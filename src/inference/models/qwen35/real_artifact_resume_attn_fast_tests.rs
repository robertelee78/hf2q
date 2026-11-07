//! Real-model Class A gate for `HF2Q_QWEN_RESUME_ATTN_FAST` (the fast path
//! of resumed-prefill route (c), `resume_attn_fast`).
//!
//! `qwen38_resume_attn_fast_real_model_exact_equal` prefills each prompt in
//! production-shaped chunks (a fresh head chunk, then resumed chunks), then
//! decodes greedily, twice in one process: once with route (c) forced to the
//! shipped bridge and once forced to the fast path (test-only thread
//! override; the switch itself is read once per process). Every chunk's
//! last-row logits, every decode step's logits and the emitted tokens must
//! be bitwise equal between the two runs.

use std::path::PathBuf;

use sha2::{Digest, Sha256};

use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::real_artifact_joint_decode_tests::{argmax, load, GateRouting, SLOTS};
use super::resume_attn_fast::{fast_route_calls, ForceResumeAttnRoute};
use super::spec_decode::positions_for_range;
use super::Qwen35LayerKind;
use crate::serve::multi_seq_kv::SlotId;

const GGUF_ENV: &str = "HF2Q_QWEN38_GGUF";
const PROMPTS_ENV: &str = "HF2Q_QWEN_RESUME_FAST_PROMPTS";
/// Production prefill shape: the first chunk, then resumed chunks.
const HEAD: usize = 796;
const CHUNK: usize = 2048;
const DECODE_STEPS: usize = 16;
/// Default prompt length: 796 + 4 × 2048 + 112, so every resumed chunk has
/// more than 32 queries (route (c)) and the last one ends at L % 64 ≠ 0.
const SYNTHETIC_TOKENS: usize = 9100;
/// Route (c) is taken for resumed chunks with more than this many queries.
const DIRECT_MAX_QUERIES: usize = 32;

/// Token-id prompts: `HF2Q_QWEN_RESUME_FAST_PROMPTS` names a JSON array of
/// 1..=4 token-id lists (the Q1 prompts format); unset, one deterministic
/// synthetic prompt. Returns the prompts and a description for the log.
fn prompts() -> (Vec<Vec<u32>>, String) {
    let Some(path) = std::env::var_os(PROMPTS_ENV).map(PathBuf::from) else {
        // Fixed LCG over ordinary text ids (no special tokens).
        let mut state: u64 = 0x005e_ed0f_9100;
        let tokens = (0..SYNTHETIC_TOKENS)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                1_000 + ((state >> 33) % 99_000) as u32
            })
            .collect();
        return (vec![tokens], format!("synthetic {SYNTHETIC_TOKENS} tokens"));
    };
    let bytes = std::fs::read(&path).expect("read prompts file");
    let prompts: Vec<Vec<u32>> =
        serde_json::from_slice(&bytes).expect("prompts file is a JSON array of token-id lists");
    assert!(
        (1..=SLOTS as usize).contains(&prompts.len()),
        "{PROMPTS_ENV} must hold 1..={SLOTS} prompts"
    );
    for tokens in &prompts {
        assert!(
            tokens.len() > HEAD,
            "every prompt must be longer than the {HEAD}-token head"
        );
    }
    let description = format!("{} sha256 {:x}", path.display(), Sha256::digest(&bytes));
    (prompts, description)
}

/// `(start, len)` of every prefill chunk of a prompt.
fn schedule(len: usize) -> Vec<(usize, usize)> {
    let mut chunks = vec![(0, HEAD.min(len))];
    let mut start = HEAD;
    while start < len {
        let n = CHUNK.min(len - start);
        chunks.push((start, n));
        start += n;
    }
    chunks
}

/// Everything one run emits, in order.
struct Run {
    /// Last-row logits of every chunk of every prompt.
    chunk_logits: Vec<Vec<f32>>,
    /// Logits of every decode step of every prompt.
    decode_logits: Vec<Vec<f32>>,
    /// Greedy tokens emitted per prompt.
    tokens: Vec<Vec<u32>>,
    /// Fast-route layer calls during the run.
    fast_calls: u64,
}

impl Run {
    fn sha256(&self) -> String {
        let mut hash = Sha256::new();
        for row in self.chunk_logits.iter().chain(&self.decode_logits) {
            for value in row {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        format!("{:x}", hash.finalize())
    }
}

/// A fresh TQ-only cache of `slots` slots for `model`, bound to the model at
/// its current address. The thread-local GPU cache is keyed by the model's
/// address, and `load` primes it before returning the model by value, so
/// the cache it primed belongs to the pre-move address; re-priming here
/// rebuilds it for this one (a no-op when it already matches).
fn tq_cache(model: &Qwen35Model, capacity: u32, slots: u32) -> HybridKvCache {
    model
        .ensure_gpu_cache_primed()
        .expect("prime the GPU cache for the model at its final address");
    let cache = model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&model.cfg, device, capacity, slots, true)
        })
        .expect("allocate TQ cache");
    assert!(
        !cache.full_attn.is_empty(),
        "model has no full-attention layer"
    );
    for (rank, slot) in cache.full_attn.iter().enumerate() {
        assert!(
            slot.k.is_none() && slot.v.is_none() && slot.tq.is_some(),
            "full-attention slot {rank} is not TQ-only: route (c) would not run"
        );
    }
    cache
}

fn full_attention_layers(model: &Qwen35Model) -> u64 {
    model
        .cfg
        .layer_types
        .iter()
        .filter(|kind| matches!(kind, Qwen35LayerKind::FullAttention))
        .count() as u64
}

/// Prefill `tokens` into `slot` of `cache` on `chunks` with route (c) forced
/// to `fast`, checking after every chunk that route (c) took the forced
/// path on every FA layer (fail fast instead of after the whole run).
/// Returns each chunk's last-row logits.
fn prefill_checked(
    model: &Qwen35Model,
    cache: &mut HybridKvCache,
    slot: u32,
    tokens: &[u32],
    chunks: &[(usize, usize)],
    fast: bool,
    label: &str,
) -> Vec<Vec<f32>> {
    let _route = ForceResumeAttnRoute::new(fast);
    let fa_layers = full_attention_layers(model);
    chunks
        .iter()
        .map(|&(start, n)| {
            let before = fast_route_calls();
            let logits = model
                .forward_gpu_last_logits(
                    &tokens[start..start + n],
                    &positions_for_range(start as i32, n),
                    cache,
                    SlotId(slot),
                )
                .unwrap_or_else(|error| {
                    panic!("{label} prompt {slot} chunk c={start} n={n}: {error:#}")
                });
            let route_c = start > 0 && n > DIRECT_MAX_QUERIES;
            let expected = if fast && route_c { fa_layers } else { 0 };
            assert_eq!(
                fast_route_calls() - before,
                expected,
                "{label} prompt {slot} chunk c={start} n={n}: fast-path layer calls"
            );
            logits
        })
        .collect()
}

/// Prefill every prompt on the production chunk schedule into its own slot
/// of a fresh cache, then decode greedily, with route (c) forced to `fast`.
fn run(model: &Qwen35Model, prompts: &[Vec<u32>], fast: bool) -> Run {
    let longest = prompts.iter().map(Vec::len).max().expect("prompts");
    let capacity =
        u32::try_from((longest + DECODE_STEPS + 16).next_multiple_of(256)).expect("capacity");
    let mut cache = tq_cache(model, capacity, prompts.len() as u32);
    let calls_before = fast_route_calls();
    let label = if fast { "fast" } else { "bridge" };
    let mut out = Run {
        chunk_logits: Vec::new(),
        decode_logits: Vec::new(),
        tokens: Vec::new(),
        fast_calls: 0,
    };
    for (slot, prompt) in prompts.iter().enumerate() {
        let slot_id = SlotId(slot as u32);
        let chunks = schedule(prompt.len());
        out.chunk_logits.extend(prefill_checked(
            model,
            &mut cache,
            slot as u32,
            prompt,
            &chunks,
            fast,
            label,
        ));
        let mut logits = out.chunk_logits.last().expect("chunk logits").clone();
        let mut emitted = Vec::with_capacity(DECODE_STEPS);
        for step in 0..DECODE_STEPS {
            let token = argmax(&logits);
            emitted.push(token);
            let position = (prompt.len() + step) as i32;
            logits = model
                .forward_gpu_last_logits(&[token], &[position; 4], &mut cache, slot_id)
                .unwrap_or_else(|error| {
                    panic!("{label} prompt {slot} decode step {step}: {error:#}")
                });
            out.decode_logits.push(logits.clone());
        }
        out.tokens.push(emitted);
    }
    out.fast_calls = fast_route_calls() - calls_before;
    out
}

/// Seconds-long check before the long runs: a 64-token head and one
/// 64-token resumed chunk of the first prompt, on each route, in a small
/// cache. Fails fast on cache binding, TQ layout, route forcing and (for
/// this short prefix) bitwise equality.
fn preflight(model: &Qwen35Model, prompt: &[u32]) {
    let chunks = [(0, 64), (64, 64)];
    let mut logits = Vec::new();
    for fast in [false, true] {
        let mut cache = tq_cache(model, 256, 1);
        let label = if fast {
            "preflight fast"
        } else {
            "preflight bridge"
        };
        logits.push(prefill_checked(
            model,
            &mut cache,
            0,
            &prompt[..128],
            &chunks,
            fast,
            label,
        ));
    }
    let (rows, elements) = mismatches(&logits[0], &logits[1]);
    eprintln!("[resume-attn-fast gate] preflight: {rows} rows, {elements} elements differ");
    assert_eq!(
        elements, 0,
        "preflight: fast path is not bitwise equal to the bridge"
    );
}

/// `(rows that differ, elements whose bits differ)` between two row lists.
fn mismatches(a: &[Vec<f32>], b: &[Vec<f32>]) -> (usize, usize) {
    assert_eq!(a.len(), b.len(), "row count");
    a.iter().zip(b).fold((0, 0), |(rows, elements), (x, y)| {
        assert_eq!(x.len(), y.len(), "vocab width");
        let differ = x
            .iter()
            .zip(y)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count();
        (rows + usize::from(differ > 0), elements + differ)
    })
}

/// Class A gate: route (c)'s fast path against the shipped bridge on the
/// real model. Run explicitly:
/// `HF2Q_QWEN38_GGUF=… [HF2Q_QWEN_RESUME_FAST_PROMPTS=prompts.json] <test binary> qwen38_resume_attn_fast_real_model_exact_equal --ignored --nocapture --test-threads=1`
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (a Qwen3.8 GGUF) and Apple Metal"]
fn qwen38_resume_attn_fast_real_model_exact_equal() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let (prompts, description) = prompts();
    let model = load(GGUF_ENV, GateRouting::Serve);
    let fa_layers = full_attention_layers(&model);
    let route_c_chunks: u64 = prompts
        .iter()
        .flat_map(|prompt| schedule(prompt.len()))
        .filter(|&(start, n)| start > 0 && n > DIRECT_MAX_QUERIES)
        .count() as u64;
    eprintln!(
        "[resume-attn-fast gate] artifact={} prompts={description} lengths={:?} head={HEAD} \
         chunk={CHUNK} decode_steps={DECODE_STEPS} route_c_chunks={route_c_chunks} fa_layers={fa_layers}",
        std::env::var(GGUF_ENV).unwrap_or_default(),
        prompts.iter().map(Vec::len).collect::<Vec<_>>()
    );
    // Everything that can fail before the long runs fails here.
    assert!(fa_layers > 0, "model has no full-attention layer");
    assert!(
        route_c_chunks > 0,
        "no resumed chunk with more than {DIRECT_MAX_QUERIES} queries"
    );
    let vocab = model.cfg.vocab_size;
    for (index, prompt) in prompts.iter().enumerate() {
        assert!(
            prompt.iter().all(|&id| id < vocab),
            "prompt {index} holds a token id outside the vocabulary ({vocab})"
        );
        assert!(
            prompt.len() + DECODE_STEPS <= model.cfg.max_position_embeddings as usize,
            "prompt {index} plus {DECODE_STEPS} decode steps exceeds the model's positions"
        );
    }
    preflight(&model, &prompts[0]);

    let bridge = run(&model, &prompts, false);
    let fast = run(&model, &prompts, true);

    let (chunk_rows, chunk_elements) = mismatches(&bridge.chunk_logits, &fast.chunk_logits);
    let (decode_rows, decode_elements) = mismatches(&bridge.decode_logits, &fast.decode_logits);
    let token_mismatches: usize = bridge
        .tokens
        .iter()
        .zip(&fast.tokens)
        .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
        .sum();
    eprintln!(
        "[resume-attn-fast gate] chunk logits: {chunk_rows}/{} rows, {chunk_elements} elements differ; \
         decode logits: {decode_rows}/{} rows, {decode_elements} elements differ; \
         tokens: {token_mismatches} differ",
        bridge.chunk_logits.len(),
        bridge.decode_logits.len()
    );
    eprintln!(
        "[resume-attn-fast gate] bridge sha256={} fast_calls={} tokens={:?}",
        bridge.sha256(),
        bridge.fast_calls,
        bridge.tokens
    );
    eprintln!(
        "[resume-attn-fast gate] fast   sha256={} fast_calls={} tokens={:?}",
        fast.sha256(),
        fast.fast_calls,
        fast.tokens
    );

    // The gate is vacuous unless each arm took the route it was forced to.
    assert_eq!(bridge.fast_calls, 0, "bridge run took the fast path");
    assert_eq!(
        fast.fast_calls,
        route_c_chunks * fa_layers,
        "fast run must take the fast path on every FA layer of every route (c) chunk"
    );
    assert_eq!(
        (chunk_elements, decode_elements, token_mismatches),
        (0, 0, 0),
        "fast path is not bitwise equal to the bridge"
    );
}

/// Model-free check of the gate's cache binding (the failure that stopped
/// its first real-model attempt): a model primed and then moved, as `load`
/// returns it, no longer owns the thread's GPU cache; [`tq_cache`] re-primes
/// for the model's current address and hands out TQ-only slots.
#[test]
fn resume_fast_gate_cache_binds_a_moved_model() {
    use super::{default_layer_types, Qwen35Config, Qwen35Variant};

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    if let Err(error) = mlx_native::MlxDevice::new() {
        eprintln!("skipping: no Metal device: {error}");
        return;
    }
    let cfg = Qwen35Config {
        variant: Qwen35Variant::Dense,
        hidden_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 256,
        linear_num_key_heads: 2,
        linear_num_value_heads: 2,
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        layer_types: default_layer_types(4, 4),
        partial_rotary_factor: 0.25,
        rope_theta: 1e7,
        rotary_dim: 64,
        mrope_section: [11, 11, 10, 0],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 4096,
        vocab_size: 128,
        attn_output_gate: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: true,
        intermediate_size: Some(64),
        moe: None,
    };
    let primed = Qwen35Model::empty_from_cfg(cfg);
    primed
        .ensure_gpu_cache_primed()
        .expect("prime before the move");
    let model = Box::new(primed);
    assert!(
        model.with_gpu_cache_mut(|_, _| Ok(())).is_err(),
        "a moved model must not own the cache primed at its old address"
    );
    let cache = tq_cache(&model, 256, 2);
    assert_eq!(cache.full_attn.len() as u64, full_attention_layers(&model));
    model
        .with_gpu_cache_mut(|_, _| Ok(()))
        .expect("the re-primed cache serves the model at its current address");
}
