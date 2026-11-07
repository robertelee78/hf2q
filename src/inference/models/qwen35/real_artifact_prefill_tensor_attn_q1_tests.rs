//! Q1 long-prefix gate for ADR-063 F3 (tensor attention for resumed Qwen
//! prefill), per the pre-declared packet
//! `~/.local/state/hf2q/v0130/q38-opparity/q1/PACKET-f3.md`.
//!
//! `qwen38_prefill_tensor_attn_q1_trace` prefills real long prompts in three
//! arms whose routes are fixed in code, then decodes teacher-forced on the
//! reference arm's top-1 and writes `hf2q.adr063.invariance.v2` stores for
//! `q1_decision_policy.py envelope`:
//! - `R/`: each prompt in one fresh chunk (c = 0, route (a));
//! - `O/`: a fresh head chunk, then fixed-size resumed chunks on the bridge
//!   (route (c));
//! - `N/`: the same schedule on the tensor route (route (c′)).
//!
//! Every arm also writes `chunks.json` (arm, agent, c, n, resolved route,
//! tensor dispatches per chunk); `R/` writes `teacher.json`.
//! `qwen38_prefill_r_route_40k_is_deterministic` is the D5a control.

use std::path::PathBuf;

use sha2::{Digest, Sha256};

use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::prefill_tensor_attn::{tensor_dispatch_count, QwenPrefillAttention};
use super::real_artifact_joint_decode_invariance_tests::cache_with_capacity;
use super::real_artifact_joint_decode_q1_tests::LogitStore;
use super::real_artifact_joint_decode_tests::{argmax, bits, load, SLOTS};
use super::spec_decode::positions_for_range;
use super::Qwen35LayerKind;
use crate::serve::multi_seq_kv::SlotId;

const PROMPTS_ENV: &str = "HF2Q_QWEN_PREFILL_Q1_PROMPTS";
const OUT_ENV: &str = "HF2Q_QWEN_PREFILL_Q1_OUT";
const STEPS_ENV: &str = "HF2Q_QWEN_PREFILL_Q1_STEPS";
const HEAD_ENV: &str = "HF2Q_QWEN_PREFILL_Q1_HEAD";
const CHUNK_ENV: &str = "HF2Q_QWEN_PREFILL_Q1_CHUNK";
const ARMS_ENV: &str = "HF2Q_QWEN_PREFILL_Q1_ARMS";
const TEACHER_ENV: &str = "HF2Q_QWEN_PREFILL_Q1_TEACHER";
const TEACHER_SCHEMA: &str = "hf2q.qwen38-prefill-q1-teacher.v1";

/// One arm of the gate: how a prompt is prefilled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arm {
    /// Reference: the whole prompt in one fresh chunk.
    R,
    /// Head chunk, then resumed chunks on the bridge (route (c)).
    O,
    /// Head chunk, then resumed chunks on the tensor route (route (c′)).
    N,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::R => "R",
            Self::O => "O",
            Self::N => "N",
        }
    }

    /// The route fixed in code for this arm (not read from the switch).
    fn route(self) -> QwenPrefillAttention {
        match self {
            Self::N => QwenPrefillAttention::Tensor,
            Self::R | Self::O => QwenPrefillAttention::Bridge,
        }
    }
}

struct Settings {
    prompts_path: PathBuf,
    prompts_sha256: String,
    prompts: Vec<Vec<u32>>,
    steps: usize,
    head: usize,
    chunk: usize,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .filter(|raw| !raw.trim().is_empty())
        .map(|raw| {
            raw.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be an integer, got {raw:?}"))
        })
        .unwrap_or(default)
}

/// Token-id prompts: a JSON array of 1..=4 token-id lists.
fn settings() -> Settings {
    let prompts_path = std::env::var_os(PROMPTS_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {PROMPTS_ENV} to a JSON array of token-id lists"));
    let bytes = std::fs::read(&prompts_path).expect("read prompts file");
    let prompts: Vec<Vec<u32>> =
        serde_json::from_slice(&bytes).expect("prompts file is a JSON array of token-id lists");
    assert!(
        (1..=SLOTS as usize).contains(&prompts.len()),
        "{PROMPTS_ENV} must hold 1..={SLOTS} prompts"
    );
    let settings = Settings {
        prompts_sha256: format!("{:x}", Sha256::digest(&bytes)),
        prompts_path,
        prompts,
        steps: env_usize(STEPS_ENV, 64),
        head: env_usize(HEAD_ENV, 796),
        chunk: env_usize(CHUNK_ENV, 2048),
    };
    assert!(
        settings.head > 0 && settings.chunk > 32,
        "head > 0 and chunk > 32 required"
    );
    for tokens in &settings.prompts {
        assert!(
            tokens.len() > settings.head,
            "prompt not longer than the head chunk"
        );
    }
    settings
}

/// `(start, len)` of every prefill chunk of a prompt in `arm`.
fn schedule(arm: Arm, len: usize, head: usize, chunk: usize) -> Vec<(usize, usize)> {
    if arm == Arm::R {
        return vec![(0, len)];
    }
    let mut chunks = vec![(0, head)];
    let mut start = head;
    while start < len {
        let n = chunk.min(len - start);
        chunks.push((start, n));
        start += n;
    }
    chunks
}

fn full_attention_layers(model: &Qwen35Model) -> usize {
    model
        .cfg
        .layer_types
        .iter()
        .filter(|kind| matches!(kind, Qwen35LayerKind::FullAttention))
        .count()
}

/// Set `arm`'s route on the model; the tensor arm requires the Metal
/// tensor API (a silent bridge fallback would make the gate vacuous).
fn fix_route(model: &mut Qwen35Model, arm: Arm) {
    if arm.route().uses_tensor() {
        let available = model
            .with_gpu_cache_mut(|device, registry| {
                mlx_native::ops::flash_attn_prefill_tensor::flash_attn_prefill_tensor_available(
                    registry, device, 256,
                )
                .map_err(anyhow::Error::from)
            })
            .expect("tensor attention probe");
        assert!(available, "arm N needs the Metal tensor API");
    }
    model.prefill_attention = arm.route();
}

/// Prefill `tokens` into `slot` on `arm`'s schedule, logging every chunk;
/// returns the last-position logits.
fn prefill(
    model: &Qwen35Model,
    cache: &mut HybridKvCache,
    arm: Arm,
    slot: u32,
    tokens: &[u32],
    settings: &Settings,
    log: &mut Vec<serde_json::Value>,
) -> Vec<f32> {
    let mut logits = Vec::new();
    for (start, n) in schedule(arm, tokens.len(), settings.head, settings.chunk) {
        let before = tensor_dispatch_count();
        logits = model
            .forward_gpu_last_logits(
                &tokens[start..start + n],
                &positions_for_range(start as i32, n),
                cache,
                SlotId(slot),
            )
            .unwrap_or_else(|error| {
                panic!("{} agent {slot} chunk c={start}: {error:#}", arm.name())
            });
        let dispatches = tensor_dispatch_count() - before;
        let route = if start == 0 {
            "fresh"
        } else if n <= 32 {
            "direct"
        } else {
            model.prefill_attention.label()
        };
        eprintln!(
            "[F3 Q1 chunk] arm={} agent=agent-{slot} c={start} n={n} route={route} tensor_dispatches={dispatches}",
            arm.name()
        );
        log.push(serde_json::json!({
            "arm": arm.name(),
            "agent": format!("agent-{slot}"),
            "c": start,
            "n": n,
            "route": route,
            "tensor_dispatches": dispatches,
        }));
    }
    logits
}

/// Teacher tokens per agent: `teacher[agent][step]` is R's top-1 at row
/// `step` (step 0 = last prompt position).
fn read_teacher(settings: &Settings) -> Vec<Vec<u32>> {
    let path = std::env::var_os(TEACHER_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!("without arm R in this process set {TEACHER_ENV} to R's teacher.json")
        });
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("read teacher")).expect("teacher JSON");
    assert_eq!(value["schema"], TEACHER_SCHEMA, "teacher schema");
    assert_eq!(
        value["prompts_sha256"],
        settings.prompts_sha256.as_str(),
        "teacher prompts"
    );
    assert_eq!(value["steps"], settings.steps, "teacher steps");
    (0..settings.prompts.len())
        .map(|agent| {
            let tokens: Vec<u32> =
                serde_json::from_value(value["agents"][format!("agent-{agent}")].clone())
                    .unwrap_or_else(|error| panic!("teacher agent-{agent}: {error}"));
            assert!(
                tokens.len() >= settings.steps,
                "teacher agent-{agent} too short"
            );
            tokens
        })
        .collect()
}

/// Run one arm: prefill every prompt, record the last-position row, then
/// `steps` teacher-forced one-token decode rows. Returns each agent's
/// top-1 per row (R's are the teacher).
fn run_arm(
    model: &mut Qwen35Model,
    arm: Arm,
    settings: &Settings,
    teacher: Option<&[Vec<u32>]>,
    out: &std::path::Path,
) -> Vec<Vec<u32>> {
    fix_route(model, arm);
    let model = &*model;
    let longest = settings
        .prompts
        .iter()
        .map(Vec::len)
        .max()
        .expect("prompts");
    let capacity =
        u32::try_from((longest + settings.steps + 16).next_multiple_of(256)).expect("capacity");
    let per_token = crate::serve::load_info::qwen35_slot_kv_bytes_per_token(&model.cfg, true);
    let fixed = crate::serve::load_info::qwen35_fixed_kv_bytes_per_slot(&model.cfg);
    eprintln!(
        "[F3 Q1] arm={} capacity={capacity} cache≈{} B ({SLOTS} slots × ({per_token} B/token × {capacity} + {fixed} B fixed incl. anchor reserve))",
        arm.name(),
        u64::from(SLOTS) * (per_token * u64::from(capacity) + fixed)
    );
    let mut cache = cache_with_capacity(model, capacity);
    let layers = model.cfg.num_hidden_layers as usize;
    let mut store = LogitStore::create(out.join(arm.name()));
    let mut log = Vec::new();
    let mut top1 = Vec::with_capacity(settings.prompts.len());
    for (agent, tokens) in settings.prompts.iter().enumerate() {
        let slot = agent as u32;
        let mut logits = prefill(model, &mut cache, arm, slot, tokens, settings, &mut log);
        let mut decisions = Vec::with_capacity(settings.steps + 1);
        for step in 0..=settings.steps {
            let row = tokens.len() - 1 + step;
            store.push(slot, row, layers, step, &logits);
            decisions.push(argmax(&logits));
            if step == settings.steps {
                break;
            }
            let token = teacher.map_or(decisions[step], |teacher| teacher[agent][step]);
            let position = (tokens.len() + step) as i32;
            logits = model
                .forward_gpu_last_logits(&[token], &[position; 4], &mut cache, SlotId(slot))
                .unwrap_or_else(|error| {
                    panic!("{} agent {slot} step {step}: {error:#}", arm.name())
                });
        }
        top1.push(decisions);
    }
    let directory = out.join(arm.name());
    std::fs::write(
        directory.join("chunks.json"),
        serde_json::to_vec_pretty(&log).expect("encode chunks"),
    )
    .expect("write chunks.json");
    let resumed: Vec<_> = log.iter().filter(|chunk| chunk["c"] != 0).collect();
    eprintln!(
        "[F3 Q1] arm={} resumed chunks={} with tensor dispatches={} (FA layers {})",
        arm.name(),
        resumed.len(),
        resumed
            .iter()
            .filter(|chunk| chunk["tensor_dispatches"].as_u64() > Some(0))
            .count(),
        full_attention_layers(model)
    );
    let provenance = serde_json::json!({
        "test": "qwen38_prefill_tensor_attn_q1_trace",
        "packet": "q38-opparity/q1/PACKET-f3.md",
        "artifact": std::env::var("HF2Q_QWEN38_GGUF").unwrap_or_default(),
        "prompts": settings.prompts_path.display().to_string(),
        "prompts_sha256": settings.prompts_sha256,
        "prompt_tokens": settings.prompts.iter().map(Vec::len).collect::<Vec<_>>(),
        "head": settings.head,
        "chunk": settings.chunk,
        "steps": settings.steps,
        "arm": arm.name(),
        "route": if arm == Arm::R { "fresh single chunk" } else { arm.route().label() },
        "teacher_forcing": "R top-1",
        "teacher": std::env::var(TEACHER_ENV).ok(),
    });
    let trace = store.finish(&provenance);
    eprintln!("[F3 Q1] arm={} trace {}", arm.name(), trace.display());
    top1
}

/// ADR-063 F3 long-prefix Q1 gate input (PACKET-f3.md): arms R, O and N
/// (or the `HF2Q_QWEN_PREFILL_Q1_ARMS` subset, comma-separated) over the
/// token-id prompts, each with `steps` teacher-forced decode rows after the
/// last-position row. Without R in the process, `HF2Q_QWEN_PREFILL_Q1_TEACHER`
/// names R's `teacher.json`. Then
/// `python3 -B scripts/four_agent/q1_decision_policy.py envelope $OUT/R/trace.json $OUT/N/trace.json`
/// (and `O`, and `O` vs `N`).
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF, HF2Q_QWEN_PREFILL_Q1_PROMPTS, HF2Q_QWEN_PREFILL_Q1_OUT and Apple Metal"]
fn qwen38_prefill_tensor_attn_q1_trace() {
    let out = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {OUT_ENV} to a new output directory"));
    assert!(!out.exists(), "{} already exists", out.display());
    let settings = settings();
    let arms: Vec<Arm> = std::env::var(ARMS_ENV)
        .ok()
        .filter(|raw| !raw.trim().is_empty())
        .map(|raw| {
            raw.split(',')
                .map(|name| match name.trim() {
                    "R" => Arm::R,
                    "O" => Arm::O,
                    "N" => Arm::N,
                    other => panic!("{ARMS_ENV}: unknown arm {other:?} (R, O, N)"),
                })
                .collect()
        })
        .unwrap_or_else(|| vec![Arm::R, Arm::O, Arm::N]);
    assert!(!arms.is_empty(), "{ARMS_ENV} selects no arm");
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = load("HF2Q_QWEN38_GGUF");
    model.ensure_gpu_cache_primed().expect("prime GPU cache");
    std::fs::create_dir_all(&out).expect("create output directory");

    let mut teacher = None;
    if arms.contains(&Arm::R) {
        let decisions = run_arm(&mut model, Arm::R, &settings, None, &out);
        let agents: serde_json::Map<String, serde_json::Value> = decisions
            .iter()
            .enumerate()
            .map(|(agent, tokens)| (format!("agent-{agent}"), serde_json::json!(tokens)))
            .collect();
        std::fs::write(
            out.join("R").join("teacher.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema": TEACHER_SCHEMA,
                "prompts_sha256": settings.prompts_sha256,
                "steps": settings.steps,
                "agents": agents,
            }))
            .expect("encode teacher"),
        )
        .expect("write teacher.json");
        teacher = Some(decisions);
    }
    let teacher = match teacher {
        Some(teacher) => teacher,
        None => read_teacher(&settings),
    };
    for arm in arms.into_iter().filter(|arm| *arm != Arm::R) {
        run_arm(&mut model, arm, &settings, Some(&teacher), &out);
    }
}

/// D5a: the reference route is deterministic at full length. Prefill the
/// first prompt of `HF2Q_QWEN_PREFILL_Q1_PROMPTS` as one fresh chunk twice
/// into fresh one-slot caches; the last-position logits must be bitwise
/// equal.
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF, HF2Q_QWEN_PREFILL_Q1_PROMPTS and Apple Metal"]
fn qwen38_prefill_r_route_40k_is_deterministic() {
    let settings = settings();
    let tokens = &settings.prompts[0];
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = load("HF2Q_QWEN38_GGUF");
    model.ensure_gpu_cache_primed().expect("prime GPU cache");
    let capacity = u32::try_from((tokens.len() + 16).next_multiple_of(256)).expect("capacity");
    let run = || {
        let mut cache = model
            .with_gpu_cache_mut(|device, _registry| {
                HybridKvCache::new_with_options(&model.cfg, device, capacity, 1, true)
            })
            .expect("one-slot TQ cache");
        model
            .forward_gpu_last_logits(
                tokens,
                &positions_for_range(0, tokens.len()),
                &mut cache,
                SlotId(0),
            )
            .expect("single-chunk prefill")
    };
    let first = run();
    let second = run();
    let differing = bits(&first)
        .iter()
        .zip(bits(&second))
        .filter(|(a, b)| **a != *b)
        .count();
    eprintln!(
        "[F3 D5a] prompts_sha256={} tokens={} differing logits={differing} top1={}",
        settings.prompts_sha256,
        tokens.len(),
        argmax(&first)
    );
    assert_eq!(
        differing, 0,
        "R route last-position logits differ across runs"
    );
}
