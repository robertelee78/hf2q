//! Teacher-forced DFlash2 verify replay for comparing Qwen3.6 MoE expert
//! routes on identical inputs.
//!
//! `qwen36_dflash2_verify_moe_route_replay` prefills up to four lanes with
//! 256 tokens of real token streams and then advances every lane through
//! width-32 DFlash2 verify steps of eight rows whose tokens are the stream's
//! own, so every route consumes the same tokens at the same positions. Each
//! row's top logits, the logit and rank of the stream's next token and a
//! fixed sample of the vocabulary are written as one JSON line.
//!
//! `HF2Q_QWEN_MOE_REPLAY_SEGMENTS=<n>` (default 1) repeats that from `n`
//! places of each stream, 128 tokens apart, each on a fresh cache after its
//! own prefill of the same tokens. A segment's first step then carries no
//! history of earlier verify steps, so `n` segments of one or two steps
//! measure a route's error without the divergence of state that a long
//! replay accumulates.
//!
//! The MoE expert route is a process setting, so each route is one process
//! over the same texts; `scripts/qwen_moe_route_replay_compare.py` reduces
//! the files against the first:
//!
//! ```text
//! export HF2Q_QWEN36_GGUF=… HF2Q_QWEN_MOE_REPLAY_TEXTS=texts.json HF2Q_QWEN_MOE_BATCHED=1
//! T="cargo test --release --locked --bin hf2q qwen36_dflash2_verify_moe_route_replay -- --ignored --exact"
//! # FP32 expert products on the native blocks: the reference
//! HF2Q_QWEN_MOE_REPLAY_OUT=matvec.jsonl $T
//! HF2Q_QWEN_MOE_STAGED_M8=1 HF2Q_QWEN_MOE_REPLAY_OUT=staged.jsonl $T
//! HF2Q_QWEN_MOE_STAGED_M8=1 HF2Q_QWEN_MOE_LIVE_TILES=1 HF2Q_QWEN_MOE_REPLAY_OUT=live.jsonl $T
//! python3 -B scripts/qwen_moe_route_replay_compare.py matvec.jsonl staged.jsonl live.jsonl
//! ```

use std::io::Write;
use std::path::PathBuf;

use super::forward_gpu::{Qwen35JointVerify, Qwen35JointVerifyLane};
use super::joint_decode_routes::{joint_ffn_shared_from_process, select_dense_ffn_route};
use super::joint_projection::JointProjectionKernel;
use super::joint_step_options::JointStepOptions;
use super::model::Qwen35Model;
use super::real_artifact_joint_decode_invariance_tests::cache_with_capacity;
use super::real_artifact_joint_decode_tests::SLOTS;
use super::spec_decode::positions_for_range;
use super::Qwen35JointOutput;
use crate::serve::multi_seq_kv::SlotId;

const TEXTS_ENV: &str = "HF2Q_QWEN_MOE_REPLAY_TEXTS";
const OUT_ENV: &str = "HF2Q_QWEN_MOE_REPLAY_OUT";
const STEPS_ENV: &str = "HF2Q_QWEN_MOE_REPLAY_STEPS";
const SEGMENTS_ENV: &str = "HF2Q_QWEN_MOE_REPLAY_SEGMENTS";
/// Tokens of each stream that a segment's prefill consumes.
const PREFILL: usize = 256;
/// Tokens between the starts of two segments.
const SEGMENT_STRIDE: usize = 128;
/// Rows of one lane's verify window.
const WINDOW: usize = 8;
/// Leading logits kept per row.
const TOP: usize = 4;
/// Vocabulary entries sampled per row for the logit error.
const SAMPLE: usize = 64;

/// The `TOP` largest logits, largest first, ties to the lower id.
fn top_logits(logits: &[f32]) -> [(u32, f32); TOP] {
    let mut top = [(u32::MAX, f32::NEG_INFINITY); TOP];
    for (id, &value) in logits.iter().enumerate() {
        if value > top[TOP - 1].1 {
            let mut at = TOP - 1;
            while at > 0 && value > top[at - 1].1 {
                top[at] = top[at - 1];
                at -= 1;
            }
            top[at] = (id as u32, value);
        }
    }
    top
}

#[test]
#[ignore = "requires HF2Q_QWEN36_GGUF, HF2Q_QWEN_MOE_REPLAY_TEXTS, HF2Q_QWEN_MOE_REPLAY_OUT and Apple Metal"]
fn qwen36_dflash2_verify_moe_route_replay() {
    let out = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {OUT_ENV} to a new output file"));
    assert!(!out.exists(), "{} already exists", out.display());
    let number = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .map(|raw| raw.parse::<usize>().expect("integer"))
            .unwrap_or(default)
    };
    let steps = number(STEPS_ENV, 64);
    let segments = number(SEGMENTS_ENV, 1);
    assert!(steps > 0 && segments > 0, "steps and segments are positive");
    let texts_path = std::env::var_os(TEXTS_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {TEXTS_ENV} to a JSON array of texts"));
    let texts: Vec<String> =
        serde_json::from_slice(&std::fs::read(&texts_path).expect("read texts"))
            .expect("texts file is a JSON array of strings");
    assert!(
        (1..=SLOTS as usize).contains(&texts.len()),
        "{TEXTS_ENV} must hold 1..={SLOTS} texts"
    );

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let gguf_path = std::env::var_os("HF2Q_QWEN36_GGUF")
        .map(PathBuf::from)
        .expect("HF2Q_QWEN36_GGUF");
    let gguf = mlx_native::gguf::GgufFile::open(&gguf_path).expect("open GGUF");
    let tokenizer = super::tokenizer::build_tokenizer_from_gguf(&gguf).expect("GGUF tokenizer");
    let segment_tokens = PREFILL + WINDOW * steps + 1;
    let needed = SEGMENT_STRIDE * (segments - 1) + segment_tokens;
    let streams: Vec<Vec<u32>> = texts
        .iter()
        .enumerate()
        .map(|(lane, text)| {
            let ids = tokenizer
                .encode(text.as_str(), false)
                .expect("tokenize text")
                .get_ids()
                .to_vec();
            assert!(
                ids.len() >= needed,
                "text {lane} has {} tokens, the replay needs {needed}",
                ids.len()
            );
            ids
        })
        .collect();

    // Load and prime in place: the thread-local GPU cache is keyed by the
    // model's address.
    let mut progress = crate::serve::header::LoadProgress::new(false, 1, 0);
    let mut model = Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load Qwen model");
    model.bind_ggml_routing_policy(super::routing_policy::resolve_qwen_ggml_routing_policy(
        &gguf_path,
    ));
    // The upload's INFO lines on stderr: the live-tile route logs how many
    // layers run its fused expert stages and how many run stage by stage.
    let load_log = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .without_time()
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .finish();
    tracing::subscriber::with_default(load_log, || model.ensure_gpu_cache_primed())
        .expect("prime GPU cache");
    let mode = model
        .joint_decode_mode()
        .unwrap_or_else(|reason| panic!("joint decode not admitted: {reason}"));
    let dense_ffn = select_dense_ffn_route(
        mode,
        joint_ffn_shared_from_process().expect("joint FFN switch"),
    );
    let options = JointStepOptions {
        projection: JointProjectionKernel::Canonical,
        pooled_outputs: true,
        verify_attention_tile: false,
        verify_mixer_fused: false,
    };
    let vocab = model.cfg.vocab_size as usize;
    let sample_ids: Vec<usize> = (0..SAMPLE)
        .map(|index| (index * 3877 + 11) % vocab)
        .collect();

    let capacity = u32::try_from((segment_tokens + 16).next_multiple_of(256)).expect("capacity");

    let mut file = std::io::BufWriter::new(std::fs::File::create(&out).expect("create output"));
    let switch = |name: &str| std::env::var(name).ok();
    let header = serde_json::json!({
        "test": "qwen36_dflash2_verify_moe_route_replay",
        "artifact": gguf_path.display().to_string(),
        "texts": texts_path.display().to_string(),
        "lanes": streams.len(),
        "prefill": PREFILL,
        "window": WINDOW,
        "steps": steps,
        "segments": segments,
        "segment_stride": SEGMENT_STRIDE,
        "moe_ffn_decode_route": model.moe_ffn_decode_route(),
        "moe_expert_decode_route": model.moe_expert_decode_route(),
        "joint_route_mode": format!("{mode:?}"),
        "switches": {
            "HF2Q_QWEN_MOE_BATCHED": switch("HF2Q_QWEN_MOE_BATCHED"),
            "HF2Q_QWEN_MOE_STAGED_M8": switch("HF2Q_QWEN_MOE_STAGED_M8"),
            "HF2Q_QWEN_MOE_LIVE_TILES": switch("HF2Q_QWEN_MOE_LIVE_TILES"),
            "HF2Q_REPACKED_GGUF_M32": switch("HF2Q_REPACKED_GGUF_M32"),
            "HF2Q_REPACKED_GGUF_M32_PROJ": switch("HF2Q_REPACKED_GGUF_M32_PROJ"),
        },
        "sample_ids": sample_ids,
    });
    writeln!(file, "{header}").expect("write header");

    for segment in 0..segments {
        // Positions restart at zero: a segment is its own conversation.
        let origin = SEGMENT_STRIDE * segment;
        let mut cache = cache_with_capacity(&model, capacity);
        for (lane, stream) in streams.iter().enumerate() {
            model
                .forward_gpu_last_logits(
                    &stream[origin..origin + PREFILL],
                    &positions_for_range(0, PREFILL),
                    &mut cache,
                    SlotId(lane as u32),
                )
                .expect("prefill");
        }
        for step in 0..steps {
            let start = PREFILL + WINDOW * step;
            let lanes: Vec<Qwen35JointVerifyLane<'_>> = streams
                .iter()
                .enumerate()
                .map(|(lane, stream)| Qwen35JointVerifyLane {
                    slot_id: SlotId(lane as u32),
                    position: start as i32,
                    tokens: &stream[origin + start..origin + start + WINDOW],
                    output: Qwen35JointOutput::Logits,
                })
                .collect();
            let Qwen35JointVerify { rows, pending, .. } = model
                .forward_gpu_dflash2_verify(&lanes, &mut cache, mode, dense_ffn, options, None)
                .unwrap_or_else(|error| panic!("segment {segment} step {step}: {error:#}"));
            for (lane, (outputs, stream)) in rows.iter().zip(&streams).enumerate() {
                for (row, output) in outputs.iter().enumerate() {
                    let logits = output.logits.as_deref().expect("verify logits");
                    let position = start + row;
                    let next = stream[origin + position + 1] as usize;
                    let next_logit = logits[next];
                    let rank = logits.iter().filter(|value| **value > next_logit).count();
                    let top = top_logits(logits);
                    let record = serde_json::json!({
                        "lane": lane,
                        "segment": segment,
                        "step": step,
                        "position": position,
                        "input": stream[origin + position],
                        "next": next,
                        "next_logit": next_logit,
                        "next_rank": rank,
                        "top": top.iter().map(|(id, value)| (*id, *value)).collect::<Vec<_>>(),
                        "sample": sample_ids.iter().map(|id| logits[*id]).collect::<Vec<_>>(),
                    });
                    writeln!(file, "{record}").expect("write row");
                }
            }
            model
                .commit_joint_verify(&mut cache, pending, &vec![WINDOW; streams.len()])
                .unwrap_or_else(|error| panic!("segment {segment} step {step} commit: {error:#}"));
        }
    }
    file.flush().expect("flush output");
    eprintln!(
        "MoE route replay ({}) written: {}",
        model.moe_expert_decode_route(),
        out.display()
    );
}
