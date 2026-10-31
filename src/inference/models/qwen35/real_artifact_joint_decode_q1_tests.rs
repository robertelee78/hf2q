//! Q1 Class B trace on real prompts for the Qwen3.8 padded joint route.
//!
//! `qwen38_joint_decode_q1_envelope_trace` advances real rendered agent
//! prompts teacher-forced through serial one-token forwards and through
//! joint steps on both dense FFN routes, and writes every full-vocabulary
//! logit row set as an `hf2q.adr063.invariance.v2` store for
//! `q1_decision_policy.py envelope`.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::real_artifact_joint_decode_invariance_tests::cache_with_capacity;
use super::real_artifact_joint_decode_tests::{argmax, bits, load, SLOTS};
use super::spec_decode::positions_for_range;
use super::{Qwen35JointOutput, Qwen35JointRow};
use crate::serve::multi_seq_kv::SlotId;

const Q1_PROMPTS_ENV: &str = "HF2Q_QWEN_JOINT_Q1_PROMPTS";
const Q1_OUT_ENV: &str = "HF2Q_QWEN_JOINT_Q1_OUT";
const Q1_STEPS_ENV: &str = "HF2Q_QWEN_JOINT_Q1_STEPS";
const Q1_TOKENIZER_ENV: &str = "HF2Q_QWEN35_TOKENIZER";
const Q1_TRACE_SCHEMA: &str = "hf2q.adr063.invariance.v2";
const PREFILL_CHUNK: usize = 512;

/// Joint arms of the trace: the production default first, then the
/// `HF2Q_QWEN_JOINT_FFN_SHARED=0` route. Each arm owns a cache and a store
/// directory under `$HF2Q_QWEN_JOINT_Q1_OUT`.
const JOINT_ARMS: [(JointDenseFfnRoute, &str); 2] = [
    (JointDenseFfnRoute::SharedWeights, "joint_shared_ffn"),
    (JointDenseFfnRoute::SerialBuilder, "joint_serial_ffn"),
];

/// Rendered prompts: a JSON array of 2..=4 entries, each either the exact
/// token IDs or the rendered chat-prompt text (tokenized with the serving
/// tokenizer at `HF2Q_QWEN35_TOKENIZER`, no special tokens added).
fn q1_prompts() -> Vec<Vec<u32>> {
    let path = std::env::var_os(Q1_PROMPTS_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {Q1_PROMPTS_ENV} to a JSON array of rendered prompts"));
    let entries: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(&path).expect("read prompts file"))
            .expect("prompts file is a JSON array");
    assert!(
        (2..=SLOTS as usize).contains(&entries.len()),
        "{Q1_PROMPTS_ENV} must hold 2..={SLOTS} prompts"
    );
    let tokenizer = entries.iter().any(serde_json::Value::is_string).then(|| {
        let path = std::env::var_os(Q1_TOKENIZER_ENV)
            .unwrap_or_else(|| panic!("text prompts require {Q1_TOKENIZER_ENV}"));
        tokenizers::Tokenizer::from_file(path).expect("load tokenizer.json")
    });
    entries
        .iter()
        .map(|entry| match entry {
            serde_json::Value::String(text) => tokenizer
                .as_ref()
                .expect("tokenizer loaded for text prompts")
                .encode(text.as_str(), false)
                .expect("tokenize rendered prompt")
                .get_ids()
                .to_vec(),
            serde_json::Value::Array(ids) => ids
                .iter()
                .map(|id| {
                    id.as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .expect("token id")
                })
                .collect(),
            other => panic!("prompt entry must be text or token ids, got {other}"),
        })
        .inspect(|tokens: &Vec<u32>| assert!(tokens.len() >= 16, "prompt shorter than 16 tokens"))
        .collect()
}

fn prefill_chunked(
    model: &Qwen35Model,
    cache: &mut HybridKvCache,
    slot: u32,
    tokens: &[u32],
) -> Vec<f32> {
    let mut logits = Vec::new();
    for start in (0..tokens.len()).step_by(PREFILL_CHUNK) {
        let chunk = &tokens[start..(start + PREFILL_CHUNK).min(tokens.len())];
        logits = model
            .forward_gpu_last_logits(
                chunk,
                &positions_for_range(start as i32, chunk.len()),
                cache,
                SlotId(slot),
            )
            .expect("chunked prefill");
    }
    logits
}

/// One `hf2q.adr063.invariance.v2` store of full-vocabulary logit rows.
struct LogitStore {
    directory: PathBuf,
    expected: Vec<serde_json::Value>,
    captured: Vec<serde_json::Value>,
}

impl LogitStore {
    fn create(directory: PathBuf) -> Self {
        std::fs::create_dir_all(&directory).expect("create trace directory");
        Self {
            directory,
            expected: Vec::new(),
            captured: Vec::new(),
        }
    }

    fn push(&mut self, agent: u32, row: usize, layer: usize, step: usize, logits: &[f32]) {
        let at = serde_json::json!({
            "stage": "Logits",
            "agent": format!("agent-{agent}"),
            "logical_row_start": row,
            "layer": layer,
            "segment": 0,
            "logical_step": step,
        });
        let payload = logits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        let file = format!("tensor-{:06}.bin", self.captured.len());
        std::fs::write(self.directory.join(&file), &payload).expect("write logit row");
        let expected = serde_json::json!({"at": at, "scalar": "F32", "shape": [1, logits.len()]});
        let mut captured = expected.clone();
        captured["file"] = serde_json::Value::from(file);
        captured["sha256"] = serde_json::Value::from(format!("{:x}", Sha256::digest(&payload)));
        self.expected.push(expected);
        self.captured.push(captured);
    }

    fn finish(self, provenance: &serde_json::Value) -> PathBuf {
        let path = self.directory.join("trace.json");
        let trace = serde_json::json!({
            "schema": Q1_TRACE_SCHEMA,
            "provenance": provenance,
            "coverage": {"expected": self.expected, "captured": self.captured, "errors": []},
        });
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&trace).expect("encode trace"),
        )
        .expect("write trace.json");
        path
    }
}

/// Q1 Class B input for the padded joint route: real rendered agent
/// prompts, teacher-forced (every path consumes the serial top-1 at every
/// step), one full-vocabulary logit row per agent and step from each path.
///
/// - `serial/`: `forward_gpu_last_logits`, one token, one slot (stock M1
///   target route);
/// - `joint_shared_ffn/`: `forward_gpu_decode_joint` over all agents with
///   the weight-shared dense FFN (production default);
/// - `joint_serial_ffn/`: the same with the serial builder's per-row fused
///   gate/up kernel (`HF2Q_QWEN_JOINT_FFN_SHARED=0`).
///
/// Then `python3 -B scripts/four_agent/q1_decision_policy.py envelope
/// $OUT/serial/trace.json $OUT/joint_shared_ffn/trace.json` (the gate), and
/// the same against `joint_serial_ffn` for the kill-switch route.
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF, HF2Q_DECODE_MV_EXT=1, HF2Q_QWEN_JOINT_Q1_PROMPTS, HF2Q_QWEN_JOINT_Q1_OUT and Apple Metal"]
fn qwen38_joint_decode_q1_envelope_trace() {
    let out = std::env::var_os(Q1_OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {Q1_OUT_ENV} to a new output directory"));
    assert!(!out.exists(), "{} already exists", out.display());
    let steps = std::env::var(Q1_STEPS_ENV)
        .ok()
        .map(|raw| raw.parse::<usize>().expect("integer step count"))
        .unwrap_or(32);
    let prompts = q1_prompts();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = load("HF2Q_QWEN38_GGUF");
    let mode = model
        .joint_decode_mode()
        .unwrap_or_else(|reason| panic!("joint decode not admitted: {reason}"));
    assert_eq!(
        mode,
        JointRouteMode::PaddedMvExt,
        "the Class B trace needs the Qwen3.8 padded policy \
         (HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1)"
    );
    let longest = prompts.iter().map(Vec::len).max().expect("prompts");
    let capacity = u32::try_from((longest + steps + 16).next_multiple_of(256)).expect("capacity");
    let mut serial = cache_with_capacity(&model, capacity);
    let mut joint = JOINT_ARMS
        .iter()
        .map(|_| cache_with_capacity(&model, capacity))
        .collect::<Vec<_>>();
    let layers = model.cfg.num_hidden_layers as usize;

    let mut next = Vec::with_capacity(prompts.len());
    for (slot, tokens) in prompts.iter().enumerate() {
        let slot = slot as u32;
        let reference = prefill_chunked(&model, &mut serial, slot, tokens);
        for cache in &mut joint {
            assert_eq!(
                bits(&reference),
                bits(&prefill_chunked(&model, cache, slot, tokens)),
                "agent {slot} prefill must be deterministic"
            );
        }
        next.push(argmax(&reference));
    }

    let mut left = LogitStore::create(out.join("serial"));
    let mut right = JOINT_ARMS
        .iter()
        .map(|(_, directory)| LogitStore::create(out.join(directory)))
        .collect::<Vec<_>>();
    for step in 0..steps {
        let rows = prompts
            .iter()
            .enumerate()
            .map(|(slot, tokens)| Qwen35JointRow {
                slot_id: SlotId(slot as u32),
                token: next[slot],
                position: (tokens.len() + step) as i32,
                output: Qwen35JointOutput::Logits,
            })
            .collect::<Vec<_>>();
        for (((dense_ffn, _), cache), store) in JOINT_ARMS.iter().zip(&mut joint).zip(&mut right) {
            let outputs = model
                .forward_gpu_decode_joint(&rows, cache, mode, *dense_ffn)
                .unwrap_or_else(|error| {
                    panic!("{} joint step {step}: {error:#}", dense_ffn.label())
                });
            for (row, output) in rows.iter().zip(outputs) {
                let logits = output.logits.expect("joint logits");
                store.push(row.slot_id.0, row.position as usize, layers, step, &logits);
            }
        }
        for row in &rows {
            let slot = row.slot_id.0;
            let serial_logits = model
                .forward_gpu_last_logits(&[row.token], &[row.position; 4], &mut serial, row.slot_id)
                .expect("serial step");
            left.push(slot, row.position as usize, layers, step, &serial_logits);
            // Teacher forcing: every cache consumes the serial decision.
            next[slot as usize] = argmax(&serial_logits);
        }
        for slot in 0..prompts.len() as u32 {
            let cursor = serial.sequence_len_for_slot(SlotId(slot)).ok();
            for cache in &joint {
                assert_eq!(
                    cache.sequence_len_for_slot(SlotId(slot)).ok(),
                    cursor,
                    "step {step} agent {slot} cursor"
                );
            }
        }
    }
    let provenance = serde_json::json!({
        "test": "qwen38_joint_decode_q1_envelope_trace",
        "artifact": std::env::var("HF2Q_QWEN38_GGUF").unwrap_or_default(),
        "prompts": std::env::var(Q1_PROMPTS_ENV).unwrap_or_default(),
        "prompt_tokens": prompts.iter().map(Vec::len).collect::<Vec<_>>(),
        "steps": steps,
        "joint_mode": mode.label(),
        "decode_mvn": std::env::var("HF2Q_DECODE_MVN").ok(),
        "decode_mv_ext": std::env::var("HF2Q_DECODE_MV_EXT").ok(),
        "teacher_forcing": "serial top-1",
    });
    let left_path = left.finish(&serde_json::json!({"path": "serial_m1", "run": provenance}));
    for ((dense_ffn, _), store) in JOINT_ARMS.iter().zip(right) {
        let path = store.finish(&serde_json::json!({
            "path": "joint",
            "dense_ffn": dense_ffn.label(),
            "run": provenance,
        }));
        assert!(Path::new(&path).is_file());
        eprintln!(
            "Q1 trace ({} dense FFN); run: python3 -B scripts/four_agent/q1_decision_policy.py \
             envelope {} {}",
            dense_ffn.label(),
            left_path.display(),
            path.display()
        );
    }
    assert!(Path::new(&left_path).is_file());
}
