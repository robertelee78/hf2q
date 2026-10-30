//! Official-artifact cohort-invariance gates and the Q1 Class B trace for
//! Qwen joint B-row decode.
//!
//! **Cohort invariance (Q1 §3.1 item 2, Class A).** A joint row must produce
//! the same bytes whichever tenants share its step and wherever it sits in
//! the packed rows. Each variant below prefills four slots identically in
//! its own cache, then advances the target slot for several steps inside a
//! different cohort (alone, pairs, a triple, and a full cohort with the
//! target packed last). The target's logits, post-norm hidden, cursor,
//! ping-pong parity, KV codes/scales and conv/recurrent bytes must be
//! bitwise equal across every variant: in exact mode, and in the padded
//! `mul_mv_ext` mode, whose packed width is always 4.
//!
//! **Q1 Class B trace.** `qwen38_joint_decode_q1_envelope_trace` advances
//! real rendered agent prompts teacher-forced through serial one-token
//! forwards and through joint steps, and writes both full-vocabulary logit
//! row sets as `hf2q.adr063.invariance.v2` stores for
//! `q1_decision_policy.py envelope`.
//!
//! ```text
//! HF2Q_QWEN36_GGUF=/path/qwen36.gguf cargo test --release --lib \
//!   qwen36_joint_decode_rows_are_cohort_invariant_bitwise -- --ignored --exact
//! HF2Q_QWEN38_GGUF=/path/qwen38.gguf cargo test --release --lib \
//!   qwen38_joint_decode_mvn_rows_are_cohort_invariant_bitwise -- --ignored --exact
//! HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 HF2Q_QWEN38_GGUF=/path/qwen38.gguf \
//!   cargo test --release --lib \
//!   qwen38_joint_decode_padded_mv_ext_rows_are_cohort_invariant_bitwise -- --ignored --exact
//! ```

use std::path::{Path, PathBuf};

use mlx_native::MlxBuffer;
use sha2::{Digest, Sha256};

use super::joint_decode_routes::JointRouteMode;
use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::real_artifact_joint_decode_tests::{
    argmax, bits, buffer_bytes, hidden_bits, load, prompt, SLOTS,
};
use super::spec_decode::positions_for_range;
use super::{Qwen35JointOutput, Qwen35JointRow};
use crate::serve::multi_seq_kv::SlotId;

const TARGET: u32 = 2;
const STEPS: usize = 4;
const CAPACITY: u32 = 160;
/// Packed orders containing `TARGET` at packed index 0, 1, 2 and 3, with
/// cohort sizes 1 through 4.
const VARIANTS: [&[u32]; 6] = [
    &[2],
    &[0, 2],
    &[2, 3],
    &[1, 2, 3],
    &[2, 3, 1],
    &[0, 1, 3, 2],
];

fn cache_with_capacity(model: &Qwen35Model, capacity: u32) -> HybridKvCache {
    model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&model.cfg, device, capacity, SLOTS, true)
        })
        .expect("allocate four-slot TQ cache")
}

/// Bytes of `slot`'s region in a buffer whose outermost axis is the slot.
fn slot_region(buffer: &MlxBuffer, slot: u32) -> Vec<u8> {
    let bytes = buffer_bytes(buffer);
    assert_eq!(bytes.len() % SLOTS as usize, 0, "buffer not split by slot");
    let per_slot = bytes.len() / SLOTS as usize;
    bytes[slot as usize * per_slot..(slot as usize + 1) * per_slot].to_vec()
}

/// Every observable byte of one slot's target cache state.
#[derive(Debug, PartialEq, Eq)]
struct SlotState {
    cursor: Option<u32>,
    full_attn: Vec<(u32, Vec<Vec<u8>>)>,
    linear_attn: Vec<(bool, Vec<Vec<u8>>)>,
}

fn slot_state(cache: &HybridKvCache, slot: u32) -> SlotState {
    let index = slot as usize;
    SlotState {
        cursor: cache.sequence_len_for_slot(SlotId(slot)).ok(),
        full_attn: cache
            .full_attn
            .iter()
            .map(|layer| {
                let mut regions = Vec::new();
                if let Some(tq) = &layer.tq {
                    for buffer in [&tq.k_packed, &tq.k_norms, &tq.v_packed, &tq.v_norms] {
                        regions.push(slot_region(buffer, slot));
                    }
                }
                for buffer in [&layer.k, &layer.v].into_iter().flatten() {
                    regions.push(slot_region(buffer, slot));
                }
                (layer.current_len[index], regions)
            })
            .collect(),
        linear_attn: cache
            .linear_attn
            .iter()
            .map(|layer| {
                (
                    layer.pp_flipped[index],
                    [
                        &layer.conv_state,
                        &layer.conv_state_scratch,
                        &layer.recurrent,
                        &layer.recurrent_scratch,
                    ]
                    .into_iter()
                    .map(|buffer| slot_region(buffer, slot))
                    .collect(),
                )
            })
            .collect(),
    }
}

/// One variant's observations of the target row: per-step logits and
/// hidden bits, then the final slot state.
struct TargetTrace {
    logits: Vec<Vec<u32>>,
    hidden: Vec<Vec<u32>>,
    state: SlotState,
}

fn run_variant(model: &Qwen35Model, cohort: &[u32], mode: JointRouteMode) -> TargetTrace {
    let hidden_size = model.cfg.hidden_size as usize;
    let mut cache = cache_with_capacity(model, CAPACITY);
    let mut next = Vec::with_capacity(SLOTS as usize);
    let mut position = Vec::with_capacity(SLOTS as usize);
    for slot in 0..SLOTS {
        let tokens = prompt(slot, 0);
        let logits = model
            .forward_gpu_last_logits(
                &tokens,
                &positions_for_range(0, tokens.len()),
                &mut cache,
                SlotId(slot),
            )
            .expect("prefill");
        next.push(argmax(&logits));
        position.push(tokens.len() as i32);
    }
    let mut target_logits = Vec::with_capacity(STEPS);
    let mut target_hidden = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let rows = cohort
            .iter()
            .map(|&slot| Qwen35JointRow {
                slot_id: SlotId(slot),
                token: next[slot as usize],
                position: position[slot as usize],
                output: if slot == TARGET {
                    Qwen35JointOutput::LogitsAndHidden
                } else {
                    Qwen35JointOutput::GreedyToken
                },
            })
            .collect::<Vec<_>>();
        let outputs = model
            .forward_gpu_decode_joint(&rows, &mut cache, mode)
            .unwrap_or_else(|error| panic!("cohort {cohort:?} step {step}: {error:#}"));
        for (row, output) in rows.iter().zip(outputs) {
            let slot = row.slot_id.0 as usize;
            next[slot] = match row.output {
                Qwen35JointOutput::GreedyToken => output.greedy_token.expect("greedy token"),
                _ => {
                    let logits = output.logits.expect("target logits");
                    target_logits.push(bits(&logits));
                    target_hidden.push(hidden_bits(
                        output.nextn_hidden.as_ref().expect("target hidden"),
                        hidden_size,
                    ));
                    argmax(&logits)
                }
            };
            position[slot] += 1;
        }
    }
    TargetTrace {
        logits: target_logits,
        hidden: target_hidden,
        state: slot_state(&cache, TARGET),
    }
}

fn run_cohort_invariance(env: &str, mode: JointRouteMode) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = load(env);
    assert_eq!(
        model.joint_decode_mode(),
        Ok(mode),
        "official artifact must be admitted in {} mode",
        mode.label()
    );
    let reference = run_variant(&model, VARIANTS[0], mode);
    assert_eq!(reference.logits.len(), STEPS);
    for &cohort in &VARIANTS[1..] {
        let variant = run_variant(&model, cohort, mode);
        for step in 0..STEPS {
            assert!(
                variant.logits[step] == reference.logits[step],
                "{} mode: target logits differ at step {step} in cohort {cohort:?} vs alone",
                mode.label()
            );
            assert!(
                variant.hidden[step] == reference.hidden[step],
                "{} mode: target hidden differs at step {step} in cohort {cohort:?} vs alone",
                mode.label()
            );
        }
        assert!(
            variant.state == reference.state,
            "{} mode: target cache state differs in cohort {cohort:?} vs alone",
            mode.label()
        );
    }
}

#[test]
#[ignore = "requires HF2Q_QWEN36_GGUF (official Qwen3.6 artifact) and Apple Metal"]
fn qwen36_joint_decode_rows_are_cohort_invariant_bitwise() {
    assert_ne!(
        std::env::var("HF2Q_DECODE_MV_EXT").as_deref(),
        Ok("1"),
        "the exact arm needs HF2Q_DECODE_MV_EXT unset"
    );
    run_cohort_invariance("HF2Q_QWEN36_GGUF", JointRouteMode::Exact);
}

#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (official Qwen3.8 artifact) and Apple Metal"]
fn qwen38_joint_decode_mvn_rows_are_cohort_invariant_bitwise() {
    assert_ne!(
        std::env::var("HF2Q_DECODE_MV_EXT").as_deref(),
        Ok("1"),
        "the exact arm needs HF2Q_DECODE_MV_EXT unset"
    );
    run_cohort_invariance("HF2Q_QWEN38_GGUF", JointRouteMode::Exact);
}

#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF, HF2Q_DECODE_MVN=0, HF2Q_DECODE_MV_EXT=1 and Apple Metal"]
fn qwen38_joint_decode_padded_mv_ext_rows_are_cohort_invariant_bitwise() {
    for (name, value) in [("HF2Q_DECODE_MVN", "0"), ("HF2Q_DECODE_MV_EXT", "1")] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(value),
            "the padded arm requires externally-set {name}={value}"
        );
    }
    run_cohort_invariance("HF2Q_QWEN38_GGUF", JointRouteMode::PaddedMvExt);
}

// ---------------------------------------------------------------------------
// Q1 Class B trace on real prompts
// ---------------------------------------------------------------------------

const Q1_PROMPTS_ENV: &str = "HF2Q_QWEN_JOINT_Q1_PROMPTS";
const Q1_OUT_ENV: &str = "HF2Q_QWEN_JOINT_Q1_OUT";
const Q1_STEPS_ENV: &str = "HF2Q_QWEN_JOINT_Q1_STEPS";
const Q1_TOKENIZER_ENV: &str = "HF2Q_QWEN35_TOKENIZER";
const Q1_TRACE_SCHEMA: &str = "hf2q.adr063.invariance.v2";
const PREFILL_CHUNK: usize = 512;

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

/// Q1 Class B input for the joint route: real rendered agent prompts,
/// teacher-forced (both paths consume the serial top-1 at every step), one
/// full-vocabulary logit row per agent and step from each path.
///
/// - left store (`serial/`): `forward_gpu_last_logits`, one token, one slot
///   (stock M1 target route);
/// - right store (`joint/`): `forward_gpu_decode_joint` over all agents in
///   the worker's mode (padded `mul_mv_ext` under the Qwen3.8 policy).
///
/// Then: `python3 -B scripts/four_agent/q1_decision_policy.py envelope
/// $OUT/serial/trace.json $OUT/joint/trace.json`.
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF, HF2Q_QWEN_JOINT_Q1_PROMPTS, HF2Q_QWEN_JOINT_Q1_OUT and Apple Metal"]
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
    let longest = prompts.iter().map(Vec::len).max().expect("prompts");
    let capacity = u32::try_from((longest + steps + 16).next_multiple_of(256)).expect("capacity");
    let mut serial = cache_with_capacity(&model, capacity);
    let mut joint = cache_with_capacity(&model, capacity);
    let layers = model.cfg.num_hidden_layers as usize;

    let mut next = Vec::with_capacity(prompts.len());
    for (slot, tokens) in prompts.iter().enumerate() {
        let slot = slot as u32;
        let a = prefill_chunked(&model, &mut serial, slot, tokens);
        let b = prefill_chunked(&model, &mut joint, slot, tokens);
        assert_eq!(
            bits(&a),
            bits(&b),
            "agent {slot} prefill must be deterministic"
        );
        next.push(argmax(&a));
    }

    let mut left = LogitStore::create(out.join("serial"));
    let mut right = LogitStore::create(out.join("joint"));
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
        let outputs = model
            .forward_gpu_decode_joint(&rows, &mut joint, mode)
            .unwrap_or_else(|error| panic!("joint step {step}: {error:#}"));
        for (row, output) in rows.iter().zip(outputs) {
            let slot = row.slot_id.0;
            let serial_logits = model
                .forward_gpu_last_logits(&[row.token], &[row.position; 4], &mut serial, row.slot_id)
                .expect("serial step");
            let joint_logits = output.logits.expect("joint logits");
            let logical_row = row.position as usize;
            left.push(slot, logical_row, layers, step, &serial_logits);
            right.push(slot, logical_row, layers, step, &joint_logits);
            // Teacher forcing: both caches consume the serial decision.
            next[slot as usize] = argmax(&serial_logits);
        }
        for slot in 0..prompts.len() as u32 {
            assert_eq!(
                serial.sequence_len_for_slot(SlotId(slot)).ok(),
                joint.sequence_len_for_slot(SlotId(slot)).ok(),
                "step {step} agent {slot} cursor"
            );
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
    let right_path = right.finish(&serde_json::json!({"path": "joint", "run": provenance}));
    eprintln!(
        "Q1 trace written; run: python3 -B scripts/four_agent/q1_decision_policy.py envelope {} {}",
        left_path.display(),
        right_path.display()
    );
    assert!(Path::new(&left_path).is_file() && Path::new(&right_path).is_file());
}
