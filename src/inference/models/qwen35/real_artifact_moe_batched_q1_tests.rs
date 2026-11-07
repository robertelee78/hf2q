//! Q1 Class B trace for the Qwen3.6 batched native-storage MoE route
//! (`HF2Q_QWEN_MOE_BATCHED`, ADR-063 Q4-QWEN36 F2) on real prompts.
//!
//! `qwen36_moe_route_q1_envelope_trace` prefills each prompt and decodes it
//! serially (`forward_gpu_last_logits`, one token, one slot) for
//! `HF2Q_QWEN_MOE_Q1_STEPS` steps (default 32), writing every
//! full-vocabulary logit row as an `hf2q.adr063.invariance.v2` store and the
//! consumed tokens as `tokens.json`. The MoE route is a process setting, so
//! the two arms are two processes on the same prompts:
//!
//! ```text
//! # reference arm: today's default route, greedy
//! HF2Q_QWEN36_GGUF=… HF2Q_QWEN_MOE_Q1_PROMPTS=prompts.json HF2Q_QWEN_MOE_Q1_OUT=off \
//!   cargo test --release --bin hf2q qwen36_moe_route_q1_envelope_trace -- --ignored --exact
//! # candidate arm: the batched route, teacher-forced on the reference tokens
//! HF2Q_QWEN_MOE_BATCHED=1 HF2Q_QWEN_MOE_Q1_FORCED=off/tokens.json HF2Q_QWEN36_GGUF=… \
//!   HF2Q_QWEN_MOE_Q1_PROMPTS=prompts.json HF2Q_QWEN_MOE_Q1_OUT=on \
//!   cargo test --release --bin hf2q qwen36_moe_route_q1_envelope_trace -- --ignored --exact
//! python3 -B scripts/four_agent/q1_decision_policy.py envelope off/serial/trace.json on/serial/trace.json
//! ```

use std::path::PathBuf;

use sha2::{Digest, Sha256};

use super::real_artifact_joint_decode_invariance_tests::cache_with_capacity;
use super::real_artifact_joint_decode_tests::{argmax, load, SLOTS};
use super::spec_decode::positions_for_range;
use crate::serve::multi_seq_kv::SlotId;

const PROMPTS_ENV: &str = "HF2Q_QWEN_MOE_Q1_PROMPTS";
const OUT_ENV: &str = "HF2Q_QWEN_MOE_Q1_OUT";
const STEPS_ENV: &str = "HF2Q_QWEN_MOE_Q1_STEPS";
const FORCED_ENV: &str = "HF2Q_QWEN_MOE_Q1_FORCED";
const TRACE_SCHEMA: &str = "hf2q.adr063.invariance.v2";
const PREFILL_CHUNK: usize = 512;

/// A JSON array of prompts, each an array of token ids (at least 16).
fn prompts() -> Vec<Vec<u32>> {
    let path = std::env::var_os(PROMPTS_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {PROMPTS_ENV} to a JSON array of token-id arrays"));
    let prompts: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read(&path).expect("read prompts")).expect("prompts JSON");
    assert!(!prompts.is_empty(), "{PROMPTS_ENV} holds no prompts");
    assert!(prompts.iter().all(|p| p.len() >= 16), "every prompt needs at least 16 tokens");
    prompts
}

fn write_store(directory: PathBuf, rows: &[(u32, usize, usize, usize, Vec<f32>)], provenance: serde_json::Value) -> PathBuf {
    std::fs::create_dir_all(&directory).expect("create store");
    let (mut expected, mut captured) = (Vec::new(), Vec::new());
    for (index, (agent, row, layer, step, logits)) in rows.iter().enumerate() {
        let at = serde_json::json!({
            "stage": "Logits",
            "agent": format!("agent-{agent}"),
            "logical_row_start": row,
            "layer": layer,
            "segment": 0,
            "logical_step": step,
        });
        let payload = logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        let file = format!("tensor-{index:06}.bin");
        std::fs::write(directory.join(&file), &payload).expect("write logit row");
        let entry = serde_json::json!({"at": at, "scalar": "F32", "shape": [1, logits.len()]});
        let mut with_file = entry.clone();
        with_file["file"] = serde_json::Value::from(file);
        with_file["sha256"] = serde_json::Value::from(format!("{:x}", Sha256::digest(&payload)));
        expected.push(entry);
        captured.push(with_file);
    }
    let path = directory.join("trace.json");
    let trace = serde_json::json!({
        "schema": TRACE_SCHEMA,
        "provenance": provenance,
        "coverage": {"expected": expected, "captured": captured, "errors": []},
    });
    std::fs::write(&path, serde_json::to_vec_pretty(&trace).expect("encode trace")).expect("write trace");
    path
}

#[test]
#[ignore = "requires HF2Q_QWEN36_GGUF, HF2Q_QWEN_MOE_Q1_PROMPTS, HF2Q_QWEN_MOE_Q1_OUT and Apple Metal"]
fn qwen36_moe_route_q1_envelope_trace() {
    let out = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {OUT_ENV} to a new output directory"));
    assert!(!out.exists(), "{} already exists", out.display());
    let steps = std::env::var(STEPS_ENV)
        .ok()
        .map(|raw| raw.parse::<usize>().expect("integer step count"))
        .unwrap_or(32);
    let forced: Option<Vec<Vec<u32>>> = std::env::var_os(FORCED_ENV).map(|path| {
        serde_json::from_slice(&std::fs::read(path).expect("read forced tokens")).expect("forced tokens JSON")
    });
    let prompts = prompts();
    if let Some(forced) = &forced {
        assert_eq!(forced.len(), prompts.len(), "one forced token list per prompt");
        assert!(forced.iter().all(|t| t.len() >= steps), "forced lists cover every step");
    }
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = load("HF2Q_QWEN36_GGUF");
    let layers = model.cfg.num_hidden_layers as usize;
    let longest = prompts.iter().map(Vec::len).max().expect("prompts");
    let capacity = u32::try_from((longest + steps + 16).next_multiple_of(256)).expect("capacity");

    let mut rows = Vec::with_capacity(prompts.len() * steps);
    let mut consumed = vec![Vec::with_capacity(steps); prompts.len()];
    for (group, chunk) in prompts.chunks(SLOTS as usize).enumerate() {
        let mut cache = cache_with_capacity(&model, capacity);
        for (offset, tokens) in chunk.iter().enumerate() {
            let agent = group * SLOTS as usize + offset;
            let slot = SlotId(offset as u32);
            let mut logits = Vec::new();
            for start in (0..tokens.len()).step_by(PREFILL_CHUNK) {
                let piece = &tokens[start..(start + PREFILL_CHUNK).min(tokens.len())];
                logits = model
                    .forward_gpu_last_logits(piece, &positions_for_range(start as i32, piece.len()), &mut cache, slot)
                    .expect("chunked prefill");
            }
            for step in 0..steps {
                // The step-0 row is the last prefill row; later rows are one-token decodes.
                let position = tokens.len() + step;
                rows.push((agent as u32, position, layers, step, logits.clone()));
                let token = forced.as_ref().map_or_else(|| argmax(&logits), |f| f[agent][step]);
                consumed[agent].push(token);
                logits = model
                    .forward_gpu_last_logits(&[token], &[position as i32; 4], &mut cache, slot)
                    .expect("decode step");
            }
        }
    }
    let provenance = serde_json::json!({
        "test": "qwen36_moe_route_q1_envelope_trace",
        "artifact": std::env::var("HF2Q_QWEN36_GGUF").unwrap_or_default(),
        "prompts": std::env::var(PROMPTS_ENV).unwrap_or_default(),
        "prompt_tokens": prompts.iter().map(Vec::len).collect::<Vec<_>>(),
        "steps": steps,
        "moe_batched": std::env::var("HF2Q_QWEN_MOE_BATCHED").ok(),
        "teacher_forcing": if forced.is_some() { "forced tokens file" } else { "own greedy top-1" },
        "forced": std::env::var(FORCED_ENV).ok(),
    });
    std::fs::create_dir_all(&out).expect("create out");
    std::fs::write(out.join("tokens.json"), serde_json::to_vec(&consumed).expect("tokens")).expect("write tokens");
    let path = write_store(out.join("serial"), &rows, serde_json::json!({"path": "serial_m1", "run": provenance}));
    eprintln!("Q1 trace written: {}", path.display());
}
