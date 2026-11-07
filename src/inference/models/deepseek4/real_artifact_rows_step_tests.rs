//! Official-artifact proof that rows steps are bit-identical to one-token
//! verifier steps over token-wise prompt work (the recovery tail and short
//! cached suffixes), including the recovery-anchor replay.
//!
//! Run on Apple Silicon with the converted checkpoint:
//! `HF2Q_DEEPSEEK4_GGUF=/path/to/deepseek4.gguf cargo test --release --bin hf2q \
//!   official_artifact_rows_step_matches_one_token_steps_bitwise -- --ignored --nocapture`
//! Run it with `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC` both unset/0 and 1.

use super::cache::Deepseek4Cache;
use super::joint_decode::MAX_JOINT_DECODE_ROWS;
use super::model::Deepseek4Model;
use super::real_artifact_decode_cohort_tests::{assert_cache_exact, assert_exact_f32, read_f32};
use super::real_artifact_tests::official_artifact;

/// Prefix rows installed by matrix prefill, then token-wise chunk widths.
/// The recovery anchor is captured before the last chunk, as serving
/// captures it before the tail.
struct Case {
    prefix: usize,
    chunks: &'static [usize],
}

const CASES: [Case; 5] = [
    // The run's cold prompt: 749 matrix rows, then the 8-token tail.
    Case { prefix: 749, chunks: &[8] },
    // A 32-token cached suffix: three rows steps of 8, the anchor, the tail.
    Case { prefix: 812, chunks: &[8, 8, 8, 8] },
    // Window ring wrap at 128, a ratio-4 emit and the ratio-128 block at 127.
    Case { prefix: 123, chunks: &[8, 8] },
    // Inside the first window, before any compressed row.
    Case { prefix: 3, chunks: &[5, 8] },
    // Every other width the step admits, across the 256 ratio-128 boundary.
    Case { prefix: 233, chunks: &[2, 3, 4, 5, 6, 7] },
];

fn token(case: usize, row: usize) -> u32 {
    ((row * 389 + case * 7_919 + 23) % 120_000) as u32
}

fn prefilled(model: &mut Deepseek4Model, case: usize, prefix: usize, capacity: usize) -> Deepseek4Cache {
    let tokens = (0..prefix).map(|row| token(case, row)).collect::<Vec<_>>();
    let mut cache = model.allocate_cache(capacity).expect("allocate cache");
    model
        .forward_verifier_prompt(&tokens, &mut cache)
        .expect("install matrix prefix");
    assert_eq!(cache.position(), prefix);
    cache
}

#[test]
#[ignore = "loads the official checkpoint and proves rows steps equal one-token steps bitwise"]
fn official_artifact_rows_step_matches_one_token_steps_bitwise() {
    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    assert_eq!(model.joint_decode_unavailable_reason(), None);
    let vocab = model.cfg.vocab_size as usize;
    let state_row = model.cfg.hyper_connection_count as usize * model.cfg.hidden_size as usize;

    for (index, case) in CASES.iter().enumerate() {
        let total: usize = case.chunks.iter().sum();
        assert!(case.chunks.iter().all(|&rows| (2..=MAX_JOINT_DECODE_ROWS).contains(&rows)));
        // A DeepSeek-V4 cache needs at least one full sliding window: the window ring is
        // min(capacity, sliding_window) rows and both the matrix-prefill and the one-token
        // attention paths require it to equal the configured window (serving caches start at
        // 131,072 tokens). Case 3's tiny prefix would otherwise get an 18-row ring.
        let capacity = (case.prefix + total + 2).max(model.cfg.sliding_window as usize);
        eprintln!("rows step case {index}: prefix {}, chunks {:?}, capacity {capacity}", case.prefix, case.chunks);
        let mut serial = prefilled(&mut model, index, case.prefix, capacity);
        let mut stepped = prefilled(&mut model, index, case.prefix, capacity);
        assert_cache_exact(&format!("case {index} prefix"), &serial, &stepped);

        // Width-one and over-wide steps are refused before any GPU work.
        for rows in [1, MAX_JOINT_DECODE_ROWS + 1] {
            let tokens = (0..rows).map(|row| token(index, case.prefix + row)).collect::<Vec<_>>();
            assert!(model.forward_verifier_rows_step(&tokens, &mut stepped).is_err());
            assert!(!stepped.is_poisoned());
            assert_eq!(stepped.position(), case.prefix);
        }

        let mut row = case.prefix;
        let mut anchor = None;
        let mut tail_reference = None;
        for (chunk, &rows) in case.chunks.iter().enumerate() {
            let tokens = (row..row + rows).map(|r| token(index, r)).collect::<Vec<_>>();
            let mut serial_states = Vec::with_capacity(rows);
            let mut serial_logits = Vec::with_capacity(rows);
            for &token in &tokens {
                let state = model
                    .forward_verifier_one(token, &mut serial)
                    .unwrap_or_else(|error| panic!("case {index} serial row: {error:#}"));
                let logits = model.forward_logits(&state).expect("serial head");
                serial_states.push(read_f32(&state, "serial state"));
                serial_logits.push(read_f32(&logits, "serial logits"));
            }
            let label = format!("case {index} chunk {chunk} rows {row}..{}", row + rows);
            if chunk == case.chunks.len() - 1 {
                anchor = Some(stepped.snapshot().expect("anchor snapshot"));
            }
            let state = model
                .forward_verifier_rows_step(&tokens, &mut stepped)
                .unwrap_or_else(|error| panic!("{label}: {error:#}"));
            assert_eq!(state.shape()[0], rows, "{label} state rows");
            let logits = model.forward_logits(&state).expect("rows step head");
            assert_eq!(logits.shape(), [rows, vocab]);
            let state = read_f32(&state, "rows step state");
            let logits = read_f32(&logits, "rows step logits");
            for r in 0..rows {
                assert_exact_f32(
                    &format!("{label} row {r} state"),
                    &serial_states[r],
                    &state[r * state_row..(r + 1) * state_row],
                );
                assert_exact_f32(
                    &format!("{label} row {r} logits"),
                    &serial_logits[r],
                    &logits[r * vocab..(r + 1) * vocab],
                );
            }
            assert_cache_exact(&label, &serial, &stepped);
            row += rows;
            if chunk == case.chunks.len() - 1 {
                tail_reference = Some((state, logits, tokens));
            }
        }

        // Anchor replay: restore the anchor and run the final chunk again.
        {
            let anchor = anchor.expect("anchor");
            let (state, logits, tokens) = tail_reference.expect("tail reference");
            stepped.restore(&anchor).expect("restore anchor");
            let replay = model
                .forward_verifier_rows_step(&tokens, &mut stepped)
                .unwrap_or_else(|error| panic!("case {index} anchor replay: {error:#}"));
            let replay_logits = model.forward_logits(&replay).expect("replay head");
            assert_exact_f32(
                &format!("case {index} replay state"),
                &state,
                &read_f32(&replay, "replay state"),
            );
            assert_exact_f32(
                &format!("case {index} replay logits"),
                &logits,
                &read_f32(&replay_logits, "replay logits"),
            );
            assert_cache_exact(&format!("case {index} replay"), &serial, &stepped);
        }

        eprintln!("rows step case {index}: rows, logits, caches and anchor replay bitwise equal");

        // A rejected commit gate poisons the cache without publishing.
        let position = stepped.position();
        let tokens = (0..2).map(|r| token(index, position + r)).collect::<Vec<_>>();
        let rejected = model.forward_verifier_rows_step_with_commit_gate(
            &tokens[..],
            &mut stepped,
            || anyhow::bail!("synthetic rows step supervisor rejection"),
        );
        assert!(rejected.is_err());
        assert!(stepped.is_poisoned());
        assert_eq!(stepped.position(), position);
    }
}
