//! Official-artifact proof that joint decode is bit-identical to serial decode.
//!
//! Run on Apple Silicon with the converted checkpoint:
//! `HF2Q_DEEPSEEK4_GGUF=/path/to/deepseek4.gguf cargo test --release --bin hf2q \
//!   official_artifact_joint_decode_matches_serial_decode_bitwise -- --ignored --nocapture`
//!
//! `official_artifact_long_context_joint_decode_matches_serial_decode_bitwise`
//! repeats the proof at long contexts (same env; any DeepSeek-V4 GGUF the
//! loader accepts, including the ds4 0731 file). The process's
//! `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC` and `HF2Q_DEEPSEEK_BATCHED_ATTN_CORES`
//! are read at model load as in serving.

use super::cache::Deepseek4Cache;
use super::joint_decode::MAX_JOINT_DECODE_ROWS;
use super::model::Deepseek4Model;
use super::real_artifact_decode_cohort_tests::{assert_cache_exact, assert_exact_f32, read_f32};
use super::real_artifact_tests::official_artifact;

const LANES: usize = 8;
/// Unequal prompt lengths: lanes inside their first sliding window, one a
/// token short of a ratio-4 group, and lanes that cross ratio-4 and
/// ratio-128 compression boundaries at different steps of the decode span.
const PREFIX_ROWS: [usize; LANES] = [148, 131, 3, 254, 200, 127, 64, 9];
const DECODE_STEPS: usize = 140;
/// Ragged step widths (B = 8, 4, 3, 5, 2, 6, 7) cover every width the model
/// admits, so every lane advances at its own rate and joint rows never share
/// a position by construction.
const STEP_LANES: [&[usize]; 7] = [
    &[0, 1, 2, 3, 4, 5, 6, 7],
    &[0, 1, 2, 3],
    &[0, 2, 3],
    &[1, 3, 4, 5, 7],
    &[1, 3],
    &[0, 2, 4, 5, 6, 7],
    &[1, 2, 3, 4, 5, 6, 7],
];

fn prefix_tokens(lane: usize) -> Vec<u32> {
    (0..PREFIX_ROWS[lane])
        .map(|row| ((row * 97 + lane * 193 + 11) % 120_000) as u32)
        .collect()
}

fn step_token(step: usize, lane: usize) -> u32 {
    ((step * 389 + lane * 7_919 + 17) % 120_000) as u32
}

fn prefilled_caches(model: &mut Deepseek4Model, capacity: usize) -> Vec<Deepseek4Cache> {
    (0..LANES)
        .map(|lane| {
            let mut cache = model.allocate_cache(capacity).expect("allocate cache");
            model
                .forward_verifier_prompt(&prefix_tokens(lane), &mut cache)
                .expect("install prefix");
            assert_eq!(cache.position(), PREFIX_ROWS[lane]);
            cache
        })
        .collect()
}

fn selected<'a>(caches: &'a mut [Deepseek4Cache], lanes: &[usize]) -> Vec<&'a mut Deepseek4Cache> {
    caches
        .iter_mut()
        .enumerate()
        .filter(|(lane, _)| lanes.contains(lane))
        .map(|(_, cache)| cache)
        .collect()
}

#[test]
#[ignore = "loads the official checkpoint and proves joint decode equals serial decode bitwise"]
fn official_artifact_joint_decode_matches_serial_decode_bitwise() {
    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    eprintln!(
        "artifact {} matvec {:?} attn cores {:?}",
        path.display(),
        // The loader resolved the family the same way.
        super::residency::MatvecKernels::from_env().expect("matvec switch"),
        model.attn_cores
    );
    assert!(model.cfg.compress_ratios.contains(&4));
    assert!(model.cfg.compress_ratios.contains(&128));
    assert_eq!(model.joint_decode_unavailable_reason(), None);
    assert!(LANES <= MAX_JOINT_DECODE_ROWS);
    let capacity = PREFIX_ROWS.iter().max().unwrap() + DECODE_STEPS + 1;
    let mut serial_caches = prefilled_caches(&mut model, capacity);
    let mut joint_caches = prefilled_caches(&mut model, capacity);
    let vocab = model.cfg.vocab_size as usize;
    let state_row = model.cfg.hyper_connection_count as usize * model.cfg.hidden_size as usize;

    // Width-one and misaligned requests are rejected before any GPU work.
    let single = model.forward_verifier_decode_joint_with_commit_gate(
        &[step_token(0, 0)],
        &mut selected(&mut joint_caches, &[0]),
        || Ok(()),
    );
    assert!(single.is_err());
    let misaligned = model.forward_verifier_decode_joint_with_commit_gate(
        &[step_token(0, 0)],
        &mut selected(&mut joint_caches, &[0, 1]),
        || Ok(()),
    );
    assert!(misaligned.is_err());
    assert!(joint_caches.iter().all(|cache| !cache.is_poisoned()));

    for step in 0..DECODE_STEPS {
        let lanes = STEP_LANES[step % STEP_LANES.len()];
        let tokens = lanes
            .iter()
            .map(|&lane| step_token(step, lane))
            .collect::<Vec<_>>();
        let mut serial_states = Vec::with_capacity(lanes.len());
        let mut serial_logits = Vec::with_capacity(lanes.len());
        for (&lane, &token) in lanes.iter().zip(&tokens) {
            let state = model
                .forward_verifier_one(token, &mut serial_caches[lane])
                .unwrap_or_else(|error| panic!("serial step {step} lane {lane}: {error:#}"));
            let logits = model
                .forward_logits(&state)
                .unwrap_or_else(|error| panic!("serial step {step} lane {lane} head: {error:#}"));
            serial_states.push(read_f32(&state, "serial state"));
            serial_logits.push(read_f32(&logits, "serial logits"));
        }

        let joint_state = model
            .forward_verifier_decode_joint_with_commit_gate(
                &tokens,
                &mut selected(&mut joint_caches, lanes),
                || Ok(()),
            )
            .unwrap_or_else(|error| panic!("joint step {step} lanes {lanes:?}: {error:#}"));
        let joint_logits = model
            .forward_logits(&joint_state)
            .unwrap_or_else(|error| panic!("joint step {step} head: {error:#}"));
        assert_eq!(joint_logits.shape(), [lanes.len(), vocab]);
        let joint_state = read_f32(&joint_state, "joint state");
        let joint_logits = read_f32(&joint_logits, "joint logits");
        for (row, &lane) in lanes.iter().enumerate() {
            assert_exact_f32(
                &format!("step {step} lane {lane} state"),
                &serial_states[row],
                &joint_state[row * state_row..(row + 1) * state_row],
            );
            assert_exact_f32(
                &format!("step {step} lane {lane} logits"),
                &serial_logits[row],
                &joint_logits[row * vocab..(row + 1) * vocab],
            );
        }
        for lane in 0..LANES {
            assert_cache_exact(
                &format!("step {step} lane {lane} cache"),
                &serial_caches[lane],
                &joint_caches[lane],
            );
        }
    }

    // A rejected commit gate poisons every joint lane without publishing.
    let positions = joint_caches
        .iter()
        .map(Deepseek4Cache::position)
        .collect::<Vec<_>>();
    let rejected = model.forward_verifier_decode_joint_with_commit_gate(
        &[step_token(DECODE_STEPS, 0), step_token(DECODE_STEPS, 3)],
        &mut selected(&mut joint_caches, &[0, 3]),
        || anyhow::bail!("synthetic joint supervisor rejection"),
    );
    assert!(rejected.is_err());
    assert!(joint_caches[0].is_poisoned() && joint_caches[3].is_poisoned());
    assert!(!joint_caches[1].is_poisoned() && !joint_caches[2].is_poisoned());
    assert_eq!(
        joint_caches
            .iter()
            .map(Deepseek4Cache::position)
            .collect::<Vec<_>>(),
        positions
    );
}

/// Long-context lanes: ratio-4 rows on the flash path at different widths
/// (1145 -> width 414, 2040 -> 638..640, 2100 -> 640; window 128 + top-k), a row
/// with two indexer top-k blocks (4220 -> 1055 keys), and one short row on the
/// score/reduce path, so one joint layer mixes flash width groups (permuted
/// flash batches) with score/reduce rows. Ratio-128 blocks complete at
/// 1151, 2047 and 4223 and ratio-4 blocks every fourth step inside the span.
const LONG_PREFIX_ROWS: [usize; 5] = [1145, 2040, 2100, 4220, 300];
const LONG_DECODE_STEPS: usize = 10;
const LONG_STEP_LANES: [&[usize]; 4] = [&[0, 1, 2, 3, 4], &[3, 1, 4, 0], &[2, 4, 1], &[0, 3]];

fn long_prefix_tokens(lane: usize) -> Vec<u32> {
    (0..LONG_PREFIX_ROWS[lane])
        .map(|row| ((row * 131 + lane * 977 + 23) % 120_000) as u32)
        .collect()
}

fn long_prefilled_caches(model: &mut Deepseek4Model, capacity: usize) -> Vec<Deepseek4Cache> {
    (0..LONG_PREFIX_ROWS.len())
        .map(|lane| {
            let mut cache = model.allocate_cache(capacity).expect("allocate cache");
            model
                .forward_verifier_prompt(&long_prefix_tokens(lane), &mut cache)
                .expect("install long prefix");
            assert_eq!(cache.position(), LONG_PREFIX_ROWS[lane]);
            cache
        })
        .collect()
}

#[test]
#[ignore = "loads a DeepSeek-V4 checkpoint and proves long-context joint decode equals serial decode bitwise"]
fn official_artifact_long_context_joint_decode_matches_serial_decode_bitwise() {
    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load artifact {}: {error:#}", path.display()));
    eprintln!(
        "artifact {} matvec {:?} attn cores {:?}",
        path.display(),
        // The loader resolved the family the same way.
        super::residency::MatvecKernels::from_env().expect("matvec switch"),
        model.attn_cores
    );
    assert_eq!(model.joint_decode_unavailable_reason(), None);
    let lanes_total = LONG_PREFIX_ROWS.len();
    assert!(lanes_total <= MAX_JOINT_DECODE_ROWS);
    let capacity = LONG_PREFIX_ROWS.iter().max().unwrap() + LONG_DECODE_STEPS + 1;
    let mut serial_caches = long_prefilled_caches(&mut model, capacity);
    let mut joint_caches = long_prefilled_caches(&mut model, capacity);
    let vocab = model.cfg.vocab_size as usize;
    let state_row = model.cfg.hyper_connection_count as usize * model.cfg.hidden_size as usize;

    for step in 0..LONG_DECODE_STEPS {
        let lanes = LONG_STEP_LANES[step % LONG_STEP_LANES.len()];
        let tokens = lanes
            .iter()
            .map(|&lane| step_token(step, lane))
            .collect::<Vec<_>>();
        let mut serial_states = Vec::with_capacity(lanes.len());
        let mut serial_logits = Vec::with_capacity(lanes.len());
        for (&lane, &token) in lanes.iter().zip(&tokens) {
            let state = model
                .forward_verifier_one(token, &mut serial_caches[lane])
                .unwrap_or_else(|error| panic!("serial step {step} lane {lane}: {error:#}"));
            let logits = model
                .forward_logits(&state)
                .unwrap_or_else(|error| panic!("serial step {step} lane {lane} head: {error:#}"));
            serial_states.push(read_f32(&state, "serial state"));
            serial_logits.push(read_f32(&logits, "serial logits"));
        }
        // Entry order differs from lane order on purpose (row permutation).
        let mut joint_lane_caches: Vec<&mut Deepseek4Cache> = Vec::with_capacity(lanes.len());
        let mut remaining: Vec<Option<&mut Deepseek4Cache>> =
            joint_caches.iter_mut().map(Some).collect();
        for &lane in lanes {
            joint_lane_caches.push(remaining[lane].take().expect("lane listed once"));
        }
        let joint_state = model
            .forward_verifier_decode_joint_with_commit_gate(&tokens, &mut joint_lane_caches, || {
                Ok(())
            })
            .unwrap_or_else(|error| panic!("joint step {step} lanes {lanes:?}: {error:#}"));
        drop(joint_lane_caches);
        drop(remaining);
        let joint_logits = model
            .forward_logits(&joint_state)
            .unwrap_or_else(|error| panic!("joint step {step} head: {error:#}"));
        assert_eq!(joint_logits.shape(), [lanes.len(), vocab]);
        let joint_state = read_f32(&joint_state, "joint state");
        let joint_logits = read_f32(&joint_logits, "joint logits");
        for (row, &lane) in lanes.iter().enumerate() {
            assert_exact_f32(
                &format!("long step {step} lane {lane} state"),
                &serial_states[row],
                &joint_state[row * state_row..(row + 1) * state_row],
            );
            assert_exact_f32(
                &format!("long step {step} lane {lane} logits"),
                &serial_logits[row],
                &joint_logits[row * vocab..(row + 1) * vocab],
            );
        }
        for lane in 0..lanes_total {
            assert_cache_exact(
                &format!("long step {step} lane {lane} cache"),
                &serial_caches[lane],
                &joint_caches[lane],
            );
        }
        eprintln!("long step {step} lanes {lanes:?}: joint == serial");
    }
}
