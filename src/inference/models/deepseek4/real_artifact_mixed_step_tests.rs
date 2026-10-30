//! ADR-063 S1: official-artifact proof that one cooperative prefill cohort
//! and simultaneously decoding nonmembers stay isolated inside one mixed
//! scheduler step, under success, late failure, poison and cancellation.
//!
//! Each scenario issues the transactions a mixed step would issue (joint
//! decode for the decoders, the supervised cooperative cohort transaction,
//! snapshot/restore for a cancelled member) on one cache set, and
//! independent serial execution (`forward_verifier_one`,
//! `forward_verifier_prefill`) on a twin set. Decoder logits are read only
//! after the same step's cohort transaction finished, as the sampler reads
//! live logits. Every comparison is bitwise.
//!
//! Run on Apple Silicon with the converted checkpoint:
//! `HF2Q_DEEPSEEK4_GGUF=/path/to/deepseek4.gguf cargo test --release --lib \
//!   official_artifact_mixed_step_isolates_cohort_from_decoders -- --ignored --nocapture`

use mlx_native::MlxBuffer;

use super::cache::{Deepseek4Cache, Deepseek4CacheSnapshot};
use super::cooperative_fault;
use super::joint_decode::MAX_JOINT_DECODE_ROWS;
use super::model::Deepseek4Model;
use super::real_artifact_decode_cohort_tests::{assert_cache_exact, assert_exact_f32, read_f32};
use super::real_artifact_tests::official_artifact;
use super::verifier_forward::MAX_COOPERATIVE_PREFILL_ROWS;

const LANES: usize = 5;
/// Slot order interleaves the cohort with the decoders.
const COHORT: [usize; 3] = [0, 2, 3];
const DECODERS: [usize; 2] = [1, 4];
/// Cohort prefixes reproduce the protected anchor's compressor remainders
/// (mod-128 = 20, mod-4 = 0); decoders complete ratio-4 and ratio-128
/// groups while the cohort is in flight.
const PREFIX_ROWS: [usize; LANES] = [148, 125, 148, 148, 254];
/// Production three-lane cooperative width: 2048 / 3 rounded down to whole
/// 128-token windows.
const CHUNK_ROWS: usize = 640;
/// Chunk 0 runs decode-then-cohort; chunk 1 runs cohort-then-decode.
const CHUNK_STEPS: [usize; 2] = [1, 3];
const STEPS: usize = 8;
const CAPACITY: usize = 2_048;
/// Fail after a ratio-4 (indexer) and a ratio-128 layer wrote member state.
const FAULT_AFTER_LAYER: usize = 3;
const CANCELLED: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fault {
    /// Every submission completes and the commit gate accepts.
    Success,
    /// Every submission completes, then the supervisor gate rejects.
    LateFailure,
    /// Execution fails after `FAULT_AFTER_LAYER` wrote every member cache.
    Poison,
    /// The cohort commits, then one member's client cancels and the member
    /// rolls back to its prompt-boundary anchor and later replays serially.
    Cancel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role {
    /// Holds the greedy token sampled from its live logits.
    Decoding(u32),
    Prefilling,
    Poisoned,
    RolledBack,
}

fn token_batch(lane: usize, phase: usize, rows: usize) -> Vec<u32> {
    (0..rows)
        .map(|row| ((row * 97 + lane * 193 + phase * 389 + 11) % 120_000) as u32)
        .collect()
}

fn first_argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate() {
        if value > logits[best] {
            best = index;
        }
    }
    best as u32
}

fn selected<'a>(caches: &'a mut [Deepseek4Cache], lanes: &[usize]) -> Vec<&'a mut Deepseek4Cache> {
    caches
        .iter_mut()
        .enumerate()
        .filter(|(lane, _)| lanes.contains(lane))
        .map(|(_, cache)| cache)
        .collect()
}

fn head(model: &mut Deepseek4Model, state: &MlxBuffer, label: &str) -> Vec<f32> {
    let last = model
        .last_token_state(state)
        .unwrap_or_else(|error| panic!("{label} last row: {error:#}"));
    let logits = model
        .forward_logits(&last)
        .unwrap_or_else(|error| panic!("{label} head: {error:#}"));
    read_f32(&logits, label)
}

fn install_prefix(model: &mut Deepseek4Model, lane: usize) -> (Deepseek4Cache, Vec<f32>) {
    let mut cache = model.allocate_cache(CAPACITY).expect("allocate cache");
    let prefix = token_batch(lane, 0, PREFIX_ROWS[lane]);
    let state = model
        .forward_verifier_prompt(&prefix, &mut cache)
        .unwrap_or_else(|error| panic!("lane {lane} prefix: {error:#}"));
    assert_eq!(cache.position(), PREFIX_ROWS[lane]);
    let logits = head(model, &state, "prefix logits");
    (cache, logits)
}

fn serial_decode(
    model: &mut Deepseek4Model,
    cache: &mut Deepseek4Cache,
    token: u32,
    label: &str,
) -> Vec<f32> {
    let state = model
        .forward_verifier_one(token, cache)
        .unwrap_or_else(|error| panic!("{label} serial decode: {error:#}"));
    let logits = model
        .forward_logits(&state)
        .unwrap_or_else(|error| panic!("{label} serial head: {error:#}"));
    read_f32(&logits, label)
}

fn serial_chunk(
    model: &mut Deepseek4Model,
    cache: &mut Deepseek4Cache,
    lane: usize,
    chunk: usize,
    label: &str,
) -> Vec<f32> {
    let tokens = token_batch(lane, chunk + 1, CHUNK_ROWS);
    let state = model
        .forward_verifier_prefill(&tokens, cache)
        .unwrap_or_else(|error| panic!("{label} serial chunk {chunk}: {error:#}"));
    head(model, &state, label)
}

/// Joint decode as the serving joint path issues it. The logits stay
/// unread until the rest of the step has run.
fn joint_decode(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    decoders: &[(usize, u32)],
    label: &str,
) -> MlxBuffer {
    let lanes = decoders.iter().map(|&(lane, _)| lane).collect::<Vec<_>>();
    let tokens = decoders.iter().map(|&(_, token)| token).collect::<Vec<_>>();
    let state = model
        .forward_verifier_decode_joint_with_commit_gate(
            &tokens,
            &mut selected(caches, &lanes),
            || Ok(()),
        )
        .unwrap_or_else(|error| panic!("{label} joint decode: {error:#}"));
    model
        .forward_logits(&state)
        .unwrap_or_else(|error| panic!("{label} joint head: {error:#}"))
}

/// One cooperative transaction with the scenario's fault injected.
fn cohort_chunk(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    lanes: &[usize],
    chunk: usize,
    fault: Fault,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let batches = lanes
        .iter()
        .map(|&lane| token_batch(lane, chunk + 1, CHUNK_ROWS))
        .collect::<Vec<_>>();
    let batch_refs = batches.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let armed =
        (fault == Fault::Poison).then(|| cooperative_fault::arm_after_layer(FAULT_AFTER_LAYER));
    let result = model.forward_verifier_prefill_cohort_with_commit_gate(
        &batch_refs,
        &mut selected(caches, lanes),
        || {
            anyhow::ensure!(
                fault != Fault::LateFailure,
                "injected supervisor rejection after every cohort submission"
            );
            Ok(())
        },
    );
    if let Some(armed) = armed.as_ref() {
        assert!(armed.fired(), "injected cohort fault never fired");
    }
    let states = result?;
    Ok(states
        .iter()
        .map(|state| head(model, state, "cohort logits"))
        .collect())
}

fn run_scenario(model: &mut Deepseek4Model, fault: Fault) {
    let label = format!("{fault:?}");
    let vocab = model.cfg.vocab_size as usize;
    let mut serial = Vec::with_capacity(LANES);
    let mut mixed = Vec::with_capacity(LANES);
    let mut roles = [Role::Prefilling; LANES];
    let mut poisoned_at = [None; LANES];
    for lane in 0..LANES {
        let (serial_cache, serial_logits) = install_prefix(model, lane);
        let (mixed_cache, mixed_logits) = install_prefix(model, lane);
        let lane_label = format!("{label} lane {lane} prefix");
        assert_exact_f32(&lane_label, &serial_logits, &mixed_logits);
        assert_cache_exact(&lane_label, &serial_cache, &mixed_cache);
        if DECODERS.contains(&lane) {
            roles[lane] = Role::Decoding(first_argmax(&serial_logits));
        }
        serial.push(serial_cache);
        mixed.push(mixed_cache);
    }

    let mut anchor: Option<Deepseek4CacheSnapshot> = None;
    for step in 0..STEPS {
        let step_label = format!("{label} step {step}");
        if fault == Fault::Cancel && step == CHUNK_STEPS[0] + 1 {
            // The member's reply closed after its cohort commit: serving
            // restores the prompt-boundary anchor. Its serial twin never ran
            // the cancelled chunk.
            let anchor = anchor.take().expect("cancel anchor");
            mixed[CANCELLED]
                .restore(&anchor)
                .expect("restore cancelled member");
            assert_cache_exact(
                &format!("{step_label} cancelled member rollback"),
                &serial[CANCELLED],
                &mixed[CANCELLED],
            );
            roles[CANCELLED] = Role::RolledBack;
        }
        let decoders = (0..LANES)
            .filter_map(|lane| match roles[lane] {
                Role::Decoding(token) => Some((lane, token)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let chunk = CHUNK_STEPS
            .iter()
            .position(|&chunk_step| chunk_step == step);
        let prefilling = COHORT
            .iter()
            .copied()
            .filter(|&lane| roles[lane] == Role::Prefilling)
            .collect::<Vec<_>>();
        let cohort = chunk
            .filter(|_| prefilling.len() >= 2)
            .map(|chunk| (chunk, prefilling));
        let chunk_fault = match &cohort {
            Some((0, _)) => fault,
            _ => Fault::Success,
        };
        let commits = matches!(chunk_fault, Fault::Success | Fault::Cancel);

        // Independent serial execution of everything that should publish.
        let serial_decode_logits = decoders
            .iter()
            .map(|&(lane, token)| {
                let lane_label = format!("{step_label} decoder {lane}");
                serial_decode(model, &mut serial[lane], token, &lane_label)
            })
            .collect::<Vec<_>>();
        let mut serial_cohort_logits = Vec::new();
        if let Some((chunk, lanes)) = cohort.as_ref().filter(|_| commits) {
            for &lane in lanes {
                if fault == Fault::Cancel && *chunk == 0 && lane == CANCELLED {
                    continue;
                }
                let lane_label = format!("{step_label} serial member {lane}");
                let logits = serial_chunk(model, &mut serial[lane], lane, *chunk, &lane_label);
                serial_cohort_logits.push((lane, logits));
            }
        }

        // The mixed step: both transactions, in this step's order.
        let decode_first = chunk != Some(1);
        if fault == Fault::Cancel && chunk == Some(0) {
            anchor = Some(mixed[CANCELLED].snapshot().expect("snapshot anchor"));
        }
        let positions_before = mixed
            .iter()
            .map(Deepseek4Cache::position)
            .collect::<Vec<_>>();
        let mut joint_logits = None;
        if decode_first {
            joint_logits = Some(joint_decode(model, &mut mixed, &decoders, &step_label));
        }
        let cohort_result = cohort
            .as_ref()
            .map(|(chunk, lanes)| cohort_chunk(model, &mut mixed, lanes, *chunk, chunk_fault));
        if fault == Fault::Cancel && chunk == Some(1) {
            // Replay over the rolled-back cache: stale rows the cancelled
            // cooperative chunk left beyond the cursor must not leak.
            let lane = CANCELLED;
            for replay in 0..CHUNK_STEPS.len() {
                let replay_label = format!("{step_label} cancelled member replay {replay}");
                let expected = serial_chunk(model, &mut serial[lane], lane, replay, &replay_label);
                let actual = serial_chunk(model, &mut mixed[lane], lane, replay, &replay_label);
                assert_exact_f32(&replay_label, &expected, &actual);
                assert_cache_exact(&replay_label, &serial[lane], &mixed[lane]);
                roles[lane] = Role::Decoding(first_argmax(&expected));
            }
        }
        if !decode_first {
            joint_logits = Some(joint_decode(model, &mut mixed, &decoders, &step_label));
        }

        // Live decoder logits are read only now, after the cohort.
        let joint_logits = read_f32(&joint_logits.expect("decoders run every step"), &step_label);
        assert_eq!(joint_logits.len(), decoders.len() * vocab);
        let decoder_rows = decoders.iter().zip(&serial_decode_logits);
        for (row, (&(lane, _), expected)) in decoder_rows.enumerate() {
            let actual = &joint_logits[row * vocab..(row + 1) * vocab];
            assert_exact_f32(
                &format!("{step_label} decoder {lane} logits"),
                expected,
                actual,
            );
            roles[lane] = Role::Decoding(first_argmax(expected));
        }

        if let (Some((chunk, lanes)), Some(result)) = (cohort, cohort_result) {
            match result {
                Ok(logits) if commits => {
                    for (&lane, actual) in lanes.iter().zip(&logits) {
                        let member_label = format!("{step_label} member {lane}");
                        let expected = serial_cohort_logits
                            .iter()
                            .find(|(serial_lane, _)| *serial_lane == lane);
                        match expected {
                            Some((_, expected)) => {
                                assert_exact_f32(&member_label, expected, actual);
                                assert_cache_exact(&member_label, &serial[lane], &mixed[lane]);
                            }
                            None => assert!(fault == Fault::Cancel && lane == CANCELLED),
                        }
                        if chunk + 1 == CHUNK_STEPS.len() {
                            roles[lane] = Role::Decoding(first_argmax(actual));
                        }
                    }
                }
                Err(error) if !commits => {
                    let expected = if fault == Fault::LateFailure {
                        "rejected before cache publication"
                    } else {
                        "partially executed; caches poisoned"
                    };
                    assert!(
                        format!("{error:#}").contains(expected),
                        "{step_label} unexpected cohort error: {error:#}"
                    );
                    for &lane in &lanes {
                        assert!(mixed[lane].is_poisoned(), "{step_label} member {lane}");
                        roles[lane] = Role::Poisoned;
                        poisoned_at[lane] = Some(positions_before[lane]);
                    }
                }
                Ok(_) => panic!("{step_label} faulted cohort published"),
                Err(error) => panic!("{step_label} cohort failed: {error:#}"),
            }
        }

        for lane in 0..LANES {
            let lane_label = format!("{step_label} lane {lane}");
            match roles[lane] {
                Role::Poisoned => {
                    assert!(mixed[lane].is_poisoned(), "{lane_label} lost poison");
                    assert_eq!(
                        Some(mixed[lane].position()),
                        poisoned_at[lane],
                        "{lane_label}"
                    );
                    assert!(mixed[lane].plan_next_step().is_err());
                }
                _ if fault == Fault::Cancel && lane == CANCELLED && step == CHUNK_STEPS[0] => {}
                _ => {
                    assert!(!mixed[lane].is_poisoned(), "{lane_label} poisoned");
                    assert_cache_exact(&lane_label, &serial[lane], &mixed[lane]);
                }
            }
        }
    }

    for lane in DECODERS {
        assert_eq!(mixed[lane].position(), PREFIX_ROWS[lane] + STEPS);
    }
    // A poisoned member leaves no hidden state behind its reset.
    for lane in COHORT {
        if roles[lane] != Role::Poisoned {
            continue;
        }
        mixed[lane].reset().expect("reset poisoned member");
        let prefix = token_batch(lane, 0, PREFIX_ROWS[lane]);
        model
            .forward_verifier_prompt(&prefix, &mut mixed[lane])
            .expect("reinstall poisoned member prefix");
        let replay_label = format!("{label} member {lane} replay after reset");
        let expected = serial_chunk(model, &mut serial[lane], lane, 0, &replay_label);
        let actual = serial_chunk(model, &mut mixed[lane], lane, 0, &replay_label);
        assert_exact_f32(&replay_label, &expected, &actual);
        assert_cache_exact(&replay_label, &serial[lane], &mixed[lane]);
    }
    eprintln!(
        "DeepSeek-V4 S1 mixed step: fault={fault:?} cohort={COHORT:?} decoders={DECODERS:?} chunk_rows={CHUNK_ROWS} steps={STEPS} exact_nonmember_state_logits_tokens=true"
    );
}

#[test]
#[ignore = "loads the official checkpoint and proves mixed cooperative prefill isolates decoders"]
fn official_artifact_mixed_step_isolates_cohort_from_decoders() {
    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    assert_eq!(model.cfg.sliding_window, 128);
    let faulted_ratios = &model.cfg.compress_ratios[..=FAULT_AFTER_LAYER];
    assert!(faulted_ratios.contains(&4) && faulted_ratios.contains(&128));
    assert_eq!(model.joint_decode_unavailable_reason(), None);
    assert!(LANES <= MAX_JOINT_DECODE_ROWS);
    assert_eq!(
        CHUNK_ROWS,
        MAX_COOPERATIVE_PREFILL_ROWS / COHORT.len() / 128 * 128
    );
    for fault in [
        Fault::Success,
        Fault::LateFailure,
        Fault::Poison,
        Fault::Cancel,
    ] {
        run_scenario(&mut model, fault);
    }
}
