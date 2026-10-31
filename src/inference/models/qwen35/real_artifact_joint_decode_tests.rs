//! Official-artifact hardware gates for Qwen joint B-row decode.
//!
//! Each gate prefills four slots of two identical `HybridKvCache`s with
//! ragged prompts, then advances one cache with independent serial one-token
//! forwards and the other with joint steps over a cohort schedule that
//! covers full and partial cohorts, a lone row, parity flips on every
//! DeltaNet layer, a row that stops (EOS/cancel) while the others continue,
//! and a slot that is reset and re-prefilled (a new conversation) before it
//! rejoins. Refused steps must leave both caches untouched.
//!
//! Run (one artifact per process; mlx-native snapshots its routing policy):
//!
//! ```text
//! HF2Q_QWEN36_GGUF=/path/qwen36.gguf cargo test --release --bin hf2q \
//!   qwen36_joint_decode_matches_serial_decode_bitwise -- --ignored --exact
//! HF2Q_QWEN38_GGUF=/path/qwen38.gguf cargo test --release --bin hf2q \
//!   qwen38_joint_decode_mvn_matches_serial_decode_bitwise -- --ignored --exact
//! HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 HF2Q_QWEN38_GGUF=/path/qwen38.gguf \
//!   cargo test --release --bin hf2q \
//!   qwen38_joint_decode_padded_mv_ext_keeps_decision_parity -- --ignored --exact
//! ```
//!
//! The padded test is a Q1 diagnostic (synthetic prompts, top-1 only) on the
//! dense FFN route `HF2Q_QWEN_JOINT_FFN_SHARED` selects (default: shared
//! weights). Cohort invariance lives in
//! `real_artifact_joint_decode_invariance_tests`, the real-prompt Q1 Class B
//! trace in `real_artifact_joint_decode_q1_tests`.

use mlx_native::{DType, MlxBuffer};

use super::joint_decode_routes::{
    joint_ffn_shared_from_process, select_dense_ffn_route, JointDenseFfnRoute, JointRouteMode,
};
use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::spec_decode::positions_for_range;
use super::{Qwen35JointOutput, Qwen35JointRow};
use crate::serve::multi_seq_kv::SlotId;

pub(super) const SLOTS: u32 = 4;
const CAPACITY: u32 = 160;
const BASE: [u32; 5] = [9707, 374, 279, 15, 1_782];

pub(super) fn load(env: &str) -> Qwen35Model {
    use crate::serve::header::LoadProgress;
    use mlx_native::gguf::GgufFile;

    let path = std::env::var_os(env)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| panic!("ignored joint decode gate requires {env}"));
    assert!(path.is_file(), "artifact is absent at {}", path.display());
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let mut progress = LoadProgress::new(false, 1, 0);
    let model = Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load Qwen model");
    model
        .ensure_gpu_cache_primed()
        .expect("prime GPU cache before allocating caches");
    model
}

fn new_cache(model: &Qwen35Model) -> HybridKvCache {
    model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&model.cfg, device, CAPACITY, SLOTS, true)
        })
        .expect("allocate four-slot TQ cache")
}

/// Ragged prompt for `slot` (≥ 16 tokens: the TQ prefill floor).
pub(super) fn prompt(slot: u32, generation: u32) -> Vec<u32> {
    let len = 17 + 5 * slot as usize + 3 * generation as usize;
    (0..len)
        .map(|index| BASE[(index + slot as usize + generation as usize) % BASE.len()])
        .collect()
}

pub(super) fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate() {
        if value > logits[best] {
            best = index;
        }
    }
    best as u32
}

pub(super) fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

pub(super) fn buffer_bytes(buffer: &MlxBuffer) -> Vec<u8> {
    match buffer.dtype() {
        DType::U8 => buffer
            .as_slice::<u8>()
            .expect("read U8 cache bytes")
            .to_vec(),
        DType::F32 => buffer
            .as_slice::<f32>()
            .expect("read F32 cache bytes")
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect(),
        other => panic!("unexpected cache dtype {other:?}"),
    }
}

pub(super) fn hidden_bits(buffer: &MlxBuffer, hidden: usize) -> Vec<u32> {
    bits(&buffer.as_slice::<f32>().expect("read hidden")[..hidden])
}

/// Cursors, parities, KV codes/scales and conv/recurrent state, all slots.
fn assert_caches_equal(serial: &HybridKvCache, joint: &HybridKvCache, label: &str) {
    for slot in 0..SLOTS {
        assert_eq!(
            serial.sequence_len_for_slot(SlotId(slot)).ok(),
            joint.sequence_len_for_slot(SlotId(slot)).ok(),
            "{label}: slot {slot} cursor"
        );
    }
    for (layer, (a, b)) in serial.full_attn.iter().zip(&joint.full_attn).enumerate() {
        assert_eq!(
            a.current_len, b.current_len,
            "{label}: full_attn[{layer}] cursors"
        );
        match (&a.tq, &b.tq) {
            (Some(a), Some(b)) => {
                for (name, x, y) in [
                    ("k_packed", &a.k_packed, &b.k_packed),
                    ("k_norms", &a.k_norms, &b.k_norms),
                    ("v_packed", &a.v_packed, &b.v_packed),
                    ("v_norms", &a.v_norms, &b.v_norms),
                ] {
                    assert!(
                        buffer_bytes(x) == buffer_bytes(y),
                        "{label}: full_attn[{layer}] TQ {name} bytes differ"
                    );
                }
            }
            (None, None) => {}
            _ => panic!("{label}: full_attn[{layer}] TQ presence differs"),
        }
        for (name, x, y) in [("k", &a.k, &b.k), ("v", &a.v, &b.v)] {
            if let (Some(x), Some(y)) = (x, y) {
                assert!(
                    buffer_bytes(x) == buffer_bytes(y),
                    "{label}: full_attn[{layer}] F32 {name} bytes differ"
                );
            }
        }
    }
    for (layer, (a, b)) in serial
        .linear_attn
        .iter()
        .zip(&joint.linear_attn)
        .enumerate()
    {
        assert_eq!(
            a.pp_flipped, b.pp_flipped,
            "{label}: linear_attn[{layer}] parity"
        );
        for (name, x, y) in [
            ("conv_state", &a.conv_state, &b.conv_state),
            (
                "conv_state_scratch",
                &a.conv_state_scratch,
                &b.conv_state_scratch,
            ),
            ("recurrent", &a.recurrent, &b.recurrent),
            (
                "recurrent_scratch",
                &a.recurrent_scratch,
                &b.recurrent_scratch,
            ),
        ] {
            assert!(
                buffer_bytes(x) == buffer_bytes(y),
                "{label}: linear_attn[{layer}] {name} bytes differ"
            );
        }
    }
}

struct Lane {
    next_token: u32,
    position: i32,
}

/// Prefill `slot` with `prompt` in both caches and return its lane.
fn prefill_slot(
    model: &Qwen35Model,
    serial: &mut HybridKvCache,
    joint: &mut HybridKvCache,
    slot: u32,
    prompt: &[u32],
) -> Lane {
    let positions = positions_for_range(0, prompt.len());
    let a = model
        .forward_gpu_last_logits(prompt, &positions, serial, SlotId(slot))
        .expect("serial prefill");
    let b = model
        .forward_gpu_last_logits(prompt, &positions, joint, SlotId(slot))
        .expect("joint-cache prefill");
    assert_eq!(
        bits(&a),
        bits(&b),
        "slot {slot} prefill must be deterministic"
    );
    Lane {
        next_token: argmax(&a),
        position: prompt.len() as i32,
    }
}

fn output_kind(slot: u32, step: usize) -> Qwen35JointOutput {
    match (slot as usize + step) % 3 {
        0 => Qwen35JointOutput::GreedyToken,
        1 => Qwen35JointOutput::Logits,
        _ => Qwen35JointOutput::LogitsAndHidden,
    }
}

/// Cohorts per step: full, partial, lone, ragged re-mixes. Slot 1 stops
/// after step 3 (EOS/cancel) and returns as a new conversation at step 6.
const SCHEDULE: [&[u32]; 9] = [
    &[0, 1, 2, 3],
    &[0, 1, 2, 3],
    &[1, 3],
    &[0, 1, 2],
    &[0, 2, 3],
    &[2],
    &[0, 1, 2, 3],
    &[1, 3],
    &[0, 1, 2, 3],
];
const REJOIN_STEP: usize = 6;

fn assert_refusal_writes_nothing(
    model: &Qwen35Model,
    serial: &HybridKvCache,
    joint: &mut HybridKvCache,
    lanes: &[Lane],
    mode: JointRouteMode,
    dense_ffn: JointDenseFfnRoute,
) {
    let row = |slot: u32| Qwen35JointRow {
        slot_id: SlotId(slot),
        token: lanes[slot as usize].next_token,
        position: lanes[slot as usize].position,
        output: Qwen35JointOutput::Logits,
    };
    let mut refused = vec![
        (vec![row(0), row(0)], dense_ffn),
        (
            vec![
                row(1),
                Qwen35JointRow {
                    slot_id: SlotId(SLOTS),
                    ..row(2)
                },
            ],
            dense_ffn,
        ),
    ];
    if mode == JointRouteMode::Exact {
        // A valid cohort on the non-bitwise shared FFN route.
        refused.push((vec![row(0), row(1)], JointDenseFfnRoute::SharedWeights));
    }
    for (rows, dense_ffn) in refused {
        assert!(
            model
                .forward_gpu_decode_joint(&rows, joint, mode, dense_ffn)
                .is_err(),
            "invalid joint step must be refused"
        );
        assert_caches_equal(serial, joint, "after refused step");
    }
}

/// Bitwise gate: every joint row equals its serial one-token forward, and
/// both caches stay byte-identical across the whole schedule.
fn run_bitwise_gate(env: &str) {
    assert_ne!(
        std::env::var("HF2Q_DECODE_MV_EXT").as_deref(),
        Ok("1"),
        "the bitwise arm needs the exact routing policy (unset HF2Q_DECODE_MV_EXT)"
    );
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = load(env);
    assert_eq!(
        model.joint_decode_mode(),
        Ok(JointRouteMode::Exact),
        "official artifact must be admitted to exact joint decode"
    );
    let hidden = model.cfg.hidden_size as usize;
    let mut serial = new_cache(&model);
    let mut joint = new_cache(&model);
    let mut lanes = (0..SLOTS)
        .map(|slot| prefill_slot(&model, &mut serial, &mut joint, slot, &prompt(slot, 0)))
        .collect::<Vec<_>>();
    assert_caches_equal(&serial, &joint, "after prefill");
    assert_refusal_writes_nothing(
        &model,
        &serial,
        &mut joint,
        &lanes,
        JointRouteMode::Exact,
        JointDenseFfnRoute::SerialBuilder,
    );

    for (step, cohort) in SCHEDULE.iter().enumerate() {
        if step == REJOIN_STEP {
            for cache in [&mut serial, &mut joint] {
                cache.reset_for_slot(SlotId(1)).expect("reset slot 1");
            }
            lanes[1] = prefill_slot(&model, &mut serial, &mut joint, 1, &prompt(1, 1));
        }
        let rows = cohort
            .iter()
            .map(|&slot| Qwen35JointRow {
                slot_id: SlotId(slot),
                token: lanes[slot as usize].next_token,
                position: lanes[slot as usize].position,
                output: output_kind(slot, step),
            })
            .collect::<Vec<_>>();
        let outputs = model
            .forward_gpu_decode_joint(
                &rows,
                &mut joint,
                JointRouteMode::Exact,
                JointDenseFfnRoute::SerialBuilder,
            )
            .unwrap_or_else(|error| panic!("joint step {step}: {error:#}"));
        for (row, output) in rows.iter().zip(&outputs) {
            let slot = row.slot_id.0;
            let positions = [row.position; 4];
            let next = match row.output {
                Qwen35JointOutput::GreedyToken => {
                    let token = model
                        .forward_gpu_greedy(&[row.token], &positions, &mut serial, row.slot_id)
                        .expect("serial greedy step");
                    assert_eq!(
                        output.greedy_token,
                        Some(token),
                        "step {step} slot {slot} greedy token"
                    );
                    token
                }
                Qwen35JointOutput::Logits => {
                    let logits = model
                        .forward_gpu_last_logits(&[row.token], &positions, &mut serial, row.slot_id)
                        .expect("serial logits step");
                    let joint_logits = output.logits.as_deref().expect("joint logits");
                    assert!(
                        bits(&logits) == bits(joint_logits),
                        "step {step} slot {slot} logits differ"
                    );
                    argmax(&logits)
                }
                Qwen35JointOutput::LogitsAndHidden => {
                    let (logits, nextn) = model
                        .forward_gpu_with_nextn_hidden(
                            &[row.token],
                            &positions,
                            &mut serial,
                            row.slot_id,
                        )
                        .expect("serial nextn step");
                    let joint_logits = output.logits.as_deref().expect("joint logits");
                    assert!(
                        bits(&logits) == bits(joint_logits),
                        "step {step} slot {slot} logits differ"
                    );
                    assert_eq!(
                        hidden_bits(&nextn, hidden),
                        hidden_bits(output.nextn_hidden.as_ref().expect("joint hidden"), hidden),
                        "step {step} slot {slot} nextn hidden differs"
                    );
                    argmax(&logits)
                }
            };
            let lane = &mut lanes[slot as usize];
            lane.next_token = next;
            lane.position += 1;
        }
        assert_caches_equal(&serial, &joint, &format!("after joint step {step}"));
    }
}

#[test]
#[ignore = "requires HF2Q_QWEN36_GGUF (official Qwen3.6 artifact) and Apple Metal"]
fn qwen36_joint_decode_matches_serial_decode_bitwise() {
    run_bitwise_gate("HF2Q_QWEN36_GGUF");
}

#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (official Qwen3.8 artifact) and Apple Metal"]
fn qwen38_joint_decode_mvn_matches_serial_decode_bitwise() {
    run_bitwise_gate("HF2Q_QWEN38_GGUF");
}

/// Decision diagnostic for the Qwen3.8 production route: joint rows use
/// `mul_mv_ext` at the fixed padded width 4, with the dense FFN route
/// `HF2Q_QWEN_JOINT_FFN_SHARED` selects. On these synthetic prompts,
/// teacher-forced top-1 decisions must equal independent sequential
/// forwards at every step, cursors and parities must match, and every slot
/// must keep making the same serial decisions for `FOLLOW_ON` further
/// tokens. Under Q1 this is diagnostic only (synthetic inputs, top-1); the
/// Class B verdict comes from `q1_decision_policy.py envelope` over the
/// real-prompt trace written by
/// `qwen38_joint_decode_q1_envelope_trace` (see the design note).
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF, HF2Q_DECODE_MVN=0, HF2Q_DECODE_MV_EXT=1 and Apple Metal"]
fn qwen38_joint_decode_padded_mv_ext_keeps_decision_parity() {
    const FOLLOW_ON: usize = 16;
    assert_eq!(
        std::env::var("HF2Q_DECODE_MVN").as_deref(),
        Ok("0"),
        "padded decision diagnostic requires externally-set HF2Q_DECODE_MVN=0"
    );
    assert_eq!(
        std::env::var("HF2Q_DECODE_MV_EXT").as_deref(),
        Ok("1"),
        "padded decision diagnostic requires externally-set HF2Q_DECODE_MV_EXT=1"
    );
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = load("HF2Q_QWEN38_GGUF");
    assert_eq!(
        model.joint_decode_mode(),
        Ok(JointRouteMode::PaddedMvExt),
        "Qwen3.8 under the production policy must select the padded mul_mv_ext mode"
    );
    let dense_ffn = select_dense_ffn_route(
        JointRouteMode::PaddedMvExt,
        joint_ffn_shared_from_process().expect("valid HF2Q_QWEN_JOINT_FFN_SHARED"),
    );
    eprintln!(
        "padded decision diagnostic: dense FFN {}",
        dense_ffn.label()
    );
    let mut serial = new_cache(&model);
    let mut joint = new_cache(&model);
    let mut lanes = (0..SLOTS)
        .map(|slot| prefill_slot(&model, &mut serial, &mut joint, slot, &prompt(slot, 0)))
        .collect::<Vec<_>>();
    assert_refusal_writes_nothing(
        &model,
        &serial,
        &mut joint,
        &lanes,
        JointRouteMode::PaddedMvExt,
        dense_ffn,
    );
    let cursors_and_parity = |serial: &HybridKvCache, joint: &HybridKvCache, label: &str| {
        for slot in 0..SLOTS {
            assert_eq!(
                serial.sequence_len_for_slot(SlotId(slot)).ok(),
                joint.sequence_len_for_slot(SlotId(slot)).ok(),
                "{label}: slot {slot} cursor"
            );
        }
        for (layer, (a, b)) in serial
            .linear_attn
            .iter()
            .zip(&joint.linear_attn)
            .enumerate()
        {
            assert_eq!(
                a.pp_flipped, b.pp_flipped,
                "{label}: linear_attn[{layer}] parity"
            );
        }
    };

    for (step, cohort) in SCHEDULE.iter().enumerate() {
        if step == REJOIN_STEP {
            for cache in [&mut serial, &mut joint] {
                cache.reset_for_slot(SlotId(1)).expect("reset slot 1");
            }
            let fresh = prompt(1, 1);
            let positions = positions_for_range(0, fresh.len());
            let a = model
                .forward_gpu_last_logits(&fresh, &positions, &mut serial, SlotId(1))
                .expect("serial re-prefill");
            let b = model
                .forward_gpu_last_logits(&fresh, &positions, &mut joint, SlotId(1))
                .expect("joint-cache re-prefill");
            assert_eq!(argmax(&a), argmax(&b), "re-prefill decision");
            lanes[1] = Lane {
                next_token: argmax(&a),
                position: fresh.len() as i32,
            };
        }
        let rows = cohort
            .iter()
            .map(|&slot| Qwen35JointRow {
                slot_id: SlotId(slot),
                token: lanes[slot as usize].next_token,
                position: lanes[slot as usize].position,
                output: Qwen35JointOutput::GreedyToken,
            })
            .collect::<Vec<_>>();
        let outputs = model
            .forward_gpu_decode_joint(&rows, &mut joint, JointRouteMode::PaddedMvExt, dense_ffn)
            .unwrap_or_else(|error| panic!("joint step {step}: {error:#}"));
        for (row, output) in rows.iter().zip(&outputs) {
            let serial_token = model
                .forward_gpu_greedy(&[row.token], &[row.position; 4], &mut serial, row.slot_id)
                .expect("serial greedy step");
            assert_eq!(
                output.greedy_token,
                Some(serial_token),
                "step {step} slot {} decision",
                row.slot_id.0
            );
            // Teacher forcing: both caches consume the serial decision.
            let lane = &mut lanes[row.slot_id.0 as usize];
            lane.next_token = serial_token;
            lane.position += 1;
        }
        cursors_and_parity(&serial, &joint, &format!("after joint step {step}"));
    }

    for step in 0..FOLLOW_ON {
        for slot in 0..SLOTS {
            let lane = &mut lanes[slot as usize];
            let positions = [lane.position; 4];
            let a = model
                .forward_gpu_greedy(&[lane.next_token], &positions, &mut serial, SlotId(slot))
                .expect("serial follow-on");
            let b = model
                .forward_gpu_greedy(&[lane.next_token], &positions, &mut joint, SlotId(slot))
                .expect("joint-cache follow-on");
            assert_eq!(a, b, "follow-on step {step} slot {slot} decision");
            lane.next_token = a;
            lane.position += 1;
        }
    }
    cursors_and_parity(&serial, &joint, "after follow-on decode");
}
