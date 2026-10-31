//! Official-artifact cohort-invariance gates for Qwen joint B-row decode
//! (Q1 §3.1 item 2, Class A).
//!
//! A joint row must produce the same bytes whichever tenants share its step
//! and wherever it sits in the packed rows. Each variant below prefills four
//! slots identically in its own cache, then advances the target slot for
//! several steps inside a different cohort (alone, pairs, a triple, and a
//! full cohort with the target packed last). The target's logits, post-norm
//! hidden, cursor, ping-pong parity, KV codes/scales and conv/recurrent
//! bytes must be bitwise equal across every variant: in exact mode, and in
//! the padded `mul_mv_ext` mode (packed width always 4) on both dense FFN
//! routes (shared weights, the default, and the serial builder).
//!
//! ```text
//! HF2Q_QWEN36_GGUF=/path/qwen36.gguf cargo test --release --bin hf2q \
//!   qwen36_joint_decode_rows_are_cohort_invariant_bitwise -- --ignored --exact
//! HF2Q_QWEN38_GGUF=/path/qwen38.gguf cargo test --release --bin hf2q \
//!   qwen38_joint_decode_mvn_rows_are_cohort_invariant_bitwise -- --ignored --exact
//! HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 HF2Q_QWEN38_GGUF=/path/qwen38.gguf \
//!   cargo test --release --bin hf2q \
//!   qwen38_joint_decode_padded_mv_ext_rows_are_cohort_invariant_bitwise -- --ignored --exact
//! ```

use mlx_native::MlxBuffer;

use super::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
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

pub(super) fn cache_with_capacity(model: &Qwen35Model, capacity: u32) -> HybridKvCache {
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

fn run_variant(
    model: &Qwen35Model,
    cohort: &[u32],
    mode: JointRouteMode,
    dense_ffn: JointDenseFfnRoute,
) -> TargetTrace {
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
            .forward_gpu_decode_joint(&rows, &mut cache, mode, dense_ffn)
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

fn run_cohort_invariance(env: &str, mode: JointRouteMode, dense_ffns: &[JointDenseFfnRoute]) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = load(env);
    assert_eq!(
        model.joint_decode_mode(),
        Ok(mode),
        "official artifact must be admitted in {} mode",
        mode.label()
    );
    for &dense_ffn in dense_ffns {
        let label = format!("{} mode, {} dense FFN", mode.label(), dense_ffn.label());
        let reference = run_variant(&model, VARIANTS[0], mode, dense_ffn);
        assert_eq!(reference.logits.len(), STEPS);
        for &cohort in &VARIANTS[1..] {
            let variant = run_variant(&model, cohort, mode, dense_ffn);
            for step in 0..STEPS {
                assert!(
                    variant.logits[step] == reference.logits[step],
                    "{label}: target logits differ at step {step} in cohort {cohort:?} vs alone"
                );
                assert!(
                    variant.hidden[step] == reference.hidden[step],
                    "{label}: target hidden differs at step {step} in cohort {cohort:?} vs alone"
                );
            }
            assert!(
                variant.state == reference.state,
                "{label}: target cache state differs in cohort {cohort:?} vs alone"
            );
        }
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
    run_cohort_invariance(
        "HF2Q_QWEN36_GGUF",
        JointRouteMode::Exact,
        &[JointDenseFfnRoute::SerialBuilder],
    );
}

#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (official Qwen3.8 artifact) and Apple Metal"]
fn qwen38_joint_decode_mvn_rows_are_cohort_invariant_bitwise() {
    assert_ne!(
        std::env::var("HF2Q_DECODE_MV_EXT").as_deref(),
        Ok("1"),
        "the exact arm needs HF2Q_DECODE_MV_EXT unset"
    );
    run_cohort_invariance(
        "HF2Q_QWEN38_GGUF",
        JointRouteMode::Exact,
        &[JointDenseFfnRoute::SerialBuilder],
    );
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
    run_cohort_invariance(
        "HF2Q_QWEN38_GGUF",
        JointRouteMode::PaddedMvExt,
        &[
            JointDenseFfnRoute::SharedWeights,
            JointDenseFfnRoute::SerialBuilder,
        ],
    );
}
