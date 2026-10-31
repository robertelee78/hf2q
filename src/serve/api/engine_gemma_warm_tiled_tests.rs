//! ADR-063 Stage 2 tiled F16 warm attention: real-artifact determinism and
//! decision-envelope gates. Joint == per-slot on the tiled route is
//! `gemma_joint_warm_prefill_matches_per_slot_bitwise` (Stage 1 harness).
//!
//! Ignored; `HF2Q_GEMMA4_GGUF` names the official Gemma 4 26B-A4B GGUF. Run
//! each test alone with `--exact --ignored --nocapture --test-threads=1`.
//! * `gemma_warm_tiled_attention_is_deterministic`: on the tiled route, each
//!   gate lane's per-slot warm prefill and the four-lane joint cohort repeat
//!   bit for bit (first token, logits bits, KV bytes, cursors).
//! * `gemma_warm_tiled_attention_decision_envelope`: per lane, the tiled
//!   route against the vec-hybrid route (`HF2Q_GEMMA_WARM_TILED_ATTENTION=0`)
//!   from identical histories: max |logit difference| at the lane's last
//!   row, top-1 agreement and the top-1/top-2 margins, printed as Q1
//!   envelope evidence. It fails only on a top-1 disagreement outside a
//!   near tie, and when a suffix below the 32-row tiled threshold is not
//!   bitwise equal across the two routes (both must run vec-hybrid).
use super::gemma_warm_batch_tests::{
    assert_lane_equal, load_gate_model, reserve_for, LaneImage, WarmFixture, HISTORIES, MAX_DECODE,
    SUFFIXES,
};
use super::*;
use crate::inference::models::gemma4::GemmaLiveKernel;

const REPEATS: usize = 5;
/// A vec-hybrid top-1/top-2 margin below this many logits is a near tie: a
/// top-1 flip there is recorded, not failed.
const NEAR_TIE_MARGIN: f32 = 0.5;

#[test]
#[ignore = "official Gemma 4 GGUF: tiled warm attention repeats bit for bit"]
fn gemma_warm_tiled_attention_is_deterministic() {
    let mut loaded = load_gate_model();
    let LoadedModel::Gemma(g) = &mut loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    guard.model.weights.warm_live_kernel = GemmaLiveKernel::TiledF16;
    let fixture = WarmFixture::build(&mut guard);
    let prompts = &fixture.prompts;

    let mut first_runs = Vec::with_capacity(4);
    for slot in 0..4 {
        let first = fixture.per_slot(
            &mut guard,
            slot,
            &prompts[slot],
            HISTORIES[slot],
            MAX_DECODE,
        );
        for repeat in 1..REPEATS {
            let again = fixture.per_slot(
                &mut guard,
                slot,
                &prompts[slot],
                HISTORIES[slot],
                MAX_DECODE,
            );
            assert_lane_equal(
                &format!("tiled per-slot slot {slot} repeat {repeat}"),
                &again,
                &first,
            );
        }
        first_runs.push(first);
    }

    let lanes: Vec<_> = (0..4).map(|slot| fixture.lane(slot)).collect();
    let reserve = reserve_for(&lanes);
    let first_joint = fixture.joint(&mut guard, &lanes, reserve).unwrap();
    for repeat in 1..REPEATS {
        let again = fixture.joint(&mut guard, &lanes, reserve).unwrap();
        for (slot, (image, first)) in again.iter().zip(&first_joint).enumerate() {
            assert_lane_equal(
                &format!("tiled joint slot {slot} repeat {repeat}"),
                image,
                first,
            );
        }
    }
    for (slot, (joint, per_slot)) in first_joint.iter().zip(&first_runs).enumerate() {
        assert_lane_equal(
            &format!("tiled joint == per-slot slot {slot}"),
            joint,
            per_slot,
        );
    }
    println!(
        "GEMMA_WARM_TILED_DETERMINISM_COMPLETE kernel=TiledF16 repeats={REPEATS} per_slot_lanes=4 joint_lanes=4 logits_bits=true kv_bytes=true cursors=true histories={HISTORIES:?} suffixes={SUFFIXES:?}"
    );
}

/// Final logits of a lane image.
fn logits(image: &LaneImage) -> Vec<f32> {
    image
        .logits
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect()
}

/// `(top-1 index, top-1 minus top-2)` of a logits row.
fn top1_margin(row: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (index, &value) in row.iter().enumerate() {
        if value > best.1 {
            second = best.1;
            best = (index, value);
        } else if value > second {
            second = value;
        }
    }
    (best.0, best.1 - second)
}

#[test]
#[ignore = "official Gemma 4 GGUF: tiled vs vec-hybrid warm attention decision envelope (Q1 evidence)"]
fn gemma_warm_tiled_attention_decision_envelope() {
    let mut loaded = load_gate_model();
    let LoadedModel::Gemma(g) = &mut loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    // Histories come from cold prefill, which never runs live attention, so
    // both routes resume from identical bytes.
    let fixture = WarmFixture::build(&mut guard);
    let prompts = &fixture.prompts;
    let run =
        |guard: &mut Gemma4KvGuard<'_>, kernel: GemmaLiveKernel, slot: usize, prompt: &[u32]| {
            guard.model.weights.warm_live_kernel = kernel;
            fixture.per_slot(guard, slot, prompt, HISTORIES[slot], MAX_DECODE)
        };

    let mut worst = 0.0f32;
    let mut agree = 0usize;
    let mut near_ties = 0usize;
    let mut min_margin = f32::INFINITY;
    let mut failures = Vec::new();
    for slot in 0..4 {
        let vec_image = run(&mut guard, GemmaLiveKernel::VecHybrid, slot, &prompts[slot]);
        let tiled_image = run(&mut guard, GemmaLiveKernel::TiledF16, slot, &prompts[slot]);
        let (vec_logits, tiled_logits) = (logits(&vec_image), logits(&tiled_image));
        let max_abs = vec_logits
            .iter()
            .zip(&tiled_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let (vec_top, vec_margin) = top1_margin(&vec_logits);
        let (tiled_top, tiled_margin) = top1_margin(&tiled_logits);
        let near_tie = vec_margin < NEAR_TIE_MARGIN;
        let top1_agree = vec_top == tiled_top;
        worst = worst.max(max_abs);
        min_margin = min_margin.min(vec_margin);
        agree += usize::from(top1_agree);
        near_ties += usize::from(near_tie);
        println!(
            "GEMMA_WARM_TILED_ENVELOPE_LANE slot={slot} history={} suffix={} max_abs_logit_diff={max_abs:.6} top1_vec={vec_top} top1_tiled={tiled_top} top1_agree={top1_agree} margin_vec={vec_margin:.6} margin_tiled={tiled_margin:.6} near_tie={near_tie} first_token_vec={} first_token_tiled={}",
            HISTORIES[slot], SUFFIXES[slot], vec_image.first_token, tiled_image.first_token
        );
        if !top1_agree && !near_tie {
            failures.push(slot);
        }
    }

    // Below 32 suffix rows the warm route keeps vec-hybrid under either
    // switch, so the two arms must be the same graph bit for bit.
    let tiny: Vec<u32> = prompts[0][..HISTORIES[0] + 31].to_vec();
    let tiny_vec = run(&mut guard, GemmaLiveKernel::VecHybrid, 0, &tiny);
    let tiny_tiled = run(&mut guard, GemmaLiveKernel::TiledF16, 0, &tiny);
    assert_lane_equal(
        "31-row suffix keeps vec-hybrid on both switches",
        &tiny_tiled,
        &tiny_vec,
    );

    println!(
        "GEMMA_WARM_TILED_ENVELOPE lanes=4 top1_agree={agree}/4 near_ties={near_ties} near_tie_margin={NEAR_TIE_MARGIN} max_abs_logit_diff={worst:.6} min_margin_vec={min_margin:.6} tiny_suffix_equal=true histories={HISTORIES:?} suffixes={SUFFIXES:?}"
    );
    assert!(
        failures.is_empty(),
        "tiled warm attention flipped top-1 outside a near tie on slots {failures:?}"
    );
}
