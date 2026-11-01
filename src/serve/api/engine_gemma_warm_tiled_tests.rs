//! ADR-063 Stage 2 tiled F16 warm attention: real-artifact determinism and
//! decision-envelope gates. Joint == per-slot on the tiled routes is
//! `gemma_joint_warm_prefill_matches_per_slot_bitwise{,_tiled_precise}`
//! (Stage 1 harness).
//!
//! Ignored; `HF2Q_GEMMA4_GGUF` names the official Gemma 4 26B-A4B GGUF. Run
//! each test alone with `--exact --ignored --nocapture --test-threads=1`.
//! * `gemma_warm_tiled_attention_is_deterministic`: on the tiled route, each
//!   gate lane's per-slot warm prefill and the four-lane joint cohort repeat
//!   bit for bit (first token, logits bits, KV bytes, cursors).
//! * `gemma_warm_tiled_attention_decision_envelope` (`TiledF16`, switch `1`)
//!   and `gemma_warm_tiled_precise_attention_decision_envelope`
//!   (`TiledF16Precise`, switch `precise`): per lane, the tiled arm against
//!   the vec-hybrid route (`HF2Q_GEMMA_WARM_TILED_ATTENTION=0`) from
//!   identical histories, at the lane's last row: max |Δ log-prob| over the
//!   vocabulary and over the vec-hybrid top-2, top-1 agreement and the
//!   top-1/top-2 margins in nats. A top-1 flip fails unless the vec-hybrid
//!   margin is a policy near tie (< [`NEAR_TIE_NAT`]). A suffix below the
//!   32-row tiled threshold must be bitwise equal across the two routes
//!   (both run vec-hybrid).
//! * `gemma_warm_tiled_attention_layer_localization` and
//!   `gemma_warm_tiled_precise_attention_layer_localization` (diagnostics):
//!   one lane (history 714, suffix 200) on vec-hybrid and the tiled arm with
//!   the per-layer probe armed; prints, per layer, the max |difference| of
//!   the attention output and of the residual after the layer, the worst
//!   row, and the first layer whose attention differs by more than 5% of its
//!   magnitude.
use super::gemma_warm_batch_tests::{
    assert_lane_equal, load_gate_model, reserve_for, LaneImage, WarmFixture, HISTORIES, MAX_DECODE,
    SUFFIXES,
};
use super::*;
use crate::inference::models::gemma4::GemmaLiveKernel;
use crate::serve::forward_prefill_batched::live_layer_probe;

const REPEATS: usize = 5;
/// Q1 decision policy near-tie threshold: a vec-hybrid top-1/top-2 log-prob
/// margin below this many nats is a near tie, where a top-1 flip is
/// recorded, not failed.
const NEAR_TIE_NAT: f64 = 0.10;

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

/// Log-probabilities (natural log, F64) of a logits row.
fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_sum = row
        .iter()
        .map(|&value| (value as f64 - max).exp())
        .sum::<f64>()
        .ln();
    row.iter()
        .map(|&value| value as f64 - max - log_sum)
        .collect()
}

/// `(top-1 index, top-2 index)` of a row.
fn top2(row: &[f64]) -> (usize, usize) {
    let mut best = (usize::MAX, f64::NEG_INFINITY);
    let mut second = best;
    for (index, &value) in row.iter().enumerate() {
        if value > best.1 {
            second = best;
            best = (index, value);
        } else if value > second.1 {
            second = (index, value);
        }
    }
    (best.0, second.0)
}

/// Max |a − b| over `indices`.
fn max_abs_at(a: &[f64], b: &[f64], indices: impl IntoIterator<Item = usize>) -> f64 {
    indices
        .into_iter()
        .map(|index| (a[index] - b[index]).abs())
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "official Gemma 4 GGUF: tiled vs vec-hybrid warm attention decision envelope (Q1 evidence)"]
fn gemma_warm_tiled_attention_decision_envelope() {
    decision_envelope(GemmaLiveKernel::TiledF16);
}

#[test]
#[ignore = "official Gemma 4 GGUF: tiled precise vs vec-hybrid warm attention decision envelope (Q1 evidence)"]
fn gemma_warm_tiled_precise_attention_decision_envelope() {
    decision_envelope(GemmaLiveKernel::TiledF16Precise);
}

fn decision_envelope(kernel: GemmaLiveKernel) {
    assert!(kernel.is_tiled(), "the envelope compares a tiled arm");
    let mut loaded = load_gate_model();
    let LoadedModel::Gemma(g) = &mut loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    // Histories come from cold prefill, which never runs live attention, so
    // both routes resume from identical bytes.
    let fixture = WarmFixture::build(&mut guard);
    let prompts = &fixture.prompts;
    let run = |guard: &mut Gemma4KvGuard<'_>, arm: GemmaLiveKernel, slot: usize, prompt: &[u32]| {
        guard.model.weights.warm_live_kernel = arm;
        fixture.per_slot(guard, slot, prompt, HISTORIES[slot], MAX_DECODE)
    };

    let mut worst_vocab = 0.0f64;
    let mut worst_top2 = 0.0f64;
    let mut agree = 0usize;
    let mut near_ties = 0usize;
    let mut min_margin = f64::INFINITY;
    let mut failures = Vec::new();
    for slot in 0..4 {
        let vec_image = run(&mut guard, GemmaLiveKernel::VecHybrid, slot, &prompts[slot]);
        let tiled_image = run(&mut guard, kernel, slot, &prompts[slot]);
        let vec_lp = log_softmax(&logits(&vec_image));
        let tiled_lp = log_softmax(&logits(&tiled_image));
        let max_vocab = max_abs_at(&vec_lp, &tiled_lp, 0..vec_lp.len());
        let (vec_top, vec_second) = top2(&vec_lp);
        let (tiled_top, tiled_second) = top2(&tiled_lp);
        let max_top2 = max_abs_at(&vec_lp, &tiled_lp, [vec_top, vec_second]);
        let vec_margin = vec_lp[vec_top] - vec_lp[vec_second];
        let tiled_margin = tiled_lp[tiled_top] - tiled_lp[tiled_second];
        let near_tie = vec_margin < NEAR_TIE_NAT;
        let top1_agree = vec_top == tiled_top;
        worst_vocab = worst_vocab.max(max_vocab);
        worst_top2 = worst_top2.max(max_top2);
        min_margin = min_margin.min(vec_margin);
        agree += usize::from(top1_agree);
        near_ties += usize::from(near_tie);
        println!(
            "GEMMA_WARM_TILED_ENVELOPE_LANE kernel={kernel:?} slot={slot} history={} suffix={} max_abs_logprob_diff={max_vocab:.6} max_abs_logprob_diff_top2={max_top2:.6} top1_vec={vec_top} top2_vec={vec_second} top1_tiled={tiled_top} top1_agree={top1_agree} logprob_vec_top2=({:.6},{:.6}) logprob_tiled_at_vec_top2=({:.6},{:.6}) margin_vec_nat={vec_margin:.6} margin_tiled_nat={tiled_margin:.6} near_tie={near_tie} first_token_vec={} first_token_tiled={}",
            HISTORIES[slot],
            SUFFIXES[slot],
            vec_lp[vec_top],
            vec_lp[vec_second],
            tiled_lp[vec_top],
            tiled_lp[vec_second],
            vec_image.first_token,
            tiled_image.first_token
        );
        if !top1_agree && !near_tie {
            failures.push(slot);
        }
    }

    // Below 32 suffix rows the warm route keeps vec-hybrid under either
    // switch, so the two arms must be the same graph bit for bit.
    let tiny: Vec<u32> = prompts[0][..HISTORIES[0] + 31].to_vec();
    let tiny_vec = run(&mut guard, GemmaLiveKernel::VecHybrid, 0, &tiny);
    let tiny_tiled = run(&mut guard, kernel, 0, &tiny);
    assert_lane_equal(
        "31-row suffix keeps vec-hybrid on both switches",
        &tiny_tiled,
        &tiny_vec,
    );

    println!(
        "GEMMA_WARM_TILED_ENVELOPE kernel={kernel:?} lanes=4 top1_agree={agree}/4 near_ties={near_ties} near_tie_nat={NEAR_TIE_NAT} max_abs_logprob_diff={worst_vocab:.6} max_abs_logprob_diff_top2={worst_top2:.6} min_margin_vec_nat={min_margin:.6} policy_failures={failures:?} tiny_suffix_equal=true histories={HISTORIES:?} suffixes={SUFFIXES:?}"
    );
    assert!(
        failures.is_empty(),
        "{kernel:?} warm attention flipped top-1 outside a {NEAR_TIE_NAT}-nat near tie on slots {failures:?}"
    );
}

/// `(max |a − b|, max |b|)`.
fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold((0.0f32, 0.0f32), |(diff, scale), (x, y)| {
            (diff.max((x - y).abs()), scale.max(y.abs()))
        })
}

/// Attention relative difference above which a layer counts as diverged.
const LARGE_ATTN_REL: f32 = 0.05;

#[test]
#[ignore = "official Gemma 4 GGUF: per-layer tiled vs vec-hybrid warm attention divergence (diagnostic)"]
fn gemma_warm_tiled_attention_layer_localization() {
    layer_localization(GemmaLiveKernel::TiledF16);
}

#[test]
#[ignore = "official Gemma 4 GGUF: per-layer tiled precise vs vec-hybrid warm attention divergence (diagnostic)"]
fn gemma_warm_tiled_precise_attention_layer_localization() {
    layer_localization(GemmaLiveKernel::TiledF16Precise);
}

fn layer_localization(kernel: GemmaLiveKernel) {
    assert!(kernel.is_tiled(), "the localization compares a tiled arm");
    let mut loaded = load_gate_model();
    let LoadedModel::Gemma(g) = &mut loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    let fixture = WarmFixture::build(&mut guard);
    let slot = 1;
    let rows = SUFFIXES[slot];
    let prompt = fixture.prompts[slot].clone();
    let mut capture = |arm: GemmaLiveKernel| {
        guard.model.weights.warm_live_kernel = arm;
        live_layer_probe::arm();
        let image = fixture.per_slot(&mut guard, slot, &prompt, HISTORIES[slot], MAX_DECODE);
        (image, live_layer_probe::take())
    };
    let (vec_image, vec_layers) = capture(GemmaLiveKernel::VecHybrid);
    let (tiled_image, tiled_layers) = capture(kernel);
    assert!(!vec_layers.is_empty(), "the probe captured no layer");
    assert_eq!(
        vec_layers.len(),
        tiled_layers.len(),
        "one warm prefill per arm"
    );

    let mut first_large = None;
    for (vec_layer, tiled_layer) in vec_layers.iter().zip(&tiled_layers) {
        assert_eq!(vec_layer.layer, tiled_layer.layer);
        let row_elems = vec_layer.attn.len() / rows;
        let hidden = vec_layer.hidden.len() / rows;
        let (attn_max, attn_scale) = diff_stats(&tiled_layer.attn, &vec_layer.attn);
        let (hidden_max, hidden_scale) = diff_stats(&tiled_layer.hidden, &vec_layer.hidden);
        let row_diff = |row: usize| {
            let span = row * row_elems..(row + 1) * row_elems;
            diff_stats(&tiled_layer.attn[span.clone()], &vec_layer.attn[span]).0
        };
        let (worst_row, worst_row_diff) =
            (0..rows)
                .map(|row| (row, row_diff(row)))
                .fold(
                    (0, 0.0f32),
                    |best, item| if item.1 > best.1 { item } else { best },
                );
        let last = (rows - 1) * hidden..rows * hidden;
        let (hidden_last, _) =
            diff_stats(&tiled_layer.hidden[last.clone()], &vec_layer.hidden[last]);
        let attn_rel = attn_max / attn_scale.max(f32::MIN_POSITIVE);
        if first_large.is_none() && attn_rel > LARGE_ATTN_REL {
            first_large = Some(vec_layer.layer);
        }
        println!(
            "GEMMA_WARM_TILED_LAYER kernel={kernel:?} layer={} sliding={} head_dim={} attn_max_abs={attn_max:.6} attn_rel={attn_rel:.6} attn_last_row_max_abs={:.6} worst_row={worst_row} worst_row_max_abs={worst_row_diff:.6} hidden_max_abs={hidden_max:.6} hidden_rel={:.6} hidden_last_row_max_abs={hidden_last:.6}",
            vec_layer.layer,
            vec_layer.sliding,
            vec_layer.head_dim,
            row_diff(rows - 1),
            hidden_max / hidden_scale.max(f32::MIN_POSITIVE),
        );
    }
    let (logit_max, _) = diff_stats(
        &tiled_image
            .logits
            .iter()
            .map(|&bits| f32::from_bits(bits))
            .collect::<Vec<_>>(),
        &vec_image
            .logits
            .iter()
            .map(|&bits| f32::from_bits(bits))
            .collect::<Vec<_>>(),
    );
    println!(
        "GEMMA_WARM_TILED_LOCALIZATION kernel={kernel:?} slot={slot} history={} suffix={rows} layers={} first_layer_attn_rel_above_{LARGE_ATTN_REL}={first_large:?} final_logit_max_abs={logit_max:.6}",
        HISTORIES[slot],
        vec_layers.len()
    );
}
