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
//! * `gemma_warm_attention_route_envelope`: on the same cold-prefilled
//!   histories, every warm route (`VecHybrid` control rerun, `TiledF16`,
//!   `TiledF16Precise`) against the vec-hybrid reference, per lane at the
//!   last row: max |Δ log-prob| over the vocabulary and over the vec-hybrid
//!   top-2, top-1 agreement and the top-1/top-2 margins in nats. Prints one
//!   `GEMMA_WARM_ROUTE_ENVELOPE_LANE` line per (route, lane), one
//!   `GEMMA_WARM_ROUTE_ENVELOPE` line per route and a final
//!   `GEMMA_WARM_ROUTE_ENVELOPE_JSON {…}` summary. A top-1 flip outside a
//!   policy near tie (vec-hybrid margin < [`NEAR_TIE_NAT`]) fails the gate on
//!   [`ENVELOPE_ASSERTED_ROUTE`] only; the control must be bitwise equal to
//!   the reference, and a suffix below the 32-row tiled threshold must be
//!   bitwise equal across routes (all run vec-hybrid).
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

/// Warm routes the envelope measures against the vec-hybrid reference, all
/// from the same histories. The `VecHybrid` arm is the control: a rerun of
/// the reference graph, bitwise equal to it (all deltas zero).
const ENVELOPE_ROUTES: [GemmaLiveKernel; 3] = [
    GemmaLiveKernel::VecHybrid,
    GemmaLiveKernel::TiledF16,
    GemmaLiveKernel::TiledF16Precise,
];

/// The route whose near-tie policy failures fail the gate. `TiledF16` is
/// measured, not asserted: it flipped top-1 at vec-hybrid margins of 0.167
/// and 0.392 nat in the Stage 2 run and has been default off since the
/// 2026-11-01 quality gate.
const ENVELOPE_ASSERTED_ROUTE: GemmaLiveKernel = GemmaLiveKernel::TiledF16Precise;

/// One route's lane against the vec-hybrid reference, at the last row.
struct LaneEnvelope {
    slot: usize,
    /// Max |Δ log-prob| over the vocabulary.
    max_vocab: f64,
    /// Max |Δ log-prob| over the vec-hybrid top-2.
    max_top2: f64,
    vec_top2: (usize, usize),
    route_top2: (usize, usize),
    vec_logprobs: (f64, f64),
    route_logprobs_at_vec_top2: (f64, f64),
    vec_margin: f64,
    route_margin: f64,
    near_tie: bool,
    top1_agree: bool,
    first_tokens: (u32, u32),
}

impl LaneEnvelope {
    fn measure(slot: usize, vec_image: &LaneImage, route_image: &LaneImage) -> Self {
        let vec_lp = log_softmax(&logits(vec_image));
        let route_lp = log_softmax(&logits(route_image));
        let (vec_top, vec_second) = top2(&vec_lp);
        let (route_top, route_second) = top2(&route_lp);
        let vec_margin = vec_lp[vec_top] - vec_lp[vec_second];
        Self {
            slot,
            max_vocab: max_abs_at(&vec_lp, &route_lp, 0..vec_lp.len()),
            max_top2: max_abs_at(&vec_lp, &route_lp, [vec_top, vec_second]),
            vec_top2: (vec_top, vec_second),
            route_top2: (route_top, route_second),
            vec_logprobs: (vec_lp[vec_top], vec_lp[vec_second]),
            route_logprobs_at_vec_top2: (route_lp[vec_top], route_lp[vec_second]),
            vec_margin,
            route_margin: route_lp[route_top] - route_lp[route_second],
            near_tie: vec_margin < NEAR_TIE_NAT,
            top1_agree: vec_top == route_top,
            first_tokens: (vec_image.first_token, route_image.first_token),
        }
    }

    /// A top-1 flip outside a policy near tie.
    fn policy_failure(&self) -> bool {
        !self.top1_agree && !self.near_tie
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "slot": self.slot,
            "history": HISTORIES[self.slot],
            "suffix": SUFFIXES[self.slot],
            "max_abs_logprob_diff": self.max_vocab,
            "max_abs_logprob_diff_top2": self.max_top2,
            "top1_vec": self.vec_top2.0,
            "top2_vec": self.vec_top2.1,
            "top1_route": self.route_top2.0,
            "top2_route": self.route_top2.1,
            "top1_agree": self.top1_agree,
            "logprob_vec_top2": [self.vec_logprobs.0, self.vec_logprobs.1],
            "logprob_route_at_vec_top2": [
                self.route_logprobs_at_vec_top2.0,
                self.route_logprobs_at_vec_top2.1
            ],
            "margin_vec_nat": self.vec_margin,
            "margin_route_nat": self.route_margin,
            "near_tie": self.near_tie,
            "policy_failure": self.policy_failure(),
            "first_token_vec": self.first_tokens.0,
            "first_token_route": self.first_tokens.1,
        })
    }
}

#[test]
#[ignore = "official Gemma 4 GGUF: vec-hybrid / tiled / tiled-precise vs vec-hybrid warm attention decision envelope (Q1 evidence)"]
fn gemma_warm_attention_route_envelope() {
    let mut loaded = load_gate_model();
    let LoadedModel::Gemma(g) = &mut loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    // Histories come from cold prefill, which never runs live attention, so
    // every route resumes from identical bytes.
    let fixture = WarmFixture::build(&mut guard);
    let prompts = &fixture.prompts;
    let run = |guard: &mut Gemma4KvGuard<'_>, arm: GemmaLiveKernel, slot: usize, prompt: &[u32]| {
        guard.model.weights.warm_live_kernel = arm;
        fixture.per_slot(guard, slot, prompt, HISTORIES[slot], MAX_DECODE)
    };

    let mut lanes: Vec<Vec<LaneEnvelope>> = ENVELOPE_ROUTES.iter().map(|_| Vec::new()).collect();
    for slot in 0..4 {
        let reference = run(&mut guard, GemmaLiveKernel::VecHybrid, slot, &prompts[slot]);
        for (route, route_lanes) in ENVELOPE_ROUTES.into_iter().zip(&mut lanes) {
            let image = run(&mut guard, route, slot, &prompts[slot]);
            if route == GemmaLiveKernel::VecHybrid {
                assert_lane_equal(
                    &format!("vec-hybrid control slot {slot}"),
                    &image,
                    &reference,
                );
            }
            let lane = LaneEnvelope::measure(slot, &reference, &image);
            println!(
                "GEMMA_WARM_ROUTE_ENVELOPE_LANE route={route:?} slot={slot} history={} suffix={} max_abs_logprob_diff={:.6} max_abs_logprob_diff_top2={:.6} top1_vec={} top2_vec={} top1_route={} top2_route={} top1_agree={} logprob_vec_top2=({:.6},{:.6}) logprob_route_at_vec_top2=({:.6},{:.6}) margin_vec_nat={:.6} margin_route_nat={:.6} near_tie={} policy_failure={} first_token_vec={} first_token_route={}",
                HISTORIES[slot],
                SUFFIXES[slot],
                lane.max_vocab,
                lane.max_top2,
                lane.vec_top2.0,
                lane.vec_top2.1,
                lane.route_top2.0,
                lane.route_top2.1,
                lane.top1_agree,
                lane.vec_logprobs.0,
                lane.vec_logprobs.1,
                lane.route_logprobs_at_vec_top2.0,
                lane.route_logprobs_at_vec_top2.1,
                lane.vec_margin,
                lane.route_margin,
                lane.near_tie,
                lane.policy_failure(),
                lane.first_tokens.0,
                lane.first_tokens.1,
            );
            route_lanes.push(lane);
        }
    }

    // Below 32 suffix rows the warm route keeps vec-hybrid under every
    // switch, so each tiled arm must be the vec-hybrid graph bit for bit.
    let tiny: Vec<u32> = prompts[0][..HISTORIES[0] + 31].to_vec();
    let tiny_vec = run(&mut guard, GemmaLiveKernel::VecHybrid, 0, &tiny);
    for route in ENVELOPE_ROUTES.into_iter().filter(|route| route.is_tiled()) {
        let tiny_route = run(&mut guard, route, 0, &tiny);
        assert_lane_equal(
            &format!("31-row suffix keeps vec-hybrid under {route:?}"),
            &tiny_route,
            &tiny_vec,
        );
    }

    let mut routes_json = Vec::with_capacity(ENVELOPE_ROUTES.len());
    for (route, route_lanes) in ENVELOPE_ROUTES.into_iter().zip(&lanes) {
        let agree = route_lanes.iter().filter(|lane| lane.top1_agree).count();
        let near_ties = route_lanes.iter().filter(|lane| lane.near_tie).count();
        let worst_vocab = route_lanes.iter().map(|lane| lane.max_vocab).fold(0.0, f64::max);
        let worst_top2 = route_lanes.iter().map(|lane| lane.max_top2).fold(0.0, f64::max);
        let min_vec_margin = route_lanes
            .iter()
            .map(|lane| lane.vec_margin)
            .fold(f64::INFINITY, f64::min);
        let failures: Vec<usize> = route_lanes
            .iter()
            .filter(|lane| lane.policy_failure())
            .map(|lane| lane.slot)
            .collect();
        println!(
            "GEMMA_WARM_ROUTE_ENVELOPE route={route:?} lanes={} top1_agree={agree}/{} near_ties={near_ties} near_tie_nat={NEAR_TIE_NAT} max_abs_logprob_diff={worst_vocab:.6} max_abs_logprob_diff_top2={worst_top2:.6} min_margin_vec_nat={min_vec_margin:.6} policy_failures={failures:?} asserted={}",
            route_lanes.len(),
            route_lanes.len(),
            route == ENVELOPE_ASSERTED_ROUTE,
        );
        routes_json.push(serde_json::json!({
            "route": format!("{route:?}"),
            "asserted": route == ENVELOPE_ASSERTED_ROUTE,
            "top1_agree": agree,
            "lanes": route_lanes.len(),
            "near_ties": near_ties,
            "max_abs_logprob_diff": worst_vocab,
            "max_abs_logprob_diff_top2": worst_top2,
            "min_margin_vec_nat": min_vec_margin,
            "policy_failures": failures,
            "policy_pass": failures.is_empty(),
            "per_lane": route_lanes.iter().map(LaneEnvelope::json).collect::<Vec<_>>(),
        }));
    }
    let summary = serde_json::json!({
        "gate": "gemma_warm_attention_route_envelope",
        "reference": "VecHybrid",
        "row": "last",
        "near_tie_nat": NEAR_TIE_NAT,
        "histories": HISTORIES,
        "suffixes": SUFFIXES,
        "vec_control_bitwise": true,
        "tiny_suffix_equal": true,
        "routes": routes_json,
    });
    println!("GEMMA_WARM_ROUTE_ENVELOPE_JSON {summary}");

    let asserted = ENVELOPE_ROUTES
        .iter()
        .position(|&route| route == ENVELOPE_ASSERTED_ROUTE)
        .expect("the asserted route is measured");
    let failures: Vec<usize> = lanes[asserted]
        .iter()
        .filter(|lane| lane.policy_failure())
        .map(|lane| lane.slot)
        .collect();
    assert!(
        failures.is_empty(),
        "{ENVELOPE_ASSERTED_ROUTE:?} warm attention flipped top-1 outside a {NEAR_TIE_NAT}-nat near tie on slots {failures:?}"
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
