//! ADR-063 Stage 2 tiled F16 warm attention: real-artifact determinism, every-row route
//! envelope and localization diagnostics. Joint == per-slot on the tiled route is
//! `gemma_joint_warm_prefill_matches_per_slot_bitwise` (Stage 1 harness).
//!
//! Ignored; `HF2Q_GEMMA4_GGUF` names the official Gemma 4 26B-A4B GGUF. Run
//! each test alone with `--exact --ignored --nocapture --test-threads=1`.
//! * `gemma_warm_tiled_attention_is_deterministic`: on the tiled route, each
//!   gate lane's per-slot warm prefill and the four-lane joint cohort repeat
//!   bit for bit (first token, logits bits, KV bytes, cursors).
//! * `gemma_warm_attention_route_envelope`: per lane, every suffix row's
//!   logits on each route in `ENVELOPE_ROUTES` against `VecHybrid` from
//!   identical histories, with the Q1 §3.2 statistic D and the 0.10-nat
//!   near-tie policy per row (recorded, not asserted). Asserted: the
//!   `VecHybrid` rerun equals the reference bit for bit, and a 31-row suffix
//!   is the vec-hybrid graph under every tiled route. The precise F32 route
//!   failed this envelope (ADR-063 Q4-GEMMA-ATTN, 72 near-tie failures over
//!   952 rows) and was removed; a future warm route plugs in here.
//! * `gemma_warm_tiled_attention_layer_localization` (diagnostic): one lane
//!   (history 714, suffix 200) on every route with the per-layer probe armed;
//!   prints, per layer, the attention and residual differences against
//!   vec-hybrid (max-relative and relative RMS).
use super::gemma_warm_batch_tests::{
    assert_lane_equal, load_gate_model, reserve_for, LaneImage, WarmFixture, HISTORIES, MAX_DECODE,
    SUFFIXES,
};
use super::*;
use crate::inference::models::gemma4::GemmaLiveKernel;
use crate::serve::forward_prefill_batched::{live_layer_probe, live_row_logits_probe};

const REPEATS: usize = 5;
/// Q1 near-tie policy (`hf2q.q1.numeric-policy.v1`): a top-1 change where
/// the baseline margin is at least this many nats is a policy failure.
const NEAR_TIE_NAT: f64 = 0.10;
/// Q1 §3.2 τ_route.
const TAU_ROUTE_NAT: f64 = 0.25;
/// Warm routes measured against `VecHybrid` by the envelope and localization diagnostics.
const ENVELOPE_ROUTES: [GemmaLiveKernel; 1] = [GemmaLiveKernel::TiledF16];

#[test]
#[ignore = "official Gemma 4 GGUF: tiled F16 warm attention repeats bit for bit"]
fn gemma_warm_tiled_attention_is_deterministic() {
    warm_route_is_deterministic(GemmaLiveKernel::TiledF16);
}

fn warm_route_is_deterministic(kernel: GemmaLiveKernel) {
    let mut loaded = load_gate_model();
    let LoadedModel::Gemma(g) = &mut loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    guard.model.weights.warm_live_kernel = kernel;
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
                &format!("{kernel:?} per-slot slot {slot} repeat {repeat}"),
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
                &format!("{kernel:?} joint slot {slot} repeat {repeat}"),
                image,
                first,
            );
        }
    }
    for (slot, (joint, per_slot)) in first_joint.iter().zip(&first_runs).enumerate() {
        assert_lane_equal(
            &format!("{kernel:?} joint == per-slot slot {slot}"),
            joint,
            per_slot,
        );
    }
    println!(
        "GEMMA_WARM_TILED_DETERMINISM_COMPLETE kernel={kernel:?} repeats={REPEATS} per_slot_lanes=4 joint_lanes=4 logits_bits=true kv_bytes=true cursors=true histories={HISTORIES:?} suffixes={SUFFIXES:?}"
    );
}

/// `(top-1, top-2)` indices of a logits row (lowest index wins a tie).
fn top2(row: &[f32]) -> (usize, usize) {
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut second = (0usize, f32::NEG_INFINITY);
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

/// `log Σ exp(row)` in F64.
fn log_normalizer(row: &[f32]) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    max + row
        .iter()
        .map(|&value| (value as f64 - max).exp())
        .sum::<f64>()
        .ln()
}

/// Q1 §3.2 decision statistic of one row.
struct RowEnvelope {
    row: usize,
    /// Baseline (vec-hybrid) top-1 `a` and the comparison token `c` (route
    /// top-1 if it differs from `a`, else the baseline's second token).
    a: usize,
    c: usize,
    /// Log-probabilities `L[a]`, `L[c]`, `R[a]`, `R[c]`.
    l: (f64, f64),
    r: (f64, f64),
    route_top1: usize,
    route_top1_logprob: f64,
    /// Baseline top-1 minus top-2 (nat).
    vec_margin: f64,
    d: f64,
}

impl RowEnvelope {
    fn measure(row: usize, reference: &[f32], route: &[f32]) -> Self {
        assert_eq!(reference.len(), route.len());
        let (a, vec_second) = top2(reference);
        let (route_top1, _) = top2(route);
        let c = if route_top1 != a { route_top1 } else { vec_second };
        let (zl, zr) = (log_normalizer(reference), log_normalizer(route));
        let lp = |values: &[f32], z: f64, index: usize| values[index] as f64 - z;
        let l = (lp(reference, zl, a), lp(reference, zl, c));
        let r = (lp(route, zr, a), lp(route, zr, c));
        Self {
            row,
            a,
            c,
            l,
            r,
            route_top1,
            route_top1_logprob: lp(route, zr, route_top1),
            vec_margin: reference[a] as f64 - reference[vec_second] as f64,
            d: ((l.0 - l.1) - (r.0 - r.1)).abs(),
        }
    }

    fn flipped(&self) -> bool {
        self.route_top1 != self.a
    }

    fn policy_failure(&self) -> bool {
        self.flipped() && self.vec_margin >= NEAR_TIE_NAT
    }
}

/// A per-slot warm prefill with every suffix row's logits captured.
fn run_rows(
    fixture: &WarmFixture,
    guard: &mut Gemma4KvGuard<'_>,
    kernel: GemmaLiveKernel,
    slot: usize,
    prompt: &[u32],
) -> (LaneImage, Vec<Vec<f32>>) {
    guard.model.weights.warm_live_kernel = kernel;
    live_row_logits_probe::arm();
    let image = fixture.per_slot(guard, slot, prompt, HISTORIES[slot], MAX_DECODE);
    let rows = live_row_logits_probe::take();
    let suffix = prompt.len() - HISTORIES[slot];
    assert_eq!(rows.len(), suffix, "{kernel:?} slot {slot}: every suffix row captured");
    for (index, (row, _)) in rows.iter().enumerate() {
        assert_eq!(*row, index, "{kernel:?} slot {slot}: rows in stream order");
    }
    let last = &rows[suffix - 1].1;
    assert!(
        last.iter().map(|value| value.to_bits()).eq(image.logits.iter().copied()),
        "{kernel:?} slot {slot}: the last captured row is the published logits"
    );
    (image, rows.into_iter().map(|(_, logits)| logits).collect())
}

fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

#[test]
#[ignore = "official Gemma 4 GGUF: every-row route envelope of the tiled warm route against vec-hybrid (diagnostic; vec control and tiny suffix asserted)"]
fn gemma_warm_attention_route_envelope() {
    let mut loaded = load_gate_model();
    let LoadedModel::Gemma(g) = &mut loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    // Histories come from cold prefill, which never runs live attention, so
    // every route resumes from identical bytes.
    let fixture = WarmFixture::build(&mut guard);
    let prompts = fixture.prompts.clone();

    let mut per_route: Vec<Vec<(usize, RowEnvelope)>> = ENVELOPE_ROUTES.iter().map(|_| Vec::new()).collect();
    for slot in 0..4 {
        let (reference_image, reference) =
            run_rows(&fixture, &mut guard, GemmaLiveKernel::VecHybrid, slot, &prompts[slot]);
        // A4: the vec-hybrid control rerun equals the reference bit for bit.
        let (control_image, control) =
            run_rows(&fixture, &mut guard, GemmaLiveKernel::VecHybrid, slot, &prompts[slot]);
        assert_lane_equal(&format!("vec-hybrid control slot {slot}"), &control_image, &reference_image);
        assert!(bits(&control) == bits(&reference), "vec-hybrid control slot {slot}: row logits bits");
        for (route, rows) in ENVELOPE_ROUTES.into_iter().zip(&mut per_route) {
            let (_, route_rows) = run_rows(&fixture, &mut guard, route, slot, &prompts[slot]);
            for (row, (l, r)) in reference.iter().zip(&route_rows).enumerate() {
                let envelope = RowEnvelope::measure(row, l, r);
                println!(
                    "GEMMA_WARM_ROUTE_ENVELOPE_ROW route={route:?} slot={slot} row={} a={} c={} logprob_vec=({:.6},{:.6}) logprob_route=({:.6},{:.6}) route_top1={} route_top1_logprob={:.6} margin_vec_nat={:.6} d_nat={:.6} flipped={} policy_failure={}",
                    envelope.row,
                    envelope.a,
                    envelope.c,
                    envelope.l.0,
                    envelope.l.1,
                    envelope.r.0,
                    envelope.r.1,
                    envelope.route_top1,
                    envelope.route_top1_logprob,
                    envelope.vec_margin,
                    envelope.d,
                    envelope.flipped(),
                    envelope.policy_failure(),
                );
                rows.push((slot, envelope));
            }
        }
    }

    let expected_rows: usize = SUFFIXES.iter().sum();
    let mut routes_json = Vec::new();
    for (route, rows) in ENVELOPE_ROUTES.into_iter().zip(&per_route) {
        // Row coverage (frozen): every suffix row of every lane.
        assert_eq!(rows.len(), expected_rows, "{route:?}: rows covered");
        let mut lanes_json = Vec::new();
        for slot in 0..4 {
            let lane: Vec<&RowEnvelope> = rows.iter().filter(|(s, _)| *s == slot).map(|(_, e)| e).collect();
            assert_eq!(lane.len(), SUFFIXES[slot]);
            let max_d = lane.iter().map(|e| e.d).fold(0.0, f64::max);
            let agree = lane.iter().filter(|e| !e.flipped()).count();
            let sensitive = lane.iter().filter(|e| e.d > TAU_ROUTE_NAT).count();
            let failures: Vec<usize> = lane.iter().filter(|e| e.policy_failure()).map(|e| e.row).collect();
            let last = lane.last().unwrap();
            println!(
                "GEMMA_WARM_ROUTE_ENVELOPE_LANE route={route:?} slot={slot} history={} suffix={} rows={} top1_agree={agree}/{} max_d_nat={max_d:.6} rows_d_above_tau={sensitive} policy_failures={failures:?} last_row_d_nat={:.6} last_row_margin_vec_nat={:.6}",
                HISTORIES[slot],
                SUFFIXES[slot],
                lane.len(),
                lane.len(),
                last.d,
                last.vec_margin,
            );
            lanes_json.push(serde_json::json!({
                "slot": slot, "history": HISTORIES[slot], "suffix": SUFFIXES[slot], "rows": lane.len(),
                "top1_agree": agree, "max_d_nat": max_d, "rows_d_above_tau": sensitive,
                "policy_failure_rows": failures, "last_row_d_nat": last.d,
            }));
        }
        let max_d = rows.iter().map(|(_, e)| e.d).fold(0.0, f64::max);
        let verdict = if rows.iter().all(|(_, e)| e.d == 0.0) {
            "ExactEqualDecision"
        } else if max_d <= TAU_ROUTE_NAT {
            "WithinDecisionEnvelope"
        } else {
            "RouteSensitive"
        };
        routes_json.push(serde_json::json!({
            "route": format!("{route:?}"),
            "rows": rows.len(), "max_d_nat": max_d, "q1_class_b": verdict,
            "policy_failures": rows.iter().filter(|(_, e)| e.policy_failure()).count(),
            "per_lane": lanes_json,
        }));
    }
    let summary = serde_json::json!({
        "gate": "gemma_warm_attention_route_envelope",
        "policy": "hf2q.q1.numeric-policy.v1",
        "reference": "VecHybrid",
        "rows": "all",
        "rows_per_route": expected_rows,
        "near_tie_nat": NEAR_TIE_NAT,
        "tau_route_nat": TAU_ROUTE_NAT,
        "histories": HISTORIES,
        "suffixes": SUFFIXES,
        "vec_control_bitwise": true,
        "tiny_suffix_equal": "checked after this summary",
        "routes": routes_json,
    });
    println!("GEMMA_WARM_ROUTE_ENVELOPE_JSON {summary}");

    // A3: below 32 suffix rows every tiled arm keeps the vec-hybrid graph. A 31-row resume never
    // reaches the batched prefill (the row probe would see no rows), so compare the full lane image:
    // first token, final logits bits, KV bytes and cursors.
    let tiny: Vec<u32> = prompts[0][..HISTORIES[0] + 31].to_vec();
    let run = |guard: &mut Gemma4KvGuard<'_>, kernel: GemmaLiveKernel| {
        guard.model.weights.warm_live_kernel = kernel;
        fixture.per_slot(guard, 0, &tiny, HISTORIES[0], MAX_DECODE)
    };
    let tiny_vec = run(&mut guard, GemmaLiveKernel::VecHybrid);
    for route in ENVELOPE_ROUTES {
        let tiny_route = run(&mut guard, route);
        assert_lane_equal(&format!("31-row suffix keeps vec-hybrid under {route:?}"), &tiny_route, &tiny_vec);
    }
    println!("GEMMA_WARM_ROUTE_ENVELOPE_A3 tiny_suffix_rows=31 tiny_suffix_equal=true routes={ENVELOPE_ROUTES:?}");
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

/// `rms(a − b) / rms(b)` (descriptive).
fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (num, den) = a.iter().zip(b).fold((0.0f64, 0.0f64), |(num, den), (&x, &y)| {
        let diff = x as f64 - y as f64;
        (num + diff * diff, den + y as f64 * y as f64)
    });
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

/// Attention relative difference above which a layer counts as diverged.
const LARGE_ATTN_REL: f32 = 0.05;

#[test]
#[ignore = "official Gemma 4 GGUF: per-layer tiled vs vec-hybrid warm attention divergence (diagnostic)"]
fn gemma_warm_tiled_attention_layer_localization() {
    let mut loaded = load_gate_model();
    let LoadedModel::Gemma(g) = &mut loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    let fixture = WarmFixture::build(&mut guard);
    let slot = 1;
    let rows = SUFFIXES[slot];
    let prompt = fixture.prompts[slot].clone();
    let mut capture = |kernel: GemmaLiveKernel| {
        guard.model.weights.warm_live_kernel = kernel;
        live_layer_probe::arm();
        let image = fixture.per_slot(&mut guard, slot, &prompt, HISTORIES[slot], MAX_DECODE);
        (image, live_layer_probe::take())
    };
    let (vec_image, vec_layers) = capture(GemmaLiveKernel::VecHybrid);
    assert!(!vec_layers.is_empty(), "the probe captured no layer");
    for route in ENVELOPE_ROUTES {
        let (route_image, route_layers) = capture(route);
        assert_eq!(vec_layers.len(), route_layers.len(), "one warm prefill per arm");
        let mut first_large = None;
        for (vec_layer, route_layer) in vec_layers.iter().zip(&route_layers) {
            assert_eq!(vec_layer.layer, route_layer.layer);
            let row_elems = vec_layer.attn.len() / rows;
            let hidden = vec_layer.hidden.len() / rows;
            let (attn_max, attn_scale) = diff_stats(&route_layer.attn, &vec_layer.attn);
            let (hidden_max, hidden_scale) = diff_stats(&route_layer.hidden, &vec_layer.hidden);
            let row_diff = |row: usize| {
                let span = row * row_elems..(row + 1) * row_elems;
                diff_stats(&route_layer.attn[span.clone()], &vec_layer.attn[span]).0
            };
            let (worst_row, worst_row_diff) = (0..rows)
                .map(|row| (row, row_diff(row)))
                .fold((0, 0.0f32), |best, item| if item.1 > best.1 { item } else { best });
            let last = (rows - 1) * hidden..rows * hidden;
            let (hidden_last, _) =
                diff_stats(&route_layer.hidden[last.clone()], &vec_layer.hidden[last]);
            let attn_rel = attn_max / attn_scale.max(f32::MIN_POSITIVE);
            if first_large.is_none() && attn_rel > LARGE_ATTN_REL {
                first_large = Some(vec_layer.layer);
            }
            println!(
                "GEMMA_WARM_TILED_LAYER route={route:?} layer={} sliding={} head_dim={} attn_max_abs={attn_max:.6e} attn_rel={attn_rel:.6e} attn_rel_rms={:.6e} attn_last_row_max_abs={:.6e} worst_row={worst_row} worst_row_max_abs={worst_row_diff:.6e} hidden_max_abs={hidden_max:.6e} hidden_rel={:.6e} hidden_last_row_max_abs={hidden_last:.6e}",
                vec_layer.layer,
                vec_layer.sliding,
                vec_layer.head_dim,
                rel_rms(&route_layer.attn, &vec_layer.attn),
                row_diff(rows - 1),
                hidden_max / hidden_scale.max(f32::MIN_POSITIVE),
            );
        }
        let to_f32 = |image: &LaneImage| -> Vec<f32> {
            image.logits.iter().map(|&bits| f32::from_bits(bits)).collect()
        };
        let (logit_max, _) = diff_stats(&to_f32(&route_image), &to_f32(&vec_image));
        println!(
            "GEMMA_WARM_TILED_LOCALIZATION route={route:?} slot={slot} history={} suffix={rows} layers={} first_layer_attn_rel_above_{LARGE_ATTN_REL}={first_large:?} final_logit_max_abs={logit_max:.6}",
            HISTORIES[slot],
            vec_layers.len()
        );
    }
}
