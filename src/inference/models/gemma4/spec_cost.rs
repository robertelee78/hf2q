//! ADR-063 Q4-GEMMA-SPEC-SERVE M0 / gate G0 (design §8.1) — the frozen
//! round-cost projection. Pure arithmetic over measured figures; the
//! real-artifact harness (`engine::gemma_spec_m0`) supplies the timings.
//!
//! - Drafter bound: `td_bound(K, n) = 1.25 · K · (Σ_s t_mv(s, n) +
//!   t_argmax(n) + 4·t_flash(n) + 30·t_disp)`.
//! - Acceptance model: geometric per-depth acceptance fitted to the peer's
//!   L(3) = 3.34, `L(K) = Σ_{i=0..K} α^i`.
//! - Round cost: `c(K, n) = (td_bound + ts + tv(K+1, n) + E[tr]) / t1(n)`,
//!   where `E[tr] = (1 − α^(K·n)) · tr(K, n)`: `tr` is the measured
//!   full-tail restore of every wrapped lane in one submission (an upper
//!   bound on any partial restore), and under the frozen model that
//!   submission runs iff any of the n lanes rejects a draft. With fewer than
//!   n wrapped lanes this over-weights `tr`, which keeps the bound
//!   conservative.
//! - Projection: `ΔD(K) = Σ_n D·h(n)·(1 − c(K, n)/L(K))`; K* maximizes it
//!   over K ∈ {1, 2, 3}; G0 passes iff `ΔD(K*) ≥ 0.40 s`.

use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};

/// The peer's measured tokens emitted per round at depth 3 (§2.2).
pub(crate) const PEER_L3: f64 = 3.34;
/// hf2q's L1 decode median in seconds (§2.1).
pub(crate) const DECODE_SECONDS: f64 = 9.203;
/// Gate G0 threshold in seconds (decision D6).
pub(crate) const G0_SECONDS: f64 = 0.40;
/// Drafter-bound dispatch-sequence factor.
pub(crate) const TD_FACTOR: f64 = 1.25;
/// Flash dispatches per draft step (four drafter layers).
pub(crate) const TD_FLASH_PER_STEP: f64 = 4.0;
/// Small dispatches per draft step covered by `t_disp`.
pub(crate) const TD_SMALL_DISPATCHES: f64 = 30.0;

/// Solve `1 + α + α² + α³ = l3` for α ∈ (0, 1) by bisection.
pub(crate) fn alpha_from_l3(l3: f64) -> Result<f64> {
    ensure!((1.0..4.0).contains(&l3), "L(3) {l3} outside [1, 4)");
    let f = |a: f64| 1.0 + a + a * a + a * a * a - l3;
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Tokens emitted per round at depth `k` under the geometric model.
pub(crate) fn acceptance(alpha: f64, k: usize) -> f64 {
    (0..=k).map(|i| alpha.powi(i as i32)).sum()
}

/// Per-width drafter-bound components, seconds.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub(crate) struct DrafterBoundParts {
    pub sum_t_mv: f64,
    pub t_argmax: f64,
    pub t_flash: f64,
    pub t_disp: f64,
}

pub(crate) fn td_bound(k: usize, parts: &DrafterBoundParts) -> f64 {
    TD_FACTOR
        * k as f64
        * (parts.sum_t_mv
            + parts.t_argmax
            + TD_FLASH_PER_STEP * parts.t_flash
            + TD_SMALL_DISPATCHES * parts.t_disp)
}

/// Measured M0 figures, seconds. Keys: `n` = decoding width, `w` = window
/// width `K + 1`, `k` = depth.
#[derive(Debug, Clone, Default)]
pub(crate) struct RoundCostMeasurements {
    pub t1: BTreeMap<usize, f64>,
    /// `(w, n) → tv`.
    pub tv: BTreeMap<(usize, usize), f64>,
    /// `(k, n) → ts`.
    pub ts: BTreeMap<(usize, usize), f64>,
    /// `(k, n) → tr` (full-tail restore).
    pub tr: BTreeMap<(usize, usize), f64>,
    pub drafter: BTreeMap<usize, DrafterBoundParts>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct DepthProjection {
    pub k: usize,
    pub acceptance: f64,
    /// `n → c(K, n)`.
    pub round_cost: BTreeMap<usize, f64>,
    /// `n → td_bound(K, n)`.
    pub td_bound: BTreeMap<usize, f64>,
    pub delta_d_seconds: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct G0Decision {
    pub alpha: f64,
    /// `n → h(n)`.
    pub width_weights: BTreeMap<usize, f64>,
    pub depths: Vec<DepthProjection>,
    pub k_star: usize,
    pub delta_d_k_star: f64,
    pub threshold_seconds: f64,
    pub pass: bool,
}

/// `h(n)` from per-width spec-off tick seconds.
pub(crate) fn width_weights(tick_seconds: &BTreeMap<usize, f64>) -> Result<BTreeMap<usize, f64>> {
    let total: f64 = tick_seconds.values().sum();
    ensure!(total > 0.0, "width histogram has no tick time");
    ensure!(
        tick_seconds.values().all(|s| *s >= 0.0 && s.is_finite()),
        "width histogram has a negative or non-finite entry"
    );
    Ok(tick_seconds
        .iter()
        .filter(|(_, s)| **s > 0.0)
        .map(|(&n, &s)| (n, s / total))
        .collect())
}

pub(crate) fn round_cost(
    m: &RoundCostMeasurements,
    alpha: f64,
    k: usize,
    n: usize,
) -> Result<(f64, f64)> {
    let t1 = *m.t1.get(&n).with_context(|| format!("t1({n}) not measured"))?;
    ensure!(t1 > 0.0, "t1({n}) must be positive");
    let tv = *m
        .tv
        .get(&(k + 1, n))
        .with_context(|| format!("tv({}, {n}) not measured", k + 1))?;
    let ts = *m.ts.get(&(k, n)).with_context(|| format!("ts({k}, {n}) not measured"))?;
    let tr = *m.tr.get(&(k, n)).with_context(|| format!("tr({k}, {n}) not measured"))?;
    let parts = m
        .drafter
        .get(&n)
        .with_context(|| format!("drafter bound parts({n}) not measured"))?;
    let td = td_bound(k, parts);
    let expected_tr = (1.0 - alpha.powi((k * n) as i32)) * tr;
    Ok(((td + ts + tv + expected_tr) / t1, td))
}

/// The frozen G0 decision over K ∈ `depths` (normally {1, 2, 3}); depths
/// whose measurements are absent for any weighted width are skipped, and
/// at least one depth must be fully measured.
pub(crate) fn decide_g0(
    m: &RoundCostMeasurements,
    tick_seconds: &BTreeMap<usize, f64>,
    depths: &[usize],
) -> Result<G0Decision> {
    let alpha = alpha_from_l3(PEER_L3)?;
    let h = width_weights(tick_seconds)?;
    let mut out = Vec::new();
    for &k in depths {
        ensure!((1..=3).contains(&k), "depth {k} outside 1..=3");
        let l = acceptance(alpha, k);
        let mut costs = BTreeMap::new();
        let mut bounds = BTreeMap::new();
        let mut delta = 0.0;
        let mut complete = true;
        for (&n, &weight) in &h {
            match round_cost(m, alpha, k, n) {
                Ok((c, td)) => {
                    costs.insert(n, c);
                    bounds.insert(n, td);
                    delta += DECODE_SECONDS * weight * (1.0 - c / l);
                }
                Err(_) => {
                    complete = false;
                    break;
                }
            }
        }
        if complete {
            out.push(DepthProjection {
                k,
                acceptance: l,
                round_cost: costs,
                td_bound: bounds,
                delta_d_seconds: delta,
            });
        }
    }
    let best = out
        .iter()
        .max_by(|a, b| a.delta_d_seconds.total_cmp(&b.delta_d_seconds))
        .context("no depth fully measured over the weighted widths")?;
    let (k_star, delta_d_k_star) = (best.k, best.delta_d_seconds);
    Ok(G0Decision {
        alpha,
        width_weights: h,
        depths: out,
        k_star,
        delta_d_k_star,
        threshold_seconds: G0_SECONDS,
        pass: delta_d_k_star >= G0_SECONDS,
    })
}

/// Median of a non-empty sample.
pub(crate) fn median(mut samples: Vec<f64>) -> Result<f64> {
    ensure!(!samples.is_empty(), "median of an empty sample");
    samples.sort_by(f64::total_cmp);
    let mid = samples.len() / 2;
    Ok(if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        0.5 * (samples[mid - 1] + samples[mid])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn gemma_spec_acceptance_model_matches_design_arithmetic() {
        let alpha = alpha_from_l3(PEER_L3).unwrap();
        assert!(close(alpha, 0.881, 5e-4), "alpha {alpha}");
        assert!(close(acceptance(alpha, 1), 1.881, 5e-4));
        assert!(close(acceptance(alpha, 2), 2.657, 1e-3));
        assert!(close(acceptance(alpha, 3), 3.34, 1e-9));
    }

    #[test]
    fn gemma_spec_g0_width4_thresholds() {
        // Width 4 only: ΔD(1) ≥ 0.40 ⇔ c(1,4) ≤ L(1)·(1 − 0.40/D) ≈ 1.80.
        let alpha = alpha_from_l3(PEER_L3).unwrap();
        let limit = acceptance(alpha, 1) * (1.0 - G0_SECONDS / DECODE_SECONDS);
        assert!(close(limit, 1.80, 5e-3), "limit {limit}");
        let limit3 = acceptance(alpha, 3) * (1.0 - G0_SECONDS / DECODE_SECONDS);
        assert!(close(limit3, 3.19, 5e-3), "limit3 {limit3}");

        let parts = DrafterBoundParts {
            sum_t_mv: 0.0,
            t_argmax: 0.0,
            t_flash: 0.0,
            t_disp: 0.0,
        };
        let mut m = RoundCostMeasurements::default();
        m.t1.insert(4, 0.022);
        m.ts.insert((1, 4), 0.0);
        m.tr.insert((1, 4), 0.0);
        m.drafter.insert(4, parts);
        let widths: BTreeMap<usize, f64> = [(4, 9.0)].into_iter().collect();
        // Just below and just above the limit.
        for (c, pass) in [(limit - 1e-6, true), (limit + 1e-6, false)] {
            m.tv.insert((2, 4), c * 0.022);
            let d = decide_g0(&m, &widths, &[1]).unwrap();
            assert_eq!(d.k_star, 1);
            assert_eq!(d.pass, pass, "c={c} ΔD={}", d.delta_d_k_star);
        }
    }

    #[test]
    fn gemma_spec_round_cost_components() {
        let alpha = alpha_from_l3(PEER_L3).unwrap();
        let parts = DrafterBoundParts {
            sum_t_mv: 1e-3,
            t_argmax: 1e-4,
            t_flash: 5e-5,
            t_disp: 2e-6,
        };
        let td = td_bound(2, &parts);
        assert!(close(td, 1.25 * 2.0 * (1e-3 + 1e-4 + 4.0 * 5e-5 + 30.0 * 2e-6), 1e-15));
        let mut m = RoundCostMeasurements::default();
        m.t1.insert(2, 0.02);
        m.tv.insert((3, 2), 0.03);
        m.ts.insert((2, 2), 0.001);
        m.tr.insert((2, 2), 0.002);
        m.drafter.insert(2, parts);
        let (c, got_td) = round_cost(&m, alpha, 2, 2).unwrap();
        let expected = (td + 0.001 + 0.03 + (1.0 - alpha.powi(4)) * 0.002) / 0.02;
        assert!(close(c, expected, 1e-12));
        assert_eq!(got_td, td);
        assert!(round_cost(&m, alpha, 1, 2).is_err());

        // h(n) is time-weighted; ΔD sums D·h(n)·(1 − c/L).
        let ticks: BTreeMap<usize, f64> = [(1, 1.0), (2, 3.0), (3, 0.0)].into_iter().collect();
        let h = width_weights(&ticks).unwrap();
        assert_eq!(h.len(), 2);
        assert!(close(h[&1], 0.25, 1e-12) && close(h[&2], 0.75, 1e-12));
        assert_eq!(median(vec![3.0, 1.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]).unwrap(), 2.5);
    }
}
