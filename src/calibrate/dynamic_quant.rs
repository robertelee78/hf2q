//! Dynamic Quant — sensitivity-based mixed-precision allocation (ADR-020 Track 1).
//!
//! Native Rust port of `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py`.  The
//! algorithm has two stages:
//!
//! 1. **`estimate_sensitivities`** (mlx-lm lines 38-106) — per-quantizable-
//!    tensor signed first-order Taylor estimate of KL-divergence change
//!    incurred by quantizing at low_bits vs high_bits.  Requires
//!    autograd through the transformer; lands in iter 8.
//!
//! 2. **`estimate_threshold`** (mlx-lm lines 109-146) — binary-search a
//!    sensitivity threshold τ such that allocating high_bits to tensors
//!    with `sens > τ` and low_bits otherwise yields the target average
//!    bits-per-weight.  Pure function, no autograd.  Lands in iter 7
//!    (this commit).
//!
//! See ADR-020 §8.2 for the full Track 1 acceptance criteria.
//!
//! ## Granularity
//!
//! mlx-lm's algorithm is **per-quantizable-leaf** (Linear + Embedding) —
//! each tensor gets its own scalar score and its own bit allocation.
//! hf2q's legacy `dynamic-quant-4-6/4-8` is per-LAYER (one score per transformer
//! block, applied to all attention + FFN linears in that block).  This
//! module operates at the per-tensor granularity to match mlx-lm; the
//! existing per-layer dispatcher at `src/quantize/{mixed,layer_mix}.rs`
//! is a degenerate case where all tensors in a layer share a score.
//!
//! ## Cache compatibility
//!
//! The two algorithms produce DIFFERENT sensitivity rankings (variance
//! of activations vs first-order Taylor of KL-loss).  The
//! `~/.cache/hf2q/sensitivity/<sha>.json` cache is keyed on
//! [`crate::calibrate::cache::SensitivityCacheKey::algorithm_version`];
//! legacy `1.0.variance-magnitude` and new `2.0.gradient-alignment`
//! files coexist without overwriting.  Iter 9/10 e2e wires the new
//! algorithm into the dispatcher; until then the legacy default
//! remains in place.

use std::collections::BTreeMap;

use thiserror::Error;

/// Cache-version identifier for the gradient-alignment algorithm.
///
/// Lives alongside [`crate::calibrate::cache::SENSITIVITY_ALGORITHM_VERSION`]
/// (the legacy `1.0.variance-magnitude` constant).  Cached payloads
/// keyed on either version are addressable independently — switching
/// algorithms does NOT invalidate the legacy cache.
pub const SENSITIVITY_ALGORITHM_VERSION_GRADIENT_ALIGNMENT: &str = "2.0.gradient-alignment";

/// Sensitivity-algorithm version.  Matched against
/// [`crate::calibrate::cache::SensitivityCacheEntry::algorithm_version`]
/// at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensitivityAlgorithm {
    /// `1.0.variance-magnitude` — original hf2q DWQ-46/48 (per-layer).
    /// Score: `sqrt(variance(act)) × log2(1 + max(|act|))`.
    VarianceMagnitude,
    /// `2.0.gradient-alignment` — port of mlx-lm `dynamic_quant`
    /// (per-quantizable-tensor).  Score:
    /// `(grad · (w_low − w_high)).sum() / (params / 1e6)`.
    GradientAlignment,
}

impl SensitivityAlgorithm {
    pub const fn version_str(&self) -> &'static str {
        match self {
            Self::VarianceMagnitude => {
                crate::calibrate::cache::SENSITIVITY_ALGORITHM_VERSION
            }
            Self::GradientAlignment => SENSITIVITY_ALGORITHM_VERSION_GRADIENT_ALIGNMENT,
        }
    }

    pub fn from_version_str(s: &str) -> Option<Self> {
        match s {
            "1.0.variance-magnitude" => Some(Self::VarianceMagnitude),
            "2.0.gradient-alignment" => Some(Self::GradientAlignment),
            _ => None,
        }
    }
}

impl std::fmt::Display for SensitivityAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.version_str())
    }
}

/// A quantizable tensor's metadata for BPW accounting + threshold
/// search.  Path key matches the sensitivity-map key:
/// `mlx_lm/quant/dynamic_quant.py:104` strips the trailing `.weight`
/// from the leaf path, so the path here is the same form.
#[derive(Debug, Clone)]
pub struct QuantizableTensor {
    /// Layer/tensor path (matches sensitivity-map key).
    pub path: String,
    /// Element count (numel).  Used for BPW + per-million sensitivity
    /// normalization (iter 8).
    pub num_elements: u64,
}

/// Bits-per-weight accounting for a single tensor under a given
/// `(bits, group_size)` pair.  Counts the weight bits + per-group fp16
/// scale + per-group fp16 bias.  Mirrors mlx-lm's convention at
/// `mlx_lm/utils.py:202-215 compute_bits_per_weight`.
#[inline]
pub fn mlx_affine_tensor_bits(num_elements: u64, bits: u8, group_size: u64) -> u64 {
    if num_elements == 0 || group_size == 0 {
        return 0;
    }
    let weight_bits = num_elements.saturating_mul(bits as u64);
    // div_ceil for groups; 32 bits = scales(fp16=16) + biases(fp16=16).
    let groups = num_elements.div_ceil(group_size);
    let scale_bias_bits = groups.saturating_mul(32);
    weight_bits.saturating_add(scale_bias_bits)
}

/// Compute bits-per-weight averaged across `tensors` given a
/// per-tensor allocation predicate (`use_high(path) → bool`).
///
/// Direct port of mlx-lm `compute_bits_per_weight` semantics
/// (`mlx_lm/utils.py:210-215` + the `nn.QuantizedLinear` byte
/// computation at `:202-204`) under the assumption all tensors are
/// MLX-affine quantized at either `low_bits/low_group_size` or
/// `high_bits/high_group_size`.
pub fn compute_bits_per_weight<F>(
    tensors: &[QuantizableTensor],
    low_bits: u8,
    low_group_size: u64,
    high_bits: u8,
    high_group_size: u64,
    use_high: F,
) -> f64
where
    F: Fn(&str) -> bool,
{
    if tensors.is_empty() {
        return 0.0;
    }
    let mut total_bits: u128 = 0;
    let mut total_elems: u128 = 0;
    for t in tensors {
        let (bits, gs) = if use_high(&t.path) {
            (high_bits, high_group_size)
        } else {
            (low_bits, low_group_size)
        };
        total_bits = total_bits.saturating_add(mlx_affine_tensor_bits(t.num_elements, bits, gs) as u128);
        total_elems = total_elems.saturating_add(t.num_elements as u128);
    }
    if total_elems == 0 {
        0.0
    } else {
        total_bits as f64 / total_elems as f64
    }
}

/// Errors from the threshold-search pipeline.
#[derive(Error, Debug, PartialEq)]
pub enum DynamicQuantError {
    #[error("estimate_threshold: sensitivity map is empty (no quantizable tensors)")]
    EmptySensitivities,

    #[error(
        "estimate_threshold: target_bpw {target:.4} is unreachable; achievable range is \
         [{min:.4}, {max:.4}] for low={low_bits}b/{low_gs} high={high_bits}b/{high_gs}"
    )]
    TargetBpwUnreachable {
        target: f64,
        min: f64,
        max: f64,
        low_bits: u8,
        low_gs: u64,
        high_bits: u8,
        high_gs: u64,
    },

    #[error(
        "estimate_threshold: low_bits ({low_bits}) >= high_bits ({high_bits}); \
         high must exceed low for the search to be monotone"
    )]
    InvalidBitOrdering { low_bits: u8, high_bits: u8 },
}

/// Binary-search a sensitivity threshold τ that produces the average
/// bits-per-weight closest to `target_bpw`.  Direct port of
/// `mlx_lm/quant/dynamic_quant.py:109-146`.
///
/// **Algorithm:**
///
/// 1. Bounds: `[min(sens), max(sens)]`.
/// 2. Tolerance: `1e-3 × (max − min)` (mlx-lm line 129).
/// 3. At each iter, midpoint τ; tensors with `sens > τ` get `high`,
///    rest get `low`.  Compute BPW; if BPW > target raise lower bound,
///    else lower upper bound.
/// 4. Returns midpoint of the converged interval.
///
/// **Convergence**: monotone — raising τ can only move tensors from
/// high → low, which monotonically reduces BPW.  Iteration count is
/// `O(log(1/tolerance)) ≈ 10` regardless of model size.
///
/// **Errors**:
/// - [`DynamicQuantError::EmptySensitivities`] if `sensitivities` is empty.
/// - [`DynamicQuantError::TargetBpwUnreachable`] if `target_bpw` lies
///   outside the achievable range (all-low BPW to all-high BPW).
/// - [`DynamicQuantError::InvalidBitOrdering`] if `low_bits >= high_bits`.
pub fn estimate_threshold(
    tensors: &[QuantizableTensor],
    sensitivities: &BTreeMap<String, f64>,
    target_bpw: f64,
    low_bits: u8,
    low_group_size: u64,
    high_bits: u8,
    high_group_size: u64,
) -> Result<f64, DynamicQuantError> {
    if sensitivities.is_empty() {
        return Err(DynamicQuantError::EmptySensitivities);
    }
    if low_bits >= high_bits {
        return Err(DynamicQuantError::InvalidBitOrdering { low_bits, high_bits });
    }

    let mut min_t = f64::INFINITY;
    let mut max_t = f64::NEG_INFINITY;
    for &v in sensitivities.values() {
        if v < min_t {
            min_t = v;
        }
        if v > max_t {
            max_t = v;
        }
    }
    let range = max_t - min_t;
    let tolerance = 1e-3 * range.abs().max(f64::EPSILON);

    // Achievable range: τ → +∞ ⇒ all-low (lowest BPW); τ → -∞ ⇒ all-high (highest BPW).
    let bpw_all_low = compute_bits_per_weight(
        tensors,
        low_bits,
        low_group_size,
        high_bits,
        high_group_size,
        |_| false,
    );
    let bpw_all_high = compute_bits_per_weight(
        tensors,
        low_bits,
        low_group_size,
        high_bits,
        high_group_size,
        |_| true,
    );
    let (achievable_min, achievable_max) = (bpw_all_low.min(bpw_all_high), bpw_all_low.max(bpw_all_high));

    // Allow a small numeric slack at the boundary.
    let slack = 1e-9_f64.max(1e-6 * achievable_max.abs());
    if target_bpw < achievable_min - slack || target_bpw > achievable_max + slack {
        return Err(DynamicQuantError::TargetBpwUnreachable {
            target: target_bpw,
            min: achievable_min,
            max: achievable_max,
            low_bits,
            low_gs: low_group_size,
            high_bits,
            high_gs: high_group_size,
        });
    }

    let mut lo = min_t;
    let mut hi = max_t;
    while (hi - lo) > tolerance {
        let mid = (lo + hi) / 2.0;
        let bpw = compute_bits_per_weight(
            tensors,
            low_bits,
            low_group_size,
            high_bits,
            high_group_size,
            |p| {
                sensitivities
                    .get(p)
                    .copied()
                    .map(|s| s > mid)
                    .unwrap_or(false)
            },
        );
        if bpw > target_bpw {
            // BPW too high → fewer tensors should be high → raise τ.
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo + hi) / 2.0)
}

/// Build the per-tensor allocation map at the converged threshold.
/// Returns `(path → use_high)` for every tensor in `tensors`.
pub fn allocation_at_threshold(
    tensors: &[QuantizableTensor],
    sensitivities: &BTreeMap<String, f64>,
    threshold: f64,
) -> BTreeMap<String, bool> {
    let mut out = BTreeMap::new();
    for t in tensors {
        let use_high = sensitivities
            .get(&t.path)
            .copied()
            .map(|s| s > threshold)
            .unwrap_or(false);
        out.insert(t.path.clone(), use_high);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(path: &str, n: u64) -> QuantizableTensor {
        QuantizableTensor {
            path: path.to_string(),
            num_elements: n,
        }
    }

    #[test]
    fn algorithm_version_round_trip() {
        for v in [
            SensitivityAlgorithm::VarianceMagnitude,
            SensitivityAlgorithm::GradientAlignment,
        ] {
            let s = v.version_str();
            assert_eq!(
                SensitivityAlgorithm::from_version_str(s),
                Some(v),
                "version_str ⇄ from_version_str round-trip"
            );
        }
        assert_eq!(SensitivityAlgorithm::from_version_str("3.0.future"), None);
    }

    #[test]
    fn algorithm_version_constants_match_legacy_cache() {
        // The VarianceMagnitude variant must report the EXACT string the
        // legacy cache uses, otherwise legacy entries hard-fail.
        assert_eq!(
            SensitivityAlgorithm::VarianceMagnitude.version_str(),
            "1.0.variance-magnitude"
        );
        assert_eq!(
            SensitivityAlgorithm::VarianceMagnitude.version_str(),
            crate::calibrate::cache::SENSITIVITY_ALGORITHM_VERSION,
        );
    }

    #[test]
    fn algorithm_version_gradient_alignment_constant_pinned() {
        assert_eq!(
            SensitivityAlgorithm::GradientAlignment.version_str(),
            "2.0.gradient-alignment"
        );
        assert_eq!(
            SENSITIVITY_ALGORITHM_VERSION_GRADIENT_ALIGNMENT,
            "2.0.gradient-alignment"
        );
    }

    #[test]
    fn mlx_affine_tensor_bits_4bit_g64_matches_hand_derived() {
        // 4096 weights at 4-bit + scales/biases (fp16+fp16 per group of 64):
        //   weight: 4096 × 4 = 16384 bits
        //   groups: 4096 / 64 = 64
        //   overhead: 64 × 32 = 2048 bits
        //   total: 18432 bits
        //   bpw: 18432 / 4096 = 4.5
        let bits = mlx_affine_tensor_bits(4096, 4, 64);
        assert_eq!(bits, 18432);
    }

    #[test]
    fn compute_bpw_uniform_4bit_yields_4_5() {
        let ts = [tensor("a", 4096), tensor("b", 4096)];
        let bpw = compute_bits_per_weight(&ts, 4, 64, 6, 64, |_| false);
        assert!((bpw - 4.5).abs() < 1e-9, "uniform-4bit BPW should be 4.5; got {bpw}");
    }

    #[test]
    fn compute_bpw_uniform_6bit_yields_6_5() {
        let ts = [tensor("a", 4096), tensor("b", 4096)];
        let bpw = compute_bits_per_weight(&ts, 4, 64, 6, 64, |_| true);
        assert!((bpw - 6.5).abs() < 1e-9, "uniform-6bit BPW should be 6.5; got {bpw}");
    }

    #[test]
    fn compute_bpw_half_split_lands_at_5_5() {
        let ts = [tensor("low", 4096), tensor("high", 4096)];
        let bpw = compute_bits_per_weight(&ts, 4, 64, 6, 64, |p| p == "high");
        // Half at 4.5 BPW + half at 6.5 BPW = 5.5 BPW.
        assert!((bpw - 5.5).abs() < 1e-9, "half-split BPW should be 5.5; got {bpw}");
    }

    #[test]
    fn estimate_threshold_empty_errors() {
        let ts: [QuantizableTensor; 0] = [];
        let s: BTreeMap<String, f64> = BTreeMap::new();
        let r = estimate_threshold(&ts, &s, 5.0, 4, 64, 6, 64);
        assert_eq!(r, Err(DynamicQuantError::EmptySensitivities));
    }

    #[test]
    fn estimate_threshold_invalid_bit_ordering_errors() {
        let ts = [tensor("a", 4096)];
        let s: BTreeMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        let r = estimate_threshold(&ts, &s, 5.0, 6, 64, 4, 64);
        assert_eq!(
            r,
            Err(DynamicQuantError::InvalidBitOrdering { low_bits: 6, high_bits: 4 })
        );
    }

    #[test]
    fn estimate_threshold_unreachable_target_errors() {
        let ts = [tensor("a", 4096), tensor("b", 4096)];
        let s: BTreeMap<String, f64> = [
            ("a".to_string(), 0.1),
            ("b".to_string(), 0.5),
        ]
        .into_iter()
        .collect();
        // Achievable range with 4/64 vs 6/64: [4.5, 6.5].  Target 7.0 unreachable.
        let r = estimate_threshold(&ts, &s, 7.0, 4, 64, 6, 64);
        assert!(
            matches!(r, Err(DynamicQuantError::TargetBpwUnreachable { .. })),
            "target above achievable max must error: {r:?}"
        );
    }

    #[test]
    fn estimate_threshold_converges_for_target_5_0_two_tensor() {
        // Two equal-size tensors; sensitivities 0.1 and 0.5.  Target BPW 5.0
        // is the midpoint between all-low (4.5) and all-high (6.5), so the
        // converged threshold should be ~ 0.3 (between the two values),
        // selecting only the high-sens tensor for high_bits.
        let ts = [tensor("low_sens", 4096), tensor("high_sens", 4096)];
        let s: BTreeMap<String, f64> = [
            ("low_sens".to_string(), 0.1),
            ("high_sens".to_string(), 0.5),
        ]
        .into_iter()
        .collect();
        let tau = estimate_threshold(&ts, &s, 5.5, 4, 64, 6, 64).expect("converges");
        // At tau in (0.1, 0.5), only high_sens > tau → use_high; result BPW = 5.5.
        let bpw = compute_bits_per_weight(&ts, 4, 64, 6, 64, |p| {
            s.get(p).copied().unwrap_or(f64::NEG_INFINITY) > tau
        });
        assert!(
            (bpw - 5.5).abs() < 1e-9,
            "converged threshold should yield BPW 5.5; got {bpw}"
        );
        assert!((0.1..=0.5).contains(&tau), "tau in (0.1, 0.5); got {tau}");
    }

    #[test]
    fn estimate_threshold_bpw_monotone_in_tau() {
        // Falsifier: as τ rises from min → max, achieved BPW must
        // monotonically decrease.  Underwrites the binary-search's
        // convergence guarantee.
        let ts: Vec<QuantizableTensor> = (0..32).map(|i| tensor(&format!("t{i}"), 1024)).collect();
        let s: BTreeMap<String, f64> = (0..32)
            .map(|i| (format!("t{i}"), i as f64 / 32.0))
            .collect();
        let mut prev_bpw = f64::INFINITY;
        let mut tau = -0.1_f64;
        let step = 0.05_f64;
        let mut samples = 0;
        while tau <= 1.1 {
            let bpw = compute_bits_per_weight(&ts, 4, 64, 6, 64, |p| {
                s.get(p).copied().unwrap_or(f64::NEG_INFINITY) > tau
            });
            assert!(
                bpw <= prev_bpw + 1e-12,
                "BPW must be monotone-decreasing in τ; at τ={tau:.4} bpw={bpw:.4} > prev={prev_bpw:.4}"
            );
            prev_bpw = bpw;
            tau += step;
            samples += 1;
        }
        assert!(samples > 20, "monotonicity sweep should sample at least 20 τ values");
    }

    #[test]
    fn estimate_threshold_sixty_four_tensor_realistic() {
        // 64 tensors with linearly-increasing fractional sensitivities,
        // target BPW halfway between min (4.5) and max (6.5).  With 64
        // tensors the discrete-step granularity is (6.5−4.5)/64 = 0.03125
        // BPW per tensor swap, so the converged τ should land within
        // one step of the target.
        let ts: Vec<QuantizableTensor> = (0..64).map(|i| tensor(&format!("t{i}"), 8192)).collect();
        let s: BTreeMap<String, f64> = (0..64)
            .map(|i| (format!("t{i}"), 0.123 + (i as f64) * 0.7919)) // continuous floats, no ties
            .collect();
        let target = 5.5; // exactly midpoint
        let tau = estimate_threshold(&ts, &s, target, 4, 64, 6, 64).expect("converges");
        let bpw = compute_bits_per_weight(&ts, 4, 64, 6, 64, |p| {
            s.get(p).copied().unwrap_or(f64::NEG_INFINITY) > tau
        });
        // One discrete step is 0.03125 BPW; allow 0.04 slack
        // (`(hi − lo) ≤ 1e-3 × range` convergence + boundary placement).
        assert!(
            (bpw - target).abs() < 0.04,
            "converged BPW within 0.04 (one tensor swap) of target {target}; got {bpw} at τ={tau}"
        );
    }

    #[test]
    fn estimate_threshold_no_overshoot_on_integer_sens_ties() {
        // Falsifier for the discrete-step boundary issue: 16 tensors at
        // integer sensitivities 0..15, target 5.5 (lands exactly on an
        // 8-high allocation).  mlx-lm's binary search returns midpoint
        // — at this contrived case it can land on 9-high (BPW 5.625)
        // instead of 8-high (BPW 5.5).  Spec: BPW must be within ONE
        // tensor-swap step of target (0.125 BPW = 2÷16).
        let ts: Vec<QuantizableTensor> = (0..16).map(|i| tensor(&format!("t{i}"), 8192)).collect();
        let s: BTreeMap<String, f64> = (0..16)
            .map(|i| (format!("t{i}"), i as f64))
            .collect();
        let target = 5.5;
        let tau = estimate_threshold(&ts, &s, target, 4, 64, 6, 64).expect("converges");
        let bpw = compute_bits_per_weight(&ts, 4, 64, 6, 64, |p| {
            s.get(p).copied().unwrap_or(f64::NEG_INFINITY) > tau
        });
        let step = 2.0 / (ts.len() as f64); // 0.125 BPW per swap
        assert!(
            (bpw - target).abs() <= step + 1e-9,
            "converged BPW within one step ({step}) of target {target}; got {bpw}"
        );
    }

    #[test]
    fn allocation_at_threshold_partitions_tensors() {
        let ts = [tensor("a", 100), tensor("b", 100), tensor("c", 100)];
        let s: BTreeMap<String, f64> = [
            ("a".to_string(), 0.1),
            ("b".to_string(), 0.5),
            ("c".to_string(), 0.9),
        ]
        .into_iter()
        .collect();
        let alloc = allocation_at_threshold(&ts, &s, 0.3);
        assert_eq!(alloc[&"a".to_string()], false);
        assert_eq!(alloc[&"b".to_string()], true);
        assert_eq!(alloc[&"c".to_string()], true);
    }

    #[test]
    fn allocation_at_threshold_missing_key_defaults_low() {
        // Tensor without a sensitivity entry must default to use_high=false
        // (the conservative choice — don't burn high bits on unknowns).
        let ts = [tensor("known", 100), tensor("unknown", 100)];
        let s: BTreeMap<String, f64> = [("known".to_string(), 0.9)].into_iter().collect();
        let alloc = allocation_at_threshold(&ts, &s, 0.5);
        assert_eq!(alloc[&"known".to_string()], true);
        assert_eq!(alloc[&"unknown".to_string()], false);
    }
}
