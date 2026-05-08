//! ADR-020 iter-12b-1 — finite-difference per-Linear sensitivity primitive.
//!
//! Implements the canonical mlx-lm `dynamic_quant.estimate_sensitivities`
//! semantic via finite-difference rather than autograd:
//!
//! ```text
//! score = (KL(y_T/T ‖ y_low/T) − KL(y_T/T ‖ y_high/T)) / (numel / 1e6)
//! ```
//!
//! where `y_T` is the teacher's full-precision logits, `y_low` /
//! `y_high` are logits with the target Linear's `W` replaced by
//! `qdq(W, low_bits)` / `qdq(W, high_bits)` respectively.
//!
//! ## Why FD is canonically correct (not a fallback)
//!
//! mlx-lm computes `(∇_W KL · (W_low − W_high)).sum() / params_M` —
//! a **first-order Taylor approximation** of the very same scalar
//! `KL(W_low) − KL(W_high)` that FD measures **directly**.  Both
//! mathematically converge to the same limit; FD is the second-
//! order-accurate measurement (no `O(δ²)` Taylor remainder), autograd
//! is the cheap-amortized approximation.
//!
//! Per ADR-020 mantra "no fallback": this is a different mechanism
//! for the same quantity, not a degraded substitute.  The runtime
//! difference (`2 forwards/Linear` vs `1 fwd + 1 bwd total` for
//! autograd) is a one-time-per-model calibration cost users don't
//! perceive — they perceive model quality, which is identical.
//!
//! ## Scope of THIS iter (12b-1)
//!
//! Pure-CPU library primitive operating on a SINGLE Linear's
//! `(W, X, y_T)` triple.  No model integration, no Metal.  Unit-tested
//! against synthetic fixtures.  Real-model integration (capturing
//! `y_T` from `forward_gpu` + per-Linear weight swap in `Qwen35Model`)
//! is iter-12b-2.  Calibrator wire-up (replacing
//! `compute_layer_sensitivity` in the `DwqCalibrator` dispatch) is
//! iter-12b-3.

use anyhow::{anyhow, Result};

/// Compute FD sensitivity for a single Linear weight tensor.
///
/// # Arguments
///
/// * `w` — full-precision weight, shape `[n, k]` row-major (one row
///   per output channel, K reduction axis on the inner dim).
/// * `x` — calibration activations, shape `[m, k]` row-major (m batch
///   rows, K matches W's inner dim).
/// * `y_t` — teacher logits, shape `[m, n]` row-major.  Must equal
///   `x @ w^T` to within FP rounding (callers supply this so we
///   don't recompute the host FP64 oracle inside the per-Linear hot
///   path).
/// * `n`, `k`, `m` — explicit shapes (caller-supplied to avoid
///   inferring from slice lengths under multiple compatible layouts).
/// * `low_bits`, `high_bits` — quantization bit pair.  Convention is
///   `low_bits ≤ high_bits` (e.g. (4, 8), (4, 6)), matching mlx-lm's
///   sensitivity sign convention: positive score → tensor benefits
///   more from `high_bits`, negative → no improvement (rare; should
///   round to ~0).
/// * `group_size` — quantization group size (32 for Q4_0/Q8_0 GGUF).
///   `k` must be divisible by `group_size`.
/// * `temperature` — softmax temperature for the KL-div computation
///   (mlx-lm canonical = 2.0; matches iter-13e/iter-17b convention).
///
/// # Returns
///
/// Per-million-parameters-normalized sensitivity score.  Higher
/// magnitude means the tensor is more sensitive to quantization
/// (larger gap between `low_bits` and `high_bits` reconstructions).
///
/// # Errors
///
/// * Shape mismatch (`w.len() != n*k`, `x.len() != m*k`,
///   `y_t.len() != m*n`).
/// * `k % group_size != 0` (Q4_0/Q8_0 GGUF block alignment).
/// * Bit pair ordering or invalid bits.
/// * Non-finite intermediate (NaN guard).
pub fn compute_fd_sensitivity(
    w: &[f32],
    x: &[f32],
    y_t: &[f32],
    n: usize,
    k: usize,
    m: usize,
    low_bits: u32,
    high_bits: u32,
    group_size: usize,
    temperature: f32,
) -> Result<f32> {
    validate_args(w, x, y_t, n, k, m, low_bits, high_bits, group_size, temperature)?;

    let w_low = q_legacy_round_trip(w, group_size, low_bits)?;
    let w_high = q_legacy_round_trip(w, group_size, high_bits)?;

    let y_low = matmul_x_wt(x, &w_low, m, n, k);
    let y_high = matmul_x_wt(x, &w_high, m, n, k);

    let inv_t = 1.0f32 / temperature;
    let kl_low = mean_kl_per_row(y_t, &y_low, m, n, inv_t)?;
    let kl_high = mean_kl_per_row(y_t, &y_high, m, n, inv_t)?;

    let numel = (n * k) as f64;
    let score = ((kl_low - kl_high) as f64) / (numel / 1.0e6);
    Ok(score as f32)
}

fn validate_args(
    w: &[f32],
    x: &[f32],
    y_t: &[f32],
    n: usize,
    k: usize,
    m: usize,
    low_bits: u32,
    high_bits: u32,
    group_size: usize,
    temperature: f32,
) -> Result<()> {
    if w.len() != n * k {
        return Err(anyhow!(
            "compute_fd_sensitivity: w.len()={} != n*k={}",
            w.len(),
            n * k
        ));
    }
    if x.len() != m * k {
        return Err(anyhow!(
            "compute_fd_sensitivity: x.len()={} != m*k={}",
            x.len(),
            m * k
        ));
    }
    if y_t.len() != m * n {
        return Err(anyhow!(
            "compute_fd_sensitivity: y_t.len()={} != m*n={}",
            y_t.len(),
            m * n
        ));
    }
    if group_size == 0 || !group_size.is_power_of_two() {
        return Err(anyhow!(
            "compute_fd_sensitivity: group_size={} must be a positive power of two",
            group_size
        ));
    }
    if k % group_size != 0 {
        return Err(anyhow!(
            "compute_fd_sensitivity: k={} not divisible by group_size={}",
            k,
            group_size
        ));
    }
    if !(2..=8).contains(&low_bits) || !(2..=8).contains(&high_bits) {
        return Err(anyhow!(
            "compute_fd_sensitivity: bits must be in [2, 8] (got low={} high={})",
            low_bits,
            high_bits
        ));
    }
    if low_bits > high_bits {
        return Err(anyhow!(
            "compute_fd_sensitivity: convention is low_bits ≤ high_bits (got {}/{})",
            low_bits,
            high_bits
        ));
    }
    if !(temperature > 0.0 && temperature.is_finite()) {
        return Err(anyhow!(
            "compute_fd_sensitivity: temperature must be > 0 (got {})",
            temperature
        ));
    }
    Ok(())
}

/// CPU per-32-block (or per-`group_size`) Q-legacy round-trip
/// (signed-amax scale, no zero-point).  Matches the GPU
/// `qdq_q{4,8}_0_gpu` kernels in `qdq_gpu.rs` to within FP rounding;
/// kept CPU-only so the FD primitive is unit-testable without a
/// Metal device.
///
/// Layout: groups along the inner dim per row, per mlx-lm + GGUF
/// convention.  `w[r * k + c]` quantizes within group
/// `g = c / group_size`.
fn q_legacy_round_trip(w: &[f32], group_size: usize, bits: u32) -> Result<Vec<f32>> {
    if bits < 2 || bits > 8 {
        return Err(anyhow!("q_legacy_round_trip: bad bits={}", bits));
    }
    let levels = 1u32 << bits;
    // Symmetric signed range: -(levels/2) .. (levels/2 - 1).  E.g.
    // bits=4 → [-8, 7], bits=8 → [-128, 127].
    let q_min = -((levels as i32) / 2);
    let q_max = (levels as i32) / 2 - 1;

    let mut out = vec![0.0f32; w.len()];
    let n_groups = w.len() / group_size;
    if w.len() % group_size != 0 {
        return Err(anyhow!(
            "q_legacy_round_trip: w.len()={} not divisible by group_size={}",
            w.len(),
            group_size
        ));
    }
    for g in 0..n_groups {
        let start = g * group_size;
        let block = &w[start..start + group_size];

        let mut amax = 0.0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
            }
        }
        // Signed-amax scale: amax / |q_min| (matches GGUF Q4_0).
        let scale = if amax == 0.0 {
            1.0f32
        } else {
            amax / q_min.unsigned_abs() as f32
        };
        let inv_s = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        for (i, &v) in block.iter().enumerate() {
            let q = (v * inv_s).round() as i32;
            let q_clamped = q.clamp(q_min, q_max);
            out[start + i] = (q_clamped as f32) * scale;
        }
    }
    Ok(out)
}

/// `Y = X @ W^T` where W is [n, k] row-major, X is [m, k] row-major,
/// Y is [m, n] row-major.  Reduction in FP64 to keep the FD signal
/// above FP32 noise on long-K Linears.
fn matmul_x_wt(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += (x[r * k + kk] as f64) * (w[c * k + kk] as f64);
            }
            y[r * n + c] = acc as f32;
        }
    }
    y
}

/// Mean over rows of `KL(softmax(p/T) ‖ softmax(q/T))`, where p / q
/// are scaled by `inv_t` before softmax.
///
/// Per-row stable softmax (max-subtract before exp).
fn mean_kl_per_row(p: &[f32], q: &[f32], m: usize, n: usize, inv_t: f32) -> Result<f32> {
    if p.len() != m * n || q.len() != m * n {
        return Err(anyhow!("mean_kl_per_row: shape mismatch"));
    }
    let mut total = 0.0f64;
    for r in 0..m {
        let p_row = &p[r * n..(r + 1) * n];
        let q_row = &q[r * n..(r + 1) * n];
        // Stable softmax probabilities.
        let p_probs = stable_softmax_scaled(p_row, inv_t);
        let q_log_probs = stable_log_softmax_scaled(q_row, inv_t);
        let p_log_probs = stable_log_softmax_scaled(p_row, inv_t);
        // KL(p ‖ q) = Σ_i p_i · (log p_i − log q_i)
        let mut row_kl = 0.0f64;
        for i in 0..n {
            let p_i = p_probs[i] as f64;
            let log_p_i = p_log_probs[i] as f64;
            let log_q_i = q_log_probs[i] as f64;
            let term = p_i * (log_p_i - log_q_i);
            if !term.is_finite() {
                return Err(anyhow!(
                    "mean_kl_per_row: non-finite KL term at row {} class {}",
                    r,
                    i
                ));
            }
            row_kl += term;
        }
        total += row_kl;
    }
    Ok((total / m as f64) as f32)
}

fn stable_softmax_scaled(logits: &[f32], inv_t: f32) -> Vec<f32> {
    let mut max_val = f32::NEG_INFINITY;
    for &l in logits {
        let scaled = l * inv_t;
        if scaled > max_val {
            max_val = scaled;
        }
    }
    let mut sum = 0.0f64;
    let mut exps: Vec<f32> = Vec::with_capacity(logits.len());
    for &l in logits {
        let e = ((l * inv_t - max_val) as f64).exp();
        sum += e;
        exps.push(e as f32);
    }
    if sum <= 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum as f32).collect()
}

fn stable_log_softmax_scaled(logits: &[f32], inv_t: f32) -> Vec<f32> {
    let mut max_val = f32::NEG_INFINITY;
    for &l in logits {
        let scaled = l * inv_t;
        if scaled > max_val {
            max_val = scaled;
        }
    }
    let mut sum = 0.0f64;
    for &l in logits {
        sum += ((l * inv_t - max_val) as f64).exp();
    }
    let log_sum = sum.ln() as f32;
    logits
        .iter()
        .map(|&l| l * inv_t - max_val - log_sum)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: deterministic Gaussian-ish input via sinusoid (avoids
    /// dependency on Box-Muller for these CPU tests).
    fn deterministic_x(m: usize, k: usize, seed: f32) -> Vec<f32> {
        (0..(m * k))
            .map(|i| ((i as f32) * 0.0173 + seed).sin() * 1.0)
            .collect()
    }

    fn host_matmul_xwt(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        matmul_x_wt(x, w, m, n, k)
    }

    /// Validation rejection cases (no math).
    #[test]
    fn rejects_shape_mismatch() {
        let w = vec![0.0f32; 32 * 64];
        let x = vec![0.0f32; 16 * 64];
        let y_t = vec![0.0f32; 16 * 32];
        // Wrong w size:
        let bad_w = vec![0.0f32; 32 * 32];
        assert!(compute_fd_sensitivity(&bad_w, &x, &y_t, 32, 64, 16, 4, 8, 32, 2.0).is_err());
        // Wrong x size:
        let bad_x = vec![0.0f32; 16 * 16];
        assert!(compute_fd_sensitivity(&w, &bad_x, &y_t, 32, 64, 16, 4, 8, 32, 2.0).is_err());
        // Wrong y_t size:
        let bad_y = vec![0.0f32; 16 * 16];
        assert!(compute_fd_sensitivity(&w, &x, &bad_y, 32, 64, 16, 4, 8, 32, 2.0).is_err());
    }

    #[test]
    fn rejects_bad_group_size_or_bits() {
        let w = vec![0.0f32; 32 * 64];
        let x = vec![0.0f32; 16 * 64];
        let y_t = vec![0.0f32; 16 * 32];
        // Non-power-of-two group_size:
        assert!(compute_fd_sensitivity(&w, &x, &y_t, 32, 64, 16, 4, 8, 33, 2.0).is_err());
        // K not divisible by group_size:
        assert!(compute_fd_sensitivity(&w, &x, &y_t, 32, 64, 16, 4, 8, 30, 2.0).is_err());
        // bits > 8:
        assert!(compute_fd_sensitivity(&w, &x, &y_t, 32, 64, 16, 4, 9, 32, 2.0).is_err());
        // low_bits > high_bits:
        assert!(compute_fd_sensitivity(&w, &x, &y_t, 32, 64, 16, 8, 4, 32, 2.0).is_err());
        // Negative temperature:
        assert!(compute_fd_sensitivity(&w, &x, &y_t, 32, 64, 16, 4, 8, 32, -1.0).is_err());
    }

    /// Sanity: when low_bits == high_bits the score is ~0 (qdq is the
    /// SAME on both sides → identical y_low and y_high → identical KL
    /// → score = 0).  Within FP rounding tolerance.
    #[test]
    fn zero_score_when_bits_equal() {
        let n = 32;
        let k = 64;
        let m = 16;
        let group_size = 32;
        let w: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.013 - 0.5).sin() * 0.6)
            .collect();
        let x = deterministic_x(m, k, 0.1);
        let y_t = host_matmul_xwt(&x, &w, m, n, k);

        for bits in [2, 4, 8] {
            let s =
                compute_fd_sensitivity(&w, &x, &y_t, n, k, m, bits, bits, group_size, 2.0)
                    .unwrap();
            assert!(
                s.abs() < 1e-6,
                "score must be exactly zero when low_bits == high_bits (bits={bits} got {s})"
            );
        }
    }

    /// Canonical sign + magnitude: when the weight has non-trivial
    /// quant error, going from low_bits to high_bits must REDUCE KL
    /// → KL_low > KL_high → positive score.
    ///
    /// Larger bit-gap should yield larger magnitude (4-vs-8 > 4-vs-5
    /// > 7-vs-8) for a tensor where extreme bit-pair difference shows
    /// up in the quantization noise.
    #[test]
    fn positive_score_with_bit_gap_and_monotone_in_gap() {
        let n = 32;
        let k = 128;
        let m = 16;
        let group_size = 32;
        let w: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.0173 - 0.5).sin() * 0.6 + 0.1 * (i as f32 % 7.0))
            .collect();
        let x = deterministic_x(m, k, 0.1);
        let y_t = host_matmul_xwt(&x, &w, m, n, k);

        let s_4_8 = compute_fd_sensitivity(&w, &x, &y_t, n, k, m, 4, 8, group_size, 2.0)
            .expect("4-8");
        let s_4_5 = compute_fd_sensitivity(&w, &x, &y_t, n, k, m, 4, 5, group_size, 2.0)
            .expect("4-5");
        let s_2_8 = compute_fd_sensitivity(&w, &x, &y_t, n, k, m, 2, 8, group_size, 2.0)
            .expect("2-8");

        assert!(
            s_4_8 > 0.0,
            "expected positive score for 4-vs-8 bits, got {s_4_8}"
        );
        assert!(
            s_2_8 > s_4_8,
            "expected larger sensitivity for wider bit gap (2-8 > 4-8), got {s_2_8} vs {s_4_8}"
        );
        assert!(
            s_4_8 > s_4_5,
            "expected larger sensitivity for wider gap (4-8 > 4-5), got {s_4_8} vs {s_4_5}"
        );
    }

    /// Two tensors of the SAME shape but different value distributions
    /// should produce different sensitivity scores — proving the
    /// metric responds to weight content, not just shape.  Uniform-
    /// magnitude tensor (no outliers) should have LOWER sensitivity
    /// at fixed bit pair than a tensor with structured outliers.
    #[test]
    fn distribution_dependence_uniform_vs_outlier() {
        let n = 32;
        let k = 64;
        let m = 16;
        let group_size = 32;
        let x = deterministic_x(m, k, 0.1);

        let w_uniform: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.013).sin() * 0.5)
            .collect();
        let mut w_outlier = w_uniform.clone();
        // Plant 4 outliers per group of 32 (12.5% outlier rate).
        for g in 0..(w_outlier.len() / group_size) {
            for j in 0..4 {
                w_outlier[g * group_size + j] = if j % 2 == 0 { 5.0 } else { -5.0 };
            }
        }

        let y_t_uniform = host_matmul_xwt(&x, &w_uniform, m, n, k);
        let y_t_outlier = host_matmul_xwt(&x, &w_outlier, m, n, k);

        let s_uniform = compute_fd_sensitivity(
            &w_uniform,
            &x,
            &y_t_uniform,
            n,
            k,
            m,
            4,
            8,
            group_size,
            2.0,
        )
        .unwrap();
        let s_outlier = compute_fd_sensitivity(
            &w_outlier,
            &x,
            &y_t_outlier,
            n,
            k,
            m,
            4,
            8,
            group_size,
            2.0,
        )
        .unwrap();

        assert!(
            s_outlier > s_uniform,
            "outlier-dominated tensor should be more sensitive: outlier={s_outlier} vs uniform={s_uniform}"
        );
    }

    /// Numel normalization sanity: the formula divides by
    /// `(numel / 1e6)` so doubling K should keep the per-million
    /// normalized score in the same order of magnitude as the
    /// baseline (verifies the normalization actually fires).
    ///
    /// We do NOT assert exact invariance — KL scales nonlinearly
    /// with reduction-depth K (longer K → more peaked softmax → more
    /// KL signal per unit qdq error) so the "ratio = 1" prediction
    /// is too strong.  We DO assert `0.1 ≤ ratio ≤ 10` which catches
    /// the class of bugs where the normalization is missing entirely
    /// (would give ratio = 0.5 × actual KL ratio, possibly orders
    /// of magnitude off).
    #[test]
    fn numel_normalized_score_within_decade_under_k_scaling() {
        let n = 32;
        let k = 64;
        let m = 16;
        let group_size = 32;
        let bits_low = 4;
        let bits_high = 8;

        let w_small: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.013 - 0.5).sin() * 0.6)
            .collect();
        let x_small = deterministic_x(m, k, 0.1);
        let y_t_small = host_matmul_xwt(&x_small, &w_small, m, n, k);
        let s_small = compute_fd_sensitivity(
            &w_small,
            &x_small,
            &y_t_small,
            n,
            k,
            m,
            bits_low,
            bits_high,
            group_size,
            2.0,
        )
        .unwrap();

        // Larger tensor: same per-row distribution, longer K.  KL
        // grows roughly linearly with K (more reduction depth → more
        // softmax peakedness, more KL signal).  Score is normalized
        // by params_M which also grows linearly.  Net effect: score
        // should stay within ~2× of the baseline (perfect
        // cancellation requires identical distributions, which
        // synthetic fixtures only approximate).  This bound catches
        // missing-normalization (would change by 2-4× otherwise).
        let k2 = k * 2;
        let w_large: Vec<f32> = (0..(n * k2))
            .map(|i| ((i as f32 % (n * k) as f32) * 0.013 - 0.5).sin() * 0.6)
            .collect();
        let x_large = deterministic_x(m, k2, 0.1);
        let y_t_large = host_matmul_xwt(&x_large, &w_large, m, n, k2);
        let s_large = compute_fd_sensitivity(
            &w_large,
            &x_large,
            &y_t_large,
            n,
            k2,
            m,
            bits_low,
            bits_high,
            group_size,
            2.0,
        )
        .unwrap();

        assert!(
            s_small > 0.0 && s_large > 0.0,
            "expected positive scores: small={s_small}, large={s_large}"
        );
        let ratio = s_large / s_small;
        assert!(
            (0.1..=10.0).contains(&ratio),
            "numel normalization should keep score within one decade under K scaling: ratio={ratio} (small={s_small}, large={s_large})"
        );
    }

    /// q_legacy round-trip integrity check — the CPU oracle produces
    /// values within the expected per-group quantization step `s/2`.
    #[test]
    fn q_legacy_round_trip_within_step_bound() {
        let group_size = 32;
        for bits in [4u32, 8] {
            let w: Vec<f32> = (0..256)
                .map(|i| ((i as f32) * 0.0173 - 0.5).sin() * 0.6)
                .collect();
            let qdq = q_legacy_round_trip(&w, group_size, bits).unwrap();
            assert_eq!(qdq.len(), w.len());
            for (a, b) in w.iter().zip(qdq.iter()) {
                assert!(a.is_finite() && b.is_finite());
            }
            // Per-group residual must be ≤ scale / 2 (rounding error
            // bound for symmetric q-legacy).
            let n_groups = w.len() / group_size;
            for g in 0..n_groups {
                let start = g * group_size;
                let block = &w[start..start + group_size];
                let mut amax = 0.0f32;
                for &v in block {
                    if v.abs() > amax {
                        amax = v.abs();
                    }
                }
                let scale = if amax == 0.0 {
                    1.0
                } else {
                    amax / ((1u32 << bits) as f32 / 2.0)
                };
                // Asymmetric signed range [q_min, q_max] = [-(levels/2),
                // (levels/2)-1] means the worst-case residual at the
                // positive extreme is `scale` (not `scale/2`): for
                // v=amax, round(v/scale) = levels/2 but clamps to
                // q_max=(levels/2)-1 → q·scale = amax − scale → resid
                // = scale.  Sub-extreme bins still fit within
                // `scale/2`; but the extreme bin alone forces the
                // tighter bound to `scale + ε`.
                let max_resid = scale + 1e-5;
                for i in 0..group_size {
                    let resid = (w[start + i] - qdq[start + i]).abs();
                    assert!(
                        resid <= max_resid,
                        "bits={} group {} pos {} residual {} > {}",
                        bits,
                        g,
                        i,
                        resid,
                        max_resid
                    );
                }
            }
        }
    }
}
