//! Kernel-equivalence parity helpers for byte-parity refactor tests.
//!
//! Apple Silicon GPU kernels can produce ULP-level (~1e-6 to ~1e-5) FP
//! non-determinism between two structurally-equivalent paths because
//! parallel reductions inside a kernel may dispatch in different orders
//! depending on thread-group scheduling, even with identical inputs.
//! Asserting byte-identity (`a.to_bits() == b.to_bits()`) on such pairs
//! produces false-positive failures on noise that does not affect
//! observable behavior (greedy decode tokens, parity vs canonical
//! references like llama.cpp).
//!
//! The right invariant for a behavior-preserving refactor (e.g. lifting
//! an arena, extracting an `_into` variant, fusing CBs) is *kernel
//! equivalence*: cosine similarity ≥ a tight threshold AND max-abs
//! per-element diff ≤ a tight threshold. This module supplies that.
//!
//! ## Recommended thresholds
//!
//! | Use case | `cos_min` | `max_abs_max` |
//! |---|---|---|
//! | Pure FP non-determinism (re-ordered parallel reductions) | 0.9999  | 1e-4  |
//! | Quantization-codec equivalence (ADR-009 R-C2 TQ-active)  | 0.9998  | (n/a) |
//! | Looser "behavior-preserving" (fused vs unfused FA)       | 0.999   | 1e-3  |
//!
//! For most internal-kernel-vs-internal-kernel parity tests, prefer
//! the first row.
//!
//! ## Why a separate module from [`cosine_sim`](super::cosine_sim)
//!
//! `cosine_sim::cosine_similarity` returns a `Result` and rejects empty
//! vectors / NaN / zero-norm — appropriate for quality-measurement
//! pipelines where every value is meaningful and bad inputs should
//! abort. Test paths want a "report regardless" API: an all-zeros
//! parallel-contention flake should produce a meaningful report
//! (cosine = 0/0 → NaN → reported as `f64::NAN` so the assertion
//! prints something useful), not an early `Err`.

/// Report from comparing two `&[f32]` buffers element-wise.
///
/// Both vectors must be the same length; that's a precondition checked
/// in [`kernel_parity_report`]. `n_diff_bits` counts elements where
/// `a.to_bits() != b.to_bits()` (the strict byte-parity bar) — useful
/// to log alongside the tolerance numbers as a diagnostic of FP
/// non-determinism severity.
#[derive(Debug, Clone, Copy)]
pub struct KernelParityReport {
    /// Cosine similarity in `[-1.0, 1.0]`. `1.0` = identical direction.
    /// `f64::NAN` if either vector has zero L2-norm (e.g. all-zeros).
    pub cosine: f64,
    /// `max_i |a[i] - b[i]|` across all elements.
    pub max_abs_diff: f32,
    /// Index of the element with the largest abs diff (for diagnostics).
    pub max_abs_idx: usize,
    /// Element count where `a.to_bits() != b.to_bits()` (strict).
    pub n_diff_bits: usize,
    /// Total element count (== `a.len()` == `b.len()`).
    pub total: usize,
}

impl KernelParityReport {
    /// `true` if `cosine >= cos_min` AND `max_abs_diff <= max_abs_max`.
    /// Returns `false` if `cosine` is NaN (treats all-zeros / zero-norm
    /// as parity FAIL — the caller should guard with their own
    /// all-zeros parallel-contention check before relying on this).
    pub fn passes(&self, cos_min: f64, max_abs_max: f32) -> bool {
        self.cosine.is_finite()
            && self.cosine >= cos_min
            && self.max_abs_diff <= max_abs_max
    }
}

impl std::fmt::Display for KernelParityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cosine={:.6}, max_abs={:.3e} at index {}, {}/{} elements bit-different",
            self.cosine, self.max_abs_diff, self.max_abs_idx, self.n_diff_bits, self.total,
        )
    }
}

/// Compute the kernel-equivalence parity report between two F32 buffers.
///
/// `a` and `b` must have the same length; panics otherwise (the test
/// would fail anyway on shape divergence — make it loud, not silent).
pub fn kernel_parity_report(a: &[f32], b: &[f32]) -> KernelParityReport {
    assert_eq!(
        a.len(),
        b.len(),
        "kernel_parity_report: length mismatch ({} vs {})",
        a.len(),
        b.len(),
    );
    let total = a.len();

    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut max_abs_diff = 0.0_f32;
    let mut max_abs_idx = 0_usize;
    let mut n_diff_bits = 0_usize;

    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;

        let d = (x - y).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
            max_abs_idx = i;
        }
        if x.to_bits() != y.to_bits() {
            n_diff_bits += 1;
        }
    }

    let cosine = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        // One or both are zero-norm — cosine is undefined.
        f64::NAN
    };

    KernelParityReport {
        cosine,
        max_abs_diff,
        max_abs_idx,
        n_diff_bits,
        total,
    }
}

/// `max_i |a[i] - b[i]|` — exposed standalone for cases where only the
/// L∞ norm is wanted without computing cosine.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Assert kernel equivalence with structured diagnostics.
///
/// Computes the report once and panics on failure with a single
/// formatted message including cosine, max-abs-diff, total, and
/// `label`. Use in tests as the one-line replacement for `assert_eq!`
/// on `to_bits()` byte-parity loops.
///
/// ## Example
/// ```ignore
/// assert_kernel_equivalence(
///     &out_arena_path, &out_no_arena_path,
///     0.9999, 1e-4,
///     "iter83 chunk_internal_arena (out)",
/// );
/// ```
#[track_caller]
pub fn assert_kernel_equivalence(a: &[f32], b: &[f32], cos_min: f64, max_abs_max: f32, label: &str) {
    let report = kernel_parity_report(a, b);
    assert!(
        report.passes(cos_min, max_abs_max),
        "kernel equivalence FAILED for `{label}`: {report} \
         (required: cosine >= {cos_min}, max_abs_diff <= {max_abs_max:.3e})"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_perfect_parity() {
        let a = vec![1.0_f32, 2.0, -3.0, 4.5, 0.0];
        let report = kernel_parity_report(&a, &a);
        assert_eq!(report.cosine, 1.0);
        assert_eq!(report.max_abs_diff, 0.0);
        assert_eq!(report.n_diff_bits, 0);
        assert_eq!(report.total, 5);
        assert!(report.passes(0.9999, 1e-6));
    }

    #[test]
    fn ulp_level_diff_passes_strict_threshold() {
        // ULP-level FP non-determinism: ~1e-6 magnitude diffs.
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0 + 1e-6, 2.0, 3.0 - 1e-6];
        let report = kernel_parity_report(&a, &b);
        assert!(report.cosine >= 0.9999);
        assert!(report.max_abs_diff < 1e-4);
        assert!(report.passes(0.9999, 1e-4));
        // But fails strict byte-parity:
        assert!(report.n_diff_bits > 0);
    }

    #[test]
    fn observed_real_drift_at_iter83_passes_tolerance() {
        // Reproduces the iter83 failure profile: max abs diff ~3e-5,
        // 407/N elements bit-different. Should PASS the tolerance gate
        // (cosine ≥ 0.9999, max_abs ≤ 1e-4) because the magnitudes are
        // ULP-level on Apple Silicon GPU FP non-determinism.
        let n = 33700_usize;
        let a: Vec<f32> = (0..n).map(|i| ((i as f32) * 1e-3).sin()).collect();
        let mut b = a.clone();
        for j in (0..407).map(|j| j * 80) {
            if j < n {
                // Inject ~1e-5 magnitude perturbations (8-32 ULPs at the
                // value range; matches observed iter83 profile of
                // max_abs=3.052e-5).
                let bits = b[j].to_bits().wrapping_add(((j % 7 + 1) * 4) as u32);
                b[j] = f32::from_bits(bits);
            }
        }
        let report = kernel_parity_report(&a, &b);
        assert!(report.cosine >= 0.9999, "cosine {} should be ≥ 0.9999", report.cosine);
        assert!(
            report.max_abs_diff <= 1e-4,
            "max_abs_diff {:e} should be ≤ 1e-4 for ULP-class perturbations",
            report.max_abs_diff,
        );
        assert!(report.passes(0.9999, 1e-4),
            "iter83 ULP profile should pass tolerance: {report}");
    }

    #[test]
    fn structural_divergence_fails_tolerance() {
        // Reproduces iter89e2-F failure profile: 19% of elements have
        // wrapper=0 vs into≈0.05 magnitude. Tolerance check should FAIL
        // because divergence is functional, not FP-noise.
        let n = 262144_usize;
        let a: Vec<f32> = (0..n).map(|i| ((i as f32) * 1e-3).sin() * 0.5).collect();
        let b = a.clone();
        // Zero out 19% of `a_mut` to simulate "wrapper writes 0 where _into
        // writes finite":
        let mut a_mut = a.clone();
        for i in (0..n).step_by(5) {
            a_mut[i] = 0.0; // wrapper position
        }
        let report = kernel_parity_report(&a_mut, &b);
        assert!(!report.passes(0.9999, 1e-4),
            "structural divergence (19% zeros vs finite) must fail tolerance: {report}");
    }

    #[test]
    fn zero_norm_vector_returns_nan_cosine() {
        let a = vec![0.0_f32; 10];
        let b = vec![1.0_f32; 10];
        let report = kernel_parity_report(&a, &b);
        assert!(report.cosine.is_nan());
        assert!(!report.passes(0.9999, 1e-4));
    }

    #[test]
    fn max_abs_diff_standalone() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0, 2.5, 2.5];
        assert!((max_abs_diff(&a, &b) - 0.5).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn length_mismatch_panics() {
        let a = vec![1.0_f32; 10];
        let b = vec![1.0_f32; 11];
        let _ = kernel_parity_report(&a, &b);
    }
}
