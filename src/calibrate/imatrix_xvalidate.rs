//! ADR-014 P6 close iter-1 — cross-validation of two GGUF v3 imatrix
//! files (PR #9400 schema).
//!
//! ## Why this module exists
//!
//! [`super::imatrix`] ports llama.cpp's `GGML_OP_MUL_MAT(_ID)` accumulator
//! and writes/reads the GGUF v3 imatrix format from llama.cpp PR #9400
//! (commit `90083283`, 2025-07-19). The remaining close gate for ADR-014
//! P6 is **byte-equivalent (with documented float tolerance)
//! cross-validation against the `llama-imatrix` binary on the same
//! model + same corpus** — proving the pure-Rust port produces
//! per-tensor `in_sum2` + `counts` numerically equivalent to llama.cpp's
//! C++ implementation.
//!
//! This iter (P6 close iter-1) lands the **comparator machinery** —
//! [`cross_validate_imatrix_gguf`] + [`XValidationReport`] +
//! [`TensorComparison`] + 5 always-on tests + 1 `#[ignore]`-gated
//! real-model cell that iter-2 wires up with a real Qwen3.5-0.6B
//! fixture and the live `llama-imatrix` subprocess (the wrapper
//! [`crate::calibrate`]'s integration test crates can already invoke
//! via `tests/common/llama_cpp_runner::run_llama_imatrix`).
//!
//! ## Comparator contract
//!
//! Given two `.imatrix.gguf` paths `a` and `b`, the comparator:
//!
//! 1. Loads both via [`super::imatrix::ImatrixCollector::load_imatrix_gguf`]
//!    — the same code path the runtime quantize loop consumes. This
//!    iter introduces **no new GGUF reader**; the gate must live or die
//!    on the loader the production code actually uses.
//! 2. Computes the symmetric tensor-name set difference. Tensors
//!    present in only one side are surfaced verbatim
//!    ([`XValidationReport::tensors_in_a_only`] and `_b_only`).
//! 3. For every tensor present in both sides:
//!    - Validates that `values.len()` and `counts.len()` agree (shape
//!      mismatch → [`XValidationError::Invariant`]).
//!    - Computes element-wise `max(abs(a - b))` over `in_sum2` (the
//!      `Stats::values` field).
//!    - Computes element-wise `max(abs(a - b) / max(abs(a), abs(b),
//!      epsilon))` over `in_sum2` for the relative-tolerance branch.
//!    - Asserts `counts` arrays are byte-equal (counts are integer
//!      token-counts and llama.cpp stores them as F32 by GGUF spec, but
//!      the values are exact integers and must match to the bit — any
//!      mismatch is a real-error semantic, never a precision artefact).
//! 4. Builds an [`XValidationReport`] with per-tensor [`TensorComparison`]
//!    rows + the gate-decision predicate [`XValidationReport::is_pass`].
//!
//! ## Tolerance defaults
//!
//! `abs_tolerance = 1e-3, rel_tolerance = 1e-2`. Justification (P7
//! iter-3x/3y dequant round-trip RMSE bounds):
//! - Q4_K round-trip RMSE ≤ 0.05 — comfortably bounds float-precision
//!   drift in `in_sum2` accumulation (which accumulates `x[col]² /
//!   n_tokens` per chunk; per-chunk f32 sums are well within 1e-3 abs
//!   on activations of order 1e0–1e1).
//! - Q5_K round-trip RMSE ≤ 0.025; Q6_K round-trip RMSE ≤ 0.012 — both
//!   tighter than the default abs tolerance, so the gate would still
//!   catch a Q6_K-level regression even if the imatrix port introduced
//!   noise at the Q4_K-precision level.
//! - 1e-2 rel tolerance is the GGUF-byte-spec headroom for f32 dot
//!   products of length 4096+ (typical hidden_size); below that, an
//!   accumulator order divergence between Rust's `for col` loop and
//!   llama.cpp's `for col` loop would surface as a divergent gate.
//!
//! Callers wanting tighter or looser tolerances pass them explicitly
//! to [`cross_validate_imatrix_gguf`].
//!
//! ## Sovereignty
//!
//! Pure Rust, no `pyo3`, no shell-out. Reuses the existing
//! [`super::imatrix::ImatrixCollector::load_imatrix_gguf`] reader so
//! the cross-validation gate exercises the same loader the runtime
//! consumes — this is the load-bearing property: a regression in the
//! loader would break both production and the gate, not just one.
//!
//! ## `#[allow(dead_code)]` rationale
//!
//! The public API surface (`XValidationReport`, `TensorComparison`,
//! `XValidationError`, `cross_validate_imatrix_gguf`) is consumed by
//! the integration test crate `tests/imatrix_xvalidation.rs` (its own
//! Cargo target — Rust integration tests compile against the public
//! `lib.rs` re-exports independently from the `bin/` target). The bin
//! currently has no use site, so without `#[allow(dead_code)]` the
//! warnings would inflate the warning count past the ≤395 budget. The
//! same pattern is used by [`super::imatrix_calibrator::ImatrixCalibrator`]
//! — the production wire-up lands in P8 (CLI calibrator dispatch) at
//! which point the allow can be lifted.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use thiserror::Error;

use super::imatrix::{ImatrixCollector, ImatrixError};

/// Per-tensor comparison report between two GGUF v3 imatrix files.
///
/// Field semantics:
///
/// - `name`: canonical tensor name (e.g. `blk.0.attn_q.weight`).
/// - `max_abs_diff_in_sum2`: `max(abs(a.values[i] - b.values[i]))` over
///   all `i` in `0..values.len()`. NaN-safe: NaN is propagated to the
///   max via `f32::max` which prefers the non-NaN argument; if both
///   sides are NaN at the same position, the result remains NaN and
///   the gate fails.
/// - `max_rel_diff_in_sum2`: `max(abs(a - b) / max(abs(a), abs(b),
///   1e-12))`. The `1e-12` denominator floor is the canonical "avoid
///   div-by-zero on both-sides-zero" guard; on both-zero, the relative
///   diff is 0 (both abs values are 0), which is correct.
/// - `counts_match`: `true` iff `a.counts == b.counts` (byte-equal i64
///   slices; counts are exact integer token counts, no precision
///   leeway).
/// - `elements`: `a.values.len()`, recorded for the markdown report
///   so reviewers can spot expert-axis count discrepancies (MoE
///   tensors carry `n_experts × row_size` elements vs dense's
///   `row_size`).
#[derive(Debug, Clone)]
pub struct TensorComparison {
    pub name: String,
    pub max_abs_diff_in_sum2: f32,
    pub max_rel_diff_in_sum2: f32,
    pub counts_match: bool,
    pub elements: usize,
}

/// Cross-validation report comparing two GGUF v3 imatrix files.
///
/// Construct via [`cross_validate_imatrix_gguf`]; consume via
/// [`XValidationReport::is_pass`] (gate decision) and
/// [`XValidationReport::to_markdown`] (commit-time documentation).
#[derive(Debug, Clone)]
pub struct XValidationReport {
    /// Tensor names present in side `a` (the first argument to
    /// [`cross_validate_imatrix_gguf`]) but absent from `b`.
    pub tensors_in_a_only: Vec<String>,
    /// Tensor names present in side `b` but absent from `a`.
    pub tensors_in_b_only: Vec<String>,
    /// Per-tensor comparison rows for the **shared** tensor set, sorted
    /// alphabetically by `name` for reproducible markdown output.
    pub per_tensor: Vec<TensorComparison>,
    /// Absolute-tolerance threshold the gate was evaluated against.
    pub abs_tolerance: f32,
    /// Relative-tolerance threshold the gate was evaluated against.
    pub rel_tolerance: f32,
}

impl XValidationReport {
    /// Sentinel constructor for the iter-1 `#[ignore]`-gated cell. The
    /// real-model gate hasn't been run; record that explicitly rather
    /// than fabricate a passing report. Recognisable via empty
    /// per-tensor list **and** zero tolerances.
    ///
    /// iter-2 replaces every call site with a real
    /// [`cross_validate_imatrix_gguf`] invocation. Until then,
    /// [`XValidationReport::is_pass`] returns `false` for this sentinel
    /// (no per-tensor data ⇒ no parity claim possible) so any caller
    /// gating on `is_pass` correctly surfaces the absence of a gate.
    pub fn not_measured() -> Self {
        Self {
            tensors_in_a_only: Vec::new(),
            tensors_in_b_only: Vec::new(),
            per_tensor: Vec::new(),
            abs_tolerance: 0.0,
            rel_tolerance: 0.0,
        }
    }

    /// True iff the gate passes:
    ///
    /// 1. **Tensor-set equality** — `tensors_in_a_only.is_empty() &&
    ///    tensors_in_b_only.is_empty()`. A missing tensor is never a
    ///    precision artefact; it always indicates a port bug.
    /// 2. **Per-tensor `in_sum2` tolerance** — every shared tensor
    ///    satisfies `max_abs_diff_in_sum2 <= abs_tolerance OR
    ///    max_rel_diff_in_sum2 <= rel_tolerance`. The OR (not AND) is
    ///    deliberate: small-magnitude tensors saturate the absolute
    ///    bound trivially; large-magnitude tensors saturate the
    ///    relative bound. Either is sufficient to declare numeric
    ///    equivalence.
    /// 3. **`counts` exact match** — every shared tensor has
    ///    `counts_match == true`. Counts are integer token counts;
    ///    they have no f32-precision leeway.
    /// 4. **Non-empty `per_tensor`** — the sentinel
    ///    [`XValidationReport::not_measured`] returns false here so
    ///    callers don't mistake "no data" for "passing gate".
    ///
    /// Returns false if any of the four conditions are violated.
    pub fn is_pass(&self) -> bool {
        if !self.tensors_in_a_only.is_empty() || !self.tensors_in_b_only.is_empty() {
            return false;
        }
        if self.per_tensor.is_empty() {
            // Sentinel `not_measured()` or a degenerate empty-vs-empty
            // file pair. Either way, the gate cannot pass without
            // observed data.
            return false;
        }
        for cmp in &self.per_tensor {
            if !cmp.counts_match {
                return false;
            }
            let abs_ok = cmp.max_abs_diff_in_sum2 <= self.abs_tolerance;
            let rel_ok = cmp.max_rel_diff_in_sum2 <= self.rel_tolerance;
            if !(abs_ok || rel_ok) {
                return false;
            }
        }
        true
    }

    /// Markdown-formatted report suitable for inclusion in the ADR-014
    /// P6 close-commit message and the daily peer-parity dashboard.
    ///
    /// Layout:
    /// ```markdown
    /// | Tensor | max abs Δ in_sum2 | max rel Δ in_sum2 | counts | elements |
    /// |--------|-------------------|-------------------|--------|----------|
    /// | blk.0.attn_q.weight | 0.000123 | 0.000456 | match | 4096 |
    /// ```
    ///
    /// Tensor-set asymmetries are surfaced **above** the table as
    /// dedicated `**Missing in B:**` / `**Missing in A:**` lines so a
    /// human reviewer reads them first before scanning per-tensor numbers.
    /// Numeric values render to 6 decimal places — sufficient resolution
    /// to distinguish 1e-6 (Q6_K dequant noise floor) from 1e-3 (default
    /// abs tolerance).
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "**Tolerances:** abs ≤ {:.6}, rel ≤ {:.6} ({} tensor pair{} compared)\n\n",
            self.abs_tolerance,
            self.rel_tolerance,
            self.per_tensor.len(),
            if self.per_tensor.len() == 1 { "" } else { "s" },
        ));
        if !self.tensors_in_a_only.is_empty() {
            out.push_str("**Missing in B (present in A only):**\n");
            for n in &self.tensors_in_a_only {
                out.push_str(&format!("- `{n}`\n"));
            }
            out.push('\n');
        }
        if !self.tensors_in_b_only.is_empty() {
            out.push_str("**Missing in A (present in B only):**\n");
            for n in &self.tensors_in_b_only {
                out.push_str(&format!("- `{n}`\n"));
            }
            out.push('\n');
        }
        out.push_str(
            "| Tensor | max abs Δ in_sum2 | max rel Δ in_sum2 | counts | elements |\n",
        );
        out.push_str(
            "|--------|-------------------|-------------------|--------|----------|\n",
        );
        for cmp in &self.per_tensor {
            out.push_str(&format!(
                "| `{}` | {:.6} | {:.6} | {} | {} |\n",
                cmp.name,
                cmp.max_abs_diff_in_sum2,
                cmp.max_rel_diff_in_sum2,
                if cmp.counts_match { "match" } else { "MISMATCH" },
                cmp.elements,
            ));
        }
        out
    }
}

/// Errors that can arise while cross-validating two GGUF v3 imatrix
/// files. Distinct from [`super::imatrix::ImatrixError`] because the
/// gate has its own failure modes (one of the inputs failed to load,
/// or the two inputs disagree on a per-tensor structural invariant
/// that's never a precision artefact).
#[derive(Debug, Error)]
pub enum XValidationError {
    /// Loading one of the two `.imatrix.gguf` files failed.
    #[error("failed to load imatrix file {path}: {source}")]
    Load {
        path: PathBuf,
        #[source]
        source: ImatrixError,
    },
    /// A shared tensor has structurally inconsistent metadata between
    /// the two files (`values.len()` mismatch or `counts.len()`
    /// mismatch). Surfaced as `Invariant` because this can never be
    /// caused by float precision — only by a port bug or a corrupted
    /// file.
    #[error("internal invariant violated: {0}")]
    Invariant(String),
}

/// Compare two GGUF v3 imatrix files (PR #9400 schema).
///
/// Reuses [`super::imatrix::ImatrixCollector::load_imatrix_gguf`] for
/// both inputs — no new GGUF reader is introduced. The two files MUST
/// be the same llama.cpp PR #9400 schema (GGUF v3 magic + `general.type
/// = "imatrix"` + `<name>.in_sum2` + `<name>.counts` tensor pairs);
/// any other GGUF file kind will be rejected by the loader and
/// surface as [`XValidationError::Load`].
///
/// Tolerance defaults (`abs = 1e-3`, `rel = 1e-2`) are documented at
/// module level. Pass tighter values for hardening tests; pass looser
/// values when comparing against an older llama.cpp build whose f32
/// summation order may differ.
///
/// Returns an [`XValidationReport`]; consume it via
/// [`XValidationReport::is_pass`] for a binary gate decision and
/// [`XValidationReport::to_markdown`] for human inspection.
pub fn cross_validate_imatrix_gguf(
    a: &Path,
    b: &Path,
    abs_tolerance: f32,
    rel_tolerance: f32,
) -> Result<XValidationReport, XValidationError> {
    let (col_a, _datasets_a) =
        ImatrixCollector::load_imatrix_gguf(a).map_err(|source| XValidationError::Load {
            path: a.to_path_buf(),
            source,
        })?;
    let (col_b, _datasets_b) =
        ImatrixCollector::load_imatrix_gguf(b).map_err(|source| XValidationError::Load {
            path: b.to_path_buf(),
            source,
        })?;

    let stats_a = col_a.stats();
    let stats_b = col_b.stats();

    // Tensor-set diff. Sorted for reproducible reports.
    let mut tensors_in_a_only: Vec<String> = stats_a
        .keys()
        .filter(|k| !stats_b.contains_key(*k))
        .cloned()
        .collect();
    tensors_in_a_only.sort();

    let mut tensors_in_b_only: Vec<String> = stats_b
        .keys()
        .filter(|k| !stats_a.contains_key(*k))
        .cloned()
        .collect();
    tensors_in_b_only.sort();

    // Shared tensor names, sorted alphabetically — matches the GGUF
    // writer's sort order so the markdown report mirrors the on-disk
    // tensor order.
    let mut shared: Vec<&String> = stats_a.keys().filter(|k| stats_b.contains_key(*k)).collect();
    shared.sort();

    let mut per_tensor: Vec<TensorComparison> = Vec::with_capacity(shared.len());
    for name in shared {
        // Both sides exist by construction (we filtered on
        // `stats_b.contains_key`). Direct indexing is safe; map access
        // is `Option<&V>` so destructure rather than `.unwrap()` to
        // honour the no-unwrap discipline.
        let sa = match stats_a.get(name) {
            Some(s) => s,
            None => {
                return Err(XValidationError::Invariant(format!(
                    "tensor `{name}` vanished from side A between key scan and value lookup"
                )));
            }
        };
        let sb = match stats_b.get(name) {
            Some(s) => s,
            None => {
                return Err(XValidationError::Invariant(format!(
                    "tensor `{name}` vanished from side B between key scan and value lookup"
                )));
            }
        };

        if sa.values.len() != sb.values.len() {
            return Err(XValidationError::Invariant(format!(
                "tensor `{name}`: in_sum2 length mismatch (a={}, b={}); shape divergence is never a precision artefact",
                sa.values.len(),
                sb.values.len(),
            )));
        }
        if sa.counts.len() != sb.counts.len() {
            return Err(XValidationError::Invariant(format!(
                "tensor `{name}`: counts length mismatch (a={}, b={}); expert-count divergence is never a precision artefact",
                sa.counts.len(),
                sb.counts.len(),
            )));
        }

        let mut max_abs: f32 = 0.0;
        let mut max_rel: f32 = 0.0;
        const REL_DENOM_FLOOR: f32 = 1e-12;
        for (av, bv) in sa.values.iter().zip(sb.values.iter()) {
            let diff = (av - bv).abs();
            if diff > max_abs || diff.is_nan() {
                max_abs = diff;
            }
            let denom = av.abs().max(bv.abs()).max(REL_DENOM_FLOOR);
            let rel = diff / denom;
            if rel > max_rel || rel.is_nan() {
                max_rel = rel;
            }
        }

        let counts_match = sa.counts == sb.counts;

        per_tensor.push(TensorComparison {
            name: name.clone(),
            max_abs_diff_in_sum2: max_abs,
            max_rel_diff_in_sum2: max_rel,
            counts_match,
            elements: sa.values.len(),
        });
    }

    Ok(XValidationReport {
        tensors_in_a_only,
        tensors_in_b_only,
        per_tensor,
        abs_tolerance,
        rel_tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_measured_sentinel_is_recognisable_and_does_not_pass() {
        let r = XValidationReport::not_measured();
        assert!(r.tensors_in_a_only.is_empty());
        assert!(r.tensors_in_b_only.is_empty());
        assert!(r.per_tensor.is_empty());
        assert_eq!(r.abs_tolerance, 0.0);
        assert_eq!(r.rel_tolerance, 0.0);
        // Critical: the sentinel must NOT be mistaken for a passing gate.
        // Otherwise the iter-2 #[ignore]-gated cell would silently
        // succeed without ever running llama-imatrix.
        assert!(
            !r.is_pass(),
            "not_measured() must never report is_pass() == true"
        );
    }

    #[test]
    fn is_pass_requires_non_empty_per_tensor() {
        // Even with empty tensor-only sets and zero tolerances, an
        // empty per_tensor must not pass — we cannot claim parity on
        // an empty observation.
        let r = XValidationReport {
            tensors_in_a_only: Vec::new(),
            tensors_in_b_only: Vec::new(),
            per_tensor: Vec::new(),
            abs_tolerance: 1e-3,
            rel_tolerance: 1e-2,
        };
        assert!(!r.is_pass());
    }

    #[test]
    fn is_pass_fails_on_a_only_or_b_only() {
        let cmp = TensorComparison {
            name: "blk.0.x".into(),
            max_abs_diff_in_sum2: 0.0,
            max_rel_diff_in_sum2: 0.0,
            counts_match: true,
            elements: 1,
        };
        let r_a = XValidationReport {
            tensors_in_a_only: vec!["only-in-a".into()],
            tensors_in_b_only: Vec::new(),
            per_tensor: vec![cmp.clone()],
            abs_tolerance: 1e-3,
            rel_tolerance: 1e-2,
        };
        assert!(!r_a.is_pass());

        let r_b = XValidationReport {
            tensors_in_a_only: Vec::new(),
            tensors_in_b_only: vec!["only-in-b".into()],
            per_tensor: vec![cmp],
            abs_tolerance: 1e-3,
            rel_tolerance: 1e-2,
        };
        assert!(!r_b.is_pass());
    }

    #[test]
    fn is_pass_or_branches_between_abs_and_rel() {
        // Large-magnitude tensor where abs diff exceeds abs tolerance
        // but rel diff is within rel tolerance — must pass.
        let cmp_rel_only = TensorComparison {
            name: "big".into(),
            max_abs_diff_in_sum2: 0.5,
            max_rel_diff_in_sum2: 1e-3,
            counts_match: true,
            elements: 1,
        };
        let r1 = XValidationReport {
            tensors_in_a_only: Vec::new(),
            tensors_in_b_only: Vec::new(),
            per_tensor: vec![cmp_rel_only],
            abs_tolerance: 1e-3,
            rel_tolerance: 1e-2,
        };
        assert!(r1.is_pass(), "abs-fail + rel-pass must pass via OR");

        // Small-magnitude tensor where rel diff is huge (small denom)
        // but abs diff is within abs tolerance — must pass.
        let cmp_abs_only = TensorComparison {
            name: "small".into(),
            max_abs_diff_in_sum2: 1e-5,
            max_rel_diff_in_sum2: 5.0,
            counts_match: true,
            elements: 1,
        };
        let r2 = XValidationReport {
            tensors_in_a_only: Vec::new(),
            tensors_in_b_only: Vec::new(),
            per_tensor: vec![cmp_abs_only],
            abs_tolerance: 1e-3,
            rel_tolerance: 1e-2,
        };
        assert!(r2.is_pass(), "rel-fail + abs-pass must pass via OR");
    }

    #[test]
    fn is_pass_fails_on_counts_mismatch_even_with_perfect_in_sum2() {
        // counts mismatch is unconditional fail — counts are exact
        // integer token counts, never a precision artefact.
        let cmp = TensorComparison {
            name: "blk.0.x".into(),
            max_abs_diff_in_sum2: 0.0,
            max_rel_diff_in_sum2: 0.0,
            counts_match: false,
            elements: 8,
        };
        let r = XValidationReport {
            tensors_in_a_only: Vec::new(),
            tensors_in_b_only: Vec::new(),
            per_tensor: vec![cmp],
            abs_tolerance: 1e-3,
            rel_tolerance: 1e-2,
        };
        assert!(!r.is_pass());
    }

    // Note on the omitted "tolerances in header" unit test:
    // the integration crate `tests/imatrix_xvalidation.rs::
    // xvalidation_markdown_report_contains_required_columns` already
    // asserts both `0.001000` and `0.010000` render in the header from
    // the same code path; duplicating it here would only inflate the
    // bin test count without adding signal.

    #[test]
    fn to_markdown_surfaces_a_only_and_b_only_above_table() {
        let r = XValidationReport {
            tensors_in_a_only: vec!["alpha.in.a".into()],
            tensors_in_b_only: vec!["beta.in.b".into()],
            per_tensor: Vec::new(),
            abs_tolerance: 1e-3,
            rel_tolerance: 1e-2,
        };
        let md = r.to_markdown();
        let a_only_pos = md.find("Missing in B").expect("a-only line present");
        let b_only_pos = md.find("Missing in A").expect("b-only line present");
        let table_pos = md.find("| Tensor |").expect("table header present");
        assert!(
            a_only_pos < table_pos && b_only_pos < table_pos,
            "asymmetry lines must appear above table for human triage"
        );
        assert!(md.contains("alpha.in.a"));
        assert!(md.contains("beta.in.b"));
    }
}
