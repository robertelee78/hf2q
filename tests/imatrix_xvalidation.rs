//! ADR-014 P6 close iter-1 — cross-validation gate machinery for two
//! GGUF v3 imatrix files (PR #9400 schema).
//!
//! Five always-on tests cover the gate-decision predicate
//! (`XValidationReport::is_pass`) and the markdown rendering shape:
//!
//! 1. `xvalidation_self_compare_passes` — identical input on both
//!    sides ⇒ pass with trivially zero diffs.
//! 2. `xvalidation_perturbed_in_sum2_fails_at_tight_tolerance` — one
//!    activation mutated by ~+0.01 in side `b` ⇒ fail at tight
//!    abs/rel tolerance with `max_abs_diff_in_sum2 ≈ 0.01`.
//! 3. `xvalidation_counts_mismatch_fails` — same `in_sum2` but counts
//!    differ by 1 ⇒ fail with `counts_match == false`.
//! 4. `xvalidation_missing_tensor_fails` — side `b` is missing one
//!    tensor present in side `a` ⇒ fail with the missing tensor
//!    surfaced in `tensors_in_a_only`.
//! 5. `xvalidation_markdown_report_contains_required_columns` — the
//!    rendered markdown includes the documented column headers and
//!    renders numeric values to 6 decimal places when populated.
//!
//! One `#[ignore]`-gated cell documents the iter-2 real-model gate
//! (`xvalidation_vs_llama_imatrix_qwen35_smoke`); it returns the
//! `XValidationReport::not_measured()` sentinel this iter.
//!
//! These tests use only the public surface of
//! `hf2q::calibrate::imatrix_xvalidate` + `hf2q::calibrate::imatrix`
//! — no introspection of private state, no direct file-byte mutation
//! (which would be testing the loader, not the gate). Each side's
//! `Stats` is built via [`ImatrixCollector::accumulate_dense`] from
//! synthetic activations chosen so the desired diff appears naturally.

// hf2q is a binary crate (no `[lib]` target); integration tests cannot
// `use hf2q::...`. The codebase's established pattern for reaching
// internal modules from `tests/*.rs` is `#[path]`-include — see
// `tests/openwebui_*.rs` which `#[path]`-include `openwebui_helpers/mod.rs`.
//
// Here we `#[path]`-include the two source files we exercise as
// sibling modules of this test crate. `imatrix.rs` has only std +
// thiserror + tracing deps (all present in this crate's Cargo.toml),
// and `imatrix_xvalidate.rs` references it as `super::imatrix` — which
// resolves correctly because both files become children of this
// integration test crate's root.
//
// `#[cfg(test)]` test modules inside the included files compile +
// run as if they were ours; that's a feature, not a bug — it pins
// the algorithm-side invariants the comparator depends on.

#[path = "../src/calibrate/imatrix.rs"]
mod imatrix;

#[path = "../src/calibrate/imatrix_xvalidate.rs"]
mod imatrix_xvalidate;

use std::path::Path;

use imatrix::ImatrixCollector;
use imatrix_xvalidate::{cross_validate_imatrix_gguf, XValidationReport};

/// Build a collector with one dense tensor of the given activations.
/// Uses `accumulate_dense` (the production path) so the resulting
/// `Stats` is identical to what the live calibrator produces.
fn collector_with_dense(
    tensor_name: &str,
    activations: &[f32],
    n_tokens: usize,
    row_size: usize,
    n_chunks: usize,
) -> ImatrixCollector {
    let mut col = ImatrixCollector::new();
    for _ in 0..n_chunks {
        col.accumulate_dense(tensor_name, activations, n_tokens, row_size)
            .expect("accumulate_dense");
    }
    for _ in 0..n_chunks {
        col.record_chunk();
    }
    col
}

fn save_collector(col: &ImatrixCollector, path: &Path) {
    col.save_imatrix_gguf(path, 512, &["test-corpus.txt"])
        .expect("save_imatrix_gguf");
}

#[test]
fn xvalidation_self_compare_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let path_a = tmp.path().join("a.imatrix.gguf");
    let path_b = tmp.path().join("b.imatrix.gguf");

    // Two identical collectors saved to two paths; cross-validating
    // them must produce a passing report with trivially zero diffs.
    let col = collector_with_dense(
        "blk.0.attn_q.weight",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        4, // n_tokens
        2, // row_size
        2, // n_chunks
    );
    save_collector(&col, &path_a);
    save_collector(&col, &path_b);

    let report =
        cross_validate_imatrix_gguf(&path_a, &path_b, 1e-3, 1e-2).expect("cross-validate");

    assert!(
        report.tensors_in_a_only.is_empty(),
        "self-compare must not surface a-only tensors"
    );
    assert!(
        report.tensors_in_b_only.is_empty(),
        "self-compare must not surface b-only tensors"
    );
    assert_eq!(report.per_tensor.len(), 1);
    let cmp = &report.per_tensor[0];
    assert_eq!(cmp.name, "blk.0.attn_q.weight");
    assert_eq!(
        cmp.max_abs_diff_in_sum2, 0.0,
        "self-compare abs diff must be exactly zero"
    );
    assert_eq!(
        cmp.max_rel_diff_in_sum2, 0.0,
        "self-compare rel diff must be exactly zero"
    );
    assert!(cmp.counts_match, "self-compare counts must match");
    assert_eq!(cmp.elements, 2, "dense layer with row_size=2");
    assert!(report.is_pass(), "self-compare must pass the gate");
}

#[test]
fn xvalidation_perturbed_in_sum2_fails_at_tight_tolerance() {
    let tmp = tempfile::tempdir().unwrap();
    let path_a = tmp.path().join("a.imatrix.gguf");
    let path_b = tmp.path().join("b.imatrix.gguf");

    // Side a: baseline. row_size=1, single token of value 1.0 per chunk
    // ⇒ in_sum2[0] == 1.0 (the per-chunk-mean accumulator with one
    // chunk: sum(1.0²) / 1 == 1.0).
    let col_a = collector_with_dense("blk.0.x.weight", &[1.0], 1, 1, 1);
    save_collector(&col_a, &path_a);

    // Side b: same shape, single token of value 1.005 ⇒
    // in_sum2[0] == 1.005² ≈ 1.010025. Diff ≈ 0.010025 from side a's 1.0.
    let col_b = collector_with_dense("blk.0.x.weight", &[1.005], 1, 1, 1);
    save_collector(&col_b, &path_b);

    // At tight tolerance abs=1e-3 rel=1e-3, the ~0.01 abs diff must
    // fail (abs > 1e-3) and the ~0.01 rel diff must also fail
    // (rel > 1e-3).
    let report =
        cross_validate_imatrix_gguf(&path_a, &path_b, 1e-3, 1e-3).expect("cross-validate");
    assert_eq!(report.per_tensor.len(), 1);
    let cmp = &report.per_tensor[0];
    assert!(cmp.counts_match, "counts unchanged across sides");
    assert!(
        (cmp.max_abs_diff_in_sum2 - 0.010025).abs() < 1e-4,
        "expected max_abs_diff ≈ 0.010025, got {}",
        cmp.max_abs_diff_in_sum2
    );
    assert!(
        !report.is_pass(),
        "perturbed in_sum2 must fail at tight tolerance"
    );

    // Sanity: the same report at loose tolerance abs=1.0 must pass —
    // this proves it's the tolerance threshold (not a structural
    // mismatch) doing the rejection.
    let report_loose =
        cross_validate_imatrix_gguf(&path_a, &path_b, 1.0, 1.0).expect("cross-validate loose");
    assert!(
        report_loose.is_pass(),
        "perturbed in_sum2 must pass at loose tolerance (proves abs diff is the only failure mode)"
    );
}

#[test]
fn xvalidation_counts_mismatch_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let path_a = tmp.path().join("a.imatrix.gguf");
    let path_b = tmp.path().join("b.imatrix.gguf");

    // To isolate counts mismatch as the sole failure mode, the two
    // sides must have identical `in_sum2` but different `counts`.
    // `ImatrixCollector::accumulate_dense` increments `counts[0]` on
    // every call, so we cannot directly desync `counts` from
    // `in_sum2` through the production API. We instead pick chunk
    // sizes + values such that `in_sum2` is byte-equal across the
    // two sides while `counts` differs by 1.
    //
    // Construction (per-chunk-mean accumulator semantics: each
    // accumulate_dense adds `sum(x²) / n_tokens` to `values[col]`):
    //
    // - Side a: 1 chunk of activation [1.0], n_tokens=1, row_size=1
    //   ⇒ values[0] = 1²/1 = 1.0; counts[0] = 1.
    // - Side b: 2 chunks of activation [sqrt(0.5)], n_tokens=1,
    //   row_size=1 ⇒ values[0] = 0.5 + 0.5 = 1.0; counts[0] = 2.
    //
    // ⇒ in_sum2 byte-equal (1.0 vs 1.0); counts differ (1 vs 2).
    // The gate must fail on counts_match alone.
    let col_a = collector_with_dense("blk.0.x.weight", &[1.0], 1, 1, 1);
    save_collector(&col_a, &path_a);

    let val_b = (0.5_f32).sqrt();
    let col_b = collector_with_dense("blk.0.x.weight", &[val_b], 1, 1, 2);
    save_collector(&col_b, &path_b);

    let report =
        cross_validate_imatrix_gguf(&path_a, &path_b, 1e-3, 1e-3).expect("cross-validate");
    assert_eq!(report.per_tensor.len(), 1);
    let cmp = &report.per_tensor[0];
    assert!(
        cmp.max_abs_diff_in_sum2 < 1e-5,
        "in_sum2 should match (1.0 vs sqrt(0.5)² + sqrt(0.5)² = 1.0); got {}",
        cmp.max_abs_diff_in_sum2
    );
    assert!(
        !cmp.counts_match,
        "counts must differ (a=1, b=2)"
    );
    assert!(
        !report.is_pass(),
        "counts mismatch must fail the gate even when in_sum2 matches exactly"
    );
}

#[test]
fn xvalidation_missing_tensor_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let path_a = tmp.path().join("a.imatrix.gguf");
    let path_b = tmp.path().join("b.imatrix.gguf");

    // Side a: two tensors.
    let mut col_a = ImatrixCollector::new();
    col_a
        .accumulate_dense("blk.0.attn_q.weight", &[1.0, 2.0], 2, 1)
        .unwrap();
    col_a
        .accumulate_dense("blk.0.attn_k.weight", &[3.0, 4.0], 2, 1)
        .unwrap();
    col_a.record_chunk();
    save_collector(&col_a, &path_a);

    // Side b: only the first tensor (missing attn_k).
    let mut col_b = ImatrixCollector::new();
    col_b
        .accumulate_dense("blk.0.attn_q.weight", &[1.0, 2.0], 2, 1)
        .unwrap();
    col_b.record_chunk();
    save_collector(&col_b, &path_b);

    let report =
        cross_validate_imatrix_gguf(&path_a, &path_b, 1e-3, 1e-3).expect("cross-validate");

    assert_eq!(
        report.tensors_in_a_only,
        vec!["blk.0.attn_k.weight".to_string()],
        "the missing tensor must be surfaced in tensors_in_a_only"
    );
    assert!(
        report.tensors_in_b_only.is_empty(),
        "no tensors should be exclusive to side b"
    );
    // The shared tensor (attn_q) compares cleanly.
    assert_eq!(report.per_tensor.len(), 1);
    let cmp = &report.per_tensor[0];
    assert_eq!(cmp.name, "blk.0.attn_q.weight");
    assert_eq!(cmp.max_abs_diff_in_sum2, 0.0);
    assert!(cmp.counts_match);
    // Despite the shared tensor being identical, the asymmetry must
    // fail the gate.
    assert!(
        !report.is_pass(),
        "missing tensor must fail the gate even if all shared tensors match"
    );
}

#[test]
fn xvalidation_markdown_report_contains_required_columns() {
    // Empty report: header table columns present, no data rows.
    let empty = XValidationReport {
        tensors_in_a_only: Vec::new(),
        tensors_in_b_only: Vec::new(),
        per_tensor: Vec::new(),
        abs_tolerance: 1e-3,
        rel_tolerance: 1e-2,
    };
    let md_empty = empty.to_markdown();
    assert!(md_empty.contains("| Tensor |"), "Tensor header column");
    assert!(
        md_empty.contains("max abs Δ in_sum2"),
        "max abs Δ in_sum2 column"
    );
    assert!(
        md_empty.contains("max rel Δ in_sum2"),
        "max rel Δ in_sum2 column"
    );
    assert!(md_empty.contains("counts"), "counts column");
    assert!(md_empty.contains("elements"), "elements column");
    // Tolerance header surfaces both thresholds rendered to 6dp.
    assert!(md_empty.contains("0.001000"), "abs tolerance rendered to 6dp");
    assert!(md_empty.contains("0.010000"), "rel tolerance rendered to 6dp");

    // Populated report: tensor name + numeric values present, 6dp.
    let tmp = tempfile::tempdir().unwrap();
    let path_a = tmp.path().join("a.imatrix.gguf");
    let path_b = tmp.path().join("b.imatrix.gguf");
    let col_a = collector_with_dense("blk.7.ffn_down.weight", &[0.1, 0.2], 2, 1, 1);
    let col_b = collector_with_dense("blk.7.ffn_down.weight", &[0.1, 0.2], 2, 1, 1);
    save_collector(&col_a, &path_a);
    save_collector(&col_b, &path_b);

    let r = cross_validate_imatrix_gguf(&path_a, &path_b, 1e-3, 1e-2).unwrap();
    let md = r.to_markdown();
    assert!(md.contains("blk.7.ffn_down.weight"));
    assert!(md.contains("0.000000"), "zero diff rendered to 6dp as 0.000000");
    assert!(md.contains("match"), "counts_match true ⇒ 'match' string");
}

/// **iter-2 hardware gate** — wires up:
///
/// 1. Tokenize a small calibration corpus (`tests/fixtures/imatrix-corpus.txt`,
///    iter-2-landed) with the Qwen3.5-0.6B tokenizer.
/// 2. Convert Qwen3.5-0.6B to GGUF (via existing `cmd_convert` or a
///    pre-cached fixture) — file size ~1 GB.
/// 3. Run hf2q's `ImatrixCalibrator` (constructed from the
///    Qwen35Dense::ActivationCapture impl) over the corpus; save to
///    `<tmp>/hf2q.imatrix.gguf` via `save_imatrix_gguf`.
/// 4. Run llama.cpp's `llama-imatrix` (resolved via
///    `tests/common/llama_cpp_runner::run_llama_imatrix`) on the same
///    GGUF + corpus; output to `<tmp>/llama.imatrix.gguf`.
/// 5. Cross-validate via [`cross_validate_imatrix_gguf`] at default
///    tolerances `abs=1e-3, rel=1e-2`. Assert `is_pass()`.
/// 6. Write the rendered markdown report to
///    `docs/adr-014-p6-xvalidation-<YYYY-MM-DD>.md` for the close
///    commit.
///
/// **Iter-1 contract**: this cell must NOT silently produce a passing
/// report — that would let the close gate paint the P6 row 🟢 without
/// ever observing real cross-tool agreement. We use
/// [`XValidationReport::not_measured`] which `is_pass() == false` so
/// any caller that re-enables this test before iter-2 is wired in
/// gets a hard fail (not a fake-green).
#[test]
#[ignore = "P6 close gate: needs real model + llama-imatrix subprocess + ~1GB disk for Qwen3.5-0.6B GGUF"]
fn xvalidation_vs_llama_imatrix_qwen35_smoke() {
    eprintln!("P6 iter-1 — gate machinery only; iter-2 wires real model");
    let report = XValidationReport::not_measured();
    // Iter-1 contract: the sentinel cannot pass the gate. Iter-2
    // replaces this with a real `cross_validate_imatrix_gguf` call.
    assert!(
        !report.is_pass(),
        "iter-1: sentinel must not be mistaken for a passing gate"
    );
}
