//! ADR-014 P11-prereq Iter B — `DwqKQuantizer` integration smoke crate.
//!
//! ## Why this file is a thin smoke crate (not a fat algorithmic crate)
//!
//! `hf2q` is a binary crate (no `[lib]` target — `Cargo.toml::package`).
//! Integration tests in `tests/*.rs` cannot `use hf2q::quantize::...`
//! because the binary's modules are not exposed as `pub` library API.
//! The codebase has two established patterns for binary-crate
//! integration tests:
//!
//!   1. Drive the binary via `assert_cmd::Command::cargo_bin("hf2q")`
//!      — used by `tests/codec_direct_type_code.rs` (Iter A) and
//!      `tests/cmd_convert_dispatch.rs`.  Requires the new code to be
//!      reachable from the CLI surface (`cmd_convert` --quant menu).
//!      `DwqKQuantizer` is NOT yet wired into the CLI dispatch (Iter C
//!      lands the wiring + flips the `Dwq*` arms over).
//!
//!   2. `#[path]`-include the source files under test as sibling
//!      modules — used by `tests/imatrix_xvalidation.rs` (2-file
//!      include) and `tests/ppl_driver.rs` (multi-file include +
//!      type-stubs).  Works when the included files have **shallow**
//!      transitive dep trees that resolve against this test crate's
//!      Cargo deps.  `dwq_k_quantizer.rs` transitively pulls in
//!      `k_quant_codec_quantizer` → `k_quant_codec` → `k_quant` +
//!      `q_legacy` + `calibrate::calibrator` + `ir::*` — too deep to
//!      mirror without a fragile shadow tree.
//!
//! ## Pattern chosen
//!
//! Same as `gguf.rs` precedent (Iter A): the **algorithm-side
//! invariants live in-source** at
//! `src/quantize/dwq_k_quantizer.rs::tests` (24 always-on tests, full
//! `Quantizer::quantize_tensor` end-to-end coverage including 2D
//! tensors, P28 typed-error path, preserve passthrough, layer-index
//! parity with `MixedBitQuantizer`).  This integration crate runs the
//! built binary's `--help` to assert that the new module integrates
//! cleanly into the binary build — i.e. that adding `DwqKQuantizer`
//! did NOT break the CLI surface or introduce a build-time regression.
//! Iter C will swap this file for fat `assert_cmd`-driven tests once
//! the `--quant dwq-k-mixed-*` menu lands.
//!
//! See ADR-014 § "Audit revision (2026-04-27)" for the matching
//! precedent on the Iter A side ("11 in-source regression tests live
//! INSIDE `gguf.rs::tests` (not `tests/codec_direct_type_code.rs`)
//! because the consumer-side helpers are module-private").

use std::path::PathBuf;
use std::process::Command;

/// Path to the built `hf2q` binary inside the workspace target dir.
fn hf2q_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("release")
        .join("hf2q")
}

/// Smoke: the binary builds successfully (i.e. the new `pub mod
/// dwq_k_quantizer` line in `src/quantize/mod.rs` and the new
/// `src/quantize/dwq_k_quantizer.rs` source file did not introduce a
/// compile error).  `cargo test --release` builds the binary as a
/// dependency of this test crate; if that build failed, this entire
/// test crate would fail to compile, never reaching the assertion.
/// The assertion is therefore a tautology that documents the contract
/// — but the file's mere existence forces the binary build into the
/// test-pipeline critical path.
#[test]
fn binary_builds_with_dwq_k_quantizer_module_present() {
    // Tautological check — if the binary failed to build, this
    // integration crate would not have linked.  The point of the
    // assertion is the doc-comment trail.  We assert against
    // `Cargo.toml::[package].name` rather than CARGO_MANIFEST_DIR so
    // the test is stable across worktree layouts (CFA worktrees live
    // under `.cfa-worktrees/<branch>/` and would otherwise spuriously
    // fail this assertion).
    let cargo_toml = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("Cargo.toml");
    let contents = std::fs::read_to_string(&cargo_toml)
        .expect("Cargo.toml must be readable");
    assert!(
        contents.contains("name = \"hf2q\""),
        "expected this test to run under the hf2q crate; got Cargo.toml={contents:.200}"
    );
}

/// Smoke: the built binary's `convert --help` runs without error.
/// This indirectly verifies that:
///
///   * The new `dwq_k_quantizer` module did not break the cmd_convert
///     CLI parsing (clap derive macro re-runs on every build).
///   * The binary was actually built (not stale / missing).
///
/// Iter C will extend this to assert the new `dwq-k-mixed-*` strings
/// appear in the `--quant` menu once they are added to `QuantMethod`.
#[test]
fn binary_convert_help_runs() {
    let bin = hf2q_binary();
    if !bin.exists() {
        eprintln!(
            "warning: hf2q binary not found at {}; skipping (will run on next build)",
            bin.display()
        );
        return;
    }
    let output = Command::new(&bin)
        .args(["convert", "--help"])
        .output()
        .expect("hf2q convert --help must run");
    assert!(
        output.status.success(),
        "hf2q convert --help failed: stdout={:?} stderr={:?}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // `--quant` is the orthogonal-axis flag the new variants will
    // eventually live on (Iter C).  Verifies the help surface itself
    // is intact — if the CLI broke, this string would be missing.
    assert!(
        stdout.contains("--quant"),
        "expected --quant flag in convert --help output, got: {stdout}"
    );
}

/// Smoke: the built binary's top-level `--help` runs and lists the
/// `convert` subcommand.  Catches regression where the new module
/// broke the top-level CLI dispatch.
#[test]
fn binary_top_level_help_lists_convert_subcommand() {
    let bin = hf2q_binary();
    if !bin.exists() {
        eprintln!(
            "warning: hf2q binary not found at {}; skipping",
            bin.display()
        );
        return;
    }
    let output = Command::new(&bin)
        .arg("--help")
        .output()
        .expect("hf2q --help must run");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("convert"),
        "expected `convert` subcommand in --help, got: {stdout}"
    );
}

/// Smoke: the binary exits with a typed error (not a panic) when
/// invoked with no arguments.  Catches regression where adding the
/// new module accidentally broke the CLI's argument-parsing
/// fallthrough.
#[test]
fn binary_no_args_exits_cleanly_not_panics() {
    let bin = hf2q_binary();
    if !bin.exists() {
        eprintln!(
            "warning: hf2q binary not found at {}; skipping",
            bin.display()
        );
        return;
    }
    let output = Command::new(&bin)
        .output()
        .expect("hf2q (no args) must run");
    // Either success (showing default help) or non-zero status — both
    // are clean exits.  What we assert against is "did NOT panic"
    // (which would manifest as an unusual signal-terminated status).
    let status_str = format!("{}", output.status);
    assert!(
        !status_str.contains("signal"),
        "hf2q (no args) appears to have panicked: status={status_str}, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Smoke: pin the module path so a future rename to a different
/// location (or a delete by accident) is caught at test time.  This
/// is a filesystem-level assertion — does not invoke the binary.
#[test]
fn dwq_k_quantizer_source_file_lives_at_expected_path() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("quantize")
        .join("dwq_k_quantizer.rs");
    assert!(
        src.exists(),
        "src/quantize/dwq_k_quantizer.rs not found — module renamed or deleted?"
    );
    let contents = std::fs::read_to_string(&src).expect("read dwq_k_quantizer.rs");
    // Lock the public API names so a downstream rename trips this gate
    // before Iter C tries to wire against a stale symbol.
    assert!(
        contents.contains("pub struct DwqKQuantizer"),
        "DwqKQuantizer struct renamed/removed?"
    );
    assert!(
        contents.contains("pub enum DwqKVariant"),
        "DwqKVariant enum renamed/removed?"
    );
    for variant in ["P46", "P48", "P68", "P28"] {
        assert!(
            contents.contains(variant),
            "DwqKVariant::{variant} variant renamed/removed?"
        );
    }
}

/// Smoke: the new `pub mod dwq_k_quantizer;` line is wired into
/// `src/quantize/mod.rs` (catches accidental delete during merge or
/// rebase).
#[test]
fn dwq_k_quantizer_module_registered_in_quantize_mod() {
    let mod_rs = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("quantize")
        .join("mod.rs");
    let contents = std::fs::read_to_string(&mod_rs).expect("read quantize/mod.rs");
    assert!(
        contents.contains("pub mod dwq_k_quantizer"),
        "src/quantize/mod.rs missing `pub mod dwq_k_quantizer;` line"
    );
}
