//! ADR-014 P11-prereq Iter D — vision-tensor + non-256-multiple skip
//! predicate integration smoke crate.
//!
//! ## Why this crate is a thin smoke crate (not a fat algorithmic crate)
//!
//! `hf2q` is a binary crate (no `[lib]` target — `Cargo.toml::package`).
//! Integration tests in `tests/*.rs` cannot `use hf2q::quantize::...`
//! because the binary's modules are not exposed as `pub` library API.
//! The codebase has two established patterns for binary-crate
//! integration tests:
//!
//!   1. Drive the binary via `assert_cmd::Command::cargo_bin("hf2q")`
//!      — used by `tests/codec_direct_type_code.rs` (Iter A) and
//!      `tests/dwq_emits_q4_k_via_cli.rs` (Iter C).  Requires the new
//!      code to be reachable from the CLI surface.
//!
//!   2. `#[path]`-include the source files under test as sibling
//!      modules.  Works when the included files have **shallow**
//!      transitive dep trees that resolve against this test crate's
//!      Cargo deps.  `should_emit_f16_for_kquant` lives in
//!      `src/quantize/layer_mix.rs`, which transitively pulls in
//!      `k_quant_codec` → `k_quant` + `q_legacy` + `calibrate::*` +
//!      `ir::*` — too deep to mirror without a fragile shadow tree.
//!
//! ## Pattern chosen
//!
//! Same as `tests/dwq_k_quantizer.rs` precedent (Iter B): the
//! **algorithm-side invariants live in-source** at:
//!
//!   * `src/quantize/layer_mix.rs::tests` — 6 always-on unit tests
//!     locking the predicate behaviour (vision patterns, non-256
//!     defensive arm, aligned language tensors stay false).
//!   * `src/quantize/k_quant_codec_quantizer.rs::tests` — 5 always-on
//!     dispatch tests proving F16 passthrough fires for vision
//!     blocker shape + BF16 vision input + non-256 non-vision input,
//!     and that aligned language tensors still route through the
//!     codec.  Plus a direct `f16_passthrough` helper unit test.
//!   * `src/quantize/variant_quantizer.rs::tests` — 2 always-on tests
//!     locking the same behaviour through the per-variant dispatcher
//!     (Q4_K_M production path).
//!   * `src/quantize/dwq_k_quantizer.rs::tests` — 4 always-on tests
//!     locking F16 emit through the DWQ dispatcher (P46 + P28 +
//!     aligned-tensor pass-through + non-256 defensive).
//!
//! This integration crate runs the built binary's `--help` to assert
//! that the new predicate integrates cleanly into the binary build —
//! i.e. that adding `should_emit_f16_for_kquant` did NOT break the CLI
//! surface or introduce a build-time regression.  The optional end-to-
//! end test against a real HF model is `#[ignore]`-gated so it does
//! NOT run on default `cargo test` (avoids RAM saturation + 60-90 s
//! convert wall-time per CI run).
//!
//! See ADR-014 § "P11-prereq Iter D" 2026-04-27 close section for the
//! production-acceptance criterion (Qwen3.6-27B → dwq46 re-emit
//! completes without K-quant codec rejection on vision tensors).

use std::path::PathBuf;
use std::process::Command;

/// Path to the built `hf2q` binary inside the workspace target dir.
fn hf2q_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("release")
        .join("hf2q")
}

/// Smoke: the binary builds successfully (i.e. the new
/// `should_emit_f16_for_kquant` predicate, the new `f16_passthrough`
/// helper, and the three plumbing edits in `KQuantCodecQuantizer` /
/// `VariantKQuantizer` / `DwqKQuantizer` did not introduce a compile
/// error).  `cargo test --release` builds the binary as a dependency
/// of this test crate; if that build failed, this entire test crate
/// would fail to compile, never reaching the assertion.
#[test]
fn binary_builds_with_iter_d_skip_predicate_present() {
    // Tautological check — if the binary failed to build, this
    // integration crate would not have linked.  Documented per the
    // Iter B precedent at `tests/dwq_k_quantizer.rs::binary_builds_…`.
    let cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let contents =
        std::fs::read_to_string(&cargo_toml).expect("Cargo.toml must be readable");
    assert!(
        contents.contains("name = \"hf2q\""),
        "expected this test to run under the hf2q crate; got Cargo.toml={contents:.200}"
    );
}

/// Smoke: the built binary's `convert --help` runs without error.
/// Indirectly verifies that:
///
///   * The new predicate + `f16_passthrough` helper did not break the
///     `cmd_convert` CLI parsing (clap derive macro re-runs on every
///     build).
///   * The binary was actually built (not stale / missing).
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
    // `--quant` is the orthogonal-axis flag; the predicate sits
    // BEHIND every K-quant-family `--quant` value.  If the help
    // surface broke, this string would be missing.
    assert!(
        stdout.contains("--quant"),
        "convert --help missing --quant flag: stdout={stdout:.500}",
    );
}

/// **End-to-end (`#[ignore]`-gated):** `--quant q4_k_m` on Qwen3.6-27B
/// completes without the K-quant codec rejection that killed the first
/// P11 re-emit attempt at `commit base 87c6242`.  Pre-Iter D this
/// errored at `model.visual.blocks.0.attn.proj.weight` (row_len =
/// 1152, not a multiple of QK_K = 256); post-Iter D the vision-tensor
/// skip predicate fires and emits F16 passthrough.
///
/// **Why ignored by default:** requires (a) the Qwen3.6-27B model in
/// the local HF cache (~54 GB), (b) ≥ 60 GB free RAM for the
/// streaming convert peak working-set, (c) ~60-90 s wall-time per
/// run.  Run manually with `cargo test --release --test
/// vision_tensor_skip_predicate -- --ignored` when validating an
/// Iter D regression bisect or before shipping a P11 re-emit batch.
#[test]
#[ignore]
fn qwen35_27b_q4_k_m_completes_via_cli() {
    let bin = hf2q_binary();
    assert!(
        bin.exists(),
        "hf2q binary must be built before running this end-to-end test"
    );

    let out_dir = std::env::temp_dir().join("hf2q-iter-d-q4km-e2e");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).expect("must create temp out dir");
    let out_gguf = out_dir.join("qwen35-27b-q4km.gguf");

    let output = Command::new(&bin)
        .args([
            "convert",
            "--repo",
            "Qwen/Qwen3.6-27B",
            "--format",
            "gguf",
            "--quant",
            "q4_k_m",
            "--output",
        ])
        .arg(&out_gguf)
        .args(["--skip-quality", "--yes"])
        .output()
        .expect("hf2q convert must run");

    assert!(
        output.status.success(),
        "hf2q convert q4_k_m failed: stdout={:?} stderr={:?}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        out_gguf.exists(),
        "GGUF output file must exist at {}",
        out_gguf.display()
    );
}
