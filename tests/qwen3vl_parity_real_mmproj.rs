//! ADR-005 iter-224 row 3 Wedge-4c.5 — Qwen3-VL real-mmproj parity scaffolding.
//!
//! Operator-gated harness for the eventual end-to-end byte-level
//! comparison between hf2q's Qwen3-VL ViT + LM-side DeepStack hooks
//! and llama.cpp's `llama-mtmd-cli` reference output. The full E2E
//! comparison (running both stacks against a shared image + prompt
//! and diffing token-1 logits at top-50 elements) cannot run in
//! ordinary CI because:
//!
//!   1. It requires a real Qwen3-VL GGUF (~3-15 GB) on disk;
//!   2. The model-load path is single-tenant under hf2q's standing
//!      OOM-prevention directive (one model-loading inference at a
//!      time), so two parallel cargo test threads holding both stacks
//!      would crash the host;
//!   3. Wedge-4d's handler-side preprocess (variable-resolution
//!      patch grid + 3D-mRoPE position synthesis) is not yet wired,
//!      so the image-bearing chat path returns 400 from
//!      `process_multimodal_content`. The full parity check requires
//!      that path to land first.
//!
//! Wedge-4c.5 status: this harness establishes the operator-gate
//! contract and the artefact-presence checks. The actual two-stack
//! comparison (POST `/v1/chat/completions` to a `hf2q serve` child
//! against `/path/to/qwen3vl.gguf` + parallel `llama-mtmd-cli` run +
//! top-50 logits diff) is pace-shifted to Wedge-4d/4e, gated on the
//! same env knobs. This file's tests verify:
//!
//!   * `qwen3vl_parity_artefacts_present_when_gate_set` (env-gated):
//!     when `HF2Q_QWEN3VL_PARITY=1` and the operator passes
//!     `--ignored`, validate that the four required artefact env
//!     vars point at existing files. Fails loud with the missing
//!     piece named so the operator gets an actionable message.
//!   * `parity_gate_only_fires_on_one_or_true`: regression-guard for
//!     the env-knob parser (no I/O).
//!
//! ## Operator recipe (when artefacts are available)
//!
//! ```bash
//! export HF2Q_QWEN3VL_PARITY=1
//! export HF2Q_QWEN3VL_GGUF=/path/to/qwen3-vl-2b-instruct.gguf
//! export HF2Q_QWEN3VL_MMPROJ=/path/to/qwen3-vl-2b-instruct-mmproj.gguf
//! export HF2Q_QWEN3VL_IMAGE=/path/to/parity_image.png
//! export HF2Q_QWEN3VL_PROMPT="Describe this image."
//! cargo test --test qwen3vl_parity_real_mmproj --release \
//!     -- --nocapture --ignored
//! ```
//!
//! The recipe runs ONLY when all four required env vars are set AND
//! the test is invoked with `--ignored` (the test is `#[ignore]` by
//! default). When `HF2Q_QWEN3VL_PARITY=1` is set but artefacts are
//! missing, the test reports the missing piece and exits cleanly so
//! the operator sees an actionable message.
//!
//! Wedge-4d/4e will graduate this harness from "scaffolding-only" to
//! a real top-50-logits diff against `llama-mtmd-cli` output, gated
//! on the same env knobs.

use std::path::PathBuf;

/// Return the path from `var_name` in env, or `None` if unset.
fn env_path(var_name: &str) -> Option<PathBuf> {
    std::env::var(var_name).ok().map(PathBuf::from)
}

/// Whether the operator-gate is set. `HF2Q_QWEN3VL_PARITY=1`.
fn parity_gate_set() -> bool {
    std::env::var("HF2Q_QWEN3VL_PARITY")
        .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Operator-gated harness driver. Default-skipped (`#[ignore]`); runs
/// only when the operator passes `--ignored` AND has set the env
/// gate `HF2Q_QWEN3VL_PARITY=1`.
///
/// **Wedge-4c.5 scope**: this is SCAFFOLDING. It validates that the
/// required env vars are set and that the fixture files exist on
/// disk; the actual two-stack comparison is Wedge-4d/4e (image-bearing
/// chat path + logits diff harness). When the gate is set with all
/// artefacts present, the test today reports "scaffolding ready;
/// Wedge-4e will land the diff" — a deliberate pacing decision
/// documented in the iter-224 row 3 plan.
#[test]
#[ignore]
fn qwen3vl_parity_artefacts_present_when_gate_set() {
    if !parity_gate_set() {
        eprintln!(
            "qwen3vl_parity_artefacts_present_when_gate_set: \
             HF2Q_QWEN3VL_PARITY not set — skipping (use `--ignored` AND set \
             HF2Q_QWEN3VL_PARITY=1 to run)"
        );
        return;
    }

    let gguf = env_path("HF2Q_QWEN3VL_GGUF");
    let mmproj = env_path("HF2Q_QWEN3VL_MMPROJ");
    let image = env_path("HF2Q_QWEN3VL_IMAGE");
    let prompt = std::env::var("HF2Q_QWEN3VL_PROMPT").ok();

    let mut missing: Vec<&str> = Vec::new();
    if gguf.is_none() {
        missing.push("HF2Q_QWEN3VL_GGUF");
    }
    if mmproj.is_none() {
        missing.push("HF2Q_QWEN3VL_MMPROJ");
    }
    if image.is_none() {
        missing.push("HF2Q_QWEN3VL_IMAGE");
    }
    if prompt.is_none() {
        missing.push("HF2Q_QWEN3VL_PROMPT");
    }
    assert!(
        missing.is_empty(),
        "qwen3vl_parity_artefacts_present_when_gate_set: HF2Q_QWEN3VL_PARITY=1 set but \
         the following env var(s) are missing: {missing:?}. \
         See tests/qwen3vl_parity_real_mmproj.rs operator recipe in the \
         module docstring."
    );

    let gguf = gguf.unwrap();
    let mmproj = mmproj.unwrap();
    let image = image.unwrap();
    assert!(
        gguf.exists(),
        "HF2Q_QWEN3VL_GGUF={} does not exist on disk",
        gguf.display()
    );
    assert!(
        mmproj.exists(),
        "HF2Q_QWEN3VL_MMPROJ={} does not exist on disk",
        mmproj.display()
    );
    assert!(
        image.exists(),
        "HF2Q_QWEN3VL_IMAGE={} does not exist on disk",
        image.display()
    );

    eprintln!(
        "qwen3vl_parity_artefacts_present_when_gate_set: scaffolding ready. \
         Wedge-4d will wire the image-bearing chat path; Wedge-4e will \
         add the top-50 logits diff vs llama-mtmd-cli. Today this \
         scaffolding only verifies that artefacts are present.\n\
         GGUF:    {}\n\
         MMPROJ:  {}\n\
         IMAGE:   {}\n\
         PROMPT:  {:?}",
        gguf.display(),
        mmproj.display(),
        image.display(),
        prompt.unwrap()
    );
}

/// Validates the env-knob taxonomy: the gate var must be exactly
/// "1" or "true" (case-insensitive); other values must NOT trigger
/// the gate. Belt-and-suspenders against typos like
/// `HF2Q_QWEN3VL_PARITY=yes` silently skipping or running.
#[test]
fn parity_gate_only_fires_on_one_or_true() {
    // We can't poke os env in a parallel-test-safe way, so this test
    // exercises the parser predicate against synthetic inputs.
    fn predicate(v: &str) -> bool {
        v.trim() == "1" || v.trim().eq_ignore_ascii_case("true")
    }
    assert!(predicate("1"));
    assert!(predicate(" 1 "));
    assert!(predicate("true"));
    assert!(predicate("TRUE"));
    assert!(predicate("True"));
    assert!(!predicate("yes"));
    assert!(!predicate("0"));
    assert!(!predicate(""));
    assert!(!predicate("false"));
}
