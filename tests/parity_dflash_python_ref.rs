//! Numerical-parity test for DFlash draft forward — ADR-034 G2 gate.
//!
//! Loads .npy dumps produced by `scripts/parity/dflash_parity.py`
//! (wrapping `/opt/dflash/dflash/model_mlx.py` 582-LOC reference) and
//! compares against `src/inference/spec_decode/dflash/forward.rs`
//! (7011-LOC scaffold, never validated against this reference per
//! ADR-034 §1.2 audit).
//!
//! ## Status (2026-05-21)
//!
//! 🟡 **Scaffold** — skips when fixtures absent. Re-enables when:
//!   1. `scripts/parity/dflash_parity.py` fully implemented
//!   2. z-lab DFlash drafter checkpoint + target safetensors downloaded
//!   3. Reference dumps produced into `tests/parity_fixtures/dflash/`
//!
//! ## Why this gate is critical
//!
//! Per ADR-034 §1.2 audit: the 7011 LOC Rust scaffold was written under
//! ADR-030 (proposed) and has NEVER been validated against the Python
//! reference. Comments throughout `src/inference/spec_decode/dflash/`
//! claim "Mirrors model_mlx.py:..." — but the claim is documentary, not
//! tested. This gate is what converts the documentation into a verified
//! invariant.
//!
//! ## Threshold ladder
//!
//! - F32 reference vs Metal BF16: max-abs-diff < 1e-3
//! - F32 reference vs Metal Q4_K_M: max-abs-diff < 5e-2

use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/parity_fixtures/dflash")
}

#[test]
fn dflash_parity_against_python_reference_2026_05_21() {
    let dir = fixtures_dir();
    if !dir.exists() {
        eprintln!(
            "skipping: tests/parity_fixtures/dflash/ does not exist.\n\
             To enable this test:\n\
             1. Implement scripts/parity/dflash_parity.py (scaffold only as of 2026-05-21)\n\
             2. Download z-lab/Qwen3.6-27B-DFlash (or similar) drafter\n\
             3. Download Qwen 3.6 27B target safetensors\n\
             4. Run: python3 scripts/parity/dflash_parity.py <target> <drafter> tests/parity_fixtures/dflash/\n\
             5. Re-run: cargo test --release dflash_parity_against_python_reference\n\
             \n\
             See ADR-034 §P4 + scripts/parity/README.md.\n"
        );
        return;
    }

    // When fixtures exist, the implementation will:
    // 1. Load target_hidden_states.npy, drafter_input_concat.npy
    // 2. Load drafter weights (DFlashWeights) from the matched drafter dir
    // 3. Run dispatch_dflash_model_forward with the captured target hidden states
    // 4. Compare drafter_logits.npy vs Rust output
    // 5. Also compare per-layer outputs for divergence localization
    //
    // The 20+ dispatch_dflash_* primitives in forward.rs decompose
    // model_mlx.py's __call__ methods — verify they compose to the
    // same end-to-end result.

    eprintln!("TODO: implement when scaffold dependencies are unblocked");
}
