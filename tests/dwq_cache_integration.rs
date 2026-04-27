//! ADR-014 P5 cache wiring smoke.
//!
//! hf2q is currently a binary-only crate, so external integration tests cannot
//! import `DwqCalibrator` or `MockActivationCapture` directly. The live
//! forward-pass counter assertion lives in
//! `src/calibrate/dwq_calibrator.rs::tests::dwq_cache_hit_skips_forward_pass_same_key`.
//! This gated integration smoke runs that bin-target test through Cargo, which
//! exercises the same compiled binary target without opening private modules.

use std::process::Command;

const ENV_GATE: &str = "HF2Q_DWQ_CACHE_INTEGRATION";

#[test]
fn dwq_cache_hit_counter_contract_from_bin_target() {
    if std::env::var(ENV_GATE).as_deref() != Ok("1") {
        eprintln!(
            "[skip] dwq cache integration smoke; set {}=1 to run the bin-target counter test",
            ENV_GATE
        );
        return;
    }

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .args([
            "test",
            "--bin",
            "hf2q",
            "--release",
            "dwq_cache_hit_skips_forward_pass_same_key",
            "--",
            "--exact",
            "--test-threads=1",
        ])
        .output()
        .expect("spawn cargo test --bin hf2q dwq_cache_hit_skips_forward_pass_same_key");

    assert!(
        output.status.success(),
        "dwq cache bin-target counter test failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
