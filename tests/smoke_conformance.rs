//! Integration tests for `hf2q smoke` (ADR-012 Decision 16).
//!
//! These tests exercise the smoke subcommand's preflight surface end-to-end
//! through the cargo-built `hf2q` binary, asserting:
//!
//!   * Each preflight failure mode produces a distinct non-zero exit code.
//!   * Unknown arches fail uniformly with a structured error.
//!   * `--dry-run` short-circuits before any convert / inference work.
//!
//! No tests here download from HuggingFace, no tests here invoke a real
//! `llama-cli` — the harness exercises the smoke gate itself, not the
//! downstream tools.  Real-model transcripts live under
//! `tests/fixtures/smoke-transcripts/` and are produced by an out-of-band
//! `hf2q smoke --arch X --quant Y` invocation; they are committed once a
//! human has run them on hardware.

use assert_cmd::Command;
use predicates::prelude::*;

/// Sequential lock — the binary inherits the parent's environment, so
/// HF_TOKEN manipulation must serialize.
fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn smoke_cmd() -> Command {
    Command::cargo_bin("hf2q").expect("hf2q binary built")
}

#[test]
fn unknown_arch_fails_uniformly() {
    // Both a typo and an arch from a future ADR (gemma4, ministral, deepseekv3)
    // hit the same dispatch error — proves Decision 20's load-bearing registry.
    for bogus in &["bogus", "gemma4", "ministral", "deepseekv3"] {
        let _g = env_lock();
        let assert = smoke_cmd()
            .env("HF_TOKEN", "test-token")
            .args(["smoke", "--arch", bogus, "--dry-run"])
            .assert()
            .failure();
        let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_string();
        assert!(
            stderr.contains("unknown arch"),
            "missing 'unknown arch' for {bogus}; stderr was: {stderr}"
        );
        assert!(
            stderr.contains("qwen35"),
            "missing known arch list for {bogus}; stderr was: {stderr}"
        );
    }
}

#[test]
fn missing_hf_token_returns_exit_2() {
    let _g = env_lock();
    smoke_cmd()
        .env_remove("HF_TOKEN")
        .args([
            "smoke",
            "--arch",
            "qwen35",
            "--dry-run",
            "--llama-cli",
            "/usr/bin/true",
            "--hf2q-binary",
            "/usr/bin/true",
        ])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("HF_TOKEN"));
}

#[test]
fn missing_llama_cli_returns_exit_4() {
    let _g = env_lock();
    smoke_cmd()
        .env("HF_TOKEN", "test-token")
        .args([
            "smoke",
            "--arch",
            "qwen35",
            "--dry-run",
            "--llama-cli",
            "/path/that/does/not/exist/llama-cli",
            "--hf2q-binary",
            "/usr/bin/true",
        ])
        .assert()
        .code(4)
        .stderr(predicate::str::contains("llama-cli not found"));
}

#[test]
fn missing_hf2q_binary_returns_exit_5() {
    let _g = env_lock();
    smoke_cmd()
        .env("HF_TOKEN", "test-token")
        .args([
            "smoke",
            "--arch",
            "qwen35",
            "--dry-run",
            "--llama-cli",
            "/usr/bin/true",
            "--hf2q-binary",
            "/path/that/does/not/exist/hf2q",
        ])
        .assert()
        .code(5)
        .stderr(predicate::str::contains("hf2q binary not found"));
}

#[test]
fn vision_request_on_moe_arch_fails_uniformly() {
    // qwen35moe has has_vision: false (vision_config dropped from the MoE
    // target).  The error message must explicitly call out the unsupported
    // path so users don't think it's a transient bug.
    let _g = env_lock();
    smoke_cmd()
        .env("HF_TOKEN", "test-token")
        .args([
            "smoke",
            "--arch",
            "qwen35moe",
            "--with-vision",
            "--dry-run",
            "--llama-cli",
            "/usr/bin/true",
            "--hf2q-binary",
            "/usr/bin/true",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("vision smoke requested"));
}

#[test]
fn dry_run_with_full_preflight_green_succeeds() {
    // Every preflight gate green; --dry-run short-circuits before convert.
    // Note: the real disk preflight may still fail on the test host if free
    // space is below the arch floor, so we accept any of:
    //   * exit 0 (real pass)
    //   * exit 3 (insufficient disk on the test host)
    //   * exit 6 (huggingface-cli unavailable / repo not resolvable)
    // The test's value is asserting that we *don't* trip exit 2/4/5 — those
    // were proven absent by the upstream args.  (The clean-pass path is
    // covered by the in-process `arch::smoke::tests` set.)
    let _g = env_lock();
    let output = smoke_cmd()
        .env("HF_TOKEN", "test-token")
        .args([
            "smoke",
            "--arch",
            "qwen35",
            "--dry-run",
            "--llama-cli",
            "/usr/bin/true",
            "--hf2q-binary",
            "/usr/bin/true",
        ])
        .output()
        .expect("ran hf2q smoke");
    let code = output.status.code().unwrap_or(-1);
    assert!(
        matches!(code, 0 | 3 | 6),
        "expected 0/3/6 from green preflight (depends on host disk + HF reachability); got {code}, stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn smoke_help_lists_subcommand() {
    smoke_cmd()
        .args(["smoke", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--arch"))
        .stdout(predicate::str::contains("--quant"))
        .stdout(predicate::str::contains("--dry-run"));
}

#[test]
fn known_arches_list_matches_adr012_p8_scope() {
    // Decision 20 acceptance: ADR-012 P8 ships exactly two registry entries.
    // Any third entry landing in this ADR is a scope violation; landing one
    // in a separate follow-up ADR is fine.  This test prevents accidental
    // expansion of the ADR-012 surface.
    let _g = env_lock();
    let assert = smoke_cmd()
        .env("HF_TOKEN", "test-token")
        .args(["smoke", "--arch", "bogus", "--dry-run"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_string();
    // Expect "known arches: qwen35, qwen35moe" verbatim.  If a future arch
    // lands in this ADR by mistake, this assertion catches it.
    assert!(
        stderr.contains("qwen35, qwen35moe"),
        "expected exactly two known arches in this ADR; stderr was: {stderr}"
    );
    assert!(
        !stderr.contains("gemma4"),
        "gemma4 must not be registered in ADR-012; landing it requires its own ADR"
    );
    assert!(
        !stderr.contains("ministral"),
        "ministral must not be registered in ADR-012 (ADR-015)"
    );
    assert!(
        !stderr.contains("deepseekv3"),
        "deepseekv3 must not be registered in ADR-012 (ADR-016)"
    );
}
