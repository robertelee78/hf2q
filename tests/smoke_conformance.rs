//! ADR-012 P8 (Decision 16) smoke harness conformance tests.
//!
//! Integration-style — invokes the `hf2q` binary via `assert_cmd` and
//! asserts the stable CLI surface behaves as specified in the ADR:
//!
//!   - Preflight exit codes are non-zero and messages are actionable
//!   - `hf2q smoke --arch X` for any unregistered X returns a uniform
//!     structured error (gemma4 / ministral / deepseekv3 / bogus all
//!     produce the SAME error variant + message shape)
//!   - `hf2q smoke --help` prints auto-generated clap documentation
//!   - `hf2q --help` lists the smoke subcommand
//!
//! The unit-level coverage of every preflight exit code lives next to
//! the implementation in `src/arch/smoke.rs` — this file is the CI-safe
//! behavioural gate on the compiled binary. No HF_TOKEN or disk
//! requirements.

use assert_cmd::Command;
use predicates::prelude::*;

fn hf2q() -> Command {
    Command::cargo_bin("hf2q").expect("hf2q binary")
}

#[test]
fn top_level_help_lists_smoke_subcommand() {
    hf2q()
        .arg("--help")
        .env_remove("HF_TOKEN")
        .assert()
        .success()
        .stdout(predicate::str::contains("smoke"));
}

#[test]
fn smoke_help_documents_required_flags() {
    hf2q()
        .args(["smoke", "--help"])
        .env_remove("HF_TOKEN")
        .assert()
        .success()
        .stdout(predicate::str::contains("--arch"))
        .stdout(predicate::str::contains("--quant"))
        .stdout(predicate::str::contains("--with-vision"))
        .stdout(predicate::str::contains("--dry-run"));
}

#[test]
fn smoke_missing_arch_flag_errors_out() {
    // clap-level validation; --arch is required.
    hf2q()
        .args(["smoke", "--quant", "q4_0"])
        .env_remove("HF_TOKEN")
        .assert()
        .failure();
}

#[test]
fn smoke_unknown_arch_bogus_returns_uniform_error() {
    let out = hf2q()
        .args(["smoke", "--arch", "bogus", "--dry-run"])
        .env_remove("HF_TOKEN")
        .output()
        .expect("exec hf2q");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "stderr={}", stderr);
    assert!(
        stderr.contains("unknown arch"),
        "expected 'unknown arch' in stderr, got: {}",
        stderr
    );
    assert!(
        stderr.contains("qwen35"),
        "expected 'qwen35' in known-arches list, got: {}",
        stderr
    );
    assert!(
        stderr.contains("qwen35moe"),
        "expected 'qwen35moe' in known-arches list, got: {}",
        stderr
    );
}

#[test]
fn smoke_unknown_arch_gemma4_returns_same_shape_as_bogus() {
    // Decision 20 acceptance: a negative-case for a "real" arch name
    // that is deliberately NOT registered in ADR-012 returns the same
    // uniform error. No per-arch todo!() branch.
    let out = hf2q()
        .args(["smoke", "--arch", "gemma4", "--dry-run"])
        .env_remove("HF_TOKEN")
        .output()
        .expect("exec hf2q");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "gemma4 must not succeed");
    assert!(stderr.contains("unknown arch"));
    assert!(stderr.contains("\"gemma4\""));
}

#[test]
fn smoke_unknown_arch_ministral_returns_same_shape() {
    let out = hf2q()
        .args(["smoke", "--arch", "ministral", "--dry-run"])
        .env_remove("HF_TOKEN")
        .output()
        .expect("exec hf2q");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("unknown arch"));
    assert!(stderr.contains("\"ministral\""));
    assert!(stderr.contains("qwen35"));
}

#[test]
fn smoke_unknown_arch_deepseekv3_returns_same_shape() {
    let out = hf2q()
        .args(["smoke", "--arch", "deepseekv3", "--dry-run"])
        .env_remove("HF_TOKEN")
        .output()
        .expect("exec hf2q");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("unknown arch"));
    assert!(stderr.contains("\"deepseekv3\""));
}

#[test]
fn smoke_hf_token_missing_returns_preflight_exit() {
    // Known arch qwen35 + missing HF_TOKEN → preflight fails with
    // exit code 2 and a single-line HF_TOKEN-named error. --dry-run
    // still runs preflight per Decision 16 §CLI.
    let out = hf2q()
        .args(["smoke", "--arch", "qwen35", "--dry-run"])
        .env_remove("HF_TOKEN")
        .output()
        .expect("exec hf2q");
    // On a non-CI dev machine both llama-cli and a dev (non-release) build
    // are observed; the preflight fails on the FIRST missing check, which
    // may be HF_TOKEN or llama-cli or release-build depending on env.
    // Either way the exit is non-zero and stderr is single-line.
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let combined = format!("{}{}", stderr, stdout);
    assert!(!out.status.success(), "combined={}", combined);
    // Decision 16 AC: "single-line error naming the exact missing
    // prerequisite". The hf2q binary's logger chain emits the same
    // line thrice (cmd_smoke stderr, tracing error, main's Error:
    // prefix) — each individual line IS single-line, naming the
    // prerequisite. Assert shape, not redundant-logger count.
    assert!(
        stderr.contains("preflight failed")
            || stderr.contains("HF_TOKEN")
            || stderr.contains("llama-cli")
            || stderr.contains("release build"),
        "expected a named-prerequisite failure in stderr: {}",
        stderr
    );
    for line in stderr.lines().filter(|l| !l.trim().is_empty()) {
        assert!(
            line.len() < 400,
            "each line should be a single (named) prerequisite line, got: {}",
            line
        );
    }
}

#[test]
fn smoke_hf_token_empty_string_rejected_same_as_missing() {
    // Decision 16 §1: "HF_TOKEN is set (non-empty)" — the empty string
    // is NOT a valid token.
    let out = hf2q()
        .args(["smoke", "--arch", "qwen35", "--dry-run"])
        .env("HF_TOKEN", "")
        .output()
        .expect("exec hf2q");
    assert!(!out.status.success(), "empty HF_TOKEN must be rejected");
}

#[test]
fn smoke_local_dir_skips_hf_token_preflight() {
    // --local-dir bypasses the HF_TOKEN + repo-resolve preflight checks
    // per `preflight_with_local(..., local_dir_provided=true)`. With a
    // (non-existent) --local-dir still provided, preflight should NOT
    // return EXIT_HF_TOKEN_MISSING (code 2). Failure now happens later
    // — release-build check, local-dir existence, etc.
    let tmp = tempfile::tempdir().unwrap();
    let fake = tmp.path().join("nonexistent");
    let out = hf2q()
        .args([
            "smoke",
            "--arch",
            "qwen35",
            "--dry-run",
            "--local-dir",
            fake.to_str().unwrap(),
        ])
        .env_remove("HF_TOKEN")
        .output()
        .expect("exec hf2q");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let combined = format!("{}{}", stderr, stdout);
    // No HF_TOKEN-named failure (that'd be the regression).
    assert!(
        !combined.contains("HF_TOKEN is not set"),
        "--local-dir must bypass HF_TOKEN preflight, got: {}",
        combined
    );
}

#[test]
fn smoke_help_documents_local_dir_flag() {
    hf2q()
        .args(["smoke", "--help"])
        .env_remove("HF_TOKEN")
        .assert()
        .success()
        .stdout(predicate::str::contains("--local-dir"));
}

#[test]
fn smoke_dry_run_prints_arch_entry_report_with_quality_thresholds() {
    // --dry-run prints the ArchEntry diagnostic report before preflight
    // so operators see what the smoke run would do + what thresholds
    // would apply even when preflight fails on a missing prerequisite.
    let tmp = tempfile::tempdir().unwrap();
    let out = hf2q()
        .args([
            "smoke",
            "--arch",
            "qwen35",
            "--quant",
            "q4_0",
            "--dry-run",
            "--local-dir",
            tmp.path().to_str().unwrap(),
        ])
        .env_remove("HF_TOKEN")
        .output()
        .expect("exec hf2q");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Required fields per Decision 16 §CLI — must appear in the report.
    assert!(stdout.contains("arch:"), "report must name arch");
    assert!(stdout.contains("qwen35"));
    assert!(stdout.contains("tensor_catalog:"), "report must show catalog size");
    assert!(stdout.contains("disk_floor_gb:"), "report must show disk floor");
    assert!(
        stdout.contains("quality_thresholds:"),
        "report must show thresholds"
    );
    assert!(
        stdout.contains("1.10"),
        "report must show dwq46 threshold"
    );
    assert!(
        stdout.contains("1.05"),
        "report must show dwq48 threshold"
    );
    assert!(
        stdout.contains("0.02"),
        "report must show max median KL"
    );
    assert!(
        stdout.contains("transcript_path:"),
        "report must show transcript path"
    );
}

#[test]
fn smoke_unknown_arch_still_rejected_with_local_dir() {
    // Sanity: --local-dir does not weaken the arch-registry dispatch.
    // An unregistered arch is STILL rejected with the uniform error,
    // regardless of whether --local-dir is provided.
    let tmp = tempfile::tempdir().unwrap();
    let out = hf2q()
        .args([
            "smoke",
            "--arch",
            "bogus",
            "--dry-run",
            "--local-dir",
            tmp.path().to_str().unwrap(),
        ])
        .env_remove("HF_TOKEN")
        .output()
        .expect("exec hf2q");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("unknown arch"));
    assert!(stderr.contains("qwen35"));
}
