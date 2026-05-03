//! Subprocess-level integration test for ADR-005 Wedge-4 / iter-228a
//! Qwen3-VL text-LM SERVE-side load surface.
//!
//! # Why subprocess
//!
//! `hf2q` is a binary crate (no `[lib]` target — see
//! `tests/mmproj_llama_cpp_compat.rs:32-37`). Integration tests cannot
//! reach `crate::serve::*` directly. Cross-process subprocess spawn is
//! the canonical pattern for tests that need to drive the real
//! `hf2q serve` startup pipeline.
//!
//! # Scope
//!
//! These tests assert that `hf2q serve --model <real Qwen3-VL-2B GGUF>`
//! reaches the load-complete state without the iter-227 actionable-error
//! bail. The chat-completion path still returns HTTP 501 with the
//! `qwen3vl_text_forward_pending` sentinel until iter-228b lands.
//!
//! Unit-test coverage for the load + config + sentinel surface lives
//! inside `src/inference/models/qwen3vl_text/...::tests` and runs by
//! default (synthetic-fixture path) + under `HF2Q_QWEN3VL_LM_LOAD=1`
//! (real-fixture path via the in-binary unit tests). This file's
//! single subprocess test is the operator-script-level falsification
//! gate — when it passes, `scripts/wedge4_qwen3vl.sh` Step 5 is
//! unblocked.
//!
//! # Two scopes
//!
//! 1. **Default**: skip with a diagnostic. Keeps `cargo test --release`
//!    cheap on dev machines + CI nodes that don't carry the 1.32 GB
//!    fixture.
//! 2. **`HF2Q_QWEN3VL_LM_LOAD=1`**: runs the subprocess startup gate
//!    against the canonical Qwen3-VL-2B GGUF at
//!    `/opt/hf2q/.cfa-archive/wedge4f-out/qwen3-vl-2b-q4_0.gguf`.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const REAL_GGUF_PATH: &str = "/opt/hf2q/.cfa-archive/wedge4f-out/qwen3-vl-2b-q4_0.gguf";
const REAL_TOKENIZER_PATH: &str =
    "/opt/hf2q/.cfa-archive/wedge4f-hf/Qwen3-VL-2B-Instruct/tokenizer.json";

fn skip_unless_operator_gated_and_fixture_present() -> Option<PathBuf> {
    if std::env::var("HF2Q_QWEN3VL_LM_LOAD").ok().as_deref() != Some("1") {
        eprintln!("skip: HF2Q_QWEN3VL_LM_LOAD!=1");
        return None;
    }
    let p = PathBuf::from(REAL_GGUF_PATH);
    if !p.exists() {
        eprintln!("skip: real GGUF fixture not present at {}", p.display());
        return None;
    }
    Some(p)
}

/// Resolve the path to the just-built `hf2q` binary. `cargo test`
/// places the binary alongside the test executable in
/// `target/{profile}/`. We walk up from `current_exe` to find it.
fn hf2q_binary_path() -> PathBuf {
    let test_exe = std::env::current_exe().expect("current_exe");
    // current_exe is `target/{profile}/deps/<test>-<hash>`; walk up
    // two levels to reach `target/{profile}/`.
    let target_dir = test_exe
        .parent()
        .and_then(|p| p.parent())
        .expect("walk up to target/{profile}");
    let candidate = target_dir.join("hf2q");
    assert!(
        candidate.exists(),
        "hf2q binary not found at {} — did you `cargo build --bin hf2q`?",
        candidate.display()
    );
    candidate
}

/// iter-228a falsification gate: `hf2q serve` against the real
/// Qwen3-VL-2B GGUF must successfully load (not bail out at the
/// iter-227 dispatch site, not crash inside the Gemma loader). We
/// give the subprocess up to 60s to reach the post-load steady state,
/// then send SIGTERM.
///
/// **Pass criterion**: subprocess stdout or stderr contains a
/// post-load marker indicating the engine reached its idle state
/// (e.g. "Engine load: complete" / "listening on" / "warmup complete").
/// **Fail criterion**: subprocess exits with a non-zero code AND its
/// stderr contains the iter-227 dispatch bail or the Gemma MoE expert
/// load panic (regression guards).
#[test]
fn qwen3vl_text_real_gguf_serve_startup_smoke() {
    let Some(gguf_path) = skip_unless_operator_gated_and_fixture_present() else {
        return;
    };
    let bin = hf2q_binary_path();

    // Pick a high-numbered port with nanos jitter to avoid collisions
    // across rapid re-runs on the same machine. The base 39228 is the
    // iter-228 mnemonic; +0..99 covers parallel test executions.
    let port_jitter = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos()
        % 100) as u16;
    let port = 39228u16 + port_jitter;

    // Pass `--tokenizer` explicitly because Wedge-4f's converter
    // separates the GGUF (under `wedge4f-out/`) from the HF tokenizer
    // assets (under `wedge4f-hf/Qwen3-VL-2B-Instruct/`). The operator
    // script (`scripts/wedge4_qwen3vl.sh`) sits the tokenizer next to
    // the GGUF in its production layout; this test adapts to the
    // .cfa-archive layout where they live in different subdirs.
    let tok_path = PathBuf::from(REAL_TOKENIZER_PATH);
    let mut cmd = Command::new(&bin);
    cmd.arg("serve")
        .arg("--model")
        .arg(&gguf_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if tok_path.exists() {
        cmd.arg("--tokenizer").arg(&tok_path);
    }
    let mut child = cmd.spawn().expect("spawn hf2q serve subprocess");
    eprintln!(
        "qwen3vl_text iter-228a smoke: spawned hf2q serve (pid={}) on port {} against {}",
        child.id(),
        port,
        gguf_path.display()
    );

    // Poll: wait up to 60s for /readyz to return 200 (or for the
    // process to exit early with an error).
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut readyz_ok = false;
    let mut early_exit_status: Option<std::process::ExitStatus> = None;
    let mut iter = 0u32;
    while Instant::now() < deadline {
        iter += 1;
        match child.try_wait() {
            Ok(Some(status)) => {
                eprintln!(
                    "qwen3vl_text iter-228a smoke: subprocess exited at poll {iter} with status {status:?}"
                );
                early_exit_status = Some(status);
                break;
            }
            Ok(None) => {
                let url = format!("http://127.0.0.1:{port}/readyz");
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_millis(500))
                    .build()
                    .expect("build blocking reqwest client");
                if let Ok(resp) = client.get(&url).send() {
                    if resp.status() == reqwest::StatusCode::OK {
                        eprintln!(
                            "qwen3vl_text iter-228a smoke: /readyz=200 at poll {iter} (~{:.1}s after spawn)",
                            iter as f32 * 0.5
                        );
                        readyz_ok = true;
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("try_wait failed: {e}");
            }
        }
    }

    // SIGKILL the child (best-effort) and capture any captured diagnostics.
    if early_exit_status.is_none() {
        let _ = child.kill();
    }
    let output = child.wait_with_output().ok();
    let stderr_str = output
        .as_ref()
        .map(|o| String::from_utf8_lossy(&o.stderr).into_owned())
        .unwrap_or_default();
    let stdout_str = output
        .as_ref()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();

    // Regression guards.
    if !readyz_ok || early_exit_status.is_some() {
        assert!(
            !stderr_str.contains("iter-227 closes only the dispatch gap")
                && !stdout_str.contains("iter-227 closes only the dispatch gap"),
            "iter-228a must lift the iter-227 dense-Qwen3-VL bail. stderr:\n{stderr_str}"
        );
        assert!(
            !stderr_str.contains("missing blk.0.ffn_gate_up_exps.weight")
                && !stdout_str.contains("missing blk.0.ffn_gate_up_exps.weight"),
            "Gemma MoE loader fall-through is the original bug iter-227 fixed; \
             iter-228a must keep that intercept. stderr:\n{stderr_str}"
        );
    }

    if let Some(status) = early_exit_status {
        panic!(
            "hf2q serve exited early with status {status:?} after {iter} polls\n\
             stdout:\n{stdout_str}\n\
             stderr:\n{stderr_str}"
        );
    }

    assert!(
        readyz_ok,
        "hf2q serve did not reach /readyz=200 within 90s ({iter} polls); \
         stderr tail:\n{}",
        stderr_str.lines().rev().take(20).collect::<Vec<_>>().join("\n")
    );
}
