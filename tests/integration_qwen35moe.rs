//! Integration tests for end-to-end Qwen3.5-MoE / Qwen3.6-MoE inference
//! via `hf2q generate` (ADR-013 Phase P13.4).
//!
//! Real-model test, requires the 25 GB apex GGUF on disk; opt-in via `--ignored`.
//! Skipped cleanly (with eprintln + Ok(())) when the GGUF is not present so CI
//! and other-machine runs don't false-fail.
//!
//! Invocation:
//!   cargo test --release -- --ignored qwen35moe
//!
//! The reference fixture path is the local apex GGUF emitted by Robert's
//! externally-converted Qwen3.6-35B-A3B-Abliterix model (see ADR-013 §Context).

use std::path::Path;
use std::process::Command;

const APEX_GGUF: &str =
    "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf";

/// Locate the release `hf2q` binary inside the active workspace (handles both
/// the main checkout and worktrees by walking up from `CARGO_MANIFEST_DIR`).
fn hf2q_release_bin() -> std::path::PathBuf {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("target").join("release").join("hf2q")
}

#[test]
#[ignore = "Real-model test, requires 25 GB apex GGUF on disk; opt-in via --ignored"]
fn qwen35moe_apex_generate_smoke() {
    if !Path::new(APEX_GGUF).exists() {
        eprintln!(
            "skip: apex GGUF not found at {APEX_GGUF}; this test only runs on machines with the model staged."
        );
        return;
    }

    let bin = hf2q_release_bin();
    if !bin.exists() {
        eprintln!(
            "skip: hf2q release binary not found at {}; run `cargo build --release` first.",
            bin.display()
        );
        return;
    }

    // Greedy (T=0), 8 tokens — minimum signal that prefill + decode + sampler
    // and the qwen35moe arch dispatch all wire correctly.
    let output = Command::new(&bin)
        .args([
            "generate",
            "--model",
            APEX_GGUF,
            "--prompt",
            "Hello",
            "--max-tokens",
            "8",
            "--temperature",
            "0",
        ])
        .output()
        .expect("failed to invoke hf2q generate");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "hf2q generate exited non-zero on apex GGUF (qwen35moe).\nstatus: {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status.code()
    );

    // Generated text lands on stdout (after the 4-line hf2q header). Some
    // bytes must be emitted; an empty body would mean the decode loop
    // produced nothing.
    assert!(
        !stdout.trim().is_empty(),
        "hf2q generate produced empty stdout on apex GGUF.\nstderr:\n{stderr}"
    );

    // The qwen35 dispatcher emits a tok/s footer to stderr in non-benchmark
    // mode (`--- mlx-native (Qwen3.5): N tokens in Xs (Y tok/s) ---`).
    assert!(
        stderr.contains("tok/s"),
        "hf2q generate stderr missing tok/s footer (decode path may not have completed).\nstderr:\n{stderr}"
    );
}
