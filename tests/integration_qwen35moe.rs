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

const APEX_GGUF: &str = "/opt/hf2q/models/qwen3.6/APEX-Q5_K_M.gguf";

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
    if let Err(e) = mlx_native::MlxDevice::new() {
        eprintln!("skip: no Metal device available for qwen35moe generate smoke: {e}");
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

#[test]
#[ignore = "Real-model test, requires 25 GB apex GGUF on disk; opt-in via --ignored"]
fn qwen35moe_apex_long_prefill_executes_native_expert_mm() {
    if !Path::new(APEX_GGUF).exists() {
        eprintln!(
            "skip: apex GGUF not found at {APEX_GGUF}; this test only runs on machines with the model staged."
        );
        return;
    }
    if let Err(e) = mlx_native::MlxDevice::new() {
        eprintln!("skip: no Metal device available for qwen35moe long-prefill gate: {e}");
        return;
    }

    let bin = hf2q_release_bin();
    assert!(
        bin.exists(),
        "hf2q release binary not found at {}; run `cargo build --release` first.",
        bin.display()
    );
    let prompt = "cobalt amber cedar river mountain orchard lantern compass harbor meadow \
        copper silver granite willow maple ocean valley forest sunrise sunset glacier thunder \
        breeze cloud pebble canyon bridge garden falcon heron otter badger walnut chestnut \
        violet indigo crimson scarlet bronze marble quartz linen cotton velvet paper pencil \
        window doorway staircase rooftop chimney village market bakery library workshop station \
        engine signal circuit tensor vector matrix kernel token context prompt answer";
    let output = Command::new(&bin)
        .env("HF2Q_LOG_MM_ID_ROUTE", "1")
        .args([
            "generate",
            "--model",
            APEX_GGUF,
            "--prompt",
            prompt,
            "--max-tokens",
            "1",
            "--temperature",
            "0",
        ])
        .output()
        .expect("failed to invoke hf2q long-prefill gate");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "hf2q long-prefill gate exited non-zero.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let prefill_tokens = stdout
        .lines()
        .find_map(|line| {
            line.strip_prefix("prefill: ")?
                .split_whitespace()
                .next()?
                .parse::<u32>()
                .ok()
        })
        .expect("long-prefill gate did not report a prefill token count");
    assert!(
        prefill_tokens > 33,
        "long-prefill gate must cross the expert mm_id boundary, got {prefill_tokens} tokens"
    );

    let q5_gate_up = stderr.lines().any(|line| {
        line.contains("dispatch_id_mm_pooled engaged: type=Q5_K")
            && line.contains("top_k=8 k=2048 n=512 n_experts=256")
    });
    let q6_down = stderr.lines().any(|line| {
        line.contains("dispatch_id_mm_pooled engaged: type=Q6_K")
            && line.contains("top_k=1 k=512 n=2048 n_experts=256")
    });
    assert!(
        q5_gate_up,
        "Q5_K gate/up did not execute the pooled expert mm_id route.\nstderr:\n{stderr}"
    );
    assert!(
        q6_down,
        "Q6_K down did not execute the pooled expert mm_id route.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("tok/s"),
        "long-prefill decode did not complete.\nstderr:\n{stderr}"
    );
}
