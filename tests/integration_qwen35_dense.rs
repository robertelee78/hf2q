//! Integration tests for end-to-end Qwen3.5 dense / Qwen3.6 dense inference
//! via `hf2q generate` (ADR-013 Phase P13.4 — symmetric dense companion to
//! `integration_qwen35moe.rs`).
//!
//! Real-model test, requires a Qwen3.5/3.6 *dense* (`general.architecture =
//! "qwen35"`, NOT `"qwen35moe"`) GGUF on disk; opt-in via `--ignored`.
//! Skipped cleanly when no dense GGUF is staged (the only Qwen3.5 GGUF
//! currently on Robert's M5 Max is the apex MoE; the dense covers the 27B
//! `qwen35` arch path that lands in the same `cmd_generate_qwen35`
//! dispatcher in `src/serve/mod.rs`).
//!
//! Override location with the env var `QWEN35_DENSE_GGUF=/abs/path/file.gguf`.
//! Without it, the test searches the conventional spot under
//! `/opt/hf2q/models/` and skips if no candidate is found.
//!
//! Invocation:
//!   cargo test --release -- --ignored qwen35_dense
//!   QWEN35_DENSE_GGUF=/path/to/qwen35-dense.gguf \
//!     cargo test --release -- --ignored qwen35_dense

use std::path::{Path, PathBuf};
use std::process::Command;

/// Walk `/opt/hf2q/models/` for a dense Qwen3.5 GGUF. Returns the first match
/// whose filename contains "qwen35" or "qwen3.5" but NOT "moe", "a3b", or
/// "a4b" (those are MoE markers — `a3b` = "active 3B params", `a4b` likewise).
fn find_dense_gguf() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("QWEN35_DENSE_GGUF") {
        let path = PathBuf::from(p);
        return path.is_file().then_some(path);
    }

    let root = Path::new("/opt/hf2q/models");
    if !root.is_dir() {
        return None;
    }

    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let name = dir.file_name()?.to_string_lossy().to_lowercase();
        let dense_arch_marker = name.contains("qwen35") || name.contains("qwen3.5");
        let moe_marker =
            name.contains("moe") || name.contains("-a3b") || name.contains("-a4b");
        if !dense_arch_marker || moe_marker {
            continue;
        }
        // Pick the largest .gguf in that dir as the model file.
        let mut best: Option<(PathBuf, u64)> = None;
        for f in std::fs::read_dir(&dir).ok()?.flatten() {
            let p = f.path();
            if p.extension().and_then(|s| s.to_str()) != Some("gguf") {
                continue;
            }
            let size = f.metadata().ok().map(|m| m.len()).unwrap_or(0);
            if best.as_ref().map(|(_, s)| size > *s).unwrap_or(true) {
                best = Some((p, size));
            }
        }
        if let Some((p, _)) = best {
            return Some(p);
        }
    }
    None
}

fn hf2q_release_bin() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("target").join("release").join("hf2q")
}

#[test]
#[ignore = "Real-model test, requires Qwen3.5 dense GGUF on disk; opt-in via --ignored"]
fn qwen35_dense_generate_smoke() {
    let gguf = match find_dense_gguf() {
        Some(p) => p,
        None => {
            eprintln!(
                "skip: no Qwen3.5/3.6 dense GGUF found under /opt/hf2q/models/ \
                 (or via QWEN35_DENSE_GGUF env var). The qwen35 dense arch path \
                 has no on-disk fixture on this machine — apex MoE is the only \
                 staged Qwen variant locally."
            );
            return;
        }
    };

    let bin = hf2q_release_bin();
    if !bin.exists() {
        eprintln!(
            "skip: hf2q release binary not found at {}; run `cargo build --release` first.",
            bin.display()
        );
        return;
    }

    let output = Command::new(&bin)
        .args([
            "generate",
            "--model",
            gguf.to_str().expect("gguf path is valid utf8"),
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
        "hf2q generate exited non-zero on Qwen3.5 dense GGUF.\ngguf: {}\nstatus: {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        gguf.display(),
        output.status.code(),
    );

    assert!(
        !stdout.trim().is_empty(),
        "hf2q generate produced empty stdout on Qwen3.5 dense GGUF.\ngguf: {}\nstderr:\n{stderr}",
        gguf.display(),
    );

    assert!(
        stderr.contains("tok/s"),
        "hf2q generate stderr missing tok/s footer for Qwen3.5 dense.\ngguf: {}\nstderr:\n{stderr}",
        gguf.display(),
    );
}
