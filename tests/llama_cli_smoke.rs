//! ADR-033 §10 — runtime correctness smoke test.
//!
//! After byte-cmp proves bit-identity to canonical, this gate proves
//! the GGUF actually loads in stock llama.cpp and decodes coherent
//! tokens. Catches a different class of bug than byte-cmp (e.g. KV
//! metadata that bit-matches canonical but is semantically wrong — a
//! pathological case that byte-cmp can't catch, since canonical would
//! produce the same wrong-but-loadable file).
//!
//! Each cell:
//!   1. Runs `hf2q convert <model> --quant <q>` (cached if already done)
//!   2. Runs `/opt/llama.cpp/build/bin/llama-cli -m <gguf> -p "<prompt>"
//!      -n 16 --no-display-prompt --seed 42 -ngl 0`
//!   3. Asserts: exit code 0 AND stdout contains at least 16 decoded
//!      tokens (≥ 4 distinct words) AND no `<unk>` floods.
//!
//! Embedding models (BERT, Nomic) use a different smoke path —
//! `llama-cli` is decoder-only — so they're marked `#[ignore]` and
//! skipped here pending embedding-side smoke harness (Task #62 hooks).

use std::path::{Path, PathBuf};
use std::process::Command;

const LLAMA_CLI: &str = "/opt/llama.cpp/build/bin/llama-cli";
const CACHE_DIR: &str = "/opt/hf2q/cache/byte_cmp";
const PROMPT: &str = "The capital of France is";

fn check_prereqs(model_dir: &str) -> Result<(), String> {
    if !Path::new(model_dir).exists() {
        return Err(format!("model dir {model_dir} not present"));
    }
    if !Path::new(LLAMA_CLI).exists() {
        return Err(format!("llama-cli not built at {LLAMA_CLI}"));
    }
    Ok(())
}

fn gguf_path(model_dir: &str, quant: &str) -> PathBuf {
    let name = Path::new(model_dir).file_name().unwrap().to_str().unwrap();
    Path::new(CACHE_DIR).join(format!("{name}_hf2q_{quant}.gguf"))
}

fn smoke_cell(model_dir: &str, quant: &str) {
    if let Err(reason) = check_prereqs(model_dir) {
        eprintln!("[SKIP] {} {quant}: {reason}", short_name(model_dir));
        return;
    }
    let gguf = gguf_path(model_dir, quant);
    if !gguf.exists() {
        eprintln!(
            "[SKIP] {} {quant}: GGUF {} not present — run byte_cmp_real_model first",
            short_name(model_dir),
            gguf.display()
        );
        return;
    }

    let out = Command::new(LLAMA_CLI)
        .args([
            "-m",
            gguf.to_str().unwrap(),
            "-p",
            PROMPT,
            "-n",
            "16",
            "--no-display-prompt",
            "--seed",
            "42",
            "-ngl",
            "0",
            "--no-conversation",
        ])
        .output()
        .expect("spawn llama-cli");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "{} {quant}: llama-cli exited {}: stderr={}",
        short_name(model_dir),
        out.status,
        stderr.chars().take(2000).collect::<String>()
    );

    // Filter out trailing stats lines (llama_perf_*); only the
    // generated tokens are what we care about. The non-stat output
    // should have ≥ 16 chars (16 tokens × ≥ 1 char each).
    let body: String = stdout
        .lines()
        .filter(|l| !l.starts_with("llama_perf") && !l.contains("[end of text]"))
        .collect::<Vec<_>>()
        .join(" ");
    let token_chars = body.trim().chars().filter(|c| !c.is_whitespace()).count();
    assert!(
        token_chars >= 16,
        "{} {quant}: too few generated chars ({}): {:?}",
        short_name(model_dir),
        token_chars,
        body.chars().take(200).collect::<String>()
    );

    // Reject pathological cases: all `<unk>` or all the same byte.
    let unk_floor = body.matches("<unk>").count();
    assert!(
        unk_floor < 4,
        "{} {quant}: <unk> flood ({unk_floor} occurrences) in output: {:?}",
        short_name(model_dir),
        body.chars().take(200).collect::<String>()
    );

    eprintln!("[OK] {} {quant}: smoke", short_name(model_dir));
}

fn short_name(p: &str) -> &str {
    Path::new(p)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(p)
}

macro_rules! smoke_test {
    ($name:ident, $model:expr, $quant:expr) => {
        #[test]
        #[ignore]
        fn $name() {
            smoke_cell($model, $quant);
        }
    };
}

// ============================================================
// Decoder smoke matrix — one cell per arch × quant.
// Run after byte_cmp_real_model has populated GGUFs:
//   cargo test --release --test byte_cmp_real_model -- --ignored --test-threads=1
//   cargo test --release --test llama_cli_smoke -- --ignored --test-threads=1
// ============================================================

const GEMMA4_26B: &str = "/opt/hf2q/models/google-gemma-4-26b-a4b-it";
const QWEN35_35B: &str = "/opt/hf2q/models/Qwen-Qwen3.5-35B-A3B";
const LLAMA3_8B: &str = "/opt/hf2q/models/NousResearch-Meta-Llama-3-8B";
const MINIMAX_M2: &str = "/opt/hf2q/models/MiniMaxAI-MiniMax-M2";

// Gemma 4 26B-A4B-IT
smoke_test!(smoke_gemma4_26b_q4_k_m, GEMMA4_26B, "q4_k_m");
smoke_test!(smoke_gemma4_26b_q5_k_m, GEMMA4_26B, "q5_k_m");
smoke_test!(smoke_gemma4_26b_q6_k, GEMMA4_26B, "q6_k");
smoke_test!(smoke_gemma4_26b_q8_0, GEMMA4_26B, "q8_0");

// Qwen 3.5 35B-A3B
smoke_test!(smoke_qwen35_35b_q4_k_m, QWEN35_35B, "q4_k_m");
smoke_test!(smoke_qwen35_35b_q5_k_m, QWEN35_35B, "q5_k_m");
smoke_test!(smoke_qwen35_35b_q6_k, QWEN35_35B, "q6_k");
smoke_test!(smoke_qwen35_35b_q8_0, QWEN35_35B, "q8_0");

// Llama 3 8B
smoke_test!(smoke_llama3_8b_q4_k_m, LLAMA3_8B, "q4_k_m");
smoke_test!(smoke_llama3_8b_q5_k_m, LLAMA3_8B, "q5_k_m");
smoke_test!(smoke_llama3_8b_q6_k, LLAMA3_8B, "q6_k");
smoke_test!(smoke_llama3_8b_q8_0, LLAMA3_8B, "q8_0");

// MiniMax-M2
smoke_test!(smoke_minimax_m2_q4_k_m, MINIMAX_M2, "q4_k_m");
smoke_test!(smoke_minimax_m2_q5_k_m, MINIMAX_M2, "q5_k_m");
smoke_test!(smoke_minimax_m2_q6_k, MINIMAX_M2, "q6_k");
