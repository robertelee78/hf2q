//! ADR-033 §P1 — real-model byte-cmp acceptance gate, in-tree.
//!
//! Each cell is `#[ignore]`'d so default `cargo test` stays fast. Run
//! the full matrix with `cargo test --release --ignored byte_cmp -- --test-threads=1`.
//!
//! Each cell checks prerequisites (HF model dir, `/opt/llama.cpp`
//! build, pinned SHA) and SKIPS via early-return when prerequisites
//! are absent — so partial environments (CI without the model on
//! disk) pass `--ignored` without spurious failures. A skipped cell
//! prints `[SKIP] ...` to stderr; a verified cell prints `[OK] ...`
//! and asserts 0 bytes diff.
//!
//! Replaces `scripts/byte_cmp_full_pipeline.sh` as the in-tree gate
//! (the shell script remains for ad-hoc operator-time runs). Once a
//! cell is enabled (model present + canonical built), `cargo test
//! --release --ignored byte_cmp_<arch>_<quant>` is the regression
//! guard: any drift in either kernel or per-arch mapping flips the
//! cell red instantly.

use std::path::{Path, PathBuf};
use std::process::Command;

const LLAMA_CPP: &str = "/opt/llama.cpp";
const CACHE_DIR: &str = "/opt/hf2q/cache/byte_cmp";
const CMP_CHUNK: usize = 64 * 1024 * 1024;

fn hf2q_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_hf2q"))
}

fn check_prereqs(model_dir: &str) -> Result<(), String> {
    if !Path::new(model_dir).exists() {
        return Err(format!("model dir {model_dir} not present"));
    }
    if !Path::new(LLAMA_CPP).join("build/bin/llama-quantize").exists() {
        return Err(format!("canonical llama-quantize not built at {LLAMA_CPP}/build/bin/"));
    }
    if let Ok(pin) = std::fs::read_to_string("data/llama_cpp_pin.txt") {
        let pinned = pin.trim();
        let head = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(LLAMA_CPP)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        if !pinned.is_empty() && head != pinned {
            return Err(format!(
                "canonical SHA {} != pinned {}",
                head.chars().take(8).collect::<String>(),
                pinned.chars().take(8).collect::<String>()
            ));
        }
    }
    Ok(())
}

fn ensure_canonical_f16(model_dir: &str) -> Result<PathBuf, String> {
    std::fs::create_dir_all(CACHE_DIR).map_err(|e| format!("mkdir cache: {e}"))?;
    let name = Path::new(model_dir).file_name().unwrap().to_str().unwrap();
    let f16 = Path::new(CACHE_DIR).join(format!("{name}_canonical_f16.gguf"));
    if f16.exists() {
        return Ok(f16);
    }
    let status = Command::new("python3")
        .arg(format!("{LLAMA_CPP}/convert_hf_to_gguf.py"))
        .arg(model_dir)
        .arg("--outtype")
        .arg("f16")
        .arg("--outfile")
        .arg(&f16)
        .status()
        .map_err(|e| format!("spawn python convert: {e}"))?;
    if !status.success() {
        return Err(format!("canonical f16 convert failed: {status}"));
    }
    Ok(f16)
}

fn ensure_canonical_quant(f16: &Path, quant: &str) -> Result<PathBuf, String> {
    let stem = f16.file_stem().unwrap().to_str().unwrap();
    let name = stem.trim_end_matches("_canonical_f16");
    let out = Path::new(CACHE_DIR).join(format!("{name}_canonical_{quant}.gguf"));
    if out.exists() {
        return Ok(out);
    }
    let status = Command::new(format!("{LLAMA_CPP}/build/bin/llama-quantize"))
        .arg(f16)
        .arg(&out)
        .arg(quant.to_uppercase())
        .status()
        .map_err(|e| format!("spawn llama-quantize: {e}"))?;
    if !status.success() {
        return Err(format!("canonical {quant} quantize failed: {status}"));
    }
    Ok(out)
}

fn run_hf2q_convert(model_dir: &str, quant: &str) -> Result<PathBuf, String> {
    let name = Path::new(model_dir).file_name().unwrap().to_str().unwrap();
    let out = Path::new(CACHE_DIR).join(format!("{name}_hf2q_{quant}.gguf"));
    let _ = std::fs::remove_file(&out);
    let status = Command::new(hf2q_bin())
        .arg("convert")
        .arg(model_dir)
        .arg("--quant")
        .arg(quant)
        .arg("-o")
        .arg(&out)
        .status()
        .map_err(|e| format!("spawn hf2q: {e}"))?;
    if !status.success() {
        return Err(format!("hf2q convert failed: {status}"));
    }
    Ok(out)
}

fn streaming_byte_cmp(a: &Path, b: &Path) -> Result<(u64, u64), String> {
    use std::io::Read;
    let mut fa = std::fs::File::open(a).map_err(|e| format!("open {a:?}: {e}"))?;
    let mut fb = std::fs::File::open(b).map_err(|e| format!("open {b:?}: {e}"))?;
    let sa = fa.metadata().unwrap().len();
    let sb = fb.metadata().unwrap().len();
    if sa != sb {
        return Err(format!("size mismatch: canonical={sa} hf2q={sb}"));
    }
    let mut buf_a = vec![0u8; CMP_CHUNK];
    let mut buf_b = vec![0u8; CMP_CHUNK];
    let mut total = 0u64;
    let mut diff = 0u64;
    loop {
        let na = fa.read(&mut buf_a).map_err(|e| format!("read a: {e}"))?;
        let nb = fb.read(&mut buf_b).map_err(|e| format!("read b: {e}"))?;
        if na == 0 && nb == 0 {
            break;
        }
        if na != nb {
            return Err(format!("short read mismatch: na={na} nb={nb}"));
        }
        for (x, y) in buf_a[..na].iter().zip(buf_b[..na].iter()) {
            if x != y {
                diff += 1;
            }
        }
        total += na as u64;
    }
    Ok((diff, total))
}

fn byte_cmp_cell(model_dir: &str, quant: &str) {
    if let Err(reason) = check_prereqs(model_dir) {
        eprintln!("[SKIP] {} {quant}: {reason}", short_name(model_dir));
        return;
    }
    let f16 = match ensure_canonical_f16(model_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[SKIP] {} {quant}: ensure_f16: {e}", short_name(model_dir));
            return;
        }
    };
    let canonical_q = match ensure_canonical_quant(&f16, quant) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[SKIP] {} {quant}: ensure_quant: {e}", short_name(model_dir));
            return;
        }
    };
    let hf2q_q = run_hf2q_convert(model_dir, quant).expect("hf2q convert");
    let (diff, total) = streaming_byte_cmp(&canonical_q, &hf2q_q).expect("streaming cmp");
    eprintln!(
        "[{}] {} {quant}: {diff}/{total} bytes diff",
        if diff == 0 { "OK" } else { "FAIL" },
        short_name(model_dir)
    );
    assert_eq!(
        diff, 0,
        "byte-cmp drift for {} {quant}: {diff}/{total}",
        short_name(model_dir)
    );
}

fn short_name(p: &str) -> &str {
    Path::new(p).file_name().and_then(|s| s.to_str()).unwrap_or(p)
}

macro_rules! byte_cmp_test {
    ($name:ident, $model:expr, $quant:expr) => {
        #[test]
        #[ignore]
        fn $name() {
            byte_cmp_cell($model, $quant);
        }
    };
}

// ============================================================
// Acceptance matrix — one #[ignore]'d test per (arch, quant) cell.
// Run a single cell:
//   cargo test --release --ignored byte_cmp_gemma4_26b_q4_k_m
// Run all cells for an arch:
//   cargo test --release --ignored byte_cmp_gemma4_26b -- --test-threads=1
// Run the full matrix:
//   cargo test --release --ignored byte_cmp -- --test-threads=1
// ============================================================

const GEMMA4_26B: &str = "/opt/hf2q/models/google-gemma-4-26b-a4b-it";
const QWEN35_35B: &str = "/opt/hf2q/models/Qwen-Qwen3.5-35B-A3B";
const BGE_LARGE_EN: &str = "/opt/hf2q/models/BAAI-bge-large-en-v1.5";
const NOMIC_EMBED: &str = "/opt/hf2q/models/nomic-ai-nomic-embed-text-v1.5";
const LLAMA3_8B: &str = "/opt/hf2q/models/meta-llama-Meta-Llama-3-8B";
const MINIMAX_M2: &str = "/opt/hf2q/models/MiniMaxAI-MiniMax-M2";
const QWEN3VL_TEXT: &str = "/opt/hf2q/models/Qwen-Qwen3-VL";

// Gemma 4 26B-A4B-IT — full 8-quant matrix
byte_cmp_test!(byte_cmp_gemma4_26b_q4_0, GEMMA4_26B, "q4_0");
byte_cmp_test!(byte_cmp_gemma4_26b_q4_k_s, GEMMA4_26B, "q4_k_s");
byte_cmp_test!(byte_cmp_gemma4_26b_q4_k_m, GEMMA4_26B, "q4_k_m");
byte_cmp_test!(byte_cmp_gemma4_26b_q5_k_s, GEMMA4_26B, "q5_k_s");
byte_cmp_test!(byte_cmp_gemma4_26b_q5_k_m, GEMMA4_26B, "q5_k_m");
byte_cmp_test!(byte_cmp_gemma4_26b_q6_k, GEMMA4_26B, "q6_k");
byte_cmp_test!(byte_cmp_gemma4_26b_q8_0, GEMMA4_26B, "q8_0");
byte_cmp_test!(byte_cmp_gemma4_26b_iq4_nl, GEMMA4_26B, "iq4_nl");

// Qwen 3.5 35B-A3B — full 8-quant matrix
byte_cmp_test!(byte_cmp_qwen35_35b_q4_0, QWEN35_35B, "q4_0");
byte_cmp_test!(byte_cmp_qwen35_35b_q4_k_s, QWEN35_35B, "q4_k_s");
byte_cmp_test!(byte_cmp_qwen35_35b_q4_k_m, QWEN35_35B, "q4_k_m");
byte_cmp_test!(byte_cmp_qwen35_35b_q5_k_s, QWEN35_35B, "q5_k_s");
byte_cmp_test!(byte_cmp_qwen35_35b_q5_k_m, QWEN35_35B, "q5_k_m");
byte_cmp_test!(byte_cmp_qwen35_35b_q6_k, QWEN35_35B, "q6_k");
byte_cmp_test!(byte_cmp_qwen35_35b_q8_0, QWEN35_35B, "q8_0");
byte_cmp_test!(byte_cmp_qwen35_35b_iq4_nl, QWEN35_35B, "iq4_nl");

// BERT (bge-large-en-v1.5) — embeddings; F16 + smaller k-quants where supported
byte_cmp_test!(byte_cmp_bert_bge_q4_0, BGE_LARGE_EN, "q4_0");
byte_cmp_test!(byte_cmp_bert_bge_q4_k_m, BGE_LARGE_EN, "q4_k_m");
byte_cmp_test!(byte_cmp_bert_bge_q5_k_m, BGE_LARGE_EN, "q5_k_m");
byte_cmp_test!(byte_cmp_bert_bge_q6_k, BGE_LARGE_EN, "q6_k");
byte_cmp_test!(byte_cmp_bert_bge_q8_0, BGE_LARGE_EN, "q8_0");

// Nomic BERT (nomic-embed-text-v1.5) — embeddings
byte_cmp_test!(byte_cmp_nomic_q4_0, NOMIC_EMBED, "q4_0");
byte_cmp_test!(byte_cmp_nomic_q4_k_m, NOMIC_EMBED, "q4_k_m");
byte_cmp_test!(byte_cmp_nomic_q5_k_m, NOMIC_EMBED, "q5_k_m");
byte_cmp_test!(byte_cmp_nomic_q6_k, NOMIC_EMBED, "q6_k");
byte_cmp_test!(byte_cmp_nomic_q8_0, NOMIC_EMBED, "q8_0");

// Llama 3 8B — dense decoder, 8-quant matrix
byte_cmp_test!(byte_cmp_llama3_8b_q4_0, LLAMA3_8B, "q4_0");
byte_cmp_test!(byte_cmp_llama3_8b_q4_k_s, LLAMA3_8B, "q4_k_s");
byte_cmp_test!(byte_cmp_llama3_8b_q4_k_m, LLAMA3_8B, "q4_k_m");
byte_cmp_test!(byte_cmp_llama3_8b_q5_k_s, LLAMA3_8B, "q5_k_s");
byte_cmp_test!(byte_cmp_llama3_8b_q5_k_m, LLAMA3_8B, "q5_k_m");
byte_cmp_test!(byte_cmp_llama3_8b_q6_k, LLAMA3_8B, "q6_k");
byte_cmp_test!(byte_cmp_llama3_8b_q8_0, LLAMA3_8B, "q8_0");
byte_cmp_test!(byte_cmp_llama3_8b_iq4_nl, LLAMA3_8B, "iq4_nl");

// MiniMax-M2 — 3rd MoE for APEX validation
byte_cmp_test!(byte_cmp_minimax_m2_q4_k_m, MINIMAX_M2, "q4_k_m");
byte_cmp_test!(byte_cmp_minimax_m2_q5_k_m, MINIMAX_M2, "q5_k_m");
byte_cmp_test!(byte_cmp_minimax_m2_q6_k, MINIMAX_M2, "q6_k");
byte_cmp_test!(byte_cmp_minimax_m2_q8_0, MINIMAX_M2, "q8_0");

// Qwen 3 VL Text
byte_cmp_test!(byte_cmp_qwen3vl_text_q4_k_m, QWEN3VL_TEXT, "q4_k_m");
byte_cmp_test!(byte_cmp_qwen3vl_text_q5_k_m, QWEN3VL_TEXT, "q5_k_m");
byte_cmp_test!(byte_cmp_qwen3vl_text_q6_k, QWEN3VL_TEXT, "q6_k");
byte_cmp_test!(byte_cmp_qwen3vl_text_q8_0, QWEN3VL_TEXT, "q8_0");
