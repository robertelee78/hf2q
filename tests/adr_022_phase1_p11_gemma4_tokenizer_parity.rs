//! ADR-022 Phase 1 P1.11 — GGUF-embedded Gemma4 tokenizer parity.
//!
//! Asserts that `hf2q::inference::models::gemma4::tokenizer::build_tokenizer_from_gguf`
//! produces token streams byte-identical to the HF on-disk
//! `tokenizer.json` for the same text on the same model.
//!
//! Skips cleanly if the fixture file isn't present locally — the test
//! is the falsifier when run, but is not a load-bearing regression
//! gate (operator-supplied fixtures live in `/opt/hf2q/models/`).

use std::path::Path;
use tokenizers::Tokenizer;

const GGUF_PATH: &str = "/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf";
const HF_TOKENIZER_PATH: &str = "/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/tokenizer.json";

#[path = "../src/inference/models/gemma4/tokenizer.rs"]
mod gemma4_tokenizer;

fn ggml_open(path: &str) -> mlx_native::gguf::GgufFile {
    mlx_native::gguf::GgufFile::open(Path::new(path)).expect("GgufFile::open")
}

fn run_parity(text: &str) {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("[skip] {} not present", GGUF_PATH);
        return;
    }
    if !Path::new(HF_TOKENIZER_PATH).exists() {
        eprintln!("[skip] {} not present", HF_TOKENIZER_PATH);
        return;
    }

    let gguf = ggml_open(GGUF_PATH);
    let from_gguf = gemma4_tokenizer::build_tokenizer_from_gguf(&gguf)
        .expect("build_tokenizer_from_gguf");
    let from_disk = Tokenizer::from_file(HF_TOKENIZER_PATH)
        .expect("Tokenizer::from_file");

    let g_enc = from_gguf.encode(text, false).expect("encode (gguf)");
    let d_enc = from_disk.encode(text, false).expect("encode (disk)");

    let g_ids = g_enc.get_ids();
    let d_ids = d_enc.get_ids();

    eprintln!("[adr-022 P1.11] text={text:?}");
    eprintln!("  gguf ids: {:?}", g_ids);
    eprintln!("  disk ids: {:?}", d_ids);

    if g_ids != d_ids {
        let g_toks = g_enc.get_tokens();
        let d_toks = d_enc.get_tokens();
        eprintln!("  gguf tokens: {:?}", g_toks);
        eprintln!("  disk tokens: {:?}", d_toks);
        panic!(
            "token stream divergence on {text:?}: gguf {:?} vs disk {:?}",
            g_ids, d_ids
        );
    }
}

#[test]
fn adr022_p11_gemma4_tokenizer_simple_question() {
    run_parity("What is 2+2?");
}

#[test]
fn adr022_p11_gemma4_tokenizer_with_special_tokens() {
    // Templated chat-style fragment that exercises the BOS / channel
    // markers (`<|turn>`, `<|channel>`) — same shape as the rendered
    // prompt the byte-equal AC validates against llama-completion.
    run_parity("<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n");
}

#[test]
fn adr022_p11_gemma4_tokenizer_multibyte() {
    run_parity("Café — the price is €5. ¿Cómo estás?");
}

#[test]
fn adr022_p11_gemma4_tokenizer_newlines_only() {
    run_parity("\n\n\n");
}

#[test]
fn adr022_p11_gemma4_tokenizer_punctuation_runs() {
    run_parity("Hello,, world!!! ()[]{}");
}
