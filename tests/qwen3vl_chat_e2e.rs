//! ADR-005 iter-224 row 4 (Wedge-4d) — Qwen3-VL end-to-end multimodal
//! chat coherence harness.
//!
//! This integration test drives the full Wedge-4d acceptance bar: a real
//! Qwen3-VL GGUF + mmproj loaded through `hf2q serve`, an OpenAI-format
//! chat-completions request carrying one synthetic test image, and a
//! coherence check on the response that fails if the model produced
//! garbage (NaN / repeating-token / empty-string output).
//!
//! It is **gated** behind `HF2Q_QWEN3VL_E2E=1` because the leg loads a
//! ~6-30 GB model and the project's standing OOM-prevention directive
//! says **one model-loading inference at a time**. Default `cargo test`
//! runs no model — only the harness compile path is exercised, which is
//! cheap.
//!
//! Two more env knobs (only consulted when `HF2Q_QWEN3VL_E2E=1`):
//!
//!   * `HF2Q_QWEN3VL_E2E_GGUF`   — path to the chat-model GGUF
//!     (Qwen3-VL-2B / Qwen3-VL-8B).
//!   * `HF2Q_QWEN3VL_E2E_MMPROJ` — path to the mmproj GGUF (the
//!     `qwen3vl_merger` projector).
//!
//! Wedge-4d coherence acceptance criteria (all asserted in this harness):
//!
//!   1. **HTTP 200** — request succeeds (no fail-loud, no 501, no 500).
//!   2. **Non-empty content** — response.choices[0].message.content is
//!      non-empty after stripping whitespace.
//!   3. **Length floor** — at least 50 generated tokens (the model
//!      doesn't immediately emit EOS / `<|im_end|>`).
//!   4. **No NaN markers** — the response text doesn't contain "NaN",
//!      "<unk>", or `\u{0}` bytes.
//!   5. **Token diversity** — no single character repeats >5x in a row
//!      (catches the "EEEEEEE..." failure mode of broken position
//!      embeddings).
//!   6. **Vision header is set** — the response carries
//!      `X-HF2Q-Soft-Tokens-Total > 0`, proving the prompt actually
//!      ran through the soft-token path (not a text-only fallback).
//!
//! When `HF2Q_QWEN3VL_E2E` is unset the test calls
//! `assert_harness_compiles_with_no_env` and exits — this is the
//! default `cargo test` path, where we want to know the harness still
//! builds against the live source tree without paying the model-load
//! cost.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Env-knob constants (mirror the gemma4v `vision_e2e_vs_mlx_vlm.rs`
/// pattern so operators have a uniform mental model).
const ENV_GATE: &str = "HF2Q_QWEN3VL_E2E";
const ENV_GGUF: &str = "HF2Q_QWEN3VL_E2E_GGUF";
const ENV_MMPROJ: &str = "HF2Q_QWEN3VL_E2E_MMPROJ";

/// Simple coherence check: returns Err(reason) on any failure.
fn coherence_check(text: &str) -> Result<(), String> {
    if text.trim().is_empty() {
        return Err("response content is empty after trim".to_string());
    }
    if text.contains("NaN") || text.contains("<unk>") || text.contains('\u{0}') {
        return Err(format!(
            "response text contains NaN / unk / null marker: {:?}",
            &text[..text.len().min(200)]
        ));
    }
    // Token-diversity check: no character repeats >5x in a row. Catches
    // the "EEEEEEE..." failure mode of broken position embeddings or
    // a misrouted soft-token-injection chunk.
    let mut last_ch: Option<char> = None;
    let mut run: usize = 0;
    for ch in text.chars() {
        if Some(ch) == last_ch {
            run += 1;
            if run > 5 && !ch.is_whitespace() {
                return Err(format!(
                    "response text repeats {ch:?} {run} times in a row \
                     (likely broken model output): {:?}",
                    &text[..text.len().min(200)]
                ));
            }
        } else {
            run = 1;
            last_ch = Some(ch);
        }
    }
    Ok(())
}

/// Materialize a single synthetic 768×768 PNG fixture (a red square)
/// for the E2E request. Size matches Qwen3-VL canonical trained input
/// so smart-resize is a no-op.
fn materialize_fixture_image(path: &Path) -> std::io::Result<()> {
    use image::{ImageBuffer, ImageFormat, Rgb, RgbImage};
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let img: RgbImage = ImageBuffer::from_fn(768, 768, |_x, _y| Rgb([220u8, 30, 30]));
    img.save_with_format(path, ImageFormat::Png)
        .map_err(|e| std::io::Error::other(e.to_string()))
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("vision")
        .join("qwen3vl_solid_red_768.png")
}

/// Default-skip test: when the env-gate is unset, assert the harness
/// compiles against the live source tree and that the coherence-check
/// helpers reject the canonical failure modes. No model is loaded.
#[test]
fn qwen3vl_chat_e2e_default_skips_when_env_gate_unset() {
    if std::env::var(ENV_GATE).ok().as_deref() == Some("1") {
        eprintln!(
            "{ENV_GATE}=1 — running real Qwen3-VL E2E in `qwen3vl_chat_e2e_real`"
        );
        return;
    }
    // Sanity: the coherence helper rejects empty + repeating-token text.
    assert!(coherence_check("").is_err());
    assert!(coherence_check("EEEEEEEEEE").is_err());
    assert!(coherence_check("hello there").is_ok());
    assert!(coherence_check("contains NaN here").is_err());
}

/// Real E2E test, gated on `HF2Q_QWEN3VL_E2E=1`. Spawns hf2q-serve as
/// a child, polls /readyz, sends a Qwen3-VL chat request with one
/// synthetic image, asserts coherence + the X-HF2Q-Soft-Tokens-Total
/// transparency header.
#[test]
#[ignore = "operator-gated; run with HF2Q_QWEN3VL_E2E=1 and a real Qwen3-VL GGUF + mmproj"]
fn qwen3vl_chat_e2e_real() {
    if std::env::var(ENV_GATE).ok().as_deref() != Some("1") {
        return;
    }
    let gguf_path = std::env::var(ENV_GGUF)
        .unwrap_or_else(|_| panic!("{ENV_GGUF} must be set when {ENV_GATE}=1"));
    let mmproj_path = std::env::var(ENV_MMPROJ)
        .unwrap_or_else(|_| panic!("{ENV_MMPROJ} must be set when {ENV_GATE}=1"));

    let fixture = fixture_path();
    materialize_fixture_image(&fixture).expect("materialize fixture image");

    let bin = env!("CARGO_BIN_EXE_hf2q");
    let port: u16 = 18760;
    let mut child = std::process::Command::new(bin)
        .args([
            "serve",
            "--model",
            &gguf_path,
            "--mmproj",
            &mmproj_path,
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
        ])
        .spawn()
        .expect("spawn hf2q serve");

    let server_url = format!("http://127.0.0.1:{port}");
    // Poll /readyz with a 180s budget — model load is the main cost.
    let deadline = Instant::now() + Duration::from_secs(180);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest client");
    loop {
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("hf2q serve /readyz did not flip to 200 within 180s");
        }
        match client.get(format!("{server_url}/readyz")).send() {
            Ok(r) if r.status().is_success() => break,
            _ => std::thread::sleep(Duration::from_millis(500)),
        }
    }

    // Discover the registered model id from /v1/models (server is the
    // source of truth — see vision_e2e_vs_mlx_vlm.rs Bug 1 for context).
    let model_id = {
        let r = client
            .get(format!("{server_url}/v1/models"))
            .send()
            .expect("GET /v1/models");
        let body: serde_json::Value = r.json().expect("parse models response");
        body["data"][0]["id"]
            .as_str()
            .expect("models[0].id is a string")
            .to_string()
    };

    // Build the OpenAI chat-completions request body. Image is a
    // base64-encoded 768x768 red square.
    let img_bytes = std::fs::read(&fixture).expect("read fixture image");
    let img_b64 = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        &img_bytes,
    );
    let data_uri = format!("data:image/png;base64,{img_b64}");
    let req_body = serde_json::json!({
        "model": model_id,
        "temperature": 0.0,
        "max_tokens": 80,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image briefly."},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }]
    });

    let resp = client
        .post(format!("{server_url}/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .json(&req_body)
        .timeout(Duration::from_secs(120))
        .send()
        .expect("POST chat-completions");
    let status = resp.status();
    let headers = resp.headers().clone();
    let body_text = resp.text().expect("read body");

    // Be sure to clean up the child even if assertions fire below.
    let mut assert_and_kill = |cond: bool, msg: String| {
        if !cond {
            let _ = child.kill();
            panic!("{msg}\nresponse body: {body_text}");
        }
    };

    assert_and_kill(
        status.is_success(),
        format!("expected 2xx, got {status}"),
    );

    // Soft-tokens header check — proves the soft-token path actually ran.
    let soft_total: usize = headers
        .get("x-hf2q-soft-tokens-total")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert_and_kill(
        soft_total > 0,
        format!("X-HF2Q-Soft-Tokens-Total header missing or zero (got {soft_total}); the request did not flow through the soft-token path"),
    );

    let body_json: serde_json::Value =
        serde_json::from_str(&body_text).expect("body is JSON");
    let content = body_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let completion_tokens = body_json["usage"]["completion_tokens"]
        .as_u64()
        .unwrap_or(0);

    assert_and_kill(
        completion_tokens >= 50,
        format!("expected >=50 generated tokens, got {completion_tokens}"),
    );

    if let Err(reason) = coherence_check(&content) {
        let _ = child.kill();
        panic!("coherence check failed: {reason}");
    }

    let _ = child.kill();
}
