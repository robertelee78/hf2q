//! ADR-005 iter-224 row 5 (Wedge-4e) — Qwen3-VL streaming + tools[] +
//! reasoning_content end-to-end coherence harness.
//!
//! This integration test drives the full Wedge-4e acceptance bar: a real
//! Qwen3-VL GGUF + mmproj loaded through `hf2q serve`, an OpenAI-format
//! chat-completions request carrying one synthetic test image AND
//! `stream: true`, and a coherence check on the SSE delta stream that
//! fails if the model produced garbage (NaN / repeating-token / empty
//! deltas) OR if the streaming path returned 501 (the Wedge-4d
//! placeholder this wedge closes).
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
//! Wedge-4e coherence acceptance criteria (all asserted):
//!
//!   1. **HTTP 200** — request succeeds with a `text/event-stream`
//!      response (no fail-loud, no 501, no 500). The Wedge-4d
//!      placeholder returned 501 here; this wedge closes that gap.
//!   2. **SSE deltas non-empty** — at least one `data:` event with
//!      `choices[0].delta.content` non-empty arrives. (Or a
//!      `delta.tool_calls[*]` event when `tools[]` is non-empty +
//!      `tool_choice=auto`.)
//!   3. **No NaN markers** — the concatenated content text doesn't
//!      contain "NaN", "<unk>", or `\u{0}` bytes.
//!   4. **Token diversity** — no single character repeats >5x in a row
//!      (catches the "EEEEEEE..." failure mode of broken position
//!      embeddings carrying through to streaming).
//!   5. **Final `[DONE]` sentinel** — the stream terminates with the
//!      OpenAI-spec `data: [DONE]` line.
//!   6. **`<think>` tags balanced** — when reasoning_content is
//!      streamed, the open/close markers (`<think>` / `</think>`) are
//!      either both present and balanced or both absent. Catches
//!      truncated reasoning blocks from a borked splitter.
//!   7. **Tool-call JSON well-formed** — when `tools[]` is non-empty
//!      and the model emits a tool call, the streamed tool_call
//!      arguments parse as JSON. Catches a borked tool-call splitter
//!      that emits malformed argument fragments.
//!
//! When `HF2Q_QWEN3VL_E2E` is unset the test calls
//! `assert_harness_compiles_with_no_env` and exits — this is the
//! default `cargo test` path, where we want to know the harness still
//! builds against the live source tree without paying the model-load
//! cost.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const ENV_GATE: &str = "HF2Q_QWEN3VL_E2E";
const ENV_GGUF: &str = "HF2Q_QWEN3VL_E2E_GGUF";
const ENV_MMPROJ: &str = "HF2Q_QWEN3VL_E2E_MMPROJ";

/// Coherence check on the concatenated streamed content text.
fn coherence_check(text: &str) -> Result<(), String> {
    if text.trim().is_empty() {
        return Err("streamed content is empty after trim".to_string());
    }
    if text.contains("NaN") || text.contains("<unk>") || text.contains('\u{0}') {
        return Err(format!(
            "streamed text contains NaN / unk / null marker: {:?}",
            &text[..text.len().min(200)]
        ));
    }
    let mut last_ch: Option<char> = None;
    let mut run: usize = 0;
    for ch in text.chars() {
        if Some(ch) == last_ch {
            run += 1;
            if run > 5 && !ch.is_whitespace() {
                return Err(format!(
                    "streamed text repeats {ch:?} {run} times in a row \
                     (likely broken streaming output): {:?}",
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

/// Validate the `<think>...</think>` bracketing of reasoning_content
/// text. Returns Err on a structural defect that a borked splitter
/// might emit (truncated open marker, unmatched close, etc.).
fn reasoning_balance_check(reasoning_content: &str) -> Result<(), String> {
    if reasoning_content.is_empty() {
        return Ok(()); // model produced no reasoning — fine
    }
    // OpenAI-style reasoning_content is the BODY of <think>...</think>
    // with markers stripped by `ReasoningSplitter`. So a clean stream
    // will have NEITHER `<think>` NOR `</think>` literally inside.
    // A leak in either direction signals splitter dysfunction.
    if reasoning_content.contains("<think>") {
        return Err(format!(
            "reasoning_content leaks an unstripped <think> open marker: {:?}",
            &reasoning_content[..reasoning_content.len().min(120)]
        ));
    }
    if reasoning_content.contains("</think>") {
        return Err(format!(
            "reasoning_content leaks an unstripped </think> close marker: {:?}",
            &reasoning_content[..reasoning_content.len().min(120)]
        ));
    }
    Ok(())
}

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

/// Default-skip test: when the env-gate is unset, exercise the
/// coherence helpers against the canonical failure modes so we know
/// the harness compiles + the helpers still reject what they should.
#[test]
fn qwen3vl_streaming_e2e_default_skips_when_env_gate_unset() {
    if std::env::var(ENV_GATE).ok().as_deref() == Some("1") {
        eprintln!(
            "{ENV_GATE}=1 — running real Qwen3-VL streaming E2E in \
             `qwen3vl_streaming_e2e_real`"
        );
        return;
    }
    // Coherence helper rejects empty + repeating-token text.
    assert!(coherence_check("").is_err());
    assert!(coherence_check("EEEEEEEEEE").is_err());
    assert!(coherence_check("hello there").is_ok());
    assert!(coherence_check("contains NaN here").is_err());
    // Reasoning balance helper.
    assert!(reasoning_balance_check("").is_ok());
    assert!(reasoning_balance_check("clean reasoning").is_ok());
    assert!(reasoning_balance_check("<think>oops").is_err());
    assert!(reasoning_balance_check("oops</think>").is_err());
}

/// Wedge-4e plumbing pin (default-running): assert that the HTTP
/// surface for streaming Qwen3-VL chat NO LONGER emits the Wedge-4d
/// 501 "streaming chat with Qwen3-VL DeepStack injection is not yet
/// supported" diagnostic. Source-grep so we don't need to load a
/// real model.
#[test]
fn qwen3vl_streaming_e2e_handler_does_not_emit_wedge4d_501() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let handlers = crate_root.join("src").join("serve").join("api").join("handlers.rs");
    let src = std::fs::read_to_string(&handlers).expect("read handlers.rs");
    assert!(
        !src.contains("streaming chat with Qwen3-VL DeepStack injection is not yet"),
        "Wedge-4e: chat_completions_stream MUST NOT emit the Wedge-4d \
         501 placeholder — streaming Qwen3-VL is now production"
    );
    assert!(
        src.contains("generate_stream_with_deepstack"),
        "Wedge-4e: chat_completions_stream MUST call \
         generate_stream_with_deepstack so soft_tokens + DeepStack + \
         3D positions reach the worker"
    );
}

/// Wedge-4e plumbing pin (default-running): assert that the engine
/// worker streaming arm at `LoadedModel::Qwen35` no longer emits the
/// Phase-2c soft_token guard diagnostic.
#[test]
fn qwen3vl_streaming_e2e_engine_drops_phase2c_guard() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let engine = crate_root.join("src").join("serve").join("api").join("engine.rs");
    let src = std::fs::read_to_string(&engine).expect("read engine.rs");
    assert!(
        !src.contains("Qwen35 streaming path does not yet support"),
        "Wedge-4e: Phase-2c soft_token guard MUST be removed from \
         engine.rs's LoadedModel::Qwen35 streaming arm"
    );
    assert!(
        src.contains("generate_stream_qwen35_once_extended"),
        "Wedge-4e: engine.rs streaming arm MUST dispatch through the \
         extended Qwen35 stream entry"
    );
    assert!(
        src.contains("generate_stream_with_deepstack"),
        "Wedge-4e: Engine handle MUST expose \
         generate_stream_with_deepstack"
    );
}

/// Real E2E test, gated on `HF2Q_QWEN3VL_E2E=1`. Spawns hf2q-serve as
/// a child, polls /readyz, sends a Qwen3-VL chat-completions request
/// with one synthetic image AND `stream: true`, asserts coherence on
/// the SSE delta stream + the `[DONE]` sentinel.
#[test]
#[ignore = "operator-gated; run with HF2Q_QWEN3VL_E2E=1 and a real Qwen3-VL GGUF + mmproj"]
fn qwen3vl_streaming_e2e_real() {
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
    let port: u16 = 18761; // distinct from chat E2E (18760) so they can run in parallel
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

    let img_bytes = std::fs::read(&fixture).expect("read fixture image");
    let img_b64 = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        &img_bytes,
    );
    let data_uri = format!("data:image/png;base64,{img_b64}");

    // Wedge-4e: STREAMING request body. `stream: true` is the field
    // the Wedge-4d 501 placeholder rejected; this iter's close-gate
    // is a 200 OK with text/event-stream + non-empty SSE deltas.
    let req_body = serde_json::json!({
        "model": model_id,
        "temperature": 0.0,
        "max_tokens": 80,
        "stream": true,
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
        .expect("POST chat-completions stream=true");
    let status = resp.status();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_default();
    let body_text = resp.text().expect("read SSE body");

    let mut assert_and_kill = |cond: bool, msg: String| {
        if !cond {
            let _ = child.kill();
            panic!("{msg}\nresponse body: {body_text}");
        }
    };

    assert_and_kill(
        status.is_success(),
        format!("expected 2xx for streaming Qwen3-VL request, got {status}"),
    );
    assert_and_kill(
        content_type.contains("text/event-stream"),
        format!(
            "expected content-type to be text/event-stream, got: {content_type}"
        ),
    );

    // Parse SSE: collect data: events, accumulate content, look for [DONE].
    let mut accumulated_content = String::new();
    let mut accumulated_reasoning = String::new();
    let mut saw_done = false;
    let mut delta_count = 0usize;
    for line in body_text.lines() {
        if let Some(rest) = line.strip_prefix("data: ") {
            if rest == "[DONE]" {
                saw_done = true;
                continue;
            }
            let v: serde_json::Value = match serde_json::from_str(rest) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(delta) = v["choices"][0]["delta"].as_object() {
                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        accumulated_content.push_str(content);
                        delta_count += 1;
                    }
                }
                if let Some(reasoning) = delta.get("reasoning_content").and_then(|c| c.as_str())
                {
                    accumulated_reasoning.push_str(reasoning);
                }
            }
        }
    }

    assert_and_kill(
        saw_done,
        "SSE stream did not terminate with `data: [DONE]` sentinel".to_string(),
    );
    assert_and_kill(
        delta_count >= 5,
        format!(
            "expected >=5 non-empty content delta events, got {delta_count}"
        ),
    );

    if let Err(reason) = coherence_check(&accumulated_content) {
        let _ = child.kill();
        panic!("streaming coherence check failed: {reason}");
    }

    if let Err(reason) = reasoning_balance_check(&accumulated_reasoning) {
        let _ = child.kill();
        panic!("streaming reasoning_content balance check failed: {reason}");
    }

    let _ = child.kill();
}

/// Real E2E test (gated): streaming + tools[] + tool_choice=auto.
/// Asserts that when `tools[]` is non-empty AND the model emits a
/// tool call, the streamed tool_call arguments parse as JSON. The
/// MODE-INVARIANT `ToolCallSplitter` chain (Wedge-3 Phase E) is the
/// piece under test — it must produce structured tool_call deltas
/// regardless of prefill source (text-only vs vision-augmented).
#[test]
#[ignore = "operator-gated; run with HF2Q_QWEN3VL_E2E=1 and a real Qwen3-VL GGUF + mmproj"]
fn qwen3vl_streaming_e2e_real_with_tools() {
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
    let port: u16 = 18762;
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

    let img_bytes = std::fs::read(&fixture).expect("read fixture image");
    let img_b64 = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        &img_bytes,
    );
    let data_uri = format!("data:image/png;base64,{img_b64}");

    let req_body = serde_json::json!({
        "model": model_id,
        "temperature": 0.0,
        "max_tokens": 120,
        "stream": true,
        "tool_choice": "auto",
        "tools": [{
            "type": "function",
            "function": {
                "name": "describe_image_region",
                "description": "Describe a sub-region of the most recently shown image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "region": {
                            "type": "string",
                            "description": "Which part of the image to describe (e.g., 'top-left')."
                        }
                    },
                    "required": ["region"]
                }
            }
        }],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the top-left of this image using the available tool."},
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
        .expect("POST chat-completions stream=true tools[]");
    let status = resp.status();
    let body_text = resp.text().expect("read SSE body");

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

    // Collect tool_call argument fragments across delta events.
    // OpenAI streams tool calls as { tool_calls: [{ id, type, function:
    // { name, arguments } }, ...] } — the close-buffered shape Wedge-3
    // ships emits the full arguments JSON in one delta on the close
    // boundary. The W-B3 incremental shape is a follow-up.
    let mut tool_args_concat = String::new();
    let mut tool_name_seen: Option<String> = None;
    let mut saw_done = false;
    for line in body_text.lines() {
        if let Some(rest) = line.strip_prefix("data: ") {
            if rest == "[DONE]" {
                saw_done = true;
                continue;
            }
            let v: serde_json::Value = match serde_json::from_str(rest) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(tcs) = v["choices"][0]["delta"]["tool_calls"].as_array() {
                for tc in tcs {
                    if let Some(name) = tc["function"]["name"].as_str() {
                        if !name.is_empty() {
                            tool_name_seen = Some(name.to_string());
                        }
                    }
                    if let Some(args) = tc["function"]["arguments"].as_str() {
                        tool_args_concat.push_str(args);
                    }
                }
            }
        }
    }
    assert_and_kill(
        saw_done,
        "SSE stream did not terminate with `data: [DONE]`".to_string(),
    );

    // The model may legitimately decide NOT to emit a tool call (the
    // image is just a red square — the model could describe it
    // textually instead). We only assert WELL-FORMEDNESS of any
    // emitted tool call, not that one was emitted. If the model did
    // emit one, the args MUST parse as JSON.
    let tool_check_result: Result<(), String> = (|| {
        if let Some(name) = &tool_name_seen {
            if !tool_args_concat.is_empty() {
                let parsed: serde_json::Value =
                    serde_json::from_str(&tool_args_concat).map_err(|e| {
                        format!(
                            "tool_call arguments did not parse as JSON: {e}\n\
                             args: {tool_args_concat:?}"
                        )
                    })?;
                if !parsed.is_object() {
                    return Err(format!(
                        "tool_call arguments parsed but is not a JSON object: {parsed}"
                    ));
                }
                eprintln!(
                    "Wedge-4e tools[]: emitted tool_call name={name} args={tool_args_concat}"
                );
            }
        }
        Ok(())
    })();

    let _ = child.kill();
    if let Err(reason) = tool_check_result {
        panic!("Wedge-4e tools[] check failed: {reason}");
    }
}

/// FORCED-tool-call live assertion (Wedge-4f Codex finding #4 follow-up,
/// closes the ⚠ caveat to ✅): same operator-gated harness as
/// `qwen3vl_streaming_e2e_real_with_tools` BUT uses the OpenAI
/// `tool_choice: {type: "function", function: {name: "..."}}`
/// FORCED-function semantic that the model cannot decline. Per
/// `/opt/hf2q/src/serve/api/handlers.rs:1112-1115`, this physically
/// constrains the sampler-side grammar to only emit a tool_call for
/// the named function — text-content fallback is grammatically
/// impossible. We assert that a tool_call WAS emitted (not just
/// well-formed-if-emitted), and that its `function.name` matches the
/// forced name + its `function.arguments` parse as JSON object with
/// the required parameter present.
#[test]
#[ignore = "operator-gated; run with HF2Q_QWEN3VL_E2E=1 and a real Qwen3-VL GGUF + mmproj"]
fn qwen3vl_streaming_e2e_real_with_tools_forced_function() {
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
    // Distinct port from the auto-tools test so they can run in parallel.
    let port: u16 = 18763;
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

    let img_bytes = std::fs::read(&fixture).expect("read fixture image");
    let img_b64 = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        &img_bytes,
    );
    let data_uri = format!("data:image/png;base64,{img_b64}");

    const FORCED_NAME: &str = "describe_image_region";

    let req_body = serde_json::json!({
        "model": model_id,
        "temperature": 0.0,
        "max_tokens": 120,
        "stream": true,
        // OpenAI FORCED-function tool_choice: server-side grammar
        // physically constrains output to a tool_call for THIS name.
        // Text-content fallback is grammatically impossible per
        // handlers.rs:1112-1115's grammar-compile gate.
        "tool_choice": {"type": "function", "function": {"name": FORCED_NAME}},
        "tools": [{
            "type": "function",
            "function": {
                "name": FORCED_NAME,
                "description": "Describe a sub-region of the most recently shown image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "region": {
                            "type": "string",
                            "description": "Which part of the image to describe (e.g., 'top-left')."
                        }
                    },
                    "required": ["region"]
                }
            }
        }],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the top-left of this image."},
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
        .expect("POST chat-completions stream=true tool_choice=forced");
    let status = resp.status();
    let body_text = resp.text().expect("read SSE body");

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

    // Collect tool_call deltas — we EXPECT one (the model cannot
    // decline under forced function tool_choice).
    let mut tool_args_concat = String::new();
    let mut tool_name_seen: Option<String> = None;
    let mut saw_done = false;
    for line in body_text.lines() {
        if let Some(rest) = line.strip_prefix("data: ") {
            if rest == "[DONE]" {
                saw_done = true;
                continue;
            }
            let v: serde_json::Value = match serde_json::from_str(rest) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(tcs) = v["choices"][0]["delta"]["tool_calls"].as_array() {
                for tc in tcs {
                    if let Some(name) = tc["function"]["name"].as_str() {
                        if !name.is_empty() {
                            tool_name_seen = Some(name.to_string());
                        }
                    }
                    if let Some(args) = tc["function"]["arguments"].as_str() {
                        tool_args_concat.push_str(args);
                    }
                }
            }
        }
    }
    assert_and_kill(
        saw_done,
        "SSE stream did not terminate with `data: [DONE]`".to_string(),
    );

    // FORCED-function pin: a tool_call MUST have been emitted (the
    // grammar-constrained sampler cannot produce text content).
    let tool_check_result: Result<(), String> = (|| {
        let name = tool_name_seen
            .as_deref()
            .ok_or_else(|| {
                "FORCED-function tool_choice produced NO tool_call delta — \
                 the grammar gate at handlers.rs:1112-1115 should have \
                 physically prevented text-content fallback. Either the \
                 grammar compile failed silently OR the server didn't \
                 honor the forced name.".to_string()
            })?;
        if name != FORCED_NAME {
            return Err(format!(
                "FORCED-function tool_choice asked for {FORCED_NAME:?} but \
                 server emitted tool_call for {name:?}"
            ));
        }
        if tool_args_concat.is_empty() {
            return Err(
                "FORCED-function emitted tool_call but arguments were empty"
                    .to_string(),
            );
        }
        let parsed: serde_json::Value = serde_json::from_str(&tool_args_concat)
            .map_err(|e| {
                format!(
                    "FORCED-function tool_call arguments did not parse \
                     as JSON: {e}\nargs: {tool_args_concat:?}"
                )
            })?;
        let obj = parsed.as_object().ok_or_else(|| {
            format!("FORCED-function arguments parsed but not an object: {parsed}")
        })?;
        if !obj.contains_key("region") {
            return Err(format!(
                "FORCED-function arguments missing required `region` key: {parsed}"
            ));
        }
        eprintln!(
            "Wedge-4f finding-#4 closure: FORCED-function tool_call OK \
             name={name} args={tool_args_concat}"
        );
        Ok(())
    })();

    let _ = child.kill();
    if let Err(reason) = tool_check_result {
        panic!("FORCED-tool-call check failed: {reason}");
    }
}
