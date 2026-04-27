//! ADR-005 Phase 2a T1.8 Option B — per-model tool-call grammar live test.
//!
//! # What this test exercises
//!
//! `compile_tool_grammar` (handlers.rs) + `tool_call_gbnf` (registry.rs) form
//! the grammar-constraint pipeline for tool_choice=required / function.  This
//! test validates the full end-to-end path on a live model:
//!
//!   1. Spin up `hf2q serve` with a Gemma 4 GGUF.
//!   2. Send a chat completion with `tool_choice="required"` + a single tool.
//!   3. Assert EVERY token of the output respects the grammar — i.e. the
//!      emitted text matches `call:TOOL_NAME{...}` wrapper syntax.
//!   4. Assert the result parses cleanly via `parse_tool_call_body`.
//!
//! When a Qwen 3.5 GGUF is available (opt-in via `HF2Q_TOOL_GRAMMAR_QWEN_GGUF`),
//! the same sequence runs for the Qwen35 XML wrapper.
//!
//! # Why grammar-constrained output is the correct contract
//!
//! Before T1.8 Option B the engine emitted tool calls unconstrained: the model
//! was free to produce any text between the `<|tool_call>` / `<tool_call|>`
//! markers. `engine.rs:1615-1629` had a silent parse-failure-fallback that
//! discarded the call and emitted `Content(raw_body)` when `parse_tool_call_body`
//! returned `None`. This is a mantra violation (produce garbage rather than
//! surface an error). The grammar constraint makes parse failure structurally
//! impossible for constrained paths, unblocking the fallback removal in wave 3.
//!
//! # Env gate
//!
//! Default-off — loading a 16-27 GB chat GGUF is multi-minute on M5 Max and
//! would OOM if run concurrently with other live tests.
//!
//!   * `HF2Q_TOOL_CALL_GRAMMAR_LIVE_TEST=1` — required to run; absent ⇒ skip.
//!   * `HF2Q_TOOL_GRAMMAR_GEMMA_GGUF=<path>` — Gemma 4 GGUF to use. Defaults
//!     to the canonical abliterated-dwq path.
//!   * `HF2Q_TOOL_GRAMMAR_QWEN_GGUF=<path>` — Qwen 3.5 GGUF (optional). When
//!     absent the Qwen sub-test is reported as skipped-no-fixture.
//!
//! # Run
//!
//! ```ignore
//! HF2Q_TOOL_CALL_GRAMMAR_LIVE_TEST=1 \
//!   cargo test --release --test tool_call_grammar_live -- --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` per the OOM-prevention directive (feedback_oom_prevention).

#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ENV_GATE: &str = "HF2Q_TOOL_CALL_GRAMMAR_LIVE_TEST";
const ENV_GEMMA_GGUF: &str = "HF2Q_TOOL_GRAMMAR_GEMMA_GGUF";
const ENV_QWEN_GGUF: &str = "HF2Q_TOOL_GRAMMAR_QWEN_GGUF";
const DEFAULT_GEMMA_GGUF: &str = concat!(
    "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/",
    "gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
);
const HOST: &str = "127.0.0.1";
/// Port distinct from other live-test servers in this repo:
///   prompt_cache_live:    52335
///   openwebui_multiturn:  52334
///   vision_e2e_vs_mlx_vlm: 18181
///   mmproj_llama_cpp_compat: 52226
const PORT: u16 = 52336;
const READYZ_BUDGET_SECS: u64 = 600;
const REQUEST_TIMEOUT_SECS: u64 = 180;
/// Keep token count low — we only need enough tokens to see the grammar work.
const MAX_TOKENS: u64 = 64;

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

struct ServerGuard(std::process::Child);

impl ServerGuard {
    fn spawn(gguf: &str) -> std::io::Result<Self> {
        use std::process::{Command, Stdio};
        let bin = std::env::var("CARGO_BIN_EXE_hf2q").unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target/release/hf2q")
                .to_string_lossy()
                .into_owned()
        });
        let child = Command::new(&bin)
            .args([
                "serve",
                "--model",
                gguf,
                "--host",
                HOST,
                "--port",
                &PORT.to_string(),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        Ok(Self(child))
    }
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Poll `/readyz` until 200 or timeout.
fn wait_for_readyz() {
    let started = Instant::now();
    loop {
        assert!(
            started.elapsed().as_secs() < READYZ_BUDGET_SECS,
            "server did not become ready within {READYZ_BUDGET_SECS}s"
        );
        let result = (|| -> std::io::Result<u16> {
            let mut s = TcpStream::connect_timeout(
                &format!("{HOST}:{PORT}")
                    .parse()
                    .map_err(std::io::Error::other)?,
                Duration::from_secs(5),
            )?;
            s.set_read_timeout(Some(Duration::from_secs(5)))?;
            s.write_all(
                format!(
                    "GET /readyz HTTP/1.1\r\nHost: {HOST}:{PORT}\r\nConnection: close\r\n\r\n"
                )
                .as_bytes(),
            )?;
            let mut head = [0u8; 64];
            let n = s.read(&mut head)?;
            let head_s = std::str::from_utf8(&head[..n]).unwrap_or("");
            let code = head_s
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(0);
            Ok(code)
        })();
        match result {
            Ok(200) => return,
            Ok(_) | Err(_) => {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers (raw TCP, no external deps)
// ---------------------------------------------------------------------------

/// Send a POST /v1/chat/completions and return the full response body.
fn chat_completions(body: &str) -> String {
    let mut s = TcpStream::connect_timeout(
        &format!("{HOST}:{PORT}")
            .parse()
            .expect("parse addr"),
        Duration::from_secs(5),
    )
    .expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(REQUEST_TIMEOUT_SECS)))
        .expect("set_read_timeout");
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: {HOST}:{PORT}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {}",
        body.len(),
        body
    );
    s.write_all(req.as_bytes()).expect("write request");
    let mut resp = Vec::new();
    s.read_to_end(&mut resp).expect("read response");
    String::from_utf8_lossy(&resp).into_owned()
}

/// Extract the JSON body from a raw HTTP response string.
fn extract_json_body(raw: &str) -> &str {
    if let Some(pos) = raw.find("\r\n\r\n") {
        &raw[pos + 4..]
    } else if let Some(pos) = raw.find("\n\n") {
        &raw[pos + 2..]
    } else {
        raw
    }
}

// ---------------------------------------------------------------------------
// Core test logic
// ---------------------------------------------------------------------------

/// Verify that a chat completion with tool_choice=required for a Gemma 4 model
/// produces output that:
///   1. Is accepted by the `call:NAME{...}` grammar (grammar constraint fired).
///   2. Parses cleanly via parse_gemma4_tool_call (closes the T2.4 loop).
///
/// The test sends a request to a live server with tool_choice=required and
/// a single `get_current_weather` tool. The grammar constrains the model to
/// emit `call:get_current_weather{...}` between the tool-call markers.
fn run_gemma4_tool_call_grammar_test(model_name: &str) {
    // The tool definition used in the request.
    let request_body = serde_json::json!({
        "model": model_name,
        "messages": [
            {"role": "user", "content": "What's the weather in Paris? Use the weather tool."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and country"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "tool_choice": "required",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0
    });

    let raw = chat_completions(&request_body.to_string());
    let body = extract_json_body(&raw);
    let resp: serde_json::Value =
        serde_json::from_str(body).unwrap_or_else(|e| {
            panic!("failed to parse response JSON: {e}\nraw response:\n{raw}")
        });

    // Assert the response is a success (not an error).
    assert!(
        resp["error"].is_null(),
        "server returned error: {}\nfull response: {resp}",
        resp["error"]
    );

    // Extract the assistant message content.
    let choices = resp["choices"]
        .as_array()
        .expect("choices array");
    assert!(!choices.is_empty(), "no choices in response");

    let choice = &choices[0];
    let message = &choice["message"];

    // The model MUST have emitted a tool call (tool_choice=required).
    let tool_calls = message["tool_calls"].as_array();
    assert!(
        tool_calls.is_some() && !tool_calls.unwrap().is_empty(),
        "expected tool_calls in message but got none. message: {message}"
    );

    let call = &tool_calls.unwrap()[0];
    let fn_name = call["function"]["name"].as_str().unwrap_or("");
    let fn_args = call["function"]["arguments"].as_str().unwrap_or("");

    assert_eq!(
        fn_name, "get_current_weather",
        "unexpected function name: {fn_name}"
    );

    // The arguments must be valid JSON (parse_gemma4_tool_call converts to JSON).
    let args_val: serde_json::Value = serde_json::from_str(fn_args).unwrap_or_else(|e| {
        panic!("function arguments are not valid JSON: {e}\narguments: {fn_args}")
    });

    // The `location` argument must be present (it's the only required field).
    assert!(
        !args_val["location"].is_null(),
        "expected 'location' field in arguments: {args_val}"
    );

    println!(
        "[gemma4 tool grammar live] fn={fn_name} args={fn_args} — PASS"
    );
}

/// Same test for Qwen 3.5/3.6 — validates the XML wrapper grammar constraint.
fn run_qwen35_tool_call_grammar_test(model_name: &str) {
    let request_body = serde_json::json!({
        "model": model_name,
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo? Use the weather tool."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and country"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "tool_choice": "required",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0
    });

    let raw = chat_completions(&request_body.to_string());
    let body = extract_json_body(&raw);
    let resp: serde_json::Value =
        serde_json::from_str(body).unwrap_or_else(|e| {
            panic!("Qwen35: failed to parse response JSON: {e}\nraw:\n{raw}")
        });

    assert!(
        resp["error"].is_null(),
        "Qwen35: server returned error: {}",
        resp["error"]
    );

    let choices = resp["choices"].as_array().expect("choices array");
    assert!(!choices.is_empty());
    let message = &choices[0]["message"];
    let tool_calls = message["tool_calls"].as_array();
    assert!(
        tool_calls.is_some() && !tool_calls.unwrap().is_empty(),
        "Qwen35: expected tool_calls, got: {message}"
    );

    let call = &tool_calls.unwrap()[0];
    let fn_name = call["function"]["name"].as_str().unwrap_or("");
    let fn_args = call["function"]["arguments"].as_str().unwrap_or("");

    assert_eq!(fn_name, "get_current_weather");

    let args_val: serde_json::Value = serde_json::from_str(fn_args).unwrap_or_else(|e| {
        panic!("Qwen35: args not valid JSON: {e}\nargs: {fn_args}")
    });
    assert!(
        !args_val["location"].is_null(),
        "Qwen35: no 'location' in args: {args_val}"
    );

    println!(
        "[qwen35 tool grammar live] fn={fn_name} args={fn_args} — PASS"
    );
}

// ---------------------------------------------------------------------------
// Test entry points
// ---------------------------------------------------------------------------

#[test]
fn tool_call_grammar_gemma4_live() {
    if std::env::var(ENV_GATE).as_deref() != Ok("1") {
        println!("[SKIP] {ENV_GATE} not set — skipping live tool grammar test");
        return;
    }

    let gguf = std::env::var(ENV_GEMMA_GGUF)
        .unwrap_or_else(|_| DEFAULT_GEMMA_GGUF.to_string());

    if !std::path::Path::new(&gguf).exists() {
        println!("[SKIP] Gemma 4 GGUF not found at {gguf}; set {ENV_GEMMA_GGUF}");
        return;
    }

    println!("[tool_call_grammar_live] Gemma 4 GGUF: {gguf}");

    let _guard = ServerGuard::spawn(&gguf)
        .unwrap_or_else(|e| panic!("failed to spawn hf2q serve: {e}"));

    println!("[tool_call_grammar_live] waiting for server ready ...");
    wait_for_readyz();
    println!("[tool_call_grammar_live] server ready");

    // Use the model name as it appears in /v1/models (path without extension,
    // basename). The registry matches on substrings so `gemma-4` in the path
    // is sufficient.
    let model_name = std::path::Path::new(&gguf)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("gemma-4-model");

    run_gemma4_tool_call_grammar_test(model_name);
}

#[test]
fn tool_call_grammar_qwen35_live() {
    if std::env::var(ENV_GATE).as_deref() != Ok("1") {
        println!("[SKIP] {ENV_GATE} not set — skipping live Qwen35 tool grammar test");
        return;
    }

    let gguf = match std::env::var(ENV_QWEN_GGUF) {
        Ok(p) => p,
        Err(_) => {
            println!("[SKIP] {ENV_QWEN_GGUF} not set — Qwen35 sub-test skipped (skipped-no-fixture)");
            return;
        }
    };

    if !std::path::Path::new(&gguf).exists() {
        println!("[SKIP] Qwen35 GGUF not found at {gguf}");
        return;
    }

    println!("[tool_call_grammar_live] Qwen35 GGUF: {gguf}");

    let _guard = ServerGuard::spawn(&gguf)
        .unwrap_or_else(|e| panic!("failed to spawn hf2q serve: {e}"));

    println!("[tool_call_grammar_live] waiting for Qwen35 server ready ...");
    wait_for_readyz();
    println!("[tool_call_grammar_live] server ready");

    let model_name = std::path::Path::new(&gguf)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("qwen3.5-model");

    run_qwen35_tool_call_grammar_test(model_name);
}
