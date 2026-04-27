//! ADR-005 Phase 2.5 W-δ — Real integration tests for the grammar-constrained
//! tool-call path (streaming + non-streaming).
//!
//! # What the wave-2 sham got wrong
//!
//! `tests/tool_call_grammar_live.rs` (wave-2) sent non-streaming requests with
//! `tool_choice="required"` and asserted `message.tool_calls` was populated.
//! That assertion is structurally impossible: `handlers.rs` assembles the
//! non-streaming response at line 396 with `tool_calls: None` hardcoded.
//! The sham "passed" only because the env gate
//! (`HF2Q_TOOL_CALL_GRAMMAR_LIVE_TEST`) was never set in CI, so the test body
//! always exited early as SKIP and never exercised the code path.
//!
//! # What these tests exercise
//!
//! **Streaming test** (`tool_call_grammar_constrained_stream`):
//!   1. Spawns `hf2q serve` with a Gemma 4 abliterated-DWQ GGUF.
//!   2. POSTs `/v1/chat/completions` with `tool_choice={type:"function",
//!      function:{name:"get_weather"}}` + `stream:true`.
//!   3. Reads and parses every SSE frame.
//!   4. Asserts:
//!      - At least one `delta.tool_calls` chunk with `index=0`,
//!        an `id`, `type="function"`, and `function.name="get_weather"`.
//!      - At least one `delta.tool_calls` chunk with a non-empty
//!        `function.arguments` fragment.
//!      - Concatenated `function.arguments` is valid JSON.
//!      - Concatenated `function.arguments` contains the `"location"` key
//!        (the only required parameter).
//!      - The final chunk has `finish_reason="tool_calls"`.
//!      - Zero `delta.content` chunks emitted (constrained path must
//!        never produce free text alongside a tool call).
//!
//! **Non-streaming test** (`tool_call_grammar_constrained_nonstream`):
//!   Same setup.  Sends `stream:false`.  Because W-α2 A3 (non-streaming
//!   `tool_calls` population fix) is a dependency, this test probes
//!   whether the fix has landed:
//!   - If `message.tool_calls` is populated and well-formed → PASS.
//!   - If `message.tool_calls` is absent/null (hardcoded `None` still in
//!     `handlers.rs:396`) → FAIL with a clear diagnostic that names the
//!     missing fix so no one papers over it.
//!   - Additionally asserts `finish_reason="tool_calls"` and
//!     `message.content` is null/absent.
//!
//! # Env gate
//!
//!   * `HF2Q_TOOL_CALL_GRAMMAR_LIVE_TEST=1` — required; absent → skip.
//!   * `HF2Q_TOOL_GRAMMAR_GEMMA_GGUF=<path>` — GGUF path; defaults to the
//!     canonical abliterated-DWQ path.
//!
//! # Run
//!
//! ```ignore
//! HF2Q_TOOL_CALL_GRAMMAR_LIVE_TEST=1 \
//!   cargo test --release --test tool_call_grammar_constrained \
//!   -- --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` per the OOM-prevention directive (one model at a time).

#![allow(dead_code)]

use futures_util::StreamExt;
use std::path::PathBuf;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ENV_GATE: &str = "HF2Q_TOOL_CALL_GRAMMAR_LIVE_TEST";
const ENV_GEMMA_GGUF: &str = "HF2Q_TOOL_GRAMMAR_GEMMA_GGUF";
const DEFAULT_GEMMA_GGUF: &str = concat!(
    "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/",
    "gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
);

const HOST: &str = "127.0.0.1";
/// Port: distinct from all live-test servers.
///   prompt_cache_live:              52335
///   tool_call_grammar_live (wave-2): 52336
///   openwebui_multiturn:            52334
///   vision_e2e_vs_mlx_vlm:         18181
///   mmproj_llama_cpp_compat:        52226
const PORT: u16 = 52337;

const READYZ_BUDGET_SECS: u64 = 600;
const REQUEST_TIMEOUT_SECS: u64 = 180;
const MAX_TOKENS: u64 = 128;

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

fn wait_for_readyz() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Instant;

    let started = Instant::now();
    let mut last_err: Option<String> = None;
    while started.elapsed().as_secs() < READYZ_BUDGET_SECS {
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
                .ok_or_else(|| std::io::Error::other(format!("bad status: {head_s:?}")))?;
            Ok(code)
        })();
        match result {
            Ok(200) => {
                eprintln!(
                    "tool_call_grammar_constrained: /readyz=200 after {}s",
                    started.elapsed().as_secs()
                );
                return;
            }
            Ok(code) => last_err = Some(format!("status={code}")),
            Err(e) => last_err = Some(format!("transport: {e}")),
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    panic!(
        "tool_call_grammar_constrained: /readyz did not reach 200 within {READYZ_BUDGET_SECS}s; \
         last_err={}",
        last_err.unwrap_or_else(|| "<none>".into())
    );
}

fn base_url() -> String {
    format!("http://{HOST}:{PORT}")
}

// ---------------------------------------------------------------------------
// Accumulated tool-call state for streaming
// ---------------------------------------------------------------------------

/// Per-stream tool-call accumulator. Mirrors the openwebui_tools.rs pattern.
#[derive(Debug, Default)]
struct AccumulatedToolCall {
    pub id: Option<String>,
    pub call_type: Option<String>,
    pub name: Option<String>,
    /// Concatenated `function.arguments` fragments.
    pub arguments: String,
}

/// Full streaming result extracted from SSE frames.
#[derive(Debug, Default)]
struct StreamResult {
    /// All accumulated tool calls indexed by `delta.tool_calls[*].index`.
    pub tool_calls: Vec<AccumulatedToolCall>,
    /// All `delta.content` text concatenated; should be empty for a pure
    /// constrained tool-call stream.
    pub content_text: String,
    pub finish_reason: Option<String>,
    pub chunk_count: usize,
    pub tool_call_chunk_count: usize,
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// Build the canonical tool-call request body for `get_weather`.
fn weather_tool_request(stream: bool, model_id: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "What is the current weather in Paris, France?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country, e.g. 'Paris, France'"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        // Constrained function choice: model MUST call get_weather.
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "get_weather"
            }
        },
        "stream": stream,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0
    })
}

/// Fetch the first chat-eligible model id from `/v1/models`.
async fn fetch_model_id(client: &reqwest::Client) -> String {
    let resp = client
        .get(&format!("{}/v1/models", base_url()))
        .send()
        .await
        .expect("GET /v1/models");
    assert_eq!(resp.status().as_u16(), 200, "/v1/models returned non-200");
    let v: serde_json::Value = resp.json().await.expect("parse /v1/models JSON");
    let data = v["data"]
        .as_array()
        .expect("/v1/models missing data array");
    // Prefer models with a context_length (chat-eligible).
    data.iter()
        .find(|m| !m["context_length"].is_null())
        .or_else(|| data.first())
        .and_then(|m| m["id"].as_str())
        .map(|s| s.to_string())
        .expect("no model id in /v1/models response")
}

/// POST `stream:true` and consume SSE frames until `[DONE]`.
async fn stream_chat_completions(
    client: &reqwest::Client,
    body: &serde_json::Value,
) -> StreamResult {
    let resp = client
        .post(&format!("{}/v1/chat/completions", base_url()))
        .json(body)
        .send()
        .await
        .expect("POST /v1/chat/completions (stream)");

    let status = resp.status().as_u16();
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    assert_eq!(
        status, 200,
        "streaming chat completion returned status={status}; Content-Type={ct}"
    );
    assert!(
        ct.contains("text/event-stream"),
        "streaming response Content-Type must contain 'text/event-stream'; got {ct:?}"
    );

    let mut result = StreamResult::default();
    let mut sse_buf = String::new();
    let mut byte_stream = resp.bytes_stream();

    while let Some(chunk) = byte_stream.next().await {
        let bytes = chunk.expect("SSE bytes_stream error");
        let s = std::str::from_utf8(&bytes).expect("SSE chunk not UTF-8");
        sse_buf.push_str(s);

        // Process complete lines from buffer.
        loop {
            let Some(newline_pos) = sse_buf.find('\n') else {
                break;
            };
            let line = sse_buf[..newline_pos].trim_end_matches('\r').to_string();
            sse_buf = sse_buf[newline_pos + 1..].to_string();

            if line == "data: [DONE]" {
                return result;
            }
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
                continue;
            };

            result.chunk_count += 1;

            // Accumulate delta.content.
            if let Some(text) = v["choices"][0]["delta"]["content"].as_str() {
                if !text.is_empty() {
                    result.content_text.push_str(text);
                }
            }

            // Accumulate finish_reason.
            if let Some(fr) = v["choices"][0]["finish_reason"].as_str() {
                result.finish_reason = Some(fr.to_string());
            }

            // Accumulate delta.tool_calls.
            if let Some(tcs) = v["choices"][0]["delta"]["tool_calls"].as_array() {
                result.tool_call_chunk_count += 1;
                for tc in tcs {
                    let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                    while result.tool_calls.len() <= idx {
                        result.tool_calls.push(AccumulatedToolCall::default());
                    }
                    let slot = &mut result.tool_calls[idx];
                    if let Some(id) = tc["id"].as_str() {
                        slot.id = Some(id.to_string());
                    }
                    if let Some(t) = tc["type"].as_str() {
                        slot.call_type = Some(t.to_string());
                    }
                    if let Some(n) = tc["function"]["name"].as_str() {
                        slot.name = Some(n.to_string());
                    }
                    if let Some(args) = tc["function"]["arguments"].as_str() {
                        slot.arguments.push_str(args);
                    }
                }
            }
        }
    }

    result
}

/// POST `stream:false` and return the parsed JSON body.
async fn nonstream_chat_completions(
    client: &reqwest::Client,
    body: &serde_json::Value,
) -> serde_json::Value {
    let resp = client
        .post(&format!("{}/v1/chat/completions", base_url()))
        .json(body)
        .send()
        .await
        .expect("POST /v1/chat/completions (non-stream)");
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
    assert_eq!(
        status, 200,
        "non-streaming chat completion returned status={status}; body={text}"
    );
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("non-streaming response is not JSON ({e}): {text}"))
}

// ---------------------------------------------------------------------------
// Test: streaming constrained path
// ---------------------------------------------------------------------------

/// Validates the grammar-constrained streaming path with
/// `tool_choice={type:function,function:{name:get_weather}}` + `stream:true`.
///
/// This is the REAL test that wave-2's sham never exercised. The grammar
/// constraint fires in the engine's decode loop, the ToolCallSplitter detects
/// the model's per-model call markers, and the SSE encoder serializes structured
/// `delta.tool_calls` frames per the OpenAI spec.
#[test]
fn tool_call_grammar_constrained_stream() {
    if std::env::var(ENV_GATE).as_deref() != Ok("1") {
        eprintln!("[SKIP] {ENV_GATE}!=1 — set {ENV_GATE}=1 to run");
        return;
    }

    let gguf = std::env::var(ENV_GEMMA_GGUF)
        .unwrap_or_else(|_| DEFAULT_GEMMA_GGUF.to_string());

    if !PathBuf::from(&gguf).exists() {
        eprintln!(
            "[SKIP] Gemma GGUF not found at {gguf}; set {ENV_GEMMA_GGUF}=<path>"
        );
        return;
    }

    eprintln!(
        "tool_call_grammar_constrained [stream]: spawning hf2q at {HOST}:{PORT} model={gguf}"
    );

    let _guard = ServerGuard::spawn(&gguf)
        .unwrap_or_else(|e| panic!("failed to spawn hf2q serve: {e}"));
    wait_for_readyz();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build()
            .expect("build reqwest client");

        let model_id = fetch_model_id(&client).await;
        eprintln!("tool_call_grammar_constrained [stream]: model_id={model_id}");

        let body = weather_tool_request(true, &model_id);
        let result = stream_chat_completions(&client, &body).await;

        eprintln!(
            "tool_call_grammar_constrained [stream]: \
             chunk_count={} tool_call_chunk_count={} \
             finish_reason={:?} tool_calls={} content_len={}",
            result.chunk_count,
            result.tool_call_chunk_count,
            result.finish_reason,
            result.tool_calls.len(),
            result.content_text.len(),
        );

        // --- Assert 1: at least one tool_calls chunk was emitted ---
        assert!(
            result.tool_call_chunk_count > 0,
            "FAIL: zero delta.tool_calls chunks received. \
             The constrained path (tool_choice.type=function) should force the \
             grammar-constrained decode path which emits GenerationEvent::ToolCallDelta \
             from engine.rs:1691-1714. If zero chunks arrived, the grammar constraint \
             did not fire or the ToolCallSplitter did not detect the model's call \
             markers. Check: (1) compile_tool_grammar returned Some(grammar) for this \
             request, (2) the model emitted its call markers (Gemma4: \
             '<|tool_call>call:get_weather{{...}}<tool_call|>'), \
             (3) ToolCallSplitter::from_registration returned Some for this model."
        );

        // --- Assert 2: index=0 slot is present with id + type + name ---
        assert!(
            !result.tool_calls.is_empty(),
            "FAIL: tool_calls accumulator is empty after parsing all SSE frames."
        );

        let slot0 = &result.tool_calls[0];

        assert!(
            slot0.id.is_some(),
            "FAIL: delta.tool_calls[0].id was never set. \
             The engine emits id on the first ToolCallDelta chunk for index=0 \
             (engine.rs:1693); it must appear in the SSE stream."
        );
        let id = slot0.id.as_deref().unwrap();
        assert!(
            !id.is_empty(),
            "FAIL: delta.tool_calls[0].id is empty string."
        );

        assert_eq!(
            slot0.call_type.as_deref(),
            Some("function"),
            "FAIL: delta.tool_calls[0].type must be 'function' per OpenAI spec; \
             got {:?}",
            slot0.call_type
        );

        assert_eq!(
            slot0.name.as_deref(),
            Some("get_weather"),
            "FAIL: delta.tool_calls[0].function.name must be 'get_weather' \
             (tool_choice.function.name constrains to exactly this function); \
             got {:?}",
            slot0.name
        );

        // --- Assert 3: arguments JSON is non-empty and valid ---
        assert!(
            !slot0.arguments.is_empty(),
            "FAIL: delta.tool_calls[0].function.arguments is empty after \
             concatenating all argument fragments. The grammar-constrained \
             decode must produce at least {{}} for the required 'location' key."
        );

        let args_val: serde_json::Value = serde_json::from_str(&slot0.arguments)
            .unwrap_or_else(|e| {
                panic!(
                    "FAIL: concatenated delta.tool_calls[0].function.arguments is \
                     not valid JSON ({e}).\n\
                     Raw arguments string: {:?}\n\
                     This means the grammar constraint produced malformed JSON \
                     or the ToolCallSplitter accumulated a partial body.",
                    slot0.arguments
                )
            });

        // --- Assert 4: required key 'location' is present ---
        assert!(
            !args_val["location"].is_null(),
            "FAIL: arguments JSON is missing required key 'location'.\n\
             Parsed arguments: {args_val}\n\
             The grammar schema has 'location' as a required property; the \
             grammar-constrained model must produce it."
        );

        eprintln!(
            "tool_call_grammar_constrained [stream]: arguments={} location={:?}",
            slot0.arguments,
            args_val["location"]
        );

        // --- Assert 5: finish_reason == "tool_calls" ---
        assert_eq!(
            result.finish_reason.as_deref(),
            Some("tool_calls"),
            "FAIL: finish_reason must be 'tool_calls' per OpenAI spec when a \
             structured tool call is emitted (engine.rs:2149-2150). Got {:?}.",
            result.finish_reason
        );

        // --- Assert 6: no delta.content emitted (constrained path only) ---
        // The grammar constrains the model to emit ONLY the call body; any
        // free text before the open marker is a model-side deviation, not a
        // system bug. We assert strictly: a pure constrained stream must not
        // produce content alongside tool_calls.
        assert!(
            result.content_text.is_empty(),
            "FAIL: delta.content was non-empty alongside delta.tool_calls.\n\
             Accumulated content text: {:?}\n\
             A grammar-constrained stream should produce zero content chunks \
             because the model is forced to emit only the call body between \
             its per-model markers. If content leaked, the ToolCallSplitter may \
             have emitted a Content event for bytes before the open marker.",
            result.content_text
        );

        eprintln!("tool_call_grammar_constrained [stream]: ALL ASSERTIONS PASSED");
        eprintln!(
            "  id={id} type=function name=get_weather \
             arguments={} finish_reason=tool_calls",
            slot0.arguments
        );
    });
}

// ---------------------------------------------------------------------------
// Test: non-streaming constrained path
// ---------------------------------------------------------------------------

/// Validates the grammar-constrained non-streaming path with
/// `tool_choice={type:function,function:{name:get_weather}}` + `stream:false`.
///
/// DEPENDENCY: W-α2 item A3 — non-streaming `tool_calls` population.
///
/// The non-streaming response assembler (`handlers.rs:384-416`) currently
/// hardcodes `tool_calls: None`. W-α2 A3 was supposed to add:
///   1. A call to `parse_tool_call_body(registration, &result.text)` on the
///      decoded text.
///   2. Populating `message.tool_calls` with the structured `ToolCall` vec.
///   3. Setting `content` to `None` when tool_calls is populated.
///   4. Overriding `finish_reason` to `"tool_calls"`.
///
/// This test detects whether A3 has landed:
///   - PASS: `message.tool_calls` is a non-empty array with correct shape.
///   - FAIL with a clear diagnostic: `message.tool_calls` is absent/null
///     (hardcoded None still present). Failing here means W-α2 A3 must be
///     implemented before this test can pass.
#[test]
fn tool_call_grammar_constrained_nonstream() {
    if std::env::var(ENV_GATE).as_deref() != Ok("1") {
        eprintln!("[SKIP] {ENV_GATE}!=1 — set {ENV_GATE}=1 to run");
        return;
    }

    let gguf = std::env::var(ENV_GEMMA_GGUF)
        .unwrap_or_else(|_| DEFAULT_GEMMA_GGUF.to_string());

    if !PathBuf::from(&gguf).exists() {
        eprintln!(
            "[SKIP] Gemma GGUF not found at {gguf}; set {ENV_GEMMA_GGUF}=<path>"
        );
        return;
    }

    eprintln!(
        "tool_call_grammar_constrained [non-stream]: spawning hf2q at {HOST}:{PORT} model={gguf}"
    );

    let _guard = ServerGuard::spawn(&gguf)
        .unwrap_or_else(|e| panic!("failed to spawn hf2q serve: {e}"));
    wait_for_readyz();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build()
            .expect("build reqwest client");

        let model_id = fetch_model_id(&client).await;
        eprintln!(
            "tool_call_grammar_constrained [non-stream]: model_id={model_id}"
        );

        let body = weather_tool_request(false, &model_id);
        let resp = nonstream_chat_completions(&client, &body).await;

        eprintln!(
            "tool_call_grammar_constrained [non-stream]: response={}",
            serde_json::to_string_pretty(&resp).unwrap_or_else(|_| resp.to_string())
        );

        let message = &resp["choices"][0]["message"];
        let finish_reason = resp["choices"][0]["finish_reason"].as_str();

        // --- Assert: message.tool_calls is present (W-α2 A3 landed) ---
        //
        // If this assertion fails, the diagnostic is:
        //   "W-α2 A3 NOT LANDED — handlers.rs:396 still has tool_calls: None"
        //
        // Fix required in handlers.rs, inside the non-streaming response
        // builder (around line 392-400):
        //   1. Look up the model registration for this request.
        //   2. Call parse_tool_call_body(registration, &result.text).
        //   3. If Some(parsed): set message.tool_calls = Some(vec![ToolCall{...}]),
        //      set message.content = None, set finish_reason = "tool_calls".
        //   4. If None: leave existing behavior (content = result.text).
        let tool_calls = message["tool_calls"].as_array();
        assert!(
            tool_calls.is_some() && !tool_calls.unwrap().is_empty(),
            "FAIL [W-α2 A3 NOT LANDED]: non-streaming response has no tool_calls.\n\
             \n\
             handlers.rs:396 hardcodes `tool_calls: None`. The non-streaming path \
             never calls parse_tool_call_body() on result.text, so even when the \
             grammar-constrained model emits a valid call body, the response always \
             has tool_calls=null.\n\
             \n\
             To fix (handlers.rs ~line 392-400):\n\
               1. Resolve model registration for this request.\n\
               2. Call parse_tool_call_body(registration, &result.text).\n\
               3. If Some(ParsedToolCall {{name, arguments_json}}):\n\
                    message.tool_calls = Some(vec![ToolCall {{\n\
                        id: generate_tool_call_id(),\n\
                        call_type: \"function\".into(),\n\
                        function: ToolCallFunction {{\n\
                            name: parsed.name,\n\
                            arguments: parsed.arguments_json,\n\
                        }},\n\
                    }}]);\n\
                    message.content = None;\n\
                    finish_reason = \"tool_calls\";\n\
             \n\
             Full response received: {resp}"
        );

        let tc = &tool_calls.unwrap()[0];

        let tc_id = tc["id"].as_str().unwrap_or("");
        assert!(!tc_id.is_empty(), "tool_calls[0].id must be non-empty");

        assert_eq!(
            tc["type"].as_str(),
            Some("function"),
            "tool_calls[0].type must be 'function'; got {:?}",
            tc["type"]
        );

        let fn_name = tc["function"]["name"].as_str().unwrap_or("");
        assert_eq!(
            fn_name, "get_weather",
            "tool_calls[0].function.name must be 'get_weather'; got {fn_name:?}"
        );

        let fn_args_str = tc["function"]["arguments"].as_str().unwrap_or("");
        assert!(
            !fn_args_str.is_empty(),
            "tool_calls[0].function.arguments must not be empty"
        );

        let args_val: serde_json::Value = serde_json::from_str(fn_args_str)
            .unwrap_or_else(|e| {
                panic!(
                    "tool_calls[0].function.arguments is not valid JSON ({e}): {fn_args_str:?}"
                )
            });

        assert!(
            !args_val["location"].is_null(),
            "tool_calls[0].function.arguments missing required key 'location': {args_val}"
        );

        // message.content must be null/absent when tool_calls is populated.
        assert!(
            message["content"].is_null(),
            "FAIL: message.content must be null when tool_calls is populated \
             (OpenAI spec: content and tool_calls are mutually exclusive in a \
             tool-call response). Got content={:?}",
            message["content"]
        );

        // finish_reason must be "tool_calls".
        assert_eq!(
            finish_reason,
            Some("tool_calls"),
            "FAIL: finish_reason must be 'tool_calls' when message.tool_calls \
             is populated; got {:?}",
            finish_reason
        );

        eprintln!(
            "tool_call_grammar_constrained [non-stream]: ALL ASSERTIONS PASSED\n\
             id={tc_id} type=function name={fn_name} arguments={fn_args_str}"
        );
    });
}
