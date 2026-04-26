//! ADR-005 Phase 2a iter-133 Open WebUI E2E test fixture / harness.
//!
//! Shared helpers for the three-iter Open WebUI scenario suite:
//!
//!   * Iter A (this iter, `tests/openwebui_multiturn.rs`) — Scenario 1
//!     text-stream multi-turn.
//!   * Iter B (planned, `tests/openwebui_multiturn.rs` extended) —
//!     Scenario 2 tool-call round-trip.
//!   * Iter C (optional, planned) — Scenario 3 reasoning-panel
//!     display via `HF2Q_REASONING_TEST_MODEL`.
//!
//! The module lives under `tests/openwebui_helpers/` (subdirectory) so
//! Cargo's integration-test discovery does NOT treat it as a separate
//! test binary — only top-level `tests/*.rs` files are auto-compiled
//! as integration tests. Test files include this module via a
//! `#[path = "openwebui_helpers/mod.rs"]` mod declaration.
//!
//! Layout in this module:
//!
//!   * Env / fixture constants (`ENV_GATE`, `DEFAULT_CHAT_GGUF`, `PORT`, …)
//!   * `ServerGuard` — RAII wrapper around `hf2q serve` subprocess.
//!   * `wait_for_readyz` — `/readyz` poll loop using a raw `TcpStream`
//!     so the dependency surface during warmup stays narrow.
//!   * `streaming_chat` / `nonstreaming_chat` — async reqwest helpers.
//!   * SSE protocol invariant assertions (`assert_streaming_invariants`,
//!     `assert_nonstreaming_invariants`).
//!   * Fixture record / replay (`write_fixture`, `replay_fixture_assert`,
//!     `normalize_chunk_for_replay`).

#![allow(dead_code)] // helpers are reused across iters A/B/C; some are unused in iter A

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Env / fixture constants
// ---------------------------------------------------------------------------

pub const ENV_GATE: &str = "HF2Q_OPENWEBUI_E2E";
pub const ENV_GGUF: &str = "HF2Q_OPENWEBUI_E2E_GGUF";
pub const ENV_RECORD: &str = "HF2Q_OPENWEBUI_E2E_RECORD";

/// Default chat GGUF path — same fixture used by
/// `tests/mmproj_llama_cpp_compat.rs` and `tests/vision_e2e_vs_mlx_vlm.rs`.
/// Gemma 4 26B chat works text-only (mmproj is optional).
pub const DEFAULT_CHAT_GGUF: &str = concat!(
    "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/",
    "gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
);

/// High-numbered fixed port distinct from `mmproj_llama_cpp_compat.rs`
/// (52226) and `vision_e2e_vs_mlx_vlm.rs` (18181). All three tests are
/// gated by separate env vars and run with `--test-threads=1` per the
/// OOM directive, so collisions are operator error, not the harness's
/// problem — but distinct ports keep the failure mode obvious.
pub const PORT: u16 = 52334;
pub const HOST: &str = "127.0.0.1";

/// `/readyz` poll budget. Cold-load + warmup of a 16GB chat GGUF on M5 Max
/// is on the order of 1-3 minutes; 10 minutes is the same budget that
/// `tests/vision_e2e_vs_mlx_vlm.rs` and `tests/mmproj_llama_cpp_compat.rs`
/// use, kept symmetric to avoid harness drift.
pub const READYZ_BUDGET_SECS: u64 = 600;

/// Streaming SSE read budget per request — generous so a slow decode pass
/// doesn't false-fail. 120s is more than enough at gemma4-26B's observed
/// ~30 tok/s decode for 16-token caps.
pub const SSE_READ_BUDGET_SECS: u64 = 120;

/// Fixture truncation cap for `write_fixture` — keeps a runaway record
/// pass from committing multi-MB SSE dumps to the tree.
pub const FIXTURE_MAX_BYTES: usize = 50 * 1024;

pub fn base_url() -> String {
    format!("http://{HOST}:{PORT}")
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

/// RAII guard around the spawned `hf2q serve` subprocess. Drop kills the
/// child so a panic mid-test never strands a 16GB-resident server.
pub struct ServerGuard(Child);

impl ServerGuard {
    pub fn spawn(gguf: &str) -> std::io::Result<Self> {
        let bin = std::env::var("CARGO_BIN_EXE_hf2q").unwrap_or_else(|_| {
            // Fallback when run outside cargo (e.g. cargo nextest). The env
            // var is the cargo-canonical lookup.
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target/release/hf2q")
                .to_string_lossy()
                .into_owned()
        });
        let child = Command::new(&bin)
            .args([
                "serve",
                "--model", gguf,
                "--host", HOST,
                "--port", &PORT.to_string(),
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

/// Poll `/readyz` until it returns 200 or `READYZ_BUDGET_SECS` elapses.
/// Uses a raw 1-shot TcpStream rather than reqwest::blocking inside the
/// hot poll loop — keeps the dependency surface narrow and avoids a
/// hot reqwest client while warmup is doing 30+ GB of GPU loads.
pub fn wait_for_readyz() {
    let started = Instant::now();
    let mut last_err: Option<String> = None;
    while started.elapsed().as_secs() < READYZ_BUDGET_SECS {
        match http_get_status(HOST, PORT, "/readyz") {
            Ok(200) => {
                eprintln!(
                    "openwebui_helpers: /readyz=200 after {}s",
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
        "openwebui_helpers: /readyz did not reach 200 within {READYZ_BUDGET_SECS}s; \
         last_err={}",
        last_err.unwrap_or_else(|| "<none>".into())
    );
}

/// Minimal HTTP/1.1 GET → status code, no body. Same idiom as
/// `tests/mmproj_llama_cpp_compat.rs`.
pub fn http_get_status(host: &str, port: u16, path: &str) -> std::io::Result<u16> {
    use std::net::TcpStream;
    let mut s = TcpStream::connect_timeout(
        &format!("{host}:{port}").parse().map_err(std::io::Error::other)?,
        Duration::from_secs(5),
    )?;
    s.set_read_timeout(Some(Duration::from_secs(5)))?;
    s.write_all(
        format!("GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n").as_bytes(),
    )?;
    let mut head = [0u8; 64];
    let n = s.read(&mut head)?;
    let head_s = std::str::from_utf8(&head[..n]).unwrap_or("");
    // "HTTP/1.1 200 OK..." → split on whitespace, second token is status.
    let mut parts = head_s.split_whitespace();
    let _http = parts.next();
    let code = parts
        .next()
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| std::io::Error::other(format!("malformed HTTP status line: {head_s:?}")))?;
    Ok(code)
}

// ---------------------------------------------------------------------------
// Async HTTP request helpers
// ---------------------------------------------------------------------------

pub fn user_msg(text: &str) -> serde_json::Value {
    serde_json::json!({"role": "user", "content": text})
}

pub fn assistant_msg(text: &str) -> serde_json::Value {
    serde_json::json!({"role": "assistant", "content": text})
}

/// GET `/v1/models` → canonical loaded model id.
///
/// The loaded model id is `general.name` from GGUF metadata, falling
/// back to the file stem (see `src/serve/api/engine.rs:511-520`).
/// A request body with the wrong `model` field returns HTTP 400
/// `model_not_loaded`; we read /v1/models so the test never depends
/// on the GGUF's `general.name` string.
pub async fn fetch_canonical_model_id() -> String {
    // Async client only — adding `reqwest`'s `blocking` feature for
    // a single one-shot GET would pull in extra dependencies for no
    // reason; the test runtime is already built by the caller.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("build reqwest client");
    let resp = client
        .get(format!("{}/v1/models", base_url()))
        .send()
        .await
        .expect("GET /v1/models failed");
    assert_eq!(
        resp.status().as_u16(),
        200,
        "/v1/models status != 200: {:?}",
        resp.status()
    );
    let v: serde_json::Value = resp.json().await.expect("parse /v1/models JSON");
    v["data"][0]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("/v1/models response missing data[0].id: {v}"))
        .to_string()
}

/// Per-stream extracted reasoning + content slots (iter C W66). The
/// "reasoning panel" UX in Open WebUI gates on `delta.reasoning_content`:
/// chunks emitted in this slot stream into a collapsible thinking
/// section while `delta.content` chunks stream into the main reply.
///
/// Returned by `streaming_chat_extract_reasoning`. Iter A's existing
/// `streaming_chat` continues to expose the simpler
/// `(chunks, accumulated_content)` shape; iter C's reasoning test needs
/// both sides + the `finish_reason` and a record of which slot was
/// observed first to assert the `reasoning_content` precedes
/// `content` invariant per Decision #21.
#[derive(Debug, Default, Clone)]
pub struct ReasoningStreamCapture {
    pub frames: Vec<String>,
    pub accumulated_content: String,
    pub accumulated_reasoning: String,
    pub finish_reason: Option<String>,
    /// Sequence of `(slot, len)` ticks per non-empty delta. Slot is
    /// `"content"` or `"reasoning"`. Used to assert the reasoning-first
    /// ordering Decision #21 requires for the Open WebUI panel UX.
    pub slot_sequence: Vec<(&'static str, usize)>,
}

/// Streaming chat with explicit `reasoning_content` extraction. Same
/// transport as `streaming_chat`; broader return shape so iter C's
/// reasoning-panel test can assert both slots + ordering.
pub async fn streaming_chat_extract_reasoning(
    model_id: &str,
    messages: &[serde_json::Value],
    max_tokens: u64,
) -> ReasoningStreamCapture {
    use futures_util::StreamExt;

    let body = serde_json::json!({
        "model": model_id,
        "messages": messages,
        "stream": true,
        "max_tokens": max_tokens,
        "temperature": 0,
    });

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(SSE_READ_BUDGET_SECS))
        .build()
        .expect("build reqwest client");

    let resp = client
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/chat/completions failed");

    let status = resp.status().as_u16();
    if status != 200 {
        let body_text = resp
            .text()
            .await
            .unwrap_or_else(|_| "<unreadable body>".into());
        panic!(
            "/v1/chat/completions reasoning-stream status != 200: {status}; body={body_text}; \
             request body sent={}",
            serde_json::to_string(&body).unwrap_or_default()
        );
    }
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        ct.contains("text/event-stream"),
        "Content-Type missing text/event-stream: {ct:?}"
    );

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut cap = ReasoningStreamCapture::default();

    while let Some(next) = stream.next().await {
        let bytes = next.expect("SSE bytes_stream chunk error");
        let s = std::str::from_utf8(&bytes).expect("SSE chunk not valid UTF-8");
        buf.push_str(s);
        loop {
            let Some(end) = buf.find("\n\n") else { break };
            let msg = buf[..end].to_string();
            buf.drain(..end + 2);
            for line in msg.lines() {
                if let Some(payload) = line.strip_prefix("data: ") {
                    cap.frames.push(payload.to_string());
                    if payload != "[DONE]" {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) {
                            let delta = &v["choices"][0]["delta"];
                            if let Some(text) = delta["content"].as_str() {
                                if !text.is_empty() {
                                    cap.slot_sequence.push(("content", text.len()));
                                }
                                cap.accumulated_content.push_str(text);
                            }
                            if let Some(text) = delta["reasoning_content"].as_str() {
                                if !text.is_empty() {
                                    cap.slot_sequence.push(("reasoning", text.len()));
                                }
                                cap.accumulated_reasoning.push_str(text);
                            }
                            if let Some(fr) =
                                v["choices"][0]["finish_reason"].as_str()
                            {
                                cap.finish_reason = Some(fr.to_string());
                            }
                        }
                    }
                }
            }
            if cap.frames.last().map(|s| s == "[DONE]").unwrap_or(false) {
                return cap;
            }
        }
    }

    cap
}

/// Drive a streaming chat-completions request and return:
/// - the raw SSE chunk payloads (everything after `data: ` and before `\n\n`),
///   in order,
/// - the accumulated `delta.content` across all chunks.
pub async fn streaming_chat(
    model_id: &str,
    messages: &[serde_json::Value],
) -> (Vec<String>, String) {
    use futures_util::StreamExt;

    let body = serde_json::json!({
        "model": model_id,
        "messages": messages,
        "stream": true,
        "max_tokens": 16,
        "temperature": 0,
    });

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(SSE_READ_BUDGET_SECS))
        .build()
        .expect("build reqwest client");

    let resp = client
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/chat/completions failed");

    let status = resp.status().as_u16();
    if status != 200 {
        // Capture the body so the test failure is actionable rather than
        // just "got 400". The handler returns OpenAI-shaped error bodies
        // (`{"error": {"message": ..., "type": ..., "param": ...}}`).
        let body_text = resp.text().await.unwrap_or_else(|_| "<unreadable body>".into());
        panic!(
            "/v1/chat/completions stream status != 200: {status}; body={body_text}; \
             request body sent={}",
            serde_json::to_string(&body).unwrap_or_default()
        );
    }
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        ct.contains("text/event-stream"),
        "Content-Type missing text/event-stream: {ct:?}"
    );

    // ---- SSE byte-stream → frame parser -----------------------------
    //
    // SSE framing per `https://html.spec.whatwg.org/multipage/server-sent-events.html`
    // and what axum's `Sse` produces for `Event::default().data(json)`:
    //
    //   data: <payload>\n\n
    //
    // axum coalesces multi-line payloads with leading `data: ` per line;
    // for `Event::default().data(...)` we always get exactly one line.
    // Our state machine accumulates `bytes_stream()` output into a
    // running buffer and splits on `\n\n` (the SSE message terminator).
    // Within each message we strip the `data: ` prefix from the single
    // data line.
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut frames: Vec<String> = Vec::new();
    let mut accumulated_content = String::new();

    while let Some(next) = stream.next().await {
        let bytes = next.expect("SSE bytes_stream chunk error");
        let s = std::str::from_utf8(&bytes)
            .expect("SSE chunk not valid UTF-8");
        buf.push_str(s);
        loop {
            // Find end of the next SSE message.
            let Some(end) = buf.find("\n\n") else { break };
            let msg = buf[..end].to_string();
            buf.drain(..end + 2);
            // Each message is one or more `data: <line>` lines; we
            // emit one frame per `data: ` line so the assertion harness
            // sees the same granularity as a curl client would.
            for line in msg.lines() {
                if let Some(payload) = line.strip_prefix("data: ") {
                    frames.push(payload.to_string());
                    if payload != "[DONE]" {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) {
                            if let Some(text) = v["choices"][0]["delta"]["content"].as_str() {
                                accumulated_content.push_str(text);
                            }
                        }
                    }
                }
            }
            // Stop after `[DONE]` so we don't keep the connection open
            // chasing trailing keepalive comments.
            if frames.last().map(|s| s == "[DONE]").unwrap_or(false) {
                return (frames, accumulated_content);
            }
        }
    }

    // Stream ended without [DONE] — assertion harness will fail,
    // but return what we have for the diagnostic.
    (frames, accumulated_content)
}

pub async fn nonstreaming_chat(
    model_id: &str,
    messages: &[serde_json::Value],
) -> serde_json::Value {
    let body = serde_json::json!({
        "model": model_id,
        "messages": messages,
        "stream": false,
        "max_tokens": 16,
        "temperature": 0,
    });

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(SSE_READ_BUDGET_SECS))
        .build()
        .expect("build reqwest client");

    let resp = client
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/chat/completions failed");
    let status = resp.status();
    let body_text = resp.text().await.expect("read non-streaming body");
    assert_eq!(
        status.as_u16(),
        200,
        "/v1/chat/completions non-stream status != 200: {status:?} body={body_text}"
    );
    serde_json::from_str(&body_text)
        .unwrap_or_else(|e| panic!("non-streaming response is not JSON ({e}): {body_text}"))
}

// ---------------------------------------------------------------------------
// SSE protocol invariant assertions (OpenAI streaming chat-completions)
// ---------------------------------------------------------------------------

/// Assert the OpenAI streaming chat-completions protocol shape:
/// - At least one chunk has `delta.role == "assistant"` (role emitter,
///   per `src/serve/api/sse.rs:152-172`).
/// - At least one chunk has `delta.content` (content delta).
/// - At least one chunk has a `finish_reason` (final chunk).
/// - Every non-[DONE] payload parses as JSON with `object ==
///   "chat.completion.chunk"`.
/// - Last frame is the literal `[DONE]` (per `sse.rs:295`).
/// - Accumulated `delta.content` is non-empty.
pub fn assert_streaming_invariants(label: &str, chunks: &[String]) {
    assert!(
        !chunks.is_empty(),
        "{label}: SSE stream produced zero chunks"
    );

    let last = chunks
        .last()
        .expect("SSE stream produced zero chunks despite non-empty assertion");
    assert_eq!(
        last, "[DONE]",
        "{label}: expected last SSE frame == \"[DONE]\", got: {last:?}"
    );

    let mut saw_role_chunk = false;
    let mut saw_content_delta = false;
    let mut saw_finish_reason = false;
    let mut accumulated_content = String::new();
    for (i, c) in chunks.iter().enumerate() {
        if c == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(c) {
            Ok(v) => v,
            Err(e) => panic!("{label}: chunk {i} is not valid JSON ({e}): {c:?}"),
        };
        assert_eq!(
            v["object"].as_str(),
            Some("chat.completion.chunk"),
            "{label}: chunk {i} object field != chat.completion.chunk: {c}"
        );
        let delta = &v["choices"][0]["delta"];
        if delta["role"].as_str() == Some("assistant") {
            saw_role_chunk = true;
        }
        if let Some(text) = delta["content"].as_str() {
            saw_content_delta = true;
            accumulated_content.push_str(text);
        }
        if v["choices"][0]["finish_reason"].is_string() {
            saw_finish_reason = true;
        }
    }
    assert!(
        saw_role_chunk,
        "{label}: never saw a chunk with delta.role == \"assistant\""
    );
    assert!(
        saw_content_delta,
        "{label}: never saw a chunk with delta.content"
    );
    assert!(
        saw_finish_reason,
        "{label}: never saw a chunk with finish_reason"
    );
    assert!(
        !accumulated_content.trim().is_empty(),
        "{label}: accumulated delta.content was empty"
    );
}

/// Assert the non-streaming chat-completions response shape (the same
/// handler returns this when `req.stream == false`; see
/// `src/serve/api/handlers.rs:80-211`).
pub fn assert_nonstreaming_invariants(resp: &serde_json::Value) {
    assert_eq!(
        resp["object"].as_str(),
        Some("chat.completion"),
        "non-streaming response object field != chat.completion: {resp}"
    );
    assert!(
        resp["id"].as_str().is_some_and(|s| s.starts_with("chatcmpl-")),
        "non-streaming response id missing or not chatcmpl-prefixed: {resp}"
    );
    let msg = &resp["choices"][0]["message"];
    assert_eq!(
        msg["role"].as_str(),
        Some("assistant"),
        "non-streaming message.role != assistant: {resp}"
    );
    assert!(
        msg["content"].as_str().is_some_and(|s| !s.trim().is_empty()),
        "non-streaming message.content empty: {resp}"
    );
    assert!(
        resp["choices"][0]["finish_reason"].is_string(),
        "non-streaming finish_reason missing: {resp}"
    );
}

// ---------------------------------------------------------------------------
// Fixture record / replay
// ---------------------------------------------------------------------------

/// Strip per-request-ephemeral fields so a replay-asserted chunk pair
/// matches across runs.
///
/// The `id` and `created` fields are stamped by the handler at request
/// time (see `src/serve/api/handlers.rs:135-136`); they MUST differ
/// between recording and replay. Everything else (object, model,
/// system_fingerprint, choices) is request-stable at temperature=0.
///
/// Iter B-2 W66 extension: also normalize the per-tool-call wall-clock
/// `id` (`call_hf2q_<16hex>`) which the engine synthesizes from
/// `SystemTime::now()` on each emission. Same OpenAI-style ephemeral
/// shape as `chatcmpl-<uuid>`; same normalization treatment.
pub fn normalize_chunk_for_replay(s: &str) -> String {
    if s == "[DONE]" {
        return s.to_string();
    }
    let Ok(mut v) = serde_json::from_str::<serde_json::Value>(s) else {
        return s.to_string();
    };
    if let Some(obj) = v.as_object_mut() {
        if obj.contains_key("id") {
            obj.insert("id".into(), serde_json::json!("<request-id>"));
        }
        if obj.contains_key("created") {
            obj.insert("created".into(), serde_json::json!(0));
        }
    }
    // Iter B-2 W66: walk choices[*].delta.tool_calls[*] and
    // normalize per-call ids. The engine synthesizes
    // `call_hf2q_<UNIX_EPOCH-ns ^ index-mix>` so each fresh run varies.
    if let Some(choices) = v.pointer_mut("/choices").and_then(|c| c.as_array_mut()) {
        for ch in choices {
            if let Some(tcs) = ch
                .pointer_mut("/delta/tool_calls")
                .and_then(|t| t.as_array_mut())
            {
                for tc in tcs {
                    if let Some(obj) = tc.as_object_mut() {
                        if obj.contains_key("id") && !obj["id"].is_null() {
                            obj.insert(
                                "id".into(),
                                serde_json::json!("<tool-call-id>"),
                            );
                        }
                    }
                }
            }
        }
    }
    serde_json::to_string(&v).unwrap_or_else(|_| s.to_string())
}

pub fn write_fixture(path: &Path, chunks: &[String]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create fixture dir");
    }
    let mut buf = String::new();
    for c in chunks {
        let line = normalize_chunk_for_replay(c);
        if buf.len() + line.len() + 1 > FIXTURE_MAX_BYTES {
            eprintln!(
                "openwebui_helpers: fixture truncated at {}/{} bytes ({}/{} chunks); \
                 increase FIXTURE_MAX_BYTES or shorten max_tokens if intentional",
                buf.len(),
                FIXTURE_MAX_BYTES,
                buf.lines().count(),
                chunks.len()
            );
            break;
        }
        buf.push_str(&line);
        buf.push('\n');
    }
    std::fs::write(path, buf).expect("write fixture");
}

pub fn replay_fixture_assert(path: &Path, actual: &[String]) {
    let raw = std::fs::read_to_string(path).expect("read fixture");
    let expected: Vec<String> = raw
        .lines()
        .filter(|l| !l.is_empty())
        .map(|s| s.to_string())
        .collect();
    let actual_norm: Vec<String> = actual
        .iter()
        .map(|s| normalize_chunk_for_replay(s))
        .collect();

    let cmp_len = expected.len().min(actual_norm.len());
    assert!(
        cmp_len > 0,
        "fixture replay: zero chunks to compare (expected={}, actual={})",
        expected.len(),
        actual_norm.len()
    );
    for i in 0..cmp_len {
        assert_eq!(
            expected[i], actual_norm[i],
            "fixture replay mismatch at chunk {i} (after normalizing id+created):\n  \
             expected: {}\n  actual:   {}",
            expected[i], actual_norm[i]
        );
    }
    // If the fixture is the FULL turn (i.e. it ends with [DONE]), assert
    // the full chunk count matches too. If the fixture was truncated, we
    // already checked the prefix.
    if expected.last().map(|s| s == "[DONE]").unwrap_or(false) {
        assert_eq!(
            expected.len(),
            actual_norm.len(),
            "fixture replay: chunk count mismatch (fixture is full but actual differs)"
        );
    }
}
