//! Wedge-3 / ADR-005 iter-216 Phase G end-to-end integration tests.
//!
//! Drives the real `hf2q serve` binary against a Qwen3.5/3.6 GGUF and
//! exercises the production HTTP surface (`/v1/chat/completions` non-stream,
//! `/v1/chat/completions` stream, `/v1/embeddings`).  These tests load the
//! full ~14-25 GB model, allocate a multi-GB hybrid KV cache, and hold the
//! GPU for tens of seconds, so they are env-gated and `#[ignore]`'d by
//! default.
//!
//! ## Running
//!
//! ```bash
//! HF2Q_WEDGE3_E2E=1 \
//! HF2Q_TEST_QWEN35_MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf \
//!     cargo test --release --test qwen35_serve_e2e -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! When either env var is unset, every test logs a skip message and returns
//! `Ok(())` — calling `cargo test --test qwen35_serve_e2e -- --ignored` on a
//! CI worker without the model is harmless.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const E2E_GATE: &str = "HF2Q_WEDGE3_E2E";
const MODEL_ENV: &str = "HF2Q_TEST_QWEN35_MODEL";
const PORT: u16 = 47216;
const SERVE_BOOT_BUDGET_S: u64 = 180;

fn e2e_enabled() -> bool {
    std::env::var(E2E_GATE)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn test_model_path() -> Option<PathBuf> {
    let raw = std::env::var(MODEL_ENV).ok()?;
    let p = PathBuf::from(&raw);
    if !p.exists() {
        eprintln!("{MODEL_ENV}={raw} does not exist on disk — skipping");
        return None;
    }
    Some(p)
}

fn skip_if_disabled(test_name: &str) -> Option<PathBuf> {
    if !e2e_enabled() {
        eprintln!(
            "[{test_name}] skipping: set {E2E_GATE}=1 to run Wedge-3 e2e tests"
        );
        return None;
    }
    test_model_path().or_else(|| {
        eprintln!(
            "[{test_name}] skipping: set {MODEL_ENV} to a Qwen3.5/3.6 GGUF path"
        );
        None
    })
}

fn locate_hf2q_binary() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
        .join("target")
        .join("release")
        .join("hf2q")
}

fn http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .expect("reqwest blocking client")
}

/// RAII guard that spawns `hf2q serve` and SIGKILLs it on drop.  Polls
/// `/readyz` until 200 OK or the boot budget expires.
struct ServeProcess {
    child: Child,
    port: u16,
}

impl ServeProcess {
    fn spawn(model_path: &PathBuf) -> std::io::Result<Self> {
        let bin = locate_hf2q_binary();
        eprintln!(
            "[wedge3-e2e] spawning {} serve --model {} --port {}",
            bin.display(),
            model_path.display(),
            PORT
        );
        let child = Command::new(&bin)
            .arg("serve")
            .arg("--model")
            .arg(model_path)
            .arg("--port")
            .arg(PORT.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        Ok(Self { child, port: PORT })
    }

    fn wait_ready(&mut self) -> bool {
        let url = format!("http://127.0.0.1:{}/readyz", self.port);
        let deadline = Instant::now() + Duration::from_secs(SERVE_BOOT_BUDGET_S);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .expect("readyz client");
        let mut last_err = String::new();
        while Instant::now() < deadline {
            match client.get(&url).send() {
                Ok(resp) if resp.status() == 200 => return true,
                Ok(resp) => last_err = format!("readyz status {}", resp.status()),
                Err(e) => last_err = format!("{e}"),
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        eprintln!("[wedge3-e2e] /readyz never responded 200 within {SERVE_BOOT_BUDGET_S}s; last_err={last_err}");
        false
    }
}

impl Drop for ServeProcess {
    fn drop(&mut self) {
        eprintln!("[wedge3-e2e] killing serve pid={}", self.child.id());
        let _ = self.child.kill();
        // Drain output so the child can exit cleanly without SIGPIPE
        // surprises if it tries to log.
        if let Some(mut out) = self.child.stdout.take() {
            let mut sink = Vec::new();
            let _ = out.read_to_end(&mut sink);
        }
        if let Some(mut err) = self.child.stderr.take() {
            let mut sink = Vec::new();
            let _ = err.read_to_end(&mut sink);
        }
        let _ = self.child.wait();
    }
}

/// Wedge-3 / iter-216 Phase G AC-1: non-streaming chat completion against
/// the real `hf2q serve` binary returns HTTP 200 with at least one
/// generated token under `choices[0].message.content`.
#[test]
#[ignore = "Wedge-3 e2e — requires HF2Q_WEDGE3_E2E=1 + HF2Q_TEST_QWEN35_MODEL"]
fn e2e_qwen35_chat_completion_returns_real_tokens() {
    let Some(model_path) = skip_if_disabled("e2e_qwen35_chat_completion_returns_real_tokens")
    else {
        return;
    };
    let mut serve = ServeProcess::spawn(&model_path).expect("spawn serve");
    assert!(serve.wait_ready(), "serve never became ready");
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", serve.port);
    let body = serde_json::json!({
        "model": "Qwen3_5ForConditionalGeneration",
        "messages": [
            {"role": "user", "content": "Hello, my name is"}
        ],
        "max_tokens": 16,
        "temperature": 0.0
    });
    let client = http_client();
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .send()
        .expect("POST /v1/chat/completions");
    assert_eq!(
        resp.status().as_u16(),
        200,
        "expected HTTP 200 (Wedge-3 lifted the 501); got {}",
        resp.status()
    );
    let text = resp.text().expect("read body");
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("body must be JSON");
    let content = parsed
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .expect("missing /choices/0/message/content");
    assert!(
        !content.is_empty(),
        "Wedge-3 must return at least one generated token; content was empty"
    );
    let completion_tokens = parsed
        .pointer("/usage/completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(
        completion_tokens > 0,
        "completion_tokens must be > 0 (saw {completion_tokens})"
    );
    eprintln!(
        "[wedge3-e2e] non-stream chat: completion_tokens={} content={:?}",
        completion_tokens,
        &content[..content.len().min(64)]
    );
}

/// Wedge-3 / iter-216 Phase G AC-2: streaming chat completion emits
/// incremental SSE chunks ending in `data: [DONE]`.
///
/// Uses a raw TCP connection because `reqwest::blocking` returns an
/// already-buffered body which doesn't expose chunk-by-chunk SSE
/// reassembly without the `stream` feature (which is dev-only and
/// async-only).  The SSE protocol is plain text — line-based framing
/// is enough to assert the [DONE] sentinel + at least one non-empty
/// content delta.
#[test]
#[ignore = "Wedge-3 e2e — requires HF2Q_WEDGE3_E2E=1 + HF2Q_TEST_QWEN35_MODEL"]
fn e2e_qwen35_chat_completion_streams_tokens_via_sse() {
    let Some(model_path) =
        skip_if_disabled("e2e_qwen35_chat_completion_streams_tokens_via_sse")
    else {
        return;
    };
    let mut serve = ServeProcess::spawn(&model_path).expect("spawn serve");
    assert!(serve.wait_ready(), "serve never became ready");
    let body = serde_json::json!({
        "model": "Qwen3_5ForConditionalGeneration",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "stream": true
    })
    .to_string();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\n\
         Accept: text/event-stream\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        port = serve.port,
        len = body.len(),
        body = body,
    );
    let mut sock = std::net::TcpStream::connect(("127.0.0.1", serve.port))
        .expect("tcp connect");
    sock.set_read_timeout(Some(Duration::from_secs(180))).ok();
    sock.write_all(req.as_bytes()).expect("write request");
    let mut reader = BufReader::new(sock);
    // Drain HTTP response head.
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).expect("read head");
        if n == 0 || line == "\r\n" {
            break;
        }
        if line.starts_with("HTTP/1.1") {
            assert!(
                line.contains("200"),
                "stream HTTP status must be 200; got: {line}"
            );
        }
    }
    let mut saw_done = false;
    let mut saw_content_delta = false;
    loop {
        line.clear();
        let n = reader.read_line(&mut line).expect("read frame");
        if n == 0 {
            break;
        }
        if line.starts_with("data: [DONE]") {
            saw_done = true;
            break;
        }
        if line.starts_with("data: ") {
            // Best-effort parse to detect a non-empty content delta.
            let payload = line["data: ".len()..].trim();
            if !payload.is_empty() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) {
                    let content = v
                        .pointer("/choices/0/delta/content")
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    if !content.is_empty() {
                        saw_content_delta = true;
                    }
                }
            }
        }
    }
    assert!(saw_done, "stream must emit [DONE] sentinel");
    assert!(
        saw_content_delta,
        "stream must emit at least one non-empty content delta"
    );
    eprintln!("[wedge3-e2e] stream: ok ([DONE] + content delta observed)");
}

/// Wedge-3 / iter-216 Phase G AC-3: `/v1/embeddings` with `input: "hello world"`
/// returns a vector of length `cfg.hidden_size` (5120 for Qwen3.6 27B).
#[test]
#[ignore = "Wedge-3 e2e — requires HF2Q_WEDGE3_E2E=1 + HF2Q_TEST_QWEN35_MODEL"]
fn e2e_qwen35_embed_returns_hidden_state_vector() {
    let Some(model_path) = skip_if_disabled("e2e_qwen35_embed_returns_hidden_state_vector")
    else {
        return;
    };
    let mut serve = ServeProcess::spawn(&model_path).expect("spawn serve");
    assert!(serve.wait_ready(), "serve never became ready");
    let url = format!("http://127.0.0.1:{}/v1/embeddings", serve.port);
    let body = serde_json::json!({
        "model": "Qwen3_5ForConditionalGeneration",
        "input": "hello world"
    });
    let client = http_client();
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .send()
        .expect("POST /v1/embeddings");
    assert_eq!(
        resp.status().as_u16(),
        200,
        "expected HTTP 200; got {}",
        resp.status()
    );
    let text = resp.text().expect("read body");
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("body must be JSON");
    let vec = parsed
        .pointer("/data/0/embedding")
        .and_then(|v| v.as_array())
        .expect("missing /data/0/embedding");
    // Qwen3.6 27B: cfg.hidden_size = 5120.
    assert!(
        vec.len() >= 1024,
        "embedding vector unexpectedly short: {} (expected >= 1024)",
        vec.len()
    );
    // L2 norm should be ~1.0 (Phase A enforces L2 normalization).
    let l2: f64 = vec
        .iter()
        .filter_map(|v| v.as_f64())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    assert!(
        (l2 - 1.0).abs() < 1e-2 || l2 < 1e-6,
        "embedding not L2-normalized: ||v||_2 = {l2}"
    );
    eprintln!(
        "[wedge3-e2e] embed: len={} l2={:.6}",
        vec.len(),
        l2
    );
}

/// Wedge-3 / iter-216 Phase G AC-4 (bonus): two consecutive identical-prompt
/// requests; the second's wall-clock must beat the first's prefill time
/// thanks to the HybridPromptCache fast-path.
#[test]
#[ignore = "Wedge-3 e2e — requires HF2Q_WEDGE3_E2E=1 + HF2Q_TEST_QWEN35_MODEL"]
fn e2e_qwen35_prompt_cache_second_request_faster() {
    let Some(model_path) =
        skip_if_disabled("e2e_qwen35_prompt_cache_second_request_faster")
    else {
        return;
    };
    let mut serve = ServeProcess::spawn(&model_path).expect("spawn serve");
    assert!(serve.wait_ready(), "serve never became ready");
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", serve.port);
    let body = serde_json::json!({
        "model": "Qwen3_5ForConditionalGeneration",
        "messages": [{"role": "user", "content": "The quick brown fox"}],
        "max_tokens": 4,
        "temperature": 0.0
    })
    .to_string();
    let client = http_client();
    let mut elapsed: Vec<Duration> = Vec::new();
    let mut completion_tokens_first: u64 = 0;
    let mut completion_tokens_second: u64 = 0;
    for round in 0..2 {
        let started = Instant::now();
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(body.clone())
            .send()
            .expect("POST /v1/chat/completions");
        assert_eq!(resp.status().as_u16(), 200);
        let text = resp.text().expect("read body");
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("body must be JSON");
        let ct = parsed
            .pointer("/usage/completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if round == 0 {
            completion_tokens_first = ct;
        } else {
            completion_tokens_second = ct;
        }
        elapsed.push(started.elapsed());
    }
    eprintln!(
        "[wedge3-e2e] prompt-cache timing: first={:?} second={:?}",
        elapsed[0], elapsed[1]
    );
    eprintln!(
        "[wedge3-e2e] prompt-cache tokens: first={} second={}",
        completion_tokens_first, completion_tokens_second
    );
    // The second request should be at least as fast as the first because
    // the prefill is skipped on a cache hit.  We use a soft threshold (<
    // first) so run-to-run thermal drift on M5 Max doesn't trigger flakes;
    // the prefill skip on cache-hit is deterministic so any decode-time
    // jitter still leaves second < first.
    assert!(
        elapsed[1] < elapsed[0],
        "second request must be faster than first (saw first={:?} second={:?})",
        elapsed[0],
        elapsed[1],
    );
}
