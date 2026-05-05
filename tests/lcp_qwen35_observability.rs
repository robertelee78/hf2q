//! ADR-017 Phase B-hybrid.1 — Qwen 3.5/3.6 LCP partial-prefill resume
//! observability falsifier.
//!
//! Asserts that the engine's LCP probe RUNS on the Qwen35 path after
//! a `HybridPromptCache::try_match` miss, advancing the
//! `hf2q_kv_lcp_lookups_total` counter on `/metrics`.
//!
//! ## Iter B.1 scope
//!
//! This test guards the Phase B-hybrid.1 observability foundation:
//! the `lcp_registry` field on `Qwen35LoadedModel`, the probe call in
//! both non-streaming (`generate_qwen35_once`) and streaming
//! (`generate_stream_qwen35_once_extended`) paths, and the
//! `kv_metrics_sink.record_lcp_probe(...)` wire-up.
//!
//! Iter B.2 (next iter) will add the byte-identity falsifier on top:
//! after wiring chunk-aligned DeltaNet checkpoints + chunked prefill,
//! a long-prompt request whose LCP exceeds one chunk-stride should
//! hit `lcp_detected_total >= 1` AND produce byte-identical decoded
//! tokens to a fresh-prefill control. iter B.1 ONLY guards that the
//! probe runs (denominator counter advances).
//!
//! ## Why a separate test file (vs lcp_partial_prefill_byte_identity.rs)
//!
//! `lcp_partial_prefill_byte_identity.rs` is Gemma-specific (uses
//! Gemma-shaped prompts and Gemma fixture). The Qwen35 path has its
//! own model fixture (`qwen3.6-27b-dwq46.gguf`) and different
//! request shape. Keeping the test files family-scoped clarifies
//! ownership and avoids fixture-mixing bugs.
//!
//! ## Gating
//!
//! Operator-gated under `HF2Q_KV_PERSIST_PHASE_D=1` AND
//! `HF2Q_KV_PERSIST_QWEN35_E2E_MODEL_PATH=<gguf>` (separate env from
//! Gemma's `HF2Q_KV_PERSIST_E2E_MODEL_PATH` so both can be run in
//! sequence with different fixtures). Default `cargo test`
//! short-circuits.

use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const ENV_PHASE_D_GATE: &str = "HF2Q_KV_PERSIST_PHASE_D";
const ENV_QWEN35_MODEL_PATH: &str = "HF2Q_KV_PERSIST_QWEN35_E2E_MODEL_PATH";
const PORT: u16 = 52401;
const HOST: &str = "127.0.0.1";
const READYZ_BUDGET: Duration = Duration::from_secs(180);

/// Two prompts that share a prefix at the chat-template-rendered token
/// level. `Q` ends mid-sentence; `P` appends a few tokens of distinct
/// continuation. Greedy decode at temperature=0 gives byte-stable
/// output deterministically. Same shape as iter-3 falsifier's PROMPT_Q
/// / PROMPT_P pair.
const PROMPT_Q: &str = "List in alphabetical order the colors red, green, blue,";
const PROMPT_P: &str =
    "List in alphabetical order the colors red, green, blue, and yellow with one descriptor each.";
const MAX_TOKENS: u32 = 16;

fn hf2q_binary_path() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir).join("target/release/hf2q")
}

fn resolve_qwen35_model_path_or_skip() -> Option<PathBuf> {
    if std::env::var(ENV_PHASE_D_GATE).as_deref() != Ok("1") {
        return None;
    }
    let path = std::env::var(ENV_QWEN35_MODEL_PATH).ok()?;
    if path.is_empty() {
        return None;
    }
    let p = PathBuf::from(path);
    if !p.exists() {
        panic!(
            "[Phase B-hybrid.1] {ENV_PHASE_D_GATE}=1 set but \
             {ENV_QWEN35_MODEL_PATH} does not point to an existing file"
        );
    }
    Some(p)
}

struct ServerGuard {
    child: Child,
    port: u16,
    stderr_tail: Arc<Mutex<Vec<String>>>,
    _stderr_thread: Option<thread::JoinHandle<()>>,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl ServerGuard {
    fn log_tail(&self) -> Vec<String> {
        self.stderr_tail
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }
}

fn spawn_server(bin: &Path, model: &Path, port: u16) -> ServerGuard {
    let mut cmd = Command::new(bin);
    cmd.args([
        "serve",
        "--model",
        &model.to_string_lossy(),
        "--host",
        HOST,
        "--port",
        &port.to_string(),
    ])
    .stdout(Stdio::null())
    .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn hf2q serve");
    let stderr_tail: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let stderr_thread = child.stderr.take().map(|stderr| {
        let tail = Arc::clone(&stderr_tail);
        thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut buf = String::new();
            loop {
                buf.clear();
                match reader.read_line(&mut buf) {
                    Ok(0) => break,
                    Ok(_) => {
                        if let Ok(mut g) = tail.lock() {
                            g.push(buf.trim_end_matches(['\n', '\r']).to_string());
                            let drain = g.len().saturating_sub(256);
                            if drain > 0 {
                                g.drain(..drain);
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
        })
    });
    ServerGuard {
        child,
        port,
        stderr_tail,
        _stderr_thread: stderr_thread,
    }
}

fn http_get_status(port: u16, path: &str) -> std::io::Result<u16> {
    let mut s = TcpStream::connect((HOST, port))?;
    s.set_read_timeout(Some(Duration::from_secs(5)))?;
    s.set_write_timeout(Some(Duration::from_secs(5)))?;
    use std::io::Write;
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: {HOST}:{port}\r\nConnection: close\r\n\r\n"
    )?;
    let mut reader = BufReader::new(s);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("malformed status line: {line:?}"),
        ));
    }
    parts[1]
        .parse::<u16>()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
}

fn wait_for_readyz(server: &ServerGuard) {
    let start = Instant::now();
    while start.elapsed() < READYZ_BUDGET {
        if let Ok(200) = http_get_status(server.port, "/readyz") {
            return;
        }
        thread::sleep(Duration::from_millis(250));
    }
    panic!(
        "/readyz not 200 within {:?}\n--- stderr_tail ---\n{}",
        READYZ_BUDGET,
        server.log_tail().join("\n")
    );
}

fn fetch_metrics(server: &ServerGuard) -> String {
    use std::io::{Read, Write};
    let mut s = TcpStream::connect((HOST, server.port)).expect("connect /metrics");
    s.set_read_timeout(Some(Duration::from_secs(10))).ok();
    write!(
        s,
        "GET /metrics HTTP/1.1\r\nHost: {HOST}:{}\r\nConnection: close\r\n\r\n",
        server.port
    )
    .unwrap();
    let mut buf = String::new();
    s.read_to_string(&mut buf).expect("read /metrics");
    let body_start = buf.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    buf[body_start..].to_string()
}

fn metric_lcp_lookups_total(metrics_body: &str) -> u64 {
    for line in metrics_body.lines() {
        if let Some(rest) = line.strip_prefix("hf2q_kv_lcp_lookups_total ") {
            if let Ok(n) = rest.trim().parse::<u64>() {
                return n;
            }
        }
    }
    0
}

fn fetch_canonical_model_id(server: &ServerGuard) -> String {
    use std::io::{Read, Write};
    let mut s = TcpStream::connect((HOST, server.port)).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(10))).ok();
    write!(
        s,
        "GET /v1/models HTTP/1.1\r\nHost: {HOST}:{}\r\nConnection: close\r\n\r\n",
        server.port
    )
    .unwrap();
    let mut buf = String::new();
    s.read_to_string(&mut buf).expect("read /v1/models");
    let body_start = buf.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    let body = &buf[body_start..];
    let key = "\"id\":\"";
    let p = body.find(key).expect("find model id in /v1/models");
    let rest = &body[p + key.len()..];
    let end = rest.find('"').expect("close quote on model id");
    rest[..end].to_string()
}

fn chat_decode(server: &ServerGuard, model: &str, prompt: &str, max_tokens: u32) -> String {
    use std::io::{Read, Write};
    let body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"{prompt}"}}],"max_tokens":{max_tokens},"temperature":0,"stream":false}}"#
    );
    let mut s = TcpStream::connect((HOST, server.port)).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(180))).ok();
    s.set_write_timeout(Some(Duration::from_secs(30))).ok();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {HOST}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        server.port,
        body.len(),
        body
    )
    .unwrap();
    let mut buf = String::new();
    s.read_to_string(&mut buf).expect("read response");

    let status_line_end = buf.find("\r\n").unwrap_or(buf.len());
    let status_line = &buf[..status_line_end];
    let status_code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("malformed HTTP status line: {status_line:?}"));
    assert_eq!(
        status_code, 200,
        "[chat_decode] expected HTTP 200, got {status_code}. \
         Status line: {status_line:?}"
    );

    let body_start = buf.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    let body = &buf[body_start..];
    assert!(
        body.contains("\"choices\":["),
        "[chat_decode] response body lacks `\"choices\":[` — likely an \
         error response. Body:\n{body}"
    );

    let key = "\"content\":\"";
    let p = body
        .find(key)
        .unwrap_or_else(|| panic!("no content field: {body}"));
    let rest = &body[p + key.len()..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => break,
            }
        } else if c == '"' {
            return out;
        } else {
            out.push(c);
        }
    }
    out
}

/// Phase B-hybrid.1 LCP observability falsifier.
///
/// Asserts that the LCP probe RUNS on the Qwen35 generate path after
/// a HybridPromptCache miss. Sends Q (PromptCache miss; first-time
/// prompt) then P (different prompt; PromptCache miss again).
///
/// Each request triggers one LCP probe call. After 2 requests, the
/// metric `hf2q_kv_lcp_lookups_total` should be ≥ 2 on this server.
///
/// Falsifier:
///   * `lcp_lookups_total < 2` ⇒ the probe didn't run on the Qwen35
///     path. Either kv_metrics_sink wasn't wired (mod.rs:2619) or the
///     probe call was skipped (engine_qwen35.rs).
///
/// Pre-iter-B.1 (today on origin/main): the probe doesn't exist on
/// Qwen35; the counter stays at 0. After iter-B.1, the counter
/// advances ≥ 1 per Qwen35 request (post-PromptCache-miss only).
#[test]
fn phase_b_hybrid_qwen35_lcp_probe_runs() {
    let model_path = match resolve_qwen35_model_path_or_skip() {
        Some(p) => p,
        None => {
            eprintln!(
                "[Phase B-hybrid.1] {ENV_PHASE_D_GATE}=1 + \
                 {ENV_QWEN35_MODEL_PATH}=PATH not set — short-circuit. \
                 Set both to run."
            );
            return;
        }
    };
    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "[Phase B-hybrid.1] hf2q binary not found at {} — \
         did `cargo build --release` run?",
        bin.display()
    );

    let server = spawn_server(&bin, &model_path, PORT);
    wait_for_readyz(&server);
    let canonical = fetch_canonical_model_id(&server);
    eprintln!(
        "[Phase B-hybrid.1] server ready on {}:{} model={}",
        HOST, PORT, canonical
    );

    let metrics_pre = fetch_metrics(&server);
    let lookups_pre = metric_lcp_lookups_total(&metrics_pre);
    eprintln!(
        "[Phase B-hybrid.1] baseline lcp_lookups_total = {}",
        lookups_pre
    );

    // Send Q (first-time prompt, PromptCache miss) — probe should run.
    let q_decoded = chat_decode(&server, &canonical, PROMPT_Q, MAX_TOKENS);
    assert!(
        !q_decoded.is_empty(),
        "[Phase B-hybrid.1] Q decoded empty"
    );
    eprintln!(
        "[Phase B-hybrid.1] Q decoded {} bytes",
        q_decoded.len()
    );

    // Send P (different from Q, PromptCache miss again) — probe runs.
    let p_decoded = chat_decode(&server, &canonical, PROMPT_P, MAX_TOKENS);
    assert!(
        !p_decoded.is_empty(),
        "[Phase B-hybrid.1] P decoded empty"
    );
    eprintln!(
        "[Phase B-hybrid.1] P decoded {} bytes",
        p_decoded.len()
    );

    // Assert: probe ran on at least one of the two requests.
    let metrics_post = fetch_metrics(&server);
    let lookups_post = metric_lcp_lookups_total(&metrics_post);
    let lookups_delta = lookups_post.saturating_sub(lookups_pre);
    eprintln!(
        "[Phase B-hybrid.1] lcp_lookups_total: {} → {} (Δ = {})",
        lookups_pre, lookups_post, lookups_delta
    );
    assert!(
        lookups_delta >= 2,
        "[Phase B-hybrid.1] FALSIFIED — lcp_lookups_total advanced by \
         only {} (expected ≥ 2 for two PromptCache-miss requests). The \
         Qwen35 LCP probe did NOT run.\n\
         Likely causes:\n\
         (a) lcp_registry field not added to Qwen35LoadedModel (engine_qwen35.rs:~103)\n\
         (b) probe call missing in generate_qwen35_once (engine_qwen35.rs:~700)\n\
         (c) kv_metrics_sink not wired in serve/mod.rs:~2619 for Qwen35 arm\n\n\
         metrics excerpt:\n{}",
        lookups_delta,
        metrics_post
            .lines()
            .filter(|l| l.contains("lcp"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    eprintln!(
        "[Phase B-hybrid.1] PASS — Qwen35 LCP probe runs (Δ = {} ≥ 2)",
        lookups_delta
    );
}
