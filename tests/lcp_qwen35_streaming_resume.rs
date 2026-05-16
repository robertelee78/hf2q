//! ADR-017 Phase E.a B.5 — streaming-path LCP resume engagement falsifier.
//!
//! Verifies the streaming-path wiring (`generate_stream_qwen35_once_extended`)
//! mirrors the non-streaming path's B.3 stride-aligned LCP resume:
//! mid-prefill snapshot store at stride boundaries, descending probe
//! scan, restore_partial + suffix-chunked prefill on hit.
//!
//! ## Test design
//!
//! 1. Spawn server A with `HF2Q_KV_LCP_CHUNKED_PREFILL=1
//!    HF2Q_KV_LCP_RESUME=1 HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=64`.
//! 2. Send turn-1 (non-streaming) — engages chunked prefill, stores
//!    snapshot at chunk_pos=64 (verified by non-streaming-path
//!    `[hf2q qwen35 lcp store] mid-prefill snapshot` log line).
//! 3. Send turn-2 (streaming, multi-turn payload [user X, assistant Y,
//!    user Z]) — should engage the streaming-path probe + restore.
//! 4. Verify A's stderr contains `[hf2q qwen35 stream lcp resume]
//!    STRIDE-ALIGNED HIT` (proves the streaming-path B.3 wiring
//!    fires).
//! 5. Verify the streamed response decodes non-empty content.
//!
//! ## Operator gating
//!
//! `HF2Q_KV_PERSIST_PHASE_D=1` AND
//! `HF2Q_KV_PERSIST_QWEN35_E2E_MODEL_PATH=<gguf>`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const ENV_PHASE_D_GATE: &str = "HF2Q_KV_PERSIST_PHASE_D";
const ENV_QWEN35_MODEL_PATH: &str = "HF2Q_KV_PERSIST_QWEN35_E2E_MODEL_PATH";
const PORT_A: u16 = 52461;
const HOST: &str = "127.0.0.1";
const READYZ_BUDGET: Duration = Duration::from_secs(180);
const TEST_STRIDE: u32 = 64;
const MAX_TOKENS: u32 = 16;

const TURN1_USER: &str =
    "I'm working on a Rust project organized by Domain-Driven Design \
     bounded contexts. Could you describe in detail how bounded contexts \
     in DDD map to Rust crate boundaries with a concrete example showing \
     order, payment, and inventory in a typical e-commerce system that \
     would help me structure my workspace?";
const TURN2_USER: &str = "Now in two sentences, summarize the main difference.";

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
            "[Phase B.5 stream] {ENV_PHASE_D_GATE}=1 set but \
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
        self.stderr_tail.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

fn spawn_server(
    bin: &Path,
    model: &Path,
    port: u16,
    extra_envs: &[(&str, &str)],
) -> ServerGuard {
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
    for (k, v) in extra_envs {
        cmd.env(k, v);
    }
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
                            let drain = g.len().saturating_sub(512);
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
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: {HOST}:{port}\r\nConnection: close\r\n\r\n"
    )?;
    let mut reader = BufReader::new(s);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(std::io::Error::other(
            format!("malformed status line: {line:?}"),
        ));
    }
    parts[1]
        .parse::<u16>()
        .map_err(|e| std::io::Error::other(e.to_string()))
}

fn wait_for_readyz(server: &ServerGuard) {
    let start = Instant::now();
    loop {
        if let Ok(200) = http_get_status(server.port, "/readyz") {
            return;
        }
        if start.elapsed() > READYZ_BUDGET {
            panic!(
                "[Phase B.5 stream] /readyz did not return 200 within {READYZ_BUDGET:?} \
                 on port {}",
                server.port
            );
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn fetch_canonical_model_id(server: &ServerGuard) -> String {
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
    let p = body.find(key).expect("find model id");
    let rest = &body[p + key.len()..];
    let end = rest.find('"').expect("close quote");
    rest[..end].to_string()
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Send chat completion and return decoded content (non-streaming variant).
fn chat_decode_nonstream(
    server: &ServerGuard,
    model: &str,
    messages_json: &str,
    max_tokens: u32,
) -> String {
    let body = format!(
        r#"{{"model":"{model}","messages":{messages_json},"max_tokens":{max_tokens},"temperature":0,"stream":false}}"#
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
    let body_start = buf.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    let body = &buf[body_start..];
    let key = "\"content\":\"";
    let p = body.find(key).unwrap_or_else(|| panic!("no content"));
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
            break;
        } else {
            out.push(c);
        }
    }
    out
}

/// Send streaming chat completion, accumulate SSE chunks' delta.content,
/// return the concatenated text.
fn chat_decode_stream(
    server: &ServerGuard,
    model: &str,
    messages_json: &str,
    max_tokens: u32,
) -> String {
    let body = format!(
        r#"{{"model":"{model}","messages":{messages_json},"max_tokens":{max_tokens},"temperature":0,"stream":true}}"#
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
    s.read_to_string(&mut buf).expect("read SSE response");

    // Find body start (after headers).
    let body_start = buf.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    let body = &buf[body_start..];

    // The server uses HTTP/1.1 chunked transfer encoding wrapped around
    // SSE chunks: each HTTP chunk is `<hex_size>\r\n<data>\r\n`, where
    // <data> contains one or more SSE `data: <json>\n\n` events.  Rather
    // than fully de-chunk, we just scan the entire body for `data: `
    // markers — they're unambiguous because the surrounding HTTP chunk
    // sizes are hex-only (no "data: " substring).
    let mut accumulated = String::new();
    let mut cursor = 0;
    while let Some(rel) = body[cursor..].find("data: ") {
        let abs = cursor + rel + "data: ".len();
        // Find end of this SSE event: either `\n\n` or end-of-body.
        let end = body[abs..]
            .find("\n\n")
            .map(|p| abs + p)
            .unwrap_or(body.len());
        let data = body[abs..end].trim();
        cursor = end + 2;
        if data == "[DONE]" {
            break;
        }
        // Find "content":"..." within this event's JSON.
        let key = "\"content\":\"";
        if let Some(p) = data.find(key) {
            let rest = &data[p + key.len()..];
            let mut chars = rest.chars();
            while let Some(c) = chars.next() {
                if c == '\\' {
                    match chars.next() {
                        Some('"') => accumulated.push('"'),
                        Some('\\') => accumulated.push('\\'),
                        Some('n') => accumulated.push('\n'),
                        Some('r') => accumulated.push('\r'),
                        Some('t') => accumulated.push('\t'),
                        Some(other) => {
                            accumulated.push('\\');
                            accumulated.push(other);
                        }
                        None => break,
                    }
                } else if c == '"' {
                    break;
                } else {
                    accumulated.push(c);
                }
            }
        }
    }
    accumulated
}

#[test]
fn phase_b5_streaming_lcp_resume_engagement() {
    let model_path = match resolve_qwen35_model_path_or_skip() {
        Some(p) => p,
        None => {
            eprintln!(
                "[Phase B.5 stream] {ENV_PHASE_D_GATE}=1 + \
                 {ENV_QWEN35_MODEL_PATH}=PATH not set — short-circuit."
            );
            return;
        }
    };
    let bin = hf2q_binary_path();
    assert!(bin.exists(), "[Phase B.5 stream] hf2q binary not found");

    let stride_str = TEST_STRIDE.to_string();
    let server_a = spawn_server(
        &bin,
        &model_path,
        PORT_A,
        &[
            ("HF2Q_KV_LCP_CHUNKED_PREFILL", "1"),
            ("HF2Q_KV_LCP_RESUME", "1"),
            ("HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE", &stride_str),
        ],
    );
    wait_for_readyz(&server_a);
    let canonical = fetch_canonical_model_id(&server_a);
    eprintln!(
        "[Phase B.5 stream] server A ready on {}:{} (LCP_RESUME=1, stride={})",
        HOST, PORT_A, TEST_STRIDE
    );

    // Step 1: turn-1 (non-streaming) to populate lcp_registry at chunk_pos=64.
    let turn1_messages = format!(
        r#"[{{"role":"user","content":"{}"}}]"#,
        json_escape(TURN1_USER)
    );
    let a_turn1 = chat_decode_nonstream(&server_a, &canonical, &turn1_messages, MAX_TOKENS);
    assert!(!a_turn1.is_empty(), "[Phase B.5 stream] turn-1 empty");
    eprintln!(
        "[Phase B.5 stream] turn-1 (non-stream) → {} bytes: {:?}",
        a_turn1.len(),
        a_turn1
    );

    // Step 2: turn-2 (streaming) to engage the streaming-path resume.
    let turn2_messages = format!(
        r#"[{{"role":"user","content":"{}"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"{}"}}]"#,
        json_escape(TURN1_USER),
        json_escape(&a_turn1),
        json_escape(TURN2_USER)
    );
    let a_turn2 = chat_decode_stream(&server_a, &canonical, &turn2_messages, MAX_TOKENS);
    eprintln!(
        "[Phase B.5 stream] turn-2 (streamed) → {} bytes: {:?}",
        a_turn2.len(),
        a_turn2
    );
    if a_turn2.is_empty() {
        let log = server_a.log_tail();
        eprintln!(
            "[Phase B.5 stream] empty streamed content — server A stderr tail:\n{}",
            log.iter()
                .rev()
                .take(40)
                .rev()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    assert!(
        !a_turn2.is_empty(),
        "[Phase B.5 stream] streamed turn-2 produced empty content"
    );

    // Engagement assertion: streaming-path log line MUST appear.
    let log = server_a.log_tail();
    let stream_resume_engaged = log
        .iter()
        .any(|l| l.contains("[hf2q qwen35 stream lcp resume] STRIDE-ALIGNED HIT"));
    assert!(
        stream_resume_engaged,
        "[Phase B.5 stream] FALSIFIED — server A's stderr lacks \
         `[hf2q qwen35 stream lcp resume] STRIDE-ALIGNED HIT` line.  \
         Streaming-path B.3 resume did NOT fire on turn-2.  Stderr tail:\n{}",
        log.iter()
            .rev()
            .take(40)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    eprintln!(
        "[Phase B.5 stream] PASS — streaming-path STRIDE-ALIGNED HIT \
         engaged on turn-2.  Streamed {} bytes via LCP resume.",
        a_turn2.len()
    );
}
