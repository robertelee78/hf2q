//! ADR-017 §B-tq.5 — shared `hf2q serve` subprocess driver.
//!
//! Extracted from the inline `phase_d_driver` mod that previously
//! lived at `tests/kv_persist_gemma4_roundtrip.rs:411-1062` so
//! multiple integration test binaries can drive the live subprocess
//! flow.  Consumers:
//!
//!   * `tests/kv_persist_gemma4_roundtrip.rs` — Phase D coherence +
//!     R-P4 perf gates (gemma4 dense KV path).
//!   * `tests/kv_persist_tq_packed_roundtrip.rs` — B-tq.4 R-C1
//!     byte-identity acceptance test (TQ-active KV path).
//!
//! ## Surface
//!
//! Generic by design — accepts an `extra_env: &[(&str, &str)]`
//! argument so callers can set `HF2Q_USE_DENSE=1` (gemma4 dense
//! path) or `HF2Q_TQ_KV=1` (TQ-active path) or any other env without
//! the driver assuming a specific mode.
//!
//! ## RAII discipline
//!
//! `ServerGuard::Drop` kills + waits the child unconditionally so a
//! panicking test never leaks a server.  `wait_for_graceful_exit`
//! pairs with `trigger_graceful_shutdown` for tests that want to
//! verify the cmd_serve drain ran AND the process exited before
//! launching a successor.

#![allow(dead_code)] // each test binary uses a subset; see tests/common/mod.rs note

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub const HOST: &str = "127.0.0.1";

/// Default port for KV-persist subprocess tests.  Distinct from
/// `kv_persist_harness::PORT_DEFAULT` (52338),
/// `multi_model_swap.rs` (52337), `openwebui` (52334),
/// `mmproj_llama_cpp_compat` (52226), `vision_e2e_vs_mlx_vlm`
/// (18181) so cells can safely run sequentially even if a prior run
/// leaked.
pub const PORT_DEFAULT: u16 = 52339;

/// `/readyz` poll budget — same envelope as
/// `kv_persist_harness::READYZ_BUDGET_SECS` (cold 16-26 GiB GGUF
/// startup on M5 Max can take 60-180 s).
pub const READYZ_BUDGET_SECS: u64 = 600;

#[derive(Debug)]
pub enum DriverError {
    BinaryNotFound(String),
    SpawnFailed(String),
    ReadyzTimeout { waited_secs: u64, last: String },
    Transport(String),
    Http { status: u16, body: String },
    Sse(String),
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DriverError::BinaryNotFound(s) => write!(f, "binary not found: {s}"),
            DriverError::SpawnFailed(s) => write!(f, "spawn: {s}"),
            DriverError::ReadyzTimeout { waited_secs, last } => {
                write!(f, "readyz timeout after {waited_secs}s; last={last}")
            }
            DriverError::Transport(s) => write!(f, "transport: {s}"),
            DriverError::Http { status, body } => write!(f, "http {status}: {body}"),
            DriverError::Sse(s) => write!(f, "sse: {s}"),
        }
    }
}
impl std::error::Error for DriverError {}

pub struct ServerGuard {
    child: Child,
    host: String,
    port: u16,
    stderr_tail: Arc<Mutex<Vec<String>>>,
    stderr_thread: Option<thread::JoinHandle<()>>,
}

impl ServerGuard {
    pub fn host(&self) -> &str {
        &self.host
    }
    pub fn port(&self) -> u16 {
        self.port
    }
    pub fn log_tail(&self) -> Vec<String> {
        self.stderr_tail.lock().map(|g| g.clone()).unwrap_or_default()
    }
    /// ADR-017 Closure iter-2 (2026-05-04): non-blocking child status
    /// check for `wait_for_graceful_exit`.  Returns
    /// `Some(exit_status)` if the child has exited, `None` if still
    /// running, `Err` only on lower-level wait failure.
    pub fn try_wait_child(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(t) = self.stderr_thread.take() {
            let _ = t.join();
        }
    }
}

/// Spawn `hf2q serve --model PATH --host HOST --port PORT
/// --kv-persist CACHE_DIR` with caller-supplied extra env vars.
///
/// `extra_env`: caller-controlled key/value pairs set on the child's
/// environment.  Typical:
///   * `&[("HF2Q_USE_DENSE", "1")]` — Gemma 4 dense KV path
///     (R-C4 byte-exact requires dense; TQ is lossy by design).
///   * `&[("HF2Q_TQ_KV", "1")]` — TurboQuant-active KV path
///     (B-tq.4 acceptance test).
///   * `&[]` — no env override (server uses operator defaults).
pub fn spawn_hf2q_serve_with_kv_persist(
    bin: &Path,
    model_path: &Path,
    cache_dir: &Path,
    host: &str,
    port: u16,
    extra_env: &[(&str, &str)],
) -> Result<ServerGuard, DriverError> {
    if !bin.exists() {
        return Err(DriverError::BinaryNotFound(bin.display().to_string()));
    }
    if !model_path.exists() {
        return Err(DriverError::SpawnFailed(format!(
            "model path does not exist: {}",
            model_path.display()
        )));
    }
    std::fs::create_dir_all(cache_dir).map_err(|e| {
        DriverError::SpawnFailed(format!(
            "mkdir cache_dir {}: {e}",
            cache_dir.display()
        ))
    })?;
    let mut cmd = Command::new(bin);
    cmd.args([
        "serve",
        "--model",
        &model_path.to_string_lossy(),
        "--host",
        host,
        "--port",
        &port.to_string(),
        "--kv-persist",
        &cache_dir.to_string_lossy(),
    ])
    .stdout(Stdio::null())
    .stderr(Stdio::piped());
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| DriverError::SpawnFailed(e.to_string()))?;
    let stderr_tail: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let stderr_thread = if let Some(stderr) = child.stderr.take() {
        let tail = Arc::clone(&stderr_tail);
        Some(thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut buf = String::new();
            loop {
                buf.clear();
                match reader.read_line(&mut buf) {
                    Ok(0) => break,
                    Ok(_) => {
                        let line = buf.trim_end_matches(['\n', '\r']).to_string();
                        if let Ok(mut g) = tail.lock() {
                            g.push(line);
                            let drain = g.len().saturating_sub(256);
                            if drain > 0 {
                                g.drain(..drain);
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
        }))
    } else {
        None
    };
    Ok(ServerGuard {
        child,
        host: host.to_string(),
        port,
        stderr_tail,
        stderr_thread,
    })
}

/// Minimal HTTP/1.1 GET status — mirrors
/// `kv_persist_harness::http_get_status` so we don't pull tokio just
/// for the readyz poll.
fn http_get_status(host: &str, port: u16, path: &str) -> std::io::Result<u16> {
    use std::net::TcpStream;
    let addr = format!("{host}:{port}")
        .parse()
        .map_err(std::io::Error::other)?;
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(5))?;
    s.set_read_timeout(Some(Duration::from_secs(5)))?;
    s.write_all(
        format!(
            "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
        )
        .as_bytes(),
    )?;
    let mut head = [0u8; 64];
    let n = s.read(&mut head)?;
    let head_s = std::str::from_utf8(&head[..n]).unwrap_or("");
    head_s
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| std::io::Error::other(format!("bad status: {head_s:?}")))
}

pub fn wait_for_readyz(server: &ServerGuard) -> Result<(), DriverError> {
    let started = Instant::now();
    let mut last = String::from("<none>");
    while started.elapsed().as_secs() < READYZ_BUDGET_SECS {
        match http_get_status(server.host(), server.port(), "/readyz") {
            Ok(200) => return Ok(()),
            Ok(c) => last = format!("status={c}"),
            Err(e) => last = format!("transport: {e}"),
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err(DriverError::ReadyzTimeout {
        waited_secs: started.elapsed().as_secs(),
        last,
    })
}

/// ADR-017 Closure iter-2 (2026-05-04): trigger graceful shutdown
/// via `POST /shutdown`.  The handler raises SIGTERM on the server
/// process, which causes the cmd_serve drain logic
/// (`drain_loaded_models_to_disk`) to evict every loaded model and
/// flush KV blocks to disk before the process exits.
///
/// Returns the JSON receipt body (status, pid, queue depth) on 202
/// Accepted.  Caller should subsequently call
/// `wait_for_graceful_exit` to block until the process actually
/// terminates (the drain runs after this response).
pub fn trigger_graceful_shutdown(
    server: &ServerGuard,
) -> Result<serde_json::Value, DriverError> {
    let url = format!(
        "http://{}:{}/shutdown",
        server.host(),
        server.port()
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| DriverError::Transport(e.to_string()))?;
    let resp = client
        .post(&url)
        .send()
        .map_err(|e| DriverError::Transport(e.to_string()))?;
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        let body = resp.text().unwrap_or_else(|_| "<unreadable>".into());
        return Err(DriverError::Http { status, body });
    }
    let body: serde_json::Value = resp
        .json()
        .map_err(|e| DriverError::Transport(e.to_string()))?;
    Ok(body)
}

/// ADR-017 Closure iter-2 (2026-05-04): block until the server
/// process exits, with a timeout.  Pairs with
/// `trigger_graceful_shutdown` for tests that want to verify the
/// drain ran AND the process is gone before launching a successor.
/// Polls `Child::try_wait` every 50 ms.
pub fn wait_for_graceful_exit(
    server: &mut ServerGuard,
    timeout: Duration,
) -> Result<std::process::ExitStatus, DriverError> {
    let start = Instant::now();
    loop {
        match server.try_wait_child() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {
                if start.elapsed() >= timeout {
                    return Err(DriverError::ReadyzTimeout {
                        waited_secs: start.elapsed().as_secs(),
                        last: format!(
                            "child still alive after {} s; \
                             graceful drain may have stalled",
                            start.elapsed().as_secs()
                        ),
                    });
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                return Err(DriverError::Transport(format!(
                    "try_wait_child: {e}"
                )));
            }
        }
    }
}

/// Fetch the canonical model id from `/v1/models` so subsequent
/// requests use a model name the server recognizes.
pub fn fetch_canonical_model_id(server: &ServerGuard) -> Result<String, DriverError> {
    let url = format!(
        "http://{}:{}/v1/models",
        server.host(),
        server.port()
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| DriverError::Transport(e.to_string()))?;
    let resp = client
        .get(&url)
        .send()
        .map_err(|e| DriverError::Transport(e.to_string()))?;
    let status = resp.status().as_u16();
    if status != 200 {
        let body = resp.text().unwrap_or_else(|_| "<unreadable>".into());
        return Err(DriverError::Http { status, body });
    }
    let v: serde_json::Value = resp
        .json()
        .map_err(|e| DriverError::Transport(e.to_string()))?;
    v["data"][0]["id"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| DriverError::Http {
            status: 200,
            body: format!("missing data[0].id: {v}"),
        })
}

/// Captured decode result from a single streaming completion.
#[derive(Clone, Debug)]
pub struct DecodeCapture {
    /// Concatenated `choices[0].delta.content` over all SSE chunks —
    /// the byte sequence the operator-visible response would have
    /// rendered.  R-C4 byte-equality test compares THIS field across
    /// never-evicted and evicted+restored runs.
    pub text: String,
    /// Wall clock from POST send to the first non-empty content
    /// delta.  R-P4 ship-gate ratio = cache_hit_ttft / no_cache_ttft.
    pub ttft_ms: f64,
    /// `usage.completion_tokens` if reported, else delta count.
    pub total_tokens: u32,
    /// `usage.prompt_tokens` if reported.
    pub prompt_tokens: Option<u32>,
}

/// Issue `/v1/chat/completions` with `stream: true`,
/// `temperature: 0`, `max_tokens: max_tokens`.  Returns the full
/// captured text + TTFT.
pub fn decode_full_text(
    server: &ServerGuard,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<DecodeCapture, DriverError> {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let url = format!(
        "http://{}:{}/v1/chat/completions",
        server.host(),
        server.port()
    );
    let client = reqwest::blocking::Client::builder()
        // Big-prefill stream legs can run minutes on M5 Max cold
        // path; budget = 10 min.
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|e| DriverError::Transport(e.to_string()))?;
    let t0 = Instant::now();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .map_err(|e| DriverError::Transport(e.to_string()))?;
    let status = resp.status().as_u16();
    if status != 200 {
        let body = resp.text().unwrap_or_else(|_| "<unreadable>".into());
        return Err(DriverError::Http { status, body });
    }
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if !ct.contains("text/event-stream") {
        let body = resp.text().unwrap_or_else(|_| "<unreadable>".into());
        return Err(DriverError::Sse(format!(
            "expected text/event-stream content-type; got {ct:?}; body={body}"
        )));
    }

    let mut ttft_ms: Option<f64> = None;
    let mut text = String::new();
    let mut total_tokens: u32 = 0;
    let mut prompt_tokens: Option<u32> = None;
    let mut reader = BufReader::new(resp);
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| DriverError::Transport(e.to_string()))?;
        if n == 0 {
            break;
        }
        let l = line.trim_end_matches(['\n', '\r']);
        let payload = match l.strip_prefix("data: ") {
            Some(p) => p,
            None => continue,
        };
        if payload == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(payload)
            .map_err(|e| DriverError::Sse(format!("malformed chunk {payload:?}: {e}")))?;
        if let Some(s) = v["choices"][0]["delta"]["content"].as_str() {
            if !s.is_empty() && ttft_ms.is_none() {
                ttft_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
            }
            text.push_str(s);
            total_tokens = total_tokens.saturating_add(1);
        }
        if let Some(c) = v["usage"]["completion_tokens"].as_u64() {
            total_tokens = c as u32;
        }
        if let Some(p) = v["usage"]["prompt_tokens"].as_u64() {
            prompt_tokens = Some(p as u32);
        }
    }
    let ttft_ms = ttft_ms.ok_or_else(|| {
        DriverError::Sse(
            "no non-empty content delta observed; cannot measure TTFT".into(),
        )
    })?;
    Ok(DecodeCapture {
        text,
        ttft_ms,
        total_tokens,
        prompt_tokens,
    })
}

/// Force a pool eviction by admitting a second model under a
/// distinct on-disk path that resolves through `pool_key_for_path`.
/// Mirrors `kv_persist_harness::subprocess_driver::measure_swap_eviction_cycle`
/// + `multi_model_swap.rs:344-384`.
///
/// Returns `(link_path, eviction_wall_ms, second_request_ttft_ms)`.
#[cfg(unix)]
pub fn force_eviction_via_symlink(
    server: &ServerGuard,
    model_path: &Path,
    tmp_link_dir: &Path,
) -> Result<(PathBuf, f64, f64), DriverError> {
    let link_path = tmp_link_dir.join("kv-persist-clone.gguf");
    std::os::unix::fs::symlink(model_path, &link_path)
        .map_err(|e| DriverError::Transport(format!("symlink: {e}")))?;
    // Mirror sibling files (config.json / tokenizer.json /
    // tokenizer_config.json / generation_config.json + mmproj GGUF)
    // — `cmd_serve` resolves siblings of the GGUF path; without
    // them the second-turn admit fails with HTTP 500.
    if let Some(model_parent) = model_path.parent() {
        for fname in &[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ] {
            let src = model_parent.join(fname);
            if src.exists() {
                let dst = tmp_link_dir.join(fname);
                if let Err(e) = std::os::unix::fs::symlink(&src, &dst) {
                    return Err(DriverError::Transport(format!(
                        "symlink sibling {fname}: {e}"
                    )));
                }
            }
        }
        if let Some(src_stem) = model_path.file_stem().and_then(|s| s.to_str()) {
            let mmproj_src = model_parent.join(format!("{src_stem}-mmproj.gguf"));
            if mmproj_src.exists() {
                if let Some(link_stem) = link_path.file_stem().and_then(|s| s.to_str()) {
                    let mmproj_dst =
                        tmp_link_dir.join(format!("{link_stem}-mmproj.gguf"));
                    if let Err(e) =
                        std::os::unix::fs::symlink(&mmproj_src, &mmproj_dst)
                    {
                        return Err(DriverError::Transport(format!(
                            "symlink mmproj: {e}"
                        )));
                    }
                }
            }
        }
    }

    // Drive a tiny request against the cloned-stem path.  This is
    // the pool-eviction trigger: distinct file_stem -> distinct
    // pool key -> cold-load path even though both paths resolve
    // to the same physical bytes.
    let body = serde_json::json!({
        "model": link_path.to_string_lossy(),
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 4,
        "temperature": 0,
        "stream": true,
    });
    let url = format!(
        "http://{}:{}/v1/chat/completions",
        server.host(),
        server.port()
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|e| DriverError::Transport(e.to_string()))?;
    let t0 = Instant::now();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .map_err(|e| DriverError::Transport(e.to_string()))?;
    let status = resp.status().as_u16();
    if status != 200 {
        let body = resp.text().unwrap_or_else(|_| "<unreadable>".into());
        return Err(DriverError::Http { status, body });
    }
    let mut ttft_ms: Option<f64> = None;
    let mut reader = BufReader::new(resp);
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| DriverError::Transport(e.to_string()))?;
        if n == 0 {
            break;
        }
        let l = line.trim_end_matches(['\n', '\r']);
        let payload = match l.strip_prefix("data: ") {
            Some(p) => p,
            None => continue,
        };
        if payload == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(payload)
            .map_err(|e| DriverError::Sse(format!("eviction chunk: {e}")))?;
        if ttft_ms.is_none() {
            if let Some(s) = v["choices"][0]["delta"]["content"].as_str() {
                if !s.is_empty() {
                    ttft_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                }
            }
        }
    }
    let eviction_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let second_ttft_ms = ttft_ms.ok_or_else(|| {
        DriverError::Sse("no content delta on swap-back-in turn".into())
    })?;
    Ok((link_path, eviction_wall_ms, second_ttft_ms))
}

#[cfg(not(unix))]
pub fn force_eviction_via_symlink(
    _server: &ServerGuard,
    _model_path: &Path,
    _tmp_link_dir: &Path,
) -> Result<(PathBuf, f64, f64), DriverError> {
    Err(DriverError::Transport(
        "symlink-distinct-pool-key trick is unix-only".into(),
    ))
}
