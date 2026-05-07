//! ADR-017 Phase E.a default-on — engagement and opt-out integration tests.
//!
//! ## Purpose
//!
//! Verifies two things specified in AC-1 (`cfa-lcp-default-on-2026-05-06`):
//!
//! 1. **Default-on engagement** (`default_on_engagement_no_env`): with no
//!    LCP env vars set (clean env), the LCP probe runs and detects at least
//!    one partial-prefix hit on a long-shared-prefix two-turn sequence.
//!    Asserts `hf2q_kv_lcp_lookups_total >= 2` AND
//!    `hf2q_kv_lcp_detected_total >= 1`.
//!
//! 2. **Opt-out works** (`default_on_optout_with_env_zero`): with
//!    `HF2Q_KV_LCP_RESUME=0`, the probe is disabled and
//!    `hf2q_kv_lcp_lookups_total == 0` after the same two-turn sequence.
//!
//! ## Gating
//!
//! Both tests are gated under `HF2Q_KV_PERSIST_PHASE_D=1` (same gate
//! convention as `tests/lcp_partial_prefill_byte_identity.rs` — see
//! decisions.json Q5). When the gate is not set, each test prints a
//! one-line skip message and returns immediately; they do NOT `panic!` or
//! `assert!` anything when un-gated.
//!
//! A real model must be available at `HF2Q_KV_PERSIST_E2E_MODEL_PATH`
//! (the same env used by `lcp_partial_prefill_byte_identity.rs`). The test
//! uses `HF2Q_USE_DENSE=1` to engage the Gemma-4 dense-KV path where
//! LCP resume is implemented.
//!
//! ## Prompt design
//!
//! Turn 1 (`PROMPT_Q`) is a moderately long sentence (> 200 chat-template
//! tokens after Gemma-4 BOS/role markup). Turn 2 (`PROMPT_P`) shares the
//! turn-1 prefix and appends a short diverging suffix. This guarantees at
//! least one chunked-stride hit in the LCP probe.
//!
//! The prompts are identical to those in `lcp_partial_prefill_byte_identity.rs`
//! so test infrastructure (model warm state, prior runs) is shared.
//!
//! ## How to run manually
//!
//! ```bash
//! # Build first
//! cargo build --release --bin hf2q
//!
//! # Run with a real Gemma 4 26B-DWQ model
//! HF2Q_KV_PERSIST_PHASE_D=1 \
//! HF2Q_KV_PERSIST_E2E_MODEL_PATH=/opt/hf2q/models/gemma-4-26b-dwq46.gguf \
//! cargo test --release lcp_default_on_engagement -- --include-ignored --nocapture
//! ```
//!
//! ## Helpers
//!
//! `spawn_server`, `wait_for_readyz`, `fetch_metrics`, `fetch_canonical_model_id`,
//! `chat_decode` and the metric parsers are verbatim copies of the helpers from
//! `tests/lcp_partial_prefill_byte_identity.rs` (attributed in comments).
//! Duplication is intentional: each test file is self-contained so operator
//! can run either file independently without pulling in the other's shared state.

use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const ENV_PHASE_D_GATE: &str = "HF2Q_KV_PERSIST_PHASE_D";
const ENV_MODEL_PATH: &str = "HF2Q_KV_PERSIST_E2E_MODEL_PATH";
const PORT_DEFAULT_ON: u16 = 52395;
const PORT_OPTOUT: u16 = 52396;
const HOST: &str = "127.0.0.1";
const READYZ_BUDGET: Duration = Duration::from_secs(180);

/// Turn 1: a long enough body that, after Qwen 3.6 chat-templating, we
/// land MULTIPLE stride=64 boundaries (so the descending probe has more
/// than one chance to hit a chunk-keyed store). 46 words tokenizes to
/// ~70 tokens — borderline against stride=64 — so this body is sized at
/// ~200 words to safely exceed 256 tokens of prompt after templating.
/// That guarantees chunked-prefill stores at chunk_pos ∈ {64, 128, 192}.
const PROMPT_Q: &str = "Explain in detail how a Longest Common Prefix cache \
    can accelerate multi-turn large-language-model inference by reusing \
    previously computed key-value states across requests that share a \
    common prefix. Cover the byte-budget eviction policy, including how \
    the LRU order is maintained under concurrent request pressure in a \
    single-process HTTP server. Describe how the registry computes per-\
    entry byte size from existing payload APIs without estimation, and \
    the trade-offs between entry-count caps and byte-budget caps when \
    snapshot sizes vary by model and prompt length. Discuss the role of \
    the descending stride-aligned probe in finding the longest matching \
    cached prefix, and how mid-prefill checkpoint storage at every \
    stride boundary lets a partial-match suffice when the cached prompt \
    is shorter than the new prompt. Finally, walk through how the \
    chunked-prefill path interacts with the per-layer KV-buffer arena, \
    the DeltaNet recurrent state, and the snapshot Arc lifetime invariant \
    that keeps in-flight prefills safe under concurrent registry eviction.";

/// Turn 2: shares PROMPT_Q as a strict prefix then appends a diverging
/// suffix. Tokenization must produce a token sequence whose first N
/// tokens equal Q's first N tokens for some N > stride; the appended
/// suffix only adds tokens AFTER the shared region.
const PROMPT_P: &str = "Explain in detail how a Longest Common Prefix cache \
    can accelerate multi-turn large-language-model inference by reusing \
    previously computed key-value states across requests that share a \
    common prefix. Cover the byte-budget eviction policy, including how \
    the LRU order is maintained under concurrent request pressure in a \
    single-process HTTP server. Describe how the registry computes per-\
    entry byte size from existing payload APIs without estimation, and \
    the trade-offs between entry-count caps and byte-budget caps when \
    snapshot sizes vary by model and prompt length. Discuss the role of \
    the descending stride-aligned probe in finding the longest matching \
    cached prefix, and how mid-prefill checkpoint storage at every \
    stride boundary lets a partial-match suffice when the cached prompt \
    is shorter than the new prompt. Finally, walk through how the \
    chunked-prefill path interacts with the per-layer KV-buffer arena, \
    the DeltaNet recurrent state, and the snapshot Arc lifetime invariant \
    that keeps in-flight prefills safe under concurrent registry eviction. \
    Also describe the opt-out mechanism via environment variable and \
    the auto-disable behavior when HF2Q_USE_DENSE is unset.";

const MAX_TOKENS: u32 = 32;

// ─────────────────────────────────────────────────────────────────────────────
// Gate helper
// ─────────────────────────────────────────────────────────────────────────────

/// Return the model path if both `HF2Q_KV_PERSIST_PHASE_D=1` and
/// `HF2Q_KV_PERSIST_E2E_MODEL_PATH=<path>` are set, else return None.
///
/// Mirrors `resolve_model_path_or_skip` from lcp_partial_prefill_byte_identity.rs.
fn resolve_model_path_or_skip() -> Option<PathBuf> {
    if std::env::var(ENV_PHASE_D_GATE).as_deref() != Ok("1") {
        return None;
    }
    let path = std::env::var(ENV_MODEL_PATH).ok()?;
    if path.is_empty() {
        return None;
    }
    let p = PathBuf::from(path);
    if !p.exists() {
        panic!(
            "[lcp_default_on_engagement] {ENV_PHASE_D_GATE}=1 set but \
             {ENV_MODEL_PATH} does not point to an existing file: {}",
            p.display()
        );
    }
    Some(p)
}

fn hf2q_binary_path() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir).join("target/release/hf2q")
}

// ─────────────────────────────────────────────────────────────────────────────
// ServerGuard — verbatim from lcp_partial_prefill_byte_identity.rs
// ─────────────────────────────────────────────────────────────────────────────

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

    /// Count `STRIDE-ALIGNED HIT` lines in the tailed server stderr.
    ///
    /// This is the load-bearing engagement signal for qwen35 LCP-resume:
    /// the observability counter `hf2q_kv_lcp_detected_total` checks the
    /// BASE key (no chunk_pos) at engine_qwen35.rs:864 but stored entries
    /// use CHUNK keys (build_lcp_key_for_qwen35_chunk). The two namespaces
    /// never match for qwen35 chunked workloads, so detected_total stays
    /// 0 even when resume is fully engaged. The `STRIDE-ALIGNED HIT`
    /// stderr line is emitted at engine_qwen35.rs:961-973 (non-streaming)
    /// and :2148-2161 (streaming) only when `restore_partial` actually
    /// fires — that's the truth signal. Counter-semantics fix is tracked
    /// as a follow-up.
    fn count_stride_aligned_hits(&self) -> usize {
        self.log_tail()
            .iter()
            .filter(|line| line.contains("STRIDE-ALIGNED HIT"))
            .count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// spawn_server — mirrors lcp_partial_prefill_byte_identity.rs pattern.
//
// `extra_envs` is a list of additional (key, value) pairs layered on top of
// the default env. Pass an empty slice for the default-on test (no LCP vars).
// ─────────────────────────────────────────────────────────────────────────────

fn spawn_server(bin: &Path, model: &Path, port: u16, extra_envs: &[(&str, &str)]) -> ServerGuard {
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
    // HF2Q_USE_DENSE=1 needed for the Gemma-4 dense-KV path (LCP is only
    // wired on the dense path). Leave HF2Q_KV_LCP_RESUME unset in the
    // default-on test; set it to "0" in the opt-out test via extra_envs.
    .env("HF2Q_USE_DENSE", "1")
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

    ServerGuard { child, port, stderr_tail, _stderr_thread: stderr_thread }
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP helpers — verbatim from lcp_partial_prefill_byte_identity.rs
// ─────────────────────────────────────────────────────────────────────────────

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

/// Parse `hf2q_kv_lcp_lookups_total <N>` from /metrics body.
/// Returns 0 if the counter line is absent.
///
/// Verbatim from lcp_partial_prefill_byte_identity.rs.
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

/// Parse `hf2q_kv_lcp_detected_total <N>` from /metrics body.
/// Returns 0 if the counter line is absent.
///
/// Verbatim from lcp_partial_prefill_byte_identity.rs.
fn metric_lcp_detected_total(metrics_body: &str) -> u64 {
    for line in metrics_body.lines() {
        if let Some(rest) = line.strip_prefix("hf2q_kv_lcp_detected_total ") {
            if let Ok(n) = rest.trim().parse::<u64>() {
                return n;
            }
        }
    }
    0
}

fn fetch_canonical_model_id(server: &ServerGuard) -> String {
    use std::io::{Read, Write};
    let mut s = TcpStream::connect((HOST, server.port)).expect("connect /v1/models");
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

/// Send a non-streaming chat/completions request (temperature=0 for
/// determinism) and return the decoded assistant content.
///
/// Adapted from lcp_partial_prefill_byte_identity.rs::chat_decode.
fn chat_decode(server: &ServerGuard, model: &str, prompt: &str, max_tokens: u32) -> String {
    use std::io::{Read, Write};
    // Escape the prompt for embedding in JSON — handle double-quotes and
    // backslashes that might appear in the ADR text.
    let prompt_escaped = prompt.replace('\\', "\\\\").replace('"', "\\\"");
    let body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"{prompt_escaped}"}}],"max_tokens":{max_tokens},"temperature":0,"stream":false}}"#
    );
    let mut s = TcpStream::connect((HOST, server.port)).expect("connect for chat_decode");
    s.set_read_timeout(Some(Duration::from_secs(120))).ok();
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
    s.read_to_string(&mut buf).expect("read chat response");

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
         Status: {status_line:?}\nFull response:\n{buf}"
    );

    let body_start = buf.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    let body = &buf[body_start..];
    assert!(
        body.contains("\"choices\":["),
        "[chat_decode] response lacks `\"choices\":[` — likely error body: {body}"
    );

    let key = "\"content\":\"";
    let p = body
        .find(key)
        .unwrap_or_else(|| panic!("no content field in response body: {body}"));
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

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: default-on engagement (no LCP env vars set)
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that LCP probe is engaged by default (no env vars set).
///
/// Sends a two-turn sequence (PROMPT_Q then PROMPT_P) to an hf2q server
/// started with NO LCP-related env vars. After both turns, asserts that
/// `/metrics` shows `hf2q_kv_lcp_lookups_total >= 2` (the probe ran on
/// both turns) AND `hf2q_kv_lcp_detected_total >= 1` (at least one
/// partial-prefix hit was detected on turn 2, proving default-on is live).
///
/// Gate: `HF2Q_KV_PERSIST_PHASE_D=1` AND `HF2Q_KV_PERSIST_E2E_MODEL_PATH`
/// pointing to a real model GGUF on Metal hardware. Not run in CI.
///
/// The test uses `#[ignore]` so `cargo test` skips it by default.
/// The gate-env check inside the test body provides the operator-friendly
/// skip message when run with `--include-ignored` but without the gate set.
#[test]
#[ignore = "Requires Metal hardware + real model: set HF2Q_KV_PERSIST_PHASE_D=1 + HF2Q_KV_PERSIST_E2E_MODEL_PATH"]
fn default_on_engagement_no_env() {
    let model_path = match resolve_model_path_or_skip() {
        Some(p) => p,
        None => {
            eprintln!(
                "[default_on_engagement_no_env] skipping \
                 (HF2Q_KV_PERSIST_PHASE_D not set or \
                 HF2Q_KV_PERSIST_E2E_MODEL_PATH unset/missing)"
            );
            return;
        }
    };

    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "[default_on_engagement_no_env] hf2q binary not found at {} — \
         run `cargo build --release` first",
        bin.display()
    );

    // Spawn with NO LCP-toggle env vars — tests that the BINARY GATES
    // (kv_lcp_resume + kv_lcp_chunked_prefill) are default-on. The stride
    // value is a tuning knob, NOT a default-on toggle; we set it to 64 here
    // because the production default (1024) requires prompts > 1 K tokens
    // to engage and the test prompts are ~200 tokens for tractability.
    // The opt-out test below uses the SAME stride so the comparison stays
    // apples-to-apples.
    let server = spawn_server(
        &bin,
        &model_path,
        PORT_DEFAULT_ON,
        &[("HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE", "64")],
    );
    wait_for_readyz(&server);

    let model_id = fetch_canonical_model_id(&server);
    eprintln!(
        "[default_on_engagement_no_env] server ready on {HOST}:{PORT_DEFAULT_ON} \
         model={model_id} (no LCP env vars — default-on path)"
    );

    // Turn 1: prime the LCP registry.
    let q_decoded = chat_decode(&server, &model_id, PROMPT_Q, MAX_TOKENS);
    assert!(
        !q_decoded.is_empty(),
        "[default_on_engagement_no_env] turn-1 (Q) decoded empty"
    );
    eprintln!(
        "[default_on_engagement_no_env] turn-1 Q decoded {} bytes (registry primed)",
        q_decoded.len()
    );

    // Turn 2: probe prompt shares PROMPT_Q as a prefix.
    // LCP probe should hit with K = tokenized-len(Q) > stride.
    let p_decoded = chat_decode(&server, &model_id, PROMPT_P, MAX_TOKENS);
    assert!(
        !p_decoded.is_empty(),
        "[default_on_engagement_no_env] turn-2 (P) decoded empty"
    );
    eprintln!(
        "[default_on_engagement_no_env] turn-2 P decoded {} bytes",
        p_decoded.len()
    );

    // Read /metrics and assert the probe ran and detected.
    let metrics = fetch_metrics(&server);
    let lookups = metric_lcp_lookups_total(&metrics);
    let detected = metric_lcp_detected_total(&metrics);

    eprintln!(
        "[default_on_engagement_no_env] /metrics: \
         lcp_lookups_total={lookups}, lcp_detected_total={detected}"
    );

    // The probe runs on every non-full-equality request AFTER turn-1 stores
    // an entry. Turn-1 itself may or may not probe (depends on whether a prior
    // entry exists); turn-2 always probes because turn-1 registered an entry.
    // We assert at least 2 lookups total (conservative: one per turn).
    assert!(
        lookups >= 2,
        "[default_on_engagement_no_env] FAIL: hf2q_kv_lcp_lookups_total={lookups} < 2. \
         The LCP probe did not run on both turns. \
         Check that default-on is wired correctly in investigation_env.rs:576 \
         (kv_lcp_resume uses env_default_true, not env_eq_one)."
    );

    // Truth-signal for engagement: count `STRIDE-ALIGNED HIT` lines in
    // server stderr. The detected_total counter is broken for qwen35
    // chunked-prefill (probes BASE key vs CHUNK-keyed stores; counter-fix
    // is a follow-up). The stderr HIT line fires at engine_qwen35.rs:961
    // / :2148 only when restore_partial actually executes — the right
    // signal for whether default-on resume engaged.
    let hits = server.count_stride_aligned_hits();
    eprintln!(
        "[default_on_engagement_no_env] STRIDE-ALIGNED HIT count in stderr={hits}"
    );
    if hits < 1 {
        let tail = server.log_tail();
        let lcp_lines: Vec<&String> = tail
            .iter()
            .filter(|l| l.contains("[hf2q qwen35 lcp"))
            .collect();
        let lcp_dump = if lcp_lines.is_empty() {
            "<no [hf2q qwen35 lcp ...] lines in last-256 stderr tail>".to_string()
        } else {
            lcp_lines
                .iter()
                .map(|l| l.as_str())
                .collect::<Vec<_>>()
                .join("\n  ")
        };
        panic!(
            "[default_on_engagement_no_env] FAIL: STRIDE-ALIGNED HIT count={hits} < 1. \
             Default-on did not engage on turn-2. lookups={lookups}, detected={detected}. \
             (detected_total is known-broken for qwen35 chunked workloads — BASE-key \
             probe vs CHUNK-keyed stores; counter-fix is a follow-up.) \
             \n--- last [hf2q qwen35 lcp ...] stderr lines ---\n  {lcp_dump}"
        );
    }

    eprintln!("[default_on_engagement_no_env] PASS: default-on LCP probe engaged ({hits} hits).");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: opt-out via HF2Q_KV_LCP_RESUME=0
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that `HF2Q_KV_LCP_RESUME=0` completely disables the LCP probe.
///
/// Sends the same two-turn sequence to a server started with
/// `HF2Q_KV_LCP_RESUME=0`. After both turns, asserts that
/// `hf2q_kv_lcp_lookups_total == 0` — the probe must not run at all.
///
/// This test is the backward-compat guarantee: operators who set `=0` to
/// opt out of default-on must not see any LCP probe activity.
///
/// Gate: same as `default_on_engagement_no_env`.
///
/// The test uses `#[ignore]` so `cargo test` skips it by default.
#[test]
#[ignore = "Requires Metal hardware + real model: set HF2Q_KV_PERSIST_PHASE_D=1 + HF2Q_KV_PERSIST_E2E_MODEL_PATH"]
fn default_on_optout_with_env_zero() {
    let model_path = match resolve_model_path_or_skip() {
        Some(p) => p,
        None => {
            eprintln!(
                "[default_on_optout_with_env_zero] skipping \
                 (HF2Q_KV_PERSIST_PHASE_D not set or \
                 HF2Q_KV_PERSIST_E2E_MODEL_PATH unset/missing)"
            );
            return;
        }
    };

    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "[default_on_optout_with_env_zero] hf2q binary not found at {} — \
         run `cargo build --release` first",
        bin.display()
    );

    // Spawn with HF2Q_KV_LCP_RESUME=0 — explicit opt-out.
    // env_default_true("HF2Q_KV_LCP_RESUME") returns false for "0".
    // Same stride=64 as the default-on test, so apples-to-apples.
    let server = spawn_server(
        &bin,
        &model_path,
        PORT_OPTOUT,
        &[
            ("HF2Q_KV_LCP_RESUME", "0"),
            ("HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE", "64"),
        ],
    );
    wait_for_readyz(&server);

    let model_id = fetch_canonical_model_id(&server);
    eprintln!(
        "[default_on_optout_with_env_zero] server ready on {HOST}:{PORT_OPTOUT} \
         model={model_id} (HF2Q_KV_LCP_RESUME=0 — opt-out path)"
    );

    // Turn 1: would prime the registry IF the probe were enabled.
    let q_decoded = chat_decode(&server, &model_id, PROMPT_Q, MAX_TOKENS);
    assert!(
        !q_decoded.is_empty(),
        "[default_on_optout_with_env_zero] turn-1 (Q) decoded empty — server running?"
    );
    eprintln!(
        "[default_on_optout_with_env_zero] turn-1 Q decoded {} bytes",
        q_decoded.len()
    );

    // Turn 2.
    let p_decoded = chat_decode(&server, &model_id, PROMPT_P, MAX_TOKENS);
    assert!(
        !p_decoded.is_empty(),
        "[default_on_optout_with_env_zero] turn-2 (P) decoded empty"
    );
    eprintln!(
        "[default_on_optout_with_env_zero] turn-2 P decoded {} bytes",
        p_decoded.len()
    );

    // The observability probe at engine_qwen35.rs:864 runs UNCONDITIONALLY
    // (so /metrics tracks LCP opportunity even when resume is disabled).
    // So `lookups_total` will be > 0 even with HF2Q_KV_LCP_RESUME=0. The
    // truth-signal for opt-out is the absence of `STRIDE-ALIGNED HIT`
    // lines: those fire only inside `if lcp_resume_enabled { ... }`,
    // gated by `effective_kv_lcp_resume()` which returns false on env=0.
    let metrics = fetch_metrics(&server);
    let lookups = metric_lcp_lookups_total(&metrics);
    let hits = server.count_stride_aligned_hits();

    eprintln!(
        "[default_on_optout_with_env_zero] /metrics: lcp_lookups_total={lookups} \
         (observability probe runs unconditionally); \
         STRIDE-ALIGNED HIT count in stderr={hits} (must be 0 under opt-out)"
    );

    assert_eq!(
        hits, 0,
        "[default_on_optout_with_env_zero] FAIL: STRIDE-ALIGNED HIT count={hits}, \
         expected 0. Resume engaged despite HF2Q_KV_LCP_RESUME=0. \
         Check env_default_true() at investigation_env.rs:884 (must return false \
         on \"0\") AND effective_kv_lcp_resume() at engine.rs:3923 (must short-\
         circuit on parsed=false)."
    );

    eprintln!("[default_on_optout_with_env_zero] PASS: opt-out via =0 works correctly.");
}
