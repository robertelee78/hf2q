//! ADR-017 Phase B-hybrid.2a — chunked-prefill byte-identity falsifier.
//!
//! Asserts that running the Qwen 3.6 prefill in fixed-size chunks
//! (with state propagation through the `kv_cache`) produces output
//! byte-identical to the monolithic single-call prefill.
//!
//! ## Why this is the foundational falsifier for Phase B.2
//!
//! Phase B-hybrid.2 (LCP partial-prefill resume for Qwen 3.5/3.6)
//! requires capturing intermediate SSM-state checkpoints during
//! prefill. The cleanest way to capture checkpoints is to run prefill
//! in chunks of `stride` tokens — after each chunk's
//! `forward_gpu_last_logits` call, the kv_cache holds the state at
//! position k_end and we can snapshot it.
//!
//! For this approach to be sound, the cumulative effect of N chunked
//! calls MUST be byte-identical to one monolithic call covering the
//! same prompt. If it is NOT, then either the DeltaNet recurrent
//! state isn't propagating correctly between calls, or the full-attn
//! `current_len[0]` cursor is mis-tracked, or some other state-leak
//! bug exists. ANY divergence here means Phase B.2 is not feasible
//! without fixing the underlying state-propagation bug first.
//!
//! ## Test design
//!
//! Two-server byte-identity:
//!   * Server A: `HF2Q_KV_LCP_CHUNKED_PREFILL=1
//!     HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=64`. Server runs
//!     prefill in chunks of 64 tokens. (Stride=64 is the smallest
//!     legal value — every chunk hits the chunk_gated_delta_rule
//!     fast-path.)
//!   * Server B: control (no env flags). Server runs prefill
//!     monolithically.
//!   * Both: send the same prompt P (chosen so its tokenized length
//!     is a multiple of 64). Compare decoded bytes.
//!
//! Falsifier: `server_A_decoded(P) != server_B_decoded(P)` ⇒ chunked
//! prefill diverges from monolithic. Phase B.2 is NOT feasible at
//! the proposed stride; abort the architecture or fix the state
//! propagation bug.
//!
//! ## Prompt length constraint
//!
//! The chunked path in `engine_qwen35.rs::generate_qwen35_once` only
//! activates when `prompt_len % stride == 0`. For stride=64, the
//! tokenized prompt length must be a multiple of 64. We pick a prompt
//! that's "long enough" (a long repetitive string) and trust the
//! tokenizer to land on a 64-multiple boundary; if it doesn't, the
//! chunked path falls back to monolithic (logged at debug level) and
//! the test still passes trivially. To make the test load-bearing,
//! we run with stride=64 and a long prompt (~100+ tokens), then
//! verify `lcp_lookups_total` differences between A and B as a sanity
//! check that the code path was exercised.
//!
//! ## Operator gating
//!
//! Operator-gated under `HF2Q_KV_PERSIST_PHASE_D=1` AND
//! `HF2Q_KV_PERSIST_QWEN35_E2E_MODEL_PATH=<gguf>` (same gates as
//! `lcp_qwen35_observability.rs`).

use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const ENV_PHASE_D_GATE: &str = "HF2Q_KV_PERSIST_PHASE_D";
const ENV_QWEN35_MODEL_PATH: &str = "HF2Q_KV_PERSIST_QWEN35_E2E_MODEL_PATH";
const PORT_A: u16 = 52411;
const PORT_B: u16 = 52412;
const HOST: &str = "127.0.0.1";
const READYZ_BUDGET: Duration = Duration::from_secs(180);

/// A "long enough" prompt for the chunked path to fire on. The
/// tokenizer determines the exact length; we just need > stride
/// tokens AND ideally a multiple-of-stride length. The test prints
/// the actual tokenized length (via /metrics if available) and
/// reports skip if the multiple-of-stride constraint isn't met.
// ADR-017 Phase E.a B.5: post-gate-lift, the previously-failing
// 130-token / 3-chunk / partial-tail=2 fixture is now byte-identical
// to monolithic.  Use the longer prompt as the canonical fixture so
// B.2a guards both the 2-chunk safe-zone AND the 3-chunk-with-tail<16
// (formerly "danger zone") byte-identity invariants.
const PROMPT: &str =
    "Write a detailed essay describing the four seasons in temperate climates. \
     Cover spring with its blossoming flowers and warming temperatures, summer \
     with long sunny days and outdoor activities, autumn with falling leaves \
     and harvest festivals, and winter with snow and shorter daylight hours. \
     Include cultural traditions associated with each season across various \
     regions of the world, and explain how plants and animals adapt their \
     behavior throughout the yearly cycle. Discuss agricultural patterns, \
     migration routes for major bird species, hibernation strategies of \
     mammals, and the impact of seasonal changes on freshwater ecosystems. \
     Include examples from Europe, North America, and East Asia.";
const MAX_TOKENS: u32 = 16;
const TEST_STRIDE: u32 = 64;

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
            "[Phase B.2a] {ENV_PHASE_D_GATE}=1 set but \
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
    let p = body.find(key).expect("find model id");
    let rest = &body[p + key.len()..];
    let end = rest.find('"').expect("close quote");
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
        "[chat_decode] expected HTTP 200, got {status_code}"
    );

    let body_start = buf.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    let body = &buf[body_start..];
    assert!(
        body.contains("\"choices\":["),
        "[chat_decode] response body lacks `\"choices\":[`"
    );

    let key = "\"content\":\"";
    let p = body
        .find(key)
        .unwrap_or_else(|| panic!("no content field"));
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

/// Phase B-hybrid.2a chunked-vs-monolithic byte-identity falsifier.
///
/// **STATUS 2026-05-05**: this test FAILS today. The chunked path
/// produces non-byte-identical output to monolithic on Qwen 3.6
/// 27B-DWQ46 (stride=64, prompt=108 tokens → 41 bytes vs 47 bytes,
/// diverges at byte 4). Root cause hypothesis: the DeltaNet kernel
/// dispatcher uses different code paths based on
/// `seq_len > 64 && seq_len % 64 == 0` (chunk-pipeline) vs
/// fallback (autoregressive); the partial-tail chunk (e.g. seq_len=44
/// at stride=64 with prompt_len=108) takes the autoregressive path,
/// and the cumulative state at end-of-prefill differs from a single
/// monolithic call (which takes only one of the two paths).
///
/// Phase B.2 (LCP partial-prefill resume for Qwen 3.5/3.6) IS
/// BLOCKED on this finding. Two possible mitigations for B.2b/B.2c:
///   1. Constrain chunked prefill to stride values where prompt_len
///      is divisible (no partial tail). Restricts when the speedup
///      applies but preserves byte-identity.
///   2. Investigate + fix the kernel-path-divergence bug (likely in
///      mlx-native chunk_gated_delta_rule vs autoregressive
///      gated_delta_net dispatch).
///
/// This test is intentionally KEPT failing as a load-bearing
/// reminder per mantra "code + test == truth". When mitigation #1
/// is shipped, this test will be updated to use a divisible stride;
/// when #2 is shipped, the partial-tail case will pass too.
#[test]
fn phase_b2a_chunked_vs_monolithic_byte_identity() {
    let model_path = match resolve_qwen35_model_path_or_skip() {
        Some(p) => p,
        None => {
            eprintln!(
                "[Phase B.2a] {ENV_PHASE_D_GATE}=1 + \
                 {ENV_QWEN35_MODEL_PATH}=PATH not set — short-circuit."
            );
            return;
        }
    };
    let bin = hf2q_binary_path();
    assert!(bin.exists(), "[Phase B.2a] hf2q binary not found");

    let stride_str = TEST_STRIDE.to_string();

    // ── Server A: chunked prefill ──
    let server_a = spawn_server(
        &bin,
        &model_path,
        PORT_A,
        &[
            ("HF2Q_KV_LCP_CHUNKED_PREFILL", "1"),
            ("HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE", &stride_str),
        ],
    );
    wait_for_readyz(&server_a);
    let canonical_a = fetch_canonical_model_id(&server_a);
    eprintln!(
        "[Phase B.2a] server A (CHUNKED_PREFILL=1, stride={}) ready on {}:{}",
        TEST_STRIDE, HOST, PORT_A
    );

    let a_decoded = chat_decode(&server_a, &canonical_a, PROMPT, MAX_TOKENS);
    assert!(
        !a_decoded.is_empty(),
        "[Phase B.2a] server A decoded empty"
    );
    eprintln!("[Phase B.2a] server A decoded {} bytes", a_decoded.len());

    // Engagement assertion: server A's stderr must contain a "chunked
    // prefill" line — otherwise the chunked path fell back to
    // monolithic (e.g., prompt_len % stride != 0) and this test
    // becomes a vacuous trivial pass.
    let a_log = server_a.log_tail();
    let chunked_engaged = a_log.iter().any(|l| l.contains("chunked prefill"));
    assert!(
        chunked_engaged,
        "[Phase B.2a] FALSIFIED — server A's stderr lacks `chunked prefill` \
         line, meaning the chunked path FELL BACK to monolithic (likely \
         prompt_len % {} != 0 OR prompt_len <= stride). Test is vacuous; \
         tighten the prompt or stride choice. Server A stderr tail (last \
         lines):\n{}",
        TEST_STRIDE,
        a_log
            .iter()
            .filter(|l| l.contains("prefill") || l.contains("chunked") || l.contains("Qwen"))
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    let chunked_line = a_log
        .iter()
        .find(|l| l.contains("chunked prefill"))
        .cloned()
        .unwrap_or_default();
    eprintln!("[Phase B.2a] chunked path engaged: {}", chunked_line);

    drop(server_a);

    // ── Server B: monolithic prefill (control) ──
    let server_b = spawn_server(&bin, &model_path, PORT_B, &[]);
    wait_for_readyz(&server_b);
    let canonical_b = fetch_canonical_model_id(&server_b);
    eprintln!(
        "[Phase B.2a] server B (control monolithic) ready on {}:{}",
        HOST, PORT_B
    );

    let b_decoded = chat_decode(&server_b, &canonical_b, PROMPT, MAX_TOKENS);
    assert!(
        !b_decoded.is_empty(),
        "[Phase B.2a] server B decoded empty"
    );
    eprintln!("[Phase B.2a] server B decoded {} bytes", b_decoded.len());

    // Falsifier: byte-for-byte identity.
    if a_decoded != b_decoded {
        let common_prefix = a_decoded
            .as_bytes()
            .iter()
            .zip(b_decoded.as_bytes())
            .take_while(|(a, b)| a == b)
            .count();
        let snippet_a = a_decoded
            .get(common_prefix..common_prefix.saturating_add(120))
            .unwrap_or("")
            .to_string();
        let snippet_b = b_decoded
            .get(common_prefix..common_prefix.saturating_add(120))
            .unwrap_or("")
            .to_string();
        panic!(
            "[Phase B.2a] FALSIFIED — chunked prefill (A) vs monolithic (B) \
             produce non-byte-identical output.\n\
             server A (chunked, stride={TEST_STRIDE}) len={} bytes\n\
             server B (monolithic) len={} bytes\n\
             diverge at byte={}\n\
             A @ {}: {:?}\n\
             B @ {}: {:?}\n\
             ⇒ either (a) the chunked path fell back to monolithic (prompt \
             not divisible by stride; check eligibility log line), or \
             (b) DeltaNet recurrent state isn't propagating correctly \
             across forward_gpu_last_logits calls, or (c) full-attn \
             current_len cursor is mis-tracked. Phase B.2 is NOT feasible \
             at this stride until the divergence is root-caused.",
            a_decoded.len(),
            b_decoded.len(),
            common_prefix,
            common_prefix,
            snippet_a,
            common_prefix,
            snippet_b
        );
    }
    eprintln!(
        "[Phase B.2a] PASS — chunked (A: {} bytes) == monolithic (B: {} bytes) \
         BYTE-IDENTICAL. Phase B.2 foundation invariant verified.",
        a_decoded.len(),
        b_decoded.len()
    );
}
