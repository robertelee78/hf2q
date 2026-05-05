//! ADR-017 Phase E.a B.4 → B.5 — formerly-danger-zone byte-identity
//! falsifier.
//!
//! **Test history note:** This test was originally written for B.4
//! path (ii) to verify that prompts in the qL ∈ [2, 15] "danger zone"
//! (`prompt_len % stride ∈ {1..15}`) had chunked prefill SKIPPED
//! (the path-ii workaround) and ran monolithically.  B.5 lifted the
//! kernel-side gate at `gpu_full_attn.rs:1909+` from `seq_len >= 16`
//! to `kv_seq_len >= 16` for the resume path, proving via the
//! mlx-native kernel probe (0/N BF16 elements differ at qL ∈ {2, 8,
//! 15}, kL=130) that the documented "qL < 16 NaN bug" applied only
//! to single-K-tile (kL ≤ 16); the chunked-prefill last-chunk regime
//! always has multi-K-tile.  B.4 path-ii was REMOVED.
//!
//! Post-B.5 this test asserts the OPPOSITE: chunked prefill DOES
//! engage on the formerly-danger-zone prompt AND produces byte-
//! identical output to monolithic.  Same fixture, same byte-identity
//! gate, opposite engagement assertion — captures the structural
//! semantics shift.
//!
//! ## Why this matters
//!
//! The FA bf16 d256 fast/resume kernels gate on `kv_seq_len >= 16`
//! (`gpu_full_attn.rs:1909+`, post-B.5).  Chunked prefill's last chunk
//! always has `kv_seq_len = cur_len + seq_len ≥ stride > 16` because
//! cur_len ≥ stride (the prior chunk's full-stride contribution).  So
//! the resume kernel handles the last chunk byte-correctly regardless
//! of `seq_len ∈ [2, 15]`.
//!
//! ## Test design
//!
//! 1. Spawn server A with `HF2Q_KV_LCP_CHUNKED_PREFILL=1
//!    HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=64`.
//! 2. Spawn server B (control, no env flags).
//! 3. Send a prompt long enough to cross 3 stride boundaries with a
//!    partial-tail chunk in the formerly-danger zone (target
//!    prompt_len ≈ 130 tokens at stride=64 → chunks {64, 64, 2}).
//! 4. **Post-B.5 assertion:** A's stderr CONTAINS
//!    `[hf2q qwen35 chunked prefill]` (chunked engaged at all lengths).
//! 5. Assert: A's decoded bytes == B's decoded bytes (byte-identity
//!    preserved at the kernel level by FA RESUME on the partial-tail
//!    chunk).
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
const PORT_A: u16 = 52451;
const PORT_B: u16 = 52452;
const HOST: &str = "127.0.0.1";
const READYZ_BUDGET: Duration = Duration::from_secs(180);

/// Long enough for the tokenized prompt to cross 3 stride=64
/// boundaries with the partial tail in the danger zone (target
/// prompt_len ≈ 130, tail = 2 ∈ [1, 15]).  Verified empirically on
/// Qwen 3.6 35B-A3B-APEX-Q5_K_M; if tokenization changes and pushes
/// the prompt length out of the danger zone (e.g., into the safe zone
/// where tail ≥ 16), the test will report `chunked engaged` and fail
/// — that's a legitimate signal to update the prompt.
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
            "[Phase B.4] {ENV_PHASE_D_GATE}=1 set but \
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
    loop {
        if let Ok(200) = http_get_status(server.port, "/readyz") {
            return;
        }
        if start.elapsed() > READYZ_BUDGET {
            panic!(
                "[Phase B.4] /readyz did not return 200 within {READYZ_BUDGET:?} \
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

fn chat_decode(server: &ServerGuard, model: &str, prompt: &str, max_tokens: u32) -> String {
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
    let p = body.find(key).unwrap_or_else(|| panic!("no content field"));
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

/// **ADR-017 Phase E.a B.4 path (ii) danger-zone bypass falsifier**:
/// when a prompt lands in the qL ∈ [2, 15] danger zone, the engine
/// correctly skips chunked prefill (preventing the kernel-level
/// divergence between FA fast/resume and legacy F32 SDPA) and produces
/// byte-identical output to the no-chunked-flag baseline.
#[test]
fn phase_b4_danger_zone_bypass_byte_identity() {
    let model_path = match resolve_qwen35_model_path_or_skip() {
        Some(p) => p,
        None => {
            eprintln!(
                "[Phase B.4] {ENV_PHASE_D_GATE}=1 + \
                 {ENV_QWEN35_MODEL_PATH}=PATH not set — short-circuit."
            );
            return;
        }
    };
    let bin = hf2q_binary_path();
    assert!(bin.exists(), "[Phase B.4] hf2q binary not found");

    let stride_str = TEST_STRIDE.to_string();

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
        "[Phase B.4] server A (CHUNKED_PREFILL=1, stride={}) ready on {}:{}",
        TEST_STRIDE, HOST, PORT_A
    );

    let a_decoded = chat_decode(&server_a, &canonical_a, PROMPT, MAX_TOKENS);
    assert!(!a_decoded.is_empty(), "[Phase B.4] A decoded empty");
    eprintln!("[Phase B.4] server A decoded {} bytes", a_decoded.len());

    // Post-B.5 engagement: chunked prefill MUST engage on the
    // formerly-danger-zone prompt (B.4 path-ii bypass was REMOVED in
    // B.5 once the kernel probe proved byte-correctness at small qL
    // multi-K-tile).  The kernel-level FA RESUME path now handles the
    // partial-tail chunk byte-identically to monolithic.
    let a_log = server_a.log_tail();
    let chunked_engaged = a_log.iter().any(|l| l.contains("chunked prefill"));
    assert!(
        chunked_engaged,
        "[Phase B.4→B.5] FALSIFIED — server A's stderr LACKS `chunked \
         prefill` line, meaning chunked SKIPPED on the formerly-danger-\
         zone prompt.  B.5 should have engaged chunked at all lengths \
         (the path-ii bypass was removed).  Either the bypass code came \
         back, OR the chunked_eligible gate is rejecting the prompt for \
         a different reason. Server A stderr tail (last 30 lines):\n{}",
        a_log
            .iter()
            .rev()
            .take(30)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    let chunked_line = a_log
        .iter()
        .find(|l| l.contains("chunked prefill"))
        .cloned()
        .unwrap_or_default();
    eprintln!(
        "[Phase B.4→B.5] post-B.5 engagement: {chunked_line}"
    );

    drop(server_a);

    let server_b = spawn_server(&bin, &model_path, PORT_B, &[]);
    wait_for_readyz(&server_b);
    let canonical_b = fetch_canonical_model_id(&server_b);
    eprintln!(
        "[Phase B.4] server B (control, no chunked) ready on {}:{}",
        HOST, PORT_B
    );

    let b_decoded = chat_decode(&server_b, &canonical_b, PROMPT, MAX_TOKENS);
    assert!(!b_decoded.is_empty(), "[Phase B.4] B decoded empty");
    eprintln!("[Phase B.4] server B decoded {} bytes", b_decoded.len());

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
            "[Phase B.4] FALSIFIED — A (chunked-flag-set, danger-zone \
             bypassed → monolithic) and B (no chunked flag → \
             monolithic) produced non-byte-identical output.\n\
             A len={} bytes, B len={} bytes, diverge at byte={}\n\
             A @ {}: {:?}\n\
             B @ {}: {:?}\n\
             ⇒ Both should have run monolithic prefill on the same \
             prompt; this divergence indicates either (a) the bypass \
             didn't actually trigger (test would have caught via \
             chunked-skipped assertion above — investigate), or (b) \
             a non-determinism bug in monolithic prefill itself.",
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
        "[Phase B.4→B.5] PASS — A ({} bytes, chunked engaged via FA \
         RESUME on partial-tail) == B ({} bytes, control monolithic) \
         BYTE-IDENTICAL. B.5 verified — formerly-danger-zone prompts \
         now correctly chunked at all stride boundaries with byte-\
         equivalent output to monolithic.",
        a_decoded.len(),
        b_decoded.len()
    );
}
