//! ADR-017 Phase E option (a) iter-3 — K<N partial-prefix byte-identity
//! falsifier (dossier §10.4 corrected spec).
//!
//! This is the LOAD-BEARING positive falsifier for the iter-3 partial-
//! prefill resume path. Iter-3 ships under env-gates `HF2Q_KV_LCP_RESUME=1`
//! + `HF2Q_USE_DENSE=1` (default OFF). The spec contract: a request whose
//! prompt shares a non-trivial prefix with a previously-served prompt
//! triggers `forward_prefill_with_soft_tokens_resume(restored_lcp=Some(K))`
//! — reusing cached `dense_kvs[*][0..K)` in place AND skipping the
//! `cache.write_pos = 0` reset. The falsifier asserts that the resumed-
//! path output is **byte-identical** to a fresh-prefill of the same
//! prompt.
//!
//! ## Why this can't be a unit test
//!
//! The failure mode (silent corruption) is end-to-end: it surfaces only
//! after prefill + decode produces decoded bytes. Unit-testing the resume
//! path requires a real Metal device + Gemma 4 weights + tokenizer. So
//! this test is gated under `HF2Q_KV_PERSIST_PHASE_D=1` and
//! `HF2Q_KV_PERSIST_E2E_MODEL_PATH=<gguf>` (mirrors the existing R-C4
//! sourdough gate at `kv_persist_gemma4_roundtrip.rs::
//! kv_persist_phase_d_coherence_e2e`).
//!
//! ## Design — two-server byte-identity comparison
//!
//! Server A (resume path under test) is launched with
//! `HF2Q_KV_LCP_RESUME=1 + HF2Q_USE_DENSE=1`. We send a priming
//! prompt `Q` (registry stores its post-decode KV state), then send a
//! probe prompt `P` whose first `K` tokens equal `Q[..K]`. Server A's
//! engine probes the registry, hits with `K`, calls `take_prefix`, and
//! dispatches `forward_prefill_with_soft_tokens_resume(Some(K))`.
//!
//! Server B (control) is launched with `HF2Q_USE_DENSE=1` only — no
//! resume. We send `P` to server B directly; the engine takes the
//! pre-iter-3 wholesale-reset path.
//!
//! Falsifier: `server_A_decoded(P) != server_B_decoded(P)` byte-for-byte.
//! On mismatch the iter-3 path produces non-byte-identical output; iter-3
//! is BROKEN and MUST NOT promote from default-OFF until the divergence is
//! root-caused.
//!
//! ## Why two distinct prompts (not one prime + one probe on same server)
//!
//! Sending the same prompt twice on server A would hit the upstream
//! `PromptCache` full-equality replay (Phase E option b, shipped at
//! `d17163b`) — replay decodes from cached output, NOT from the resumed
//! KV state. PromptCache hits BEFORE LCP probe runs, so the second-turn
//! request bypasses iter-3 entirely. To exercise the iter-3 path we
//! need `P != Q` (PromptCache misses) AND `P` shares a non-trivial
//! prefix with `Q` (LcpRegistry hits with `K = LCP(P, Q) > 0` and
//! `K < P.len()`).

use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const ENV_PHASE_D_GATE: &str = "HF2Q_KV_PERSIST_PHASE_D";
const ENV_MODEL_PATH: &str = "HF2Q_KV_PERSIST_E2E_MODEL_PATH";
const PORT_A: u16 = 52391;
const PORT_B: u16 = 52392;
const HOST: &str = "127.0.0.1";
const READYZ_BUDGET: Duration = Duration::from_secs(180);

/// Prompts: Q is a strict prefix of P at the chat-template-rendered
/// level so the LCP probe finds K = Q.len() shared tokens.
///
/// Concrete content: Q ends mid-sentence; P appends 4-6 tokens of
/// continuation text. Greedy decode at temperature 0 gives byte-stable
/// output deterministically.
const PROMPT_Q: &str = "List in alphabetical order the colors red, green, blue,";
const PROMPT_P: &str =
    "List in alphabetical order the colors red, green, blue, and yellow with one descriptor each.";
const MAX_TOKENS: u32 = 64;

fn hf2q_binary_path() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir).join("target/release/hf2q")
}

fn resolve_model_path_or_skip() -> Option<PathBuf> {
    let gate_set = std::env::var(ENV_PHASE_D_GATE).as_deref() == Ok("1");
    if !gate_set {
        return None;
    }
    let path = std::env::var(ENV_MODEL_PATH).ok()?;
    if path.is_empty() {
        return None;
    }
    let p = PathBuf::from(path);
    if !p.exists() {
        panic!(
            "[iter-3 falsifier] {ENV_PHASE_D_GATE}=1 set but {ENV_MODEL_PATH} \
             does not point to an existing file"
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
    write!(s, "GET {path} HTTP/1.1\r\nHost: {HOST}:{port}\r\nConnection: close\r\n\r\n")?;
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

/// Fetch the full `/metrics` text body from the server and return it
/// for line-level inspection.
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

/// Parse `hf2q_kv_lcp_detected_total <N>` out of the /metrics body.
/// Returns 0 if the line is absent (parseable failure mode).
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
    use std::io::Write;
    let mut s = TcpStream::connect((HOST, server.port)).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(10))).ok();
    write!(
        s,
        "GET /v1/models HTTP/1.1\r\nHost: {HOST}:{}\r\nConnection: close\r\n\r\n",
        server.port
    )
    .unwrap();
    let mut buf = String::new();
    use std::io::Read;
    s.read_to_string(&mut buf).expect("read /v1/models");
    let body_start = buf.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    let body = &buf[body_start..];
    let key = "\"id\":\"";
    let p = body.find(key).expect("find model id in /v1/models");
    let rest = &body[p + key.len()..];
    let end = rest.find('"').expect("close quote on model id");
    rest[..end].to_string()
}

/// Send a /v1/chat/completions request with greedy decode (T=0) and
/// return the assistant text from `choices[0].message.content`.
fn chat_decode(server: &ServerGuard, model: &str, prompt: &str, max_tokens: u32) -> String {
    use std::io::{Read, Write};
    let body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"{prompt}"}}],"max_tokens":{max_tokens},"temperature":0,"stream":false}}"#
    );
    let mut s = TcpStream::connect((HOST, server.port)).expect("connect");
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
    s.read_to_string(&mut buf).expect("read response");
    let body_start = buf.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    let body = &buf[body_start..];

    // Naive JSON content extractor — sufficient for the well-known
    // shape `{"choices":[{"message":{"content":"...","role":"assistant"...}}]}`.
    let key = "\"content\":\"";
    let p = body
        .find(key)
        .unwrap_or_else(|| panic!("no content field in response body: {body}"));
    let rest = &body[p + key.len()..];
    // Walk through characters handling escape sequences until unescaped
    // closing quote. The OpenAI spec serializes content as a JSON string;
    // \\" must be unescaped, raw \" terminates.
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

/// ADR-017 Phase E option (a) iter-3 — K<N partial-prefix byte-identity
/// falsifier (dossier §10.4 corrected spec).
///
/// Two servers, byte-identity comparison:
///
///   * **Server A**: `HF2Q_KV_LCP_RESUME=1 + HF2Q_USE_DENSE=1`. Send Q
///     (priming, registry stores), then send P. P's prompt shares
///     prefix with Q at the rendered-template level; engine's LCP probe
///     hits, take_prefix consumes Q's entry, install dense_kvs into
///     weights, dispatch `forward_prefill_with_soft_tokens_resume(Some(K))`.
///   * **Server B**: `HF2Q_USE_DENSE=1` only (no resume). Send P
///     directly; engine takes the pre-iter-3 wholesale-reset path.
///
/// Falsifier: `server_A_decoded(P) != server_B_decoded(P)` byte-for-byte.
///
/// Heavy: spawns two Gemma 4 26B engines back-to-back (2 × ~14 GB
/// RAM, 2 × ~25s load). Operator-gated under
/// `HF2Q_KV_PERSIST_PHASE_D=1`; default `cargo test` short-circuits.
#[test]
fn iter3_partial_prefix_byte_identity() {
    let model_path = match resolve_model_path_or_skip() {
        Some(p) => p,
        None => {
            eprintln!(
                "[iter-3 falsifier] {ENV_PHASE_D_GATE}=1 not set — short-circuit. \
                 Set {ENV_PHASE_D_GATE}=1 + {ENV_MODEL_PATH}=PATH to run."
            );
            return;
        }
    };
    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "[iter-3 falsifier] hf2q binary not found at {} — \
         did `cargo build --release` run?",
        bin.display()
    );

    // ── Server A: resume path under test ──
    let server_a = spawn_server(
        &bin,
        &model_path,
        PORT_A,
        &[("HF2Q_KV_LCP_RESUME", "1")],
    );
    wait_for_readyz(&server_a);
    let canonical_a = fetch_canonical_model_id(&server_a);
    eprintln!(
        "[iter-3 falsifier] server A (LCP_RESUME=1) ready on {}:{} model={}",
        HOST, PORT_A, canonical_a
    );

    // Prime the LCP registry with Q on server A.
    let q_decoded_a = chat_decode(&server_a, &canonical_a, PROMPT_Q, MAX_TOKENS);
    assert!(
        !q_decoded_a.is_empty(),
        "[iter-3 falsifier] Q decoded empty on server A"
    );
    eprintln!(
        "[iter-3 falsifier] server A Q decoded {} bytes (priming registry)",
        q_decoded_a.len()
    );

    // Send P to server A — registry has Q's entry, LCP probe should
    // hit at K = LCP(P, Q) > 0 (PROMPT_Q is a literal prefix of
    // PROMPT_P modulo the differing suffix). The resume path engages.
    let p_decoded_a = chat_decode(&server_a, &canonical_a, PROMPT_P, MAX_TOKENS);
    assert!(
        !p_decoded_a.is_empty(),
        "[iter-3 falsifier] P decoded empty on server A (resume path)"
    );
    eprintln!(
        "[iter-3 falsifier] server A P decoded {} bytes (resume path under test)",
        p_decoded_a.len()
    );

    // Engagement assertion: confirm the iter-3 resume path actually
    // FIRED on server A by reading /metrics. Without this check the
    // test could trivially pass if the LCP probe silently failed (both
    // servers would run fresh prefill and trivially match). The
    // detected_total counter increments whenever a non-trivial
    // partial-prefix opportunity is seen (0 < K < N) — engagement
    // requires at least 1 such hit by the time we reach this point.
    let metrics_body_a = fetch_metrics(&server_a);
    let lcp_detected_a = metric_lcp_detected_total(&metrics_body_a);
    assert!(
        lcp_detected_a >= 1,
        "[iter-3 falsifier] server A's /metrics shows hf2q_kv_lcp_detected_total={} \
         — the LCP probe did NOT detect a partial-prefix hit on the Q+P sequence. \
         The byte-identity check below is therefore VACUOUS (server A took the \
         fresh-prefill path, same as server B). Test design assumption violated; \
         verify PROMPT_Q tokenizes to a strict prefix of PROMPT_P.\n\n\
         /metrics body excerpt:\n{}",
        lcp_detected_a,
        metrics_body_a
            .lines()
            .filter(|l| l.contains("lcp"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    eprintln!(
        "[iter-3 falsifier] server A engagement confirmed: \
         hf2q_kv_lcp_detected_total={}",
        lcp_detected_a
    );

    // Drop server A so Metal device + RAM are free before server B
    // loads (constrained-memory operators).
    drop(server_a);

    // ── Server B: control (no resume) ──
    let server_b = spawn_server(&bin, &model_path, PORT_B, &[]);
    wait_for_readyz(&server_b);
    let canonical_b = fetch_canonical_model_id(&server_b);
    eprintln!(
        "[iter-3 falsifier] server B (control, no resume) ready on {}:{} model={}",
        HOST, PORT_B, canonical_b
    );

    // Send P directly to server B — no priming, no LCP entry, fresh
    // wholesale-reset prefill.
    let p_decoded_b = chat_decode(&server_b, &canonical_b, PROMPT_P, MAX_TOKENS);
    assert!(
        !p_decoded_b.is_empty(),
        "[iter-3 falsifier] P decoded empty on server B (control)"
    );
    eprintln!(
        "[iter-3 falsifier] server B P decoded {} bytes (control)",
        p_decoded_b.len()
    );

    // Falsifier: byte-for-byte identity.
    if p_decoded_a != p_decoded_b {
        let common_prefix = p_decoded_a
            .as_bytes()
            .iter()
            .zip(p_decoded_b.as_bytes())
            .take_while(|(a, b)| a == b)
            .count();
        let snippet_a = p_decoded_a
            .get(common_prefix..common_prefix.saturating_add(120))
            .unwrap_or("")
            .to_string();
        let snippet_b = p_decoded_b
            .get(common_prefix..common_prefix.saturating_add(120))
            .unwrap_or("")
            .to_string();
        panic!(
            "[iter-3 falsifier] FAIL — P decoded bytes differ between resume (A) \
             and fresh (B) paths.\n\
             server A (resume) len={} bytes\n\
             server B (control) len={} bytes\n\
             diverge at byte offset={}\n\
             server A @ {}: {:?}\n\
             server B @ {}: {:?}\n\
             ⇒ iter-3 partial-prefill resume produces non-byte-identical \
             output. Phase E.a is FALSIFIED at K<N. Do NOT promote env-gate \
             from default-OFF.",
            p_decoded_a.len(),
            p_decoded_b.len(),
            common_prefix,
            common_prefix,
            snippet_a,
            common_prefix,
            snippet_b
        );
    }
    eprintln!(
        "[iter-3 falsifier] PASS — server A (resume) bytes ({} bytes) \
         == server B (control) bytes ({} bytes) BYTE-IDENTICAL",
        p_decoded_a.len(),
        p_decoded_b.len()
    );
}
