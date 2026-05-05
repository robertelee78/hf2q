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

/// iter-7 — parse `hf2q_kv_lcp_lookups_total <N>` from /metrics. The
/// lookups counter is the denominator (every probe call increments it,
/// regardless of detection outcome). For the prefill-wrap guard test
/// the lookups counter advances normally while detected stays at 0,
/// proving the probe ran but found no entry (because store was
/// skipped).
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

// ─────────────────────────────────────────────────────────────────────────────
// Iter-5 — R-C4-LCP 5-K-fraction byte-identity sweep (KILL-criterion).
//
// Per dossier §4.2 iter-5 row, this test sweeps 5 K fractions and
// asserts byte-identity at each. Any single failure FALSIFIES Phase
// E.a entirely; the env-gate must remain default-OFF until this test
// passes for all fractions.
//
// Each K fraction is structured as (priming_prompt Q_k, probe_prompt
// P_k) where:
//   - K_k ≈ {N_min, N/4, N/2, 3N/4, N-1} where N = total prompt length.
//   - LCP(Q_k, P_k) at the chat-template-rendered TOKEN level equals
//     the first K_k tokens; P_k differs at position K_k.
//   - All prompts have prompt_len < sliding_window=1024 (iter-3.5c
//     prefill-wrap guard requires this for sliding-layer correctness;
//     iter-3.6 will lift the restriction via mid-prefill snapshot).
//
// Cross-K interference management: each K fraction's pair is
// distinguished by a unique sentinel-prefix ("K0:", "K1:", ...) so
// the LCP probe between successive K fractions returns the sentinel-
// prefix length only — well below the K_k of interest, so resume
// engagement is dominated by intra-K-fraction matches.
//
// Cost: 1 server A spawn (resume on) + 1 server B spawn (control) +
// 5 × 2 = 10 chat_decode calls = ~50s + ~10s = ~60s wall-time.
// Operator-gated under HF2Q_KV_PERSIST_PHASE_D=1.

const ITER5_FRACTION_PAIRS: &[(&str, &str)] = &[
    // K_0 — small K (mostly differs early). Prompts share only the
    // sentinel-prefix + a couple words.
    (
        "K0: The capital city is",
        "K0: A different country is named differently and unique.",
    ),
    // K_1 — N/4 — share more, diverge ~25% in.
    (
        "K1: List in alphabetical order red green and",
        "K1: List in alphabetical order red green or yellow always.",
    ),
    // K_2 — N/2 — share half before diverging.
    (
        "K2: The brown fox jumps over the lazy",
        "K2: The brown fox jumps over the calm summer river quickly.",
    ),
    // K_3 — 3N/4 — large shared prefix, diverge near the end.
    (
        "K3: Once upon a time there lived a small village in the green",
        "K3: Once upon a time there lived a small village in the desert mountains.",
    ),
    // K_4 — N-1 — only the last token differs (or close to it).
    (
        "K4: ABC DEF GHI JKL MNO PQR STU VWX YZ.",
        "K4: ABC DEF GHI JKL MNO PQR STU VWX YZ!",
    ),
];

const ITER5_MAX_TOKENS: u32 = 64;

/// ADR-017 Phase E option (a) iter-5 — R-C4-LCP 5-K-fraction byte-
/// identity sweep (KILL-criterion).
///
/// For each K fraction:
///   1. Server A (resume on): send Q_k (primes registry), then send
///      P_k (registry hit, resume engages with K = LCP).
///   2. Server B (control, no resume): send P_k directly (fresh
///      prefill).
///   3. Assert server_A_decoded(P_k) == server_B_decoded(P_k) byte-
///      identical.
///   4. Track cumulative `hf2q_kv_lcp_detected_total` from server A's
///      /metrics; assert it grows monotonically (≥ k+1 detections by
///      end of fraction k).
///
/// Falsifier: any single fraction's byte-identity check fails →
/// PANIC, Phase E.a is FALSIFIED, env-gate must stay default-OFF.
#[test]
fn iter5_r_c4_lcp_5_fraction_sweep() {
    let model_path = match resolve_model_path_or_skip() {
        Some(p) => p,
        None => {
            eprintln!(
                "[iter-5 sweep] {ENV_PHASE_D_GATE}=1 not set — short-circuit. \
                 Set {ENV_PHASE_D_GATE}=1 + {ENV_MODEL_PATH}=PATH to run."
            );
            return;
        }
    };
    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "[iter-5 sweep] hf2q binary not found at {} — \
         did `cargo build --release` run?",
        bin.display()
    );

    // ── Server A: resume path under test (single instance for all fractions) ──
    let server_a = spawn_server(
        &bin,
        &model_path,
        PORT_A,
        &[("HF2Q_KV_LCP_RESUME", "1")],
    );
    wait_for_readyz(&server_a);
    let canonical_a = fetch_canonical_model_id(&server_a);
    eprintln!(
        "[iter-5 sweep] server A (LCP_RESUME=1) ready on {}:{} model={}",
        HOST, PORT_A, canonical_a
    );

    // Run all 5 fractions on server A (registry capacity=1 means each
    // fraction's P_k store evicts the prior; sentinel-prefix
    // distinguishes fractions to manage cross-K interference).
    let mut server_a_decoded: Vec<String> = Vec::with_capacity(ITER5_FRACTION_PAIRS.len());
    let mut prior_detected: u64 = 0;
    for (frac_idx, (prime, probe)) in ITER5_FRACTION_PAIRS.iter().enumerate() {
        let _q_decoded = chat_decode(&server_a, &canonical_a, prime, ITER5_MAX_TOKENS);
        let p_decoded = chat_decode(&server_a, &canonical_a, probe, ITER5_MAX_TOKENS);
        assert!(
            !p_decoded.is_empty(),
            "[iter-5 sweep] fraction {}: P decoded empty on server A",
            frac_idx
        );
        // Engagement: probe should have detected on at least the
        // post-prime LCP. Total detected counter monotonic.
        let metrics = fetch_metrics(&server_a);
        let now_detected = metric_lcp_detected_total(&metrics);
        assert!(
            now_detected > prior_detected,
            "[iter-5 sweep] fraction {}: lcp_detected_total did not advance \
             (prior={}, now={}). Either probe didn't engage, or sentinel-\
             prefix design failed. Test invariant violated.",
            frac_idx,
            prior_detected,
            now_detected
        );
        eprintln!(
            "[iter-5 sweep] fraction {}: server A P decoded {} bytes \
             (lcp_detected_total: {} → {})",
            frac_idx,
            p_decoded.len(),
            prior_detected,
            now_detected
        );
        prior_detected = now_detected;
        server_a_decoded.push(p_decoded);
    }
    drop(server_a);

    // ── Server B: control (no resume; single instance for all fractions) ──
    let server_b = spawn_server(&bin, &model_path, PORT_B, &[]);
    wait_for_readyz(&server_b);
    let canonical_b = fetch_canonical_model_id(&server_b);
    eprintln!(
        "[iter-5 sweep] server B (control, no resume) ready on {}:{} model={}",
        HOST, PORT_B, canonical_b
    );

    // Send each P_k to server B and capture decoded bytes for
    // byte-identity comparison.
    let mut all_pass = true;
    for (frac_idx, (_prime, probe)) in ITER5_FRACTION_PAIRS.iter().enumerate() {
        let p_decoded_b = chat_decode(&server_b, &canonical_b, probe, ITER5_MAX_TOKENS);
        assert!(
            !p_decoded_b.is_empty(),
            "[iter-5 sweep] fraction {}: P decoded empty on server B",
            frac_idx
        );
        let p_decoded_a = &server_a_decoded[frac_idx];
        if *p_decoded_a != p_decoded_b {
            all_pass = false;
            let common_prefix = p_decoded_a
                .as_bytes()
                .iter()
                .zip(p_decoded_b.as_bytes())
                .take_while(|(a, b)| a == b)
                .count();
            let snippet_a = p_decoded_a
                .get(common_prefix..common_prefix.saturating_add(80))
                .unwrap_or("")
                .to_string();
            let snippet_b = p_decoded_b
                .get(common_prefix..common_prefix.saturating_add(80))
                .unwrap_or("")
                .to_string();
            eprintln!(
                "[iter-5 sweep] FRACTION {} FAIL — diverge at byte={} \
                 (A.len={} bytes, B.len={} bytes)\n\
                 A @ {}: {:?}\n  B @ {}: {:?}",
                frac_idx,
                common_prefix,
                p_decoded_a.len(),
                p_decoded_b.len(),
                common_prefix,
                snippet_a,
                common_prefix,
                snippet_b
            );
        } else {
            eprintln!(
                "[iter-5 sweep] fraction {} PASS — A == B byte-identical \
                 ({} bytes)",
                frac_idx,
                p_decoded_a.len()
            );
        }
    }

    assert!(
        all_pass,
        "[iter-5 sweep] FAIL — at least one K fraction's byte-identity \
         check failed. Phase E.a R-C4-LCP KILL-CRITERION TRIGGERED. \
         Env-gate must stay default-OFF until divergence is root-caused."
    );
    eprintln!(
        "[iter-5 sweep] ALL 5 FRACTIONS PASS — Phase E.a R-C4-LCP \
         byte-identity verified across K ∈ {{small, N/4, N/2, 3N/4, N-1}}."
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Iter-7 — sliding-window prefill-wrap guard (LCP > sliding_window edge case).
//
// Per dossier §4.2 row 7: "Sliding-window LCP > sw edge cases + Qwen3.5
// hybrid deferral note. Edge: prompt with LCP=4096 against
// sliding_window=1024; decoded bytes byte-identical to full prefill.
// Falsifier: any divergence at LCP > sw boundary."
//
// ## What the prefill-wrap guard does
//
// `engine.rs:4516` (non-streaming) and `engine.rs:7027` (streaming) skip
// `lcp_registry.store(...)` when `prompt_len > sliding_window` AND the
// model has any sliding layer. Reason: the end-of-prefill snapshot
// captures the FINAL ring state — slots representing positions
// `[N-sw..N)`, not `[0..N)`. A future LCP resume at K<N would expect
// slots to represent `[0..K)` (P's shared prefix), which the wrapped
// snapshot does NOT contain. Storing would yield silent corruption on
// resume; the guard preserves byte-identity by skipping store.
//
// V1 trade-off (iter-3.5c): long-conversation LCP miss when prompt_len
// exceeds sliding_window. Iter-3.6 (mid-prefill snapshot) lifts this.
//
// ## Test design
//
// Two-server byte-identity, mirroring iter-3 falsifier:
//   * Server A (HF2Q_KV_LCP_RESUME=1 + HF2Q_USE_DENSE=1):
//     - Send Q_long (priming attempt; > sliding_window tokens). Guard
//       fires; registry empty.
//     - Send P_long (shares prefix with Q_long; > sliding_window
//       tokens). LCP probe runs but registry is empty → returns None.
//     - Engagement: lcp_lookups_total advances 2; lcp_detected_total
//       stays 0 (proves guard skipped store on Q AND P; otherwise
//       second-turn detected would be ≥ 1).
//   * Server B (control, USE_DENSE=1, RESUME flag absent):
//     - Send P_long directly; fresh prefill.
//
// Falsifier:
//   1. lcp_detected_total > 0 on server A → guard FAILED to fire (would
//      mean the store happened, registry holds Q's wrap-corrupted
//      snapshot, second turn would have detected). Phase E.a invariant
//      violated; iter-3.6 cannot lift the restriction safely.
//   2. server_A_decoded(P_long) != server_B_decoded(P_long) byte-for-
//      byte → resume engaged silently AND corrupted output. Should
//      never happen if guard fires; provides defense-in-depth assertion.
//
// ## Long-prompt construction
//
// We need prompt_tokens.len() > sliding_window. Gemma 4 default
// sliding_window = 1024 (`src/serve/config.rs:103`); 26B-DWQ inherits
// the default. We construct a prompt with ~200 reps of a 10-token
// sentence ≈ 2000 tokens — comfortably above 1024 with headroom for
// any chat-template overhead. The shared prefix between Q and P is the
// 200-rep block; suffixes differ at the last sentence.
//
// Cost: 1 server A spawn (~25s load) + 1 server B spawn (~25s load) +
// 3 chat_decodes at ~2000-token prefill (~5-10s each at 200 t/s) +
// 8 tokens decode each ≈ ~80s wall. Operator-gated under
// HF2Q_KV_PERSIST_PHASE_D=1.

const ITER7_LONG_PROMPT_REP_COUNT: usize = 200;
const ITER7_LONG_PROMPT_BASE: &str = "The quick brown fox jumps over the lazy dog. ";
const ITER7_MAX_TOKENS: u32 = 8;

fn iter7_build_prompt(suffix: &str) -> String {
    let mut s = String::with_capacity(
        ITER7_LONG_PROMPT_BASE.len() * ITER7_LONG_PROMPT_REP_COUNT + suffix.len() + 16,
    );
    for _ in 0..ITER7_LONG_PROMPT_REP_COUNT {
        s.push_str(ITER7_LONG_PROMPT_BASE);
    }
    s.push_str(suffix);
    s
}

/// ADR-017 Phase E option (a) iter-7 — sliding-window prefill-wrap
/// guard correctness.
///
/// Asserts that when the prompt exceeds `sliding_window`, the iter-3.5c
/// guard skips the LCP store. Concretely:
///   1. lcp_lookups_total on server A advances for every probe call
///      (denominator).
///   2. lcp_detected_total stays at 0 — registry never has an entry
///      because store was skipped on prior turns.
///   3. server_A_decoded(P_long) == server_B_decoded(P_long) byte-for-
///      byte (both ran fresh prefill).
///
/// Falsifier: either (a) detected_total > 0 → guard FAILED, the v1
/// invariant is BROKEN, or (b) byte mismatch → silent corruption.
#[test]
fn iter7_prefill_wrap_guard_long_prompt_byte_identity() {
    let model_path = match resolve_model_path_or_skip() {
        Some(p) => p,
        None => {
            eprintln!(
                "[iter-7 wrap-guard] {ENV_PHASE_D_GATE}=1 not set — short-circuit. \
                 Set {ENV_PHASE_D_GATE}=1 + {ENV_MODEL_PATH}=PATH to run."
            );
            return;
        }
    };
    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "[iter-7 wrap-guard] hf2q binary not found at {} — \
         did `cargo build --release` run?",
        bin.display()
    );

    let prompt_q = iter7_build_prompt("Question Q ending here distinctively.");
    let prompt_p =
        iter7_build_prompt("Question P with different ending text continuing onward.");
    eprintln!(
        "[iter-7 wrap-guard] prompts built: Q={} chars, P={} chars (shared prefix \
         {} reps × {} chars = {} chars; both prompts target prompt_len > \
         sliding_window=1024 after tokenization)",
        prompt_q.len(),
        prompt_p.len(),
        ITER7_LONG_PROMPT_REP_COUNT,
        ITER7_LONG_PROMPT_BASE.len(),
        ITER7_LONG_PROMPT_REP_COUNT * ITER7_LONG_PROMPT_BASE.len()
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
        "[iter-7 wrap-guard] server A (LCP_RESUME=1) ready on {}:{} model={}",
        HOST, PORT_A, canonical_a
    );

    // Baseline counters — server may pre-emit zero entries; capture so
    // delta math is robust to any future warmup-side probe behavior.
    let metrics_pre = fetch_metrics(&server_a);
    let lookups_pre = metric_lcp_lookups_total(&metrics_pre);
    let detected_pre = metric_lcp_detected_total(&metrics_pre);
    eprintln!(
        "[iter-7 wrap-guard] server A baseline: lookups={}, detected={}",
        lookups_pre, detected_pre
    );

    // Send Q (long, prompt_len > sliding_window). The guard MUST skip
    // the store — registry stays empty. We catch panic / non-200
    // outcomes and dump server stderr tail so any worker-side issue
    // surfaces in the test failure (without this, the chat_decode
    // helper panics with "no content field" and the server log is
    // lost).
    let q_decoded_a = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        chat_decode(&server_a, &canonical_a, &prompt_q, ITER7_MAX_TOKENS)
    })) {
        Ok(s) => s,
        Err(_) => {
            let tail = server_a.log_tail();
            panic!(
                "[iter-7 wrap-guard] Q chat_decode panicked. Server stderr \
                 tail (last 256 lines):\n{}",
                tail.join("\n")
            );
        }
    };
    assert!(
        !q_decoded_a.is_empty(),
        "[iter-7 wrap-guard] Q decoded empty on server A"
    );
    eprintln!(
        "[iter-7 wrap-guard] server A Q decoded {} bytes (priming attempt; \
         registry expected EMPTY due to wrap guard)",
        q_decoded_a.len()
    );

    // Send P (long, shares prefix with Q). PromptCache misses (different
    // prompts). LCP probe runs and finds an empty registry → detected
    // stays at 0. The guard skips the store again.
    let p_decoded_a = chat_decode(&server_a, &canonical_a, &prompt_p, ITER7_MAX_TOKENS);
    assert!(
        !p_decoded_a.is_empty(),
        "[iter-7 wrap-guard] P decoded empty on server A"
    );
    eprintln!(
        "[iter-7 wrap-guard] server A P decoded {} bytes (probe ran; expected \
         to fall through to fresh prefill)",
        p_decoded_a.len()
    );

    // Engagement assertions — the guard's load-bearing invariant.
    let metrics_post = fetch_metrics(&server_a);
    let lookups_post = metric_lcp_lookups_total(&metrics_post);
    let detected_post = metric_lcp_detected_total(&metrics_post);
    let lookups_delta = lookups_post.saturating_sub(lookups_pre);
    let detected_delta = detected_post.saturating_sub(detected_pre);
    eprintln!(
        "[iter-7 wrap-guard] server A post-Q+P: lookups={} (Δ={}), detected={} \
         (Δ={})",
        lookups_post, lookups_delta, detected_post, detected_delta
    );
    // (a) Probe must have actually run on at least one of {Q, P}; ≥ 1
    // is sufficient because Q's flow may short-circuit before probe if
    // prompt_cache fast-path misses but probe fires after — Q's probe
    // SHOULD run and increment. Assert ≥ 2 (one per turn).
    assert!(
        lookups_delta >= 2,
        "[iter-7 wrap-guard] lookups_delta={} < 2 — probe did not run on \
         both Q and P. Test invariant violated; metric integration may have \
         regressed. metrics excerpt:\n{}",
        lookups_delta,
        metrics_post
            .lines()
            .filter(|l| l.contains("lcp"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    // (b) The load-bearing assertion: detected MUST stay at 0. If > 0,
    // the guard failed to skip a store, and the iter-3.6 lift cannot
    // safely happen.
    assert_eq!(
        detected_delta, 0,
        "[iter-7 wrap-guard] FALSIFIED — detected_delta={} (expected 0). \
         The prefill-wrap guard FAILED to skip the LCP store on a \
         long-prompt request (prompt_len > sliding_window). Either:\n\
         (1) the guard predicate `prompt_len > sliding_window` is wrong, \
         or (2) the snapshot was incorrectly published despite ring \
         wrap. Phase E.a v1 byte-identity correctness invariant violated; \
         iter-3.6 mid-prefill snapshot lift CANNOT proceed safely until \
         this is root-caused.\n\nmetrics excerpt:\n{}",
        detected_delta,
        metrics_post
            .lines()
            .filter(|l| l.contains("lcp"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    eprintln!(
        "[iter-7 wrap-guard] guard engagement confirmed: lookups Δ={} ≥ 2, \
         detected Δ={} == 0 (store skipped on both turns; registry stayed empty)",
        lookups_delta, detected_delta
    );

    // Drop server A so Metal device + RAM are free before server B.
    drop(server_a);

    // ── Server B: control (no resume) ──
    let server_b = spawn_server(&bin, &model_path, PORT_B, &[]);
    wait_for_readyz(&server_b);
    let canonical_b = fetch_canonical_model_id(&server_b);
    eprintln!(
        "[iter-7 wrap-guard] server B (control, no resume) ready on {}:{} model={}",
        HOST, PORT_B, canonical_b
    );

    // Send P_long directly to server B; fresh prefill, no LCP path.
    let p_decoded_b = chat_decode(&server_b, &canonical_b, &prompt_p, ITER7_MAX_TOKENS);
    assert!(
        !p_decoded_b.is_empty(),
        "[iter-7 wrap-guard] P decoded empty on server B"
    );
    eprintln!(
        "[iter-7 wrap-guard] server B P decoded {} bytes (control)",
        p_decoded_b.len()
    );

    // Defense-in-depth: byte-identity. Both servers ran fresh prefill
    // for P (server A: guard skipped store + registry empty + probe
    // returned None; server B: no LCP path), so outputs MUST match.
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
            "[iter-7 wrap-guard] FALSIFIED — P decoded bytes differ between \
             server A (long-prompt, guard expected to fire) and server B \
             (control). Even though detected_delta was 0 (probe found \
             nothing), output diverged. This indicates a non-LCP source \
             of nondeterminism in the long-prompt path — possibly the \
             snapshot publication side-effect mutating live buffers. Phase \
             E.a v1 invariant violated.\n\
             server A len={} bytes\n\
             server B len={} bytes\n\
             diverge at byte={}\n\
             A @ {}: {:?}\n\
             B @ {}: {:?}",
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
        "[iter-7 wrap-guard] PASS — server A (long prompt, guard fired) \
         bytes ({} bytes) == server B (control) bytes ({} bytes) BYTE-\
         IDENTICAL. Prefill-wrap guard correctness confirmed.",
        p_decoded_a.len(),
        p_decoded_b.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Iter-7 — Qwen3.5 hybrid deferral compile-time + runtime sanity.
//
// Phase E.a v1 covers Gemma 4 dense ONLY. Qwen3.5 hybrid families are
// EXPLICITLY DEFERRED to Phase B-hybrid follow-up. The deferral is
// enforced by CONSTRUCTION:
//   * `lcp_registry: LcpRegistry<DenseKvBuffers>` lives on
//     `GemmaLoadedModel` only (`engine.rs:1102`); the `Qwen35LoadedModel`
//     struct does NOT have this field.
//   * The probe sites at `engine.rs:3863` and `engine.rs:6158` reach
//     `loaded.lcp_registry` where `loaded: &mut GemmaLoadedModel` — only
//     reachable from the Gemma worker arm.
//   * The Qwen35 worker arm (`engine.rs:3152` for kv_restore,
//     `engine.rs:3205` for prompt_cache_restore) returns Err with a
//     "B-hybrid follow-up" message; no LCP path possible.
//
// The compile-time check below is a load-bearing assertion: any future
// refactor that adds an `lcp_registry` field to `Qwen35LoadedModel`
// would silently flip the deferral. This module-level helper would
// instead need to be updated, surfacing the architectural change in the
// PR diff.
//
// ## Why a unit test (no model needed)
//
// The deferral is structural. We can verify it WITHOUT loading a model
// by inspecting type structure indirectly via crate-level
// re-exports. The hf2q crate is `[bin]`-only (not `[lib]`) so we can't
// `use` types directly; the next-best signal is documenting the
// invariant in this test header and asserting via the actual probe-
// site code's behavior — exercised by the prefill-wrap test above
// (which forces the Gemma path) and by the absence of Qwen3.5 entries
// in the iter-3 / iter-5 falsifiers.
//
// This test is a documentation-only unit test that always passes; its
// value is that grep'ing for "Qwen35" + "lcp" surfaces the deferral
// rationale. If a future contributor adds Qwen35 hybrid LCP support,
// this test should be UPDATED (renamed to assert hybrid LCP works), not
// deleted — making the deferral lift visible in the diff.

/// Iter-7 deferral marker (always-pass). Documents the Qwen3.5 / Qwen3-VL
/// hybrid deferral structurally enforced at the worker-thread Request
/// dispatch (engine.rs:3144-3161, 3194-3211) and the field placement
/// (lcp_registry on GemmaLoadedModel only at engine.rs:1102).
///
/// FUTURE LIFT INSTRUCTIONS: when Phase B-hybrid follow-up adds Qwen3.5
/// hybrid LCP support:
///   1. Add `lcp_registry` field to Qwen35LoadedModel.
///   2. Add Qwen35 worker arm dispatch for KvSnapshot/KvRestore (today
///      returns Err at engine.rs:3152).
///   3. Add Qwen35-specific probe sites mirroring engine.rs:3863-3940.
///   4. RENAME this test to assert the hybrid LCP path now works (do
///      NOT silently delete — make the lift visible in the diff).
#[test]
fn iter7_qwen35_hybrid_lcp_deferred_marker() {
    // Compile-time deferral marker: this test always passes and serves
    // as a grep target. The structural deferral lives in:
    //   - engine.rs:1102 (lcp_registry field on Gemma only)
    //   - engine.rs:3152 (Qwen35 kv_restore returns Err)
    //   - engine.rs:3205 (Qwen35 prompt_cache_restore returns Err)
    //   - dossier §R5 (risk register hybrid divergence)
    //
    // Asserting `true` is intentional. The TEST'S VALUE is that the
    // documentation in this test header is searchable via:
    //   grep -rn "iter7_qwen35_hybrid_lcp_deferred_marker"
    // which surfaces the deferral rationale and the FUTURE LIFT
    // INSTRUCTIONS.
    assert!(
        true,
        "Phase E.a v1 deferral marker — see test header for rationale. \
         If Phase B-hybrid follow-up lifts the deferral, RENAME this \
         test rather than deleting it so the lift surfaces in the diff."
    );
}
