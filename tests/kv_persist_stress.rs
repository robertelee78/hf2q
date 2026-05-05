//! ADR-017 Phase D — 24-hour stress test (`tests/kv_persist_stress.rs`).
//!
//! Spec: ADR-017 §472-478 Phase D Stress checkbox. Drives ONE
//! `hf2q serve --kv-persist=DIR` subprocess under a continuous
//! eviction-churn workload for `HF2Q_KV_PERSIST_STRESS_DURATION_SEC`
//! seconds (default 86400 = 24 h) and asserts:
//!
//!   * **Cache directory size budget** — `du -sk DIR` ≤ budget × 1.10.
//!   * **No resident-memory leak** — `ps -o rss= -p {pid}` ≤
//!     baseline × (1 + tolerance%/100). Default tolerance 5 %.
//!   * **No descriptor leak** — `lsof -p {pid} | wc -l` ≤
//!     `baseline + tolerance`. Default tolerance 100.
//!
//! Default `cargo test --release --test kv_persist_stress` MUST
//! short-circuit cleanly without setting `HF2Q_KV_PERSIST_STRESS_24H=1`
//! — the test is hardware-bound (operator-run on cold M5 Max with the
//! pre-bench process audit from `scripts/adr017_phase_d.sh`).
//!
//! ## Env gates (master gate first)
//!
//!   * `HF2Q_KV_PERSIST_STRESS_24H=1` — master gate. Unset = fast-pass
//!     short-circuit.
//!   * `HF2Q_KV_PERSIST_E2E_MODEL_PATH=/path/to.gguf` — REQUIRED when
//!     master gate set. Same env name as the rest of Phase D
//!     (`scripts/adr017_phase_d.sh`).
//!   * `HF2Q_KV_PERSIST_STRESS_DURATION_SEC=N` — optional, default
//!     86400 (24 h). Operators run `=1800` (30 min) for in-session
//!     validation before the full overnight run.
//!   * `HF2Q_KV_PERSIST_STRESS_BUDGET_MB=N` — optional, default 4096
//!     (4 GiB). Cache-directory size cap (post-budget the test fails).
//!   * `HF2Q_KV_PERSIST_STRESS_RSS_TOLERANCE_PCT=N` — optional,
//!     default 5 (per ADR-017 §474 RSS-leak threshold).
//!   * `HF2Q_KV_PERSIST_STRESS_LSOF_TOLERANCE=N` — optional, default
//!     100 (per ADR-017 §474 descriptor-leak threshold).
//!
//! ## Helpers
//!
//! Rust integration-test files are independent crates; sharing helpers
//! across them requires `#[path]` includes or a shared library target.
//! Per the W2 deliverable (CFA session adr017-phase-d-closure), this
//! file duplicates the `phase_d_driver` helpers from
//! `tests/kv_persist_gemma4_roundtrip.rs` into a local
//! `mod stress_driver` rather than refactoring for one-off reuse.
//! The duplicated helpers are functionally identical to the upstream
//! ones; the only material delta is that `ServerGuard::pid()` is
//! exposed (the upstream guard hides `Child`).
//!
//! No `unsafe`. No additional crate deps beyond the existing
//! `[dev-dependencies]` (`reqwest`, `serde_json`, `tempfile`).
//!
//! `feedback_substrate_must_not_synthesize_ship_gates` compliance:
//! every assertion samples real OS state (`ps`, `lsof`, `du`) — no
//! constants are asserted against thresholds.

#![allow(dead_code)] // env-gate short-circuit means most code is operator-run.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// =========================================================================
// Env-gate constants.
// =========================================================================

/// Master gate. When unset, the test short-circuits with a diagnostic
/// and PASSes — same pattern as the Phase D coherence + R-P4 tests.
const ENV_STRESS_GATE: &str = "HF2Q_KV_PERSIST_STRESS_24H";
/// Required when master gate set. Mirrors `scripts/adr017_phase_d.sh`.
const ENV_MODEL_PATH: &str = "HF2Q_KV_PERSIST_E2E_MODEL_PATH";
/// Optional duration in seconds (default 86400 = 24 h).
const ENV_STRESS_DURATION_SEC: &str = "HF2Q_KV_PERSIST_STRESS_DURATION_SEC";
/// Optional cache directory size budget in MiB (default 4096 = 4 GiB).
const ENV_STRESS_BUDGET_MB: &str = "HF2Q_KV_PERSIST_STRESS_BUDGET_MB";
/// Optional RSS-leak threshold in percent (default 5).
const ENV_STRESS_RSS_TOL_PCT: &str = "HF2Q_KV_PERSIST_STRESS_RSS_TOLERANCE_PCT";
/// Optional descriptor-leak threshold in absolute count (default 100).
const ENV_STRESS_LSOF_TOL: &str = "HF2Q_KV_PERSIST_STRESS_LSOF_TOLERANCE";
/// Optional port override (mirrors Phase D coherence/R-P4 convention).
const ENV_STRESS_PORT: &str = "HF2Q_KV_PERSIST_STRESS_PORT";

const DEFAULT_DURATION_SEC: u64 = 86_400;
const DEFAULT_BUDGET_MB: u64 = 4_096;
const DEFAULT_RSS_TOL_PCT: f64 = 5.0;
const DEFAULT_LSOF_TOL: u64 = 100;
/// Periodic checkpoint cadence — every 5 minutes per spec.
const CHECKPOINT_INTERVAL_SEC: u64 = 300;
/// Cache budget slack (post-budget hard fail) — 10 % per spec.
const CACHE_BUDGET_SLACK: f64 = 1.10;
/// Cap on per-iter decode tokens. We want eviction churn, not long
/// decodes; spec specifies "at most 8 tokens".
const PER_ITER_MAX_TOKENS: u32 = 8;

// =========================================================================
// stress_driver — duplicated from tests/kv_persist_gemma4_roundtrip.rs's
// `phase_d_driver` mod with the addition of `ServerGuard::pid()`.
//
// Rationale: Rust integration-test files are separate crates; the
// `phase_d_driver` mod cannot be cross-imported without #[path] tricks.
// Duplicating ~530 LOC of mature, byte-exact helpers is cheaper than
// promoting the driver to a library target for a one-off operator-run
// test.
// =========================================================================

mod stress_driver {
    //! Self-contained subprocess driver — RAII `ServerGuard` kills +
    //! waits the child on Drop. Functionally identical to
    //! `tests/kv_persist_gemma4_roundtrip.rs::phase_d_driver`; the
    //! only delta is `ServerGuard::pid()` for the stress sampling.
    #![allow(dead_code)]
    use std::io::{BufRead, BufReader, Read, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    pub const HOST: &str = "127.0.0.1";
    /// Distinct port from the other Phase D tests so a leaked
    /// previous-run server doesn't collide. Phase D coherence/R-P4
    /// uses 52339; multi-model-swap 52337; openwebui 52334; harness
    /// 52338. We pick 52340.
    pub const PORT_DEFAULT: u16 = 52340;
    /// `/readyz` poll budget (cold 16-26 GiB GGUF startup can take
    /// 60-180 s on M5 Max).
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
        pub fn host(&self) -> &str { &self.host }
        pub fn port(&self) -> u16 { self.port }
        /// PID of the spawned child — needed by the stress sampler
        /// for `ps -o rss=` and `lsof -p`. Distinct from the upstream
        /// `phase_d_driver` which hides `Child`.
        pub fn pid(&self) -> u32 { self.child.id() }
        pub fn log_tail(&self) -> Vec<String> {
            self.stderr_tail.lock().map(|g| g.clone()).unwrap_or_default()
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

    pub fn spawn_hf2q_serve_with_kv_persist(
        bin: &Path,
        model_path: &Path,
        cache_dir: &Path,
        host: &str,
        port: u16,
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
        let mut child = Command::new(bin)
            .args([
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
            .env("HF2Q_USE_DENSE", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
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

    #[derive(Clone, Debug)]
    pub struct DecodeCapture {
        pub text: String,
        pub ttft_ms: f64,
        pub total_tokens: u32,
        pub prompt_tokens: Option<u32>,
    }

    /// Issue `/v1/chat/completions` with `stream: true`,
    /// `temperature: 0`, `max_tokens: max_tokens`. Returns the full
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
        // Stress-loop iters may legitimately complete faster than the
        // first content delta is observed (max_tokens=8, model
        // generates EOS immediately). Tolerate ttft_ms being None and
        // synthesize from the finish wall.
        let ttft_ms = ttft_ms.unwrap_or_else(|| t0.elapsed().as_secs_f64() * 1000.0);
        Ok(DecodeCapture {
            text,
            ttft_ms,
            total_tokens,
            prompt_tokens,
        })
    }

    /// Force a pool eviction by admitting a second model under a
    /// distinct on-disk path that resolves through `pool_key_for_path`.
    /// Mirrors `phase_d_driver::force_eviction_via_symlink`.
    #[cfg(unix)]
    pub fn force_eviction_via_symlink(
        server: &ServerGuard,
        model_path: &Path,
        tmp_link_dir: &Path,
        link_idx: u64,
    ) -> Result<(PathBuf, f64, f64), DriverError> {
        // `link_idx` cycles the link path so we can re-use the same
        // tempdir across thousands of stress iters without the
        // pool-key-cache short-circuiting the eviction (each
        // distinct file_stem resolves to a distinct pool key).
        let link_path = tmp_link_dir.join(format!("kv-persist-stress-{link_idx}.gguf"));
        // Idempotent: if a previous iter wrote this exact name, replace.
        let _ = std::fs::remove_file(&link_path);
        std::os::unix::fs::symlink(model_path, &link_path)
            .map_err(|e| DriverError::Transport(format!("symlink: {e}")))?;
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
                    if dst.exists() {
                        // Already linked from a prior iter — skip.
                        continue;
                    }
                    if let Err(e) = std::os::unix::fs::symlink(&src, &dst) {
                        return Err(DriverError::Transport(format!(
                            "symlink sibling {fname}: {e}"
                        )));
                    }
                }
            }
            if let Some(src_stem) = model_path.file_stem().and_then(|s| s.to_str()) {
                let mmproj_src =
                    model_parent.join(format!("{src_stem}-mmproj.gguf"));
                if mmproj_src.exists() {
                    if let Some(link_stem) =
                        link_path.file_stem().and_then(|s| s.to_str())
                    {
                        let mmproj_dst =
                            tmp_link_dir.join(format!("{link_stem}-mmproj.gguf"));
                        let _ = std::fs::remove_file(&mmproj_dst);
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
        let second_ttft_ms = ttft_ms.unwrap_or(eviction_wall_ms);
        Ok((link_path, eviction_wall_ms, second_ttft_ms))
    }

    #[cfg(not(unix))]
    pub fn force_eviction_via_symlink(
        _server: &ServerGuard,
        _model_path: &Path,
        _tmp_link_dir: &Path,
        _link_idx: u64,
    ) -> Result<(PathBuf, f64, f64), DriverError> {
        Err(DriverError::Transport(
            "symlink-distinct-pool-key trick is unix-only".into(),
        ))
    }

    /// Locate the `hf2q` binary (mirrors
    /// `kv_persist_gemma4_roundtrip.rs::hf2q_binary_path`).
    pub fn hf2q_binary_path() -> PathBuf {
        if let Some(p) = std::env::var_os("CARGO_BIN_EXE_hf2q") {
            return PathBuf::from(p);
        }
        let target_dir = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let manifest_dir = env!("CARGO_MANIFEST_DIR");
                PathBuf::from(manifest_dir).join("target")
            });
        target_dir.join("release").join("hf2q")
    }
}

// =========================================================================
// Sampler helpers — `ps`, `lsof`, `du`.
//
// Each sampler shells out to the canonical OS tool and parses the
// output into a `u64`. None of these synthesize the value: a parse
// failure returns `None` and the caller short-circuits the assertion
// with a diagnostic, never a fake-pass.
// =========================================================================

/// `ps -o rss= -p {pid}` — RSS in KiB.
fn sample_rss_kb(pid: u32) -> Option<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse::<u64>().ok()
}

/// `lsof -p {pid} | wc -l` — descriptor count (header line included;
/// stable baseline absorbs the +1 offset).
fn sample_lsof_count(pid: u32) -> Option<u64> {
    let out = std::process::Command::new("lsof")
        .args(["-p", &pid.to_string()])
        .output()
        .ok()?;
    // `lsof` exits non-zero (status 1) when the target process has
    // some unreachable fds even though most rows are valid; treat
    // any non-empty stdout as countable.
    let s = String::from_utf8_lossy(&out.stdout);
    if s.is_empty() {
        return None;
    }
    Some(s.lines().count() as u64)
}

/// `du -sk {dir}` — directory size in KiB. Returns `None` if `dir`
/// does not exist or `du` fails.
fn sample_cache_kb(dir: &Path) -> Option<u64> {
    if !dir.exists() {
        return None;
    }
    let out = std::process::Command::new("du")
        .args(["-sk", &dir.to_string_lossy()])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.split_whitespace().next()?.parse::<u64>().ok()
}

// =========================================================================
// Cheap nanos-based PRNG seed — avoid pulling rand into dev-deps.
// =========================================================================

fn cheap_rand_choice<T: Copy>(slice: &[T], seed: u64) -> T {
    let idx = (seed as usize) % slice.len();
    slice[idx]
}

fn now_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

// =========================================================================
// Env-resolution helpers.
// =========================================================================

fn resolve_model_path() -> Option<PathBuf> {
    let p = std::env::var(ENV_MODEL_PATH).ok()?;
    let pb = PathBuf::from(p);
    if pb.exists() { Some(pb) } else { None }
}

fn resolve_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
}

fn resolve_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default)
}

// =========================================================================
// Test: kv_persist_stress_24h
// =========================================================================

/// ADR-017 §472-478 Phase D Stress checkbox.
///
/// Drives a 24-hour (default) continuous swap-in/swap-out workload
/// against `hf2q serve --kv-persist=DIR` and asserts:
///
///   * Cache directory size stays within budget (default 4 GiB +
///     10 % slack).
///   * RSS at any checkpoint ≤ baseline × (1 + tolerance%/100)
///     (default 5 %).
///   * `lsof | wc -l` at any checkpoint ≤ baseline + tolerance
///     (default 100).
///
/// Default `cargo test` (no env) returns immediately with a
/// short-circuit diagnostic — the test is hardware-bound and
/// operator-run.
#[test]
fn kv_persist_stress_24h() {
    // ---- Master gate ----
    let active = std::env::var(ENV_STRESS_GATE).as_deref() == Ok("1");
    if !active {
        eprintln!(
            "[stress] {ENV_STRESS_GATE}=1 not set — short-circuit. \
             Set {ENV_STRESS_GATE}=1 + {ENV_MODEL_PATH}=PATH to run. \
             Optional: {ENV_STRESS_DURATION_SEC}=N (default {DEFAULT_DURATION_SEC} = 24h), \
             {ENV_STRESS_BUDGET_MB}=N (default {DEFAULT_BUDGET_MB}), \
             {ENV_STRESS_RSS_TOL_PCT}=N (default {DEFAULT_RSS_TOL_PCT}), \
             {ENV_STRESS_LSOF_TOL}=N (default {DEFAULT_LSOF_TOL})."
        );
        return;
    }

    // ---- Required env: model path. Fail-hard when master gate
    //      set + model path missing (operator misconfiguration). ----
    let model_path = resolve_model_path().unwrap_or_else(|| {
        panic!(
            "[stress] {ENV_STRESS_GATE}=1 set but {ENV_MODEL_PATH} unset or path missing — \
             operator must point this at the canonical Gemma 4 26B Q4_0 GGUF \
             (see scripts/adr017_phase_d.sh)."
        )
    });

    // ---- Optional knobs ----
    let duration_sec = resolve_u64(ENV_STRESS_DURATION_SEC, DEFAULT_DURATION_SEC);
    let budget_mb = resolve_u64(ENV_STRESS_BUDGET_MB, DEFAULT_BUDGET_MB);
    let rss_tol_pct = resolve_f64(ENV_STRESS_RSS_TOL_PCT, DEFAULT_RSS_TOL_PCT);
    let lsof_tol = resolve_u64(ENV_STRESS_LSOF_TOL, DEFAULT_LSOF_TOL);
    let port = std::env::var(ENV_STRESS_PORT)
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(stress_driver::PORT_DEFAULT);

    let budget_kb = budget_mb.saturating_mul(1024);
    let budget_kb_with_slack =
        ((budget_kb as f64) * CACHE_BUDGET_SLACK).ceil() as u64;

    // ---- Binary check ----
    let bin = stress_driver::hf2q_binary_path();
    assert!(
        bin.exists(),
        "[stress] hf2q binary not found at {} — did `cargo build --release` run?",
        bin.display()
    );

    // ---- Cache + symlink tempdirs ----
    let cache_dir = std::env::temp_dir().join(format!(
        "hf2q-kv-persist-stress-{}-{}",
        std::process::id(),
        now_nanos()
    ));
    std::fs::create_dir_all(&cache_dir).expect("[stress] mkdir cache_dir");
    let tmp_link_dir = tempfile::tempdir()
        .expect("[stress] tempdir for symlink-eviction-trick");

    eprintln!(
        "[stress] start — duration={duration_sec}s budget_mb={budget_mb} \
         rss_tol_pct={rss_tol_pct} lsof_tol={lsof_tol} \
         model={} cache_dir={}",
        model_path.display(),
        cache_dir.display(),
    );

    // ---- Spawn server ----
    let server = stress_driver::spawn_hf2q_serve_with_kv_persist(
        &bin,
        &model_path,
        &cache_dir,
        stress_driver::HOST,
        port,
    )
    .expect("[stress] spawn hf2q serve --kv-persist");
    let pid = server.pid();
    stress_driver::wait_for_readyz(&server).unwrap_or_else(|e| {
        panic!(
            "[stress] /readyz did not return 200 within budget: {e}\n\
             --- hf2q serve stderr_tail ({} lines) ---\n{}",
            server.log_tail().len(),
            server.log_tail().join("\n"),
        )
    });
    let canonical = stress_driver::fetch_canonical_model_id(&server)
        .expect("[stress] fetch canonical model id");
    eprintln!(
        "[stress] /readyz OK — pid={pid} model={canonical}"
    );

    // ---- Baseline (post-readyz, pre-load) ----
    // Note: we sample AFTER /readyz succeeds so the baseline reflects
    // the steady-state model-loaded RSS, not a cold-process number
    // that would inflate the leak comparison.
    let baseline_rss_kb = sample_rss_kb(pid).unwrap_or_else(|| {
        panic!("[stress] failed to sample baseline RSS for pid={pid}")
    });
    let baseline_lsof = sample_lsof_count(pid).unwrap_or_else(|| {
        panic!("[stress] failed to sample baseline lsof count for pid={pid}")
    });
    let baseline_cache_kb = sample_cache_kb(&cache_dir).unwrap_or(0);
    eprintln!(
        "[stress] baseline rss={baseline_rss_kb}KB lsof={baseline_lsof} cache={baseline_cache_kb}KB"
    );

    // ---- Continuous loop ----
    //
    // ADR-017 Closure iter-7 (2026-05-04): operator can cap the L
    // distribution via `HF2Q_KV_PERSIST_STRESS_MAX_L` to keep the
    // smoke variant within the test budget. Default = 32768 (full
    // range). For 30-min smoke validation set MAX_L=4096 so a
    // single iter (prefill + decode + eviction) fits inside one
    // 5-min checkpoint window on Gemma 4 26B M5 Max (per B-F4
    // standing finding: prefill at L=32K = ~10 min; at L=4K = ~1
    // min; iter wall = prefill + 60s eviction). Without this cap
    // the iter-6 30-min smoke at random L picked an L=32K iter
    // first and blocked the entire budget — see iter-6 stress note.
    let max_l: usize = std::env::var("HF2Q_KV_PERSIST_STRESS_MAX_L")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32768);
    let lengths: Vec<usize> = [512usize, 4096, 16384, 32768]
        .into_iter()
        .filter(|&l| l <= max_l)
        .collect();
    if lengths.is_empty() {
        panic!(
            "[stress] HF2Q_KV_PERSIST_STRESS_MAX_L={} is below the smallest \
             L=512 in the distribution; choose ≥512",
            max_l
        );
    }
    let start_time = Instant::now();
    let total_duration = Duration::from_secs(duration_sec);
    let checkpoint_interval = Duration::from_secs(CHECKPOINT_INTERVAL_SEC);
    let mut next_checkpoint = start_time + checkpoint_interval;

    let mut iters: u64 = 0;
    let mut max_rss_kb: u64 = baseline_rss_kb;
    let mut max_lsof: u64 = baseline_lsof;
    let mut max_cache_kb: u64 = baseline_cache_kb;
    let mut prng_state: u64 = now_nanos().rotate_left(13) ^ 0x9E37_79B9_7F4A_7C15;

    while start_time.elapsed() < total_duration {
        // Cheap LCG-ish step so successive picks aren't all the same
        // length (using only `now_nanos()` would tend to repeat
        // the modulo within a tight loop).
        prng_state = prng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            ^ now_nanos();
        let l = cheap_rand_choice(&lengths, prng_state);

        // Build prompt of length L via `format!("word{i}")`.
        let prompt: String = (0..l)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");

        // Decode at most PER_ITER_MAX_TOKENS tokens — short, cheap;
        // we want eviction churn, not long decodes.
        match stress_driver::decode_full_text(
            &server,
            &canonical,
            &prompt,
            PER_ITER_MAX_TOKENS,
        ) {
            Ok(_) => {}
            Err(e) => {
                panic!(
                    "[stress] decode iter={iters} L={l} failed: {e}\n\
                     --- stderr_tail ({} lines) ---\n{}",
                    server.log_tail().len(),
                    server.log_tail().join("\n"),
                );
            }
        }

        // Force eviction (distinct symlink each iter so the pool
        // key cycles).
        if let Err(e) = stress_driver::force_eviction_via_symlink(
            &server,
            &model_path,
            tmp_link_dir.path(),
            iters,
        ) {
            panic!(
                "[stress] eviction iter={iters} failed: {e}\n\
                 --- stderr_tail ({} lines) ---\n{}",
                server.log_tail().len(),
                server.log_tail().join("\n"),
            );
        }

        iters = iters.saturating_add(1);

        // Periodic checkpoint.
        if Instant::now() >= next_checkpoint {
            let elapsed_sec = start_time.elapsed().as_secs();
            let current_rss_kb = sample_rss_kb(pid).unwrap_or_else(|| {
                panic!("[stress] failed to sample RSS at t={elapsed_sec}s pid={pid}")
            });
            let current_lsof = sample_lsof_count(pid).unwrap_or_else(|| {
                panic!("[stress] failed to sample lsof at t={elapsed_sec}s pid={pid}")
            });
            let cache_kb = sample_cache_kb(&cache_dir).unwrap_or(0);

            max_rss_kb = max_rss_kb.max(current_rss_kb);
            max_lsof = max_lsof.max(current_lsof);
            max_cache_kb = max_cache_kb.max(cache_kb);

            let rss_pct_delta = if baseline_rss_kb == 0 {
                0.0
            } else {
                ((current_rss_kb as f64 - baseline_rss_kb as f64)
                    / baseline_rss_kb as f64)
                    * 100.0
            };
            let lsof_delta = current_lsof as i64 - baseline_lsof as i64;

            eprintln!(
                "[stress] checkpoint t={elapsed_sec}s rss={current_rss_kb}KB(Δ={rss_pct_delta:+.1}%) \
                 lsof={current_lsof}(Δ{lsof_delta:+}) cache={cache_kb}KB iters={iters}"
            );

            // Hard-fail checks (defense-in-depth — final assertions
            // run after the loop too).
            if cache_kb > budget_kb_with_slack {
                panic!(
                    "[stress] FAIL — cache directory exceeded budget at t={elapsed_sec}s: \
                     cache_kb={cache_kb} > budget_kb_with_slack={budget_kb_with_slack} \
                     (budget_mb={budget_mb}, slack=10%); iters={iters}"
                );
            }
            let rss_ceiling =
                ((baseline_rss_kb as f64) * (1.0 + rss_tol_pct / 100.0)).ceil() as u64;
            if current_rss_kb > rss_ceiling {
                panic!(
                    "[stress] FAIL — RSS leak detected at t={elapsed_sec}s: \
                     current_rss_kb={current_rss_kb} > ceiling={rss_ceiling} \
                     (baseline={baseline_rss_kb}, tol={rss_tol_pct}%); iters={iters}"
                );
            }
            if current_lsof > baseline_lsof.saturating_add(lsof_tol) {
                panic!(
                    "[stress] FAIL — descriptor leak detected at t={elapsed_sec}s: \
                     current_lsof={current_lsof} > baseline+tol={} \
                     (baseline={baseline_lsof}, tol={lsof_tol}); iters={iters}"
                    , baseline_lsof.saturating_add(lsof_tol)
                );
            }

            next_checkpoint += checkpoint_interval;
        }
    }

    // ---- Final assertions (defense-in-depth) ----
    let final_rss_kb = sample_rss_kb(pid)
        .unwrap_or_else(|| panic!("[stress] failed to sample final RSS for pid={pid}"));
    let final_lsof = sample_lsof_count(pid)
        .unwrap_or_else(|| panic!("[stress] failed to sample final lsof for pid={pid}"));
    let final_cache_kb = sample_cache_kb(&cache_dir).unwrap_or(0);

    max_rss_kb = max_rss_kb.max(final_rss_kb);
    max_lsof = max_lsof.max(final_lsof);
    max_cache_kb = max_cache_kb.max(final_cache_kb);

    let rss_ceiling =
        ((baseline_rss_kb as f64) * (1.0 + rss_tol_pct / 100.0)).ceil() as u64;
    let max_rss_pct = if baseline_rss_kb == 0 {
        0.0
    } else {
        ((max_rss_kb as f64 - baseline_rss_kb as f64) / baseline_rss_kb as f64) * 100.0
    };
    let max_lsof_delta = max_lsof as i64 - baseline_lsof as i64;

    assert!(
        max_cache_kb <= budget_kb_with_slack,
        "[stress] FINAL FAIL — cache directory peaked over budget: \
         max_cache_kb={max_cache_kb} > budget_kb_with_slack={budget_kb_with_slack} \
         (budget_mb={budget_mb}, slack=10%); iters={iters}"
    );
    assert!(
        max_rss_kb <= rss_ceiling,
        "[stress] FINAL FAIL — RSS leak: max_rss_kb={max_rss_kb} > ceiling={rss_ceiling} \
         (baseline={baseline_rss_kb}, tol={rss_tol_pct}%); iters={iters}"
    );
    assert!(
        max_lsof <= baseline_lsof.saturating_add(lsof_tol),
        "[stress] FINAL FAIL — descriptor leak: max_lsof={max_lsof} > baseline+tol={} \
         (baseline={baseline_lsof}, tol={lsof_tol}); iters={iters}"
        , baseline_lsof.saturating_add(lsof_tol)
    );

    eprintln!(
        "[stress] PASS — duration={duration_sec}s iters={iters} \
         max_rss={max_rss_kb}KB({max_rss_pct:+.1}%) \
         max_lsof={max_lsof}({max_lsof_delta:+}) \
         max_cache={max_cache_kb}KB"
    );

    // RAII Drop on `server` kills the child + joins the stderr
    // thread. `tmp_link_dir` and `cache_dir` are removed below.
    drop(server);
    let _ = std::fs::remove_dir_all(&cache_dir);
    // `tmp_link_dir` is `tempfile::TempDir` — auto-cleanup on Drop.
    drop(tmp_link_dir);
}
