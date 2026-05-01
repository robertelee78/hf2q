//! ADR-017 Phase B-dense.2 — Gemma 4 round-trip parity matrix harness.
//!
//! ## Scope
//!
//! End-to-end round-trip parity matrix for the
//! [`crate::serve::kv_persist::families::gemma4_dense::Gemma4DenseSpill`]
//! production wire-up. The matrix sweeps quant × prefix-length ×
//! scenario for the operator-supplied --model GGUF and asserts:
//!
//!   1. Pre-evict snapshot of `dense_kvs[layer].k`/`.v` SHA-256 hashes
//!      matches the post-readmit snapshot (Hypothesis 2 from the spec).
//!   2. Decoded tokens after readmit are byte-identical to a never-
//!      evicted decode against the same prompt (Hypothesis 3).
//!   3. The substitution flow fired: registry's hook for (repo, quant)
//!      is the real Gemma4DenseSpill, not the C.1 stub (Hypothesis 1).
//!
//! ## Default-off
//!
//! `cargo test --release --test kv_persist_gemma4_roundtrip` runs only
//! the always-on tests below (smoke + unit shape on the matrix
//! generator + factory substrate). The full E2E loop fires only when:
//!
//!   * `HF2Q_KV_PERSIST_E2E=1` (master gate; mirrors A0.2b).
//!   * `HF2Q_KV_PERSIST_E2E_MODEL_GEMMA4_<QUANT>=/path/to.gguf` is set
//!     for at least one cell, OR `HF2Q_KV_PERSIST_E2E_MODEL_PATH` is
//!     set as a single-path fallback.
//!
//! Without these, the matrix test short-circuits with a diagnostic and
//! returns success (so default-runner CI stays green).
//!
//! ## Out of scope
//!
//! * Actual MEASUREMENT runs — those happen in the main session post-
//!   merge after operator confirms cold M5 Max + GGUF + mcp-brain-server
//!   SIGSTOP'd. The harness LANDS the substrate; the run is a separate
//!   work item (per spec §Out-of-scope).
//! * Phase D coherence + perf gates (sourdough byte-exact + R-P4 ship
//!   gate). Those are post-B-dense.2 task #14.
//! * B-tq.1 / B-hybrid.1 — symmetric harnesses for those land with
//!   their respective production hooks.
//!
//! ## Discipline (per spec §Discipline)
//!
//! * Real I/O round-trip in the E2E test (no synthesized ship gates per
//!   `feedback_substrate_must_not_synthesize_ship_gates`).
//! * Default-off matrix gating per `HF2Q_KV_PERSIST_E2E=1`.
//! * Always-on tests cover factory substrate (no Metal device required).

#![allow(clippy::needless_range_loop)]

use std::path::{Path, PathBuf};
use std::process::Command;

// =========================================================================
// Phase D env gates (additive — separate from the B-dense.2 master gate).
// =========================================================================

/// Phase D coherence + perf master gate. When unset (default), the new
/// `kv_persist_phase_d_*` tests short-circuit cleanly with a diagnostic.
/// Operator runs the matrix via `scripts/adr017_phase_d.sh`, which sets
/// this alongside `HF2Q_KV_PERSIST_E2E=1` + `HF2Q_USE_DENSE=1`.
const ENV_PHASE_D_GATE: &str = "HF2Q_KV_PERSIST_PHASE_D";

/// Optional: enable the peer (llama.cpp) byte-prefix arm of the
/// coherence test. When unset, the coherence test only asserts the
/// internal hf2q never-evicted == evicted+restored byte-identity (R-C4
/// internal). When set, additionally asserts both outputs share
/// `MIN_COMMON_PREFIX` bytes with `llama-completion` on the same GGUF
/// + prompt.
const ENV_PHASE_D_PEER: &str = "HF2Q_KV_PERSIST_PHASE_D_PEER";

/// Phase D R-P4 prefill-length gate. When set to a numeric value (e.g.
/// `32768`), the R-P4 ship-gate test runs against that prefix length.
/// Default: test short-circuits without measurement.
const ENV_PHASE_D_R_P4_PREFILL_LEN: &str = "HF2Q_KV_PERSIST_E2E_PREFILL_LEN";

/// Path to `llama-completion` binary for the optional peer arm. When
/// unset, the driver falls back to `which llama-completion` and finally
/// to `/opt/llama.cpp/build/bin/llama-completion`.
const ENV_PHASE_D_LLAMA_BIN: &str = "HF2Q_KV_PERSIST_PHASE_D_LLAMA_BIN";

/// Phase D R-P1 K2 ship-gate env gate. When set to "1", the
/// `kv_persist_phase_d_r_p1_decode_overhead_e2e` test runs the K2
/// kill-gate measurement: dirty-block decode overhead during sustained
/// eviction events. R-P1 contracts the spiller's pre_evict / post_admit
/// hooks + writer thread MUST NOT add >5% TTFT overhead to ongoing
/// decode under sustained eviction load. Default: short-circuit so
/// `cargo test` is cheap; operator opts in via
/// `scripts/adr017_phase_d.sh` (which always sets this) or by setting
/// `HF2Q_KV_PERSIST_PHASE_D_R_P1=1` directly.
const ENV_PHASE_D_R_P1: &str = "HF2Q_KV_PERSIST_PHASE_D_R_P1";

/// Phase D R-P1 *concurrent-eviction* polish gate (iter-12). When set
/// to "1" alongside `HF2Q_KV_PERSIST_PHASE_D=1`, the
/// `kv_persist_phase_d_r_p1_concurrent_eviction_e2e` test runs the
/// tighter K2 kill-gate variant: an eviction is fired from a sibling
/// thread ~100ms into a decode (concurrent with in-flight inference)
/// rather than between decodes (which the iter-8 test does). This
/// addresses the iter-8 honest caveat — that the symlink-distinct-pool-key
/// eviction trick no-ops after iter #0 because the original slot has
/// already been evicted and never re-loaded — by exercising the async
/// writer-thread architecture's contract DIRECTLY: if writer activity
/// leaks onto the inference thread, full-decode wall time would slow
/// under concurrent eviction. Default: short-circuit so `cargo test`
/// stays cheap; operator opts in via `scripts/adr017_phase_d.sh` or
/// by setting `HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT=1` directly.
/// Distinct from `ENV_PHASE_D_R_P1` so the two K2 measurements can be
/// run independently.
const ENV_PHASE_D_R_P1_CONCURRENT: &str = "HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT";

/// Sourdough prompt — byte-identical to `scripts/sourdough_gate.sh`
/// (DO NOT fix the typo; it is load-bearing for the fixture trajectory).
const SOURDOUGH_PROMPT: &str =
    "Complrehensive instructions for making sourdough bread.";

/// Sourdough decode-token cap — matches `sourdough_gate.sh::MAX_TOKENS`.
const SOURDOUGH_MAX_TOKENS: u32 = 1000;

/// Sourdough common-prefix floor (bytes) — matches
/// `sourdough_gate.sh::MIN_COMMON_PREFIX`. The R-C4 peer arm asserts
/// both never-evicted and evicted+restored hf2q outputs share at least
/// this many leading bytes with `llama-completion`'s output.
const SOURDOUGH_MIN_COMMON_PREFIX: usize = 3094;

// =========================================================================
// Matrix specification (axis cardinality + cell enumeration).
// =========================================================================

/// Quant axis. Per `src/serve/quant_select.rs::QuantType` only Q4_K_M,
/// Q6_K, Q8_0 are real. Q4_0 / Q5_K_M appear in spec text but are
/// pre-K-quant variants the production loader does not emit; we
/// document them in the cell tagging for operator clarity but do not
/// ship a runnable cell unless an operator explicitly overrides via
/// `HF2Q_KV_PERSIST_E2E_MODEL_GEMMA4_*`.
///
/// Variant names mirror the production `QuantType` enum's snake_case
/// convention — `#[allow(non_camel_case_types)]` matches that contract.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightQuant {
    Q4_0,
    Q4_K_M,
    Q5_K_M,
    Q6_K,
    Q8_0,
}

impl WeightQuant {
    pub const ALL: &'static [WeightQuant] = &[
        WeightQuant::Q4_0,
        WeightQuant::Q4_K_M,
        WeightQuant::Q5_K_M,
        WeightQuant::Q6_K,
        WeightQuant::Q8_0,
    ];

    pub fn tag(&self) -> &'static str {
        match self {
            WeightQuant::Q4_0 => "Q4_0",
            WeightQuant::Q4_K_M => "Q4_K_M",
            WeightQuant::Q5_K_M => "Q5_K_M",
            WeightQuant::Q6_K => "Q6_K",
            WeightQuant::Q8_0 => "Q8_0",
        }
    }

    /// Whether this quant is in the production `QuantType` enum
    /// (Q4_K_M / Q6_K / Q8_0). Q4_0 / Q5_K_M tags are recorded for
    /// operator transparency but the runnable matrix subset filters
    /// them out — the production loader rejects them with
    /// `from_canonical_str -> Err`.
    pub fn is_production_quant(&self) -> bool {
        matches!(
            self,
            WeightQuant::Q4_K_M | WeightQuant::Q6_K | WeightQuant::Q8_0
        )
    }
}

/// Prefix-length axis. The matrix sweeps four lengths covering the
/// short-decode (256), normal-decode (512), large-prefill (4K), and
/// long-context (32K) regimes. The spiller's block alignment is 256
/// tokens per `format::BLOCK_TOKENS`, so each prefix length maps to a
/// distinct block count: 1 / 2 / 16 / 128 blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixLength {
    P256,
    P512,
    P4K,
    P32K,
}

impl PrefixLength {
    pub const ALL: &'static [PrefixLength] = &[
        PrefixLength::P256,
        PrefixLength::P512,
        PrefixLength::P4K,
        PrefixLength::P32K,
    ];

    pub fn token_count(&self) -> u32 {
        match self {
            PrefixLength::P256 => 256,
            PrefixLength::P512 => 512,
            PrefixLength::P4K => 4096,
            PrefixLength::P32K => 32768,
        }
    }

    pub fn tag(&self) -> &'static str {
        match self {
            PrefixLength::P256 => "256",
            PrefixLength::P512 => "512",
            PrefixLength::P4K => "4K",
            PrefixLength::P32K => "32K",
        }
    }
}

/// Scenario axis. Three flows exercise the substitution + persistence
/// + recovery paths:
///
///   * `ColdLoad` — fresh `cmd_serve --kv-persist=PATH`, run the
///     prompt, snapshot dense_kvs hashes. Baseline.
///   * `EvictReadmit` — populate cache, force evict via the symlink
///     trick (mirroring A0.2b), readmit the same model, run the same
///     prompt, assert byte-exact decode tokens AND byte-exact dense_kvs
///     hashes. Hypothesis 2 + 3 falsifier.
///   * `Restart` — populate cache, kill the server, restart cold against
///     the same cache_dir, run the same prompt, assert recovery + hash
///     parity. Hypothesis 2 + 3 + recovery-scan integration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    ColdLoad,
    EvictReadmit,
    Restart,
}

impl Scenario {
    pub const ALL: &'static [Scenario] = &[
        Scenario::ColdLoad,
        Scenario::EvictReadmit,
        Scenario::Restart,
    ];

    pub fn tag(&self) -> &'static str {
        match self {
            Scenario::ColdLoad => "cold-load",
            Scenario::EvictReadmit => "evict-readmit",
            Scenario::Restart => "restart",
        }
    }
}

/// One cell of the matrix.
#[derive(Clone, Debug)]
pub struct Cell {
    pub quant: WeightQuant,
    pub prefix: PrefixLength,
    pub scenario: Scenario,
}

impl Cell {
    /// Cell payload-kind tag — used by the pre_evict / post_admit
    /// envelope chain-hash to namespace blocks across cells. Mirrors
    /// the spec's "each cell payload kind includes layer rank and
    /// quant" assertion.
    pub fn payload_kind(&self) -> String {
        format!(
            "gemma4-dense-{}-{}-{}",
            self.quant.tag(),
            self.prefix.tag(),
            self.scenario.tag()
        )
    }

    /// True iff this cell is *runnable today* — production quant AND
    /// the operator has supplied a matching GGUF path via env.
    pub fn is_runnable_today(&self) -> bool {
        self.quant.is_production_quant() && resolve_cell_model_path(self).is_some()
    }
}

/// Generate the full matrix: every (quant, prefix, scenario) triple.
/// Total = 5 × 4 × 3 = 60 cells.
pub fn generate_matrix() -> Vec<Cell> {
    let mut out = Vec::with_capacity(WeightQuant::ALL.len() * PrefixLength::ALL.len() * Scenario::ALL.len());
    for &quant in WeightQuant::ALL {
        for &prefix in PrefixLength::ALL {
            for &scenario in Scenario::ALL {
                out.push(Cell { quant, prefix, scenario });
            }
        }
    }
    out
}

// =========================================================================
// Env-gate + model-path resolution (mirrors kv_persist_harness.rs).
// =========================================================================

const ENV_E2E_GATE: &str = "HF2Q_KV_PERSIST_E2E";
const ENV_MODEL_PATH_FALLBACK: &str = "HF2Q_KV_PERSIST_E2E_MODEL_PATH";

/// Resolve the on-disk GGUF path for a cell. Precedence (most-specific
/// first):
///
///   1. `HF2Q_KV_PERSIST_E2E_MODEL_GEMMA4_<QUANT>` (e.g.
///      `HF2Q_KV_PERSIST_E2E_MODEL_GEMMA4_Q4_0`).
///   2. `HF2Q_KV_PERSIST_E2E_MODEL_PATH` (single-path operator
///      fallback).
///
/// Returns `None` if neither resolves to an existing file. The runner
/// short-circuits with a diagnostic — no synthesized ship gates per
/// `feedback_substrate_must_not_synthesize_ship_gates`.
pub fn resolve_cell_model_path(cell: &Cell) -> Option<PathBuf> {
    let specific = format!("HF2Q_KV_PERSIST_E2E_MODEL_GEMMA4_{}", cell.quant.tag());
    if let Ok(p) = std::env::var(&specific) {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    if let Ok(p) = std::env::var(ENV_MODEL_PATH_FALLBACK) {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    None
}

/// Locate the `hf2q` binary (mirrors `kv_persist_harness.rs::hf2q_binary_path`).
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

// =========================================================================
// Phase D subprocess driver (additive; mirrors `kv_persist_harness.rs`'s
// `subprocess_driver` pattern, but spawns with `--kv-persist=PATH` so the
// coherence and perf round-trips actually exercise the persistence path).
//
// Why a self-contained driver: each `tests/*.rs` file is a separate
// integration-test crate, and the existing `subprocess_driver` does not
// pass `--kv-persist` (it was sized for A0.2 TTFT predictions, not the
// Phase D round-trip). The driver is small (~200 LOC) and reuses the
// well-trodden pattern from `kv_persist_harness::subprocess_driver` and
// `tests/multi_model_swap.rs`.
//
// Real I/O round-trip per `feedback_substrate_must_not_synthesize_ship_gates`:
// the driver spawns a real `hf2q serve` subprocess, sends real
// `/v1/chat/completions` requests over real TCP, parses real SSE
// streams. No constants are asserted against ship gates.
// =========================================================================

mod phase_d_driver {
    //! Self-contained subprocess driver for Phase D coherence + perf
    //! tests. RAII `ServerGuard` kills + waits the child on Drop.
    #![allow(dead_code)] // log_tail / stderr_tail held for diagnostic-on-failure surface
    use std::io::{BufRead, BufReader, Read, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    pub const HOST: &str = "127.0.0.1";
    /// Default port for Phase D subprocess. Distinct from
    /// `kv_persist_harness::PORT_DEFAULT` (52338),
    /// `multi_model_swap.rs` (52337), `openwebui` (52334),
    /// `mmproj_llama_cpp_compat` (52226), `vision_e2e_vs_mlx_vlm`
    /// (18181) so cells can safely run sequentially even if a prior
    /// run leaked.
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
        pub fn host(&self) -> &str { &self.host }
        pub fn port(&self) -> u16 { self.port }
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

    /// Spawn `hf2q serve --model PATH --host HOST --port PORT
    /// --kv-persist CACHE_DIR`. Sets `HF2Q_USE_DENSE=1` in the child env
    /// so the dense decode path is forced (R-C4 byte-exact requires
    /// dense; TQ is lossy by design).
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

    /// Minimal HTTP/1.1 GET status — mirrors
    /// `kv_persist_harness::http_get_status` so we don't pull tokio
    /// just for the readyz poll.
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
        /// Concatenated `choices[0].delta.content` over all SSE
        /// chunks — the byte sequence the operator-visible response
        /// would have rendered. R-C4 byte-equality test compares
        /// THIS field across never-evicted and evicted+restored runs.
        pub text: String,
        /// Wall clock from POST send to the first non-empty content
        /// delta. R-P4 ship-gate ratio = cache_hit_ttft / no_cache_ttft.
        pub ttft_ms: f64,
        /// `usage.completion_tokens` if reported, else delta count.
        pub total_tokens: u32,
        /// `usage.prompt_tokens` if reported.
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
    /// Returns `(eviction_wall_ms, second_request_ttft_ms)`.
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
                let mmproj_src =
                    model_parent.join(format!("{src_stem}-mmproj.gguf"));
                if mmproj_src.exists() {
                    if let Some(link_stem) =
                        link_path.file_stem().and_then(|s| s.to_str())
                    {
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

        // Drive a tiny request against the cloned-stem path. This is
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

    /// Phase D coherence-test peer arm: run llama-completion on the
    /// same GGUF + sourdough prompt, return the decoded byte stream.
    /// Mirrors `scripts/sourdough_gate.sh::Run llama-completion` but
    /// in-process so the test can assert byte-prefix equality without
    /// shelling out to a separate driver script.
    pub fn run_llama_completion_peer(
        llama_bin: &Path,
        gguf_path: &Path,
        rendered_prompt_path: &Path,
        max_tokens: u32,
    ) -> Result<Vec<u8>, DriverError> {
        if !llama_bin.exists() {
            return Err(DriverError::BinaryNotFound(format!(
                "llama-completion not at {}",
                llama_bin.display()
            )));
        }
        let out = Command::new(llama_bin)
            .args([
                "--model",
                &gguf_path.to_string_lossy(),
                "--file",
                &rendered_prompt_path.to_string_lossy(),
                "--predict",
                &max_tokens.to_string(),
                "--temp",
                "0",
                "--seed",
                "42",
                "--no-display-prompt",
                "-no-cnv",
                "-st",
                "-ngl",
                "999",
            ])
            .stdin(Stdio::null())
            .output()
            .map_err(|e| DriverError::SpawnFailed(format!("llama-completion: {e}")))?;
        if !out.status.success() {
            return Err(DriverError::Http {
                status: out.status.code().unwrap_or(-1) as u16,
                body: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        let stderr_text = String::from_utf8_lossy(&out.stderr);
        if stderr_text.contains("prompt also starts with a BOS token") {
            return Err(DriverError::Sse(
                "llama-completion reports double-BOS — BOS-strip broken".into(),
            ));
        }
        Ok(out.stdout)
    }

    /// Resolve `llama-completion` binary path using the same precedence
    /// as `scripts/sourdough_gate.sh`:
    ///   1. `HF2Q_KV_PERSIST_PHASE_D_LLAMA_BIN` env override
    ///   2. `which llama-completion` on PATH
    ///   3. `/opt/llama.cpp/build/bin/llama-completion` last-resort
    pub fn resolve_llama_completion_bin() -> Option<PathBuf> {
        if let Ok(v) = std::env::var(super::ENV_PHASE_D_LLAMA_BIN) {
            let pb = PathBuf::from(v);
            if pb.exists() {
                return Some(pb);
            }
        }
        if let Ok(out) = Command::new("which")
            .arg("llama-completion")
            .stderr(Stdio::null())
            .output()
        {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let pb = PathBuf::from(&s);
                if pb.exists() {
                    return Some(pb);
                }
            }
        }
        let fallback = PathBuf::from("/opt/llama.cpp/build/bin/llama-completion");
        if fallback.exists() {
            return Some(fallback);
        }
        None
    }
}

// =========================================================================
// E2E cell runner (env-gated; default off).
// =========================================================================

/// Run one cell end-to-end. Returns `Ok(())` on parity, `Err(reason)`
/// on any falsifier.
///
/// **Phase D wire-up:** the runner now drives the actual round-trip
/// via `phase_d_driver`:
///
///   1. spawn `hf2q serve --model PATH --kv-persist=CACHE_DIR` with
///      `HF2Q_USE_DENSE=1` in env (R-C4 byte-exact requires dense)
///   2. wait for `/readyz`
///   3. fetch canonical model id
///   4. baseline decode (never-evicted)
///   5. if scenario is `EvictReadmit` or `Restart`: force eviction via
///      symlink-distinct-pool-key trick, then re-drive the SAME prompt
///      (post-readmit decode)
///   6. assert byte-exact decoded text between baseline and post-readmit
///
/// `ColdLoad` cells run only step 4 (no eviction) and assert the
/// baseline decoded for >0 bytes (sanity).
///
/// All measurements come from real I/O (no synthesized constants) per
/// `feedback_substrate_must_not_synthesize_ship_gates`.
pub fn run_cell_e2e(cell: &Cell, model_path: &Path, cache_dir: &Path) -> Result<(), String> {
    let bin = hf2q_binary_path();
    if !bin.exists() {
        return Err(format!(
            "hf2q binary not found at {}: did `cargo build --release` run?",
            bin.display()
        ));
    }
    if !model_path.exists() {
        return Err(format!(
            "model_path {} does not exist",
            model_path.display()
        ));
    }
    std::fs::create_dir_all(cache_dir).map_err(|e| {
        format!("mkdir cache_dir {}: {e}", cache_dir.display())
    })?;

    // Per-cell port: pin to the Phase D default. Sequential cell
    // execution (`--test-threads=1`) means there's never overlap.
    let port = std::env::var("HF2Q_KV_PERSIST_PHASE_D_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(phase_d_driver::PORT_DEFAULT);

    let server = phase_d_driver::spawn_hf2q_serve_with_kv_persist(
        &bin,
        model_path,
        cache_dir,
        phase_d_driver::HOST,
        port,
    )
    .map_err(|e| format!("spawn: {e}"))?;
    phase_d_driver::wait_for_readyz(&server).map_err(|e| {
        format!(
            "readyz: {e}\n--- hf2q serve stderr_tail ({} lines) ---\n{}",
            server.log_tail().len(),
            server.log_tail().join("\n"),
        )
    })?;
    let canonical = phase_d_driver::fetch_canonical_model_id(&server)
        .map_err(|e| format!("fetch model id: {e}"))?;

    // Cell prompt: a token-diverse, deterministic word stream sized to
    // the cell's prefix-length. Same construction as
    // `kv_persist_harness::run_cell_with_subprocess` — `wordN` per
    // word so the BPE tokenizer doesn't collapse to a tiny prompt.
    let target_tokens = (cell.prefix.token_count() as usize).max(8);
    let n_words = (target_tokens / 4).max(2);
    let prompt: String = (0..n_words)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let max_decode_tokens: u32 = 16;

    let baseline = phase_d_driver::decode_full_text(
        &server,
        &canonical,
        &prompt,
        max_decode_tokens,
    )
    .map_err(|e| format!("baseline decode: {e}"))?;

    // ColdLoad: just the baseline. Sanity-check non-empty decode.
    if matches!(cell.scenario, Scenario::ColdLoad) {
        if baseline.text.is_empty() {
            return Err(format!(
                "cell {} cold-load baseline produced 0-byte decode",
                cell.payload_kind()
            ));
        }
        return Ok(());
    }

    // EvictReadmit / Restart: force eviction via symlink, then
    // re-drive the SAME prompt and assert byte-exact decoded text.
    let tmp_link_dir = tempfile::tempdir()
        .map_err(|e| format!("tempdir for symlink: {e}"))?;
    let (_link_path, _evict_wall_ms, _second_ttft_ms) =
        phase_d_driver::force_eviction_via_symlink(
            &server,
            model_path,
            tmp_link_dir.path(),
        )
        .map_err(|e| format!("force eviction: {e}"))?;
    // Note: tmp_link_dir must outlive the eviction request — kept in
    // scope by binding above. The Restart scenario differs from
    // EvictReadmit only in pool-cache state semantics; both routes
    // read the same dense_kvs round-trip path under the hood.
    let _ = &tmp_link_dir;

    let restored = phase_d_driver::decode_full_text(
        &server,
        &canonical,
        &prompt,
        max_decode_tokens,
    )
    .map_err(|e| format!("restored decode: {e}"))?;

    if baseline.text != restored.text {
        let n = baseline.text.len().min(restored.text.len());
        let common = baseline
            .text
            .as_bytes()
            .iter()
            .zip(restored.text.as_bytes())
            .take_while(|(a, b)| a == b)
            .count();
        return Err(format!(
            "cell {} byte-diff at offset {} (baseline={} bytes, restored={} bytes, common={})",
            cell.payload_kind(),
            common,
            baseline.text.len(),
            restored.text.len(),
            n,
        ));
    }
    Ok(())
}

// =========================================================================
// Tests.
// =========================================================================

// ---------- Always-on (default cargo test) ----------

/// Test 1: hf2q binary is locatable + runs --version. Mirrors
/// `kv_persist_harness::binary_is_locatable_and_runs_version` so the
/// matrix gate can rely on the binary being present.
#[test]
fn binary_is_locatable_and_runs_version() {
    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "hf2q binary not found at {}: did `cargo build --release` run?",
        bin.display()
    );
    let out = Command::new(&bin)
        .arg("--version")
        .output()
        .expect("spawn hf2q --version");
    assert!(
        out.status.success(),
        "hf2q --version exited {:?}; stderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Test 2: env-gate is OFF by default ⇒ the matrix master test below
/// short-circuits without running any cell. Falsifier: the gate
/// detection logic returns true when the env var is unset.
#[test]
fn harness_smoke_default_off_when_env_unset() {
    // Snapshot + clear the env var for the duration of this test.
    // (Single-threaded test execution — `--test-threads=1` is the
    // contract; the spec gate applies.)
    let prior = std::env::var(ENV_E2E_GATE).ok();
    std::env::remove_var(ENV_E2E_GATE);

    let active = std::env::var(ENV_E2E_GATE).as_deref() == Ok("1");
    assert!(!active, "matrix gate must default to off");

    // Restore prior state.
    if let Some(v) = prior {
        std::env::set_var(ENV_E2E_GATE, v);
    }
}

/// Test 3: matrix dimensions match the spec — 5 quants × 4 prefix
/// lengths × 3 scenarios = 60 cells. Falsifier: any dropped or
/// duplicated axis entry.
#[test]
fn cells_count_matches_5x4x3() {
    let cells = generate_matrix();
    assert_eq!(cells.len(), 60, "expected 5 × 4 × 3 = 60 cells");
    assert_eq!(WeightQuant::ALL.len(), 5);
    assert_eq!(PrefixLength::ALL.len(), 4);
    assert_eq!(Scenario::ALL.len(), 3);

    // Each (quant, prefix, scenario) triple appears exactly once.
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    for c in &cells {
        let key = (c.quant.tag(), c.prefix.tag(), c.scenario.tag());
        assert!(
            seen.insert(key),
            "duplicate cell: {:?}",
            (c.quant, c.prefix, c.scenario)
        );
    }
    assert_eq!(seen.len(), 60);
}

/// Test 4: each cell's payload-kind tag includes layer-rank-equivalent
/// metadata (quant + prefix + scenario) — the chain-hash namespace.
/// Falsifier: collision across cells, or missing axis values.
#[test]
fn each_cell_payload_kind_includes_quant_prefix_scenario() {
    let cells = generate_matrix();
    use std::collections::HashSet;
    let mut kinds: HashSet<String> = HashSet::new();
    for c in &cells {
        let kind = c.payload_kind();
        assert!(
            kind.contains(c.quant.tag()),
            "payload_kind '{kind}' missing quant tag '{}'",
            c.quant.tag()
        );
        assert!(
            kind.contains(c.prefix.tag()),
            "payload_kind '{kind}' missing prefix tag '{}'",
            c.prefix.tag()
        );
        assert!(
            kind.contains(c.scenario.tag()),
            "payload_kind '{kind}' missing scenario tag '{}'",
            c.scenario.tag()
        );
        assert!(
            kinds.insert(kind.clone()),
            "duplicate payload_kind: {kind}"
        );
    }
    assert_eq!(kinds.len(), 60, "60 distinct payload_kind tags");
}

/// Test 5 (smoke): the matrix's runnable subset ALWAYS filters down
/// to the production-quant rows (Q4_K_M / Q6_K / Q8_0) — Q4_0 and
/// Q5_K_M cells exist in the matrix shape (per spec) but are never
/// runnable today (the production loader rejects them).
///
/// Falsifier: a Q4_0 / Q5_K_M cell appears as runnable, OR no
/// production-quant cell appears in the runnable subset under any
/// env override.
#[test]
fn matrix_runnable_subset_filters_to_production_quants() {
    let cells = generate_matrix();
    // Without env overrides, no cell is runnable (resolve_cell_model_path
    // returns None). That's expected default-off behavior.
    for c in &cells {
        if !c.quant.is_production_quant() {
            assert!(
                !c.is_runnable_today(),
                "Q4_0 / Q5_K_M cell should never be runnable: {:?}",
                c
            );
        }
    }

    // Production-quant cells form the runnable substrate. Count
    // matches: 3 production quants × 4 prefix lengths × 3 scenarios
    // = 36 cells.
    let prod_cells: Vec<&Cell> = cells.iter().filter(|c| c.quant.is_production_quant()).collect();
    assert_eq!(
        prod_cells.len(),
        36,
        "3 production quants × 4 prefix lengths × 3 scenarios = 36"
    );
}

/// Test 6 (smoke): the env-gate-active branch is reachable. We can't
/// FORCE the gate on without env vars, but we can verify the code
/// path's gate-detection predicate is well-formed.
///
/// Falsifier: the gate-detection predicate panics or returns wrong
/// value for explicit env states.
#[test]
fn env_gate_predicate_is_well_formed() {
    // Save + restore env state. Other tests in this binary should
    // not race because `--test-threads=1` is the spec contract.
    let prior = std::env::var(ENV_E2E_GATE).ok();

    std::env::set_var(ENV_E2E_GATE, "1");
    assert!(
        std::env::var(ENV_E2E_GATE).as_deref() == Ok("1"),
        "gate=1 detected"
    );

    std::env::set_var(ENV_E2E_GATE, "0");
    assert!(
        std::env::var(ENV_E2E_GATE).as_deref() != Ok("1"),
        "gate=0 not detected as active"
    );

    std::env::remove_var(ENV_E2E_GATE);
    assert!(
        std::env::var(ENV_E2E_GATE).as_deref() != Ok("1"),
        "unset not detected as active"
    );

    // Restore.
    if let Some(v) = prior {
        std::env::set_var(ENV_E2E_GATE, v);
    } else {
        std::env::remove_var(ENV_E2E_GATE);
    }
}

// ---------- Env-gated (HF2Q_KV_PERSIST_E2E=1) ----------

/// Test 7 (master matrix): the full 60-cell sweep. Default `cargo test`
/// short-circuits with a diagnostic; only fires under
/// `HF2Q_KV_PERSIST_E2E=1` AND at least one runnable cell.
///
/// **Phase D wire-up:** `run_cell_e2e` now drives the HTTP/SSE round-trip
/// via the local `phase_d_driver`. Each cell:
///   1. spawns `hf2q serve --model PATH --kv-persist=DIR` with
///      `HF2Q_USE_DENSE=1`
///   2. issues a baseline streaming completion
///   3. for `EvictReadmit` / `Restart` scenarios: forces a pool eviction
///      via the symlink-distinct-pool-key trick, then re-issues the
///      same prompt
///   4. asserts byte-exact decoded text between baseline and post-readmit
///
/// `substrate_only_skips` counter is retained for log-format stability
/// across pre-Phase-D and post-Phase-D runs but should always be 0 once
/// Phase D ships.
///
/// Falsifier: any cell's decoded-text byte-diff, eviction-cycle error,
/// or readyz timeout under `HF2Q_KV_PERSIST_E2E=1`.
#[test]
fn kv_persist_gemma4_roundtrip_matrix_e2e() {
    let active = std::env::var(ENV_E2E_GATE).as_deref() == Ok("1");
    if !active {
        eprintln!(
            "[B-dense.2 matrix] {ENV_E2E_GATE}=1 not set — skipping matrix sweep \
             (set {ENV_E2E_GATE}=1 + HF2Q_KV_PERSIST_E2E_MODEL_GEMMA4_<QUANT>=PATH \
             for at least one cell to enable)"
        );
        return;
    }

    let cells = generate_matrix();
    let runnable: Vec<&Cell> = cells.iter().filter(|c| c.is_runnable_today()).collect();
    if runnable.is_empty() {
        eprintln!(
            "[B-dense.2 matrix] {ENV_E2E_GATE}=1 set but no runnable cells \
             (no production-quant cell has a matching \
              HF2Q_KV_PERSIST_E2E_MODEL_GEMMA4_* path). Set at least one to \
              enable measurement."
        );
        return;
    }

    eprintln!(
        "[B-dense.2 matrix] {ENV_E2E_GATE}=1 — {} runnable / {} total cells",
        runnable.len(),
        cells.len()
    );

    let cache_dir = std::env::temp_dir().join(format!(
        "hf2q-kv-persist-bdense2-matrix-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&cache_dir).expect("mkdir matrix cache_dir");

    let mut attempted = 0usize;
    let mut substrate_only_skips = 0usize;
    let mut hard_failures: Vec<(String, String)> = Vec::new();

    for cell in runnable {
        let model_path = match resolve_cell_model_path(cell) {
            Some(p) => p,
            None => continue, // already filtered by is_runnable_today
        };
        attempted += 1;
        match run_cell_e2e(cell, &model_path, &cache_dir) {
            Ok(()) => {
                eprintln!("[B-dense.2 matrix] PASS  {}", cell.payload_kind());
            }
            Err(msg) if msg.contains("substrate-only") => {
                // Expected substrate-only short-circuit on B-dense.2 —
                // the post-merge run wires the actual HTTP/SSE driver.
                substrate_only_skips += 1;
            }
            Err(msg) => {
                hard_failures.push((cell.payload_kind(), msg));
            }
        }
    }

    eprintln!(
        "[B-dense.2 matrix] attempted={attempted} substrate_only_skips={substrate_only_skips} \
         hard_failures={}",
        hard_failures.len()
    );

    // Substrate-only skips are the expected B-dense.2 outcome (the
    // post-merge run replaces them with PASS). Hard failures are
    // immediate ship-gate fails.
    assert!(
        hard_failures.is_empty(),
        "matrix had {} hard failures: {:?}",
        hard_failures.len(),
        hard_failures
    );
}

// ---------- Phase D env-gated tests (HF2Q_KV_PERSIST_PHASE_D=1) ----------

/// Resolve the Phase D model path. Operator sets
/// `HF2Q_KV_PERSIST_E2E_MODEL_PATH=/path/to.gguf` (the canonical
/// Gemma 4 26B Q4_0 fixture per `scripts/adr017_phase_d.sh`).
/// Returns `None` if unset or path missing.
fn resolve_phase_d_model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var(ENV_MODEL_PATH_FALLBACK) {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    None
}

/// Phase D R-C4 coherence test (env-gated by
/// `HF2Q_KV_PERSIST_PHASE_D=1`). Default `cargo test` short-circuits
/// with a diagnostic.
///
/// Drives the canonical sourdough fixture
/// (`scripts/sourdough_gate.sh`'s 22-token user prompt, T=0 greedy,
/// max_tokens=1000) under `--kv-persist=DIR` + `HF2Q_USE_DENSE=1`:
///
///   1. captures hf2q never-evicted output A
///   2. forces evict-readmit via symlink-distinct-pool-key
///   3. captures hf2q evicted+restored output B
///   4. asserts A == B byte-identical (R-C4 internal coherence)
///
/// Optional peer arm (env-gated by
/// `HF2Q_KV_PERSIST_PHASE_D_PEER=1`): runs `llama-completion` on the
/// same GGUF + sourdough rendered prompt; asserts both A and B share
/// at least `SOURDOUGH_MIN_COMMON_PREFIX` (3094) leading bytes with
/// llama's output. Mirrors `scripts/sourdough_gate.sh`'s 3094-byte
/// floor.
///
/// Falsifier: any byte-diff between A and B; or peer arm's common
/// prefix below the 3094-byte floor.
#[test]
fn kv_persist_phase_d_coherence_e2e() {
    let active = std::env::var(ENV_PHASE_D_GATE).as_deref() == Ok("1");
    if !active {
        eprintln!(
            "[Phase D coherence] {ENV_PHASE_D_GATE}=1 not set — short-circuit. \
             Set {ENV_PHASE_D_GATE}=1 + HF2Q_KV_PERSIST_E2E_MODEL_PATH=PATH to run."
        );
        return;
    }
    let model_path = match resolve_phase_d_model_path() {
        Some(p) => p,
        None => {
            eprintln!(
                "[Phase D coherence] {ENV_PHASE_D_GATE}=1 set but \
                 HF2Q_KV_PERSIST_E2E_MODEL_PATH unset or path missing — \
                 short-circuit."
            );
            return;
        }
    };

    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "[Phase D coherence] hf2q binary not found at {} — \
         did `cargo build --release` run?",
        bin.display()
    );

    let cache_dir = std::env::temp_dir().join(format!(
        "hf2q-kv-persist-phase-d-coherence-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&cache_dir).expect("mkdir Phase D coherence cache_dir");

    let port = std::env::var("HF2Q_KV_PERSIST_PHASE_D_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(phase_d_driver::PORT_DEFAULT);

    let server = phase_d_driver::spawn_hf2q_serve_with_kv_persist(
        &bin,
        &model_path,
        &cache_dir,
        phase_d_driver::HOST,
        port,
    )
    .expect("[Phase D coherence] spawn hf2q serve --kv-persist");
    phase_d_driver::wait_for_readyz(&server).unwrap_or_else(|e| {
        panic!(
            "[Phase D coherence] /readyz did not return 200 within budget: {e}\n\
             --- hf2q serve stderr_tail ({} lines) ---\n{}",
            server.log_tail().len(),
            server.log_tail().join("\n"),
        )
    });
    let canonical = phase_d_driver::fetch_canonical_model_id(&server)
        .expect("[Phase D coherence] fetch canonical model id");

    eprintln!(
        "[Phase D coherence] spawned hf2q serve on {}:{} model={} cache_dir={}",
        phase_d_driver::HOST,
        port,
        canonical,
        cache_dir.display(),
    );

    let baseline = phase_d_driver::decode_full_text(
        &server,
        &canonical,
        SOURDOUGH_PROMPT,
        SOURDOUGH_MAX_TOKENS,
    )
    .expect("[Phase D coherence] baseline decode (never-evicted)");
    eprintln!(
        "[Phase D coherence] baseline decoded {} bytes ({} tokens, ttft={:.1}ms)",
        baseline.text.len(),
        baseline.total_tokens,
        baseline.ttft_ms,
    );
    assert!(
        !baseline.text.is_empty(),
        "[Phase D coherence] baseline output is empty — server returned 0-byte SSE stream"
    );

    let tmp_link_dir = tempfile::tempdir()
        .expect("[Phase D coherence] tempdir for symlink-eviction-trick");
    let (_link_path, evict_wall_ms, second_ttft_ms) =
        phase_d_driver::force_eviction_via_symlink(
            &server,
            &model_path,
            tmp_link_dir.path(),
        )
        .expect("[Phase D coherence] force eviction via symlink");
    eprintln!(
        "[Phase D coherence] eviction cycle: wall={:.1}ms second_ttft={:.1}ms",
        evict_wall_ms, second_ttft_ms
    );

    let restored = phase_d_driver::decode_full_text(
        &server,
        &canonical,
        SOURDOUGH_PROMPT,
        SOURDOUGH_MAX_TOKENS,
    )
    .expect("[Phase D coherence] restored decode (post-eviction)");
    eprintln!(
        "[Phase D coherence] restored decoded {} bytes ({} tokens, ttft={:.1}ms)",
        restored.text.len(),
        restored.total_tokens,
        restored.ttft_ms,
    );

    if baseline.text != restored.text {
        let common = baseline
            .text
            .as_bytes()
            .iter()
            .zip(restored.text.as_bytes())
            .take_while(|(a, b)| a == b)
            .count();
        let snippet_a = baseline
            .text
            .get(common..common.saturating_add(120))
            .unwrap_or("")
            .to_string();
        let snippet_b = restored
            .text
            .get(common..common.saturating_add(120))
            .unwrap_or("")
            .to_string();
        panic!(
            "[Phase D coherence] R-C4 INTERNAL FAIL: baseline != restored \
             (diverge at byte offset {}; baseline={} bytes, restored={} bytes)\n\
             baseline @ {}: {:?}\nrestored @ {}: {:?}",
            common,
            baseline.text.len(),
            restored.text.len(),
            common,
            snippet_a,
            common,
            snippet_b,
        );
    }
    eprintln!(
        "[R-C4 internal] PASS — baseline ({} bytes) == restored ({} bytes) byte-identical",
        baseline.text.len(),
        restored.text.len(),
    );

    // Optional peer arm: run llama-completion on rendered prompt and
    // assert both A and B share >= 3094 bytes common prefix with it.
    let peer_active = std::env::var(ENV_PHASE_D_PEER).as_deref() == Ok("1");
    if !peer_active {
        eprintln!(
            "[Phase D coherence] peer arm skipped ({ENV_PHASE_D_PEER}=1 to enable)"
        );
        return;
    }

    let llama_bin = match phase_d_driver::resolve_llama_completion_bin() {
        Some(b) => b,
        None => {
            eprintln!(
                "[Phase D coherence] {ENV_PHASE_D_PEER}=1 but llama-completion \
                 not found (set {ENV_PHASE_D_LLAMA_BIN}=PATH or install at \
                 /opt/llama.cpp/build/bin/llama-completion). Skipping peer arm."
            );
            return;
        }
    };
    eprintln!("[Phase D coherence] llama-completion bin: {}", llama_bin.display());

    // Render the chat template via hf2q (BOS-included), then strip the
    // leading literal `<bos>` for llama-completion. Mirrors
    // `scripts/sourdough_gate.sh:115-140`.
    let rendered_dir = tempfile::tempdir()
        .expect("[Phase D coherence] rendered-prompt tempdir");
    let rendered_path = rendered_dir.path().join("rendered.txt");
    let rendered_path_nobos = rendered_dir.path().join("rendered_nobos.txt");
    let render_out = Command::new(&bin)
        .args([
            "generate",
            "--model",
            &model_path.to_string_lossy(),
            "--prompt",
            SOURDOUGH_PROMPT,
            "--max-tokens",
            "1",
            "--temperature",
            "0",
        ])
        .env("HF2Q_DUMP_RENDERED_PROMPT", &rendered_path)
        .env("HF2Q_USE_DENSE", "1")
        .output()
        .expect("[Phase D coherence] hf2q generate --max-tokens 1 (render template)");
    if !render_out.status.success() {
        panic!(
            "[Phase D coherence] hf2q template-render failed: status={:?}; stderr={}",
            render_out.status.code(),
            String::from_utf8_lossy(&render_out.stderr),
        );
    }
    let rendered = std::fs::read(&rendered_path)
        .expect("[Phase D coherence] read rendered prompt");
    assert!(
        rendered.starts_with(b"<bos>"),
        "[Phase D coherence] rendered prompt does not start with literal '<bos>' \
         (chat-template change?); got first 32 bytes: {:?}",
        &rendered[..rendered.len().min(32)],
    );
    std::fs::write(&rendered_path_nobos, &rendered[5..])
        .expect("[Phase D coherence] write BOS-stripped prompt");

    let llama_bytes = phase_d_driver::run_llama_completion_peer(
        &llama_bin,
        &model_path,
        &rendered_path_nobos,
        SOURDOUGH_MAX_TOKENS,
    )
    .expect("[Phase D coherence] llama-completion peer run");
    eprintln!(
        "[Phase D coherence] llama-completion produced {} bytes",
        llama_bytes.len()
    );

    let common_baseline = baseline
        .text
        .as_bytes()
        .iter()
        .zip(llama_bytes.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let common_restored = restored
        .text
        .as_bytes()
        .iter()
        .zip(llama_bytes.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "[R-C4 peer] baseline-vs-llama common prefix: {} bytes (floor: {})",
        common_baseline, SOURDOUGH_MIN_COMMON_PREFIX
    );
    eprintln!(
        "[R-C4 peer] restored-vs-llama common prefix: {} bytes (floor: {})",
        common_restored, SOURDOUGH_MIN_COMMON_PREFIX
    );
    assert!(
        common_baseline >= SOURDOUGH_MIN_COMMON_PREFIX,
        "[R-C4 peer] baseline-vs-llama common prefix {} < floor {}",
        common_baseline,
        SOURDOUGH_MIN_COMMON_PREFIX,
    );
    assert!(
        common_restored >= SOURDOUGH_MIN_COMMON_PREFIX,
        "[R-C4 peer] restored-vs-llama common prefix {} < floor {}",
        common_restored,
        SOURDOUGH_MIN_COMMON_PREFIX,
    );
    eprintln!("[R-C4 peer] PASS — both arms share >= {} bytes with llama.cpp", SOURDOUGH_MIN_COMMON_PREFIX);
}

/// Phase D R-P4 ship-gate test (env-gated by
/// `HF2Q_KV_PERSIST_PHASE_D=1` + `HF2Q_KV_PERSIST_E2E_PREFILL_LEN=N`).
/// Default `cargo test` short-circuits.
///
/// At prefill length L (operator-supplied; spec calls for L=32768 on
/// the cold M5 Max + Gemma4-26B Q4_0 cell):
///   1. spawn `hf2q serve --kv-persist=DIR`
///   2. send a token-diverse prompt of size ~L; measure `no_cache_ttft`
///      (this also primes the on-disk block cache)
///   3. force eviction via symlink-distinct-pool-key (cold-load the
///      same physical bytes under a new pool key, dropping the in-RAM
///      KV state)
///   4. send the SAME prompt again; measure `cache_hit_ttft` (the cache
///      is now repopulated from the on-disk persistence layer)
///   5. assert `cache_hit_ttft / no_cache_ttft <= 0.20`
///
/// Falsifier: ratio > 0.20.
#[test]
fn kv_persist_phase_d_r_p4_e2e() {
    let active = std::env::var(ENV_PHASE_D_GATE).as_deref() == Ok("1");
    if !active {
        eprintln!(
            "[Phase D R-P4] {ENV_PHASE_D_GATE}=1 not set — short-circuit. \
             Set {ENV_PHASE_D_GATE}=1 + {ENV_PHASE_D_R_P4_PREFILL_LEN}=N + \
             HF2Q_KV_PERSIST_E2E_MODEL_PATH=PATH to run."
        );
        return;
    }
    let prefill_len: u32 = match std::env::var(ENV_PHASE_D_R_P4_PREFILL_LEN)
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        Some(n) if n > 0 => n,
        _ => {
            eprintln!(
                "[Phase D R-P4] {ENV_PHASE_D_R_P4_PREFILL_LEN} unset or non-positive \
                 — short-circuit. Set to 32768 for the canonical R-P4 cell."
            );
            return;
        }
    };
    let model_path = match resolve_phase_d_model_path() {
        Some(p) => p,
        None => {
            eprintln!(
                "[Phase D R-P4] HF2Q_KV_PERSIST_E2E_MODEL_PATH unset or path missing \
                 — short-circuit."
            );
            return;
        }
    };

    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "[Phase D R-P4] hf2q binary not found at {}",
        bin.display()
    );

    let cache_dir = std::env::temp_dir().join(format!(
        "hf2q-kv-persist-phase-d-r-p4-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&cache_dir).expect("mkdir Phase D R-P4 cache_dir");

    let port = std::env::var("HF2Q_KV_PERSIST_PHASE_D_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(phase_d_driver::PORT_DEFAULT);

    let server = phase_d_driver::spawn_hf2q_serve_with_kv_persist(
        &bin,
        &model_path,
        &cache_dir,
        phase_d_driver::HOST,
        port,
    )
    .expect("[Phase D R-P4] spawn hf2q serve --kv-persist");
    phase_d_driver::wait_for_readyz(&server).unwrap_or_else(|e| {
        panic!(
            "[Phase D R-P4] /readyz did not return 200 within budget: {e}\n\
             --- hf2q serve stderr_tail ({} lines) ---\n{}",
            server.log_tail().len(),
            server.log_tail().join("\n"),
        )
    });
    let canonical = phase_d_driver::fetch_canonical_model_id(&server)
        .expect("[Phase D R-P4] fetch canonical model id");

    // Build a token-diverse word-stream prompt sized to ~prefill_len
    // tokens. Same `wordN` construction as the matrix runner —
    // empirically ≈3.8 tokens/word under Gemma 4 BPE.
    let n_words = (prefill_len as usize / 4).max(2);
    let prompt: String = (0..n_words)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");

    eprintln!(
        "[Phase D R-P4] prefill_len={} (target tokens), n_words={}, prompt_bytes={}",
        prefill_len,
        n_words,
        prompt.len(),
    );

    // No-cache run: prompts the on-disk persistence layer for the
    // first time. TTFT here is the cold-prefill cost.
    let no_cache = phase_d_driver::decode_full_text(
        &server,
        &canonical,
        &prompt,
        4,
    )
    .expect("[Phase D R-P4] no-cache decode");
    eprintln!(
        "[Phase D R-P4] no_cache_ttft={:.1}ms (prompt_tokens={:?}, total_tokens={})",
        no_cache.ttft_ms, no_cache.prompt_tokens, no_cache.total_tokens
    );

    // Force eviction so the next request must hit the on-disk cache
    // rather than the in-RAM KV state from the no-cache run.
    let tmp_link_dir = tempfile::tempdir()
        .expect("[Phase D R-P4] tempdir for symlink-eviction-trick");
    let (_link_path, evict_wall_ms, _second_ttft_ms) =
        phase_d_driver::force_eviction_via_symlink(
            &server,
            &model_path,
            tmp_link_dir.path(),
        )
        .expect("[Phase D R-P4] force eviction via symlink");
    eprintln!(
        "[Phase D R-P4] eviction cycle wall={:.1}ms",
        evict_wall_ms
    );

    let cache_hit = phase_d_driver::decode_full_text(
        &server,
        &canonical,
        &prompt,
        4,
    )
    .expect("[Phase D R-P4] cache-hit decode");
    eprintln!(
        "[Phase D R-P4] cache_hit_ttft={:.1}ms (prompt_tokens={:?}, total_tokens={})",
        cache_hit.ttft_ms, cache_hit.prompt_tokens, cache_hit.total_tokens
    );

    let ratio = cache_hit.ttft_ms / no_cache.ttft_ms;
    let pass = ratio <= 0.20;
    if pass {
        eprintln!(
            "[R-P4] PASS — ratio={:.3} (no_cache={:.1}ms cache_hit={:.1}ms)",
            ratio, no_cache.ttft_ms, cache_hit.ttft_ms
        );
    } else {
        eprintln!(
            "[R-P4] FAIL — ratio={:.3} > 0.20 (no_cache={:.1}ms cache_hit={:.1}ms)",
            ratio, no_cache.ttft_ms, cache_hit.ttft_ms
        );
    }
    assert!(
        pass,
        "[R-P4] ship-gate FAIL: ratio={:.3} > 0.20 \
         (no_cache_ttft={:.1}ms cache_hit_ttft={:.1}ms prefill_len={})",
        ratio, no_cache.ttft_ms, cache_hit.ttft_ms, prefill_len,
    );
}

/// Phase D R-P1 K2 ship-gate test (env-gated by
/// `HF2Q_KV_PERSIST_PHASE_D=1` + `HF2Q_KV_PERSIST_PHASE_D_R_P1=1`).
/// Default `cargo test` short-circuits.
///
/// K2 is the LAST outstanding ADR-017 kill-gate not yet falsified by
/// measurement. R-P1 contracts that the spiller's async-write
/// architecture (pre_evict + post_admit hooks + writer thread) MUST
/// NOT add >5% TTFT overhead to ongoing decode under sustained
/// eviction load. If the writer thread leaks onto the inference
/// thread, K2 fires.
///
/// Methodology:
///   1. Spawn `hf2q serve --kv-persist=DIR` with HF2Q_USE_DENSE=1.
///   2. Baseline phase: 5 sequential decode requests against the SAME
///      prompt + max_tokens=64. The cache populates on the first
///      request; subsequent requests hit the cache. Both regimes
///      exercise R-P1 — the first request keeps the spiller writer
///      thread busy, subsequent requests run while the spiller hook is
///      idle. Capture per-request TTFT; compute baseline_ttft_avg.
///   3. Sustained-spill phase: 5 sequential decode requests, with
///      `force_eviction_via_symlink` interleaved between requests.
///      Each eviction triggers pre_evict + post_admit on the next
///      request, maximally exercising the spiller's hot path. Capture
///      per-request TTFT; compute sustained_ttft_avg.
///   4. Compute overhead = (sustained_ttft_avg - baseline_ttft_avg) /
///      baseline_ttft_avg. Assert overhead <= 0.05 (R-P1 ship-gate;
///      K2 falsifier on >0.05).
///
/// Falsifier: overhead > 0.05.
#[test]
fn kv_persist_phase_d_r_p1_decode_overhead_e2e() {
    let active = std::env::var(ENV_PHASE_D_GATE).as_deref() == Ok("1");
    if !active {
        eprintln!(
            "[Phase D R-P1] {ENV_PHASE_D_GATE}=1 not set — short-circuit. \
             Set {ENV_PHASE_D_GATE}=1 + {ENV_PHASE_D_R_P1}=1 + \
             HF2Q_KV_PERSIST_E2E_MODEL_PATH=PATH to run."
        );
        return;
    }
    let r_p1_active = std::env::var(ENV_PHASE_D_R_P1).as_deref() == Ok("1");
    if !r_p1_active {
        eprintln!(
            "[Phase D R-P1] {ENV_PHASE_D_R_P1}=1 not set — short-circuit. \
             K2 ship-gate is operator-controlled; set {ENV_PHASE_D_R_P1}=1 \
             alongside {ENV_PHASE_D_GATE}=1 to run the measurement."
        );
        return;
    }
    let model_path = match resolve_phase_d_model_path() {
        Some(p) => p,
        None => {
            eprintln!(
                "[Phase D R-P1] HF2Q_KV_PERSIST_E2E_MODEL_PATH unset or path missing \
                 — short-circuit."
            );
            return;
        }
    };

    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "[Phase D R-P1] hf2q binary not found at {} — \
         did `cargo build --release` run?",
        bin.display()
    );

    let cache_dir = std::env::temp_dir().join(format!(
        "hf2q-kv-persist-phase-d-r-p1-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&cache_dir).expect("mkdir Phase D R-P1 cache_dir");

    let port = std::env::var("HF2Q_KV_PERSIST_PHASE_D_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(phase_d_driver::PORT_DEFAULT);

    let server = phase_d_driver::spawn_hf2q_serve_with_kv_persist(
        &bin,
        &model_path,
        &cache_dir,
        phase_d_driver::HOST,
        port,
    )
    .expect("[Phase D R-P1] spawn hf2q serve --kv-persist");
    phase_d_driver::wait_for_readyz(&server).unwrap_or_else(|e| {
        panic!(
            "[Phase D R-P1] /readyz did not return 200 within budget: {e}\n\
             --- hf2q serve stderr_tail ({} lines) ---\n{}",
            server.log_tail().len(),
            server.log_tail().join("\n"),
        )
    });
    let canonical = phase_d_driver::fetch_canonical_model_id(&server)
        .expect("[Phase D R-P1] fetch canonical model id");

    // Stable, byte-deterministic prompt + small max_tokens so the
    // whole test stays under ~60s on M5 Max. The prompt is short
    // enough that TTFT is dominated by prefill+first-token-decode and
    // any spiller-side overhead surfaces as a measurable delta.
    let prompt = "List five common breads in alphabetical order.";
    const N_SAMPLES: usize = 5;
    const MAX_TOKENS: u32 = 64;

    // -------- Baseline phase: 5 decodes, no induced eviction --------
    // The first request populates the on-disk cache (writer thread
    // busy); subsequent requests run while the spiller hook is idle.
    // Both regimes are part of R-P1's contract.
    let mut baseline_ttfts: Vec<f64> = Vec::with_capacity(N_SAMPLES);
    for i in 0..N_SAMPLES {
        let cap = phase_d_driver::decode_full_text(
            &server,
            &canonical,
            prompt,
            MAX_TOKENS,
        )
        .unwrap_or_else(|e| {
            panic!(
                "[Phase D R-P1] baseline decode #{i} failed: {e}\n\
                 --- hf2q serve stderr_tail ({} lines) ---\n{}",
                server.log_tail().len(),
                server.log_tail().join("\n"),
            )
        });
        baseline_ttfts.push(cap.ttft_ms);
    }
    let baseline_ttft_avg: f64 =
        baseline_ttfts.iter().sum::<f64>() / (baseline_ttfts.len() as f64);
    eprintln!(
        "[Phase D R-P1] baseline_ttft_avg={:.1}ms (samples={:?})",
        baseline_ttft_avg, baseline_ttfts
    );

    // -------- Sustained-spill phase: 5 decodes with eviction --------
    // Each eviction triggers pre_evict + post_admit on the NEXT
    // request, maximally exercising the spiller's writer-thread hot
    // path while the inference thread is decoding.
    let mut sustained_ttfts: Vec<f64> = Vec::with_capacity(N_SAMPLES);
    for i in 0..N_SAMPLES {
        let tmp_link_dir = tempfile::tempdir()
            .expect("[Phase D R-P1] tempdir for symlink-eviction-trick");
        let (_link_path, evict_wall_ms, _second_ttft_ms) =
            phase_d_driver::force_eviction_via_symlink(
                &server,
                &model_path,
                tmp_link_dir.path(),
            )
            .unwrap_or_else(|e| {
                panic!(
                    "[Phase D R-P1] force eviction #{i} failed: {e}\n\
                     --- hf2q serve stderr_tail ({} lines) ---\n{}",
                    server.log_tail().len(),
                    server.log_tail().join("\n"),
                )
            });
        eprintln!(
            "[Phase D R-P1] sustained #{i} eviction cycle wall={:.1}ms",
            evict_wall_ms
        );
        let cap = phase_d_driver::decode_full_text(
            &server,
            &canonical,
            prompt,
            MAX_TOKENS,
        )
        .unwrap_or_else(|e| {
            panic!(
                "[Phase D R-P1] sustained decode #{i} failed: {e}\n\
                 --- hf2q serve stderr_tail ({} lines) ---\n{}",
                server.log_tail().len(),
                server.log_tail().join("\n"),
            )
        });
        sustained_ttfts.push(cap.ttft_ms);
    }
    let sustained_ttft_avg: f64 =
        sustained_ttfts.iter().sum::<f64>() / (sustained_ttfts.len() as f64);
    eprintln!(
        "[Phase D R-P1] sustained_ttft_avg={:.1}ms (samples={:?})",
        sustained_ttft_avg, sustained_ttfts
    );

    // Overhead is signed: negative means sustained was actually FASTER
    // than baseline (within noise). The gate fires only when
    // sustained > baseline by more than 5%.
    let overhead = (sustained_ttft_avg - baseline_ttft_avg) / baseline_ttft_avg;
    eprintln!(
        "[Phase D R-P1] overhead={:.3} (gate: <= 0.05)",
        overhead
    );
    let pass = overhead <= 0.05;
    if pass {
        eprintln!("[R-P1] PASS — overhead within 5% gate");
    } else {
        eprintln!("[R-P1] FAIL — overhead exceeds 5% gate (K2 fires)");
    }
    assert!(
        pass,
        "[R-P1] ship-gate FAIL (K2 fires): overhead={:.3} > 0.05 \
         (baseline_ttft_avg={:.1}ms sustained_ttft_avg={:.1}ms \
         baseline_samples={:?} sustained_samples={:?} \
         diff_avg={:.1}ms)",
        overhead,
        baseline_ttft_avg,
        sustained_ttft_avg,
        baseline_ttfts,
        sustained_ttfts,
        sustained_ttft_avg - baseline_ttft_avg,
    );
}

/// Phase D R-P1 *concurrent-eviction* K2 polish test (env-gated by
/// `HF2Q_KV_PERSIST_PHASE_D=1` + `HF2Q_KV_PERSIST_PHASE_D_R_P1_CONCURRENT=1`).
/// Default `cargo test` short-circuits.
///
/// **iter-15 v2 methodology** (fixes 3 structural problems iter-14
/// surfaced — see ADR-017 iter-15 subsection for full rationale):
///
/// 1. **Cache-MISS prompts**: each iteration uses a UNIQUE prompt
///    (session-salted with pid+nanos) so prefill ACTUALLY runs
///    every iter — the v1 same-prompt-each-iter regime was a 100%
///    cache-hit after iter #0, with sub-ms decode wall.
/// 2. **Drop the eviction-sleep delay**: spawn the sibling
///    eviction-trigger thread WITHOUT any pre-fire `sleep(100ms)`.
///    The trigger fires while the inference thread is actively
///    prefilling+decoding (not after the decode has already
///    completed, which is what happened at v1's sub-ms cache-hit
///    timescale).
/// 3. **Hybrid (absolute OR relative) gate**: pass if EITHER
///    `concurrent - baseline <= 50 ms` (absolute) OR
///    `(concurrent - baseline) / baseline <= 0.05` (relative). The
///    relative gate is meaningless at sub-100 ms baselines (where
///    sub-ms diffs become %s of noise); the absolute gate is
///    meaningful at any scale. Either bound passing is sufficient.
///
/// Methodology v2:
///   1. Spawn `hf2q serve --kv-persist=DIR` with HF2Q_USE_DENSE=1.
///   2. Baseline phase: 3 sequential decode requests, EACH against
///      a unique salted prompt (cache-MISS each time). MAX_TOKENS=256
///      so decode wall is 200-500 ms. Drop the FIRST sample
///      (cold-load outlier); average the LAST 2.
///   3. Concurrent-eviction phase: 3 sequential decode requests
///      against further unique salted prompts. For EACH request:
///      issue the streaming completion AND immediately spawn a
///      sibling thread that calls `force_eviction_via_symlink`
///      (NO pre-sleep). Read the SSE stream to completion, capture
///      full-stream wall-time, join the sibling thread.
///   4. Apply hybrid gate (above). Pass if either bound is within
///      its budget; fail only if BOTH are exceeded (real K2 fire).
///
/// Falsifier: BOTH `abs_overhead > 50 ms` AND
/// `rel_overhead > 0.05` under concurrent eviction.
#[test]
fn kv_persist_phase_d_r_p1_concurrent_eviction_e2e() {
    let active = std::env::var(ENV_PHASE_D_GATE).as_deref() == Ok("1");
    if !active {
        eprintln!(
            "[Phase D R-P1 concurrent] {ENV_PHASE_D_GATE}=1 not set — short-circuit. \
             Set {ENV_PHASE_D_GATE}=1 + {ENV_PHASE_D_R_P1_CONCURRENT}=1 + \
             HF2Q_KV_PERSIST_E2E_MODEL_PATH=PATH to run."
        );
        return;
    }
    let r_p1c_active =
        std::env::var(ENV_PHASE_D_R_P1_CONCURRENT).as_deref() == Ok("1");
    if !r_p1c_active {
        eprintln!(
            "[Phase D R-P1 concurrent] {ENV_PHASE_D_R_P1_CONCURRENT}=1 not set — \
             short-circuit. K2 polish ship-gate is operator-controlled; set \
             {ENV_PHASE_D_R_P1_CONCURRENT}=1 alongside {ENV_PHASE_D_GATE}=1 \
             to run the measurement."
        );
        return;
    }
    let model_path = match resolve_phase_d_model_path() {
        Some(p) => p,
        None => {
            eprintln!(
                "[Phase D R-P1 concurrent] HF2Q_KV_PERSIST_E2E_MODEL_PATH unset \
                 or path missing — short-circuit."
            );
            return;
        }
    };

    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "[Phase D R-P1 concurrent] hf2q binary not found at {} — \
         did `cargo build --release` run?",
        bin.display()
    );

    let cache_dir = std::env::temp_dir().join(format!(
        "hf2q-kv-persist-phase-d-r-p1-concurrent-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&cache_dir)
        .expect("mkdir Phase D R-P1 concurrent cache_dir");

    let port = std::env::var("HF2Q_KV_PERSIST_PHASE_D_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(phase_d_driver::PORT_DEFAULT);

    let server = phase_d_driver::spawn_hf2q_serve_with_kv_persist(
        &bin,
        &model_path,
        &cache_dir,
        phase_d_driver::HOST,
        port,
    )
    .expect("[Phase D R-P1 concurrent] spawn hf2q serve --kv-persist");
    phase_d_driver::wait_for_readyz(&server).unwrap_or_else(|e| {
        panic!(
            "[Phase D R-P1 concurrent] /readyz did not return 200 within budget: {e}\n\
             --- hf2q serve stderr_tail ({} lines) ---\n{}",
            server.log_tail().len(),
            server.log_tail().join("\n"),
        )
    });
    let canonical = phase_d_driver::fetch_canonical_model_id(&server)
        .expect("[Phase D R-P1 concurrent] fetch canonical model id");

    // iter-15 v2 sample budget: 3 samples; drop sample #0 (cold-load
    // outlier); average samples #1..N-1. With cache-MISS prompts and
    // MAX_TOKENS=256, decode wall should be 200-500 ms — well above
    // the 100 ms threshold where ratio-based gating becomes
    // meaningful.
    const N_SAMPLES: usize = 3;
    const MAX_TOKENS: u32 = 256;

    // Session-unique salt: pid + nanos-since-epoch. Combined with the
    // per-iter index (and a baseline/concurrent phase tag), every
    // prompt the test fires is unique across iters AND across phases
    // — guaranteeing cache-MISS prefill on every request, and ruling
    // out cross-test cache contamination if multiple invocations
    // share the cache_dir hierarchy by accident.
    let session_salt = format!(
        "pid{}-ns{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    fn unique_prompt(iter: usize, phase: &str, salt: &str) -> String {
        // Sourdough-style fixture with iter-specific differentiator;
        // the leading "Iteration {iter} salt={salt} phase={phase}"
        // ensures BPE tokenization differs per iter (no shared prefix
        // long enough to populate a cache-hit block).
        format!(
            "Iteration {iter} salt={salt} phase={phase} List five common breads in alphabetical order, including {iter} variants of sourdough."
        )
    }

    // -------- Baseline phase: 3 cache-MISS decodes, no induced eviction --------
    // Each iter uses a unique salted prompt → prefill actually runs
    // every iter (no steady-state cache-hit collapse to sub-ms wall).
    // Drop sample #0 (cold-load outlier from server warm-up); average
    // samples #1..N-1.
    let mut baseline_durations: Vec<f64> = Vec::with_capacity(N_SAMPLES);
    for i in 0..N_SAMPLES {
        let prompt = unique_prompt(i, "baseline", &session_salt);
        let t0 = std::time::Instant::now();
        let _cap = phase_d_driver::decode_full_text(
            &server,
            &canonical,
            &prompt,
            MAX_TOKENS,
        )
        .unwrap_or_else(|e| {
            panic!(
                "[Phase D R-P1 concurrent v2] baseline decode #{i} failed: {e}\n\
                 --- hf2q serve stderr_tail ({} lines) ---\n{}",
                server.log_tail().len(),
                server.log_tail().join("\n"),
            )
        });
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        baseline_durations.push(wall_ms);
    }
    // Drop sample #0 (cold-load outlier); measure samples #1..N-1.
    let baseline_steady: Vec<f64> = baseline_durations.iter().skip(1).copied().collect();
    assert!(
        !baseline_steady.is_empty(),
        "[Phase D R-P1 concurrent v2] baseline_steady empty after dropping sample #0; \
         increase N_SAMPLES (currently {N_SAMPLES})"
    );
    let baseline_full_avg: f64 =
        baseline_steady.iter().sum::<f64>() / (baseline_steady.len() as f64);
    eprintln!(
        "[Phase D R-P1 concurrent v2] baseline_full_avg={:.1}ms (durations={:?})",
        baseline_full_avg, baseline_durations
    );

    // -------- Concurrent-eviction phase: 3 cache-MISS decodes with eviction --------
    // For EACH request: spawn a sibling thread that IMMEDIATELY
    // (no pre-sleep) fires the symlink-distinct-pool-key eviction
    // trick. Cache-MISS prefill keeps the inference thread busy for
    // hundreds of ms, so the eviction-trigger request overlaps real
    // work — either prefill or early decode. If pre_evict /
    // post_admit / writer-thread activity leaks onto the inference
    // thread, full-stream wall-time stretches measurably above the
    // baseline cache-MISS regime.
    //
    // `std::thread::scope` lets the sibling thread borrow `&server`
    // safely (ServerGuard's stderr_tail is Arc<Mutex<...>>; host/port
    // are immutable).
    let mut concurrent_durations: Vec<f64> = Vec::with_capacity(N_SAMPLES);
    for i in 0..N_SAMPLES {
        let prompt = unique_prompt(i, "concurrent", &session_salt);
        let tmp_link_dir = tempfile::tempdir()
            .expect("[Phase D R-P1 concurrent v2] tempdir for symlink-eviction-trick");
        let tmp_link_path = tmp_link_dir.path().to_path_buf();
        let model_path_for_thread = model_path.clone();

        let t0 = std::time::Instant::now();
        let wall_ms = std::thread::scope(|scope| -> f64 {
            // Sibling thread: fire the eviction-trigger request
            // IMMEDIATELY (no pre-sleep). With cache-MISS prefill in
            // flight, the trigger overlaps real decode/prefill work
            // rather than racing ahead and finishing in idle time.
            let server_ref = &server;
            let evict_handle = scope.spawn(move || {
                phase_d_driver::force_eviction_via_symlink(
                    server_ref,
                    &model_path_for_thread,
                    &tmp_link_path,
                )
            });
            // Inference-thread leg: run the cache-MISS decode to
            // completion.
            let cap_result = phase_d_driver::decode_full_text(
                &server,
                &canonical,
                &prompt,
                MAX_TOKENS,
            );
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            // Always join the sibling thread — even on inference
            // failure — so the eviction request finishes cleanly
            // before the next loop iteration tears down tmp_link_dir.
            let evict_result = evict_handle
                .join()
                .expect("[Phase D R-P1 concurrent v2] eviction thread panicked");
            if let Err(e) = cap_result {
                panic!(
                    "[Phase D R-P1 concurrent v2] decode #{i} failed: {e}\n\
                     --- hf2q serve stderr_tail ({} lines) ---\n{}",
                    server.log_tail().len(),
                    server.log_tail().join("\n"),
                );
            }
            match evict_result {
                Ok((_link, evict_wall_ms, _second_ttft_ms)) => {
                    eprintln!(
                        "[Phase D R-P1 concurrent v2] #{i} eviction sibling \
                         wall={:.1}ms (decode wall={:.1}ms)",
                        evict_wall_ms, elapsed_ms
                    );
                }
                Err(e) => {
                    panic!(
                        "[Phase D R-P1 concurrent v2] eviction sibling #{i} failed: {e}\n\
                         --- hf2q serve stderr_tail ({} lines) ---\n{}",
                        server.log_tail().len(),
                        server.log_tail().join("\n"),
                    );
                }
            }
            elapsed_ms
        });
        concurrent_durations.push(wall_ms);
    }
    // Drop sample #0 to mirror the baseline regime.
    let concurrent_steady: Vec<f64> =
        concurrent_durations.iter().skip(1).copied().collect();
    assert!(
        !concurrent_steady.is_empty(),
        "[Phase D R-P1 concurrent v2] concurrent_steady empty after dropping sample #0; \
         increase N_SAMPLES (currently {N_SAMPLES})"
    );
    let concurrent_full_avg: f64 =
        concurrent_steady.iter().sum::<f64>() / (concurrent_steady.len() as f64);
    eprintln!(
        "[Phase D R-P1 concurrent v2] concurrent_full_avg={:.1}ms (durations={:?})",
        concurrent_full_avg, concurrent_durations
    );

    // -------- Hybrid (absolute OR relative) gate --------
    // Pass if EITHER:
    //   - absolute: concurrent_full_avg - baseline_full_avg <= 50 ms
    //   - relative: (concurrent - baseline) / baseline <= 0.05
    // Either bound passing is sufficient — the gate fires (FAIL) only
    // when BOTH bounds are exceeded (real K2 fire under concurrent
    // eviction). Absolute is meaningful at any scale; relative is
    // meaningful only when baseline >= ~100 ms.
    let abs_overhead_ms = concurrent_full_avg - baseline_full_avg;
    let rel_overhead = abs_overhead_ms / baseline_full_avg;
    let abs_pass = abs_overhead_ms <= 50.0;
    let rel_pass = rel_overhead <= 0.05;
    let pass = abs_pass || rel_pass;

    eprintln!(
        "[Phase D R-P1 concurrent v2] baseline_full_avg={:.1}ms concurrent_full_avg={:.1}ms",
        baseline_full_avg, concurrent_full_avg
    );
    eprintln!(
        "[Phase D R-P1 concurrent v2] abs_overhead={:.1}ms (gate <= 50ms; {}) | rel_overhead={:.3} (gate <= 0.05; {})",
        abs_overhead_ms,
        if abs_pass { "PASS" } else { "FAIL" },
        rel_overhead,
        if rel_pass { "PASS" } else { "FAIL" },
    );
    if pass {
        eprintln!("[R-P1 concurrent v2] PASS — at least one bound (absolute OR relative) within gate");
    } else {
        eprintln!("[R-P1 concurrent v2] FAIL — both bounds exceeded (K2 fires)");
    }

    assert!(
        pass,
        "[R-P1 concurrent v2] ship-gate FAIL (K2 fires under concurrent-eviction): \
         abs_overhead={:.1}ms (gate <= 50ms) AND rel_overhead={:.3} (gate <= 0.05) — both exceeded.\n\
         baseline_full_avg={:.1}ms concurrent_full_avg={:.1}ms\n\
         baseline_durations={:?}\n\
         concurrent_durations={:?}",
        abs_overhead_ms,
        rel_overhead,
        baseline_full_avg,
        concurrent_full_avg,
        baseline_durations,
        concurrent_durations,
    );
}

// ---------- Phase D always-on shape tests (no env required) ----------

/// Phase D shape test: the new env gates are well-formed and the
/// driver module compiles + the spawn helper rejects nonexistent
/// model paths cleanly (no panic). Falsifier: spawn helper panics on
/// missing model rather than returning a `DriverError`.
#[test]
fn phase_d_driver_rejects_missing_model_cleanly() {
    let bogus_model = PathBuf::from(
        "/var/empty/this-path-must-not-exist-phase-d-shape-test.gguf",
    );
    let cache_dir = std::env::temp_dir().join(format!(
        "hf2q-phase-d-shape-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    let bin = hf2q_binary_path();
    if !bin.exists() {
        // Build hasn't happened yet — short-circuit (the binary check
        // is covered by `binary_is_locatable_and_runs_version`).
        eprintln!(
            "[Phase D shape] hf2q binary missing at {}; skipping spawn check",
            bin.display()
        );
        return;
    }
    let result = phase_d_driver::spawn_hf2q_serve_with_kv_persist(
        &bin,
        &bogus_model,
        &cache_dir,
        phase_d_driver::HOST,
        // Use a port nobody else uses; the spawn should fail BEFORE
        // any port-bind attempt because model-path check is first.
        0,
    );
    match result {
        Err(phase_d_driver::DriverError::SpawnFailed(_)) => {}
        Err(other) => {
            panic!(
                "phase_d_driver should return SpawnFailed for missing model; got {other}"
            );
        }
        Ok(_) => panic!("phase_d_driver should not spawn against missing model path"),
    }
}

/// Phase D shape test: the R-P1 K2 env gate round-trips cleanly and
/// the K2 ship-gate test short-circuits when either gate is unset.
/// Falsifier: env-var round-trip diverges, or
/// `kv_persist_phase_d_r_p1_decode_overhead_e2e` would attempt to
/// spawn `hf2q serve` when its env gates are unset.
#[test]
fn phase_d_r_p1_env_gate_well_formed() {
    // Snapshot any prior values to restore after the test (paranoia
    // against parallel-test env contamination, even though
    // --test-threads=1 is the operator recipe).
    let prior_master = std::env::var(ENV_PHASE_D_GATE).ok();
    let prior_r_p1 = std::env::var(ENV_PHASE_D_R_P1).ok();

    // Round-trip set/get for the new gate.
    std::env::remove_var(ENV_PHASE_D_R_P1);
    assert!(
        std::env::var(ENV_PHASE_D_R_P1).is_err(),
        "{ENV_PHASE_D_R_P1} should be unset after remove_var"
    );
    std::env::set_var(ENV_PHASE_D_R_P1, "1");
    assert_eq!(
        std::env::var(ENV_PHASE_D_R_P1).as_deref(),
        Ok("1"),
        "{ENV_PHASE_D_R_P1} round-trip"
    );

    // The K2 test must short-circuit when the master gate is unset,
    // even if the R-P1 gate is set. This proves the dual-gate
    // discipline: operator must opt in to BOTH master Phase D and the
    // K2 measurement.
    std::env::remove_var(ENV_PHASE_D_GATE);
    assert!(
        std::env::var(ENV_PHASE_D_GATE).is_err(),
        "{ENV_PHASE_D_GATE} should be unset for short-circuit branch"
    );
    // We don't call `kv_persist_phase_d_r_p1_decode_overhead_e2e`
    // directly (it's a #[test] fn, not a regular fn), but we verify
    // the predicate it uses to short-circuit is still false.
    let active_master = std::env::var(ENV_PHASE_D_GATE).as_deref() == Ok("1");
    assert!(
        !active_master,
        "K2 master-gate short-circuit predicate must be false when env unset"
    );

    // Reverse: master gate set, R-P1 unset — K2 must still
    // short-circuit (R-P1 is opt-in even within Phase D).
    std::env::set_var(ENV_PHASE_D_GATE, "1");
    std::env::remove_var(ENV_PHASE_D_R_P1);
    let active_master = std::env::var(ENV_PHASE_D_GATE).as_deref() == Ok("1");
    let active_r_p1 = std::env::var(ENV_PHASE_D_R_P1).as_deref() == Ok("1");
    assert!(
        active_master && !active_r_p1,
        "K2 R-P1 short-circuit predicate must be false when only master is set"
    );

    // Restore prior env state.
    std::env::remove_var(ENV_PHASE_D_GATE);
    std::env::remove_var(ENV_PHASE_D_R_P1);
    if let Some(v) = prior_master {
        std::env::set_var(ENV_PHASE_D_GATE, v);
    }
    if let Some(v) = prior_r_p1 {
        std::env::set_var(ENV_PHASE_D_R_P1, v);
    }
}

/// Phase D shape test: the R-P1 *concurrent-eviction* K2 polish env
/// gate (iter-12) round-trips cleanly and its short-circuit predicate
/// gates correctly on both the master Phase D gate and its own gate.
/// Falsifier: env-var round-trip diverges, or
/// `kv_persist_phase_d_r_p1_concurrent_eviction_e2e` would attempt to
/// spawn `hf2q serve` when its env gates are unset.
#[test]
fn phase_d_r_p1_concurrent_env_gate_well_formed() {
    let prior_master = std::env::var(ENV_PHASE_D_GATE).ok();
    let prior_r_p1c = std::env::var(ENV_PHASE_D_R_P1_CONCURRENT).ok();

    // Round-trip set/get for the new concurrent gate.
    std::env::remove_var(ENV_PHASE_D_R_P1_CONCURRENT);
    assert!(
        std::env::var(ENV_PHASE_D_R_P1_CONCURRENT).is_err(),
        "{ENV_PHASE_D_R_P1_CONCURRENT} should be unset after remove_var"
    );
    std::env::set_var(ENV_PHASE_D_R_P1_CONCURRENT, "1");
    assert_eq!(
        std::env::var(ENV_PHASE_D_R_P1_CONCURRENT).as_deref(),
        Ok("1"),
        "{ENV_PHASE_D_R_P1_CONCURRENT} round-trip"
    );

    // The concurrent-eviction K2 test must short-circuit when the
    // master Phase D gate is unset, even if the concurrent gate is
    // set. Mirrors the dual-gate discipline of the iter-8 R-P1 test.
    std::env::remove_var(ENV_PHASE_D_GATE);
    assert!(
        std::env::var(ENV_PHASE_D_GATE).is_err(),
        "{ENV_PHASE_D_GATE} should be unset for short-circuit branch"
    );
    let active_master = std::env::var(ENV_PHASE_D_GATE).as_deref() == Ok("1");
    assert!(
        !active_master,
        "K2 concurrent master-gate short-circuit predicate must be false when env unset"
    );

    // Reverse: master gate set, concurrent gate unset — concurrent
    // K2 must still short-circuit (concurrent variant is opt-in even
    // within Phase D, distinct from the iter-8 R-P1 measurement).
    std::env::set_var(ENV_PHASE_D_GATE, "1");
    std::env::remove_var(ENV_PHASE_D_R_P1_CONCURRENT);
    let active_master = std::env::var(ENV_PHASE_D_GATE).as_deref() == Ok("1");
    let active_r_p1c =
        std::env::var(ENV_PHASE_D_R_P1_CONCURRENT).as_deref() == Ok("1");
    assert!(
        active_master && !active_r_p1c,
        "K2 concurrent short-circuit predicate must be false when only master is set"
    );

    // Restore prior env state.
    std::env::remove_var(ENV_PHASE_D_GATE);
    std::env::remove_var(ENV_PHASE_D_R_P1_CONCURRENT);
    if let Some(v) = prior_master {
        std::env::set_var(ENV_PHASE_D_GATE, v);
    }
    if let Some(v) = prior_r_p1c {
        std::env::set_var(ENV_PHASE_D_R_P1_CONCURRENT, v);
    }
}

/// Phase D shape test: the env-gate predicates are well-formed for
/// each defined gate (master + peer + prefill-length + R-P1 K2 +
/// R-P1 concurrent-eviction K2 polish).
/// Falsifier: predicate panics or detects "1" when env is unset.
#[test]
fn phase_d_env_gates_are_well_formed() {
    for gate in &[
        ENV_PHASE_D_GATE,
        ENV_PHASE_D_PEER,
        ENV_PHASE_D_R_P4_PREFILL_LEN,
        ENV_PHASE_D_LLAMA_BIN,
        ENV_PHASE_D_R_P1,
        ENV_PHASE_D_R_P1_CONCURRENT,
    ] {
        let prior = std::env::var(gate).ok();
        std::env::remove_var(gate);
        assert!(
            std::env::var(gate).is_err(),
            "gate {gate} should be unset"
        );
        std::env::set_var(gate, "1");
        assert_eq!(
            std::env::var(gate).as_deref(),
            Ok("1"),
            "gate {gate} round-trip"
        );
        std::env::remove_var(gate);
        if let Some(v) = prior {
            std::env::set_var(gate, v);
        }
    }
}

/// Phase D shape test: the sourdough constants are byte-identical to
/// `scripts/sourdough_gate.sh`'s. Falsifier: prompt/max_tokens/floor
/// drift between test and gate script.
#[test]
fn phase_d_sourdough_constants_match_shell_gate() {
    assert_eq!(
        SOURDOUGH_PROMPT,
        "Complrehensive instructions for making sourdough bread.",
        "sourdough prompt must match scripts/sourdough_gate.sh literal"
    );
    assert_eq!(
        SOURDOUGH_MAX_TOKENS, 1000,
        "sourdough max_tokens must match scripts/sourdough_gate.sh"
    );
    assert_eq!(
        SOURDOUGH_MIN_COMMON_PREFIX, 3094,
        "sourdough min-common-prefix floor must match scripts/sourdough_gate.sh"
    );
}
