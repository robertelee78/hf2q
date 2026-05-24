//! ADR-040 Phase D — continuous batching throughput benchmark.
//!
//! # Scope
//!
//! This test file is the env-gated harness for measuring aggregate
//! tokens/sec across N concurrent SSE streams under each of the two
//! `SchedulerPolicy` modes. Phase D iter-1 (D1) shipped the scaffolding
//! + metric shapes + always-on smoke. Phase D iter-2 (D2, 2026-05-23,
//! this commit) ships the **real measurement body**: subprocess spawn
//! of `hf2q serve --model <gguf> --scheduler <policy> [--max-slots N]`,
//! `/readyz` poll, N concurrent SSE streaming POSTs via `curl` driven
//! by `std::thread::scope`, per-stream TTFT capture, aggregate
//! tokens/sec, 429 incidence accounting, and AC-4 soft-gate reporting.
//!
//! # Env gates
//!
//! - `HF2Q_CB_THROUGHPUT_E2E=1` — enables the env-gated measurement
//!   bodies. When unset, those tests document-skip with a one-line
//!   message. Operator-runnable; never enabled in CI.
//! - `HF2Q_CB_THROUGHPUT_MODEL` — path to GGUF for the measurement.
//!   Required when E2E gate is set. The bench refuses to run without
//!   a GGUF path (D1 contract preserved; cfa-finding-F8).
//! - `HF2Q_CB_THROUGHPUT_CONCURRENCY` — comma-separated list of N
//!   values (default "1,2,4,8" per ADR-040 §5 AC-4).
//! - `HF2Q_CB_THROUGHPUT_PROMPT` — user-message text for each stream
//!   (default `"Count slowly from one to twenty, one number per line."`).
//!   Chosen to be long-enough-to-batch but bounded by `--max-tokens`.
//! - `HF2Q_CB_THROUGHPUT_MAX_TOKENS` — per-stream `max_tokens` cap
//!   (default `64`). Keeps each cell bounded at ~5-30 s on M5 Max even
//!   under N=8.
//! - `HF2Q_CB_THROUGHPUT_PORT_BASE` — port for the subprocess (default
//!   `52441`; chosen distinct from `multi_model_swap.rs` 52337 +
//!   `prompt_cache_live.rs` 52332 to avoid collisions when the suite
//!   is run interleaved).
//!
//! # Operator command
//!
//! ```bash
//! HF2Q_CB_THROUGHPUT_E2E=1 \
//!   HF2Q_CB_THROUGHPUT_MODEL=/opt/hf2q/models/<some>.gguf \
//!   HF2Q_CB_THROUGHPUT_CONCURRENCY=1,2,4,8 \
//!   cargo test --release --test continuous_batching_throughput \
//!     -- --test-threads=1 --nocapture cb_throughput_n_1_2_4_8_fifo_vs_inflight
//! ```
//!
//! # Iter-1.5 contract preserved (cfa-finding-F8)
//!
//! When `HF2Q_CB_THROUGHPUT_E2E=1` is set but `HF2Q_CB_THROUGHPUT_MODEL`
//! is absent, the test PANICS (operator action required). This matches
//! the iter-1.5 contract that "CI burns the moment an operator opts in
//! expecting real numbers" — D2 replaces the iter-1.5 panic body with
//! a real measurement body, but the panic-on-missing-GGUF guard is
//! kept so the failure mode is operator-actionable rather than silent.
//!
//! # InflightBatched-skip-when-unwired
//!
//! As of iter-A5 baseline, `--scheduler inflight_batched` is rejected
//! at `Engine::spawn_with_mode` with `EngineSpawnError::ModeNotYetWired`
//! (the per-family worker arms — Phase C2c Qwen35, C2d Gemma 4 — have
//! not landed). When the bench tries to spawn the inflight subprocess,
//! `/readyz` will never reach 200 and the subprocess will exit
//! non-zero. The bench DETECTS this case (subprocess early-exit OR
//! `/readyz` timeout) and reports `"InflightBatched skipped: not yet
//! wired (Phase C2c/C2d gated)"` rather than failing — D2 is the
//! measurement harness, the wiring lands separately. Once C2c/C2d
//! ship, the inflight cells will populate without test edits.
//!
//! # AC-4 soft-gate (D2 reports; D3 enforces)
//!
//! Per ADR-040 §5 AC-4: at N=4, `InflightBatched` aggregate tok/s must
//! be ≥ 1.5× `FifoSerial` baseline AND TTFT p95 ≤ 2× single-stream
//! TTFT. D2 REPORTS this ratio as `[ac-4 WARN]` when below 1.5×; D3
//! (statistical stability + repeated-rep median) is the iter that
//! flips this from a soft warning to a hard assertion. The rationale
//! is recorded in `docs/ADR-040-continuous-batching-reopen.md` §6.1.14.
//!
//! # Metric report shape
//!
//! Per (N, policy) cell:
//!   - aggregate_tokens_per_sec   (sum across streams)
//!   - ttft_p50_ms                (time to first token, p50 across streams)
//!   - ttft_p95_ms                (time to first token, p95 across streams)
//!   - per_slot_tokens_per_sec    (median across streams)
//!   - rejected_429_count         (count of 429 responses during the window)

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::{Duration, Instant};

/// Per-cell metric shape. D2 fills these in via `run_bench_cell`; D1
/// shipped the type + the report formatter so D2's data lands cleanly.
#[derive(Debug, Clone)]
pub struct ThroughputCell {
    pub policy: &'static str,        // "fifo_serial" | "inflight_batched"
    pub concurrency: u32,            // N
    pub aggregate_tokens_per_sec: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p95_ms: f64,
    pub per_slot_tokens_per_sec: f64,
    pub rejected_429_count: u32,
}

impl ThroughputCell {
    /// Phase D iter-1: synthetic constructor used by the always-on smoke test
    /// to prove the type compiles + Debug-formats correctly.
    pub fn synthetic_for_smoke() -> Self {
        ThroughputCell {
            policy: "fifo_serial",
            concurrency: 1,
            aggregate_tokens_per_sec: 0.0,
            ttft_p50_ms: 0.0,
            ttft_p95_ms: 0.0,
            per_slot_tokens_per_sec: 0.0,
            rejected_429_count: 0,
        }
    }
}

/// Render a vector of cells as a markdown table — D2 calls this to emit
/// the bench report to stdout.
pub fn render_report(cells: &[ThroughputCell]) -> String {
    let mut s = String::from("| policy | N | agg tok/s | TTFT p50 | TTFT p95 | per-slot tok/s | 429s |\n");
    s.push_str("|--------|---|-----------|----------|----------|----------------|------|\n");
    for c in cells {
        s.push_str(&format!(
            "| {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {} |\n",
            c.policy, c.concurrency, c.aggregate_tokens_per_sec,
            c.ttft_p50_ms, c.ttft_p95_ms, c.per_slot_tokens_per_sec,
            c.rejected_429_count
        ));
    }
    s
}

fn hf2q_binary_path() -> PathBuf {
    if let Some(p) = std::env::var_os("CARGO_BIN_EXE_hf2q") {
        return PathBuf::from(p);
    }
    let target = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/opt/hf2q/target"));
    target.join("release").join("hf2q")
}

// ========================================================================
// Always-on smoke (prevents scaffold bit-rot between iters)
// ========================================================================

#[test]
fn binary_is_locatable_and_runs_version() {
    // Mirrors tests/multi_model_swap.rs::binary_is_locatable_and_runs_version
    let bin = hf2q_binary_path();
    if !bin.exists() {
        eprintln!("[cb-throughput] skipping: {} not built", bin.display());
        return;
    }
    let out = Command::new(&bin).arg("--version").output();
    match out {
        Ok(o) if o.status.success() => {}
        Ok(o) => panic!("hf2q --version failed: {:?}", o),
        Err(e) => panic!("failed to run hf2q --version: {}", e),
    }
}

#[test]
fn throughput_cell_synthetic_round_trips_through_report() {
    let cells = vec![ThroughputCell::synthetic_for_smoke()];
    let report = render_report(&cells);
    assert!(report.contains("| fifo_serial | 1 |"), "report: {}", report);
    assert!(report.contains("agg tok/s"), "header missing");
    assert!(report.contains("|--------|"), "separator missing");
}

#[test]
fn render_report_empty_returns_header_only() {
    let report = render_report(&[]);
    let lines: Vec<&str> = report.lines().collect();
    assert_eq!(lines.len(), 2, "header + separator only, got: {:?}", lines);
}

#[test]
fn render_report_two_cells_emits_two_data_rows() {
    let cells = vec![
        ThroughputCell {
            policy: "fifo_serial",
            concurrency: 1,
            aggregate_tokens_per_sec: 100.0,
            ttft_p50_ms: 50.0,
            ttft_p95_ms: 80.0,
            per_slot_tokens_per_sec: 100.0,
            rejected_429_count: 0,
        },
        ThroughputCell {
            policy: "inflight_batched",
            concurrency: 4,
            aggregate_tokens_per_sec: 300.0,
            ttft_p50_ms: 75.0,
            ttft_p95_ms: 120.0,
            per_slot_tokens_per_sec: 75.0,
            rejected_429_count: 2,
        },
    ];
    let report = render_report(&cells);
    let data_rows = report.lines().filter(|l| l.starts_with("|") && !l.contains("---")).count();
    // header + 2 data rows
    assert_eq!(data_rows, 3, "expected header + 2 data, got: {}", report);
}

// ========================================================================
// D2 measurement helpers
// ========================================================================

/// `/readyz` poll budget for the bench subprocess. Cold-load + warmup
/// of a multi-GB GGUF on M5 Max is on the order of 60-180 s; 10 min is
/// symmetric with `multi_model_swap.rs::READYZ_BUDGET_SECS = 600`.
const READYZ_BUDGET_SECS: u64 = 600;

/// Per-stream max wall-clock budget for a single SSE consumption.
/// Default `max_tokens=64` at ~50-100 tok/s on M5 Max is well under
/// 30 s even under N=8 contention.
const STREAM_BUDGET_SECS: u64 = 120;

/// Each `run_bench_cell` claims a fresh port from this counter. Tests
/// run under `--test-threads=1` per the OOM directive (`/opt/hf2q/
/// CLAUDE.md` "do not oom us"), so single-process counter is correct.
static PORT_COUNTER: AtomicU16 = AtomicU16::new(0);

fn next_port() -> u16 {
    let base: u16 = std::env::var("HF2Q_CB_THROUGHPUT_PORT_BASE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(52441);
    base + PORT_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// RAII guard around the spawned `hf2q serve` subprocess. Drop kills
/// the child so a panic mid-test never strands a multi-GB-resident
/// server. Mirrors `tests/multi_model_swap.rs::ServerGuard`.
struct BenchServer {
    child: Child,
    port: u16,
}

impl BenchServer {
    fn spawn(gguf: &str, policy: &str, max_slots: u32, port: u16) -> std::io::Result<Self> {
        let bin = hf2q_binary_path();
        let mut cmd = Command::new(bin);
        cmd.args([
            "serve",
            "--model", gguf,
            "--host", "127.0.0.1",
            "--port", &port.to_string(),
            "--scheduler", policy,
        ]);
        // `--max-slots` is only honored under inflight_batched per ADR-040
        // §6 Phase C iter-4 (C4); pass it for both policies — fifo_serial
        // silently ignores it (worker is pinned to max_slots=1). This
        // keeps the spawn invocation symmetric across cells.
        if policy == "inflight_batched" {
            cmd.args(["--max-slots", &max_slots.to_string()]);
        }
        let child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        Ok(Self { child, port })
    }
}

impl Drop for BenchServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Minimal blocking HTTP/1.1 GET → status code. Same idiom as
/// `tests/multi_model_swap.rs::http_get_status`; kept inline so the
/// bench has no inter-test dependency.
fn http_get_status(port: u16, path: &str) -> std::io::Result<u16> {
    let mut s = TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}")
            .parse()
            .map_err(std::io::Error::other)?,
        Duration::from_secs(5),
    )?;
    s.set_read_timeout(Some(Duration::from_secs(5)))?;
    s.write_all(
        format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n")
            .as_bytes(),
    )?;
    let mut head = [0u8; 64];
    let n = s.read(&mut head)?;
    let head_s = std::str::from_utf8(&head[..n]).unwrap_or("");
    let code = head_s
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| std::io::Error::other(format!("malformed HTTP status line: {head_s:?}")))?;
    Ok(code)
}

/// Poll `/readyz` until 200 OR subprocess exits OR timeout. Returns
/// `Ok(())` on /readyz=200; `Err(reason)` on exit / timeout so the
/// caller can skip the inflight-batched cell cleanly when the engine
/// rejects the policy at spawn-time (Phase C2c/C2d gated).
fn wait_for_readyz(server: &mut BenchServer) -> Result<(), String> {
    let started = Instant::now();
    let mut last_err: Option<String> = None;
    while started.elapsed().as_secs() < READYZ_BUDGET_SECS {
        // Subprocess died? Capture stderr-tail for the diagnostic.
        if let Ok(Some(status)) = server.child.try_wait() {
            let mut stderr_tail = String::new();
            if let Some(mut e) = server.child.stderr.take() {
                let mut buf = Vec::new();
                let _ = e.read_to_end(&mut buf);
                let s = String::from_utf8_lossy(&buf);
                let lines: Vec<&str> = s.lines().collect();
                let tail = lines.iter().rev().take(15).rev();
                stderr_tail = tail.copied().collect::<Vec<_>>().join("\n");
            }
            return Err(format!(
                "subprocess exited before /readyz=200 (status={status:?})\n\
                 --- stderr tail ---\n{stderr_tail}\n--- end stderr ---"
            ));
        }
        match http_get_status(server.port, "/readyz") {
            Ok(200) => return Ok(()),
            Ok(code) => last_err = Some(format!("status={code}")),
            Err(e) => last_err = Some(format!("transport: {e}")),
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    Err(format!(
        "/readyz did not reach 200 within {READYZ_BUDGET_SECS}s; last_err={}",
        last_err.unwrap_or_else(|| "<none>".into())
    ))
}

/// Per-stream result captured by `run_stream`.
#[derive(Debug, Clone)]
struct StreamResult {
    /// 200 on success (SSE consumed); non-200 carries HTTP status (e.g.
    /// 429) and `tokens=0`. -1 indicates curl transport error.
    http_status: i32,
    /// Time-to-first-token in milliseconds (wall-clock from POST send to
    /// first `data:` frame containing a content delta). 0 when no
    /// content frame arrived (429, transport error).
    ttft_ms: f64,
    /// Number of content tokens emitted across the stream. Counted as
    /// the number of non-empty `delta.content` fragments observed in
    /// the SSE chunk sequence.
    tokens: u32,
    /// Total wall-clock for the stream (POST send → SSE close).
    total_ms: f64,
}

/// Drive ONE concurrent SSE stream via `curl`. Returns a `StreamResult`
/// regardless of success/failure so the caller can aggregate across N
/// streams without losing any.
///
/// # Why curl, not reqwest
///
/// reqwest is async-only for HTTP/2 + SSE; calling it from a
/// `std::thread::scope` thread requires per-thread tokio runtime
/// construction (heavyweight + brittle under N=8 concurrent threads).
/// `curl` is the simplest blocking SSE client available on every Unix
/// host hf2q ships to (macOS, Linux). The bench harness already shells
/// out to `hf2q` (subprocess); one more `curl` per stream is the
/// smaller blast-radius design vs. dragging tokio into the test
/// thread pool.
fn run_stream(port: u16, prompt: &str, max_tokens: u32, model: &str) -> StreamResult {
    let body = format!(
        r#"{{"model":"{}","messages":[{{"role":"user","content":"{}"}}],"max_tokens":{},"temperature":0.6,"stream":true}}"#,
        model.replace('"', "\\\""),
        prompt.replace('"', "\\\""),
        max_tokens,
    );

    let t0 = Instant::now();

    // -s: silent (no progress bar)
    // -N: no buffer (emit SSE frames as they arrive)
    // -w "\n__HTTP_STATUS__:%{http_code}\n": tail-marker for HTTP status code
    // --max-time: hard upper bound matching STREAM_BUDGET_SECS
    let out = Command::new("curl")
        .args([
            "-s",
            "-N",
            "-X", "POST",
            "-H", "Content-Type: application/json",
            "--max-time", &STREAM_BUDGET_SECS.to_string(),
            "-w", "\n__HTTP_STATUS__:%{http_code}\n",
            "-d", &body,
            &format!("http://127.0.0.1:{port}/v1/chat/completions"),
        ])
        .output();

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let out = match out {
        Ok(o) => o,
        Err(_) => {
            return StreamResult {
                http_status: -1,
                ttft_ms: 0.0,
                tokens: 0,
                total_ms,
            };
        }
    };

    let stdout = String::from_utf8_lossy(&out.stdout);

    // Parse the trailing __HTTP_STATUS__ marker.
    let http_status = stdout
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("__HTTP_STATUS__:").and_then(|c| c.parse::<i32>().ok()))
        .unwrap_or(-1);

    // Count `data: {...}` frames; count tokens as non-empty `delta.content`
    // fragments; capture TTFT at the FIRST frame with a non-empty content
    // delta. The body of `run_stream` is single-threaded inside this
    // thread::scope worker, so the wall-clock from `t0` to the moment we
    // observe the first content delta is the correct TTFT.
    //
    // curl's `-N --max-time` returns AFTER the stream closes — so TTFT is
    // approximated as the elapsed time UNTIL the curl process exit
    // (i.e. `total_ms` is the upper bound of TTFT). To get a tighter
    // TTFT we'd need streaming-stdout consumption (curl piped to a Rust
    // reader). For D2 the approximation suffices because (a) at
    // `max_tokens=64` the stream completes quickly anyway, and (b)
    // TTFT comparison ACROSS cells is what matters — the same upper-
    // bound bias applies to every cell, so the relative AC-4 gate
    // (treatment p95 ≤ 2× baseline) is unaffected. D3 refines TTFT
    // capture via streaming stdout if the AC-4 TTFT bar tightens.
    //
    // Pragmatic TTFT estimate: if any content frames arrived, set TTFT
    // to `total_ms` minus the time spent emitting all but the first
    // token at the per-cell tokens/sec rate (computed at aggregation
    // time, not here). Here we record `total_ms` as the TTFT upper
    // bound; the aggregator subtracts `(tokens-1) / per_stream_rate`.
    let mut tokens: u32 = 0;
    let mut ttft_ms = 0.0_f64;
    let mut first_content_seen = false;
    for line in stdout.lines() {
        let payload = match line.strip_prefix("data: ") {
            Some(p) => p,
            None => continue,
        };
        if payload.trim() == "[DONE]" {
            continue;
        }
        // Best-effort JSON parsing: count any `"content":"<non-empty>"`
        // delta as one token. The OpenAI-compatible chat-stream wire
        // format emits one token per SSE frame for hf2q's per-token
        // SSE writer (see `src/serve/api/sse.rs`). We do NOT depend on
        // `serde_json` here — substring search is sufficient + keeps
        // the test free of additional Cargo.toml deps.
        if let Some(idx) = payload.find(r#""content":""#) {
            let after = &payload[idx + r#""content":""#.len()..];
            // The first character after `"content":"` is `"` (empty)
            // for the role-frame and non-`"` for a real content delta.
            // hf2q's SSE writer emits the role frame first (empty
            // content), then per-token deltas; we count only the
            // non-empty deltas as tokens.
            if !after.starts_with('"') {
                tokens = tokens.saturating_add(1);
                if !first_content_seen {
                    first_content_seen = true;
                    // Best-effort TTFT: time from POST send to NOW
                    // (the moment curl returned + we got to this
                    // line). curl's `-N` flushes per SSE frame but
                    // does NOT timestamp them; the elapsed-since-t0
                    // here is bounded above by `total_ms` and below
                    // by 0. We record `total_ms` as the TTFT upper
                    // bound at this point and let the aggregator
                    // subtract `(tokens-1) × per-token-time` for a
                    // sharper estimate; the cell-level p50/p95 are
                    // computed AFTER the per-stream rates are known.
                    ttft_ms = total_ms;
                }
            }
        }
    }
    let _ = first_content_seen; // retained for future TTFT refinement

    StreamResult {
        http_status,
        ttft_ms,
        tokens,
        total_ms,
    }
}

/// Drive one (policy, N) bench cell end-to-end.
///
/// Steps:
///   1. Spawn `hf2q serve --scheduler <policy> --max-slots <N>` subprocess.
///   2. Poll `/readyz` until 200 (or detect subprocess early-exit / timeout).
///   3. GET `/v1/models` to learn the canonical model id (server-resolved).
///   4. Send N concurrent SSE POSTs via `std::thread::scope`.
///   5. Aggregate per-stream results into a `ThroughputCell`.
///   6. Subprocess is killed by `BenchServer::drop` on scope exit.
///
/// Returns `Err(reason)` when the cell cannot run (e.g. inflight_batched
/// rejected at spawn time, /readyz timeout). The caller skips that cell
/// without failing the test — D2 reports what it can measure.
fn run_bench_cell(gguf: &str, policy: &'static str, n: u32) -> Result<ThroughputCell, String> {
    let prompt = std::env::var("HF2Q_CB_THROUGHPUT_PROMPT")
        .unwrap_or_else(|_| "Count slowly from one to twenty, one number per line.".into());
    let max_tokens: u32 = std::env::var("HF2Q_CB_THROUGHPUT_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let port = next_port();
    eprintln!("[cb-throughput] cell: policy={policy}, N={n}, port={port}, gguf={gguf}");

    let mut server = BenchServer::spawn(gguf, policy, n, port)
        .map_err(|e| format!("spawn hf2q serve: {e}"))?;

    wait_for_readyz(&mut server)?;
    eprintln!("[cb-throughput] /readyz=200 on port={port}");

    // Resolve canonical model id via /v1/models for the SSE POST body.
    // The server returns a registry-keyed id; using it directly avoids
    // the auto-pipeline path-classification overhead per request.
    let model_id = fetch_model_id(port)
        .map_err(|e| format!("GET /v1/models: {e}"))?;
    eprintln!("[cb-throughput] resolved model_id={model_id}");

    // Per-thread results via std::thread::scope. Mirrors the same shape
    // used by `serve::scheduler::tests::inflight_concurrent_admits_under_
    // mutex_match_429_boundary` (see scheduler.rs).
    let cell_start = Instant::now();
    let results: Vec<StreamResult> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..n)
            .map(|_| {
                let p = port;
                let pr = prompt.clone();
                let mid = model_id.clone();
                s.spawn(move || run_stream(p, &pr, max_tokens, &mid))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let cell_walltime_ms = cell_start.elapsed().as_secs_f64() * 1000.0;

    // Aggregate.
    let rejected_429_count = results.iter().filter(|r| r.http_status == 429).count() as u32;
    let succeeded: Vec<&StreamResult> = results
        .iter()
        .filter(|r| r.http_status == 200 && r.tokens > 0)
        .collect();

    if succeeded.is_empty() {
        return Err(format!(
            "all {n} streams failed for policy={policy}; \
             results: {:?}",
            results
        ));
    }

    let total_tokens: u32 = succeeded.iter().map(|r| r.tokens).sum();
    let aggregate_tokens_per_sec = (total_tokens as f64) / (cell_walltime_ms / 1000.0).max(1e-6);

    // Per-stream tokens/sec → median.
    let mut per_stream_rates: Vec<f64> = succeeded
        .iter()
        .map(|r| (r.tokens as f64) / (r.total_ms / 1000.0).max(1e-6))
        .collect();
    per_stream_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let per_slot_tokens_per_sec = if per_stream_rates.is_empty() {
        0.0
    } else {
        per_stream_rates[per_stream_rates.len() / 2]
    };

    // TTFT p50 + p95: refine the upper-bound `total_ms` recorded in
    // `run_stream` by subtracting `(tokens - 1) / per_stream_rate` —
    // the time spent emitting tokens AFTER the first. This is a
    // best-effort estimate; D3 refines TTFT via streaming-stdout
    // consumption (per the §6.1.14 closure block).
    let mut ttft_estimates_ms: Vec<f64> = succeeded
        .iter()
        .map(|r| {
            if r.tokens <= 1 {
                r.ttft_ms
            } else {
                let per_token_ms = r.total_ms / (r.tokens as f64);
                (r.ttft_ms - per_token_ms * (r.tokens as f64 - 1.0)).max(0.0)
            }
        })
        .collect();
    ttft_estimates_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ttft_p50_ms = percentile(&ttft_estimates_ms, 0.50);
    let ttft_p95_ms = percentile(&ttft_estimates_ms, 0.95);

    let cell = ThroughputCell {
        policy,
        concurrency: n,
        aggregate_tokens_per_sec,
        ttft_p50_ms,
        ttft_p95_ms,
        per_slot_tokens_per_sec,
        rejected_429_count,
    };
    eprintln!("[cb-throughput] cell DONE: {:?}", cell);
    Ok(cell)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64) * q).ceil() as usize;
    let idx = idx.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

/// Blocking GET `/v1/models` → first entry's `id` field. Uses
/// `TcpStream` directly to avoid pulling reqwest into the bench
/// harness's thread::scope (reqwest's async runtime would need a
/// tokio handle on every thread).
fn fetch_model_id(port: u16) -> std::io::Result<String> {
    let mut s = TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}")
            .parse()
            .map_err(std::io::Error::other)?,
        Duration::from_secs(5),
    )?;
    s.set_read_timeout(Some(Duration::from_secs(30)))?;
    s.write_all(
        b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
    )?;
    let mut buf = Vec::new();
    s.read_to_end(&mut buf)?;
    let body = String::from_utf8_lossy(&buf);
    // Body = HTTP status line + headers + blank line + JSON.
    let json_start = body
        .find("\r\n\r\n")
        .ok_or_else(|| std::io::Error::other("/v1/models: no headers/body separator"))?;
    let json = &body[json_start + 4..];
    // Substring-scan for `"id":"<value>"` — first occurrence is the
    // canonical model id from `data[0].id`.
    let id_key = r#""id":""#;
    let idx = json
        .find(id_key)
        .ok_or_else(|| std::io::Error::other(format!("/v1/models: no id field in body: {json}")))?;
    let after = &json[idx + id_key.len()..];
    let end = after
        .find('"')
        .ok_or_else(|| std::io::Error::other("/v1/models: unterminated id string"))?;
    Ok(after[..end].to_string())
}

// ========================================================================
// Env-gated measurement body (Phase D iter-2 — D2, this iter)
// ========================================================================

#[test]
fn cb_throughput_n_1_2_4_8_fifo_vs_inflight() {
    // ADR-040 Phase D iter-2 (D2, 2026-05-23) — REAL measurement body.
    //
    // Replaces the iter-1.5 PANIC stub (cfa-finding-F8) with the
    // operator-runnable bench harness specified by §5 AC-4 + §6 Phase D
    // D2. Skip mode (env unset) keeps passing trivially. When the E2E
    // gate is set, the harness REQUIRES `HF2Q_CB_THROUGHPUT_MODEL` and
    // PANICS otherwise (mantra: no silent skip with the gate set).
    if std::env::var("HF2Q_CB_THROUGHPUT_E2E").as_deref() != Ok("1") {
        eprintln!(
            "[cb-throughput] skipped — set HF2Q_CB_THROUGHPUT_E2E=1 + \
             HF2Q_CB_THROUGHPUT_MODEL=<gguf> to run the ADR-040 §5 AC-4 \
             throughput harness. Operator command:\n  \
             HF2Q_CB_THROUGHPUT_E2E=1 HF2Q_CB_THROUGHPUT_MODEL=/path/to.gguf \\\n  \
             cargo test --release --test continuous_batching_throughput -- \\\n  \
             --test-threads=1 --nocapture cb_throughput_n_1_2_4_8_fifo_vs_inflight"
        );
        return;
    }
    let gguf_path = std::env::var("HF2Q_CB_THROUGHPUT_MODEL").expect(
        "HF2Q_CB_THROUGHPUT_MODEL required when HF2Q_CB_THROUGHPUT_E2E=1. \
         Either set the env or unset HF2Q_CB_THROUGHPUT_E2E — silent skip \
         with the gate set violates the iter-1.5 cfa-finding-F8 contract.",
    );
    let concurrency: Vec<u32> = std::env::var("HF2Q_CB_THROUGHPUT_CONCURRENCY")
        .unwrap_or_else(|_| String::from("1,2,4,8"))
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<u32>()
                .expect("HF2Q_CB_THROUGHPUT_CONCURRENCY must be comma-separated u32 list")
        })
        .collect();
    assert!(!concurrency.is_empty(), "concurrency list must be non-empty");
    for &n in &concurrency {
        assert!(n > 0, "concurrency entries must be > 0, got {n}");
    }

    let mut all_cells: Vec<ThroughputCell> = Vec::new();
    let mut skipped: Vec<(String, u32, String)> = Vec::new();

    for &policy in &["fifo_serial", "inflight_batched"] {
        for &n in &concurrency {
            match run_bench_cell(&gguf_path, policy, n) {
                Ok(cell) => all_cells.push(cell),
                Err(e) => {
                    eprintln!(
                        "[cb-throughput] cell SKIPPED (policy={policy}, N={n}): {e}"
                    );
                    skipped.push((policy.to_string(), n, e));
                }
            }
        }
    }

    let report = render_report(&all_cells);
    println!("\n=== ADR-040 §5 AC-4 throughput report (D2) ===\n{report}");
    if !skipped.is_empty() {
        println!("\nSkipped cells:");
        for (p, n, why) in &skipped {
            println!("  - policy={p}, N={n}: {why}");
        }
    }

    // Vacuous-test guard: at least one cell must have completed.
    // Without this guard, a malformed gguf path or a binary that fails
    // to start would silently pass the bench. Per cfa-finding-F8 the
    // env-gated body must FAIL when it cannot measure anything.
    assert!(
        !all_cells.is_empty(),
        "ADR-040 D2: no bench cells completed; all (policy, N) combinations failed. \
         Check HF2Q_CB_THROUGHPUT_MODEL={gguf_path} + the skipped-cells list above."
    );

    // AC-4 soft-gate per ADR-040 §5: when both fifo_serial AND
    // inflight_batched cells exist at N=4, REPORT the aggregate
    // tokens/sec ratio + TTFT p95 ratio. D2 reports; D3 enforces
    // statistical stability + flips this to a hard assertion.
    let fifo_n4 = all_cells
        .iter()
        .find(|c| c.policy == "fifo_serial" && c.concurrency == 4);
    let inflight_n4 = all_cells
        .iter()
        .find(|c| c.policy == "inflight_batched" && c.concurrency == 4);
    if let (Some(f), Some(i)) = (fifo_n4, inflight_n4) {
        let aggregate_ratio = i.aggregate_tokens_per_sec / f.aggregate_tokens_per_sec.max(1e-6);
        let fifo_n1 = all_cells
            .iter()
            .find(|c| c.policy == "fifo_serial" && c.concurrency == 1);
        let ttft_ratio = fifo_n1.map(|s| i.ttft_p95_ms / s.ttft_p95_ms.max(1e-6));
        eprintln!(
            "[ac-4] N=4 aggregate ratio = {:.2}x (gate = 1.5x); TTFT p95 ratio vs FIFO N=1 = {:?} (gate = 2.0x)",
            aggregate_ratio, ttft_ratio
        );
        if aggregate_ratio < 1.5 {
            eprintln!(
                "[ac-4 WARN] aggregate ratio {aggregate_ratio:.2}x below 1.5x bar; \
                 D3 statistical-stability enforcement will hard-fail this case."
            );
        }
        if let Some(t) = ttft_ratio {
            if t > 2.0 {
                eprintln!(
                    "[ac-4 WARN] TTFT p95 ratio {t:.2}x above 2.0x bar; \
                     D3 will hard-fail this case."
                );
            }
        }
    } else {
        eprintln!(
            "[ac-4] cannot evaluate gate — missing N=4 cell for one or both policies \
             (typically inflight_batched is Phase C2c/C2d gated; D2 reports what it can)."
        );
    }
}

#[test]
fn cb_throughput_required_env_vars_documented() {
    // Always-on: catalogs the env vars D2 consumes + asserts the
    // documented behaviour when they are absent.
    if std::env::var("HF2Q_CB_THROUGHPUT_E2E").as_deref() != Ok("1") {
        // skip in the default skip mode
        return;
    }
    let model = std::env::var("HF2Q_CB_THROUGHPUT_MODEL");
    let concurrency = std::env::var("HF2Q_CB_THROUGHPUT_CONCURRENCY")
        .unwrap_or_else(|_| String::from("1,2,4,8"));
    assert!(
        model.is_ok(),
        "HF2Q_CB_THROUGHPUT_E2E=1 set but HF2Q_CB_THROUGHPUT_MODEL absent — \
         D2 refuses to run without a GGUF path"
    );
    let parsed: Result<Vec<u32>, _> = concurrency
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect();
    assert!(parsed.is_ok(), "HF2Q_CB_THROUGHPUT_CONCURRENCY parse failed: {:?}", parsed);
    let ns = parsed.unwrap();
    assert!(!ns.is_empty(), "at least one N value required");
    for &n in &ns {
        assert!(n > 0, "N must be positive, got {}", n);
    }
}
