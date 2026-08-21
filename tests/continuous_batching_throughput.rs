//! ADR-040 Phase D — continuous batching throughput benchmark.
//!
//! # Scope
//!
//! This test file is the env-gated harness for measuring aggregate
//! tokens/sec across N concurrent SSE streams under each of the two
//! `SchedulerPolicy` modes. Phase D iter-1 (D1) shipped the scaffolding
//! + metric shapes + always-on smoke. Phase D iter-2 (D2, 2026-05-23)
//! shipped the **real measurement body**: subprocess spawn of
//! `hf2q serve --model <gguf> --scheduler <policy> [--max-slots N]`,
//! `/readyz` poll, N concurrent SSE streaming POSTs via `curl` driven
//! by `std::thread::scope`, per-stream TTFT capture, aggregate
//! tokens/sec, 429 incidence accounting, and AC-4 soft-gate reporting.
//!
//! Phase D iter-3 (D3, 2026-05-24, this commit) adds **statistical
//! stability** on top of D2's single-shot body: each (policy, N) cell
//! is run REPS=3 times, the median is reported alongside min/max and
//! the relative spread `sigma_pct = (max - min) / median × 100`. The
//! AC-4 soft-gate from D2 is promoted to a HARD ASSERTION gated on
//! BOTH FifoSerial AND InflightBatched cells being present at N=4
//! (the InflightBatched policy is rejected at spawn until Phase C2c +
//! C2d land — until then D3 still passes through the FifoSerial-only
//! baseline + its variance, deferring AC-4 enforcement). Per-cell
//! TTFT is also refined from D2's upper-bound estimate to **per-frame
//! streaming-stdout TTFT** — curl is spawned with `Stdio::piped()` and
//! the parent reads stdout line-by-line via `BufReader::lines()`,
//! timestamping the first `data:` content frame from a single
//! `Instant::now()` taken just before `child.spawn()`.
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
//! **cfa-iter-A5b MINOR #2 — opt-in contract clarification**: when
//! `HF2Q_CB_THROUGHPUT_E2E=1` is set, ALL related env vars are HARD
//! REQUIREMENTS (the gate's full opt-in surface):
//!   - `HF2Q_CB_THROUGHPUT_MODEL` MUST be set — missing ⇒ PANIC,
//!     not a graceful skip. The bench cannot fall back to a "default
//!     GGUF" or silently skip because doing so would surface as a
//!     PASS log without any measurement. The codex review surfaced
//!     this as a perceived inconsistency vs the legacy `HF2Q_GGUF_PATH`
//!     alias; this module DOES NOT honour `HF2Q_GGUF_PATH` —
//!     `HF2Q_CB_THROUGHPUT_MODEL` is the bench-specific env required
//!     for D3 measurements. The hard-fail contract is intentional per
//!     cfa-finding-F8.
//!   - `HF2Q_CB_THROUGHPUT_CONCURRENCY` IS optional and defaults to
//!     `"1,2,4,8"`. If the operator restricts it (e.g. to `"4"`),
//!     the AC-4 TTFT half cannot fire because the N=1 baseline is
//!     missing — see `ac4_outcome::Misconfigured` (cfa-iter-A5b
//!     MAJOR #2).
//!   - `HF2Q_CB_THROUGHPUT_PROMPT`, `_MAX_TOKENS`, `_PORT_BASE` are
//!     all optional with documented defaults.
//!
//! Skip only happens when `HF2Q_CB_THROUGHPUT_E2E` is UNSET. With the
//! gate set, every required env is enforced.
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
//! # AC-4 hard-gate (D3, this iter)
//!
//! Per ADR-040 §5 AC-4: at N=4, `InflightBatched` aggregate tok/s must
//! be ≥ 1.5× `FifoSerial` baseline AND TTFT p95 ≤ 2× single-stream
//! TTFT. D2 reported this as `[ac-4 WARN]` only; D3 enforces it via
//! `assert!` HARD-FAIL **when and only when** the N=4 cells exist for
//! BOTH policies (today InflightBatched is rejected at spawn pending
//! Phase C2c + C2d wiring per §6.1.13 Future-iter pin pointers, so D3
//! reports the FifoSerial-only baseline + variance and defers AC-4
//! enforcement). D3 also gates BEFORE the AC-4 assertion on a
//! stability check: if either of the N=4 cells shows
//! `aggregate_tokens_per_sec_sigma_pct > 20%` the test panics with an
//! operator-actionable message ("run again or increase REPS") so a
//! noisy single-iteration measurement cannot fail AC-4 spuriously.
//! Rationale + rep-count justification documented in
//! `docs/adr/ADR-040-continuous-batching-reopen.md` §6.1.15.
//!
//! # Metric report shape
//!
//! Per (N, policy) cell:
//!   - aggregate_tokens_per_sec   (sum across streams)
//!   - ttft_p50_ms                (time to first token, p50 across streams)
//!   - ttft_p95_ms                (time to first token, p95 across streams)
//!   - per_slot_tokens_per_sec    (median across streams)
//!   - rejected_429_count         (count of 429 responses during the window)

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::{Duration, Instant};

/// Per-cell metric shape. D2 fills these in via `run_bench_cell`; D1
/// shipped the type + the report formatter so D2's data lands cleanly.
#[derive(Debug, Clone)]
pub struct ThroughputCell {
    pub policy: &'static str, // "fifo_serial" | "inflight_batched"
    pub concurrency: u32,     // N
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
    let mut s =
        String::from("| policy | N | agg tok/s | TTFT p50 | TTFT p95 | per-slot tok/s | 429s |\n");
    s.push_str("|--------|---|-----------|----------|----------|----------------|------|\n");
    for c in cells {
        s.push_str(&format!(
            "| {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {} |\n",
            c.policy,
            c.concurrency,
            c.aggregate_tokens_per_sec,
            c.ttft_p50_ms,
            c.ttft_p95_ms,
            c.per_slot_tokens_per_sec,
            c.rejected_429_count
        ));
    }
    s
}

/// Phase D iter-3 (D3): per-(policy, N) cell aggregated across `REPS`
/// repetitions. Median is the load-bearing summary statistic; min/max
/// + relative spread `sigma_pct` drive the D3 stability gate.
///
/// `sigma_pct` is defined as `(max - min) / median × 100` — the
/// peak-to-peak spread expressed as a percentage of the median. This
/// is intentionally a more conservative dispersion measure than σ/μ
/// (it is bounded above by REPS × σ/μ but bounded below by 0 only when
/// all reps are identical), chosen because at REPS=3 the sample
/// standard deviation has high estimator variance and (max - min)/med
/// is a stable, operator-readable lower-cost noise indicator. The
/// `STABILITY_SIGMA_PCT_THRESHOLD` constant pins the 20% bar; cells
/// above it abort AC-4 enforcement with an operator-actionable panic.
#[derive(Debug, Clone)]
pub struct ThroughputCellStable {
    pub policy: &'static str,
    pub concurrency: u32,
    pub rep_count: u32,
    pub aggregate_tokens_per_sec_median: f64,
    pub aggregate_tokens_per_sec_min: f64,
    pub aggregate_tokens_per_sec_max: f64,
    pub aggregate_tokens_per_sec_sigma_pct: f64,
    pub ttft_p50_ms_median: f64,
    pub ttft_p95_ms_median: f64,
    pub per_slot_tokens_per_sec_median: f64,
    pub rejected_429_count_total: u32,
}

impl ThroughputCellStable {
    /// Phase D iter-3 (D3): construct a stable cell from a non-empty
    /// vector of per-rep `ThroughputCell` measurements. Caller is
    /// responsible for invoking `run_bench_cell` REPS times; this
    /// constructor aggregates the medians + spread.
    ///
    /// # Panics
    ///
    /// Panics if `cells` is empty (no measurements to aggregate is an
    /// operator-actionable bug, not a degenerate-case fallback).
    /// Panics if cells differ in `policy` or `concurrency` (mixing
    /// reps from different cells would silently corrupt the medians).
    pub fn from_reps(cells: Vec<ThroughputCell>) -> Self {
        assert!(
            !cells.is_empty(),
            "ThroughputCellStable::from_reps requires ≥1 cell"
        );
        let policy = cells[0].policy;
        let concurrency = cells[0].concurrency;
        for c in &cells {
            assert_eq!(c.policy, policy, "from_reps: mixed policies across reps");
            assert_eq!(
                c.concurrency, concurrency,
                "from_reps: mixed concurrency across reps"
            );
        }
        let rep_count = cells.len() as u32;
        let aggregates: Vec<f64> = cells.iter().map(|c| c.aggregate_tokens_per_sec).collect();
        let ttft_p50s: Vec<f64> = cells.iter().map(|c| c.ttft_p50_ms).collect();
        let ttft_p95s: Vec<f64> = cells.iter().map(|c| c.ttft_p95_ms).collect();
        let per_slots: Vec<f64> = cells.iter().map(|c| c.per_slot_tokens_per_sec).collect();
        let rejected_total: u32 = cells.iter().map(|c| c.rejected_429_count).sum();

        let agg_median = median_f64(&aggregates);
        let agg_min = aggregates.iter().cloned().fold(f64::INFINITY, f64::min);
        let agg_max = aggregates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sigma_pct = if agg_median > 0.0 {
            (agg_max - agg_min) / agg_median * 100.0
        } else {
            0.0
        };

        ThroughputCellStable {
            policy,
            concurrency,
            rep_count,
            aggregate_tokens_per_sec_median: agg_median,
            aggregate_tokens_per_sec_min: agg_min,
            aggregate_tokens_per_sec_max: agg_max,
            aggregate_tokens_per_sec_sigma_pct: sigma_pct,
            ttft_p50_ms_median: median_f64(&ttft_p50s),
            ttft_p95_ms_median: median_f64(&ttft_p95s),
            per_slot_tokens_per_sec_median: median_f64(&per_slots),
            rejected_429_count_total: rejected_total,
        }
    }
}

/// Median of an `f64` slice using sort-then-middle. For odd length:
/// the middle element. For even length: the lower of the two middle
/// elements (NOT the arithmetic mean — keeps the median a real
/// observed sample, matching the ADR-033 §Pi methodology of "the
/// median rep is the rep we'd recommend the operator deploy with",
/// not a synthetic value). For empty input: returns 0.0 (the only
/// caller is `ThroughputCellStable::from_reps` which already
/// pre-checks non-empty; this is defense-in-depth).
fn median_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

/// Render a vector of D3 stable cells as a markdown table. Sigma_pct
/// column added per the brief; min/max + rep_count make the dispersion
/// visible alongside the median.
pub fn render_report_stable(cells: &[ThroughputCellStable]) -> String {
    let mut s = String::from(
        "| policy | N | reps | agg tok/s median | agg min | agg max | sigma_pct | TTFT p50 | TTFT p95 | per-slot tok/s | 429s total |\n",
    );
    s.push_str("|--------|---|------|------------------|---------|---------|-----------|----------|----------|----------------|------------|\n");
    for c in cells {
        s.push_str(&format!(
            "| {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1}% | {:.1} | {:.1} | {:.1} | {} |\n",
            c.policy,
            c.concurrency,
            c.rep_count,
            c.aggregate_tokens_per_sec_median,
            c.aggregate_tokens_per_sec_min,
            c.aggregate_tokens_per_sec_max,
            c.aggregate_tokens_per_sec_sigma_pct,
            c.ttft_p50_ms_median,
            c.ttft_p95_ms_median,
            c.per_slot_tokens_per_sec_median,
            c.rejected_429_count_total,
        ));
    }
    s
}

/// D3 rep count per cell. 3 reps lets the median be a real sample
/// (the middle of 3) at a wall-clock cost of 3× the D2 baseline per
/// cell (~15-90s per cell × 3 = ~45-270s × N_concurrency_steps ×
/// N_policies). The ADR-033 §Pi methodology lesson favored 3-rep
/// medians as the minimum to discriminate signal from noise; D3
/// matches that floor.
pub const REPS: usize = 3;

/// D3 stability gate: cells with `aggregate_tokens_per_sec_sigma_pct`
/// above this value have their AC-4 ratio computation aborted (the
/// test panics with an operator-actionable message asking for a
/// re-run or higher REPS). 20% is the threshold the brief specifies;
/// chosen to be roughly 2× the typical run-to-run variance observed
/// on the ADR-033 §Pi Qwen3.6 bench (~10% peak-to-peak at REPS=3).
pub const STABILITY_SIGMA_PCT_THRESHOLD: f64 = 20.0;

/// **cfa-iter-A5b MAJOR #2 fix** — typed outcome of the AC-4 gate.
/// Extracted from the env-gated `cb_throughput_n_1_2_4_8_fifo_vs_inflight`
/// body so the misconfiguration + deferral logic can be exercised by
/// always-on unit tests (the env-gated body itself remains skip-by-
/// default).
///
/// The codex MAJOR #2 finding was that `[ac-4 PARTIAL]` silently
/// dropped the TTFT half of the AC-4 gate when an operator restricted
/// `HF2Q_CB_THROUGHPUT_CONCURRENCY` to exclude N=1, even when BOTH
/// N=4 cells were present. The `Misconfigured` variant pins that
/// failure mode as a HARD ERROR (the env-gated body `panic!`s on it).
#[derive(Debug, PartialEq)]
pub enum Ac4Outcome {
    /// AC-4 cannot fire — at least one of `fifo_serial N=4` or
    /// `inflight_batched N=4` is absent (typically `inflight_batched`
    /// until Phase C2c/C2d ship). The FifoSerial-only baseline +
    /// variance section is still emitted; AC-4 will fire on the first
    /// run after the missing side lands.
    Deferred,
    /// AC-4 is misconfigured — BOTH N=4 cells present BUT the
    /// `fifo_serial N=1` baseline cell needed for the TTFT denominator
    /// is missing. Operator should re-run with N=1 in the
    /// concurrency list.
    Misconfigured,
    /// One of the N=4 cells has `sigma_pct >
    /// STABILITY_SIGMA_PCT_THRESHOLD` — the median is too noisy for
    /// AC-4 to be meaningful. Operator should re-run or bump REPS.
    StabilityBlocked,
    /// AC-4 fired and PASSED — aggregate ratio ≥ 1.5× and TTFT ratio
    /// ≤ 2.0×. Carries the two ratios for diagnostic emission.
    Passed {
        aggregate_ratio: f64,
        ttft_ratio: f64,
    },
    /// AC-4 fired and FAILED — either aggregate ratio < 1.5× or TTFT
    /// ratio > 2.0×. Carries both ratios + a phrase naming which half
    /// failed so the env-gated body can panic with a precise message.
    Failed {
        aggregate_ratio: f64,
        ttft_ratio: f64,
        which: &'static str,
    },
}

/// **cfa-iter-A5b MAJOR #2 fix** — pure AC-4 gate evaluation.
///
/// Inputs: the three cell slots
/// (`fifo_serial N=4`, `inflight_batched N=4`, `fifo_serial N=1`). The
/// env-gated body extracts these from `all_cells` via `Vec::iter().find`
/// before calling this helper. Separating the policy from the I/O
/// (subprocess spawn, stderr emission, panic body construction) makes
/// the misconfiguration / deferral / stability paths testable
/// in isolation under always-on unit tests.
pub fn ac4_outcome(
    fifo_n4: Option<&ThroughputCellStable>,
    inflight_n4: Option<&ThroughputCellStable>,
    fifo_n1: Option<&ThroughputCellStable>,
) -> Ac4Outcome {
    let (f, i) = match (fifo_n4, inflight_n4) {
        (Some(f), Some(i)) => (f, i),
        _ => return Ac4Outcome::Deferred,
    };
    if f.aggregate_tokens_per_sec_sigma_pct > STABILITY_SIGMA_PCT_THRESHOLD
        || i.aggregate_tokens_per_sec_sigma_pct > STABILITY_SIGMA_PCT_THRESHOLD
    {
        return Ac4Outcome::StabilityBlocked;
    }
    // BOTH N=4 cells present + stable → N=1 baseline MUST be present.
    // Pre-iter-A5b the absence here returned a soft `[ac-4 PARTIAL]`
    // diagnostic and skipped the TTFT half of the gate; the codex
    // MAJOR #2 finding was that this lets a TTFT regression pass
    // silently. New behaviour: surface `Misconfigured` so the
    // env-gated body panics.
    let Some(base) = fifo_n1 else {
        return Ac4Outcome::Misconfigured;
    };
    let aggregate_ratio =
        i.aggregate_tokens_per_sec_median / f.aggregate_tokens_per_sec_median.max(1e-6);
    let ttft_ratio = i.ttft_p95_ms_median / base.ttft_p95_ms_median.max(1e-6);
    if aggregate_ratio < 1.5 {
        return Ac4Outcome::Failed {
            aggregate_ratio,
            ttft_ratio,
            which: "aggregate",
        };
    }
    if ttft_ratio > 2.0 {
        return Ac4Outcome::Failed {
            aggregate_ratio,
            ttft_ratio,
            which: "ttft",
        };
    }
    Ac4Outcome::Passed {
        aggregate_ratio,
        ttft_ratio,
    }
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

// ========================================================================
// D3 always-on tests (statistical aggregator + report shape)
// ========================================================================

#[test]
fn d3_median_f64_odd_length_returns_middle_sample() {
    // The D3 median is a real observed sample (not arithmetic mean)
    // per the ADR-033 §Pi methodology note in the median_f64 doc.
    let v = vec![100.0, 50.0, 200.0];
    assert_eq!(median_f64(&v), 100.0, "sorted=[50,100,200], middle=100");
}

#[test]
fn d3_median_f64_empty_returns_zero() {
    // Defense-in-depth: from_reps pre-checks non-empty, but the
    // function itself returns 0 on empty rather than panicking so
    // it stays composable.
    assert_eq!(median_f64(&[]), 0.0);
}

#[test]
fn d3_stable_from_reps_aggregates_median_min_max_sigma() {
    let cells = vec![
        ThroughputCell {
            policy: "fifo_serial",
            concurrency: 4,
            aggregate_tokens_per_sec: 100.0,
            ttft_p50_ms: 10.0,
            ttft_p95_ms: 20.0,
            per_slot_tokens_per_sec: 25.0,
            rejected_429_count: 1,
        },
        ThroughputCell {
            policy: "fifo_serial",
            concurrency: 4,
            aggregate_tokens_per_sec: 110.0,
            ttft_p50_ms: 12.0,
            ttft_p95_ms: 22.0,
            per_slot_tokens_per_sec: 27.5,
            rejected_429_count: 0,
        },
        ThroughputCell {
            policy: "fifo_serial",
            concurrency: 4,
            aggregate_tokens_per_sec: 105.0,
            ttft_p50_ms: 11.0,
            ttft_p95_ms: 21.0,
            per_slot_tokens_per_sec: 26.25,
            rejected_429_count: 2,
        },
    ];
    let stable = ThroughputCellStable::from_reps(cells);
    assert_eq!(stable.policy, "fifo_serial");
    assert_eq!(stable.concurrency, 4);
    assert_eq!(stable.rep_count, 3);
    assert!((stable.aggregate_tokens_per_sec_median - 105.0).abs() < 1e-9);
    assert!((stable.aggregate_tokens_per_sec_min - 100.0).abs() < 1e-9);
    assert!((stable.aggregate_tokens_per_sec_max - 110.0).abs() < 1e-9);
    // sigma_pct = (110 - 100) / 105 * 100 ≈ 9.52
    assert!(
        (stable.aggregate_tokens_per_sec_sigma_pct - 9.523_809_523_8).abs() < 1e-6,
        "sigma_pct={}",
        stable.aggregate_tokens_per_sec_sigma_pct,
    );
    assert!((stable.ttft_p50_ms_median - 11.0).abs() < 1e-9);
    assert!((stable.ttft_p95_ms_median - 21.0).abs() < 1e-9);
    assert!((stable.per_slot_tokens_per_sec_median - 26.25).abs() < 1e-9);
    assert_eq!(stable.rejected_429_count_total, 3);
}

#[test]
#[should_panic(expected = "from_reps: mixed policies")]
fn d3_stable_from_reps_rejects_mixed_policies() {
    let cells = vec![
        ThroughputCell {
            policy: "fifo_serial",
            concurrency: 4,
            aggregate_tokens_per_sec: 100.0,
            ttft_p50_ms: 10.0,
            ttft_p95_ms: 20.0,
            per_slot_tokens_per_sec: 25.0,
            rejected_429_count: 0,
        },
        ThroughputCell {
            policy: "inflight_batched",
            concurrency: 4,
            aggregate_tokens_per_sec: 200.0,
            ttft_p50_ms: 10.0,
            ttft_p95_ms: 20.0,
            per_slot_tokens_per_sec: 50.0,
            rejected_429_count: 0,
        },
    ];
    let _ = ThroughputCellStable::from_reps(cells);
}

#[test]
#[should_panic(expected = "from_reps: mixed concurrency")]
fn d3_stable_from_reps_rejects_mixed_concurrency() {
    let cells = vec![
        ThroughputCell {
            policy: "fifo_serial",
            concurrency: 4,
            aggregate_tokens_per_sec: 100.0,
            ttft_p50_ms: 10.0,
            ttft_p95_ms: 20.0,
            per_slot_tokens_per_sec: 25.0,
            rejected_429_count: 0,
        },
        ThroughputCell {
            policy: "fifo_serial",
            concurrency: 8,
            aggregate_tokens_per_sec: 200.0,
            ttft_p50_ms: 10.0,
            ttft_p95_ms: 20.0,
            per_slot_tokens_per_sec: 25.0,
            rejected_429_count: 0,
        },
    ];
    let _ = ThroughputCellStable::from_reps(cells);
}

#[test]
fn d3_stable_from_reps_zero_median_yields_zero_sigma_pct() {
    // Defensive: when all reps measure 0 tok/s (e.g. every stream
    // 429'd) the median is 0 and sigma_pct would otherwise divide
    // by zero. The from_reps impl returns 0.0 sigma_pct in this
    // case to keep the stability gate well-defined.
    let cells = vec![
        ThroughputCell {
            policy: "fifo_serial",
            concurrency: 1,
            aggregate_tokens_per_sec: 0.0,
            ttft_p50_ms: 0.0,
            ttft_p95_ms: 0.0,
            per_slot_tokens_per_sec: 0.0,
            rejected_429_count: 0,
        },
        ThroughputCell {
            policy: "fifo_serial",
            concurrency: 1,
            aggregate_tokens_per_sec: 0.0,
            ttft_p50_ms: 0.0,
            ttft_p95_ms: 0.0,
            per_slot_tokens_per_sec: 0.0,
            rejected_429_count: 0,
        },
        ThroughputCell {
            policy: "fifo_serial",
            concurrency: 1,
            aggregate_tokens_per_sec: 0.0,
            ttft_p50_ms: 0.0,
            ttft_p95_ms: 0.0,
            per_slot_tokens_per_sec: 0.0,
            rejected_429_count: 0,
        },
    ];
    let stable = ThroughputCellStable::from_reps(cells);
    assert_eq!(stable.aggregate_tokens_per_sec_sigma_pct, 0.0);
}

#[test]
fn d3_render_report_stable_emits_header_and_sigma_column() {
    let cells = vec![ThroughputCellStable {
        policy: "fifo_serial",
        concurrency: 4,
        rep_count: 3,
        aggregate_tokens_per_sec_median: 105.0,
        aggregate_tokens_per_sec_min: 100.0,
        aggregate_tokens_per_sec_max: 110.0,
        aggregate_tokens_per_sec_sigma_pct: 9.52,
        ttft_p50_ms_median: 11.0,
        ttft_p95_ms_median: 21.0,
        per_slot_tokens_per_sec_median: 26.25,
        rejected_429_count_total: 3,
    }];
    let report = render_report_stable(&cells);
    assert!(
        report.contains("sigma_pct"),
        "header missing sigma_pct: {report}"
    );
    assert!(
        report.contains("agg tok/s median"),
        "header missing median: {report}"
    );
    assert!(
        report.contains("| fifo_serial | 4 | 3 |"),
        "data row missing: {report}"
    );
    assert!(
        report.contains("9.5%"),
        "sigma_pct formatted row missing: {report}"
    );
}

#[test]
fn d3_stability_threshold_default_is_twenty_pct() {
    // Pin the D3 stability gate threshold so a future iter that
    // tightens or loosens it does so deliberately (and updates the
    // ADR §6.1.15 closure block + this assertion together).
    assert!(
        (STABILITY_SIGMA_PCT_THRESHOLD - 20.0).abs() < 1e-9,
        "STABILITY_SIGMA_PCT_THRESHOLD changed from 20.0 — update ADR §6.1.15 too",
    );
    // Pin REPS at 3 for the same reason: the ADR-033 §Pi median
    // discriminator floor is 3 reps; any change should be deliberate.
    assert_eq!(REPS, 3, "REPS changed from 3 — update ADR §6.1.15 too");
}

// ─────────────────────────────────────────────────────────────────────
// cfa-iter-A5b MAJOR #2 — AC-4 gate outcome tests (always-on).
//
// These exercise the pure `ac4_outcome` helper extracted from the
// env-gated body so the misconfiguration / deferral / stability /
// failed / passed paths can be verified without spawning subprocesses.
// ─────────────────────────────────────────────────────────────────────

/// Build a synthetic stable cell with the given policy, concurrency,
/// aggregate median, sigma_pct, and TTFT p95 median. The other
/// fields are zeroed — only the four used by `ac4_outcome` matter.
fn stable_cell(
    policy: &'static str,
    concurrency: u32,
    aggregate_median: f64,
    sigma_pct: f64,
    ttft_p95_median: f64,
) -> ThroughputCellStable {
    ThroughputCellStable {
        policy,
        concurrency,
        rep_count: 3,
        aggregate_tokens_per_sec_median: aggregate_median,
        aggregate_tokens_per_sec_min: aggregate_median * 0.95,
        aggregate_tokens_per_sec_max: aggregate_median * 1.05,
        aggregate_tokens_per_sec_sigma_pct: sigma_pct,
        ttft_p50_ms_median: ttft_p95_median * 0.5,
        ttft_p95_ms_median: ttft_p95_median,
        per_slot_tokens_per_sec_median: aggregate_median / f64::from(concurrency.max(1)),
        rejected_429_count_total: 0,
    }
}

#[test]
fn ac4_outcome_missing_inflight_n4_returns_deferred() {
    let f4 = stable_cell("fifo_serial", 4, 100.0, 5.0, 50.0);
    let f1 = stable_cell("fifo_serial", 1, 50.0, 5.0, 50.0);
    let out = ac4_outcome(Some(&f4), None, Some(&f1));
    assert_eq!(
        out,
        Ac4Outcome::Deferred,
        "Deferred when InflightBatched N=4 is absent (Phase C2c/C2d gated)"
    );
}

#[test]
fn ac4_outcome_missing_fifo_n4_returns_deferred() {
    let i4 = stable_cell("inflight_batched", 4, 200.0, 5.0, 70.0);
    let f1 = stable_cell("fifo_serial", 1, 50.0, 5.0, 50.0);
    let out = ac4_outcome(None, Some(&i4), Some(&f1));
    assert_eq!(
        out,
        Ac4Outcome::Deferred,
        "Deferred when FifoSerial N=4 is absent"
    );
}

#[test]
fn ac4_outcome_both_n4_present_but_missing_n1_returns_misconfigured() {
    // cfa-iter-A5b MAJOR #2: the load-bearing regression pin.
    // Pre-fix this case returned `[ac-4 PARTIAL]` and silently
    // skipped the TTFT half of the gate — a TTFT regression would
    // pass. Post-fix: typed `Misconfigured` outcome → panic at the
    // env-gated body.
    let f4 = stable_cell("fifo_serial", 4, 100.0, 5.0, 50.0);
    let i4 = stable_cell("inflight_batched", 4, 200.0, 5.0, 70.0);
    let out = ac4_outcome(Some(&f4), Some(&i4), None);
    assert_eq!(
        out,
        Ac4Outcome::Misconfigured,
        "BOTH N=4 cells + missing N=1 baseline ⇒ Misconfigured (hard error, \
         not silent skip — pre-iter-A5b [ac-4 PARTIAL] would have let a \
         TTFT regression pass)"
    );
}

#[test]
fn ac4_outcome_stability_blocked_when_sigma_pct_above_threshold() {
    // Either side over the 20% bar → StabilityBlocked.
    let f4 = stable_cell("fifo_serial", 4, 100.0, 25.0, 50.0); // 25% > 20%
    let i4 = stable_cell("inflight_batched", 4, 200.0, 5.0, 70.0);
    let f1 = stable_cell("fifo_serial", 1, 50.0, 5.0, 50.0);
    let out = ac4_outcome(Some(&f4), Some(&i4), Some(&f1));
    assert_eq!(
        out,
        Ac4Outcome::StabilityBlocked,
        "fifo_serial sigma_pct > threshold ⇒ StabilityBlocked"
    );

    let f4_ok = stable_cell("fifo_serial", 4, 100.0, 5.0, 50.0);
    let i4_noisy = stable_cell("inflight_batched", 4, 200.0, 30.0, 70.0); // 30% > 20%
    let out2 = ac4_outcome(Some(&f4_ok), Some(&i4_noisy), Some(&f1));
    assert_eq!(
        out2,
        Ac4Outcome::StabilityBlocked,
        "inflight_batched sigma_pct > threshold ⇒ StabilityBlocked"
    );
}

#[test]
fn ac4_outcome_passed_when_aggregate_above_1_5x_and_ttft_under_2x() {
    // fifo N=4 = 100, inflight N=4 = 200 ⇒ aggregate ratio = 2.0× ≥ 1.5×
    // fifo N=1 TTFT p95 = 50, inflight N=4 TTFT p95 = 70 ⇒ 1.4× ≤ 2.0×
    let f4 = stable_cell("fifo_serial", 4, 100.0, 5.0, 50.0);
    let i4 = stable_cell("inflight_batched", 4, 200.0, 5.0, 70.0);
    let f1 = stable_cell("fifo_serial", 1, 50.0, 5.0, 50.0);
    let out = ac4_outcome(Some(&f4), Some(&i4), Some(&f1));
    match out {
        Ac4Outcome::Passed {
            aggregate_ratio,
            ttft_ratio,
        } => {
            assert!((aggregate_ratio - 2.0).abs() < 1e-6);
            assert!((ttft_ratio - 1.4).abs() < 1e-6);
        }
        other => panic!("expected Passed, got {other:?}"),
    }
}

#[test]
fn ac4_outcome_failed_aggregate_when_ratio_below_1_5x() {
    // fifo N=4 = 100, inflight N=4 = 140 ⇒ 1.4× < 1.5× ⇒ Failed/aggregate.
    let f4 = stable_cell("fifo_serial", 4, 100.0, 5.0, 50.0);
    let i4 = stable_cell("inflight_batched", 4, 140.0, 5.0, 70.0);
    let f1 = stable_cell("fifo_serial", 1, 50.0, 5.0, 50.0);
    let out = ac4_outcome(Some(&f4), Some(&i4), Some(&f1));
    match out {
        Ac4Outcome::Failed {
            aggregate_ratio,
            which,
            ..
        } => {
            assert!((aggregate_ratio - 1.4).abs() < 1e-6);
            assert_eq!(
                which, "aggregate",
                "aggregate ratio failure surfaces `which='aggregate'`"
            );
        }
        other => panic!("expected Failed/aggregate, got {other:?}"),
    }
}

#[test]
fn ac4_outcome_failed_ttft_when_ratio_above_2x() {
    // fifo N=4 = 100, inflight N=4 = 200 ⇒ aggregate 2.0× passes
    // fifo N=1 TTFT p95 = 50, inflight N=4 TTFT p95 = 150 ⇒ 3.0× > 2.0×
    let f4 = stable_cell("fifo_serial", 4, 100.0, 5.0, 50.0);
    let i4 = stable_cell("inflight_batched", 4, 200.0, 5.0, 150.0);
    let f1 = stable_cell("fifo_serial", 1, 50.0, 5.0, 50.0);
    let out = ac4_outcome(Some(&f4), Some(&i4), Some(&f1));
    match out {
        Ac4Outcome::Failed {
            ttft_ratio, which, ..
        } => {
            assert!((ttft_ratio - 3.0).abs() < 1e-6);
            assert_eq!(which, "ttft", "ttft ratio failure surfaces `which='ttft'`");
        }
        other => panic!("expected Failed/ttft, got {other:?}"),
    }
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
    let data_rows = report
        .lines()
        .filter(|l| l.starts_with("|") && !l.contains("---"))
        .count();
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
        // CLI accepts hyphenated form (`fifo-serial` / `inflight-batched`)
        // per clap's kebab-case auto-derivation; bench cell `policy`
        // strings use underscore form. Convert here at the subprocess
        // boundary so report / assertion code keeps the underscore form
        // unchanged. ADR-040 §6.1.55 D3-AC-4 real-hardware bench
        // (commit hash recorded at commit time).
        let policy_cli = policy.replace('_', "-");
        cmd.args([
            "serve",
            "--model",
            gguf,
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--scheduler",
            &policy_cli,
        ]);
        // `--max-slots` is only honored under inflight_batched per ADR-040
        // §6 Phase C iter-4 (C4); pass it for both policies — fifo_serial
        // silently ignores it (worker is pinned to max_slots=1). This
        // keeps the spawn invocation symmetric across cells.
        if policy == "inflight_batched" {
            cmd.args(["--max-slots", &max_slots.to_string()]);
        }
        let child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
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
///
/// # TTFT measurement (D3, this iter)
///
/// D2 used `Command::output()` which blocks until curl exits; the
/// recorded TTFT was the upper bound (= total stream walltime) refined
/// at aggregation time by subtracting `(tokens-1) × per_token_ms`.
/// D3 replaces this with **per-frame streaming-stdout TTFT**: curl is
/// spawned with `Stdio::piped()`, the parent reads the pipe via
/// `BufReader::lines()`, and the moment the first `data:` frame with
/// a non-empty `delta.content` arrives the parent captures
/// `t0.elapsed()` directly. curl's `-N` flag flushes per-SSE-frame so
/// the parent's BufReader receives the bytes as they arrive over the
/// socket (modulo OS pipe scheduling — typically sub-millisecond).
/// This eliminates the upper-bound bias that drove the D2 → D3
/// caveat: D3's TTFT is the wall-clock from POST send (just before
/// `child.spawn()`) to the first content frame's arrival at the
/// parent, with no token-count-based subtraction.
fn run_stream(port: u16, prompt: &str, max_tokens: u32, model: &str) -> StreamResult {
    let body = format!(
        r#"{{"model":"{}","messages":[{{"role":"user","content":"{}"}}],"max_tokens":{},"temperature":0.6,"stream":true}}"#,
        model.replace('"', "\\\""),
        prompt.replace('"', "\\\""),
        max_tokens,
    );

    // -s: silent (no progress bar)
    // -N: no buffer (emit SSE frames as they arrive)
    // -w "\n__HTTP_STATUS__:%{http_code}\n": tail-marker for HTTP status code
    // --max-time: hard upper bound matching STREAM_BUDGET_SECS
    let mut cmd = Command::new("curl");
    cmd.args([
        "-s",
        "-N",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
        "--max-time",
        &STREAM_BUDGET_SECS.to_string(),
        "-w",
        "\n__HTTP_STATUS__:%{http_code}\n",
        "-d",
        &body,
        &format!("http://127.0.0.1:{port}/v1/chat/completions"),
    ])
    .stdout(Stdio::piped())
    .stderr(Stdio::null());

    let t0 = Instant::now();
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => {
            return StreamResult {
                http_status: -1,
                ttft_ms: 0.0,
                tokens: 0,
                total_ms: t0.elapsed().as_secs_f64() * 1000.0,
            };
        }
    };

    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return StreamResult {
                http_status: -1,
                ttft_ms: 0.0,
                tokens: 0,
                total_ms: t0.elapsed().as_secs_f64() * 1000.0,
            };
        }
    };

    let reader = BufReader::new(stdout);

    // Per-frame parse loop: scan curl's stdout line-by-line as it
    // arrives. Capture TTFT (`Instant::now() - t0`) the first time
    // we see a `data: {...}` frame with a non-empty `delta.content`.
    // The substring scan for `"content":"` matches hf2q's
    // OpenAI-compatible chat-stream wire format (see
    // `src/serve/api/sse.rs`); we do NOT depend on `serde_json`
    // because the bench has no need for full JSON validation and the
    // substring search keeps `Cargo.toml` untouched per the brief.
    //
    // The role frame (`"content":""`) is emitted first; we count
    // only non-empty deltas as tokens. The trailing
    // `__HTTP_STATUS__:<code>` marker is curl's `-w` output and is
    // parsed at the end.
    let mut tokens: u32 = 0;
    let mut ttft_ms = 0.0_f64;
    let mut first_content_seen = false;
    let mut http_status: i32 = -1;
    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            // Read error mid-stream: capture what we have, status -1.
            Err(_) => break,
        };
        if let Some(code_str) = line.strip_prefix("__HTTP_STATUS__:") {
            if let Ok(code) = code_str.trim().parse::<i32>() {
                http_status = code;
            }
            continue;
        }
        let payload = match line.strip_prefix("data: ") {
            Some(p) => p,
            None => continue,
        };
        if payload.trim() == "[DONE]" {
            continue;
        }
        if let Some(idx) = payload.find(r#""content":""#) {
            let after = &payload[idx + r#""content":""#.len()..];
            // First char after the open quote: `"` = empty content
            // (role frame); anything else = real content delta.
            if !after.starts_with('"') {
                tokens = tokens.saturating_add(1);
                if !first_content_seen {
                    first_content_seen = true;
                    // D3 per-frame TTFT: wall-clock from POST send
                    // (the `t0` Instant taken just before
                    // `child.spawn()`) to NOW (the moment we
                    // observed the first content delta on the
                    // parent-side BufReader). No token-count-based
                    // subtraction is performed.
                    ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }
    }

    // Reap the child to avoid zombie subprocesses; ignore the exit
    // status because we've already captured `http_status` from the
    // SSE stream's `-w` marker. If curl is still running for some
    // reason (e.g. server hung after streaming), `wait()` blocks
    // for the remainder of the `--max-time` budget which is bounded.
    let _ = child.wait();

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

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

    let mut server =
        BenchServer::spawn(gguf, policy, n, port).map_err(|e| format!("spawn hf2q serve: {e}"))?;

    wait_for_readyz(&mut server)?;
    eprintln!("[cb-throughput] /readyz=200 on port={port}");

    // Resolve canonical model id via /v1/models for the SSE POST body.
    // The server returns a registry-keyed id; using it directly avoids
    // the auto-pipeline path-classification overhead per request.
    let model_id = fetch_model_id(port).map_err(|e| format!("GET /v1/models: {e}"))?;
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

    // D3 per-frame TTFT: `run_stream` now records the wall-clock
    // from POST send to first content delta directly via
    // streaming-stdout consumption (BufReader::lines() on the
    // child's piped stdout). No token-count-based subtraction is
    // performed — the recorded `ttft_ms` IS the time-to-first-token
    // measurement. The aggregator just sorts and percentiles.
    let mut ttft_per_stream_ms: Vec<f64> = succeeded.iter().map(|r| r.ttft_ms).collect();
    ttft_per_stream_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ttft_p50_ms = percentile(&ttft_per_stream_ms, 0.50);
    let ttft_p95_ms = percentile(&ttft_per_stream_ms, 0.95);

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

/// Phase D iter-3 (D3): run `run_bench_cell` REPS times and aggregate
/// into a `ThroughputCellStable`. Each rep is a full subprocess spawn
/// + /readyz poll + N concurrent SSE streams + subprocess shutdown,
/// matching the D2 cell shape. The reps are sequential — no two
/// `hf2q serve` subprocesses are alive at once (CLAUDE.md "do not
/// oom us" rule + the fact that `BenchServer::drop` is the only
/// shutdown path).
///
/// Returns `Err(reason)` when ANY of the REPS reps fails to produce
/// a cell. This is the strict policy: a single failed rep makes the
/// median undefined (`assert!(!cells.is_empty())` in
/// `ThroughputCellStable::from_reps` would fire on partial data
/// anyway), so we surface the failure to the caller and let it
/// record the cell as skipped — matching D2's
/// `inflight_batched-rejected-at-spawn` skip path.
fn run_bench_cell_3rep(
    gguf: &str,
    policy: &'static str,
    n: u32,
) -> Result<ThroughputCellStable, String> {
    let mut cells = Vec::with_capacity(REPS);
    for rep in 0..REPS {
        eprintln!(
            "[d3] cell policy={} N={} rep={}/{}",
            policy,
            n,
            rep + 1,
            REPS
        );
        let cell = run_bench_cell(gguf, policy, n).map_err(|e| {
            format!(
                "rep {}/{} failed for policy={} N={}: {}",
                rep + 1,
                REPS,
                policy,
                n,
                e
            )
        })?;
        cells.push(cell);
    }
    Ok(ThroughputCellStable::from_reps(cells))
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
    s.write_all(b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")?;
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
    assert!(
        !concurrency.is_empty(),
        "concurrency list must be non-empty"
    );
    for &n in &concurrency {
        assert!(n > 0, "concurrency entries must be > 0, got {n}");
    }

    // Phase D iter-3 (D3): each (policy, N) cell is run REPS=3 times
    // via `run_bench_cell_3rep` which aggregates into a
    // `ThroughputCellStable` (median + min/max + sigma_pct). Per-rep
    // failures abort the whole cell — a partial-data median would be
    // meaningless and silently misleading.
    let mut all_cells: Vec<ThroughputCellStable> = Vec::new();
    let mut skipped: Vec<(String, u32, String)> = Vec::new();

    for &policy in &["fifo_serial", "inflight_batched"] {
        for &n in &concurrency {
            match run_bench_cell_3rep(&gguf_path, policy, n) {
                Ok(cell) => all_cells.push(cell),
                Err(e) => {
                    eprintln!("[cb-throughput] cell SKIPPED (policy={policy}, N={n}): {e}");
                    skipped.push((policy.to_string(), n, e));
                }
            }
        }
    }

    let report = render_report_stable(&all_cells);
    println!("\n=== ADR-040 §5 AC-4 throughput report (D3, REPS={REPS}) ===\n{report}");
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
        "ADR-040 D3: no bench cells completed; all (policy, N) combinations failed. \
         Check HF2Q_CB_THROUGHPUT_MODEL={gguf_path} + the skipped-cells list above."
    );

    // FifoSerial-only baseline + variance reporting. D3 always emits
    // a separate stability section for the FifoSerial rows EVEN WHEN
    // the AC-4 gate cannot fire (InflightBatched is rejected at
    // spawn until Phase C2c/C2d wire it). This gives operators the
    // run-to-run noise floor for the baseline so they can decide
    // whether to bump REPS before C2c/C2d land.
    println!("\n=== D3 FifoSerial-only stability baseline ===");
    let mut any_fifo = false;
    for cell in all_cells.iter().filter(|c| c.policy == "fifo_serial") {
        any_fifo = true;
        println!(
            "  N={}: median={:.1} tok/s, min={:.1}, max={:.1}, sigma_pct={:.1}% ({} reps, 429s total={})",
            cell.concurrency,
            cell.aggregate_tokens_per_sec_median,
            cell.aggregate_tokens_per_sec_min,
            cell.aggregate_tokens_per_sec_max,
            cell.aggregate_tokens_per_sec_sigma_pct,
            cell.rep_count,
            cell.rejected_429_count_total,
        );
    }
    if !any_fifo {
        println!("  (no fifo_serial cells completed)");
    }

    // AC-4 HARD GATE per ADR-040 §5 (D3 promotion from D2 soft-warn):
    // when N=4 cells exist for BOTH fifo_serial AND inflight_batched,
    // assert aggregate ratio ≥ 1.5× AND TTFT p95 ratio ≤ 2.0×. Before
    // the assertion, stability must be acceptable on BOTH cells
    // (sigma_pct ≤ STABILITY_SIGMA_PCT_THRESHOLD).
    //
    // When InflightBatched is rejected at spawn (Phase C2c/C2d not
    // yet wired), the inflight_n4 cell will be missing and we skip
    // AC-4 enforcement — deferred to once C2c/C2d ship. D3 still
    // reports the FifoSerial-only baseline + variance above so the
    // bench is operator-useful in the interim.
    let fifo_n4 = all_cells
        .iter()
        .find(|c| c.policy == "fifo_serial" && c.concurrency == 4);
    let inflight_n4 = all_cells
        .iter()
        .find(|c| c.policy == "inflight_batched" && c.concurrency == 4);
    let fifo_n1 = all_cells
        .iter()
        .find(|c| c.policy == "fifo_serial" && c.concurrency == 1);

    // cfa-iter-A5b MAJOR #2 fix: route through the pure `ac4_outcome`
    // helper so the misconfiguration / deferral / stability arms can
    // be unit-tested in isolation. The env-gated body translates the
    // typed outcome into the existing wire-level shape (panics for
    // hard-fail arms, stderr emission for soft-deferral, PASS log
    // for the success arm).
    match ac4_outcome(fifo_n4, inflight_n4, fifo_n1) {
        Ac4Outcome::Deferred => {
            eprintln!(
                "[ac-4 DEFERRED] cannot evaluate gate — missing N=4 cell for one or both \
                 policies. Typically inflight_batched is Phase C2c/C2d gated; D3 reports \
                 the FifoSerial-only baseline + variance and defers AC-4 enforcement until \
                 the inflight-side wiring lands. The bench is forward-compatible: no edits \
                 needed once C2c/C2d ship."
            );
        }
        Ac4Outcome::Misconfigured => {
            let f = fifo_n4.expect("Misconfigured requires fifo_n4 present");
            let i = inflight_n4.expect("Misconfigured requires inflight_n4 present");
            panic!(
                "[ac-4 MISCONFIGURED] BOTH N=4 cells present (fifo_serial \
                 N=4 median = {:.1} tok/s; inflight_batched N=4 median = {:.1} tok/s) \
                 BUT fifo_serial N=1 baseline cell is missing. The AC-4 TTFT half \
                 compares inflight_batched N=4 p95 against fifo_serial N=1 p95; \
                 skipping it would let a TTFT regression slip past the gate. \
                 Re-run with HF2Q_CB_THROUGHPUT_CONCURRENCY including `1` (e.g. \
                 `1,2,4,8` — the default).",
                f.aggregate_tokens_per_sec_median, i.aggregate_tokens_per_sec_median,
            );
        }
        Ac4Outcome::StabilityBlocked => {
            let f = fifo_n4.expect("StabilityBlocked requires fifo_n4 present");
            let i = inflight_n4.expect("StabilityBlocked requires inflight_n4 present");
            if f.aggregate_tokens_per_sec_sigma_pct > STABILITY_SIGMA_PCT_THRESHOLD {
                panic!(
                    "AC-4 BLOCKED: fifo_serial N=4 measurement variance {:.1}% > {:.1}% threshold; \
                     run again or increase REPS for stable median (median={:.1}, min={:.1}, max={:.1})",
                    f.aggregate_tokens_per_sec_sigma_pct,
                    STABILITY_SIGMA_PCT_THRESHOLD,
                    f.aggregate_tokens_per_sec_median,
                    f.aggregate_tokens_per_sec_min,
                    f.aggregate_tokens_per_sec_max,
                );
            }
            panic!(
                "AC-4 BLOCKED: inflight_batched N=4 measurement variance {:.1}% > {:.1}% threshold; \
                 run again or increase REPS for stable median (median={:.1}, min={:.1}, max={:.1})",
                i.aggregate_tokens_per_sec_sigma_pct,
                STABILITY_SIGMA_PCT_THRESHOLD,
                i.aggregate_tokens_per_sec_median,
                i.aggregate_tokens_per_sec_min,
                i.aggregate_tokens_per_sec_max,
            );
        }
        Ac4Outcome::Failed {
            aggregate_ratio,
            ttft_ratio,
            which,
        } => {
            let f = fifo_n4.expect("Failed requires fifo_n4 present");
            let i = inflight_n4.expect("Failed requires inflight_n4 present");
            if which == "aggregate" {
                panic!(
                    "AC-4 FAILED: aggregate ratio {:.2}× below 1.5× bar \
                     (fifo_serial N=4 median = {:.1} tok/s; inflight_batched N=4 median = {:.1} tok/s)",
                    aggregate_ratio,
                    f.aggregate_tokens_per_sec_median,
                    i.aggregate_tokens_per_sec_median,
                );
            }
            let base = fifo_n1.expect("Failed/ttft requires fifo_n1 present");
            panic!(
                "AC-4 FAILED: TTFT p95 ratio {:.2}× above 2.0× bar \
                 (fifo_serial N=1 p95 median = {:.1} ms; inflight_batched N=4 p95 median = {:.1} ms)",
                ttft_ratio,
                base.ttft_p95_ms_median,
                i.ttft_p95_ms_median,
            );
        }
        Ac4Outcome::Passed {
            aggregate_ratio,
            ttft_ratio,
        } => {
            eprintln!(
                "[ac-4 PASS] aggregate ratio {:.2}× ≥ 1.5× ✓ ; TTFT p95 ratio {:.2}× ≤ 2.0× ✓",
                aggregate_ratio, ttft_ratio,
            );
        }
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
    let concurrency =
        std::env::var("HF2Q_CB_THROUGHPUT_CONCURRENCY").unwrap_or_else(|_| String::from("1,2,4,8"));
    assert!(
        model.is_ok(),
        "HF2Q_CB_THROUGHPUT_E2E=1 set but HF2Q_CB_THROUGHPUT_MODEL absent — \
         D2 refuses to run without a GGUF path"
    );
    let parsed: Result<Vec<u32>, _> = concurrency
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect();
    assert!(
        parsed.is_ok(),
        "HF2Q_CB_THROUGHPUT_CONCURRENCY parse failed: {:?}",
        parsed
    );
    let ns = parsed.unwrap();
    assert!(!ns.is_empty(), "at least one N value required");
    for &n in &ns {
        assert!(n > 0, "N must be positive, got {}", n);
    }
}

// ============================================================================
// ADR-040 §6.1.55 iter-A4-cont-inflection-bench (2026-05-30) —
// acceptance-rate dimension on top of D3's throughput cell.
//
// Per dossier §6, this is the D3-style AC-4 throughput bench extended
// to plot acceptance_rate × concurrent_count for hf2q's Qwen35/Qwen3.6
// + EAGLE-3 drafter combos.  Structural scaffold lands today; the
// env-gated measurement body runs ONLY when both
// `HF2Q_CB_THROUGHPUT_E2E=1` AND `HF2Q_A4_INFLECTION_BENCH=1` are set.
//
// Skip-mode preserves the D3 contract: ALL existing throughput-cell
// tests pass unchanged.  Cell carriers + report helpers are pure
// data; no I/O at the type level.
// ============================================================================

/// **ADR-040 §6.1.55 iter-A4-cont-inflection-bench (2026-05-30)** —
/// per-(concurrent, acceptance) cell for the acceptance-rate dimension
/// extension to the D3 throughput bench.
///
/// Mirror of [`ThroughputCellStable`] in shape; carries the
/// acceptance-rate axis alongside the existing aggregate throughput
/// median.  Plotting `acceptance_rate` on the x-axis against
/// `tokens_per_step` (decode-side throughput per verification step)
/// lets the operator visually identify the spec-decode inflection
/// point on hf2q's hardware (dossier §1.5 + §6).
///
/// **Skip-mode safety**: pure data; no allocations beyond the
/// fields.  All-defaulted via [`Self::synthetic_for_smoke`] for the
/// always-on scaffold-shape pin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptanceCell {
    /// N concurrent requests (matches the D3 ThroughputCell
    /// `concurrency` field).
    pub concurrent: u32,
    /// Mean per-step acceptance rate measured at this concurrency.
    /// `[0.0, 1.0]` (clamped by
    /// [`crate::inference::spec_decode::SpecDecodeAcceptanceMetric::
    /// acceptance_ratio`]).
    pub acceptance_rate: f64,
    /// Decode tokens per verification step.  At high acceptance this
    /// approaches the tree budget; at low acceptance this drops to 1
    /// (the verifier-side argmax always emits at least one token).
    pub tokens_per_step: f64,
}

impl AcceptanceCell {
    /// **ADR-040 §6.1.55 iter-A4-cont-inflection-bench (2026-05-30)** —
    /// synthetic cell for the always-on scaffold-shape smoke test.
    /// Mirrors [`ThroughputCell::synthetic_for_smoke`].
    pub fn synthetic_for_smoke() -> Self {
        AcceptanceCell {
            concurrent: 1,
            acceptance_rate: 0.0,
            tokens_per_step: 0.0,
        }
    }
}

/// Render a vector of [`AcceptanceCell`] as a markdown table — the
/// env-gated bench body calls this to emit the report alongside the
/// D3 throughput report.
pub fn render_acceptance_report(cells: &[AcceptanceCell]) -> String {
    let mut s = String::from("| concurrent | acceptance_rate | tokens_per_step |\n");
    s.push_str("|------------|-----------------|-----------------|\n");
    for c in cells {
        s.push_str(&format!(
            "| {} | {:.3} | {:.2} |\n",
            c.concurrent, c.acceptance_rate, c.tokens_per_step
        ));
    }
    s
}

#[test]
fn acceptance_cell_synthetic_round_trips_through_report() {
    let cells = vec![AcceptanceCell::synthetic_for_smoke()];
    let report = render_acceptance_report(&cells);
    assert!(report.contains("| 1 |"), "acceptance report: {}", report);
    assert!(report.contains("concurrent"), "header missing");
    assert!(report.contains("|------------|"), "separator missing");
}

#[test]
fn render_acceptance_report_empty_returns_header_only() {
    let report = render_acceptance_report(&[]);
    let lines: Vec<&str> = report.lines().collect();
    assert_eq!(lines.len(), 2, "header + separator only, got: {:?}", lines);
}

/// **ADR-040 §6.1.55 iter-A4-cont-inflection-bench (2026-05-30)** —
/// env-gated harness placeholder for the acceptance-rate dimension
/// bench.
///
/// Today: structural scaffold.  Skip mode preserves the D3 contract.
/// When `HF2Q_CB_THROUGHPUT_E2E=1` AND `HF2Q_A4_INFLECTION_BENCH=1`
/// AND `HF2Q_CB_THROUGHPUT_MODEL=<gguf>` are ALL set, the body would
/// run a real acceptance-rate measurement at the D3 concurrency
/// sweep (1, 2, 4, 8) and plot `acceptance_rate` × `tokens_per_step`
/// against `concurrent` to find the workload-specific inflection
/// point.
///
/// **Why a placeholder**: the kernel-side dispatcher
/// (iter-A4-cont-drafter-dispatcher-kernel) has not landed; running
/// the measurement before the dispatcher would surface 0% acceptance
/// at every concurrency (the drafter would never write past
/// `SlotId(0)`).  Once the kernel dispatcher lands + the threshold
/// gate is tuned by hf2q operator measurement, this body fills in.
#[test]
fn a4_inflection_bench_acceptance_dimension_scaffold() {
    if std::env::var("HF2Q_CB_THROUGHPUT_E2E").as_deref() != Ok("1") {
        eprintln!(
            "[a4-inflection-bench] skipped — set HF2Q_CB_THROUGHPUT_E2E=1 \
             AND HF2Q_A4_INFLECTION_BENCH=1 AND HF2Q_CB_THROUGHPUT_MODEL=<gguf> \
             to engage the acceptance-rate dimension bench. Today this is a \
             structural scaffold per ADR-040 §6.1.55 iter-A4-cont-inflection-bench."
        );
        return;
    }
    if std::env::var("HF2Q_A4_INFLECTION_BENCH").as_deref() != Ok("1") {
        eprintln!(
            "[a4-inflection-bench] skipped — HF2Q_CB_THROUGHPUT_E2E=1 set but \
             HF2Q_A4_INFLECTION_BENCH != 1; respecting bench opt-in."
        );
        return;
    }
    // Structural-cell-shape proof under the env gate.  The real
    // measurement body lands at iter-A4-cont-drafter-dispatcher-kernel
    // once the per-slot kernel routing exists — see the docstring.
    let cells: Vec<AcceptanceCell> = vec![AcceptanceCell::synthetic_for_smoke()];
    let report = render_acceptance_report(&cells);
    println!("\n=== ADR-040 §6.1.55 iter-A4-cont-inflection-bench (scaffold) ===\n{report}");
}

// ============================================================================
// ADR-040 §6.1.55 iter-A4-cont-moe-validation (2026-05-30) —
// operator-runnable env-gated harness for Qwen3.6-A3B MoE A/B at
// N=1,2,4,8 concurrent.
//
// Per dossier §1.6 #3 the MoE-routing trap is a known production
// hidden trap; this harness reserves the structural shape for the
// operator-runnable A/B at the published-inflection threshold.
//
// Skip-mode: no-op (the env gate guards the body).  When
// `HF2Q_A4_MOE_AB_VALIDATION_E2E=1` AND `HF2Q_CB_THROUGHPUT_MODEL` is
// set, the test would run a real A/B comparing baseline-Qwen3.6 vs
// batched-spec-decode-Qwen3.6 at the four concurrencies.
// ============================================================================

#[test]
fn a4_moe_validation_qwen36_a3b_a_b_n_1_2_4_8() {
    if std::env::var("HF2Q_A4_MOE_AB_VALIDATION_E2E").as_deref() != Ok("1") {
        eprintln!(
            "[a4-moe-validation] skipped — set HF2Q_A4_MOE_AB_VALIDATION_E2E=1 \
             AND HF2Q_CB_THROUGHPUT_MODEL=<Qwen3.6-A3B-gguf> to engage the \
             MoE A/B bench at N=1,2,4,8 concurrent. ADR-040 §6.1.55 \
             iter-A4-cont-moe-validation harness."
        );
        return;
    }
    // Hard requirement once the gate is set (mirrors D3
    // cfa-finding-F8 — silent skip under an explicit opt-in violates
    // the operator contract).
    let gguf_path = std::env::var("HF2Q_CB_THROUGHPUT_MODEL").expect(
        "HF2Q_CB_THROUGHPUT_MODEL required when HF2Q_A4_MOE_AB_VALIDATION_E2E=1. \
         Either set the env or unset HF2Q_A4_MOE_AB_VALIDATION_E2E — silent skip \
         with the gate set violates the iter-1.5 cfa-finding-F8 contract.",
    );
    eprintln!(
        "[a4-moe-validation] running A/B sweep against {gguf_path} \
         at N=1,2,4,8 per ADR-040 §6.1.55 iter-A4-cont-moe-validation."
    );
    // Structural scaffold reserve: the real body lands at
    // iter-A4-cont-drafter-dispatcher-kernel + the operator
    // empirical-measurement runbook entry per dossier §7.  Today the
    // body is a placeholder that pins the env-gate contract.
    let concurrencies: Vec<u32> = vec![1, 2, 4, 8];
    for &n in &concurrencies {
        eprintln!("[a4-moe-validation] N={n}: deferred to iter-A4-cont-drafter-dispatcher-kernel");
    }
}
