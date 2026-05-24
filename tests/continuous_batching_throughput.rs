//! ADR-040 Phase D iter-1 — continuous batching throughput benchmark scaffolding.
//!
//! # Scope
//!
//! This test file is the env-gated harness for measuring aggregate
//! tokens/sec across N concurrent SSE streams under each of the two
//! `SchedulerPolicy` modes. Phase D iter-1 (this iter) ships the
//! scaffolding + metric shapes + always-on smoke; iter-2 wires the
//! real measurement body once Phases A, B, and C iter-2+ have landed
//! enough plumbing to make `SchedulerPolicy::InflightBatched`
//! functional.
//!
//! # Env gates
//!
//! - `HF2Q_CB_THROUGHPUT_E2E=1` — enables the env-gated measurement
//!   bodies. When unset, those tests document-skip with a one-line
//!   "Phase D iter-2 implementation pending" message.
//! - `HF2Q_CB_THROUGHPUT_MODEL` — path to GGUF for the measurement.
//!   Required when E2E gate is set.
//! - `HF2Q_CB_THROUGHPUT_CONCURRENCY` — comma-separated list of N
//!   values (default "1,2,4,8" per ADR-040 §5 AC-4).
//!
//! # Metric report shape
//!
//! Per (N, policy) cell:
//!   - aggregate_tokens_per_sec   (sum across streams)
//!   - ttft_p50_ms                (time to first token, p50 across streams)
//!   - ttft_p95_ms                (time to first token, p95 across streams)
//!   - per_slot_tokens_per_sec    (median across streams)
//!   - rejected_429_count         (count of 429 responses during the window)

use std::path::PathBuf;
use std::process::Command;

/// Per-cell metric shape. Phase D iter-2 fills these in; iter-1 ships
/// the type + the report formatter so iter-2's data lands cleanly.
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

/// Render a vector of cells as a markdown table — iter-2 calls this to emit
/// the bench report to stdout / a results file.
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
// Env-gated measurement bodies (Phase D iter-2 implementation pending)
// ========================================================================

#[test]
fn cb_throughput_n_1_2_4_8_fifo_vs_inflight() {
    // Phase D iter-1: env-gated stub. Iter-2 fills in the body once
    // Phase A iter-2+ multi-seq KV impls + Phase B iter-3 InflightBatched
    // step + Phase C iter-2 Engine slot-aware wiring all land.
    if std::env::var("HF2Q_CB_THROUGHPUT_E2E").as_deref() != Ok("1") {
        eprintln!("[cb-throughput] skipped (HF2Q_CB_THROUGHPUT_E2E != 1)");
        return;
    }
    // Iter-1 contract: when the env gate is set but the upstream phases
    // aren't ready yet, document-skip with a clear, actionable message
    // rather than producing a misleading zero result.
    eprintln!(
        "[cb-throughput] ADR-040 Phase D iter-2 implementation pending; \
         requires Phases A iter-2+, B iter-3+, C iter-2+ to land first."
    );
}

#[test]
fn cb_throughput_required_env_vars_documented() {
    // Always-on: catalogs the env vars iter-2 will consume + asserts the
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
         iter-2 will refuse to run without a GGUF path"
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
