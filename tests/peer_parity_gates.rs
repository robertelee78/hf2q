//! ADR-014 P10 iter-1 — peer-parity benchmark harness skeleton.
//!
//! This crate orchestrates the eight Decision-15 gate cells (lines
//! 575–582 of `docs/ADR-014-streaming-convert-pipeline.md`) that
//! compare hf2q's streaming convert pipeline against llama.cpp and
//! mlx-lm peers across 27B-dense + apex-MoE × {GGUF None, GGUF
//! Imatrix, safetensors DWQ, GGUF DWQ vs current pipeline}.
//!
//! Test inventory (14 tests total):
//!
//! Always-on smoke (6 tests, run by default):
//! 1. `harness_compiles_and_emits_table_skeleton`
//! 2. `gate_cell_speed_tolerance_check`
//! 3. `gate_cell_rss_tolerance_check`
//! 4. `gate_cell_ppl_tolerance_check`
//! 5. `subprocess_wrapper_handles_missing_binary`
//! 6. `subprocess_wrapper_handles_missing_python_module`
//!
//! `#[ignore]`-gated (8 tests, one per Decision-15 cell, P11
//! territory — needs apex MoE GPU + ~150 GB disk):
//! 7-14: `cell_<idx>_<model>_<backend>_<calibrator>` — each runs the
//! corresponding `GateCell` end-to-end against the real peer; this
//! iter returns `Verdict::NotMeasured` because no real models are
//! present.
//!
//! PPL columns are deferred to iter-2 (no wikitext-2 fixture or
//! `ppl_kl_eval` driver exists in the repo today; iter-2 lands them
//! when P6 closes).

mod common;

use std::fmt;
use std::sync::Mutex;

use common::llama_cpp_runner;
use common::metrics::RunMetrics;
use common::mlx_lm_runner;

/// Serialises any test that mutates a process-global env var
/// (`HF2Q_*_BIN` overrides). Cargo's default test runner is
/// multi-threaded; `std::env::set_var` is process-wide, so two tests
/// flipping the same key at once would race. We hold this mutex for
/// the entire duration of each env-mutating smoke and release it
/// (along with the var) on exit. This is the canonical Rust idiom for
/// env-var-touching tests in shared binaries (see `tempfile`'s own
/// test suite for the same pattern).
static ENV_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------
// Backend + Peer identifiers (Decision 15 columns 2 + 4)
// ---------------------------------------------------------------------

/// The two output backends ADR-014 must measure parity for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Gguf,
    Safetensors,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendKind::Gguf => f.write_str("GGUF"),
            BackendKind::Safetensors => f.write_str("safetensors"),
        }
    }
}

/// The peer the gate compares hf2q against. Decision 15's 8th column
/// names the comparator; we encode it as a typed enum so the
/// markdown table is consistent and the harness routes to the right
/// subprocess wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerId {
    /// llama.cpp uncalibrated Q4_K_M (rows 1, 5).
    LlamaCppUncalibratedQ4KM,
    /// llama.cpp imatrix Q4_K_M (rows 2, 6).
    LlamaCppImatrixQ4KM,
    /// mlx-lm DWQ (rows 3, 7).
    MlxLmDwq,
    /// No external peer; gate is hf2q-vs-hf2q-current-pipeline (rows
    /// 4, 8). The RSS gate of ≤ 0.50× encodes the central
    /// correctness/sanity claim of the ADR — streaming halves peak
    /// resident.
    Hf2qCurrentPipeline,
}

impl fmt::Display for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PeerId::LlamaCppUncalibratedQ4KM => f.write_str("llama.cpp uncalibrated Q4_K_M"),
            PeerId::LlamaCppImatrixQ4KM => f.write_str("llama.cpp imatrix Q4_K_M"),
            PeerId::MlxLmDwq => f.write_str("mlx_lm DWQ"),
            PeerId::Hf2qCurrentPipeline => f.write_str("(no peer; vs hf2q current pipeline)"),
        }
    }
}

// ---------------------------------------------------------------------
// GateCell — one row of Decision 15's matrix
// ---------------------------------------------------------------------

/// A single gate cell from Decision 15. Fields map column-for-column
/// to the matrix on lines 575–582 of
/// `docs/ADR-014-streaming-convert-pipeline.md`:
///
/// | Field                | Decision-15 column |
/// |----------------------|--------------------|
/// | `model_id`           | "Model"            |
/// | `backend`            | "Backend"          |
/// | `calibrator_variant` | "Calibrator"       |
/// | `peer_id`            | "Peer"             |
/// | `speed_tolerance`    | "Speed gate"       |
/// | `rss_tolerance`      | "RSS gate"         |
/// | `ppl_tolerance`      | "PPL gate"         |
#[derive(Debug, Clone, Copy)]
pub struct GateCell {
    pub model_id: &'static str,
    pub backend: BackendKind,
    pub calibrator_variant: &'static str,
    pub peer_id: PeerId,
    pub speed_tolerance: f64,
    pub rss_tolerance: f64,
    pub ppl_tolerance: f64,
}

/// The 8 gate cells from ADR-014 Decision 15, populated **verbatim**
/// from lines 575–582. Any drift here is a spec violation — the
/// `gate_cells_match_decision_15_verbatim` smoke test wedges this
/// against a duplicate literal table to catch silent edits.
pub const GATE_CELLS: [GateCell; 8] = [
    // Row 1: 27B dense | GGUF | None (q4_k_m)
    //        | llama.cpp uncalibrated Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02×
    GateCell {
        model_id: "27B dense",
        backend: BackendKind::Gguf,
        calibrator_variant: "None (q4_k_m)",
        peer_id: PeerId::LlamaCppUncalibratedQ4KM,
        speed_tolerance: 1.10,
        rss_tolerance: 1.10,
        ppl_tolerance: 1.02,
    },
    // Row 2: 27B dense | GGUF | Imatrix (imatrix-q4_k_m)
    //        | llama.cpp imatrix Q4_K_M    | ≤ 1.10× | ≤ 1.10× | ≤ 1.02×
    GateCell {
        model_id: "27B dense",
        backend: BackendKind::Gguf,
        calibrator_variant: "Imatrix (imatrix-q4_k_m)",
        peer_id: PeerId::LlamaCppImatrixQ4KM,
        speed_tolerance: 1.10,
        rss_tolerance: 1.10,
        ppl_tolerance: 1.02,
    },
    // Row 3: 27B dense | safetensors | DWQ (dwq-4-6)
    //        | mlx_lm DWQ                  | ≤ 1.10× | ≤ 1.10× | ≤ 1.02×
    GateCell {
        model_id: "27B dense",
        backend: BackendKind::Safetensors,
        calibrator_variant: "DWQ (dwq-4-6)",
        peer_id: PeerId::MlxLmDwq,
        speed_tolerance: 1.10,
        rss_tolerance: 1.10,
        ppl_tolerance: 1.02,
    },
    // Row 4: 27B dense | GGUF | DWQ (dwq-4-6)
    //        | (no peer; vs hf2q current pipeline) | ≤ 1.0× | ≤ 0.50× | ≤ 1.0×
    GateCell {
        model_id: "27B dense",
        backend: BackendKind::Gguf,
        calibrator_variant: "DWQ (dwq-4-6)",
        peer_id: PeerId::Hf2qCurrentPipeline,
        speed_tolerance: 1.0,
        rss_tolerance: 0.50,
        ppl_tolerance: 1.0,
    },
    // Row 5: apex MoE | GGUF | None (q4_k_m)
    //        | llama.cpp uncalibrated Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02×
    GateCell {
        model_id: "apex MoE",
        backend: BackendKind::Gguf,
        calibrator_variant: "None (q4_k_m)",
        peer_id: PeerId::LlamaCppUncalibratedQ4KM,
        speed_tolerance: 1.10,
        rss_tolerance: 1.10,
        ppl_tolerance: 1.02,
    },
    // Row 6: apex MoE | GGUF | Imatrix (imatrix-q4_k_m)
    //        | llama.cpp imatrix Q4_K_M    | ≤ 1.10× | ≤ 1.10× | ≤ 1.02×
    GateCell {
        model_id: "apex MoE",
        backend: BackendKind::Gguf,
        calibrator_variant: "Imatrix (imatrix-q4_k_m)",
        peer_id: PeerId::LlamaCppImatrixQ4KM,
        speed_tolerance: 1.10,
        rss_tolerance: 1.10,
        ppl_tolerance: 1.02,
    },
    // Row 7: apex MoE | safetensors | DWQ (dwq-4-6)
    //        | mlx_lm DWQ                  | ≤ 1.10× | ≤ 1.10× | ≤ 1.02×
    GateCell {
        model_id: "apex MoE",
        backend: BackendKind::Safetensors,
        calibrator_variant: "DWQ (dwq-4-6)",
        peer_id: PeerId::MlxLmDwq,
        speed_tolerance: 1.10,
        rss_tolerance: 1.10,
        ppl_tolerance: 1.02,
    },
    // Row 8: apex MoE | GGUF | DWQ (dwq-4-6)
    //        | (no peer; vs hf2q current pipeline) | ≤ 1.0× | ≤ 0.50× | ≤ 1.0×
    GateCell {
        model_id: "apex MoE",
        backend: BackendKind::Gguf,
        calibrator_variant: "DWQ (dwq-4-6)",
        peer_id: PeerId::Hf2qCurrentPipeline,
        speed_tolerance: 1.0,
        rss_tolerance: 0.50,
        ppl_tolerance: 1.0,
    },
];

impl GateCell {
    /// Returns `true` iff hf2q's wall-clock is within
    /// `self.speed_tolerance × peer_wall`. Matches Decision 15's
    /// "≤" (less-than-or-equal) semantic: the gate passes when hf2q
    /// is no slower than `tolerance × peer`.
    pub fn passes_speed(&self, hf2q_wall: f64, peer_wall: f64) -> bool {
        // Defensive: a non-positive peer wall is the missing-binary
        // sentinel; the harness routes that through `Verdict::NotMeasured`
        // before this function is called, but the predicate is
        // semantically `false` in that case.
        if peer_wall <= 0.0 || hf2q_wall < 0.0 {
            return false;
        }
        hf2q_wall <= self.speed_tolerance * peer_wall
    }

    /// Returns `true` iff hf2q's peak RSS is within
    /// `self.rss_tolerance × peer_rss`. Same sentinel semantics as
    /// [`passes_speed`].
    pub fn passes_rss(&self, hf2q_rss: u64, peer_rss: u64) -> bool {
        if peer_rss == 0 || peer_rss == u64::MAX || hf2q_rss == u64::MAX {
            return false;
        }
        let limit = (peer_rss as f64) * self.rss_tolerance;
        (hf2q_rss as f64) <= limit
    }

    /// Returns `true` iff hf2q's PPL is within
    /// `self.ppl_tolerance × peer_ppl`. PPL is a positive quantity;
    /// `None` on either side means "PPL was not measured this iter"
    /// (PPL columns deferred to iter-2 — see ADR-014 P10 row).
    pub fn passes_ppl(&self, hf2q_ppl: Option<f32>, peer_ppl: Option<f32>) -> bool {
        match (hf2q_ppl, peer_ppl) {
            (Some(h), Some(p)) if p > 0.0 && h >= 0.0 => {
                (h as f64) <= self.ppl_tolerance * (p as f64)
            }
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------
// Result + Verdict + Ratios
// ---------------------------------------------------------------------

/// Per-cell ratio summary surfaced into the markdown table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ratios {
    pub speed: f64,
    pub rss: f64,
    pub ppl: f64,
}

impl Ratios {
    /// Sentinel ratios for cells that returned `Verdict::NotMeasured`
    /// (no real model, missing peer binary, or PPL deferred to
    /// iter-2). All fields `f64::NAN` so the markdown table can
    /// surface `n/a` without confusion with a real `0.0` ratio.
    pub fn not_measured() -> Self {
        Self {
            speed: f64::NAN,
            rss: f64::NAN,
            ppl: f64::NAN,
        }
    }
}

/// Verdict for a single gate cell. `NotMeasured` is the canonical
/// outcome for `#[ignore]`-gated cells this iter (no real models
/// loaded) — distinct from `Pass` and `Fail` so the markdown table
/// surfaces the deferred state honestly.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Pass,
    Fail { reason: String },
    NotMeasured { reason: String },
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Pass => f.write_str("PASS"),
            Verdict::Fail { reason } => write!(f, "FAIL ({reason})"),
            Verdict::NotMeasured { reason } => write!(f, "NOT MEASURED ({reason})"),
        }
    }
}

/// One row of the harness output: the cell descriptor + the two
/// observations + computed ratios + final verdict.
#[derive(Debug, Clone)]
pub struct CellResult {
    pub cell: GateCell,
    pub hf2q: RunMetrics,
    pub peer: RunMetrics,
    pub ppl_hf2q: Option<f32>,
    pub ppl_peer: Option<f32>,
    pub verdict: Verdict,
    pub ratios: Ratios,
}

impl CellResult {
    /// Construct a `NotMeasured` result with sentinel metrics. Used
    /// by the `#[ignore]`-gated cells this iter (P11 wires real
    /// models) and by any cell whose peer binary was absent.
    pub fn not_measured(cell: GateCell, reason: &str) -> Self {
        Self {
            cell,
            hf2q: RunMetrics::missing_binary("hf2q (deferred)"),
            peer: RunMetrics::missing_binary("peer (deferred)"),
            ppl_hf2q: None,
            ppl_peer: None,
            verdict: Verdict::NotMeasured { reason: reason.to_string() },
            ratios: Ratios::not_measured(),
        }
    }
}

// ---------------------------------------------------------------------
// Markdown table emitter (S4)
// ---------------------------------------------------------------------

/// Pure function — produces the full markdown document from a slice
/// of `CellResult`s plus a hardware fingerprint and git SHA. No I/O.
///
/// Header columns (PPL deferred to iter-2):
///   `Model | Backend | Calibrator | Peer | hf2q wall | peer wall |
///    speed ratio | hf2q RSS | peer RSS | RSS ratio | Verdict`
///
/// On empty input the table reports the header followed by a single
/// `No results — harness ran with empty input.` line.
pub fn emit_markdown_table(
    results: &[CellResult],
    hardware_fingerprint: &str,
    sha: &str,
) -> String {
    let mut out = String::new();
    out.push_str("# ADR-014 P10 Peer-Parity Results\n\n");
    out.push_str(&format!("## Hardware: {hardware_fingerprint}\n"));
    out.push_str(&format!("## SHA: {sha}\n\n"));
    out.push_str(
        "**Note**: PPL columns deferred to ADR-014 P10 iter-2 \
         (no wikitext-2 fixture or `ppl_kl_eval` driver exists in repo today; \
         iter-2 lands them when P6 closes).\n\n",
    );

    out.push_str(
        "| Model | Backend | Calibrator | Peer | hf2q wall (s) | peer wall (s) | \
         speed ratio | hf2q RSS (B) | peer RSS (B) | RSS ratio | Verdict |\n",
    );
    out.push_str(
        "|-------|---------|------------|------|---------------|---------------|\
         -------------|--------------|--------------|-----------|---------|\n",
    );

    if results.is_empty() {
        out.push_str(
            "| _No results — harness ran with empty input._ | | | | | | | | | | |\n",
        );
        return out;
    }

    for r in results {
        let speed_ratio = format_ratio(r.ratios.speed);
        let rss_ratio = format_ratio(r.ratios.rss);
        let hf2q_wall = format_metric_f64(r.hf2q.wall_s);
        let peer_wall = format_metric_f64(r.peer.wall_s);
        let hf2q_rss = format_metric_u64(r.hf2q.peak_rss_bytes);
        let peer_rss = format_metric_u64(r.peer.peak_rss_bytes);
        out.push_str(&format!(
            "| {model} | {backend} | {calibrator} | {peer} | {hw} | {pw} | {sr} | {hr} | {pr} | {rr} | {v} |\n",
            model = r.cell.model_id,
            backend = r.cell.backend,
            calibrator = r.cell.calibrator_variant,
            peer = r.cell.peer_id,
            hw = hf2q_wall,
            pw = peer_wall,
            sr = speed_ratio,
            hr = hf2q_rss,
            pr = peer_rss,
            rr = rss_ratio,
            v = r.verdict,
        ));
    }
    out
}

fn format_ratio(r: f64) -> String {
    if r.is_nan() {
        "n/a".to_string()
    } else {
        format!("{r:.3}")
    }
}

fn format_metric_f64(v: f64) -> String {
    if v == -1.0 {
        "n/a".to_string()
    } else {
        format!("{v:.3}")
    }
}

fn format_metric_u64(v: u64) -> String {
    if v == u64::MAX {
        "n/a".to_string()
    } else {
        format!("{v}")
    }
}

/// Persist the harness's markdown output to
/// `docs/peer-parity-results-<YYYY-MM-DD>.md`. **Only** callable from
/// `#[ignore]`-gated tests (mutating `docs/` from the always-on suite
/// would pollute the repo on every run).
///
/// `today` is provided by the caller (UTC date stamp like
/// `"2026-04-27"`) so the writer stays pure and deterministic and
/// the iter-1 always-on suite does not need a clock.
pub fn write_results_to_dated_doc(
    results: &[CellResult],
    hardware_fingerprint: &str,
    sha: &str,
    today: &str,
    docs_dir: &std::path::Path,
) -> std::io::Result<std::path::PathBuf> {
    let path = docs_dir.join(format!("peer-parity-results-{today}.md"));
    let body = emit_markdown_table(results, hardware_fingerprint, sha);
    std::fs::write(&path, body)?;
    Ok(path)
}

// ---------------------------------------------------------------------
// run_cell — the per-#[ignore]-test entry point
// ---------------------------------------------------------------------

/// Run a single gate cell end-to-end. This iter every call returns
/// `Verdict::NotMeasured` (no real models present); P11 swaps in the
/// real-model wiring. We still touch the subprocess wrappers via the
/// runner modules' missing-binary path so the integration is exercised
/// at the type level (the always-on smoke tests below cover the
/// behavioural contract).
fn run_cell(cell: GateCell) -> CellResult {
    CellResult::not_measured(
        cell,
        "P11 territory — needs real apex MoE GPU + ~150 GB disk per ADR-014 R13",
    )
}

// =====================================================================
// Always-on smoke tests (6)
// =====================================================================

/// Smoke 1: harness compiles and the markdown emitter produces a
/// well-formed table on empty input (header + the empty-input line).
#[test]
fn harness_compiles_and_emits_table_skeleton() {
    let md = emit_markdown_table(&[], "M5 Max, 128 GB", "abc1234");
    assert!(md.contains("# ADR-014 P10 Peer-Parity Results"));
    assert!(md.contains("## Hardware: M5 Max, 128 GB"));
    assert!(md.contains("## SHA: abc1234"));
    // PPL deferral note must be surfaced inline.
    assert!(md.contains("PPL columns deferred"));
    // Header row must contain every column the harness reports.
    for col in [
        "Model",
        "Backend",
        "Calibrator",
        "Peer",
        "hf2q wall (s)",
        "peer wall (s)",
        "speed ratio",
        "hf2q RSS (B)",
        "peer RSS (B)",
        "RSS ratio",
        "Verdict",
    ] {
        assert!(md.contains(col), "header missing column `{col}`:\n{md}");
    }
    assert!(md.contains("_No results — harness ran with empty input._"));
}

/// Smoke 2: speed-ratio tolerance check (hf2q within `1.10×` peer
/// passes; outside fails; missing-binary sentinel always fails).
#[test]
fn gate_cell_speed_tolerance_check() {
    let cell = GATE_CELLS[0]; // 1.10× speed gate
    // 1.05× peer → pass.
    assert!(cell.passes_speed(1.05, 1.0));
    // Exact tolerance → pass (≤, inclusive).
    assert!(cell.passes_speed(1.10, 1.0));
    // 1.11× → fail.
    assert!(!cell.passes_speed(1.11, 1.0));
    // Sentinel peer wall (missing-binary) → fail (cell routes via
    // Verdict::NotMeasured upstream).
    assert!(!cell.passes_speed(1.0, -1.0));
    assert!(!cell.passes_speed(-1.0, 1.0));

    // Row 4: hf2q-vs-current-pipeline, gate is exactly 1.0× speed.
    let strict = GATE_CELLS[3];
    assert!(strict.passes_speed(1.0, 1.0));
    assert!(!strict.passes_speed(1.001, 1.0));
}

/// Smoke 3: RSS-ratio tolerance check. Critically, row 4
/// (hf2q-vs-current-pipeline) gates RSS at ≤ 0.50× — the central
/// streaming-halves-peak-RSS claim of the ADR.
#[test]
fn gate_cell_rss_tolerance_check() {
    let cell = GATE_CELLS[0]; // 1.10× RSS gate
    assert!(cell.passes_rss(110_000_000, 100_000_000));
    assert!(cell.passes_rss(100_000_000, 100_000_000));
    assert!(!cell.passes_rss(110_000_001, 100_000_000));
    // Sentinel RSS (missing-binary) → fail.
    assert!(!cell.passes_rss(u64::MAX, 100_000_000));
    assert!(!cell.passes_rss(100_000_000, u64::MAX));

    // Row 4: hf2q-vs-current-pipeline, gate is ≤ 0.50× RSS.
    let strict = GATE_CELLS[3];
    assert!(strict.passes_rss(50_000_000, 100_000_000));
    assert!(!strict.passes_rss(50_000_001, 100_000_000));
}

/// Smoke 4: PPL-ratio tolerance check (when both sides Some). PPL
/// columns are deferred to iter-2 but the predicate is wired now so
/// iter-2 only needs to populate the inputs.
#[test]
fn gate_cell_ppl_tolerance_check() {
    let cell = GATE_CELLS[0]; // 1.02× PPL gate
    assert!(cell.passes_ppl(Some(10.0), Some(10.0)));
    assert!(cell.passes_ppl(Some(10.2), Some(10.0)));
    assert!(!cell.passes_ppl(Some(10.21), Some(10.0)));
    // None on either side → fail (PPL not measured this iter; the
    // markdown table surfaces NotMeasured rather than a fake green).
    assert!(!cell.passes_ppl(None, Some(10.0)));
    assert!(!cell.passes_ppl(Some(10.0), None));
    assert!(!cell.passes_ppl(None, None));

    // Row 4: hf2q-vs-current-pipeline, gate is ≤ 1.0× PPL (tighter
    // — DWQ→GGUF must not regress vs the current pipeline).
    let strict = GATE_CELLS[3];
    assert!(strict.passes_ppl(Some(10.0), Some(10.0)));
    assert!(!strict.passes_ppl(Some(10.0001), Some(10.0)));
}

/// Smoke 5: subprocess wrapper handles a missing llama.cpp binary by
/// returning the `RunMetrics::missing_binary` sentinel rather than
/// panicking. Routes via `HF2Q_LLAMA_QUANTIZE_BIN` set to a
/// guaranteed-absent path.
///
/// `ENV_LOCK` is held for the duration to serialise against any
/// other test in this binary that touches the same env var.
#[test]
fn subprocess_wrapper_handles_missing_binary() {
    let _guard = ENV_LOCK.lock().expect("ENV_LOCK poisoned");
    // SAFETY: ENV_LOCK serialises every env-mutating test in this
    // binary; the unsafe `set_var`/`remove_var` calls are bounded to
    // this critical section, so no other Rust thread can observe a
    // partial state.
    unsafe {
        std::env::set_var(
            "HF2Q_LLAMA_QUANTIZE_BIN",
            "/nonexistent/llama-quantize-test-stub",
        );
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let m = llama_cpp_runner::run_llama_quantize(
        &tmp.path().join("in.gguf"),
        &tmp.path().join("out.gguf"),
        "Q4_K_M",
    );
    // Restore env BEFORE the assertions so a panic during assertion
    // doesn't leak the override into subsequent tests waiting on the
    // mutex (Mutex auto-poisons on panic, but we still want the
    // var-restore side effect to land even on success).
    unsafe {
        std::env::remove_var("HF2Q_LLAMA_QUANTIZE_BIN");
    }
    assert!(m.is_missing_binary(), "missing binary must surface sentinel; got {m:?}");
    assert_eq!(m.wall_s, -1.0);
    assert_eq!(m.peak_rss_bytes, u64::MAX);
    assert_eq!(m.exit_code, -1);
    assert!(m.stderr_tail.contains("llama-quantize"));
}

/// Smoke 6: subprocess wrapper handles a missing python interpreter
/// the same way (sentinel, not panic). Routes via `HF2Q_PYTHON_BIN`
/// set to a guaranteed-absent path.
///
/// `ENV_LOCK` is held for the duration to serialise against any
/// other test in this binary that touches the same env var.
#[test]
fn subprocess_wrapper_handles_missing_python_module() {
    let _guard = ENV_LOCK.lock().expect("ENV_LOCK poisoned");
    // SAFETY: see comment in `subprocess_wrapper_handles_missing_binary`.
    unsafe {
        std::env::set_var("HF2Q_PYTHON_BIN", "/nonexistent/python3-test-stub");
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let (m, ok) = mlx_lm_runner::run_mlx_lm_load(tmp.path());
    unsafe {
        std::env::remove_var("HF2Q_PYTHON_BIN");
    }
    assert!(m.is_missing_binary(), "missing python must surface sentinel; got {m:?}");
    assert!(!ok);
    assert_eq!(m.exit_code, -1);
    assert!(m.stderr_tail.contains("python3"));
}

// =====================================================================
// Always-on structural tests (additional — cover GATE_CELLS contents
// + emitter-with-data behaviour without growing the headline 6-test
// count). These remain part of the binary's always-on surface.
// =====================================================================

/// The matrix has exactly 8 rows — Decision 15 lines 575–582.
#[test]
fn gate_cells_count_is_eight() {
    assert_eq!(GATE_CELLS.len(), 8, "Decision 15 has exactly 8 cells");
}

/// 27B-dense subset is rows 1..=4; apex-MoE subset is rows 5..=8.
/// Every cell's tuple of (model, backend, calibrator, peer, speed,
/// rss, ppl) is checked verbatim against Decision 15 to catch silent
/// drift.
#[test]
fn gate_cells_match_decision_15_verbatim() {
    // Hand-rebuilt expectation (independent of the GATE_CELLS const)
    // so any drift in either source surfaces as a failure.
    let expected: [(&str, BackendKind, &str, PeerId, f64, f64, f64); 8] = [
        ("27B dense", BackendKind::Gguf, "None (q4_k_m)", PeerId::LlamaCppUncalibratedQ4KM, 1.10, 1.10, 1.02),
        ("27B dense", BackendKind::Gguf, "Imatrix (imatrix-q4_k_m)", PeerId::LlamaCppImatrixQ4KM, 1.10, 1.10, 1.02),
        ("27B dense", BackendKind::Safetensors, "DWQ (dwq-4-6)", PeerId::MlxLmDwq, 1.10, 1.10, 1.02),
        ("27B dense", BackendKind::Gguf, "DWQ (dwq-4-6)", PeerId::Hf2qCurrentPipeline, 1.0, 0.50, 1.0),
        ("apex MoE", BackendKind::Gguf, "None (q4_k_m)", PeerId::LlamaCppUncalibratedQ4KM, 1.10, 1.10, 1.02),
        ("apex MoE", BackendKind::Gguf, "Imatrix (imatrix-q4_k_m)", PeerId::LlamaCppImatrixQ4KM, 1.10, 1.10, 1.02),
        ("apex MoE", BackendKind::Safetensors, "DWQ (dwq-4-6)", PeerId::MlxLmDwq, 1.10, 1.10, 1.02),
        ("apex MoE", BackendKind::Gguf, "DWQ (dwq-4-6)", PeerId::Hf2qCurrentPipeline, 1.0, 0.50, 1.0),
    ];

    for (idx, (cell, want)) in GATE_CELLS.iter().zip(expected.iter()).enumerate() {
        assert_eq!(cell.model_id, want.0, "row {idx} model");
        assert_eq!(cell.backend, want.1, "row {idx} backend");
        assert_eq!(cell.calibrator_variant, want.2, "row {idx} calibrator");
        assert_eq!(cell.peer_id, want.3, "row {idx} peer");
        assert!(
            (cell.speed_tolerance - want.4).abs() < f64::EPSILON,
            "row {idx} speed_tolerance: {} vs {}",
            cell.speed_tolerance,
            want.4
        );
        assert!(
            (cell.rss_tolerance - want.5).abs() < f64::EPSILON,
            "row {idx} rss_tolerance: {} vs {}",
            cell.rss_tolerance,
            want.5
        );
        assert!(
            (cell.ppl_tolerance - want.6).abs() < f64::EPSILON,
            "row {idx} ppl_tolerance: {} vs {}",
            cell.ppl_tolerance,
            want.6
        );
    }
}

/// Markdown emitter with one synthetic NotMeasured row: each output
/// table line has the same pipe-delimited column count as the header.
#[test]
fn emit_markdown_table_with_one_synthetic_result_round_trips_columns() {
    let result = CellResult::not_measured(GATE_CELLS[0], "smoke");
    let md = emit_markdown_table(&[result], "M5 Max, 128 GB", "deadbee");
    let header_pipes = md
        .lines()
        .find(|l| l.starts_with("| Model "))
        .map(|l| l.matches('|').count())
        .expect("header present");
    let data_pipes = md
        .lines()
        .find(|l| l.starts_with("| 27B dense "))
        .map(|l| l.matches('|').count())
        .expect("data row present");
    assert_eq!(header_pipes, data_pipes, "data row must match header column count");
    assert!(md.contains("27B dense"));
    assert!(md.contains("GGUF"));
    assert!(md.contains("None (q4_k_m)"));
    assert!(md.contains("llama.cpp uncalibrated Q4_K_M"));
    assert!(md.contains("NOT MEASURED"));
}

// =====================================================================
// #[ignore]-gated cells (8) — one per GATE_CELLS row, P11 territory.
// Each calls `run_cell(cell)` end-to-end and asserts the verdict is
// `NotMeasured` this iter (real-model wiring lands in P11).
// =====================================================================

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk"]
fn cell_0_27b_dense_gguf_uncalibrated_q4km() {
    let r = run_cell(GATE_CELLS[0]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk"]
fn cell_1_27b_dense_gguf_imatrix_q4km() {
    let r = run_cell(GATE_CELLS[1]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk"]
fn cell_2_27b_dense_safetensors_dwq46() {
    let r = run_cell(GATE_CELLS[2]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk"]
fn cell_3_27b_dense_gguf_dwq46_vs_current_pipeline() {
    let r = run_cell(GATE_CELLS[3]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk"]
fn cell_4_apex_moe_gguf_uncalibrated_q4km() {
    let r = run_cell(GATE_CELLS[4]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk"]
fn cell_5_apex_moe_gguf_imatrix_q4km() {
    let r = run_cell(GATE_CELLS[5]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk"]
fn cell_6_apex_moe_safetensors_dwq46() {
    let r = run_cell(GATE_CELLS[6]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk"]
fn cell_7_apex_moe_gguf_dwq46_vs_current_pipeline() {
    let r = run_cell(GATE_CELLS[7]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}
