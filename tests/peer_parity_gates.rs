//! ADR-014 P10 iter-1 + iter-2a + iter-2b + iter-2c — peer-parity
//! benchmark harness.
//!
//! This crate orchestrates the eight Decision-15 gate cells (lines
//! 575–582 of `docs/ADR-014-streaming-convert-pipeline.md`) that
//! compare hf2q's streaming convert pipeline against llama.cpp and
//! mlx-lm peers across 27B-dense + apex-MoE × {GGUF None, GGUF
//! Imatrix, safetensors DWQ, GGUF DWQ vs current pipeline}.
//!
//! iter-2a lands the **peer-side** half of PPL: a
//! `tests/common/llama_cpp_runner.rs::run_llama_perplexity` wrapper
//! around `/opt/homebrew/bin/llama-perplexity`, a 512-token smoke
//! fixture at `tests/fixtures/ppl-corpus/wikitext2-smoke.tokens`,
//! and three new columns in `emit_markdown_table` (`hf2q PPL`,
//! `peer PPL`, `PPL ratio`).
//!
//! iter-2b landed the hf2q-side driver
//! (`src/quality/ppl_driver.rs::measure_ppl_qwen35_dense`) and wired
//! the 4 dense cells (cells 0-3) through it.
//!
//! iter-2c (this iter) renamed the driver to `measure_ppl_qwen35`
//! after a Chesterton's-fence audit confirmed
//! `Qwen35Model::load_from_gguf` (model.rs:172) and `forward_cpu`
//! (forward_cpu.rs:75) both dispatch on `cfg.variant` internally.
//! The 4 MoE cells (cells 4-7) now record `hf2q_ppl` via the same
//! variant-agnostic driver — 8-cell PPL coverage complete. Sentinel
//! `/var/empty/...gguf` paths still surface `PplDriverError::Gguf` ⇒
//! `hf2q_ppl = None` ⇒ `Verdict::NotMeasured` for all 8 cells until
//! P11 swaps real model paths in.
//!
//! Test inventory (now 23 always-on + 8 ignored = 31 total):
//!
//! Always-on smoke + structural (run by default):
//!  - iter-1 baseline: 19 tests covering markdown empty-input,
//!    speed/RSS/PPL tolerance predicates, missing-binary sentinel,
//!    GATE_CELLS Decision-15 verbatim audit.
//!  - iter-2a additions (4 tests):
//!    1. `wikitext2_smoke_fixture_loads_to_512_tokens`
//!    2. `peer_perplexity_wrapper_handles_missing_binary`
//!    3. `markdown_table_renders_ppl_columns`
//!    4. `markdown_table_renders_full_ppl_row_when_both_measured`
//!
//! `#[ignore]`-gated (8 tests, one per Decision-15 cell, P11
//! territory — needs apex MoE GPU + ~150 GB disk):
//! `cell_<idx>_<model>_<backend>_<calibrator>` — each runs the
//! corresponding `GateCell` end-to-end against the real peer; this
//! iter routes the peer side through `run_llama_perplexity` against
//! the 512-token smoke fixture so the wiring is exercised, and
//! returns `Verdict::NotMeasured` because hf2q-side PPL is deferred
//! to iter-2b and no real models are present yet.

mod common;

// ---------------------------------------------------------------------
// iter-2b + iter-2c: hf2q-side PPL driver wiring
// ---------------------------------------------------------------------
//
// The hf2q driver lives at `src/quality/ppl_driver.rs` and depends on
// `src/quality/perplexity.rs` + the `Qwen35Model` public API in
// `src/inference/models/qwen35/`. hf2q is a binary crate (no `[lib]`
// target — confirmed at Cargo.toml:1-160), so `tests/*.rs` cannot say
// `use hf2q::quality::ppl_driver::measure_ppl_qwen35`. The established
// pattern (see `tests/imatrix_xvalidation.rs:48-52`) is
// `#[path]`-include of the production source files; we follow that
// pattern here. (iter-2c rename: `measure_ppl_qwen35_dense` →
// `measure_ppl_qwen35`; one variant-agnostic entry point handles both
// `Qwen35Variant::Dense` and `Qwen35Variant::Moe`.)
//
// `ppl_driver.rs` references the `Qwen35Model` public API via
// `use crate::inference::models::qwen35::...` paths. Those modules
// belong to ADR-013 and form a deeply-interconnected web (model.rs,
// forward_cpu.rs, ffn.rs, full_attn.rs, delta_net.rs, weight_loader.rs,
// kernels.rs, …) — `#[path]`-mirroring the whole tree into this test
// crate is impractical. Instead we provide minimal type-stubs in the
// `inference::models::qwen35` namespace below: the public-API surface
// `ppl_driver` calls (`Qwen35Model::load_from_gguf`,
// `Qwen35Model::forward_cpu`, `Qwen35Variant::Dense`, the
// `text_positions` helper) is enough to make the include compile.
//
// All 8 `#[ignore]`-gated cells in this binary (4 dense + 4 MoE
// post-iter-2c) use sentinel model paths (`/var/empty/...gguf`) that
// don't exist on disk, so `GgufFile::open` short-circuits with
// `MlxError::IoError` long before the driver would ever reach the
// stubbed `Qwen35Model::load_from_gguf`. The cells therefore exercise
// the GGUF-error path of the driver honestly; the stubs are dead code
// at runtime in this test binary. P11 swaps the sentinel paths for
// real 27B-dense + apex-MoE GGUFs and adds a `[lib]` target (or moves
// the wiring into `src/main.rs`'s test scaffolding) so the real-model
// load is invoked through the production qwen35 code, not these
// stubs.

#[path = "../src/quality/perplexity.rs"]
pub mod perplexity;

#[path = "../src/quality/ppl_driver.rs"]
mod ppl_driver;

mod quality {
    pub use super::perplexity;
}

mod inference {
    pub mod models {
        pub mod qwen35 {
            //! Type-stubs sufficient to satisfy `ppl_driver.rs`'s
            //! `use crate::inference::models::qwen35::...` lines. NEVER
            //! invoked at runtime by the smoke tests in this binary —
            //! every cell uses a sentinel `/var/empty/...gguf` path so
            //! the driver fails at `GgufFile::open` long before it
            //! would touch this stubbed surface.

            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub enum Qwen35Variant {
                Dense,
                Moe,
            }

            pub mod forward_cpu {
                /// Mirror of `src/inference/models/qwen35/forward_cpu.rs::text_positions`
                /// — kept identical (text-convention `[i, i, i, i]`) so a
                /// future change to the production helper trips a divergence
                /// test rather than silently desyncing.
                pub fn text_positions(seq_len: u32) -> Vec<[i32; 4]> {
                    (0..seq_len as i32).map(|i| [i, i, i, i]).collect()
                }
            }

            pub mod model {
                use super::Qwen35Variant;
                use anyhow::{anyhow, Result};
                use mlx_native::gguf::GgufFile;

                /// Stub of `Qwen35Config`. Only the fields ppl_driver
                /// reads are populated.
                pub struct Qwen35Config {
                    pub vocab_size: u32,
                    pub max_position_embeddings: u32,
                    pub variant: Qwen35Variant,
                }

                /// Stub of `Qwen35Model`. `load_from_gguf` and
                /// `forward_cpu` always error: this binary's smoke
                /// tests never reach them (sentinel-path GGUFs fail at
                /// open). Real-model wiring is P11 territory.
                pub struct Qwen35Model {
                    pub cfg: Qwen35Config,
                }

                impl Qwen35Model {
                    pub fn load_from_gguf(_gguf: &GgufFile) -> Result<Self> {
                        Err(anyhow!(
                            "tests/peer_parity_gates.rs: stubbed Qwen35Model::load_from_gguf \
                             — production driver lives at src/quality/ppl_driver.rs and \
                             reaches the real Qwen35Model only when wired through a \
                             [lib] target (deferred to P11)"
                        ))
                    }

                    pub fn forward_cpu(
                        &self,
                        _tokens: &[u32],
                        _positions: &[[i32; 4]],
                    ) -> Result<Vec<f32>> {
                        Err(anyhow!(
                            "tests/peer_parity_gates.rs: stubbed Qwen35Model::forward_cpu \
                             — see load_from_gguf comment"
                        ))
                    }
                }
            }
        }
    }
}

use ppl_driver::measure_ppl_qwen35;

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

    /// Build a `CellResult` from raw measurements (iter-2a entry
    /// point — the always-on suite drives this directly to verify
    /// verdict logic, and `run_cell` calls it after invoking the
    /// peer wrapper).
    ///
    /// Verdict semantics (iter-2a):
    /// - If **either** side's wall_s is the missing-binary sentinel
    ///   (`-1.0`) **or** hf2q_ppl is `None` (hf2q-side driver lands
    ///   at iter-2b), the cell is `Verdict::NotMeasured` — never
    ///   fake-green and never fake-red. The `reason` field carries
    ///   the disqualifier so the markdown table surfaces *why*.
    /// - Otherwise (all three signals measured on both sides) the
    ///   verdict is `Pass` iff every gate passes:
    ///   - speed: `hf2q_wall ≤ tolerance × peer_wall`
    ///   - RSS:   `hf2q_rss  ≤ tolerance × peer_rss`
    ///   - PPL:   `hf2q_ppl  ≤ tolerance × peer_ppl`
    ///
    ///   Any gate failure produces `Fail { reason }` naming the
    ///   first failing gate.
    pub fn from_measurements(
        cell: GateCell,
        hf2q: RunMetrics,
        peer: RunMetrics,
        ppl_hf2q: Option<f32>,
        ppl_peer: Option<f32>,
    ) -> Self {
        // Compute ratios up front — `NaN` for the dimensions that
        // can't be ratio'd (sentinel inputs / missing PPL).
        let speed_ratio = if peer.wall_s > 0.0 && hf2q.wall_s >= 0.0 {
            hf2q.wall_s / peer.wall_s
        } else {
            f64::NAN
        };
        let rss_ratio = if peer.peak_rss_bytes > 0
            && peer.peak_rss_bytes != u64::MAX
            && hf2q.peak_rss_bytes != u64::MAX
        {
            (hf2q.peak_rss_bytes as f64) / (peer.peak_rss_bytes as f64)
        } else {
            f64::NAN
        };
        let ppl_ratio = match (ppl_hf2q, ppl_peer) {
            (Some(h), Some(p)) if p > 0.0 && h.is_finite() && p.is_finite() => {
                (h as f64) / (p as f64)
            }
            _ => f64::NAN,
        };

        let ratios = Ratios {
            speed: speed_ratio,
            rss: rss_ratio,
            ppl: ppl_ratio,
        };

        // Verdict: NotMeasured if any input is missing; else gate.
        let verdict = if hf2q.is_missing_binary() {
            Verdict::NotMeasured {
                reason: "hf2q-side measurement missing (driver lands at iter-2b)".to_string(),
            }
        } else if peer.is_missing_binary() {
            Verdict::NotMeasured {
                reason: format!("peer binary missing: {}", peer.stderr_tail),
            }
        } else if ppl_hf2q.is_none() {
            Verdict::NotMeasured {
                reason: "hf2q PPL not measured (iter-2b deferred)".to_string(),
            }
        } else if ppl_peer.is_none() {
            Verdict::NotMeasured {
                reason: "peer PPL not parsed from llama-perplexity stderr".to_string(),
            }
        } else if !cell.passes_speed(hf2q.wall_s, peer.wall_s) {
            Verdict::Fail {
                reason: format!(
                    "speed gate: {:.3}× > {:.2}× tolerance",
                    speed_ratio, cell.speed_tolerance
                ),
            }
        } else if !cell.passes_rss(hf2q.peak_rss_bytes, peer.peak_rss_bytes) {
            Verdict::Fail {
                reason: format!(
                    "RSS gate: {:.3}× > {:.2}× tolerance",
                    rss_ratio, cell.rss_tolerance
                ),
            }
        } else if !cell.passes_ppl(ppl_hf2q, ppl_peer) {
            Verdict::Fail {
                reason: format!(
                    "PPL gate: {:.4}× > {:.2}× tolerance",
                    ppl_ratio, cell.ppl_tolerance
                ),
            }
        } else {
            Verdict::Pass
        };

        Self {
            cell,
            hf2q,
            peer,
            ppl_hf2q,
            ppl_peer,
            verdict,
            ratios,
        }
    }
}

// ---------------------------------------------------------------------
// Markdown table emitter (S4)
// ---------------------------------------------------------------------

/// Pure function — produces the full markdown document from a slice
/// of `CellResult`s plus a hardware fingerprint and git SHA. No I/O.
///
/// Header columns (iter-2a appends `hf2q PPL`, `peer PPL`,
/// `PPL ratio` at the **end** so the iter-1 column order is preserved
/// and the iter-1 always-on tests pass unchanged):
///   `Model | Backend | Calibrator | Peer | hf2q wall (s) |
///    peer wall (s) | speed ratio | hf2q RSS (B) | peer RSS (B) |
///    RSS ratio | Verdict | hf2q PPL | peer PPL | PPL ratio`
///
/// PPL cells render `f32` to 4 decimal places; un-measured PPLs
/// (Option::None) render as the em-dash `—` so the deferred state
/// is visually distinct from a real `0.0000`. The PPL ratio is
/// `hf2q_ppl / peer_ppl` to 4 decimals when both sides are measured,
/// `—` when either is missing.
///
/// On empty input the table reports the header followed by a single
/// `No results — harness ran with empty input.` line, with the
/// pipe-count matching the 14-column header so downstream
/// markdown-table consumers don't choke.
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
        "**Note (iter-2a)**: peer-side PPL is wired via \
         `tests/common/llama_cpp_runner.rs::run_llama_perplexity`; \
         hf2q-side PPL stays `NotMeasured` until iter-2b lands the \
         hf2q-side driver wrapping `Qwen35Dense::from_gguf` + \
         chunked forward-pass + `compute_perplexity`.\n\n",
    );

    out.push_str(
        "| Model | Backend | Calibrator | Peer | hf2q wall (s) | peer wall (s) | \
         speed ratio | hf2q RSS (B) | peer RSS (B) | RSS ratio | Verdict | \
         hf2q PPL | peer PPL | PPL ratio |\n",
    );
    out.push_str(
        "|-------|---------|------------|------|---------------|---------------|\
         -------------|--------------|--------------|-----------|---------|\
         ----------|----------|-----------|\n",
    );

    if results.is_empty() {
        // 14 columns ⇒ 15 pipes (one before each cell + one trailing).
        out.push_str(
            "| _No results — harness ran with empty input._ | | | | | | | | | | | | | |\n",
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
        let hf2q_ppl = format_ppl(r.ppl_hf2q);
        let peer_ppl = format_ppl(r.ppl_peer);
        let ppl_ratio = format_ppl_ratio(r.ppl_hf2q, r.ppl_peer);
        out.push_str(&format!(
            "| {model} | {backend} | {calibrator} | {peer} | {hw} | {pw} | {sr} | {hr} | {pr} | {rr} | {v} | {hp} | {pp} | {pr2} |\n",
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
            hp = hf2q_ppl,
            pp = peer_ppl,
            pr2 = ppl_ratio,
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

/// Renders a measured PPL to 4 decimal places, or the em-dash when
/// the cell is unmeasured (PPL was deferred or peer binary missing).
/// The em-dash keeps `0.0000` visually distinct from "no measurement".
fn format_ppl(v: Option<f32>) -> String {
    match v {
        Some(p) => format!("{p:.4}"),
        None => "—".to_string(),
    }
}

/// Renders the hf2q/peer PPL ratio to 4 decimals when both sides are
/// measured. Missing on either side → em-dash. A non-positive peer
/// PPL would yield ±inf so we route that through the em-dash too —
/// the harness does not surface fake ratios.
fn format_ppl_ratio(hf2q: Option<f32>, peer: Option<f32>) -> String {
    match (hf2q, peer) {
        (Some(h), Some(p)) if p > 0.0 && h.is_finite() && p.is_finite() => {
            format!("{:.4}", (h as f64) / (p as f64))
        }
        _ => "—".to_string(),
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

/// Resolves the iter-2a smoke fixture path
/// (`tests/fixtures/ppl-corpus/wikitext2-smoke.tokens`) relative to
/// `CARGO_MANIFEST_DIR`. The fixture is committed to the repo (2 KB
/// deterministic ramp — see the sibling README); this resolver
/// avoids hard-coding `/opt/hf2q` so the harness works from any
/// worktree.
fn smoke_corpus_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ppl-corpus")
        .join("wikitext2-smoke.tokens")
}

/// Read the iter-2a smoke fixture into a `Vec<u32>`. Returns `None`
/// if the file is missing or has the wrong byte length. Used by both
/// the peer-side wrapper (which passes the file path directly to
/// `llama-perplexity`) and the hf2q-side driver (which needs the
/// tokens in memory).
///
/// The fixture is documented at
/// `tests/fixtures/ppl-corpus/README.md`: 512 little-endian u32s
/// generated by `(i * 17 + 3) % 32000`. iter-2a's
/// `wikitext2_smoke_fixture_loads_to_512_tokens` test pins this.
fn load_smoke_corpus_tokens() -> Option<Vec<u32>> {
    let path = smoke_corpus_path();
    let bytes = std::fs::read(&path).ok()?;
    if bytes.len() != 512 * 4 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Resolve the hf2q-side model path for a gate cell. P11 swaps these
/// sentinel paths for real model artifacts staged on disk; iter-2b
/// uses sentinels so `GgufFile::open` short-circuits the driver with
/// `PplDriverError::Gguf` without touching the stubbed
/// `Qwen35Model::load_from_gguf`. The path encoding is mechanical
/// (model + backend + calibrator) so the harness's per-cell error
/// surface names the cell that produced it.
fn hf2q_model_path(cell: &GateCell) -> std::path::PathBuf {
    std::path::PathBuf::from(format!(
        "/var/empty/{}-{}-{}-deferred.gguf",
        cell.model_id.replace(' ', "-"),
        cell.backend,
        cell.calibrator_variant.replace(' ', "-"),
    ))
}

/// Run a single gate cell end-to-end. iter-2a wired the peer-side
/// PPL via `run_llama_perplexity` against the smoke fixture; iter-2b
/// wired the hf2q-side PPL via the dense-named driver for cells 0-3;
/// iter-2c (this file) wires the hf2q-side PPL via the renamed
/// variant-agnostic [`measure_ppl_qwen35`] for **all 8 cells (0-7)**
/// — both `27B dense` and `apex MoE` model_ids dispatch through the
/// same driver. `Qwen35Model::load_from_gguf` (model.rs:172) and
/// `forward_cpu` (forward_cpu.rs:75) inspect `cfg.variant` and
/// dispatch internally; the driver never branches on variant.
/// Wall + RSS stay sentinel for both sides this iter — real-model
/// wiring lands in P11.
///
/// Cell-routing rules (post-iter-2c):
/// - All 8 cells call [`measure_ppl_qwen35`] with the cell's hf2q
///   model path and the smoke corpus. With sentinel paths the
///   driver returns `PplDriverError::Gguf` ⇒ `hf2q_ppl = None` ⇒
///   verdict `NotMeasured`. The wiring is exercised for both
///   variants; P11 swaps real 27B-dense + apex-MoE paths.
fn run_cell(cell: GateCell) -> CellResult {
    let corpus_path = smoke_corpus_path();
    let model_path = hf2q_model_path(&cell);

    // Peer-side PPL: invoke the wrapper. The model path is
    // intentionally a sentinel (no real models present yet — P11);
    // the binary will fail to load it, exit non-zero, and the
    // parser will return None for the PPL. That's the correct
    // observable: peer wrapper present, but PPL un-measured because
    // the model wasn't loadable. Once P11 wires real GGUF paths,
    // this same call site starts producing real PPL values.
    let (peer_metrics, peer_ppl) = if corpus_path.is_file() {
        llama_cpp_runner::run_llama_perplexity(&model_path, &corpus_path)
    } else {
        // Fixture missing (should never happen — it's checked in)
        // but stay safe rather than panic. The verdict will surface
        // the missing-fixture state via the peer sentinel.
        (RunMetrics::missing_binary("smoke fixture"), None)
    };

    // hf2q-side PPL: all 8 cells (dense + MoE) route through the
    // variant-agnostic `measure_ppl_qwen35` driver. The model_id
    // discriminator is no longer load-bearing for routing — it
    // remains in `GateCell` because the markdown table still labels
    // each row (`27B dense` vs `apex MoE`), and the future P11
    // wiring will pick model paths via that field.
    let hf2q_ppl: Option<f32> = match load_smoke_corpus_tokens() {
        Some(tokens) => match measure_ppl_qwen35(&model_path, &tokens, None) {
            Ok(ppl) => Some(ppl),
            Err(_) => None,
        },
        None => None,
    };

    // Wall + RSS for hf2q stay sentinel this iter (real-model
    // forward-pass timing lands in P11 alongside the real GGUFs).
    let hf2q_metrics = RunMetrics::missing_binary("hf2q (iter-2c: PPL via variant-agnostic driver; wall+RSS pending P11)");

    CellResult::from_measurements(cell, hf2q_metrics, peer_metrics, hf2q_ppl, peer_ppl)
}

// =====================================================================
// Always-on smoke tests (6)
// =====================================================================

/// Smoke 1: harness compiles and the markdown emitter produces a
/// well-formed table on empty input (header + the empty-input line).
/// iter-2a updates the inline note to advertise the peer-side PPL
/// wiring (the iter-1 "deferred" note is gone now that
/// `run_llama_perplexity` exists and the 3 PPL columns are present).
#[test]
fn harness_compiles_and_emits_table_skeleton() {
    let md = emit_markdown_table(&[], "M5 Max, 128 GB", "abc1234");
    assert!(md.contains("# ADR-014 P10 Peer-Parity Results"));
    assert!(md.contains("## Hardware: M5 Max, 128 GB"));
    assert!(md.contains("## SHA: abc1234"));
    // iter-2a inline note: peer-side PPL wired; hf2q-side deferred
    // to iter-2b. The iter-1 "PPL columns deferred" string is gone —
    // those columns are now present.
    assert!(
        md.contains("peer-side PPL is wired"),
        "iter-2a inline note must advertise peer-side PPL wiring"
    );
    assert!(
        md.contains("iter-2b"),
        "iter-2a inline note must point forward to iter-2b for hf2q-side"
    );
    assert!(
        !md.contains("PPL columns deferred"),
        "iter-1 deferral language must NOT survive into iter-2a output"
    );
    // Header row must contain every column the harness reports
    // (iter-1 columns + iter-2a's three PPL columns).
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
        "hf2q PPL",
        "peer PPL",
        "PPL ratio",
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
// iter-2a always-on smoke tests (4) — peer-side PPL plumbing
// =====================================================================

/// iter-2a #1: the 512-token deterministic smoke fixture loads off
/// disk, has the documented byte length, and the parsed u32 stream
/// matches the documented generation rule `(i * 17 + 3) % 32000`.
/// This guards (a) the fixture survived git, (b) the format pledge
/// in the sibling README is honoured, and (c) iter-2b's hf2q-side
/// driver will read exactly the bytes it expects.
#[test]
fn wikitext2_smoke_fixture_loads_to_512_tokens() {
    let path = smoke_corpus_path();
    assert!(
        path.is_file(),
        "smoke fixture missing at {:?} — must be checked in to git",
        path
    );
    let bytes = std::fs::read(&path).expect("read smoke fixture");
    assert_eq!(
        bytes.len(),
        512 * 4,
        "smoke fixture must be exactly 512 little-endian u32s ⇒ 2048 bytes; got {}",
        bytes.len()
    );

    let tokens: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(tokens.len(), 512);

    // Verify the generation rule end-to-end: every token must equal
    // `(i * 17 + 3) % 32000` per README. This is the byte-identity
    // check iter-2b relies on.
    for (i, &tok) in tokens.iter().enumerate() {
        let expected = ((i as u32) * 17 + 3) % 32000;
        assert_eq!(
            tok, expected,
            "token at index {i} must satisfy the documented ramp formula"
        );
    }

    // Sanity boundaries called out verbatim in the README.
    assert_eq!(&tokens[..4], &[3, 20, 37, 54]);
    assert_eq!(&tokens[508..], &[8639, 8656, 8673, 8690]);
}

/// iter-2a #2: the peer-side PPL wrapper handles a missing binary
/// the same way every other peer wrapper does — tracing::warn +
/// `RunMetrics::missing_binary` sentinel + `None` PPL. Routes via
/// `HF2Q_LLAMA_PERPLEXITY_BIN` set to a guaranteed-absent path. The
/// `is_missing_binary()` accessor (already on `RunMetrics` from
/// iter-1, lines 51–55 of `tests/common/metrics.rs`) is the
/// canonical sentinel detector — no duck-typing against `wall_s`.
///
/// `ENV_LOCK` is held for the duration to serialise against any
/// other test in this binary that touches the same env var.
#[test]
fn peer_perplexity_wrapper_handles_missing_binary() {
    let _guard = ENV_LOCK.lock().expect("ENV_LOCK poisoned");
    // SAFETY: ENV_LOCK serialises every env-mutating test in this
    // binary; the unsafe set/remove calls are bounded to this
    // critical section.
    unsafe {
        std::env::set_var(
            "HF2Q_LLAMA_PERPLEXITY_BIN",
            "/nonexistent/llama-perplexity-test-stub",
        );
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let (m, ppl) = llama_cpp_runner::run_llama_perplexity(
        &tmp.path().join("model.gguf"),
        &tmp.path().join("corpus.tokens"),
    );
    // Restore env BEFORE assertions so a panic doesn't leak the
    // override into siblings waiting on the mutex.
    unsafe {
        std::env::remove_var("HF2Q_LLAMA_PERPLEXITY_BIN");
    }
    assert!(
        m.is_missing_binary(),
        "missing binary must surface sentinel; got {m:?}"
    );
    assert!(
        ppl.is_none(),
        "missing binary must yield None ppl (no fake-green); got {ppl:?}"
    );
    assert_eq!(m.exit_code, -1);
    assert_eq!(m.wall_s, -1.0);
    assert_eq!(m.peak_rss_bytes, u64::MAX);
    assert!(
        m.stderr_tail.contains("llama-perplexity"),
        "stderr_tail must name the missing binary; got `{}`",
        m.stderr_tail
    );
}

/// iter-2a #3: the markdown table emits the three new PPL columns
/// in their documented header order, and a half-measured row
/// (hf2q PPL = NotMeasured, peer PPL = Some(5.42)) renders as
/// "—" + "5.4200" + "—" — the em-dash distinguishes deferred from
/// real `0.0000`.
#[test]
fn markdown_table_renders_ppl_columns() {
    // Build a synthetic CellResult with peer PPL measured + hf2q
    // PPL deferred. Wall/RSS sentinels drive the verdict to
    // NotMeasured (hf2q-side missing) per from_measurements.
    let result = CellResult {
        cell: GATE_CELLS[0],
        hf2q: RunMetrics::missing_binary("hf2q (iter-2b deferred)"),
        peer: RunMetrics {
            wall_s: 12.345,
            peak_rss_bytes: 1_024 * 1_024 * 1_024,
            exit_code: 0,
            stderr_tail: "Final estimate: PPL = 5.4200 +/- 0.05000\n".to_string(),
        },
        ppl_hf2q: None,
        ppl_peer: Some(5.42),
        verdict: Verdict::NotMeasured {
            reason: "iter-2a smoke".to_string(),
        },
        ratios: Ratios::not_measured(),
    };
    let md = emit_markdown_table(&[result], "M5 Max, 128 GB", "iter2a01");

    // Header columns added in order at the END of the row.
    for col in ["hf2q PPL", "peer PPL", "PPL ratio"] {
        assert!(md.contains(col), "header missing PPL column `{col}`:\n{md}");
    }

    // Header pipe count must equal data-row pipe count (15 each
    // for 14 columns).
    let header_pipes = md
        .lines()
        .find(|l| l.starts_with("| Model "))
        .map(|l| l.matches('|').count())
        .expect("header present");
    assert_eq!(
        header_pipes, 15,
        "iter-2a header must have 15 pipes (14 columns); got {header_pipes}"
    );
    let data_pipes = md
        .lines()
        .find(|l| l.starts_with("| 27B dense "))
        .map(|l| l.matches('|').count())
        .expect("data row present");
    assert_eq!(
        header_pipes, data_pipes,
        "data row pipe count must match header"
    );

    // Half-measured row: hf2q PPL "—", peer PPL "5.4200", ratio "—".
    let data_line = md
        .lines()
        .find(|l| l.starts_with("| 27B dense "))
        .expect("data row");
    let cells: Vec<&str> = data_line.split('|').map(str::trim).collect();
    // cells[0] is empty (leading pipe) and cells[15] is empty
    // (trailing pipe); the 14 real cells live at indices 1..=14.
    assert_eq!(cells[12], "—", "hf2q PPL must render as em-dash; got `{}`", cells[12]);
    assert_eq!(cells[13], "5.4200", "peer PPL must render to 4 dp; got `{}`", cells[13]);
    assert_eq!(cells[14], "—", "PPL ratio must em-dash when hf2q missing; got `{}`", cells[14]);
}

/// iter-2a #4: when both PPLs are measured, the row renders the
/// ratio to 4 decimals AND the verdict logic on `ppl_tolerance`
/// fires — `from_measurements` builds Pass when within tolerance
/// and Fail when over. We exercise both branches.
#[test]
fn markdown_table_renders_full_ppl_row_when_both_measured() {
    // Within tolerance: 27B dense | None | llama.cpp uncalibrated
    // (ppl_tolerance = 1.02). Pick hf2q = 5.50, peer = 5.45 ⇒
    // ratio 1.00917… which is ≤ 1.02 ⇒ Pass (provided speed + RSS
    // also pass — we set both to identity 1.0× / 1.0×).
    let happy_hf2q = RunMetrics {
        wall_s: 10.0,
        peak_rss_bytes: 1_000_000_000,
        exit_code: 0,
        stderr_tail: String::new(),
    };
    let happy_peer = RunMetrics {
        wall_s: 10.0,
        peak_rss_bytes: 1_000_000_000,
        exit_code: 0,
        stderr_tail: "Final estimate: PPL = 5.4500 +/- 0.05000\n".to_string(),
    };
    let pass_result = CellResult::from_measurements(
        GATE_CELLS[0],
        happy_hf2q.clone(),
        happy_peer.clone(),
        Some(5.50),
        Some(5.45),
    );
    assert!(
        matches!(pass_result.verdict, Verdict::Pass),
        "full-measurement within tolerance must Pass; got {:?}",
        pass_result.verdict
    );
    let pass_md =
        emit_markdown_table(std::slice::from_ref(&pass_result), "M5 Max, 128 GB", "iter2a02");
    let pass_line = pass_md
        .lines()
        .find(|l| l.starts_with("| 27B dense "))
        .expect("data row");
    let pass_cells: Vec<&str> = pass_line.split('|').map(str::trim).collect();
    assert_eq!(pass_cells[12], "5.5000", "hf2q PPL must render to 4 dp");
    assert_eq!(pass_cells[13], "5.4500", "peer PPL must render to 4 dp");
    // ratio = 5.50 / 5.45 ≈ 1.0091743 ⇒ "1.0092" at 4 dp.
    assert_eq!(
        pass_cells[14], "1.0092",
        "PPL ratio must render to 4 dp; got `{}`",
        pass_cells[14]
    );
    assert!(pass_line.contains("PASS"), "verdict cell must say PASS");

    // Over tolerance: hf2q = 5.60, peer = 5.45 ⇒ ratio 1.02752 > 1.02 ⇒ Fail.
    // Reasoning: 5.60 / 5.45 = 1.027523 which exceeds the 1.02 ppl_tolerance
    // for row 0; speed + RSS gates still pass at 1.0× each.
    let fail_result = CellResult::from_measurements(
        GATE_CELLS[0],
        happy_hf2q,
        happy_peer,
        Some(5.60),
        Some(5.45),
    );
    assert!(
        matches!(fail_result.verdict, Verdict::Fail { .. }),
        "full-measurement over tolerance must Fail; got {:?}",
        fail_result.verdict
    );
    if let Verdict::Fail { ref reason } = fail_result.verdict {
        assert!(
            reason.contains("PPL gate"),
            "Fail reason must name PPL gate; got `{reason}`"
        );
    }
    let fail_md = emit_markdown_table(&[fail_result], "M5 Max, 128 GB", "iter2a03");
    assert!(fail_md.contains("FAIL"), "verdict must surface FAIL");
    assert!(
        fail_md.contains("PPL gate"),
        "FAIL reason must mention PPL gate"
    );
}

// =====================================================================
// #[ignore]-gated cells (8) — one per GATE_CELLS row, P11 territory.
// Each calls `run_cell(cell)` end-to-end and asserts the verdict is
// `NotMeasured` this iter (real-model wiring lands in P11).
// =====================================================================

#[test]
#[ignore = "P11 hardware gate: needs 27B-dense GGUF + ~100GB disk + Qwen35Model forward_cpu warm"]
fn cell_0_27b_dense_gguf_uncalibrated_q4km() {
    let r = run_cell(GATE_CELLS[0]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs 27B-dense GGUF + ~100GB disk + Qwen35Model forward_cpu warm"]
fn cell_1_27b_dense_gguf_imatrix_q4km() {
    let r = run_cell(GATE_CELLS[1]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs 27B-dense GGUF + ~100GB disk + Qwen35Model forward_cpu warm"]
fn cell_2_27b_dense_safetensors_dwq46() {
    let r = run_cell(GATE_CELLS[2]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs 27B-dense GGUF + ~100GB disk + Qwen35Model forward_cpu warm"]
fn cell_3_27b_dense_gguf_dwq46_vs_current_pipeline() {
    let r = run_cell(GATE_CELLS[3]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk + Qwen35Model::load_from_gguf for Variant::Moe"]
fn cell_4_apex_moe_gguf_uncalibrated_q4km() {
    let r = run_cell(GATE_CELLS[4]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk + Qwen35Model::load_from_gguf for Variant::Moe"]
fn cell_5_apex_moe_gguf_imatrix_q4km() {
    let r = run_cell(GATE_CELLS[5]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk + Qwen35Model::load_from_gguf for Variant::Moe"]
fn cell_6_apex_moe_safetensors_dwq46() {
    let r = run_cell(GATE_CELLS[6]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}

#[test]
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk + Qwen35Model::load_from_gguf for Variant::Moe"]
fn cell_7_apex_moe_gguf_dwq46_vs_current_pipeline() {
    let r = run_cell(GATE_CELLS[7]);
    assert!(matches!(r.verdict, Verdict::NotMeasured { .. }));
}
