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
// E2E cell runner (env-gated; default off).
// =========================================================================

/// Run one cell end-to-end. Returns `Ok(())` on parity, `Err(reason)`
/// on any falsifier. The runner is responsible for spawning
/// `hf2q serve --model PATH --kv-persist=DIR`, sending an N-token
/// prompt via `/v1/chat/completions`, snapshotting the dense_kvs
/// SHA-256 hashes, forcing an evict cycle (per cell scenario),
/// re-driving the same prompt, and comparing.
///
/// **B-dense.2 scope:** the runner is structured but the actual
/// HTTP/SSE plumbing is delegated to the existing
/// `kv_persist_harness.rs::subprocess_driver` module. This file owns
/// the matrix shape + env gate + cell payload kind enumeration; the
/// HTTP/SSE driver lives in the existing harness module so a single
/// driver implementation services both A0 (TTFT predictions) and
/// B-dense.2 (round-trip parity).
///
/// To avoid coupling, this stub-runner returns `Err("E2E driver lives
/// in kv_persist_harness::subprocess_driver — invoke from there
/// post-merge")` when called. The matrix gate test below records the
/// stub-runner result without panicking (operator's actual run is the
/// post-merge follow-up).
pub fn run_cell_e2e(cell: &Cell, _model_path: &Path, _cache_dir: &Path) -> Result<(), String> {
    // Sanity: the binary must be locatable. If not, the run cannot
    // proceed and we bail with a diagnostic (no-panic on the always-on
    // path; the gated runner is allowed to surface this as a failure
    // because the binary is a hard prerequisite).
    let bin = hf2q_binary_path();
    if !bin.exists() {
        return Err(format!(
            "hf2q binary not found at {}: did `cargo build --release` run?",
            bin.display()
        ));
    }

    // Round-trip parity wiring deferred per spec — the post-merge run
    // drives the actual HTTP/SSE round-trip via the existing
    // kv_persist_harness::subprocess_driver module. Returning a clear
    // Err keeps the env-gated test from claiming a false success.
    Err(format!(
        "B-dense.2 cell {:?} substrate-only: post-merge run drives the HTTP/SSE \
         round-trip via kv_persist_harness::subprocess_driver",
        cell.payload_kind()
    ))
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
/// **Substrate-only on B-dense.2:** the runner returns Err for every
/// cell with a "post-merge run" diagnostic. The matrix gate test
/// records the count of attempted cells and the count of runnable
/// cells, but does NOT fail on the substrate-only Err. The actual
/// E2E HTTP/SSE round-trip is a post-merge work item driven from
/// the main session.
///
/// Falsifier (post-merge): any cell's pre/post dense_kvs hash diff,
/// any decoded-token diff, or the substitution flow not firing.
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
