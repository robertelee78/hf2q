//! ADR-014 P10 iter-1 — shared test infrastructure for the
//! peer-parity benchmark harness (`tests/peer_parity_gates.rs`) and the
//! P9-deferred mlx-lm cross-validation gates
//! (`tests/safetensors_mlx_lm_round_trip.rs`).
//!
//! Sub-modules:
//! - [`metrics`] — `RunMetrics` struct + `missing_binary` sentinel
//!   constructor. Every subprocess wrapper returns this type.
//! - [`llama_cpp_runner`] — wrappers around `llama-quantize`,
//!   `convert_hf_to_gguf`, and `llama-imatrix`. Each shells out via
//!   `std::process::Command` (no `shell=true`); on missing binary,
//!   tracing::warn fires and the wrapper returns
//!   [`metrics::RunMetrics::missing_binary`] so the harness can report
//!   `Verdict::NotMeasured` rather than panic.
//! - [`mlx_lm_runner`] — wrappers around `python3 -c "import mlx_lm; …"`
//!   for the mlx-lm cross-validation gates. Treats `ImportError` as a
//!   missing module and returns the same `missing_binary` sentinel.
//!
//! All wrappers obey the ADR-014 P10 sovereignty contract (Decision 21):
//! no link to mlx-lm at build time, runtime subprocess for
//! cross-validation only, behind explicit `--ignored` opt-in for the
//! gates that require real peers.
//!
//! Sovereignty + missing-binary contract:
//! - subprocess wrappers must use `std::process::Command` with explicit
//!   binary path lookup — no `shell=true`, no `bash -c`;
//! - missing binary → `tracing::warn!` + `RunMetrics::missing_binary`;
//!   never panic, never silent fallback, never fake-green;
//! - the harness inspects `wall_s == -1.0` to surface
//!   `Verdict::NotMeasured` for that cell.
//!
//! Allow-list rationale: `dead_code` is permitted at the module level
//! because individual integration test binaries only consume a subset
//! of these helpers (each `tests/*.rs` becomes its own crate per
//! Cargo's integration-test model — symbols not referenced by that
//! particular file are dead from its perspective).

#![allow(dead_code)]

pub mod llama_cpp_runner;
pub mod metrics;
pub mod mlx_lm_runner;
