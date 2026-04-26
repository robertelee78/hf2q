//! Arch-table-driven scaffolding (ADR-012 Decision 20).
//!
//! Every architecture (`qwen35`, `qwen35moe`, future `gemma4`, `ministral`,
//! `deepseekv3`, …) registers into a single [`ArchRegistry`] singleton.  All
//! conformance tooling — the `hf2q smoke` binary subcommand (Decision 16),
//! MTP round-trip gate (Decision 19), mmproj round-trip (Decision 18 Layer B),
//! and DWQ PPL/KL eval (Decision 17) — reads its per-arch knobs from the
//! registered [`ArchEntry`] rather than hardcoded Qwen paths.
//!
//! # What lives here
//!
//! * [`registry`] — the [`ArchEntry`] struct and the [`Registry`] singleton.
//! * [`catalog`] — the [`TensorCatalog`] type plus per-tensor metadata.
//! * [`conformance`] — arch-generic helpers (smoke, round-trip, future PPL/KL).
//! * [`smoke`] — the `hf2q smoke …` subcommand implementation.
//! * [`entries`] — one Rust file per registered arch.  ADR-012 P8 ships only
//!   `qwen35` and `qwen35moe`; future arches add their own file in their own
//!   ADR.  No placeholder files.

pub mod catalog;
pub mod conformance;
pub mod entries;
pub mod registry;
pub mod smoke;

// Public re-exports — kept narrow to avoid bringing in unused-import warnings
// in `main.rs` that only consumes the smoke entry points and the registry.
// Catalog / TensorSpec / EvalCorpus / QualityThresholds are reachable via
// `crate::arch::{catalog, registry}::…` for tests and future phases.
#[allow(unused_imports)]
pub use registry::{ArchEntry, Registry, RegistryError};
#[allow(unused_imports)]
pub use smoke::{run_smoke, SmokeOptions, SmokeQuant};

/// Human-readable list of every arch the registry knows about.  Used for
/// "unknown arch" error messages so failures consistently surface what *is*
/// available without per-arch placeholder branches.
pub fn known_arches() -> Vec<&'static str> {
    Registry::global().known()
}
