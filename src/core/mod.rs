//! `hf2q` foundation module — types and traits shared by both the
//! convert pipeline and the inference/serve pipeline.
//!
//! This module is the in-place precursor to the planned `hf2q-core`
//! crate (workspace v0.1.0 split). It exists so that every type or
//! trait crossing the convert ↔ serve boundary lives in exactly one
//! place, with no circular module dependencies.
//!
//! ## Boundary rule
//!
//! Anything referenced by **both** `crate::{quantize, backends, input,
//! calibrate, quality, models, intelligence}` (the future
//! `hf2q-convert` set) **and** `crate::{inference, serve}` (the future
//! `hf2q-serve` set) lives here. Anything specific to one side stays
//! in its current module.
//!
//! Cyclic dependencies (notably `calibrate` ↔ `inference` around DWQ
//! activation capture) are resolved by extracting the *trait* into
//! this module while the concrete implementations stay in their
//! respective sides.
//!
//! ## Planned submodules
//!
//! Substeps fill these in sequentially; each leaves a `#[deprecated]`
//! re-export shim at the old path so in-flight branches still compile
//! until the workspace split removes the shims.
//!
//! - `provenance/` — GGUF metadata key constants, source-bundle
//!   sha256, `SourceShard` (was `src/serve/{provenance,cache}`).
//! - `integrity` — `ShardIntegrity`, `verify_shard`, `verify_repo`
//!   (was `src/input/integrity`).
//! - `hardware` — `HardwareProfile`, `HardwareProfiler`,
//!   `lookup_memory_bandwidth_gbs` (was `src/intelligence/hardware`).
//! - `traits::activation_capture` — `ActivationCapture` trait,
//!   `LayerActivations`, `MockActivationCapture` (was
//!   `src/inference/models/qwen35/activation_capture`).
//! - `mlx_linear` — `MlxAffineLinear`, `MlxAffineLinearBytes` (was
//!   `src/calibrate/mlx_safetensors_loader`).
//! - `kernel_parity` — `assert_kernel_equivalence` debug helper (was
//!   `src/quality/kernel_parity`).
//!
//! `src/backends/chat_templates.rs` does NOT move here; it lands in
//! `src/serve/chat_templates.rs` (serve-only concern; only consumer is
//! `src/serve/mod.rs`).

// B1.2 landed: provenance keys + Provenance enum + detect + sha256 file
// helpers.  `SourceShard` + `compute_source_bundle_sha256` deferred to
// B1.3 (alongside `integrity` migration — SourceShard's `from_integrity`
// adapter depends on ShardIntegrity from `src/input/integrity`).
pub mod integrity;
pub mod provenance;
pub mod sha256;
