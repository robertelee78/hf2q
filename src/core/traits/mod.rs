//! Cross-boundary traits — runtime-agnostic interfaces shared by
//! `crate::{calibrate, quantize}` (the future `hf2q-convert` crate)
//! and `crate::{inference, serve}` (the future `hf2q-serve` crate).
//!
//! Migrated 2026-05-16 at B1.5 of the v0.1.0 workspace split.  Each
//! submodule here defines a trait whose concrete implementations live
//! on one side of the convert ↔ serve split, but whose callers live
//! on the other side — extracting the trait into core resolves the
//! cyclic dependency that would otherwise force callers to reach
//! across the crate boundary.
//!
//! ## Submodules
//!
//! - [`activation_capture`] — DWQ / imatrix calibration interface.
//!   Trait + `LayerActivations` data type + `MockActivationCapture`
//!   reference impl.  Concrete `RealActivationCapture` for Qwen35 lives
//!   in `src/inference/models/qwen35/activation_capture_real.rs` and
//!   `impl`s this trait via the absolute path.

pub mod activation_capture;
