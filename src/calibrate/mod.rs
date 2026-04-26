//! Calibration module — pure-Rust implementations of weight-quantization
//! calibrators used by hf2q's IR-quantize and native-quantize paths.
//!
//! ## ADR-014 P6 — Imatrix calibrator (this module's first inhabitant)
//!
//! [`imatrix`] is the pure-Rust port of llama.cpp's importance-matrix
//! algorithm (`/opt/llama.cpp/tools/imatrix/imatrix.cpp`). At a forward
//! pass through a calibration corpus, the calibrator captures the
//! input activation vector at every Linear layer and accumulates
//! `x[col]² · 1.0` per column. The mean of squared activations becomes
//! the **per-column importance weight** consumed by the per-column-
//! weighted MSE inside the k-quant codebook search at quantize time
//! (ADR-014 Decision 11, lands in P7 alongside the
//! [`Calibrator`]-trait orthogonal split).
//!
//! ## ADR-014 P7 — orthogonal `Calibrator × OutputFormat` split (future)
//!
//! P7 will add `Calibrator` (trait), `NoneCalibrator`, `DwqCalibrator`,
//! `ImatrixCalibrator` impls + the path migration of
//! `dwq.rs`/`dwq_activation.rs`/`sensitivity.rs`/`apex.rs` from
//! `src/quantize/` into this module. Until then, only the imatrix
//! algorithm lives here and the eager DWQ path stays in
//! `src/quantize/`.

pub mod cache;
pub mod imatrix;
