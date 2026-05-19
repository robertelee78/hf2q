//! `QuantizeError` — typed error taxonomy for the new ggml_quants
//! pipeline. Per ADR-033 §"Quantizer trait" and the no-fallback rule
//! ([[feedback-no-loop-suppression-2026-05-17]]): every error is typed
//! at the trait boundary; the kernel layer never silently demotes to
//! a different format.

use thiserror::Error;

use super::GgmlType;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum QuantizeError {
    /// The requested `GgmlType` has no `Quantizer` impl in this build.
    /// Per ADR Decision §"Quantizer trait", this is the no-fallback
    /// guard — the pipeline returns this typed error rather than
    /// silently emitting F16.
    #[error("no Quantizer impl for ggml_type {0:?}")]
    NoQuantizerForType(GgmlType),

    /// Numeric `u32` value didn't decode to a known `GgmlType` (holes
    /// in the enum's numeric space).
    #[error("unknown ggml_type value {0} (not in supported set)")]
    UnknownGgmlType(u32),

    /// `n_per_row` isn't a multiple of the type's `block_size`. Per
    /// ADR §"shape_fallback contract" this is a hard error, not a
    /// silent F16 demotion.
    #[error("n_per_row {n_per_row} not a multiple of block_size {block_size} for {ggml_type:?}")]
    NotBlockAligned {
        ggml_type: GgmlType,
        n_per_row: usize,
        block_size: usize,
    },

    /// `src.len()` isn't a multiple of `n_per_row`.
    #[error("src len {src_len} not a multiple of n_per_row {n_per_row}")]
    NotRowAligned { src_len: usize, n_per_row: usize },

    /// Imatrix vector length doesn't match `n_per_row`. Per
    /// llama.cpp's convention (and codex's 0bd0e7eb review), the
    /// imatrix is per-row (length `n_per_row`), reused across rows
    /// without advancement.
    #[error("imatrix len {im_len} must equal n_per_row {n_per_row} (per-row weights)")]
    ImatrixLenMismatch { n_per_row: usize, im_len: usize },
}
