//! Output format backends.
//!
//! Post-ADR-033 P6: the legacy two-pass `gguf.rs` + `safetensors_out.rs`
//! backends were retired alongside the convert/calibrate/quality stack.
//! Only the seek-back GGUF writer subdirectory (`gguf/{writer,types}.rs`)
//! survives; it is consumed by `crate::convert::orchestrator`.
//!
//! `BackendError` is the typed error surface used by both the legacy
//! (now-deleted) trait callers and the new convert-v2 writer paths.
//! It is kept here as a shared module to host that error type.

pub mod gguf;

use thiserror::Error;

/// Errors from output backend operations.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum BackendError {
    #[error("Output format not supported: {format}")]
    UnsupportedFormat { format: String },

    #[error("Backend validation failed: {reason}")]
    ValidationFailed { reason: String },

    #[error("Failed to write output: {reason}")]
    WriteFailed { reason: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
