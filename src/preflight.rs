//! Pre-conversion validation (Epic 2).
//!
//! Runs before any expensive work to catch configuration issues early.
//! Not yet implemented.

use thiserror::Error;

/// Errors from preflight validation.
#[derive(Error, Debug)]
pub enum PreflightError {
    #[error("Preflight validation is not yet implemented (Epic 2)")]
    NotImplemented,

    #[error("Unsupported layer types found: {layers:?}. Use --unsupported-layers=passthrough to proceed with f16 fallback.")]
    UnsupportedLayers { layers: Vec<String> },

    #[error("Output format '{format}' is not compatible with model architecture '{architecture}': {reason}")]
    IncompatibleFormat {
        format: String,
        architecture: String,
        reason: String,
    },

    #[error("Sensitive layer range {range} exceeds model layer count ({max_layers})")]
    InvalidLayerRange { range: String, max_layers: u32 },

    #[error("Insufficient disk space: need {needed_bytes} bytes, have {available_bytes} bytes")]
    InsufficientDisk {
        needed_bytes: u64,
        available_bytes: u64,
    },

    #[error("Insufficient memory: estimated {needed_bytes} bytes, available {available_bytes} bytes")]
    InsufficientMemory {
        needed_bytes: u64,
        available_bytes: u64,
    },
}
