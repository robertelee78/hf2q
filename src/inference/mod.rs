//! Inference module — wraps mlx-rs for forward passes.
//!
//! Used by both DWQ calibration and quality measurement.
//! Platform-specific #[cfg] is isolated here.
//! No other module imports mlx_rs directly.

pub mod mlx_runner;
pub mod stub_runner;

use thiserror::Error;

/// Errors from inference operations.
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Inference runner is not yet implemented (Epic 4)")]
    NotImplemented,

    #[error("Unsupported platform: MLX requires Apple Silicon")]
    UnsupportedPlatform,

    #[error("Model loading failed: {reason}")]
    LoadFailed { reason: String },

    #[error("Forward pass failed: {reason}")]
    ForwardFailed { reason: String },
}

/// Trait for inference runners.
///
/// All implementations must be Send + Sync.
pub trait InferenceRunner: Send + Sync {
    /// Human-readable name.
    fn name(&self) -> &str;

    /// Whether this runner is available on the current platform.
    fn is_available(&self) -> bool;
}
