//! Output format backends — each backend writes a specific model format.
//!
//! Trait-based architecture: adding a new format = adding a new file + registry entry.
//! Every backend validates its output before writing.

#[allow(dead_code)]
pub mod coreml;
#[allow(dead_code)]
pub mod gguf;
pub mod mlx;
#[allow(dead_code)]
pub mod nvfp4;

use std::path::Path;

use thiserror::Error;

use crate::ir::{FormatWarning, OutputManifest, QuantizedModel};
use crate::progress::ProgressReporter;

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

/// Trait for output format backends.
///
/// All implementations must be Send + Sync for future parallelism.
#[allow(dead_code)]
pub trait OutputBackend: Send + Sync {
    /// Human-readable name of this output format.
    fn name(&self) -> &str;

    /// Validate the quantized model for this format before writing.
    /// Returns warnings (non-fatal issues) or errors (fatal).
    fn validate(&self, model: &QuantizedModel) -> Result<Vec<FormatWarning>, BackendError>;

    /// Write the quantized model to the output directory.
    fn write(
        &self,
        model: &QuantizedModel,
        input_dir: &Path,
        output_dir: &Path,
        progress: &ProgressReporter,
    ) -> Result<OutputManifest, BackendError>;
}
