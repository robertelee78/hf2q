//! Output format backends — each backend writes a specific model format.
//!
//! Trait-based architecture: adding a new format = adding a new file + registry entry.
//! Every backend validates its output before writing.

pub mod coreml;
#[allow(dead_code)]
pub mod gguf;
pub mod mlx;
#[allow(dead_code)]
pub mod nvfp4;

use std::collections::HashMap;
use std::path::Path;

use thiserror::Error;

use crate::ir::{FormatWarning, OutputManifest, QuantizedModel, ModelMetadata, TensorMap};
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

    /// Quantize original f16 weights using the backend's native algorithm and write output.
    ///
    /// Some formats (e.g., MLX) have specific quantization algorithms baked into their
    /// inference kernels. For these, the backend must perform quantization itself rather
    /// than receiving pre-quantized IR output.
    ///
    /// Default: not supported (returns error). Backends that support this override it.
    fn quantize_and_write(
        &self,
        tensor_map: &TensorMap,
        metadata: &ModelMetadata,
        bits: u8,
        group_size: usize,
        bit_overrides: Option<&HashMap<String, u8>>,
        input_dir: &Path,
        output_dir: &Path,
        progress: &ProgressReporter,
    ) -> Result<OutputManifest, BackendError> {
        let _ = (tensor_map, metadata, bits, group_size, bit_overrides, input_dir, output_dir, progress);
        Err(BackendError::UnsupportedFormat {
            format: format!("{} does not support native quantization", self.name()),
        })
    }

    /// Whether this backend requires native quantization (quantize_and_write)
    /// rather than receiving pre-quantized IR.
    fn requires_native_quantization(&self) -> bool {
        false
    }
}
