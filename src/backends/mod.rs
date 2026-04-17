//! Output format backends — each backend writes a specific model format.
//!
//! Trait-based architecture: adding a new format = adding a new file + registry entry.
//! Every backend validates its output before writing.

pub mod gguf;
pub mod safetensors_out;

use std::collections::HashMap;
use std::path::Path;

use thiserror::Error;

use crate::ir::{FormatWarning, OutputManifest, QuantizedModel, ModelMetadata, TensorMap};
use crate::progress::ProgressReporter;

/// Quantization knobs for backends that implement their own quantization
/// algorithm (see [`OutputBackend::quantize_and_write`]).
///
/// `bits` is the default bit-width; `bit_overrides` is an optional
/// per-tensor override map (e.g. keep embeddings at f16, lm_head at q6).
pub struct QuantizeConfig<'a> {
    pub bits: u8,
    pub group_size: usize,
    pub bit_overrides: Option<&'a HashMap<String, u8>>,
}

/// Errors from output backend operations.
#[derive(Error, Debug)]
pub enum BackendError {
    #[error("Output format not supported: {format}")]
    UnsupportedFormat { format: String },

    #[error("Backend validation failed: {reason}")]
    #[allow(dead_code)]
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
    /// Some formats have specific quantization algorithms baked into their
    /// inference kernels. For these, the backend must perform quantization itself rather
    /// than receiving pre-quantized IR output.
    ///
    /// Default: not supported (returns error). Backends that support this override it.
    fn quantize_and_write(
        &self,
        tensor_map: &TensorMap,
        metadata: &ModelMetadata,
        config: &QuantizeConfig<'_>,
        input_dir: &Path,
        output_dir: &Path,
        progress: &ProgressReporter,
    ) -> Result<OutputManifest, BackendError> {
        let _ = (tensor_map, metadata, config, input_dir, output_dir, progress);
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
