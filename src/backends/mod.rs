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
    ///
    /// ADR-014 P9 iter-1: the default is `false` and **no shipped backend**
    /// overrides it any more.  Both [`gguf::GgufBackend`] and
    /// [`safetensors_out::SafetensorsBackend`] now flow through the
    /// Calibrator → Quantizer → `QuantizedModel` → `write` chain so the
    /// IR-level quantize loop is the single dispatch path.
    ///
    /// The trait method itself is retained so future backends that *do*
    /// own a non-IR algorithm (e.g. an MLX-Python emitter) can still opt
    /// in.  Until such a backend lands, every call site in
    /// `cmd_convert` (the ~6 sites below the Phase 2 dispatch) takes the
    /// `false` branch — Chesterton's fence on the dead arms is
    /// intentional.
    fn requires_native_quantization(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::gguf::GgufBackend;
    use crate::backends::safetensors_out::SafetensorsBackend;

    /// ADR-014 P9 iter-1 §S1 — `SafetensorsBackend` no longer overrides
    /// `requires_native_quantization`; it inherits the trait default
    /// (`false`) so the cmd_convert dispatch routes safetensors through
    /// the same Calibrator → Quantizer chain as GGUF.
    #[test]
    fn safetensors_backend_does_not_require_native_quantization() {
        let backend = SafetensorsBackend::new();
        assert!(
            !backend.requires_native_quantization(),
            "SafetensorsBackend must inherit the trait default \
             `requires_native_quantization() = false` (ADR-014 P9 iter-1 §S1)"
        );
    }

    /// ADR-014 P9 iter-1 §S1 — the trait default is `false`. Codified so
    /// future backend authors aren't surprised by the dispatch behaviour.
    /// Uses `GgufBackend` (also no override) as the default-witness; if
    /// either backend ever regains a native-quant override the assert
    /// must move first.
    #[test]
    fn output_backend_default_requires_native_quantization_is_false() {
        let backend = GgufBackend::new();
        assert!(
            !backend.requires_native_quantization(),
            "OutputBackend trait default (and GgufBackend inherit) must \
             remain `requires_native_quantization() = false` until a \
             future backend deliberately opts in"
        );
    }
}
