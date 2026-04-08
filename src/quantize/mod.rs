//! Quantization module — transforms TensorMap into QuantizedModel.
//!
//! Implements the `Quantizer` trait with dispatch by QuantMethod.
//! Sub-modules:
//! - `static_quant`: f16, q8, q4, q2 round-to-nearest
//! - `mixed`: Mixed-bit with --sensitive-layers
//! - `dwq`: DWQ calibration via InferenceRunner

pub mod apex;
pub mod dwq;
pub mod dwq_activation;
pub mod mixed;
pub mod sensitivity;
pub mod static_quant;

use thiserror::Error;

use crate::ir::{QuantizedTensor, TensorRef};
use crate::progress::ProgressReporter;

/// Errors from quantization operations.
#[derive(Error, Debug)]
pub enum QuantizeError {
    #[error("Quantization failed for tensor '{tensor}': {reason}")]
    TensorQuantizeFailed { tensor: String, reason: String },

    #[error("Unsupported quantization method: {method}")]
    UnsupportedMethod { method: String },

    #[error("Group size {group_size} does not evenly divide tensor dimension {dim} for tensor '{tensor}'")]
    GroupSizeMismatch {
        tensor: String,
        group_size: usize,
        dim: usize,
    },

    #[error("IR error: {0}")]
    IrError(#[from] crate::ir::IrError),
}

/// Configuration for quantizing a single layer/tensor.
#[derive(Debug, Clone)]
pub struct LayerQuantConfig {
    /// Bit width for quantization
    pub bits: u8,
    /// Group size for block quantization
    pub group_size: usize,
    /// Whether to preserve this tensor at full precision
    pub preserve: bool,
}

/// Trait for quantization implementations.
///
/// All implementations must be Send + Sync for rayon parallelism.
pub trait Quantizer: Send + Sync {
    /// Human-readable name of this quantization method.
    fn name(&self) -> &str;

    /// Whether this quantizer requires calibration data (forward passes).
    fn requires_calibration(&self) -> bool;

    /// Quantize a single tensor according to the given config.
    fn quantize_tensor(
        &self,
        tensor: &TensorRef,
        config: &LayerQuantConfig,
    ) -> Result<QuantizedTensor, QuantizeError>;
}

/// Validate that group size evenly divides the tensor dimension.
///
/// Returns an error if the group size does not divide the last dimension of the tensor.
/// This is used for strict validation when exact group alignment is required.
pub fn validate_group_size(tensor: &TensorRef, group_size: usize) -> Result<(), QuantizeError> {
    if group_size == 0 || !tensor.is_weight() {
        return Ok(());
    }
    let dim = tensor.numel();
    if dim > group_size && dim % group_size != 0 {
        return Err(QuantizeError::GroupSizeMismatch {
            tensor: tensor.name.clone(),
            group_size,
            dim,
        });
    }
    Ok(())
}

/// Quantize an entire TensorMap using the given quantizer.
pub fn quantize_model(
    tensor_map: &crate::ir::TensorMap,
    metadata: &crate::ir::ModelMetadata,
    quantizer: &dyn Quantizer,
    bits: u8,
    group_size: usize,
    progress: &ProgressReporter,
) -> Result<crate::ir::QuantizedModel, QuantizeError> {
    use std::collections::HashMap;

    let total = tensor_map.len() as u64;
    let pb = progress.bar(total, "Quantizing");

    let mut quantized_tensors = HashMap::new();

    // Sort tensor names for deterministic output
    let mut tensor_names: Vec<&String> = tensor_map.tensors.keys().collect();
    tensor_names.sort();

    for name in tensor_names {
        let tensor = &tensor_map.tensors[name];

        // Log group size misalignment (non-fatal: quantizer will pad)
        if let Err(e) = validate_group_size(tensor, group_size) {
            tracing::debug!("{}", e);
        }

        let config = LayerQuantConfig {
            bits,
            group_size,
            preserve: !tensor.is_weight(),
        };

        let quantized = quantizer.quantize_tensor(tensor, &config)?;
        quantized_tensors.insert(name.clone(), quantized);

        pb.inc(1);
    }

    pb.finish_with_message(format!("Quantized {} tensors", quantized_tensors.len()));

    Ok(crate::ir::QuantizedModel {
        metadata: metadata.clone(),
        tensors: quantized_tensors,
        quant_method: quantizer.name().to_string(),
        group_size,
        bits,
    })
}
