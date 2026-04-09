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
            preserve: !tensor.is_weight() || tensor.is_vision_tensor(),
        };

        if tensor.is_vision_tensor() && tensor.is_weight() {
            tracing::debug!("Preserving vision tensor as F16: {}", name);
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, ModelMetadata, TensorMap, TensorRef};
    use crate::quantize::static_quant::StaticQuantizer;

    fn dummy_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "TestArch".to_string(),
            model_type: "test".to_string(),
            param_count: 1000,
            hidden_size: 64,
            num_layers: 1,
            layer_types: vec!["attention".to_string()],
            num_attention_heads: 4,
            num_kv_heads: None,
            vocab_size: 256,
            dtype: "float16".to_string(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some(128),
            raw_config: serde_json::Value::Null,
        }
    }

    /// Create a fake F16 weight tensor with valid data for the given shape.
    fn make_tensor(name: &str, shape: Vec<usize>) -> TensorRef {
        let numel: usize = shape.iter().product();
        TensorRef {
            name: name.to_string(),
            shape,
            dtype: DType::F16,
            data: vec![0u8; numel * 2], // F16 = 2 bytes per element
        }
    }

    #[test]
    fn test_vision_tensor_preserved() {
        let mut tensor_map = TensorMap::new();

        // Vision weight tensor — should be preserved despite being a weight
        tensor_map.insert(make_tensor(
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight",
            vec![64, 64],
        ));

        // Regular weight tensor — should be quantized
        tensor_map.insert(make_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            vec![64, 64],
        ));

        let metadata = dummy_metadata();
        let quantizer = StaticQuantizer::new("q4").unwrap();
        let progress = crate::progress::ProgressReporter::new();

        let result = quantize_model(&tensor_map, &metadata, &quantizer, 4, 32, &progress).unwrap();

        // Vision tensor should be preserved (F16)
        let vision_t = &result.tensors["model.vision_tower.encoder.layers.0.self_attn.q_proj.weight"];
        assert!(
            vision_t.quant_info.preserved,
            "Vision weight tensor should be preserved at full precision"
        );

        // Regular weight tensor should NOT be preserved (quantized)
        let regular_t = &result.tensors["model.layers.0.self_attn.q_proj.weight"];
        assert!(
            !regular_t.quant_info.preserved,
            "Regular weight tensor should be quantized, not preserved"
        );
    }
}
