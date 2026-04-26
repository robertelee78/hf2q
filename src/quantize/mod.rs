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
pub mod intermediate_moe_q8;
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

/// ADR-014 P2 iter-1: streaming quantize loop.
///
/// Consumes a [`crate::ir::lazy::LazyTensorMap`] one tensor at a time:
/// materialise → (optional bf16→f16) → quantise → accumulate. The
/// per-iteration input bytes are dropped between quantizations, so the
/// peak resident bytes for input tensors are bounded by ~one tensor at
/// a time (~750 MB apex MoE BF16 merged tile) instead of the ~70 GB
/// eager `materialize_all + convert_bf16_to_f16 + quantize_model`
/// pattern that this function replaces on the IR-quantize path.
///
/// **Memory-budget delta vs. `quantize_model`** (apex MoE BF16, 27B):
///
/// | Stage                        | Eager pipeline | Streaming (this fn) |
/// | ---------------------------- | -------------- | ------------------- |
/// | Input tensors resident       | ~70 GB         | ~750 MB peak/tensor |
/// | Output `QuantizedModel`      | ~30 GB         | ~30 GB              |
/// | Total peak                   | ~100 GB        | ~30 GB              |
///
/// The output `QuantizedModel` accumulation is unchanged from the
/// eager path — it is the next P2 iter's target (refactor backends
/// to write per-tensor as it's quantised, eliminating the
/// `QuantizedModel` accumulation entirely; ~750 MB peak total).
///
/// **bf16_to_f16:** matches the existing eager-path conversion gate
/// (`if !backend.requires_native_quantization() { convert_bf16_to_f16 }`
/// at `src/main.rs::cmd_convert` Phase 2 prelude). Caller passes
/// `true` for the IR-quantize path (GGUF), `false` for the native
/// quantization path (DWQ on safetensors). Conversion happens
/// per-tensor inside the streaming loop, immediately after materialise.
///
/// **Determinism:** [`crate::ir::lazy::LazyTensorMap`]'s `BTreeMap`
/// iteration order is deterministic; the resulting `QuantizedModel`'s
/// `tensors` `HashMap` is order-independent so byte-identical output
/// to `quantize_model` on the same input is guaranteed (verified by
/// `tests::quantize_streaming_byte_identical_to_quantize_model`).
///
/// **Error path:** materialise / bf16-cast / quantize errors all bail
/// the loop with the failed tensor's name in the error context.
/// Partial accumulation is dropped on `?`.
pub fn quantize_streaming(
    map: crate::ir::lazy::LazyTensorMap,
    metadata: &crate::ir::ModelMetadata,
    quantizer: &dyn Quantizer,
    bits: u8,
    group_size: usize,
    progress: &ProgressReporter,
    bf16_to_f16: bool,
) -> Result<crate::ir::QuantizedModel, QuantizeError> {
    use std::collections::HashMap;

    let total = map.len() as u64;
    let pb = progress.bar(total, "Quantizing (streaming)");

    let mut quantized_tensors = HashMap::new();
    let mut bf16_converted = 0usize;

    // BTreeMap iteration → deterministic order. Each iteration:
    // materialise (consumes the LazyTensor, drops the underlying mmap
    // ref count when this is the last reference) → optional
    // bf16-to-f16 cast → quantise → accumulate → drop materialised
    // input.
    for (name, lazy) in map.into_iter() {
        let mut tensor =
            lazy.materialize()
                .map_err(|e| QuantizeError::TensorQuantizeFailed {
                    tensor: name.clone(),
                    reason: format!("materialize: {e}"),
                })?;

        if bf16_to_f16 && tensor.dtype == crate::ir::DType::BF16 {
            tensor = tensor.to_f16().map_err(|e| QuantizeError::TensorQuantizeFailed {
                tensor: name.clone(),
                reason: format!("bf16→f16: {e}"),
            })?;
            bf16_converted += 1;
        }

        if let Err(e) = validate_group_size(&tensor, group_size) {
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

        let quantized = quantizer.quantize_tensor(&tensor, &config)?;
        quantized_tensors.insert(name, quantized);

        // `tensor` (owned TensorRef with materialised bytes) drops here
        // — frees per-iteration input memory before the next tensor is
        // materialised.
        pb.inc(1);
    }

    pb.finish_with_message(format!(
        "Quantized {} tensors (streaming){}",
        quantized_tensors.len(),
        if bf16_converted > 0 {
            format!(", {} bf16→f16", bf16_converted)
        } else {
            String::new()
        }
    ));

    Ok(crate::ir::QuantizedModel {
        metadata: metadata.clone(),
        tensors: quantized_tensors,
        quant_method: quantizer.name().to_string(),
        group_size,
        bits,
    })
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
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
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

    /// ADR-014 P2 iter-1 byte-identity regression:
    /// `quantize_streaming(LazyTensorMap, ...)` produces a
    /// `QuantizedModel` byte-equal to `quantize_model(&TensorMap, ...)`
    /// on the same fixture.
    #[test]
    fn quantize_streaming_byte_identical_to_quantize_model() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};

        let mut tensor_map = TensorMap::new();
        let mut lazy_map = LazyTensorMap::new();

        // Build distinguishing payloads so a swap or drop would be caught.
        let make_payload = |seed: u8, len: usize| -> Vec<u8> {
            (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
        };

        let fixtures: Vec<(&str, Vec<usize>, DType, Vec<u8>)> = vec![
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![64, 64],
                DType::F16,
                make_payload(0x10, 64 * 64 * 2),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                vec![64, 32],
                DType::F16,
                make_payload(0x20, 64 * 32 * 2),
            ),
            (
                "model.layers.0.input_layernorm.weight",
                vec![64],
                DType::F16,
                make_payload(0x30, 64 * 2),
            ),
        ];

        for (name, shape, dtype, data) in &fixtures {
            tensor_map.insert(TensorRef {
                name: name.to_string(),
                shape: shape.clone(),
                dtype: *dtype,
                data: data.clone(),
            });
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), shape.clone(), *dtype),
                data.clone(),
            ));
        }

        let metadata = dummy_metadata();
        let quantizer = StaticQuantizer::new("q4").unwrap();
        let progress = crate::progress::ProgressReporter::new();

        let eager = quantize_model(&tensor_map, &metadata, &quantizer, 4, 32, &progress).unwrap();
        let streaming = quantize_streaming(
            lazy_map, &metadata, &quantizer, 4, 32, &progress, /* bf16_to_f16 */ false,
        )
        .unwrap();

        assert_eq!(eager.tensors.len(), streaming.tensors.len());
        for (name, _shape, _dtype, _data) in &fixtures {
            let e = eager.tensors.get(*name).unwrap();
            let s = streaming.tensors.get(*name).unwrap();
            assert_eq!(e.shape, s.shape, "{name} shape");
            assert_eq!(e.original_dtype, s.original_dtype, "{name} dtype");
            assert_eq!(e.data, s.data, "{name} bytes");
            assert_eq!(e.quant_info.method, s.quant_info.method, "{name} method");
            assert_eq!(e.quant_info.bits, s.quant_info.bits, "{name} bits");
            assert_eq!(
                e.quant_info.preserved, s.quant_info.preserved,
                "{name} preserved"
            );
        }
        // Quantization metadata fields preserved.
        assert_eq!(eager.quant_method, streaming.quant_method);
        assert_eq!(eager.group_size, streaming.group_size);
        assert_eq!(eager.bits, streaming.bits);
    }

    /// `bf16_to_f16: true` casts BF16 inputs before quantization;
    /// equivalent to `tensor_map.convert_bf16_to_f16()` then
    /// `quantize_model` on the eager path.
    #[test]
    fn quantize_streaming_bf16_to_f16_path() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};

        // Synthetic bf16 tensor — known value 1.0.
        let bf16_one_bytes = half::bf16::from_f32(1.0).to_le_bytes();
        let bf16_data: Vec<u8> = (0..64 * 64 * 2).step_by(2)
            .flat_map(|_| bf16_one_bytes)
            .collect();

        let mut tensor_map = TensorMap::new();
        let mut lazy_map = LazyTensorMap::new();
        for name in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.input_layernorm.weight",
        ] {
            let shape: Vec<usize> = if name.contains("layernorm") {
                vec![64]
            } else {
                vec![64, 64]
            };
            let len = shape.iter().product::<usize>() * 2;
            let data: Vec<u8> = (0..len).map(|i| (i & 0xFF) as u8).collect();
            tensor_map.insert(TensorRef {
                name: name.to_string(),
                shape: shape.clone(),
                dtype: DType::BF16,
                data: data.clone(),
            });
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), shape, DType::BF16),
                data,
            ));
        }
        let _ = bf16_data; // unused after refactor; keep above for shape ref

        // Eager path: convert_bf16_to_f16, then quantize_model.
        tensor_map.convert_bf16_to_f16().unwrap();
        let metadata = dummy_metadata();
        let quantizer = StaticQuantizer::new("q4").unwrap();
        let progress = crate::progress::ProgressReporter::new();
        let eager = quantize_model(&tensor_map, &metadata, &quantizer, 4, 32, &progress).unwrap();

        // Streaming path: bf16_to_f16=true does the same conversion per-tensor.
        let streaming = quantize_streaming(
            lazy_map, &metadata, &quantizer, 4, 32, &progress, /* bf16_to_f16 */ true,
        )
        .unwrap();

        assert_eq!(eager.tensors.len(), streaming.tensors.len());
        for name in eager.tensors.keys() {
            let e = eager.tensors.get(name).unwrap();
            let s = streaming.tensors.get(name).unwrap();
            assert_eq!(e.shape, s.shape, "{name} shape");
            assert_eq!(e.data, s.data, "{name} bytes after bf16→f16+quantize");
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
