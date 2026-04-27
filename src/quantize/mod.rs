//! Quantization module — transforms TensorMap into QuantizedModel.
//!
//! Implements the `Quantizer` trait with dispatch by QuantMethod.
//! Sub-modules:
//! - `static_quant`: f16, q8, q4, q2 round-to-nearest
//! - `mixed`: Mixed-bit with --sensitive-layers
//! - `dwq`: DWQ calibration via InferenceRunner

pub mod apex;
pub mod dwq;
pub mod k_quant;
pub mod k_quant_codec;
pub mod k_quant_codec_quantizer;
pub mod layer_mix;
pub mod q_legacy;
pub mod variant_quantizer;
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

/// ADR-014 P3 (Decision 5): rayon-parallel variant of [`quantize_streaming`].
///
/// Distributes the per-tensor quantize work across a rayon thread
/// pool. The pipeline shape per ADR-014 Decision 5:
///
/// ```text
/// producer                    : materialise via map.into_iter()
/// worker pool (n threads)     : quantise (CPU-bound)
/// collector                   : accumulate into HashMap
/// ```
///
/// Implemented via `rayon::iter::IntoParallelIterator` on a
/// `Vec<(String, LazyTensor)>` collected from the lazy map. rayon's
/// thread pool handles work distribution and scheduling; bounded
/// in-flight work is enforced by the pool size (`min(available
/// parallelism, 16)` matches the ADR's `min(num_cpus, 16)` cap —
/// Apple Silicon performance cores cap out at 12, oversubscription
/// degrades).
///
/// **Memory-budget delta vs. serial [`quantize_streaming`]** (apex MoE):
///
/// - Serial: 1 tensor materialised at a time → ~750 MB peak input
/// - Parallel (n=8 workers): ~8 tensors materialised concurrently
///   → ~6 GB peak input (still bounded; well under the 30 GB+ apex
///   resident envelope from Decision 6).
/// - Parallel (n=16): ~12 GB peak input (cap; per ADR's "memory
///   bandwidth saturates beyond 12 cores on M5 Max").
///
/// **Byte-identical to serial**: the order of `quantizer.quantize_tensor()`
/// invocations differs from serial, but each invocation is deterministic
/// per `(tensor_bytes, config)`. The output `HashMap<name, QuantizedTensor>`
/// has the same set of `(name, quantized_bytes)` pairs as serial, so
/// `cargo test --bin hf2q quantize_streaming_parallel_byte_identical_to_serial`
/// pins the contract.
///
/// **Cancellation**: rayon's parallel iter doesn't have explicit
/// cancellation; if a worker errors, subsequent workers continue
/// (rayon collects all errors then `?` short-circuits). For SIGINT
/// cancellation at this layer, the caller should check
/// `INTERRUPTED.load()` after the parallel iter returns. Mid-batch
/// cancellation needs the StreamingBackend pattern from P2 iter-3
/// where the writer thread can short-circuit on the interrupt flag.
pub fn quantize_streaming_parallel(
    map: crate::ir::lazy::LazyTensorMap,
    metadata: &crate::ir::ModelMetadata,
    quantizer: &(dyn Quantizer + Send + Sync),
    bits: u8,
    group_size: usize,
    progress: &ProgressReporter,
    bf16_to_f16: bool,
    n_workers: Option<usize>,
) -> Result<crate::ir::QuantizedModel, QuantizeError> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use std::collections::HashMap;
    use std::sync::Mutex;

    // Resolve worker count: caller-specified, or system parallelism
    // capped at 16 (Apple Silicon performance cores cap out at 12;
    // oversubscription degrades — ADR-014 Decision 5).
    let n_workers = n_workers
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(8)
        .clamp(1, 16);

    let total = map.len();
    let pb = progress.bar(total as u64, "Quantizing (parallel)");
    let pb_mutex = Mutex::new(pb);
    let bf16_counter = std::sync::atomic::AtomicUsize::new(0);

    // Materialise the iterator into a Vec so rayon can distribute the
    // work. Each entry is (name, LazyTensor); the materialisation of
    // the *bytes* still happens inside each worker's closure body —
    // the LazyTensor is small (meta + boxed closure ptr).
    let entries: Vec<(String, crate::ir::lazy::LazyTensor)> = map.into_iter().collect();

    // Build a custom rayon thread pool sized to `n_workers`; the
    // global pool may be a different size (or shared with other rayon
    // calls). Using a dedicated pool keeps the thread count
    // deterministic.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_workers)
        .build()
        .map_err(|e| QuantizeError::TensorQuantizeFailed {
            tensor: "<pool>".to_string(),
            reason: format!("rayon pool init: {e}"),
        })?;

    let quantized_results: Result<Vec<(String, crate::ir::QuantizedTensor)>, QuantizeError> = pool
        .install(|| {
            entries
                .into_par_iter()
                .map(|(name, lazy)| -> Result<(String, crate::ir::QuantizedTensor), QuantizeError> {
                    let mut tensor = lazy.materialize().map_err(|e| {
                        QuantizeError::TensorQuantizeFailed {
                            tensor: name.clone(),
                            reason: format!("materialize: {e}"),
                        }
                    })?;

                    if bf16_to_f16 && tensor.dtype == crate::ir::DType::BF16 {
                        tensor = tensor.to_f16().map_err(|e| {
                            QuantizeError::TensorQuantizeFailed {
                                tensor: name.clone(),
                                reason: format!("bf16→f16: {e}"),
                            }
                        })?;
                        bf16_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

                    // Per-worker progress increment under mutex —
                    // ProgressReporter::ProgressBar is internally
                    // atomic-counter-backed but the API requires
                    // &self. Mutex is fine; contention is bounded by
                    // worker count (8-16).
                    if let Ok(pb) = pb_mutex.lock() {
                        pb.inc(1);
                    }

                    Ok((name, quantized))
                })
                .collect()
        });

    let quantized_tensors: HashMap<String, crate::ir::QuantizedTensor> = quantized_results?
        .into_iter()
        .collect();

    if let Ok(pb) = pb_mutex.into_inner() {
        let bf16 = bf16_counter.load(std::sync::atomic::Ordering::Relaxed);
        pb.finish_with_message(format!(
            "Quantized {} tensors (parallel n={}){}",
            quantized_tensors.len(),
            n_workers,
            if bf16 > 0 {
                format!(", {} bf16→f16", bf16)
            } else {
                String::new()
            }
        ));
    }

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

    /// ADR-014 P3 byte-identity regression:
    /// `quantize_streaming_parallel(LazyTensorMap, ...)` produces a
    /// `QuantizedModel` byte-equal to `quantize_streaming(LazyTensorMap, ...)`
    /// (and therefore byte-equal to `quantize_model(&TensorMap, ...)`)
    /// on the same fixture. Verifies that rayon work distribution
    /// doesn't introduce non-determinism in per-tensor quantization.
    #[test]
    fn quantize_streaming_parallel_byte_identical_to_serial() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};

        // Fresh LazyTensorMaps for each call (consumed).
        let make_lazy_map = || -> LazyTensorMap {
            let make_payload = |seed: u8, len: usize| -> Vec<u8> {
                (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
            };
            let mut m = LazyTensorMap::new();
            for (name, shape, seed) in [
                ("model.layers.0.self_attn.q_proj.weight", vec![64, 64], 0x10u8),
                ("model.layers.0.self_attn.k_proj.weight", vec![64, 32], 0x20u8),
                ("model.layers.0.mlp.gate_proj.weight", vec![64, 64], 0x30u8),
                ("model.layers.0.mlp.up_proj.weight", vec![64, 64], 0x40u8),
                ("model.layers.0.mlp.down_proj.weight", vec![64, 64], 0x50u8),
                ("model.layers.1.self_attn.q_proj.weight", vec![64, 64], 0x60u8),
                ("model.layers.0.input_layernorm.weight", vec![64], 0x70u8),
            ] {
                let len = shape.iter().product::<usize>() * 2;
                m.insert(LazyTensor::from_bytes(
                    LazyMeta::new(name.into(), shape, DType::F16),
                    make_payload(seed, len),
                ));
            }
            m
        };

        let metadata = dummy_metadata();
        let quantizer = StaticQuantizer::new("q4").unwrap();
        let progress = crate::progress::ProgressReporter::new();

        let serial = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &quantizer,
            4,
            32,
            &progress,
            false,
        )
        .unwrap();

        for n_workers in [1, 2, 4, 8] {
            let parallel = quantize_streaming_parallel(
                make_lazy_map(),
                &metadata,
                &quantizer,
                4,
                32,
                &progress,
                false,
                Some(n_workers),
            )
            .unwrap();

            assert_eq!(
                serial.tensors.len(),
                parallel.tensors.len(),
                "n_workers={}: tensor count",
                n_workers
            );
            for (name, s) in &serial.tensors {
                let p = parallel.tensors.get(name).unwrap_or_else(|| {
                    panic!("n_workers={}: missing tensor {}", n_workers, name)
                });
                assert_eq!(s.shape, p.shape, "n_workers={} {name} shape", n_workers);
                assert_eq!(s.original_dtype, p.original_dtype, "n_workers={} {name} dtype", n_workers);
                assert_eq!(s.data, p.data, "n_workers={} {name} bytes", n_workers);
                assert_eq!(s.quant_info.method, p.quant_info.method);
                assert_eq!(s.quant_info.bits, p.quant_info.bits);
                assert_eq!(s.quant_info.preserved, p.quant_info.preserved);
            }
            assert_eq!(serial.quant_method, parallel.quant_method);
            assert_eq!(serial.group_size, parallel.group_size);
            assert_eq!(serial.bits, parallel.bits);
        }
    }

    /// Worker count is clamped to [1, 16] regardless of input.
    /// `n_workers: Some(0)` → 1; `n_workers: Some(100)` → 16.
    #[test]
    fn quantize_streaming_parallel_worker_clamp() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};

        let make_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            for i in 0..3 {
                let name = format!("model.layers.{}.weight", i);
                m.insert(LazyTensor::from_bytes(
                    LazyMeta::new(name, vec![32, 32], DType::F16),
                    vec![0u8; 32 * 32 * 2],
                ));
            }
            m
        };
        let metadata = dummy_metadata();
        let quantizer = StaticQuantizer::new("q4").unwrap();
        let progress = crate::progress::ProgressReporter::new();

        // n_workers=0 → clamp to 1, succeeds.
        let r = quantize_streaming_parallel(
            make_map(),
            &metadata,
            &quantizer,
            4,
            32,
            &progress,
            false,
            Some(0),
        );
        assert!(r.is_ok(), "n_workers=0 must clamp to 1 and succeed");

        // n_workers=100 → clamp to 16, succeeds.
        let r = quantize_streaming_parallel(
            make_map(),
            &metadata,
            &quantizer,
            4,
            32,
            &progress,
            false,
            Some(100),
        );
        assert!(r.is_ok(), "n_workers=100 must clamp to 16 and succeed");
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

    /// ADR-014 P7 iter-3t — `VariantKQuantizer` end-to-end through
    /// `quantize_streaming(LazyTensorMap, ...)` on canonical GGUF tensor
    /// names: confirms the per-tensor target dispatch (`layer_mix`
    /// policy → `k_quant_codec` block format) flows through the streaming
    /// loop intact, and produces a `QuantizedModel` byte-identical to the
    /// eager `quantize_model(&TensorMap, ...)` path on the same input.
    ///
    /// This is the production-shape test that would catch any future
    /// regression coupling P0 (LazyTensorMap), P2 (streaming loop), P3
    /// (BTreeMap iteration determinism), and P7 (variant Quantizer trait
    /// impl). Existing `variant_end_to_end_pipeline` covered the
    /// `quantize_tensor`-direct surface — this covers the streaming
    /// surface that production cmd_convert will call.
    #[test]
    fn variant_streaming_byte_identical_to_eager() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;
        use crate::calibrate::calibrator::CalibrationData;

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;

        // F16 fixtures using a deterministic ramp per tensor (varying
        // seed offset so quantized bytes differ across tensors — catches
        // any swap or cache bug).
        let make_f16_payload = |seed: f32, len: usize| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(len * 2);
            for i in 0..len {
                let v = (i as f32 / len as f32) * 2.0 - 1.0 + seed * 0.01;
                let h = half::f16::from_f32(v);
                bytes.extend_from_slice(&h.to_le_bytes());
            }
            bytes
        };

        // Four canonical tensor names spanning all four `Q4_K_M` policy
        // branches in `layer_mix::target_for`:
        //   1. output.weight       → Q6_K bump (M variant)
        //   2. blk.0.attn_v.weight → Q6_K (use_more_bits at layer 0)
        //   3. blk.5.attn_q.weight → Q4_K (attn_q never bumps)
        //   4. blk.10.ffn_down.weight → Q4_K (use_more_bits=false at
        //      layer 10 of 32: 10 ≥ n/8=4, 10 < 7n/8=28, (10-4)%3=0)
        let fixtures: Vec<(&str, Vec<usize>, &str, usize)> = vec![
            ("output.weight",            vec![1, QK_K], "Q6_K", 210),
            ("blk.0.attn_v.weight",      vec![1, QK_K], "Q6_K", 210),
            ("blk.5.attn_q.weight",      vec![1, QK_K], "Q4_K", 144),
            ("blk.10.ffn_down.weight",   vec![1, QK_K], "Q4_K", 144),
        ];

        let mut tensor_map = TensorMap::new();
        let mut lazy_map = LazyTensorMap::new();

        for (i, (name, shape, _expected_type, _expected_bytes)) in fixtures.iter().enumerate() {
            let len: usize = shape.iter().product();
            let data = make_f16_payload(i as f32, len);
            tensor_map.insert(TensorRef {
                name: name.to_string(),
                shape: shape.clone(),
                dtype: DType::F16,
                data: data.clone(),
            });
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), shape.clone(), DType::F16),
                data,
            ));
        }

        let metadata = dummy_metadata();
        let quantizer = VariantKQuantizer::new(
            KQuantVariant::Q4_K_M,
            CalibrationData::None,
            N_LAYERS,
        );
        let progress = crate::progress::ProgressReporter::new();

        // Eager and streaming paths against same input.
        let eager =
            quantize_model(&tensor_map, &metadata, &quantizer, 0, 0, &progress).unwrap();
        let streaming = quantize_streaming(
            lazy_map, &metadata, &quantizer, 0, 0, &progress, /* bf16_to_f16 */ false,
        )
        .unwrap();

        // Per-tensor structural + policy + byte-identity checks.
        assert_eq!(eager.tensors.len(), 4);
        assert_eq!(streaming.tensors.len(), 4);
        for (name, _shape, expected_type, expected_bytes) in &fixtures {
            let e = eager.tensors.get(*name).expect("eager tensor missing");
            let s = streaming.tensors.get(*name).expect("streaming tensor missing");

            // Policy branch: per-tensor target picked correctly.
            assert_eq!(
                e.quant_info.ggml_type.as_deref(),
                Some(*expected_type),
                "eager {name} ggml_type mismatch"
            );
            assert_eq!(
                s.quant_info.ggml_type.as_deref(),
                Some(*expected_type),
                "streaming {name} ggml_type mismatch"
            );

            // Codec path: bytes-per-block emitted by the codec match the
            // target's canonical block size.
            assert_eq!(
                e.data.len(), *expected_bytes,
                "eager {name} block size mismatch"
            );
            assert_eq!(
                s.data.len(), *expected_bytes,
                "streaming {name} block size mismatch"
            );

            // Byte-identity: streaming and eager produce literally
            // identical output bytes for the same Quantizer + input.
            assert_eq!(e.data, s.data, "streaming/eager byte mismatch on {name}");
            assert_eq!(e.shape, s.shape, "{name} shape diverged");
            assert_eq!(e.original_dtype, s.original_dtype, "{name} dtype diverged");
            assert_eq!(
                e.quant_info.method, s.quant_info.method,
                "{name} method diverged"
            );
            assert_eq!(
                e.quant_info.method,
                crate::quantize::k_quant_codec_quantizer::METHOD_K_QUANT_CODEC_DIRECT,
                "{name} method should be k_quant_codec_direct (production codec path)"
            );
            assert_eq!(
                e.quant_info.preserved, false,
                "{name} should be quantized, not preserved"
            );
        }

        // Top-level QuantizedModel metadata: name() reports the variant.
        assert_eq!(eager.quant_method, "Q4_K_M");
        assert_eq!(streaming.quant_method, "Q4_K_M");
    }

    /// ADR-014 P7 iter-4 — `VariantKQuantizer` parallel-streaming
    /// byte-identity gate. Extends iter-3t's serial-vs-eager check to
    /// cover the rayon path: `quantize_streaming_parallel` + variant
    /// dispatch must produce a `QuantizedModel` byte-equal to
    /// `quantize_streaming` (serial) + variant dispatch on the same
    /// fixture.
    ///
    /// Why: the existing `quantize_streaming_parallel_byte_identical_to_serial`
    /// test only exercises `StaticQuantizer`.  A future regression
    /// where the variant's per-tensor target dispatch (parsing
    /// `blk.<N>` from tensor names + calling `layer_mix::target_for`)
    /// developed thread-unsafe state would silently corrupt parallel
    /// output without affecting serial output.  This gate locks the
    /// invariant.
    ///
    /// Also adds a compile-time `Send + Sync` check on `VariantKQuantizer`
    /// because `quantize_streaming_parallel` requires `&(dyn Quantizer
    /// + Send + Sync)`.  If a future field addition (e.g. interior
    /// mutability via `Cell<…>`) breaks this bound, the test fails to
    /// compile rather than runs and produces garbage.
    #[test]
    fn variant_streaming_parallel_byte_identical_to_serial() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;
        use crate::calibrate::calibrator::CalibrationData;

        // compile-time Send + Sync guard: this fn won't compile if
        // VariantKQuantizer or CalibrationData lose Send+Sync.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VariantKQuantizer>();

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;

        let make_f16_payload = |seed: f32, len: usize| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(len * 2);
            for i in 0..len {
                let v = (i as f32 / len as f32) * 2.0 - 1.0 + seed * 0.01;
                let h = half::f16::from_f32(v);
                bytes.extend_from_slice(&h.to_le_bytes());
            }
            bytes
        };

        // 6 tensors covering 4 distinct policy branches, multiple
        // layers, and a passthrough — exercises rayon work
        // distribution across tensors with heterogeneous targets and
        // confirms parallel dispatch handles each branch correctly.
        let fixtures: Vec<(&str, Vec<usize>)> = vec![
            ("output.weight",                vec![1, QK_K]),     // Q6_K bump
            ("blk.0.attn_v.weight",          vec![1, QK_K]),     // Q6_K bump
            ("blk.5.attn_q.weight",          vec![1, QK_K]),     // Q4_K base
            ("blk.10.ffn_down.weight",       vec![1, QK_K]),     // Q4_K base
            ("blk.20.ffn_down_exps.weight",  vec![1, QK_K]),     // Q4_K base (MoE)
            ("blk.5.attn_norm.weight",       vec![16]),          // 1-D < 32 → preserve
        ];

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            for (i, (name, shape)) in fixtures.iter().enumerate() {
                let len: usize = shape.iter().product();
                let data = make_f16_payload(i as f32, len);
                m.insert(LazyTensor::from_bytes(
                    LazyMeta::new(name.to_string(), shape.clone(), DType::F16),
                    data,
                ));
            }
            m
        };

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = VariantKQuantizer::new(
            KQuantVariant::Q4_K_M,
            CalibrationData::None,
            N_LAYERS,
        );

        let serial = quantize_streaming(
            make_lazy_map(), &metadata, &q, 0, 0, &progress, false,
        )
        .unwrap();

        // Force >1 worker so rayon fan-out fires.  None defaults to
        // num_cpus; passing Some(4) clamps for determinism.
        let parallel = quantize_streaming_parallel(
            make_lazy_map(), &metadata, &q, 0, 0, &progress, false, Some(4),
        )
        .unwrap();

        assert_eq!(serial.tensors.len(), parallel.tensors.len(),
            "tensor count diverged");
        assert_eq!(serial.quant_method, parallel.quant_method);
        assert_eq!(serial.bits, parallel.bits);
        assert_eq!(serial.group_size, parallel.group_size);

        for (name, _shape) in &fixtures {
            let s = serial.tensors.get(*name)
                .unwrap_or_else(|| panic!("serial missing {name}"));
            let p = parallel.tensors.get(*name)
                .unwrap_or_else(|| panic!("parallel missing {name}"));
            assert_eq!(s.shape, p.shape, "{name} shape");
            assert_eq!(s.original_dtype, p.original_dtype, "{name} dtype");
            assert_eq!(s.quant_info.method, p.quant_info.method,
                "{name} method");
            assert_eq!(
                s.quant_info.ggml_type.as_deref(),
                p.quant_info.ggml_type.as_deref(),
                "{name} ggml_type"
            );
            assert_eq!(s.quant_info.preserved, p.quant_info.preserved,
                "{name} preserved");
            // The actual byte-identity check.
            assert_eq!(s.data, p.data,
                "{name}: parallel bytes diverged from serial — \
                 rayon work distribution is non-deterministic for \
                 the variant Quantizer or VariantKQuantizer is not \
                 thread-safe");
        }
    }

    /// ADR-014 P7 iter-3z — proves imatrix actually **improves** the
    /// importance-weighted error vs uncalibrated, not just produces
    /// different bytes (iter-3w).  Closes the divergence-direction
    /// gap: a regression that flips the imatrix codebook search to
    /// minimise the **wrong** objective (or accidentally inverts the
    /// importance vector) would still pass iter-3w's "bytes differ"
    /// gate but would fail this gate.
    ///
    /// Mechanism: `make_qkx3_quants` minimises `sum(weight[c] * err[c]^2)`
    /// where `weight[c]` is the imatrix importance for column `c`.
    /// `make_qkx2_quants` (the uncalibrated path) minimises uniform
    /// `sum(err[c]^2)`.  When the importance vector is highly non-uniform
    /// (here: 10000× ratio on first 16 columns), the imatrix path's
    /// importance-weighted SSE must be **lower** than the uncalibrated
    /// path's, even though its uniform-RMSE on the full row may be
    /// higher (by design — imatrix trades low-importance accuracy for
    /// high-importance accuracy).
    #[test]
    fn variant_imatrix_lowers_importance_weighted_error() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::quantize::k_quant::dequantize_row_q4_k_bytes;

        const QK_K: usize = 256;
        const N_BLOCKS: usize = 4;
        const TOTAL: usize = N_BLOCKS * QK_K;
        const TENSOR: &str = "blk.5.attn_q.weight";
        const HIGH_IMPORTANCE_COLS: usize = 16;

        // Activations → after `accumulate_dense`, ImatrixCollector stores
        // `values[c] = sum_t activations[c]^2`.  With 1 token and
        // activations[c] = 10.0 for c<16 / 0.1 for c≥16, importance
        // becomes 100 vs 0.01 → 10000× ratio.
        let acts: Vec<f32> = (0..QK_K)
            .map(|c| if c < HIGH_IMPORTANCE_COLS { 10.0 } else { 0.1 })
            .collect();

        let mut col = ImatrixCollector::new();
        col.accumulate_dense(TENSOR, &acts, 1, QK_K).unwrap();
        col.record_chunk();
        let imatrix = CalibrationData::from_imatrix_collector(&col);

        // Per-column importance weight for the SSE computation: square
        // of the activation magnitude (matches what the codec sees).
        let importance: Vec<f64> = (0..QK_K)
            .map(|c| if c < HIGH_IMPORTANCE_COLS { 100.0_f64 } else { 0.01_f64 })
            .collect();

        // Adversarial-ish input that exposes codebook tradeoffs:
        // high-importance columns have wide magnitude range (forces
        // the codebook to spend bits on them) while low-importance
        // columns have small steady values (cheap to skimp on).  On a
        // smooth ramp Q4_K_M is already near-optimal in both codebooks
        // and the imatrix improvement washes out in the noise.
        let original_f32: Vec<f32> = (0..TOTAL)
            .map(|i| {
                let c = i % QK_K;
                let r = (i / QK_K) as f32;
                if c < HIGH_IMPORTANCE_COLS {
                    // wide range: -3 .. 3 across rows × column index
                    let t = (c as f32 / HIGH_IMPORTANCE_COLS as f32) + r * 0.5;
                    -3.0 + 6.0 * (t.fract())
                } else {
                    // small steady values: ~ -0.1 .. 0.1
                    let t = ((c as f32) / QK_K as f32) + r * 0.01;
                    -0.1 + 0.2 * (t.fract())
                }
            })
            .collect();
        let f16_bytes: Vec<u8> = original_f32
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let expected_post_f16: Vec<f32> = original_f32
            .iter()
            .map(|v| half::f16::from_f32(*v).to_f32())
            .collect();

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            m.insert(LazyTensor::from_bytes(
                LazyMeta::new(TENSOR.to_string(), vec![N_BLOCKS, QK_K], DType::F16),
                f16_bytes.clone(),
            ));
            m
        };

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        let q_none = VariantKQuantizer::new(
            KQuantVariant::Q4_K_M, CalibrationData::None, 32,
        );
        let q_imatrix = VariantKQuantizer::new(
            KQuantVariant::Q4_K_M, imatrix, 32,
        );

        let out_none = quantize_streaming(
            make_lazy_map(), &metadata, &q_none, 0, 0, &progress, false,
        ).unwrap();
        let out_imatrix = quantize_streaming(
            make_lazy_map(), &metadata, &q_imatrix, 0, 0, &progress, false,
        ).unwrap();

        let bytes_none = &out_none.tensors[TENSOR].data;
        let bytes_imatrix = &out_imatrix.tensors[TENSOR].data;

        let mut decoded_none = vec![0.0_f32; TOTAL];
        let mut decoded_imatrix = vec![0.0_f32; TOTAL];
        dequantize_row_q4_k_bytes(bytes_none, &mut decoded_none).unwrap();
        dequantize_row_q4_k_bytes(bytes_imatrix, &mut decoded_imatrix).unwrap();

        // Importance-weighted SSE: sum over (row, col) of weight[col] * err^2.
        let weighted_sse = |decoded: &[f32]| -> f64 {
            let mut s = 0.0_f64;
            for r in 0..N_BLOCKS {
                for c in 0..QK_K {
                    let i = r * QK_K + c;
                    let e = (decoded[i] as f64) - (expected_post_f16[i] as f64);
                    s += importance[c] * e * e;
                }
            }
            s
        };

        let wsse_none = weighted_sse(&decoded_none);
        let wsse_imatrix = weighted_sse(&decoded_imatrix);

        // The gate: imatrix path's importance-weighted SSE must be
        // strictly lower.  If they're equal, calibration is being
        // ignored.  If imatrix is higher, the codec is minimising the
        // wrong objective (sign-flipped weights, etc).
        assert!(
            wsse_imatrix < wsse_none,
            "imatrix-weighted SSE {wsse_imatrix:.6} should be < uncalibrated \
             SSE {wsse_none:.6} — imatrix path is failing to minimise the \
             importance-weighted objective"
        );

        // Sanity: the high-importance columns specifically should see
        // a meaningful improvement (>= 5% reduction in SSE on those
        // columns alone).  Catches a bug where the gate technically
        // passes via tiny noise improvements unrelated to the
        // importance vector.
        let high_col_sse = |decoded: &[f32]| -> f64 {
            let mut s = 0.0_f64;
            for r in 0..N_BLOCKS {
                for c in 0..HIGH_IMPORTANCE_COLS {
                    let i = r * QK_K + c;
                    let e = (decoded[i] as f64) - (expected_post_f16[i] as f64);
                    s += e * e;
                }
            }
            s
        };
        let high_none = high_col_sse(&decoded_none);
        let high_imatrix = high_col_sse(&decoded_imatrix);
        assert!(
            high_imatrix < high_none * 0.95,
            "high-importance-column SSE: imatrix={high_imatrix:.6} \
             uncalibrated={high_none:.6}; imatrix should reduce error \
             on these columns by ≥ 5% (got {:.2}%)",
            (1.0 - high_imatrix / high_none) * 100.0
        );
    }

    /// ADR-014 P7 iter-3y — round-trip RMSE bounds for Q5_K_M and Q6_K
    /// through the streaming pipeline, paralleling iter-3x's Q4_K_M
    /// gate. Confirms the variant dispatch correctly routes to the
    /// Q5_K and Q6_K codecs (not just Q4_K) and the resulting bytes
    /// dequantize to numerically meaningful values within the
    /// canonical bounds:
    ///
    /// - Q5_K: ≤ 0.025 RMSE (5.5-bpw codebook → ~half Q4_K error)
    /// - Q6_K: ≤ 0.012 RMSE (6.5-bpw codebook → ~quarter Q4_K error)
    ///
    /// Uses the same `blk.5.attn_q.weight` layer-5/32 setup as iter-3x
    /// so attn_q never bumps and each variant lands at its base
    /// target — Q5_K_M → Q5_K, Q6_K → Q6_K.  This means a regression
    /// in the policy that silently routes Q5_K_M to Q4_K (or any other
    /// target) would round-trip with wrong bytes-per-block AND wrong
    /// RMSE.  The two assertions catch the bug at different layers.
    #[test]
    fn variant_streaming_q5km_q6k_round_trip_rmse_bounds() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;
        use crate::calibrate::calibrator::CalibrationData;
        use crate::quantize::k_quant::{
            dequantize_row_q5_k_bytes, dequantize_row_q6_k_bytes,
        };

        const QK_K: usize = 256;
        const N_BLOCKS: usize = 4;
        const TOTAL: usize = N_BLOCKS * QK_K;
        const TENSOR: &str = "blk.5.attn_q.weight";

        let original_f32: Vec<f32> = (0..TOTAL)
            .map(|i| (i as f32 / TOTAL as f32) * 2.0 - 1.0)
            .collect();

        let f16_bytes: Vec<u8> = original_f32
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();

        let expected_post_f16: Vec<f32> = original_f32
            .iter()
            .map(|v| half::f16::from_f32(*v).to_f32())
            .collect();

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        // Per-variant: (variant, expected_target, expected_block_size, rmse_bound, dequant_fn)
        type DequantFn = fn(&[u8], &mut [f32])
            -> Result<usize, crate::quantize::k_quant::KQuantError>;
        let cases: Vec<(KQuantVariant, &str, usize, f64, DequantFn)> = vec![
            (
                KQuantVariant::Q5_K_M, "Q5_K", N_BLOCKS * 176, 0.025,
                dequantize_row_q5_k_bytes as DequantFn,
            ),
            (
                KQuantVariant::Q6_K, "Q6_K", N_BLOCKS * 210, 0.012,
                dequantize_row_q6_k_bytes as DequantFn,
            ),
        ];

        for (variant, expected_type, expected_bytes, rmse_bound, dequant) in cases {
            let mut lazy_map = LazyTensorMap::new();
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(TENSOR.to_string(), vec![N_BLOCKS, QK_K], DType::F16),
                f16_bytes.clone(),
            ));

            let q = VariantKQuantizer::new(variant, CalibrationData::None, 32);
            let out = quantize_streaming(
                lazy_map, &metadata, &q, 0, 0, &progress, /* bf16_to_f16 */ false,
            )
            .unwrap_or_else(|e| panic!("streaming failed for {variant}: {e}"));

            let t = out.tensors.get(TENSOR)
                .unwrap_or_else(|| panic!("{variant}: missing tensor"));

            // structural: variant routed to the expected base target
            assert_eq!(
                t.data.len(), expected_bytes,
                "{variant}: bytes ({}) != expected ({expected_bytes})",
                t.data.len()
            );
            assert_eq!(
                t.quant_info.ggml_type.as_deref(), Some(expected_type),
                "{variant}: ggml_type mismatch"
            );

            // dequantize via the matching codec path
            let mut decoded = vec![0.0_f32; TOTAL];
            let n_decoded = dequant(&t.data, &mut decoded)
                .unwrap_or_else(|e| panic!("{variant}: dequant failed: {e:?}"));
            assert_eq!(n_decoded, TOTAL,
                "{variant}: dequant should populate every element");

            // RMSE bound vs F16-cast original (apples-to-apples — codec
            // saw the F16-cast values).
            let sse: f64 = decoded
                .iter()
                .zip(expected_post_f16.iter())
                .map(|(d, e)| {
                    let diff = (*d as f64) - (*e as f64);
                    diff * diff
                })
                .sum();
            let rmse = (sse / TOTAL as f64).sqrt();

            assert!(
                rmse <= rmse_bound,
                "{variant}: round-trip RMSE {rmse:.6} exceeds {rmse_bound} bound — \
                 quantize→dequantize chain through VariantKQuantizer is producing \
                 numerically wrong {expected_type} blocks"
            );

            // sanity: not all zeros
            let nonzero = decoded.iter().filter(|v| v.abs() > 1e-6).count();
            assert!(
                nonzero > TOTAL / 2,
                "{variant}: decoded mostly zero — likely target/tag mismatch"
            );
        }
    }

    /// ADR-014 P7 iter-3x — round-trip dequant RMSE bound on
    /// `VariantKQuantizer` output, exercised through `quantize_streaming`.
    /// Closes the quantize→dequantize loop end-to-end via the variant
    /// dispatch (rather than `quantize_row_q4_k`/`dequantize_row_q4_k`
    /// in isolation, which `k_quant.rs::tests` already covers).
    ///
    /// Why this matters: a regression in `quantize_tensor_2d_to_bytes`
    /// (the multi-row helper) or `target_to_ggml_name` (which sets the
    /// downstream dequant dispatch tag) would produce bytes that pass
    /// every existing structural test (right size, right tag) but
    /// would round-trip to garbage.  Locking RMSE proves the bytes are
    /// numerically meaningful Q4_K blocks, not just well-shaped opaque
    /// data.
    ///
    /// Bound matches the `quantize_row_q4_k_round_trip_smooth_ramp_rmse`
    /// gate in `k_quant.rs::tests` (≤ 0.05 on a smooth [-1, 1] ramp).
    /// We use the same input pattern so any divergence between the
    /// streaming path's RMSE and the isolated codec path's RMSE
    /// indicates a layering bug.
    #[test]
    fn variant_streaming_q4km_round_trip_rmse_bound() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;
        use crate::calibrate::calibrator::CalibrationData;
        use crate::quantize::k_quant::dequantize_row_q4_k_bytes;

        // Use multiple super-blocks (4 × QK_K = 1024 elements) so RMSE
        // averages over enough samples to be statistically meaningful
        // and not dominated by edge artefacts.
        const QK_K: usize = 256;
        const N_BLOCKS: usize = 4;
        const TOTAL: usize = N_BLOCKS * QK_K;
        const TENSOR: &str = "blk.5.attn_q.weight"; // attn_q never bumps; stays at Q4_K base

        let original_f32: Vec<f32> = (0..TOTAL)
            .map(|i| (i as f32 / TOTAL as f32) * 2.0 - 1.0)
            .collect();

        // F16 input (matches mainline streaming path: cmd_convert
        // converts BF16→F16 before the streaming quantize loop).
        let f16_bytes: Vec<u8> = original_f32
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        assert_eq!(f16_bytes.len(), TOTAL * 2);

        let mut lazy_map = LazyTensorMap::new();
        lazy_map.insert(LazyTensor::from_bytes(
            LazyMeta::new(TENSOR.to_string(), vec![N_BLOCKS, QK_K], DType::F16),
            f16_bytes,
        ));

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = VariantKQuantizer::new(
            KQuantVariant::Q4_K_M,
            CalibrationData::None,
            32,
        );

        let out = quantize_streaming(
            lazy_map, &metadata, &q, 0, 0, &progress, /* bf16_to_f16 */ false,
        )
        .unwrap();

        let t = out.tensors.get(TENSOR).expect("missing tensor");
        // structural sanity: 4 super-blocks × 144 bytes = 576 bytes Q4_K
        assert_eq!(t.data.len(), N_BLOCKS * 144,
            "Q4_K bytes count mismatch: {} blocks × 144", N_BLOCKS);
        assert_eq!(t.quant_info.ggml_type.as_deref(), Some("Q4_K"));

        // Dequantize back to F32 via the same flat-bytes path the GGUF
        // backend (and downstream consumers) will use.
        let mut decoded = vec![0.0_f32; TOTAL];
        let n_decoded = dequantize_row_q4_k_bytes(&t.data, &mut decoded)
            .expect("dequantize_row_q4_k_bytes failed");
        assert_eq!(n_decoded, TOTAL,
            "dequant should populate every element");

        // The F16 cast at input introduces ~6e-4 RMSE at most — well
        // below the Q4_K bound — so we compare decoded vs the F16-cast
        // value for an apples-to-apples comparison (matching what the
        // codec actually saw).
        let expected_post_f16: Vec<f32> = original_f32
            .iter()
            .map(|v| half::f16::from_f32(*v).to_f32())
            .collect();

        let sse: f64 = decoded
            .iter()
            .zip(expected_post_f16.iter())
            .map(|(d, e)| {
                let diff = (*d as f64) - (*e as f64);
                diff * diff
            })
            .sum();
        let rmse = (sse / TOTAL as f64).sqrt();

        // Bound ≤ 0.05 matches `k_quant.rs::tests` direct-codec gate.
        assert!(
            rmse <= 0.05,
            "Q4_K round-trip RMSE {rmse:.6} exceeds 0.05 bound — \
             quantize→dequantize chain through VariantKQuantizer is \
             producing numerically wrong blocks (was a structural-only \
             test pre-iter-3x)"
        );

        // Sanity: decoded values are not all-zero (catches a bytes-tag
        // mismatch where dequant treats the buffer as zeros).
        let nonzero_count = decoded.iter().filter(|v| v.abs() > 1e-6).count();
        assert!(nonzero_count > TOTAL / 2,
            "decoded output is mostly zero — likely a target/tag mismatch");
    }

    /// ADR-014 P7 iter-3w — proves the imatrix calibration path is
    /// genuinely end-to-end functional through `VariantKQuantizer` +
    /// `quantize_streaming`: imatrix-weighted quantization produces a
    /// **different** byte sequence than uncalibrated (`CalibrationData::None`)
    /// on the same F32 input, while keeping the same block size.
    ///
    /// Why this matters: iter-3v's variant-menu smoke used
    /// `CalibrationData::None` only.  Without an explicit divergence
    /// gate, a future regression could silently route the imatrix
    /// path back through the uncalibrated codec (e.g. by passing the
    /// wrong calibration value down the call stack) and every existing
    /// test would still pass.  Locking the divergence makes that
    /// failure mode visible.
    ///
    /// Mechanism: the imatrix-weighted `make_qkx3_quants` codebook
    /// search at `ggml-quants.c:902` minimises the **importance-weighted**
    /// L2 error, while `make_qkx2_quants` minimises uniform L2.  When
    /// the importance vector is non-uniform (here: 100× boost on the
    /// first 16 columns, 0.01× elsewhere), the two searches converge
    /// to different codebooks → different quantized bytes.
    #[test]
    fn variant_imatrix_diverges_from_none_through_streaming() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;
        // attn_q at middle layer → never bumps for Q4_K_M, so the codec
        // hits the Q4_K target on both paths and bytes are directly
        // comparable.
        const TENSOR: &str = "blk.5.attn_q.weight";

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        // F32 → F16 input row with realistic spread.
        let make_f16_payload = |len: usize| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(len * 2);
            for i in 0..len {
                let v = (i as f32 / len as f32) * 2.0 - 1.0;
                let h = half::f16::from_f32(v);
                bytes.extend_from_slice(&h.to_le_bytes());
            }
            bytes
        };

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            m.insert(LazyTensor::from_bytes(
                LazyMeta::new(TENSOR.to_string(), vec![1, QK_K], DType::F16),
                make_f16_payload(QK_K),
            ));
            m
        };

        // Build calibration: 100× boost first 16 columns, 0.01× rest.
        let mut col = ImatrixCollector::new();
        let acts: Vec<f32> = (0..QK_K)
            .map(|i| if i < 16 { 100.0 } else { 0.01 })
            .collect();
        col.accumulate_dense(TENSOR, &acts, 1, QK_K).unwrap();
        col.record_chunk();
        let imatrix = CalibrationData::from_imatrix_collector(&col);
        // sanity: the calibration carries entries
        assert!(imatrix.is_some(), "imatrix should be populated");

        // Run the SAME input through BOTH paths.
        let q_none = VariantKQuantizer::new(
            KQuantVariant::Q4_K_M,
            CalibrationData::None,
            N_LAYERS,
        );
        let q_imatrix = VariantKQuantizer::new(
            KQuantVariant::Q4_K_M,
            imatrix,
            N_LAYERS,
        );

        let out_none = quantize_streaming(
            make_lazy_map(), &metadata, &q_none, 0, 0, &progress, false,
        )
        .unwrap();
        let out_imatrix = quantize_streaming(
            make_lazy_map(), &metadata, &q_imatrix, 0, 0, &progress, false,
        )
        .unwrap();

        let t_none = out_none.tensors.get(TENSOR).expect("none-path missing");
        let t_imatrix = out_imatrix.tensors.get(TENSOR).expect("imatrix-path missing");

        // Both produce Q4_K blocks (same target, same size).
        assert_eq!(t_none.data.len(), 144,
            "none path should emit one Q4_K block");
        assert_eq!(t_imatrix.data.len(), 144,
            "imatrix path should emit one Q4_K block");
        assert_eq!(
            t_none.quant_info.ggml_type.as_deref(),
            t_imatrix.quant_info.ggml_type.as_deref(),
            "ggml_type tag should match"
        );

        // The divergence gate: bytes MUST differ.  If they're equal,
        // the imatrix calibration is silently being ignored.
        assert_ne!(
            t_none.data, t_imatrix.data,
            "imatrix-weighted Q4_K bytes must differ from uncalibrated \
             Q4_K bytes when the importance vector is non-uniform — \
             otherwise the calibration plumbing is broken"
        );
    }

    /// ADR-014 P7 iter-3v — every `KQuantVariant` in `all()` produces
    /// non-empty quantized output through the streaming pipeline with
    /// the correct ggml_type tag and block-size byte count for the
    /// variant's base target.  Future-proofs P8 CLI registration: if
    /// a new variant is added to `KQuantVariant::all()`, this smoke
    /// test exercises it without code changes here.
    #[test]
    fn variant_menu_smoke_through_streaming() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;
        use crate::calibrate::calibrator::CalibrationData;

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;

        // single Q4_K-eligible tensor at a non-bumping layer (5 of 32)
        // so the base target — not an upgrade — fires.
        let make_f16_payload = |seed: f32, len: usize| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(len * 2);
            for i in 0..len {
                let v = (i as f32 / len as f32) * 2.0 - 1.0 + seed * 0.01;
                let h = half::f16::from_f32(v);
                bytes.extend_from_slice(&h.to_le_bytes());
            }
            bytes
        };

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        for variant in KQuantVariant::all() {
            // expected block size from the variant's base target
            let (expected_type, expected_bytes): (&str, usize) = match variant.base_target() {
                crate::quantize::k_quant_codec::KQuantTarget::Q4K => ("Q4_K", 144),
                crate::quantize::k_quant_codec::KQuantTarget::Q5K => ("Q5_K", 176),
                crate::quantize::k_quant_codec::KQuantTarget::Q6K => ("Q6_K", 210),
                _ => unreachable!("only K-family base targets supported"),
            };

            let mut lazy_map = LazyTensorMap::new();
            let name = "blk.5.attn_q.weight"; // attn_q never bumps → base
            let data = make_f16_payload(0.0, QK_K);
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                data,
            ));

            let q = VariantKQuantizer::new(*variant, CalibrationData::None, N_LAYERS);
            let out = quantize_streaming(
                lazy_map, &metadata, &q, 0, 0, &progress, /* bf16_to_f16 */ false,
            )
            .unwrap_or_else(|e| panic!("streaming failed for {variant}: {e}"));

            assert_eq!(out.tensors.len(), 1, "{variant}: tensor count");
            let t = out.tensors.get(name).unwrap();
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some(expected_type),
                "{variant}: ggml_type"
            );
            assert_eq!(
                t.data.len(), expected_bytes,
                "{variant}: emitted block size"
            );
            assert_eq!(out.quant_method, variant.name(),
                "{variant}: quant_method tag");
        }
    }

    /// ADR-014 P7 iter-3u — `VariantKQuantizer` honors the
    /// `should_skip_quantization` rule for `ffn_gate_inp.weight` AND
    /// the new MoE-aware classification (`*_exps`, `*_shexp`) through
    /// the streaming pipeline.  Locks two iter-3u behaviors end-to-end:
    ///
    /// 1. `blk.X.ffn_gate_inp.weight` passes through preserved (no
    ///    quantization) regardless of `LayerQuantConfig.preserve` —
    ///    matches `llama-quant.cpp:307`.
    /// 2. `blk.X.ffn_down_exps.weight` gets the Q4_K_M `use_more_bits`
    ///    Q6_K bump at qualifying layers (was a parity gap pre-iter-3u
    ///    where MoE expert tensors fell through to the `Other` category).
    #[test]
    fn variant_streaming_honors_iter3u_moe_classification() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;
        use crate::calibrate::calibrator::CalibrationData;

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;

        let make_f16_payload = |seed: f32, len: usize| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(len * 2);
            for i in 0..len {
                let v = (i as f32 / len as f32) * 2.0 - 1.0 + seed * 0.01;
                let h = half::f16::from_f32(v);
                bytes.extend_from_slice(&h.to_le_bytes());
            }
            bytes
        };

        // Three MoE-style fixtures:
        //   1. ffn_gate_inp.weight   → skip-quantize → passthrough preserved
        //   2. ffn_down_exps.weight  layer 0 → Q6_K (use_more_bits + new
        //                                            substring-classify)
        //   3. ffn_up_exps.weight    layer 0 → Q4_K base (FfnUp is
        //                                            classified now but
        //                                            policy still falls
        //                                            to base — locks
        //                                            the documented
        //                                            iter-3u behavior)
        let fixtures: Vec<(&str, Vec<usize>, &str, Option<&str>)> = vec![
            ("blk.0.ffn_gate_inp.weight",  vec![1, QK_K], "passthrough",         None),
            ("blk.0.ffn_down_exps.weight", vec![1, QK_K], "k_quant_codec_direct", Some("Q6_K")),
            ("blk.0.ffn_up_exps.weight",   vec![1, QK_K], "k_quant_codec_direct", Some("Q4_K")),
        ];

        let mut lazy_map = LazyTensorMap::new();
        for (i, (name, shape, _expected_method, _expected_type)) in fixtures.iter().enumerate() {
            let len: usize = shape.iter().product();
            let data = make_f16_payload(i as f32, len);
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), shape.clone(), DType::F16),
                data,
            ));
        }

        let metadata = dummy_metadata();
        let quantizer = VariantKQuantizer::new(
            KQuantVariant::Q4_K_M,
            CalibrationData::None,
            N_LAYERS,
        );
        let progress = crate::progress::ProgressReporter::new();

        let result = quantize_streaming(
            lazy_map, &metadata, &quantizer, 0, 0, &progress, /* bf16_to_f16 */ false,
        )
        .unwrap();

        for (name, _shape, expected_method, expected_type) in &fixtures {
            let t = result.tensors.get(*name).unwrap_or_else(||
                panic!("streaming output missing {name}"));
            assert_eq!(
                t.quant_info.method, *expected_method,
                "{name} method mismatch (iter-3u)"
            );
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                *expected_type,
                "{name} ggml_type mismatch (iter-3u)"
            );

            if name.contains("ffn_gate_inp") {
                // ffn_gate_inp passes through — original F16 bytes
                // preserved 1:1, NOT block-quantized.
                let len: usize = QK_K;
                assert_eq!(
                    t.data.len(), len * 2,
                    "{name} should preserve F16 bytes (2 per element)"
                );
                assert!(t.quant_info.preserved, "{name} should be marked preserved");
            } else {
                // expert tensors get block-quantized to either Q4_K (144) or
                // Q6_K (210) bytes per super-block.
                let expected_bytes = match expected_type.unwrap() {
                    "Q4_K" => 144,
                    "Q6_K" => 210,
                    _ => unreachable!(),
                };
                assert_eq!(
                    t.data.len(), expected_bytes,
                    "{name} should emit {expected_bytes}-byte blocks for {}",
                    expected_type.unwrap()
                );
                assert!(!t.quant_info.preserved, "{name} should be quantized");
            }
        }
    }
}
