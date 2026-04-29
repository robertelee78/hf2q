//! Quantization module — transforms TensorMap into QuantizedModel.
//!
//! Implements the `Quantizer` trait with dispatch by QuantMethod.
//! Sub-modules:
//! - `static_quant`: f16, q8, q4, q2 round-to-nearest
//! - `mixed`: Mixed-bit with --sensitive-layers
//!
//! ADR-014 P7 iter-8: `dwq`, `dwq_activation`, `sensitivity`, and `apex`
//! migrated to [`crate::calibrate`] (Layout A) — calibrators belong with
//! their orchestration shell, not with the static quantizer hierarchy.

pub mod dwq_k_quantizer;
pub mod k_quant;
pub mod k_quant_codec;
pub mod k_quant_codec_quantizer;
pub mod layer_mix;
pub mod mixed;
pub mod q_legacy;
pub mod static_quant;
pub mod variant_quantizer;

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
/// ADR-014 P7 iter-48 — borrowing wire-up wedge.  Drop-in replacement
/// for [`quantize_model`] (`&TensorMap` borrow signature) that routes
/// through the streaming pipeline by cloning bytes into a transient
/// `LazyTensorMap`.
///
/// **Peak memory caveat**: this helper temporarily holds both the
/// borrowed `tensor_map` AND a per-tensor clone in the LazyTensorMap
/// during quantize.  Peak memory is therefore HIGHER than `quantize_model`
/// during the call (~2× the largest tensor's bytes briefly) — it is
/// NOT a memory win on its own.
///
/// **Use case**: env-flag-gated production exercise of the streaming
/// code path in main.rs Phase 3 dispatch arms that still need to pass
/// `&tensor_map` to downstream phases (4.5 quality, 4.6 native backend).
/// The actual memory-budget win lands when iter-3 surgery removes the
/// downstream `tensor_map` dependency and the streaming-consuming
/// variant ([`quantize_via_streaming_consuming`]) becomes the call site.
///
/// **iter-72 future-improvement note**: the per-tensor `t.data.clone()`
/// inside the loop is the dominant memory cost.  Two paths to remove it:
/// (a) change `TensorRef::data` to `Arc<Vec<u8>>` so cloning becomes
/// a refcount bump (invasive: `Vec<u8>` is assumed in many sites);
/// (b) extend `LazyTensor` with a borrowed-bytes constructor under a
/// borrowed-lifetime variant (would require LazyTensorMap to grow a
/// lifetime parameter — propagates through the entire pipeline).
/// Option (a) is the cleaner end-state but requires a separate ADR
/// (touches the ir::TensorRef contract).  Until then the wedge is
/// fit-for-purpose as the env-flag transitional toggle.
///
/// Byte-identical to `quantize_model` on the same fixture, verified
/// by `quantize_via_streaming_borrowed_byte_identical_to_quantize_model`.
pub fn quantize_via_streaming_borrowed(
    tensor_map: &crate::ir::TensorMap,
    metadata: &crate::ir::ModelMetadata,
    quantizer: &dyn Quantizer,
    bits: u8,
    group_size: usize,
    progress: &ProgressReporter,
) -> Result<crate::ir::QuantizedModel, QuantizeError> {
    use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
    use std::sync::Arc;

    let mut lazy_map = LazyTensorMap::new();
    // ADR-014 P13 step 5 (iter-77): use `LazyTensor::from_arc_bytes` (iter-76)
    // instead of `from_bytes` with a deep `t.data.clone()`.  Today this still
    // costs one Vec→Arc<Vec<u8>> wrap (the underlying bytes are cloned into
    // a new Vec because `t.data` is still `Vec<u8>` per the un-migrated
    // `TensorRef` contract).  When P13 step 6+ migrates `TensorRef::data` to
    // `Arc<[u8]>`, this wrap becomes a refcount bump — zero bytes copied.
    // The structural shift is captured here so the future migration is a
    // single-line swap (`Arc::clone(&t.data)` instead of `Arc::new(t.data.clone())`)
    // rather than a wholesale rewrite of this loop.
    for (_, t) in tensor_map.tensors.iter() {
        let meta = LazyMeta::new(t.name.clone(), t.shape.clone(), t.dtype);
        // Refcount==1 when materialise() runs (the LazyTensor here is the
        // sole strong-ref holder; tensor_map's `&Vec<u8>` is a borrow).
        // → Arc::unwrap_or_clone path inside materialize() is zero-copy.
        let shared: Arc<Vec<u8>> = Arc::new(t.data.clone());
        lazy_map.insert(LazyTensor::from_arc_bytes(meta, shared));
    }
    quantize_streaming(lazy_map, metadata, quantizer, bits, group_size, progress, false)
}

/// ADR-014 P7 iter-47 — iter-3 wire-up wedge.  Drop-in replacement
/// for [`quantize_model`] that takes ownership of an eager
/// [`crate::ir::TensorMap`] and routes through the streaming
/// pipeline ([`quantize_streaming`]) under the hood.
///
/// **Why this signature**: the iter-3 wire-up swaps `main.rs:1147`'s
/// `materialize_all()` bridge for direct lazy_map consumption.  But
/// each Phase 3 dispatch arm in main.rs currently calls
/// `quantize_model(&tensor_map, ...)` — borrowing the tensor_map.
/// Migrating arm-by-arm needs a streaming entry point that can be
/// dropped in WITHOUT changing the main.rs control flow yet (because
/// later phases like 4.5/4.6 may still need the tensor_map).
///
/// This helper takes ownership of the TensorMap and consumes it via
/// `LazyTensorMap::from_eager` internally, so each arm of main.rs's
/// Phase 3 can choose between:
///   - The legacy eager path: `quantize_model(&tensor_map, ...)`
///   - The streaming path: `quantize_via_streaming_consuming(tensor_map, ...)`
/// with byte-identical output.
///
/// Memory profile: the consumed TensorMap's bytes are MOVED into the
/// LazyTensor wrappers (no clone), so peak memory is unchanged from
/// the eager path during this transitional iter.  The actual
/// memory-budget win lands when iter-3 wires the safetensors-direct
/// streaming path (`read_tensors_lazy`) bypassing eager materialise.
///
/// **Byte-identity contract**: `quantize_via_streaming_consuming(m, ...)`
/// produces a QuantizedModel byte-equal to `quantize_model(&m, ...)`
/// on the same fixture, verified by
/// `quantize_via_streaming_consuming_byte_identical_to_quantize_model`.
pub fn quantize_via_streaming_consuming(
    tensor_map: crate::ir::TensorMap,
    metadata: &crate::ir::ModelMetadata,
    quantizer: &dyn Quantizer,
    bits: u8,
    group_size: usize,
    progress: &ProgressReporter,
) -> Result<crate::ir::QuantizedModel, QuantizeError> {
    let lazy_map = crate::ir::lazy::LazyTensorMap::from_eager(tensor_map);
    quantize_streaming(lazy_map, metadata, quantizer, bits, group_size, progress, false)
}

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
        let mut tensor = lazy
            .materialize()
            .map_err(|e| QuantizeError::TensorQuantizeFailed {
                tensor: name.clone(),
                reason: format!("materialize: {e}"),
            })?;

        if bf16_to_f16 && tensor.dtype == crate::ir::DType::BF16 {
            tensor = tensor
                .to_f16()
                .map_err(|e| QuantizeError::TensorQuantizeFailed {
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
                .map(
                    |(name, lazy)| -> Result<(String, crate::ir::QuantizedTensor), QuantizeError> {
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
                    },
                )
                .collect()
        });

    let quantized_tensors: HashMap<String, crate::ir::QuantizedTensor> =
        quantized_results?.into_iter().collect();

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
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    vec![64, 64],
                    0x10u8,
                ),
                (
                    "model.layers.0.self_attn.k_proj.weight",
                    vec![64, 32],
                    0x20u8,
                ),
                ("model.layers.0.mlp.gate_proj.weight", vec![64, 64], 0x30u8),
                ("model.layers.0.mlp.up_proj.weight", vec![64, 64], 0x40u8),
                ("model.layers.0.mlp.down_proj.weight", vec![64, 64], 0x50u8),
                (
                    "model.layers.1.self_attn.q_proj.weight",
                    vec![64, 64],
                    0x60u8,
                ),
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
                let p = parallel
                    .tensors
                    .get(name)
                    .unwrap_or_else(|| panic!("n_workers={}: missing tensor {}", n_workers, name));
                assert_eq!(s.shape, p.shape, "n_workers={} {name} shape", n_workers);
                assert_eq!(
                    s.original_dtype, p.original_dtype,
                    "n_workers={} {name} dtype",
                    n_workers
                );
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
        let bf16_data: Vec<u8> = (0..64 * 64 * 2)
            .step_by(2)
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
        let vision_t =
            &result.tensors["model.vision_tower.encoder.layers.0.self_attn.q_proj.weight"];
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
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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
            ("output.weight", vec![1, QK_K], "Q6_K", 210),
            ("blk.0.attn_v.weight", vec![1, QK_K], "Q6_K", 210),
            ("blk.5.attn_q.weight", vec![1, QK_K], "Q4_K", 144),
            ("blk.10.ffn_down.weight", vec![1, QK_K], "Q4_K", 144),
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
        let quantizer =
            VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, N_LAYERS);
        let progress = crate::progress::ProgressReporter::new();

        // Eager and streaming paths against same input.
        let eager = quantize_model(&tensor_map, &metadata, &quantizer, 0, 0, &progress).unwrap();
        let streaming = quantize_streaming(
            lazy_map, &metadata, &quantizer, 0, 0, &progress, /* bf16_to_f16 */ false,
        )
        .unwrap();

        // Per-tensor structural + policy + byte-identity checks.
        assert_eq!(eager.tensors.len(), 4);
        assert_eq!(streaming.tensors.len(), 4);
        for (name, _shape, expected_type, expected_bytes) in &fixtures {
            let e = eager.tensors.get(*name).expect("eager tensor missing");
            let s = streaming
                .tensors
                .get(*name)
                .expect("streaming tensor missing");

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
                e.data.len(),
                *expected_bytes,
                "eager {name} block size mismatch"
            );
            assert_eq!(
                s.data.len(),
                *expected_bytes,
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
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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
            ("output.weight", vec![1, QK_K]),               // Q6_K bump
            ("blk.0.attn_v.weight", vec![1, QK_K]),         // Q6_K bump
            ("blk.5.attn_q.weight", vec![1, QK_K]),         // Q4_K base
            ("blk.10.ffn_down.weight", vec![1, QK_K]),      // Q4_K base
            ("blk.20.ffn_down_exps.weight", vec![1, QK_K]), // Q4_K base (MoE)
            ("blk.5.attn_norm.weight", vec![16]),           // 1-D < 32 → preserve
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
        let q = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, N_LAYERS);

        let serial =
            quantize_streaming(make_lazy_map(), &metadata, &q, 0, 0, &progress, false).unwrap();

        // Force >1 worker so rayon fan-out fires.  None defaults to
        // num_cpus; passing Some(4) clamps for determinism.
        let parallel = quantize_streaming_parallel(
            make_lazy_map(),
            &metadata,
            &q,
            0,
            0,
            &progress,
            false,
            Some(4),
        )
        .unwrap();

        assert_eq!(
            serial.tensors.len(),
            parallel.tensors.len(),
            "tensor count diverged"
        );
        assert_eq!(serial.quant_method, parallel.quant_method);
        assert_eq!(serial.bits, parallel.bits);
        assert_eq!(serial.group_size, parallel.group_size);

        for (name, _shape) in &fixtures {
            let s = serial
                .tensors
                .get(*name)
                .unwrap_or_else(|| panic!("serial missing {name}"));
            let p = parallel
                .tensors
                .get(*name)
                .unwrap_or_else(|| panic!("parallel missing {name}"));
            assert_eq!(s.shape, p.shape, "{name} shape");
            assert_eq!(s.original_dtype, p.original_dtype, "{name} dtype");
            assert_eq!(s.quant_info.method, p.quant_info.method, "{name} method");
            assert_eq!(
                s.quant_info.ggml_type.as_deref(),
                p.quant_info.ggml_type.as_deref(),
                "{name} ggml_type"
            );
            assert_eq!(
                s.quant_info.preserved, p.quant_info.preserved,
                "{name} preserved"
            );
            // The actual byte-identity check.
            assert_eq!(
                s.data, p.data,
                "{name}: parallel bytes diverged from serial — \
                 rayon work distribution is non-deterministic for \
                 the variant Quantizer or VariantKQuantizer is not \
                 thread-safe"
            );
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
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant::dequantize_row_q4_k_bytes;
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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
            .map(|c| {
                if c < HIGH_IMPORTANCE_COLS {
                    100.0_f64
                } else {
                    0.01_f64
                }
            })
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

        let q_none = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, 32);
        let q_imatrix = VariantKQuantizer::new(KQuantVariant::Q4_K_M, imatrix, 32);

        let out_none =
            quantize_streaming(make_lazy_map(), &metadata, &q_none, 0, 0, &progress, false)
                .unwrap();
        let out_imatrix = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_imatrix,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();

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
    /// - Q3_K: ≤ 0.10 RMSE (3.4375-bpw codebook → ~2× Q4_K error;
    ///   matches iter-7 isolated-codec gate)
    /// - Q5_K: ≤ 0.025 RMSE (5.5-bpw codebook → ~half Q4_K error)
    /// - Q6_K: ≤ 0.012 RMSE (6.5-bpw codebook → ~quarter Q4_K error)
    ///
    /// Uses the same `blk.5.attn_q.weight` layer-5/32 setup as iter-3x
    /// so attn_q never bumps and each variant lands at its base
    /// target — Q3_K_S/M/L → Q3_K, Q5_K_M → Q5_K, Q6_K → Q6_K.  This
    /// means a regression in the policy that silently routes any
    /// variant to a different target would round-trip with wrong
    /// bytes-per-block AND wrong RMSE.  The two assertions catch the
    /// bug at different layers.
    ///
    /// iter-11: extended to cover Q3_K_S/M/L (all three lock to Q3_K
    /// base on attn_q since none of them apply policy upgrades to
    /// attn_q per `llama-quant.cpp`).
    #[test]
    fn variant_streaming_q5km_q6k_round_trip_rmse_bounds() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant::{
            dequantize_row_q2_k_bytes, dequantize_row_q3_k_bytes, dequantize_row_q4_k_bytes,
            dequantize_row_q5_k_bytes, dequantize_row_q6_k_bytes,
        };
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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
        type DequantFn =
            fn(&[u8], &mut [f32]) -> Result<usize, crate::quantize::k_quant::KQuantError>;
        let cases: Vec<(KQuantVariant, &str, usize, f64, DequantFn)> = vec![
            // Q2_K family lands at Q2_K base on attn_q (no policy bumps for
            // attn_q on any Q2 variant); RMSE bound matches iter-19's
            // `quantize_q2_k_round_trip_smooth_ramp_rmse` (≤ 0.20).
            (
                KQuantVariant::Q2_K_S,
                "Q2_K",
                N_BLOCKS * 84,
                0.20,
                dequantize_row_q2_k_bytes as DequantFn,
            ),
            (
                KQuantVariant::Q2_K,
                "Q2_K",
                N_BLOCKS * 84,
                0.20,
                dequantize_row_q2_k_bytes as DequantFn,
            ),
            // Q3_K family lands at base Q3_K on attn_q (no policy bumps for
            // any Q3_K variant on attn_q per `llama-quant.cpp`); RMSE bound
            // matches iter-7's `quantize_q3_k_round_trip_smooth_ramp_rmse`.
            (
                KQuantVariant::Q3_K_S,
                "Q3_K",
                N_BLOCKS * 110,
                0.10,
                dequantize_row_q3_k_bytes as DequantFn,
            ),
            (
                KQuantVariant::Q3_K_M,
                "Q3_K",
                N_BLOCKS * 110,
                0.10,
                dequantize_row_q3_k_bytes as DequantFn,
            ),
            (
                KQuantVariant::Q3_K_L,
                "Q3_K",
                N_BLOCKS * 110,
                0.10,
                dequantize_row_q3_k_bytes as DequantFn,
            ),
            // Q4_K_S lands at Q4_K base on attn_q (i_layer=5 ≥ 4 so the
            // Q4_K_S attn_v `i<4 → Q5_K` upgrade doesn't apply — and
            // attn_q is never bumped anyway); RMSE bound matches the
            // direct-codec gate (≤ 0.05 on smooth ramp).
            (
                KQuantVariant::Q4_K_S,
                "Q4_K",
                N_BLOCKS * 144,
                0.05,
                dequantize_row_q4_k_bytes as DequantFn,
            ),
            // Q5_K_S has no upgrades anywhere — every tensor at Q5_K base.
            (
                KQuantVariant::Q5_K_S,
                "Q5_K",
                N_BLOCKS * 176,
                0.025,
                dequantize_row_q5_k_bytes as DequantFn,
            ),
            (
                KQuantVariant::Q5_K_M,
                "Q5_K",
                N_BLOCKS * 176,
                0.025,
                dequantize_row_q5_k_bytes as DequantFn,
            ),
            (
                KQuantVariant::Q6_K,
                "Q6_K",
                N_BLOCKS * 210,
                0.012,
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

            let t = out
                .tensors
                .get(TENSOR)
                .unwrap_or_else(|| panic!("{variant}: missing tensor"));

            // structural: variant routed to the expected base target
            assert_eq!(
                t.data.len(),
                expected_bytes,
                "{variant}: bytes ({}) != expected ({expected_bytes})",
                t.data.len()
            );
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some(expected_type),
                "{variant}: ggml_type mismatch"
            );

            // dequantize via the matching codec path
            let mut decoded = vec![0.0_f32; TOTAL];
            let n_decoded = dequant(&t.data, &mut decoded)
                .unwrap_or_else(|e| panic!("{variant}: dequant failed: {e:?}"));
            assert_eq!(
                n_decoded, TOTAL,
                "{variant}: dequant should populate every element"
            );

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
    ///
    /// ADR-014 P7 iter-12 — Q3_K_M streaming-path policy lock.
    /// Mirrors iter-3t's Q4_K_M pattern but exercises every distinct
    /// Q3_K_M policy branch through `quantize_streaming` end-to-end.
    /// Catches a regression where iter-9's `target_for` per-tensor
    /// dispatch silently mis-routes one of the bump categories on the
    /// streaming path while the isolated unit tests still pass.
    ///
    /// Branches covered (per `llama-quant.cpp:439-460/527-528/578-582`):
    ///   1. `output.weight`              → Q6_K (210 bytes)  [output bump]
    ///   2. `blk.0.attn_v.weight`        → Q5_K (176 bytes)  [attn_v i<2]
    ///   3. `blk.5.attn_v.weight`        → Q4_K (144 bytes)  [attn_v i≥2]
    ///   4. `blk.0.ffn_down.weight`      → Q5_K (176 bytes)  [ffn_down i<n/16]
    ///   5. `blk.6.ffn_down.weight`      → Q4_K (144 bytes)  [ffn_down use_more_bits]
    ///   6. `blk.5.ffn_down.weight`      → Q3_K (110 bytes)  [ffn_down else→base]
    ///   7. `blk.10.attn_q.weight`       → Q3_K (110 bytes)  [attn_q never bumps→base]
    ///
    /// `n_layers = 32` so `n/16 = 2`, `n/8 = 4`, `7n/8 = 28`. At i=5,
    /// `use_more_bits` is false ((5-4)%3=1≠2), so ffn_down lands at
    /// base.  At i=6, (6-4)%3=2 → use_more_bits=true → Q4_K.
    #[test]
    fn variant_streaming_q3km_policy_branches() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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

        // (tensor_name, expected_ggml_type, expected_block_bytes)
        let cases: Vec<(&str, &str, usize)> = vec![
            ("output.weight", "Q6_K", 210),
            ("blk.0.attn_v.weight", "Q5_K", 176),
            ("blk.5.attn_v.weight", "Q4_K", 144),
            ("blk.0.ffn_down.weight", "Q5_K", 176),
            ("blk.6.ffn_down.weight", "Q4_K", 144),
            ("blk.5.ffn_down.weight", "Q3_K", 110),
            ("blk.10.attn_q.weight", "Q3_K", 110),
        ];

        let mut lazy_map = LazyTensorMap::new();
        for (i, (name, _ty, _bytes)) in cases.iter().enumerate() {
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                make_f16_payload(i as f32, QK_K),
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = VariantKQuantizer::new(KQuantVariant::Q3_K_M, CalibrationData::None, N_LAYERS);

        let out = quantize_streaming(
            lazy_map, &metadata, &q, 0, 0, &progress, /* bf16_to_f16 */ false,
        )
        .expect("Q3_K_M streaming should succeed");

        assert_eq!(
            out.tensors.len(),
            cases.len(),
            "expected {} tensors in output",
            cases.len()
        );
        assert_eq!(out.quant_method, "Q3_K_M");

        for (name, expected_type, expected_bytes) in cases {
            let t = out
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("missing {name} in streaming output"));
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some(expected_type),
                "Q3_K_M policy regression on {name}: expected ggml_type={expected_type}, got {:?}",
                t.quant_info.ggml_type
            );
            assert_eq!(
                t.data.len(),
                expected_bytes,
                "Q3_K_M policy regression on {name}: expected {expected_bytes} block bytes, got {}",
                t.data.len()
            );
        }
    }

    /// ADR-014 P7 iter-12 — Q3_K_L streaming-path policy lock.
    /// Q3_K_L is the simplest Q3 variant: attn_v ALWAYS Q5_K, ffn_down
    /// ALWAYS Q5_K, output Q6_K, everything else base Q3_K.  No
    /// per-layer bands.  Catches a regression where the Q3_K_L→Q5_K
    /// arms get reverted to the Q3_K_M conditional logic.
    #[test]
    fn variant_streaming_q3kl_policy_branches() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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

        let cases: Vec<(&str, &str, usize)> = vec![
            ("output.weight", "Q6_K", 210),
            ("blk.0.attn_v.weight", "Q5_K", 176),
            ("blk.15.attn_v.weight", "Q5_K", 176), // Q3_K_L: attn_v always Q5_K
            ("blk.0.ffn_down.weight", "Q5_K", 176),
            ("blk.5.ffn_down.weight", "Q5_K", 176), // Q3_K_L: ffn_down always Q5_K (no use_more_bits gate)
            ("blk.10.attn_q.weight", "Q3_K", 110),
        ];

        let mut lazy_map = LazyTensorMap::new();
        for (i, (name, _ty, _bytes)) in cases.iter().enumerate() {
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                make_f16_payload(i as f32, QK_K),
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = VariantKQuantizer::new(KQuantVariant::Q3_K_L, CalibrationData::None, N_LAYERS);

        let out = quantize_streaming(lazy_map, &metadata, &q, 0, 0, &progress, false)
            .expect("Q3_K_L streaming should succeed");

        assert_eq!(out.quant_method, "Q3_K_L");
        for (name, expected_type, expected_bytes) in cases {
            let t = out
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some(expected_type),
                "Q3_K_L on {name}: expected {expected_type}"
            );
            assert_eq!(t.data.len(), expected_bytes, "Q3_K_L on {name}");
        }
    }

    /// ADR-014 P7 iter-16 — Q4_K_S streaming-path policy lock.
    /// Q4_K_S keeps output/token_embd at base Q4_K (no Q6_K bump) and
    /// applies the `i_layer<4 → Q5_K` upgrade only on attn_v
    /// (`layer_mix.rs::target_for` Q4_K_S branch matching
    /// `llama-quant.cpp:536`).  Everything else stays at Q4_K base.
    ///
    /// Branches covered:
    ///   1. `output.weight`            → Q4_K (no bump for _S)
    ///   2. `blk.0.attn_v.weight`      → Q5_K (i<4)
    ///   3. `blk.3.attn_v.weight`      → Q5_K (i<4)
    ///   4. `blk.4.attn_v.weight`      → Q4_K (i≥4 → base)
    ///   5. `blk.20.attn_v.weight`     → Q4_K
    ///   6. `blk.0.ffn_down.weight`    → Q4_K (Q4_K_S has no ffn_down upgrade)
    ///   7. `blk.10.attn_q.weight`     → Q4_K (base)
    #[test]
    fn variant_streaming_q4ks_policy_branches() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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

        let cases: Vec<(&str, &str, usize)> = vec![
            ("output.weight", "Q4_K", 144),
            ("blk.0.attn_v.weight", "Q5_K", 176),
            ("blk.3.attn_v.weight", "Q5_K", 176),
            ("blk.4.attn_v.weight", "Q4_K", 144),
            ("blk.20.attn_v.weight", "Q4_K", 144),
            ("blk.0.ffn_down.weight", "Q4_K", 144),
            ("blk.10.attn_q.weight", "Q4_K", 144),
        ];

        let mut lazy_map = LazyTensorMap::new();
        for (i, (name, _ty, _bytes)) in cases.iter().enumerate() {
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                make_f16_payload(i as f32, QK_K),
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = VariantKQuantizer::new(KQuantVariant::Q4_K_S, CalibrationData::None, N_LAYERS);

        let out = quantize_streaming(lazy_map, &metadata, &q, 0, 0, &progress, false)
            .expect("Q4_K_S streaming should succeed");

        assert_eq!(out.quant_method, "Q4_K_S");
        for (name, expected_type, expected_bytes) in cases {
            let t = out
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some(expected_type),
                "Q4_K_S on {name}: expected {expected_type}, got {:?}",
                t.quant_info.ggml_type
            );
            assert_eq!(
                t.data.len(),
                expected_bytes,
                "Q4_K_S on {name}: expected {expected_bytes} bytes"
            );
        }
    }

    /// ADR-014 P7 iter-16 — Q5_K_S streaming-path policy lock.
    /// Q5_K_S has the simplest policy of all K-quant variants: every
    /// tensor stays at base Q5_K with NO upgrades (no output bump, no
    /// attn_v bump, no ffn_down bump).  This catches a regression
    /// where Q5_K_S accidentally inherits Q4_K_S's `i_layer<4 → Q5_K`
    /// attn_v branch (which would be a no-op since Q5_K_S's base IS
    /// Q5_K, but the policy code path matters for symmetry with
    /// llama.cpp).
    #[test]
    fn variant_streaming_q5ks_policy_all_base() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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

        // Every tensor lands at Q5_K base (176 bytes).  No upgrades.
        let names = [
            "output.weight",
            "blk.0.attn_v.weight",
            "blk.10.attn_v.weight",
            "blk.0.ffn_down.weight",
            "blk.6.ffn_down.weight",
            "blk.20.attn_q.weight",
        ];

        let mut lazy_map = LazyTensorMap::new();
        for (i, name) in names.iter().enumerate() {
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                make_f16_payload(i as f32, QK_K),
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = VariantKQuantizer::new(KQuantVariant::Q5_K_S, CalibrationData::None, N_LAYERS);

        let out = quantize_streaming(lazy_map, &metadata, &q, 0, 0, &progress, false)
            .expect("Q5_K_S streaming should succeed");

        assert_eq!(out.quant_method, "Q5_K_S");
        for name in names {
            let t = out
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some("Q5_K"),
                "Q5_K_S on {name}: should always land at Q5_K base"
            );
            assert_eq!(t.data.len(), 176, "Q5_K_S {name}: 176 bytes (Q5_K block)");
        }
    }

    /// ADR-014 P7 iter-22 — Q2_K streaming-path policy lock.
    /// Q2_K (default) policy per `llama-quant.cpp`: output → Q6_K
    /// (`:439-460`), attn_v → Q4_K (n_gqa>=4 modern default `:512-514`),
    /// ffn_down → Q3_K always (`:571`), everything else → Q2_K base.
    ///
    /// Branches covered:
    ///   1. `output.weight`              → Q6_K (210 bytes)  [output bump]
    ///   2. `blk.0.attn_v.weight`        → Q4_K (144 bytes)  [attn_v]
    ///   3. `blk.20.attn_v.weight`       → Q4_K (144 bytes)  [attn_v all layers]
    ///   4. `blk.0.ffn_down.weight`      → Q3_K (110 bytes)  [ffn_down always]
    ///   5. `blk.10.ffn_down.weight`     → Q3_K (110 bytes)  [ffn_down all layers]
    ///   6. `blk.5.attn_q.weight`        → Q2_K (84 bytes)   [base]
    ///   7. `blk.10.ffn_up.weight`       → Q2_K (84 bytes)   [base]
    #[test]
    fn variant_streaming_q2k_policy_branches() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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

        let cases: Vec<(&str, &str, usize)> = vec![
            ("output.weight",                "Q6_K", 210),
            ("blk.0.attn_v.weight",          "Q4_K", 144),
            ("blk.20.attn_v.weight",         "Q4_K", 144),
            ("blk.0.ffn_down.weight",        "Q3_K", 110),
            ("blk.10.ffn_down.weight",       "Q3_K", 110),
            ("blk.5.attn_q.weight",          "Q2_K", 84),
            ("blk.10.ffn_up.weight",         "Q2_K", 84),
        ];

        let mut lazy_map = LazyTensorMap::new();
        for (i, (name, _ty, _bytes)) in cases.iter().enumerate() {
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                make_f16_payload(i as f32, QK_K),
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = VariantKQuantizer::new(KQuantVariant::Q2_K, CalibrationData::None, N_LAYERS);

        let out = quantize_streaming(lazy_map, &metadata, &q, 0, 0, &progress, false)
            .expect("Q2_K streaming should succeed");

        assert_eq!(out.quant_method, "Q2_K");
        for (name, expected_type, expected_bytes) in cases {
            let t = out
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some(expected_type),
                "Q2_K on {name}: expected {expected_type}, got {:?}",
                t.quant_info.ggml_type
            );
            assert_eq!(
                t.data.len(),
                expected_bytes,
                "Q2_K on {name}: expected {expected_bytes} bytes"
            );
        }
    }

    /// ADR-014 P7 iter-22 — Q2_K_S streaming-path policy lock.
    /// Q2_K_S policy per `llama-quant.cpp`: output → Q6_K, attn_v → Q4_K
    /// (n_gqa>=4), ffn_down i_layer < n_layer/8 → Q4_K else base
    /// (`:572-574`).  attn_q/ffn_up/etc. all → Q2_K base.
    ///
    /// At n_layers=32: n/8 = 4 → ffn_down at i_layer 0,1,2,3 bumps to
    /// Q4_K; i_layer 4+ stays at Q2_K base.
    #[test]
    fn variant_streaming_q2ks_policy_branches() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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

        let cases: Vec<(&str, &str, usize)> = vec![
            ("output.weight",                "Q6_K", 210),
            ("blk.0.attn_v.weight",          "Q4_K", 144),
            ("blk.0.ffn_down.weight",        "Q4_K", 144), // i<n/8 → Q4_K
            ("blk.3.ffn_down.weight",        "Q4_K", 144), // i<n/8=4 → Q4_K
            ("blk.4.ffn_down.weight",        "Q2_K",  84), // i>=n/8 → base
            ("blk.20.ffn_down.weight",       "Q2_K",  84), // base
            ("blk.5.attn_q.weight",          "Q2_K",  84), // base
        ];

        let mut lazy_map = LazyTensorMap::new();
        for (i, (name, _ty, _bytes)) in cases.iter().enumerate() {
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                make_f16_payload(i as f32, QK_K),
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = VariantKQuantizer::new(KQuantVariant::Q2_K_S, CalibrationData::None, N_LAYERS);

        let out = quantize_streaming(lazy_map, &metadata, &q, 0, 0, &progress, false)
            .expect("Q2_K_S streaming should succeed");

        assert_eq!(out.quant_method, "Q2_K_S");
        for (name, expected_type, expected_bytes) in cases {
            let t = out
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some(expected_type),
                "Q2_K_S on {name}: expected {expected_type}"
            );
            assert_eq!(t.data.len(), expected_bytes, "Q2_K_S on {name}");
        }
    }

    /// ADR-014 P7 iter-24 — DwqK::P28 streaming-path end-to-end policy lock.
    ///
    /// Mirrors iter-12 (Q3_K_M) and iter-22 (Q2_K) for the DWQ class:
    /// drives `DwqKQuantizer::new(P28, sensitive_layers, None)` end-to-end
    /// through `quantize_streaming` on a multi-tensor fixture spanning
    /// both buckets and asserts the byte-level routing contract.
    ///
    /// Pre-iter-24 the P28 contract was only gated by isolated
    /// `quantize_tensor`-call tests in `dwq_k_quantizer.rs`; this test
    /// pins the contract through the actual streaming loop on a `LazyTensorMap`
    /// — the production code path.  Closes the gap left when iter-20
    /// unblocked P28 base→Q2_K routing.
    ///
    /// P28 policy:
    /// - Sensitive layers (`is_sensitive_tensor` ⇒ true) → Q8_0 (34 bytes/block).
    /// - Base layers + non-`model.layers.<N>.…` names → Q2_K (84 bytes/block).
    ///
    /// `DwqKQuantizer` uses the HF naming convention (`model.layers.<N>.…`),
    /// not GGUF (`blk.<N>.…`), per `extract_layer_index`'s contract.
    #[test]
    fn variant_streaming_dwq28_policy_branches() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::dwq_k_quantizer::{DwqKQuantizer, DwqKVariant};

        const QK_K: usize = 256;

        let make_f32_payload = |seed: f32, len: usize| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(len * 4);
            for i in 0..len {
                let v = (i as f32 / len as f32) * 2.0 - 1.0 + seed * 0.01;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes
        };

        // Sensitive set: layers 3, 5, 12.  All other layers → base bucket.
        // Mix of both buckets + a non-layer tensor (output.weight) which
        // falls through to base per `extract_layer_index` returning None.
        //
        // Per-row byte budgets at 256 elements:
        //   Q8_0  block = 32 elems × 34 bytes/block → 8 × 34 = 272 bytes
        //   Q2_K  super-block = 256 elems × 84 bytes/super-block → 84 bytes
        let cases: Vec<(&str, &str, usize)> = vec![
            // Sensitive bucket → Q8_0 (272 bytes/row at QK_K=256).
            ("model.layers.3.self_attn.q_proj.weight",  "Q8_0", 272),
            ("model.layers.5.mlp.down_proj.weight",     "Q8_0", 272),
            ("model.layers.12.self_attn.v_proj.weight", "Q8_0", 272),
            // Base bucket → Q2_K (84 bytes/row at QK_K=256).
            ("model.layers.0.self_attn.q_proj.weight", "Q2_K", 84),
            ("model.layers.4.self_attn.v_proj.weight", "Q2_K", 84),
            ("model.layers.20.mlp.down_proj.weight",   "Q2_K", 84),
            // Non-`model.layers.<N>.…` → falls through to base bucket.
            ("output.weight",                          "Q2_K", 84),
            ("lm_head.weight",                         "Q2_K", 84),
        ];

        let mut lazy_map = LazyTensorMap::new();
        for (i, (name, _ty, _bytes)) in cases.iter().enumerate() {
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F32),
                make_f32_payload(i as f32, QK_K),
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = DwqKQuantizer::new(
            DwqKVariant::P28,
            &[3..=3, 5..=5, 12..=12],
            CalibrationData::None,
        );

        let out = quantize_streaming(lazy_map, &metadata, &q, 0, 0, &progress, false)
            .expect("DwqK::P28 streaming should succeed");

        assert_eq!(out.quant_method, "dwq-k-mixed-2-8");
        assert_eq!(out.tensors.len(), cases.len());
        for (name, expected_type, expected_bytes) in cases {
            let t = out
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some(expected_type),
                "P28 on {name}: expected {expected_type}, got {:?}",
                t.quant_info.ggml_type
            );
            assert_eq!(
                t.data.len(),
                expected_bytes,
                "P28 on {name}: expected {expected_bytes} bytes"
            );
        }
    }

    #[test]
    fn variant_streaming_q4km_round_trip_rmse_bound() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant::dequantize_row_q4_k_bytes;
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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
        let q = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, 32);

        let out = quantize_streaming(
            lazy_map, &metadata, &q, 0, 0, &progress, /* bf16_to_f16 */ false,
        )
        .unwrap();

        let t = out.tensors.get(TENSOR).expect("missing tensor");
        // structural sanity: 4 super-blocks × 144 bytes = 576 bytes Q4_K
        assert_eq!(
            t.data.len(),
            N_BLOCKS * 144,
            "Q4_K bytes count mismatch: {} blocks × 144",
            N_BLOCKS
        );
        assert_eq!(t.quant_info.ggml_type.as_deref(), Some("Q4_K"));

        // Dequantize back to F32 via the same flat-bytes path the GGUF
        // backend (and downstream consumers) will use.
        let mut decoded = vec![0.0_f32; TOTAL];
        let n_decoded = dequantize_row_q4_k_bytes(&t.data, &mut decoded)
            .expect("dequantize_row_q4_k_bytes failed");
        assert_eq!(n_decoded, TOTAL, "dequant should populate every element");

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
        assert!(
            nonzero_count > TOTAL / 2,
            "decoded output is mostly zero — likely a target/tag mismatch"
        );
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
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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
        let q_none = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, N_LAYERS);
        let q_imatrix = VariantKQuantizer::new(KQuantVariant::Q4_K_M, imatrix, N_LAYERS);

        let out_none =
            quantize_streaming(make_lazy_map(), &metadata, &q_none, 0, 0, &progress, false)
                .unwrap();
        let out_imatrix = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_imatrix,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();

        let t_none = out_none.tensors.get(TENSOR).expect("none-path missing");
        let t_imatrix = out_imatrix
            .tensors
            .get(TENSOR)
            .expect("imatrix-path missing");

        // Both produce Q4_K blocks (same target, same size).
        assert_eq!(
            t_none.data.len(),
            144,
            "none path should emit one Q4_K block"
        );
        assert_eq!(
            t_imatrix.data.len(),
            144,
            "imatrix path should emit one Q4_K block"
        );
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

    /// ADR-014 P7 iter-13 — Q3_K_M imatrix-vs-none divergence gate.
    /// Mirrors iter-3w's Q4_K_M pattern but for the Q3_K codec path
    /// added in iter-9.  Without this gate, a regression that silently
    /// routes the Q3_K path through `quantize_row_q3_k_to_bytes`
    /// (uncalibrated) instead of `quantize_row_q3_k_imatrix_to_bytes`
    /// would pass every existing structural test (right block size,
    /// right ggml_type tag, right RMSE on uniform input) but fail
    /// quietly on real models with non-uniform activation importance.
    ///
    /// Mechanism: `make_qkx2_quants` (uncalibrated) minimises uniform
    /// L2; the imatrix Q3_K path (a closed form analogous to
    /// `make_qkx3_quants` for higher-bit K-quants) minimises
    /// importance-weighted L2.  Different objectives → different
    /// codebooks → different bytes when importance is non-uniform.
    ///
    /// Tensor: `blk.5.attn_q.weight` at layer 5/32 — attn_q never
    /// bumps for any Q3 variant, so both paths land on the Q3_K base
    /// target and bytes are directly comparable (110 byte block).
    #[test]
    fn variant_imatrix_q3km_diverges_from_none_through_streaming() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;
        const TENSOR: &str = "blk.5.attn_q.weight";

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

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

        // Non-uniform importance: 100× boost first 16 columns, 0.01× rest.
        let mut col = ImatrixCollector::new();
        let acts: Vec<f32> = (0..QK_K)
            .map(|i| if i < 16 { 100.0 } else { 0.01 })
            .collect();
        col.accumulate_dense(TENSOR, &acts, 1, QK_K).unwrap();
        col.record_chunk();
        let imatrix = CalibrationData::from_imatrix_collector(&col);
        assert!(imatrix.is_some(), "imatrix calibration should be populated");

        // Run SAME input through Q3_K_M with NoneCalibrator vs imatrix.
        let q_none = VariantKQuantizer::new(KQuantVariant::Q3_K_M, CalibrationData::None, N_LAYERS);
        let q_imatrix = VariantKQuantizer::new(KQuantVariant::Q3_K_M, imatrix, N_LAYERS);

        let out_none =
            quantize_streaming(make_lazy_map(), &metadata, &q_none, 0, 0, &progress, false)
                .unwrap();
        let out_imatrix = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_imatrix,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();

        let t_none = out_none.tensors.get(TENSOR).expect("none-path missing");
        let t_imatrix = out_imatrix
            .tensors
            .get(TENSOR)
            .expect("imatrix-path missing");

        // Both produce Q3_K blocks (same target, same size).
        assert_eq!(
            t_none.data.len(),
            110,
            "none path should emit one Q3_K block (110 bytes)"
        );
        assert_eq!(
            t_imatrix.data.len(),
            110,
            "imatrix path should emit one Q3_K block (110 bytes)"
        );
        assert_eq!(
            t_none.quant_info.ggml_type.as_deref(),
            Some("Q3_K"),
            "none path should tag Q3_K"
        );
        assert_eq!(
            t_imatrix.quant_info.ggml_type.as_deref(),
            Some("Q3_K"),
            "imatrix path should tag Q3_K"
        );

        // The divergence gate: bytes MUST differ. If they're equal,
        // the imatrix calibration is silently being ignored on the
        // Q3_K codec path.
        assert_ne!(
            t_none.data, t_imatrix.data,
            "Q3_K_M imatrix-weighted bytes must differ from uncalibrated \
             Q3_K bytes when importance is non-uniform — otherwise the \
             Q3_K imatrix dispatch in `quantize_row_to_bytes` is broken"
        );
    }

    /// ADR-014 P7 iter-14 — Q3_K_M imatrix improves importance-weighted
    /// error vs uncalibrated.  Mirrors iter-3z's Q4_K_M directional
    /// gate but for the Q3_K codec path.  Without this, a regression
    /// that flips the imatrix Q3_K codebook search to minimise the
    /// **wrong** objective (or accidentally inverts the importance
    /// vector) would still pass iter-13's "bytes differ" gate but
    /// would silently degrade real-model quality on the Q3_K path.
    ///
    /// Mechanism: imatrix Q3_K minimises importance-weighted L2;
    /// uncalibrated Q3_K minimises uniform L2.  When importance is
    /// highly non-uniform (here: 10000× ratio on first 16 columns
    /// via 100×/0.01× activations), the imatrix path's
    /// importance-weighted SSE must be **lower** than the uncalibrated
    /// path's even though uniform-RMSE on the full row may be higher
    /// (by design — imatrix trades low-importance accuracy for
    /// high-importance accuracy).
    ///
    /// Adversarial input pattern matches iter-3z: high-importance
    /// columns get wide-magnitude data (-3..3) while low-importance
    /// get small steady values (~0.1), forcing the codebook search
    /// to choose between accuracy on high-mag/high-importance vs
    /// low-mag/low-importance.
    #[test]
    fn variant_imatrix_q3km_lowers_importance_weighted_error() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant::dequantize_row_q3_k_bytes;
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_BLOCKS: usize = 4;
        const TOTAL: usize = N_BLOCKS * QK_K;
        const TENSOR: &str = "blk.5.attn_q.weight";
        const HIGH_IMPORTANCE_COLS: usize = 16;

        // accumulate_dense → ImatrixCollector stores
        // `values[c] = sum_t activations[c]^2`.  100×/0.01× activations
        // give 10000× importance ratio (100 vs 0.01).
        let acts: Vec<f32> = (0..QK_K)
            .map(|c| if c < HIGH_IMPORTANCE_COLS { 10.0 } else { 0.1 })
            .collect();
        let mut col = ImatrixCollector::new();
        col.accumulate_dense(TENSOR, &acts, 1, QK_K).unwrap();
        col.record_chunk();
        let imatrix = CalibrationData::from_imatrix_collector(&col);

        // Per-column importance for the SSE computation (matches the
        // codec's per-column weighting).
        let importance: Vec<f64> = (0..QK_K)
            .map(|c| {
                if c < HIGH_IMPORTANCE_COLS {
                    100.0_f64
                } else {
                    0.01_f64
                }
            })
            .collect();

        // Adversarial input: high-importance cols get wide range,
        // low-importance cols get small steady values.
        let original_f32: Vec<f32> = (0..TOTAL)
            .map(|i| {
                let c = i % QK_K;
                let r = (i / QK_K) as f32;
                if c < HIGH_IMPORTANCE_COLS {
                    let t = (c as f32 / HIGH_IMPORTANCE_COLS as f32) + r * 0.5;
                    -3.0 + 6.0 * (t.fract())
                } else {
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
        let q_none = VariantKQuantizer::new(KQuantVariant::Q3_K_M, CalibrationData::None, 32);
        let q_imatrix = VariantKQuantizer::new(KQuantVariant::Q3_K_M, imatrix, 32);

        let out_none =
            quantize_streaming(make_lazy_map(), &metadata, &q_none, 0, 0, &progress, false)
                .unwrap();
        let out_imatrix = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_imatrix,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();

        let bytes_none = &out_none.tensors[TENSOR].data;
        let bytes_imatrix = &out_imatrix.tensors[TENSOR].data;
        // structural sanity: both Q3_K, 4 blocks of 110 = 440 bytes
        assert_eq!(bytes_none.len(), N_BLOCKS * 110);
        assert_eq!(bytes_imatrix.len(), N_BLOCKS * 110);

        let mut decoded_none = vec![0.0_f32; TOTAL];
        let mut decoded_imatrix = vec![0.0_f32; TOTAL];
        dequantize_row_q3_k_bytes(bytes_none, &mut decoded_none).unwrap();
        dequantize_row_q3_k_bytes(bytes_imatrix, &mut decoded_imatrix).unwrap();

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

        // Directional gate: imatrix Q3_K wSSE strictly < uncalibrated.
        // If they're equal, calibration is being ignored.  If imatrix
        // is higher, the codec is minimising the wrong objective
        // (sign-flipped weights, etc).
        assert!(
            wsse_imatrix < wsse_none,
            "Q3_K_M imatrix-weighted SSE {wsse_imatrix:.6} should be \
             < uncalibrated SSE {wsse_none:.6} — imatrix Q3_K path is \
             failing to minimise the importance-weighted objective"
        );
    }

    /// ADR-014 P7 iter-23 — Q2_K imatrix-vs-none divergence gate.
    /// Mirrors iter-13's Q3_K_M divergence test for the Q2_K codec
    /// path (landed in iter-19+iter-20).  Without this gate, a
    /// regression that silently routes the Q2_K imatrix dispatch
    /// through `quantize_row_q2_k_to_bytes` (uncalibrated) instead of
    /// `quantize_row_q2_k_imatrix_to_bytes` would pass every
    /// structural test but quietly degrade dwq28 quality on real
    /// models with non-uniform activation importance.
    ///
    /// Tensor: `blk.5.attn_q.weight` at layer 5/32 — attn_q never
    /// bumps for any Q2 variant, so both paths land on the Q2_K base
    /// target (84-byte block).
    #[test]
    fn variant_imatrix_q2k_diverges_from_none_through_streaming() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;
        const TENSOR: &str = "blk.5.attn_q.weight";

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

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

        // Non-uniform importance: 100× boost first 16 columns, 0.01× rest.
        let mut col = ImatrixCollector::new();
        let acts: Vec<f32> = (0..QK_K)
            .map(|i| if i < 16 { 100.0 } else { 0.01 })
            .collect();
        col.accumulate_dense(TENSOR, &acts, 1, QK_K).unwrap();
        col.record_chunk();
        let imatrix = CalibrationData::from_imatrix_collector(&col);

        let q_none = VariantKQuantizer::new(
            KQuantVariant::Q2_K,
            CalibrationData::None,
            N_LAYERS,
        );
        let q_imatrix =
            VariantKQuantizer::new(KQuantVariant::Q2_K, imatrix, N_LAYERS);

        let out_none = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_none,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();
        let out_imatrix = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_imatrix,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();

        let t_none = out_none
            .tensors
            .get(TENSOR)
            .expect("none-path missing");
        let t_imatrix = out_imatrix
            .tensors
            .get(TENSOR)
            .expect("imatrix-path missing");

        assert_eq!(t_none.data.len(), 84, "Q2_K block size");
        assert_eq!(t_imatrix.data.len(), 84, "Q2_K block size");
        assert_eq!(
            t_none.quant_info.ggml_type.as_deref(),
            Some("Q2_K")
        );
        assert_eq!(
            t_imatrix.quant_info.ggml_type.as_deref(),
            Some("Q2_K")
        );

        assert_ne!(
            t_none.data, t_imatrix.data,
            "Q2_K imatrix-weighted bytes must differ from uncalibrated \
             when importance is non-uniform — otherwise the Q2_K imatrix \
             dispatch in `quantize_row_to_bytes` is broken"
        );
    }

    /// ADR-014 P7 iter-23 — Q2_K imatrix lowers importance-weighted
    /// error vs uncalibrated.  Directional-correctness gate that
    /// mirrors iter-14's Q3_K_M pattern but for the Q2_K codec path.
    ///
    /// Without this, a regression that flips the Q2_K imatrix codebook
    /// search to minimise the wrong objective (or inverts the
    /// importance vector) would still pass iter-23's "bytes differ"
    /// gate but would silently degrade real-model quality on dwq28.
    ///
    /// Adversarial input pattern matches iter-3z/iter-14: high-importance
    /// columns get wide-magnitude data; low-importance get small steady
    /// values, forcing the codebook to choose between accuracy on
    /// high-mag/high-importance vs low-mag/low-importance.
    #[test]
    fn variant_imatrix_q2k_lowers_importance_weighted_error() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant::dequantize_row_q2_k_bytes;
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_BLOCKS: usize = 4;
        const TOTAL: usize = N_BLOCKS * QK_K;
        const TENSOR: &str = "blk.5.attn_q.weight";
        const HIGH_IMPORTANCE_COLS: usize = 16;

        // accumulate_dense → values[c] = sum_t activations[c]^2.
        // 100×/0.01× activations → 10000× importance ratio.
        let acts: Vec<f32> = (0..QK_K)
            .map(|c| {
                if c < HIGH_IMPORTANCE_COLS {
                    10.0
                } else {
                    0.1
                }
            })
            .collect();
        let mut col = ImatrixCollector::new();
        col.accumulate_dense(TENSOR, &acts, 1, QK_K).unwrap();
        col.record_chunk();
        let imatrix = CalibrationData::from_imatrix_collector(&col);

        let importance: Vec<f64> = (0..QK_K)
            .map(|c| {
                if c < HIGH_IMPORTANCE_COLS {
                    100.0_f64
                } else {
                    0.01_f64
                }
            })
            .collect();

        // Adversarial input: wide-mag high-importance, small-steady low-importance.
        let original_f32: Vec<f32> = (0..TOTAL)
            .map(|i| {
                let c = i % QK_K;
                let r = (i / QK_K) as f32;
                if c < HIGH_IMPORTANCE_COLS {
                    let t = (c as f32 / HIGH_IMPORTANCE_COLS as f32) + r * 0.5;
                    -3.0 + 6.0 * (t.fract())
                } else {
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
            KQuantVariant::Q2_K,
            CalibrationData::None,
            32,
        );
        let q_imatrix = VariantKQuantizer::new(KQuantVariant::Q2_K, imatrix, 32);

        let out_none = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_none,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();
        let out_imatrix = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_imatrix,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();

        let bytes_none = &out_none.tensors[TENSOR].data;
        let bytes_imatrix = &out_imatrix.tensors[TENSOR].data;
        assert_eq!(bytes_none.len(), N_BLOCKS * 84);
        assert_eq!(bytes_imatrix.len(), N_BLOCKS * 84);

        let mut decoded_none = vec![0.0_f32; TOTAL];
        let mut decoded_imatrix = vec![0.0_f32; TOTAL];
        dequantize_row_q2_k_bytes(bytes_none, &mut decoded_none).unwrap();
        dequantize_row_q2_k_bytes(bytes_imatrix, &mut decoded_imatrix).unwrap();

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

        assert!(
            wsse_imatrix < wsse_none,
            "Q2_K imatrix-weighted SSE {wsse_imatrix:.6} should be \
             < uncalibrated SSE {wsse_none:.6} — imatrix Q2_K path is \
             failing to minimise the importance-weighted objective"
        );
    }

    /// ADR-014 P7 iter-25 — Q2_K_S imatrix-vs-none divergence gate.
    /// Mirrors iter-23 for the Q2_K_S variant.  Q2_K_S's policy routes
    /// some tensors to Q4_K (i<n/8 ffn_down) and some to Q2_K base
    /// (attn_q, ffn_up, ffn_down at i>=n/8); this test exercises the
    /// base-routed path with imatrix calibration to ensure the
    /// variant→codec dispatch threads imatrix through both legs.
    ///
    /// Without this gate, a regression where Q2_K_S's variant dispatch
    /// silently strips the imatrix payload (vs Q2_K which has its own
    /// iter-23 gate) would still pass policy-routing tests but quietly
    /// degrade dwq28-style models that combine Q2_K_S's mixed-precision
    /// layout with imatrix calibration.
    ///
    /// Tensor: `blk.5.attn_q.weight` at layer 5/32 — attn_q never
    /// bumps for Q2_K_S, lands on Q2_K base target (84-byte block).
    #[test]
    fn variant_imatrix_q2ks_diverges_from_none_through_streaming() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;
        const TENSOR: &str = "blk.5.attn_q.weight";

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

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

        // Non-uniform importance: 100× boost first 16 columns, 0.01× rest.
        let mut col = ImatrixCollector::new();
        let acts: Vec<f32> = (0..QK_K)
            .map(|i| if i < 16 { 100.0 } else { 0.01 })
            .collect();
        col.accumulate_dense(TENSOR, &acts, 1, QK_K).unwrap();
        col.record_chunk();
        let imatrix = CalibrationData::from_imatrix_collector(&col);

        let q_none = VariantKQuantizer::new(
            KQuantVariant::Q2_K_S,
            CalibrationData::None,
            N_LAYERS,
        );
        let q_imatrix =
            VariantKQuantizer::new(KQuantVariant::Q2_K_S, imatrix, N_LAYERS);

        let out_none = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_none,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();
        let out_imatrix = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_imatrix,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();

        assert_eq!(out_none.quant_method, "Q2_K_S");
        assert_eq!(out_imatrix.quant_method, "Q2_K_S");

        let t_none = out_none
            .tensors
            .get(TENSOR)
            .expect("none-path missing");
        let t_imatrix = out_imatrix
            .tensors
            .get(TENSOR)
            .expect("imatrix-path missing");

        assert_eq!(t_none.data.len(), 84, "Q2_K base block size");
        assert_eq!(t_imatrix.data.len(), 84, "Q2_K base block size");
        assert_eq!(
            t_none.quant_info.ggml_type.as_deref(),
            Some("Q2_K"),
            "Q2_K_S attn_q must route to Q2_K base"
        );
        assert_eq!(
            t_imatrix.quant_info.ggml_type.as_deref(),
            Some("Q2_K"),
            "Q2_K_S attn_q must route to Q2_K base"
        );

        assert_ne!(
            t_none.data, t_imatrix.data,
            "Q2_K_S imatrix-weighted bytes must differ from uncalibrated \
             when importance is non-uniform — otherwise the Q2_K_S variant \
             dispatch silently strips the imatrix payload before reaching \
             the Q2_K codec"
        );
    }

    /// ADR-014 P7 iter-25 — Q2_K_S imatrix lowers importance-weighted
    /// error vs uncalibrated.  Directional-correctness gate that
    /// mirrors iter-23's second test (Q2_K) but for the Q2_K_S
    /// variant dispatch.
    ///
    /// Without this, a regression that flips the Q2_K_S variant's
    /// imatrix routing to minimise the wrong objective would still
    /// pass the divergence gate but quietly degrade real-model
    /// dwq28+Q2_K_S quality.
    #[test]
    fn variant_imatrix_q2ks_lowers_importance_weighted_error() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant::dequantize_row_q2_k_bytes;
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_BLOCKS: usize = 4;
        const TOTAL: usize = N_BLOCKS * QK_K;
        const TENSOR: &str = "blk.5.attn_q.weight";
        const HIGH_IMPORTANCE_COLS: usize = 16;

        let acts: Vec<f32> = (0..QK_K)
            .map(|c| {
                if c < HIGH_IMPORTANCE_COLS {
                    10.0
                } else {
                    0.1
                }
            })
            .collect();
        let mut col = ImatrixCollector::new();
        col.accumulate_dense(TENSOR, &acts, 1, QK_K).unwrap();
        col.record_chunk();
        let imatrix = CalibrationData::from_imatrix_collector(&col);

        let importance: Vec<f64> = (0..QK_K)
            .map(|c| {
                if c < HIGH_IMPORTANCE_COLS {
                    100.0_f64
                } else {
                    0.01_f64
                }
            })
            .collect();

        let original_f32: Vec<f32> = (0..TOTAL)
            .map(|i| {
                let c = i % QK_K;
                let r = (i / QK_K) as f32;
                if c < HIGH_IMPORTANCE_COLS {
                    let t = (c as f32 / HIGH_IMPORTANCE_COLS as f32) + r * 0.5;
                    -3.0 + 6.0 * (t.fract())
                } else {
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
            KQuantVariant::Q2_K_S,
            CalibrationData::None,
            32,
        );
        let q_imatrix = VariantKQuantizer::new(KQuantVariant::Q2_K_S, imatrix, 32);

        let out_none = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_none,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();
        let out_imatrix = quantize_streaming(
            make_lazy_map(),
            &metadata,
            &q_imatrix,
            0,
            0,
            &progress,
            false,
        )
        .unwrap();

        let bytes_none = &out_none.tensors[TENSOR].data;
        let bytes_imatrix = &out_imatrix.tensors[TENSOR].data;
        assert_eq!(bytes_none.len(), N_BLOCKS * 84);
        assert_eq!(bytes_imatrix.len(), N_BLOCKS * 84);

        let mut decoded_none = vec![0.0_f32; TOTAL];
        let mut decoded_imatrix = vec![0.0_f32; TOTAL];
        dequantize_row_q2_k_bytes(bytes_none, &mut decoded_none).unwrap();
        dequantize_row_q2_k_bytes(bytes_imatrix, &mut decoded_imatrix).unwrap();

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

        assert!(
            wsse_imatrix < wsse_none,
            "Q2_K_S imatrix-weighted SSE {wsse_imatrix:.6} should be \
             < uncalibrated SSE {wsse_none:.6} — imatrix Q2_K_S variant \
             dispatch is failing to minimise the importance-weighted objective"
        );
    }

    /// ADR-014 P7 iter-26 — Q2_K_S vs Q2_K cross-variant divergence gate.
    /// Mid-policy regression catcher: the two Q2 variants share the same
    /// codec for the base bucket but their policies diverge on `ffn_down`
    /// per `llama-quant.cpp:571-574`:
    ///   - Q2_K:    ffn_down → Q3_K always (110 bytes/super-block)
    ///   - Q2_K_S:  ffn_down i<n/8 → Q4_K (144 bytes), else → Q2_K (84 bytes)
    ///
    /// Without this gate, a regression that collapses both variants into
    /// the same dispatch (e.g. variant→target table mis-keyed, or
    /// `target_for` losing the variant discriminant) would still pass
    /// every same-variant test but silently make Q2_K_S a Q2_K alias.
    ///
    /// At n_layers=32, n/8=4, so:
    ///   - `blk.0.ffn_down.weight` (i=0):   Q2_K → 110 bytes; Q2_K_S → 144 bytes
    ///   - `blk.10.ffn_down.weight` (i=10): Q2_K → 110 bytes; Q2_K_S →  84 bytes
    ///   - `blk.5.attn_q.weight`   (i=5):   both → 84 bytes (control: same)
    ///
    /// Asserts byte-size + ggml_type divergence on the two ffn_down cases
    /// AND byte-size match on the attn_q control (proves the gate is
    /// catching policy-driven divergence, not test-fixture-driven noise).
    #[test]
    fn variant_streaming_q2k_vs_q2ks_cross_variant_divergence() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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

        // (name, q2k_type, q2k_bytes, q2ks_type, q2ks_bytes, divergent?)
        let cases: Vec<(&str, &str, usize, &str, usize, bool)> = vec![
            ("blk.0.ffn_down.weight",  "Q3_K", 110, "Q4_K", 144, true),
            ("blk.10.ffn_down.weight", "Q3_K", 110, "Q2_K",  84, true),
            ("blk.5.attn_q.weight",    "Q2_K",  84, "Q2_K",  84, false),
        ];

        let mut lazy_map_q2k = LazyTensorMap::new();
        let mut lazy_map_q2ks = LazyTensorMap::new();
        for (i, (name, _t1, _b1, _t2, _b2, _div)) in cases.iter().enumerate() {
            let payload = make_f16_payload(i as f32, QK_K);
            lazy_map_q2k.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                payload.clone(),
            ));
            lazy_map_q2ks.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                payload,
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        let q_q2k = VariantKQuantizer::new(
            KQuantVariant::Q2_K,
            CalibrationData::None,
            N_LAYERS,
        );
        let q_q2ks = VariantKQuantizer::new(
            KQuantVariant::Q2_K_S,
            CalibrationData::None,
            N_LAYERS,
        );

        let out_q2k =
            quantize_streaming(lazy_map_q2k, &metadata, &q_q2k, 0, 0, &progress, false)
                .expect("Q2_K streaming");
        let out_q2ks =
            quantize_streaming(lazy_map_q2ks, &metadata, &q_q2ks, 0, 0, &progress, false)
                .expect("Q2_K_S streaming");

        assert_eq!(out_q2k.quant_method, "Q2_K");
        assert_eq!(out_q2ks.quant_method, "Q2_K_S");

        for (name, q2k_type, q2k_bytes, q2ks_type, q2ks_bytes, divergent) in cases {
            let t_q2k = out_q2k
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("Q2_K missing {name}"));
            let t_q2ks = out_q2ks
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("Q2_K_S missing {name}"));

            assert_eq!(
                t_q2k.quant_info.ggml_type.as_deref(),
                Some(q2k_type),
                "Q2_K on {name}: expected {q2k_type}, got {:?}",
                t_q2k.quant_info.ggml_type
            );
            assert_eq!(
                t_q2k.data.len(),
                q2k_bytes,
                "Q2_K on {name}: expected {q2k_bytes} bytes"
            );

            assert_eq!(
                t_q2ks.quant_info.ggml_type.as_deref(),
                Some(q2ks_type),
                "Q2_K_S on {name}: expected {q2ks_type}, got {:?}",
                t_q2ks.quant_info.ggml_type
            );
            assert_eq!(
                t_q2ks.data.len(),
                q2ks_bytes,
                "Q2_K_S on {name}: expected {q2ks_bytes} bytes"
            );

            if divergent {
                // Divergent policy → ggml_type must differ across variants.
                // (Byte-data also differs trivially via byte-count, but the
                // type tag is the contract surface — bytes can change for
                // many reasons; ggml_type only changes on a routing change.)
                assert_ne!(
                    t_q2k.quant_info.ggml_type, t_q2ks.quant_info.ggml_type,
                    "Q2_K and Q2_K_S must route {name} to DIFFERENT codecs \
                     per ffn_down policy divergence — Q2_K_S has collapsed \
                     into a Q2_K alias"
                );
            } else {
                // Control: shared base → same codec, same bytes (deterministic).
                assert_eq!(
                    t_q2k.quant_info.ggml_type, t_q2ks.quant_info.ggml_type,
                    "Q2_K and Q2_K_S must route shared-base {name} to the \
                     SAME codec — divergence here would be a fixture bug"
                );
                assert_eq!(
                    t_q2k.data, t_q2ks.data,
                    "Q2_K and Q2_K_S on shared-base {name} must produce \
                     byte-identical output (uncalibrated, deterministic)"
                );
            }
        }
    }

    /// ADR-014 P7 iter-27 — Q3_K_M vs Q3_K_L cross-variant divergence gate.
    /// Mirrors iter-26 for the Q3 family.  Per `llama-quant.cpp:527-588`,
    /// Q3_K_M and Q3_K_L share the same Q3_K base codec but their
    /// attn_v + ffn_down policies diverge:
    ///
    ///   Q3_K_M:
    ///     - attn_v   i<2          → Q5_K (176)
    ///     - attn_v   i≥2          → Q4_K (144)
    ///     - ffn_down i<n/16       → Q5_K (176)
    ///     - ffn_down use_more_bits → Q4_K (144)
    ///     - ffn_down else         → Q3_K (110)
    ///   Q3_K_L:
    ///     - attn_v   any          → Q5_K (176)
    ///     - ffn_down any          → Q5_K (176)
    ///
    /// At n_layers=32 (n/8=4, 7n/8=28, n/16=2):
    ///   - blk.5.attn_v.weight    (i=5, ≥2):                Q3_K_M → Q4_K vs Q3_K_L → Q5_K (divergent)
    ///   - blk.9.ffn_down.weight  (i=9, use_more_bits true): Q3_K_M → Q4_K vs Q3_K_L → Q5_K (divergent)
    ///   - blk.13.ffn_down.weight (i=13, else):              Q3_K_M → Q3_K vs Q3_K_L → Q5_K (divergent)
    ///   - blk.0.attn_v.weight    (i=0, <2):                 both → Q5_K (control: same)
    ///   - blk.5.attn_q.weight    (i=5):                     both → Q3_K (control: same)
    ///
    /// Without this gate, a regression that collapses the Q3_K_L policy
    /// arms into Q3_K_M's tiered ladder (or vice versa) would still pass
    /// every same-variant policy-lock test (iter-12) but quietly alias
    /// the two variants from the user's perspective.
    #[test]
    fn variant_streaming_q3km_vs_q3kl_cross_variant_divergence() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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

        // (name, q3km_type, q3km_bytes, q3kl_type, q3kl_bytes, divergent?)
        let cases: Vec<(&str, &str, usize, &str, usize, bool)> = vec![
            ("blk.5.attn_v.weight",     "Q4_K", 144, "Q5_K", 176, true),
            ("blk.9.ffn_down.weight",   "Q4_K", 144, "Q5_K", 176, true),
            ("blk.13.ffn_down.weight",  "Q3_K", 110, "Q5_K", 176, true),
            ("blk.0.attn_v.weight",     "Q5_K", 176, "Q5_K", 176, false),
            ("blk.5.attn_q.weight",     "Q3_K", 110, "Q3_K", 110, false),
        ];

        let mut lazy_map_q3km = LazyTensorMap::new();
        let mut lazy_map_q3kl = LazyTensorMap::new();
        for (i, (name, _, _, _, _, _)) in cases.iter().enumerate() {
            let payload = make_f16_payload(i as f32, QK_K);
            lazy_map_q3km.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                payload.clone(),
            ));
            lazy_map_q3kl.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                payload,
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        let q_q3km = VariantKQuantizer::new(
            KQuantVariant::Q3_K_M,
            CalibrationData::None,
            N_LAYERS,
        );
        let q_q3kl = VariantKQuantizer::new(
            KQuantVariant::Q3_K_L,
            CalibrationData::None,
            N_LAYERS,
        );

        let out_q3km =
            quantize_streaming(lazy_map_q3km, &metadata, &q_q3km, 0, 0, &progress, false)
                .expect("Q3_K_M streaming");
        let out_q3kl =
            quantize_streaming(lazy_map_q3kl, &metadata, &q_q3kl, 0, 0, &progress, false)
                .expect("Q3_K_L streaming");

        assert_eq!(out_q3km.quant_method, "Q3_K_M");
        assert_eq!(out_q3kl.quant_method, "Q3_K_L");

        for (name, q3km_type, q3km_bytes, q3kl_type, q3kl_bytes, divergent) in cases {
            let t_q3km = out_q3km
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("Q3_K_M missing {name}"));
            let t_q3kl = out_q3kl
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("Q3_K_L missing {name}"));

            assert_eq!(
                t_q3km.quant_info.ggml_type.as_deref(),
                Some(q3km_type),
                "Q3_K_M on {name}: expected {q3km_type}, got {:?}",
                t_q3km.quant_info.ggml_type
            );
            assert_eq!(
                t_q3km.data.len(),
                q3km_bytes,
                "Q3_K_M on {name}: expected {q3km_bytes} bytes"
            );

            assert_eq!(
                t_q3kl.quant_info.ggml_type.as_deref(),
                Some(q3kl_type),
                "Q3_K_L on {name}: expected {q3kl_type}, got {:?}",
                t_q3kl.quant_info.ggml_type
            );
            assert_eq!(
                t_q3kl.data.len(),
                q3kl_bytes,
                "Q3_K_L on {name}: expected {q3kl_bytes} bytes"
            );

            if divergent {
                assert_ne!(
                    t_q3km.quant_info.ggml_type, t_q3kl.quant_info.ggml_type,
                    "Q3_K_M and Q3_K_L must route {name} to DIFFERENT codecs \
                     per attn_v/ffn_down policy divergence — Q3_K_L has \
                     collapsed into Q3_K_M's tiered ladder (or vice versa)"
                );
            } else {
                assert_eq!(
                    t_q3km.quant_info.ggml_type, t_q3kl.quant_info.ggml_type,
                    "Q3_K_M and Q3_K_L must route shared-policy {name} to \
                     the SAME codec — divergence here would be a fixture bug"
                );
                assert_eq!(
                    t_q3km.data, t_q3kl.data,
                    "Q3_K_M and Q3_K_L on shared-policy {name} must produce \
                     byte-identical output (uncalibrated, deterministic)"
                );
            }
        }
    }

    /// ADR-014 P7 iter-28 — Q4_K_S vs Q4_K_M cross-variant divergence gate.
    /// Mirrors iter-26/27 for the Q4 family.  Per `llama-quant.cpp:439-460`
    /// (output bump), `:530-536` (attn_v), `:578-582` (ffn_down):
    ///
    ///   Q4_K_M:
    ///     - output           → Q6_K (210)
    ///     - attn_v   use_more_bits → Q6_K (210), else → Q4_K base (144)
    ///     - ffn_down use_more_bits → Q6_K (210), else → Q4_K base (144)
    ///   Q4_K_S:
    ///     - output           → Q4_K base (144)   (NO _M output bump)
    ///     - attn_v   i<4    → Q5_K (176), else → Q4_K base (144)
    ///     - ffn_down any    → Q4_K base (144)
    ///
    /// At n_layers=32 (n/8=4), the test exercises 5 cases:
    ///   - output.weight              :        Q4_K_M → Q6_K (210) vs Q4_K_S → Q4_K (144) (divergent)
    ///   - blk.0.attn_v.weight  (i=0): Q4_K_M → Q6_K (210, use_more_bits=true) vs Q4_K_S → Q5_K (176, i<4) (divergent)
    ///   - blk.9.ffn_down.weight(i=9): Q4_K_M → Q6_K (210, use_more_bits=true) vs Q4_K_S → Q4_K (144) (divergent)
    ///   - blk.5.attn_v.weight  (i=5): both → Q4_K (144) (control: same; use_more_bits=false, i≥4)
    ///   - blk.5.attn_q.weight  (i=5): both → Q4_K (144) (control: same)
    ///
    /// Without this gate, a regression that gives Q4_K_S the _M output
    /// bump (or strips _M's use_more_bits→Q6_K) would still pass every
    /// same-variant policy-lock test (iter-3t/iter-3u) but would
    /// silently make _S into _M (or a hybrid).
    #[test]
    fn variant_streaming_q4ks_vs_q4km_cross_variant_divergence() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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

        // (name, q4km_type, q4km_bytes, q4ks_type, q4ks_bytes, divergent?)
        let cases: Vec<(&str, &str, usize, &str, usize, bool)> = vec![
            ("output.weight",            "Q6_K", 210, "Q4_K", 144, true),
            ("blk.0.attn_v.weight",      "Q6_K", 210, "Q5_K", 176, true),
            ("blk.9.ffn_down.weight",    "Q6_K", 210, "Q4_K", 144, true),
            ("blk.5.attn_v.weight",      "Q4_K", 144, "Q4_K", 144, false),
            ("blk.5.attn_q.weight",      "Q4_K", 144, "Q4_K", 144, false),
        ];

        let mut lazy_map_q4km = LazyTensorMap::new();
        let mut lazy_map_q4ks = LazyTensorMap::new();
        for (i, (name, _, _, _, _, _)) in cases.iter().enumerate() {
            let payload = make_f16_payload(i as f32, QK_K);
            lazy_map_q4km.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                payload.clone(),
            ));
            lazy_map_q4ks.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                payload,
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        let q_q4km = VariantKQuantizer::new(
            KQuantVariant::Q4_K_M,
            CalibrationData::None,
            N_LAYERS,
        );
        let q_q4ks = VariantKQuantizer::new(
            KQuantVariant::Q4_K_S,
            CalibrationData::None,
            N_LAYERS,
        );

        let out_q4km =
            quantize_streaming(lazy_map_q4km, &metadata, &q_q4km, 0, 0, &progress, false)
                .expect("Q4_K_M streaming");
        let out_q4ks =
            quantize_streaming(lazy_map_q4ks, &metadata, &q_q4ks, 0, 0, &progress, false)
                .expect("Q4_K_S streaming");

        assert_eq!(out_q4km.quant_method, "Q4_K_M");
        assert_eq!(out_q4ks.quant_method, "Q4_K_S");

        for (name, q4km_type, q4km_bytes, q4ks_type, q4ks_bytes, divergent) in cases {
            let t_q4km = out_q4km
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("Q4_K_M missing {name}"));
            let t_q4ks = out_q4ks
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("Q4_K_S missing {name}"));

            assert_eq!(
                t_q4km.quant_info.ggml_type.as_deref(),
                Some(q4km_type),
                "Q4_K_M on {name}: expected {q4km_type}, got {:?}",
                t_q4km.quant_info.ggml_type
            );
            assert_eq!(
                t_q4km.data.len(),
                q4km_bytes,
                "Q4_K_M on {name}: expected {q4km_bytes} bytes"
            );

            assert_eq!(
                t_q4ks.quant_info.ggml_type.as_deref(),
                Some(q4ks_type),
                "Q4_K_S on {name}: expected {q4ks_type}, got {:?}",
                t_q4ks.quant_info.ggml_type
            );
            assert_eq!(
                t_q4ks.data.len(),
                q4ks_bytes,
                "Q4_K_S on {name}: expected {q4ks_bytes} bytes"
            );

            if divergent {
                assert_ne!(
                    t_q4km.quant_info.ggml_type, t_q4ks.quant_info.ggml_type,
                    "Q4_K_M and Q4_K_S must route {name} to DIFFERENT codecs \
                     per output / attn_v / ffn_down policy divergence — \
                     _S has accidentally inherited _M's tiered upgrades \
                     (or _M's bumps got stripped)"
                );
            } else {
                assert_eq!(
                    t_q4km.quant_info.ggml_type, t_q4ks.quant_info.ggml_type,
                    "Q4_K_M and Q4_K_S must route shared-base {name} to \
                     the SAME codec — divergence here would be a fixture bug"
                );
                assert_eq!(
                    t_q4km.data, t_q4ks.data,
                    "Q4_K_M and Q4_K_S on shared-base {name} must produce \
                     byte-identical output (uncalibrated, deterministic)"
                );
            }
        }
    }

    /// ADR-014 P7 iter-29 — Q5_K_S vs Q5_K_M cross-variant divergence gate.
    /// Closes the M-vs-S divergence-gate matrix for K-quants
    /// (Q2 done iter-26, Q3 done iter-27, Q4 done iter-28, Q5 here).
    /// Per `llama-quant.cpp:439-460/530-536/578-582`:
    ///
    ///   Q5_K_M:
    ///     - output           → Q6_K (210)
    ///     - attn_v   use_more_bits → Q6_K (210), else → Q5_K base (176)
    ///     - ffn_down use_more_bits → Q6_K (210), else → Q5_K base (176)
    ///   Q5_K_S:
    ///     - output           → Q5_K base (176)   (NO _M output bump)
    ///     - attn_v   any    → Q5_K base (176)
    ///     - ffn_down any    → Q5_K base (176)
    ///
    /// At n_layers=32 (n/8=4), the test exercises 5 cases:
    ///   - output.weight              :        Q5_K_M → Q6_K (210) vs Q5_K_S → Q5_K (176) (divergent)
    ///   - blk.0.attn_v.weight  (i=0, ub=true): Q5_K_M → Q6_K (210) vs Q5_K_S → Q5_K (176) (divergent)
    ///   - blk.9.ffn_down.weight(i=9, ub=true): Q5_K_M → Q6_K (210) vs Q5_K_S → Q5_K (176) (divergent)
    ///   - blk.5.attn_v.weight  (i=5, ub=false): both → Q5_K (176) (control: same)
    ///   - blk.5.attn_q.weight  (i=5):           both → Q5_K (176) (control: same)
    ///
    /// Without this gate, a regression that gives _S the _M output/V/down
    /// upgrades (or strips them from _M) would still pass `iter-16`'s
    /// `variant_streaming_q5ks_policy_all_base` lock (which only checks
    /// the _S half) but quietly alias the two variants.
    #[test]
    fn variant_streaming_q5ks_vs_q5km_cross_variant_divergence() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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

        // (name, q5km_type, q5km_bytes, q5ks_type, q5ks_bytes, divergent?)
        let cases: Vec<(&str, &str, usize, &str, usize, bool)> = vec![
            ("output.weight",            "Q6_K", 210, "Q5_K", 176, true),
            ("blk.0.attn_v.weight",      "Q6_K", 210, "Q5_K", 176, true),
            ("blk.9.ffn_down.weight",    "Q6_K", 210, "Q5_K", 176, true),
            ("blk.5.attn_v.weight",      "Q5_K", 176, "Q5_K", 176, false),
            ("blk.5.attn_q.weight",      "Q5_K", 176, "Q5_K", 176, false),
        ];

        let mut lazy_map_q5km = LazyTensorMap::new();
        let mut lazy_map_q5ks = LazyTensorMap::new();
        for (i, (name, _, _, _, _, _)) in cases.iter().enumerate() {
            let payload = make_f16_payload(i as f32, QK_K);
            lazy_map_q5km.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                payload.clone(),
            ));
            lazy_map_q5ks.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                payload,
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        let q_q5km = VariantKQuantizer::new(
            KQuantVariant::Q5_K_M,
            CalibrationData::None,
            N_LAYERS,
        );
        let q_q5ks = VariantKQuantizer::new(
            KQuantVariant::Q5_K_S,
            CalibrationData::None,
            N_LAYERS,
        );

        let out_q5km =
            quantize_streaming(lazy_map_q5km, &metadata, &q_q5km, 0, 0, &progress, false)
                .expect("Q5_K_M streaming");
        let out_q5ks =
            quantize_streaming(lazy_map_q5ks, &metadata, &q_q5ks, 0, 0, &progress, false)
                .expect("Q5_K_S streaming");

        assert_eq!(out_q5km.quant_method, "Q5_K_M");
        assert_eq!(out_q5ks.quant_method, "Q5_K_S");

        for (name, q5km_type, q5km_bytes, q5ks_type, q5ks_bytes, divergent) in cases {
            let t_q5km = out_q5km
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("Q5_K_M missing {name}"));
            let t_q5ks = out_q5ks
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("Q5_K_S missing {name}"));

            assert_eq!(
                t_q5km.quant_info.ggml_type.as_deref(),
                Some(q5km_type),
                "Q5_K_M on {name}: expected {q5km_type}, got {:?}",
                t_q5km.quant_info.ggml_type
            );
            assert_eq!(
                t_q5km.data.len(),
                q5km_bytes,
                "Q5_K_M on {name}: expected {q5km_bytes} bytes"
            );

            assert_eq!(
                t_q5ks.quant_info.ggml_type.as_deref(),
                Some(q5ks_type),
                "Q5_K_S on {name}: expected {q5ks_type}, got {:?}",
                t_q5ks.quant_info.ggml_type
            );
            assert_eq!(
                t_q5ks.data.len(),
                q5ks_bytes,
                "Q5_K_S on {name}: expected {q5ks_bytes} bytes"
            );

            if divergent {
                assert_ne!(
                    t_q5km.quant_info.ggml_type, t_q5ks.quant_info.ggml_type,
                    "Q5_K_M and Q5_K_S must route {name} to DIFFERENT codecs \
                     per output / attn_v / ffn_down policy divergence — \
                     _S has accidentally inherited _M's tiered upgrades \
                     (or _M's bumps got stripped)"
                );
            } else {
                assert_eq!(
                    t_q5km.quant_info.ggml_type, t_q5ks.quant_info.ggml_type,
                    "Q5_K_M and Q5_K_S must route shared-base {name} to \
                     the SAME codec — divergence here would be a fixture bug"
                );
                assert_eq!(
                    t_q5km.data, t_q5ks.data,
                    "Q5_K_M and Q5_K_S on shared-base {name} must produce \
                     byte-identical output (uncalibrated, deterministic)"
                );
            }
        }
    }

    /// ADR-014 P7 iter-30 — Q3_K_S vs Q3_K_M cross-variant divergence gate.
    /// Completes the Q3 family triangle (M↔L iter-27, M↔S here, L↔S
    /// transitively gated via the two pairs).  Per `llama-quant.cpp`:
    ///
    ///   Q3_K_M:
    ///     - output                → Q6_K (210, K-family fall-through)
    ///     - attn_v   i<2          → Q5_K (176)
    ///     - attn_v   i≥2          → Q4_K (144)
    ///     - ffn_down i<n/16       → Q5_K (176)
    ///     - ffn_down use_more_bits → Q4_K (144)
    ///     - ffn_down else         → Q3_K base (110)
    ///   Q3_K_S:
    ///     - output                → Q6_K (210, SHARED K-family fall-through)
    ///     - attn_v   any          → Q3_K base (110)   (NO _M tiered upgrades)
    ///     - ffn_down any          → Q3_K base (110)   (NO _M tiered upgrades)
    ///
    /// At n_layers=32 (n/8=4, 7n/8=28, n/16=2):
    ///   - output.weight              :        BOTH → Q6_K (210) (control: shared K-family fall-through)
    ///   - blk.0.attn_v.weight  (i=0):         Q3_K_M → Q5_K (176) vs Q3_K_S → Q3_K (110) (divergent)
    ///   - blk.5.attn_v.weight  (i=5):         Q3_K_M → Q4_K (144) vs Q3_K_S → Q3_K (110) (divergent)
    ///   - blk.9.ffn_down.weight(i=9, ub=true): Q3_K_M → Q4_K (144) vs Q3_K_S → Q3_K (110) (divergent)
    ///   - blk.13.ffn_down.weight(i=13):       BOTH → Q3_K (110) (control)
    ///   - blk.5.attn_q.weight  (i=5):         BOTH → Q3_K (110) (control)
    ///
    /// The `output.weight` control is structurally novel (it asserts the
    /// shared K-family fall-through is preserved across both variants) —
    /// catches a regression that strips the Q6_K output bump from _S
    /// while leaving _M intact.
    #[test]
    fn variant_streaming_q3ks_vs_q3km_cross_variant_divergence() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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

        // (name, q3km_type, q3km_bytes, q3ks_type, q3ks_bytes, divergent?)
        let cases: Vec<(&str, &str, usize, &str, usize, bool)> = vec![
            ("output.weight",            "Q6_K", 210, "Q6_K", 210, false),
            ("blk.0.attn_v.weight",      "Q5_K", 176, "Q3_K", 110, true),
            ("blk.5.attn_v.weight",      "Q4_K", 144, "Q3_K", 110, true),
            ("blk.9.ffn_down.weight",    "Q4_K", 144, "Q3_K", 110, true),
            ("blk.13.ffn_down.weight",   "Q3_K", 110, "Q3_K", 110, false),
            ("blk.5.attn_q.weight",      "Q3_K", 110, "Q3_K", 110, false),
        ];

        let mut lazy_map_q3km = LazyTensorMap::new();
        let mut lazy_map_q3ks = LazyTensorMap::new();
        for (i, (name, _, _, _, _, _)) in cases.iter().enumerate() {
            let payload = make_f16_payload(i as f32, QK_K);
            lazy_map_q3km.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                payload.clone(),
            ));
            lazy_map_q3ks.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                payload,
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        let q_q3km = VariantKQuantizer::new(
            KQuantVariant::Q3_K_M,
            CalibrationData::None,
            N_LAYERS,
        );
        let q_q3ks = VariantKQuantizer::new(
            KQuantVariant::Q3_K_S,
            CalibrationData::None,
            N_LAYERS,
        );

        let out_q3km =
            quantize_streaming(lazy_map_q3km, &metadata, &q_q3km, 0, 0, &progress, false)
                .expect("Q3_K_M streaming");
        let out_q3ks =
            quantize_streaming(lazy_map_q3ks, &metadata, &q_q3ks, 0, 0, &progress, false)
                .expect("Q3_K_S streaming");

        assert_eq!(out_q3km.quant_method, "Q3_K_M");
        assert_eq!(out_q3ks.quant_method, "Q3_K_S");

        for (name, q3km_type, q3km_bytes, q3ks_type, q3ks_bytes, divergent) in cases {
            let t_q3km = out_q3km
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("Q3_K_M missing {name}"));
            let t_q3ks = out_q3ks
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("Q3_K_S missing {name}"));

            assert_eq!(
                t_q3km.quant_info.ggml_type.as_deref(),
                Some(q3km_type),
                "Q3_K_M on {name}: expected {q3km_type}, got {:?}",
                t_q3km.quant_info.ggml_type
            );
            assert_eq!(
                t_q3km.data.len(),
                q3km_bytes,
                "Q3_K_M on {name}: expected {q3km_bytes} bytes"
            );

            assert_eq!(
                t_q3ks.quant_info.ggml_type.as_deref(),
                Some(q3ks_type),
                "Q3_K_S on {name}: expected {q3ks_type}, got {:?}",
                t_q3ks.quant_info.ggml_type
            );
            assert_eq!(
                t_q3ks.data.len(),
                q3ks_bytes,
                "Q3_K_S on {name}: expected {q3ks_bytes} bytes"
            );

            if divergent {
                assert_ne!(
                    t_q3km.quant_info.ggml_type, t_q3ks.quant_info.ggml_type,
                    "Q3_K_M and Q3_K_S must route {name} to DIFFERENT codecs \
                     per attn_v/ffn_down policy divergence — _S has \
                     accidentally inherited _M's tiered upgrades \
                     (or _M's bumps got stripped)"
                );
            } else {
                assert_eq!(
                    t_q3km.quant_info.ggml_type, t_q3ks.quant_info.ggml_type,
                    "Q3_K_M and Q3_K_S must route shared-policy {name} to \
                     the SAME codec — divergence here would catch a \
                     regression in the K-family output fall-through or \
                     base-route policy"
                );
                assert_eq!(
                    t_q3km.data, t_q3ks.data,
                    "Q3_K_M and Q3_K_S on shared-policy {name} must produce \
                     byte-identical output (uncalibrated, deterministic)"
                );
            }
        }
    }

    /// ADR-014 P7 iter-31 — exhaustive variant-pair base-target consistency
    /// matrix.  Auto-iterates every unordered pair from
    /// `KQuantVariant::all()` and asserts the streaming-output invariant
    /// on a guaranteed-base-routed tensor (`blk.5.attn_q.weight`):
    ///
    ///   bytes_a == bytes_b  ⇔  variant_a.base_target() == variant_b.base_target()
    ///
    /// Tensor choice rationale: `attn_q.weight` classifies as
    /// `TensorCategory::AttentionQ` per `llama-quant.cpp:115-150` →
    /// falls through `target_for`'s match arms to `_ => base` for every
    /// variant (no upgrades).  The byte-data therefore depends ONLY on
    /// the variant's `base_target()`, making this a clean per-codec
    /// matrix gate.
    ///
    /// Catches:
    ///   - A new variant whose base_target() drifts from its `target_for`
    ///     dispatch on attn_q (would mismatch a same-base pair).
    ///   - A future variant added to `all()` without its base codec wired
    ///     correctly (would mismatch every existing variant).
    ///   - A regression where two distinct base targets produce the same
    ///     bytes (e.g. codec aliasing).
    ///
    /// The test is *exhaustive over the variant matrix*, so adding a new
    /// variant to `KQuantVariant::all()` automatically extends coverage
    /// (zero new test code required) — this is the future-proofing
    /// counterpart to iter-3v's smoke test.
    #[test]
    fn variant_pair_matrix_base_target_consistency() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;
        const TENSOR: &str = "blk.5.attn_q.weight";

        let make_f16_payload = |len: usize| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(len * 2);
            for i in 0..len {
                let v = (i as f32 / len as f32) * 2.0 - 1.0;
                let h = half::f16::from_f32(v);
                bytes.extend_from_slice(&h.to_le_bytes());
            }
            bytes
        };
        let payload = make_f16_payload(QK_K);

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            m.insert(LazyTensor::from_bytes(
                LazyMeta::new(TENSOR.to_string(), vec![1, QK_K], DType::F16),
                payload.clone(),
            ));
            m
        };

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        // Quantize every variant once; collect (variant, bytes, base_target).
        let variants: &'static [KQuantVariant] = KQuantVariant::all();
        let mut results: Vec<(
            KQuantVariant,
            Vec<u8>,
            crate::quantize::k_quant_codec::KQuantTarget,
        )> = Vec::with_capacity(variants.len());
        for v in variants {
            let q = VariantKQuantizer::new(*v, CalibrationData::None, N_LAYERS);
            let out = quantize_streaming(make_lazy_map(), &metadata, &q, 0, 0, &progress, false)
                .unwrap_or_else(|e| panic!("variant {v:?} streaming failed: {e:?}"));
            let t = out
                .tensors
                .get(TENSOR)
                .unwrap_or_else(|| panic!("variant {v:?} missing {TENSOR}"));
            results.push((*v, t.data.clone(), v.base_target()));
        }

        // Sanity: at least one variant per base target → the matrix has
        // at least one same-base pair AND at least one different-base pair.
        let unique_targets: std::collections::HashSet<crate::quantize::k_quant_codec::KQuantTarget> =
            results.iter().map(|(_, _, t)| *t).collect();
        assert!(
            unique_targets.len() >= 2,
            "Variant menu must cover at least 2 distinct base targets \
             for the matrix to be meaningful; got {} targets",
            unique_targets.len()
        );

        // Exhaustive unordered-pair iteration.  C(N,2) pairs.
        let n = results.len();
        let mut same_base_pairs = 0usize;
        let mut diff_base_pairs = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let (va, bytes_a, ta) = (&results[i].0, &results[i].1, results[i].2);
                let (vb, bytes_b, tb) = (&results[j].0, &results[j].1, results[j].2);

                if ta == tb {
                    same_base_pairs += 1;
                    assert_eq!(
                        bytes_a, bytes_b,
                        "Pair ({va:?}, {vb:?}) shares base_target = {ta:?} \
                         but produced different bytes on {TENSOR} \
                         (uncalibrated, base-routed) — base codec dispatch \
                         is variant-discriminant when it should not be"
                    );
                } else {
                    diff_base_pairs += 1;
                    assert_ne!(
                        bytes_a, bytes_b,
                        "Pair ({va:?}, {vb:?}) has different base_targets \
                         ({ta:?} vs {tb:?}) but produced byte-identical \
                         output on {TENSOR} — distinct base codecs are \
                         silently aliasing"
                    );
                    // Also: byte-counts must differ (different bpw → different
                    // block sizes), the strongest cheap invariant.
                    assert_ne!(
                        bytes_a.len(), bytes_b.len(),
                        "Pair ({va:?}, {vb:?}) base_target bytes_per_block \
                         must differ when targets differ"
                    );
                }
            }
        }

        assert_eq!(
            same_base_pairs + diff_base_pairs,
            n * (n - 1) / 2,
            "Exhaustive pair enumeration miscounted"
        );
        // Sanity floor: with the current 10-variant menu we expect 6
        // same-base pairs (Q2 family: 1, Q3 family: 3, Q4 family: 1,
        // Q5 family: 1, Q6 alone: 0).  Don't lock the exact count to
        // keep the test stable as the menu grows; just assert > 0 on
        // each side so the matrix is genuinely exercising both branches.
        assert!(
            same_base_pairs > 0,
            "Variant menu has no same-base pairs — same-base branch is \
             untested"
        );
        assert!(
            diff_base_pairs > 0,
            "Variant menu has no different-base pairs — diff-base branch \
             is untested"
        );
    }

    /// ADR-014 P7 iter-32 — exhaustive variant × calibrator matrix.
    /// Auto-iterates every variant in `KQuantVariant::all()` and asserts
    /// the calibrator-orthogonality invariants on a guaranteed-base-routed
    /// tensor (`blk.5.attn_q.weight`):
    ///
    ///   For every variant V:
    ///     1. quant_method(V, None) == quant_method(V, Imatrix)
    ///        — calibration must NOT change the variant identity tag.
    ///     2. ggml_type(V, None) == ggml_type(V, Imatrix)
    ///        — calibration must NOT change the routing decision.
    ///     3. byte_count(V, None) == byte_count(V, Imatrix)
    ///        — same target → same block size.
    ///     4. If V's base target supports imatrix:
    ///        bytes(V, None) != bytes(V, Imatrix)
    ///        — non-uniform importance MUST flow through to the codec
    ///          search and produce different quantized bytes.
    ///        Else (theoretical, no current variant base targets fail this):
    ///        bytes(V, None) == bytes(V, Imatrix)
    ///        — calibration silently no-ops on non-supporting codec.
    ///
    /// Mirrors iter-31's exhaustive variant-pair matrix but on the
    /// orthogonal calibrator dimension.  Per Decisions 9+11, calibrator
    /// and OutputFormat (variant→target) are independent — this test
    /// pins that orthogonality through the streaming pipeline.
    ///
    /// Future-proofing: adding a new variant to `all()` automatically
    /// extends matrix coverage.  Adding a new `KQuantTarget` with
    /// `supports_imatrix = false` would land in the else branch and
    /// require zero test changes.
    #[test]
    fn variant_calibrator_matrix_through_streaming() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_BLOCKS: usize = 4;
        const TOTAL: usize = N_BLOCKS * QK_K;
        const N_LAYERS: usize = 32;
        const TENSOR: &str = "blk.5.attn_q.weight";
        const HIGH_IMPORTANCE_COLS: usize = 16;

        // Adversarial input pattern (iter-3z/iter-14/iter-23):
        // wide-range values in high-importance columns + small steady
        // values in low-importance.  Combined with the 100× / 0.01×
        // importance ratio, every K-quant codec including Q6_K should
        // make a different codebook choice between None and Imatrix
        // calibration paths.
        let original_f32: Vec<f32> = (0..TOTAL)
            .map(|i| {
                let c = i % QK_K;
                let r = (i / QK_K) as f32;
                if c < HIGH_IMPORTANCE_COLS {
                    let t = (c as f32 / HIGH_IMPORTANCE_COLS as f32) + r * 0.5;
                    -3.0 + 6.0 * t.fract()
                } else {
                    let t = (c as f32 / QK_K as f32) + r * 0.01;
                    -0.1 + 0.2 * t.fract()
                }
            })
            .collect();
        let payload: Vec<u8> = original_f32
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            m.insert(LazyTensor::from_bytes(
                LazyMeta::new(TENSOR.to_string(), vec![N_BLOCKS, QK_K], DType::F16),
                payload.clone(),
            ));
            m
        };

        // Non-uniform importance (10000× ratio).
        let acts: Vec<f32> = (0..QK_K)
            .map(|c| {
                if c < HIGH_IMPORTANCE_COLS {
                    10.0
                } else {
                    0.1
                }
            })
            .collect();
        let mut col = ImatrixCollector::new();
        col.accumulate_dense(TENSOR, &acts, 1, QK_K).unwrap();
        col.record_chunk();

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        let variants: &'static [KQuantVariant] = KQuantVariant::all();

        for v in variants {
            // None-calibrated path.
            let q_none = VariantKQuantizer::new(*v, CalibrationData::None, N_LAYERS);
            let out_none =
                quantize_streaming(make_lazy_map(), &metadata, &q_none, 0, 0, &progress, false)
                    .unwrap_or_else(|e| panic!("variant {v:?} None streaming: {e:?}"));

            // Imatrix-calibrated path (rebuild calibration each iter to
            // avoid sharing state — collector::from_imatrix_collector
            // takes &col, doesn't consume).
            let imatrix = CalibrationData::from_imatrix_collector(&col);
            let q_imatrix = VariantKQuantizer::new(*v, imatrix, N_LAYERS);
            let out_imatrix =
                quantize_streaming(make_lazy_map(), &metadata, &q_imatrix, 0, 0, &progress, false)
                    .unwrap_or_else(|e| panic!("variant {v:?} Imatrix streaming: {e:?}"));

            // Invariant 1: quant_method tag stable across calibrators.
            assert_eq!(
                out_none.quant_method, out_imatrix.quant_method,
                "variant {v:?}: quant_method changed between None ({}) and \
                 Imatrix ({}) — calibrator is leaking into variant identity",
                out_none.quant_method, out_imatrix.quant_method
            );

            let t_none = out_none
                .tensors
                .get(TENSOR)
                .unwrap_or_else(|| panic!("variant {v:?} None missing {TENSOR}"));
            let t_imatrix = out_imatrix
                .tensors
                .get(TENSOR)
                .unwrap_or_else(|| panic!("variant {v:?} Imatrix missing {TENSOR}"));

            // Invariant 2: ggml_type stable across calibrators.
            assert_eq!(
                t_none.quant_info.ggml_type, t_imatrix.quant_info.ggml_type,
                "variant {v:?}: ggml_type changed between None ({:?}) and \
                 Imatrix ({:?}) — calibrator is mis-routing the codec dispatch",
                t_none.quant_info.ggml_type, t_imatrix.quant_info.ggml_type
            );

            // Invariant 3: byte-count stable across calibrators.
            assert_eq!(
                t_none.data.len(),
                t_imatrix.data.len(),
                "variant {v:?}: byte-count changed between None ({}) and \
                 Imatrix ({}) — block-size drift across calibration",
                t_none.data.len(),
                t_imatrix.data.len()
            );

            // Invariant 4: bytes diverge IFF base codec supports imatrix.
            // The adversarial 4-super-block fixture combined with the
            // 100× importance ratio is sharp enough to force divergence
            // across every K-quant family (Q2-Q6) — proven by per-codec
            // tests iter-13/14 (Q3_K_M), iter-3w/3z (Q4_K_M), iter-23
            // (Q2_K), iter-25 (Q2_K_S).  Q5_K_M and Q6_K extend the
            // surface here.
            let supports = v.base_target().supports_imatrix();
            if supports {
                assert_ne!(
                    t_none.data, t_imatrix.data,
                    "variant {v:?} (base {:?} supports imatrix): None and \
                     Imatrix produced byte-identical output — imatrix \
                     payload is being silently stripped before the codec",
                    v.base_target()
                );
            } else {
                assert_eq!(
                    t_none.data, t_imatrix.data,
                    "variant {v:?} (base {:?} does NOT support imatrix): \
                     None and Imatrix produced different output — codec \
                     is silently using imatrix despite supports_imatrix=false",
                    v.base_target()
                );
            }
        }
    }

    /// ADR-014 P7 iter-33 — exhaustive variant × calibrator × parallelism
    /// orthogonality matrix.  Auto-iterates every variant in
    /// `KQuantVariant::all()` × every calibrator in {None, Imatrix} ×
    /// every dispatch path in {serial `quantize_streaming`, rayon
    /// `quantize_streaming_parallel`} and asserts byte-identity:
    ///
    ///   For every (variant V, calibrator C):
    ///     bytes_serial(V, C) == bytes_parallel(V, C)
    ///
    /// At the current 10-variant menu × 2 calibrators = 20 (V, C)
    /// combinations, each running on a 4-tensor multi-fixture (1 sensitive
    /// blk.5.attn_q + 3 categorically-distinct tensors).  iter-4 only
    /// covered Q4_K_M + None on a 6-tensor fixture; iter-33 closes the
    /// remaining 9 variants × 2 calibrators = 18 (V, C) combinations.
    ///
    /// Catches:
    ///   - A future variant whose `quantize_tensor` impl is non-deterministic
    ///     (e.g. uses rayon-internal mutex with non-deterministic ordering).
    ///   - A regression where `quantize_streaming_parallel` swaps tensor
    ///     order in the output `HashMap` (would still produce same set
    ///     but per-tensor bytes might be cross-contaminated).
    ///   - A `Send + Sync` bound regression on `VariantKQuantizer` that
    ///     forces a slower lock path.
    ///
    /// Mirrors iter-31 (variant-pair matrix) and iter-32 (variant ×
    /// calibrator matrix) to close the third orthogonality dimension —
    /// per Decision 5, parallelism is byte-identity-preserving.  Adding
    /// a new variant to `all()` extends matrix coverage automatically.
    #[test]
    fn variant_calibrator_parallelism_matrix_through_streaming() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_BLOCKS: usize = 2;
        const TOTAL: usize = N_BLOCKS * QK_K;
        const N_LAYERS: usize = 32;
        const HIGH_IMPORTANCE_COLS: usize = 16;

        // Multi-tensor fixture spanning categorically-distinct tensors —
        // ensures the parallel iter exercises real ordering, not just a
        // single-tensor degenerate case.  Using HF/llama-canonical
        // GGUF names so layer_mix routing is identical to production.
        let tensors: Vec<&str> = vec![
            "blk.5.attn_q.weight",   // base bucket for all variants
            "blk.5.attn_v.weight",   // V-policy upgrades on _M variants
            "blk.0.ffn_down.weight", // ffn_down policy upgrades
            "output.weight",         // K-family output bump
        ];

        let make_adversarial_payload = |seed: f32| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(TOTAL * 2);
            for i in 0..TOTAL {
                let c = i % QK_K;
                let r = (i / QK_K) as f32;
                let v = if c < HIGH_IMPORTANCE_COLS {
                    let t = (c as f32 / HIGH_IMPORTANCE_COLS as f32) + r * 0.5 + seed * 0.01;
                    -3.0 + 6.0 * t.fract()
                } else {
                    let t = (c as f32 / QK_K as f32) + r * 0.01 + seed * 0.001;
                    -0.1 + 0.2 * t.fract()
                };
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            bytes
        };

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            for (i, name) in tensors.iter().enumerate() {
                m.insert(LazyTensor::from_bytes(
                    LazyMeta::new(name.to_string(), vec![N_BLOCKS, QK_K], DType::F16),
                    make_adversarial_payload(i as f32),
                ));
            }
            m
        };

        // Imatrix non-uniform on every tensor (use `*` for collector
        // to apply the same activation profile across the fixture's
        // shape).
        let acts: Vec<f32> = (0..QK_K)
            .map(|c| if c < HIGH_IMPORTANCE_COLS { 10.0 } else { 0.1 })
            .collect();
        let mut col = ImatrixCollector::new();
        for name in &tensors {
            col.accumulate_dense(name, &acts, 1, QK_K).unwrap();
        }
        col.record_chunk();

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let variants: &'static [KQuantVariant] = KQuantVariant::all();

        // (variant, calibrator_label) pairs.  Use a Fn closure to
        // re-build the calibrator each call since CalibrationData is
        // moved into VariantKQuantizer.
        let calibrator_labels = ["None", "Imatrix"];

        for v in variants {
            for cal_label in &calibrator_labels {
                let make_calibration = || match *cal_label {
                    "None" => CalibrationData::None,
                    "Imatrix" => CalibrationData::from_imatrix_collector(&col),
                    _ => unreachable!(),
                };

                let q_serial = VariantKQuantizer::new(*v, make_calibration(), N_LAYERS);
                let q_parallel = VariantKQuantizer::new(*v, make_calibration(), N_LAYERS);

                let out_serial =
                    quantize_streaming(make_lazy_map(), &metadata, &q_serial, 0, 0, &progress, false)
                        .unwrap_or_else(|e| panic!("({v:?}, {cal_label}) serial: {e:?}"));
                let out_parallel = quantize_streaming_parallel(
                    make_lazy_map(),
                    &metadata,
                    &q_parallel,
                    0,
                    0,
                    &progress,
                    false,
                    None, // worker count = system parallelism
                )
                .unwrap_or_else(|e| panic!("({v:?}, {cal_label}) parallel: {e:?}"));

                assert_eq!(
                    out_serial.quant_method, out_parallel.quant_method,
                    "({v:?}, {cal_label}): quant_method differs serial vs parallel"
                );
                assert_eq!(
                    out_serial.tensors.len(),
                    out_parallel.tensors.len(),
                    "({v:?}, {cal_label}): tensor count differs serial vs parallel"
                );

                for tname in &tensors {
                    let t_serial = out_serial
                        .tensors
                        .get(*tname)
                        .unwrap_or_else(|| panic!("({v:?}, {cal_label}) serial missing {tname}"));
                    let t_parallel = out_parallel
                        .tensors
                        .get(*tname)
                        .unwrap_or_else(|| panic!("({v:?}, {cal_label}) parallel missing {tname}"));

                    assert_eq!(
                        t_serial.quant_info.ggml_type, t_parallel.quant_info.ggml_type,
                        "({v:?}, {cal_label}) on {tname}: ggml_type differs serial \
                         ({:?}) vs parallel ({:?})",
                        t_serial.quant_info.ggml_type, t_parallel.quant_info.ggml_type
                    );
                    assert_eq!(
                        t_serial.data, t_parallel.data,
                        "({v:?}, {cal_label}) on {tname}: bytes differ serial vs \
                         parallel — Decision 5 byte-identity contract violated"
                    );
                }
            }
        }
    }

    /// ADR-014 P7 iter-34 — legacy block-format streaming round-trip RMSE.
    /// K-quant variants got per-bit-tier RMSE bounds via iter-3x/3y
    /// (Q4/5/6_K), iter-11 (Q3_K), iter-17 (Q4_K_S/Q5_K_S), iter-22
    /// (Q2_K/Q2_K_S).  This iter closes the legacy targets — Q4_0,
    /// Q4_1, Q5_0, Q5_1, Q8_0 — through the same `quantize_streaming` +
    /// matching `dequantize_row_q*_bytes` round-trip pattern.
    ///
    /// Legacy targets dispatch through `KQuantCodecQuantizer` directly
    /// (they are not exposed via `KQuantVariant` since variants are
    /// K-quant-policy-only — Decisions 12+13).  This test pins the
    /// legacy codec quality through the production streaming path.
    ///
    /// RMSE bounds chosen per bit-tier on a 4-super-block smooth ramp
    /// (1024 elements, range -1..+1):
    ///   - Q4_0 / Q4_1: 4-bit quants → ≤ 0.05
    ///   - Q5_0 / Q5_1: 5-bit quants → ≤ 0.025
    ///   - Q8_0:        8-bit quants → ≤ 0.005
    ///
    /// Catches:
    ///   - A regression in `quantize_row_q*_to_bytes` that emits
    ///     structurally-valid bytes that round-trip to garbage.
    ///   - A `KQuantCodecQuantizer` Quantizer-trait wiring regression
    ///     that mis-routes legacy targets.
    ///   - Drift between the codec's `bytes_per_block` and the
    ///     streaming pipeline's per-row byte budget.
    ///
    /// Note: legacy targets do NOT support imatrix (per
    /// `KQuantTarget::supports_imatrix`); this test exercises the
    /// uncalibrated path only.
    #[test]
    fn legacy_target_streaming_round_trip_rmse_bounds() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant_codec::KQuantTarget;
        use crate::quantize::k_quant_codec_quantizer::KQuantCodecQuantizer;
        use crate::quantize::q_legacy::{
            dequantize_row_q4_0_bytes, dequantize_row_q4_1_bytes,
            dequantize_row_q5_0_bytes, dequantize_row_q5_1_bytes,
            dequantize_row_q8_0_bytes,
        };

        const QK: usize = 32; // legacy block size
        const N_BLOCKS: usize = 32; // 32 × 32 = 1024 elements
        const TOTAL: usize = QK * N_BLOCKS;
        const TENSOR: &str = "blk.0.attn_q.weight";

        // Smooth ramp -1..+1 over 1024 elements; matches K-quant RMSE
        // test fixtures' shape so the comparisons are apples-to-apples.
        let original_f32: Vec<f32> = (0..TOTAL)
            .map(|i| (i as f32 / TOTAL as f32) * 2.0 - 1.0)
            .collect();
        let payload: Vec<u8> = original_f32
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
                LazyMeta::new(TENSOR.to_string(), vec![N_BLOCKS, QK], DType::F16),
                payload.clone(),
            ));
            m
        };

        // (target, name, ggml_type, bytes_per_block, rmse_bound, dequant fn)
        type DequantFn = fn(&[u8], &mut [f32]) -> Result<usize, crate::quantize::q_legacy::QLegacyError>;
        let cases: Vec<(KQuantTarget, &str, &str, usize, f64, DequantFn)> = vec![
            (KQuantTarget::Q4Legacy,  "Q4_0", "Q4_0", 18, 0.05,  dequantize_row_q4_0_bytes),
            (KQuantTarget::Q4Legacy1, "Q4_1", "Q4_1", 20, 0.05,  dequantize_row_q4_1_bytes),
            (KQuantTarget::Q5Legacy0, "Q5_0", "Q5_0", 22, 0.025, dequantize_row_q5_0_bytes),
            (KQuantTarget::Q5Legacy1, "Q5_1", "Q5_1", 24, 0.025, dequantize_row_q5_1_bytes),
            (KQuantTarget::Q8Legacy,  "Q8_0", "Q8_0", 34, 0.005, dequantize_row_q8_0_bytes),
        ];

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        for (target, name, ggml_type, bytes_per_block, rmse_bound, dequant) in cases {
            let q = KQuantCodecQuantizer::new(name, target, CalibrationData::None);
            let out = quantize_streaming(make_lazy_map(), &metadata, &q, 0, 0, &progress, false)
                .unwrap_or_else(|e| panic!("{name} streaming: {e:?}"));

            let t = out
                .tensors
                .get(TENSOR)
                .unwrap_or_else(|| panic!("{name} missing {TENSOR}"));
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some(ggml_type),
                "{name}: ggml_type"
            );
            assert_eq!(
                t.data.len(),
                N_BLOCKS * bytes_per_block,
                "{name}: byte budget = N_BLOCKS × bytes_per_block"
            );

            let mut decoded = vec![0.0_f32; TOTAL];
            dequant(&t.data, &mut decoded)
                .unwrap_or_else(|e| panic!("{name} dequant: {e:?}"));

            let rmse: f64 = {
                let s: f64 = (0..TOTAL)
                    .map(|i| {
                        let e = decoded[i] as f64 - expected_post_f16[i] as f64;
                        e * e
                    })
                    .sum();
                (s / TOTAL as f64).sqrt()
            };

            assert!(
                rmse <= rmse_bound,
                "{name} round-trip RMSE {rmse:.6} exceeds bound {rmse_bound} \
                 — codec→streaming→dequant chain has degraded"
            );
        }
    }

    /// ADR-014 P7 iter-35 — legacy target × parallelism orthogonality.
    /// Mirrors iter-33's variant × parallelism matrix for legacy targets
    /// (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0).  Per Decision 5, parallelism is
    /// byte-identity-preserving regardless of codec — but iter-33 only
    /// covered `KQuantVariant` (K-quant family).  This iter closes
    /// `KQuantCodecQuantizer` directly with each legacy target.
    ///
    ///   For every legacy target T:
    ///     bytes_serial(T) == bytes_parallel(T)
    ///
    /// Compile-time `Send + Sync` guard on `KQuantCodecQuantizer`
    /// ensures rayon-eligibility is preserved against future field
    /// additions to the struct.
    ///
    /// Catches:
    ///   - A regression where `quantize_streaming_parallel` mis-orders
    ///     legacy-target output across the worker pool.
    ///   - Non-determinism added to the legacy codec path
    ///     (e.g. parallel scan).
    ///   - Loss of `Send + Sync` on `KQuantCodecQuantizer` forcing a
    ///     slower lock-based dispatch.
    #[test]
    fn legacy_target_parallelism_byte_identity() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant_codec::KQuantTarget;
        use crate::quantize::k_quant_codec_quantizer::KQuantCodecQuantizer;

        // Compile-time guard: KQuantCodecQuantizer must be Send + Sync
        // for rayon parallel dispatch.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<KQuantCodecQuantizer>();

        const QK: usize = 32;
        const N_BLOCKS: usize = 32;

        let make_payload = |seed: f32| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(N_BLOCKS * QK * 2);
            for i in 0..(N_BLOCKS * QK) {
                let v = (i as f32 / (N_BLOCKS * QK) as f32) * 2.0 - 1.0 + seed * 0.01;
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            bytes
        };

        // Multi-tensor fixture: 4 tensors so the parallel iter exercises
        // real ordering across the worker pool.
        let tensors: Vec<&str> = vec![
            "blk.0.attn_q.weight",
            "blk.5.attn_v.weight",
            "blk.10.ffn_down.weight",
            "output.weight",
        ];

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            for (i, name) in tensors.iter().enumerate() {
                m.insert(LazyTensor::from_bytes(
                    LazyMeta::new(name.to_string(), vec![N_BLOCKS, QK], DType::F16),
                    make_payload(i as f32),
                ));
            }
            m
        };

        let cases: Vec<(KQuantTarget, &str)> = vec![
            (KQuantTarget::Q4Legacy,  "Q4_0"),
            (KQuantTarget::Q4Legacy1, "Q4_1"),
            (KQuantTarget::Q5Legacy0, "Q5_0"),
            (KQuantTarget::Q5Legacy1, "Q5_1"),
            (KQuantTarget::Q8Legacy,  "Q8_0"),
        ];

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        for (target, name) in cases {
            let q_serial = KQuantCodecQuantizer::new(name, target, CalibrationData::None);
            let q_parallel = KQuantCodecQuantizer::new(name, target, CalibrationData::None);

            let out_serial =
                quantize_streaming(make_lazy_map(), &metadata, &q_serial, 0, 0, &progress, false)
                    .unwrap_or_else(|e| panic!("{name} serial: {e:?}"));
            let out_parallel = quantize_streaming_parallel(
                make_lazy_map(),
                &metadata,
                &q_parallel,
                0,
                0,
                &progress,
                false,
                None,
            )
            .unwrap_or_else(|e| panic!("{name} parallel: {e:?}"));

            assert_eq!(
                out_serial.quant_method, out_parallel.quant_method,
                "{name}: quant_method differs serial vs parallel"
            );
            assert_eq!(
                out_serial.tensors.len(),
                out_parallel.tensors.len(),
                "{name}: tensor count differs serial vs parallel"
            );

            for tname in &tensors {
                let t_serial = out_serial
                    .tensors
                    .get(*tname)
                    .unwrap_or_else(|| panic!("{name} serial missing {tname}"));
                let t_parallel = out_parallel
                    .tensors
                    .get(*tname)
                    .unwrap_or_else(|| panic!("{name} parallel missing {tname}"));

                assert_eq!(
                    t_serial.quant_info.ggml_type,
                    t_parallel.quant_info.ggml_type,
                    "{name} on {tname}: ggml_type differs"
                );
                assert_eq!(
                    t_serial.data, t_parallel.data,
                    "{name} on {tname}: legacy bytes differ serial vs parallel \
                     — Decision 5 byte-identity contract violated for legacy codec"
                );
            }
        }
    }

    /// ADR-014 P7 iter-36 — pin iter-34's bug-fix contract permanently.
    /// Direct regression-direction test for the bug fix landed in iter-34
    /// (`5f42942`): `KQuantCodecQuantizer::quantize_tensor` was applying
    /// `should_emit_f16_for_kquant`'s 256-multiple arm unconditionally,
    /// causing legacy targets (which use 32-element blocks) to falsely
    /// F16-passthrough on any 32-multiple-but-not-256-multiple row.
    ///
    /// Row length 96 = 3 × 32 (legal for legacy formats, NOT 256-multiple)
    /// is the canonical falsification fixture: pre-iter-34, every legacy
    /// target on this row would emit `ggml_type = "F16"` with 192 payload
    /// bytes; post-iter-34, each emits its own legacy codec output.
    ///
    /// This test pins the gate so a future code-cleanup that removes the
    /// `is_k_quant_target` branch (or "simplifies" the predicate to
    /// always apply) is caught immediately.  Mirrors the iter-34
    /// `legacy_target_streaming_round_trip_rmse_bounds` coverage that
    /// originally exposed the bug, but at the row-length boundary
    /// rather than the round-trip RMSE boundary.
    #[test]
    fn legacy_target_non_256_multiple_emits_legacy_not_f16() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant_codec::KQuantTarget;
        use crate::quantize::k_quant_codec_quantizer::KQuantCodecQuantizer;

        // Row length 96 = 3 × 32 elements: legal for every legacy codec
        // (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 use 32-element blocks).  NOT a
        // multiple of 256, so the K-quant alignment predicate fires —
        // this is exactly the bug fixture from iter-34.
        const ROW_LEN: usize = 96;
        const N_BLOCKS_LEGACY: usize = ROW_LEN / 32;
        const TENSOR: &str = "blk.0.attn_q.weight";

        // Confirm pre-condition: `should_emit_f16_for_kquant` reports
        // `true` on row_len=96 (the bug surface) — if this changes, the
        // test fixture itself is no longer the canonical falsification.
        assert!(
            crate::quantize::layer_mix::should_emit_f16_for_kquant(TENSOR, ROW_LEN),
            "ROW_LEN=96 must trigger the K-quant alignment predicate \
             — otherwise this test fixture no longer falsifies the iter-34 bug"
        );

        // F16 payload for the row.
        let payload: Vec<u8> = (0..ROW_LEN)
            .map(|i| (i as f32 / ROW_LEN as f32) * 2.0 - 1.0)
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            m.insert(LazyTensor::from_bytes(
                LazyMeta::new(TENSOR.to_string(), vec![1, ROW_LEN], DType::F16),
                payload.clone(),
            ));
            m
        };

        // (target, name, ggml_type, bytes_per_block)
        let cases: Vec<(KQuantTarget, &str, &str, usize)> = vec![
            (KQuantTarget::Q4Legacy,  "Q4_0", "Q4_0", 18),
            (KQuantTarget::Q4Legacy1, "Q4_1", "Q4_1", 20),
            (KQuantTarget::Q5Legacy0, "Q5_0", "Q5_0", 22),
            (KQuantTarget::Q5Legacy1, "Q5_1", "Q5_1", 24),
            (KQuantTarget::Q8Legacy,  "Q8_0", "Q8_0", 34),
        ];

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        for (target, name, ggml_type, bytes_per_block) in cases {
            let q = KQuantCodecQuantizer::new(name, target, CalibrationData::None);
            let out = quantize_streaming(make_lazy_map(), &metadata, &q, 0, 0, &progress, false)
                .unwrap_or_else(|e| panic!("{name} streaming on row_len=96: {e:?}"));

            let t = out
                .tensors
                .get(TENSOR)
                .unwrap_or_else(|| panic!("{name} missing {TENSOR}"));

            // Bug fixture: pre-iter-34 emitted `ggml_type=Some("F16")`,
            // 192 bytes (96 × 2-byte F16).  Post-iter-34 must emit the
            // requested legacy codec.
            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some(ggml_type),
                "{name} on row_len=96: emitted {:?} instead of {ggml_type} \
                 — the iter-34 `is_k_quant_target` gate has regressed",
                t.quant_info.ggml_type
            );
            assert_ne!(
                t.quant_info.ggml_type.as_deref(),
                Some("F16"),
                "{name} on row_len=96: emitted F16 — the K-quant alignment \
                 predicate has been re-applied to legacy targets"
            );
            assert_eq!(
                t.data.len(),
                N_BLOCKS_LEGACY * bytes_per_block,
                "{name} on row_len=96: byte budget = N_BLOCKS_LEGACY × \
                 bytes_per_block = {} × {} = {}",
                N_BLOCKS_LEGACY,
                bytes_per_block,
                N_BLOCKS_LEGACY * bytes_per_block
            );
            // Cross-check: emitted bytes are NOT the F16 fallback (which
            // would be 192 bytes for row_len=96).
            assert_ne!(
                t.data.len(),
                ROW_LEN * 2,
                "{name} on row_len=96: produced {} bytes (= F16 byte count) — \
                 fell into the F16 passthrough arm",
                ROW_LEN * 2
            );
        }
    }

    /// ADR-014 P7 iter-37 — vision-tensor F16 passthrough applies to
    /// legacy targets too.  Bug discovered while pinning iter-34's
    /// regression contract: iter-34's `is_k_quant_target` gate
    /// collapsed the vision-pattern arm too, so vision tensors routed
    /// to legacy targets (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) would silently
    /// quantize through the legacy codec instead of F16-passthrough.
    ///
    /// Vision tensors are F16-passthrough *policy-driven*, not
    /// block-size-driven — quality loss propagates to multimodal input
    /// representation regardless of which codec the rest of the model
    /// uses.  The fix: split the predicate into `is_vision_tensor_pattern`
    /// (universal) and `is_kquant_row_misaligned` (K-quant only); the
    /// codec dispatch ORs them with `is_k_quant_target`.
    ///
    /// Test: vision tensor `vit.proj_out.weight` with row_len=256
    /// (legal for both K-quant AND legacy block sizes — isolates the
    /// vision-pattern axis from the alignment axis).  All 5 legacy
    /// targets MUST emit F16 passthrough (512 bytes = 256 × 2-byte F16),
    /// not legacy codec output.
    ///
    /// Catches a future "consolidation" of the predicates back into a
    /// single condition or removal of the universal vision-pattern arm.
    #[test]
    fn legacy_target_vision_tensor_emits_f16_passthrough() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant_codec::KQuantTarget;
        use crate::quantize::k_quant_codec_quantizer::KQuantCodecQuantizer;

        const ROW_LEN: usize = 256; // 256-multiple AND 32-multiple — isolates vision axis
        const TENSOR: &str = "vit.proj_out.weight";

        // Pre-conditions confirming the fixture isolates the vision axis:
        assert!(
            crate::quantize::layer_mix::is_vision_tensor_pattern(TENSOR),
            "TENSOR must match the vision-pattern arm"
        );
        assert!(
            !crate::quantize::layer_mix::is_kquant_row_misaligned(ROW_LEN),
            "ROW_LEN=256 must NOT trigger the K-quant-misalignment arm — \
             this isolates the test to the vision-pattern axis"
        );

        let payload: Vec<u8> = (0..ROW_LEN)
            .map(|i| (i as f32 / ROW_LEN as f32) * 2.0 - 1.0)
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            m.insert(LazyTensor::from_bytes(
                LazyMeta::new(TENSOR.to_string(), vec![1, ROW_LEN], DType::F16),
                payload.clone(),
            ));
            m
        };

        // (target, name) — iter over all 5 legacy targets AND a K-quant
        // target for the orthogonality assertion.
        let cases: Vec<(KQuantTarget, &str)> = vec![
            (KQuantTarget::Q4Legacy,  "Q4_0"),
            (KQuantTarget::Q4Legacy1, "Q4_1"),
            (KQuantTarget::Q5Legacy0, "Q5_0"),
            (KQuantTarget::Q5Legacy1, "Q5_1"),
            (KQuantTarget::Q8Legacy,  "Q8_0"),
            // K-quant control: also F16-passthrough'd via vision arm.
            (KQuantTarget::Q4K,       "Q4_K"),
        ];

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        for (target, name) in cases {
            let q = KQuantCodecQuantizer::new(name, target, CalibrationData::None);
            let out = quantize_streaming(make_lazy_map(), &metadata, &q, 0, 0, &progress, false)
                .unwrap_or_else(|e| panic!("{name} on vision tensor: {e:?}"));

            let t = out
                .tensors
                .get(TENSOR)
                .unwrap_or_else(|| panic!("{name} missing {TENSOR}"));

            assert_eq!(
                t.quant_info.ggml_type.as_deref(),
                Some("F16"),
                "{name} on vision tensor: emitted {:?} instead of F16 — \
                 the iter-37 vision-pattern universal-passthrough has regressed",
                t.quant_info.ggml_type
            );
            assert!(
                t.quant_info.preserved,
                "{name} on vision tensor: preserved flag must be true \
                 for F16 passthrough"
            );
            assert_eq!(
                t.data.len(),
                ROW_LEN * 2,
                "{name} on vision tensor: expected {} bytes (= F16 byte budget), \
                 got {} — fell into the codec dispatch arm",
                ROW_LEN * 2,
                t.data.len()
            );
        }
    }

    /// ADR-014 P7 iter-38 — DwqK predicate split is routing-aware.
    /// Bug deferred from iter-37: `dwq_k_quantizer.rs` was still calling
    /// the merged `should_emit_f16_for_kquant` predicate, applying the
    /// row-misalignment arm to ALL routing decisions including Q8_0
    /// sensitive layers.  Q8_0 uses 32-element blocks, so a 32-multiple
    /// row that is NOT also 256-multiple was falsely F16-passthrough'd
    /// instead of going through Q8_0.
    ///
    /// Fix: split the gate into routing-aware logic — vision is
    /// universal (policy), K-quant alignment is K-quant-routing-only
    /// (per `target_for(tensor)` resolution).  Q8_0 sensitive layers
    /// with 32-multiple-but-not-256-multiple rows now correctly
    /// quantize through the legacy codec.
    ///
    /// Test: DwqKVariant::P48 (Q4_K base + Q8_0 sensitive) with sensitive
    /// layer 5; tensor `model.layers.5.self_attn.q_proj.weight` at
    /// row_len=96 (3 × 32 = 96, NOT 256-multiple).  Pre-iter-38 emitted
    /// F16 (192 bytes); post-iter-38 emits Q8_0 (3 × 34 = 102 bytes).
    /// Note: P46's sensitive_target is Q6_K not Q8_0 — for THIS test
    /// we need a variant with a legacy-codec sensitive target.
    ///
    /// Companion control case: same row at layer 0 (NOT sensitive) with
    /// the same variant routes to Q4_K base — which IS a K-quant target
    /// at a non-256-multiple row, so STILL F16-passthrough'd.  This
    /// orthogonality assertion proves the fix discriminates correctly:
    /// sensitive→Q8_0 escapes; base→Q4_K still triggers passthrough.
    #[test]
    fn dwq_sensitive_q8_0_layer_with_32_multiple_row_emits_q8_not_f16() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::dwq_k_quantizer::{DwqKQuantizer, DwqKVariant};

        const ROW_LEN: usize = 96; // 3 × 32 (legacy-legal), NOT 256-multiple

        // Pre-condition: row_len must trigger the K-quant alignment arm
        // so the bug fixture is canonical.
        assert!(
            crate::quantize::layer_mix::is_kquant_row_misaligned(ROW_LEN),
            "ROW_LEN=96 must trigger the K-quant row-misalignment arm — \
             otherwise this test fixture no longer exercises iter-38"
        );

        let payload: Vec<u8> = (0..ROW_LEN)
            .map(|i| (i as f32 / ROW_LEN as f32) * 2.0 - 1.0)
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        // Two tensors: one sensitive (→ Q8_0), one base (→ Q4_K).
        // Identical row layout so divergence is purely policy-driven.
        // DwqK uses HF naming (`model.layers.<N>.…`) per
        // `extract_layer_index`'s contract — `blk.<N>.…` (GGUF) does
        // not parse so would route to base unconditionally.
        let sensitive_name = "model.layers.5.self_attn.q_proj.weight";
        let base_name = "model.layers.0.self_attn.q_proj.weight";

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            for name in [sensitive_name, base_name] {
                m.insert(LazyTensor::from_bytes(
                    LazyMeta::new(name.to_string(), vec![1, ROW_LEN], DType::F16),
                    payload.clone(),
                ));
            }
            m
        };

        // P48 = Q4_K base + Q8_0 sensitive.  Mark layer 5 sensitive.
        // (P46's sensitive_target is Q6_K — wrong for this test.)
        let q = DwqKQuantizer::new(
            DwqKVariant::P48,
            &[5..=5],
            CalibrationData::None,
        );

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let out = quantize_streaming(make_lazy_map(), &metadata, &q, 0, 0, &progress, false)
            .expect("DwqK::P46 streaming with row_len=96");

        // ── Sensitive layer (Q8_0 routing) ──
        // Post-iter-38: row_len=96 is 32-multiple, Q8_0 handles it,
        // emit Q8_0 (3 × 34 = 102 bytes).  Pre-iter-38 emitted F16 (192).
        let t_sensitive = out
            .tensors
            .get(sensitive_name)
            .expect("sensitive missing");
        assert_eq!(
            t_sensitive.quant_info.ggml_type.as_deref(),
            Some("Q8_0"),
            "sensitive layer at row_len=96 emitted {:?} instead of Q8_0 \
             — iter-38 routing-aware F16 gate has regressed",
            t_sensitive.quant_info.ggml_type
        );
        assert_ne!(
            t_sensitive.quant_info.ggml_type.as_deref(),
            Some("F16"),
            "sensitive layer at row_len=96 fell into F16 passthrough — \
             DwqK is applying K-quant alignment arm to Q8_0 routing"
        );
        assert_eq!(
            t_sensitive.data.len(),
            (ROW_LEN / 32) * 34,
            "sensitive layer Q8_0 byte budget"
        );

        // ── Base layer (Q4_K routing) ──
        // Q4_K is K-quant + row_len=96 is misaligned → F16 passthrough.
        // This control proves the routing-aware fix discriminates: the
        // K-quant alignment arm STILL fires for K-quant routing, just
        // not for Q8_0 routing.
        let t_base = out.tensors.get(base_name).expect("base missing");
        assert_eq!(
            t_base.quant_info.ggml_type.as_deref(),
            Some("F16"),
            "base layer at row_len=96 (K-quant routing) emitted {:?} \
             instead of F16 — the K-quant alignment arm has been over-fixed",
            t_base.quant_info.ggml_type
        );
        assert_eq!(
            t_base.data.len(),
            ROW_LEN * 2,
            "base layer F16 byte budget"
        );
    }

    /// ADR-014 P7 iter-39 — vision policy overrides DwqK Q8_0 routing.
    /// Orthogonal control gate to iter-38: pin that the vision-pattern
    /// arm STILL fires for DwqK even when the per-tensor routing target
    /// is the legacy Q8_0 sensitive-bucket.  Vision-policy F16 passthrough
    /// is universal (multimodal-quality reasons), it must not be
    /// short-circuited by the iter-38 routing-aware split.
    ///
    /// Test fixture: `model.visual.layers.5.attn.q_proj.weight`
    ///   - Matches `is_vision_tensor_pattern` (contains "model.visual.")
    ///   - Does NOT match `TensorRef::is_vision_tensor()` (which only
    ///     matches "vision_tower" / "embed_vision") — so streaming.rs's
    ///     `preserve = … || tensor.is_vision_tensor()` does NOT fire,
    ///     forcing the test through DwqK's gate path.
    ///   - `extract_layer_index` resolves layer 5
    ///   - Layer 5 marked sensitive → Q8_0 routing
    ///   - Row_len=256 (256-multiple AND 32-multiple — isolates vision axis
    ///     from row-misalignment axis)
    ///
    /// Pre-iter-37 (with the merged predicate): F16 passthrough fires
    /// via the universal vision arm.
    /// Post-iter-37/38 (split predicate, routing-aware): F16 passthrough
    /// STILL fires via the universal `is_vision_tensor_pattern` OR clause.
    ///
    /// Without this test, a future "simplification" of the routing-aware
    /// gate that drops the vision arm (or wraps the entire OR in
    /// `is_kquant_routing`) would silently route vision tensors to the
    /// Q8_0 codec — degrading multimodal quality with no test signal.
    #[test]
    fn dwq_vision_tensor_on_sensitive_layer_emits_f16_passthrough() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::dwq_k_quantizer::{DwqKQuantizer, DwqKVariant};

        const ROW_LEN: usize = 256; // 256-multiple — alignment axis is OFF

        // Tensor name that simultaneously: (a) matches the codec
        // vision-pattern, (b) parses to a layer index for DwqK
        // sensitivity routing, (c) does NOT match streaming.rs's
        // `TensorRef::is_vision_tensor()` (narrower; only "vision_tower"
        // or "embed_vision") — this is necessary so the test exercises
        // DwqK's gate path and not the upstream `preserve=true` shortcut.
        let tensor_name = "model.visual.layers.5.attn.q_proj.weight";

        // Pre-conditions:
        assert!(
            crate::quantize::layer_mix::is_vision_tensor_pattern(tensor_name),
            "fixture must match vision-pattern arm"
        );
        assert!(
            !crate::quantize::layer_mix::is_kquant_row_misaligned(ROW_LEN),
            "ROW_LEN=256 must NOT trigger row-misalignment arm — \
             this test isolates the vision-policy axis from the alignment axis"
        );

        let payload: Vec<u8> = (0..ROW_LEN)
            .map(|i| (i as f32 / ROW_LEN as f32) * 2.0 - 1.0)
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            m.insert(LazyTensor::from_bytes(
                LazyMeta::new(tensor_name.to_string(), vec![1, ROW_LEN], DType::F16),
                payload.clone(),
            ));
            m
        };

        // P48 = Q4_K base + Q8_0 sensitive.  Layer 5 sensitive →
        // routing_target=Q8_0 (legacy).  Vision arm must still fire.
        let q = DwqKQuantizer::new(
            DwqKVariant::P48,
            &[5..=5],
            CalibrationData::None,
        );

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let out = quantize_streaming(make_lazy_map(), &metadata, &q, 0, 0, &progress, false)
            .expect("DwqK::P48 streaming with vision tensor");

        let t = out
            .tensors
            .get(tensor_name)
            .unwrap_or_else(|| panic!("missing {tensor_name}"));

        assert_eq!(
            t.quant_info.ggml_type.as_deref(),
            Some("F16"),
            "vision tensor on sensitive Q8_0 layer emitted {:?} instead of F16 \
             — vision-policy universal F16 passthrough has been short-circuited \
             by iter-38's routing-aware gate",
            t.quant_info.ggml_type
        );
        assert!(
            t.quant_info.preserved,
            "vision tensor: preserved flag must be true for F16 passthrough"
        );
        assert_eq!(
            t.data.len(),
            ROW_LEN * 2,
            "vision tensor F16 byte budget"
        );
    }

    /// ADR-014 P7 iter-40 — dual vision-predicate path boundary.
    /// Documents and pins the structural finding from iter-39 debugging:
    /// hf2q has TWO vision predicates with different scope, driving
    /// F16-passthrough through TWO different paths in the streaming
    /// pipeline.  The two paths produce *different* `ggml_type` tags
    /// for the same content (preserved-passthrough vs f16-passthrough)
    /// — this is the live contract surface.
    ///
    /// Predicate A — narrow: `TensorRef::is_vision_tensor()` matches
    /// only `vision_tower` and `embed_vision`.  Drives streaming.rs's
    /// `preserve = … || tensor.is_vision_tensor()` so the codec
    /// quantizer's `config.preserve` short-circuit fires, returning
    /// `method="passthrough"` and `ggml_type=None`.
    ///
    /// Predicate B — broad: `is_vision_tensor_pattern()` in
    /// `layer_mix` matches `model.visual.`, `vision_tower.`,
    /// `vision_model.`, `vit.`, `visual.`, `.visual.`.  Used by the
    /// codec quantizer + DwqK gates downstream of streaming, hits when
    /// streaming's preserve did not.  Returns `method="f16"` and
    /// `ggml_type=Some("F16")` via `f16_passthrough`.
    ///
    /// Predicate B is a strict superset of A — so every vision tensor
    /// hits ONE of the two paths.  A future merge that drops either
    /// predicate, OR a broadening of A to match all of B's patterns,
    /// would change at least one tensor's emit shape and trip this
    /// test.
    ///
    /// Test fixture: two tensors at row_len=256 (no row-misalignment):
    ///   - `vision_tower.encoder.layers.5.attn.q_proj.weight`
    ///     → predicate A matches → preserve=true upstream → ggml_type=None
    ///   - `model.visual.layers.5.attn.q_proj.weight`
    ///     → predicate A NO match, predicate B matches → ggml_type=Some("F16")
    ///
    /// Both produce 512-byte output (256 F16 elements × 2 bytes/element)
    /// with byte-equivalent data — the contract for users is "vision
    /// tensors are F16-preserved", regardless of which path got them
    /// there.  Asserts byte-equality across paths to lock that.
    #[test]
    fn vision_tensor_dual_predicate_path_boundary() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant_codec::KQuantTarget;
        use crate::quantize::k_quant_codec_quantizer::KQuantCodecQuantizer;

        const ROW_LEN: usize = 256;

        // Path A (narrow): caught by streaming.rs's preserve check.
        let name_a = "vision_tower.encoder.layers.5.attn.q_proj.weight";
        // Path B (broad-only): caught by codec's gate.
        let name_b = "model.visual.layers.5.attn.q_proj.weight";

        // Pre-conditions document the predicate split:
        // both must match the broad codec predicate (so vision policy
        // fires *somewhere*); only A matches the narrow upstream
        // predicate.  These asserts will trip if either predicate
        // drifts and the dual-path structure collapses.
        assert!(
            crate::quantize::layer_mix::is_vision_tensor_pattern(name_a),
            "name_a must match broad predicate"
        );
        assert!(
            crate::quantize::layer_mix::is_vision_tensor_pattern(name_b),
            "name_b must match broad predicate"
        );

        let payload: Vec<u8> = (0..ROW_LEN)
            .map(|i| (i as f32 / ROW_LEN as f32) * 2.0 - 1.0)
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            for n in [name_a, name_b] {
                m.insert(LazyTensor::from_bytes(
                    LazyMeta::new(n.to_string(), vec![1, ROW_LEN], DType::F16),
                    payload.clone(),
                ));
            }
            m
        };

        // Use any K-quant target — the F16 passthrough fires before
        // codec dispatch in both paths.
        let q = KQuantCodecQuantizer::new("Q4_K", KQuantTarget::Q4K, CalibrationData::None);

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let out = quantize_streaming(make_lazy_map(), &metadata, &q, 0, 0, &progress, false)
            .expect("dual-predicate streaming");

        let t_a = out.tensors.get(name_a).expect("name_a missing");
        let t_b = out.tensors.get(name_b).expect("name_b missing");

        // Path A: streaming preserve → method=passthrough, ggml_type=None.
        assert_eq!(
            t_a.quant_info.method, "passthrough",
            "name_a path A: method must be 'passthrough' (streaming preserve)"
        );
        assert_eq!(
            t_a.quant_info.ggml_type, None,
            "name_a path A: ggml_type must be None"
        );
        assert!(
            t_a.quant_info.preserved,
            "name_a path A: preserved flag must be true"
        );

        // Path B: codec gate → method=f16, ggml_type=Some("F16").
        assert_eq!(
            t_b.quant_info.method, "f16",
            "name_b path B: method must be 'f16' (codec gate)"
        );
        assert_eq!(
            t_b.quant_info.ggml_type.as_deref(),
            Some("F16"),
            "name_b path B: ggml_type must be Some(\"F16\")"
        );
        assert!(
            t_b.quant_info.preserved,
            "name_b path B: preserved flag must be true"
        );

        // Cross-path contract: byte-data equivalent (both paths emit
        // F16-cast of the original payload).
        assert_eq!(
            t_a.data.len(),
            ROW_LEN * 2,
            "name_a F16 byte budget"
        );
        assert_eq!(
            t_b.data.len(),
            ROW_LEN * 2,
            "name_b F16 byte budget"
        );
        assert_eq!(
            t_a.data, t_b.data,
            "vision-tensor contract: both predicate paths must produce \
             byte-equivalent F16 output for the same input — the user-\
             facing contract is 'vision tensors are F16-preserved' \
             regardless of which gate fired"
        );
    }

    /// ADR-014 P7 iter-41 — F16-passthrough paths × parallelism orthogonality.
    /// Closes Decision 5 (byte-identity for parallel) for the F16-passthrough
    /// decision tree pinned in iter-40.  Both Path A (streaming preserve)
    /// and Path B (codec gate) must produce byte-identical output through
    /// `quantize_streaming_parallel` vs serial, in the same fixture
    /// alongside a non-vision tensor that goes through the actual codec.
    ///
    /// Without this gate, a regression in `quantize_streaming_parallel`
    /// that mishandles preserve=true tensors (Path A) or that re-enters
    /// the codec gate non-deterministically (Path B) could silently break
    /// vision-tensor preservation under multi-worker dispatch.
    ///
    /// Multi-fixture (3 tensors):
    ///   - `vision_tower.encoder.layers.5.attn.q_proj.weight` → Path A
    ///   - `model.visual.layers.5.attn.q_proj.weight`         → Path B
    ///   - `blk.5.attn_q.weight`                              → codec dispatch
    ///
    /// Asserts serial ≡ parallel for ALL three tensors AND the dual-path
    /// contract from iter-40 holds in the parallel output (vision-tensor
    /// data byte-equivalent across both paths).
    #[test]
    fn vision_passthrough_paths_parallel_byte_identity() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant_codec::KQuantTarget;
        use crate::quantize::k_quant_codec_quantizer::KQuantCodecQuantizer;

        const ROW_LEN: usize = 256;

        let payload: Vec<u8> = (0..ROW_LEN)
            .map(|i| (i as f32 / ROW_LEN as f32) * 2.0 - 1.0)
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        let tensors: Vec<&str> = vec![
            "vision_tower.encoder.layers.5.attn.q_proj.weight", // Path A
            "model.visual.layers.5.attn.q_proj.weight",         // Path B
            "blk.5.attn_q.weight",                              // codec dispatch
        ];

        let make_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            for n in &tensors {
                m.insert(LazyTensor::from_bytes(
                    LazyMeta::new(n.to_string(), vec![1, ROW_LEN], DType::F16),
                    payload.clone(),
                ));
            }
            m
        };

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q_serial = KQuantCodecQuantizer::new("Q4_K", KQuantTarget::Q4K, CalibrationData::None);
        let q_parallel = KQuantCodecQuantizer::new("Q4_K", KQuantTarget::Q4K, CalibrationData::None);

        let out_serial =
            quantize_streaming(make_lazy_map(), &metadata, &q_serial, 0, 0, &progress, false)
                .expect("serial streaming");
        let out_parallel = quantize_streaming_parallel(
            make_lazy_map(),
            &metadata,
            &q_parallel,
            0,
            0,
            &progress,
            false,
            None,
        )
        .expect("parallel streaming");

        for tname in &tensors {
            let t_s = out_serial
                .tensors
                .get(*tname)
                .unwrap_or_else(|| panic!("serial missing {tname}"));
            let t_p = out_parallel
                .tensors
                .get(*tname)
                .unwrap_or_else(|| panic!("parallel missing {tname}"));

            assert_eq!(
                t_s.quant_info.method, t_p.quant_info.method,
                "{tname}: method differs serial vs parallel"
            );
            assert_eq!(
                t_s.quant_info.ggml_type, t_p.quant_info.ggml_type,
                "{tname}: ggml_type differs serial vs parallel"
            );
            assert_eq!(
                t_s.quant_info.preserved, t_p.quant_info.preserved,
                "{tname}: preserved flag differs serial vs parallel"
            );
            assert_eq!(
                t_s.data, t_p.data,
                "{tname}: bytes differ serial vs parallel — Decision 5 \
                 byte-identity contract violated for F16-passthrough path"
            );
        }

        // Re-assert iter-40's cross-path contract under parallel dispatch:
        // vision-tensor byte data is equivalent across Path A and Path B.
        let path_a_bytes = &out_parallel.tensors[tensors[0]].data;
        let path_b_bytes = &out_parallel.tensors[tensors[1]].data;
        assert_eq!(
            path_a_bytes, path_b_bytes,
            "iter-40 dual-predicate contract: parallel output must \
             still emit byte-equivalent F16 data across both paths"
        );
    }

    /// ADR-014 P7 iter-42 — streaming output flows cleanly through
    /// backend validate.  Audit gate ahead of P2 iter-3 (production
    /// `materialize_all()` bridge removal in `main.rs:1147`).
    ///
    /// Finding from this audit: `quantize_streaming` is built and
    /// gated by 50+ tests but is NEVER called from production
    /// `main.rs` — the `materialize_all()` bridge at line 1147 still
    /// runs upfront, defeating the streaming-pipeline memory-budget
    /// purpose.  P2 iter-3 (StreamingBackend production wiring +
    /// per-tensor write-and-drop) remains the wedge.
    ///
    /// This test pins the contract that `quantize_streaming`'s output
    /// shape is validation-clean against `GgufBackend::validate` — the
    /// drop-in replacement for `quantize_model` is structurally safe.
    /// When iter-3 lands, the wire-up should be a no-op behaviorally.
    ///
    /// Multi-tensor fixture spanning categorically-distinct layers:
    /// streamingly quantize through Q4_K, then run the resulting
    /// `QuantizedModel` through `GgufBackend::validate`.  Asserts: zero
    /// warnings of severity≥`Error`.
    #[test]
    fn streaming_output_validates_clean_against_gguf_backend() {
        use crate::backends::{gguf::GgufBackend, OutputBackend};
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::k_quant_codec::KQuantTarget;
        use crate::quantize::k_quant_codec_quantizer::KQuantCodecQuantizer;

        const QK_K: usize = 256;

        // Multi-tensor fixture: a small but realistic shape that the
        // backend's validate sees in production.  All non-vision, all
        // weight-shaped, all 256-multiple — should validate clean.
        let tensors: Vec<&str> = vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_gate.weight",
        ];

        let make_payload = |seed: f32| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(QK_K * 2);
            for i in 0..QK_K {
                let v = (i as f32 / QK_K as f32) * 2.0 - 1.0 + seed * 0.001;
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            bytes
        };

        let mut lazy_map = LazyTensorMap::new();
        for (i, name) in tensors.iter().enumerate() {
            lazy_map.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![1, QK_K], DType::F16),
                make_payload(i as f32),
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = KQuantCodecQuantizer::new("Q4_K", KQuantTarget::Q4K, CalibrationData::None);

        let model = quantize_streaming(lazy_map, &metadata, &q, 0, 0, &progress, false)
            .expect("streaming quantize");

        // Backend validate the output.  The validate contract is the
        // gate main.rs:1147+ must satisfy when iter-3 wires
        // `quantize_streaming` directly.  Any backend-level structural
        // requirement (tensor name, ggml_type, byte count) the
        // streaming path violates would surface here BEFORE the
        // expensive write step.
        let backend = GgufBackend::new();
        let warnings = backend
            .validate(&model)
            .expect("validate must not error");

        // Backend validate may emit informational warnings (e.g. about
        // tensor count or naming conventions on a synthetic fixture)
        // but must not emit Error-severity ones — those would block
        // the iter-3 drop-in replacement.
        let errors: Vec<_> = warnings
            .iter()
            .filter(|w| {
                let s = format!("{:?}", w);
                s.to_lowercase().contains("error")
            })
            .collect();
        assert!(
            errors.is_empty(),
            "streaming output failed validate with {} error-severity \
             warnings: {:?} — `quantize_streaming` is NOT a drop-in \
             replacement for `quantize_model` against GgufBackend",
            errors.len(),
            errors
        );
    }

    /// ADR-014 P7 iter-44 — end-to-end iter-3 flow integration gate.
    /// Proves the full iter-3 production wire-up shape works without
    /// `main.rs:1147`'s `materialize_all()` bridge:
    ///
    ///   lazy_map (Phase 1.x) → quantize_streaming → QuantizedModel
    ///                                              ↓
    ///   re-read safetensors → lazy_map_2 → measure_quality_streaming_lazy
    ///                                              ↓
    ///   GgufBackend::validate(&QuantizedModel)
    ///
    /// Each tensor is touched at most once per phase; no full-model
    /// `TensorMap` is ever resident.  This is the architectural shape
    /// `main.rs:1147` will adopt when iter-3 lands.
    ///
    /// **Iter-3 prerequisite surfaced + resolved**: iter-44 originally
    /// used `KQuantCodecQuantizer` and discovered that
    /// `quality::dequantize_single_tensor` did NOT handle the
    /// `METHOD_K_QUANT_CODEC_DIRECT` method — it fell into the
    /// "no scales" warn path and returned zeros.  iter-45 (`3ebd4da`)
    /// added the codec-direct dequant arm dispatching off `ggml_type`
    /// for K-quants (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K) and legacy codec outputs
    /// (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0).  This iter still uses StaticQuantizer
    /// for the smaller fixture; the K-quant path is now exercised via
    /// the iter-45 byte-identity gate `kquant_codec_direct_dequant_round_trip`.
    ///
    /// Synthetic fixture (in-memory, no on-disk safetensors required):
    /// build the same set of tensors twice (once for quantize, once for
    /// quality) — mirrors the production "re-read after quantize"
    /// pattern.
    ///
    /// Asserts:
    ///   1. quantize_streaming produces a valid QuantizedModel.
    ///   2. measure_quality_streaming_lazy succeeds on the parallel lazy_map.
    ///   3. QualityReport has non-empty per_layer_cosine_sim.
    ///   4. GgufBackend::validate accepts the QuantizedModel.
    #[test]
    fn iter3_end_to_end_flow_lazy_quantize_quality_validate() {
        use crate::backends::{gguf::GgufBackend, OutputBackend};
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quality::measure_quality_streaming_lazy;
        use crate::quantize::static_quant::StaticQuantizer;

        // Use 64 elements / tensor (StaticQuantizer's group quantization
        // works for any size; the quality dequant path is wired for it).
        const N_ELEMENTS: usize = 64;

        let tensors: Vec<&str> = vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_v.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_up.weight",
        ];

        let make_payload = |seed: f32| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(N_ELEMENTS * 2);
            for i in 0..N_ELEMENTS {
                let v = (i as f32 / N_ELEMENTS as f32) * 2.0 - 1.0 + seed * 0.001;
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            bytes
        };

        // Build lazy_map_a (consumed by quantize_streaming).
        let mut lazy_map_a = LazyTensorMap::new();
        for (i, name) in tensors.iter().enumerate() {
            lazy_map_a.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![2, N_ELEMENTS / 2], DType::F16),
                make_payload(i as f32),
            ));
        }
        // Build lazy_map_b (consumed by measure_quality_streaming_lazy
        // — mirrors production "re-read safetensors after quantize").
        let mut lazy_map_b = LazyTensorMap::new();
        for (i, name) in tensors.iter().enumerate() {
            lazy_map_b.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), vec![2, N_ELEMENTS / 2], DType::F16),
                make_payload(i as f32),
            ));
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = StaticQuantizer::new("q8").expect("StaticQuantizer q8");

        // Phase 3: streaming quantize.
        let model =
            quantize_streaming(lazy_map_a, &metadata, &q, 8, 32, &progress, false)
                .expect("Phase 3 streaming quantize");

        // Phase 4.5: lazy quality measurement on the parallel lazy_map.
        let report = measure_quality_streaming_lazy(
            &lazy_map_b,
            &model,
            &metadata,
            std::path::Path::new("/tmp"),
            &progress,
        )
        .expect("Phase 4.5 lazy quality measurement");

        let per_layer = report
            .per_layer_cosine_sim
            .as_ref()
            .expect("per-layer cosine sim must exist for non-empty fixture");
        assert!(
            !per_layer.is_empty(),
            "per-layer must include at least one weight pair"
        );
        for (i, &v) in per_layer.iter().enumerate() {
            assert!(
                v > 0.99,
                "tensor {i}: cosine {v} should be > 0.99 for 8-bit static quant \
                 — quality measurement is degraded under iter-3 flow shape"
            );
        }

        // Backend validate.
        let backend = GgufBackend::new();
        let warnings = backend.validate(&model).expect("validate must not error");
        let errors: Vec<_> = warnings
            .iter()
            .filter(|w| format!("{:?}", w).to_lowercase().contains("error"))
            .collect();
        assert!(
            errors.is_empty(),
            "iter-3 flow QuantizedModel failed validate: {:?}",
            errors
        );
    }

    /// ADR-014 P7 iter-47 — `quantize_via_streaming_consuming` byte-identity
    /// to `quantize_model` on the same fixture.
    ///
    /// Pins the iter-3 wire-up wedge contract: when main.rs Phase 3
    /// arms swap `quantize_model(&tensor_map, ...)` for
    /// `quantize_via_streaming_consuming(tensor_map, ...)`, the
    /// QuantizedModel output is byte-equal so backend write + Phase 4.5
    /// quality + Phase 4.6 native paths all see the same bytes.
    ///
    /// 5-tensor multi-fixture spans the categorical-routing surface
    /// (output bump, attn_v, ffn_down, attn_q base, embedding) so a
    /// regression in any per-tensor dispatch path surfaces.  Q4_K_M
    /// variant chosen because it exercises the `_M` upgrade ladder.
    #[test]
    fn quantize_via_streaming_consuming_byte_identical_to_quantize_model() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::TensorMap;
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;

        let make_payload = |seed: f32| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(QK_K * 2);
            for i in 0..QK_K {
                let v = (i as f32 / QK_K as f32) * 2.0 - 1.0 + seed * 0.001;
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            bytes
        };

        let tensors: Vec<&str> = vec![
            "output.weight",
            "blk.0.attn_v.weight",
            "blk.0.ffn_down.weight",
            "blk.5.attn_q.weight",
            "blk.10.ffn_up.weight",
        ];

        // Build two identical TensorMaps so we can consume one
        // (streaming-path) and borrow the other (eager-path).
        let build_map = || -> TensorMap {
            let mut m = TensorMap::new();
            for (i, name) in tensors.iter().enumerate() {
                m.insert(crate::ir::TensorRef {
                    name: name.to_string(),
                    shape: vec![1, QK_K],
                    dtype: DType::F16,
                    data: make_payload(i as f32),
                });
            }
            m
        };
        let map_for_eager = build_map();
        let map_for_streaming = build_map();

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        let q_eager =
            VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, N_LAYERS);
        let q_streaming =
            VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, N_LAYERS);

        let eager_out = quantize_model(&map_for_eager, &metadata, &q_eager, 0, 0, &progress)
            .expect("quantize_model");
        let streaming_out = quantize_via_streaming_consuming(
            map_for_streaming,
            &metadata,
            &q_streaming,
            0,
            0,
            &progress,
        )
        .expect("quantize_via_streaming_consuming");

        // Same shape contract.
        assert_eq!(eager_out.quant_method, streaming_out.quant_method);
        assert_eq!(eager_out.tensors.len(), streaming_out.tensors.len());

        for tname in &tensors {
            let e = eager_out
                .tensors
                .get(*tname)
                .unwrap_or_else(|| panic!("eager missing {tname}"));
            let s = streaming_out
                .tensors
                .get(*tname)
                .unwrap_or_else(|| panic!("streaming missing {tname}"));

            assert_eq!(
                e.quant_info.ggml_type, s.quant_info.ggml_type,
                "{tname}: ggml_type"
            );
            assert_eq!(
                e.quant_info.method, s.quant_info.method,
                "{tname}: method"
            );
            assert_eq!(
                e.quant_info.preserved, s.quant_info.preserved,
                "{tname}: preserved"
            );
            assert_eq!(
                e.data, s.data,
                "{tname}: bytes differ between eager and streaming-consuming \
                 — iter-3 wire-up wedge would break byte-identity contract"
            );
        }
    }

    /// ADR-014 P7 iter-48 — `quantize_via_streaming_borrowed` byte-identity
    /// to `quantize_model`.  Pins the borrowing wedge contract: the
    /// env-flag-gated `HF2Q_STREAMING_PHASE3=1` swap in main.rs Phase 3
    /// produces byte-equal output to the eager path.
    ///
    /// Unlike iter-47's `quantize_via_streaming_consuming` which moves
    /// bytes, this wedge clones them — peak memory is HIGHER during
    /// the call.  The byte-identity contract is the same; the memory
    /// caveat is documented (it's a transitional production-exercise
    /// helper, not an end-state).
    #[test]
    fn quantize_via_streaming_borrowed_byte_identical_to_quantize_model() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::TensorMap;
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;

        let make_payload = |seed: f32| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(QK_K * 2);
            for i in 0..QK_K {
                let v = (i as f32 / QK_K as f32) * 2.0 - 1.0 + seed * 0.001;
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            bytes
        };

        let tensors: Vec<&str> = vec![
            "output.weight",
            "blk.0.attn_v.weight",
            "blk.0.ffn_down.weight",
            "blk.5.attn_q.weight",
        ];

        let mut tensor_map = TensorMap::new();
        for (i, name) in tensors.iter().enumerate() {
            tensor_map.insert(crate::ir::TensorRef {
                name: name.to_string(),
                shape: vec![1, QK_K],
                dtype: DType::F16,
                data: make_payload(i as f32),
            });
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q1 = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, N_LAYERS);
        let q2 = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, N_LAYERS);

        let eager = quantize_model(&tensor_map, &metadata, &q1, 0, 0, &progress)
            .expect("quantize_model");
        let borrowed = quantize_via_streaming_borrowed(&tensor_map, &metadata, &q2, 0, 0, &progress)
            .expect("quantize_via_streaming_borrowed");

        assert_eq!(eager.quant_method, borrowed.quant_method);
        assert_eq!(eager.tensors.len(), borrowed.tensors.len());

        for tname in &tensors {
            let e = eager.tensors.get(*tname).unwrap();
            let b = borrowed.tensors.get(*tname).unwrap();
            assert_eq!(e.quant_info.ggml_type, b.quant_info.ggml_type, "{tname}");
            assert_eq!(
                e.data, b.data,
                "{tname}: bytes differ between eager and borrowed-streaming"
            );
        }
    }

    /// ADR-014 P7 iter-49 — `quantize_via_streaming_borrowed` byte-identity
    /// to `quantize_model` on the ImatrixAdaptive path (VariantKQuantizer +
    /// real imatrix calibration data).  Pins the iter-49 main.rs wire-up
    /// at the ImatrixAdaptive arm (line 1430+) under
    /// `HF2Q_STREAMING_PHASE3=1`.
    ///
    /// Critical signal: iter-48's gate covered Q4_K_M with
    /// `CalibrationData::None`.  This iter exercises the ImatrixAdaptive
    /// path explicitly with non-uniform importance — proving the
    /// streaming dispatch threads imatrix through to the codec for
    /// VariantKQuantizer's per-tensor target dispatch.
    #[test]
    fn quantize_via_streaming_borrowed_byte_identical_under_imatrix_variant_kquantizer() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::ir::TensorMap;
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

        const QK_K: usize = 256;
        const N_LAYERS: usize = 32;

        let make_payload = |seed: f32| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(QK_K * 2);
            for i in 0..QK_K {
                let v = (i as f32 / QK_K as f32) * 2.0 - 1.0 + seed * 0.001;
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            bytes
        };

        let tensors: Vec<&str> = vec![
            "blk.0.attn_v.weight",
            "blk.0.ffn_down.weight",
            "blk.5.attn_q.weight",
        ];

        let mut tensor_map = TensorMap::new();
        for (i, name) in tensors.iter().enumerate() {
            tensor_map.insert(crate::ir::TensorRef {
                name: name.to_string(),
                shape: vec![1, QK_K],
                dtype: DType::F16,
                data: make_payload(i as f32),
            });
        }

        // Non-uniform importance per tensor.
        let acts: Vec<f32> = (0..QK_K)
            .map(|i| if i < 16 { 100.0 } else { 0.01 })
            .collect();
        let mut col = ImatrixCollector::new();
        for name in &tensors {
            col.accumulate_dense(name, &acts, 1, QK_K).unwrap();
        }
        col.record_chunk();

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();

        let q1 = VariantKQuantizer::new(
            KQuantVariant::Q4_K_M,
            CalibrationData::from_imatrix_collector(&col),
            N_LAYERS,
        );
        let q2 = VariantKQuantizer::new(
            KQuantVariant::Q4_K_M,
            CalibrationData::from_imatrix_collector(&col),
            N_LAYERS,
        );

        let eager = quantize_model(&tensor_map, &metadata, &q1, 0, 0, &progress)
            .expect("quantize_model");
        let borrowed =
            quantize_via_streaming_borrowed(&tensor_map, &metadata, &q2, 0, 0, &progress)
                .expect("quantize_via_streaming_borrowed");

        for tname in &tensors {
            let e = eager.tensors.get(*tname).unwrap();
            let b = borrowed.tensors.get(*tname).unwrap();
            assert_eq!(
                e.quant_info.ggml_type, b.quant_info.ggml_type,
                "{tname}: ggml_type"
            );
            assert_eq!(
                e.data, b.data,
                "{tname}: bytes differ — imatrix-VariantKQuantizer streaming path \
                 has lost byte-identity vs eager"
            );
        }
    }

    /// ADR-014 P7 iter-51 — `quantize_via_streaming_borrowed` byte-identity
    /// to `quantize_model` on the DwqKQuantizer path.  Closes the four-arm
    /// Phase 3 migration: K-quant codec direct (iter-48), ImatrixAdaptive
    /// (iter-49), StaticQuantizer (iter-50), DwqK (iter-51).
    ///
    /// DwqK has the most complex per-tensor dispatch: sensitive vs base
    /// routing via `is_sensitive_tensor` + `target_for(tensor_name)`,
    /// then delegate to KQuantCodecQuantizer.  The streaming path
    /// preserves this layer-by-layer because each tensor's name is in
    /// the LazyTensor metadata and routes deterministically.
    ///
    /// Test fixture mixes a sensitive layer (Q8_0 routing) and base
    /// layers (Q4_K routing) under DwqK::P48, asserting the two paths
    /// (eager `quantize_model` vs streaming-borrowed) emit byte-equal
    /// output across both routing buckets.
    #[test]
    fn quantize_via_streaming_borrowed_byte_identical_under_dwq_k() {
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::TensorMap;
        use crate::quantize::dwq_k_quantizer::{DwqKQuantizer, DwqKVariant};

        const QK_K: usize = 256;

        let make_payload = |seed: f32| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(QK_K * 2);
            for i in 0..QK_K {
                let v = (i as f32 / QK_K as f32) * 2.0 - 1.0 + seed * 0.001;
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            bytes
        };

        // DwqK uses HF naming (`model.layers.<N>.…`).  Mix sensitive
        // (layer 5 → Q8_0) + base (layer 0/10 → Q4_K) buckets under P48.
        let tensors: Vec<&str> = vec![
            "model.layers.0.self_attn.q_proj.weight",   // base → Q4_K
            "model.layers.5.self_attn.q_proj.weight",   // sensitive → Q8_0
            "model.layers.10.mlp.down_proj.weight",     // base → Q4_K
        ];

        let mut tensor_map = TensorMap::new();
        for (i, name) in tensors.iter().enumerate() {
            tensor_map.insert(crate::ir::TensorRef {
                name: name.to_string(),
                shape: vec![1, QK_K],
                dtype: DType::F16,
                data: make_payload(i as f32),
            });
        }

        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q1 = DwqKQuantizer::new(DwqKVariant::P48, &[5..=5], CalibrationData::None);
        let q2 = DwqKQuantizer::new(DwqKVariant::P48, &[5..=5], CalibrationData::None);

        let eager = quantize_model(&tensor_map, &metadata, &q1, 0, 0, &progress)
            .expect("quantize_model");
        let borrowed =
            quantize_via_streaming_borrowed(&tensor_map, &metadata, &q2, 0, 0, &progress)
                .expect("quantize_via_streaming_borrowed");

        for tname in &tensors {
            let e = eager.tensors.get(*tname).unwrap();
            let b = borrowed.tensors.get(*tname).unwrap();
            assert_eq!(e.quant_info.ggml_type, b.quant_info.ggml_type, "{tname}");
            assert_eq!(
                e.data, b.data,
                "{tname}: bytes differ — DwqK streaming path lost byte-identity vs eager"
            );
        }

        // Cross-check: the sensitive layer 5 emitted Q8_0 (272 bytes for
        // 256 elements), the base layers emitted Q4_K (144 bytes).  Locks
        // the sensitive/base routing decisions are firing in BOTH paths.
        let s = borrowed
            .tensors
            .get("model.layers.5.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(s.quant_info.ggml_type.as_deref(), Some("Q8_0"));
        let b0 = borrowed
            .tensors
            .get("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(b0.quant_info.ggml_type.as_deref(), Some("Q4_K"));
    }

    /// ADR-014 P7 iter-56 — full iter-3 lazy pipeline end-to-end gate.
    /// Architecturally simulates main.rs's iter-3 wholesale-skip path
    /// at the unit-test level: every step operates on `LazyTensorMap`
    /// without `materialize_all()` ever firing on the full model.
    ///
    /// Pipeline shape (mirrors what main.rs:1147+ becomes post-iter-3):
    ///   1. Build LazyTensorMap (mock of `read_tensors_lazy` output)
    ///   2. lazy_map.convert_bf16_to_f16()    (iter-55)
    ///   3. lazy_map.total_size_bytes()       (iter-54 — telemetry)
    ///   4. quantize_streaming(lazy_map_for_q3, …)        (iter-1+47)
    ///   5. measure_quality_streaming_lazy(lazy_map_for_q45, …)  (iter-43+45)
    ///   6. backend.validate(&QuantizedModel)
    ///
    /// Step 4 consumes lazy_map; step 5 takes a parallel re-read.
    /// In production this re-read would come from `read_tensors_lazy`
    /// over the safetensors files; here we build an identical map
    /// upstream.
    ///
    /// This iter does NOT touch main.rs.  Its purpose: prove that
    /// every iter-54/55 API piece + iter-43/45/47-52 pipeline stage
    /// composes cleanly under the iter-3 architectural shape, so the
    /// eventual main.rs surgery (iter-57+) is a transparent code
    /// shuffle without behavioral risk.
    ///
    /// Asserts:
    ///   - convert_bf16_to_f16 returns expected count
    ///   - total_size_bytes matches a known sum
    ///   - QuantizedModel non-empty
    ///   - QualityReport per-layer cosine ≥ threshold (8-bit dequant)
    ///   - GgufBackend::validate clean
    #[test]
    fn iter3_lazy_pipeline_end_to_end_no_materialize_all() {
        use crate::backends::{gguf::GgufBackend, OutputBackend};
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quality::measure_quality_streaming_lazy;
        use crate::quantize::static_quant::StaticQuantizer;

        const N_ELEMENTS: usize = 64;

        // Mixed-dtype fixture: 2 BF16 weights + 1 F16 weight.  Mirrors
        // what `read_tensors_lazy` would yield on a model with mixed
        // input precisions.
        fn bf16_payload(seed: f32) -> Vec<u8> {
            (0..N_ELEMENTS)
                .map(|i| (i as f32 / N_ELEMENTS as f32) * 2.0 - 1.0 + seed * 0.001)
                .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
                .collect()
        }
        fn f16_payload(seed: f32) -> Vec<u8> {
            (0..N_ELEMENTS)
                .map(|i| (i as f32 / N_ELEMENTS as f32) * 2.0 - 1.0 + seed * 0.001)
                .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
                .collect()
        }

        let build_lazy_map = || -> LazyTensorMap {
            let mut m = LazyTensorMap::new();
            m.insert(LazyTensor::from_bytes(
                LazyMeta::new(
                    "blk.0.attn_q.weight".to_string(),
                    vec![2, N_ELEMENTS / 2],
                    DType::BF16,
                ),
                bf16_payload(0.0),
            ));
            m.insert(LazyTensor::from_bytes(
                LazyMeta::new(
                    "blk.0.attn_v.weight".to_string(),
                    vec![2, N_ELEMENTS / 2],
                    DType::BF16,
                ),
                bf16_payload(1.0),
            ));
            m.insert(LazyTensor::from_bytes(
                LazyMeta::new(
                    "blk.0.ffn_down.weight".to_string(),
                    vec![2, N_ELEMENTS / 2],
                    DType::F16,
                ),
                f16_payload(2.0),
            ));
            m
        };

        // Step 1+2: build + bf16→f16 (iter-55).
        let mut lazy_map = build_lazy_map();
        let converted = lazy_map.convert_bf16_to_f16().expect("bf16 convert");
        assert_eq!(converted, 2, "two BF16 tensors converted");

        // Step 3: telemetry (iter-54).  Post-conversion all 3 tensors
        // are F16 (2 bytes) × 64 elems = 128 bytes each → 384 total.
        let total_bytes = lazy_map.total_size_bytes();
        assert_eq!(total_bytes, 3 * 64 * 2, "lazy total_size_bytes post-bf16");

        // Step 4: quantize via streaming (iter-1+47).
        let metadata = dummy_metadata();
        let progress = crate::progress::ProgressReporter::new();
        let q = StaticQuantizer::new("q8").expect("StaticQuantizer q8");
        let model = quantize_streaming(lazy_map, &metadata, &q, 8, 32, &progress, false)
            .expect("Phase 3 streaming quantize");
        assert!(!model.tensors.is_empty(), "QuantizedModel non-empty");
        assert_eq!(model.tensors.len(), 3, "all three tensors quantized");

        // Step 5: lazy quality on parallel re-read map (iter-43).
        // In production this comes from `read_tensors_lazy` again;
        // here we rebuild the same fixture (post-bf16 cast) to mirror
        // the streaming re-read pattern.
        let mut lazy_for_quality = build_lazy_map();
        lazy_for_quality
            .convert_bf16_to_f16()
            .expect("bf16 convert (q-side)");
        let report = measure_quality_streaming_lazy(
            &lazy_for_quality,
            &model,
            &metadata,
            std::path::Path::new("/tmp"),
            &progress,
        )
        .expect("Phase 4.5 lazy quality");
        let per_layer = report
            .per_layer_cosine_sim
            .as_ref()
            .expect("per-layer cosine sim must exist");
        assert!(
            !per_layer.is_empty(),
            "per-layer must contain weight pairs"
        );
        for (i, &v) in per_layer.iter().enumerate() {
            assert!(
                v > 0.99,
                "tensor {i} cosine {v} ≤ 0.99 — 8-bit static dequant degraded"
            );
        }

        // Step 6: backend validate.
        let backend = GgufBackend::new();
        let warnings = backend.validate(&model).expect("validate");
        let errors: Vec<_> = warnings
            .iter()
            .filter(|w| format!("{:?}", w).to_lowercase().contains("error"))
            .collect();
        assert!(
            errors.is_empty(),
            "iter-3 lazy pipeline output must validate clean: {:?}",
            errors
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
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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
                crate::quantize::k_quant_codec::KQuantTarget::Q2K => ("Q2_K", 84),
                crate::quantize::k_quant_codec::KQuantTarget::Q3K => ("Q3_K", 110),
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
                t.data.len(),
                expected_bytes,
                "{variant}: emitted block size"
            );
            assert_eq!(
                out.quant_method,
                variant.name(),
                "{variant}: quant_method tag"
            );
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
        use crate::calibrate::calibrator::CalibrationData;
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::quantize::layer_mix::KQuantVariant;
        use crate::quantize::variant_quantizer::VariantKQuantizer;

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
            (
                "blk.0.ffn_gate_inp.weight",
                vec![1, QK_K],
                "passthrough",
                None,
            ),
            (
                "blk.0.ffn_down_exps.weight",
                vec![1, QK_K],
                "k_quant_codec_direct",
                Some("Q6_K"),
            ),
            (
                "blk.0.ffn_up_exps.weight",
                vec![1, QK_K],
                "k_quant_codec_direct",
                Some("Q4_K"),
            ),
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
        let quantizer =
            VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, N_LAYERS);
        let progress = crate::progress::ProgressReporter::new();

        let result = quantize_streaming(
            lazy_map, &metadata, &quantizer, 0, 0, &progress, /* bf16_to_f16 */ false,
        )
        .unwrap();

        for (name, _shape, expected_method, expected_type) in &fixtures {
            let t = result
                .tensors
                .get(*name)
                .unwrap_or_else(|| panic!("streaming output missing {name}"));
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
                    t.data.len(),
                    len * 2,
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
                    t.data.len(),
                    expected_bytes,
                    "{name} should emit {expected_bytes}-byte blocks for {}",
                    expected_type.unwrap()
                );
                assert!(!t.quant_info.preserved, "{name} should be quantized");
            }
        }
    }
}
