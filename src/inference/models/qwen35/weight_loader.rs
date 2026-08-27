//! Real GGUF weight loading into the Qwen3.5 CPU reference types.
//!
//! Bridges mlx-native's [`GgufFile::load_tensor_f32`] to hf2q's
//! [`FullAttnLayerWeights`], [`DeltaNetLayerWeights`], and FFN variant
//! weight structs.
//!
//! # Memory strategy
//!
//! The full apex GGUF dequantized to f32 is ~80 GB — too large for a
//! single in-memory model. This loader is **per-layer on-demand**: the
//! caller loads ONE layer at a time, runs whatever validation or forward
//! pass it needs, then drops the layer before loading the next one. For
//! a 40-layer MoE, per-layer dequantized memory is ~200 MB-2 GB (varies
//! by layer kind) — well within a single process's budget.
//!
//! Production GPU inference uses the native loaders in this module to retain
//! quantized `MlxBuffer`s and pass them directly to mlx-native's GGML kernels.
//! The per-layer F32 loaders remain CPU-reference and synthetic-fixture tools.
//!
//! # Layout conversion
//!
//! GGUF's `load_tensor_f32` returns a `Vec<f32>` with dims in "outer-first"
//! shape order (see `TensorInfo.shape`). For matrices, this usually matches
//! the `[out_dim, in_dim]` convention our CPU refs expect. Single-vector
//! tensors (norms, biases) are 1-D and drop in directly.

use anyhow::{anyhow, Context, Result};
use mlx_native::gguf::{GgufFile, GgufMappedTensorSet, TensorInfo};
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType as MlxDType, MlxBuffer, MlxDevice};
use std::collections::BTreeMap;

use crate::ir::lazy::{LazyTensor, LazyTensorMap};
use crate::ir::{DType as IrDType, TensorRef};

use super::delta_net::DeltaNetLayerWeights;
use super::ffn::{DenseFfnWeights, MoeFfnWeights};
use super::full_attn::FullAttnLayerWeights;
use super::gpu_delta_net::DeltaNetWeightsGpu;
use super::gpu_full_attn::{FullAttnQGateWeightsGpu, FullAttnWeightsGpu};
use super::in_memory_loader::{
    bf16_bytes_to_f32, f16_bytes_to_f32, f32_bytes_to_f32, quantize_f32_to_q8_0_buffer,
};
use super::model::{Qwen35FfnWeights, Qwen35LayerWeights, Qwen35Model};
use super::{default_layer_types, Qwen35Config, Qwen35LayerKind, Qwen35MoeConfig, Qwen35Variant};
use crate::serve::forward_mlx_shared::{map_native_gguf_tensor_view, MlxQWeight};

// ============================================================================
// Quantized MoE weight container
// ============================================================================

/// Per-layer MoE FFN weights retaining every production matrix in the exact
/// storage representation declared by the GGUF.
///
/// This struct is the bridge between GGUF disk bytes and
/// `MoeFfnWeightsGpuQ`: it holds the raw Metal buffers that `GgufFile::load_tensor`
/// produced. Router and shared-expert matrices use the same native dense
/// dispatch abstraction as ordinary projections; they are never expanded and
/// uploaded as a BF16 shadow.
pub struct MoeFfnWeightsQ {
    /// Router: `[num_experts, hidden_size]` artifact-native matrix.
    pub router: MlxQWeight,
    /// Stacked expert gate_proj in exact artifact storage: scalar dtype for
    /// F32/F16/BF16, U8-packed bytes for block codecs.
    pub expert_gate_q: MlxBuffer,
    /// Stacked expert up_proj in exact artifact storage.
    pub expert_up_q: MlxBuffer,
    /// Stacked expert down_proj in exact artifact storage.
    pub expert_down_q: MlxBuffer,
    /// GGML quantization type for the gate expert buffer.
    pub ggml_type_gate: GgmlType,
    /// GGML quantization type for the up expert buffer.
    pub ggml_type_up: GgmlType,
    /// GGML quantization type for the down expert buffer.
    /// In the apex GGUF this is Q6_K.
    pub ggml_type_down: GgmlType,
    /// Shared-expert sigmoid gate: `[1, hidden_size]` artifact-native matrix.
    pub shared_gate_logit: MlxQWeight,
    /// Shared-expert SwiGLU matrices in their declared artifact codecs.
    pub shared_gate: MlxQWeight,
    pub shared_up: MlxQWeight,
    pub shared_down: MlxQWeight,
    /// ADR-020 AC#5 Iter C2.4 — DWQ-overlay mlx-affine expert stacks
    /// (packed-U32 weight + BF16 scales + BF16 biases).  When `Some`,
    /// the corresponding `expert_*_q` buffer above stays
    /// resident-but-unused; gpu_ffn dispatch sites gate on these
    /// `Option`s and route to `mlx_native::quantized_matmul_id_into`.
    /// Qwen35 splits gate + up (no fused gate_up like Gemma 4).
    pub expert_gate_affine: Option<crate::serve::forward_mlx_shared::MlxAffineMoeStack>,
    pub expert_up_affine: Option<crate::serve::forward_mlx_shared::MlxAffineMoeStack>,
    pub expert_down_affine: Option<crate::serve::forward_mlx_shared::MlxAffineMoeStack>,
}

// ============================================================================
// Quantized Dense FFN weight container
// ============================================================================

/// Per-layer dense SwiGLU FFN weights kept in their native GGML quantization.
///
/// This mirrors [`MoeFfnWeightsQ`] for the dense path: the three projection
/// buffers (`gate`, `up`, `down`) are raw GGML blocks (DType::U8 on Metal),
/// exactly as they came off disk.  The Metal `quantized_matmul_ggml` kernel
/// dequantizes on-the-fly during the matmul, so no F32 expansion occurs.
///
/// On the current Qwen3.6 27B DWQ GGUF, most dense FFN projection tensors are
/// Q4_0 and the final layers are Q6_K.  Keeping those blocks native avoids
/// expanding gate/up/down to F32 scratch weights during load, which was the
/// path that pushed the 27B dense fixture over Metal's working-set limits.
pub struct DenseFfnWeightsQ {
    /// Gate projection raw GGML blocks: `[intermediate_size, hidden_size]`.
    pub gate_q: MlxBuffer,
    /// Up projection raw GGML blocks: `[intermediate_size, hidden_size]`.
    pub up_q: MlxBuffer,
    /// Down projection raw GGML blocks: `[hidden_size, intermediate_size]`.
    pub down_q: MlxBuffer,
    /// GGML quantization type for the gate projection.
    pub ggml_type_gate: GgmlType,
    /// GGML quantization type for the up projection.
    pub ggml_type_up: GgmlType,
    /// GGML quantization type for the down projection.
    pub ggml_type_down: GgmlType,
    /// Dense FFN intermediate dimension (number of rows in gate/up weight).
    pub intermediate_size: u32,
    /// Model hidden dimension (number of columns in gate/up weight).
    pub hidden_size: u32,
}

/// Per-layer dense SwiGLU weights retaining native scalar GGUF storage.
///
/// F32, F16, and BF16 buffers remain typed Metal buffers exactly as loaded;
/// execution selects the matching dense kernel per projection. This is
/// intentionally separate from [`DenseFfnWeights`] so production GGUF loads
/// can never materialize a second F32 copy as an incidental CPU-reference
/// representation.
pub struct DenseFfnWeightsNative {
    pub gate: MlxBuffer,
    pub up: MlxBuffer,
    pub down: MlxBuffer,
    pub gate_type: GgmlType,
    pub up_type: GgmlType,
    pub down_type: GgmlType,
    pub intermediate_size: u32,
    pub hidden_size: u32,
}

/// Load a tensor from the GGUF, dequantize to f32, and download into
/// a `Vec<f32>`.
pub fn load_f32_tensor(gguf: &GgufFile, name: &str, device: &MlxDevice) -> Result<Vec<f32>> {
    let buf = gguf
        .load_tensor_f32(name, device)
        .map_err(|e| anyhow!("load_tensor_f32({name}): {e}"))?;
    let slice: &[f32] = buf
        .as_slice()
        .map_err(|e| anyhow!("as_slice({name}): {e}"))?;
    #[cfg(test)]
    super::execution_observation::observe_loaded_f32(name, slice)?;
    Ok(slice.to_vec())
}

/// Retain one exact F32 state tensor from the model's shared GGUF mapping.
///
/// Production GGUF loads run the complete role/codec preflight before this
/// helper is reached. Re-checking the storage contract here keeps the binding
/// fail closed even when a future caller is added outside that load path. The
/// returned buffer aliases the artifact mapping; no anonymous upload, dtype
/// conversion, or weight shadow is created.
pub(super) fn map_f32_state_with_residency(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    name: &str,
    device: &MlxDevice,
) -> Result<MlxBuffer> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| anyhow!("Qwen F32 state tensor '{name}' not found"))?;
    anyhow::ensure!(
        info.ggml_type == GgmlType::F32,
        "Qwen state tensor '{name}' must retain F32 storage, got {:?}",
        info.ggml_type
    );
    let expected_bytes = info
        .shape
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| anyhow!("Qwen F32 state tensor '{name}' byte extent overflow"))?;
    anyhow::ensure!(
        info.byte_len == expected_bytes,
        "Qwen F32 state tensor '{name}' has {} bytes, expected {expected_bytes}",
        info.byte_len
    );
    let buffer = map_native_gguf_tensor_view(mapped, info)
        .with_context(|| format!("retain native Qwen F32 state '{name}'"))?;
    anyhow::ensure!(
        buffer.dtype() == MlxDType::F32
            && buffer.data_byte_len() == expected_bytes
            && buffer.is_file_backed(),
        "Qwen F32 state tensor '{name}' did not retain its exact mapped payload"
    );
    super::weight_pool::register_weight_buffer(device, &buffer)
        .map_err(|e| anyhow!("register_weight_buffer({name}): {e}"))?;
    #[cfg(test)]
    {
        let values = buffer
            .as_slice::<f32>()
            .map_err(|e| anyhow!("read mapped F32 state {name}: {e}"))?;
        super::execution_observation::observe_loaded_f32(name, values)?;
    }
    Ok(buffer)
}

/// Load a quantized tensor as raw GGML blocks (DType::U8 on Metal) and
/// register the resulting Metal buffer with the thread-local weight pool's
/// `MTLResidencySet`.
///
/// **Wave 5b.7 iter 2:** this is the residency-aware drop-in for the bare
/// `GgufFile::load_tensor` calls used by per-layer MoE / dense quantized
/// loaders.  It is a thin wrapper around the public
/// [`mlx_native::gguf::GgufFile::load_tensor`] + the public
/// [`super::weight_pool::register_weight_buffer`] helper.  No bucket-rounding
/// — buffers are allocated at their exact GGML byte length.  No-op for the
/// residency call when `HF2Q_NO_RESIDENCY=1`.
fn map_tensor_with_residency(
    mapped: &GgufMappedTensorSet<'_>,
    info: &mlx_native::gguf::TensorInfo,
    device: &MlxDevice,
) -> Result<MlxBuffer> {
    let name = info.name.as_str();
    let buf = map_native_gguf_tensor_view(mapped, info)?;
    super::weight_pool::register_weight_buffer(device, &buf)
        .map_err(|e| anyhow!("register_weight_buffer({name}): {e}"))?;
    #[cfg(test)]
    super::execution_observation::observe_loaded_ggml(name, &buf)?;
    Ok(buf)
}

pub(super) fn load_native_projection(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    name: &str,
    rows: usize,
    cols: usize,
    device: &MlxDevice,
) -> Result<(MlxBuffer, GgmlType)> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| anyhow!("native Qwen projection '{name}' not found"))?;
    let expected = validate_native_projection_info(name, info, rows, cols)?;
    let buffer = map_tensor_with_residency(mapped, info, device)?;
    let expected_dtype = match info.ggml_type {
        GgmlType::F32 => MlxDType::F32,
        GgmlType::F16 => MlxDType::F16,
        GgmlType::BF16 => MlxDType::BF16,
        _ => MlxDType::U8,
    };
    anyhow::ensure!(
        buffer.dtype() == expected_dtype,
        "native Qwen projection '{name}' {:?} tensor loaded as {:?}",
        info.ggml_type,
        buffer.dtype()
    );
    let loaded_bytes = buffer.data_byte_len();
    anyhow::ensure!(
        loaded_bytes == expected,
        "native Qwen projection '{name}' loaded byte length {loaded_bytes} != {expected}"
    );
    Ok((buffer, info.ggml_type))
}

/// Descriptor-only admission shared by hosted preflight and the production
/// projection loader. It deliberately validates the exact packed extent as
/// well as shape and dispatch type, before any Metal buffer is allocated.
pub(crate) fn validate_native_projection_info(
    name: &str,
    info: &TensorInfo,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    anyhow::ensure!(
        info.shape.as_slice() == [rows, cols],
        "native Qwen projection '{name}' shape {:?} != [{rows}, {cols}]",
        info.shape
    );
    anyhow::ensure!(
        qwen35_native_projection_type_supported(info.ggml_type),
        "native Qwen projection '{name}' uses {:?}, which has no complete scalar decode/prefill route",
        info.ggml_type
    );
    anyhow::ensure!(
        cols % info.ggml_type.block_values() as usize == 0,
        "native Qwen projection '{name}' row width {cols} is not aligned to {:?}'s {}-value blocks",
        info.ggml_type,
        info.ggml_type.block_values()
    );
    let expected = rows
        .checked_mul(cols / info.ggml_type.block_values() as usize)
        .and_then(|v| v.checked_mul(info.ggml_type.block_bytes() as usize))
        .ok_or_else(|| anyhow!("native Qwen projection '{name}' byte length overflow"))?;
    anyhow::ensure!(
        info.byte_len == expected,
        "native Qwen projection '{name}' byte length {} != expected {expected} for {:?}",
        info.byte_len,
        info.ggml_type
    );
    Ok(expected)
}

/// Descriptor-only contract for the shared-expert sigmoid gate. GGUF stores
/// this logical one-row projection as an exact rank-one vector; accepting a
/// rank-two squeeze here would make preflight, target loading, and MTP loading
/// disagree about the artifact's native representation.
pub(crate) fn validate_native_row_projection_info(
    name: &str,
    info: &TensorInfo,
    cols: usize,
) -> Result<usize> {
    anyhow::ensure!(
        info.shape.as_slice() == [cols],
        "native Qwen row projection '{name}' must be exact rank 1 with shape [{cols}], got {:?}",
        info.shape
    );
    anyhow::ensure!(
        qwen35_native_projection_type_supported(info.ggml_type),
        "native Qwen row projection '{name}' uses {:?}, which has no complete scalar decode/prefill route",
        info.ggml_type
    );
    anyhow::ensure!(
        cols > 0 && cols % info.ggml_type.block_values() as usize == 0,
        "native Qwen row projection '{name}' width {cols} is not aligned to {:?}'s {}-value blocks",
        info.ggml_type,
        info.ggml_type.block_values()
    );
    let expected = (cols / info.ggml_type.block_values() as usize)
        .checked_mul(info.ggml_type.block_bytes() as usize)
        .ok_or_else(|| anyhow!("native Qwen row projection '{name}' byte length overflow"))?;
    anyhow::ensure!(
        info.byte_len == expected,
        "native Qwen row projection '{name}' byte length {} != expected {expected} for {:?}",
        info.byte_len,
        info.ggml_type
    );
    Ok(expected)
}

pub(crate) fn qwen35_native_projection_type_supported(t: GgmlType) -> bool {
    matches!(
        t,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_0
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::IQ4_NL
            | GgmlType::IQ4_XS
    )
}

fn load_native_row_projection(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    name: &str,
    cols: usize,
    device: &MlxDevice,
) -> Result<MlxQWeight> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| anyhow!("native Qwen row projection '{name}' not found"))?;
    validate_native_row_projection_info(name, info, cols)?;
    let weight = MlxQWeight::from_mapped_gguf_row_vector(mapped, info, cols)
        .with_context(|| format!("retain native Qwen row projection '{name}'"))?;
    super::weight_pool::register_weight_buffer(device, &weight.buffer)
        .map_err(|error| anyhow!("register_weight_buffer({name}): {error}"))?;
    #[cfg(test)]
    super::execution_observation::observe_loaded_ggml(name, &weight.buffer)?;
    Ok(weight)
}

/// Load one full-attention block with the conversion-emitted quantized
/// representation intact. The fused Q/gate matrix stays fused and native;
/// inference projects it once and deinterleaves only the F32 activation.
pub fn load_full_attn_layer_native(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    cfg: &Qwen35Config,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<FullAttnWeightsGpu> {
    let p = format!("blk.{layer_idx}");
    let hidden = cfg.hidden_size as usize;
    let heads = cfg.num_attention_heads as usize;
    let kv_heads = cfg.num_key_value_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_total = heads * head_dim;
    let kv_total = kv_heads * head_dim;

    let fused_name = format!("{p}.attn_q.weight");
    let (fused, fused_type) =
        load_native_projection(gguf, mapped, &fused_name, 2 * q_total, hidden, device)?;
    let (wk, wk_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.attn_k.weight"),
        kv_total,
        hidden,
        device,
    )?;
    let (wv, wv_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.attn_v.weight"),
        kv_total,
        hidden,
        device,
    )?;
    let (wo, wo_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.attn_output.weight"),
        hidden,
        q_total,
        device,
    )?;

    Ok(FullAttnWeightsGpu {
        attn_norm: map_f32_state_with_residency(
            gguf,
            mapped,
            &format!("{p}.attn_norm.weight"),
            device,
        )?,
        post_attn_norm: map_f32_state_with_residency(
            gguf,
            mapped,
            &format!("{p}.post_attention_norm.weight"),
            device,
        )?,
        q_gate: FullAttnQGateWeightsGpu::Fused {
            weight: fused,
            ggml_type: fused_type,
        },
        wk,
        wk_ggml_type,
        wv,
        wv_ggml_type,
        attn_q_norm: map_f32_state_with_residency(
            gguf,
            mapped,
            &format!("{p}.attn_q_norm.weight"),
            device,
        )?,
        attn_k_norm: map_f32_state_with_residency(
            gguf,
            mapped,
            &format!("{p}.attn_k_norm.weight"),
            device,
        )?,
        wo,
        wo_ggml_type,
    })
}

/// Load one DeltaNet block with all five large projections in their native
/// GGML representation. Small F32 state/norm tensors keep their declared F32
/// storage; the rank-two convolution matrix remains the exact mapped GGUF view.
pub fn load_delta_net_layer_native(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    cfg: &Qwen35Config,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<DeltaNetWeightsGpu> {
    let p = format!("blk.{layer_idx}");
    let hidden = cfg.hidden_size as usize;
    let nk = cfg.linear_num_key_heads as usize;
    let nv = cfg.linear_num_value_heads as usize;
    let dk = cfg.linear_key_head_dim as usize;
    let dv = cfg.linear_value_head_dim as usize;
    let k_width = cfg.linear_conv_kernel_dim as usize;
    let qkv_channels = 2 * nk * dk + nv * dv;
    let z_channels = nv * dv;

    let (attn_qkv, attn_qkv_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.attn_qkv.weight"),
        qkv_channels,
        hidden,
        device,
    )?;
    let (attn_gate, attn_gate_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.attn_gate.weight"),
        z_channels,
        hidden,
        device,
    )?;
    let (ssm_alpha, ssm_alpha_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.ssm_alpha.weight"),
        nv,
        hidden,
        device,
    )?;
    let (ssm_beta, ssm_beta_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.ssm_beta.weight"),
        nv,
        hidden,
        device,
    )?;
    let (ssm_out, ssm_out_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.ssm_out.weight"),
        hidden,
        z_channels,
        device,
    )?;

    let (ssm_conv1d, ssm_conv1d_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.ssm_conv1d.weight"),
        qkv_channels,
        k_width,
        device,
    )?;
    anyhow::ensure!(
        ssm_conv1d_type == GgmlType::F32,
        "layer {layer_idx}: ssm_conv1d must retain F32 storage, got {ssm_conv1d_type:?}"
    );
    let ssm_dt_bias =
        map_f32_state_with_residency(gguf, mapped, &format!("{p}.ssm_dt.bias"), device)?;
    let ssm_a = map_f32_state_with_residency(gguf, mapped, &format!("{p}.ssm_a"), device)?;
    let ssm_norm =
        map_f32_state_with_residency(gguf, mapped, &format!("{p}.ssm_norm.weight"), device)?;
    let ssm_dt_bias_cpu = ssm_dt_bias
        .as_slice::<f32>()
        .map_err(|error| anyhow!("read mapped {p}.ssm_dt.bias: {error}"))?
        .to_vec();
    let ssm_a_cpu = ssm_a
        .as_slice::<f32>()
        .map_err(|error| anyhow!("read mapped {p}.ssm_a: {error}"))?
        .to_vec();
    let ssm_norm_cpu = ssm_norm
        .as_slice::<f32>()
        .map_err(|error| anyhow!("read mapped {p}.ssm_norm.weight: {error}"))?
        .to_vec();

    Ok(DeltaNetWeightsGpu {
        attn_norm: map_f32_state_with_residency(
            gguf,
            mapped,
            &format!("{p}.attn_norm.weight"),
            device,
        )?,
        post_attn_norm: map_f32_state_with_residency(
            gguf,
            mapped,
            &format!("{p}.post_attention_norm.weight"),
            device,
        )?,
        attn_qkv,
        attn_qkv_ggml_type,
        attn_gate,
        attn_gate_ggml_type,
        ssm_conv1d,
        ssm_alpha,
        ssm_alpha_ggml_type,
        ssm_dt_bias,
        ssm_dt_bias_cpu,
        ssm_beta,
        ssm_beta_ggml_type,
        ssm_a,
        ssm_a_cpu,
        ssm_norm,
        ssm_norm_cpu,
        ssm_out,
        ssm_out_ggml_type,
    })
}

/// Load the global tensors (`token_embd`, `output`, `output_norm`) from
/// a GGUF into flat f32 vectors.
pub fn load_global_tensors(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    device: &MlxDevice,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let _ = cfg; // reserved for future shape validation
    let token_embd =
        load_f32_tensor(gguf, "token_embd.weight", device).context("token_embd.weight")?;
    let output_weight = load_f32_tensor(gguf, "output.weight", device).context("output.weight")?;
    let output_norm =
        load_f32_tensor(gguf, "output_norm.weight", device).context("output_norm.weight")?;
    Ok((token_embd, output_weight, output_norm))
}

// ============================================================================
// LazyTensorMap-backed model load (ADR-014 P4)
// ============================================================================

struct LazyQwen35Lookup<'a> {
    tensors: BTreeMap<String, &'a LazyTensor>,
}

impl<'a> LazyQwen35Lookup<'a> {
    fn new(map: &'a LazyTensorMap) -> Self {
        let mut tensors = BTreeMap::new();
        for (name, lazy) in map.iter() {
            let gguf_name = qwen35_lazy_name_to_gguf(name);
            tensors.entry(gguf_name).or_insert(lazy);
        }
        Self { tensors }
    }

    fn get(&self, name: &str) -> Result<&'a LazyTensor> {
        self.tensors
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("lazy qwen35 load: tensor '{name}' not found"))
    }

    fn maybe(&self, name: &str) -> Option<&'a LazyTensor> {
        self.tensors.get(name).copied()
    }
}

fn qwen35_lazy_name_to_gguf(name: &str) -> String {
    let name = name.replace("language_model.", "");
    match name.as_str() {
        "model.embed_tokens.weight" => return "token_embd.weight".to_string(),
        "model.norm.weight" => return "output_norm.weight".to_string(),
        "lm_head.weight" => return "output.weight".to_string(),
        _ => {}
    }

    let Some(rest) = name.strip_prefix("model.layers.") else {
        return name;
    };
    let Some(dot_pos) = rest.find('.') else {
        return name;
    };
    let layer_num = &rest[..dot_pos];
    if !layer_num.chars().all(|c| c.is_ascii_digit()) {
        return name;
    }
    let suffix = &rest[dot_pos + 1..];
    let mapped_suffix = match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
        "linear_attn.in_proj_z.weight" => "attn_gate.weight",
        "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
        "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
        "linear_attn.out_proj.weight" => "ssm_out.weight",
        "linear_attn.A_log" => "ssm_a",
        "linear_attn.dt_bias" | "linear_attn.dt_proj.bias" => "ssm_dt.bias",
        "linear_attn.conv1d.weight" => "ssm_conv1d.weight",
        "linear_attn.norm.weight" => "ssm_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "mlp.gate.weight" => "ffn_gate_inp.weight",
        "mlp.shared_expert_gate.weight" => "ffn_gate_inp_shexp.weight",
        "mlp.shared_expert.gate_proj.weight" => "ffn_gate_shexp.weight",
        "mlp.shared_expert.up_proj.weight" => "ffn_up_shexp.weight",
        "mlp.shared_expert.down_proj.weight" => "ffn_down_shexp.weight",
        "mlp.experts.gate_proj.weight" => "ffn_gate_exps.weight",
        "mlp.experts.up_proj.weight" => "ffn_up_exps.weight",
        "mlp.experts.down_proj.weight" => "ffn_down_exps.weight",
        "eh_proj.weight" => "nextn.eh_proj.weight",
        "enorm.weight" => "nextn.enorm.weight",
        "hnorm.weight" => "nextn.hnorm.weight",
        "embed_tokens.weight" => "nextn.embed_tokens.weight",
        other => other,
    };
    format!("blk.{layer_num}.{mapped_suffix}")
}

fn parse_blk_tensor(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let dot_pos = rest.find('.')?;
    let layer = rest[..dot_pos].parse::<u32>().ok()?;
    Some((layer, &rest[dot_pos + 1..]))
}

fn infer_lazy_qwen35_config(lookup: &LazyQwen35Lookup<'_>) -> Result<Qwen35Config> {
    let hidden_size = lookup
        .get("output_norm.weight")?
        .shape()
        .first()
        .copied()
        .ok_or_else(|| anyhow!("output_norm.weight has empty shape"))? as u32;
    let vocab_size = lookup
        .get("token_embd.weight")?
        .shape()
        .first()
        .copied()
        .ok_or_else(|| anyhow!("token_embd.weight has empty shape"))? as u32;

    let mut max_layer = None::<u32>;
    let mut has_moe = false;
    for name in lookup.tensors.keys() {
        if let Some((layer, suffix)) = parse_blk_tensor(name) {
            if !suffix.starts_with("nextn.") {
                max_layer = Some(max_layer.map_or(layer, |m| m.max(layer)));
            }
            if suffix == "ffn_gate_exps.weight" {
                has_moe = true;
            }
        }
    }
    let num_hidden_layers = max_layer
        .map(|layer| layer + 1)
        .ok_or_else(|| anyhow!("lazy qwen35 load: no blk.<layer> tensors found"))?;

    let mut layer_types = Vec::with_capacity(num_hidden_layers as usize);
    for layer in 0..num_hidden_layers {
        let full_name = format!("blk.{layer}.attn_q.weight");
        let linear_name = format!("blk.{layer}.attn_qkv.weight");
        let kind = if lookup.maybe(&full_name).is_some() {
            Qwen35LayerKind::FullAttention
        } else if lookup.maybe(&linear_name).is_some() {
            Qwen35LayerKind::LinearAttention
        } else {
            return Err(anyhow!(
                "lazy qwen35 load: layer {layer} has neither attn_q nor attn_qkv"
            ));
        };
        layer_types.push(kind);
    }

    let full_attention_interval = layer_types
        .iter()
        .position(|kind| *kind == Qwen35LayerKind::FullAttention)
        .map(|idx| idx as u32 + 1)
        .unwrap_or(0);
    let layer_types = if full_attention_interval > 0 {
        default_layer_types(num_hidden_layers, full_attention_interval)
    } else {
        layer_types
    };

    let full_layer_idx = layer_types
        .iter()
        .position(|kind| *kind == Qwen35LayerKind::FullAttention)
        .unwrap_or(0) as u32;
    let head_dim = lookup
        .maybe(&format!("blk.{full_layer_idx}.attn_q_norm.weight"))
        .and_then(|t| t.shape().first().copied())
        .or_else(|| {
            lookup
                .maybe(&format!("blk.{full_layer_idx}.attn_k_norm.weight"))
                .and_then(|t| t.shape().first().copied())
        })
        .unwrap_or(32) as u32;
    let attn_q_rows = lookup
        .maybe(&format!("blk.{full_layer_idx}.attn_q.weight"))
        .and_then(|t| t.shape().first().copied())
        .unwrap_or(head_dim as usize * 2);
    let q_rows = if attn_q_rows % 2 == 0 {
        attn_q_rows / 2
    } else {
        attn_q_rows
    };
    let num_attention_heads = ((q_rows as u32) / head_dim).max(1);
    let kv_rows = lookup
        .maybe(&format!("blk.{full_layer_idx}.attn_k.weight"))
        .and_then(|t| t.shape().first().copied())
        .unwrap_or(head_dim as usize);
    let num_key_value_heads = ((kv_rows as u32) / head_dim).max(1);

    let linear_layer_idx = layer_types
        .iter()
        .position(|kind| *kind == Qwen35LayerKind::LinearAttention)
        .unwrap_or(0) as u32;
    let linear_value_head_dim = lookup
        .maybe(&format!("blk.{linear_layer_idx}.ssm_norm.weight"))
        .and_then(|t| t.shape().first().copied())
        .unwrap_or(head_dim as usize) as u32;
    let linear_key_head_dim = linear_value_head_dim;
    let linear_num_value_heads = lookup
        .maybe(&format!("blk.{linear_layer_idx}.ssm_a"))
        .and_then(|t| t.shape().first().copied())
        .unwrap_or(num_key_value_heads as usize) as u32;
    let attn_qkv_rows = lookup
        .maybe(&format!("blk.{linear_layer_idx}.attn_qkv.weight"))
        .and_then(|t| t.shape().first().copied())
        .unwrap_or(
            (2 * num_key_value_heads * linear_key_head_dim
                + linear_num_value_heads * linear_value_head_dim) as usize,
        ) as u32;
    let v_rows = linear_num_value_heads * linear_value_head_dim;
    let linear_num_key_heads = if attn_qkv_rows > v_rows && linear_key_head_dim > 0 {
        ((attn_qkv_rows - v_rows) / (2 * linear_key_head_dim)).max(1)
    } else {
        num_key_value_heads.max(1)
    };
    let linear_conv_kernel_dim = lookup
        .maybe(&format!("blk.{linear_layer_idx}.ssm_conv1d.weight"))
        .and_then(|t| t.shape().last().copied())
        .unwrap_or(4) as u32;

    let variant = if has_moe {
        Qwen35Variant::Moe
    } else {
        Qwen35Variant::Dense
    };
    let (intermediate_size, moe) = match variant {
        Qwen35Variant::Dense => {
            let m = lookup
                .get("blk.0.ffn_gate.weight")?
                .shape()
                .first()
                .copied()
                .ok_or_else(|| anyhow!("blk.0.ffn_gate.weight has empty shape"))?
                as u32;
            (Some(m), None)
        }
        Qwen35Variant::Moe => {
            let expert_shape = lookup.get("blk.0.ffn_gate_exps.weight")?.shape();
            if expert_shape.len() < 3 {
                return Err(anyhow!(
                    "blk.0.ffn_gate_exps.weight shape {:?} is not [experts, inter, hidden]",
                    expert_shape
                ));
            }
            let shared_intermediate = lookup
                .get("blk.0.ffn_gate_shexp.weight")?
                .shape()
                .first()
                .copied()
                .ok_or_else(|| anyhow!("blk.0.ffn_gate_shexp.weight has empty shape"))?
                as u32;
            (
                None,
                Some(Qwen35MoeConfig {
                    num_experts: expert_shape[0] as u32,
                    moe_intermediate_size: expert_shape[1] as u32,
                    num_experts_per_tok: 1,
                    shared_expert_intermediate_size: shared_intermediate,
                }),
            )
        }
    };

    let rotary_dim = head_dim / 2;
    Ok(Qwen35Config {
        variant,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_conv_kernel_dim,
        full_attention_interval,
        layer_types,
        partial_rotary_factor: if head_dim == 0 {
            0.0
        } else {
            rotary_dim as f32 / head_dim as f32
        },
        rope_theta: 10_000_000.0,
        rotary_dim,
        mrope_section: [rotary_dim / 4, rotary_dim / 4, rotary_dim / 4, 0],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 131_072,
        vocab_size,
        attn_output_gate: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: true,
        intermediate_size,
        moe,
    })
}

fn tensor_ref_to_f32(mut tensor: TensorRef) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    match tensor.dtype {
        IrDType::F32 => f32_bytes_to_f32(&tensor.data, &mut out),
        IrDType::F16 => f16_bytes_to_f32(&tensor.data, &mut out),
        IrDType::BF16 => bf16_bytes_to_f32(&tensor.data, &mut out),
        other => {
            return Err(anyhow!(
                "tensor '{}' has dtype {:?}; expected F32/F16/BF16",
                tensor.name,
                other
            ));
        }
    }
    // ADR-014 P13 step 2 (iter-80): replace `tensor.data.clear()` (in-place
    // mutation) with assignment-replacement so the same code works whether
    // `tensor.data` is `Vec<u8>` (today) or `Arc<[u8]>` (post-iter-82+
    // P13 type migration).  The early-drop semantic is preserved — the
    // old Vec/Arc is dropped immediately when the assignment overwrites
    // the field.
    tensor.data = std::sync::Arc::new(Vec::new());
    Ok(out)
}

fn load_lazy_f32(lookup: &LazyQwen35Lookup<'_>, name: &str) -> Result<Vec<f32>> {
    let tensor = lookup
        .get(name)?
        .materialize_cloned()
        .with_context(|| format!("materialize {name}"))?;
    tensor_ref_to_f32(tensor).with_context(|| format!("convert {name} to f32"))
}

fn upload_lazy_raw_u8(
    lookup: &LazyQwen35Lookup<'_>,
    name: &str,
    device: &MlxDevice,
) -> Result<MlxBuffer> {
    let tensor = lookup
        .get(name)?
        .materialize_cloned()
        .with_context(|| format!("materialize {name}"))?;
    if tensor.dtype != IrDType::U8 {
        return Err(anyhow!(
            "tensor '{}' has dtype {:?}; expected U8 GGML block bytes",
            tensor.name,
            tensor.dtype
        ));
    }
    let mut buf = device
        .alloc_buffer(tensor.data.len(), MlxDType::U8, tensor.shape.clone())
        .map_err(|e| anyhow!("alloc U8 buffer for {name}: {e}"))?;
    {
        let dst: &mut [u8] = buf
            .as_mut_slice()
            .map_err(|e| anyhow!("as_mut_slice({name}): {e}"))?;
        dst.copy_from_slice(&tensor.data);
    }
    // W-5b.7 iter 2: register lazy-loaded raw U8 weight blocks with the
    // weight pool's residency set.
    super::weight_pool::register_weight_buffer(device, &buf)
        .map_err(|e| anyhow!("register_weight_buffer({name}): {e}"))?;
    Ok(buf)
}

fn load_lazy_expert_q8_0(
    lookup: &LazyQwen35Lookup<'_>,
    name: &str,
    device: &MlxDevice,
) -> Result<MlxBuffer> {
    let lazy = lookup.get(name)?;
    if lazy.dtype() == IrDType::U8 {
        return upload_lazy_raw_u8(lookup, name, device);
    }
    let shape = lazy.shape().to_vec();
    let f32_data = tensor_ref_to_f32(
        lazy.materialize_cloned()
            .with_context(|| format!("materialize {name}"))?,
    )
    .with_context(|| format!("convert {name} expert tensor to f32"))?;
    quantize_f32_to_q8_0_buffer(&f32_data, shape, device)
        .with_context(|| format!("Q8_0 upload for {name}"))
}

fn load_lazy_full_attn_layer(
    lookup: &LazyQwen35Lookup<'_>,
    cfg: &Qwen35Config,
    layer_idx: u32,
) -> Result<FullAttnLayerWeights> {
    let p = format!("blk.{}", layer_idx);
    let attn_norm = load_lazy_f32(lookup, &format!("{p}.attn_norm.weight"))?;
    let q_fused = load_lazy_f32(lookup, &format!("{p}.attn_q.weight"))?;
    let wk = load_lazy_f32(lookup, &format!("{p}.attn_k.weight"))?;
    let wv = load_lazy_f32(lookup, &format!("{p}.attn_v.weight"))?;
    let attn_q_norm = load_lazy_f32(lookup, &format!("{p}.attn_q_norm.weight"))?;
    let attn_k_norm = load_lazy_f32(lookup, &format!("{p}.attn_k_norm.weight"))?;
    let wo = load_lazy_f32(lookup, &format!("{p}.attn_output.weight"))?;
    let post_attn_norm = load_lazy_f32(lookup, &format!("{p}.post_attention_norm.weight"))?;

    let h = cfg.hidden_size as usize;
    let nh = cfg.num_attention_heads as usize;
    let nkv = cfg.num_key_value_heads as usize;
    let d = cfg.head_dim as usize;
    let q_total = nh * d;
    let kv_total = nkv * d;
    if q_fused.len() != 2 * q_total * h {
        return Err(anyhow!(
            "fused attn_q layer {layer_idx}: got {} floats, expected {}",
            q_fused.len(),
            2 * q_total * h
        ));
    }
    if wk.len() != kv_total * h || wv.len() != kv_total * h {
        return Err(anyhow!("layer {layer_idx}: K/V shape mismatch"));
    }

    let mut wq = vec![0.0f32; q_total * h];
    let mut w_gate = vec![0.0f32; q_total * h];
    for head_idx in 0..nh {
        let src_q_start = (head_idx * 2 * d) * h;
        let src_g_start = ((head_idx * 2 + 1) * d) * h;
        let dst_start = head_idx * d * h;
        wq[dst_start..dst_start + d * h]
            .copy_from_slice(&q_fused[src_q_start..src_q_start + d * h]);
        w_gate[dst_start..dst_start + d * h]
            .copy_from_slice(&q_fused[src_g_start..src_g_start + d * h]);
    }

    Ok(FullAttnLayerWeights {
        attn_norm,
        post_attn_norm,
        wq,
        wk,
        wv,
        w_gate,
        attn_q_norm,
        attn_k_norm,
        wo,
    })
}

fn load_lazy_delta_net_layer(
    lookup: &LazyQwen35Lookup<'_>,
    cfg: &Qwen35Config,
    layer_idx: u32,
) -> Result<DeltaNetLayerWeights> {
    let p = format!("blk.{}", layer_idx);
    let attn_norm = load_lazy_f32(lookup, &format!("{p}.attn_norm.weight"))?;
    let post_attn_norm = load_lazy_f32(lookup, &format!("{p}.post_attention_norm.weight"))?;
    let attn_qkv = load_lazy_f32(lookup, &format!("{p}.attn_qkv.weight"))?;
    let attn_gate = load_lazy_f32(lookup, &format!("{p}.attn_gate.weight"))?;
    let ssm_conv1d_gguf = load_lazy_f32(lookup, &format!("{p}.ssm_conv1d.weight"))?;
    let ssm_alpha = load_lazy_f32(lookup, &format!("{p}.ssm_alpha.weight"))?;
    let ssm_dt_bias = load_lazy_f32(lookup, &format!("{p}.ssm_dt.bias"))?;
    let ssm_beta = load_lazy_f32(lookup, &format!("{p}.ssm_beta.weight"))?;
    let ssm_a = load_lazy_f32(lookup, &format!("{p}.ssm_a"))?;
    let ssm_norm = load_lazy_f32(lookup, &format!("{p}.ssm_norm.weight"))?;
    let ssm_out = load_lazy_f32(lookup, &format!("{p}.ssm_out.weight"))?;

    let nk = cfg.linear_num_key_heads as usize;
    let nv = cfg.linear_num_value_heads as usize;
    let dk = cfg.linear_key_head_dim as usize;
    let dv = cfg.linear_value_head_dim as usize;
    let k_width = cfg.linear_conv_kernel_dim as usize;
    let qkv_channels = 2 * nk * dk + nv * dv;
    if ssm_conv1d_gguf.len() != qkv_channels * k_width {
        return Err(anyhow!("layer {layer_idx}: ssm_conv1d shape mismatch"));
    }
    let mut ssm_conv1d = vec![0.0f32; k_width * qkv_channels];
    for c in 0..qkv_channels {
        for ki in 0..k_width {
            ssm_conv1d[ki * qkv_channels + c] = ssm_conv1d_gguf[c * k_width + ki];
        }
    }

    Ok(DeltaNetLayerWeights {
        attn_norm,
        post_attn_norm,
        attn_qkv,
        attn_gate,
        ssm_conv1d,
        ssm_alpha,
        ssm_dt_bias,
        ssm_beta,
        ssm_a,
        ssm_norm,
        ssm_out,
    })
}

fn load_lazy_dense_ffn(lookup: &LazyQwen35Lookup<'_>, layer_idx: u32) -> Result<DenseFfnWeights> {
    let p = format!("blk.{}", layer_idx);
    Ok(DenseFfnWeights {
        gate: load_lazy_f32(lookup, &format!("{p}.ffn_gate.weight"))?,
        up: load_lazy_f32(lookup, &format!("{p}.ffn_up.weight"))?,
        down: load_lazy_f32(lookup, &format!("{p}.ffn_down.weight"))?,
    })
}

fn load_lazy_moe_ffn_quantized(
    lookup: &LazyQwen35Lookup<'_>,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<MoeFfnWeightsQ> {
    let p = format!("blk.{}", layer_idx);
    let as_test_f32_matrix = |name: &str, rows: usize, cols: usize| -> Result<MlxQWeight> {
        let values = load_lazy_f32(lookup, name)?;
        anyhow::ensure!(
            values.len() == rows * cols,
            "lazy matrix {name} has {} values, expected {} for [{rows},{cols}]",
            values.len(),
            rows * cols
        );
        let mut buffer = device
            .alloc_buffer(values.len() * 4, MlxDType::F32, vec![rows, cols])
            .map_err(|error| anyhow!("allocate lazy matrix {name}: {error}"))?;
        buffer
            .as_mut_slice::<f32>()
            .map_err(|error| anyhow!("map lazy matrix {name}: {error}"))?
            .copy_from_slice(&values);
        Ok(MlxQWeight {
            buffer,
            info: crate::serve::gpu::QuantWeightInfo {
                ggml_dtype: GgmlType::F32,
                rows,
                cols,
            },
            affine: None,
            decode_record_q6k_m1: std::sync::OnceLock::new(),
        })
    };
    let cfg = infer_lazy_qwen35_config(lookup)?;
    let moe = cfg
        .moe
        .as_ref()
        .context("lazy MoE matrix load requires MoE config")?;
    let h = cfg.hidden_size as usize;
    let ne = moe.num_experts as usize;
    let shared = moe.shared_expert_intermediate_size as usize;
    Ok(MoeFfnWeightsQ {
        router: as_test_f32_matrix(&format!("{p}.ffn_gate_inp.weight"), ne, h)?,
        expert_gate_q: load_lazy_expert_q8_0(lookup, &format!("{p}.ffn_gate_exps.weight"), device)?,
        expert_up_q: load_lazy_expert_q8_0(lookup, &format!("{p}.ffn_up_exps.weight"), device)?,
        expert_down_q: load_lazy_expert_q8_0(lookup, &format!("{p}.ffn_down_exps.weight"), device)?,
        ggml_type_gate: GgmlType::Q8_0,
        ggml_type_up: GgmlType::Q8_0,
        ggml_type_down: GgmlType::Q8_0,
        shared_gate_logit: as_test_f32_matrix(&format!("{p}.ffn_gate_inp_shexp.weight"), 1, h)?,
        shared_gate: as_test_f32_matrix(&format!("{p}.ffn_gate_shexp.weight"), shared, h)?,
        shared_up: as_test_f32_matrix(&format!("{p}.ffn_up_shexp.weight"), shared, h)?,
        shared_down: as_test_f32_matrix(&format!("{p}.ffn_down_shexp.weight"), h, shared)?,
        expert_gate_affine: None,
        expert_up_affine: None,
        expert_down_affine: None,
    })
}

impl Qwen35Model {
    /// Load a complete Qwen3.5 model directly from a transformed lazy tensor
    /// map, without emitting and re-reading an intermediate GGUF.
    ///
    /// The map may contain either post-transform HF tensor names or the GGUF
    /// names used by the inference loader. Iteration is deterministic through
    /// `LazyTensorMap`'s `BTreeMap`; each requested tensor is materialized,
    /// converted/uploaded, and then dropped before the next tensor is loaded.
    pub fn load_from_lazy_tensor_map(model: &LazyTensorMap) -> Result<Self> {
        let lookup = LazyQwen35Lookup::new(model);
        let mut cfg = infer_lazy_qwen35_config(&lookup)?;
        let device = super::forward_gpu::new_model_load_device()
            .map_err(|e| anyhow!("MlxDevice::new for lazy qwen35 loading: {e}"))?;

        let mut token_embd = load_lazy_f32(&lookup, "token_embd.weight")?;
        let output_weight = load_lazy_f32(&lookup, "output.weight")?;
        let output_norm = load_lazy_f32(&lookup, "output_norm.weight")?;

        let h = cfg.hidden_size as usize;
        if h > 0 {
            let physical_vocab = token_embd.len() / h;
            if (physical_vocab as u32) != cfg.vocab_size {
                cfg.vocab_size = physical_vocab as u32;
            }
        }

        if h > 0 {
            const QWEN35_FULL_VOCAB: u32 = 248_320;
            let current_vocab = cfg.vocab_size;
            if current_vocab < QWEN35_FULL_VOCAB && (QWEN35_FULL_VOCAB - current_vocab) < 2048 {
                token_embd.resize(QWEN35_FULL_VOCAB as usize * h, 0.0f32);
            }
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            let kind = cfg
                .layer_types
                .get(i as usize)
                .copied()
                .ok_or_else(|| anyhow!("layer_idx {i} out of range"))?;
            let ffn = match cfg.variant {
                Qwen35Variant::Dense => Qwen35FfnWeights::Dense(load_lazy_dense_ffn(&lookup, i)?),
                Qwen35Variant::Moe => {
                    Qwen35FfnWeights::MoeQ(load_lazy_moe_ffn_quantized(&lookup, i, &device)?)
                }
            };
            let layer = match kind {
                Qwen35LayerKind::FullAttention => Qwen35LayerWeights::FullAttn {
                    attn: load_lazy_full_attn_layer(&lookup, &cfg, i)?,
                    ffn,
                },
                Qwen35LayerKind::LinearAttention => Qwen35LayerWeights::LinearAttn {
                    attn: load_lazy_delta_net_layer(&lookup, &cfg, i)?,
                    ffn,
                },
            };
            layers.push(layer);
        }

        Ok(Self {
            activation_epoch: Self::next_activation_epoch(),
            native_routes_activated: std::sync::atomic::AtomicBool::new(false),
            ggml_routing_policy: mlx_native::ggml_routing_policy_from_environment(),
            cfg,
            layers,
            token_embd,
            token_embd_native: None,
            output_weight,
            output_weight_native: None,
            tied_word_embeddings: false,
            output_norm,
            output_norm_native: None,
            mtp: None,
            #[cfg(test)]
            loaded_candidate_identity: None,
        })
    }
}

/// Load a single full-attention layer's weights from the GGUF.
pub fn load_full_attn_layer(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<FullAttnLayerWeights> {
    let p = format!("blk.{}", layer_idx);
    let attn_norm = load_f32_tensor(gguf, &format!("{p}.attn_norm.weight"), device)
        .with_context(|| format!("layer {layer_idx} attn_norm"))?;

    // APEX LAYOUT DISCOVERY: full-attention layers have a FUSED `attn_q.weight`
    // holding Q + output-gate in sequence (shape `[2 * n_head * head_dim, hidden]`),
    // not a separate `attn_gate.weight` tensor. This matches the peer's in-memory
    // `wq` convention where `wq` has output dim `2 * head_dim * n_head` with Q
    // in the lower half and gate in the upper half. Our CPU reference keeps wq and
    // w_gate separate, so we split after loading.
    // not a separate `attn_gate.weight` tensor. Rows are interleaved per head;
    // the CPU reference keeps Q and gate separate, so we split after loading.
    let q_fused = load_f32_tensor(gguf, &format!("{p}.attn_q.weight"), device)
        .with_context(|| format!("layer {layer_idx} attn_q (fused Q+gate)"))?;

    let wk = load_f32_tensor(gguf, &format!("{p}.attn_k.weight"), device)
        .with_context(|| format!("layer {layer_idx} attn_k"))?;
    let wv = load_f32_tensor(gguf, &format!("{p}.attn_v.weight"), device)
        .with_context(|| format!("layer {layer_idx} attn_v"))?;
    let attn_q_norm = load_f32_tensor(gguf, &format!("{p}.attn_q_norm.weight"), device)
        .with_context(|| format!("layer {layer_idx} attn_q_norm"))?;
    let attn_k_norm = load_f32_tensor(gguf, &format!("{p}.attn_k_norm.weight"), device)
        .with_context(|| format!("layer {layer_idx} attn_k_norm"))?;
    let wo = load_f32_tensor(gguf, &format!("{p}.attn_output.weight"), device)
        .with_context(|| format!("layer {layer_idx} attn_output"))?;
    let post_attn_norm = load_f32_tensor(gguf, &format!("{p}.post_attention_norm.weight"), device)
        .with_context(|| format!("layer {layer_idx} post_attention_norm"))?;

    // Sanity check shapes.
    let h = cfg.hidden_size as usize;
    let nh = cfg.num_attention_heads as usize;
    let nkv = cfg.num_key_value_heads as usize;
    let d = cfg.head_dim as usize;
    let q_total = nh * d;
    let kv_total = nkv * d;
    assert_eq!(attn_norm.len(), h, "attn_norm layer {layer_idx} shape");
    assert_eq!(
        q_fused.len(),
        2 * q_total * h,
        "fused attn_q layer {layer_idx}: got {} floats, expected 2 * n_head*d * hidden = {}",
        q_fused.len(),
        2 * q_total * h
    );
    assert_eq!(wk.len(), kv_total * h, "attn_k layer {layer_idx} shape");
    assert_eq!(wv.len(), kv_total * h, "attn_v layer {layer_idx} shape");
    assert_eq!(attn_q_norm.len(), d, "attn_q_norm layer {layer_idx} shape");
    assert_eq!(attn_k_norm.len(), d, "attn_k_norm layer {layer_idx} shape");
    assert_eq!(wo.len(), h * q_total, "attn_output layer {layer_idx} shape");
    assert_eq!(
        post_attn_norm.len(),
        h,
        "post_attn_norm layer {layer_idx} shape"
    );

    // De-interleave fused q_fused into wq and w_gate.
    // The peer's layout (confirmed from build_layer_attn): Q and gate are INTERLEAVED
    // De-interleave fused q_fused into wq and w_gate. Q and gate are interleaved
    // at head granularity. For head h: rows [2*h*d .. (2*h+1)*d-1] = Q[h], rows
    // [(2*h+1)*d .. (2*h+2)*d-1] = gate[h]. Each "row" is h (hidden_size) floats wide.
    // So in the flat vec: head h Q starts at offset (2*h*d)*h, gate starts at (2*h+1)*d*h.
    let mut wq = vec![0.0f32; q_total * h];
    let mut w_gate = vec![0.0f32; q_total * h];
    for head_idx in 0..nh {
        let src_q_start = (head_idx * 2 * d) * h;
        let src_g_start = ((head_idx * 2 + 1) * d) * h;
        let dst_start = head_idx * d * h;
        wq[dst_start..dst_start + d * h]
            .copy_from_slice(&q_fused[src_q_start..src_q_start + d * h]);
        w_gate[dst_start..dst_start + d * h]
            .copy_from_slice(&q_fused[src_g_start..src_g_start + d * h]);
    }
    drop(q_fused);

    Ok(FullAttnLayerWeights {
        attn_norm,
        post_attn_norm,
        wq,
        wk,
        wv,
        w_gate,
        attn_q_norm,
        attn_k_norm,
        wo,
    })
}

/// Load a single linear-attention (DeltaNet) layer's weights.
///
/// # V-head ordering (tiled, matches mlx-native 0.4.1 fused kernel)
///
/// `convert_hf_to_gguf.py`'s `_LinearAttentionVReorderBase._reorder_v_heads`
/// permutes V-head dimensions from HF "grouped" order `[n_k, n_vpk, d]` to
/// GGUF "tiled" order `[n_vpk, n_k, d]` (i.e. `v_head = i_vpk * n_k + i_k`)
/// so that ggml's broadcast semantics align K and V heads when the fused
/// GDN op is enabled (`fused_gdn_ar` / `fused_gdn_ch` paths in
/// `qwen35moe.cpp::build_layer_attn_linear`).
///
/// The peer's fused GDN kernel — and now mlx-native's `gated_delta_net_f32`
/// kernel as of mlx-native 0.4.1 (commit `4f00f6e`) — performs the GQA
/// mlx-native's `gated_delta_net_f32` kernel performs the GQA
/// mapping internally as `k_head = v_head % n_k_heads`, which is the
/// inverse of the GGUF tiling: with `v_head = i_vpk * n_k + i_k`,
/// `v_head % n_k = i_k`, recovering the correct K-head for any V-head.
///
/// Therefore: every V-head-axis tensor MUST be left in the GGUF's natural
/// tiled order. No un-reordering. Earlier hf2q snapshots un-reordered to
/// "grouped" order to compensate for an old (block-style) mlx-native kernel
/// that used `k_head = v_head / group_ratio`; that kernel was retired in
/// `4f00f6e` to reach byte-parity with the peer.
/// `4f00f6e` after the byte-parity gate failed.
///
/// Affected tensors that stay in GGUF tiled V-head order (apex GGUF:
/// n_k=16, n_vpk=2, d_v=128, hidden=2048):
/// - `attn_qkv.weight`     (V rows only, the trailing `n_v * d_v` rows)
/// - `attn_gate.weight`    (all rows, `[n_v * d_v, hidden]`)
/// - `ssm_alpha.weight`    (all rows, `[n_v, hidden]`)
/// - `ssm_beta.weight`     (all rows, `[n_v, hidden]`)
/// - `ssm_a`               (1-D `[n_v]`)
/// - `ssm_dt.bias`         (1-D `[n_v]`)
/// - `ssm_conv1d.weight`   (V channels only)
/// - `ssm_out.weight`      (V-head column blocks of the `[hidden, n_v * d_v]`
///                          shape)
pub fn load_delta_net_layer(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<DeltaNetLayerWeights> {
    let p = format!("blk.{}", layer_idx);

    // Key dimensions.
    let nk = cfg.linear_num_key_heads as usize;
    let nv = cfg.linear_num_value_heads as usize;
    let dk = cfg.linear_key_head_dim as usize;
    let dv = cfg.linear_value_head_dim as usize;
    let h = cfg.hidden_size as usize;
    let k_width = cfg.linear_conv_kernel_dim as usize;
    let qkv_channels = 2 * nk * dk + nv * dv;

    let attn_norm = load_f32_tensor(gguf, &format!("{p}.attn_norm.weight"), device)?;
    let post_attn_norm = load_f32_tensor(gguf, &format!("{p}.post_attention_norm.weight"), device)
        .with_context(|| format!("layer {layer_idx} post_attention_norm"))?;

    // ---- attn_qkv ----
    // GGUF shape: [qkv_total, hidden] = [(2*nk*dk + nv*dv), h].
    // V rows are in GGUF tiled order (`v_head = i_vpk * n_k + i_k`); Q/K rows
    // are unchanged. The mlx-native fused GDN kernel maps `k_head = v_head %
    // n_k_heads` internally, so this layout is consumed directly — no reorder.
    let attn_qkv = load_f32_tensor(gguf, &format!("{p}.attn_qkv.weight"), device)?;
    let qk_rows = 2 * nk * dk;
    let v_rows = nv * dv;
    assert_eq!(attn_qkv.len(), (qk_rows + v_rows) * h, "attn_qkv shape");

    // ---- attn_gate ----
    // GGUF shape: [nv*dv, h] in tiled V-head order. Consumed by op-8 (output
    // gate) which multiplies element-wise with the GDN output; the GDN output
    // inherits V's tiled order, so the gate must match — i.e. stay tiled.
    let attn_gate = load_f32_tensor(gguf, &format!("{p}.attn_gate.weight"), device)?;
    assert_eq!(attn_gate.len(), nv * dv * h, "attn_gate shape");

    // ---- ssm_conv1d ----
    // GGUF layout: [channels, K] (channels = qkv_channels). V-channels are in
    // tiled order, matching attn_qkv. Per-channel conv, so channel ordering
    // is opaque to the kernel — we just transpose from [channels, K] to
    // [K, channels] (the order the CPU reference's `ssm_conv_scalar` reads).
    let ssm_conv1d_gguf = load_f32_tensor(gguf, &format!("{p}.ssm_conv1d.weight"), device)?;
    assert_eq!(
        ssm_conv1d_gguf.len(),
        qkv_channels * k_width,
        "ssm_conv1d shape"
    );
    let ssm_conv1d = {
        // Transpose [channels, K] → [K, channels].
        let mut out = vec![0.0f32; k_width * qkv_channels];
        for c in 0..qkv_channels {
            for ki in 0..k_width {
                out[ki * qkv_channels + c] = ssm_conv1d_gguf[c * k_width + ki];
            }
        }
        out
    };
    drop(ssm_conv1d_gguf);

    // ---- ssm_alpha ----
    // GGUF shape: [nv, h] in tiled V-head order. Produces `g[t, vh]` consumed
    // by GDN — must share the kernel's V-head order (= GGUF tiled).
    let ssm_alpha = load_f32_tensor(gguf, &format!("{p}.ssm_alpha.weight"), device)?;
    assert_eq!(ssm_alpha.len(), nv * h, "ssm_alpha shape");

    // ---- ssm_dt_bias ----
    // GGUF shape: [nv] in tiled order. Added per-V-head before softplus to
    // produce `g`; consumed by GDN.
    let ssm_dt_bias = load_f32_tensor(gguf, &format!("{p}.ssm_dt.bias"), device)?;
    assert_eq!(ssm_dt_bias.len(), nv, "ssm_dt_bias shape");

    // ---- ssm_beta ----
    // GGUF shape: [nv, h] in tiled V-head order. Produces `beta[t, vh]`.
    let ssm_beta = load_f32_tensor(gguf, &format!("{p}.ssm_beta.weight"), device)?;
    assert_eq!(ssm_beta.len(), nv * h, "ssm_beta shape");

    // ---- ssm_a ----
    // GGUF shape: [nv] in tiled order. Per-V-head decay base.
    let ssm_a = load_f32_tensor(gguf, &format!("{p}.ssm_a"), device)?;
    assert_eq!(ssm_a.len(), nv, "ssm_a shape");

    // ---- ssm_norm ----
    // GGUF shape: [dv] (one norm shared across all V-heads — NOT [nv*dv]).
    // Per-element broadcast across heads — head ordering is irrelevant.
    let ssm_norm = load_f32_tensor(gguf, &format!("{p}.ssm_norm.weight"), device)?;

    // ---- ssm_out ----
    // GGUF shape: [hidden, nv*dv] (output projection). The column dimension
    // is in tiled V-head order, matching the GDN output's V-head order. Keep
    // as-is — the projection then mixes V-heads back into the residual stream.
    let ssm_out = load_f32_tensor(gguf, &format!("{p}.ssm_out.weight"), device)?;
    assert_eq!(ssm_out.len(), h * nv * dv, "ssm_out shape");

    Ok(DeltaNetLayerWeights {
        attn_norm,
        post_attn_norm,
        attn_qkv,
        attn_gate,
        ssm_conv1d,
        ssm_alpha,
        ssm_dt_bias,
        ssm_beta,
        ssm_a,
        ssm_norm,
        ssm_out,
    })
}

/// Load an MoE FFN layer's weights.
pub fn load_moe_ffn(gguf: &GgufFile, layer_idx: u32, device: &MlxDevice) -> Result<MoeFfnWeights> {
    let p = format!("blk.{}", layer_idx);
    let router = load_f32_tensor(gguf, &format!("{p}.ffn_gate_inp.weight"), device)?;
    let expert_gate = load_f32_tensor(gguf, &format!("{p}.ffn_gate_exps.weight"), device)?;
    let expert_up = load_f32_tensor(gguf, &format!("{p}.ffn_up_exps.weight"), device)?;
    let expert_down = load_f32_tensor(gguf, &format!("{p}.ffn_down_exps.weight"), device)?;
    let shared_gate_logit =
        load_f32_tensor(gguf, &format!("{p}.ffn_gate_inp_shexp.weight"), device)?;
    let shared_gate = load_f32_tensor(gguf, &format!("{p}.ffn_gate_shexp.weight"), device)?;
    let shared_up = load_f32_tensor(gguf, &format!("{p}.ffn_up_shexp.weight"), device)?;
    let shared_down = load_f32_tensor(gguf, &format!("{p}.ffn_down_shexp.weight"), device)?;
    Ok(MoeFfnWeights {
        router,
        expert_gate,
        expert_up,
        expert_down,
        shared_gate_logit,
        shared_gate,
        shared_up,
        shared_down,
    })
}

/// Load an MoE FFN layer's weights, keeping expert tensors in their native
/// GGML quantization (e.g. Q6_K).
///
/// Expert stacks (`ffn_{gate,up,down}_exps`) retain exact file-backed GGML
/// views. Router and shared-expert matrices use the same mapped native matrix
/// representation and dispatch directly from their declared storage type.
/// No production MoE matrix is dequantized or re-encoded during load.
pub fn load_moe_ffn_quantized(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    cfg: &Qwen35Config,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<MoeFfnWeightsQ> {
    let p = format!("blk.{}", layer_idx);
    let moe = cfg
        .moe
        .as_ref()
        .context("native MoE load requires MoE configuration")?;
    let h = cfg.hidden_size as usize;
    let ne = moe.num_experts as usize;
    let expert = moe.moe_intermediate_size as usize;
    let shared = moe.shared_expert_intermediate_size as usize;
    let load_matrix = |name: &str, rows: usize, cols: usize| -> Result<MlxQWeight> {
        let info = gguf
            .tensor_info(name)
            .ok_or_else(|| anyhow!("layer {layer_idx}: {name} not found in GGUF"))?;
        anyhow::ensure!(
            info.shape.as_slice() == [rows, cols],
            "layer {layer_idx}: {name} shape {:?} != [{rows},{cols}]",
            info.shape
        );
        let weight = MlxQWeight::from_mapped_gguf_tensor(mapped, info)
            .with_context(|| format!("layer {layer_idx}: retain {name}"))?;
        super::weight_pool::register_weight_buffer(device, &weight.buffer)
            .with_context(|| format!("register {name}"))?;
        #[cfg(test)]
        super::execution_observation::observe_loaded_ggml(name, &weight.buffer)?;
        Ok(weight)
    };
    let router = load_matrix(&format!("{p}.ffn_gate_inp.weight"), ne, h)?;
    let shared_gate_logit = load_native_row_projection(
        gguf,
        mapped,
        &format!("{p}.ffn_gate_inp_shexp.weight"),
        h,
        device,
    )?;
    let shared_gate = load_matrix(&format!("{p}.ffn_gate_shexp.weight"), shared, h)?;
    let shared_up = load_matrix(&format!("{p}.ffn_up_shexp.weight"), shared, h)?;
    let shared_down = load_matrix(&format!("{p}.ffn_down_shexp.weight"), h, shared)?;

    // Expert weights: load raw GGML blocks, preserving quantization.
    // Residency registration retains the same mapped Metal allocation.
    let gate_name = format!("{p}.ffn_gate_exps.weight");
    let up_name = format!("{p}.ffn_up_exps.weight");
    let down_name = format!("{p}.ffn_down_exps.weight");
    let gate_info = gguf
        .tensor_info(&gate_name)
        .ok_or_else(|| anyhow!("layer {layer_idx}: ffn_gate_exps not found in GGUF"))?;
    let up_info = gguf
        .tensor_info(&up_name)
        .ok_or_else(|| anyhow!("layer {layer_idx}: ffn_up_exps not found in GGUF"))?;
    let down_info = gguf
        .tensor_info(&down_name)
        .ok_or_else(|| anyhow!("layer {layer_idx}: ffn_down_exps not found in GGUF"))?;
    anyhow::ensure!(
        gate_info.shape.as_slice() == [ne, expert, h]
            && up_info.shape.as_slice() == [ne, expert, h]
            && down_info.shape.as_slice() == [ne, h, expert],
        "layer {layer_idx}: malformed expert stack shapes gate={:?} up={:?} down={:?}",
        gate_info.shape,
        up_info.shape,
        down_info.shape
    );
    let expert_gate_q = map_tensor_with_residency(mapped, gate_info, device)
        .with_context(|| format!("layer {layer_idx} ffn_gate_exps (native mapped)"))?;
    let expert_up_q = map_tensor_with_residency(mapped, up_info, device)
        .with_context(|| format!("layer {layer_idx} ffn_up_exps (native mapped)"))?;
    let expert_down_q = map_tensor_with_residency(mapped, down_info, device)
        .with_context(|| format!("layer {layer_idx} ffn_down_exps (native mapped)"))?;

    // Every expert projection retains and dispatches from its own declared
    // artifact codec. Gate/up fusion is an execution optimization and is
    // considered later only when both codecs are equal and natively fused.
    let ggml_type_gate = gate_info.ggml_type;
    let ggml_type_up = up_info.ggml_type;
    let ggml_type_down = down_info.ggml_type;

    let supported = qwen35_moe_expert_type_supported;

    // Validate that every stored type has an artifact-native expert-ID route.
    // Scalar and block codecs remain in their mapped GGUF representation.
    if !supported(ggml_type_gate) {
        return Err(anyhow!(
            "layer {layer_idx}: gate expert weights have unsupported quant type {:?}",
            ggml_type_gate
        ));
    }
    if !supported(ggml_type_up) {
        return Err(anyhow!(
            "layer {layer_idx}: up expert weights have unsupported quant type {:?}",
            ggml_type_up
        ));
    }
    if !supported(ggml_type_down) {
        return Err(anyhow!(
            "layer {layer_idx}: down expert weights have unsupported quant type {:?}",
            ggml_type_down
        ));
    }

    Ok(MoeFfnWeightsQ {
        router,
        expert_gate_q,
        expert_up_q,
        expert_down_q,
        ggml_type_gate,
        ggml_type_up,
        ggml_type_down,
        shared_gate_logit,
        shared_gate,
        shared_up,
        shared_down,
        expert_gate_affine: None,
        expert_up_affine: None,
        expert_down_affine: None,
    })
}

pub(crate) fn qwen35_moe_expert_type_supported(t: GgmlType) -> bool {
    matches!(
        t,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_0
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::IQ4_NL
            // ADR-033 §Pi Task #18 2026-05-22 — IQ4_XS mv_id ported
            // at mlx-native@ff13e58. Required for apex-quality /
            // apex-i-quality MoE GGUFs (mudler's canonical mid-layer
            // expert quant).
            | GgmlType::IQ4_XS
    )
}

/// Load a dense FFN layer's weights.
pub fn load_dense_ffn(
    gguf: &GgufFile,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<DenseFfnWeights> {
    let p = format!("blk.{}", layer_idx);
    let gate = load_f32_tensor(gguf, &format!("{p}.ffn_gate.weight"), device)?;
    let up = load_f32_tensor(gguf, &format!("{p}.ffn_up.weight"), device)?;
    let down = load_f32_tensor(gguf, &format!("{p}.ffn_down.weight"), device)?;
    Ok(DenseFfnWeights { gate, up, down })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DenseFfnStorage {
    NativeScalar,
    Quantized,
    MixedNative,
}

pub(crate) fn qwen35_dense_ffn_type_supported(t: GgmlType) -> bool {
    matches!(t, GgmlType::F16 | GgmlType::BF16 | GgmlType::F32)
        || qwen35_dense_ffn_quant_type_supported(t)
}

fn qwen35_dense_ffn_quant_type_supported(t: GgmlType) -> bool {
    matches!(
        t,
        GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_0
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::IQ4_NL
            | GgmlType::IQ4_XS
    )
}

pub(super) fn dense_ffn_storage(
    layer_idx: u32,
    gate: GgmlType,
    up: GgmlType,
    down: GgmlType,
) -> Result<DenseFfnStorage> {
    let is_scalar = |t: GgmlType| matches!(t, GgmlType::F16 | GgmlType::BF16 | GgmlType::F32);
    let is_supported_quant = qwen35_dense_ffn_quant_type_supported;

    if is_scalar(gate) && is_scalar(up) && is_scalar(down) {
        return Ok(DenseFfnStorage::NativeScalar);
    }
    if is_supported_quant(gate) && is_supported_quant(up) && is_supported_quant(down) {
        return Ok(DenseFfnStorage::Quantized);
    }
    if [gate, up, down]
        .into_iter()
        .all(|storage| is_scalar(storage) || is_supported_quant(storage))
    {
        return Ok(DenseFfnStorage::MixedNative);
    }

    Err(anyhow!(
        "layer {layer_idx}: unsupported dense FFN storage: \
         gate={gate:?}, up={up:?}, down={down:?}; refusing a silent transform"
    ))
}

/// Shared admission boundary for bounded hosted preflight and the runtime
/// loader. Keeping the relation here prevents either caller from accepting a
/// gate/up dispatch mismatch or rejecting the all-float fixture/runtime path.
pub(crate) fn validate_qwen35_dense_ffn_storage(
    layer_idx: u32,
    gate: GgmlType,
    up: GgmlType,
    down: GgmlType,
) -> Result<()> {
    dense_ffn_storage(layer_idx, gate, up, down).map(|_| ())
}

/// Load native scalar dense-FFN projections without changing their storage
/// dtype or allocating a dequantized/re-encoded shadow.
pub fn load_dense_ffn_native(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    layer_idx: u32,
    cfg: &Qwen35Config,
    device: &MlxDevice,
) -> Result<DenseFfnWeightsNative> {
    let p = format!("blk.{layer_idx}");
    let hidden_size = cfg.hidden_size;
    let intermediate_size = cfg
        .intermediate_size
        .ok_or_else(|| anyhow!("layer {layer_idx}: dense FFN but intermediate size is absent"))?;
    let (gate, gate_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.ffn_gate.weight"),
        intermediate_size as usize,
        hidden_size as usize,
        device,
    )?;
    let (up, up_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.ffn_up.weight"),
        intermediate_size as usize,
        hidden_size as usize,
        device,
    )?;
    let (down, down_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.ffn_down.weight"),
        hidden_size as usize,
        intermediate_size as usize,
        device,
    )?;
    Ok(DenseFfnWeightsNative {
        gate,
        up,
        down,
        gate_type,
        up_type,
        down_type,
        intermediate_size,
        hidden_size,
    })
}

pub(super) fn dense_ffn_tensor_types(
    gguf: &GgufFile,
    layer_idx: u32,
) -> Result<(GgmlType, GgmlType, GgmlType)> {
    let p = format!("blk.{layer_idx}");
    let tensor_type = |role: &str| {
        gguf.tensor_info(&format!("{p}.ffn_{role}.weight"))
            .map(|info| info.ggml_type)
            .ok_or_else(|| anyhow!("layer {layer_idx}: ffn_{role}.weight not found in GGUF"))
    };
    Ok((
        tensor_type("gate")?,
        tensor_type("up")?,
        tensor_type("down")?,
    ))
}

/// Load a dense FFN layer's weights, keeping every projection in its native
/// GGML scalar or block representation. Sibling gate/up/down matrices may
/// deliberately use different codecs.
///
/// Gate/up/down projection buffers are retained as exact mapped views rather
/// than expanded through `load_tensor_f32`. This avoids a load-time
/// dequantize/re-encode cycle and its full-precision memory expansion.
///
/// Returns `Err` if any weight tensor has an unsupported quantization type.
/// Float storage is selected explicitly by [`load_layer`]; an error here must
/// never trigger a silent full-model F32 expansion.
pub fn load_dense_ffn_quantized(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    layer_idx: u32,
    cfg: &Qwen35Config,
    device: &MlxDevice,
) -> Result<DenseFfnWeightsQ> {
    let p = format!("blk.{}", layer_idx);

    // Classify metadata before loading any buffers. Every projection retains
    // an independent artifact codec; fusion is an execution-only decision.
    let (ggml_type_gate, ggml_type_up, ggml_type_down) = dense_ffn_tensor_types(gguf, layer_idx)?;
    anyhow::ensure!(
        matches!(
            dense_ffn_storage(layer_idx, ggml_type_gate, ggml_type_up, ggml_type_down)?,
            DenseFfnStorage::Quantized | DenseFfnStorage::MixedNative
        ),
        "layer {layer_idx}: dense FFN does not require the encoded native path"
    );

    // Load raw GGML blocks — DType::U8 on Metal, no F32 expansion.
    // Residency registration retains the same mapped Metal allocation.
    let map_matrix = |name: &str| -> Result<MlxBuffer> {
        let info = gguf
            .tensor_info(name)
            .ok_or_else(|| anyhow!("layer {layer_idx}: {name} not found in GGUF"))?;
        let weight = MlxQWeight::from_mapped_gguf_tensor(mapped, info)
            .with_context(|| format!("layer {layer_idx} {name} (native mapped)"))?;
        super::weight_pool::register_weight_buffer(device, &weight.buffer)
            .with_context(|| format!("register {name}"))?;
        #[cfg(test)]
        super::execution_observation::observe_loaded_ggml(name, &weight.buffer)?;
        Ok(weight.buffer)
    };
    let gate_q = map_matrix(&format!("{p}.ffn_gate.weight"))?;
    let up_q = map_matrix(&format!("{p}.ffn_up.weight"))?;
    let down_q = map_matrix(&format!("{p}.ffn_down.weight"))?;

    // Use config values as authoritative (already validated against GGUF metadata
    // by Qwen35Config::from_gguf).
    let hidden_size = cfg.hidden_size;
    let intermediate_size = cfg
        .intermediate_size
        .ok_or_else(|| anyhow!("layer {layer_idx}: dense FFN but cfg.intermediate_size is None"))?;

    Ok(DenseFfnWeightsQ {
        gate_q,
        up_q,
        down_q,
        ggml_type_gate,
        ggml_type_up,
        ggml_type_down,
        intermediate_size,
        hidden_size,
    })
}

/// Load a complete layer (attention + FFN) per its `Qwen35LayerKind` and
/// the model's FFN variant.
pub fn load_layer(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<Qwen35LayerWeights> {
    let mapped = gguf
        .map_tensor_data(device)
        .context("map GGUF tensor data for standalone layer load")?;
    let kind = cfg
        .layer_types
        .get(layer_idx as usize)
        .copied()
        .ok_or_else(|| anyhow!("layer_idx {layer_idx} out of range"))?;

    let ffn = load_ffn(gguf, &mapped, cfg, layer_idx, device)?;

    match kind {
        Qwen35LayerKind::FullAttention => {
            let attn = load_full_attn_layer(gguf, cfg, layer_idx, device)?;
            Ok(Qwen35LayerWeights::FullAttn { attn, ffn })
        }
        Qwen35LayerKind::LinearAttention => {
            let attn = load_delta_net_layer(gguf, cfg, layer_idx, device)?;
            Ok(Qwen35LayerWeights::LinearAttn { attn, ffn })
        }
    }
}

/// Load only the FFN portion of a layer. Production model loading pairs this
/// with a native attention variant so quantized attention is never expanded.
pub fn load_ffn(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    cfg: &Qwen35Config,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<Qwen35FfnWeights> {
    // Dense variant: select the storage class from all three projection
    // tensors. Errors never fall back to F32: doing so can silently expand a
    // quantized dense model by tens of GiB and replace its optimized kernels.
    //
    // MoE variant: always uses quantized MoeFfnWeightsQ (see comment in the
    // MoeQ branch below for rationale).
    Ok(match cfg.variant {
        Qwen35Variant::Dense => {
            let (gate, up, down) = dense_ffn_tensor_types(gguf, layer_idx)?;
            match dense_ffn_storage(layer_idx, gate, up, down)? {
                DenseFfnStorage::Quantized | DenseFfnStorage::MixedNative => {
                    Qwen35FfnWeights::DenseQ(load_dense_ffn_quantized(
                        gguf, mapped, layer_idx, cfg, device,
                    )?)
                }
                DenseFfnStorage::NativeScalar => Qwen35FfnWeights::DenseNative(
                    load_dense_ffn_native(gguf, mapped, layer_idx, cfg, device)?,
                ),
            }
        }
        Qwen35Variant::Moe => {
            // MoeFfnWeightsQ preserves the GGUF's native Q6_K/Q8_0 expert tensor layout
            // and avoids the 128 GB F32 expansion that OOMs on the real 35B-A3B model
            // (256 experts × 40 layers × 3 tensors × 2048×512 × 4 bytes exceeds Metal's
            // 112 GB working set cap). The F32 `load_moe_ffn` / `Qwen35FfnWeights::Moe`
            // variant is preserved for synthetic-weight unit tests that deliberately use
            // F32 inputs (see gpu_ffn.rs::build_moe_ffn_layer_gpu).
            Qwen35FfnWeights::MoeQ(load_moe_ffn_quantized(
                gguf, mapped, cfg, layer_idx, device,
            )?)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::gguf::writer::GgufWriter;
    use crate::inference::models::qwen35::model::Qwen35Model;
    use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
    use crate::ir::DType;
    use crate::quantize::ggml_quants::GgmlType as WriterGgmlType;

    #[test]
    fn production_qwen_projection_retains_file_backed_artifact_storage() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Some(device) = MlxDevice::new().ok() else {
            eprintln!("[skip] Metal device unavailable");
            return;
        };
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("qwen-native-projection.gguf");
        let rows = 4usize;
        let cols = 256usize;
        let payload: Vec<f32> = (0..rows * cols).map(|value| value as f32).collect();
        {
            let file = std::fs::File::create(&path).expect("create GGUF");
            let mut writer = GgufWriter::new(file);
            writer.write_header(1, 0).expect("header");
            let tensor = writer
                .reserve_tensor_info(
                    "blk.0.attn_q.weight",
                    &[cols as u64, rows as u64],
                    WriterGgmlType::F32,
                )
                .expect("tensor info");
            writer.pad_to_alignment().expect("alignment");
            writer
                .stream_tensor_payload(tensor, bytemuck::cast_slice(&payload))
                .expect("tensor payload");
            writer.finalize().expect("finalize");
        }

        let gguf = GgufFile::open(&path).expect("open GGUF");
        let mapped = gguf.map_tensor_data(&device).expect("map GGUF");
        let (projection, storage) =
            load_native_projection(&gguf, &mapped, "blk.0.attn_q.weight", rows, cols, &device)
                .expect("load projection");
        assert_eq!(storage, GgmlType::F32);
        assert!(
            projection.is_file_backed(),
            "production Qwen matrices must be views of the scoped GGUF mapping"
        );
        assert_eq!(projection.data_byte_len(), payload.len() * 4);
    }

    #[test]
    fn production_shared_gate_retains_exact_rank_one_row_view_only() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Some(device) = MlxDevice::new().ok() else {
            eprintln!("[skip] Metal device unavailable");
            return;
        };
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("qwen-native-row-vector.gguf");
        let cols = 256usize;
        let payload = vec![0u8; cols * 4];
        {
            let file = std::fs::File::create(&path).expect("create GGUF");
            let mut writer = GgufWriter::new(file);
            writer.write_header(2, 0).expect("header");
            let row = writer
                .reserve_tensor_info(
                    "blk.0.ffn_gate_inp_shexp.weight",
                    &[cols as u64],
                    WriterGgmlType::F32,
                )
                .expect("row tensor info");
            let rank_two = writer
                .reserve_tensor_info(
                    "blk.0.rank_two.weight",
                    &[cols as u64, 1],
                    WriterGgmlType::F32,
                )
                .expect("rank-two tensor info");
            writer.pad_to_alignment().expect("alignment");
            writer
                .stream_tensor_payload(row, &payload)
                .expect("row payload");
            writer
                .stream_tensor_payload(rank_two, &payload)
                .expect("rank-two payload");
            writer.finalize().expect("finalize");
        }

        let gguf = GgufFile::open(&path).expect("open GGUF");
        let mapped = gguf.map_tensor_data(&device).expect("map GGUF");
        let row = load_native_row_projection(
            &gguf,
            &mapped,
            "blk.0.ffn_gate_inp_shexp.weight",
            cols,
            &device,
        )
        .expect("load exact rank-one row projection");
        assert_eq!((row.info.rows, row.info.cols), (1, cols));
        assert!(row.buffer.is_file_backed());
        assert_eq!(row.buffer.data_byte_len(), payload.len());

        let ordinary_error = MlxQWeight::from_mapped_gguf_tensor(
            &mapped,
            gguf.tensor_info("blk.0.ffn_gate_inp_shexp.weight").unwrap(),
        )
        .err()
        .expect("ordinary native matrices must remain rank two");
        assert!(ordinary_error.to_string().contains("must be rank 2"));

        let squeeze_error =
            load_native_row_projection(&gguf, &mapped, "blk.0.rank_two.weight", cols, &device)
                .err()
                .expect("rank-two storage must not be implicitly squeezed");
        assert!(format!("{squeeze_error:#}").contains("must be exact rank 1"));
    }

    #[test]
    fn qwen38_dense_q4k_q6k_storage_stays_quantized() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        assert_eq!(
            dense_ffn_storage(0, GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q6_K)
                .expect("Q4_K gate/up with Q6_K down is a supported dense layout"),
            DenseFfnStorage::Quantized
        );
    }

    #[test]
    fn qwen38_q2_q3_dense_and_moe_storage_stays_quantized() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        for (gate_up, down) in [
            (GgmlType::Q2_K, GgmlType::Q3_K),
            (GgmlType::Q3_K, GgmlType::Q4_K),
        ] {
            assert_eq!(
                dense_ffn_storage(0, gate_up, gate_up, down).unwrap(),
                DenseFfnStorage::Quantized
            );
            assert!(qwen35_moe_expert_type_supported(gate_up));
            assert!(qwen35_moe_expert_type_supported(down));
        }
    }

    #[test]
    fn every_native_quantized_text_codec_retains_artifact_blocks() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        for ggml_type in [
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::Q4_0,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
            GgmlType::IQ4_NL,
            GgmlType::IQ4_XS,
        ] {
            assert_eq!(
                dense_ffn_storage(0, ggml_type, ggml_type, ggml_type)
                    .expect("linked quantized text artifact must retain native blocks"),
                DenseFfnStorage::Quantized
            );
        }
    }

    #[test]
    fn dense_scalar_fixtures_keep_native_storage() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        for storage in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
            assert_eq!(
                dense_ffn_storage(0, storage, storage, storage)
                    .expect("native scalar fixtures remain supported"),
                DenseFfnStorage::NativeScalar
            );
        }
        assert_eq!(
            dense_ffn_storage(1, GgmlType::F32, GgmlType::F16, GgmlType::BF16)
                .expect("mixed native scalar storage is dispatched per projection"),
            DenseFfnStorage::NativeScalar
        );
    }

    #[test]
    fn dense_storage_admits_cross_class_and_mixed_quant_codecs_without_substitution() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        assert_eq!(
            dense_ffn_storage(7, GgmlType::Q4_K, GgmlType::BF16, GgmlType::F16)
                .expect("each cross-class sibling has an exact native dispatch"),
            DenseFfnStorage::MixedNative
        );

        assert_eq!(
            dense_ffn_storage(8, GgmlType::Q2_K, GgmlType::Q5_K, GgmlType::IQ4_NL)
                .expect("each projection has an independent exact native dispatch"),
            DenseFfnStorage::Quantized
        );
    }

    fn f32_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
        values.flat_map(|v| v.to_le_bytes()).collect()
    }

    fn insert_f32(map: &mut LazyTensorMap, name: &str, shape: Vec<usize>, base: f32) {
        let numel: usize = shape.iter().product();
        let data = f32_bytes((0..numel).map(|i| base + i as f32));
        let meta = LazyMeta::new(name.to_string(), shape, DType::F32);
        map.insert(LazyTensor::from_bytes(meta, data));
    }

    fn insert_zeros(map: &mut LazyTensorMap, name: &str, shape: Vec<usize>) {
        insert_f32(map, name, shape, 0.0);
    }

    fn synthetic_four_layer_dense_map() -> LazyTensorMap {
        let mut map = LazyTensorMap::new();
        let h = 32usize;
        let d = 8usize;
        let q_heads = 2usize;
        let kv_heads = 1usize;
        let lin_k_heads = 1usize;
        let lin_v_heads = 2usize;
        let inter = 32usize;

        insert_f32(&mut map, "token_embd.weight", vec![16, h], 100.0);
        insert_f32(&mut map, "output.weight", vec![16, h], 200.0);
        insert_f32(&mut map, "output_norm.weight", vec![h], 300.0);

        for layer in 0..4 {
            let p = format!("blk.{layer}");
            insert_zeros(&mut map, &format!("{p}.attn_norm.weight"), vec![h]);
            insert_zeros(
                &mut map,
                &format!("{p}.post_attention_norm.weight"),
                vec![h],
            );
            if layer == 3 {
                insert_f32(
                    &mut map,
                    &format!("{p}.attn_q.weight"),
                    vec![2 * q_heads * d, h],
                    1_000.0,
                );
                insert_zeros(
                    &mut map,
                    &format!("{p}.attn_k.weight"),
                    vec![kv_heads * d, h],
                );
                insert_zeros(
                    &mut map,
                    &format!("{p}.attn_v.weight"),
                    vec![kv_heads * d, h],
                );
                insert_zeros(&mut map, &format!("{p}.attn_q_norm.weight"), vec![d]);
                insert_zeros(&mut map, &format!("{p}.attn_k_norm.weight"), vec![d]);
                insert_zeros(
                    &mut map,
                    &format!("{p}.attn_output.weight"),
                    vec![h, q_heads * d],
                );
            } else {
                let qkv_rows = 2 * lin_k_heads * d + lin_v_heads * d;
                insert_zeros(&mut map, &format!("{p}.attn_qkv.weight"), vec![qkv_rows, h]);
                insert_zeros(
                    &mut map,
                    &format!("{p}.attn_gate.weight"),
                    vec![lin_v_heads * d, h],
                );
                insert_zeros(
                    &mut map,
                    &format!("{p}.ssm_conv1d.weight"),
                    vec![qkv_rows, 4],
                );
                insert_zeros(
                    &mut map,
                    &format!("{p}.ssm_alpha.weight"),
                    vec![lin_v_heads, h],
                );
                insert_zeros(&mut map, &format!("{p}.ssm_dt.bias"), vec![lin_v_heads]);
                insert_zeros(
                    &mut map,
                    &format!("{p}.ssm_beta.weight"),
                    vec![lin_v_heads, h],
                );
                insert_zeros(&mut map, &format!("{p}.ssm_a"), vec![lin_v_heads]);
                insert_zeros(&mut map, &format!("{p}.ssm_norm.weight"), vec![d]);
                insert_zeros(
                    &mut map,
                    &format!("{p}.ssm_out.weight"),
                    vec![h, lin_v_heads * d],
                );
            }
            insert_zeros(&mut map, &format!("{p}.ffn_gate.weight"), vec![inter, h]);
            insert_zeros(&mut map, &format!("{p}.ffn_up.weight"), vec![inter, h]);
            insert_zeros(&mut map, &format!("{p}.ffn_down.weight"), vec![h, inter]);
        }

        map
    }

    fn synthetic_four_layer_moe_map() -> LazyTensorMap {
        let mut map = synthetic_four_layer_dense_map();
        for layer in 0..4 {
            let p = format!("blk.{layer}");
            map.remove(&format!("{p}.ffn_gate.weight"));
            map.remove(&format!("{p}.ffn_up.weight"));
            map.remove(&format!("{p}.ffn_down.weight"));
            insert_zeros(&mut map, &format!("{p}.ffn_gate_inp.weight"), vec![2, 32]);
            insert_f32(
                &mut map,
                &format!("{p}.ffn_gate_exps.weight"),
                vec![2, 16, 32],
                10.0,
            );
            insert_f32(
                &mut map,
                &format!("{p}.ffn_up_exps.weight"),
                vec![2, 16, 32],
                20.0,
            );
            insert_f32(
                &mut map,
                &format!("{p}.ffn_down_exps.weight"),
                vec![2, 32, 16],
                30.0,
            );
            insert_zeros(
                &mut map,
                &format!("{p}.ffn_gate_inp_shexp.weight"),
                vec![32],
            );
            insert_zeros(
                &mut map,
                &format!("{p}.ffn_gate_shexp.weight"),
                vec![16, 32],
            );
            insert_zeros(&mut map, &format!("{p}.ffn_up_shexp.weight"), vec![16, 32]);
            insert_zeros(
                &mut map,
                &format!("{p}.ffn_down_shexp.weight"),
                vec![32, 16],
            );
        }
        map
    }

    fn load_lazy_or_skip_without_metal(map: &LazyTensorMap) -> Option<Qwen35Model> {
        match Qwen35Model::load_from_lazy_tensor_map(map) {
            Ok(model) => Some(model),
            Err(err) if format!("{err}").contains("No Metal GPU device found") => {
                eprintln!("skipping GPU-backed lazy load test: {err}");
                None
            }
            Err(err) => panic!("lazy load: {err:#}"),
        }
    }

    #[test]
    fn load_from_lazy_tensor_map_infers_four_layer_dense_config() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let map = synthetic_four_layer_dense_map();
        let Some(model) = load_lazy_or_skip_without_metal(&map) else {
            return;
        };
        assert_eq!(model.cfg.variant, Qwen35Variant::Dense);
        assert_eq!(model.cfg.num_hidden_layers, 4);
        assert_eq!(model.cfg.hidden_size, 32);
        assert_eq!(model.cfg.full_attention_interval, 4);
        assert_eq!(model.cfg.num_attention_heads, 2);
        assert_eq!(model.cfg.num_key_value_heads, 1);
        assert_eq!(model.cfg.linear_num_key_heads, 1);
        assert_eq!(model.cfg.linear_num_value_heads, 2);
        assert_eq!(model.token_embd[0], 100.0);
        assert_eq!(model.output_weight[0], 200.0);
        assert_eq!(model.output_norm[0], 300.0);
    }

    #[test]
    fn load_from_lazy_tensor_map_splits_fused_full_attention_q_gate() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let map = synthetic_four_layer_dense_map();
        let Some(model) = load_lazy_or_skip_without_metal(&map) else {
            return;
        };
        let attn = match &model.layers[3] {
            Qwen35LayerWeights::FullAttn { attn, .. } => attn,
            _ => panic!("layer 3 should be full attention"),
        };
        let h = model.cfg.hidden_size as usize;
        let d = model.cfg.head_dim as usize;
        assert_eq!(attn.wq[0], 1_000.0);
        assert_eq!(attn.w_gate[0], 1_000.0 + (d * h) as f32);
        assert_eq!(attn.wq[d * h], 1_000.0 + (2 * d * h) as f32);
        assert_eq!(attn.w_gate[d * h], 1_000.0 + (3 * d * h) as f32);
    }

    #[test]
    fn load_from_lazy_tensor_map_quantizes_moe_experts_to_q8_0() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let map = synthetic_four_layer_moe_map();
        let Some(model) = load_lazy_or_skip_without_metal(&map) else {
            return;
        };
        assert_eq!(model.cfg.variant, Qwen35Variant::Moe);
        let ffn = model.layers[0].ffn();
        let moe = match ffn {
            Qwen35FfnWeights::MoeQ(moe) => moe,
            other => panic!("expected MoeQ, got {}", other.variant()),
        };
        assert_eq!(moe.ggml_type_gate, GgmlType::Q8_0);
        assert_eq!(moe.ggml_type_up, GgmlType::Q8_0);
        assert_eq!(moe.ggml_type_down, GgmlType::Q8_0);
        assert_eq!(moe.expert_gate_q.dtype(), mlx_native::DType::U8);
        let gate_bytes = moe.expert_gate_q.as_slice::<u8>().expect("gate bytes");
        assert_eq!(gate_bytes.len(), (2 * 16 * 32 / 32) * 34);
    }

    /// **Integration test**: load a single linear-attention layer (layer 0)
    /// from the real apex GGUF. Verifies:
    /// - All 10 DeltaNet tensors + 8 MoE FFN tensors load successfully.
    /// - Shapes match what the CPU reference expects.
    /// - Values are finite and non-degenerate (non-zero stddev).
    ///
    /// Runtime-skips when artefact absent. Memory cost when run:
    /// ~1-2 GB of dequantized f32 for one MoE linear layer.
    #[test]
    fn load_real_apex_linear_attn_layer_0() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let path = std::path::PathBuf::from(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             APEX-Q5_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: apex GGUF not at expected path");
            return;
        }
        let gguf = match GgufFile::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let cfg = Qwen35Model::load_config_only(&gguf).expect("config");

        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };

        // Layer 0 is linear-attention (apex cfg: interval=4, so layer 3 is first full).
        assert_eq!(cfg.layer_types[0], Qwen35LayerKind::LinearAttention);

        let layer = load_layer(&gguf, &cfg, 0, &device).expect("load layer 0");

        // Inspect shapes + stats.
        let attn = match &layer {
            Qwen35LayerWeights::LinearAttn { attn, .. } => attn,
            _ => panic!("expected LinearAttn"),
        };
        let nv = cfg.linear_num_value_heads as usize;
        let _dk = cfg.linear_key_head_dim as usize;
        let dv = cfg.linear_value_head_dim as usize;
        let z_channels = nv * dv;

        assert_eq!(attn.attn_norm.len(), cfg.hidden_size as usize);
        assert_eq!(attn.ssm_a.len(), nv);
        assert_eq!(attn.ssm_dt_bias.len(), nv);
        // APEX LAYOUT DISCOVERY: ssm_norm is per-head-shared at shape [D_v],
        // NOT [n_v_heads * D_v]. One norm weight broadcasts across all v_heads.
        // This is a correction to the schema documented in delta_net.rs;
        // noted here so the future fix to DeltaNetLayerWeights.ssm_norm
        // carries through to delta_net_layer_cpu_ref.
        assert_eq!(attn.ssm_norm.len(), dv);
        assert_eq!(attn.ssm_out.len(), (cfg.hidden_size as usize) * z_channels);

        // All tensors finite + non-degenerate.
        for (name, data) in [
            ("attn_norm", &attn.attn_norm),
            ("attn_qkv", &attn.attn_qkv),
            ("ssm_conv1d", &attn.ssm_conv1d),
            ("ssm_a", &attn.ssm_a),
            ("ssm_dt_bias", &attn.ssm_dt_bias),
            ("ssm_out", &attn.ssm_out),
        ] {
            let n_nan = data.iter().filter(|v| v.is_nan()).count();
            let n_inf = data
                .iter()
                .filter(|v| !v.is_finite() && !v.is_nan())
                .count();
            assert_eq!(n_nan, 0, "{}: NaN values present", name);
            assert_eq!(n_inf, 0, "{}: Inf values present", name);

            let n = data.len() as f64;
            let sum: f64 = data.iter().map(|v| *v as f64).sum();
            let sum_sq: f64 = data.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let mean = sum / n;
            let variance = (sum_sq / n - mean * mean).max(0.0);
            let stddev = variance.sqrt();

            eprintln!(
                "  {}: len={}, mean={:.6}, stddev={:.6}",
                name,
                data.len(),
                mean,
                stddev
            );
            assert!(
                stddev > 1e-9 || name == "ssm_dt_bias" || data.iter().all(|v| v.abs() < 1e-9),
                "{}: degenerate (all-equal) content (stddev={})",
                name,
                stddev
            );
        }

        // MoE FFN tensors also loaded + finite.
        // Production loader (`load_layer` for MoE variant) returns
        // `Qwen35FfnWeights::MoeQ` (native GGML blocks on Metal — no F32
        // expansion of 256 experts) per the OOM-prevention path. The
        // F32-expanded `Qwen35FfnWeights::Moe` variant is used only by
        // synthetic-test fixtures via `empty_from_cfg`. Accept either.
        let moe_cfg = cfg.moe.as_ref().expect("moe cfg");
        let expected_router_len = (moe_cfg.num_experts * cfg.hidden_size) as usize;
        match layer.ffn() {
            Qwen35FfnWeights::Moe(m) => {
                assert_eq!(m.router.len(), expected_router_len);
                assert_eq!(
                    m.expert_gate.len(),
                    (moe_cfg.num_experts * moe_cfg.moe_intermediate_size * cfg.hidden_size)
                        as usize
                );
                let router_finite = m.router.iter().all(|v| v.is_finite());
                assert!(router_finite, "router has non-finite values");
            }
            Qwen35FfnWeights::MoeQ(m) => {
                assert_eq!(m.router.info.rows * m.router.info.cols, expected_router_len);
                let router_finite = m
                    .router
                    .buffer
                    .as_slice::<f32>()
                    .expect("synthetic router F32 storage")
                    .iter()
                    .all(|v| v.is_finite());
                assert!(router_finite, "router has non-finite values");
                // Expert tensors are GGML blocks on the device — assert
                // dtype is U8 (block bytes) and byte count is non-zero.
                assert_eq!(
                    m.expert_gate_q.dtype(),
                    mlx_native::DType::U8,
                    "expert_gate_q must be raw GGML blocks (U8)"
                );
                assert!(
                    m.expert_gate_q.element_count() > 0,
                    "expert_gate_q must have non-zero byte count"
                );
                assert!(
                    m.expert_up_q.element_count() > 0,
                    "expert_up_q must have non-zero byte count"
                );
                assert!(
                    m.expert_down_q.element_count() > 0,
                    "expert_down_q must have non-zero byte count"
                );
            }
            _ => panic!(
                "expected MoE FFN (Moe or MoeQ), got {}",
                layer.ffn().variant()
            ),
        }
    }

    /// Integration test for a full-attention layer (layer 3 in apex).
    #[test]
    fn load_real_apex_full_attn_layer_3() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let path = std::path::PathBuf::from(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             APEX-Q5_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: apex GGUF not at expected path");
            return;
        }
        let gguf = match GgufFile::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let cfg = Qwen35Model::load_config_only(&gguf).expect("config");
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };

        assert_eq!(cfg.layer_types[3], Qwen35LayerKind::FullAttention);
        let layer = load_layer(&gguf, &cfg, 3, &device).expect("load layer 3");

        let attn = match &layer {
            Qwen35LayerWeights::FullAttn { attn, .. } => attn,
            _ => panic!("expected FullAttn"),
        };

        // Sanity on full-attention tensors.
        let h = cfg.hidden_size as usize;
        let nh = cfg.num_attention_heads as usize;
        let nkv = cfg.num_key_value_heads as usize;
        let d = cfg.head_dim as usize;
        assert_eq!(attn.wq.len(), nh * d * h);
        assert_eq!(attn.wk.len(), nkv * d * h);
        assert_eq!(attn.wv.len(), nkv * d * h);

        for (name, data) in [("wq", &attn.wq), ("wk", &attn.wk), ("wv", &attn.wv)] {
            let n_nan = data.iter().filter(|v| v.is_nan()).count();
            assert_eq!(n_nan, 0, "{}: NaN values present", name);
            let stddev = {
                let n = data.len() as f64;
                let sum: f64 = data.iter().map(|v| *v as f64).sum();
                let sum_sq: f64 = data.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let mean = sum / n;
                ((sum_sq / n - mean * mean).max(0.0)).sqrt()
            };
            eprintln!("  {}: len={}, stddev={:.6}", name, data.len(), stddev);
            assert!(stddev > 1e-9, "{}: degenerate", name);
        }
    }

    /// Global tensors loadable from real apex.
    #[test]
    fn load_real_apex_global_tensors() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let path = std::path::PathBuf::from(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             APEX-Q5_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: apex GGUF not at expected path");
            return;
        }
        let gguf = match GgufFile::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let cfg = Qwen35Model::load_config_only(&gguf).expect("config");
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };

        let (embd, out_w, out_norm) = load_global_tensors(&gguf, &cfg, &device).expect("globals");
        let vocab = cfg.vocab_size as usize;
        let h = cfg.hidden_size as usize;
        assert_eq!(embd.len(), vocab * h, "token_embd shape");
        assert_eq!(out_w.len(), vocab * h, "output.weight shape");
        assert_eq!(out_norm.len(), h, "output_norm shape");

        // Spot-check non-degenerate.
        let embd_stddev = {
            let n = embd.len() as f64;
            let sum: f64 = embd.iter().map(|v| *v as f64).sum();
            let sum_sq: f64 = embd.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let mean = sum / n;
            ((sum_sq / n - mean * mean).max(0.0)).sqrt()
        };
        eprintln!(
            "  token_embd: {} values, stddev = {:.6}",
            embd.len(),
            embd_stddev
        );
        assert!(embd_stddev > 1e-6, "token_embd degenerate");
    }
}
