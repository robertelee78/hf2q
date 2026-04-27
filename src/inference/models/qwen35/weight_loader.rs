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
//! For production GPU inference, a separate loader (follow-up iter) will
//! hold tensors as quantized `MlxBuffer`s and pass them directly to
//! mlx-native's `quantized_matmul_ggml` kernels, avoiding the dequant
//! bounce entirely.
//!
//! # Layout conversion
//!
//! GGUF's `load_tensor_f32` returns a `Vec<f32>` with dims in "outer-first"
//! shape order (see `TensorInfo.shape`). For matrices, this usually matches
//! the `[out_dim, in_dim]` convention our CPU refs expect. Single-vector
//! tensors (norms, biases) are 1-D and drop in directly.

use anyhow::{anyhow, Context, Result};
use std::collections::BTreeMap;
use mlx_native::gguf::GgufFile;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType as MlxDType, MlxBuffer, MlxDevice};

use crate::ir::lazy::{LazyTensor, LazyTensorMap};
use crate::ir::{DType as IrDType, TensorRef};

use super::delta_net::DeltaNetLayerWeights;
use super::ffn::{DenseFfnWeights, MoeFfnWeights};
use super::full_attn::FullAttnLayerWeights;
use super::in_memory_loader::{
    bf16_bytes_to_f32, f16_bytes_to_f32, f32_bytes_to_f32, quantize_f32_to_q8_0_buffer,
};
use super::model::{Qwen35FfnWeights, Qwen35LayerWeights, Qwen35Model};
use super::{default_layer_types, Qwen35Config, Qwen35LayerKind, Qwen35MoeConfig, Qwen35Variant};

// ============================================================================
// Quantized MoE weight container
// ============================================================================

/// Per-layer MoE FFN weights with expert tensors kept in their native GGML
/// quantization.  Small tensors (router, shared-expert) are still f32.
///
/// This struct is the bridge between GGUF disk bytes and
/// `MoeFfnWeightsGpuQ`: it holds the raw Metal buffers that `GgufFile::load_tensor`
/// produced, plus the f32 scalars needed for routing and shared-expert computation.
pub struct MoeFfnWeightsQ {
    /// Router: `[num_experts, hidden_size]` F32.
    pub router: Vec<f32>,
    /// Stacked expert gate_proj: raw GGML blocks, dtype U8 on Metal.
    pub expert_gate_q: MlxBuffer,
    /// Stacked expert up_proj: raw GGML blocks, dtype U8 on Metal.
    pub expert_up_q: MlxBuffer,
    /// Stacked expert down_proj: raw GGML blocks, dtype U8 on Metal.
    pub expert_down_q: MlxBuffer,
    /// GGML quantization type for the gate and up expert buffers (must match).
    /// In the apex GGUF these are Q5_K.
    pub ggml_type_gate_up: GgmlType,
    /// GGML quantization type for the down expert buffer (may differ from gate/up).
    /// In the apex GGUF this is Q6_K.
    pub ggml_type_down: GgmlType,
    /// Shared-expert sigmoid gate: `[hidden_size]` F32.
    pub shared_gate_logit: Vec<f32>,
    /// Shared-expert SwiGLU weights (F32).
    pub shared_gate: Vec<f32>,
    pub shared_up: Vec<f32>,
    pub shared_down: Vec<f32>,
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
/// For a 27B dense GGUF (hidden=5120, intermediate=17408, Q4_K weights):
///   F32 expansion: 17408×5120×2×4 bytes ≈ 714 MB per layer × 64 layers ≈ 46 GB
///   Q4_K on-disk:  ≈ 7 GB per layer (4 bits/weight × 1.06× overhead)
///   Total savings: ~39 GB Metal working set, eliminating the 129 GB OOM.
pub struct DenseFfnWeightsQ {
    /// Gate projection raw GGML blocks: `[intermediate_size, hidden_size]`.
    pub gate_q: MlxBuffer,
    /// Up projection raw GGML blocks: `[intermediate_size, hidden_size]`.
    pub up_q: MlxBuffer,
    /// Down projection raw GGML blocks: `[hidden_size, intermediate_size]`.
    pub down_q: MlxBuffer,
    /// GGML quantization type for gate/up (must be same — they share k=hidden_size).
    pub ggml_type_gate_up: GgmlType,
    /// GGML quantization type for down (may differ from gate/up in mixed-quant GGUFs).
    pub ggml_type_down: GgmlType,
    /// Dense FFN intermediate dimension (number of rows in gate/up weight).
    pub intermediate_size: u32,
    /// Model hidden dimension (number of columns in gate/up weight).
    pub hidden_size: u32,
}

/// Load a tensor from the GGUF, dequantize to f32, and download into
/// a `Vec<f32>`.
pub fn load_f32_tensor(
    gguf: &GgufFile,
    name: &str,
    device: &MlxDevice,
) -> Result<Vec<f32>> {
    let buf = gguf
        .load_tensor_f32(name, device)
        .map_err(|e| anyhow!("load_tensor_f32({name}): {e}"))?;
    let slice: &[f32] = buf
        .as_slice()
        .map_err(|e| anyhow!("as_slice({name}): {e}"))?;
    Ok(slice.to_vec())
}

/// Load the global tensors (`token_embd`, `output`, `output_norm`) from
/// a GGUF into flat f32 vectors.
pub fn load_global_tensors(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    device: &MlxDevice,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let _ = cfg; // reserved for future shape validation
    let token_embd = load_f32_tensor(gguf, "token_embd.weight", device)
        .context("token_embd.weight")?;
    let output_weight = load_f32_tensor(gguf, "output.weight", device)
        .context("output.weight")?;
    let output_norm = load_f32_tensor(gguf, "output_norm.weight", device)
        .context("output_norm.weight")?;
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
        .or_else(|| lookup.maybe(&format!("blk.{full_layer_idx}.attn_k_norm.weight"))
            .and_then(|t| t.shape().first().copied()))
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
        .unwrap_or((2 * num_key_value_heads * linear_key_head_dim
            + linear_num_value_heads * linear_value_head_dim) as usize) as u32;
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
                .ok_or_else(|| anyhow!("blk.0.ffn_gate.weight has empty shape"))? as u32;
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
    tensor.data.clear();
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

fn load_lazy_dense_ffn(
    lookup: &LazyQwen35Lookup<'_>,
    layer_idx: u32,
) -> Result<DenseFfnWeights> {
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
    Ok(MoeFfnWeightsQ {
        router: load_lazy_f32(lookup, &format!("{p}.ffn_gate_inp.weight"))?,
        expert_gate_q: load_lazy_expert_q8_0(lookup, &format!("{p}.ffn_gate_exps.weight"), device)?,
        expert_up_q: load_lazy_expert_q8_0(lookup, &format!("{p}.ffn_up_exps.weight"), device)?,
        expert_down_q: load_lazy_expert_q8_0(lookup, &format!("{p}.ffn_down_exps.weight"), device)?,
        ggml_type_gate_up: GgmlType::Q8_0,
        ggml_type_down: GgmlType::Q8_0,
        shared_gate_logit: load_lazy_f32(lookup, &format!("{p}.ffn_gate_inp_shexp.weight"))?,
        shared_gate: load_lazy_f32(lookup, &format!("{p}.ffn_gate_shexp.weight"))?,
        shared_up: load_lazy_f32(lookup, &format!("{p}.ffn_up_shexp.weight"))?,
        shared_down: load_lazy_f32(lookup, &format!("{p}.ffn_down_shexp.weight"))?,
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
        let device = MlxDevice::new()
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
            if current_vocab < QWEN35_FULL_VOCAB
                && (QWEN35_FULL_VOCAB - current_vocab) < 2048
            {
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
            cfg,
            layers,
            token_embd,
            output_weight,
            output_norm,
            mtp: None,
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
    // not a separate `attn_gate.weight` tensor. This matches llama.cpp's in-memory
    // `wq` convention where `wq` has output dim `2 * head_dim * n_head` with Q
    // in the lower half and gate in the upper half. Our CPU reference keeps wq and
    // w_gate separate, so we split after loading.
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
    assert_eq!(post_attn_norm.len(), h, "post_attn_norm layer {layer_idx} shape");

    // De-interleave fused q_fused into wq and w_gate.
    // llama.cpp layout (confirmed from build_layer_attn): Q and gate are INTERLEAVED
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
/// llama.cpp's fused GDN kernel — and now mlx-native's `gated_delta_net_f32`
/// kernel as of mlx-native 0.4.1 (commit `4f00f6e`) — performs the GQA
/// mapping internally as `k_head = v_head % n_k_heads`, which is the
/// inverse of the GGUF tiling: with `v_head = i_vpk * n_k + i_k`,
/// `v_head % n_k = i_k`, recovering the correct K-head for any V-head.
///
/// Therefore: every V-head-axis tensor MUST be left in the GGUF's natural
/// tiled order. No un-reordering. Earlier hf2q snapshots un-reordered to
/// "grouped" order to compensate for an old (block-style) mlx-native kernel
/// that used `k_head = v_head / group_ratio`; that kernel was retired in
/// `4f00f6e` to reach byte-parity with llama.cpp.
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
    let nk     = cfg.linear_num_key_heads as usize;
    let nv     = cfg.linear_num_value_heads as usize;
    let dk     = cfg.linear_key_head_dim as usize;
    let dv     = cfg.linear_value_head_dim as usize;
    let h      = cfg.hidden_size as usize;
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
    let v_rows  = nv * dv;
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
    assert_eq!(ssm_conv1d_gguf.len(), qkv_channels * k_width, "ssm_conv1d shape");
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
pub fn load_moe_ffn(
    gguf: &GgufFile,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<MoeFfnWeights> {
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
/// Expert weight buffers (`ffn_{gate,up,down}_exps`) are loaded via
/// `GgufFile::load_tensor` (raw GGML blocks, DType::U8 on Metal) rather than
/// `load_tensor_f32`.  This avoids the ~3.2 GB per-layer F32 expansion that
/// causes the OOM on the 35B apex model.
///
/// Small tensors (router, shared-expert) are still dequantized to f32 because
/// they are projected with the existing F32 dense kernel.
pub fn load_moe_ffn_quantized(
    gguf: &GgufFile,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<MoeFfnWeightsQ> {
    let p = format!("blk.{}", layer_idx);

    // Router and shared-expert weights are small — dequantize to f32.
    let router = load_f32_tensor(gguf, &format!("{p}.ffn_gate_inp.weight"), device)?;
    let shared_gate_logit =
        load_f32_tensor(gguf, &format!("{p}.ffn_gate_inp_shexp.weight"), device)?;
    let shared_gate = load_f32_tensor(gguf, &format!("{p}.ffn_gate_shexp.weight"), device)?;
    let shared_up   = load_f32_tensor(gguf, &format!("{p}.ffn_up_shexp.weight"), device)?;
    let shared_down = load_f32_tensor(gguf, &format!("{p}.ffn_down_shexp.weight"), device)?;

    // Expert weights: load raw GGML blocks, preserving quantization.
    let expert_gate_q = gguf
        .load_tensor(&format!("{p}.ffn_gate_exps.weight"), device)
        .with_context(|| format!("layer {layer_idx} ffn_gate_exps (quantized)"))?;
    let expert_up_q = gguf
        .load_tensor(&format!("{p}.ffn_up_exps.weight"), device)
        .with_context(|| format!("layer {layer_idx} ffn_up_exps (quantized)"))?;
    let expert_down_q = gguf
        .load_tensor(&format!("{p}.ffn_down_exps.weight"), device)
        .with_context(|| format!("layer {layer_idx} ffn_down_exps (quantized)"))?;

    // Gate and up may have a different quant type than down (e.g. Q5_K vs Q6_K
    // in the apex GGUF).  Read each separately.
    let gate_info = gguf.tensor_info(&format!("{p}.ffn_gate_exps.weight"))
        .ok_or_else(|| anyhow!("layer {layer_idx}: ffn_gate_exps not found in GGUF"))?;
    let ggml_type_gate_up = gate_info.ggml_type;

    let down_info = gguf.tensor_info(&format!("{p}.ffn_down_exps.weight"))
        .ok_or_else(|| anyhow!("layer {layer_idx}: ffn_down_exps not found in GGUF"))?;
    let ggml_type_down = down_info.ggml_type;

    let supported = |t: GgmlType| matches!(t,
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q5_K | GgmlType::Q6_K
    );

    // Validate that the types are supported by quantized_matmul_id_ggml.
    // Q5_K uses the mv_id kernel (mm_id not yet ported); Q4_0/Q8_0/Q6_K
    // also use mv_id for decode and mm_id for prefill batches > 8 tokens.
    if !supported(ggml_type_gate_up) {
        return Err(anyhow!(
            "layer {layer_idx}: gate/up expert weights have unsupported quant type {:?} \
             (expected Q4_0, Q8_0, Q5_K, or Q6_K)",
            ggml_type_gate_up
        ));
    }
    if !supported(ggml_type_down) {
        return Err(anyhow!(
            "layer {layer_idx}: down expert weights have unsupported quant type {:?} \
             (expected Q4_0, Q8_0, Q5_K, or Q6_K)",
            ggml_type_down
        ));
    }

    Ok(MoeFfnWeightsQ {
        router,
        expert_gate_q,
        expert_up_q,
        expert_down_q,
        ggml_type_gate_up,
        ggml_type_down,
        shared_gate_logit,
        shared_gate,
        shared_up,
        shared_down,
    })
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

/// Load a dense FFN layer's weights, keeping projections in their native GGML
/// quantization (e.g. Q4_K, Q6_K) — the production path for 27B dense GGUFs.
///
/// Gate/up/down projection buffers are loaded via `GgufFile::load_tensor` (raw
/// GGML blocks, DType::U8 on Metal) rather than `load_tensor_f32`.  This
/// eliminates the Q4_0→F32 round-trip that expands a 27B model from ~14 GB
/// on-disk to ~129 GB in RAM, causing an OOM hang on M5 Max 128 GB.
///
/// Returns `Err` if any weight tensor has an unsupported quantization type
/// (F32 and F16 are explicitly rejected; the caller's `load_layer` falls back
/// to `load_dense_ffn` for synthetic/tiny models that use float weights).
pub fn load_dense_ffn_quantized(
    gguf: &GgufFile,
    layer_idx: u32,
    cfg: &Qwen35Config,
    device: &MlxDevice,
) -> Result<DenseFfnWeightsQ> {
    let p = format!("blk.{}", layer_idx);

    // Read quantization types from GGUF tensor metadata BEFORE loading buffers,
    // so we can reject unsupported types cheaply without touching the tensor data.
    let gate_info = gguf.tensor_info(&format!("{p}.ffn_gate.weight"))
        .ok_or_else(|| anyhow!("layer {layer_idx}: ffn_gate.weight not found in GGUF"))?;
    let ggml_type_gate_up = gate_info.ggml_type;

    let down_info = gguf.tensor_info(&format!("{p}.ffn_down.weight"))
        .ok_or_else(|| anyhow!("layer {layer_idx}: ffn_down.weight not found in GGUF"))?;
    let ggml_type_down = down_info.ggml_type;

    // Reject float types — callers must use `load_dense_ffn` for those.
    let supported = |t: GgmlType| matches!(
        t,
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q6_K
    );
    if !supported(ggml_type_gate_up) {
        return Err(anyhow!(
            "layer {layer_idx}: gate/up dense weights have unsupported quant type {:?} \
             (expected Q4_0, Q8_0, Q4_K, or Q6_K for the quantized dense path; \
             use load_dense_ffn for F32/F16 weights)",
            ggml_type_gate_up
        ));
    }
    if !supported(ggml_type_down) {
        return Err(anyhow!(
            "layer {layer_idx}: down dense weight has unsupported quant type {:?} \
             (expected Q4_0, Q8_0, Q4_K, or Q6_K for the quantized dense path; \
             use load_dense_ffn for F32/F16 weights)",
            ggml_type_down
        ));
    }

    // Load raw GGML blocks — DType::U8 on Metal, no F32 expansion.
    let gate_q = gguf
        .load_tensor(&format!("{p}.ffn_gate.weight"), device)
        .with_context(|| format!("layer {layer_idx} ffn_gate.weight (quantized)"))?;
    let up_q = gguf
        .load_tensor(&format!("{p}.ffn_up.weight"), device)
        .with_context(|| format!("layer {layer_idx} ffn_up.weight (quantized)"))?;
    let down_q = gguf
        .load_tensor(&format!("{p}.ffn_down.weight"), device)
        .with_context(|| format!("layer {layer_idx} ffn_down.weight (quantized)"))?;

    // Use config values as authoritative (already validated against GGUF metadata
    // by Qwen35Config::from_gguf).
    let hidden_size = cfg.hidden_size;
    let intermediate_size = cfg.intermediate_size
        .ok_or_else(|| anyhow!("layer {layer_idx}: dense FFN but cfg.intermediate_size is None"))?;

    Ok(DenseFfnWeightsQ {
        gate_q,
        up_q,
        down_q,
        ggml_type_gate_up,
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
    let kind = cfg
        .layer_types
        .get(layer_idx as usize)
        .copied()
        .ok_or_else(|| anyhow!("layer_idx {layer_idx} out of range"))?;

    // Dense variant: prefer the quantized Q path (DenseQ) to avoid the
    // Q4_0→F32 round-trip that causes the 129 GB OOM on 27B dense GGUFs.
    // Fall back to F32 `load_dense_ffn` when the GGUF uses float weights
    // (F32/F16), which covers synthetic test models and future architectures.
    //
    // MoE variant: always uses quantized MoeFfnWeightsQ (see comment in the
    // MoeQ branch below for rationale).
    let ffn = match cfg.variant {
        Qwen35Variant::Dense => {
            // Try quantized path first; fall back to F32 for float-weight GGUFs
            // (e.g. synthetic-weight unit tests that use F32 projections).
            match load_dense_ffn_quantized(gguf, layer_idx, cfg, device) {
                Ok(w) => Qwen35FfnWeights::DenseQ(w),
                Err(_) => Qwen35FfnWeights::Dense(load_dense_ffn(gguf, layer_idx, device)?),
            }
        }
        Qwen35Variant::Moe => {
            // MoeFfnWeightsQ preserves the GGUF's native Q6_K/Q8_0 expert tensor layout
            // and avoids the 128 GB F32 expansion that OOMs on the real 35B-A3B model
            // (256 experts × 40 layers × 3 tensors × 2048×512 × 4 bytes exceeds Metal's
            // 112 GB working set cap). The F32 `load_moe_ffn` / `Qwen35FfnWeights::Moe`
            // variant is preserved for synthetic-weight unit tests that deliberately use
            // F32 inputs (see gpu_ffn.rs::build_moe_ffn_layer_gpu).
            Qwen35FfnWeights::MoeQ(load_moe_ffn_quantized(gguf, layer_idx, device)?)
        }
    };

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
    use crate::ir::DType;
    use crate::inference::models::qwen35::model::Qwen35Model;

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
            insert_zeros(&mut map, &format!("{p}.post_attention_norm.weight"), vec![h]);
            if layer == 3 {
                insert_f32(
                    &mut map,
                    &format!("{p}.attn_q.weight"),
                    vec![2 * q_heads * d, h],
                    1_000.0,
                );
                insert_zeros(&mut map, &format!("{p}.attn_k.weight"), vec![kv_heads * d, h]);
                insert_zeros(&mut map, &format!("{p}.attn_v.weight"), vec![kv_heads * d, h]);
                insert_zeros(&mut map, &format!("{p}.attn_q_norm.weight"), vec![d]);
                insert_zeros(&mut map, &format!("{p}.attn_k_norm.weight"), vec![d]);
                insert_zeros(&mut map, &format!("{p}.attn_output.weight"), vec![h, q_heads * d]);
            } else {
                let qkv_rows = 2 * lin_k_heads * d + lin_v_heads * d;
                insert_zeros(&mut map, &format!("{p}.attn_qkv.weight"), vec![qkv_rows, h]);
                insert_zeros(&mut map, &format!("{p}.attn_gate.weight"), vec![lin_v_heads * d, h]);
                insert_zeros(&mut map, &format!("{p}.ssm_conv1d.weight"), vec![qkv_rows, 4]);
                insert_zeros(&mut map, &format!("{p}.ssm_alpha.weight"), vec![lin_v_heads, h]);
                insert_zeros(&mut map, &format!("{p}.ssm_dt.bias"), vec![lin_v_heads]);
                insert_zeros(&mut map, &format!("{p}.ssm_beta.weight"), vec![lin_v_heads, h]);
                insert_zeros(&mut map, &format!("{p}.ssm_a"), vec![lin_v_heads]);
                insert_zeros(&mut map, &format!("{p}.ssm_norm.weight"), vec![d]);
                insert_zeros(&mut map, &format!("{p}.ssm_out.weight"), vec![h, lin_v_heads * d]);
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
            insert_f32(&mut map, &format!("{p}.ffn_gate_exps.weight"), vec![2, 16, 32], 10.0);
            insert_f32(&mut map, &format!("{p}.ffn_up_exps.weight"), vec![2, 16, 32], 20.0);
            insert_f32(&mut map, &format!("{p}.ffn_down_exps.weight"), vec![2, 32, 16], 30.0);
            insert_zeros(&mut map, &format!("{p}.ffn_gate_inp_shexp.weight"), vec![32]);
            insert_zeros(&mut map, &format!("{p}.ffn_gate_shexp.weight"), vec![16, 32]);
            insert_zeros(&mut map, &format!("{p}.ffn_up_shexp.weight"), vec![16, 32]);
            insert_zeros(&mut map, &format!("{p}.ffn_down_shexp.weight"), vec![32, 16]);
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
        assert_eq!(moe.ggml_type_gate_up, GgmlType::Q8_0);
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
    /// `#[ignore]`d so regular `cargo test` doesn't touch the 25 GB file.
    /// Memory cost: ~1-2 GB of dequantized f32 for one MoE linear layer.
    #[test]
    #[ignore]
    fn load_real_apex_linear_attn_layer_0() {
        let path = std::path::PathBuf::from(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf",
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
            let n_inf = data.iter().filter(|v| !v.is_finite() && !v.is_nan()).count();
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
                // Router stays F32 (small, projected with F32 dense kernel).
                assert_eq!(m.router.len(), expected_router_len);
                let router_finite = m.router.iter().all(|v| v.is_finite());
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
    #[ignore]
    fn load_real_apex_full_attn_layer_3() {
        let path = std::path::PathBuf::from(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf",
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
    #[ignore]
    fn load_real_apex_global_tensors() {
        let path = std::path::PathBuf::from(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf",
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

        let (embd, out_w, out_norm) =
            load_global_tensors(&gguf, &cfg, &device).expect("globals");
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
        eprintln!("  token_embd: {} values, stddev = {:.6}", embd.len(), embd_stddev);
        assert!(embd_stddev > 1e-6, "token_embd degenerate");
    }
}
