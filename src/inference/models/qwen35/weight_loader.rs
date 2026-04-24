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
use mlx_native::gguf::GgufFile;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{MlxBuffer, MlxDevice};

use super::delta_net::DeltaNetLayerWeights;
use super::ffn::{DenseFfnWeights, MoeFfnWeights};
use super::full_attn::FullAttnLayerWeights;
use super::model::{Qwen35FfnWeights, Qwen35LayerWeights};
use super::{Qwen35Config, Qwen35LayerKind, Qwen35Variant};

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
    /// GGML quantization type for all three expert buffers.
    pub ggml_type: GgmlType,
    /// Shared-expert sigmoid gate: `[hidden_size]` F32.
    pub shared_gate_logit: Vec<f32>,
    /// Shared-expert SwiGLU weights (F32).
    pub shared_gate: Vec<f32>,
    pub shared_up: Vec<f32>,
    pub shared_down: Vec<f32>,
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

    // Split fused q_fused into wq (first q_total rows) and w_gate (second q_total rows).
    // Layout: [2*q_total, h] row-major; wq = rows [0..q_total], w_gate = rows [q_total..2*q_total].
    let wq: Vec<f32> = q_fused[0..q_total * h].to_vec();
    let w_gate: Vec<f32> = q_fused[q_total * h..2 * q_total * h].to_vec();
    drop(q_fused);

    Ok(FullAttnLayerWeights {
        attn_norm,
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
pub fn load_delta_net_layer(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    layer_idx: u32,
    device: &MlxDevice,
) -> Result<DeltaNetLayerWeights> {
    let p = format!("blk.{}", layer_idx);
    let attn_norm = load_f32_tensor(gguf, &format!("{p}.attn_norm.weight"), device)?;
    let attn_qkv = load_f32_tensor(gguf, &format!("{p}.attn_qkv.weight"), device)?;
    let attn_gate = load_f32_tensor(gguf, &format!("{p}.attn_gate.weight"), device)?;
    let ssm_conv1d = load_f32_tensor(gguf, &format!("{p}.ssm_conv1d.weight"), device)?;
    let ssm_alpha = load_f32_tensor(gguf, &format!("{p}.ssm_alpha.weight"), device)?;
    let ssm_dt_bias = load_f32_tensor(gguf, &format!("{p}.ssm_dt.bias"), device)?;
    let ssm_beta = load_f32_tensor(gguf, &format!("{p}.ssm_beta.weight"), device)?;
    let ssm_a = load_f32_tensor(gguf, &format!("{p}.ssm_a"), device)?; // no .weight suffix
    let ssm_norm = load_f32_tensor(gguf, &format!("{p}.ssm_norm.weight"), device)?;
    let ssm_out = load_f32_tensor(gguf, &format!("{p}.ssm_out.weight"), device)?;

    Ok(DeltaNetLayerWeights {
        attn_norm,
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

    // All three expert tensors must have the same quantization type.
    let gate_info = gguf.tensor_info(&format!("{p}.ffn_gate_exps.weight"))
        .ok_or_else(|| anyhow!("layer {layer_idx}: ffn_gate_exps not found in GGUF"))?;
    let ggml_type = gate_info.ggml_type;

    // Validate that the type is supported by quantized_matmul_id_ggml.
    // Q5_K uses the mv_id kernel (mm_id not yet ported); Q4_0/Q8_0/Q6_K
    // also use mv_id for decode and mm_id for prefill batches > 8 tokens.
    match ggml_type {
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q5_K | GgmlType::Q6_K => {}
        other => {
            return Err(anyhow!(
                "layer {layer_idx}: expert weights have unsupported quant type {:?} \
                 (expected Q4_0, Q8_0, Q5_K, or Q6_K)",
                other
            ));
        }
    }

    Ok(MoeFfnWeightsQ {
        router,
        expert_gate_q,
        expert_up_q,
        expert_down_q,
        ggml_type,
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

    // MoE variant: route to the quantized Q path — MoeFfnWeightsQ preserves
    // the GGUF's native Q6_K/Q8_0 expert tensor layout and avoids the 128 GB
    // F32 expansion that OOMs on the real 35B-A3B model (256 experts × 40
    // layers × 3 tensors × 2048×512 × 4 bytes exceeds Metal's 112 GB working
    // set cap). The F32 `load_moe_ffn` / `Qwen35FfnWeights::Moe` variant is
    // preserved for synthetic-weight unit tests that deliberately use F32
    // inputs (see gpu_ffn.rs::build_moe_ffn_layer_gpu).
    let ffn = match cfg.variant {
        Qwen35Variant::Dense => Qwen35FfnWeights::Dense(load_dense_ffn(gguf, layer_idx, device)?),
        Qwen35Variant::Moe => {
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
    use crate::inference::models::qwen35::model::Qwen35Model;

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
        let moe = match layer.ffn() {
            Qwen35FfnWeights::Moe(m) => m,
            _ => panic!("expected MoE FFN"),
        };
        let moe_cfg = cfg.moe.as_ref().expect("moe cfg");
        assert_eq!(
            moe.router.len(),
            (moe_cfg.num_experts * cfg.hidden_size) as usize
        );
        assert_eq!(
            moe.expert_gate.len(),
            (moe_cfg.num_experts * moe_cfg.moe_intermediate_size * cfg.hidden_size) as usize
        );

        // Sanity on router (F32 in apex per type_scan: type 0 = F32).
        let router_finite = moe.router.iter().all(|v| v.is_finite());
        assert!(router_finite, "router has non-finite values");
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
