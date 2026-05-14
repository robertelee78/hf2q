//! DFlash drafter forward dispatch primitives (ADR-030 Phase 2).
//!
//! This module composes mlx-native dispatchers into the small forward
//! steps that make up `DFlashDraftModel.__call__`
//! (`/opt/dflash/dflash/model_mlx.py:181-198`) and `DFlashAttention.__call__`
//! (`model_mlx.py:82-116`).
//!
//! Builds up incrementally — each commit adds one composable piece with
//! its own smoke test against real drafter weights on M5 Max Metal.
//! Per mantra (no stubs, no fallback): every function COMPUTES SOMETHING
//! USEFUL and ships tested. The final `DFlashDraftModel::forward` is
//! composed from these pieces in a later iter.
//!
//! ## Current iter (iter-18): input_layernorm + Q projection
//!
//! ```text
//!   h: [L, hidden] F32
//!         │
//!         ▼  input_layernorm (RMSNorm, weight from DFlashLayerTensors)
//!   normed: [L, hidden] F32
//!         │
//!         ▼  q_proj (BF16 weight @ F32 input via apply_linear_projection_f32)
//!   q: [L, num_q_heads * head_dim] F32
//! ```
//!
//! Future iters add: K/V proj, q_norm/k_norm, RoPE, SDPA, concat, O proj
//! (attention complete); SwiGLU MLP; layer composition (residuals);
//! model composition (fc + per-layer + final norm + lm_head).

use super::config::DFlashConfig;
use super::tensors::DFlashLayerTensors;
use anyhow::{anyhow, Context, Result};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};
use mlx_native::ops::rms_norm::dispatch_rms_norm;
use crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32;

/// Build the `[eps, dim]` F32 params buffer required by [`dispatch_rms_norm`].
///
/// Per `mlx_native::ops::rms_norm` doc, the kernel reads two F32 values
/// from this buffer. We allocate a tiny dedicated buffer per call; the
/// cost is amortized over the rms_norm dispatch itself.
fn alloc_rms_norm_params(device: &MlxDevice, eps: f32, dim: u32) -> Result<MlxBuffer> {
    let mut params = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc rms_norm params: {e}"))?;
    let slice = params
        .as_mut_slice::<f32>()
        .map_err(|e| anyhow!("rms_norm params slice: {e}"))?;
    slice[0] = eps;
    slice[1] = dim as f32;
    Ok(params)
}

/// Dispatch input_layernorm (`RMSNorm(h)`) into the encoder.
///
/// Mirrors `model_mlx.py:DFlashDecoderLayer.__call__` first stage:
/// `self.input_layernorm(x)`.
///
/// # Arguments
///
/// - `h_input`: `[L, hidden_size]` F32 input hidden state
/// - `layer`: per-layer drafter weights (uses `input_layernorm`)
/// - Returns: `[L, hidden_size]` F32 normalized output
///
/// The output buffer is allocated fresh; the encoder is NOT committed
/// by this function (caller composes multiple dispatches before commit).
pub fn dispatch_dflash_input_layernorm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    h_input: &MlxBuffer,
    layer: &DFlashLayerTensors,
    cfg: &DFlashConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size as u32;
    let element_count = (seq_len as usize) * (hidden as usize);
    if h_input.element_count() != element_count {
        return Err(anyhow!(
            "dflash input_layernorm: h_input element count {} != L({}) * hidden({})",
            h_input.element_count(),
            seq_len,
            hidden
        ));
    }
    let normed = device
        .alloc_buffer(element_count * 4, DType::F32, vec![seq_len as usize, hidden as usize])
        .map_err(|e| anyhow!("alloc input_layernorm output: {e}"))?;
    let params = alloc_rms_norm_params(device, cfg.rms_norm_eps, hidden)?;
    dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        h_input,
        &layer.input_layernorm,
        &normed,
        &params,
        seq_len,
        hidden,
    )
    .context("dispatch_rms_norm input_layernorm")?;
    Ok(normed)
}

/// Dispatch the Q projection: `q_proj @ normed`.
///
/// Mirrors `model_mlx.py:DFlashAttention.__call__` line 92:
/// `queries = self.q_proj(x)`.
///
/// # Arguments
///
/// - `normed`: `[L, hidden_size]` F32 input (output of input_layernorm)
/// - `layer`: per-layer drafter weights (uses `q_proj` BF16 weight
///   shape `[num_q_heads * head_dim, hidden_size]`)
/// - Returns: `[L, num_q_heads * head_dim]` F32 (unreshaped)
///
/// Reshape into `[L, num_q_heads, head_dim]` for q_norm + RoPE happens
/// in a later piece. The encoder is NOT committed.
pub fn dispatch_dflash_q_proj(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    normed: &MlxBuffer,
    layer: &DFlashLayerTensors,
    cfg: &DFlashConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size as u32;
    let q_out_dim = (cfg.num_attention_heads * cfg.head_dim) as u32;
    apply_linear_projection_f32(
        encoder,
        registry,
        device,
        normed,
        &layer.q_proj,
        seq_len,
        hidden,
        q_out_dim,
    )
    .context("dispatch_dflash_q_proj")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::dflash::{
        config::DFlashConfig,
        tensors::DFlashModelTensors,
        weights::{DFlashWeights, DFlashWeightsFile},
    };

    fn gemma4_26b_a4b_dflash_config() -> DFlashConfig {
        DFlashConfig::from_json_str(
            crate::inference::spec_decode::dflash::config::tests::GEMMA4_26B_A4B_DFLASH_CONFIG,
        )
        .expect("test fixture must parse")
    }

    /// Smoke test: load real drafter weights to GPU, dispatch
    /// input_layernorm + q_proj on a deterministic F32 input, verify
    /// the Q output shape + that all values are finite.
    ///
    /// At zero-mean ones input through RMSNorm:
    ///   sum_sq = hidden * 1 = hidden
    ///   inv_rms = sqrt(hidden/hidden + eps).recip() ≈ 1
    ///   normed[i] = 1 * 1 * weight[i] = weight[i]
    /// Then q_proj is a regular matmul with finite BF16 weights — all
    /// outputs are finite if and only if the dispatch plumbing works.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_input_norm_and_q_proj() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        // Load + upload weights.
        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");

        // Allocate a [L=block_size, hidden] F32 input filled with 1.0.
        let block_size = 8u32; // K=7 per Phase 1.5 optimal
        let hidden = cfg.hidden_size as u32;
        let elem = (block_size as usize) * (hidden as usize);
        let mut h_input = device
            .alloc_buffer(elem * 4, DType::F32, vec![block_size as usize, hidden as usize])
            .expect("alloc h_input");
        {
            let slice = h_input.as_mut_slice::<f32>().expect("h_input slice");
            for v in slice.iter_mut() {
                *v = 1.0;
            }
        }

        // Dispatch into a single encoder.
        let mut encoder = device.command_encoder().expect("encoder");
        let layer = &tensors.layers[0];
        let normed = dispatch_dflash_input_layernorm(
            &mut encoder, &mut registry, &device, &h_input, layer, &cfg, block_size,
        )
        .expect("input_layernorm dispatch");
        encoder.memory_barrier(); // RAW: q_proj reads `normed`
        let q = dispatch_dflash_q_proj(
            &mut encoder, &mut registry, &device, &normed, layer, &cfg, block_size,
        )
        .expect("q_proj dispatch");
        encoder.commit_and_wait().expect("commit");

        // Validate shape: [block_size, num_q_heads * head_dim]
        let q_dim = (cfg.num_attention_heads * cfg.head_dim) as usize;
        assert_eq!(q.element_count(), (block_size as usize) * q_dim);
        assert_eq!(q.byte_len(), q.element_count() * 4); // F32

        // Validate finite values via host read.
        let host: &[f32] = q.as_slice::<f32>().expect("q host slice");
        let n_total = host.len();
        let n_finite = host.iter().filter(|v| v.is_finite()).count();
        let n_zero = host.iter().filter(|v| **v == 0.0).count();
        assert_eq!(
            n_finite, n_total,
            "all Q output values must be finite (got {n_finite}/{n_total}; n_zero={n_zero})"
        );
        // Sanity: not all zeros — input was 1.0s through a non-trivial transform.
        assert!(n_zero < n_total / 2, "Q output suspiciously sparse (n_zero={n_zero}/{n_total})");
    }
}
