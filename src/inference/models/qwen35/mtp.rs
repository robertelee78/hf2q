//! Multi-Token Prediction (MTP) draft block for Qwen3.5.
//!
//! Qwen3.5 stores the single NextN/MTP block at `blk.{num_hidden_layers}`.
//! Wrapper tensors live under `blk.N.nextn.*`; the inner block itself uses
//! normal full-attention/dense-FFN tensor names at `blk.N.*`. The main verifier
//! stack never executes this block directly: speculative decoding calls
//! [`MtpWeights::forward_draft`] with the verifier hidden state and the
//! embedding of the just-accepted token.

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::elementwise::elementwise_add;
use mlx_native::ops::rms_norm;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::ffn::DenseFfnShape;
use super::gpu_ffn::{build_dense_ffn_layer_gpu, DenseFfnWeightsGpu};
use super::gpu_full_attn::{
    apply_imrope, apply_linear_projection_f32, apply_q_or_k_per_head_rms_norm,
    apply_sdpa_with_kv_cache, apply_sigmoid_gate_multiply,
};
use super::kv_cache::HybridKvCache;
use super::Qwen35Config;
use mlx_native::ops::fused_norm_add::dispatch_fused_residual_norm_f32;

pub use super::mtp_weights_load::load_mtp_weights_if_present;

/// Fully-loaded GPU MTP block. Projection weights are uploaded once as BF16;
/// residual activations and logits are F32.
pub struct MtpWeights {
    pub layer_index: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub intermediate_size: u32,
    pub(super) loaded_tensor_names: Vec<String>,
    pub(super) enorm: MlxBuffer,
    pub(super) hnorm: MlxBuffer,
    pub(super) eh_proj_embed: MlxBuffer,
    pub(super) eh_proj_hidden: MlxBuffer,
    #[allow(dead_code)]
    pub(super) embed_tokens: MlxBuffer,
    pub(super) shared_head_norm: MlxBuffer,
    pub(super) shared_head_head: MlxBuffer,
    pub(super) attn: MtpFullAttnWeightsGpu,
    pub(super) ffn: DenseFfnWeightsGpu,
}

pub(super) struct MtpFullAttnWeightsGpu {
    pub(super) attn_norm: MlxBuffer,
    pub(super) post_attn_norm: MlxBuffer,
    pub(super) wq: MlxBuffer,
    pub(super) wk: MlxBuffer,
    pub(super) wv: MlxBuffer,
    pub(super) w_gate: Option<MlxBuffer>,
    pub(super) attn_q_norm: MlxBuffer,
    pub(super) attn_k_norm: MlxBuffer,
    pub(super) wo: MlxBuffer,
}

impl MtpWeights {
    pub fn len(&self) -> usize {
        self.loaded_tensor_names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.loaded_tensor_names.is_empty()
    }

    pub fn has_tensor_suffix(&self, suffix: &str) -> bool {
        let direct_prefix = format!("blk.{}.", self.layer_index);
        let nextn_prefix = format!("blk.{}.nextn.", self.layer_index);
        self.loaded_tensor_names.iter().any(|name| {
            name.strip_prefix(&nextn_prefix) == Some(suffix)
                || name.strip_prefix(&direct_prefix) == Some(suffix)
        })
    }

    /// Run the MTP block for a single-token draft step.
    ///
    /// Inputs:
    /// - `prev_hidden`: verifier hidden state for token `t`, shape `[1, H]`.
    /// - `embed_t`: embedding for accepted token `t + 1`, shape `[1, H]`.
    /// - `position_ids`: IMROPE text positions for `t + 1`, flat `[4]`.
    ///
    /// Returns draft logits for token `t + 2`, shape `[1, vocab]`, F32.
    pub fn forward_draft(
        &self,
        prev_hidden: &MlxBuffer,
        embed_t: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        position_ids: &[i32],
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<MlxBuffer> {
        ensure!(
            position_ids.len() == 4,
            "MTP forward_draft expects exactly 4 IMROPE position ids, got {}",
            position_ids.len()
        );
        let h = self.hidden_size;
        ensure!(
            prev_hidden.element_count() == h as usize,
            "MTP prev_hidden has {} elements, expected {}",
            prev_hidden.element_count(),
            h
        );
        ensure!(
            embed_t.element_count() == h as usize,
            "MTP embed_t has {} elements, expected {}",
            embed_t.element_count(),
            h
        );

        let pos_buf = upload_i32(position_ids, device).context("MTP upload positions")?;
        let projected = self.project_embedding_and_hidden(embed_t, prev_hidden, device, registry)?;
        let attn_out = self.forward_full_attention(&projected, &pos_buf, kv_cache, device, registry, cfg)?;
        let hidden = self.forward_ffn_residual(&projected, &attn_out, device, registry, cfg)?;
        self.forward_shared_head(&hidden, device, registry, cfg.rms_norm_eps)
    }

    fn project_embedding_and_hidden(
        &self,
        embed_t: &MlxBuffer,
        prev_hidden: &MlxBuffer,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
    ) -> Result<MlxBuffer> {
        let h = self.hidden_size;
        let mut enc = device.command_encoder().context("MTP enc eh_proj")?;
        let embed_norm = rms_norm_with_weight(
            &mut enc,
            registry,
            device,
            embed_t,
            &self.enorm,
            1,
            h,
            1e-6,
        )?;
        let hidden_norm = rms_norm_with_weight(
            &mut enc,
            registry,
            device,
            prev_hidden,
            &self.hnorm,
            1,
            h,
            1e-6,
        )?;
        enc.memory_barrier();
        let embed_part = apply_linear_projection_f32(
            &mut enc,
            registry,
            device,
            &embed_norm,
            &self.eh_proj_embed,
            1,
            h,
            h,
        )?;
        let hidden_part = apply_linear_projection_f32(
            &mut enc,
            registry,
            device,
            &hidden_norm,
            &self.eh_proj_hidden,
            1,
            h,
            h,
        )?;
        enc.memory_barrier();
        let out = device
            .alloc_buffer((h as usize) * 4, DType::F32, vec![1, h as usize])
            .map_err(|e| anyhow!("MTP alloc eh_proj sum: {e}"))?;
        elementwise_add(
            &mut enc,
            registry,
            device.metal_device(),
            &embed_part,
            &hidden_part,
            &out,
            h as usize,
            DType::F32,
        )
        .context("MTP eh_proj sum")?;
        enc.commit();
        Ok(out)
    }

    fn forward_full_attention(
        &self,
        x: &MlxBuffer,
        positions: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<MlxBuffer> {
        let h = self.hidden_size;
        let q_total = cfg.num_attention_heads * cfg.head_dim;
        let kv_total = cfg.num_key_value_heads * cfg.head_dim;
        let attn = &self.attn;

        let (q_rope, k_rope, v_flat, gate_flat) = {
            let mut enc = device.command_encoder().context("MTP enc attn qkv")?;
            let x_norm = rms_norm_with_weight(
                &mut enc,
                registry,
                device,
                x,
                &attn.attn_norm,
                1,
                h,
                cfg.rms_norm_eps,
            )?;
            enc.memory_barrier();
            let q_flat = apply_linear_projection_f32(
                &mut enc, registry, device, &x_norm, &attn.wq, 1, h, q_total,
            )?;
            let k_flat = apply_linear_projection_f32(
                &mut enc, registry, device, &x_norm, &attn.wk, 1, h, kv_total,
            )?;
            let v_flat = apply_linear_projection_f32(
                &mut enc, registry, device, &x_norm, &attn.wv, 1, h, kv_total,
            )?;
            let gate_flat = match &attn.w_gate {
                Some(w) => Some(apply_linear_projection_f32(
                    &mut enc, registry, device, &x_norm, w, 1, h, q_total,
                )?),
                None => None,
            };
            enc.memory_barrier();
            let q_normed = apply_q_or_k_per_head_rms_norm(
                &mut enc,
                registry,
                device,
                &q_flat,
                &attn.attn_q_norm,
                1,
                cfg.num_attention_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?;
            let k_normed = apply_q_or_k_per_head_rms_norm(
                &mut enc,
                registry,
                device,
                &k_flat,
                &attn.attn_k_norm,
                1,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?;
            enc.memory_barrier();
            let q_rope = apply_imrope(
                &mut enc,
                registry,
                device,
                &q_normed,
                positions,
                1,
                cfg.num_attention_heads,
                cfg.head_dim,
                cfg.rotary_dim,
                cfg.rope_theta as f32,
                cfg.mrope_section,
            )?;
            let k_rope = apply_imrope(
                &mut enc,
                registry,
                device,
                &k_normed,
                positions,
                1,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rotary_dim,
                cfg.rope_theta as f32,
                cfg.mrope_section,
            )?;
            enc.commit();
            (q_rope, k_rope, v_flat, gate_flat)
        };

        let slot = kv_cache
            .mtp_slot
            .as_mut()
            .ok_or_else(|| anyhow!("MTP forward_draft requires HybridKvCache.mtp_slot"))?;
        let attn_out = apply_sdpa_with_kv_cache(
            device,
            registry,
            &q_rope,
            &k_rope,
            &v_flat,
            slot,
            1,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            kv_cache.max_seq_len,
        )
        .context("MTP SDPA")?;

        let mut enc = device.command_encoder().context("MTP enc attn output")?;
        let gated_or_attn = if let Some(gate) = gate_flat.as_ref() {
            apply_sigmoid_gate_multiply(
                &mut enc,
                registry,
                device,
                &attn_out,
                gate,
                q_total,
            )?
        } else {
            attn_out
        };
        let out = apply_linear_projection_f32(
            &mut enc,
            registry,
            device,
            &gated_or_attn,
            &attn.wo,
            1,
            q_total,
            h,
        )?;
        enc.commit();
        Ok(out)
    }

    fn forward_ffn_residual(
        &self,
        residual: &MlxBuffer,
        attn_out: &MlxBuffer,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<MlxBuffer> {
        let h = self.hidden_size;
        let ffn_input = device
            .alloc_buffer((h as usize) * 4, DType::F32, vec![1, h as usize])
            .map_err(|e| anyhow!("MTP alloc ffn_input: {e}"))?;
        let ffn_residual = device
            .alloc_buffer((h as usize) * 4, DType::F32, vec![1, h as usize])
            .map_err(|e| anyhow!("MTP alloc ffn_residual: {e}"))?;
        let mut enc = device.command_encoder().context("MTP enc residual norm")?;
        dispatch_fused_residual_norm_f32(
            &mut enc,
            registry,
            device.metal_device(),
            residual,
            attn_out,
            &self.attn.post_attn_norm,
            &ffn_input,
            Some(&ffn_residual),
            1,
            h,
            cfg.rms_norm_eps,
        )
        .context("MTP fused residual norm")?;
        enc.commit();

        build_dense_ffn_layer_gpu(
            device,
            registry,
            &ffn_input,
            &self.ffn,
            DenseFfnShape {
                hidden_size: h,
                intermediate_size: self.intermediate_size,
            },
            Some(&ffn_residual),
        )
        .context("MTP dense FFN")
    }

    fn forward_shared_head(
        &self,
        hidden: &MlxBuffer,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        eps: f32,
    ) -> Result<MlxBuffer> {
        let h = self.hidden_size;
        let normed = {
            let mut enc = device.command_encoder().context("MTP enc head norm")?;
            let out = rms_norm_with_weight(
                &mut enc,
                registry,
                device,
                hidden,
                &self.shared_head_norm,
                1,
                h,
                eps,
            )?;
            enc.commit();
            out
        };

        let mut enc = device.command_encoder().context("MTP enc shared head")?;
        let logits = apply_linear_projection_f32(
            &mut enc,
            registry,
            device,
            &normed,
            &self.shared_head_head,
            1,
            h,
            self.vocab_size,
        )
        .context("MTP shared head projection")?;
        enc.commit_and_wait().context("MTP commit logits")?;
        Ok(logits)
    }
}

fn rms_norm_with_weight(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    seq_len: u32,
    hidden_size: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    let out = device
        .alloc_buffer(
            (seq_len * hidden_size) as usize * 4,
            DType::F32,
            vec![seq_len as usize, hidden_size as usize],
        )
        .map_err(|e| anyhow!("alloc rms_norm out: {e}"))?;
    let mut params = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc rms_norm params: {e}"))?;
    {
        let s = params.as_mut_slice::<f32>().map_err(|e| anyhow!("{e}"))?;
        s[0] = eps;
        s[1] = hidden_size as f32;
    }
    rms_norm::dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        weight,
        &out,
        &params,
        seq_len,
        hidden_size,
    )
    .context("dispatch_rms_norm")?;
    Ok(out)
}

#[cfg(test)]
#[path = "mtp_tests.rs"]
mod tests;

fn upload_i32(data: &[i32], device: &MlxDevice) -> Result<MlxBuffer> {
    let mut buf = device
        .alloc_buffer(data.len() * 4, DType::I32, vec![data.len()])
        .map_err(|e| anyhow!("alloc i32 buffer: {e}"))?;
    buf.as_mut_slice::<i32>()
        .map_err(|e| anyhow!("i32 mut_slice: {e}"))?
        .copy_from_slice(data);
    Ok(buf)
}
