//! Multi-Token Prediction (MTP) draft block for Qwen3.5.
//!
//! Qwen3.5 stores the single NextN/MTP block at `blk.{num_hidden_layers}`.
//! Wrapper tensors live under `blk.N.nextn.*`; the inner block itself uses
//! normal full-attention/dense-FFN tensor names at `blk.N.*`. The main verifier
//! stack never executes this block directly: speculative decoding calls
//! [`MtpWeights::forward_draft`] with the verifier hidden state and the
//! embedding of the just-accepted token.

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::argmax::dispatch_argmax_f32;
use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::ops::rms_norm;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::ffn::{DenseFfnShape, MoeFfnShape};
use super::gpu_ffn::{
    build_dense_ffn_layer_gpu, build_dense_ffn_layer_gpu_q, build_moe_ffn_layer_gpu_q_into,
    DenseFfnWeightsGpu, DenseFfnWeightsGpuQ, MoeFfnWeightsGpuQ,
};
use super::gpu_full_attn::{
    append_kv_to_cache_without_attention, apply_imrope, apply_linear_projection_f32_with_ggml_type,
    apply_q_gate_projection_f32, apply_q_or_k_per_head_rms_norm, apply_sdpa_with_kv_cache,
    apply_sdpa_with_kv_cache_decode_into, apply_sigmoid_gate_multiply, FullAttnQGateWeightsGpu,
};
use super::joint_decode_core::row_view;
use super::kv_cache::HybridKvCache;
use super::Qwen35Config;
use crate::serve::forward_mlx_shared::MlxQWeight;
use mlx_native::ops::fused_norm_add::dispatch_fused_residual_norm_f32;

pub use super::mtp_weights_load::{
    load_mtp_weights_if_present, load_mtp_weights_if_present_with_shared_head,
};

/// Fully-loaded GPU MTP block. GGUF projections retain their declared native
/// representation; residual activations and logits are F32.
pub struct MtpWeights {
    pub layer_index: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    /// For dense MTP this is the dense FFN intermediate dim. For MoE MTP
    /// it's the per-expert (moe) intermediate dim — useful for diagnostics
    /// only; dispatch consults [`MtpFfnWeightsGpu`] directly.
    pub intermediate_size: u32,
    pub(super) loaded_tensor_names: Vec<String>,
    pub(super) enorm: MlxBuffer,
    pub(super) hnorm: MlxBuffer,
    pub(super) eh_proj: MlxBuffer,
    pub(super) eh_proj_ggml_type: GgmlType,
    /// MTP token-embedding table.
    ///
    /// `Some(...)` when the GGUF carries a dedicated `blk.{N}.nextn.embed_tokens.weight`
    /// (Qwen3.5 MTP convention; HF flag `mtp_use_dedicated_embeddings == True`).
    ///
    /// `None` when the MTP block shares the main verifier's `token_embd.weight`
    /// (Qwen3.6 27B + 35B-A3B convention; HF flag `False`). At draft time the
    /// caller of `forward_draft` already supplies the embedding (`embed_t`); the
    /// verifier embedding table itself lives on `Qwen35Model::token_embd` and is
    /// reused via the hot embed_tokens lookup path — no buffer duplication.
    #[allow(dead_code)]
    pub(super) embed_tokens: Option<MlxQWeight>,
    pub(super) shared_head_norm: MlxBuffer,
    pub(super) shared_head_head: MlxBuffer,
    pub(super) shared_head_head_ggml_type: GgmlType,
    pub(super) attn: MtpFullAttnWeightsGpu,
    pub(super) ffn: MtpFfnWeightsGpu,
}

/// Inner-FFN variant for the MTP block.
///
/// Qwen 3.6 27B dense-MTP target emits a SwiGLU dense FFN at the MTP block:
/// `blk.{N}.ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight`.
///
/// Qwen 3.5/3.6 35B-A3B MoE-MTP target emits the same MoE FFN schema used by
/// regular MoE layers at the MTP block: 8 tensors (`ffn_gate_inp`,
/// `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, plus 4 shared-expert).
/// The MoE variant uses the production quantized path
/// ([`MoeFfnWeightsGpuQ`]) so expert weights stay native GGML blocks on Metal,
/// matching the rest of the verifier stack (no F32 expansion).
pub(super) enum MtpFfnWeightsGpu {
    /// Explicit floating-point storage, retained and executed as F32.
    Dense {
        weights: DenseFfnWeightsGpu,
        intermediate_size: u32,
    },
    /// Dense SwiGLU FFN retaining the GGUF's native quantized blocks.
    DenseQ { weights: DenseFfnWeightsGpuQ },
    /// Quantized MoE FFN (Qwen 3.5/3.6 35B-A3B MoE-MTP convention).
    Moe {
        weights: MoeFfnWeightsGpuQ,
        shape: MoeFfnShape,
    },
}

pub(super) struct MtpFullAttnWeightsGpu {
    pub(super) attn_norm: MlxBuffer,
    pub(super) post_attn_norm: MlxBuffer,
    pub(super) q_gate: MtpQGateWeightsGpu,
    pub(super) wk: MlxBuffer,
    pub(super) wk_ggml_type: GgmlType,
    pub(super) wv: MlxBuffer,
    pub(super) wv_ggml_type: GgmlType,
    pub(super) attn_q_norm: MlxBuffer,
    pub(super) attn_k_norm: MlxBuffer,
    pub(super) wo: MlxBuffer,
    pub(super) wo_ggml_type: GgmlType,
}

pub(super) enum MtpQGateWeightsGpu {
    Ungated {
        wq: MlxBuffer,
        wq_ggml_type: GgmlType,
    },
    Gated(FullAttnQGateWeightsGpu),
}

/// Test-friendly indicator for which inner-FFN variant a loaded MTP block
/// carries. Used by integration tests that need to assert the loader took
/// the dense or MoE path on a real GGUF without exposing the GPU buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpFfnKind {
    Dense,
    Moe,
}

struct MtpDraftBody {
    hidden: MlxBuffer,
    profile_enabled: bool,
    projection_ms: f64,
    attention_ms: f64,
    ffn_ms: f64,
}

/// One lane of [`MtpWeights::forward_draft_greedy_batch`].
#[derive(Clone, Copy, Debug)]
pub struct MtpDraftLane<'a> {
    pub slot_id: crate::serve::multi_seq_kv::SlotId,
    /// `[1, H]` hidden carried into this depth: the verifier's carry row at
    /// depth 0, the previous depth's normalized MTP row after it.
    pub prev_hidden: &'a MlxBuffer,
    /// The token this depth embeds (the pending token, then each draft).
    pub token: u32,
    /// Its text position; the slot's MTP cursor must equal it.
    pub position: i32,
}

impl MtpWeights {
    pub fn len(&self) -> usize {
        self.loaded_tensor_names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.loaded_tensor_names.is_empty()
    }

    /// Variant indicator for the inner FFN block. Mainly used by tests
    /// validating that the loader picked the right dispatch path for a
    /// given GGUF (dense for Qwen 3.6 27B; MoE for Qwen 3.5/3.6 35B-A3B).
    pub fn ffn_kind(&self) -> MtpFfnKind {
        match &self.ffn {
            MtpFfnWeightsGpu::Dense { .. } => MtpFfnKind::Dense,
            MtpFfnWeightsGpu::DenseQ { .. } => MtpFfnKind::Dense,
            MtpFfnWeightsGpu::Moe { .. } => MtpFfnKind::Moe,
        }
    }

    pub fn has_tensor_suffix(&self, suffix: &str) -> bool {
        let direct_prefix = format!("blk.{}.", self.layer_index);
        let nextn_prefix = format!("blk.{}.nextn.", self.layer_index);
        self.loaded_tensor_names.iter().any(|name| {
            name.strip_prefix(&nextn_prefix) == Some(suffix)
                || name.strip_prefix(&direct_prefix) == Some(suffix)
        })
    }

    /// Resolve the token embedding table required by this MTP artifact.
    /// Shared-table Qwen3.8 returns the caller-provided main embeddings;
    /// dedicated-table variants gather selected rows directly from their
    /// declared native representation.
    pub fn embeddings_for_tokens(
        &self,
        tokens: &[u32],
        shared_embed_rows: &MlxBuffer,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
    ) -> Result<MlxBuffer> {
        ensure!(
            !tokens.is_empty(),
            "MTP embeddings: tokens must be non-empty"
        );
        let hidden_size = self.hidden_size as usize;
        ensure!(
            shared_embed_rows.element_count() == tokens.len() * hidden_size,
            "MTP embeddings: shared rows elements {} != {}",
            shared_embed_rows.element_count(),
            tokens.len() * hidden_size
        );
        let Some(dedicated) = self.embed_tokens.as_ref() else {
            return Ok(shared_embed_rows.clone());
        };
        ensure!(
            dedicated.affine.is_none() && dedicated.info.cols == hidden_size,
            "MTP dedicated embedding must be a native [vocab,{hidden_size}] GGUF tensor"
        );
        let vocab = dedicated.info.rows;
        let mut output = device
            .alloc_buffer(
                tokens.len() * hidden_size * 4,
                DType::F32,
                vec![tokens.len(), hidden_size],
            )
            .map_err(|e| anyhow!("MTP alloc dedicated embeddings: {e}"))?;
        let vocab_u32 =
            u32::try_from(vocab).context("MTP dedicated embedding vocab exceeds u32")?;
        super::forward_gpu::embed_tokens_gpu_into(
            tokens,
            &[],
            Some(dedicated),
            vocab_u32,
            self.hidden_size,
            device,
            registry,
            &mut output,
            #[cfg(test)]
            None,
        )
        .context("MTP dedicated native embedding gather")?;
        Ok(output)
    }

    /// Run the MTP block for a single-token draft step. Convenience wrapper
    /// over [`MtpWeights::forward_draft_with_hidden`] that drops the returned
    /// hidden buffer — use the `_with_hidden` variant when you intend to
    /// chain a second MTP step (K=N speculative decoding).
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
        slot_id: crate::serve::multi_seq_kv::SlotId,
        position_ids: &[i32],
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<MlxBuffer> {
        let (logits, _hidden) = self.forward_draft_with_hidden(
            prev_hidden,
            embed_t,
            kv_cache,
            slot_id,
            position_ids,
            device,
            registry,
            cfg,
        )?;
        Ok(logits)
    }

    /// Artifact-aware draft entry point. Callers provide the verifier's
    /// shared embedding row, but a dedicated-table artifact replaces it with
    /// its own exact row before executing the draft block.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_draft_for_token(
        &self,
        prev_hidden: &MlxBuffer,
        token: u32,
        shared_embed_t: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        position_ids: &[i32],
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<(MlxBuffer, MlxBuffer)> {
        let embed = self.embeddings_for_tokens(&[token], shared_embed_t, device, registry)?;
        self.forward_draft_with_hidden(
            prev_hidden,
            &embed,
            kv_cache,
            slot_id,
            position_ids,
            device,
            registry,
            cfg,
        )
    }

    /// Greedy artifact-aware draft entry point for the server hot path.
    ///
    /// This is semantically identical to `forward_draft_for_token` followed
    /// by GPU argmax, but encodes shared-head RMSNorm, vocabulary projection,
    /// and argmax in one command buffer with a single terminal wait. The
    /// target verifier remains authoritative; this only removes an avoidable
    /// host/GPU round trip from each chained draft token.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_draft_greedy_for_token(
        &self,
        prev_hidden: &MlxBuffer,
        token: u32,
        shared_embed_t: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        position_ids: &[i32],
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<(u32, MlxBuffer)> {
        let embed = self.embeddings_for_tokens(&[token], shared_embed_t, device, registry)?;
        let body = self.forward_draft_body(
            prev_hidden,
            &embed,
            kv_cache,
            slot_id,
            position_ids,
            device,
            registry,
            cfg,
        )?;
        let head_started = std::time::Instant::now();
        let (drafts, nextn_hidden) =
            self.forward_shared_head_greedy(&body.hidden, device, registry, cfg.rms_norm_eps)?;
        self.report_draft_profile(&body, head_started.elapsed().as_secs_f64() * 1000.0);
        Ok((drafts[0], nextn_hidden))
    }

    /// One greedy MTP draft step for several slots in one forward (joint
    /// MTP drafting, one call per draft depth). Row `i` is `lanes[i]`:
    /// its K/V is written at its slot's MTP cursor (advancing it) and it
    /// attends over its own slot through the single-row decode SDPA; the
    /// projections, norms, FFN and shared head run over the packed rows, and
    /// each row's draft is one GPU argmax over its own logits. Returns per
    /// lane its draft and its normalized hidden row, the next depth's
    /// `prev_hidden`.
    ///
    /// Deterministic. A packed row may differ in its last bits from the
    /// one-lane [`Self::forward_draft_greedy_for_token`]; drafts only decide
    /// how many rows the target accepts, never the committed tokens.
    /// `shared_embed_rows` holds the verifier's embedding of every lane's
    /// token, `[lanes, H]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_draft_greedy_batch(
        &self,
        lanes: &[MtpDraftLane<'_>],
        shared_embed_rows: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<Vec<(u32, MlxBuffer)>> {
        ensure!(!lanes.is_empty(), "MTP batched draft needs a lane");
        let rows = lanes.len();
        let hs = self.hidden_size as usize;
        for (index, lane) in lanes.iter().enumerate() {
            ensure!(
                lane.prev_hidden.element_count() == hs,
                "MTP batched draft lane {index} prev_hidden has {} elements, expected {hs}",
                lane.prev_hidden.element_count()
            );
            ensure!(
                lanes[..index]
                    .iter()
                    .all(|earlier| earlier.slot_id != lane.slot_id),
                "MTP batched draft slot {} appears twice",
                lane.slot_id.0
            );
        }
        let tokens = lanes.iter().map(|lane| lane.token).collect::<Vec<_>>();
        let embed = self.embeddings_for_tokens(&tokens, shared_embed_rows, device, registry)?;
        let prev = device
            .alloc_buffer(rows * hs * 4, DType::F32, vec![rows, hs])
            .map_err(|e| anyhow!("MTP alloc batched prev hidden: {e}"))?;
        {
            let mut enc = device
                .command_encoder()
                .map(crate::inference::native_observation::encoder)
                .context("MTP enc batched prev hidden")?;
            for (row, lane) in lanes.iter().enumerate() {
                dispatch_copy_f32(
                    &mut enc,
                    registry,
                    device.metal_device(),
                    lane.prev_hidden,
                    &prev,
                    0,
                    row * hs,
                    hs,
                )
                .with_context(|| format!("MTP pack prev hidden row {row}"))?;
            }
            enc.commit();
        }
        let mut positions = vec![0_i32; 4 * rows];
        for axis in 0..4 {
            for (row, lane) in lanes.iter().enumerate() {
                positions[axis * rows + row] = lane.position;
            }
        }
        let pos_buf = upload_i32(&positions, device).context("MTP upload batched positions")?;
        let rows_u32 = u32::try_from(rows).context("MTP batched draft rows exceed u32")?;
        let projected = self.project_embedding_and_hidden(
            &embed,
            &prev,
            rows_u32,
            device,
            registry,
            "adr065.mtp_draft",
        )?;
        let slots = lanes.iter().map(|lane| lane.slot_id).collect::<Vec<_>>();
        let attn_out = self.forward_full_attention_rows(
            &projected, &pos_buf, kv_cache, &slots, device, registry, cfg,
        )?;
        let hidden = self.forward_ffn_residual(&projected, &attn_out, device, registry, cfg)?;
        let (drafts, normed) =
            self.forward_shared_head_greedy(&hidden, device, registry, cfg.rms_norm_eps)?;
        drafts
            .into_iter()
            .enumerate()
            .map(|(row, draft)| {
                Ok((
                    draft,
                    super::spec_decode::nth_hidden_row(&normed, self.hidden_size, row as u64)?,
                ))
            })
            .collect()
    }

    /// Attention of the batched draft step: Q/K/V, norms and IMROPE over
    /// the packed rows, then per row the single-row decode route against
    /// its own slot's MTP cache (K/V write at the slot cursor, SDPA over
    /// it), then the gate and output projection over the packed rows.
    #[allow(clippy::too_many_arguments)]
    fn forward_full_attention_rows(
        &self,
        x: &MlxBuffer,
        positions: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        slots: &[crate::serve::multi_seq_kv::SlotId],
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<MlxBuffer> {
        let h = self.hidden_size;
        let rows = slots.len();
        let rows_u32 = rows as u32;
        let (n_heads, n_kv_heads, head_dim) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        ensure!(
            head_dim % 32 == 0,
            "MTP batched draft attention requires head_dim % 32 == 0, got {head_dim}"
        );
        let q_total = n_heads * head_dim;
        let kv_total = n_kv_heads * head_dim;
        let attn = &self.attn;
        let max_seq_len = kv_cache.max_seq_len;
        let slot = kv_cache
            .mtp_slot
            .as_mut()
            .context("MTP batched draft requires HybridKvCache.mtp_slot")?;
        let mut enc = device
            .command_encoder()
            .map(crate::inference::native_observation::encoder)
            .context("MTP enc batched attention")?;
        let x_norm = rms_norm_with_weight(
            &mut enc,
            registry,
            device,
            x,
            &attn.attn_norm,
            rows_u32,
            h,
            cfg.rms_norm_eps,
        )?;
        enc.memory_barrier();
        let (q_flat, gate_flat) = match &attn.q_gate {
            MtpQGateWeightsGpu::Ungated { wq, wq_ggml_type } => (
                apply_linear_projection_f32_with_ggml_type(
                    &mut enc,
                    registry,
                    device,
                    &x_norm,
                    wq,
                    *wq_ggml_type,
                    rows_u32,
                    h,
                    q_total,
                )?,
                None,
            ),
            MtpQGateWeightsGpu::Gated(q_gate) => {
                let (q, gate) = apply_q_gate_projection_f32(
                    &mut enc, registry, device, &x_norm, q_gate, rows_u32, n_heads, head_dim, h,
                )?;
                (q, Some(gate))
            }
        };
        let k_flat = apply_linear_projection_f32_with_ggml_type(
            &mut enc,
            registry,
            device,
            &x_norm,
            &attn.wk,
            attn.wk_ggml_type,
            rows_u32,
            h,
            kv_total,
        )?;
        let v_flat = apply_linear_projection_f32_with_ggml_type(
            &mut enc,
            registry,
            device,
            &x_norm,
            &attn.wv,
            attn.wv_ggml_type,
            rows_u32,
            h,
            kv_total,
        )?;
        enc.memory_barrier();
        let q_normed = apply_q_or_k_per_head_rms_norm(
            &mut enc,
            registry,
            device,
            &q_flat,
            &attn.attn_q_norm,
            rows_u32,
            n_heads,
            head_dim,
            cfg.rms_norm_eps,
        )?;
        let k_normed = apply_q_or_k_per_head_rms_norm(
            &mut enc,
            registry,
            device,
            &k_flat,
            &attn.attn_k_norm,
            rows_u32,
            n_kv_heads,
            head_dim,
            cfg.rms_norm_eps,
        )?;
        enc.memory_barrier();
        let q_rope = apply_imrope(
            &mut enc,
            registry,
            device,
            &q_normed,
            positions,
            rows_u32,
            n_heads,
            head_dim,
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
            rows_u32,
            n_kv_heads,
            head_dim,
            cfg.rotary_dim,
            cfg.rope_theta as f32,
            cfg.mrope_section,
        )?;
        enc.memory_barrier();
        let (nh, nkv, hd) = (n_heads as usize, n_kv_heads as usize, head_dim as usize);
        let mut row_outputs = Vec::with_capacity(rows);
        for (row, &slot_id) in slots.iter().enumerate() {
            if row > 0 {
                enc.memory_barrier();
            }
            row_outputs.push(
                apply_sdpa_with_kv_cache_decode_into(
                    &mut enc,
                    device,
                    registry,
                    &row_view(&q_rope, row, vec![1, nh, hd])?,
                    &row_view(&k_rope, row, vec![1, nkv, hd])?,
                    &row_view(&v_flat, row, vec![1, kv_total as usize])?,
                    slot,
                    1,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_seq_len,
                    slot_id,
                )
                .with_context(|| format!("MTP batched draft SDPA row {row}"))?,
            );
        }
        enc.memory_barrier();
        let qt = q_total as usize;
        let attention = device
            .alloc_buffer(rows * qt * 4, DType::F32, vec![rows, qt])
            .map_err(|e| anyhow!("MTP alloc batched attention: {e}"))?;
        for (row, output) in row_outputs.iter().enumerate() {
            dispatch_copy_f32(
                &mut enc,
                registry,
                device.metal_device(),
                output,
                &attention,
                0,
                row * qt,
                qt,
            )
            .with_context(|| format!("MTP gather attention row {row}"))?;
        }
        enc.memory_barrier();
        let gated_or_attn = match gate_flat.as_ref() {
            Some(gate) => {
                let gated = apply_sigmoid_gate_multiply(
                    &mut enc,
                    registry,
                    device,
                    &attention,
                    gate,
                    rows_u32 * q_total,
                )?;
                enc.memory_barrier();
                gated
            }
            None => attention,
        };
        let out = apply_linear_projection_f32_with_ggml_type(
            &mut enc,
            registry,
            device,
            &gated_or_attn,
            &attn.wo,
            attn.wo_ggml_type,
            rows_u32,
            q_total,
            h,
        )?;
        enc.commit();
        Ok(out)
    }

    /// Same as [`forward_draft`] but also returns the MTP block's normalized
    /// `h_nextn` row (AFTER `shared_head_norm`). The normalized row is the
    /// only valid input for a chained second MTP step.
    ///
    /// Shape contract: `hidden.element_count() == hidden_size` (single-token
    /// draft step).
    pub fn forward_draft_with_hidden(
        &self,
        prev_hidden: &MlxBuffer,
        embed_t: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        position_ids: &[i32],
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<(MlxBuffer, MlxBuffer)> {
        let body = self.forward_draft_body(
            prev_hidden,
            embed_t,
            kv_cache,
            slot_id,
            position_ids,
            device,
            registry,
            cfg,
        )?;
        let head_started = std::time::Instant::now();
        let (logits, nextn_hidden) =
            self.forward_shared_head(&body.hidden, device, registry, cfg.rms_norm_eps)?;
        self.report_draft_profile(&body, head_started.elapsed().as_secs_f64() * 1000.0);
        Ok((logits, nextn_hidden))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_draft_body(
        &self,
        prev_hidden: &MlxBuffer,
        embed_t: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        position_ids: &[i32],
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<MtpDraftBody> {
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

        // ADR-028 iter-156: per-sub-step GPU-timing harness. Sets
        // commit_and_wait barriers between sub-steps when HF2Q_MTP_PROFILE=1
        // is set. Measurement-only — adds ~1-2ms total per draft. Default
        // path commits each sub-step's CB without sync (Apple Metal pipelines
        // them across the boundary).
        let mtp_substep_profile = std::env::var("HF2Q_MTP_PROFILE").as_deref() == Ok("1");
        let pos_buf = upload_i32(position_ids, device).context("MTP upload positions")?;

        let t0 = std::time::Instant::now();
        let projected = self.project_embedding_and_hidden(
            embed_t,
            prev_hidden,
            1,
            device,
            registry,
            "adr065.mtp_draft",
        )?;
        if mtp_substep_profile {
            // Force GPU sync to measure sub-step time accurately.
            let mut enc = device
                .command_encoder()
                .map(crate::inference::native_observation::encoder)
                .context("MTP profile sync 1")?;
            enc.commit_and_wait().ok();
        }
        let t_proj = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let attn_out = self.forward_full_attention(
            &projected, &pos_buf, kv_cache, slot_id, 1, device, registry, cfg,
        )?;
        if mtp_substep_profile {
            let mut enc = device
                .command_encoder()
                .map(crate::inference::native_observation::encoder)
                .context("MTP profile sync 2")?;
            enc.commit_and_wait().ok();
        }
        let t_attn = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = std::time::Instant::now();
        let hidden = self.forward_ffn_residual(&projected, &attn_out, device, registry, cfg)?;
        if mtp_substep_profile {
            let mut enc = device
                .command_encoder()
                .map(crate::inference::native_observation::encoder)
                .context("MTP profile sync 3")?;
            enc.commit_and_wait().ok();
        }
        let t_ffn = t2.elapsed().as_secs_f64() * 1000.0;

        Ok(MtpDraftBody {
            hidden,
            profile_enabled: mtp_substep_profile,
            projection_ms: t_proj,
            attention_ms: t_attn,
            ffn_ms: t_ffn,
        })
    }

    fn report_draft_profile(&self, body: &MtpDraftBody, head_ms: f64) {
        if body.profile_enabled {
            eprintln!(
                "[MTP_SUBSTEP] proj={:.2}ms attn={:.2}ms ffn={:.2}ms head={:.2}ms total={:.2}ms",
                body.projection_ms,
                body.attention_ms,
                body.ffn_ms,
                head_ms,
                body.projection_ms + body.attention_ms + body.ffn_ms + head_ms,
            );
        }
    }

    /// Reconcile a target batch into the MTP attention cache.
    ///
    /// `target_nextn` is the target model's post-output-RMSNorm hidden for
    /// the same tokens as `embed_rows`. MTP consumes that hidden shifted one
    /// row right: row zero is `pending_target_hidden` (or zero for a cold
    /// prompt), and row `i > 0` is target row `i - 1`.
    ///
    /// This is used for initial/full-prompt catch-up and after every verifier
    /// batch. It intentionally stops after the attention K/V write: MTP FFN
    /// and vocabulary logits have no persistent state.
    #[allow(clippy::too_many_arguments)]
    pub fn process_target_batch(
        &self,
        tokens: &[u32],
        pending_target_hidden: Option<&MlxBuffer>,
        target_nextn: &MlxBuffer,
        shared_embed_rows: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        position_ids: &[i32],
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<()> {
        let hidden_size = self.hidden_size as usize;
        let embed_rows = self.embeddings_for_tokens(tokens, shared_embed_rows, device, registry)?;
        ensure!(
            hidden_size > 0,
            "MTP process_target_batch: hidden_size must be > 0"
        );
        ensure!(
            embed_rows.element_count() % hidden_size == 0,
            "MTP process_target_batch: embed elements {} not divisible by hidden_size {hidden_size}",
            embed_rows.element_count()
        );
        let seq_len = embed_rows.element_count() / hidden_size;
        ensure!(
            tokens.len() == seq_len,
            "MTP process_target_batch: tokens {} != seq_len {seq_len}",
            tokens.len()
        );
        ensure!(seq_len > 0, "MTP process_target_batch: empty batch");
        ensure!(
            target_nextn.element_count() == seq_len * hidden_size,
            "MTP process_target_batch: target nextn elements {} != {}",
            target_nextn.element_count(),
            seq_len * hidden_size
        );
        ensure!(
            position_ids.len() == seq_len * 4,
            "MTP process_target_batch: positions {} != 4*seq_len {}",
            position_ids.len(),
            seq_len * 4
        );
        if let Some(pending) = pending_target_hidden {
            ensure!(
                pending.element_count() == hidden_size,
                "MTP process_target_batch: pending hidden elements {} != hidden_size {hidden_size}",
                pending.element_count()
            );
        }

        // mlx-native allocations are zero-initialized. Thus a cold prompt's
        // first shifted row remains exactly zero without a prompt-sized host
        // allocation or download.
        let shifted = device
            .alloc_buffer(
                seq_len * hidden_size * 4,
                DType::F32,
                vec![seq_len, hidden_size],
            )
            .map_err(|e| anyhow!("MTP alloc shifted target nextn: {e}"))?;
        let plan = shifted_nextn_copy_plan(seq_len, hidden_size, pending_target_hidden.is_some())?;
        let mut enc = device
            .command_encoder()
            .map(crate::inference::native_observation::encoder)
            .context("MTP enc shifted target nextn")?;
        if let (Some(pending), Some(copy)) = (pending_target_hidden, plan.pending) {
            dispatch_copy_f32(
                &mut enc,
                registry,
                device.metal_device(),
                pending,
                &shifted,
                copy.src_offset,
                copy.dst_offset,
                copy.count,
            )
            .context("MTP copy pending target nextn")?;
        }
        if let Some(copy) = plan.target_prefix {
            dispatch_copy_f32(
                &mut enc,
                registry,
                device.metal_device(),
                target_nextn,
                &shifted,
                copy.src_offset,
                copy.dst_offset,
                copy.count,
            )
            .context("MTP shift target nextn prefix")?;
        }
        if crate::inference::stage_profile::enabled() {
            enc.commit_and_wait_labeled("adr065.mtp_catchup.shifted_copy")
                .context("profile MTP shifted target copy")?;
        } else {
            enc.commit();
        }
        // Release the submitted copy now: the native observer retains a
        // command for its checked drain only once the encoder drops, so an
        // encoder held past the drain below stays unchecked until some later
        // same-queue wait, or forever when this catch-up is the last GPU work.
        drop(enc);

        let seq_len_u32 = u32::try_from(seq_len).context("MTP batch seq_len exceeds u32")?;
        let pos_buf = upload_i32(position_ids, device).context("MTP upload batch positions")?;
        let projected = self.project_embedding_and_hidden(
            &embed_rows,
            &shifted,
            seq_len_u32,
            device,
            registry,
            "adr065.mtp_catchup",
        )?;
        self.append_attention_kv(
            &projected,
            &pos_buf,
            kv_cache,
            slot_id,
            seq_len_u32,
            device,
            registry,
            cfg,
        )?;

        // The KV-only projection/write chain submits without a wait. Drain
        // before its temporary buffers drop; the MTP cursor and valid prefix
        // are transactional on return.
        let mut drain = device
            .command_encoder()
            .map(crate::inference::native_observation::encoder)
            .context("MTP target-batch cache drain")?;
        drain
            .commit_and_wait_labeled("mtp.process_target_batch")
            .context("MTP target-batch cache drain")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn append_attention_kv(
        &self,
        x: &MlxBuffer,
        positions: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        seq_len: u32,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<()> {
        let h = self.hidden_size;
        let kv_total = cfg.num_key_value_heads * cfg.head_dim;
        let attn = &self.attn;
        let (k_rope, v_flat) = {
            let mut enc = device
                .command_encoder()
                .map(crate::inference::native_observation::encoder)
                .context("MTP enc KV-only projections")?;
            let x_norm = rms_norm_with_weight(
                &mut enc,
                registry,
                device,
                x,
                &attn.attn_norm,
                seq_len,
                h,
                cfg.rms_norm_eps,
            )?;
            enc.memory_barrier();
            let k_flat = apply_linear_projection_f32_with_ggml_type(
                &mut enc,
                registry,
                device,
                &x_norm,
                &attn.wk,
                attn.wk_ggml_type,
                seq_len,
                h,
                kv_total,
            )?;
            let v_flat = apply_linear_projection_f32_with_ggml_type(
                &mut enc,
                registry,
                device,
                &x_norm,
                &attn.wv,
                attn.wv_ggml_type,
                seq_len,
                h,
                kv_total,
            )?;
            enc.memory_barrier();
            let k_normed = apply_q_or_k_per_head_rms_norm(
                &mut enc,
                registry,
                device,
                &k_flat,
                &attn.attn_k_norm,
                seq_len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?;
            enc.memory_barrier();
            let k_rope = apply_imrope(
                &mut enc,
                registry,
                device,
                &k_normed,
                positions,
                seq_len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rotary_dim,
                cfg.rope_theta as f32,
                cfg.mrope_section,
            )?;
            if crate::inference::stage_profile::enabled() {
                enc.commit_and_wait_labeled("adr065.mtp_catchup.kv_projections")
                    .context("profile MTP catch-up KV projections")?;
            } else {
                enc.commit_labeled("mtp.kv_only_projections");
            }
            (k_rope, v_flat)
        };

        let max_seq_len = kv_cache.max_seq_len;
        let slot = kv_cache
            .mtp_slot
            .as_mut()
            .context("MTP KV-only append requires HybridKvCache.mtp_slot")?;
        append_kv_to_cache_without_attention(
            device,
            registry,
            &k_rope,
            &v_flat,
            slot,
            seq_len,
            cfg.num_key_value_heads,
            cfg.head_dim,
            max_seq_len,
            slot_id,
        )
        .context("MTP target-batch KV-only append")
    }

    fn project_embedding_and_hidden(
        &self,
        embed_t: &MlxBuffer,
        prev_hidden: &MlxBuffer,
        seq_len: u32,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        profile_label: &'static str,
    ) -> Result<MlxBuffer> {
        let h = self.hidden_size;
        let mut enc = device
            .command_encoder()
            .map(crate::inference::native_observation::encoder)
            .context("MTP enc eh_proj")?;
        let embed_norm = rms_norm_with_weight(
            &mut enc,
            registry,
            device,
            embed_t,
            &self.enorm,
            seq_len,
            h,
            1e-6,
        )?;
        let hidden_norm = rms_norm_with_weight(
            &mut enc,
            registry,
            device,
            prev_hidden,
            &self.hnorm,
            seq_len,
            h,
            1e-6,
        )?;
        enc.memory_barrier();
        let concat = device
            .alloc_buffer(
                (seq_len * 2 * h) as usize * 4,
                DType::F32,
                vec![seq_len as usize, (2 * h) as usize],
            )
            .map_err(|e| anyhow!("MTP alloc eh_proj concat: {e}"))?;
        mlx_native::ops::feature_concat::register(registry);
        mlx_native::ops::feature_concat::dispatch_feature_concat_f32(
            &mut enc,
            registry,
            device.metal_device(),
            &embed_norm,
            &concat,
            seq_len,
            h,
            0,
            2 * h,
        )
        .context("MTP eh_proj concat embedding")?;
        mlx_native::ops::feature_concat::dispatch_feature_concat_f32(
            &mut enc,
            registry,
            device.metal_device(),
            &hidden_norm,
            &concat,
            seq_len,
            h,
            h,
            2 * h,
        )
        .context("MTP eh_proj concat hidden")?;
        enc.memory_barrier();
        let out = apply_linear_projection_f32_with_ggml_type(
            &mut enc,
            registry,
            device,
            &concat,
            &self.eh_proj,
            self.eh_proj_ggml_type,
            seq_len,
            2 * h,
            h,
        )?;
        if crate::inference::stage_profile::enabled() {
            enc.commit_and_wait_labeled(profile_label)
                .context("profile MTP embedding and hidden projection")?;
        } else {
            enc.commit();
        }
        Ok(out)
    }

    fn forward_full_attention(
        &self,
        x: &MlxBuffer,
        positions: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        seq_len: u32,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<MlxBuffer> {
        let h = self.hidden_size;
        let q_total = cfg.num_attention_heads * cfg.head_dim;
        let kv_total = cfg.num_key_value_heads * cfg.head_dim;
        let attn = &self.attn;

        let (q_rope, k_rope, v_flat, gate_flat) = {
            let mut enc = device
                .command_encoder()
                .map(crate::inference::native_observation::encoder)
                .context("MTP enc attn qkv")?;
            let x_norm = rms_norm_with_weight(
                &mut enc,
                registry,
                device,
                x,
                &attn.attn_norm,
                seq_len,
                h,
                cfg.rms_norm_eps,
            )?;
            enc.memory_barrier();
            let (q_flat, gate_flat) = match &attn.q_gate {
                MtpQGateWeightsGpu::Ungated { wq, wq_ggml_type } => (
                    apply_linear_projection_f32_with_ggml_type(
                        &mut enc,
                        registry,
                        device,
                        &x_norm,
                        wq,
                        *wq_ggml_type,
                        seq_len,
                        h,
                        q_total,
                    )?,
                    None,
                ),
                MtpQGateWeightsGpu::Gated(q_gate) => {
                    let (q, gate) = apply_q_gate_projection_f32(
                        &mut enc,
                        registry,
                        device,
                        &x_norm,
                        q_gate,
                        seq_len,
                        cfg.num_attention_heads,
                        cfg.head_dim,
                        h,
                    )?;
                    (q, Some(gate))
                }
            };
            let k_flat = apply_linear_projection_f32_with_ggml_type(
                &mut enc,
                registry,
                device,
                &x_norm,
                &attn.wk,
                attn.wk_ggml_type,
                seq_len,
                h,
                kv_total,
            )?;
            let v_flat = apply_linear_projection_f32_with_ggml_type(
                &mut enc,
                registry,
                device,
                &x_norm,
                &attn.wv,
                attn.wv_ggml_type,
                seq_len,
                h,
                kv_total,
            )?;
            enc.memory_barrier();
            let q_normed = apply_q_or_k_per_head_rms_norm(
                &mut enc,
                registry,
                device,
                &q_flat,
                &attn.attn_q_norm,
                seq_len,
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
                seq_len,
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
                seq_len,
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
                seq_len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rotary_dim,
                cfg.rope_theta as f32,
                cfg.mrope_section,
            )?;
            if crate::inference::stage_profile::enabled() {
                enc.commit_and_wait_labeled("adr065.mtp_draft.attention_projections")
                    .context("profile MTP draft attention projections")?;
            } else {
                enc.commit();
            }
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
            seq_len,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            kv_cache.max_seq_len,
            None,
            slot_id,
        )
        .context("MTP SDPA")?;

        let mut enc = device
            .command_encoder()
            .map(crate::inference::native_observation::encoder)
            .context("MTP enc attn output")?;
        let gated_or_attn = if let Some(gate) = gate_flat.as_ref() {
            let gated = apply_sigmoid_gate_multiply(
                &mut enc,
                registry,
                device,
                &attn_out,
                gate,
                seq_len * q_total,
            )?;
            // RAW: the output projection reads `gated`. The encoder
            // dispatches concurrently, so without this barrier the
            // projection can read partially written rows and the MTP draft
            // changes from run to run (the target path's ADR-015 iter21 fix).
            enc.memory_barrier();
            gated
        } else {
            attn_out
        };
        let out = apply_linear_projection_f32_with_ggml_type(
            &mut enc,
            registry,
            device,
            &gated_or_attn,
            &attn.wo,
            attn.wo_ggml_type,
            seq_len,
            q_total,
            h,
        )?;
        if crate::inference::stage_profile::enabled() {
            enc.commit_and_wait_labeled("adr065.mtp_draft.attention_output")
                .context("profile MTP draft attention output")?;
        } else {
            enc.commit();
        }
        Ok(out)
    }

    /// Post-attention residual norm and FFN over `[rows, H]` (one row on the
    /// single-draft path). MoE runs each row through the one-row expert path.
    fn forward_ffn_residual(
        &self,
        residual: &MlxBuffer,
        attn_out: &MlxBuffer,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Qwen35Config,
    ) -> Result<MlxBuffer> {
        let h = self.hidden_size;
        let hs = h as usize;
        ensure!(
            residual.element_count().is_multiple_of(hs) && residual.element_count() > 0,
            "MTP FFN residual has {} elements, not a positive multiple of {h}",
            residual.element_count()
        );
        let rows = residual.element_count() / hs;
        let ffn_input = device
            .alloc_buffer(rows * hs * 4, DType::F32, vec![rows, hs])
            .map_err(|e| anyhow!("MTP alloc ffn_input: {e}"))?;
        let ffn_residual = device
            .alloc_buffer(rows * hs * 4, DType::F32, vec![rows, hs])
            .map_err(|e| anyhow!("MTP alloc ffn_residual: {e}"))?;
        let mut enc = device
            .command_encoder()
            .map(crate::inference::native_observation::encoder)
            .context("MTP enc residual norm")?;
        dispatch_fused_residual_norm_f32(
            &mut enc,
            registry,
            device.metal_device(),
            residual,
            attn_out,
            &self.attn.post_attn_norm,
            &ffn_input,
            Some(&ffn_residual),
            rows as u32,
            h,
            cfg.rms_norm_eps,
        )
        .context("MTP fused residual norm")?;
        if crate::inference::stage_profile::enabled() {
            enc.commit_and_wait_labeled("adr065.mtp_draft.residual_norm")
                .context("profile MTP draft residual norm")?;
        } else {
            enc.commit();
        }
        // Release before the FFN, whose teacher-scope route may wait.
        drop(enc);

        match &self.ffn {
            MtpFfnWeightsGpu::Dense {
                weights,
                intermediate_size,
            } => build_dense_ffn_layer_gpu(
                device,
                registry,
                &ffn_input,
                weights,
                DenseFfnShape {
                    hidden_size: h,
                    intermediate_size: *intermediate_size,
                },
                Some(&ffn_residual),
            )
            .context("MTP exact F32 dense FFN"),
            MtpFfnWeightsGpu::DenseQ { weights } => build_dense_ffn_layer_gpu_q(
                device,
                registry,
                &ffn_input,
                weights,
                Some(&ffn_residual),
            )
            .context("MTP native dense FFN"),
            MtpFfnWeightsGpu::Moe { weights, shape } => {
                // ADR-034 post-codex audit (2026-05-21): route through the
                // external-encoder variant with the REAL MTP layer index
                // (`self.layer_index`, typically num_hidden_layers — e.g. 40
                // for Qwen 3.5 35B-A3B) so the imatrix intercept tag emitted
                // by `build_moe_ffn_layer_gpu_q_into` reflects the actual
                // MTP block name (`blk.{layer_index}.ffn_*_exps.weight`).
                // The legacy wrapper `build_moe_ffn_layer_gpu_q` hardcodes
                // `layer_idx=0` (gpu_ffn.rs:2263); using it from production
                // would silently mis-tag MTP expert records.
                let mut enc = device
                    .command_encoder()
                    .map(crate::inference::native_observation::encoder)
                    .context("MTP enc moe_ffn_q")?;
                let out = if rows == 1 {
                    build_moe_ffn_layer_gpu_q_into(
                        &mut enc,
                        device,
                        registry,
                        &ffn_input,
                        weights,
                        *shape,
                        Some(&ffn_residual),
                        self.layer_index as usize,
                    )
                    .context("MTP MoE FFN")?
                } else {
                    // Routed experts differ per row: one row at a time
                    // through the one-row path, then gathered.
                    let mut per_row = Vec::with_capacity(rows);
                    for row in 0..rows {
                        if row > 0 {
                            enc.memory_barrier();
                        }
                        per_row.push(
                            build_moe_ffn_layer_gpu_q_into(
                                &mut enc,
                                device,
                                registry,
                                &row_view(&ffn_input, row, vec![1, hs])?,
                                weights,
                                *shape,
                                Some(&row_view(&ffn_residual, row, vec![1, hs])?),
                                self.layer_index as usize,
                            )
                            .with_context(|| format!("MTP MoE FFN row {row}"))?,
                        );
                    }
                    enc.memory_barrier();
                    let packed = device
                        .alloc_buffer(rows * hs * 4, DType::F32, vec![rows, hs])
                        .map_err(|e| anyhow!("MTP alloc packed MoE output: {e}"))?;
                    for (row, output) in per_row.iter().enumerate() {
                        dispatch_copy_f32(
                            &mut enc,
                            registry,
                            device.metal_device(),
                            output,
                            &packed,
                            0,
                            row * hs,
                            hs,
                        )
                        .with_context(|| format!("MTP gather MoE row {row}"))?;
                    }
                    packed
                };
                // Match the legacy wrapper's commit policy: seq=1 (the only
                // MTP draft shape) uses non-blocking commit so the next
                // command buffer pipelines across the boundary on Metal.
                if crate::inference::stage_profile::enabled() {
                    enc.commit_and_wait_labeled("adr065.mtp_draft.moe")
                        .context("profile MTP draft MoE")?;
                } else {
                    enc.commit();
                }
                Ok(out)
            }
        }
    }

    fn forward_shared_head(
        &self,
        hidden: &MlxBuffer,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        eps: f32,
    ) -> Result<(MlxBuffer, MlxBuffer)> {
        // ADR-028 iter-155: consolidated single-CB shared-head — merges
        // the prior 2-buffer chain (head_norm + lm_head) into one
        // command buffer with a memory_barrier between RAW dependents.
        // Saves ~1ms per draft step on Apple Metal at decode shape.
        let h = self.hidden_size;
        let mut enc = device
            .command_encoder()
            .map(crate::inference::native_observation::encoder)
            .context("MTP enc shared head")?;
        let normed = rms_norm_with_weight(
            &mut enc,
            registry,
            device,
            hidden,
            &self.shared_head_norm,
            1,
            h,
            eps,
        )?;
        // RAW: lm_head reads `normed` produced by rms_norm above. Apple
        // Metal compute encoders run threadgroups in parallel by default;
        // memory_barrier required so the projection sees finalized norm.
        enc.memory_barrier();
        let logits = apply_linear_projection_f32_with_ggml_type(
            &mut enc,
            registry,
            device,
            &normed,
            &self.shared_head_head,
            self.shared_head_head_ggml_type,
            1,
            h,
            self.vocab_size,
        )
        .context("MTP shared head projection")?;
        if crate::inference::stage_profile::enabled() {
            enc.commit_and_wait_labeled("adr065.mtp_draft.shared_head")
                .context("profile MTP commit logits")?;
        } else {
            enc.commit_and_wait().context("MTP commit logits")?;
        }
        Ok((logits, normed))
    }

    /// Shared-head norm, vocabulary projection and one GPU argmax per row
    /// over `[rows, H]`, in one command buffer with a single wait. Returns
    /// each row's draft and the normalized rows.
    fn forward_shared_head_greedy(
        &self,
        hidden: &MlxBuffer,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        eps: f32,
    ) -> Result<(Vec<u32>, MlxBuffer)> {
        let h = self.hidden_size;
        let hs = h as usize;
        ensure!(
            hidden.element_count().is_multiple_of(hs) && hidden.element_count() > 0,
            "MTP shared head input has {} elements, not a positive multiple of {h}",
            hidden.element_count()
        );
        let rows = hidden.element_count() / hs;
        let vocab = self.vocab_size as usize;
        let mut enc = device
            .command_encoder()
            .map(crate::inference::native_observation::encoder)
            .context("MTP enc fused shared head argmax")?;
        let normed = rms_norm_with_weight(
            &mut enc,
            registry,
            device,
            hidden,
            &self.shared_head_norm,
            rows as u32,
            h,
            eps,
        )?;
        enc.memory_barrier();
        let logits = apply_linear_projection_f32_with_ggml_type(
            &mut enc,
            registry,
            device,
            &normed,
            &self.shared_head_head,
            self.shared_head_head_ggml_type,
            rows as u32,
            h,
            self.vocab_size,
        )
        .context("MTP fused greedy shared-head projection")?;
        enc.memory_barrier();

        let mut params = device
            .alloc_buffer(4, DType::U32, vec![1])
            .map_err(|error| anyhow!("MTP alloc greedy argmax params: {error}"))?;
        params
            .as_mut_slice::<u32>()
            .map_err(|error| anyhow!("MTP greedy argmax params slice: {error}"))?[0] =
            self.vocab_size;
        let mut indices = Vec::with_capacity(rows);
        for row in 0..rows {
            let out_index = device
                .alloc_buffer(4, DType::U32, vec![1])
                .map_err(|error| anyhow!("MTP alloc greedy argmax index: {error}"))?;
            let out_value = device
                .alloc_buffer(4, DType::F32, vec![1])
                .map_err(|error| anyhow!("MTP alloc greedy argmax value: {error}"))?;
            let row_logits = if rows == 1 {
                logits.clone()
            } else {
                row_view(&logits, row, vec![1, vocab])?
            };
            dispatch_argmax_f32(
                &mut enc,
                registry,
                device.metal_device(),
                &row_logits,
                &out_index,
                &out_value,
                &params,
                self.vocab_size,
            )
            .with_context(|| format!("MTP fused greedy argmax row {row}"))?;
            indices.push((out_index, out_value));
        }
        enc.commit_and_wait_labeled("mtp.shared_head_argmax")
            .context("MTP commit fused shared-head argmax")?;
        let tokens = indices
            .iter()
            .map(|(index, _)| {
                index
                    .as_slice::<u32>()
                    .map(|slice| slice[0])
                    .map_err(|error| anyhow!("MTP greedy argmax index slice: {error}"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((tokens, normed))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HiddenCopy {
    src_offset: usize,
    dst_offset: usize,
    count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShiftedNextnCopyPlan {
    pending: Option<HiddenCopy>,
    target_prefix: Option<HiddenCopy>,
}

fn shifted_nextn_copy_plan(
    seq_len: usize,
    hidden_size: usize,
    has_pending: bool,
) -> Result<ShiftedNextnCopyPlan> {
    ensure!(seq_len > 0, "shifted nextn plan requires a non-empty batch");
    ensure!(
        hidden_size > 0,
        "shifted nextn plan requires hidden_size > 0"
    );
    let pending = has_pending.then_some(HiddenCopy {
        src_offset: 0,
        dst_offset: 0,
        count: hidden_size,
    });
    let target_prefix = (seq_len > 1).then_some(HiddenCopy {
        src_offset: 0,
        dst_offset: hidden_size,
        count: (seq_len - 1)
            .checked_mul(hidden_size)
            .context("shifted nextn target-prefix size overflow")?,
    });
    Ok(ShiftedNextnCopyPlan {
        pending,
        target_prefix,
    })
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
