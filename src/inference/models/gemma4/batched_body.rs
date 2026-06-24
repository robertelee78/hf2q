//! ADR-040 Phase F M2.2 / S2+S3 — the `[N,hidden]` batched decode BODY.
//!
//! S1 batched the lm_head (post-body). S2/S3 batch the BODY itself — the layer
//! loop that dominates decode time (MoE 1.76 ms/token + dense projections). The
//! MoE sits mid-layer after per-slot attention, so batching it requires the N
//! slots' MoE inputs gathered: i.e. running the whole layer in `[N,hidden]`.
//!
//! `forward_decode_body_batched` replaces `decode_batch_gemma4` pass-1's N
//! sequential per-slot bodies with ONE batched pass, leaving each slot's final
//! hidden in `[N,hidden]`; the existing (proven) batched head + finalize then
//! complete the tick. Premise PROVEN at the kernel level before this restructure
//! (mantra): dense projections per-row bit-identical (H-S1-rowparity, mlx
//! 272b2a3), MoE `_id` per-token bit-identical across n_tokens (H-S2-tokenparity,
//! mlx 32b5045), rms_norm per-row identical (implicit via slot_aware_n4),
//! attention per-slot (N independent `flash_attn_vec_tq_hb` dispatches). So a
//! correct `[N,hidden]` body MUST be bit-identical to N serial bodies — the
//! `slot_aware_n1`/`slot_aware_n4` gates confirm it.
//!
//! This module is built incrementally with the gate held at each step:
//! 1. `BatchedDecodeBuffers` — the `[N,...]` activation scratch (this commit).
//! 2. `encode_one_layer_batched` — production hybrid-TQ `[N,hidden]` layer.
//! 3. `forward_decode_body_batched` — embed + layer loop, wired into pass-1.

use anyhow::Result;
use mlx_native::graph::GraphSession;
use mlx_native::{DType, GraphExecutor, KernelRegistry, MlxBuffer, MlxDevice};

use crate::quantize::imatrix::ImatrixHint;
use crate::serve::config::LayerType;
use crate::serve::forward_mlx_shared::{
    dispatch_qmatmul, dispatch_rms_norm_unit_perhead, RmsNormPerHeadArgs,
};
use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};

use super::kv_cache::MultiSeqHybridKvBuffers;
use super::model::{MlxActivationBuffers, MlxModelWeights};
use crate::serve::gpu::GpuContext;
use crate::serve::multi_seq_kv::SlotId;

/// `[N,...]` mirror of the production decode activation scratch. Each buffer is
/// sized `N × (the scalar buffer's element count)` — read straight from the
/// model's existing `MlxActivationBuffers` so the per-token/per-head dimensions
/// are exactly the proven scalar sizes (largest-layer-sized), never re-derived.
/// Row-major: row `i` (slot `i`) of buffer `b` occupies `b[i*stride..(i+1)*stride]`
/// where `stride` is the scalar buffer's element count.
pub struct BatchedDecodeBuffers {
    /// Batch width (number of slots this scratch is sized for).
    pub n: usize,
    /// `[N, hidden]` residual-stream hidden state (the body's input + output).
    pub hidden: MlxBuffer,
    /// `[N, hidden]` RMS-norm output (pre-attn / pre-FF / final norms reuse it).
    pub norm_out: MlxBuffer,
    /// `[N, num_heads*head_dim]` Q projection.
    pub attn_q: MlxBuffer,
    /// `[N, num_kv_heads*head_dim]` K projection.
    pub attn_k: MlxBuffer,
    /// `[N, num_kv_heads*head_dim]` V projection.
    pub attn_v: MlxBuffer,
    /// `[N, num_heads*head_dim]` Q after per-head norm + RoPE.
    pub attn_q_normed: MlxBuffer,
    /// `[N, num_kv_heads*head_dim]` K after per-head norm + RoPE.
    pub attn_k_normed: MlxBuffer,
    /// `[N, num_heads*head_dim]` SDPA output (one row per slot's attention).
    pub sdpa_out: MlxBuffer,
    /// `[N, hidden]` O-projection output.
    pub attn_out: MlxBuffer,
    /// `[N, intermediate]` dense-MLP gate.
    pub mlp_gate: MlxBuffer,
    /// `[N, intermediate]` dense-MLP up.
    pub mlp_up: MlxBuffer,
    /// `[N, max(intermediate, moe_intermediate)]` fused SwiGLU scratch.
    pub mlp_fused: MlxBuffer,
    /// `[N, hidden]` dense-MLP down output.
    pub mlp_down: MlxBuffer,
    /// `[N, hidden]` residual scratch.
    pub residual: MlxBuffer,
    /// `[N, hidden]` MoE router-input norm.
    pub moe_norm_out: MlxBuffer,
    /// `[N, hidden]` router norm (concurrent with pre-FF norm).
    pub router_norm_out: MlxBuffer,
    /// `[N, num_experts]` router logits.
    pub moe_router_logits: MlxBuffer,
    /// `[N, top_k]` selected expert ids (U32).
    pub moe_expert_ids: MlxBuffer,
    /// `[N, top_k]` pre-scaled routing weights.
    pub moe_routing_weights_gpu: MlxBuffer,
    /// `[N, top_k, 2*moe_intermediate]` gate_up `_id` output.
    pub moe_gate_up_id_out: MlxBuffer,
    /// `[N, top_k, moe_intermediate]` SwiGLU `_id` output.
    pub moe_swiglu_id_out: MlxBuffer,
    /// `[N, top_k, hidden]` down `_id` output.
    pub moe_down_id_out: MlxBuffer,
}

/// Element count of an F32/U32 buffer (4 bytes/element).
fn elems(b: &MlxBuffer) -> usize {
    b.byte_len() / 4
}

impl BatchedDecodeBuffers {
    /// Allocate `[N,...]` scratch sized `N ×` each scalar buffer. `acts` is the
    /// model's live `MlxActivationBuffers` (the proven scalar sizes).
    pub fn new(device: &MlxDevice, acts: &MlxActivationBuffers, n: usize) -> Result<Self> {
        let f32n = |scalar: &MlxBuffer, name: &str| -> Result<MlxBuffer> {
            let count = elems(scalar) * n;
            device
                .alloc_buffer(count * 4, DType::F32, vec![count])
                .map_err(|e| anyhow::anyhow!("BatchedDecodeBuffers alloc {name} ({count} f32): {e}"))
        };
        let u32n = |scalar: &MlxBuffer, name: &str| -> Result<MlxBuffer> {
            let count = elems(scalar) * n;
            device
                .alloc_buffer(count * 4, DType::U32, vec![count])
                .map_err(|e| anyhow::anyhow!("BatchedDecodeBuffers alloc {name} ({count} u32): {e}"))
        };
        Ok(Self {
            n,
            hidden: f32n(&acts.hidden, "hidden")?,
            norm_out: f32n(&acts.norm_out, "norm_out")?,
            attn_q: f32n(&acts.attn_q, "attn_q")?,
            attn_k: f32n(&acts.attn_k, "attn_k")?,
            attn_v: f32n(&acts.attn_v, "attn_v")?,
            attn_q_normed: f32n(&acts.attn_q_normed, "attn_q_normed")?,
            attn_k_normed: f32n(&acts.attn_k_normed, "attn_k_normed")?,
            sdpa_out: f32n(&acts.sdpa_out, "sdpa_out")?,
            attn_out: f32n(&acts.attn_out, "attn_out")?,
            mlp_gate: f32n(&acts.mlp_gate, "mlp_gate")?,
            mlp_up: f32n(&acts.mlp_up, "mlp_up")?,
            mlp_fused: f32n(&acts.mlp_fused, "mlp_fused")?,
            mlp_down: f32n(&acts.mlp_down, "mlp_down")?,
            residual: f32n(&acts.residual, "residual")?,
            moe_norm_out: f32n(&acts.moe_norm_out, "moe_norm_out")?,
            router_norm_out: f32n(&acts.router_norm_out, "router_norm_out")?,
            moe_router_logits: f32n(&acts.moe_router_logits, "moe_router_logits")?,
            moe_expert_ids: u32n(&acts.moe_expert_ids, "moe_expert_ids")?,
            moe_routing_weights_gpu: f32n(&acts.moe_routing_weights_gpu, "moe_routing_weights_gpu")?,
            moe_gate_up_id_out: f32n(&acts.moe_gate_up_id_out, "moe_gate_up_id_out")?,
            moe_swiglu_id_out: f32n(&acts.moe_swiglu_id_out, "moe_swiglu_id_out")?,
            moe_down_id_out: f32n(&acts.moe_down_id_out, "moe_down_id_out")?,
        })
    }

    /// Per-slot stride (element count of one row) for buffer family `hidden`.
    #[inline]
    pub fn hidden_stride(&self) -> usize {
        elems(&self.hidden) / self.n
    }
}

/// Byte offset of slot `i`'s row in an `[N, stride]` buffer (f32/u32, 4 bytes).
#[inline]
fn row_off(stride: usize, i: usize) -> u64 {
    (i * stride * 4) as u64
}

impl MlxModelWeights {
    /// ADR-040 S2/S3 — one `[N,hidden]` decode layer (production hybrid-TQ path).
    ///
    /// Position-INDEPENDENT ops (input-norm, QKV, O-proj, post-attn norm, dense
    /// MLP, MoE) run BATCHED on the full `[N,...]` buffers (rows=N / m=N /
    /// n_tokens=N — proven bit-identical per H-S1-rowparity + H-S2-tokenparity).
    /// Position-DEPENDENT ops (Q/K norm+RoPE, V-norm, hybrid KV-encode,
    /// `flash_attn_vec_hybrid`) loop per-slot over `slice_view` row-views, running
    /// the EXACT scalar ops — bit-identical by reuse. At N=1 each row-view is the
    /// whole buffer ⇒ structurally identical to the scalar body.
    ///
    /// WORK IN PROGRESS — built op-by-op against the scalar `encode_one_layer`,
    /// gated by `slot_aware_n1` (N=1 bit-identical) then `slot_aware_n4`. Not yet
    /// wired into `decode_batch_gemma4` until the full layer + body verify.
    #[allow(dead_code)]
    pub(crate) fn encode_one_layer_batched(
        &self,
        layer_idx: usize,
        bufs: &BatchedDecodeBuffers,
        n: usize,
        positions_buf: &MlxBuffer,
        slot_ids: &[SlotId],
        seq_positions: &[usize],
        multi_seq_kv_hybrid: &mut [MultiSeqHybridKvBuffers],
        tq_scale_factor_d512: f32,
        tq_codebook_bits: u32,
        session: &mut GraphSession<'_>,
        exec: &GraphExecutor,
        reg: &mut KernelRegistry,
    ) -> Result<()> {
        let dev = exec.device();
        let metal_dev = dev.metal_device();
        let hs = self.hidden_size;
        let nu = n as u32;
        let hd = self.layers[layer_idx].head_dim;
        let nkv = self.layers[layer_idx].num_kv_heads;
        let nh = self.num_attention_heads;
        let is_sliding = self.layers[layer_idx].layer_type == LayerType::Sliding;
        let eps = self.rms_norm_eps;
        let q_stride = elems(&bufs.attn_q) / n;
        let k_stride = elems(&bufs.attn_k) / n;
        let v_stride = elems(&bufs.attn_v) / n;

        // -- Pre-attention RMS norm (BATCHED rows=N): hidden -> norm_out --
        // norm_params is [eps, hs], per-element and identical for every row.
        session.barrier_between(
            &[&bufs.hidden, &self.layers[layer_idx].norms.input_layernorm],
            &[&bufs.norm_out],
        );
        session
            .rms_norm(
                reg,
                metal_dev,
                &bufs.hidden,
                &self.layers[layer_idx].norms.input_layernorm,
                &bufs.norm_out,
                &self.activations.norm_params,
                nu,
                hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched pre-attn norm L{layer_idx}: {e}"))?;

        // -- QKV projections (BATCHED m=N): all read norm_out, write disjoint --
        session.barrier_between(
            &[&bufs.norm_out],
            &[&bufs.attn_q, &bufs.attn_k, &bufs.attn_v],
        );
        dispatch_qmatmul(
            session, reg, dev, &bufs.norm_out,
            &self.layers[layer_idx].attn.q_proj, &bufs.attn_q, nu,
            ImatrixHint::Layered { tag: "attn_q", layer: layer_idx },
        )?;
        dispatch_qmatmul(
            session, reg, dev, &bufs.norm_out,
            &self.layers[layer_idx].attn.k_proj, &bufs.attn_k, nu,
            ImatrixHint::Layered { tag: "attn_k", layer: layer_idx },
        )?;
        let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
        if !v_is_k {
            dispatch_qmatmul(
                session, reg, dev, &bufs.norm_out,
                self.layers[layer_idx].attn.v_proj.as_ref().unwrap(), &bufs.attn_v, nu,
                ImatrixHint::Layered { tag: "attn_v", layer: layer_idx },
            )?;
        }

        // -- Per-head RMS norm + RoPE on Q and K (PER-SLOT row-views) --
        // Position-dependent: each slot's row uses its own position. Mirrors the
        // scalar fused_head_norm_rope (gpu_full_attn.rs:150-170) per slot.
        let half_rope = (hd / 2) as u32;
        let ff_gpu = if is_sliding {
            None
        } else {
            Some(&self.activations.rope_freq_factors_gpu)
        };
        let theta = if is_sliding {
            self.rope_theta_sliding
        } else {
            self.rope_theta_global
        };
        session.barrier_between(
            &[&bufs.attn_q, &bufs.attn_k],
            &[&bufs.attn_q_normed, &bufs.attn_k_normed],
        );
        for i in 0..n {
            let pos_i = positions_buf.slice_view((i * 4) as u64, 1);
            let q_in = bufs.attn_q.slice_view(row_off(q_stride, i), q_stride);
            let q_out = bufs.attn_q_normed.slice_view(row_off(q_stride, i), q_stride);
            mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                session.encoder_mut(), reg, metal_dev,
                &q_in, &q_out,
                Some(&self.layers[layer_idx].attn.q_norm_weight),
                &pos_i, ff_gpu,
                nh as u32, hd as u32, half_rope, eps, theta,
            )
            .map_err(|e| anyhow::anyhow!("batched Q norm+RoPE L{layer_idx} slot{i}: {e}"))?;
            let k_in = bufs.attn_k.slice_view(row_off(k_stride, i), k_stride);
            let k_out = bufs.attn_k_normed.slice_view(row_off(k_stride, i), k_stride);
            mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                session.encoder_mut(), reg, metal_dev,
                &k_in, &k_out,
                Some(&self.layers[layer_idx].attn.k_norm_weight),
                &pos_i, ff_gpu,
                nkv as u32, hd as u32, half_rope, eps, theta,
            )
            .map_err(|e| anyhow::anyhow!("batched K norm+RoPE L{layer_idx} slot{i}: {e}"))?;
        }

        // -- V norm (PER-SLOT row-views) — gemma4 v_is_k case writes attn_v --
        // Per-head RMS norm with the per-head (sliding/global) norm params.
        let hd_norm_params = if is_sliding {
            &self.activations.norm_params_sliding_hd
        } else {
            &self.activations.norm_params_global_hd
        };
        if v_is_k {
            session.barrier_between(&[&bufs.attn_k], &[&bufs.attn_v]);
            for i in 0..n {
                let vk_in = bufs.attn_k.slice_view(row_off(k_stride, i), k_stride);
                let v_out = bufs.attn_v.slice_view(row_off(v_stride, i), v_stride);
                dispatch_rms_norm_unit_perhead(
                    session.encoder_mut(), reg, metal_dev,
                    &RmsNormPerHeadArgs {
                        input: &vk_in,
                        output: &v_out,
                        params_buf: hd_norm_params,
                        rows: nkv as u32,
                        dim: hd as u32,
                    },
                )?;
            }
        }

        // -- ATTENTION (PER-SLOT): hybrid KV-encode (F16-K copy + FWHT-V quant)
        // -> flash_attn_vec_hybrid -> fwht_sign_undo, each against this slot's
        // multi_seq_kv_hybrid[L] region (slice_view at slot byte offset). Mirrors
        // the scalar default hybrid path (gpu_full_attn.rs:416-494, 1180-1221,
        // FWHT-undo). The encode↔SDPA↔undo FWHT coherence is version-churned in
        // the scalar source; this build applies the undo, and the N=1 gate
        // (slot_aware_n1, bit-identical to serial) settles it — if N=1 diverges,
        // toggle the undo / params per the dumped divergence (RUNTIME PARITY,
        // not read-replication). --
        let q_norm_stride = elems(&bufs.attn_q_normed) / n;
        let k_norm_stride = elems(&bufs.attn_k_normed) / n;
        let sdpa_stride = elems(&bufs.sdpa_out) / n;
        for i in 0..n {
            let slot = slot_ids[i].0 as u64;
            let seq_pos_i = seq_positions[i];
            let buf = &multi_seq_kv_hybrid[layer_idx];
            let cap = buf.capacity;
            let is_ring = buf.is_sliding;
            let cache_pos: u32 = if is_ring {
                (seq_pos_i % cap) as u32
            } else {
                seq_pos_i as u32
            };
            // Per-slot KV views (offset math mirrors forward_prefill.rs:4636-4735).
            let k_elems = nkv * cap * hd;
            let v_dtype_size = buf.v_packed.dtype().size_of() as u64;
            let k_view = buf
                .k
                .slice_view(slot * (k_elems as u64) * 2, k_elems)
                .with_shape(vec![nkv, cap, hd])
                .map_err(|e| anyhow::anyhow!("batched K slot-view L{layer_idx} s{i}: {e}"))?;
            let v_view = buf
                .v_packed
                .slice_view(slot * (k_elems as u64) * v_dtype_size, k_elems)
                .with_shape(vec![nkv, cap, hd])
                .map_err(|e| anyhow::anyhow!("batched V slot-view L{layer_idx} s{i}: {e}"))?;
            let norms_per_pos = buf.norms_per_pos;
            let v_norms_view = if buf.v_norms.byte_len() == 4 {
                buf.v_norms.slice_view(0, 1).with_shape(vec![1])
                    .map_err(|e| anyhow::anyhow!("batched Vnorms dummy L{layer_idx} s{i}: {e}"))?
            } else {
                let ne = nkv * cap * norms_per_pos;
                let shp = if norms_per_pos == 1 { vec![nkv, cap] } else { vec![nkv, cap, norms_per_pos] };
                buf.v_norms.slice_view(slot * (ne as u64) * 4, ne).with_shape(shp)
                    .map_err(|e| anyhow::anyhow!("batched Vnorms slot-view L{layer_idx} s{i}: {e}"))?
            };
            // Row-views of the batched Q/K/V activations for this slot.
            let q_i = bufs.attn_q_normed.slice_view(row_off(q_norm_stride, i), q_norm_stride);
            let kn_i = bufs.attn_k_normed.slice_view(row_off(k_norm_stride, i), k_norm_stride);
            let v_i = bufs.attn_v.slice_view(row_off(v_stride, i), v_stride);
            let sdpa_i = bufs.sdpa_out.slice_view(row_off(sdpa_stride, i), sdpa_stride);

            // F16-K copy: attn_k_normed -> hybrid K cache (F16).
            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                session.encoder_mut(), reg, metal_dev,
                &kn_i, &k_view, nkv as u32, hd as u32, cap as u32, cache_pos,
            )
            .map_err(|e| anyhow::anyhow!("batched F16-K write L{layer_idx} s{i}: {e}"))?;
            // FWHT-V quantize: attn_v -> hybrid V (TQ-HB packed + norms).
            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                session.encoder_mut(), reg, metal_dev,
                &v_i, &v_view, &v_norms_view,
                nkv as u32, hd as u32, cap as u32, cache_pos,
                is_ring, tq_scale_factor_d512, tq_codebook_bits,
            )
            .map_err(|e| anyhow::anyhow!("batched FWHT-V quant L{layer_idx} s{i}: {e}"))?;

            // SDPA: flash_attn_vec_hybrid (raw Q, F16-K, TQ-HB-V).
            let kv_seq_len = if is_ring {
                ((seq_pos_i + 1).min(cap)) as u32
            } else {
                (seq_pos_i + 1) as u32
            };
            let ring_start = if is_ring && kv_seq_len as usize >= cap {
                ((seq_pos_i + 1) % cap) as u32
            } else {
                0u32
            };
            let p_hyb = mlx_native::ops::flash_attn_vec_hybrid::FlashAttnVecTqHbParams {
                num_heads: nh as u32,
                num_kv_heads: nkv as u32,
                head_dim: hd as u32,
                kv_seq_len,
                kv_capacity: cap as u32,
                scale: 1.0,
                mask_type: if is_sliding { 2 } else { 1 },
                sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                softcap: 0.0,
                ring_start,
                scale_factor_d512: tq_scale_factor_d512,
                codebook_bits: tq_codebook_bits,
                fuse_fwht_pre: 0,
                nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(kv_seq_len),
            };
            mlx_native::ops::flash_attn_vec_hybrid::flash_attn_vec_hybrid(
                session.encoder_mut(), reg, dev,
                &q_i, &k_view, &v_view, &v_norms_view,
                &sdpa_i, &self.activations.sdpa_tmp, &p_hyb,
            )
            .map_err(|e| anyhow::anyhow!("batched flash_attn_vec_hybrid L{layer_idx} s{i}: {e}"))?;
            // FWHT-undo (V was FWHT-rotated pre-quant ⇒ SDPA out in FWHT domain).
            mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
                session.encoder_mut(), reg, metal_dev,
                &sdpa_i, nh as u32, hd as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched FWHT-undo L{layer_idx} s{i}: {e}"))?;
        }
        //
        // NB: the ops below read `bufs.sdpa_out`, which the (not-yet-built)
        // attention produces. Until the attention crux lands, this layer is
        // structurally complete but numerically incomplete — WIP, not wired, not
        // verified.

        let num_experts = self.num_experts;
        let top_k = self.layers[layer_idx].moe.top_k;
        let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;
        let interm = self.intermediate_size;

        // -- O-proj (BATCHED m=N): sdpa_out -> attn_out --
        session.barrier_between(
            &[&bufs.sdpa_out, &self.layers[layer_idx].attn.o_proj.buffer],
            &[&bufs.attn_out],
        );
        dispatch_qmatmul(
            session, reg, dev, &bufs.sdpa_out,
            &self.layers[layer_idx].attn.o_proj, &bufs.attn_out, nu,
            ImatrixHint::Layered { tag: "attn_output", layer: layer_idx },
        )?;

        // -- Fused post-attn norm + residual add (BATCHED rows=N): residual =
        // norm(attn_out, post_attn_w) + hidden. Default (non-split) path. --
        session.barrier_between(
            &[&bufs.hidden, &bufs.attn_out],
            &[&bufs.residual],
        );
        mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
            session.encoder_mut(), reg, metal_dev,
            &bufs.hidden,
            &bufs.attn_out,
            &self.layers[layer_idx].norms.post_attention_layernorm,
            &bufs.residual,
            hs as u32, nu, eps,
        )
        .map_err(|e| anyhow::anyhow!("batched post-attn norm+add L{layer_idx}: {e}"))?;

        // -- B8: pre-FF norm1 + pre-FF norm2 + router norm (BATCHED rows=N) --
        // Plain rms_norm rows=N (the scalar's `rms_norm_f32_hs_cached` is a
        // rows=1 pipeline-cache optimization; same math, per-row bit-identical).
        session.barrier_between(
            &[&bufs.residual],
            &[&bufs.norm_out, &bufs.moe_norm_out, &bufs.router_norm_out],
        );
        session
            .rms_norm(
                reg, metal_dev, &bufs.residual,
                &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                &bufs.norm_out, &self.activations.norm_params, nu, hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched pre-FF norm L{layer_idx}: {e}"))?;
        session
            .rms_norm(
                reg, metal_dev, &bufs.residual,
                &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                &bufs.moe_norm_out, &self.activations.norm_params, nu, hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched pre-FF norm 2 L{layer_idx}: {e}"))?;
        session
            .rms_norm(
                reg, metal_dev, &bufs.residual,
                &self.layers[layer_idx].moe.router_combined_weight,
                &bufs.router_norm_out, &self.activations.norm_params, nu, hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched router norm L{layer_idx}: {e}"))?;

        // -- B9: dense gate + dense up + router logits (BATCHED m=N) --
        session.barrier_between(
            &[&bufs.norm_out, &bufs.router_norm_out],
            &[&bufs.mlp_gate, &bufs.mlp_up, &bufs.moe_router_logits],
        );
        dispatch_qmatmul(
            session, reg, dev, &bufs.norm_out,
            &self.layers[layer_idx].mlp.gate_proj, &bufs.mlp_gate, nu,
            ImatrixHint::Layered { tag: "ffn_gate", layer: layer_idx },
        )?;
        dispatch_qmatmul(
            session, reg, dev, &bufs.norm_out,
            &self.layers[layer_idx].mlp.up_proj, &bufs.mlp_up, nu,
            ImatrixHint::Layered { tag: "ffn_up", layer: layer_idx },
        )?;
        dispatch_qmatmul(
            session, reg, dev, &bufs.router_norm_out,
            &self.layers[layer_idx].moe.router_proj, &bufs.moe_router_logits, nu,
            ImatrixHint::Layered { tag: "ffn_gate_inp", layer: layer_idx },
        )?;

        // -- B10: fused_gelu_mul (BATCHED, elementwise over N*intermediate) +
        // fused_moe_routing (PER-SLOT: each token's router logits -> top_k) --
        session.barrier_between(
            &[&bufs.mlp_gate, &bufs.mlp_up, &bufs.moe_router_logits],
            &[&bufs.mlp_fused, &bufs.moe_expert_ids, &bufs.moe_routing_weights_gpu],
        );
        {
            let total = (interm * n) as u32;
            let n_elements_bytes = total.to_ne_bytes();
            let pipeline = reg.get_pipeline("fused_gelu_mul", metal_dev)?;
            encode_with_args(
                session.encoder_mut(), pipeline,
                &[
                    (0, KernelArg::Buffer(&bufs.mlp_gate)),
                    (1, KernelArg::Buffer(&bufs.mlp_up)),
                    (2, KernelArg::Buffer(&bufs.mlp_fused)),
                    (3, KernelArg::Bytes(&n_elements_bytes)),
                ],
                mlx_native::MTLSize::new(total as u64, 1, 1),
                mlx_native::MTLSize::new(std::cmp::min(256, total as u64), 1, 1),
            );
        }
        // Per-slot routing: router_logits[i] -> expert_ids[i*top_k], weights.
        let rl_stride = elems(&bufs.moe_router_logits) / n; // num_experts
        let ids_stride = elems(&bufs.moe_expert_ids) / n; // top_k
        for i in 0..n {
            let rl_i = bufs.moe_router_logits.slice_view(row_off(rl_stride, i), rl_stride);
            let ids_i = bufs.moe_expert_ids.slice_view(row_off(ids_stride, i), ids_stride);
            let w_i = bufs
                .moe_routing_weights_gpu
                .slice_view(row_off(ids_stride, i), ids_stride);
            mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32(
                session.encoder_mut(), reg, metal_dev,
                &rl_i, &ids_i, &w_i,
                &self.layers[layer_idx].moe.per_expert_scale,
                num_experts as u32, top_k as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched MoE routing L{layer_idx} slot{i}: {e}"))?;
        }

        // -- B11: dense down (BATCHED m=N): mlp_fused -> mlp_down --
        session.barrier_between(
            &[&bufs.mlp_fused, &self.layers[layer_idx].mlp.down_proj.buffer],
            &[&bufs.mlp_down],
        );
        dispatch_qmatmul(
            session, reg, dev, &bufs.mlp_fused,
            &self.layers[layer_idx].mlp.down_proj, &bufs.mlp_down, nu,
            ImatrixHint::Layered { tag: "ffn_down", layer: layer_idx },
        )?;

        // -- MoE gate_up_id (BATCHED n_tokens=N — H-S2-tokenparity) --
        let stacked_gate_up = self
            .layers[layer_idx]
            .moe
            .stacked_gate_up
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("batched body requires fused _id MoE (stacked_gate_up) L{layer_idx}"))?;
        let stacked_down = self
            .layers[layer_idx]
            .moe
            .stacked_down
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("batched body requires fused _id MoE (stacked_down) L{layer_idx}"))?;
        session.barrier_between(
            &[&bufs.moe_norm_out, &bufs.moe_expert_ids, stacked_gate_up],
            &[&bufs.moe_gate_up_id_out],
        );
        let gu_params = mlx_native::GgmlQuantizedMatmulIdParams {
            n_tokens: nu,
            top_k: top_k as u32,
            n: (2 * moe_int) as u32,
            k: hs as u32,
            n_experts: num_experts as u32,
            expert_stride: self.layers[layer_idx].moe.gate_up_expert_stride,
            ggml_type: self.layers[layer_idx].moe.gate_up_ggml_dtype,
        };
        session
            .quantized_matmul_id_ggml(
                reg, dev, &bufs.moe_norm_out, stacked_gate_up,
                &bufs.moe_expert_ids, &bufs.moe_gate_up_id_out, &gu_params,
            )
            .map_err(|e| anyhow::anyhow!("batched gate_up _id L{layer_idx}: {e}"))?;

        // -- swiglu (BATCHED over N*top_k expert rows) --
        session.barrier_between(&[&bufs.moe_gate_up_id_out], &[&bufs.moe_swiglu_id_out]);
        mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
            session.encoder_mut(), reg, metal_dev,
            &bufs.moe_gate_up_id_out, &bufs.moe_swiglu_id_out,
            moe_int, top_k * n,
        )
        .map_err(|e| anyhow::anyhow!("batched swiglu L{layer_idx}: {e}"))?;

        // -- down_id (BATCHED n_tokens=N*top_k) --
        session.barrier_between(
            &[&bufs.moe_swiglu_id_out, &bufs.moe_expert_ids, stacked_down],
            &[&bufs.moe_down_id_out],
        );
        let dn_params = mlx_native::GgmlQuantizedMatmulIdParams {
            n_tokens: (top_k * n) as u32,
            top_k: 1,
            n: hs as u32,
            k: moe_int as u32,
            n_experts: num_experts as u32,
            expert_stride: self.layers[layer_idx].moe.down_expert_stride,
            ggml_type: self.layers[layer_idx].moe.down_ggml_dtype,
        };
        session
            .quantized_matmul_id_ggml(
                reg, dev, &bufs.moe_swiglu_id_out, stacked_down,
                &bufs.moe_expert_ids, &bufs.moe_down_id_out, &dn_params,
            )
            .map_err(|e| anyhow::anyhow!("batched down _id L{layer_idx}: {e}"))?;

        // -- post-FF norm 1 (BATCHED rows=N): mlp_down -> attn_out --
        session.barrier_between(&[&bufs.mlp_down], &[&bufs.attn_out]);
        session
            .rms_norm(
                reg, metal_dev, &bufs.mlp_down,
                &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                &bufs.attn_out, &self.activations.norm_params, nu, hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched post-FF norm1 L{layer_idx}: {e}"))?;

        // -- weighted_sum (PER-SLOT: each token's top_k experts -> its accum row) --
        // moe_accum reuses bufs.moe_norm_out? No — needs its own [N,hidden]; the
        // scalar uses self.activations.moe_accum. Here we reuse bufs.residual is
        // taken; route through a dedicated [N,hidden] view of moe_norm_out is
        // unsafe (still needed). Use bufs.mlp_gate as scratch [N,intermediate] is
        // wrong size. Allocate-free: weighted_sum writes [N,hidden]; reuse
        // bufs.norm_out (free after B9 consumed it into mlp_gate/up). norm_out is
        // [N,hidden]. Safe: B8/B9 already consumed norm_out; nothing reads it
        // again this layer.
        let down_stride = elems(&bufs.moe_down_id_out) / n; // top_k*hs
        let w_stride = elems(&bufs.moe_routing_weights_gpu) / n; // top_k
        let acc_stride = hs; // moe_accum row = hidden
        for i in 0..n {
            let din_i = bufs.moe_down_id_out.slice_view(row_off(down_stride, i), down_stride);
            let w_i = bufs.moe_routing_weights_gpu.slice_view(row_off(w_stride, i), w_stride);
            let acc_i = bufs.norm_out.slice_view(row_off(acc_stride, i), acc_stride);
            mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                session.encoder_mut(), reg, metal_dev,
                &din_i, &w_i, &acc_i, hs, top_k,
            )
            .map_err(|e| anyhow::anyhow!("batched weighted_sum L{layer_idx} slot{i}: {e}"))?;
        }

        // -- post-FF norm2 + combine (BATCHED rows=N): mlp_down = norm(moe_accum,
        // post_ff_norm_2) + attn_out  (moe_accum == bufs.norm_out, see above) --
        session.barrier_between(&[&bufs.attn_out, &bufs.norm_out], &[&bufs.mlp_down]);
        mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
            session.encoder_mut(), reg, metal_dev,
            &bufs.attn_out, &bufs.norm_out,
            &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
            &bufs.mlp_down, hs as u32, nu, eps,
        )
        .map_err(|e| anyhow::anyhow!("batched post-FF norm2+combine L{layer_idx}: {e}"))?;

        // -- end-of-layer (BATCHED rows=N): hidden = norm(mlp_down, post_ff_norm)
        // + residual, then * layer_scalar --
        let scalar_is_vector = self.layers[layer_idx].layer_scalar.element_count() > 1;
        session.barrier_between(&[&bufs.residual, &bufs.mlp_down], &[&bufs.hidden]);
        mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_scalar_f32(
            session.encoder_mut(), reg, metal_dev,
            &bufs.residual, &bufs.mlp_down,
            &self.layers[layer_idx].norms.post_feedforward_layernorm,
            &bufs.hidden, &self.layers[layer_idx].layer_scalar,
            nu, hs as u32, eps, scalar_is_vector,
        )
        .map_err(|e| anyhow::anyhow!("batched end-of-layer L{layer_idx}: {e}"))?;

        // Layer complete: bufs.hidden holds this layer's output for all N slots.
        Ok(())
    }

    /// ADR-040 S2/S3 — the `[N,hidden]` batched decode BODY: embed-gather N
    /// tokens → run the layer loop in `[N,hidden]` (per-slot attention against
    /// each slot's `multi_seq_kv_hybrid` region) → return each slot's final
    /// hidden row `[n, hidden]` (pre-final-norm — the same value scalar
    /// `forward_decode_capture_hidden` leaves in `self.activations.hidden`). The
    /// SlotAware worker feeds those rows to the proven batched head + finalize.
    ///
    /// `tokens[i]` / `slot_ids[i]` / `seq_positions[i]` describe slot `i`'s
    /// current decode step. Production hybrid-TQ path only (errors otherwise via
    /// the layer encode). WIP — gated by `slot_aware_n1` (N=1 bit-identical to
    /// scalar body) then `slot_aware_n4`; not wired until those pass.
    #[allow(dead_code)]
    pub(crate) fn forward_decode_body_batched(
        &self,
        tokens: &[u32],
        slot_ids: &[SlotId],
        seq_positions: &[usize],
        multi_seq_kv_hybrid: &mut [MultiSeqHybridKvBuffers],
        gpu: &mut GpuContext,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        let hs = self.hidden_size;
        if n == 0 {
            return Ok(Vec::new());
        }
        // tq params — read identically to the scalar (forward_gpu.rs:517/560).
        let tq_scale_factor_d512: f32 = match std::env::var("HF2Q_SCALE_FORMULA").as_deref() {
            Ok("sqrt256") => 16.0,
            Ok("sqrt512") => 512.0_f32.sqrt(),
            _ => 1.0,
        };
        let tq_codebook_bits: u32 = match std::env::var("HF2Q_TQ_CODEBOOK_BITS").as_deref() {
            Ok("4") => 0,
            Ok("5") => 5,
            Ok("6") => 6,
            _ => 8,
        };

        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let metal_dev = dev.metal_device();

        let bufs = BatchedDecodeBuffers::new(dev, &self.activations, n)?;
        let h_stride = bufs.hidden_stride();

        // Positions buffer [N] u32 for per-slot RoPE.
        let mut positions_buf = dev
            .alloc_buffer(n * 4, DType::U32, vec![n])
            .map_err(|e| anyhow::anyhow!("body_batched positions alloc: {e}"))?;
        {
            let p: &mut [u32] = positions_buf
                .as_mut_slice()
                .map_err(|e| anyhow::anyhow!("body_batched positions write: {e}"))?;
            for (i, &sp) in seq_positions.iter().enumerate() {
                p[i] = sp as u32;
            }
        }

        let mut s = exec
            .begin()
            .map_err(|e| anyhow::anyhow!("body_batched session begin: {e}"))?;

        // Embed-gather (PER-SLOT): token i -> bufs.hidden row i, scaled sqrt(hs).
        let scale = (hs as f32).sqrt();
        for i in 0..n {
            let h_i = bufs.hidden.slice_view(row_off(h_stride, i), h_stride);
            mlx_native::ops::elementwise::embedding_gather_scale_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.embed_weight, &h_i, tokens[i], hs, scale,
            )
            .map_err(|e| anyhow::anyhow!("body_batched embed slot{i}: {e}"))?;
        }

        // Layer loop in [N,hidden].
        let num_layers = self.layers.len();
        for layer_idx in 0..num_layers {
            self.encode_one_layer_batched(
                layer_idx, &bufs, n, &positions_buf,
                slot_ids, seq_positions, multi_seq_kv_hybrid,
                tq_scale_factor_d512, tq_codebook_bits,
                &mut s, exec, reg,
            )?;
        }

        s.finish()
            .map_err(|e| anyhow::anyhow!("body_batched session finish: {e}"))?;

        // Final hidden rows [n, hidden] (pre-final-norm).
        let out: &[f32] = bufs
            .hidden
            .as_slice()
            .map_err(|e| anyhow::anyhow!("body_batched read hidden: {e}"))?;
        Ok(out[..n * hs].to_vec())
    }
}
