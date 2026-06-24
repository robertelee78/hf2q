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
use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::model::MlxActivationBuffers;

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
