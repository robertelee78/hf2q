//! Hybrid KV cache for Qwen3.5 (full-attn KV + linear-attn SSM state).
//!
//! ADR-013 Decision 11. The Qwen3.5 layer stack is heterogeneous: full-
//! attention layers need a token-indexed K/V cache (standard transformer
//! behavior); linear-attention (Gated DeltaNet) layers need a recurrent
//! state matrix plus a 1D conv ring-buffer. This module owns all three,
//! allocated up-front and indexed per-layer.
//!
//! # Layout summary
//!
//! ```text
//! HybridKvCache
//!   full_attn:  Vec<FullAttnKvSlot>   len = # full-attention layers
//!     ┌─ k: MlxBuffer [head_dim, n_kv, max_seq_len, n_seqs]  f32
//!     ├─ v: MlxBuffer [head_dim, n_kv, max_seq_len, n_seqs]  f32
//!     └─ current_len:  Vec<u32>        one per seq
//!   mtp_slot: Option<FullAttnKvSlot>   present when nextn_predict_layers > 0
//!   linear_attn: Vec<LinearAttnStateSlot>  len = # linear-attention layers
//!     ├─ conv_state:         MlxBuffer [conv_channels, K-1, n_seqs] f32 (kernel native)
//!     ├─ conv_state_scratch: MlxBuffer [conv_channels, K-1, n_seqs] f32 (ping-pong)
//!     └─ recurrent:          MlxBuffer [D_k, D_v, num_v_heads, n_seqs] f32
//! ```
//!
//! # Per-layer ordering
//!
//! The `full_attn` vec is indexed by full-attention *rank* (0, 1, 2, ... for
//! the N-th full-attention layer in the model), NOT by original layer index.
//! Same for `linear_attn`. Callers use [`HybridKvCache::slot_index_for_layer`]
//! to translate a model layer index to the correct slot.
//!
//! For Qwen3.5-MoE (40 layers, full_attention_interval=4):
//! - Layer indices 3, 7, 11, ..., 39 are full-attention → full_attn[0..10].
//! - All other layers are linear-attention → linear_attn[0..30].
//!
//! # CPU reference
//!
//! The scalar CPU reference implementation for Gated DeltaNet (used as the
//! P7/P8 parity oracle) lives in
//! [`mlx_native::ops::gated_delta_net::cpu_reference_f32`] — we re-export
//! rather than duplicate.

use anyhow::{anyhow, Context, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

pub use super::gqa_q2_policy::Qwen35TqSdpaParams;
#[allow(unused_imports)]
pub use mlx_native::ops::gated_delta_net::cpu_reference_f32 as gated_delta_net_cpu_ref;

use super::gqa_q2_policy::use_gqa_q2_tq_sdpa;
use super::{Qwen35Config, Qwen35LayerKind};
use tq_arena::TqArenaLayout;

mod source_teacher;
mod tq_arena;

pub(super) use source_teacher::{
    plan_qwen35_base_text_cache, prepare_qwen35_base_text_cache, PreparedQwen35BaseTextCacheV1,
};

/// Per-full-attention-layer KV slot.
pub struct FullAttnKvSlot {
    /// Conventional keys buffer
    /// `[n_seqs, n_kv_heads, max_seq_len, head_dim]` F32. Present only
    /// when TQ KV is disabled; production TQ slots do not retain an F32
    /// shadow copy.
    pub k: Option<MlxBuffer>,
    /// Values buffer — same shape and dtype as `k`.
    pub v: Option<MlxBuffer>,
    /// Per-seq write cursor. `current_len[s]` = number of tokens already
    /// stored for sequence s.
    pub current_len: Vec<u32>,
    /// Production TQ K/V buffers. `Some` when the cache is constructed
    /// with `tq_kv_active = true` (the default and canonical-launcher
    /// path); `None` for the explicit `HF2Q_TQ_KV=0` F32 control path.
    pub tq: Option<TqFullAttnKvBuffers>,
}

/// Per-linear-attention-layer SSM state + conv ring buffer.
pub struct LinearAttnStateSlot {
    /// DeltaNet conv1d ring buffer (active read buffer): `[conv_channels, K-1, n_seqs]` f32.
    ///
    /// Layout matches the ssm_conv kernel's expected `state[i, c, s]` at offset
    /// `s * (K-1) * channels + c * (K-1) + i`, i.e. channels-major with K-1 stride 1.
    /// Ping-pong semantics (ADR-040 M-QWEN, PER-SLOT): which physical
    /// buffer is "current" vs "scratch" for a given slot is decided by
    /// that slot's [`Self::pp_flipped`] parity — read via
    /// [`LinearAttnStateSlot::conv_bufs_for_slot`], flip after a slot's
    /// step via [`LinearAttnStateSlot::swap_for_slot`]. Never assume this
    /// named field is current for any particular slot.
    pub conv_state: MlxBuffer,
    /// DeltaNet conv1d ring buffer (scratch, write target for ssm_conv kernel).
    /// Same shape as `conv_state`.  Swapped after each decode step.
    pub conv_state_scratch: MlxBuffer,
    /// DeltaNet recurrent state (current): `[D_k, D_v, num_v_heads, n_seqs]` f32.
    ///
    /// Ping-pong semantics (ADR-040 M-QWEN, PER-SLOT): current/scratch
    /// roles per slot are decided by [`Self::pp_flipped`] — read via
    /// [`LinearAttnStateSlot::recurrent_bufs_for_slot`], flip via
    /// [`LinearAttnStateSlot::swap_for_slot`] (zero copies, zero
    /// allocations, and — unlike the pre-M-QWEN whole-buffer swap —
    /// zero effect on other slots).
    pub recurrent: MlxBuffer,
    /// DeltaNet recurrent state (scratch, write target for GDN kernel).
    /// Same shape as `recurrent`.  Swapped with `recurrent` each decode step.
    pub recurrent_scratch: MlxBuffer,
    /// ADR-034 task #90 Step 2 (2026-05-21) — per-position recurrent
    /// state capture buffer for K=N speculative decoding partial-reject
    /// rollback.
    ///
    /// Shape: `[D_k, D_v, num_v_heads, n_tokens_max, n_seqs]` f32 with
    /// `D_k` innermost — same as `recurrent` extended with a `n_tokens_max`
    /// axis. `n_tokens_max` = `MAX_SPEC_DEPTH + 1` (≤ 8 for current
    /// implementation; the prefill chunk path never uses this slot).
    ///
    /// `None` (the default) when the cache was constructed in non-spec
    /// mode — non-capture `dispatch_gated_delta_net_decode` is used and
    /// the recurrent-only ping-pong is byte-identical to pre-#90 behavior.
    ///
    /// `Some(buf)` when the cache was constructed via
    /// [`HybridKvCache::new_with_spec_decode_capacity`]. The K=N spec
    /// runner routes `build_delta_net_layer` through
    /// `dispatch_gated_delta_net_decode_with_capture`, which writes
    /// per-position state into `capture_states` each token. On partial-
    /// reject of K drafts, the runner copies
    /// `capture_states[..., accepted_idx, ...]` → the slot's CURRENT
    /// recurrent buffer (parity-aware, ADR-040 M-QWEN) via
    /// [`HybridKvCache::rollback_la_to`].
    ///
    /// Memory cost (Qwen 3.5/3.6 D_k=D_v=128, n_v_heads=8, n_seqs=1,
    /// n_tokens_max=4): 2 MB per LA layer. ~60-90 MB total per forward
    /// across 30+ LA layers. Allocated once per spec-decode cache
    /// construction; freed when the cache drops.
    pub capture_states: Option<MlxBuffer>,
    /// ADR-034 task #90 Step 4c (2026-05-21) — per-position conv1d
    /// state capture buffer for K=N speculative decoding rollback.
    ///
    /// Shape: `[n_seqs, n_tokens_max, K-1, channels]` F32 with channels
    /// innermost — matches the mlx-native
    /// `dispatch_ssm_conv_with_capture` kernel (commit 92e322b) buffer 4
    /// contract.
    ///
    /// `None` (default) when cache was constructed in non-spec mode —
    /// non-capture `dispatch_ssm_conv` is used and conv ping-pong is
    /// byte-identical to pre-#90 behavior.
    ///
    /// `Some(buf)` when [`HybridKvCache::ensure_la_capture`] has been
    /// called. The K=N spec runner routes `build_delta_net_layer`
    /// through the capture variant, which writes per-position conv state.
    /// On partial-reject of K drafts, the runner copies
    /// `conv_capture_states[..., accepted_idx, ...]` → the slot's CURRENT
    /// conv buffer (parity-aware, ADR-040 M-QWEN) via
    /// [`HybridKvCache::rollback_la_to`] — paired with the
    /// recurrent capture rollback to fully restore DeltaNet state.
    ///
    /// Memory cost (Qwen 3.5/3.6 conv_channels=8192, K-1=3, n_seqs=1,
    /// n_tokens_max=4): 384 KB per LA layer. ~12 MB total per forward
    /// across 30+ LA layers.
    pub conv_capture_states: Option<MlxBuffer>,
    /// ADR-040 M-QWEN (2026-07-01) — PER-SLOT ping-pong parity.
    ///
    /// `pp_flipped[slot] == false` ⇒ that slot's CURRENT state lives in the
    /// `conv_state`/`recurrent` fields and its WRITE target is the
    /// `*_scratch` fields; `true` ⇒ roles reversed. One bit per slot
    /// because conv + recurrent always swap together (all swap sites).
    ///
    /// WHY: the buffers hold ALL `n_seqs` slots' state (`[.., n_seqs]`),
    /// but a decode tick writes only the ticking slot's region — the old
    /// whole-buffer `std::mem::swap` after one slot's tick flipped the
    /// read/write roles under every OTHER active slot, so their next tick
    /// read stale state. Invisible at N=1 (serial + engine-N=1 pins byte-
    /// exact); corrupted output at N≥2 concurrent (M-QWEN N=8 gate,
    /// 2026-07-01). Access the buffers via
    /// [`LinearAttnStateSlot::conv_bufs_for_slot`] /
    /// [`LinearAttnStateSlot::recurrent_bufs_for_slot`] and swap via
    /// [`LinearAttnStateSlot::swap_for_slot`]; never assume the named
    /// field is "current" for a given slot.
    pub pp_flipped: Vec<bool>,
}

impl FullAttnKvSlot {
    /// ADR-027 Phase B iter-9 — encode one token's K and V into the
    /// TQ-active byte-packed buffers via mlx-native's
    /// `dispatch_hadamard_quantize_kv_hb`.
    ///
    /// The kernel applies in-place FWHT + Lloyd-Max quantization onto
    /// `k_token` / `v_token` (both F32, shape `[n_kv_heads, head_dim]`)
    /// and writes the resulting U8 indices + F32 norms into
    /// `self.tq.k_packed` / `k_norms` / `v_packed` / `v_norms` at
    /// `write_pos`.
    ///
    /// **Caller contract (matches the GPU kernel's invariant):**
    /// - `self.tq` MUST be `Some` — the slot must have been constructed
    ///   via [`HybridKvCache::new_with_options`] with `tq_kv_active = true`.
    /// - `head_dim` must be 256 or 512 (kernel requirement).
    /// - `codebook_bits` must be 5, 6, or 8.
    /// - `cache_capacity` must equal the slot's `max_seq_len` from
    ///   construction time (the kernel computes the linear offset
    ///   `head*capacity*head_dim + write_pos*head_dim + dim`).
    /// - `write_pos < cache_capacity` for the global path; the kernel
    ///   wraps for the sliding path.
    ///
    /// **Production call site (iter-10):** the qwen35 forward path
    /// (`gpu_full_attn::full_attn_layer_gpu`) calls this once per
    /// (full-attn-layer × token) when `slot.tq.is_some()`. The decoded
    /// SDPA dispatch via `flash_attn_vec_tq_hb` reads from the same
    /// buffers without an F32 round-trip.
    ///
    /// **Iter-9 scope:** wrapper + GPU dispatch tests only. Iter-10
    /// wires this into `full_attn_layer_gpu`; iter-11 ships the SDPA
    /// dispatch + NRMSE-vs-F32 parity validation.
    ///
    /// # Errors
    ///
    /// - Returns `Err` if `self.tq.is_none()` (mantra: fail loud, no
    ///   silent fallback to F32 path).
    /// - Propagates errors from the GPU encode kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_token_to_tq(
        &mut self,
        k_token: &MlxBuffer,
        v_token: &MlxBuffer,
        n_kv_heads: u32,
        head_dim: u32,
        cache_capacity: u32,
        write_pos: u32,
        is_sliding: bool,
        scale_factor_d512: f32,
        codebook_bits: u32,
        encoder: &mut mlx_native::CommandEncoder,
        registry: &mut mlx_native::KernelRegistry,
        device: &MlxDevice,
    ) -> Result<()> {
        let tq = self.tq.as_mut().ok_or_else(|| {
            anyhow!(
                "FullAttnKvSlot::encode_token_to_tq: slot.tq is None — slot was not \
                 constructed in TQ-active mode (HybridKvCache::new_with_options \
                 tq_kv_active=true required)"
            )
        })?;
        let metal_dev = device.metal_device();
        // K side.
        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
            encoder,
            registry,
            metal_dev,
            k_token,
            &tq.k_packed,
            &tq.k_norms,
            n_kv_heads,
            head_dim,
            cache_capacity,
            write_pos,
            is_sliding,
            scale_factor_d512,
            codebook_bits,
        )
        .map_err(|e| anyhow!("encode_token_to_tq: dispatch_hadamard_quantize_kv_hb K: {e}"))?;
        // V side.
        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
            encoder,
            registry,
            metal_dev,
            v_token,
            &tq.v_packed,
            &tq.v_norms,
            n_kv_heads,
            head_dim,
            cache_capacity,
            write_pos,
            is_sliding,
            scale_factor_d512,
            codebook_bits,
        )
        .map_err(|e| anyhow!("encode_token_to_tq: dispatch_hadamard_quantize_kv_hb V: {e}"))?;
        Ok(())
    }

    /// ADR-027 Phase B iter-14 — multi-token TQ encode for prefill.
    ///
    /// Loops mlx-native's `dispatch_hadamard_quantize_kv_hb_seq` (per-token
    /// dispatch with successive `src_offset` values) to encode `n_tokens`
    /// positions of the seq-major K or V buffer into this slot's TQ
    /// buffers, starting at cache slot `cache_write_pos_start`.
    ///
    /// **Caller contract:**
    /// - `self.tq` MUST be `Some` (TQ-active mode required).
    /// - `kv_seq_major` is F32 with at least
    ///   `n_tokens × num_kv_heads × head_dim` elements (seq-major layout
    ///   `[n_tokens, num_kv_heads, head_dim]`); typical production
    ///   passing the K or V projection output before it lands in the
    ///   F32 cache.
    /// - `is_k = true` selects the K-side TQ buffers; `false` selects V.
    ///   This keeps the prefill encode loop in `gpu_full_attn` clean —
    ///   one call per side per layer per chunk.
    ///
    /// Iter-15 wires this at all 4 KV write sites in
    /// `gpu_full_attn::full_attn_layer_gpu` (decode, prefill, fused
    /// stage_ab prefill, decode_into).
    ///
    /// # Errors
    ///
    /// - `Err` if `self.tq.is_none()`.
    /// - Propagates errors from the GPU encode kernel (head_dim ∈
    ///   {256, 512}, codebook_bits ∈ {5, 6, 8}, src_size validation,
    ///   non-sliding overflow at `write_pos_start + n_tokens >
    ///   cache_capacity`).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_seq_tokens_to_tq_for_slot(
        &mut self,
        kv_seq_major: &MlxBuffer,
        is_k: bool,
        n_tokens: u32,
        n_kv_heads: u32,
        head_dim: u32,
        cache_capacity: u32,
        cache_write_pos_start: u32,
        src_tok_offset: u32,
        is_sliding: bool,
        scale_factor_d512: f32,
        codebook_bits: u32,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        encoder: &mut mlx_native::CommandEncoder,
        registry: &mut mlx_native::KernelRegistry,
        device: &MlxDevice,
    ) -> Result<()> {
        let tq = self.tq.as_mut().ok_or_else(|| {
            anyhow!(
                "encode_seq_tokens_to_tq: slot.tq is None — slot was not \
                 constructed in TQ-active mode"
            )
        })?;
        let views = tq.slot_views(slot_id, n_kv_heads, cache_capacity, head_dim)?;
        let (packed, norms) = if is_k {
            (&views.k_packed, &views.k_norms)
        } else {
            (&views.v_packed, &views.v_norms)
        };
        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq(
            encoder,
            registry,
            device.metal_device(),
            kv_seq_major,
            packed,
            norms,
            n_kv_heads,
            head_dim,
            views.capacity_tokens,
            cache_write_pos_start,
            n_tokens,
            src_tok_offset,
            is_sliding,
            scale_factor_d512,
            codebook_bits,
        )
        .map_err(|e| {
            anyhow!(
                "encode_seq_tokens_to_tq: dispatch_hadamard_quantize_kv_hb_seq \
                 ({} side, n_tokens={n_tokens}, write_pos_start={cache_write_pos_start}): {e}",
                if is_k { "K" } else { "V" }
            )
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_seq_tokens_to_tq(
        &mut self,
        kv_seq_major: &MlxBuffer,
        is_k: bool,
        n_tokens: u32,
        n_kv_heads: u32,
        head_dim: u32,
        cache_capacity: u32,
        cache_write_pos_start: u32,
        src_tok_offset: u32,
        is_sliding: bool,
        scale_factor_d512: f32,
        codebook_bits: u32,
        encoder: &mut mlx_native::CommandEncoder,
        registry: &mut mlx_native::KernelRegistry,
        device: &MlxDevice,
    ) -> Result<()> {
        self.encode_seq_tokens_to_tq_for_slot(
            kv_seq_major,
            is_k,
            n_tokens,
            n_kv_heads,
            head_dim,
            cache_capacity,
            cache_write_pos_start,
            src_tok_offset,
            is_sliding,
            scale_factor_d512,
            codebook_bits,
            crate::serve::multi_seq_kv::SlotId(0),
            encoder,
            registry,
            device,
        )
    }

    /// ADR-027 Phase B iter-10 — dispatch the TQ SDPA kernel
    /// (`flash_attn_vec_tq_hb`) consuming this slot's `tq` buffers.
    ///
    /// **Caller contract (mirrors the GPU kernel):**
    /// - `self.tq` MUST be `Some` (constructed via
    ///   [`HybridKvCache::new_with_options`] with `tq_kv_active=true`).
    /// - `q` MUST be FWHT-rotated by the caller before this call (see
    ///   `mlx_native::ops::fwht_standalone::dispatch_fwht_f32`).
    ///   Shape: `[num_heads, head_dim]` F32.
    /// - `output` is the F32 destination buffer; the caller MUST apply
    ///   inverse FWHT to it after this call returns.
    ///   Shape: `[num_heads, head_dim]` F32.
    /// - `tmp` scratch buffer sized via
    ///   `mlx_native::ops::flash_attn_vec_tq_hb::tmp_buffer_bytes(...)`
    ///   (only used when NWG > 1; the kernel writes directly to
    ///   `output` when NWG == 1).
    /// - If K/V were encoded earlier on the same command encoder, the caller
    ///   MUST issue `encoder.memory_barrier()` before this dispatch. Metal may
    ///   overlap concurrent compute dispatches otherwise; production Qwen
    ///   decode does this at both cache-write call sites.
    ///
    /// **Iter-10 scope (this method):** dispatch wrapper + GPU sanity
    /// tests (output is finite + non-zero on real Metal). The full
    /// F32-baseline NRMSE-vs-TQ parity test is iter-11; the
    /// production-decode integration in `gpu_full_attn::full_attn_
    /// layer_gpu` is also iter-11.
    ///
    /// # Errors
    ///
    /// - Returns `Err` if `self.tq.is_none()` (mantra: fail loud).
    /// - Propagates errors from the GPU SDPA kernel (head_dim ∈
    ///   {256, 512}, codebook_bits ∈ {5, 6, 8}, kv_seq_len > 0,
    ///   kv_capacity ≥ kv_seq_len, …).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_tq_sdpa_for_slot(
        &self,
        q: &MlxBuffer,
        output: &MlxBuffer,
        tmp: &MlxBuffer,
        params: &Qwen35TqSdpaParams,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        encoder: &mut mlx_native::CommandEncoder,
        registry: &mut mlx_native::KernelRegistry,
        device: &MlxDevice,
    ) -> Result<()> {
        let tq = self.tq.as_ref().ok_or_else(|| {
            anyhow!(
                "FullAttnKvSlot::dispatch_tq_sdpa: slot.tq is None — slot was \
                 not constructed in TQ-active mode (HybridKvCache::new_with_options \
                 tq_kv_active=true required)"
            )
        })?;
        let views = tq.slot_views(
            slot_id,
            params.num_kv_heads,
            params.kv_capacity,
            params.head_dim,
        )?;
        let kernel_params = mlx_native::ops::flash_attn_vec_tq_hb::FlashAttnVecTqHbParams {
            num_heads: params.num_heads,
            num_kv_heads: params.num_kv_heads,
            head_dim: params.head_dim,
            kv_seq_len: params.kv_seq_len,
            kv_capacity: views.capacity_tokens,
            scale: params.scale,
            mask_type: params.mask_type,
            sliding_window: params.sliding_window,
            softcap: params.softcap,
            ring_start: params.ring_start,
            scale_factor_d512: params.scale_factor_d512,
            codebook_bits: params.codebook_bits,
            // ADR-028 iter-106: caller pre-rotates Q (qwen35 path keeps
            // current FWHT-pre dispatch).
            fuse_fwht_pre: 0,
            // ADR-028 iter-127a Path D: NSG axis. iter-127a scaffolds with
            // NSG=1 default (byte-identical). Adaptive policy lands once
            // cross-simdgroup reduce is verified at NSG=2,4.
            nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(params.kv_seq_len),
        };
        if use_gqa_q2_tq_sdpa(params) {
            static FIRST_GQA_Q2: std::sync::Once = std::sync::Once::new();
            FIRST_GQA_Q2.call_once(|| {
                tracing::info!(
                    kv_seq_len = params.kv_seq_len,
                    num_heads = params.num_heads,
                    num_kv_heads = params.num_kv_heads,
                    "Qwen TQ-HB decode selected GQA-cooperative Q2 attention"
                );
            });
            mlx_native::ops::flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_gqa(
                encoder,
                registry,
                device,
                q,
                &views.k_packed,
                &views.k_norms,
                &views.v_packed,
                &views.v_norms,
                output,
                tmp,
                &kernel_params,
                mlx_native::ops::flash_attn_vec_tq_hb::GqaTile::Q2,
            )
            .map_err(|e| anyhow!("dispatch_tq_sdpa: cooperative TQ-HB Q2: {e}"))?;
        } else {
            mlx_native::ops::flash_attn_vec_tq_hb::flash_attn_vec_tq_hb(
                encoder,
                registry,
                device,
                q,
                &views.k_packed,
                &views.k_norms,
                &views.v_packed,
                &views.v_norms,
                output,
                tmp,
                &kernel_params,
            )
            .map_err(|e| anyhow!("dispatch_tq_sdpa: flash_attn_vec_tq_hb: {e}"))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_tq_sdpa(
        &self,
        q: &MlxBuffer,
        output: &MlxBuffer,
        tmp: &MlxBuffer,
        params: &Qwen35TqSdpaParams,
        encoder: &mut mlx_native::CommandEncoder,
        registry: &mut mlx_native::KernelRegistry,
        device: &MlxDevice,
    ) -> Result<()> {
        self.dispatch_tq_sdpa_for_slot(
            q,
            output,
            tmp,
            params,
            crate::serve::multi_seq_kv::SlotId(0),
            encoder,
            registry,
            device,
        )
    }

    /// ADR-027 Phase B iter-31 (sub-sub-iter 23c-β.2) — dequantize a
    /// sequence of TQ-encoded K (or V) positions back to F32 in the
    /// FWHT-rotated domain, into a fresh GPU temp buffer. This is the
    /// bridge that lets the existing dense F32 prefill SDPA kernel read
    /// from a TQ-only KV cache (post-iter-32 F32-alloc-drop).
    ///
    /// **Output layout:** `[num_kv_heads, n_tokens, head_dim]` F32 — the
    /// head-major layout the dense prefill SDPA already expects (matches
    /// hf2q's full-attn KV cache shape `[n_seqs=1, n_kv_heads, max_seq,
    /// head_dim]` minus the leading `n_seqs` axis).
    ///
    /// **Output domain:** FWHT-rotated. The caller of this helper is
    /// expected to pre-rotate Q with the same FWHT before SDPA, then
    /// post-rotate the SDPA output to undo. This is the same convention
    /// the iter-15 decode-TQ chain (`dispatch_decode_sdpa_with_optional_tq`)
    /// uses; the prefill wiring (iter-32) follows it.
    ///
    /// **Codebook bits:** sourced from
    /// `crate::debug::INVESTIGATION_ENV.tq_codebook_bits` (matches the
    /// production write-side default — Gemma at `forward_mlx.rs:2313`,
    /// hf2q `gpu_full_attn::write_kv_with_optional_tq_encode`); falls
    /// back to 8 if env contains an unexpected value.
    ///
    /// # Arguments
    ///
    /// * `is_k`           — `true` to dequant K, `false` to dequant V.
    /// * `n_tokens`       — number of consecutive cache positions
    ///   `[start_pos..start_pos+n_tokens)` to dequant.
    /// * `start_pos`      — first cache position to read (inclusive).
    /// * `cache_capacity` — must equal the slot's `max_seq_len` from
    ///   construction time (the kernel uses it as the per-head stride).
    ///
    /// # Errors
    ///
    /// - `Err` if `self.tq.is_none()` (mantra: fail loud — caller must
    ///   construct the slot via `HybridKvCache::new_with_options(..,
    ///   tq_kv_active=true)`).
    /// - `Err` if `start_pos + n_tokens > cache_capacity` (preflight
    ///   inside `dispatch_tq_dequantize_hb_kv_seq`).
    /// - Propagates GPU alloc / dispatch errors.
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_seq_to_temp_f32_for_slot(
        &self,
        is_k: bool,
        n_tokens: u32,
        start_pos: u32,
        cache_capacity: u32,
        n_kv_heads: u32,
        head_dim: u32,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        encoder: &mut mlx_native::CommandEncoder,
        registry: &mut mlx_native::KernelRegistry,
        device: &MlxDevice,
    ) -> Result<MlxBuffer> {
        let tq = self.tq.as_ref().ok_or_else(|| {
            anyhow!(
                "FullAttnKvSlot::dequant_seq_to_temp_f32: slot.tq is None — slot \
                 was not constructed in TQ-active mode (HybridKvCache::new_with_options \
                 tq_kv_active=true required)"
            )
        })?;

        // Codebook-bits source: same INVESTIGATION_ENV cache the
        // write-side uses, with the same {5,6,8} validation + fallback to
        // 8 (mirrors gpu_full_attn::write_kv_with_optional_tq_encode and
        // dispatch_decode_sdpa_with_optional_tq). Reading via LazyLock
        // avoids per-call env::var().
        let cb_env = crate::debug::INVESTIGATION_ENV.tq_codebook_bits;
        let codebook_bits: u32 = if matches!(cb_env, 5 | 6 | 8) {
            cb_env
        } else {
            8
        };

        let n_elems = (n_kv_heads as usize) * (n_tokens as usize) * (head_dim as usize);
        let dst = device
            .alloc_buffer(
                n_elems * 4,
                DType::F32,
                vec![n_kv_heads as usize, n_tokens as usize, head_dim as usize],
            )
            .map_err(|e| {
                anyhow!(
                "dequant_seq_to_temp_f32: alloc temp [{n_kv_heads},{n_tokens},{head_dim}] f32: {e}"
            )
            })?;

        let views = tq.slot_views(slot_id, n_kv_heads, cache_capacity, head_dim)?;
        let (packed, norms) = if is_k {
            (&views.k_packed, &views.k_norms)
        } else {
            (&views.v_packed, &views.v_norms)
        };

        // scale_factor_d512=1.0: matches the "bare" per-block norm
        // convention the iter-15 decode TQ chain uses (and the iter-13
        // GPU litmus test PASS confirms is correct under NRMSE 0.008).
        mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv_seq(
            encoder,
            registry,
            device.metal_device(),
            packed,
            norms,
            &dst,
            n_kv_heads,
            head_dim,
            views.capacity_tokens,
            start_pos,
            n_tokens,
            /*scale_factor_d512=*/ 1.0,
            codebook_bits,
        )
        .map_err(|e| anyhow!("dequant_seq_to_temp_f32: dispatch_tq_dequantize_hb_kv_seq: {e}"))?;

        Ok(dst)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dequant_seq_to_temp_f32(
        &self,
        is_k: bool,
        n_tokens: u32,
        start_pos: u32,
        cache_capacity: u32,
        n_kv_heads: u32,
        head_dim: u32,
        encoder: &mut mlx_native::CommandEncoder,
        registry: &mut mlx_native::KernelRegistry,
        device: &MlxDevice,
    ) -> Result<MlxBuffer> {
        self.dequant_seq_to_temp_f32_for_slot(
            is_k,
            n_tokens,
            start_pos,
            cache_capacity,
            n_kv_heads,
            head_dim,
            crate::serve::multi_seq_kv::SlotId(0),
            encoder,
            registry,
            device,
        )
    }

    /// ADR-027 Phase B iter-32 (sub-sub-iter 23c-β.3) — dequant + un-rotate
    /// chain: dequant TQ to temp F32 (rotated domain), then apply
    /// `FWHT × sign-undo` per-(head, position) chunk so the output is in the
    /// **original (unrotated) F32 K/V domain**.
    ///
    /// **Why this exists:** the existing dense F32 prefill SDPA kernel
    /// (`apply_flash_attn_prefill_seq_major_resume` and friends) reads
    /// K/V in the unrotated domain. Dropping in
    /// `dequant_seq_to_temp_f32_unrotated` as a `slot.k.as_ref()` replacement
    /// makes the dense prefill SDPA work against TQ-only KV with no
    /// kernel changes (iter-33 wires this into the production path).
    ///
    /// **Round-trip property:** for any K written via
    /// `encode_seq_tokens_to_tq`, this helper recovers K to within the
    /// quant round-trip floor (iter-13 NRMSE 0.008 on single-position;
    /// `dequant_seq_to_temp_f32_unrotated_recovers_original_within_nrmse_threshold`
    /// validates this seq variant under the same 0.15 ADR-007 §F-0.3
    /// threshold at production cache shape).
    ///
    /// Output layout: same as `dequant_seq_to_temp_f32`
    /// (`[n_kv_heads, n_tokens, head_dim]` head-major F32) — only the
    /// values change (now in the unrotated domain).
    ///
    /// **Internal pipeline (single GPU encoder):**
    /// 1. `dispatch_tq_dequantize_hb_kv_seq` (iter-30) →  temp_f32 (rotated).
    /// 2. RAW barrier (FWHT-undo reads what dequant just wrote).
    /// 3. `dispatch_fwht_sign_undo_f32` with `num_heads = n_kv_heads * n_tokens`
    ///    — each (head, token) chunk of `head_dim` elements is one
    ///    independent rotation group; the kernel's threadgroup-per-head
    ///    grid fans out across all `(n_kv_heads × n_tokens)` chunks.
    ///
    /// # Errors
    ///
    /// Same as `dequant_seq_to_temp_f32` plus FWHT dispatch errors
    /// (`head_dim` ∉ {256, 512}).
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_seq_to_temp_f32_unrotated_for_slot(
        &self,
        is_k: bool,
        n_tokens: u32,
        start_pos: u32,
        cache_capacity: u32,
        n_kv_heads: u32,
        head_dim: u32,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        encoder: &mut mlx_native::CommandEncoder,
        registry: &mut mlx_native::KernelRegistry,
        device: &MlxDevice,
    ) -> Result<MlxBuffer> {
        // (1) Dequant in the rotated domain.
        let dst = self.dequant_seq_to_temp_f32_for_slot(
            is_k,
            n_tokens,
            start_pos,
            cache_capacity,
            n_kv_heads,
            head_dim,
            slot_id,
            encoder,
            registry,
            device,
        )?;

        // (2) RAW barrier: FWHT-undo kernel reads what dequant just wrote.
        encoder.memory_barrier();

        // (3) Per-(head, token) FWHT × sign-undo. The fwht_sign_undo
        // kernel processes `num_heads` independent chunks of `head_dim`
        // elements; we fan out across all `n_kv_heads × n_tokens` chunks
        // by passing the product as `num_heads`. Layout: temp_f32 is
        // `[n_kv_heads, n_tokens, head_dim]` flattened — each (h, t) chunk
        // of `head_dim` elements is one rotation group at offset
        // `(h * n_tokens + t) * head_dim`.
        let total_chunks = n_kv_heads.checked_mul(n_tokens).ok_or_else(|| {
            anyhow!(
                "dequant_seq_to_temp_f32_unrotated: n_kv_heads ({n_kv_heads}) × \
                 n_tokens ({n_tokens}) overflow u32"
            )
        })?;
        mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
            encoder,
            registry,
            device.metal_device(),
            &dst,
            total_chunks,
            head_dim,
        )
        .map_err(|e| {
            anyhow!("dequant_seq_to_temp_f32_unrotated: dispatch_fwht_sign_undo_f32: {e}")
        })?;

        Ok(dst)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dequant_seq_to_temp_f32_unrotated(
        &self,
        is_k: bool,
        n_tokens: u32,
        start_pos: u32,
        cache_capacity: u32,
        n_kv_heads: u32,
        head_dim: u32,
        encoder: &mut mlx_native::CommandEncoder,
        registry: &mut mlx_native::KernelRegistry,
        device: &MlxDevice,
    ) -> Result<MlxBuffer> {
        self.dequant_seq_to_temp_f32_unrotated_for_slot(
            is_k,
            n_tokens,
            start_pos,
            cache_capacity,
            n_kv_heads,
            head_dim,
            crate::serve::multi_seq_kv::SlotId(0),
            encoder,
            registry,
            device,
        )
    }
}

impl LinearAttnStateSlot {
    /// ADR-040 M-QWEN — (current, scratch) conv-state buffers FOR ONE SLOT,
    /// honoring that slot's ping-pong parity. "current" is the read buffer
    /// for the slot's next forward; "scratch" is its write target. Callers
    /// narrow to the slot region downstream (`narrow_la_ping_pong_to_slot`);
    /// the parity only decides which physical buffer plays which role for
    /// THIS slot.
    #[inline]
    pub fn conv_bufs_for_slot(
        &self,
        slot: crate::serve::scheduler::SlotId,
    ) -> (&MlxBuffer, &MlxBuffer) {
        if self.pp_flipped[slot.0 as usize] {
            (&self.conv_state_scratch, &self.conv_state)
        } else {
            (&self.conv_state, &self.conv_state_scratch)
        }
    }

    /// ADR-040 M-QWEN — (current, scratch) recurrent-state buffers for one
    /// slot, honoring that slot's ping-pong parity (see
    /// [`Self::conv_bufs_for_slot`]).
    #[inline]
    pub fn recurrent_bufs_for_slot(
        &self,
        slot: crate::serve::scheduler::SlotId,
    ) -> (&MlxBuffer, &MlxBuffer) {
        if self.pp_flipped[slot.0 as usize] {
            (&self.recurrent_scratch, &self.recurrent)
        } else {
            (&self.recurrent, &self.recurrent_scratch)
        }
    }

    /// ADR-040 M-QWEN — mutable CURRENT conv-state buffer for one slot
    /// (parity-aware). For writers that must land in the slot's live
    /// state (e.g. spec-decode rollback), never the named field directly.
    #[inline]
    pub fn conv_current_mut(&mut self, slot: crate::serve::scheduler::SlotId) -> &mut MlxBuffer {
        if self.pp_flipped[slot.0 as usize] {
            &mut self.conv_state_scratch
        } else {
            &mut self.conv_state
        }
    }

    /// ADR-040 M-QWEN — mutable CURRENT recurrent-state buffer for one
    /// slot (parity-aware). See [`Self::conv_current_mut`].
    #[inline]
    pub fn recurrent_current_mut(
        &mut self,
        slot: crate::serve::scheduler::SlotId,
    ) -> &mut MlxBuffer {
        if self.pp_flipped[slot.0 as usize] {
            &mut self.recurrent_scratch
        } else {
            &mut self.recurrent
        }
    }

    /// ADR-040 M-QWEN — flip ONE slot's ping-pong parity after its decode/
    /// prefill step wrote new state into that slot's scratch. O(1) bit
    /// flip; the physical buffers never move, so other slots' read/write
    /// roles are untouched (the whole-buffer `std::mem::swap` this
    /// replaces corrupted every other active slot at N≥2 concurrent —
    /// M-QWEN root cause, 2026-07-01). Covers BOTH conv and recurrent
    /// (they always swap together).
    #[inline]
    pub fn swap_for_slot(&mut self, slot: crate::serve::scheduler::SlotId) {
        let i = slot.0 as usize;
        self.pp_flipped[i] = !self.pp_flipped[i];
    }
}

/// Top-level hybrid cache holding both full-attention and linear-attention
/// per-layer state.
pub struct HybridKvCache {
    pub full_attn: Vec<FullAttnKvSlot>,
    /// Full-attention KV slot for the appended MTP block at
    /// `layer_idx == cfg.num_hidden_layers`; absent for non-MTP GGUFs.
    pub mtp_slot: Option<FullAttnKvSlot>,
    pub linear_attn: Vec<LinearAttnStateSlot>,
    /// Maximum tokens the full-attn K/V buffers can hold per sequence.
    pub max_seq_len: u32,
    pub n_seqs: u32,
    /// Number of DeltaNet conv channels (derived from config; cached here so
    /// tests and update helpers don't need to recompute).
    pub conv_channels: u32,
    /// Precomputed `full_attn_rank` for each model layer index, for O(1)
    /// lookup in the hot path.
    per_layer_slot: Vec<LayerSlot>,
    /// ADR-027 Phase B iter-28 (sub-iter 23b) — cache records its own
    /// TQ-active mode at construction time. Today this mirrors
    /// `slot.tq.is_some()` for every full-attn slot, but having it on the
    /// cache itself is the precondition for sub-iter 23c, where
    /// `FullAttnKvSlot.k`/`v` become `Option<MlxBuffer>` and the alloc
    /// branch needs to know whether to skip the F32 K/V allocation. Kept
    /// `pub` for symmetry with `n_seqs` / `max_seq_len` (read-only state
    /// derived from constructor inputs).
    pub tq_kv_active: bool,
    /// Capture storage may remain allocated between agentic turns, but the
    /// capture kernels must run only while this flag is set. Keeping activity
    /// separate from allocation avoids re-creating hundreds of megabytes of
    /// per-position DeltaNet buffers on every short cached continuation.
    la_capture_active_tokens: Option<u32>,
}

/// Resolved slot index for a given model layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSlot {
    Full(u32),   // index into `full_attn`
    Linear(u32), // index into `linear_attn`
}

impl std::fmt::Debug for HybridKvCacheSnapshot {
    /// Surface only counts + total bytes — `MlxBuffer` does not implement
    /// `Debug` (Metal device handles can't be safely printed) and
    /// dumping per-element contents would be useless at this scale (GB).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridKvCacheSnapshot")
            .field("full_attn_layers", &self.full_attn_k.len())
            .field("linear_attn_layers", &self.linear_conv.len())
            .field("has_mtp", &self.mtp.is_some())
            .field("total_bytes", &self.total_bytes())
            .finish()
    }
}

/// Deep-copy snapshot of a [`HybridKvCache`] — owns fresh `MlxBuffer`
/// allocations holding byte-equal contents at snapshot time.
///
/// Wedge-3 / ADR-005 iter-216 Phase B.  Used by `HybridPromptCache`
/// (engine_qwen35.rs Phase C) to save post-prefill cache state for replay
/// on the next equivalent prompt.  See [`HybridKvCache::snapshot`] for
/// the deep-copy contract and the DeltaNet ping-pong note.
///
/// **ADR-027 Phase B sub-sub-iter 23a-β (Optional fields)**: full-attn
/// K/V are Optional so iter-34 (sub-sub-iter 23c-β.5) can drop the F32
/// backing in TQ mode without producing zero-byte garbage. iter-23a-β
/// added the type; iter-34 flipped the alloc to actually emit `None` in
/// TQ-active mode for the realized 3.94× per-slot memory savings. The
/// codec at `qwen35_hybrid_persistor.rs` extracts via
/// `.as_ref().expect()` with explicit pinning; v3 codec (iter-36)
/// adds `kv_present: u8` + `tq_present: u8` per-slot flags so the
/// envelope round-trips both Optional K/V AND Optional TQ state.
pub struct HybridKvCacheSnapshot {
    /// One per full-attn layer (e.g. 16 for Qwen3.6 27B): K matrix
    /// bytes. `None` in TQ-only mode (iter-34 alloc-drop production
    /// path); `Some(buf)` on F32 path (legacy `tq_kv_active=false`).
    pub full_attn_k: Vec<Option<MlxBuffer>>,
    /// One per full-attn layer: V matrix bytes. Same Optional
    /// semantics as `full_attn_k`.
    pub full_attn_v: Vec<Option<MlxBuffer>>,
    /// One per full-attn layer: per-seq write cursor at snapshot time.
    pub full_attn_current_len: Vec<Vec<u32>>,
    /// **ADR-027 Phase B iter-35 (sub-iter 23d-α):** one per full-attn
    /// layer — TQ-encoded K/V state at snapshot time. `Some(_)` when
    /// the source slot had `slot.tq.is_some()` (i.e. `tq_kv_active=true`
    /// at construction); `None` for legacy F32-only slots.
    ///
    /// Pairs with [`Self::full_attn_k`]: iter-34 (F32 alloc-drop) made
    /// `full_attn_k`/`full_attn_v` `None` per slot when in TQ mode —
    /// without this TQ snapshot field, restore would leave the new
    /// cache's cursor-visible TQ rows unwritten and decode would produce
    /// garbage (LCP-resume in TQ-only mode would silently break).
    /// iter-35 closes that gap by mirroring TQ state into the snapshot.
    pub full_attn_tq: Vec<Option<TqKvSnapshot>>,
    /// MTP slot snapshot (present only when the source cache had one).
    pub mtp: Option<MtpKvSnapshot>,
    /// One per linear-attn (DeltaNet) layer: active conv-state bytes.
    /// Scratch is intentionally NOT snapshotted — see [`HybridKvCache::snapshot`].
    pub linear_conv: Vec<MlxBuffer>,
    /// One per linear-attn layer: active recurrent state bytes.
    pub linear_recurrent: Vec<MlxBuffer>,
}

/// Slot-local prompt-boundary checkpoint for agentic replay with different
/// generation parameters.
///
/// Full-attention K/V is append-only, so decoding after a prompt does not
/// modify the prompt rows. Rewinding one physical slot therefore needs only
/// its per-layer cursors plus the fixed-size DeltaNet state. Keeping the
/// sequence K/V in place avoids a prompt-length-sized duplicate allocation
/// for every agent slot.
pub struct HybridKvSlotAnchor {
    prompt_len: usize,
    full_attn_current_len: Vec<u32>,
    mtp_current_len: Option<u32>,
    linear_conv: Vec<Vec<u8>>,
    linear_recurrent: Vec<Vec<u8>>,
}

/// Lightweight rollback point for one target forward transaction.
///
/// A Qwen target forward writes each DeltaNet layer into the inactive
/// ping-pong buffer exactly once and then flips that slot's parity. The prior
/// state bytes therefore remain intact until another target forward begins.
/// Capturing cursors plus the selected buffer for each layer is sufficient to
/// restore an exact pre-forward boundary without copying the large recurrent
/// matrices on every decode tick.
pub(crate) struct HybridKvSlotTransaction {
    full_attn_current_len: Vec<u32>,
    mtp_current_len: Option<u32>,
    linear_pp_flipped: Vec<bool>,
}

impl HybridKvSlotAnchor {
    /// Physical bytes retained outside the live cache for this checkpoint.
    pub fn total_bytes(&self) -> usize {
        self.linear_conv
            .iter()
            .chain(self.linear_recurrent.iter())
            .map(Vec::len)
            .sum()
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }
}

/// **ADR-027 Phase B iter-35 (sub-iter 23d-α)** — deep-copy snapshot of
/// one full-attn slot's TQ-encoded K/V buffers (mirrors
/// [`TqFullAttnKvBuffers`]). Owned `MlxBuffer` allocations whose contents
/// byte-equal the source `slot.tq.k_packed`/`k_norms`/`v_packed`/`v_norms`
/// at snapshot time.
///
/// Why deep-copy and NOT Arc::clone: same rationale as the F32
/// snapshot path (see [`HybridKvCacheSnapshot`] doc-comment) — the live
/// cache's TQ buffers continue to be written by subsequent decode
/// steps; aliasing would let the snapshot drift in lockstep, defeating
/// the purpose of capturing pre-decode state.
pub struct TqKvSnapshot {
    /// Byte-packed K indices `[n_seqs, n_kv_heads, max_seq_len, head_dim]` U8.
    pub k_packed: MlxBuffer,
    /// K per-(seq, head, position) F32 norms.
    pub k_norms: MlxBuffer,
    /// Byte-packed V indices, same shape as `k_packed`.
    pub v_packed: MlxBuffer,
    /// V per-(seq, head, position) F32 norms, same shape as `k_norms`.
    pub v_norms: MlxBuffer,
    /// Mirrors [`TqFullAttnKvBuffers::norms_per_pos`] — captured so
    /// restore can validate shape consistency without re-deriving from
    /// `head_dim`.
    pub norms_per_pos: u32,
}

impl TqKvSnapshot {
    /// Total owned bytes — sum of all 4 buffer byte_lens.
    pub fn total_bytes(&self) -> usize {
        self.k_packed.byte_len()
            + self.k_norms.byte_len()
            + self.v_packed.byte_len()
            + self.v_norms.byte_len()
    }
}

/// ADR-027 Phase B iter-18 — full-attention KV byte breakdown.
///
/// Returned by [`HybridKvCache::full_attn_bytes_breakdown`]. Captures
/// per-component byte counts so operators can quantify TQ memory cost
/// vs F32 baseline empirically (and verify the iter-19 F32-drop savings
/// land as projected).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullAttnKvBytesBreakdown {
    /// Sum of `slot.k.byte_len() + slot.v.byte_len()` across every
    /// full-attn slot (regular + optional MTP). Always non-zero today;
    /// iter-19 will make this zero in TQ mode.
    pub f32_k_v_bytes: usize,
    /// Sum of `slot.tq.k_packed.byte_len() + slot.tq.v_packed.byte_len()`
    /// across every TQ-active slot. Zero when `tq_kv_active=false`.
    pub tq_packed_bytes: usize,
    /// Sum of `slot.tq.k_norms.byte_len() + slot.tq.v_norms.byte_len()`
    /// across every TQ-active slot. Zero when `tq_kv_active=false`.
    pub tq_norms_bytes: usize,
    /// Number of full-attn slots in `full_attn` (does NOT include MTP).
    pub n_full_attn_slots: usize,
    /// `true` iff `mtp_slot` is `Some` (one extra slot's worth of bytes
    /// is in the totals above).
    pub has_mtp_slot: bool,
}

impl FullAttnKvBytesBreakdown {
    /// Total bytes (F32 + TQ packed + TQ norms). Useful for `kv_alloc`
    /// banner reporting + memory budget enforcement.
    pub fn total_bytes(&self) -> usize {
        self.f32_k_v_bytes + self.tq_packed_bytes + self.tq_norms_bytes
    }

    /// Total TQ bytes (packed + norms). Zero when not in TQ mode.
    pub fn tq_total_bytes(&self) -> usize {
        self.tq_packed_bytes + self.tq_norms_bytes
    }

    /// Projected savings ratio (`f32_bytes / tq_bytes`) once iter-19
    /// drops the F32 backing. Returns `None` when `tq_total_bytes() == 0`
    /// (legacy F32-only path; no TQ buffers to compare against).
    pub fn projected_iter19_savings_ratio(&self) -> Option<f64> {
        if self.tq_total_bytes() == 0 {
            return None;
        }
        Some(self.f32_k_v_bytes as f64 / self.tq_total_bytes() as f64)
    }
}

/// MTP slot snapshot — same shape as a `FullAttnKvSlot` snapshot but kept
/// as a dedicated struct so `Option<MtpKvSnapshot>` is explicit rather
/// than overloading `full_attn_k`/`full_attn_v` with a sentinel.
///
/// **ADR-027 Phase B sub-sub-iter 23a-α (Optional fields) + iter-34
/// (alloc-drop)**: K/V are Optional so iter-34 can drop the F32 backing
/// in TQ mode without producing zero-byte garbage. iter-23a-α added the
/// type; iter-34 flipped alloc to actually emit `None`. Producers and
/// consumers always emit/expect `Some` (no behavior change today).
/// `MtpKvSnapshot` itself stays `Option<MtpKvSnapshot>` at the
/// `HybridKvCacheSnapshot.mtp` level — that signals "MTP slot present
/// at all"; the inner `Option<MlxBuffer>` signals "F32 backing present
/// for the MTP slot's K/V".
pub struct MtpKvSnapshot {
    pub k: Option<MlxBuffer>,
    pub v: Option<MlxBuffer>,
    pub current_len: Vec<u32>,
    /// **ADR-027 Phase B iter-35 (sub-iter 23d-α):** MTP slot's TQ
    /// snapshot. Same Optional semantics as `HybridKvCacheSnapshot::full_attn_tq`.
    pub tq: Option<TqKvSnapshot>,
}

impl HybridKvCacheSnapshot {
    /// Total bytes the snapshot owns across all KV / SSM slots.  Useful
    /// for memory accounting + tracing the per-prompt cache footprint.
    pub fn total_bytes(&self) -> usize {
        let mut n = 0usize;
        let tq_bytes = |tq: &TqKvSnapshot| {
            tq.k_packed.byte_len()
                + tq.k_norms.byte_len()
                + tq.v_packed.byte_len()
                + tq.v_norms.byte_len()
        };
        // ADR-027 sub-sub-iter 23a-β: Optional full-attn K/V — sum only Some.
        for k in &self.full_attn_k {
            if let Some(buf) = k {
                n += buf.byte_len();
            }
        }
        for v in &self.full_attn_v {
            if let Some(buf) = v {
                n += buf.byte_len();
            }
        }
        for tq in self.full_attn_tq.iter().flatten() {
            n += tq_bytes(tq);
        }
        if let Some(s) = &self.mtp {
            // ADR-027 sub-sub-iter 23a-α: Optional MTP K/V — sum only Some.
            if let Some(buf) = &s.k {
                n += buf.byte_len();
            }
            if let Some(buf) = &s.v {
                n += buf.byte_len();
            }
            if let Some(tq) = &s.tq {
                n += tq_bytes(tq);
            }
        }
        for c in &self.linear_conv {
            n += c.byte_len();
        }
        for r in &self.linear_recurrent {
            n += r.byte_len();
        }
        n
    }
}

impl crate::serve::kv_persist::lcp_registry::ByteSized for HybridKvCacheSnapshot {
    /// Exact byte count of the snapshot across all KV / SSM slots.
    /// Delegates to `self.total_bytes()` which sums every `MlxBuffer::byte_len()`.
    fn byte_len(&self) -> u64 {
        self.total_bytes() as u64
    }
}

/// Allocate a fresh `MlxBuffer` of the same byte-length / dtype / shape
/// as `src`, and memcpy the source bytes into it.  Used by the snapshot
/// path to produce buffers that DON'T alias the source.
fn deep_copy_buffer(device: &MlxDevice, src: &MlxBuffer) -> Result<MlxBuffer> {
    let byte_len = src.byte_len();
    let dtype = src.dtype();
    let shape = src.shape().to_vec();
    let mut dst = device
        .alloc_buffer(byte_len, dtype, shape)
        .map_err(|e| anyhow!("deep_copy_buffer allocation: {e}"))?;
    let src_bytes = src
        .as_slice::<u8>()
        .map_err(|e| anyhow!("deep_copy_buffer src as_slice: {e}"))?;
    let dst_bytes = dst
        .as_mut_slice::<u8>()
        .map_err(|e| anyhow!("deep_copy_buffer dst as_mut_slice: {e}"))?;
    anyhow::ensure!(
        src_bytes.len() == dst_bytes.len(),
        "deep_copy_buffer byte-length mismatch (src={} dst={})",
        src_bytes.len(),
        dst_bytes.len()
    );
    dst_bytes.copy_from_slice(src_bytes);
    Ok(dst)
}

/// Deep-copy only the first `n_tokens` positions of a rank-4 sequence
/// buffer into a compact allocation whose sequence axis is exactly
/// `n_tokens`.  LCP checkpoints need the valid prefix, not the unused tail
/// of the request-sized cache allocation.
fn deep_copy_buffer_prefix(
    device: &MlxDevice,
    src: &MlxBuffer,
    n_tokens: usize,
    name: &str,
) -> Result<MlxBuffer> {
    let src_shape = src.shape();
    anyhow::ensure!(
        src_shape.len() == 4,
        "deep_copy_buffer_prefix ({name}): shape rank {} != 4",
        src_shape.len()
    );
    anyhow::ensure!(
        n_tokens > 0 && n_tokens <= src_shape[2],
        "deep_copy_buffer_prefix ({name}): n_tokens={n_tokens} outside 1..={}",
        src_shape[2]
    );
    let mut dst_shape = src_shape.to_vec();
    dst_shape[2] = n_tokens;
    let byte_len = dst_shape
        .iter()
        .try_fold(src.dtype().size_of(), |bytes, dim| bytes.checked_mul(*dim))
        .ok_or_else(|| anyhow!("deep_copy_buffer_prefix ({name}): byte length overflow"))?;
    let mut dst = device
        .alloc_buffer(byte_len, src.dtype(), dst_shape)
        .map_err(|e| anyhow!("deep_copy_buffer_prefix ({name}) allocation: {e}"))?;
    let src_bytes = src
        .as_slice::<u8>()
        .map_err(|e| anyhow!("deep_copy_buffer_prefix ({name}) src as_slice: {e}"))?;
    let dst_bytes = dst
        .as_mut_slice::<u8>()
        .map_err(|e| anyhow!("deep_copy_buffer_prefix ({name}) dst as_mut_slice: {e}"))?;
    let n_seqs = src_shape[0];
    let n_kv_heads = src_shape[1];
    let src_max_seq = src_shape[2];
    let head_pos_bytes = src_shape[3] * src.dtype().size_of();
    let copy_bytes = n_tokens * head_pos_bytes;
    let src_head_stride_bytes = src_max_seq * head_pos_bytes;
    let dst_head_stride_bytes = copy_bytes;
    let src_seq_stride_bytes = n_kv_heads * src_head_stride_bytes;
    let dst_seq_stride_bytes = n_kv_heads * dst_head_stride_bytes;
    for seq in 0..n_seqs {
        let src_seq_off = seq * src_seq_stride_bytes;
        let dst_seq_off = seq * dst_seq_stride_bytes;
        for head in 0..n_kv_heads {
            let src_off = src_seq_off + head * src_head_stride_bytes;
            let dst_off = dst_seq_off + head * dst_head_stride_bytes;
            dst_bytes[dst_off..dst_off + copy_bytes]
                .copy_from_slice(&src_bytes[src_off..src_off + copy_bytes]);
        }
    }
    Ok(dst)
}

fn deep_copy_snapshot_sequence_buffer(
    device: &MlxDevice,
    src: &MlxBuffer,
    prefix_tokens: Option<usize>,
    name: &str,
) -> Result<MlxBuffer> {
    match prefix_tokens {
        Some(n_tokens) => deep_copy_buffer_prefix(device, src, n_tokens, name),
        None => deep_copy_buffer(device, src),
    }
}

/// ADR-017 Phase E.a B.5 — partial-position copy of full-attn slot
/// K/V buffers.  Both source and destination have shape
/// `[n_seqs, n_kv_heads, max_seq_len_*, head_dim]` (rank-4, the
/// `FullAttnKvSlot::new` layout) with F32 elements; we copy the first
/// `n_tokens` positions per (seq, head).
///
/// The two buffers may have DIFFERENT `max_seq_len` dimensions; the
/// per-head stride differs accordingly.  All other dimensions
/// (`n_seqs`, `n_kv_heads`, `head_dim`) MUST match.
fn partial_copy_slot(
    src: &MlxBuffer,
    dst: &mut MlxBuffer,
    n_tokens: usize,
    name: &str,
) -> Result<()> {
    let src_shape = src.shape();
    let dst_shape = dst.shape();
    anyhow::ensure!(
        src_shape.len() == 4,
        "partial_copy_slot ({name}): src shape rank {} != 4 (expected \
         [n_seqs, n_kv_heads, max_seq_len, head_dim])",
        src_shape.len()
    );
    anyhow::ensure!(
        dst_shape.len() == 4,
        "partial_copy_slot ({name}): dst shape rank {} != 4 (expected \
         [n_seqs, n_kv_heads, max_seq_len, head_dim])",
        dst_shape.len()
    );
    let src_n_seqs = src_shape[0];
    let src_n_kv = src_shape[1];
    let src_max_seq = src_shape[2];
    let src_d = src_shape[3];
    let dst_n_seqs = dst_shape[0];
    let dst_n_kv = dst_shape[1];
    let dst_max_seq = dst_shape[2];
    let dst_d = dst_shape[3];
    anyhow::ensure!(
        src_n_seqs == dst_n_seqs && src_n_kv == dst_n_kv && src_d == dst_d,
        "partial_copy_slot ({name}): non-seq-dim mismatch — \
         src=[{src_n_seqs}, {src_n_kv}, _, {src_d}] vs \
         dst=[{dst_n_seqs}, {dst_n_kv}, _, {dst_d}]"
    );
    anyhow::ensure!(
        n_tokens <= src_max_seq && n_tokens <= dst_max_seq,
        "partial_copy_slot ({name}): n_tokens={n_tokens} exceeds capacity \
         (src_max_seq={src_max_seq}, dst_max_seq={dst_max_seq})"
    );
    if n_tokens == 0 {
        return Ok(());
    }
    let elem_size = src.dtype().size_of();
    anyhow::ensure!(
        elem_size == dst.dtype().size_of(),
        "partial_copy_slot ({name}): dtype size mismatch"
    );
    // Innermost (head_dim) is contiguous; per-head positions are
    // `head_dim` elements at the same stride for both buffers.
    let head_pos_bytes = src_d * elem_size;
    let copy_bytes = n_tokens * head_pos_bytes;
    // Per-head stride: max_seq_len * head_dim * elem_size.
    let src_head_stride_bytes = src_max_seq * head_pos_bytes;
    let dst_head_stride_bytes = dst_max_seq * head_pos_bytes;
    // Per-seq stride: n_kv_heads * max_seq_len * head_dim * elem_size.
    let src_seq_stride_bytes = src_n_kv * src_head_stride_bytes;
    let dst_seq_stride_bytes = dst_n_kv * dst_head_stride_bytes;

    let src_bytes = src
        .as_slice::<u8>()
        .map_err(|e| anyhow!("partial_copy_slot ({name}) src as_slice: {e}"))?;
    let dst_bytes = dst
        .as_mut_slice::<u8>()
        .map_err(|e| anyhow!("partial_copy_slot ({name}) dst as_mut_slice: {e}"))?;

    for seq in 0..src_n_seqs {
        let src_seq_off = seq * src_seq_stride_bytes;
        let dst_seq_off = seq * dst_seq_stride_bytes;
        for head in 0..src_n_kv {
            let src_off = src_seq_off + head * src_head_stride_bytes;
            let dst_off = dst_seq_off + head * dst_head_stride_bytes;
            dst_bytes[dst_off..dst_off + copy_bytes]
                .copy_from_slice(&src_bytes[src_off..src_off + copy_bytes]);
        }
    }
    Ok(())
}

/// Memcpy bytes from `src` to `dst`.  Both buffers must have equal
/// `byte_len`; mismatches are caller bugs (different cache shapes) and
/// surface as Err.
fn copy_buffer_bytes(src: &MlxBuffer, dst: &mut MlxBuffer) -> Result<()> {
    anyhow::ensure!(
        src.byte_len() == dst.byte_len(),
        "copy_buffer_bytes: byte-length mismatch (src={} dst={})",
        src.byte_len(),
        dst.byte_len()
    );
    let src_bytes = src
        .as_slice::<u8>()
        .map_err(|e| anyhow!("copy_buffer_bytes src as_slice: {e}"))?;
    let dst_bytes = dst
        .as_mut_slice::<u8>()
        .map_err(|e| anyhow!("copy_buffer_bytes dst as_mut_slice: {e}"))?;
    dst_bytes.copy_from_slice(src_bytes);
    Ok(())
}

fn copy_slot_region_out(
    src: &MlxBuffer,
    slot_idx: usize,
    n_seqs: usize,
    name: &str,
) -> Result<Vec<u8>> {
    anyhow::ensure!(n_seqs > 0, "copy_slot_region_out ({name}): n_seqs is zero");
    anyhow::ensure!(
        slot_idx < n_seqs,
        "copy_slot_region_out ({name}): slot {slot_idx} outside n_seqs={n_seqs}"
    );
    let src_bytes = src
        .as_slice::<u8>()
        .with_context(|| format!("copy_slot_region_out ({name}) as_slice"))?;
    anyhow::ensure!(
        src_bytes.len() % n_seqs == 0,
        "copy_slot_region_out ({name}): byte length {} not divisible by n_seqs={n_seqs}",
        src_bytes.len()
    );
    let per_slot = src_bytes.len() / n_seqs;
    let start = slot_idx * per_slot;
    Ok(src_bytes[start..start + per_slot].to_vec())
}

fn copy_slot_region_in(
    src: &[u8],
    dst: &mut MlxBuffer,
    slot_idx: usize,
    n_seqs: usize,
    name: &str,
) -> Result<()> {
    anyhow::ensure!(n_seqs > 0, "copy_slot_region_in ({name}): n_seqs is zero");
    anyhow::ensure!(
        slot_idx < n_seqs,
        "copy_slot_region_in ({name}): slot {slot_idx} outside n_seqs={n_seqs}"
    );
    let dst_bytes = dst
        .as_mut_slice::<u8>()
        .with_context(|| format!("copy_slot_region_in ({name}) as_mut_slice"))?;
    anyhow::ensure!(
        dst_bytes.len() % n_seqs == 0,
        "copy_slot_region_in ({name}): byte length {} not divisible by n_seqs={n_seqs}",
        dst_bytes.len()
    );
    let per_slot = dst_bytes.len() / n_seqs;
    anyhow::ensure!(
        src.len() == per_slot,
        "copy_slot_region_in ({name}): checkpoint bytes {} != destination slot bytes {per_slot}",
        src.len()
    );
    let start = slot_idx * per_slot;
    dst_bytes[start..start + per_slot].copy_from_slice(src);
    Ok(())
}

/// DeltaNet 1D conv kernel width — Qwen3.5 uses 4; kept as a constant here so
/// the conv-state allocation math is explicit. If the config ever varies, the
/// value is the runtime authority (`cfg.linear_conv_kernel_dim`).
pub const DELTA_NET_CONV_K: u32 = 4;

impl HybridKvCache {
    /// Capture the exact cursor and DeltaNet buffer-selection boundary before
    /// at most one target forward mutates this slot.
    pub(crate) fn begin_slot_transaction(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
        target_cursor: u32,
    ) -> Result<HybridKvSlotTransaction> {
        let slot_idx = slot.0 as usize;
        anyhow::ensure!(
            slot_idx < self.n_seqs as usize,
            "begin_slot_transaction: slot {} outside n_seqs={}",
            slot.0,
            self.n_seqs
        );
        let mut full_attn_current_len = Vec::with_capacity(self.full_attn.len());
        for (layer_idx, full) in self.full_attn.iter().enumerate() {
            let cursor = full.current_len.get(slot_idx).copied().ok_or_else(|| {
                anyhow!("begin_slot_transaction: full_attn[{layer_idx}] cursor missing")
            })?;
            anyhow::ensure!(
                cursor == target_cursor,
                "begin_slot_transaction: full_attn[{layer_idx}] cursor={cursor} != target_cursor={target_cursor}"
            );
            full_attn_current_len.push(cursor);
        }
        let mtp_current_len = self
            .mtp_slot
            .as_ref()
            .map(|mtp| {
                mtp.current_len.get(slot_idx).copied().ok_or_else(|| {
                    anyhow!(
                        "begin_slot_transaction: MTP cursor missing for slot {}",
                        slot.0
                    )
                })
            })
            .transpose()?;
        let linear_pp_flipped = self
            .linear_attn
            .iter()
            .enumerate()
            .map(|(layer_idx, linear)| {
                linear.pp_flipped.get(slot_idx).copied().ok_or_else(|| {
                    anyhow!(
                        "begin_slot_transaction: linear_attn[{layer_idx}] parity missing for slot {}",
                        slot.0
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(HybridKvSlotTransaction {
            full_attn_current_len,
            mtp_current_len,
            linear_pp_flipped,
        })
    }

    /// Restore a transaction captured by [`Self::begin_slot_transaction`].
    ///
    /// This must run before another target forward reuses the inactive
    /// DeltaNet buffers. Appended full-attention/MTP K/V rows are left in
    /// place and made unobservable by rewinding their cursors.
    pub(crate) fn rollback_slot_transaction(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        transaction: &HybridKvSlotTransaction,
    ) -> Result<()> {
        let slot_idx = slot.0 as usize;
        anyhow::ensure!(
            slot_idx < self.n_seqs as usize,
            "rollback_slot_transaction: slot {} outside n_seqs={}",
            slot.0,
            self.n_seqs
        );
        anyhow::ensure!(
            transaction.full_attn_current_len.len() == self.full_attn.len(),
            "rollback_slot_transaction: full-attention layer count mismatch"
        );
        anyhow::ensure!(
            transaction.linear_pp_flipped.len() == self.linear_attn.len(),
            "rollback_slot_transaction: linear-attention layer count mismatch"
        );
        // Validate the complete rollback point before changing any cursor or
        // ping-pong selector. A late validation failure must leave the live
        // slot wholly untouched so the serving layer can safely hard-reset
        // it instead of inheriting a partially rewound cache.
        for (layer_idx, (full, &saved)) in self
            .full_attn
            .iter()
            .zip(&transaction.full_attn_current_len)
            .enumerate()
        {
            let live = full.current_len.get(slot_idx).ok_or_else(|| {
                anyhow!("rollback_slot_transaction: full_attn[{layer_idx}] cursor missing")
            })?;
            anyhow::ensure!(
                *live >= saved,
                "rollback_slot_transaction: full_attn[{layer_idx}] live cursor {} is behind saved cursor {saved}",
                *live
            );
        }
        match (self.mtp_slot.as_ref(), transaction.mtp_current_len) {
            (Some(mtp), Some(saved)) => {
                let live = mtp.current_len.get(slot_idx).ok_or_else(|| {
                    anyhow!(
                        "rollback_slot_transaction: MTP cursor missing for slot {}",
                        slot.0
                    )
                })?;
                anyhow::ensure!(
                    *live >= saved,
                    "rollback_slot_transaction: MTP live cursor {} is behind saved cursor {saved}",
                    *live
                );
            }
            (None, None) => {}
            _ => anyhow::bail!("rollback_slot_transaction: MTP presence mismatch"),
        }
        for (layer_idx, linear) in self.linear_attn.iter().enumerate() {
            linear.pp_flipped.get(slot_idx).ok_or_else(|| {
                anyhow!("rollback_slot_transaction: linear_attn[{layer_idx}] parity missing")
            })?;
        }

        for (full, &saved) in self
            .full_attn
            .iter_mut()
            .zip(&transaction.full_attn_current_len)
        {
            full.current_len[slot_idx] = saved;
        }
        if let (Some(mtp), Some(saved)) = (self.mtp_slot.as_mut(), transaction.mtp_current_len) {
            mtp.current_len[slot_idx] = saved;
        }
        for (linear, &saved) in self
            .linear_attn
            .iter_mut()
            .zip(&transaction.linear_pp_flipped)
        {
            linear.pp_flipped[slot_idx] = saved;
        }
        Ok(())
    }

    /// Authoritative number of valid full-attention tokens for one physical
    /// agent slot. Bytes at or above this cursor are not observable.
    pub(crate) fn sequence_len_for_slot(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<u32, crate::serve::multi_seq_kv::MultiSeqError> {
        crate::serve::multi_seq_kv::MultiSeqKvCache::seq_len(self, slot)
    }

    /// Verify that every verifier full-attention layer has committed the same
    /// request-local cursor before the serving driver publishes a resumable
    /// prefill boundary.
    ///
    /// `sequence_len_for_slot` intentionally remains a cheap canonical read
    /// for metrics. Scheduler transaction boundaries need a release-mode
    /// check: a command-buffer or restore defect must not let layer 0 advance
    /// the public token ledger while another verifier layer remains behind.
    /// The optional MTP cache has an independent speculative cursor: ordinary
    /// base prefill does not advance it, and speculative decoding advances it
    /// relative to its own proposal window. It must therefore be snapshotted
    /// and restored, but never compared to the verifier prompt cursor here.
    pub(crate) fn validate_sequence_len_for_slot(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
        expected: usize,
    ) -> Result<()> {
        let slot_idx = slot.0 as usize;
        anyhow::ensure!(
            slot_idx < self.n_seqs as usize,
            "validate_sequence_len_for_slot: slot {} outside n_seqs={}",
            slot.0,
            self.n_seqs
        );
        let expected = u32::try_from(expected)
            .context("validate_sequence_len_for_slot: expected cursor exceeds u32")?;
        anyhow::ensure!(
            !self.full_attn.is_empty(),
            "validate_sequence_len_for_slot: Qwen cache has no full-attention layers"
        );
        for (layer_idx, full) in self.full_attn.iter().enumerate() {
            let actual = full.current_len.get(slot_idx).copied().ok_or_else(|| {
                anyhow!(
                    "validate_sequence_len_for_slot: full_attn[{layer_idx}] cursor missing for slot {}",
                    slot.0
                )
            })?;
            anyhow::ensure!(
                actual == expected,
                "validate_sequence_len_for_slot: full_attn[{layer_idx}] cursor={actual} != expected={expected} for slot {}",
                slot.0
            );
        }
        Ok(())
    }

    /// Verify the target and optional MTP attention cursors at an exact
    /// speculative transaction boundary.
    ///
    /// Ordinary Qwen serving intentionally keeps the MTP cursor independent;
    /// callers must opt into this stronger invariant only after mirroring the
    /// same target batch through `MtpWeights::process_target_batch`.
    pub(crate) fn validate_speculative_cursors_for_slot(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
        expected: usize,
    ) -> Result<()> {
        self.validate_sequence_len_for_slot(slot, expected)?;
        let slot_idx = slot.0 as usize;
        let expected = u32::try_from(expected)
            .context("validate_speculative_cursors_for_slot: expected cursor exceeds u32")?;
        let mtp = self
            .mtp_slot
            .as_ref()
            .context("validate_speculative_cursors_for_slot: MTP slot missing")?;
        let actual = mtp.current_len.get(slot_idx).copied().ok_or_else(|| {
            anyhow!(
                "validate_speculative_cursors_for_slot: MTP cursor missing for slot {}",
                slot.0
            )
        })?;
        anyhow::ensure!(
            actual == expected,
            "validate_speculative_cursors_for_slot: MTP cursor={actual} != expected={expected} for slot {}",
            slot.0
        );
        Ok(())
    }

    /// Capture the prompt boundary for one physical agent slot without
    /// duplicating its append-only full-attention K/V rows.
    pub(crate) fn snapshot_slot_anchor(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
        prompt_len: usize,
    ) -> Result<HybridKvSlotAnchor> {
        let slot_idx = slot.0 as usize;
        let n_seqs = self.n_seqs as usize;
        anyhow::ensure!(
            slot_idx < n_seqs,
            "snapshot_slot_anchor: slot {} outside n_seqs={}",
            slot.0,
            self.n_seqs
        );
        anyhow::ensure!(
            prompt_len > 0 && prompt_len <= self.max_seq_len as usize,
            "snapshot_slot_anchor: prompt_len={prompt_len} outside 1..={}",
            self.max_seq_len
        );

        let mut full_attn_current_len = Vec::with_capacity(self.full_attn.len());
        for (layer_idx, full) in self.full_attn.iter().enumerate() {
            let cursor = *full.current_len.get(slot_idx).ok_or_else(|| {
                anyhow!("snapshot_slot_anchor: full_attn[{layer_idx}] cursor missing")
            })?;
            anyhow::ensure!(
                cursor as usize == prompt_len,
                "snapshot_slot_anchor: full_attn[{layer_idx}] cursor={cursor} != prompt_len={prompt_len} for slot {}",
                slot.0
            );
            full_attn_current_len.push(cursor);
        }
        let mtp_current_len = self
            .mtp_slot
            .as_ref()
            .map(|mtp| {
                let cursor = mtp.current_len.get(slot_idx).copied().ok_or_else(|| {
                    anyhow!(
                        "snapshot_slot_anchor: MTP cursor missing for slot {}",
                        slot.0
                    )
                })?;
                Ok::<u32, anyhow::Error>(cursor)
            })
            .transpose()?;

        let mut linear_conv = Vec::with_capacity(self.linear_attn.len());
        let mut linear_recurrent = Vec::with_capacity(self.linear_attn.len());
        for (layer_idx, linear) in self.linear_attn.iter().enumerate() {
            let (conv, _) = linear.conv_bufs_for_slot(slot);
            let (recurrent, _) = linear.recurrent_bufs_for_slot(slot);
            linear_conv.push(copy_slot_region_out(
                conv,
                slot_idx,
                n_seqs,
                &format!("linear_attn[{layer_idx}].conv"),
            )?);
            linear_recurrent.push(copy_slot_region_out(
                recurrent,
                slot_idx,
                n_seqs,
                &format!("linear_attn[{layer_idx}].recurrent"),
            )?);
        }

        Ok(HybridKvSlotAnchor {
            prompt_len,
            full_attn_current_len,
            mtp_current_len,
            linear_conv,
            linear_recurrent,
        })
    }

    /// Rewind one physical agent slot to a previously captured prompt
    /// boundary. Peer cursors, peer DeltaNet state, and all full-attention
    /// K/V bytes remain untouched.
    pub(crate) fn restore_slot_anchor(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        anchor: &HybridKvSlotAnchor,
    ) -> Result<()> {
        let slot_idx = slot.0 as usize;
        let n_seqs = self.n_seqs as usize;
        anyhow::ensure!(
            slot_idx < n_seqs,
            "restore_slot_anchor: slot {} outside n_seqs={}",
            slot.0,
            self.n_seqs
        );
        anyhow::ensure!(
            anchor.prompt_len <= self.max_seq_len as usize,
            "restore_slot_anchor: prompt_len={} exceeds max_seq_len={}",
            anchor.prompt_len,
            self.max_seq_len
        );
        anyhow::ensure!(
            anchor.full_attn_current_len.len() == self.full_attn.len(),
            "restore_slot_anchor: full-attn layer count mismatch"
        );
        anyhow::ensure!(
            anchor.linear_conv.len() == self.linear_attn.len()
                && anchor.linear_recurrent.len() == self.linear_attn.len(),
            "restore_slot_anchor: linear-attn layer count mismatch"
        );

        // The anchor intentionally owns no full-attention K/V copy. Refuse
        // the rewind unless the live append-only cursor proves those rows
        // are still populated in this same slot.
        for (layer_idx, (full, &saved_cursor)) in self
            .full_attn
            .iter_mut()
            .zip(anchor.full_attn_current_len.iter())
            .enumerate()
        {
            let live_cursor = full.current_len.get_mut(slot_idx).ok_or_else(|| {
                anyhow!("restore_slot_anchor: full_attn[{layer_idx}] cursor missing")
            })?;
            anyhow::ensure!(
                *live_cursor >= saved_cursor,
                "restore_slot_anchor: full_attn[{layer_idx}] live cursor {} is behind saved cursor {saved_cursor}; prompt K/V cannot be proven intact",
                *live_cursor
            );
            *live_cursor = saved_cursor;
        }
        match (self.mtp_slot.as_mut(), anchor.mtp_current_len) {
            (Some(mtp), Some(saved_cursor)) => {
                let live_cursor = mtp.current_len.get_mut(slot_idx).ok_or_else(|| {
                    anyhow!(
                        "restore_slot_anchor: MTP cursor missing for slot {}",
                        slot.0
                    )
                })?;
                anyhow::ensure!(
                    *live_cursor >= saved_cursor,
                    "restore_slot_anchor: MTP live cursor {} is behind saved cursor {saved_cursor}",
                    *live_cursor
                );
                *live_cursor = saved_cursor;
            }
            (None, None) => {}
            _ => anyhow::bail!("restore_slot_anchor: MTP presence mismatch"),
        }

        for (layer_idx, linear) in self.linear_attn.iter_mut().enumerate() {
            copy_slot_region_in(
                &anchor.linear_conv[layer_idx],
                &mut linear.conv_state,
                slot_idx,
                n_seqs,
                &format!("linear_attn[{layer_idx}].conv"),
            )?;
            copy_slot_region_in(
                &anchor.linear_recurrent[layer_idx],
                &mut linear.recurrent,
                slot_idx,
                n_seqs,
                &format!("linear_attn[{layer_idx}].recurrent"),
            )?;
            // The named buffers now hold the restored current state for this
            // slot. Peer parity remains exactly as it was.
            linear.pp_flipped[slot_idx] = false;
        }
        Ok(())
    }

    /// Allocate the full hybrid cache for a Qwen3.5 (dense or MoE) model.
    ///
    /// Allocates:
    /// - For each full-attention layer in `cfg.layer_types`: two f32 buffers
    ///   of shape `[head_dim, n_kv_heads, max_seq_len, n_seqs]`.
    /// - For each linear-attention layer: conv-state of shape `[K-1, conv_channels, n_seqs]`
    ///   and recurrent state of shape `[D_k, D_v, num_v_heads, n_seqs]`.
    ///
    /// Recurrent semantic state is explicitly zero-initialized at the end of
    /// `new()` via [`Self::reset`]. Full-attention and TQ arenas use the
    /// overwrite contract: their tails are uninitialized and inaccessible
    /// until the per-slot cursor makes a row visible.
    ///
    /// **ADR-015 iter61a (broken-window fix):** the prior implementation
    /// relied on `MTLResourceOptions::StorageModeShared` returning zeroed
    /// pages "on first access" via the OS page-zeroing path.  Empirically
    /// this is NOT guaranteed on macOS / Apple Silicon — a freshly
    /// allocated Metal buffer can contain residual bytes from a recently
    /// freed allocation in the same process / device heap region (the
    /// Metal allocator coalesces and recycles pages within its private
    /// pool before the OS sees the free).  In a cold process this even
    /// surfaces as run-to-run non-determinism: the heap state at the
    /// moment Metal services `newBufferWithLength` differs across cold
    /// invocations.
    ///
    /// Concretely this caused divergent decoded tokens at temperature=0
    /// (greedy) on Qwen3.5/3.6: the DeltaNet `ssm_conv` kernel reads
    /// `conv_state_in` (K-1 history rows) on the very first prefill call
    /// before any decode step has populated it, and the
    /// `gated_delta_net` kernel similarly reads `state_in` (the
    /// recurrent state) on the same first call.  Garbage in those
    /// buffers contaminates the prefill logits, which are argmax'd to
    /// produce the first decoded token — different garbage on each cold
    /// run, different first tokens, different generations.  The
    /// `feedback_no_broken_windows` standing directive applies: fix at
    /// the source rather than relying on undefined initialization.
    ///
    /// # Memory footprint
    ///
    /// The full-attention K/V caches dominate at long context. Example
    /// (Qwen3.5-MoE at max_position_embeddings = 262144, n_seqs = 1):
    /// - Per full-attn layer: 256*2*262144*1*4 = 512 MB × 2 (K+V) = 1 GB
    /// - Total for 10 full-attn layers ≈ 10 GB of KV cache alone.
    ///
    /// Callers should pick `max_seq_len` for their actual use (e.g. 8192 or
    /// 32768) rather than always using `cfg.max_position_embeddings`. See
    /// ADR-013 Risk R8.
    ///
    /// # Errors
    ///
    /// Returns an error if any buffer allocation fails or if `max_seq_len`
    /// or `n_seqs` is zero.
    pub fn new(
        cfg: &Qwen35Config,
        device: &MlxDevice,
        max_seq_len: u32,
        n_seqs: u32,
    ) -> Result<Self> {
        // ADR-027 Phase B iter-8: legacy constructor delegates to the
        // tq-aware variant with tq_kv_active=false. ALL 71 existing
        // call sites stay unchanged; production TQ-active dispatch
        // routes through `new_with_options` from iter-9 forward.
        Self::new_with_options(cfg, device, max_seq_len, n_seqs, false)
    }

    /// ADR-027 Phase B iter-8 — tq-aware constructor. When
    /// `tq_kv_active = true` each full-attention slot (including the
    /// optional MTP slot) is augmented with a [`TqFullAttnKvBuffers`]
    /// alongside its existing F32 K/V buffers (shadow-cache pattern,
    /// mirrors Gemma's `dense_kvs` + `leg_hb_encoded` co-existence at
    /// `forward_mlx.rs:739+824`).
    ///
    /// In iter-8 the TQ buffers were allocation-only scratch;
    /// the SDPA dispatch + KV-write branches that consume them are
    /// iter-9 scope. iter-11 (post-NRMSE-parity) drops the F32 backing
    /// in TQ mode for the full 3.94× memory savings claim from §1.
    ///
    /// Linear-attn slots are unchanged regardless of `tq_kv_active`
    /// (DeltaNet SSM state is already compressed; per ADR-027 §3
    /// non-goal "TQ on linear-attn DeltaNet state").
    ///
    /// # Errors
    ///
    /// Same preconditions as [`Self::new`] plus any TQ allocation
    /// failure (propagated from [`alloc_tq_full_attn_buffers`]).
    pub fn new_with_options(
        cfg: &Qwen35Config,
        device: &MlxDevice,
        max_seq_len: u32,
        n_seqs: u32,
        tq_kv_active: bool,
    ) -> Result<Self> {
        let mut cache = Self::allocate_with_profile(
            cfg,
            device,
            max_seq_len,
            n_seqs,
            tq_kv_active,
            true,
            None,
        )?;
        // Full-attention storage is intentionally uninitialized: its cursor
        // is zero and every readable position is overwritten before the
        // cursor advances. Recurrent DeltaNet state remains semantic-zero.
        cache.reset_all_buffers();
        Ok(cache)
    }

    /// Construct a multi-sequence cache whose full-attention TQ buffers hold
    /// only a small physical seed per slot. `max_seq_len` remains the full
    /// logical context limit; callers must grow a slot before making rows
    /// beyond its current physical capacity visible.
    pub fn new_with_growable_tq(
        cfg: &Qwen35Config,
        device: &MlxDevice,
        max_seq_len: u32,
        n_seqs: u32,
        initial_capacity_tokens: u32,
    ) -> Result<Self> {
        let mut cache = Self::allocate_with_profile(
            cfg,
            device,
            max_seq_len,
            n_seqs,
            true,
            true,
            Some(initial_capacity_tokens),
        )?;
        cache.reset_all_buffers();
        Ok(cache)
    }

    fn allocate_with_profile(
        cfg: &Qwen35Config,
        device: &MlxDevice,
        max_seq_len: u32,
        n_seqs: u32,
        tq_kv_active: bool,
        include_mtp: bool,
        tq_initial_capacity: Option<u32>,
    ) -> Result<Self> {
        if max_seq_len == 0 {
            return Err(anyhow!("HybridKvCache: max_seq_len must be > 0"));
        }
        if n_seqs == 0 {
            return Err(anyhow!("HybridKvCache: n_seqs must be > 0"));
        }

        let conv_channels = conv_channels_for(cfg);
        let k_minus1 = cfg.linear_conv_kernel_dim.saturating_sub(1).max(1);

        let mut full_attn = Vec::new();
        let mut linear_attn = Vec::new();
        let mut per_layer_slot = Vec::with_capacity(cfg.layer_types.len());

        for (layer_idx, kind) in cfg.layer_types.iter().enumerate() {
            match kind {
                Qwen35LayerKind::FullAttention => {
                    let rank = full_attn.len() as u32;
                    per_layer_slot.push(LayerSlot::Full(rank));
                    let mut slot =
                        alloc_full_attn_slot(cfg, device, max_seq_len, n_seqs, tq_kv_active)
                            .with_context(|| format!("alloc full-attn slot (layer {layer_idx})"))?;
                    if tq_kv_active {
                        let physical_capacity = tq_initial_capacity.unwrap_or(max_seq_len);
                        slot.tq = Some(
                            alloc_tq_full_attn_buffers(cfg, device, physical_capacity, n_seqs)
                                .with_context(|| {
                                    format!("alloc tq full-attn buffers (layer {layer_idx})")
                                })?,
                        );
                    }
                    full_attn.push(slot);
                }
                Qwen35LayerKind::LinearAttention => {
                    let rank = linear_attn.len() as u32;
                    per_layer_slot.push(LayerSlot::Linear(rank));
                    linear_attn.push(
                        alloc_linear_attn_slot(cfg, device, conv_channels, k_minus1, n_seqs)
                            .with_context(|| {
                                format!("alloc linear-attn slot (layer {layer_idx})")
                            })?,
                    );
                }
            }
        }

        let mtp_slot = if include_mtp && cfg.mtp_num_hidden_layers > 0 {
            let mut slot = alloc_full_attn_slot(cfg, device, max_seq_len, n_seqs, tq_kv_active)
                .context("alloc MTP full-attn slot")?;
            if tq_kv_active {
                let physical_capacity = tq_initial_capacity.unwrap_or(max_seq_len);
                slot.tq = Some(
                    alloc_tq_full_attn_buffers(cfg, device, physical_capacity, n_seqs)
                        .context("alloc tq full-attn buffers (MTP slot)")?,
                );
            }
            Some(slot)
        } else {
            None
        };

        Ok(HybridKvCache {
            full_attn,
            mtp_slot,
            linear_attn,
            max_seq_len,
            n_seqs,
            conv_channels,
            per_layer_slot,
            tq_kv_active,
            la_capture_active_tokens: None,
        })
    }

    /// Reset semantic state without touching unread full-attention pages.
    ///
    /// Full K/V is valid only below `current_len`; lowering the cursor makes
    /// prior bytes unobservable. Zeroing the entire logical capacity here
    /// would commit every page of every slot and defeat full-context virtual
    /// reservation. DeltaNet recurrent/conv state is semantic input, so
    /// [`Self::reset`] still zeros those comparatively small buffers.
    ///
    /// `pub(crate)` because the qwen35 `--benchmark` 5-iter loop in
    /// `src/serve/mod.rs::cmd_generate_qwen35` calls this between
    /// iterations to re-establish the exact byte-state a freshly
    /// constructed cache would have, without paying the allocator cost
    /// of full reallocation each iter.
    pub(crate) fn reset_all_buffers(&mut self) {
        self.reset();
    }

    /// Translate a model layer index (0..num_hidden_layers) to the matching
    /// slot in this cache.
    pub fn slot_index_for_layer(&self, layer_idx: u32) -> Option<LayerSlot> {
        self.per_layer_slot.get(layer_idx as usize).copied()
    }

    /// ADR-027 Phase B iter-18 — full-attention KV memory breakdown.
    ///
    /// Sums byte counts across every full-attn slot (regular + optional
    /// MTP) split into:
    /// - F32 K/V backing buffers (legacy + shadow-cache mode)
    /// - TQ packed indices (U8, present iff `tq_kv_active=true`)
    /// - TQ per-position norms (F32, present iff `tq_kv_active=true`)
    ///
    /// **Operator-driven mantra**: "TQ for all models we support, as well
    /// or better than peers." Peer KV-quant systems (KIVI, vLLM) ship
    /// 3-4× memory savings vs F32. Iter-15 wired the TQ chain alongside
    /// F32 (shadow cache) so output matches F32 byte-identically; iter-19
    /// will drop the F32 backing in TQ mode for the full 3.94× savings
    /// at qwen36 8K shape (33.55 MB F32 → 8.52 MB TQ per slot).
    ///
    /// This method gives operators the empirical numbers to size that
    /// gap before iter-19 lands. Tests pin the breakdown at qwen36 8K
    /// AND 32K shapes so any silent allocator drift surfaces immediately.
    pub fn full_attn_bytes_breakdown(&self) -> FullAttnKvBytesBreakdown {
        let mut f32_k_v_bytes: usize = 0;
        let mut tq_packed_bytes: usize = 0;
        let mut tq_norms_bytes: usize = 0;
        for slot in &self.full_attn {
            // iter-29 (sub-sub-iter 23c-α) + iter-34 (sub-sub-iter
            // 23c-β.5): None means TQ-only mode (iter-34 alloc-drop
            // production path); contributes 0 F32 bytes — exactly the
            // load-bearing 3.94× memory savings the iter-34 regression-pin
            // test `full_attn_bytes_breakdown_tq_on_drops_f32_at_qwen36_32k`
            // checks (340 MiB total at 32K vs 1.34 GB F32-only baseline).
            if let Some(buf) = slot.k.as_ref() {
                f32_k_v_bytes += buf.byte_len();
            }
            if let Some(buf) = slot.v.as_ref() {
                f32_k_v_bytes += buf.byte_len();
            }
            if let Some(tq) = &slot.tq {
                tq_packed_bytes += tq.k_packed.byte_len() + tq.v_packed.byte_len();
                tq_norms_bytes += tq.k_norms.byte_len() + tq.v_norms.byte_len();
            }
        }
        if let Some(slot) = self.mtp_slot.as_ref() {
            if let Some(buf) = slot.k.as_ref() {
                f32_k_v_bytes += buf.byte_len();
            }
            if let Some(buf) = slot.v.as_ref() {
                f32_k_v_bytes += buf.byte_len();
            }
            if let Some(tq) = &slot.tq {
                tq_packed_bytes += tq.k_packed.byte_len() + tq.v_packed.byte_len();
                tq_norms_bytes += tq.k_norms.byte_len() + tq.v_norms.byte_len();
            }
        }
        FullAttnKvBytesBreakdown {
            f32_k_v_bytes,
            tq_packed_bytes,
            tq_norms_bytes,
            n_full_attn_slots: self.full_attn.len(),
            has_mtp_slot: self.mtp_slot.is_some(),
        }
    }

    /// Reset all per-seq write cursors and zero out the recurrent/conv state.
    /// Does NOT zero the K/V buffers (callers overwrite them on subsequent
    /// tokens).
    /// Truncate every full-attention slot's `current_len[0]` to
    /// `new_len`. Independently from the MTP slot (see
    /// [`HybridKvCache::truncate_mtp_to`]) because the two slot families
    /// have different base offsets: full-attn slots get populated during
    /// prefill (`current_len` starts at `prompt_len`), but MTP slot
    /// starts empty at decode (`current_len = 0`).
    ///
    /// **Use case (ADR-034 K=N speculative, partial reject)**: after a
    /// batched verifier forward has written `spec_k + 1` positions but
    /// only `accepted + 1` of them are valid (the rest were drafted
    /// off-path tokens that the target rejected), this method rolls
    /// each full-attn slot back so next iter's forward writes start at
    /// the correct slot index. Without rollback the next iter's writes
    /// append AFTER the stale entries, leaving duplicate rope-positions
    /// in the slot — attention double-counts those positions and quality
    /// degrades (the "the the the…" / "** ** **" attractor bug at K=N
    /// chain).
    ///
    /// Targets only `full_attn` slots. Linear-attention DeltaNet state
    /// is invariant under this call (its recurrent / conv-state buffers
    /// don't carry a per-token cursor).
    pub fn truncate_full_attn_to(&mut self, new_len: u32) {
        for slot in self.full_attn.iter_mut() {
            for c in slot.current_len.iter_mut() {
                if *c > new_len {
                    *c = new_len;
                }
            }
        }
    }

    /// Truncate the optional MTP slot's `current_len[0]` to `new_len`.
    /// Counterpart to [`HybridKvCache::truncate_full_attn_to`] for the
    /// MTP-draft slot in K=N speculative decoding.
    pub fn truncate_mtp_to(&mut self, new_len: u32) {
        if let Some(slot) = self.mtp_slot.as_mut() {
            for c in slot.current_len.iter_mut() {
                if *c > new_len {
                    *c = new_len;
                }
            }
        }
    }

    /// ADR-040 Phase B4d (2026-05-30) — per-slot variant of
    /// [`Self::truncate_full_attn_to`].  Decrements only
    /// `current_len[slot.0]` on every full-attn slot; sibling slots'
    /// cursors are byte-untouched.  Bounds-first per A2b iter-1.5
    /// cfa-finding-F5: returns `Err` BEFORE any cursor mutation if
    /// `slot.0 >= n_seqs` (read from the canonical
    /// `current_len.len()` shape on the first full-attn slot — same
    /// invariant `seq_len(slot)` enforces).
    ///
    /// **Use case (ADR-040 Phase B4d spec-decode at SlotId(N>0))**:
    /// the K=N partial-reject path in [`super::spec_decode::SpecDecode::run_prompt`]
    /// rolls back the active slot's `current_len[slot.0]` after a
    /// batched verify wrote `spec_k + 1` positions but only
    /// `accepted + 1` were valid.  Sibling slots in other in-flight
    /// requests under the SlotAware scheduler (Phase C2c/C2d) must
    /// not have their cursors touched by this rollback.
    ///
    /// # Errors
    /// - `slot.0 >= n_seqs` (where `n_seqs` is the configured n_seqs
    ///   axis on the cache): returns `Err` with a clear "ADR-040
    ///   Phase B4d per-slot truncate contract" diagnostic.
    pub fn truncate_full_attn_to_for_slot(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        new_len: u32,
    ) -> Result<()> {
        // Bounds-FIRST per A2b iter-1.5 cfa-finding-F5.  The canonical
        // `n_seqs` for the cursor axis is the length of
        // `current_len` on the first full-attn slot.  An empty
        // `full_attn` Vec also fails (the spec-decode path requires
        // at least one full-attn slot via the prefill contract).
        let n_seqs = self
            .full_attn
            .first()
            .map(|s| s.current_len.len() as u32)
            .ok_or_else(|| {
                anyhow!(
                    "HybridKvCache::truncate_full_attn_to_for_slot({:?}, new_len={}): \
                     ADR-040 Phase B4d per-slot truncate contract — empty full_attn slot vec",
                    slot,
                    new_len,
                )
            })?;
        if slot.0 >= n_seqs {
            return Err(anyhow!(
                "HybridKvCache::truncate_full_attn_to_for_slot({:?}, new_len={}): \
                 ADR-040 Phase B4d per-slot truncate contract — slot {} >= n_seqs {}",
                slot,
                new_len,
                slot.0,
                n_seqs,
            ));
        }
        for slot_data in self.full_attn.iter_mut() {
            // Defensive shape guard: if a sibling layer disagrees with
            // the canonical n_seqs it's a multi-layer invariant
            // violation; treat as the bounds-check above already
            // would have caught.
            if let Some(cur) = slot_data.current_len.get_mut(slot.0 as usize) {
                if *cur > new_len {
                    *cur = new_len;
                }
            }
        }
        Ok(())
    }

    /// ADR-040 Phase B4d (2026-05-30) — per-slot variant of
    /// [`Self::truncate_mtp_to`].  Decrements only the MTP slot's
    /// `current_len[slot.0]`; sibling slots' MTP cursors are
    /// byte-untouched.  Bounds-first per A2b iter-1.5 cfa-finding-F5.
    /// No-op if `self.mtp_slot.is_none()` (the model lacks MTP).
    ///
    /// # Errors
    /// - `slot.0 >= n_seqs` on the MTP slot's `current_len` axis:
    ///   returns `Err` with a clear "ADR-040 Phase B4d per-slot
    ///   truncate contract" diagnostic.
    pub fn truncate_mtp_to_for_slot(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        new_len: u32,
    ) -> Result<()> {
        let Some(mtp) = self.mtp_slot.as_mut() else {
            return Ok(());
        };
        let n_seqs = mtp.current_len.len() as u32;
        if slot.0 >= n_seqs {
            return Err(anyhow!(
                "HybridKvCache::truncate_mtp_to_for_slot({:?}, new_len={}): \
                 ADR-040 Phase B4d per-slot truncate contract — slot {} >= n_seqs {}",
                slot,
                new_len,
                slot.0,
                n_seqs,
            ));
        }
        if let Some(cur) = mtp.current_len.get_mut(slot.0 as usize) {
            if *cur > new_len {
                *cur = new_len;
            }
        }
        Ok(())
    }

    /// ADR-034 task #90 Step 2 (2026-05-21) — lazily allocate the
    /// per-position capture buffer on every linear-attention slot for
    /// K=N speculative decoding. Idempotent: re-calling with the SAME
    /// `n_tokens_max` is a no-op; re-calling with a LARGER value
    /// reallocates (re-allocations clear the existing capture content).
    ///
    /// Caller MUST invoke this once before entering a K=N spec-decode
    /// loop with `n_tokens_max = MAX_SPEC_DEPTH + 1`. After this returns
    /// Ok, every `linear_attn[i].capture_states` is `Some(buf)` with
    /// shape `[D_k, D_v, num_v_heads, n_tokens_max, n_seqs]` F32
    /// (matching the mlx-native `dispatch_gated_delta_net_decode_with_capture`
    /// kernel's buffer 9 contract — see project_adr034_task90_gdn_kernel_shipped).
    ///
    /// On partial-reject of K drafts, [`Self::rollback_la_to`] copies
    /// `capture_states[..., accepted_idx, ...]` → `recurrent` (active).
    ///
    /// # Errors
    /// Returns `Err` when buffer alloc fails (OOM).
    pub fn ensure_la_capture(
        &mut self,
        cfg: &Qwen35Config,
        device: &MlxDevice,
        n_tokens_max: u32,
    ) -> Result<()> {
        if n_tokens_max == 0 {
            return Err(anyhow!("ensure_la_capture: n_tokens_max must be > 0"));
        }
        // Fail closed if allocation below is interrupted. The retained
        // buffers remain valid storage, but no forward may select capture
        // kernels until every linear-attention slot is ready.
        self.la_capture_active_tokens = None;
        let d_k = cfg.linear_key_head_dim as usize;
        let d_v = cfg.linear_value_head_dim as usize;
        let n_v_heads = cfg.linear_num_value_heads as usize;
        let n_seqs = self.n_seqs as usize;
        let state_elems = d_k * d_v * n_v_heads * n_seqs;
        let capture_elems = state_elems * (n_tokens_max as usize);
        let shape = vec![d_k, d_v, n_v_heads, n_tokens_max as usize, n_seqs];

        // ADR-034 task #90 Step 4c (2026-05-21) — also allocate the
        // companion conv1d state capture buffer. Layout per the
        // mlx-native `dispatch_ssm_conv_with_capture` kernel contract
        // (commit 92e322b): [n_seqs, n_tokens_max, K-1, channels] F32
        // with channels innermost.
        let conv_channels = conv_channels_for(cfg) as usize;
        let k_minus1 = (cfg.linear_conv_kernel_dim.saturating_sub(1)) as usize;
        let conv_state_elems = conv_channels * k_minus1 * n_seqs;
        let conv_capture_elems = (n_seqs) * (n_tokens_max as usize) * k_minus1 * conv_channels;
        let conv_shape = vec![n_seqs, n_tokens_max as usize, k_minus1, conv_channels];

        for slot in self.linear_attn.iter_mut() {
            // Recurrent capture: an existing larger allocation is also valid.
            // Kernel indexing derives its token capacity from the buffer shape,
            // while callers only write the requested suffix length.
            if let Some(buf) = slot.capture_states.as_ref() {
                if (n_seqs == 1 && buf.element_count() >= capture_elems)
                    || buf.element_count() == capture_elems
                {
                    // continue to conv check
                } else {
                    let cap_buf = device
                        .alloc_buffer(capture_elems * 4, DType::F32, shape.clone())
                        .map_err(|e| anyhow!("alloc capture_states: {e}"))?;
                    slot.capture_states = Some(cap_buf);
                }
            } else {
                let cap_buf = device
                    .alloc_buffer(capture_elems * 4, DType::F32, shape.clone())
                    .map_err(|e| anyhow!("alloc capture_states: {e}"))?;
                slot.capture_states = Some(cap_buf);
            }
            // Conv capture (Step 4c): same grow-only semantics.
            if let Some(buf) = slot.conv_capture_states.as_ref() {
                if (n_seqs == 1 && buf.element_count() >= conv_capture_elems)
                    || buf.element_count() == conv_capture_elems
                {
                    continue;
                }
            }
            let conv_cap_buf = device
                .alloc_buffer(conv_capture_elems * 4, DType::F32, conv_shape.clone())
                .map_err(|e| anyhow!("alloc conv_capture_states: {e}"))?;
            // Sanity: conv_state_elems must equal one per-token slice
            // of the capture buffer (defensive — wiring contract for
            // rollback_la_to).
            debug_assert_eq!(
                conv_state_elems,
                conv_capture_elems / (n_tokens_max as usize),
                "ensure_la_capture: conv per-token elems ({}) must equal conv_state_elems ({})",
                conv_capture_elems / (n_tokens_max as usize),
                conv_state_elems,
            );
            slot.conv_capture_states = Some(conv_cap_buf);
        }
        self.la_capture_active_tokens = Some(n_tokens_max);
        Ok(())
    }

    /// End a bounded capture operation while retaining its grow-only storage.
    /// Ordinary single-token decode sees capture as inactive and therefore
    /// stays on the non-capture kernels. A later capture of equal or smaller
    /// depth reuses the same buffers without Metal allocation or zero-fill.
    pub fn clear_la_capture(&mut self) {
        self.la_capture_active_tokens = None;
    }

    /// Whether the next forward must write per-position DeltaNet capture
    /// state. Allocation alone is deliberately not an activity signal.
    #[inline]
    pub fn la_capture_active(&self) -> bool {
        self.la_capture_active_tokens.is_some()
    }

    /// Requested capture depth for the active forward. Retained buffers may
    /// have a larger physical capacity; callers bind a zero-copy prefix view
    /// of exactly this many token positions to the capture kernels.
    #[inline]
    pub fn la_capture_active_tokens(&self) -> Option<u32> {
        self.la_capture_active_tokens
    }

    /// ADR-034 task #90 Step 2 (2026-05-21) + ADR-040 Phase A2b (2026-05-29) —
    /// roll back ONE sequence slot's linear-attention recurrent + conv state
    /// to `capture_states[..., accepted_idx, ...]` for that slot. Called on
    /// partial-reject in K=N spec-decode.
    ///
    /// **ADR-040 Phase A2b multi-seq lift (2026-05-29):**
    /// Pre-A2b the signature was `rollback_la_to(accepted_idx: u32)` and the
    /// rollback was inherently slot-blind (used `state_elems = whole_recurrent`
    /// which only coincides with per-seq elems at `n_seqs == 1`). The new
    /// signature takes an explicit `slot: SlotId` and rolls back ONLY that
    /// slot's per-seq slice. Other slots' recurrent + conv_state buffers are
    /// byte-untouched.
    ///
    /// **Layout — recurrent (col-major in shape vec)**:
    /// - shape `[D_k, D_v, n_v_heads, n_seqs]` (kv_cache.rs:2284-2289)
    /// - per-seq elems = `D_k * D_v * n_v_heads` (NOT `recurrent.element_count()`)
    /// - slot `s` offset in `recurrent` = `s * per_seq_elems`
    ///
    /// **Layout — capture (col-major in shape vec)**:
    /// - shape `[D_k, D_v, n_v_heads, n_tokens_max, n_seqs]` (kv_cache.rs:1479)
    /// - matches mlx-native `dispatch_gated_delta_net_decode_with_capture`:
    ///   `state_capture_seq_stride = n_tokens * (n_v_heads * D_v * D_k)`,
    ///   `state_capture_token_stride = n_v_heads * D_v * D_k`
    ///   (see `gated_delta_net_decode_capture.metal` lines 37-46)
    /// - slot `s`, token `t` offset = `s * (n_tokens_max * per_seq_elems) +
    ///   t * per_seq_elems`
    ///
    /// **Layout — conv_state (col-major in shape vec)**:
    /// - shape `[channels, K-1, n_seqs]` (kv_cache.rs:2268)
    /// - per-seq elems = `channels * (K-1)`
    /// - slot `s` offset = `s * per_seq_elems`
    ///
    /// **Layout — conv_capture (row-major in shape vec)**:
    /// - shape `[n_seqs, n_tokens_max, K-1, channels]` (kv_cache.rs:1493)
    /// - per-seq-token elems = `(K-1) * channels`
    /// - slot `s`, token `t` offset = `s * (n_tokens_max * per_seq_elems) +
    ///   t * per_seq_elems`
    ///
    /// Pre-conditions:
    /// - [`Self::ensure_la_capture`] was called for this cache.
    /// - `slot.0 < self.n_seqs`.
    /// - `accepted_idx < n_tokens_max` (the value passed to
    ///   `ensure_la_capture`).
    /// - The most-recent forward through these LA slots used
    ///   `dispatch_gated_delta_net_decode_with_capture` (i.e. wrote
    ///   per-position states into `capture_states`).
    ///
    /// Post-condition: every `linear_attn[i].recurrent` and
    /// `linear_attn[i].conv_state`'s slice for `slot` contains
    /// `capture_states[..., accepted_idx, slot]` and
    /// `conv_capture_states[slot, accepted_idx, ..., ...]` respectively.
    /// All other slots' bytes are unchanged. The `recurrent_scratch` /
    /// `conv_state_scratch` buffers are left untouched (overwritten by the
    /// next forward).
    ///
    /// # Errors
    /// - `slot.0 >= self.n_seqs` (bounds-first per iter-1.5 cfa-finding-F5)
    /// - Any slot lacks `capture_states` (caller bug: forgot
    ///   `ensure_la_capture`)
    /// - `accepted_idx >= n_tokens_max`
    pub fn rollback_la_to(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        accepted_idx: u32,
    ) -> Result<()> {
        // ADR-040 Phase A2b (2026-05-29) — bounds-first per iter-1.5
        // cfa-finding-F5 ordering. The legacy n_seqs > 1 guard at
        // kv_cache.rs:1567 is REPLACED with a real per-slot routing
        // path. Layout proof + slice math in the doc-comment above.
        if slot.0 >= self.n_seqs {
            return Err(anyhow!(
                "rollback_la_to: SlotOutOfRange slot={} max_slots={} \
                 (ADR-040 Phase A2b multi-seq lift; HybridKvCache constructed \
                 with n_seqs={})",
                slot.0,
                self.n_seqs,
                self.n_seqs,
            ));
        }
        let slot_idx = slot.0 as usize;
        let n_seqs = self.n_seqs as usize;
        // ADR-034 task #90 Step 4c (2026-05-21) — rollback also copies
        // the per-position conv state. Both buffers must be allocated
        // (ensure_la_capture allocates them in lockstep) for the
        // rollback to be consistent. Mismatch is a caller bug.
        for (i, slot_data) in self.linear_attn.iter_mut().enumerate() {
            let capture = slot_data.capture_states.as_ref().ok_or_else(|| {
                anyhow!(
                    "rollback_la_to: linear_attn[{}].capture_states is None — \
                     caller must call ensure_la_capture before rollback",
                    i
                )
            })?;
            // ADR-040 Phase A2b — per-seq math (NOT whole-buffer):
            //   recurrent_total_elems = per_seq_elems * n_seqs
            //   capture_total_elems   = per_seq_elems * n_tokens_max * n_seqs
            let recurrent_total = slot_data.recurrent.element_count();
            let capture_total = capture.element_count();
            if recurrent_total % n_seqs != 0 {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}].recurrent elements {} \
                     not divisible by n_seqs {} (layout invariant broken)",
                    i,
                    recurrent_total,
                    n_seqs
                ));
            }
            let per_seq_elems = recurrent_total / n_seqs;
            if per_seq_elems == 0 {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}] per_seq_elems == 0 \
                     (degenerate cfg)",
                    i
                ));
            }
            if capture_total % (per_seq_elems * n_seqs) != 0 {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}].capture_states elements {} \
                     not a multiple of per_seq_elems*n_seqs ({} * {} = {})",
                    i,
                    capture_total,
                    per_seq_elems,
                    n_seqs,
                    per_seq_elems * n_seqs
                ));
            }
            let n_tokens_max = capture_total / (per_seq_elems * n_seqs);
            if (accepted_idx as usize) >= n_tokens_max {
                return Err(anyhow!(
                    "rollback_la_to: accepted_idx {} >= n_tokens_max {} \
                     for linear_attn[{}]",
                    accepted_idx,
                    n_tokens_max,
                    i
                ));
            }
            // Copy capture[slot, accepted_idx, ...] → recurrent[slot, ...]
            // per the layout proof above.
            let capture_slice = capture
                .as_slice::<f32>()
                .map_err(|e| anyhow!("rollback_la_to: linear_attn[{}].capture as_slice: {e}", i))?;
            let capture_seq_stride = n_tokens_max * per_seq_elems;
            let src_offset =
                slot_idx * capture_seq_stride + (accepted_idx as usize) * per_seq_elems;
            let src_end = src_offset + per_seq_elems;
            if src_end > capture_slice.len() {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}] capture src range [{}..{}) \
                     exceeds buffer len {} (slot={} accepted_idx={} \
                     n_tokens_max={} per_seq_elems={})",
                    i,
                    src_offset,
                    src_end,
                    capture_slice.len(),
                    slot_idx,
                    accepted_idx,
                    n_tokens_max,
                    per_seq_elems
                ));
            }
            // Copy into a temporary so we can drop the immutable borrow
            // before taking &mut on recurrent.
            let src_owned: Vec<f32> = capture_slice[src_offset..src_end].to_vec();
            // ADR-040 M-QWEN: rollback must land in the slot's CURRENT
            // recurrent buffer (parity-aware), not the named field.
            let dst = slot_data
                .recurrent_current_mut(slot)
                .as_mut_slice::<f32>()
                .map_err(|e| {
                    anyhow!(
                        "rollback_la_to: linear_attn[{}].recurrent as_mut_slice: {e}",
                        i
                    )
                })?;
            let dst_offset = slot_idx * per_seq_elems;
            let dst_end = dst_offset + per_seq_elems;
            if dst_end > dst.len() {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}] recurrent dst range \
                     [{}..{}) exceeds buffer len {} (slot={} per_seq_elems={})",
                    i,
                    dst_offset,
                    dst_end,
                    dst.len(),
                    slot_idx,
                    per_seq_elems
                ));
            }
            dst[dst_offset..dst_end].copy_from_slice(&src_owned);

            // ADR-034 task #90 Step 4c (2026-05-21) + ADR-040 Phase A2b
            // (2026-05-29) — also roll back the conv1d ring buffer.
            //
            // Active conv_state layout: `[channels, K-1, n_seqs]` col-major
            // ⇒ slot `s` offset in conv_state = s * (channels * K-1).
            //
            // Capture conv layout: `[n_seqs, n_tokens_max, K-1, channels]`
            // row-major ⇒ slot `s`, token `t` offset in conv_capture =
            // s * (n_tokens_max * K-1 * channels) + t * (K-1 * channels).
            //
            // Per-token slice in capture is `[K-1, channels]` (channels
            // innermost). Active layout is `[channels, K-1]` (K-1
            // innermost), so we re-index per (k_i, c) — same as legacy.
            let conv_capture = slot_data.conv_capture_states.as_ref().ok_or_else(|| {
                anyhow!(
                    "rollback_la_to: linear_attn[{}].conv_capture_states is None — \
                     ensure_la_capture must allocate both buffers in lockstep",
                    i
                )
            })?;
            let conv_state_total = slot_data.conv_state.element_count();
            let conv_capture_total = conv_capture.element_count();
            if conv_state_total % n_seqs != 0 {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}].conv_state elements {} \
                     not divisible by n_seqs {} (layout invariant broken)",
                    i,
                    conv_state_total,
                    n_seqs
                ));
            }
            let conv_per_seq = conv_state_total / n_seqs;
            if conv_per_seq == 0 {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}] conv_per_seq == 0",
                    i
                ));
            }
            if conv_capture_total % (conv_per_seq * n_seqs) != 0 {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}].conv_capture_states elems {} \
                     not a multiple of conv_per_seq*n_seqs ({} * {} = {})",
                    i,
                    conv_capture_total,
                    conv_per_seq,
                    n_seqs,
                    conv_per_seq * n_seqs
                ));
            }
            let conv_n_tokens_max = conv_capture_total / (conv_per_seq * n_seqs);
            if (accepted_idx as usize) >= conv_n_tokens_max {
                return Err(anyhow!(
                    "rollback_la_to: accepted_idx {} >= conv n_tokens_max {} \
                     for linear_attn[{}]",
                    accepted_idx,
                    conv_n_tokens_max,
                    i
                ));
            }
            let conv_per_t = conv_per_seq;
            let conv_capture_slice = conv_capture.as_slice::<f32>().map_err(|e| {
                anyhow!(
                    "rollback_la_to: linear_attn[{}].conv_capture as_slice: {e}",
                    i
                )
            })?;
            let conv_capture_seq_stride = conv_n_tokens_max * conv_per_t;
            let conv_src_offset =
                slot_idx * conv_capture_seq_stride + (accepted_idx as usize) * conv_per_t;
            let conv_src_end = conv_src_offset + conv_per_t;
            if conv_src_end > conv_capture_slice.len() {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}] conv_capture src range \
                     [{}..{}) exceeds buffer len {} (slot={} accepted_idx={} \
                     conv_n_tokens_max={} conv_per_t={})",
                    i,
                    conv_src_offset,
                    conv_src_end,
                    conv_capture_slice.len(),
                    slot_idx,
                    accepted_idx,
                    conv_n_tokens_max,
                    conv_per_t
                ));
            }
            let conv_src_owned: Vec<f32> =
                conv_capture_slice[conv_src_offset..conv_src_end].to_vec();

            // Re-index from capture [K-1, channels] (channels innermost)
            // to active conv_state [channels, K-1] (K-1 innermost).
            let conv_shape = slot_data.conv_state.shape().to_vec();
            // conv_state shape per alloc_linear_attn_slot: [channels, K-1, n_seqs]
            if conv_shape.len() < 2 {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}].conv_state shape too short: {:?}",
                    i,
                    conv_shape
                ));
            }
            let channels = conv_shape[0];
            let k_minus1 = conv_shape[1];
            if channels * k_minus1 != conv_per_t {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}] conv_state channels*k_minus1 ({}*{}={}) != \
                     per_t ({})",
                    i,
                    channels,
                    k_minus1,
                    channels * k_minus1,
                    conv_per_t
                ));
            }
            // ADR-040 M-QWEN: parity-aware CURRENT conv buffer (see above).
            let conv_dst = slot_data
                .conv_current_mut(slot)
                .as_mut_slice::<f32>()
                .map_err(|e| {
                    anyhow!(
                        "rollback_la_to: linear_attn[{}].conv_state as_mut_slice: {e}",
                        i
                    )
                })?;
            // Slot offset into conv_dst: per the col-major layout above.
            let conv_dst_slot_offset = slot_idx * conv_per_seq;
            let conv_dst_slot_end = conv_dst_slot_offset + conv_per_seq;
            if conv_dst_slot_end > conv_dst.len() {
                return Err(anyhow!(
                    "rollback_la_to: linear_attn[{}] conv_state dst slot range \
                     [{}..{}) exceeds buffer len {} (slot={} conv_per_seq={})",
                    i,
                    conv_dst_slot_offset,
                    conv_dst_slot_end,
                    conv_dst.len(),
                    slot_idx,
                    conv_per_seq
                ));
            }
            let conv_dst_slot = &mut conv_dst[conv_dst_slot_offset..conv_dst_slot_end];
            // Capture layout: capture[i, c] at offset i*channels + c
            // (channels innermost). conv_state layout: state[c, i] at
            // offset c*k_minus1 + i (k_minus1 innermost). Re-index:
            for k_i in 0..k_minus1 {
                for c in 0..channels {
                    let src_idx = k_i * channels + c;
                    let dst_idx = c * k_minus1 + k_i;
                    conv_dst_slot[dst_idx] = conv_src_owned[src_idx];
                }
            }
        }
        Ok(())
    }

    /// Re-select the pre-forward DeltaNet ping-pong buffers for one slot.
    ///
    /// A batched decode writes every linear-attention layer into its inactive
    /// buffer and flips that slot exactly once after the forward completes.
    /// When a speculative transaction is rejected in full, the previous
    /// recurrent and convolution state is therefore still intact in the
    /// opposite buffer. Flipping once restores it without a CPU copy; the
    /// caller must then replay the valid target token normally.
    pub fn rewind_la_ping_pong_for_slot(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<()> {
        anyhow::ensure!(
            slot.0 < self.n_seqs,
            "rewind_la_ping_pong_for_slot: slot {} outside n_seqs={}",
            slot.0,
            self.n_seqs
        );
        for linear in &mut self.linear_attn {
            linear.swap_for_slot(slot);
        }
        Ok(())
    }

    /// **ADR-040 iter-C2d-cont-kernel iter-1 (2026-05-29)** — per-slot
    /// reset for the persistent multi-seq `HybridKvCache` worker hot path.
    ///
    /// Counterpart to [`Self::reset`] (which zeros **every** slot) that
    /// targets a SINGLE [`SlotId`]'s per-seq slice. Used by
    /// `engine_qwen35::generate_qwen35_once_slot_aware` to clear a slot's
    /// state at request entry + exit so the persistent cache is
    /// request-isolated within the slot — the next request to land on
    /// the same slot sees a zero-cursor full-attn cache + zero
    /// recurrent/conv state.
    ///
    /// **Layout proof** (mirror of A2b §6.1.23 `rollback_la_to`):
    /// - **full_attn.current_len**: `Vec<u32>` of length `n_seqs`. Per-slot
    ///   reset → set `current_len[slot_idx] = 0`; other slots untouched.
    /// - **full_attn.k / v (F32, when present)**: shape
    ///   `[n_seqs, n_kv_heads, max_seq_len, head_dim]` row-major.
    ///   Per-seq elems = `n_kv_heads * max_seq_len * head_dim`.
    ///   Slot `s` offset = `s * per_seq_elems`. **NOT zeroed**: the SDPA
    ///   read path masks against `current_len[slot_idx]`, so stale bytes
    ///   beyond the cursor are unreadable — matches the existing
    ///   per-request `alloc_kv_cache_for_request` path (the cursor, not tail
    ///   contents, defines readability).
    /// - **full_attn.tq (when present)**: same `[n_seqs, n_kv_heads,
    ///   max_seq_len, head_dim]` shape over k_packed / k_norms / v_packed /
    ///   v_norms. Same logic as F32 K/V — NOT zeroed; cursor masks.
    /// - **mtp_slot (when present)**: same layout as full_attn; reset
    ///   `current_len[slot_idx] = 0`.
    /// - **linear_attn.conv_state**: shape `[conv_channels, K-1, n_seqs]`
    ///   col-major. Per-seq elems = `conv_channels * (K-1)`. Slot `s`
    ///   offset = `s * per_seq_elems`. **MUST be zeroed**: the DeltaNet
    ///   conv1d kernel reads the ring buffer unconditionally (no cursor
    ///   mask), so stale bytes WOULD corrupt the next request.
    /// - **linear_attn.conv_state_scratch**: ping-pong scratch.
    ///   ALSO zeroed (the kernel swaps active/scratch; a stale scratch
    ///   slot would flip into active and be read).
    /// - **linear_attn.recurrent**: shape `[D_k, D_v, n_v_heads, n_seqs]`
    ///   col-major. Per-seq elems = `D_k * D_v * n_v_heads`. Slot `s`
    ///   offset = `s * per_seq_elems`. **MUST be zeroed**: recurrent
    ///   state has no cursor; the next request's first delta-net step
    ///   would read stale state and corrupt the run.
    /// - **linear_attn.recurrent_scratch**: ping-pong scratch. ALSO
    ///   zeroed (same reason as conv_state_scratch).
    /// - **capture_states / conv_capture_states**: spec-decode-only;
    ///   NOT zeroed by this fn (the spec-decode runner explicitly
    ///   captures every step before reading, so stale bytes are
    ///   structurally unreachable).
    ///
    /// # Errors
    /// - `slot.0 >= self.n_seqs` (bounds-first per A2b iter-1.5
    ///   cfa-finding-F5 ordering).
    ///
    /// # Per-slot byte-equivalence pin
    ///
    /// At `slot = SlotId(0)` AND `n_seqs == 1` this is byte-equivalent
    /// to [`Self::reset`] (the for-loop iterates exactly one slot,
    /// zeros exactly the same bytes). H53 pins this in the test
    /// module via element-count + slice-offset assertions.
    pub fn reset_for_slot(&mut self, slot: crate::serve::multi_seq_kv::SlotId) -> Result<()> {
        // Bounds-first per A2b §6.1.23 iter-1.5 cfa-finding-F5.
        if slot.0 >= self.n_seqs {
            return Err(anyhow!(
                "reset_for_slot: SlotOutOfRange slot={} max_slots={} \
                 (ADR-040 iter-C2d-cont-kernel iter-1 multi-seq lift; \
                 HybridKvCache constructed with n_seqs={})",
                slot.0,
                self.n_seqs,
                self.n_seqs,
            ));
        }
        let slot_idx = slot.0 as usize;
        let n_seqs = self.n_seqs as usize;

        // 1. full_attn slots — reset per-slot current_len cursor.
        for fa in self.full_attn.iter_mut() {
            if let Some(c) = fa.current_len.get_mut(slot_idx) {
                *c = 0;
            }
        }
        // 2. mtp_slot (optional) — reset per-slot current_len cursor.
        if let Some(fa) = self.mtp_slot.as_mut() {
            if let Some(c) = fa.current_len.get_mut(slot_idx) {
                *c = 0;
            }
        }
        // 3. linear_attn slots — zero per-slot conv_state +
        // conv_state_scratch + recurrent + recurrent_scratch slices.
        // Capture buffers (capture_states / conv_capture_states) are
        // spec-decode-only and explicitly overwritten by the capture
        // dispatch on every step; not zeroed here.
        for (i, la) in self.linear_attn.iter_mut().enumerate() {
            // conv_state — layout [conv_channels, K-1, n_seqs] col-major
            //                     → per_seq_elems = conv_state.element_count() / n_seqs.
            let total_conv = la.conv_state.element_count();
            if total_conv % n_seqs != 0 {
                return Err(anyhow!(
                    "reset_for_slot: linear_attn[{}].conv_state elements {} \
                     not divisible by n_seqs {} (layout invariant broken)",
                    i,
                    total_conv,
                    n_seqs
                ));
            }
            let per_seq_conv = total_conv / n_seqs;
            // recurrent — layout [D_k, D_v, n_v_heads, n_seqs] col-major.
            let total_rec = la.recurrent.element_count();
            if total_rec % n_seqs != 0 {
                return Err(anyhow!(
                    "reset_for_slot: linear_attn[{}].recurrent elements {} \
                     not divisible by n_seqs {} (layout invariant broken)",
                    i,
                    total_rec,
                    n_seqs
                ));
            }
            let per_seq_rec = total_rec / n_seqs;
            // Pair scratch buffers — same shapes by construction.
            let total_conv_scratch = la.conv_state_scratch.element_count();
            let total_rec_scratch = la.recurrent_scratch.element_count();
            if total_conv_scratch != total_conv {
                return Err(anyhow!(
                    "reset_for_slot: linear_attn[{}] conv_state_scratch elements \
                     {} != conv_state elements {} (ping-pong shape broken)",
                    i,
                    total_conv_scratch,
                    total_conv
                ));
            }
            if total_rec_scratch != total_rec {
                return Err(anyhow!(
                    "reset_for_slot: linear_attn[{}] recurrent_scratch elements \
                     {} != recurrent elements {} (ping-pong shape broken)",
                    i,
                    total_rec_scratch,
                    total_rec
                ));
            }
            // Zero per-slot slice in each of the 4 buffers.
            {
                let s = la.conv_state.as_mut_slice::<f32>().map_err(|e| {
                    anyhow!(
                        "reset_for_slot: linear_attn[{}].conv_state as_mut_slice: {e}",
                        i
                    )
                })?;
                let start = slot_idx * per_seq_conv;
                let end = start + per_seq_conv;
                for v in &mut s[start..end] {
                    *v = 0.0;
                }
            }
            {
                let s = la.conv_state_scratch.as_mut_slice::<f32>().map_err(|e| {
                    anyhow!(
                        "reset_for_slot: linear_attn[{}].conv_state_scratch as_mut_slice: {e}",
                        i
                    )
                })?;
                let start = slot_idx * per_seq_conv;
                let end = start + per_seq_conv;
                for v in &mut s[start..end] {
                    *v = 0.0;
                }
            }
            {
                let s = la.recurrent.as_mut_slice::<f32>().map_err(|e| {
                    anyhow!(
                        "reset_for_slot: linear_attn[{}].recurrent as_mut_slice: {e}",
                        i
                    )
                })?;
                let start = slot_idx * per_seq_rec;
                let end = start + per_seq_rec;
                for v in &mut s[start..end] {
                    *v = 0.0;
                }
            }
            {
                let s = la.recurrent_scratch.as_mut_slice::<f32>().map_err(|e| {
                    anyhow!(
                        "reset_for_slot: linear_attn[{}].recurrent_scratch as_mut_slice: {e}",
                        i
                    )
                })?;
                let start = slot_idx * per_seq_rec;
                let end = start + per_seq_rec;
                for v in &mut s[start..end] {
                    *v = 0.0;
                }
            }
            // ADR-040 M-QWEN: both ping-pong buffers are zeroed for this
            // slot → parity back to canonical (named fields = current).
            la.pp_flipped[slot_idx] = false;
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        for slot in self.full_attn.iter_mut() {
            for c in slot.current_len.iter_mut() {
                *c = 0;
            }
        }
        if let Some(slot) = self.mtp_slot.as_mut() {
            for c in slot.current_len.iter_mut() {
                *c = 0;
            }
        }
        for slot in self.linear_attn.iter_mut() {
            // Zero f32 buffers in place. Safe because f32 all-zero bit pattern
            // is a valid 0.0.
            if let Ok(s) = slot.conv_state.as_mut_slice::<f32>() {
                for v in s.iter_mut() {
                    *v = 0.0;
                }
            }
            if let Ok(s) = slot.conv_state_scratch.as_mut_slice::<f32>() {
                for v in s.iter_mut() {
                    *v = 0.0;
                }
            }
            if let Ok(s) = slot.recurrent.as_mut_slice::<f32>() {
                for v in s.iter_mut() {
                    *v = 0.0;
                }
            }
            if let Ok(s) = slot.recurrent_scratch.as_mut_slice::<f32>() {
                for v in s.iter_mut() {
                    *v = 0.0;
                }
            }
            // Reset ping-pong ownership after zeroing semantic state.
            for f in slot.pp_flipped.iter_mut() {
                *f = false;
            }
        }
    }

    /// Take a cursor-bounded deep-copy snapshot of the live serial cache.
    ///
    /// Wedge-3 / ADR-005 iter-216 Phase B.  The snapshot owns *fresh*
    /// `MlxBuffer` allocations whose contents byte-equal the corresponding
    /// buffers at snapshot time.  Used by `HybridPromptCache` to save
    /// post-prefill cache state and replay it for the next equivalent
    /// prompt, mirroring Gemma's `PromptCache` shape but with the hybrid
    /// (full-attn K/V + DeltaNet conv-state + recurrent state) surface.
    ///
    /// # Why deep-copy and NOT Arc::clone
    ///
    /// `MlxBuffer`'s underlying allocation is an `Arc<MetalBuffer>` — an
    /// `Arc::clone` would alias the buffer and a subsequent decode call
    /// (which writes into the cache through `forward_gpu`) would mutate
    /// the snapshot in lock-step with the live cache, defeating the
    /// purpose of caching pre-decode state.  Deep-copy via
    /// `device.alloc_buffer` + byte-level memcpy detaches the snapshot
    /// from the live cache so the snapshot is stable across any number of
    /// subsequent forward passes.
    ///
    /// # Ping-pong note (DeltaNet)
    ///
    /// `LinearAttnStateSlot::conv_state` and `recurrent` are the *active*
    /// (read) buffers under the kernel's ping-pong contract.  After each
    /// decode step the caller swaps them with the corresponding scratch
    /// buffer.  The snapshot only captures the active buffers — the
    /// scratch contents at snapshot time are post-write garbage that the
    /// next forward pass overwrites unconditionally, so they carry no
    /// semantic state.  On restore, scratch is left untouched (the next
    /// forward will write into it then swap; the swap exchange is
    /// purely a pointer operation, no copy).
    ///
    /// # Errors
    ///
    /// Propagates from any `MlxDevice::alloc_buffer` call (zero-byte
    /// alloc, OOM) and from `MlxBuffer::as_slice<u8>` / `as_mut_slice<u8>`
    /// (impossible in correct operation: every snapshot buffer is sized
    /// to its source's byte length).
    pub fn snapshot(&self, device: &MlxDevice) -> Result<HybridKvCacheSnapshot> {
        anyhow::ensure!(
            self.n_seqs == 1,
            "snapshot: full-cache snapshots are unsafe for overwrite-backed multi-sequence tails; use snapshot_prefix for an explicit live prefix"
        );
        let live_tokens =
            self.sequence_len_for_slot(crate::serve::multi_seq_kv::SlotId(0))? as usize;
        anyhow::ensure!(
            live_tokens > 0,
            "snapshot: cache has no cursor-visible sequence bytes"
        );
        self.snapshot_inner(device, Some(live_tokens), None)
    }

    /// Take an LCP snapshot whose full-attention/MTP sequence buffers own
    /// exactly `n_tokens` positions. DeltaNet state remains a full copy
    /// because it is fixed-size recurrent state rather than a sequence-axis
    /// cache. The compact snapshot is restored with [`Self::restore_partial`].
    pub fn snapshot_prefix(
        &self,
        device: &MlxDevice,
        n_tokens: usize,
    ) -> Result<HybridKvCacheSnapshot> {
        anyhow::ensure!(n_tokens > 0, "snapshot_prefix: n_tokens must be > 0");
        self.snapshot_inner(device, Some(n_tokens), None)
    }

    /// Take a compact LCP snapshot after a captured multi-token forward while
    /// preserving the live cache at the end of that forward. Full-attention
    /// and MTP buffers are sliced at `n_tokens`; DeltaNet recurrent and conv
    /// state are read from the per-position capture buffers at
    /// `capture_index` rather than from the live end-of-forward state.
    ///
    /// This lets agentic serving process a short changed suffix in one GPU
    /// forward and still retain the stable boundary immediately before a
    /// generation-only chat-template tail. The ordinary alternative requires
    /// two forwards (stable prefix, then template tail), whose fixed dispatch
    /// cost dominates short follow-up turns.
    ///
    /// The current LCP serving path is serial (`n_seqs == 1`). Slot-aware
    /// scheduling does not expose LCP resume, so this method rejects a
    /// multi-sequence cache rather than inventing cross-slot capture semantics.
    pub fn snapshot_prefix_from_capture(
        &self,
        device: &MlxDevice,
        n_tokens: usize,
        capture_index: usize,
    ) -> Result<HybridKvCacheSnapshot> {
        anyhow::ensure!(
            self.n_seqs == 1,
            "snapshot_prefix_from_capture: n_seqs={} != 1",
            self.n_seqs
        );
        let mut snapshot = self.snapshot_inner(device, Some(n_tokens), Some(capture_index))?;

        for lengths in &mut snapshot.full_attn_current_len {
            for length in lengths {
                *length = n_tokens as u32;
            }
        }
        if let Some(mtp) = snapshot.mtp.as_mut() {
            for length in &mut mtp.current_len {
                *length = n_tokens as u32;
            }
        }
        Ok(snapshot)
    }

    fn snapshot_inner(
        &self,
        device: &MlxDevice,
        prefix_tokens: Option<usize>,
        linear_capture_index: Option<usize>,
    ) -> Result<HybridKvCacheSnapshot> {
        let mut full_attn_k = Vec::with_capacity(self.full_attn.len());
        let mut full_attn_v = Vec::with_capacity(self.full_attn.len());
        let mut full_attn_current_len = Vec::with_capacity(self.full_attn.len());
        // ADR-027 Phase B iter-35 (sub-iter 23d-α): per-slot TQ snapshot
        // mirrors slot.tq state so iter-34's TQ-only F32-drop survives
        // LCP-resume (snapshot → restore would otherwise leave TQ
        // cursor-visible rows unwritten in the new request's cache → garbage
        // decode).
        let mut full_attn_tq = Vec::with_capacity(self.full_attn.len());
        for slot in &self.full_attn {
            // ADR-027 sub-sub-iter 23c-α: slot.k/v are Optional. None
            // marks iter-30 TQ-only state (no F32 backing); snapshot
            // pushes None to mirror. iter-34 makes None the production
            // norm under tq_kv_active=true.
            full_attn_k.push(match slot.k.as_ref() {
                Some(buf) => Some(
                    deep_copy_snapshot_sequence_buffer(device, buf, prefix_tokens, "full_attn.k")
                        .context("snapshot full_attn.k")?,
                ),
                None => None,
            });
            full_attn_v.push(match slot.v.as_ref() {
                Some(buf) => Some(
                    deep_copy_snapshot_sequence_buffer(device, buf, prefix_tokens, "full_attn.v")
                        .context("snapshot full_attn.v")?,
                ),
                None => None,
            });
            // iter-35: capture slot.tq when present (deep-copy each of
            // the 4 TQ buffers so the snapshot is detached from the
            // live cache and stable across subsequent decode writes).
            full_attn_tq.push(match slot.tq.as_ref() {
                Some(tq) => Some(TqKvSnapshot {
                    k_packed: deep_copy_snapshot_sequence_buffer(
                        device,
                        &tq.k_packed,
                        prefix_tokens,
                        "full_attn.tq.k_packed",
                    )
                    .context("snapshot full_attn.tq.k_packed")?,
                    k_norms: deep_copy_snapshot_sequence_buffer(
                        device,
                        &tq.k_norms,
                        prefix_tokens,
                        "full_attn.tq.k_norms",
                    )
                    .context("snapshot full_attn.tq.k_norms")?,
                    v_packed: deep_copy_snapshot_sequence_buffer(
                        device,
                        &tq.v_packed,
                        prefix_tokens,
                        "full_attn.tq.v_packed",
                    )
                    .context("snapshot full_attn.tq.v_packed")?,
                    v_norms: deep_copy_snapshot_sequence_buffer(
                        device,
                        &tq.v_norms,
                        prefix_tokens,
                        "full_attn.tq.v_norms",
                    )
                    .context("snapshot full_attn.tq.v_norms")?,
                    norms_per_pos: tq.norms_per_pos,
                }),
                None => None,
            });
            full_attn_current_len.push(slot.current_len.clone());
        }
        let mtp = match &self.mtp_slot {
            Some(slot) => Some(MtpKvSnapshot {
                // iter-23c-α: same Optional bridge as full_attn above.
                k: match slot.k.as_ref() {
                    Some(buf) => Some(
                        deep_copy_snapshot_sequence_buffer(device, buf, prefix_tokens, "mtp.k")
                            .context("snapshot mtp.k")?,
                    ),
                    None => None,
                },
                v: match slot.v.as_ref() {
                    Some(buf) => Some(
                        deep_copy_snapshot_sequence_buffer(device, buf, prefix_tokens, "mtp.v")
                            .context("snapshot mtp.v")?,
                    ),
                    None => None,
                },
                current_len: slot.current_len.clone(),
                // iter-35: MTP slot's TQ snapshot (same shape as
                // full-attn slots when present).
                tq: match slot.tq.as_ref() {
                    Some(tq) => Some(TqKvSnapshot {
                        k_packed: deep_copy_snapshot_sequence_buffer(
                            device,
                            &tq.k_packed,
                            prefix_tokens,
                            "mtp.tq.k_packed",
                        )
                        .context("snapshot mtp.tq.k_packed")?,
                        k_norms: deep_copy_snapshot_sequence_buffer(
                            device,
                            &tq.k_norms,
                            prefix_tokens,
                            "mtp.tq.k_norms",
                        )
                        .context("snapshot mtp.tq.k_norms")?,
                        v_packed: deep_copy_snapshot_sequence_buffer(
                            device,
                            &tq.v_packed,
                            prefix_tokens,
                            "mtp.tq.v_packed",
                        )
                        .context("snapshot mtp.tq.v_packed")?,
                        v_norms: deep_copy_snapshot_sequence_buffer(
                            device,
                            &tq.v_norms,
                            prefix_tokens,
                            "mtp.tq.v_norms",
                        )
                        .context("snapshot mtp.tq.v_norms")?,
                        norms_per_pos: tq.norms_per_pos,
                    }),
                    None => None,
                },
            }),
            None => None,
        };
        let mut linear_conv = Vec::with_capacity(self.linear_attn.len());
        let mut linear_recurrent = Vec::with_capacity(self.linear_attn.len());
        // ADR-040 M-QWEN: snapshots are PARITY-CANONICAL — each slot's
        // region is taken from whichever physical buffer is CURRENT for
        // that slot (per-slot ping-pong parity), assembled into one
        // canonical buffer. Keeps the on-disk/codec format unchanged
        // (restore_from writes named fields + resets parity to false).
        // Both layouts are slot-major (slot region = total/n_seqs,
        // offset slot*per_seq — same math as rollback_la_to).
        let canonicalize = |named: &MlxBuffer,
                            scratch: &MlxBuffer,
                            pp: &[bool],
                            what: &str|
         -> Result<MlxBuffer> {
            let mut out =
                deep_copy_buffer(device, named).with_context(|| format!("snapshot {what}"))?;
            if pp.iter().any(|&f| f) {
                let s = scratch
                    .as_slice::<f32>()
                    .with_context(|| format!("snapshot {what} scratch as_slice"))?;
                let d = out
                    .as_mut_slice::<f32>()
                    .with_context(|| format!("snapshot {what} out as_mut_slice"))?;
                let n = pp.len();
                anyhow::ensure!(
                    n > 0 && d.len() % n == 0,
                    "snapshot {what}: elements {} not divisible by n_seqs {}",
                    d.len(),
                    n
                );
                let per = d.len() / n;
                for (i, &flipped) in pp.iter().enumerate() {
                    if flipped {
                        d[i * per..(i + 1) * per].copy_from_slice(&s[i * per..(i + 1) * per]);
                    }
                }
            }
            Ok(out)
        };
        for (layer_idx, slot) in self.linear_attn.iter().enumerate() {
            if let Some(capture_index) = linear_capture_index {
                let recurrent_capture = slot.capture_states.as_ref().ok_or_else(|| {
                    anyhow!(
                        "snapshot_prefix_from_capture: linear_attn[{layer_idx}] recurrent capture missing"
                    )
                })?;
                let recurrent_per_token = slot.recurrent.element_count();
                anyhow::ensure!(
                    recurrent_per_token > 0
                        && recurrent_capture.element_count() % recurrent_per_token == 0,
                    "snapshot_prefix_from_capture: linear_attn[{layer_idx}] recurrent capture shape mismatch"
                );
                let recurrent_capacity = recurrent_capture.element_count() / recurrent_per_token;
                anyhow::ensure!(
                    capture_index < recurrent_capacity,
                    "snapshot_prefix_from_capture: capture_index={capture_index} >= recurrent capacity={recurrent_capacity} at layer {layer_idx}"
                );
                let recurrent_src = recurrent_capture.as_slice::<u8>().with_context(|| {
                    format!("linear_attn[{layer_idx}] recurrent capture as_slice")
                })?;
                let recurrent_start = capture_index * recurrent_per_token;
                let recurrent_end = recurrent_start + recurrent_per_token;
                let recurrent_start_bytes = recurrent_start * std::mem::size_of::<f32>();
                let recurrent_end_bytes = recurrent_end * std::mem::size_of::<f32>();
                let mut recurrent_snapshot = device
                    .alloc_buffer(
                        slot.recurrent.byte_len(),
                        slot.recurrent.dtype(),
                        slot.recurrent.shape().to_vec(),
                    )
                    .with_context(|| {
                        format!("linear_attn[{layer_idx}] recurrent snapshot allocation")
                    })?;
                recurrent_snapshot
                    .as_mut_slice::<u8>()
                    .with_context(|| {
                        format!("linear_attn[{layer_idx}] recurrent snapshot as_mut_slice")
                    })?
                    .copy_from_slice(&recurrent_src[recurrent_start_bytes..recurrent_end_bytes]);

                let conv_capture = slot.conv_capture_states.as_ref().ok_or_else(|| {
                    anyhow!(
                        "snapshot_prefix_from_capture: linear_attn[{layer_idx}] conv capture missing"
                    )
                })?;
                let conv_shape = slot.conv_state.shape();
                anyhow::ensure!(
                    conv_shape.len() == 3 && conv_shape[2] == 1,
                    "snapshot_prefix_from_capture: linear_attn[{layer_idx}] conv shape {:?} is not [channels, K-1, 1]",
                    conv_shape
                );
                let channels = conv_shape[0];
                let k_minus_one = conv_shape[1];
                let conv_per_token = channels * k_minus_one;
                anyhow::ensure!(
                    conv_per_token > 0 && conv_capture.element_count() % conv_per_token == 0,
                    "snapshot_prefix_from_capture: linear_attn[{layer_idx}] conv capture shape mismatch"
                );
                let conv_capacity = conv_capture.element_count() / conv_per_token;
                anyhow::ensure!(
                    capture_index < conv_capacity,
                    "snapshot_prefix_from_capture: capture_index={capture_index} >= conv capacity={conv_capacity} at layer {layer_idx}"
                );
                let conv_src = conv_capture
                    .as_slice::<u8>()
                    .with_context(|| format!("linear_attn[{layer_idx}] conv capture as_slice"))?;
                let conv_start = capture_index * conv_per_token;
                let f32_bytes = std::mem::size_of::<f32>();
                let conv_start_bytes = conv_start * f32_bytes;
                let conv_src =
                    &conv_src[conv_start_bytes..conv_start_bytes + conv_per_token * f32_bytes];
                let mut conv_snapshot = device
                    .alloc_buffer(
                        slot.conv_state.byte_len(),
                        slot.conv_state.dtype(),
                        slot.conv_state.shape().to_vec(),
                    )
                    .with_context(|| {
                        format!("linear_attn[{layer_idx}] conv snapshot allocation")
                    })?;
                let conv_dst = conv_snapshot.as_mut_slice::<u8>().with_context(|| {
                    format!("linear_attn[{layer_idx}] conv snapshot as_mut_slice")
                })?;
                // Capture is [K-1, channels]; the active cache is
                // [channels, K-1]. Preserve every f32 payload bit.
                for k_idx in 0..k_minus_one {
                    for channel in 0..channels {
                        let src_byte = (k_idx * channels + channel) * f32_bytes;
                        let dst_byte = (channel * k_minus_one + k_idx) * f32_bytes;
                        conv_dst[dst_byte..dst_byte + f32_bytes]
                            .copy_from_slice(&conv_src[src_byte..src_byte + f32_bytes]);
                    }
                }
                linear_conv.push(conv_snapshot);
                linear_recurrent.push(recurrent_snapshot);
            } else {
                linear_conv.push(canonicalize(
                    &slot.conv_state,
                    &slot.conv_state_scratch,
                    &slot.pp_flipped,
                    "conv_state",
                )?);
                linear_recurrent.push(canonicalize(
                    &slot.recurrent,
                    &slot.recurrent_scratch,
                    &slot.pp_flipped,
                    "recurrent",
                )?);
            }
        }
        Ok(HybridKvCacheSnapshot {
            full_attn_k,
            full_attn_v,
            full_attn_current_len,
            full_attn_tq,
            mtp,
            linear_conv,
            linear_recurrent,
        })
    }

    /// Memcpy the snapshot's per-slot bytes back into this cache's owned
    /// buffers and restore per-seq write cursors.
    ///
    /// Wedge-3 / ADR-005 iter-216 Phase B.  Pairs with [`Self::snapshot`].
    /// The cache's existing `MlxBuffer` allocations are reused — only their
    /// contents are overwritten, so the cache shape (max_seq_len, n_seqs,
    /// per-layer-slot vectors) MUST match the snapshot's source cache.
    /// Mismatches surface as length-comparison errors.
    ///
    /// # Errors
    ///
    /// Returns Err when:
    /// - the snapshot's slot count doesn't match `self`'s
    /// - any per-slot byte length disagrees (would mean a different
    ///   `cfg`-shape cache — caller bug)
    /// - any `as_slice` / `as_mut_slice` call fails
    pub fn restore_from(&mut self, snapshot: &HybridKvCacheSnapshot) -> Result<()> {
        anyhow::ensure!(
            snapshot.full_attn_k.len() == self.full_attn.len(),
            "restore_from: full_attn slot count mismatch ({} snapshot vs {} cache)",
            snapshot.full_attn_k.len(),
            self.full_attn.len()
        );
        anyhow::ensure!(
            snapshot.linear_conv.len() == self.linear_attn.len(),
            "restore_from: linear_attn slot count mismatch ({} snapshot vs {} cache)",
            snapshot.linear_conv.len(),
            self.linear_attn.len()
        );
        // iter-35 (sub-iter 23d-α): full_attn_tq must align with full_attn_k
        // count (snapshot producer iter-35 always pushes one entry per slot).
        anyhow::ensure!(
            snapshot.full_attn_tq.len() == snapshot.full_attn_k.len(),
            "restore_from: snapshot full_attn_tq.len ({}) != full_attn_k.len ({})",
            snapshot.full_attn_tq.len(),
            snapshot.full_attn_k.len()
        );
        for (slot, (k_snap, (v_snap, (tq_snap, len_snap)))) in self.full_attn.iter_mut().zip(
            snapshot.full_attn_k.iter().zip(
                snapshot.full_attn_v.iter().zip(
                    snapshot
                        .full_attn_tq
                        .iter()
                        .zip(snapshot.full_attn_current_len.iter()),
                ),
            ),
        ) {
            // ADR-027 sub-sub-iter 23c-α: Optional full-attn K/V on
            // BOTH source (iter-23a-β) AND destination (iter-23c-α).
            // Restore is a no-op when either side is None — matches
            // iter-30 TQ-only mode where SDPA reads slot.tq directly
            // and F32 backing is absent on both sides.
            if let (Some(k_buf), Some(dst_k)) = (k_snap, slot.k.as_mut()) {
                copy_buffer_bytes(k_buf, dst_k).context("restore full_attn.k")?;
            }
            if let (Some(v_buf), Some(dst_v)) = (v_snap, slot.v.as_mut()) {
                copy_buffer_bytes(v_buf, dst_v).context("restore full_attn.v")?;
            }
            // iter-35 (sub-iter 23d-α): TQ restore. Mirrors F32 path's
            // (Some,Some) source/destination guard. When source has TQ
            // (iter-34 production case under tq_kv_active=true) AND
            // destination slot has TQ buffers, copy all 4 byte payloads.
            if let (Some(tq_src), Some(tq_dst)) = (tq_snap, slot.tq.as_mut()) {
                copy_buffer_bytes(&tq_src.k_packed, &mut tq_dst.k_packed)
                    .context("restore full_attn.tq.k_packed")?;
                copy_buffer_bytes(&tq_src.k_norms, &mut tq_dst.k_norms)
                    .context("restore full_attn.tq.k_norms")?;
                copy_buffer_bytes(&tq_src.v_packed, &mut tq_dst.v_packed)
                    .context("restore full_attn.tq.v_packed")?;
                copy_buffer_bytes(&tq_src.v_norms, &mut tq_dst.v_norms)
                    .context("restore full_attn.tq.v_norms")?;
            }
            anyhow::ensure!(
                len_snap.len() == slot.current_len.len(),
                "restore_from: full_attn current_len shape mismatch"
            );
            slot.current_len.copy_from_slice(len_snap);
        }
        match (&snapshot.mtp, self.mtp_slot.as_mut()) {
            (Some(snap), Some(slot)) => {
                // ADR-027 sub-sub-iter 23c-α: Optional MTP K/V on
                // BOTH source (iter-23a-α) AND destination
                // (iter-23c-α). Restore is a no-op when either side is
                // None — matches iter-30 TQ-only mode.
                if let (Some(snap_k), Some(dst_k)) = (&snap.k, slot.k.as_mut()) {
                    copy_buffer_bytes(snap_k, dst_k).context("restore mtp.k")?;
                }
                if let (Some(snap_v), Some(dst_v)) = (&snap.v, slot.v.as_mut()) {
                    copy_buffer_bytes(snap_v, dst_v).context("restore mtp.v")?;
                }
                // iter-35: MTP TQ restore.
                if let (Some(tq_src), Some(tq_dst)) = (&snap.tq, slot.tq.as_mut()) {
                    copy_buffer_bytes(&tq_src.k_packed, &mut tq_dst.k_packed)
                        .context("restore mtp.tq.k_packed")?;
                    copy_buffer_bytes(&tq_src.k_norms, &mut tq_dst.k_norms)
                        .context("restore mtp.tq.k_norms")?;
                    copy_buffer_bytes(&tq_src.v_packed, &mut tq_dst.v_packed)
                        .context("restore mtp.tq.v_packed")?;
                    copy_buffer_bytes(&tq_src.v_norms, &mut tq_dst.v_norms)
                        .context("restore mtp.tq.v_norms")?;
                }
                anyhow::ensure!(
                    snap.current_len.len() == slot.current_len.len(),
                    "restore_from: mtp current_len shape mismatch"
                );
                slot.current_len.copy_from_slice(&snap.current_len);
            }
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => {
                anyhow::bail!(
                    "restore_from: mtp_slot presence mismatch between snapshot and cache"
                );
            }
        }
        for (slot, (conv_snap, rec_snap)) in self.linear_attn.iter_mut().zip(
            snapshot
                .linear_conv
                .iter()
                .zip(snapshot.linear_recurrent.iter()),
        ) {
            copy_buffer_bytes(conv_snap, &mut slot.conv_state).context("restore conv_state")?;
            copy_buffer_bytes(rec_snap, &mut slot.recurrent).context("restore recurrent")?;
            // ADR-040 M-QWEN: snapshots are parity-canonical (current
            // state assembled into the named fields), so restoring makes
            // the named fields current for EVERY slot.
            for f in slot.pp_flipped.iter_mut() {
                *f = false;
            }
        }
        Ok(())
    }

    /// ADR-017 Phase E.a B.5 — partial-position restore for LCP resume
    /// across requests with DIFFERENT max_seq_len.
    ///
    /// `restore_from` requires byte-equal slot K/V buffer sizes (same
    /// max_seq_len at snapshot time and restore time).  For LCP partial-
    /// prefill resume, the snapshot's source request and the new request
    /// typically have DIFFERENT prompt lengths and therefore different
    /// per-request `max_seq_len` allocations (see
    /// `engine_qwen35.rs::alloc_kv_cache_for_request` — `max_seq =
    /// (prompt_len + max_tokens + 64).max(128)`).  Byte-copy fails.
    ///
    /// `restore_partial` instead copies, per full-attn head, only the
    /// first `n_tokens` positions of K and V — the slot positions that
    /// hold the cached prefix at snapshot time.  The destination
    /// `max_seq_len` may be larger; the unused tail [n_tokens..max_seq]
    /// is left zero-initialised (which the kernel never reads thanks
    /// to `kL`-aware tile bounds).  Sets `slot.current_len[0] =
    /// n_tokens` for each full-attn slot.
    ///
    /// TQ-active mode (ADR-027 sub-iter 23d-γ): when a slot carries TQ
    /// buffers on both sides (`slot.tq` Some in snapshot AND cache),
    /// the same first-`n_tokens`-per-head partial copy is applied to
    /// `k_packed` / `v_packed` (U8) and `k_norms` / `v_norms` (F32) so
    /// the TQ-only decode/resume chain sees the restored prefix state.
    ///
    /// DeltaNet recurrent + conv state buffers are NOT sized to
    /// `max_seq_len` (they're sized to model dimensions only) — those
    /// are byte-copied directly via `copy_buffer_bytes`, same as
    /// `restore_from`.
    ///
    /// MTP slot: same partial-position semantics as the regular
    /// full-attn slots when present.
    ///
    /// # Errors
    ///
    /// * Slot count mismatch (different model architecture).
    /// * `n_tokens` exceeds either source or destination per-head
    ///   capacity.
    /// * Per-head buffer size derivation fails (snapshot or destination
    ///   not in `[n_kv_heads, max_seq, head_dim]` shape).
    pub fn restore_partial(
        &mut self,
        snapshot: &HybridKvCacheSnapshot,
        n_tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            snapshot.full_attn_k.len() == self.full_attn.len(),
            "restore_partial: full_attn slot count mismatch ({} snapshot vs {} cache)",
            snapshot.full_attn_k.len(),
            self.full_attn.len()
        );
        anyhow::ensure!(
            snapshot.linear_conv.len() == self.linear_attn.len(),
            "restore_partial: linear_attn slot count mismatch ({} snapshot vs {} cache)",
            snapshot.linear_conv.len(),
            self.linear_attn.len()
        );

        // iter-35 (sub-iter 23d-α) guard mirrored from `restore_from`:
        // snapshot producers always push one TQ entry per full-attn slot.
        anyhow::ensure!(
            snapshot.full_attn_tq.len() == snapshot.full_attn_k.len(),
            "restore_partial: snapshot full_attn_tq.len ({}) != full_attn_k.len ({})",
            snapshot.full_attn_tq.len(),
            snapshot.full_attn_k.len()
        );

        // Per-slot partial-position copy.  Each slot has shape
        // [n_kv_heads, max_seq_len, head_dim] with F32 elements.  Copy
        // first n_tokens positions per head.
        for (slot, (k_snap, (v_snap, tq_snap))) in self.full_attn.iter_mut().zip(
            snapshot.full_attn_k.iter().zip(
                snapshot
                    .full_attn_v
                    .iter()
                    .zip(snapshot.full_attn_tq.iter()),
            ),
        ) {
            // ADR-027 sub-sub-iter 23c-α: Optional full-attn K/V on
            // BOTH source AND destination. Restore is a no-op when
            // either side is None.
            if let (Some(k_buf), Some(dst_k)) = (k_snap, slot.k.as_mut()) {
                partial_copy_slot(k_buf, dst_k, n_tokens, "full_attn.k")?;
            }
            if let (Some(v_buf), Some(dst_v)) = (v_snap, slot.v.as_mut()) {
                partial_copy_slot(v_buf, dst_v, n_tokens, "full_attn.v")?;
            }
            // ADR-027 sub-iter 23d-γ (2026-08-03): TQ partial restore.
            // Without this branch an LCP resume under production
            // `tq_kv_active=true` left every TQ buffer ZERO-INITIALIZED
            // while `current_len` advanced past the boundary — the
            // resumed request attended over zeroed K/V for the entire
            // cached prefix (silent coherence corruption; surfaced as
            // the serve-side disk-persist panic investigation).
            // `partial_copy_slot` is dtype-size generic (raw u8 slices),
            // so the U8 packed buffers and the F32 norms buffers both
            // route through it; the norms buffer's innermost axis is
            // `norms_per_pos`, which the per-head stride math handles
            // identically.
            if let (Some(tq_src), Some(tq_dst)) = (tq_snap, slot.tq.as_mut()) {
                partial_copy_slot(
                    &tq_src.k_packed,
                    &mut tq_dst.k_packed,
                    n_tokens,
                    "full_attn.tq.k_packed",
                )?;
                partial_copy_slot(
                    &tq_src.k_norms,
                    &mut tq_dst.k_norms,
                    n_tokens,
                    "full_attn.tq.k_norms",
                )?;
                partial_copy_slot(
                    &tq_src.v_packed,
                    &mut tq_dst.v_packed,
                    n_tokens,
                    "full_attn.tq.v_packed",
                )?;
                partial_copy_slot(
                    &tq_src.v_norms,
                    &mut tq_dst.v_norms,
                    n_tokens,
                    "full_attn.tq.v_norms",
                )?;
            }
            // current_len[0] = n_tokens (the LCP boundary the snapshot
            // was taken at; subsequent prefill chunks will write at
            // positions [n_tokens..]).
            anyhow::ensure!(
                !slot.current_len.is_empty(),
                "restore_partial: slot.current_len is empty"
            );
            slot.current_len[0] = n_tokens as u32;
            for v in slot.current_len.iter_mut().skip(1) {
                *v = n_tokens as u32;
            }
        }

        // MTP slot (when present).
        match (&snapshot.mtp, self.mtp_slot.as_mut()) {
            (Some(snap), Some(slot)) => {
                // ADR-027 sub-sub-iter 23c-α: Optional MTP K/V on BOTH
                // source AND destination.
                if let (Some(snap_k), Some(dst_k)) = (&snap.k, slot.k.as_mut()) {
                    partial_copy_slot(snap_k, dst_k, n_tokens, "mtp.k")?;
                }
                if let (Some(snap_v), Some(dst_v)) = (&snap.v, slot.v.as_mut()) {
                    partial_copy_slot(snap_v, dst_v, n_tokens, "mtp.v")?;
                }
                // ADR-027 sub-iter 23d-γ: MTP TQ partial restore (same
                // gap + same fix as the full-attn slots above).
                if let (Some(tq_src), Some(tq_dst)) = (&snap.tq, slot.tq.as_mut()) {
                    partial_copy_slot(
                        &tq_src.k_packed,
                        &mut tq_dst.k_packed,
                        n_tokens,
                        "mtp.tq.k_packed",
                    )?;
                    partial_copy_slot(
                        &tq_src.k_norms,
                        &mut tq_dst.k_norms,
                        n_tokens,
                        "mtp.tq.k_norms",
                    )?;
                    partial_copy_slot(
                        &tq_src.v_packed,
                        &mut tq_dst.v_packed,
                        n_tokens,
                        "mtp.tq.v_packed",
                    )?;
                    partial_copy_slot(
                        &tq_src.v_norms,
                        &mut tq_dst.v_norms,
                        n_tokens,
                        "mtp.tq.v_norms",
                    )?;
                }
                anyhow::ensure!(
                    !slot.current_len.is_empty(),
                    "restore_partial: mtp slot.current_len is empty"
                );
                slot.current_len[0] = n_tokens as u32;
                for v in slot.current_len.iter_mut().skip(1) {
                    *v = n_tokens as u32;
                }
            }
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => {
                anyhow::bail!(
                    "restore_partial: mtp_slot presence mismatch between snapshot and cache"
                );
            }
        }

        // DeltaNet conv + recurrent state are NOT sized to max_seq_len
        // (they're per-head/per-model dimensions only) — byte-copy
        // directly.  Snapshot taken at any prompt position has correct
        // recurrent state at THAT position; we want exactly that state.
        for (slot, (conv_snap, rec_snap)) in self.linear_attn.iter_mut().zip(
            snapshot
                .linear_conv
                .iter()
                .zip(snapshot.linear_recurrent.iter()),
        ) {
            copy_buffer_bytes(conv_snap, &mut slot.conv_state)
                .context("restore_partial conv_state")?;
            copy_buffer_bytes(rec_snap, &mut slot.recurrent)
                .context("restore_partial recurrent")?;
            // ADR-040 M-QWEN: snapshot is parity-canonical → named fields
            // become current for every slot after restore.
            for f in slot.pp_flipped.iter_mut() {
                *f = false;
            }
        }
        Ok(())
    }

    /// Total allocated bytes across all slots (for memory accounting / logs).
    pub fn total_bytes(&self) -> usize {
        let mut n = 0usize;
        for s in &self.full_attn {
            // iter-29 (sub-sub-iter 23c-α): Optional K/V — 0 bytes when
            // None (iter-30 TQ-only mode); element_count×4 when Some.
            if let Some(b) = s.k.as_ref() {
                n += b.element_count() * 4;
            }
            if let Some(b) = s.v.as_ref() {
                n += b.element_count() * 4;
            }
        }
        if let Some(s) = &self.mtp_slot {
            if let Some(b) = s.k.as_ref() {
                n += b.element_count() * 4;
            }
            if let Some(b) = s.v.as_ref() {
                n += b.element_count() * 4;
            }
        }
        for s in &self.linear_attn {
            n += s.conv_state.element_count() * 4
                + s.conv_state_scratch.element_count() * 4
                + s.recurrent.element_count() * 4
                + s.recurrent_scratch.element_count() * 4;
            // ADR-034 task #90 Step 2 (2026-05-21) — count capture
            // buffer when allocated (None in non-spec mode = 0 bytes).
            if let Some(buf) = s.capture_states.as_ref() {
                n += buf.element_count() * 4;
            }
            // ADR-034 task #90 Step 4c (2026-05-21) — count conv capture
            // companion buffer when allocated.
            if let Some(buf) = s.conv_capture_states.as_ref() {
                n += buf.element_count() * 4;
            }
        }
        n
    }
}

/// DeltaNet conv1d input channel count = Q + K + V total per-token width:
///
///   conv_channels = 2 * (n_k_heads * D_k) + n_v_heads * D_v
///
/// For Qwen3.5-MoE: 2*16*128 + 32*128 = 8192.
/// For Qwen3.5 dense: 2*16*128 + 48*128 = 10240.
pub fn conv_channels_for(cfg: &Qwen35Config) -> u32 {
    2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
        + cfg.linear_num_value_heads * cfg.linear_value_head_dim
}

fn alloc_full_attn_slot(
    cfg: &Qwen35Config,
    device: &MlxDevice,
    max_seq_len: u32,
    n_seqs: u32,
    tq_kv_active: bool,
) -> Result<FullAttnKvSlot> {
    // ADR-027 Phase B iter-34 (sub-sub-iter 23c-β.5) — the LOAD-BEARING
    // memory-savings switch.
    //
    // When tq_kv_active=true, the slot's F32 K/V backing is dropped
    // (k=None, v=None). The SDPA read path for production qwen35
    // (head_dim=256) is fully covered by the TQ-only chain:
    //   * Decode: dispatch_decode_sdpa_with_optional_tq (iter-15) reads
    //     slot.tq directly; F32 fallback is unreachable when head_dim
    //     ∈ {256, 512} AND slot.tq.is_some().
    //   * Prefill RESUME: apply_flash_attn_prefill_seq_major_resume_via_tq_cache
    //     (iter-33) dequants slot.tq → temp F32 (unrotated, head-major),
    //     dispatches the same dense resume kernel against the temp
    //     buffers. iter-33 NRMSE 0.003 vs F32 baseline confirms parity.
    //   * Prefill FRESH (cur_len=0, fast path): reads k_seq_major directly
    //     (the just-computed chunk K/V), not from the cache; cache-write
    //     side is gated below to skip the F32 write (write_kv_with_optional_tq_encode).
    //
    // Memory savings at qwen36 35B-A3B-APEX 8K shape: 33.55 MB F32 K+V
    // per slot dropped → 8.52 MB TQ packed+norms only = 3.94×.
    // Regression-pin: full_attn_bytes_breakdown_tq_on_drops_f32_at_qwen36_*.
    if tq_kv_active {
        return Ok(FullAttnKvSlot {
            k: None,
            v: None,
            current_len: vec![0; n_seqs as usize],
            // tq is populated by HybridKvCache::new_with_options' subsequent
            // alloc_tq_full_attn_buffers call (separate from this fn's
            // F32 alloc; see new_with_options for the wiring).
            tq: None,
        });
    }

    // Conventional F32 control path (`HF2Q_TQ_KV=0`).
    // Layout: [n_seqs, n_kv_heads, max_seq_len, head_dim] — matches SDPA kernel's
    // expected K/V layout: [batch, n_kv_heads, kv_seq_len, head_dim] (head_dim innermost).
    // kv_capacity = max_seq_len; kv_seq_len = current_len at forward time.
    let elems = (n_seqs as usize)
        * (cfg.num_key_value_heads as usize)
        * (max_seq_len as usize)
        * (cfg.head_dim as usize);
    let bytes = elems * 4;
    let shape = vec![
        n_seqs as usize,
        cfg.num_key_value_heads as usize,
        max_seq_len as usize,
        cfg.head_dim as usize,
    ];
    // SAFETY: full-attention reads are bounded by `current_len`, which is
    // advanced only after the producer writes the corresponding positions.
    let k = unsafe { device.alloc_buffer_for_overwrite(bytes, DType::F32, shape.clone()) }
        .map_err(|e| anyhow!("alloc full-attn K: {e}"))?;
    // SAFETY: same cursor-before-read invariant as K.
    let v = unsafe { device.alloc_buffer_for_overwrite(bytes, DType::F32, shape) }
        .map_err(|e| anyhow!("alloc full-attn V: {e}"))?;

    Ok(FullAttnKvSlot {
        k: Some(k),
        v: Some(v),
        current_len: vec![0; n_seqs as usize],
        // ADR-027 Phase B iter-8: tq is None on the legacy F32 path.
        // Set by `HybridKvCache::new_with_options` when tq_kv_active=true.
        tq: None,
    })
}

fn alloc_linear_attn_slot(
    cfg: &Qwen35Config,
    device: &MlxDevice,
    conv_channels: u32,
    k_minus1: u32,
    n_seqs: u32,
) -> Result<LinearAttnStateSlot> {
    // Conv state ping-pong: [conv_channels, K-1, n_seqs] — kernel native layout.
    // The ssm_conv kernel expects state[i, c, s] at offset
    // s*(K-1)*channels + c*(K-1) + i, which corresponds to column-major
    // [channels, K-1] per sequence — i.e. channels-major with K-1 stride-1.
    // Storing in this layout avoids per-token CPU transpose + upload/download.
    let conv_elems = (conv_channels as usize) * (k_minus1 as usize) * (n_seqs as usize);
    let conv_shape = vec![conv_channels as usize, k_minus1 as usize, n_seqs as usize];
    let conv_state = device
        .alloc_buffer(conv_elems * 4, DType::F32, conv_shape.clone())
        .map_err(|e| anyhow!("alloc conv_state: {e}"))?;
    // Scratch buffer for ping-pong: ssm_conv writes new state here; caller
    // swaps conv_state ↔ conv_state_scratch after each decode step.
    let conv_state_scratch = device
        .alloc_buffer(conv_elems * 4, DType::F32, conv_shape)
        .map_err(|e| anyhow!("alloc conv_state_scratch: {e}"))?;

    // Recurrent state: [D_k, D_v, num_v_heads, n_seqs] — d_k innermost (matches
    // mlx-native's gated_delta_net kernel layout).
    let rec_elems = (cfg.linear_key_head_dim as usize)
        * (cfg.linear_value_head_dim as usize)
        * (cfg.linear_num_value_heads as usize)
        * (n_seqs as usize);
    let rec_shape = vec![
        cfg.linear_key_head_dim as usize,
        cfg.linear_value_head_dim as usize,
        cfg.linear_num_value_heads as usize,
        n_seqs as usize,
    ];
    let recurrent = device
        .alloc_buffer(rec_elems * 4, DType::F32, rec_shape.clone())
        .map_err(|e| anyhow!("alloc recurrent: {e}"))?;
    // Scratch buffer for ping-pong: same shape, zero-initialized.
    // GDN kernel writes here; after each decode step the caller swaps
    // `recurrent` and `recurrent_scratch`, making the new output the
    // new current state without any CPU copy.
    let recurrent_scratch = device
        .alloc_buffer(rec_elems * 4, DType::F32, rec_shape)
        .map_err(|e| anyhow!("alloc recurrent_scratch: {e}"))?;

    Ok(LinearAttnStateSlot {
        conv_state,
        conv_state_scratch,
        recurrent,
        recurrent_scratch,
        // ADR-034 task #90 Step 2 (2026-05-21) — recurrent capture
        // buffer. Step 4c adds the companion conv_capture_states field.
        // Both lazy-allocate via `HybridKvCache::ensure_la_capture`.
        capture_states: None,
        conv_capture_states: None,
        // ADR-040 M-QWEN — per-slot ping-pong parity, all canonical
        // (false = named fields are current) at alloc.
        pp_flipped: vec![false; n_seqs as usize],
    })
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-027 Phase B iter-7 — TQ-active full-attn KV buffer infra (additive)
// ──────────────────────────────────────────────────────────────────────────
//
// Qwen full-attention TQ storage extends the native three-axis kernel layout
// with an outer sequence/agent axis. The production allocator, cache writer,
// decode attention, prefill resume, and persistence paths all consume this
// representation directly; F32 buffers exist only in the explicit opt-out
// regime.

/// ADR-027 Phase B iter-7 — TQ-active K/V buffer set for one full-attn
/// slot (qwen35). Holds Hadamard-rotated 8-bit-quantized K/V indices and
/// per-position F32 norms.
///
/// Shape convention matches the qwen35 F32 cache layout (4D with `n_seqs`
/// as the outer axis), differing from Gemma's HbKvBuffers shape which is
/// 3D (no batch axis). The mlx-native `flash_attn_vec_tq_hb` kernel reads
/// the inner three axes `[n_kv_heads, max_seq_len, head_dim]` per
/// sequence; the n_seqs outer dimension is consumed at the call site.
///
/// Constructed by [`alloc_tq_full_attn_buffers`] and installed by the
/// `HybridKvCache` allocator when TQ mode is active.
pub struct TqFullAttnKvBuffers {
    /// Byte-packed K indices `[n_seqs, n_kv_heads, max_seq_len, head_dim]`
    /// U8.  One byte per element (8-bit Lloyd-Max codebook index).
    pub k_packed: MlxBuffer,
    /// K per-(seq, head, position) F32 norms.  Shape:
    /// `[n_seqs, n_kv_heads, max_seq_len, norms_per_pos]` F32.
    /// At head_dim=256 (qwen35 / qwen35moe) `norms_per_pos = 1`;
    /// at head_dim=512 it would be 2 (matches Gemma's formula).
    pub k_norms: MlxBuffer,
    /// Byte-packed V indices, same shape as `k_packed`.
    pub v_packed: MlxBuffer,
    /// V per-(seq, head, position) F32 norms, same shape as `k_norms`.
    pub v_norms: MlxBuffer,
    /// Number of F32 norms per position (1 for head_dim=256;
    /// 2 for head_dim=512).  Cached so SDPA dispatch (iter-8) doesn't
    /// recompute from `head_dim`.
    pub norms_per_pos: u32,
    /// Physical per-slot segment map. Logical context remains on the owning
    /// `HybridKvCache`; this map records only rows allocated in this arena.
    layout: TqArenaLayout,
}

struct TqFullAttnSlotViews {
    k_packed: MlxBuffer,
    k_norms: MlxBuffer,
    v_packed: MlxBuffer,
    v_norms: MlxBuffer,
    capacity_tokens: u32,
}

impl TqFullAttnKvBuffers {
    /// Zero-copy views for one sequence in the outer `n_seqs` axis. The MLX
    /// kernels remain single-sequence kernels; Metal buffer offsets select
    /// the agent slot without changing their math.
    fn slot_views(
        &self,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        n_kv_heads: u32,
        cache_capacity: u32,
        head_dim: u32,
    ) -> Result<TqFullAttnSlotViews> {
        let n_seqs = self.layout.n_seqs();
        anyhow::ensure!(
            (slot_id.0 as usize) < n_seqs,
            "TQ slot {} out of range for n_seqs={n_seqs}",
            slot_id.0
        );
        let segment = self.layout.segment(slot_id.0 as usize)?;
        anyhow::ensure!(
            cache_capacity >= segment.capacity_tokens,
            "TQ slot {} physical capacity {} exceeds caller logical capacity {}",
            slot_id.0,
            segment.capacity_tokens,
            cache_capacity
        );
        let packed_elems = (n_kv_heads as usize)
            .checked_mul(segment.capacity_tokens as usize)
            .and_then(|value| value.checked_mul(head_dim as usize))
            .ok_or_else(|| anyhow!("TQ packed slot extent overflow"))?;
        let norms_elems = (n_kv_heads as usize)
            .checked_mul(segment.capacity_tokens as usize)
            .and_then(|value| value.checked_mul(self.norms_per_pos as usize))
            .ok_or_else(|| anyhow!("TQ norm slot extent overflow"))?;
        let packed_offset =
            self.layout
                .packed_base_elements(slot_id.0 as usize, n_kv_heads, head_dim)?;
        let norms_offset = self
            .layout
            .norms_base_elements(slot_id.0 as usize, n_kv_heads, self.norms_per_pos)?
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| anyhow!("TQ norm slot offset overflow"))?;
        Ok(TqFullAttnSlotViews {
            k_packed: self.k_packed.slice_view(packed_offset, packed_elems),
            k_norms: self.k_norms.slice_view(norms_offset, norms_elems),
            v_packed: self.v_packed.slice_view(packed_offset, packed_elems),
            v_norms: self.v_norms.slice_view(norms_offset, norms_elems),
            capacity_tokens: segment.capacity_tokens,
        })
    }
}

/// Number of F32 norms per (seq, head, position) for a given head_dim.
/// Mirrors mlx-native's formula at `forward_mlx.rs:2326`:
/// `(head_dim / 256).max(1)`.
///
/// Returns 1 for head_dim ∈ [1, 256] (qwen35 + qwen35moe production at
/// head_dim=256).  Returns 2 for head_dim=512.  Returns 3 for head_dim
/// ∈ [768, 1023] etc. — but production qwen35 head_dim is always 256,
/// so this is conservative future-proofing only.
#[inline]
pub fn tq_norms_per_pos_for(head_dim: u32) -> u32 {
    (head_dim / 256).max(1)
}

/// Allocate one full-attn slot's worth of TQ-active K/V buffers (U8
/// packed + F32 norms) for overwrite. Mirrors the production shape
/// the mlx-native `flash_attn_vec_tq_hb` kernel consumes per sequence,
/// extended with the qwen35 `n_seqs` outer axis.
///
/// **Iter-7 scope:** standalone allocator only — no `HybridKvCache`
/// integration yet.  Iter-8 wires this into the per-slot allocator.
///
/// # Errors
///
/// Returns an error if any buffer allocation fails or if `max_seq_len`
/// or `n_seqs` is zero (mirrors `HybridKvCache::new`'s preflight).
pub fn alloc_tq_full_attn_buffers(
    cfg: &Qwen35Config,
    device: &MlxDevice,
    max_seq_len: u32,
    n_seqs: u32,
) -> Result<TqFullAttnKvBuffers> {
    if max_seq_len == 0 {
        return Err(anyhow!(
            "alloc_tq_full_attn_buffers: max_seq_len must be > 0"
        ));
    }
    if n_seqs == 0 {
        return Err(anyhow!("alloc_tq_full_attn_buffers: n_seqs must be > 0"));
    }

    let layout = TqArenaLayout::uniform(n_seqs, max_seq_len)?;
    alloc_tq_full_attn_buffers_with_layout(cfg, device, layout)
}

fn alloc_tq_full_attn_buffers_with_layout(
    cfg: &Qwen35Config,
    device: &MlxDevice,
    layout: TqArenaLayout,
) -> Result<TqFullAttnKvBuffers> {
    let n_kv_heads = cfg.num_key_value_heads as usize;
    let head_dim = cfg.head_dim;
    let norms_per_pos = tq_norms_per_pos_for(head_dim);
    let total_capacity_tokens = layout.total_capacity_tokens() as usize;
    let first_capacity = layout.segment(0)?.capacity_tokens as usize;
    let uniform_capacity = layout
        .capacities()
        .all(|capacity| capacity as usize == first_capacity)
        .then_some(first_capacity);

    // Packed arena: concatenated per-slot head-major segments. A static
    // uniform layout remains byte-identical to the historical outer-axis
    // allocation; a growable layout may assign different capacities.
    // 1 byte per element (8-bit Lloyd-Max index).
    let packed_elems = total_capacity_tokens * n_kv_heads * (head_dim as usize);
    let packed_bytes = packed_elems; // U8 → 1 byte/elem
    let packed_shape = match uniform_capacity {
        Some(capacity) => vec![layout.n_seqs(), n_kv_heads, capacity, head_dim as usize],
        None => vec![packed_elems],
    };

    // Norm arena mirrors the packed segment order.
    let norms_elems = total_capacity_tokens * n_kv_heads * (norms_per_pos as usize);
    let norms_bytes = norms_elems * std::mem::size_of::<f32>();
    let norms_shape = match uniform_capacity {
        Some(capacity) => vec![
            layout.n_seqs(),
            n_kv_heads,
            capacity,
            norms_per_pos as usize,
        ],
        None => vec![norms_elems],
    };

    // SAFETY: all TQ attention readers are bounded by the owning slot's
    // `current_len`; encoder writes packed values and norms before advancing
    // that cursor.
    let k_packed =
        unsafe { device.alloc_buffer_for_overwrite(packed_bytes, DType::U8, packed_shape.clone()) }
            .map_err(|e| anyhow!("alloc TQ full-attn K packed: {e}"))?;
    let k_norms =
        unsafe { device.alloc_buffer_for_overwrite(norms_bytes, DType::F32, norms_shape.clone()) }
            .map_err(|e| anyhow!("alloc TQ full-attn K norms: {e}"))?;
    let v_packed =
        unsafe { device.alloc_buffer_for_overwrite(packed_bytes, DType::U8, packed_shape) }
            .map_err(|e| anyhow!("alloc TQ full-attn V packed: {e}"))?;
    let v_norms =
        unsafe { device.alloc_buffer_for_overwrite(norms_bytes, DType::F32, norms_shape) }
            .map_err(|e| anyhow!("alloc TQ full-attn V norms: {e}"))?;

    Ok(TqFullAttnKvBuffers {
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        norms_per_pos,
        layout,
    })
}

/// Total bytes the TQ-active full-attn slot occupies (sum of all 4
/// buffers).  Useful for memory accounting + the parity test.
impl TqFullAttnKvBuffers {
    pub fn physical_capacity_for_slot(
        &self,
        slot_id: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<u32> {
        Ok(self.layout.segment(slot_id.0 as usize)?.capacity_tokens)
    }

    /// Exact component bytes after growing one slot and compacting the arena.
    /// This is a planning primitive only; allocation/copy/swap happens in the
    /// owning cache so all layers change atomically.
    pub fn planned_total_bytes_after_slot_growth(
        &self,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        required_tokens: u32,
        logical_max_tokens: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) -> Result<usize> {
        let layout =
            self.layout
                .grow_slot(slot_id.0 as usize, required_tokens, logical_max_tokens)?;
        let total_tokens = layout.total_capacity_tokens() as usize;
        let packed = total_tokens
            .checked_mul(n_kv_heads as usize)
            .and_then(|value| value.checked_mul(head_dim as usize))
            .ok_or_else(|| anyhow!("TQ arena packed byte plan overflow"))?;
        let norms = total_tokens
            .checked_mul(n_kv_heads as usize)
            .and_then(|value| value.checked_mul(self.norms_per_pos as usize))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| anyhow!("TQ arena norm byte plan overflow"))?;
        packed
            .checked_add(norms)
            .and_then(|one_side| one_side.checked_mul(2))
            .ok_or_else(|| anyhow!("TQ arena total byte plan overflow"))
    }

    pub fn total_bytes(&self) -> usize {
        self.k_packed.byte_len()
            + self.k_norms.byte_len()
            + self.v_packed.byte_len()
            + self.v_norms.byte_len()
    }
}

impl std::fmt::Debug for TqFullAttnKvBuffers {
    /// Surface only counts + total bytes — `MlxBuffer` does not implement
    /// `Debug` (Metal device handles can't be safely printed). Mirrors
    /// the `HybridKvCacheSnapshot` Debug impl above.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TqFullAttnKvBuffers")
            .field("k_packed_bytes", &self.k_packed.byte_len())
            .field("k_norms_bytes", &self.k_norms.byte_len())
            .field("v_packed_bytes", &self.v_packed.byte_len())
            .field("v_norms_bytes", &self.v_norms.byte_len())
            .field("norms_per_pos", &self.norms_per_pos)
            .field("total_bytes", &self.total_bytes())
            .finish()
    }
}

/// Compute the F32 K+V byte count for one full-attn slot at the given
/// shape.  Matches the existing `alloc_full_attn_slot` formula.  Used
/// by the iter-7 parity test to assert the TQ savings ratio.
pub fn full_attn_slot_f32_bytes(cfg: &Qwen35Config, max_seq_len: u32, n_seqs: u32) -> usize {
    let elems = (n_seqs as usize)
        * (cfg.num_key_value_heads as usize)
        * (max_seq_len as usize)
        * (cfg.head_dim as usize);
    // K + V, 4 bytes each (F32).
    2 * elems * std::mem::size_of::<f32>()
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A2a iter-2a — MultiSeqKvCache impl for HybridKvCache.
//
// Scope (per dossier §4 iter-2a + §2.10 R1):
//   * FULL-ATTENTION slot lift + MTP slot lift.  Both buffers carry
//     `n_seqs` as the outermost axis in their 4-D shape (kv_cache.rs:
//     2226-2236), and their per-seq cursor is `current_len: Vec<u32>`
//     of length `n_seqs` (kv_cache.rs:2213, 2247).  No new kernels
//     needed for the iter-2a surface (which mutates cursors only —
//     buffer-content writes land in Phase B iter-3 forward-path slot
//     threading per ADR-040 §2.2 + dossier §2.10 R2).
//
//   * LINEAR-ATTENTION: DEFERRED to Phase A2b.  Per dossier §2.1.4
//     and §2.10 R1, the spec-decode capture buffer at
//     kv_cache.rs:1476-1480 has the n_tokens_max axis OUTSIDE n_seqs,
//     and the `rollback_la_to` guard at kv_cache.rs:1567 explicitly
//     errors when `n_seqs > 1`.  The recurrent state alone scales
//     linearly (H1 verifies this), but `LinearAttnStateSlot` has no
//     logical "cursor" — recurrent state is updated in-kernel during
//     decode, not via a per-call cursor bump.  Therefore the trait's
//     `append_for_seq` / `drop_seq` on linear-attn slots are no-ops
//     in Phase A2a (the trait mutates cursors only; the linear-attn
//     state will be lifted when Phase A2b reshapes the capture buffer
//     and lifts the rollback guard).
//
//   * `fork_seq` cross-slot: DEFERRED to Phase A2b/A2c.  Per dossier
//     §2.10 R5, same-buffer cross-region memcpy via
//     `dispatch_kv_cache_copy_seq_*` is a NEW kernel pattern that
//     needs its own unit-test arc.  Phase A2a returns `SlotOom` for
//     any cross-slot fork to signal "kernel-dispatch path not yet
//     implemented"; the same-slot (`src == dst`) case is a no-op
//     success.  This is a sequenced contract, not a stub: the impl
//     ships as soon as the kernel arc lands.  Per cfa-finding-F2
//     (no `Err::NotImplemented` variant on `MultiSeqError`) we re-use
//     the existing `SlotOom` discriminant with sentinel byte values
//     `(needed_bytes=0, budget_bytes=0)` — operators reading the
//     error message see a clear "out of memory" shape that maps to
//     the Decision #19 429 + Retry-After upstream path while iter-A2c
//     wires the real kernel dispatch.
//
//   * Persistor multi-seq round-trip test: deferred to Phase A2a
//     follow-up (the persistor wire format at
//     `qwen35_hybrid_persistor.rs:171-175` already threads `n_seqs`,
//     but the test lives in a different file tree under
//     `src/serve/kv_persist/families/` and is out of scope for the
//     `src/inference/models/qwen35/kv_cache.rs`-only edit boundary
//     of Phase A2a).
//
// LayoutNotSupported is NEVER returned: HybridKvCache only supports
// `SeparateSlots`, and `Paged` is an alternate construction that this
// type does not expose.  Bounds-first ordering per iter-1.5
// cfa-finding-F5 is preserved across all four methods.
// ──────────────────────────────────────────────────────────────────────────

impl crate::serve::multi_seq_kv::MultiSeqKvCache for HybridKvCache {
    fn layout(&self) -> crate::serve::multi_seq_kv::MultiSeqLayout {
        crate::serve::multi_seq_kv::MultiSeqLayout::SeparateSlots
    }

    fn slot_count(&self) -> u32 {
        // `HybridKvCache::n_seqs` is already `u32` (kv_cache.rs:695); no cast.
        self.n_seqs
    }

    fn seq_len(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<u32, crate::serve::multi_seq_kv::MultiSeqError> {
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // Per dossier §4 iter-2a step 2: full-attn cursors are
        // homogeneous across full_attn slots in production (every
        // full-attn layer advances together per token).  Reading from
        // `full_attn[0]` is the canonical source. The optional MTP cache has
        // an independent speculative cursor: ordinary verifier prefill does
        // not advance it, while proposal processing does. It must never be
        // included in the public verifier-length invariant. If `full_attn`
        // is empty (degenerate config — Qwen35 production always has at
        // least one full-attn layer), fall back to 0 to keep the trait total.
        if self.full_attn.is_empty() {
            return Ok(0);
        }
        let canonical = self.full_attn[0].current_len[slot.0 as usize];

        // iter-2.5 C4: defensive cursor-homogeneity check.  Production
        // wiring MUST keep `current_len[slot.0]` identical across all
        // `full_attn[i]` (and the MTP slot if present) because
        // `append_for_seq` bumps them in lockstep.  A desync
        // (checkpoint replay, partial rollback, kernel error) would
        // silently lie via this accessor — debug builds fail-fast so
        // the desync is caught in dev/CI; release builds trust the
        // invariant and return the canonical_0 reading (consistent
        // runtime behaviour, no panic, no Result-shape change).  If a
        // future incident reveals desync in prod, escalate to a
        // Result-return that includes the per-layer cursor vector.
        debug_assert!(
            self.full_attn
                .iter()
                .all(|s| s.current_len[slot.0 as usize] == canonical),
            "HybridKvCache::seq_len({:?}): current_len desynchronized across \
             full_attn layers; canonical=full_attn[0].current_len[{}]={} but \
             at least one slot disagrees",
            slot,
            slot.0,
            canonical
        );
        Ok(canonical)
    }

    fn append_for_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        n_tokens: u32,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds (iter-1.5 cfa-finding-F5 ordering).
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots is the only layout HybridKvCache supports
        //    — no LayoutNotSupported branch.
        // 3. Budget: SeparateSlots cannot SlotOom on append (buffers are
        //    pre-allocated at construction; cursor overflow is bounded by
        //    `max_seq_len` and protected by `saturating_add` below).
        //
        // ADR-040 Phase A2a scope: bump current_len across all full_attn
        // slots + the MTP slot.  Linear-attn cursor lift is DEFERRED to
        // Phase A2b — `LinearAttnStateSlot` has no logical cursor and the
        // `rollback_la_to` guard at kv_cache.rs:1567 explicitly rejects
        // n_seqs > 1 until the capture-buffer layout is re-derived.
        for slot_data in &mut self.full_attn {
            let cur = &mut slot_data.current_len[slot.0 as usize];
            *cur = cur.saturating_add(n_tokens);
        }
        if let Some(ref mut mtp) = self.mtp_slot {
            let cur = &mut mtp.current_len[slot.0 as usize];
            *cur = cur.saturating_add(n_tokens);
        }
        Ok(())
    }

    fn drop_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds (iter-1.5 cfa-finding-F5 ordering).
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only — no LayoutNotSupported.
        // 3. Budget: drop is a pure release; SlotOom is unreachable here
        //    (per trait doc-comment at multi_seq_kv.rs:396-399).
        //
        // ADR-040 Phase A2a scope: zero current_len[slot] across all
        // full_attn slots + MTP.  Do NOT zero recurrent state — that is
        // Phase A2b's responsibility, gated on the `rollback_la_to`
        // guard at kv_cache.rs:1567 being lifted.  The recurrent state
        // remains at its prior contents; the next forward pass that
        // touches `slot` will overwrite per the linear-attn kernel's
        // ping-pong protocol (recurrent ↔ recurrent_scratch).  This is
        // sound for Phase A2a's full-attn-only forward routing because
        // forward paths under multi-seq do not yet dispatch linear-attn
        // (see Phase B iter-3 + dossier §2.10 R2).
        for slot_data in &mut self.full_attn {
            slot_data.current_len[slot.0 as usize] = 0;
        }
        if let Some(ref mut mtp) = self.mtp_slot {
            mtp.current_len[slot.0 as usize] = 0;
        }
        Ok(())
    }

    fn fork_seq(
        &mut self,
        src: crate::serve::multi_seq_kv::SlotId,
        dst: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds — src FIRST per iter-1.5 cfa-finding-F5 (so a fully
        //    invalid (src, dst) pair surfaces src as the OOR victim
        //    deterministically — pinned by the fixture-parity test in
        //    `serve::multi_seq_kv::tests`).
        if src.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot: src,
                max_slots: self.n_seqs,
            });
        }
        if dst.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot: dst,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only — no LayoutNotSupported.
        // 3. Same-slot fork is a no-op per trait spec — every reader of
        //    `dst` after this call sees the same bytes (src's bytes).
        if src == dst {
            return Ok(());
        }
        // ──────────────────────────────────────────────────────────────
        // ADR-040 Phase A2c (2026-05-30) — REAL cross-slot fork.
        //
        // Replaces the prior `CapabilityUnsupported` typed-deferral with
        // same-buffer cross-region memcpy via slice `copy_within` on
        // every per-slot byte region.  The per-slot byte-offset formulas
        // here MIRROR the slice_view byte-offset formulas the forward
        // path already uses for per-slot KV writes:
        //
        //   * Full-attn K / V (F32):
        //       `[n_seqs, n_kv_heads, max_seq_len, head_dim]`, OUTERMOST
        //       n_seqs ⇒ slot stride = `n_kv_heads * max_seq_len * head_dim * 4`
        //       — same as `slot_k_v_region_for_full_attn` at
        //       `gpu_full_attn.rs:102-116`.
        //
        //   * Full-attn TQ packed (U8) + TQ norms (F32):
        //       packed `[n_seqs, n_kv_heads, max_seq_len, head_dim]`, norms
        //       `[n_seqs, n_kv_heads, max_seq_len, norms_per_pos]`.  Stride
        //       formulas match `alloc_tq_full_attn_buffers` at
        //       `kv_cache.rs:2735-2763`.
        //
        //   * MTP slot: identical shape to full-attn slot per
        //       `HybridKvCache::new_with_mtp` discipline (the MTP slot
        //       block is appended at `layer_idx == num_hidden_layers`).
        //
        //   * Linear-attn recurrent / conv_state / recurrent_scratch /
        //       conv_state_scratch: same per-slot layout proofs as
        //       `gpu_delta_net.rs:160-181` per §6.1.40 iter-A2b-cont.
        //
        //   * Linear-attn capture_states + conv_capture_states (K=N
        //       spec-decode): optional buffers, per-slot stride includes
        //       n_tokens_max axis per `gpu_delta_net.rs:172-181`.
        //
        // Cursor copy: `current_len[dst] = current_len[src]` across every
        // full_attn slot + MTP slot.  Linear-attn slots carry no cursor
        // (recurrent state is in-buffer; the byte copy above handles it).
        //
        // PERFORMANCE: per-trait-doc `MultiSeqLayout::SeparateSlots` →
        // O(seq_len) per-slot copy.  This is the production reality of
        // prefix-share on SeparateSlots layouts — no zero-copy until the
        // Paged layout kernel arc lands.
        // ──────────────────────────────────────────────────────────────

        let src_idx = src.0 as usize;
        let dst_idx = dst.0 as usize;
        let n_seqs = self.n_seqs as usize;

        // (1) Full-attn slots (F32 K/V + optional TQ buffers + cursor).
        for slot in self.full_attn.iter_mut() {
            let cur_src = slot.current_len[src_idx];
            // F32 K/V: copy slot region bytes when Some.
            if let Some(ref mut k) = slot.k {
                copy_buffer_slot_prefix(k, src_idx, dst_idx, n_seqs, cur_src as usize).map_err(
                    |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: full-attn K copy failed ({e})"
                        )),
                    },
                )?;
            }
            if let Some(ref mut v) = slot.v {
                copy_buffer_slot_prefix(v, src_idx, dst_idx, n_seqs, cur_src as usize).map_err(
                    |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: full-attn V copy failed ({e})"
                        )),
                    },
                )?;
            }
            // TQ-active shadow buffers (packed U8 + norms F32).
            if let Some(ref mut tq) = slot.tq {
                copy_buffer_slot_prefix(
                    &mut tq.k_packed,
                    src_idx,
                    dst_idx,
                    n_seqs,
                    cur_src as usize,
                )
                .map_err(|e| {
                    crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: TQ K packed copy failed ({e})"
                        )),
                    }
                })?;
                copy_buffer_slot_prefix(
                    &mut tq.v_packed,
                    src_idx,
                    dst_idx,
                    n_seqs,
                    cur_src as usize,
                )
                .map_err(|e| {
                    crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: TQ V packed copy failed ({e})"
                        )),
                    }
                })?;
                copy_buffer_slot_prefix(
                    &mut tq.k_norms,
                    src_idx,
                    dst_idx,
                    n_seqs,
                    cur_src as usize,
                )
                .map_err(|e| {
                    crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: TQ K norms copy failed ({e})"
                        )),
                    }
                })?;
                copy_buffer_slot_prefix(
                    &mut tq.v_norms,
                    src_idx,
                    dst_idx,
                    n_seqs,
                    cur_src as usize,
                )
                .map_err(|e| {
                    crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: TQ V norms copy failed ({e})"
                        )),
                    }
                })?;
            }
            // Cursor copy AFTER buffer copy.
            slot.current_len[dst_idx] = cur_src;
        }

        // (2) MTP slot (same shape as full-attn; cursor + buffers).
        if let Some(ref mut mtp) = self.mtp_slot {
            let cur_src = mtp.current_len[src_idx];
            if let Some(ref mut k) = mtp.k {
                copy_buffer_slot_prefix(k, src_idx, dst_idx, n_seqs, cur_src as usize).map_err(
                    |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!("fork_seq: MTP K copy failed ({e})")),
                    },
                )?;
            }
            if let Some(ref mut v) = mtp.v {
                copy_buffer_slot_prefix(v, src_idx, dst_idx, n_seqs, cur_src as usize).map_err(
                    |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!("fork_seq: MTP V copy failed ({e})")),
                    },
                )?;
            }
            if let Some(ref mut tq) = mtp.tq {
                copy_buffer_slot_prefix(
                    &mut tq.k_packed,
                    src_idx,
                    dst_idx,
                    n_seqs,
                    cur_src as usize,
                )
                .map_err(|e| {
                    crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: MTP TQ K packed copy failed ({e})"
                        )),
                    }
                })?;
                copy_buffer_slot_prefix(
                    &mut tq.v_packed,
                    src_idx,
                    dst_idx,
                    n_seqs,
                    cur_src as usize,
                )
                .map_err(|e| {
                    crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: MTP TQ V packed copy failed ({e})"
                        )),
                    }
                })?;
                copy_buffer_slot_prefix(
                    &mut tq.k_norms,
                    src_idx,
                    dst_idx,
                    n_seqs,
                    cur_src as usize,
                )
                .map_err(|e| {
                    crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: MTP TQ K norms copy failed ({e})"
                        )),
                    }
                })?;
                copy_buffer_slot_prefix(
                    &mut tq.v_norms,
                    src_idx,
                    dst_idx,
                    n_seqs,
                    cur_src as usize,
                )
                .map_err(|e| {
                    crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: MTP TQ V norms copy failed ({e})"
                        )),
                    }
                })?;
            }
            mtp.current_len[dst_idx] = cur_src;
        }

        // (3) Linear-attn slots: recurrent + conv_state + scratches +
        // optional capture buffers.  Layout proofs at
        // `gpu_delta_net.rs:160-181` (recurrent col-major n_seqs
        // outermost; conv_state col-major n_seqs outermost; capture
        // recurrent col-major n_seqs outermost; conv_capture row-major
        // n_seqs outermost).  For copy_within purposes the AXIS ordering
        // doesn't matter — only that n_seqs is the OUTERMOST axis so
        // each slot's region is contiguous.  All four base buffers have
        // n_seqs as outermost per `alloc_linear_attn_slot` at
        // `kv_cache.rs:2575-2632`.
        for slot in self.linear_attn.iter_mut() {
            // recurrent + recurrent_scratch.
            copy_buffer_slot_region(&mut slot.recurrent, src_idx, dst_idx, n_seqs).map_err(
                |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                    capability: leak_static_str(format!(
                        "fork_seq: LA recurrent copy failed ({e})"
                    )),
                },
            )?;
            copy_buffer_slot_region(&mut slot.recurrent_scratch, src_idx, dst_idx, n_seqs)
                .map_err(
                    |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: LA recurrent_scratch copy failed ({e})"
                        )),
                    },
                )?;
            // conv_state + conv_state_scratch.
            copy_buffer_slot_region(&mut slot.conv_state, src_idx, dst_idx, n_seqs).map_err(
                |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                    capability: leak_static_str(format!(
                        "fork_seq: LA conv_state copy failed ({e})"
                    )),
                },
            )?;
            copy_buffer_slot_region(&mut slot.conv_state_scratch, src_idx, dst_idx, n_seqs)
                .map_err(
                    |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: LA conv_state_scratch copy failed ({e})"
                        )),
                    },
                )?;
            // Optional capture buffers (K=N spec-decode).  Same n_seqs
            // outermost discipline.
            if let Some(ref mut cap) = slot.capture_states {
                copy_buffer_slot_region(cap, src_idx, dst_idx, n_seqs).map_err(|e| {
                    crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: LA capture_states copy failed ({e})"
                        )),
                    }
                })?;
            }
            if let Some(ref mut ccap) = slot.conv_capture_states {
                copy_buffer_slot_region(ccap, src_idx, dst_idx, n_seqs).map_err(|e| {
                    crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                        capability: leak_static_str(format!(
                            "fork_seq: LA conv_capture_states copy failed ({e})"
                        )),
                    }
                })?;
            }
            // ADR-040 M-QWEN: BOTH physical buffers' src regions were
            // copied to dst, so dst's (current, scratch) roles must match
            // src's — carry the ping-pong parity across the fork.
            slot.pp_flipped[dst_idx] = slot.pp_flipped[src_idx];
        }

        Ok(())
    }
}

/// ADR-040 Phase A2c (2026-05-30) — leak a `String` into a `&'static
/// str` for `MultiSeqError::CapabilityUnsupported` payloads constructed
/// from runtime context.  The error payload is `&'static str` (per the
/// iter-2.5 M1 surface); buffer-copy failures during fork are
/// production defects (every buffer is pre-allocated at construction
/// time, and `copy_within` only fails on out-of-bounds — which our
/// per-slot byte-offset formulas preclude by construction at the
/// MultiSeqKvCache impl level), so the leak is bounded.
#[inline]
fn leak_static_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// ADR-040 Phase A2c (2026-05-30) — same-buffer cross-region byte copy
/// keyed by an explicit `n_seqs` (the n_seqs axis position in the shape
/// vector varies across buffer types; per-slot byte stride is always
/// `total_bytes / n_seqs` because n_seqs is the outermost-in-memory
/// axis on every Qwen35 multi-seq buffer per the layout proofs at
/// `kv_cache.rs:2546-2632` + `alloc_tq_full_attn_buffers:2737-2776`).
///
/// For full-attn F32 K/V (`[n_seqs, n_kv_heads, max_seq_len, head_dim]`,
/// row-major n_seqs outermost) per-slot byte stride =
/// `n_kv_heads * max_seq_len * head_dim * 4` — same formula
/// `slot_k_v_region_for_full_attn` at `gpu_full_attn.rs:102-116` uses
/// for forward-path slice_view.
///
/// For full-attn TQ packed/norms (`[n_seqs, n_kv_heads, max_seq_len,
/// {head_dim,norms_per_pos}]`) per-slot byte stride =
/// `n_kv_heads * max_seq_len * {head_dim,norms_per_pos} * elem_size`
/// — `alloc_tq_full_attn_buffers:2737-2776`.
///
/// For linear-attn recurrent/conv_state (col-major col-major
/// `[..., n_seqs]` — n_seqs is the LAST shape dim ⇒ outermost in
/// memory) per-slot byte stride = `D_k * D_v * n_v_heads * 4`
/// (recurrent) / `channels * (K-1) * 4` (conv_state) per
/// `gpu_delta_net.rs:160-181`.
///
/// For linear-attn capture buffers (recurrent capture col-major n_seqs
/// outermost; conv_capture row-major n_seqs outermost) per-slot byte
/// stride collapses to `total_bytes / n_seqs` by the same outermost-
/// in-memory invariant.
fn copy_buffer_slot_prefix(
    buf: &mut MlxBuffer,
    src_idx: usize,
    dst_idx: usize,
    n_seqs: usize,
    live_tokens: usize,
) -> Result<()> {
    let shape = buf.shape().to_vec();
    anyhow::ensure!(
        shape.len() == 4 && shape[0] == n_seqs,
        "fork_seq prefix: expected [n_seqs, heads, capacity, inner], got {:?}",
        shape
    );
    anyhow::ensure!(
        src_idx < n_seqs && dst_idx < n_seqs && live_tokens <= shape[2],
        "fork_seq prefix: src={src_idx} dst={dst_idx} live_tokens={live_tokens} outside n_seqs={n_seqs} capacity={}",
        shape[2]
    );
    if live_tokens == 0 {
        return Ok(());
    }
    let heads = shape[1];
    let capacity = shape[2];
    let bytes_per_position = shape[3]
        .checked_mul(buf.dtype().size_of())
        .ok_or_else(|| anyhow!("fork_seq prefix byte extent overflow"))?;
    let head_stride = capacity
        .checked_mul(bytes_per_position)
        .ok_or_else(|| anyhow!("fork_seq prefix head stride overflow"))?;
    let slot_stride = heads
        .checked_mul(head_stride)
        .ok_or_else(|| anyhow!("fork_seq prefix slot stride overflow"))?;
    let copy_bytes = live_tokens
        .checked_mul(bytes_per_position)
        .ok_or_else(|| anyhow!("fork_seq prefix copy extent overflow"))?;
    let bytes = buf
        .as_mut_slice::<u8>()
        .map_err(|error| anyhow!("fork_seq prefix as_mut_slice<u8>: {error}"))?;
    for head in 0..heads {
        let src_start = src_idx * slot_stride + head * head_stride;
        let dst_start = dst_idx * slot_stride + head * head_stride;
        bytes.copy_within(src_start..src_start + copy_bytes, dst_start);
    }
    Ok(())
}

fn copy_buffer_slot_region(
    buf: &mut MlxBuffer,
    src_idx: usize,
    dst_idx: usize,
    n_seqs: usize,
) -> Result<()> {
    anyhow::ensure!(n_seqs > 0, "fork_seq: n_seqs must be > 0");
    let total_bytes = buf.byte_len();
    anyhow::ensure!(
        total_bytes % n_seqs == 0,
        "fork_seq: total_bytes={} not divisible by n_seqs={}",
        total_bytes,
        n_seqs
    );
    let per_slot_bytes = total_bytes / n_seqs;
    anyhow::ensure!(
        src_idx < n_seqs && dst_idx < n_seqs,
        "fork_seq: src/dst out of buffer range \
         (src={src_idx}, dst={dst_idx}, n_seqs={n_seqs})"
    );
    if per_slot_bytes == 0 {
        return Ok(());
    }
    let bytes = buf
        .as_mut_slice::<u8>()
        .map_err(|e| anyhow!("fork_seq: as_mut_slice<u8>: {e}"))?;
    let src_off = src_idx * per_slot_bytes;
    bytes.copy_within(src_off..src_off + per_slot_bytes, dst_idx * per_slot_bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::{
        default_layer_types, Qwen35LayerKind, Qwen35MoeConfig, Qwen35Variant,
    };
    use mlx_native::DType;

    fn moe_cfg_40layer() -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Moe,
            hidden_size: 2048,
            num_hidden_layers: 40,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            layer_types: default_layer_types(40, 4),
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rotary_dim: 64,
            mrope_section: [11, 11, 10, 0],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262144,
            vocab_size: 248320,
            attn_output_gate: true,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: true,
            intermediate_size: None,
            moe: Some(Qwen35MoeConfig {
                moe_intermediate_size: 512,
                num_experts: 256,
                num_experts_per_tok: 8,
                shared_expert_intermediate_size: 512,
            }),
        }
    }

    fn dense_cfg_64layer() -> Qwen35Config {
        let mut cfg = moe_cfg_40layer();
        cfg.variant = Qwen35Variant::Dense;
        cfg.num_hidden_layers = 64;
        cfg.layer_types = default_layer_types(64, 4);
        cfg.hidden_size = 5120;
        cfg.num_attention_heads = 24;
        cfg.num_key_value_heads = 4;
        cfg.linear_num_value_heads = 48;
        cfg.intermediate_size = Some(17408);
        cfg.moe = None;
        cfg
    }

    #[test]
    fn tq_slot_views_address_independent_outer_sequence_regions() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping: no Metal device: {error}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let capacity = 64u32;
        let mut tq = alloc_tq_full_attn_buffers(&cfg, &device, capacity, 2)
            .expect("two-sequence TQ allocation");
        tq.k_packed
            .as_mut_slice::<u8>()
            .expect("packed CPU mapping")
            .fill(0);
        tq.k_norms
            .as_mut_slice::<f32>()
            .expect("norm CPU mapping")
            .fill(0.0);

        let packed_per_slot = (cfg.num_key_value_heads * capacity * cfg.head_dim) as usize;
        let norms_per_slot = (cfg.num_key_value_heads * capacity * tq.norms_per_pos) as usize;
        let mut views = tq
            .slot_views(
                crate::serve::multi_seq_kv::SlotId(1),
                cfg.num_key_value_heads,
                capacity,
                cfg.head_dim,
            )
            .expect("slot 1 views");
        views
            .k_packed
            .as_mut_slice::<u8>()
            .expect("packed view mapping")[0] = 0xA5;
        views
            .k_norms
            .as_mut_slice::<f32>()
            .expect("norm view mapping")[0] = 3.25;

        let packed = tq.k_packed.as_slice::<u8>().expect("packed root mapping");
        assert_eq!(packed[0], 0, "slot 0 packed region changed");
        assert_eq!(packed[packed_per_slot], 0xA5, "slot 1 packed offset");
        let norms = tq.k_norms.as_slice::<f32>().expect("norm root mapping");
        assert_eq!(norms[0], 0.0, "slot 0 norm region changed");
        assert_eq!(norms[norms_per_slot], 3.25, "slot 1 norm offset");
    }

    #[test]
    fn tq_gpu_encode_writes_slot_one_without_touching_slot_zero() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping: no Metal device: {error}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let capacity = 64u32;
        let mut cache = HybridKvCache::new_with_options(&cfg, &device, capacity, 2, true)
            .expect("two-sequence TQ cache");
        let slot = &mut cache.full_attn[0];
        let tq = slot.tq.as_mut().expect("TQ buffers");
        for buffer in [&mut tq.k_packed, &mut tq.v_packed] {
            buffer
                .as_mut_slice::<u8>()
                .expect("packed CPU mapping")
                .fill(0);
        }
        for buffer in [&mut tq.k_norms, &mut tq.v_norms] {
            buffer
                .as_mut_slice::<f32>()
                .expect("norm CPU mapping")
                .fill(0.0);
        }

        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let k = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 31);
        let v = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 37);
        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        let slot_one = crate::serve::multi_seq_kv::SlotId(1);
        slot.encode_seq_tokens_to_tq_for_slot(
            &k,
            true,
            1,
            n_kv_heads,
            head_dim,
            capacity,
            0,
            0,
            false,
            1.0,
            8,
            slot_one,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("slot-one K encode");
        slot.encode_seq_tokens_to_tq_for_slot(
            &v,
            false,
            1,
            n_kv_heads,
            head_dim,
            capacity,
            0,
            0,
            false,
            1.0,
            8,
            slot_one,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("slot-one V encode");
        encoder.commit_and_wait().expect("TQ encode completion");

        let tq = slot.tq.as_ref().expect("TQ buffers after encode");
        let packed_per_slot = (n_kv_heads * capacity * head_dim) as usize;
        let norms_per_slot = (n_kv_heads * capacity * tq.norms_per_pos) as usize;
        let packed = tq.k_packed.as_slice::<u8>().expect("packed readback");
        assert!(
            packed[..packed_per_slot].iter().all(|byte| *byte == 0),
            "slot-one encode modified slot-zero packed bytes"
        );
        assert!(
            packed[packed_per_slot..].iter().any(|byte| *byte != 0),
            "slot-one packed region was not written"
        );
        let norms = tq.k_norms.as_slice::<f32>().expect("norm readback");
        assert!(
            norms[..norms_per_slot].iter().all(|value| *value == 0.0),
            "slot-one encode modified slot-zero norms"
        );
        assert!(
            norms[norms_per_slot..].iter().any(|value| *value > 0.0),
            "slot-one norm region was not written"
        );

        // Exercise the read side against the same slot. Slot zero remains
        // all-zero, so a non-zero finite result also proves SDPA did not bind
        // the old hard-coded outer-axis origin.
        let num_heads = cfg.num_attention_heads;
        let q = synth_token_buffer(&device, num_heads as usize, head_dim as usize, 41);
        let output = device
            .alloc_buffer(
                (num_heads * head_dim * 4) as usize,
                DType::F32,
                vec![num_heads as usize, head_dim as usize],
            )
            .expect("TQ SDPA output");
        let tmp_bytes =
            mlx_native::ops::flash_attn_vec_tq_hb::tmp_buffer_bytes(num_heads, head_dim);
        let tmp = device
            .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
            .expect("TQ SDPA scratch");
        let mut encoder = device.command_encoder().expect("SDPA encoder");
        mlx_native::ops::fwht_standalone::dispatch_fwht_sign_premult_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &q,
            num_heads,
            head_dim,
        )
        .expect("Q rotation");
        encoder.memory_barrier();
        slot.dispatch_tq_sdpa_for_slot(
            &q,
            &output,
            &tmp,
            &Qwen35TqSdpaParams {
                num_heads,
                num_kv_heads: n_kv_heads,
                head_dim,
                kv_seq_len: 1,
                kv_capacity: capacity,
                scale: 1.0 / (head_dim as f32).sqrt(),
                mask_type: 0,
                sliding_window: 0,
                softcap: 0.0,
                ring_start: 0,
                scale_factor_d512: 1.0,
                codebook_bits: 8,
            },
            slot_one,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("slot-one TQ SDPA");
        encoder.memory_barrier();
        mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &output,
            num_heads,
            head_dim,
        )
        .expect("output inverse rotation");
        encoder.commit_and_wait().expect("TQ SDPA completion");
        let output_values = output.as_slice::<f32>().expect("TQ SDPA readback");
        assert!(output_values.iter().all(|value| value.is_finite()));
        assert!(output_values.iter().any(|value| value.abs() > 1e-6));
    }

    #[test]
    fn conv_channels_moe_8192() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        assert_eq!(conv_channels_for(&cfg), 8192);
    }

    #[test]
    fn conv_channels_dense_10240() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = dense_cfg_64layer();
        assert_eq!(conv_channels_for(&cfg), 10240);
    }

    /// ADR-013 acceptance criterion: 40-layer MoE with full_attention_interval=4
    /// produces 10 full-attn slots + 30 linear-attn slots.
    #[test]
    fn moe_40layer_slot_counts() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        // Use small max_seq_len for quick alloc.
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc cache");
        assert_eq!(cache.full_attn.len(), 10);
        assert_eq!(cache.linear_attn.len(), 30);
        assert_eq!(cache.full_attn.len() + cache.linear_attn.len(), 40);
    }

    #[test]
    fn dense_64layer_slot_counts() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = dense_cfg_64layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc cache");
        assert_eq!(cache.full_attn.len(), 16); // 64 / 4
        assert_eq!(cache.linear_attn.len(), 48);
    }

    #[test]
    fn layer_slot_lookup_matches_layer_types() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc");

        for (i, kind) in cfg.layer_types.iter().enumerate() {
            let slot = cache
                .slot_index_for_layer(i as u32)
                .expect("has slot for layer");
            match (kind, slot) {
                (Qwen35LayerKind::FullAttention, LayerSlot::Full(_)) => {}
                (Qwen35LayerKind::LinearAttention, LayerSlot::Linear(_)) => {}
                _ => panic!(
                    "layer {} kind {:?} resolved to mismatched slot {:?}",
                    i, kind, slot
                ),
            }
        }
    }

    #[test]
    fn slot_lookup_out_of_range_none() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc");
        assert!(cache.slot_index_for_layer(40).is_none());
        assert!(cache.slot_index_for_layer(9999).is_none());
    }

    #[test]
    fn full_attn_slot_shape_and_dtype() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 64, 2).expect("alloc");
        let s = &cache.full_attn[0];
        // iter-29 (sub-sub-iter 23c-α): legacy `new()` always emits
        // Some K/V; iter-30's tq_kv_active=true alloc branch is the
        // None case.
        let sk = s.k.as_ref().expect("legacy new()⇒Some(k)");
        let sv = s.v.as_ref().expect("legacy new()⇒Some(v)");
        assert_eq!(sk.dtype(), DType::F32);
        assert_eq!(sv.dtype(), DType::F32);
        // Expected element count: n_seqs * n_kv * max_seq_len * head_dim
        // = 2 * 2 * 64 * 256 = 65536.  Layout is SDPA-native [n_seqs, n_kv, max_seq, head_dim].
        assert_eq!(sk.element_count(), 2 * 2 * 64 * 256);
        assert_eq!(sv.element_count(), 2 * 2 * 64 * 256);
        assert_eq!(s.current_len.len(), 2);
        assert!(s.current_len.iter().all(|&c| c == 0));
    }

    #[test]
    fn linear_attn_slot_shape_matches_kernel_layout() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc");
        let s = &cache.linear_attn[0];
        // conv_state: [K-1=3, conv_channels=8192, n_seqs=1]
        assert_eq!(s.conv_state.element_count(), 3 * 8192 * 1);
        // recurrent: [D_k=128, D_v=128, num_v_heads=32, n_seqs=1]
        assert_eq!(s.recurrent.element_count(), 128 * 128 * 32 * 1);
    }

    /// Overwrite-backed attention storage begins cursor-invisible. Semantic
    /// recurrent state still begins at zero because it is read before the
    /// first DeltaNet update.
    #[test]
    fn new_hides_lazy_attention_tails_and_zeros_semantic_state() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 64, 2).expect("alloc");

        // Lazy attention bytes are deliberately unreadable until their
        // cursors advance; only the cursor state is observable here.
        for (idx, slot) in cache.full_attn.iter().enumerate() {
            assert!(
                slot.current_len.iter().all(|&c| c == 0),
                "full_attn[{idx}] starts cursor-visible"
            );
        }
        // Every linear-attn SSM-state byte must be zero.
        for (idx, slot) in cache.linear_attn.iter().enumerate() {
            let conv = slot.conv_state.as_slice::<f32>().expect("conv slice");
            assert!(
                conv.iter().all(|v| v.to_bits() == 0),
                "linear_attn[{}].conv_state has non-zero bytes after new()",
                idx
            );
            let conv_s = slot
                .conv_state_scratch
                .as_slice::<f32>()
                .expect("conv_scratch slice");
            assert!(
                conv_s.iter().all(|v| v.to_bits() == 0),
                "linear_attn[{}].conv_state_scratch has non-zero bytes after new()",
                idx
            );
            let rec = slot.recurrent.as_slice::<f32>().expect("rec slice");
            assert!(
                rec.iter().all(|v| v.to_bits() == 0),
                "linear_attn[{}].recurrent has non-zero bytes after new()",
                idx
            );
            let rec_s = slot
                .recurrent_scratch
                .as_slice::<f32>()
                .expect("rec_scratch slice");
            assert!(
                rec_s.iter().all(|v| v.to_bits() == 0),
                "linear_attn[{}].recurrent_scratch has non-zero bytes after new()",
                idx
            );
        }
    }

    #[test]
    fn reset_zeros_state_and_resets_cursors() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 2).expect("alloc");

        // Dirty the caches.
        for slot in cache.linear_attn.iter_mut().take(2) {
            let s = slot.recurrent.as_mut_slice::<f32>().expect("rec mut");
            for v in s.iter_mut().take(10) {
                *v = 1.0;
            }
        }
        for slot in cache.full_attn.iter_mut() {
            slot.current_len[0] = 5;
            slot.current_len[1] = 3;
        }

        cache.reset();

        for slot in &cache.full_attn {
            assert!(slot.current_len.iter().all(|&c| c == 0));
        }
        for slot in cache.linear_attn.iter_mut().take(2) {
            let s = slot.recurrent.as_slice::<f32>().expect("rec");
            for v in s.iter().take(10) {
                assert_eq!(*v, 0.0);
            }
        }
    }

    #[test]
    fn rejects_zero_seqs() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        assert!(HybridKvCache::new(&cfg, &device, 16, 0).is_err());
        assert!(HybridKvCache::new(&cfg, &device, 0, 1).is_err());
    }

    #[test]
    fn total_bytes_matches_expected_footprint() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 32, 1).expect("alloc");

        // Full-attn: 10 × 2 × 256 × 2 × 32 × 1 × 4 bytes = 10 × 131072 = 1.3 MB
        // (no ping-pong on full-attn KV cache — single buffer per slot)
        let full_expected = 10 * 2 * (256 * 2 * 32 * 1) * 4;
        // Linear-attn (post P13.3): each slot allocates ping-pong buffers
        // (active + scratch) for both conv_state and recurrent. The swap
        // happens on each decode step (LinearAttnStateSlot::swap_*); both
        // buffers are resident together. Per-slot footprint:
        //   conv_state             : 3 × 8192 × 1 × 4 = 98304 bytes
        //   conv_state_scratch     : 3 × 8192 × 1 × 4 = 98304 bytes  (ping-pong)
        //   recurrent              : 128 × 128 × 32 × 1 × 4 = 2097152 bytes
        //   recurrent_scratch      : 128 × 128 × 32 × 1 × 4 = 2097152 bytes (ping-pong)
        //   each slot: 4_390_912 bytes × 30 = 131_727_360
        let conv_bytes = 3 * 8192 * 1 * 4;
        let rec_bytes = 128 * 128 * 32 * 1 * 4;
        let linear_expected = 30 * (2 * conv_bytes + 2 * rec_bytes);
        let expected = full_expected + linear_expected;
        assert_eq!(cache.total_bytes(), expected);
    }

    // -- Wedge-3 / iter-216 Phase B: snapshot + restore ----------------

    /// Wedge-3 / iter-216 Phase B: snapshot captures byte-exact contents
    /// of every owned KV / SSM buffer, and restore_from puts them back
    /// after intervening mutation.
    #[test]
    fn hybrid_kv_cache_snapshot_round_trip_preserves_bytes() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc");

        // Plant non-zero canary values so the snapshot has something
        // unique to compare against zero / mutated bytes.
        // iter-29 (sub-sub-iter 23c-α): legacy `new()` always emits
        // Some K/V; tests `.expect("legacy new()⇒Some(_)")` to surface
        // any regression toward None on the F32 path.
        for (i, slot) in cache.full_attn.iter_mut().enumerate() {
            let kbuf = slot.k.as_mut().expect("legacy new()⇒Some(k)");
            let s = kbuf.as_mut_slice::<f32>().expect("k mut");
            s[0] = (i as f32) + 0.25;
            s[7] = (i as f32) + 0.5;
            let vbuf = slot.v.as_mut().expect("legacy new()⇒Some(v)");
            let s = vbuf.as_mut_slice::<f32>().expect("v mut");
            s[0] = -(i as f32) - 0.125;
            slot.current_len[0] = 8;
        }
        for (i, slot) in cache.linear_attn.iter_mut().enumerate() {
            let s = slot.conv_state.as_mut_slice::<f32>().expect("conv mut");
            s[0] = (i as f32) * 2.0 + 1.0;
            let s = slot.recurrent.as_mut_slice::<f32>().expect("rec mut");
            s[0] = (i as f32) * 0.5 + 0.125;
        }

        let snap = cache.snapshot(&device).expect("snapshot");

        // Capture canary values pre-mutation for later compare.
        let mut expect_full_k0: Vec<f32> = Vec::new();
        let mut expect_full_v0: Vec<f32> = Vec::new();
        let mut expect_full_lens: Vec<u32> = Vec::new();
        for slot in &cache.full_attn {
            let kbuf = slot.k.as_ref().expect("legacy new()⇒Some(k)");
            let vbuf = slot.v.as_ref().expect("legacy new()⇒Some(v)");
            expect_full_k0.push(kbuf.as_slice::<f32>().unwrap()[0]);
            expect_full_v0.push(vbuf.as_slice::<f32>().unwrap()[0]);
            expect_full_lens.push(slot.current_len[0]);
        }
        let mut expect_lin_conv0: Vec<f32> = Vec::new();
        let mut expect_lin_rec0: Vec<f32> = Vec::new();
        for slot in &cache.linear_attn {
            expect_lin_conv0.push(slot.conv_state.as_slice::<f32>().unwrap()[0]);
            expect_lin_rec0.push(slot.recurrent.as_slice::<f32>().unwrap()[0]);
        }

        // Mutate the live cache: zero out everything + change cursors.
        cache.reset();
        for slot in cache.full_attn.iter_mut() {
            let kbuf = slot.k.as_mut().expect("legacy new()⇒Some(k)");
            for v in kbuf.as_mut_slice::<f32>().unwrap().iter_mut() {
                *v = 999.0;
            }
            let vbuf = slot.v.as_mut().expect("legacy new()⇒Some(v)");
            for v in vbuf.as_mut_slice::<f32>().unwrap().iter_mut() {
                *v = -999.0;
            }
            slot.current_len[0] = 42;
        }

        // Restore — byte-equality across all canary positions.
        cache
            .restore_partial(&snap, 8)
            .expect("restore live prefix");
        for (i, slot) in cache.full_attn.iter().enumerate() {
            let kbuf = slot.k.as_ref().expect("legacy new()⇒Some(k)");
            let vbuf = slot.v.as_ref().expect("legacy new()⇒Some(v)");
            assert_eq!(
                kbuf.as_slice::<f32>().unwrap()[0],
                expect_full_k0[i],
                "full_attn[{i}].k[0] not restored"
            );
            assert_eq!(
                vbuf.as_slice::<f32>().unwrap()[0],
                expect_full_v0[i],
                "full_attn[{i}].v[0] not restored"
            );
            assert_eq!(
                slot.current_len[0], expect_full_lens[i],
                "full_attn[{i}].current_len[0] not restored"
            );
        }
        for (i, slot) in cache.linear_attn.iter().enumerate() {
            assert_eq!(
                slot.conv_state.as_slice::<f32>().unwrap()[0],
                expect_lin_conv0[i],
                "linear_attn[{i}].conv_state[0] not restored"
            );
            assert_eq!(
                slot.recurrent.as_slice::<f32>().unwrap()[0],
                expect_lin_rec0[i],
                "linear_attn[{i}].recurrent[0] not restored"
            );
        }
    }

    /// ADR-027 Phase B iter-35 (sub-iter 23d-α) — TQ snapshot round-trip
    /// preserves byte-equal TQ-buffer state across snapshot → mutate →
    /// restore_from cycles.
    ///
    /// **Load-bearing test for LCP-resume in TQ-only mode.** After
    /// iter-34 dropped the F32 K/V backing in TQ-active mode, the
    /// snapshot/restore path was the LAST place that still depended
    /// on slot.k/v being Some — without this iter's TQ snapshot fields
    /// + restore branch, an LCP-resume that hit a TQ-only cached
    /// snapshot would copy nothing into the new request's slot.tq
    /// buffers (zero-init), and decode would produce garbage.
    ///
    /// Sequence:
    /// (1) Build TQ-active cache (post-iter-34: slot.k=None, slot.tq=Some).
    /// (2) Plant canary bytes in slot.tq.k_packed[0..N] and v_norms.
    /// (3) snapshot() — captures slot.tq via deep-copy.
    /// (4) Mutate live slot.tq.k_packed[0..N] (set to different bytes).
    /// (5) restore_from(snapshot) — copies canary bytes back.
    /// (6) Assert slot.tq.k_packed bytes match original canary.
    #[test]
    fn hybrid_kv_cache_snapshot_round_trip_preserves_tq_bytes() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, 16, 1, true).expect("kv tq-on");

        // iter-34 invariant: slot.k/v are None, slot.tq is Some.
        assert!(
            cache.full_attn[0].k.is_none(),
            "iter-34: slot.k must be None"
        );
        assert!(cache.full_attn[0].tq.is_some(), "tq alloc must be Some");

        // Plant canary bytes in TQ buffers of slot 0.
        const CANARY_K_BYTE: u8 = 0xA5;
        const CANARY_V_BYTE: u8 = 0x5A;
        let canary_k_norm: f32 = 1.234567;
        let canary_v_norm: f32 = -2.345678;
        {
            let tq = cache.full_attn[0].tq.as_mut().expect("tq mut");
            tq.k_packed.as_mut_slice::<u8>().expect("k_packed mut")[0] = CANARY_K_BYTE;
            tq.v_packed.as_mut_slice::<u8>().expect("v_packed mut")[0] = CANARY_V_BYTE;
            tq.k_norms.as_mut_slice::<f32>().expect("k_norms mut")[0] = canary_k_norm;
            tq.v_norms.as_mut_slice::<f32>().expect("v_norms mut")[0] = canary_v_norm;
        }
        set_all_sequence_lengths(&mut cache, 1);

        // Take snapshot.
        let snap = cache.snapshot(&device).expect("snapshot");
        // iter-35 contract: snapshot.full_attn_tq must have one entry per
        // slot, all Some(_) when source had tq.
        assert_eq!(
            snap.full_attn_tq.len(),
            cache.full_attn.len(),
            "snapshot.full_attn_tq must align with cache.full_attn"
        );
        for (i, tq_snap) in snap.full_attn_tq.iter().enumerate() {
            assert!(
                tq_snap.is_some(),
                "snapshot.full_attn_tq[{i}] must be Some when slot.tq is Some"
            );
        }

        // Mutate live cache: blow away the TQ canaries.
        {
            let tq = cache.full_attn[0].tq.as_mut().expect("tq mut");
            tq.k_packed.as_mut_slice::<u8>().expect("k_packed mut")[0] = 0xFF;
            tq.v_packed.as_mut_slice::<u8>().expect("v_packed mut")[0] = 0x00;
            tq.k_norms.as_mut_slice::<f32>().expect("k_norms mut")[0] = -999.0;
            tq.v_norms.as_mut_slice::<f32>().expect("v_norms mut")[0] = 999.0;
        }

        // Restore from snapshot.
        cache
            .restore_partial(&snap, 1)
            .expect("restore live prefix");

        // Assert canary bytes recovered.
        let tq_restored = cache.full_attn[0].tq.as_ref().expect("tq ref");
        assert_eq!(
            tq_restored
                .k_packed
                .as_slice::<u8>()
                .expect("k_packed slice")[0],
            CANARY_K_BYTE,
            "tq.k_packed[0] not restored"
        );
        assert_eq!(
            tq_restored
                .v_packed
                .as_slice::<u8>()
                .expect("v_packed slice")[0],
            CANARY_V_BYTE,
            "tq.v_packed[0] not restored"
        );
        assert_eq!(
            tq_restored
                .k_norms
                .as_slice::<f32>()
                .expect("k_norms slice")[0],
            canary_k_norm,
            "tq.k_norms[0] not restored"
        );
        assert_eq!(
            tq_restored
                .v_norms
                .as_slice::<f32>()
                .expect("v_norms slice")[0],
            canary_v_norm,
            "tq.v_norms[0] not restored"
        );
    }

    /// ADR-027 Phase B iter-35 — defensive: snapshot/restore in legacy
    /// F32-only mode (no TQ) must continue to work bit-identically.
    /// snapshot.full_attn_tq is all-None, restore_from is a no-op for
    /// the TQ branch.
    #[test]
    fn hybrid_kv_cache_snapshot_restore_legacy_f32_unaffected_by_iter35() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        // Legacy F32-only mode (tq_kv_active=false).
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("kv legacy");
        assert!(cache.full_attn[0].k.is_some(), "legacy⇒Some(k)");
        assert!(cache.full_attn[0].tq.is_none(), "legacy⇒tq None");
        set_all_sequence_lengths(&mut cache, 1);

        let snap = cache.snapshot(&device).expect("snapshot");
        // iter-35 contract: snapshot.full_attn_tq is all-None when source
        // has no TQ buffers.
        for (i, tq_snap) in snap.full_attn_tq.iter().enumerate() {
            assert!(
                tq_snap.is_none(),
                "snapshot.full_attn_tq[{i}] must be None when slot.tq is None (legacy mode)"
            );
        }

        // Plant + mutate + restore F32 K canary (legacy-style round-trip).
        let canary_value: f32 = 7.5;
        cache.full_attn[0]
            .k
            .as_mut()
            .unwrap()
            .as_mut_slice::<f32>()
            .unwrap()[0] = canary_value;
        let snap2 = cache.snapshot(&device).expect("snapshot2");
        cache.full_attn[0]
            .k
            .as_mut()
            .unwrap()
            .as_mut_slice::<f32>()
            .unwrap()[0] = -1.0;
        cache
            .restore_partial(&snap2, 1)
            .expect("restore live prefix");
        assert_eq!(
            cache.full_attn[0]
                .k
                .as_ref()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()[0],
            canary_value,
            "legacy F32 round-trip MUST still work after iter-35 TQ field added"
        );
    }

    /// Wedge-3 / iter-216 Phase B: snapshot does NOT alias the source
    /// — mutating the source post-snapshot leaves snapshot bytes intact.
    #[test]
    fn hybrid_kv_cache_snapshot_does_not_alias() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc");

        // Plant a canary in slot 0.
        // iter-29 (sub-sub-iter 23c-α): legacy new()⇒Some K/V.
        cache.full_attn[0]
            .k
            .as_mut()
            .expect("legacy new()⇒Some(k)")
            .as_mut_slice::<f32>()
            .unwrap()[0] = 7.5;
        cache.linear_attn[0]
            .recurrent
            .as_mut_slice::<f32>()
            .unwrap()[0] = 3.25;
        set_all_sequence_lengths(&mut cache, 1);

        let snap = cache.snapshot(&device).expect("snapshot");
        // Canary values inside the snapshot.
        let snap_full_k0 = snap.full_attn_k[0]
            .as_ref()
            .expect("snap.k[0] some")
            .as_slice::<f32>()
            .unwrap()[0];
        let snap_lin_rec0 = snap.linear_recurrent[0].as_slice::<f32>().unwrap()[0];
        assert_eq!(snap_full_k0, 7.5);
        assert_eq!(snap_lin_rec0, 3.25);

        // Mutate the live cache — snapshot must NOT see this.
        cache.full_attn[0]
            .k
            .as_mut()
            .expect("legacy new()⇒Some(k)")
            .as_mut_slice::<f32>()
            .unwrap()[0] = -123.0;
        cache.linear_attn[0]
            .recurrent
            .as_mut_slice::<f32>()
            .unwrap()[0] = -456.0;

        // Snapshot still holds the original canaries (deep-copy, not Arc::clone).
        assert_eq!(
            snap.full_attn_k[0]
                .as_ref()
                .expect("snap.k[0] some")
                .as_slice::<f32>()
                .unwrap()[0],
            7.5,
            "snapshot aliased live cache (full_attn.k)"
        );
        assert_eq!(
            snap.linear_recurrent[0].as_slice::<f32>().unwrap()[0],
            3.25,
            "snapshot aliased live cache (linear recurrent)"
        );
    }

    /// Wedge-3 / iter-216 Phase B: total_bytes accounting on the snapshot
    /// equals the cache it came from (snapshot owns the same shape × counts).
    #[test]
    fn hybrid_kv_cache_snapshot_total_bytes_matches_source() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc");
        set_all_sequence_lengths(&mut cache, 1);
        let snap = cache.snapshot(&device).expect("snapshot");
        // Snapshot total_bytes = full_attn (k+v) + linear_attn (conv + recurrent),
        // i.e. excludes the live cache's scratch/ping-pong buffers (which the
        // snapshot doesn't own).  So snap.total_bytes <= cache.total_bytes.
        // Equality holds for the active-only subset.
        // iter-29 (sub-sub-iter 23c-α): legacy new()⇒Some on every slot.
        let cache_active_only: usize = cache
            .full_attn
            .iter()
            .map(|s| {
                let k = s.k.as_ref().expect("legacy new()⇒Some(k)");
                let per_token = k.shape()[1] * k.shape()[3] * k.dtype().size_of();
                2 * per_token
            })
            .sum::<usize>()
            + cache
                .linear_attn
                .iter()
                .map(|s| s.conv_state.element_count() * 4 + s.recurrent.element_count() * 4)
                .sum::<usize>();
        assert_eq!(snap.total_bytes(), cache_active_only);
    }

    /// Sanity smoke for the re-exported CPU reference: it exists and has
    /// the expected signature. Actual correctness is already tested in
    /// mlx-native (test_gated_delta_net.rs).
    #[test]
    fn re_exported_cpu_ref_callable() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use mlx_native::ops::gated_delta_net::GatedDeltaNetParams;

        let p = GatedDeltaNetParams {
            d_k: 4,
            d_v: 4,
            n_k_heads: 1,
            n_v_heads: 1,
            n_tokens: 1,
            n_seqs: 1,
        };
        let q = vec![0.0f32; 4];
        let k = vec![0.0f32; 4];
        let v = vec![0.1f32; 4];
        let g = vec![0.1f32; 1];
        let beta = vec![0.5f32; 1];
        let state_in = vec![0.0f32; 16];
        let (out, _state) = gated_delta_net_cpu_ref(&q, &k, &v, &g, &beta, &state_in, p);
        assert_eq!(out.len(), 4);
    }

    /// ADR-017 Phase E.a B.5 unit test: `partial_copy_slot` correctly
    /// copies the first `n_tokens` positions per (seq, head) across
    /// differently-sized source and destination buffers.
    ///
    /// Verifies:
    /// * Pattern preservation: known F32 values at positions
    ///   `[0..n_tokens]` per (seq, head) round-trip from src → dst
    ///   via the per-head stride math.
    /// * Tail isolation: dst positions `[n_tokens..dst_max_seq]`
    ///   remain untouched (zero-initialised).
    /// * Cross-head isolation: source head N's bytes don't leak into
    ///   destination head M (different stride bases).
    #[test]
    fn partial_copy_slot_per_head_position_round_trip() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("MlxDevice");
        let n_seqs = 1usize;
        let n_kv_heads = 2usize;
        let head_dim = 4usize;
        let src_max_seq = 8usize;
        let dst_max_seq = 16usize;
        let n_tokens = 5usize;

        // Build src buffer with a unique known F32 pattern per
        // (seq, head, pos, elem) so per-head + per-position
        // isolation is verifiable: value = 1000 + 100*seq + 10*head +
        // pos + 0.01*elem.
        let src_elems = n_seqs * n_kv_heads * src_max_seq * head_dim;
        let mut src_data = vec![0.0f32; src_elems];
        for seq in 0..n_seqs {
            for head in 0..n_kv_heads {
                for pos in 0..src_max_seq {
                    for elem in 0..head_dim {
                        let idx = ((seq * n_kv_heads + head) * src_max_seq + pos) * head_dim + elem;
                        src_data[idx] = 1000.0
                            + 100.0 * seq as f32
                            + 10.0 * head as f32
                            + pos as f32
                            + 0.01 * elem as f32;
                    }
                }
            }
        }
        let src_bytes = src_elems * 4;
        let src_shape = vec![n_seqs, n_kv_heads, src_max_seq, head_dim];
        let mut src_buf = device
            .alloc_buffer(src_bytes, DType::F32, src_shape)
            .expect("alloc src");
        src_buf
            .as_mut_slice::<f32>()
            .expect("src as_mut_slice")
            .copy_from_slice(&src_data);

        // dst zero-initialised at a different (larger) max_seq_len.
        let dst_elems = n_seqs * n_kv_heads * dst_max_seq * head_dim;
        let dst_bytes = dst_elems * 4;
        let dst_shape = vec![n_seqs, n_kv_heads, dst_max_seq, head_dim];
        let mut dst_buf = device
            .alloc_buffer(dst_bytes, DType::F32, dst_shape)
            .expect("alloc dst");

        partial_copy_slot(&src_buf, &mut dst_buf, n_tokens, "test_partial_copy")
            .expect("partial_copy_slot");

        // Verify dst contents.
        let dst_after = dst_buf.as_slice::<f32>().expect("dst as_slice").to_vec();

        // Per (seq, head, pos, elem):
        //   pos < n_tokens : MUST equal src's value.
        //   pos >= n_tokens: MUST be 0.0 (zero-initialised tail).
        for seq in 0..n_seqs {
            for head in 0..n_kv_heads {
                for pos in 0..dst_max_seq {
                    for elem in 0..head_dim {
                        let dst_idx =
                            ((seq * n_kv_heads + head) * dst_max_seq + pos) * head_dim + elem;
                        if pos < n_tokens {
                            // Compare to src[seq, head, pos, elem].
                            let expected = 1000.0
                                + 100.0 * seq as f32
                                + 10.0 * head as f32
                                + pos as f32
                                + 0.01 * elem as f32;
                            assert!(
                                (dst_after[dst_idx] - expected).abs() < 1e-6,
                                "partial_copy_slot: mismatch at \
                                 seq={seq}, head={head}, pos={pos}, elem={elem} \
                                 — got {}, expected {expected}",
                                dst_after[dst_idx]
                            );
                        } else {
                            assert_eq!(
                                dst_after[dst_idx], 0.0,
                                "partial_copy_slot: tail bleed at \
                                 seq={seq}, head={head}, pos={pos} (>= n_tokens={n_tokens}) \
                                 elem={elem} — got {}, expected 0.0",
                                dst_after[dst_idx]
                            );
                        }
                    }
                }
            }
        }
    }

    /// ADR-017 Phase E.a B.5 unit test: `partial_copy_slot` rejects
    /// rank mismatch (rank-3 instead of rank-4).
    #[test]
    fn partial_copy_slot_rejects_wrong_rank() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("MlxDevice");
        let bad_src = device
            .alloc_buffer(64, DType::F32, vec![2, 4, 2]) // rank 3
            .expect("alloc bad_src");
        let mut good_dst = device
            .alloc_buffer(64, DType::F32, vec![1, 2, 4, 2])
            .expect("alloc good_dst");
        let result = partial_copy_slot(&bad_src, &mut good_dst, 1, "test_rank");
        assert!(
            result.is_err(),
            "partial_copy_slot should reject rank-3 source"
        );
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("rank") || err_msg.contains("expected"),
            "error should mention rank/expected: {err_msg}"
        );
    }

    /// ADR-017 Phase E.a B.5 unit test: `partial_copy_slot` rejects
    /// `n_tokens > capacity`.
    #[test]
    fn partial_copy_slot_rejects_overshoot() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("MlxDevice");
        let src = device
            .alloc_buffer(64, DType::F32, vec![1, 2, 4, 2])
            .expect("alloc src");
        let mut dst = device
            .alloc_buffer(64, DType::F32, vec![1, 2, 4, 2])
            .expect("alloc dst");
        let result = partial_copy_slot(&src, &mut dst, 100, "test_overshoot");
        assert!(
            result.is_err(),
            "partial_copy_slot should reject n_tokens > capacity"
        );
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("exceeds capacity"),
            "error should mention capacity overshoot: {err_msg}"
        );
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-7 — TQ-active full-attn KV alloc tests
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn tq_norms_per_pos_for_qwen35_head_dim_256_is_one() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Qwen 3.5 / 3.6 production head_dim = 256 (verified by
        // Qwen35Config::head_dim default + APEX-Q5_K_M GGUF metadata).
        // Mirrors mlx-native `forward_mlx.rs:2326` formula exactly.
        assert_eq!(tq_norms_per_pos_for(256), 1);
        // Boundary cases — head_dim < 256 still rounds to 1.
        assert_eq!(tq_norms_per_pos_for(1), 1);
        assert_eq!(tq_norms_per_pos_for(64), 1);
        assert_eq!(tq_norms_per_pos_for(128), 1);
        assert_eq!(tq_norms_per_pos_for(255), 1);
        // head_dim = 512 → 2 (Gemma-class shape, future-proof for any
        // qwen variant that lifts head_dim).
        assert_eq!(tq_norms_per_pos_for(512), 2);
        // head_dim = 768 → 3 (purely for the saturating math; no
        // production model uses this today).
        assert_eq!(tq_norms_per_pos_for(768), 3);
    }

    #[test]
    fn tq_full_attn_buffers_alloc_byte_count_qwen36_apex_shape() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        // Qwen 3.6 35B-A3B-APEX-Q5_K_M production shape:
        //   n_kv_heads = 2, head_dim = 256, max_seq_len = 8192,
        //   n_seqs = 1.  These are the exact values
        //   `alloc_kv_cache_for_request` would pass for an 8K-token
        //   request.  Test asserts the exact byte counts so any future
        //   shape drift surfaces immediately.
        let cfg = moe_cfg_40layer();
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);

        let max_seq_len: u32 = 8192;
        let n_seqs: u32 = 1;
        let buffers = alloc_tq_full_attn_buffers(&cfg, &device, max_seq_len, n_seqs)
            .expect("alloc_tq_full_attn_buffers");

        // Expected byte counts at qwen36 APEX shape:
        //   k_packed: 1 × 2 × 8192 × 256 × 1 byte  = 4_194_304 bytes
        //   k_norms : 1 × 2 × 8192 × 1   × 4 bytes =    65_536 bytes
        //   v_packed: same as k_packed             = 4_194_304 bytes
        //   v_norms : same as k_norms              =    65_536 bytes
        //   total                                  = 8_519_680 bytes
        let expected_packed = 1 * 2 * 8192 * 256;
        let expected_norms = 1 * 2 * 8192 * 1 * 4;
        let expected_total = 2 * expected_packed + 2 * expected_norms;
        assert_eq!(buffers.k_packed.byte_len(), expected_packed);
        assert_eq!(buffers.k_norms.byte_len(), expected_norms);
        assert_eq!(buffers.v_packed.byte_len(), expected_packed);
        assert_eq!(buffers.v_norms.byte_len(), expected_norms);
        assert_eq!(buffers.total_bytes(), expected_total);
        assert_eq!(buffers.norms_per_pos, 1);
    }

    #[test]
    fn tq_full_attn_buffers_byte_count_3p94x_smaller_than_f32() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        // Qwen36 APEX shape — proves the 3.2× peer-parity claim from
        // ADR-027 §1's KV-memory table is achievable.  At
        // (n_seqs=1, n_kv_heads=2, max_seq_len=8192, head_dim=256):
        //
        //   F32 K+V : 2 × (1 × 2 × 8192 × 256 × 4) = 33_554_432 bytes
        //   TQ K+V  : (4_194_304 + 65_536) × 2     =  8_519_680 bytes
        //   ratio   : 33_554_432 / 8_519_680       = 3.94×
        //
        // The ADR §1 quote says 3.2× total cache reduction including
        // linear-attn (which stays F32) — at the FULL-ATTN-SLOT level
        // (this test's measurement) the ratio is closer to 4× because
        // norms overhead is small at head_dim=256.
        let cfg = moe_cfg_40layer();
        let max_seq_len: u32 = 8192;
        let n_seqs: u32 = 1;
        let f32_bytes = full_attn_slot_f32_bytes(&cfg, max_seq_len, n_seqs);
        let tq_buffers = alloc_tq_full_attn_buffers(&cfg, &device, max_seq_len, n_seqs)
            .expect("alloc_tq_full_attn_buffers");
        let tq_bytes = tq_buffers.total_bytes();

        let ratio = f32_bytes as f64 / tq_bytes as f64;
        assert!(
            (3.5..=4.5).contains(&ratio),
            "TQ savings ratio {ratio:.3}× outside expected [3.5, 4.5] window. \
             f32_bytes={f32_bytes}, tq_bytes={tq_bytes}"
        );
        // Spot-check the exact byte counts so any silent shape drift
        // surfaces.
        assert_eq!(f32_bytes, 33_554_432);
        assert_eq!(tq_bytes, 8_519_680);
    }

    #[test]
    fn tq_full_attn_buffers_alloc_rejects_zero_max_seq_len() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let err = alloc_tq_full_attn_buffers(&cfg, &device, 0, 1).unwrap_err();
        assert!(
            format!("{err:#}").contains("max_seq_len must be > 0"),
            "expected max_seq_len-zero error"
        );
    }

    #[test]
    fn tq_full_attn_buffers_alloc_rejects_zero_n_seqs() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let err = alloc_tq_full_attn_buffers(&cfg, &device, 8192, 0).unwrap_err();
        assert!(
            format!("{err:#}").contains("n_seqs must be > 0"),
            "expected n_seqs-zero error"
        );
    }

    #[test]
    fn tq_full_attn_buffers_start_cursor_invisible() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache = HybridKvCache::new_with_options(&cfg, &device, 32, 2, true)
            .expect("allocate cursor-guarded TQ cache");
        for slot in &cache.full_attn {
            assert!(
                slot.current_len.iter().all(|&len| len == 0),
                "uninitialized TQ bytes must remain invisible until writes advance the cursor"
            );
            assert!(slot.k.is_none() && slot.v.is_none());
            assert!(slot.tq.is_some());
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-8 — HybridKvCache::new_with_options tests
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn hybrid_kv_cache_new_with_options_tq_off_keeps_tq_none_per_slot() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Default path (tq_kv_active=false): every full-attn slot has
        // tq=None. Mirrors the legacy `HybridKvCache::new(...)` behavior
        // exactly. This test pins the regression contract for all 71
        // existing call sites.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache = HybridKvCache::new_with_options(&cfg, &device, 64, 1, false).expect("kv");
        assert!(
            !cache.full_attn.is_empty(),
            "test fixture has full-attn layers"
        );
        for (i, slot) in cache.full_attn.iter().enumerate() {
            assert!(
                slot.tq.is_none(),
                "full_attn[{i}].tq must be None when tq_kv_active=false"
            );
        }
        // Legacy `new()` is byte-identical to `new_with_options(... false)`.
        let legacy = HybridKvCache::new(&cfg, &device, 64, 1).expect("kv legacy");
        assert_eq!(legacy.full_attn.len(), cache.full_attn.len());
        for slot in legacy.full_attn.iter() {
            assert!(slot.tq.is_none(), "legacy `new()` keeps tq=None");
        }
    }

    #[test]
    fn hybrid_kv_cache_new_with_options_tq_on_populates_tq_per_full_attn_slot() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // tq_kv_active=true: every full-attn slot gets a populated
        // TqFullAttnKvBuffers alongside its existing F32 K/V buffers
        // (shadow-cache pattern; iter-11 drops the F32 backing).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache = HybridKvCache::new_with_options(&cfg, &device, 64, 1, true).expect("kv tq-on");
        assert!(!cache.full_attn.is_empty());
        let n_full_attn = cache.full_attn.len();
        for (i, slot) in cache.full_attn.iter().enumerate() {
            assert!(
                slot.tq.is_some(),
                "full_attn[{i}].tq must be Some when tq_kv_active=true"
            );
            let tq = slot.tq.as_ref().unwrap();
            assert_eq!(tq.norms_per_pos, 1, "head_dim=256 → norms_per_pos=1");
            // iter-34 (sub-sub-iter 23c-β.5): F32 K/V backing is dropped
            // when tq_kv_active=true. The slot now carries ONLY the TQ
            // buffers + current_len cursor (no F32 K/V allocation).
            // This is the load-bearing memory-savings invariant.
            assert!(
                slot.k.is_none(),
                "iter-34: slot.k must be None when tq_kv_active=true (F32 alloc dropped)"
            );
            assert!(
                slot.v.is_none(),
                "iter-34: slot.v must be None when tq_kv_active=true (F32 alloc dropped)"
            );
        }
        // MTP slot: tq present iff cfg has MTP. moe_cfg_40layer() sets
        // mtp_num_hidden_layers=0 → mtp_slot is None entirely.
        assert!(cache.mtp_slot.is_none(), "moe_cfg_40layer has no MTP");
        // Linear-attn slots are unchanged (no TQ field — DeltaNet SSM
        // state stays F32 per ADR-027 §3 non-goal).
        assert_eq!(
            cache.full_attn.len() + cache.linear_attn.len(),
            cfg.layer_types.len(),
            "every model layer maps to exactly one slot"
        );
        let _ = n_full_attn;
    }

    #[test]
    fn hybrid_kv_cache_new_with_options_tq_on_byte_count_at_qwen36_apex_shape() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Empirical byte-count parity at qwen36 35B-A3B-APEX shape:
        // each full-attn slot now holds ONLY TQ packed K+V (8.13 MB)
        // + TQ norms K+V (128 KB) = 8_519_680 bytes per slot.
        //
        // iter-34 (sub-sub-iter 23c-β.5): F32 K+V backing dropped (was
        // 16 MB each in shadow mode pre-iter-34). Per-slot total
        // 8_519_680 bytes — the load-bearing 3.94× memory savings vs
        // the 33.55 MB F32-only baseline (1×2×8192×256×4 each for K+V).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let max_seq_len: u32 = 8192;
        let n_seqs: u32 = 1;
        let cache = HybridKvCache::new_with_options(&cfg, &device, max_seq_len, n_seqs, true)
            .expect("kv tq-on");
        let slot = &cache.full_attn[0];
        assert!(slot.tq.is_some());
        let tq = slot.tq.as_ref().unwrap();
        // iter-34: F32 K and V are dropped — slot.k and slot.v are None.
        assert!(
            slot.k.is_none(),
            "iter-34: slot.k must be None when tq_kv_active=true (was 16 MB F32 in shadow mode)"
        );
        assert!(
            slot.v.is_none(),
            "iter-34: slot.v must be None when tq_kv_active=true (was 16 MB F32 in shadow mode)"
        );
        // TQ K_packed + V_packed: 1×2×8192×256 each (U8) = 4 MB each.
        assert_eq!(tq.k_packed.byte_len(), 1 * 2 * 8192 * 256);
        assert_eq!(tq.v_packed.byte_len(), 1 * 2 * 8192 * 256);
        // TQ K_norms + V_norms: 1×2×8192×1×4 each = 64 KB.
        assert_eq!(tq.k_norms.byte_len(), 1 * 2 * 8192 * 1 * 4);
        assert_eq!(tq.v_norms.byte_len(), 1 * 2 * 8192 * 1 * 4);
        // **Load-bearing 3.94× memory savings regression-pin:**
        // per-slot total = TQ packed+norms only (no F32 backing).
        // Pre-iter-34 shadow mode: 2 × 16 MB F32 + 8.52 MB TQ = 42_074_112.
        // Post-iter-34: 8_519_680 (3.94× smaller; 33.55 MB saved per slot).
        let per_slot_total = tq.total_bytes();
        assert_eq!(
            per_slot_total, 8_519_680,
            "iter-34: per-slot total must be TQ-only (3.94× savings vs F32+TQ shadow mode)"
        );
        // Reference: pre-iter-34 shadow total was 42_074_112 bytes
        // (2 × 16 MB F32 K+V + 8.52 MB TQ). Now 8_519_680 bytes.
        let pre_iter34_shadow_total = 2 * (1 * 2 * 8192 * 256 * 4) + 8_519_680;
        assert_eq!(pre_iter34_shadow_total, 42_074_112);
        // **The dossier-quoted 3.94× savings is vs the F32-ONLY baseline**
        // (legacy `HybridKvCache::new()` mode = 33_554_432 bytes per slot,
        // TQ buffers absent). iter-34 TQ-only = 8_519_680 bytes.
        // 33_554_432 / 8_519_680 = 3.937× ≈ 3.94×.
        let f32_only_baseline = 1 * 2 * 8192 * 256 * 4 * 2; // K + V each at 16 MB
        assert_eq!(f32_only_baseline, 33_554_432);
        let savings_ratio_vs_f32_only = f32_only_baseline as f64 / per_slot_total as f64;
        assert!(
            savings_ratio_vs_f32_only > 3.93 && savings_ratio_vs_f32_only < 3.95,
            "expected 3.94× F32-only→TQ-only savings, got {savings_ratio_vs_f32_only:.4}×"
        );
        // Bonus: vs pre-iter-34 shadow mode (which carried F32 + TQ),
        // savings is 4.94×.
        let savings_ratio_vs_shadow = pre_iter34_shadow_total as f64 / per_slot_total as f64;
        assert!(
            savings_ratio_vs_shadow > 4.93 && savings_ratio_vs_shadow < 4.95,
            "expected 4.94× shadow→TQ-only savings, got {savings_ratio_vs_shadow:.4}×"
        );
    }

    #[test]
    fn hybrid_kv_cache_new_with_options_tq_on_with_mtp_populates_mtp_tq() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Synthetic cfg with MTP enabled — the MTP full-attn slot
        // should ALSO get a populated tq when tq_kv_active=true.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let mut cfg = moe_cfg_40layer();
        cfg.mtp_num_hidden_layers = 1;
        let cache =
            HybridKvCache::new_with_options(&cfg, &device, 64, 1, true).expect("kv tq-on with mtp");
        assert!(cache.mtp_slot.is_some(), "cfg has MTP layer");
        let mtp = cache.mtp_slot.as_ref().unwrap();
        assert!(
            mtp.tq.is_some(),
            "MTP slot should ALSO have tq populated when tq_kv_active=true"
        );
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-28 (sub-iter 23b) — HybridKvCache.tq_kv_active
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn hybrid_kv_cache_tq_kv_active_field_matches_constructor_arg() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // The cache itself records its TQ-mode at construction. iter-29
        // (sub-iter 23c) keys the F32 K/V alloc branch off this field;
        // until then it must mirror `slot.tq.is_some()` for every
        // full-attn slot (and for the MTP slot if present).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();

        // tq_kv_active=false: field reads false; every slot.tq is None.
        let off = HybridKvCache::new_with_options(&cfg, &device, 64, 1, false).expect("kv tq-off");
        assert!(!off.tq_kv_active, "tq_kv_active must propagate (false)");
        for (i, slot) in off.full_attn.iter().enumerate() {
            assert!(
                slot.tq.is_none(),
                "tq_kv_active=false implies full_attn[{i}].tq.is_none()"
            );
        }

        // tq_kv_active=true: field reads true; every slot.tq is Some.
        let on = HybridKvCache::new_with_options(&cfg, &device, 64, 1, true).expect("kv tq-on");
        assert!(on.tq_kv_active, "tq_kv_active must propagate (true)");
        for (i, slot) in on.full_attn.iter().enumerate() {
            assert!(
                slot.tq.is_some(),
                "tq_kv_active=true implies full_attn[{i}].tq.is_some()"
            );
        }

        // Legacy `new()` defaults to false (regression contract).
        let legacy = HybridKvCache::new(&cfg, &device, 64, 1).expect("kv legacy");
        assert!(!legacy.tq_kv_active, "legacy `new()` ⇒ tq_kv_active=false");
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-9 — encode_token_to_tq GPU dispatch tests
    // ──────────────────────────────────────────────────────────────────

    /// Build a synthetic K/V token buffer of shape `[n_kv_heads, head_dim]`
    /// F32 with deterministic non-trivial values. The kernel applies FWHT
    /// + L2-norm + quant; non-zero input ensures non-zero norm + at least
    /// one non-zero packed index.
    fn synth_token_buffer(
        device: &MlxDevice,
        n_kv_heads: usize,
        head_dim: usize,
        salt: u32,
    ) -> MlxBuffer {
        let elems = n_kv_heads * head_dim;
        let bytes = elems * std::mem::size_of::<f32>();
        let mut buf = device
            .alloc_buffer(bytes, DType::F32, vec![n_kv_heads, head_dim])
            .expect("alloc token buf");
        {
            let s = buf.as_mut_slice::<f32>().expect("token mut slice");
            for (i, v) in s.iter_mut().enumerate() {
                // Non-trivial pattern: scaled sinusoid + salt offset.
                let x = ((i as u32 + salt) % 1000) as f32 / 1000.0;
                *v = (x * 6.28318).sin() * 0.5;
            }
        }
        buf
    }

    #[test]
    fn encode_token_to_tq_errors_when_slot_lacks_tq_buffers() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Mantra: fail loud, no silent fallback. Calling encode on a
        // legacy F32-only slot must error explicitly.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, 64, 1, false).expect("kv tq-off");
        // Pick a real full-attn slot.
        let slot = &mut cache.full_attn[0];
        assert!(slot.tq.is_none());
        let n_kv_heads = cfg.num_key_value_heads as u32;
        let head_dim = cfg.head_dim;
        let k_token = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 1);
        let v_token = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 2);
        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        let err = slot
            .encode_token_to_tq(
                &k_token,
                &v_token,
                n_kv_heads,
                head_dim,
                64,
                0,
                false,
                1.0,
                8,
                &mut encoder,
                &mut registry,
                &device,
            )
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("slot.tq is None"),
            "expected fail-loud None-tq error, got: {msg}"
        );
    }

    #[test]
    fn encode_token_to_tq_writes_packed_at_write_pos_only() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Encode one token at write_pos=5 in a TQ-active slot. Verify:
        // - k_packed bytes at position 5 are non-zero (post-quant indices)
        // - k_packed bytes at OTHER positions (0..5, 6..) retain a sentinel
        // This pins the kernel's positional addressing.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 64;
        let mut cache = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];
        assert!(slot.tq.is_some());
        const UNWRITTEN: u8 = 0xCC;
        slot.tq
            .as_mut()
            .expect("tq")
            .k_packed
            .as_mut_slice::<u8>()
            .expect("seed lazy tail")
            .fill(UNWRITTEN);
        let n_kv_heads = cfg.num_key_value_heads as u32;
        let head_dim = cfg.head_dim;
        let k_token = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 1);
        let v_token = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 2);

        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        let write_pos: u32 = 5;
        slot.encode_token_to_tq(
            &k_token,
            &v_token,
            n_kv_heads,
            head_dim,
            cache_capacity,
            write_pos,
            false,
            1.0,
            8,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("encode_token_to_tq dispatch");
        // commit + sync so the GPU writes are visible to as_slice.
        encoder.commit_and_wait().expect("encoder commit_and_wait");

        let tq = slot.tq.as_ref().unwrap();
        let k_packed_bytes = tq.k_packed.as_slice::<u8>().expect("k_packed slice");
        // Positional addressing: kernel writes at offset
        // `head*capacity*head_dim + write_pos*head_dim + dim_idx`.
        let head_dim_us = head_dim as usize;
        let cap_us = cache_capacity as usize;
        for head in 0..n_kv_heads as usize {
            let base = head * cap_us * head_dim_us;
            // At write_pos: at least one byte must differ from the unreadable
            // tail sentinel after the GPU producer runs.
            let pos_offset = base + (write_pos as usize) * head_dim_us;
            let pos_slice = &k_packed_bytes[pos_offset..pos_offset + head_dim_us];
            let wrote_at_pos = pos_slice.iter().any(|&b| b != UNWRITTEN);
            assert!(
                wrote_at_pos,
                "head={head} pos={write_pos}: encoder did not overwrite the sentinel"
            );
            // At other positions: bytes retain the pre-seeded sentinel.
            for other_pos in 0..cap_us {
                if other_pos as u32 == write_pos {
                    continue;
                }
                let other_offset = base + other_pos * head_dim_us;
                let other_slice = &k_packed_bytes[other_offset..other_offset + head_dim_us];
                assert!(
                    other_slice.iter().all(|&b| b == UNWRITTEN),
                    "head={head} pos={other_pos}: kernel must NOT write outside write_pos"
                );
            }
        }
    }

    #[test]
    fn encode_token_to_tq_writes_positive_norms() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // After FWHT + L2-norm extraction, the stored norm scalar must
        // be > 0 for any non-zero input. This pins the norm pipeline.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 16;
        let mut cache = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];
        let n_kv_heads = cfg.num_key_value_heads as u32;
        let head_dim = cfg.head_dim;
        let k_token = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 11);
        let v_token = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 13);

        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        slot.encode_token_to_tq(
            &k_token,
            &v_token,
            n_kv_heads,
            head_dim,
            cache_capacity,
            3,
            false,
            1.0,
            8,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("encode dispatch");
        encoder.commit_and_wait().expect("encoder commit_and_wait");

        let tq = slot.tq.as_ref().unwrap();
        let k_norms = tq.k_norms.as_slice::<f32>().expect("k_norms slice");
        let v_norms = tq.v_norms.as_slice::<f32>().expect("v_norms slice");
        // norms layout: [n_kv_heads, cache_capacity, norms_per_pos=1].
        // At write_pos=3 each head's norm must be > 0.
        for head in 0..n_kv_heads as usize {
            let idx = head * (cache_capacity as usize) * 1 + 3 * 1 + 0;
            assert!(
                k_norms[idx] > 0.0,
                "head={head} pos=3: expected positive K norm, got {}",
                k_norms[idx]
            );
            assert!(
                v_norms[idx] > 0.0,
                "head={head} pos=3: expected positive V norm, got {}",
                v_norms[idx]
            );
        }
    }

    #[test]
    fn encode_token_to_tq_at_two_positions_writes_both_independently() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Encode token A at pos=2 then token B at pos=7 — both positions
        // must have populated bytes; positions 0,1,3,4,5,6,8+ stay zero.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 16;
        let mut cache = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];
        let n_kv_heads = cfg.num_key_value_heads as u32;
        let head_dim = cfg.head_dim;
        let k_a = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 100);
        let v_a = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 200);
        let k_b = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 300);
        let v_b = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 400);

        let mut registry = mlx_native::KernelRegistry::new();
        // Dispatch A then B in the SAME encoder (production pattern: one
        // encoder per per-layer per-token write).
        let mut encoder = device.command_encoder().expect("encoder");
        slot.encode_token_to_tq(
            &k_a,
            &v_a,
            n_kv_heads,
            head_dim,
            cache_capacity,
            2,
            false,
            1.0,
            8,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("encode A");
        slot.encode_token_to_tq(
            &k_b,
            &v_b,
            n_kv_heads,
            head_dim,
            cache_capacity,
            7,
            false,
            1.0,
            8,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("encode B");
        encoder.commit_and_wait().expect("encoder commit_and_wait");

        let tq = slot.tq.as_ref().unwrap();
        let k_packed = tq.k_packed.as_slice::<u8>().expect("k_packed");
        let head_dim_us = head_dim as usize;
        let cap_us = cache_capacity as usize;
        for head in 0..n_kv_heads as usize {
            let base = head * cap_us * head_dim_us;
            for pos in 0..cap_us {
                let off = base + pos * head_dim_us;
                let slice = &k_packed[off..off + head_dim_us];
                let any_nonzero = slice.iter().any(|&b| b != 0);
                let expected_nonzero = pos == 2 || pos == 7;
                assert_eq!(
                    any_nonzero, expected_nonzero,
                    "head={head} pos={pos}: expected_nonzero={expected_nonzero}, \
                     got any_nonzero={any_nonzero}"
                );
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-10 — dispatch_tq_sdpa GPU dispatch tests
    // ──────────────────────────────────────────────────────────────────

    /// Helper: alloc the F32 destination/scratch buffers for the SDPA
    /// dispatch at qwen35 shape.  Returns (q, output, tmp).
    fn alloc_sdpa_buffers(
        device: &MlxDevice,
        num_heads: u32,
        head_dim: u32,
    ) -> (MlxBuffer, MlxBuffer, MlxBuffer) {
        let q_elems = (num_heads as usize) * (head_dim as usize);
        let q = device
            .alloc_buffer(
                q_elems * std::mem::size_of::<f32>(),
                DType::F32,
                vec![num_heads as usize, head_dim as usize],
            )
            .expect("alloc q");
        let output = device
            .alloc_buffer(
                q_elems * std::mem::size_of::<f32>(),
                DType::F32,
                vec![num_heads as usize, head_dim as usize],
            )
            .expect("alloc output");
        let tmp_bytes =
            mlx_native::ops::flash_attn_vec_tq_hb::tmp_buffer_bytes(num_heads, head_dim);
        let tmp = device
            .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
            .expect("alloc tmp");
        (q, output, tmp)
    }

    #[test]
    fn dispatch_tq_sdpa_errors_when_slot_lacks_tq_buffers() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Mantra: fail loud, no silent fallback. Calling SDPA on a
        // legacy F32-only slot must error explicitly.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache =
            HybridKvCache::new_with_options(&cfg, &device, 64, 1, false).expect("kv tq-off");
        let slot = &cache.full_attn[0];
        assert!(slot.tq.is_none());
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let (q, output, tmp) = alloc_sdpa_buffers(&device, num_heads, head_dim);
        let params = Qwen35TqSdpaParams {
            num_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            kv_seq_len: 1,
            kv_capacity: 64,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
        };
        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        let err = slot
            .dispatch_tq_sdpa(
                &q,
                &output,
                &tmp,
                &params,
                &mut encoder,
                &mut registry,
                &device,
            )
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("slot.tq is None"),
            "expected fail-loud None-tq error, got: {msg}"
        );
    }

    #[test]
    fn dispatch_tq_sdpa_produces_finite_nonzero_output_at_qwen35_shape() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Encode a single token's K, V via encode_token_to_tq, then
        // dispatch SDPA with kv_seq_len=1. Output must be:
        //   - finite (no NaN / no Inf)
        //   - non-zero (the kernel actually wrote something)
        // This is the iter-10 sanity check — full F32-baseline NRMSE
        // parity is iter-11.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 64;
        let mut cache = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];
        let n_kv_heads = cfg.num_key_value_heads;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;

        // Allocate K, V tokens with deterministic non-trivial values.
        let k_token = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 11);
        let v_token = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 13);

        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        // Encode the single KV token at write_pos=0.
        slot.encode_token_to_tq(
            &k_token,
            &v_token,
            n_kv_heads,
            head_dim,
            cache_capacity,
            0,
            false,
            1.0,
            8,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("encode");
        // RAW dependency: SDPA below reads the packed K/V and norms written by
        // the two encode dispatches above. This is part of the production
        // pattern, not an optional test synchronization aid.
        encoder.memory_barrier();

        // Build Q (FWHT-rotation skipped — sanity test only checks
        // finite/non-zero output, not numerical correctness).
        let (mut q_buf, output, tmp) = alloc_sdpa_buffers(&device, num_heads, head_dim);
        {
            let s = q_buf.as_mut_slice::<f32>().expect("q mut");
            for (i, v) in s.iter_mut().enumerate() {
                *v = ((i as f32) * 0.001).cos() * 0.5;
            }
        }
        let params = Qwen35TqSdpaParams {
            num_heads,
            num_kv_heads: n_kv_heads,
            head_dim,
            kv_seq_len: 1,
            kv_capacity: cache_capacity,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
        };

        // Dispatch SDPA on the SAME encoder (production pattern: encode
        // → dispatch in one CB).
        slot.dispatch_tq_sdpa(
            &q_buf,
            &output,
            &tmp,
            &params,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("dispatch_tq_sdpa");
        encoder.commit_and_wait().expect("commit_and_wait");

        let out = output.as_slice::<f32>().expect("output slice");
        let mut any_nonzero = false;
        for &v in out.iter() {
            assert!(v.is_finite(), "SDPA output must be finite; got {v}");
            if v != 0.0 {
                any_nonzero = true;
            }
        }
        assert!(
            any_nonzero,
            "SDPA output must be non-zero (kernel produced no writes)"
        );
    }

    #[test]
    fn dispatch_tq_sdpa_two_position_kv_finite_output() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Encode TWO KV positions then dispatch SDPA with kv_seq_len=2.
        // Output must remain finite + non-zero at qwen35 shape.
        // Pins regression that the kernel correctly handles
        // multi-position KV cache reads.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 16;
        let mut cache = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];
        let n_kv_heads = cfg.num_key_value_heads;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;

        let k0 = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 100);
        let v0 = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 200);
        let k1 = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 300);
        let v1 = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 400);

        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        slot.encode_token_to_tq(
            &k0,
            &v0,
            n_kv_heads,
            head_dim,
            cache_capacity,
            0,
            false,
            1.0,
            8,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("encode pos 0");
        slot.encode_token_to_tq(
            &k1,
            &v1,
            n_kv_heads,
            head_dim,
            cache_capacity,
            1,
            false,
            1.0,
            8,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("encode pos 1");
        encoder.memory_barrier();

        let (mut q_buf, output, tmp) = alloc_sdpa_buffers(&device, num_heads, head_dim);
        {
            let s = q_buf.as_mut_slice::<f32>().expect("q mut");
            for (i, v) in s.iter_mut().enumerate() {
                *v = ((i as f32) * 0.0017).sin() * 0.5;
            }
        }
        let params = Qwen35TqSdpaParams {
            num_heads,
            num_kv_heads: n_kv_heads,
            head_dim,
            kv_seq_len: 2,
            kv_capacity: cache_capacity,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
        };

        slot.dispatch_tq_sdpa(
            &q_buf,
            &output,
            &tmp,
            &params,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("dispatch_tq_sdpa");
        encoder.commit_and_wait().expect("commit_and_wait");

        let out = output.as_slice::<f32>().expect("output slice");
        let mut any_nonzero = false;
        for &v in out.iter() {
            assert!(
                v.is_finite(),
                "SDPA output must be finite at kv_seq_len=2; got {v}"
            );
            if v != 0.0 {
                any_nonzero = true;
            }
        }
        assert!(any_nonzero, "kv_seq_len=2 SDPA output must be non-zero");
    }

    #[test]
    fn dispatch_tq_sdpa_rejects_kv_seq_len_zero() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Defensive: kernel param validation propagates through the
        // wrapper.  kv_seq_len=0 must fail loud (kernel
        // validate_params rejects).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache = HybridKvCache::new_with_options(&cfg, &device, 64, 1, true).expect("kv tq-on");
        let slot = &cache.full_attn[0];
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let (q, output, tmp) = alloc_sdpa_buffers(&device, num_heads, head_dim);
        let params = Qwen35TqSdpaParams {
            num_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            kv_seq_len: 0, // invalid
            kv_capacity: 64,
            scale: 1.0,
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
        };
        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        let err = slot
            .dispatch_tq_sdpa(
                &q,
                &output,
                &tmp,
                &params,
                &mut encoder,
                &mut registry,
                &device,
            )
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("kv_seq_len must be > 0"),
            "expected kv_seq_len-zero validation error, got: {msg}"
        );
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-11 — NRMSE-vs-F32 baseline parity (the litmus)
    // ──────────────────────────────────────────────────────────────────

    /// **D1 SRHT sign table for D=256.** Verbatim from
    /// `mlx-native/src/shaders/hadamard_quantize_kv_fast.metal:21-26` and
    /// `fwht_standalone.metal:21-26`.  Bit `j` of the byte at
    /// `table[j>>3]` is the sign bit for element `j`: bit=1 → -1, bit=0 → +1.
    /// Both encode and Q pre-rotation use the SAME table, so attention
    /// scores after sign×FWHT round-trip equal the F32 baseline modulo
    /// quantization (sign[i]^2 = 1 cancels under Q@K^T).
    const TBQ_SIGNS_256: [u8; 32] = [
        0xa7, 0x3b, 0x91, 0xf4, 0x6d, 0xc2, 0x58, 0x0e, 0xb3, 0x7f, 0x24, 0xd6, 0x89, 0x45, 0xea,
        0x1c, 0x63, 0xaf, 0xd8, 0x52, 0x97, 0x0b, 0xe1, 0x3d, 0x76, 0xc4, 0x19, 0xfe, 0x4a, 0x85,
        0x2c, 0xdb,
    ];

    /// Apply the D1 sign pattern in-place (TBQ_SIGNS_256). Self-inverse.
    fn apply_d1_sign_d256(x: &mut [f32]) {
        assert_eq!(x.len(), 256, "D1 sign d256 requires len=256");
        for (j, v) in x.iter_mut().enumerate() {
            let sign_byte = TBQ_SIGNS_256[j >> 3];
            let bit = (sign_byte >> (j & 7)) & 1;
            if bit != 0 {
                *v = -*v;
            }
        }
    }

    /// Sign × FWHT pre-rotation (mirrors GPU `fwht_sign_premult_f32_d256`).
    /// Used to rotate Q into the same basis as the encoded K, V.
    fn sign_premult_fwht_d256(x: &mut [f32]) {
        apply_d1_sign_d256(x);
        mlx_native::turboquant::fwht_inplace(x).expect("FWHT");
    }

    /// FWHT × sign undo (mirrors GPU `fwht_sign_undo_f32_d256`).  Used
    /// to inverse-rotate the SDPA output back into the standard basis.
    fn fwht_sign_undo_d256(x: &mut [f32]) {
        mlx_native::turboquant::fwht_inplace(x).expect("FWHT undo");
        apply_d1_sign_d256(x);
    }

    /// Compute NRMSE = sqrt(sum((a-b)^2) / sum(b^2)) — relative error
    /// vs the reference signal. Mirrors `mlx_native::turboquant::tests::nrmse`.
    fn nrmse(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "NRMSE requires equal-length slices");
        let mut sum_sq_diff = 0.0_f32;
        let mut sum_sq_ref = 0.0_f32;
        for (av, bv) in a.iter().zip(b.iter()) {
            let d = av - bv;
            sum_sq_diff += d * d;
            sum_sq_ref += bv * bv;
        }
        if sum_sq_ref == 0.0 {
            return 0.0;
        }
        (sum_sq_diff / sum_sq_ref).sqrt()
    }

    /// Build a synthetic single-token K, V at qwen35 shape with non-trivial
    /// values, return (cpu_floats, gpu_buffer) so we can both upload to GPU
    /// for encoding AND compute the F32 reference SDPA on CPU.
    fn synth_token_with_cpu_mirror(
        device: &MlxDevice,
        n_kv_heads: usize,
        head_dim: usize,
        salt: u32,
    ) -> (Vec<Vec<f32>>, MlxBuffer) {
        let mut cpu: Vec<Vec<f32>> = Vec::with_capacity(n_kv_heads);
        for h in 0..n_kv_heads {
            let mut head: Vec<f32> = Vec::with_capacity(head_dim);
            for i in 0..head_dim {
                let x = ((i as u32 + h as u32 * 31 + salt) % 1000) as f32 / 1000.0;
                head.push((x * 6.28318).sin() * 0.5);
            }
            cpu.push(head);
        }
        let elems = n_kv_heads * head_dim;
        let mut buf = device
            .alloc_buffer(
                elems * std::mem::size_of::<f32>(),
                DType::F32,
                vec![n_kv_heads, head_dim],
            )
            .expect("alloc token buf");
        {
            let s = buf.as_mut_slice::<f32>().expect("token mut slice");
            for h in 0..n_kv_heads {
                for d in 0..head_dim {
                    s[h * head_dim + d] = cpu[h][d];
                }
            }
        }
        (cpu, buf)
    }

    #[test]
    fn dispatch_tq_sdpa_nrmse_vs_f32_baseline_under_threshold() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // **ITER-11 LITMUS TEST** — does the qwen35 TQ encode + GPU SDPA
        // pipeline produce numerically-correct outputs vs an F32 baseline?
        //
        // Method (kv_seq_len=1 closed-form simplification):
        // 1. Generate synthetic F32 K, V, Q at qwen35 shape.
        // 2. Upload K, V to GPU + encode via dispatch_hadamard_quantize_kv_hb
        //    (in-place FWHT + Lloyd-Max 8-bit quant). Read back packed/norms.
        // 3. Apply CPU FWHT to Q (mirrors the GPU pre-rotation that the
        //    forward path will do via dispatch_fwht_f32 in iter-12).
        // 4. Call flash_attn_vec_tq_hb_oracle (CPU F32 mirror of the GPU
        //    SDPA kernel). Output is in FWHT basis.
        // 5. Apply inverse CPU FWHT to oracle output → output_tq in
        //    standard basis.
        // 6. F32 reference at kv_seq_len=1: softmax over 1 score = 1.0 →
        //    output_ref[h] = V[kv_head(h)] (broadcast across query
        //    heads via GQA: kv_head(h) = h / heads_per_kv).
        // 7. NRMSE(output_tq, output_ref) — measures the cumulative
        //    quantization error end-to-end.
        //
        // Threshold: NRMSE < 0.15 per ADR-007 §F-0.3 (Gemma path's
        // empirically-validated TQ-vs-F32 ceiling). Failure indicates
        // a fundamental kernel-level mismatch and would falsify Phase B.
        //
        // Why kv_seq_len=1: at single-position KV, the F32 reference
        // simplifies to the cached V vector itself (softmax(scalar) = 1.0).
        // This gives a closed-form baseline without writing a full SDPA
        // CPU oracle. iter-12 extends to multi-token KV with a fuller
        // CPU SDPA reference.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 64;
        let mut cache = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];
        let n_kv_heads = cfg.num_key_value_heads;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        assert_eq!(head_dim, 256, "qwen35 production head_dim");

        // Step 1: synthetic K, V with both CPU mirrors (for reference) and
        // GPU buffers (for encoding).
        let (k_cpu, k_buf) =
            synth_token_with_cpu_mirror(&device, n_kv_heads as usize, head_dim as usize, 7);
        let (v_cpu, v_buf) =
            synth_token_with_cpu_mirror(&device, n_kv_heads as usize, head_dim as usize, 11);

        // Step 2: GPU encode at write_pos=0.
        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        slot.encode_token_to_tq(
            &k_buf,
            &v_buf,
            n_kv_heads,
            head_dim,
            cache_capacity,
            0,
            false,
            1.0,
            8,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("encode");
        encoder.commit_and_wait().expect("encode commit");

        // Step 2b: read back packed/norms to CPU.
        let tq = slot.tq.as_ref().unwrap();
        let k_packed_bytes: Vec<u8> = tq.k_packed.as_slice::<u8>().unwrap().to_vec();
        let k_norms_floats: Vec<f32> = tq.k_norms.as_slice::<f32>().unwrap().to_vec();
        let v_packed_bytes: Vec<u8> = tq.v_packed.as_slice::<u8>().unwrap().to_vec();
        let v_norms_floats: Vec<f32> = tq.v_norms.as_slice::<f32>().unwrap().to_vec();

        // Step 3: synthetic Q (n_heads × head_dim) — non-trivial values.
        let mut q_orig: Vec<Vec<f32>> = Vec::with_capacity(num_heads as usize);
        for h in 0..num_heads as usize {
            let mut head = Vec::with_capacity(head_dim as usize);
            for i in 0..head_dim as usize {
                let x = ((i + h * 17) % 1000) as f32 / 1000.0;
                head.push((x * 3.14159).cos() * 0.4);
            }
            q_orig.push(head);
        }
        // Apply D1 sign × FWHT to each head of Q (mirrors GPU
        // dispatch_fwht_sign_premult_f32 — the Q pre-rotation Gemma's
        // production path uses; iter-12 will dispatch this on GPU).
        let mut q_fwht: Vec<f32> = Vec::with_capacity((num_heads as usize) * (head_dim as usize));
        for head in &q_orig {
            let mut buf = head.clone();
            sign_premult_fwht_d256(&mut buf);
            q_fwht.extend(buf);
        }

        // Step 4: call CPU oracle.
        let oracle_params = mlx_native::tq_oracle::TqHbOracleParams {
            num_heads,
            num_kv_heads: n_kv_heads,
            head_dim,
            kv_seq_len: 1,
            kv_capacity: cache_capacity,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
        };
        let mut oracle_output = vec![0.0_f32; (num_heads as usize) * (head_dim as usize)];
        mlx_native::tq_oracle::flash_attn_vec_tq_hb_oracle(
            &q_fwht,
            &k_packed_bytes,
            &k_norms_floats,
            &v_packed_bytes,
            &v_norms_floats,
            &mut oracle_output,
            &oracle_params,
        )
        .expect("oracle");

        // Step 5: inverse rotation on oracle output (FWHT × sign undo).
        // Mirrors GPU dispatch_fwht_sign_undo_f32.
        let mut output_tq_flat = oracle_output.clone();
        for h in 0..num_heads as usize {
            let off = h * head_dim as usize;
            fwht_sign_undo_d256(&mut output_tq_flat[off..off + head_dim as usize]);
        }

        // Step 6: F32 reference at kv_seq_len=1 (closed form).
        // softmax over a single score = 1.0; output = V[kv_head(h)].
        let heads_per_kv = (num_heads / n_kv_heads) as usize;
        let mut output_ref_flat: Vec<f32> =
            Vec::with_capacity((num_heads as usize) * (head_dim as usize));
        for h in 0..num_heads as usize {
            let kv_head = h / heads_per_kv;
            output_ref_flat.extend_from_slice(&v_cpu[kv_head]);
        }

        // Step 7: NRMSE.
        let nrmse_value = nrmse(&output_tq_flat, &output_ref_flat);

        // ADR-007 §F-0.3 threshold: TQ-vs-F32 NRMSE ≤ 0.15.
        // qwen35 / qwen36 KV distribution post-FWHT must approximate
        // N(0,1) for 8-bit Lloyd-Max codebook to be accurate; threshold
        // failure = falsifies Phase B (would require per-(layer, head)
        // calibration per ADR-007 F-2 path).
        eprintln!(
            "[iter-11 NRMSE litmus] qwen35 TQ-vs-F32 NRMSE = {nrmse_value:.6} \
             (threshold 0.15)"
        );
        assert!(
            nrmse_value < 0.15,
            "iter-11 NRMSE litmus FAILED: {nrmse_value:.6} >= 0.15. \
             qwen35 TQ-on path is NOT shippable at 8-bit codebook with \
             standard FWHT. Investigate per-(layer, head) calibration \
             (ADR-007 F-2 path) before proceeding."
        );

        // Held to silence unused warnings — k_cpu retained for completeness
        // but not used (V dominates the kv_seq_len=1 closed form; iter-12
        // multi-position test uses k_cpu in the full CPU SDPA reference).
        let _ = k_cpu;
        let _ = q_orig;
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-13 — GPU end-to-end NRMSE litmus
    // ──────────────────────────────────────────────────────────────────

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-14 — encode_seq_tokens_to_tq prefill encode
    // ──────────────────────────────────────────────────────────────────

    /// Build a synthetic seq-major K (or V) buffer at qwen35 shape with
    /// deterministic non-trivial values: shape `[seq_len, n_kv_heads,
    /// head_dim]` F32. Used by both the multi-token encode test and
    /// the per-token equivalence test below.
    fn synth_seq_kv_buffer(
        device: &MlxDevice,
        seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        salt: u32,
    ) -> MlxBuffer {
        let elems = seq_len * n_kv_heads * head_dim;
        let mut buf = device
            .alloc_buffer(
                elems * std::mem::size_of::<f32>(),
                DType::F32,
                vec![seq_len, n_kv_heads, head_dim],
            )
            .expect("alloc seq kv buf");
        {
            let s = buf.as_mut_slice::<f32>().expect("seq kv mut slice");
            for t in 0..seq_len {
                for h in 0..n_kv_heads {
                    for d in 0..head_dim {
                        let i = (t * n_kv_heads + h) * head_dim + d;
                        let x = ((i as u32 + salt) % 1000) as f32 / 1000.0;
                        s[i] = (x * 6.28318).sin() * 0.5;
                    }
                }
            }
        }
        buf
    }

    #[test]
    fn encode_seq_tokens_to_tq_errors_when_slot_lacks_tq_buffers() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, 64, 1, false).expect("kv tq-off");
        let slot = &mut cache.full_attn[0];
        assert!(slot.tq.is_none());
        let seq_kv = synth_seq_kv_buffer(
            &device,
            4,
            cfg.num_key_value_heads as usize,
            cfg.head_dim as usize,
            17,
        );
        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        let err = slot
            .encode_seq_tokens_to_tq(
                &seq_kv,
                true,
                4,
                cfg.num_key_value_heads,
                cfg.head_dim,
                64,
                0,
                0,
                false,
                1.0,
                8,
                &mut encoder,
                &mut registry,
                &device,
            )
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("slot.tq is None"),
            "expected fail-loud None-tq error, got: {msg}"
        );
    }

    #[test]
    fn encode_seq_tokens_to_tq_byte_equal_to_per_token_loop() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // **iter-14 equivalence test** — proves the multi-token
        // dispatch (`dispatch_hadamard_quantize_kv_hb_seq`) produces
        // byte-identical packed/norms output to a manual per-token
        // loop calling `dispatch_hadamard_quantize_kv_hb` once per
        // position. This pins the `_seq` variant's loop semantics +
        // src_offset stride so production wiring (iter-15) can use
        // the bulk dispatch with confidence.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 32;
        let n_tokens: u32 = 5;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // Reference path: 5 separate single-token tokens encoded via
        // encode_token_to_tq into reference cache slot.
        let mut cache_ref = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv ref tq-on");
        let slot_ref = &mut cache_ref.full_attn[0];

        // Build N single-token K and V buffers (each shape
        // [n_kv_heads, head_dim]).
        let mut single_k_bufs: Vec<MlxBuffer> = Vec::new();
        let mut single_v_bufs: Vec<MlxBuffer> = Vec::new();
        for t in 0..n_tokens as usize {
            single_k_bufs.push(synth_token_buffer(
                &device,
                n_kv_heads as usize,
                head_dim as usize,
                100 + t as u32,
            ));
            single_v_bufs.push(synth_token_buffer(
                &device,
                n_kv_heads as usize,
                head_dim as usize,
                200 + t as u32,
            ));
        }

        let mut registry = mlx_native::KernelRegistry::new();
        let mut enc_ref = device.command_encoder().expect("encoder ref");
        for (t, (k_buf, v_buf)) in single_k_bufs.iter().zip(single_v_bufs.iter()).enumerate() {
            slot_ref
                .encode_token_to_tq(
                    k_buf,
                    v_buf,
                    n_kv_heads,
                    head_dim,
                    cache_capacity,
                    t as u32,
                    false,
                    1.0,
                    8,
                    &mut enc_ref,
                    &mut registry,
                    &device,
                )
                .expect("encode_token_to_tq per-token");
        }
        enc_ref.commit_and_wait().expect("ref commit");

        // Multi-token dispatch path: build a single seq-major K + V
        // buffer carrying the SAME data laid out as
        // [n_tokens, n_kv_heads, head_dim], then call
        // encode_seq_tokens_to_tq once per side.
        let mut seq_k = device
            .alloc_buffer(
                (n_tokens as usize) * (n_kv_heads as usize) * (head_dim as usize) * 4,
                DType::F32,
                vec![n_tokens as usize, n_kv_heads as usize, head_dim as usize],
            )
            .expect("alloc seq_k");
        let mut seq_v = device
            .alloc_buffer(
                (n_tokens as usize) * (n_kv_heads as usize) * (head_dim as usize) * 4,
                DType::F32,
                vec![n_tokens as usize, n_kv_heads as usize, head_dim as usize],
            )
            .expect("alloc seq_v");
        {
            let dst_k = seq_k.as_mut_slice::<f32>().expect("seq_k mut");
            let dst_v = seq_v.as_mut_slice::<f32>().expect("seq_v mut");
            let stride = (n_kv_heads as usize) * (head_dim as usize);
            for t in 0..n_tokens as usize {
                let k_src = single_k_bufs[t].as_slice::<f32>().expect("k src");
                let v_src = single_v_bufs[t].as_slice::<f32>().expect("v src");
                dst_k[t * stride..(t + 1) * stride].copy_from_slice(k_src);
                dst_v[t * stride..(t + 1) * stride].copy_from_slice(v_src);
            }
        }

        let mut cache_seq = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv seq tq-on");
        let slot_seq = &mut cache_seq.full_attn[0];
        let mut enc_seq = device.command_encoder().expect("encoder seq");
        slot_seq
            .encode_seq_tokens_to_tq(
                &seq_k,
                true,
                n_tokens,
                n_kv_heads,
                head_dim,
                cache_capacity,
                0,
                0,
                false,
                1.0,
                8,
                &mut enc_seq,
                &mut registry,
                &device,
            )
            .expect("encode K seq");
        slot_seq
            .encode_seq_tokens_to_tq(
                &seq_v,
                false,
                n_tokens,
                n_kv_heads,
                head_dim,
                cache_capacity,
                0,
                0,
                false,
                1.0,
                8,
                &mut enc_seq,
                &mut registry,
                &device,
            )
            .expect("encode V seq");
        enc_seq.commit_and_wait().expect("seq commit");

        // Byte-equal comparison: per-token loop and bulk _seq must
        // produce identical packed + norms bytes.
        let tq_ref = slot_ref.tq.as_ref().unwrap();
        let tq_seq = slot_seq.tq.as_ref().unwrap();
        assert_eq!(
            tq_ref.k_packed.as_slice::<u8>().unwrap(),
            tq_seq.k_packed.as_slice::<u8>().unwrap(),
            "k_packed bytes diverge between per-token loop and _seq dispatch"
        );
        assert_eq!(
            tq_ref.k_norms.as_slice::<f32>().unwrap(),
            tq_seq.k_norms.as_slice::<f32>().unwrap(),
            "k_norms bytes diverge between per-token loop and _seq dispatch"
        );
        assert_eq!(
            tq_ref.v_packed.as_slice::<u8>().unwrap(),
            tq_seq.v_packed.as_slice::<u8>().unwrap(),
            "v_packed bytes diverge between per-token loop and _seq dispatch"
        );
        assert_eq!(
            tq_ref.v_norms.as_slice::<f32>().unwrap(),
            tq_seq.v_norms.as_slice::<f32>().unwrap(),
            "v_norms bytes diverge between per-token loop and _seq dispatch"
        );
    }

    /// ADR-027 Phase B iter-31 (sub-sub-iter 23c-β.2) — `dequant_seq_to_temp_f32`
    /// shadow-cache parity test.
    ///
    /// Threads the iter-30 mlx-native parity guarantee
    /// (`tq_dequantize_hb_kv_seq_n1_byte_identical_to_per_position`)
    /// through hf2q's actual TQ encode pipeline at production cache shape.
    ///
    /// Sequence:
    /// (1) Synthesize N tokens of K, encode via `encode_seq_tokens_to_tq`
    ///     into a TQ-active slot.
    /// (2) Reference: per-position dispatch
    ///     `dispatch_tq_dequantize_hb_kv` for each position individually
    ///     into separate F32 buffers.
    /// (3) Under test: `dequant_seq_to_temp_f32` for the entire range
    ///     `[0..N)` in one call.
    /// (4) Byte-equal compare: per-position outputs[h, :] vs
    ///     seq output[h, t, :] for each (h, t) pair.
    ///
    /// Without this contract, iter-32's prefill SDPA wiring (which reads
    /// `dequant_seq_to_temp_f32` output) would risk silent drift vs the
    /// shadow-cache F32 baseline — the cross-axis sweep harness is too
    /// coarse to catch a per-(h,t)-position dequant bug.
    #[test]
    fn dequant_seq_to_temp_f32_byte_equal_to_per_position_dispatch() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 32;
        let n_tokens: u32 = 6;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        assert_eq!(head_dim, 256);

        // Build a TQ-active cache + encode N tokens of K into slot 0
        // via the production seq-encode path.
        let mut cache = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];

        // Build a [n_tokens, n_kv_heads, head_dim] f32 source buffer
        // with deterministic non-trivial values. Same shape the
        // production path passes to encode_seq_tokens_to_tq.
        let stride = (n_kv_heads as usize) * (head_dim as usize);
        let total_elems = (n_tokens as usize) * stride;
        let mut seq_k = device
            .alloc_buffer(
                total_elems * 4,
                DType::F32,
                vec![n_tokens as usize, n_kv_heads as usize, head_dim as usize],
            )
            .expect("alloc seq_k");
        {
            let dst = seq_k.as_mut_slice::<f32>().expect("seq_k mut");
            for t in 0..n_tokens as usize {
                for h in 0..n_kv_heads as usize {
                    for d in 0..head_dim as usize {
                        // Deterministic pattern with non-trivial inter-
                        // position variance so dequant correctness can
                        // be observed per (t, h, d).
                        let v = ((t * 31 + h * 17 + d) as f32 / 137.0).sin() * 0.4;
                        dst[t * stride + h * head_dim as usize + d] = v;
                    }
                }
            }
        }

        let mut registry = mlx_native::KernelRegistry::new();

        // Encode K seq into TQ.
        let mut enc = device.command_encoder().expect("encoder");
        slot.encode_seq_tokens_to_tq(
            &seq_k,
            /*is_k=*/ true,
            n_tokens,
            n_kv_heads,
            head_dim,
            cache_capacity,
            /*write_pos=*/ 0,
            /*src_tok_offset=*/ 0,
            /*sliding=*/ false,
            /*scale_factor_d512=*/ 1.0,
            /*codebook_bits=*/ 8,
            &mut enc,
            &mut registry,
            &device,
        )
        .expect("encode K seq");
        enc.commit_and_wait().expect("encode commit");

        // Reference: per-position dispatch into separate buffers.
        let mut ref_per_pos: Vec<MlxBuffer> = Vec::with_capacity(n_tokens as usize);
        for _ in 0..n_tokens as usize {
            ref_per_pos.push(
                device
                    .alloc_buffer(
                        (n_kv_heads as usize) * (head_dim as usize) * 4,
                        DType::F32,
                        vec![n_kv_heads as usize, head_dim as usize],
                    )
                    .expect("alloc ref_per_pos"),
            );
        }
        {
            let mut enc = device.command_encoder().expect("encoder ref");
            let tq = slot.tq.as_ref().unwrap();
            for t in 0..n_tokens {
                mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv(
                    &mut enc,
                    &mut registry,
                    device.metal_device(),
                    &tq.k_packed,
                    &tq.k_norms,
                    &ref_per_pos[t as usize],
                    n_kv_heads,
                    head_dim,
                    cache_capacity,
                    /*read_pos=*/ t,
                    /*scale_factor_d512=*/ 1.0,
                    /*codebook_bits=*/ 8,
                )
                .expect("per-pos dispatch");
            }
            enc.commit_and_wait().expect("ref commit");
        }

        // Under test: dequant_seq_to_temp_f32 for the entire range
        // [0..n_tokens). Output shape [n_kv_heads, n_tokens, head_dim].
        let temp_f32 = {
            let mut enc = device.command_encoder().expect("encoder seq");
            let buf = slot
                .dequant_seq_to_temp_f32(
                    /*is_k=*/ true,
                    n_tokens,
                    /*start_pos=*/ 0,
                    cache_capacity,
                    n_kv_heads,
                    head_dim,
                    &mut enc,
                    &mut registry,
                    &device,
                )
                .expect("dequant_seq_to_temp_f32");
            enc.commit_and_wait().expect("seq commit");
            buf
        };
        assert_eq!(
            temp_f32.element_count(),
            (n_kv_heads as usize) * (n_tokens as usize) * (head_dim as usize),
            "temp_f32 element count must equal nkv × n_tokens × head_dim"
        );

        // Byte-equal compare per (h, t) chunk.
        let seq_slice = temp_f32.as_slice::<f32>().expect("temp_f32 slice");
        for h in 0..n_kv_heads {
            for t in 0..n_tokens {
                // Reference layout: ref_per_pos[t][h, 0..hd].
                let pp_slice = ref_per_pos[t as usize].as_slice::<f32>().expect("pp slice");
                let pp_off = (h as usize) * (head_dim as usize);
                let pp = &pp_slice[pp_off..pp_off + head_dim as usize];

                // Seq output layout: temp_f32[h, t, 0..hd]
                // = seq_slice[h * n_tokens * hd + t * hd + 0..hd].
                let seq_off = (h as usize) * (n_tokens as usize) * (head_dim as usize)
                    + (t as usize) * (head_dim as usize);
                let s = &seq_slice[seq_off..seq_off + head_dim as usize];

                assert_eq!(
                    pp, s,
                    "h={h} t={t}: dequant_seq output diverges from per-position \
                     dispatch — iter-32 prefill wiring would silently drift."
                );
            }
        }
    }

    /// ADR-027 Phase B iter-32 (sub-sub-iter 23c-β.3) —
    /// `dequant_seq_to_temp_f32_unrotated` round-trip recovery test.
    ///
    /// **Round-trip property:** for any K written via
    /// `encode_seq_tokens_to_tq`, the dequant + FWHT-undo + sign-undo
    /// chain recovers K to within the quant round-trip floor (iter-13
    /// measured NRMSE 0.008 on single-position; this test validates
    /// the seq variant under the same 0.15 ADR-007 §F-0.3 threshold
    /// at production cache shape: cfg=moe_cfg_40layer, n_tokens=6,
    /// head_dim=256).
    ///
    /// Without this contract, iter-33's drop-in replacement of
    /// `slot.k.as_ref()` with `dequant_seq_to_temp_f32_unrotated`
    /// output would silently degrade dense prefill SDPA accuracy.
    /// This test is the load-bearing parity gate.
    ///
    /// Sequence:
    /// (1) Build TQ-active cache + synthesize N tokens of F32 K.
    /// (2) Encode K via `encode_seq_tokens_to_tq` (writes TQ buffers).
    /// (3) `dequant_seq_to_temp_f32_unrotated` reads TQ + un-rotates.
    /// (4) Download both original F32 K and recovered K to CPU; compute
    ///     NRMSE per (kv_head, token, dim) flattened.
    /// (5) Assert NRMSE < 0.15 (ADR-007 §F-0.3 threshold).
    #[test]
    fn dequant_seq_to_temp_f32_unrotated_recovers_original_within_nrmse_threshold() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 32;
        let n_tokens: u32 = 6;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        assert_eq!(head_dim, 256);

        let mut cache = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];

        // Build a [n_tokens, n_kv_heads, head_dim] f32 K source with
        // deterministic values that span both signs and magnitudes
        // (so the quant codebook coverage is exercised).
        let stride = (n_kv_heads as usize) * (head_dim as usize);
        let total_elems = (n_tokens as usize) * stride;
        let mut k_orig_cpu = vec![0f32; total_elems];
        for t in 0..n_tokens as usize {
            for h in 0..n_kv_heads as usize {
                for d in 0..head_dim as usize {
                    let v = ((t * 31 + h * 17 + d) as f32 / 137.0).sin() * 0.4
                        + ((t + h + d) as f32 * 0.0011).cos() * 0.15;
                    k_orig_cpu[t * stride + h * head_dim as usize + d] = v;
                }
            }
        }
        let mut seq_k = device
            .alloc_buffer(
                total_elems * 4,
                DType::F32,
                vec![n_tokens as usize, n_kv_heads as usize, head_dim as usize],
            )
            .expect("alloc seq_k");
        seq_k
            .as_mut_slice::<f32>()
            .expect("seq_k mut")
            .copy_from_slice(&k_orig_cpu);

        let mut registry = mlx_native::KernelRegistry::new();

        // Encode K seq into TQ.
        {
            let mut enc = device.command_encoder().expect("encoder");
            slot.encode_seq_tokens_to_tq(
                &seq_k,
                /*is_k=*/ true,
                n_tokens,
                n_kv_heads,
                head_dim,
                cache_capacity,
                /*write_pos=*/ 0,
                /*src_tok_offset=*/ 0,
                /*sliding=*/ false,
                /*scale_factor_d512=*/ 1.0,
                /*codebook_bits=*/ 8,
                &mut enc,
                &mut registry,
                &device,
            )
            .expect("encode K seq");
            enc.commit_and_wait().expect("encode commit");
        }

        // Dequant + un-rotate via the iter-32 helper.
        let recovered = {
            let mut enc = device.command_encoder().expect("encoder dequant");
            let buf = slot
                .dequant_seq_to_temp_f32_unrotated(
                    /*is_k=*/ true,
                    n_tokens,
                    /*start_pos=*/ 0,
                    cache_capacity,
                    n_kv_heads,
                    head_dim,
                    &mut enc,
                    &mut registry,
                    &device,
                )
                .expect("dequant_seq_to_temp_f32_unrotated");
            enc.commit_and_wait().expect("dequant commit");
            buf
        };

        // Output layout: [n_kv_heads, n_tokens, head_dim].
        // Reference (k_orig_cpu) layout: [n_tokens, n_kv_heads, head_dim].
        // Permute to compare.
        let recovered_slice = recovered.as_slice::<f32>().expect("recovered slice");
        let mut recovered_seq_major = vec![0f32; total_elems];
        for h in 0..n_kv_heads as usize {
            for t in 0..n_tokens as usize {
                for d in 0..head_dim as usize {
                    let head_major_off =
                        h * (n_tokens as usize) * (head_dim as usize) + t * (head_dim as usize) + d;
                    let seq_major_off = t * stride + h * (head_dim as usize) + d;
                    recovered_seq_major[seq_major_off] = recovered_slice[head_major_off];
                }
            }
        }

        // NRMSE between original K and recovered K.
        let nrmse_value = nrmse(&recovered_seq_major, &k_orig_cpu);
        assert!(
            nrmse_value < 0.15,
            "TQ round-trip NRMSE {nrmse_value:.6} >= 0.15 (ADR-007 §F-0.3 threshold)"
        );
        // Iter-13 single-position measured 0.008. Seq variant should be in
        // the same ballpark — failing this is a regression signal even if
        // technically under threshold.
        eprintln!("[iter-32 round-trip NRMSE] {nrmse_value:.6} (iter-13 single-pos: ~0.008)");
    }

    /// ADR-027 Phase B iter-33 (sub-sub-iter 23c-β.4) — TQ-cache-backed
    /// prefill resume parity vs F32-shadow-cache prefill resume.
    ///
    /// **Load-bearing test for iter-34's F32 alloc-drop.** When iter-34
    /// makes `slot.k = None` in TQ-active mode and the production call
    /// site at `gpu_full_attn::apply_sdpa_with_kv_cache:2382+` routes
    /// prefill resume through
    /// `apply_flash_attn_prefill_seq_major_resume_via_tq_cache` (this
    /// iter's helper, defined in `gpu_full_attn.rs`), the cross-axis
    /// sweep harness depends on the resulting prefill output matching
    /// F32 baseline within the quant round-trip floor. This test pins
    /// that contract at production cache shape (cfg=moe_cfg_40layer,
    /// head_dim=256, n_kv_heads=2, n_heads=16) BEFORE iter-34 lands.
    ///
    /// Sequence:
    /// (1) Build TQ-active cache (both F32 K/V and TQ allocated in
    ///     shadow-cache mode).
    /// (2) Synthesize K, V seq-major source `[n_tokens=24, n_kv_heads,
    ///     head_dim]`.
    /// (3) Permute seq-major → head-major and write into slot.k /
    ///     slot.v at positions [0..24) (manually populates the F32 path).
    /// (4) `encode_seq_tokens_to_tq` writes K, V (seq-major source)
    ///     into slot.tq for positions [0..24).
    /// (5) Synthesize Q chunk `[seq_len=8, n_heads=16, head_dim=256]`.
    /// (6) Path A (REFERENCE):
    ///     `gpu_full_attn::apply_flash_attn_prefill_seq_major_resume`
    ///     on (Q, slot.k, slot.v, ..) → out_a.
    /// (7) Path B (UNDER TEST):
    ///     `gpu_full_attn::apply_flash_attn_prefill_seq_major_resume_via_tq_cache`
    ///     on (slot, Q, ..) → out_b.
    /// (8) NRMSE(out_a, out_b) < 0.15 (ADR-007 §F-0.3 threshold).
    #[test]
    fn apply_flash_attn_prefill_seq_major_resume_via_tq_cache_nrmse_vs_f32() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use super::super::gpu_full_attn::{
            apply_flash_attn_prefill_seq_major_resume,
            apply_flash_attn_prefill_seq_major_resume_via_tq_cache,
            apply_tq_prefill_seq_major_resume_direct,
        };
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 64;
        let n_tokens: u32 = 24;
        let chunk2_seq_len: u32 = 8;
        let cur_len: u32 = n_tokens - chunk2_seq_len;
        let n_kv_heads = cfg.num_key_value_heads;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        assert_eq!(head_dim, 256);

        let mut cache = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv tq-on");

        // Synthesize K, V seq-major.
        let stride_seq = (n_kv_heads as usize) * (head_dim as usize);
        let kv_total_elems = (n_tokens as usize) * stride_seq;
        let mut k_seq_major_cpu = vec![0f32; kv_total_elems];
        let mut v_seq_major_cpu = vec![0f32; kv_total_elems];
        for t in 0..n_tokens as usize {
            for h in 0..n_kv_heads as usize {
                for d in 0..head_dim as usize {
                    let off = t * stride_seq + h * head_dim as usize + d;
                    k_seq_major_cpu[off] = ((t * 31 + h * 17 + d * 7) as f32 / 137.0).sin() * 0.4;
                    v_seq_major_cpu[off] = ((t * 13 + h * 23 + d * 5) as f32 / 211.0).cos() * 0.35;
                }
            }
        }

        // iter-34 (sub-sub-iter 23c-β.5): slot.k/v are None when
        // tq_kv_active=true (the F32 alloc was dropped for the 3.94×
        // memory savings). For Path A (the F32 reference path) we
        // allocate F32 K/V buffers LOCALLY at full slot capacity
        // shape `[1, n_kv_heads, max_seq_len, head_dim]` and populate
        // them with the same source data the TQ path encodes from.
        let cap = cache_capacity as usize;
        let f32_kv_elems = (n_kv_heads as usize) * cap * (head_dim as usize);
        let mut local_k_f32 = device
            .alloc_buffer(
                f32_kv_elems * 4,
                DType::F32,
                vec![1, n_kv_heads as usize, cap, head_dim as usize],
            )
            .expect("alloc local F32 K (Path A reference)");
        let mut local_v_f32 = device
            .alloc_buffer(
                f32_kv_elems * 4,
                DType::F32,
                vec![1, n_kv_heads as usize, cap, head_dim as usize],
            )
            .expect("alloc local F32 V (Path A reference)");
        {
            let dst_k = local_k_f32.as_mut_slice::<f32>().expect("local k mut");
            let dst_v = local_v_f32.as_mut_slice::<f32>().expect("local v mut");
            // Zero the unused [n_tokens..max_seq_len) tail so the kernel
            // attends over a well-defined region.
            for v in dst_k.iter_mut() {
                *v = 0.0;
            }
            for v in dst_v.iter_mut() {
                *v = 0.0;
            }
            for h in 0..n_kv_heads as usize {
                for t in 0..n_tokens as usize {
                    for d in 0..head_dim as usize {
                        let src_off = t * stride_seq + h * head_dim as usize + d;
                        let dst_off = h * cap * head_dim as usize + t * head_dim as usize + d;
                        dst_k[dst_off] = k_seq_major_cpu[src_off];
                        dst_v[dst_off] = v_seq_major_cpu[src_off];
                    }
                }
            }
        }

        // Encode K, V into slot.tq via the seq-batch encoder.
        let mut seq_k = device
            .alloc_buffer(
                kv_total_elems * 4,
                DType::F32,
                vec![n_tokens as usize, n_kv_heads as usize, head_dim as usize],
            )
            .expect("alloc seq_k");
        let mut seq_v = device
            .alloc_buffer(
                kv_total_elems * 4,
                DType::F32,
                vec![n_tokens as usize, n_kv_heads as usize, head_dim as usize],
            )
            .expect("alloc seq_v");
        seq_k
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&k_seq_major_cpu);
        seq_v
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&v_seq_major_cpu);

        let mut registry = mlx_native::KernelRegistry::new();
        // The flash-attn-prefill kernel entry points are registered
        // separately from the default registry — match production
        // (forward_gpu.rs:1874).
        mlx_native::ops::flash_attn_prefill::register(&mut registry);
        {
            let slot = &mut cache.full_attn[0];
            let mut enc = device.command_encoder().expect("encoder");
            slot.encode_seq_tokens_to_tq(
                &seq_k,
                true,
                n_tokens,
                n_kv_heads,
                head_dim,
                cache_capacity,
                0,
                0,
                false,
                1.0,
                8,
                &mut enc,
                &mut registry,
                &device,
            )
            .expect("encode K seq");
            slot.encode_seq_tokens_to_tq(
                &seq_v,
                false,
                n_tokens,
                n_kv_heads,
                head_dim,
                cache_capacity,
                0,
                0,
                false,
                1.0,
                8,
                &mut enc,
                &mut registry,
                &device,
            )
            .expect("encode V seq");
            enc.commit_and_wait().expect("encode commit");
        }

        // Synthesize Q chunk.
        let q_total_elems = (chunk2_seq_len as usize) * (n_heads as usize) * (head_dim as usize);
        let mut q_cpu = vec![0f32; q_total_elems];
        for t in 0..chunk2_seq_len as usize {
            for h in 0..n_heads as usize {
                for d in 0..head_dim as usize {
                    let off =
                        t * (n_heads as usize) * (head_dim as usize) + h * head_dim as usize + d;
                    q_cpu[off] = ((t * 19 + h * 11 + d * 3) as f32 / 173.0).sin() * 0.3;
                }
            }
        }
        let mut q_gpu = device
            .alloc_buffer(
                q_total_elems * 4,
                DType::F32,
                vec![chunk2_seq_len as usize, n_heads as usize, head_dim as usize],
            )
            .expect("alloc q");
        q_gpu.as_mut_slice::<f32>().unwrap().copy_from_slice(&q_cpu);

        // Path A (REFERENCE): F32 prefill resume reading from
        // locally-allocated F32 K/V (slot.k/v are None in iter-34's
        // TQ-only mode).
        let out_a = apply_flash_attn_prefill_seq_major_resume(
            &device,
            &mut registry,
            &q_gpu,
            &local_k_f32,
            &local_v_f32,
            chunk2_seq_len,
            cur_len,
            n_tokens,
            cache_capacity,
            n_heads,
            n_kv_heads,
            head_dim,
        )
        .expect("F32 prefill resume");

        let slot_ref = &cache.full_attn[0];
        // iter-34 invariant pin: in tq_kv_active=true mode the slot's
        // F32 K/V are dropped at alloc time.
        assert!(
            slot_ref.k.is_none(),
            "iter-34: slot.k must be None when tq_kv_active=true"
        );
        assert!(slot_ref.v.is_none(), "iter-34: slot.v must be None");

        // Path B (UNDER TEST).
        let out_b = apply_flash_attn_prefill_seq_major_resume_via_tq_cache(
            &device,
            &mut registry,
            slot_ref,
            &q_gpu,
            chunk2_seq_len,
            cur_len,
            n_tokens,
            cache_capacity,
            n_heads,
            n_kv_heads,
            head_dim,
        )
        .expect("TQ-cache prefill resume");

        // Path C (UNDER TEST): direct byte-packed TQ attention. The helper
        // deliberately commits without a host wait; this explicit terminal
        // drain proves both its output and pool-retained scratch lifetimes.
        let out_c = apply_tq_prefill_seq_major_resume_direct(
            &device,
            &mut registry,
            slot_ref,
            &q_gpu,
            chunk2_seq_len,
            cur_len,
            n_tokens,
            cache_capacity,
            n_heads,
            n_kv_heads,
            head_dim,
        )
        .expect("direct TQ-cache prefill resume");
        device
            .command_encoder()
            .expect("direct TQ terminal encoder")
            .commit_and_wait()
            .expect("direct TQ terminal wait");

        // NRMSE.
        let a = out_a.as_slice::<f32>().expect("out_a slice");
        let b = out_b.as_slice::<f32>().expect("out_b slice");
        assert_eq!(a.len(), b.len(), "out_a / out_b element count mismatch");
        let mut sum_sq_diff = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for (av, bv) in a.iter().zip(b.iter()) {
            let diff = (*av - *bv) as f64;
            sum_sq_diff += diff * diff;
            sum_sq_ref += (*av as f64) * (*av as f64);
        }
        let nrmse_value = (sum_sq_diff / sum_sq_ref.max(1e-30)).sqrt() as f32;
        assert!(
            nrmse_value < 0.15,
            "TQ-cache prefill resume NRMSE {nrmse_value:.6} >= 0.15 \
             (ADR-007 §F-0.3 threshold)"
        );
        eprintln!(
            "[iter-33 prefill resume NRMSE F32 vs TQ-cache] {nrmse_value:.6} \
             (cur_len={cur_len}, kv_seq={n_tokens}, qL={chunk2_seq_len})"
        );

        let c = out_c.as_slice::<f32>().expect("out_c slice");
        assert_eq!(a.len(), c.len(), "out_a / out_c element count mismatch");
        let mut direct_sum_sq_diff = 0.0f64;
        for (av, cv) in a.iter().zip(c.iter()) {
            let diff = (*av - *cv) as f64;
            direct_sum_sq_diff += diff * diff;
        }
        let direct_nrmse = (direct_sum_sq_diff / sum_sq_ref.max(1e-30)).sqrt() as f32;
        assert!(
            direct_nrmse < 0.15,
            "direct TQ-cache prefill resume NRMSE {direct_nrmse:.6} >= 0.15 \
             (ADR-007 §F-0.3 threshold)"
        );
        assert!(
            c.iter().all(|value| value.is_finite()),
            "direct TQ-cache prefill produced a non-finite output"
        );
        eprintln!(
            "[direct TQ prefill resume NRMSE F32 vs byte-packed] {direct_nrmse:.6} \
             (cur_len={cur_len}, kv_seq={n_tokens}, qL={chunk2_seq_len})"
        );
    }

    /// ADR-027 Phase B iter-33 — defensive: TQ-cache helper errors loud
    /// when caller passes a slot constructed without TQ buffers.
    #[test]
    fn apply_flash_attn_prefill_seq_major_resume_via_tq_cache_errors_when_slot_lacks_tq() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use super::super::gpu_full_attn::apply_flash_attn_prefill_seq_major_resume_via_tq_cache;
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache =
            HybridKvCache::new_with_options(&cfg, &device, 64, 1, false).expect("kv tq-off");
        let slot = &cache.full_attn[0];
        assert!(slot.tq.is_none(), "test precondition: slot.tq must be None");

        let q_gpu = device
            .alloc_buffer(
                8 * cfg.num_attention_heads as usize * cfg.head_dim as usize * 4,
                DType::F32,
                vec![8, cfg.num_attention_heads as usize, cfg.head_dim as usize],
            )
            .expect("alloc q");

        let mut registry = mlx_native::KernelRegistry::new();
        let res = apply_flash_attn_prefill_seq_major_resume_via_tq_cache(
            &device,
            &mut registry,
            slot,
            &q_gpu,
            8,
            16,
            24,
            64,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        assert!(res.is_err(), "must error when slot.tq is None");
        let msg = format!("{:?}", res.err().unwrap());
        assert!(
            msg.contains("slot.tq is None"),
            "error msg must mention slot.tq is None, got: {msg}"
        );
    }

    /// ADR-027 Phase B iter-31 — defensive: helper errors loud when
    /// caller passes a slot constructed without TQ buffers (mantra:
    /// no fallback, no stub — Result::Err with clear context).
    #[test]
    fn dequant_seq_to_temp_f32_errors_when_slot_lacks_tq_buffers() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        // tq_kv_active=false: slot.tq is None.
        let cache =
            HybridKvCache::new_with_options(&cfg, &device, 32, 1, false).expect("kv tq-off");
        let slot = &cache.full_attn[0];
        assert!(slot.tq.is_none(), "test precondition: slot.tq must be None");

        let mut registry = mlx_native::KernelRegistry::new();
        let mut enc = device.command_encoder().expect("encoder");
        let res = slot.dequant_seq_to_temp_f32(
            true,
            1,
            0,
            32,
            cfg.num_key_value_heads,
            cfg.head_dim,
            &mut enc,
            &mut registry,
            &device,
        );
        assert!(res.is_err(), "must error when slot.tq is None");
        let msg = format!("{:?}", res.err().unwrap());
        assert!(
            msg.contains("slot.tq is None"),
            "error msg must mention slot.tq is None, got: {msg}"
        );
    }

    #[test]
    fn encode_seq_tokens_to_tq_with_src_tok_offset_skips_leading_tokens() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Defensive: src_tok_offset > 0 must skip leading source tokens
        // (matches dispatch_hadamard_quantize_kv_seq semantics for the
        // 4-bit path). Encode tokens [2, 3] of a 5-token source into
        // cache slots [0, 1] — slot[0,1] should match a per-token
        // encode of source positions [2, 3].
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 16;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let total_src_tokens: u32 = 5;
        let n_tokens_to_encode: u32 = 2;
        let src_tok_offset: u32 = 2;

        // Build source seq buffer (5 tokens).
        let seq_k = synth_seq_kv_buffer(
            &device,
            total_src_tokens as usize,
            n_kv_heads as usize,
            head_dim as usize,
            333,
        );

        // Reference: encode tokens [2, 3] via per-token loop using
        // single-token buffers extracted from positions 2 and 3.
        let mut cache_ref = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv ref");
        let slot_ref = &mut cache_ref.full_attn[0];
        let mut registry = mlx_native::KernelRegistry::new();
        let mut enc_ref = device.command_encoder().expect("encoder ref");
        let stride = (n_kv_heads as usize) * (head_dim as usize);
        // MLX_UNRETAINED_REFS=1 matches production command-buffer ownership:
        // every dispatch input must outlive the command buffer. Keep the
        // extracted per-token sources alive until `enc_ref` completes.
        let mut token_sources = Vec::with_capacity(n_tokens_to_encode as usize);
        for (cache_slot, src_pos) in
            (src_tok_offset..src_tok_offset + n_tokens_to_encode).enumerate()
        {
            let mut tok_buf = device
                .alloc_buffer(
                    stride * 4,
                    DType::F32,
                    vec![n_kv_heads as usize, head_dim as usize],
                )
                .expect("alloc tok");
            {
                let dst = tok_buf.as_mut_slice::<f32>().expect("tok mut");
                let src_slice = seq_k.as_slice::<f32>().expect("seq_k slice");
                let src_offset = (src_pos as usize) * stride;
                dst.copy_from_slice(&src_slice[src_offset..src_offset + stride]);
            }
            // Use the same buffer for K + V (test only cares about K side).
            slot_ref
                .encode_token_to_tq(
                    &tok_buf,
                    &tok_buf,
                    n_kv_heads,
                    head_dim,
                    cache_capacity,
                    cache_slot as u32,
                    false,
                    1.0,
                    8,
                    &mut enc_ref,
                    &mut registry,
                    &device,
                )
                .expect("encode token");
            token_sources.push(tok_buf);
        }
        enc_ref.commit_and_wait().expect("ref commit");

        // Test path: encode_seq_tokens_to_tq with src_tok_offset=2.
        let mut cache_seq = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv seq");
        let slot_seq = &mut cache_seq.full_attn[0];
        let mut enc_seq = device.command_encoder().expect("encoder seq");
        slot_seq
            .encode_seq_tokens_to_tq(
                &seq_k,
                true,
                n_tokens_to_encode,
                n_kv_heads,
                head_dim,
                cache_capacity,
                0,
                src_tok_offset,
                false,
                1.0,
                8,
                &mut enc_seq,
                &mut registry,
                &device,
            )
            .expect("encode seq K");
        enc_seq.commit_and_wait().expect("seq commit");

        // K side bytes must match.
        assert_eq!(
            slot_ref
                .tq
                .as_ref()
                .unwrap()
                .k_packed
                .as_slice::<u8>()
                .unwrap(),
            slot_seq
                .tq
                .as_ref()
                .unwrap()
                .k_packed
                .as_slice::<u8>()
                .unwrap(),
            "src_tok_offset semantics mismatch on k_packed"
        );
        assert_eq!(
            slot_ref
                .tq
                .as_ref()
                .unwrap()
                .k_norms
                .as_slice::<f32>()
                .unwrap(),
            slot_seq
                .tq
                .as_ref()
                .unwrap()
                .k_norms
                .as_slice::<f32>()
                .unwrap(),
            "src_tok_offset semantics mismatch on k_norms"
        );
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-18 — full-attn KV memory breakdown tests
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn full_attn_bytes_breakdown_tq_off_only_f32_at_qwen36_8k() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Default F32 path at qwen36 8K shape: every full-attn slot has
        // F32 K + V (16 MB each at 1×2×8192×256×4 = 16,777,216 bytes per
        // buffer). TQ counts must be zero (no shadow-cache when env=0).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        // At default_layer_types(40, 4), every 4th layer is full-attn:
        // layers [0, 4, 8, 12, 16, 20, 24, 28, 32, 36] = 10 full-attn slots.
        let cache =
            HybridKvCache::new_with_options(&cfg, &device, 8192, 1, false).expect("kv tq-off");
        let breakdown = cache.full_attn_bytes_breakdown();
        assert_eq!(breakdown.n_full_attn_slots, 10);
        assert!(!breakdown.has_mtp_slot, "moe_cfg_40layer has no MTP");
        // Per-slot F32 K+V = 2 * 16_777_216 = 33_554_432 bytes.
        // 10 slots × 33_554_432 = 335_544_320 bytes total.
        assert_eq!(breakdown.f32_k_v_bytes, 10 * 33_554_432);
        assert_eq!(breakdown.tq_packed_bytes, 0);
        assert_eq!(breakdown.tq_norms_bytes, 0);
        assert_eq!(breakdown.total_bytes(), 335_544_320);
        assert_eq!(breakdown.projected_iter19_savings_ratio(), None);
    }

    #[test]
    fn full_attn_bytes_breakdown_tq_on_drops_f32_at_qwen36_8k() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // ADR-027 Phase B iter-34 (sub-sub-iter 23c-β.5): TQ-only mode
        // (alloc-drop). F32 K+V backing absent (iter-34 alloc skip);
        // only TQ packed+norms allocated. Per slot:
        //   F32 K+V       = 0 (was 33_554_432 in shadow mode pre-iter-34)
        //   TQ packed K+V = 8_388_608
        //   TQ norms K+V  = 131_072
        //   Per-slot total = 8_519_680 bytes (3.94× smaller than F32-only baseline).
        // 10 slots × 8_519_680 = 85_196_800 bytes total
        // (vs pre-iter-34 shadow 420_741_120 = 4.94× shadow→TQ savings;
        //  vs F32-only baseline 335_544_320 = 3.94× absolute savings).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache =
            HybridKvCache::new_with_options(&cfg, &device, 8192, 1, true).expect("kv tq-on");
        let breakdown = cache.full_attn_bytes_breakdown();
        assert_eq!(breakdown.n_full_attn_slots, 10);
        assert!(!breakdown.has_mtp_slot);
        // **iter-34 LOAD-BEARING REGRESSION-PIN: F32 K/V alloc dropped.**
        assert_eq!(
            breakdown.f32_k_v_bytes, 0,
            "iter-34: f32_k_v_bytes MUST be 0 in TQ-only mode (alloc-drop)"
        );
        // TQ packed: 1×2×8192×256 (U8) = 4_194_304 per K, ×2 (K+V) ×10 slots.
        assert_eq!(breakdown.tq_packed_bytes, 10 * 2 * 4_194_304);
        // TQ norms: 1×2×8192×1 (F32) = 65_536 per K, ×2 (K+V) ×10 slots.
        assert_eq!(breakdown.tq_norms_bytes, 10 * 2 * 65_536);
        // Total = 0 (F32) + 83_886_080 (TQ packed) + 1_310_720 (TQ norms).
        assert_eq!(breakdown.total_bytes(), 85_196_800);
        // Pre-iter-34 shadow total reference: 420_741_120 bytes.
        // Reduction: 420_741_120 / 85_196_800 = 4.94×.
    }

    #[test]
    fn full_attn_bytes_breakdown_tq_on_drops_f32_at_qwen36_32k() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // ADR-027 Phase B iter-34 (sub-sub-iter 23c-β.5): TQ-only mode
        // at production-realistic 32K context. The dossier-quoted
        // 3.94× memory savings vs F32-only baseline:
        //   F32-only baseline (pre-Phase B): 1.34 GB (= 10 × 134_217_728)
        //   iter-34 TQ-only: 340_787_200 bytes ≈ 325 MiB
        //   Savings: 1_342_177_280 / 340_787_200 = 3.94×
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache = HybridKvCache::new_with_options(&cfg, &device, 32768, 1, true)
            .expect("kv tq-on at 32K");
        let breakdown = cache.full_attn_bytes_breakdown();
        assert_eq!(breakdown.n_full_attn_slots, 10);
        // **iter-34 LOAD-BEARING REGRESSION-PIN AT 32K SHAPE.**
        assert_eq!(
            breakdown.f32_k_v_bytes, 0,
            "iter-34 at 32K: f32_k_v_bytes MUST be 0 in TQ-only mode"
        );
        assert_eq!(breakdown.tq_packed_bytes, 10 * 33_554_432);
        assert_eq!(breakdown.tq_norms_bytes, 10 * 524_288);
        // Per-slot total: 33_554_432 + 524_288 = 34_078_720 bytes.
        // 10 slots × 34_078_720 = 340_787_200 bytes ≈ 325 MiB.
        assert_eq!(breakdown.total_bytes(), 340_787_200);
        // **The 3.94× savings claim VS F32-ONLY baseline:**
        let f32_only_baseline_per_slot: usize = 1 * 2 * 32768 * 256 * 4 * 2; // K+V
        assert_eq!(f32_only_baseline_per_slot, 134_217_728);
        let f32_only_total = 10 * f32_only_baseline_per_slot;
        assert_eq!(f32_only_total, 1_342_177_280); // 1.34 GB matches §1 ADR claim
        let savings_ratio = f32_only_total as f64 / breakdown.total_bytes() as f64;
        assert!(
            (3.93..=3.95).contains(&savings_ratio),
            "iter-34 32K F32-only→TQ-only savings: expected ~3.94×, got {savings_ratio:.4}×"
        );
    }

    #[test]
    fn full_attn_bytes_breakdown_with_mtp_includes_mtp_slot() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // ADR-027 iter-34: MTP slot ALSO drops F32 in TQ mode and
        // contributes only TQ packed+norms to the breakdown.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let mut cfg = moe_cfg_40layer();
        cfg.mtp_num_hidden_layers = 1;
        let cache = HybridKvCache::new_with_options(&cfg, &device, 1024, 1, true)
            .expect("kv tq-on with mtp");
        let breakdown = cache.full_attn_bytes_breakdown();
        assert!(breakdown.has_mtp_slot);
        // 10 regular full-attn + 1 MTP = 11 slots' worth of TQ; no F32.
        let per_slot_tq_packed = 1 * 2 * 1024 * 256 * 2;
        let per_slot_tq_norms = 1 * 2 * 1024 * 1 * 4 * 2;
        assert_eq!(breakdown.n_full_attn_slots, 10);
        // iter-34 invariant — MTP slot also dropped F32.
        assert_eq!(
            breakdown.f32_k_v_bytes, 0,
            "iter-34: MTP slot must also drop F32 K/V"
        );
        assert_eq!(breakdown.tq_packed_bytes, 11 * per_slot_tq_packed);
        assert_eq!(breakdown.tq_norms_bytes, 11 * per_slot_tq_norms);
    }

    #[test]
    fn full_attn_bytes_breakdown_tq_off_returns_no_savings_ratio() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // F32-only mode: projected_iter19_savings_ratio() must return
        // None (no TQ buffers to compare against).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache = HybridKvCache::new(&cfg, &device, 64, 1).expect("kv legacy");
        let breakdown = cache.full_attn_bytes_breakdown();
        assert!(breakdown.tq_packed_bytes == 0);
        assert!(breakdown.tq_norms_bytes == 0);
        assert_eq!(breakdown.projected_iter19_savings_ratio(), None);
    }

    #[test]
    fn dispatch_tq_sdpa_gpu_end_to_end_nrmse_vs_f32_baseline_under_threshold() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // **ITER-13 GPU LITMUS** — validates the FULL GPU chain:
        // (a) GPU encode (dispatch_hadamard_quantize_kv_hb)
        // (b) GPU Q pre-rotation (dispatch_fwht_sign_premult_f32_d256)
        // (c) GPU TQ SDPA (flash_attn_vec_tq_hb via dispatch_tq_sdpa)
        // (d) GPU output inverse-rotation (dispatch_fwht_sign_undo_f32_d256)
        //
        // Compares against the F32 closed-form reference at kv_seq_len=1
        // (output[h] = V[kv_head(h)] since softmax over a single score = 1.0).
        //
        // iter-11 proved (a)+CPU oracle correctness (NRMSE 0.008). iter-13
        // re-runs the same test using the actual GPU SDPA kernel so the
        // production wiring (iter-14) has a parity-validated path.
        //
        // Threshold: NRMSE < 0.15 per ADR-007 §F-0.3. iter-11 measured
        // 0.008 on the CPU oracle path; the GPU path SHOULD match within
        // small numerical drift (different FP rounding order).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache_capacity: u32 = 64;
        let mut cache = HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
            .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];
        let n_kv_heads = cfg.num_key_value_heads;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        assert_eq!(head_dim, 256);

        // Synthesize K, V tokens with both CPU mirrors + GPU buffers.
        let (_k_cpu, k_buf) =
            synth_token_with_cpu_mirror(&device, n_kv_heads as usize, head_dim as usize, 7);
        let (v_cpu, v_buf) =
            synth_token_with_cpu_mirror(&device, n_kv_heads as usize, head_dim as usize, 11);

        // Synthesize Q with both CPU mirror (for closed-form ref) AND
        // GPU buffer (for the GPU FWHT pre-rotation + SDPA).
        let mut q_orig: Vec<Vec<f32>> = Vec::with_capacity(num_heads as usize);
        for h in 0..num_heads as usize {
            let mut head = Vec::with_capacity(head_dim as usize);
            for i in 0..head_dim as usize {
                let x = ((i + h * 17) % 1000) as f32 / 1000.0;
                head.push((x * 3.14159).cos() * 0.4);
            }
            q_orig.push(head);
        }
        let mut q_gpu = device
            .alloc_buffer(
                (num_heads as usize) * (head_dim as usize) * 4,
                DType::F32,
                vec![num_heads as usize, head_dim as usize],
            )
            .expect("alloc q");
        {
            let s = q_gpu.as_mut_slice::<f32>().expect("q mut");
            for h in 0..num_heads as usize {
                for d in 0..head_dim as usize {
                    s[h * head_dim as usize + d] = q_orig[h][d];
                }
            }
        }

        // Output + scratch.
        let output = device
            .alloc_buffer(
                (num_heads as usize) * (head_dim as usize) * 4,
                DType::F32,
                vec![num_heads as usize, head_dim as usize],
            )
            .expect("alloc output");
        let tmp_bytes =
            mlx_native::ops::flash_attn_vec_tq_hb::tmp_buffer_bytes(num_heads, head_dim);
        let tmp = device
            .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
            .expect("alloc tmp");

        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        // (a) GPU encode K, V at write_pos=0.
        slot.encode_token_to_tq(
            &k_buf,
            &v_buf,
            n_kv_heads,
            head_dim,
            cache_capacity,
            0,
            false,
            1.0,
            8,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("encode_token_to_tq");
        encoder.memory_barrier();

        // (b) GPU Q pre-rotation: sign × FWHT (in-place on q_gpu).
        mlx_native::ops::fwht_standalone::dispatch_fwht_sign_premult_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &q_gpu,
            num_heads,
            head_dim,
        )
        .expect("fwht sign-premult Q");
        encoder.memory_barrier();

        // (c) GPU TQ SDPA dispatch.
        let params = Qwen35TqSdpaParams {
            num_heads,
            num_kv_heads: n_kv_heads,
            head_dim,
            kv_seq_len: 1,
            kv_capacity: cache_capacity,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
        };
        slot.dispatch_tq_sdpa(
            &q_gpu,
            &output,
            &tmp,
            &params,
            &mut encoder,
            &mut registry,
            &device,
        )
        .expect("dispatch_tq_sdpa");
        encoder.memory_barrier();

        // (d) GPU output inverse-rotation: FWHT × sign-undo (in-place on output).
        mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &output,
            num_heads,
            head_dim,
        )
        .expect("fwht sign-undo output");

        encoder.commit_and_wait().expect("commit chain");

        // Read GPU output to CPU + compare to F32 closed-form reference.
        let output_gpu_flat: Vec<f32> = output.as_slice::<f32>().expect("output slice").to_vec();
        let heads_per_kv = (num_heads / n_kv_heads) as usize;
        let mut output_ref_flat: Vec<f32> =
            Vec::with_capacity((num_heads as usize) * (head_dim as usize));
        for h in 0..num_heads as usize {
            let kv_head = h / heads_per_kv;
            output_ref_flat.extend_from_slice(&v_cpu[kv_head]);
        }

        let nrmse_value = nrmse(&output_gpu_flat, &output_ref_flat);
        eprintln!(
            "[iter-13 GPU NRMSE litmus] qwen35 GPU TQ-vs-F32 NRMSE = {nrmse_value:.6} \
             (threshold 0.15; iter-11 CPU oracle measured 0.008)"
        );
        assert!(
            nrmse_value < 0.15,
            "iter-13 GPU NRMSE litmus FAILED: {nrmse_value:.6} >= 0.15. \
             GPU TQ chain produces incorrect output even though CPU oracle path \
             passed at iter-11. Investigate kernel/host shape mismatch."
        );
    }

    #[test]
    fn hybrid_kv_cache_new_with_options_tq_off_with_mtp_keeps_mtp_tq_none() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Same MTP cfg but tq_kv_active=false: MTP slot has tq=None.
        // Ensures the MTP arm honors the flag identically to regular
        // full-attn slots.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let mut cfg = moe_cfg_40layer();
        cfg.mtp_num_hidden_layers = 1;
        let cache = HybridKvCache::new_with_options(&cfg, &device, 64, 1, false)
            .expect("kv tq-off with mtp");
        assert!(cache.mtp_slot.is_some());
        assert!(
            cache.mtp_slot.as_ref().unwrap().tq.is_none(),
            "MTP slot tq=None when tq_kv_active=false"
        );
    }

    #[test]
    fn tq_full_attn_buffers_alloc_shape_at_n_seqs_2() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Defensive: prove the n_seqs outer axis is honored correctly
        // (Gemma's HbKvBuffers is 3-D; qwen35's 4-D shape is the new
        // contract).  Matters for spec-decode prefill where n_seqs > 1.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let buffers =
            alloc_tq_full_attn_buffers(&cfg, &device, 64, 2).expect("alloc_tq_full_attn_buffers");
        // Expected: k_packed = [n_seqs=2, n_kv_heads=2, max_seq_len=64,
        // head_dim=256] = 2*2*64*256 = 65_536 bytes (U8).
        assert_eq!(buffers.k_packed.byte_len(), 65_536);
        assert_eq!(buffers.k_packed.shape(), &[2, 2, 64, 256]);
        // k_norms = [n_seqs=2, n_kv_heads=2, max_seq_len=64,
        // norms_per_pos=1] = 2*2*64*1 elems × 4 bytes = 1024 bytes.
        assert_eq!(buffers.k_norms.byte_len(), 1024);
        assert_eq!(buffers.k_norms.shape(), &[2, 2, 64, 1]);
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-034 task #90 Step 2 (2026-05-21) — capture_states allocator
    // + rollback_la_to tests.
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn ensure_la_capture_allocates_when_none_2026_05_21() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return, // Skip on systems without Metal.
        };
        let cfg = moe_cfg_40layer();
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("cache");
        // Pre-condition: every LA slot has capture_states = None.
        for s in &cache.linear_attn {
            assert!(
                s.capture_states.is_none(),
                "pre-ensure: capture_states must be None"
            );
        }
        cache
            .ensure_la_capture(&cfg, &device, 4)
            .expect("ensure_la_capture");
        // Post-condition: every LA slot has a properly-sized capture buffer.
        let expected_elems = (cfg.linear_key_head_dim as usize)
            * (cfg.linear_value_head_dim as usize)
            * (cfg.linear_num_value_heads as usize)
            * 4   // n_tokens_max
            * 1; // n_seqs
        for (i, s) in cache.linear_attn.iter().enumerate() {
            let buf = s
                .capture_states
                .as_ref()
                .unwrap_or_else(|| panic!("LA[{i}] capture None after ensure"));
            assert_eq!(
                buf.element_count(),
                expected_elems,
                "LA[{i}] capture element_count mismatch"
            );
            assert_eq!(buf.dtype(), DType::F32, "LA[{i}] capture must be F32");
        }
    }

    #[test]
    fn ensure_la_capture_idempotent_at_same_n_tokens_2026_05_21() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = moe_cfg_40layer();
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("cache");
        cache
            .ensure_la_capture(&cfg, &device, 4)
            .expect("first call");
        // Snapshot the buffer pointer/identity for one slot.
        let first_elems = cache.linear_attn[0]
            .capture_states
            .as_ref()
            .unwrap()
            .element_count();
        cache
            .ensure_la_capture(&cfg, &device, 4)
            .expect("second call — same size");
        let second_elems = cache.linear_attn[0]
            .capture_states
            .as_ref()
            .unwrap()
            .element_count();
        assert_eq!(
            first_elems, second_elems,
            "idempotent ensure at same n_tokens_max must preserve buffer size"
        );
    }

    #[test]
    fn ensure_la_capture_reallocs_when_larger_2026_05_21() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = moe_cfg_40layer();
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("cache");
        cache
            .ensure_la_capture(&cfg, &device, 4)
            .expect("first call");
        let first_elems = cache.linear_attn[0]
            .capture_states
            .as_ref()
            .unwrap()
            .element_count();
        cache
            .ensure_la_capture(&cfg, &device, 8)
            .expect("second call — larger");
        let second_elems = cache.linear_attn[0]
            .capture_states
            .as_ref()
            .unwrap()
            .element_count();
        assert!(
            second_elems > first_elems,
            "larger n_tokens_max must reallocate to bigger buffer"
        );
        assert_eq!(
            second_elems,
            2 * first_elems,
            "n_tokens_max=8 should double the buffer vs n_tokens_max=4"
        );
    }

    #[test]
    fn clear_la_capture_deactivates_but_retains_grow_only_storage() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = moe_cfg_40layer();
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("cache");
        cache
            .ensure_la_capture(&cfg, &device, 8)
            .expect("allocate eight-token capture storage");
        assert!(cache.la_capture_active());
        let recurrent_elems = cache.linear_attn[0]
            .capture_states
            .as_ref()
            .expect("recurrent capture")
            .element_count();
        let conv_elems = cache.linear_attn[0]
            .conv_capture_states
            .as_ref()
            .expect("conv capture")
            .element_count();

        cache.clear_la_capture();
        assert!(!cache.la_capture_active());
        assert!(cache
            .linear_attn
            .iter()
            .all(|slot| { slot.capture_states.is_some() && slot.conv_capture_states.is_some() }));

        cache
            .ensure_la_capture(&cfg, &device, 4)
            .expect("reuse larger capture storage for smaller request");
        assert!(cache.la_capture_active());
        assert_eq!(
            cache.linear_attn[0]
                .capture_states
                .as_ref()
                .expect("recurrent capture after reuse")
                .element_count(),
            recurrent_elems
        );
        assert_eq!(
            cache.linear_attn[0]
                .conv_capture_states
                .as_ref()
                .expect("conv capture after reuse")
                .element_count(),
            conv_elems
        );
    }

    #[test]
    fn ensure_la_capture_rejects_zero_2026_05_21() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = moe_cfg_40layer();
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("cache");
        assert!(
            cache.ensure_la_capture(&cfg, &device, 0).is_err(),
            "n_tokens_max=0 must reject"
        );
    }

    #[test]
    fn rollback_la_to_copies_capture_slice_2026_05_21() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = moe_cfg_40layer();
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("cache");
        cache.ensure_la_capture(&cfg, &device, 4).expect("ensure");

        // Construct a known-pattern capture buffer for LA[0]:
        // capture[i, j, h, t, s] = (t * 1000 + (i*100) + j) as f32.
        let state_elems = cache.linear_attn[0].recurrent.element_count();
        let n_tokens_max = 4usize;
        {
            let cap = cache.linear_attn[0].capture_states.as_mut().unwrap();
            let cap_slice = cap.as_mut_slice::<f32>().expect("cap mut");
            assert_eq!(cap_slice.len(), state_elems * n_tokens_max);
            for t in 0..n_tokens_max {
                for (idx, v) in cap_slice[t * state_elems..(t + 1) * state_elems]
                    .iter_mut()
                    .enumerate()
                {
                    *v = (t * 1000 + idx) as f32;
                }
            }
        }
        // Also fill LA[1] capture with a different pattern.
        {
            let cap = cache.linear_attn[1].capture_states.as_mut().unwrap();
            let cap_slice = cap.as_mut_slice::<f32>().expect("cap mut");
            for t in 0..n_tokens_max {
                for (idx, v) in cap_slice[t * state_elems..(t + 1) * state_elems]
                    .iter_mut()
                    .enumerate()
                {
                    *v = (t * 1000 + idx + 99) as f32;
                }
            }
        }

        cache
            .rollback_la_to(crate::serve::multi_seq_kv::SlotId(0), 2)
            .expect("rollback to idx=2");

        // LA[0].recurrent should now equal capture[2*state_elems..]
        let rec0 = cache.linear_attn[0]
            .recurrent
            .as_slice::<f32>()
            .expect("rec0");
        for (idx, &v) in rec0.iter().enumerate() {
            assert_eq!(
                v,
                (2 * 1000 + idx) as f32,
                "LA[0].recurrent[{idx}] after rollback to idx=2"
            );
        }
        // LA[1].recurrent should equal capture[2*state_elems..] from LA[1]'s buffer
        let rec1 = cache.linear_attn[1]
            .recurrent
            .as_slice::<f32>()
            .expect("rec1");
        for (idx, &v) in rec1.iter().enumerate() {
            assert_eq!(
                v,
                (2 * 1000 + idx + 99) as f32,
                "LA[1].recurrent[{idx}] after rollback to idx=2"
            );
        }
    }

    #[test]
    fn rollback_la_to_rejects_no_capture_2026_05_21() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = moe_cfg_40layer();
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("cache");
        // Did NOT call ensure_la_capture.
        assert!(
            cache
                .rollback_la_to(crate::serve::multi_seq_kv::SlotId(0), 0)
                .is_err(),
            "rollback without ensure_la_capture must error"
        );
    }

    #[test]
    fn rollback_la_to_rejects_out_of_range_idx_2026_05_21() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = moe_cfg_40layer();
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("cache");
        cache.ensure_la_capture(&cfg, &device, 4).expect("ensure");
        assert!(
            cache
                .rollback_la_to(crate::serve::multi_seq_kv::SlotId(0), 4)
                .is_err(),
            "rollback idx=4 with n_tokens_max=4 must error (need idx < n_tokens_max)"
        );
        assert!(
            cache
                .rollback_la_to(crate::serve::multi_seq_kv::SlotId(0), 99)
                .is_err(),
            "rollback idx=99 must error"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // ADR-040 Phase A2a iter-2a — multi-seq lift hypotheses + trait impl
    // tests.  See `docs/research/adr040-kv-cache-lift-dossier-2026-05-23.md`
    // §2.8 for H1–H5 falsification statements.
    //
    // Order in this block matches the dossier §4 iter-2a sequencing:
    //   H1 (allocator byte-scale)          — must PASS before the
    //                                        `impl MultiSeqKvCache` block
    //                                        is trusted.
    //   H2 (slot-0 byte-equivalence)       — pins ADR §5 AC-1.
    //   H3 (per-slot isolation)            — pins per-slot O(1) bound.
    //   Trait-surface pins (slot_count,    — exercise the methods directly.
    //     out-of-range, drop, fork-to-self,
    //     fork-cross-slot deferral,
    //     layout discriminant)
    //
    // H4 (recurrent-state outermost-axis stride) and H5 (gpu_delta_net.rs
    // dispatch hard-codes) are DEFERRED to Phase A2b per dossier §4 +
    // §2.10 R1 (the `rollback_la_to` guard at kv_cache.rs:1567 is the
    // real linear-attn multi-seq blocker; lifting it is not in scope
    // for Phase A2a, which is full-attn + MTP slot lift ONLY).
    // ───────────────────────────────────────────────────────────────────────

    /// Synthetic tiny dense Qwen35Config sized so n_seqs=4 allocation fits
    /// trivially on any test machine but the buffers still exercise the
    /// 4-D shape `[n_seqs, n_kv, max_seq, head_dim]` with non-degenerate
    /// inner axes.
    ///
    /// Shape choices (per dossier §4 iter-2a step 1 + kv_cache.rs:2226-2236):
    /// - `num_hidden_layers=4` + `full_attention_interval=2`
    ///   ⇒ layers = [Linear, Full, Linear, Full]
    ///   ⇒ `full_attn.len()=2` AND `linear_attn.len()=2` so BOTH the F32
    ///     full-attn buffer scaling AND the linear-attn recurrent
    ///     scaling get exercised in one cache.
    /// - `num_key_value_heads=2`, `head_dim=32`, `max_seq_len=64`
    ///   ⇒ baseline K bytes per slot = 1 * 2 * 64 * 32 * 4 = 16384 B
    ///   ⇒ n_seqs=4 K bytes per slot = 4 * 16384 = 65536 B (easy fit).
    /// - `linear_key_head_dim=8`, `linear_value_head_dim=8`,
    ///   `linear_num_value_heads=4`
    ///   ⇒ baseline recurrent bytes = 8 * 8 * 4 * 1 * 4 = 1024 B,
    ///     n_seqs=4 = 4096 B.
    /// - `moe = None` (dense variant ⇒ no MoE allocator path involvement).
    ///
    /// Anything larger here would slow down the test for no diagnostic
    /// benefit; anything smaller risks a degenerate axis collapsing the
    /// byte-scaling assertion (e.g. `max_seq_len=1` would make the
    /// n_seqs vs n_kv axis swap byte-undetectable).
    fn tiny_dense_cfg_4layer_for_multi_seq_tests() -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Dense,
            hidden_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 32,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            layer_types: default_layer_types(4, 2),
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rotary_dim: 8,
            mrope_section: [2, 2, 2, 2],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 4096,
            vocab_size: 256,
            attn_output_gate: true,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: true,
            intermediate_size: Some(128),
            moe: None,
        }
    }

    #[test]
    fn growable_tq_constructor_keeps_logical_context_but_seeds_small_physical_arenas() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("Metal device for test");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let cache = HybridKvCache::new_with_growable_tq(&cfg, &device, 4096, 16, 1)
            .expect("growable cache");

        assert_eq!(cache.max_seq_len, 4096, "logical context must not shrink");
        assert_eq!(cache.n_seqs, 16);
        for slot in &cache.full_attn {
            let tq = slot.tq.as_ref().expect("TQ storage");
            for slot_id in 0..16 {
                assert_eq!(
                    tq.physical_capacity_for_slot(crate::serve::multi_seq_kv::SlotId(slot_id))
                        .unwrap(),
                    1
                );
            }
            let static_full_context_bytes =
                16 * 2 * 4096 * 32 * 2 + 16 * 2 * 4096 * tq.norms_per_pos as usize * 4 * 2;
            assert!(
                tq.total_bytes() < static_full_context_bytes,
                "physical seed must not allocate the logical full-context product"
            );
        }
    }

    /// Dossier §2.8 H1 — falsifies the ADR-040 §1.3 structural claim
    /// ("structural shape supports `n_seqs > 1` with no buffer-layout
    /// change") on the allocator side.
    ///
    /// Falsifier (any one of these fires ⇒ ADR §1.3 falsified for Phase A2a):
    /// 1. `HybridKvCache::new(.., n_seqs=4)` panics or errors.
    /// 2. `cache.n_seqs != 4` after construction.
    /// 3. Full-attn K (or V) byte length at `n_seqs=4` is NOT exactly
    ///    4× the `n_seqs=1` baseline.
    /// 4. Linear-attn recurrent byte length at `n_seqs=4` is NOT exactly
    ///    4× the `n_seqs=1` baseline.
    ///
    /// The capture-buffer (5-D shape with the n_tokens_max axis OUTSIDE
    /// n_seqs per kv_cache.rs:1476-1480) is intentionally NOT asserted
    /// here — dossier §2.1.4 + §2.10 R1 flag it as the linear-attn
    /// multi-seq deferral boundary, and Phase A2a ships full-attn +
    /// MTP lift ONLY.
    #[test]
    fn h1_hybrid_kv_cache_alloc_n_seqs_4_byte_scale() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("cpu device for test");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let max_seq_len = 64u32;

        let cache_1 = HybridKvCache::new(&cfg, &device, max_seq_len, 1).expect("alloc at n_seqs=1");
        let cache_4 = HybridKvCache::new(&cfg, &device, max_seq_len, 4).expect("alloc at n_seqs=4");

        assert_eq!(cache_1.n_seqs, 1, "H1: n_seqs=1 baseline construction");
        assert_eq!(cache_4.n_seqs, 4, "H1: n_seqs=4 lift surfaced");

        // Falsifier 3: full-attn K + V scale exactly 4× with n_seqs.
        // (n_seqs is the outermost axis at kv_cache.rs:2231-2236; the
        // alloc multiplies by `n_seqs as usize` at kv_cache.rs:2226.)
        assert!(!cache_1.full_attn.is_empty(), "tiny cfg has full-attn slot");
        let baseline_k = cache_1.full_attn[0]
            .k
            .as_ref()
            .expect("F32 K present (legacy non-TQ path)")
            .byte_len();
        let lifted_k = cache_4.full_attn[0]
            .k
            .as_ref()
            .expect("F32 K present (legacy non-TQ path)")
            .byte_len();
        assert_eq!(
            lifted_k,
            baseline_k * 4,
            "H1 FALSIFIED: full-attn K does not scale linearly with n_seqs \
             ({} != {} * 4 = {}); ADR-040 §1.3 structural claim broken",
            lifted_k,
            baseline_k,
            baseline_k * 4
        );

        let baseline_v = cache_1.full_attn[0]
            .v
            .as_ref()
            .expect("F32 V present (legacy non-TQ path)")
            .byte_len();
        let lifted_v = cache_4.full_attn[0]
            .v
            .as_ref()
            .expect("F32 V present (legacy non-TQ path)")
            .byte_len();
        assert_eq!(
            lifted_v,
            baseline_v * 4,
            "H1 FALSIFIED: full-attn V does not scale linearly with n_seqs \
             ({} != {} * 4 = {})",
            lifted_v,
            baseline_v,
            baseline_v * 4
        );

        // Falsifier 4: linear-attn recurrent scales exactly 4× with n_seqs.
        // (Recurrent shape `[D_k, D_v, num_v_heads, n_seqs]` per
        // kv_cache.rs:2284-2289 — n_seqs is OUTERMOST.)
        if !cache_1.linear_attn.is_empty() {
            let baseline_r = cache_1.linear_attn[0].recurrent.byte_len();
            let lifted_r = cache_4.linear_attn[0].recurrent.byte_len();
            assert_eq!(
                lifted_r,
                baseline_r * 4,
                "H1 FALSIFIED: linear-attn recurrent does not scale \
                 linearly with n_seqs ({} != {} * 4 = {})",
                lifted_r,
                baseline_r,
                baseline_r * 4
            );

            // Capture-buffer assertion intentionally OMITTED — see dossier
            // §2.1.4 + §2.10 R1: the 5-D capture buffer asserts n_seqs=1
            // at kv_cache.rs:1567 and is deferred to Phase A2b.
        }

        // current_len cursor vec also scales with n_seqs by construction
        // at kv_cache.rs:2213 + 2247 — pin this so a future refactor
        // can't silently regress the per-slot bookkeeping.
        assert_eq!(
            cache_1.full_attn[0].current_len.len(),
            1,
            "H1: baseline current_len Vec length tracks n_seqs"
        );
        assert_eq!(
            cache_4.full_attn[0].current_len.len(),
            4,
            "H1: lifted current_len Vec length tracks n_seqs"
        );

        // ── iter-2.5 M5: shape/stride proof ────────────────────────────
        //
        // byte_len() 4× scaling is necessary but NOT sufficient.  An
        // axis-order swap (e.g. n_seqs↔n_kv_heads) would produce the
        // identical byte count yet break per-slot indexing because
        // the kernel walks the shape in a fixed order.  M5 adds
        // shape-axis assertions so the test catches:
        //   - n_seqs landing on the wrong axis position
        //   - a non-n_seqs dim changing between n_seqs=1 and n_seqs=4
        //   - dtype reinterpretation (caught implicitly via the
        //     by-shape product check)
        //
        // **Layout conventions** (per kv_cache.rs alloc sites):
        //   - Full-attn K/V (line 2231-2236): row-major shape vec
        //     `[n_seqs, n_kv_heads, max_seq_len, head_dim]` — `n_seqs`
        //     is at shape[0] (outermost in row-major; head_dim
        //     innermost stride-1).
        //   - Linear-attn recurrent (line 2284-2289):
        //     column-major-style shape vec `[D_k, D_v, num_v_heads,
        //     n_seqs]` — `n_seqs` is at shape.last() (outermost in
        //     column-major; D_k innermost stride-1; comment at line
        //     2278 confirms "d_k innermost").
        //
        // These two layouts pick different conventions because each
        // matches its respective kernel's native traversal order;
        // the M5 assertions hard-code the per-buffer convention
        // rather than trying to pick a single "outermost" idea.

        // Full-attn K: shape[0] must be n_seqs; other dims invariant.
        let k_shape_1 = cache_1.full_attn[0].k.as_ref().unwrap().shape().to_vec();
        let k_shape_4 = cache_4.full_attn[0].k.as_ref().unwrap().shape().to_vec();
        assert_eq!(
            k_shape_1.len(),
            4,
            "M5: full-attn K must be 4-D; got shape {:?}",
            k_shape_1
        );
        assert_eq!(
            k_shape_4.len(),
            4,
            "M5: full-attn K (n_seqs=4) must be 4-D; got shape {:?}",
            k_shape_4
        );
        assert_eq!(
            k_shape_1[0], 1,
            "M5: baseline full-attn K shape[0] must be n_seqs=1; got {:?}",
            k_shape_1
        );
        assert_eq!(
            k_shape_4[0], 4,
            "M5 FALSIFIED: full-attn K shape[0] must be n_seqs=4 \
             (n_seqs landed on the wrong axis — kernel per-slot indexing \
             will silently corrupt); got {:?}",
            k_shape_4
        );
        // All non-n_seqs dims invariant between cache_1 and cache_4 —
        // catches an axis-permutation where n_seqs is correctly
        // outermost but, e.g., n_kv_heads and head_dim swap.
        assert_eq!(
            &k_shape_4[1..],
            &k_shape_1[1..],
            "M5 FALSIFIED: non-n_seqs dims diverge between n_seqs=1 \
             ({:?}) and n_seqs=4 ({:?}) — silent axis swap",
            k_shape_1,
            k_shape_4
        );

        // Full-attn V: same convention as K.  Catches an asymmetric
        // K-vs-V layout regression (e.g. K stays correct, V swaps).
        let v_shape_1 = cache_1.full_attn[0].v.as_ref().unwrap().shape().to_vec();
        let v_shape_4 = cache_4.full_attn[0].v.as_ref().unwrap().shape().to_vec();
        assert_eq!(
            v_shape_1[0], 1,
            "M5: baseline full-attn V shape[0] must be n_seqs=1; got {:?}",
            v_shape_1
        );
        assert_eq!(
            v_shape_4[0], 4,
            "M5 FALSIFIED: full-attn V shape[0] must be n_seqs=4; got {:?}",
            v_shape_4
        );
        assert_eq!(
            &v_shape_4[1..],
            &v_shape_1[1..],
            "M5 FALSIFIED: V non-n_seqs dims diverge ({:?} vs {:?})",
            v_shape_1,
            v_shape_4
        );

        // Linear-attn recurrent: shape.last() must be n_seqs;
        // preceding dims invariant.  Convention differs from
        // full-attn (see comment above).
        if !cache_1.linear_attn.is_empty() {
            let r_shape_1 = cache_1.linear_attn[0].recurrent.shape().to_vec();
            let r_shape_4 = cache_4.linear_attn[0].recurrent.shape().to_vec();
            assert_eq!(
                r_shape_1.len(),
                4,
                "M5: linear-attn recurrent must be 4-D; got {:?}",
                r_shape_1
            );
            assert_eq!(
                r_shape_4.len(),
                4,
                "M5: linear-attn recurrent (n_seqs=4) must be 4-D; got {:?}",
                r_shape_4
            );
            assert_eq!(
                r_shape_1.last().copied(),
                Some(1),
                "M5: baseline linear-attn recurrent shape.last() must be \
                 n_seqs=1; got {:?}",
                r_shape_1
            );
            assert_eq!(
                r_shape_4.last().copied(),
                Some(4),
                "M5 FALSIFIED: linear-attn recurrent shape.last() must be \
                 n_seqs=4 (n_seqs landed on the wrong axis — kernel \
                 per-slot indexing will silently corrupt); got {:?}",
                r_shape_4
            );
            // Non-n_seqs dims invariant — catches an axis permutation
            // among [D_k, D_v, num_v_heads].
            let r_inner_1 = &r_shape_1[..r_shape_1.len() - 1];
            let r_inner_4 = &r_shape_4[..r_shape_4.len() - 1];
            assert_eq!(
                r_inner_4, r_inner_1,
                "M5 FALSIFIED: linear-attn recurrent non-n_seqs dims \
                 diverge between n_seqs=1 ({:?}) and n_seqs=4 ({:?}) — \
                 silent axis swap within [D_k, D_v, num_v_heads]",
                r_shape_1, r_shape_4
            );
        }
    }

    /// iter-2.5 H1-tq pin — sibling to H1 that exercises the TQ-active
    /// production KV path per dossier §2.1.7.  H1 uses
    /// `HybridKvCache::new(..)` which is the legacy F32-only allocator
    /// (`tq_kv_active=false`); a TQ-active build constructs via
    /// `new_with_options(.., tq_kv_active=true)` which adds U8-packed
    /// K/V + F32 norms buffers (`alloc_tq_full_attn_buffers` at
    /// kv_cache.rs:2393) and DROPS the F32 K/V backing per
    /// iter-34's 3.94× memory savings flip.
    ///
    /// Falsifiers (any one ⇒ iter-2.5 H1-tq broken):
    /// 1. `HybridKvCache::new_with_options(.., n_seqs=4, true)` panics
    ///    or errors at construction.
    /// 2. `cache.tq_kv_active` is not propagated.
    /// 3. TQ K/V packed buffers at `n_seqs=4` are NOT exactly 4× the
    ///    `n_seqs=1` baseline.
    /// 4. TQ K/V norms buffers at `n_seqs=4` are NOT exactly 4× the
    ///    `n_seqs=1` baseline.
    /// 5. `n_seqs` is NOT shape[0] on the TQ packed/norms buffers
    ///    (axis-order swap — same M5-class regression as the F32
    ///    path).
    ///
    /// **NOT a strict superset of H1** — H1 covers F32 buffers
    /// (`slot.k.is_some()` and `slot.v.is_some()`) which are
    /// dropped in TQ-active mode (iter-34); the two tests are
    /// complementary halves of the n_seqs lift coverage matrix.
    #[test]
    fn h1_tq_active_hybrid_kv_cache_alloc_n_seqs_4_byte_scale() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("cpu device for test");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let max_seq_len = 64u32;

        let cache_1 = HybridKvCache::new_with_options(&cfg, &device, max_seq_len, 1, true)
            .expect("TQ-active alloc at n_seqs=1");
        let cache_4 = HybridKvCache::new_with_options(&cfg, &device, max_seq_len, 4, true)
            .expect("TQ-active alloc at n_seqs=4");

        // Falsifier 2: tq_kv_active flag propagated.
        assert!(
            cache_1.tq_kv_active,
            "H1-tq: tq_kv_active must be true after new_with_options(.., true)"
        );
        assert!(
            cache_4.tq_kv_active,
            "H1-tq: tq_kv_active must be true after new_with_options(.., true)"
        );

        // Falsifier 1: n_seqs propagated.
        assert_eq!(cache_1.n_seqs, 1, "H1-tq: n_seqs=1 baseline");
        assert_eq!(cache_4.n_seqs, 4, "H1-tq: n_seqs=4 lift");

        // Falsifier (iter-34 contract): F32 K/V are DROPPED in
        // TQ-active mode.  Without this assert a future regression
        // that re-introduces shadow-mode F32 backing would silently
        // double the memory; the byte-scale check below would still
        // pass because both sides scale 4×.
        if cache_1.full_attn.is_empty() {
            eprintln!("H1-tq: cfg yields no full-attn layers; vacuous");
            return;
        }
        assert!(
            cache_4.full_attn[0].k.is_none(),
            "H1-tq: TQ-active full-attn slot.k must be None (iter-34 \
             dropped F32 backing for 3.94× savings)"
        );
        assert!(
            cache_4.full_attn[0].v.is_none(),
            "H1-tq: TQ-active full-attn slot.v must be None (iter-34)"
        );
        // TQ buffers MUST be present.
        let tq_1 = cache_1.full_attn[0]
            .tq
            .as_ref()
            .expect("H1-tq: tq present when tq_kv_active=true at n_seqs=1");
        let tq_4 = cache_4.full_attn[0]
            .tq
            .as_ref()
            .expect("H1-tq: tq present when tq_kv_active=true at n_seqs=4");

        // Falsifier 3: TQ packed scales 4×.
        let baseline_kp = tq_1.k_packed.byte_len();
        let lifted_kp = tq_4.k_packed.byte_len();
        assert_eq!(
            lifted_kp,
            baseline_kp * 4,
            "H1-tq FALSIFIED: TQ K-packed does not scale 4× with n_seqs \
             ({} != {} * 4 = {})",
            lifted_kp,
            baseline_kp,
            baseline_kp * 4
        );
        let baseline_vp = tq_1.v_packed.byte_len();
        let lifted_vp = tq_4.v_packed.byte_len();
        assert_eq!(
            lifted_vp,
            baseline_vp * 4,
            "H1-tq FALSIFIED: TQ V-packed does not scale 4× ({} != {})",
            lifted_vp,
            baseline_vp * 4
        );

        // Falsifier 4: TQ norms scales 4×.
        let baseline_kn = tq_1.k_norms.byte_len();
        let lifted_kn = tq_4.k_norms.byte_len();
        assert_eq!(
            lifted_kn,
            baseline_kn * 4,
            "H1-tq FALSIFIED: TQ K-norms does not scale 4× ({} != {})",
            lifted_kn,
            baseline_kn * 4
        );
        let baseline_vn = tq_1.v_norms.byte_len();
        let lifted_vn = tq_4.v_norms.byte_len();
        assert_eq!(
            lifted_vn,
            baseline_vn * 4,
            "H1-tq FALSIFIED: TQ V-norms does not scale 4× ({} != {})",
            lifted_vn,
            baseline_vn * 4
        );

        // Falsifier 5: M5-style shape proof for TQ buffers.  Per
        // `alloc_tq_full_attn_buffers` (kv_cache.rs:2421-2426 +
        // 2437-2442) the convention is `[n_seqs, n_kv_heads,
        // max_seq_len, head_dim]` and `[n_seqs, n_kv_heads,
        // max_seq_len, norms_per_pos]` — n_seqs at shape[0].
        let kp_shape_1 = tq_1.k_packed.shape().to_vec();
        let kp_shape_4 = tq_4.k_packed.shape().to_vec();
        assert_eq!(
            kp_shape_1.len(),
            4,
            "H1-tq M5: TQ K-packed must be 4-D; got {:?}",
            kp_shape_1
        );
        assert_eq!(
            kp_shape_1[0], 1,
            "H1-tq M5: baseline TQ K-packed shape[0] must be n_seqs=1; got {:?}",
            kp_shape_1
        );
        assert_eq!(
            kp_shape_4[0], 4,
            "H1-tq M5 FALSIFIED: TQ K-packed shape[0] must be n_seqs=4; got {:?}",
            kp_shape_4
        );
        assert_eq!(
            &kp_shape_4[1..],
            &kp_shape_1[1..],
            "H1-tq M5 FALSIFIED: TQ K-packed non-n_seqs dims diverge \
             ({:?} vs {:?})",
            kp_shape_1,
            kp_shape_4
        );
        // Same for K-norms.
        let kn_shape_1 = tq_1.k_norms.shape().to_vec();
        let kn_shape_4 = tq_4.k_norms.shape().to_vec();
        assert_eq!(
            kn_shape_1[0], 1,
            "H1-tq M5: baseline TQ K-norms shape[0] must be n_seqs=1; got {:?}",
            kn_shape_1
        );
        assert_eq!(
            kn_shape_4[0], 4,
            "H1-tq M5 FALSIFIED: TQ K-norms shape[0] must be n_seqs=4; got {:?}",
            kn_shape_4
        );
        assert_eq!(
            &kn_shape_4[1..],
            &kn_shape_1[1..],
            "H1-tq M5 FALSIFIED: TQ K-norms non-n_seqs dims diverge"
        );

        // current_len cursor vec also scales with n_seqs (same as H1).
        assert_eq!(
            cache_1.full_attn[0].current_len.len(),
            1,
            "H1-tq: baseline current_len Vec length tracks n_seqs"
        );
        assert_eq!(
            cache_4.full_attn[0].current_len.len(),
            4,
            "H1-tq: lifted current_len Vec length tracks n_seqs"
        );
    }

    // Trait-surface tests use the local `MultiSeqKvCache` impl (above the
    // tests module).  Pulling the trait + types into scope here keeps the
    // production code at the parent module untouched by test-only imports.
    use crate::serve::multi_seq_kv::{MultiSeqError, MultiSeqKvCache as _, MultiSeqLayout, SlotId};

    /// Pin: `slot_count()` returns the constructor's `n_seqs` verbatim.
    /// Falsifies any future refactor that introduces a u32→u64 cast or
    /// silently caps the value.
    #[test]
    fn qwen35_hybrid_kv_slot_count_matches_n_seqs() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let cache1 = HybridKvCache::new(&cfg, &device, 64, 1).expect("alloc 1");
        let cache4 = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc 4");
        assert_eq!(cache1.slot_count(), 1);
        assert_eq!(cache4.slot_count(), 4);
    }

    /// Pin: `layout()` returns `SeparateSlots` (HybridKvCache does not
    /// expose Paged — bounds-first ordering means this trip is only
    /// observable through this getter, not via append/drop/fork error
    /// shapes).
    #[test]
    fn qwen35_hybrid_kv_layout_is_separate_slots() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");
        assert_eq!(cache.layout(), MultiSeqLayout::SeparateSlots);
    }

    /// Pin (iter-1.5 cfa-finding-F5): out-of-range `SlotId` surfaces as
    /// `SlotOutOfRange { slot, max_slots }` with BOTH fields populated —
    /// not a partial error.  Bounds-first ordering rules out
    /// `LayoutNotSupported` masking the slot bug.
    #[test]
    fn qwen35_hybrid_kv_slot_out_of_range_errors_named() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");

        // seq_len OOR
        let err = cache
            .seq_len(SlotId(4))
            .expect_err("slot 4 OOR for n_seqs=4");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(4),
                max_slots: 4
            }
        );
        let err = cache.seq_len(SlotId(99)).expect_err("slot 99 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(99),
                max_slots: 4
            }
        );

        // append_for_seq OOR
        let err = cache.append_for_seq(SlotId(4), 1).expect_err("append OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(4),
                max_slots: 4
            }
        );

        // drop_seq OOR
        let err = cache.drop_seq(SlotId(4)).expect_err("drop OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(4),
                max_slots: 4
            }
        );

        // fork_seq src OOR FIRST (deterministic per fixture-parity contract).
        let err = cache
            .fork_seq(SlotId(4), SlotId(5))
            .expect_err("fork: src OOR first");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(4),
                max_slots: 4
            }
        );
        // fork_seq src valid, dst OOR.
        let err = cache
            .fork_seq(SlotId(0), SlotId(4))
            .expect_err("fork: dst OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(4),
                max_slots: 4
            }
        );
    }

    /// Pin: `append_for_seq` advances ONLY the named slot's cursor —
    /// surface-level isolation evidence for H3 (the per-buffer GPU write
    /// isolation lands in Phase B iter-3 forward-path slot threading;
    /// iter-2a's trait surface only owns the cursor bookkeeping).
    #[test]
    fn qwen35_hybrid_kv_append_advances_target_slot_only() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");

        // All slots start at 0.
        for s in 0..4 {
            assert_eq!(cache.seq_len(SlotId(s)).expect("seq_len in range"), 0);
        }

        cache.append_for_seq(SlotId(0), 5).expect("append slot 0");
        cache.append_for_seq(SlotId(2), 3).expect("append slot 2");

        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 5);
        assert_eq!(cache.seq_len(SlotId(1)).unwrap(), 0, "slot 1 untouched");
        assert_eq!(cache.seq_len(SlotId(2)).unwrap(), 3);
        assert_eq!(cache.seq_len(SlotId(3)).unwrap(), 0, "slot 3 untouched");
    }

    /// Dossier §2.8 H3 — per-slot isolation.  Writes to slot 0 and slot 2
    /// MUST NOT mutate slot 1's cursor.  The test seeds slot 1 with a
    /// known cursor via the trait surface (the only mutation API in
    /// Phase A2a), then exercises slots 0 and 2, then re-reads slot 1.
    #[test]
    fn qwen35_hybrid_kv_per_slot_isolation_n_seqs_4() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");

        // Seed slot 1.
        cache.append_for_seq(SlotId(1), 7).expect("seed slot 1");
        assert_eq!(cache.seq_len(SlotId(1)).unwrap(), 7);

        // Exercise slots 0 + 2.
        cache.append_for_seq(SlotId(0), 5).expect("write slot 0");
        cache.append_for_seq(SlotId(2), 11).expect("write slot 2");

        // H3 falsifier: slot 1 must be byte-equal-cursor to its seed.
        assert_eq!(
            cache.seq_len(SlotId(1)).unwrap(),
            7,
            "H3 FALSIFIED: slot 1 cursor mutated by writes to slots 0/2"
        );
        // Sanity: 0 and 2 took the expected increments.
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 5);
        assert_eq!(cache.seq_len(SlotId(2)).unwrap(), 11);
        assert_eq!(cache.seq_len(SlotId(3)).unwrap(), 0);
    }

    /// Dossier §2.8 H2 — at n_seqs=4, slot 0's `current_len` evolves
    /// identically to the n_seqs=1 baseline under the same append
    /// sequence.  This is the cursor-level analogue of the full byte-
    /// equivalence claim (the GPU-buffer-content side lands when Phase
    /// B iter-3 wires the forward path to per-slot offsets; the trait
    /// surface that Phase A2a ships owns ONLY the cursor side).
    ///
    /// Falsifier: any inequality between the n_seqs=1 cursor and the
    /// n_seqs=4 slot-0 cursor after the same op sequence ⇒ the lift
    /// is not invisible to slot-0 readers, and ADR §5 AC-1 byte-
    /// equivalence is broken at the trait-surface level.
    #[test]
    fn qwen35_hybrid_kv_byte_identical_at_slot_0_n_seqs_4_vs_1() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache1 = HybridKvCache::new(&cfg, &device, 64, 1).expect("alloc 1");
        let mut cache4 = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc 4");

        // Identical op sequence against slot 0 of each cache.
        for &n in &[1u32, 3, 5, 2] {
            cache1.append_for_seq(SlotId(0), n).expect("append cache1");
            cache4
                .append_for_seq(SlotId(0), n)
                .expect("append cache4 slot 0");
        }

        let l1 = cache1.seq_len(SlotId(0)).unwrap();
        let l4 = cache4.seq_len(SlotId(0)).unwrap();
        assert_eq!(
            l1, l4,
            "H2 FALSIFIED: slot 0 cursor at n_seqs=4 ({}) drifts from \
             n_seqs=1 baseline ({}) under identical append sequence",
            l4, l1
        );
        assert_eq!(l1, 1 + 3 + 5 + 2, "sanity: cursor sum matches op stream");

        // Per-layer pin: the underlying `current_len[0]` Vec entry on
        // EVERY full-attn slot must equal the trait's view (homogeneous
        // current_len assumption from dossier §4 step 2 — TRUE in
        // production because all full-attn layers advance together).
        for (idx, slot) in cache4.full_attn.iter().enumerate() {
            assert_eq!(
                slot.current_len[0], l4,
                "full_attn slot {} cursor drift at n_seqs=4 slot 0",
                idx
            );
        }
    }

    /// iter-2.5 C4 pin: `append_for_seq` keeps `current_len[slot.0]`
    /// byte-identical across every `full_attn[i]` slot AND the MTP
    /// slot (if present).  This is the production-wiring invariant
    /// that `seq_len()`'s canonical-from-`full_attn[0]` read depends
    /// on; the C4 fix added a `debug_assert!` against per-layer
    /// desync, and this test pins the production-side invariant
    /// (every layer's cursor is the same after a clean append).
    ///
    /// Falsifier: any `full_attn[i].current_len[slot]` that diverges
    /// from `full_attn[0].current_len[slot]` after a sequence of
    /// `append_for_seq(slot, _)` calls ⇒ the seq_len() canonical
    /// assumption is unsafe and the iter-2.5 C4 debug_assert is
    /// load-bearing for catching the regression.
    #[test]
    fn qwen35_hybrid_kv_seq_len_canonical_across_full_attn_layers() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");

        // Append a non-trivial sequence to slot 1.  Use multiple
        // bumps so a per-layer rounding/silent-truncation defect
        // would surface as a divergence, not coincidentally match
        // after one bump.
        cache.append_for_seq(SlotId(1), 2).unwrap();
        cache.append_for_seq(SlotId(1), 3).unwrap();
        // Total slot-1 cursor should now be 5 on every full-attn
        // layer.  The tiny cfg has 2 full-attn layers per
        // `tiny_dense_cfg_4layer_for_multi_seq_tests()` so the
        // assertion exercises >1 layer (not a vacuous single-layer
        // case).
        assert!(
            cache.full_attn.len() >= 2,
            "fixture sanity: tiny cfg yields ≥2 full-attn layers (got {})",
            cache.full_attn.len()
        );
        let canonical = cache.full_attn[0].current_len[1];
        assert_eq!(
            canonical, 5,
            "slot 1 canonical cursor must be 2+3=5 after the append sequence"
        );
        for (idx, slot) in cache.full_attn.iter().enumerate() {
            assert_eq!(
                slot.current_len[1], canonical,
                "C4 FALSIFIED: full_attn[{idx}].current_len[1] = {} \
                 diverges from canonical full_attn[0].current_len[1] = {}; \
                 the iter-2.5 C4 debug_assert in seq_len() would trip in \
                 debug builds — production wiring must keep cursors in \
                 lockstep across all full-attn layers.",
                slot.current_len[1], canonical
            );
        }

        // Other slots must be untouched (per-slot isolation pin).
        for slot_idx in [0u32, 2, 3] {
            for (layer, full) in cache.full_attn.iter().enumerate() {
                assert_eq!(
                    full.current_len[slot_idx as usize], 0,
                    "slot {slot_idx} on full_attn[{layer}] must remain 0 \
                     after slot-1 appends (per-slot isolation invariant)"
                );
            }
        }

        // And seq_len() returns the canonical value (the cursor read
        // is the load-bearing application of the invariant).
        assert_eq!(
            cache.seq_len(SlotId(1)).expect("seq_len 1 in range"),
            canonical
        );
    }

    #[test]
    fn qwen35_release_boundary_validation_keeps_mtp_cursor_independent() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        cfg.mtp_num_hidden_layers = 1;
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 2).expect("alloc");
        let slot = SlotId(1);
        cache.append_for_seq(slot, 5).expect("seed cursors");
        cache
            .validate_sequence_len_for_slot(slot, 5)
            .expect("homogeneous boundary");

        assert!(cache.full_attn.len() >= 2, "fixture needs a later layer");
        cache.full_attn[1].current_len[1] = 4;
        let layer_error = cache
            .validate_sequence_len_for_slot(slot, 5)
            .expect_err("later full-attention cursor desync must fail closed");
        assert!(format!("{layer_error:#}").contains("full_attn[1] cursor=4"));
        cache.full_attn[1].current_len[1] = 5;

        let mtp = cache.mtp_slot.as_mut().expect("fixture has MTP slot");
        mtp.current_len[1] = 3;
        assert_eq!(
            cache.seq_len(slot).expect("public length follows verifier"),
            5,
            "an independent MTP proposal cursor must not poison verifier length"
        );
        cache
            .validate_sequence_len_for_slot(slot, 5)
            .expect("base verifier boundary must not require MTP lockstep");
        let anchor = cache
            .snapshot_slot_anchor(slot, 5)
            .expect("prompt anchor captures the independent MTP cursor");
        assert_eq!(anchor.mtp_current_len, Some(3));
    }

    #[test]
    fn qwen35_speculative_boundary_requires_target_mtp_cursor_equality() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        cfg.mtp_num_hidden_layers = 1;
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 2).expect("alloc");
        let slot = SlotId(1);
        cache.append_for_seq(slot, 5).expect("seed target cursors");
        cache
            .mtp_slot
            .as_mut()
            .expect("fixture has MTP")
            .current_len[1] = 0;

        let mismatch = cache
            .validate_speculative_cursors_for_slot(slot, 5)
            .expect_err("empty MTP cursor must fail the speculative boundary");
        assert!(format!("{mismatch:#}").contains("MTP cursor=0 != expected=5"));

        cache
            .mtp_slot
            .as_mut()
            .expect("fixture has MTP")
            .current_len[1] = 5;
        cache
            .validate_speculative_cursors_for_slot(slot, 5)
            .expect("target and MTP cursor equality");
    }

    #[test]
    fn qwen38_spec_reject_rewinds_only_target_slot_ping_pong() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");
        let target = SlotId(2);

        for linear in &mut cache.linear_attn {
            linear.swap_for_slot(target);
        }
        assert!(
            cache
                .linear_attn
                .iter()
                .all(|linear| linear.pp_flipped[target.0 as usize]),
            "fixture must model one completed target forward"
        );

        cache
            .rewind_la_ping_pong_for_slot(target)
            .expect("reject rewinds target slot");
        for linear in &cache.linear_attn {
            assert!(!linear.pp_flipped[target.0 as usize]);
            for sibling in [0usize, 1, 3] {
                assert!(!linear.pp_flipped[sibling], "sibling slot changed");
            }
        }
    }

    /// Pin: drop resets ONLY the target slot's cursor (across all
    /// full-attn slots + MTP if present).  Dossier §4 iter-2a step 2:
    /// recurrent state intentionally NOT zeroed in Phase A2a — pinned
    /// by `qwen35_hybrid_kv_drop_does_not_zero_recurrent_buffer_a2a`.
    #[test]
    fn qwen35_hybrid_kv_drop_resets_seq_len_for_target_slot_only() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");

        // Seed every slot.
        cache.append_for_seq(SlotId(0), 10).unwrap();
        cache.append_for_seq(SlotId(1), 20).unwrap();
        cache.append_for_seq(SlotId(2), 30).unwrap();
        cache.append_for_seq(SlotId(3), 40).unwrap();

        // Drop slot 2.
        cache.drop_seq(SlotId(2)).expect("drop slot 2");

        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 10);
        assert_eq!(cache.seq_len(SlotId(1)).unwrap(), 20);
        assert_eq!(cache.seq_len(SlotId(2)).unwrap(), 0, "slot 2 reset");
        assert_eq!(cache.seq_len(SlotId(3)).unwrap(), 40);

        // Pin: drop wipes the cursor on EVERY full-attn slot, not just
        // the one `seq_len()` happens to read.
        for slot in &cache.full_attn {
            assert_eq!(
                slot.current_len[2], 0,
                "every full-attn slot's cursor[2] reset"
            );
            assert_eq!(
                slot.current_len[0], 10,
                "every full-attn slot's cursor[0] preserved"
            );
            assert_eq!(slot.current_len[1], 20);
            assert_eq!(slot.current_len[3], 40);
        }
    }

    /// Dossier §4 iter-2a step 2 + §2.10 R1 pin: Phase A2a's `drop_seq`
    /// must NOT zero the linear-attn recurrent state.  Lifting that
    /// behaviour is Phase A2b's responsibility (gated on the
    /// `rollback_la_to` guard at kv_cache.rs:1567 being lifted, which
    /// requires the spec-decode capture-buffer layout to be re-derived
    /// for n_seqs > 1).
    ///
    /// Falsifier: any byte change to `linear_attn[0].recurrent` after a
    /// `drop_seq` call ⇒ Phase A2a has crossed into the linear-attn
    /// carve-out's territory.
    ///
    /// **iter-2.5 M4 strengthening**: the iter-2a version only
    /// compared `byte_len()` before/after, which proves NOTHING about
    /// content invariance — allocation length staying constant is
    /// vacuously true under any reasonable `drop_seq` impl, including
    /// a buggy one that zeros the bytes in place.  This version
    /// fills the recurrent buffer with a deterministic non-zero
    /// pattern via direct `as_mut_slice::<f32>()` write, snapshots
    /// the bytes, calls `drop_seq`, snapshots again, and asserts
    /// byte-by-byte equality.  Any in-place mutation by `drop_seq`
    /// surfaces here.
    #[test]
    fn qwen35_hybrid_kv_drop_does_not_zero_recurrent_buffer_a2a() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");
        if cache.linear_attn.is_empty() {
            // Defensive: tiny_cfg has linear_attn slots, but if a future
            // cfg drop changes this, skip rather than false-pass.
            eprintln!(
                "qwen35_hybrid_kv_drop_does_not_zero_recurrent_buffer_a2a: \
                 cfg has no linear_attn; vacuous"
            );
            return;
        }

        // Step 1: fill recurrent buffer of layer 0 with a deterministic
        // non-zero pattern.  MlxBuffer is CPU-accessible on Apple
        // Silicon (StorageModeShared) so `as_mut_slice::<f32>()` is
        // a direct host write — no kernel dispatch needed, no
        // download/upload helper.  Production write path lives in
        // gpu_delta_net.rs; this test uses the host-side accessor
        // because the contract under audit is "drop_seq does NOT
        // touch this buffer", which is observable purely from a
        // host-side byte snapshot.
        let total_f32 = cache.linear_attn[0].recurrent.byte_len() / std::mem::size_of::<f32>();
        assert!(
            total_f32 > 0,
            "fixture sanity: recurrent buffer must have non-zero element count"
        );
        {
            let slice = cache.linear_attn[0]
                .recurrent
                .as_mut_slice::<f32>()
                .expect("recurrent is F32 + StorageModeShared (Apple Silicon)");
            assert_eq!(slice.len(), total_f32, "as_mut_slice element count");
            for (i, dst) in slice.iter_mut().enumerate() {
                // Pattern: 0.42 * (i+1) keeps values non-zero and
                // distinguishable across positions, so a partial-zero
                // bug (e.g. "zero only the first N bytes for slot 0")
                // surfaces as a position-dependent diff.
                *dst = 0.42_f32 * (i as f32 + 1.0_f32);
            }
        }

        // Step 2: snapshot the recurrent buffer bytes after the
        // deterministic upload.  Clone the f32 slice into an owned
        // Vec so the snapshot is detached from the live buffer.
        let before: Vec<f32> = cache.linear_attn[0]
            .recurrent
            .as_slice::<f32>()
            .expect("recurrent f32 view")
            .to_vec();
        assert_eq!(before.len(), total_f32);
        // Confirm the upload itself worked — at least one element is
        // the expected non-zero pattern.  Defends against a future
        // refactor that silently breaks `as_mut_slice` for this
        // buffer kind.
        assert!(
            before.iter().any(|&v| v != 0.0),
            "M4 fixture sanity: deterministic upload must produce \
             non-zero bytes (else the test is vacuous)"
        );

        // Step 3: call drop_seq(SlotId(0)).  Per Phase A2a contract
        // (dossier §4 iter-2a step 2 + §2.10 R1) this MUST NOT touch
        // recurrent contents at all.
        cache.drop_seq(SlotId(0)).expect("drop slot 0");

        // Step 4: snapshot again.
        let after: Vec<f32> = cache.linear_attn[0]
            .recurrent
            .as_slice::<f32>()
            .expect("recurrent f32 view (after)")
            .to_vec();

        // Step 5: full byte-by-byte (f32-by-f32) equality.  Any
        // mutation by drop_seq — including partial zero, partial
        // overwrite, in-place ping-pong swap — surfaces here.  The
        // previous iter-2a assertion (byte_len equality) would
        // false-pass on every single one of those bug patterns.
        assert_eq!(
            before.len(),
            after.len(),
            "Phase A2a contract: recurrent buffer length must not change \
             across drop_seq (was {}, now {})",
            before.len(),
            after.len()
        );
        assert_eq!(
            before, after,
            "Phase A2a contract (iter-2.5 M4): drop_seq mutated \
             linear_attn[0].recurrent contents.  Per dossier R1, A2a \
             does NOT touch linear-attn state; this test pins that \
             contract via byte-for-byte content comparison, NOT the \
             previous vacuous byte_len() check."
        );
    }

    /// Pin: `fork_seq(src, src)` is a successful no-op per trait spec.
    /// Iter-1 fixture parity contract.
    #[test]
    fn qwen35_hybrid_kv_fork_to_self_is_noop_ok() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");
        cache.append_for_seq(SlotId(2), 9).unwrap();
        // src == dst — no-op success.
        cache.fork_seq(SlotId(2), SlotId(2)).expect("fork self ok");
        // Cursor unchanged.
        assert_eq!(cache.seq_len(SlotId(2)).unwrap(), 9);
        // Other slots untouched.
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 0);
        assert_eq!(cache.seq_len(SlotId(1)).unwrap(), 0);
        assert_eq!(cache.seq_len(SlotId(3)).unwrap(), 0);
    }

    /// **HISTORICAL** — Phase A2a / iter-2.5 M1 typed-clamp pin
    /// (renamed from `qwen35_hybrid_kv_fork_cross_slot_returns_capability_unsupported_at_phase_a2a`
    /// at iter-A2c per ADR-040 brief "rename to historical_ if they
    /// were pinning the clamp shape").
    ///
    /// **Prior contract** (A2a → A2c): cross-slot fork returned
    /// `CapabilityUnsupported` with a capability label naming the
    /// deferred Phase A2c kernel arc + dossier R5.  This pinned the
    /// typed-clamp envelope (HTTP 501) before the real same-buffer
    /// cross-region memcpy landed.
    ///
    /// **Closure (iter-A2c, 2026-05-30)**: the real fork dispatch
    /// shipped at `kv_cache.rs` `HybridKvCache::fork_seq`.  This
    /// historical test ASSERTS the closure by pinning the NEW
    /// contract: cross-slot fork must return `Ok(())` (the discriminant
    /// flip from `Err(CapabilityUnsupported)` to `Ok(())` is the iter
    /// closure signal per the prior comment "When Phase A2c ships the
    /// real kernel dispatch, ... flip the assertion to `expect('fork
    /// ok after A2c')`").  The full byte-equality + cursor-copy
    /// pin lives at H158 + H163-H165.
    #[test]
    fn historical_qwen35_hybrid_kv_fork_cross_slot_closure_at_phase_a2c() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");
        cache.append_for_seq(SlotId(0), 7).unwrap();
        // iter-A2c closure: fork now returns Ok(()) (was previously
        // CapabilityUnsupported per iter-2.5 M1 typed-clamp).
        cache.fork_seq(SlotId(0), SlotId(1)).expect(
            "iter-A2c closure: cross-slot fork must return Ok(()) — \
                     was previously CapabilityUnsupported per A2a typed-clamp; \
                     A2c (this iter) ships the real same-buffer cross-region memcpy",
        );
        // Cursor copy invariant: dst.seq_len == src.seq_len after fork
        // (H165 sub-pin; fully exercised at H158/H165).
        assert_eq!(
            cache.seq_len(SlotId(1)).unwrap(),
            7,
            "iter-A2c closure: fork_seq must copy src's seq_len to dst"
        );
        // src unchanged (H163 sub-pin).
        assert_eq!(
            cache.seq_len(SlotId(0)).unwrap(),
            7,
            "iter-A2c closure: fork_seq must NOT modify src's seq_len"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // ADR-040 Phase A2b iter-A2b — linear-attn capture-buffer multi-seq lift
    // hypotheses H31-H35 (2026-05-29).
    //
    // A2a shipped the full-attn + MTP n_seqs lift (kv_cache.rs:2226-2247) and
    // documented the linear-attn capture buffer + `rollback_la_to` guard at
    // kv_cache.rs:1567 as the deferred sub-iter per dossier §1.3 / §2.1.4 /
    // §2.10 R1.  Iter-A2b lifts the rollback math to per-slot routing using
    // the real layout proofs:
    //
    //   - recurrent: `[D_k, D_v, n_v_heads, n_seqs]`  col-major
    //                ⇒ slot s offset = s * (D_k * D_v * n_v_heads)
    //
    //   - capture:   `[D_k, D_v, n_v_heads, n_tokens_max, n_seqs]`  col-major
    //                ⇒ slot s, token t offset = s * (n_tokens_max * D_k * D_v
    //                  * n_v_heads) + t * (D_k * D_v * n_v_heads)
    //                (matches mlx-native `gated_delta_net_decode_capture.metal`
    //                 lines 37-46: state_capture_seq_stride = n_tokens *
    //                 state_capture_token_stride)
    //
    //   - conv_state: `[channels, K-1, n_seqs]`  col-major
    //                ⇒ slot s offset = s * (channels * (K-1))
    //
    //   - conv_capture: `[n_seqs, n_tokens_max, K-1, channels]`  row-major
    //                ⇒ slot s, token t offset = s * (n_tokens_max * (K-1) *
    //                  channels) + t * ((K-1) * channels)
    //
    // Forward-path linear-attn dispatch sites in `gpu_delta_net.rs` (the H5
    // `n_seqs = 1u32` hard-codes) are intentionally NOT lifted in this iter
    // — they live behind the existing serial dispatch path and are gated on
    // iter-A2b-cont (parallel to Qwen35 B4a → B4a-cont split per dossier).
    //
    // Order in this block:
    //   H31 — capture buffer 5-D byte-scale at n_seqs=4
    //   H32 — per-slot capture isolation: write slot 0 → slot 1 untouched
    //   H33 — per-slot rollback isolation: rollback slot 0 → slot 1 recurrent
    //         + conv_state untouched
    //   H34 — n_seqs=1 byte-equivalence: rollback math matches pre-A2b
    //   H35 — slot out-of-range typed error names ADR-040 Phase A2b
    // ──────────────────────────────────────────────────────────────────────

    /// H31 — linear-attn capture buffer byte-scale at n_seqs=4.
    ///
    /// Pins that `ensure_la_capture` allocates the recurrent capture
    /// (`[D_k, D_v, n_v_heads, n_tokens_max, n_seqs]` F32) AND the conv
    /// capture (`[n_seqs, n_tokens_max, K-1, channels]` F32) at the
    /// expected byte size for `n_seqs=4`.
    ///
    /// Falsifier: byte-len at n_seqs=4 not exactly 4× the n_seqs=1 baseline,
    /// OR closed-form formula `n_seqs * n_tokens_max * per_seq_elems * 4`
    /// (recurrent) disagrees with the alloc'd byte_len.
    ///
    /// **Layout proof (ADR §6.1.23):**
    /// - recurrent capture per_seq_elems = D_k * D_v * n_v_heads
    /// - recurrent capture bytes = n_seqs * n_tokens_max * per_seq_elems * 4
    /// - conv capture per_seq_elems = channels * (K-1)
    /// - conv capture bytes = n_seqs * n_tokens_max * per_seq_elems * 4
    #[test]
    fn h31_la_capture_buffer_byte_scale_n_seqs_4_2026_05_29() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let max_seq_len = 64u32;
        let n_tokens_max = 4u32;

        let mut cache_1 =
            HybridKvCache::new(&cfg, &device, max_seq_len, 1).expect("alloc n_seqs=1");
        cache_1
            .ensure_la_capture(&cfg, &device, n_tokens_max)
            .expect("ensure n_seqs=1");

        let mut cache_4 =
            HybridKvCache::new(&cfg, &device, max_seq_len, 4).expect("alloc n_seqs=4");
        cache_4
            .ensure_la_capture(&cfg, &device, n_tokens_max)
            .expect("ensure n_seqs=4");

        assert!(
            !cache_1.linear_attn.is_empty(),
            "tiny cfg has linear-attn slot"
        );

        // Falsifier (1): recurrent capture byte-len scales 4×.
        let baseline_cap = cache_1.linear_attn[0]
            .capture_states
            .as_ref()
            .expect("ensure_la_capture allocated capture_states at n_seqs=1")
            .byte_len();
        let lifted_cap = cache_4.linear_attn[0]
            .capture_states
            .as_ref()
            .expect("ensure_la_capture allocated capture_states at n_seqs=4")
            .byte_len();
        assert_eq!(
            lifted_cap,
            baseline_cap * 4,
            "H31 FALSIFIED: recurrent capture does not scale linearly with n_seqs \
             ({} != {} * 4 = {})",
            lifted_cap,
            baseline_cap,
            baseline_cap * 4
        );

        // Closed-form check: bytes = n_seqs * n_tokens_max * per_seq_elems * 4.
        let per_seq_elems = (cfg.linear_key_head_dim as usize)
            * (cfg.linear_value_head_dim as usize)
            * (cfg.linear_num_value_heads as usize);
        let expected_bytes_4 = 4 * (n_tokens_max as usize) * per_seq_elems * 4;
        assert_eq!(
            lifted_cap, expected_bytes_4,
            "H31 closed-form: capture bytes at n_seqs=4 must equal \
             4 * {n_tokens_max} * {per_seq_elems} * 4 = {expected_bytes_4}; \
             got {lifted_cap}"
        );

        // Falsifier (2): conv_capture byte-len scales 4×.
        let baseline_conv = cache_1.linear_attn[0]
            .conv_capture_states
            .as_ref()
            .expect("ensure_la_capture allocated conv_capture at n_seqs=1")
            .byte_len();
        let lifted_conv = cache_4.linear_attn[0]
            .conv_capture_states
            .as_ref()
            .expect("ensure_la_capture allocated conv_capture at n_seqs=4")
            .byte_len();
        assert_eq!(
            lifted_conv,
            baseline_conv * 4,
            "H31 FALSIFIED: conv_capture does not scale linearly with n_seqs \
             ({} != {} * 4 = {})",
            lifted_conv,
            baseline_conv,
            baseline_conv * 4
        );

        let conv_channels = conv_channels_for(&cfg) as usize;
        let k_minus1 = (cfg.linear_conv_kernel_dim.saturating_sub(1)) as usize;
        let conv_per_seq = conv_channels * k_minus1;
        let expected_conv_bytes_4 = 4 * (n_tokens_max as usize) * conv_per_seq * 4;
        assert_eq!(
            lifted_conv, expected_conv_bytes_4,
            "H31 closed-form: conv_capture bytes at n_seqs=4 must equal \
             4 * {n_tokens_max} * {conv_per_seq} * 4 = {expected_conv_bytes_4}; \
             got {lifted_conv}"
        );
    }

    /// H32 — per-slot capture write isolation.
    ///
    /// Pins that writing a known F32 pattern into slot 0's per-seq region of
    /// `capture_states` leaves slot 1's region byte-identical to its
    /// initial-allocated state (zero-init via the allocator).
    ///
    /// Falsifier: any byte in slot 1's per-seq capture region changed after
    /// writing only slot 0's region.
    ///
    /// Layout: slot s offset = `s * (n_tokens_max * per_seq_elems)`
    /// per-element. Per-seq slice = `[slot_off .. slot_off +
    /// n_tokens_max*per_seq_elems]`.
    #[test]
    fn h32_la_capture_per_slot_write_isolation_2026_05_29() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let n_tokens_max = 4u32;

        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");
        cache
            .ensure_la_capture(&cfg, &device, n_tokens_max)
            .expect("ensure");

        let per_seq_elems = (cfg.linear_key_head_dim as usize)
            * (cfg.linear_value_head_dim as usize)
            * (cfg.linear_num_value_heads as usize);
        let seq_stride = (n_tokens_max as usize) * per_seq_elems;

        // Snapshot slot 1's capture region BEFORE writing slot 0.
        let snapshot_slot1: Vec<f32> = {
            let cap = cache.linear_attn[0]
                .capture_states
                .as_ref()
                .expect("capture allocated");
            let slice = cap.as_slice::<f32>().expect("capture as_slice");
            assert_eq!(
                slice.len(),
                seq_stride * 4,
                "capture total elems must equal seq_stride * n_seqs"
            );
            slice[seq_stride..2 * seq_stride].to_vec()
        };

        // Write a non-zero pattern into slot 0's per-seq region.
        {
            let cap = cache.linear_attn[0]
                .capture_states
                .as_mut()
                .expect("capture mut");
            let slice = cap.as_mut_slice::<f32>().expect("capture mut slice");
            for (i, v) in slice[..seq_stride].iter_mut().enumerate() {
                *v = (i + 1) as f32 * 7.0;
            }
        }

        // Slot 1's per-seq region must be byte-identical to the snapshot.
        let after_slot1: Vec<f32> = {
            let cap = cache.linear_attn[0]
                .capture_states
                .as_ref()
                .expect("capture re-borrow");
            let slice = cap.as_slice::<f32>().expect("capture as_slice 2");
            slice[seq_stride..2 * seq_stride].to_vec()
        };
        assert_eq!(
            after_slot1, snapshot_slot1,
            "H32 FALSIFIED: writing slot 0's capture region perturbed slot 1's region \
             (capture write isolation broken — slot stride must be exactly {} elems)",
            seq_stride
        );

        // Also verify slots 2 and 3 are byte-untouched.
        let after_slot2: Vec<f32> = {
            let cap = cache.linear_attn[0].capture_states.as_ref().unwrap();
            let slice = cap.as_slice::<f32>().unwrap();
            slice[2 * seq_stride..3 * seq_stride].to_vec()
        };
        assert!(
            after_slot2.iter().all(|&v| v == 0.0),
            "H32: slot 2's capture region must remain zero-init after slot 0 write"
        );
        let after_slot3: Vec<f32> = {
            let cap = cache.linear_attn[0].capture_states.as_ref().unwrap();
            let slice = cap.as_slice::<f32>().unwrap();
            slice[3 * seq_stride..4 * seq_stride].to_vec()
        };
        assert!(
            after_slot3.iter().all(|&v| v == 0.0),
            "H32: slot 3's capture region must remain zero-init after slot 0 write"
        );
    }

    /// H33 — per-slot `rollback_la_to` isolation.
    ///
    /// Writes distinct patterns into slot 0's and slot 1's capture regions
    /// (recurrent + conv), seeds non-zero contents into BOTH slots' active
    /// `recurrent` + `conv_state` buffers, then calls
    /// `rollback_la_to(SlotId(0), 2)` and asserts:
    ///   1. Slot 0's recurrent + conv_state regions now contain slot 0's
    ///      capture-at-token-2 pattern.
    ///   2. Slot 1's recurrent + conv_state regions are byte-untouched.
    ///   3. Slots 2 and 3 (n_seqs=4) are also byte-untouched.
    ///
    /// Falsifier: any byte in slot 1's, slot 2's, or slot 3's recurrent or
    /// conv_state region changed after rolling back ONLY slot 0.
    #[test]
    fn h33_rollback_la_to_per_slot_isolation_2026_05_29() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let n_tokens_max = 4u32;

        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");
        cache
            .ensure_la_capture(&cfg, &device, n_tokens_max)
            .expect("ensure");

        let per_seq_elems = (cfg.linear_key_head_dim as usize)
            * (cfg.linear_value_head_dim as usize)
            * (cfg.linear_num_value_heads as usize);
        let cap_seq_stride = (n_tokens_max as usize) * per_seq_elems;

        let conv_channels = conv_channels_for(&cfg) as usize;
        let k_minus1 = (cfg.linear_conv_kernel_dim.saturating_sub(1)) as usize;
        let conv_per_seq = conv_channels * k_minus1;
        let conv_cap_seq_stride = (n_tokens_max as usize) * conv_per_seq;

        // Seed capture buffers (per-slot distinct patterns).
        // Slot 0 token t element idx: value = 1000 + t*100 + idx
        // Slot 1 token t element idx: value = 9000 + t*100 + idx
        // Recurrent capture (col-major; slot stride = cap_seq_stride):
        {
            let cap = cache.linear_attn[0]
                .capture_states
                .as_mut()
                .expect("cap mut");
            let slice = cap.as_mut_slice::<f32>().expect("cap mut slice");
            for t in 0..(n_tokens_max as usize) {
                for idx in 0..per_seq_elems {
                    let s0 = 0 * cap_seq_stride + t * per_seq_elems + idx;
                    slice[s0] = (1000 + t * 100 + idx) as f32;
                    let s1 = 1 * cap_seq_stride + t * per_seq_elems + idx;
                    slice[s1] = (9000 + t * 100 + idx) as f32;
                }
            }
        }
        // Conv capture (row-major; slot stride = conv_cap_seq_stride):
        {
            let cap = cache.linear_attn[0]
                .conv_capture_states
                .as_mut()
                .expect("conv cap mut");
            let slice = cap.as_mut_slice::<f32>().expect("conv cap mut slice");
            for t in 0..(n_tokens_max as usize) {
                for idx in 0..conv_per_seq {
                    let s0 = 0 * conv_cap_seq_stride + t * conv_per_seq + idx;
                    slice[s0] = (2000 + t * 200 + idx) as f32;
                    let s1 = 1 * conv_cap_seq_stride + t * conv_per_seq + idx;
                    slice[s1] = (8000 + t * 200 + idx) as f32;
                }
            }
        }

        // Seed active recurrent + conv_state for slots 1, 2, 3 with
        // distinguishable patterns so the post-rollback snapshot can prove
        // they were untouched.
        {
            let rec = &mut cache.linear_attn[0].recurrent;
            let s = rec.as_mut_slice::<f32>().expect("rec mut");
            assert_eq!(s.len(), per_seq_elems * 4);
            for slot in 0..4usize {
                for i in 0..per_seq_elems {
                    s[slot * per_seq_elems + i] = (5_000_000 + slot * 1000 + i) as f32;
                }
            }
        }
        {
            let cs = &mut cache.linear_attn[0].conv_state;
            let s = cs.as_mut_slice::<f32>().expect("conv_state mut");
            assert_eq!(s.len(), conv_per_seq * 4);
            for slot in 0..4usize {
                for i in 0..conv_per_seq {
                    s[slot * conv_per_seq + i] = (6_000_000 + slot * 2000 + i) as f32;
                }
            }
        }

        // Snapshot slots 1, 2, 3 BEFORE rollback.
        let pre_rec_slot1: Vec<f32> = cache.linear_attn[0].recurrent.as_slice::<f32>().unwrap()
            [per_seq_elems..2 * per_seq_elems]
            .to_vec();
        let pre_rec_slot2: Vec<f32> = cache.linear_attn[0].recurrent.as_slice::<f32>().unwrap()
            [2 * per_seq_elems..3 * per_seq_elems]
            .to_vec();
        let pre_rec_slot3: Vec<f32> = cache.linear_attn[0].recurrent.as_slice::<f32>().unwrap()
            [3 * per_seq_elems..4 * per_seq_elems]
            .to_vec();
        let pre_conv_slot1: Vec<f32> = cache.linear_attn[0].conv_state.as_slice::<f32>().unwrap()
            [conv_per_seq..2 * conv_per_seq]
            .to_vec();
        let pre_conv_slot2: Vec<f32> = cache.linear_attn[0].conv_state.as_slice::<f32>().unwrap()
            [2 * conv_per_seq..3 * conv_per_seq]
            .to_vec();
        let pre_conv_slot3: Vec<f32> = cache.linear_attn[0].conv_state.as_slice::<f32>().unwrap()
            [3 * conv_per_seq..4 * conv_per_seq]
            .to_vec();

        // Rollback ONLY slot 0 to token index 2.
        cache
            .rollback_la_to(crate::serve::multi_seq_kv::SlotId(0), 2)
            .expect("rollback slot 0 ok");

        // Verify slot 0's recurrent now contains the capture[s=0, t=2] pattern.
        let post_rec_slot0: Vec<f32> =
            cache.linear_attn[0].recurrent.as_slice::<f32>().unwrap()[0..per_seq_elems].to_vec();
        for (idx, &v) in post_rec_slot0.iter().enumerate() {
            assert_eq!(
                v,
                (1000 + 2 * 100 + idx) as f32,
                "H33: slot 0 recurrent[{idx}] after rollback to (slot=0, t=2)"
            );
        }

        // Slot 1's, 2's, 3's recurrent must be byte-identical to pre-rollback.
        let post_rec_slot1: Vec<f32> = cache.linear_attn[0].recurrent.as_slice::<f32>().unwrap()
            [per_seq_elems..2 * per_seq_elems]
            .to_vec();
        assert_eq!(
            post_rec_slot1, pre_rec_slot1,
            "H33 FALSIFIED: slot 1 recurrent perturbed by rollback of slot 0"
        );
        let post_rec_slot2: Vec<f32> = cache.linear_attn[0].recurrent.as_slice::<f32>().unwrap()
            [2 * per_seq_elems..3 * per_seq_elems]
            .to_vec();
        assert_eq!(
            post_rec_slot2, pre_rec_slot2,
            "H33 FALSIFIED: slot 2 recurrent perturbed by rollback of slot 0"
        );
        let post_rec_slot3: Vec<f32> = cache.linear_attn[0].recurrent.as_slice::<f32>().unwrap()
            [3 * per_seq_elems..4 * per_seq_elems]
            .to_vec();
        assert_eq!(
            post_rec_slot3, pre_rec_slot3,
            "H33 FALSIFIED: slot 3 recurrent perturbed by rollback of slot 0"
        );

        // Slot 1's, 2's, 3's conv_state must be byte-identical too.
        let post_conv_slot1: Vec<f32> = cache.linear_attn[0].conv_state.as_slice::<f32>().unwrap()
            [conv_per_seq..2 * conv_per_seq]
            .to_vec();
        assert_eq!(
            post_conv_slot1, pre_conv_slot1,
            "H33 FALSIFIED: slot 1 conv_state perturbed by rollback of slot 0"
        );
        let post_conv_slot2: Vec<f32> = cache.linear_attn[0].conv_state.as_slice::<f32>().unwrap()
            [2 * conv_per_seq..3 * conv_per_seq]
            .to_vec();
        assert_eq!(
            post_conv_slot2, pre_conv_slot2,
            "H33 FALSIFIED: slot 2 conv_state perturbed by rollback of slot 0"
        );
        let post_conv_slot3: Vec<f32> = cache.linear_attn[0].conv_state.as_slice::<f32>().unwrap()
            [3 * conv_per_seq..4 * conv_per_seq]
            .to_vec();
        assert_eq!(
            post_conv_slot3, pre_conv_slot3,
            "H33 FALSIFIED: slot 3 conv_state perturbed by rollback of slot 0"
        );
    }

    /// H34 — n_seqs=1 byte-equivalence (regression pin).
    ///
    /// Pins that at `n_seqs=1`, the new per-slot `rollback_la_to(SlotId(0),
    /// accepted_idx)` produces a recurrent + conv_state byte-identical to
    /// the pre-A2b legacy `rollback_la_to(accepted_idx)` (which used
    /// `state_elems = recurrent.element_count()` — coincidentally equal to
    /// per-seq elems at n_seqs=1).
    ///
    /// The legacy code path is reconstructed inline (whole-buffer
    /// `state_elems` math + flat memcpy) on a second cache with the same
    /// seed; the two `recurrent` + `conv_state` buffers must be
    /// bit-exact.
    #[test]
    fn h34_rollback_la_to_n_seqs_1_byte_equivalence_2026_05_29() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let n_tokens_max = 4u32;

        // Cache (a): exercise new per-slot rollback path.
        let mut cache_a = HybridKvCache::new(&cfg, &device, 64, 1).expect("alloc a");
        cache_a
            .ensure_la_capture(&cfg, &device, n_tokens_max)
            .expect("ensure a");

        // Cache (b): identical seed, used as the "shadow" for legacy math.
        let mut cache_b = HybridKvCache::new(&cfg, &device, 64, 1).expect("alloc b");
        cache_b
            .ensure_la_capture(&cfg, &device, n_tokens_max)
            .expect("ensure b");

        let per_seq_elems = (cfg.linear_key_head_dim as usize)
            * (cfg.linear_value_head_dim as usize)
            * (cfg.linear_num_value_heads as usize);
        let conv_channels = conv_channels_for(&cfg) as usize;
        let k_minus1 = (cfg.linear_conv_kernel_dim.saturating_sub(1)) as usize;
        let conv_per_seq = conv_channels * k_minus1;

        // Identical capture pattern for both caches.
        let seed_capture = |cache: &mut HybridKvCache| {
            for la in cache.linear_attn.iter_mut() {
                let cap = la.capture_states.as_mut().unwrap();
                let s = cap.as_mut_slice::<f32>().unwrap();
                for t in 0..(n_tokens_max as usize) {
                    for idx in 0..per_seq_elems {
                        s[t * per_seq_elems + idx] = (t as f32) * 1000.0 + idx as f32 + 0.5;
                    }
                }
                let conv_cap = la.conv_capture_states.as_mut().unwrap();
                let cs = conv_cap.as_mut_slice::<f32>().unwrap();
                for t in 0..(n_tokens_max as usize) {
                    for idx in 0..conv_per_seq {
                        cs[t * conv_per_seq + idx] = (t as f32) * 3000.0 + idx as f32 + 0.25;
                    }
                }
            }
        };
        seed_capture(&mut cache_a);
        seed_capture(&mut cache_b);

        // Cache (a): use the production per-slot rollback.
        cache_a
            .rollback_la_to(crate::serve::multi_seq_kv::SlotId(0), 2)
            .expect("rollback a");

        // Cache (b): simulate the pre-A2b legacy math (whole-buffer
        // state_elems + flat memcpy + conv re-index loop) inline. At
        // n_seqs=1 this is provably identical to per-seq math.
        for slot_data in cache_b.linear_attn.iter_mut() {
            let capture = slot_data.capture_states.as_ref().unwrap();
            let state_elems = slot_data.recurrent.element_count();
            // At n_seqs=1: state_elems == per_seq_elems.
            assert_eq!(state_elems, per_seq_elems);
            let cap_slice = capture.as_slice::<f32>().unwrap();
            let src_offset = 2 * state_elems;
            let src_owned: Vec<f32> = cap_slice[src_offset..src_offset + state_elems].to_vec();
            let dst = slot_data.recurrent.as_mut_slice::<f32>().unwrap();
            dst.copy_from_slice(&src_owned);

            let conv_capture = slot_data.conv_capture_states.as_ref().unwrap();
            let conv_state_elems = slot_data.conv_state.element_count();
            assert_eq!(conv_state_elems, conv_per_seq);
            let conv_cap_slice = conv_capture.as_slice::<f32>().unwrap();
            let conv_src_offset = 2 * conv_state_elems;
            let conv_src_owned: Vec<f32> =
                conv_cap_slice[conv_src_offset..conv_src_offset + conv_state_elems].to_vec();
            let conv_dst = slot_data.conv_state.as_mut_slice::<f32>().unwrap();
            for k_i in 0..k_minus1 {
                for c in 0..conv_channels {
                    let src_idx = k_i * conv_channels + c;
                    let dst_idx = c * k_minus1 + k_i;
                    conv_dst[dst_idx] = conv_src_owned[src_idx];
                }
            }
        }

        // Byte-equality of recurrent + conv_state across both caches.
        for (la_a, la_b) in cache_a.linear_attn.iter().zip(cache_b.linear_attn.iter()) {
            let ra = la_a.recurrent.as_slice::<f32>().unwrap();
            let rb = la_b.recurrent.as_slice::<f32>().unwrap();
            assert_eq!(
                ra, rb,
                "H34 FALSIFIED: recurrent bytes differ between A2b per-slot \
                 path and legacy whole-buffer path at n_seqs=1"
            );
            let ca = la_a.conv_state.as_slice::<f32>().unwrap();
            let cb = la_b.conv_state.as_slice::<f32>().unwrap();
            assert_eq!(
                ca, cb,
                "H34 FALSIFIED: conv_state bytes differ between A2b per-slot \
                 path and legacy whole-buffer path at n_seqs=1"
            );
        }
    }

    /// H35 — slot out-of-range typed error.
    ///
    /// Pins that `rollback_la_to(SlotId(99), 0)` returns `Err` whose Display
    /// message names "SlotOutOfRange" and "ADR-040 Phase A2b" — bounds-first
    /// per iter-1.5 cfa-finding-F5 ordering.
    ///
    /// Also pins that the error is raised BEFORE any
    /// `ensure_la_capture` check: a fresh cache without ensure_la_capture
    /// at slot=99 still surfaces SlotOutOfRange (not the
    /// "capture_states is None" message).
    #[test]
    fn h35_rollback_la_to_slot_out_of_range_typed_2026_05_29() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();

        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc n_seqs=4");
        cache.ensure_la_capture(&cfg, &device, 4).expect("ensure");

        // SlotId(4) is one past the valid range [0, 3].
        let err = cache
            .rollback_la_to(crate::serve::multi_seq_kv::SlotId(4), 0)
            .expect_err("slot 4 OOR for n_seqs=4");
        let msg = format!("{err}");
        assert!(
            msg.contains("SlotOutOfRange"),
            "H35: error message must contain 'SlotOutOfRange'; got: {msg}"
        );
        assert!(
            msg.contains("slot=4"),
            "H35: error message must surface slot id; got: {msg}"
        );
        assert!(
            msg.contains("max_slots=4"),
            "H35: error message must surface max_slots; got: {msg}"
        );
        assert!(
            msg.contains("ADR-040 Phase A2b"),
            "H35: error message must name the iter that introduced bounds; got: {msg}"
        );

        // SlotId(99) — same family.
        let err = cache
            .rollback_la_to(crate::serve::multi_seq_kv::SlotId(99), 0)
            .expect_err("slot 99 OOR");
        assert!(
            format!("{err}").contains("SlotOutOfRange"),
            "H35: SlotId(99) must also surface SlotOutOfRange"
        );

        // Bounds-first ordering: cache WITHOUT ensure_la_capture at slot=99
        // still surfaces SlotOutOfRange (not the "capture_states is None"
        // message). This pins cfa-finding-F5's "bounds before pre-condition"
        // ordering against any future iter that re-orders the validation.
        let mut cache_no_cap = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");
        let err = cache_no_cap
            .rollback_la_to(crate::serve::multi_seq_kv::SlotId(99), 0)
            .expect_err("slot OOR before ensure_la_capture");
        let msg = format!("{err}");
        assert!(
            msg.contains("SlotOutOfRange"),
            "H35: bounds-first — SlotOutOfRange must surface BEFORE \
             capture_states None check; got: {msg}"
        );
        assert!(
            !msg.contains("capture_states is None"),
            "H35: bounds-first — capture_states-None message must NOT \
             leak past the slot-OOR guard; got: {msg}"
        );
    }

    /// **iter-C2d-cont-kernel iter-1 — reset_for_slot per-slot
    /// isolation (2026-05-29)**.
    ///
    /// Pin: `reset_for_slot(SlotId(s))` ONLY zeros the slot-`s` region
    /// in linear_attn conv_state + conv_state_scratch + recurrent +
    /// recurrent_scratch (per-slot slice math at offset
    /// `s * per_seq_elems`); other slots' bytes are byte-untouched.
    /// And full_attn current_len[slot=s] = 0; other slots' cursors
    /// untouched.
    ///
    /// Falsifier shape: seed every slot with distinct non-zero
    /// patterns, call `reset_for_slot(SlotId(1))`, then assert
    /// (a) slot 1's per-seq region is zero in all 4 LA buffers and
    /// (b) slots 0, 2, 3 keep their seeded bytes verbatim.
    #[test]
    fn iter_c2d_cont_kernel_iter1_reset_for_slot_per_slot_isolation_2026_05_29() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let n_seqs: u32 = 4;
        let mut cache = HybridKvCache::new(&cfg, &device, 64, n_seqs).expect("alloc");

        let per_seq_rec = (cfg.linear_key_head_dim as usize)
            * (cfg.linear_value_head_dim as usize)
            * (cfg.linear_num_value_heads as usize);
        let conv_channels = conv_channels_for(&cfg) as usize;
        let k_minus1 = (cfg.linear_conv_kernel_dim.saturating_sub(1)) as usize;
        let per_seq_conv = conv_channels * k_minus1;

        // Seed every slot's per-seq region in every LA buffer with
        // distinct patterns. Slot s, buffer kind k → base = s*1000 +
        // k*100 + 1.0 — guarantees non-zero everywhere.
        for la in cache.linear_attn.iter_mut() {
            for (kind, total) in [
                (0u32, la.conv_state.element_count()),
                (1u32, la.conv_state_scratch.element_count()),
            ] {
                let buf = if kind == 0 {
                    la.conv_state.as_mut_slice::<f32>().unwrap()
                } else {
                    la.conv_state_scratch.as_mut_slice::<f32>().unwrap()
                };
                assert_eq!(total, n_seqs as usize * per_seq_conv);
                for s in 0..(n_seqs as usize) {
                    let start = s * per_seq_conv;
                    for idx in 0..per_seq_conv {
                        buf[start + idx] = (s as f32) * 1000.0
                            + (kind as f32) * 100.0
                            + (idx as f32) * 0.001
                            + 1.0;
                    }
                }
            }
            for (kind, total) in [
                (2u32, la.recurrent.element_count()),
                (3u32, la.recurrent_scratch.element_count()),
            ] {
                let buf = if kind == 2 {
                    la.recurrent.as_mut_slice::<f32>().unwrap()
                } else {
                    la.recurrent_scratch.as_mut_slice::<f32>().unwrap()
                };
                assert_eq!(total, n_seqs as usize * per_seq_rec);
                for s in 0..(n_seqs as usize) {
                    let start = s * per_seq_rec;
                    for idx in 0..per_seq_rec {
                        buf[start + idx] = (s as f32) * 1000.0
                            + (kind as f32) * 100.0
                            + (idx as f32) * 0.001
                            + 1.0;
                    }
                }
            }
        }
        // Seed full_attn current_len cursors with distinct non-zero
        // values per slot.
        for fa in cache.full_attn.iter_mut() {
            for s in 0..(n_seqs as usize) {
                fa.current_len[s] = (s as u32) + 17;
            }
        }

        // Call reset_for_slot(SlotId(1)).
        cache
            .reset_for_slot(crate::serve::multi_seq_kv::SlotId(1))
            .expect("reset_for_slot(1)");

        // Slot 1's per-seq region is zero in all 4 LA buffers;
        // other slots untouched.
        for la in cache.linear_attn.iter() {
            for (kind, buf_slice) in [
                (0u32, la.conv_state.as_slice::<f32>().unwrap()),
                (1u32, la.conv_state_scratch.as_slice::<f32>().unwrap()),
            ] {
                for s in 0..(n_seqs as usize) {
                    let start = s * per_seq_conv;
                    for idx in 0..per_seq_conv {
                        let v = buf_slice[start + idx];
                        if s == 1 {
                            assert!(
                                v == 0.0,
                                "iter-1: slot 1 conv buf kind={kind} idx={idx} \
                                 must be 0 after reset_for_slot(1); got {v}"
                            );
                        } else {
                            let expected = (s as f32) * 1000.0
                                + (kind as f32) * 100.0
                                + (idx as f32) * 0.001
                                + 1.0;
                            assert!(
                                (v - expected).abs() < 1e-6,
                                "iter-1: slot {s} conv buf kind={kind} idx={idx} \
                                 must be untouched (={expected}); got {v}"
                            );
                        }
                    }
                }
            }
            for (kind, buf_slice) in [
                (2u32, la.recurrent.as_slice::<f32>().unwrap()),
                (3u32, la.recurrent_scratch.as_slice::<f32>().unwrap()),
            ] {
                for s in 0..(n_seqs as usize) {
                    let start = s * per_seq_rec;
                    for idx in 0..per_seq_rec {
                        let v = buf_slice[start + idx];
                        if s == 1 {
                            assert!(
                                v == 0.0,
                                "iter-1: slot 1 rec buf kind={kind} idx={idx} \
                                 must be 0 after reset_for_slot(1); got {v}"
                            );
                        } else {
                            let expected = (s as f32) * 1000.0
                                + (kind as f32) * 100.0
                                + (idx as f32) * 0.001
                                + 1.0;
                            assert!(
                                (v - expected).abs() < 1e-6,
                                "iter-1: slot {s} rec buf kind={kind} idx={idx} \
                                 must be untouched (={expected}); got {v}"
                            );
                        }
                    }
                }
            }
        }
        // Slot 1's full_attn cursor must be 0; others untouched.
        for fa in cache.full_attn.iter() {
            for s in 0..(n_seqs as usize) {
                if s == 1 {
                    assert_eq!(fa.current_len[s], 0, "iter-1: slot 1 current_len must be 0");
                } else {
                    assert_eq!(
                        fa.current_len[s],
                        (s as u32) + 17,
                        "iter-1: slot {s} current_len must be untouched"
                    );
                }
            }
        }
    }

    /// **iter-C2d-cont-kernel iter-1 — reset_for_slot bounds-first
    /// typed error (2026-05-29)**.
    ///
    /// Mirror of H35 for the new per-slot reset primitive. Pin:
    /// `reset_for_slot(SlotId(s)) where s >= n_seqs` returns Err
    /// with `SlotOutOfRange` + the iter cite in the message.
    #[test]
    fn iter_c2d_cont_kernel_iter1_reset_for_slot_bounds_typed_2026_05_29() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");

        let err = cache
            .reset_for_slot(crate::serve::multi_seq_kv::SlotId(4))
            .expect_err("slot 4 OOR for n_seqs=4");
        let msg = format!("{err}");
        assert!(
            msg.contains("SlotOutOfRange"),
            "iter-1: error message must contain 'SlotOutOfRange'; got: {msg}"
        );
        assert!(
            msg.contains("slot=4"),
            "iter-1: error message must surface slot id; got: {msg}"
        );
        assert!(
            msg.contains("max_slots=4"),
            "iter-1: error message must surface max_slots; got: {msg}"
        );
        assert!(
            msg.contains("iter-C2d-cont-kernel iter-1"),
            "iter-1: error must name implementing iter; got: {msg}"
        );

        // SlotId(0) on a valid n_seqs=1 cache is the byte-equivalence
        // case — must succeed (zero-elements zeroed but no error).
        let mut cache1 = HybridKvCache::new(&cfg, &device, 64, 1).expect("alloc n_seqs=1");
        cache1
            .reset_for_slot(crate::serve::multi_seq_kv::SlotId(0))
            .expect("SlotId(0) at n_seqs=1 must succeed");
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-040 Phase A2c + A3c (2026-05-30) — fork_seq REAL cross-slot
    // copy hypothesis bank.  Pinning the iter-A2c (Qwen35) +
    // iter-A3c (Gemma 4 — see gemma4/kv_cache.rs) joint dispatcher
    // closure per dossier §2.3.3.
    //
    // Qwen35 scope (this file): H158 + H163-H166 — the HybridKvCache
    // full-attn + linear-attn + MTP + capture-buffer end-to-end fork
    // proof.  The Gemma 4 sibling-struct lifts H159-H162 land in
    // gemma4/kv_cache.rs (one test per sibling struct).
    // ──────────────────────────────────────────────────────────────────

    /// **H158** — Qwen35 `HybridKvCache::fork_seq` cross-slot copy
    /// returns `Ok(())`, copies only cursor-visible full-attn K/V rows,
    /// and leaves the destination's lazy tail untouched. Replaces the A2a typed-clamp
    /// `CapabilityUnsupported` envelope at `kv_cache.rs:3044-3099`.
    ///
    /// Falsifier (any one of these fires ⇒ H158 broken):
    /// 1. `fork_seq(src, dst)` returns Err.
    /// 2. dst's live full-attn K/V prefix differs from src.
    /// 3. dst's lazy full-attn K/V tail is overwritten.
    #[test]
    fn h158_qwen35_hybrid_kv_fork_seq_copies_only_live_prefix() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let max_seq_len = 64u32;
        let mut cache = HybridKvCache::new(&cfg, &device, max_seq_len, 4).expect("alloc n_seqs=4");

        // Seed slot 0's full-attn K/V bytes with deterministic
        // non-zero patterns per layer.  Production write path is the
        // forward-path kernel dispatcher; this test exercises the
        // fork-copy contract via host-side byte writes.
        let nkv = cfg.num_key_value_heads as usize;
        let hd = cfg.head_dim as usize;
        let cap = max_seq_len as usize;
        let slot_elems = nkv * cap * hd;
        let slot_bytes_f32 = slot_elems * std::mem::size_of::<f32>();
        const DST_TAIL: u8 = 0xE5;

        for (layer_idx, slot) in cache.full_attn.iter_mut().enumerate() {
            if let Some(ref mut k) = slot.k {
                let s = k.as_mut_slice::<u8>().expect("K u8");
                for (i, b) in s[..slot_bytes_f32].iter_mut().enumerate() {
                    *b = (((layer_idx * 13 + i) % 251) + 1) as u8;
                }
                s[slot_bytes_f32..2 * slot_bytes_f32].fill(DST_TAIL);
            }
            if let Some(ref mut v) = slot.v {
                let s = v.as_mut_slice::<u8>().expect("V u8");
                for (i, b) in s[..slot_bytes_f32].iter_mut().enumerate() {
                    *b = (((layer_idx * 19 + i) % 253) + 1) as u8;
                }
                s[slot_bytes_f32..2 * slot_bytes_f32].fill(DST_TAIL);
            }
        }
        // Bump slot 0's cursor. Only these seven positions are readable and
        // may be copied into the destination's overwrite-backed region.
        cache.append_for_seq(SlotId(0), 7).unwrap();

        // iter-A2c closure: fork must return Ok(()).
        cache
            .fork_seq(SlotId(0), SlotId(1))
            .expect("H158: fork_seq must succeed post-A2c");

        // Per-layer byte-equality for the cursor-visible prefix, while the
        // destination tail keeps its sentinel.
        for (layer_idx, slot) in cache.full_attn.iter().enumerate() {
            for (name, buffer) in [("K", slot.k.as_ref()), ("V", slot.v.as_ref())] {
                let Some(buffer) = buffer else { continue };
                let bytes = buffer.as_slice::<u8>().expect("fork bytes");
                let head_stride = cap * hd * std::mem::size_of::<f32>();
                let live_bytes = 7 * hd * std::mem::size_of::<f32>();
                for head in 0..nkv {
                    let src = head * head_stride;
                    let dst = slot_bytes_f32 + head * head_stride;
                    assert_eq!(
                        &bytes[dst..dst + live_bytes],
                        &bytes[src..src + live_bytes],
                        "H158 FALSIFIED: full_attn[{layer_idx}] {name} head {head} live prefix"
                    );
                    assert!(
                        bytes[dst + live_bytes..dst + head_stride]
                            .iter()
                            .all(|&byte| byte == DST_TAIL),
                        "H158 FALSIFIED: full_attn[{layer_idx}] {name} head {head} tail overwritten"
                    );
                }
            }
        }
    }

    /// **H163** — `HybridKvCache::fork_seq` does NOT modify the source
    /// slot's bytes (copy not move).
    ///
    /// Falsifier: any per-layer K/V byte at slot 0's region differs
    /// from the pre-fork snapshot.
    #[test]
    fn h163_qwen35_hybrid_kv_fork_seq_src_unchanged() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let max_seq_len = 64u32;
        let mut cache = HybridKvCache::new(&cfg, &device, max_seq_len, 4).expect("alloc n_seqs=4");
        let nkv = cfg.num_key_value_heads as usize;
        let hd = cfg.head_dim as usize;
        let cap = max_seq_len as usize;
        let slot_bytes_f32 = nkv * cap * hd * std::mem::size_of::<f32>();
        // Seed slot 0's K/V with non-zero data.
        for (layer_idx, slot) in cache.full_attn.iter_mut().enumerate() {
            if let Some(ref mut k) = slot.k {
                let s = k.as_mut_slice::<u8>().expect("K u8");
                for (i, b) in s[..slot_bytes_f32].iter_mut().enumerate() {
                    *b = (((layer_idx * 23 + i) % 251) + 1) as u8;
                }
            }
        }
        cache.append_for_seq(SlotId(0), 9).unwrap();

        // Snapshot SOURCE slot 0's bytes BEFORE the fork.
        let src_before: Vec<Vec<u8>> = cache
            .full_attn
            .iter()
            .map(|s| {
                s.k.as_ref()
                    .map(|k| k.as_slice::<u8>().unwrap()[..slot_bytes_f32].to_vec())
                    .unwrap_or_default()
            })
            .collect();
        let src_cursor_before: Vec<u32> =
            cache.full_attn.iter().map(|s| s.current_len[0]).collect();

        cache.fork_seq(SlotId(0), SlotId(2)).expect("H163: fork ok");

        // src slot 0's bytes must be UNCHANGED.
        for (layer_idx, slot) in cache.full_attn.iter().enumerate() {
            if let Some(ref k) = slot.k {
                let src_after: Vec<u8> = k.as_slice::<u8>().unwrap()[..slot_bytes_f32].to_vec();
                assert_eq!(
                    src_before[layer_idx], src_after,
                    "H163 FALSIFIED: full_attn[{layer_idx}] src slot 0 K bytes \
                     mutated by fork_seq"
                );
            }
        }
        // src slot 0's cursor must also be unchanged.
        for (layer_idx, slot) in cache.full_attn.iter().enumerate() {
            assert_eq!(
                slot.current_len[0], src_cursor_before[layer_idx],
                "H163 FALSIFIED: full_attn[{layer_idx}] src slot 0 cursor mutated \
                 by fork_seq"
            );
        }
    }

    /// **H164** — `HybridKvCache::fork_seq`: dst slot bytes are
    /// byte-identical to src slot bytes for EVERY buffer the cache
    /// carries (full_attn K/V, MTP K/V if present, linear-attn
    /// recurrent + conv_state, capture buffers if present).  Extends
    /// H158 with the linear-attn surface.
    #[test]
    fn h164_qwen35_hybrid_kv_fork_seq_dst_matches_src_all_buffers() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let max_seq_len = 64u32;
        let n_seqs = 4u32;
        let mut cache = HybridKvCache::new(&cfg, &device, max_seq_len, n_seqs).expect("alloc");

        // Seed linear-attn recurrent + conv_state bytes for slot 0
        // (n_seqs is the LAST shape dim ⇒ outermost in memory; per-slot
        // byte stride = total_bytes / n_seqs).
        for (layer_idx, slot) in cache.linear_attn.iter_mut().enumerate() {
            let total_rec = slot.recurrent.byte_len();
            assert_eq!(total_rec % (n_seqs as usize), 0);
            let per_slot_rec = total_rec / (n_seqs as usize);
            let s = slot.recurrent.as_mut_slice::<u8>().expect("recurrent u8");
            for (i, b) in s[..per_slot_rec].iter_mut().enumerate() {
                *b = (((layer_idx * 29 + i) % 251) + 1) as u8;
            }
            let total_conv = slot.conv_state.byte_len();
            assert_eq!(total_conv % (n_seqs as usize), 0);
            let per_slot_conv = total_conv / (n_seqs as usize);
            let s = slot.conv_state.as_mut_slice::<u8>().expect("conv_state u8");
            for (i, b) in s[..per_slot_conv].iter_mut().enumerate() {
                *b = (((layer_idx * 31 + i) % 253) + 1) as u8;
            }
        }
        cache.append_for_seq(SlotId(0), 5).unwrap();

        cache.fork_seq(SlotId(0), SlotId(3)).expect("H164: fork ok");

        // Per-layer dst slot 3 byte-equality on linear-attn buffers.
        for (layer_idx, slot) in cache.linear_attn.iter().enumerate() {
            let per_slot_rec = slot.recurrent.byte_len() / (n_seqs as usize);
            let bytes = slot.recurrent.as_slice::<u8>().unwrap();
            let src_off = 0;
            let dst_off = 3 * per_slot_rec;
            assert_eq!(
                &bytes[src_off..src_off + per_slot_rec],
                &bytes[dst_off..dst_off + per_slot_rec],
                "H164 FALSIFIED: linear_attn[{layer_idx}] recurrent dst slot 3 \
                 bytes do not match src slot 0"
            );
            let per_slot_conv = slot.conv_state.byte_len() / (n_seqs as usize);
            let cbytes = slot.conv_state.as_slice::<u8>().unwrap();
            let csrc_off = 0;
            let cdst_off = 3 * per_slot_conv;
            assert_eq!(
                &cbytes[csrc_off..csrc_off + per_slot_conv],
                &cbytes[cdst_off..cdst_off + per_slot_conv],
                "H164 FALSIFIED: linear_attn[{layer_idx}] conv_state dst slot 3 \
                 bytes do not match src slot 0"
            );
        }
    }

    /// **H165** — `HybridKvCache::fork_seq` copies cursor (per-layer
    /// `current_len`) from src to dst.
    #[test]
    fn h165_qwen35_hybrid_kv_fork_seq_cursor_copied() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");
        cache.append_for_seq(SlotId(0), 11).unwrap();
        cache.append_for_seq(SlotId(2), 5).unwrap();
        // Pre-fork: src cursor = 11, dst (slot 3) cursor = 0.
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 11);
        assert_eq!(cache.seq_len(SlotId(3)).unwrap(), 0);
        cache.fork_seq(SlotId(0), SlotId(3)).expect("H165: fork ok");
        // Post-fork: dst cursor must equal src cursor.
        assert_eq!(
            cache.seq_len(SlotId(3)).unwrap(),
            11,
            "H165 FALSIFIED: dst cursor != src cursor after fork"
        );
        // src cursor unchanged.
        assert_eq!(
            cache.seq_len(SlotId(0)).unwrap(),
            11,
            "H165 FALSIFIED: src cursor mutated by fork"
        );
        // Untouched sibling slot 2 unchanged.
        assert_eq!(
            cache.seq_len(SlotId(2)).unwrap(),
            5,
            "H165 FALSIFIED: untouched sibling slot 2 cursor mutated"
        );
    }

    /// **H166** — `HybridKvCache::fork_seq` returns typed errors for
    /// out-of-range src/dst (src checked first per iter-1.5
    /// cfa-finding-F5).  Same-slot fork (src == dst) is a successful
    /// no-op per trait spec.
    #[test]
    fn h166_qwen35_hybrid_kv_fork_seq_typed_errors() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 4).expect("alloc");

        // (1) src OOR returns SlotOutOfRange (src reported FIRST per
        // iter-1.5 cfa-finding-F5 deterministic ordering).
        let err = cache
            .fork_seq(SlotId(4), SlotId(0))
            .expect_err("H166: src OOR");
        match err {
            MultiSeqError::SlotOutOfRange { slot, max_slots } => {
                assert_eq!(slot.0, 4);
                assert_eq!(max_slots, 4);
            }
            other => panic!("H166: expected SlotOutOfRange; got {other:?}"),
        }

        // (2) dst OOR returns SlotOutOfRange after src bounds-check pass.
        let err = cache
            .fork_seq(SlotId(0), SlotId(4))
            .expect_err("H166: dst OOR");
        match err {
            MultiSeqError::SlotOutOfRange { slot, max_slots } => {
                assert_eq!(slot.0, 4);
                assert_eq!(max_slots, 4);
            }
            other => panic!("H166: expected SlotOutOfRange; got {other:?}"),
        }

        // (3) BOTH src and dst OOR — src is reported first
        // (deterministic ordering).
        let err = cache
            .fork_seq(SlotId(7), SlotId(8))
            .expect_err("H166: both OOR");
        match err {
            MultiSeqError::SlotOutOfRange { slot, max_slots } => {
                assert_eq!(slot.0, 7, "H166: src reported first (not dst)");
                assert_eq!(max_slots, 4);
            }
            other => panic!("H166: expected SlotOutOfRange; got {other:?}"),
        }

        // (4) Same-slot fork is a successful no-op per trait spec.
        cache.append_for_seq(SlotId(2), 7).unwrap();
        cache
            .fork_seq(SlotId(2), SlotId(2))
            .expect("H166: same-slot fork must be a successful no-op");
        assert_eq!(
            cache.seq_len(SlotId(2)).unwrap(),
            7,
            "H166: same-slot fork preserves cursor"
        );
    }

    /// ADR-040 M-QWEN (2026-07-01) — per-slot ping-pong parity semantics.
    ///
    /// The N≥2 concurrent divergence root cause was the whole-buffer
    /// `std::mem::swap` of `LinearAttnStateSlot` conv/recurrent ping-pong
    /// buffers: one slot's post-tick swap flipped read/write roles under
    /// every OTHER active slot. This pins the replacement semantics:
    /// (1) `swap_for_slot` flips ONE slot's roles and leaves the others'
    ///     current-state reads untouched;
    /// (2) `snapshot()` is parity-canonical (each slot's region taken from
    ///     its CURRENT buffer);
    /// (3) `fork_seq` carries parity src→dst so dst-current == src-current;
    /// (4) `reset_for_slot` returns the slot to canonical parity;
    /// (5) `rollback_la_to` under flipped parity lands in the slot's
    ///     CURRENT (scratch-named) buffer.
    #[test]
    fn la_ping_pong_per_slot_parity_semantics_2026_07_01() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Ok(device) = MlxDevice::new() else {
            eprintln!("[skip] la_ping_pong_per_slot_parity_semantics — no Metal device");
            return;
        };
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let n_seqs = 4u32;
        let mut cache = HybridKvCache::new(&cfg, &device, 64, n_seqs).expect("alloc n_seqs=4");
        assert!(!cache.linear_attn.is_empty(), "cfg must have linear layers");

        // Sentinel-fill layer 0's four buffers: current-slot-region s = 100+s,
        // scratch-slot-region s = 200+s (conv), 300+s / 400+s (recurrent).
        {
            let la = &mut cache.linear_attn[0];
            let n = n_seqs as usize;
            let fill = |buf: &mut MlxBuffer, base: f32| {
                let s = buf.as_mut_slice::<f32>().expect("as_mut_slice");
                let per = s.len() / n;
                for i in 0..n {
                    for v in &mut s[i * per..(i + 1) * per] {
                        *v = base + i as f32;
                    }
                }
            };
            fill(&mut la.conv_state, 100.0);
            fill(&mut la.conv_state_scratch, 200.0);
            fill(&mut la.recurrent, 300.0);
            fill(&mut la.recurrent_scratch, 400.0);
        }
        let read_slot0th = |buf: &MlxBuffer, slot: usize, n: usize| -> f32 {
            let s = buf.as_slice::<f32>().expect("as_slice");
            let per = s.len() / n;
            s[slot * per]
        };

        // (1) Flip slot 1 only.
        cache.linear_attn[0].swap_for_slot(SlotId(1));
        {
            let la = &cache.linear_attn[0];
            let n = n_seqs as usize;
            // Slot 1's CURRENT is now the scratch-named buffer.
            let (c1, s1) = la.conv_bufs_for_slot(SlotId(1));
            assert_eq!(
                read_slot0th(c1, 1, n),
                201.0,
                "slot1 conv current = scratch region"
            );
            assert_eq!(
                read_slot0th(s1, 1, n),
                101.0,
                "slot1 conv scratch = named region"
            );
            let (r1, _) = la.recurrent_bufs_for_slot(SlotId(1));
            assert_eq!(
                read_slot0th(r1, 1, n),
                401.0,
                "slot1 rec current = scratch region"
            );
            // Slots 0/2/3 untouched: current still the named buffers.
            for s in [0usize, 2, 3] {
                let (c, _) = la.conv_bufs_for_slot(SlotId(s as u32));
                assert_eq!(
                    read_slot0th(c, s, n),
                    100.0 + s as f32,
                    "slot{s} conv current unchanged"
                );
                let (r, _) = la.recurrent_bufs_for_slot(SlotId(s as u32));
                assert_eq!(
                    read_slot0th(r, s, n),
                    300.0 + s as f32,
                    "slot{s} rec current unchanged"
                );
            }
        }

        // (2) Snapshot canonicalization: slot 1 region comes from scratch.
        for slot in &mut cache.full_attn {
            if let Some(k) = slot.k.as_mut() {
                k.as_mut_slice::<u8>().expect("seed k").fill(0);
            }
            if let Some(v) = slot.v.as_mut() {
                v.as_mut_slice::<u8>().expect("seed v").fill(0);
            }
            if let Some(tq) = slot.tq.as_mut() {
                for buf in [
                    &mut tq.k_packed,
                    &mut tq.k_norms,
                    &mut tq.v_packed,
                    &mut tq.v_norms,
                ] {
                    buf.as_mut_slice::<u8>().expect("seed tq").fill(0);
                }
            }
        }
        let snap = cache
            .snapshot_inner(&device, None, None)
            .expect("fully initialized test snapshot");
        {
            let n = n_seqs as usize;
            assert_eq!(
                read_slot0th(&snap.linear_conv[0], 0, n),
                100.0,
                "snap slot0 conv = current(named)"
            );
            assert_eq!(
                read_slot0th(&snap.linear_conv[0], 1, n),
                201.0,
                "snap slot1 conv = current(scratch)"
            );
            assert_eq!(
                read_slot0th(&snap.linear_recurrent[0], 1, n),
                401.0,
                "snap slot1 rec = current(scratch)"
            );
        }

        // (3) fork_seq 1 → 2 carries parity; dst-current == src-current.
        {
            use crate::serve::multi_seq_kv::MultiSeqKvCache;
            cache.fork_seq(SlotId(1), SlotId(2)).expect("fork_seq 1→2");
            let la = &cache.linear_attn[0];
            assert!(la.pp_flipped[2], "fork carries parity");
            let n = n_seqs as usize;
            let (c2, _) = la.conv_bufs_for_slot(SlotId(2));
            assert_eq!(
                read_slot0th(c2, 2, n),
                201.0,
                "slot2 conv current == slot1's forked current"
            );
        }

        // (4) reset_for_slot returns canonical parity + zeroes.
        cache.reset_for_slot(SlotId(1)).expect("reset_for_slot 1");
        {
            let la = &cache.linear_attn[0];
            assert!(!la.pp_flipped[1], "reset returns slot1 to canonical parity");
            let n = n_seqs as usize;
            let (c1, _) = la.conv_bufs_for_slot(SlotId(1));
            assert_eq!(read_slot0th(c1, 1, n), 0.0, "reset zeroed slot1 current");
        }

        // (5) rollback under flipped parity lands in the slot's CURRENT.
        cache
            .ensure_la_capture(&cfg, &device, 2)
            .expect("ensure_la_capture");
        {
            // Fill capture position 0 for slot 3 with 777.0 (recurrent) and
            // conv capture with 888.0.
            let n = n_seqs as usize;
            let la = &mut cache.linear_attn[0];
            {
                let cap = la.capture_states.as_mut().expect("capture_states");
                let total = cap.element_count();
                let s = cap.as_mut_slice::<f32>().expect("cap slice");
                let per_seq = total / n; // [.., n_tokens_max, n_seqs] slot-major per rollback math
                for v in &mut s[3 * per_seq..3 * per_seq + per_seq] {
                    *v = 777.0;
                }
            }
            {
                let ccap = la
                    .conv_capture_states
                    .as_mut()
                    .expect("conv_capture_states");
                let s = ccap.as_mut_slice::<f32>().expect("ccap slice");
                let total = s.len();
                let per_seq = total / n;
                for v in &mut s[3 * per_seq..3 * per_seq + per_seq] {
                    *v = 888.0;
                }
            }
            la.swap_for_slot(SlotId(3)); // flip slot 3
        }
        cache
            .rollback_la_to(SlotId(3), 0)
            .expect("rollback_la_to slot3");
        {
            let la = &cache.linear_attn[0];
            let n = n_seqs as usize;
            let (r3, _) = la.recurrent_bufs_for_slot(SlotId(3));
            assert_eq!(
                read_slot0th(r3, 3, n),
                777.0,
                "rollback landed in slot3's CURRENT recurrent (parity-aware)"
            );
            let (c3, _) = la.conv_bufs_for_slot(SlotId(3));
            assert_eq!(
                read_slot0th(c3, 3, n),
                888.0,
                "rollback landed in slot3's CURRENT conv (parity-aware)"
            );
        }
    }

    // ---------------------------------------------------------------------
    // ADR-027 sub-iter 23d-γ (2026-08-03) — restore_partial TQ coverage
    // ---------------------------------------------------------------------

    /// Tiny dense cfg with an MTP slot (mirrors the multi-seq fixture but
    /// with `mtp_num_hidden_layers = 1` so the MTP branch is exercised).
    fn tiny_dense_cfg_4layer_with_mtp() -> Qwen35Config {
        let mut cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        cfg.mtp_num_hidden_layers = 1;
        cfg
    }

    /// Fill a TQ buffer set with deterministic, per-buffer-distinct byte
    /// patterns so a restore mismatch surfaces as an exact-byte diff.
    fn plant_tq_pattern(tq: &mut TqFullAttnKvBuffers, seed: usize) {
        let bufs: [&mut MlxBuffer; 4] = [
            &mut tq.k_packed,
            &mut tq.k_norms,
            &mut tq.v_packed,
            &mut tq.v_norms,
        ];
        for (bi, buf) in bufs.into_iter().enumerate() {
            let s = buf.as_mut_slice::<u8>().expect("tq mut_slice");
            for (i, b) in s.iter_mut().enumerate() {
                *b = ((seed * 13 + bi * 5 + i) % 251) as u8;
            }
        }
    }

    fn set_all_sequence_lengths(cache: &mut HybridKvCache, n_tokens: u32) {
        for slot in &mut cache.full_attn {
            slot.current_len.fill(n_tokens);
        }
        if let Some(mtp) = cache.mtp_slot.as_mut() {
            mtp.current_len.fill(n_tokens);
        }
    }

    /// Read the first `n_tokens` positions of head `head` (seq 0) from a
    /// 4-rank `[n_seqs, n_kv, max_seq, inner]` buffer as raw bytes —
    /// mirrors `partial_copy_slot`'s per-head stride math.
    fn read_head_prefix(buf: &MlxBuffer, head: usize, n_tokens: usize) -> Vec<u8> {
        let shape = buf.shape();
        let (_n_kv, max_seq, inner) = (shape[1], shape[2], shape[3]);
        let elem = buf.dtype().size_of();
        let head_stride = max_seq * inner * elem;
        let all = buf.as_slice::<u8>().expect("slice");
        all[head * head_stride..head * head_stride + n_tokens * inner * elem].to_vec()
    }

    /// LCP checkpoints must own only the addressable prefix. This pins both
    /// the reduced allocation shape and exact restoration into a larger live
    /// cache under the production TQ-only substrate.
    #[test]
    fn snapshot_prefix_compacts_sequence_buffers_and_restores_exactly() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_with_mtp();
        let max_seq_len = 64u32;
        let n_tokens = 40usize;

        let mut src = HybridKvCache::new_with_options(&cfg, &device, max_seq_len, 1, true)
            .expect("alloc src");
        for slot in &mut src.full_attn {
            plant_tq_pattern(slot.tq.as_mut().expect("tq"), 7);
        }
        plant_tq_pattern(
            src.mtp_slot.as_mut().expect("mtp").tq.as_mut().expect("tq"),
            11,
        );
        src.linear_attn[0].recurrent.as_mut_slice::<f32>().unwrap()[0] = 17.25;

        // Every overwrite-backed TQ byte was initialized by plant_tq_pattern,
        // so this internal full-allocation comparison is safe in the test.
        let full = src
            .snapshot_inner(&device, None, None)
            .expect("fully initialized test snapshot");
        let compact = src
            .snapshot_prefix(&device, n_tokens)
            .expect("prefix snapshot");
        let compact_tq = compact.full_attn_tq[0].as_ref().expect("compact tq");
        for buf in [
            &compact_tq.k_packed,
            &compact_tq.k_norms,
            &compact_tq.v_packed,
            &compact_tq.v_norms,
        ] {
            assert_eq!(buf.shape()[2], n_tokens, "snapshot sequence axis");
        }
        assert!(
            compact.total_bytes() < full.total_bytes(),
            "prefix snapshot must own fewer bytes (compact={} full={})",
            compact.total_bytes(),
            full.total_bytes()
        );

        let mut dst = HybridKvCache::new_with_options(&cfg, &device, max_seq_len, 1, true)
            .expect("alloc dst");
        dst.restore_partial(&compact, n_tokens)
            .expect("restore compact prefix");
        for (slot_index, slot) in dst.full_attn.iter().enumerate() {
            let dst_tq = slot.tq.as_ref().expect("dst tq");
            let src_tq = src.full_attn[slot_index].tq.as_ref().expect("src tq");
            for (dst_buf, src_buf) in [
                (&dst_tq.k_packed, &src_tq.k_packed),
                (&dst_tq.k_norms, &src_tq.k_norms),
                (&dst_tq.v_packed, &src_tq.v_packed),
                (&dst_tq.v_norms, &src_tq.v_norms),
            ] {
                for head in 0..dst_buf.shape()[1] {
                    assert_eq!(
                        read_head_prefix(dst_buf, head, n_tokens),
                        read_head_prefix(src_buf, head, n_tokens)
                    );
                }
            }
        }
        assert_eq!(
            dst.linear_attn[0].recurrent.as_slice::<f32>().unwrap()[0],
            17.25,
            "fixed-size DeltaNet state must survive compact snapshot restore"
        );

        assert!(src.snapshot_prefix(&device, 0).is_err());
        assert!(src
            .snapshot_prefix(&device, max_seq_len as usize + 1)
            .is_err());
    }

    #[test]
    fn snapshot_prefix_from_capture_uses_intermediate_deltanet_state_without_mutating_live() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_with_mtp();
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, 64, 1, true).expect("alloc cache");
        cache
            .ensure_la_capture(&cfg, &device, 3)
            .expect("capture buffers");

        for (layer_idx, slot) in cache.linear_attn.iter_mut().enumerate() {
            slot.pp_flipped[0] = true;
            slot.recurrent
                .as_mut_slice::<f32>()
                .expect("live recurrent")
                .fill(900.0 + layer_idx as f32);
            slot.conv_state
                .as_mut_slice::<f32>()
                .expect("live conv")
                .fill(800.0 + layer_idx as f32);
            slot.recurrent_scratch
                .as_mut_slice::<f32>()
                .expect("scratch recurrent")
                .fill(f32::NAN);
            slot.conv_state_scratch
                .as_mut_slice::<f32>()
                .expect("scratch conv")
                .fill(f32::NAN);

            let recurrent_per_token = slot.recurrent.element_count();
            let recurrent_capture = slot
                .capture_states
                .as_mut()
                .expect("recurrent capture")
                .as_mut_slice::<f32>()
                .expect("recurrent capture slice");
            recurrent_capture[recurrent_per_token..2 * recurrent_per_token]
                .fill(40.0 + layer_idx as f32);

            let conv_shape = slot.conv_state.shape().to_vec();
            let channels = conv_shape[0];
            let k_minus_one = conv_shape[1];
            let conv_per_token = channels * k_minus_one;
            let conv_capture = slot
                .conv_capture_states
                .as_mut()
                .expect("conv capture")
                .as_mut_slice::<f32>()
                .expect("conv capture slice");
            let captured = &mut conv_capture[conv_per_token..2 * conv_per_token];
            for k_idx in 0..k_minus_one {
                for channel in 0..channels {
                    captured[k_idx * channels + channel] =
                        (layer_idx * 10_000 + k_idx * 1_000 + channel) as f32;
                }
            }
        }

        let snapshot = cache
            .snapshot_prefix_from_capture(&device, 20, 1)
            .expect("captured prefix snapshot");
        for (layer_idx, slot) in cache.linear_attn.iter().enumerate() {
            assert!(snapshot.linear_recurrent[layer_idx]
                .as_slice::<f32>()
                .expect("snapshot recurrent")
                .iter()
                .all(|&v| v == 40.0 + layer_idx as f32));
            let conv_shape = slot.conv_state.shape();
            let channels = conv_shape[0];
            let k_minus_one = conv_shape[1];
            let captured_conv = snapshot.linear_conv[layer_idx]
                .as_slice::<f32>()
                .expect("snapshot conv");
            for channel in 0..channels {
                for k_idx in 0..k_minus_one {
                    assert_eq!(
                        captured_conv[channel * k_minus_one + k_idx],
                        (layer_idx * 10_000 + k_idx * 1_000 + channel) as f32
                    );
                }
            }
            assert!(slot
                .recurrent
                .as_slice::<f32>()
                .expect("live recurrent unchanged")
                .iter()
                .all(|&v| v == 900.0 + layer_idx as f32));
            assert!(slot
                .conv_state
                .as_slice::<f32>()
                .expect("live conv unchanged")
                .iter()
                .all(|&v| v == 800.0 + layer_idx as f32));
            assert!(slot.pp_flipped[0], "capture snapshot must not alter parity");
            assert!(slot
                .recurrent_scratch
                .as_slice::<f32>()
                .expect("scratch recurrent unchanged")
                .iter()
                .all(|v| v.is_nan()));
            assert!(slot
                .conv_state_scratch
                .as_slice::<f32>()
                .expect("scratch conv unchanged")
                .iter()
                .all(|v| v.is_nan()));
        }
        assert!(snapshot
            .full_attn_current_len
            .iter()
            .flatten()
            .all(|&len| len == 20));
        assert!(snapshot
            .mtp
            .as_ref()
            .expect("mtp snapshot")
            .current_len
            .iter()
            .all(|&len| len == 20));

        cache.clear_la_capture();
        assert!(!cache.la_capture_active());
        assert!(cache
            .linear_attn
            .iter()
            .all(|slot| slot.capture_states.is_some() && slot.conv_capture_states.is_some()));
    }

    /// ADR-027 sub-iter 23d-γ — the load-bearing regression pin for the
    /// silent-corruption gap: under production TQ-only mode,
    /// `restore_partial` MUST copy the first n_tokens positions of all
    /// four TQ buffers per slot (pre-23d-γ they were left zeroed while
    /// `current_len` advanced — the resumed request attended over zeroed
    /// K/V for the whole cached prefix).
    #[test]
    fn restore_partial_tq_only_mode_restores_tq_prefix_bytes() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_with_mtp();
        let max_seq_len = 64u32;
        let n_tokens = 40usize;

        // Source cache in TQ-only mode with planted deterministic bytes.
        let mut src = HybridKvCache::new_with_options(&cfg, &device, max_seq_len, 1, true)
            .expect("alloc src");
        assert!(src.tq_kv_active, "fixture must be TQ-active");
        assert!(src.full_attn[0].k.is_none(), "TQ-only: F32 dropped");
        assert!(src.full_attn[0].tq.is_some(), "TQ-only: tq populated");
        assert!(
            src.mtp_slot.as_ref().expect("mtp").k.is_none(),
            "TQ-only MTP: F32 dropped"
        );
        for slot in src.full_attn.iter_mut() {
            plant_tq_pattern(slot.tq.as_mut().expect("tq"), 1);
        }
        plant_tq_pattern(
            src.mtp_slot.as_mut().expect("mtp").tq.as_mut().expect("tq"),
            2,
        );
        // Linear-attn state must also survive the round-trip (unchanged path).
        src.linear_attn[0].recurrent.as_mut_slice::<f32>().unwrap()[0] = 9.5;
        set_all_sequence_lengths(&mut src, n_tokens as u32);

        let snap = src.snapshot(&device).expect("snapshot");
        assert!(snap.full_attn_k[0].is_none(), "TQ-only snapshot: k None");
        assert!(snap.full_attn_tq[0].is_some(), "TQ-only snapshot: tq Some");
        assert!(
            snap.mtp.as_ref().expect("mtp snap").k.is_none(),
            "TQ-only MTP snap: k None"
        );
        assert!(
            snap.mtp.as_ref().expect("mtp snap").tq.is_some(),
            "TQ-only MTP snap: tq Some"
        );

        // Seed the destination's cursor-invisible tails. Partial restore must
        // not read or overwrite anything beyond the requested prefix.
        const UNWRITTEN: u8 = 0xD3;
        let mut dst = HybridKvCache::new_with_options(&cfg, &device, max_seq_len, 1, true)
            .expect("alloc dst");
        for slot in &mut dst.full_attn {
            let tq = slot.tq.as_mut().expect("dst tq");
            for buf in [
                &mut tq.k_packed,
                &mut tq.k_norms,
                &mut tq.v_packed,
                &mut tq.v_norms,
            ] {
                buf.as_mut_slice::<u8>()
                    .expect("seed destination tail")
                    .fill(UNWRITTEN);
            }
        }
        {
            let tq = dst
                .mtp_slot
                .as_mut()
                .expect("dst mtp")
                .tq
                .as_mut()
                .expect("dst mtp tq");
            for buf in [
                &mut tq.k_packed,
                &mut tq.k_norms,
                &mut tq.v_packed,
                &mut tq.v_norms,
            ] {
                buf.as_mut_slice::<u8>()
                    .expect("seed destination MTP tail")
                    .fill(UNWRITTEN);
            }
        }
        dst.restore_partial(&snap, n_tokens)
            .expect("restore_partial");

        // Every full-attn slot: all four TQ buffers carry the prefix.
        for (i, slot) in dst.full_attn.iter().enumerate() {
            let tq_dst = slot.tq.as_ref().expect("dst tq");
            let tq_src = src.full_attn[i].tq.as_ref().expect("src tq");
            for (name, d, s) in [
                ("k_packed", &tq_dst.k_packed, &tq_src.k_packed),
                ("k_norms", &tq_dst.k_norms, &tq_src.k_norms),
                ("v_packed", &tq_dst.v_packed, &tq_src.v_packed),
                ("v_norms", &tq_dst.v_norms, &tq_src.v_norms),
            ] {
                let n_kv = d.shape()[1];
                for head in 0..n_kv {
                    assert_eq!(
                        read_head_prefix(d, head, n_tokens),
                        read_head_prefix(s, head, n_tokens),
                        "full_attn[{i}].tq.{name}[head {head}] prefix diverged after restore_partial"
                    );
                    // Tail beyond the boundary remains untouched.
                    let tail = read_head_prefix(d, head, max_seq_len as usize);
                    let inner = d.shape()[3] * d.dtype().size_of();
                    assert!(
                        tail[n_tokens * inner..].iter().all(|&b| b == UNWRITTEN),
                        "full_attn[{i}].tq.{name}[head {head}] tail overwritten by partial restore"
                    );
                }
            }
            assert_eq!(
                slot.current_len[0] as usize, n_tokens,
                "full_attn[{i}].current_len[0] must advance to the LCP boundary"
            );
        }

        // MTP slot: same pin.
        let dst_mtp = dst.mtp_slot.as_ref().expect("dst mtp");
        let src_mtp = src.mtp_slot.as_ref().expect("src mtp");
        let (dt, st) = (
            dst_mtp.tq.as_ref().expect("dst mtp tq"),
            src_mtp.tq.as_ref().expect("src mtp tq"),
        );
        for (name, d, s) in [
            ("k_packed", &dt.k_packed, &st.k_packed),
            ("k_norms", &dt.k_norms, &st.k_norms),
            ("v_packed", &dt.v_packed, &st.v_packed),
            ("v_norms", &dt.v_norms, &st.v_norms),
        ] {
            for head in 0..d.shape()[1] {
                assert_eq!(
                    read_head_prefix(d, head, n_tokens),
                    read_head_prefix(s, head, n_tokens),
                    "mtp.tq.{name}[head {head}] prefix diverged after restore_partial"
                );
                let tail = read_head_prefix(d, head, max_seq_len as usize);
                let inner = d.shape()[3] * d.dtype().size_of();
                assert!(
                    tail[n_tokens * inner..].iter().all(|&b| b == UNWRITTEN),
                    "mtp.tq.{name}[head {head}] tail overwritten by partial restore"
                );
            }
        }
        assert_eq!(dst_mtp.current_len[0] as usize, n_tokens);

        // Linear-attn state restored (byte-copy path, unchanged by 23d-γ).
        assert_eq!(
            dst.linear_attn[0].recurrent.as_slice::<f32>().unwrap()[0],
            9.5,
            "linear recurrent must survive restore_partial"
        );
    }

    /// Mirror pin for the legacy F32-only regime: TQ branches are
    /// no-ops when either side lacks TQ, and the F32 partial copy is
    /// untouched by the 23d-γ additions.
    #[test]
    fn restore_partial_f32_only_mode_tq_branches_are_noop() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_with_mtp();
        let max_seq_len = 64u32;
        let n_tokens = 40usize;

        let mut src = HybridKvCache::new(&cfg, &device, max_seq_len, 1).expect("alloc src");
        assert!(!src.tq_kv_active && src.full_attn[0].tq.is_none());
        // Plant F32 canary bytes in k[0].
        {
            let k = src.full_attn[0].k.as_mut().expect("f32 k");
            let s = k.as_mut_slice::<f32>().unwrap();
            for (i, v) in s.iter_mut().enumerate() {
                *v = (i % 97) as f32;
            }
        }
        set_all_sequence_lengths(&mut src, n_tokens as u32);
        let snap = src.snapshot(&device).expect("snapshot");
        let mut dst = HybridKvCache::new(&cfg, &device, max_seq_len, 1).expect("alloc dst");
        dst.restore_partial(&snap, n_tokens)
            .expect("restore_partial");

        let d = dst.full_attn[0]
            .k
            .as_ref()
            .expect("dst k")
            .as_slice::<f32>()
            .unwrap();
        let s = src.full_attn[0]
            .k
            .as_ref()
            .expect("src k")
            .as_slice::<f32>()
            .unwrap();
        let inner =
            d.len() / (src.full_attn[0].k.as_ref().unwrap().shape()[1] * max_seq_len as usize);
        for head in 0..2usize {
            let stride = max_seq_len as usize * inner;
            assert_eq!(
                &d[head * stride..head * stride + n_tokens * inner],
                &s[head * stride..head * stride + n_tokens * inner],
                "F32 k prefix diverged (23d-γ must not perturb the legacy path)"
            );
        }
    }

    #[test]
    fn slot_anchor_rewinds_only_target_cursor_and_linear_state() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 3).expect("alloc");
        let target = SlotId(1);
        let peer = SlotId(0);
        cache
            .append_for_seq(target, 9)
            .expect("target prompt cursor");
        cache.append_for_seq(peer, 7).expect("peer cursor");

        for (layer_idx, linear) in cache.linear_attn.iter_mut().enumerate() {
            let conv_per_slot = linear.conv_state.byte_len() / 3;
            let rec_per_slot = linear.recurrent.byte_len() / 3;

            // Target's current prompt state lives in scratch (flipped).
            linear.pp_flipped[target.0 as usize] = true;
            let target_conv = &mut linear.conv_state_scratch.as_mut_slice::<u8>().unwrap()
                [conv_per_slot..2 * conv_per_slot];
            target_conv.fill((31 + layer_idx) as u8);
            let target_rec = &mut linear.recurrent_scratch.as_mut_slice::<u8>().unwrap()
                [rec_per_slot..2 * rec_per_slot];
            target_rec.fill((71 + layer_idx) as u8);

            // Peer canaries cover both physical buffers and non-canonical
            // parity; a target restore must not touch any of them.
            linear.pp_flipped[peer.0 as usize] = true;
            linear.conv_state.as_mut_slice::<u8>().unwrap()[..conv_per_slot].fill(11);
            linear.conv_state_scratch.as_mut_slice::<u8>().unwrap()[..conv_per_slot].fill(12);
            linear.recurrent.as_mut_slice::<u8>().unwrap()[..rec_per_slot].fill(13);
            linear.recurrent_scratch.as_mut_slice::<u8>().unwrap()[..rec_per_slot].fill(14);
        }

        let anchor = cache
            .snapshot_slot_anchor(target, 9)
            .expect("slot-local anchor");
        assert_eq!(anchor.prompt_len(), 9);
        assert!(anchor.total_bytes() > 0);

        // Simulate decode mutating only the target slot after the prompt.
        cache
            .append_for_seq(target, 5)
            .expect("target decode cursor");
        for linear in &mut cache.linear_attn {
            let conv_per_slot = linear.conv_state.byte_len() / 3;
            let rec_per_slot = linear.recurrent.byte_len() / 3;
            linear.conv_state.as_mut_slice::<u8>().unwrap()[conv_per_slot..2 * conv_per_slot]
                .fill(201);
            linear.recurrent.as_mut_slice::<u8>().unwrap()[rec_per_slot..2 * rec_per_slot]
                .fill(202);
            linear.pp_flipped[target.0 as usize] = false;
        }

        cache
            .restore_slot_anchor(target, &anchor)
            .expect("slot-local restore");
        assert_eq!(cache.seq_len(target).unwrap(), 9);
        assert_eq!(cache.seq_len(peer).unwrap(), 7, "peer cursor changed");

        for (layer_idx, linear) in cache.linear_attn.iter().enumerate() {
            let conv_per_slot = linear.conv_state.byte_len() / 3;
            let rec_per_slot = linear.recurrent.byte_len() / 3;
            assert!(
                linear.conv_state.as_slice::<u8>().unwrap()[conv_per_slot..2 * conv_per_slot]
                    .iter()
                    .all(|&byte| byte == (31 + layer_idx) as u8)
            );
            assert!(
                linear.recurrent.as_slice::<u8>().unwrap()[rec_per_slot..2 * rec_per_slot]
                    .iter()
                    .all(|&byte| byte == (71 + layer_idx) as u8)
            );
            assert!(!linear.pp_flipped[target.0 as usize]);

            assert!(linear.conv_state.as_slice::<u8>().unwrap()[..conv_per_slot]
                .iter()
                .all(|&byte| byte == 11));
            assert!(
                linear.conv_state_scratch.as_slice::<u8>().unwrap()[..conv_per_slot]
                    .iter()
                    .all(|&byte| byte == 12)
            );
            assert!(linear.recurrent.as_slice::<u8>().unwrap()[..rec_per_slot]
                .iter()
                .all(|&byte| byte == 13));
            assert!(
                linear.recurrent_scratch.as_slice::<u8>().unwrap()[..rec_per_slot]
                    .iter()
                    .all(|&byte| byte == 14)
            );
            assert!(linear.pp_flipped[peer.0 as usize], "peer parity changed");
        }
    }

    #[test]
    fn slot_transaction_rollback_validation_is_fail_atomic() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_dense_cfg_4layer_for_multi_seq_tests();
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 1).expect("alloc");
        let slot = SlotId(0);
        cache.append_for_seq(slot, 5).expect("seed cursor");
        let mut transaction = cache
            .begin_slot_transaction(slot, 5)
            .expect("capture transaction");

        cache
            .append_for_seq(slot, 2)
            .expect("simulate target append");
        for linear in &mut cache.linear_attn {
            linear.pp_flipped[0] = true;
        }
        let cursors_before: Vec<u32> = cache
            .full_attn
            .iter()
            .map(|full| full.current_len[0])
            .collect();
        let parities_before: Vec<bool> = cache
            .linear_attn
            .iter()
            .map(|linear| linear.pp_flipped[0])
            .collect();

        // Make only the second layer invalid. A validate-as-you-mutate
        // rollback would rewind layer zero before discovering this error.
        transaction.full_attn_current_len[1] = cursors_before[1] + 1;
        let error = cache
            .rollback_slot_transaction(slot, &transaction)
            .expect_err("late-layer cursor mismatch must fail");
        assert!(error.to_string().contains("full_attn[1] live cursor"));
        assert_eq!(
            cache
                .full_attn
                .iter()
                .map(|full| full.current_len[0])
                .collect::<Vec<_>>(),
            cursors_before,
            "failed rollback partially rewound full-attention cursors"
        );
        assert_eq!(
            cache
                .linear_attn
                .iter()
                .map(|linear| linear.pp_flipped[0])
                .collect::<Vec<_>>(),
            parities_before,
            "failed rollback changed DeltaNet ping-pong selection"
        );
    }
}
