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

#[allow(unused_imports)]
pub use mlx_native::ops::gated_delta_net::cpu_reference_f32 as gated_delta_net_cpu_ref;

use super::{Qwen35Config, Qwen35LayerKind};

/// Per-full-attention-layer KV slot.
pub struct FullAttnKvSlot {
    /// Keys buffer `[head_dim, n_kv_heads, max_seq_len, n_seqs]` f32.
    ///
    /// **ADR-027 Phase B iter-29 (sub-sub-iter 23c-α):** wrapped in
    /// `Option` so iter-30 (sub-sub-iter 23c-β) can skip the F32 K/V
    /// allocation entirely when the cache is constructed with
    /// `tq_kv_active=true` — the actual 3.94× per-slot memory savings
    /// deliverable. Today (iter-29 structural prep) the alloc path
    /// always emits `Some(..)`; consumers handle Optional via
    /// `.as_ref().expect("F32 K/V required — iter-23c-β alloc branch
    /// dropped this; check tq_kv_active path")` at the per-call F32
    /// path (already gated by `slot.tq.is_none()` in iter-15's
    /// `dispatch_decode_sdpa_with_optional_tq`) and via `if let Some`
    /// at reset / snapshot / persist sites where None becomes a real
    /// possibility in iter-30.
    pub k: Option<MlxBuffer>,
    /// Values buffer — same shape and dtype as `k`.
    pub v: Option<MlxBuffer>,
    /// Per-seq write cursor. `current_len[s]` = number of tokens already
    /// stored for sequence s.
    pub current_len: Vec<u32>,
    /// ADR-027 Phase B iter-8 — TQ-active K/V buffers. `Some` when the
    /// containing `HybridKvCache` was constructed via
    /// `new_with_options(.., tq_kv_active = true)` (production path:
    /// `HF2Q_TQ_KV=1`); `None` in the legacy F32-only path (default,
    /// preserves all 71 existing `HybridKvCache::new(...)` callers).
    ///
    /// **Iter-8 scope (this commit):** allocator branching only. The
    /// SDPA dispatch + KV write branches that consume these buffers
    /// are iter-9 scope. In iter-8, when `tq.is_some()` the F32 `k` /
    /// `v` are STILL allocated alongside (shadow-cache pattern;
    /// mirrors Gemma's `dense_kvs` + `leg_hb_encoded` co-existence at
    /// `forward_mlx.rs:739+824`).  iter-11 (post-NRMSE-parity) drops
    /// the F32 backing in TQ mode for the full 3.94× memory savings.
    pub tq: Option<TqFullAttnKvBuffers>,
}

/// Per-linear-attention-layer SSM state + conv ring buffer.
pub struct LinearAttnStateSlot {
    /// DeltaNet conv1d ring buffer (active read buffer): `[conv_channels, K-1, n_seqs]` f32.
    ///
    /// Layout matches the ssm_conv kernel's expected `state[i, c, s]` at offset
    /// `s * (K-1) * channels + c * (K-1) + i`, i.e. channels-major with K-1 stride 1.
    /// Ping-pong semantics: `conv_state` is the active (read) buffer.
    /// `conv_state_scratch` is the inactive (write) buffer.  After each decode
    /// step the caller swaps via [`LinearAttnStateSlot::swap_conv_state`].
    pub conv_state: MlxBuffer,
    /// DeltaNet conv1d ring buffer (scratch, write target for ssm_conv kernel).
    /// Same shape as `conv_state`.  Swapped after each decode step.
    pub conv_state_scratch: MlxBuffer,
    /// DeltaNet recurrent state (current): `[D_k, D_v, num_v_heads, n_seqs]` f32.
    ///
    /// Ping-pong semantics: `recurrent` is the active (read) state buffer.
    /// `recurrent_scratch` is the inactive (write) buffer.  After each
    /// decode step the caller swaps the two handles via
    /// [`LinearAttnStateSlot::swap_recurrent`], turning last step's output
    /// into this step's input — zero copies, zero allocations.
    pub recurrent: MlxBuffer,
    /// DeltaNet recurrent state (scratch, write target for GDN kernel).
    /// Same shape as `recurrent`.  Swapped with `recurrent` each decode step.
    pub recurrent_scratch: MlxBuffer,
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
        .map_err(|e| {
            anyhow!("encode_token_to_tq: dispatch_hadamard_quantize_kv_hb K: {e}")
        })?;
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
        .map_err(|e| {
            anyhow!("encode_token_to_tq: dispatch_hadamard_quantize_kv_hb V: {e}")
        })?;
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
        let tq = self.tq.as_mut().ok_or_else(|| {
            anyhow!(
                "encode_seq_tokens_to_tq: slot.tq is None — slot was not \
                 constructed in TQ-active mode"
            )
        })?;
        let (packed, norms) = if is_k {
            (&tq.k_packed, &tq.k_norms)
        } else {
            (&tq.v_packed, &tq.v_norms)
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
            cache_capacity,
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
        let tq = self.tq.as_ref().ok_or_else(|| {
            anyhow!(
                "FullAttnKvSlot::dispatch_tq_sdpa: slot.tq is None — slot was \
                 not constructed in TQ-active mode (HybridKvCache::new_with_options \
                 tq_kv_active=true required)"
            )
        })?;
        let kernel_params = mlx_native::ops::flash_attn_vec_tq_hb::FlashAttnVecTqHbParams {
            num_heads: params.num_heads,
            num_kv_heads: params.num_kv_heads,
            head_dim: params.head_dim,
            kv_seq_len: params.kv_seq_len,
            kv_capacity: params.kv_capacity,
            scale: params.scale,
            mask_type: params.mask_type,
            sliding_window: params.sliding_window,
            softcap: params.softcap,
            ring_start: params.ring_start,
            scale_factor_d512: params.scale_factor_d512,
            codebook_bits: params.codebook_bits,
        };
        mlx_native::ops::flash_attn_vec_tq_hb::flash_attn_vec_tq_hb(
            encoder,
            registry,
            device,
            q,
            &tq.k_packed,
            &tq.k_norms,
            &tq.v_packed,
            &tq.v_norms,
            output,
            tmp,
            &kernel_params,
        )
        .map_err(|e| anyhow!("dispatch_tq_sdpa: flash_attn_vec_tq_hb: {e}"))?;
        Ok(())
    }
}

/// ADR-027 Phase B iter-10 — parameters for the qwen35 TQ SDPA dispatch.
/// Mirrors `mlx_native::ops::flash_attn_vec_tq_hb::FlashAttnVecTqHbParams`
/// but lives in the qwen35 namespace so the engine call site doesn't need
/// to import mlx-native types directly. Iter-11 wires this into
/// `gpu_full_attn::full_attn_layer_gpu`'s decode dispatch.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35TqSdpaParams {
    /// Q heads (e.g. 16 for qwen36 35B-A3B-APEX).
    pub num_heads: u32,
    /// K/V heads (e.g. 2 for qwen36).
    pub num_kv_heads: u32,
    /// head_dim (must be 256 or 512; production qwen35 = 256).
    pub head_dim: u32,
    /// Number of KV positions populated (cur_len at dispatch time).
    pub kv_seq_len: u32,
    /// Cache capacity (max_seq_len from `HybridKvCache` construction).
    pub kv_capacity: u32,
    /// Scale (typically `1 / sqrt(head_dim)`).
    pub scale: f32,
    /// Mask type: 0 = none, 1 = causal, 2 = sliding-window.
    pub mask_type: u32,
    /// Sliding window length (mask_type=2 only).
    pub sliding_window: u32,
    /// Softcap value (0 = disabled).
    pub softcap: f32,
    /// Ring buffer start slot for sliding-window cache (0 for global).
    pub ring_start: u32,
    /// D=512 per-block scale divisor (1.0 for d=256 = qwen35 production).
    pub scale_factor_d512: f32,
    /// Codebook bit-width (5, 6, or 8 — qwen35 default = 8).
    pub codebook_bits: u32,
}

impl LinearAttnStateSlot {
    /// Swap the active and scratch conv state buffers (O(1) pointer swap).
    /// Call this after every decode step to make the just-written scratch the
    /// new current conv state.
    #[inline]
    pub fn swap_conv_state(&mut self) {
        std::mem::swap(&mut self.conv_state, &mut self.conv_state_scratch);
    }

    /// Swap the active and scratch recurrent state buffers (O(1) pointer swap).
    /// Call this after every decode step to make the just-written scratch the
    /// new current state.
    #[inline]
    pub fn swap_recurrent(&mut self) {
        std::mem::swap(&mut self.recurrent, &mut self.recurrent_scratch);
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
/// K/V are Optional so iter-23c+ can drop the F32 backing in TQ mode
/// without producing zero-byte garbage. iter-23a-β scope: producers and
/// consumers always emit/expect `Some` (no behavior change today). The
/// codec at `qwen35_hybrid_persistor.rs` extracts via
/// `.as_ref().expect()` with explicit pinning for iter-23a-γ which
/// extends the codec with a `kv_present: u8` per-slot flag.
pub struct HybridKvCacheSnapshot {
    /// One per full-attn layer (e.g. 16 for Qwen3.6 27B): K matrix
    /// bytes. `None` in TQ-only mode (iter-23c+); `Some(buf)` on F32
    /// path. Producers in iter-23a-β always emit `Some`.
    pub full_attn_k: Vec<Option<MlxBuffer>>,
    /// One per full-attn layer: V matrix bytes. Same Optional
    /// semantics as `full_attn_k`.
    pub full_attn_v: Vec<Option<MlxBuffer>>,
    /// One per full-attn layer: per-seq write cursor at snapshot time.
    pub full_attn_current_len: Vec<Vec<u32>>,
    /// MTP slot snapshot (present only when the source cache had one).
    pub mtp: Option<MtpKvSnapshot>,
    /// One per linear-attn (DeltaNet) layer: active conv-state bytes.
    /// Scratch is intentionally NOT snapshotted — see [`HybridKvCache::snapshot`].
    pub linear_conv: Vec<MlxBuffer>,
    /// One per linear-attn layer: active recurrent state bytes.
    pub linear_recurrent: Vec<MlxBuffer>,
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
/// **ADR-027 Phase B sub-sub-iter 23a-α (Optional fields)**: K/V are
/// Optional so iter-23c+ can drop the F32 backing in TQ mode without
/// producing zero-byte garbage. iter-23a-α scope: producers and
/// consumers always emit/expect `Some` (no behavior change today).
/// `MtpKvSnapshot` itself stays `Option<MtpKvSnapshot>` at the
/// `HybridKvCacheSnapshot.mtp` level — that signals "MTP slot present
/// at all"; the inner `Option<MlxBuffer>` signals "F32 backing present
/// for the MTP slot's K/V".
pub struct MtpKvSnapshot {
    pub k: Option<MlxBuffer>,
    pub v: Option<MlxBuffer>,
    pub current_len: Vec<u32>,
}

impl HybridKvCacheSnapshot {
    /// Total bytes the snapshot owns across all KV / SSM slots.  Useful
    /// for memory accounting + tracing the per-prompt cache footprint.
    pub fn total_bytes(&self) -> usize {
        let mut n = 0usize;
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
        if let Some(s) = &self.mtp {
            // ADR-027 sub-sub-iter 23a-α: Optional MTP K/V — sum only Some.
            if let Some(buf) = &s.k {
                n += buf.byte_len();
            }
            if let Some(buf) = &s.v {
                n += buf.byte_len();
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
        .map_err(|e| anyhow!("deep_copy_buffer alloc: {e}"))?;
    let src_bytes: &[u8] = src
        .as_slice::<u8>()
        .map_err(|e| anyhow!("deep_copy_buffer src as_slice: {e}"))?;
    let dst_bytes: &mut [u8] = dst
        .as_mut_slice::<u8>()
        .map_err(|e| anyhow!("deep_copy_buffer dst as_mut_slice: {e}"))?;
    anyhow::ensure!(
        src_bytes.len() == dst_bytes.len(),
        "deep_copy_buffer: byte-length mismatch (src={} dst={})",
        src_bytes.len(),
        dst_bytes.len()
    );
    dst_bytes.copy_from_slice(src_bytes);
    Ok(dst)
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

    let src_bytes: &[u8] = src
        .as_slice::<u8>()
        .map_err(|e| anyhow!("partial_copy_slot ({name}) src as_slice: {e}"))?;
    let dst_bytes: &mut [u8] = dst
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
    let src_bytes: &[u8] = src
        .as_slice::<u8>()
        .map_err(|e| anyhow!("copy_buffer_bytes src as_slice: {e}"))?;
    let dst_bytes: &mut [u8] = dst
        .as_mut_slice::<u8>()
        .map_err(|e| anyhow!("copy_buffer_bytes dst as_mut_slice: {e}"))?;
    dst_bytes.copy_from_slice(src_bytes);
    Ok(())
}

/// DeltaNet 1D conv kernel width — Qwen3.5 uses 4; kept as a constant here so
/// the conv-state allocation math is explicit. If the config ever varies, the
/// value is the runtime authority (`cfg.linear_conv_kernel_dim`).
pub const DELTA_NET_CONV_K: u32 = 4;

impl HybridKvCache {
    /// Allocate the full hybrid cache for a Qwen3.5 (dense or MoE) model.
    ///
    /// Allocates:
    /// - For each full-attention layer in `cfg.layer_types`: two f32 buffers
    ///   of shape `[head_dim, n_kv_heads, max_seq_len, n_seqs]`.
    /// - For each linear-attention layer: conv-state of shape `[K-1, conv_channels, n_seqs]`
    ///   and recurrent state of shape `[D_k, D_v, num_v_heads, n_seqs]`.
    ///
    /// All buffers are explicitly zero-initialized at the end of `new()`
    /// via [`Self::reset`].
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
    /// In iter-8 the TQ buffers are allocated + zero-initialized only;
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
                    let mut slot = alloc_full_attn_slot(
                        cfg, device, max_seq_len, n_seqs,
                    )
                    .with_context(|| format!("alloc full-attn slot (layer {layer_idx})"))?;
                    if tq_kv_active {
                        slot.tq = Some(
                            alloc_tq_full_attn_buffers(
                                cfg, device, max_seq_len, n_seqs,
                            )
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
                    linear_attn.push(alloc_linear_attn_slot(
                        cfg,
                        device,
                        conv_channels,
                        k_minus1,
                        n_seqs,
                    )
                    .with_context(|| format!("alloc linear-attn slot (layer {layer_idx})"))?);
                }
            }
        }

        let mtp_slot = if cfg.mtp_num_hidden_layers > 0 {
            let mut slot = alloc_full_attn_slot(cfg, device, max_seq_len, n_seqs)
                .context("alloc MTP full-attn slot")?;
            if tq_kv_active {
                slot.tq = Some(
                    alloc_tq_full_attn_buffers(cfg, device, max_seq_len, n_seqs)
                        .context("alloc tq full-attn buffers (MTP slot)")?,
                );
            }
            Some(slot)
        } else {
            None
        };

        let mut cache = HybridKvCache {
            full_attn,
            mtp_slot,
            linear_attn,
            max_seq_len,
            n_seqs,
            conv_channels,
            per_layer_slot,
            tq_kv_active,
        };
        // ADR-015 iter61a (broken-window fix): explicitly zero every owned
        // GPU buffer to defend against StorageModeShared returning recycled,
        // non-zero memory at allocation time.  See the doc-comment on
        // `new()` above for the full rationale.  The cost is a one-time
        // memset over the cache footprint at construction; on the
        // 27B-dwq46 / 35B-apex configs this is dominated by full-attn
        // K/V (~10 GB at max_seq=262144) but `cmd_generate_qwen35`
        // sizes the cache to `prompt_len + max_tokens + 64` so in
        // practice this is sub-100ms and amortized across the entire
        // generation request.
        cache.reset_all_buffers();
        Ok(cache)
    }

    /// Internal helper used by `new()` to zero every owned GPU buffer at
    /// construction time.  Distinct from the public `reset()` which only
    /// zeros the linear-attention SSM state — full-attention K/V is
    /// covered here for correctness against any future kernel that reads
    /// past `current_len` (today's SDPA/flash-attn paths mask, but the
    /// cost of belt-and-braces zeroing at construction is one memset).
    fn reset_all_buffers(&mut self) {
        // Cursor reset (matches `reset()`).
        for slot in self.full_attn.iter_mut() {
            for c in slot.current_len.iter_mut() {
                *c = 0;
            }
            // Zero K/V buffers.  Float zero == bit zero, so writing
            // through `as_mut_slice::<f32>` is well-defined.
            // iter-29 (sub-sub-iter 23c-α): slot.k/v are Optional. None
            // is the iter-30 TQ-active state (no F32 backing); skip
            // cleanly. Today the alloc path always emits Some (no
            // observable change).
            if let Some(buf) = slot.k.as_mut() {
                if let Ok(s) = buf.as_mut_slice::<f32>() {
                    for v in s.iter_mut() {
                        *v = 0.0;
                    }
                }
            }
            if let Some(buf) = slot.v.as_mut() {
                if let Ok(s) = buf.as_mut_slice::<f32>() {
                    for v in s.iter_mut() {
                        *v = 0.0;
                    }
                }
            }
        }
        if let Some(slot) = self.mtp_slot.as_mut() {
            for c in slot.current_len.iter_mut() {
                *c = 0;
            }
            if let Some(buf) = slot.k.as_mut() {
                if let Ok(s) = buf.as_mut_slice::<f32>() {
                    for v in s.iter_mut() {
                        *v = 0.0;
                    }
                }
            }
            if let Some(buf) = slot.v.as_mut() {
                if let Ok(s) = buf.as_mut_slice::<f32>() {
                    for v in s.iter_mut() {
                        *v = 0.0;
                    }
                }
            }
        }
        for slot in self.linear_attn.iter_mut() {
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
        }
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
            // iter-29 (sub-sub-iter 23c-α): None means TQ-only mode
            // (iter-30 alloc branch); contributes 0 F32 bytes — exactly
            // the load-bearing memory savings the iter-30 regression-pin
            // test (full_attn_bytes_breakdown_tq_on_drops_f32_*) checks.
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
        }
    }

    /// Take a deep-copy snapshot of every owned KV / SSM buffer in this
    /// cache.
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
        let mut full_attn_k = Vec::with_capacity(self.full_attn.len());
        let mut full_attn_v = Vec::with_capacity(self.full_attn.len());
        let mut full_attn_current_len = Vec::with_capacity(self.full_attn.len());
        for slot in &self.full_attn {
            // ADR-027 sub-sub-iter 23c-α: slot.k/v are Optional. None
            // marks iter-30 TQ-only state (no F32 backing); snapshot
            // pushes None to mirror. Today the alloc path always emits
            // Some so this is always the Some branch (zero behavior
            // change).
            full_attn_k.push(match slot.k.as_ref() {
                Some(buf) => Some(
                    deep_copy_buffer(device, buf).context("snapshot full_attn.k")?,
                ),
                None => None,
            });
            full_attn_v.push(match slot.v.as_ref() {
                Some(buf) => Some(
                    deep_copy_buffer(device, buf).context("snapshot full_attn.v")?,
                ),
                None => None,
            });
            full_attn_current_len.push(slot.current_len.clone());
        }
        let mtp = match &self.mtp_slot {
            Some(slot) => Some(MtpKvSnapshot {
                // iter-23c-α: same Optional bridge as full_attn above.
                k: match slot.k.as_ref() {
                    Some(buf) => Some(
                        deep_copy_buffer(device, buf).context("snapshot mtp.k")?,
                    ),
                    None => None,
                },
                v: match slot.v.as_ref() {
                    Some(buf) => Some(
                        deep_copy_buffer(device, buf).context("snapshot mtp.v")?,
                    ),
                    None => None,
                },
                current_len: slot.current_len.clone(),
            }),
            None => None,
        };
        let mut linear_conv = Vec::with_capacity(self.linear_attn.len());
        let mut linear_recurrent = Vec::with_capacity(self.linear_attn.len());
        for slot in &self.linear_attn {
            linear_conv.push(
                deep_copy_buffer(device, &slot.conv_state).context("snapshot conv_state")?,
            );
            linear_recurrent.push(
                deep_copy_buffer(device, &slot.recurrent).context("snapshot recurrent")?,
            );
        }
        Ok(HybridKvCacheSnapshot {
            full_attn_k,
            full_attn_v,
            full_attn_current_len,
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
        for (slot, (k_snap, (v_snap, len_snap))) in self.full_attn.iter_mut().zip(
            snapshot
                .full_attn_k
                .iter()
                .zip(snapshot.full_attn_v.iter().zip(snapshot.full_attn_current_len.iter())),
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
                anyhow::ensure!(
                    snap.current_len.len() == slot.current_len.len(),
                    "restore_from: mtp current_len shape mismatch"
                );
                slot.current_len.copy_from_slice(&snap.current_len);
            }
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => {
                anyhow::bail!("restore_from: mtp_slot presence mismatch between snapshot and cache");
            }
        }
        for (slot, (conv_snap, rec_snap)) in self
            .linear_attn
            .iter_mut()
            .zip(snapshot.linear_conv.iter().zip(snapshot.linear_recurrent.iter()))
        {
            copy_buffer_bytes(conv_snap, &mut slot.conv_state).context("restore conv_state")?;
            copy_buffer_bytes(rec_snap, &mut slot.recurrent).context("restore recurrent")?;
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

        // Per-slot partial-position copy.  Each slot has shape
        // [n_kv_heads, max_seq_len, head_dim] with F32 elements.  Copy
        // first n_tokens positions per head.
        for (slot, (k_snap, v_snap)) in self.full_attn.iter_mut().zip(
            snapshot.full_attn_k.iter().zip(snapshot.full_attn_v.iter()),
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
        for (slot, (conv_snap, rec_snap)) in self
            .linear_attn
            .iter_mut()
            .zip(snapshot.linear_conv.iter().zip(snapshot.linear_recurrent.iter()))
        {
            copy_buffer_bytes(conv_snap, &mut slot.conv_state)
                .context("restore_partial conv_state")?;
            copy_buffer_bytes(rec_snap, &mut slot.recurrent)
                .context("restore_partial recurrent")?;
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
) -> Result<FullAttnKvSlot> {
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
    let k = device
        .alloc_buffer(bytes, DType::F32, shape.clone())
        .map_err(|e| anyhow!("alloc full-attn K: {e}"))?;
    let v = device
        .alloc_buffer(bytes, DType::F32, shape)
        .map_err(|e| anyhow!("alloc full-attn V: {e}"))?;

    Ok(FullAttnKvSlot {
        // iter-29 (sub-sub-iter 23c-α): wrap in Some today; iter-30
        // (sub-sub-iter 23c-β) adds a tq_kv_active branch here that
        // skips the alloc and emits None for the 3.94× memory win.
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
    let conv_elems =
        (conv_channels as usize) * (k_minus1 as usize) * (n_seqs as usize);
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
    })
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-027 Phase B iter-7 — TQ-active full-attn KV buffer infra (additive)
// ──────────────────────────────────────────────────────────────────────────
//
// Mirrors mlx-native `forward_mlx.rs::HbKvBuffers` (Gemma 4 TQ-active path)
// shape contract, extended with the qwen35 `n_seqs` axis. Iter-7 ships only
// the buffer types + allocator + tests so iter-8's SDPA dispatch (via
// `flash_attn_vec_tq_hb` from mlx-native) has a stable target.
//
// **Iter-7 scope (this file region):**
// - `TqFullAttnKvBuffers` struct (parallel to `FullAttnKvSlot` in TQ mode).
// - `tq_norms_per_pos_for(head_dim) -> u32` (1 for head_dim=256; 2 for
//   head_dim=512; mirror of `forward_mlx.rs:2326`).
// - `alloc_tq_full_attn_buffers(cfg, device, max_seq_len, n_seqs)` —
//   returns a fully-allocated TQ buffer set with U8 packed indices +
//   F32 norms zero-initialized.
// - Tests prove byte-count parity (~3.94× smaller than F32 at qwen36 APEX
//   shape) + correct shape per qwen35 cache layout.
//
// **NOT yet wired into `HybridKvCache::new`** — that's iter-8 along with
// the SDPA dispatch branch. Iter-7 keeps the existing F32 path completely
// untouched (Chesterton's fence on the live serve path).

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
/// Constructed by [`alloc_tq_full_attn_buffers`]. Iter-8 wires this into
/// the `HybridKvCache::new` allocator branch + the SDPA dispatch.
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
/// packed + F32 norms) zero-initialized.  Mirrors the production shape
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
        return Err(anyhow!(
            "alloc_tq_full_attn_buffers: n_seqs must be > 0"
        ));
    }

    let n_kv_heads = cfg.num_key_value_heads as usize;
    let head_dim = cfg.head_dim;
    let norms_per_pos = tq_norms_per_pos_for(head_dim);

    // Packed: [n_seqs, n_kv_heads, max_seq_len, head_dim] U8.
    // 1 byte per element (8-bit Lloyd-Max index).
    let packed_elems = (n_seqs as usize)
        * n_kv_heads
        * (max_seq_len as usize)
        * (head_dim as usize);
    let packed_bytes = packed_elems; // U8 → 1 byte/elem
    let packed_shape = vec![
        n_seqs as usize,
        n_kv_heads,
        max_seq_len as usize,
        head_dim as usize,
    ];

    // Norms: [n_seqs, n_kv_heads, max_seq_len, norms_per_pos] F32.
    // norms_per_pos=1 collapses to a 3-D view at the kernel level,
    // but we keep the 4-D shape on the buffer so cfg-shape validation
    // is unambiguous (every dim is explicit).
    let norms_elems = (n_seqs as usize)
        * n_kv_heads
        * (max_seq_len as usize)
        * (norms_per_pos as usize);
    let norms_bytes = norms_elems * std::mem::size_of::<f32>();
    let norms_shape = vec![
        n_seqs as usize,
        n_kv_heads,
        max_seq_len as usize,
        norms_per_pos as usize,
    ];

    let mut k_packed = device
        .alloc_buffer(packed_bytes, DType::U8, packed_shape.clone())
        .map_err(|e| anyhow!("alloc TQ full-attn K packed: {e}"))?;
    let mut k_norms = device
        .alloc_buffer(norms_bytes, DType::F32, norms_shape.clone())
        .map_err(|e| anyhow!("alloc TQ full-attn K norms: {e}"))?;
    let mut v_packed = device
        .alloc_buffer(packed_bytes, DType::U8, packed_shape)
        .map_err(|e| anyhow!("alloc TQ full-attn V packed: {e}"))?;
    let mut v_norms = device
        .alloc_buffer(norms_bytes, DType::F32, norms_shape)
        .map_err(|e| anyhow!("alloc TQ full-attn V norms: {e}"))?;

    // Zero-init all four buffers — mirrors `HybridKvCache::reset_all_buffers`
    // discipline (defends against StorageModeShared returning recycled
    // non-zero memory).  ADR-015 iter61a.
    if let Ok(s) = k_packed.as_mut_slice::<u8>() {
        s.fill(0);
    }
    if let Ok(s) = v_packed.as_mut_slice::<u8>() {
        s.fill(0);
    }
    if let Ok(s) = k_norms.as_mut_slice::<f32>() {
        s.fill(0.0);
    }
    if let Ok(s) = v_norms.as_mut_slice::<f32>() {
        s.fill(0.0);
    }

    Ok(TqFullAttnKvBuffers {
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        norms_per_pos,
    })
}

/// Total bytes the TQ-active full-attn slot occupies (sum of all 4
/// buffers).  Useful for memory accounting + the parity test.
impl TqFullAttnKvBuffers {
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
pub fn full_attn_slot_f32_bytes(
    cfg: &Qwen35Config,
    max_seq_len: u32,
    n_seqs: u32,
) -> usize {
    let elems = (n_seqs as usize)
        * (cfg.num_key_value_heads as usize)
        * (max_seq_len as usize)
        * (cfg.head_dim as usize);
    // K + V, 4 bytes each (F32).
    2 * elems * std::mem::size_of::<f32>()
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
    fn conv_channels_moe_8192() {
        let cfg = moe_cfg_40layer();
        assert_eq!(conv_channels_for(&cfg), 8192);
    }

    #[test]
    fn conv_channels_dense_10240() {
        let cfg = dense_cfg_64layer();
        assert_eq!(conv_channels_for(&cfg), 10240);
    }

    /// ADR-013 acceptance criterion: 40-layer MoE with full_attention_interval=4
    /// produces 10 full-attn slots + 30 linear-attn slots.
    #[test]
    fn moe_40layer_slot_counts() {
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
        let cfg = dense_cfg_64layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc cache");
        assert_eq!(cache.full_attn.len(), 16); // 64 / 4
        assert_eq!(cache.linear_attn.len(), 48);
    }

    #[test]
    fn layer_slot_lookup_matches_layer_types() {
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
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc");
        assert!(cache.slot_index_for_layer(40).is_none());
        assert!(cache.slot_index_for_layer(9999).is_none());
    }

    #[test]
    fn full_attn_slot_shape_and_dtype() {
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
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc");
        let s = &cache.linear_attn[0];
        // conv_state: [K-1=3, conv_channels=8192, n_seqs=1]
        assert_eq!(s.conv_state.element_count(), 3 * 8192 * 1);
        // recurrent: [D_k=128, D_v=128, num_v_heads=32, n_seqs=1]
        assert_eq!(s.recurrent.element_count(), 128 * 128 * 32 * 1);
    }

    /// ADR-015 iter61a: `HybridKvCache::new` MUST return cache buffers
    /// whose contents are bit-identical zero in every owned `MlxBuffer`.
    /// Relying on `MTLResourceOptions::StorageModeShared` first-touch
    /// page zeroing was empirically false (cold-process greedy decode
    /// produced different first tokens each cold run on Qwen3.6 27B
    /// dwq46 + apex q4_0-flat).  Regression bar: every f32 in every
    /// owned slot reads as `0.0_f32` (bit pattern `0x0000_0000`).
    #[test]
    fn new_returns_zero_initialized_buffers_iter61a() {
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 64, 2).expect("alloc");

        // Every full-attn K/V byte must be zero.
        // iter-29 (sub-sub-iter 23c-α): Optional-aware. legacy new()
        // always emits Some today; iter-30 TQ-only is the None branch
        // (no F32 backing → trivially zero F32 contribution).
        for (idx, slot) in cache.full_attn.iter().enumerate() {
            let sk = slot.k.as_ref().expect("legacy new()⇒Some(k)");
            let sv = slot.v.as_ref().expect("legacy new()⇒Some(v)");
            let k = sk.as_slice::<f32>().expect("k slice");
            assert!(
                k.iter().all(|v| v.to_bits() == 0),
                "full_attn[{}].k has non-zero bytes after new()",
                idx
            );
            let v = sv.as_slice::<f32>().expect("v slice");
            assert!(
                v.iter().all(|x| x.to_bits() == 0),
                "full_attn[{}].v has non-zero bytes after new()",
                idx
            );
            assert!(slot.current_len.iter().all(|&c| c == 0));
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
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        assert!(HybridKvCache::new(&cfg, &device, 16, 0).is_err());
        assert!(HybridKvCache::new(&cfg, &device, 0, 1).is_err());
    }

    #[test]
    fn total_bytes_matches_expected_footprint() {
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
            slot.current_len[0] = (i as u32) + 1;
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
        cache.restore_from(&snap).expect("restore");
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
                slot.current_len[0],
                expect_full_lens[i],
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

    /// Wedge-3 / iter-216 Phase B: snapshot does NOT alias the source
    /// — mutating the source post-snapshot leaves snapshot bytes intact.
    #[test]
    fn hybrid_kv_cache_snapshot_does_not_alias() {
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let mut cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc");

        // Plant a canary in slot 0.
        // iter-29 (sub-sub-iter 23c-α): legacy new()⇒Some K/V.
        cache.full_attn[0].k.as_mut().expect("legacy new()⇒Some(k)").as_mut_slice::<f32>().unwrap()[0] = 7.5;
        cache.linear_attn[0].recurrent.as_mut_slice::<f32>().unwrap()[0] = 3.25;

        let snap = cache.snapshot(&device).expect("snapshot");
        // Canary values inside the snapshot.
        let snap_full_k0 = snap.full_attn_k[0].as_ref().expect("snap.k[0] some").as_slice::<f32>().unwrap()[0];
        let snap_lin_rec0 = snap.linear_recurrent[0].as_slice::<f32>().unwrap()[0];
        assert_eq!(snap_full_k0, 7.5);
        assert_eq!(snap_lin_rec0, 3.25);

        // Mutate the live cache — snapshot must NOT see this.
        cache.full_attn[0].k.as_mut().expect("legacy new()⇒Some(k)").as_mut_slice::<f32>().unwrap()[0] = -123.0;
        cache.linear_attn[0].recurrent.as_mut_slice::<f32>().unwrap()[0] = -456.0;

        // Snapshot still holds the original canaries (deep-copy, not Arc::clone).
        assert_eq!(
            snap.full_attn_k[0].as_ref().expect("snap.k[0] some").as_slice::<f32>().unwrap()[0],
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
        let cfg = moe_cfg_40layer();
        let device = MlxDevice::new().expect("device");
        let cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("alloc");
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
                s.k.as_ref().expect("legacy new()⇒Some(k)").element_count() * 4
                    + s.v.as_ref().expect("legacy new()⇒Some(v)").element_count() * 4
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
        use mlx_native::ops::gated_delta_net::GatedDeltaNetParams;

        let p = GatedDeltaNetParams {
            d_k: 4, d_v: 4, n_k_heads: 1, n_v_heads: 1, n_tokens: 1, n_seqs: 1,
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
                        let idx = ((seq * n_kv_heads + head) * src_max_seq + pos)
                            * head_dim
                            + elem;
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
        let dst_after = dst_buf
            .as_slice::<f32>()
            .expect("dst as_slice")
            .to_vec();

        // Per (seq, head, pos, elem):
        //   pos < n_tokens : MUST equal src's value.
        //   pos >= n_tokens: MUST be 0.0 (zero-initialised tail).
        for seq in 0..n_seqs {
            for head in 0..n_kv_heads {
                for pos in 0..dst_max_seq {
                    for elem in 0..head_dim {
                        let dst_idx =
                            ((seq * n_kv_heads + head) * dst_max_seq + pos)
                                * head_dim
                                + elem;
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
        let buffers = alloc_tq_full_attn_buffers(
            &cfg, &device, max_seq_len, n_seqs,
        )
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
        let tq_buffers = alloc_tq_full_attn_buffers(
            &cfg, &device, max_seq_len, n_seqs,
        )
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
    fn tq_full_attn_buffers_alloc_initializes_to_zero() {
        // ADR-015 iter61a discipline: every owned GPU buffer must be
        // zero-initialized so the SDPA dispatch (iter-8) cannot read
        // recycled non-zero StorageModeShared bytes pre-write.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let buffers = alloc_tq_full_attn_buffers(&cfg, &device, 32, 1)
            .expect("alloc_tq_full_attn_buffers");
        // U8 packed buffers all-zero.
        for byte in buffers.k_packed.as_slice::<u8>().expect("k_packed slice") {
            assert_eq!(*byte, 0u8);
        }
        for byte in buffers.v_packed.as_slice::<u8>().expect("v_packed slice") {
            assert_eq!(*byte, 0u8);
        }
        // F32 norms all-zero.
        for f in buffers.k_norms.as_slice::<f32>().expect("k_norms slice") {
            assert_eq!(*f, 0.0_f32);
        }
        for f in buffers.v_norms.as_slice::<f32>().expect("v_norms slice") {
            assert_eq!(*f, 0.0_f32);
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-8 — HybridKvCache::new_with_options tests
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn hybrid_kv_cache_new_with_options_tq_off_keeps_tq_none_per_slot() {
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
        let cache = HybridKvCache::new_with_options(&cfg, &device, 64, 1, false)
            .expect("kv");
        assert!(!cache.full_attn.is_empty(), "test fixture has full-attn layers");
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
        let cache = HybridKvCache::new_with_options(&cfg, &device, 64, 1, true)
            .expect("kv tq-on");
        assert!(!cache.full_attn.is_empty());
        let n_full_attn = cache.full_attn.len();
        for (i, slot) in cache.full_attn.iter().enumerate() {
            assert!(
                slot.tq.is_some(),
                "full_attn[{i}].tq must be Some when tq_kv_active=true"
            );
            let tq = slot.tq.as_ref().unwrap();
            assert_eq!(tq.norms_per_pos, 1, "head_dim=256 → norms_per_pos=1");
            // K/V F32 buffers also remain allocated (shadow cache).
            // iter-29 (sub-sub-iter 23c-α): iter-30 will drop these to
            // None for the actual 3.94× memory savings; this assertion
            // pins the iter-29 shadow-cache invariant.
            assert!(slot.k.as_ref().expect("iter-29 shadow K still Some").byte_len() > 0);
            assert!(slot.v.as_ref().expect("iter-29 shadow V still Some").byte_len() > 0);
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
        // Empirical byte-count parity at qwen36 35B-A3B-APEX shape:
        // each full-attn slot now holds F32 K (16 MB) + F32 V (16 MB)
        // + TQ packed K+V (8.13 MB) + TQ norms K+V (128 KB) =
        // 40_517_632 bytes per slot (up from 33_554_432 F32-only).
        // iter-11 will drop the F32 K+V backing, restoring 33.55 MB →
        // 8.52 MB savings (3.94×).
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
        let cache = HybridKvCache::new_with_options(
            &cfg, &device, max_seq_len, n_seqs, true,
        )
        .expect("kv tq-on");
        let slot = &cache.full_attn[0];
        assert!(slot.tq.is_some());
        let tq = slot.tq.as_ref().unwrap();
        // F32 K and V each 16 MB at this shape (1×2×8192×256×4).
        // iter-29 (sub-sub-iter 23c-α): iter-30 drops these to None.
        let f32_each = 1 * 2 * 8192 * 256 * 4;
        assert_eq!(slot.k.as_ref().expect("iter-29 shadow K Some").byte_len(), f32_each);
        assert_eq!(slot.v.as_ref().expect("iter-29 shadow V Some").byte_len(), f32_each);
        // TQ K_packed + V_packed: 1×2×8192×256 each (U8) = 4 MB each.
        assert_eq!(tq.k_packed.byte_len(), 1 * 2 * 8192 * 256);
        assert_eq!(tq.v_packed.byte_len(), 1 * 2 * 8192 * 256);
        // TQ K_norms + V_norms: 1×2×8192×1×4 each = 64 KB.
        assert_eq!(tq.k_norms.byte_len(), 1 * 2 * 8192 * 1 * 4);
        assert_eq!(tq.v_norms.byte_len(), 1 * 2 * 8192 * 1 * 4);
        // Per-slot total in shadow-cache mode (iter-8): F32 + TQ.
        // F32 K+V = 33_554_432; TQ K+V (packed+norms) = 8_519_680;
        // shadow total = 42_074_112 bytes (iter-11 drops F32 → 8.5 MB only).
        let per_slot_total = 2 * f32_each + tq.total_bytes();
        assert_eq!(per_slot_total, 42_074_112);
        // Once iter-11 drops F32 in TQ mode, per_slot_total ==
        // tq.total_bytes() == 8_519_680. Documenting the target here
        // so iter-11 has a regression-pin.
        assert_eq!(tq.total_bytes(), 8_519_680);
    }

    #[test]
    fn hybrid_kv_cache_new_with_options_tq_on_with_mtp_populates_mtp_tq() {
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
        let cache = HybridKvCache::new_with_options(&cfg, &device, 64, 1, true)
            .expect("kv tq-on with mtp");
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
        let off = HybridKvCache::new_with_options(&cfg, &device, 64, 1, false)
            .expect("kv tq-off");
        assert!(!off.tq_kv_active, "tq_kv_active must propagate (false)");
        for (i, slot) in off.full_attn.iter().enumerate() {
            assert!(
                slot.tq.is_none(),
                "tq_kv_active=false implies full_attn[{i}].tq.is_none()"
            );
        }

        // tq_kv_active=true: field reads true; every slot.tq is Some.
        let on = HybridKvCache::new_with_options(&cfg, &device, 64, 1, true)
            .expect("kv tq-on");
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
        let mut cache = HybridKvCache::new_with_options(&cfg, &device, 64, 1, false)
            .expect("kv tq-off");
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
                &k_token, &v_token, n_kv_heads, head_dim, 64, 0, false, 1.0, 8,
                &mut encoder, &mut registry, &device,
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
        // Encode one token at write_pos=5 in a TQ-active slot. Verify:
        // - k_packed bytes at position 5 are non-zero (post-quant indices)
        // - k_packed bytes at OTHER positions (0..5, 6..) remain zero
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
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
                .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];
        assert!(slot.tq.is_some());
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
            // At write_pos: at least one byte must be non-zero (post-quant
            // index for the FWHT-rotated K).
            let pos_offset = base + (write_pos as usize) * head_dim_us;
            let pos_slice = &k_packed_bytes[pos_offset..pos_offset + head_dim_us];
            let nonzero_at_pos = pos_slice.iter().any(|&b| b != 0);
            assert!(
                nonzero_at_pos,
                "head={head} pos={write_pos}: expected non-zero packed bytes after encode"
            );
            // At other positions: bytes must remain zero (init-state).
            for other_pos in 0..cap_us {
                if other_pos as u32 == write_pos {
                    continue;
                }
                let other_offset = base + other_pos * head_dim_us;
                let other_slice = &k_packed_bytes[other_offset..other_offset + head_dim_us];
                assert!(
                    other_slice.iter().all(|&b| b == 0),
                    "head={head} pos={other_pos}: kernel must NOT write outside write_pos"
                );
            }
        }
    }

    #[test]
    fn encode_token_to_tq_writes_positive_norms() {
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
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
                .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];
        let n_kv_heads = cfg.num_key_value_heads as u32;
        let head_dim = cfg.head_dim;
        let k_token = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 11);
        let v_token = synth_token_buffer(&device, n_kv_heads as usize, head_dim as usize, 13);

        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        slot.encode_token_to_tq(
            &k_token, &v_token, n_kv_heads, head_dim, cache_capacity,
            3, false, 1.0, 8, &mut encoder, &mut registry, &device,
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
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
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
            &k_a, &v_a, n_kv_heads, head_dim, cache_capacity, 2, false, 1.0, 8,
            &mut encoder, &mut registry, &device,
        )
        .expect("encode A");
        slot.encode_token_to_tq(
            &k_b, &v_b, n_kv_heads, head_dim, cache_capacity, 7, false, 1.0, 8,
            &mut encoder, &mut registry, &device,
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
        let tmp_bytes = mlx_native::ops::flash_attn_vec_tq_hb::tmp_buffer_bytes(
            num_heads, head_dim,
        );
        let tmp = device
            .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
            .expect("alloc tmp");
        (q, output, tmp)
    }

    #[test]
    fn dispatch_tq_sdpa_errors_when_slot_lacks_tq_buffers() {
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
        let cache = HybridKvCache::new_with_options(&cfg, &device, 64, 1, false)
            .expect("kv tq-off");
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
            .dispatch_tq_sdpa(&q, &output, &tmp, &params, &mut encoder, &mut registry, &device)
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("slot.tq is None"),
            "expected fail-loud None-tq error, got: {msg}"
        );
    }

    #[test]
    fn dispatch_tq_sdpa_produces_finite_nonzero_output_at_qwen35_shape() {
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
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
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
            &k_token, &v_token, n_kv_heads, head_dim, cache_capacity,
            0, false, 1.0, 8, &mut encoder, &mut registry, &device,
        )
        .expect("encode");

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
            &q_buf, &output, &tmp, &params, &mut encoder, &mut registry, &device,
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
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
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
            &k0, &v0, n_kv_heads, head_dim, cache_capacity, 0, false, 1.0, 8,
            &mut encoder, &mut registry, &device,
        )
        .expect("encode pos 0");
        slot.encode_token_to_tq(
            &k1, &v1, n_kv_heads, head_dim, cache_capacity, 1, false, 1.0, 8,
            &mut encoder, &mut registry, &device,
        )
        .expect("encode pos 1");

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
            &q_buf, &output, &tmp, &params, &mut encoder, &mut registry, &device,
        )
        .expect("dispatch_tq_sdpa");
        encoder.commit_and_wait().expect("commit_and_wait");

        let out = output.as_slice::<f32>().expect("output slice");
        let mut any_nonzero = false;
        for &v in out.iter() {
            assert!(v.is_finite(), "SDPA output must be finite at kv_seq_len=2; got {v}");
            if v != 0.0 {
                any_nonzero = true;
            }
        }
        assert!(any_nonzero, "kv_seq_len=2 SDPA output must be non-zero");
    }

    #[test]
    fn dispatch_tq_sdpa_rejects_kv_seq_len_zero() {
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
        let cache = HybridKvCache::new_with_options(&cfg, &device, 64, 1, true)
            .expect("kv tq-on");
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
            .dispatch_tq_sdpa(&q, &output, &tmp, &params, &mut encoder, &mut registry, &device)
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
        0xa7, 0x3b, 0x91, 0xf4, 0x6d, 0xc2, 0x58, 0x0e,
        0xb3, 0x7f, 0x24, 0xd6, 0x89, 0x45, 0xea, 0x1c,
        0x63, 0xaf, 0xd8, 0x52, 0x97, 0x0b, 0xe1, 0x3d,
        0x76, 0xc4, 0x19, 0xfe, 0x4a, 0x85, 0x2c, 0xdb,
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
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
                .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];
        let n_kv_heads = cfg.num_key_value_heads;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        assert_eq!(head_dim, 256, "qwen35 production head_dim");

        // Step 1: synthetic K, V with both CPU mirrors (for reference) and
        // GPU buffers (for encoding).
        let (k_cpu, k_buf) = synth_token_with_cpu_mirror(
            &device, n_kv_heads as usize, head_dim as usize, 7,
        );
        let (v_cpu, v_buf) = synth_token_with_cpu_mirror(
            &device, n_kv_heads as usize, head_dim as usize, 11,
        );

        // Step 2: GPU encode at write_pos=0.
        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        slot.encode_token_to_tq(
            &k_buf, &v_buf, n_kv_heads, head_dim, cache_capacity,
            0, false, 1.0, 8, &mut encoder, &mut registry, &device,
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
        let mut q_fwht: Vec<f32> =
            Vec::with_capacity((num_heads as usize) * (head_dim as usize));
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
        let mut oracle_output =
            vec![0.0_f32; (num_heads as usize) * (head_dim as usize)];
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
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, 64, 1, false)
                .expect("kv tq-off");
        let slot = &mut cache.full_attn[0];
        assert!(slot.tq.is_none());
        let seq_kv = synth_seq_kv_buffer(
            &device, 4, cfg.num_key_value_heads as usize,
            cfg.head_dim as usize, 17,
        );
        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");
        let err = slot
            .encode_seq_tokens_to_tq(
                &seq_kv, true, 4,
                cfg.num_key_value_heads, cfg.head_dim, 64,
                0, 0, false, 1.0, 8,
                &mut encoder, &mut registry, &device,
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
        let mut cache_ref =
            HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
                .expect("kv ref tq-on");
        let slot_ref = &mut cache_ref.full_attn[0];

        // Build N single-token K and V buffers (each shape
        // [n_kv_heads, head_dim]).
        let mut single_k_bufs: Vec<MlxBuffer> = Vec::new();
        let mut single_v_bufs: Vec<MlxBuffer> = Vec::new();
        for t in 0..n_tokens as usize {
            single_k_bufs.push(synth_token_buffer(
                &device, n_kv_heads as usize, head_dim as usize,
                100 + t as u32,
            ));
            single_v_bufs.push(synth_token_buffer(
                &device, n_kv_heads as usize, head_dim as usize,
                200 + t as u32,
            ));
        }

        let mut registry = mlx_native::KernelRegistry::new();
        let mut enc_ref = device.command_encoder().expect("encoder ref");
        for (t, (k_buf, v_buf)) in
            single_k_bufs.iter().zip(single_v_bufs.iter()).enumerate()
        {
            slot_ref
                .encode_token_to_tq(
                    k_buf, v_buf, n_kv_heads, head_dim, cache_capacity,
                    t as u32, false, 1.0, 8, &mut enc_ref,
                    &mut registry, &device,
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
                (n_tokens as usize) * (n_kv_heads as usize)
                    * (head_dim as usize) * 4,
                DType::F32,
                vec![n_tokens as usize, n_kv_heads as usize, head_dim as usize],
            )
            .expect("alloc seq_k");
        let mut seq_v = device
            .alloc_buffer(
                (n_tokens as usize) * (n_kv_heads as usize)
                    * (head_dim as usize) * 4,
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

        let mut cache_seq =
            HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
                .expect("kv seq tq-on");
        let slot_seq = &mut cache_seq.full_attn[0];
        let mut enc_seq = device.command_encoder().expect("encoder seq");
        slot_seq
            .encode_seq_tokens_to_tq(
                &seq_k, true, n_tokens, n_kv_heads, head_dim, cache_capacity,
                0, 0, false, 1.0, 8, &mut enc_seq, &mut registry, &device,
            )
            .expect("encode K seq");
        slot_seq
            .encode_seq_tokens_to_tq(
                &seq_v, false, n_tokens, n_kv_heads, head_dim, cache_capacity,
                0, 0, false, 1.0, 8, &mut enc_seq, &mut registry, &device,
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

    #[test]
    fn encode_seq_tokens_to_tq_with_src_tok_offset_skips_leading_tokens() {
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
            &device, total_src_tokens as usize,
            n_kv_heads as usize, head_dim as usize, 333,
        );

        // Reference: encode tokens [2, 3] via per-token loop using
        // single-token buffers extracted from positions 2 and 3.
        let mut cache_ref =
            HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
                .expect("kv ref");
        let slot_ref = &mut cache_ref.full_attn[0];
        let mut registry = mlx_native::KernelRegistry::new();
        let mut enc_ref = device.command_encoder().expect("encoder ref");
        let stride = (n_kv_heads as usize) * (head_dim as usize);
        for (cache_slot, src_pos) in
            (src_tok_offset..src_tok_offset + n_tokens_to_encode).enumerate()
        {
            let mut tok_buf = device
                .alloc_buffer(
                    stride * 4, DType::F32,
                    vec![n_kv_heads as usize, head_dim as usize],
                )
                .expect("alloc tok");
            {
                let dst = tok_buf.as_mut_slice::<f32>().expect("tok mut");
                let src_slice =
                    seq_k.as_slice::<f32>().expect("seq_k slice");
                let src_offset = (src_pos as usize) * stride;
                dst.copy_from_slice(&src_slice[src_offset..src_offset + stride]);
            }
            // Use the same buffer for K + V (test only cares about K side).
            slot_ref
                .encode_token_to_tq(
                    &tok_buf, &tok_buf, n_kv_heads, head_dim, cache_capacity,
                    cache_slot as u32, false, 1.0, 8, &mut enc_ref,
                    &mut registry, &device,
                )
                .expect("encode token");
        }
        enc_ref.commit_and_wait().expect("ref commit");

        // Test path: encode_seq_tokens_to_tq with src_tok_offset=2.
        let mut cache_seq =
            HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
                .expect("kv seq");
        let slot_seq = &mut cache_seq.full_attn[0];
        let mut enc_seq = device.command_encoder().expect("encoder seq");
        slot_seq
            .encode_seq_tokens_to_tq(
                &seq_k, true, n_tokens_to_encode, n_kv_heads, head_dim,
                cache_capacity, 0, src_tok_offset, false, 1.0, 8,
                &mut enc_seq, &mut registry, &device,
            )
            .expect("encode seq K");
        enc_seq.commit_and_wait().expect("seq commit");

        // K side bytes must match.
        assert_eq!(
            slot_ref.tq.as_ref().unwrap().k_packed.as_slice::<u8>().unwrap(),
            slot_seq.tq.as_ref().unwrap().k_packed.as_slice::<u8>().unwrap(),
            "src_tok_offset semantics mismatch on k_packed"
        );
        assert_eq!(
            slot_ref.tq.as_ref().unwrap().k_norms.as_slice::<f32>().unwrap(),
            slot_seq.tq.as_ref().unwrap().k_norms.as_slice::<f32>().unwrap(),
            "src_tok_offset semantics mismatch on k_norms"
        );
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-18 — full-attn KV memory breakdown tests
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn full_attn_bytes_breakdown_tq_off_only_f32_at_qwen36_8k() {
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
        let cache = HybridKvCache::new_with_options(&cfg, &device, 8192, 1, false)
            .expect("kv tq-off");
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
    fn full_attn_bytes_breakdown_tq_on_shadow_at_qwen36_8k() {
        // Shadow-cache mode (iter-15 design): F32 + TQ both alloc.
        // Per slot: F32 K+V = 33_554_432 + TQ packed K+V = 8_388_608
        // + TQ norms K+V = 131_072 = 42_074_112 bytes.
        // 10 slots × 42_074_112 = 420_741_120 bytes total.
        // Projected iter-19 savings ratio per slot:
        //   f32_bytes / tq_bytes = 33_554_432 / 8_519_680 ≈ 3.94×
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer();
        let cache = HybridKvCache::new_with_options(&cfg, &device, 8192, 1, true)
            .expect("kv tq-on");
        let breakdown = cache.full_attn_bytes_breakdown();
        assert_eq!(breakdown.n_full_attn_slots, 10);
        assert!(!breakdown.has_mtp_slot);
        // F32 backing unchanged from tq_off.
        assert_eq!(breakdown.f32_k_v_bytes, 10 * 33_554_432);
        // TQ packed: 1×2×8192×256 (U8) = 4_194_304 per K, ×2 (K+V) ×10 slots.
        assert_eq!(breakdown.tq_packed_bytes, 10 * 2 * 4_194_304);
        // TQ norms: 1×2×8192×1 (F32) = 65_536 per K, ×2 (K+V) ×10 slots.
        assert_eq!(breakdown.tq_norms_bytes, 10 * 2 * 65_536);
        // Total = 335_544_320 (F32) + 83_886_080 (TQ packed) + 1_310_720 (TQ norms).
        assert_eq!(breakdown.total_bytes(), 420_741_120);
        // Projected iter-19 savings — drop F32 leaves only TQ.
        let ratio = breakdown.projected_iter19_savings_ratio().unwrap();
        assert!(
            (3.5..=4.5).contains(&ratio),
            "projected_iter19_savings_ratio {ratio:.4} outside expected [3.5, 4.5] window"
        );
    }

    #[test]
    fn full_attn_bytes_breakdown_tq_on_shadow_at_qwen36_32k() {
        // 32K context — verifies the §1 ADR claim at production-realistic
        // scale (max_seq_len=32768). Per slot: F32 K+V = 4× 8K =
        // 134_217_728; TQ packed K+V = 4× 8K = 33_554_432; TQ norms K+V
        // = 4× 8K = 524_288. Per-slot total in shadow mode = 168_296_448.
        // 10 slots × 168_296_448 = 1_682_964_480 bytes ≈ 1.57 GiB.
        // Iter-19 target: 10 × 34_078_720 (TQ only) = 340_787_200 bytes ≈ 325 MiB.
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
        assert_eq!(breakdown.f32_k_v_bytes, 10 * 134_217_728);
        assert_eq!(breakdown.tq_packed_bytes, 10 * 33_554_432);
        assert_eq!(breakdown.tq_norms_bytes, 10 * 524_288);
        assert_eq!(breakdown.total_bytes(), 1_682_964_480);
        // The §1 ADR table claim: post-iter-19 total = ~340 MiB at 32K
        // (down from F32-only 1.34 GB). Verify the projection holds.
        let projected_iter19_total =
            breakdown.tq_total_bytes();
        assert_eq!(projected_iter19_total, 340_787_200); // ~325 MiB
        // §1 says "1.34 GB total" for F32 dense at 32K → matches our
        // 1_342_177_280 = 10 × 134_217_728 byte count exactly.
        assert_eq!(breakdown.f32_k_v_bytes, 1_342_177_280);
        // Savings ratio at 32K should match 8K (shape-invariant).
        let ratio = breakdown.projected_iter19_savings_ratio().unwrap();
        assert!(
            (3.5..=4.5).contains(&ratio),
            "32K savings ratio {ratio:.4} outside [3.5, 4.5]"
        );
    }

    #[test]
    fn full_attn_bytes_breakdown_with_mtp_includes_mtp_slot() {
        // MTP slot bytes count toward both F32 and TQ totals.
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
        // 10 regular full-attn + 1 MTP = 11 slots' worth.
        let per_slot_f32 = 1 * 2 * 1024 * 256 * 4 * 2; // K + V
        let per_slot_tq_packed = 1 * 2 * 1024 * 256 * 2;
        let per_slot_tq_norms = 1 * 2 * 1024 * 1 * 4 * 2;
        assert_eq!(breakdown.n_full_attn_slots, 10);
        assert_eq!(breakdown.f32_k_v_bytes, 11 * per_slot_f32);
        assert_eq!(breakdown.tq_packed_bytes, 11 * per_slot_tq_packed);
        assert_eq!(breakdown.tq_norms_bytes, 11 * per_slot_tq_norms);
    }

    #[test]
    fn full_attn_bytes_breakdown_tq_off_returns_no_savings_ratio() {
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
        let mut cache =
            HybridKvCache::new_with_options(&cfg, &device, cache_capacity, 1, true)
                .expect("kv tq-on");
        let slot = &mut cache.full_attn[0];
        let n_kv_heads = cfg.num_key_value_heads;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        assert_eq!(head_dim, 256);

        // Synthesize K, V tokens with both CPU mirrors + GPU buffers.
        let (_k_cpu, k_buf) = synth_token_with_cpu_mirror(
            &device, n_kv_heads as usize, head_dim as usize, 7,
        );
        let (v_cpu, v_buf) = synth_token_with_cpu_mirror(
            &device, n_kv_heads as usize, head_dim as usize, 11,
        );

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
        let tmp_bytes = mlx_native::ops::flash_attn_vec_tq_hb::tmp_buffer_bytes(
            num_heads, head_dim,
        );
        let tmp = device
            .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
            .expect("alloc tmp");

        let mut registry = mlx_native::KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        // (a) GPU encode K, V at write_pos=0.
        slot.encode_token_to_tq(
            &k_buf, &v_buf, n_kv_heads, head_dim, cache_capacity,
            0, false, 1.0, 8, &mut encoder, &mut registry, &device,
        )
        .expect("encode_token_to_tq");
        encoder.memory_barrier();

        // (b) GPU Q pre-rotation: sign × FWHT (in-place on q_gpu).
        mlx_native::ops::fwht_standalone::dispatch_fwht_sign_premult_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &q_gpu, num_heads, head_dim,
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
            &q_gpu, &output, &tmp, &params, &mut encoder, &mut registry, &device,
        )
        .expect("dispatch_tq_sdpa");
        encoder.memory_barrier();

        // (d) GPU output inverse-rotation: FWHT × sign-undo (in-place on output).
        mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &output, num_heads, head_dim,
        )
        .expect("fwht sign-undo output");

        encoder.commit_and_wait().expect("commit chain");

        // Read GPU output to CPU + compare to F32 closed-form reference.
        let output_gpu_flat: Vec<f32> =
            output.as_slice::<f32>().expect("output slice").to_vec();
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
        let buffers = alloc_tq_full_attn_buffers(&cfg, &device, 64, 2)
            .expect("alloc_tq_full_attn_buffers");
        // Expected: k_packed = [n_seqs=2, n_kv_heads=2, max_seq_len=64,
        // head_dim=256] = 2*2*64*256 = 65_536 bytes (U8).
        assert_eq!(buffers.k_packed.byte_len(), 65_536);
        assert_eq!(buffers.k_packed.shape(), &[2, 2, 64, 256]);
        // k_norms = [n_seqs=2, n_kv_heads=2, max_seq_len=64,
        // norms_per_pos=1] = 2*2*64*1 elems × 4 bytes = 1024 bytes.
        assert_eq!(buffers.k_norms.byte_len(), 1024);
        assert_eq!(buffers.k_norms.shape(), &[2, 2, 64, 1]);
    }
}
