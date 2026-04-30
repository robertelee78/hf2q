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
    pub k: MlxBuffer,
    /// Values buffer — same shape and dtype as `k`.
    pub v: MlxBuffer,
    /// Per-seq write cursor. `current_len[s]` = number of tokens already
    /// stored for sequence s.
    pub current_len: Vec<u32>,
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
}

/// Resolved slot index for a given model layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSlot {
    Full(u32),   // index into `full_attn`
    Linear(u32), // index into `linear_attn`
}

/// Deep-copy snapshot of a [`HybridKvCache`] — owns fresh `MlxBuffer`
/// allocations holding byte-equal contents at snapshot time.
///
/// Wedge-3 / ADR-005 iter-216 Phase B.  Used by `HybridPromptCache`
/// (engine_qwen35.rs Phase C) to save post-prefill cache state for replay
/// on the next equivalent prompt.  See [`HybridKvCache::snapshot`] for
/// the deep-copy contract and the DeltaNet ping-pong note.
pub struct HybridKvCacheSnapshot {
    /// One per full-attn layer (e.g. 16 for Qwen3.6 27B): K matrix bytes.
    pub full_attn_k: Vec<MlxBuffer>,
    /// One per full-attn layer: V matrix bytes.
    pub full_attn_v: Vec<MlxBuffer>,
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

/// MTP slot snapshot — same shape as a `FullAttnKvSlot` snapshot but kept
/// as a dedicated struct so `Option<MtpKvSnapshot>` is explicit rather
/// than overloading `full_attn_k`/`full_attn_v` with a sentinel.
pub struct MtpKvSnapshot {
    pub k: MlxBuffer,
    pub v: MlxBuffer,
    pub current_len: Vec<u32>,
}

impl HybridKvCacheSnapshot {
    /// Total bytes the snapshot owns across all KV / SSM slots.  Useful
    /// for memory accounting + tracing the per-prompt cache footprint.
    pub fn total_bytes(&self) -> usize {
        let mut n = 0usize;
        for k in &self.full_attn_k {
            n += k.byte_len();
        }
        for v in &self.full_attn_v {
            n += v.byte_len();
        }
        if let Some(s) = &self.mtp {
            n += s.k.byte_len() + s.v.byte_len();
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
    /// All buffers are initialized to zero by MlxDevice::alloc_buffer (which
    /// uses StorageModeShared → the OS zeroes pages on first access; no
    /// explicit memset needed).
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
                    full_attn.push(alloc_full_attn_slot(
                        cfg, device, max_seq_len, n_seqs,
                    )
                    .with_context(|| format!("alloc full-attn slot (layer {layer_idx})"))?);
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
            Some(
                alloc_full_attn_slot(cfg, device, max_seq_len, n_seqs)
                    .context("alloc MTP full-attn slot")?,
            )
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
        })
    }

    /// Translate a model layer index (0..num_hidden_layers) to the matching
    /// slot in this cache.
    pub fn slot_index_for_layer(&self, layer_idx: u32) -> Option<LayerSlot> {
        self.per_layer_slot.get(layer_idx as usize).copied()
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
            full_attn_k.push(deep_copy_buffer(device, &slot.k).context("snapshot full_attn.k")?);
            full_attn_v.push(deep_copy_buffer(device, &slot.v).context("snapshot full_attn.v")?);
            full_attn_current_len.push(slot.current_len.clone());
        }
        let mtp = match &self.mtp_slot {
            Some(slot) => Some(MtpKvSnapshot {
                k: deep_copy_buffer(device, &slot.k).context("snapshot mtp.k")?,
                v: deep_copy_buffer(device, &slot.v).context("snapshot mtp.v")?,
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
            copy_buffer_bytes(k_snap, &mut slot.k).context("restore full_attn.k")?;
            copy_buffer_bytes(v_snap, &mut slot.v).context("restore full_attn.v")?;
            anyhow::ensure!(
                len_snap.len() == slot.current_len.len(),
                "restore_from: full_attn current_len shape mismatch"
            );
            slot.current_len.copy_from_slice(len_snap);
        }
        match (&snapshot.mtp, self.mtp_slot.as_mut()) {
            (Some(snap), Some(slot)) => {
                copy_buffer_bytes(&snap.k, &mut slot.k).context("restore mtp.k")?;
                copy_buffer_bytes(&snap.v, &mut slot.v).context("restore mtp.v")?;
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

    /// Total allocated bytes across all slots (for memory accounting / logs).
    pub fn total_bytes(&self) -> usize {
        let mut n = 0usize;
        for s in &self.full_attn {
            n += s.k.element_count() * 4 + s.v.element_count() * 4;
        }
        if let Some(s) = &self.mtp_slot {
            n += s.k.element_count() * 4 + s.v.element_count() * 4;
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
        k,
        v,
        current_len: vec![0; n_seqs as usize],
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
        assert_eq!(s.k.dtype(), DType::F32);
        assert_eq!(s.v.dtype(), DType::F32);
        // Expected element count: n_seqs * n_kv * max_seq_len * head_dim
        // = 2 * 2 * 64 * 256 = 65536.  Layout is SDPA-native [n_seqs, n_kv, max_seq, head_dim].
        assert_eq!(s.k.element_count(), 2 * 2 * 64 * 256);
        assert_eq!(s.v.element_count(), 2 * 2 * 64 * 256);
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
        for (i, slot) in cache.full_attn.iter_mut().enumerate() {
            let s = slot.k.as_mut_slice::<f32>().expect("k mut");
            s[0] = (i as f32) + 0.25;
            s[7] = (i as f32) + 0.5;
            let s = slot.v.as_mut_slice::<f32>().expect("v mut");
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
            expect_full_k0.push(slot.k.as_slice::<f32>().unwrap()[0]);
            expect_full_v0.push(slot.v.as_slice::<f32>().unwrap()[0]);
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
            for v in slot.k.as_mut_slice::<f32>().unwrap().iter_mut() {
                *v = 999.0;
            }
            for v in slot.v.as_mut_slice::<f32>().unwrap().iter_mut() {
                *v = -999.0;
            }
            slot.current_len[0] = 42;
        }

        // Restore — byte-equality across all canary positions.
        cache.restore_from(&snap).expect("restore");
        for (i, slot) in cache.full_attn.iter().enumerate() {
            assert_eq!(
                slot.k.as_slice::<f32>().unwrap()[0],
                expect_full_k0[i],
                "full_attn[{i}].k[0] not restored"
            );
            assert_eq!(
                slot.v.as_slice::<f32>().unwrap()[0],
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
        cache.full_attn[0].k.as_mut_slice::<f32>().unwrap()[0] = 7.5;
        cache.linear_attn[0].recurrent.as_mut_slice::<f32>().unwrap()[0] = 3.25;

        let snap = cache.snapshot(&device).expect("snapshot");
        // Canary values inside the snapshot.
        let snap_full_k0 = snap.full_attn_k[0].as_slice::<f32>().unwrap()[0];
        let snap_lin_rec0 = snap.linear_recurrent[0].as_slice::<f32>().unwrap()[0];
        assert_eq!(snap_full_k0, 7.5);
        assert_eq!(snap_lin_rec0, 3.25);

        // Mutate the live cache — snapshot must NOT see this.
        cache.full_attn[0].k.as_mut_slice::<f32>().unwrap()[0] = -123.0;
        cache.linear_attn[0].recurrent.as_mut_slice::<f32>().unwrap()[0] = -456.0;

        // Snapshot still holds the original canaries (deep-copy, not Arc::clone).
        assert_eq!(
            snap.full_attn_k[0].as_slice::<f32>().unwrap()[0],
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
        let cache_active_only: usize = cache
            .full_attn
            .iter()
            .map(|s| s.k.element_count() * 4 + s.v.element_count() * 4)
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
}
