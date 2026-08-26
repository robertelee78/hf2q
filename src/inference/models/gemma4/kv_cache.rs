//! Gemma 4 KV-cache buffer types and decode-regime control.
//!
//! Extracted from `src/serve/forward_mlx.rs` by ADR-038 Step 2.
//! Mirrors the `qwen35/kv_cache.rs` pattern.

use anyhow::{anyhow, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

/// Per-layer KV cache buffers for the mlx-native path (TurboQuant compressed).
///
/// ADR-007 Phase 1.2: KV cache is stored as 4-bit nibble-packed indices
/// with per-position F32 norms, replacing F16 dense buffers.  This halves
/// KV memory bandwidth during SDPA and enables 262K context.
pub struct MlxKvCache {
    /// K packed indices `[num_kv_heads, capacity, head_dim/2]` U8 (nibble-packed).
    pub k_packed: MlxBuffer,
    /// K per-position norms `[num_kv_heads, capacity]` F32.
    pub k_norms: MlxBuffer,
    /// V packed indices `[num_kv_heads, capacity, head_dim/2]` U8 (nibble-packed).
    pub v_packed: MlxBuffer,
    /// V per-position norms `[num_kv_heads, capacity]` F32.
    pub v_norms: MlxBuffer,
    /// Cache capacity (max_seq_len for global, sliding_window for sliding).
    pub capacity: usize,
    /// Whether this is a sliding window cache.
    pub is_sliding: bool,
    /// Current write position (next position to write).
    pub write_pos: usize,
    /// Number of valid positions in the cache.
    pub seq_len: usize,
}

impl MlxKvCache {
    /// ADR-028 iter-229 / ADR-028 §iter-227 work item D: ds4-style counter
    /// rollback for speculative decode infrastructure.
    ///
    /// Logically discards the most-recent `n_back` positions from the cache.
    /// Following ds4's pattern (`DS4_MTP_KEEP_ACCEPTED` macro, ds4.c:16246),
    /// no actual cache bytes are cleared — `seq_len` is decremented to make
    /// the trailing positions invisible to subsequent SDPA reads.  Future
    /// writes (via `write_pos`) will overwrite those slots naturally.
    ///
    /// Returns the new `seq_len` for caller assertion.
    ///
    /// # Semantics
    /// - **Linear cache** (`is_sliding=false`): `write_pos == seq_len`, both
    ///   decrement by `n_back`.  Subsequent writes resume at the new
    ///   `write_pos`.
    /// - **Sliding cache** (`is_sliding=true`): more complex — `write_pos`
    ///   wraps modulo `capacity`.  Implemented for the linear case here;
    ///   sliding-aware rollback requires position-tracking metadata and is
    ///   deferred until iter-227 work item C (SD state machine).
    ///
    /// # Errors
    /// Returns `Err` if `n_back > seq_len` or sliding cache (not yet supported).
    pub fn trim(&mut self, n_back: usize) -> Result<usize, &'static str> {
        if self.is_sliding {
            // Sliding cache rollback requires logical-position tracking
            // (the slot index ≠ logical position when wrapped).  For ds4-style
            // SD on gemma4, sliding layers may need a counter parallel to
            // `write_pos` that tracks logical end-position separately.
            // Deferred to iter-227 work item C.
            return Err("trim() not yet supported on sliding cache");
        }
        if n_back > self.seq_len {
            return Err("trim n_back exceeds seq_len");
        }
        // Linear cache: write_pos == seq_len. Both decrement.
        self.seq_len -= n_back;
        self.write_pos = self.seq_len;
        Ok(self.seq_len)
    }

    /// Returns the count of valid (visible) positions.  Equivalent to
    /// `seq_len` post-iter-229 but exposed as named API for the SD
    /// state machine (matches ds4's `s->graph.mtp_n_raw` semantic).
    #[inline]
    pub fn visible_len(&self) -> usize {
        self.seq_len
    }
}

/// Per-layer byte-packed higher-bit (5/6-bit) KV buffers (iter-21 Track B).
///
/// **Single-seq legacy shape** — every buffer is 3-D `[nkv, cap, hd]` (K/V
/// packed) or `[nkv, cap, norms_per_pos]` (norms).  The 3 production alloc
/// sites (`src/serve/forward_prefill.rs`, `src/serve/forward_prefill_batched.rs`,
/// `src/inference/models/gemma4/forward_gpu.rs`) construct this type via
/// inline struct literals at n_seqs=1 implicit; this struct stays unchanged
/// to keep those sites byte-for-byte compatible while the multi-seq lift
/// rolls in via the sibling [`MultiSeqHbKvBuffers`] (ADR-040 Phase A3a).
pub struct HbKvBuffers {
    /// Byte-packed K indices `[nkv_heads, capacity, head_dim]` U8.
    pub k_packed: MlxBuffer,
    /// K per-position norms (same layout as 4-bit: D=256 → 1/pos, D=512 → 2/pos).
    pub k_norms: MlxBuffer,
    /// Byte-packed V indices `[nkv_heads, capacity, head_dim]` U8.
    pub v_packed: MlxBuffer,
    /// V per-position norms.
    pub v_norms: MlxBuffer,
    /// Cache capacity in positions.
    pub capacity: usize,
    /// True if ring-buffer (sliding) semantics.
    pub is_sliding: bool,
    /// Norms per position (1 for D=256, 2 for D=512).
    #[allow(dead_code)]
    pub norms_per_pos: usize,
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A3a iter-3 — multi-seq variant of HbKvBuffers.
//
// Per dossier `docs/research/adr040-kv-cache-lift-dossier-2026-05-23.md`
// §2.2.4 + §4 iter-3, the Gemma 4 production-default KV variant
// (HbKvBuffers, TQ-active 8-bit per ADR-007) needs an n_seqs axis to
// support ADR-040's multi-seq scheduling.  This sibling struct ships
// the lift WITHOUT touching the 3 existing alloc sites (those keep
// allocating the legacy 3-D `HbKvBuffers` at implicit n_seqs=1 until
// Phase B4c re-routes them through `alloc_hb_kv_for_layer`).
//
// **Why a sibling struct, not a struct extension** (deviation from
// the dossier §4 iter-3 line 511 sketch):
//   * The brief explicitly forbids touching `forward_prefill.rs`,
//     `forward_prefill_batched.rs`, and `forward_gpu.rs` (constraint
//     #8).  Adding required fields to `HbKvBuffers` would break the
//     3 inline struct-literal alloc sites (Rust requires every public
//     field to be initialised in a `Struct { .. }` literal).  The
//     wrapper approach honours every brief constraint AND mirrors
//     Qwen35's pattern where `HybridKvCache` is the multi-seq
//     aggregate distinct from the per-layer buffer primitive.
//   * Sequencing: Phase B4c will refactor the 3 alloc sites to
//     emit `MultiSeqHbKvBuffers` via `alloc_hb_kv_for_layer`.  At
//     that point `HbKvBuffers` becomes pure legacy and the sites
//     simply produce the 4-D buffer through the unified helper.
//
// Scope (per dossier §4 iter-3):
//   * HbKvBuffers analogue with n_seqs as the OUTERMOST axis on
//     k_packed/v_packed (`[n_seqs, nkv, cap, hd]`) and k_norms/
//     v_norms (`[n_seqs, nkv, cap, norms_per_pos]`).
//   * Per-seq cursor `seq_lens: Vec<u32>` of length n_seqs.
//   * MultiSeqKvCache impl mirroring the Qwen35 A2a pattern at
//     `src/inference/models/qwen35/kv_cache.rs:2579-2780`.
//
// A3b SCOPE (this iter — graduated lift):
//   * HybridKvBuffers — FULL multi-seq lift via sibling struct
//     `MultiSeqHybridKvBuffers` + `alloc_multi_seq_hybrid_kv_for_layer`
//     helper.  Mirrors A3a's HbKvBuffers pattern verbatim.  Production
//     default since ADR-029 iter-13 (H10 falsification — see below).
//   * DenseKvBuffers — TYPED CLAMP (slot_count() == 1; slot > 0 returns
//     `MultiSeqError::SlotOutOfRange { max_slots, requested }`; in-bounds
//     unsupported mutations return `CapabilityUnsupported { capability }`
//     with an iter-A3b-2 grep label).  Full lift in iter-A3b-2.
//   * MlxKvCache — TYPED CLAMP (same shape; capability label names
//     iter-A3b-3 + "legacy 4-bit").  Full lift in iter-A3b-3.
//
// H10 STATUS: FALSIFIED at iter-A3a investigation.  `HF2Q_HYBRID_KV`
// defaults ON per `src/debug/investigation_env.rs:878` since ADR-029
// iter-13.  The dossier's "defer to A3b" framing for HybridKvBuffers
// stays intact (this IS the A3b iter), but the priority elevation is
// honoured: HybridKvBuffers gets the FULL lift in iter-1 of A3b, not
// a clamp.
//
// DEFERRED to A3c per dossier R5 (mirrors Qwen35 A2c):
//   * fork_seq cross-slot kernel dispatch (same-buffer cross-region
//     memcpy via `dispatch_kv_cache_copy_seq_*` between slot byte
//     offsets).  iter-A3a returns `CapabilityUnsupported` with a
//     label naming the deferred A3c arc — same shape as Qwen35's
//     iter-2.5 M1 closure.
//
// LayoutNotSupported is NEVER returned: MultiSeqHbKvBuffers only
// supports SeparateSlots, and Paged is an alternate construction
// that this type does not expose.  Bounds-first ordering per
// iter-1.5 cfa-finding-F5 is preserved across all four methods.
// ──────────────────────────────────────────────────────────────────────────

/// Multi-seq variant of [`HbKvBuffers`] — ADR-040 Phase A3a.
///
/// Outermost axis on every buffer is `n_seqs`.  Buffer layouts:
/// - K/V packed: `[n_seqs, nkv_heads, capacity, head_dim]` U8 (1 byte/elem)
/// - K/V norms:  `[n_seqs, nkv_heads, capacity, norms_per_pos]` F32
///
/// Per-seq cursor [`seq_lens`](Self::seq_lens) is `Vec<u32>` of length
/// `n_seqs` (parallel to Qwen35 `FullAttnKvSlot::current_len`).  A
/// per-slot byte offset for kernel writes is
/// `slot.0 * nkv * cap * hd` (packed) / `slot.0 * nkv * cap * norms_per_pos *
/// 4` (norms) — Phase B4c will thread this through the 3 existing
/// `dispatch_hadamard_quantize_kv_hb_*` callers via `MlxBuffer::slice_view`,
/// the same primitive Qwen35 B4a-cont uses.
///
/// Per dossier H7 verification (production Gemma 4 has mixed
/// `LayerType::Full` + `LayerType::Sliding` layers per
/// `src/inference/models/gemma4/model.rs:1250`), `is_sliding` is
/// recorded on this struct just like the legacy `HbKvBuffers`; the
/// per-slot ring-wrap math stays within each slot's region by
/// construction (n_seqs is outermost, so slot N's `capacity` window
/// is contiguous and disjoint from slot M's).
pub struct MultiSeqHbKvBuffers {
    /// Number of physical slots — the outermost axis on every buffer.
    /// Set at construction via [`alloc_hb_kv_for_layer`]; once set,
    /// cannot change without reallocation.
    pub n_seqs: u32,
    /// Byte-packed K indices `[n_seqs, nkv_heads, capacity, head_dim]` U8.
    pub k_packed: MlxBuffer,
    /// K per-position norms `[n_seqs, nkv_heads, capacity, norms_per_pos]` F32.
    pub k_norms: MlxBuffer,
    /// Byte-packed V indices `[n_seqs, nkv_heads, capacity, head_dim]` U8.
    pub v_packed: MlxBuffer,
    /// V per-position norms `[n_seqs, nkv_heads, capacity, norms_per_pos]` F32.
    pub v_norms: MlxBuffer,
    /// Cache capacity in positions (same as the legacy [`HbKvBuffers`]).
    pub capacity: usize,
    /// True if ring-buffer (sliding) semantics.
    pub is_sliding: bool,
    /// Norms per position (1 for D=256, 2 for D=512).
    #[allow(dead_code)]
    pub norms_per_pos: usize,
    /// Per-seq write cursor; `seq_lens[slot.0]` is the number of valid
    /// positions stored in slot `slot.0`.  `len() == n_seqs as usize`
    /// by construction (see [`alloc_hb_kv_for_layer`]).  Mirrors
    /// Qwen35's `FullAttnKvSlot::current_len` discipline.
    pub seq_lens: Vec<u32>,
}

/// Allocate persistent multi-slot KV storage without committing every page of
/// a full-context logical stride at server startup.
///
/// Sliding-window rings stay on the ordinary zeroed, residency-managed path:
/// they are small and may wrap over previously written positions. Full
/// attention is cursor-masked; every position below the cursor is written
/// before attention can read it, so untouched tail pages are not observable.
fn alloc_multi_seq_kv_storage(
    dev: &MlxDevice,
    byte_len: usize,
    dtype: DType,
    shape: Vec<usize>,
    is_ring: bool,
) -> mlx_native::Result<MlxBuffer> {
    if is_ring {
        dev.alloc_buffer(byte_len, dtype, shape)
    } else {
        // SAFETY: every full-attention reader is bounded by the per-slot
        // cursor, which advances only after the producer writes that position.
        // Reset paths lower the cursor and never expose the unwritten tail.
        unsafe { dev.alloc_buffer_for_overwrite(byte_len, dtype, shape) }
    }
}

/// ADR-040 Phase A3a iter-3 — unified [`MultiSeqHbKvBuffers`] allocator.
///
/// **Why this helper exists** (dossier H8):
/// The 3 production HbKvBuffers alloc sites
/// (`src/serve/forward_prefill.rs:843-882`,
/// `src/serve/forward_prefill_batched.rs:443-475`,
/// `src/inference/models/gemma4/forward_gpu.rs:443-459`) currently
/// duplicate the per-layer buffer shape formula inline.  Phase B4c
/// will refactor those 3 sites to call this helper; A3a ships the
/// helper now so the multi-seq buffer-allocation pattern lands in
/// a single place, eliminating drift risk per dossier H8.
///
/// Mirrors [`alloc_hybrid_kv_for_layer`] (same file, line 218) in
/// signature shape; the extra `n_seqs` parameter is the lift this
/// helper introduces.
///
/// At `n_seqs=1` the byte counts are identical to the 3 sites'
/// inline allocs (`nkv * cap * hd` packed + `nkv * cap *
/// norms_per_pos * 4` norms); the only observable shape difference
/// is the leading dimension on every buffer (`[1, nkv, cap, hd]` vs
/// `[nkv, cap, hd]`).  The H8 test pins this byte-equivalence.
///
/// # Errors
///
/// Returns `Err` for `n_seqs == 0`, `nkv == 0`, `hd == 0`, or
/// `cap == 0` — buffer alloc would otherwise underflow the kernel's
/// shape preconditions.  Mirrors `alloc_tq_full_attn_buffers`
/// (Qwen35 `kv_cache.rs:2399-2408`) defensive pre-flight.
pub fn alloc_hb_kv_for_layer(
    dev: &MlxDevice,
    layer_idx: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    is_ring: bool,
    n_seqs: u32,
) -> Result<MultiSeqHbKvBuffers> {
    if n_seqs == 0 {
        return Err(anyhow!(
            "alloc_hb_kv_for_layer L{layer_idx}: n_seqs must be > 0"
        ));
    }
    if nkv == 0 || hd == 0 || cap == 0 {
        return Err(anyhow!(
            "alloc_hb_kv_for_layer L{layer_idx}: nkv/hd/cap must be > 0 \
             (got nkv={nkv}, hd={hd}, cap={cap})"
        ));
    }
    let norms_per_pos = (hd / 256).max(1);
    let n = n_seqs as usize;

    // Packed: [n_seqs, nkv, cap, hd] U8 (1 byte/elem).  Outer n_seqs
    // matches Qwen35 `alloc_tq_full_attn_buffers` convention
    // (`kv_cache.rs:2421-2426`), keeping per-slot byte offsets
    // contiguous and addressable as `slot.0 * (nkv*cap*hd)` for the
    // kernel-dispatcher slot-offset work in Phase B4c.
    let packed_bytes = n * nkv * cap * hd; // U8 → 1 byte/elem
    let packed_shape = vec![n, nkv, cap, hd];

    // Norms: [n_seqs, nkv, cap, norms_per_pos] F32.  norms_per_pos=1
    // (D=256) collapses to an effective 3-D `[n, nkv, cap]` at the
    // kernel level (consistent with the legacy 3-D `vec![nkv, cap]`
    // form when n_seqs=1 + norms_per_pos=1); we keep the 4-D shape
    // here so cfg-shape validation is unambiguous (every dim is
    // explicit, matching the Qwen35 norms-shape pin at
    // `kv_cache.rs:2437-2442`).
    let norms_elems = n * nkv * cap * norms_per_pos;
    let norms_bytes = norms_elems * std::mem::size_of::<f32>();
    let norms_shape = vec![n, nkv, cap, norms_per_pos];

    let k_packed =
        alloc_multi_seq_kv_storage(dev, packed_bytes, DType::U8, packed_shape.clone(), is_ring)
            .map_err(|e| anyhow!("hb_kv L{layer_idx} K packed: {e}"))?;
    let k_norms =
        alloc_multi_seq_kv_storage(dev, norms_bytes, DType::F32, norms_shape.clone(), is_ring)
            .map_err(|e| anyhow!("hb_kv L{layer_idx} K norms: {e}"))?;
    let v_packed = alloc_multi_seq_kv_storage(dev, packed_bytes, DType::U8, packed_shape, is_ring)
        .map_err(|e| anyhow!("hb_kv L{layer_idx} V packed: {e}"))?;
    let v_norms = alloc_multi_seq_kv_storage(dev, norms_bytes, DType::F32, norms_shape, is_ring)
        .map_err(|e| anyhow!("hb_kv L{layer_idx} V norms: {e}"))?;

    Ok(MultiSeqHbKvBuffers {
        n_seqs,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        capacity: cap,
        is_sliding: is_ring,
        norms_per_pos,
        seq_lens: vec![0u32; n],
    })
}

/// **ADR-040 §3.5 iter-A5c (cfa-A5b MAJOR #3)** — pure mapping from a
/// Gemma 4 layer type to the `(is_ring, capacity)` pair that
/// [`alloc_hb_kv_for_layer`] (and the existing per-layer KV-cache
/// allocator at `src/inference/models/gemma4/model.rs:1247-1257`)
/// consumes.
///
/// This helper extracts the per-layer-type branch that was inlined at
/// `model.rs:1249-1257` so the cfa-iter-A5b MAJOR #3 mixed-layer test
/// can walk the SAME mapping the production allocator uses, instead of
/// asserting the boolean argument of `alloc_hb_kv_for_layer` is honoured
/// (the prior iter-A5b test only verified that the boolean is honoured —
/// it did NOT verify the production `LayerType::Full/Sliding` →
/// `(is_ring, capacity)` mapping).
///
/// **Mapping** (mirrors `gemma4/model.rs:1249-1257`):
///
/// | `LayerType` | `is_ring` | `capacity` |
/// |---|---:|---:|
/// | `Sliding` | `true` (ring) | `sliding_window` |
/// | `Full` | `false` (linear) | `max_position_embeddings` |
///
/// **Falsifier-by-design**: swapping the two arms in this helper makes
/// the regression test below fail. Future per-layer-type allocator
/// changes (e.g. ADR-040 Phase B4c slot-level reallocation) MUST route
/// through this helper so the test bank catches branch swaps before
/// they reach production.
pub fn layer_type_to_alloc_params(
    layer_type: crate::serve::config::LayerType,
    sliding_window: usize,
    max_position_embeddings: usize,
) -> (bool, usize) {
    use crate::serve::config::LayerType;
    match layer_type {
        LayerType::Sliding => (true, sliding_window),
        LayerType::Full => (false, max_position_embeddings),
    }
}

/// ADR-040 Phase F `iter-F-kvcap` — per-slot (continuous-batching) variant of
/// [`layer_type_to_alloc_params`].
///
/// The multi-seq (`SlotAware`) KV scaffold reserves a `max_slots × capacity`
/// virtual buffer per layer. Every slot receives the model's complete logical
/// context capacity. Untouched full-attention pages are left uncommitted by
/// [`alloc_multi_seq_kv_storage`], so logical addressability no longer implies
/// eager physical residency. Aggregate physical KV use is governed by the
/// serving admission policy, not by silently shrinking each slot.
///
/// Full-attention storage carries one non-addressable guard position beyond
/// the model context. On the M5 Max, Gemma's production shape at exactly
/// 262,144 positions makes each F16 K layer exactly 1 GiB; the slot-aware
/// Metal path produced divergent decode while otherwise identical capacities
/// 32K, 64K, 128K, 262,143, and 262,145 were coherent. Padding the physical
/// head stride by one position removes that exact-power boundary without
/// changing the advertised context or admitting token 262,145. This is storage
/// padding, not extra logical context.
///
/// **Sliding layers are unchanged** — they are ring buffers capped at
/// `sliding_window` (already per-slot-independent and small); dividing them
/// would corrupt the ring-window semantics.
///
/// `max_slots` is retained in the signature because it remains part of the
/// allocation-policy call site, but it does not alter logical capacity.
pub fn layer_type_to_alloc_params_per_slot(
    layer_type: crate::serve::config::LayerType,
    sliding_window: usize,
    max_position_embeddings: usize,
    max_slots: usize,
) -> (bool, usize) {
    use crate::serve::config::LayerType;
    match layer_type {
        LayerType::Sliding => (true, sliding_window),
        LayerType::Full => {
            let _ = max_slots;
            (false, max_position_embeddings.saturating_add(1))
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A3a iter-3 — MultiSeqKvCache impl for MultiSeqHbKvBuffers.
//
// Mirrors the Qwen35 A2a impl at `src/inference/models/qwen35/kv_cache.rs:
// 2579-2780` in structure (bounds-first per iter-1.5 cfa-finding-F5;
// fork_seq returns `CapabilityUnsupported` per iter-2.5 M1) and in
// invariants (per-slot cursor isolation; drop_seq does NOT zero the
// underlying buffer bytes — Phase B4c will reuse the slot's region on
// next admission; recurrent-content invariance is the per-slot
// analogue of Qwen35 M4).
//
// Phase A3a scope: per-slot CURSOR bookkeeping only.  GPU buffer
// content writes land via the 3 existing alloc sites' `dispatch_
// hadamard_quantize_kv_hb_*` callers at Phase B4c; iter-A3a's trait
// surface mutates `seq_lens[slot.0]` and validates bounds.  This
// matches the dossier R2 mitigation: the per-cache trait owns the
// cursor; the forward-path slot threading is a separate phase.
// ──────────────────────────────────────────────────────────────────────────

impl crate::serve::multi_seq_kv::MultiSeqKvCache for MultiSeqHbKvBuffers {
    fn layout(&self) -> crate::serve::multi_seq_kv::MultiSeqLayout {
        crate::serve::multi_seq_kv::MultiSeqLayout::SeparateSlots
    }

    fn slot_count(&self) -> u32 {
        // `MultiSeqHbKvBuffers::n_seqs` is already `u32` (see struct
        // definition above); no cast.
        self.n_seqs
    }

    fn seq_len(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<u32, crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST (iter-1.5 cfa-finding-F5 ordering).
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only — MultiSeqHbKvBuffers does
        //    not expose Paged.
        // 3. Return the per-seq cursor directly; `seq_lens.len() ==
        //    n_seqs` by construction (alloc_hb_kv_for_layer).  Unlike
        //    Qwen35's multi-layer canonical-from-`full_attn[0]` read,
        //    this is a per-layer struct — there is exactly one cursor
        //    per slot per buffer, no canonical-vs-per-layer
        //    homogeneity concern, no debug_assert needed.
        Ok(self.seq_lens[slot.0 as usize])
    }

    fn append_for_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        n_tokens: u32,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST (iter-1.5 cfa-finding-F5).
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only.
        // 3. Budget: SeparateSlots cannot SlotOom on append (buffers
        //    are pre-allocated at construction; cursor overflow is
        //    bounded by `capacity` and protected by `saturating_add`).
        //
        // ADR-040 Phase A3a scope: bump the per-seq cursor.  The
        // underlying k_packed / v_packed / norms bytes for slot
        // `slot.0` are written by the kernel dispatcher at Phase B4c
        // via `MlxBuffer::slice_view(byte_offset, n_elements)` (same
        // primitive Qwen35 B4a-cont uses).
        let cur = &mut self.seq_lens[slot.0 as usize];
        *cur = cur.saturating_add(n_tokens);
        Ok(())
    }

    fn drop_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only — no LayoutNotSupported.
        // 3. Budget: drop is a pure release; SlotOom unreachable.
        //
        // ADR-040 Phase A3a scope: cursor-only reset.  The underlying
        // K/V bytes are NOT zeroed; the next `append_for_seq` into
        // this slot will overwrite them via the kernel dispatcher at
        // Phase B4c.  This matches Qwen35 A2a's discipline (the
        // `qwen35_hybrid_kv_drop_does_not_zero_recurrent_buffer_a2a`
        // pin at `qwen35/kv_cache.rs:6949+` — recurrent-content
        // invariance under drop_seq).
        self.seq_lens[slot.0 as usize] = 0;
        Ok(())
    }

    fn fork_seq(
        &mut self,
        src: crate::serve::multi_seq_kv::SlotId,
        dst: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds — src FIRST per iter-1.5 cfa-finding-F5 (so a
        //    fully invalid `(src, dst)` pair surfaces src as the OOR
        //    victim deterministically; matches the trait doc + the
        //    NoopMultiSeqKvCache fixture-parity contract).
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
        // 3. Same-slot fork is a no-op per trait spec.
        if src == dst {
            return Ok(());
        }
        // ──────────────────────────────────────────────────────────────
        // ADR-040 Phase A3c (2026-05-30) — REAL cross-slot fork.
        //
        // Replaces the prior `CapabilityUnsupported` typed-deferral
        // with same-buffer cross-region memcpy on the four MultiSeq
        // HB buffers (k_packed U8 / k_norms F32 / v_packed U8 /
        // v_norms F32) per the layout proofs in
        // `alloc_hb_kv_for_layer` at `kv_cache.rs:252-334` — n_seqs
        // OUTERMOST on every buffer so per-slot byte stride =
        // `total_bytes / n_seqs`.
        //
        // Cursor copy: `seq_lens[dst] = seq_lens[src]`.
        //
        // Parallel to Qwen35 A2c at `qwen35/kv_cache.rs:3044+`.
        // ──────────────────────────────────────────────────────────────
        let src_idx = src.0 as usize;
        let dst_idx = dst.0 as usize;
        let n_seqs = self.n_seqs as usize;
        gemma4_copy_buffer_slot_region(&mut self.k_packed, src_idx, dst_idx, n_seqs).map_err(
            |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: gemma4_leak_static_str(format!(
                    "fork_seq: MultiSeqHbKvBuffers k_packed copy failed ({e})"
                )),
            },
        )?;
        gemma4_copy_buffer_slot_region(&mut self.k_norms, src_idx, dst_idx, n_seqs).map_err(
            |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: gemma4_leak_static_str(format!(
                    "fork_seq: MultiSeqHbKvBuffers k_norms copy failed ({e})"
                )),
            },
        )?;
        gemma4_copy_buffer_slot_region(&mut self.v_packed, src_idx, dst_idx, n_seqs).map_err(
            |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: gemma4_leak_static_str(format!(
                    "fork_seq: MultiSeqHbKvBuffers v_packed copy failed ({e})"
                )),
            },
        )?;
        gemma4_copy_buffer_slot_region(&mut self.v_norms, src_idx, dst_idx, n_seqs).map_err(
            |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: gemma4_leak_static_str(format!(
                    "fork_seq: MultiSeqHbKvBuffers v_norms copy failed ({e})"
                )),
            },
        )?;
        // Cursor copy AFTER buffer copy.
        self.seq_lens[dst_idx] = self.seq_lens[src_idx];
        Ok(())
    }
}

/// ADR-040 Phase A3c (2026-05-30) — leak a `String` into a `&'static
/// str` for `MultiSeqError::CapabilityUnsupported` payloads
/// constructed from runtime context.  Sibling of Qwen35 A2c's
/// `leak_static_str` at `qwen35/kv_cache.rs`.
#[inline]
fn gemma4_leak_static_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// ADR-040 Phase A3c (2026-05-30) — same-buffer cross-region byte copy
/// for a Gemma 4 multi-seq buffer at `[n_seqs, ...]` outermost.
///
/// Per-slot byte stride is `buf.byte_len() / n_seqs`.  Mirrors Qwen35
/// A2c's `copy_buffer_slot_region` at `qwen35/kv_cache.rs:3382+`.
/// Same n_seqs-outermost invariant holds for all four Gemma 4 multi-seq
/// sibling structs (`MultiSeqHbKvBuffers` at line 197+,
/// `MultiSeqHybridKvBuffers` at line 870+, `MultiSeqDenseKvBuffers` at
/// line 1319+, `MultiSeqMlxKvCache` at line 1694+) — the alloc helpers
/// `alloc_hb_kv_for_layer` / `alloc_multi_seq_hybrid_kv_for_layer` /
/// `alloc_multi_seq_dense_kv_for_layer` / `alloc_multi_seq_mlx_kv_for_layer`
/// all emit shape `[n, ...]` with n=`n_seqs` as the leading dim ⇒
/// per-slot region is contiguous of size `total_bytes / n_seqs`.
fn gemma4_copy_buffer_slot_region(
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

/// Copy only cursor-visible rows between overwrite-backed non-ring slots.
/// Buffer shape is `[n_seqs, heads, capacity, inner]` or
/// `[n_seqs, heads, capacity]`; each head's live prefix is contiguous but
/// heads are separated by the full capacity stride.
fn gemma4_copy_buffer_slot_prefix(
    buf: &mut MlxBuffer,
    src_idx: usize,
    dst_idx: usize,
    n_seqs: usize,
    live_tokens: usize,
) -> Result<()> {
    let shape = buf.shape().to_vec();
    anyhow::ensure!(
        matches!(shape.len(), 3 | 4) && shape[0] == n_seqs,
        "Gemma fork prefix expected [n_seqs, heads, capacity, ...], got {:?}",
        shape
    );
    anyhow::ensure!(
        src_idx < n_seqs && dst_idx < n_seqs && live_tokens <= shape[2],
        "Gemma fork prefix outside allocation: src={src_idx} dst={dst_idx} live_tokens={live_tokens} n_seqs={n_seqs} capacity={}",
        shape[2]
    );
    if live_tokens == 0 {
        return Ok(());
    }
    let inner_elements = shape.get(3).copied().unwrap_or(1);
    let bytes_per_position = inner_elements
        .checked_mul(buf.dtype().size_of())
        .ok_or_else(|| anyhow!("Gemma fork prefix byte extent overflow"))?;
    let head_stride = shape[2]
        .checked_mul(bytes_per_position)
        .ok_or_else(|| anyhow!("Gemma fork prefix head stride overflow"))?;
    let slot_stride = shape[1]
        .checked_mul(head_stride)
        .ok_or_else(|| anyhow!("Gemma fork prefix slot stride overflow"))?;
    let copy_bytes = live_tokens
        .checked_mul(bytes_per_position)
        .ok_or_else(|| anyhow!("Gemma fork prefix copy extent overflow"))?;
    let bytes = buf
        .as_mut_slice::<u8>()
        .map_err(|error| anyhow!("Gemma fork prefix as_mut_slice<u8>: {error}"))?;
    for head in 0..shape[1] {
        let src_start = src_idx * slot_stride + head * head_stride;
        let dst_start = dst_idx * slot_stride + head * head_stride;
        bytes.copy_within(src_start..src_start + copy_bytes, dst_start);
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 iter-B4c-kernel iter-1 — per-slot reset primitive for
// MultiSeqHbKvBuffers (Gemma 4 mirror of Qwen35's
// HybridKvCache::reset_for_slot per §6.1.27).
// ──────────────────────────────────────────────────────────────────────────

impl MultiSeqHbKvBuffers {
    /// **ADR-040 iter-B4c-kernel iter-1** (2026-05-30) — per-slot
    /// reset for the persistent multi-seq `MultiSeqHbKvBuffers` worker
    /// hot path.
    ///
    /// Cross-architecture mirror of Qwen35
    /// `HybridKvCache::reset_for_slot` (per §6.1.27 closure block).
    /// Used by `engine::generate_gemma4_once_slot_aware` to clear a
    /// slot's state at request entry + exit so the persistent per-layer
    /// `MultiSeqHbKvBuffers` is request-isolated within the slot — the
    /// next request to land on the same slot sees a zero-cursor cache.
    ///
    /// **Layout proof** (mirror of A2b §6.1.23 / iter-C2d-cont-kernel
    /// iter-1 §6.1.27 reset_for_slot discipline):
    /// - **seq_lens**: `Vec<u32>` of length `n_seqs`. Per-slot reset →
    ///   set `seq_lens[slot_idx] = 0`; other slots untouched.  This is
    ///   the load-bearing cursor that bounds the HB-packed K/V SDPA
    ///   read path.
    /// - **k_packed / v_packed (U8, `[n_seqs, nkv, capacity, head_dim]`
    ///   row-major)**: per-slot region size = `nkv * capacity *
    ///   head_dim` bytes.  **NOT zeroed** — same discipline as Qwen35
    ///   full_attn: the HB-SDPA read path masks against
    ///   `seq_lens[slot_idx]` (positions ≥ cursor are unreadable to
    ///   the kernel).  Stale bytes beyond the cursor are structurally
    ///   unreachable, matching the existing `drop_seq` invariant pinned
    ///   by `gemma4_hb_kv_drop_does_not_zero_k_packed_buffer`.
    /// - **k_norms / v_norms (F32, `[n_seqs, nkv, capacity,
    ///   norms_per_pos]`)**: same cursor-masked discipline — NOT
    ///   zeroed.  The norms read path is gated on the same
    ///   `seq_lens[slot_idx]` cursor that gates packed.
    ///
    /// # Errors
    ///
    /// - `slot.0 >= self.n_seqs` (bounds-first per A2b iter-1.5
    ///   cfa-finding-F5 ordering) — returns typed
    ///   [`crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange`].
    ///
    /// # Per-slot byte-equivalence pin
    ///
    /// At `slot = SlotId(0)` AND `n_seqs == 1` this is byte-equivalent
    /// to setting `seq_lens[0] = 0` directly (the existing `drop_seq`
    /// shape).  H80 pins this in the test module.
    ///
    /// # Why distinct from `drop_seq`
    ///
    /// `drop_seq` is the `MultiSeqKvCache` trait method called by the
    /// scheduler on per-slot release.  `reset_for_slot` is the
    /// orchestrator-level entry point called at iter-B4c-kernel iter-1's
    /// `generate_gemma4_once_slot_aware` entry + exit — the inherent
    /// method gives the engine.rs orchestrator a named API that mirrors
    /// Qwen35 iter-1's `HybridKvCache::reset_for_slot` 1:1 for
    /// cross-architecture grep-symmetry, without depending on the
    /// `MultiSeqKvCache` trait (which lives in a separate import
    /// surface).  Bodies are structurally identical today; if iter-B4c-
    /// kernel-iter-{2,3,4} ever lifts the K/V byte zeroing discipline
    /// (e.g. to defend against a future kernel-write-past-cursor bug),
    /// the two methods diverge — orchestrator-driven entry/exit
    /// resets honour the lift, scheduler-driven `drop_seq` does not.
    pub fn reset_for_slot(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // Bounds-first per A2b §6.1.23 iter-1.5 cfa-finding-F5.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // Reset the per-slot cursor.  K/V packed + norms bytes are NOT
        // zeroed (cursor-masked read path — see layout proof above;
        // matches `drop_seq` invariant).
        self.seq_lens[slot.0 as usize] = 0;
        Ok(())
    }
}

/// Per-layer dense F32/F16 KV buffers for dense attention path (ADR-009).
///
/// **Legacy single-seq variant.**  ADR-040 Phase A3b iter-2 shipped the
/// multi-seq sibling [`MultiSeqDenseKvBuffers`] +
/// [`alloc_multi_seq_dense_kv_for_layer`] for `HF2Q_USE_DENSE=1` paths
/// that need `n_seqs > 1`.  This struct remains the legacy single-seq
/// path used by the 3 inline production alloc sites
/// (`forward_prefill.rs:705`, `forward_prefill_batched.rs:367`,
/// `engine.rs:6836`) at implicit `n_seqs=1` byte-for-byte unchanged
/// until Phase B4c re-routes those sites through
/// `alloc_multi_seq_dense_kv_for_layer`.  The
/// `impl MultiSeqKvCache for DenseKvBuffers` below is a TYPED CLAMP
/// (`slot_count() == 1`; in-bounds slot 0 ops return
/// `CapabilityUnsupported` pointing at the multi-seq sibling).
pub struct DenseKvBuffers {
    pub k: MlxBuffer,
    pub v: MlxBuffer,
    /// Capacity (positions) in this layer's cache. Sliding layers use
    /// ring-buffer mode: capacity == sliding_window, writes wrap.
    /// Global layers use linear mode: capacity >= seq_len + max_tokens.
    pub capacity: usize,
    /// True if this is a sliding layer (ring-buffer semantics).
    pub is_sliding: bool,
    /// ADR-017 Phase E.a iter-3.5a (Codex round-1 LOW #4) — KV element
    /// dtype invariant. Today every engine in a process uses one
    /// `HF2Q_F16_KV` setting (parsed at LazyLock init), so all
    /// `DenseKvBuffers` in the live LcpRegistry necessarily share
    /// dtype. Recording it explicitly:
    ///
    ///   * makes the invariant a load-bearing struct field (not a
    ///     process-implicit assumption that could regress under any
    ///     future hot-reload / multi-engine path),
    ///   * lets the engine's `take_prefix` capacity-check site assert
    ///     `cached.dtype == model.kv_dtype` per layer alongside the
    ///     existing capacity + is_sliding checks, closing a class of
    ///     silent-corruption bugs at the type level.
    ///
    /// Populated at every construction site: `forward_prefill.rs` per-
    /// layer alloc, `forward_prefill_batched.rs` per-layer alloc, and
    /// `engine.rs::kv_restore_gemma` per-layer alloc.
    pub dtype: DType,
}

impl crate::serve::kv_persist::lcp_registry::ByteSized for DenseKvBuffers {
    /// Exact byte count of the K + V buffers for this layer.
    /// Uses `MlxBuffer::byte_len()` — the same API used at forward_mlx.rs:1167+
    /// and 5158+. No estimation.
    fn byte_len(&self) -> u64 {
        (self.k.byte_len() + self.v.byte_len()) as u64
    }
}

/// ADR-028 Phase 10 (iter-347): hybrid K storage — F16 K alongside TQ-HB-packed V.
///
/// Motivation (ADR-028 §iter-346 audit):
///   * Pure TQ-HB (today's default): 504 MB raw F32 → 128 MB packed (3.94× saving).
///   * Pure F16 K + F16 V (peer): 504 MB → 252 MB (2× saving).
///   * Hybrid (F16 K + TQ-HB V): 504 MB → 158 MB (3.19× saving — 81% of TQ-HB).
///
/// The structural decode-side gap vs the peer (1.81× per-dispatch wall on
/// our TQ-HB SDPA, formally measured iter-326..342) is owned by the K-side
/// scalar dequant loop inside `flash_attn_vec_tq_hb`: peer's K is F16 and consumed
/// by `simdgroup_matrix` matmul, ours is byte-packed and consumed by per-thread
/// scalar lookup against the codebook. Storing K as F16 (this struct) and using a
/// new `flash_attn_vec_hybrid_dk256` SDPA kernel (Phase 10d) brings the K-side
/// throughput up to peer-equivalent simdgroup math while the V-side stays in
/// 1-byte-per-element TQ-HB packing.
///
/// Field layout mirrors the union of `DenseKvBuffers` (K) and `HbKvBuffers` (V) so
/// existing snapshot / restore / KV-persist code that walks the K and V buffers
/// can pattern-match by field name without learning a new shape.
///
/// Allocation gate: env `HF2Q_HYBRID_KV` (parsed in `investigation_env.rs`).
/// Default OFF until parity + bench gates pass (Phase 10f/g). When ON, the
/// per-layer alloc site at `forward_decode` (currently building `HbKvBuffers`)
/// instead builds `HybridKvBuffers`, the K-encode dispatch is skipped, and the
/// SDPA dispatcher routes to the hybrid kernel.
pub struct HybridKvBuffers {
    /// Dense F16 K cache `[nkv_heads, capacity, head_dim]`. dtype is always
    /// `mlx_native::DType::F16` for this struct (the whole point — F16 K → peer
    /// SDPA-equivalent simdgroup matmul). No F32 variant: F32 K would erase the
    /// memory advantage of the hybrid design (158 MB → 284 MB at gemma4 32K
    /// context, defeating the purpose of mixing in TQ-HB on V at all).
    pub k: MlxBuffer,
    /// Byte-packed V indices `[nkv_heads, capacity, head_dim]` U8 — same layout
    /// and codec as `HbKvBuffers::v_packed`. The V-encode dispatch
    /// (`hadamard_quantize_kv_*`) writes here unchanged.
    pub v_packed: MlxBuffer,
    /// V per-position norms — same layout as `HbKvBuffers::v_norms`.
    /// (D=256 → 1/pos, D=512 → 2/pos.)
    pub v_norms: MlxBuffer,
    /// Cache capacity in positions (matches `DenseKvBuffers::capacity` and
    /// `HbKvBuffers::capacity` — populated identically at alloc site).
    pub capacity: usize,
    /// True if ring-buffer (sliding) semantics — same as the sibling structs.
    pub is_sliding: bool,
    /// Norms per position (1 for D=256, 2 for D=512). Mirrors
    /// `HbKvBuffers::norms_per_pos`.
    #[allow(dead_code)]
    pub norms_per_pos: usize,
    /// ADR-030 iter-96: BF16 K cache for DFlash spec-decode xlen verify
    /// path. Same layout `[nkv_heads, capacity, head_dim]` as `k` but BF16
    /// dtype. Populated from `pf_k_perm` BF16 (head_norm_rope's bf16 output)
    /// via `dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major`. Used by
    /// xlen branch's SDPA to read BF16 K bit-identical to what Option C
    /// reads from pf_k_perm — avoids the F16-roundtrip precision drift
    /// root-caused at iter-92/93.  Lazy-alloc'd on first xlen-mode call.
    pub bf16_xlen_k: Option<MlxBuffer>,
    /// ADR-030 iter-96: BF16 V cache, same semantics as `bf16_xlen_k`.
    pub bf16_xlen_v: Option<MlxBuffer>,
}

impl crate::serve::kv_persist::lcp_registry::ByteSized for HybridKvBuffers {
    /// Exact byte count: F16 K + U8 V + F32 V-norms. Used by the LcpRegistry
    /// byte budget the same way `DenseKvBuffers::byte_len` is.
    fn byte_len(&self) -> u64 {
        (self.k.byte_len() + self.v_packed.byte_len() + self.v_norms.byte_len()) as u64
    }
}

/// Exact bytes owned by one hybrid-KV LCP snapshot layer at `snapshot_capacity`.
///
/// The live buffer dtype is the representation authority. Packed V owns one
/// U8 code per head-dimension element plus `norms_per_pos` F32 norms per
/// position. Full-F16 V owns two bytes per element and the canonical shared
/// four-byte norms dummy; it must not be charged as if the dummy had a token
/// axis.
pub(crate) fn hybrid_lcp_snapshot_layer_bytes(
    num_kv_heads: usize,
    snapshot_capacity: usize,
    head_dim: usize,
    norms_per_pos: usize,
    k_dtype: DType,
    v_dtype: DType,
) -> Result<u64> {
    anyhow::ensure!(
        num_kv_heads > 0 && snapshot_capacity > 0 && head_dim > 0 && norms_per_pos > 0,
        "hybrid LCP snapshot dimensions must be nonzero"
    );
    anyhow::ensure!(
        k_dtype == DType::F16,
        "hybrid LCP snapshot K must be F16, got {k_dtype:?}"
    );
    anyhow::ensure!(
        matches!(v_dtype, DType::U8 | DType::F16),
        "hybrid LCP snapshot V must be U8 or F16, got {v_dtype:?}"
    );

    let positions = u64::try_from(num_kv_heads)
        .ok()
        .and_then(|heads| heads.checked_mul(snapshot_capacity as u64))
        .ok_or_else(|| anyhow!("hybrid LCP snapshot position count overflow"))?;
    let elements = positions
        .checked_mul(head_dim as u64)
        .ok_or_else(|| anyhow!("hybrid LCP snapshot element count overflow"))?;
    let k_bytes = elements
        .checked_mul(k_dtype.size_of() as u64)
        .ok_or_else(|| anyhow!("hybrid LCP snapshot K byte count overflow"))?;
    let v_bytes = elements
        .checked_mul(v_dtype.size_of() as u64)
        .ok_or_else(|| anyhow!("hybrid LCP snapshot V byte count overflow"))?;
    let norms_bytes = if v_dtype == DType::F16 {
        std::mem::size_of::<f32>() as u64
    } else {
        positions
            .checked_mul(norms_per_pos as u64)
            .and_then(|norms| norms.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or_else(|| anyhow!("hybrid LCP snapshot V-norm byte count overflow"))?
    };
    k_bytes
        .checked_add(v_bytes)
        .and_then(|bytes| bytes.checked_add(norms_bytes))
        .ok_or_else(|| anyhow!("hybrid LCP snapshot layer byte count overflow"))
}

/// Snapshot the populated prefix of every hybrid-KV layer without changing its
/// stored V representation.
///
/// Serial and batched prefill use the same cache contract, so keeping this copy
/// in the representation owner prevents their dtype and dummy-buffer handling
/// from drifting apart again.
pub(crate) fn snapshot_hybrid_kv_for_lcp(
    dev: &MlxDevice,
    live_hybrid: &[HybridKvBuffers],
    sequence_len: usize,
    snapshot_capacity: usize,
) -> Result<Vec<std::sync::Arc<HybridKvBuffers>>> {
    anyhow::ensure!(
        sequence_len <= snapshot_capacity,
        "hybrid LCP snapshot sequence {sequence_len} exceeds capacity {snapshot_capacity}"
    );
    let mut snapshot = Vec::with_capacity(live_hybrid.len());
    for (layer_idx, live_layer) in live_hybrid.iter().enumerate() {
        let shape = live_layer.k.shape();
        anyhow::ensure!(
            shape.len() == 3,
            "hybrid LCP snapshot K L{layer_idx} must be rank 3, got {shape:?}"
        );
        let num_kv_heads = shape[0];
        let live_capacity = shape[1];
        let head_dim = shape[2];
        anyhow::ensure!(
            live_capacity == live_layer.capacity,
            "hybrid LCP snapshot K L{layer_idx} shape capacity {live_capacity} != metadata {}",
            live_layer.capacity
        );
        anyhow::ensure!(
            sequence_len <= live_capacity,
            "hybrid LCP snapshot sequence {sequence_len} exceeds live L{layer_idx} capacity {live_capacity}"
        );
        let norms_per_pos = live_layer.norms_per_pos;
        let k_dtype = live_layer.k.dtype();
        let v_dtype = live_layer.v_packed.dtype();
        let _ = hybrid_lcp_snapshot_layer_bytes(
            num_kv_heads,
            snapshot_capacity,
            head_dim,
            norms_per_pos,
            k_dtype,
            v_dtype,
        )?;
        anyhow::ensure!(
            live_layer.v_norms.dtype() == DType::F32,
            "hybrid LCP snapshot V norms L{layer_idx} must be F32, got {:?}",
            live_layer.v_norms.dtype()
        );

        let checked_elements = |capacity: usize, inner: usize, what: &str| -> Result<usize> {
            num_kv_heads
                .checked_mul(capacity)
                .and_then(|elements| elements.checked_mul(inner))
                .ok_or_else(|| anyhow!("hybrid LCP snapshot {what} L{layer_idx} extent overflow"))
        };
        let k_elements = checked_elements(snapshot_capacity, head_dim, "K")?;
        let v_elements = checked_elements(snapshot_capacity, head_dim, "V")?;
        let k_bytes = k_elements
            .checked_mul(k_dtype.size_of())
            .ok_or_else(|| anyhow!("hybrid LCP snapshot K L{layer_idx} byte overflow"))?;
        let v_bytes = v_elements
            .checked_mul(v_dtype.size_of())
            .ok_or_else(|| anyhow!("hybrid LCP snapshot V L{layer_idx} byte overflow"))?;
        let mut k_snapshot = dev
            .alloc_buffer(
                k_bytes,
                k_dtype,
                vec![num_kv_heads, snapshot_capacity, head_dim],
            )
            .map_err(|error| anyhow!("hybrid LCP snapshot K L{layer_idx} alloc: {error}"))?;
        let mut v_snapshot = dev
            .alloc_buffer(
                v_bytes,
                v_dtype,
                vec![num_kv_heads, snapshot_capacity, head_dim],
            )
            .map_err(|error| anyhow!("hybrid LCP snapshot V L{layer_idx} alloc: {error}"))?;

        let mut norms_snapshot = if v_dtype == DType::F16 {
            anyhow::ensure!(
                live_layer.v_norms.data_byte_len() == std::mem::size_of::<f32>(),
                "hybrid LCP snapshot F16 V norms L{layer_idx} must be the canonical four-byte dummy, got {} bytes",
                live_layer.v_norms.data_byte_len()
            );
            dev.alloc_buffer(std::mem::size_of::<f32>(), DType::F32, vec![1])
                .map_err(|error| {
                    anyhow!("hybrid LCP snapshot V norms dummy L{layer_idx} alloc: {error}")
                })?
        } else {
            let norm_elements = checked_elements(snapshot_capacity, norms_per_pos, "V norms")?;
            let norm_bytes = norm_elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| anyhow!("hybrid LCP snapshot V norms L{layer_idx} byte overflow"))?;
            let norm_shape = if norms_per_pos == 1 {
                vec![num_kv_heads, snapshot_capacity]
            } else {
                vec![num_kv_heads, snapshot_capacity, norms_per_pos]
            };
            dev.alloc_buffer(norm_bytes, DType::F32, norm_shape)
                .map_err(|error| {
                    anyhow!("hybrid LCP snapshot V norms L{layer_idx} alloc: {error}")
                })?
        };

        let copy_prefix = |source: &MlxBuffer,
                           destination: &mut MlxBuffer,
                           element_bytes: usize,
                           inner: usize,
                           what: &str|
         -> Result<()> {
            let source: &[u8] = source.as_slice().map_err(|error| {
                anyhow!("hybrid LCP snapshot {what} L{layer_idx} source: {error}")
            })?;
            let destination: &mut [u8] = destination.as_mut_slice().map_err(|error| {
                anyhow!("hybrid LCP snapshot {what} L{layer_idx} destination: {error}")
            })?;
            let copy_len = sequence_len
                .checked_mul(inner)
                .and_then(|elements| elements.checked_mul(element_bytes))
                .ok_or_else(|| {
                    anyhow!("hybrid LCP snapshot {what} L{layer_idx} copy extent overflow")
                })?;
            let source_stride = live_capacity
                .checked_mul(inner)
                .and_then(|elements| elements.checked_mul(element_bytes))
                .ok_or_else(|| {
                    anyhow!("hybrid LCP snapshot {what} L{layer_idx} source stride overflow")
                })?;
            let destination_stride = snapshot_capacity
                .checked_mul(inner)
                .and_then(|elements| elements.checked_mul(element_bytes))
                .ok_or_else(|| {
                    anyhow!("hybrid LCP snapshot {what} L{layer_idx} destination stride overflow")
                })?;
            for head in 0..num_kv_heads {
                let source_offset = head * source_stride;
                let destination_offset = head * destination_stride;
                let source_row = source
                    .get(source_offset..source_offset + copy_len)
                    .ok_or_else(|| {
                        anyhow!("hybrid LCP snapshot {what} L{layer_idx} source is truncated")
                    })?;
                let destination_row = destination
                    .get_mut(destination_offset..destination_offset + copy_len)
                    .ok_or_else(|| {
                        anyhow!("hybrid LCP snapshot {what} L{layer_idx} destination is truncated")
                    })?;
                destination_row.copy_from_slice(source_row);
            }
            Ok(())
        };
        copy_prefix(
            &live_layer.k,
            &mut k_snapshot,
            k_dtype.size_of(),
            head_dim,
            "K",
        )?;
        copy_prefix(
            &live_layer.v_packed,
            &mut v_snapshot,
            v_dtype.size_of(),
            head_dim,
            "V",
        )?;
        if v_dtype == DType::F16 {
            let source: &[u8] = live_layer.v_norms.as_slice().map_err(|error| {
                anyhow!("hybrid LCP snapshot V norms dummy L{layer_idx} source: {error}")
            })?;
            let destination: &mut [u8] = norms_snapshot.as_mut_slice().map_err(|error| {
                anyhow!("hybrid LCP snapshot V norms dummy L{layer_idx} destination: {error}")
            })?;
            anyhow::ensure!(
                source.len() == destination.len(),
                "hybrid LCP snapshot V norms dummy L{layer_idx} extent changed ({} != {})",
                source.len(),
                destination.len()
            );
            destination.copy_from_slice(source);
        } else {
            copy_prefix(
                &live_layer.v_norms,
                &mut norms_snapshot,
                std::mem::size_of::<f32>(),
                norms_per_pos,
                "V norms",
            )?;
        }

        snapshot.push(std::sync::Arc::new(HybridKvBuffers {
            k: k_snapshot,
            v_packed: v_snapshot,
            v_norms: norms_snapshot,
            capacity: snapshot_capacity,
            is_sliding: live_layer.is_sliding,
            norms_per_pos,
            bf16_xlen_k: None,
            bf16_xlen_v: None,
        }));
    }
    Ok(snapshot)
}

/// ADR-017 Phase E.a sub-iter "gemma-hybrid-lcp" (2026-08-03) — per-layer
/// payload for Gemma 4's LCP partial-prefill registry across KV regimes.
///
/// Why this exists: the registry previously stored `Vec<Arc<DenseKvBuffers>>`
/// and the end-of-prefill snapshot was gated `kv_lcp_resume && use_dense` —
/// under the PRODUCTION hybrid regime (`HF2Q_HYBRID_KV` default-ON,
/// ADR-029 iter-13) no snapshot was ever taken, so Gemma 4 had NO LCP
/// prefix resume at all in production (safe but slow; every multi-turn
/// request re-prefilled the full conversation).
///
/// The dual-leg requirement comes from how prefill + decode consume the
/// caches: prefill attention reads the DENSE SDPA path
/// (`forward_prefill.rs` `flash_attn_vec` over `dense_kvs`), while decode
/// under the hybrid regime reads `hybrid_kv` (F16 K + TQ-HB V). An LCP
/// resume must therefore restore BOTH legs — dense for the resumed
/// prefill's attention over `[0..k)`, hybrid for the subsequent decode —
/// or the resumed request attends over zeroed bytes on the unrestored
/// leg (the same silent-corruption class ADR-027 sub-iter 23d-γ closed
/// for qwen35).
///
/// Regime matrix (process-fixed at startup):
///   * `use_dense=1` (legacy dense regime) → `Dense` (today's path,
///     unchanged).
///   * hybrid production (`hybrid_kv=1`) → `DenseAndHybrid`.
///   * HB-encoded opt-out (`hybrid_kv=0` without use_dense) → `Dense`
///     payload, but the engine gate keeps LCP auto-disabled there
///     (`effective_kv_lcp_resume` — packed-K restore is NOT covered by
///     this sub-iter; restoring only the dense leg would corrupt decode
///     on the packed leg).
pub enum GemmaLcpLayerKv {
    /// Dense-only payload (legacy dense regime; unchanged semantics).
    Dense(DenseKvBuffers),
    /// Production hybrid payload: dense leg (prefill SDPA) + hybrid leg
    /// (decode). Tuple order is (dense, hybrid) — the same order the
    /// install site mounts them (`weights.dense_kvs`, then
    /// `weights.hybrid_kv`).
    DenseAndHybrid(DenseKvBuffers, HybridKvBuffers),
}

/// Manual Debug (the leg types don't implement it — MlxBuffer fields).
/// Compact: variant + capacities only, no buffer contents.
impl std::fmt::Debug for GemmaLcpLayerKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(d) => f
                .debug_struct("Dense")
                .field("capacity", &d.capacity)
                .field("is_sliding", &d.is_sliding)
                .field("dtype", &d.dtype)
                .finish(),
            Self::DenseAndHybrid(d, h) => f
                .debug_struct("DenseAndHybrid")
                .field("dense_capacity", &d.capacity)
                .field("hybrid_capacity", &h.capacity)
                .field("is_sliding", &d.is_sliding)
                .finish(),
        }
    }
}

impl GemmaLcpLayerKv {
    /// The dense leg, regardless of variant (always present).
    pub fn dense(&self) -> &DenseKvBuffers {
        match self {
            Self::Dense(d) => d,
            Self::DenseAndHybrid(d, _) => d,
        }
    }

    /// The hybrid leg when present.
    pub fn hybrid(&self) -> Option<&HybridKvBuffers> {
        match self {
            Self::Dense(_) => None,
            Self::DenseAndHybrid(_, h) => Some(h),
        }
    }
}

impl crate::serve::kv_persist::lcp_registry::ByteSized for GemmaLcpLayerKv {
    /// Exact byte count across both legs (hybrid variant) or the dense
    /// leg alone — no estimation, same discipline as the sibling impls.
    fn byte_len(&self) -> u64 {
        match self {
            Self::Dense(d) => crate::serve::kv_persist::lcp_registry::ByteSized::byte_len(d),
            Self::DenseAndHybrid(d, h) => {
                crate::serve::kv_persist::lcp_registry::ByteSized::byte_len(d)
                    + crate::serve::kv_persist::lcp_registry::ByteSized::byte_len(h)
            }
        }
    }
}

/// ADR-028 Phase 10c (iter-348): per-layer F16-K + TQ-HB-V buffer allocator.
///
/// Single-source-of-truth for the hybrid allocation shape — called from
/// 3 sites (decode lazy-alloc, per-token prefill alloc, batched-prefill
/// alloc) so all three stay in lockstep.
///
/// F16 K: 2 bytes/elem, shape `[nkv, cap, hd]`.
/// V layout identical to legacy `HbKvBuffers` V-side (1 byte/elem packed +
/// per-pos F32 norms with `norms_per_pos = max(1, hd / 256)`).
pub(crate) fn alloc_hybrid_kv_for_layer(
    dev: &MlxDevice,
    layer_idx: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    is_ring: bool,
) -> anyhow::Result<HybridKvBuffers> {
    let norms_per_pos = (hd / 256).max(1);
    let norms_n = nkv * cap * norms_per_pos;
    // F16 K: byte_count = elements * 2.
    let k = dev
        .alloc_buffer(nkv * cap * hd * 2, DType::F16, vec![nkv, cap, hd])
        .map_err(|e| anyhow!("hybrid F16 K L{layer_idx}: {e}"))?;
    // ADR-029 iter-20 H27: when HF2Q_FULL_F16_KV is set, V is F16 (2 bytes/elem)
    // and v_norms is a small dummy buffer (kernel ignores it when v_is_f16=1).
    // Otherwise: legacy TQ-HB packed V (1 byte/elem) + per-position F32 norms.
    let full_f16_v = std::env::var("HF2Q_FULL_F16_KV")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "on"))
        .unwrap_or(false);
    let (v_packed, v_norms) = if full_f16_v {
        let v_f16 = dev
            .alloc_buffer(nkv * cap * hd * 2, DType::F16, vec![nkv, cap, hd])
            .map_err(|e| anyhow!("hybrid F16 V L{layer_idx}: {e}"))?;
        // Dummy norms buffer (unused but kept for ABI compat with hybrid SDPA
        // signature; kernel's v_is_f16 FC=1 skips the read).
        let v_norms_dummy = dev
            .alloc_buffer(4, DType::F32, vec![1])
            .map_err(|e| anyhow!("hybrid V norms (dummy) L{layer_idx}: {e}"))?;
        (v_f16, v_norms_dummy)
    } else {
        let v_p = dev
            .alloc_buffer(nkv * cap * hd, DType::U8, vec![nkv, cap, hd])
            .map_err(|e| anyhow!("hybrid V packed L{layer_idx}: {e}"))?;
        let v_n = dev
            .alloc_buffer(
                norms_n * 4,
                DType::F32,
                if norms_per_pos == 1 {
                    vec![nkv, cap]
                } else {
                    vec![nkv, cap, norms_per_pos]
                },
            )
            .map_err(|e| anyhow!("hybrid V norms L{layer_idx}: {e}"))?;
        (v_p, v_n)
    };
    // ADR-030 iter-96: lazy-alloc the BF16 xlen cache only when env opted-in.
    // Saves ~55MB at gemma-4 when xlen mode disabled.
    let xlen_mode = std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() == Ok("1");
    let (bf16_xlen_k, bf16_xlen_v) = if xlen_mode {
        let bk = dev
            .alloc_buffer(nkv * cap * hd * 2, DType::BF16, vec![nkv, cap, hd])
            .map_err(|e| anyhow!("bf16 xlen K L{layer_idx}: {e}"))?;
        let bv = dev
            .alloc_buffer(nkv * cap * hd * 2, DType::BF16, vec![nkv, cap, hd])
            .map_err(|e| anyhow!("bf16 xlen V L{layer_idx}: {e}"))?;
        (Some(bk), Some(bv))
    } else {
        (None, None)
    };
    Ok(HybridKvBuffers {
        k,
        v_packed,
        v_norms,
        capacity: cap,
        is_sliding: is_ring,
        norms_per_pos,
        bf16_xlen_k,
        bf16_xlen_v,
    })
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A3b iter-1 — multi-seq variant of HybridKvBuffers.
//
// HybridKvBuffers is the PRODUCTION DEFAULT KV variant for Gemma 4
// since ADR-029 iter-13 (H10 falsification confirmed at A3a
// investigation — `HF2Q_HYBRID_KV` default-ON per
// `src/debug/investigation_env.rs:878`).  This sibling struct mirrors
// A3a's `MultiSeqHbKvBuffers` lift verbatim, adding an outermost
// `n_seqs` axis to every per-layer buffer + a per-seq cursor.
//
// Sibling-struct rationale (same as A3a — see A3a's block above):
//   * The brief forbids touching `forward_prefill.rs`,
//     `forward_prefill_batched.rs`, and `forward_gpu.rs` (constraint
//     #8).  Adding required fields to `HybridKvBuffers` would break
//     the 3 inline struct-literal alloc sites.  Sibling-struct
//     wraps the lift while keeping the legacy struct unchanged;
//     Phase B4c re-routes those sites through
//     `alloc_multi_seq_hybrid_kv_for_layer`.
//
// Scope (A3b iter-1):
//   * Per-buffer n_seqs OUTERMOST: K F16 `[n_seqs, nkv, cap, hd]`,
//     V packed U8 `[n_seqs, nkv, cap, hd]`, V norms F32
//     `[n_seqs, nkv, cap, norms_per_pos]`, optional BF16 xlen K/V
//     `[n_seqs, nkv, cap, hd]`.
//   * Per-seq cursor `seq_lens: Vec<u32>` of length n_seqs.
//   * MultiSeqKvCache impl mirroring A3a's MultiSeqHbKvBuffers.
//   * Honours both `HF2Q_FULL_F16_KV` (V also F16) and
//     `HF2Q_DFLASH_XLEN_SDPA` (extra BF16 K/V buffers) env gates,
//     read at alloc-time identically to the legacy
//     `alloc_hybrid_kv_for_layer`.
//
// DEFERRED to A3c (parallel to Qwen35 A2c per dossier R5):
//   * fork_seq cross-slot kernel dispatch.  iter-A3b-1 returns
//     `CapabilityUnsupported` per iter-2.5 M1 mantra-compliance —
//     same shape as A3a / Qwen35.
// ──────────────────────────────────────────────────────────────────────────

/// Multi-seq variant of [`HybridKvBuffers`] — ADR-040 Phase A3b iter-1.
///
/// Outermost axis on every buffer is `n_seqs`.  Buffer layouts:
/// - K (F16):              `[n_seqs, nkv_heads, capacity, head_dim]`
/// - V packed (U8 / F16):  `[n_seqs, nkv_heads, capacity, head_dim]`
/// - V norms (F32):        `[n_seqs, nkv_heads, capacity, norms_per_pos]`
///   (or 4-byte dummy when `HF2Q_FULL_F16_KV=1`; kernel reads are
///   gated on the v_is_f16 function constant — same as legacy)
/// - Optional BF16 xlen K/V (when `HF2Q_DFLASH_XLEN_SDPA=1`):
///   `[n_seqs, nkv_heads, capacity, head_dim]`
///
/// Per-seq cursor [`seq_lens`](Self::seq_lens) is `Vec<u32>` of length
/// `n_seqs` (parallel to A3a's `MultiSeqHbKvBuffers::seq_lens`).  A
/// per-slot byte offset for kernel writes is
/// `slot.0 * nkv * cap * hd * elem_bytes` (K/V packed) /
/// `slot.0 * nkv * cap * norms_per_pos * 4` (V norms) — Phase B4c
/// will thread this through the `alloc_hybrid_kv_for_layer` callers
/// via `MlxBuffer::slice_view`, same primitive Qwen35 B4a-cont uses.
///
/// Per dossier H7 verification, `is_sliding` is recorded on this
/// struct just like the legacy `HybridKvBuffers`; the per-slot
/// ring-wrap math stays within each slot's region by construction
/// (n_seqs is outermost, so slot N's `capacity` window is contiguous
/// and disjoint from slot M's).
pub struct MultiSeqHybridKvBuffers {
    /// Number of physical slots — the outermost axis on every buffer.
    /// Set at construction via [`alloc_multi_seq_hybrid_kv_for_layer`];
    /// once set, cannot change without reallocation.
    pub n_seqs: u32,
    /// Dense F16 K cache `[n_seqs, nkv_heads, capacity, head_dim]`.
    pub k: MlxBuffer,
    /// V packed indices `[n_seqs, nkv_heads, capacity, head_dim]` U8
    /// (or F16 when `HF2Q_FULL_F16_KV=1` was set at alloc-time).
    pub v_packed: MlxBuffer,
    /// V per-position norms `[n_seqs, nkv_heads, capacity, norms_per_pos]`
    /// F32 (or 4-byte dummy when `HF2Q_FULL_F16_KV=1`).
    pub v_norms: MlxBuffer,
    /// Cache capacity in positions (same as the legacy [`HybridKvBuffers`]).
    pub capacity: usize,
    /// True if ring-buffer (sliding) semantics.
    pub is_sliding: bool,
    /// Norms per position (1 for D=256, 2 for D=512).
    #[allow(dead_code)]
    pub norms_per_pos: usize,
    /// Optional BF16 K cache for DFlash xlen verify path (ADR-030
    /// iter-96).  Same layout as `k` but BF16 dtype; allocated only
    /// when `HF2Q_DFLASH_XLEN_SDPA=1` at construction time.
    pub bf16_xlen_k: Option<MlxBuffer>,
    /// Optional BF16 V cache, same semantics as `bf16_xlen_k`.
    pub bf16_xlen_v: Option<MlxBuffer>,
    /// Per-seq write cursor; `seq_lens[slot.0]` is the number of valid
    /// positions stored in slot `slot.0`.  `len() == n_seqs as usize`
    /// by construction (see [`alloc_multi_seq_hybrid_kv_for_layer`]).
    /// Mirrors A3a's `MultiSeqHbKvBuffers::seq_lens` discipline.
    pub seq_lens: Vec<u32>,
}

/// Prompt-boundary checkpoint for one Gemma agent slot.
///
/// Full-attention rows are append-only, so decoding after the prompt cannot
/// alter them. Sliding layers are rings: decode overwrites prompt rows after
/// wrap, so an exact retry with different generation policy must restore the
/// small fixed-size ring. This checkpoint therefore scales with
/// `sliding_window`, not with the model's 262K logical context.
pub struct GemmaHybridSlotAnchor {
    prompt_len: usize,
    layers: Vec<Option<GemmaHybridSlidingLayerAnchor>>,
}

struct GemmaHybridSlidingLayerAnchor {
    k: Vec<u8>,
    v_packed: Vec<u8>,
    v_norms: Option<Vec<u8>>,
    bf16_xlen_k: Option<Vec<u8>>,
    bf16_xlen_v: Option<Vec<u8>>,
}

impl GemmaHybridSlotAnchor {
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub fn total_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|layer| {
                layer.k.len()
                    + layer.v_packed.len()
                    + layer.v_norms.as_ref().map_or(0, Vec::len)
                    + layer.bf16_xlen_k.as_ref().map_or(0, Vec::len)
                    + layer.bf16_xlen_v.as_ref().map_or(0, Vec::len)
            })
            .sum()
    }

    /// Exact reclaimable heap ownership retained by this payload.
    ///
    /// The enclosing `AnchorStore` charges the top-level entry and committed
    /// vector control storage. This method charges every allocation owned by
    /// the opaque family payload, including spare `Vec` capacity and the
    /// per-layer option table.
    pub fn owned_bytes(&self) -> u64 {
        let layer_table = (self.layers.capacity() as u64)
            .saturating_mul(std::mem::size_of::<Option<GemmaHybridSlidingLayerAnchor>>() as u64);
        self.layers
            .iter()
            .flatten()
            .fold(layer_table, |sum, layer| {
                let vec_bytes = |bytes: &Vec<u8>| bytes.capacity() as u64;
                sum.saturating_add(vec_bytes(&layer.k))
                    .saturating_add(vec_bytes(&layer.v_packed))
                    .saturating_add(layer.v_norms.as_ref().map(vec_bytes).unwrap_or(0))
                    .saturating_add(layer.bf16_xlen_k.as_ref().map(vec_bytes).unwrap_or(0))
                    .saturating_add(layer.bf16_xlen_v.as_ref().map(vec_bytes).unwrap_or(0))
            })
    }

    #[cfg(test)]
    pub(crate) fn synthetic(prompt_len: usize, layer_payload_bytes: &[usize]) -> Self {
        let mut layers = Vec::with_capacity(layer_payload_bytes.len());
        for &bytes in layer_payload_bytes {
            layers.push(Some(GemmaHybridSlidingLayerAnchor {
                k: vec![0; bytes],
                v_packed: Vec::new(),
                v_norms: None,
                bf16_xlen_k: None,
                bf16_xlen_v: None,
            }));
        }
        Self { prompt_len, layers }
    }
}

fn gemma4_copy_slot_region_out(
    buf: &MlxBuffer,
    slot_idx: usize,
    n_seqs: usize,
    name: &str,
) -> Result<Vec<u8>> {
    anyhow::ensure!(n_seqs > 0, "{name}: n_seqs must be positive");
    anyhow::ensure!(
        slot_idx < n_seqs,
        "{name}: slot {slot_idx} outside {n_seqs}"
    );
    let bytes = buf
        .as_slice::<u8>()
        .map_err(|e| anyhow!("{name}: as_slice<u8>: {e}"))?;
    anyhow::ensure!(
        bytes.len() % n_seqs == 0,
        "{name}: byte length {} not divisible by n_seqs={n_seqs}",
        bytes.len()
    );
    let per_slot = bytes.len() / n_seqs;
    let start = slot_idx * per_slot;
    Ok(bytes[start..start + per_slot].to_vec())
}

fn gemma4_copy_slot_region_in(
    src: &[u8],
    dst: &mut MlxBuffer,
    slot_idx: usize,
    n_seqs: usize,
    name: &str,
) -> Result<()> {
    anyhow::ensure!(n_seqs > 0, "{name}: n_seqs must be positive");
    anyhow::ensure!(
        slot_idx < n_seqs,
        "{name}: slot {slot_idx} outside {n_seqs}"
    );
    let bytes = dst
        .as_mut_slice::<u8>()
        .map_err(|e| anyhow!("{name}: as_mut_slice<u8>: {e}"))?;
    anyhow::ensure!(
        bytes.len() % n_seqs == 0,
        "{name}: byte length {} not divisible by n_seqs={n_seqs}",
        bytes.len()
    );
    let per_slot = bytes.len() / n_seqs;
    anyhow::ensure!(
        src.len() == per_slot,
        "{name}: checkpoint bytes {} != slot bytes {per_slot}",
        src.len()
    );
    let start = slot_idx * per_slot;
    bytes[start..start + per_slot].copy_from_slice(src);
    Ok(())
}

fn gemma4_preflight_slot_region_in(
    src: &[u8],
    dst: &MlxBuffer,
    slot_idx: usize,
    n_seqs: usize,
    name: &str,
) -> Result<()> {
    anyhow::ensure!(n_seqs > 0, "{name}: n_seqs must be positive");
    anyhow::ensure!(
        slot_idx < n_seqs,
        "{name}: slot {slot_idx} outside {n_seqs}"
    );
    let bytes = dst
        .as_slice::<u8>()
        .map_err(|e| anyhow!("{name}: as_slice<u8>: {e}"))?;
    anyhow::ensure!(
        bytes.len() % n_seqs == 0,
        "{name}: byte length {} not divisible by n_seqs={n_seqs}",
        bytes.len()
    );
    let per_slot = bytes.len() / n_seqs;
    anyhow::ensure!(
        src.len() == per_slot,
        "{name}: checkpoint bytes {} != slot bytes {per_slot}",
        src.len()
    );
    Ok(())
}

/// Validate every layer, optional buffer, and destination span before the
/// restore mutates the first byte or cursor. A failed preflight therefore
/// leaves the physical slot untouched and lets the caller hard-reset and
/// invalidate the whole checkpoint lineage deterministically.
pub(crate) fn preflight_gemma_hybrid_slot_anchor_restore(
    scaffold: &[MultiSeqHybridKvBuffers],
    slot: crate::serve::multi_seq_kv::SlotId,
    anchor: &GemmaHybridSlotAnchor,
    resume_len: usize,
) -> Result<()> {
    anyhow::ensure!(
        resume_len <= anchor.prompt_len,
        "Gemma slot anchor resume length {resume_len} exceeds checkpoint {}",
        anchor.prompt_len
    );
    anyhow::ensure!(
        scaffold.len() == anchor.layers.len(),
        "Gemma slot anchor layer count mismatch: live={} saved={}",
        scaffold.len(),
        anchor.layers.len()
    );
    for (layer_idx, (buf, saved)) in scaffold.iter().zip(&anchor.layers).enumerate() {
        let slot_idx = slot.0 as usize;
        let n_seqs = buf.n_seqs as usize;
        anyhow::ensure!(
            slot_idx < n_seqs && slot_idx < buf.seq_lens.len(),
            "Gemma slot anchor restore L{layer_idx}: slot {} outside n_seqs={n_seqs} / cursors={}",
            slot.0,
            buf.seq_lens.len()
        );
        match (buf.is_sliding, saved) {
            (false, None) => {}
            (true, Some(saved)) => {
                gemma4_preflight_slot_region_in(
                    &saved.k,
                    &buf.k,
                    slot_idx,
                    n_seqs,
                    &format!("Gemma slot anchor restore L{layer_idx} K"),
                )?;
                gemma4_preflight_slot_region_in(
                    &saved.v_packed,
                    &buf.v_packed,
                    slot_idx,
                    n_seqs,
                    &format!("Gemma slot anchor restore L{layer_idx} V"),
                )?;
                match (&saved.v_norms, buf.v_norms.byte_len() == 4) {
                    (Some(bytes), false) => gemma4_preflight_slot_region_in(
                        bytes,
                        &buf.v_norms,
                        slot_idx,
                        n_seqs,
                        &format!("Gemma slot anchor restore L{layer_idx} V norms"),
                    )?,
                    (None, true) => {}
                    _ => anyhow::bail!(
                        "Gemma slot anchor restore L{layer_idx}: V-norm layout changed"
                    ),
                }
                match (&saved.bf16_xlen_k, buf.bf16_xlen_k.as_ref()) {
                    (Some(bytes), Some(dst)) => gemma4_preflight_slot_region_in(
                        bytes,
                        dst,
                        slot_idx,
                        n_seqs,
                        &format!("Gemma slot anchor restore L{layer_idx} xlen K"),
                    )?,
                    (None, None) => {}
                    _ => anyhow::bail!(
                        "Gemma slot anchor restore L{layer_idx}: xlen K layout changed"
                    ),
                }
                match (&saved.bf16_xlen_v, buf.bf16_xlen_v.as_ref()) {
                    (Some(bytes), Some(dst)) => gemma4_preflight_slot_region_in(
                        bytes,
                        dst,
                        slot_idx,
                        n_seqs,
                        &format!("Gemma slot anchor restore L{layer_idx} xlen V"),
                    )?,
                    (None, None) => {}
                    _ => anyhow::bail!(
                        "Gemma slot anchor restore L{layer_idx}: xlen V layout changed"
                    ),
                }
            }
            _ => {
                anyhow::bail!("Gemma slot anchor restore L{layer_idx}: sliding/full layout changed")
            }
        }
    }
    Ok(())
}

/// Capture one slot immediately after prompt prefill and before decode.
pub fn snapshot_gemma_hybrid_slot_anchor(
    scaffold: &[MultiSeqHybridKvBuffers],
    slot: crate::serve::multi_seq_kv::SlotId,
    prompt_len: usize,
) -> Result<GemmaHybridSlotAnchor> {
    anyhow::ensure!(
        prompt_len > 0,
        "Gemma slot anchor requires a non-empty prompt"
    );
    let mut layers = Vec::with_capacity(scaffold.len());
    for (layer_idx, buf) in scaffold.iter().enumerate() {
        let slot_idx = slot.0 as usize;
        let n_seqs = buf.n_seqs as usize;
        anyhow::ensure!(
            slot_idx < n_seqs,
            "Gemma slot anchor L{layer_idx}: slot {} outside n_seqs={n_seqs}",
            slot.0
        );
        if !buf.is_sliding {
            layers.push(None);
            continue;
        }
        let k = gemma4_copy_slot_region_out(
            &buf.k,
            slot_idx,
            n_seqs,
            &format!("Gemma slot anchor L{layer_idx} K"),
        )?;
        let v_packed = gemma4_copy_slot_region_out(
            &buf.v_packed,
            slot_idx,
            n_seqs,
            &format!("Gemma slot anchor L{layer_idx} V"),
        )?;
        let v_norms = if buf.v_norms.byte_len() == 4 {
            None
        } else {
            Some(gemma4_copy_slot_region_out(
                &buf.v_norms,
                slot_idx,
                n_seqs,
                &format!("Gemma slot anchor L{layer_idx} V norms"),
            )?)
        };
        let bf16_xlen_k = buf
            .bf16_xlen_k
            .as_ref()
            .map(|buffer| {
                gemma4_copy_slot_region_out(
                    buffer,
                    slot_idx,
                    n_seqs,
                    &format!("Gemma slot anchor L{layer_idx} xlen K"),
                )
            })
            .transpose()?;
        let bf16_xlen_v = buf
            .bf16_xlen_v
            .as_ref()
            .map(|buffer| {
                gemma4_copy_slot_region_out(
                    buffer,
                    slot_idx,
                    n_seqs,
                    &format!("Gemma slot anchor L{layer_idx} xlen V"),
                )
            })
            .transpose()?;
        layers.push(Some(GemmaHybridSlidingLayerAnchor {
            k,
            v_packed,
            v_norms,
            bf16_xlen_k,
            bf16_xlen_v,
        }));
    }
    Ok(GemmaHybridSlotAnchor { prompt_len, layers })
}

/// Restore one idle slot to a captured prompt boundary.
pub fn restore_gemma_hybrid_slot_anchor(
    scaffold: &mut [MultiSeqHybridKvBuffers],
    slot: crate::serve::multi_seq_kv::SlotId,
    anchor: &GemmaHybridSlotAnchor,
    resume_len: usize,
) -> Result<()> {
    preflight_gemma_hybrid_slot_anchor_restore(scaffold, slot, anchor, resume_len)?;
    for (layer_idx, (buf, saved)) in scaffold.iter_mut().zip(&anchor.layers).enumerate() {
        let slot_idx = slot.0 as usize;
        let n_seqs = buf.n_seqs as usize;
        anyhow::ensure!(
            slot_idx < n_seqs,
            "Gemma slot anchor restore L{layer_idx}: slot {} outside n_seqs={n_seqs}",
            slot.0
        );
        match (buf.is_sliding, saved) {
            (false, None) => {}
            (true, Some(saved)) => {
                gemma4_copy_slot_region_in(
                    &saved.k,
                    &mut buf.k,
                    slot_idx,
                    n_seqs,
                    &format!("Gemma slot anchor restore L{layer_idx} K"),
                )?;
                gemma4_copy_slot_region_in(
                    &saved.v_packed,
                    &mut buf.v_packed,
                    slot_idx,
                    n_seqs,
                    &format!("Gemma slot anchor restore L{layer_idx} V"),
                )?;
                match (&saved.v_norms, buf.v_norms.byte_len() == 4) {
                    (Some(bytes), false) => gemma4_copy_slot_region_in(
                        bytes,
                        &mut buf.v_norms,
                        slot_idx,
                        n_seqs,
                        &format!("Gemma slot anchor restore L{layer_idx} V norms"),
                    )?,
                    (None, true) => {}
                    _ => anyhow::bail!(
                        "Gemma slot anchor restore L{layer_idx}: V-norm layout changed"
                    ),
                }
                match (&saved.bf16_xlen_k, buf.bf16_xlen_k.as_mut()) {
                    (Some(bytes), Some(dst)) => gemma4_copy_slot_region_in(
                        bytes,
                        dst,
                        slot_idx,
                        n_seqs,
                        &format!("Gemma slot anchor restore L{layer_idx} xlen K"),
                    )?,
                    (None, None) => {}
                    _ => anyhow::bail!(
                        "Gemma slot anchor restore L{layer_idx}: xlen K layout changed"
                    ),
                }
                match (&saved.bf16_xlen_v, buf.bf16_xlen_v.as_mut()) {
                    (Some(bytes), Some(dst)) => gemma4_copy_slot_region_in(
                        bytes,
                        dst,
                        slot_idx,
                        n_seqs,
                        &format!("Gemma slot anchor restore L{layer_idx} xlen V"),
                    )?,
                    (None, None) => {}
                    _ => anyhow::bail!(
                        "Gemma slot anchor restore L{layer_idx}: xlen V layout changed"
                    ),
                }
            }
            _ => {
                anyhow::bail!("Gemma slot anchor restore L{layer_idx}: sliding/full layout changed")
            }
        }
        // A later native chat turn can rewrite the prompt's trailing
        // generation cue.  The restored ring is still exact through the LCP;
        // rewind the logical cursor so suffix prefill overwrites the changed
        // cue instead of treating checkpoint-tail rows as live future KV.
        buf.seq_lens[slot_idx] = resume_len.min(u32::MAX as usize) as u32;
    }
    Ok(())
}

impl crate::serve::kv_persist::lcp_registry::ByteSized for MultiSeqHybridKvBuffers {
    /// Exact byte count: F16/F32 K + (U8|F16) V + (F32|dummy) V-norms +
    /// optional BF16 xlen K + optional BF16 xlen V.  Used by the
    /// LcpRegistry byte budget identically to `HybridKvBuffers::byte_len`
    /// — the lift to N slots scales every buffer by N at alloc-time, so
    /// `byte_len()` automatically reports the per-slot totals × N.
    fn byte_len(&self) -> u64 {
        let mut sum =
            (self.k.byte_len() + self.v_packed.byte_len() + self.v_norms.byte_len()) as u64;
        if let Some(ref bk) = self.bf16_xlen_k {
            sum += bk.byte_len() as u64;
        }
        if let Some(ref bv) = self.bf16_xlen_v {
            sum += bv.byte_len() as u64;
        }
        sum
    }
}

/// ADR-040 Phase A3b iter-1 — unified [`MultiSeqHybridKvBuffers`] allocator.
///
/// Mirrors [`alloc_hybrid_kv_for_layer`] (same file, line 649) in
/// signature shape; the extra `n_seqs` parameter is the lift this
/// helper introduces.  Honours the SAME env gates the legacy helper
/// does (`HF2Q_FULL_F16_KV`, `HF2Q_DFLASH_XLEN_SDPA`) so an A3b lift
/// of the 3 alloc-site callers at Phase B4c is purely a "drop-in
/// substitution + n_seqs argument added" diff — no behavioural change
/// per-slot.
///
/// At `n_seqs=1` the byte counts are byte-equivalent to the legacy
/// allocator's output (H11 hypothesis); the only observable shape
/// difference is the leading `1` dimension on every buffer.
///
/// # Errors
///
/// Returns `Err` for `n_seqs == 0`, `nkv == 0`, `hd == 0`, or `cap == 0`
/// — buffer alloc would otherwise underflow the kernel's shape
/// preconditions.  Mirrors A3a's `alloc_hb_kv_for_layer` pre-flight.
pub fn alloc_multi_seq_hybrid_kv_for_layer(
    dev: &MlxDevice,
    layer_idx: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    is_ring: bool,
    n_seqs: u32,
) -> Result<MultiSeqHybridKvBuffers> {
    if n_seqs == 0 {
        return Err(anyhow!(
            "alloc_multi_seq_hybrid_kv_for_layer L{layer_idx}: n_seqs must be > 0"
        ));
    }
    if nkv == 0 || hd == 0 || cap == 0 {
        return Err(anyhow!(
            "alloc_multi_seq_hybrid_kv_for_layer L{layer_idx}: nkv/hd/cap must be \
             > 0 (got nkv={nkv}, hd={hd}, cap={cap})"
        ));
    }
    let norms_per_pos = (hd / 256).max(1);
    let n = n_seqs as usize;

    // F16 K: `[n_seqs, nkv, cap, hd]`, 2 bytes/elem.  n_seqs is
    // OUTERMOST so per-slot byte offset = `slot.0 * (nkv*cap*hd*2)`
    // is a contiguous slab Phase B4c can address via
    // `MlxBuffer::slice_view`.
    let k_elems = n * nkv * cap * hd;
    let k_bytes = k_elems * 2;
    let k = alloc_multi_seq_kv_storage(dev, k_bytes, DType::F16, vec![n, nkv, cap, hd], is_ring)
        .map_err(|e| anyhow!("multi-seq hybrid F16 K L{layer_idx}: {e}"))?;

    // V: honour `HF2Q_FULL_F16_KV` exactly like the legacy allocator
    // — when set, V is F16 (2 bytes/elem) and v_norms is a 4-byte
    // dummy.  Otherwise legacy TQ-HB packed U8 + F32 per-position
    // norms.  The env-read is at alloc-time so the lift inherits
    // the legacy allocator's behaviour byte-for-byte at n_seqs=1.
    let full_f16_v = std::env::var("HF2Q_FULL_F16_KV")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "on"))
        .unwrap_or(false);
    let (v_packed, v_norms) = if full_f16_v {
        let v_elems = n * nkv * cap * hd;
        let v_bytes = v_elems * 2;
        let v_f16 =
            alloc_multi_seq_kv_storage(dev, v_bytes, DType::F16, vec![n, nkv, cap, hd], is_ring)
                .map_err(|e| anyhow!("multi-seq hybrid F16 V L{layer_idx}: {e}"))?;
        // Dummy norms buffer — same 4-byte shape as the legacy
        // allocator emits when full_f16_v is set; kernel's
        // v_is_f16 function constant skips the read entirely.  The
        // dummy is shared across slots (no per-slot dimension)
        // because it carries no data; a per-slot dummy would just
        // waste 4*(N-1) bytes for zero kernel benefit.
        let v_norms_dummy = dev
            .alloc_buffer(4, DType::F32, vec![1])
            .map_err(|e| anyhow!("multi-seq hybrid V norms (dummy) L{layer_idx}: {e}"))?;
        (v_f16, v_norms_dummy)
    } else {
        let v_packed_elems = n * nkv * cap * hd;
        let v_packed_bytes = v_packed_elems; // U8 = 1 byte/elem
        let v_p = alloc_multi_seq_kv_storage(
            dev,
            v_packed_bytes,
            DType::U8,
            vec![n, nkv, cap, hd],
            is_ring,
        )
        .map_err(|e| anyhow!("multi-seq hybrid V packed L{layer_idx}: {e}"))?;
        let v_norms_elems = n * nkv * cap * norms_per_pos;
        let v_norms_bytes = v_norms_elems * std::mem::size_of::<f32>();
        let v_n = alloc_multi_seq_kv_storage(
            dev,
            v_norms_bytes,
            DType::F32,
            vec![n, nkv, cap, norms_per_pos],
            is_ring,
        )
        .map_err(|e| anyhow!("multi-seq hybrid V norms L{layer_idx}: {e}"))?;
        (v_p, v_n)
    };

    // Optional BF16 xlen K/V — ADR-030 iter-96.  Lazy at alloc-time
    // mirrors the legacy allocator's behaviour.  ~110 MB/slot at
    // production Gemma 4 shapes when the env gate is OFF, so
    // None-by-default matters even more at N>1.
    let xlen_mode = std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() == Ok("1");
    let (bf16_xlen_k, bf16_xlen_v) = if xlen_mode {
        let xlen_elems = n * nkv * cap * hd;
        let xlen_bytes = xlen_elems * 2;
        let bk = alloc_multi_seq_kv_storage(
            dev,
            xlen_bytes,
            DType::BF16,
            vec![n, nkv, cap, hd],
            is_ring,
        )
        .map_err(|e| anyhow!("multi-seq hybrid bf16 xlen K L{layer_idx}: {e}"))?;
        let bv = alloc_multi_seq_kv_storage(
            dev,
            xlen_bytes,
            DType::BF16,
            vec![n, nkv, cap, hd],
            is_ring,
        )
        .map_err(|e| anyhow!("multi-seq hybrid bf16 xlen V L{layer_idx}: {e}"))?;
        (Some(bk), Some(bv))
    } else {
        (None, None)
    };

    Ok(MultiSeqHybridKvBuffers {
        n_seqs,
        k,
        v_packed,
        v_norms,
        capacity: cap,
        is_sliding: is_ring,
        norms_per_pos,
        bf16_xlen_k,
        bf16_xlen_v,
        seq_lens: vec![0u32; n],
    })
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A3b iter-1 — MultiSeqKvCache impl for MultiSeqHybridKvBuffers.
//
// Mirrors A3a's MultiSeqHbKvBuffers impl in structure (bounds-first
// per iter-1.5 cfa-finding-F5; fork_seq returns
// `CapabilityUnsupported` per iter-2.5 M1) and in invariants
// (per-slot cursor isolation; drop_seq does NOT zero the underlying
// buffer bytes — Phase B4c reuses the slot's region on next admission).
//
// Phase A3b iter-1 scope: per-slot CURSOR bookkeeping only.  GPU
// buffer content writes land via the `alloc_hybrid_kv_for_layer`
// callers + `dispatch_hadamard_quantize_kv_*` dispatchers at Phase B4c.
// ──────────────────────────────────────────────────────────────────────────

impl crate::serve::multi_seq_kv::MultiSeqKvCache for MultiSeqHybridKvBuffers {
    fn layout(&self) -> crate::serve::multi_seq_kv::MultiSeqLayout {
        crate::serve::multi_seq_kv::MultiSeqLayout::SeparateSlots
    }

    fn slot_count(&self) -> u32 {
        // `MultiSeqHybridKvBuffers::n_seqs` is already `u32`; no cast.
        self.n_seqs
    }

    fn seq_len(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<u32, crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST (iter-1.5 cfa-finding-F5 ordering).
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only — MultiSeqHybridKvBuffers
        //    does not expose Paged.
        // 3. Return the per-seq cursor directly; `seq_lens.len() ==
        //    n_seqs` by construction.  Mirrors A3a's
        //    MultiSeqHbKvBuffers: per-layer struct, single cursor per
        //    slot per buffer.
        Ok(self.seq_lens[slot.0 as usize])
    }

    fn append_for_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        n_tokens: u32,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only.
        // 3. Budget: SeparateSlots cannot SlotOom on append (buffers
        //    pre-allocated; cursor protected by saturating_add).
        //
        // ADR-040 Phase A3b iter-1 scope: bump the per-seq cursor.
        // The underlying K / V packed / norms (and optional xlen
        // bf16) bytes for slot `slot.0` are written by the kernel
        // dispatcher at Phase B4c via `MlxBuffer::slice_view(...)`,
        // identically to the A3a MultiSeqHbKvBuffers pattern.
        let cur = &mut self.seq_lens[slot.0 as usize];
        *cur = cur.saturating_add(n_tokens);
        Ok(())
    }

    fn drop_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only — no LayoutNotSupported.
        // 3. Budget: drop is a pure release; SlotOom unreachable.
        //
        // Cursor-only reset.  The underlying K / V (packed | F16) /
        // norms / optional xlen bytes are NOT zeroed; the next
        // `append_for_seq` into this slot overwrites them via the
        // kernel dispatcher at Phase B4c.  Matches A3a's discipline
        // (gemma4_hb_kv_drop_does_not_zero_k_packed_buffer pin above).
        self.seq_lens[slot.0 as usize] = 0;
        Ok(())
    }

    fn fork_seq(
        &mut self,
        src: crate::serve::multi_seq_kv::SlotId,
        dst: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds — src FIRST per iter-1.5 cfa-finding-F5.
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
        // 3. Same-slot fork is a no-op per trait spec.
        if src == dst {
            return Ok(());
        }
        // ──────────────────────────────────────────────────────────────
        // ADR-040 Phase A3c (2026-05-30) — REAL cross-slot fork.
        //
        // Replaces the prior `CapabilityUnsupported` typed-deferral
        // with same-buffer cross-region memcpy on the K (F16) +
        // V packed (U8|F16) + V norms (F32|dummy) + optional BF16
        // xlen K/V buffers per `alloc_multi_seq_hybrid_kv_for_layer`
        // at `kv_cache.rs:942+`.  n_seqs OUTERMOST on every buffer
        // EXCEPT the dummy `v_norms_dummy` (4-byte total when
        // `HF2Q_FULL_F16_KV=1`) — that buffer is shared across
        // slots by construction (line 996-998) and is excluded
        // from per-slot copy via the `total_bytes % n_seqs == 0`
        // guard (when the dummy is in use total_bytes=4 and
        // n_seqs ≥ 1; if n_seqs ∤ 4 the guard returns Err — but
        // the dummy IS shared so we recognise the case explicitly
        // by total_bytes < n_seqs and skip).  See helper docs.
        //
        // Cursor copy: `seq_lens[dst] = seq_lens[src]`.
        // ──────────────────────────────────────────────────────────────
        let src_idx = src.0 as usize;
        let dst_idx = dst.0 as usize;
        let n_seqs = self.n_seqs as usize;
        let live_tokens = self.seq_lens[src_idx] as usize;
        gemma4_copy_buffer_slot_prefix(&mut self.k, src_idx, dst_idx, n_seqs, live_tokens)
            .map_err(
                |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                    capability: gemma4_leak_static_str(format!(
                        "fork_seq: MultiSeqHybridKvBuffers k copy failed ({e})"
                    )),
                },
            )?;
        gemma4_copy_buffer_slot_prefix(&mut self.v_packed, src_idx, dst_idx, n_seqs, live_tokens)
            .map_err(
            |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: gemma4_leak_static_str(format!(
                    "fork_seq: MultiSeqHybridKvBuffers v_packed copy failed ({e})"
                )),
            },
        )?;
        // v_norms: skip if it's the 4-byte dummy (HF2Q_FULL_F16_KV=1
        // path — alloc'd at `alloc_multi_seq_hybrid_kv_for_layer:996-998`
        // with byte_len=4 shared across slots).  Detection: byte_len
        // < n_seqs ⇒ structurally cannot be a per-slot buffer.
        if self.v_norms.byte_len() >= n_seqs {
            gemma4_copy_buffer_slot_prefix(
                &mut self.v_norms,
                src_idx,
                dst_idx,
                n_seqs,
                live_tokens,
            )
            .map_err(|e| {
                crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                    capability: gemma4_leak_static_str(format!(
                        "fork_seq: MultiSeqHybridKvBuffers v_norms copy failed ({e})"
                    )),
                }
            })?;
        }
        // Optional BF16 xlen K/V (HF2Q_DFLASH_XLEN_SDPA=1 path).  Same
        // n_seqs outermost layout per `alloc_multi_seq_hybrid_kv_for_layer:1019-1031`.
        if let Some(ref mut bk) = self.bf16_xlen_k {
            gemma4_copy_buffer_slot_prefix(bk, src_idx, dst_idx, n_seqs, live_tokens).map_err(
                |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                    capability: gemma4_leak_static_str(format!(
                        "fork_seq: MultiSeqHybridKvBuffers bf16_xlen_k copy failed ({e})"
                    )),
                },
            )?;
        }
        if let Some(ref mut bv) = self.bf16_xlen_v {
            gemma4_copy_buffer_slot_prefix(bv, src_idx, dst_idx, n_seqs, live_tokens).map_err(
                |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                    capability: gemma4_leak_static_str(format!(
                        "fork_seq: MultiSeqHybridKvBuffers bf16_xlen_v copy failed ({e})"
                    )),
                },
            )?;
        }
        // Cursor copy AFTER buffer copy.
        self.seq_lens[dst_idx] = self.seq_lens[src_idx];
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 iter-B4c-kernel iter-1 — per-slot reset primitive for
// MultiSeqHybridKvBuffers (Gemma 4 hybrid variant; mirrors A3a sibling).
// ──────────────────────────────────────────────────────────────────────────

impl MultiSeqHybridKvBuffers {
    /// **ADR-040 iter-B4c-kernel iter-1** (2026-05-30) — per-slot
    /// reset for the persistent multi-seq `MultiSeqHybridKvBuffers`
    /// (Gemma 4 hybrid K-F16 + TQ-HB-V variant).
    ///
    /// Sibling of [`MultiSeqHbKvBuffers::reset_for_slot`] for the
    /// `HF2Q_FULL_F16_KV=1` / `HF2Q_DFLASH_XLEN_SDPA=1` codepath that
    /// engages `MultiSeqHybridKvBuffers`.  Used by
    /// `engine::generate_gemma4_once_slot_aware` to clear a slot's
    /// state at request entry + exit so the persistent per-layer
    /// `MultiSeqHybridKvBuffers` is request-isolated within the slot.
    ///
    /// **Layout proof**:
    /// - **seq_lens**: `Vec<u32>` of length `n_seqs`. Per-slot reset →
    ///   set `seq_lens[slot_idx] = 0`; other slots untouched.  Same
    ///   load-bearing cursor as the A3a sibling.
    /// - **k (F16, `[n_seqs, nkv, capacity, head_dim]`)**: NOT zeroed
    ///   — same cursor-masked discipline as A3a.
    /// - **v_packed / v_norms**: NOT zeroed; cursor-masked.
    /// - **bf16_xlen_k / bf16_xlen_v (optional BF16)**: NOT zeroed —
    ///   the xlen verify path reads only up to `seq_lens[slot_idx]`
    ///   positions per ADR-030 iter-96, matching the F16 K + packed V
    ///   discipline.
    ///
    /// # Errors
    ///
    /// - `slot.0 >= self.n_seqs` (bounds-first per A2b iter-1.5
    ///   cfa-finding-F5 ordering).
    ///
    /// # Per-slot byte-equivalence pin
    ///
    /// At `slot = SlotId(0)` AND `n_seqs == 1` this is byte-equivalent
    /// to setting `seq_lens[0] = 0` directly (the existing `drop_seq`
    /// shape).  Sibling pin to the A3a per-slot byte-equivalence.
    pub fn reset_for_slot(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // Bounds-first per A2b §6.1.23 iter-1.5 cfa-finding-F5.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // Reset the per-slot cursor.  K (F16) / V (packed | F16) / V
        // norms (F32 | dummy) / optional xlen bf16 K/V bytes are NOT
        // zeroed (cursor-masked read path — see layout proof above;
        // matches `drop_seq` invariant).
        self.seq_lens[slot.0 as usize] = 0;
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A3b iter-2 — multi-seq variant of DenseKvBuffers.
//
// DenseKvBuffers is the DEV/DEBUG KV variant for Gemma 4 reachable via
// `HF2Q_USE_DENSE=1` (off-default; the production-default path is
// HybridKvBuffers since ADR-029 iter-13 per H10 falsification).  This
// sibling struct mirrors A3b iter-1's `MultiSeqHybridKvBuffers` lift
// verbatim, adding an outermost `n_seqs` axis to the K and V buffers
// + a per-seq cursor — the same shape A3a's HbKvBuffers and A3b iter-1's
// HybridKvBuffers use.
//
// Sibling-struct rationale (same as A3a / A3b iter-1):
//   * The legacy `DenseKvBuffers` struct is used by 3 production
//     alloc sites (`forward_prefill.rs:705`,
//     `forward_prefill_batched.rs:367`, `engine.rs:6836`) at implicit
//     `n_seqs=1` via struct-literal init.  Adding required fields
//     would break those sites; sibling-struct keeps the legacy
//     unchanged + adds the multi-seq shape additively.  Phase B4c
//     re-routes the alloc sites through
//     `alloc_multi_seq_dense_kv_for_layer` when the kernel-side
//     slot-offset wiring lands.
//
// Scope (A3b iter-2):
//   * Per-buffer n_seqs OUTERMOST: K `[n_seqs, nkv, cap, hd]`,
//     V `[n_seqs, nkv, cap, hd]` — dtype is per-call (F32 default,
//     F16 when `HF2Q_F16_KV=1`).
//   * Per-seq cursor `seq_lens: Vec<u32>` of length n_seqs.
//   * MultiSeqKvCache impl mirroring A3b iter-1's MultiSeqHybridKvBuffers
//     (bounds-first per iter-1.5 cfa-finding-F5; fork cross-slot →
//     CapabilityUnsupported A3c deferral).
//   * `reset_for_slot` inherent method mirroring the A3a/A3b iter-1
//     siblings for the iter-B4c-kernel iter-1 reset-on-entry/exit
//     discipline.
//
// At n_seqs=1 the byte counts are byte-equivalent to the legacy
// `DenseKvBuffers` per-layer K + V allocs (H148 hypothesis).  The
// only observable shape difference is the leading `1` dimension on
// every buffer.
//
// DEFERRED to A3c (parallel to Qwen35 A2c per dossier R5):
//   * fork_seq cross-slot kernel dispatch.  iter-A3b-2 returns
//     `CapabilityUnsupported` per iter-2.5 M1 mantra-compliance —
//     same shape as A3a / A3b iter-1 / Qwen35.
// ──────────────────────────────────────────────────────────────────────────

/// Multi-seq variant of [`DenseKvBuffers`] — ADR-040 Phase A3b iter-2.
///
/// Outermost axis on every buffer is `n_seqs`.  Buffer layouts:
/// - K: `[n_seqs, nkv_heads, capacity, head_dim]` of [`Self::dtype`]
///   (F32 default, F16 when `HF2Q_F16_KV=1` was set at alloc-time).
/// - V: `[n_seqs, nkv_heads, capacity, head_dim]` of [`Self::dtype`]
///   (same dtype as K — ADR-017 Phase E.a iter-3.5a invariant).
///
/// Per-seq cursor [`seq_lens`](Self::seq_lens) is `Vec<u32>` of length
/// `n_seqs` (parallel to A3b iter-1's `MultiSeqHybridKvBuffers::seq_lens`).
/// A per-slot byte offset for kernel writes is
/// `slot.0 * nkv * cap * hd * dtype.size_of()` (K) / `slot.0 * nkv *
/// cap * hd * dtype.size_of()` (V) — Phase B4c will thread this
/// through the `alloc_multi_seq_dense_kv_for_layer` callers via
/// `MlxBuffer::slice_view`, same primitive Qwen35 B4a-cont uses.
///
/// `is_sliding` is recorded on this struct just like the legacy
/// `DenseKvBuffers`; the per-slot ring-wrap math stays within each
/// slot's region by construction (n_seqs is outermost, so slot N's
/// `capacity` window is contiguous and disjoint from slot M's).
pub struct MultiSeqDenseKvBuffers {
    /// Number of physical slots — the outermost axis on every buffer.
    /// Set at construction via [`alloc_multi_seq_dense_kv_for_layer`];
    /// once set, cannot change without reallocation.
    pub n_seqs: u32,
    /// Dense K cache `[n_seqs, nkv_heads, capacity, head_dim]` of
    /// [`Self::dtype`].
    pub k: MlxBuffer,
    /// Dense V cache `[n_seqs, nkv_heads, capacity, head_dim]` of
    /// [`Self::dtype`] (same dtype as K per ADR-017 Phase E.a
    /// iter-3.5a invariant).
    pub v: MlxBuffer,
    /// Cache capacity in positions (same as the legacy [`DenseKvBuffers`]).
    pub capacity: usize,
    /// True if ring-buffer (sliding) semantics.
    pub is_sliding: bool,
    /// KV element dtype.  Mirrors `DenseKvBuffers::dtype` (ADR-017
    /// Phase E.a iter-3.5a invariant); same dtype applies to both K
    /// and V (no mixed-dtype layout).
    pub dtype: DType,
    /// Per-seq write cursor; `seq_lens[slot.0]` is the number of valid
    /// positions stored in slot `slot.0`.  `len() == n_seqs as usize`
    /// by construction (see [`alloc_multi_seq_dense_kv_for_layer`]).
    /// Mirrors A3b iter-1's `MultiSeqHybridKvBuffers::seq_lens` discipline.
    pub seq_lens: Vec<u32>,
}

impl crate::serve::kv_persist::lcp_registry::ByteSized for MultiSeqDenseKvBuffers {
    /// Exact byte count: K + V.  Used by the LcpRegistry byte budget
    /// identically to `DenseKvBuffers::byte_len` — the lift to N slots
    /// scales every buffer by N at alloc-time, so `byte_len()`
    /// automatically reports the per-slot totals × N.
    fn byte_len(&self) -> u64 {
        (self.k.byte_len() + self.v.byte_len()) as u64
    }
}

/// ADR-040 Phase A3b iter-2 — unified [`MultiSeqDenseKvBuffers`] allocator.
///
/// Mirrors A3b iter-1's [`alloc_multi_seq_hybrid_kv_for_layer`]
/// (same file) in signature shape; the `dtype` parameter is the
/// per-call invariant the legacy 3 production sites carry inline
/// (`forward_prefill.rs:701-703` passes `kv_dtype` from
/// `INVESTIGATION_ENV.f16_kv`).
///
/// At `n_seqs=1` the byte counts are byte-equivalent to the legacy
/// inline-alloc sites' K + V allocs (H148 hypothesis); the only
/// observable shape difference is the leading `1` dimension on every
/// buffer.
///
/// # Errors
///
/// Returns `Err` for `n_seqs == 0`, `nkv == 0`, `hd == 0`, or
/// `cap == 0` — buffer alloc would otherwise underflow the kernel's
/// shape preconditions.  Mirrors A3a / A3b iter-1 pre-flight.
pub fn alloc_multi_seq_dense_kv_for_layer(
    dev: &MlxDevice,
    layer_idx: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    is_ring: bool,
    dtype: DType,
    n_seqs: u32,
) -> Result<MultiSeqDenseKvBuffers> {
    if n_seqs == 0 {
        return Err(anyhow!(
            "alloc_multi_seq_dense_kv_for_layer L{layer_idx}: n_seqs must be > 0"
        ));
    }
    if nkv == 0 || hd == 0 || cap == 0 {
        return Err(anyhow!(
            "alloc_multi_seq_dense_kv_for_layer L{layer_idx}: nkv/hd/cap must be \
             > 0 (got nkv={nkv}, hd={hd}, cap={cap})"
        ));
    }
    let n = n_seqs as usize;
    let elem_bytes = dtype.size_of();

    // Dense K: `[n_seqs, nkv, cap, hd]`.  n_seqs is OUTERMOST so
    // per-slot byte offset = `slot.0 * (nkv*cap*hd*elem_bytes)` is a
    // contiguous slab Phase B4c can address via `MlxBuffer::slice_view`.
    let k_elems = n * nkv * cap * hd;
    let k_bytes = k_elems * elem_bytes;
    let k = alloc_multi_seq_kv_storage(dev, k_bytes, dtype, vec![n, nkv, cap, hd], is_ring)
        .map_err(|e| anyhow!("multi-seq dense K L{layer_idx}: {e}"))?;

    // Dense V: same shape + dtype as K (ADR-017 Phase E.a iter-3.5a
    // invariant — both buffers in the legacy DenseKvBuffers share
    // dtype).
    let v_elems = n * nkv * cap * hd;
    let v_bytes = v_elems * elem_bytes;
    let v = alloc_multi_seq_kv_storage(dev, v_bytes, dtype, vec![n, nkv, cap, hd], is_ring)
        .map_err(|e| anyhow!("multi-seq dense V L{layer_idx}: {e}"))?;

    Ok(MultiSeqDenseKvBuffers {
        n_seqs,
        k,
        v,
        capacity: cap,
        is_sliding: is_ring,
        dtype,
        seq_lens: vec![0u32; n],
    })
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A3b iter-2 — MultiSeqKvCache impl for MultiSeqDenseKvBuffers.
//
// Mirrors A3b iter-1's MultiSeqHybridKvBuffers impl in structure
// (bounds-first per iter-1.5 cfa-finding-F5; fork_seq returns
// `CapabilityUnsupported` per iter-2.5 M1) and in invariants (per-slot
// cursor isolation; drop_seq does NOT zero the underlying K/V buffer
// bytes — Phase B4c reuses the slot's region on next admission).
//
// Phase A3b iter-2 scope: per-slot CURSOR bookkeeping only.  GPU
// buffer content writes land via the `alloc_multi_seq_dense_kv_for_layer`
// callers + the dense-attention dispatchers at Phase B4c.
// ──────────────────────────────────────────────────────────────────────────

impl crate::serve::multi_seq_kv::MultiSeqKvCache for MultiSeqDenseKvBuffers {
    fn layout(&self) -> crate::serve::multi_seq_kv::MultiSeqLayout {
        crate::serve::multi_seq_kv::MultiSeqLayout::SeparateSlots
    }

    fn slot_count(&self) -> u32 {
        // `MultiSeqDenseKvBuffers::n_seqs` is already `u32`; no cast.
        self.n_seqs
    }

    fn seq_len(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<u32, crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST (iter-1.5 cfa-finding-F5 ordering).
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only — MultiSeqDenseKvBuffers
        //    does not expose Paged.
        // 3. Return the per-seq cursor directly; `seq_lens.len() ==
        //    n_seqs` by construction (alloc_multi_seq_dense_kv_for_layer).
        Ok(self.seq_lens[slot.0 as usize])
    }

    fn append_for_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        n_tokens: u32,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only.
        // 3. Budget: SeparateSlots cannot SlotOom on append (buffers
        //    pre-allocated; cursor protected by saturating_add).
        //
        // ADR-040 Phase A3b iter-2 scope: bump the per-seq cursor.
        // The underlying K / V bytes for slot `slot.0` are written by
        // the dense-attention kernel dispatcher at Phase B4c via
        // `MlxBuffer::slice_view(...)`, identically to the A3b iter-1
        // MultiSeqHybridKvBuffers pattern.
        let cur = &mut self.seq_lens[slot.0 as usize];
        *cur = cur.saturating_add(n_tokens);
        Ok(())
    }

    fn drop_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only — no LayoutNotSupported.
        // 3. Budget: drop is a pure release; SlotOom unreachable.
        //
        // Cursor-only reset.  The underlying K / V bytes are NOT
        // zeroed; the next `append_for_seq` into this slot overwrites
        // them via the dense-attention dispatcher at Phase B4c.
        // Matches A3a / A3b iter-1 discipline (cursor-masked read).
        self.seq_lens[slot.0 as usize] = 0;
        Ok(())
    }

    fn fork_seq(
        &mut self,
        src: crate::serve::multi_seq_kv::SlotId,
        dst: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds — src FIRST per iter-1.5 cfa-finding-F5.
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
        // 3. Same-slot fork is a no-op per trait spec.
        if src == dst {
            return Ok(());
        }
        // ──────────────────────────────────────────────────────────────
        // ADR-040 Phase A3c (2026-05-30) — REAL cross-slot fork.
        //
        // Replaces the prior `CapabilityUnsupported` typed-deferral
        // with same-buffer cross-region memcpy on K + V per
        // `alloc_multi_seq_dense_kv_for_layer` at `kv_cache.rs:1374+`.
        // n_seqs OUTERMOST on both buffers ⇒ per-slot byte stride =
        // `total_bytes / n_seqs`.
        //
        // Cursor copy: `seq_lens[dst] = seq_lens[src]`.
        // ──────────────────────────────────────────────────────────────
        let src_idx = src.0 as usize;
        let dst_idx = dst.0 as usize;
        let n_seqs = self.n_seqs as usize;
        let live_tokens = self.seq_lens[src_idx] as usize;
        gemma4_copy_buffer_slot_prefix(&mut self.k, src_idx, dst_idx, n_seqs, live_tokens)
            .map_err(
                |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                    capability: gemma4_leak_static_str(format!(
                        "fork_seq: MultiSeqDenseKvBuffers k copy failed ({e})"
                    )),
                },
            )?;
        gemma4_copy_buffer_slot_prefix(&mut self.v, src_idx, dst_idx, n_seqs, live_tokens)
            .map_err(
                |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                    capability: gemma4_leak_static_str(format!(
                        "fork_seq: MultiSeqDenseKvBuffers v copy failed ({e})"
                    )),
                },
            )?;
        // Cursor copy AFTER buffer copy.
        self.seq_lens[dst_idx] = self.seq_lens[src_idx];
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 iter-B4c-kernel iter-1 — per-slot reset primitive for
// MultiSeqDenseKvBuffers (Gemma 4 dense variant; mirrors A3a/A3b iter-1
// siblings).
// ──────────────────────────────────────────────────────────────────────────

impl MultiSeqDenseKvBuffers {
    /// **ADR-040 iter-B4c-kernel iter-1** (shipped on this iter
    /// A3b iter-2) — per-slot reset for the persistent multi-seq
    /// `MultiSeqDenseKvBuffers` (Gemma 4 dense F32/F16 KV variant).
    ///
    /// Sibling of [`MultiSeqHbKvBuffers::reset_for_slot`] +
    /// [`MultiSeqHybridKvBuffers::reset_for_slot`] for the
    /// `HF2Q_USE_DENSE=1` codepath that engages `MultiSeqDenseKvBuffers`.
    /// When the Gemma 4 SlotAware worker arms are wired through the
    /// dense scaffold at Phase B4c, this primitive will be called at
    /// request entry + exit so the persistent per-layer cache is
    /// request-isolated within the slot.
    ///
    /// **Layout proof**:
    /// - **seq_lens**: `Vec<u32>` of length `n_seqs`. Per-slot reset →
    ///   set `seq_lens[slot_idx] = 0`; other slots untouched.  Same
    ///   load-bearing cursor as the A3a / A3b iter-1 siblings.
    /// - **k / v (dense `[n_seqs, nkv, capacity, head_dim]`)**: NOT
    ///   zeroed — cursor-masked discipline; the next
    ///   `append_for_seq`-then-kernel-write sequence overwrites the
    ///   per-slot K/V region from the dense-attention dispatcher.
    ///
    /// # Errors
    ///
    /// - `slot.0 >= self.n_seqs` (bounds-first per A2b iter-1.5
    ///   cfa-finding-F5 ordering).
    ///
    /// # Per-slot byte-equivalence pin
    ///
    /// At `slot = SlotId(0)` AND `n_seqs == 1` this is byte-equivalent
    /// to setting `seq_lens[0] = 0` directly (the existing `drop_seq`
    /// shape).  Sibling pin to the A3a / A3b iter-1 per-slot
    /// byte-equivalence (see H150).
    pub fn reset_for_slot(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // Bounds-first per A2b §6.1.23 iter-1.5 cfa-finding-F5.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // Reset the per-slot cursor.  K / V bytes are NOT zeroed
        // (cursor-masked read path — see layout proof above; matches
        // `drop_seq` invariant).
        self.seq_lens[slot.0 as usize] = 0;
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A3b iter-3 — FULL multi-seq lift for MlxKvCache (LEGACY
// 4-bit nibble-packed path; off-default since ADR-007 default-on TQ 8-bit).
//
// Direct mirror of A3b iter-2's `MultiSeqDenseKvBuffers` (§6.1.41) — NEW
// sibling struct `MultiSeqMlxKvCache` carries the lift with `n_seqs`
// outermost on every buffer + per-seq `seq_lens: Vec<u32>` cursor; the
// LEGACY `MlxKvCache` (lines 14-31) retains its typed clamp until Phase
// B4c re-routes the single production alloc site at
// `gemma4/model.rs:1277-1290` through the NEW
// `alloc_multi_seq_mlx_kv_for_layer` helper.
//
// Sibling-struct pattern (chain of 3 prior iters):
//   * A3a (§6.1.11)        — MultiSeqHbKvBuffers       (HbKvBuffers       lift)
//   * A3b iter-1 (§6.1.19) — MultiSeqHybridKvBuffers   (HybridKvBuffers   lift)
//   * A3b iter-2 (§6.1.41) — MultiSeqDenseKvBuffers    (DenseKvBuffers    lift)
//   * A3b iter-3 (THIS)    — MultiSeqMlxKvCache        (MlxKvCache        lift)
//
// Buffer shape table (per-slot bytes scale by `n_seqs` outermost):
//   * k_packed: U8  [n_seqs, nkv, capacity, hd/2]
//   * k_norms : F32 [n_seqs, nkv, capacity]                     (norms_per_pos=1)
//             or F32 [n_seqs, nkv, capacity, norms_per_pos]      (norms_per_pos>1)
//   * v_packed: U8  [n_seqs, nkv, capacity, hd/2]
//   * v_norms : F32 [n_seqs, nkv, capacity] / [n_seqs, nkv, capacity, norms_per_pos]
//
// Per-slot byte-offset formulas (Phase B4c will thread these through
// `MlxBuffer::slice_view` exactly as Qwen35 B4a-cont and Gemma 4
// iter-B4c-kernel iter-2B do):
//   * k_packed / v_packed slot.0 offset = slot.0 * (nkv * cap * (hd/2))
//   * k_norms  / v_norms  slot.0 offset = slot.0 * (nkv * cap * norms_per_pos * 4)
//
// `seq_len: usize` + `write_pos: usize` on the legacy struct are
// replaced by a single `seq_lens: Vec<u32>` of length `n_seqs` — the
// legacy invariant `write_pos == seq_len` (linear cache only;
// `trim()` line 56 enforces this) folds into the per-seq cursor
// since both fields advanced together on the linear path.
// ──────────────────────────────────────────────────────────────────────────

/// **ADR-040 Phase A3b iter-3** (this iter) — full multi-seq sibling of
/// the legacy [`MlxKvCache`].  Mirrors A3b iter-2's
/// [`MultiSeqDenseKvBuffers`] (LEGACY 4-bit nibble-packed flavour) and
/// A3b iter-1's [`MultiSeqHybridKvBuffers`] (production-default
/// hybrid flavour) sibling-struct patterns verbatim.
///
/// Outermost axis on every buffer is `n_seqs`.  Buffer layouts:
/// - K packed: `[n_seqs, nkv_heads, capacity, head_dim/2]` U8
///   (4-bit nibble-packed indices).
/// - K norms : `[n_seqs, nkv_heads, capacity]` F32 (norms_per_pos=1)
///   OR `[n_seqs, nkv_heads, capacity, norms_per_pos]` F32
///   (norms_per_pos > 1; per AmesianX iter-15 per-block norm at D=512).
/// - V packed: same shape + dtype as K packed.
/// - V norms : same shape + dtype as K norms.
///
/// Per-seq cursor [`seq_lens`](Self::seq_lens) is `Vec<u32>` of length
/// `n_seqs` (parallel to A3b iter-2's `MultiSeqDenseKvBuffers::seq_lens`).
/// Replaces the legacy [`MlxKvCache::seq_len`] + [`MlxKvCache::write_pos`]
/// pair — both advanced together on the linear path (`trim()` line 68:
/// `self.seq_len -= n_back; self.write_pos = self.seq_len;`), so a
/// single per-seq cursor preserves the invariant.
///
/// `is_sliding` is recorded on this struct just like the legacy
/// `MlxKvCache`; the per-slot ring-wrap math stays within each slot's
/// region by construction (`n_seqs` is outermost, so slot N's
/// `capacity` window is contiguous and disjoint from slot M's).
pub struct MultiSeqMlxKvCache {
    /// Number of physical slots — the outermost axis on every buffer.
    /// Set at construction via [`alloc_multi_seq_mlx_kv_for_layer`];
    /// once set, cannot change without reallocation.
    pub n_seqs: u32,
    /// K packed indices `[n_seqs, nkv_heads, capacity, head_dim/2]`
    /// U8 (nibble-packed).
    pub k_packed: MlxBuffer,
    /// K per-position norms.  Shape is
    /// `[n_seqs, nkv_heads, capacity]` when `norms_per_pos == 1` (D=256
    /// layers) and `[n_seqs, nkv_heads, capacity, norms_per_pos]` when
    /// `norms_per_pos > 1` (D=512 layers — per-block norm per
    /// AmesianX iter-15).  Dtype is F32.
    pub k_norms: MlxBuffer,
    /// V packed indices `[n_seqs, nkv_heads, capacity, head_dim/2]`
    /// U8 (nibble-packed).
    pub v_packed: MlxBuffer,
    /// V per-position norms — same shape + dtype as `k_norms`.
    pub v_norms: MlxBuffer,
    /// Cache capacity in positions (same as the legacy [`MlxKvCache`];
    /// max_seq_len for global, sliding_window for sliding).
    pub capacity: usize,
    /// True if ring-buffer (sliding window) semantics.
    pub is_sliding: bool,
    /// Norms shape selector — `1` for D=256 layers, `2` for D=512
    /// layers (per `gemma4/model.rs:1273` `(hd / 256).max(1)` formula).
    /// Pinned on the struct so the per-slot byte-offset formula
    /// `slot.0 * (nkv * cap * norms_per_pos * 4)` is reconstructable
    /// at the Phase B4c slot-view call site without reading the
    /// `MlxBuffer::shape()` len.
    pub norms_per_pos: usize,
    /// Per-seq write cursor; `seq_lens[slot.0]` is the number of valid
    /// positions stored in slot `slot.0` (this is the per-seq
    /// equivalent of the legacy `MlxKvCache::seq_len` field — and,
    /// since `trim()` keeps `write_pos == seq_len` on the linear path,
    /// also of `MlxKvCache::write_pos`).  `len() == n_seqs as usize`
    /// by construction (see [`alloc_multi_seq_mlx_kv_for_layer`]).
    /// Mirrors A3b iter-2's `MultiSeqDenseKvBuffers::seq_lens` discipline.
    pub seq_lens: Vec<u32>,
}

impl crate::serve::kv_persist::lcp_registry::ByteSized for MultiSeqMlxKvCache {
    /// Exact byte count: `k_packed + k_norms + v_packed + v_norms`.
    /// Used by the LcpRegistry byte budget; the lift to N slots scales
    /// every buffer by N at alloc-time, so `byte_len()` automatically
    /// reports the per-slot totals × N (parallel to A3b iter-2's
    /// `MultiSeqDenseKvBuffers::byte_len`).
    fn byte_len(&self) -> u64 {
        (self.k_packed.byte_len()
            + self.k_norms.byte_len()
            + self.v_packed.byte_len()
            + self.v_norms.byte_len()) as u64
    }
}

/// ADR-040 Phase A3b iter-3 — unified [`MultiSeqMlxKvCache`] allocator.
///
/// Mirrors A3b iter-2's [`alloc_multi_seq_dense_kv_for_layer`] in
/// signature shape; the `norms_per_pos` parameter is the per-layer
/// invariant the legacy single-seq production site carries inline
/// (`gemma4/model.rs:1273` computes `(hd / 256).max(1)` and threads
/// it into both `k_norms` and `v_norms` shape vectors).
///
/// At `n_seqs=1` the byte counts are byte-equivalent to the legacy
/// inline-alloc site's K + V packed + K + V norms allocs (H155
/// hypothesis); the only observable shape difference is the leading
/// `1` dimension on every buffer.
///
/// # Errors
///
/// Returns `Err` for `n_seqs == 0`, `nkv == 0`, `hd == 0`, `cap == 0`,
/// or `norms_per_pos == 0`, OR for an odd `hd` (4-bit nibble-packing
/// requires `hd` to be even so `hd/2` is exact — buffer alloc would
/// otherwise round-down silently and corrupt the per-position stride).
/// Mirrors A3a / A3b iter-{1,2} pre-flight + adds the iter-3-specific
/// `hd` evenness check the legacy path implicitly relies on at
/// `gemma4/model.rs:1272` (`nkv * capacity * (hd / 2)`).
pub fn alloc_multi_seq_mlx_kv_for_layer(
    dev: &MlxDevice,
    layer_idx: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    is_ring: bool,
    norms_per_pos: usize,
    n_seqs: u32,
) -> Result<MultiSeqMlxKvCache> {
    if n_seqs == 0 {
        return Err(anyhow!(
            "alloc_multi_seq_mlx_kv_for_layer L{layer_idx}: n_seqs must be > 0"
        ));
    }
    if nkv == 0 || hd == 0 || cap == 0 || norms_per_pos == 0 {
        return Err(anyhow!(
            "alloc_multi_seq_mlx_kv_for_layer L{layer_idx}: nkv/hd/cap/norms_per_pos \
             must be > 0 (got nkv={nkv}, hd={hd}, cap={cap}, norms_per_pos={norms_per_pos})"
        ));
    }
    if hd % 2 != 0 {
        return Err(anyhow!(
            "alloc_multi_seq_mlx_kv_for_layer L{layer_idx}: hd must be even for 4-bit \
             nibble-packed K/V (hd/2 stride; got hd={hd})"
        ));
    }
    let n = n_seqs as usize;

    // K packed: `[n_seqs, nkv, cap, hd/2]` U8 (4-bit nibble-packed).
    // n_seqs OUTERMOST so per-slot byte offset = `slot.0 * (nkv*cap*(hd/2))`
    // is a contiguous slab Phase B4c can address via `MlxBuffer::slice_view`.
    let hd_half = hd / 2;
    let k_packed_elems = n * nkv * cap * hd_half;
    let k_packed_bytes = k_packed_elems; // U8 = 1 byte/elem.
    let k_packed = alloc_multi_seq_kv_storage(
        dev,
        k_packed_bytes,
        DType::U8,
        vec![n, nkv, cap, hd_half],
        is_ring,
    )
    .map_err(|e| anyhow!("multi-seq MLX K packed L{layer_idx}: {e}"))?;

    // K norms: shape switches on norms_per_pos per the legacy formula
    // at `gemma4/model.rs:1280-1283`.  Dtype F32 = 4 bytes/elem.
    let k_norms_elems = n * nkv * cap * norms_per_pos;
    let k_norms_bytes = k_norms_elems * 4;
    let k_norms_shape: Vec<usize> = if norms_per_pos == 1 {
        vec![n, nkv, cap]
    } else {
        vec![n, nkv, cap, norms_per_pos]
    };
    let k_norms =
        alloc_multi_seq_kv_storage(dev, k_norms_bytes, DType::F32, k_norms_shape, is_ring)
            .map_err(|e| anyhow!("multi-seq MLX K norms L{layer_idx}: {e}"))?;

    // V packed: same shape + dtype as K packed.
    let v_packed_elems = n * nkv * cap * hd_half;
    let v_packed_bytes = v_packed_elems;
    let v_packed = alloc_multi_seq_kv_storage(
        dev,
        v_packed_bytes,
        DType::U8,
        vec![n, nkv, cap, hd_half],
        is_ring,
    )
    .map_err(|e| anyhow!("multi-seq MLX V packed L{layer_idx}: {e}"))?;

    // V norms: same shape + dtype as K norms.
    let v_norms_elems = n * nkv * cap * norms_per_pos;
    let v_norms_bytes = v_norms_elems * 4;
    let v_norms_shape: Vec<usize> = if norms_per_pos == 1 {
        vec![n, nkv, cap]
    } else {
        vec![n, nkv, cap, norms_per_pos]
    };
    let v_norms =
        alloc_multi_seq_kv_storage(dev, v_norms_bytes, DType::F32, v_norms_shape, is_ring)
            .map_err(|e| anyhow!("multi-seq MLX V norms L{layer_idx}: {e}"))?;

    Ok(MultiSeqMlxKvCache {
        n_seqs,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        capacity: cap,
        is_sliding: is_ring,
        norms_per_pos,
        seq_lens: vec![0u32; n],
    })
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A3b iter-3 — MultiSeqKvCache impl for MultiSeqMlxKvCache.
//
// Mirrors A3b iter-2's MultiSeqDenseKvBuffers impl in structure
// (bounds-first per iter-1.5 cfa-finding-F5; fork_seq returns
// `CapabilityUnsupported` per iter-2.5 M1) and in invariants (per-slot
// cursor isolation; drop_seq does NOT zero the underlying K/V byte
// buffers — Phase B4c reuses the slot's region on next admission via
// cursor-masked read).
//
// Phase A3b iter-3 scope: per-slot CURSOR bookkeeping only.  GPU
// buffer content writes land via the `alloc_multi_seq_mlx_kv_for_layer`
// callers + the TQ-active SDPA kernel dispatchers at Phase B4c.
// ──────────────────────────────────────────────────────────────────────────

impl crate::serve::multi_seq_kv::MultiSeqKvCache for MultiSeqMlxKvCache {
    fn layout(&self) -> crate::serve::multi_seq_kv::MultiSeqLayout {
        crate::serve::multi_seq_kv::MultiSeqLayout::SeparateSlots
    }

    fn slot_count(&self) -> u32 {
        // `MultiSeqMlxKvCache::n_seqs` is already `u32`; no cast.
        self.n_seqs
    }

    fn seq_len(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<u32, crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST (iter-1.5 cfa-finding-F5 ordering).
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only — MultiSeqMlxKvCache does not
        //    expose Paged.
        // 3. Return the per-seq cursor directly; `seq_lens.len() ==
        //    n_seqs` by construction (alloc_multi_seq_mlx_kv_for_layer).
        Ok(self.seq_lens[slot.0 as usize])
    }

    fn append_for_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        n_tokens: u32,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only.
        // 3. Budget: SeparateSlots cannot SlotOom on append (buffers
        //    pre-allocated; cursor protected by saturating_add).
        //
        // ADR-040 Phase A3b iter-3 scope: bump the per-seq cursor.
        // The underlying K packed / K norms / V packed / V norms bytes
        // for slot `slot.0` are written by the TQ-active SDPA kernel
        // dispatcher at Phase B4c via `MlxBuffer::slice_view(...)`,
        // identically to the A3b iter-{1,2} sibling patterns.
        let cur = &mut self.seq_lens[slot.0 as usize];
        *cur = cur.saturating_add(n_tokens);
        Ok(())
    }

    fn drop_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only — no LayoutNotSupported.
        // 3. Budget: drop is a pure release; SlotOom unreachable.
        //
        // Cursor-only reset.  The underlying K packed / K norms / V
        // packed / V norms bytes are NOT zeroed; the next
        // `append_for_seq` into this slot overwrites them via the
        // TQ-active SDPA dispatcher at Phase B4c.  Matches A3a / A3b
        // iter-{1,2} discipline (cursor-masked read).
        self.seq_lens[slot.0 as usize] = 0;
        Ok(())
    }

    fn fork_seq(
        &mut self,
        src: crate::serve::multi_seq_kv::SlotId,
        dst: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds — src FIRST per iter-1.5 cfa-finding-F5.
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
        // 3. Same-slot fork is a no-op per trait spec.
        if src == dst {
            return Ok(());
        }
        // ──────────────────────────────────────────────────────────────
        // ADR-040 Phase A3c (2026-05-30) — REAL cross-slot fork.
        //
        // Replaces the prior `CapabilityUnsupported` typed-deferral
        // with same-buffer cross-region memcpy on the four 4-bit-
        // nibble-packed buffers (k_packed U8 / k_norms F32 /
        // v_packed U8 / v_norms F32) per
        // `alloc_multi_seq_mlx_kv_for_layer` at `kv_cache.rs:1771+`.
        // n_seqs OUTERMOST on every buffer ⇒ per-slot byte stride =
        // `total_bytes / n_seqs`.
        //
        // Cursor copy: `seq_lens[dst] = seq_lens[src]`.
        // ──────────────────────────────────────────────────────────────
        let src_idx = src.0 as usize;
        let dst_idx = dst.0 as usize;
        let n_seqs = self.n_seqs as usize;
        let live_tokens = self.seq_lens[src_idx] as usize;
        gemma4_copy_buffer_slot_prefix(&mut self.k_packed, src_idx, dst_idx, n_seqs, live_tokens)
            .map_err(
            |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: gemma4_leak_static_str(format!(
                    "fork_seq: MultiSeqMlxKvCache k_packed copy failed ({e})"
                )),
            },
        )?;
        gemma4_copy_buffer_slot_prefix(&mut self.k_norms, src_idx, dst_idx, n_seqs, live_tokens)
            .map_err(
                |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                    capability: gemma4_leak_static_str(format!(
                        "fork_seq: MultiSeqMlxKvCache k_norms copy failed ({e})"
                    )),
                },
            )?;
        gemma4_copy_buffer_slot_prefix(&mut self.v_packed, src_idx, dst_idx, n_seqs, live_tokens)
            .map_err(
            |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: gemma4_leak_static_str(format!(
                    "fork_seq: MultiSeqMlxKvCache v_packed copy failed ({e})"
                )),
            },
        )?;
        gemma4_copy_buffer_slot_prefix(&mut self.v_norms, src_idx, dst_idx, n_seqs, live_tokens)
            .map_err(
                |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                    capability: gemma4_leak_static_str(format!(
                        "fork_seq: MultiSeqMlxKvCache v_norms copy failed ({e})"
                    )),
                },
            )?;
        // Cursor copy AFTER buffer copy.
        self.seq_lens[dst_idx] = self.seq_lens[src_idx];
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 iter-B4c-kernel iter-1 — per-slot reset primitive for
// MultiSeqMlxKvCache (Gemma 4 legacy 4-bit nibble-packed variant;
// mirrors A3a / A3b iter-{1,2} siblings).
// ──────────────────────────────────────────────────────────────────────────

impl MultiSeqMlxKvCache {
    /// **ADR-040 iter-B4c-kernel iter-1** (shipped on this iter
    /// A3b iter-3) — per-slot reset for the persistent multi-seq
    /// `MultiSeqMlxKvCache` (Gemma 4 LEGACY 4-bit nibble-packed KV
    /// variant; off-default since ADR-007 default-on TQ 8-bit).
    ///
    /// Sibling of [`MultiSeqHbKvBuffers::reset_for_slot`] +
    /// [`MultiSeqHybridKvBuffers::reset_for_slot`] +
    /// [`MultiSeqDenseKvBuffers::reset_for_slot`] for the legacy
    /// 4-bit codepath that engages `MultiSeqMlxKvCache`.  When the
    /// Gemma 4 SlotAware worker arms are wired through the legacy
    /// 4-bit scaffold at Phase B4c, this primitive will be called at
    /// request entry + exit so the persistent per-layer cache is
    /// request-isolated within the slot.
    ///
    /// **Layout proof**:
    /// - **seq_lens**: `Vec<u32>` of length `n_seqs`. Per-slot reset →
    ///   set `seq_lens[slot_idx] = 0`; other slots untouched.  Same
    ///   load-bearing cursor as the A3a / A3b iter-{1,2} siblings.
    /// - **k_packed / v_packed / k_norms / v_norms** (4-D, `n_seqs`
    ///   OUTERMOST): NOT zeroed — cursor-masked discipline; the next
    ///   `append_for_seq`-then-kernel-write sequence overwrites the
    ///   per-slot K packed / K norms / V packed / V norms region from
    ///   the TQ-active SDPA dispatcher.
    ///
    /// # Errors
    ///
    /// - `slot.0 >= self.n_seqs` (bounds-first per A2b iter-1.5
    ///   cfa-finding-F5 ordering).
    ///
    /// # Per-slot byte-equivalence pin
    ///
    /// At `slot = SlotId(0)` AND `n_seqs == 1` this is byte-equivalent
    /// to setting `seq_lens[0] = 0` directly (the existing `drop_seq`
    /// shape).  Sibling pin to the A3a / A3b iter-{1,2} per-slot
    /// byte-equivalence (see H157).
    pub fn reset_for_slot(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        // Bounds-first per A2b §6.1.23 iter-1.5 cfa-finding-F5.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // Reset the per-slot cursor.  K packed / K norms / V packed /
        // V norms bytes are NOT zeroed (cursor-masked read path — see
        // layout proof above; matches `drop_seq` invariant).
        self.seq_lens[slot.0 as usize] = 0;
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A3b iter-1 — TYPED CLAMP impls for DenseKvBuffers + MlxKvCache.
//
// Both variants are NON-DEFAULT today:
//   * DenseKvBuffers: reachable via `HF2Q_USE_DENSE=1` (off-default).
//     The MULTI-SEQ SIBLING `MultiSeqDenseKvBuffers` SHIPPED in
//     iter-A3b-2 (see §6.1.41 closure block in ADR-040) provides the
//     full lift; the LEGACY `DenseKvBuffers` retains its typed clamp
//     until Phase B4c re-routes the 3 production alloc sites
//     (`forward_prefill.rs:705`, `forward_prefill_batched.rs:367`,
//     `engine.rs:6836`) through `alloc_multi_seq_dense_kv_for_layer`.
//   * MlxKvCache: legacy 4-bit nibble-packed path (off-default since
//     ADR-007 default-on TQ 8-bit).
//     The MULTI-SEQ SIBLING `MultiSeqMlxKvCache` SHIPPED in
//     iter-A3b-3 (see §6.1.42 closure block in ADR-040) provides the
//     full lift; the LEGACY `MlxKvCache` retains its typed clamp
//     until Phase B4c re-routes the single production alloc site
//     (`gemma4/model.rs:1277-1290`) through
//     `alloc_multi_seq_mlx_kv_for_layer`.
//
// Per dossier R3 mitigation, each clamp:
//   * Returns `slot_count() == 1` (single-seq by construction).
//   * `seq_len(SlotId(0))` returns `Ok(internal_cursor as u32)`.
//   * Any operation on `slot.0 > 0` returns
//     `MultiSeqError::SlotOutOfRange { max_slots: 1, requested }`
//     (out-of-range discriminant, matching the canonical contract).
//   * In-bounds mutating operations that are NOT yet implemented
//     (append/drop/rollback at slot 0) return
//     `MultiSeqError::CapabilityUnsupported { capability }` with an
//     operator-grep'able label naming the deferred iter — so the
//     discriminant distinguishes "bad slot index" from "staged
//     capability" rather than collapsing both into one error.
//
// FULL LIFT status:
//   * iter-A3b-2 — DenseKvBuffers full multi-seq via sibling
//     `MultiSeqDenseKvBuffers` + `alloc_multi_seq_dense_kv_for_layer`
//     SHIPPED on this iter; Phase B4c re-routes the 3 production
//     alloc sites for kernel-side engagement.
//   * iter-A3b-3 — MlxKvCache full multi-seq via sibling
//     `MultiSeqMlxKvCache` + `alloc_multi_seq_mlx_kv_for_layer`
//     SHIPPED on this iter; Phase B4c re-routes the single legacy
//     production alloc site at `gemma4/model.rs:1277-1290` for
//     kernel-side engagement.  Legacy 4-bit path is off-default since
//     ADR-007 default-on TQ 8-bit; remains low-priority for the
//     production cutover.
//
// The clamps are non-vaporware (production paths that flip the
// env gate get an honest typed error pointing at the next iter)
// AND mantra-aligned (no stub, no fallback — every method has a
// real impl returning a real error with operator-grep'able context).
// ──────────────────────────────────────────────────────────────────────────

impl crate::serve::multi_seq_kv::MultiSeqKvCache for DenseKvBuffers {
    fn layout(&self) -> crate::serve::multi_seq_kv::MultiSeqLayout {
        crate::serve::multi_seq_kv::MultiSeqLayout::SeparateSlots
    }

    fn slot_count(&self) -> u32 {
        // Clamp: single-seq by construction.  Iter-A3b-2 lifts to N.
        1
    }

    fn seq_len(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<u32, crate::serve::multi_seq_kv::MultiSeqError> {
        if slot.0 >= 1 {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: 1,
            });
        }
        // DenseKvBuffers carries no internal cursor — seq_len is
        // tracked externally by the caller (the `MlxModelWeights`
        // owner) for this variant.  The single-seq clamp reports
        // `0` as a sentinel meaning "external cursor; consult the
        // owner".  Iter-A3b-2 adds the per-seq Vec<u32> cursor.
        Ok(0)
    }

    fn append_for_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        _n_tokens: u32,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        if slot.0 >= 1 {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: 1,
            });
        }
        // Single-seq clamp: external cursor.  Iter-A3b-2 SHIPPED
        // the multi-seq sibling `MultiSeqDenseKvBuffers` +
        // `alloc_multi_seq_dense_kv_for_layer`; this LEGACY struct
        // retains the typed clamp because the 3 production alloc
        // sites still emit `DenseKvBuffers` until Phase B4c re-routes
        // them through the multi-seq allocator.  Returning Ok(())
        // here would let a scheduler-side accountant proceed against
        // an unaware backing buffer; we honour the clamp by returning
        // a typed `CapabilityUnsupported` even at slot 0 — the label
        // points to the multi-seq sibling as the production path.
        Err(
            crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: "DenseKvBuffers::append_for_seq (legacy single-seq path; full multi-seq lift shipped in ADR-040 Phase A3b iter-2 — use MultiSeqDenseKvBuffers via alloc_multi_seq_dense_kv_for_layer)",
            },
        )
    }

    fn drop_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        if slot.0 >= 1 {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: 1,
            });
        }
        // Single-seq clamp: cursor lives outside this struct.
        // Iter-A3b-2 SHIPPED the multi-seq sibling
        // `MultiSeqDenseKvBuffers` with the real cursor reset; the
        // LEGACY struct retains the typed clamp until Phase B4c
        // re-routes the 3 production alloc sites.
        Err(
            crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: "DenseKvBuffers::drop_seq (legacy single-seq path; full multi-seq lift shipped in ADR-040 Phase A3b iter-2 — use MultiSeqDenseKvBuffers via alloc_multi_seq_dense_kv_for_layer)",
            },
        )
    }

    fn fork_seq(
        &mut self,
        src: crate::serve::multi_seq_kv::SlotId,
        dst: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        if src.0 >= 1 {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot: src,
                max_slots: 1,
            });
        }
        if dst.0 >= 1 {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot: dst,
                max_slots: 1,
            });
        }
        // Single-seq: src==dst==0 is the only valid combination,
        // and that is a no-op per trait spec.
        Ok(())
    }
}

impl crate::serve::multi_seq_kv::MultiSeqKvCache for MlxKvCache {
    fn layout(&self) -> crate::serve::multi_seq_kv::MultiSeqLayout {
        crate::serve::multi_seq_kv::MultiSeqLayout::SeparateSlots
    }

    fn slot_count(&self) -> u32 {
        // Clamp: single-seq by construction.  Iter-A3b-3 lifts to N.
        1
    }

    fn seq_len(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<u32, crate::serve::multi_seq_kv::MultiSeqError> {
        if slot.0 >= 1 {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: 1,
            });
        }
        // MlxKvCache carries `seq_len: usize` internally (legacy
        // single-seq cursor).  Report it via the trait surface for
        // diagnostic parity; clamp to u32::MAX defensively.
        Ok(u32::try_from(self.seq_len).unwrap_or(u32::MAX))
    }

    fn append_for_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
        _n_tokens: u32,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        if slot.0 >= 1 {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: 1,
            });
        }
        // Single-seq clamp: legacy path mutates `seq_len`/`write_pos`
        // via direct field access at production callsites (the trait
        // surface is not the canonical mutation path for the legacy
        // single-seq route).  Iter-A3b-3 SHIPPED the multi-seq sibling
        // `MultiSeqMlxKvCache` + `alloc_multi_seq_mlx_kv_for_layer`;
        // this LEGACY struct retains the typed clamp because the
        // single production alloc site at `gemma4/model.rs:1277-1290`
        // still emits `MlxKvCache` until Phase B4c re-routes it
        // through the multi-seq allocator.  Returning Ok(()) here
        // would let a scheduler-side accountant proceed against an
        // unaware backing buffer; we honour the clamp by returning a
        // typed `CapabilityUnsupported` even at slot 0 — the label
        // points to the multi-seq sibling as the production path.
        Err(
            crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: "MlxKvCache::append_for_seq (legacy 4-bit single-seq path; full multi-seq lift shipped in ADR-040 Phase A3b iter-3 — use MultiSeqMlxKvCache via alloc_multi_seq_mlx_kv_for_layer)",
            },
        )
    }

    fn drop_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        if slot.0 >= 1 {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: 1,
            });
        }
        // Single-seq clamp: cursor lives outside this struct.
        // Iter-A3b-3 SHIPPED the multi-seq sibling `MultiSeqMlxKvCache`
        // with the real cursor reset; the LEGACY struct retains the
        // typed clamp until Phase B4c re-routes the single legacy
        // production alloc site (`gemma4/model.rs:1277-1290`).
        Err(
            crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: "MlxKvCache::drop_seq (legacy 4-bit single-seq path; full multi-seq lift shipped in ADR-040 Phase A3b iter-3 — use MultiSeqMlxKvCache via alloc_multi_seq_mlx_kv_for_layer)",
            },
        )
    }

    fn fork_seq(
        &mut self,
        src: crate::serve::multi_seq_kv::SlotId,
        dst: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        if src.0 >= 1 {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot: src,
                max_slots: 1,
            });
        }
        if dst.0 >= 1 {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot: dst,
                max_slots: 1,
            });
        }
        // Single-seq: src==dst==0 is the only valid combo, no-op.
        Ok(())
    }
}

/// Per-call decode regime override for ADR-007 Gate H two-regime-one-process
/// runs (W12 iter-108a blocker #3).
///
/// Set on `MlxModelWeights` before each prefill+decode trajectory via
/// [`crate::inference::models::gemma4::MlxModelWeights::set_decode_regime`].
/// Consulted at the SDPA-mode gate inside `forward_decode` (the
/// `use_dense_sdpa` check); the four codebook-bits gates stay env-var-driven
/// because the codebook width is consistent across both regimes.
///
/// - `Default` (the zero value): preserve today's env-var behavior.
/// - `ForceTq`: ignore env, behave as if `HF2Q_USE_DENSE` were unset and
///   `HF2Q_LAYER_POLICY=tq_all` — TQ-active SDPA on every layer.
/// - `ForceDense`: ignore env, behave as if `HF2Q_USE_DENSE=1` were set —
///   dense-active SDPA on every layer.
///
/// Gate H uses one `MlxModelWeights` instance and runs (a)
/// `set_decode_regime(ForceDense)` → fresh prefill + decode loop → capture
/// tokens + per-token NLL + SDPA-output dumps; then (b)
/// `set_decode_regime(ForceTq)` → fresh prefill + decode loop with the same
/// prompt → cosine the SDPA outputs and PPL the NLLs. The per-instance step
/// counter is reset by `set_decode_regime` so each regime's stderr lines
/// start at `step=0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeRegime {
    /// Honor `HF2Q_USE_DENSE` / `HF2Q_LAYER_POLICY` env vars (today's path).
    #[default]
    Default,
    /// Force TQ-active SDPA regardless of env.
    /// (Wired by iter-108b's release-check.sh Gate 5 harness; no in-tree
    /// caller as of iter-108a.)
    #[allow(dead_code)]
    ForceTq,
    /// Force dense-active SDPA regardless of env.
    /// (Wired by iter-108b's release-check.sh Gate 5 harness.)
    #[allow(dead_code)]
    ForceDense,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_dev() -> Option<MlxDevice> {
        match MlxDevice::new() {
            Ok(d) => Some(d),
            Err(_) => {
                eprintln!("skip: no MlxDevice");
                None
            }
        }
    }

    #[test]
    fn gemma_anchor_owned_bytes_charge_all_family_allocations() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let anchor = GemmaHybridSlotAnchor::synthetic(7, &[11, 13, 17]);
        let layer_table = (anchor.layers.capacity() as u64)
            * std::mem::size_of::<Option<GemmaHybridSlidingLayerAnchor>>() as u64;
        let heap_payload: u64 = anchor
            .layers
            .iter()
            .flatten()
            .map(|layer| layer.k.capacity() as u64)
            .sum();
        assert_eq!(anchor.owned_bytes(), layer_table + heap_payload);
        assert_eq!(anchor.total_bytes(), 41);
    }

    #[test]
    fn gemma_anchor_restore_preflights_all_layers_before_first_mutation() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Some(dev) = skip_dev() else {
            return;
        };
        let mut scaffold = vec![
            alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, 2, 256, 4, true, 2).expect("layer zero"),
            alloc_multi_seq_hybrid_kv_for_layer(&dev, 1, 2, 256, 4, true, 2).expect("layer one"),
        ];
        for layer in &mut scaffold {
            layer.seq_lens[0] = 3;
            layer.k.as_mut_slice::<u8>().unwrap().fill(0x11);
            layer.v_packed.as_mut_slice::<u8>().unwrap().fill(0x22);
            layer.v_norms.as_mut_slice::<u8>().unwrap().fill(0x33);
            if let Some(buffer) = layer.bf16_xlen_k.as_mut() {
                buffer.as_mut_slice::<u8>().unwrap().fill(0x44);
            }
            if let Some(buffer) = layer.bf16_xlen_v.as_mut() {
                buffer.as_mut_slice::<u8>().unwrap().fill(0x55);
            }
        }
        let mut anchor =
            snapshot_gemma_hybrid_slot_anchor(&scaffold, crate::serve::multi_seq_kv::SlotId(0), 3)
                .expect("snapshot");

        // Make layer zero visibly different from its checkpoint, then corrupt
        // only the final layer's saved layout. A write-before-full-preflight
        // implementation would silently restore layer zero before failing.
        scaffold[0].k.as_mut_slice::<u8>().unwrap().fill(0xA5);
        let layer_zero_before = scaffold[0].k.as_slice::<u8>().unwrap().to_vec();
        anchor.layers[1].as_mut().expect("sliding layer").k.pop();
        let cursors_before: Vec<Vec<u32>> = scaffold
            .iter()
            .map(|layer| layer.seq_lens.clone())
            .collect();

        let error = restore_gemma_hybrid_slot_anchor(
            &mut scaffold,
            crate::serve::multi_seq_kv::SlotId(0),
            &anchor,
            2,
        )
        .expect_err("late-layer mismatch must fail");
        assert!(
            error.to_string().contains("L1 K"),
            "unexpected error: {error:#}"
        );
        assert_eq!(
            scaffold[0].k.as_slice::<u8>().unwrap(),
            layer_zero_before,
            "layer zero mutated before layer one failed preflight"
        );
        assert_eq!(
            scaffold
                .iter()
                .map(|layer| &layer.seq_lens)
                .collect::<Vec<_>>(),
            cursors_before.iter().collect::<Vec<_>>(),
            "no cursor may change on failed preflight"
        );
    }

    #[test]
    fn mlx_kv_cache_trim_linear_decrements_seq_len() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let mut cache = MlxKvCache {
            k_packed: buf(),
            k_norms: buf(),
            v_packed: buf(),
            v_norms: buf(),
            capacity: 16,
            is_sliding: false,
            write_pos: 8,
            seq_len: 8,
        };
        let new_len = cache.trim(3).unwrap();
        assert_eq!(new_len, 5);
        assert_eq!(cache.seq_len, 5);
        assert_eq!(cache.write_pos, 5);
    }

    #[test]
    fn mlx_kv_cache_trim_sliding_errors() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let mut cache = MlxKvCache {
            k_packed: buf(),
            k_norms: buf(),
            v_packed: buf(),
            v_norms: buf(),
            capacity: 16,
            is_sliding: true,
            write_pos: 4,
            seq_len: 4,
        };
        assert!(cache.trim(1).is_err());
    }

    #[test]
    fn mlx_kv_cache_trim_overflow_errors() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let mut cache = MlxKvCache {
            k_packed: buf(),
            k_norms: buf(),
            v_packed: buf(),
            v_norms: buf(),
            capacity: 16,
            is_sliding: false,
            write_pos: 3,
            seq_len: 3,
        };
        assert!(cache.trim(10).is_err());
    }

    #[test]
    fn mlx_kv_cache_visible_len_eq_seq_len() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let cache = MlxKvCache {
            k_packed: buf(),
            k_norms: buf(),
            v_packed: buf(),
            v_norms: buf(),
            capacity: 32,
            is_sliding: false,
            write_pos: 7,
            seq_len: 7,
        };
        assert_eq!(cache.visible_len(), cache.seq_len);
    }

    #[test]
    fn decode_regime_default_via_default_trait() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let r: DecodeRegime = Default::default();
        assert_eq!(r, DecodeRegime::Default);
    }

    #[test]
    fn decode_regime_variants_distinct() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        assert_ne!(DecodeRegime::Default, DecodeRegime::ForceTq);
        assert_ne!(DecodeRegime::Default, DecodeRegime::ForceDense);
        assert_ne!(DecodeRegime::ForceTq, DecodeRegime::ForceDense);
    }

    #[test]
    fn hybrid_kv_buffers_byte_len_sums_fields() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2;
        let cap = 4;
        let hd = 256;
        let k = dev
            .alloc_buffer(nkv * cap * hd * 2, DType::F16, vec![nkv, cap, hd])
            .unwrap();
        let v_packed = dev
            .alloc_buffer(nkv * cap * hd, DType::U8, vec![nkv, cap, hd])
            .unwrap();
        let v_norms = dev
            .alloc_buffer(nkv * cap * 4, DType::F32, vec![nkv, cap])
            .unwrap();
        let k_bytes = k.byte_len();
        let vp_bytes = v_packed.byte_len();
        let vn_bytes = v_norms.byte_len();
        let buf = HybridKvBuffers {
            k,
            v_packed,
            v_norms,
            capacity: cap,
            is_sliding: false,
            norms_per_pos: 1,
            bf16_xlen_k: None,
            bf16_xlen_v: None,
        };
        use crate::serve::kv_persist::lcp_registry::ByteSized;
        assert_eq!(buf.byte_len(), (k_bytes + vp_bytes + vn_bytes) as u64);
    }

    #[test]
    fn dense_kv_buffers_byte_len_sums_k_plus_v() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2;
        let cap = 8;
        let hd = 256;
        let k = dev
            .alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd])
            .unwrap();
        let v = dev
            .alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd])
            .unwrap();
        let kb = k.byte_len();
        let vb = v.byte_len();
        let buf = DenseKvBuffers {
            k,
            v,
            capacity: cap,
            is_sliding: false,
            dtype: DType::F32,
        };
        use crate::serve::kv_persist::lcp_registry::ByteSized;
        assert_eq!(buf.byte_len(), (kb + vb) as u64);
    }

    #[test]
    fn alloc_hybrid_kv_for_layer_no_xlen_no_full_f16() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        // Ensure env gates are off for this test.
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        let buf = alloc_hybrid_kv_for_layer(&dev, 0, 2, 256, 8, false).unwrap();
        assert!(buf.bf16_xlen_k.is_none());
        assert!(buf.bf16_xlen_v.is_none());
        assert_eq!(buf.capacity, 8);
        assert!(!buf.is_sliding);
        assert_eq!(buf.norms_per_pos, 1);
    }

    #[test]
    fn alloc_hybrid_kv_for_layer_full_f16_v_allocates_f16() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        std::env::set_var("HF2Q_FULL_F16_KV", "1");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        let buf = alloc_hybrid_kv_for_layer(&dev, 1, 2, 256, 4, true).unwrap();
        // v_norms is the dummy 4-byte buffer when full_f16_v=true.
        assert_eq!(buf.v_norms.byte_len(), 4);
        assert!(buf.is_sliding);
        std::env::remove_var("HF2Q_FULL_F16_KV");
    }

    #[test]
    fn alloc_hybrid_kv_for_layer_xlen_allocates_bf16_buffers() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::set_var("HF2Q_DFLASH_XLEN_SDPA", "1");
        let buf = alloc_hybrid_kv_for_layer(&dev, 2, 2, 256, 4, false).unwrap();
        assert!(buf.bf16_xlen_k.is_some());
        assert!(buf.bf16_xlen_v.is_some());
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
    }

    #[test]
    fn alloc_hybrid_kv_for_layer_norms_per_pos_d256_d512() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        // D=256: norms_per_pos == 1.
        let buf256 = alloc_hybrid_kv_for_layer(&dev, 0, 2, 256, 4, false).unwrap();
        assert_eq!(buf256.norms_per_pos, 1);
        // D=512: norms_per_pos == 2.
        let buf512 = alloc_hybrid_kv_for_layer(&dev, 0, 2, 512, 4, false).unwrap();
        assert_eq!(buf512.norms_per_pos, 2);
    }

    // ───────────────────────────────────────────────────────────────────────
    // ADR-040 Phase A3a iter-3 — multi-seq lift hypotheses + trait impl
    // tests.  See `docs/research/adr040-kv-cache-lift-dossier-2026-05-23.md`
    // §2.9 for H6–H10 falsification statements.
    //
    // Order mirrors the dossier §4 iter-3 sequencing + the Qwen35 A2a
    // test block layout at `src/inference/models/qwen35/kv_cache.rs:
    // 6176-7115`:
    //   H6 (allocator byte-scale + multi-seq via offset)  — must PASS
    //                                                        before the
    //                                                        `impl
    //                                                        MultiSeqKvCache`
    //                                                        block is
    //                                                        trusted.
    //   H7 (sliding-window per-slot isolation)            — pins ring-
    //                                                        buffer
    //                                                        independence.
    //   H8 (alloc helper byte-equivalence vs inline)      — pins drift-
    //                                                        risk
    //                                                        elimination
    //                                                        for B4c.
    //   Trait-surface pins (slot_count, OOR, drop, fork-  — exercise
    //     to-self, fork cross-slot deferral, append          methods
    //     isolation, layout discriminant, append-only-       directly.
    //     target, drop-content-invariance).
    //   M5-equivalent shape pin                            — pins n_seqs
    //                                                        axis position.
    //
    // H9 (mixed Gemma 4 layer_types) is a HYPOTHESIS-ONLY pin: production
    // Gemma 4 has mixed `LayerType::Full` + `LayerType::Sliding` per
    // `src/inference/models/gemma4/model.rs:1250`, but this struct lift
    // is per-layer-agnostic (the struct stores `is_sliding` and the lift
    // applies to both branches uniformly).  Verified by code-reading;
    // no synthetic test fixture required.
    //
    // H10 status: FALSIFIED.  `HF2Q_HYBRID_KV` is default-ON since
    // ADR-029 iter-13 per `src/debug/investigation_env.rs:878` (the
    // dossier's pre-iter-13 assumption that the production default
    // was HbKvBuffers is stale).  This raises A3b's priority but does
    // NOT block A3a: HbKvBuffers is still reached on the
    // `HF2Q_HYBRID_KV=0` opt-out path, and the multi-seq lift here
    // is a structural prerequisite for the HybridKvBuffers lift in
    // A3b regardless of which variant is the production default.
    // See the §2.10 R3 mitigation in the dossier — the brief's
    // "defer to A3b" framing stays intact.
    // ───────────────────────────────────────────────────────────────────────

    /// Test-imports for the trait-surface pins.  Pulled into the test
    /// module rather than the parent module so production code carries
    /// no test-only imports.
    use crate::serve::multi_seq_kv::{MultiSeqError, MultiSeqKvCache as _, MultiSeqLayout, SlotId};

    /// Dossier §2.9 H6 falsifier — extending `HbKvBuffers` (here as the
    /// sibling [`MultiSeqHbKvBuffers`] per the brief-deviation rationale
    /// documented at the struct definition) to carry `n_seqs` outermost
    /// requires no kernel changes IF the kernel-side write address is
    /// derivable from `(n_seqs, cache_capacity, write_pos)` via byte
    /// arithmetic at the caller.
    ///
    /// This iter-3 H6 pin checks the **allocator side** of that claim:
    /// the 4-D buffers at n_seqs=4 must have exactly 4× the bytes of
    /// their n_seqs=1 counterparts (k_packed, v_packed, k_norms,
    /// v_norms).  Per-slot byte offset = `slot.0 * (nkv*cap*hd)`
    /// packed / `slot.0 * (nkv*cap*norms_per_pos*4)` norms — the
    /// alloc multiplies by `n` in the helper.
    ///
    /// Phase B4c will verify the kernel-side half (write to slot 1
    /// via byte-offset Q/K dispatch, read back slot 0 unchanged); A3a
    /// owns the alloc-side byte-equivalence which is the falsifier
    /// for the dossier H6 structural claim.
    ///
    /// Falsifier (any one ⇒ H6 broken):
    /// 1. `alloc_hb_kv_for_layer(.., n_seqs=4)` panics or errors.
    /// 2. K-packed at n_seqs=4 is NOT exactly 4× n_seqs=1 baseline.
    /// 3. V-packed at n_seqs=4 is NOT exactly 4×.
    /// 4. K-norms at n_seqs=4 is NOT exactly 4×.
    /// 5. V-norms at n_seqs=4 is NOT exactly 4×.
    /// 6. `seq_lens.len() != n_seqs`.
    #[test]
    fn h6_hb_kv_buffers_n_seqs_4_byte_scale() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        // Per dossier H6: shape choices identical to existing
        // alloc_hybrid_kv_for_layer test fixtures (nkv=2, hd=256,
        // cap=8) so the byte-count formula matches what production
        // sees at the 3 inline alloc sites.
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;
        let baseline =
            alloc_hb_kv_for_layer(&dev, 0, nkv, hd, cap, false, 1).expect("H6: alloc at n_seqs=1");
        let lifted =
            alloc_hb_kv_for_layer(&dev, 0, nkv, hd, cap, false, 4).expect("H6: alloc at n_seqs=4");

        // n_seqs field propagated.
        assert_eq!(baseline.n_seqs, 1, "H6: baseline n_seqs=1");
        assert_eq!(lifted.n_seqs, 4, "H6: lifted n_seqs=4");

        // Falsifier 2: K-packed 4×.
        assert_eq!(
            lifted.k_packed.byte_len(),
            baseline.k_packed.byte_len() * 4,
            "H6 FALSIFIED: k_packed does not scale 4× ({} != {} * 4 = {})",
            lifted.k_packed.byte_len(),
            baseline.k_packed.byte_len(),
            baseline.k_packed.byte_len() * 4
        );
        // Falsifier 3: V-packed 4×.
        assert_eq!(
            lifted.v_packed.byte_len(),
            baseline.v_packed.byte_len() * 4,
            "H6 FALSIFIED: v_packed does not scale 4× ({} != {} * 4 = {})",
            lifted.v_packed.byte_len(),
            baseline.v_packed.byte_len(),
            baseline.v_packed.byte_len() * 4
        );
        // Falsifier 4: K-norms 4×.
        assert_eq!(
            lifted.k_norms.byte_len(),
            baseline.k_norms.byte_len() * 4,
            "H6 FALSIFIED: k_norms does not scale 4× ({} != {})",
            lifted.k_norms.byte_len(),
            baseline.k_norms.byte_len() * 4
        );
        // Falsifier 5: V-norms 4×.
        assert_eq!(
            lifted.v_norms.byte_len(),
            baseline.v_norms.byte_len() * 4,
            "H6 FALSIFIED: v_norms does not scale 4×"
        );

        // Falsifier 6: per-seq cursor vec length tracks n_seqs.
        assert_eq!(baseline.seq_lens.len(), 1, "H6: baseline seq_lens.len()");
        assert_eq!(lifted.seq_lens.len(), 4, "H6: lifted seq_lens.len()");
        // All cursors start at 0.
        assert!(
            baseline.seq_lens.iter().all(|&x| x == 0),
            "H6: baseline seq_lens initialized to 0"
        );
        assert!(
            lifted.seq_lens.iter().all(|&x| x == 0),
            "H6: lifted seq_lens initialized to 0"
        );
    }

    /// Dossier §2.9 H7 falsifier — Gemma 4's sliding-window path
    /// (is_sliding=true ring buffer) is per-slot isolated: a write to
    /// slot 0's region must NOT touch slot 1's bytes.
    ///
    /// At Phase A3a the kernel-dispatcher slot-offset routing is not
    /// yet wired (Phase B4c scope per the brief's NOTE under Step 3),
    /// so this test verifies isolation at the **buffer-region level**
    /// via direct host-side writes: write a deterministic pattern
    /// into slot 1's K/V regions, advance slot 0's cursor (no buffer
    /// mutation in A3a), assert slot 1's bytes are unchanged.
    ///
    /// This pins the structural precondition Phase B4c will rely on:
    /// the per-slot byte-offset formula
    /// `slot.0 * (nkv*cap*hd)` produces disjoint contiguous regions
    /// for sliding AND linear caches identically (the `is_sliding`
    /// flag only changes the kernel's wrap-on-write behaviour
    /// WITHIN a slot, not the inter-slot byte offset).
    ///
    /// Falsifier: any byte change in slot 1's K/V region after the
    /// slot-0 cursor advance ⇒ H7 broken; per-slot isolation
    /// assumption invalid.
    #[test]
    fn h7_hb_kv_sliding_per_slot_isolation() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 4usize; // small ring for a sliding cache
                          // is_ring=true mirrors Gemma 4 LayerType::Sliding allocation
                          // (forward_prefill.rs:847-861 sliding-layer branch).
        let mut cache = alloc_hb_kv_for_layer(&dev, 0, nkv, hd, cap, true, 2)
            .expect("H7: alloc n_seqs=2 sliding");
        assert!(cache.is_sliding, "H7: sliding flag propagated");
        assert_eq!(cache.n_seqs, 2);

        // Per-slot region size for K/V packed: nkv * cap * hd bytes (U8).
        let slot_packed = nkv * cap * hd;
        let total_packed = cache.k_packed.byte_len();
        assert_eq!(
            total_packed,
            2 * slot_packed,
            "H7 fixture sanity: total bytes = 2 * slot_packed"
        );

        // Write a deterministic non-zero pattern into slot 1's K and
        // V packed regions (offsets [slot_packed .. 2*slot_packed)).
        {
            let k_slice = cache
                .k_packed
                .as_mut_slice::<u8>()
                .expect("k_packed u8 mut");
            for (i, b) in k_slice[slot_packed..2 * slot_packed].iter_mut().enumerate() {
                // Pattern: position-dependent so a partial-zero bug
                // (zero only first N bytes of slot 1) surfaces.
                *b = ((i % 251) + 1) as u8;
            }
            // Slot 0 region intentionally left at the zero-init from
            // alloc_hb_kv_for_layer — proves slot 0's region is
            // untouched by the slot-1 write.
        }
        {
            let v_slice = cache
                .v_packed
                .as_mut_slice::<u8>()
                .expect("v_packed u8 mut");
            for (i, b) in v_slice[slot_packed..2 * slot_packed].iter_mut().enumerate() {
                *b = ((i % 253) + 1) as u8;
            }
        }

        // Snapshot slot 1's bytes for later comparison.
        let k_slot1_before: Vec<u8> = cache.k_packed.as_slice::<u8>().expect("k_packed u8")
            [slot_packed..2 * slot_packed]
            .to_vec();
        let v_slot1_before: Vec<u8> = cache.v_packed.as_slice::<u8>().expect("v_packed u8")
            [slot_packed..2 * slot_packed]
            .to_vec();
        // Sanity: slot 0 region is all-zero before the cursor advance.
        let k_slot0_before: Vec<u8> =
            cache.k_packed.as_slice::<u8>().expect("k_packed u8")[0..slot_packed].to_vec();
        assert!(
            k_slot0_before.iter().all(|&b| b == 0),
            "H7 fixture sanity: slot 0 K region zero-init"
        );

        // A3a-scope cursor advance on slot 0.  Per the brief's NOTE
        // under Step 3, kernel-dispatch slot-offset routing is B4c
        // scope; A3a's trait surface only mutates `seq_lens[0]`.
        cache
            .append_for_seq(SlotId(0), 3)
            .expect("H7: append slot 0 cursor");
        assert_eq!(cache.seq_lens[0], 3, "H7: slot 0 cursor advanced");
        assert_eq!(cache.seq_lens[1], 0, "H7: slot 1 cursor untouched");

        // H7 falsifier: slot 1's K/V bytes must be byte-identical to
        // the snapshot taken before the slot-0 cursor advance.
        let k_slot1_after: Vec<u8> = cache.k_packed.as_slice::<u8>().expect("k_packed u8")
            [slot_packed..2 * slot_packed]
            .to_vec();
        let v_slot1_after: Vec<u8> = cache.v_packed.as_slice::<u8>().expect("v_packed u8")
            [slot_packed..2 * slot_packed]
            .to_vec();
        assert_eq!(
            k_slot1_before, k_slot1_after,
            "H7 FALSIFIED: slot 1's k_packed bytes changed after slot-0 \
             cursor advance — per-slot isolation invariant broken"
        );
        assert_eq!(
            v_slot1_before, v_slot1_after,
            "H7 FALSIFIED: slot 1's v_packed bytes changed after slot-0 \
             cursor advance"
        );
    }

    /// Dossier §2.9 H8 falsifier — `alloc_hb_kv_for_layer(.., n_seqs=1)`
    /// produces byte counts byte-equivalent to the 3 inline alloc sites
    /// (`forward_prefill.rs:843-882`, `forward_prefill_batched.rs:443-475`,
    /// `forward_gpu.rs:443-459`), eliminating drift risk for Phase B4c's
    /// refactor.
    ///
    /// The 3 sites' formula is:
    ///   k_packed_bytes = nkv * cap * hd       (U8 → 1 byte/elem)
    ///   v_packed_bytes = same
    ///   norms_bytes    = nkv * cap * norms_per_pos * 4 (F32)
    ///   where norms_per_pos = (hd / 256).max(1)
    ///
    /// At n_seqs=1 the helper's formula multiplies by n=1, producing
    /// identical byte counts.  Shape differs trivially (4-D
    /// `[1, nkv, cap, hd]` vs 3-D `[nkv, cap, hd]`) — this is the
    /// observable difference Phase B4c documents when wiring the
    /// helper into the 3 sites.
    ///
    /// Falsifier: any byte-count mismatch ⇒ H8 broken; B4c refactor
    /// would silently change buffer sizes.
    #[test]
    fn h8_alloc_hb_kv_for_layer_byte_equivalent_to_pre_refactor() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;
        let norms_per_pos = (hd / 256).max(1);

        // The 3 inline alloc sites' byte formula (verbatim from
        // forward_prefill.rs:864-875).
        let expected_packed_bytes = nkv * cap * hd; // U8
        let expected_norms_bytes = nkv * cap * norms_per_pos * std::mem::size_of::<f32>();

        let helper =
            alloc_hb_kv_for_layer(&dev, 0, nkv, hd, cap, false, 1).expect("H8: helper at n_seqs=1");

        // K/V packed bytes match the inline formula.
        assert_eq!(
            helper.k_packed.byte_len(),
            expected_packed_bytes,
            "H8 FALSIFIED: k_packed bytes diverge from inline formula \
             ({} != {})",
            helper.k_packed.byte_len(),
            expected_packed_bytes
        );
        assert_eq!(
            helper.v_packed.byte_len(),
            expected_packed_bytes,
            "H8 FALSIFIED: v_packed bytes diverge from inline formula"
        );

        // K/V norms bytes match.
        assert_eq!(
            helper.k_norms.byte_len(),
            expected_norms_bytes,
            "H8 FALSIFIED: k_norms bytes diverge from inline formula \
             ({} != {})",
            helper.k_norms.byte_len(),
            expected_norms_bytes
        );
        assert_eq!(
            helper.v_norms.byte_len(),
            expected_norms_bytes,
            "H8 FALSIFIED: v_norms bytes diverge from inline formula"
        );

        // Shape difference is the documented observable: 4-D vs 3-D.
        // Helper at n_seqs=1 yields `[1, nkv, cap, hd]`; the inline
        // sites use `[nkv, cap, hd]`.  Phase B4c notes this in the
        // refactor diff — the byte count is invariant, the shape rank
        // changes from 3 to 4 with the leading 1.
        assert_eq!(
            helper.k_packed.shape(),
            &[1, nkv, cap, hd],
            "H8: helper k_packed shape includes leading n_seqs=1 axis"
        );
        // Also verify norms_per_pos branch matches.
        assert_eq!(helper.norms_per_pos, norms_per_pos);
        assert_eq!(helper.capacity, cap);
    }

    /// M5-equivalent shape proof for [`MultiSeqHbKvBuffers`] — pins
    /// `n_seqs` as the OUTERMOST axis on every buffer (shape[0]).
    /// Mirrors Qwen35 H1's M5 strengthening at `qwen35/kv_cache.rs:
    /// 6345-6474`.
    ///
    /// Falsifier: any buffer where shape[0] != n_seqs ⇒ per-slot
    /// byte-offset arithmetic at the caller (Phase B4c) would index
    /// the wrong slot, silently corrupting the cache.  byte_len()
    /// alone cannot catch this (n_seqs and another axis can swap
    /// without changing the product).
    #[test]
    fn gemma4_hb_kv_n_seqs_outermost_axis() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let cache_1 = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 1).expect("alloc n_seqs=1");
        let cache_4 = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4).expect("alloc n_seqs=4");

        // All 4 buffers must be 4-D with n_seqs at shape[0].
        for (name, b1, b4) in [
            ("k_packed", &cache_1.k_packed, &cache_4.k_packed),
            ("v_packed", &cache_1.v_packed, &cache_4.v_packed),
            ("k_norms", &cache_1.k_norms, &cache_4.k_norms),
            ("v_norms", &cache_1.v_norms, &cache_4.v_norms),
        ] {
            let s1 = b1.shape().to_vec();
            let s4 = b4.shape().to_vec();
            assert_eq!(
                s1.len(),
                4,
                "M5: {name} (n_seqs=1) must be 4-D; got {:?}",
                s1
            );
            assert_eq!(
                s4.len(),
                4,
                "M5: {name} (n_seqs=4) must be 4-D; got {:?}",
                s4
            );
            assert_eq!(
                s1[0], 1,
                "M5: {name} baseline shape[0] must be n_seqs=1; got {:?}",
                s1
            );
            assert_eq!(
                s4[0], 4,
                "M5 FALSIFIED: {name} shape[0] must be n_seqs=4 \
                 (n_seqs landed on wrong axis); got {:?}",
                s4
            );
            // Non-n_seqs dims invariant — catches axis permutation.
            assert_eq!(
                &s4[1..],
                &s1[1..],
                "M5 FALSIFIED: {name} non-n_seqs dims diverge between \
                 n_seqs=1 ({:?}) and n_seqs=4 ({:?})",
                s1,
                s4
            );
        }
    }

    /// Pin: `slot_count()` returns the constructor's `n_seqs` verbatim.
    /// Falsifies any future refactor that introduces a u32→u64 cast or
    /// silently caps the value.  Mirrors `qwen35_hybrid_kv_slot_count_
    /// matches_n_seqs` (qwen35/kv_cache.rs:6669-6677).
    #[test]
    fn gemma4_hb_kv_slot_count_matches_n_seqs() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let c1 = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 1).expect("alloc 1");
        let c4 = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4).expect("alloc 4");
        assert_eq!(c1.slot_count(), 1);
        assert_eq!(c4.slot_count(), 4);
    }

    /// Pin: `layout()` returns `SeparateSlots`.  MultiSeqHbKvBuffers
    /// does not expose Paged — bounds-first ordering means this trip
    /// is only observable through this getter.
    #[test]
    fn gemma4_hb_kv_layout_is_separate_slots() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4).expect("alloc");
        assert_eq!(c.layout(), MultiSeqLayout::SeparateSlots);
    }

    /// Pin (iter-1.5 cfa-finding-F5): out-of-range `SlotId` surfaces as
    /// `SlotOutOfRange { slot, max_slots }` with both fields populated
    /// across every trait method — bounds-first ordering preserved.
    /// Mirrors `qwen35_hybrid_kv_slot_out_of_range_errors_named`
    /// (qwen35/kv_cache.rs:6695-6721).
    #[test]
    fn gemma4_hb_kv_slot_out_of_range_errors_named() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let mut c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4).expect("alloc");

        // seq_len OOR
        let err = c.seq_len(SlotId(4)).expect_err("slot 4 OOR for n_seqs=4");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(4),
                max_slots: 4
            }
        );
        let err = c.seq_len(SlotId(99)).expect_err("slot 99 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(99),
                max_slots: 4
            }
        );

        // append_for_seq OOR
        let err = c.append_for_seq(SlotId(4), 1).expect_err("append OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(4),
                max_slots: 4
            }
        );

        // drop_seq OOR
        let err = c.drop_seq(SlotId(4)).expect_err("drop OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(4),
                max_slots: 4
            }
        );

        // fork_seq src OOR FIRST (deterministic per fixture-parity).
        let err = c
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
        let err = c.fork_seq(SlotId(0), SlotId(4)).expect_err("fork: dst OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(4),
                max_slots: 4
            }
        );
    }

    /// Pin: `append_for_seq` advances ONLY the named slot's cursor.
    /// Surface-level isolation evidence for H6's per-slot O(1) bound
    /// (the per-buffer GPU write isolation lands in Phase B4c).
    /// Mirrors `qwen35_hybrid_kv_append_advances_target_slot_only`
    /// (qwen35/kv_cache.rs:6727-6745).
    #[test]
    fn gemma4_hb_kv_append_advances_target_slot_only() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let mut c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4).expect("alloc");
        // All slots start at 0.
        for s in 0..4 {
            assert_eq!(c.seq_len(SlotId(s)).expect("seq_len in range"), 0);
        }
        c.append_for_seq(SlotId(0), 5).expect("append slot 0");
        c.append_for_seq(SlotId(2), 3).expect("append slot 2");
        assert_eq!(c.seq_len(SlotId(0)).unwrap(), 5);
        assert_eq!(c.seq_len(SlotId(1)).unwrap(), 0, "slot 1 untouched");
        assert_eq!(c.seq_len(SlotId(2)).unwrap(), 3);
        assert_eq!(c.seq_len(SlotId(3)).unwrap(), 0, "slot 3 untouched");
    }

    /// Pin: drop resets ONLY the target slot's cursor.  Other slots'
    /// cursors AND the underlying K/V bytes are invariant.  The K/V
    /// content half is the structural analogue of Qwen35 A2a's
    /// recurrent-content M4 pin at `qwen35/kv_cache.rs:6949+` (drop
    /// must not zero the buffer — Phase A3c's fork kernel will
    /// re-use the slot's region on next admission).
    #[test]
    fn gemma4_hb_kv_drop_resets_seq_len_for_target_slot_only() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let mut c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4).expect("alloc");
        // Seed every slot.
        c.append_for_seq(SlotId(0), 10).unwrap();
        c.append_for_seq(SlotId(1), 20).unwrap();
        c.append_for_seq(SlotId(2), 30).unwrap();
        c.append_for_seq(SlotId(3), 40).unwrap();
        // Drop slot 2.
        c.drop_seq(SlotId(2)).expect("drop slot 2");
        assert_eq!(c.seq_len(SlotId(0)).unwrap(), 10);
        assert_eq!(c.seq_len(SlotId(1)).unwrap(), 20);
        assert_eq!(c.seq_len(SlotId(2)).unwrap(), 0, "slot 2 reset");
        assert_eq!(c.seq_len(SlotId(3)).unwrap(), 40);
        // Direct cursor read (single buffer per layer; no canonical-
        // vs-per-layer concern like Qwen35's full_attn-vec).
        assert_eq!(c.seq_lens[2], 0, "underlying cursor wiped");
        assert_eq!(c.seq_lens[0], 10, "untouched cursors preserved");
    }

    /// Iter-2.5 M4 analogue for Gemma 4: `drop_seq` must NOT mutate
    /// the underlying K/V packed buffer bytes for the target slot
    /// (or any other slot).  The reasoning mirrors Qwen35's recurrent-
    /// content invariance: the trait surface owns cursor bookkeeping
    /// only; buffer writes are kernel-dispatcher-owned at Phase B4c.
    /// A future regression that zeros bytes inside `drop_seq` would
    /// break the next admission's buffer-reuse correctness — pin it
    /// here so it surfaces at the trait-surface boundary.
    ///
    /// Falsifier: any byte change in slot 0's K or V region after
    /// `drop_seq(SlotId(0))` ⇒ Phase A3a has crossed into kernel-
    /// dispatcher territory.
    #[test]
    fn gemma4_hb_kv_drop_does_not_zero_k_packed_buffer() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 4usize;
        let mut c = alloc_hb_kv_for_layer(&dev, 0, nkv, hd, cap, false, 2).expect("alloc n_seqs=2");

        // Step 1: fill slot 0's K and V packed regions with a
        // deterministic non-zero pattern via direct host write.
        // StorageModeShared makes this a CPU mutation — no kernel
        // dispatch needed, no download/upload helper.  Production
        // write path lives in the kernel dispatcher; this test
        // exercises the contract "drop_seq does NOT touch this
        // buffer" via host-side byte snapshot.
        let slot_packed = nkv * cap * hd;
        {
            let k = c.k_packed.as_mut_slice::<u8>().expect("k_packed u8 mut");
            for (i, b) in k[..slot_packed].iter_mut().enumerate() {
                // Distinct non-zero pattern from H7's slot-1 fill —
                // makes test failures distinguishable from H7's
                // fixture state under shared-binary debugging.
                *b = (((i * 7) % 251) + 1) as u8;
            }
        }
        {
            let v = c.v_packed.as_mut_slice::<u8>().expect("v_packed u8 mut");
            for (i, b) in v[..slot_packed].iter_mut().enumerate() {
                *b = (((i * 11) % 253) + 1) as u8;
            }
        }

        // Step 2: cursor bump on slot 0 (then snapshot bytes BEFORE
        // drop_seq).
        c.append_for_seq(SlotId(0), 2).expect("append slot 0");
        let k_before: Vec<u8> =
            c.k_packed.as_slice::<u8>().expect("k_packed u8")[..slot_packed].to_vec();
        let v_before: Vec<u8> =
            c.v_packed.as_slice::<u8>().expect("v_packed u8")[..slot_packed].to_vec();
        // Fixture sanity: at least one byte is the deterministic
        // pattern, not zero.  Defends against a future regression
        // that breaks `as_mut_slice` for this buffer kind.
        assert!(
            k_before.iter().any(|&b| b != 0),
            "M4-G fixture sanity: deterministic upload must produce \
             non-zero bytes (else test is vacuous)"
        );

        // Step 3: call drop_seq(SlotId(0)).  Per Phase A3a contract,
        // this MUST NOT touch the K/V packed bytes at all.
        c.drop_seq(SlotId(0)).expect("drop slot 0");
        assert_eq!(c.seq_lens[0], 0, "cursor reset");

        // Step 4: snapshot again.
        let k_after: Vec<u8> =
            c.k_packed.as_slice::<u8>().expect("k_packed u8 after")[..slot_packed].to_vec();
        let v_after: Vec<u8> =
            c.v_packed.as_slice::<u8>().expect("v_packed u8 after")[..slot_packed].to_vec();

        // Step 5: byte-by-byte equality.  Any mutation by drop_seq
        // — including partial zero, partial overwrite, in-place
        // swap — surfaces here.
        assert_eq!(
            k_before, k_after,
            "M4-G FALSIFIED: drop_seq mutated k_packed contents for \
             slot 0.  Per Phase A3a contract, drop_seq is cursor-only; \
             buffer-content reset is kernel-dispatcher-owned at Phase \
             B4c.  An in-place zero here would break the next \
             admission's buffer-reuse correctness."
        );
        assert_eq!(
            v_before, v_after,
            "M4-G FALSIFIED: drop_seq mutated v_packed contents for slot 0."
        );
    }

    /// Pin: `fork_seq(src, src)` is a successful no-op per trait spec.
    /// Mirrors `qwen35_hybrid_kv_fork_to_self_is_noop_ok` (qwen35/
    /// kv_cache.rs:7050-7063).
    #[test]
    fn gemma4_hb_kv_fork_to_self_is_noop_ok() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let mut c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4).expect("alloc");
        c.append_for_seq(SlotId(2), 9).unwrap();
        // src == dst — no-op success.
        c.fork_seq(SlotId(2), SlotId(2)).expect("fork self ok");
        // Cursor unchanged.
        assert_eq!(c.seq_len(SlotId(2)).unwrap(), 9);
        // Other slots untouched.
        assert_eq!(c.seq_len(SlotId(0)).unwrap(), 0);
        assert_eq!(c.seq_len(SlotId(1)).unwrap(), 0);
        assert_eq!(c.seq_len(SlotId(3)).unwrap(), 0);
    }

    /// **HISTORICAL** — Phase A3a / iter-2.5 M1 typed-clamp pin
    /// (renamed from `gemma4_hb_kv_fork_cross_slot_returns_capability_unsupported`
    /// at iter-A3c per ADR-040 brief "rename to historical_ if they
    /// were pinning the clamp shape").
    ///
    /// **Prior contract** (A3a → A3c): cross-slot fork returned
    /// `CapabilityUnsupported` with a capability label naming the
    /// deferred Phase A3c kernel arc + dossier R5.
    ///
    /// **Closure (iter-A3c, 2026-05-30)**: real cross-slot fork landed
    /// on `MultiSeqHbKvBuffers::fork_seq` (same-buffer cross-region
    /// memcpy on k_packed / k_norms / v_packed / v_norms + cursor
    /// copy).  This historical test asserts the closure pin: cross-
    /// slot fork must return `Ok(())`.  Full byte-equality +
    /// cursor-copy pin lives at H159 + H163-H165.
    #[test]
    fn historical_gemma4_hb_kv_fork_cross_slot_closure_at_phase_a3c() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let mut c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4).expect("alloc");
        c.append_for_seq(SlotId(0), 7).unwrap();
        // iter-A3c closure: fork now returns Ok(()).
        c.fork_seq(SlotId(0), SlotId(1)).expect(
            "iter-A3c closure: cross-slot fork must return Ok(()) — \
                     was previously CapabilityUnsupported per A3a typed-clamp",
        );
        // Cursor copy invariant (sub-pin for H165).
        assert_eq!(
            c.seq_len(SlotId(1)).unwrap(),
            7,
            "iter-A3c closure: fork_seq must copy src's seq_len to dst"
        );
        assert_eq!(
            c.seq_len(SlotId(0)).unwrap(),
            7,
            "iter-A3c closure: fork_seq must NOT modify src's seq_len (sub-pin for H163)"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // cfa-iter-A5b MAJOR #3 — mixed Gemma 4 layer_types fixture (closes the
    // H9 "verified by code-reading only" gap the codex review surfaced).
    //
    // Production Gemma 4 has heterogeneous `layer_types` (per
    // `src/inference/models/gemma4/model.rs:1250` —
    // `LayerType::{Full, Sliding}` interleaved). The pre-iter-A5b test
    // bank exercised allocation only at uniformly Full or uniformly
    // Sliding; this test walks a synthetic `[Full, Sliding, Full,
    // Sliding]` config and asserts each layer's `MultiSeqHbKvBuffers`
    // honours its layer-type's `is_sliding` flag. A future regression
    // in the per-layer iteration / layer-type plumbing would surface
    // here as a failed assertion rather than silent corruption.
    //
    // NOTE on "Null layers": the codex finding mentioned
    // "Null layers" but `enum LayerType` (`src/serve/config.rs`) only
    // has `Sliding` and `Full` variants — there is no Null/absent
    // variant for Gemma 4. The closest "null" semantics in the
    // codebase is `LoadedModel::Gemma4` MoE-only norm absence
    // (handled at load time, not at KV-alloc time). The test
    // therefore exercises the realistic [Full, Sliding] mixed-vector
    // case + documents the Null-absence inline so a future code
    // reader does not chase a non-existent enum variant.
    // ───────────────────────────────────────────────────────────────────────

    /// **cfa-iter-A5b MAJOR #3** — mixed Gemma 4 `layer_types` allocator
    /// fixture. Walks `[Full, Sliding, Full, Sliding]` and verifies
    /// each layer's buffer carries the right `is_sliding` flag, the
    /// right capacity, and the right per-layer byte count.
    ///
    /// Falsifier (any one ⇒ mixed-layer allocator broken):
    /// 1. The per-layer `alloc_hb_kv_for_layer` call panics or errors.
    /// 2. A `Full` layer's allocated buffer reports `is_sliding == true`.
    /// 3. A `Sliding` layer's allocated buffer reports `is_sliding == false`.
    /// 4. A Sliding-layer capacity does not match `sliding_window`.
    /// 5. A Full-layer capacity does not match `max_seq_len`.
    /// 6. Per-layer seq_lens cursors are not zero-initialised.
    /// 7. `n_seqs` propagated from the call site differs across layers.
    #[test]
    fn a3a_mixed_layer_alloc_full_sliding_byte_isolation() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        use crate::serve::config::LayerType;
        // Synthetic 4-layer mixed config: alternating Full / Sliding,
        // matching the production Gemma 4 27B-A4B pattern
        // (`gemma4/model.rs:1250` — every Nth layer is Full).
        let layer_types: Vec<LayerType> = vec![
            LayerType::Full,
            LayerType::Sliding,
            LayerType::Full,
            LayerType::Sliding,
        ];
        let n_seqs: u32 = 2;
        let nkv: usize = 2;
        let hd: usize = 256;
        let max_seq_len: usize = 32;
        let sliding_window: usize = 8;

        for (layer_idx, lt) in layer_types.iter().enumerate() {
            // ADR-040 §3.5 iter-A5c (cfa-A5b MAJOR #3): route through the
            // PRODUCTION helper that `gemma4/model.rs:1247-1257` also
            // uses. A future branch-swap of Full/Sliding in
            // `layer_type_to_alloc_params` would surface here as wrong
            // capacity OR wrong is_ring — load-bearing falsifier.
            let (is_ring, cap) =
                super::layer_type_to_alloc_params(*lt, sliding_window, max_seq_len);
            let buf = alloc_hb_kv_for_layer(&dev, layer_idx, nkv, hd, cap, is_ring, n_seqs)
                .unwrap_or_else(|e| {
                    panic!("L{layer_idx} ({lt:?}): alloc_hb_kv_for_layer must succeed; got {e}")
                });

            // Falsifier 2/3 — is_sliding flag matches layer type.
            assert_eq!(
                buf.is_sliding, is_ring,
                "L{layer_idx} ({lt:?}): is_sliding={} does NOT match \
                 expected={is_ring} (layer-type plumbing broken)",
                buf.is_sliding,
            );

            // Falsifier 4/5 — capacity matches layer-type-specific cap.
            let cap_label = if is_ring {
                "sliding_window"
            } else {
                "max_seq_len"
            };
            assert_eq!(
                buf.capacity, cap,
                "L{layer_idx} ({lt:?}): capacity={} does NOT match \
                 expected={cap} ({cap_label})",
                buf.capacity,
            );

            // Falsifier 6 — per-seq cursors zero-initialised.
            assert_eq!(
                buf.seq_lens.len(),
                n_seqs as usize,
                "L{layer_idx}: seq_lens.len() must equal n_seqs"
            );
            assert!(
                buf.seq_lens.iter().all(|&x| x == 0),
                "L{layer_idx}: seq_lens must be zero-initialised"
            );

            // Falsifier 7 — n_seqs propagated.
            assert_eq!(
                buf.n_seqs, n_seqs,
                "L{layer_idx}: n_seqs must propagate from call site"
            );

            // Byte-count cross-check: Full layers cap=32 vs Sliding cap=8
            // ⇒ Full byte count = 4× Sliding byte count for same n_seqs
            // /nkv/hd. We don't assert the ratio inline (it differs per
            // layer) but pin the per-layer byte count against the formula
            // for sanity:
            let expected_packed_bytes = (n_seqs as usize) * nkv * cap * hd;
            assert_eq!(
                buf.k_packed.byte_len(),
                expected_packed_bytes,
                "L{layer_idx} ({lt:?}): k_packed byte_len mismatch",
            );
            assert_eq!(
                buf.v_packed.byte_len(),
                expected_packed_bytes,
                "L{layer_idx} ({lt:?}): v_packed byte_len mismatch",
            );
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // cfa-iter-A5c (was-A5b) MAJOR #3 — closes the codex finding that the
    // prior `a3a_mixed_layer_alloc_full_sliding_byte_isolation` test only
    // verified that `alloc_hb_kv_for_layer` honoured its boolean argument
    // (not that production's `LayerType::{Full, Sliding}` → `(is_ring,
    // capacity)` mapping was correct). The iter-A5c fix:
    //   (a) extracts the mapping into `layer_type_to_alloc_params` (pure fn),
    //   (b) routes BOTH the production `gemma4/model.rs:1247-1257` call site
    //       AND the mixed-layer test above through it,
    //   (c) adds an explicit branch-swap falsifier test below.
    // A future swap of the two arms in `layer_type_to_alloc_params` makes
    // the test below fail with a clear "Full mapped to ring buffer with
    // sliding_window capacity" message; the mixed-layer test above ALSO
    // fails because alloc_hb_kv_for_layer would receive swapped
    // (is_ring, cap) per layer.
    // ───────────────────────────────────────────────────────────────────────

    /// **cfa-iter-A5c MAJOR #3** — explicit branch-swap falsifier for
    /// `layer_type_to_alloc_params`. Pins the production mapping:
    /// - `LayerType::Sliding` → `(is_ring=true, capacity=sliding_window)`.
    /// - `LayerType::Full` → `(is_ring=false, capacity=max_position_embeddings)`.
    ///
    /// Each assertion is a separate falsifier so the failure message
    /// names the exact axis (is_ring vs capacity) that drifted.
    #[test]
    fn a5c_layer_type_to_alloc_params_mapping_pinned() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use crate::serve::config::LayerType;
        let sliding_window: usize = 4_096;
        let max_pos: usize = 131_072;

        let (is_ring_s, cap_s) =
            super::layer_type_to_alloc_params(LayerType::Sliding, sliding_window, max_pos);
        assert!(is_ring_s, "Sliding MUST map to is_ring=true (ring buffer)");
        assert_eq!(
            cap_s, sliding_window,
            "Sliding MUST map to capacity=sliding_window={sliding_window}"
        );

        let (is_ring_f, cap_f) =
            super::layer_type_to_alloc_params(LayerType::Full, sliding_window, max_pos);
        assert!(!is_ring_f, "Full MUST map to is_ring=false (linear buffer)");
        assert_eq!(
            cap_f, max_pos,
            "Full MUST map to capacity=max_position_embeddings={max_pos}"
        );

        // Cross-arm sanity: a Full layer NEVER takes the sliding_window
        // capacity and a Sliding layer NEVER takes the max_pos capacity.
        assert_ne!(
            cap_s, cap_f,
            "Sliding + Full MUST yield distinct capacities in a realistic \
             production config (sliding_window != max_position_embeddings); \
             a swap of the two arms in `layer_type_to_alloc_params` would \
             make these equal and break the assertion above"
        );
    }

    /// **cfa-iter-A5c MAJOR #3** — production-path cross-check: the
    /// mixed-layer fixture (above) walks through
    /// `super::layer_type_to_alloc_params`; this test ALSO walks the
    /// production call site at `gemma4/model.rs:1247-1257` through the
    /// SAME helper (it routes through it as of this iter), proving the
    /// two paths cannot diverge.
    ///
    /// The test instantiates the helper with the canonical Gemma 4 27B
    /// sliding_window=1024 + max_position_embeddings=131_072 and asserts
    /// the per-layer-type capacity is what `model.rs` will see — pinning
    /// the contract that future allocator changes must continue to honour.
    #[test]
    fn a5c_production_gemma4_model_routes_through_layer_type_helper() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use crate::serve::config::LayerType;
        let sliding_window: usize = 1_024;
        let max_pos: usize = 131_072;

        // Sliding layer — production must allocate a ring buffer of
        // capacity sliding_window (per gemma4/model.rs:1253-1257
        // pre-iter-A5c logic; post-iter-A5c routes through this helper).
        let (is_ring, cap) =
            super::layer_type_to_alloc_params(LayerType::Sliding, sliding_window, max_pos);
        assert!(
            is_ring && cap == sliding_window,
            "production Sliding layer alloc shape: (ring=true, cap=1024); \
             got (ring={is_ring}, cap={cap}) — gemma4/model.rs:1247-1257 \
             would allocate the wrong shape if this mapping drifts"
        );

        // Full layer — production must allocate a linear buffer of
        // capacity max_position_embeddings.
        let (is_ring, cap) =
            super::layer_type_to_alloc_params(LayerType::Full, sliding_window, max_pos);
        assert!(
            !is_ring && cap == max_pos,
            "production Full layer alloc shape: (ring=false, cap=131072); \
             got (ring={is_ring}, cap={cap})"
        );
    }

    /// ADR-040 Phase F `iter-F-kvcap` — falsifier for the per-slot allocator
    /// helper [`super::layer_type_to_alloc_params_per_slot`]. Pins the
    /// continuous-batching mapping:
    /// - `Sliding` → `(ring=true, sliding_window)` — UNCHANGED vs single-seq
    ///   (ring window is per-slot-independent; must NOT be divided).
    /// - `Full` → `(ring=false, max_position_embeddings + 1)` for every slot;
    ///   the final physical position is a guard and is never logical context.
    /// Also pins the load-bearing invariant that adding slots never reduces an
    /// agent's logical context capacity. Physical residency is accounted by
    /// high-water usage at admission time.
    #[test]
    fn iter_f_kvcap_per_slot_alloc_params_mapping_pinned() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use crate::serve::config::LayerType;
        let sliding_window: usize = 1_024;
        let max_pos: usize = 262_144;

        // Sliding is per-slot-independent: identical for any max_slots.
        for &n in &[1usize, 2, 4, 8] {
            let (is_ring, cap) = super::layer_type_to_alloc_params_per_slot(
                LayerType::Sliding,
                sliding_window,
                max_pos,
                n,
            );
            assert!(is_ring, "Sliding MUST stay a ring buffer (max_slots={n})");
            assert_eq!(
                cap, sliding_window,
                "Sliding capacity MUST stay sliding_window regardless of \
                 max_slots — dividing the ring window would corrupt its \
                 semantics (max_slots={n})"
            );
        }

        // Full logical capacity is independent of the number of slots. The
        // slot-aware storage stride has one guard position beyond it.
        let expected_storage_capacity = max_pos + 1;
        for &n in &[1usize, 2, 4, 8] {
            let (_, per_slot) = super::layer_type_to_alloc_params_per_slot(
                LayerType::Full,
                sliding_window,
                max_pos,
                n,
            );
            assert_eq!(
                per_slot, expected_storage_capacity,
                "slot count must not shrink logical Full KV capacity or remove \
                 its physical guard: max_slots={n}, got {per_slot}, \
                 expected {expected_storage_capacity}"
            );
        }
    }

    /// **cfa-iter-A5b MAJOR #3** — Null-layer absence documentation
    /// pin. `LayerType` has exactly two variants (`Sliding`, `Full`);
    /// any future addition of a `Null`-like variant MUST land
    /// alongside an extension of the mixed-layer test above so the
    /// allocator's per-layer dispatch surface is exercised
    /// exhaustively. This test pins the current variant set; the
    /// `match` exhaustiveness check is the load-bearing assertion.
    #[test]
    fn a3a_layer_type_variants_are_full_and_sliding_only() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use crate::serve::config::LayerType;
        // Exhaustive match on every variant — if a Null / Absent
        // variant lands without test coverage, this match will fail
        // to compile and the operator MUST extend
        // `a3a_mixed_layer_alloc_full_sliding_byte_isolation`.
        fn name(lt: LayerType) -> &'static str {
            match lt {
                LayerType::Full => "Full",
                LayerType::Sliding => "Sliding",
            }
        }
        assert_eq!(name(LayerType::Full), "Full");
        assert_eq!(name(LayerType::Sliding), "Sliding");
    }

    // ───────────────────────────────────────────────────────────────────────
    // ADR-040 Phase A3b iter-1 — multi-seq lift hypotheses + clamp pins for
    // HybridKvBuffers (FULL lift), DenseKvBuffers + MlxKvCache (typed
    // clamps).  See `docs/research/adr040-kv-cache-lift-dossier-2026-05-23.md`
    // §Gemma 4 KV variants + R3 + H10 falsification for grounding.
    //
    // Hypothesis order:
    //   H11 (HybridKvBuffers byte-scale at n_seqs=4 — 4× per buffer)
    //   H12 (HybridKvBuffers per-slot byte isolation under host-side writes)
    //   H13 (HybridKvBuffers cursor independence — slot 0 advance, slot 1
    //        unchanged)
    //   H14 (HybridKvBuffers optional xlen BF16 K/V coexists with U8 V +
    //        F32 v_norms in the n_seqs lift)
    //   H15 (DenseKvBuffers typed clamp — slot > 0 returns
    //        SlotOutOfRange{slot, max_slots=1})
    //   H16 (MlxKvCache typed clamp — same shape)
    // ───────────────────────────────────────────────────────────────────────

    /// **H11** — `alloc_multi_seq_hybrid_kv_for_layer(.., n_seqs=4)`
    /// produces buffers byte-scaled exactly 4× the n_seqs=1 baseline
    /// across K (F16), V packed (U8), and V norms (F32).  Mirrors H6
    /// for HbKvBuffers; H11 is the HybridKvBuffers analogue.
    ///
    /// Falsifier (any one ⇒ H11 broken):
    /// 1. Allocation at n_seqs=4 panics or errors.
    /// 2. K F16 at n_seqs=4 is NOT exactly 4× the n_seqs=1 baseline.
    /// 3. V packed at n_seqs=4 is NOT exactly 4×.
    /// 4. V norms at n_seqs=4 is NOT exactly 4×.
    /// 5. `seq_lens.len() != n_seqs`.
    #[test]
    fn h11_multi_seq_hybrid_kv_n_seqs_4_byte_scale() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        // Mirror H6's fixture shapes for cross-test grep parity.
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;

        let baseline = alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, nkv, hd, cap, false, 1)
            .expect("H11: alloc at n_seqs=1");
        let lifted = alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, nkv, hd, cap, false, 4)
            .expect("H11: alloc at n_seqs=4");

        assert_eq!(baseline.n_seqs, 1, "H11: baseline n_seqs=1");
        assert_eq!(lifted.n_seqs, 4, "H11: lifted n_seqs=4");

        // Falsifier 2: K F16 4×.
        assert_eq!(
            lifted.k.byte_len(),
            baseline.k.byte_len() * 4,
            "H11 FALSIFIED: F16 K does not scale 4× ({} != {} * 4 = {})",
            lifted.k.byte_len(),
            baseline.k.byte_len(),
            baseline.k.byte_len() * 4
        );
        // Falsifier 3: V packed 4×.
        assert_eq!(
            lifted.v_packed.byte_len(),
            baseline.v_packed.byte_len() * 4,
            "H11 FALSIFIED: V packed does not scale 4× ({} != {} * 4 = {})",
            lifted.v_packed.byte_len(),
            baseline.v_packed.byte_len(),
            baseline.v_packed.byte_len() * 4
        );
        // Falsifier 4: V norms 4×.
        assert_eq!(
            lifted.v_norms.byte_len(),
            baseline.v_norms.byte_len() * 4,
            "H11 FALSIFIED: V norms does not scale 4× ({} != {})",
            lifted.v_norms.byte_len(),
            baseline.v_norms.byte_len() * 4
        );

        // Falsifier 5: per-seq cursor vec length tracks n_seqs.
        assert_eq!(baseline.seq_lens.len(), 1, "H11: baseline seq_lens.len()");
        assert_eq!(lifted.seq_lens.len(), 4, "H11: lifted seq_lens.len()");
        assert!(
            baseline.seq_lens.iter().all(|&x| x == 0),
            "H11: baseline seq_lens zero-init"
        );
        assert!(
            lifted.seq_lens.iter().all(|&x| x == 0),
            "H11: lifted seq_lens zero-init"
        );

        // Shape pin: n_seqs OUTERMOST on every buffer (4-D `[n_seqs,
        // nkv, cap, hd]` for K / V packed / V norms when
        // norms_per_pos==1 the trailing axis is 1).  M5-equivalent
        // shape proof.
        for (name, b) in [
            ("k", &lifted.k),
            ("v_packed", &lifted.v_packed),
            ("v_norms", &lifted.v_norms),
        ] {
            let s = b.shape().to_vec();
            assert_eq!(s.len(), 4, "H11 M5: {name} must be 4-D; got {:?}", s);
            assert_eq!(
                s[0], 4,
                "H11 M5 FALSIFIED: {name} shape[0] must be n_seqs=4 (n_seqs landed \
                 on wrong axis); got {:?}",
                s
            );
        }

        // iter-A3b iter-1.5 (codex /cfa request_changes major): pin the
        // EXACT total byte formula at n_seqs=4 — not just 4× scale.  This
        // catches any regression where a per-buffer formula was 4× but the
        // composition (K + V_packed + V_norms + optional xlen) was wrong.
        //
        // Formula (default path, HF2Q_FULL_F16_KV unset, HF2Q_DFLASH_XLEN_SDPA unset):
        //   norms_per_pos = max(hd/256, 1)             = 1  (hd=256)
        //   k_bytes       = n * nkv * cap * hd * 2     = 4 * 2 * 8 * 256 * 2 = 32768
        //   v_packed_bytes= n * nkv * cap * hd * 1     = 4 * 2 * 8 * 256     = 16384
        //   v_norms_bytes = n * nkv * cap * 1 * 4      = 4 * 2 * 8 * 1 * 4   = 256
        //   xlen_bytes    = 0 (None)
        //   TOTAL         = 49408 bytes
        let expected_k_bytes = 4usize * nkv * cap * hd * 2; // 32768
        let expected_v_packed_bytes = 4usize * nkv * cap * hd; // 16384
        let expected_v_norms_bytes = 4usize * nkv * cap * 1 * 4; // 256
        let expected_total = expected_k_bytes + expected_v_packed_bytes + expected_v_norms_bytes;
        assert_eq!(
            lifted.k.byte_len(),
            expected_k_bytes,
            "H11 EXACT FORMULA FALSIFIED: K F16 bytes ({}) != n*nkv*cap*hd*2 ({})",
            lifted.k.byte_len(),
            expected_k_bytes
        );
        assert_eq!(
            lifted.v_packed.byte_len(),
            expected_v_packed_bytes,
            "H11 EXACT FORMULA FALSIFIED: V packed U8 bytes ({}) != n*nkv*cap*hd ({})",
            lifted.v_packed.byte_len(),
            expected_v_packed_bytes
        );
        assert_eq!(
            lifted.v_norms.byte_len(),
            expected_v_norms_bytes,
            "H11 EXACT FORMULA FALSIFIED: V norms F32 bytes ({}) != n*nkv*cap*1*4 ({})",
            lifted.v_norms.byte_len(),
            expected_v_norms_bytes
        );
        let actual_total =
            lifted.k.byte_len() + lifted.v_packed.byte_len() + lifted.v_norms.byte_len();
        assert_eq!(
            actual_total, expected_total,
            "H11 EXACT FORMULA FALSIFIED: composition K+V_packed+V_norms = {} != {}",
            actual_total, expected_total
        );
        // Concrete pin so any future refactor that changes per-buffer
        // shape OR composition breaks this test rather than silently
        // mis-allocating production memory.
        assert_eq!(
            actual_total, 49408,
            "H11 EXACT FORMULA FALSIFIED at concrete value: expected 49408 bytes \
             for n_seqs=4 nkv=2 cap=8 hd=256 default (no full-F16, no xlen), got {}",
            actual_total
        );
        // xlen=None is the default path: confirm the optional fields are
        // genuinely absent (the formula above assumed 0 xlen bytes).
        assert!(
            lifted.bf16_xlen_k.is_none() && lifted.bf16_xlen_v.is_none(),
            "H11 EXACT FORMULA: xlen buffers must be None on default path"
        );
    }

    /// **H11r** (iter-A3b iter-1.5) — realistic Gemma 4 sliding-attention
    /// shape pin.  Codex /cfa on iter-1 flagged H11's cap=8 fixture as
    /// non-representative.  This test uses canonical Gemma 4 27B sliding
    /// shape (nkv=8, hd=256) with a moderate cap=512 (truncated from prod
    /// 2048 to keep test alloc under 12 MB) and pins the EXACT total byte
    /// count, proving the formula scales correctly at production-class
    /// fan-outs not just the H11 tiny fixture.
    #[test]
    fn h11r_multi_seq_hybrid_kv_realistic_sliding_shape_byte_formula() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        // Canonical Gemma 4 27B sliding shape: 8 KV heads × 256 head_dim.
        // cap=512 keeps total alloc at ~10 MB (cap=2048 prod would be ~40 MB).
        let nkv = 8usize;
        let hd = 256usize;
        let cap = 512usize;
        let n_seqs = 4u32;
        let lifted = alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, nkv, hd, cap, false, n_seqs)
            .expect("H11r: realistic shape alloc");
        let n = n_seqs as usize;
        let expected_k = n * nkv * cap * hd * 2; // 4*8*512*256*2 = 8_388_608
        let expected_v = n * nkv * cap * hd; // 4*8*512*256   = 4_194_304
        let expected_norms = n * nkv * cap * 1 * 4; // 4*8*512*1*4   = 65_536
        let expected_total = expected_k + expected_v + expected_norms;
        assert_eq!(lifted.k.byte_len(), expected_k, "H11r: K F16");
        assert_eq!(lifted.v_packed.byte_len(), expected_v, "H11r: V packed U8");
        assert_eq!(
            lifted.v_norms.byte_len(),
            expected_norms,
            "H11r: V norms F32"
        );
        let actual = lifted.k.byte_len() + lifted.v_packed.byte_len() + lifted.v_norms.byte_len();
        assert_eq!(actual, expected_total, "H11r: composition");
        assert_eq!(
            actual, 12_648_448,
            "H11r CONCRETE FALSIFIED: realistic Gemma 4 sliding shape n=4 nkv=8 \
             hd=256 cap=512 should sum to 12_648_448 bytes (~12 MB); got {}",
            actual
        );
    }

    /// **H12** — `MultiSeqHybridKvBuffers` per-slot byte isolation:
    /// host-side write of a deterministic non-zero pattern into slot
    /// 0's K (F16) + V packed (U8) + V norms (F32) regions leaves
    /// slot 1's bytes byte-identical.  Mirrors H7 for HbKvBuffers.
    ///
    /// At Phase A3b iter-1 the kernel-dispatcher slot-offset routing
    /// is not yet wired (Phase B4c scope); this test verifies isolation
    /// at the **buffer-region level** via the per-slot byte-offset
    /// formula `slot.0 * (nkv*cap*hd*elem_bytes)`.  The cursor advance
    /// is a no-op against the underlying bytes — the test pins that
    /// no cross-slot byte mutation occurs.
    ///
    /// Falsifier: any byte change in slot 1's K / V packed / V norms
    /// region after writing to slot 0's region ⇒ H12 broken; the
    /// per-slot byte-offset formula does not produce disjoint regions.
    #[test]
    fn h12_multi_seq_hybrid_kv_per_slot_byte_isolation() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 4usize;
        let mut cache = alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, nkv, hd, cap, false, 2)
            .expect("H12: alloc n_seqs=2");
        assert_eq!(cache.n_seqs, 2);

        // Per-slot region sizes:
        //   K (F16):   slot bytes = nkv * cap * hd * 2
        //   V packed:  slot bytes = nkv * cap * hd (U8)
        //   V norms:   slot bytes = nkv * cap * 1 * 4 (norms_per_pos=1
        //              at hd=256; 4 = sizeof(f32))
        let slot_k_bytes = nkv * cap * hd * 2;
        let slot_v_bytes = nkv * cap * hd;
        let slot_vn_bytes = nkv * cap * 1 * 4;

        // Sanity: total bytes match `2 * slot_bytes` for each buffer.
        assert_eq!(
            cache.k.byte_len(),
            2 * slot_k_bytes,
            "H12 fixture sanity: K total = 2 * slot_k_bytes"
        );
        assert_eq!(
            cache.v_packed.byte_len(),
            2 * slot_v_bytes,
            "H12 fixture sanity: V packed total = 2 * slot_v_bytes"
        );
        assert_eq!(
            cache.v_norms.byte_len(),
            2 * slot_vn_bytes,
            "H12 fixture sanity: V norms total = 2 * slot_vn_bytes"
        );

        // Overwrite-backed non-ring tails are not initialized by allocation.
        // Seed the peer slot before observing it so isolation is explicit.
        cache.k.as_mut_slice::<u8>().expect("seed K peer")[slot_k_bytes..2 * slot_k_bytes]
            .fill(0xA7);
        cache.v_packed.as_mut_slice::<u8>().expect("seed V peer")[slot_v_bytes..2 * slot_v_bytes]
            .fill(0xB8);
        cache
            .v_norms
            .as_mut_slice::<f32>()
            .expect("seed norms peer")[(nkv * cap)..2 * (nkv * cap)]
            .fill(42.25);

        // Write deterministic non-zero pattern into slot 0's K
        // (interpret as u8 bytes for fixture simplicity — the kernel
        // writes F16 but byte-level isolation is what we're pinning).
        {
            let k_slice = cache.k.as_mut_slice::<u8>().expect("k F16 as u8 mut");
            for (i, b) in k_slice[..slot_k_bytes].iter_mut().enumerate() {
                *b = (((i * 7) % 251) + 1) as u8;
            }
        }
        {
            let v_slice = cache
                .v_packed
                .as_mut_slice::<u8>()
                .expect("v_packed u8 mut");
            for (i, b) in v_slice[..slot_v_bytes].iter_mut().enumerate() {
                *b = (((i * 11) % 253) + 1) as u8;
            }
        }
        {
            let vn_slice = cache
                .v_norms
                .as_mut_slice::<f32>()
                .expect("v_norms f32 mut");
            let slot_vn_f32 = nkv * cap * 1; // norms_per_pos=1
            for (i, f) in vn_slice[..slot_vn_f32].iter_mut().enumerate() {
                *f = (i as f32) * 0.123_45;
            }
        }

        // Snapshot slot 1's regions.
        let k_slot1_before: Vec<u8> =
            cache.k.as_slice::<u8>().expect("k F16 as u8")[slot_k_bytes..2 * slot_k_bytes].to_vec();
        let v_slot1_before: Vec<u8> = cache.v_packed.as_slice::<u8>().expect("v_packed u8")
            [slot_v_bytes..2 * slot_v_bytes]
            .to_vec();
        let vn_slot1_before: Vec<f32> = cache.v_norms.as_slice::<f32>().expect("v_norms f32")
            [(nkv * cap * 1)..2 * (nkv * cap * 1)]
            .to_vec();

        // Sanity: slot 1 contains the explicit sentinel.
        assert!(
            k_slot1_before.iter().all(|&b| b == 0xA7),
            "H12 fixture sanity: slot 1 K sentinel"
        );
        assert!(
            v_slot1_before.iter().all(|&b| b == 0xB8),
            "H12 fixture sanity: slot 1 V packed sentinel"
        );
        assert!(
            vn_slot1_before.iter().all(|&f| f == 42.25),
            "H12 fixture sanity: slot 1 V norms sentinel"
        );

        // A3b iter-1 cursor advance on slot 0 (no buffer mutation).
        cache
            .append_for_seq(SlotId(0), 3)
            .expect("H12: append slot 0");
        assert_eq!(cache.seq_lens[0], 3);
        assert_eq!(cache.seq_lens[1], 0);

        // H12 falsifier: slot 1's bytes must be byte-identical.
        let k_slot1_after: Vec<u8> =
            cache.k.as_slice::<u8>().expect("k F16 as u8")[slot_k_bytes..2 * slot_k_bytes].to_vec();
        let v_slot1_after: Vec<u8> = cache.v_packed.as_slice::<u8>().expect("v_packed u8")
            [slot_v_bytes..2 * slot_v_bytes]
            .to_vec();
        let vn_slot1_after: Vec<f32> = cache.v_norms.as_slice::<f32>().expect("v_norms f32")
            [(nkv * cap * 1)..2 * (nkv * cap * 1)]
            .to_vec();

        assert_eq!(
            k_slot1_before, k_slot1_after,
            "H12 FALSIFIED: slot 1 K bytes changed after slot-0 write"
        );
        assert_eq!(
            v_slot1_before, v_slot1_after,
            "H12 FALSIFIED: slot 1 V packed bytes changed"
        );
        assert_eq!(
            vn_slot1_before, vn_slot1_after,
            "H12 FALSIFIED: slot 1 V norms changed"
        );
    }

    /// **H13** — `MultiSeqHybridKvBuffers` cursor independence:
    /// `append_for_seq(SlotId(0), N)` advances slot 0's cursor and
    /// leaves slot 1's cursor at 0.  Mirrors H6 for HbKvBuffers
    /// at the trait-surface level (gemma4_hb_kv_append_advances_
    /// target_slot_only).
    #[test]
    fn h13_multi_seq_hybrid_kv_cursor_independence() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        let mut c =
            alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, 2, 256, 8, false, 4).expect("alloc");
        // All slots start at 0.
        for s in 0..4 {
            assert_eq!(c.seq_len(SlotId(s)).expect("seq_len in range"), 0);
        }
        c.append_for_seq(SlotId(0), 5).expect("append slot 0");
        c.append_for_seq(SlotId(2), 3).expect("append slot 2");
        // Falsifier: any other slot ≠ 0 after the targeted appends.
        assert_eq!(c.seq_len(SlotId(0)).unwrap(), 5);
        assert_eq!(
            c.seq_len(SlotId(1)).unwrap(),
            0,
            "H13 FALSIFIED: slot 1 cursor touched by slot 0/2 append"
        );
        assert_eq!(c.seq_len(SlotId(2)).unwrap(), 3);
        assert_eq!(
            c.seq_len(SlotId(3)).unwrap(),
            0,
            "H13 FALSIFIED: slot 3 cursor touched by slot 0/2 append"
        );
        // Drop slot 0 — slots 2/1/3 cursors unchanged.
        c.drop_seq(SlotId(0)).expect("drop slot 0");
        assert_eq!(c.seq_len(SlotId(0)).unwrap(), 0, "H13: slot 0 reset");
        assert_eq!(
            c.seq_len(SlotId(2)).unwrap(),
            3,
            "H13: slot 2 preserved through slot 0 drop"
        );
    }

    /// **H14** — `MultiSeqHybridKvBuffers` optional BF16 xlen K/V
    /// coexists with the U8 V packed + F32 v_norms layout under the
    /// n_seqs lift.  Pinned with both env-gate states:
    ///   (a) `HF2Q_DFLASH_XLEN_SDPA=1` → `bf16_xlen_k`/`_v` are
    ///       `Some(_)` with 4-D shape `[n_seqs, nkv, cap, hd]`
    ///       and byte-len = `n * nkv * cap * hd * 2`.
    ///   (b) `HF2Q_DFLASH_XLEN_SDPA` unset → both fields are `None`.
    ///
    /// Same env-gate discipline the legacy `alloc_hybrid_kv_for_layer`
    /// follows; pinning that the lift inherits the gate behaviour
    /// at n_seqs=N.
    #[test]
    fn h14_multi_seq_hybrid_kv_xlen_optional_coexists_with_u8_v() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        // (a) xlen ON.
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::set_var("HF2Q_DFLASH_XLEN_SDPA", "1");
        let xlen_on = alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, 2, 256, 4, false, 3)
            .expect("H14: alloc xlen on");
        assert_eq!(xlen_on.n_seqs, 3);
        assert!(
            xlen_on.bf16_xlen_k.is_some(),
            "H14 FALSIFIED: xlen K must be Some when env gate set"
        );
        assert!(
            xlen_on.bf16_xlen_v.is_some(),
            "H14 FALSIFIED: xlen V must be Some when env gate set"
        );
        // Byte counts and shapes pin the n_seqs lift on the BF16 path.
        let bk = xlen_on.bf16_xlen_k.as_ref().unwrap();
        let bv = xlen_on.bf16_xlen_v.as_ref().unwrap();
        let expected_xlen_bytes = 3 * 2 * 4 * 256 * 2; // n_seqs * nkv * cap * hd * 2 (BF16)
        assert_eq!(
            bk.byte_len(),
            expected_xlen_bytes,
            "H14 FALSIFIED: xlen K bytes wrong ({} != {})",
            bk.byte_len(),
            expected_xlen_bytes
        );
        assert_eq!(
            bv.byte_len(),
            expected_xlen_bytes,
            "H14 FALSIFIED: xlen V bytes wrong"
        );
        // Shape: n_seqs OUTERMOST on xlen buffers too.
        assert_eq!(
            bk.shape(),
            &[3, 2, 4, 256],
            "H14: xlen K shape n_seqs outermost"
        );
        assert_eq!(
            bv.shape(),
            &[3, 2, 4, 256],
            "H14: xlen V shape n_seqs outermost"
        );

        // U8 V packed + F32 v_norms coexist unchanged.
        // V packed: n * nkv * cap * hd (U8) = 3 * 2 * 4 * 256 = 6144.
        assert_eq!(
            xlen_on.v_packed.byte_len(),
            3 * 2 * 4 * 256,
            "H14: U8 V packed coexists with xlen"
        );
        // V norms: n * nkv * cap * 1 * 4 (F32 at hd=256 → norms_per_pos=1) = 96.
        assert_eq!(
            xlen_on.v_norms.byte_len(),
            3 * 2 * 4 * 1 * 4,
            "H14: F32 V norms coexists with xlen"
        );

        // (b) xlen OFF.
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        let xlen_off = alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, 2, 256, 4, false, 3)
            .expect("H14: alloc xlen off");
        assert!(
            xlen_off.bf16_xlen_k.is_none(),
            "H14 FALSIFIED: xlen K must be None when env gate unset"
        );
        assert!(
            xlen_off.bf16_xlen_v.is_none(),
            "H14 FALSIFIED: xlen V must be None when env gate unset"
        );
        // U8 V + F32 v_norms still allocated identically.
        assert_eq!(xlen_off.v_packed.byte_len(), 3 * 2 * 4 * 256);
        assert_eq!(xlen_off.v_norms.byte_len(), 3 * 2 * 4 * 1 * 4);
    }

    /// **H15** — DenseKvBuffers typed clamp.  `slot_count() == 1`;
    /// any operation against `SlotId(slot > 0)` returns
    /// `SlotOutOfRange { slot, max_slots: 1 }`.  Append/drop at
    /// the clamp boundary itself (slot 0) returns
    /// `CapabilityUnsupported` naming iter-A3b-2 (so the typed
    /// error reaches operators rather than silently no-op'ing
    /// the bookkeeping).
    #[test]
    fn h15_dense_kv_buffers_typed_clamp_slot_count_one() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use crate::serve::multi_seq_kv::MultiSeqKvCache;
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2;
        let cap = 8;
        let hd = 256;
        let k = dev
            .alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd])
            .unwrap();
        let v = dev
            .alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd])
            .unwrap();
        let mut buf = DenseKvBuffers {
            k,
            v,
            capacity: cap,
            is_sliding: false,
            dtype: DType::F32,
        };

        // slot_count == 1.
        assert_eq!(
            buf.slot_count(),
            1,
            "H15 FALSIFIED: DenseKvBuffers slot_count must be 1"
        );
        assert_eq!(buf.layout(), MultiSeqLayout::SeparateSlots);

        // seq_len(SlotId(0)) returns Ok(0).
        assert_eq!(buf.seq_len(SlotId(0)).unwrap(), 0);

        // seq_len(SlotId(1)) returns SlotOutOfRange.
        let err = buf.seq_len(SlotId(1)).expect_err("slot 1 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(1),
                max_slots: 1
            },
            "H15 FALSIFIED: SlotOutOfRange shape wrong; got {err:?}"
        );

        // append_for_seq(SlotId(2)) returns SlotOutOfRange (bounds first).
        let err = buf
            .append_for_seq(SlotId(2), 1)
            .expect_err("append slot 2 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(2),
                max_slots: 1
            }
        );

        // append_for_seq(SlotId(0)) returns CapabilityUnsupported naming iter-A3b-2.
        let err = buf
            .append_for_seq(SlotId(0), 1)
            .expect_err("append clamped to iter-A3b-2");
        match err {
            MultiSeqError::CapabilityUnsupported { capability } => {
                assert!(
                    capability.contains("DenseKvBuffers"),
                    "label must name struct: {capability}"
                );
                assert!(
                    capability.contains("A3b iter-2"),
                    "label must name deferral: {capability}"
                );
            }
            other => panic!("H15: expected CapabilityUnsupported; got {other:?}"),
        }

        // drop_seq same shape.
        let err = buf.drop_seq(SlotId(5)).expect_err("drop slot 5 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(5),
                max_slots: 1
            }
        );
        let err = buf.drop_seq(SlotId(0)).expect_err("drop clamped");
        assert!(matches!(err, MultiSeqError::CapabilityUnsupported { .. }));

        // fork_seq(SlotId(0), SlotId(0)) is the only valid combo — Ok(()) no-op.
        buf.fork_seq(SlotId(0), SlotId(0))
            .expect("self-fork ok no-op");
        // fork_seq src OOR.
        let err = buf
            .fork_seq(SlotId(1), SlotId(0))
            .expect_err("fork src OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(1),
                max_slots: 1
            }
        );
        // fork_seq dst OOR.
        let err = buf
            .fork_seq(SlotId(0), SlotId(2))
            .expect_err("fork dst OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(2),
                max_slots: 1
            }
        );
    }

    /// **H16** — MlxKvCache typed clamp.  Same shape as H15:
    /// `slot_count() == 1`; slot > 0 returns SlotOutOfRange;
    /// in-bounds slot operations return CapabilityUnsupported
    /// naming iter-A3b-3.  `seq_len(SlotId(0))` reports the
    /// legacy single-seq cursor (`self.seq_len as u32`).
    #[test]
    fn h16_mlx_kv_cache_typed_clamp_slot_count_one() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use crate::serve::multi_seq_kv::MultiSeqKvCache;
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let mut cache = MlxKvCache {
            k_packed: buf(),
            k_norms: buf(),
            v_packed: buf(),
            v_norms: buf(),
            capacity: 16,
            is_sliding: false,
            write_pos: 5,
            seq_len: 5,
        };

        // slot_count == 1.
        assert_eq!(
            cache.slot_count(),
            1,
            "H16 FALSIFIED: MlxKvCache slot_count must be 1"
        );
        assert_eq!(cache.layout(), MultiSeqLayout::SeparateSlots);

        // seq_len(SlotId(0)) reports the legacy cursor.
        assert_eq!(
            cache.seq_len(SlotId(0)).unwrap(),
            5,
            "H16: seq_len(0) reports legacy cursor (was 5)"
        );

        // seq_len(SlotId(1)) returns SlotOutOfRange.
        let err = cache.seq_len(SlotId(1)).expect_err("slot 1 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(1),
                max_slots: 1
            },
            "H16 FALSIFIED: SlotOutOfRange shape wrong; got {err:?}"
        );

        // append_for_seq OOR vs clamp.
        let err = cache.append_for_seq(SlotId(3), 1).expect_err("append OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(3),
                max_slots: 1
            }
        );
        let err = cache
            .append_for_seq(SlotId(0), 1)
            .expect_err("append clamped");
        match err {
            MultiSeqError::CapabilityUnsupported { capability } => {
                assert!(
                    capability.contains("MlxKvCache"),
                    "label must name struct: {capability}"
                );
                assert!(
                    capability.contains("A3b iter-3"),
                    "label must name deferral: {capability}"
                );
                assert!(
                    capability.contains("legacy 4-bit"),
                    "label must name legacy path: {capability}"
                );
            }
            other => panic!("H16: expected CapabilityUnsupported; got {other:?}"),
        }

        // drop_seq same shape.
        let err = cache.drop_seq(SlotId(7)).expect_err("drop OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(7),
                max_slots: 1
            }
        );
        let err = cache.drop_seq(SlotId(0)).expect_err("drop clamped");
        assert!(matches!(err, MultiSeqError::CapabilityUnsupported { .. }));

        // fork self ok no-op.
        cache
            .fork_seq(SlotId(0), SlotId(0))
            .expect("self-fork no-op");
    }

    /// **H10 verification (defence-in-depth, post-falsification pin)** —
    /// `HF2Q_HYBRID_KV` default value is TRUE per ADR-029 iter-13.
    /// The dossier's pre-iter-13 "default-off" assumption is stale; this
    /// test pins the current `env_default_true` discipline so a future
    /// regression that flips the default back to false (perhaps via a
    /// `env_default_false` rename) trips here, naming the H10 footnote
    /// in the failure message.  Mirrors A3a's H10 status note at
    /// kv_cache.rs:932-942.
    #[test]
    fn h10_post_falsification_hybrid_kv_default_is_on() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // The investigation_env module is the production source of
        // truth.  We read the raw env-default helper to avoid any
        // process-state pollution from other tests setting the
        // variable explicitly.
        std::env::remove_var("HF2Q_HYBRID_KV");
        // Re-construct an InvestigationEnv with the env unset so we
        // observe the default-on behaviour directly.
        let env = crate::debug::investigation_env::InvestigationEnv::from_env();
        assert!(
            env.hybrid_kv,
            "H10 post-falsification FALSIFIED: HF2Q_HYBRID_KV default flipped to OFF; \
             A3b iter-1 assumed default-ON per ADR-029 iter-13 + \
             `src/debug/investigation_env.rs:878` (env_default_true). \
             If this flip is intentional, update the A3b iter-1 block \
             comment at kv_cache.rs and re-examine whether HybridKvBuffers \
             remains the production-default variant for Gemma 4."
        );
    }

    /// **ADR-040 iter-B4c-kernel iter-1 — `MultiSeqHbKvBuffers::
    /// reset_for_slot` per-slot cursor isolation (2026-05-30)**.
    ///
    /// Pin: `reset_for_slot(SlotId(s))` ONLY zeros `seq_lens[s]`;
    /// other slots' cursors are untouched, AND the K/V packed + norms
    /// bytes of EVERY slot (including slot s) are byte-identical to
    /// pre-call (cursor-masked read discipline — see layout proof at
    /// the method).
    ///
    /// Mirror of Qwen35
    /// `iter_c2d_cont_kernel_iter1_reset_for_slot_per_slot_isolation`
    /// (qwen35/kv_cache.rs:8113).
    #[test]
    fn iter_b4c_kernel_iter1_multi_seq_hb_kv_reset_for_slot_per_slot_isolation_2026_05_30() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use crate::serve::multi_seq_kv::SlotId;
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 4usize;
        let n_seqs: u32 = 4;
        let mut cache =
            alloc_hb_kv_for_layer(&dev, 0, nkv, hd, cap, false, n_seqs).expect("alloc n_seqs=4");

        // Seed every slot's cursor + K/V packed bytes with distinct
        // non-zero patterns.
        for s in 0..(n_seqs as usize) {
            cache.seq_lens[s] = (s as u32) + 11;
        }
        let slot_packed = nkv * cap * hd;
        {
            let k_slice = cache.k_packed.as_mut_slice::<u8>().expect("k_packed u8");
            for s in 0..(n_seqs as usize) {
                let start = s * slot_packed;
                for (i, b) in k_slice[start..start + slot_packed].iter_mut().enumerate() {
                    *b = (((s * 17 + i) % 251) + 1) as u8;
                }
            }
        }
        {
            let v_slice = cache.v_packed.as_mut_slice::<u8>().expect("v_packed u8");
            for s in 0..(n_seqs as usize) {
                let start = s * slot_packed;
                for (i, b) in v_slice[start..start + slot_packed].iter_mut().enumerate() {
                    *b = (((s * 19 + i) % 253) + 1) as u8;
                }
            }
        }
        // Snapshot K/V packed for every slot before the reset.
        let k_before: Vec<u8> = cache
            .k_packed
            .as_slice::<u8>()
            .expect("k_packed read")
            .to_vec();
        let v_before: Vec<u8> = cache
            .v_packed
            .as_slice::<u8>()
            .expect("v_packed read")
            .to_vec();

        // Reset slot 1.
        cache
            .reset_for_slot(SlotId(1))
            .expect("reset_for_slot(1) on n_seqs=4");

        // Slot 1's cursor must be 0; others untouched.
        for s in 0..(n_seqs as usize) {
            if s == 1 {
                assert_eq!(
                    cache.seq_lens[s], 0,
                    "iter-B4c-kernel iter-1: slot 1 cursor must be 0 after reset_for_slot(1)"
                );
            } else {
                assert_eq!(
                    cache.seq_lens[s],
                    (s as u32) + 11,
                    "iter-B4c-kernel iter-1: slot {s} cursor must be untouched"
                );
            }
        }
        // K/V packed bytes of EVERY slot are byte-identical pre/post
        // (cursor-masked discipline — no K/V byte zeroing on reset).
        let k_after: Vec<u8> = cache
            .k_packed
            .as_slice::<u8>()
            .expect("k_packed read 2")
            .to_vec();
        let v_after: Vec<u8> = cache
            .v_packed
            .as_slice::<u8>()
            .expect("v_packed read 2")
            .to_vec();
        assert_eq!(
            k_before, k_after,
            "iter-B4c-kernel iter-1: reset_for_slot must NOT zero K packed bytes \
             (cursor-masked discipline; matches drop_seq invariant)"
        );
        assert_eq!(
            v_before, v_after,
            "iter-B4c-kernel iter-1: reset_for_slot must NOT zero V packed bytes"
        );
    }

    /// **ADR-040 iter-B4c-kernel iter-1 — `MultiSeqHbKvBuffers::
    /// reset_for_slot` bounds-first typed error (2026-05-30)**.
    ///
    /// Bounds-first per A2b iter-1.5 cfa-finding-F5: OOR slot returns
    /// typed `MultiSeqError::SlotOutOfRange`.  SlotId(0) on n_seqs=1
    /// is the byte-equivalence case — must succeed.
    ///
    /// Mirror of Qwen35
    /// `iter_c2d_cont_kernel_iter1_reset_for_slot_bounds_typed`
    /// (qwen35/kv_cache.rs:8270).
    #[test]
    fn iter_b4c_kernel_iter1_multi_seq_hb_kv_reset_for_slot_bounds_typed_2026_05_30() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use crate::serve::multi_seq_kv::{MultiSeqError, SlotId};
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let mut cache =
            alloc_hb_kv_for_layer(&dev, 0, 2, 256, 4, false, 4).expect("alloc n_seqs=4");
        let err = cache.reset_for_slot(SlotId(4)).expect_err("slot 4 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(4),
                max_slots: 4
            },
            "iter-B4c-kernel iter-1: OOR must surface SlotOutOfRange"
        );

        // SlotId(0) on n_seqs=1 is the byte-equivalence case.
        let mut cache1 =
            alloc_hb_kv_for_layer(&dev, 0, 2, 256, 4, false, 1).expect("alloc n_seqs=1");
        cache1.seq_lens[0] = 7;
        cache1
            .reset_for_slot(SlotId(0))
            .expect("SlotId(0) at n_seqs=1 must succeed");
        assert_eq!(cache1.seq_lens[0], 0);
    }

    /// **ADR-040 iter-B4c-kernel iter-1 — `MultiSeqHybridKvBuffers::
    /// reset_for_slot` per-slot cursor isolation (2026-05-30)**.
    ///
    /// Sibling pin for the hybrid (F16-K + TQ-HB-V) variant.
    #[test]
    fn iter_b4c_kernel_iter1_multi_seq_hybrid_kv_reset_for_slot_per_slot_isolation_2026_05_30() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use crate::serve::multi_seq_kv::SlotId;
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        // Ensure xlen + F16 are env-unset so the legacy F16-K + U8-V
        // shape is exercised; the reset_for_slot semantics are
        // env-invariant but the alloc-time bytes differ.
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        let mut cache = alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, 2, 256, 4, false, 4)
            .expect("alloc multi-seq hybrid n_seqs=4");
        for s in 0..4 {
            cache.seq_lens[s] = (s as u32) * 5 + 3;
        }
        cache.reset_for_slot(SlotId(2)).expect("reset_for_slot(2)");
        assert_eq!(cache.seq_lens[0], 3);
        assert_eq!(cache.seq_lens[1], 8);
        assert_eq!(
            cache.seq_lens[2], 0,
            "iter-B4c-kernel iter-1: slot 2 cursor must be 0 after reset"
        );
        assert_eq!(cache.seq_lens[3], 18);

        // OOR.
        let err = cache.reset_for_slot(SlotId(99)).expect_err("slot 99 OOR");
        assert!(matches!(
            err,
            crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange { .. }
        ));
    }

    // ───────────────────────────────────────────────────────────────────────
    // ADR-040 Phase A3b iter-2 — MultiSeqDenseKvBuffers FULL LIFT
    // hypothesis bank H144-H150.
    //
    // Mirrors A3b iter-1's H11-H14 + iter-1.5's H11r structure for the
    // DenseKvBuffers variant.  The lift is additive (legacy DenseKvBuffers
    // typed clamp stays — see H15 — until Phase B4c re-routes the 3
    // production alloc sites; iter-A3b-3 MlxKvCache typed clamp also
    // stays — see H16 — until iter-A3b-3 ships).
    //
    // Hypothesis register:
    //   H144 — sibling struct + alloc helper exist; n_seqs/seq_lens
    //          discipline matches A3b iter-1.
    //   H145 — alloc helper pre-flight: n_seqs=0 / nkv=0 / hd=0 / cap=0
    //          all return Err (no panic).
    //   H146 — byte-scale: n_seqs=4 yields exactly 4× the n_seqs=1
    //          baseline on K + V (both dtypes); EXACT concrete formula
    //          pinned mirroring H11's iter-1.5 hygiene fix.
    //   H147 — per-slot byte isolation: host-side writes to slot 0's
    //          K/V regions leave slot 1's bytes byte-identical.
    //   H148 — n_seqs=1 byte-equivalence: byte counts match the
    //          legacy inline DenseKvBuffers per-layer alloc.
    //   H149 — MultiSeqKvCache impl: slot_count() == n_seqs;
    //          bounds-first SlotOutOfRange; cursor advances per slot;
    //          fork cross-slot → CapabilityUnsupported naming A3c.
    //   H150 — reset_for_slot inherent method: per-slot cursor reset
    //          with K/V byte preservation (cursor-masked discipline)
    //          and bounds-first typed OOR.
    // ───────────────────────────────────────────────────────────────────────

    /// **H144** — Sibling struct `MultiSeqDenseKvBuffers` exists with
    /// `n_seqs` outermost discipline + per-seq cursor + correct dtype/
    /// is_sliding propagation through [`alloc_multi_seq_dense_kv_for_layer`].
    ///
    /// Falsifier (any one ⇒ H144 broken):
    /// 1. Struct missing or misnamed.
    /// 2. n_seqs field absent / wrong type.
    /// 3. seq_lens not Vec<u32> length n_seqs.
    /// 4. dtype not propagated from alloc-time argument.
    /// 5. is_sliding not propagated.
    #[test]
    fn h144_multi_seq_dense_kv_buffers_sibling_struct_exists() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;
        let n_seqs = 3u32;

        // F32 + linear path.
        let buf_f32_lin = alloc_multi_seq_dense_kv_for_layer(
            &dev,
            0,
            nkv,
            hd,
            cap,
            /*is_ring=*/ false,
            DType::F32,
            n_seqs,
        )
        .expect("H144: alloc F32 linear");
        assert_eq!(buf_f32_lin.n_seqs, n_seqs, "H144: n_seqs propagation");
        assert_eq!(buf_f32_lin.dtype, DType::F32, "H144: dtype propagation F32");
        assert!(
            !buf_f32_lin.is_sliding,
            "H144: is_sliding=false propagation"
        );
        assert_eq!(buf_f32_lin.capacity, cap, "H144: capacity propagation");
        assert_eq!(
            buf_f32_lin.seq_lens.len(),
            n_seqs as usize,
            "H144 FALSIFIED: seq_lens.len() must equal n_seqs"
        );
        assert!(
            buf_f32_lin.seq_lens.iter().all(|&x| x == 0),
            "H144 FALSIFIED: seq_lens zero-init"
        );
        // Shape: n_seqs OUTERMOST on K + V.
        assert_eq!(
            buf_f32_lin.k.shape(),
            &[n_seqs as usize, nkv, cap, hd],
            "H144 FALSIFIED: K shape n_seqs outermost"
        );
        assert_eq!(
            buf_f32_lin.v.shape(),
            &[n_seqs as usize, nkv, cap, hd],
            "H144 FALSIFIED: V shape n_seqs outermost"
        );

        // F16 + sliding path (HF2Q_F16_KV codepath).
        let buf_f16_ring = alloc_multi_seq_dense_kv_for_layer(
            &dev,
            7,
            nkv,
            hd,
            cap,
            /*is_ring=*/ true,
            DType::F16,
            n_seqs,
        )
        .expect("H144: alloc F16 sliding");
        assert_eq!(
            buf_f16_ring.dtype,
            DType::F16,
            "H144: dtype propagation F16"
        );
        assert!(buf_f16_ring.is_sliding, "H144: is_sliding=true propagation");
    }

    /// **H145** — `alloc_multi_seq_dense_kv_for_layer` pre-flight:
    /// `n_seqs == 0`, `nkv == 0`, `hd == 0`, `cap == 0` all return
    /// `Err` (NOT panic).  Mirrors A3a / A3b iter-1 pre-flight.
    #[test]
    fn h145_alloc_multi_seq_dense_kv_for_layer_preflight_errors() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        // n_seqs = 0
        assert!(
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, 2, 256, 8, false, DType::F32, 0).is_err(),
            "H145 FALSIFIED: n_seqs=0 must error"
        );
        // nkv = 0
        assert!(
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, 0, 256, 8, false, DType::F32, 1).is_err(),
            "H145 FALSIFIED: nkv=0 must error"
        );
        // hd = 0
        assert!(
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, 2, 0, 8, false, DType::F32, 1).is_err(),
            "H145 FALSIFIED: hd=0 must error"
        );
        // cap = 0
        assert!(
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, 2, 256, 0, false, DType::F32, 1).is_err(),
            "H145 FALSIFIED: cap=0 must error"
        );
    }

    /// **H146** — `alloc_multi_seq_dense_kv_for_layer(.., n_seqs=4)`
    /// produces buffers byte-scaled exactly 4× the n_seqs=1 baseline
    /// across K + V on BOTH F32 and F16 dtypes.  EXACT concrete formula
    /// pinned (mirrors H11 iter-1.5 hygiene fix).
    ///
    /// Formula (dtype = F32, hd=256, nkv=2, cap=8):
    ///   k_bytes = n * nkv * cap * hd * 4 = 4 * 2 * 8 * 256 * 4 = 65536
    ///   v_bytes = n * nkv * cap * hd * 4 = 65536
    ///   TOTAL   = 131072
    /// Formula (dtype = F16, same shape):
    ///   k_bytes = 65536 / 2 = 32768
    ///   v_bytes = 32768
    ///   TOTAL   = 65536
    #[test]
    fn h146_multi_seq_dense_kv_n_seqs_4_byte_scale_exact_formula() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;

        // F32 path.
        let f32_baseline =
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, nkv, hd, cap, false, DType::F32, 1)
                .expect("H146: F32 alloc n_seqs=1");
        let f32_lifted =
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, nkv, hd, cap, false, DType::F32, 4)
                .expect("H146: F32 alloc n_seqs=4");
        assert_eq!(f32_baseline.n_seqs, 1);
        assert_eq!(f32_lifted.n_seqs, 4);
        // 4× scaling on K + V.
        assert_eq!(
            f32_lifted.k.byte_len(),
            f32_baseline.k.byte_len() * 4,
            "H146 FALSIFIED: F32 K not 4× scale"
        );
        assert_eq!(
            f32_lifted.v.byte_len(),
            f32_baseline.v.byte_len() * 4,
            "H146 FALSIFIED: F32 V not 4× scale"
        );
        // EXACT formula at n_seqs=4 (F32).
        let expected_f32_k = 4usize * nkv * cap * hd * 4; // 65536
        let expected_f32_v = expected_f32_k;
        let expected_f32_total = expected_f32_k + expected_f32_v;
        assert_eq!(
            f32_lifted.k.byte_len(),
            expected_f32_k,
            "H146 EXACT FORMULA FALSIFIED: F32 K bytes != n*nkv*cap*hd*4"
        );
        assert_eq!(
            f32_lifted.v.byte_len(),
            expected_f32_v,
            "H146 EXACT FORMULA FALSIFIED: F32 V bytes != n*nkv*cap*hd*4"
        );
        let actual_f32_total = f32_lifted.k.byte_len() + f32_lifted.v.byte_len();
        assert_eq!(
            actual_f32_total, expected_f32_total,
            "H146 EXACT FORMULA FALSIFIED: F32 composition"
        );
        assert_eq!(
            actual_f32_total, 131_072,
            "H146 EXACT FORMULA FALSIFIED at concrete value: F32 expected 131072 \
             bytes for n_seqs=4 nkv=2 cap=8 hd=256; got {}",
            actual_f32_total
        );

        // F16 path (HF2Q_F16_KV codepath).
        let f16_baseline =
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, nkv, hd, cap, false, DType::F16, 1)
                .expect("H146: F16 alloc n_seqs=1");
        let f16_lifted =
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, nkv, hd, cap, false, DType::F16, 4)
                .expect("H146: F16 alloc n_seqs=4");
        assert_eq!(
            f16_lifted.k.byte_len(),
            f16_baseline.k.byte_len() * 4,
            "H146 FALSIFIED: F16 K not 4× scale"
        );
        assert_eq!(
            f16_lifted.v.byte_len(),
            f16_baseline.v.byte_len() * 4,
            "H146 FALSIFIED: F16 V not 4× scale"
        );
        // EXACT formula (F16: 2 bytes/elem instead of 4).
        let expected_f16_k = 4usize * nkv * cap * hd * 2; // 32768
        let expected_f16_v = expected_f16_k;
        let expected_f16_total = expected_f16_k + expected_f16_v;
        assert_eq!(
            f16_lifted.k.byte_len(),
            expected_f16_k,
            "H146 EXACT FORMULA FALSIFIED: F16 K bytes != n*nkv*cap*hd*2"
        );
        let actual_f16_total = f16_lifted.k.byte_len() + f16_lifted.v.byte_len();
        assert_eq!(
            actual_f16_total, expected_f16_total,
            "H146 EXACT FORMULA FALSIFIED: F16 composition"
        );
        assert_eq!(
            actual_f16_total, 65_536,
            "H146 EXACT FORMULA FALSIFIED at concrete value: F16 expected 65536 \
             bytes for n_seqs=4 nkv=2 cap=8 hd=256; got {}",
            actual_f16_total
        );

        // Per-seq cursor vec length tracks n_seqs.
        assert_eq!(f32_baseline.seq_lens.len(), 1);
        assert_eq!(f32_lifted.seq_lens.len(), 4);
        assert_eq!(f16_lifted.seq_lens.len(), 4);
        assert!(f32_lifted.seq_lens.iter().all(|&x| x == 0));

        // Shape pin: n_seqs OUTERMOST on every buffer.
        for (name, b) in [("k", &f32_lifted.k), ("v", &f32_lifted.v)] {
            let s = b.shape().to_vec();
            assert_eq!(s.len(), 4, "H146 M5: {name} must be 4-D; got {:?}", s);
            assert_eq!(
                s[0], 4,
                "H146 M5 FALSIFIED: {name} shape[0] must be n_seqs=4 (n_seqs \
                 landed on wrong axis); got {:?}",
                s
            );
        }
    }

    /// **H147** — `MultiSeqDenseKvBuffers` per-slot byte isolation:
    /// host-side writes of a deterministic non-zero pattern into slot
    /// 0's K + V regions leave slot 1's bytes byte-identical.  Mirrors
    /// H12 for the dense F32 variant.
    ///
    /// Falsifier: any byte change in slot 1's K / V region after
    /// writing to slot 0's region ⇒ H147 broken; the per-slot
    /// byte-offset formula does not produce disjoint regions.
    #[test]
    fn h147_multi_seq_dense_kv_per_slot_byte_isolation() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 4usize;
        let mut cache =
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, nkv, hd, cap, false, DType::F32, 2)
                .expect("H147: alloc n_seqs=2");
        assert_eq!(cache.n_seqs, 2);

        // Per-slot region sizes (F32 = 4 bytes/elem):
        //   K: slot bytes = nkv * cap * hd * 4
        //   V: slot bytes = nkv * cap * hd * 4
        let slot_k_bytes = nkv * cap * hd * 4;
        let slot_v_bytes = nkv * cap * hd * 4;

        // Sanity: total bytes match 2 * slot_bytes for each buffer.
        assert_eq!(cache.k.byte_len(), 2 * slot_k_bytes, "H147: K total");
        assert_eq!(cache.v.byte_len(), 2 * slot_v_bytes, "H147: V total");

        cache.k.as_mut_slice::<u8>().expect("seed K peer")[slot_k_bytes..2 * slot_k_bytes]
            .fill(0xC3);
        cache.v.as_mut_slice::<u8>().expect("seed V peer")[slot_v_bytes..2 * slot_v_bytes]
            .fill(0xD4);

        // Write deterministic non-zero pattern into slot 0's K + V
        // (interpret as u8 bytes for fixture simplicity; the kernel
        // writes F32 but byte-level isolation is what we're pinning).
        {
            let k_slice = cache.k.as_mut_slice::<u8>().expect("K F32 as u8 mut");
            for (i, b) in k_slice[..slot_k_bytes].iter_mut().enumerate() {
                *b = (((i * 7) % 251) + 1) as u8;
            }
        }
        {
            let v_slice = cache.v.as_mut_slice::<u8>().expect("V F32 as u8 mut");
            for (i, b) in v_slice[..slot_v_bytes].iter_mut().enumerate() {
                *b = (((i * 11) % 253) + 1) as u8;
            }
        }

        // Snapshot slot 1's regions.
        let k_slot1_before: Vec<u8> =
            cache.k.as_slice::<u8>().expect("K read")[slot_k_bytes..2 * slot_k_bytes].to_vec();
        let v_slot1_before: Vec<u8> =
            cache.v.as_slice::<u8>().expect("V read")[slot_v_bytes..2 * slot_v_bytes].to_vec();

        // Sanity: slot 1 contains the explicit sentinel.
        assert!(
            k_slot1_before.iter().all(|&b| b == 0xC3),
            "H147 fixture sanity: slot 1 K sentinel"
        );
        assert!(
            v_slot1_before.iter().all(|&b| b == 0xD4),
            "H147 fixture sanity: slot 1 V sentinel"
        );

        // A3b iter-2 cursor advance on slot 0 (no buffer mutation).
        cache
            .append_for_seq(SlotId(0), 3)
            .expect("H147: append slot 0");
        assert_eq!(cache.seq_lens[0], 3);
        assert_eq!(cache.seq_lens[1], 0);

        // H147 falsifier: slot 1's bytes must be byte-identical.
        let k_slot1_after: Vec<u8> =
            cache.k.as_slice::<u8>().expect("K read 2")[slot_k_bytes..2 * slot_k_bytes].to_vec();
        let v_slot1_after: Vec<u8> =
            cache.v.as_slice::<u8>().expect("V read 2")[slot_v_bytes..2 * slot_v_bytes].to_vec();

        assert_eq!(
            k_slot1_before, k_slot1_after,
            "H147 FALSIFIED: slot 1 K bytes changed after slot-0 write"
        );
        assert_eq!(
            v_slot1_before, v_slot1_after,
            "H147 FALSIFIED: slot 1 V bytes changed after slot-0 write"
        );
    }

    /// **H148** — n_seqs=1 byte-equivalence: allocating
    /// `MultiSeqDenseKvBuffers` at `n_seqs=1` produces buffer byte
    /// counts EQUAL to a legacy `DenseKvBuffers` per-layer K + V
    /// allocation at the same `(nkv, cap, hd, dtype)` parameters.
    ///
    /// Pins the H148 hypothesis: the iter-A3b-2 sibling-struct lift is
    /// byte-equivalent at n_seqs=1 to the 3 legacy production alloc
    /// sites (`forward_prefill.rs:705`, `forward_prefill_batched.rs:367`,
    /// `engine.rs:6836`) which each emit `DenseKvBuffers` with
    /// `nkv*cap*hd*dtype.size_of()` bytes per buffer.  Phase B4c re-route
    /// will be byte-safe by construction.
    #[test]
    fn h148_multi_seq_dense_kv_n_seqs_1_byte_equivalent_to_legacy() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;

        for &dtype in &[DType::F32, DType::F16] {
            let multi = alloc_multi_seq_dense_kv_for_layer(&dev, 0, nkv, hd, cap, false, dtype, 1)
                .expect("H148: alloc multi-seq n_seqs=1");

            // Legacy per-layer formula (mirrors forward_prefill.rs:700-704):
            //   n = nkv * capacity * hd
            //   alloc_buffer(n * kv_elem_bytes, kv_dtype, vec![nkv, cap, hd])
            let legacy_bytes = nkv * cap * hd * dtype.size_of();

            assert_eq!(
                multi.k.byte_len(),
                legacy_bytes,
                "H148 FALSIFIED ({:?}): K bytes {} != legacy {}",
                dtype,
                multi.k.byte_len(),
                legacy_bytes
            );
            assert_eq!(
                multi.v.byte_len(),
                legacy_bytes,
                "H148 FALSIFIED ({:?}): V bytes {} != legacy {}",
                dtype,
                multi.v.byte_len(),
                legacy_bytes
            );

            // Total parity vs legacy DenseKvBuffers (K + V).
            let legacy_total = 2 * legacy_bytes;
            use crate::serve::kv_persist::lcp_registry::ByteSized;
            assert_eq!(
                multi.byte_len(),
                legacy_total as u64,
                "H148 FALSIFIED ({:?}): total byte_len {} != legacy K+V {}",
                dtype,
                multi.byte_len(),
                legacy_total
            );
        }
    }

    /// **H149** — `MultiSeqKvCache` impl for `MultiSeqDenseKvBuffers`:
    /// `slot_count() == n_seqs` (NOT 1 — the multi-seq sibling is no
    /// longer clamped); bounds-first SlotOutOfRange on the OOR path;
    /// per-slot cursor advance + drop; fork same-slot Ok; fork cross-slot
    /// → CapabilityUnsupported naming A3c.  Mirrors H11/H12/H13 +
    /// gemma4_hb_kv_fork_cross_slot_returns_capability_unsupported.
    #[test]
    fn h149_multi_seq_dense_kv_multi_seq_kv_cache_impl() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let n_seqs = 4u32;
        let mut cache =
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, 2, 256, 8, false, DType::F32, n_seqs)
                .expect("H149: alloc n_seqs=4");

        // 1. slot_count() == n_seqs (NOT the clamp's 1).
        assert_eq!(
            cache.slot_count(),
            n_seqs,
            "H149 FALSIFIED: slot_count must equal n_seqs={n_seqs}"
        );
        assert_eq!(cache.layout(), MultiSeqLayout::SeparateSlots);

        // 2. All slots start at cursor 0.
        for s in 0..n_seqs {
            assert_eq!(
                cache.seq_len(SlotId(s)).expect("seq_len in range"),
                0,
                "H149: slot {s} starts at cursor 0"
            );
        }

        // 3. Bounds-first OOR on seq_len.
        let err = cache.seq_len(SlotId(n_seqs)).expect_err("OOR n_seqs");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(n_seqs),
                max_slots: n_seqs,
            },
            "H149 FALSIFIED: seq_len OOR shape; got {err:?}"
        );

        // 4. Per-slot cursor advance.
        cache.append_for_seq(SlotId(0), 5).expect("append slot 0");
        cache.append_for_seq(SlotId(2), 3).expect("append slot 2");
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 5);
        assert_eq!(
            cache.seq_len(SlotId(1)).unwrap(),
            0,
            "H149 FALSIFIED: slot 1 cursor touched by slot 0/2 append"
        );
        assert_eq!(cache.seq_len(SlotId(2)).unwrap(), 3);
        assert_eq!(
            cache.seq_len(SlotId(3)).unwrap(),
            0,
            "H149 FALSIFIED: slot 3 cursor touched by slot 0/2 append"
        );

        // 5. Bounds-first OOR on append.
        let err = cache
            .append_for_seq(SlotId(n_seqs + 1), 1)
            .expect_err("append OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(n_seqs + 1),
                max_slots: n_seqs,
            }
        );

        // 6. drop_seq resets target cursor, leaves siblings.
        cache.drop_seq(SlotId(0)).expect("drop slot 0");
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 0, "H149: slot 0 reset");
        assert_eq!(
            cache.seq_len(SlotId(2)).unwrap(),
            3,
            "H149 FALSIFIED: slot 2 preserved through slot 0 drop"
        );

        // 7. Bounds-first OOR on drop.
        let err = cache.drop_seq(SlotId(99)).expect_err("drop OOR");
        assert!(matches!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(99),
                max_slots: 4
            }
        ));

        // 8. fork_seq same slot is a no-op Ok.
        cache
            .fork_seq(SlotId(1), SlotId(1))
            .expect("self-fork no-op");

        // 9. Bounds-first OOR on fork (src then dst).
        let err = cache
            .fork_seq(SlotId(99), SlotId(0))
            .expect_err("fork src OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(99),
                max_slots: 4
            }
        );
        let err = cache
            .fork_seq(SlotId(0), SlotId(99))
            .expect_err("fork dst OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(99),
                max_slots: 4
            }
        );

        // 10. fork cross-slot → Ok(()) post-iter-A3c (was previously
        // CapabilityUnsupported naming A3c per A3b iter-2 typed-clamp).
        // Seed slot 0's cursor + slot 2's cursor for the post-fork copy
        // assertion (slot 2 untouched).
        cache
            .append_for_seq(SlotId(0), 5)
            .expect("re-seed slot 0 for fork");
        let slot0_before = cache.seq_len(SlotId(0)).unwrap();
        let slot2_before = cache.seq_len(SlotId(2)).unwrap();
        cache
            .fork_seq(SlotId(0), SlotId(1))
            .expect("iter-A3c closure: cross-slot fork must return Ok(())");
        assert_eq!(
            cache.seq_len(SlotId(1)).unwrap(),
            slot0_before,
            "H149 closure: fork must copy src's seq_len to dst"
        );
        assert_eq!(
            cache.seq_len(SlotId(0)).unwrap(),
            slot0_before,
            "H149 closure: fork must NOT mutate src's seq_len"
        );
        assert_eq!(
            cache.seq_len(SlotId(2)).unwrap(),
            slot2_before,
            "H149 closure: fork must NOT mutate non-src non-dst slots"
        );
    }

    /// **H150** — `MultiSeqDenseKvBuffers::reset_for_slot` inherent
    /// method: per-slot cursor reset with K + V byte preservation
    /// (cursor-masked discipline matching A3a / A3b iter-1 siblings);
    /// bounds-first typed OOR; SlotId(0) at n_seqs=1 is byte-equivalent
    /// case (must succeed).
    ///
    /// Mirrors `iter_b4c_kernel_iter1_multi_seq_hybrid_kv_reset_for_slot_*`
    /// tests for the dense variant.
    #[test]
    fn h150_multi_seq_dense_kv_reset_for_slot_per_slot_isolation_and_bounds() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 4usize;
        let n_seqs = 4u32;
        let mut cache =
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, nkv, hd, cap, false, DType::F32, n_seqs)
                .expect("alloc n_seqs=4");

        // Seed every slot's cursor + K/V bytes with distinct patterns.
        for s in 0..(n_seqs as usize) {
            cache.seq_lens[s] = (s as u32) + 11;
        }
        let slot_k_bytes = nkv * cap * hd * 4; // F32
        let slot_v_bytes = nkv * cap * hd * 4;
        {
            let k_slice = cache.k.as_mut_slice::<u8>().expect("k u8");
            for s in 0..(n_seqs as usize) {
                let start = s * slot_k_bytes;
                for (i, b) in k_slice[start..start + slot_k_bytes].iter_mut().enumerate() {
                    *b = (((s * 17 + i) % 251) + 1) as u8;
                }
            }
        }
        {
            let v_slice = cache.v.as_mut_slice::<u8>().expect("v u8");
            for s in 0..(n_seqs as usize) {
                let start = s * slot_v_bytes;
                for (i, b) in v_slice[start..start + slot_v_bytes].iter_mut().enumerate() {
                    *b = (((s * 19 + i) % 253) + 1) as u8;
                }
            }
        }
        // Snapshot K/V for every slot before the reset.
        let k_before: Vec<u8> = cache.k.as_slice::<u8>().expect("k read").to_vec();
        let v_before: Vec<u8> = cache.v.as_slice::<u8>().expect("v read").to_vec();

        // Reset slot 1.
        cache
            .reset_for_slot(SlotId(1))
            .expect("reset_for_slot(1) on n_seqs=4");

        // Slot 1's cursor must be 0; others untouched.
        for s in 0..(n_seqs as usize) {
            if s == 1 {
                assert_eq!(
                    cache.seq_lens[s], 0,
                    "H150 FALSIFIED: slot 1 cursor must be 0 after reset_for_slot(1)"
                );
            } else {
                assert_eq!(
                    cache.seq_lens[s],
                    (s as u32) + 11,
                    "H150 FALSIFIED: slot {s} cursor must be untouched"
                );
            }
        }

        // K/V bytes of EVERY slot are byte-identical pre/post
        // (cursor-masked discipline — no K/V byte zeroing on reset).
        let k_after: Vec<u8> = cache.k.as_slice::<u8>().expect("k read 2").to_vec();
        let v_after: Vec<u8> = cache.v.as_slice::<u8>().expect("v read 2").to_vec();
        assert_eq!(
            k_before, k_after,
            "H150 FALSIFIED: reset_for_slot must NOT zero K bytes \
             (cursor-masked discipline; matches drop_seq invariant)"
        );
        assert_eq!(
            v_before, v_after,
            "H150 FALSIFIED: reset_for_slot must NOT zero V bytes"
        );

        // Bounds-first OOR.
        let err = cache.reset_for_slot(SlotId(99)).expect_err("slot 99 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(99),
                max_slots: 4
            },
            "H150 FALSIFIED: reset OOR shape; got {err:?}"
        );

        // SlotId(0) at n_seqs=1 byte-equivalence case (must succeed).
        let mut cache1 =
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, 2, 256, 4, false, DType::F32, 1)
                .expect("alloc n_seqs=1");
        cache1.seq_lens[0] = 7;
        cache1
            .reset_for_slot(SlotId(0))
            .expect("SlotId(0) at n_seqs=1 must succeed (byte-equivalence case)");
        assert_eq!(cache1.seq_lens[0], 0);
    }

    // ───────────────────────────────────────────────────────────────────────
    // ADR-040 Phase A3b iter-3 — MultiSeqMlxKvCache FULL LIFT
    // hypothesis bank H151-H157.
    //
    // Mirrors A3b iter-2's H144-H150 structure for the LEGACY 4-bit
    // nibble-packed `MlxKvCache` variant (off-default since ADR-007
    // default-on TQ 8-bit).  The lift is additive (legacy MlxKvCache
    // typed clamp stays — see H16 — until Phase B4c re-routes the
    // single production alloc site at `gemma4/model.rs:1277-1290`).
    //
    // Hypothesis register:
    //   H151 — sibling struct + alloc helper exist; n_seqs/seq_lens
    //          discipline matches A3b iter-2.
    //   H152 — alloc helper pre-flight: n_seqs=0 / nkv=0 / hd=0 / cap=0
    //          / norms_per_pos=0 all return Err (no panic); odd hd also
    //          returns Err (iter-3-specific 4-bit-packed evenness gate).
    //   H153 — byte-scale: n_seqs=4 yields exactly 4× the n_seqs=1
    //          baseline on k_packed / k_norms / v_packed / v_norms;
    //          EXACT concrete formula pinned mirroring H146.
    //   H154 — per-slot byte isolation: host-side writes to slot 0's
    //          K packed / K norms / V packed / V norms regions leave
    //          slot 1's bytes byte-identical.
    //   H155 — n_seqs=1 byte-equivalence: byte counts match the legacy
    //          inline MlxKvCache per-layer K packed + K norms + V packed
    //          + V norms alloc (`gemma4/model.rs:1272-1290`).
    //   H156 — MultiSeqKvCache impl: slot_count() == n_seqs;
    //          bounds-first SlotOutOfRange; cursor advances per slot;
    //          fork cross-slot → CapabilityUnsupported naming A3c.
    //   H157 — reset_for_slot inherent method: per-slot cursor reset
    //          with K packed / K norms / V packed / V norms byte
    //          preservation (cursor-masked discipline) and bounds-first
    //          typed OOR.
    // ───────────────────────────────────────────────────────────────────────

    /// **H151** — Sibling struct `MultiSeqMlxKvCache` exists with
    /// `n_seqs` outermost discipline + per-seq cursor + correct
    /// is_sliding / norms_per_pos propagation through
    /// [`alloc_multi_seq_mlx_kv_for_layer`].
    ///
    /// Falsifier (any one ⇒ H151 broken):
    /// 1. Struct missing or misnamed.
    /// 2. n_seqs field absent / wrong type.
    /// 3. seq_lens not Vec<u32> length n_seqs.
    /// 4. norms_per_pos not propagated from alloc-time argument.
    /// 5. is_sliding not propagated.
    /// 6. Shape on any of k_packed / k_norms / v_packed / v_norms does
    ///    NOT carry n_seqs as leading axis.
    #[test]
    fn h151_multi_seq_mlx_kv_cache_sibling_struct_exists() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;
        let n_seqs = 3u32;

        // norms_per_pos=1 (D=256) + linear path.
        let buf_lin = alloc_multi_seq_mlx_kv_for_layer(
            &dev, 0, nkv, hd, cap, /*is_ring=*/ false, /*norms_per_pos=*/ 1, n_seqs,
        )
        .expect("H151: alloc norms_per_pos=1 linear");
        assert_eq!(buf_lin.n_seqs, n_seqs, "H151: n_seqs propagation");
        assert_eq!(buf_lin.norms_per_pos, 1, "H151: norms_per_pos propagation");
        assert!(!buf_lin.is_sliding, "H151: is_sliding=false propagation");
        assert_eq!(buf_lin.capacity, cap, "H151: capacity propagation");
        assert_eq!(
            buf_lin.seq_lens.len(),
            n_seqs as usize,
            "H151 FALSIFIED: seq_lens.len() must equal n_seqs"
        );
        assert!(
            buf_lin.seq_lens.iter().all(|&x| x == 0),
            "H151 FALSIFIED: seq_lens zero-init"
        );
        // Shape: n_seqs OUTERMOST on every buffer.
        assert_eq!(
            buf_lin.k_packed.shape(),
            &[n_seqs as usize, nkv, cap, hd / 2],
            "H151 FALSIFIED: k_packed shape n_seqs outermost"
        );
        assert_eq!(
            buf_lin.v_packed.shape(),
            &[n_seqs as usize, nkv, cap, hd / 2],
            "H151 FALSIFIED: v_packed shape n_seqs outermost"
        );
        assert_eq!(
            buf_lin.k_norms.shape(),
            &[n_seqs as usize, nkv, cap],
            "H151 FALSIFIED: k_norms shape (norms_per_pos=1) n_seqs outermost"
        );
        assert_eq!(
            buf_lin.v_norms.shape(),
            &[n_seqs as usize, nkv, cap],
            "H151 FALSIFIED: v_norms shape (norms_per_pos=1) n_seqs outermost"
        );

        // norms_per_pos=2 (D=512) + sliding path.
        let hd_big = 512usize;
        let buf_ring = alloc_multi_seq_mlx_kv_for_layer(
            &dev, 7, nkv, hd_big, cap, /*is_ring=*/ true, /*norms_per_pos=*/ 2, n_seqs,
        )
        .expect("H151: alloc norms_per_pos=2 sliding");
        assert_eq!(
            buf_ring.norms_per_pos, 2,
            "H151: norms_per_pos=2 propagation"
        );
        assert!(buf_ring.is_sliding, "H151: is_sliding=true propagation");
        // K norms shape switches to 4-D when norms_per_pos > 1.
        assert_eq!(
            buf_ring.k_norms.shape(),
            &[n_seqs as usize, nkv, cap, 2],
            "H151 FALSIFIED: k_norms shape (norms_per_pos=2) must be 4-D"
        );
        assert_eq!(
            buf_ring.v_norms.shape(),
            &[n_seqs as usize, nkv, cap, 2],
            "H151 FALSIFIED: v_norms shape (norms_per_pos=2) must be 4-D"
        );
    }

    /// **H152** — `alloc_multi_seq_mlx_kv_for_layer` pre-flight:
    /// `n_seqs == 0`, `nkv == 0`, `hd == 0`, `cap == 0`,
    /// `norms_per_pos == 0` all return `Err` (NOT panic); ALSO odd
    /// `hd` returns `Err` (iter-3-specific 4-bit nibble-packed
    /// evenness gate the legacy path implicitly relies on at
    /// `gemma4/model.rs:1272` `nkv * capacity * (hd / 2)`).
    #[test]
    fn h152_alloc_multi_seq_mlx_kv_for_layer_preflight_errors() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        // n_seqs = 0
        assert!(
            alloc_multi_seq_mlx_kv_for_layer(&dev, 0, 2, 256, 8, false, 1, 0).is_err(),
            "H152 FALSIFIED: n_seqs=0 must error"
        );
        // nkv = 0
        assert!(
            alloc_multi_seq_mlx_kv_for_layer(&dev, 0, 0, 256, 8, false, 1, 1).is_err(),
            "H152 FALSIFIED: nkv=0 must error"
        );
        // hd = 0
        assert!(
            alloc_multi_seq_mlx_kv_for_layer(&dev, 0, 2, 0, 8, false, 1, 1).is_err(),
            "H152 FALSIFIED: hd=0 must error"
        );
        // cap = 0
        assert!(
            alloc_multi_seq_mlx_kv_for_layer(&dev, 0, 2, 256, 0, false, 1, 1).is_err(),
            "H152 FALSIFIED: cap=0 must error"
        );
        // norms_per_pos = 0
        assert!(
            alloc_multi_seq_mlx_kv_for_layer(&dev, 0, 2, 256, 8, false, 0, 1).is_err(),
            "H152 FALSIFIED: norms_per_pos=0 must error"
        );
        // odd hd (4-bit packing requires hd to be even)
        assert!(
            alloc_multi_seq_mlx_kv_for_layer(&dev, 0, 2, 257, 8, false, 1, 1).is_err(),
            "H152 FALSIFIED: odd hd must error (4-bit nibble-packed)"
        );
    }

    /// **H153** — `alloc_multi_seq_mlx_kv_for_layer(.., n_seqs=4)`
    /// produces buffers byte-scaled exactly 4× the n_seqs=1 baseline
    /// across k_packed / k_norms / v_packed / v_norms.  EXACT concrete
    /// formula pinned (mirrors H146 iter-A3b-2 hygiene fix).
    ///
    /// Formula (norms_per_pos=1, hd=256, nkv=2, cap=8):
    ///   k_packed bytes = n * nkv * cap * hd/2 * 1 = 4 * 2 * 8 * 128 * 1 = 8192
    ///   v_packed bytes = same = 8192
    ///   k_norms bytes  = n * nkv * cap * norms_per_pos * 4
    ///                  = 4 * 2 * 8 * 1 * 4 = 256
    ///   v_norms bytes  = same = 256
    ///   TOTAL          = 16_896
    /// Formula (norms_per_pos=2, hd=512, nkv=2, cap=8):
    ///   k_packed bytes = 4 * 2 * 8 * 256 * 1 = 16_384
    ///   v_packed bytes = 16_384
    ///   k_norms bytes  = 4 * 2 * 8 * 2 * 4 = 512
    ///   v_norms bytes  = 512
    ///   TOTAL          = 33_792
    #[test]
    fn h153_multi_seq_mlx_kv_n_seqs_4_byte_scale_exact_formula() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;

        // norms_per_pos=1 path (D=256).
        let baseline_1 = alloc_multi_seq_mlx_kv_for_layer(&dev, 0, nkv, hd, cap, false, 1, 1)
            .expect("H153: alloc norms_per_pos=1 n_seqs=1");
        let lifted_1 = alloc_multi_seq_mlx_kv_for_layer(&dev, 0, nkv, hd, cap, false, 1, 4)
            .expect("H153: alloc norms_per_pos=1 n_seqs=4");
        assert_eq!(baseline_1.n_seqs, 1);
        assert_eq!(lifted_1.n_seqs, 4);

        // 4× scaling on every buffer.
        assert_eq!(
            lifted_1.k_packed.byte_len(),
            baseline_1.k_packed.byte_len() * 4,
            "H153 FALSIFIED: k_packed not 4× scale"
        );
        assert_eq!(
            lifted_1.v_packed.byte_len(),
            baseline_1.v_packed.byte_len() * 4,
            "H153 FALSIFIED: v_packed not 4× scale"
        );
        assert_eq!(
            lifted_1.k_norms.byte_len(),
            baseline_1.k_norms.byte_len() * 4,
            "H153 FALSIFIED: k_norms not 4× scale"
        );
        assert_eq!(
            lifted_1.v_norms.byte_len(),
            baseline_1.v_norms.byte_len() * 4,
            "H153 FALSIFIED: v_norms not 4× scale"
        );

        // EXACT formula at n_seqs=4 (norms_per_pos=1).
        let expected_k_packed = 4usize * nkv * cap * (hd / 2); // 8192
        let expected_v_packed = expected_k_packed;
        let expected_k_norms = 4usize * nkv * cap * 1 * 4; // 256
        let expected_v_norms = expected_k_norms;
        let expected_total =
            expected_k_packed + expected_v_packed + expected_k_norms + expected_v_norms;
        assert_eq!(
            lifted_1.k_packed.byte_len(),
            expected_k_packed,
            "H153 EXACT FORMULA FALSIFIED: k_packed bytes != n*nkv*cap*hd/2"
        );
        assert_eq!(
            lifted_1.v_packed.byte_len(),
            expected_v_packed,
            "H153 EXACT FORMULA FALSIFIED: v_packed bytes != n*nkv*cap*hd/2"
        );
        assert_eq!(
            lifted_1.k_norms.byte_len(),
            expected_k_norms,
            "H153 EXACT FORMULA FALSIFIED: k_norms bytes != n*nkv*cap*1*4"
        );
        assert_eq!(
            lifted_1.v_norms.byte_len(),
            expected_v_norms,
            "H153 EXACT FORMULA FALSIFIED: v_norms bytes != n*nkv*cap*1*4"
        );
        use crate::serve::kv_persist::lcp_registry::ByteSized;
        let actual_total = lifted_1.byte_len() as usize;
        assert_eq!(
            actual_total, expected_total,
            "H153 EXACT FORMULA FALSIFIED: total composition"
        );
        assert_eq!(
            actual_total, 16_896,
            "H153 EXACT FORMULA FALSIFIED at concrete value: norms_per_pos=1 \
             expected 16896 bytes for n_seqs=4 nkv=2 cap=8 hd=256; got {}",
            actual_total
        );

        // norms_per_pos=2 path (D=512 — per AmesianX iter-15 per-block norm).
        let hd_big = 512usize;
        let baseline_2 = alloc_multi_seq_mlx_kv_for_layer(&dev, 0, nkv, hd_big, cap, false, 2, 1)
            .expect("H153: alloc norms_per_pos=2 n_seqs=1");
        let lifted_2 = alloc_multi_seq_mlx_kv_for_layer(&dev, 0, nkv, hd_big, cap, false, 2, 4)
            .expect("H153: alloc norms_per_pos=2 n_seqs=4");
        assert_eq!(
            lifted_2.k_packed.byte_len(),
            baseline_2.k_packed.byte_len() * 4,
            "H153 FALSIFIED: k_packed norms_per_pos=2 not 4× scale"
        );
        assert_eq!(
            lifted_2.k_norms.byte_len(),
            baseline_2.k_norms.byte_len() * 4,
            "H153 FALSIFIED: k_norms norms_per_pos=2 not 4× scale"
        );
        // EXACT formula (norms_per_pos=2, hd=512).
        let expected_k_packed_2 = 4usize * nkv * cap * (hd_big / 2); // 16384
        let expected_k_norms_2 = 4usize * nkv * cap * 2 * 4; // 512
        let expected_total_2 = 2 * expected_k_packed_2 + 2 * expected_k_norms_2;
        let actual_total_2 = lifted_2.byte_len() as usize;
        assert_eq!(
            actual_total_2, expected_total_2,
            "H153 EXACT FORMULA FALSIFIED: norms_per_pos=2 composition"
        );
        assert_eq!(
            actual_total_2, 33_792,
            "H153 EXACT FORMULA FALSIFIED at concrete value: norms_per_pos=2 \
             expected 33792 bytes for n_seqs=4 nkv=2 cap=8 hd=512; got {}",
            actual_total_2
        );

        // Per-seq cursor vec length tracks n_seqs.
        assert_eq!(baseline_1.seq_lens.len(), 1);
        assert_eq!(lifted_1.seq_lens.len(), 4);
        assert_eq!(lifted_2.seq_lens.len(), 4);
        assert!(lifted_1.seq_lens.iter().all(|&x| x == 0));

        // Shape pin: n_seqs OUTERMOST on every buffer.
        for (name, b) in [
            ("k_packed", &lifted_1.k_packed),
            ("v_packed", &lifted_1.v_packed),
            ("k_norms", &lifted_1.k_norms),
            ("v_norms", &lifted_1.v_norms),
        ] {
            let s = b.shape().to_vec();
            assert!(s.len() >= 3, "H153: {name} must be ≥3-D; got {:?}", s);
            assert_eq!(
                s[0], 4,
                "H153 FALSIFIED: {name} shape[0] must be n_seqs=4 (n_seqs \
                 landed on wrong axis); got {:?}",
                s
            );
        }
    }

    /// **H154** — `MultiSeqMlxKvCache` per-slot byte isolation:
    /// host-side writes of a deterministic non-zero pattern into slot
    /// 0's k_packed / k_norms / v_packed / v_norms regions leave slot
    /// 1's bytes byte-identical.  Mirrors H147 for the legacy 4-bit
    /// variant.
    ///
    /// Falsifier: any byte change in slot 1's region after writing
    /// to slot 0's region ⇒ H154 broken; the per-slot byte-offset
    /// formula does not produce disjoint regions.
    #[test]
    fn h154_multi_seq_mlx_kv_per_slot_byte_isolation() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 4usize;
        let mut cache = alloc_multi_seq_mlx_kv_for_layer(
            &dev, 0, nkv, hd, cap, false, /*norms_per_pos=*/ 1, 2,
        )
        .expect("H154: alloc n_seqs=2");
        assert_eq!(cache.n_seqs, 2);

        // Per-slot region sizes:
        //   k_packed: slot bytes = nkv * cap * hd/2 (U8 = 1 byte/elem)
        //   v_packed: same
        //   k_norms:  slot bytes = nkv * cap * 1 * 4 (F32 = 4 bytes/elem)
        //   v_norms:  same
        let slot_kp_bytes = nkv * cap * (hd / 2);
        let slot_vp_bytes = nkv * cap * (hd / 2);
        let slot_kn_bytes = nkv * cap * 1 * 4;
        let slot_vn_bytes = nkv * cap * 1 * 4;

        // Sanity: total bytes match 2 * slot_bytes for each buffer.
        assert_eq!(
            cache.k_packed.byte_len(),
            2 * slot_kp_bytes,
            "H154: k_packed total"
        );
        assert_eq!(
            cache.v_packed.byte_len(),
            2 * slot_vp_bytes,
            "H154: v_packed total"
        );
        assert_eq!(
            cache.k_norms.byte_len(),
            2 * slot_kn_bytes,
            "H154: k_norms total"
        );
        assert_eq!(
            cache.v_norms.byte_len(),
            2 * slot_vn_bytes,
            "H154: v_norms total"
        );

        for (buffer, start, end, sentinel) in [
            (&mut cache.k_packed, slot_kp_bytes, 2 * slot_kp_bytes, 0x91),
            (&mut cache.v_packed, slot_vp_bytes, 2 * slot_vp_bytes, 0xA2),
            (&mut cache.k_norms, slot_kn_bytes, 2 * slot_kn_bytes, 0xB3),
            (&mut cache.v_norms, slot_vn_bytes, 2 * slot_vn_bytes, 0xC4),
        ] {
            buffer.as_mut_slice::<u8>().expect("seed peer slot")[start..end].fill(sentinel);
        }

        // Write deterministic non-zero pattern into slot 0's region
        // for ALL FOUR buffers (interpret as u8 bytes for fixture
        // simplicity; the kernel writes U8 / F32 but byte-level
        // isolation is what we're pinning).
        {
            let s = cache.k_packed.as_mut_slice::<u8>().expect("kp u8 mut");
            for (i, b) in s[..slot_kp_bytes].iter_mut().enumerate() {
                *b = (((i * 7) % 251) + 1) as u8;
            }
        }
        {
            let s = cache.v_packed.as_mut_slice::<u8>().expect("vp u8 mut");
            for (i, b) in s[..slot_vp_bytes].iter_mut().enumerate() {
                *b = (((i * 11) % 251) + 1) as u8;
            }
        }
        {
            let s = cache.k_norms.as_mut_slice::<u8>().expect("kn u8 mut");
            for (i, b) in s[..slot_kn_bytes].iter_mut().enumerate() {
                *b = (((i * 13) % 251) + 1) as u8;
            }
        }
        {
            let s = cache.v_norms.as_mut_slice::<u8>().expect("vn u8 mut");
            for (i, b) in s[..slot_vn_bytes].iter_mut().enumerate() {
                *b = (((i * 17) % 251) + 1) as u8;
            }
        }

        // Snapshot slot 1's regions on all four buffers.
        let kp_slot1_before: Vec<u8> = cache.k_packed.as_slice::<u8>().expect("kp r")
            [slot_kp_bytes..2 * slot_kp_bytes]
            .to_vec();
        let vp_slot1_before: Vec<u8> = cache.v_packed.as_slice::<u8>().expect("vp r")
            [slot_vp_bytes..2 * slot_vp_bytes]
            .to_vec();
        let kn_slot1_before: Vec<u8> = cache.k_norms.as_slice::<u8>().expect("kn r")
            [slot_kn_bytes..2 * slot_kn_bytes]
            .to_vec();
        let vn_slot1_before: Vec<u8> = cache.v_norms.as_slice::<u8>().expect("vn r")
            [slot_vn_bytes..2 * slot_vn_bytes]
            .to_vec();

        // Sanity: slot 1 contains the explicit sentinels.
        assert!(
            kp_slot1_before.iter().all(|&b| b == 0x91),
            "H154 sanity: kp slot1 sentinel"
        );
        assert!(
            vp_slot1_before.iter().all(|&b| b == 0xA2),
            "H154 sanity: vp slot1 sentinel"
        );
        assert!(
            kn_slot1_before.iter().all(|&b| b == 0xB3),
            "H154 sanity: kn slot1 sentinel"
        );
        assert!(
            vn_slot1_before.iter().all(|&b| b == 0xC4),
            "H154 sanity: vn slot1 sentinel"
        );

        // A3b iter-3 cursor advance on slot 0 (no buffer mutation).
        cache
            .append_for_seq(SlotId(0), 3)
            .expect("H154: append slot 0");
        assert_eq!(cache.seq_lens[0], 3);
        assert_eq!(cache.seq_lens[1], 0);

        // H154 falsifier: slot 1's bytes must be byte-identical on ALL FOUR buffers.
        let kp_slot1_after: Vec<u8> = cache.k_packed.as_slice::<u8>().expect("kp r2")
            [slot_kp_bytes..2 * slot_kp_bytes]
            .to_vec();
        let vp_slot1_after: Vec<u8> = cache.v_packed.as_slice::<u8>().expect("vp r2")
            [slot_vp_bytes..2 * slot_vp_bytes]
            .to_vec();
        let kn_slot1_after: Vec<u8> = cache.k_norms.as_slice::<u8>().expect("kn r2")
            [slot_kn_bytes..2 * slot_kn_bytes]
            .to_vec();
        let vn_slot1_after: Vec<u8> = cache.v_norms.as_slice::<u8>().expect("vn r2")
            [slot_vn_bytes..2 * slot_vn_bytes]
            .to_vec();
        assert_eq!(
            kp_slot1_before, kp_slot1_after,
            "H154 FALSIFIED: slot 1 k_packed bytes changed after slot-0 write"
        );
        assert_eq!(
            vp_slot1_before, vp_slot1_after,
            "H154 FALSIFIED: slot 1 v_packed bytes changed after slot-0 write"
        );
        assert_eq!(
            kn_slot1_before, kn_slot1_after,
            "H154 FALSIFIED: slot 1 k_norms bytes changed after slot-0 write"
        );
        assert_eq!(
            vn_slot1_before, vn_slot1_after,
            "H154 FALSIFIED: slot 1 v_norms bytes changed after slot-0 write"
        );
    }

    /// **H155** — n_seqs=1 byte-equivalence: allocating
    /// `MultiSeqMlxKvCache` at `n_seqs=1` produces buffer byte counts
    /// EQUAL to a legacy `MlxKvCache` per-layer K packed + K norms + V
    /// packed + V norms allocation at the same
    /// `(nkv, cap, hd, norms_per_pos)` parameters.
    ///
    /// Pins the H155 hypothesis: the iter-A3b-3 sibling-struct lift is
    /// byte-equivalent at n_seqs=1 to the legacy production alloc site
    /// at `gemma4/model.rs:1272-1290` which emits `MlxKvCache` with
    /// the formulas:
    ///   packed_bytes = nkv * capacity * (hd / 2)
    ///   norms_bytes  = nkv * capacity * norms_per_pos * 4
    /// Phase B4c re-route will be byte-safe by construction.
    #[test]
    fn h155_multi_seq_mlx_kv_n_seqs_1_byte_equivalent_to_legacy() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let cap = 8usize;

        for &(hd, norms_per_pos) in &[(256usize, 1usize), (512usize, 2usize)] {
            let multi =
                alloc_multi_seq_mlx_kv_for_layer(&dev, 0, nkv, hd, cap, false, norms_per_pos, 1)
                    .expect("H155: alloc multi-seq n_seqs=1");

            // Legacy per-layer formulas (mirror gemma4/model.rs:1272-1290).
            let legacy_packed_bytes = nkv * cap * (hd / 2);
            let legacy_norms_bytes = nkv * cap * norms_per_pos * 4;

            assert_eq!(
                multi.k_packed.byte_len(),
                legacy_packed_bytes,
                "H155 FALSIFIED (hd={hd} norms_per_pos={norms_per_pos}): \
                 k_packed bytes {} != legacy {}",
                multi.k_packed.byte_len(),
                legacy_packed_bytes
            );
            assert_eq!(
                multi.v_packed.byte_len(),
                legacy_packed_bytes,
                "H155 FALSIFIED (hd={hd} norms_per_pos={norms_per_pos}): \
                 v_packed bytes {} != legacy {}",
                multi.v_packed.byte_len(),
                legacy_packed_bytes
            );
            assert_eq!(
                multi.k_norms.byte_len(),
                legacy_norms_bytes,
                "H155 FALSIFIED (hd={hd} norms_per_pos={norms_per_pos}): \
                 k_norms bytes {} != legacy {}",
                multi.k_norms.byte_len(),
                legacy_norms_bytes
            );
            assert_eq!(
                multi.v_norms.byte_len(),
                legacy_norms_bytes,
                "H155 FALSIFIED (hd={hd} norms_per_pos={norms_per_pos}): \
                 v_norms bytes {} != legacy {}",
                multi.v_norms.byte_len(),
                legacy_norms_bytes
            );

            // Total parity vs legacy MlxKvCache (K packed + K norms + V packed + V norms).
            let legacy_total = 2 * legacy_packed_bytes + 2 * legacy_norms_bytes;
            use crate::serve::kv_persist::lcp_registry::ByteSized;
            assert_eq!(
                multi.byte_len(),
                legacy_total as u64,
                "H155 FALSIFIED (hd={hd} norms_per_pos={norms_per_pos}): \
                 total byte_len {} != legacy 4-buffer sum {}",
                multi.byte_len(),
                legacy_total
            );
        }
    }

    /// **H156** — `MultiSeqKvCache` impl for `MultiSeqMlxKvCache`:
    /// `slot_count() == n_seqs` (NOT 1 — the multi-seq sibling is no
    /// longer clamped); bounds-first SlotOutOfRange on the OOR path;
    /// per-slot cursor advance + drop; fork same-slot Ok; fork cross-slot
    /// → CapabilityUnsupported naming A3c.  Mirrors H149 for the legacy
    /// 4-bit variant.
    #[test]
    fn h156_multi_seq_mlx_kv_multi_seq_kv_cache_impl() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let n_seqs = 4u32;
        let mut cache = alloc_multi_seq_mlx_kv_for_layer(&dev, 0, 2, 256, 8, false, 1, n_seqs)
            .expect("H156: alloc n_seqs=4");

        // 1. slot_count() == n_seqs (NOT the clamp's 1).
        assert_eq!(
            cache.slot_count(),
            n_seqs,
            "H156 FALSIFIED: slot_count must equal n_seqs={n_seqs}"
        );
        assert_eq!(cache.layout(), MultiSeqLayout::SeparateSlots);

        // 2. All slots start at cursor 0.
        for s in 0..n_seqs {
            assert_eq!(
                cache.seq_len(SlotId(s)).expect("seq_len in range"),
                0,
                "H156: slot {s} starts at cursor 0"
            );
        }

        // 3. Bounds-first OOR on seq_len.
        let err = cache.seq_len(SlotId(n_seqs)).expect_err("OOR n_seqs");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(n_seqs),
                max_slots: n_seqs,
            },
            "H156 FALSIFIED: seq_len OOR shape; got {err:?}"
        );

        // 4. Per-slot cursor advance independence.
        cache.append_for_seq(SlotId(0), 5).expect("append slot 0");
        cache.append_for_seq(SlotId(2), 3).expect("append slot 2");
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 5);
        assert_eq!(
            cache.seq_len(SlotId(1)).unwrap(),
            0,
            "H156 FALSIFIED: slot 1 cursor touched by slot 0/2 append"
        );
        assert_eq!(cache.seq_len(SlotId(2)).unwrap(), 3);
        assert_eq!(
            cache.seq_len(SlotId(3)).unwrap(),
            0,
            "H156 FALSIFIED: slot 3 cursor touched by slot 0/2 append"
        );

        // 5. Bounds-first OOR on append.
        let err = cache
            .append_for_seq(SlotId(n_seqs + 1), 1)
            .expect_err("append OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(n_seqs + 1),
                max_slots: n_seqs,
            }
        );

        // 6. drop_seq resets target cursor, leaves siblings.
        cache.drop_seq(SlotId(0)).expect("drop slot 0");
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 0, "H156: slot 0 reset");
        assert_eq!(
            cache.seq_len(SlotId(2)).unwrap(),
            3,
            "H156 FALSIFIED: slot 2 preserved through slot 0 drop"
        );

        // 7. Bounds-first OOR on drop.
        let err = cache.drop_seq(SlotId(99)).expect_err("drop OOR");
        assert!(matches!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(99),
                max_slots: 4
            }
        ));

        // 8. fork_seq same slot is a no-op Ok.
        cache
            .fork_seq(SlotId(1), SlotId(1))
            .expect("self-fork no-op");

        // 9. Bounds-first OOR on fork (src then dst).
        let err = cache
            .fork_seq(SlotId(99), SlotId(0))
            .expect_err("fork src OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(99),
                max_slots: 4
            }
        );
        let err = cache
            .fork_seq(SlotId(0), SlotId(99))
            .expect_err("fork dst OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(99),
                max_slots: 4
            }
        );

        // 10. fork cross-slot → Ok(()) post-iter-A3c (was previously
        // CapabilityUnsupported naming A3c per A3b iter-3 typed-clamp).
        cache
            .append_for_seq(SlotId(0), 5)
            .expect("re-seed slot 0 for fork");
        let slot0_before = cache.seq_len(SlotId(0)).unwrap();
        let slot2_before = cache.seq_len(SlotId(2)).unwrap();
        cache
            .fork_seq(SlotId(0), SlotId(1))
            .expect("iter-A3c closure: cross-slot fork must return Ok(())");
        assert_eq!(
            cache.seq_len(SlotId(1)).unwrap(),
            slot0_before,
            "H156 closure: fork must copy src's seq_len to dst"
        );
        assert_eq!(
            cache.seq_len(SlotId(0)).unwrap(),
            slot0_before,
            "H156 closure: fork must NOT mutate src's seq_len"
        );
        assert_eq!(
            cache.seq_len(SlotId(2)).unwrap(),
            slot2_before,
            "H156 closure: fork must NOT mutate non-src non-dst slots"
        );
    }

    /// **H157** — `MultiSeqMlxKvCache::reset_for_slot` inherent
    /// method: per-slot cursor reset with k_packed / k_norms / v_packed
    /// / v_norms byte preservation (cursor-masked discipline matching
    /// A3a / A3b iter-{1,2} siblings); bounds-first typed OOR;
    /// SlotId(0) at n_seqs=1 is byte-equivalence case (must succeed).
    ///
    /// Mirrors H150 for the legacy 4-bit variant.
    #[test]
    fn h157_multi_seq_mlx_kv_reset_for_slot_per_slot_isolation_and_bounds() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 4usize;
        let n_seqs = 4u32;
        let mut cache = alloc_multi_seq_mlx_kv_for_layer(&dev, 0, nkv, hd, cap, false, 1, n_seqs)
            .expect("alloc n_seqs=4");

        // Seed every slot's cursor + buffer bytes with distinct patterns.
        for s in 0..(n_seqs as usize) {
            cache.seq_lens[s] = (s as u32) + 11;
        }
        let slot_kp_bytes = nkv * cap * (hd / 2);
        let slot_vp_bytes = nkv * cap * (hd / 2);
        let slot_kn_bytes = nkv * cap * 1 * 4;
        let slot_vn_bytes = nkv * cap * 1 * 4;
        {
            let s = cache.k_packed.as_mut_slice::<u8>().expect("kp u8");
            for slot in 0..(n_seqs as usize) {
                let start = slot * slot_kp_bytes;
                for (i, b) in s[start..start + slot_kp_bytes].iter_mut().enumerate() {
                    *b = (((slot * 17 + i) % 251) + 1) as u8;
                }
            }
        }
        {
            let s = cache.v_packed.as_mut_slice::<u8>().expect("vp u8");
            for slot in 0..(n_seqs as usize) {
                let start = slot * slot_vp_bytes;
                for (i, b) in s[start..start + slot_vp_bytes].iter_mut().enumerate() {
                    *b = (((slot * 19 + i) % 251) + 1) as u8;
                }
            }
        }
        {
            let s = cache.k_norms.as_mut_slice::<u8>().expect("kn u8");
            for slot in 0..(n_seqs as usize) {
                let start = slot * slot_kn_bytes;
                for (i, b) in s[start..start + slot_kn_bytes].iter_mut().enumerate() {
                    *b = (((slot * 23 + i) % 251) + 1) as u8;
                }
            }
        }
        {
            let s = cache.v_norms.as_mut_slice::<u8>().expect("vn u8");
            for slot in 0..(n_seqs as usize) {
                let start = slot * slot_vn_bytes;
                for (i, b) in s[start..start + slot_vn_bytes].iter_mut().enumerate() {
                    *b = (((slot * 29 + i) % 251) + 1) as u8;
                }
            }
        }
        // Snapshot all four buffers before the reset.
        let kp_before: Vec<u8> = cache.k_packed.as_slice::<u8>().expect("kp r").to_vec();
        let vp_before: Vec<u8> = cache.v_packed.as_slice::<u8>().expect("vp r").to_vec();
        let kn_before: Vec<u8> = cache.k_norms.as_slice::<u8>().expect("kn r").to_vec();
        let vn_before: Vec<u8> = cache.v_norms.as_slice::<u8>().expect("vn r").to_vec();

        // Reset slot 1.
        cache
            .reset_for_slot(SlotId(1))
            .expect("reset_for_slot(1) on n_seqs=4");

        // Slot 1's cursor must be 0; others untouched.
        for s in 0..(n_seqs as usize) {
            if s == 1 {
                assert_eq!(
                    cache.seq_lens[s], 0,
                    "H157 FALSIFIED: slot 1 cursor must be 0 after reset_for_slot(1)"
                );
            } else {
                assert_eq!(
                    cache.seq_lens[s],
                    (s as u32) + 11,
                    "H157 FALSIFIED: slot {s} cursor must be untouched"
                );
            }
        }

        // All four buffers byte-identical pre/post (cursor-masked
        // discipline — no byte zeroing on reset).
        let kp_after: Vec<u8> = cache.k_packed.as_slice::<u8>().expect("kp r2").to_vec();
        let vp_after: Vec<u8> = cache.v_packed.as_slice::<u8>().expect("vp r2").to_vec();
        let kn_after: Vec<u8> = cache.k_norms.as_slice::<u8>().expect("kn r2").to_vec();
        let vn_after: Vec<u8> = cache.v_norms.as_slice::<u8>().expect("vn r2").to_vec();
        assert_eq!(
            kp_before, kp_after,
            "H157 FALSIFIED: reset_for_slot must NOT zero k_packed bytes \
             (cursor-masked discipline; matches drop_seq invariant)"
        );
        assert_eq!(
            vp_before, vp_after,
            "H157 FALSIFIED: reset_for_slot must NOT zero v_packed bytes"
        );
        assert_eq!(
            kn_before, kn_after,
            "H157 FALSIFIED: reset_for_slot must NOT zero k_norms bytes"
        );
        assert_eq!(
            vn_before, vn_after,
            "H157 FALSIFIED: reset_for_slot must NOT zero v_norms bytes"
        );

        // Bounds-first OOR.
        let err = cache.reset_for_slot(SlotId(99)).expect_err("slot 99 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(99),
                max_slots: 4
            },
            "H157 FALSIFIED: reset OOR shape; got {err:?}"
        );

        // SlotId(0) at n_seqs=1 byte-equivalence case (must succeed).
        let mut cache1 = alloc_multi_seq_mlx_kv_for_layer(&dev, 0, 2, 256, 4, false, 1, 1)
            .expect("alloc n_seqs=1");
        cache1.seq_lens[0] = 7;
        cache1
            .reset_for_slot(SlotId(0))
            .expect("SlotId(0) at n_seqs=1 must succeed (byte-equivalence case)");
        assert_eq!(cache1.seq_lens[0], 0);
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-040 Phase A3c (2026-05-30) — fork_seq REAL cross-slot copy
    // for the four Gemma 4 multi-seq sibling structs.  H159-H162 pin
    // the iter-A3c closure (per-sibling) — one dispatcher closes the
    // typed-clamp on all four structs per dossier §2.3.3.
    //
    // Qwen35 sibling H158 + H163-H166 land in qwen35/kv_cache.rs (the
    // architecture-level full-attn + linear-attn + MTP fork proof).
    // ──────────────────────────────────────────────────────────────────

    /// **H159** — Gemma 4 `MultiSeqHbKvBuffers::fork_seq` cross-slot
    /// copy returns `Ok(())` AND produces byte-identical k_packed /
    /// k_norms / v_packed / v_norms regions between src and dst.
    /// Also asserts the cursor copy + src invariance (H163-H165 sub-pins
    /// at the per-sibling-struct level).
    #[test]
    fn h159_multi_seq_hb_kv_fork_seq_cross_slot_copies_all_buffers() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;
        let n_seqs = 4u32;
        let mut c =
            alloc_hb_kv_for_layer(&dev, 0, nkv, hd, cap, false, n_seqs).expect("alloc n_seqs=4");

        // Per-slot byte sizes (4-D, n_seqs outermost):
        //   k_packed/v_packed: nkv * cap * hd  (U8 = 1 byte/elem)
        //   k_norms/v_norms:   nkv * cap * 1 * 4 (F32, norms_per_pos=1)
        let slot_kp = nkv * cap * hd;
        let slot_vp = nkv * cap * hd;
        let slot_kn = nkv * cap * 1 * 4;
        let slot_vn = nkv * cap * 1 * 4;

        // Seed slot 0 with deterministic non-zero patterns.
        {
            let s = c.k_packed.as_mut_slice::<u8>().expect("kp u8");
            for (i, b) in s[..slot_kp].iter_mut().enumerate() {
                *b = (((i * 7) % 251) + 1) as u8;
            }
        }
        {
            let s = c.v_packed.as_mut_slice::<u8>().expect("vp u8");
            for (i, b) in s[..slot_vp].iter_mut().enumerate() {
                *b = (((i * 11) % 253) + 1) as u8;
            }
        }
        {
            let s = c.k_norms.as_mut_slice::<u8>().expect("kn u8");
            for (i, b) in s[..slot_kn].iter_mut().enumerate() {
                *b = (((i * 13) % 251) + 1) as u8;
            }
        }
        {
            let s = c.v_norms.as_mut_slice::<u8>().expect("vn u8");
            for (i, b) in s[..slot_vn].iter_mut().enumerate() {
                *b = (((i * 17) % 251) + 1) as u8;
            }
        }
        c.append_for_seq(SlotId(0), 5).unwrap();

        // Snapshot src bytes pre-fork.
        let src_kp = c.k_packed.as_slice::<u8>().unwrap()[..slot_kp].to_vec();
        let src_vp = c.v_packed.as_slice::<u8>().unwrap()[..slot_vp].to_vec();
        let src_kn = c.k_norms.as_slice::<u8>().unwrap()[..slot_kn].to_vec();
        let src_vn = c.v_norms.as_slice::<u8>().unwrap()[..slot_vn].to_vec();

        // iter-A3c closure: fork returns Ok(()).
        c.fork_seq(SlotId(0), SlotId(2))
            .expect("H159: fork must succeed post-A3c");

        // dst (slot 2) bytes byte-identical to src (slot 0).
        let dst_kp = c.k_packed.as_slice::<u8>().unwrap()[2 * slot_kp..3 * slot_kp].to_vec();
        let dst_vp = c.v_packed.as_slice::<u8>().unwrap()[2 * slot_vp..3 * slot_vp].to_vec();
        let dst_kn = c.k_norms.as_slice::<u8>().unwrap()[2 * slot_kn..3 * slot_kn].to_vec();
        let dst_vn = c.v_norms.as_slice::<u8>().unwrap()[2 * slot_vn..3 * slot_vn].to_vec();
        assert_eq!(src_kp, dst_kp, "H159 FALSIFIED: k_packed dst != src");
        assert_eq!(src_vp, dst_vp, "H159 FALSIFIED: v_packed dst != src");
        assert_eq!(src_kn, dst_kn, "H159 FALSIFIED: k_norms dst != src");
        assert_eq!(src_vn, dst_vn, "H159 FALSIFIED: v_norms dst != src");

        // Cursor copy.
        assert_eq!(c.seq_len(SlotId(2)).unwrap(), 5, "H159: cursor copied");
        // src cursor unchanged.
        assert_eq!(
            c.seq_len(SlotId(0)).unwrap(),
            5,
            "H159: src cursor unchanged"
        );
        // src bytes unchanged.
        let src_kp_after = c.k_packed.as_slice::<u8>().unwrap()[..slot_kp].to_vec();
        assert_eq!(src_kp, src_kp_after, "H159 FALSIFIED: src k_packed mutated");
    }

    /// **H160** — Gemma 4 `MultiSeqHybridKvBuffers::fork_seq` cross-
    /// slot copy returns `Ok(())`, copies only cursor-visible K/V rows,
    /// and leaves the lazy destination tail untouched.
    #[test]
    fn h160_multi_seq_hybrid_kv_fork_seq_copies_only_live_prefix() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;
        let n_seqs = 4u32;
        let mut c = alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, nkv, hd, cap, false, n_seqs)
            .expect("alloc n_seqs=4");

        // Per-slot byte sizes (4-D, n_seqs outermost):
        //   k F16:          nkv * cap * hd * 2
        //   v_packed U8:    nkv * cap * hd
        //   v_norms F32:    nkv * cap * 1 * 4
        let slot_k_bytes = nkv * cap * hd * 2;
        let slot_vp_bytes = nkv * cap * hd;
        let slot_vn_bytes = nkv * cap * 1 * 4;

        // Seed slot 1 with deterministic non-zero patterns.
        {
            let s = c.k.as_mut_slice::<u8>().expect("k u8");
            let start = 1 * slot_k_bytes;
            for (i, b) in s[start..start + slot_k_bytes].iter_mut().enumerate() {
                *b = (((i * 23) % 251) + 1) as u8;
            }
        }
        {
            let s = c.v_packed.as_mut_slice::<u8>().expect("vp u8");
            let start = 1 * slot_vp_bytes;
            for (i, b) in s[start..start + slot_vp_bytes].iter_mut().enumerate() {
                *b = (((i * 29) % 253) + 1) as u8;
            }
        }
        {
            let s = c.v_norms.as_mut_slice::<u8>().expect("vn u8");
            let start = 1 * slot_vn_bytes;
            for (i, b) in s[start..start + slot_vn_bytes].iter_mut().enumerate() {
                *b = (((i * 31) % 251) + 1) as u8;
            }
        }
        let live = 5usize;
        for buf in [&mut c.k, &mut c.v_packed, &mut c.v_norms] {
            let slot_bytes = buf.byte_len() / n_seqs as usize;
            let dst = 3 * slot_bytes;
            buf.as_mut_slice::<u8>().unwrap()[dst..dst + slot_bytes].fill(0xa5);
        }
        c.append_for_seq(SlotId(1), live as u32).unwrap();

        let src_k = c.k.as_slice::<u8>().unwrap()[slot_k_bytes..2 * slot_k_bytes].to_vec();
        let src_vp =
            c.v_packed.as_slice::<u8>().unwrap()[slot_vp_bytes..2 * slot_vp_bytes].to_vec();
        let src_vn = c.v_norms.as_slice::<u8>().unwrap()[slot_vn_bytes..2 * slot_vn_bytes].to_vec();

        c.fork_seq(SlotId(1), SlotId(3))
            .expect("H160: fork must succeed post-A3c");

        // dst (slot 3): each head's live prefix is copied, while the lazy
        // tail remains the destination sentinel rather than being read from
        // the source allocation.
        let dst_k = c.k.as_slice::<u8>().unwrap()[3 * slot_k_bytes..4 * slot_k_bytes].to_vec();
        let dst_vp =
            c.v_packed.as_slice::<u8>().unwrap()[3 * slot_vp_bytes..4 * slot_vp_bytes].to_vec();
        let dst_vn =
            c.v_norms.as_slice::<u8>().unwrap()[3 * slot_vn_bytes..4 * slot_vn_bytes].to_vec();
        for head in 0..nkv {
            let k_row = hd * 2;
            let vp_row = hd;
            let vn_row = 4;
            let k_base = head * cap * k_row;
            let vp_base = head * cap * vp_row;
            let vn_base = head * cap * vn_row;
            assert_eq!(
                &dst_k[k_base..k_base + live * k_row],
                &src_k[k_base..k_base + live * k_row]
            );
            assert!(dst_k[k_base + live * k_row..k_base + cap * k_row]
                .iter()
                .all(|&b| b == 0xa5));
            assert_eq!(
                &dst_vp[vp_base..vp_base + live * vp_row],
                &src_vp[vp_base..vp_base + live * vp_row]
            );
            assert!(dst_vp[vp_base + live * vp_row..vp_base + cap * vp_row]
                .iter()
                .all(|&b| b == 0xa5));
            assert_eq!(
                &dst_vn[vn_base..vn_base + live * vn_row],
                &src_vn[vn_base..vn_base + live * vn_row]
            );
            assert!(dst_vn[vn_base + live * vn_row..vn_base + cap * vn_row]
                .iter()
                .all(|&b| b == 0xa5));
        }

        // Cursor + src invariance.
        assert_eq!(
            c.seq_len(SlotId(3)).unwrap(),
            live as u32,
            "H160: cursor copied"
        );
        assert_eq!(
            c.seq_len(SlotId(1)).unwrap(),
            live as u32,
            "H160: src cursor unchanged"
        );
        let src_k_after = c.k.as_slice::<u8>().unwrap()[slot_k_bytes..2 * slot_k_bytes].to_vec();
        assert_eq!(src_k, src_k_after, "H160 FALSIFIED: src k mutated");
    }

    /// **H161** — Gemma 4 `MultiSeqDenseKvBuffers::fork_seq` cross-
    /// slot copy returns `Ok(())`, copies only cursor-visible dense K/V
    /// rows, and preserves the destination's lazy tail.
    #[test]
    fn h161_multi_seq_dense_kv_fork_seq_copies_only_live_prefix() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 4usize;
        let n_seqs = 4u32;
        let mut c =
            alloc_multi_seq_dense_kv_for_layer(&dev, 0, nkv, hd, cap, false, DType::F32, n_seqs)
                .expect("alloc n_seqs=4");

        // Per-slot byte sizes (F32 = 4 bytes/elem):
        let slot_k_bytes = nkv * cap * hd * 4;
        let slot_v_bytes = nkv * cap * hd * 4;

        // Seed slot 2 with deterministic non-zero patterns.
        {
            let s = c.k.as_mut_slice::<u8>().expect("k u8");
            let start = 2 * slot_k_bytes;
            for (i, b) in s[start..start + slot_k_bytes].iter_mut().enumerate() {
                *b = (((i * 37) % 251) + 1) as u8;
            }
        }
        {
            let s = c.v.as_mut_slice::<u8>().expect("v u8");
            let start = 2 * slot_v_bytes;
            for (i, b) in s[start..start + slot_v_bytes].iter_mut().enumerate() {
                *b = (((i * 41) % 253) + 1) as u8;
            }
        }
        let live = 3usize;
        c.k.as_mut_slice::<u8>().unwrap()[..slot_k_bytes].fill(0xa6);
        c.v.as_mut_slice::<u8>().unwrap()[..slot_v_bytes].fill(0xa6);
        c.append_for_seq(SlotId(2), live as u32).unwrap();

        let src_k = c.k.as_slice::<u8>().unwrap()[2 * slot_k_bytes..3 * slot_k_bytes].to_vec();
        let src_v = c.v.as_slice::<u8>().unwrap()[2 * slot_v_bytes..3 * slot_v_bytes].to_vec();

        c.fork_seq(SlotId(2), SlotId(0))
            .expect("H161: fork must succeed post-A3c");

        let dst_k = c.k.as_slice::<u8>().unwrap()[..slot_k_bytes].to_vec();
        let dst_v = c.v.as_slice::<u8>().unwrap()[..slot_v_bytes].to_vec();
        for head in 0..nkv {
            let row = hd * 4;
            let base = head * cap * row;
            assert_eq!(
                &dst_k[base..base + live * row],
                &src_k[base..base + live * row]
            );
            assert!(dst_k[base + live * row..base + cap * row]
                .iter()
                .all(|&b| b == 0xa6));
            assert_eq!(
                &dst_v[base..base + live * row],
                &src_v[base..base + live * row]
            );
            assert!(dst_v[base + live * row..base + cap * row]
                .iter()
                .all(|&b| b == 0xa6));
        }

        assert_eq!(c.seq_len(SlotId(0)).unwrap(), 3, "H161: cursor copied");
        assert_eq!(
            c.seq_len(SlotId(2)).unwrap(),
            3,
            "H161: src cursor unchanged"
        );
        let src_k_after =
            c.k.as_slice::<u8>().unwrap()[2 * slot_k_bytes..3 * slot_k_bytes].to_vec();
        assert_eq!(src_k, src_k_after, "H161 FALSIFIED: src k mutated");
    }

    /// **H162** — Gemma 4 `MultiSeqMlxKvCache::fork_seq` cross-slot
    /// copy returns `Ok(())`, copies cursor-visible packed/norm rows,
    /// and preserves the destination's lazy tail.
    #[test]
    fn h162_multi_seq_mlx_kv_fork_seq_copies_only_live_prefix() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = match skip_dev() {
            Some(d) => d,
            None => return,
        };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 4usize;
        let n_seqs = 4u32;
        let mut c = alloc_multi_seq_mlx_kv_for_layer(&dev, 0, nkv, hd, cap, false, 1, n_seqs)
            .expect("alloc n_seqs=4");

        // Per-slot byte sizes (4-D, n_seqs outermost):
        //   k_packed/v_packed: nkv * cap * (hd/2)  (U8 = 1 byte/elem)
        //   k_norms/v_norms:   nkv * cap * 1 * 4   (F32, norms_per_pos=1)
        let slot_kp = nkv * cap * (hd / 2);
        let slot_vp = nkv * cap * (hd / 2);
        let slot_kn = nkv * cap * 1 * 4;
        let slot_vn = nkv * cap * 1 * 4;

        // Seed slot 0 with deterministic non-zero patterns.
        {
            let s = c.k_packed.as_mut_slice::<u8>().expect("kp u8");
            for (i, b) in s[..slot_kp].iter_mut().enumerate() {
                *b = (((i * 43) % 251) + 1) as u8;
            }
        }
        {
            let s = c.v_packed.as_mut_slice::<u8>().expect("vp u8");
            for (i, b) in s[..slot_vp].iter_mut().enumerate() {
                *b = (((i * 47) % 253) + 1) as u8;
            }
        }
        {
            let s = c.k_norms.as_mut_slice::<u8>().expect("kn u8");
            for (i, b) in s[..slot_kn].iter_mut().enumerate() {
                *b = (((i * 53) % 251) + 1) as u8;
            }
        }
        {
            let s = c.v_norms.as_mut_slice::<u8>().expect("vn u8");
            for (i, b) in s[..slot_vn].iter_mut().enumerate() {
                *b = (((i * 59) % 251) + 1) as u8;
            }
        }
        let live = 3usize;
        for buf in [
            &mut c.k_packed,
            &mut c.v_packed,
            &mut c.k_norms,
            &mut c.v_norms,
        ] {
            let slot_bytes = buf.byte_len() / n_seqs as usize;
            let dst = 3 * slot_bytes;
            buf.as_mut_slice::<u8>().unwrap()[dst..dst + slot_bytes].fill(0xa7);
        }
        c.append_for_seq(SlotId(0), live as u32).unwrap();

        let src_kp = c.k_packed.as_slice::<u8>().unwrap()[..slot_kp].to_vec();
        let src_vp = c.v_packed.as_slice::<u8>().unwrap()[..slot_vp].to_vec();
        let src_kn = c.k_norms.as_slice::<u8>().unwrap()[..slot_kn].to_vec();
        let src_vn = c.v_norms.as_slice::<u8>().unwrap()[..slot_vn].to_vec();

        c.fork_seq(SlotId(0), SlotId(3))
            .expect("H162: fork must succeed post-A3c");

        let dst_kp = c.k_packed.as_slice::<u8>().unwrap()[3 * slot_kp..4 * slot_kp].to_vec();
        let dst_vp = c.v_packed.as_slice::<u8>().unwrap()[3 * slot_vp..4 * slot_vp].to_vec();
        let dst_kn = c.k_norms.as_slice::<u8>().unwrap()[3 * slot_kn..4 * slot_kn].to_vec();
        let dst_vn = c.v_norms.as_slice::<u8>().unwrap()[3 * slot_vn..4 * slot_vn].to_vec();
        for head in 0..nkv {
            let packed_row = hd / 2;
            let norm_row = 4;
            let packed_base = head * cap * packed_row;
            let norm_base = head * cap * norm_row;
            assert_eq!(
                &dst_kp[packed_base..packed_base + live * packed_row],
                &src_kp[packed_base..packed_base + live * packed_row]
            );
            assert!(
                dst_kp[packed_base + live * packed_row..packed_base + cap * packed_row]
                    .iter()
                    .all(|&b| b == 0xa7)
            );
            assert_eq!(
                &dst_vp[packed_base..packed_base + live * packed_row],
                &src_vp[packed_base..packed_base + live * packed_row]
            );
            assert!(
                dst_vp[packed_base + live * packed_row..packed_base + cap * packed_row]
                    .iter()
                    .all(|&b| b == 0xa7)
            );
            assert_eq!(
                &dst_kn[norm_base..norm_base + live * norm_row],
                &src_kn[norm_base..norm_base + live * norm_row]
            );
            assert!(
                dst_kn[norm_base + live * norm_row..norm_base + cap * norm_row]
                    .iter()
                    .all(|&b| b == 0xa7)
            );
            assert_eq!(
                &dst_vn[norm_base..norm_base + live * norm_row],
                &src_vn[norm_base..norm_base + live * norm_row]
            );
            assert!(
                dst_vn[norm_base + live * norm_row..norm_base + cap * norm_row]
                    .iter()
                    .all(|&b| b == 0xa7)
            );
        }

        assert_eq!(
            c.seq_len(SlotId(3)).unwrap(),
            live as u32,
            "H162: cursor copied"
        );
        assert_eq!(
            c.seq_len(SlotId(0)).unwrap(),
            live as u32,
            "H162: src cursor unchanged"
        );
        let src_kp_after = c.k_packed.as_slice::<u8>().unwrap()[..slot_kp].to_vec();
        assert_eq!(src_kp, src_kp_after, "H162 FALSIFIED: src k_packed mutated");
    }

    // -----------------------------------------------------------------
    // "gemma-hybrid-lcp" (2026-08-03) — GemmaLcpLayerKv payload contract
    // -----------------------------------------------------------------

    #[test]
    fn hybrid_lcp_snapshot_estimate_counts_packed_v_and_all_norm_blocks() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let (num_kv_heads, capacity, head_dim, norms_per_pos) = (2usize, 10usize, 512usize, 2usize);
        let bytes = hybrid_lcp_snapshot_layer_bytes(
            num_kv_heads,
            capacity,
            head_dim,
            norms_per_pos,
            DType::F16,
            DType::U8,
        )
        .expect("packed hybrid snapshot estimate");
        let k = num_kv_heads * capacity * head_dim * DType::F16.size_of();
        let v = num_kv_heads * capacity * head_dim * DType::U8.size_of();
        let norms = num_kv_heads * capacity * norms_per_pos * std::mem::size_of::<f32>();
        assert_eq!(bytes, (k + v + norms) as u64);
    }

    #[test]
    fn hybrid_lcp_snapshot_estimate_counts_f16_v_and_one_dummy() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let (num_kv_heads, capacity, head_dim, norms_per_pos) = (2usize, 10usize, 512usize, 2usize);
        let bytes = hybrid_lcp_snapshot_layer_bytes(
            num_kv_heads,
            capacity,
            head_dim,
            norms_per_pos,
            DType::F16,
            DType::F16,
        )
        .expect("full-F16 hybrid snapshot estimate");
        let k = num_kv_heads * capacity * head_dim * DType::F16.size_of();
        let v = num_kv_heads * capacity * head_dim * DType::F16.size_of();
        assert_eq!(
            bytes,
            (k + v + std::mem::size_of::<f32>()) as u64,
            "full-F16 snapshot owns one canonical norms dummy, not a token-shaped norms buffer"
        );
    }

    #[test]
    fn hybrid_lcp_snapshot_copy_preserves_packed_and_f16_v_layouts() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Some(device) = skip_dev() else {
            return;
        };
        const HEADS: usize = 2;
        const LIVE_CAPACITY: usize = 4;
        const SNAPSHOT_CAPACITY: usize = 6;
        const SEQUENCE_LEN: usize = 3;
        const HEAD_DIM: usize = 256;

        let assert_head_prefixes = |source: &MlxBuffer,
                                    snapshot: &MlxBuffer,
                                    inner: usize,
                                    element_bytes: usize| {
            let source = source.as_slice::<u8>().expect("live snapshot source bytes");
            let snapshot = snapshot
                .as_slice::<u8>()
                .expect("copied snapshot destination bytes");
            let copy_bytes = SEQUENCE_LEN * inner * element_bytes;
            let source_stride = LIVE_CAPACITY * inner * element_bytes;
            let snapshot_stride = SNAPSHOT_CAPACITY * inner * element_bytes;
            for head in 0..HEADS {
                assert_eq!(
                    &snapshot[head * snapshot_stride..head * snapshot_stride + copy_bytes],
                    &source[head * source_stride..head * source_stride + copy_bytes],
                    "head {head} populated prefix changed during LCP snapshot copy"
                );
            }
        };

        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        for full_f16_v in [false, true] {
            if full_f16_v {
                std::env::set_var("HF2Q_FULL_F16_KV", "1");
            } else {
                std::env::remove_var("HF2Q_FULL_F16_KV");
            }
            let mut live = alloc_hybrid_kv_for_layer(
                &device,
                usize::from(full_f16_v),
                HEADS,
                HEAD_DIM,
                LIVE_CAPACITY,
                true,
            )
            .expect("allocate live hybrid layer");
            for (index, byte) in live
                .k
                .as_mut_slice::<u8>()
                .expect("live K bytes")
                .iter_mut()
                .enumerate()
            {
                *byte = (index as u8).wrapping_mul(17).wrapping_add(3);
            }
            for (index, byte) in live
                .v_packed
                .as_mut_slice::<u8>()
                .expect("live V bytes")
                .iter_mut()
                .enumerate()
            {
                *byte = (index as u8).wrapping_mul(29).wrapping_add(5);
            }
            for (index, byte) in live
                .v_norms
                .as_mut_slice::<u8>()
                .expect("live V norms bytes")
                .iter_mut()
                .enumerate()
            {
                *byte = (index as u8).wrapping_mul(11).wrapping_add(7);
            }

            let copied = snapshot_hybrid_kv_for_lcp(
                &device,
                std::slice::from_ref(&live),
                SEQUENCE_LEN,
                SNAPSHOT_CAPACITY,
            )
            .expect("copy hybrid LCP snapshot");
            assert_eq!(copied.len(), 1);
            let copied = &copied[0];
            assert_eq!(copied.k.dtype(), live.k.dtype());
            assert_eq!(copied.v_packed.dtype(), live.v_packed.dtype());
            assert_eq!(copied.v_norms.dtype(), DType::F32);
            assert_eq!(copied.capacity, SNAPSHOT_CAPACITY);
            assert_eq!(copied.is_sliding, live.is_sliding);
            assert_eq!(copied.norms_per_pos, live.norms_per_pos);
            assert_head_prefixes(
                &live.k,
                &copied.k,
                HEAD_DIM,
                live.k.dtype().size_of(),
            );
            assert_head_prefixes(
                &live.v_packed,
                &copied.v_packed,
                HEAD_DIM,
                live.v_packed.dtype().size_of(),
            );
            if full_f16_v {
                assert_eq!(copied.v_norms.shape(), &[1]);
                assert_eq!(
                    copied.v_norms.as_slice::<u8>().expect("copied dummy bytes"),
                    live.v_norms.as_slice::<u8>().expect("live dummy bytes")
                );
            } else {
                assert_head_prefixes(
                    &live.v_norms,
                    &copied.v_norms,
                    live.norms_per_pos,
                    std::mem::size_of::<f32>(),
                );
            }
        }
        std::env::remove_var("HF2Q_FULL_F16_KV");
    }

    /// ByteSized must sum both legs exactly (no estimation — the LCP
    /// registry's byte budget is enforced off this value).
    #[test]
    fn gemma_lcp_layer_kv_bytesized_sums_both_legs() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let dev = MlxDevice::new().expect("device");
        let (nkv, cap, hd) = (2usize, 8usize, 4usize);
        let dense = DenseKvBuffers {
            k: dev
                .alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd])
                .unwrap(),
            v: dev
                .alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd])
                .unwrap(),
            capacity: cap,
            is_sliding: false,
            dtype: DType::F32,
        };
        let hybrid = HybridKvBuffers {
            k: dev
                .alloc_buffer(nkv * cap * hd * 2, DType::F16, vec![nkv, cap, hd])
                .unwrap(),
            v_packed: dev
                .alloc_buffer(nkv * cap * hd, DType::U8, vec![nkv, cap, hd])
                .unwrap(),
            v_norms: dev
                .alloc_buffer(nkv * cap * 4, DType::F32, vec![nkv, cap])
                .unwrap(),
            capacity: cap,
            is_sliding: false,
            norms_per_pos: 1,
            bf16_xlen_k: None,
            bf16_xlen_v: None,
        };
        use crate::serve::kv_persist::lcp_registry::ByteSized;
        let d_bytes = ByteSized::byte_len(&dense);
        let h_bytes = ByteSized::byte_len(&hybrid);
        assert_eq!(
            d_bytes,
            (nkv * cap * hd * 4 * 2) as u64,
            "dense = 2 F32 buffers"
        );
        assert_eq!(
            h_bytes,
            (nkv * cap * hd * 2 + nkv * cap * hd + nkv * cap * 4) as u64
        );

        let dense_only = GemmaLcpLayerKv::Dense(DenseKvBuffers {
            k: dev
                .alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd])
                .unwrap(),
            v: dev
                .alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd])
                .unwrap(),
            capacity: cap,
            is_sliding: false,
            dtype: DType::F32,
        });
        assert_eq!(ByteSized::byte_len(&dense_only), d_bytes);
        let both = GemmaLcpLayerKv::DenseAndHybrid(
            DenseKvBuffers {
                k: dev
                    .alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd])
                    .unwrap(),
                v: dev
                    .alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd])
                    .unwrap(),
                capacity: cap,
                is_sliding: false,
                dtype: DType::F32,
            },
            hybrid,
        );
        assert_eq!(
            ByteSized::byte_len(&both),
            d_bytes + h_bytes,
            "DenseAndHybrid byte_len must sum dense + hybrid legs exactly"
        );
        // Accessors.
        assert!(both.hybrid().is_some());
        assert!(dense_only.hybrid().is_none());
        assert_eq!(both.dense().capacity, cap);
    }
}
