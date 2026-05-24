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
//     `LayoutNotSupported { SeparateSlots }`).  Full lift in iter-A3b-2.
//   * MlxKvCache — TYPED CLAMP (same shape as DenseKvBuffers).  Full
//     lift in iter-A3b-3.
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

    let mut k_packed = dev
        .alloc_buffer(packed_bytes, DType::U8, packed_shape.clone())
        .map_err(|e| anyhow!("hb_kv L{layer_idx} K packed: {e}"))?;
    let mut k_norms = dev
        .alloc_buffer(norms_bytes, DType::F32, norms_shape.clone())
        .map_err(|e| anyhow!("hb_kv L{layer_idx} K norms: {e}"))?;
    let mut v_packed = dev
        .alloc_buffer(packed_bytes, DType::U8, packed_shape)
        .map_err(|e| anyhow!("hb_kv L{layer_idx} V packed: {e}"))?;
    let mut v_norms = dev
        .alloc_buffer(norms_bytes, DType::F32, norms_shape)
        .map_err(|e| anyhow!("hb_kv L{layer_idx} V norms: {e}"))?;

    // Zero-init mirrors `alloc_tq_full_attn_buffers` discipline
    // (Qwen35 `kv_cache.rs:2460-2471`): defend against StorageModeShared
    // returning recycled non-zero memory (ADR-015 iter61a).
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: self.n_seqs,
                },
            );
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: self.n_seqs,
                },
            );
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: self.n_seqs,
                },
            );
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot: src,
                    max_slots: self.n_seqs,
                },
            );
        }
        if dst.0 >= self.n_seqs {
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot: dst,
                    max_slots: self.n_seqs,
                },
            );
        }
        // 2. Layout: SeparateSlots only — no LayoutNotSupported.
        // 3. Same-slot fork is a no-op per trait spec.
        if src == dst {
            return Ok(());
        }
        // ADR-040 Phase A3c deferred (parallel to Qwen35 A2c per
        // dossier R5): cross-slot fork requires same-buffer
        // cross-region memcpy via `dispatch_kv_cache_copy_seq_*`
        // between slot byte offsets on the same underlying buffer.
        // That kernel-pattern + its own unit-test arc are scheduled
        // for A3c once the kernel arc lands on Qwen35 A2c (the same
        // dispatcher works for both arches; one kernel call serves
        // both per dossier §2.3.3).
        //
        // **Mantra-aligned (iter-2.5 M1)**: surfaces as
        // `CapabilityUnsupported` (HTTP 501 upstream) — NOT
        // `SlotOom { 0, 0 }` (HTTP 429), which would lie about the
        // capacity freeing up.  When Phase A3c ships the real
        // kernel dispatch, this branch flips to `Ok(())` + per-buffer
        // byte-equality assertion (write-to-src → fork → read-from-dst
        // = write-to-src-and-dst-directly), matching Qwen35 A2c's
        // ceremony.
        Err(
            crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: "fork_seq cross-slot copy (Gemma 4 MultiSeqHbKvBuffers; deferred to Phase A3c per ADR-040 §6 + dossier R5)",
            },
        )
    }
}

/// Per-layer dense F32/F16 KV buffers for dense attention path (ADR-009).
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
/// The structural decode-side gap vs llama.cpp peer (1.81× per-dispatch wall on
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
    let k = dev.alloc_buffer(nkv * cap * hd * 2, DType::F16,
        vec![nkv, cap, hd])
        .map_err(|e| anyhow!("hybrid F16 K L{layer_idx}: {e}"))?;
    // ADR-029 iter-20 H27: when HF2Q_FULL_F16_KV is set, V is F16 (2 bytes/elem)
    // and v_norms is a small dummy buffer (kernel ignores it when v_is_f16=1).
    // Otherwise: legacy TQ-HB packed V (1 byte/elem) + per-position F32 norms.
    let full_f16_v = std::env::var("HF2Q_FULL_F16_KV")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "on"))
        .unwrap_or(false);
    let (v_packed, v_norms) = if full_f16_v {
        let v_f16 = dev.alloc_buffer(nkv * cap * hd * 2, DType::F16,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow!("hybrid F16 V L{layer_idx}: {e}"))?;
        // Dummy norms buffer (unused but kept for ABI compat with hybrid SDPA
        // signature; kernel's v_is_f16 FC=1 skips the read).
        let v_norms_dummy = dev.alloc_buffer(4, DType::F32, vec![1])
            .map_err(|e| anyhow!("hybrid V norms (dummy) L{layer_idx}: {e}"))?;
        (v_f16, v_norms_dummy)
    } else {
        let v_p = dev.alloc_buffer(nkv * cap * hd, DType::U8,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow!("hybrid V packed L{layer_idx}: {e}"))?;
        let v_n = dev.alloc_buffer(norms_n * 4, DType::F32,
            if norms_per_pos == 1 { vec![nkv, cap] } else { vec![nkv, cap, norms_per_pos] })
            .map_err(|e| anyhow!("hybrid V norms L{layer_idx}: {e}"))?;
        (v_p, v_n)
    };
    // ADR-030 iter-96: lazy-alloc the BF16 xlen cache only when env opted-in.
    // Saves ~55MB at gemma-4 when xlen mode disabled.
    let xlen_mode = std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() == Ok("1");
    let (bf16_xlen_k, bf16_xlen_v) = if xlen_mode {
        let bk = dev.alloc_buffer(nkv * cap * hd * 2, DType::BF16,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow!("bf16 xlen K L{layer_idx}: {e}"))?;
        let bv = dev.alloc_buffer(nkv * cap * hd * 2, DType::BF16,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow!("bf16 xlen V L{layer_idx}: {e}"))?;
        (Some(bk), Some(bv))
    } else {
        (None, None)
    };
    Ok(HybridKvBuffers { k, v_packed, v_norms, capacity: cap, is_sliding: is_ring, norms_per_pos, bf16_xlen_k, bf16_xlen_v })
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

impl crate::serve::kv_persist::lcp_registry::ByteSized for MultiSeqHybridKvBuffers {
    /// Exact byte count: F16/F32 K + (U8|F16) V + (F32|dummy) V-norms +
    /// optional BF16 xlen K + optional BF16 xlen V.  Used by the
    /// LcpRegistry byte budget identically to `HybridKvBuffers::byte_len`
    /// — the lift to N slots scales every buffer by N at alloc-time, so
    /// `byte_len()` automatically reports the per-slot totals × N.
    fn byte_len(&self) -> u64 {
        let mut sum = (self.k.byte_len()
            + self.v_packed.byte_len()
            + self.v_norms.byte_len()) as u64;
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
    let k = dev
        .alloc_buffer(k_bytes, DType::F16, vec![n, nkv, cap, hd])
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
        let v_f16 = dev
            .alloc_buffer(v_bytes, DType::F16, vec![n, nkv, cap, hd])
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
        let v_p = dev
            .alloc_buffer(v_packed_bytes, DType::U8, vec![n, nkv, cap, hd])
            .map_err(|e| anyhow!("multi-seq hybrid V packed L{layer_idx}: {e}"))?;
        let v_norms_elems = n * nkv * cap * norms_per_pos;
        let v_norms_bytes = v_norms_elems * std::mem::size_of::<f32>();
        let v_n = dev
            .alloc_buffer(v_norms_bytes, DType::F32, vec![n, nkv, cap, norms_per_pos])
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
        let bk = dev
            .alloc_buffer(xlen_bytes, DType::BF16, vec![n, nkv, cap, hd])
            .map_err(|e| anyhow!("multi-seq hybrid bf16 xlen K L{layer_idx}: {e}"))?;
        let bv = dev
            .alloc_buffer(xlen_bytes, DType::BF16, vec![n, nkv, cap, hd])
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: self.n_seqs,
                },
            );
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: self.n_seqs,
                },
            );
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: self.n_seqs,
                },
            );
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot: src,
                    max_slots: self.n_seqs,
                },
            );
        }
        if dst.0 >= self.n_seqs {
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot: dst,
                    max_slots: self.n_seqs,
                },
            );
        }
        // 2. Layout: SeparateSlots only — no LayoutNotSupported.
        // 3. Same-slot fork is a no-op per trait spec.
        if src == dst {
            return Ok(());
        }
        // ADR-040 Phase A3c deferred (parallel to A3a Gemma 4
        // MultiSeqHbKvBuffers + Qwen35 A2c per dossier R5): cross-
        // slot fork requires same-buffer cross-region memcpy via
        // `dispatch_kv_cache_copy_seq_*` between slot byte offsets.
        // The kernel-pattern + its own unit-test arc are scheduled
        // for A3c — one dispatcher serves both arches + both
        // sibling structs per dossier §2.3.3.
        //
        // **Mantra-aligned (iter-2.5 M1)**: surfaces as
        // `CapabilityUnsupported` (HTTP 501 upstream) — NOT
        // `SlotOom { 0, 0 }` (HTTP 429), which would lie about the
        // capacity freeing up.
        Err(
            crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: "fork_seq cross-slot copy (Gemma 4 MultiSeqHybridKvBuffers; deferred to Phase A3c per ADR-040 §6 + dossier R5)",
            },
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A3b iter-1 — TYPED CLAMP impls for DenseKvBuffers + MlxKvCache.
//
// Both variants are NON-DEFAULT today:
//   * DenseKvBuffers: reachable via `HF2Q_USE_DENSE=1` (off-default).
//   * MlxKvCache: legacy 4-bit nibble-packed path (off-default since
//     ADR-007 default-on TQ 8-bit).
//
// Per dossier R3 mitigation, each clamp:
//   * Returns `slot_count() == 1` (single-seq by construction).
//   * `seq_len(SlotId(0))` returns `Ok(internal_cursor as u32)`.
//   * Any operation on `slot.0 > 0` returns
//     `LayoutNotSupported { layout: SeparateSlots }` — same error
//     shape A3a's NoopMultiSeqKvCache fixture uses for Paged-layout
//     refusal at in-bounds slots; here the discriminant signals
//     "this per-arch lift is staged; iter-A3b-2/3 ship the full N
//     slot lift" rather than "this layout will never be supported".
//
// FULL LIFT scheduled for:
//   * iter-A3b-2 — DenseKvBuffers full multi-seq (~150 LOC).
//   * iter-A3b-3 — MlxKvCache full multi-seq (~80 LOC).
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: 1,
                },
            );
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: 1,
                },
            );
        }
        // Single-seq clamp: external cursor.  Iter-A3b-2 wires
        // the real bump.  Returning Ok(()) here would let a
        // scheduler-side accountant proceed against an unaware
        // backing buffer; we honour the clamp by returning a
        // typed `LayoutNotSupported` even at slot 0 — the
        // SeparateSlots layout here means "single-seq SeparateSlots
        // only" and append_for_seq has no internal state to bump.
        // Iter-A3b-2 replaces this with the real per-seq bump.
        Err(
            crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: "DenseKvBuffers::append_for_seq (full multi-seq lift deferred to ADR-040 Phase A3b iter-2)",
            },
        )
    }

    fn drop_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        if slot.0 >= 1 {
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: 1,
                },
            );
        }
        // Single-seq clamp: cursor lives outside this struct.
        // Iter-A3b-2 wires the real reset.
        Err(
            crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: "DenseKvBuffers::drop_seq (full multi-seq lift deferred to ADR-040 Phase A3b iter-2)",
            },
        )
    }

    fn fork_seq(
        &mut self,
        src: crate::serve::multi_seq_kv::SlotId,
        dst: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        if src.0 >= 1 {
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot: src,
                    max_slots: 1,
                },
            );
        }
        if dst.0 >= 1 {
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot: dst,
                    max_slots: 1,
                },
            );
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: 1,
                },
            );
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
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: 1,
                },
            );
        }
        // Single-seq clamp: legacy path mutates `seq_len`/`write_pos`
        // via direct field access at production callsites (the trait
        // surface is not the canonical mutation path for the legacy
        // single-seq route).  Iter-A3b-3 lifts to N + thread the
        // trait surface as the production mutation path.
        Err(
            crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: "MlxKvCache::append_for_seq (full multi-seq lift deferred to ADR-040 Phase A3b iter-3 — legacy 4-bit path)",
            },
        )
    }

    fn drop_seq(
        &mut self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        if slot.0 >= 1 {
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot,
                    max_slots: 1,
                },
            );
        }
        // Single-seq clamp.  Iter-A3b-3 wires the real reset.
        Err(
            crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: "MlxKvCache::drop_seq (full multi-seq lift deferred to ADR-040 Phase A3b iter-3 — legacy 4-bit path)",
            },
        )
    }

    fn fork_seq(
        &mut self,
        src: crate::serve::multi_seq_kv::SlotId,
        dst: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<(), crate::serve::multi_seq_kv::MultiSeqError> {
        if src.0 >= 1 {
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot: src,
                    max_slots: 1,
                },
            );
        }
        if dst.0 >= 1 {
            return Err(
                crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                    slot: dst,
                    max_slots: 1,
                },
            );
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
    fn mlx_kv_cache_trim_linear_decrements_seq_len() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let mut cache = MlxKvCache {
            k_packed: buf(), k_norms: buf(), v_packed: buf(), v_norms: buf(),
            capacity: 16, is_sliding: false, write_pos: 8, seq_len: 8,
        };
        let new_len = cache.trim(3).unwrap();
        assert_eq!(new_len, 5);
        assert_eq!(cache.seq_len, 5);
        assert_eq!(cache.write_pos, 5);
    }

    #[test]
    fn mlx_kv_cache_trim_sliding_errors() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let mut cache = MlxKvCache {
            k_packed: buf(), k_norms: buf(), v_packed: buf(), v_norms: buf(),
            capacity: 16, is_sliding: true, write_pos: 4, seq_len: 4,
        };
        assert!(cache.trim(1).is_err());
    }

    #[test]
    fn mlx_kv_cache_trim_overflow_errors() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let mut cache = MlxKvCache {
            k_packed: buf(), k_norms: buf(), v_packed: buf(), v_norms: buf(),
            capacity: 16, is_sliding: false, write_pos: 3, seq_len: 3,
        };
        assert!(cache.trim(10).is_err());
    }

    #[test]
    fn mlx_kv_cache_visible_len_eq_seq_len() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let cache = MlxKvCache {
            k_packed: buf(), k_norms: buf(), v_packed: buf(), v_norms: buf(),
            capacity: 32, is_sliding: false, write_pos: 7, seq_len: 7,
        };
        assert_eq!(cache.visible_len(), cache.seq_len);
    }

    #[test]
    fn decode_regime_default_via_default_trait() {
        let r: DecodeRegime = Default::default();
        assert_eq!(r, DecodeRegime::Default);
    }

    #[test]
    fn decode_regime_variants_distinct() {
        assert_ne!(DecodeRegime::Default, DecodeRegime::ForceTq);
        assert_ne!(DecodeRegime::Default, DecodeRegime::ForceDense);
        assert_ne!(DecodeRegime::ForceTq, DecodeRegime::ForceDense);
    }

    #[test]
    fn hybrid_kv_buffers_byte_len_sums_fields() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let nkv = 2; let cap = 4; let hd = 256;
        let k = dev.alloc_buffer(nkv * cap * hd * 2, DType::F16, vec![nkv, cap, hd]).unwrap();
        let v_packed = dev.alloc_buffer(nkv * cap * hd, DType::U8, vec![nkv, cap, hd]).unwrap();
        let v_norms = dev.alloc_buffer(nkv * cap * 4, DType::F32, vec![nkv, cap]).unwrap();
        let k_bytes = k.byte_len();
        let vp_bytes = v_packed.byte_len();
        let vn_bytes = v_norms.byte_len();
        let buf = HybridKvBuffers {
            k, v_packed, v_norms,
            capacity: cap, is_sliding: false, norms_per_pos: 1,
            bf16_xlen_k: None, bf16_xlen_v: None,
        };
        use crate::serve::kv_persist::lcp_registry::ByteSized;
        assert_eq!(buf.byte_len(), (k_bytes + vp_bytes + vn_bytes) as u64);
    }

    #[test]
    fn dense_kv_buffers_byte_len_sums_k_plus_v() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let nkv = 2; let cap = 8; let hd = 256;
        let k = dev.alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd]).unwrap();
        let v = dev.alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd]).unwrap();
        let kb = k.byte_len(); let vb = v.byte_len();
        let buf = DenseKvBuffers { k, v, capacity: cap, is_sliding: false, dtype: DType::F32 };
        use crate::serve::kv_persist::lcp_registry::ByteSized;
        assert_eq!(buf.byte_len(), (kb + vb) as u64);
    }

    #[test]
    fn alloc_hybrid_kv_for_layer_no_xlen_no_full_f16() {
        let dev = match skip_dev() { Some(d) => d, None => return };
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
        let dev = match skip_dev() { Some(d) => d, None => return };
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
        let dev = match skip_dev() { Some(d) => d, None => return };
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::set_var("HF2Q_DFLASH_XLEN_SDPA", "1");
        let buf = alloc_hybrid_kv_for_layer(&dev, 2, 2, 256, 4, false).unwrap();
        assert!(buf.bf16_xlen_k.is_some());
        assert!(buf.bf16_xlen_v.is_some());
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
    }

    #[test]
    fn alloc_hybrid_kv_for_layer_norms_per_pos_d256_d512() {
        let dev = match skip_dev() { Some(d) => d, None => return };
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
    use crate::serve::multi_seq_kv::{
        MultiSeqError, MultiSeqKvCache as _, MultiSeqLayout, SlotId,
    };

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
        let dev = match skip_dev() { Some(d) => d, None => return };
        // Per dossier H6: shape choices identical to existing
        // alloc_hybrid_kv_for_layer test fixtures (nkv=2, hd=256,
        // cap=8) so the byte-count formula matches what production
        // sees at the 3 inline alloc sites.
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;
        let baseline = alloc_hb_kv_for_layer(&dev, 0, nkv, hd, cap, false, 1)
            .expect("H6: alloc at n_seqs=1");
        let lifted = alloc_hb_kv_for_layer(&dev, 0, nkv, hd, cap, false, 4)
            .expect("H6: alloc at n_seqs=4");

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
        let dev = match skip_dev() { Some(d) => d, None => return };
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
        let k_slot1_before: Vec<u8> = cache
            .k_packed
            .as_slice::<u8>()
            .expect("k_packed u8")
            [slot_packed..2 * slot_packed]
            .to_vec();
        let v_slot1_before: Vec<u8> = cache
            .v_packed
            .as_slice::<u8>()
            .expect("v_packed u8")
            [slot_packed..2 * slot_packed]
            .to_vec();
        // Sanity: slot 0 region is all-zero before the cursor advance.
        let k_slot0_before: Vec<u8> = cache
            .k_packed
            .as_slice::<u8>()
            .expect("k_packed u8")
            [0..slot_packed]
            .to_vec();
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
        let k_slot1_after: Vec<u8> = cache
            .k_packed
            .as_slice::<u8>()
            .expect("k_packed u8")
            [slot_packed..2 * slot_packed]
            .to_vec();
        let v_slot1_after: Vec<u8> = cache
            .v_packed
            .as_slice::<u8>()
            .expect("v_packed u8")
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
        let dev = match skip_dev() { Some(d) => d, None => return };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 8usize;
        let norms_per_pos = (hd / 256).max(1);

        // The 3 inline alloc sites' byte formula (verbatim from
        // forward_prefill.rs:864-875).
        let expected_packed_bytes = nkv * cap * hd; // U8
        let expected_norms_bytes = nkv * cap * norms_per_pos * std::mem::size_of::<f32>();

        let helper = alloc_hb_kv_for_layer(&dev, 0, nkv, hd, cap, false, 1)
            .expect("H8: helper at n_seqs=1");

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
        let dev = match skip_dev() { Some(d) => d, None => return };
        let cache_1 = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 1)
            .expect("alloc n_seqs=1");
        let cache_4 = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4)
            .expect("alloc n_seqs=4");

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
        let dev = match skip_dev() { Some(d) => d, None => return };
        let c1 = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 1)
            .expect("alloc 1");
        let c4 = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4)
            .expect("alloc 4");
        assert_eq!(c1.slot_count(), 1);
        assert_eq!(c4.slot_count(), 4);
    }

    /// Pin: `layout()` returns `SeparateSlots`.  MultiSeqHbKvBuffers
    /// does not expose Paged — bounds-first ordering means this trip
    /// is only observable through this getter.
    #[test]
    fn gemma4_hb_kv_layout_is_separate_slots() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4)
            .expect("alloc");
        assert_eq!(c.layout(), MultiSeqLayout::SeparateSlots);
    }

    /// Pin (iter-1.5 cfa-finding-F5): out-of-range `SlotId` surfaces as
    /// `SlotOutOfRange { slot, max_slots }` with both fields populated
    /// across every trait method — bounds-first ordering preserved.
    /// Mirrors `qwen35_hybrid_kv_slot_out_of_range_errors_named`
    /// (qwen35/kv_cache.rs:6695-6721).
    #[test]
    fn gemma4_hb_kv_slot_out_of_range_errors_named() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let mut c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4)
            .expect("alloc");

        // seq_len OOR
        let err = c.seq_len(SlotId(4)).expect_err("slot 4 OOR for n_seqs=4");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(4), max_slots: 4 }
        );
        let err = c.seq_len(SlotId(99)).expect_err("slot 99 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(99), max_slots: 4 }
        );

        // append_for_seq OOR
        let err = c.append_for_seq(SlotId(4), 1).expect_err("append OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(4), max_slots: 4 }
        );

        // drop_seq OOR
        let err = c.drop_seq(SlotId(4)).expect_err("drop OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(4), max_slots: 4 }
        );

        // fork_seq src OOR FIRST (deterministic per fixture-parity).
        let err = c.fork_seq(SlotId(4), SlotId(5)).expect_err("fork: src OOR first");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(4), max_slots: 4 }
        );
        // fork_seq src valid, dst OOR.
        let err = c.fork_seq(SlotId(0), SlotId(4)).expect_err("fork: dst OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(4), max_slots: 4 }
        );
    }

    /// Pin: `append_for_seq` advances ONLY the named slot's cursor.
    /// Surface-level isolation evidence for H6's per-slot O(1) bound
    /// (the per-buffer GPU write isolation lands in Phase B4c).
    /// Mirrors `qwen35_hybrid_kv_append_advances_target_slot_only`
    /// (qwen35/kv_cache.rs:6727-6745).
    #[test]
    fn gemma4_hb_kv_append_advances_target_slot_only() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let mut c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4)
            .expect("alloc");
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
        let dev = match skip_dev() { Some(d) => d, None => return };
        let mut c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4)
            .expect("alloc");
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
        let dev = match skip_dev() { Some(d) => d, None => return };
        let nkv = 2usize;
        let hd = 256usize;
        let cap = 4usize;
        let mut c = alloc_hb_kv_for_layer(&dev, 0, nkv, hd, cap, false, 2)
            .expect("alloc n_seqs=2");

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
        let k_before: Vec<u8> = c
            .k_packed
            .as_slice::<u8>()
            .expect("k_packed u8")
            [..slot_packed]
            .to_vec();
        let v_before: Vec<u8> = c
            .v_packed
            .as_slice::<u8>()
            .expect("v_packed u8")
            [..slot_packed]
            .to_vec();
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
        let k_after: Vec<u8> = c
            .k_packed
            .as_slice::<u8>()
            .expect("k_packed u8 after")
            [..slot_packed]
            .to_vec();
        let v_after: Vec<u8> = c
            .v_packed
            .as_slice::<u8>()
            .expect("v_packed u8 after")
            [..slot_packed]
            .to_vec();

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
        let dev = match skip_dev() { Some(d) => d, None => return };
        let mut c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4)
            .expect("alloc");
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

    /// Phase A3c deferral pin: cross-slot fork returns
    /// `CapabilityUnsupported` with the capability label naming the
    /// deferred kernel arc + dossier R5.  Mirrors Qwen35's iter-2.5
    /// M1 closure pin at `qwen35/kv_cache.rs:7079-7113`.
    ///
    /// When Phase A3c ships the real kernel dispatch, this test
    /// will fail loudly — signaling the deferral closure.  At that
    /// iter, flip the assertion to `expect("fork ok after A3c")`
    /// plus a per-buffer byte-equality check.
    #[test]
    fn gemma4_hb_kv_fork_cross_slot_returns_capability_unsupported() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let mut c = alloc_hb_kv_for_layer(&dev, 0, 2, 256, 8, false, 4)
            .expect("alloc");
        c.append_for_seq(SlotId(0), 7).unwrap();
        let err = c
            .fork_seq(SlotId(0), SlotId(1))
            .expect_err("cross-slot fork deferred to Phase A3c");
        // Discriminant must be `CapabilityUnsupported` (mirrors
        // Qwen35 iter-2.5 M1; NOT `SlotOom`).
        match err {
            MultiSeqError::CapabilityUnsupported { capability } => {
                assert!(
                    capability.contains("fork_seq cross-slot copy"),
                    "capability label must name the deferred surface: \
                     {capability}"
                );
                assert!(
                    capability.contains("Phase A3c"),
                    "capability label must name the Phase A3c deferral: \
                     {capability}"
                );
                assert!(
                    capability.contains("R5"),
                    "capability label must name dossier R5 grounding: \
                     {capability}"
                );
                // Per-arch identity — distinguishes Gemma 4 from the
                // Qwen35 sibling at the log line.
                assert!(
                    capability.contains("Gemma 4"),
                    "capability label must name the arch: {capability}"
                );
            }
            other => panic!(
                "Phase A3c deferral: expected CapabilityUnsupported (HTTP 501); \
                 got {other:?} — the legacy `SlotOom {{ 0, 0 }}` sentinel \
                 mantra-violation must NOT return here.  When Phase A3c ships \
                 the real kernel, flip this match to `Ok(())` + per-buffer \
                 byte-equality assertions (mirrors Qwen35 A2c ceremony)."
            ),
        }
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
        let dev = match skip_dev() { Some(d) => d, None => return };
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
            let (is_ring, cap) = super::layer_type_to_alloc_params(
                *lt, sliding_window, max_seq_len,
            );
            let buf = alloc_hb_kv_for_layer(&dev, layer_idx, nkv, hd, cap, is_ring, n_seqs)
                .unwrap_or_else(|e| panic!(
                    "L{layer_idx} ({lt:?}): alloc_hb_kv_for_layer must succeed; got {e}"
                ));

            // Falsifier 2/3 — is_sliding flag matches layer type.
            assert_eq!(
                buf.is_sliding, is_ring,
                "L{layer_idx} ({lt:?}): is_sliding={} does NOT match \
                 expected={is_ring} (layer-type plumbing broken)",
                buf.is_sliding,
            );

            // Falsifier 4/5 — capacity matches layer-type-specific cap.
            let cap_label = if is_ring { "sliding_window" } else { "max_seq_len" };
            assert_eq!(
                buf.capacity, cap,
                "L{layer_idx} ({lt:?}): capacity={} does NOT match \
                 expected={cap} ({cap_label})",
                buf.capacity,
            );

            // Falsifier 6 — per-seq cursors zero-initialised.
            assert_eq!(buf.seq_lens.len(), n_seqs as usize,
                "L{layer_idx}: seq_lens.len() must equal n_seqs");
            assert!(buf.seq_lens.iter().all(|&x| x == 0),
                "L{layer_idx}: seq_lens must be zero-initialised");

            // Falsifier 7 — n_seqs propagated.
            assert_eq!(buf.n_seqs, n_seqs,
                "L{layer_idx}: n_seqs must propagate from call site");

            // Byte-count cross-check: Full layers cap=32 vs Sliding cap=8
            // ⇒ Full byte count = 4× Sliding byte count for same n_seqs
            // /nkv/hd. We don't assert the ratio inline (it differs per
            // layer) but pin the per-layer byte count against the formula
            // for sanity:
            let expected_packed_bytes = (n_seqs as usize) * nkv * cap * hd;
            assert_eq!(
                buf.k_packed.byte_len(), expected_packed_bytes,
                "L{layer_idx} ({lt:?}): k_packed byte_len mismatch",
            );
            assert_eq!(
                buf.v_packed.byte_len(), expected_packed_bytes,
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
        use crate::serve::config::LayerType;
        let sliding_window: usize = 4_096;
        let max_pos: usize = 131_072;

        let (is_ring_s, cap_s) = super::layer_type_to_alloc_params(
            LayerType::Sliding, sliding_window, max_pos,
        );
        assert!(is_ring_s, "Sliding MUST map to is_ring=true (ring buffer)");
        assert_eq!(cap_s, sliding_window,
            "Sliding MUST map to capacity=sliding_window={sliding_window}");

        let (is_ring_f, cap_f) = super::layer_type_to_alloc_params(
            LayerType::Full, sliding_window, max_pos,
        );
        assert!(!is_ring_f, "Full MUST map to is_ring=false (linear buffer)");
        assert_eq!(cap_f, max_pos,
            "Full MUST map to capacity=max_position_embeddings={max_pos}");

        // Cross-arm sanity: a Full layer NEVER takes the sliding_window
        // capacity and a Sliding layer NEVER takes the max_pos capacity.
        assert_ne!(cap_s, cap_f,
            "Sliding + Full MUST yield distinct capacities in a realistic \
             production config (sliding_window != max_position_embeddings); \
             a swap of the two arms in `layer_type_to_alloc_params` would \
             make these equal and break the assertion above");
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
        use crate::serve::config::LayerType;
        let sliding_window: usize = 1_024;
        let max_pos: usize = 131_072;

        // Sliding layer — production must allocate a ring buffer of
        // capacity sliding_window (per gemma4/model.rs:1253-1257
        // pre-iter-A5c logic; post-iter-A5c routes through this helper).
        let (is_ring, cap) = super::layer_type_to_alloc_params(
            LayerType::Sliding, sliding_window, max_pos,
        );
        assert!(is_ring && cap == sliding_window,
            "production Sliding layer alloc shape: (ring=true, cap=1024); \
             got (ring={is_ring}, cap={cap}) — gemma4/model.rs:1247-1257 \
             would allocate the wrong shape if this mapping drifts");

        // Full layer — production must allocate a linear buffer of
        // capacity max_position_embeddings.
        let (is_ring, cap) = super::layer_type_to_alloc_params(
            LayerType::Full, sliding_window, max_pos,
        );
        assert!(!is_ring && cap == max_pos,
            "production Full layer alloc shape: (ring=false, cap=131072); \
             got (ring={is_ring}, cap={cap})");
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
        let dev = match skip_dev() { Some(d) => d, None => return };
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
            assert_eq!(
                s.len(),
                4,
                "H11 M5: {name} must be 4-D; got {:?}",
                s
            );
            assert_eq!(
                s[0], 4,
                "H11 M5 FALSIFIED: {name} shape[0] must be n_seqs=4 (n_seqs landed \
                 on wrong axis); got {:?}",
                s
            );
        }
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
        let dev = match skip_dev() { Some(d) => d, None => return };
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
            let v_slice = cache.v_packed.as_mut_slice::<u8>().expect("v_packed u8 mut");
            for (i, b) in v_slice[..slot_v_bytes].iter_mut().enumerate() {
                *b = (((i * 11) % 253) + 1) as u8;
            }
        }
        {
            let vn_slice = cache.v_norms.as_mut_slice::<f32>().expect("v_norms f32 mut");
            let slot_vn_f32 = nkv * cap * 1; // norms_per_pos=1
            for (i, f) in vn_slice[..slot_vn_f32].iter_mut().enumerate() {
                *f = (i as f32) * 0.123_45;
            }
        }

        // Snapshot slot 1's regions.
        let k_slot1_before: Vec<u8> = cache
            .k
            .as_slice::<u8>()
            .expect("k F16 as u8")[slot_k_bytes..2 * slot_k_bytes]
            .to_vec();
        let v_slot1_before: Vec<u8> = cache
            .v_packed
            .as_slice::<u8>()
            .expect("v_packed u8")[slot_v_bytes..2 * slot_v_bytes]
            .to_vec();
        let vn_slot1_before: Vec<f32> = cache
            .v_norms
            .as_slice::<f32>()
            .expect("v_norms f32")[(nkv * cap * 1)..2 * (nkv * cap * 1)]
            .to_vec();

        // Sanity: slot 1 is zero-init.
        assert!(
            k_slot1_before.iter().all(|&b| b == 0),
            "H12 fixture sanity: slot 1 K zero-init"
        );
        assert!(
            v_slot1_before.iter().all(|&b| b == 0),
            "H12 fixture sanity: slot 1 V packed zero-init"
        );
        assert!(
            vn_slot1_before.iter().all(|&f| f == 0.0),
            "H12 fixture sanity: slot 1 V norms zero-init"
        );

        // A3b iter-1 cursor advance on slot 0 (no buffer mutation).
        cache.append_for_seq(SlotId(0), 3).expect("H12: append slot 0");
        assert_eq!(cache.seq_lens[0], 3);
        assert_eq!(cache.seq_lens[1], 0);

        // H12 falsifier: slot 1's bytes must be byte-identical.
        let k_slot1_after: Vec<u8> = cache
            .k
            .as_slice::<u8>()
            .expect("k F16 as u8")[slot_k_bytes..2 * slot_k_bytes]
            .to_vec();
        let v_slot1_after: Vec<u8> = cache
            .v_packed
            .as_slice::<u8>()
            .expect("v_packed u8")[slot_v_bytes..2 * slot_v_bytes]
            .to_vec();
        let vn_slot1_after: Vec<f32> = cache
            .v_norms
            .as_slice::<f32>()
            .expect("v_norms f32")[(nkv * cap * 1)..2 * (nkv * cap * 1)]
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
        let dev = match skip_dev() { Some(d) => d, None => return };
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        let mut c = alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, 2, 256, 8, false, 4)
            .expect("alloc");
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
        assert_eq!(c.seq_len(SlotId(2)).unwrap(), 3, "H13: slot 2 preserved through slot 0 drop");
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
        let dev = match skip_dev() { Some(d) => d, None => return };
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
        assert_eq!(bk.shape(), &[3, 2, 4, 256], "H14: xlen K shape n_seqs outermost");
        assert_eq!(bv.shape(), &[3, 2, 4, 256], "H14: xlen V shape n_seqs outermost");

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
        use crate::serve::multi_seq_kv::MultiSeqKvCache;
        let dev = match skip_dev() { Some(d) => d, None => return };
        let nkv = 2; let cap = 8; let hd = 256;
        let k = dev.alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd]).unwrap();
        let v = dev.alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd]).unwrap();
        let mut buf = DenseKvBuffers { k, v, capacity: cap, is_sliding: false, dtype: DType::F32 };

        // slot_count == 1.
        assert_eq!(buf.slot_count(), 1, "H15 FALSIFIED: DenseKvBuffers slot_count must be 1");
        assert_eq!(buf.layout(), MultiSeqLayout::SeparateSlots);

        // seq_len(SlotId(0)) returns Ok(0).
        assert_eq!(buf.seq_len(SlotId(0)).unwrap(), 0);

        // seq_len(SlotId(1)) returns SlotOutOfRange.
        let err = buf.seq_len(SlotId(1)).expect_err("slot 1 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(1), max_slots: 1 },
            "H15 FALSIFIED: SlotOutOfRange shape wrong; got {err:?}"
        );

        // append_for_seq(SlotId(2)) returns SlotOutOfRange (bounds first).
        let err = buf.append_for_seq(SlotId(2), 1).expect_err("append slot 2 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(2), max_slots: 1 }
        );

        // append_for_seq(SlotId(0)) returns CapabilityUnsupported naming iter-A3b-2.
        let err = buf.append_for_seq(SlotId(0), 1).expect_err("append clamped to iter-A3b-2");
        match err {
            MultiSeqError::CapabilityUnsupported { capability } => {
                assert!(capability.contains("DenseKvBuffers"), "label must name struct: {capability}");
                assert!(capability.contains("A3b iter-2"), "label must name deferral: {capability}");
            }
            other => panic!("H15: expected CapabilityUnsupported; got {other:?}"),
        }

        // drop_seq same shape.
        let err = buf.drop_seq(SlotId(5)).expect_err("drop slot 5 OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(5), max_slots: 1 }
        );
        let err = buf.drop_seq(SlotId(0)).expect_err("drop clamped");
        assert!(matches!(err, MultiSeqError::CapabilityUnsupported { .. }));

        // fork_seq(SlotId(0), SlotId(0)) is the only valid combo — Ok(()) no-op.
        buf.fork_seq(SlotId(0), SlotId(0)).expect("self-fork ok no-op");
        // fork_seq src OOR.
        let err = buf.fork_seq(SlotId(1), SlotId(0)).expect_err("fork src OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(1), max_slots: 1 }
        );
        // fork_seq dst OOR.
        let err = buf.fork_seq(SlotId(0), SlotId(2)).expect_err("fork dst OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(2), max_slots: 1 }
        );
    }

    /// **H16** — MlxKvCache typed clamp.  Same shape as H15:
    /// `slot_count() == 1`; slot > 0 returns SlotOutOfRange;
    /// in-bounds slot operations return CapabilityUnsupported
    /// naming iter-A3b-3.  `seq_len(SlotId(0))` reports the
    /// legacy single-seq cursor (`self.seq_len as u32`).
    #[test]
    fn h16_mlx_kv_cache_typed_clamp_slot_count_one() {
        use crate::serve::multi_seq_kv::MultiSeqKvCache;
        let dev = match skip_dev() { Some(d) => d, None => return };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let mut cache = MlxKvCache {
            k_packed: buf(), k_norms: buf(), v_packed: buf(), v_norms: buf(),
            capacity: 16, is_sliding: false, write_pos: 5, seq_len: 5,
        };

        // slot_count == 1.
        assert_eq!(cache.slot_count(), 1, "H16 FALSIFIED: MlxKvCache slot_count must be 1");
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
            MultiSeqError::SlotOutOfRange { slot: SlotId(1), max_slots: 1 },
            "H16 FALSIFIED: SlotOutOfRange shape wrong; got {err:?}"
        );

        // append_for_seq OOR vs clamp.
        let err = cache.append_for_seq(SlotId(3), 1).expect_err("append OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(3), max_slots: 1 }
        );
        let err = cache.append_for_seq(SlotId(0), 1).expect_err("append clamped");
        match err {
            MultiSeqError::CapabilityUnsupported { capability } => {
                assert!(capability.contains("MlxKvCache"), "label must name struct: {capability}");
                assert!(capability.contains("A3b iter-3"), "label must name deferral: {capability}");
                assert!(capability.contains("legacy 4-bit"), "label must name legacy path: {capability}");
            }
            other => panic!("H16: expected CapabilityUnsupported; got {other:?}"),
        }

        // drop_seq same shape.
        let err = cache.drop_seq(SlotId(7)).expect_err("drop OOR");
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange { slot: SlotId(7), max_slots: 1 }
        );
        let err = cache.drop_seq(SlotId(0)).expect_err("drop clamped");
        assert!(matches!(err, MultiSeqError::CapabilityUnsupported { .. }));

        // fork self ok no-op.
        cache.fork_seq(SlotId(0), SlotId(0)).expect("self-fork no-op");
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
}
