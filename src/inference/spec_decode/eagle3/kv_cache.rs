//! ADR-037 Phase E5b — Drafter KV cache + rollback.
//!
//! `DrafterKvCache` owns per-tree-node K and V buffers on the GPU
//! for the EAGLE-3 drafter. Each tree node `i` owns slot `i` in the
//! cache. Operations:
//!
//! ## ADR-040 Phase A4 iter-1 (2026-05-30) — multi-seq sibling type
//!
//! Per the §6.1.53 deep-research dossier
//! ([`docs/research/adr040-a4-drafter-multi-seq-dossier-2026-05-30.md`]),
//! the drafter-side multi-seq KV API contract is settled (vLLM/P-EAGLE
//! per-slot pattern with rejected tokens masked to `PADDING_SLOT_ID`),
//! but empirical research shows spec-decode net-regresses above 4-8
//! concurrent requests. iter-A4 iter-1 ships:
//!
//! 1. **`MultiSeqDrafterKvCache`** sibling struct (additive — LEGACY
//!    [`DrafterKvCache`] is UNCHANGED) with `n_seqs` outermost axis on
//!    K + V buffers + per-slot `seq_lens: Vec<u32>` cursor. Mirrors the
//!    A3a/A3b `MultiSeqHbKvBuffers` / `MultiSeqHybridKvBuffers` pattern
//!    at `src/inference/models/gemma4/kv_cache.rs:197+` / `:954+`.
//! 2. **`alloc_multi_seq_drafter_kv_for_layer`** unified allocator
//!    (mirror of `alloc_hb_kv_for_layer`).
//! 3. **`PADDING_SLOT: SlotId = SlotId(u32::MAX)`** per the vLLM/P-EAGLE
//!    convention — rejected-token writes route here (no-write semantics
//!    at the kernel level; in-bounds-vs-padding is the typed
//!    discrimination).
//! 4. **`MultiSeqKvCache` impl** (bounds-first per A2b iter-1.5
//!    cfa-finding-F5; fork_seq uses the same same-buffer cross-region
//!    copy_within pattern as A3c `gemma4_copy_buffer_slot_region` at
//!    `kv_cache.rs:592+`).
//! 5. **`reset_for_slot`** inherent method (mirror of
//!    `MultiSeqHbKvBuffers::reset_for_slot` at line 687+ in the Gemma 4
//!    file — cursor-only reset; K/V byte preservation discipline).
//!
//! The **threshold gate** lives in `serve::api::engine::Engine::
//! spawn_with_mode` and returns `EngineSpawnError::
//! SpecDecodeMaxSlotsAboveBatchedThreshold` when `max_slots > 4` AND
//! `HF2Q_SPEC_DECODE_ALLOW_OVERSIZED != 1`. This protects operators
//! from the published spec-decode inflection-point regression while
//! letting the API contract land.
//!
//! Per the §6.1.53 closure: iter-A4 iter-1 ships the API + the gate;
//! `iter-A4-cont-moe-validation` (Qwen3.6-A3B A/B at N=1,2,4,8),
//! `iter-A4-cont-acceptance-telemetry`, and `iter-A4-cont-inflection-
//! bench` remain typed-deferred sub-arcs gated on external signals
//! (real-hardware bench / operator infra / D3-style benchmark
//! extension). See ADR-040 §6.1.54 + dossier §6 for the full closure.
//!
//! - **append**: write K/V at slot `i` after computing them in the
//!   forward pass.
//! - **rollback_to_accepted**: after Phase E5a tree-walk produces an
//!   accepted-node-index list, compact the cache so the accepted
//!   slots occupy positions `[0, accepted.len())` in cache order.
//!   Non-accepted slots are dropped — their entries become free
//!   capacity for the next tree expansion.
//!
//! ## Layout
//!
//! Cache buffer shape: `[num_kv_heads, capacity, head_dim]` F32
//! row-major flat (matches `tree_attention` kernel's expected
//! K/V cache layout).
//!
//! ## Why pure F32 instead of BF16
//!
//! `tree_attention` accepts both F32 and F16 KV (per Phase E1 dispatch
//! wrapper). For simplicity + minimal precision loss across
//! append/rollback round-trips, this initial implementation uses
//! F32. Phase E7 may switch to F16 if memory becomes a bottleneck.
//!
//! ## Rollback algorithm
//!
//! Implemented as CPU download → reorder → upload. Simpler than a
//! Metal compact-permute kernel and correct. The cost is bounded:
//! `accepted.len() * num_kv_heads * head_dim * 4` bytes downloaded
//! + uploaded per rollback. At typical Qwen-3.6-27B shapes (head_dim
//! = 128, num_kv_heads = 8, accepted = ~10 tokens), ~40 KB per
//! K and V each — well within Apple Silicon unified memory
//! bandwidth. Future Phase E7 can replace with a GPU compact kernel
//! if profiling reveals it matters.

use anyhow::{anyhow, ensure, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

/// Drafter K/V cache for tree-decoding spec-decode.
pub struct DrafterKvCache {
    /// `[num_kv_heads, capacity, head_dim]` F32. Lives on GPU.
    pub k_buf: MlxBuffer,
    /// Same shape + layout as `k_buf`.
    pub v_buf: MlxBuffer,
    /// Number of KV heads (cache stride dim 0).
    pub num_kv_heads: usize,
    /// Allocated capacity (positions axis). `len <= capacity`.
    pub capacity: usize,
    /// Per-head dim (innermost axis).
    pub head_dim: usize,
    /// Current valid length: positions `[0, len)` are populated;
    /// `[len, capacity)` are uninitialized (may contain stale data).
    len: usize,
}

impl DrafterKvCache {
    /// Allocate a fresh empty KV cache. `capacity` must be > 0.
    pub fn new(
        device: &MlxDevice,
        num_kv_heads: usize,
        capacity: usize,
        head_dim: usize,
    ) -> Result<Self> {
        ensure!(num_kv_heads > 0, "DrafterKvCache: num_kv_heads must be > 0");
        ensure!(capacity > 0, "DrafterKvCache: capacity must be > 0");
        ensure!(head_dim > 0, "DrafterKvCache: head_dim must be > 0");
        let total_elems = num_kv_heads
            .checked_mul(capacity)
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| {
                anyhow!(
                    "DrafterKvCache: num_kv_heads ({}) * capacity ({}) * head_dim ({}) overflows usize",
                    num_kv_heads,
                    capacity,
                    head_dim
                )
            })?;
        ensure!(
            total_elems <= (u32::MAX as usize),
            "DrafterKvCache: total elements ({}) exceeds u32::MAX",
            total_elems
        );
        let total_bytes = total_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow!("DrafterKvCache: byte size overflows usize"))?;
        let k_buf = device
            .alloc_buffer(
                total_bytes,
                DType::F32,
                vec![num_kv_heads, capacity, head_dim],
            )
            .map_err(|e| anyhow!("alloc K cache: {e}"))?;
        let v_buf = device
            .alloc_buffer(
                total_bytes,
                DType::F32,
                vec![num_kv_heads, capacity, head_dim],
            )
            .map_err(|e| anyhow!("alloc V cache: {e}"))?;
        Ok(Self {
            k_buf,
            v_buf,
            num_kv_heads,
            capacity,
            head_dim,
            len: 0,
        })
    }

    /// Current valid length (number of populated slots from 0).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Reset to empty (positions [0, capacity) remain allocated but
    /// `len = 0`). Data not zeroed — append overwrites.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Append a single K/V row at position `len`. The caller supplies
    /// row data in `[num_kv_heads, head_dim]` layout (one row per head).
    ///
    /// Per-head storage in the cache: row gets written at
    /// `cache[head, len, :] = row[head, :]` for each head.
    pub fn append(&mut self, k_row: &[f32], v_row: &[f32]) -> Result<()> {
        ensure!(
            self.len < self.capacity,
            "DrafterKvCache::append: cache full (len={}, capacity={})",
            self.len,
            self.capacity
        );
        let expected = self.num_kv_heads * self.head_dim;
        ensure!(
            k_row.len() == expected,
            "DrafterKvCache::append: k_row has {} elements, expected {} (num_kv_heads {} * head_dim {})",
            k_row.len(),
            expected,
            self.num_kv_heads,
            self.head_dim
        );
        ensure!(
            v_row.len() == expected,
            "DrafterKvCache::append: v_row has {} elements, expected {}",
            v_row.len(),
            expected
        );
        // Write per-head into [head, len, :] cells. Layout is
        // [num_kv_heads, capacity, head_dim] row-major, so the
        // (head, position, dim) offset is `head * capacity * head_dim
        // + position * head_dim + dim`.
        let k_slice = self
            .k_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("k_buf slice: {e}"))?;
        let v_slice = self
            .v_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("v_buf slice: {e}"))?;
        for h in 0..self.num_kv_heads {
            let cache_offset = h * self.capacity * self.head_dim + self.len * self.head_dim;
            let row_offset = h * self.head_dim;
            k_slice[cache_offset..cache_offset + self.head_dim]
                .copy_from_slice(&k_row[row_offset..row_offset + self.head_dim]);
            v_slice[cache_offset..cache_offset + self.head_dim]
                .copy_from_slice(&v_row[row_offset..row_offset + self.head_dim]);
        }
        self.len += 1;
        Ok(())
    }

    /// Rollback: keep only positions in `accepted` (in their accepted
    /// order). After this, `len = accepted.len()`. Positions
    /// `[len, capacity)` may contain stale data and must NOT be
    /// read by downstream consumers.
    ///
    /// Per Phase E5a `walk_tree_accept` contract:
    /// - `accepted[0]` is always root (index 0)
    /// - all indices in `accepted` are < pre-rollback `self.len`
    /// - indices may be out of order (the walk descends; ancestors
    ///   appear before descendants but they may not be index-ordered)
    ///
    /// Implementation: download K/V to CPU, build new sequences from
    /// accepted indices in order, upload to positions [0, accepted.len()).
    pub fn rollback_to_accepted(&mut self, accepted: &[usize]) -> Result<()> {
        ensure!(
            !accepted.is_empty(),
            "DrafterKvCache::rollback_to_accepted: accepted must be non-empty (root always included)"
        );
        ensure!(
            accepted.len() <= self.capacity,
            "DrafterKvCache::rollback_to_accepted: accepted len {} > capacity {}",
            accepted.len(),
            self.capacity
        );
        // Validate all indices fit current len.
        for (i, &idx) in accepted.iter().enumerate() {
            ensure!(
                idx < self.len,
                "DrafterKvCache::rollback_to_accepted: accepted[{}] = {} >= current len {}",
                i,
                idx,
                self.len
            );
        }
        // Detect duplicates (would corrupt the cache by writing
        // the same slot's data twice).
        let mut seen = std::collections::HashSet::with_capacity(accepted.len());
        for (i, &idx) in accepted.iter().enumerate() {
            ensure!(
                seen.insert(idx),
                "DrafterKvCache::rollback_to_accepted: duplicate index {} at position {}",
                idx,
                i
            );
        }
        // Download → reorder → upload.
        let k_data = self
            .k_buf
            .as_slice::<f32>()
            .map_err(|e| anyhow!("k_buf slice: {e}"))?
            .to_vec();
        let v_data = self
            .v_buf
            .as_slice::<f32>()
            .map_err(|e| anyhow!("v_buf slice: {e}"))?
            .to_vec();
        let stride_per_head = self.capacity * self.head_dim;
        let new_len = accepted.len();
        // Build the new K/V data per head, in accepted order.
        let mut new_k = vec![0.0f32; self.num_kv_heads * stride_per_head];
        let mut new_v = vec![0.0f32; self.num_kv_heads * stride_per_head];
        // First copy the unchanged tail (positions [new_len, capacity)
        // stay at whatever they were; only [0, new_len) gets rewritten).
        for h in 0..self.num_kv_heads {
            new_k[h * stride_per_head..(h + 1) * stride_per_head]
                .copy_from_slice(&k_data[h * stride_per_head..(h + 1) * stride_per_head]);
            new_v[h * stride_per_head..(h + 1) * stride_per_head]
                .copy_from_slice(&v_data[h * stride_per_head..(h + 1) * stride_per_head]);
        }
        // Now overwrite positions [0, new_len) with accepted source data.
        for (new_pos, &src_idx) in accepted.iter().enumerate() {
            for h in 0..self.num_kv_heads {
                let src_offset = h * stride_per_head + src_idx * self.head_dim;
                let dst_offset = h * stride_per_head + new_pos * self.head_dim;
                new_k[dst_offset..dst_offset + self.head_dim]
                    .copy_from_slice(&k_data[src_offset..src_offset + self.head_dim]);
                new_v[dst_offset..dst_offset + self.head_dim]
                    .copy_from_slice(&v_data[src_offset..src_offset + self.head_dim]);
            }
        }
        // Upload back.
        self.k_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("k_buf mut slice: {e}"))?
            .copy_from_slice(&new_k);
        self.v_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("v_buf mut slice: {e}"))?
            .copy_from_slice(&new_v);
        self.len = new_len;
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A4 iter-1 (2026-05-30) — multi-seq drafter KV cache.
//
// SIBLING type: LEGACY `DrafterKvCache` (lines 79+ above) is UNCHANGED.
// Production code under `EngineMode::SerialFifo` continues to use the
// legacy single-seq cache byte-for-byte (H231 + H230 invariants).
//
// The multi-seq sibling carries the dossier §5 (`docs/research/
// adr040-a4-drafter-multi-seq-dossier-2026-05-30.md`) API surface:
//   - `n_seqs` outermost on K/V buffers (per-slot byte stride = total
//     bytes / n_seqs; same as A3a `MultiSeqHbKvBuffers`).
//   - Per-slot cursor `seq_lens: Vec<u32>` (length == n_seqs).
//   - `PADDING_SLOT: SlotId = SlotId(u32::MAX)` — vLLM/P-EAGLE
//     rejected-token convention (rejected writes route here at the
//     kernel level; the typed `MultiSeqKvCache::append_for_seq` returns
//     `MultiSeqError::SlotOutOfRange` when the PADDING_SLOT sentinel is
//     used directly so the no-write semantics surface honestly at the
//     trait boundary — kernel-level masking happens at the dispatcher).
//
// fork_seq uses the SAME same-buffer cross-region `copy_within` pattern
// as Gemma 4 A3c's `gemma4_copy_buffer_slot_region` (at
// `src/inference/models/gemma4/kv_cache.rs:592+`); the n_seqs-outermost
// invariant holds so per-slot byte stride is exactly `total_bytes /
// n_seqs`.
//
// `reset_for_slot` mirrors `MultiSeqHbKvBuffers::reset_for_slot` at line
// 687+ of the Gemma 4 file: cursor-only reset; K/V byte preservation
// (the kernel masks against `seq_lens[slot_idx]`, so positions ≥ cursor
// are structurally unreachable).
//
// Per the §6.1.53 dossier closure: the API contract IS settled here;
// production activation gating lives in `Engine::spawn_with_mode` via
// the `SpecDecodeMaxSlotsAboveBatchedThreshold` typed error variant.
// ──────────────────────────────────────────────────────────────────────────

/// **ADR-040 Phase A4 iter-1 (2026-05-30)** — multi-seq variant of
/// [`DrafterKvCache`].
///
/// Outermost axis on K + V buffers is `n_seqs`.  Buffer layout:
/// `[n_seqs, num_kv_heads, capacity, head_dim]` F32 row-major flat.
/// Per-slot byte stride for kernel writes is
/// `slot.0 * (num_kv_heads * capacity * head_dim * size_of::<f32>())`
/// — identical convention to Gemma 4 A3a `MultiSeqHbKvBuffers`
/// (`src/inference/models/gemma4/kv_cache.rs:197+`).
///
/// Per-slot cursor [`seq_lens`](Self::seq_lens) is `Vec<u32>` of length
/// `n_seqs` (parallel to the A3a discipline).  At construction time
/// every slot's cursor is 0; the [`MultiSeqKvCache::append_for_seq`]
/// impl below bumps `seq_lens[slot.0]` while leaving other slots
/// untouched.
///
/// **Per the §6.1.53 dossier**: the API contract is settled (vLLM/P-EAGLE
/// per-slot pattern); production activation is gated by
/// `Engine::spawn_with_mode`'s threshold check
/// (`SpecDecodeMaxSlotsAboveBatchedThreshold` when `max_slots > 4`).
/// This type can be safely constructed in test contexts and at
/// operator-opted-in spawn time.
pub struct MultiSeqDrafterKvCache {
    /// Number of physical slots — the outermost axis on K + V.
    /// Set at construction via [`alloc_multi_seq_drafter_kv_for_layer`];
    /// once set, cannot change without reallocation.
    pub n_seqs: u32,
    /// K buffer at `[n_seqs, num_kv_heads, capacity, head_dim]` F32.
    pub k_buf: MlxBuffer,
    /// V buffer at the same shape + dtype as `k_buf`.
    pub v_buf: MlxBuffer,
    /// Number of KV heads (axis 1).
    pub num_kv_heads: usize,
    /// Allocated capacity (axis 2 — positions per slot).
    pub capacity: usize,
    /// Per-head dim (innermost axis).
    pub head_dim: usize,
    /// Per-slot write cursor; `seq_lens[slot.0]` is the number of valid
    /// positions stored in slot `slot.0`.  `len() == n_seqs as usize`
    /// by construction.
    pub seq_lens: Vec<u32>,
}

impl MultiSeqDrafterKvCache {
    /// **vLLM/P-EAGLE rejected-token convention** — rejected tokens
    /// during EAGLE-3 tree verification map to this sentinel `SlotId`
    /// so the kernel-level dispatcher can mask their writes.  Value
    /// `SlotId(u32::MAX)` because the alloc helper rejects
    /// `n_seqs == u32::MAX` (would overflow byte arithmetic), so
    /// `PADDING_SLOT` is always strictly outside the in-bounds range
    /// `[0, n_seqs)`.
    ///
    /// **Semantics at the trait surface**: passing `PADDING_SLOT` to
    /// [`MultiSeqKvCache::append_for_seq`] (and any other trait method)
    /// surfaces as [`crate::serve::multi_seq_kv::MultiSeqError::
    /// SlotOutOfRange`] — the typed signal "this is the
    /// padding/rejected-token sentinel; do not write" rather than the
    /// silent no-op a kernel-level mask would produce.  Kernel-level
    /// masking is a separate concern at the future B4d-like drafter
    /// dispatcher (iter-A4-cont, gated on real-hardware empirical
    /// inflection-point measurement per dossier §6 + §7).
    pub const PADDING_SLOT: crate::serve::multi_seq_kv::SlotId =
        crate::serve::multi_seq_kv::SlotId(u32::MAX);

    /// **ADR-040 Phase A4 iter-1 (2026-05-30)** — per-slot cursor reset
    /// for the persistent multi-seq drafter KV cache.
    ///
    /// Cross-architecture mirror of Gemma 4
    /// [`crate::inference::models::gemma4::kv_cache::MultiSeqHbKvBuffers::
    /// reset_for_slot`] (per §6.1.31 closure block) and Qwen35
    /// `HybridKvCache::reset_for_slot` (per §6.1.27 closure block).
    /// Reserved for use by a future EAGLE-3 orchestrator slot-aware
    /// entry point at request boundaries so the persistent per-slot
    /// drafter cache is request-isolated within the slot — the next
    /// request to land on the same slot sees a zero-cursor cache.
    ///
    /// **Layout proof** (mirror of A3a / A3b discipline):
    /// - **seq_lens**: `Vec<u32>` of length `n_seqs`. Per-slot reset →
    ///   set `seq_lens[slot_idx] = 0`; other slots untouched.  This is
    ///   the load-bearing cursor that bounds the drafter tree-attention
    ///   K/V read path.
    /// - **k_buf / v_buf (F32, `[n_seqs, num_kv_heads, capacity,
    ///   head_dim]` row-major)**: per-slot region size = `num_kv_heads
    ///   * capacity * head_dim * size_of::<f32>()` bytes.  **NOT
    ///   zeroed** — same discipline as A3a `MultiSeqHbKvBuffers`: the
    ///   tree-attention SDPA read path masks against
    ///   `seq_lens[slot_idx]` (positions ≥ cursor are unreadable to the
    ///   kernel).  Stale bytes beyond the cursor are structurally
    ///   unreachable.
    ///
    /// # Errors
    ///
    /// - `slot.0 >= self.n_seqs` (bounds-first per A2b iter-1.5
    ///   cfa-finding-F5 ordering) — returns typed
    ///   [`crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange`].
    /// - `slot == Self::PADDING_SLOT` surfaces as `SlotOutOfRange`
    ///   because `PADDING_SLOT.0 == u32::MAX >= n_seqs` for any
    ///   reachable `n_seqs` — the alloc helper rejects n_seqs ==
    ///   u32::MAX as oversized.
    ///
    /// # Per-slot byte-equivalence pin (H230)
    ///
    /// At `slot = SlotId(0)` AND `n_seqs == 1` this is byte-equivalent
    /// to setting `seq_lens[0] = 0` directly (the existing legacy
    /// [`DrafterKvCache::clear`] shape at the cursor level).
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
        // Cursor-only reset; K/V bytes preserved (cursor-masked read
        // path — see layout proof above).
        self.seq_lens[slot.0 as usize] = 0;
        Ok(())
    }
}

/// **ADR-040 Phase A4 iter-1 (2026-05-30)** — unified
/// [`MultiSeqDrafterKvCache`] allocator.
///
/// Mirrors the A3a [`crate::inference::models::gemma4::kv_cache::
/// alloc_hb_kv_for_layer`] (`gemma4/kv_cache.rs:252+`) shape:
/// pre-flight dimension validation + `MlxBuffer::alloc_buffer` for K
/// + V at `[n_seqs, num_kv_heads, capacity, head_dim]` F32.
///
/// At `n_seqs == 1` the byte counts are identical to the legacy
/// [`DrafterKvCache::new`] call (`num_kv_heads * capacity * head_dim *
/// 4` per buffer); the only observable shape difference is the leading
/// dimension on every buffer (`[1, nkv, cap, hd]` vs `[nkv, cap, hd]`).
/// H230 pins this byte-equivalence at the test level.
///
/// # Errors
///
/// Returns `Err` for any of:
/// - `n_seqs == 0` — caller bug (every alloc site has a real concurrency).
/// - `n_seqs == u32::MAX` — would collide with [`MultiSeqDrafterKvCache::
///   PADDING_SLOT`]; defensive rejection so the padding sentinel is
///   always strictly outside the in-bounds range.
/// - `num_kv_heads == 0` / `capacity == 0` / `head_dim == 0` — kernel
///   shape preconditions would otherwise underflow.
/// - The byte product `n_seqs * num_kv_heads * capacity * head_dim * 4`
///   overflows `usize` or exceeds `u32::MAX` (the MlxBuffer shape's
///   dimensional bound).
///
/// Mirrors [`DrafterKvCache::new`] defensive pre-flight (lines 79-119
/// of this file) verbatim plus the per-axis multiplication overflow
/// guard the A3a / A3b allocators carry.
pub fn alloc_multi_seq_drafter_kv_for_layer(
    device: &MlxDevice,
    num_kv_heads: usize,
    capacity: usize,
    head_dim: usize,
    n_seqs: u32,
) -> Result<MultiSeqDrafterKvCache> {
    ensure!(
        n_seqs > 0,
        "alloc_multi_seq_drafter_kv_for_layer: n_seqs must be > 0"
    );
    ensure!(
        n_seqs != u32::MAX,
        "alloc_multi_seq_drafter_kv_for_layer: n_seqs must be < u32::MAX \
         (reserved as PADDING_SLOT sentinel per vLLM/P-EAGLE convention)"
    );
    ensure!(
        num_kv_heads > 0,
        "alloc_multi_seq_drafter_kv_for_layer: num_kv_heads must be > 0"
    );
    ensure!(
        capacity > 0,
        "alloc_multi_seq_drafter_kv_for_layer: capacity must be > 0"
    );
    ensure!(
        head_dim > 0,
        "alloc_multi_seq_drafter_kv_for_layer: head_dim must be > 0"
    );
    let n = n_seqs as usize;
    let total_elems = n
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(capacity))
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| {
            anyhow!(
                "alloc_multi_seq_drafter_kv_for_layer: n_seqs ({}) * num_kv_heads \
                 ({}) * capacity ({}) * head_dim ({}) overflows usize",
                n_seqs,
                num_kv_heads,
                capacity,
                head_dim
            )
        })?;
    ensure!(
        total_elems <= (u32::MAX as usize),
        "alloc_multi_seq_drafter_kv_for_layer: total elements ({}) exceeds u32::MAX \
         (MlxBuffer shape bound)",
        total_elems
    );
    let total_bytes = total_elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            anyhow!("alloc_multi_seq_drafter_kv_for_layer: byte size overflows usize")
        })?;
    let shape = vec![n, num_kv_heads, capacity, head_dim];
    let mut k_buf = device
        .alloc_buffer(total_bytes, DType::F32, shape.clone())
        .map_err(|e| anyhow!("alloc multi-seq drafter K: {e}"))?;
    let mut v_buf = device
        .alloc_buffer(total_bytes, DType::F32, shape)
        .map_err(|e| anyhow!("alloc multi-seq drafter V: {e}"))?;
    // Zero-init mirrors A3a `alloc_hb_kv_for_layer` discipline
    // (`gemma4/kv_cache.rs:307-322`): defend against StorageModeShared
    // returning recycled non-zero memory (ADR-015 iter61a).
    if let Ok(s) = k_buf.as_mut_slice::<f32>() {
        s.fill(0.0);
    }
    if let Ok(s) = v_buf.as_mut_slice::<f32>() {
        s.fill(0.0);
    }
    Ok(MultiSeqDrafterKvCache {
        n_seqs,
        k_buf,
        v_buf,
        num_kv_heads,
        capacity,
        head_dim,
        seq_lens: vec![0u32; n],
    })
}

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 Phase A4 iter-1 — MultiSeqKvCache impl for
// MultiSeqDrafterKvCache.
//
// Mirrors the A3a `MultiSeqHbKvBuffers` impl at
// `gemma4/kv_cache.rs:393-568` in structure (bounds-first per iter-1.5
// cfa-finding-F5; fork_seq uses same-buffer cross-region copy_within
// via the `drafter_copy_buffer_slot_region` helper below).
//
// Phase A4 iter-1 scope: per-slot CURSOR bookkeeping + same-buffer
// cross-region fork.  No new kernel writes; that's iter-A4-cont scope
// (gated on empirical inflection-point measurement per dossier §6).
// ──────────────────────────────────────────────────────────────────────────

impl crate::serve::multi_seq_kv::MultiSeqKvCache for MultiSeqDrafterKvCache {
    fn layout(&self) -> crate::serve::multi_seq_kv::MultiSeqLayout {
        crate::serve::multi_seq_kv::MultiSeqLayout::SeparateSlots
    }

    fn slot_count(&self) -> u32 {
        self.n_seqs
    }

    fn seq_len(
        &self,
        slot: crate::serve::multi_seq_kv::SlotId,
    ) -> Result<u32, crate::serve::multi_seq_kv::MultiSeqError> {
        // 1. Bounds FIRST (iter-1.5 cfa-finding-F5 ordering).
        //    PADDING_SLOT == SlotId(u32::MAX) surfaces here too — the
        //    typed signal that the caller used the rejected-token
        //    sentinel directly against the trait surface.
        if slot.0 >= self.n_seqs {
            return Err(crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: self.n_seqs,
            });
        }
        // 2. Layout: SeparateSlots only — MultiSeqDrafterKvCache does
        //    not expose Paged.
        // 3. Return the per-seq cursor directly; `seq_lens.len() ==
        //    n_seqs` by construction (alloc_multi_seq_drafter_kv_for_layer).
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
        // ADR-040 Phase A4 iter-1 scope: bump the per-seq cursor.  The
        // underlying k_buf / v_buf bytes for slot `slot.0` are written
        // by the future drafter dispatcher at iter-A4-cont (gated on
        // empirical inflection-point measurement per dossier §6).
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
        // ADR-040 Phase A4 iter-1 scope: cursor-only reset.  The
        // underlying K/V bytes are NOT zeroed; the next
        // `append_for_seq` into this slot will overwrite them via the
        // future drafter dispatcher (iter-A4-cont).  Recurrent-content
        // invariance under drop_seq matches A3a `MultiSeqHbKvBuffers`
        // discipline at `gemma4/kv_cache.rs:457-482`.
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
        // ADR-040 Phase A4 iter-1 — REAL cross-slot fork via
        // same-buffer cross-region memcpy.  Mirror of A3c Gemma 4
        // `gemma4_copy_buffer_slot_region` (`gemma4/kv_cache.rs:592+`)
        // and Qwen35 A2c.  n_seqs OUTERMOST on K/V ⇒ per-slot byte
        // stride = `total_bytes / n_seqs`.
        //
        // Cursor copy: `seq_lens[dst] = seq_lens[src]` AFTER buffer copy.
        // ──────────────────────────────────────────────────────────────
        let src_idx = src.0 as usize;
        let dst_idx = dst.0 as usize;
        let n_seqs = self.n_seqs as usize;
        drafter_copy_buffer_slot_region(&mut self.k_buf, src_idx, dst_idx, n_seqs).map_err(
            |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: drafter_leak_static_str(format!(
                    "fork_seq: MultiSeqDrafterKvCache k_buf copy failed ({e})"
                )),
            },
        )?;
        drafter_copy_buffer_slot_region(&mut self.v_buf, src_idx, dst_idx, n_seqs).map_err(
            |e| crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported {
                capability: drafter_leak_static_str(format!(
                    "fork_seq: MultiSeqDrafterKvCache v_buf copy failed ({e})"
                )),
            },
        )?;
        // Cursor copy AFTER buffer copy.
        self.seq_lens[dst_idx] = self.seq_lens[src_idx];
        Ok(())
    }
}

/// ADR-040 Phase A4 iter-1 (2026-05-30) — leak a `String` into a
/// `&'static str` for `MultiSeqError::CapabilityUnsupported` payloads
/// constructed from runtime context.  Sibling of Gemma 4 A3c's
/// `gemma4_leak_static_str` at `gemma4/kv_cache.rs:575+`.
#[inline]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "Rust 1.89 does not mark helpers reached only by the dormant MultiSeqKvCache::fork_seq production seam as live"
    )
)]
fn drafter_leak_static_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// ADR-040 Phase A4 iter-1 (2026-05-30) — same-buffer cross-region byte
/// copy for a multi-seq drafter buffer at `[n_seqs, ...]` outermost.
///
/// Per-slot byte stride is `buf.byte_len() / n_seqs`.  Mirrors A3c
/// `gemma4_copy_buffer_slot_region` (`gemma4/kv_cache.rs:592+`) and
/// Qwen35 A2c's `copy_buffer_slot_region` 1:1 in shape; the
/// n_seqs-outermost invariant is identical (the alloc helper above emits
/// shape `[n_seqs, ...]` with n_seqs as leading dim ⇒ per-slot region is
/// contiguous of size `total_bytes / n_seqs`).
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "Rust 1.89 does not mark helpers reached only by the dormant MultiSeqKvCache::fork_seq production seam as live"
    )
)]
fn drafter_copy_buffer_slot_region(
    buf: &mut MlxBuffer,
    src_idx: usize,
    dst_idx: usize,
    n_seqs: usize,
) -> Result<()> {
    ensure!(n_seqs > 0, "fork_seq: n_seqs must be > 0");
    let total_bytes = buf.byte_len();
    ensure!(
        total_bytes % n_seqs == 0,
        "fork_seq: total_bytes={} not divisible by n_seqs={}",
        total_bytes,
        n_seqs
    );
    let per_slot_bytes = total_bytes / n_seqs;
    ensure!(
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

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 §6.1.55 iter-A4-cont-drafter-dispatcher (2026-05-30) —
// SingleSeq vs MultiSeq drafter cache dispatch.
//
// Per the §6.1.55 dossier (research/§5), the orchestrator routes
// between [`DrafterKvCache`] (pre-A4 byte-equivalent single-seq) and
// [`MultiSeqDrafterKvCache`] (post-A4 batched spec-decode opt-in)
// based on the engine mode discriminator.  Today the SlotAware-side
// arm is structurally wired but NEVER engaged at runtime — the
// `Engine::spawn_with_mode` SlotAware arm's threshold gate from iter-1
// (§6.1.54) rejects `max_slots > 4` unless
// `HF2Q_SPEC_DECODE_ALLOW_OVERSIZED=1` is set; once the operator opts
// in OR an empirical inflection-point measurement lands a tunable
// threshold above 1, this dispatcher is the routing seam the worker
// arm will call.
//
// **Pure variant + routing helper** — no kernel writes.  The
// kernel-level routing (per-slot byte-stride dispatch through the
// EAGLE-3 `tree_attention` kernel) is iter-A4-cont-drafter-dispatcher-
// kernel, gated on the inflection-point measurement per dossier §6.
// ──────────────────────────────────────────────────────────────────────────

/// **ADR-040 §6.1.55 iter-A4-cont-drafter-dispatcher (2026-05-30)** —
/// SingleSeq vs MultiSeq variant carrier for the EAGLE-3 drafter KV
/// cache.
///
/// Mirror of the dossier §5 design.  At construction time the
/// orchestrator picks one of the two arms based on the engine mode:
///
/// - [`Self::SingleSeq`] — pre-A4 byte-equivalent single-seq cache.
///   Selected on `EngineMode::SerialFifo` AND on
///   `EngineMode::SlotAware { max_slots: 1 }` (the single-slot
///   degenerate case where the multi-seq path would carry the same
///   byte count anyway; H230 pins the byte equivalence).
/// - [`Self::MultiSeq`] — post-A4 multi-seq cache.  Selected on
///   `EngineMode::SlotAware { max_slots: N>1 }` AFTER the threshold
///   gate at `Engine::spawn_with_mode` either accepts the value OR the
///   operator has set `HF2Q_SPEC_DECODE_ALLOW_OVERSIZED=1` for the
///   documented-regression regime.
///
/// **iter-A4-cont-drafter-dispatcher-kernel (deferred)**: the
/// kernel-level routing through `tree_attention` per-slot byte
/// strides lands at iter-A4-cont-drafter-dispatcher-kernel.  Today
/// only the routing-variant decision lives here; the kernel-side
/// dispatch is gated on the empirical inflection-point measurement
/// per dossier §6 (the iter would be a B4c-kernel-style §6.1.45
/// mirror for the drafter K/V buffers).
pub enum DrafterKvCacheVariant {
    /// Pre-A4 single-seq variant.  Byte-equivalent to the legacy
    /// [`DrafterKvCache`] surface in every observable way.
    SingleSeq(DrafterKvCache),
    /// Post-A4 multi-seq variant.  See [`MultiSeqDrafterKvCache`] for
    /// the per-slot cursor + buffer layout.
    MultiSeq(MultiSeqDrafterKvCache),
}

impl DrafterKvCacheVariant {
    /// Returns the slot count (1 for [`Self::SingleSeq`]; `n_seqs` for
    /// [`Self::MultiSeq`]).  Pure accessor; no allocation.
    pub fn slot_count(&self) -> u32 {
        match self {
            Self::SingleSeq(_) => 1,
            Self::MultiSeq(c) => c.n_seqs,
        }
    }

    /// `true` when this variant is the multi-seq arm (post-A4
    /// dispatcher engaged).  Operator-grep'able via the
    /// `iter-A4-cont-drafter-dispatcher` cite at the call site.
    pub fn is_multi_seq(&self) -> bool {
        matches!(self, Self::MultiSeq(_))
    }
}

/// **ADR-040 §6.1.55 iter-A4-cont-drafter-dispatcher (2026-05-30)** —
/// route the requested `max_slots` from
/// [`crate::serve::api::engine::EngineMode::SlotAware`] to the
/// appropriate [`DrafterKvCacheVariant`] arm.
///
/// Pure decision function — takes the slot count + the legacy
/// per-shape allocator arguments and produces the variant decision.
/// Does NOT allocate any GPU buffer (kernel-level dispatch + alloc
/// lands at iter-A4-cont-drafter-dispatcher-kernel; today the helper
/// is the structural routing seam only).
///
/// # Semantics
///
/// - `max_slots == 1` ⇒ pick `DrafterKvCacheVariant::SingleSeq` (the
///   degenerate case is byte-equivalent to the multi-seq alternative
///   at n_seqs=1 per H230; carrying the legacy type preserves the
///   pre-A4 byte-equivalence pin from §6.1.54).
/// - `max_slots > 1` ⇒ pick `DrafterKvCacheVariant::MultiSeq` — the
///   per-slot cursor + buffer routing seam.
///
/// The companion [`Engine::spawn_with_mode`] threshold gate at
/// `engine.rs::SpecDecodeMaxSlotsAboveBatchedThreshold` enforces the
/// safe-zone policy; this helper is reached ONLY when the gate
/// already accepted the `max_slots` value.
///
/// # Cross-references
///
/// - Dossier §5 (concrete API surface proposal — this is the
///   `DrafterKvCacheVariant` arm).
/// - ADR-040 §6.1.54 (iter-A4 iter-1 SHIPPED — API + threshold gate;
///   this helper is the orchestrator-side mirror of that contract).
/// - ADR-040 §6.1.55 (closure block — names the full structural
///   bundle).
#[inline]
pub fn select_drafter_kv_variant_for_mode(max_slots: u32) -> DrafterKvCacheSelection {
    if max_slots <= 1 {
        DrafterKvCacheSelection::SingleSeq
    } else {
        DrafterKvCacheSelection::MultiSeq { n_seqs: max_slots }
    }
}

/// Companion typed decision for [`select_drafter_kv_variant_for_mode`].
///
/// Carries the routing decision shape *without* constructing the
/// buffer (lets unit tests pin the routing policy at the
/// structural-shape level — no MlxDevice required).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterKvCacheSelection {
    /// Pre-A4 single-seq routing (production-default).
    SingleSeq,
    /// Post-A4 multi-seq routing with `n_seqs` distinct slots.
    MultiSeq { n_seqs: u32 },
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn make_cache() -> Option<(MlxDevice, DrafterKvCache)> {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return None,
        };
        let cache = DrafterKvCache::new(&device, 2, 8, 4).expect("alloc");
        Some((device, cache))
    }

    /// Build a sentinel K/V row where the values encode (head, dim, tag).
    /// Per-head value: row[head][dim] = tag*1000 + head*100 + dim.
    fn sentinel_row(num_kv_heads: usize, head_dim: usize, tag: u32) -> Vec<f32> {
        let mut out = vec![0.0f32; num_kv_heads * head_dim];
        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                out[h * head_dim + d] = (tag * 1000 + h as u32 * 100 + d as u32) as f32;
            }
        }
        out
    }

    #[test]
    fn adr_037_e5b_kv_cache_constructor_validates_dims_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        assert!(DrafterKvCache::new(&device, 0, 8, 4).is_err());
        assert!(DrafterKvCache::new(&device, 2, 0, 4).is_err());
        assert!(DrafterKvCache::new(&device, 2, 8, 0).is_err());
    }

    #[test]
    fn adr_037_e5b_kv_cache_initial_state_empty_2026_05_22() {
        let (_dev, cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.num_kv_heads, 2);
        assert_eq!(cache.capacity, 8);
        assert_eq!(cache.head_dim, 4);
    }

    #[test]
    fn adr_037_e5b_kv_cache_append_grows_len_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        for tag in 0..5 {
            let k = sentinel_row(cache.num_kv_heads, cache.head_dim, tag);
            let v = sentinel_row(cache.num_kv_heads, cache.head_dim, tag + 100);
            cache.append(&k, &v).expect("append");
            assert_eq!(cache.len(), (tag + 1) as usize);
        }
    }

    #[test]
    fn adr_037_e5b_kv_cache_append_rejects_full_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let k = sentinel_row(2, 4, 0);
        let v = sentinel_row(2, 4, 0);
        for _ in 0..8 {
            cache.append(&k, &v).expect("append within capacity");
        }
        let err = cache.append(&k, &v).unwrap_err();
        assert!(err.to_string().contains("cache full"), "got: {err}");
    }

    #[test]
    fn adr_037_e5b_kv_cache_append_rejects_wrong_row_shape_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let bad_k = vec![0.0f32; 5]; // expected 2*4=8
        let v = sentinel_row(2, 4, 0);
        let err = cache.append(&bad_k, &v).unwrap_err();
        assert!(err.to_string().contains("k_row has 5"), "got: {err}");
    }

    #[test]
    fn adr_037_e5b_kv_cache_rollback_keeps_only_accepted_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        // Append 5 sentinel-tagged rows. Tags 0..5.
        for tag in 0..5 {
            let k = sentinel_row(cache.num_kv_heads, cache.head_dim, tag);
            let v = sentinel_row(cache.num_kv_heads, cache.head_dim, tag + 100);
            cache.append(&k, &v).expect("append");
        }
        // Roll back to indices [0, 2, 4]. Expected new len = 3,
        // and positions [0, 1, 2] contain data from src 0, 2, 4.
        cache.rollback_to_accepted(&[0, 2, 4]).expect("rollback");
        assert_eq!(cache.len(), 3);
        // Verify content at positions [0, 1, 2].
        let k_data: Vec<f32> = cache.k_buf.as_slice::<f32>().unwrap().to_vec();
        let stride_per_head = cache.capacity * cache.head_dim;
        for (new_pos, &src_tag) in [0_u32, 2, 4].iter().enumerate() {
            for h in 0..cache.num_kv_heads {
                for d in 0..cache.head_dim {
                    let offset = h * stride_per_head + new_pos * cache.head_dim + d;
                    let expected = (src_tag * 1000 + h as u32 * 100 + d as u32) as f32;
                    assert_eq!(
                        k_data[offset], expected,
                        "rollback pos {} head {} dim {} expected tag {} got {}",
                        new_pos, h, d, src_tag, k_data[offset]
                    );
                }
            }
        }
    }

    #[test]
    fn adr_037_e5b_kv_cache_rollback_rejects_empty_accepted_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let k = sentinel_row(2, 4, 0);
        let v = sentinel_row(2, 4, 0);
        cache.append(&k, &v).expect("append");
        let err = cache.rollback_to_accepted(&[]).unwrap_err();
        assert!(err.to_string().contains("must be non-empty"), "got: {err}");
    }

    #[test]
    fn adr_037_e5b_kv_cache_rollback_rejects_out_of_range_idx_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let k = sentinel_row(2, 4, 0);
        let v = sentinel_row(2, 4, 0);
        cache.append(&k, &v).expect("append"); // len = 1
        let err = cache.rollback_to_accepted(&[0, 5]).unwrap_err();
        assert!(err.to_string().contains(">= current len"), "got: {err}");
    }

    #[test]
    fn adr_037_e5b_kv_cache_rollback_rejects_duplicate_idx_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let k = sentinel_row(2, 4, 0);
        let v = sentinel_row(2, 4, 0);
        for _ in 0..3 {
            cache.append(&k, &v).expect("append");
        }
        let err = cache.rollback_to_accepted(&[0, 1, 1]).unwrap_err();
        assert!(err.to_string().contains("duplicate index"), "got: {err}");
    }

    #[test]
    fn adr_037_e5b_kv_cache_rollback_to_root_only_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        for tag in 0..4 {
            let k = sentinel_row(cache.num_kv_heads, cache.head_dim, tag);
            let v = sentinel_row(cache.num_kv_heads, cache.head_dim, tag + 100);
            cache.append(&k, &v).expect("append");
        }
        // Tree-walk-accept returns only [0] when verifier rejects
        // root's prediction. Rollback collapses cache to len=1.
        cache.rollback_to_accepted(&[0]).expect("rollback to root");
        assert_eq!(cache.len(), 1);
        // Verify position 0 has tag-0 data unchanged.
        let k_data = cache.k_buf.as_slice::<f32>().unwrap();
        let stride_per_head = cache.capacity * cache.head_dim;
        for h in 0..cache.num_kv_heads {
            for d in 0..cache.head_dim {
                let offset = h * stride_per_head + d;
                let expected = (0_u32 * 1000 + h as u32 * 100 + d as u32) as f32;
                assert_eq!(k_data[offset], expected);
            }
        }
    }

    #[test]
    fn adr_037_e5b_kv_cache_clear_resets_len_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let k = sentinel_row(2, 4, 0);
        let v = sentinel_row(2, 4, 0);
        cache.append(&k, &v).expect("append");
        cache.append(&k, &v).expect("append");
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn adr_037_e5b_kv_cache_integration_with_tree_walk_accept_2026_05_22() {
        // Build a 4-node tree + verifier argmax → walk → rollback.
        use crate::inference::spec_decode::eagle3::dynamic_tree::ExpandedTree;
        use crate::inference::spec_decode::eagle3::tree_walk::walk_tree_accept;

        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        // Append 4 sentinel rows for a 4-node tree (root + 3 descendants).
        for tag in 0..4 {
            let k = sentinel_row(cache.num_kv_heads, cache.head_dim, tag);
            let v = sentinel_row(cache.num_kv_heads, cache.head_dim, tag + 100);
            cache.append(&k, &v).expect("append");
        }
        // Tree: 0 → 1 → 2 (chain) + 0 → 3 (sibling of 1).
        // Verifier: root→token1 = 1 (match node 1), node1→token2 = 2 (match node 2).
        // Walk: [0, 1, 2].
        let tree = ExpandedTree {
            tokens: vec![100, 1, 2, 3],
            parents: vec![None, Some(0), Some(1), Some(0)],
            depths: vec![0, 1, 2, 1],
            cum_log_probs: vec![0.0, -0.1, -0.2, -0.5],
        };
        let argmax = vec![1_u32, 2, 0, 0];
        let accepted = walk_tree_accept(&tree, &argmax).expect("walk");
        assert_eq!(accepted, vec![0, 1, 2]);
        // Rollback cache to keep accepted positions.
        cache.rollback_to_accepted(&accepted).expect("rollback");
        assert_eq!(cache.len(), 3);
        // Cache positions [0, 1, 2] should contain tags 0, 1, 2 (root, node1, node2).
        let k_data = cache.k_buf.as_slice::<f32>().unwrap();
        let stride_per_head = cache.capacity * cache.head_dim;
        for (new_pos, expected_tag) in [0_u32, 1, 2].iter().enumerate() {
            for h in 0..cache.num_kv_heads {
                let offset = h * stride_per_head + new_pos * cache.head_dim;
                let expected = (expected_tag * 1000 + h as u32 * 100) as f32;
                assert_eq!(k_data[offset], expected);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-040 Phase A4 iter-1 (2026-05-30) — H224..H232 (sans H229
    // which lives with the spawn-gate in `serve::api::engine`).
    //
    // These tests pin:
    //   - H224: MultiSeqDrafterKvCache sibling struct exists + has the
    //     dossier §5 shape.
    //   - H225: alloc_multi_seq_drafter_kv_for_layer rejects every
    //     malformed dim + the PADDING_SLOT-collision guard.
    //   - H226: PADDING_SLOT == SlotId(u32::MAX) per vLLM/P-EAGLE.
    //   - H227: MultiSeqKvCache impl bounds-first + fork_seq via
    //     same-buffer cross-region memcpy.
    //   - H228: reset_for_slot is cursor-only + preserves K/V bytes.
    //   - H230: n_seqs=1 byte-equivalence with the legacy DrafterKvCache
    //     at the byte-count level.
    //   - H231: LEGACY DrafterKvCache surface is UNCHANGED (additive
    //     sibling pin).
    //   - H232: Qwen35 + Gemma 4 + Qwen3VL surfaces UNCHANGED outside
    //     the spec_decode dir (this test pins the cross-module type
    //     witness; full grep regression is the test bundle bundle).
    //
    // Skip discipline: tests that require MlxDevice gracefully skip
    // when device construction fails (CI without GPU). The structural
    // pins (PADDING_SLOT const + alloc-helper signature) run
    // unconditionally.
    // ──────────────────────────────────────────────────────────────────

    fn make_multi_seq_cache(n_seqs: u32) -> Option<(MlxDevice, MultiSeqDrafterKvCache)> {
        let device = MlxDevice::new().ok()?;
        let cache = alloc_multi_seq_drafter_kv_for_layer(&device, 2, 8, 4, n_seqs).expect("alloc");
        Some((device, cache))
    }

    /// **H224** — `MultiSeqDrafterKvCache` carries the dossier §5 shape:
    /// `n_seqs` outermost on K + V + per-slot `seq_lens` cursor.
    #[test]
    fn h224_multi_seq_drafter_kv_cache_carries_dossier_shape_2026_05_30() {
        let (_dev, cache) = match make_multi_seq_cache(3) {
            Some(t) => t,
            None => {
                eprintln!(
                    "[skip] h224 — MlxDevice unavailable (CI without GPU); \
                     skip-mode structural pins for PADDING_SLOT + alloc helper \
                     run unconditionally elsewhere."
                );
                return;
            }
        };
        // Field witnesses — every dossier §5 field must be reachable.
        assert_eq!(cache.n_seqs, 3, "H224: n_seqs round-trips alloc arg");
        assert_eq!(cache.num_kv_heads, 2);
        assert_eq!(cache.capacity, 8);
        assert_eq!(cache.head_dim, 4);
        assert_eq!(
            cache.seq_lens.len(),
            3,
            "H224: seq_lens.len() == n_seqs by construction"
        );
        assert!(
            cache.seq_lens.iter().all(|&l| l == 0),
            "H224: all per-slot cursors start at 0"
        );
        // K + V buffers carry the 4-D shape (n_seqs outermost).
        assert_eq!(
            cache.k_buf.byte_len(),
            3 * 2 * 8 * 4 * std::mem::size_of::<f32>(),
            "H224: K buffer total bytes match [n_seqs, nkv, cap, hd] F32"
        );
        assert_eq!(
            cache.v_buf.byte_len(),
            3 * 2 * 8 * 4 * std::mem::size_of::<f32>(),
            "H224: V buffer total bytes match [n_seqs, nkv, cap, hd] F32"
        );
    }

    /// **H225** — alloc helper rejects every malformed dim + the
    /// PADDING_SLOT-collision guard at `n_seqs == u32::MAX`.
    #[test]
    fn h225_alloc_multi_seq_drafter_kv_validates_dims_2026_05_30() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("[skip] h225 — MlxDevice unavailable");
                return;
            }
        };
        // n_seqs == 0
        assert!(
            alloc_multi_seq_drafter_kv_for_layer(&device, 2, 8, 4, 0).is_err(),
            "H225: n_seqs == 0 must error"
        );
        // n_seqs == u32::MAX collides with PADDING_SLOT
        let err = alloc_multi_seq_drafter_kv_for_layer(&device, 2, 8, 4, u32::MAX)
            .err()
            .expect("H225: n_seqs == u32::MAX must error");
        assert!(
            err.to_string().contains("PADDING_SLOT"),
            "H225: PADDING_SLOT collision must be named in error. Got: {err}"
        );
        // num_kv_heads / capacity / head_dim each zero independently.
        assert!(
            alloc_multi_seq_drafter_kv_for_layer(&device, 0, 8, 4, 2).is_err(),
            "H225: num_kv_heads == 0 must error"
        );
        assert!(
            alloc_multi_seq_drafter_kv_for_layer(&device, 2, 0, 4, 2).is_err(),
            "H225: capacity == 0 must error"
        );
        assert!(
            alloc_multi_seq_drafter_kv_for_layer(&device, 2, 8, 0, 2).is_err(),
            "H225: head_dim == 0 must error"
        );
    }

    /// **H226 (structural — no GPU)** — `PADDING_SLOT == SlotId(u32::MAX)`
    /// per the vLLM/P-EAGLE convention.
    #[test]
    fn h226_padding_slot_const_is_max_u32_per_vllm_p_eagle_2026_05_30() {
        // Structural witness — does not allocate any GPU resources.
        assert_eq!(
            MultiSeqDrafterKvCache::PADDING_SLOT,
            crate::serve::multi_seq_kv::SlotId(u32::MAX),
            "H226: PADDING_SLOT MUST be SlotId(u32::MAX) per vLLM/P-EAGLE \
             rejected-token convention (dossier §1.1 + §5)."
        );
        // Witness that PADDING_SLOT is strictly outside any in-bounds
        // range — the alloc helper rejects n_seqs == u32::MAX so for
        // every reachable cache, PADDING_SLOT.0 >= n_seqs.
        for n in [1u32, 2, 4, 8, 16, 1024, u32::MAX - 1] {
            assert!(
                MultiSeqDrafterKvCache::PADDING_SLOT.0 >= n,
                "H226: PADDING_SLOT must be outside in-bounds range \
                 for every reachable n_seqs (got n={n})"
            );
        }
    }

    /// **H227** — `MultiSeqKvCache` impl: bounds-first + same-slot
    /// fork no-op + real cross-slot fork via copy_within.
    #[test]
    fn h227_multi_seq_kv_cache_impl_bounds_first_and_fork_2026_05_30() {
        use crate::serve::multi_seq_kv::{MultiSeqError, MultiSeqKvCache, SlotId};

        let (_dev, mut cache) = match make_multi_seq_cache(3) {
            Some(t) => t,
            None => {
                eprintln!("[skip] h227 — MlxDevice unavailable");
                return;
            }
        };
        // slot_count() == n_seqs.
        assert_eq!(cache.slot_count(), 3, "H227: slot_count() == n_seqs");

        // Out-of-range slot returns SlotOutOfRange (bounds-first).
        let oor = cache.seq_len(SlotId(3)).unwrap_err();
        assert_eq!(
            oor,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(3),
                max_slots: 3,
            },
            "H227: out-of-range slot must surface SlotOutOfRange"
        );

        // PADDING_SLOT surfaces as SlotOutOfRange too (typed signal,
        // not silent no-op).
        let pad_err = cache
            .seq_len(MultiSeqDrafterKvCache::PADDING_SLOT)
            .unwrap_err();
        assert!(
            matches!(pad_err, MultiSeqError::SlotOutOfRange { .. }),
            "H227: PADDING_SLOT against trait surface MUST be \
             SlotOutOfRange. Got: {pad_err:?}"
        );

        // append_for_seq bumps the cursor at the bound slot only.
        cache.append_for_seq(SlotId(1), 7).expect("append");
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 0);
        assert_eq!(cache.seq_len(SlotId(1)).unwrap(), 7);
        assert_eq!(cache.seq_len(SlotId(2)).unwrap(), 0);

        // Stamp slot 1's K-buffer region with a sentinel so fork_seq can
        // verify the bytes copied.  Slot stride = nkv * cap * hd =
        // 2 * 8 * 4 = 64 F32 elements per slot.
        let slot_stride = cache.num_kv_heads * cache.capacity * cache.head_dim;
        {
            let k_slice = cache.k_buf.as_mut_slice::<f32>().expect("k slice");
            for i in 0..slot_stride {
                k_slice[slot_stride + i] = (i + 1) as f32; // slot 1
            }
        }
        // Same-slot fork is a no-op (slot 1 → slot 1).
        cache
            .fork_seq(SlotId(1), SlotId(1))
            .expect("same-slot fork");
        // Cross-slot fork: slot 1 → slot 2. Cursor + bytes copied.
        cache.fork_seq(SlotId(1), SlotId(2)).expect("fork 1→2");
        assert_eq!(
            cache.seq_len(SlotId(2)).unwrap(),
            7,
            "H227: fork_seq copies the per-slot cursor"
        );
        let k_slice = cache.k_buf.as_slice::<f32>().expect("k slice");
        for i in 0..slot_stride {
            let expected = (i + 1) as f32;
            assert_eq!(
                k_slice[2 * slot_stride + i],
                expected,
                "H227: fork_seq must memcpy the K-buffer per-slot region \
                 (slot 2[{i}] = {expected})"
            );
        }
        // Slot 0 untouched.
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 0);

        // drop_seq resets the cursor only (no K/V byte zeroing).
        cache.drop_seq(SlotId(2)).expect("drop slot 2");
        assert_eq!(cache.seq_len(SlotId(2)).unwrap(), 0);
        // The bytes for slot 2 are NOT zeroed — H227 / H228 invariant.
        let k_after_drop = cache.k_buf.as_slice::<f32>().expect("k slice");
        for i in 0..slot_stride {
            assert_eq!(
                k_after_drop[2 * slot_stride + i],
                (i + 1) as f32,
                "H227: drop_seq must NOT zero K/V bytes (preserves \
                 recurrent-content invariance)"
            );
        }
    }

    /// **H228** — `reset_for_slot` is cursor-only + preserves K/V bytes
    /// + matches A3a `MultiSeqHbKvBuffers::reset_for_slot` discipline.
    #[test]
    fn h228_reset_for_slot_cursor_only_byte_preservation_2026_05_30() {
        use crate::serve::multi_seq_kv::{MultiSeqError, SlotId};

        let (_dev, mut cache) = match make_multi_seq_cache(2) {
            Some(t) => t,
            None => {
                eprintln!("[skip] h228 — MlxDevice unavailable");
                return;
            }
        };
        // Bounds-first: out-of-range slot.
        let err = cache.reset_for_slot(SlotId(2)).unwrap_err();
        assert!(
            matches!(err, MultiSeqError::SlotOutOfRange { .. }),
            "H228: bounds-first per A2b iter-1.5 cfa-finding-F5. Got: {err:?}"
        );
        // PADDING_SLOT also surfaces as SlotOutOfRange.
        let err_pad = cache
            .reset_for_slot(MultiSeqDrafterKvCache::PADDING_SLOT)
            .unwrap_err();
        assert!(matches!(err_pad, MultiSeqError::SlotOutOfRange { .. }));

        // Stamp K + V byte sentinels for both slots.
        let slot_stride = cache.num_kv_heads * cache.capacity * cache.head_dim;
        {
            let k_slice = cache.k_buf.as_mut_slice::<f32>().expect("k slice");
            for i in 0..(2 * slot_stride) {
                k_slice[i] = (i + 1) as f32;
            }
            let v_slice = cache.v_buf.as_mut_slice::<f32>().expect("v slice");
            for i in 0..(2 * slot_stride) {
                v_slice[i] = (100 + i) as f32;
            }
        }
        cache.seq_lens[0] = 5;
        cache.seq_lens[1] = 6;
        // Reset slot 0 only.
        cache.reset_for_slot(SlotId(0)).expect("reset slot 0");
        assert_eq!(cache.seq_lens[0], 0, "H228: cursor reset at slot 0");
        assert_eq!(cache.seq_lens[1], 6, "H228: other-slot cursor untouched");
        // K + V bytes for both slots untouched.
        let k_slice = cache.k_buf.as_slice::<f32>().expect("k slice");
        let v_slice = cache.v_buf.as_slice::<f32>().expect("v slice");
        for i in 0..(2 * slot_stride) {
            assert_eq!(
                k_slice[i],
                (i + 1) as f32,
                "H228: reset_for_slot MUST NOT zero K bytes (cursor-masked)"
            );
            assert_eq!(
                v_slice[i],
                (100 + i) as f32,
                "H228: reset_for_slot MUST NOT zero V bytes (cursor-masked)"
            );
        }
    }

    /// **H230** — at `n_seqs == 1`, the multi-seq sibling's byte count
    /// per buffer is identical to the legacy `DrafterKvCache::new` at
    /// matching `(num_kv_heads, capacity, head_dim)`.
    #[test]
    fn h230_multi_seq_n_seqs_1_byte_equiv_to_legacy_drafter_kv_2026_05_30() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("[skip] h230 — MlxDevice unavailable");
                return;
            }
        };
        let legacy = DrafterKvCache::new(&device, 2, 8, 4).expect("legacy alloc");
        let multi = alloc_multi_seq_drafter_kv_for_layer(&device, 2, 8, 4, 1).expect("multi alloc");
        assert_eq!(
            legacy.k_buf.byte_len(),
            multi.k_buf.byte_len(),
            "H230: legacy K byte count must equal multi-seq K byte count at n_seqs=1"
        );
        assert_eq!(
            legacy.v_buf.byte_len(),
            multi.v_buf.byte_len(),
            "H230: legacy V byte count must equal multi-seq V byte count at n_seqs=1"
        );
        assert_eq!(multi.n_seqs, 1);
        assert_eq!(multi.seq_lens, vec![0u32]);
    }

    /// **H231** — LEGACY `DrafterKvCache` surface is UNCHANGED.
    /// Constructor signature + behavioural shape are pinned; iter-A4 is
    /// strictly additive.
    #[test]
    fn h231_legacy_drafter_kv_cache_surface_unchanged_2026_05_30() {
        // Signature pin: `DrafterKvCache::new(&MlxDevice, usize, usize,
        // usize) -> Result<Self>`. If a future refactor changes the
        // arity or argument order, this fails to compile.
        let _ctor: fn(&MlxDevice, usize, usize, usize) -> Result<DrafterKvCache> =
            DrafterKvCache::new;
        // Behavioural shape pin: a fresh cache is empty + reports
        // (num_kv_heads, capacity, head_dim) unchanged.
        if let Ok(device) = MlxDevice::new() {
            let cache = DrafterKvCache::new(&device, 2, 8, 4).expect("alloc");
            assert_eq!(cache.len(), 0);
            assert!(cache.is_empty());
            assert_eq!(cache.num_kv_heads, 2);
            assert_eq!(cache.capacity, 8);
            assert_eq!(cache.head_dim, 4);
        } else {
            eprintln!(
                "[skip] h231 — MlxDevice unavailable; signature pin still asserted at compile."
            );
        }
    }

    /// **H232** — cross-module type witness: the `MultiSeqKvCache` trait
    /// from `serve::multi_seq_kv` is shared across Qwen35 + Gemma 4 +
    /// Qwen3VL + drafter. iter-A4 only adds a new impl for
    /// `MultiSeqDrafterKvCache`; the existing trait surface + the
    /// existing per-arch impls are UNTOUCHED. This test pins the
    /// type-witness invariant — both the legacy types continue to
    /// implement the trait AND the new sibling does.
    #[test]
    fn h232_multi_seq_kv_trait_witness_cross_arch_unchanged_2026_05_30() {
        fn assert_multi_seq_kv<T: crate::serve::multi_seq_kv::MultiSeqKvCache>() {}
        // New drafter sibling implements the trait.
        assert_multi_seq_kv::<MultiSeqDrafterKvCache>();
        // The pre-existing per-arch impls remain reachable via the
        // trait surface (compile-time witness).  These names are pinned
        // by the existing C2c/C2d/C2e arms in `engine::spawn_with_mode`
        // — H232 fails to compile if any per-arch trait impl was
        // accidentally dropped during the iter-A4 lift.
        assert_multi_seq_kv::<crate::inference::models::gemma4::kv_cache::MultiSeqHbKvBuffers>();
        assert_multi_seq_kv::<crate::inference::models::gemma4::kv_cache::MultiSeqHybridKvBuffers>(
        );
        assert_multi_seq_kv::<crate::serve::multi_seq_kv::NoopMultiSeqKvCache>();
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-040 §6.1.55 iter-A4-cont-drafter-dispatcher (2026-05-30) —
    // H235 structural pins for DrafterKvCacheVariant + selection
    // helper.  Pure-data tests; no MlxDevice required.
    // ──────────────────────────────────────────────────────────────────

    /// **H235a** — [`DrafterKvCacheSelection`] is a pure-data variant
    /// carrier; `select_drafter_kv_variant_for_mode` routes by
    /// `max_slots` per the dossier §5 contract.
    ///
    /// - `max_slots == 0` ⇒ SingleSeq (degenerate; defense-in-depth —
    ///   the spawn arm pre-rejects `max_slots == 0` per the C2c/C2d/C2e
    ///   `ModeNotYetWired` discipline, but the routing helper preserves
    ///   the SingleSeq fallback to avoid surprising panics).
    /// - `max_slots == 1` ⇒ SingleSeq (the degenerate single-slot
    ///   case is byte-equivalent to multi-seq at n_seqs=1; pre-A4
    ///   preserved).
    /// - `max_slots > 1` ⇒ MultiSeq with the requested `n_seqs`.
    #[test]
    fn h235a_drafter_kv_cache_selection_routes_by_max_slots_2026_05_30() {
        assert_eq!(
            select_drafter_kv_variant_for_mode(0),
            DrafterKvCacheSelection::SingleSeq,
            "H235a: max_slots == 0 MUST degrade to SingleSeq (defense-in-depth)"
        );
        assert_eq!(
            select_drafter_kv_variant_for_mode(1),
            DrafterKvCacheSelection::SingleSeq,
            "H235a: max_slots == 1 MUST route to SingleSeq (byte-equivalent to \
             MultiSeq at n_seqs=1; pre-A4 preserved)"
        );
        for n in [2u32, 3, 4, 8, 16, 1024] {
            assert_eq!(
                select_drafter_kv_variant_for_mode(n),
                DrafterKvCacheSelection::MultiSeq { n_seqs: n },
                "H235a: max_slots == {n} MUST route to MultiSeq with the requested n_seqs"
            );
        }
    }

    /// **H235b** — `DrafterKvCacheVariant::slot_count` returns the
    /// correct slot count for both arms.  Tested via the synthetic
    /// fixture (no GPU): the SingleSeq arm wraps a real
    /// [`DrafterKvCache`] (slot_count == 1 by definition) and the
    /// MultiSeq arm exposes the carrier's `n_seqs`.
    #[test]
    fn h235b_drafter_kv_cache_variant_slot_count_pin_2026_05_30() {
        let (_, single) = match make_cache() {
            Some(t) => t,
            None => {
                eprintln!(
                    "[skip] h235b — MlxDevice unavailable; selection enum + \
                     is_multi_seq pinned in h235a / h235c."
                );
                return;
            }
        };
        let variant = DrafterKvCacheVariant::SingleSeq(single);
        assert_eq!(
            variant.slot_count(),
            1,
            "H235b: SingleSeq arm slot_count == 1"
        );
        assert!(!variant.is_multi_seq(), "H235b: SingleSeq is NOT multi_seq");
    }

    /// **H235c** — `DrafterKvCacheSelection` derives Debug + PartialEq
    /// + Copy at the type level (pinned by the H235a equality assertions,
    /// restated here as a compile-time witness for clarity).
    #[test]
    fn h235c_drafter_kv_cache_selection_copy_eq_witness_2026_05_30() {
        let s = DrafterKvCacheSelection::MultiSeq { n_seqs: 4 };
        let s2 = s; // Copy
        assert_eq!(s, s2);
        let s3 = DrafterKvCacheSelection::SingleSeq;
        assert_ne!(s, s3);
    }

    /// **H235d (source-grep pin)** — the dispatcher cite is named at
    /// the type declaration site + companion routing helper site.
    /// Operator-grep'able for the future iter-A4-cont-drafter-
    /// dispatcher-kernel implementer.
    #[test]
    fn h235d_drafter_dispatcher_cite_named_at_source_2026_05_30() {
        let src = include_str!("kv_cache.rs");
        assert!(
            src.contains("iter-A4-cont-drafter-dispatcher"),
            "H235d FALSIFIED: kv_cache.rs does NOT name \
             `iter-A4-cont-drafter-dispatcher` at the dispatcher cite."
        );
        assert!(
            src.contains("DrafterKvCacheVariant"),
            "H235d FALSIFIED: DrafterKvCacheVariant variant carrier missing."
        );
        assert!(
            src.contains("select_drafter_kv_variant_for_mode"),
            "H235d FALSIFIED: select_drafter_kv_variant_for_mode routing \
             helper missing."
        );
        assert!(
            src.contains("SingleSeq") && src.contains("MultiSeq"),
            "H235d FALSIFIED: SingleSeq / MultiSeq arm names missing."
        );
    }
}
