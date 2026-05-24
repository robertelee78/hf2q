//! ADR-040 Phase A iter-1 — multi-seq KV cache trait surface + types.
//!
//! This module is the **pure trait + ID-type primitive** that the per-model
//! KV caches (Qwen35 `HybridKvCache`, Gemma 4 dense KV cache, EAGLE-3 /
//! DFlash drafter caches) will implement in Phase A iter-2+.  It contains
//! *no* production callsite, *no* GPU code, *no* per-model impls.  Pattern
//! mirrors `serve::multi_model` (iter-206 W74) and `serve::quant_select`
//! (iter-201 W51): a synthetic-fixture-tested data primitive that later
//! iters glue into the live serve path.
//!
//! # What this module ships in iter-1
//!
//! - [`SeqId`] / [`SlotId`] opaque-newtype IDs that the compiler refuses to
//!   confuse with raw `u32` or with each other.  Same trick the rest of the
//!   ADR-040 stack will use to keep "slot index" and "logical sequence ID"
//!   distinguishable in `forward_prefill.rs` + `forward_prefill_batched.rs`
//!   signatures (Phase B iter-4).
//! - [`MultiSeqLayout`] enum naming the two layouts ADR-040 considers:
//!   `SeparateSlots` (the Phase A default per ADR-040 §3.1) and `Paged`
//!   (reserved for a future ADR if Phase D §3.1 measurement justifies the
//!   kernel port).
//! - [`MultiSeqError`] enum covering the failure modes per-slot append /
//!   fork / drop can return.  OOM maps to 429 upstream per ADR-040 §3.5.
//! - [`MultiSeqKvCache`] trait surface — the contract every per-model KV
//!   cache implements in Phase A iter-2+.  Methods are the §5 AC-1
//!   enumeration verbatim: `slot_count`, `seq_len`, `append_for_seq`,
//!   `drop_seq`, `fork_seq`, plus `layout` for diagnostics.
//! - [`NoopMultiSeqKvCache`] — a fully-working test fixture backed by a
//!   `Vec<u32>` of per-slot lengths.  Two reasons it exists per the ADR
//!   mantra "no fallback, no stub (todo later) code":
//!     1. It IS the byte-equivalence target for iter-2's `HybridKvCache`
//!        impl — the trait surface stops measuring "I called the right
//!        method" and starts measuring "I produced the right lengths"
//!        only when a concrete impl is testable.
//!     2. Phase B iter-1's scheduler scaffold (sibling module
//!        `serve::scheduler`) can use a concrete `MultiSeqKvCache` type
//!        in its unit tests without dragging Metal / GGUF / model
//!        weights into Phase B's compile graph.
//!
//! # What this module does NOT do
//!
//! - Touch any per-model KV cache (`src/inference/models/qwen35/kv_cache.rs`,
//!   `src/inference/models/gemma4/kv_cache.rs`) — those impls land in
//!   Phase A iter-2+.
//! - Touch the `Engine` (`src/serve/api/engine.rs`) — Phase C iter-2 wires
//!   the `SchedulerPolicy` enum through `Engine::spawn`.
//! - Touch `forward_prefill.rs` / `forward_prefill_batched.rs` — Phase B
//!   iter-3 adds the `slot_id` parameter.
//! - Touch the `HotSwapManager` pool (`src/serve/multi_model.rs`) — the
//!   ADR-005 Phase 4 pool sits ABOVE this trait; per-model multi-seq
//!   slotting lives WITHIN each loaded engine.
//!
//! # Why the newtypes
//!
//! [`SeqId`] and [`SlotId`] both wrap `u32`.  If they were both
//! `pub type SlotId = u32;` aliases, a signature like
//! `fn append_for_seq(seq: u32, slot: u32, n: u32)` becomes a foot-gun:
//! at the callsite `append_for_seq(slot, seq, n)` compiles and silently
//! corrupts the cache.  Wrapping them in distinct newtype structs forces
//! the caller to write `SlotId(slot_idx)` / `SeqId(seq_idx)` explicitly,
//! and the compiler rejects the wrong-typed argument at the boundary.
//! This is the same discipline `LoadedHandle.repo_id: String` (always
//! `org/name`) vs `pool_key_str` (always `org/name@QUANT`) keeps in
//! `serve::multi_model` — distinct shapes get distinct types.
//!
//! In Phase A iter-1 the trait surface only takes `SlotId` parameters
//! (the slot is the physical thing being mutated; the sequence ID is
//! tracked by the scheduler in Phase B iter-3).  `SeqId` exists in the
//! module surface because Phase B iter-3's scheduler needs to thread a
//! logical-sequence identifier from the request envelope to the per-slot
//! KV append site; pinning the newtype here means iter-3 doesn't have to
//! revisit the trait surface to add it.
//!
//! # ADR-040 §5 AC-1 mapping
//!
//! AC-1 says:
//!
//! > `MultiSeqKvCache` trait lives in `src/serve/multi_seq_kv.rs` with
//! > `append_for_seq`, `drop_seq`, `fork_seq`, `seq_len`, `slot_count`
//! > methods.
//!
//! This module ships the trait surface verbatim.  Phase A iter-2+ closes
//! the rest of AC-1 (`HybridKvCache` impl, byte-equivalence at slot 0
//! vs `n_seqs=1`, Gemma 4 dense impl, per-slot O(1) append/drop bench).
//!
//! # Tests
//!
//! Synthetic-fixture unit tests cover:
//! - fresh cache: every slot starts at `seq_len = 0`
//! - `append_for_seq(n)` advances `seq_len(slot)` by `n` without touching
//!   other slots
//! - out-of-range `slot >= slot_count` returns
//!   [`MultiSeqError::SlotOutOfRange`] with both `slot` and `max_slots`
//!   populated (for `append_for_seq`, `seq_len`, `drop_seq`, `fork_seq`)
//! - `drop_seq` resets `seq_len` to 0
//! - `fork_seq(src, dst)` copies `src`'s `seq_len` to `dst` and leaves
//!   `src` unchanged
//! - `MultiSeqLayout::Paged` is reserved: constructor succeeds but
//!   `append_for_seq` returns [`MultiSeqError::LayoutNotSupported`]
//! - [`SeqId`] / [`SlotId`] do NOT interconvert at the type level
//!   (compile-time test via a doc-test snippet)
//! - [`MultiSeqError`] `Debug` formatting names the slot id + relevant
//!   context for each variant

/// Logical-sequence opaque ID.
///
/// Wraps `u32` to keep "sequence ID" distinguishable from "slot index" at
/// the type level.  Constructed by the scheduler (Phase B iter-3) at
/// request admission time; flows through to the per-slot KV append in
/// Phase B iter-4 via the `forward_prefill` / `forward_prefill_batched`
/// signatures.
///
/// Iter-1 surfaces this type to pin the newtype here so subsequent iters
/// don't have to revisit the trait surface.  The Phase A iter-1
/// [`MultiSeqKvCache`] trait methods themselves only take [`SlotId`] —
/// the sequence ID is metadata the scheduler tracks, not state the KV
/// cache stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId(pub u32);

impl SeqId {
    /// Sentinel sequence ID used by Phase B iter-3's scheduler when an
    /// admission slot has not yet been assigned a sequence.  Distinct
    /// from `SeqId(0)` (which IS a valid sequence ID, assigned to the
    /// first admitted request).
    pub const UNASSIGNED: SeqId = SeqId(u32::MAX);
}

/// Physical slot opaque ID.
///
/// Indexes into the per-loaded-model multi-seq KV cache (`slot_count`
/// slots per engine).  Wraps `u32` to keep "slot index" distinguishable
/// from "sequence ID" at the type level — see the module-level docs for
/// why this matters.
///
/// The valid range is `0..slot_count`; an out-of-range `SlotId` produces
/// [`MultiSeqError::SlotOutOfRange`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlotId(pub u32);

/// Per-engine multi-seq KV layout strategy.
///
/// Per ADR-040 §3.1, Phase A iter-1 ships [`Self::SeparateSlots`] as the
/// default — extends the existing `[..., max_seq_len, n_seqs]` buffer
/// shape (e.g., `HybridKvCache.k/v: MlxBuffer` at
/// `src/inference/models/qwen35/kv_cache.rs:14-16`) from `n_seqs = 1` to
/// `n_seqs = N` with per-slot `current_len`.  Reuses every existing
/// kernel; no Metal work.
///
/// [`Self::Paged`] is reserved for a future ADR.  Phase D's benchmark
/// (ADR-040 §6 Phase D / §3.1 alternative) decides whether the
/// PagedAttention kernel port is worth doing: only if `SeparateSlots`
/// shows ≥30% memory waste under N=8 concurrent at production context
/// lengths.  Iter-1 carries the variant so the trait + error types lock
/// the eventual surface shape, but the [`NoopMultiSeqKvCache`] fixture
/// refuses to append on Paged with [`MultiSeqError::LayoutNotSupported`]
/// (the per-model impls in Phase A iter-2+ will do the same).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiSeqLayout {
    /// Per-slot independent allocation: shape `[..., max_seq_len,
    /// n_seqs]` with `n_seqs = slot_count`.  Phase A default.
    SeparateSlots,
    /// Block-paged allocation à la vLLM PagedAttention.  Reserved for a
    /// future ADR if Phase D's benchmark surfaces SeparateSlots waste.
    /// Phase A iter-1 carries the variant for surface stability; the
    /// `NoopMultiSeqKvCache` fixture and the Phase A iter-2+ per-model
    /// impls refuse to operate under this layout until that ADR ships.
    Paged,
}

/// Per-slot append / fork / drop failure cases.
///
/// Per ADR-040 §3.5, `SlotOom` maps to a 429 + Retry-After response
/// upstream (Decision #19 contract preserved).  `SlotOutOfRange` is a
/// caller-side defect — the scheduler should never have handed out an
/// out-of-range `SlotId`.  `LayoutNotSupported` is operator config drift
/// (asked for `MultiSeqLayout::Paged` before the future PagedAttention
/// ADR shipped).
///
/// `Clone + PartialEq + Eq` so test assertions can match the full error
/// shape (including the populated fields, not just the discriminant) via
/// `assert_eq!`.  `Debug` is derived so the assertion failure messages
/// name the slot id + relevant context inline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiSeqError {
    /// The caller passed a `SlotId` outside `0..slot_count`.  Defensive
    /// — the scheduler (Phase B iter-3+) is expected to bound-check
    /// before reaching the per-slot append site, but the trait surface
    /// validates anyway so a misbehaving caller cannot corrupt the
    /// cache.  Both fields are populated so the operator-facing message
    /// can name what was attempted vs what was permitted.
    SlotOutOfRange {
        /// The out-of-range slot ID the caller passed in.
        slot: SlotId,
        /// The current `slot_count` of the cache.  Valid IDs are
        /// `0..max_slots`.
        max_slots: u32,
    },
    /// The per-slot KV budget (ADR-040 §3.5: total
    /// `kv_cache_budget_bytes` divided equally across slots) cannot
    /// accommodate the requested append.  Mapped to 429 + Retry-After
    /// upstream per Decision #19.  `needed_bytes` is the append size
    /// the caller attempted; `budget_bytes` is the per-slot allotment.
    SlotOom {
        /// The slot that ran out of budget.
        slot: SlotId,
        /// Bytes the append would have needed.
        needed_bytes: u64,
        /// Per-slot budget remaining.
        budget_bytes: u64,
    },
    /// The cache was constructed with a [`MultiSeqLayout`] variant that
    /// the current build does not implement.  Iter-1 always trips on
    /// [`MultiSeqLayout::Paged`]; a future ADR that ports
    /// PagedAttention will lift this restriction by shipping a real
    /// impl for that variant.
    LayoutNotSupported {
        /// The unsupported layout the caller selected.
        layout: MultiSeqLayout,
    },
}

impl std::fmt::Display for MultiSeqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SlotOutOfRange { slot, max_slots } => write!(
                f,
                "multi-seq KV cache slot {:?} out of range; valid slots are 0..{}",
                slot.0, max_slots
            ),
            Self::SlotOom {
                slot,
                needed_bytes,
                budget_bytes,
            } => write!(
                f,
                "multi-seq KV cache slot {:?} out of memory: append needed \
                 {needed_bytes} bytes; per-slot budget {budget_bytes} bytes \
                 (ADR-040 §3.5 — map to 429 + Retry-After upstream)",
                slot.0
            ),
            Self::LayoutNotSupported { layout } => write!(
                f,
                "multi-seq KV cache layout {layout:?} is not supported in this build \
                 (Phase A iter-1 ships SeparateSlots only; Paged is reserved for a \
                 future PagedAttention ADR per ADR-040 §3.1)"
            ),
        }
    }
}

impl std::error::Error for MultiSeqError {}

/// Multi-seq KV cache contract (ADR-040 §5 AC-1).
///
/// Implemented in Phase A iter-2+ by:
/// - `HybridKvCache` (Qwen35) at `src/inference/models/qwen35/kv_cache.rs`
/// - Gemma 4 dense KV cache at `src/inference/models/gemma4/kv_cache.rs`
/// - EAGLE-3 / DFlash drafter caches at
///   `src/inference/spec_decode/{eagle3,dflash}/kv_cache.rs`
///   (research-quality, gated on Phase E1 per ADR-040 §4 question 5)
///
/// **Per-slot O(1) bound** (AC-1): `append_for_seq` and `drop_seq` must
/// not iterate over other slots.  The Phase A iter-2 `HybridKvCache`
/// impl satisfies this trivially because per-slot `current_len: Vec<u32>`
/// is already indexed by slot; the GPU buffer writes are slot-local by
/// construction.
///
/// **Byte-equivalence at slot 0** (AC-1): for `slot_count = 1`, the per-
/// model impl's behaviour must be byte-equivalent to its pre-ADR-040
/// `n_seqs = 1` baseline at slot 0.  This is the ADR-040 §3.6 backward
/// compatibility contract pinned at the per-model level; Phase A iter-6
/// adds the per-family parity gate.
pub trait MultiSeqKvCache {
    /// The KV layout in use.  Returned for diagnostics (e.g.
    /// `/v1/models` extension fields, `/metrics` labels) and so the
    /// scheduler can route fork operations differently if Phase D's
    /// future PagedAttention impl needs block-aware admission.
    fn layout(&self) -> MultiSeqLayout;

    /// Number of physical slots the cache is provisioned for.  Valid
    /// `SlotId`s are `0..slot_count`.  Set at engine construction time
    /// from the `max_slots` knob (ADR-040 §3.4 default 4, env override
    /// `HF2Q_MAX_SLOTS`).
    fn slot_count(&self) -> u32;

    /// Current sequence length stored in `slot`.  Returns
    /// [`MultiSeqError::SlotOutOfRange`] when `slot.0 >= slot_count()`.
    /// Iter-2+ per-model impls return the slot's `current_len` field
    /// directly.
    fn seq_len(&self, slot: SlotId) -> Result<u32, MultiSeqError>;

    /// Append `n_tokens` to `slot`'s KV cache, advancing the slot's
    /// `seq_len` by `n_tokens`.  O(1) — does NOT iterate over other
    /// slots.
    ///
    /// Errors:
    /// - [`MultiSeqError::SlotOutOfRange`] when `slot.0 >= slot_count()`
    /// - [`MultiSeqError::SlotOom`] when the per-slot budget cannot
    ///   accommodate the append (maps to 429 upstream per ADR-040 §3.5)
    /// - [`MultiSeqError::LayoutNotSupported`] when the cache was
    ///   constructed with [`MultiSeqLayout::Paged`] (iter-1; future
    ///   PagedAttention ADR lifts this)
    fn append_for_seq(&mut self, slot: SlotId, n_tokens: u32) -> Result<(), MultiSeqError>;

    /// Drop `slot`'s sequence: reset `seq_len` to 0 and release any
    /// per-slot resources back to the cache's free pool.  Returns
    /// [`MultiSeqError::SlotOutOfRange`] when `slot.0 >= slot_count()`.
    ///
    /// O(1).  Does NOT free the underlying GPU buffer (the buffer is
    /// engine-lifetime; the slot becomes available for the next
    /// admission immediately).
    fn drop_seq(&mut self, slot: SlotId) -> Result<(), MultiSeqError>;

    /// Copy `src`'s sequence state into `dst`.  After a successful
    /// fork, `dst`'s `seq_len` equals `src`'s `seq_len` and `src` is
    /// unchanged.  Used by Phase B iter-6's prefix-share path
    /// (admitting a new request that shares a prefix with an
    /// in-flight slot — fork is cheaper than re-prefilling).
    ///
    /// Errors:
    /// - [`MultiSeqError::SlotOutOfRange`] when either `src.0` or
    ///   `dst.0` is `>= slot_count()`.  Iter-1's fixture validates `src`
    ///   first then `dst` (deterministic for tests); per-model impls
    ///   must follow the same order so the error is reproducible.
    /// - [`MultiSeqError::SlotOom`] if the per-slot budget cannot
    ///   accommodate the copy (only fires under PagedAttention or
    ///   future block-shared layouts; SeparateSlots iter-2 cannot
    ///   `SlotOom` on fork because it is a memcpy into pre-allocated
    ///   per-slot buffers).
    /// - [`MultiSeqError::LayoutNotSupported`] under Paged in iter-1.
    fn fork_seq(&mut self, src: SlotId, dst: SlotId) -> Result<(), MultiSeqError>;
}

/// Test-only fixture impl backed by a `Vec<u32>` of per-slot lengths.
///
/// Two reasons this exists per the ADR mantra "no fallback, no stub
/// (todo later) code" (ADR-040 §7):
///
/// 1. **Byte-equivalence target.**  Iter-2's `HybridKvCache` impl needs
///    a concrete reference to compare against.  The fixture's `Vec<u32>`
///    per-slot length tracking is the trivial-but-correct semantics the
///    real impl must match (modulo GPU buffer writes the fixture does
///    not perform).
///
/// 2. **Phase B scaffold concrete type.**  The sibling
///    `serve::scheduler` module (Phase B iter-1) needs a concrete
///    `MultiSeqKvCache` type for its unit tests without dragging Metal /
///    GGUF / model weights into Phase B's compile graph.  The fixture
///    type is `pub` (not `pub(crate)`) for this reason — Phase B
///    iter-1's tests construct it directly.
///
/// **NOT a fallback.**  Production code MUST use the per-model impl
/// (`HybridKvCache` etc.).  This fixture is for tests only; the
/// `#[allow(dead_code)]` on the module declaration at
/// `src/serve/mod.rs:27` is the same staging pattern as
/// `serve::quant_select`'s pre-wiring state.
#[derive(Debug, Clone)]
pub struct NoopMultiSeqKvCache {
    layout: MultiSeqLayout,
    /// `slot_lens[i]` is `seq_len(SlotId(i))`.  `len()` is the
    /// `slot_count`.  All entries start at `0`; `append_for_seq` adds
    /// to the indexed entry, `drop_seq` resets it, `fork_seq` copies.
    slot_lens: Vec<u32>,
}

impl NoopMultiSeqKvCache {
    /// Construct a fixture with `slot_count` slots, each starting at
    /// `seq_len = 0`.  `layout` is recorded as-is; for
    /// [`MultiSeqLayout::Paged`] the constructor still succeeds (per
    /// iter-1 surface contract — we want to be able to *construct* the
    /// reserved-layout cache and observe that *append* trips on
    /// [`MultiSeqError::LayoutNotSupported`], not have the constructor
    /// itself refuse).
    pub fn new(slot_count: u32, layout: MultiSeqLayout) -> Self {
        Self {
            layout,
            slot_lens: vec![0u32; slot_count as usize],
        }
    }

    /// Internal helper: validate `slot.0 < slot_count`.  Pulled into a
    /// helper so every trait method returns the SAME error shape (same
    /// field ordering, same `max_slots` value) on out-of-range.
    fn check_slot(&self, slot: SlotId) -> Result<(), MultiSeqError> {
        let max = self.slot_lens.len() as u32;
        if slot.0 >= max {
            Err(MultiSeqError::SlotOutOfRange {
                slot,
                max_slots: max,
            })
        } else {
            Ok(())
        }
    }

    /// Internal helper: refuse on the Paged layout per iter-1 contract.
    fn check_layout_supported(&self) -> Result<(), MultiSeqError> {
        match self.layout {
            MultiSeqLayout::SeparateSlots => Ok(()),
            MultiSeqLayout::Paged => Err(MultiSeqError::LayoutNotSupported {
                layout: MultiSeqLayout::Paged,
            }),
        }
    }
}

impl MultiSeqKvCache for NoopMultiSeqKvCache {
    fn layout(&self) -> MultiSeqLayout {
        self.layout
    }

    fn slot_count(&self) -> u32 {
        self.slot_lens.len() as u32
    }

    fn seq_len(&self, slot: SlotId) -> Result<u32, MultiSeqError> {
        self.check_slot(slot)?;
        // `seq_len` is a read; it does not depend on the layout — even
        // a Paged cache should be able to report its current per-slot
        // length when one exists.  The fixture never enters a state
        // where `slot_lens` is meaningless, so this returns directly.
        Ok(self.slot_lens[slot.0 as usize])
    }

    fn append_for_seq(&mut self, slot: SlotId, n_tokens: u32) -> Result<(), MultiSeqError> {
        // Layout check FIRST so a Paged out-of-range append returns
        // LayoutNotSupported (the more diagnostic error for an
        // operator config drift) rather than SlotOutOfRange.  Per-model
        // impls in iter-2+ follow the same ordering.
        self.check_layout_supported()?;
        self.check_slot(slot)?;
        // Saturating add so a pathological `n_tokens = u32::MAX` test
        // cannot panic in debug builds; production callers bound
        // `n_tokens` by the per-slot max_seq_len long before reaching
        // this method.
        let entry = &mut self.slot_lens[slot.0 as usize];
        *entry = entry.saturating_add(n_tokens);
        Ok(())
    }

    fn drop_seq(&mut self, slot: SlotId) -> Result<(), MultiSeqError> {
        // Layout check FIRST per the append ordering.
        self.check_layout_supported()?;
        self.check_slot(slot)?;
        self.slot_lens[slot.0 as usize] = 0;
        Ok(())
    }

    fn fork_seq(&mut self, src: SlotId, dst: SlotId) -> Result<(), MultiSeqError> {
        self.check_layout_supported()?;
        // Validate src first then dst — the trait docs pin this order
        // so error reproduction is deterministic for tests.
        self.check_slot(src)?;
        self.check_slot(dst)?;
        let src_len = self.slot_lens[src.0 as usize];
        self.slot_lens[dst.0 as usize] = src_len;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Compile-time type-distinctness check.
//
// The doc-test below demonstrates that `SeqId` and `SlotId` are NOT
// interchangeable at the type level — passing the wrong-typed argument
// is a compile error.  Per ADR-040 §3 the newtypes exist exactly to
// catch this class of bug at the boundary.
//
// Doc-test (compile_fail) — does NOT run as a unit test, but the
// `cargo test --doc` pass + the inline reasoning serve as the proof.
// ─────────────────────────────────────────────────────────────────────

/// Compile-time proof that [`SeqId`] and [`SlotId`] do not interconvert.
///
/// ```compile_fail
/// use hf2q::serve::multi_seq_kv::{SeqId, SlotId, NoopMultiSeqKvCache, MultiSeqLayout, MultiSeqKvCache};
/// let mut cache = NoopMultiSeqKvCache::new(4, MultiSeqLayout::SeparateSlots);
/// // Mistakenly pass a SeqId where a SlotId is expected — must NOT compile.
/// let seq = SeqId(0);
/// cache.append_for_seq(seq, 3).unwrap();
/// ```
///
/// ```compile_fail
/// use hf2q::serve::multi_seq_kv::{SeqId, SlotId};
/// // u32 → SlotId is explicit; u32 → SeqId is explicit; SeqId ↛ SlotId.
/// fn takes_slot(_: SlotId) {}
/// takes_slot(SeqId(0));
/// ```
#[allow(dead_code)]
const _ID_NEWTYPES_ARE_DISTINCT: () = ();

#[cfg(test)]
mod tests {
    use super::*;

    // ── starting state ──────────────────────────────────────────────────

    #[test]
    fn noop_cache_starts_with_zero_seq_len_per_slot() {
        let cache = NoopMultiSeqKvCache::new(4, MultiSeqLayout::SeparateSlots);
        assert_eq!(cache.slot_count(), 4);
        assert_eq!(cache.layout(), MultiSeqLayout::SeparateSlots);
        for i in 0..4 {
            assert_eq!(cache.seq_len(SlotId(i)).unwrap(), 0, "slot {i} not zero");
        }
    }

    // ── append ──────────────────────────────────────────────────────────

    #[test]
    fn noop_cache_append_advances_seq_len() {
        let mut cache = NoopMultiSeqKvCache::new(4, MultiSeqLayout::SeparateSlots);
        cache.append_for_seq(SlotId(1), 3).unwrap();
        assert_eq!(cache.seq_len(SlotId(1)).unwrap(), 3);
        // Other slots must be untouched (per-slot O(1) invariant).
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 0);
        assert_eq!(cache.seq_len(SlotId(2)).unwrap(), 0);
        assert_eq!(cache.seq_len(SlotId(3)).unwrap(), 0);
        // Repeated append accumulates.
        cache.append_for_seq(SlotId(1), 5).unwrap();
        assert_eq!(cache.seq_len(SlotId(1)).unwrap(), 8);
    }

    // ── out-of-range slot ──────────────────────────────────────────────

    #[test]
    fn noop_cache_slot_out_of_range_errors_named() {
        let mut cache = NoopMultiSeqKvCache::new(2, MultiSeqLayout::SeparateSlots);

        // append
        let err = cache.append_for_seq(SlotId(2), 1).unwrap_err();
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(2),
                max_slots: 2
            },
            "append OOR must populate both fields; got {err:?}"
        );

        // seq_len
        let err = cache.seq_len(SlotId(7)).unwrap_err();
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(7),
                max_slots: 2
            },
            "seq_len OOR must populate both fields; got {err:?}"
        );

        // drop
        let err = cache.drop_seq(SlotId(42)).unwrap_err();
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(42),
                max_slots: 2
            },
            "drop OOR must populate both fields; got {err:?}"
        );
    }

    // ── drop ───────────────────────────────────────────────────────────

    #[test]
    fn noop_cache_drop_resets_seq_len_to_zero() {
        let mut cache = NoopMultiSeqKvCache::new(2, MultiSeqLayout::SeparateSlots);
        cache.append_for_seq(SlotId(0), 11).unwrap();
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 11);
        cache.drop_seq(SlotId(0)).unwrap();
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 0);
        // Drop is idempotent: a second drop on an already-empty slot
        // succeeds without changing anything (per-slot O(1) reset).
        cache.drop_seq(SlotId(0)).unwrap();
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 0);
    }

    // ── fork ───────────────────────────────────────────────────────────

    #[test]
    fn noop_cache_fork_copies_seq_len_from_src_to_dst() {
        let mut cache = NoopMultiSeqKvCache::new(3, MultiSeqLayout::SeparateSlots);
        cache.append_for_seq(SlotId(0), 5).unwrap();
        cache.fork_seq(SlotId(0), SlotId(2)).unwrap();
        assert_eq!(cache.seq_len(SlotId(2)).unwrap(), 5, "dst must match src");
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 5, "src must be unchanged");
        // Untouched slot remains zero (per-slot O(1) invariant).
        assert_eq!(cache.seq_len(SlotId(1)).unwrap(), 0);
    }

    #[test]
    fn noop_cache_fork_src_oob_errors() {
        let mut cache = NoopMultiSeqKvCache::new(2, MultiSeqLayout::SeparateSlots);
        let err = cache.fork_seq(SlotId(5), SlotId(0)).unwrap_err();
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(5),
                max_slots: 2
            },
            "fork src OOR must surface src ID first; got {err:?}"
        );
    }

    #[test]
    fn noop_cache_fork_dst_oob_errors() {
        let mut cache = NoopMultiSeqKvCache::new(2, MultiSeqLayout::SeparateSlots);
        // src=0 is valid; dst=9 must surface as the OOR victim.
        let err = cache.fork_seq(SlotId(0), SlotId(9)).unwrap_err();
        assert_eq!(
            err,
            MultiSeqError::SlotOutOfRange {
                slot: SlotId(9),
                max_slots: 2
            },
            "fork dst OOR must surface dst ID; got {err:?}"
        );
    }

    // ── reserved Paged layout ──────────────────────────────────────────

    #[test]
    fn noop_cache_paged_layout_errors_on_append() {
        // Constructor still works (per iter-1 surface contract — see
        // NoopMultiSeqKvCache::new docs).
        let mut cache = NoopMultiSeqKvCache::new(4, MultiSeqLayout::Paged);
        assert_eq!(cache.slot_count(), 4);
        assert_eq!(cache.layout(), MultiSeqLayout::Paged);
        // Append is refused with the diagnostic LayoutNotSupported.
        let err = cache.append_for_seq(SlotId(0), 1).unwrap_err();
        assert_eq!(
            err,
            MultiSeqError::LayoutNotSupported {
                layout: MultiSeqLayout::Paged
            },
            "Paged append must trip LayoutNotSupported; got {err:?}"
        );
        // Drop + fork are also refused under Paged (same reservation).
        let err = cache.drop_seq(SlotId(0)).unwrap_err();
        assert_eq!(
            err,
            MultiSeqError::LayoutNotSupported {
                layout: MultiSeqLayout::Paged
            }
        );
        let err = cache.fork_seq(SlotId(0), SlotId(1)).unwrap_err();
        assert_eq!(
            err,
            MultiSeqError::LayoutNotSupported {
                layout: MultiSeqLayout::Paged
            }
        );
    }

    // ── newtype distinctness (runtime sanity for the compile-time guard) ──

    #[test]
    fn seq_id_and_slot_id_are_distinct_types() {
        // Runtime sanity: the wrapped u32 round-trips, but the two
        // types are not `==` even when wrapping the same number.  The
        // load-bearing proof lives in the `compile_fail` doc-test on
        // `_ID_NEWTYPES_ARE_DISTINCT`; this unit test is the
        // companion runtime-side check that the newtypes carry the
        // value through without alteration.
        let s = SlotId(7);
        let q = SeqId(7);
        assert_eq!(s.0, 7);
        assert_eq!(q.0, 7);
        // SeqId::UNASSIGNED is distinct from SeqId(0).
        assert_ne!(SeqId::UNASSIGNED, SeqId(0));
        assert_eq!(SeqId::UNASSIGNED.0, u32::MAX);
        // The following line, if uncommented, MUST fail to compile:
        //     let _: SlotId = q;
        // The doc-test on `_ID_NEWTYPES_ARE_DISTINCT` automates that
        // proof under `cargo test --doc`.
    }

    // ── Debug formatting names slot id + context ────────────────────────

    #[test]
    fn multi_seq_error_display_names_fields() {
        // SlotOutOfRange — Display + Debug both name the slot + max.
        let e = MultiSeqError::SlotOutOfRange {
            slot: SlotId(11),
            max_slots: 4,
        };
        let d = format!("{e:?}");
        assert!(d.contains("SlotOutOfRange"), "Debug missing variant: {d}");
        assert!(d.contains("11"), "Debug missing slot id: {d}");
        assert!(d.contains("4"), "Debug missing max_slots: {d}");
        let s = format!("{e}");
        assert!(s.contains("11"), "Display missing slot id: {s}");
        assert!(s.contains("0..4"), "Display missing valid range: {s}");

        // SlotOom — names slot, needed, budget.
        let e = MultiSeqError::SlotOom {
            slot: SlotId(2),
            needed_bytes: 1024,
            budget_bytes: 256,
        };
        let d = format!("{e:?}");
        assert!(d.contains("SlotOom"), "Debug missing variant: {d}");
        assert!(d.contains('2'), "Debug missing slot id: {d}");
        assert!(d.contains("1024"), "Debug missing needed: {d}");
        assert!(d.contains("256"), "Debug missing budget: {d}");
        let s = format!("{e}");
        assert!(s.contains("1024"), "Display missing needed: {s}");
        assert!(s.contains("256"), "Display missing budget: {s}");
        assert!(s.contains("429"), "Display must reference the 429 mapping: {s}");

        // LayoutNotSupported — names the layout.
        let e = MultiSeqError::LayoutNotSupported {
            layout: MultiSeqLayout::Paged,
        };
        let d = format!("{e:?}");
        assert!(d.contains("LayoutNotSupported"), "Debug missing variant: {d}");
        assert!(d.contains("Paged"), "Debug missing layout: {d}");
        let s = format!("{e}");
        assert!(s.contains("Paged"), "Display missing layout: {s}");
        assert!(
            s.contains("PagedAttention") || s.contains("future"),
            "Display must point to the future ADR: {s}"
        );
    }

    // ── trait object usability ─────────────────────────────────────────

    #[test]
    fn noop_cache_usable_as_trait_object() {
        // Phase B iter-1's scheduler scaffold will hold a
        // `Box<dyn MultiSeqKvCache>` so it can swap between the fixture
        // (in tests) and the per-model impls (in production).  Pin
        // here that the trait is object-safe.
        let mut cache: Box<dyn MultiSeqKvCache> = Box::new(NoopMultiSeqKvCache::new(
            2,
            MultiSeqLayout::SeparateSlots,
        ));
        assert_eq!(cache.slot_count(), 2);
        cache.append_for_seq(SlotId(0), 4).unwrap();
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 4);
        cache.drop_seq(SlotId(0)).unwrap();
        assert_eq!(cache.seq_len(SlotId(0)).unwrap(), 0);
    }
}
