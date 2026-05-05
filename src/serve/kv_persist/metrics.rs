//! ADR-017 §R-F7 — cache-side telemetry seam.
//!
//! Defines the trait + label-constant set that the Phase A.2 substrate
//! (`block_store::DiskBlockStore`, `recovery::*`) calls into when a
//! quarantine or eviction event fires. The concrete sink lives in
//! `serve::api::state::KvSpillCounters` (bin-side), but the substrate
//! deliberately depends only on this trait so the narrow `kv_persist`
//! lib facade (`src/lib.rs`) can compile without pulling the bin-side
//! `serve::api` module graph.
//!
//! ## Why a trait (not a direct struct ref)
//!
//! `src/lib.rs` re-exports `kv_persist::block_store` + `kv_persist::recovery`
//! into a narrow lib target so `tests/kv_persist_writer_kill_minus_9.rs`
//! can drive the real production types from a forked child. That lib
//! target intentionally OMITS `serve::api` (and its long tail of
//! `multi_model` + `intelligence::hardware` deps). Importing
//! `KvSpillCounters` directly inside `block_store.rs` / `recovery.rs`
//! would break the lib build.
//!
//! Threading the seam as `Option<&Arc<dyn KvCacheMetricsSink>>`
//! decouples the substrate from the concrete sink: the lib build
//! doesn't see `KvSpillCounters` at all (nothing in the lib's import
//! graph mentions it); the bin build wires the production
//! `KvSpillCounters` instance through `Arc<dyn KvCacheMetricsSink>`.
//! Tests that don't care about telemetry pass `None`.
//!
//! ## Why the label-constant arrays live here too
//!
//! `KV_QUARANTINE_REASONS` and `KV_EVICTION_TRIGGERS` are the closed
//! enums of `/metrics` label values. Both are referenced by
//! (a) the bump sites here in `kv_persist`, and
//! (b) the `/metrics` handler in `serve::api::handlers`, which emits
//!     them as the cardinality preamble.
//! Hosting them in `kv_persist::metrics` keeps the two consumers in
//! lockstep — one place to amend if a label is added, one compile-time
//! match for both sides.

use std::sync::Arc;

/// ADR-017 §R-F7: closed enum of `hf2q_kv_quarantined_total{reason="..."}`
/// label values. Order is load-bearing — index lookups in
/// `KvCacheMetricsSink::record_quarantine` use the corresponding
/// `KvQuarantineReason` variant index, and the `/metrics` emit walks
/// this array in order so successive scrapes stay diff-stable.
///
/// Adding a new variant MUST append (not insert mid-array). The
/// constant length ([`KV_QUARANTINE_REASON_COUNT`]) is what
/// `KvSpillCounters` uses to size its `[AtomicU64; N]` storage.
pub const KV_QUARANTINE_REASONS: &[&str] = &["trunc", "verbump", "bodyhash", "parity"];

/// Cardinality of [`KV_QUARANTINE_REASONS`]. Used by `KvSpillCounters`
/// as the storage-array length.
pub const KV_QUARANTINE_REASON_COUNT: usize = 4;

/// ADR-017 §R-F7: closed enum of `hf2q_kv_cache_evictions_total{trigger="..."}`
/// label values. Today only `"budget_overflow"` fires (the only path
/// through `evict_lru_until_under_budget`). Future triggers (e.g.
/// `"manual"` for an operator-driven cache flush) MUST append to the
/// end of this array to preserve scrape ordering.
pub const KV_EVICTION_TRIGGERS: &[&str] = &["budget_overflow"];

/// Cardinality of [`KV_EVICTION_TRIGGERS`]. Sized for the single
/// trigger today.
pub const KV_EVICTION_TRIGGER_COUNT: usize = 1;

/// ADR-017 §R-F7 quarantine-reason mirror enum, decoupled from
/// `kv_persist::recovery::QuarantineReason` so the metrics seam doesn't
/// pull `recovery` into modules that only need to bump telemetry.
/// `From<recovery::QuarantineReason>` in `recovery.rs` keeps the two
/// enums in 1:1 correspondence (exhaustive match — adding a variant to
/// either side without the other is a compile error).
///
/// Variant order matches [`KV_QUARANTINE_REASONS`] index ordering:
///   0. `TruncatedHeader` → `"trunc"`
///   1. `VersionMismatch` → `"verbump"`
///   2. `BodyHashMismatch` → `"bodyhash"`
///   3. `ParityFail` → `"parity"`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvQuarantineReason {
    TruncatedHeader,
    VersionMismatch,
    BodyHashMismatch,
    ParityFail,
}

impl KvQuarantineReason {
    /// Index into [`KV_QUARANTINE_REASONS`] / the
    /// `KvSpillCounters.quarantines` storage array. Match is exhaustive
    /// so the compiler enforces lockstep between the variant set, the
    /// label array, and the storage size.
    pub fn index(self) -> usize {
        match self {
            KvQuarantineReason::TruncatedHeader => 0,
            KvQuarantineReason::VersionMismatch => 1,
            KvQuarantineReason::BodyHashMismatch => 2,
            KvQuarantineReason::ParityFail => 3,
        }
    }

    /// Metric label string ([`KV_QUARANTINE_REASONS`] entry at
    /// `self.index()`). Convenience for callers that want the label
    /// without going through the array index.
    pub fn as_metric_label(self) -> &'static str {
        KV_QUARANTINE_REASONS[self.index()]
    }
}

/// ADR-017 §R-F7 telemetry sink — implemented by
/// `serve::api::state::KvSpillCounters` (bin-side). The Phase A.2
/// substrate (`block_store`, `recovery`) takes
/// `Option<&Arc<dyn KvCacheMetricsSink>>` at every call site so:
///
///   * The narrow `kv_persist` lib facade compiles without pulling
///     `serve::api` into its module graph (see `src/lib.rs`).
///   * Tests that don't care about telemetry pass `None` and the bump
///     becomes a no-op.
///   * The production wiring (`cmd_serve`) hands the same Arc the
///     `/metrics` handler reads, so bump and scrape see one shared
///     table without an extra hop through global state.
///
/// `Send + Sync` because the manager's eviction path may run from a
/// concurrent tokio task; `KvSpillCounters` is already Send + Sync so
/// the bound is a no-cost discipline check at trait-impl time.
pub trait KvCacheMetricsSink: Send + Sync {
    /// Bump `hf2q_kv_quarantined_total{reason=<reason.as_metric_label()>}`
    /// by 1. Called by the recovery scan and read-path quarantine sites
    /// AFTER the rename to `kv-quarantine/` succeeds (so a partial-move
    /// failure never overcounts).
    fn record_quarantine(&self, reason: KvQuarantineReason);

    /// Bump `hf2q_kv_cache_evictions_total{trigger="budget_overflow"}`
    /// by 1. Called by `block_store::DiskBlockStore::evict_lru_until_under_budget`
    /// PER block actually removed (pinned / racing-removed blocks do
    /// NOT bump — the per-call NOT per-block invariant from
    /// `KvSpillCounters::record_spill` extends to this sink as well).
    fn record_eviction_budget_overflow(&self);

    /// ADR-017 Phase E option (a) iter-2 — per-request LCP probe
    /// observation.  Bumps `hf2q_kv_lcp_lookups_total` unconditionally
    /// (every post-`PromptCache`-miss probe counts as a lookup) and
    /// `hf2q_kv_lcp_detected_total` when `detected_k.is_some()` (a
    /// non-trivial partial-prefix opportunity exists, i.e. `0 < K <
    /// new_tokens.len()`). The optional `k_value` lets future
    /// histogram-style sinks bucket by LCP length; the production
    /// `KvSpillCounters` impl stores the count of detection events
    /// only.
    ///
    /// Default impl is a no-op so non-production / mock sinks (tests)
    /// can opt out without code churn.
    fn record_lcp_probe(&self, _detected_k: Option<usize>) {}
}

/// Convenience alias used by trigger sites: the optional reference the
/// bump-site receives. `None` ⇒ no-op (tests, lib build).
pub type MetricsSinkRef<'a> = Option<&'a Arc<dyn KvCacheMetricsSink>>;
