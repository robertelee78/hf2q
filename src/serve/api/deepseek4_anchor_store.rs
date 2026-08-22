//! DeepSeek-V4 payload, budget, and telemetry for the model-neutral anchor store.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{ensure, Context, Result};

use crate::inference::models::deepseek4::cache::Deepseek4CacheSnapshot;

use super::anchor_store::{
    effective_committed_depth, simultaneous_pending_capacity_slots, AnchorEntry, AnchorStore,
    PruneResult, Publication, StagePending,
};

pub(super) const DEFAULT_MAX_COMMITTED_ANCHORS: usize = 4;
pub(super) type Deepseek4AnchorStore = AnchorStore<Deepseek4PromptAnchor>;

pub(super) struct Deepseek4PromptAnchor {
    pub prompt_tokens: Box<[u32]>,
    pub snapshot: Deepseek4CacheSnapshot,
    pub capture_duration: Duration,
    lineage_epoch: u64,
}

impl Deepseek4PromptAnchor {
    pub(super) fn new(
        prompt_tokens: &[u32],
        snapshot: Deepseek4CacheSnapshot,
        capture_duration: Duration,
    ) -> Self {
        Self {
            prompt_tokens: prompt_tokens.into(),
            snapshot,
            capture_duration,
            lineage_epoch: u64::MAX,
        }
    }
}

impl AnchorEntry for Deepseek4PromptAnchor {
    fn token_count(&self) -> usize {
        self.prompt_tokens.len()
    }

    fn lineage_epoch(&self) -> u64 {
        self.lineage_epoch
    }

    fn set_lineage_epoch(&mut self, epoch: u64) {
        self.lineage_epoch = epoch;
    }

    fn owned_bytes(&self) -> u64 {
        self.snapshot.owned_bytes().saturating_add(
            (self.prompt_tokens.len() as u64).saturating_mul(std::mem::size_of::<u32>() as u64),
        )
    }
}

/// One immutable aggregate grant shared by a SlotAware worker's stores.
/// Stores are mutated by one worker thread, while atomics keep telemetry and
/// conservation checks independent of swap order.
pub(super) struct Deepseek4AnchorBudget {
    aggregate_budget_bytes: u64,
    control_owned_bytes: u64,
    configured_slots: usize,
    owned_bytes: AtomicU64,
}

impl Deepseek4AnchorBudget {
    pub(super) fn new(
        configured_slots: usize,
        aggregate_budget_bytes: u64,
        initial_owned_bytes: u64,
    ) -> Result<Arc<Self>> {
        ensure!(
            initial_owned_bytes <= aggregate_budget_bytes,
            "DeepSeek-V4 anchor control bytes {initial_owned_bytes} exceed aggregate grant {aggregate_budget_bytes}"
        );
        record_configuration(
            configured_slots,
            aggregate_budget_bytes,
            initial_owned_bytes,
        );
        Ok(Arc::new(Self {
            aggregate_budget_bytes,
            control_owned_bytes: initial_owned_bytes,
            configured_slots,
            owned_bytes: AtomicU64::new(initial_owned_bytes),
        }))
    }

    pub(super) fn aggregate_budget_bytes(&self) -> u64 {
        self.aggregate_budget_bytes
    }

    pub(super) fn owned_bytes(&self) -> u64 {
        self.owned_bytes.load(Ordering::Relaxed)
    }

    fn slot_budget(&self, slot_owned_bytes: u64) -> u64 {
        let other_owned = self.owned_bytes().saturating_sub(slot_owned_bytes);
        self.aggregate_budget_bytes.saturating_sub(other_owned)
    }

    fn effective_depth(&self, anchor_bytes: u64) -> usize {
        effective_committed_depth(
            DEFAULT_MAX_COMMITTED_ANCHORS,
            self.aggregate_budget_bytes
                .saturating_sub(self.control_owned_bytes),
            self.configured_slots,
            anchor_bytes,
        )
    }

    fn pending_capacity(&self, anchor_bytes: u64, committed_depth: usize) -> usize {
        simultaneous_pending_capacity_slots(
            self.aggregate_budget_bytes
                .saturating_sub(self.control_owned_bytes),
            self.configured_slots,
            anchor_bytes,
            committed_depth,
        )
    }

    fn replace_store_charge(&self, before: u64, after: u64) -> Result<u64> {
        let aggregate_before = self.owned_bytes();
        ensure!(
            aggregate_before >= before,
            "DeepSeek-V4 aggregate anchor charge {aggregate_before} is below slot charge {before}"
        );
        let aggregate_after = aggregate_before
            .checked_sub(before)
            .and_then(|value| value.checked_add(after))
            .context("DeepSeek-V4 aggregate anchor charge overflow")?;
        ensure!(
            aggregate_after <= self.aggregate_budget_bytes,
            "DeepSeek-V4 aggregate anchor charge {aggregate_after} exceeds grant {}",
            self.aggregate_budget_bytes
        );
        self.owned_bytes.store(aggregate_after, Ordering::Relaxed);
        TELEMETRY
            .aggregate_owned_bytes
            .store(aggregate_after, Ordering::Relaxed);
        raise_peak(
            &TELEMETRY.aggregate_peak_committed_pending_bytes,
            aggregate_after,
        );
        Ok(aggregate_after)
    }
}

pub(super) fn stage_pending(
    store: &mut Deepseek4AnchorStore,
    budget: &Deepseek4AnchorBudget,
    anchor: Deepseek4PromptAnchor,
    capture_source: &'static str,
) -> Result<StagePending> {
    let before = store.owned_bytes();
    let anchor_bytes = anchor.owned_bytes();
    let capture_duration = anchor.capture_duration;
    let prompt_tokens = anchor.prompt_tokens.len();
    let effective_depth = budget.effective_depth(anchor_bytes);
    let pending_capacity = budget.pending_capacity(anchor_bytes, effective_depth);
    let outcome = store.stage_pending(
        anchor,
        effective_depth,
        budget.slot_budget(store.owned_bytes()),
    );
    let aggregate_after = budget.replace_store_charge(before, store.owned_bytes())?;
    record_capture(
        outcome,
        capture_duration,
        store.owned_bytes(),
        aggregate_after,
        effective_depth,
        pending_capacity,
    );
    tracing::info!(
        target: "hf2q::serve::api::deepseek4_anchor",
        prompt_tokens,
        anchor_bytes,
        capture_ms = capture_duration.as_secs_f64() * 1000.0,
        aggregate_owned_bytes = aggregate_after,
        aggregate_budget_bytes = budget.aggregate_budget_bytes,
        effective_committed_depth = effective_depth,
        simultaneous_pending_capacity_slots = pending_capacity,
        outcome = ?outcome,
        capture_source,
        "DeepSeek-V4 slot-local boundary capture aggregate-budget preflight"
    );
    Ok(outcome)
}

pub(super) fn publish_pending(
    store: &mut Deepseek4AnchorStore,
    budget: &Deepseek4AnchorBudget,
) -> Result<Option<Publication>> {
    if !store.has_pending() {
        return Ok(None);
    }
    let pending_bytes = store.pending_bytes();
    let effective_depth = budget.effective_depth(pending_bytes);
    let before = store.owned_bytes();
    let publication = match store.publish_pending(effective_depth) {
        Ok(publication) => publication,
        Err(error) => {
            store.clear_all();
            budget.replace_store_charge(before, store.owned_bytes())?;
            record_lineage_clear();
            return Err(error.context("publish DeepSeek-V4 pending anchor"));
        }
    };
    budget.replace_store_charge(before, store.owned_bytes())?;
    record_publication(publication);
    Ok(Some(publication))
}

pub(super) fn discard_pending(
    store: &mut Deepseek4AnchorStore,
    budget: &Deepseek4AnchorBudget,
) -> Result<bool> {
    let before = store.owned_bytes();
    let discarded = store.discard_pending();
    budget.replace_store_charge(before, store.owned_bytes())?;
    Ok(discarded)
}

pub(super) fn prune_after_restore(
    store: &mut Deepseek4AnchorStore,
    budget: &Deepseek4AnchorBudget,
    restored_index: usize,
) -> Result<PruneResult> {
    let before = store.owned_bytes();
    let pruned = store.prune_descendants_after_restore(restored_index)?;
    budget.replace_store_charge(before, store.owned_bytes())?;
    Ok(pruned)
}

pub(super) fn cancel_at_cursor(
    store: &mut Deepseek4AnchorStore,
    budget: &Deepseek4AnchorBudget,
    cursor: usize,
) -> Result<PruneResult> {
    let before = store.owned_bytes();
    let pruned = store.cancel_request_at_cursor(cursor);
    budget.replace_store_charge(before, store.owned_bytes())?;
    record_cancellation(pruned);
    Ok(pruned)
}

pub(super) fn clear_all(
    store: &mut Deepseek4AnchorStore,
    budget: &Deepseek4AnchorBudget,
    cause: &'static str,
) -> Result<()> {
    let before = store.owned_bytes();
    let cleared = store.clear_all();
    let aggregate_after = budget.replace_store_charge(before, store.owned_bytes())?;
    record_lineage_clear();
    tracing::info!(
        target: "hf2q::serve::api::deepseek4_anchor",
        cleared_committed = cleared.committed,
        pending_discarded = cleared.pending_discarded,
        aggregate_owned_bytes = aggregate_after,
        cause,
        "DeepSeek-V4 slot-local anchor lineage cleared"
    );
    Ok(())
}

pub(super) fn record_restore_hit(tokens_saved: usize, prune: PruneResult) {
    TELEMETRY
        .restore_attempts_total
        .fetch_add(1, Ordering::Relaxed);
    TELEMETRY.restore_hits_total.fetch_add(1, Ordering::Relaxed);
    TELEMETRY
        .tokens_saved_total
        .fetch_add(tokens_saved as u64, Ordering::Relaxed);
    TELEMETRY
        .descendants_pruned_total
        .fetch_add(prune.pruned as u64, Ordering::Relaxed);
}

pub(super) fn record_restore_miss() {
    TELEMETRY
        .restore_attempts_total
        .fetch_add(1, Ordering::Relaxed);
    TELEMETRY
        .restore_misses_total
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) struct AnchorTelemetry {
    pub captures_total: AtomicU64,
    pub capture_budget_skips_total: AtomicU64,
    pub capture_nanos_total: AtomicU64,
    pub restore_attempts_total: AtomicU64,
    pub restore_hits_total: AtomicU64,
    pub restore_misses_total: AtomicU64,
    pub tokens_saved_total: AtomicU64,
    pub descendants_pruned_total: AtomicU64,
    pub evictions_total: AtomicU64,
    pub cancellations_total: AtomicU64,
    pub lineage_clears_total: AtomicU64,
    pub peak_committed_pending_bytes: AtomicU64,
    pub aggregate_peak_committed_pending_bytes: AtomicU64,
    pub aggregate_budget_bytes: AtomicU64,
    pub aggregate_owned_bytes: AtomicU64,
    pub configured_slots: AtomicU64,
    pub effective_committed_depth: AtomicU64,
    pub simultaneous_pending_capacity_slots: AtomicU64,
}

pub(crate) static TELEMETRY: AnchorTelemetry = AnchorTelemetry {
    captures_total: AtomicU64::new(0),
    capture_budget_skips_total: AtomicU64::new(0),
    capture_nanos_total: AtomicU64::new(0),
    restore_attempts_total: AtomicU64::new(0),
    restore_hits_total: AtomicU64::new(0),
    restore_misses_total: AtomicU64::new(0),
    tokens_saved_total: AtomicU64::new(0),
    descendants_pruned_total: AtomicU64::new(0),
    evictions_total: AtomicU64::new(0),
    cancellations_total: AtomicU64::new(0),
    lineage_clears_total: AtomicU64::new(0),
    peak_committed_pending_bytes: AtomicU64::new(0),
    aggregate_peak_committed_pending_bytes: AtomicU64::new(0),
    aggregate_budget_bytes: AtomicU64::new(0),
    aggregate_owned_bytes: AtomicU64::new(0),
    configured_slots: AtomicU64::new(0),
    effective_committed_depth: AtomicU64::new(0),
    simultaneous_pending_capacity_slots: AtomicU64::new(0),
};

fn record_configuration(configured_slots: usize, budget: u64, initial_owned: u64) {
    TELEMETRY
        .configured_slots
        .store(configured_slots as u64, Ordering::Relaxed);
    TELEMETRY
        .aggregate_budget_bytes
        .store(budget, Ordering::Relaxed);
    TELEMETRY
        .aggregate_owned_bytes
        .store(initial_owned, Ordering::Relaxed);
    TELEMETRY
        .peak_committed_pending_bytes
        .store(0, Ordering::Relaxed);
    TELEMETRY
        .aggregate_peak_committed_pending_bytes
        .store(initial_owned, Ordering::Relaxed);
}

fn record_capture(
    outcome: StagePending,
    duration: Duration,
    slot_owned_bytes: u64,
    aggregate_owned_bytes: u64,
    effective_depth: usize,
    pending_capacity: usize,
) {
    TELEMETRY.captures_total.fetch_add(1, Ordering::Relaxed);
    TELEMETRY.capture_nanos_total.fetch_add(
        duration.as_nanos().min(u128::from(u64::MAX)) as u64,
        Ordering::Relaxed,
    );
    if matches!(
        outcome,
        StagePending::NoCommittedCapacity | StagePending::BudgetExceeded { .. }
    ) {
        TELEMETRY
            .capture_budget_skips_total
            .fetch_add(1, Ordering::Relaxed);
    }
    TELEMETRY
        .aggregate_owned_bytes
        .store(aggregate_owned_bytes, Ordering::Relaxed);
    TELEMETRY
        .effective_committed_depth
        .store(effective_depth as u64, Ordering::Relaxed);
    TELEMETRY
        .simultaneous_pending_capacity_slots
        .store(pending_capacity as u64, Ordering::Relaxed);
    raise_peak(&TELEMETRY.peak_committed_pending_bytes, slot_owned_bytes);
}

fn record_publication(publication: Publication) {
    TELEMETRY
        .evictions_total
        .fetch_add(publication.evicted as u64, Ordering::Relaxed);
}

fn record_cancellation(prune: PruneResult) {
    TELEMETRY
        .cancellations_total
        .fetch_add(1, Ordering::Relaxed);
    TELEMETRY
        .descendants_pruned_total
        .fetch_add(prune.pruned as u64, Ordering::Relaxed);
}

fn record_lineage_clear() {
    TELEMETRY
        .lineage_clears_total
        .fetch_add(1, Ordering::Relaxed);
}

fn raise_peak(peak: &AtomicU64, observed: u64) {
    let mut current = peak.load(Ordering::Relaxed);
    while observed > current {
        match peak.compare_exchange_weak(current, observed, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

#[cfg(test)]
pub(super) fn assert_budget_conservation(
    budget: &Deepseek4AnchorBudget,
    stores: impl IntoIterator<Item = u64>,
) {
    assert_eq!(stores.into_iter().sum::<u64>(), budget.owned_bytes());
    assert!(budget.owned_bytes() <= budget.aggregate_budget_bytes());
}
