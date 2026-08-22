//! Gemma 4 policy and telemetry for the model-neutral anchor state machine.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use super::anchor_store::{PruneResult, Publication, StagePending};

pub(crate) const DEFAULT_MAX_COMMITTED_ANCHORS: usize = 4;

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
    pub configured_slots: AtomicU64,
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
    configured_slots: AtomicU64::new(0),
};

fn raise_peak(peak: &AtomicU64, observed: u64) {
    let mut current = peak.load(Ordering::Relaxed);
    while observed > current {
        match peak.compare_exchange_weak(current, observed, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

pub(crate) fn record_configuration(configured_slots: usize, aggregate_budget_bytes: u64) {
    TELEMETRY
        .configured_slots
        .store(configured_slots as u64, Ordering::Relaxed);
    TELEMETRY
        .aggregate_budget_bytes
        .store(aggregate_budget_bytes, Ordering::Relaxed);
    TELEMETRY
        .peak_committed_pending_bytes
        .store(0, Ordering::Relaxed);
    TELEMETRY
        .aggregate_peak_committed_pending_bytes
        .store(0, Ordering::Relaxed);
}

pub(crate) fn record_capture(
    outcome: StagePending,
    duration: Duration,
    slot_owned_bytes: u64,
    aggregate_owned_bytes: u64,
) {
    TELEMETRY.captures_total.fetch_add(1, Ordering::Relaxed);
    TELEMETRY.capture_nanos_total.fetch_add(
        duration.as_nanos().min(u128::from(u64::MAX)) as u64,
        Ordering::Relaxed,
    );
    if matches!(
        outcome,
        StagePending::BudgetExceeded { .. } | StagePending::NoCommittedCapacity
    ) {
        TELEMETRY
            .capture_budget_skips_total
            .fetch_add(1, Ordering::Relaxed);
    }
    raise_peak(&TELEMETRY.peak_committed_pending_bytes, slot_owned_bytes);
    raise_peak(
        &TELEMETRY.aggregate_peak_committed_pending_bytes,
        aggregate_owned_bytes,
    );
}

pub(crate) fn record_publication(publication: Publication) {
    TELEMETRY
        .evictions_total
        .fetch_add(publication.evicted as u64, Ordering::Relaxed);
}

pub(crate) fn record_restore_hit(tokens_saved: usize, prune: PruneResult) {
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

pub(crate) fn record_restore_miss() {
    TELEMETRY
        .restore_attempts_total
        .fetch_add(1, Ordering::Relaxed);
    TELEMETRY
        .restore_misses_total
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_cancellation(prune: PruneResult) {
    TELEMETRY
        .cancellations_total
        .fetch_add(1, Ordering::Relaxed);
    TELEMETRY
        .descendants_pruned_total
        .fetch_add(prune.pruned as u64, Ordering::Relaxed);
}

pub(crate) fn record_restore_hit_during_cancellation(tokens_saved: usize, prune: PruneResult) {
    TELEMETRY
        .restore_attempts_total
        .fetch_add(1, Ordering::Relaxed);
    TELEMETRY.restore_hits_total.fetch_add(1, Ordering::Relaxed);
    TELEMETRY
        .tokens_saved_total
        .fetch_add(tokens_saved as u64, Ordering::Relaxed);
    record_cancellation(prune);
}

pub(crate) fn record_lineage_clear() {
    TELEMETRY
        .lineage_clears_total
        .fetch_add(1, Ordering::Relaxed);
}
