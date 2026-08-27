use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

pub(crate) use super::anchor_store::{
    effective_committed_depth, simultaneous_pending_capacity_slots, StagePending,
};
use super::anchor_store::{
    emit_anchor_restore_event, AnchorEntry, AnchorRestoreEvent, AnchorStore,
};

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
    pub peak_committed_pending_bytes: AtomicU64,
    pub aggregate_peak_committed_pending_bytes: AtomicU64,
    pub aggregate_budget_bytes: AtomicU64,
    pub configured_slots: AtomicU64,
    pub effective_committed_depth: AtomicU64,
    pub simultaneous_pending_capacity_slots: AtomicU64,
    pub prefill_transaction_ceiling_tokens: AtomicU64,
    pub spec_boundary_capable: AtomicU64,
    pub partial_capacity_captures_total: AtomicU64,
    pub spec_boundary_restore_tokens_total: AtomicU64,
    pub post_admission_prefill_failures_total: AtomicU64,
    pub stable_boundary_compound_prefills_total: AtomicU64,
    pub rectangular_prefill_cohorts_total: AtomicU64,
    pub cohort_staging_invariant_failures_total: AtomicU64,
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
    peak_committed_pending_bytes: AtomicU64::new(0),
    aggregate_peak_committed_pending_bytes: AtomicU64::new(0),
    aggregate_budget_bytes: AtomicU64::new(0),
    configured_slots: AtomicU64::new(0),
    effective_committed_depth: AtomicU64::new(0),
    simultaneous_pending_capacity_slots: AtomicU64::new(0),
    prefill_transaction_ceiling_tokens: AtomicU64::new(0),
    spec_boundary_capable: AtomicU64::new(0),
    partial_capacity_captures_total: AtomicU64::new(0),
    spec_boundary_restore_tokens_total: AtomicU64::new(0),
    post_admission_prefill_failures_total: AtomicU64::new(0),
    stable_boundary_compound_prefills_total: AtomicU64::new(0),
    rectangular_prefill_cohorts_total: AtomicU64::new(0),
    cohort_staging_invariant_failures_total: AtomicU64::new(0),
};

/// One-shot acceptance fault for ADR-049's post-admission recovery gate.
///
/// The trigger is immutable for the worker lifetime. It can fire only after
/// a matching request completed a non-empty GPU prefill slice, so equality
/// hits and pre-admission validation failures cannot satisfy the gate by
/// accident. Normal serving constructs this with `None` unless the centrally
/// parsed, unsafe-acknowledged investigation variable is active.
#[derive(Debug, Default)]
pub(crate) struct PostAdmissionPrefillFailure {
    trigger_max_tokens: Option<usize>,
    fired: bool,
}

impl PostAdmissionPrefillFailure {
    pub(crate) fn new(trigger_max_tokens: Option<usize>) -> Self {
        Self {
            trigger_max_tokens,
            fired: false,
        }
    }

    pub(crate) fn should_fail(
        &mut self,
        request_max_tokens: usize,
        advanced_tokens: usize,
    ) -> bool {
        if self.fired || advanced_tokens == 0 {
            return false;
        }
        if self.trigger_max_tokens != Some(request_max_tokens) {
            return false;
        }
        self.fired = true;
        true
    }
}

pub(crate) fn record_post_admission_prefill_failure() {
    TELEMETRY
        .post_admission_prefill_failures_total
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_stable_boundary_compound_prefill() {
    TELEMETRY
        .stable_boundary_compound_prefills_total
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_rectangular_prefill_cohort() {
    TELEMETRY
        .rectangular_prefill_cohorts_total
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_cohort_staging_invariant_failure() {
    TELEMETRY
        .cohort_staging_invariant_failures_total
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_configuration(
    configured_slots: usize,
    aggregate_budget_bytes: u64,
    prefill_transaction_ceiling_tokens: u32,
    spec_boundary_capable: bool,
) {
    TELEMETRY
        .configured_slots
        .store(configured_slots as u64, Ordering::Relaxed);
    TELEMETRY
        .aggregate_budget_bytes
        .store(aggregate_budget_bytes, Ordering::Relaxed);
    TELEMETRY.prefill_transaction_ceiling_tokens.store(
        u64::from(prefill_transaction_ceiling_tokens),
        Ordering::Relaxed,
    );
    TELEMETRY
        .spec_boundary_capable
        .store(u64::from(spec_boundary_capable), Ordering::Relaxed);
    TELEMETRY
        .peak_committed_pending_bytes
        .store(0, Ordering::Relaxed);
    TELEMETRY
        .aggregate_peak_committed_pending_bytes
        .store(0, Ordering::Relaxed);
    TELEMETRY
        .effective_committed_depth
        .store(0, Ordering::Relaxed);
    TELEMETRY
        .simultaneous_pending_capacity_slots
        .store(0, Ordering::Relaxed);
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

pub(crate) fn record_capture(
    outcome: StagePending,
    duration: Duration,
    slot_peak_bytes: u64,
    aggregate_owned_bytes: u64,
    effective_committed_depth: usize,
    simultaneous_pending_capacity_slots: usize,
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
    if has_partial_capacity(
        effective_committed_depth,
        simultaneous_pending_capacity_slots,
        TELEMETRY.configured_slots.load(Ordering::Relaxed) as usize,
    ) {
        TELEMETRY
            .partial_capacity_captures_total
            .fetch_add(1, Ordering::Relaxed);
    }
    TELEMETRY
        .effective_committed_depth
        .store(effective_committed_depth as u64, Ordering::Relaxed);
    TELEMETRY.simultaneous_pending_capacity_slots.store(
        simultaneous_pending_capacity_slots as u64,
        Ordering::Relaxed,
    );
    raise_peak(&TELEMETRY.peak_committed_pending_bytes, slot_peak_bytes);
    raise_peak(
        &TELEMETRY.aggregate_peak_committed_pending_bytes,
        aggregate_owned_bytes,
    );
}

fn has_partial_capacity(
    effective_committed_depth: usize,
    simultaneous_pending_capacity_slots: usize,
    configured_slots: usize,
) -> bool {
    effective_committed_depth < DEFAULT_MAX_COMMITTED_ANCHORS
        || simultaneous_pending_capacity_slots < configured_slots
}

pub(crate) fn record_committed_budget_skips(
    skipped: usize,
    aggregate_owned_bytes: u64,
    effective_committed_depth: usize,
    simultaneous_pending_capacity_slots: usize,
    configured_slots: usize,
) {
    let skipped = skipped as u64;
    TELEMETRY
        .captures_total
        .fetch_add(skipped, Ordering::Relaxed);
    TELEMETRY
        .capture_budget_skips_total
        .fetch_add(skipped, Ordering::Relaxed);
    if has_partial_capacity(
        effective_committed_depth,
        simultaneous_pending_capacity_slots,
        configured_slots,
    ) {
        TELEMETRY
            .partial_capacity_captures_total
            .fetch_add(skipped, Ordering::Relaxed);
    }
    TELEMETRY
        .effective_committed_depth
        .store(effective_committed_depth as u64, Ordering::Relaxed);
    TELEMETRY.simultaneous_pending_capacity_slots.store(
        simultaneous_pending_capacity_slots as u64,
        Ordering::Relaxed,
    );
    raise_peak(
        &TELEMETRY.aggregate_peak_committed_pending_bytes,
        aggregate_owned_bytes,
    );
}

pub(crate) fn discard_cohort_pending<A: AnchorEntry>(
    stores: &mut [AnchorStore<A>],
    slot_indices: &[usize],
) -> anyhow::Result<usize> {
    let mut discarded = 0;
    for &slot_idx in slot_indices {
        let store = stores
            .get_mut(slot_idx)
            .ok_or_else(|| anyhow::anyhow!("Qwen cohort anchor slot {slot_idx} missing"))?;
        discarded += usize::from(store.discard_pending());
        store.validate()?;
    }
    Ok(discarded)
}

pub(crate) fn record_restore(event: AnchorRestoreEvent) {
    TELEMETRY
        .restore_attempts_total
        .fetch_add(1, Ordering::Relaxed);
    if event.outcome.is_hit() {
        TELEMETRY.restore_hits_total.fetch_add(1, Ordering::Relaxed);
    } else {
        TELEMETRY
            .restore_misses_total
            .fetch_add(1, Ordering::Relaxed);
    }
    TELEMETRY
        .tokens_saved_total
        .fetch_add(event.tokens_saved as u64, Ordering::Relaxed);
    TELEMETRY
        .descendants_pruned_total
        .fetch_add(event.descendant_prune_count as u64, Ordering::Relaxed);
    emit_anchor_restore_event(event);
}

pub(crate) fn record_spec_boundary_restore(tokens: usize) {
    TELEMETRY
        .spec_boundary_restore_tokens_total
        .fetch_add(tokens as u64, Ordering::Relaxed);
}

pub(crate) fn record_evictions(evicted: usize) {
    TELEMETRY
        .evictions_total
        .fetch_add(evicted as u64, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::super::anchor_store::{AnchorEntry, AnchorPublicationDisposition, AnchorStore};
    use super::{
        effective_committed_depth, has_partial_capacity, simultaneous_pending_capacity_slots,
        PostAdmissionPrefillFailure, StagePending, DEFAULT_MAX_COMMITTED_ANCHORS,
    };

    fn default_store<A: AnchorEntry>() -> AnchorStore<A> {
        AnchorStore::with_committed_capacity(DEFAULT_MAX_COMMITTED_ANCHORS)
    }

    #[derive(Default)]
    struct ReferenceStore {
        committed: Vec<(usize, u64)>,
        pending: Option<(usize, u64)>,
        lineage_epoch: u64,
    }

    impl ReferenceStore {
        fn stage(&mut self, depth: usize, bytes: u64) {
            assert!(self.pending.replace((depth, bytes)).is_none());
        }

        fn publish(&mut self, keep: usize) {
            let pending = self.pending.take().expect("reference pending");
            if self
                .committed
                .last()
                .is_some_and(|last| last.0 == pending.0)
            {
                self.committed.pop();
            }
            self.committed.push(pending);
            let evict = self.committed.len().saturating_sub(keep);
            self.committed.drain(0..evict);
        }

        fn restore_then_branch(&mut self, index: usize) {
            self.pending = None;
            self.committed.truncate(index + 1);
            self.lineage_epoch = self.lineage_epoch.wrapping_add(1);
        }

        fn cancel(&mut self, live_cursor: usize) {
            self.pending = None;
            let keep = self
                .committed
                .partition_point(|(depth, _)| *depth <= live_cursor);
            if keep < self.committed.len() {
                self.committed.truncate(keep);
                self.lineage_epoch = self.lineage_epoch.wrapping_add(1);
            }
        }

        fn clear(&mut self) {
            self.committed.clear();
            self.pending = None;
            self.lineage_epoch = self.lineage_epoch.wrapping_add(1);
        }

        fn owned_bytes(&self) -> u64 {
            self.committed
                .iter()
                .map(|(_, bytes)| bytes)
                .chain(self.pending.iter().map(|(_, bytes)| bytes))
                .sum()
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct FakeAnchor {
        tokens: Vec<u32>,
        epoch: u64,
        bytes: u64,
        publication_disposition: AnchorPublicationDisposition,
    }

    impl FakeAnchor {
        fn new(tokens: &[u32], bytes: u64) -> Self {
            Self {
                tokens: tokens.to_vec(),
                epoch: u64::MAX,
                bytes,
                publication_disposition: AnchorPublicationDisposition::Unpublished,
            }
        }
    }

    impl AnchorEntry for FakeAnchor {
        fn token_count(&self) -> usize {
            self.tokens.len()
        }

        fn lineage_epoch(&self) -> u64 {
            self.epoch
        }

        fn set_lineage_epoch(&mut self, epoch: u64) {
            self.epoch = epoch;
        }

        fn owned_bytes(&self) -> u64 {
            self.bytes
        }

        fn publication_disposition(&self) -> AnchorPublicationDisposition {
            self.publication_disposition
        }

        fn set_publication_disposition(&mut self, disposition: AnchorPublicationDisposition) {
            self.publication_disposition = disposition;
        }
    }

    fn deepest(store: &AnchorStore<FakeAnchor>, prompt: &[u32]) -> Option<usize> {
        store.deepest_matching_index(|anchor| prompt.starts_with(&anchor.tokens))
    }

    fn assert_matches_reference(store: &AnchorStore<FakeAnchor>, reference: &ReferenceStore) {
        store.validate().expect("production state must be valid");
        assert_eq!(
            store
                .committed
                .iter()
                .map(|anchor| (anchor.token_count(), anchor.owned_bytes()))
                .collect::<Vec<_>>(),
            reference.committed
        );
        assert_eq!(
            store
                .pending
                .as_ref()
                .map(|anchor| (anchor.token_count(), anchor.owned_bytes())),
            reference.pending
        );
        assert_eq!(store.lineage_epoch(), reference.lineage_epoch);
        assert_eq!(store.payload_owned_bytes(), reference.owned_bytes());
    }

    #[test]
    fn partial_capacity_metric_matches_its_depth_or_pending_contract() {
        assert!(!has_partial_capacity(4, 16, 16));
        assert!(has_partial_capacity(3, 16, 16));
        assert!(has_partial_capacity(4, 15, 16));
    }

    #[test]
    fn state_machine_matches_independent_reference_model() {
        let mut store = default_store();
        let mut reference = ReferenceStore::default();

        for (depth, bytes) in [(2, 11), (4, 13), (6, 17), (8, 19), (10, 23)] {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&vec![1; depth], bytes), 4, 1_000),
                StagePending::Staged
            );
            reference.stage(depth, bytes);
            assert_matches_reference(&store, &reference);
            store.publish_pending(4).expect("production publication");
            reference.publish(4);
            assert_matches_reference(&store, &reference);
        }

        store
            .prune_descendants_after_restore(1)
            .expect("production rewind");
        reference.restore_then_branch(1);
        assert_matches_reference(&store, &reference);

        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1; 11], 29), 4, 1_000),
            StagePending::Staged
        );
        reference.stage(11, 29);
        store
            .publish_pending(4)
            .expect("production divergent branch");
        reference.publish(4);
        assert_matches_reference(&store, &reference);

        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1; 12], 31), 4, 1_000),
            StagePending::Staged
        );
        reference.stage(12, 31);
        store.cancel_request_at_cursor(8);
        reference.cancel(8);
        assert_matches_reference(&store, &reference);

        store.clear_all();
        reference.clear();
        assert_matches_reference(&store, &reference);
    }

    #[test]
    fn pending_is_invisible_until_terminal_publication() {
        let mut store = default_store();
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1, 2], 20), 4, 1_000),
            StagePending::Staged
        );
        assert_eq!(deepest(&store, &[1, 2, 3]), None);
        assert_eq!(store.committed_len(), 0);
        assert_eq!(store.pending_bytes(), 20);

        let publication = store.publish_pending(4).expect("publish");
        assert_eq!(publication.evicted, 0);
        assert_eq!(deepest(&store, &[1, 2, 3]), Some(0));
        assert_eq!(store.committed_len(), 1);
        assert_eq!(store.pending_bytes(), 0);
        store.validate().expect("valid store");
    }

    #[test]
    fn aggregate_budget_reduces_depth_for_wide_qwen38_without_overcommit() {
        const MIB: u64 = 1024 * 1024;
        let budget = 4_608 * MIB;
        let qwen38_anchor = 151 * MIB;
        let qwen36_anchor = 64 * MIB;

        assert_eq!(effective_committed_depth(4, budget, 4, qwen38_anchor), 4);
        assert_eq!(effective_committed_depth(4, budget, 8, qwen38_anchor), 3);
        assert_eq!(effective_committed_depth(4, budget, 16, qwen38_anchor), 1);
        assert_eq!(effective_committed_depth(4, budget, 16, qwen36_anchor), 4);

        let n16_qwen38_charge = qwen38_anchor
            .checked_mul(16)
            .expect("scaled aggregate charge");
        assert!(n16_qwen38_charge <= budget);
        assert_eq!(
            simultaneous_pending_capacity_slots(budget, 16, qwen38_anchor, 1),
            14
        );
        assert_eq!(
            simultaneous_pending_capacity_slots(budget, 8, qwen38_anchor, 3),
            6
        );
        assert_eq!(
            simultaneous_pending_capacity_slots(budget, 16, qwen36_anchor, 4),
            8
        );
    }

    #[test]
    fn aggregate_budget_depth_fails_closed_for_invalid_or_pending_only_capacity() {
        assert_eq!(effective_committed_depth(4, 1_000, 0, 10), 0);
        assert_eq!(effective_committed_depth(4, 1_000, 4, 0), 0);
        assert_eq!(effective_committed_depth(0, 1_000, 4, 10), 0);
        assert_eq!(effective_committed_depth(4, 10, 4, 10), 0);
    }

    #[test]
    fn aggregate_preflight_makes_n16_partial_pending_availability_explicit() {
        let mut stores: Vec<AnchorStore<FakeAnchor>> = (0..16)
            .map(|_| AnchorStore::with_committed_capacity(4))
            .collect();
        let control_bytes: u64 = stores.iter().map(AnchorStore::owned_bytes).sum();
        let budget = control_bytes + 4_608;

        for (slot, store) in stores.iter_mut().enumerate() {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&vec![1; slot + 1], 151), 1, budget),
                StagePending::Staged
            );
            store.publish_pending(1).expect("publish N16 baseline");
        }

        let mut staged = 0usize;
        let mut skipped = 0usize;
        for slot in 0..stores.len() {
            let aggregate_before: u64 = stores.iter().map(AnchorStore::owned_bytes).sum();
            let other_owned = aggregate_before - stores[slot].owned_bytes();
            let slot_budget = budget.saturating_sub(other_owned);
            match stores[slot].stage_pending(
                FakeAnchor::new(&vec![1; slot + 17], 151),
                1,
                slot_budget,
            ) {
                StagePending::Staged => staged += 1,
                StagePending::BudgetExceeded { .. } => skipped += 1,
                outcome => panic!("unexpected N16 stage outcome: {outcome:?}"),
            }
        }
        let aggregate_after: u64 = stores.iter().map(AnchorStore::owned_bytes).sum();
        assert_eq!(staged, 14);
        assert_eq!(skipped, 2);
        assert!(aggregate_after <= budget);
    }

    #[test]
    fn preflight_charges_k_committed_plus_one_pending() {
        let mut store = AnchorStore::with_committed_capacity(4);
        let budget = store.owned_bytes() + 50;
        for depth in 1..=4 {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&vec![7; depth], 10), 4, budget),
                StagePending::Staged
            );
            store.publish_pending(4).expect("publish");
        }
        assert_eq!(store.committed_bytes(), 40);
        let owned_before_failed_preflight = store.owned_bytes();
        let control_before_failed_preflight = store.control_owned_bytes();
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[7; 5], 11), 4, budget),
            StagePending::BudgetExceeded {
                needed_bytes: store.committed_control_bytes() + 51,
                budget_bytes: budget,
            }
        );
        assert_eq!(store.committed_len(), 4);
        assert!(!store.has_pending());
        assert_eq!(store.owned_bytes(), owned_before_failed_preflight);
        assert_eq!(store.control_owned_bytes(), control_before_failed_preflight);
        assert_eq!(store.payload_owned_bytes(), 40);
        store.validate().expect("valid store");
    }

    #[test]
    fn zero_control_capacity_skips_without_allocating_or_exceeding_tiny_grant() {
        let mut store = AnchorStore::with_committed_capacity(0);
        assert_eq!(store.owned_bytes(), 0);
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1], 10), 1, 1),
            StagePending::NoCommittedCapacity
        );
        assert_eq!(store.owned_bytes(), 0);
        assert_eq!(store.control_owned_bytes(), 0);
        assert!(!store.has_pending());
        store.validate().expect("zero-capacity store remains valid");
    }

    #[test]
    fn positional_eviction_keeps_newest_k_not_recently_restored() {
        let mut store = default_store();
        for depth in 1..=5 {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&vec![3; depth], 10), 4, 1_000),
                StagePending::Staged
            );
            store.publish_pending(4).expect("publish");
        }
        assert_eq!(store.committed_token_counts(), vec![2, 3, 4, 5]);
        assert_eq!(store.payload_owned_bytes(), 40);
        store.validate().expect("valid store");
    }

    #[test]
    fn rewind_then_write_invalidates_descendants_and_old_c_cannot_restore() {
        let mut store = default_store();
        for tokens in [&[1, 2][..], &[1, 2, 3][..], &[1, 2, 3, 4][..]] {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(tokens, 10), 4, 1_000),
                StagePending::Staged
            );
            store.publish_pending(4).expect("publish");
        }
        let old_epoch = store.lineage_epoch();
        assert_eq!(deepest(&store, &[1, 2, 9]), Some(0));
        let prune = store
            .prune_descendants_after_restore(0)
            .expect("restore pruning");
        assert_eq!(prune.pruned, 2);
        assert_ne!(store.lineage_epoch(), old_epoch);
        assert_eq!(store.committed_token_counts(), vec![2]);
        assert_eq!(deepest(&store, &[1, 2, 3, 4, 5]), Some(0));

        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1, 2, 9], 10), 4, 1_000),
            StagePending::Staged
        );
        store.publish_pending(4).expect("publish branch X");
        assert_eq!(store.committed_token_counts(), vec![2, 3]);
        assert_eq!(deepest(&store, &[1, 2, 3, 4, 5]), Some(0));
        store.validate().expect("valid store");
    }

    #[test]
    fn cancellation_discards_pending_and_prunes_only_unreachable_committed() {
        let mut store = default_store();
        for depth in [2, 4, 6] {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&vec![1; depth], 10), 4, 1_000),
                StagePending::Staged
            );
            store.publish_pending(4).expect("publish");
        }
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1; 8], 10), 4, 1_000),
            StagePending::Staged
        );
        let result = store.cancel_request_at_cursor(4);
        assert!(result.pending_discarded);
        assert_eq!(result.pruned, 1);
        assert_eq!(store.committed_token_counts(), vec![2, 4]);
        assert_eq!(store.payload_owned_bytes(), 20);
        store.validate().expect("valid store");
    }

    #[test]
    fn reset_poison_or_failed_restore_clears_the_whole_store() {
        for _cause in ["reset", "poison", "restore-failure"] {
            let mut store = default_store();
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&[1, 2], 10), 4, 1_000),
                StagePending::Staged
            );
            store.publish_pending(4).expect("publish");
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&[1, 2, 3], 10), 4, 1_000),
                StagePending::Staged
            );
            let prior_epoch = store.lineage_epoch();
            let cleared = store.clear_all();
            assert_eq!(cleared.committed, 1);
            assert!(cleared.pending_discarded);
            assert_ne!(store.lineage_epoch(), prior_epoch);
            assert_eq!(store.payload_owned_bytes(), 0);
            assert_eq!(deepest(&store, &[1, 2, 3]), None);
            store.validate().expect("valid store");
        }
    }

    fn store_with_pending() -> AnchorStore<FakeAnchor> {
        let mut store = default_store();
        for depth in [2, 4, 6] {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&vec![1; depth], 10), 4, 1_000),
                StagePending::Staged
            );
            store.publish_pending(4).expect("publish fixture anchor");
        }
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1; 8], 10), 4, 1_000),
            StagePending::Staged
        );
        store.validate().expect("reference fixture is valid");
        store
    }

    #[test]
    fn invariant_battery_detects_all_seventeen_injected_mutations() {
        let mut caught = 0usize;
        for mutation in 0..17 {
            let mut store = store_with_pending();
            match mutation {
                0 => store.committed[0].epoch = 99,
                1 => store.committed[2].epoch = 99,
                2 => store.pending.as_mut().unwrap().epoch = 99,
                3 => store.lineage_epoch = 99,
                4 => store.owned_bytes += 1,
                5 => store.owned_bytes -= 1,
                6 => store.committed.swap(0, 1),
                7 => store.committed[1].tokens = store.committed[0].tokens.clone(),
                8 => store.committed.reverse(),
                9 => store.committed[0].tokens.clear(),
                10 => store.pending.as_mut().unwrap().tokens.clear(),
                11 => store.pending.as_mut().unwrap().tokens = vec![1; 3],
                12 => {
                    store.pending = None;
                }
                13 => {
                    store.committed.pop();
                }
                14 => store.committed[0].bytes += 7,
                15 => store.pending.as_mut().unwrap().bytes += 7,
                16 => store.committed.push(FakeAnchor {
                    tokens: vec![1; 9],
                    epoch: store.lineage_epoch,
                    bytes: 10,
                    publication_disposition: AnchorPublicationDisposition::Unpublished,
                }),
                _ => unreachable!(),
            }
            if store.validate().is_err() {
                caught += 1;
            }
        }
        assert_eq!(caught, 17, "every injected state mutation must fail closed");
    }

    #[test]
    fn post_admission_prefill_fault_requires_a_real_matching_slice_and_fires_once() {
        let mut fault = PostAdmissionPrefillFailure::new(Some(39));

        assert!(!fault.should_fail(39, 0), "an equality hit did no GPU work");
        assert!(
            !fault.should_fail(38, 128),
            "a different request must not consume the one-shot fault"
        );
        assert!(fault.should_fail(39, 128), "the matching GPU slice fires");
        assert!(
            !fault.should_fail(39, 128),
            "the process must recover after exactly one injected request failure"
        );
    }

    #[test]
    fn disabled_post_admission_prefill_fault_cannot_change_serving() {
        let mut fault = PostAdmissionPrefillFailure::new(None);
        for max_tokens in [1, 39, usize::MAX] {
            assert!(!fault.should_fail(max_tokens, 2_048));
        }
    }
}
