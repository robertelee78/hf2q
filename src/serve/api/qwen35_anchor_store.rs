use anyhow::{ensure, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

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

pub(crate) fn record_capture(outcome: StagePending, duration: Duration, peak_bytes: u64) {
    TELEMETRY.captures_total.fetch_add(1, Ordering::Relaxed);
    TELEMETRY.capture_nanos_total.fetch_add(
        duration.as_nanos().min(u128::from(u64::MAX)) as u64,
        Ordering::Relaxed,
    );
    if matches!(outcome, StagePending::BudgetExceeded { .. }) {
        TELEMETRY
            .capture_budget_skips_total
            .fetch_add(1, Ordering::Relaxed);
    }
    raise_peak(&TELEMETRY.peak_committed_pending_bytes, peak_bytes);
}

pub(crate) fn record_restore_hit(tokens_saved: usize, descendants_pruned: usize) {
    TELEMETRY
        .restore_attempts_total
        .fetch_add(1, Ordering::Relaxed);
    TELEMETRY.restore_hits_total.fetch_add(1, Ordering::Relaxed);
    TELEMETRY
        .tokens_saved_total
        .fetch_add(tokens_saved as u64, Ordering::Relaxed);
    TELEMETRY
        .descendants_pruned_total
        .fetch_add(descendants_pruned as u64, Ordering::Relaxed);
}

pub(crate) fn record_restore_miss() {
    TELEMETRY
        .restore_attempts_total
        .fetch_add(1, Ordering::Relaxed);
    TELEMETRY
        .restore_misses_total
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_evictions(evicted: usize) {
    TELEMETRY
        .evictions_total
        .fetch_add(evicted as u64, Ordering::Relaxed);
}

/// Payload facts the store needs to enforce lineage and exact ownership.
/// Model tensors remain opaque to the state machine.
pub(crate) trait AnchorEntry {
    fn token_count(&self) -> usize;
    fn lineage_epoch(&self) -> u64;
    fn set_lineage_epoch(&mut self, epoch: u64);
    fn owned_bytes(&self) -> u64;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StagePending {
    Staged,
    PendingOccupied,
    NoCommittedCapacity,
    BudgetExceeded {
        needed_bytes: u64,
        budget_bytes: u64,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Publication {
    pub evicted: usize,
    pub replaced_equal_depth: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PruneResult {
    pub pruned: usize,
    pub pending_discarded: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ClearResult {
    pub committed: usize,
    pub pending_discarded: bool,
}

/// Slot-local checkpoints over one mutable physical KV lineage.
///
/// `owned_bytes` is the exact reclaimable sum of committed plus pending
/// payloads. It intentionally does not participate in the scheduler's
/// monotonic Metal high-water counter.
pub(crate) struct AnchorStore<A: AnchorEntry> {
    committed: Vec<A>,
    pending: Option<A>,
    lineage_epoch: u64,
    owned_bytes: u64,
}

impl<A: AnchorEntry> Default for AnchorStore<A> {
    fn default() -> Self {
        Self {
            committed: Vec::new(),
            pending: None,
            lineage_epoch: 0,
            owned_bytes: 0,
        }
    }
}

impl<A: AnchorEntry> AnchorStore<A> {
    pub(crate) fn committed_len(&self) -> usize {
        self.committed.len()
    }

    pub(crate) fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub(crate) fn pending(&self) -> Option<&A> {
        self.pending
            .as_ref()
            .filter(|anchor| anchor.lineage_epoch() == self.lineage_epoch)
    }

    pub(crate) fn lineage_epoch(&self) -> u64 {
        self.lineage_epoch
    }

    pub(crate) fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub(crate) fn committed_bytes(&self) -> u64 {
        self.committed.iter().map(AnchorEntry::owned_bytes).sum()
    }

    pub(crate) fn pending_bytes(&self) -> u64 {
        self.pending
            .as_ref()
            .map(AnchorEntry::owned_bytes)
            .unwrap_or(0)
    }

    pub(crate) fn committed(&self, index: usize) -> Option<&A> {
        self.committed
            .get(index)
            .filter(|anchor| anchor.lineage_epoch() == self.lineage_epoch)
    }

    pub(crate) fn committed_mut(&mut self, index: usize) -> Option<&mut A> {
        let epoch = self.lineage_epoch;
        self.committed
            .get_mut(index)
            .filter(|anchor| anchor.lineage_epoch() == epoch)
    }

    pub(crate) fn newest_committed_at_or_before(&self, token_count: usize) -> Option<usize> {
        self.committed
            .iter()
            .enumerate()
            .rev()
            .find(|(_, anchor)| {
                anchor.lineage_epoch() == self.lineage_epoch && anchor.token_count() <= token_count
            })
            .map(|(index, _)| index)
    }

    pub(crate) fn deepest_matching_index(
        &self,
        mut matches: impl FnMut(&A) -> bool,
    ) -> Option<usize> {
        self.committed
            .iter()
            .enumerate()
            .rev()
            .find(|(_, anchor)| anchor.lineage_epoch() == self.lineage_epoch && matches(anchor))
            .map(|(index, _)| index)
    }

    /// Capture is request-local until terminal cache+ledger success. The
    /// budget is checked against committed K plus this one pending payload.
    pub(crate) fn stage_pending(
        &mut self,
        mut anchor: A,
        max_committed: usize,
        byte_budget: u64,
    ) -> StagePending {
        if max_committed == 0 {
            return StagePending::NoCommittedCapacity;
        }
        if self.pending.is_some() {
            return StagePending::PendingOccupied;
        }
        let needed_bytes = self.owned_bytes.saturating_add(anchor.owned_bytes());
        if needed_bytes > byte_budget {
            return StagePending::BudgetExceeded {
                needed_bytes,
                budget_bytes: byte_budget,
            };
        }
        anchor.set_lineage_epoch(self.lineage_epoch);
        self.owned_bytes = needed_bytes;
        self.pending = Some(anchor);
        StagePending::Staged
    }

    /// Atomically expose the request-local capture, then apply positional
    /// keep-newest-K eviction. Equal-depth publication replaces that boundary.
    pub(crate) fn publish_pending(&mut self, max_committed: usize) -> Result<Publication> {
        ensure!(max_committed > 0, "anchor publication requires K > 0");
        let pending = self
            .pending
            .take()
            .ok_or_else(|| anyhow::anyhow!("anchor publication requires pending payload"))?;
        ensure!(
            pending.lineage_epoch() == self.lineage_epoch,
            "pending anchor epoch {} != live epoch {}",
            pending.lineage_epoch(),
            self.lineage_epoch
        );

        let mut publication = Publication::default();
        if let Some(last) = self.committed.last() {
            ensure!(
                pending.token_count() >= last.token_count(),
                "pending anchor depth {} is behind newest committed depth {}",
                pending.token_count(),
                last.token_count()
            );
            if pending.token_count() == last.token_count() {
                let replaced = self.committed.pop().expect("last existed");
                self.owned_bytes = self.owned_bytes.saturating_sub(replaced.owned_bytes());
                publication.replaced_equal_depth = true;
            }
        }
        self.committed.push(pending);

        if self.committed.len() > max_committed {
            publication.evicted = self.committed.len() - max_committed;
            for evicted in self.committed.drain(0..publication.evicted) {
                self.owned_bytes = self.owned_bytes.saturating_sub(evicted.owned_bytes());
            }
        }
        self.validate()?;
        Ok(publication)
    }

    pub(crate) fn discard_pending(&mut self) -> bool {
        let Some(pending) = self.pending.take() else {
            return false;
        };
        self.owned_bytes = self.owned_bytes.saturating_sub(pending.owned_bytes());
        true
    }

    /// A successful restore selects one valid ancestor. Before any new KV
    /// write, all descendants and request-local pending state are invalidated,
    /// then surviving ancestors are retagged onto the new physical lineage.
    pub(crate) fn prune_descendants_after_restore(
        &mut self,
        restored_index: usize,
    ) -> Result<PruneResult> {
        let restored = self
            .committed
            .get(restored_index)
            .ok_or_else(|| anyhow::anyhow!("restored anchor index {restored_index} missing"))?;
        ensure!(
            restored.lineage_epoch() == self.lineage_epoch,
            "restored anchor epoch {} != live epoch {}",
            restored.lineage_epoch(),
            self.lineage_epoch
        );
        let pending_discarded = self.discard_pending();
        let pruned = self.committed.len().saturating_sub(restored_index + 1);
        for descendant in self.committed.drain(restored_index + 1..) {
            self.owned_bytes = self.owned_bytes.saturating_sub(descendant.owned_bytes());
        }
        self.bump_and_retag_survivors();
        self.validate()?;
        Ok(PruneResult {
            pruned,
            pending_discarded,
        })
    }

    /// Cancellation may preserve only checkpoints whose physical rows still
    /// exist at the recovered cursor. Request-local pending never publishes.
    pub(crate) fn cancel_request_at_cursor(&mut self, live_cursor: usize) -> PruneResult {
        let pending_discarded = self.discard_pending();
        let keep = self
            .committed
            .partition_point(|anchor| anchor.token_count() <= live_cursor);
        let pruned = self.committed.len().saturating_sub(keep);
        for anchor in self.committed.drain(keep..) {
            self.owned_bytes = self.owned_bytes.saturating_sub(anchor.owned_bytes());
        }
        if pruned > 0 {
            self.bump_and_retag_survivors();
        }
        PruneResult {
            pruned,
            pending_discarded,
        }
    }

    /// Cold reset, poison, or any failed restore destroys all authority for
    /// the slot's old mutable KV lineage.
    pub(crate) fn clear_all(&mut self) -> ClearResult {
        let result = ClearResult {
            committed: self.committed.len(),
            pending_discarded: self.pending.is_some(),
        };
        self.committed.clear();
        self.pending = None;
        self.owned_bytes = 0;
        self.lineage_epoch = self.lineage_epoch.wrapping_add(1);
        result
    }

    fn bump_and_retag_survivors(&mut self) {
        self.lineage_epoch = self.lineage_epoch.wrapping_add(1);
        for anchor in &mut self.committed {
            anchor.set_lineage_epoch(self.lineage_epoch);
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.committed
                .iter()
                .all(|anchor| anchor.lineage_epoch() == self.lineage_epoch),
            "committed anchor carries a stale lineage epoch"
        );
        ensure!(
            self.pending
                .as_ref()
                .is_none_or(|anchor| anchor.lineage_epoch() == self.lineage_epoch),
            "pending anchor carries a stale lineage epoch"
        );
        ensure!(
            self.committed
                .windows(2)
                .all(|pair| pair[0].token_count() < pair[1].token_count()),
            "committed anchors are not a strict positional chain"
        );
        ensure!(
            self.committed.iter().all(|anchor| anchor.token_count() > 0)
                && self
                    .pending
                    .as_ref()
                    .is_none_or(|anchor| anchor.token_count() > 0),
            "anchor token depth must be non-zero"
        );
        ensure!(
            self.pending.as_ref().is_none_or(|pending| {
                self.committed
                    .last()
                    .is_none_or(|last| pending.token_count() >= last.token_count())
            }),
            "pending anchor is behind the newest committed depth"
        );
        let recomputed = self
            .committed
            .iter()
            .map(AnchorEntry::owned_bytes)
            .chain(self.pending.iter().map(AnchorEntry::owned_bytes))
            .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
            .ok_or_else(|| anyhow::anyhow!("anchor owned-byte accounting overflow"))?;
        ensure!(
            recomputed == self.owned_bytes,
            "anchor owned-byte accounting mismatch: stored={} recomputed={recomputed}",
            self.owned_bytes
        );
        Ok(())
    }

    #[cfg(test)]
    fn committed_token_counts(&self) -> Vec<usize> {
        self.committed
            .iter()
            .map(AnchorEntry::token_count)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{AnchorEntry, AnchorStore, StagePending};

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
    }

    impl FakeAnchor {
        fn new(tokens: &[u32], bytes: u64) -> Self {
            Self {
                tokens: tokens.to_vec(),
                epoch: u64::MAX,
                bytes,
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
        assert_eq!(store.owned_bytes(), reference.owned_bytes());
    }

    #[test]
    fn state_machine_matches_independent_reference_model() {
        let mut store = AnchorStore::default();
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
        let mut store = AnchorStore::default();
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1, 2], 20), 4, 100),
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
    fn preflight_charges_k_committed_plus_one_pending() {
        let mut store = AnchorStore::default();
        for depth in 1..=4 {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&vec![7; depth], 10), 4, 50),
                StagePending::Staged
            );
            store.publish_pending(4).expect("publish");
        }
        assert_eq!(store.committed_bytes(), 40);
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[7; 5], 11), 4, 50),
            StagePending::BudgetExceeded {
                needed_bytes: 51,
                budget_bytes: 50,
            }
        );
        assert_eq!(store.committed_len(), 4);
        assert!(!store.has_pending());
        assert_eq!(store.owned_bytes(), 40);
        store.validate().expect("valid store");
    }

    #[test]
    fn positional_eviction_keeps_newest_k_not_recently_restored() {
        let mut store = AnchorStore::default();
        for depth in 1..=5 {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&vec![3; depth], 10), 4, 100),
                StagePending::Staged
            );
            store.publish_pending(4).expect("publish");
        }
        assert_eq!(store.committed_token_counts(), vec![2, 3, 4, 5]);
        assert_eq!(store.owned_bytes(), 40);
        store.validate().expect("valid store");
    }

    #[test]
    fn rewind_then_write_invalidates_descendants_and_old_c_cannot_restore() {
        let mut store = AnchorStore::default();
        for tokens in [&[1, 2][..], &[1, 2, 3][..], &[1, 2, 3, 4][..]] {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(tokens, 10), 4, 100),
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
            store.stage_pending(FakeAnchor::new(&[1, 2, 9], 10), 4, 100),
            StagePending::Staged
        );
        store.publish_pending(4).expect("publish branch X");
        assert_eq!(store.committed_token_counts(), vec![2, 3]);
        assert_eq!(deepest(&store, &[1, 2, 3, 4, 5]), Some(0));
        store.validate().expect("valid store");
    }

    #[test]
    fn cancellation_discards_pending_and_prunes_only_unreachable_committed() {
        let mut store = AnchorStore::default();
        for depth in [2, 4, 6] {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&vec![1; depth], 10), 4, 100),
                StagePending::Staged
            );
            store.publish_pending(4).expect("publish");
        }
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1; 8], 10), 4, 100),
            StagePending::Staged
        );
        let result = store.cancel_request_at_cursor(4);
        assert!(result.pending_discarded);
        assert_eq!(result.pruned, 1);
        assert_eq!(store.committed_token_counts(), vec![2, 4]);
        assert_eq!(store.owned_bytes(), 20);
        store.validate().expect("valid store");
    }

    #[test]
    fn reset_poison_or_failed_restore_clears_the_whole_store() {
        for _cause in ["reset", "poison", "restore-failure"] {
            let mut store = AnchorStore::default();
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&[1, 2], 10), 4, 100),
                StagePending::Staged
            );
            store.publish_pending(4).expect("publish");
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&[1, 2, 3], 10), 4, 100),
                StagePending::Staged
            );
            let prior_epoch = store.lineage_epoch();
            let cleared = store.clear_all();
            assert_eq!(cleared.committed, 1);
            assert!(cleared.pending_discarded);
            assert_ne!(store.lineage_epoch(), prior_epoch);
            assert_eq!(store.owned_bytes(), 0);
            assert_eq!(deepest(&store, &[1, 2, 3]), None);
            store.validate().expect("valid store");
        }
    }

    fn store_with_pending() -> AnchorStore<FakeAnchor> {
        let mut store = AnchorStore::default();
        for depth in [2, 4, 6] {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&vec![1; depth], 10), 4, 100),
                StagePending::Staged
            );
            store.publish_pending(4).expect("publish fixture anchor");
        }
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1; 8], 10), 4, 100),
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
                }),
                _ => unreachable!(),
            }
            if store.validate().is_err() {
                caught += 1;
            }
        }
        assert_eq!(caught, 17, "every injected state mutation must fail closed");
    }
}
