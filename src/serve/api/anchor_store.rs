//! Model-neutral slot-local checkpoint lineage.
//!
//! Payloads remain family-owned. This module only enforces publication,
//! lineage, positional eviction, cancellation, and exact reclaimable-byte
//! accounting for checkpoints over one mutable physical KV log.

use anyhow::{ensure, Result};

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
/// payloads and the preallocated committed-vector control storage. It must
/// not participate in a scheduler's monotonic device-allocation high-water.
pub(crate) struct AnchorStore<A: AnchorEntry> {
    pub(super) committed: Vec<A>,
    pub(super) pending: Option<A>,
    pub(super) lineage_epoch: u64,
    pub(super) owned_bytes: u64,
}

impl<A: AnchorEntry> AnchorStore<A> {
    pub(crate) fn with_committed_capacity(capacity: usize) -> Self {
        let committed = Vec::with_capacity(capacity);
        let owned_bytes =
            (committed.capacity() as u64).saturating_mul(std::mem::size_of::<A>() as u64);
        Self {
            committed,
            pending: None,
            lineage_epoch: 0,
            owned_bytes,
        }
    }

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

    pub(crate) fn control_owned_bytes(&self) -> u64 {
        self.committed_control_bytes()
    }

    pub(crate) fn committed(&self, index: usize) -> Option<&A> {
        self.committed
            .get(index)
            .filter(|anchor| anchor.lineage_epoch() == self.lineage_epoch)
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
        if max_committed > self.committed.capacity() {
            return StagePending::NoCommittedCapacity;
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
            .as_ref()
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
            publication.replaced_equal_depth = pending.token_count() == last.token_count();
        }
        ensure!(
            max_committed <= self.committed.capacity(),
            "anchor publication K {max_committed} exceeds preflighted committed capacity {}",
            self.committed.capacity()
        );

        // No state has changed above this line. A rejected publication keeps
        // the request-local pending payload intact and byte accounting exact.
        let pending = self.pending.take().expect("pending preflighted");
        if publication.replaced_equal_depth {
            let replaced = self.committed.pop().expect("last existed");
            self.owned_bytes = self.owned_bytes.saturating_sub(replaced.owned_bytes());
        }
        if !publication.replaced_equal_depth && self.committed.len() >= max_committed {
            publication.evicted = self.committed.len() - max_committed + 1;
            for evicted in self.committed.drain(0..publication.evicted) {
                self.owned_bytes = self.owned_bytes.saturating_sub(evicted.owned_bytes());
            }
        }
        ensure!(
            self.committed.len() < self.committed.capacity(),
            "anchor publication has no preflighted committed control slot"
        );
        self.committed.push(pending);
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
        self.owned_bytes = self.committed_control_bytes();
        self.lineage_epoch = self.lineage_epoch.wrapping_add(1);
        result
    }

    fn bump_and_retag_survivors(&mut self) {
        self.lineage_epoch = self.lineage_epoch.wrapping_add(1);
        for anchor in &mut self.committed {
            anchor.set_lineage_epoch(self.lineage_epoch);
        }
    }

    pub(super) fn committed_control_bytes(&self) -> u64 {
        (self.committed.capacity() as u64).saturating_mul(std::mem::size_of::<A>() as u64)
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
            .try_fold(self.committed_control_bytes(), |sum, bytes| {
                sum.checked_add(bytes)
            })
            .ok_or_else(|| anyhow::anyhow!("anchor owned-byte accounting overflow"))?;
        ensure!(
            recomputed == self.owned_bytes,
            "anchor owned-byte accounting mismatch: stored={} recomputed={recomputed}",
            self.owned_bytes
        );
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn committed_token_counts(&self) -> Vec<usize> {
        self.committed
            .iter()
            .map(AnchorEntry::token_count)
            .collect()
    }

    #[cfg(test)]
    pub(super) fn payload_owned_bytes(&self) -> u64 {
        self.owned_bytes
            .saturating_sub(self.committed_control_bytes())
    }
}

/// Maximum committed depth whose retained payloads fit for every configured
/// slot. Pending capture is admitted separately against the live aggregate.
pub(crate) fn effective_committed_depth(
    max_committed: usize,
    aggregate_budget_bytes: u64,
    n_slots: usize,
    anchor_bytes: u64,
) -> usize {
    if max_committed == 0 || n_slots == 0 || anchor_bytes == 0 {
        return 0;
    }
    let per_depth_all_slots = anchor_bytes.saturating_mul(n_slots as u64);
    if per_depth_all_slots == 0 {
        return 0;
    }
    max_committed.min((aggregate_budget_bytes / per_depth_all_slots) as usize)
}

pub(crate) fn simultaneous_pending_capacity_slots(
    aggregate_budget_bytes: u64,
    n_slots: usize,
    anchor_bytes: u64,
    committed_depth: usize,
) -> usize {
    if n_slots == 0 || anchor_bytes == 0 {
        return 0;
    }
    let committed_charge = anchor_bytes
        .saturating_mul(n_slots as u64)
        .saturating_mul(committed_depth as u64);
    let free = aggregate_budget_bytes.saturating_sub(committed_charge);
    n_slots.min((free / anchor_bytes) as usize)
}

#[cfg(test)]
mod tests {
    use super::{AnchorEntry, AnchorStore, StagePending};

    #[derive(Clone)]
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

    fn publish(store: &mut AnchorStore<FakeAnchor>, tokens: &[u32]) {
        assert_eq!(
            store.stage_pending(FakeAnchor::new(tokens, 10), 4, 1_000),
            StagePending::Staged
        );
        store.publish_pending(4).expect("publish");
    }

    #[test]
    fn pending_is_invisible_until_publication() {
        let mut store = AnchorStore::with_committed_capacity(4);
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1, 2], 10), 4, 1_000),
            StagePending::Staged
        );
        assert_eq!(
            store.deepest_matching_index(|anchor| [1, 2, 3].starts_with(&anchor.tokens)),
            None
        );
        store.publish_pending(4).expect("publish");
        assert_eq!(
            store.deepest_matching_index(|anchor| [1, 2, 3].starts_with(&anchor.tokens)),
            Some(0)
        );
    }

    #[test]
    fn rejected_publication_preserves_pending_and_exact_accounting() {
        let mut store = AnchorStore::with_committed_capacity(4);
        publish(&mut store, &[1, 2, 3, 4]);
        let before = store.owned_bytes();
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1, 2], 10), 4, 1_000),
            StagePending::Staged
        );
        let staged = store.owned_bytes();
        assert_eq!(staged, before + 10);
        let error = store
            .publish_pending(4)
            .expect_err("behind-lineage publication must fail atomically");
        assert!(error.to_string().contains("behind newest committed"));
        assert!(store.has_pending());
        assert_eq!(store.owned_bytes(), staged);
        assert_eq!(store.committed_token_counts(), vec![4]);
        store.discard_pending();
        assert_eq!(store.owned_bytes(), before);
        store
            .validate()
            .expect("rejected publication remains valid");
    }

    #[test]
    fn rewind_a_then_write_invalidates_old_b_and_c() {
        let mut store = AnchorStore::with_committed_capacity(4);
        publish(&mut store, &[1, 2]);
        publish(&mut store, &[1, 2, 3]);
        publish(&mut store, &[1, 2, 3, 4]);

        let epoch = store.lineage_epoch();
        let prune = store
            .prune_descendants_after_restore(0)
            .expect("rewind to A");
        assert_eq!(prune.pruned, 2);
        assert_ne!(store.lineage_epoch(), epoch);
        publish(&mut store, &[1, 2, 9]);
        assert_eq!(store.committed_token_counts(), vec![2, 3]);
        assert_eq!(
            store.deepest_matching_index(|anchor| [1, 2, 3, 4].starts_with(&anchor.tokens)),
            Some(0),
            "old B/C must not regain authority after the branch"
        );
    }

    #[test]
    fn cancellation_discards_pending_and_keeps_only_reachable_committed() {
        let mut store = AnchorStore::with_committed_capacity(4);
        publish(&mut store, &[1, 2]);
        publish(&mut store, &[1, 2, 3, 4]);
        publish(&mut store, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1; 8], 10), 4, 1_000),
            StagePending::Staged
        );
        let prune = store.cancel_request_at_cursor(4);
        assert!(prune.pending_discarded);
        assert_eq!(prune.pruned, 1);
        assert_eq!(store.committed_token_counts(), vec![2, 4]);
        store.validate().expect("valid after cancellation");
    }

    #[test]
    fn cancellation_before_publication_preserves_the_pre_request_committed_set() {
        let mut store = AnchorStore::with_committed_capacity(4);
        publish(&mut store, &[1, 2]);
        publish(&mut store, &[1, 2, 3, 4]);
        publish(&mut store, &[1, 2, 3, 4, 5, 6]);
        let committed_before = store.committed_token_counts();
        let bytes_before = store.owned_bytes();
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1; 8], 10), 4, 1_000),
            StagePending::Staged
        );

        let prune = store.cancel_request_at_cursor(6);
        assert!(prune.pending_discarded);
        assert_eq!(prune.pruned, 0);
        assert_eq!(store.committed_token_counts(), committed_before);
        assert_eq!(store.owned_bytes(), bytes_before);
        store.validate().expect("valid after cancellation rollback");
    }

    #[test]
    fn reset_poison_and_failed_restore_clear_all_authority() {
        for _cause in ["reset", "poison", "failed-restore"] {
            let mut store = AnchorStore::with_committed_capacity(4);
            publish(&mut store, &[1, 2]);
            assert_eq!(
                store.stage_pending(FakeAnchor::new(&[1, 2, 3], 10), 4, 1_000),
                StagePending::Staged
            );
            let previous_epoch = store.lineage_epoch();
            let cleared = store.clear_all();
            assert_eq!(cleared.committed, 1);
            assert!(cleared.pending_discarded);
            assert_ne!(store.lineage_epoch(), previous_epoch);
            assert_eq!(store.committed_len(), 0);
            store.validate().expect("valid cleared store");
        }
    }

    #[test]
    fn owned_bytes_are_exact_across_publication_prune_and_clear() {
        let mut store = AnchorStore::with_committed_capacity(4);
        let control = store.control_owned_bytes();
        publish(&mut store, &[1, 2]);
        publish(&mut store, &[1, 2, 3]);
        assert_eq!(store.owned_bytes(), control + 20);
        store
            .prune_descendants_after_restore(0)
            .expect("prune descendant");
        assert_eq!(store.owned_bytes(), control + 10);
        store.clear_all();
        assert_eq!(store.owned_bytes(), control);
    }
}
