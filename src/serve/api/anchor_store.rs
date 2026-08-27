//! Model-neutral slot-local checkpoint lineage.
//!
//! Payloads remain family-owned. This module only enforces publication,
//! lineage, positional eviction, cancellation, and exact reclaimable-byte
//! accounting for checkpoints over one mutable physical KV log.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{ensure, Result};

static ANCHOR_RESTORE_FAILURE_FIRED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AnchorRestoreFaultFamily {
    Gemma4,
    Deepseek4,
}

impl AnchorRestoreFaultFamily {
    fn label(self) -> &'static str {
        match self {
            Self::Gemma4 => "gemma4",
            Self::Deepseek4 => "deepseek4",
        }
    }
}

fn consume_one_shot_restore_failure(
    trigger_max_tokens: Option<usize>,
    request_max_tokens: Option<usize>,
    fired: &AtomicBool,
) -> bool {
    let Some(trigger_max_tokens) = trigger_max_tokens else {
        return false;
    };
    if request_max_tokens != Some(trigger_max_tokens) {
        return false;
    }
    fired
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

/// Inject the centrally parsed ADR-049 restore fault at a real family-owned
/// prompt-anchor restore. Cancellation passes `None` and can never consume the
/// request-selected fault. The family caller must route the returned error
/// through its ordinary fail-closed reset and full-lineage invalidation path.
pub(crate) fn maybe_inject_anchor_restore_failure(
    family: AnchorRestoreFaultFamily,
    request_max_tokens: Option<usize>,
) -> Result<()> {
    if !consume_one_shot_restore_failure(
        crate::debug::INVESTIGATION_ENV.anchor_restore_failure_max_tokens,
        request_max_tokens,
        &ANCHOR_RESTORE_FAILURE_FIRED,
    ) {
        return Ok(());
    }
    tracing::warn!(
        family = family.label(),
        request_max_tokens,
        "injecting one-shot prompt-anchor restore failure"
    );
    anyhow::bail!(
        "injected_{family}_anchor_restore_failure",
        family = family.label()
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AnchorPublicationDisposition {
    Unpublished,
    Added,
    ReplacedEqualDepth,
    PositionalKeepNewestK { evicted: usize },
}

impl AnchorPublicationDisposition {
    pub(crate) fn reason(self) -> &'static str {
        match self {
            Self::Unpublished => "unpublished",
            Self::Added => "none",
            Self::ReplacedEqualDepth => "replace_equal_depth",
            Self::PositionalKeepNewestK { .. } => "positional_keep_newest_k",
        }
    }

    pub(crate) fn evicted_count(self) -> usize {
        match self {
            Self::PositionalKeepNewestK { evicted } => evicted,
            Self::Unpublished | Self::Added | Self::ReplacedEqualDepth => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AnchorRestoreOutcome {
    Hit,
    MissNoMatch,
    RestoreFailedResetSucceeded,
    FailedCleanup,
}

impl AnchorRestoreOutcome {
    fn label(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::MissNoMatch => "miss_no_match",
            Self::RestoreFailedResetSucceeded => "restore_failed_reset_succeeded",
            Self::FailedCleanup => "failed_cleanup",
        }
    }

    pub(crate) fn is_hit(self) -> bool {
        matches!(self, Self::Hit)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct AnchorDivergence {
    pub position: usize,
    pub distance: usize,
}

impl AnchorDivergence {
    pub(crate) fn between(old_live: &[u32], incoming: &[u32]) -> Self {
        let position = old_live
            .iter()
            .zip(incoming.iter())
            .take_while(|(left, right)| left == right)
            .count();
        Self {
            position,
            distance: old_live.len().saturating_sub(position),
        }
    }

    pub(crate) fn rewind(live_cursor: usize, restored_tokens: usize) -> Self {
        Self {
            position: restored_tokens,
            distance: live_cursor.saturating_sub(restored_tokens),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AnchorRestoreEvent {
    pub family: &'static str,
    pub slot: Option<u32>,
    pub cause: &'static str,
    pub outcome: AnchorRestoreOutcome,
    pub attempted_hit_depth: usize,
    pub hit_depth: usize,
    pub divergence: AnchorDivergence,
    pub tokens_saved: usize,
    pub descendant_prune_count: usize,
    pub pending_discarded: bool,
    pub publication_disposition: Option<AnchorPublicationDisposition>,
    pub capture_duration: Duration,
    pub peak_committed_pending_bytes: u64,
}

pub(crate) fn emit_anchor_restore_event(event: AnchorRestoreEvent) {
    let (eviction_reason, publication_evicted_count) = event
        .publication_disposition
        .map(|disposition| (disposition.reason(), disposition.evicted_count()))
        .unwrap_or(("not_applicable", 0));
    tracing::info!(
        target: "hf2q::serve::api::anchor_restore",
        family = event.family,
        slot = ?event.slot,
        cause = event.cause,
        outcome = event.outcome.label(),
        attempted_hit_depth = event.attempted_hit_depth,
        hit_depth = event.hit_depth,
        divergence_position = event.divergence.position,
        divergence_distance = event.divergence.distance,
        tokens_saved = event.tokens_saved,
        descendant_prune_count = event.descendant_prune_count,
        pending_discarded = event.pending_discarded,
        eviction_reason,
        publication_evicted_count,
        capture_ms = event.capture_duration.as_secs_f64() * 1000.0,
        peak_committed_pending_bytes = event.peak_committed_pending_bytes,
        "slot-local anchor restore outcome"
    );
}

/// Payload facts the store needs to enforce lineage and exact ownership.
/// Model tensors remain opaque to the state machine.
pub(crate) trait AnchorEntry {
    fn token_count(&self) -> usize;
    fn lineage_epoch(&self) -> u64;
    fn set_lineage_epoch(&mut self, epoch: u64);
    fn owned_bytes(&self) -> u64;
    fn publication_disposition(&self) -> AnchorPublicationDisposition;
    fn set_publication_disposition(&mut self, disposition: AnchorPublicationDisposition);
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

pub(crate) fn capture_if_anchor_admitted<T>(
    admission: StagePending,
    capture: impl FnOnce() -> Result<T>,
) -> Result<Option<T>> {
    match admission {
        StagePending::Staged => capture().map(Some),
        StagePending::NoCommittedCapacity | StagePending::BudgetExceeded { .. } => Ok(None),
        StagePending::PendingOccupied => {
            anyhow::bail!("anchor capture preflight found an occupied pending payload")
        }
    }
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
    peak_owned_bytes: u64,
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
            peak_owned_bytes: owned_bytes,
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

    /// Lifetime high-water for this one store, including committed-vector
    /// control storage plus committed and request-local pending payloads.
    /// Keeping it on the store makes the value follow the physical slot when
    /// a family swaps slot state through a shared execution surface.
    pub(crate) fn peak_owned_bytes(&self) -> u64 {
        self.peak_owned_bytes
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
    pub(crate) fn preflight_stage_pending(
        &self,
        anchor_owned_bytes: u64,
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
        let needed_bytes = self.owned_bytes.saturating_add(anchor_owned_bytes);
        if needed_bytes > byte_budget {
            return StagePending::BudgetExceeded {
                needed_bytes,
                budget_bytes: byte_budget,
            };
        }
        StagePending::Staged
    }

    /// Capture is request-local until terminal cache+ledger success. The
    /// budget is checked against committed K plus this one pending payload.
    pub(crate) fn stage_pending(
        &mut self,
        mut anchor: A,
        max_committed: usize,
        byte_budget: u64,
    ) -> StagePending {
        let anchor_owned_bytes = anchor.owned_bytes();
        let admission =
            self.preflight_stage_pending(anchor_owned_bytes, max_committed, byte_budget);
        if admission != StagePending::Staged {
            return admission;
        }
        let needed_bytes = self.owned_bytes.saturating_add(anchor_owned_bytes);
        anchor.set_lineage_epoch(self.lineage_epoch);
        self.owned_bytes = needed_bytes;
        self.peak_owned_bytes = self.peak_owned_bytes.max(needed_bytes);
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
        if !publication.replaced_equal_depth && self.committed.len() >= max_committed {
            publication.evicted = self.committed.len() - max_committed + 1;
        }
        let disposition = if publication.replaced_equal_depth {
            AnchorPublicationDisposition::ReplacedEqualDepth
        } else if publication.evicted > 0 {
            AnchorPublicationDisposition::PositionalKeepNewestK {
                evicted: publication.evicted,
            }
        } else {
            AnchorPublicationDisposition::Added
        };
        let mut pending = self.pending.take().expect("pending preflighted");
        pending.set_publication_disposition(disposition);
        if publication.replaced_equal_depth {
            let replaced = self.committed.pop().expect("last existed");
            self.owned_bytes = self.owned_bytes.saturating_sub(replaced.owned_bytes());
        }
        if publication.evicted > 0 {
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
        ensure!(
            self.peak_owned_bytes >= self.owned_bytes,
            "anchor per-store peak {} is below current owned bytes {}",
            self.peak_owned_bytes,
            self.owned_bytes
        );
        Ok(())
    }

    /// Ordered token depths of the epoch-valid committed lineage.
    ///
    /// Production cancellation telemetry uses this compact identity to prove
    /// that a request-local rollback did not publish or discard an anchor. It
    /// contains lengths only; prompt tokens and payload bytes remain private.
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
    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use tracing_subscriber::prelude::*;

    use super::{
        capture_if_anchor_admitted, consume_one_shot_restore_failure, emit_anchor_restore_event,
        AnchorDivergence, AnchorEntry, AnchorPublicationDisposition, AnchorRestoreEvent,
        AnchorRestoreOutcome, AnchorStore, StagePending,
    };

    #[derive(Clone)]
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

    fn publish(store: &mut AnchorStore<FakeAnchor>, tokens: &[u32]) {
        assert_eq!(
            store.stage_pending(FakeAnchor::new(tokens, 10), 4, 1_000),
            StagePending::Staged
        );
        store.publish_pending(4).expect("publish");
    }

    #[test]
    fn pending_preflight_matches_stage_without_mutating_the_store() {
        let mut store = AnchorStore::<FakeAnchor>::with_committed_capacity(4);
        let owned_before = store.owned_bytes();

        assert_eq!(
            store.preflight_stage_pending(10, 4, owned_before + 9),
            StagePending::BudgetExceeded {
                needed_bytes: owned_before + 10,
                budget_bytes: owned_before + 9,
            }
        );
        assert_eq!(store.owned_bytes(), owned_before);
        assert!(!store.has_pending());

        assert_eq!(
            store.preflight_stage_pending(10, 4, owned_before + 10),
            StagePending::Staged
        );
        assert_eq!(store.owned_bytes(), owned_before);
        assert!(!store.has_pending());
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1], 10), 4, owned_before + 10),
            StagePending::Staged
        );
    }

    #[test]
    fn optional_capture_never_calls_the_copy_after_a_capacity_rejection() {
        let mut called = false;
        let captured = capture_if_anchor_admitted::<u64>(
            StagePending::NoCommittedCapacity,
            || {
                called = true;
                Ok(99)
            },
        )
        .unwrap();
        assert_eq!(captured, None);
        assert!(!called, "capacity rejection must happen before payload copying");

        assert!(capture_if_anchor_admitted::<u64>(StagePending::PendingOccupied, || Ok(99))
            .is_err());
    }

    #[test]
    fn restore_failure_selector_requires_exact_request_and_fires_once() {
        let fired = AtomicBool::new(false);
        assert!(!consume_one_shot_restore_failure(None, Some(47), &fired));
        assert!(!consume_one_shot_restore_failure(Some(47), None, &fired));
        assert!(!consume_one_shot_restore_failure(
            Some(47),
            Some(46),
            &fired
        ));
        assert!(consume_one_shot_restore_failure(Some(47), Some(47), &fired));
        assert!(!consume_one_shot_restore_failure(
            Some(47),
            Some(47),
            &fired
        ));
    }

    #[derive(Clone, Default)]
    struct RecordingLayer {
        events: Arc<Mutex<Vec<BTreeMap<String, String>>>>,
    }

    struct FieldVisitor<'a> {
        fields: &'a mut BTreeMap<String, String>,
    }

    impl tracing::field::Visit for FieldVisitor<'_> {
        fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.fields
                .insert(field.name().to_string(), format!("{value:?}"));
        }
    }

    impl<S> tracing_subscriber::Layer<S> for RecordingLayer
    where
        S: tracing::Subscriber,
    {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut fields = BTreeMap::new();
            event.record(&mut FieldVisitor {
                fields: &mut fields,
            });
            if fields.contains_key("family") {
                self.events.lock().expect("events lock").push(fields);
            }
        }
    }

    #[test]
    fn publication_disposition_survives_until_restore() {
        let mut store = AnchorStore::with_committed_capacity(2);
        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1, 2], 10), 2, 1_000),
            StagePending::Staged
        );
        store.publish_pending(2).unwrap();
        assert_eq!(
            store.committed(0).unwrap().publication_disposition(),
            AnchorPublicationDisposition::Added
        );

        assert_eq!(
            store.stage_pending(FakeAnchor::new(&[1, 2], 10), 2, 1_000),
            StagePending::Staged
        );
        store.publish_pending(2).unwrap();
        assert_eq!(
            store.committed(0).unwrap().publication_disposition(),
            AnchorPublicationDisposition::ReplacedEqualDepth
        );

        for tokens in [&[1, 2, 3, 4][..], &[1, 2, 3, 4, 5, 6][..]] {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(tokens, 10), 2, 1_000),
                StagePending::Staged
            );
            store.publish_pending(2).unwrap();
        }
        assert_eq!(store.committed_token_counts(), vec![4, 6]);
        assert_eq!(
            store.committed(1).unwrap().publication_disposition(),
            AnchorPublicationDisposition::PositionalKeepNewestK { evicted: 1 }
        );
    }

    #[test]
    fn anchor_divergence_measures_discarded_live_tail() {
        assert_eq!(
            AnchorDivergence::between(&[1, 2, 3, 4, 9, 10], &[1, 2, 3, 4, 8]),
            AnchorDivergence {
                position: 4,
                distance: 2,
            }
        );
        assert_eq!(
            AnchorDivergence::rewind(10, 6),
            AnchorDivergence {
                position: 6,
                distance: 4,
            }
        );
    }

    #[test]
    fn restore_event_has_fixed_complete_schema_for_every_outcome() {
        let layer = RecordingLayer::default();
        let events = layer.events.clone();
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            for (outcome, attempted_hit_depth, disposition) in [
                (
                    AnchorRestoreOutcome::Hit,
                    3,
                    Some(AnchorPublicationDisposition::PositionalKeepNewestK { evicted: 2 }),
                ),
                (AnchorRestoreOutcome::MissNoMatch, 0, None),
                (
                    AnchorRestoreOutcome::RestoreFailedResetSucceeded,
                    3,
                    Some(AnchorPublicationDisposition::Added),
                ),
                (
                    AnchorRestoreOutcome::FailedCleanup,
                    3,
                    Some(AnchorPublicationDisposition::Added),
                ),
                (AnchorRestoreOutcome::FailedCleanup, 0, None),
            ] {
                emit_anchor_restore_event(AnchorRestoreEvent {
                    family: "test-family",
                    slot: Some(7),
                    cause: "schema-test",
                    outcome,
                    attempted_hit_depth,
                    hit_depth: usize::from(outcome.is_hit()) * 3,
                    divergence: AnchorDivergence {
                        position: 11,
                        distance: 7,
                    },
                    tokens_saved: usize::from(outcome.is_hit()) * 11,
                    descendant_prune_count: usize::from(outcome.is_hit()),
                    pending_discarded: outcome.is_hit(),
                    publication_disposition: disposition,
                    capture_duration: if attempted_hit_depth > 0 {
                        Duration::from_millis(5)
                    } else {
                        Duration::ZERO
                    },
                    peak_committed_pending_bytes: 99,
                });
            }
        });

        let captured = events.lock().expect("events lock").clone();
        assert_eq!(captured.len(), 5);
        for event in &captured {
            for field in [
                "family",
                "slot",
                "cause",
                "outcome",
                "attempted_hit_depth",
                "hit_depth",
                "divergence_position",
                "divergence_distance",
                "tokens_saved",
                "descendant_prune_count",
                "pending_discarded",
                "eviction_reason",
                "publication_evicted_count",
                "capture_ms",
                "peak_committed_pending_bytes",
            ] {
                assert!(event.contains_key(field), "restore event omitted {field}");
            }
        }
        assert_eq!(captured[0]["outcome"], "hit");
        assert_eq!(captured[0]["attempted_hit_depth"], "3");
        assert_eq!(captured[0]["hit_depth"], "3");
        assert_eq!(captured[1]["outcome"], "miss_no_match");
        assert_eq!(captured[1]["attempted_hit_depth"], "0");
        assert_eq!(captured[1]["hit_depth"], "0");
        assert_eq!(captured[2]["outcome"], "restore_failed_reset_succeeded");
        assert_eq!(captured[2]["hit_depth"], "0");
        assert_eq!(captured[3]["outcome"], "failed_cleanup");
        assert_eq!(captured[3]["attempted_hit_depth"], "3");
        assert_eq!(captured[4]["outcome"], "failed_cleanup");
        assert_eq!(captured[4]["attempted_hit_depth"], "0");
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

    #[test]
    fn peak_owned_bytes_is_lifetime_and_store_local() {
        let mut slot_zero = AnchorStore::with_committed_capacity(4);
        let mut slot_one = AnchorStore::with_committed_capacity(4);
        let control = slot_zero.control_owned_bytes();
        assert_eq!(slot_zero.peak_owned_bytes(), control);
        assert_eq!(slot_one.peak_owned_bytes(), control);

        publish(&mut slot_zero, &[1, 2]);
        assert_eq!(slot_zero.peak_owned_bytes(), control + 10);
        assert_eq!(slot_one.peak_owned_bytes(), control);

        assert_eq!(
            slot_one.stage_pending(FakeAnchor::new(&[1, 2, 3], 30), 4, 1_000),
            StagePending::Staged
        );
        assert_eq!(slot_one.peak_owned_bytes(), control + 30);
        slot_one.discard_pending();
        slot_zero.clear_all();

        assert_eq!(slot_zero.owned_bytes(), control);
        assert_eq!(slot_zero.peak_owned_bytes(), control + 10);
        assert_eq!(slot_one.owned_bytes(), control);
        assert_eq!(slot_one.peak_owned_bytes(), control + 30);
    }
}
