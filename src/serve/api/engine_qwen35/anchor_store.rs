use anyhow::{anyhow, Result};
use mlx_native::{MlxBuffer, MlxDevice};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    LazyLock, Mutex,
};
use std::time::Duration;

use crate::inference::models::qwen35::kv_cache::HybridKvSlotAnchor;

use super::Qwen35SpecPrefixBoundary;

#[cfg(test)]
#[path = "anchor_store/c1_differential_tests.rs"]
mod c1_differential_tests;

#[derive(Clone)]
pub(in crate::serve::api) struct Qwen35Anchor {
    pub id: u64,
    pub request: u64,
    pub prompt_tokens: Vec<u32>,
    pub kv: HybridKvSlotAnchor,
    pub prefill_logits: Vec<f32>,
    pub vision_fingerprint: Option<[u8; 32]>,
    pub spec: Option<Qwen35SpecPrefixBoundary>,
}

impl Qwen35Anchor {
    pub fn owned(&self) -> AnchorOwnedBytes {
        AnchorOwnedBytes {
            payload_image: self.kv.total_bytes(),
            logits: self
                .prefill_logits
                .len()
                .saturating_mul(std::mem::size_of::<f32>()),
            hidden: self
                .spec
                .as_ref()
                .map_or(0, |spec| spec.pending_target_hidden.data_byte_len()),
            tokens: self
                .prompt_tokens
                .len()
                .saturating_mul(std::mem::size_of::<u32>()),
            metadata: std::mem::size_of::<Self>(),
        }
    }

    pub fn owned_bytes(&self) -> usize {
        self.owned().total()
    }
}

/// Copy a speculative hidden-row view into a right-sized owner before an
/// anchor retains it. MTP prefill and verify return row views into multi-row
/// outputs; retaining such a view would pin the whole transient allocation
/// and make the anchor's byte accounting false.
pub(in crate::serve::api) fn owned_spec_boundary(
    device: &MlxDevice,
    token_count: usize,
    hidden: &MlxBuffer,
) -> Result<Qwen35SpecPrefixBoundary> {
    let pending_target_hidden = if hidden.byte_len() == hidden.data_byte_len() {
        hidden.clone()
    } else {
        let mut owned = device
            .alloc_buffer(
                hidden.data_byte_len(),
                hidden.dtype(),
                hidden.shape().to_vec(),
            )
            .map_err(|error| anyhow!("allocate Qwen anchor speculative hidden: {error}"))?;
        owned
            .as_mut_slice::<u8>()
            .map_err(|error| anyhow!("map Qwen anchor speculative hidden destination: {error}"))?
            .copy_from_slice(
                hidden.as_slice::<u8>().map_err(|error| {
                    anyhow!("map Qwen anchor speculative hidden source: {error}")
                })?,
            );
        owned
    };
    Ok(Qwen35SpecPrefixBoundary {
        token_count,
        pending_target_hidden,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::serve::api) struct AnchorOwnedBytes {
    pub payload_image: usize,
    pub logits: usize,
    pub hidden: usize,
    pub tokens: usize,
    pub metadata: usize,
}

impl AnchorOwnedBytes {
    pub fn total(self) -> usize {
        self.payload_image
            .saturating_add(self.logits)
            .saturating_add(self.hidden)
            .saturating_add(self.tokens)
            .saturating_add(self.metadata)
    }

    fn add(&mut self, add: Self) {
        self.payload_image = self.payload_image.saturating_add(add.payload_image);
        self.logits = self.logits.saturating_add(add.logits);
        self.hidden = self.hidden.saturating_add(add.hidden);
        self.tokens = self.tokens.saturating_add(add.tokens);
        self.metadata = self.metadata.saturating_add(add.metadata);
    }
}

fn incremental_owned(anchor: &Qwen35Anchor, earlier: &[Qwen35Anchor]) -> AnchorOwnedBytes {
    let mut owned = anchor.owned();
    if earlier.iter().any(|old| anchor.kv.shares_payload_owner_with(&old.kv)) {
        owned.payload_image = 0;
    }
    if let Some(spec) = &anchor.spec {
        let hidden = &spec.pending_target_hidden;
        if earlier.iter().filter_map(|old| old.spec.as_ref()).any(|old| {
            let previous = &old.pending_target_hidden;
            match (hidden.root_origin_snapshot(), previous.root_origin_snapshot()) {
                (mlx_native::RootOriginSnapshot::KnownNative { id: left, .. },
                 mlx_native::RootOriginSnapshot::KnownNative { id: right, .. }) => left == right,
                _ => std::ptr::eq(&**hidden.metal_buffer(), &**previous.metal_buffer()),
            }
        }) {
            owned.hidden = 0;
        }
    }
    owned
}

pub(in crate::serve::api) fn anchor_non_payload_bytes(
    token_count: usize,
    logits_count: usize,
    hidden_bytes: usize,
) -> Result<usize> {
    token_count
        .checked_mul(std::mem::size_of::<u32>())
        .and_then(|tokens| {
            logits_count
                .checked_mul(std::mem::size_of::<f32>())
                .and_then(|logits| tokens.checked_add(logits))
        })
        .and_then(|bytes| bytes.checked_add(hidden_bytes))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Qwen35Anchor>()))
        .ok_or_else(|| anyhow!("Qwen anchor owned-byte estimate overflow"))
}

pub(in crate::serve::api) fn record_anchor_capture_skipped_quota() {
    ANCHOR_TELEMETRY
        .capture_skipped_quota
        .fetch_add(1, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::serve::api) struct Qwen35AnchorGauge {
    pub slot: usize,
    pub committed_count: usize,
    pub committed: AnchorOwnedBytes,
    pub pending: AnchorOwnedBytes,
}

static ANCHOR_GAUGES: LazyLock<Mutex<Vec<Qwen35AnchorGauge>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

pub(in crate::serve::api) fn anchor_gauges() -> Vec<Qwen35AnchorGauge> {
    ANCHOR_GAUGES
        .lock()
        .expect("Qwen anchor gauge lock poisoned")
        .clone()
}

pub(in crate::serve::api) struct Qwen35AnchorTelemetry {
    pub captures_staged: AtomicU64,
    pub capture_skipped_oversized_owner: AtomicU64,
    pub capture_skipped_quota: AtomicU64,
    pub committed: AtomicU64,
    pub equality_hits: AtomicU64,
    pub restore_refused: AtomicU64,
    pub restore_refusals_by_reason: [AtomicU64; 5],
    pub tombstoned_cursor: AtomicU64,
    pub tombstoned_cold_reset: AtomicU64,
    pub evicted: AtomicU64,
    pub restores_by_rank: [AtomicU64; 8],
    pub snapshot_count: AtomicU64,
    pub snapshot_nanos: AtomicU64,
    pub restore_count: AtomicU64,
    pub restore_nanos: AtomicU64,
}

pub(in crate::serve::api) static ANCHOR_TELEMETRY: Qwen35AnchorTelemetry = Qwen35AnchorTelemetry {
    captures_staged: AtomicU64::new(0),
    capture_skipped_oversized_owner: AtomicU64::new(0),
    capture_skipped_quota: AtomicU64::new(0),
    committed: AtomicU64::new(0),
    equality_hits: AtomicU64::new(0),
    restore_refused: AtomicU64::new(0),
    restore_refusals_by_reason: [const { AtomicU64::new(0) }; 5],
    tombstoned_cursor: AtomicU64::new(0),
    tombstoned_cold_reset: AtomicU64::new(0),
    evicted: AtomicU64::new(0),
    restores_by_rank: [const { AtomicU64::new(0) }; 8],
    snapshot_count: AtomicU64::new(0),
    snapshot_nanos: AtomicU64::new(0),
    restore_count: AtomicU64::new(0),
    restore_nanos: AtomicU64::new(0),
};

pub(in crate::serve::api) fn record_anchor_snapshot(elapsed: Duration) {
    ANCHOR_TELEMETRY
        .snapshot_count
        .fetch_add(1, Ordering::Relaxed);
    ANCHOR_TELEMETRY.snapshot_nanos.fetch_add(
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

pub(in crate::serve::api) fn record_anchor_restore(elapsed: Duration) {
    ANCHOR_TELEMETRY
        .restore_count
        .fetch_add(1, Ordering::Relaxed);
    ANCHOR_TELEMETRY.restore_nanos.fetch_add(
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

pub(in crate::serve::api) fn record_anchor_restore_refused() {
    ANCHOR_TELEMETRY
        .restore_refused
        .fetch_add(1, Ordering::Relaxed);
    ANCHOR_TELEMETRY.restore_refusals_by_reason[AnchorRestoreRefusal::Validation as usize]
        .fetch_add(1, Ordering::Relaxed);
}

pub(in crate::serve::api) fn record_anchor_equality_hit() {
    ANCHOR_TELEMETRY
        .equality_hits
        .fetch_add(1, Ordering::Relaxed);
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::serve::api) struct AnchorTicket {
    origin: u64,
    epoch: u64,
    anchor_id: u64,
    operation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::serve::api) enum AnchorRestoreRefusal {
    Poisoned,
    Transaction,
    Unavailable,
    Stale,
    Validation,
}

impl AnchorRestoreRefusal {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Poisoned => "poisoned",
            Self::Transaction => "transaction",
            Self::Unavailable => "unavailable",
            Self::Stale => "stale",
            Self::Validation => "validation",
        }
    }

    fn counted(self) -> anyhow::Error {
        ANCHOR_TELEMETRY.restore_refused.fetch_add(1, Ordering::Relaxed);
        ANCHOR_TELEMETRY.restore_refusals_by_reason[self as usize].fetch_add(1, Ordering::Relaxed);
        self.into()
    }
}

impl std::fmt::Display for AnchorRestoreRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Qwen anchor restore refused: {}", self.label())
    }
}
impl std::error::Error for AnchorRestoreRefusal {}

/// Keep construction tickets live across prefill's possible boundary snapshots.
/// The owning worker is synchronous; no caller can publish a pending anchor
/// until every ticket in this scope has closed. Uses the scheduled handle slice
/// rather than allocating a second ticket/owner collection.
pub(in crate::serve::api) fn capture_during_prefill<T>(
    stores: &mut [Qwen35AnchorStore],
    handles: &[crate::serve::scheduler::SlotHandle],
    run: impl FnOnce() -> T,
) -> Result<T> {
    for (opened, handle) in handles.iter().enumerate() {
        if let Err(error) = stores[handle.slot_id.0 as usize].begin_capture() {
            for earlier in &handles[..opened] {
                let store = &mut stores[earlier.slot_id.0 as usize];
                let (_, ticket) = store.transaction.expect("opened by this scope");
                store.finish_capture(ticket)?;
            }
            return Err(error);
        }
    }
    let result = run();
    for handle in handles {
        let store = &mut stores[handle.slot_id.0 as usize];
        let (_, ticket) = store.transaction.expect("prefill construction retains its ticket");
        store.finish_capture(ticket)?;
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::serve::api) enum AnchorTransaction {
    Capture,
    Restore,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::serve::api) enum AnchorRestoreFault {
    Preflight,
    Encode,
    PostCommit,
}

#[derive(Clone)]
pub(in crate::serve::api) struct Qwen35AnchorStore {
    keep: usize,
    quota_bytes: Option<usize>,
    metric_slot: usize,
    origin: u64,
    epoch: u64,
    next_id: u64,
    poisoned: bool,
    committed: Vec<Qwen35Anchor>,
    pending: Option<Qwen35Anchor>,
    transaction: Option<(AnchorTransaction, AnchorTicket)>,
    restore_rank: Option<usize>,
    next_operation: u64,
    #[cfg(test)]
    pub restore_fault: Option<AnchorRestoreFault>,
    #[cfg(test)]
    pub pruning_disabled: bool,
}

impl Qwen35AnchorStore {
    pub(crate) fn include_carry_owners(
        &self, owners: &mut crate::inference::models::qwen35::carry_memory::CarryOwners,
    ) {
        for anchor in self.committed.iter().chain(self.pending.iter()) {
            if let Some(spec) = anchor.spec.as_ref() {
                owners.include(&spec.pending_target_hidden);
            }
        }
    }

    pub fn new(keep: usize) -> Result<Self> {
        Self::new_for_slot(keep, None, 0)
    }

    pub fn new_with_quota(keep: usize, quota_bytes: Option<usize>) -> Result<Self> {
        Self::new_for_slot(keep, quota_bytes, 0)
    }

    pub fn new_for_slot(
        keep: usize,
        quota_bytes: Option<usize>,
        metric_slot: usize,
    ) -> Result<Self> {
        if !(1..=8).contains(&keep) {
            return Err(anyhow!(
                "HF2Q_QWEN_ANCHOR_DEPTH must be in 1..=8 (got {keep})"
            ));
        }
        if quota_bytes == Some(0) {
            return Err(anyhow!("Qwen anchor quota must be greater than zero"));
        }
        let store = Self {
            keep,
            quota_bytes,
            metric_slot,
            origin: next_origin(),
            epoch: 0,
            next_id: 1,
            poisoned: false,
            committed: Vec::new(),
            pending: None,
            transaction: None,
            restore_rank: None,
            next_operation: 1,
            #[cfg(test)]
            restore_fault: None,
            #[cfg(test)]
            pruning_disabled: false,
        };
        store.publish_gauge();
        Ok(store)
    }

    pub fn empty_like(&self) -> Self {
        Self::new_for_slot(self.keep, self.quota_bytes, self.metric_slot)
            .expect("validated Qwen anchor store")
    }

    pub fn rebind_slot(&mut self, metric_slot: usize) {
        self.metric_slot = metric_slot;
        self.publish_gauge();
    }

    pub fn quota_headroom(&self) -> Option<usize> {
        self.quota_bytes
            .map(|quota| quota.saturating_sub(self.committed_owned().total()))
    }

    pub fn is_empty(&self) -> bool {
        self.committed.is_empty() && self.pending.is_none()
    }

    pub fn committed(&self) -> &[Qwen35Anchor] {
        &self.committed
    }

    pub fn pending(&self) -> Option<&Qwen35Anchor> {
        self.pending.as_ref()
    }


    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn open_transaction(&self) -> Option<AnchorTransaction> {
        self.transaction.map(|(kind, _)| kind)
    }

    pub fn begin_capture(&mut self) -> Result<AnchorTicket> {
        anyhow::ensure!(!self.poisoned && self.transaction.is_none(),
            "Qwen anchor capture blocked by open transaction or poison");
        let ticket = AnchorTicket {
            origin: self.origin, epoch: self.epoch, anchor_id: 0,
            operation: self.next_operation,
        };
        self.next_operation = self.next_operation.checked_add(1).expect("Qwen anchor operation overflow");
        self.transaction = Some((AnchorTransaction::Capture, ticket));
        Ok(ticket)
    }

    pub fn finish_capture(&mut self, ticket: AnchorTicket) -> Result<()> {
        anyhow::ensure!(self.transaction == Some((AnchorTransaction::Capture, ticket)),
            "stale Qwen anchor capture ticket");
        self.transaction = None;
        Ok(())
    }

    pub fn abort_transaction(&mut self, ticket: AnchorTicket) -> Result<()> {
        anyhow::ensure!(self.transaction.is_some_and(|(_, open)| open == ticket),
            "stale Qwen anchor transaction ticket");
        self.transaction = None;
        self.restore_rank = None;
        Ok(())
    }
    pub fn deepest(&self) -> Option<&Qwen35Anchor> {
        self.committed.last()
    }

    pub fn deepest_mut(&mut self) -> Option<&mut Qwen35Anchor> {
        self.committed.last_mut()
    }

    pub fn deepest_prefix_of(
        &self,
        prompt: &[u32],
        vision_fingerprint: Option<[u8; 32]>,
        text_anchor_reuse_limit: Option<usize>,
    ) -> Option<&Qwen35Anchor> {
        if self.poisoned {
            return None;
        }
        self.committed
            .iter()
            .filter(|anchor| {
                anchor.vision_fingerprint == vision_fingerprint
                    && !anchor.prompt_tokens.is_empty()
                    && prompt.starts_with(&anchor.prompt_tokens)
                    && text_anchor_reuse_limit
                        .is_none_or(|limit| anchor.prompt_tokens.len() <= limit)
            })
            .max_by_key(|anchor| anchor.prompt_tokens.len())
    }

    pub fn anchor(&self, id: u64) -> Option<&Qwen35Anchor> {
        self.committed.iter().find(|anchor| anchor.id == id)
    }

    pub fn anchor_mut(&mut self, id: u64) -> Option<&mut Qwen35Anchor> {
        self.committed.iter_mut().find(|anchor| anchor.id == id)
    }

    pub fn stage(&mut self, request: u64, mut anchor: Qwen35Anchor) -> Result<u64> {
        anyhow::ensure!(self.transaction.is_none(), "cannot stage during an open Qwen anchor transaction");
        if self.poisoned {
            return Err(anyhow!("Qwen anchor store is poisoned"));
        }
        if self
            .pending
            .as_ref()
            .is_some_and(|old| old.request != request)
        {
            return Err(anyhow!(
                "Qwen anchor store already has another request pending"
            ));
        }
        anchor.prompt_tokens.shrink_to_fit();
        anchor.prefill_logits.shrink_to_fit();
        if let Some(spec) = &anchor.spec {
            if spec.pending_target_hidden.byte_len() != spec.pending_target_hidden.data_byte_len() {
                ANCHOR_TELEMETRY
                    .capture_skipped_oversized_owner
                    .fetch_add(1, Ordering::Relaxed);
                return Err(anyhow!(
                    "Qwen anchor speculative hidden owner is not right-sized"
                ));
            }
        }
        let projected = self
            .committed_owned()
            .total()
            .checked_add(incremental_owned(&anchor, &self.committed).total())
            .ok_or_else(|| anyhow!("Qwen anchor owned-byte quota arithmetic overflow"))?;
        if self.quota_bytes.is_some_and(|quota| projected > quota) {
            ANCHOR_TELEMETRY
                .capture_skipped_quota
                .fetch_add(1, Ordering::Relaxed);
            return Err(anyhow!(
                "Qwen anchor capture exceeds slot quota: projected={projected}"
            ));
        }
        anchor.id = self.next_id;
        anchor.request = request;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| anyhow!("Qwen anchor id overflow"))?;
        let id = anchor.id;
        self.pending = Some(anchor);
        self.changed();
        ANCHOR_TELEMETRY
            .captures_staged
            .fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        super::anchor_exactness_tests::fresh_http::anchor_staged(
            self.metric_slot, self.origin, self.epoch, id, request,
            self.pending.as_ref().expect("staged anchor exists").prompt_tokens.len(),
        );
        Ok(id)
    }

    pub fn commit(&mut self, request: u64) -> Result<bool> {
        anyhow::ensure!(self.transaction.is_none(), "cannot commit during an open Qwen anchor transaction");
        let Some(anchor) = self.pending.take() else {
            return Ok(false);
        };
        if anchor.request != request {
            self.pending = Some(anchor);
            return Err(anyhow!("Qwen anchor completion request mismatch"));
        }
        let depth = anchor.prompt_tokens.len();
        self.committed
            .retain(|old| old.prompt_tokens.len() != depth);
        self.committed.push(anchor);
        self.committed
            .sort_by_key(|anchor| anchor.prompt_tokens.len());
        if self.committed.len() > self.keep {
            let evict = self.committed.len() - self.keep;
            self.committed.drain(..evict);
            ANCHOR_TELEMETRY
                .evicted
                .fetch_add(evict as u64, Ordering::Relaxed);
        }
        ANCHOR_TELEMETRY.committed.fetch_add(1, Ordering::Relaxed);
        self.changed();
        Ok(true)
    }

    pub fn discard_pending(&mut self, request: u64) -> bool {
        if self
            .pending
            .as_ref()
            .is_some_and(|anchor| anchor.request == request)
        {
            self.pending = None;
            self.changed();
            true
        } else {
            false
        }
    }

    pub fn on_cursor_set(&mut self, cursor: usize) {
        #[cfg(test)]
        if self.pruning_disabled { return; }
        let before = self.committed.len() + usize::from(self.pending.is_some());
        self.committed
            .retain(|anchor| anchor.kv.prompt_len() <= cursor);
        if self
            .pending
            .as_ref()
            .is_some_and(|anchor| anchor.kv.prompt_len() > cursor)
        {
            self.pending = None;
        }
        let removed = before - (self.committed.len() + usize::from(self.pending.is_some()));
        ANCHOR_TELEMETRY
            .tombstoned_cursor
            .fetch_add(removed as u64, Ordering::Relaxed);
        self.changed();
    }

    pub fn ticket(&mut self, anchor_id: u64) -> Result<AnchorTicket> {
        if self.poisoned {
            return Err(AnchorRestoreRefusal::Poisoned.counted());
        }
        if self.transaction.is_some() {
            return Err(AnchorRestoreRefusal::Transaction.counted());
        }
        if self.anchor(anchor_id).is_none() {
            return Err(AnchorRestoreRefusal::Unavailable.counted());
        }
        let ticket = AnchorTicket {
            origin: self.origin,
            epoch: self.epoch,
            anchor_id,
            operation: self.next_operation,
        };
        self.transaction = Some((AnchorTransaction::Restore, ticket));
        self.next_operation = self.next_operation.checked_add(1).expect("Qwen anchor operation overflow");
        Ok(ticket)
    }

    pub fn validate_ticket(&self, ticket: AnchorTicket) -> Result<&Qwen35Anchor> {
        if self.poisoned {
            return Err(AnchorRestoreRefusal::Poisoned.counted());
        }
        if ticket.origin != self.origin || ticket.epoch != self.epoch
            || self.transaction != Some((AnchorTransaction::Restore, ticket)) {
            return Err(AnchorRestoreRefusal::Stale.counted());
        }
        self.anchor(ticket.anchor_id).ok_or_else(|| AnchorRestoreRefusal::Unavailable.counted())
    }

    pub fn before_restore(&mut self, ticket: AnchorTicket) -> Result<()> {
        anyhow::ensure!(self.transaction == Some((AnchorTransaction::Restore, ticket)),
            "Qwen anchor restore ticket is not open");
        self.validate_ticket(ticket)?;
        let rank = self
            .committed
            .iter()
            .rev()
            .position(|anchor| anchor.id == ticket.anchor_id)
            .expect("validated anchor rank");
        let depth = self
            .anchor(ticket.anchor_id)
            .expect("validated anchor")
            .kv
            .prompt_len();
        // C2 Phase B discards any unpublished request payload, including an
        // ancestor whose live rows happen not to overlap the ensuing writes.
        let pending = usize::from(self.pending.take().is_some());
        self.on_cursor_set(depth);
        if pending != 0 {
            ANCHOR_TELEMETRY.tombstoned_cursor.fetch_add(pending as u64, Ordering::Relaxed);
            self.changed();
        }
        self.restore_rank = Some(rank);
        Ok(())
    }


    pub fn finish_restore(&mut self, ticket: AnchorTicket, success: bool) -> Result<usize> {
        anyhow::ensure!(self.transaction == Some((AnchorTransaction::Restore, ticket)),
            "stale Qwen anchor restore completion");
        let rank = self.restore_rank.take().ok_or_else(|| anyhow!("Qwen restore never entered Phase B"))?;
        self.transaction = None;
        if success {
            ANCHOR_TELEMETRY.restores_by_rank[rank].fetch_add(1, Ordering::Relaxed);
            self.changed();
        }
        Ok(rank)
    }
    pub fn poison_and_clear(&mut self) {
        self.committed.clear();
        self.pending = None;
        self.transaction = None;
        self.restore_rank = None;
        self.poisoned = true;
        self.changed();
    }

    pub fn reset(&mut self) {
        let removed = self.committed.len() + usize::from(self.pending.is_some());
        ANCHOR_TELEMETRY.tombstoned_cold_reset.fetch_add(removed as u64, Ordering::Relaxed);
        self.committed.clear();
        self.pending = None;
        self.poisoned = false;
        self.transaction = None;
        self.restore_rank = None;
        self.origin = next_origin();
        self.epoch = 0;
        self.publish_gauge();
    }

    pub fn committed_owned(&self) -> AnchorOwnedBytes {
        let mut owned = AnchorOwnedBytes::default();
        for (index, anchor) in self.committed.iter().enumerate() {
            owned.add(incremental_owned(anchor, &self.committed[..index]));
        }
        owned
    }

    pub fn pending_owned(&self) -> AnchorOwnedBytes {
        self.pending.as_ref().map_or_else(AnchorOwnedBytes::default,
            |anchor| incremental_owned(anchor, &self.committed))
    }

    pub fn owned_bytes(&self) -> usize {
        let committed = self.committed_owned().total();
        committed.saturating_add(self.pending_owned().total())
    }

    fn changed(&mut self) {
        self.epoch = self
            .epoch
            .checked_add(1)
            .expect("Qwen anchor store epoch overflow");
        self.publish_gauge();
    }

    fn publish_gauge(&self) {
        let mut gauges = ANCHOR_GAUGES
            .lock()
            .expect("Qwen anchor gauge lock poisoned");
        if gauges.len() <= self.metric_slot {
            gauges.resize_with(self.metric_slot + 1, Qwen35AnchorGauge::default);
        }
        gauges[self.metric_slot] = Qwen35AnchorGauge {
            slot: self.metric_slot,
            committed_count: self.committed.len(),
            committed: self.committed_owned(),
            pending: self.pending_owned(),
        };
    }
}

/// Admission and cancellation share the same atomic restore and failure path.
pub(in crate::serve::api) fn restore_committed_anchor(
    kv: &mut crate::inference::models::qwen35::kv_cache::HybridKvCache,
    slot: crate::serve::multi_seq_kv::SlotId,
    store: &mut Qwen35AnchorStore,
    anchor_id: u64,
    supervisor: &crate::serve::api::engine_supervisor::EngineSupervisor,
) -> Result<usize> {
    use anyhow::Context;
    let ticket = store.ticket(anchor_id)?;
    let mut mutating = false;
    let started = std::time::Instant::now();
    let result = (|| {
        let anchor = store.validate_ticket(ticket)?;
        #[cfg(test)]
        if store.restore_fault == Some(AnchorRestoreFault::Preflight) {
            let invalid = anchor.kv.test_with_invalid_last_layer()?;
            kv.plan_slot_anchor_restore(slot, &invalid)?;
            anyhow::bail!("last-layer preflight fault was not refused");
        }
        let plan = kv.plan_slot_anchor_restore(slot, &anchor.kv)?;
        #[cfg(test)]
        let plan = {
            let mut plan = plan;
            match store.restore_fault {
                Some(AnchorRestoreFault::Encode) => plan.test_fail_after_encode(),
                Some(AnchorRestoreFault::PostCommit) => plan.test_fail_after_commit(),
                _ => {}
            }
            plan
        };
        store.before_restore(ticket)?;
        mutating = true;
        let lease = supervisor.arm(
            "Qwen35 anchor restore", super::QWEN35_WORKER_TRANSACTION_TIMEOUT,
        )?;
        let applied = kv.apply_slot_anchor_restore(plan);
        let completed = lease.finish();
        // Do not mask the typed GPU failure with a watchdog completion error.
        applied.context("apply Qwen anchor restore")?;
        completed?;
        let rank = store.finish_restore(ticket, true)?;
        #[cfg(test)]
        super::anchor_exactness_tests::fresh_http::anchor_restore_ticket(
            slot, ticket.origin, ticket.epoch, ticket.anchor_id, ticket.operation,
        );
        Ok(rank)
    })();
    record_anchor_restore(started.elapsed());
    if result.is_err() {
        if store.open_transaction().is_some() {
            store.abort_transaction(ticket)?;
        }
        record_anchor_restore_refused();
        if mutating {
            store.poison_and_clear();
        }
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::serve::api) enum AnchorRecovery {
    Preserved { tokens: usize },
    Restored { rank: usize, tokens: usize },
    Reset,
}

/// The owning cancellation state transition, also used by real-artifact tests.
pub(in crate::serve::api) fn recover_cancelled_prefix(
    kv: &mut crate::inference::models::qwen35::kv_cache::HybridKvCache,
    slot: crate::serve::multi_seq_kv::SlotId,
    request: u64,
    store: &mut Qwen35AnchorStore,
    tokens: &mut Vec<u32>,
    spec: &mut Option<Qwen35SpecPrefixBoundary>,
    preserve: bool,
    supervisor: &crate::serve::api::engine_supervisor::EngineSupervisor,
) -> Result<AnchorRecovery> {
    use anyhow::Context;
    store.discard_pending(request);
    let live = kv.sequence_len_for_slot(slot)? as usize;
    let graft = kv.graft_region_for_slot(slot)? as usize;
    let physical_tokens = graft.checked_add(tokens.len()).context("Qwen recovery cursor overflow")?;
    if preserve && !tokens.is_empty() && live == physical_tokens {
        kv.validate_sequence_len_for_slot(slot, physical_tokens)?;
        if spec.as_ref().is_some_and(|boundary| boundary.token_count != tokens.len()
            || kv.validate_speculative_cursors_for_slot(slot, physical_tokens).is_err()) {
            *spec = None;
        }
        return Ok(AnchorRecovery::Preserved { tokens: tokens.len() });
    }
    if let Some(id) = store.committed().iter().rev()
        .find(|anchor| anchor.kv.prompt_len() <= live && !anchor.prompt_tokens.is_empty())
        .map(|anchor| anchor.id)
    {
        match restore_committed_anchor(kv, slot, store, id, supervisor) {
            Ok(rank) => {
                let anchor = store.anchor(id).expect("successful restore retains selected anchor");
                tokens.clone_from(&anchor.prompt_tokens);
                *spec = anchor.spec.clone().filter(|boundary|
                    boundary.token_count == tokens.len()
                    && kv.validate_speculative_cursors_for_slot(slot, anchor.kv.prompt_len()).is_ok());
                return Ok(AnchorRecovery::Restored { rank, tokens: tokens.len() });
            }
            Err(error) if !supervisor.is_healthy()
                || crate::serve::api::engine::is_fatal_command_buffer_error(&error) => {
                return Err(error.context("Qwen cancellation checkpoint restore"));
            }
            Err(error) => {
                tracing::warn!(slot = slot.0, error = %error,
                    "Qwen cancellation checkpoint unavailable; resetting slot");
            }
        }
    }
    kv.reset_for_slot(slot).context("reset Qwen slot after cancellation")?;
    tokens.clear();
    *spec = None;
    store.reset();
    Ok(AnchorRecovery::Reset)
}

pub(in crate::serve::api) fn configured_anchor_depth() -> Result<usize> {
    match std::env::var("HF2Q_QWEN_ANCHOR_DEPTH") {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| anyhow!("HF2Q_QWEN_ANCHOR_DEPTH must be an integer in 1..=8"))
            .and_then(|depth| {
                if (1..=8).contains(&depth) {
                    Ok(depth)
                } else {
                    Err(anyhow!(
                        "HF2Q_QWEN_ANCHOR_DEPTH must be in 1..=8 (got {depth})"
                    ))
                }
            }),
        Err(std::env::VarError::NotPresent) => Ok(1),
        Err(error) => Err(anyhow!("read HF2Q_QWEN_ANCHOR_DEPTH: {error}")),
    }
}

pub(in crate::serve::api) fn configured_anchor_quota() -> Result<Option<usize>> {
    match std::env::var("HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX") {
        Ok(value) => {
            let bytes = value.parse::<usize>().map_err(|_| {
                anyhow!("HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX must be a positive integer")
            })?;
            if bytes == 0 {
                Err(anyhow!(
                    "HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX must be greater than zero"
                ))
            } else {
                Ok(Some(bytes))
            }
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(anyhow!("read HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX: {error}")),
    }
}

fn next_origin() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let mut current = NEXT.load(Ordering::Relaxed);
    loop {
        let next = current
            .checked_add(1)
            .expect("Qwen anchor store origin overflow");
        match NEXT.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return current,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor(depth: usize, request: u64, seed: u32) -> Qwen35Anchor {
        Qwen35Anchor {
            id: 0,
            request,
            prompt_tokens: (0..depth).map(|i| seed + i as u32).collect(),
            kv: HybridKvSlotAnchor::test_stub(depth),
            prefill_logits: vec![seed as f32],
            vision_fingerprint: None,
            spec: None,
        }
    }

    #[test]
    fn anchor_cpu_nonzero_aliases_are_charged_once_across_publication() {
        let owner: std::sync::Arc<[u8]> = vec![7u8; 4096].into();
        let mut first = anchor(2, 1, 7);
        first.kv = HybridKvSlotAnchor::test_with_payload_owner(2, owner.clone());
        let mut second = anchor(4, 2, 7);
        second.kv = HybridKvSlotAnchor::test_with_payload_owner(4, owner.clone());
        let mut store = Qwen35AnchorStore::new(4).unwrap();
        store.stage(1, first).unwrap();
        assert_eq!(store.pending_owned().payload_image, 4096);
        store.commit(1).unwrap();
        store.stage(2, second).unwrap();
        assert_eq!(store.committed_owned().payload_image, 4096);
        assert_eq!(store.pending_owned().payload_image, 0);
        assert_eq!(std::sync::Arc::strong_count(&owner), 3);
        store.commit(2).unwrap();
        assert_eq!(store.committed_owned().payload_image, 4096);
        let mut third = anchor(6, 3, 7);
        third.kv = HybridKvSlotAnchor::test_with_payload_owner(6, vec![7u8; 4096].into());
        store.stage(3, third).unwrap();
        assert_eq!(store.pending_owned().payload_image, 4096, "equal bytes are not shared ownership");
        store.discard_pending(3);
        store.on_cursor_set(2);
        assert_eq!(store.committed_owned().payload_image, 4096);
        assert_eq!(std::sync::Arc::strong_count(&owner), 2);
        store.poison_and_clear();
        assert_eq!(store.owned_bytes(), 0);
        assert_eq!(std::sync::Arc::strong_count(&owner), 1);
    }

    #[test]
    fn anchor_capture_scope_closes_before_success_or_error_publication() {
        use crate::serve::{multi_seq_kv::SlotId, scheduler::SlotHandle};
        let handle = SlotHandle { slot_id: SlotId(0), generation: 1 };
        let mut stores = [Qwen35AnchorStore::new(4).unwrap()];
        let failure: Result<()> = capture_during_prefill(&mut stores, &[handle],
            || Err(anyhow!("snapshot encode refused"))).unwrap();
        assert!(failure.is_err());
        assert_eq!(stores[0].open_transaction(), None);
        let success = capture_during_prefill(&mut stores, &[handle], || 17).unwrap();
        assert_eq!(success, 17);
        assert_eq!(stores[0].open_transaction(), None);
        let capture = stores[0].begin_capture().unwrap();
        assert!(capture_during_prefill(&mut stores, &[handle], || 0).is_err());
        assert_eq!(stores[0].open_transaction(), Some(AnchorTransaction::Capture));
        stores[0].finish_capture(capture).unwrap();
    }

    #[test]
    fn pending_is_invisible_and_commit_keeps_deepest_bound() {
        let mut store = Qwen35AnchorStore::new(2).unwrap();
        for (request, depth) in [(1, 2), (2, 4), (3, 6)] {
            store.stage(request, anchor(depth, request, 7)).unwrap();
            assert!(store
                .deepest_prefix_of(&[7, 8, 9, 10, 11, 12], None, None)
                .is_none_or(|a| a.prompt_tokens.len() < depth));
            store.commit(request).unwrap();
        }
        assert_eq!(
            store
                .committed()
                .iter()
                .map(|anchor| anchor.prompt_tokens.len())
                .collect::<Vec<_>>(),
            vec![4, 6]
        );
    }

    #[test]
    fn cursor_rewind_prunes_descendants_before_reuse() {
        let mut store = Qwen35AnchorStore::new(4).unwrap();
        for (request, depth) in [(1, 2), (2, 4), (3, 6)] {
            store.stage(request, anchor(depth, request, 7)).unwrap();
            store.commit(request).unwrap();
        }
        store.on_cursor_set(4);
        assert_eq!(store.committed().len(), 2);
        assert_eq!(
            store
                .deepest_prefix_of(&[7, 8, 9, 10, 11, 12], None, None)
                .unwrap()
                .prompt_tokens
                .len(),
            4
        );
    }

    #[test]
    fn restore_ticket_is_epoch_bound() {
        let mut store = Qwen35AnchorStore::new(2).unwrap();
        let id = store.stage(1, anchor(2, 1, 7)).unwrap();
        store.commit(1).unwrap();
        let ticket = store.ticket(id).unwrap();
        store.abort_transaction(ticket).unwrap();
        store.stage(2, anchor(4, 2, 7)).unwrap();
        assert!(store.validate_ticket(ticket).is_err());
    }

    #[test]
    fn anchor_transactions_are_closed_and_pending_publication_is_not_a_ticket() {
        let mut store = Qwen35AnchorStore::new(4).unwrap();
        let capture = store.begin_capture().unwrap();
        assert_eq!(store.open_transaction(), Some(AnchorTransaction::Capture));
        assert!(store.begin_capture().is_err());
        store.finish_capture(capture).unwrap();
        let next = store.begin_capture().unwrap();
        assert!(store.finish_capture(capture).is_err(), "old capture cannot close a new operation");
        store.finish_capture(next).unwrap();
        let id = store.stage(1, anchor(2, 1, 7)).unwrap();
        assert_eq!(store.open_transaction(), None);
        store.commit(1).unwrap();
        assert!(!store.commit(1).unwrap());
        let restore = store.ticket(id).unwrap();
        store.before_restore(restore).unwrap();
        store.finish_restore(restore, true).unwrap();
        assert!(store.finish_restore(restore, true).is_err());
        assert_eq!(store.open_transaction(), None);
        store.poison_and_clear();
        assert!(store.is_poisoned());
        assert!(store.begin_capture().is_err());
        assert!(store.ticket(id).is_err());
        assert_eq!(store.owned_bytes(), 0);
        store.reset();
        assert!(!store.is_poisoned());
        let capture = store.begin_capture().unwrap();
        store.abort_transaction(capture).unwrap();
    }

    #[test]
    fn anchor_u8_real_open_tickets_refuse_joint_admission_nonvacuously() {
        use crate::serve::api::engine::qwen35_joint_decode::{
            joint_anchor_admission, record_joint_ineligible, JOINT_TELEMETRY,
        };
        use crate::serve::scheduler::SlotHandle;
        use crate::serve::multi_seq_kv::SlotId;
        let handle = SlotHandle { slot_id: SlotId(0), generation: 1 };
        let mut stores = [Qwen35AnchorStore::new(4).unwrap()];
        let capture = stores[0].begin_capture().unwrap();
        let before = JOINT_TELEMETRY.anchor_capture.load(Ordering::Relaxed);
        let reason = joint_anchor_admission(&[handle], &stores).unwrap();
        assert_eq!(reason.label(), "anchor_capture");
        record_joint_ineligible(&reason);
        assert_eq!(JOINT_TELEMETRY.anchor_capture.load(Ordering::Relaxed), before + 1);
        stores[0].finish_capture(capture).unwrap();
        let id = stores[0].stage(1, anchor(2, 1, 7)).unwrap();
        assert!(joint_anchor_admission(&[handle], &stores).is_none(), "pending is not open capture");
        stores[0].commit(1).unwrap();
        let restore = stores[0].ticket(id).unwrap();
        let before = JOINT_TELEMETRY.anchor_restore.load(Ordering::Relaxed);
        let reason = joint_anchor_admission(&[handle], &stores).unwrap();
        assert_eq!(reason.label(), "anchor_restore");
        record_joint_ineligible(&reason);
        assert_eq!(JOINT_TELEMETRY.anchor_restore.load(Ordering::Relaxed), before + 1);
        stores[0].before_restore(restore).unwrap();
        stores[0].finish_restore(restore, true).unwrap();
        assert!(joint_anchor_admission(&[handle], &stores).is_none());
    }

    #[test]
    fn quota_refusal_does_not_publish_pending_state() {
        let candidate = anchor(8, 1, 7);
        let quota = candidate.owned_bytes() - 1;
        let mut store = Qwen35AnchorStore::new_with_quota(4, Some(quota)).unwrap();
        let error = store.stage(1, candidate).unwrap_err();
        assert!(format!("{error:#}").contains("exceeds slot quota"));
        assert!(store.is_empty());
        assert_eq!(store.owned_bytes(), 0);
    }

    #[test]
    fn speculative_boundary_materializes_a_right_sized_hidden_row() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().unwrap();
        let mut parent = device
            .alloc_buffer(
                8 * std::mem::size_of::<f32>(),
                mlx_native::DType::F32,
                vec![8],
            )
            .unwrap();
        parent
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let row = parent.slice_view(4 * std::mem::size_of::<f32>() as u64, 4);
        assert_ne!(row.byte_len(), row.data_byte_len());

        let boundary = owned_spec_boundary(&device, 17, &row).unwrap();

        assert_eq!(boundary.token_count, 17);
        assert_eq!(
            boundary.pending_target_hidden.byte_len(),
            boundary.pending_target_hidden.data_byte_len()
        );
        assert_eq!(
            boundary.pending_target_hidden.as_slice::<f32>().unwrap(),
            &[5.0, 6.0, 7.0, 8.0]
        );
    }
}
