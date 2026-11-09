use anyhow::{anyhow, Result};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    LazyLock, Mutex,
};
use std::time::Duration;

use crate::inference::models::qwen35::kv_cache::HybridKvSlotAnchor;

use super::Qwen35SpecPrefixBoundary;

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

    fn add_anchor(&mut self, anchor: &Qwen35Anchor) {
        let add = anchor.owned();
        self.payload_image = self.payload_image.saturating_add(add.payload_image);
        self.logits = self.logits.saturating_add(add.logits);
        self.hidden = self.hidden.saturating_add(add.hidden);
        self.tokens = self.tokens.saturating_add(add.tokens);
        self.metadata = self.metadata.saturating_add(add.metadata);
    }
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
    pub tombstoned_cursor: AtomicU64,
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
    tombstoned_cursor: AtomicU64::new(0),
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
}

impl Qwen35AnchorStore {
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
            .checked_add(anchor.owned_bytes())
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
        Ok(id)
    }

    pub fn commit(&mut self, request: u64) -> Result<bool> {
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

    pub fn ticket(&self, anchor_id: u64) -> Result<AnchorTicket> {
        if self.poisoned || self.pending.is_some() {
            record_anchor_restore_refused();
            return Err(anyhow!("Qwen anchor restore blocked by store state"));
        }
        if self.anchor(anchor_id).is_none() {
            record_anchor_restore_refused();
            return Err(anyhow!("Qwen anchor {anchor_id} is not committed"));
        }
        Ok(AnchorTicket {
            origin: self.origin,
            epoch: self.epoch,
            anchor_id,
        })
    }

    pub fn validate_ticket(&self, ticket: AnchorTicket) -> Result<&Qwen35Anchor> {
        if self.poisoned || ticket.origin != self.origin || ticket.epoch != self.epoch {
            record_anchor_restore_refused();
            return Err(anyhow!("stale Qwen anchor restore ticket"));
        }
        self.anchor(ticket.anchor_id).ok_or_else(|| {
            record_anchor_restore_refused();
            anyhow!("Qwen anchor {} was retired", ticket.anchor_id)
        })
    }

    pub fn before_restore(&mut self, ticket: AnchorTicket) -> Result<()> {
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
            .prompt_tokens
            .len();
        self.on_cursor_set(depth);
        ANCHOR_TELEMETRY.restores_by_rank[rank].fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn poison_and_clear(&mut self) {
        self.committed.clear();
        self.pending = None;
        self.poisoned = true;
        self.changed();
    }

    pub fn reset(&mut self) {
        self.committed.clear();
        self.pending = None;
        self.poisoned = false;
        self.origin = next_origin();
        self.epoch = 0;
        self.publish_gauge();
    }

    pub fn committed_owned(&self) -> AnchorOwnedBytes {
        let mut owned = AnchorOwnedBytes::default();
        for anchor in &self.committed {
            owned.add_anchor(anchor);
        }
        owned
    }

    pub fn pending_owned(&self) -> AnchorOwnedBytes {
        let mut owned = AnchorOwnedBytes::default();
        if let Some(anchor) = &self.pending {
            owned.add_anchor(anchor);
        }
        owned
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
    NEXT.try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        value.checked_add(1)
    })
    .expect("Qwen anchor store origin overflow")
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
        store.stage(2, anchor(4, 2, 7)).unwrap();
        assert!(store.validate_ticket(ticket).is_err());
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
}
