//! Issue #727 (spill and restore on reclaim) for the Qwen SlotAware worker.
//!
//! With more agent conversations than slots, admission hands a request a
//! slot that still holds another conversation's resumable state. Before this
//! module, that state was discarded, so which conversation later resumed and
//! which recomputed its prefix depended on arrival order. Cold prefill and
//! live-tail resume differ numerically, so the order changed the output.
//!
//! Admission now copies the discarded state into a host spill store: the
//! slot's cache state ([`HybridKvSlotSpill`]) plus exactly the ledger slot
//! affinity reads (the retained token prefix with its speculative boundary,
//! and the prompt anchor). A later request whose prompt matches a spilled
//! ledger better than the slot it was granted, by the same longest-prefix
//! rule affinity applies to idle slots, gets the spill restored into its
//! granted slot. Admission then takes the same live-tail or anchor resume it
//! would have taken had the conversation kept its slot.
//!
//! The store has a fixed byte budget (`serve.slot_spill_budget`) and a fixed
//! entry cap ([`QWEN35_SLOT_SPILL_ENTRIES_PER_SLOT`] per worker slot); it
//! never reads available memory. Entries are stamped from a monotonic
//! counter when spilled; eviction removes the oldest stamp first; a restored
//! entry leaves the store. Every outcome is therefore a function of the
//! request sequence alone. A spill runs only when admission would otherwise
//! discard a granted slot's state: the request matches nothing in the slot
//! (cold), a better-matching spill is restored over it, the request starts
//! new content exactly at the slot's anchor (a fork, or another conversation
//! sharing that anchor), or an embedding takes the slot. A conversation's own
//! next turn or retry in its own slot resumes in place (live tail, or its
//! stable anchor plus the shared assistant header) and copies nothing, and
//! the lookup is skipped while the store is empty. A worker whose requests
//! each continue (or retry) the conversation already in their slot, or land
//! in an empty slot, therefore never copies anything.

use anyhow::Context;

use super::{
    qwen35_anchor_vision_compatible, qwen35_slot_affinity, Qwen35PromptAnchor, Qwen35RetainedPrefix,
    Qwen35SlotPreference, SlotId,
};
use crate::inference::models::qwen35::kv_cache::{HybridKvCache, HybridKvSlotSpill};
use crate::serve::api::engine_qwen35::Qwen35SpecPrefixBoundary;

const LOG_TARGET: &str = "hf2q::serve::api::engine_qwen35::slot_spill";

/// Spill entries kept per worker slot. The byte budget is normally the
/// binding limit; this cap bounds the prefix lookup done on admission.
pub(super) const QWEN35_SLOT_SPILL_ENTRIES_PER_SLOT: usize = 4;

pub(super) struct SlotSpillEntry<T> {
    pub(super) stamp: u64,
    pub(super) bytes: u64,
    pub(super) payload: T,
}

pub(super) enum SpillInsert<T> {
    Stored {
        stamp: u64,
        evicted: Vec<SlotSpillEntry<T>>,
    },
    /// The entry alone exceeds the budget (or the store holds no entries);
    /// the caller keeps ownership and the conversation goes cold later.
    Rejected(T),
}

/// Deterministic byte- and count-bounded store with oldest-stamp eviction.
pub(super) struct SlotSpillStore<T> {
    budget_bytes: u64,
    max_entries: usize,
    next_stamp: u64,
    resident_bytes: u64,
    /// Ascending stamp order: the front is the oldest entry.
    entries: Vec<SlotSpillEntry<T>>,
}

impl<T> SlotSpillStore<T> {
    pub(super) fn new(budget_bytes: u64, max_entries: usize) -> Self {
        Self {
            budget_bytes,
            max_entries,
            next_stamp: 0,
            resident_bytes: 0,
            entries: Vec::new(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Whether an entry of `bytes` can be stored at all.
    pub(super) fn admits(&self, bytes: u64) -> bool {
        self.max_entries > 0 && bytes <= self.budget_bytes
    }

    /// Store `payload`, first evicting oldest entries until both the entry
    /// cap and the byte budget hold.
    pub(super) fn insert(&mut self, bytes: u64, payload: T) -> SpillInsert<T> {
        if !self.admits(bytes) {
            return SpillInsert::Rejected(payload);
        }
        let mut evicted = Vec::new();
        while self.entries.len() >= self.max_entries
            || self.resident_bytes.saturating_add(bytes) > self.budget_bytes
        {
            evicted.push(self.take(0));
        }
        let stamp = self.next_stamp;
        self.next_stamp += 1;
        self.resident_bytes += bytes;
        self.entries.push(SlotSpillEntry {
            stamp,
            bytes,
            payload,
        });
        SpillInsert::Stored { stamp, evicted }
    }

    /// Entries with their store index, newest stamp first.
    pub(super) fn newest_first(&self) -> impl Iterator<Item = (usize, &SlotSpillEntry<T>)> {
        self.entries.iter().enumerate().rev()
    }

    pub(super) fn take(&mut self, index: usize) -> SlotSpillEntry<T> {
        let entry = self.entries.remove(index);
        self.resident_bytes -= entry.bytes;
        entry
    }
}

/// One reclaimed conversation: its slot's cache state and its resume ledger.
pub(super) struct Qwen35SlotSpill {
    source_slot: u32,
    kv: HybridKvSlotSpill,
    retained: Qwen35RetainedPrefix,
    anchor: Option<Qwen35PromptAnchor>,
}

pub(super) type Qwen35SlotSpillStore = SlotSpillStore<Qwen35SlotSpill>;

/// What happens to the granted slot's previous conversation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum VictimSpill {
    /// Nothing is discarded (an in-place live-tail resume, or an empty slot).
    Keep,
    /// The slot's ledger is cleared or replaced: move it into the spill.
    Move,
    /// An in-place anchor rewind keeps the slot's anchor for this request,
    /// and the request starts new content exactly at that anchor (another
    /// conversation sharing it, or a fork): the occupant's live tail beyond
    /// the anchor would be lost, so spill a copy of the ledger.
    Copy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35ReclaimPlan {
    /// Store index to restore into the granted slot and the resume it offers.
    pub(super) restore: Option<(usize, Qwen35SlotPreference)>,
    pub(super) victim: VictimSpill,
}

/// Decide the reclaim for one admission. `in_place` is affinity's choice for
/// the granted slot (`None` when the request was not granted the slot it
/// matched, or matched none); `spill_match` is the best stored conversation;
/// `victim_retained` is the occupant's retained token ledger. A spill is
/// restored only when it resumes strictly more tokens than the granted slot
/// does in place.
///
/// An anchor rewind copies the occupant only when `prompt_tokens` diverges
/// from `victim_retained` exactly at the anchor boundary. A conversation's
/// own next turn or retry also rewinds to its stable anchor (the generation
/// cue and reasoning are rewritten), but it shares at least the assistant
/// header beyond the anchor, so it keeps the slot without a copy.
pub(super) fn plan_qwen35_reclaim(
    in_place: Option<Qwen35SlotPreference>,
    spill_match: Option<(usize, Qwen35SlotPreference)>,
    prompt_tokens: &[u32],
    victim_retained: &[u32],
    victim_has_ledger: bool,
) -> Qwen35ReclaimPlan {
    let in_place_tokens = in_place.map_or(0, |preference| preference.cached_tokens);
    let restore = spill_match.filter(|(_, preference)| preference.cached_tokens > in_place_tokens);
    let victim = match in_place {
        _ if !victim_has_ledger => VictimSpill::Keep,
        _ if restore.is_some() => VictimSpill::Move,
        None => VictimSpill::Move,
        Some(preference)
            if preference.restore_prompt_anchor
                && victim_retained.len() > in_place_tokens
                && common_prefix_len(prompt_tokens, victim_retained) == in_place_tokens =>
        {
            VictimSpill::Copy
        }
        Some(_) => VictimSpill::Keep,
    };
    Qwen35ReclaimPlan { restore, victim }
}

fn common_prefix_len(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

/// The (retained, anchor) prefixes a conversation's ledger offers a request,
/// filtered by vision compatibility exactly as slot affinity filters idle
/// slots: a retained prefix is usable only beside an anchor with the
/// request's vision fingerprint.
pub(super) fn qwen35_resume_candidate<'a>(
    retained: &'a Qwen35RetainedPrefix,
    anchor: Option<&'a Qwen35PromptAnchor>,
    request_vision_fingerprint: Option<[u8; 32]>,
    text_anchor_reuse_limit: Option<usize>,
) -> (&'a [u32], Option<&'a [u32]>) {
    let exact_anchor_matches =
        anchor.is_some_and(|anchor| anchor.vision_fingerprint == request_vision_fingerprint);
    let anchor_matches = anchor.is_some_and(|anchor| {
        qwen35_anchor_vision_compatible(
            anchor.vision_fingerprint,
            anchor.prompt_tokens.len(),
            request_vision_fingerprint,
            text_anchor_reuse_limit,
        )
    });
    (
        if exact_anchor_matches {
            retained.as_slice()
        } else {
            &[]
        },
        anchor
            .filter(|_| anchor_matches)
            .map(|anchor| anchor.prompt_tokens.as_slice()),
    )
}

/// The longest-prefix candidate among `(index, retained, anchor)` ledgers,
/// by slot affinity's rule. Candidates come newest first, so equal matches
/// resolve to the newest. The returned preference's `slot` is meaningless
/// (it carries the index); the caller substitutes the granted slot.
pub(super) fn qwen35_best_ledger_match<'a>(
    candidates: impl IntoIterator<Item = (usize, &'a [u32], Option<&'a [u32]>)>,
    prompt_tokens: &[u32],
) -> Option<(usize, Qwen35SlotPreference)> {
    let preference = qwen35_slot_affinity(
        candidates
            .into_iter()
            .map(|(index, retained, anchor)| (index, retained, anchor, None, false)),
        prompt_tokens,
    )?
    .preference();
    Some((preference.slot.0 as usize, preference))
}

fn best_spill_match(
    store: &Qwen35SlotSpillStore,
    prompt_tokens: &[u32],
    request_vision_fingerprint: Option<[u8; 32]>,
    text_anchor_reuse_limit: Option<usize>,
) -> Option<(usize, Qwen35SlotPreference)> {
    if store.is_empty() {
        return None;
    }
    qwen35_best_ledger_match(
        store.newest_first().map(|(index, entry)| {
            let (retained, anchor) = qwen35_resume_candidate(
                &entry.payload.retained,
                entry.payload.anchor.as_ref(),
                request_vision_fingerprint,
                text_anchor_reuse_limit,
            );
            (index, retained, anchor)
        }),
        prompt_tokens,
    )
}

fn ledger_tokens(retained: &Qwen35RetainedPrefix, anchor: Option<&Qwen35PromptAnchor>) -> usize {
    retained
        .len()
        .max(anchor.map_or(0, |anchor| anchor.prompt_tokens.len()))
}

/// Host-side bytes of a ledger, charged to the spill budget with the cache
/// copy so the store bound covers everything an entry keeps alive.
fn ledger_bytes(retained: &Qwen35RetainedPrefix, anchor: Option<&Qwen35PromptAnchor>) -> usize {
    let spec = |spec: &Option<Qwen35SpecPrefixBoundary>| {
        spec.as_ref()
            .map_or(0, |spec| spec.pending_target_hidden.byte_len())
    };
    let token = std::mem::size_of::<u32>();
    retained.len() * token
        + spec(&retained.spec)
        + anchor.map_or(0, |anchor| {
            anchor.prompt_tokens.len() * token
                + anchor.kv.total_bytes()
                + anchor.prefill_logits.len() * std::mem::size_of::<f32>()
                + spec(&anchor.spec)
        })
}

/// Spill the conversation that held `slot` (see [`VictimSpill`]). A spill
/// that cannot be taken or stored is logged; the conversation then goes cold
/// exactly as it did before #727.
fn spill_qwen35_victim(
    kv: &HybridKvCache,
    store: &mut Qwen35SlotSpillStore,
    slot: SlotId,
    retained: &mut Qwen35RetainedPrefix,
    anchor: &mut Option<Qwen35PromptAnchor>,
    mode: VictimSpill,
) {
    if mode == VictimSpill::Keep {
        return;
    }
    let tokens = ledger_tokens(retained, anchor.as_ref());
    let kv_bytes = match kv.slot_spill_bytes(slot) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(target: LOG_TARGET, slot = slot.0, tokens, "Qwen35 slot spill skipped: {error:#}");
            return;
        }
    };
    let bytes = u64::try_from(kv_bytes + ledger_bytes(retained, anchor.as_ref())).unwrap_or(u64::MAX);
    if !store.admits(bytes) {
        tracing::debug!(
            target: LOG_TARGET,
            slot = slot.0,
            tokens,
            bytes,
            "Qwen35 slot spill not stored: entry exceeds the spill budget"
        );
        return;
    }
    let spilled = match kv.spill_slot(slot) {
        Ok(spilled) => spilled,
        Err(error) => {
            tracing::warn!(target: LOG_TARGET, slot = slot.0, tokens, "Qwen35 slot spill skipped: {error:#}");
            return;
        }
    };
    let (retained, anchor) = match mode {
        VictimSpill::Move => (std::mem::take(retained), anchor.take()),
        VictimSpill::Copy => (retained.clone(), anchor.clone()),
        VictimSpill::Keep => unreachable!("Keep returned above"),
    };
    let payload = Qwen35SlotSpill {
        source_slot: slot.0,
        kv: spilled,
        retained,
        anchor,
    };
    match store.insert(bytes, payload) {
        SpillInsert::Stored { stamp, evicted } => {
            for entry in evicted {
                tracing::debug!(
                    target: LOG_TARGET,
                    slot = entry.payload.source_slot,
                    stamp = entry.stamp,
                    tokens = ledger_tokens(&entry.payload.retained, entry.payload.anchor.as_ref()),
                    bytes = entry.bytes,
                    "Qwen35 slot spill evicted (oldest stamp)"
                );
            }
            tracing::debug!(
                target: LOG_TARGET,
                slot = slot.0,
                stamp,
                tokens,
                bytes,
                copy = mode == VictimSpill::Copy,
                entries = store.len(),
                resident_bytes = store.resident_bytes(),
                "Qwen35 slot spilled"
            );
        }
        SpillInsert::Rejected(_) => unreachable!("admission was checked before the copy"),
    }
}

/// Spill a slot an embedding request takes over: its ledger is cleared.
pub(super) fn spill_reclaimed_qwen35_slot(
    kv: &HybridKvCache,
    store: &mut Qwen35SlotSpillStore,
    slot: SlotId,
    retained: &mut Qwen35RetainedPrefix,
    anchor: &mut Option<Qwen35PromptAnchor>,
) {
    if !retained.is_empty() || anchor.is_some() {
        spill_qwen35_victim(kv, store, slot, retained, anchor, VictimSpill::Move);
    }
}

/// Generation admission's reclaim step, run once the scheduler granted
/// `slot`. Spills the slot's previous conversation when this admission
/// discards it, restores the best-matching spilled conversation when it
/// resumes more of the prompt than the slot does in place, and returns the
/// resume preference for the granted slot.
///
/// An `Err` means a restore wrote part of the slot; the caller must reset
/// the slot and clear its ledger.
#[allow(clippy::too_many_arguments)]
pub(super) fn reclaim_qwen35_slot(
    kv: &mut HybridKvCache,
    store: &mut Qwen35SlotSpillStore,
    slot: SlotId,
    retained: &mut Qwen35RetainedPrefix,
    anchor: &mut Option<Qwen35PromptAnchor>,
    in_place: Option<Qwen35SlotPreference>,
    prompt_tokens: &[u32],
    request_vision_fingerprint: Option<[u8; 32]>,
    text_anchor_reuse_limit: Option<usize>,
) -> anyhow::Result<Option<Qwen35SlotPreference>> {
    let spill_match = best_spill_match(
        store,
        prompt_tokens,
        request_vision_fingerprint,
        text_anchor_reuse_limit,
    );
    let plan = plan_qwen35_reclaim(
        in_place,
        spill_match,
        prompt_tokens,
        retained.as_slice(),
        !retained.is_empty() || anchor.is_some(),
    );
    // Take the matched entry before spilling the victim so the victim's
    // insertion can never evict it.
    let restore = plan
        .restore
        .map(|(index, preference)| (store.take(index), preference));
    spill_qwen35_victim(kv, store, slot, retained, anchor, plan.victim);
    let Some((entry, preference)) = restore else {
        return Ok(in_place);
    };
    let tokens = ledger_tokens(&entry.payload.retained, entry.payload.anchor.as_ref());
    kv.restore_slot_spill(slot, &entry.payload.kv)
        .with_context(|| {
            format!(
                "Qwen35 slot spill restore into slot {} (spilled from slot {})",
                slot.0, entry.payload.source_slot
            )
        })?;
    *retained = entry.payload.retained;
    *anchor = entry.payload.anchor;
    tracing::debug!(
        target: LOG_TARGET,
        slot = slot.0,
        from_slot = entry.payload.source_slot,
        stamp = entry.stamp,
        tokens,
        bytes = entry.bytes,
        cached_tokens = preference.cached_tokens,
        anchor = preference.restore_prompt_anchor,
        "Qwen35 slot spill restored"
    );
    Ok(Some(Qwen35SlotPreference { slot, ..preference }))
}

#[cfg(test)]
#[path = "qwen35_slot_spill_tests.rs"]
mod tests;
