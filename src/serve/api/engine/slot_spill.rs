//! Issue #727 (spill and restore on reclaim): the family-neutral half shared
//! by the Qwen and Gemma SlotAware workers (`qwen35_slot_spill`,
//! `gemma4_slot_spill`).
//!
//! * [`SlotSpillStore`]: the host store of reclaimed conversations. It has a
//!   fixed byte budget (`serve.slot_spill_budget`) and a fixed entry cap
//!   ([`SLOT_SPILL_ENTRIES_PER_SLOT`] per worker slot) and never reads
//!   available memory. Entries are stamped from a monotonic counter when
//!   spilled; eviction removes the oldest stamp first; a restored entry
//!   leaves the store. Every outcome is therefore a function of the request
//!   sequence alone.
//! * [`plan_reclaim`]: what one admission does with the granted slot's
//!   previous conversation and with the best stored conversation.

/// Spill entries kept per worker slot. The byte budget is normally the
/// binding limit; this cap bounds the prefix lookup done on admission.
pub(super) const SLOT_SPILL_ENTRIES_PER_SLOT: usize = 4;

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

/// The resume one ledger (a slot's or a spill's) offers a request: the
/// family's slot-affinity preference.
pub(super) trait ResumePreference: Copy {
    fn cached_tokens(&self) -> usize;
    fn restores_prompt_anchor(&self) -> bool;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ReclaimPlan<I, P> {
    /// The ledger to restore into the granted slot and the resume it offers.
    pub(super) restore: Option<(I, P)>,
    pub(super) victim: VictimSpill,
}

/// Decide the reclaim for one admission. `in_place` is affinity's choice for
/// the granted slot (`None` when the request was not granted the slot it
/// matched, or matched none); `candidate` is the best ledger held elsewhere
/// (a stored conversation); `victim_retained` is the occupant's retained
/// token ledger. A candidate is restored only when it resumes strictly more
/// tokens than the granted slot does in place.
///
/// An anchor rewind copies the occupant only when `prompt_tokens` diverges
/// from `victim_retained` exactly at the anchor boundary. A conversation's
/// own next turn or retry also rewinds to its stable anchor (the generation
/// cue and reasoning are rewritten), but it shares at least the assistant
/// header beyond the anchor, so it keeps the slot without a copy.
pub(super) fn plan_reclaim<I, P: ResumePreference>(
    in_place: Option<P>,
    candidate: Option<(I, P)>,
    prompt_tokens: &[u32],
    victim_retained: &[u32],
    victim_has_ledger: bool,
) -> ReclaimPlan<I, P> {
    let in_place_tokens = in_place.map_or(0, |preference| preference.cached_tokens());
    let restore = candidate.filter(|(_, preference)| preference.cached_tokens() > in_place_tokens);
    let victim = match in_place {
        _ if !victim_has_ledger => VictimSpill::Keep,
        _ if restore.is_some() => VictimSpill::Move,
        None => VictimSpill::Move,
        Some(preference)
            if preference.restores_prompt_anchor()
                && victim_retained.len() > in_place_tokens
                && common_prefix_len(prompt_tokens, victim_retained) == in_place_tokens =>
        {
            VictimSpill::Copy
        }
        Some(_) => VictimSpill::Keep,
    };
    ReclaimPlan { restore, victim }
}

pub(super) fn common_prefix_len(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
#[path = "slot_spill_tests.rs"]
mod tests;
