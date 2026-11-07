//! Issue #727 (spill and restore on reclaim) for the Gemma 4 SlotAware
//! worker.
//!
//! With more agent conversations than slots, admission hands a request a
//! slot that still holds another conversation's resumable state. Before this
//! module, that state was discarded, so which conversation later resumed and
//! which recomputed its prefix depended on arrival order. Cold prefill and
//! resume differ numerically, so the order changed the output.
//!
//! A Gemma slot's state is its KV rows and cursors
//! ([`GemmaKvSlotSpill`]: the live scaffold's global rows and sliding rings,
//! every scaffold's cursors, the graft rows below them) plus the ledger slot
//! affinity reads: the retained token prefix and the published prompt anchor
//! (stable-boundary, full-input or exact-image checkpoint, with its saved
//! sliding rings). A Gemma ledger is resumable only beside its anchor (the
//! request-aware affinity rule binds the retained prefix to the anchor's
//! vision identity), so a slot without an anchor holds nothing a later
//! request could resume and is never spilled.
//!
//! Admission ([`reclaim_gemma4_slot`]), once the scheduler has granted a
//! slot:
//! - **Restore.** The best ledger held elsewhere is restored into the
//!   granted slot when it resumes strictly more of the prompt than the slot
//!   does in place, by the longest-prefix rule affinity applies to idle
//!   slots. It is either a spilled conversation, or the request's own idle
//!   slot when the scheduler did not grant it (the shared KV budget refused
//!   that slot's growth): that state moves into the granted slot in place.
//!   Admission then takes the same live-tail or anchor resume it would have
//!   taken in the original slot, on the same bytes.
//! - **Spill.** A granted slot whose conversation this admission discards
//!   (a cold request, or a restore over it) is spilled first; so is a slot
//!   an embedding or a cold cohort lane takes ([`spill_reclaimed_gemma4_slot`]).
//!
//! The store and the decision table are shared with the Qwen worker
//! (`slot_spill`), with one difference: an in-place anchor rewind never
//! spills a copy. A Gemma full-input checkpoint ends exactly where a tool
//! turn's generation begins, so a conversation's own next turn can diverge
//! from its retained tail right at the anchor, which the shared rule would
//! read as a fork and copy on every turn. A fork of a published anchor
//! therefore still replaces the occupant's live tail, as before #727.
//!
//! A conversation's own next turn or retry in its own slot copies nothing,
//! and the lookup is skipped while the store is empty: a worker whose
//! requests each continue the conversation already in their slot, or land in
//! an empty slot, never copies anything. The spill copy reads shared-storage
//! buffers on the CPU between GPU transactions, at the admission point the
//! anchor checkpoints use.

use anyhow::Context;

use super::slot_spill::{
    plan_reclaim, ResumePreference, SlotSpillStore, SpillInsert, VictimSpill,
    SLOT_SPILL_ENTRIES_PER_SLOT,
};
use super::{gemma4_slot_affinity, Gemma4PromptAnchor, Gemma4SlotPreference, SlotId};
use crate::inference::models::gemma4::kv_cache::slot_spill::{
    GemmaKvSlotSpill, GemmaSlotKvRegime, GemmaSlotScaffolds,
};

const LOG_TARGET: &str = "hf2q::serve::api::engine::gemma4_slot_spill";

impl ResumePreference for Gemma4SlotPreference {
    fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    fn restores_prompt_anchor(&self) -> bool {
        self.restore_prompt_anchor
    }
}

/// One reclaimed conversation: its slot's cache state and its resume ledger.
pub(super) struct Gemma4SlotSpill {
    source_slot: u32,
    kv: GemmaKvSlotSpill,
    /// Graft rows below the cursors (the bound graft; checked on restore).
    graft_len: usize,
    retained: Vec<u32>,
    /// The published anchor, parked without an anchor-quota lease.
    anchor: Option<Gemma4PromptAnchor>,
}

/// The worker's spill store and the KV scaffold it spills.
pub(super) struct Gemma4SlotSpills {
    store: SlotSpillStore<Gemma4SlotSpill>,
    regime: GemmaSlotKvRegime,
}

impl Gemma4SlotSpills {
    pub(super) fn new(budget_bytes: u64, slots: usize, regime: GemmaSlotKvRegime) -> Self {
        Self {
            store: SlotSpillStore::new(
                budget_bytes,
                slots.saturating_mul(SLOT_SPILL_ENTRIES_PER_SLOT),
            ),
            regime,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.store.len()
    }

    #[cfg(test)]
    pub(super) fn resident_bytes(&self) -> u64 {
        self.store.resident_bytes()
    }

    /// The longest-prefix spilled conversation for a request, newest first
    /// on ties, skipping store indices `excluded` marks. `None` while the
    /// store is empty, without a lookup. The returned preference's `slot`
    /// carries the store index.
    pub(super) fn best_match(
        &self,
        prompt_tokens: &[u32],
        vision_fingerprint: Option<[u8; 32]>,
        excluded: &[bool],
    ) -> Option<(usize, Gemma4SlotPreference)> {
        if self.store.is_empty() {
            return None;
        }
        gemma4_best_ledger_match(
            self.store
                .newest_first()
                .filter(|(index, _)| !excluded.get(*index).copied().unwrap_or(false))
                .map(|(index, entry)| {
                    let (retained, anchor) = gemma4_resume_candidate(
                        &entry.payload.retained,
                        entry.payload.anchor.as_ref(),
                        vision_fingerprint,
                    );
                    (index, retained, anchor)
                }),
            prompt_tokens,
        )
    }

    /// Tokens the best spilled conversation resumes for a request; batch
    /// planners treat a request with a spilled prefix as warm.
    pub(super) fn resume_tokens(
        &self,
        prompt_tokens: &[u32],
        vision_fingerprint: Option<[u8; 32]>,
    ) -> usize {
        self.best_match(prompt_tokens, vision_fingerprint, &[])
            .map_or(0, |(_, preference)| preference.cached_tokens)
    }
}

/// The (retained, anchor) prefixes a Gemma ledger offers a request: a
/// retained prefix is usable only beside an anchor with the request's vision
/// identity, exactly as `preferred_gemma4_slot_for_request` filters idle
/// slots.
pub(super) fn gemma4_resume_candidate<'a>(
    retained: &'a [u32],
    anchor: Option<&'a Gemma4PromptAnchor>,
    vision_fingerprint: Option<[u8; 32]>,
) -> (&'a [u32], Option<&'a [u32]>) {
    let identity_matches =
        anchor.is_some_and(|anchor| anchor.vision_fingerprint == vision_fingerprint);
    (
        if identity_matches { retained } else { &[] },
        anchor
            .filter(|_| identity_matches)
            .map(|anchor| anchor.prompt_tokens.as_slice()),
    )
}

/// The longest-prefix candidate among `(index, retained, anchor)` ledgers,
/// by slot affinity's rule. Candidates come newest first, so equal matches
/// resolve to the newest. The returned preference's `slot` is meaningless
/// (it carries the index); the caller substitutes the granted slot.
pub(super) fn gemma4_best_ledger_match<'a>(
    candidates: impl IntoIterator<Item = (usize, &'a [u32], Option<&'a [u32]>)>,
    prompt_tokens: &[u32],
) -> Option<(usize, Gemma4SlotPreference)> {
    let preference = gemma4_slot_affinity(
        candidates
            .into_iter()
            .map(|(index, retained, anchor)| (index, retained, anchor, None, false, false)),
        prompt_tokens,
    )?
    .preference();
    Some((preference.slot.0 as usize, preference))
}

/// Host-side bytes of a ledger, charged to the spill budget with the cache
/// copy so the store bound covers everything an entry keeps alive.
fn ledger_bytes(retained: &[u32], anchor: Option<&Gemma4PromptAnchor>) -> usize {
    let token = std::mem::size_of::<u32>();
    retained.len() * token
        + anchor.map_or(0, |anchor| {
            anchor.prompt_tokens.len() * token + anchor.kv.total_bytes()
        })
}

fn ledger_tokens(retained: &[u32], anchor: Option<&Gemma4PromptAnchor>) -> usize {
    retained
        .len()
        .max(anchor.map_or(0, |anchor| anchor.prompt_tokens.len()))
}

/// Global rows a copy of a slot keeps beyond its cursor: the published
/// anchor's physical boundary (graft plus prompt rows), which its in-place
/// restore reads from the append-only global layers.
fn anchor_rows_floor(anchor: Option<&Gemma4PromptAnchor>) -> usize {
    anchor.map_or(0, |anchor| anchor.kv.prompt_len())
}

/// Move the conversation that held `slot` into the store. A spill that
/// cannot be taken or stored is logged and the ledger is left in place; the
/// caller then discards it and the conversation goes cold, exactly as before
/// #727.
fn spill_gemma4_victim(
    kv: &GemmaSlotScaffolds<'_>,
    spills: &mut Gemma4SlotSpills,
    graft_len: usize,
    slot: SlotId,
    retained: &mut Vec<u32>,
    anchor: &mut Option<Gemma4PromptAnchor>,
) {
    let tokens = ledger_tokens(retained, anchor.as_ref());
    let rows_floor = anchor_rows_floor(anchor.as_ref());
    let kv_bytes = match kv.spill_bytes(spills.regime, slot, rows_floor) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(target: LOG_TARGET, slot = slot.0, tokens, "Gemma4 slot spill skipped: {error:#}");
            return;
        }
    };
    let bytes =
        u64::try_from(kv_bytes + ledger_bytes(retained, anchor.as_ref())).unwrap_or(u64::MAX);
    if !spills.store.admits(bytes) {
        tracing::debug!(
            target: LOG_TARGET,
            slot = slot.0,
            tokens,
            bytes,
            "Gemma4 slot spill not stored: entry exceeds the spill budget"
        );
        return;
    }
    let spilled = match kv.spill(spills.regime, slot, rows_floor) {
        Ok(spilled) => spilled,
        Err(error) => {
            tracing::warn!(target: LOG_TARGET, slot = slot.0, tokens, "Gemma4 slot spill skipped: {error:#}");
            return;
        }
    };
    let mut anchor = anchor.take();
    if let Some(anchor) = anchor.as_mut() {
        anchor.kv.release_quota();
    }
    let payload = Gemma4SlotSpill {
        source_slot: slot.0,
        kv: spilled,
        graft_len,
        retained: std::mem::take(retained),
        anchor,
    };
    match spills.store.insert(bytes, payload) {
        SpillInsert::Stored { stamp, evicted } => {
            for entry in evicted {
                tracing::debug!(
                    target: LOG_TARGET,
                    slot = entry.payload.source_slot,
                    stamp = entry.stamp,
                    tokens = ledger_tokens(&entry.payload.retained, entry.payload.anchor.as_ref()),
                    bytes = entry.bytes,
                    "Gemma4 slot spill evicted (oldest stamp)"
                );
            }
            tracing::debug!(
                target: LOG_TARGET,
                slot = slot.0,
                stamp,
                tokens,
                bytes,
                entries = spills.store.len(),
                resident_bytes = spills.store.resident_bytes(),
                "Gemma4 slot spilled"
            );
        }
        SpillInsert::Rejected(_) => unreachable!("admission was checked before the copy"),
    }
}

/// Spill the conversation of a slot an embedding or a cold cohort lane takes
/// over; the caller then clears the slot's ledger and resets it.
pub(super) fn spill_reclaimed_gemma4_slot(
    kv: &GemmaSlotScaffolds<'_>,
    spills: &mut Gemma4SlotSpills,
    graft_len: usize,
    slot: SlotId,
    retained: &mut Vec<u32>,
    anchor: &mut Option<Gemma4PromptAnchor>,
) {
    if anchor.is_some() {
        spill_gemma4_victim(kv, spills, graft_len, slot, retained, anchor);
    }
}

/// Where a restored conversation comes from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RestoreSource {
    /// A spill store index.
    Spill(usize),
    /// The request's own idle slot, which the scheduler did not grant.
    Slot(SlotId),
}

/// The restore and victim decision for one admission. `preferred` is slot
/// affinity's idle-slot choice for the request (the slot the scheduler was
/// asked for), `spill_match` the best stored conversation.
fn plan_gemma4_reclaim(
    slot: SlotId,
    preferred: Option<Gemma4SlotPreference>,
    spill_match: Option<(usize, Gemma4SlotPreference)>,
    prompt_tokens: &[u32],
    victim_retained: &[u32],
    victim_has_anchor: bool,
) -> (Option<(RestoreSource, Gemma4SlotPreference)>, VictimSpill) {
    let in_place = preferred.filter(|preference| preference.slot == slot);
    let relocation = preferred
        .filter(|preference| preference.slot != slot)
        .map(|preference| (RestoreSource::Slot(preference.slot), preference));
    let spill_match =
        spill_match.map(|(index, preference)| (RestoreSource::Spill(index), preference));
    let candidate = match (relocation, spill_match) {
        (Some(own), Some(spilled)) if spilled.1.cached_tokens > own.1.cached_tokens => {
            Some(spilled)
        }
        (Some(own), _) => Some(own),
        (None, spilled) => spilled,
    };
    let plan = plan_reclaim(
        in_place,
        candidate,
        prompt_tokens,
        victim_retained,
        victim_has_anchor,
    );
    let victim = match plan.victim {
        // See the module documentation: no copy on an in-place anchor rewind.
        VictimSpill::Copy => VictimSpill::Keep,
        victim => victim,
    };
    (plan.restore, victim)
}

/// Generation admission's reclaim step, run once the scheduler granted
/// `slot` for a request whose idle-slot preference was `preferred`. Spills
/// the slot's previous conversation when this admission discards it,
/// restores the best ledger held elsewhere (a spill, or the request's own
/// ungranted slot) when it resumes more of the prompt than the slot does in
/// place, and returns the resume preference for the granted slot.
///
/// An `Err` means a restore may have written part of the slot; the caller
/// must reset the slot and clear its ledger.
#[allow(clippy::too_many_arguments)]
pub(super) fn reclaim_gemma4_slot(
    kv: &mut GemmaSlotScaffolds<'_>,
    spills: &mut Gemma4SlotSpills,
    graft_len: usize,
    slot: SlotId,
    retained_tokens: &mut [Vec<u32>],
    prompt_anchors: &mut [Option<Gemma4PromptAnchor>],
    preferred: Option<Gemma4SlotPreference>,
    prompt_tokens: &[u32],
    vision_fingerprint: Option<[u8; 32]>,
) -> anyhow::Result<Option<Gemma4SlotPreference>> {
    let slot_idx = slot.0 as usize;
    anyhow::ensure!(
        slot_idx < retained_tokens.len() && slot_idx < prompt_anchors.len(),
        "Gemma4 reclaim: slot {} outside the worker's {} slots",
        slot.0,
        retained_tokens.len()
    );
    let (restore, victim) = plan_gemma4_reclaim(
        slot,
        preferred,
        spills.best_match(prompt_tokens, vision_fingerprint, &[]),
        prompt_tokens,
        &retained_tokens[slot_idx],
        prompt_anchors[slot_idx].is_some(),
    );
    // Take the matched entry before spilling the victim so the victim's
    // insertion can never evict it.
    let restore = restore.map(|(source, preference)| {
        let source = match source {
            RestoreSource::Spill(index) => Ok(spills.store.take(index)),
            RestoreSource::Slot(own) => Err(own),
        };
        (source, preference)
    });
    if victim == VictimSpill::Move {
        spill_gemma4_victim(
            kv,
            spills,
            graft_len,
            slot,
            &mut retained_tokens[slot_idx],
            &mut prompt_anchors[slot_idx],
        );
    }
    let Some((source, preference)) = restore else {
        return Ok(preferred.filter(|preference| preference.slot == slot));
    };
    match source {
        Ok(entry) => {
            let tokens = ledger_tokens(&entry.payload.retained, entry.payload.anchor.as_ref());
            anyhow::ensure!(
                entry.payload.graft_len == graft_len,
                "Gemma4 slot spill holds {} graft rows, the worker binds {graft_len}",
                entry.payload.graft_len
            );
            let mut anchor = entry.payload.anchor;
            if let Some(anchor) = anchor.as_mut() {
                let hybrid = kv.hybrid.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("Gemma4 spilled prompt anchor requires the hybrid KV scaffold")
                })?;
                anchor
                    .kv
                    .reacquire_quota(hybrid, slot)
                    .context("Gemma4 slot spill restore: anchor quota")?;
            }
            kv.restore(slot, &entry.payload.kv).with_context(|| {
                format!(
                    "Gemma4 slot spill restore into slot {} (spilled from slot {})",
                    slot.0, entry.payload.source_slot
                )
            })?;
            retained_tokens[slot_idx] = entry.payload.retained;
            prompt_anchors[slot_idx] = anchor;
            tracing::debug!(
                target: LOG_TARGET,
                slot = slot.0,
                from_slot = entry.payload.source_slot,
                stamp = entry.stamp,
                tokens,
                bytes = entry.bytes,
                cached_tokens = preference.cached_tokens,
                anchor = preference.restore_prompt_anchor,
                "Gemma4 slot spill restored"
            );
        }
        Err(own) => {
            let own_idx = own.0 as usize;
            let rows_floor = anchor_rows_floor(prompt_anchors[own_idx].as_ref());
            kv.copy_slot(spills.regime, own, slot, rows_floor)
                .with_context(|| {
                    format!(
                        "Gemma4 slot move from its own slot {} into granted slot {}",
                        own.0, slot.0
                    )
                })?;
            retained_tokens[slot_idx] = std::mem::take(&mut retained_tokens[own_idx]);
            prompt_anchors[slot_idx] = prompt_anchors[own_idx].take();
            tracing::debug!(
                target: LOG_TARGET,
                slot = slot.0,
                from_slot = own.0,
                cached_tokens = preference.cached_tokens,
                anchor = preference.restore_prompt_anchor,
                "Gemma4 conversation moved into its granted slot (own slot not granted)"
            );
        }
    }
    Ok(Some(Gemma4SlotPreference { slot, ..preference }))
}

#[cfg(test)]
#[path = "gemma4_slot_spill_tests.rs"]
mod tests;
