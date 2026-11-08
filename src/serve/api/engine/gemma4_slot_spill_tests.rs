//! Issue #727 (Gemma): admission decides spill, restore and own-slot moves
//! by longest prefix; a reclaimed conversation is restored bitwise into
//! whatever slot it is granted; the store accounts cache and ledger bytes and
//! parks anchors without an anchor-quota lease.

use mlx_native::MlxDevice;

use super::super::slot_spill::VictimSpill;
use super::super::{Gemma4PromptAnchor, Gemma4SlotPreference, SlotId};
use super::{
    gemma4_best_ledger_match, gemma4_resume_candidate, plan_gemma4_reclaim, reclaim_gemma4_slot,
    spill_reclaimed_gemma4_slot, Gemma4SlotSpills, RestoreSource,
};
use crate::inference::models::gemma4::kv_cache::slot_spill::{
    GemmaKvSlotSpill, GemmaSlotKvRegime, GemmaSlotScaffolds,
};
use crate::inference::models::gemma4::kv_cache::{
    alloc_hb_kv_for_layer, alloc_multi_seq_hybrid_kv_for_layer, anchor_budget,
    snapshot_gemma_hybrid_slot_anchor, MultiSeqHbKvBuffers, MultiSeqHybridKvBuffers,
};

const SLOTS: usize = 4;
const REGIME: GemmaSlotKvRegime = GemmaSlotKvRegime::Hybrid;

fn preference(slot: u32, cached_tokens: usize, anchor: bool) -> Gemma4SlotPreference {
    Gemma4SlotPreference {
        slot: SlotId(slot),
        cached_tokens,
        restore_prompt_anchor: anchor,
    }
}

#[test]
fn gemma_reclaim_plan_restores_only_longer_ledgers_and_never_copies() {
    let occupant: Vec<u32> = (0..30).collect();
    let mut fork: Vec<u32> = occupant[..12].to_vec();
    fork.extend([700, 701]);
    let other: Vec<u32> = (500..520).collect();
    let granted = SlotId(1);

    // Cold admission into an anchored slot moves its conversation out.
    let (restore, victim) = plan_gemma4_reclaim(granted, None, None, &other, &occupant, true);
    assert_eq!((restore, victim), (None, VictimSpill::Move));
    // A slot without an anchor holds nothing resumable: nothing to spill.
    let (_, victim) = plan_gemma4_reclaim(granted, None, None, &other, &occupant, false);
    assert_eq!(victim, VictimSpill::Keep);
    // An in-place live resume keeps the slot.
    let live = preference(1, 30, false);
    let (restore, victim) =
        plan_gemma4_reclaim(granted, Some(live), None, &occupant, &occupant, true);
    assert_eq!((restore, victim), (None, VictimSpill::Keep));
    // An in-place anchor rewind that diverges exactly at the anchor is not
    // copied: a full-input checkpoint ends where the next turn diverges.
    let anchor = preference(1, 12, true);
    let (restore, victim) =
        plan_gemma4_reclaim(granted, Some(anchor), None, &fork, &occupant, true);
    assert_eq!((restore, victim), (None, VictimSpill::Keep));

    // A spill restores only when it resumes strictly more.
    let (restore, victim) = plan_gemma4_reclaim(
        granted,
        Some(anchor),
        Some((3, preference(3, 16, false))),
        &fork,
        &occupant,
        true,
    );
    assert_eq!(
        restore,
        Some((RestoreSource::Spill(3), preference(3, 16, false)))
    );
    assert_eq!(victim, VictimSpill::Move);
    let (restore, _) = plan_gemma4_reclaim(
        granted,
        Some(live),
        Some((3, preference(3, 30, false))),
        &occupant,
        &occupant,
        true,
    );
    assert_eq!(restore, None);

    // The request's own idle slot was not granted: it moves into the
    // granted slot, unless a spill resumes strictly more.
    let own = preference(0, 20, true);
    let (restore, victim) = plan_gemma4_reclaim(granted, Some(own), None, &fork, &occupant, true);
    assert_eq!(restore, Some((RestoreSource::Slot(SlotId(0)), own)));
    assert_eq!(victim, VictimSpill::Move);
    let (restore, _) = plan_gemma4_reclaim(
        granted,
        Some(own),
        Some((2, preference(2, 20, false))),
        &fork,
        &[],
        false,
    );
    assert_eq!(
        restore,
        Some((RestoreSource::Slot(SlotId(0)), own)),
        "ties keep the own slot"
    );
    let (restore, victim) = plan_gemma4_reclaim(
        granted,
        Some(own),
        Some((2, preference(2, 21, false))),
        &fork,
        &[],
        false,
    );
    assert_eq!(
        restore,
        Some((RestoreSource::Spill(2), preference(2, 21, false)))
    );
    assert_eq!(
        victim,
        VictimSpill::Keep,
        "an empty granted slot spills nothing"
    );
}

#[test]
fn gemma_ledger_match_binds_the_retained_prefix_to_its_anchor_identity() {
    let prompt: Vec<u32> = (1..=20).collect();
    // Newest first: index 2 is newest.
    let candidates = [
        (2, &prompt[..8], Some(&prompt[..4])),
        (1, &prompt[..12], Some(&prompt[..4])),
        (0, &prompt[..12], Some(&prompt[..4])),
    ];
    let (index, best) = gemma4_best_ledger_match(candidates, &prompt).expect("match");
    assert_eq!(
        (index, best.cached_tokens),
        (1, 12),
        "longest, newest on ties"
    );
    assert!(!best.restore_prompt_anchor);
    // An exact retry of the anchor replays its last token.
    let (_, best) =
        gemma4_best_ledger_match([(0, &[][..], Some(&prompt[..]))], &prompt).expect("retry");
    assert_eq!((best.cached_tokens, best.restore_prompt_anchor), (19, true));
    assert!(gemma4_best_ledger_match([(0, &prompt[..], None)], &prompt).is_none());

    // Without an anchor (or with another vision identity) the retained
    // prefix is not offered, exactly as for an idle slot.
    let (retained, anchor) = gemma4_resume_candidate(&prompt[..12], None, None);
    assert!(retained.is_empty() && anchor.is_none());
}

struct Kv {
    hb: Vec<MultiSeqHbKvBuffers>,
    hybrid: Vec<MultiSeqHybridKvBuffers>,
}

impl Kv {
    fn new(device: &MlxDevice) -> Self {
        let shapes = [(64usize, false), (8usize, true)];
        Self {
            hb: shapes
                .iter()
                .enumerate()
                .map(|(layer, &(cap, ring))| {
                    alloc_hb_kv_for_layer(device, layer, 2, 16, cap, ring, SLOTS as u32).unwrap()
                })
                .collect(),
            hybrid: shapes
                .iter()
                .enumerate()
                .map(|(layer, &(cap, ring))| {
                    alloc_multi_seq_hybrid_kv_for_layer(
                        device,
                        layer,
                        2,
                        16,
                        cap,
                        ring,
                        SLOTS as u32,
                    )
                    .unwrap()
                })
                .collect(),
        }
    }

    fn view(&mut self) -> GemmaSlotScaffolds<'_> {
        GemmaSlotScaffolds {
            hb: &mut self.hb,
            hybrid: Some(&mut self.hybrid),
            dense: None,
            nibble: None,
        }
    }

    /// Plant a seed-distinct pattern in `slot`'s region of every hybrid
    /// buffer and set its cursors.
    fn plant(&mut self, slot: usize, seed: usize, cursor: u32) {
        for (layer_index, layer) in self.hybrid.iter_mut().enumerate() {
            for (buffer_index, buffer) in [&mut layer.k, &mut layer.v_packed, &mut layer.v_norms]
                .into_iter()
                .enumerate()
            {
                let bytes = buffer.as_mut_slice::<u8>().unwrap();
                let per_slot = bytes.len() / SLOTS;
                for (offset, byte) in bytes[slot * per_slot..(slot + 1) * per_slot]
                    .iter_mut()
                    .enumerate()
                {
                    *byte = ((offset * 7 + seed * 131 + layer_index * 11 + buffer_index * 3) % 253)
                        as u8;
                }
            }
            layer.seq_lens[slot] = cursor;
        }
        for layer in &mut self.hb {
            layer.seq_lens[slot] = cursor;
        }
    }

    fn image(&mut self, slot: usize, rows_floor: usize) -> GemmaKvSlotSpill {
        self.view()
            .spill(REGIME, SlotId(slot as u32), rows_floor)
            .unwrap()
    }
}

struct Ledgers {
    retained: Vec<Vec<u32>>,
    anchors: Vec<Option<Gemma4PromptAnchor>>,
}

impl Ledgers {
    fn new() -> Self {
        Self {
            retained: vec![Vec::new(); SLOTS],
            anchors: (0..SLOTS).map(|_| None).collect(),
        }
    }
}

/// Conversation in `slot`: an anchor at `anchor_len` tokens (captured from
/// the slot's rings), then a live tail up to `tokens.len()`.
fn seat(
    kv: &mut Kv,
    ledgers: &mut Ledgers,
    slot: usize,
    tokens: &[u32],
    anchor_len: usize,
    seed: usize,
) {
    kv.plant(slot, seed, anchor_len as u32);
    let anchor =
        snapshot_gemma_hybrid_slot_anchor(&kv.hybrid, SlotId(slot as u32), anchor_len).unwrap();
    kv.plant(slot, seed + 1000, tokens.len() as u32);
    ledgers.retained[slot] = tokens.to_vec();
    ledgers.anchors[slot] = Some(Gemma4PromptAnchor {
        prompt_tokens: tokens[..anchor_len].to_vec(),
        kv: anchor,
        vision_fingerprint: None,
    });
}

#[allow(clippy::too_many_arguments)]
fn admit(
    kv: &mut Kv,
    spills: &mut Gemma4SlotSpills,
    ledgers: &mut Ledgers,
    granted: usize,
    preferred: Option<Gemma4SlotPreference>,
    prompt: &[u32],
) -> Option<Gemma4SlotPreference> {
    reclaim_gemma4_slot(
        &mut kv.view(),
        spills,
        0,
        SlotId(granted as u32),
        &mut ledgers.retained,
        &mut ledgers.anchors,
        preferred,
        prompt,
        None,
    )
    .expect("reclaim")
}

#[test]
fn gemma_reclaimed_conversation_is_spilled_then_restored_bitwise_into_its_granted_slot() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    crate::debug::investigation_env::gemma_full_input::set_worker_active(true);
    let device = MlxDevice::new().expect("Metal device");
    let mut kv = Kv::new(&device);
    let mut ledgers = Ledgers::new();
    let mut spills = Gemma4SlotSpills::new(1 << 30, SLOTS, REGIME);

    let a: Vec<u32> = (1..=24).collect();
    let b: Vec<u32> = (100..130).collect();
    seat(&mut kv, &mut ledgers, 1, &a, 10, 1);
    seat(&mut kv, &mut ledgers, 3, &b, 12, 2);
    let a_image = kv.image(1, 10);
    let quota_before = anchor_budget::used_bytes();
    assert!(ledgers.anchors[1].as_ref().unwrap().kv.holds_quota());

    // A cold request granted slot 1 moves A into the store.
    let cold: Vec<u32> = (900..940).collect();
    assert_eq!(
        admit(&mut kv, &mut spills, &mut ledgers, 1, None, &cold),
        None
    );
    assert_eq!(spills.len(), 1);
    assert!(ledgers.retained[1].is_empty() && ledgers.anchors[1].is_none());
    let a_ledger = (a.len() + 10) * 4 + {
        let entry = spills.store.newest_first().next().unwrap().1;
        entry.payload.anchor.as_ref().unwrap().kv.total_bytes()
    };
    assert_eq!(
        spills.resident_bytes(),
        (a_image.total_bytes() + a_ledger) as u64,
        "cache rows, cursors, retained tokens, anchor tokens and saved rings"
    );
    let parked = &spills.store.newest_first().next().unwrap().1.payload;
    assert!(
        !parked.anchor.as_ref().unwrap().kv.holds_quota(),
        "parked anchors hold no lease"
    );
    assert!(anchor_budget::used_bytes() < quota_before);
    // The cold request's prefill overwrites slot 1.
    kv.plant(1, 77, 40);

    // A's live-tail follow-up is granted slot 3 (B's): B is spilled, A is
    // restored and resumes its whole retained prefix.
    let mut follow_up = a.clone();
    follow_up.extend([501, 502, 503]);
    let b_image = kv.image(3, 12);
    let resumed = admit(&mut kv, &mut spills, &mut ledgers, 3, None, &follow_up).expect("restored");
    assert_eq!(resumed, preference(3, a.len(), false));
    assert!(kv.image(3, 10) == a_image, "A's rows and cursors in slot 3");
    assert_eq!(ledgers.retained[3], a);
    assert_eq!(ledgers.anchors[3].as_ref().unwrap().prompt_tokens, a[..10]);
    assert!(
        ledgers.anchors[3].as_ref().unwrap().kv.holds_quota(),
        "lease reacquired"
    );
    assert_eq!(spills.len(), 1, "A left the store, B entered it");

    // B's anchor follow-up (it shares only B's anchor) is granted the empty
    // slot 2: B is restored and rewinds to its anchor.
    let mut b_anchor_turn = b[..12].to_vec();
    b_anchor_turn.extend([7, 8, 9]);
    let resumed =
        admit(&mut kv, &mut spills, &mut ledgers, 2, None, &b_anchor_turn).expect("restored");
    assert_eq!(resumed, preference(2, 12, true));
    assert!(kv.image(2, 12) == b_image, "B's rows and cursors in slot 2");
    assert_eq!(spills.len(), 0);

    // A's own next turn in its own slot copies nothing.
    let resumed = admit(
        &mut kv,
        &mut spills,
        &mut ledgers,
        3,
        Some(preference(3, a.len(), false)),
        &follow_up,
    );
    assert_eq!(resumed, Some(preference(3, a.len(), false)));
    assert_eq!(spills.len(), 0);
    crate::debug::investigation_env::gemma_full_input::set_worker_active(false);
}

#[test]
fn gemma_conversation_whose_own_slot_is_not_granted_moves_in_place() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let mut kv = Kv::new(&device);
    let mut ledgers = Ledgers::new();
    let mut spills = Gemma4SlotSpills::new(1 << 30, SLOTS, REGIME);
    let a: Vec<u32> = (1..=20).collect();
    let c: Vec<u32> = (300..330).collect();
    seat(&mut kv, &mut ledgers, 0, &a, 8, 5);
    seat(&mut kv, &mut ledgers, 2, &c, 9, 6);
    let a_image = kv.image(0, 8);
    let c_image = kv.image(2, 9);
    let mut follow_up = a.clone();
    follow_up.push(42);

    // A preferred slot 0; the shared KV budget refused it and slot 2 was
    // granted. C is spilled, A moves 0 -> 2 and resumes its live tail.
    let resumed = admit(
        &mut kv,
        &mut spills,
        &mut ledgers,
        2,
        Some(preference(0, a.len(), false)),
        &follow_up,
    )
    .expect("moved");
    assert_eq!(resumed, preference(2, a.len(), false));
    assert!(kv.image(2, 8) == a_image, "A's rows and cursors in slot 2");
    assert!(kv.image(0, 8) == a_image, "the source slot keeps its bytes");
    assert!(ledgers.retained[0].is_empty() && ledgers.anchors[0].is_none());
    assert_eq!(ledgers.retained[2], a);
    assert_eq!(spills.len(), 1, "C was spilled");

    // C returns to slot 1 and is restored bitwise.
    let mut c_turn = c.clone();
    c_turn.push(43);
    let resumed = admit(&mut kv, &mut spills, &mut ledgers, 1, None, &c_turn).expect("restored");
    assert_eq!(resumed, preference(1, c.len(), false));
    assert!(kv.image(1, 9) == c_image);
}

#[test]
fn gemma_spill_above_the_budget_is_not_stored_and_the_victim_goes_cold() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let mut kv = Kv::new(&device);
    let mut ledgers = Ledgers::new();
    let mut spills = Gemma4SlotSpills::new(64, SLOTS, REGIME);
    let a: Vec<u32> = (1..=20).collect();
    seat(&mut kv, &mut ledgers, 1, &a, 8, 3);
    let cold: Vec<u32> = (900..930).collect();
    assert_eq!(
        admit(&mut kv, &mut spills, &mut ledgers, 1, None, &cold),
        None
    );
    assert_eq!(spills.len(), 0);
    assert_eq!(
        ledgers.retained[1], a,
        "the caller discards the ledger, as before #727"
    );

    // An embedding (or a cold cohort lane) taking a slot spills it too.
    let mut spills = Gemma4SlotSpills::new(1 << 30, SLOTS, REGIME);
    spill_reclaimed_gemma4_slot(
        &kv.view(),
        &mut spills,
        0,
        SlotId(1),
        &mut ledgers.retained[1],
        &mut ledgers.anchors[1],
    );
    assert_eq!(spills.len(), 1);
    assert!(ledgers.anchors[1].is_none());
    // A disabled store (budget 0) never stores.
    let mut disabled = Gemma4SlotSpills::new(0, SLOTS, REGIME);
    seat(&mut kv, &mut ledgers, 1, &a, 8, 3);
    assert_eq!(
        admit(&mut kv, &mut disabled, &mut ledgers, 1, None, &cold),
        None
    );
    assert_eq!(disabled.len(), 0);
    assert_eq!(disabled.resume_tokens(&a, None), 0);
}
