//! Issue #727: admission decides spill/restore by longest prefix, and a
//! reclaimed conversation is restored bitwise into whatever slot it is
//! granted on its return. The shared store's own tests are in
//! `slot_spill_tests.rs`.

use mlx_native::MlxDevice;

use super::super::slot_spill::{plan_reclaim, ReclaimPlan, SlotSpillStore, VictimSpill};
use super::super::{Qwen35PromptAnchor, Qwen35RetainedPrefix, Qwen35SlotPreference, SlotId};
use super::{
    qwen35_best_ledger_match, reclaim_qwen35_slot, spill_reclaimed_qwen35_slot,
    Qwen35SlotSpillStore,
};
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::{default_layer_types, Qwen35Config, Qwen35Variant};

fn preference(cached_tokens: usize, anchor: bool) -> Qwen35SlotPreference {
    Qwen35SlotPreference {
        slot: SlotId(0),
        cached_tokens,
        restore_prompt_anchor: anchor,
        use_prefill_logits: false,
    }
}

/// The shared reclaim plan at Qwen's types.
fn qwen_plan(
    in_place: Option<Qwen35SlotPreference>,
    spill_match: Option<(usize, Qwen35SlotPreference)>,
    prompt_tokens: &[u32],
    victim_retained: &[u32],
    victim_has_ledger: bool,
) -> ReclaimPlan<usize, Qwen35SlotPreference> {
    plan_reclaim(
        in_place,
        spill_match,
        prompt_tokens,
        victim_retained,
        victim_has_ledger,
    )
}

#[test]
fn reclaim_plan_spills_only_discarded_state_and_restores_only_longer_matches() {
    let live = preference(30, false);
    let anchor = preference(12, true);
    let spill = |tokens| Some((3, preference(tokens, false)));
    // The occupant: stable anchor 0..12, then the assistant header 12..14,
    // reasoning and answer up to 30.
    let occupant: Vec<u32> = (0..30).collect();
    let other: Vec<u32> = (500..520).collect();
    let mut continuation: Vec<u32> = occupant[..14].to_vec();
    continuation.extend([900, 901, 902]);
    let mut fork: Vec<u32> = occupant[..12].to_vec();
    fork.extend([700, 701]);
    let mut extended = occupant.clone();
    extended.push(31);

    // Cold admission into an occupied slot: move its ledger into the store.
    let plan = qwen_plan(None, None, &other, &occupant, true);
    assert_eq!((plan.restore, plan.victim), (None, VictimSpill::Move));
    // Cold admission into an empty slot spills nothing.
    let plan = qwen_plan(None, None, &other, &[], false);
    assert_eq!(plan.victim, VictimSpill::Keep);
    // A live-tail resume keeps the slot's conversation.
    let plan = qwen_plan(Some(live), None, &extended, &occupant, true);
    assert_eq!((plan.restore, plan.victim), (None, VictimSpill::Keep));
    // The occupant's own next turn or retry rewinds to its stable anchor but
    // shares the assistant header beyond it: no copy.
    let plan = qwen_plan(Some(anchor), None, &continuation, &occupant, true);
    assert_eq!(plan.victim, VictimSpill::Keep);
    // A request that starts new content exactly at the anchor (a fork, or
    // another conversation sharing the anchor) would discard the occupant's
    // tail: copy the ledger, the anchor stays in place for this request.
    let plan = qwen_plan(Some(anchor), None, &fork, &occupant, true);
    assert_eq!(plan.victim, VictimSpill::Copy);
    // No tail beyond the anchor: nothing to lose.
    let plan = qwen_plan(Some(anchor), None, &fork, &occupant[..12], true);
    assert_eq!(plan.victim, VictimSpill::Keep);

    // A spill restores only when it resumes strictly more tokens.
    let plan = qwen_plan(Some(anchor), spill(16), &continuation, &occupant, true);
    assert_eq!(plan.restore, spill(16));
    assert_eq!(plan.victim, VictimSpill::Move);
    // A spill matching no more than the live tail (30 tokens) stays stored.
    let plan = qwen_plan(Some(live), spill(30), &extended, &occupant, true);
    assert_eq!((plan.restore, plan.victim), (None, VictimSpill::Keep));
    let plan = qwen_plan(None, spill(8), &other, &[], false);
    assert_eq!((plan.restore, plan.victim), (spill(8), VictimSpill::Keep));
}

#[test]
fn ledger_match_is_longest_prefix_then_newest() {
    let prompt: Vec<u32> = (1..=20).collect();
    let other: Vec<u32> = (100..120).collect();
    // Newest first: index 2 is newest.
    let candidates = [
        (2, &prompt[..8], Some(&prompt[..4])),
        (1, &prompt[..12], Some(&prompt[..4])),
        (0, &prompt[..12], Some(&prompt[..4])),
    ];
    let (index, best) = qwen35_best_ledger_match(candidates, &prompt).expect("match");
    assert_eq!((index, best.cached_tokens), (1, 12), "longest, newest on ties");
    assert!(!best.restore_prompt_anchor);

    // A full-prompt anchor beats a shorter live tail and reuses its logits.
    let candidates = [
        (0, &prompt[..15], None),
        (1, &other[..], Some(&prompt[..])),
    ];
    let (index, best) = qwen35_best_ledger_match(candidates, &prompt).expect("match");
    assert_eq!((index, best.cached_tokens), (1, 20));
    assert!(best.restore_prompt_anchor && best.use_prefill_logits);

    // A live tail must be a strict prefix; unrelated ledgers never match.
    assert!(qwen35_best_ledger_match([(0, &prompt[..], None)], &prompt).is_none());
    assert!(qwen35_best_ledger_match([(0, &other[..5], Some(&other[..3]))], &prompt).is_none());
}

/// Two full-attention layers, two DeltaNet layers, no MTP.
fn tiny_cfg() -> Qwen35Config {
    Qwen35Config {
        variant: Qwen35Variant::Dense,
        hidden_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 32,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 2,
        layer_types: default_layer_types(4, 2),
        partial_rotary_factor: 0.25,
        rope_theta: 1e7,
        rotary_dim: 8,
        mrope_section: [2, 2, 2, 2],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 4096,
        vocab_size: 256,
        attn_output_gate: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: true,
        intermediate_size: Some(128),
        moe: None,
    }
}

fn set_cursor(cache: &mut HybridKvCache, slot: SlotId, cursor: u32) {
    for block in &mut cache.full_attn {
        block.current_len[slot.0 as usize] = cursor;
    }
}

/// Fill `slot`'s region of every buffer with a seed-distinct pattern.
fn plant(cache: &mut HybridKvCache, slot: SlotId, seed: usize) {
    let n_seqs = cache.n_seqs as usize;
    let fill = |buffer: &mut mlx_native::MlxBuffer, index: usize| {
        let region = buffer.byte_len() / n_seqs;
        let start = slot.0 as usize * region;
        let bytes = buffer.as_mut_slice::<u8>().expect("cache bytes");
        for (offset, byte) in bytes[start..start + region].iter_mut().enumerate() {
            *byte = ((seed * 29 + index * 13 + offset) % 251) as u8;
        }
    };
    let mut index = 0;
    for block in &mut cache.full_attn {
        let tq = block.tq.as_mut().expect("TQ cache");
        for buffer in [
            &mut tq.k_packed,
            &mut tq.k_norms,
            &mut tq.v_packed,
            &mut tq.v_norms,
        ] {
            fill(buffer, index);
            index += 1;
        }
    }
    for linear in &mut cache.linear_attn {
        for buffer in [
            &mut linear.conv_state,
            &mut linear.conv_state_scratch,
            &mut linear.recurrent,
            &mut linear.recurrent_scratch,
        ] {
            fill(buffer, index);
            index += 1;
        }
        linear.pp_flipped[slot.0 as usize] = seed % 2 == 1;
    }
}

/// Conversation A in `slot`: an anchor at `anchor_len` tokens, then a live
/// tail up to `tokens.len()`.
fn seat_conversation(
    cache: &mut HybridKvCache,
    slot: SlotId,
    tokens: &[u32],
    anchor_len: usize,
    seed: usize,
) -> (Qwen35RetainedPrefix, Option<Qwen35PromptAnchor>) {
    plant(cache, slot, seed);
    set_cursor(cache, slot, anchor_len as u32);
    let kv = cache
        .snapshot_slot_anchor(slot, anchor_len)
        .expect("anchor checkpoint");
    set_cursor(cache, slot, tokens.len() as u32);
    (
        Qwen35RetainedPrefix {
            tokens: tokens.to_vec(),
            spec: None,
        },
        Some(Qwen35PromptAnchor {
            prompt_tokens: tokens[..anchor_len].to_vec(),
            kv,
            prefill_logits: vec![0.25; 8],
            vision_fingerprint: None,
            spec: None,
            forced_close: None,
        }),
    )
}

#[test]
fn reclaimed_conversation_is_spilled_then_restored_bitwise_into_its_granted_slot() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let mut cache = HybridKvCache::new_with_options(&tiny_cfg(), &device, 32, 3, true)
        .expect("three-slot cache");
    let mut retained: Vec<Qwen35RetainedPrefix> =
        (0..3).map(|_| Qwen35RetainedPrefix::default()).collect();
    let mut anchors: Vec<Option<Qwen35PromptAnchor>> = (0..3).map(|_| None).collect();
    let mut store: Qwen35SlotSpillStore = SlotSpillStore::new(1 << 30, 8);

    let a: Vec<u32> = (1..=10).collect();
    (retained[0], anchors[0]) = seat_conversation(&mut cache, SlotId(0), &a, 6, 1);
    let a_state = cache.spill_slot(SlotId(0)).expect("reference copy of A");

    // A cold request C is granted A's idle slot: A moves into the store.
    let c = [50u32, 51, 52];
    let granted = reclaim_qwen35_slot(
        &mut cache,
        &mut store,
        SlotId(0),
        &mut retained[0],
        &mut anchors[0],
        None,
        &c,
        None,
        None,
    )
    .expect("cold reclaim");
    assert_eq!(granted, None, "C stays cold");
    assert_eq!(store.len(), 1);
    assert!(retained[0].is_empty() && anchors[0].is_none());

    // C runs in slot 0 and overwrites it.
    cache.reset_for_slot(SlotId(0)).expect("cold reset");
    plant(&mut cache, SlotId(0), 2);
    set_cursor(&mut cache, SlotId(0), 4);
    retained[0].tokens = vec![50, 51, 52, 53];

    // A returns with one more turn and is granted the empty slot 2.
    let mut a_turn: Vec<u32> = a.clone();
    a_turn.extend([11, 12]);
    let granted = reclaim_qwen35_slot(
        &mut cache,
        &mut store,
        SlotId(2),
        &mut retained[2],
        &mut anchors[2],
        None,
        &a_turn,
        None,
        None,
    )
    .expect("restore A");
    assert_eq!(
        granted,
        Some(Qwen35SlotPreference {
            slot: SlotId(2),
            cached_tokens: 10,
            restore_prompt_anchor: false,
            use_prefill_logits: false,
        }),
        "A resumes its live tail in slot 2"
    );
    assert!(store.is_empty(), "a restored spill leaves the store");
    assert_eq!(retained[2].tokens, a);
    assert_eq!(
        anchors[2].as_ref().map(|anchor| anchor.prompt_tokens.as_slice()),
        Some(&a[..6])
    );
    assert!(
        cache.spill_slot(SlotId(2)).expect("restored A") == a_state,
        "slot 2 holds A's state bitwise"
    );
    assert_eq!(retained[0].tokens, vec![50, 51, 52, 53], "C is untouched");

    // D shares A's anchor and rewinds slot 2 to it: A's tail is copied out
    // while the anchor stays in place for D.
    let mut d = a[..6].to_vec();
    d.push(99);
    let in_place = Some(Qwen35SlotPreference {
        slot: SlotId(2),
        cached_tokens: 6,
        restore_prompt_anchor: true,
        use_prefill_logits: false,
    });
    let granted = reclaim_qwen35_slot(
        &mut cache,
        &mut store,
        SlotId(2),
        &mut retained[2],
        &mut anchors[2],
        in_place,
        &d,
        None,
        None,
    )
    .expect("anchor reclaim");
    assert_eq!(granted, in_place);
    assert_eq!(store.len(), 1);
    assert_eq!(retained[2].tokens, a, "the slot ledger is unchanged");
    assert!(anchors[2].is_some());

    // A's own next turn also rewinds slot 2 to the anchor, but it shares the
    // tokens after it (the assistant header): no copy is taken.
    let mut continuation = a[..8].to_vec();
    continuation.push(77);
    let granted = reclaim_qwen35_slot(
        &mut cache,
        &mut store,
        SlotId(2),
        &mut retained[2],
        &mut anchors[2],
        in_place,
        &continuation,
        None,
        None,
    )
    .expect("continuation keeps its slot");
    assert_eq!(granted, in_place);
    assert_eq!(store.len(), 1, "a continuation copies nothing");

    // A returns again, granted slot 1: its copied tail is restored.
    a_turn.push(13);
    let granted = reclaim_qwen35_slot(
        &mut cache,
        &mut store,
        SlotId(1),
        &mut retained[1],
        &mut anchors[1],
        None,
        &a_turn,
        None,
        None,
    )
    .expect("restore A again");
    assert_eq!(granted.map(|preference| preference.cached_tokens), Some(10));
    assert!(store.is_empty());
    assert!(cache.spill_slot(SlotId(1)).expect("restored A") == a_state);

    // An embedding request takes slot 1 over: A is spilled again.
    spill_reclaimed_qwen35_slot(
        &cache,
        &mut store,
        SlotId(1),
        &mut retained[1],
        &mut anchors[1],
    );
    assert_eq!(store.len(), 1);
    assert!(retained[1].is_empty() && anchors[1].is_none());
}

#[test]
fn a_spill_above_the_budget_is_not_stored_and_the_victim_goes_cold() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let mut cache = HybridKvCache::new_with_options(&tiny_cfg(), &device, 32, 2, true)
        .expect("two-slot cache");
    let a: Vec<u32> = (1..=10).collect();
    let (mut retained, mut anchor) = seat_conversation(&mut cache, SlotId(0), &a, 6, 1);
    let mut store: Qwen35SlotSpillStore = SlotSpillStore::new(1024, 8);
    let granted = reclaim_qwen35_slot(
        &mut cache,
        &mut store,
        SlotId(0),
        &mut retained,
        &mut anchor,
        None,
        &[7, 7, 7],
        None,
        None,
    )
    .expect("cold reclaim");
    assert_eq!(granted, None);
    assert!(store.is_empty(), "a 1 KiB budget cannot hold the slot");
    // The ledger stays for admission's ordinary cold clear.
    assert_eq!(retained.tokens, a);
}
