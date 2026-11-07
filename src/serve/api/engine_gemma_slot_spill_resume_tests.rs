//! Issue #727 gates (official Gemma 4 artifact, Class A): a conversation
//! whose slot is reclaimed and later restored into another slot continues
//! bitwise as if it had kept its slot, and the same conversation is bitwise
//! equal in every slot.
//!
//! * `gemma_spilled_slot_resumes_bitwise_in_any_slot`:
//!   1. **Mid-request moves.** The target decodes 0, 1, 8 or 24 ticks in
//!      slot 2, is spilled (`GemmaSlotScaffolds::spill`), slot 2 is wiped, and
//!      the spill is restored into a destination holding another
//!      conversation; it finishes there. Reference: the same request in slot
//!      2 without a move.
//!   2. **Between turns, through the production reclaim**
//!      (`reclaim_gemma4_slot`). Turn 1 finishes in slot 2 and publishes its
//!      ledger as serving does (the input checkpoint at `P1` and the
//!      retained prefix). A cold request reclaims slot 2 (the conversation is
//!      spilled) and prefills over it; the follow-up is admitted into another
//!      slot and restored there:
//!      - **live tail:** the follow-up extends the retained prefix;
//!      - **anchor:** the follow-up shares only `P1`; the restored anchor
//!        rewinds the slot;
//!      - **own slot not granted:** the follow-up's own idle slot is refused
//!        and the conversation moves into the granted slot in place.
//!      Reference: the same follow-up in place in slot 2.
//!   3. **Joint-warm cohort lane.** The restored live-tail follow-up runs as
//!      a `PerSlotExact` joint warm lane beside two peers; reference: the
//!      same cohort with the target in slot 2.
//!
//!   Every comparison is `gemma_warm_batch_tests::assert_lane_equal`: first
//!   token, final logits bits, valid KV rows (global rows and live ring rows
//!   of K, V codes and V scales) and every scaffold cursor; decode ticks also
//!   compare their token and logits bits.
//! * `gemma_resume_is_slot_independent_bitwise`: the same two-turn
//!   conversation in slot 0 and in slots 1, 2 and 3 (peers occupied by other
//!   conversations) produces identical images and ticks.
//!
//! ```text
//! HF2Q_GEMMA4_GGUF=/abs/gemma-4-26B-A4B.gguf \
//!   cargo test --release --bin hf2q gemma_spilled_slot_resumes_bitwise_in_any_slot \
//!   -- --ignored --exact --nocapture --test-threads=1
//! HF2Q_GEMMA4_GGUF=/abs/gemma-4-26B-A4B.gguf \
//!   cargo test --release --bin hf2q gemma_resume_is_slot_independent_bitwise \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```

use super::gemma4_slot_spill::{reclaim_gemma4_slot, Gemma4SlotSpills};
use super::gemma_warm_batch_tests::{
    assert_lane_equal, lane_image, load_gate_model, synthetic_prompt, LaneImage, MAX_DECODE,
};
use super::*;
use crate::inference::models::gemma4::kv_cache::slot_spill::GemmaSlotKvRegime;
use crate::inference::models::gemma4::kv_cache::{
    restore_gemma_hybrid_slot_anchor, snapshot_gemma_hybrid_slot_anchor,
};

const SOURCE: usize = 2;
const HISTORY: usize = 700;
const FOLLOW_UP: usize = 120;
const TURN_TICKS: usize = 24;
const PEER_TICKS: usize = 5;
/// Mid-request moves: (destination, decode ticks before the move).
const MOVES: [(usize, usize); 4] = [(2, 0), (0, 1), (3, 8), (1, 24)];

/// One decode tick: the token it fed and its logits bits.
type Tick = (u32, Vec<u32>);

struct Trace {
    prefill: LaneImage,
    ticks: Vec<Tick>,
    last: LaneImage,
}

fn assert_trace(label: &str, actual: &Trace, reference: &Trace) {
    assert_lane_equal(
        &format!("{label}: prefill"),
        &actual.prefill,
        &reference.prefill,
    );
    assert_eq!(actual.ticks.len(), reference.ticks.len(), "{label}: ticks");
    for (index, (actual, reference)) in actual.ticks.iter().zip(&reference.ticks).enumerate() {
        assert_eq!(actual.0, reference.0, "{label}: tick {index} token");
        assert!(actual.1 == reference.1, "{label}: tick {index} logits bits");
    }
    assert_lane_equal(&format!("{label}: final"), &actual.last, &reference.last);
}

fn guard_of(loaded: &mut LoadedModel) -> Gemma4KvGuard<'_> {
    let LoadedModel::Gemma(g) = loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let guard = Gemma4KvGuard::take(g).unwrap();
    assert_eq!(guard.model.weights.kv_graft_len(), 0, "no graft bound");
    assert_eq!(
        GemmaSlotKvRegime::live(),
        GemmaSlotKvRegime::Hybrid,
        "production KV regime"
    );
    guard
}

fn prompt(seed: usize, len: usize, guard: &Gemma4KvGuard<'_>) -> Vec<u32> {
    synthetic_prompt(seed, len, guard.model.weights.vocab_size)
}

/// The reclaim wipe: every scaffold cursor of `slot` back to zero.
fn wipe(guard: &mut Gemma4KvGuard<'_>, slot: usize) {
    install_gemma4_slot_cursor(guard, SlotId(slot as u32), 0).unwrap();
}

/// Cold prefill of `prompt` into a wiped `slot`: (first token, logits).
fn cold(guard: &mut Gemma4KvGuard<'_>, slot: usize, prompt: &[u32]) -> (u32, Vec<f32>) {
    wipe(guard, slot);
    clear_gemma4_self_mounts(guard.model);
    let first = guard
        .model
        .weights
        .forward_prefill_with_soft_tokens_slot_aware(
            prompt,
            &[],
            MAX_DECODE + TURN_TICKS,
            &mut guard.model.ctx,
            SlotId(slot as u32),
            &mut guard.kv,
            guard.hybrid.as_mut(),
            guard.dense.as_mut(),
            guard.mlx.as_mut(),
        )
        .unwrap();
    clear_gemma4_self_mounts(guard.model);
    commit_gemma4_slot_cursor(guard, SlotId(slot as u32), 0, prompt.len()).unwrap();
    (first, guard.model.weights.logits_view().unwrap().to_vec())
}

/// Warm prefill of `prompt[cached..]` onto `slot`'s cursor at `cached`.
fn resume(
    guard: &mut Gemma4KvGuard<'_>,
    slot: usize,
    prompt: &[u32],
    cached: usize,
) -> (u32, Vec<f32>) {
    clear_gemma4_self_mounts(guard.model);
    let first = guard
        .model
        .weights
        .forward_prefill_with_soft_tokens_slot_aware_resume(
            prompt,
            &[],
            MAX_DECODE + TURN_TICKS,
            &mut guard.model.ctx,
            SlotId(slot as u32),
            &mut guard.kv,
            guard.hybrid.as_mut(),
            guard.dense.as_mut(),
            guard.mlx.as_mut(),
            cached,
        )
        .unwrap();
    clear_gemma4_self_mounts(guard.model);
    commit_gemma4_slot_cursor(guard, SlotId(slot as u32), cached, prompt.len()).unwrap();
    (first, guard.model.weights.logits_view().unwrap().to_vec())
}

/// `ticks` greedy batched-body decode ticks of `slot`, feeding `token` at
/// the slot's cursor and committing each tick as the worker does.
fn decode(
    guard: &mut Gemma4KvGuard<'_>,
    slot: usize,
    mut token: u32,
    ticks: usize,
) -> (u32, Vec<Tick>) {
    let vocab = guard.model.weights.vocab_size;
    let mut out = Vec::with_capacity(ticks);
    for _ in 0..ticks {
        let position = guard.hybrid.as_ref().unwrap()[0].seq_lens[slot] as usize;
        let mut regime = crate::inference::models::gemma4::batched_body::BatchedKvRegime::Hybrid(
            guard.hybrid.as_mut().unwrap().as_mut_slice(),
        );
        let mut head = None;
        let lm = &mut *guard.model;
        let hidden = lm
            .weights
            .forward_decode_body_batched(
                &[token],
                &[SlotId(slot as u32)],
                &[position],
                &mut regime,
                &mut head,
                &mut lm.ctx,
            )
            .unwrap();
        let head = match head {
            Some(head) => head,
            None => lm.weights.lm_head_batched(&hidden, 1, &mut lm.ctx).unwrap(),
        };
        let logits = &head.logits[..vocab];
        out.push((token, logits.iter().map(|value| value.to_bits()).collect()));
        commit_gemma4_decode_cursors(guard, &[(SlotId(slot as u32), position + 1)]).unwrap();
        token = argmax_f32_first_max(logits).0;
    }
    (token, out)
}

fn last_image(guard: &Gemma4KvGuard<'_>, slot: usize, next: u32, ticks: &[Tick]) -> LaneImage {
    let valid = guard.hybrid.as_ref().unwrap()[0].seq_lens[slot] as usize;
    let logits: Vec<f32> = ticks
        .last()
        .map(|(_, bits)| bits.iter().map(|&bits| f32::from_bits(bits)).collect())
        .unwrap_or_default();
    lane_image(guard, slot, valid, next, &logits)
}

/// Another conversation, mid-decode, in each listed slot.
fn occupy(guard: &mut Gemma4KvGuard<'_>, slots: &[usize], seed: usize) {
    for &slot in slots {
        let peer = prompt(seed + slot, 300 + 40 * slot, guard);
        let (first, _) = cold(guard, slot, &peer);
        decode(guard, slot, first, PEER_TICKS);
    }
}

/// Move `from` into `to` as a reclaim does: spill, wipe, restore.
fn move_slot(guard: &mut Gemma4KvGuard<'_>, from: usize, to: usize) {
    let spill = guard
        .slot_scaffolds()
        .spill(GemmaSlotKvRegime::Hybrid, SlotId(from as u32), 0)
        .unwrap();
    wipe(guard, from);
    guard
        .slot_scaffolds()
        .restore(SlotId(to as u32), &spill)
        .unwrap();
}

/// The target request: cold prefill in `SOURCE`, moved to `dest` after
/// `move_after` ticks (`None`: never), finished there.
fn mid_request(
    guard: &mut Gemma4KvGuard<'_>,
    target: &[u32],
    moved: Option<(usize, usize)>,
) -> Trace {
    occupy(guard, &[0, 1, 3], 50);
    let (first, logits) = cold(guard, SOURCE, target);
    let prefill = lane_image(guard, SOURCE, target.len(), first, &logits);
    let (dest, before) = moved.unwrap_or((SOURCE, TURN_TICKS));
    let (next, mut ticks) = decode(guard, SOURCE, first, before);
    if moved.is_some() {
        move_slot(guard, SOURCE, dest);
    }
    let (next, rest) = decode(guard, dest, next, TURN_TICKS - before);
    ticks.extend(rest);
    let last = last_image(guard, dest, next, &ticks);
    Trace {
        prefill,
        ticks,
        last,
    }
}

struct Ledgers {
    retained: Vec<Vec<u32>>,
    anchors: Vec<Option<Gemma4PromptAnchor>>,
}

/// Turn 1 in `slot`: cold prefill of `p1`, its input checkpoint (published as
/// the slot's anchor), `TURN_TICKS` decode ticks, and the retained prefix.
fn turn_one(guard: &mut Gemma4KvGuard<'_>, ledgers: &mut Ledgers, slot: usize, p1: &[u32]) {
    let (first, _) = cold(guard, slot, p1);
    let kv = snapshot_gemma_hybrid_slot_anchor(
        guard.hybrid.as_ref().unwrap(),
        SlotId(slot as u32),
        p1.len(),
    )
    .unwrap();
    let (_, ticks) = decode(guard, slot, first, TURN_TICKS);
    let mut retained = p1.to_vec();
    retained.extend(ticks.iter().map(|(token, _)| *token));
    ledgers.retained[slot] = retained;
    ledgers.anchors[slot] = Some(Gemma4PromptAnchor {
        prompt_tokens: p1.to_vec(),
        kv,
        vision_fingerprint: None,
    });
}

/// Admission's resume of `prompt` in `slot` for `preference`, as
/// `admit_gemma4_slot_routed` performs it, then its decode.
fn follow_up(
    guard: &mut Gemma4KvGuard<'_>,
    ledgers: &Ledgers,
    slot: usize,
    prompt: &[u32],
    preference: Gemma4SlotPreference,
) -> Trace {
    assert_eq!(preference.slot, SlotId(slot as u32));
    let cached = preference.cached_tokens;
    if preference.restore_prompt_anchor {
        let anchor = ledgers.anchors[slot].as_ref().expect("anchor");
        restore_gemma_hybrid_slot_anchor(
            guard.hybrid.as_mut().unwrap(),
            SlotId(slot as u32),
            &anchor.kv,
            cached,
        )
        .unwrap();
        install_gemma4_slot_cursor(guard, SlotId(slot as u32), cached).unwrap();
    } else {
        validate_gemma4_slot_cursor(guard, SlotId(slot as u32), cached).unwrap();
    }
    let (first, logits) = resume(guard, slot, prompt, cached);
    let prefill = lane_image(guard, slot, prompt.len(), first, &logits);
    let (next, ticks) = decode(guard, slot, first, TURN_TICKS);
    let last = last_image(guard, slot, next, &ticks);
    Trace {
        prefill,
        ticks,
        last,
    }
}

fn ledgers() -> Ledgers {
    Ledgers {
        retained: vec![Vec::new(); 4],
        anchors: (0..4).map(|_| None).collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn reclaim(
    guard: &mut Gemma4KvGuard<'_>,
    spills: &mut Gemma4SlotSpills,
    ledgers: &mut Ledgers,
    granted: usize,
    preferred: Option<Gemma4SlotPreference>,
    prompt: &[u32],
) -> Option<Gemma4SlotPreference> {
    reclaim_gemma4_slot(
        &mut guard.slot_scaffolds(),
        spills,
        0,
        SlotId(granted as u32),
        &mut ledgers.retained,
        &mut ledgers.anchors,
        preferred,
        prompt,
        None,
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Form {
    LiveTail,
    Anchor,
    OwnSlotNotGranted,
}

fn follow_up_prompt(
    guard: &Gemma4KvGuard<'_>,
    ledgers: &Ledgers,
    p1: &[u32],
    form: Form,
) -> Vec<u32> {
    let mut prompt = match form {
        Form::Anchor => p1.to_vec(),
        Form::LiveTail | Form::OwnSlotNotGranted => ledgers.retained[SOURCE].clone(),
    };
    prompt.extend(synthetic_prompt(
        77,
        FOLLOW_UP,
        guard.model.weights.vocab_size,
    ));
    prompt
}

/// In place: turn 1 and the follow-up both in `SOURCE`.
fn between_turns_reference(guard: &mut Gemma4KvGuard<'_>, p1: &[u32], form: Form) -> Trace {
    occupy(guard, &[0, 1, 3], 60);
    let mut ledgers = ledgers();
    turn_one(guard, &mut ledgers, SOURCE, p1);
    let prompt = follow_up_prompt(guard, &ledgers, p1, form);
    let preference = preferred_gemma4_slot_for_request(
        &ledgers.retained,
        &ledgers.anchors,
        &[None, None, None, None],
        &prompt,
        None,
    )
    .expect("in-place resume");
    follow_up(guard, &ledgers, SOURCE, &prompt, preference)
}

/// Turn 1 in `SOURCE`, reclaimed and resumed in `dest` through
/// `reclaim_gemma4_slot`.
fn between_turns_moved(
    guard: &mut Gemma4KvGuard<'_>,
    p1: &[u32],
    form: Form,
    dest: usize,
) -> Trace {
    occupy(guard, &[0, 1, 3], 60);
    let mut ledgers = ledgers();
    let mut spills = Gemma4SlotSpills::new(1 << 36, 4, GemmaSlotKvRegime::Hybrid);
    turn_one(guard, &mut ledgers, SOURCE, p1);
    let prompt = follow_up_prompt(guard, &ledgers, p1, form);
    let preferred = preferred_gemma4_slot_for_request(
        &ledgers.retained,
        &ledgers.anchors,
        &[None, None, None, None],
        &prompt,
        None,
    )
    .expect("own-slot preference");
    let granted = match form {
        Form::OwnSlotNotGranted => reclaim(
            guard,
            &mut spills,
            &mut ledgers,
            dest,
            Some(preferred),
            &prompt,
        ),
        Form::LiveTail | Form::Anchor => {
            // A cold request takes `SOURCE` and prefills over it.
            let other = prompt_cold(guard);
            assert_eq!(
                reclaim(guard, &mut spills, &mut ledgers, SOURCE, None, &other),
                None
            );
            assert_eq!(spills.len(), 1, "the conversation was spilled");
            ledgers.retained[SOURCE].clear();
            ledgers.anchors[SOURCE] = None;
            cold(guard, SOURCE, &other);
            reclaim(guard, &mut spills, &mut ledgers, dest, None, &prompt)
        }
    }
    .expect("restored into the granted slot");
    assert_eq!(
        (granted.cached_tokens, granted.restore_prompt_anchor),
        (preferred.cached_tokens, preferred.restore_prompt_anchor),
        "{form:?}: the same resume as in place"
    );
    follow_up(guard, &ledgers, dest, &prompt, granted)
}

fn prompt_cold(guard: &Gemma4KvGuard<'_>) -> Vec<u32> {
    synthetic_prompt(901, 500, guard.model.weights.vocab_size)
}

/// The live-tail follow-up as a `PerSlotExact` joint warm lane beside peers
/// in slots 0 and 1; `dest` is where the target runs. Returns the target's
/// and the peers' images, in lane order.
fn joint_cohort(guard: &mut Gemma4KvGuard<'_>, p1: &[u32], dest: usize) -> Vec<LaneImage> {
    occupy(guard, &[0, 1, 3], 70);
    let mut ledgers = ledgers();
    let mut spills = Gemma4SlotSpills::new(1 << 36, 4, GemmaSlotKvRegime::Hybrid);
    turn_one(guard, &mut ledgers, SOURCE, p1);
    let prompt = follow_up_prompt(guard, &ledgers, p1, Form::LiveTail);
    let cached = ledgers.retained[SOURCE].len();
    if dest != SOURCE {
        let other = prompt_cold(guard);
        assert_eq!(
            reclaim(guard, &mut spills, &mut ledgers, SOURCE, None, &other),
            None
        );
        cold(guard, SOURCE, &other);
        let granted =
            reclaim(guard, &mut spills, &mut ledgers, dest, None, &prompt).expect("restored");
        assert_eq!(
            (granted.cached_tokens, granted.restore_prompt_anchor),
            (cached, false)
        );
    }
    assert!(gemma_warm_batch::lane_eligible(prompt.len(), cached));
    // Two peers with their own histories and joint-eligible suffixes.
    let mut lanes = vec![(prompt.clone(), dest, cached)];
    for (slot, history) in [(0usize, 465usize), (1, 714)] {
        let peer = synthetic_prompt(
            300 + slot,
            history + 56 + 100 * slot,
            guard.model.weights.vocab_size,
        );
        cold(guard, slot, &peer[..history]);
        lanes.push((peer, slot, history));
    }
    let seqs: Vec<(Vec<u32>, SlotId, usize)> = lanes
        .iter()
        .map(|(prompt, slot, cached)| (prompt[*cached..].to_vec(), SlotId(*slot as u32), *cached))
        .collect();
    let reserve = gemma4_batch_decode_reserve(
        lanes
            .iter()
            .map(|(prompt, _, _)| (prompt.len(), MAX_DECODE)),
    );
    clear_gemma4_self_mounts(guard.model);
    let output = guard
        .model
        .weights
        .forward_prefill_batched_multi_seq_live(
            &seqs,
            guard.hybrid.as_deref().unwrap(),
            reserve,
            &mut guard.model.ctx,
        )
        .unwrap();
    clear_gemma4_self_mounts(guard.model);
    let commits: Vec<_> = lanes
        .iter()
        .map(|(prompt, slot, cached)| (SlotId(*slot as u32), *cached, prompt.len()))
        .collect();
    commit_gemma4_slot_cursors(guard, &commits).unwrap();
    lanes
        .iter()
        .zip(output.first_tokens)
        .zip(output.logits)
        .map(|(((prompt, slot, _), first), logits)| {
            lane_image(guard, *slot, prompt.len(), first, &logits)
        })
        .collect()
}

#[test]
#[ignore = "official Gemma 4 GGUF: a reclaimed conversation restored into any slot resumes bitwise (#727)"]
fn gemma_spilled_slot_resumes_bitwise_in_any_slot() {
    let mut loaded = load_gate_model();
    let mut guard = guard_of(&mut loaded);
    let target = prompt(11, HISTORY, &guard);

    let reference = mid_request(&mut guard, &target, None);
    for (dest, before) in MOVES {
        let moved = mid_request(&mut guard, &target, Some((dest, before)));
        assert_trace(
            &format!("mid-request move to slot {dest} after {before} ticks"),
            &moved,
            &reference,
        );
    }

    for form in [Form::LiveTail, Form::Anchor, Form::OwnSlotNotGranted] {
        let reference = between_turns_reference(&mut guard, &target, form);
        for dest in [0usize, 1, 3] {
            let moved = between_turns_moved(&mut guard, &target, form, dest);
            assert_trace(
                &format!("{form:?} follow-up in slot {dest}"),
                &moved,
                &reference,
            );
        }
    }

    let reference = joint_cohort(&mut guard, &target, SOURCE);
    let moved = joint_cohort(&mut guard, &target, 3);
    for (index, (moved, reference)) in moved.iter().zip(&reference).enumerate() {
        assert_lane_equal(&format!("joint warm cohort lane {index}"), moved, reference);
    }
    println!(
        "GEMMA_SPILLED_SLOT_RESUME_BITWISE_COMPLETE moves={MOVES:?} follow_ups=[live,anchor,own_slot]x[0,1,3] joint_cohort=true history={HISTORY} follow_up={FOLLOW_UP} ticks={TURN_TICKS}"
    );
}

#[test]
#[ignore = "official Gemma 4 GGUF: the same conversation is bitwise equal in every slot (#727)"]
fn gemma_resume_is_slot_independent_bitwise() {
    let mut loaded = load_gate_model();
    let mut guard = guard_of(&mut loaded);
    let target = prompt(13, HISTORY, &guard);
    let run = |guard: &mut Gemma4KvGuard<'_>, slot: usize| -> (Trace, Trace) {
        let peers: Vec<usize> = (0..4).filter(|&peer| peer != slot).collect();
        occupy(guard, &peers, 80 + slot);
        let (first, logits) = cold(guard, slot, &target);
        let prefill = lane_image(guard, slot, target.len(), first, &logits);
        let (next, ticks) = decode(guard, slot, first, TURN_TICKS);
        let last = last_image(guard, slot, next, &ticks);
        let mut retained = target.clone();
        retained.extend(ticks.iter().map(|(token, _)| *token));
        let mut follow_up = retained.clone();
        follow_up.extend(synthetic_prompt(
            79,
            FOLLOW_UP,
            guard.model.weights.vocab_size,
        ));
        let (first, logits) = resume(guard, slot, &follow_up, retained.len());
        let resumed = lane_image(guard, slot, follow_up.len(), first, &logits);
        let (next, resumed_ticks) = decode(guard, slot, first, TURN_TICKS);
        let resumed_last = last_image(guard, slot, next, &resumed_ticks);
        (
            Trace {
                prefill,
                ticks,
                last,
            },
            Trace {
                prefill: resumed,
                ticks: resumed_ticks,
                last: resumed_last,
            },
        )
    };
    let (turn, follow) = run(&mut guard, 0);
    for slot in 1..4 {
        let (other_turn, other_follow) = run(&mut guard, slot);
        assert_trace(
            &format!("turn 1 in slot {slot} vs slot 0"),
            &other_turn,
            &turn,
        );
        assert_trace(
            &format!("live follow-up in slot {slot} vs slot 0"),
            &other_follow,
            &follow,
        );
    }
    println!("GEMMA_RESUME_SLOT_INDEPENDENT_BITWISE_COMPLETE slots=0..4 history={HISTORY} follow_up={FOLLOW_UP} ticks={TURN_TICKS}");
}
