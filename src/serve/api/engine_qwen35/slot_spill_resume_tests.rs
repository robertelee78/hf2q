//! Issue #727 gate (official artifact): a conversation whose slot is
//! reclaimed and later restored into another slot continues bitwise as if it
//! had kept its slot.
//!
//! Every move goes through the reclaim path: `HybridKvCache::spill_slot(2)`,
//! `reset_for_slot(2)` (the reclaim wipe), then `restore_slot_spill(dest)`.
//! Before the move, `dest` holds another conversation mid-decode. Two
//! families of variants run, for both a copy target (history rounds) and a
//! free-generation target (long K3 chains):
//!
//! 1. **Between turns (the production path).** Turn 1 finishes in slot 2 and
//!    publishes its ledger exactly as serving does: the full-prompt anchor
//!    (`snapshot_slot_anchor`, the prefill logits, and the speculative
//!    boundary) captured at the prompt boundary as `finish_qwen35_prefill`
//!    does, and the retained prefix with its speculative boundary
//!    (`retained_prefix` / `spec_prefix_candidate` at the final cursor) as
//!    decode completion does. The slot moves to `dest`, and a follow-up turn
//!    is admitted through `Qwen35PrefillState::begin` the way admission
//!    resumes it:
//!    - **anchor:** the follow-up shares only the turn-1 prompt; the slot is
//!      rewound with `restore_slot_anchor`;
//!    - **live tail:** the follow-up extends the retained prefix, carrying
//!      its speculative boundary (the pending verifier hidden row).
//!
//!    The reference runs the same follow-up in place in slot 2.
//! 2. **Mid-request.** The target request moves after 0, 1, 8 or 40 decode
//!    ticks and finishes in `dest`; the reference is the slot-independence
//!    gate's in-place run.
//!
//! Compared bitwise with the slot-independence gate's `assert_same`:
//! - the prefill logits;
//! - per tick: the route, committed tokens, verified queue, target and MTP
//!   cursors, and the MTP verifier hidden row;
//! - the final state: per-layer cursors, valid KV code/scale (or F32 K/V)
//!   rows, valid MTP cache rows, and DeltaNet parity with conv/recurrent
//!   current and scratch regions;
//! - the logits of one probe forward over the final state.
//!
//! ```text
//! # The Qwen3.8 padded route comes from the model's own routing policy
//! # (artifact path names qwen3.8/qwen38; no HF2Q_DECODE_MVN/MV_EXT override).
//! HF2Q_QWEN38_GGUF=/abs/Qwen3.8-27B-vanilla.gguf \
//!   cargo test --release --bin hf2q qwen38_spilled_slot_resumes_bitwise_in_any_slot \
//!   -- --ignored --exact --nocapture --test-threads=1
//! HF2Q_QWEN36_GGUF=/abs/qwen36.gguf \
//!   cargo test --release --bin hf2q qwen36_spilled_slot_resumes_bitwise_in_any_slot \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```

use super::round_slot_independence_tests::{
    assert_padded_policy, assert_same, bits, cursors, load_gate_artifact, new_cache, prefill,
    run_variant, slot_state, tick_peer, tick_target, user_turn, Peer, Peers, Route, TargetTrace,
    Tick, K3_TARGET_TEXT, MAX_TARGET_TICKS, PEER_DECODE_STEPS, PEER_MAX_TOKENS, PEER_TEXTS,
    PREFILL_CHUNK, REFERENCE_SLOT, TARGET_MAX_TOKENS, TARGET_TEXT,
};
use super::{
    Qwen35DecodeState, Qwen35LoadedModel, Qwen35PrefillAdvance, Qwen35PrefillState,
    Qwen35SpecPrefixBoundary,
};
use crate::inference::models::qwen35::kv_cache::{HybridKvCache, HybridKvSlotAnchor};
use crate::serve::api::engine::{LoadedModel, SamplingParams};
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::multi_seq_kv::SlotId;

/// Mid-request moves: (destination slot, decode ticks before the move). The
/// source is slot 2; zero ticks moves the request at its prompt boundary.
const MOVES: [(SlotId, usize); 4] = [
    (SlotId(2), 0),
    (SlotId(0), 1),
    (SlotId(3), 8),
    (SlotId(1), 40),
];
/// Between-turn moves: the same slot after the wipe, and two others.
const TURN_DESTINATIONS: [SlotId; 3] = [SlotId(2), SlotId(0), SlotId(3)];
const FIRST_TURN_MAX_TOKENS: usize = 64;
const FOLLOW_UP_MAX_TOKENS: usize = 96;
/// Follow-up text after the turn-1 prompt for the anchor form. It must be
/// shorter than turn 1's generation so the retained prefix cannot match.
const ANCHOR_FOLLOW_UP: &str = "Sure, here it is:";
const LIVE_FOLLOW_UP: &str = "Now summarize your answer in one sentence.";

/// Place another conversation mid-decode in `slot`.
fn occupy(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    slot: SlotId,
    peer_prompt: &[u32],
    supervisor: &EngineSupervisor,
) {
    let (state, _) = prefill(qwen, kv, slot, peer_prompt, PEER_MAX_TOKENS, supervisor);
    let mut peer = Peer {
        slot,
        state,
        live: true,
    };
    for _ in 0..PEER_DECODE_STEPS {
        tick_peer(qwen, kv, &mut peer, supervisor);
    }
}

/// The reclaim of `source` and the restore of its spill into `dest`.
fn move_slot(kv: &mut HybridKvCache, source: SlotId, dest: SlotId, label: &str) {
    let spill = kv
        .spill_slot(source)
        .unwrap_or_else(|error| panic!("{label}: spill: {error:#}"));
    kv.reset_for_slot(source)
        .unwrap_or_else(|error| panic!("{label}: reclaim wipe: {error:#}"));
    kv.restore_slot_spill(dest, &spill)
        .unwrap_or_else(|error| panic!("{label}: restore: {error:#}"));
}

/// Final slot state and probe logits of a finished request in `slot`.
#[allow(clippy::too_many_arguments)]
fn trace(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    slot: SlotId,
    target: &Qwen35DecodeState,
    prompt_len: usize,
    prefill_logits: &[f32],
    ticks: Vec<Tick>,
    label: &str,
) -> TargetTrace {
    let state = slot_state(kv, slot);
    let next = *target
        .generated_tokens
        .last()
        .expect("target committed tokens");
    let position = prompt_len + target.generated_tokens.len() - 1;
    // As in the slot-independence gate: a request that finished inside a
    // verify round has no defined probe input; the cursors compare instead.
    let probe = if cursors(kv, slot).0 as usize == position {
        qwen.model
            .forward_gpu_last_logits(&[next], &[position as i32; 4], kv, slot)
            .unwrap_or_else(|error| panic!("{label}: probe forward: {error:#}"))
    } else {
        Vec::new()
    };
    TargetTrace {
        prefill_logits: bits(prefill_logits),
        ticks,
        state,
        probe_logits: bits(&probe),
    }
}

/// Serve the target from the reference slot, move it to `dest` through
/// spill → wipe → restore after `ticks_before_move` decode ticks, and finish
/// it there.
fn run_moved(
    qwen: &mut Qwen35LoadedModel,
    target_prompt: &[u32],
    peer_prompt: &[u32],
    dest: SlotId,
    ticks_before_move: usize,
) -> TargetTrace {
    let source = REFERENCE_SLOT;
    let label = format!("move to slot {} after {ticks_before_move} ticks", dest.0);
    let supervisor = EngineSupervisor::new();
    let mut kv = new_cache(qwen);
    if dest != source {
        occupy(qwen, &mut kv, dest, peer_prompt, &supervisor);
    }
    let (mut target, prefill_logits) = prefill(
        qwen,
        &mut kv,
        source,
        target_prompt,
        TARGET_MAX_TOKENS,
        &supervisor,
    );

    let mut slot = source;
    let mut moved = false;
    let mut ticks: Vec<Tick> = Vec::new();
    while !ticks.last().is_some_and(|tick: &Tick| tick.finished) {
        assert!(
            ticks.len() < MAX_TARGET_TICKS,
            "{label}: target did not finish within {MAX_TARGET_TICKS} ticks"
        );
        if !moved && ticks.len() == ticks_before_move {
            move_slot(&mut kv, source, dest, &label);
            target.slot_id = dest;
            slot = dest;
            moved = true;
        }
        ticks.push(tick_target(qwen, &mut kv, slot, &mut target, &supervisor));
    }
    assert!(moved, "{label}: the request finished before its move");
    assert!(supervisor.is_healthy(), "{label}: GPU supervision failed");
    trace(
        qwen,
        &mut kv,
        slot,
        &target,
        target_prompt.len(),
        &prefill_logits,
        ticks,
        &label,
    )
}

/// What a finished turn publishes for its next admission.
struct Ledger {
    /// Full-prompt anchor captured at the prompt boundary.
    anchor: HybridKvSlotAnchor,
    anchor_spec: Option<Qwen35SpecPrefixBoundary>,
    /// Exact token prefix of the live cache at completion.
    retained: Vec<u32>,
    retained_spec: Option<Qwen35SpecPrefixBoundary>,
}

/// Serve turn 1 in `slot` to completion and publish its ledger the way
/// `finish_qwen35_prefill` and decode completion do.
fn serve_first_turn(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    slot: SlotId,
    prompt: &[u32],
    supervisor: &EngineSupervisor,
    label: &str,
) -> Ledger {
    let (mut state, _) = prefill(qwen, kv, slot, prompt, FIRST_TURN_MAX_TOKENS, supervisor);
    let anchor = kv
        .snapshot_slot_anchor(slot, prompt.len())
        .unwrap_or_else(|error| panic!("{label}: prompt anchor: {error:#}"));
    let anchor_spec = state
        .spec_prefix_candidate(prompt.len())
        .filter(|spec| {
            kv.validate_speculative_cursors_for_slot(slot, spec.token_count)
                .is_ok()
        });
    let mut finished = false;
    for _ in 0..MAX_TARGET_TICKS {
        let outcome = state
            .decode_tick(qwen, kv, supervisor)
            .unwrap_or_else(|error| panic!("{label}: turn-1 tick: {error:#}"));
        if state.complete_tick(outcome).finished {
            finished = true;
            break;
        }
    }
    assert!(finished, "{label}: turn 1 did not finish");
    let valid = kv
        .sequence_len_for_slot(slot)
        .expect("turn-1 retained cursor") as usize;
    let retained = state.retained_prefix(valid);
    let retained_spec = state.spec_prefix_candidate(valid).filter(|spec| {
        kv.validate_speculative_cursors_for_slot(slot, spec.token_count)
            .is_ok()
    });
    Ledger {
        anchor,
        anchor_spec,
        retained,
        retained_spec,
    }
}

#[derive(Clone, Copy, Debug)]
enum FollowUp {
    Anchor,
    LiveTail,
}

/// Admit and serve a follow-up turn in `slot` from a resumed prefix, as
/// `admit_qwen35_slot` does after its anchor or live-tail decision.
#[allow(clippy::too_many_arguments)]
fn serve_follow_up(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    slot: SlotId,
    prompt: &[u32],
    cached_tokens: usize,
    cached_spec: Option<Qwen35SpecPrefixBoundary>,
    supervisor: &EngineSupervisor,
    label: &str,
) -> TargetTrace {
    let params = SamplingParams {
        max_tokens: FOLLOW_UP_MAX_TOKENS,
        ..SamplingParams::default()
    };
    let mut pending = Qwen35PrefillState::begin(
        prompt.to_vec(),
        params,
        None,
        kv,
        slot,
        cached_tokens,
        None,
        cached_spec,
        None,
        qwen.hidden_size,
        None,
    )
    .unwrap_or_else(|error| panic!("{label}: follow-up admission: {error:#}"));
    let (mut target, prefill_logits) = loop {
        match pending
            .advance(qwen, None, kv, PREFILL_CHUNK, supervisor)
            .unwrap_or_else(|error| panic!("{label}: follow-up prefill: {error:#}"))
        {
            Qwen35PrefillAdvance::Pending { state, .. } => pending = state,
            Qwen35PrefillAdvance::Ready {
                state,
                prefill_logits,
                ..
            } => break (state, prefill_logits),
        }
    };
    assert!(
        !target.finished_at_seed(),
        "{label}: follow-up finished at its seed token"
    );
    let mut ticks: Vec<Tick> = Vec::new();
    while !ticks.last().is_some_and(|tick: &Tick| tick.finished) {
        assert!(
            ticks.len() < MAX_TARGET_TICKS,
            "{label}: follow-up did not finish within {MAX_TARGET_TICKS} ticks"
        );
        ticks.push(tick_target(qwen, kv, slot, &mut target, supervisor));
    }
    assert!(supervisor.is_healthy(), "{label}: GPU supervision failed");
    trace(
        qwen,
        kv,
        slot,
        &target,
        prompt.len(),
        &prefill_logits,
        ticks,
        label,
    )
}

/// Turn 1 in slot 2, then a follow-up resumed from its ledger: in place
/// (`dest == None`) or after moving the slot to `dest`. Returns the
/// follow-up trace and whether a speculative boundary was resumed.
fn run_follow_up(
    qwen: &mut Qwen35LoadedModel,
    first_turn: &[u32],
    peer_prompt: &[u32],
    suffix: &[u32],
    form: FollowUp,
    dest: Option<SlotId>,
) -> (TargetTrace, bool) {
    let source = REFERENCE_SLOT;
    let slot = dest.unwrap_or(source);
    let label = match dest {
        Some(dest) => format!("{form:?} follow-up after a move to slot {}", dest.0),
        None => format!("{form:?} follow-up in place"),
    };
    let supervisor = EngineSupervisor::new();
    let mut kv = new_cache(qwen);
    if dest.is_some_and(|dest| dest != source) {
        occupy(qwen, &mut kv, slot, peer_prompt, &supervisor);
    }
    let ledger = serve_first_turn(qwen, &mut kv, source, first_turn, &supervisor, &label);
    if dest.is_some() {
        move_slot(&mut kv, source, slot, &label);
    }

    let (prompt, cached_tokens, candidate) = match form {
        FollowUp::Anchor => {
            kv.restore_slot_anchor(slot, &ledger.anchor)
                .unwrap_or_else(|error| panic!("{label}: anchor restore: {error:#}"));
            let prompt: Vec<u32> = first_turn.iter().chain(suffix).copied().collect();
            assert!(
                !(ledger.retained.len() < prompt.len() && prompt.starts_with(&ledger.retained)),
                "{label}: the follow-up must resume the anchor, not the live tail"
            );
            (prompt, first_turn.len(), ledger.anchor_spec)
        }
        FollowUp::LiveTail => {
            let prompt: Vec<u32> = ledger.retained.iter().chain(suffix).copied().collect();
            (prompt, ledger.retained.len(), ledger.retained_spec)
        }
    };
    // Admission's speculative-boundary check after either resume.
    let cached_spec = candidate.filter(|spec| {
        spec.token_count == cached_tokens
            && kv
                .validate_speculative_cursors_for_slot(slot, cached_tokens)
                .is_ok()
    });
    let carried = cached_spec.is_some();
    let trace = serve_follow_up(
        qwen,
        &mut kv,
        slot,
        &prompt,
        cached_tokens,
        cached_spec,
        &supervisor,
        &label,
    );
    (trace, carried)
}

fn run_gate(env: &str, expect_padded: bool) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut loaded = load_gate_artifact(env);
    let LoadedModel::Qwen35(qwen) = &mut loaded else {
        unreachable!("load_gate_artifact returns a Qwen model")
    };
    if expect_padded {
        assert_padded_policy(qwen, env);
    }
    let tokenizer = qwen.tokenizer.clone();
    let tokens = |text: &str| -> Vec<u32> {
        tokenizer
            .encode(text, false)
            .expect("tokenize gate text")
            .get_ids()
            .to_vec()
    };
    let encode = |text: &str| tokens(user_turn(text).as_str());
    let peer_prompts = PEER_TEXTS.map(encode);
    let anchor_suffix = tokens(ANCHOR_FOLLOW_UP);
    let live_suffix = tokens(&format!("<|im_end|>\n{}", user_turn(LIVE_FOLLOW_UP)));

    let mut live_spec_resumed = false;
    for (name, text) in [("copy", TARGET_TEXT), ("k3", K3_TARGET_TEXT)] {
        let target_prompt = encode(text);

        // Between turns: the production reclaim → return path.
        for (form, suffix) in [
            (FollowUp::Anchor, &anchor_suffix),
            (FollowUp::LiveTail, &live_suffix),
        ] {
            let (reference, reference_spec) =
                run_follow_up(qwen, &target_prompt, &peer_prompts[0], suffix, form, None);
            if matches!(form, FollowUp::LiveTail) {
                live_spec_resumed |= reference_spec;
            }
            eprintln!(
                "[slot-spill] {env} {name}: {form:?} follow-up, {} ticks, speculative boundary resumed: {reference_spec}",
                reference.ticks.len()
            );
            for dest in TURN_DESTINATIONS {
                let (variant, variant_spec) = run_follow_up(
                    qwen,
                    &target_prompt,
                    &peer_prompts[0],
                    suffix,
                    form,
                    Some(dest),
                );
                let label = format!("{name}: {form:?} follow-up after a move to slot {}", dest.0);
                assert_eq!(
                    variant_spec, reference_spec,
                    "{label}: speculative boundary resume differs"
                );
                assert_same(&reference, &variant, &label);
                eprintln!("[slot-spill] {env}: {label} bitwise equal");
            }
        }

        // Mid-request moves.
        let reference = run_variant(
            qwen,
            &target_prompt,
            &peer_prompts,
            REFERENCE_SLOT,
            Peers::Empty,
        );
        let routes = reference.routes();
        let k3_rounds = routes.iter().filter(|&&route| route == Route::K3).count();
        eprintln!(
            "[slot-spill] {env} {name}: {} target ticks, {k3_rounds} K3 rounds",
            routes.len()
        );
        assert!(k3_rounds > 0, "{name}: no K3 round ran: {routes:?}");
        for (dest, ticks_before_move) in MOVES {
            assert!(
                ticks_before_move < routes.len(),
                "{name}: the reference finishes before tick {ticks_before_move}"
            );
            let variant = run_moved(qwen, &target_prompt, &peer_prompts[0], dest, ticks_before_move);
            let label = format!(
                "{name}: moved to slot {} after {ticks_before_move} ticks",
                dest.0
            );
            assert_same(&reference, &variant, &label);
            eprintln!("[slot-spill] {env}: {label} bitwise equal");
        }
    }
    assert!(
        live_spec_resumed,
        "no live-tail follow-up resumed a speculative boundary; the pending-hidden path went untested"
    );
}

/// #727 on the official Qwen3.8 artifact, padded `mul_mv_ext` decode route:
/// spill, reclaim wipe and restore into any slot, between turns or
/// mid-request, leave every later tick, the final cache state and the probe
/// logits bitwise identical to keeping the slot.
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (official Qwen3.8 artifact with its MTP head) and Apple Metal"]
fn qwen38_spilled_slot_resumes_bitwise_in_any_slot() {
    run_gate("HF2Q_QWEN38_GGUF", true);
}

/// #727 on the official Qwen3.6 artifact: spill, reclaim wipe and restore
/// into any slot, between turns or mid-request, leave every later tick, the
/// final cache state and the probe logits bitwise identical to keeping the
/// slot.
#[test]
#[ignore = "requires HF2Q_QWEN36_GGUF (official Qwen3.6 artifact with its MTP head) and Apple Metal"]
fn qwen36_spilled_slot_resumes_bitwise_in_any_slot() {
    run_gate("HF2Q_QWEN36_GGUF", false);
}
