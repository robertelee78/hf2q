//! Deterministic, weightless consumer semantics; no Metal dispatch or models.
use std::sync::Arc;
use std::time::Duration;

use ahash::AHashMap;
use tokenizers::decoders::byte_level::ByteLevel;
use tokenizers::models::bpe::BPE;
use tokenizers::Tokenizer;

use super::*;
use super::super::{
    plan_qwen35_verified_block, Qwen35CanonicalDecision, Qwen35DecisionSource,
    Qwen35SpecSemanticState, Qwen35ThinkingBudgetState,
};
use super::super::speculation_qualification_tests::loaded_model;
use crate::serve::api::engine::SamplingParams;
use crate::serve::api::grammar::parse;
use crate::serve::api::qwen35_speculation::QwenSpeculationPolicy;
use crate::serve::api::registry::{ToolCallSplitter, QWEN35};
use crate::serve::api::token_text::token_bytes_table;
use crate::serve::multi_seq_kv::SlotId;

const SEED: u32 = 0;
const A: u32 = 1;
const B: u32 = 2;
const C: u32 = 3;
const X: u32 = 4;
const OPEN: u32 = 5;
const MERGED: u32 = 6;
const OPEN_LEFT: u32 = 7;
const OPEN_RIGHT: u32 = 8;
const LEAD: u32 = 9;
const CONT: u32 = 10;
const CONT_OPEN: u32 = 11;
const EMPTY: u32 = 15;
const EOS: u32 = 63;

fn model() -> Qwen35LoadedModel {
    let pieces = [
        (SEED, "p"), (A, "a"), (B, "b"), (C, "c"), (X, "x"),
        (OPEN, "<tool_call>"), (MERGED, "<tool_call>a"),
        (OPEN_LEFT, "<tool_"), (OPEN_RIGHT, "call>"),
        // GPT-2 byte-level pieces: C3 A9 is the UTF-8 encoding of é.
        (LEAD, "\u{00c3}"), (CONT, "\u{00a9}"),
        (CONT_OPEN, "\u{00a9}<tool_call>"),
    ];
    let vocab: AHashMap<String, u32> = pieces.into_iter()
        .map(|(id, piece)| (piece.to_string(), id)).collect();
    let bpe = BPE::builder().vocab_and_merges(vocab, Vec::new())
        .build().expect("byte-level fixture");
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_decoder(Some(ByteLevel::default()));
    let mut qwen = loaded_model(QwenSpeculationPolicy::Off);
    qwen.tokenizer = tokenizer;
    qwen
}

fn state(qwen: &Qwen35LoadedModel, first: u32, grammar: &str) -> Qwen35DecodeState {
    let mut logits = vec![-100.0; qwen.vocab_size];
    logits[first as usize] = 20.0;
    let mut state = Qwen35DecodeState::from_prefill_logits(
        qwen, vec![SEED], SamplingParams { max_tokens: 64, ..SamplingParams::default() },
        None, SlotId(0), 0, &logits, Duration::ZERO, 1, None,
    ).expect("seed decode state");
    let grammar = parse(grammar).expect("fixture grammar");
    let root = grammar.rule_id("root").unwrap();
    state.params.grammar = Some(grammar.clone());
    state.params.token_bytes = Some(Arc::new(token_bytes_table(&qwen.tokenizer, qwen.vocab_size)));
    state.grammar_runtime = Some(GrammarRuntime::new(grammar, root).unwrap());
    state
}

fn choose(
    semantic: &mut Qwen35SpecSemanticState,
    qwen: &Qwen35LoadedModel,
    params: &SamplingParams,
    preferred: u32,
    prepared: Option<&mut PreparedDFlash2Grammar>,
) -> Qwen35CanonicalDecision {
    let mut logits = vec![-100.0; qwen.vocab_size];
    logits[preferred as usize] = 20.0;
    semantic.select_and_observe_from(
        qwen, params, Qwen35DecisionSource::Logits(&mut logits), prepared,
    ).expect("canonical decision")
}

/// Compare actual canonical decisions and committed text/history/router state,
/// including decisions AFTER an early mismatch. Preparation cannot mutate the
/// source: the real ordinary commit path below remains an independent oracle.
fn sequence(
    qwen: &Qwen35LoadedModel,
    state: &mut Qwen35DecodeState,
    drafts: &[u32],
    decisions: &[(u32, u32)],
    expected_text: &str,
) {
    let mut ordinary = Qwen35SpecSemanticState::from_decode(state);
    let mut projected = ordinary.clone();
    let mut prepared = state.prepare_dflash2_grammar(qwen, drafts, false).expect("prepare");
    for &(preferred, expected) in decisions {
        let left = choose(&mut ordinary, qwen, &state.params, preferred, None);
        let right = choose(&mut projected, qwen, &state.params, preferred, prepared.as_mut());
        assert_eq!(left, right);
        assert_eq!(right.token, expected);
        assert_eq!(right.terminal, expected == EOS);
        let tick = state.commit_ordinary_target_token(qwen, expected, Duration::ZERO)
            .expect("ordinary output commit");
        let tick = state.complete_tick(tick);
        assert_eq!(tick.finished, expected == EOS);
        assert_eq!(projected.generated_tokens, state.generated_tokens);
        assert_eq!(projected.sampling_history, state.sampling_history);
    }
    assert_eq!(state.decoded_text, expected_text);
    assert_eq!(projected.text_decoder.clone().finish(), ordinary.text_decoder.clone().finish());
    assert_eq!(
        projected.tool_splitter.clone().and_then(|mut splitter| splitter.finish()),
        state.tool_splitter.clone().and_then(|mut splitter| splitter.finish()),
    );
    assert_eq!(
        projected.tool_splitter.as_ref().map(ToolCallSplitter::in_tool_call),
        state.tool_splitter.as_ref().map(ToolCallSplitter::in_tool_call),
    );
}

#[test]
fn ranked_canonical_window_and_retained_prefix_are_unchanged() {
    let qwen = model();
    let state = state(&qwen, SEED, "root ::= \"abc\"\n");
    let mut ordinary = Qwen35SpecSemanticState::from_decode(&state);
    let mut projected = ordinary.clone();
    let mut prepared = state.prepare_dflash2_grammar(&qwen, &[A, B, C], false).unwrap();
    let decide = |semantic: &mut Qwen35SpecSemanticState, row: usize,
                      prepared: Option<&mut PreparedDFlash2Grammar>| {
        let mut logits = vec![-100.0; qwen.vocab_size];
        // The top logit is invalid; the unchanged ranked-validity sampler
        // must still make the grammar-valid canonical choice on every row.
        logits[X as usize] = 30.0;
        logits[[A, B, C, EOS][row] as usize] = 20.0;
        semantic.select_and_observe_from(
            &qwen, &state.params, Qwen35DecisionSource::Logits(&mut logits), prepared,
        )
    };
    let left = plan_qwen35_verified_block(&[A, B, C], |row| decide(&mut ordinary, row, None)).unwrap();
    let right = plan_qwen35_verified_block(&[A, B, C], |row| decide(&mut projected, row, prepared.as_mut())).unwrap();
    assert_eq!(left, right);
    assert_eq!(right.output.into_iter().collect::<Vec<_>>(), vec![A, B, C]);
    assert!(right.terminal_after_pending);
    assert_eq!((right.matched_drafts, right.valid_input_tokens, right.carry_hidden_row), (3, 4, 3));
    assert_eq!(state.generated_tokens, vec![SEED]);
}

#[test]
fn mismatch_and_invalid_proposals_do_not_poison_later_decisions() {
    let qwen = model();
    let mut mismatch = state(&qwen, SEED, "root ::= \"abc\" | \"acb\"\n");
    sequence(&qwen, &mut mismatch, &[A, B, C],
        &[(A, A), (C, C), (B, B), (EOS, EOS)], "pacb");
    for invalid in [X, EMPTY, EOS] {
        let mut truncated = state(&qwen, SEED, "root ::= \"abc\"\n");
        sequence(&qwen, &mut truncated, &[A, invalid, C],
            &[(A, A), (B, B), (C, C), (EOS, EOS)], "pabc");
    }
}

#[test]
fn eos_before_acceptance_and_after_acceptance_preserves_ledger() {
    let qwen = model();
    let mut incomplete = state(&qwen, SEED, "root ::= \"a\"\n");
    sequence(&qwen, &mut incomplete, &[EOS], &[(A, A), (EOS, EOS)], "pa");
    let mut accepted = state(&qwen, SEED, "root ::= \"a\"\n");
    sequence(&qwen, &mut accepted, &[A, EOS, X], &[(A, A), (EOS, EOS)], "pa");
    let mut optional = state(&qwen, SEED, "root ::= \"a\"\n");
    optional.grammar_runtime.as_mut().unwrap().set_lazy_trigger(b"<tool_call>");
    sequence(&qwen, &mut optional, &[EOS], &[(EOS, EOS)], "p");
}

#[test]
fn lazy_split_and_same_token_marker_body_follow_native_acceptance() {
    let qwen = model();
    for drafts in [vec![MERGED, B, EOS], vec![OPEN_LEFT, OPEN_RIGHT, A, B, EOS]] {
        let mut state = state(&qwen, SEED, "root ::= \"ab\"\n");
        state.grammar_runtime.as_mut().unwrap().set_lazy_trigger(b"<tool_call>");
        state.tool_splitter = ToolCallSplitter::from_registration(&QWEN35);
        let decisions: Vec<_> = drafts.iter().map(|&token| (token, token)).collect();
        sequence(&qwen, &mut state, &drafts, &decisions, "p<tool_call>ab");
    }
}

#[test]
fn partial_utf8_and_decoder_driven_tool_trigger_are_not_replayed() {
    let qwen = model();
    let mut utf8 = state(&qwen, SEED, "root ::= \"éa\"\n");
    // EOS after an incomplete character truncates only the projection.
    sequence(&qwen, &mut utf8, &[LEAD, EOS],
        &[(LEAD, LEAD), (CONT, CONT), (A, A), (EOS, EOS)], "péa");
    let mut split = state(&qwen, LEAD, "root ::= \"a\"\n");
    split.grammar_runtime.as_mut().unwrap().set_awaiting_trigger(true);
    split.tool_splitter = ToolCallSplitter::from_registration(&QWEN35);
    sequence(&qwen, &mut split, &[CONT_OPEN, A, EOS],
        &[(CONT_OPEN, CONT_OPEN), (A, A), (EOS, EOS)], "é<tool_call>a");
}

#[test]
fn forced_token_matches_or_discards_the_hypothetical_prefix() {
    let qwen = model();
    for drafts in [vec![A, C, X], vec![B, C, A]] {
        let mut state = state(&qwen, SEED, "root ::= \"acx\" | \"bca\"\n");
        state.params.reasoning_forced_open = true;
        state.params.thinking_token_budget = Some(1);
        state.params.reasoning_end_tokens = Some(Arc::new(vec![B]));
        state.params.reasoning_close_tokens = Some(Arc::new(vec![B]));
        state.thinking_budget = Qwen35ThinkingBudgetState::from_params(&state.params);
        state.thinking_budget.as_mut().unwrap().observe_generated(&state.generated_tokens, false);
        sequence(&qwen, &mut state, &drafts,
            &[(A, B), (C, C), (A, A), (EOS, EOS)], "pbca");
    }
}

#[test]
fn projections_and_diagnostic_clones_are_isolated() {
    let qwen = model();
    let lane = state(&qwen, SEED, "root ::= \"abc\" | \"acb\"\n");
    let mut prepared = lane.prepare_dflash2_grammar(&qwen, &[A, B, C], false).unwrap().unwrap();
    let mut diagnostic = prepared.clone();
    let mut left = Qwen35SpecSemanticState::from_decode(&lane);
    let mut right = left.clone();
    for token in [A, B, C, EOS] {
        let decision = choose(&mut left, &qwen, &lane.params, token, Some(&mut prepared));
        assert_eq!(decision.token, token);
    }
    // Consuming one projection did not consume its diagnostic sibling.
    for token in [A, B, C, EOS] {
        assert_eq!(choose(&mut right, &qwen, &lane.params, token, Some(&mut diagnostic)).token, token);
    }
    let mut peer = state(&qwen, SEED, "root ::= \"acb\"\n");
    peer.slot_id = SlotId(1);
    sequence(&qwen, &mut peer, &[A, C, B], &[(A, A), (C, C), (B, B), (EOS, EOS)], "pacb");
    let mut untouched = Qwen35SpecSemanticState::from_decode(&lane);
    assert_eq!(choose(&mut untouched, &qwen, &lane.params, A, None).token, A);
    assert_eq!(choose(&mut untouched, &qwen, &lane.params, C, None).token, C);
}

#[test]
fn authoritative_metadata_errors_are_not_grammar_misses() {
    let qwen = model();
    let mut state = state(&qwen, SEED, "root ::= \"a\"\n");
    assert!(state.prepare_dflash2_grammar(&qwen, &[qwen.vocab_size as u32], false)
        .err().unwrap().to_string().contains("outside the authoritative vocabulary"));
    assert!(state.prepare_dflash2_grammar(&qwen, &[A; 8], false)
        .err().unwrap().to_string().contains("exceeds seven"));
    state.params.token_bytes = None;
    assert!(state.prepare_dflash2_grammar(&qwen, &[A], false)
        .err().unwrap().to_string().contains("requires authoritative token bytes"));
    state.params.token_bytes = Some(Arc::new(vec![b"a".to_vec()]));
    assert!(state.prepare_dflash2_grammar(&qwen, &[A], false)
        .err().unwrap().to_string().contains("table length"));
    state.grammar_runtime = None;
    state.params.grammar = None;
    state.params.token_bytes = None;
    for collect_work in [false, true] {
        assert!(state.prepare_dflash2_grammar(&qwen, &[A], collect_work).unwrap().is_none());
    }
    assert_eq!(choose(&mut Qwen35SpecSemanticState::from_decode(&state),
        &qwen, &state.params, X, None).token, X);
}

#[test]
fn work_tracks_matching_prefix_consumption_and_discard_without_clone_leakage() {
    use DFlash2GrammarOutcome::{DivergenceDiscarded, PlanningConsumed, Prepared};
    let qwen = model();
    let state = state(&qwen, SEED, "root ::= \"abc\" | \"acb\"\n");
    let mut prepared = state.prepare_dflash2_grammar(&qwen, &[A, B, C], true)
        .unwrap().unwrap();
    let mut diagnostic = prepared.clone();
    let mut semantic = Qwen35SpecSemanticState::from_decode(&state);
    for token in [A, C, B] {
        assert_eq!(choose(&mut semantic, &qwen, &state.params, token,
            Some(&mut prepared)).token, token);
    }
    let work = prepared.work().unwrap();
    assert_eq!(work.actual_proposals, 3);
    assert_eq!(work.attempts.iter().flatten().map(|attempt| attempt.outcome)
        .collect::<Vec<_>>(), [PlanningConsumed, DivergenceDiscarded, DivergenceDiscarded]);
    assert!(diagnostic.work().unwrap().attempts.iter().flatten()
        .all(|attempt| attempt.outcome == Prepared));
    let mut independent = Qwen35SpecSemanticState::from_decode(&state);
    for token in [A, B, C] {
        assert_eq!(choose(&mut independent, &qwen, &state.params, token,
            Some(&mut diagnostic)).token, token);
    }
    assert!(diagnostic.work().unwrap().attempts.iter().flatten()
        .all(|attempt| attempt.outcome == PlanningConsumed));
    assert!(state.prepare_dflash2_grammar(&qwen, &[A], false)
        .unwrap().unwrap().work().is_none());
    let empty = state.prepare_dflash2_grammar(&qwen, &[], true)
        .unwrap().unwrap().work().unwrap();
    assert_eq!(empty.actual_proposals, 0);
    assert!(empty.attempts.iter().all(Option::is_none));
}

#[test]
fn work_does_not_count_invalid_or_eos_tail_as_prepared_or_consumed() {
    use DFlash2GrammarOutcome::{Invalid, PlanningConsumed};
    let qwen = model();
    let incomplete = state(&qwen, SEED, "root ::= \"abc\"\n");
    let mut prepared = incomplete.prepare_dflash2_grammar(&qwen, &[A, X, C], true)
        .unwrap().unwrap();
    let mut semantic = Qwen35SpecSemanticState::from_decode(&incomplete);
    for token in [A, B] {
        assert_eq!(choose(&mut semantic, &qwen, &incomplete.params, token,
            Some(&mut prepared)).token, token);
    }
    let work = prepared.work().unwrap();
    assert_eq!(work.actual_proposals, 3);
    assert_eq!(work.attempts[0].unwrap().outcome, PlanningConsumed);
    assert_eq!(work.attempts[1].unwrap().outcome, Invalid);
    assert!(work.attempts[2..].iter().all(Option::is_none));

    let complete = state(&qwen, SEED, "root ::= \"a\"\n");
    let mut prepared = complete.prepare_dflash2_grammar(&qwen, &[A, EOS, X], true)
        .unwrap().unwrap();
    let mut semantic = Qwen35SpecSemanticState::from_decode(&complete);
    for token in [A, EOS] {
        assert_eq!(choose(&mut semantic, &qwen, &complete.params, token,
            Some(&mut prepared)).token, token);
    }
    let work = prepared.work().unwrap();
    assert_eq!(work.actual_proposals, 3);
    assert!(work.attempts[1].unwrap().eos);
    assert_eq!(work.attempts[1].unwrap().outcome, PlanningConsumed);
    assert!(work.attempts[2..].iter().all(Option::is_none));
}

#[test]
fn work_separates_lazy_tracking_from_active_consumed_transitions() {
    use DFlash2GrammarOutcome::PlanningConsumed;
    let qwen = model();
    let mut state = state(&qwen, SEED, "root ::= \"ab\"\n");
    state.grammar_runtime.as_mut().unwrap().set_lazy_trigger(b"<tool_call>");
    state.tool_splitter = ToolCallSplitter::from_registration(&QWEN35);
    let mut prepared = state.prepare_dflash2_grammar(&qwen, &[MERGED, B, EOS], true)
        .unwrap().unwrap();
    let mut semantic = Qwen35SpecSemanticState::from_decode(&state);
    for token in [MERGED, B, EOS] {
        assert_eq!(choose(&mut semantic, &qwen, &state.params, token,
            Some(&mut prepared)).token, token);
    }
    let work = prepared.work().unwrap();
    assert!(work.awaiting_trigger_at_start);
    assert_eq!(work.attempts.iter().flatten().map(|attempt| attempt.awaiting_trigger)
        .collect::<Vec<_>>(), [true, false, false]);
    assert!(work.attempts.iter().flatten()
        .all(|attempt| attempt.outcome == PlanningConsumed));
}

fn checked_sequence(
    qwen: &Qwen35LoadedModel,
    state: &Qwen35DecodeState,
    drafts: &[u32],
    decisions: &[(u32, u32)],
) {
    let mut projected = Qwen35SpecSemanticState::from_decode(state);
    let mut prepared = state.prepare_dflash2_grammar(qwen, drafts, true)
        .unwrap().unwrap();
    let mut ordinary = OrdinaryDFlash2Check::begin(
        Some(&prepared), &projected, &state.params,
    ).unwrap();
    for &(preferred, expected) in decisions {
        let mut logits = vec![-100.0; qwen.vocab_size];
        logits[preferred as usize] = 20.0;
        let decision = projected.select_and_observe_with_ordinary_check(
            qwen, &state.params, Qwen35DecisionSource::Logits(&mut logits),
            Some(&mut prepared), Some(&mut ordinary),
        ).unwrap();
        assert_eq!(decision.token, expected);
        assert_eq!(decision.terminal, expected == EOS);
    }
    assert_eq!(ordinary.rows(), decisions.len());
    assert_eq!(projected.generated_tokens, ordinary.semantic.generated_tokens);
    assert_eq!(projected.sampling_history, ordinary.semantic.sampling_history);
}

#[test]
fn persistent_ordinary_check_covers_mismatch_lazy_forced_and_terminal_rows() {
    let qwen = model();
    let mut mismatch = state(&qwen, SEED, "root ::= \"abc\" | \"acb\"\n");
    mismatch.params.temperature = 0.0;
    checked_sequence(&qwen, &mismatch, &[A, B, C],
        &[(A, A), (C, C), (B, B), (EOS, EOS)]);

    let mut lazy = state(&qwen, SEED, "root ::= \"ab\"\n");
    lazy.params.temperature = 0.0;
    lazy.grammar_runtime.as_mut().unwrap().set_lazy_trigger(b"<tool_call>");
    lazy.tool_splitter = ToolCallSplitter::from_registration(&QWEN35);
    checked_sequence(&qwen, &lazy, &[MERGED, B, EOS],
        &[(MERGED, MERGED), (B, B), (EOS, EOS)]);

    let mut forced = state(&qwen, SEED, "root ::= \"acx\" | \"bca\"\n");
    forced.params.temperature = 0.0;
    forced.params.reasoning_forced_open = true;
    forced.params.thinking_token_budget = Some(1);
    forced.params.reasoning_end_tokens = Some(Arc::new(vec![B]));
    forced.params.reasoning_close_tokens = Some(Arc::new(vec![B]));
    forced.thinking_budget = Qwen35ThinkingBudgetState::from_params(&forced.params);
    forced.thinking_budget.as_mut().unwrap()
        .observe_generated(&forced.generated_tokens, false);
    checked_sequence(&qwen, &forced, &[A, C, X],
        &[(A, B), (C, C), (A, A), (EOS, EOS)]);
}

#[test]
fn persistent_ordinary_check_detects_a_later_poisoned_snapshot_decision() {
    let qwen = model();
    let mut original = state(&qwen, SEED, "root ::= \"abc\"\n");
    original.params.temperature = 0.0;
    let wrong = state(&qwen, SEED, "root ::= \"acb\"\n");
    let mut prepared = original.prepare_dflash2_grammar(&qwen, &[A, B, C], true)
        .unwrap().unwrap();
    let mut poison = wrong.prepare_dflash2_grammar(&qwen, &[A, C, B], false)
        .unwrap().unwrap();
    prepared.transitions[0] = poison.transitions[0].take();
    let mut projected = Qwen35SpecSemanticState::from_decode(&original);
    let mut ordinary = OrdinaryDFlash2Check::begin(
        Some(&prepared), &projected, &original.params,
    ).unwrap();
    for (index, token) in [A, B].into_iter().enumerate() {
        let mut logits = vec![-100.0; qwen.vocab_size];
        logits[token as usize] = 20.0;
        let result = projected.select_and_observe_with_ordinary_check(
            &qwen, &original.params, Qwen35DecisionSource::Logits(&mut logits),
            Some(&mut prepared), Some(&mut ordinary),
        );
        if index == 0 {
            assert_eq!(result.unwrap().token, A);
        } else {
            assert!(result.is_err());
        }
    }
    assert_eq!(ordinary.rows(), 1);
    assert_eq!(projected.generated_tokens, [SEED, A, C]);
    assert_eq!(ordinary.semantic.generated_tokens, [SEED, A, B]);
}
