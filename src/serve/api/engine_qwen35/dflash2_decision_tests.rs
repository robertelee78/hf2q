//! Differential test of the decision candidates
//! (`HF2Q_DFLASH2_GPU_DECISIONS=1`) at the level the served text is decided:
//! the plan of a verified window.
//!
//! Each case is one recorded verify window: a request state (repetition
//! window, thinking budget, grammar), eight rows of target logits and seven
//! drafts. The window is planned twice from the same state, by the host
//! sampler over every row's logits (today's route) and from the candidates
//! the kernels return for those logits, and the two plans must be equal:
//! the output tokens, the terminal flag, the matched and rejected drafts and
//! the retained rows. The cases cover what this workload's requests carry:
//!
//! - a repetition penalty over histories of every length, with the target's
//!   best token inside the history in most rows;
//! - a thinking budget whose forced close starts inside the window, with
//!   the drafts agreeing and disagreeing with the forced tokens;
//! - a lazy tool grammar that is waiting (candidates decide), that is
//!   triggered by a token inside the window, and that constrains the whole
//!   window (those rows must take the host sampler over their own logits);
//! - end-of-generation decisions, full, partial and zero acceptance.
//!
//! The test also holds the route itself: rows without an active grammar are
//! decided by the candidates (no silent fallback), rows under an active
//! grammar never are. Needs Apple Metal:
//!
//! ```text
//! cargo test --release --bin hf2q dflash2_scheduler::decision_tests
//! ```

use std::sync::Arc;
use std::time::Duration;

use mlx_native::{KernelRegistry, MlxDevice};

use super::super::speculation_qualification_tests::loaded_model;
use super::super::{Qwen35SpecSemanticState, Qwen35ThinkingBudgetState};
use super::*;
use crate::inference::models::qwen35::verify_decision::candidates_for_test;
use crate::serve::api::engine::{SamplingParams, ToolCallPolicy};
use crate::serve::api::grammar::GrammarRuntime;
use crate::serve::api::qwen35_speculation::QwenSpeculationPolicy;
use crate::serve::multi_seq_kv::SlotId;

const VOCAB: usize = 64;
const EOS: u32 = 63;
const ROWS: usize = 8;
/// Thinking-budget close: a transition token, then the close marker. Both
/// are letters the grammar admits, and no target row prefers them.
const FORCED: [u32; 2] = [52, 53];
/// The token whose text triggers the lazy grammar.
const TRIGGER: u32 = 40;

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 33
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Grammar {
    None,
    /// Waiting for [`TRIGGER`]; a row whose decision is the trigger starts it.
    Waiting,
    /// Triggered before the window.
    Active,
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    penalty: f32,
    budget: bool,
    grammar: Grammar,
}

/// Token text: a letter per token, so the grammar `[a-m]+` admits tokens
/// 0..13 of each 26; the trigger token is `<`, end of generation is empty.
fn token_bytes() -> Vec<Vec<u8>> {
    (0..VOCAB as u32)
        .map(|token| match token {
            EOS => Vec::new(),
            TRIGGER => b"<".to_vec(),
            token => vec![b'a' + (token % 26) as u8],
        })
        .collect()
}

fn state(qwen: &Qwen35LoadedModel, shape: Shape, seed: &mut u64) -> Qwen35DecodeState {
    let prompt_len = 1 + (next(seed) % 90) as usize;
    let prompt = (0..prompt_len)
        .map(|_| (next(seed) % 49) as u32)
        .collect::<Vec<_>>();
    let mut params = SamplingParams {
        max_tokens: 256,
        repetition_penalty: shape.penalty,
        ..SamplingParams::default()
    };
    if shape.budget {
        params.reasoning_forced_open = true;
        params.thinking_token_budget = Some(1 + (next(seed) % 6) as usize);
        params.reasoning_end_tokens = Some(Arc::new(FORCED.to_vec()));
        params.reasoning_close_tokens = Some(Arc::new(vec![FORCED[1]]));
    }
    let grammar = (shape.grammar != Grammar::None)
        .then(|| crate::serve::api::grammar::parse("root ::= [a-m]+\n").expect("letter grammar"));
    if let Some(grammar) = grammar.as_ref() {
        params.grammar = Some(grammar.clone());
        params.token_bytes = Some(Arc::new(token_bytes()));
        params.tool_call_policy = ToolCallPolicy::AutoLazyGrammar;
    }
    let mut seed_logits = vec![0.0_f32; VOCAB];
    seed_logits[(next(seed) % 49) as usize] = 10.0;
    // Seeded without the grammar (the seed token is free), then armed.
    let mut seed_params = params.clone();
    seed_params.grammar = None;
    seed_params.token_bytes = None;
    seed_params.tool_call_policy = ToolCallPolicy::Auto;
    let mut state = Qwen35DecodeState::from_prefill_logits(
        qwen,
        prompt.clone(),
        seed_params,
        None,
        SlotId(0),
        0,
        &seed_logits,
        Duration::ZERO,
        prompt.len(),
        None,
    )
    .expect("seed decode state");
    state.history_lookup = None;
    state.params = params;
    state.thinking_budget = Qwen35ThinkingBudgetState::from_params(&state.params);
    assert_eq!(state.thinking_budget.is_some(), shape.budget);
    if let Some(grammar) = grammar {
        let root = grammar.rule_id("root").unwrap();
        let mut runtime = GrammarRuntime::new(grammar, root).unwrap();
        if shape.grammar == Grammar::Waiting {
            runtime.set_lazy_trigger(b"<");
        }
        state.grammar_runtime = Some(runtime);
    }
    state
}

/// Eight rows of logits whose two best tokens are close enough for the
/// penalty to choose between them; the best one is drawn from the recent
/// tokens in most rows.
fn target_rows(state: &Qwen35DecodeState, shape: Shape, seed: &mut u64) -> Vec<Vec<f32>> {
    (0..ROWS)
        .map(|row| {
            let mut logits = (0..VOCAB)
                .map(|_| (next(seed) % 2_000) as f32 / 500.0 - 2.0)
                .collect::<Vec<_>>();
            let recent = &state.sampling_history;
            let top = match next(seed) % 10 {
                0..=5 if !recent.is_empty() => {
                    recent[(next(seed) % recent.len() as u64) as usize] as usize
                }
                6 if shape.grammar == Grammar::Waiting && row > 0 => TRIGGER as usize,
                7 if row > 1 => EOS as usize,
                _ => (next(seed) % 49) as usize,
            };
            let runner_up = (next(seed) % 49) as usize;
            logits[runner_up] = 9.8;
            logits[top] = 10.0 + (next(seed) % 5) as f32 / 10.0;
            logits
        })
        .collect()
}

/// The decisions the host sampler makes along the window when every draft
/// agrees: what a perfect drafter would have proposed.
fn canonical(qwen: &Qwen35LoadedModel, state: &Qwen35DecodeState, rows: &[Vec<f32>]) -> Vec<u32> {
    let mut semantic = Qwen35SpecSemanticState::from_decode(state);
    let mut tokens = Vec::new();
    for row in rows {
        let Ok(decision) = semantic.select_and_observe(qwen, &state.params, &mut row.clone())
        else {
            break;
        };
        tokens.push(decision.token);
        if decision.terminal {
            break;
        }
    }
    tokens
}

#[test]
fn candidate_plans_are_the_host_samplers_plans() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut registry = KernelRegistry::new();
    let qwen = loaded_model(QwenSpeculationPolicy::Off);
    assert_eq!(qwen.vocab_size, VOCAB);
    assert_eq!(qwen.eos_token_ids, [EOS]);

    let mut seed = 0x5eed_u64;
    let (mut windows, mut decided, mut fallen_back) = (0, 0, 0);
    let (mut forced_closes, mut triggered, mut terminal, mut full, mut zero) = (0, 0, 0, 0, 0);
    let mut penalty_decided = 0;
    for case in 0..600 {
        let shape = Shape {
            penalty: [1.0, 1.05, 1.05, 1.3][case % 4],
            budget: case % 3 == 1,
            grammar: [
                Grammar::None,
                Grammar::Waiting,
                Grammar::None,
                Grammar::Active,
            ][(case / 4) % 4],
        };
        let state = state(&qwen, shape, &mut seed);
        let rows = target_rows(&state, shape, &mut seed);
        let ideal = canonical(&qwen, &state, &rows);

        // Drafts: the ideal ones up to a random row, then something else.
        let agree = (next(&mut seed) % 9) as usize;
        let mut drafts = Vec::with_capacity(ROWS - 1);
        for row in 0..ROWS - 1 {
            let wrong = (next(&mut seed) % 62) as u32;
            drafts.push(match ideal.get(row) {
                Some(&token) if row < agree => token,
                Some(&token) if wrong == token => (token + 1) % 62,
                _ => wrong,
            });
        }
        let mut tokens = vec![*state.generated_tokens.last().unwrap()];
        tokens.extend_from_slice(&drafts);

        let window = Qwen35DFlash2Window {
            pending_token: tokens[0],
            position: 0,
            greedy: false,
            maximum_rows: ROWS,
            decisions: DecisionRoute::Check,
        };
        assert!(
            state.decision_candidates_eligible(&window),
            "case {case} {shape:?}: the request is not represented"
        );
        let flat = rows.iter().flatten().copied().collect::<Vec<_>>();
        let candidates = candidates_for_test(
            &device,
            &mut registry,
            VOCAB,
            &[(0, ROWS)],
            &[Some(state.sampling_history.as_slice())],
            &tokens,
            &flat,
        );
        let mut outputs = rows
            .iter()
            .zip(candidates)
            .map(|(logits, decision)| Qwen35JointRowOutput {
                greedy_token: None,
                logits: Some(logits.clone()),
                nextn_hidden: None,
                decision,
            })
            .collect::<Vec<_>>();

        let host = state
            .plan_dflash2_rows(
                &qwen,
                &window,
                &drafts,
                &mut outputs,
                DecisionRoute::HostLogits,
            )
            .expect("host plan");
        let candidate = state
            .plan_dflash2_rows(
                &qwen,
                &window,
                &drafts,
                &mut outputs,
                DecisionRoute::Candidates,
            )
            .expect("candidate plan");
        assert_eq!(
            candidate.plan, host.plan,
            "case {case} {shape:?}: the candidates plan another window"
        );
        assert!(
            outputs
                .iter()
                .zip(&rows)
                .all(|(output, row)| output.logits.as_deref() == Some(row.as_slice())),
            "case {case}: a checked plan changed the rows it was given"
        );
        // The served entry point in check mode returns the host's plan and
        // reports no mismatch.
        let before = DECISION_MISMATCHES.load(std::sync::atomic::Ordering::Relaxed);
        let checked = state
            .plan_selected_dflash2_window(&qwen, &window, &tokens, outputs)
            .expect("checked plan");
        assert_eq!(checked.plan, host.plan);
        assert_eq!(
            DECISION_MISMATCHES.load(std::sync::atomic::Ordering::Relaxed),
            before
        );

        // The route: every planned row had candidates; without an active
        // grammar the candidates decided it.
        assert_eq!(host.candidate_rows + host.fallback_rows, 0);
        assert!(
            candidate.candidate_rows + candidate.fallback_rows >= 1,
            "case {case}: no row was planned"
        );
        match shape.grammar {
            Grammar::None => assert_eq!(
                candidate.fallback_rows, 0,
                "case {case} {shape:?}: a row without a grammar left the candidates"
            ),
            Grammar::Active => assert_eq!(
                candidate.candidate_rows, 0,
                "case {case}: a grammar-constrained row was decided by candidates"
            ),
            Grammar::Waiting => {}
        }
        windows += 1;
        decided += candidate.candidate_rows;
        fallen_back += candidate.fallback_rows;
        penalty_decided += usize::from(shape.penalty != 1.0 && candidate.candidate_rows > 0);
        let output = &host.plan.output;
        forced_closes +=
            usize::from(shape.budget && output.iter().any(|&token| token == FORCED[0]));
        triggered += usize::from(
            shape.grammar == Grammar::Waiting
                && output.contains(&TRIGGER)
                && candidate.fallback_rows > 0,
        );
        terminal += usize::from(host.plan.terminal_after_pending);
        full += usize::from(host.plan.matched_drafts == drafts.len());
        zero += usize::from(host.plan.matched_drafts == 0);
    }
    eprintln!(
        "decision plans: {windows} windows, {decided} candidate rows, {fallen_back} fallback rows, \
         {forced_closes} forced closes, {triggered} triggers inside a window, {terminal} terminal, \
         {full} full, {zero} zero acceptance"
    );
    // The cases are what the header says they are.
    assert!(decided > 1_000 && fallen_back > 100 && penalty_decided > 200);
    assert!(forced_closes > 20 && triggered > 5 && terminal > 10 && full > 10 && zero > 20);
}

/// Requests the candidates do not represent keep the host sampler.
#[test]
fn unrepresented_requests_keep_the_host_sampler() {
    let qwen = loaded_model(QwenSpeculationPolicy::Off);
    let mut seed = 7_u64;
    let shape = Shape {
        penalty: 1.05,
        budget: false,
        grammar: Grammar::None,
    };
    let window = |greedy| Qwen35DFlash2Window {
        pending_token: 1,
        position: 0,
        greedy,
        maximum_rows: ROWS,
        decisions: DecisionRoute::HostLogits,
    };
    let base = state(&qwen, shape, &mut seed);
    assert!(base.decision_candidates_eligible(&window(false)));
    assert!(!base.decision_candidates_eligible(&window(true)));

    let mut logprobs = state(&qwen, shape, &mut seed);
    logprobs.want_logprobs = true;
    assert!(!logprobs.decision_candidates_eligible(&window(false)));

    type Edit = fn(&mut SamplingParams);
    let edits: [Edit; 8] = [
        |params| params.temperature = 0.7,
        |params| params.top_k = 40,
        |params| params.top_p = 0.9,
        |params| params.min_p = 0.05,
        |params| params.seed = Some(3),
        |params| {
            params.logit_bias.insert(4, 2.0);
        },
        |params| params.stop_strings.push("stop".to_owned()),
        |params| params.frequency_penalty = 0.5,
    ];
    for (index, edit) in edits.iter().enumerate() {
        let mut state = state(&qwen, shape, &mut seed);
        edit(&mut state.params);
        assert!(
            !state.decision_candidates_eligible(&window(false)),
            "edit {index} left the request eligible"
        );
    }

    assert_eq!(parse_decision_route(None), DecisionRoute::HostLogits);
    assert_eq!(parse_decision_route(Some("0")), DecisionRoute::HostLogits);
    assert_eq!(
        parse_decision_route(Some("true")),
        DecisionRoute::HostLogits
    );
    assert_eq!(parse_decision_route(Some("1")), DecisionRoute::Candidates);
    assert_eq!(parse_decision_route(Some("check")), DecisionRoute::Check);
}
