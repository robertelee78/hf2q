//! Repeatable speculation route (ADR-063, `adr063-qwen-spec-repeatable.md`):
//! a tick's kind comes from the request's own state only, and the joint
//! partition admits only ordinary ticks. Model-free; MTP-holding states use a
//! host-allocated verifier-hidden row.

use std::time::Duration;

use mlx_native::MlxDevice;

use super::speculation_qualification_tests::loaded_model;
use super::{Qwen35DecodeState, Qwen35LoadedModel, Qwen35SlotMtpState, Qwen35TickKind};
use crate::inference::spec_decode::cost_controller::SpeculationCostController;
use crate::inference::spec_decode::ngram_proposer::{HistoryLookupConfig, HistoryLookupIndex};

use crate::serve::api::engine::SamplingParams;
use crate::serve::api::qwen35_speculation::{QwenSpeculationPolicy, QwenSpeculationRoute};
use crate::serve::multi_seq_kv::SlotId;

/// `[1..=8, 9, 10, 11, 1..=7]`: with the seeded `8`, the six-token suffix
/// `3..=8` recurs and the history proposes `9, 10, 11`.
const PROMPT: [u32; 18] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7];
const SEED: u32 = 8;

fn state(qwen: &Qwen35LoadedModel, max_tokens: usize) -> Qwen35DecodeState {
    let mut logits = vec![0.0_f32; qwen.vocab_size];
    logits[SEED as usize] = 10.0;
    let params = SamplingParams {
        max_tokens,
        ..SamplingParams::default()
    };
    let mut state = Qwen35DecodeState::from_prefill_logits(
        qwen,
        PROMPT.to_vec(),
        params,
        None,
        SlotId(0),
        0,
        &logits,
        Duration::ZERO,
        PROMPT.len(),
        None,
    )
    .expect("seed Qwen decode state");
    assert!(!state.finished_at_seed());
    state.history_lookup = None;
    state
}

fn with_history(state: &mut Qwen35DecodeState) {
    let mut lookup = HistoryLookupIndex::new(HistoryLookupConfig {
        min_match: 6,
        max_match: 12,
        max_draft_tokens: 3,
        max_model_len: 1024,
    });
    lookup.reset(&state.prompt_tokens);
    state.history_lookup = Some(lookup);
}

fn with_mtp(state: &mut Qwen35DecodeState, device: &MlxDevice) {
    let verifier_hidden = device
        .alloc_buffer(
            64 * std::mem::size_of::<f32>(),
            mlx_native::DType::F32,
            vec![1, 64],
        )
        .expect("verifier hidden row");
    state.mtp = Some(Qwen35SlotMtpState { verifier_hidden });
}

fn warm(state: &mut Qwen35DecodeState, elapsed: Duration) {
    state.mtp_cost.observe_ordinary_target(elapsed);
    state.history_cost.observe_ordinary_target(elapsed);
}

#[test]
fn pending_verified_output_is_a_commit_tick() {
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);
    let mut state = state(&qwen, 64);
    state.pending_speculation_output.push_back(12);
    assert_eq!(state.next_tick_kind().unwrap(), Qwen35TickKind::Commit);
}

#[test]
fn history_drafts_plan_a_history_round_only_after_an_ordinary_observation() {
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);
    let mut state = state(&qwen, 64);
    with_history(&mut state);
    assert_eq!(state.next_tick_kind().unwrap(), Qwen35TickKind::Ordinary);
    warm(&mut state, Duration::from_millis(30));
    assert_eq!(
        state.next_tick_kind().unwrap(),
        Qwen35TickKind::History(vec![9, 10, 11])
    );
    // The ledger is synchronized from the committed output, as before.
    assert_eq!(
        state.history_lookup.as_ref().unwrap().verified_len(),
        PROMPT.len() + state.generated_tokens.len()
    );
    // Drafts never run past the output limit: two tokens left keep one draft.
    let mut short = self::state(&qwen, 3);
    with_history(&mut short);
    warm(&mut short, Duration::from_millis(30));
    assert_eq!(
        short.next_tick_kind().unwrap(),
        Qwen35TickKind::History(vec![9])
    );
}

#[test]
fn mtp_plans_k3_only_with_four_outputs_left_and_an_open_gate() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);

    let mut admitted = state(&qwen, 64);
    with_mtp(&mut admitted, &device);
    assert_eq!(
        admitted.next_tick_kind().unwrap(),
        Qwen35TickKind::Ordinary,
        "warm-up first"
    );
    warm(&mut admitted, Duration::from_millis(30));
    assert_eq!(admitted.next_tick_kind().unwrap(), Qwen35TickKind::K3);

    let mut tail = state(&qwen, 4);
    with_mtp(&mut tail, &device);
    warm(&mut tail, Duration::from_millis(30));
    assert_eq!(
        tail.next_tick_kind().unwrap(),
        Qwen35TickKind::Ordinary,
        "three outputs left"
    );

    // A modelled gate disabled by two windows of one decision per round.
    let mut disabled = state(&qwen, 64);
    with_mtp(&mut disabled, &device);
    disabled.mtp_cost = SpeculationCostController::modelled(1.26);
    warm(&mut disabled, Duration::from_millis(30));
    for _ in 0..8 {
        disabled
            .mtp_cost
            .observe_speculative_round(1, Duration::from_millis(1));
    }
    assert_eq!(disabled.next_tick_kind().unwrap(), Qwen35TickKind::Ordinary);
}

#[test]
fn history_miss_with_a_cost_disabled_mtp_drops_mtp_and_plans_ordinary() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);
    let mut state = state(&qwen, 64);
    // A history that cannot match: the seed never recurs.
    state.generated_tokens = vec![40];
    with_history(&mut state);
    with_mtp(&mut state, &device);
    warm(&mut state, Duration::from_millis(30));
    // History misses, MTP is open: a K3 round, MTP kept.
    assert_eq!(state.next_tick_kind().unwrap(), Qwen35TickKind::K3);
    assert!(state.mtp.is_some());
    state.mtp_cost = SpeculationCostController::modelled(1.26);
    state
        .mtp_cost
        .observe_ordinary_target(Duration::from_millis(30));
    for _ in 0..8 {
        state
            .mtp_cost
            .observe_speculative_round(1, Duration::from_millis(1));
    }
    assert_eq!(state.next_tick_kind().unwrap(), Qwen35TickKind::Ordinary);
    assert!(
        state.mtp.is_none(),
        "a cost-disabled MTP is dropped on a history miss"
    );
}

/// Two identical requests whose ordinary decisions took different wall time
/// plan the same ticks under the repeatable (modelled) gates.
#[test]
fn repeatable_gates_plan_identically_under_different_timing() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);
    assert!(
        qwen.speculation.cost_model().is_some(),
        "repeatable is the default route"
    );
    let plan = |ordinary: Duration, round: Duration| {
        let mut state = state(&qwen, 64);
        with_mtp(&mut state, &device);
        warm(&mut state, ordinary);
        let mut kinds = Vec::new();
        for decisions in [1, 1, 1, 1, 2, 1, 1, 1, 4, 1] {
            let kind = state.next_tick_kind().unwrap();
            if kind == Qwen35TickKind::K3 {
                state.mtp_cost.observe_speculative_round(decisions, round);
            }
            kinds.push(kind);
        }
        kinds
    };
    let fast = plan(Duration::from_millis(90), Duration::from_millis(5));
    let slow = plan(Duration::from_millis(5), Duration::from_millis(900));
    assert_eq!(fast, slow);
    assert!(fast.contains(&Qwen35TickKind::K3) && fast.contains(&Qwen35TickKind::Ordinary));
}

/// Under the repeatable route only a planned ordinary tick may join a joint
/// step; a speculating, committing or unplanned lane stays serial. The
/// adaptive route keeps the phase-1 class.
#[test]
fn joint_partition_admits_only_planned_ordinary_ticks_under_the_repeatable_route() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);

    let mut ordinary = state(&qwen, 64);
    assert_eq!(ordinary.joint_lane_class(), None, "unplanned");
    assert_eq!(
        ordinary.plan_tick().unwrap(),
        Some(&Qwen35TickKind::Ordinary)
    );
    assert_eq!(ordinary.joint_lane_class(), Some(true));

    let mut k3 = state(&qwen, 64);
    with_mtp(&mut k3, &device);
    warm(&mut k3, Duration::from_millis(30));
    assert_eq!(k3.plan_tick().unwrap(), Some(&Qwen35TickKind::K3));
    assert_eq!(k3.joint_lane_class(), None);

    let mut history = state(&qwen, 64);
    with_history(&mut history);
    warm(&mut history, Duration::from_millis(30));
    assert!(matches!(
        history.plan_tick().unwrap(),
        Some(Qwen35TickKind::History(_))
    ));
    assert_eq!(history.joint_lane_class(), None);

    // A warmed-up MTP lane whose tick is ordinary (fewer than four left)
    // joins as its coherent ordinary row.
    let mut coherent = state(&qwen, 3);
    with_mtp(&mut coherent, &device);
    warm(&mut coherent, Duration::from_millis(30));
    assert_eq!(
        coherent.plan_tick().unwrap(),
        Some(&Qwen35TickKind::Ordinary)
    );
    assert_eq!(coherent.joint_lane_class(), Some(true));

    // A lane at its step bound gets no plan, so it stays serial and
    // `decode_tick` finishes it without consulting a proposer.
    let mut bound = state(&qwen, 64);
    with_history(&mut bound);
    warm(&mut bound, Duration::from_millis(30));
    bound.step = bound.max_tokens;
    assert_eq!(bound.plan_tick().unwrap(), None);
    assert_eq!(bound.joint_lane_class(), None);

    let mut adaptive = state(&qwen, 64);
    with_mtp(&mut adaptive, &device);
    warm(&mut adaptive, Duration::from_millis(30));
    adaptive.route = QwenSpeculationRoute::Adaptive;
    assert_eq!(
        adaptive.joint_lane_class(),
        Some(false),
        "phase 1: joins only in company"
    );
}
