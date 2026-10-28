use super::*;
use crate::inference::spec_decode::cost_controller::SpeculationCostController;
use crate::serve::api::engine::SamplingParams;
use super::super::{is_greedy_eligible, is_qwen_server_speculation_exact_eligible};
use std::time::Duration;

fn cold() -> Facts {
    Facts { step: 1, max_tokens: 8192, generated_tokens: 1, pending_outputs: 0,
        history_present: true, history_may_speculate: false, mtp_present: true,
        mtp_may_speculate: false, greedy_without_logprobs: false }
}
fn handle(slot: u32, generation: u64) -> SlotHandle { SlotHandle { slot_id: SlotId(slot), generation } }

#[test]
fn real_profile_is_constrained_deterministic_not_gpu_greedy() {
    let params = SamplingParams { temperature: 0.0, repetition_penalty: 1.05,
        grammar: Some(crate::serve::api::grammar::parse("root ::= \"<tool_call>\"\n").unwrap()),
        thinking_token_budget: Some(2048), ..SamplingParams::default() };
    assert!(!is_greedy_eligible(&params));
    assert!(is_qwen_server_speculation_exact_eligible(&params));
    let history = SpeculationCostController::new();
    let mtp = SpeculationCostController::new();
    let before = (format!("{history:?}"), format!("{mtp:?}"));
    let action = classify(Facts { history_may_speculate: history.may_speculate(),
        mtp_may_speculate: mtp.may_speculate(),
        greedy_without_logprobs: is_greedy_eligible(&params) && !params.logprobs,
        ..cold() });
    assert_eq!(action.dispatch, Dispatch::History); // wrapper is preserved
    assert_eq!(action.ordinary, OrdinaryLeaf::CoherentMtpFullLogits);
    assert_eq!(before, (format!("{history:?}"), format!("{mtp:?}")));
}

#[test]
fn actual_cost_warmup_changes_known_leaf_without_disabling_auto() {
    let mut mtp = SpeculationCostController::new();
    assert_eq!(classify(cold()).ordinary, OrdinaryLeaf::CoherentMtpFullLogits);
    mtp.observe_ordinary_target(Duration::from_millis(10));
    let f = Facts { mtp_may_speculate: mtp.may_speculate(), ..cold() };
    assert_eq!(classify(f).dispatch, Dispatch::History);
    assert_eq!(classify(f).ordinary, OrdinaryLeaf::MtpProposal);
    // Actual complete nonprofitable windows disable only the cost gate.
    for _ in 0..8 { mtp.observe_speculative_round(1, Duration::from_millis(11)); }
    assert!(!mtp.may_speculate());
    assert_eq!(classify(Facts { mtp_may_speculate: mtp.may_speculate(), ..cold() }).ordinary,
        OrdinaryLeaf::CoherentMtpFullLogits);
}

#[test]
fn active_history_requires_real_lookup_decision_not_option_guess() {
    let a = classify(Facts { history_may_speculate: true, ..cold() });
    assert_eq!(a.dispatch, Dispatch::History);
    assert_eq!(a.ordinary, OrdinaryLeaf::HistoryDecisionRequired);
    assert_eq!(preview(a, SlotId(2), handle(2, 7), handle(2, 7), false), OwnerPreview::NotKnownOrdinary);
}

#[test]
fn verified_output_and_terminal_bound_precede_every_forward_leaf() {
    let a = classify(Facts { pending_outputs: 3, ..cold() });
    assert_eq!(a, NextAction { dispatch: Dispatch::EmitVerified, ordinary: OrdinaryLeaf::NoTarget });
    assert_eq!(preview(a, SlotId(0), handle(0, 7), handle(0, 7), false), OwnerPreview::NotKnownOrdinary);
    assert_eq!(classify(Facts { step: 8192, pending_outputs: 3, ..cold() }),
        NextAction { dispatch: Dispatch::Finished, ordinary: OrdinaryLeaf::NoTarget });
}

#[test]
fn mtp_tail_and_plain_greedy_remain_distinct() {
    let a = classify(Facts { history_present: false, mtp_may_speculate: true,
        generated_tokens: 8189, ..cold() });
    assert_eq!(a.dispatch, Dispatch::Mtp);
    assert_eq!(a.ordinary, OrdinaryLeaf::CoherentMtpFullLogits);
    assert_eq!(classify(Facts { history_present: false, mtp_present: false,
        greedy_without_logprobs: true, ..cold() }),
        NextAction { dispatch: Dispatch::Ordinary, ordinary: OrdinaryLeaf::GpuGreedy });
    assert_eq!(classify(Facts { history_present: false, mtp_present: false, ..cold() }).ordinary,
        OrdinaryLeaf::FullLogits);
}

#[test]
fn cancellation_generation_and_physical_slot_refuse_known_leaf_preview() {
    let action = classify(cold());
    assert_eq!(preview(action, SlotId(1), handle(1, 7), handle(1, 7), false),
        OwnerPreview::KnownLeaf(OrdinaryLeaf::CoherentMtpFullLogits));
    assert_eq!(preview(action, SlotId(1), handle(1, 7), handle(1, 7), true), OwnerPreview::Cancelled);
    assert_eq!(preview(action, SlotId(1), handle(1, 8), handle(1, 7), false), OwnerPreview::RefusedOwner);
    assert_eq!(preview(action, SlotId(1), handle(1, 7), handle(2, 7), false), OwnerPreview::RefusedOwner);
    assert_eq!(preview(action, SlotId(3), handle(1, 7), handle(1, 7), false), OwnerPreview::RefusedOwner);
}

#[test]
fn nested_history_without_mtp_can_be_full_logits_not_false_missing() {
    let a = classify(Facts { mtp_present: false, ..cold() });
    assert_eq!(a.dispatch, Dispatch::History);
    assert_eq!(a.ordinary, OrdinaryLeaf::FullLogits);
}
