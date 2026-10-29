//! Synthetic HOST state only; exercises real lookup/cost/continuation methods.
//! No model identity, acquisition, device or GPU behavior is represented.
use super::*;
pub(in crate::serve::api) fn host_state(slot:SlotId)->Qwen35DecodeState {
    let prompt=vec![1,2,3,4,5,6,7,8,9,1,2,3,4,5,6];
    let mut lookup=HistoryLookupIndex::new(HistoryLookupConfig {min_match:6,max_match:12,max_draft_tokens:3,max_model_len:262144});
    lookup.reset(&prompt);lookup.extend_verified(&[7]);
    Qwen35DecodeState {slot_id:slot,prompt_len:prompt.len(),decode_position_base:prompt.len(),prompt_tokens:prompt,
        max_tokens:64,is_greedy:false,want_logprobs:false,logprobs_vec:None,cached_tokens:0,
        params:SamplingParams {repetition_penalty:1.05,..SamplingParams::default()},
        grammar_runtime:None,tool_splitter:None,next_token:7,generated_tokens:vec![7],sampling_history:vec![7],
        decoded_text:String::new(),stop_strings:Vec::new(),finish_reason:"length",step:1,answer_event_reported:false,
        thinking_budget:None,prefill_duration:Duration::ZERO,decode_start:Instant::now(),mtp:None,
        history_lookup:Some(lookup),pending_speculation_output:VecDeque::new(),terminal_after_pending:false,
        mtp_cost:SpeculationCostController::new(),history_cost:SpeculationCostController::new()}
}
#[test]fn cold_cost_continuation_owns_then_restores_exact_lookup_on_error() {
    let mut s=host_state(SlotId(0));let before=s.history_lookup.as_ref().unwrap().verified_len();
    let HistoryDecision::Ordinary(c)=s.prepare_history_decision().unwrap() else {panic!("cold baseline required")};
    assert!(s.history_lookup.is_none());
    // No target committed: same path restores lookup even if scalar fails.
    c.restore(&mut s);assert_eq!(s.history_lookup.as_ref().unwrap().verified_len(),before);
    assert!(!s.history_cost.may_speculate());
}
#[test]fn actual_history_proposal_is_retained_not_replaced_by_ordinary() {
    let mut s=host_state(SlotId(0));s.history_cost.observe_ordinary_target(Duration::from_millis(10));
    let HistoryDecision::Proposals(d)=s.prepare_history_decision().unwrap() else {panic!("real repeated history")};
    assert_eq!(d,vec![8,9,1]);assert!(s.history_lookup.is_some());
}
#[test]fn actual_empty_draft_continuation_extends_only_newly_committed_suffix() {
    let mut s=host_state(SlotId(1));s.prompt_tokens=(0..20).collect();s.prompt_len=20;s.generated_tokens=vec![250];
    s.history_lookup.as_mut().unwrap().reset(&s.prompt_tokens);s.history_lookup.as_mut().unwrap().extend_verified(&[250]);
    s.history_cost.observe_ordinary_target(Duration::from_millis(10));
    let HistoryDecision::Ordinary(c)=s.prepare_history_decision().unwrap() else {panic!("no actual repeated suffix")};
    assert!(s.history_lookup.is_none());s.generated_tokens.push(249);c.restore(&mut s);
    assert_eq!(s.history_lookup.as_ref().unwrap().verified_len(),22);
}
