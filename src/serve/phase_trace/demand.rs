//! Prospective diagnostic demand. Executor availability remains separately false.
use super::SlotState;
use crate::serve::scheduler::SlotHandle;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cell::RefCell;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub population_policy: Policy,
    pub qwen_prefill_chunk_tokens: u32,
}
#[derive(Clone, Copy, Deserialize, Serialize)]
pub(super) enum Policy {
    #[serde(rename = "queued_forward_demand_v3")]
    QueuedForwardDemandV3,
}
impl Config {
    pub(super) fn validate(&self, diagnostic: Option<bool>) -> Result<(), String> {
        if diagnostic != Some(true) || self.qwen_prefill_chunk_tokens != 128 {
            return Err("queued demand requires explicit native diagnostic and fixed128 quantum".into());
        }
        Ok(())
    }
}

pub(super) fn deserialize_optional<'de, D: serde::Deserializer<'de>>(d: D)
    -> Result<Option<Config>, D::Error> {
    Config::deserialize(d).map(Some)
}

// Actual owning worker only; no HTTP thread or scheduler-phase inference.
thread_local! {
    static READY: RefCell<Vec<SlotState>> = const { RefCell::new(Vec::new()) };
}
fn same_owner(a: &SlotState, b: &SlotState) -> bool {
    a.slot_id == b.slot_id && a.generation == b.generation
        && a.request_id == b.request_id && a.phase == b.phase
}
fn input_ready(s: &SlotState) -> bool {
    s.forward_eligibility == Some(true) && s.runnable_forward == Some(true)
        && s.blocked.is_none()
}
fn decorate(states: &[SlotState], cause: &str, selected: &[SlotHandle],
    previous: &[SlotState]) -> Value {
    json!(states.iter().map(|s| {
        let mut value = json!(s);
        let selected_owner = selected.iter().any(|h| h.slot_id.0 == s.slot_id
            && h.generation == s.generation);
        let state = if cause == "owning_worker_work_state" {
            if input_ready(s) { "QueuedInputReady" }
            else if s.runnable_forward.is_some() { "NoForwardDemand" }
            else { "Unknown" }
        } else if cause == "owning_worker_enter_transaction" {
            match previous.iter().find(|old| same_owner(old, s)) {
                Some(old) if input_ready(old) && s.forward_eligibility == Some(true) => {
                    if selected_owner { "SelectedForwardPending" }
                    else { "WaitingForOwningExecutor" }
                }
                Some(_) if s.forward_eligibility == Some(false) => "NoForwardDemand",
                _ => "Unknown",
            }
        } else { "Unknown" };
        value["forward_demand_state"] = json!(state);
        value["selected_forward_owner"] = json!(selected_owner);
        value
    }).collect::<Vec<_>>())
}

pub(super) fn enrich(states: &[SlotState], cause: &str, selected: &[SlotHandle]) -> Value {
    READY.with(|ready| {
        let mut previous = ready.borrow_mut();
        let result = decorate(states, cause, selected, &previous);
        if cause == "owning_worker_work_state" { *previous = states.to_vec(); }
        else if cause != "owning_worker_enter_transaction" { previous.clear(); }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::scheduler::SlotId;
    fn slot(i: u32) -> SlotState {
        SlotState {request_id:i as u64,slot_id:i,generation:7,phase:"prefill",active:true,
            forward_eligibility:Some(true),runnable_forward:Some(true),runnable_emit:Some(false),
            inflight:None,blocked:None}
    }
    #[test]
    fn selected_owner_and_waiters_never_gain_executor_availability() {
        let before:Vec<_>=(0..4).map(slot).collect();
        let mut busy=before.clone();
        for s in &mut busy {s.runnable_forward=Some(false);s.blocked=Some("OwningWorkerTransaction");}
        let result=decorate(&busy,"owning_worker_enter_transaction",
            &[SlotHandle {slot_id:SlotId(0),generation:7}],&before);
        assert_eq!(result[0]["forward_demand_state"],"SelectedForwardPending");
        for i in 1..4 {
            assert_eq!(result[i]["forward_demand_state"],"WaitingForOwningExecutor");
            assert_eq!(result[i]["selected_forward_owner"],false);
            assert_eq!(result[i]["runnable_forward"],false);
        }
    }
    #[test]
    fn pending_output_terminal_stale_and_unknown_never_gain_demand() {
        let mut old=slot(0);old.runnable_forward=Some(false);
        old.blocked=Some("VerifiedOutputPendingSemanticUnknown");
        assert_eq!(decorate(&[old.clone()],"owning_worker_work_state",&[],&[])[0]["forward_demand_state"],"NoForwardDemand");
        let mut now=slot(0);now.generation+=1;
        assert_eq!(decorate(&[now],"owning_worker_enter_transaction",&[],&[slot(0)])[0]["forward_demand_state"],"Unknown");
        for reason in ["ClientClosed","RequestTerminal","Cancelled","WorkerError"] {
            old.blocked=Some(reason);old.forward_eligibility=Some(false);
            assert_eq!(decorate(&[old.clone()],"owning_worker_work_state",&[],&[])[0]["forward_demand_state"],"NoForwardDemand");
        }
    }
    #[test]
    fn absent_policy_is_not_null_and_unknown_fields_are_not_ignored() {
        #[derive(Deserialize)]
        struct Envelope {
            #[serde(default, deserialize_with="deserialize_optional")]
            policy: Option<Config>,
        }
        assert!(serde_json::from_str::<Envelope>("{}").unwrap().policy.is_none());
        assert!(serde_json::from_str::<Envelope>(r#"{"policy":null}"#).is_err());
    }
    #[test]
    fn closed_profile_and_fixed_quantum_require_actual_diagnostic_opt_in() {
        let c:Config=serde_json::from_str(r#"{"population_policy":"queued_forward_demand_v3","qwen_prefill_chunk_tokens":128}"#).unwrap();
        assert!(c.validate(Some(true)).is_ok());assert!(c.validate(None).is_err());
        assert!(c.validate(Some(false)).is_err());
        for value in [r#"{"population_policy":"forward_demand_v2","qwen_prefill_chunk_tokens":128}"#,
            r#"{"population_policy":"queued_forward_demand_v3","qwen_prefill_chunk_tokens":128,"extra":true}"#] {
            assert!(serde_json::from_str::<Config>(value).is_err());
        }
        let wrong=Config {population_policy:Policy::QueuedForwardDemandV3,qwen_prefill_chunk_tokens:64};
        assert!(wrong.validate(Some(true)).is_err());
    }
}
