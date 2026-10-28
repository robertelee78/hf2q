//! Source-local worker readiness facts. No scheduling or model mutations.
use super::*;
use crate::serve::phase_trace::WorkState;

fn fact(
    handle: SlotHandle,
    reply: &SlotReply,
    forward: Option<bool>,
    emit: Option<bool>,
    blocked: Option<&'static str>,
) -> WorkState {
    if reply.client_closed() {
        WorkState {
            handle,
            forward_eligibility: Some(false),
            runnable_forward: Some(false),
            runnable_emit: Some(false),
            blocked: Some("ClientClosed"),
        }
    } else {
        WorkState {
            handle,
            forward_eligibility: Some(!matches!(
                blocked,
                Some(
                    "RequestTerminal" | "Cancelled" | "WorkerError" | "ColdCohortCompletionBarrier"
                )
            )),
            runnable_forward: forward,
            runnable_emit: emit,
            blocked,
        }
    }
}

pub(super) fn qwen(scheduler: &InflightBatchedScheduler, slots: &[Option<Qwen35Slot>]) {
    if !crate::serve::phase_trace::enabled() {
        return;
    }
    let facts = slots
        .iter()
        .flatten()
        .map(|(work, reply, handle)| match work {
            Qwen35SlotWork::Prefill(state) => fact(
                *handle,
                reply,
                Some(state.f2_forward_ready()),
                Some(false),
                None,
            ),
            Qwen35SlotWork::Decode(state) => {
                let (forward, pending_tokens) = state.f2_work_readiness();
                fact(
                    *handle,
                    reply,
                    Some(forward),
                    if pending_tokens { None } else { Some(false) },
                    if pending_tokens {
                        Some("VerifiedOutputPendingSemanticUnknown")
                    } else if !forward {
                        Some("RequestTerminal")
                    } else {
                        None
                    },
                )
            }
        })
        .collect();
    scheduler.trace_work_states(facts);
}

pub(super) fn gemma(scheduler: &InflightBatchedScheduler, slots: &[Option<Gemma4Slot>]) {
    if !crate::serve::phase_trace::enabled() {
        return;
    }
    let facts = slots
        .iter()
        .flatten()
        .map(|(work, reply, handle)| match work {
            Gemma4SlotWork::Prefill(state) => fact(
                *handle,
                reply,
                Some(state.progress.cursor() < state.prompt_tokens.len()),
                Some(false),
                None,
            ),
            Gemma4SlotWork::Decode(state) => fact(
                *handle,
                reply,
                Some(!state.finished_at_seed()),
                Some(false),
                if state.finished_at_seed() {
                    Some("RequestTerminal")
                } else {
                    None
                },
            ),
        })
        .collect();
    scheduler.trace_work_states(facts);
}

pub(super) fn deepseek(scheduler: &InflightBatchedScheduler, slots: &[Option<Deepseek4Slot>]) {
    if !crate::serve::phase_trace::enabled() {
        return;
    }
    let facts = slots
        .iter()
        .flatten()
        .map(|(work, reply, handle)| match work {
            // A sampled token may terminate before cache advancement. Do not
            // predict prepare_tick's decision or count logits as buffered output.
            Deepseek4SlotWork::Decode { .. } => fact(
                *handle,
                reply,
                None,
                None,
                Some("SampleBeforeForwardDecision"),
            ),
            Deepseek4SlotWork::ParkedCompletion { .. } => fact(
                *handle,
                reply,
                Some(false),
                Some(false),
                Some("ColdCohortCompletionBarrier"),
            ),
            Deepseek4SlotWork::Prefill(state) => fact(
                *handle,
                reply,
                Some(state.f2_forward_ready()),
                Some(false),
                None,
            ),
        })
        .collect();
    scheduler.trace_work_states(facts);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qwen_verified_pending_tokens_block_forward_without_semantic_emit_claim() {
        use super::super::super::engine_qwen35::f2_qwen_decode_buffer_state as readiness;
        assert_eq!(readiness(1, 8, 0), (true, false));
        assert_eq!(readiness(1, 8, 3), (false, true));
        assert_eq!(readiness(8, 8, 3), (false, false));
        assert_eq!(readiness(9, 8, 0), (false, false));
    }

    #[test]
    fn closed_reply_cannot_be_forward_or_emit_ready() {
        let (tx, rx) = oneshot::channel();
        drop(rx);
        let value = fact(
            SlotHandle {
                slot_id: SlotId(0),
                generation: 9,
            },
            &SlotReply::Unary(tx),
            Some(true),
            Some(true),
            None,
        );
        assert_eq!(value.runnable_forward, Some(false));
        assert_eq!(value.runnable_emit, Some(false));
        assert_eq!(value.forward_eligibility, Some(false));
        assert_eq!(value.blocked, Some("ClientClosed"));
    }
}
