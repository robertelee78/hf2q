//! Metadata-only tests of actual private command routing. No model/device.
use super::*;

fn arm() -> Request {
    let (reply, _receive) = oneshot::channel();
    Request::ArmNativeObservation {
        nonce: "n".repeat(32),
        reply,
    }
}

#[test]
fn native_arm_is_not_a_generation_or_a_slot_reservation() {
    let request = arm();
    assert!(qwen35_request_prompt_tokens(&request).is_none());
    assert_eq!(
        qwen35_pending_disposition(&request, false, false, None),
        Qwen35PendingDisposition::ReadyWithoutSlot
    );
    assert_eq!(
        gemma4_pending_disposition(&request, &[], &[], &[]),
        Gemma4PendingDisposition::ReadyControl
    );
    assert!(!gemma4_request_runs_inline_gpu(&request));
}

#[test]
fn fatal_buffer_drain_replies_to_actual_arm_instead_of_silently_dropping_it() {
    let (reply, mut receive) = oneshot::channel();
    fail_slotaware_buffered_request(
        Request::ArmNativeObservation {
            nonce: "n".repeat(32),
            reply,
        },
        "source control: existing worker failed",
    );
    assert!(receive
        .try_recv()
        .unwrap()
        .unwrap_err()
        .to_string()
        .contains("existing worker failed"));
}
