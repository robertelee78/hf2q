//! Actual production cancellation recovery over real checked transactions.
//! No pending capture is promoted and no shallower fallback is invented here.
use super::*;
use super::{lifecycle::Lane, trace::State};
fn begin_without_capture(fixture: &mut Fixture, lane: &mut Lane, prompt: &[u32], cached: usize) -> Result<u64> {
    let request = lane.admit(prompt.len(), DECODE_LIMIT)?;
    let qwen = fixture.qwen();
    let pending = Qwen35PrefillState::begin(prompt.to_vec(), SamplingParams {
        max_tokens: DECODE_LIMIT, ..SamplingParams::default()
    }, None, &mut lane.cache, SLOT, cached, None, lane.spec.clone(), None,
        qwen.hidden_size, qwen.model.kv_graft.as_ref())?;
    // A real admitted prefill exists, but no advance/capture has executed.
    drop(pending);
    Ok(request)
}
pub(super) fn run(fixture: &mut Fixture) -> Result<()> {
    for window in ["beforecapture", "pending", "resumed-decode"] {
        let mut lane = Lane::new(fixture)?;
        let ids = super::lifecycle::build(fixture, &mut lane, 2)?;
        let script = super::lifecycle::script(&mut lane, &ids)?;
        fs::write(fixture.packet.output.join(format!("cancel-{window}-prefix-script.json")),
            serde_json::to_vec_pretty(&script)?)?;
        let reference = super::lifecycle::independent_checkpoint(fixture, &script, 1)?;
        let expected = State::read(&reference.cache, SLOT)?;
        let saved_tokens = reference.retained.clone();
        drop(reference);
        lane.restore(ids[1], &fixture.supervisor)?;
        let mut prompt = saved_tokens.clone();
        prompt.extend_from_slice(&fixture.packet.workload.lanes[0][128..144]);
        let request = if window == "beforecapture" {
            begin_without_capture(fixture, &mut lane, &prompt, saved_tokens.len())?
        } else {
            let served = super::lifecycle::prefill(fixture, &mut lane, &prompt, saved_tokens.len(), true)?;
            let request = served.request.context("cancelled generation lacks its actual admission")?;
            if window == "resumed-decode" {
                super::lifecycle::tick(fixture, &mut lane, served.state)?;
            }
            ensure!(lane.store.pending().is_some(), "actual pending capture missing");
            request
        };
        let recovery = recover_cancelled_prefix(&mut lane.cache, SLOT, request, &mut lane.store,
            &mut lane.retained, &mut lane.spec, window == "beforecapture", &fixture.supervisor)?;
        lane.cancel_admission()?;
        let actual = State::read(&lane.cache, SLOT)?;
        super::state_pair::write(fixture, &format!("cancel-{window}"), &actual, &expected,
            &lane.retained, &saved_tokens)?;
        expected.same(&actual, "cancel recovery versus never-cancelled recovered-depth twin")?;
        ensure!(lane.retained == saved_tokens && lane.store.pending().is_none(),
            "cancel published uncommitted state");
        let expected_disposition = if window == "beforecapture" {
            AnchorRecovery::Preserved { tokens: saved_tokens.len() }
        } else { AnchorRecovery::Restored { rank: 0, tokens: saved_tokens.len() } };
        ensure!(recovery == expected_disposition, "cancellation reported the wrong actual recovery branch");
        fs::write(fixture.packet.output.join(format!("cancel-{window}.json")),
            serde_json::to_vec_pretty(&serde_json::json!({"schema":"hf2q.c2.cancel-matrix.v1",
                "window":window,"actual_recovery":format!("{recovery:?}"),
                "recovered_tokens":lane.retained.len(),"state":"ExactEqual","pending":false}))?)?;
    }
    // Fresh admission cancelled before its first capture has no committed
    // recovery point: the REAL helper must reset, not publish a zero-byte stub.
    let mut lane = Lane::new(fixture)?;
    let baseline = State::read(&lane.cache, SLOT)?;
    let prompt = fixture.packet.workload.lanes[0][..64].to_vec();
    let request = begin_without_capture(fixture, &mut lane, &prompt, 0)?;
    let recovery = recover_cancelled_prefix(&mut lane.cache, SLOT, request, &mut lane.store,
        &mut lane.retained, &mut lane.spec, true, &fixture.supervisor)?;
    lane.cancel_admission()?;
    ensure!(recovery == AnchorRecovery::Reset && lane.retained.is_empty() && lane.spec.is_none()
        && lane.store.is_empty(), "fresh pre-capture cancellation did not reset0");
    let state = State::read(&lane.cache, SLOT)?;
    super::state_pair::write(fixture, "cancel-fresh-reset", &state, &baseline, &[], &[])?;
    baseline.same(&state, "fresh cancelled/reset state versus cold zero-cursor twin")?;
    fs::write(fixture.packet.output.join("cancel-fresh-reset.json"), serde_json::to_vec_pretty(
        &serde_json::json!({"schema":"hf2q.c2.cancel-matrix.v1","window":"fresh-beforecapture",
            "actual_recovery":"Reset","recovered_tokens":0,"state":"ExactEqual","pending":false}))?)?;
    Ok(())
}
