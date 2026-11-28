//! Actual cancellation and never-cancelled recovery references have their own
//! required request/state/terminal custody, including genuine no-advance arms.
use super::*;
use super::{lifecycle::Lane, registration::StepWork, request_kind::{Outcome, Purpose, Window}, trace};

fn begin_without_capture(fixture: &mut Fixture, lane: &mut Lane, prompt: &[u32], cached: usize) -> Result<u64> {
    let step = trace::acquire::begin_step(StepWork::StateOnly)?;
    let request = lane.admit(prompt.len(), DECODE_LIMIT)?;
    let qwen = fixture.qwen();
    let pending = Qwen35PrefillState::begin(prompt.to_vec(), SamplingParams {
        max_tokens: DECODE_LIMIT, ..SamplingParams::default()
    }, None, &mut lane.cache, SLOT, cached, None, lane.spec.clone(), None,
        qwen.hidden_size, qwen.model.kv_graft.as_ref())?;
    // Real admission and prepared state, but deliberately no advance/forward.
    drop(pending);
    trace::transaction(&lane.cache, SLOT)?;
    step.finish()?;
    Ok(request)
}

fn reference(fixture: &Fixture, lane: &Lane, prompt: &[u32], name: &str, window: Window)
    -> Result<super::live_ram::OwnedState> {
    let scope = super::lifecycle::begin_acquisition(fixture, lane, prompt, name, "twin")?;
    let step = trace::acquire::begin_step(StepWork::StateOnly)?;
    trace::transaction(&lane.cache, SLOT)?;
    step.finish()?;
    trace::acquire::settle_outcome(&lane.cache, &lane.retained,
        Outcome::Reference { window, recovered_tokens: lane.retained.len() })?;
    scope.finish_transactions()?;
    let observed = scope.take()?;
    ensure!(observed.forwards.is_empty() && observed.decisions.is_empty(),
        "recovered-depth reference invented target work");
    drop(scope);
    super::state_pair::snapshot(fixture, name, "twin", &lane.cache, &lane.retained)
}

fn close_actual(fixture: &Fixture, lane: &Lane, scope: trace::Scope, name: &str,
    window: Window, request: u64, recovery: &AnchorRecovery) -> Result<()> {
    trace::acquire::settle_outcome(&lane.cache, &lane.retained,
        Outcome::cancellation(window, request, recovery, lane.retained.len())?)?;
    scope.finish_transactions()?;
    let observed = scope.take()?;
    if Purpose::for_case(name, "actual")?.raw_required() {
        super::trace_pair::write_arm(fixture, name, "actual", &observed)?;
    } else {
        ensure!(observed.forwards.is_empty() && observed.decisions.is_empty(),
            "no-advance cancellation invented raw target rows");
    }
    Ok(())
}

pub(super) fn run(fixture: &mut Fixture) -> Result<()> {
    for (label, window) in [("beforecapture", Window::BeforeCapture), ("pending", Window::Pending),
        ("resumed-decode", Window::ResumedDecode)] {
        let name = format!("cancel-{label}");
        if !fixture.selected(&name)? { continue; }
        let mut lane = Lane::new(fixture)?;
        let ids = super::lifecycle::build(fixture, &mut lane, 2)?;
        let script = super::lifecycle::script(&mut lane, &ids)?;
        super::case_evidence::authority(fixture, &name, &script)?;
        let reference_lane = super::lifecycle::independent_checkpoint(fixture, &script, 1)?;
        let saved_tokens = reference_lane.retained.clone();
        let mut prompt = saved_tokens.clone();
        prompt.extend_from_slice(&fixture.packet.workload.lanes[0][128..144]);
        let expected = reference(fixture, &reference_lane, &prompt, &name, window)?;
        drop(reference_lane);
        // The request obligation precedes restore, admission, target work,
        // cancellation and recovery. Every state transition is required.
        let scope = super::lifecycle::begin_acquisition(fixture, &lane, &prompt, &name, "actual")?;
        let restore = trace::acquire::begin_step(StepWork::StateOnly)?;
        lane.restore(ids[1], &fixture.supervisor)?;
        scope.publish(&lane.retained);
        trace::transaction(&lane.cache, SLOT)?;
        restore.finish()?;
        let request = if window == Window::BeforeCapture {
            begin_without_capture(fixture, &mut lane, &prompt, saved_tokens.len())?
        } else {
            let served = super::lifecycle::prefill(fixture, &mut lane, &prompt, saved_tokens.len(), true)?;
            let request = served.request.context("cancelled generation lacks its actual admission")?;
            if window == Window::ResumedDecode { super::lifecycle::tick(fixture, &mut lane, served.state)?; }
            ensure!(lane.store.pending().is_some(), "actual pending capture missing");
            request
        };
        let recovering = trace::acquire::begin_step(StepWork::StateOnly)?;
        let recovery = recover_cancelled_prefix(&mut lane.cache, SLOT, request, &mut lane.store,
            &mut lane.retained, &mut lane.spec, window == Window::BeforeCapture, &fixture.supervisor)?;
        lane.cancel_admission()?;
        scope.publish(&lane.retained);
        trace::transaction(&lane.cache, SLOT)?;
        recovering.finish()?;
        close_actual(fixture, &lane, scope, &name, window, request, &recovery)?;
        let actual = super::state_pair::snapshot(fixture, &name, "actual", &lane.cache, &lane.retained)?;
        expected.same_semantic(&actual, "cancel recovery versus never-cancelled recovered-depth twin")?;
        ensure!(lane.retained == saved_tokens && lane.store.pending().is_none(), "cancel published uncommitted state");
        let expected_disposition = if window == Window::BeforeCapture {
            AnchorRecovery::Preserved { tokens: saved_tokens.len() }
        } else { AnchorRecovery::Restored { rank: 0, tokens: saved_tokens.len() } };
        ensure!(recovery == expected_disposition, "cancellation reported the wrong actual recovery branch");
        super::case_evidence::classification(fixture, &name, &serde_json::json!({
            "schema":"hf2q.c2.cancel-matrix.v1","window":label,"actual_recovery":format!("{recovery:?}"),
            "recovered_tokens":lane.retained.len(),"state":"ExactEqual","pending":false,
        }))?;
        drop((lane, expected, actual));
        fixture.completed_case(&name)?;
    }
    let name = "cancel-fresh-reset";
    if !fixture.selected(name)? { return Ok(()); }
    let mut lane = Lane::new(fixture)?;
    let prompt = fixture.packet.workload.lanes[0][..64].to_vec();
    super::case_evidence::authority(fixture, name, &serde_json::json!({
        "schema":"hf2q.c2.no-checkpoint-cancellation-input.v1","prompt":prompt,"committed_checkpoint":false,
    }))?;
    let baseline = reference(fixture, &lane, &prompt, name, Window::FreshReset)?;
    let scope = super::lifecycle::begin_acquisition(fixture, &lane, &prompt, name, "actual")?;
    let request = begin_without_capture(fixture, &mut lane, &prompt, 0)?;
    let recovering = trace::acquire::begin_step(StepWork::StateOnly)?;
    let recovery = recover_cancelled_prefix(&mut lane.cache, SLOT, request, &mut lane.store,
        &mut lane.retained, &mut lane.spec, true, &fixture.supervisor)?;
    lane.cancel_admission()?;
    scope.publish(&lane.retained);
    trace::transaction(&lane.cache, SLOT)?;
    recovering.finish()?;
    ensure!(recovery == AnchorRecovery::Reset && lane.retained.is_empty() && lane.spec.is_none()
        && lane.store.is_empty(), "fresh pre-capture cancellation did not reset0");
    close_actual(fixture, &lane, scope, name, Window::FreshReset, request, &recovery)?;
    let state = super::state_pair::snapshot(fixture, name, "actual", &lane.cache, &[])?;
    baseline.same(&state, "fresh cancelled/reset state versus cold zero-cursor twin")?;
    super::case_evidence::classification(fixture, name, &serde_json::json!({
        "schema":"hf2q.c2.cancel-matrix.v1","window":"fresh-beforecapture",
        "actual_recovery":"Reset","recovered_tokens":0,"state":"ExactEqual","pending":false,
    }))?;
    drop((lane, baseline, state));
    fixture.completed_case(name)?;
    Ok(())
}
