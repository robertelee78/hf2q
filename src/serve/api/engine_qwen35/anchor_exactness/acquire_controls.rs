//! No model/device: real guards, pre-call plans, strict stores and reloaded proof.
//! Synthetic acquisition is deliberately unqualified, never a runtime substitute.
use super::*;
use super::super::super::{transaction_trace, state_layout::StateLayout, registered_plan};
use super::super::super::request_kind::{Outcome, Recovery, Window};

fn cpu_state(settlement: bool) -> Result<()> {
    cpu_state_outcome(settlement.then_some(Outcome::Completed { emitted_tokens: 1 }))
}
fn cpu_state_outcome(outcome: Option<Outcome>) -> Result<()> {
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let active = active.as_mut().context("CPU acquisition scope absent")?;
        let session = active.session.as_mut().context("CPU acquisition session absent")?;
        let ticket = if let Some(outcome) = outcome { session.settlement_before(outcome)? }
            else { session.state_before_capture()? };
        let Transactions::Recording(writer) = &mut active.transactions else { anyhow::bail!("CPU writer absent"); };
        if outcome.is_some() { writer.begin(ticket, EventKind::Settlement)?; }
        ensure!(writer.current()?.ticket == ticket, "CPU state belongs to another parent ticket");
        let capture = writer.prepare(StateLayout { cursors: 1, parity: 1, regions: vec![8, 0], ledger: 1 })?;
        let state = State { target: vec![1], mtp: None, parity: vec![false], regions: vec![vec![0; 8], Vec::new()] };
        writer.record(capture, &state, &[1])?;
        if outcome.is_some() { session.settlement_after(ticket) } else { session.observed_state(ticket) }
    })
}

fn native_step() -> Result<()> {
    let step = begin_step(StepWork::Native)?;
    let native = native_begin()?;
    native_contract([NativeHead { slot: SLOT, start: 0, input_rows: 1, projected_rows: 1,
        vocab: 2, hidden_width: 1, gpu_argmax: true,
        input: super::super::super::native_inputs::InputBinding::host(&[1])? }])?;
    plan_native()?;
    super::super::gpu_argmax(SLOT, 0, 1)?;
    super::super::forward(SLOT, 0, &[1], "synthetic CPU projection", &[0.0, 1.0], 2, Some(&[0.5]), 1)?;
    native.finish()?;
    require_decision(SLOT, 0, 2, 1)?;
    super::super::decision_for(SLOT, None, 0, 1)?;
    cpu_state(false)?;
    step.finish()
}

#[test]
fn c2_transaction_scope_rejects_unfinished_and_incomplete_capture() {
    let (_root, writer) = transaction_trace::cpu_fixture::writer("actual");
    let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
    assert!(scope.take().is_err(), "open execution cannot publish observations");
    assert!(scope.finish_transactions().is_err(), "missing required state/settlement must fail");
    assert!(scope.take().is_err(), "ignored coverage failure must remain fatal");
}

#[test]
fn c2_transaction_scope_rejects_duplicate_completion() {
    let (_root, writer) = transaction_trace::cpu_fixture::writer("actual");
    let context = writer.context();
    let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
    native_step().unwrap();
    cpu_state(true).unwrap();
    scope.finish_transactions().unwrap();
    let observation = scope.take().unwrap();
    assert_eq!(observation.decisions, [(0, 1)]);
    let budget = context.budget.borrow();
    let binding = budget.artifact(&format!("{}-actual-execution", context.case)).unwrap();
    let expected = registered_plan::ExpectedScope { root: &context.root, identity: &context.identity,
        acquisition: &context.acquisition, registration_sha256: &context.registration_sha256,
        case: &context.case, arm: &context.arm };
    let plan = registered_plan::load_checked(&expected, binding).unwrap();
    plan.forwards[&0].check(&observation.forwards[0]).unwrap();
    assert_eq!(plan.decisions[&0].physical_position, 0);
    assert_eq!(plan.gpu_positions, [0]);
    drop(budget);
    assert!(scope.finish_transactions().is_err(), "completion is one-shot");
    assert!(scope.take().is_err(), "ignored duplicate completion must remain fatal");
}

#[test]
fn c2_required_api_rejects_symmetric_whole_lower_hook_omission() {
    for arm in ["actual", "twin"] {
        let (_root, writer) = transaction_trace::cpu_fixture::writer(arm);
        let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
        let step = begin_step(StepWork::Native).unwrap();
        let native = native_begin().unwrap();
        native_contract([NativeHead { slot: SLOT, start: 0, input_rows: 1, projected_rows: 1,
            vocab: 2, hidden_width: 1, gpu_argmax: false,
            input: super::super::super::native_inputs::InputBinding::host(&[1]).unwrap() }]).unwrap();
        // Neither lower planner nor observer is invoked in either arm.
        assert!(native.finish().is_err());
        assert!(step.finish().is_err());
        assert!(scope.finish_transactions().is_err());
        assert!(scope.take().is_err());
    }
}

#[test]
fn c2_required_step_rejects_whole_native_api_omission() {
    let (_root, writer) = transaction_trace::cpu_fixture::writer("actual");
    let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
    let step = begin_step(StepWork::Native).unwrap();
    cpu_state(false).unwrap();
    assert!(step.finish().is_err(), "state alone cannot stand in for selected native work");
    assert!(scope.finish_transactions().is_err());
}

#[test]
fn c2_request_rejects_missing_terminal_settlement() {
    let (_root, writer) = transaction_trace::cpu_fixture::writer("actual");
    let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
    native_step().unwrap();
    assert!(scope.finish_transactions().is_err(), "completed lower observations do not settle the request");
    assert!(scope.take().is_err());
}

#[test]
fn c2_required_capture_rejects_lost_writer() {
    let (_root, writer) = transaction_trace::cpu_fixture::writer("actual");
    let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
    ACTIVE.with(|cell| cell.borrow_mut().as_mut().unwrap().transactions = Transactions::Disabled);
    assert!(begin_step(StepWork::StateOnly).is_err());
    assert!(scope.finish_transactions().is_err());
    assert!(scope.take().is_err());
}

#[test]
fn c2_registered_execution_rejects_other_nonce_and_other_root() {
    let (_root, writer) = transaction_trace::cpu_fixture::writer("actual");
    let context = writer.context();
    let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
    native_step().unwrap(); cpu_state(true).unwrap(); scope.finish_transactions().unwrap();
    let budget = context.budget.borrow();
    let binding = budget.artifact(&format!("{}-actual-execution", context.case)).unwrap();
    let mut other = context.acquisition.clone();
    other.nonce = if other.nonce == "9".repeat(64) { "8".repeat(64) } else { "9".repeat(64) };
    let expected = registered_plan::ExpectedScope { root: &context.root, identity: &context.identity,
        acquisition: &other, registration_sha256: &context.registration_sha256,
        case: &context.case, arm: &context.arm };
    assert!(registered_plan::load_checked(&expected, binding).is_err());
    let other_root = tempfile::tempdir().unwrap();
    let other_root = other_root.path().canonicalize().unwrap();
    let copied = other_root.join(&binding.path);
    fs::create_dir_all(copied.parent().unwrap()).unwrap();
    fs::copy(context.root.join(&binding.path), &copied).unwrap();
    other = context.acquisition.clone(); other.output_root = other_root.clone();
    let expected = registered_plan::ExpectedScope { root: &other_root, identity: &context.identity,
        acquisition: &other, registration_sha256: &context.registration_sha256,
        case: &context.case, arm: &context.arm };
    assert!(registered_plan::load_checked(&expected, binding).is_err());
}

#[test]
fn c2_whole_case_rejects_missing_arm_bundle() {
    let (_root, writer) = transaction_trace::cpu_fixture::writer("actual");
    let context = writer.context();
    let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
    native_step().unwrap(); cpu_state(true).unwrap(); scope.finish_transactions().unwrap();
    assert!(context.budget.borrow_mut().completed_case(&context.case).is_err(),
        "completed execution cannot substitute for missing raw/final-state/other-arm artifacts");
    assert!(context.budget.borrow_mut().finish().is_err());
}

#[test]
fn c2_no_advance_cancellation_and_reference_publish_no_fake_target_rows() {
    for arm in ["actual", "twin"] {
        let (_root, writer) = transaction_trace::cpu_fixture::writer_for_case(arm, "cancel-beforecapture");
        let context = writer.context();
        let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
        let step = begin_step(StepWork::StateOnly).unwrap();
        cpu_state(false).unwrap(); step.finish().unwrap();
        let outcome = if arm == "actual" {
            Outcome::Cancelled { window: Window::BeforeCapture, request: 17,
                recovery: Recovery::Preserved, recovered_tokens: 1 }
        } else { Outcome::Reference { window: Window::BeforeCapture, recovered_tokens: 1 } };
        cpu_state_outcome(Some(outcome)).unwrap();
        scope.finish_transactions().unwrap();
        let observed = scope.take().unwrap();
        assert!(observed.forwards.is_empty() && observed.decisions.is_empty());
        let budget = context.budget.borrow();
        let binding = budget.artifact(&format!("{}-{arm}-execution", context.case)).unwrap();
        let expected = registered_plan::ExpectedScope { root: &context.root, identity: &context.identity,
            acquisition: &context.acquisition, registration_sha256: &context.registration_sha256,
            case: &context.case, arm: &context.arm };
        let plan = registered_plan::load_checked(&expected, binding).unwrap();
        assert!(plan.raw_expected.is_empty(), "no-advance must not fabricate Q1 target tensors");
    }
}

#[test]
fn c2_cancellation_cannot_publish_a_success_terminal_or_omit_its_whole_state_phase() {
    for omit_state in [false, true] {
        let (_root, writer) = transaction_trace::cpu_fixture::writer_for_case("actual", "cancel-beforecapture");
        let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
        if omit_state {
            cpu_state_outcome(Some(Outcome::Cancelled { window: Window::BeforeCapture,
                request: 17, recovery: Recovery::Preserved, recovered_tokens: 1 })).unwrap();
        } else {
            let step = begin_step(StepWork::StateOnly).unwrap();
            cpu_state(false).unwrap(); step.finish().unwrap();
            assert!(cpu_state_outcome(Some(Outcome::Completed { emitted_tokens: 0 })).is_err());
        }
        assert!(scope.finish_transactions().is_err());
        assert!(scope.take().is_err());
    }
}

#[test]
fn c2_native_host_input_rejects_same_length_token_substitution_before_publication() {
    let (_root, writer) = transaction_trace::cpu_fixture::writer("actual");
    let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
    let step = begin_step(StepWork::Native).unwrap();
    let native = native_begin().unwrap();
    native_contract([NativeHead { slot: SLOT, start: 0, input_rows: 1, projected_rows: 1,
        vocab: 2, hidden_width: 1, gpu_argmax: false,
        input: super::super::super::native_inputs::InputBinding::host(&[1]).unwrap() }]).unwrap();
    plan_native().unwrap();
    assert!(super::super::forward(SLOT, 0, &[0], "synthetic substituted native input",
        &[0.0, 1.0], 2, Some(&[0.5]), 1).is_err());
    assert!(native.finish().is_err());
    assert!(step.finish().is_err());
    assert!(scope.finish_transactions().is_err());
    assert!(scope.take().is_err());
}

#[test]
fn c2_native_input_records_reject_postcompletion_mutation_and_whole_record_omission() {
    let (_root, writer) = transaction_trace::cpu_fixture::writer("actual");
    let context = writer.context();
    let scope = Scope::begin_transactions(writer, &[1], 0).unwrap();
    native_step().unwrap(); cpu_state(true).unwrap(); scope.finish_transactions().unwrap();
    let mut observed = scope.take().unwrap();
    let binding = context.budget.borrow().artifact(&format!("{}-actual-execution", context.case)).unwrap().clone();
    let expected = registered_plan::ExpectedScope { root: &context.root, identity: &context.identity,
        acquisition: &context.acquisition, registration_sha256: &context.registration_sha256,
        case: &context.case, arm: &context.arm };
    let plan = registered_plan::load_checked(&expected, &binding).unwrap();
    let root = context.root.join("actual-input-proof");
    fs::create_dir(&root).unwrap();
    super::super::super::native_inputs::write(context.header(), &context.root, &context.budget,
        &observed, &plan, &root).unwrap();
    observed.forwards[0].input[0] = 0;
    assert!(super::super::super::native_inputs::write(context.header(), &context.root, &context.budget,
        &observed, &plan, &context.root.join("mutated")).is_err());
    observed.forwards.clear();
    assert!(super::super::super::native_inputs::write(context.header(), &context.root, &context.budget,
        &observed, &plan, &context.root.join("omitted")).is_err());
}
