//! Final state is planned and its strict store is opened BEFORE tensor readback.
//! Every arm owns its full state store; canonical single-lane parity reuses the
//! actual/single state pair instead of duplicating the actual state's payload.
use super::*;
use super::{state_layout::StateLayout, live_ram::OwnedState};

#[derive(Serialize)]
struct Plan<'a> {
    schema: &'static str,
    identity: &'a Identity,
    scope_acquisition: &'a acquisition::Acquisition,
    registration_sha256: &'a str,
    case: &'a str,
    arm: &'a str,
    slot: u32,
    store_root: &'a Path,
    layout: &'a StateLayout,
    expected: &'a [ExpectedTensor],
    derived_before_state_read: bool,
}

#[derive(Serialize)]
struct Binding<'a> {
    schema: &'static str,
    identity: &'a Identity,
    acquisition: &'a acquisition::Acquisition,
    registration_sha256: &'a str,
    case: &'a str,
    arm: &'a str,
    slot: u32,
    plan: acquisition::FileBinding,
    trace: acquisition::FileBinding,
}

pub(super) fn snapshot(fixture: &Fixture, name: &str, arm: &str,
    cache: &HybridKvCache, ledger: &[u32]) -> Result<OwnedState> {
    ensure!(fixture.selected(name)? && matches!(arm, "actual" | "twin" | "single"),
        "final state case/arm was not independently registered");
    let budget = fixture.matrix_budget()?;
    let registration = fixture.packet.registration.as_ref().context("matrix registration missing")?;
    let layout = StateLayout::from_cache(cache, SLOT, ledger.len())?;
    let agent = format!("{name}/slot0");
    let expected = layout.expected(&agent, 0)?;
    let count = expected.len();
    {
        let mut budget = budget.borrow_mut();
        budget.state_readback(layout.logical_bytes()?)?;
        budget.reserve_store(count, layout.logical_bytes()?)?;
    }
    let key = format!("{name}-{arm}-state");
    let root = fixture.packet.output.join(&key);
    let observed = fixture.observed(&fixture.packet.output)?;
    let plan_path = fixture.packet.output.join(format!("{key}-plan.json"));
    let plan_sha256 = super::metadata::write(&plan_path, budget.as_ref(), &Plan {
        schema: "hf2q.c2.final-state-plan.v1", identity: &fixture.packet.identity,
        scope_acquisition: &observed, registration_sha256: &registration.sha256, case: name, arm,
        slot: 0, store_root: &root, layout: &layout, expected: &expected,
        derived_before_state_read: true,
    })?;
    let mut store = Store::new(&root, fixture.packet.identity.clone(),
        format!("C2-ClassA-complete-semantic-state-{arm}"), expected,
        fixture.packet.host_payload_limit, count).map_err(anyhow::Error::msg)?;
    store.bind_acquisition(observed.for_store_root(&root)?).map_err(anyhow::Error::msg)?;
    let state = OwnedState::read(&budget, &layout, cache, SLOT)?;
    super::state_layout::capture(&mut store, &layout, &state, ledger, &agent, 0)?;
    let trace = store.finish().map_err(anyhow::Error::msg)?;
    ensure!(trace.coverage.errors.is_empty() && trace.coverage.missing_native_capabilities.is_empty()
        && trace.coverage.expected.len() == trace.coverage.captured.len(),
        "Incomplete semantic cache/ledger evidence");
    let trace_path = root.join("trace.json");
    let trace_sha256 = super::metadata::hash_file(&trace_path, budget.as_ref())?;
    let binding_path = root.join("binding.json");
    let binding_sha256 = super::metadata::write(&binding_path, budget.as_ref(), &Binding {
        schema: "hf2q.c2.final-state-binding.v1", identity: &trace.identity,
        acquisition: trace.acquisition.as_ref().context("state store lost native acquisition binding")?,
        registration_sha256: &registration.sha256, case: name, arm, slot: 0,
        plan: acquisition::FileBinding { path: plan_path.strip_prefix(&fixture.packet.output)?.into(),
            sha256: plan_sha256 },
        trace: acquisition::FileBinding { path: trace_path.strip_prefix(&fixture.packet.output)?.into(),
            sha256: trace_sha256 },
    })?;
    budget.borrow_mut().record_artifact(name, &key, &fixture.packet.output, &binding_path, binding_sha256)?;
    Ok(state)
}
