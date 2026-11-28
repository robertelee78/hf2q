//! Strict raw twins consume pre-call source registrations, never observed counts.
use super::*;
use super::trace::{Observations, coordinate};
use std::collections::BTreeSet;

pub(super) fn write_pair(fixture: &Fixture, name: &str, actual: &Observations,
    expected: &Observations) -> Result<()> {
    write_arm(fixture, name, "actual", actual)?;
    write_arm(fixture, name, "twin", expected)
}

#[derive(Serialize)]
struct Binding<'a> {
    schema: &'static str,
    identity: &'a Identity,
    scope_acquisition: acquisition::Acquisition,
    case: &'a str,
    arm: &'a str,
    execution: &'a acquisition::FileBinding,
    trace: acquisition::FileBinding,
    route_observation: acquisition::FileBinding,
    native_inputs: acquisition::FileBinding,
}

pub(super) fn write_arm(fixture: &Fixture, name: &str, arm: &str,
    observations: &Observations) -> Result<()> {
    ensure!(matches!(arm, "actual" | "twin" | "single"), "unknown matrix oracle arm");
    let agent = format!("{name}/slot0");
    let mut plan = super::registered_plan::load(fixture, name, arm)?;
    ensure!(observations.forwards.len() == plan.forwards.len()
        && observations.decisions.len() == plan.decisions.len(),
        "Incomplete: actual rows/selectors differ from the required source set");
    let mut gpu_positions: Vec<_> = observations.gpu_argmax.iter().map(|&(position, _)| position).collect();
    gpu_positions.sort_unstable();
    plan.gpu_positions.sort_unstable();
    ensure!(gpu_positions == plan.gpu_positions, "Incomplete: actual GPU selectors differ from their required source set");
    let root = fixture.packet.output.join(format!("{name}-{arm}"));
    let records = plan.raw_expected.len();
    let mut store = Store::new(&root, fixture.packet.identity.clone(),
        format!("C2-ClassA-{name}-{arm}"), std::mem::take(&mut plan.raw_expected),
        fixture.packet.host_payload_limit, records).map_err(anyhow::Error::msg)?;
    store.bind_acquisition(fixture.observed(&root)?).map_err(anyhow::Error::msg)?;
    let budget = fixture.matrix_budget()?;
    let route_path = root.join("route-observation.json");
    let route_sha256 = super::metadata::write(&route_path, budget.as_ref(), &serde_json::json!({
        "schema":"hf2q.c2.route-observation.v1",
        "actual_gpu_argmax":observations.gpu_argmax,
        "routes":observations.forwards.iter().map(|f| &f.route).collect::<Vec<_>>(),
        "completed_route_events":observations.route_events,
        "unforced":true,"initial_readiness":format!("{:?}", observations.readiness),
    }))?;
    let mut seen = BTreeSet::new();
    for forward in &observations.forwards {
        let sequence = forward.planned_sequence.context("actual row lacks its source-assigned projection identity")?;
        ensure!(seen.insert(sequence), "actual source projection observed twice");
        let request = *plan.forwards.get(&sequence).context("actual projection was not required")?;
        request.check(forward)?;
        let first = request.first_row()?;
        let transaction = usize::try_from(sequence)?;
        for (stage, values, width) in [(Stage::Logits, &forward.logits, request.vocab),
            (Stage::LayerOutput, &forward.hidden, request.hidden_width)] {
            for row in 0..request.projected_rows {
                trace::capture_bits(&mut store, coordinate(&agent, transaction, first + row, stage),
                    &values[row * width..(row + 1) * width], vec![1, width])?;
            }
        }
    }
    for (index, token) in &observations.decisions {
        let request = plan.decisions.get(index).context("actual semantic selector was not required")?;
        store.capture(request.raw_expected(&agent).at, Scalar::U32,
            vec![1], token as *const u32 as usize as u64, 4, 0, None,
            "production semantic selector after checked native terminal", &token.to_le_bytes())
            .map_err(anyhow::Error::msg)?;
    }
    let trace = store.finish().map_err(anyhow::Error::msg)?;
    ensure!(trace.coverage.errors.is_empty() && trace.coverage.missing_native_capabilities.is_empty()
        && trace.coverage.expected.len() == trace.coverage.captured.len(),
        "Incomplete: {name}/{arm} differs from independently registered full extents");
    let trace_path = root.join("trace.json");
    let trace_sha256 = super::metadata::hash_file(&trace_path, budget.as_ref())?;
    let native_inputs = super::native_inputs::write(super::registration::HeaderRef {
        identity: &plan.context.identity, acquisition: &plan.context.acquisition,
        registration_sha256: &plan.context.registration_sha256,
        case: &plan.context.case, arm: &plan.context.arm, slot: plan.context.slot,
    }, &fixture.packet.output, &budget, observations, &plan, &root)?;
    let binding_path = root.join("binding.json");
    let binding_sha256 = super::metadata::write(&binding_path, budget.as_ref(), &Binding {
        schema: "hf2q.c2.raw-arm-binding.v1", identity: &fixture.packet.identity,
        scope_acquisition: fixture.observed(&fixture.packet.output)?, case: name, arm,
        execution: &plan.execution,
        trace: acquisition::FileBinding { path: trace_path.strip_prefix(&fixture.packet.output)?.into(), sha256: trace_sha256 },
        route_observation: acquisition::FileBinding { path: route_path.strip_prefix(&fixture.packet.output)?.into(), sha256: route_sha256 },
        native_inputs,
    })?;
    budget.borrow_mut().record_artifact(name, &format!("{name}-{arm}"),
        &fixture.packet.output, &binding_path, binding_sha256)?;
    Ok(())
}
