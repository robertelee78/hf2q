//! Independent extent registration is packet-owned. Captured row counts are
//! NEVER used to manufacture an expected manifest, including for negative controls.
use super::*;
use super::trace::{Observations, coordinate};

pub(super) fn write_pair(fixture: &Fixture, name: &str, actual: &Observations,
    expected: &Observations) -> Result<()> {
    write_arm(fixture, name, "actual", actual)?;
    write_arm(fixture, name, "twin", expected)
}

pub(super) fn write_arm(fixture: &Fixture, name: &str, arm: &str,
    observations: &Observations) -> Result<()> {
    ensure!(matches!(arm, "actual" | "twin" | "single"), "unknown matrix oracle arm");
    let agent = format!("{name}/slot0");
    let extents = fixture.expected(&format!("{name}-{arm}"))?;
    ensure!(!extents.is_empty() && extents.iter().all(|e| e.at.agent == agent),
        "independent extent agent binding differs for {name}/{arm}");
    let root = fixture.packet.output.join(format!("{name}-{arm}"));
    let records = extents.len();
    let mut store = Store::new(&root, fixture.packet.identity.clone(),
        format!("C2-ClassA-{name}-{arm}"), extents, fixture.packet.host_payload_limit, records)
        .map_err(anyhow::Error::msg)?;
    store.bind_acquisition(fixture.observed(&root)?).map_err(anyhow::Error::msg)?;
    fs::write(root.join("route-observation.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema":"hf2q.c2.route-observation.v1",
        "actual_gpu_argmax":observations.gpu_argmax,
        "routes":observations.forwards.iter().map(|f| &f.route).collect::<Vec<_>>(),
        "completed_route_events":observations.route_events,
        "unforced":true,"initial_readiness":format!("{:?}", observations.readiness),
    }))?)?;
    for (transaction, forward) in observations.forwards.iter().filter(|f| f.slot == 0).enumerate() {
        ensure!(forward.vocab != 0 && forward.logits.len() % forward.vocab == 0,
            "actual full-vocabulary row extent malformed");
        let count = forward.logits.len() / forward.vocab;
        ensure!(count <= forward.input.len() && forward.hidden_width != 0
            && forward.hidden.len() == count * forward.hidden_width,
            "actual projected logit/normalized-hidden rows differ");
        let first = forward.start + forward.input.len() - count;
        for (row, logits) in forward.logits.chunks_exact(forward.vocab).enumerate() {
            let values: Vec<_> = logits.iter().map(|v| f32::from_bits(*v)).collect();
            trace::capture_f32(&mut store, coordinate(&agent, transaction, first + row, Stage::Logits),
                &values, vec![1, forward.vocab])?;
        }
        for (row, hidden) in forward.hidden.chunks_exact(forward.hidden_width).enumerate() {
            let values: Vec<_> = hidden.iter().map(|v| f32::from_bits(*v)).collect();
            trace::capture_f32(&mut store, coordinate(&agent, transaction, first + row,
                Stage::LayerOutput), &values, vec![1, forward.hidden_width])?;
        }
    }
    for (index, token) in &observations.decisions {
        store.capture(coordinate(&agent, *index, *index, Stage::SpecCanonicalIds), Scalar::U32,
            vec![1], token as *const u32 as usize as u64, 4, 0, None,
            "production semantic selector after checked native terminal", &token.to_le_bytes())
            .map_err(anyhow::Error::msg)?;
    }
    let trace = store.finish().map_err(anyhow::Error::msg)?;
    ensure!(trace.coverage.errors.is_empty()
        && trace.coverage.expected.len() == trace.coverage.captured.len(),
        "Incomplete: {name}/{arm} differs from independently registered full extents");
    Ok(())
}
