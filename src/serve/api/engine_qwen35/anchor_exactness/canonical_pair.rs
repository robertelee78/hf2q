//! Four-slot per-position parity is separate from same-transaction parity:
//! production cap8 legitimately changes rejected verifier tails with lane width.
//! Every accepted semantic decision (terminal included) still owes its COMPLETE
//! raw vocabulary row and corresponding actual hidden row to the single-slot twin.
use super::*;

struct Row<'a> {
    index: usize,
    position: usize,
    token: u32,
    logits: &'a [u32],
    hidden: &'a [u32],
    route: &'a str,
}
fn rows<'a>(observed: &'a Observations, prompt: &[u32], tokens: &[u32], graft: usize) -> Result<Vec<Row<'a>>> {
    let decisions = &observed.decisions;
    ensure!(decisions.len() >= tokens.len()
        && decisions.iter().enumerate().all(|(index, &(actual, _))| index == actual),
        "Incomplete: canonical decisions omit, repeat or reorder an emitted/intermediate position");
    for (index, &token) in tokens.iter().enumerate() {
        ensure!(decisions.get(index).is_some_and(|&(_, selected)| selected == token),
            "emitted token differs from its actual canonical selector");
    }
    let mut result = Vec::with_capacity(decisions.len());
    for &(index, token) in decisions {
        let position = graft + prompt.len() - 1 + index;
        let mut selected: Option<Row<'a>> = None;
        for forward in observed.forwards.iter().filter(|forward| forward.slot == SLOT.0) {
            ensure!(forward.vocab > 0 && forward.logits.len() % forward.vocab == 0,
                "malformed actual vocabulary-row extent");
            let count = forward.logits.len() / forward.vocab;
            let first = (forward.start + forward.input.len()).checked_sub(count)
                .context("actual forward output rows exceed transaction input")?;
            if position < first || position >= first + count { continue; }
            // Rejected proposal tails are checked in full by the same-width
            // raw twin. They are not alternate canonical decision candidates.
            let canonical_input = forward.input.iter().enumerate().all(|(row, actual)| {
                let physical = forward.start + row;
                if physical > position { return true; }
                // Bound graft-bank rows have no logical prompt-token index.
                // Their complete native rows remain covered by the raw twin.
                if physical < graft { return true; }
                let logical = physical - graft;
                prompt.get(logical).or_else(|| logical.checked_sub(prompt.len())
                    .and_then(|index| tokens.get(index))) == Some(actual)
            });
            if !canonical_input { continue; }
            ensure!(forward.hidden_width > 0 && forward.hidden.len() == count * forward.hidden_width,
                "Incomplete: canonical full-logit row has no corresponding actual hidden row");
            let row = position - first;
            let candidate = Row { index, position, token,
                logits: &forward.logits[row * forward.vocab..(row + 1) * forward.vocab],
                hidden: &forward.hidden[row * forward.hidden_width..(row + 1) * forward.hidden_width],
                route: &forward.route };
            if let Some(prior) = &selected {
                ensure!(prior.logits == candidate.logits && prior.hidden == candidate.hidden,
                    "DivergedExactInvariant: repeated canonical target row changed");
            }
            selected = Some(candidate);
        }
        result.push(selected.context("Incomplete: actual accepted/terminal decision lacks its full target row")?);
    }
    Ok(result)
}
pub(super) fn write_and_compare(fixture: &Fixture, case: &str, prompt: &[u32],
    actual: &Observations, actual_tokens: &[u32], twin: &Observations, twin_tokens: &[u32],
    graft: usize) -> Result<usize> {
    ensure!(actual_tokens == twin_tokens, "DivergedExactInvariant: four-slot/single-slot emitted token witness");
    let actual = rows(actual, prompt, actual_tokens, graft)?;
    let twin = rows(twin, prompt, twin_tokens, graft)?;
    ensure!(actual.len() == twin.len(), "DivergedExactInvariant: complete accepted/terminal decision extent");
    let name = format!("{case}-single-slot-parity");
    let agent = format!("{name}/slot0");
    let budget = fixture.matrix_budget()?;
    for (arm, source_arm, rows) in [("actual", "actual", &actual), ("twin", "single", &twin)] {
        let key = format!("{name}-{arm}");
        let mut plan = super::super::registered_plan::load(fixture, case, source_arm)?;
        ensure!(rows.len() == plan.decisions.len(), "canonical rows omit required semantic selectors");
        let expected = std::mem::take(&mut plan.canonical_expected);
        ensure!(expected.iter().all(|extent| extent.at.agent == agent), "canonical extent agent differs");
        let records = expected.len();
        let root = fixture.packet.output.join(&key);
        let mut store = Store::new(&root, fixture.packet.identity.clone(),
            format!("C2-ClassA-complete-canonical-single-slot-parity-{arm}"), expected,
            fixture.packet.host_payload_limit, records).map_err(anyhow::Error::msg)?;
        store.bind_acquisition(fixture.observed(&root)?).map_err(anyhow::Error::msg)?;
        for row in rows {
            let request = plan.decisions.get(&row.index).context("canonical row has no source selector")?;
            ensure!(u64::try_from(row.position)? == request.physical_position
                && row.logits.len() == request.vocab && row.hidden.len() == request.hidden_width,
                "canonical row differs from its source-declared full geometry");
            for (stage, values) in [(Stage::Logits, row.logits), (Stage::LayerOutput, row.hidden)] {
                trace::capture_bits(&mut store, trace::coordinate(&agent, row.index, row.position, stage),
                    values, vec![1, values.len()])?;
            }
            store.capture(trace::coordinate(&agent, row.index, row.position, Stage::SpecCanonicalIds),
                Scalar::U32, vec![1], &row.token as *const u32 as usize as u64, 4, 0, None,
                "actual canonical semantic selector including terminal decisions", &row.token.to_le_bytes())
                .map_err(anyhow::Error::msg)?;
        }
        let route_path = root.join("actual-row-routes.json");
        let route_sha256 = super::super::metadata::write(&route_path, budget.as_ref(),
            &rows.iter().map(|row| (row.index, row.position, row.route)).collect::<Vec<_>>())?;
        let receipt = store.finish().map_err(anyhow::Error::msg)?;
        ensure!(receipt.coverage.errors.is_empty() && receipt.coverage.expected.len() == receipt.coverage.captured.len(),
            "Incomplete: canonical rows differ from independently registered full extents");
        let trace_path = root.join("trace.json");
        let trace_sha256 = super::super::metadata::hash_file(&trace_path, budget.as_ref())?;
        let binding_path = root.join("binding.json");
        let binding_sha256 = super::super::metadata::write(&binding_path, budget.as_ref(), &serde_json::json!({
            "schema":"hf2q.c2.canonical-arm-binding.v1","identity":fixture.packet.identity,
            "scope_acquisition":fixture.observed(&fixture.packet.output)?,"case":case,"arm":arm,
            "source_arm":source_arm,"execution":plan.execution,
            "trace":{"path":trace_path.strip_prefix(&fixture.packet.output)?,"sha256":trace_sha256},
            "route_observation":{"path":route_path.strip_prefix(&fixture.packet.output)?,"sha256":route_sha256},
        }))?;
        budget.borrow_mut().record_artifact(case, &key, &fixture.packet.output, &binding_path, binding_sha256)?;
    }
    for (actual, twin) in actual.iter().zip(&twin) {
        ensure!(actual.index == twin.index && actual.position == twin.position && actual.token == twin.token
            && actual.logits == twin.logits && actual.hidden == twin.hidden,
            "DivergedExactInvariant: complete canonical per-position logits/hidden/token single-slot parity");
    }
    Ok(actual.len())
}

#[cfg(test)]
mod controls {
    use super::*;

    #[test]
    fn c2_canonical_graft_rows_preserve_logical_prefix_validation() {
        let mut observed = Observations::default();
        observed.decisions.push((0, 1));
        observed.forwards.push(trace::Forward { slot: 0, start: 0, input: vec![1, 0, 1],
            route: "synthetic CPU graft-prefix geometry".into(), logits: vec![0, 1.0_f32.to_bits()],
            hidden: vec![0.5_f32.to_bits()], vocab: 2, hidden_width: 1, planned_sequence: Some(0), input_sha256: None });
        let accepted = rows(&observed, &[0, 1], &[1], 1).unwrap();
        assert_eq!(accepted[0].position, 2);
        drop(accepted);
        observed.forwards[0].input[1] = 1;
        assert!(rows(&observed, &[0, 1], &[1], 1).is_err(),
            "virtual bank rows must not hide a changed logical prompt token");
    }
}
