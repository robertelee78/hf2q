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
                let Some(logical) = physical.checked_sub(graft) else { return false; };
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
    for (arm, rows) in [("actual", &actual), ("twin", &twin)] {
        let key = format!("{name}-{arm}");
        let expected = fixture.expected(&key)?;
        ensure!(expected.iter().all(|extent| extent.at.agent == agent), "canonical extent agent differs");
        let records = expected.len();
        let root = fixture.packet.output.join(&key);
        let mut store = Store::new(&root, fixture.packet.identity.clone(),
            format!("C2-ClassA-complete-canonical-single-slot-parity-{arm}"), expected,
            fixture.packet.host_payload_limit, records).map_err(anyhow::Error::msg)?;
        store.bind_acquisition(fixture.observed(&root)?).map_err(anyhow::Error::msg)?;
        for row in rows {
            for (stage, scalar, values) in [(Stage::Logits, Scalar::F32, row.logits),
                (Stage::LayerOutput, Scalar::F32, row.hidden)] {
                let bytes: Vec<_> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
                store.capture(trace::coordinate(&agent, row.index, row.position, stage), scalar,
                    vec![1, values.len()], values.as_ptr() as usize as u64, bytes.len() as u64, 0, None,
                    "complete actual canonical target row after checked native completion", &bytes)
                    .map_err(anyhow::Error::msg)?;
            }
            store.capture(trace::coordinate(&agent, row.index, row.index, Stage::SpecCanonicalIds),
                Scalar::U32, vec![1], &row.token as *const u32 as usize as u64, 4, 0, None,
                "actual canonical semantic selector including terminal decisions", &row.token.to_le_bytes())
                .map_err(anyhow::Error::msg)?;
        }
        fs::write(root.join("actual-row-routes.json"), serde_json::to_vec_pretty(
            &rows.iter().map(|row| (row.index, row.position, row.route)).collect::<Vec<_>>())?)?;
        let receipt = store.finish().map_err(anyhow::Error::msg)?;
        ensure!(receipt.coverage.errors.is_empty() && receipt.coverage.expected.len() == receipt.coverage.captured.len(),
            "Incomplete: canonical rows differ from independently registered full extents");
    }
    for (actual, twin) in actual.iter().zip(&twin) {
        ensure!(actual.index == twin.index && actual.position == twin.position && actual.token == twin.token
            && actual.logits == twin.logits && actual.hidden == twin.hidden,
            "DivergedExactInvariant: complete canonical per-position logits/hidden/token single-slot parity");
    }
    Ok(actual.len())
}
