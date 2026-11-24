//! Full final semantic cache state, never a digest/final-logit proxy. Scratch
//! and rows at/above the cursor are outside the restore contract and omitted.
use super::*;
use super::trace::State;
fn capture(store: &mut Store, agent: &str, segment: u32, scalar: Scalar, shape: Vec<usize>,
    bytes: &[u8]) -> Result<()> {
    store.capture(Coordinate { logical_step: 0, layer: 0, stage: Stage::CacheState,
        agent: agent.into(), logical_row_start: 0, segment }, scalar, shape,
        bytes.as_ptr() as usize as u64, bytes.len() as u64, 0, None,
        "complete semantic slot state after production checked transaction", bytes).map_err(anyhow::Error::msg)
}
pub(super) fn write(fixture: &Fixture, name: &str, actual: &State, twin: &State,
    actual_ledger: &[u32], twin_ledger: &[u32]) -> Result<()> {
    for (arm, state, ledger) in [("actual", actual, actual_ledger), ("twin", twin, twin_ledger)] {
        let key = format!("{name}-{arm}-state");
        let expected = fixture.expected(&key)?;
        let records = expected.len();
        let root = fixture.packet.output.join(&key);
        let mut store = Store::new(&root, fixture.packet.identity.clone(),
            format!("C2-ClassA-complete-semantic-state-{arm}"), expected,
            fixture.packet.host_payload_limit, records).map_err(anyhow::Error::msg)?;
        store.bind_acquisition(fixture.observed(&root)?).map_err(anyhow::Error::msg)?;
        let agent = format!("{name}/slot0");
        let cursors: Vec<_> = state.target.iter().copied().chain(state.mtp).collect();
        let bytes: Vec<_> = cursors.iter().flat_map(|v| v.to_le_bytes()).collect();
        capture(&mut store, &agent, 0, Scalar::U32, vec![cursors.len()], &bytes)?;
        let parity: Vec<_> = state.parity.iter().map(|p| u8::from(*p)).collect();
        capture(&mut store, &agent, 1, Scalar::U8, vec![parity.len()], &parity)?;
        for (index, bytes) in state.regions.iter().enumerate() {
            if bytes.is_empty() { continue; }
            capture(&mut store, &agent, u32::try_from(index + 2)?, Scalar::U8, vec![bytes.len()], bytes)?;
        }
        if !ledger.is_empty() {
            let tokens: Vec<_> = ledger.iter().flat_map(|v| v.to_le_bytes()).collect();
            capture(&mut store, &agent, u32::MAX, Scalar::U32, vec![ledger.len()], &tokens)?;
        }
        let trace = store.finish().map_err(anyhow::Error::msg)?;
        ensure!(trace.coverage.errors.is_empty(), "Incomplete semantic cache/ledger evidence");
    }
    Ok(())
}
