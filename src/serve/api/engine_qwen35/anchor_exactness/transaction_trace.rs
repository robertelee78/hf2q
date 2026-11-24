//! Every successful scheduler-visible prefill/decode transaction streams its
//! complete actual semantic cache/ledger into an independently registered Q1
//! store. No state digest, last-row sample, or output-derived expected extent.
use super::*;
use super::trace::State;
pub(super) struct Writer {
    store: Store,
    agent: String,
    step: usize,
}
impl Writer {
    pub(super) fn new(fixture: &Fixture, case: &str, arm: &str) -> Result<Self> {
        let key = format!("{case}-{arm}-transactions");
        let expected = fixture.expected(&key)?;
        let count = expected.len();
        let root = fixture.packet.output.join(&key);
        let mut store = Store::new(&root, fixture.packet.identity.clone(),
            format!("C2-ClassA-complete-actual-transactions-{arm}"), expected,
            fixture.packet.host_payload_limit, count).map_err(anyhow::Error::msg)?;
        store.bind_acquisition(fixture.observed(&root)?).map_err(anyhow::Error::msg)?;
        Ok(Self { store, agent: format!("{case}/slot0"), step: 0 })
    }
    pub(super) fn capture(&mut self, cache: &HybridKvCache, slot: SlotId, ledger: &[u32]) -> Result<()> {
        let state = State::read(cache, slot)?;
        let mut record = |segment, scalar, shape, bytes: &[u8]| -> Result<()> {
            self.store.capture(Coordinate { logical_step: self.step as u64, layer: 0,
                stage: Stage::CacheState, agent: self.agent.clone(), logical_row_start: 0, segment },
                scalar, shape, bytes.as_ptr() as usize as u64, bytes.len() as u64, 0, None,
                "actual successful scheduler-visible transaction returned after checked native terminal",
                bytes).map_err(anyhow::Error::msg)
        };
        let cursors: Vec<_> = state.target.iter().copied().chain(state.mtp).collect();
        let bytes: Vec<_> = cursors.iter().flat_map(|v| v.to_le_bytes()).collect();
        record(0, Scalar::U32, vec![cursors.len()], &bytes)?;
        let parity: Vec<_> = state.parity.iter().map(|p| u8::from(*p)).collect();
        record(1, Scalar::U8, vec![parity.len()], &parity)?;
        for (index, bytes) in state.regions.iter().enumerate() {
            if !bytes.is_empty() { record(u32::try_from(index + 2)?, Scalar::U8, vec![bytes.len()], bytes)?; }
        }
        if !ledger.is_empty() {
            let tokens: Vec<_> = ledger.iter().flat_map(|v| v.to_le_bytes()).collect();
            record(u32::MAX, Scalar::U32, vec![ledger.len()], &tokens)?;
        }
        self.step += 1;
        Ok(())
    }
    pub(super) fn finish(self) -> Result<()> {
        let trace = self.store.finish().map_err(anyhow::Error::msg)?;
        ensure!(trace.coverage.errors.is_empty(), "Incomplete: transaction extent differs from independent script");
        Ok(())
    }
}
