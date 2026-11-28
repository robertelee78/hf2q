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

#[cfg(test)]
mod scope_controls {
    use super::*;

    // Actual bounded stores with synthetic CPU bytes, never model acquisition.
    fn writer(captured: bool) -> (tempfile::TempDir, Writer) {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().canonicalize().unwrap().join("trace");
        let identity = Identity {
            family: Family::Qwen38,
            source_commit: "a".repeat(40),
            binary_sha256: "b".repeat(64),
            native_version: env!("HF2Q_MLX_NATIVE_VERSION").into(),
            native_checksum: "c".repeat(64),
            model_sha256: "d".repeat(64),
            tokenizer_sha256: "e".repeat(64),
            template_sha256: "f".repeat(64),
            config_sha256: "1".repeat(64),
            token_vectors_sha256: "2".repeat(64),
        };
        let at = Coordinate { logical_step: 0, layer: 0, stage: Stage::CacheState,
            agent: "scope/slot0".into(), logical_row_start: 0, segment: 0 };
        let expected = vec![ExpectedTensor { at: at.clone(), scalar: Scalar::U32, shape: vec![1] }];
        let mut store = Store::new(&root, identity, "model-free-scope-control".into(),
            expected, 4, 1).unwrap();
        if captured {
            let bytes = 7_u32.to_le_bytes();
            store.capture(at, Scalar::U32, vec![1], bytes.as_ptr() as usize as u64,
                bytes.len() as u64, 0, None, "synthetic CPU scope control", &bytes).unwrap();
        }
        (parent, Writer { store, agent: "scope/slot0".into(), step: 0 })
    }

    #[test]
    fn c2_transaction_scope_rejects_unfinished_and_incomplete_capture() {
        let (_root, writer) = writer(false);
        let scope = trace::Scope::begin_transactions(writer, &[]).unwrap();
        assert!(scope.take().is_err(), "an open writer cannot publish observations");
        assert!(scope.finish_transactions().is_err(), "missing registered bytes must fail");
        assert!(scope.take().is_err(), "ignored coverage failure must remain fatal");
    }

    #[test]
    fn c2_transaction_scope_rejects_duplicate_completion() {
        let (_root, writer) = writer(true);
        let scope = trace::Scope::begin_transactions(writer, &[]).unwrap();
        scope.finish_transactions().unwrap();
        assert!(scope.take().is_ok(), "complete registered bytes may publish observations");
        assert!(scope.finish_transactions().is_err(), "completion is one-shot");
        assert!(scope.take().is_err(), "ignored duplicate completion must remain fatal");
    }
}
