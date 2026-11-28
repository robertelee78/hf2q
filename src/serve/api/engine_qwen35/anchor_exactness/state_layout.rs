//! Semantic extents derived from allocation descriptors and CPU cursors before
//! any tensor value is read. Empty regions remain explicit in the layout but
//! never become invalid zero-sized Q1 tensors.
use super::*;
use crate::inference::models::qwen35::kv_cache::FullAttnKvSlot;
use mlx_native::{DType, MlxBuffer};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StateLayout {
    pub cursors: usize,
    pub parity: usize,
    pub regions: Vec<usize>,
    pub ledger: usize,
}

fn full_bytes(shape: &[usize], scalar_bytes: usize, slot: SlotId, valid: usize) -> Result<usize> {
    ensure!(shape.len() == 4 && (slot.0 as usize) < shape[0] && valid <= shape[2],
        "independent full-attention layout exceeds allocator/cursor");
    shape[1].checked_mul(valid).and_then(|n| n.checked_mul(shape[3]))
        .and_then(|n| n.checked_mul(scalar_bytes))
        .context("full-attention semantic layout overflow")
}

fn full(layer: &FullAttnKvSlot, slot: SlotId, regions: &mut Vec<usize>) -> Result<()> {
    let valid = *layer.current_len.get(slot.0 as usize)
        .context("independent full-attention cursor absent")? as usize;
    let mut add = |buffer: &MlxBuffer| -> Result<()> {
        ensure!(matches!(buffer.dtype(), DType::U8 | DType::F32),
            "independent semantic layout requires explicit U8/F32 storage");
        let bytes = full_bytes(buffer.shape(), buffer.dtype().size_of(), slot, valid)?;
        ensure!(bytes <= buffer.data_byte_len(), "semantic region exceeds logical allocation");
        regions.push(bytes);
        Ok(())
    };
    if let Some(tq) = &layer.tq {
        for buffer in [&tq.k_packed, &tq.k_norms, &tq.v_packed, &tq.v_norms] { add(buffer)?; }
    }
    for buffer in [&layer.k, &layer.v].into_iter().flatten() { add(buffer)?; }
    Ok(())
}

fn linear(buffer: &MlxBuffer, slots: usize, slot: SlotId) -> Result<usize> {
    ensure!(slots > 0 && (slot.0 as usize) < slots
        && buffer.dtype() == DType::F32 && buffer.shape().last() == Some(&slots)
        && buffer.element_count() % slots == 0,
        "independent linear layout requires the allocator's contiguous slot span");
    (buffer.element_count() / slots).checked_mul(buffer.dtype().size_of())
        .context("linear semantic layout overflow")
}

impl StateLayout {
    pub(super) fn from_cache(cache: &HybridKvCache, slot: SlotId, ledger: usize) -> Result<Self> {
        ensure!(slot.0 < cache.n_seqs, "independent semantic slot outside allocation");
        let mut regions = Vec::with_capacity(
            (cache.full_attn.len() + usize::from(cache.mtp_slot.is_some())) * 6
                + cache.linear_attn.len() * 2);
        for layer in &cache.full_attn { full(layer, slot, &mut regions)?; }
        if let Some(mtp) = &cache.mtp_slot { full(mtp, slot, &mut regions)?; }
        for layer in &cache.linear_attn {
            ensure!((slot.0 as usize) < layer.pp_flipped.len(), "independent parity slot absent");
            regions.push(linear(layer.conv_bufs_for_slot(slot).0, cache.n_seqs as usize, slot)?);
            regions.push(linear(layer.recurrent_bufs_for_slot(slot).0, cache.n_seqs as usize, slot)?);
        }
        let result = Self { cursors: cache.full_attn.len() + usize::from(cache.mtp_slot.is_some()),
            parity: cache.linear_attn.len(), regions, ledger };
        ensure!(result.cursors > 0 && result.parity > 0,
            "C2 hybrid artifacts require target cursors and DeltaNet parity");
        result.logical_bytes()?;
        Ok(result)
    }

    pub(super) fn logical_bytes(&self) -> Result<u64> {
        let mut bytes = self.cursors.checked_mul(4).and_then(|n| n.checked_add(self.parity))
            .and_then(|n| self.ledger.checked_mul(4).and_then(|ledger| n.checked_add(ledger)))
            .context("semantic cursor/parity/ledger byte count overflow")?;
        for &region in &self.regions {
            bytes = bytes.checked_add(region).context("semantic region byte count overflow")?;
        }
        Ok(u64::try_from(bytes)?)
    }

    pub(super) fn expected(&self, agent: &str, logical_step: u64) -> Result<Vec<ExpectedTensor>> {
        ensure!(self.cursors > 0 && self.parity > 0, "empty hybrid state layout");
        self.logical_bytes()?;
        let mut expected = Vec::with_capacity(self.regions.len() + 3);
        let mut add = |segment, scalar, length| {
            expected.push(ExpectedTensor { at: Coordinate { logical_step, layer: 0,
                stage: Stage::CacheState, agent: agent.into(), logical_row_start: 0, segment },
                scalar, shape: vec![length] });
        };
        add(0, Scalar::U32, self.cursors);
        add(1, Scalar::U8, self.parity);
        for (index, &bytes) in self.regions.iter().enumerate() {
            if bytes != 0 { add(u32::try_from(index + 2)?, Scalar::U8, bytes); }
        }
        if self.ledger != 0 { add(u32::MAX, Scalar::U32, self.ledger); }
        Ok(expected)
    }

    pub(super) fn check(&self, state: &super::trace::State, ledger: &[u32]) -> Result<()> {
        ensure!(state.target.len() + usize::from(state.mtp.is_some()) == self.cursors
            && state.parity.len() == self.parity && state.regions.len() == self.regions.len()
            && ledger.len() == self.ledger,
            "observed semantic metadata differs from independent layout");
        ensure!(state.regions.iter().zip(&self.regions).all(|(bytes, expected)| bytes.len() == *expected),
            "observed semantic bytes differ from allocator-derived layout");
        Ok(())
    }
}

#[test]
fn c2_state_layout_refuses_out_of_cursor_slot_and_overflow_extents() {
    assert_eq!(full_bytes(&[4, 3, 8, 5], 4, SlotId(2), 4).unwrap(), 240);
    assert_eq!(full_bytes(&[4, 3, 8, 5], 4, SlotId(2), 0).unwrap(), 0);
    assert!(full_bytes(&[4, 3, 8, 5], 4, SlotId(2), 9).is_err());
    assert!(full_bytes(&[4, 3, 8, 5], 4, SlotId(4), 1).is_err());
    assert!(full_bytes(&[4, 3, 8], 4, SlotId(0), 1).is_err());
    assert!(full_bytes(&[4, usize::MAX, 8, 5], 4, SlotId(0), 8).is_err());
}

#[test]
fn c2_state_layout_cannot_hide_bytes_in_a_declared_empty_region() {
    let layout = StateLayout { cursors: 1, parity: 1, regions: vec![8, 0], ledger: 2 };
    let mut state = super::trace::State { target: vec![2], mtp: None, parity: vec![false],
        regions: vec![vec![0; 8], Vec::new()] };
    layout.check(&state, &[1, 2]).unwrap();
    state.regions[1].push(0);
    assert!(layout.check(&state, &[1, 2]).is_err());
    state.regions[1].clear();
    state.regions[0].pop();
    assert!(layout.check(&state, &[1, 2]).is_err());
}

pub(super) fn capture(store: &mut Store, layout: &StateLayout, state: &super::trace::State,
    ledger: &[u32], agent: &str, logical_step: u64) -> Result<()> {
    layout.check(state, ledger)?;
    let mut record = |segment, scalar, length, bytes: &[u8]| -> Result<()> {
        store.capture(Coordinate { logical_step, layer: 0, stage: Stage::CacheState,
            agent: agent.into(), logical_row_start: 0, segment }, scalar, vec![length],
            bytes.as_ptr() as usize as u64, bytes.len() as u64, 0, None,
            "complete semantic state after checked native terminal; layout registered before readback",
            bytes).map_err(anyhow::Error::msg)
    };
    let mut cursors = Vec::with_capacity(layout.cursors * 4);
    for cursor in state.target.iter().copied().chain(state.mtp) {
        cursors.extend_from_slice(&cursor.to_le_bytes());
    }
    record(0, Scalar::U32, layout.cursors, &cursors)?;
    let parity: Vec<_> = state.parity.iter().map(|p| u8::from(*p)).collect();
    record(1, Scalar::U8, layout.parity, &parity)?;
    for (index, (bytes, &length)) in state.regions.iter().zip(&layout.regions).enumerate() {
        if length != 0 { record(u32::try_from(index + 2)?, Scalar::U8, length, bytes)?; }
    }
    if layout.ledger != 0 {
        let mut tokens = Vec::with_capacity(layout.ledger * 4);
        for token in ledger { tokens.extend_from_slice(&token.to_le_bytes()); }
        record(u32::MAX, Scalar::U32, layout.ledger, &tokens)?;
    }
    Ok(())
}
