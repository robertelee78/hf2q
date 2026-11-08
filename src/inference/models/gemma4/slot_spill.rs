//! Issue #727 (spill and restore on reclaim): copy one physical Gemma 4
//! agent slot to host memory and restore it into any slot index, or move it
//! between two slots in place.
//!
//! Every per-slot buffer of the `MultiSeq*` scaffolds keeps the slot on its
//! outermost axis, and none of a slot's bytes depend on the slot index, so a
//! copy restored into another slot holds exactly the bytes the source slot
//! held. A spill copies the slot's observable state:
//!
//! - the rows of each layer of the *live* scaffold, the one the slot-aware
//!   prefill and decode dispatch reads and writes ([`GemmaSlotKvRegime::live`]):
//!   - global layers: rows `0..cursor` of every head;
//!   - sliding layers: the ring rows `0..min(cursor, capacity)`; once the
//!     ring has wrapped every ring row is live;
//!   - per buffer: hybrid F16 K, TQ-HB V codes (or F16 V) and V norms (the
//!     full-F16-V dummy norm buffer is shared, not per slot), and the BF16
//!     xlen K/V when allocated; HB and nibble K/V codes and norms; dense K
//!     and V.
//! - the slot's cursor in every layer of every provisioned scaffold. The
//!   engine keeps them uniform (graft rows plus token rows), and a restore
//!   reproduces each one.
//!
//! The graft rows sit below the cursors and are copied with them. Rows at or
//! above a cursor are unobservable, with one exception the caller states:
//! a published prompt anchor restores only the sliding rings and relies on
//! the append-only global rows below its boundary, so `rows_floor` extends
//! the copy to that boundary. Provisioned scaffolds that are not live are
//! never written by the forward; their cursors are their only state.
//!
//! Storage is host `Vec<u8>` filled by CPU `memcpy` over the shared-storage
//! Metal buffers, the same access the slot-anchor checkpoint uses at the same
//! admission point. The caller must not have GPU work in flight on either
//! slot.

use anyhow::{anyhow, Result};
use mlx_native::MlxBuffer;

use super::{
    MultiSeqDenseKvBuffers, MultiSeqHbKvBuffers, MultiSeqHybridKvBuffers, MultiSeqMlxKvCache,
};
use crate::inference::kv_slot_rows::{copy_rows_between_slots, read_rows, write_rows, SlotRows};
use crate::serve::multi_seq_kv::SlotId;

/// The Gemma 4 SlotAware KV scaffolds, in provisioning order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GemmaSlotKvRegime {
    /// TQ-HB K and V (`HF2Q_HYBRID_KV=0`).
    Hb,
    /// F16 K and TQ-HB V, the production default.
    Hybrid,
    /// Dense F32/F16 K and V (`HF2Q_USE_DENSE=1`).
    Dense,
    /// Legacy 4-bit nibble-packed K and V (`HF2Q_TQ_CODEBOOK_BITS=4`).
    Nibble,
}

const REGIMES: [GemmaSlotKvRegime; 4] = [
    GemmaSlotKvRegime::Hb,
    GemmaSlotKvRegime::Hybrid,
    GemmaSlotKvRegime::Dense,
    GemmaSlotKvRegime::Nibble,
];

impl GemmaSlotKvRegime {
    /// The scaffold the slot-aware forwards read and write, by the dispatch
    /// precedence of `forward_prefill_with_soft_tokens_slot_aware` and
    /// `forward_decode_slot_aware`: dense, then the 4-bit codebook, then
    /// hybrid, then HB.
    pub(crate) fn live() -> Self {
        let env = &crate::debug::INVESTIGATION_ENV;
        if env.use_dense {
            Self::Dense
        } else if crate::serve::api::tq_packed_descriptor::effective_gemma_tq_codebook_bits() == 0 {
            Self::Nibble
        } else if env.hybrid_kv {
            Self::Hybrid
        } else {
            Self::Hb
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Hb => 0,
            Self::Hybrid => 1,
            Self::Dense => 2,
            Self::Nibble => 3,
        }
    }
}

/// One layer of a slot-major Gemma scaffold.
trait SlotLayer {
    fn n_seqs(&self) -> usize;
    fn capacity(&self) -> usize;
    fn seq_lens(&self) -> &[u32];
    fn seq_lens_mut(&mut self) -> &mut [u32];
    /// The per-slot buffers in a fixed order. Shared buffers (the full-F16-V
    /// dummy norms) hold no slot state and are excluded.
    fn slot_buffers(&self) -> Vec<&MlxBuffer>;
    fn slot_buffers_mut(&mut self) -> Vec<&mut MlxBuffer>;
}

impl SlotLayer for MultiSeqHbKvBuffers {
    fn n_seqs(&self) -> usize {
        self.n_seqs as usize
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn seq_lens(&self) -> &[u32] {
        &self.seq_lens
    }
    fn seq_lens_mut(&mut self) -> &mut [u32] {
        &mut self.seq_lens
    }
    fn slot_buffers(&self) -> Vec<&MlxBuffer> {
        vec![&self.k_packed, &self.k_norms, &self.v_packed, &self.v_norms]
    }
    fn slot_buffers_mut(&mut self) -> Vec<&mut MlxBuffer> {
        vec![
            &mut self.k_packed,
            &mut self.k_norms,
            &mut self.v_packed,
            &mut self.v_norms,
        ]
    }
}

/// The full-F16-V scaffold allocates a 4-byte V-norm placeholder shared by
/// every slot (`cache_scaffold_plan`); the anchor checkpoint skips it too.
fn hybrid_has_slot_v_norms(layer: &MultiSeqHybridKvBuffers) -> bool {
    layer.v_norms.byte_len() != 4
}

impl SlotLayer for MultiSeqHybridKvBuffers {
    fn n_seqs(&self) -> usize {
        self.n_seqs as usize
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn seq_lens(&self) -> &[u32] {
        &self.seq_lens
    }
    fn seq_lens_mut(&mut self) -> &mut [u32] {
        &mut self.seq_lens
    }
    fn slot_buffers(&self) -> Vec<&MlxBuffer> {
        let mut buffers = vec![&self.k, &self.v_packed];
        if hybrid_has_slot_v_norms(self) {
            buffers.push(&self.v_norms);
        }
        buffers.extend(self.bf16_xlen_k.as_ref());
        buffers.extend(self.bf16_xlen_v.as_ref());
        buffers
    }
    fn slot_buffers_mut(&mut self) -> Vec<&mut MlxBuffer> {
        let slot_norms = hybrid_has_slot_v_norms(self);
        let Self {
            k,
            v_packed,
            v_norms,
            bf16_xlen_k,
            bf16_xlen_v,
            ..
        } = self;
        let mut buffers = vec![k, v_packed];
        if slot_norms {
            buffers.push(v_norms);
        }
        buffers.extend(bf16_xlen_k.as_mut());
        buffers.extend(bf16_xlen_v.as_mut());
        buffers
    }
}

impl SlotLayer for MultiSeqDenseKvBuffers {
    fn n_seqs(&self) -> usize {
        self.n_seqs as usize
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn seq_lens(&self) -> &[u32] {
        &self.seq_lens
    }
    fn seq_lens_mut(&mut self) -> &mut [u32] {
        &mut self.seq_lens
    }
    fn slot_buffers(&self) -> Vec<&MlxBuffer> {
        vec![&self.k, &self.v]
    }
    fn slot_buffers_mut(&mut self) -> Vec<&mut MlxBuffer> {
        vec![&mut self.k, &mut self.v]
    }
}

impl SlotLayer for MultiSeqMlxKvCache {
    fn n_seqs(&self) -> usize {
        self.n_seqs as usize
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn seq_lens(&self) -> &[u32] {
        &self.seq_lens
    }
    fn seq_lens_mut(&mut self) -> &mut [u32] {
        &mut self.seq_lens
    }
    fn slot_buffers(&self) -> Vec<&MlxBuffer> {
        vec![&self.k_packed, &self.k_norms, &self.v_packed, &self.v_norms]
    }
    fn slot_buffers_mut(&mut self) -> Vec<&mut MlxBuffer> {
        vec![
            &mut self.k_packed,
            &mut self.k_norms,
            &mut self.v_packed,
            &mut self.v_norms,
        ]
    }
}

/// The worker's scaffolds (`Gemma4KvGuard`): HB is always provisioned, the
/// others when their regime is enabled at spawn.
pub(crate) struct GemmaSlotScaffolds<'a> {
    pub(crate) hb: &'a mut [MultiSeqHbKvBuffers],
    pub(crate) hybrid: Option<&'a mut [MultiSeqHybridKvBuffers]>,
    pub(crate) dense: Option<&'a mut [MultiSeqDenseKvBuffers]>,
    pub(crate) nibble: Option<&'a mut [MultiSeqMlxKvCache]>,
}

/// One live-scaffold layer of a spilled slot.
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
struct SpilledLayer {
    rows: usize,
    /// Rows `0..rows` of every head of each [`SlotLayer::slot_buffers`]
    /// buffer, head-major.
    buffers: Vec<Vec<u8>>,
}

/// Host copy of one physical Gemma slot's complete observable cache state.
/// See the module documentation for what it holds.
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(crate) struct GemmaKvSlotSpill {
    regime: GemmaSlotKvRegime,
    layers: Vec<SpilledLayer>,
    /// The slot's per-layer cursor in each scaffold, indexed by
    /// [`GemmaSlotKvRegime`] order; `None` for an unprovisioned scaffold.
    cursors: [Option<Vec<u32>>; 4],
}

impl GemmaKvSlotSpill {
    /// Host bytes owned by this spill (`spill_bytes` reports the same
    /// without copying).
    #[cfg(test)]
    pub(crate) fn total_bytes(&self) -> usize {
        let rows: usize = self
            .layers
            .iter()
            .flat_map(|layer| layer.buffers.iter())
            .map(Vec::len)
            .sum();
        let cursors: usize = self
            .cursors
            .iter()
            .flatten()
            .map(|cursors| cursors.len() * std::mem::size_of::<u32>())
            .sum();
        rows + cursors
    }

    /// The live-scaffold cursor of the first layer (all layers agree).
    #[cfg(test)]
    pub(crate) fn cursor(&self) -> Option<u32> {
        self.cursors[self.regime.index()]
            .as_ref()
            .and_then(|cursors| cursors.first().copied())
    }
}

fn layer_cursor(layer: &dyn SlotLayer, slot_idx: usize, name: &str) -> Result<usize> {
    anyhow::ensure!(
        slot_idx < layer.n_seqs() && slot_idx < layer.seq_lens().len(),
        "{name}: slot {slot_idx} outside n_seqs={}",
        layer.n_seqs()
    );
    Ok(layer.seq_lens()[slot_idx] as usize)
}

/// Rows of one layer a spill copies: the cursor (or the caller's floor),
/// bounded by the layer's ring or full capacity.
fn spilled_rows(
    layer: &dyn SlotLayer,
    slot_idx: usize,
    rows_floor: usize,
    name: &str,
) -> Result<usize> {
    Ok(layer_cursor(layer, slot_idx, name)?
        .max(rows_floor)
        .min(layer.capacity()))
}

impl GemmaSlotScaffolds<'_> {
    fn layers(&self, regime: GemmaSlotKvRegime) -> Option<Vec<&dyn SlotLayer>> {
        fn erase<T: SlotLayer + 'static>(layers: &[T]) -> Vec<&dyn SlotLayer> {
            layers.iter().map(|layer| layer as &dyn SlotLayer).collect()
        }
        match regime {
            GemmaSlotKvRegime::Hb => Some(erase(&*self.hb)),
            GemmaSlotKvRegime::Hybrid => self.hybrid.as_deref().map(erase),
            GemmaSlotKvRegime::Dense => self.dense.as_deref().map(erase),
            GemmaSlotKvRegime::Nibble => self.nibble.as_deref().map(erase),
        }
    }

    fn layers_mut(&mut self, regime: GemmaSlotKvRegime) -> Option<Vec<&mut dyn SlotLayer>> {
        fn erase<T: SlotLayer + 'static>(layers: &mut [T]) -> Vec<&mut dyn SlotLayer> {
            layers
                .iter_mut()
                .map(|layer| layer as &mut dyn SlotLayer)
                .collect()
        }
        match regime {
            GemmaSlotKvRegime::Hb => Some(erase(&mut *self.hb)),
            GemmaSlotKvRegime::Hybrid => self.hybrid.as_deref_mut().map(erase),
            GemmaSlotKvRegime::Dense => self.dense.as_deref_mut().map(erase),
            GemmaSlotKvRegime::Nibble => self.nibble.as_deref_mut().map(erase),
        }
    }

    fn live_layers(
        &self,
        regime: GemmaSlotKvRegime,
        operation: &str,
    ) -> Result<Vec<&dyn SlotLayer>> {
        self.layers(regime)
            .ok_or_else(|| anyhow!("{operation}: live {regime:?} KV scaffold is not provisioned"))
    }

    /// Host bytes [`Self::spill`] would copy, without copying.
    pub(crate) fn spill_bytes(
        &self,
        regime: GemmaSlotKvRegime,
        slot: SlotId,
        rows_floor: usize,
    ) -> Result<usize> {
        let slot_idx = slot.0 as usize;
        let mut total = 0usize;
        for (index, layer) in self
            .live_layers(regime, "gemma slot_spill_bytes")?
            .into_iter()
            .enumerate()
        {
            let name = format!("gemma {regime:?} L{index}");
            let rows = spilled_rows(layer, slot_idx, rows_floor, &name)?;
            for buffer in layer.slot_buffers() {
                total += SlotRows::of(buffer, layer.n_seqs(), &name)?.spilled_len(rows, &name)?;
            }
        }
        for scaffold in REGIMES {
            if let Some(layers) = self.layers(scaffold) {
                total += layers.len() * std::mem::size_of::<u32>();
            }
        }
        Ok(total)
    }

    /// Copy `slot`'s complete observable state to host memory. The scaffolds
    /// are not modified.
    pub(crate) fn spill(
        &self,
        regime: GemmaSlotKvRegime,
        slot: SlotId,
        rows_floor: usize,
    ) -> Result<GemmaKvSlotSpill> {
        let slot_idx = slot.0 as usize;
        let layers = self
            .live_layers(regime, "gemma spill_slot")?
            .into_iter()
            .enumerate()
            .map(|(index, layer)| {
                let name = format!("gemma {regime:?} L{index}");
                let rows = spilled_rows(layer, slot_idx, rows_floor, &name)?;
                let buffers = layer
                    .slot_buffers()
                    .into_iter()
                    .enumerate()
                    .map(|(buffer_index, buffer)| {
                        read_rows(
                            buffer,
                            slot_idx,
                            layer.n_seqs(),
                            rows,
                            &format!("{name} buffer {buffer_index}"),
                        )
                    })
                    .collect::<Result<_>>()?;
                Ok(SpilledLayer { rows, buffers })
            })
            .collect::<Result<_>>()?;
        let mut cursors: [Option<Vec<u32>>; 4] = Default::default();
        for scaffold in REGIMES {
            if let Some(layers) = self.layers(scaffold) {
                cursors[scaffold.index()] = Some(
                    layers
                        .into_iter()
                        .enumerate()
                        .map(|(index, layer)| {
                            layer_cursor(layer, slot_idx, &format!("gemma {scaffold:?} L{index}"))
                                .map(|cursor| cursor as u32)
                        })
                        .collect::<Result<_>>()?,
                );
            }
        }
        Ok(GemmaKvSlotSpill {
            regime,
            layers,
            cursors,
        })
    }

    /// Check `spill` against these scaffolds' geometry for `slot` before
    /// any byte is written.
    fn validate_restore(&self, slot_idx: usize, spill: &GemmaKvSlotSpill) -> Result<()> {
        let live = self.live_layers(spill.regime, "gemma restore_slot_spill")?;
        anyhow::ensure!(
            live.len() == spill.layers.len(),
            "gemma restore_slot_spill: spill holds {} {:?} layers, cache holds {}",
            spill.layers.len(),
            spill.regime,
            live.len()
        );
        for (index, (layer, spilled)) in live.into_iter().zip(&spill.layers).enumerate() {
            let name = format!("gemma {:?} L{index}", spill.regime);
            layer_cursor(layer, slot_idx, &name)?;
            let buffers = layer.slot_buffers();
            anyhow::ensure!(
                buffers.len() == spilled.buffers.len(),
                "{name}: spill holds {} buffers, cache holds {} (KV layout changed)",
                spilled.buffers.len(),
                buffers.len()
            );
            for (buffer_index, (buffer, rows)) in
                buffers.into_iter().zip(&spilled.buffers).enumerate()
            {
                let expected = SlotRows::of(buffer, layer.n_seqs(), &name)?
                    .spilled_len(spilled.rows, &name)?;
                anyhow::ensure!(
                    rows.len() == expected,
                    "{name} buffer {buffer_index}: spilled {} bytes, cache geometry needs {expected}",
                    rows.len()
                );
            }
        }
        for scaffold in REGIMES {
            match (self.layers(scaffold), spill.cursors[scaffold.index()].as_ref()) {
                (Some(layers), Some(cursors)) => {
                    anyhow::ensure!(
                        layers.len() == cursors.len(),
                        "gemma restore_slot_spill: {scaffold:?} holds {} layers, spill {}",
                        layers.len(),
                        cursors.len()
                    );
                    for (index, layer) in layers.into_iter().enumerate() {
                        layer_cursor(layer, slot_idx, &format!("gemma {scaffold:?} L{index}"))?;
                    }
                }
                (None, None) => {}
                _ => anyhow::bail!(
                    "gemma restore_slot_spill: {scaffold:?} scaffold presence differs between spill and cache"
                ),
            }
        }
        Ok(())
    }

    /// Restore a spill into `slot`, which may differ from the slot it was
    /// taken from. Afterwards every cursor of `slot` and every spilled row
    /// equal the source slot at spill time; peers are untouched. The whole
    /// spill is validated before the first byte is written.
    pub(crate) fn restore(&mut self, slot: SlotId, spill: &GemmaKvSlotSpill) -> Result<()> {
        let slot_idx = slot.0 as usize;
        self.validate_restore(slot_idx, spill)?;
        let regime = spill.regime;
        let live = self
            .layers_mut(regime)
            .expect("live scaffold validated before the restore");
        for (index, (layer, spilled)) in live.into_iter().zip(&spill.layers).enumerate() {
            let n_seqs = layer.n_seqs();
            for (buffer_index, (buffer, rows)) in layer
                .slot_buffers_mut()
                .into_iter()
                .zip(&spilled.buffers)
                .enumerate()
            {
                write_rows(
                    rows,
                    buffer,
                    slot_idx,
                    n_seqs,
                    spilled.rows,
                    &format!("gemma {regime:?} L{index} buffer {buffer_index}"),
                )?;
            }
        }
        for scaffold in REGIMES {
            if let (Some(layers), Some(cursors)) = (
                self.layers_mut(scaffold),
                spill.cursors[scaffold.index()].as_ref(),
            ) {
                for (layer, &cursor) in layers.into_iter().zip(cursors) {
                    layer.seq_lens_mut()[slot_idx] = cursor;
                }
            }
        }
        Ok(())
    }

    /// Move `source`'s state into `dest` in place: the rows a spill of
    /// `source` would copy and every cursor, without a host copy. `source`
    /// keeps its bytes and cursors; `dest`'s previous state is overwritten.
    pub(crate) fn copy_slot(
        &mut self,
        regime: GemmaSlotKvRegime,
        source: SlotId,
        dest: SlotId,
        rows_floor: usize,
    ) -> Result<()> {
        let (src_idx, dst_idx) = (source.0 as usize, dest.0 as usize);
        for scaffold in REGIMES {
            if let Some(layers) = self.layers(scaffold) {
                for (index, layer) in layers.into_iter().enumerate() {
                    let name = format!("gemma {scaffold:?} L{index}");
                    layer_cursor(layer, src_idx, &name)?;
                    layer_cursor(layer, dst_idx, &name)?;
                }
            }
        }
        let live = self.layers_mut(regime).ok_or_else(|| {
            anyhow!("gemma copy_slot: live {regime:?} KV scaffold is not provisioned")
        })?;
        for (index, layer) in live.into_iter().enumerate() {
            let name = format!("gemma {regime:?} L{index}");
            let rows = spilled_rows(&*layer, src_idx, rows_floor, &name)?;
            let n_seqs = layer.n_seqs();
            for (buffer_index, buffer) in layer.slot_buffers_mut().into_iter().enumerate() {
                copy_rows_between_slots(
                    buffer,
                    src_idx,
                    dst_idx,
                    n_seqs,
                    rows,
                    &format!("{name} buffer {buffer_index}"),
                )?;
            }
        }
        for scaffold in REGIMES {
            if let Some(layers) = self.layers_mut(scaffold) {
                for layer in layers {
                    let cursor = layer.seq_lens()[src_idx];
                    layer.seq_lens_mut()[dst_idx] = cursor;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "slot_spill_tests.rs"]
mod tests;
