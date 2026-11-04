//! ADR-063 Q4-GEMMA-SPEC-SERVE — verify-route (R-A) building blocks.
//!
//! - [`DecodeWindow`] / [`WindowRows`] / [`layout_windows`]: the rank-major
//!   row layout of a windowed decode tick (design §5.3). Rank 0 holds every
//!   window's first row in window order, so a tick of width-1 windows is
//!   exactly today's one-row-per-slot batch.
//! - [`RingPreimagePlan`]: which sliding-ring rows a verify of width `K+1`
//!   at `pos0` destroys, and which of them are restored after `m` rows were
//!   committed (design §5.5).
//! - [`GemmaRingPreimage`]: the per-lane pre-image buffers and the save
//!   (linearize) / restore (N1) dispatches over every sliding layer and
//!   payload (F16 K, V codes, F32 V norms).
// The serving round (design §5.1) consumes the pre-image plan and buffers;
// until it lands, only the M0 harness and the unit tests do.
#![cfg_attr(not(test), allow(dead_code))]

use anyhow::{ensure, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::kv_cache::MultiSeqHybridKvBuffers;
use crate::serve::multi_seq_kv::SlotId;

/// Widest verify window (K = 3 drafts plus the sampled token).
pub(crate) const MAX_WINDOW: usize = 4;

/// One lane's decode window: `tokens[j]` is processed at `start_pos + j`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodeWindow {
    pub slot: SlotId,
    pub start_pos: usize,
    pub tokens: Vec<u32>,
}

impl DecodeWindow {
    /// An ordinary width-1 decode row.
    pub(crate) fn single(slot: SlotId, start_pos: usize, token: u32) -> Self {
        Self {
            slot,
            start_pos,
            tokens: vec![token],
        }
    }
}

/// Rank-major row map of a windowed tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowRows {
    /// Rows `[rank_offsets[j], rank_offsets[j+1])` hold rank `j` of every
    /// window that has one, in window order. `len() == w_max + 1`.
    pub rank_offsets: Vec<usize>,
    /// `row_of[window][rank]` is the row index of that window position.
    pub row_of: Vec<Vec<usize>>,
}

impl WindowRows {
    /// Total rows `R`.
    pub(crate) fn rows(&self) -> usize {
        *self.rank_offsets.last().unwrap_or(&0)
    }

    /// True iff any window is wider than one row.
    pub(crate) fn is_windowed(&self) -> bool {
        self.rank_offsets.len() > 2
    }
}

/// Per-row inputs of a windowed tick in rank-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RowLayout {
    pub tokens: Vec<u32>,
    pub slot_ids: Vec<SlotId>,
    pub positions: Vec<usize>,
    pub rows: WindowRows,
}

/// Build the rank-major layout (U8). Windows must be non-empty, at most
/// [`MAX_WINDOW`] wide, and address distinct slots.
pub(crate) fn layout_windows(windows: &[DecodeWindow]) -> Result<RowLayout> {
    let mut seen = std::collections::BTreeSet::new();
    let mut w_max = 0usize;
    for window in windows {
        ensure!(
            !window.tokens.is_empty() && window.tokens.len() <= MAX_WINDOW,
            "Gemma decode window width {} outside 1..={MAX_WINDOW}",
            window.tokens.len()
        );
        ensure!(
            seen.insert(window.slot.0),
            "Gemma decode windows address slot {} twice",
            window.slot.0
        );
        w_max = w_max.max(window.tokens.len());
    }
    let total: usize = windows.iter().map(|w| w.tokens.len()).sum();
    let mut tokens = Vec::with_capacity(total);
    let mut slot_ids = Vec::with_capacity(total);
    let mut positions = Vec::with_capacity(total);
    let mut rank_offsets = Vec::with_capacity(w_max + 1);
    let mut row_of: Vec<Vec<usize>> = windows
        .iter()
        .map(|w| Vec::with_capacity(w.tokens.len()))
        .collect();
    rank_offsets.push(0);
    for rank in 0..w_max {
        for (index, window) in windows.iter().enumerate() {
            if let Some(&token) = window.tokens.get(rank) {
                row_of[index].push(tokens.len());
                tokens.push(token);
                slot_ids.push(window.slot);
                positions.push(window.start_pos + rank);
            }
        }
        rank_offsets.push(tokens.len());
    }
    Ok(RowLayout {
        tokens,
        slot_ids,
        positions,
        rows: WindowRows {
            rank_offsets,
            row_of,
        },
    })
}

/// Pre-image save of a lane whose verify window reaches the wrapped ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RingSave {
    /// First verify position whose ring write destroys readable history.
    pub r_lo: usize,
    /// Rows saved (`pos0 + K − r_lo + 1`, 1..=K).
    pub n_saved: usize,
    /// Exclusive logical end passed to the linearize (`pos0 + K + 1 − cap`).
    pub logical_end: usize,
}

/// Tail restore after the walk committed `m` of the window's rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RingRestore {
    pub src_row_offset: usize,
    pub n_restore: usize,
    pub logical_end: usize,
}

/// The sliding-ring rollback plan for one lane's window (U6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RingPreimagePlan {
    pub pos0: usize,
    pub k: usize,
    pub save: Option<RingSave>,
}

impl RingPreimagePlan {
    /// Plan for a verify window `[pos0, pos0 + k]` over a ring of `cap` rows.
    /// Rows at positions `< cap` overwrite never-written slots, and the
    /// rank-0 write at `pos0` destroys `pos0 − cap`, which no cursor
    /// `≥ pos0 + 1` reads: neither has a pre-image.
    pub(crate) fn new(pos0: usize, k: usize, cap: usize) -> Result<Self> {
        ensure!(cap > 0, "ring capacity must be positive");
        ensure!(k < MAX_WINDOW, "draft depth {k} exceeds {}", MAX_WINDOW - 1);
        ensure!(k < cap, "draft depth {k} must be below ring capacity {cap}");
        let end = pos0 + k;
        let save = if k > 0 && end >= cap {
            let r_lo = (pos0 + 1).max(cap);
            Some(RingSave {
                r_lo,
                n_saved: end - r_lo + 1,
                logical_end: end + 1 - cap,
            })
        } else {
            None
        };
        Ok(Self { pos0, k, save })
    }

    /// Restore after committing `m` rows (`1 ..= k + 1`); `None` when every
    /// rejected row had no pre-image (or nothing was rejected).
    pub(crate) fn restore(&self, m: usize) -> Result<Option<RingRestore>> {
        ensure!(
            (1..=self.k + 1).contains(&m),
            "committed rows {m} outside 1..={}",
            self.k + 1
        );
        let Some(save) = self.save else {
            return Ok(None);
        };
        let end = self.pos0 + self.k;
        let start = (self.pos0 + m).max(save.r_lo);
        if m == self.k + 1 || start > end {
            return Ok(None);
        }
        Ok(Some(RingRestore {
            src_row_offset: start - save.r_lo,
            n_restore: end - start + 1,
            logical_end: save.logical_end,
        }))
    }
}

/// One sliding layer payload of one slot: `(byte offset, data bytes)` of
/// the slot view plus the head-major geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PayloadGeometry {
    slot_bytes: usize,
    row_bytes: usize,
}

/// Head-major geometry of one sliding layer's three payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlidingLayerGeometry {
    pub layer_idx: usize,
    pub n_kv_heads: usize,
    pub capacity: usize,
    payloads: [PayloadGeometry; 3],
}

/// Derive the sliding layers' payload geometry from the live scaffold.
/// Refuses layouts the pre-image cannot express (dummy V norms, per-slot
/// extents that do not factor into `[nkv, capacity, row]`).
pub(crate) fn sliding_layer_geometry(
    hybrid: &[MultiSeqHybridKvBuffers],
    n_kv_heads_of: impl Fn(usize) -> usize,
) -> Result<Vec<SlidingLayerGeometry>> {
    let mut out = Vec::new();
    for (layer_idx, layer) in hybrid.iter().enumerate() {
        if !layer.is_sliding {
            continue;
        }
        let n_seqs = layer.n_seqs as usize;
        let nkv = n_kv_heads_of(layer_idx);
        ensure!(n_seqs > 0 && nkv > 0, "sliding layer {layer_idx}: empty geometry");
        ensure!(
            layer.v_norms.byte_len() != 4,
            "sliding layer {layer_idx}: dummy V norms (full F16 KV) have no pre-image"
        );
        let mut payloads = [PayloadGeometry {
            slot_bytes: 0,
            row_bytes: 0,
        }; 3];
        for (index, buffer) in [&layer.k, &layer.v_packed, &layer.v_norms]
            .into_iter()
            .enumerate()
        {
            let slot_bytes = buffer.byte_len() / n_seqs;
            ensure!(
                slot_bytes * n_seqs == buffer.byte_len()
                    && slot_bytes % (nkv * layer.capacity) == 0,
                "sliding layer {layer_idx} payload {index}: {} bytes do not factor into \
                 [{n_seqs}, {nkv}, {}, row]",
                buffer.byte_len(),
                layer.capacity
            );
            payloads[index] = PayloadGeometry {
                slot_bytes,
                row_bytes: slot_bytes / (nkv * layer.capacity),
            };
        }
        out.push(SlidingLayerGeometry {
            layer_idx,
            n_kv_heads: nkv,
            capacity: layer.capacity,
            payloads,
        });
    }
    Ok(out)
}

/// Per-lane pre-image storage: for every sliding layer and payload a
/// head-major `[nkv, MAX_WINDOW − 1, row]` region.
pub(crate) struct GemmaRingPreimage {
    geometry: Vec<SlidingLayerGeometry>,
    /// `regions[layer][payload] = (byte offset, bytes)` inside `buffer`.
    regions: Vec<[(usize, usize); 3]>,
    buffer: MlxBuffer,
}

const PREIMAGE_ROWS: usize = MAX_WINDOW - 1;

impl GemmaRingPreimage {
    pub(crate) fn new(device: &MlxDevice, geometry: Vec<SlidingLayerGeometry>) -> Result<Self> {
        let mut regions = Vec::with_capacity(geometry.len());
        let mut total = 0usize;
        for layer in &geometry {
            let mut layer_regions = [(0, 0); 3];
            for (index, payload) in layer.payloads.iter().enumerate() {
                let bytes = layer.n_kv_heads * PREIMAGE_ROWS * payload.row_bytes;
                layer_regions[index] = (total, bytes);
                // 16-byte aligned regions.
                total += bytes.div_ceil(16) * 16;
            }
            regions.push(layer_regions);
        }
        let buffer = device
            .alloc_buffer(total.max(16), DType::U8, vec![total.max(16)])
            .map_err(|e| anyhow::anyhow!("Gemma ring pre-image alloc ({total} B): {e}"))?;
        Ok(Self {
            geometry,
            regions,
            buffer,
        })
    }

    /// The slot's view of one payload, bounds-checked against the whole
    /// buffer: the save-side linearize checks `byte_len` only (U6).
    fn slot_view(buffer: &MlxBuffer, slot: SlotId, slot_bytes: usize) -> Result<MlxBuffer> {
        let start = slot.0 as usize * slot_bytes;
        ensure!(
            start + slot_bytes <= buffer.byte_len(),
            "slot {} view [{start}, {}) exceeds payload buffer {} B",
            slot.0,
            start + slot_bytes,
            buffer.byte_len()
        );
        Ok(buffer
            .slice_view(start as u64, slot_bytes / buffer.dtype().size_of())
            .with_shape(vec![slot_bytes / buffer.dtype().size_of()])
            .map_err(|e| anyhow::anyhow!("slot view: {e}"))?)
    }

    fn region(&self, layer: usize, payload: usize) -> MlxBuffer {
        let (offset, bytes) = self.regions[layer][payload];
        self.buffer.slice_view(offset as u64, bytes)
    }

    fn payload_buffers(layer: &MultiSeqHybridKvBuffers) -> [&MlxBuffer; 3] {
        [&layer.k, &layer.v_packed, &layer.v_norms]
    }

    /// Encode the pre-image save of `slot` per `save` over every sliding
    /// layer and payload. Must be encoded before the verify forward.
    pub(crate) fn encode_save(
        &self,
        session: &mut GraphSession<'_>,
        reg: &mut KernelRegistry,
        dev: &MlxDevice,
        hybrid: &[MultiSeqHybridKvBuffers],
        slot: SlotId,
        save: &RingSave,
    ) -> Result<()> {
        ensure!(save.n_saved <= PREIMAGE_ROWS, "pre-image rows {}", save.n_saved);
        for (index, layer) in self.geometry.iter().enumerate() {
            let kv = hybrid
                .get(layer.layer_idx)
                .context("pre-image layer missing from scaffold")?;
            for (payload, buffer) in Self::payload_buffers(kv).into_iter().enumerate() {
                let geometry = layer.payloads[payload];
                let src = Self::slot_view(buffer, slot, geometry.slot_bytes)?;
                let dst = self.region(index, payload);
                session.barrier_between(&[&src], &[&dst]);
                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_linearize_ring_bytes(
                    session.encoder_mut(),
                    reg,
                    dev.metal_device(),
                    &src,
                    &dst,
                    layer.n_kv_heads as u32,
                    layer.capacity as u32,
                    PREIMAGE_ROWS as u32,
                    save.n_saved as u32,
                    save.logical_end as u32,
                    geometry.row_bytes as u32,
                )
                .map_err(|e| {
                    anyhow::anyhow!("pre-image save L{} payload {payload}: {e}", layer.layer_idx)
                })?;
            }
        }
        Ok(())
    }

    /// Encode the tail restore of `slot` (N1) over every sliding layer and
    /// payload. Must run after the walk and before the cursor commit.
    pub(crate) fn encode_restore(
        &self,
        session: &mut GraphSession<'_>,
        reg: &mut KernelRegistry,
        dev: &MlxDevice,
        hybrid: &[MultiSeqHybridKvBuffers],
        slot: SlotId,
        restore: &RingRestore,
    ) -> Result<()> {
        for (index, layer) in self.geometry.iter().enumerate() {
            let kv = hybrid
                .get(layer.layer_idx)
                .context("pre-image layer missing from scaffold")?;
            for (payload, buffer) in Self::payload_buffers(kv).into_iter().enumerate() {
                let geometry = layer.payloads[payload];
                let dst = Self::slot_view(buffer, slot, geometry.slot_bytes)?;
                let src = self.region(index, payload);
                session.barrier_between(&[&src], &[&dst]);
                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_restore_ring_bytes(
                    session.encoder_mut(),
                    reg,
                    dev.metal_device(),
                    &src,
                    &dst,
                    layer.n_kv_heads as u32,
                    PREIMAGE_ROWS as u32,
                    layer.capacity as u32,
                    restore.src_row_offset as u32,
                    restore.n_restore as u32,
                    restore.logical_end as u32,
                    geometry.row_bytes as u32,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "pre-image restore L{} payload {payload}: {e}",
                        layer.layer_idx
                    )
                })?;
            }
        }
        Ok(())
    }

    /// Number of save (or restore) dispatches per lane.
    pub(crate) fn dispatches_per_lane(&self) -> usize {
        self.geometry.len() * 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(slot: u32, pos: usize, tokens: &[u32]) -> DecodeWindow {
        DecodeWindow {
            slot: SlotId(slot),
            start_pos: pos,
            tokens: tokens.to_vec(),
        }
    }

    #[test]
    fn gemma_spec_window_rows_layout() {
        // Width-1 windows: exactly today's one-row-per-slot batch.
        let ordinary = [
            DecodeWindow::single(SlotId(2), 700, 11),
            DecodeWindow::single(SlotId(0), 900, 12),
            DecodeWindow::single(SlotId(3), 1400, 13),
        ];
        let layout = layout_windows(&ordinary).unwrap();
        assert_eq!(layout.tokens, [11, 12, 13]);
        assert_eq!(layout.slot_ids, [SlotId(2), SlotId(0), SlotId(3)]);
        assert_eq!(layout.positions, [700, 900, 1400]);
        assert_eq!(layout.rows.rank_offsets, [0, 3]);
        assert!(!layout.rows.is_windowed());

        // Mixed widths: speculating (w = 2, 4) and ordinary lanes.
        let mixed = [
            window(1, 100, &[1, 2]),
            window(0, 50, &[3]),
            window(3, 1022, &[4, 5, 6, 7]),
            window(2, 7, &[8, 9]),
        ];
        let layout = layout_windows(&mixed).unwrap();
        assert_eq!(layout.rows.rank_offsets, [0, 4, 7, 8, 9]);
        assert_eq!(layout.rows.rows(), 9);
        assert!(layout.rows.is_windowed());
        // Slice 0 equals the one-row-per-slot batch in window order.
        assert_eq!(&layout.tokens[..4], &[1, 3, 4, 8]);
        assert_eq!(&layout.slot_ids[..4], &[SlotId(1), SlotId(0), SlotId(3), SlotId(2)]);
        assert_eq!(&layout.positions[..4], &[100, 50, 1022, 7]);
        // Slice j holds rank j of every window that has one, in window order.
        assert_eq!(&layout.tokens[4..7], &[2, 5, 9]);
        assert_eq!(&layout.positions[4..7], &[101, 1023, 8]);
        assert_eq!(&layout.tokens[7..], &[6, 7]);
        assert_eq!(&layout.positions[7..], &[1024, 1025]);
        assert_eq!(
            layout.rows.row_of,
            vec![vec![0, 4], vec![1], vec![2, 5, 7, 8], vec![3, 6]]
        );
        for (index, rows) in layout.rows.row_of.iter().enumerate() {
            for (rank, &row) in rows.iter().enumerate() {
                assert_eq!(layout.tokens[row], mixed[index].tokens[rank]);
                assert_eq!(layout.slot_ids[row], mixed[index].slot);
                assert_eq!(layout.positions[row], mixed[index].start_pos + rank);
                assert!(
                    (layout.rows.rank_offsets[rank]..layout.rows.rank_offsets[rank + 1])
                        .contains(&row)
                );
            }
        }

        assert!(layout_windows(&[window(0, 1, &[])]).is_err());
        assert!(layout_windows(&[window(0, 1, &[1, 2, 3, 4, 5])]).is_err());
        assert!(layout_windows(&[window(0, 1, &[1]), window(0, 9, &[2])]).is_err());
    }

    /// Reference: simulate the ring as `slot → logical position held` and
    /// check that the plan's restore returns exactly the readable state.
    #[test]
    fn gemma_spec_ring_preimage_plan() {
        let cap = 1024usize;
        let mut pos_cases: Vec<usize> = (0..8).collect();
        pos_cases.extend(1015..1030);
        pos_cases.extend([2047, 2048, 2049, 5000, 4093]);
        for pos0 in pos_cases {
            for k in 1..=3usize {
                let plan = RingPreimagePlan::new(pos0, k, cap).unwrap();
                let end = pos0 + k;
                // No save before wrap.
                assert_eq!(plan.save.is_some(), end >= cap, "pos0={pos0} k={k}");
                if let Some(save) = plan.save {
                    assert_eq!(save.r_lo, (pos0 + 1).max(cap));
                    assert!((1..=k).contains(&save.n_saved));
                    assert_eq!(save.n_saved, end - save.r_lo + 1);
                    assert_eq!(save.logical_end, end + 1 - cap);
                    // Linearize row i is the slot verify position r_lo+i overwrites.
                    for i in 0..save.n_saved {
                        let logical = save.logical_end - save.n_saved + i;
                        assert_eq!(logical + cap, save.r_lo + i);
                    }
                }
                for m in 1..=k + 1 {
                    // Ring before the round: slot s holds the newest p ≤ pos0-1 with p ≡ s.
                    let held_before = |slot: usize| -> Option<usize> {
                        let q = pos0.checked_sub(1)?;
                        let back = (q % cap + cap - slot) % cap;
                        q.checked_sub(back)
                    };
                    let mut ring: Vec<Option<usize>> = (0..cap).map(held_before).collect();
                    for r in pos0..=end {
                        ring[r % cap] = Some(r);
                    }
                    let restore = plan.restore(m).unwrap();
                    if let (Some(save), Some(restore)) = (plan.save, restore) {
                        assert!(restore.src_row_offset + restore.n_restore <= save.n_saved);
                        for i in 0..restore.n_restore {
                            let logical = restore.logical_end - restore.n_restore + i;
                            let pre_row = restore.src_row_offset + i;
                            // Pre-image row pre_row holds logical r_lo+pre_row-cap.
                            assert_eq!(logical, save.r_lo + pre_row - cap);
                            // Accepted rows are never restored.
                            assert!(logical + cap >= pos0 + m);
                            ring[logical % cap] = Some(logical);
                        }
                    }
                    if m == k + 1 {
                        assert_eq!(restore, None);
                    }
                    // Next decode at c reads logical [c-1023, c-1] over the ring.
                    let c = pos0 + m;
                    for p in c.saturating_sub(cap - 1)..c {
                        assert_eq!(ring[p % cap], Some(p), "pos0={pos0} k={k} m={m} p={p}");
                    }
                }
                assert!(plan.restore(0).is_err());
                assert!(plan.restore(k + 2).is_err());
            }
        }
        assert!(RingPreimagePlan::new(2000, 4, cap).is_err());
    }
}
