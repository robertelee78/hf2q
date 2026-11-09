//! Issue #727 (spill and restore on reclaim): copy one physical agent slot of
//! a multi-slot [`HybridKvCache`] to host memory and restore it into any slot
//! index.
//!
//! Every per-slot buffer keeps the slot on its outermost axis, and none of a
//! slot's bytes depend on the slot index (the ADR-063 slot-independence gate
//! proves a request is bitwise equal in every slot). A spill copies exactly
//! the slot's observable state:
//!
//! - each full-attention layer and the MTP block: rows `0..cursor` of every
//!   head of the TQ codes and norms, or of the F32 K/V when TQ is off, plus
//!   that block's cursor. The MTP cursor moves independently of the target
//!   cursor, so each block keeps its own.
//! - each DeltaNet layer: the slot's region of both physical conv buffers and
//!   both physical recurrent buffers, plus the slot's ping-pong parity. The
//!   restored slot then holds the same bytes in the same roles, so its next
//!   read and write see exactly what the source slot would have seen.
//! - the slot's graft-region length. The graft rows sit below the cursors
//!   and are copied with them.
//!
//! Rows at or above a cursor are unobservable and are not copied. The
//! per-position DeltaNet capture buffers are scratch inside one speculative
//! transaction; spill and restore refuse to run while capture is active,
//! which it never is between requests.
//!
//! Storage is host `Vec<u8>` filled by CPU `memcpy` over the shared-storage
//! Metal buffers: the same access the slot-anchor checkpoint uses at the same
//! admission point. The caller must not have GPU work in flight on the slot.

use anyhow::{anyhow, Result};
use mlx_native::MlxBuffer;

use super::{copy_slot_region_in, copy_slot_region_out, FullAttnKvSlot, HybridKvCache};
use crate::inference::kv_slot_rows::{read_rows, write_rows, SlotRows};
use crate::serve::multi_seq_kv::SlotId;

/// Cursor-visible state of one full-attention block (a layer or the MTP
/// block) for one slot.
#[cfg_attr(test, derive(PartialEq, Eq))]
struct SpilledAttention {
    cursor: u32,
    /// Rows `0..cursor` of every head, head-major, of each K/V buffer the
    /// block holds, in [`attention_buffers`] order.
    rows: Vec<Vec<u8>>,
}

/// One DeltaNet layer's state for one slot.
#[cfg_attr(test, derive(PartialEq, Eq))]
struct SpilledLinear {
    pp_flipped: bool,
    /// The slot's region of each buffer, in [`linear_buffers`] order.
    regions: Vec<Vec<u8>>,
}

/// Host copy of one physical slot's complete observable cache state. See the
/// module documentation for what it holds.
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct HybridKvSlotSpill {
    full_attn: Vec<SpilledAttention>,
    mtp: Option<SpilledAttention>,
    linear_attn: Vec<SpilledLinear>,
    graft_len: u32,
}

impl HybridKvSlotSpill {
    /// Host bytes owned by this spill.
    pub fn total_bytes(&self) -> usize {
        let attention = self
            .full_attn
            .iter()
            .chain(self.mtp.iter())
            .flat_map(|block| block.rows.iter())
            .map(Vec::len);
        let linear = self
            .linear_attn
            .iter()
            .flat_map(|layer| layer.regions.iter())
            .map(Vec::len);
        attention.chain(linear).sum()
    }
}

/// The K/V buffers of one block in a fixed order: TQ codes/norms, then F32.
fn attention_buffers(block: &FullAttnKvSlot) -> Vec<&MlxBuffer> {
    let mut buffers = Vec::with_capacity(6);
    if let Some(tq) = &block.tq {
        buffers.extend([&tq.k_packed, &tq.k_norms, &tq.v_packed, &tq.v_norms]);
    }
    buffers.extend(block.k.as_ref());
    buffers.extend(block.v.as_ref());
    buffers
}

fn attention_buffers_mut(block: &mut FullAttnKvSlot) -> Vec<&mut MlxBuffer> {
    let mut buffers = Vec::with_capacity(6);
    if let Some(tq) = block.tq.as_mut() {
        buffers.extend([
            &mut tq.k_packed,
            &mut tq.k_norms,
            &mut tq.v_packed,
            &mut tq.v_norms,
        ]);
    }
    buffers.extend(block.k.as_mut());
    buffers.extend(block.v.as_mut());
    buffers
}

const LINEAR_BUFFER_NAMES: [&str; 4] = [
    "conv_state",
    "conv_state_scratch",
    "recurrent",
    "recurrent_scratch",
];

fn linear_buffers(layer: &super::LinearAttnStateSlot) -> [&MlxBuffer; 4] {
    [
        &layer.conv_state,
        &layer.conv_state_scratch,
        &layer.recurrent,
        &layer.recurrent_scratch,
    ]
}

fn linear_buffers_mut(layer: &mut super::LinearAttnStateSlot) -> [&mut MlxBuffer; 4] {
    [
        &mut layer.conv_state,
        &mut layer.conv_state_scratch,
        &mut layer.recurrent,
        &mut layer.recurrent_scratch,
    ]
}

fn block_cursor(block: &FullAttnKvSlot, slot_idx: usize, name: &str) -> Result<u32> {
    block
        .current_len
        .get(slot_idx)
        .copied()
        .ok_or_else(|| anyhow!("slot spill ({name}): cursor missing for slot {slot_idx}"))
}

fn spill_attention(
    block: &FullAttnKvSlot,
    slot_idx: usize,
    n_seqs: usize,
    name: &str,
) -> Result<SpilledAttention> {
    let cursor = block_cursor(block, slot_idx, name)?;
    let rows = attention_buffers(block)
        .into_iter()
        .enumerate()
        .map(|(index, buffer)| {
            read_rows(
                buffer,
                slot_idx,
                n_seqs,
                cursor as usize,
                &format!("{name} buffer {index}"),
            )
        })
        .collect::<Result<_>>()?;
    Ok(SpilledAttention { cursor, rows })
}

fn attention_spill_bytes(
    block: &FullAttnKvSlot,
    slot_idx: usize,
    n_seqs: usize,
    name: &str,
) -> Result<usize> {
    let cursor = block_cursor(block, slot_idx, name)? as usize;
    attention_buffers(block)
        .into_iter()
        .map(|buffer| SlotRows::of(buffer, n_seqs, name)?.spilled_len(cursor, name))
        .sum()
}

fn validate_attention(
    block: &FullAttnKvSlot,
    spilled: &SpilledAttention,
    slot_idx: usize,
    n_seqs: usize,
    name: &str,
) -> Result<()> {
    block_cursor(block, slot_idx, name)?;
    let buffers = attention_buffers(block);
    anyhow::ensure!(
        buffers.len() == spilled.rows.len(),
        "slot spill ({name}): spill holds {} K/V buffers, cache holds {} (TQ/F32 mode mismatch)",
        spilled.rows.len(),
        buffers.len()
    );
    for (index, (buffer, rows)) in buffers.into_iter().zip(&spilled.rows).enumerate() {
        let expected =
            SlotRows::of(buffer, n_seqs, name)?.spilled_len(spilled.cursor as usize, name)?;
        anyhow::ensure!(
            rows.len() == expected,
            "slot spill ({name} buffer {index}): spilled {} bytes, cache geometry needs {expected}",
            rows.len()
        );
    }
    Ok(())
}

fn restore_attention(
    block: &mut FullAttnKvSlot,
    spilled: &SpilledAttention,
    slot_idx: usize,
    n_seqs: usize,
    name: &str,
) -> Result<()> {
    for (index, (buffer, rows)) in attention_buffers_mut(block)
        .into_iter()
        .zip(&spilled.rows)
        .enumerate()
    {
        write_rows(
            rows,
            buffer,
            slot_idx,
            n_seqs,
            spilled.cursor as usize,
            &format!("{name} buffer {index}"),
        )?;
    }
    block.current_len[slot_idx] = spilled.cursor;
    Ok(())
}

impl HybridKvCache {
    fn slot_spill_index(&self, slot: SlotId, operation: &str) -> Result<usize> {
        let slot_idx = slot.0 as usize;
        anyhow::ensure!(
            slot_idx < self.n_seqs as usize,
            "{operation}: slot {} outside n_seqs={}",
            slot.0,
            self.n_seqs
        );
        anyhow::ensure!(
            !self.la_capture_active(),
            "{operation}: DeltaNet capture is active; a slot can only move between transactions"
        );
        Ok(slot_idx)
    }

    /// Host bytes [`Self::spill_slot`] would copy for `slot`, without copying.
    pub(crate) fn slot_spill_bytes(&self, slot: SlotId) -> Result<usize> {
        let slot_idx = self.slot_spill_index(slot, "slot_spill_bytes")?;
        let n_seqs = self.n_seqs as usize;
        let mut total = 0usize;
        for (layer, block) in self.full_attn.iter().enumerate() {
            total +=
                attention_spill_bytes(block, slot_idx, n_seqs, &format!("full_attn[{layer}]"))?;
        }
        if let Some(mtp) = &self.mtp_slot {
            total += attention_spill_bytes(mtp, slot_idx, n_seqs, "mtp")?;
        }
        for (layer, linear) in self.linear_attn.iter().enumerate() {
            for (buffer, name) in linear_buffers(linear).into_iter().zip(LINEAR_BUFFER_NAMES) {
                anyhow::ensure!(
                    buffer.byte_len() % n_seqs == 0,
                    "slot_spill_bytes: linear_attn[{layer}].{name} is not split by slot"
                );
                total += buffer.byte_len() / n_seqs;
            }
        }
        Ok(total)
    }

    /// Copy `slot`'s complete observable state to host memory. The cache is
    /// not modified.
    pub(crate) fn spill_slot(&self, slot: SlotId) -> Result<HybridKvSlotSpill> {
        let slot_idx = self.slot_spill_index(slot, "spill_slot")?;
        let n_seqs = self.n_seqs as usize;
        let full_attn = self
            .full_attn
            .iter()
            .enumerate()
            .map(|(layer, block)| {
                spill_attention(block, slot_idx, n_seqs, &format!("full_attn[{layer}]"))
            })
            .collect::<Result<_>>()?;
        let mtp = self
            .mtp_slot
            .as_ref()
            .map(|mtp| spill_attention(mtp, slot_idx, n_seqs, "mtp"))
            .transpose()?;
        let linear_attn = self
            .linear_attn
            .iter()
            .enumerate()
            .map(|(layer, linear)| {
                let pp_flipped = *linear.pp_flipped.get(slot_idx).ok_or_else(|| {
                    anyhow!("spill_slot: linear_attn[{layer}] parity missing for slot {slot_idx}")
                })?;
                let regions = linear_buffers(linear)
                    .into_iter()
                    .zip(LINEAR_BUFFER_NAMES)
                    .map(|(buffer, name)| {
                        copy_slot_region_out(
                            buffer,
                            slot_idx,
                            n_seqs,
                            &format!("linear_attn[{layer}].{name}"),
                        )
                    })
                    .collect::<Result<_>>()?;
                Ok(SpilledLinear {
                    pp_flipped,
                    regions,
                })
            })
            .collect::<Result<_>>()?;
        let graft_len = *self
            .graft_len
            .get(slot_idx)
            .ok_or_else(|| anyhow!("spill_slot: graft length missing for slot {slot_idx}"))?;
        Ok(HybridKvSlotSpill {
            full_attn,
            mtp,
            linear_attn,
            graft_len,
        })
    }

    /// Restore a spill into `slot`, which may differ from the slot it was
    /// taken from. Afterwards `slot`'s cursors, parity, graft length, every
    /// cursor-visible K/V row and every DeltaNet byte equal the source slot
    /// at spill time; peers are untouched. The whole spill is validated
    /// against this cache's geometry before the first byte is written.
    pub(crate) fn restore_slot_spill(
        &mut self,
        slot: SlotId,
        spill: &HybridKvSlotSpill,
    ) -> Result<()> {
        let slot_idx = self.slot_spill_index(slot, "restore_slot_spill")?;
        let n_seqs = self.n_seqs as usize;
        anyhow::ensure!(
            spill.full_attn.len() == self.full_attn.len(),
            "restore_slot_spill: spill holds {} full-attention layers, cache holds {}",
            spill.full_attn.len(),
            self.full_attn.len()
        );
        anyhow::ensure!(
            spill.linear_attn.len() == self.linear_attn.len(),
            "restore_slot_spill: spill holds {} DeltaNet layers, cache holds {}",
            spill.linear_attn.len(),
            self.linear_attn.len()
        );
        for (layer, (block, spilled)) in self.full_attn.iter().zip(&spill.full_attn).enumerate() {
            validate_attention(
                block,
                spilled,
                slot_idx,
                n_seqs,
                &format!("full_attn[{layer}]"),
            )?;
        }
        match (self.mtp_slot.as_ref(), spill.mtp.as_ref()) {
            (Some(mtp), Some(spilled)) => {
                validate_attention(mtp, spilled, slot_idx, n_seqs, "mtp")?
            }
            (None, None) => {}
            _ => anyhow::bail!("restore_slot_spill: MTP presence differs between spill and cache"),
        }
        for (layer, (linear, spilled)) in
            self.linear_attn.iter().zip(&spill.linear_attn).enumerate()
        {
            anyhow::ensure!(
                slot_idx < linear.pp_flipped.len(),
                "restore_slot_spill: linear_attn[{layer}] parity missing for slot {slot_idx}"
            );
            anyhow::ensure!(
                spilled.regions.len() == LINEAR_BUFFER_NAMES.len(),
                "restore_slot_spill: linear_attn[{layer}] spill holds {} regions",
                spilled.regions.len()
            );
            for ((buffer, name), region) in linear_buffers(linear)
                .into_iter()
                .zip(LINEAR_BUFFER_NAMES)
                .zip(&spilled.regions)
            {
                anyhow::ensure!(
                    buffer.byte_len() % n_seqs == 0 && region.len() == buffer.byte_len() / n_seqs,
                    "restore_slot_spill: linear_attn[{layer}].{name} spilled {} bytes, slot region is {}",
                    region.len(),
                    buffer.byte_len() / n_seqs
                );
            }
        }
        anyhow::ensure!(
            slot_idx < self.graft_len.len(),
            "restore_slot_spill: graft length missing for slot {slot_idx}"
        );

        for (layer, (block, spilled)) in self.full_attn.iter_mut().zip(&spill.full_attn).enumerate()
        {
            restore_attention(
                block,
                spilled,
                slot_idx,
                n_seqs,
                &format!("full_attn[{layer}]"),
            )?;
        }
        if let (Some(mtp), Some(spilled)) = (self.mtp_slot.as_mut(), spill.mtp.as_ref()) {
            restore_attention(mtp, spilled, slot_idx, n_seqs, "mtp")?;
        }
        for (layer, (linear, spilled)) in self
            .linear_attn
            .iter_mut()
            .zip(&spill.linear_attn)
            .enumerate()
        {
            for ((buffer, name), region) in linear_buffers_mut(linear)
                .into_iter()
                .zip(LINEAR_BUFFER_NAMES)
                .zip(&spilled.regions)
            {
                copy_slot_region_in(
                    region,
                    buffer,
                    slot_idx,
                    n_seqs,
                    &format!("linear_attn[{layer}].{name}"),
                )?;
            }
            linear.pp_flipped[slot_idx] = spilled.pp_flipped;
        }
        self.graft_len[slot_idx] = spill.graft_len;
        Ok(())
    }
}

#[cfg(test)]
#[path = "slot_spill_tests.rs"]
mod tests;
