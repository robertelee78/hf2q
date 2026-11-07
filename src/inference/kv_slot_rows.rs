//! Issue #727: row copies over slot-major KV buffers, shared by every
//! family's reclaimed-slot spill (`qwen35::kv_cache::slot_spill`,
//! `gemma4::kv_cache::slot_spill`).
//!
//! A slot-major buffer is `[n_seqs, heads, capacity]` or
//! `[n_seqs, heads, capacity, inner]`: each slot owns one contiguous region
//! and each head one contiguous run of `capacity` rows inside it. A spill
//! copies rows `0..rows` of every head, head-major, so the copy holds only
//! the rows a cursor makes observable.
//!
//! Access is CPU `memcpy` over shared-storage Metal buffers; the caller must
//! not have GPU work in flight on either slot.

use anyhow::{anyhow, Context, Result};
use mlx_native::MlxBuffer;

/// Byte geometry of one slot-major K/V buffer.
pub(crate) struct SlotRows {
    heads: usize,
    capacity: usize,
    row_bytes: usize,
    head_stride: usize,
    slot_stride: usize,
}

impl SlotRows {
    pub(crate) fn of(buffer: &MlxBuffer, n_seqs: usize, name: &str) -> Result<Self> {
        let shape = buffer.shape();
        let (heads, capacity, inner) = match *shape {
            [seqs, heads, capacity] if seqs == n_seqs => (heads, capacity, 1),
            [seqs, heads, capacity, inner] if seqs == n_seqs => (heads, capacity, inner),
            _ => anyhow::bail!(
                "slot spill ({name}): expected [n_seqs={n_seqs}, heads, capacity, inner], got {shape:?}"
            ),
        };
        let row_bytes = inner
            .checked_mul(buffer.dtype().size_of())
            .ok_or_else(|| anyhow!("slot spill ({name}): row extent overflow"))?;
        let head_stride = capacity
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow!("slot spill ({name}): head stride overflow"))?;
        let slot_stride = heads
            .checked_mul(head_stride)
            .ok_or_else(|| anyhow!("slot spill ({name}): slot stride overflow"))?;
        let extent = slot_stride
            .checked_mul(n_seqs)
            .ok_or_else(|| anyhow!("slot spill ({name}): buffer extent overflow"))?;
        anyhow::ensure!(
            extent <= buffer.byte_len(),
            "slot spill ({name}): shape covers {extent} bytes but the buffer holds {}",
            buffer.byte_len()
        );
        Ok(Self {
            heads,
            capacity,
            row_bytes,
            head_stride,
            slot_stride,
        })
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes of rows `0..rows` over every head of one slot.
    pub(crate) fn spilled_len(&self, rows: usize, name: &str) -> Result<usize> {
        anyhow::ensure!(
            rows <= self.capacity,
            "slot spill ({name}): cursor {rows} exceeds capacity {}",
            self.capacity
        );
        Ok(self.heads * rows * self.row_bytes)
    }

    fn head_start(&self, slot_idx: usize, head: usize) -> usize {
        slot_idx * self.slot_stride + head * self.head_stride
    }
}

/// Rows `0..rows` of every head of `slot_idx`, head-major.
pub(crate) fn read_rows(
    buffer: &MlxBuffer,
    slot_idx: usize,
    n_seqs: usize,
    rows: usize,
    name: &str,
) -> Result<Vec<u8>> {
    let layout = SlotRows::of(buffer, n_seqs, name)?;
    let mut out = Vec::with_capacity(layout.spilled_len(rows, name)?);
    let bytes = buffer
        .as_slice::<u8>()
        .with_context(|| format!("slot spill ({name}) as_slice"))?;
    let copy = rows * layout.row_bytes;
    for head in 0..layout.heads {
        let start = layout.head_start(slot_idx, head);
        out.extend_from_slice(&bytes[start..start + copy]);
    }
    Ok(out)
}

/// Write [`read_rows`] output for `rows` rows into `slot_idx`.
pub(crate) fn write_rows(
    src: &[u8],
    buffer: &mut MlxBuffer,
    slot_idx: usize,
    n_seqs: usize,
    rows: usize,
    name: &str,
) -> Result<()> {
    let layout = SlotRows::of(buffer, n_seqs, name)?;
    anyhow::ensure!(
        src.len() == layout.spilled_len(rows, name)?,
        "slot spill ({name}): spilled {} bytes for {rows} rows, expected {}",
        src.len(),
        layout.heads * rows * layout.row_bytes
    );
    let copy = rows * layout.row_bytes;
    if copy == 0 {
        return Ok(());
    }
    let bytes = buffer
        .as_mut_slice::<u8>()
        .with_context(|| format!("slot spill ({name}) as_mut_slice"))?;
    for (head, chunk) in src.chunks_exact(copy).enumerate() {
        let start = layout.head_start(slot_idx, head);
        bytes[start..start + copy].copy_from_slice(chunk);
    }
    Ok(())
}

/// Copy rows `0..rows` of every head from `src_slot` to `dst_slot` inside
/// one buffer, without a host intermediate.
pub(crate) fn copy_rows_between_slots(
    buffer: &mut MlxBuffer,
    src_slot: usize,
    dst_slot: usize,
    n_seqs: usize,
    rows: usize,
    name: &str,
) -> Result<()> {
    let layout = SlotRows::of(buffer, n_seqs, name)?;
    layout.spilled_len(rows, name)?;
    anyhow::ensure!(
        src_slot < n_seqs && dst_slot < n_seqs,
        "slot spill ({name}): slots {src_slot}->{dst_slot} outside n_seqs={n_seqs}"
    );
    let copy = rows * layout.row_bytes;
    if copy == 0 || src_slot == dst_slot {
        return Ok(());
    }
    let bytes = buffer
        .as_mut_slice::<u8>()
        .with_context(|| format!("slot spill ({name}) as_mut_slice"))?;
    for head in 0..layout.heads {
        let src = layout.head_start(src_slot, head);
        bytes.copy_within(src..src + copy, layout.head_start(dst_slot, head));
    }
    Ok(())
}
