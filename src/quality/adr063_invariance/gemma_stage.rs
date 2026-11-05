//! Existing Gemma graph-stage localizer. A selected stage introduces a checked
//! terminal boundary and continues on the SAME executor/device/queue. Its
//! receipt must be labelled diagnostic split, never grouped-serving parity.
//!
//! [`capture`] serves the single-row decode body; under a
//! `gemma_prefill::serial_lane` lease its row belongs to that lane.
//! [`capture_batched`] serves the `[N, ..]` batched decode body
//! (`batched_body.rs`): row `i` belongs to `slot_ids[i]` at `positions[i]`.
use super::{
    family,
    gemma_prefill::current_serial_lane,
    receipt::{Family, Stage},
};
use anyhow::Result;
use mlx_native::{graph::GraphSession, GraphExecutor, MlxBuffer};
/// Finish the current session on the same executor and continue in a fresh
/// one. Allocate the replacement encoder before submitting: if creation fails,
/// the caller retains the old session and propagates the error normally.
fn checked_split<'a>(session: &mut GraphSession<'a>, exec: &'a GraphExecutor) -> Result<()> {
    let replacement = exec.begin().map(crate::inference::native_observation::session)?;
    let completed = std::mem::replace(session, replacement);
    completed.finish()?;
    Ok(())
}
fn active_view(buffer: &MlxBuffer, byte_offset: u64, shape: &[usize]) -> Result<MlxBuffer> {
    let elements = shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| anyhow::anyhow!("Gemma active extent overflow"))?;
    let end = elements
        .checked_mul(buffer.dtype().size_of())
        .and_then(|bytes| (bytes as u64).checked_add(byte_offset))
        .ok_or_else(|| anyhow::anyhow!("Gemma active bytes overflow"))?;
    anyhow::ensure!(
        end <= buffer.data_byte_len() as u64,
        "Gemma active capture exceeds source scratch"
    );
    Ok(buffer.slice_view(byte_offset, elements).with_shape(shape.to_vec())?)
}
pub fn capture<'a>(
    session: &mut GraphSession<'a>,
    exec: &'a GraphExecutor,
    position: usize,
    layer: usize,
    buffers: &[(Stage, &MlxBuffer, Vec<usize>)],
) -> Result<()> {
    if !buffers.iter().any(|(stage, _, _)| {
        family::selected(Family::Gemma4, position as u64, layer as u32, *stage)
    }) {
        return Ok(());
    }
    checked_split(session, exec)?;
    let drain = "Gemma selected-stage diagnostic split: same GraphExecutor checked finish";
    let lane = current_serial_lane();
    for (stage, buffer, shape) in buffers {
        let active = active_view(buffer, 0, shape)?;
        match lane {
            Some(lane) => family::drained_owned_rows(Family::Gemma4, position as u64, layer as u32,
                *stage, &active, &[(lane, position)], drain),
            None => family::drained(Family::Gemma4, position as u64, layer as u32, *stage, &active, drain),
        }
        .map_err(anyhow::Error::msg)?;
    }
    Ok(())
}
/// One batched-decode role: `(stage, [N, stride] buffer, row stride in
/// elements, logical row shape)`. Row `i` is read at `i * stride`.
pub type BatchedRole<'b> = (Stage, &'b MlxBuffer, usize, Vec<usize>);
/// Batched-decode stage capture. Each selected row is labelled by its own
/// slot (the Q1 cohort binds slot `i` to workload lane `i`) and position; it
/// is refused inside a serial-lane lease, whose single lane cannot own N rows.
pub fn capture_batched<'a>(
    session: &mut GraphSession<'a>,
    exec: &'a GraphExecutor,
    layer: usize,
    slot_ids: &[crate::serve::multi_seq_kv::SlotId],
    positions: &[usize],
    roles: &[BatchedRole<'_>],
) -> Result<()> {
    let selected = |stage: Stage, row: usize| {
        family::selected(Family::Gemma4, positions[row] as u64, layer as u32, stage)
    };
    anyhow::ensure!(slot_ids.len() == positions.len(), "batched decode rows/slots differ");
    if !roles
        .iter()
        .any(|(stage, ..)| (0..positions.len()).any(|row| selected(*stage, row)))
    {
        return Ok(());
    }
    anyhow::ensure!(
        current_serial_lane().is_none(),
        "batched Gemma decode rows inside a serial-lane scope"
    );
    checked_split(session, exec)?;
    for (stage, buffer, stride, shape) in roles {
        let row_elements: usize = shape.iter().product();
        anyhow::ensure!(row_elements <= *stride, "Gemma batched role row exceeds its stride");
        let mut row_shape = vec![1];
        row_shape.extend_from_slice(shape);
        for (row, (&slot, &position)) in slot_ids.iter().zip(positions).enumerate() {
            if !selected(*stage, row) {
                continue;
            }
            let offset = row
                .checked_mul(*stride)
                .and_then(|n| n.checked_mul(buffer.dtype().size_of()))
                .ok_or_else(|| anyhow::anyhow!("Gemma batched row offset overflow"))?;
            let active = active_view(buffer, offset as u64, &row_shape)?;
            family::drained_owned_rows(
                Family::Gemma4,
                position as u64,
                layer as u32,
                *stage,
                &active,
                &[(slot.0 as usize, position)],
                "Gemma batched-decode selected-stage diagnostic split: same GraphExecutor checked finish",
            )
            .map_err(anyhow::Error::msg)?;
        }
    }
    Ok(())
}
