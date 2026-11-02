//! Existing Gemma graph-stage localizer. A selected stage introduces a checked
//! terminal boundary and continues on the SAME executor/device/queue. Its
//! receipt must be labelled diagnostic split, never grouped-serving parity.
use super::{
    family,
    receipt::{Family, Stage},
};
use anyhow::Result;
use mlx_native::{graph::GraphSession, GraphExecutor, MlxBuffer};
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
    // Allocate the replacement encoder before submitting. If creation fails,
    // the caller retains the old session and propagates the error normally.
    let replacement = exec.begin()?;
    let completed = std::mem::replace(session, replacement);
    completed.finish()?;
    for (stage, buffer, shape) in buffers {
        let elements = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| anyhow::anyhow!("Gemma active extent overflow"))?;
        let bytes = elements
            .checked_mul(buffer.dtype().size_of())
            .ok_or_else(|| anyhow::anyhow!("Gemma active bytes overflow"))?;
        anyhow::ensure!(
            bytes <= buffer.data_byte_len(),
            "Gemma active capture exceeds source scratch"
        );
        let active = buffer.slice_view(0, elements).with_shape(shape.clone())?;
        family::drained(
            Family::Gemma4,
            position as u64,
            layer as u32,
            *stage,
            &active,
            "Gemma selected-stage diagnostic split: same GraphExecutor checked finish",
        )
        .map_err(anyhow::Error::msg)?;
    }
    Ok(())
}
