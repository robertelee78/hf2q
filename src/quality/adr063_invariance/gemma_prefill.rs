//! Whole-row Gemma prefill localizer reads at the EXISTING per-layer terminal.
//! Source keeps these roles distinct until the next layer. Pre-attention norm
//! is deliberately absent here: pf_norm_out was overwritten in session B.
use super::{
    family,
    receipt::{Family, Stage},
};
use anyhow::{ensure, Result};
use mlx_native::MlxBuffer;
pub(crate) fn require_checked_terminal(active: bool, checked: bool) -> Result<()> {
    ensure!(
        !active || checked,
        "Gemma Q1 capture refused: this branch did not return from a checked layer terminal"
    );
    Ok(())
}
pub fn after_layer(
    checked_terminal: bool,
    position: usize,
    layer: usize,
    buffers: &[(Stage, &MlxBuffer, Vec<usize>)],
) -> Result<()> {
    let active = family::enabled(Family::Gemma4);
    require_checked_terminal(active, checked_terminal)?;
    if !active {
        return Ok(());
    }
    for (stage, buffer, shape) in buffers {
        if !family::selected(Family::Gemma4, position as u64, layer as u32, *stage) {
            continue;
        }
        let elements = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| anyhow::anyhow!("Gemma prefill active extent overflow"))?;
        let bytes = elements
            .checked_mul(buffer.dtype().size_of())
            .ok_or_else(|| anyhow::anyhow!("Gemma prefill active byte overflow"))?;
        ensure!(
            bytes <= buffer.data_byte_len(),
            "Gemma prefill active extent exceeds actual role"
        );
        let view = buffer.slice_view(0, elements).with_shape(shape.clone())?;
        family::drained(Family::Gemma4,position as u64,layer as u32,*stage,&view,
            "Gemma actual batched per-layer checked finish, diagnostic sync, before next-layer reuse")
            .map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

#[cfg(test)]
mod terminal_controls {
    use super::*;
    #[test]
    fn q1_gemma_prefill_capture_requires_owning_checked_terminal() {
        assert!(require_checked_terminal(true, true).is_ok());
        assert!(require_checked_terminal(true, false).is_err());
        assert!(require_checked_terminal(false, false).is_ok());
    }
}
