//! Test-only Q4-GEMMA-DECODE A1 router snapshots, adapted from #699.
//! Copies use the real step encoder; readback follows its existing final wait.

use std::cell::{Cell, RefCell};

use anyhow::Result;
use mlx_native::{DType, GraphSession, MlxBuffer, MlxDevice};

pub(crate) type Buffers = (MlxBuffer, MlxBuffer);

pub(crate) struct Step {
    pub n: usize,
    pub stride: usize,
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
}

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static LAST: RefCell<Option<Step>> = const { RefCell::new(None) };
}

pub(crate) fn arm(on: bool) {
    ARMED.with(|value| value.set(on));
    LAST.with(|value| value.borrow_mut().take());
}

pub(crate) fn take() -> Option<Step> {
    LAST.with(|value| value.borrow_mut().take())
}

pub(crate) fn allocate(
    device: &MlxDevice,
    layers: usize,
    n: usize,
    stride: usize,
) -> Result<Option<Buffers>> {
    if !ARMED.with(Cell::get) {
        return Ok(None);
    }
    let count = layers * n * stride;
    Ok(Some((
        device.alloc_buffer(count * 4, DType::U32, vec![count])?,
        device.alloc_buffer(count * 4, DType::F32, vec![count])?,
    )))
}

pub(crate) fn record(
    session: &mut GraphSession<'_>,
    buffers: &Option<Buffers>,
    ids: &MlxBuffer,
    weights: &MlxBuffer,
    layer: usize,
    elements: usize,
) -> Result<()> {
    if let Some((ids_out, weights_out)) = buffers {
        let bytes = (elements * 4) as u64;
        let offset = layer as u64 * bytes;
        let encoder = session.encoder_mut();
        encoder.blit_copy_bytes(ids, 0, ids_out, offset, bytes)?;
        encoder.blit_copy_bytes(weights, 0, weights_out, offset, bytes)?;
    }
    Ok(())
}

/// Called only after the entire actual step is known complete, including the
/// asynchronous command-buffer chunks preceding its final same-queue wait.
pub(crate) fn publish(buffers: &Option<Buffers>, n: usize, stride: usize) -> Result<()> {
    if let Some((ids, weights)) = buffers {
        let step = Step {
            n,
            stride,
            ids: ids.as_slice::<u32>()?.to_vec(),
            weights: weights.as_slice::<f32>()?.to_vec(),
        };
        LAST.with(|value| *value.borrow_mut() = Some(step));
    }
    Ok(())
}
