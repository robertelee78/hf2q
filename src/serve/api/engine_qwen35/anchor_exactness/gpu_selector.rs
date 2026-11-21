//! Test-only selector input to the REAL GPU acceptance/commit tail. The target
//! graph, lm_head, actual argmax buffer and readback remain untouched. This is
//! never enabled by a production build or by an unforced lifecycle Scope.
use super::*;
use mlx_native::{CommandEncoder, KernelRegistry, MlxDevice};

pub(crate) struct SelectorOwners {
    pub(crate) selected: MlxBuffer,
    _frozen: MlxBuffer,
}
pub(crate) fn selector(
    encoder: &mut CommandEncoder, _registry: &mut KernelRegistry, device: &MlxDevice,
    actual: &MlxBuffer, rows: &[(SlotId, usize)],
) -> Result<Option<SelectorOwners>> {
    let mut frozen = ACTIVE.with(|cell| {
        let active = cell.borrow();
        let Some(active) = active.as_ref() else { return Vec::new() };
        let Some(frozen) = active.frozen.as_ref() else { return Vec::new() };
        rows.iter().enumerate().filter_map(|(row, (slot, position))| {
            if *slot != active.target_slot { return None; }
            let index = position.checked_sub(active.prediction_origin)?;
            frozen.get(index).map(|token| (row, *token))
        }).collect::<Vec<_>>()
    });
    if frozen.is_empty() {
        for (row, (slot, position)) in rows.iter().enumerate() {
            if let Some(token) = super::super::fullw_http::gpu_token(*slot, *position)? {
                frozen.push((row, token));
            }
        }
    }
    if frozen.is_empty() { return Ok(None); }
    ensure!(actual.dtype() == DType::U32 && actual.element_count() >= rows.len(),
        "frozen GPU selector requires actual u32 argmax rows");
    let mut values = device.alloc_buffer(frozen.len() * 4, DType::U32, vec![frozen.len()])?;
    for (out, (_, token)) in values.as_mut_slice::<u32>()?.iter_mut().zip(&frozen) { *out = *token; }
    let selected = device.alloc_buffer(actual.data_byte_len(), DType::U32, actual.shape().to_vec())?;
    // Commands, not host reads of an in-flight argmax. The ordinary argmax is
    // preserved in `actual`; only the diagnostic tail receives `selected`.
    encoder.blit_copy_bytes(actual, 0, &selected, 0, actual.data_byte_len() as u64)?;
    for (index, (row, _)) in frozen.iter().enumerate() {
        encoder.blit_copy_bytes(&values, (index * 4) as u64, &selected, (row * 4) as u64, 4)?;
        super::super::fullw_http::gpu_selected(rows[*row].0);
    }
    ACTIVE.with(|cell| {
        if let Some(active) = cell.borrow_mut().as_mut() {
            active.observations.frozen_gpu_selector_used = true;
        }
    });
    Ok(Some(SelectorOwners { selected, _frozen: values }))
}
pub(crate) fn set_prediction_origin(origin: usize) {
    ACTIVE.with(|cell| { if let Some(active) = cell.borrow_mut().as_mut() { active.prediction_origin = origin; } });
}
pub(crate) fn frozen_selector_digest() -> Option<String> {
    ACTIVE.with(|cell| cell.borrow().as_ref().and_then(|active| active.frozen.as_ref().map(|frozen|
        hash(&serde_json::to_vec(&(active.prediction_origin, frozen)).expect("finite frozen selector")))))
}
