//! Actual-buffer adapter boundary. The caller must provide a checked drain;
//! retaining a wrapper, queue ordering, or a submission label is insufficient.
use super::{
    receipt::{Coordinate, Scalar},
    store::Store,
};
use mlx_native::{DType, MlxBuffer};

/// Copy the COMPLETE logical CPU-visible extent after the owning execution
/// path's successful terminal drain. Physical padding is deliberately excluded.
/// This function never submits work and never treats a last-row slice as its
/// parent's full active extent. The required coordinate/shape packet is checked
/// by Store independently of the allocation's metadata.
pub fn after_checked_drain(
    store: &mut Store,
    at: Coordinate,
    buffer: &MlxBuffer,
    expected_shape: &[usize],
    submission_id: Option<u64>,
    checked_drain: &str,
) -> Result<(), String> {
    if buffer.shape() != expected_shape {
        return Err("actual logical shape differs from capture packet".into());
    }
    let scalar = match buffer.dtype() {
        DType::F32 => Scalar::F32,
        DType::F16 => Scalar::F16,
        DType::BF16 => Scalar::Bf16,
        DType::I32 => Scalar::I32,
        DType::U32 => Scalar::U32,
        DType::U8 => Scalar::U8,
        _ => return Err("unsupported diagnostic scalar".into()),
    };
    let bytes = buffer.as_logical_slice::<u8>().map_err(|e| e.to_string())?;
    store.capture(
        at,
        scalar,
        expected_shape.to_vec(),
        buffer.contents_ptr() as usize as u64,
        buffer.byte_len() as u64,
        buffer.byte_offset() as u64,
        submission_id,
        checked_drain,
        bytes,
    )
}

/// Normalize a complete matrix transaction into logical rows. All rows are
/// visited; this is not a sample or a last-row view. Caller supplies the same
/// logical base for chunked and unchunked prefill packets.
pub fn rows_after_checked_drain(
    store: &mut Store,
    base: Coordinate,
    buffer: &MlxBuffer,
    submission_id: Option<u64>,
    checked_drain: &str,
) -> Result<(), String> {
    let shape = buffer.shape();
    if shape.is_empty() || shape[0] == 0 {
        return Err("empty row capture".into());
    }
    let row_elements = shape[1..]
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or("row extent overflow")?;
    let row_bytes = row_elements
        .checked_mul(buffer.dtype().size_of())
        .ok_or("row byte overflow")?;
    let all = buffer.as_logical_slice::<u8>().map_err(|e| e.to_string())?;
    if row_bytes.checked_mul(shape[0]) != Some(all.len()) {
        return Err("actual row extent mismatch".into());
    }
    let scalar = match buffer.dtype() {
        DType::F32 => Scalar::F32,
        DType::F16 => Scalar::F16,
        DType::BF16 => Scalar::Bf16,
        DType::I32 => Scalar::I32,
        DType::U32 => Scalar::U32,
        DType::U8 => Scalar::U8,
        _ => return Err("unsupported row scalar".into()),
    };
    let mut row_shape = shape.to_vec();
    row_shape[0] = 1;
    for (row, bytes) in all.chunks_exact(row_bytes).enumerate() {
        let mut at = base.clone();
        at.logical_row_start = at
            .logical_row_start
            .checked_add(row as u64)
            .ok_or("logical row overflow")?;
        let offset = buffer
            .byte_offset()
            .checked_add((row * row_bytes) as u64)
            .ok_or("physical row overflow")?;
        store.capture(
            at,
            scalar,
            row_shape.clone(),
            buffer.contents_ptr() as usize as u64,
            buffer.byte_len() as u64,
            offset,
            submission_id,
            checked_drain,
            bytes,
        )?;
    }
    Ok(())
}
