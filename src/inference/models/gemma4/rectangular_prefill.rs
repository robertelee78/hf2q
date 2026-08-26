use crate::serve::multi_seq_kv::SlotId;
use anyhow::{ensure, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};
use std::collections::BTreeSet;

use crate::quantize::imatrix::ImatrixHint;
use crate::serve::forward_mlx_shared::{dispatch_qmatmul, MlxQWeight};

const PROVEN_RECTANGULAR_ROWS_PER_LANE: usize = 32;

fn is_proven_rectangular_shape(lanes: usize, rows_per_lane: usize) -> bool {
    matches!(lanes, 2 | 4) && rows_per_lane == PROVEN_RECTANGULAR_ROWS_PER_LANE
}

/// Physical shape admitted by Gemma's equality-preserving multi-slot prefill.
///
/// `rows_per_lane` remains the scalar operator width. Callers may flatten the
/// storage to `lanes * rows_per_lane` only for operators already proven
/// independent per row; shape-sensitive kernels retain the explicit lane axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RectangularPrefillShape {
    pub(crate) lanes: usize,
    pub(crate) rows_per_lane: usize,
    pub(crate) start_position: usize,
}

pub(crate) fn validate_rectangular_prefill_layout(
    seq_lens: &[usize],
    seq_offsets: &[usize],
    start_positions: &[usize],
    slot_ids: &[SlotId],
) -> Result<RectangularPrefillShape> {
    let lanes = seq_lens.len();
    ensure!(
        lanes >= 2,
        "rectangular Gemma prefill requires at least two lanes"
    );
    ensure!(
        seq_offsets.len() == lanes
            && start_positions.len() == lanes
            && slot_ids.len() == lanes,
        "rectangular Gemma prefill descriptor lengths disagree: lens={lanes} offsets={} starts={} slots={}",
        seq_offsets.len(),
        start_positions.len(),
        slot_ids.len()
    );

    let rows_per_lane = seq_lens[0];
    ensure!(rows_per_lane > 0, "rectangular Gemma prefill has zero rows");
    ensure!(
        seq_lens.iter().all(|&rows| rows == rows_per_lane),
        "rectangular Gemma prefill requires equal lane widths: {seq_lens:?}"
    );
    for (lane, &offset) in seq_offsets.iter().enumerate() {
        let expected = lane
            .checked_mul(rows_per_lane)
            .context("rectangular Gemma prefill offset overflow")?;
        ensure!(
            offset == expected,
            "rectangular Gemma prefill lane {lane} offset {offset} != {expected}"
        );
    }

    let start_position = start_positions[0];
    ensure!(
        start_positions.iter().all(|&start| start == start_position),
        "rectangular Gemma prefill requires equal logical starts: {start_positions:?}"
    );
    let distinct_slots = slot_ids.iter().map(|slot| slot.0).collect::<BTreeSet<_>>();
    ensure!(
        distinct_slots.len() == lanes,
        "rectangular Gemma prefill requires distinct physical slots: {slot_ids:?}"
    );

    Ok(RectangularPrefillShape {
        lanes,
        rows_per_lane,
        start_position,
    })
}

/// Classify a live multi-sequence descriptor without weakening validation.
///
/// Only B2/B4 with M32 is a proven production geometry. Other lane counts,
/// widths, unequal lane widths, or unequal logical starts retain the existing
/// aggregate route. Once a descriptor claims proven rectangular eligibility,
/// every remaining structural invariant is mandatory: malformed offsets,
/// descriptor extents, or duplicate slots are propagated as errors instead of
/// silently falling back to a known shape-sensitive aggregate projection.
pub(crate) fn rectangular_prefill_shape_if_eligible(
    seq_lens: &[usize],
    seq_offsets: &[usize],
    start_positions: &[usize],
    slot_ids: &[SlotId],
) -> Result<Option<RectangularPrefillShape>> {
    if seq_lens.len() < 2 {
        return Ok(None);
    }
    let rows_per_lane = seq_lens[0];
    if !is_proven_rectangular_shape(seq_lens.len(), rows_per_lane) {
        return Ok(None);
    }
    let Some(&start_position) = start_positions.first() else {
        return validate_rectangular_prefill_layout(
            seq_lens,
            seq_offsets,
            start_positions,
            slot_ids,
        )
        .map(Some);
    };
    if !seq_lens.iter().all(|&rows| rows == rows_per_lane)
        || !start_positions.iter().all(|&start| start == start_position)
    {
        return Ok(None);
    }
    validate_rectangular_prefill_layout(seq_lens, seq_offsets, start_positions, slot_ids).map(Some)
}

/// Return one lane of a logical rectangular F32 matrix without copying it.
///
/// The parent may be an oversized scratch allocation (Gemma's attention
/// scratch is sized for the largest head width across all layers), but it must
/// contain the complete active rectangle. The returned view names exactly
/// `rows_per_lane * cols` F32 elements and retains the parent's Metal storage
/// plus its existing byte offset.
pub(crate) fn checked_f32_lane_view(
    buffer: &MlxBuffer,
    shape: RectangularPrefillShape,
    cols: usize,
    lane: usize,
    operation: &str,
) -> Result<MlxBuffer> {
    ensure!(cols > 0, "{operation}: rectangular matrix has zero columns");
    ensure!(
        lane < shape.lanes,
        "{operation}: lane {lane} out of range for {} lanes",
        shape.lanes
    );
    ensure!(
        buffer.dtype() == DType::F32,
        "{operation}: rectangular lane slicing requires F32 storage, got {:?}",
        buffer.dtype()
    );

    let lane_elements = shape
        .rows_per_lane
        .checked_mul(cols)
        .context("rectangular lane element count overflow")?;
    let active_elements = shape
        .lanes
        .checked_mul(lane_elements)
        .context("rectangular matrix element count overflow")?;
    let active_bytes = active_elements
        .checked_mul(std::mem::size_of::<f32>())
        .context("rectangular matrix byte extent overflow")?;
    ensure!(
        buffer.element_count() >= active_elements && buffer.data_byte_len() >= active_bytes,
        "{operation}: active F32 rectangle needs {active_elements} elements/{active_bytes} bytes, got {} elements/{} bytes",
        buffer.element_count(),
        buffer.data_byte_len()
    );

    let lane_bytes = lane_elements
        .checked_mul(std::mem::size_of::<f32>())
        .context("rectangular lane byte extent overflow")?;
    let relative_offset = lane
        .checked_mul(lane_bytes)
        .context("rectangular lane byte offset overflow")?;
    Ok(buffer.slice_view(
        u64::try_from(relative_offset).context("rectangular lane offset does not fit u64")?,
        lane_elements,
    ))
}

/// Execute a rectangular projection with the exact scalar per-sequence
/// operator width while retaining a single graph session.
///
/// This entry point independently enforces the proven B2/B4 M32 contract so a
/// future caller cannot bypass admission by constructing a shape directly.
///
/// Each lane aliases its row-contiguous region of the shared activation
/// buffers. Calling the canonical dispatcher with `m = rows_per_lane` keeps
/// kernel selection and reduction order identical to one scalar sequence;
/// only command-buffer ownership is shared across lanes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_rectangular_qmatmul(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxQWeight,
    output: &MlxBuffer,
    shape: RectangularPrefillShape,
    input_cols: usize,
    output_cols: usize,
    imatrix_hint: ImatrixHint<'_>,
) -> Result<()> {
    ensure!(
        is_proven_rectangular_shape(shape.lanes, shape.rows_per_lane),
        "rectangular projection shape B{}xM{} is outside the proven B2/B4 M32 contract",
        shape.lanes,
        shape.rows_per_lane
    );
    ensure!(
        weight.info.cols == input_cols && weight.info.rows == output_cols,
        "rectangular projection shape mismatch: weight=[{},{}], input_cols={input_cols}, output_cols={output_cols}",
        weight.info.rows,
        weight.info.cols
    );
    let scalar_m = u32::try_from(shape.rows_per_lane)
        .context("rectangular scalar row count does not fit u32")?;
    for lane in 0..shape.lanes {
        let lane_input = checked_f32_lane_view(
            input,
            shape,
            input_cols,
            lane,
            "rectangular projection input",
        )?;
        let lane_output = checked_f32_lane_view(
            output,
            shape,
            output_cols,
            lane,
            "rectangular projection output",
        )?;
        dispatch_qmatmul(
            session,
            registry,
            device,
            &lane_input,
            weight,
            &lane_output,
            scalar_m,
            imatrix_hint,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_equal_rows_equal_starts_and_noncontiguous_distinct_slots() {
        assert_eq!(
            validate_rectangular_prefill_layout(
                &[32, 32, 32, 32],
                &[0, 32, 64, 96],
                &[64, 64, 64, 64],
                &[SlotId(6), SlotId(1), SlotId(7), SlotId(3)],
            )
            .expect("valid rectangle"),
            RectangularPrefillShape {
                lanes: 4,
                rows_per_lane: 32,
                start_position: 64,
            }
        );
    }

    #[test]
    fn rejects_every_shape_or_slot_ambiguity() {
        let cases = [
            (vec![32], vec![0], vec![64], vec![SlotId(0)]),
            (
                vec![32, 31],
                vec![0, 32],
                vec![64, 64],
                vec![SlotId(0), SlotId(1)],
            ),
            (
                vec![32, 32],
                vec![0, 31],
                vec![64, 64],
                vec![SlotId(0), SlotId(1)],
            ),
            (
                vec![32, 32],
                vec![0, 32],
                vec![64, 65],
                vec![SlotId(0), SlotId(1)],
            ),
            (
                vec![32, 32],
                vec![0, 32],
                vec![64, 64],
                vec![SlotId(0), SlotId(0)],
            ),
        ];
        for (lens, offsets, starts, slots) in cases {
            assert!(
                validate_rectangular_prefill_layout(&lens, &offsets, &starts, &slots).is_err(),
                "ambiguous rectangle was admitted: lens={lens:?} offsets={offsets:?} starts={starts:?} slots={slots:?}"
            );
        }
    }

    #[test]
    fn eligibility_distinguishes_nonrectangles_from_malformed_rectangles() {
        assert_eq!(
            rectangular_prefill_shape_if_eligible(
                &[32, 32],
                &[0, 32],
                &[64, 64],
                &[SlotId(3), SlotId(7)],
            )
            .expect("valid rectangle"),
            Some(RectangularPrefillShape {
                lanes: 2,
                rows_per_lane: 32,
                start_position: 64,
            })
        );
        assert_eq!(
            rectangular_prefill_shape_if_eligible(
                &[32, 31],
                &[0, 32],
                &[64, 64],
                &[SlotId(0), SlotId(1)],
            )
            .expect("unequal widths are a supported nonrectangle"),
            None
        );
        assert_eq!(
            rectangular_prefill_shape_if_eligible(
                &[32, 32],
                &[0, 32],
                &[64, 65],
                &[SlotId(0), SlotId(1)],
            )
            .expect("unequal starts are a supported nonrectangle"),
            None
        );
        for (offsets, slots) in [
            (vec![0, 31], vec![SlotId(0), SlotId(1)]),
            (vec![0, 32], vec![SlotId(0), SlotId(0)]),
        ] {
            assert!(
                rectangular_prefill_shape_if_eligible(&[32, 32], &offsets, &[64, 64], &slots,)
                    .is_err(),
                "malformed admitted rectangle fell back: offsets={offsets:?} slots={slots:?}"
            );
        }
    }

    #[test]
    fn eligibility_only_admits_proven_b2_b4_m32_shapes() {
        for lanes in [2usize, 4] {
            let seq_lens = vec![32; lanes];
            let seq_offsets = (0..lanes).map(|lane| lane * 32).collect::<Vec<_>>();
            let start_positions = vec![96; lanes];
            let slot_ids = (0..lanes)
                .map(|lane| SlotId((lane * 3 + 1) as u32))
                .collect::<Vec<_>>();
            assert_eq!(
                rectangular_prefill_shape_if_eligible(
                    &seq_lens,
                    &seq_offsets,
                    &start_positions,
                    &slot_ids,
                )
                .expect("proven rectangle"),
                Some(RectangularPrefillShape {
                    lanes,
                    rows_per_lane: 32,
                    start_position: 96,
                })
            );
        }

        for (lanes, rows) in [(3usize, 32usize), (8, 32), (2, 1), (2, 31), (4, 64)] {
            let seq_lens = vec![rows; lanes];
            let seq_offsets = (0..lanes).map(|lane| lane * rows).collect::<Vec<_>>();
            let start_positions = vec![96; lanes];
            let slot_ids = (0..lanes)
                .map(|lane| SlotId(lane as u32))
                .collect::<Vec<_>>();
            assert_eq!(
                rectangular_prefill_shape_if_eligible(
                    &seq_lens,
                    &seq_offsets,
                    &start_positions,
                    &slot_ids,
                )
                .expect("unsupported equal rectangle must fall back"),
                None,
                "unexpected admission for B{lanes} M{rows}"
            );
        }

        for (offsets, slots) in [
            (
                vec![0, 32, 64, 95],
                vec![SlotId(0), SlotId(1), SlotId(2), SlotId(3)],
            ),
            (
                vec![0, 32, 64, 96],
                vec![SlotId(0), SlotId(1), SlotId(2), SlotId(2)],
            ),
        ] {
            assert!(
                rectangular_prefill_shape_if_eligible(
                    &[32, 32, 32, 32],
                    &offsets,
                    &[96, 96, 96, 96],
                    &slots,
                )
                .is_err(),
                "malformed admitted B4 M32 rectangle fell back: offsets={offsets:?} slots={slots:?}"
            );
        }
    }

    #[test]
    fn f32_lane_views_are_exact_zero_copy_aliases() {
        let Some(device) = MlxDevice::new().ok() else {
            eprintln!("skipping Metal-only rectangular alias test");
            return;
        };
        let shape = RectangularPrefillShape {
            lanes: 3,
            rows_per_lane: 2,
            start_position: 64,
        };
        let cols = 5;
        let elements = shape.lanes * shape.rows_per_lane * cols;
        let mut parent = device
            .alloc_buffer(
                elements * DType::F32.size_of(),
                DType::F32,
                vec![shape.lanes * shape.rows_per_lane, cols],
            )
            .expect("F32 parent");
        parent
            .as_mut_slice::<f32>()
            .expect("writable parent")
            .iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = index as f32);

        let lane = checked_f32_lane_view(&parent, shape, cols, 1, "alias test").expect("lane view");
        assert_eq!(lane.contents_ptr(), parent.contents_ptr());
        assert_eq!(
            lane.byte_offset(),
            (shape.rows_per_lane * cols * DType::F32.size_of()) as u64
        );
        assert_eq!(
            lane.data_byte_len(),
            shape.rows_per_lane * cols * DType::F32.size_of()
        );
        assert_eq!(
            lane.as_slice::<f32>().expect("typed lane"),
            &(10..20).map(|value| value as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn f32_lane_views_reject_bad_dtype_extent_and_lane() {
        let Some(device) = MlxDevice::new().ok() else {
            eprintln!("skipping Metal-only rectangular extent test");
            return;
        };
        let shape = RectangularPrefillShape {
            lanes: 2,
            rows_per_lane: 3,
            start_position: 64,
        };
        let short = device
            .alloc_buffer(23 * DType::F32.size_of(), DType::F32, vec![23])
            .expect("short F32 buffer");
        let wrong_dtype = device
            .alloc_buffer(24 * DType::BF16.size_of(), DType::BF16, vec![24])
            .expect("BF16 buffer");
        assert!(checked_f32_lane_view(&short, shape, 4, 0, "short").is_err());
        assert!(checked_f32_lane_view(&wrong_dtype, shape, 4, 0, "dtype").is_err());
        assert!(checked_f32_lane_view(&short, shape, 4, 2, "lane").is_err());
    }
}
