use crate::serve::multi_seq_kv::SlotId;
use anyhow::{ensure, Context, Result};
use std::collections::BTreeSet;

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
}
