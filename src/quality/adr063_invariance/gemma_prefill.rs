//! Whole-row Gemma prefill localizer reads at the EXISTING per-layer terminal.
//! Source keeps these roles distinct until the next layer. Pre-attention norm
//! is deliberately absent here: pf_norm_out was overwritten in session B.
//!
//! Row owners: a packed multi-sequence prefill (Q1 T2, 512 rows at
//! `lane*128`) maps every row through `gemma_endpoint::packed_owners`; a
//! serial four-lane localizer names the lane of the current single-sequence
//! call with [`serial_lane`]. Neither present = the fixed-lane localizer.
use super::{
    family,
    gemma_endpoint::{packed_owners, PackedLayout},
    receipt::{Family, Stage},
};
use anyhow::{ensure, Result};
use mlx_native::MlxBuffer;
use std::cell::Cell;
pub(crate) fn require_checked_terminal(active: bool, checked: bool) -> Result<()> {
    ensure!(
        !active || checked,
        "Gemma Q1 capture refused: this branch did not return from a checked layer terminal"
    );
    Ok(())
}
thread_local! {static SERIAL_LANE:Cell<Option<usize>>=const{Cell::new(None)};}
/// Same-thread lease naming the workload lane of every single-sequence Gemma
/// prefill/decode call until dropped. Not nestable.
#[must_use]
pub struct SerialLane {
    _same_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for SerialLane {
    fn drop(&mut self) {
        SERIAL_LANE.with(|l| l.set(None));
    }
}
pub fn serial_lane(lane: usize) -> Result<SerialLane> {
    ensure!(lane < 4, "serial localizer lane outside the four workload lanes");
    SERIAL_LANE.with(|l| {
        ensure!(l.get().is_none(), "nested serial localizer lane");
        l.set(Some(lane));
        Ok(SerialLane { _same_thread: std::marker::PhantomData })
    })
}
pub(crate) fn current_serial_lane() -> Option<usize> {
    SERIAL_LANE.with(|l| l.get())
}
/// `(lane, absolute position)` of each of `rows` rows starting at `position`,
/// or `None` for the fixed-lane route.
pub(crate) fn row_owners(
    packed: Option<&PackedLayout>,
    serial: Option<usize>,
    position: usize,
    rows: usize,
) -> Result<Option<Vec<(usize, usize)>>> {
    match (packed, serial) {
        (Some(_), Some(_)) => anyhow::bail!("packed Gemma prefill inside a serial-lane scope"),
        (Some(layout), None) => {
            let owners = packed_owners(layout).map_err(anyhow::Error::msg)?;
            ensure!(
                position == 0 && owners.len() == rows,
                "packed Gemma prefill rows differ from the four-sequence owner map"
            );
            Ok(Some(owners))
        }
        (None, Some(lane)) => Ok(Some((position..position + rows).map(|p| (lane, p)).collect())),
        (None, None) => Ok(None),
    }
}
pub fn after_layer(
    checked_terminal: bool,
    position: usize,
    layer: usize,
    buffers: &[(Stage, &MlxBuffer, Vec<usize>)],
    packed: Option<&PackedLayout>,
) -> Result<()> {
    let active = family::enabled(Family::Gemma4);
    require_checked_terminal(active, checked_terminal)?;
    if !active {
        return Ok(());
    }
    let mut owners = None;
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
        if owners.is_none() {
            owners = Some(row_owners(packed, current_serial_lane(), position, shape[0])?);
        }
        let drain = "Gemma actual batched per-layer checked finish, diagnostic sync, before next-layer reuse";
        match owners.as_ref().and_then(Option::as_ref) {
            Some(rows) => family::drained_owned_rows(Family::Gemma4, position as u64, layer as u32, *stage, &view, rows, drain),
            None => family::drained(Family::Gemma4, position as u64, layer as u32, *stage, &view, drain),
        }
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
    #[test]
    fn q1_gemma_prefill_row_owners_map_packed_rows_and_serial_lanes() {
        let layout: PackedLayout = (0..4).map(|lane| (lane as u32, 0, 128, lane * 128)).collect();
        let packed = row_owners(Some(&layout), None, 0, 512).unwrap().unwrap();
        assert_eq!(packed.len(), 512);
        assert_eq!((packed[0], packed[127], packed[128], packed[383], packed[511]),
            ((0, 0), (0, 127), (1, 0), (2, 127), (3, 127)));
        // A packed stream must start at 0, cover exactly 512 rows, and never
        // run inside a serial-lane scope.
        assert!(row_owners(Some(&layout), None, 0, 511).is_err());
        assert!(row_owners(Some(&layout), None, 64, 512).is_err());
        assert!(row_owners(Some(&layout), Some(1), 0, 512).is_err());
        assert!(row_owners(Some(&vec![(0, 0, 512, 0)]), None, 0, 512).is_err());
        assert_eq!(row_owners(None, Some(2), 64, 3).unwrap().unwrap(), vec![(2, 64), (2, 65), (2, 66)]);
        assert_eq!(row_owners(None, None, 0, 128).unwrap(), None);
    }
    #[test]
    fn q1_gemma_serial_lane_lease_is_bounded_and_not_nestable() {
        assert!(serial_lane(4).is_err());
        {
            let _lane = serial_lane(3).unwrap();
            assert_eq!(current_serial_lane(), Some(3));
            assert!(serial_lane(0).is_err());
        }
        assert_eq!(current_serial_lane(), None);
    }
}
