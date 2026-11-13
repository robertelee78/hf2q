//! Generation-cue fold for the bounded prompt prefill
//! (`HF2Q_QWEN_PREFILL_FOLD_CUE=1`).
//!
//! A served prompt ends in a short generation cue. The rows before it are the
//! stable prompt prefix, and the prompt cache keeps a checkpoint exactly
//! there: the slot's DeltaNet state, its cursors and the logits of the last
//! prefix row. The bounded prefill therefore stops at that boundary, takes
//! the checkpoint, and runs the cue as a second forward of a few rows, which
//! streams every layer's weights a second time.
//!
//! The fold runs the last content chunk and the cue as one forward and takes
//! the checkpoint inside it. Every operation whose output row depends on that
//! input row alone (norms, the prepared row-tile projections, the dense FFN,
//! the elementwise stages) runs once over all rows. The operations that mix
//! rows run as the two forwards would have run them:
//!
//! - DeltaNet convolution and recurrence are dispatched twice, first over the
//!   prefix rows from the slot state into the checkpoint image, then over the
//!   cue rows from the checkpoint image into the slot state.
//! - A full-attention layer and a MoE FFN layer are built twice, once per
//!   segment, with the row count, cursor and scratch each forward would have
//!   had.
//! - The output head projects the last prefix row and the last row.
//!
//! Each segment therefore runs the kernels, parameters and inputs of its own
//! forward, and the slot state, the checkpoint, the captured target layers and
//! both logit rows are bit-identical to the two-forward path
//! (`prefill_fold_tests`). That holds only while every shared product is row
//! exact, so the fold is taken only inside a routed
//! [`super::prefill_prepared::PrefillScope`] and only for models whose shared
//! products all take the prepared kernel ([`Qwen35Model::prefill_fold_exact`]).
//!
//! [`Qwen35Model::prefill_fold_exact`]: super::model::Qwen35Model

use std::cell::RefCell;

use anyhow::{anyhow, ensure, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::joint_step_options::parse_switch;
use super::kv_cache::LaAnchorImage;
use crate::inference::spec_decode::dflash2::route_profile;

/// Switch, default off: `1`/`on`/`true` folds the generation cue into the
/// last content chunk of a bounded prompt prefill.
pub(crate) const QWEN_PREFILL_FOLD_CUE_ENV: &str = "HF2Q_QWEN_PREFILL_FOLD_CUE";

/// Longest cue the fold accepts. The served cues are four to eight tokens;
/// the bound keeps the second segment on the short-chunk kernels the tests
/// cover.
pub(crate) const MAX_CUE_ROWS: usize = 32;

/// The switch as resolved for the model this thread serves. An invalid value
/// warns once and keeps the fold off.
fn resolved() -> bool {
    const SWITCH: route_profile::Switch = route_profile::switch(QWEN_PREFILL_FOLD_CUE_ENV);
    route_profile::with_value(SWITCH, |raw| parse_switch(QWEN_PREFILL_FOLD_CUE_ENV, raw))
        .unwrap_or_else(|error| {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(error, fallback = false, "invalid Qwen cue-fold switch");
            });
            false
        })
}

/// Whether the operator switched the fold on.
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(switches) = super::prefill_prepared::SwitchesForTest::current() {
        return switches.fold_cue;
    }
    resolved()
}

/// Forward routes that have no folded form: with any of them a prompt
/// prefill keeps its two forwards. GLP steering writes the residual stream
/// between layers, the dump-bisect instrument downloads per-layer tensors,
/// and the dense-Q split profile commits each FFN stage on its own; none of
/// the three is split at the prompt boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UnfoldedRoutes {
    pub(crate) glp_bound: bool,
    pub(crate) dump_bisect: bool,
    pub(crate) dense_q_split_profile: bool,
}

impl UnfoldedRoutes {
    pub(crate) fn any(self) -> bool {
        self.glp_bound || self.dump_bisect || self.dense_q_split_profile
    }
}

/// Whether a chunk of `prefix_rows` content rows followed by `cue_rows` cue
/// rows has the shape the fold serves: both segments are multi-row forwards
/// (a one-row forward is the decode graph) and the cue is short.
pub(crate) fn rows_admitted(prefix_rows: usize, cue_rows: usize) -> bool {
    prefix_rows >= 2 && (2..=MAX_CUE_ROWS).contains(&cue_rows)
}

/// Whether folding these rows can pay. A content chunk of at most 32 rows
/// runs on the short-chunk wait cadence and one row tile; a fold that
/// carries it past 32 rows waits after every layer and starts a second tile
/// (measured about 10 ms slower per turn on Qwen3.8), so such a chunk keeps
/// its two forwards.
pub(crate) fn rows_worth_folding(prefix_rows: usize, cue_rows: usize) -> bool {
    const SHORT_CHUNK_ROWS: usize = 32;
    prefix_rows > SHORT_CHUNK_ROWS || prefix_rows + cue_rows <= SHORT_CHUNK_ROWS
}

/// The boundary a folded forward captures.
pub(crate) struct BoundaryFold {
    /// Rows before the boundary (the content segment).
    pub(crate) prefix_rows: u32,
    /// Checkpoint image the DeltaNet layers write their boundary state into.
    pub(crate) image: LaAnchorImage,
}

thread_local! {
    static REQUEST: RefCell<Option<BoundaryFold>> = const { RefCell::new(None) };
    static BOUNDARY_LOGITS: RefCell<Option<Vec<f32>>> = const { RefCell::new(None) };
}

/// Run `forward` with `request` armed for the next target forward on this
/// thread and return its result with the boundary-row logits.
///
/// The forward must consume the request ([`take_request`]) and publish the
/// boundary logits ([`publish_boundary_logits`]); anything else is an error,
/// so a route that ignores the fold cannot pass for one that took it.
pub(crate) fn with_request<R>(
    request: BoundaryFold,
    forward: impl FnOnce() -> Result<R>,
) -> Result<(R, Vec<f32>)> {
    REQUEST.with(|cell| {
        let mut slot = cell.borrow_mut();
        ensure!(slot.is_none(), "a cue fold is already armed on this thread");
        *slot = Some(request);
        Ok(())
    })?;
    BOUNDARY_LOGITS.with(|cell| cell.borrow_mut().take());
    let result = forward();
    let unconsumed = REQUEST.with(|cell| cell.borrow_mut().take()).is_some();
    let logits = BOUNDARY_LOGITS.with(|cell| cell.borrow_mut().take());
    let result = result?;
    ensure!(!unconsumed, "the target forward did not take the cue fold");
    let logits =
        logits.ok_or_else(|| anyhow!("the folded forward published no boundary logits"))?;
    Ok((result, logits))
}

/// Take the armed fold for a forward of `seq_len` rows, if any.
pub(crate) fn take_request(seq_len: u32) -> Result<Option<BoundaryFold>> {
    let Some(request) = REQUEST.with(|cell| cell.borrow_mut().take()) else {
        return Ok(None);
    };
    let cue_rows = seq_len.checked_sub(request.prefix_rows).ok_or_else(|| {
        anyhow!(
            "cue fold boundary {} beyond {seq_len} rows",
            request.prefix_rows
        )
    })?;
    ensure!(
        rows_admitted(request.prefix_rows as usize, cue_rows as usize),
        "cue fold of {} + {cue_rows} rows is outside the served shape",
        request.prefix_rows
    );
    Ok(Some(request))
}

/// Hand the boundary-row logits of the folded forward back to its caller.
pub(crate) fn publish_boundary_logits(logits: Vec<f32>) {
    BOUNDARY_LOGITS.with(|cell| *cell.borrow_mut() = Some(logits));
}

/// Row range `[first, first + rows)` of a row-major buffer of `width`
/// elements per row, as a view with exactly that many elements.
pub(crate) fn row_view(buffer: &MlxBuffer, first: u32, rows: u32, width: usize) -> MlxBuffer {
    let element_bytes = buffer.dtype().size_of();
    buffer.slice_view(
        (first as usize * width * element_bytes) as u64,
        rows as usize * width,
    )
}

/// The axis-major position rows `[first, first + rows)` of a forward's
/// `[4, seq_len]` position table, as their own `[4, rows]` buffer.
pub(crate) fn segment_positions(
    device: &MlxDevice,
    positions_flat: &[i32],
    seq_len: u32,
    first: u32,
    rows: u32,
) -> Result<MlxBuffer> {
    let (seq, first, rows) = (seq_len as usize, first as usize, rows as usize);
    ensure!(
        positions_flat.len() == 4 * seq && first + rows <= seq,
        "cue fold positions: {} entries for {seq} rows, segment {first}+{rows}",
        positions_flat.len()
    );
    let mut buffer = device
        .alloc_buffer(4 * rows * 4, DType::I32, vec![4 * rows])
        .map_err(|error| anyhow!("alloc cue fold positions: {error}"))?;
    let destination = buffer
        .as_mut_slice::<i32>()
        .map_err(|error| anyhow!("map cue fold positions: {error}"))?;
    for axis in 0..4 {
        destination[axis * rows..(axis + 1) * rows]
            .copy_from_slice(&positions_flat[axis * seq + first..axis * seq + first + rows]);
    }
    Ok(buffer)
}

/// Per-forward scratch of the DeltaNet boundary split: the convolution state
/// at the boundary in the slot layout, and the parameter blocks of the two
/// segment dispatches (the arena's own blocks describe the whole chunk).
pub struct DnBoundaryScratch {
    pub(crate) conv_state: MlxBuffer,
    pub(crate) ssm_params: [MlxBuffer; 2],
    pub(crate) gdn_params: [MlxBuffer; 2],
}

impl DnBoundaryScratch {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &MlxDevice,
        segment_rows: [u32; 2],
        n_k_heads: u32,
        n_v_heads: u32,
        d_k: u32,
        d_v: u32,
        k_width: u32,
        n_seqs: u32,
    ) -> Result<Self> {
        let channels = 2 * n_k_heads * d_k + n_v_heads * d_v;
        let conv_elements = channels as usize * (k_width as usize - 1);
        let conv_state = device
            .alloc_buffer(conv_elements * 4, DType::F32, vec![conv_elements])
            .map_err(|error| anyhow!("alloc cue fold conv state: {error}"))?;
        let params = |values: &[u32]| -> Result<MlxBuffer> {
            let mut buffer = device
                .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
                .map_err(|error| anyhow!("alloc cue fold params: {error}"))?;
            buffer
                .as_mut_slice::<u32>()
                .map_err(|error| anyhow!("map cue fold params: {error}"))?
                .copy_from_slice(values);
            Ok(buffer)
        };
        let ssm = |rows: u32| params(&[channels, rows, n_seqs, k_width]);
        let gdn = |rows: u32| params(&[d_k, d_v, n_k_heads, n_v_heads, rows, n_seqs, 0, 0, 0]);
        Ok(Self {
            conv_state,
            ssm_params: [ssm(segment_rows[0])?, ssm(segment_rows[1])?],
            gdn_params: [gdn(segment_rows[0])?, gdn(segment_rows[1])?],
        })
    }
}

/// One DeltaNet layer's boundary split.
pub struct DnBoundarySplit<'a> {
    /// Rows before the boundary.
    pub(crate) prefix_rows: u32,
    pub(crate) scratch: &'a DnBoundaryScratch,
    /// This layer's recurrent state in the checkpoint image (slot layout).
    pub(crate) recurrent: MlxBuffer,
    /// This layer's convolution state in the checkpoint image (`[K-1, C]`).
    pub(crate) conv_image: MlxBuffer,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::joint_step_options::parse_switch;

    #[test]
    fn switch_is_off_unless_set_on() {
        let parse = |raw| parse_switch(QWEN_PREFILL_FOLD_CUE_ENV, raw);
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some(" ")), Ok(false));
        assert_eq!(parse(Some("on")), Ok(true));
        assert_eq!(parse(Some("0")), Ok(false));
        assert!(parse(Some("cue")).is_err());
    }

    #[test]
    fn a_short_chunk_is_not_folded_past_the_short_chunk_cadence() {
        assert!(rows_worth_folding(720, 5));
        assert!(
            rows_worth_folding(157, 5),
            "a new tile, but one forward less"
        );
        assert!(rows_worth_folding(33, 5));
        assert!(rows_worth_folding(27, 5), "32 rows: still a short chunk");
        assert!(!rows_worth_folding(28, 5));
        assert!(!rows_worth_folding(31, 5));
        assert!(!rows_worth_folding(32, 2));
    }

    #[test]
    fn only_two_multi_row_segments_with_a_short_cue_fold() {
        assert!(rows_admitted(720, 5));
        assert!(rows_admitted(2, 2));
        assert!(rows_admitted(31, MAX_CUE_ROWS));
        assert!(!rows_admitted(1, 5), "a one-row prefix is the decode graph");
        assert!(!rows_admitted(720, 1), "a one-row cue is the decode graph");
        assert!(!rows_admitted(720, 0));
        assert!(!rows_admitted(720, MAX_CUE_ROWS + 1));
    }
}
