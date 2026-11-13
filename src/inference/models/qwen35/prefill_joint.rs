//! Joint prompt prefill: one target forward for the pending prefill chunks
//! of several slots (`HF2Q_QWEN_PREFILL_JOINT=1`), and the cue fold of
//! [`super::prefill_fold`] as its one-lane case.
//!
//! A bounded prompt prefill streams every layer's weights once per forward,
//! whatever the row count. When several slots wait for a prefill at the same
//! time, one forward can carry all their rows and pay that cost once.
//!
//! Each waiting slot is a *lane*. A lane's rows are the chunk its own bounded
//! prefill would run next, cut into the *segments* its own forwards would be:
//! the content rows, and the generation-cue rows when the chunk folds the cue
//! (the stable-boundary checkpoint is then captured between the two).
//!
//! Every operation whose output row depends on that input row alone (the
//! embedding, the norms, the prepared row-tile projections, the dense FFN,
//! the elementwise stages) runs once over all rows of the forward. The
//! operations that mix rows run per lane and segment, with the kernels, row
//! counts, cursors, positions and scratch that lane's own forward has:
//!
//! - a full-attention layer and a MoE FFN layer are built once per segment,
//!   against that lane's cache slot;
//! - the DeltaNet convolution and recurrence are dispatched once per
//!   segment, from the lane's state (or its boundary checkpoint image) into
//!   the lane's state (or the image);
//! - the output head projects each lane's last row and each boundary row.
//!
//! So every lane's cache rows, DeltaNet state, checkpoint, captured target
//! layers and logits are bit-identical to its solo prefill
//! (`prefill_joint_tests`). As for the fold, that holds only while every
//! shared product is row exact: inside a routed
//! [`super::prefill_prepared::PrefillScope`] and for models whose shared
//! products all take the prepared kernel
//! ([`Qwen35Model::prefill_fold_exact`]).
//!
//! # Shared prefix
//!
//! Cold lanes whose prompts start with the same tokens share those rows in
//! the forward. The shared rows of the residual stream are computed once;
//! they are, bit for bit, the rows each sharing lane's own forward computes,
//! because the tokens, the positions and the (empty) history are the same.
//! Nothing a lane keeps is shared: each lane's attention (and MoE FFN) is
//! built over a private copy of the shared rows followed by its own rows and
//! writes all of them into that lane's cache; the DeltaNet scan of the shared
//! rows ends in a per-forward scratch state from which every lane scans its
//! own rows into its own state. A scan split at a row is bit-identical to
//! one scan (both DeltaNet kernels walk the rows in order and carry F32
//! state), so sharing leaves each lane's bits those of its solo prefill.
//!
//! [`Qwen35Model::prefill_fold_exact`]: super::model::Qwen35Model

use std::cell::RefCell;

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::joint_step_options::parse_switch;
use crate::inference::spec_decode::dflash2::route_profile;
use super::kv_cache::{LaAnchorImage, LinearAttnStateSlot};
use crate::serve::multi_seq_kv::SlotId;

/// Switch, default off: `1`/`on`/`true` prefills the waiting slots of a
/// multi-slot worker in one forward. Needs the prepared prefill route.
pub(crate) const QWEN_PREFILL_JOINT_ENV: &str = "HF2Q_QWEN_PREFILL_JOINT";

/// Shortest common prefix worth sharing: below one row tile the private
/// copies cost more than the shared rows save.
pub(crate) const MIN_SHARED_ROWS: usize = 32;

/// A prompt-prefill MoE FFN call of more than this many rows computes each
/// output row from that input row alone, with the same kernels whatever the
/// row count (tensor-matmul router, quantized MM shared expert,
/// expert-grouped `mm_id`; `moe_prefill_rows_tests`). At or below it the
/// layer takes per-pair expert matvecs (`MM_ID_ROUTING_THRESHOLD`) and, at
/// eight rows or fewer, the row-exact router matvec, whose bits differ.
pub(crate) const MOE_ROW_EXACT_MIN_ROWS: u32 = 32;

// The threshold covers every row-count switch of the prefill MoE FFN at
// mlx-native's stock values. `HF2Q_MM_ID_ROUTING_THRESHOLD` moves one of
// them at run time; a MoE model then admits neither the fold nor a joint
// forward (`Qwen35Model::prefill_fold_exact`).
const _: () = assert!(
    MOE_ROW_EXACT_MIN_ROWS >= mlx_native::ops::quantized_matmul_id_ggml::MM_ID_ROUTING_THRESHOLD
        && MOE_ROW_EXACT_MIN_ROWS
            >= mlx_native::ops::dense_matvec_f32_rows::DENSE_MATVEC_F32_ROWS_MAX_ROWS
);

/// Whether this model's resolved route selects joint prefill.
pub(crate) fn enabled() -> bool {
    const SWITCH: route_profile::Switch = route_profile::switch(QWEN_PREFILL_JOINT_ENV);
    route_profile::with_value(SWITCH, |raw| parse_switch(QWEN_PREFILL_JOINT_ENV, raw))
        .unwrap_or_else(|error| {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(error, fallback = false, "invalid Qwen joint-prefill switch");
            });
            false
        })
}

/// The stable-boundary checkpoint a lane's chunk crosses.
pub(crate) struct LaneBoundary {
    /// Rows of the lane's chunk before the boundary (the content segment).
    pub(crate) prefix_rows: u32,
    /// Checkpoint image the DeltaNet layers write the boundary state into.
    pub(crate) image: LaAnchorImage,
}

/// One slot's chunk in a joint forward.
pub(crate) struct LaneRequest {
    pub(crate) slot: SlotId,
    /// Rows of the lane's chunk.
    pub(crate) rows: u32,
    /// Whether the chunk's leading rows are the forward's shared prefix.
    pub(crate) shares_prefix: bool,
    pub(crate) boundary: Option<LaneBoundary>,
}

/// A joint forward: the shared prefix rows first, then every lane's own
/// rows, in lane order.
pub(crate) struct JointRequest {
    /// Rows the sharing lanes have in common (0 = nothing is shared).
    pub(crate) shared_rows: u32,
    pub(crate) lanes: Vec<LaneRequest>,
}

/// The logits a joint forward produced for one lane.
#[derive(Default)]
pub(crate) struct LaneLogits {
    /// Logits of the lane's last row. Empty for the last lane, whose last
    /// row is the forward's own and comes back as the forward's result.
    pub(crate) last: Vec<f32>,
    /// Logits of the lane's last content row, when it has a boundary.
    pub(crate) boundary: Option<Vec<f32>>,
}

thread_local! {
    static REQUEST: RefCell<Option<JointRequest>> = const { RefCell::new(None) };
    static LOGITS: RefCell<Option<Vec<LaneLogits>>> = const { RefCell::new(None) };
}

/// Run `forward` with `request` armed for the next target forward on this
/// thread and return its result with the per-lane logits.
///
/// The forward must consume the request ([`take_request`]) and publish the
/// lane logits ([`publish_logits`]); anything else is an error, so a route
/// that ignores the request cannot pass for one that took it.
pub(crate) fn with_request<R>(
    request: JointRequest,
    forward: impl FnOnce() -> Result<R>,
) -> Result<(R, Vec<LaneLogits>)> {
    let lanes = request.lanes.len();
    REQUEST.with(|cell| {
        let mut slot = cell.borrow_mut();
        ensure!(
            slot.is_none(),
            "a joint prefill is already armed on this thread"
        );
        *slot = Some(request);
        Ok(())
    })?;
    LOGITS.with(|cell| cell.borrow_mut().take());
    let result = forward();
    let unconsumed = REQUEST.with(|cell| cell.borrow_mut().take()).is_some();
    let logits = LOGITS.with(|cell| cell.borrow_mut().take());
    let result = result?;
    ensure!(
        !unconsumed,
        "the target forward did not take the joint prefill"
    );
    let logits = logits.ok_or_else(|| anyhow!("the joint forward published no lane logits"))?;
    ensure!(
        logits.len() == lanes,
        "the joint forward published logits of {} lanes, {lanes} were armed",
        logits.len()
    );
    Ok((result, logits))
}

/// Hand the per-lane logits of a joint forward back to its caller.
pub(crate) fn publish_logits(logits: Vec<LaneLogits>) {
    LOGITS.with(|cell| *cell.borrow_mut() = Some(logits));
}

/// A run of consecutive rows of the forward: `first..first + rows`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RowRun {
    pub(crate) first: u32,
    pub(crate) rows: u32,
}

/// One lane of an armed forward, placed in the forward's rows.
pub(crate) struct PlanLane {
    pub(crate) slot: SlotId,
    /// The forward row of the lane's first own (unshared) row.
    pub(crate) own_first: u32,
    /// Rows of the lane's chunk, shared rows included.
    pub(crate) rows: u32,
    /// Leading rows of the chunk that are the forward's shared prefix.
    pub(crate) shared_rows: u32,
    pub(crate) boundary: Option<LaneBoundary>,
}

impl PlanLane {
    /// The lane's segments, as `(first, rows)` in the lane's own row count:
    /// the rows its own forwards would each have run.
    pub(crate) fn segments(&self) -> Vec<(u32, u32)> {
        match &self.boundary {
            Some(boundary) => vec![
                (0, boundary.prefix_rows),
                (boundary.prefix_rows, self.rows - boundary.prefix_rows),
            ],
            None => vec![(0, self.rows)],
        }
    }

    /// The forward rows holding lane rows `first..first + rows`, in order:
    /// the shared part (if any), then the lane's own part.
    pub(crate) fn runs(&self, first: u32, rows: u32) -> Vec<RowRun> {
        let end = first + rows;
        let mut runs = Vec::with_capacity(2);
        if first < self.shared_rows {
            runs.push(RowRun {
                first,
                rows: end.min(self.shared_rows) - first,
            });
        }
        if end > self.shared_rows {
            let own = first.max(self.shared_rows);
            runs.push(RowRun {
                first: self.own_first + (own - self.shared_rows),
                rows: end - own,
            });
        }
        runs
    }

    /// The forward row of lane row `row` (an own row).
    fn own_row(&self, row: u32) -> u32 {
        self.own_first + (row - self.shared_rows)
    }

    /// The forward row of the lane's last row.
    pub(crate) fn last_row(&self) -> u32 {
        self.own_row(self.rows - 1)
    }

    /// The forward row of the lane's last content row, when it has a
    /// boundary.
    pub(crate) fn boundary_row(&self) -> Option<u32> {
        self.boundary
            .as_ref()
            .map(|boundary| self.own_row(boundary.prefix_rows - 1))
    }
}

/// An armed joint forward, validated against the forward's row count.
pub(crate) struct JointPlan {
    pub(crate) shared_rows: u32,
    pub(crate) lanes: Vec<PlanLane>,
}

/// One segment of one lane: the unit a row-mixing layer is built for.
pub(crate) struct Segment {
    pub(crate) slot: SlotId,
    /// Rows of the segment.
    pub(crate) rows: u32,
    /// The forward rows holding the segment's rows, in order.
    pub(crate) runs: Vec<RowRun>,
    /// Whether the segment's shared rows are the ones the forward keeps as
    /// the shared rows' output (the first sharing lane's; every sharing
    /// lane computes the same bits).
    pub(crate) emits_shared: bool,
}

impl Segment {
    /// Whether the MoE FFN of the segment's own forward is row exact, so
    /// that a call over more rows gives the segment's rows the same bits.
    pub(crate) fn moe_row_exact(&self) -> bool {
        self.rows > MOE_ROW_EXACT_MIN_ROWS
    }

    /// `(segment row, forward rows)` of each part of the segment's output
    /// that the forward keeps.
    pub(crate) fn emitted(&self, shared_rows: u32) -> Vec<(u32, RowRun)> {
        let mut at = 0;
        let mut emitted = Vec::with_capacity(self.runs.len());
        for run in &self.runs {
            let shared = run.first < shared_rows;
            if !shared || self.emits_shared {
                emitted.push((at, *run));
            }
            at += run.rows;
        }
        emitted
    }
}

impl JointPlan {
    /// Every segment of every lane, in lane order.
    pub(crate) fn segments(&self) -> Vec<Segment> {
        let mut shared_emitted = false;
        let mut segments = Vec::new();
        for lane in &self.lanes {
            for (first, rows) in lane.segments() {
                let runs = lane.runs(first, rows);
                let has_shared = runs.first().is_some_and(|run| run.first < self.shared_rows)
                    && lane.shared_rows > 0;
                let emits_shared = has_shared && !shared_emitted;
                shared_emitted |= has_shared;
                segments.push(Segment {
                    slot: lane.slot,
                    rows,
                    runs,
                    emits_shared,
                });
            }
        }
        segments
    }
}

/// Take the armed joint forward for a forward of `seq_len` rows, if any.
pub(crate) fn take_request(seq_len: u32, n_seqs: u32) -> Result<Option<JointPlan>> {
    let Some(request) = REQUEST.with(|cell| cell.borrow_mut().take()) else {
        return Ok(None);
    };
    plan(request, seq_len, n_seqs).map(Some)
}

fn plan(request: JointRequest, seq_len: u32, n_seqs: u32) -> Result<JointPlan> {
    let shared = request.shared_rows;
    ensure!(!request.lanes.is_empty(), "a joint prefill needs a lane");
    let sharing = request
        .lanes
        .iter()
        .filter(|lane| lane.shares_prefix)
        .count();
    ensure!(
        (shared == 0 && sharing == 0) || (shared > 0 && sharing >= 2),
        "joint prefill: {sharing} lanes share a prefix of {shared} rows"
    );
    let mut next = shared;
    let mut lanes = Vec::with_capacity(request.lanes.len());
    for (index, lane) in request.lanes.into_iter().enumerate() {
        ensure!(
            lane.slot.0 < n_seqs,
            "joint prefill lane {index}: slot {} outside the cache's {n_seqs} slots",
            lane.slot.0
        );
        ensure!(
            lanes.iter().all(|other: &PlanLane| other.slot != lane.slot),
            "joint prefill: slot {} appears twice",
            lane.slot.0
        );
        let lane_shared = if lane.shares_prefix { shared } else { 0 };
        // Every dispatch of a row-mixing kernel carries at least two rows
        // (a one-row forward is the decode graph): the lane's own rows
        // before its boundary, and the rows after it.
        let content_rows = lane
            .boundary
            .as_ref()
            .map_or(lane.rows, |boundary| boundary.prefix_rows);
        ensure!(
            content_rows >= lane_shared + 2 && content_rows <= lane.rows,
            "joint prefill lane {index}: {content_rows} content rows of {} with {lane_shared} shared",
            lane.rows
        );
        if let Some(boundary) = &lane.boundary {
            ensure!(
                super::prefill_fold::rows_admitted(
                    boundary.prefix_rows as usize,
                    (lane.rows - boundary.prefix_rows) as usize
                ),
                "joint prefill lane {index}: cue fold of {} + {} rows is outside the served shape",
                boundary.prefix_rows,
                lane.rows - boundary.prefix_rows
            );
        }
        let own_rows = lane.rows - lane_shared;
        lanes.push(PlanLane {
            slot: lane.slot,
            own_first: next,
            rows: lane.rows,
            shared_rows: lane_shared,
            boundary: lane.boundary,
        });
        next = next
            .checked_add(own_rows)
            .ok_or_else(|| anyhow!("joint prefill row count overflow"))?;
    }
    ensure!(
        next == seq_len,
        "joint prefill lanes hold {next} rows, the forward has {seq_len}"
    );
    Ok(JointPlan {
        shared_rows: shared,
        lanes,
    })
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

/// The axis-major position rows of `runs` out of a forward's
/// `[4, seq_len]` position table, as their own `[4, rows]` buffer.
pub(crate) fn segment_positions(
    device: &MlxDevice,
    positions_flat: &[i32],
    seq_len: u32,
    runs: &[RowRun],
) -> Result<MlxBuffer> {
    let seq = seq_len as usize;
    let rows: usize = runs.iter().map(|run| run.rows as usize).sum();
    ensure!(
        positions_flat.len() == 4 * seq
            && runs
                .iter()
                .all(|run| (run.first + run.rows) as usize <= seq),
        "joint prefill positions: {} entries for {seq} rows, segment {runs:?}",
        positions_flat.len()
    );
    let mut buffer = device
        .alloc_buffer(4 * rows * 4, DType::I32, vec![4 * rows])
        .map_err(|error| anyhow!("alloc joint prefill positions: {error}"))?;
    let destination = buffer
        .as_mut_slice::<i32>()
        .map_err(|error| anyhow!("map joint prefill positions: {error}"))?;
    for axis in 0..4 {
        let mut at = axis * rows;
        for run in runs {
            let (first, count) = (run.first as usize, run.rows as usize);
            destination[at..at + count]
                .copy_from_slice(&positions_flat[axis * seq + first..axis * seq + first + count]);
            at += count;
        }
    }
    Ok(buffer)
}

/// A `[rows, width]` F32 scratch buffer owned by the forward.
pub(crate) fn rows_scratch(device: &MlxDevice, rows: u32, width: usize) -> Result<MlxBuffer> {
    device
        .alloc_buffer(
            rows as usize * width * 4,
            DType::F32,
            vec![rows as usize, width],
        )
        .map_err(|error| anyhow!("alloc joint prefill rows scratch: {error}"))
}

/// Copy rows between row-major F32 buffers of `width` elements per row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_rows(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut mlx_native::KernelRegistry,
    device: &MlxDevice,
    source: &MlxBuffer,
    source_first: u32,
    destination: &MlxBuffer,
    destination_first: u32,
    rows: u32,
    width: u32,
) -> Result<()> {
    mlx_native::ops::copy::dispatch_strided_copy_f32(
        encoder,
        registry,
        device.metal_device(),
        &row_view(source, source_first, rows, width as usize),
        &row_view(destination, destination_first, rows, width as usize),
        &mlx_native::ops::copy::StridedCopyParams {
            rows,
            cols: width,
            stride_row: width,
            stride_col: 1,
        },
    )
    .map_err(|error| anyhow!("joint prefill row copy: {error}"))
}

/// Gather a segment's rows out of `source` (a buffer over the forward's
/// rows) into the contiguous `destination`.
pub(crate) fn gather_segment(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut mlx_native::KernelRegistry,
    device: &MlxDevice,
    source: &MlxBuffer,
    segment: &Segment,
    destination: &MlxBuffer,
    width: u32,
) -> Result<()> {
    let mut at = 0;
    for run in &segment.runs {
        copy_rows(
            encoder,
            registry,
            device,
            source,
            run.first,
            destination,
            at,
            run.rows,
            width,
        )?;
        at += run.rows;
    }
    Ok(())
}

/// Scatter the kept rows of a segment's contiguous output into
/// `destination` (a buffer over the forward's rows).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_segment(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut mlx_native::KernelRegistry,
    device: &MlxDevice,
    source: &MlxBuffer,
    segment: &Segment,
    shared_rows: u32,
    destination: &MlxBuffer,
    width: u32,
) -> Result<()> {
    for (at, run) in segment.emitted(shared_rows) {
        copy_rows(
            encoder,
            registry,
            device,
            source,
            at,
            destination,
            run.first,
            run.rows,
            width,
        )?;
    }
    Ok(())
}

/// One dispatch of the DeltaNet convolution and of the recurrence: rows
/// `first..first + rows` of the forward, from one state into another.
pub struct DnRun {
    pub(crate) first: u32,
    pub(crate) rows: u32,
    /// Convolution state read and written, in the slot layout.
    pub(crate) conv_in: MlxBuffer,
    pub(crate) conv_out: MlxBuffer,
    /// Recurrent state read and written, in the slot layout.
    pub(crate) recurrent_in: MlxBuffer,
    pub(crate) recurrent_out: MlxBuffer,
    /// Where the convolution state written by this run is also kept in the
    /// checkpoint image's `[K-1, channels]` layout (a boundary run).
    pub(crate) conv_image: Option<MlxBuffer>,
    /// Parameter blocks of the two dispatches.
    pub(crate) ssm_params: MlxBuffer,
    pub(crate) gdn_params: MlxBuffer,
}

/// Per-forward scratch of the DeltaNet runs: the state at the end of the
/// shared prefix, each boundary lane's convolution state at its boundary in
/// the slot layout, and the parameter blocks of every run (the arena's own
/// blocks describe the whole forward). The same runs repeat in every layer.
pub(crate) struct DnRunScratch {
    shared_conv: Option<MlxBuffer>,
    shared_recurrent: Option<MlxBuffer>,
    boundary_conv: Vec<Option<MlxBuffer>>,
    /// `(first, rows, ssm params, gdn params)` of every run, in run order.
    runs: Vec<(u32, u32, MlxBuffer, MlxBuffer)>,
    dims: DnDims,
}

/// The DeltaNet shape the runs are dispatched with.
#[derive(Clone, Copy)]
pub(crate) struct DnDims {
    pub(crate) n_k_heads: u32,
    pub(crate) n_v_heads: u32,
    pub(crate) d_k: u32,
    pub(crate) d_v: u32,
    pub(crate) k_width: u32,
    pub(crate) n_seqs: u32,
}

impl DnDims {
    fn channels(&self) -> u32 {
        2 * self.n_k_heads * self.d_k + self.n_v_heads * self.d_v
    }
}

impl DnRunScratch {
    pub(crate) fn new(device: &MlxDevice, plan: &JointPlan, dims: DnDims) -> Result<Self> {
        let channels = dims.channels();
        let conv_elements = channels as usize * (dims.k_width as usize - 1);
        let recurrent_elements = dims.d_k as usize * dims.d_v as usize * dims.n_v_heads as usize;
        let state = |elements: usize| -> Result<MlxBuffer> {
            device
                .alloc_buffer(elements * 4, DType::F32, vec![elements])
                .map_err(|error| anyhow!("alloc joint prefill DeltaNet state: {error}"))
        };
        let params = |values: &[u32]| -> Result<MlxBuffer> {
            let mut buffer = device
                .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
                .map_err(|error| anyhow!("alloc joint prefill DeltaNet params: {error}"))?;
            buffer
                .as_mut_slice::<u32>()
                .map_err(|error| anyhow!("map joint prefill DeltaNet params: {error}"))?
                .copy_from_slice(values);
            Ok(buffer)
        };
        let mut runs = Vec::new();
        let mut push = |first: u32, rows: u32| -> Result<()> {
            runs.push((
                first,
                rows,
                params(&[channels, rows, dims.n_seqs, dims.k_width])?,
                params(&[
                    dims.d_k,
                    dims.d_v,
                    dims.n_k_heads,
                    dims.n_v_heads,
                    rows,
                    dims.n_seqs,
                    0,
                    0,
                    0,
                ])?,
            ));
            Ok(())
        };
        if plan.shared_rows > 0 {
            push(0, plan.shared_rows)?;
        }
        let mut boundary_conv = Vec::with_capacity(plan.lanes.len());
        for lane in &plan.lanes {
            let content = lane
                .boundary
                .as_ref()
                .map_or(lane.rows, |boundary| boundary.prefix_rows);
            push(lane.own_first, content - lane.shared_rows)?;
            if lane.boundary.is_some() {
                push(
                    lane.own_first + (content - lane.shared_rows),
                    lane.rows - content,
                )?;
                boundary_conv.push(Some(state(conv_elements)?));
            } else {
                boundary_conv.push(None);
            }
        }
        let shared = plan.shared_rows > 0;
        Ok(Self {
            shared_conv: shared.then(|| state(conv_elements)).transpose()?,
            shared_recurrent: shared.then(|| state(recurrent_elements)).transpose()?,
            boundary_conv,
            runs,
            dims,
        })
    }

    /// The runs of one DeltaNet layer: `slot` is the layer's state in the
    /// cache, `layer` its index among the linear-attention layers (the
    /// index of its region in a checkpoint image).
    pub(crate) fn layer_runs(
        &self,
        plan: &JointPlan,
        slot: &LinearAttnStateSlot,
        layer: usize,
    ) -> Result<Vec<DnRun>> {
        let dims = self.dims;
        let mut params = self.runs.iter();
        let mut next = |conv_in: &MlxBuffer,
                        conv_out: &MlxBuffer,
                        recurrent_in: &MlxBuffer,
                        recurrent_out: &MlxBuffer,
                        conv_image: Option<MlxBuffer>|
         -> Result<DnRun> {
            let (first, rows, ssm_params, gdn_params) = params
                .next()
                .ok_or_else(|| anyhow!("joint prefill DeltaNet run scratch is short"))?;
            Ok(DnRun {
                first: *first,
                rows: *rows,
                conv_in: conv_in.clone(),
                conv_out: conv_out.clone(),
                recurrent_in: recurrent_in.clone(),
                recurrent_out: recurrent_out.clone(),
                conv_image,
                ssm_params: ssm_params.clone(),
                gdn_params: gdn_params.clone(),
            })
        };
        let lane_state = |lane: &PlanLane| {
            let (conv_current, conv_scratch) = slot.conv_bufs_for_slot(lane.slot);
            let (recurrent_current, recurrent_scratch) = slot.recurrent_bufs_for_slot(lane.slot);
            super::gpu_delta_net::narrow_la_ping_pong_to_slot(
                conv_current,
                conv_scratch,
                recurrent_current,
                recurrent_scratch,
                lane.slot,
                dims.d_k,
                dims.d_v,
                dims.n_v_heads,
                dims.channels(),
                dims.k_width - 1,
            )
        };
        let mut runs = Vec::with_capacity(self.runs.len());
        if plan.shared_rows > 0 {
            // The shared rows start from the state of a sharing lane: all of
            // them are cold, so any one's (zeroed) state is every one's.
            let donor = plan
                .lanes
                .iter()
                .find(|lane| lane.shared_rows > 0)
                .ok_or_else(|| anyhow!("joint prefill: a shared prefix without a sharing lane"))?;
            let state = lane_state(donor);
            let (conv, recurrent) = self.shared_state()?;
            runs.push(next(
                &state.conv_state_in,
                conv,
                &state.recurrent_in,
                recurrent,
                None,
            )?);
        }
        for (index, lane) in plan.lanes.iter().enumerate() {
            let state = lane_state(lane);
            let (conv_in, recurrent_in) = if lane.shared_rows > 0 {
                self.shared_state()?
            } else {
                (&state.conv_state_in, &state.recurrent_in)
            };
            match (&lane.boundary, &self.boundary_conv[index]) {
                (Some(boundary), Some(boundary_conv)) => {
                    let views = boundary
                        .image
                        .boundary_views(layer)
                        .with_context(|| format!("joint prefill lane {index} checkpoint image"))?;
                    runs.push(next(
                        conv_in,
                        boundary_conv,
                        recurrent_in,
                        &views.recurrent,
                        Some(views.conv),
                    )?);
                    runs.push(next(
                        boundary_conv,
                        &state.conv_state_out,
                        &views.recurrent,
                        &state.recurrent_out,
                        None,
                    )?);
                }
                (None, None) => runs.push(next(
                    conv_in,
                    &state.conv_state_out,
                    recurrent_in,
                    &state.recurrent_out,
                    None,
                )?),
                _ => {
                    return Err(anyhow!(
                        "joint prefill lane {index}: boundary scratch mismatch"
                    ))
                }
            }
        }
        ensure!(
            runs.len() == self.runs.len(),
            "joint prefill: {} DeltaNet runs for {} parameter blocks",
            runs.len(),
            self.runs.len()
        );
        Ok(runs)
    }

    fn shared_state(&self) -> Result<(&MlxBuffer, &MlxBuffer)> {
        match (&self.shared_conv, &self.shared_recurrent) {
            (Some(conv), Some(recurrent)) => Ok((conv, recurrent)),
            _ => Err(anyhow!(
                "joint prefill: shared DeltaNet state scratch is missing"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::joint_step_options::parse_switch;

    fn lane(slot: u32, rows: u32, shares_prefix: bool) -> LaneRequest {
        LaneRequest {
            slot: SlotId(slot),
            rows,
            shares_prefix,
            boundary: None,
        }
    }

    #[test]
    fn switch_is_off_unless_set_on() {
        let parse = |raw| parse_switch(QWEN_PREFILL_JOINT_ENV, raw);
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some("1")), Ok(true));
        assert_eq!(parse(Some("off")), Ok(false));
        assert!(parse(Some("share")).is_err());
    }

    #[test]
    fn lanes_are_laid_out_behind_the_shared_prefix() {
        let request = JointRequest {
            shared_rows: 40,
            lanes: vec![lane(2, 100, true), lane(0, 7, false), lane(3, 50, true)],
        };
        let plan = plan(request, 40 + 60 + 7 + 10, 4).expect("plan");
        assert_eq!(plan.lanes[0].own_first, 40);
        assert_eq!(plan.lanes[1].own_first, 100);
        assert_eq!(plan.lanes[2].own_first, 107);
        assert_eq!(
            plan.lanes[0].runs(0, 100),
            vec![
                RowRun { first: 0, rows: 40 },
                RowRun {
                    first: 40,
                    rows: 60
                }
            ]
        );
        assert_eq!(
            plan.lanes[1].runs(0, 7),
            vec![RowRun {
                first: 100,
                rows: 7
            }]
        );
        assert_eq!(plan.lanes[2].last_row(), 116);
        let segments = plan.segments();
        assert_eq!(segments.len(), 3);
        assert!(segments[0].emits_shared && !segments[2].emits_shared);
        assert_eq!(
            segments[0].emitted(40),
            vec![
                (0, RowRun { first: 0, rows: 40 }),
                (
                    40,
                    RowRun {
                        first: 40,
                        rows: 60
                    }
                )
            ]
        );
        assert_eq!(
            segments[2].emitted(40),
            vec![(
                40,
                RowRun {
                    first: 107,
                    rows: 10
                }
            )]
        );
    }

    #[test]
    fn inadmissible_plans_are_refused() {
        let refuse = |request: JointRequest, seq_len: u32| plan(request, seq_len, 4).is_err();
        // Row count of the forward.
        assert!(refuse(
            JointRequest {
                shared_rows: 0,
                lanes: vec![lane(0, 5, false), lane(1, 5, false)]
            },
            11
        ));
        // A slot twice, a slot outside the cache.
        assert!(refuse(
            JointRequest {
                shared_rows: 0,
                lanes: vec![lane(1, 5, false), lane(1, 5, false)]
            },
            10
        ));
        assert!(refuse(
            JointRequest {
                shared_rows: 0,
                lanes: vec![lane(4, 5, false)]
            },
            5
        ));
        // A one-row lane; a shared prefix that leaves a lane one own row.
        assert!(refuse(
            JointRequest {
                shared_rows: 0,
                lanes: vec![lane(0, 1, false)]
            },
            1
        ));
        assert!(refuse(
            JointRequest {
                shared_rows: 40,
                lanes: vec![lane(0, 41, true), lane(1, 50, true)]
            },
            51
        ));
        // A prefix nobody or only one lane shares.
        assert!(refuse(
            JointRequest {
                shared_rows: 40,
                lanes: vec![lane(0, 50, true), lane(1, 50, false)]
            },
            100
        ));
        assert!(refuse(
            JointRequest {
                shared_rows: 0,
                lanes: vec![lane(0, 50, true), lane(1, 50, true)]
            },
            100
        ));
    }
}
