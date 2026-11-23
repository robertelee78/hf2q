//! Separately admitted C23 localization only. Copies use the original encoder;
//! blits can split compute encoders. Original-route interpretation additionally
//! requires exact same-K equality of all 60 saved endpoint rows, outside this sink.
use anyhow::{ensure, Context, Result};
use crate::adr063_invariance::{receipt::*, store::Store};
use crate::serve::multi_seq_kv::SlotId;
use mlx_native::{CommandEncoder, DType, MlxBuffer, MlxDevice};
use std::{cell::RefCell, collections::BTreeSet};
use super::{Qwen35Config, Qwen35Variant};

const MAX_PAYLOAD: usize = 16 * 1024 * 1024 * 1024;
const MAX_METADATA: usize = 96 * 1024 * 1024;
const MAX_ROWS: usize = 4096;
#[derive(Clone, Debug)]
pub(crate) struct Owner {
    pub slot: SlotId,
    pub generation: u64,
    pub lane: usize,
    pub turn: usize,
    pub ticket: String,
}
struct Pending {
    id: u64,
    rows: Vec<(usize, usize)>, // actual physical packed row, absolute logical row
    points: BTreeSet<usize>,
    holds: Vec<MlxBuffer>,
}
struct Sink {
    owner: Owner,
    store: Store,
    first: usize,
    end: usize,
    width: usize,
    layers: usize,
    bank: Option<MlxBuffer>,
    pending: Option<Pending>,
    entered: BTreeSet<usize>,
    completed: BTreeSet<usize>,
    metadata: Vec<serde_json::Value>,
    metadata_bytes: usize,
    failed: bool,
}
thread_local! { static ACTIVE: RefCell<Option<Sink>> = const { RefCell::new(None) }; }
static NEXT_FORWARD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
fn point(layers: usize, layer: usize, stage: Stage) -> Result<usize> {
    ensure!(layer < layers, "physical decoder layer out of range");
    Ok(match stage {
        Stage::Input if layer == 0 => 0,
        Stage::Residual => 1 + layer * 2,
        Stage::LayerOutput => 2 + layer * 2,
        _ => anyhow::bail!("unsupported actual layer boundary"),
    })
}
fn visit(points: &mut BTreeSet<usize>, layers: usize, layer: usize, stage: Stage) -> Result<usize> {
    let index = point(layers, layer, stage)?;
    ensure!(points.insert(index), "duplicate layer snapshot");
    Ok(index)
}
fn complete_points(points: &BTreeSet<usize>, layers: usize) -> bool {
    points.iter().copied().eq(0..1 + 2 * layers)
}
fn boundary(index: usize) -> (usize, Stage) {
    if index == 0 { return (0, Stage::Input); }
    ((index - 1) / 2, [Stage::Residual, Stage::LayerOutput][(index - 1) % 2])
}
fn layout(cfg: &Qwen35Config, first: usize, end: usize) -> Result<(usize, usize, usize)> {
    ensure!(first < end && end - first <= MAX_ROWS && end <= i32::MAX as usize,
        "localization row quota/range");
    ensure!(cfg.variant == Qwen35Variant::Dense, "localization requires dense Qwen");
    let width = cfg.hidden_size as usize;
    let layers = cfg.num_hidden_layers as usize;
    ensure!(width > 0 && layers > 0 && layers <= 256, "localization model geometry");
    let bytes = (end - first).checked_mul(width).and_then(|n| n.checked_mul(1 + 2 * layers))
        .and_then(|n| n.checked_mul(4)).context("localization payload overflow")?;
    ensure!(bytes <= MAX_PAYLOAD, "localization payload quota");
    Ok((width, layers, bytes))
}
fn coordinate(lane: usize, turn: usize, first: usize, layer: usize, stage: Stage) -> Coordinate {
    Coordinate { logical_step: turn as u64, layer: layer as u32, stage,
        agent: format!("c23-lane-{lane}"), logical_row_start: first as u64, segment: 0 }
}
pub(crate) fn expected(cfg: &Qwen35Config, lane: usize, turn: usize, first: usize, end: usize)
    -> Result<Vec<ExpectedTensor>> {
    let (width, layers, _) = layout(cfg, first, end)?;
    Ok((0..1 + 2 * layers).map(|index| {
        let (layer, stage) = boundary(index);
        ExpectedTensor { at: coordinate(lane, turn, first, layer, stage), scalar: Scalar::F32,
            shape: vec![end - first, width] }
    }).collect())
}
pub(crate) fn install(owner: Owner, store: Store, first_row: usize, end_row_exclusive: usize) -> Result<()> {
    ensure!(!owner.ticket.is_empty() && owner.ticket.len() <= 256
        && first_row < end_row_exclusive && end_row_exclusive - first_row <= MAX_ROWS
        && end_row_exclusive <= i32::MAX as usize,
        "invalid separately admitted localization owner/range");
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        ensure!(active.is_none(), "nested/stale localization sink");
        *active = Some(Sink { owner, store, first: first_row, end: end_row_exclusive,
            width: 0, layers: 0, bank: None, pending: None, entered: BTreeSet::new(),
            completed: BTreeSet::new(), metadata: vec![], metadata_bytes: 0, failed: false });
        Ok(())
    })
}
fn metadata(sink: &mut Sink, value: serde_json::Value) -> Result<()> {
    let bytes = serde_json::to_vec(&value)?.len();
    sink.metadata_bytes = sink.metadata_bytes.checked_add(bytes).context("metadata overflow")?;
    ensure!(sink.metadata_bytes <= MAX_METADATA, "localization metadata quota");
    sink.metadata.push(value);
    Ok(())
}
/// Same-thread guard. Dropping before the existing checked terminal poisons and
/// leaks the bounded bank and source owners: uncertain GPU retirement is not free.
pub(crate) struct Forward { id: u64, drained: bool, _thread: std::marker::PhantomData<std::rc::Rc<()>> }
pub(crate) fn selected() -> bool { ACTIVE.with(|cell| cell.borrow().is_some()) }
/// `cohort` lists every REAL packed row, never padding; `axes` is actual axis-major
/// positions. Slot generation is the explicit admission owner, not guessed from KV.
pub(crate) fn begin(device: &MlxDevice, cfg: &Qwen35Config, cohort: &[(SlotId, i32, u32)],
    axes: &[i32], packed_width: usize, witness: impl FnOnce() -> serde_json::Value) -> Result<Option<Forward>> {
    let result = ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let Some(sink) = active.as_mut() else { return Ok(None); };
        ensure!(!sink.failed && sink.pending.is_none(), "poisoned/overlapping localization forward");
        ensure!(packed_width <= MAX_ROWS && packed_width >= cohort.len()
            && axes.len() == 4 * packed_width,
            "localization packed axes extent");
        let mut rows = Vec::new();
        for (physical, &(slot, position, _)) in cohort.iter().enumerate() {
            if slot != sink.owner.slot { continue; }
            let logical = usize::try_from(position).context("negative logical position")?;
            if logical < sink.first || logical >= sink.end { continue; }
            ensure!((0..4).all(|axis| axes[axis * packed_width + physical] == position),
                "localization axis-major position mismatch");
            ensure!(sink.entered.insert(logical), "duplicate/stale selected logical row");
            rows.push((physical, logical));
        }
        if rows.is_empty() { return Ok(None); }
        let (width, layers, bytes) = layout(cfg, sink.first, sink.end)?;
        if sink.bank.is_none() {
            sink.bank = Some(device.alloc_buffer(bytes, DType::F32,
                vec![1 + 2 * layers, sink.end - sink.first, width])?);
            sink.width = width; sink.layers = layers;
        }
        ensure!(sink.width == width && sink.layers == layers, "stale model geometry");
        let id = NEXT_FORWARD.fetch_update(std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| anyhow::anyhow!("forward id overflow"))?;
        let coordinates = cohort.iter().map(|&(slot, position, _)|
            Ok((slot, usize::try_from(position)?))).collect::<Result<Vec<_>>>()?;
        let cohort_owners = crate::serve::api::engine_qwen35::anchor_exactness_tests::fresh_http::localization_cohort(&coordinates)?;
        validate_cohort_owner(&sink.owner, &rows, &cohort_owners)?;
        metadata(sink, serde_json::json!({"kind":"forward", "id":id,
            "owner": {"slot":sink.owner.slot.0,"generation":sink.owner.generation,
                "lane":sink.owner.lane,"turn":sink.owner.turn,"ticket":sink.owner.ticket},
            "actual_real_cohort":cohort.iter().map(|&(slot,pos,token)|
                serde_json::json!({"slot":slot.0,"position":pos,"token":token})).collect::<Vec<_>>(),
            "actual_axis_major_positions":axes,"packed_width":packed_width,"cohort_owners":cohort_owners,
            "selected_physical_logical_rows":rows,"dispatch":witness()}))?;
        sink.pending = Some(Pending { id, rows, points: BTreeSet::new(), holds: vec![] });
        Ok(Some(Forward { id, drained: false, _thread: std::marker::PhantomData }))
    });
    poison_error(&result);
    result
}
fn validate_cohort_owner(owner: &Owner, rows: &[(usize, usize)], value: &serde_json::Value) -> Result<()> {
    let cohort = value.as_array().context("localization cohort owner witness absent")?;
    for &(physical, logical) in rows {
        let row = cohort.get(physical).context("localization cohort physical row absent")?;
        let actual = &row["owner"];
        ensure!(row["physical_row"].as_u64() == Some(physical as u64)
            && row["logical_position"].as_u64() == Some(logical as u64)
            && actual["slot"].as_u64() == Some(owner.slot.0 as u64)
            && actual["generation"].as_u64() == Some(owner.generation)
            && actual["lane"].as_u64() == Some(owner.lane as u64)
            && actual["turn"].as_u64() == Some(owner.turn as u64)
            && actual["ticket"].as_str() == Some(owner.ticket.as_str()),
            "stale localization physical row/slot/generation/ticket owner");
    }
    Ok(())
}
pub(crate) fn snapshot(enc: &mut CommandEncoder, source: &MlxBuffer, layer: usize,
    stage: Stage, witness: impl FnOnce() -> serde_json::Value) -> Result<()> {
    let result = ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let Some(sink) = active.as_mut() else { return Ok(()); };
        let Some(pending) = sink.pending.as_mut() else { return Ok(()); };
        let index = visit(&mut pending.points, sink.layers, layer, stage)?;
        let bank = sink.bank.as_ref().context("snapshot bank absent")?;
        ensure!(source.dtype() == DType::F32, "snapshot actual source dtype");
        let row_bytes = sink.width.checked_mul(4).context("row byte overflow")?;
        pending.holds.push(source.clone()); // before ANY blit can partially succeed
        let mut row = 0;
        while row < pending.rows.len() {
            let (physical, logical) = pending.rows[row];
            let mut count = 1;
            while row + count < pending.rows.len()
                && pending.rows[row + count] == (physical + count, logical + count) {
                count += 1;
            }
            let bytes = count.checked_mul(row_bytes).context("span byte overflow")?;
            let src = physical.checked_mul(row_bytes).context("physical row overflow")?;
            let dst = (index * (sink.end - sink.first) + logical - sink.first) * row_bytes;
            ensure!(src.checked_add(bytes).is_some_and(|end| end <= source.data_byte_len())
                && dst.checked_add(bytes).is_some_and(|end| end <= bank.data_byte_len()),
                "snapshot logical source/destination bounds");
            enc.blit_copy_bytes(source, src as u64, bank, dst as u64, bytes as u64)?;
            row += count;
        }
        let id = pending.id;
        metadata(sink, serde_json::json!({"kind":"boundary","forward":id,
            "physical_decoder_layer":layer,"stage":stage,"point":index,
            "source_identity":source.contents_ptr() as usize as u64,
            "source_parent_bytes":source.byte_len(),"source_view_offset":source.byte_offset(),
            "source_logical_bytes":source.data_byte_len(),"row_bytes":row_bytes,
            "source_shape":source.shape(),"bank_layout":"[boundary,absolute_row-first_row,hidden]",
            "host_encode_witness_not_gpu_completion":true,"dispatch":witness()}))
    });
    poison_error(&result);
    result
}
impl Forward {
    pub(crate) fn finish_after_checked_output_head(mut self) -> Result<()> {
        self.drained = true;
        ACTIVE.with(|cell| {
            let mut active = cell.borrow_mut();
            let sink = active.as_mut().context("localization sink disappeared")?;
            ensure!(sink.pending.as_ref().is_some_and(|pending| pending.id == self.id),
                "stale localization retirement owner");
            let pending = sink.pending.take().context("localization forward disappeared")?;
            ensure!(!sink.failed, "poisoned localization retirement");
            if !complete_points(&pending.points, sink.layers) {
                sink.failed = true;
                sink.store.fail("missing actual layer snapshot boundary");
                anyhow::bail!("missing actual layer snapshot boundary");
            }
            sink.completed.extend(pending.rows.iter().map(|&(_, logical)| logical));
            Ok(())
        })
    }
}
fn retain_uncertain(sink: &mut Sink, reason: &str) {
    sink.failed = true;
    sink.store.fail(reason);
    if let Some(bank) = sink.bank.take() { std::mem::forget(bank); }
    if let Some(mut pending) = sink.pending.take() {
        for hold in pending.holds.drain(..) { std::mem::forget(hold); }
    }
}
fn poison_error<T>(result: &Result<T>) {
    if let Err(error) = result {
        ACTIVE.with(|cell| {
            if let Some(sink) = cell.borrow_mut().as_mut() {
                sink.failed = true;
                sink.store.fail(format!("{error:#}"));
            }
        });
    }
}
impl Drop for Forward {
    fn drop(&mut self) {
        if !self.drained {
            ACTIVE.with(|cell| { if let Some(sink) = cell.borrow_mut().as_mut() {
                if sink.pending.as_ref().is_some_and(|pending| pending.id == self.id) {
                    retain_uncertain(sink, "unknown checked GPU retirement; snapshot owners retained to process exit");
                }
            }});
        }
    }
}
pub(crate) fn abort(reason: &str) -> Result<()> {
    ACTIVE.with(|cell| {
        let mut sink = cell.borrow_mut().take().context("localization sink absent")?;
        if sink.pending.is_some() { retain_uncertain(&mut sink, reason); }
        else { sink.store.fail(reason); }
        sink.store.finish().map_err(anyhow::Error::msg)?;
        Ok(())
    })
}
pub(crate) fn finish() -> Result<serde_json::Value> {
    ACTIVE.with(|cell| {
        let mut sink = cell.borrow_mut().take().context("localization sink absent")?;
        if sink.pending.is_some() { retain_uncertain(&mut sink, "finish before checked GPU retirement"); }
        if sink.failed || !sink.completed.iter().copied().eq(sink.first..sink.end) {
            sink.store.fail("missing/stale/uncertain selected logical row coverage");
            sink.store.finish().map_err(anyhow::Error::msg)?;
            anyhow::bail!("incomplete localization retained; no success");
        }
        let bank = sink.bank.as_ref().context("localization bank absent")?;
        let data = bank.as_logical_slice::<f32>()?; // all forwards already checked-drained
        let elements = (sink.end - sink.first) * sink.width;
        for index in 0..1 + 2 * sink.layers {
            let (layer, stage) = boundary(index);
            sink.store.capture(coordinate(sink.owner.lane, sink.owner.turn, sink.first, layer, stage),
                Scalar::F32, vec![sink.end - sink.first, sink.width],
                bank.contents_ptr() as usize as u64, bank.byte_len() as u64,
                bank.byte_offset() + (index * elements * 4) as u64, None,
                "original owning encoder snapshot; all actual existing output-head checked drains complete",
                bytemuck::cast_slice(&data[index * elements..(index + 1) * elements]))
                .map_err(anyhow::Error::msg)?;
        }
        let trace = sink.store.finish().map_err(anyhow::Error::msg)?;
        ensure!(trace.coverage.errors.is_empty(), "failed localization store; no success");
        Ok(serde_json::json!({"schema":"hf2q.c23.layer-localization.v1",
            "records":trace.coverage.captured.len(),"payload_bytes":data.len()*4,
            "first_row":sink.first,"end_row_exclusive":sink.end,
            "hidden_width":sink.width,"physical_decoder_layers":sink.layers,
            "metadata":sink.metadata,"additional_checked_waits":0,
            "blit_changes_compute_encoder_partitions":true,
            "standalone_ffn_output":"NonexistentAtFusedResidualProjection; LayerOutputIsActualFusedSum",
            "original_route_endpoint_equality":"RequiredAll60SameKBeforeInterpretation",
            "native_per_projection_kernel_route":"UnknownNotObserved; dispatch branches only",
            "numeric_policy":"Unspecified","original_c23":"PermanentlyClosed4of4; failure retained"}))
    })
}
#[cfg(test)]
#[path = "c23_layer_capture/tests.rs"]
mod tests;
