//! Explicit test-only SAME-producer routing witnesses. Requires the exact
//! published native named by [`WITNESS_NATIVE_VERSION`].
//! No CPU softmax, extra submission/wait, inferred selector or default activation.
use super::{acquisition::Acquisition, receipt::Identity};
use anyhow::{ensure, Context, Result};
use mlx_native::ops::routing_selection_witness::{
    dispatch_gemma_batch_with_witness, dispatch_gemma_with_witness, read_after_checked_drain,
    SelectionBoundary, SelectionWitness,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    cell::RefCell,
    collections::BTreeSet,
    fs::{File, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::Path,
};
const LAYERS: usize = 30;
const K: usize = 8;
const E: usize = 128;
// T5: four*(initial128 + actual full-prefix verifier132 + at most16 continuation).
// T6: four*(canonical144 + unpublished64), smaller. No retired-owner refund.
const ROW_CAP: usize = 4 * (128 + 132 + 16);
const ENTRY_CAP: usize = ROW_CAP * LAYERS;
pub const PAYLOAD_BYTES: u64 = (ENTRY_CAP * 84) as u64;
pub const METADATA_BYTES: u64 = 64 * 1024 * 1024;
pub const CPU_LEDGER_BYTES: u64 = 16 * 1024 * 1024;
/// Exact published native this witness (and the routing-recipe prep) runs on.
/// Re-reviewed 0.15.11 -> 0.15.15 (registry crates `7d689ee` -> `9c67e1f`,
/// 4 commits): `src/ops/routing_selection_witness.rs` is byte-identical
/// (sha256 6d086d61...4d16a17 in both); `src/ops/**`, `src/shaders/**`,
/// `kernel_registry.rs`, `device.rs`, `build.rs` and `build_support/**` are
/// unchanged. The only changes are additive: bounded GGUF header/map-plan
/// entries (legacy `GgufFile::from_file` now calls the shared parser with the
/// same 100k/1M/256 MiB/64M caps and alignment), buffer backing descriptors,
/// `command_buffer_uses_unretained_references`, opt-in completed-GPU
/// timestamps (off unless the new `install_..._with_gpu_timestamps` is used)
/// and read-only page observation. `CommandEncoder`, `KernelRegistry`,
/// `MlxBuffer`, `MlxDevice`, `blit_copy_bytes` and the witness dispatch keep
/// their semantics, so the route label `native-routing1511-v1` (the witness
/// protocol introduced in 0.15.11) is kept.
/// Re-reviewed 0.15.15 -> 0.15.16 (`9c67e1f` -> `20540a51`): the witness
/// shaders gain the threadgroup barrier after the softmax-max broadcast read
/// (the same WAR race fix as the production decode routing kernels), with no
/// arithmetic change; `src/ops/routing_selection_witness.rs` is unchanged.
/// Re-reviewed 0.15.16 -> 0.15.18 (0.15.17 was never pinned): the changes are
/// the registry's runtime-source fallbacks and IQ4_XS sources, the precise
/// flash-attention prefill variants, IQ2_XXS matmuls and the partial-rotary
/// `rope_multi` pairing fix. `src/ops/routing_selection_witness.rs`, the
/// routing and routing-witness shaders are unchanged, and the witness kernels
/// are not registration fallbacks, so their pipeline resolution is unchanged.
/// Re-reviewed 0.15.18 -> 0.16.0 (`dc236dc` -> `1a5fc09`, tag `v0.16.0`): the changes are the
/// compact expert-matmul tile schedule (`map0c`, `mm_id` `_c` kernels,
/// `IdMmScratch` accessors, routing-policy field `expert_mm_schedule`) and the
/// ring-bytes KV restore. `src/ops/routing_selection_witness.rs` and the
/// routing and routing-witness shaders are unchanged.
pub const WITNESS_NATIVE_VERSION: &str = "0.16.0";
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub payload_bytes: u64,
    pub metadata_bytes: u64,
    pub cpu_ledger_bytes: u64,
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum Mode {
    Ordinary,
    Cancelled,
    Transaction,
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Namespace {
    Canonical,
    UnpublishedCancelled,
    T5Initial,
    T5Verify,
    T5Continuation,
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub struct Row {
    pub lane: usize,
    pub position: usize,
}
#[derive(Clone, Debug, Serialize)]
struct Record {
    invocation: u64,
    namespace: Namespace,
    layer: usize,
    row: Row,
    kernel: &'static str,
    native_variant: &'static str,
    domain: &'static str,
    entry: usize,
    source_row: usize,
    producer_row: usize,
    producer_rows: usize,
}
struct Call {
    id: u64,
    namespace: Namespace,
    rows: Vec<Row>,
    seen: BTreeSet<(usize, usize)>,
}
struct Bank {
    scores: MlxBuffer,
    ids: MlxBuffer,
    status: MlxBuffer,
    selected: MlxBuffer,
    weights: MlxBuffer,
}
struct Active {
    bank: Bank,
    records: Vec<Record>,
    call: Option<Call>,
    seen: BTreeSet<(Namespace, usize, usize, usize)>,
    mode: Mode,
    next: u64,
    read: usize,
    pending: bool,
    meta: File,
    metadata: u64,
}
thread_local! {static ACTIVE:RefCell<Option<Active>>=const{RefCell::new(None)};}
pub struct Scope {
    _same_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|s| {
            if let Some(a) = s.borrow_mut().take() {
                // No checked callback after failure => cannot promise GPU no longer references
                // bank. Retain its real payload/metadata until fresh supervised process exits.
                if a.pending {
                    std::mem::forget(a);
                }
            }
        });
    }
}
pub fn validate_budget(b: Option<&Budget>) -> Result<()> {
    if let Some(b) = b {
        ensure!(
            b.payload_bytes == PAYLOAD_BYTES
                && b.metadata_bytes == METADATA_BYTES
                && b.cpu_ledger_bytes == CPU_LEDGER_BYTES,
            "exact routing bank+metadata+CPU ledger admission"
        );
        ensure!(
            env!("HF2Q_MLX_NATIVE_VERSION") == WITNESS_NATIVE_VERSION,
            "exact published routing-witness native pin required BEFORE model load"
        );
    }
    Ok(())
}
fn rows_ok(rows: &[Row]) -> Result<()> {
    ensure!(
        !rows.is_empty() && rows.len() <= 512,
        "bounded actual row map"
    );
    let mut seen = BTreeSet::new();
    ensure!(
        rows.iter()
            .all(|r| r.lane < 4 && r.position < 144 && seen.insert((r.lane, r.position))),
        "duplicate/out-of-range routing owner/position"
    );
    Ok(())
}
fn event(a: &mut Active, v: serde_json::Value) -> Result<()> {
    let data = serde_json::to_vec(&v)?;
    ensure!(
        data.len() <= 65536
            && a.metadata
                .checked_add(data.len() as u64 + 1)
                .is_some_and(|n| n <= METADATA_BYTES),
        "routing metadata quota"
    );
    a.meta.write_all(&data)?;
    a.meta.write_all(b"\n")?;
    a.metadata += data.len() as u64 + 1;
    Ok(())
}
#[path = "gemma_routing_witness/allocation.rs"]
mod allocation;
pub use allocation::install;
fn namespace_rows(mode: Mode, ns: Namespace, rows: &[Row]) -> Result<()> {
    rows_ok(rows)?;
    let serial_span = |start: usize, end: usize| {
        rows.len() == end - start
            && rows
                .iter()
                .enumerate()
                .all(|(i, r)| r.lane == rows[0].lane && r.position == start + i)
    };
    ensure!(
        match (mode, ns) {
            (Mode::Ordinary, Namespace::Canonical) | (Mode::Cancelled, Namespace::Canonical) =>
                true,
            (Mode::Cancelled, Namespace::UnpublishedCancelled) => serial_span(64, 128),
            (Mode::Transaction, Namespace::T5Initial) => serial_span(0, 128),
            (Mode::Transaction, Namespace::T5Verify) => serial_span(0, 132),
            (Mode::Transaction, Namespace::T5Continuation) =>
                rows.len() == 1 && (129..144).contains(&rows[0].position),
            _ => false,
        },
        "namespace/source spans incompatible with actual trajectory"
    );
    Ok(())
}
pub fn begin(namespace: Namespace, rows: Vec<Row>) -> Result<()> {
    ACTIVE.with(|s| {
        let mut s = s.borrow_mut();
        let Some(a) = s.as_mut() else { return Ok(()) };
        rows_ok(&rows)?;
        ensure!(
            a.call.is_none() && !a.pending,
            "previous routing invocation not checked"
        );
        namespace_rows(a.mode, namespace, &rows)?;
        ensure!(
            a.records
                .len()
                .checked_add(rows.len() * LAYERS)
                .is_some_and(|n| n <= ENTRY_CAP),
            "preallocated routing bank exhausted before encode"
        );
        a.next = a.next.checked_add(1).context("invocation overflow")?;
        a.call = Some(Call {
            id: a.next,
            namespace,
            rows,
            seen: BTreeSet::new(),
        });
        Ok(())
    })
}
pub fn serial(namespace: Namespace, lane: usize, start: usize, count: usize) -> Result<()> {
    ensure!(
        start.checked_add(count).is_some_and(|n| n <= 144),
        "bounded source span"
    );
    begin(
        namespace,
        (start..start + count)
            .map(|position| Row { lane, position })
            .collect(),
    )
}
fn views(
    b: &Bank,
    start: usize,
    rows: usize,
) -> Result<(MlxBuffer, MlxBuffer, MlxBuffer, MlxBuffer, MlxBuffer)> {
    let v = |b: &MlxBuffer, width: usize| {
        b.slice_view((start * width * 4) as u64, rows * width)
            .with_shape(vec![rows, width])
    };
    Ok((
        v(&b.scores, 2)?,
        v(&b.ids, 2)?,
        v(&b.status, 1)?,
        v(&b.selected, K)?,
        v(&b.weights, K)?,
    ))
}
fn check_rows(
    c: &Call,
    global: &BTreeSet<(Namespace, usize, usize, usize)>,
    used: usize,
    layer: usize,
    actual: &[Row],
    source_start: usize,
) -> Result<()> {
    ensure!(
        layer < LAYERS
            && source_start
                .checked_add(actual.len())
                .is_some_and(|n| n <= c.rows.len())
            && c.rows[source_start..source_start + actual.len()] == *actual,
        "actual producer geometry/owner map differs"
    );
    ensure!(
        used.checked_add(actual.len())
            .is_some_and(|n| n <= ENTRY_CAP),
        "routing capacity before producer"
    );
    for (i, r) in actual.iter().enumerate() {
        ensure!(
            !c.seen.contains(&(layer, source_start + i))
                && !global.contains(&(c.namespace, r.lane, r.position, layer)),
            "duplicate logical routing row/layer"
        );
    }
    Ok(())
}
fn reserve(a: &mut Active, layer: usize, actual: &[Row], source_start: usize) -> Result<usize> {
    let c = a
        .call
        .as_ref()
        .context("active routing bank lacks explicit invocation row map")?;
    check_rows(c, &a.seen, a.records.len(), layer, actual, source_start)?;
    Ok(a.records.len())
}
#[allow(clippy::too_many_arguments)]
fn encode(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    device: &MlxDevice,
    logits: &MlxBuffer,
    ids: &MlxBuffer,
    weights: &MlxBuffer,
    scales: &MlxBuffer,
    experts: u32,
    k: u32,
    layer: usize,
    actual: Vec<Row>,
    source_start: usize,
    batch: bool,
) -> Result<bool> {
    ACTIVE.with(|s|{let mut s=s.borrow_mut();let Some(a)=s.as_mut()else{return Ok(false)};
    ensure!(experts as usize==E&&k as usize==K,"selected actual Gemma expert geometry only");
    let start=reserve(a,layer,&actual,source_start)?;let n=actual.len();let (scores,bids,status,selected,stable_weights)=views(&a.bank,start,n)?;
    let witness=SelectionWitness{scores:&scores,ids:&bids,status:&status};a.pending=true;
    let dispatch=if batch {dispatch_gemma_batch_with_witness(enc,reg,device,logits,ids,weights,scales,experts,k,n as u32,Some(witness))?}
        else {ensure!(n==1,"scalar routing is one actual row");dispatch_gemma_with_witness(enc,reg,device,logits,ids,weights,scales,experts,k,Some(witness))?};
    ensure!(dispatch.witness_encoded&&matches!(dispatch.domain,mlx_native::ops::routing_selection_witness::SelectionDomain::SoftmaxBeforeTopKRenorm),"actual native producer/domain receipt");
    // Copies use SAME original IDs/weights that the following experts consume.
    // Blit transitions/barriers are native ordering; never commit or wait here.
    enc.blit_copy_bytes(ids,0,&selected,0,(n*K*4)as u64)?;
    enc.blit_copy_bytes(weights,0,&stable_weights,0,(n*K*4)as u64)?;
    let native_variant=match dispatch.variant {
        mlx_native::ops::routing_selection_witness::RoutingSelectionVariant::GemmaV1=>"GemmaV1",
        mlx_native::ops::routing_selection_witness::RoutingSelectionVariant::GemmaV2=>"GemmaV2",
        mlx_native::ops::routing_selection_witness::RoutingSelectionVariant::GemmaV3=>"GemmaV3",
        _=>anyhow::bail!("wrong actual native family producer"),
    };
    ensure!(!batch||native_variant!="GemmaV2","no scalar V2 approximation for batch");
    let c=a.call.as_mut().unwrap();
    for (i,r) in actual.into_iter().enumerate(){c.seen.insert((layer,source_start+i));a.seen.insert((c.namespace,r.lane,r.position,layer));
        a.records.push(Record{invocation:c.id,namespace:c.namespace,layer,row:r,kernel:dispatch.kernel,native_variant,
            domain:"SoftmaxBeforeTopKRenorm",entry:start+i,source_row:source_start+i,producer_row:i,producer_rows:n});}
    Ok(true)
})
}
#[allow(clippy::too_many_arguments)]
pub fn scalar(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    device: &MlxDevice,
    logits: &MlxBuffer,
    ids: &MlxBuffer,
    weights: &MlxBuffer,
    scales: &MlxBuffer,
    e: u32,
    k: u32,
    layer: usize,
    position: usize,
    slot: Option<usize>,
    source_row: usize,
) -> Result<bool> {
    let row = ACTIVE.with(|s| {
        let s = s.borrow();
        let Some(a) = s.as_ref() else { return Ok(None) };
        let c = a.call.as_ref().context("missing scalar invocation")?;
        ensure!(
            slot.is_some() || (c.rows.len() == 1 && source_row == 0),
            "scalar source lacks actual packed slot descriptor"
        );
        let r = *c.rows.get(source_row).context("physical row missing")?;
        ensure!(
            r.position == position && slot.is_none_or(|slot| slot == r.lane),
            "actual scalar cursor/slot differs"
        );
        Ok(Some(r))
    })?;
    let Some(row) = row else { return Ok(false) };
    encode(
        enc,
        reg,
        device,
        logits,
        ids,
        weights,
        scales,
        e,
        k,
        layer,
        vec![row],
        source_row,
        false,
    )
}
#[allow(clippy::too_many_arguments)]
pub fn batch(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    device: &MlxDevice,
    logits: &MlxBuffer,
    ids: &MlxBuffer,
    weights: &MlxBuffer,
    scales: &MlxBuffer,
    e: u32,
    k: u32,
    layer: usize,
    start: usize,
    rows: usize,
    layout: Option<Vec<(u32, usize, usize, usize)>>,
) -> Result<bool> {
    let actual = ACTIVE.with(|s| {
        let s = s.borrow();
        let Some(a) = s.as_ref() else { return Ok(None) };
        let c = a.call.as_ref().context("missing batch invocation")?;
        let actual = if let Some(layout) = layout {
            let mut out = Vec::new();
            for (slot, pos, len, offset) in layout {
                ensure!(offset == out.len(), "actual packed row gap");
                out.extend((pos..pos + len).map(|position| Row {
                    lane: slot as usize,
                    position,
                }));
            }
            out
        } else {
            let lane = c.rows.first().context("empty scope")?.lane;
            (start..start + rows)
                .map(|position| Row { lane, position })
                .collect()
        };
        ensure!(actual.len() == rows, "actual batch geometry length");
        rows_ok(&actual)?;
        Ok(Some(actual))
    })?;
    let Some(actual) = actual else {
        return Ok(false);
    };
    encode(
        enc, reg, device, logits, ids, weights, scales, e, k, layer, actual, 0, true,
    )
}
#[path = "gemma_routing_witness/output.rs"]
mod output;
pub use output::{end_checked, finish};
#[cfg(test)]
#[path = "gemma_routing_witness_controls.rs"]
mod controls;

#[path = "gemma_routing_witness/gpu_control.rs"]
mod gpu_control;
