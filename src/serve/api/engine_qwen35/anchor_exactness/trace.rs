//! Test-only, thread-local acquisition. Observing a returned tensor never changes
//! its route, synchronization, or production decision. Frozen decisions are a
//! separate, explicitly labelled diagnostic mode; lifecycle tests never use it.
use super::*;
use std::{cell::RefCell, collections::BTreeMap, ops::Range};
use mlx_native::{DType, MlxBuffer};
use crate::inference::models::qwen35::kv_cache::FullAttnKvSlot;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct State {
    pub target: Vec<u32>,
    pub mtp: Option<u32>,
    pub parity: Vec<bool>,
    pub regions: Vec<Vec<u8>>,
}
fn elements(buffer: &MlxBuffer, range: Range<usize>) -> Result<Vec<u8>> {
    ensure!(range.end.checked_mul(buffer.dtype().size_of()).is_some_and(|n| n <= buffer.data_byte_len()),
        "observable region exceeds logical extent");
    match buffer.dtype() {
        DType::U8 => Ok(buffer.as_logical_slice::<u8>()?[range].to_vec()),
        DType::F32 => Ok(buffer.as_logical_slice::<f32>()?[range].iter()
            .flat_map(|v| v.to_bits().to_le_bytes()).collect()),
        _ => anyhow::bail!("C2 cache observation requires explicit U8/F32 representation"),
    }
}
fn slot_region(buffer: &MlxBuffer, slot: SlotId) -> Result<Vec<u8>> {
    let shape = buffer.shape();
    ensure!(!shape.is_empty() && (slot.0 as usize) < shape[0], "slot axis mismatch");
    let width: usize = shape[1..].iter().product();
    elements(buffer, slot.0 as usize * width..(slot.0 as usize + 1) * width)
}
fn rows(buffer: &MlxBuffer, slot: SlotId, valid: u32) -> Result<Vec<u8>> {
    let shape = buffer.shape();
    ensure!(shape.len() == 4 && (slot.0 as usize) < shape[0] && valid as usize <= shape[2],
        "full-attention valid extent mismatch");
    let mut bytes = Vec::new();
    for head in 0..shape[1] {
        let first = (slot.0 as usize * shape[1] + head) * shape[2] * shape[3];
        bytes.extend(elements(buffer, first..first + valid as usize * shape[3])?);
    }
    Ok(bytes)
}
fn full(layer: &FullAttnKvSlot, slot: SlotId, regions: &mut Vec<Vec<u8>>) -> Result<u32> {
    let valid = *layer.current_len.get(slot.0 as usize).context("full-attention cursor")?;
    if let Some(tq) = &layer.tq {
        for buffer in [&tq.k_packed, &tq.k_norms, &tq.v_packed, &tq.v_norms] {
            regions.push(rows(buffer, slot, valid)?);
        }
    }
    for buffer in [&layer.k, &layer.v].into_iter().flatten() {
        regions.push(rows(buffer, slot, valid)?);
    }
    Ok(valid)
}
impl State {
    pub(super) fn read(kv: &HybridKvCache, slot: SlotId) -> Result<Self> {
        let mut regions = Vec::new();
        let target = kv.full_attn.iter().map(|l| full(l, slot, &mut regions)).collect::<Result<Vec<_>>>()?;
        let mtp = kv.mtp_slot.as_ref().map(|l| full(l, slot, &mut regions)).transpose()?;
        let mut parity = Vec::new();
        for layer in &kv.linear_attn {
            parity.push(layer.pp_flipped[slot.0 as usize]);
            // Scratch is not semantic state and is intentionally not restored.
            regions.push(slot_region(layer.conv_bufs_for_slot(slot).0, slot)?);
            regions.push(slot_region(layer.recurrent_bufs_for_slot(slot).0, slot)?);
        }
        Ok(Self { target, mtp, parity, regions })
    }
    pub(super) fn same(&self, other: &Self, context: &str) -> Result<()> {
        ensure!(self == other, "DivergedExactInvariant: {context}"); Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Forward {
    pub slot: u32,
    pub start: usize,
    pub input: Vec<u32>,
    pub route: String,
    pub logits: Vec<u32>,
    pub hidden: Vec<u32>,
    pub vocab: usize,
    pub hidden_width: usize,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Observations {
    pub forwards: Vec<Forward>,
    pub decisions: Vec<(usize, u32)>,
    pub gpu_argmax: Vec<(usize, u32)>,
    pub readiness: Option<std::result::Result<(), &'static str>>,
    pub route_events: Vec<&'static str>,
    pub frozen_gpu_selector_used: bool,
}
struct Active {
    observations: Observations,
    frozen: Option<Vec<u32>>,
    decision_base: usize,
    target_slot: SlotId,
    prediction_origin: usize,
    writer: Option<super::transaction_trace::Writer>,
    published_ledger: Vec<u32>,
}
thread_local! { static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) }; }
pub(super) struct Scope;
impl Scope {
    pub(super) fn begin(frozen: Option<Vec<u32>>, decision_base: usize) -> Result<Self> {
        ACTIVE.with(|cell| {
            let mut active = cell.borrow_mut();
            ensure!(active.is_none(), "nested C2 acquisition");
            *active = Some(Active { observations: Observations::default(), frozen, decision_base,
                target_slot: SLOT, prediction_origin: 0, writer: None, published_ledger: Vec::new() });
            Ok(Self)
        })
    }
    pub(super) fn observe_slot(&self, slot: SlotId) {
        ACTIVE.with(|cell| {
            if let Some(active) = cell.borrow_mut().as_mut() { active.target_slot = slot; }
        });
    }
    pub(super) fn attach(&self, writer: super::transaction_trace::Writer, ledger: &[u32]) -> Result<()> {
        ACTIVE.with(|cell| {
            let mut active = cell.borrow_mut();
            let active = active.as_mut().context("C2 scope absent")?;
            ensure!(active.writer.is_none(), "duplicate transaction writer");
            active.writer = Some(writer);
            active.published_ledger = ledger.to_vec();
            Ok(())
        })
    }
    pub(super) fn publish(&self, ledger: &[u32]) {
        ACTIVE.with(|cell| {
            if let Some(active) = cell.borrow_mut().as_mut() { active.published_ledger = ledger.to_vec(); }
        });
    }
    pub(super) fn finish_transactions(&self) -> Result<()> {
        let writer = ACTIVE.with(|cell| cell.borrow_mut().as_mut().and_then(|a| a.writer.take()));
        if let Some(writer) = writer { writer.finish()?; }
        Ok(())
    }
    pub(super) fn inspect<T>(&self, f:impl FnOnce(&Observations)->T) -> Result<T> {
        ACTIVE.with(|cell| {
            let active=cell.borrow();
            Ok(f(&active.as_ref().context("C2 scope absent")?.observations))
        })
    }
    pub(super) fn take(&self) -> Result<Observations> {
        ACTIVE.with(|cell| Ok(cell.borrow_mut().as_mut().context("C2 scope absent")?
            .observations.clone()))
    }
}
impl Drop for Scope { fn drop(&mut self) { ACTIVE.with(|cell| { cell.borrow_mut().take(); }); } }
thread_local! { static COMPARISON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
pub(crate) struct Comparison(bool);
impl Comparison {
    pub(crate) fn begin() -> Self { Self(COMPARISON.with(|cell| cell.replace(true))) }
}
impl Drop for Comparison { fn drop(&mut self) { COMPARISON.with(|cell| cell.set(self.0)); } }
pub(crate) fn enabled() -> bool { ACTIVE.with(|cell| cell.borrow().is_some()) || super::fullw_http::enabled() }
// Called only after the existing checked drain, while the actual output owners
// remain alive. No extra forward, alternate output mode or synthetic row.
pub(crate) fn forward(
    slot: SlotId, start: usize, input: &[u32], route: &str,
    logits: &[f32], vocab: usize, hidden: Option<&[f32]>, hidden_width: usize,
) -> Result<()> {
    super::fullw_http::forward(slot, start, input, route, logits, vocab, hidden, hidden_width)?;
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let Some(active) = active.as_mut() else { return Ok(()) };
        if slot != active.target_slot { return Ok(()); }
        ensure!(vocab > 0 && !logits.is_empty() && logits.len() % vocab == 0,
            "actual C2 logits extent is not full vocabulary");
        let hidden = hidden.unwrap_or(&[]);
        ensure!(hidden.is_empty() || (hidden_width > 0 && hidden.len() % hidden_width == 0),
            "actual C2 hidden extent mismatch");
        active.observations.forwards.push(Forward { slot: slot.0, start, input: input.to_vec(),
            route: route.into(), logits: logits.iter().map(|v| v.to_bits()).collect(),
            hidden: hidden.iter().map(|v| v.to_bits()).collect(), vocab, hidden_width });
        Ok(())
    })
}
pub(crate) fn decision(slot: SlotId, index: usize, actual: u32) -> Result<u32> {
    decision_for(slot,None,index,actual)
}
pub(crate) fn decision_for(slot:SlotId,ticket:Option<&str>,index:usize,actual:u32)->Result<u32> {
    if COMPARISON.with(|cell| cell.get()) { return Ok(actual); }
    if super::fresh_http::enabled() {super::fresh_http::require_owner(slot,ticket)?;}
    let actual = super::fullw_http::decision(slot, index, actual)?;
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let Some(active) = active.as_mut() else { return Ok(actual) };
        if slot != active.target_slot { return Ok(actual); }
        let token = if let Some(frozen) = &active.frozen {
            *frozen.get(index.checked_sub(active.decision_base).context("decision before frozen start")?)
                .context("actual decision exceeds independent frozen extent")?
        } else { actual };
        active.observations.decisions.push((index, token));
        Ok(token)
    })
}
pub(crate) fn stopping(slot:SlotId,ticket:Option<&str>,token:u32,cause:&str)->Result<()> {
    if !COMPARISON.with(|cell|cell.get()) && super::fresh_http::enabled() {
        super::fresh_http::stopping(slot,ticket,token,cause)?;
    }
    Ok(())
}
pub(crate) fn gpu_argmax(slot: SlotId, position: usize, token: u32) {
    super::fullw_http::argmax(slot, position, token);
    ACTIVE.with(|cell| {
        if let Some(active) = cell.borrow_mut().as_mut() {
            if slot != active.target_slot { return; }
            active.observations.gpu_argmax.push((position, token));
        }
    });
}
pub(super) fn transaction(cache: &HybridKvCache, slot: SlotId) -> Result<()> {
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let Some(active) = active.as_mut() else { return Ok(()) };
        if slot == active.target_slot {
            if let Some(writer) = active.writer.as_mut() {
                writer.capture(cache, slot, &active.published_ledger)?;
            }
        }
        Ok(())
    })
}
pub(crate) fn route(slot: SlotId, route: &'static str) {
    super::fullw_http::route(slot, route);
    ACTIVE.with(|cell| {
        if let Some(active) = cell.borrow_mut().as_mut() {
            if slot == active.target_slot { active.observations.route_events.push(route); }
        }
    });
}
#[path = "gpu_selector.rs"]
mod gpu_selector;
pub(crate) use gpu_selector::selector as gpu_selector;

pub(super) fn coordinate(agent: &str, turn: usize, row: usize, stage: Stage) -> Coordinate {
    Coordinate { logical_step: turn as u64, layer: 0, stage,
        agent: agent.into(), logical_row_start: row as u64, segment: 0 }
}
pub(super) fn capture_f32(store: &mut Store, at: Coordinate, values: &[f32], shape: Vec<usize>) -> Result<()> {
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    store.capture(at, Scalar::F32, shape, values.as_ptr() as usize as u64,
        bytes.len() as u64, 0, None, "production forward returned after its checked native drain", &bytes)
        .map_err(anyhow::Error::msg)
}
// Emit canonical decision rows, not a final-row proxy. Every planned row must
// have a source forward. Different verifier chunking is allowed only Class B.
pub(super) fn decision_rows(observed: &Observations, slot: u32, prompt: &[u32],
    frozen: &[u32]) -> Result<BTreeMap<usize, Vec<f32>>> {
    let mut result = BTreeMap::new();
    for (index, token) in &observed.decisions {
        ensure!(*index < frozen.len() && *token == frozen[*index], "teacher-force decision join failed");
        let position = prompt.len() - 1 + index;
        let mut found = None;
        for forward in &observed.forwards {
            if forward.slot != slot { continue; }
            let count = forward.logits.len() / forward.vocab;
            let first = forward.start + forward.input.len() - count;
            if position >= first && position < first + count {
                // Earlier verifier tails can use rejected proposals. Only a
                // forward whose entire consumed prefix is frozen is canonical.
                let matches = forward.input.iter().enumerate().all(|(i, token)| {
                    let p = forward.start + i;
                    p <= position && prompt.get(p).or_else(|| frozen.get(p.saturating_sub(prompt.len()))) == Some(token)
                        || p > position
                });
                if matches {
                    let offset = (position - first) * forward.vocab;
                    found = Some(forward.logits[offset..offset + forward.vocab].iter()
                        .map(|v| f32::from_bits(*v)).collect());
                }
            }
        }
        ensure!(result.insert(*index, found.context("missing actual full-vocabulary decision row")?).is_none(),
            "canonical decision was observed twice");
    }
    ensure!(result.len() == frozen.len() && (0..frozen.len()).all(|i| result.contains_key(&i)),
        "Incomplete: actual decisions differ from independent frozen extent");
    Ok(result)
}

pub(crate) fn publish_ledger(slot: SlotId, ledger: &[u32]) {
    ACTIVE.with(|cell| {
        if let Some(active) = cell.borrow_mut().as_mut() {
            if active.target_slot == slot {
                active.published_ledger.clear();
                active.published_ledger.extend_from_slice(ledger);
            }
        }
    });
}

/// The actual normalized projection row already observed at this checkpoint.
/// Equality diagnostics retain this witness; they do not execute a new head.
pub(crate) fn checkpoint_hidden(slot: SlotId, position: usize) -> Result<Vec<f32>> {
    ACTIVE.with(|cell| {
        let active = cell.borrow();
        let active = active.as_ref().context("checkpoint observation scope absent")?;
        for forward in active.observations.forwards.iter().rev().filter(|f| f.slot == slot.0) {
            ensure!(forward.vocab != 0 && forward.logits.len() % forward.vocab == 0,
                "checkpoint full-vocabulary extent invalid");
            let rows = forward.logits.len() / forward.vocab;
            ensure!(rows <= forward.input.len(), "checkpoint projected/input row mismatch");
            let first = forward.start + forward.input.len() - rows;
            if position < first || position >= first + rows { continue; }
            ensure!(forward.hidden_width != 0 && forward.hidden.len() == rows * forward.hidden_width,
                "checkpoint normalized hidden does not match actual projected rows");
            let offset = (position - first) * forward.hidden_width;
            return Ok(forward.hidden[offset..offset + forward.hidden_width].iter()
                .map(|bits| f32::from_bits(*bits)).collect());
        }
        anyhow::bail!("AcquisitionBlocked: actual normalized output-head checkpoint witness missing")
    })
}
