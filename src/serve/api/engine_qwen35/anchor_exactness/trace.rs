//! Test-only, thread-local acquisition. Observing a returned tensor never changes
//! its route, synchronization, or production decision. Frozen decisions are a
//! separate, explicitly labelled diagnostic mode; lifecycle tests never use it.
use super::*;
use std::{cell::RefCell, collections::BTreeMap};
use mlx_native::{DType, MlxBuffer};
use crate::inference::models::qwen35::kv_cache::FullAttnKvSlot;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct State {
    pub target: Vec<u32>,
    pub mtp: Option<u32>,
    pub parity: Vec<bool>,
    pub regions: Vec<Vec<u8>>,
}
// Shape descriptors place n_seqs last, but the owning snapshot/restore
// operations address contiguous per-slot spans (la_restore.rs:370-424,
// 563-585). Follow those physical semantics, not an inferred shape stride.
fn linear_slot_bytes(values: &[f32], slots: usize, slot: SlotId) -> Result<Vec<u8>> {
    ensure!(slots > 0 && (slot.0 as usize) < slots && values.len() % slots == 0,
        "linear slot extent mismatch");
    let width = values.len() / slots;
    let start = slot.0 as usize * width;
    Ok(values[start..start + width].iter().flat_map(|v| v.to_bits().to_le_bytes()).collect())
}
fn slot_region(buffer: &MlxBuffer, slot: SlotId, slots: usize) -> Result<Vec<u8>> {
    ensure!(buffer.dtype() == DType::F32 && buffer.shape().last() == Some(&slots),
        "linear observation requires allocator slot descriptor");
    linear_slot_bytes(buffer.as_logical_slice::<f32>()?, slots, slot)
}
pub(super) fn rows(buffer: &MlxBuffer, slot: SlotId, valid: u32) -> Result<Vec<u8>> {
    let shape = buffer.shape();
    ensure!(shape.len() == 4 && (slot.0 as usize) < shape[0] && valid as usize <= shape[2],
        "full-attention valid extent mismatch");
    let length = shape[1].checked_mul(valid as usize).and_then(|n| n.checked_mul(shape[3]))
        .and_then(|n| n.checked_mul(buffer.dtype().size_of())).context("cache row byte count overflow")?;
    let mut bytes = Vec::with_capacity(length);
    for head in 0..shape[1] {
        let first = (slot.0 as usize * shape[1] + head) * shape[2] * shape[3];
        let range = first..first + valid as usize * shape[3];
        ensure!(range.end.checked_mul(buffer.dtype().size_of()).is_some_and(|n| n <= buffer.data_byte_len()),
            "observable region exceeds logical extent");
        match buffer.dtype() {
            DType::U8 => bytes.extend_from_slice(&buffer.as_logical_slice::<u8>()?[range]),
            DType::F32 => bytes.extend(buffer.as_logical_slice::<f32>()?[range].iter()
                .flat_map(|value| value.to_bits().to_le_bytes())),
            _ => anyhow::bail!("C2 cache observation requires explicit U8/F32 representation"),
        }
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
        let mut regions = Vec::with_capacity(6 * (kv.full_attn.len() + usize::from(kv.mtp_slot.is_some()))
            + 2 * kv.linear_attn.len());
        let target = kv.full_attn.iter().map(|l| full(l, slot, &mut regions)).collect::<Result<Vec<_>>>()?;
        let mtp = kv.mtp_slot.as_ref().map(|l| full(l, slot, &mut regions)).transpose()?;
        let mut parity = Vec::with_capacity(kv.linear_attn.len());
        for layer in &kv.linear_attn {
            parity.push(layer.pp_flipped[slot.0 as usize]);
            // Scratch is not semantic state and is intentionally not restored.
            regions.push(slot_region(layer.conv_bufs_for_slot(slot).0, slot, kv.n_seqs as usize)?);
            regions.push(slot_region(layer.recurrent_bufs_for_slot(slot).0, slot, kv.n_seqs as usize)?);
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
    pub input_sha256: Option<String>,
    pub planned_sequence: Option<u64>,
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
enum Transactions {
    Disabled,
    Recording(super::transaction_trace::Writer),
    Finished,
    Failed,
}
struct Active {
    observations: Observations,
    frozen: Option<Vec<u32>>,
    decision_base: usize,
    target_slot: SlotId,
    prediction_origin: usize,
    transactions: Transactions,
    published_ledger: Vec<u32>,
    session: Option<super::registration::Session>,
    raw_ram: Option<super::live_ram::Reservation>,
}
#[path = "acquire.rs"]
pub(crate) mod acquire;
thread_local! { static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) }; }
pub(super) struct Scope;
impl Scope {
    pub(super) fn begin(frozen: Option<Vec<u32>>, decision_base: usize) -> Result<Self> {
        ACTIVE.with(|cell| {
            let mut active = cell.borrow_mut();
            ensure!(active.is_none(), "nested C2 acquisition");
            *active = Some(Active { observations: Observations::default(), frozen, decision_base,
                target_slot: SLOT, prediction_origin: 0, transactions: Transactions::Disabled,
                published_ledger: Vec::new(), session: None, raw_ram: None });
            Ok(Self)
        })
    }
    pub(super) fn observe_slot(&self, slot: SlotId) {
        ACTIVE.with(|cell| {
            if let Some(active) = cell.borrow_mut().as_mut() { active.target_slot = slot; }
        });
    }
    pub(super) fn begin_transactions(writer: super::transaction_trace::Writer, ledger: &[u32],
        prediction_origin: usize) -> Result<Self> {
        let scope = Self::begin(None, 0)?;
        let session = super::registration::Session::new(writer.context(), prediction_origin)?;
        ACTIVE.with(|cell| {
            let mut active = cell.borrow_mut();
            let active = active.as_mut().context("C2 scope absent")?;
            active.transactions = Transactions::Recording(writer);
            active.session = Some(session);
            active.prediction_origin = prediction_origin;
            active.published_ledger = ledger.to_vec();
            Ok::<_, anyhow::Error>(())
        })?;
        Ok(scope)
    }
    pub(super) fn publish(&self, ledger: &[u32]) {
        ACTIVE.with(|cell| {
            if let Some(active) = cell.borrow_mut().as_mut() { active.published_ledger = ledger.to_vec(); }
        });
    }
    pub(super) fn finish_transactions(&self) -> Result<()> {
        ACTIVE.with(|cell| {
            let mut active = cell.borrow_mut();
            let active = active.as_mut().context("C2 scope absent")?;
            // Failure is sticky, including a failed or repeated close. An
            // ignored error must never leave observations publishable.
            let writer = match std::mem::replace(&mut active.transactions, Transactions::Failed) {
                Transactions::Recording(writer) => writer,
                Transactions::Disabled => anyhow::bail!("Incomplete: transaction writer was not attached"),
                Transactions::Finished => anyhow::bail!("Incomplete: transaction capture already finished"),
                Transactions::Failed => anyhow::bail!("Incomplete: transaction capture previously failed"),
            };
            writer.finish()?;
            active.session.take().context("required acquisition session was not attached")?.finish()?;
            active.transactions = Transactions::Finished;
            Ok(())
        })
    }
    pub(super) fn inspect<T>(&self, f:impl FnOnce(&Observations)->T) -> Result<T> {
        ACTIVE.with(|cell| {
            let active=cell.borrow();
            Ok(f(&active.as_ref().context("C2 scope absent")?.observations))
        })
    }
    pub(super) fn take(&self) -> Result<Observations> {
        ACTIVE.with(|cell| {
            let mut active = cell.borrow_mut();
            let active = active.as_mut().context("C2 scope absent")?;
            ensure!(matches!(active.transactions, Transactions::Finished),
                "Incomplete: transaction capture is unfinished or failed");
            Ok(std::mem::take(&mut active.observations))
        })
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
        ensure!(route.len() <= 256, "raw route metadata exceeds its retained grant");
        let (planned_sequence, input_sha256) = if let Some(session) = active.session.as_mut() {
            let (sequence, digest) = session.forward(slot, start, input, vocab, hidden_width, logits.len(), hidden.len())?;
            (Some(sequence), Some(digest))
        } else { (None, None) };
        if active.session.is_none() {
            let bytes = logits.len().checked_add(hidden.len()).and_then(|n| n.checked_add(input.len()))
                .and_then(|n| n.checked_mul(4)).and_then(|n| n.checked_add(route.len()))
                .and_then(|n| n.checked_add(2 * std::mem::size_of::<Forward>()))
                .context("prefix raw observation allocation overflow")?;
            super::live_ram::observe_raw(&mut active.raw_ram, u64::try_from(bytes)?)?;
        }
        active.observations.forwards.push(Forward { slot: slot.0, start, input: input.to_vec(),
            route: route.into(), logits: logits.iter().map(|v| v.to_bits()).collect(),
            hidden: hidden.iter().map(|v| v.to_bits()).collect(), vocab, hidden_width, planned_sequence, input_sha256 });
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
        if let Some(session) = active.session.as_mut() { session.decision(index)?; }
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
pub(crate) fn gpu_argmax(slot: SlotId, position: usize, token: u32) -> Result<()> {
    super::fullw_http::argmax(slot, position, token);
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        if let Some(active) = active.as_mut() {
            if slot != active.target_slot { return Ok(()); }
            if let Some(session) = active.session.as_mut() { session.gpu_argmax(position)?; }
            active.observations.gpu_argmax.push((position, token));
        }
        Ok(())
    })
}
pub(super) fn transaction(cache: &HybridKvCache, slot: SlotId) -> Result<()> {
    if acquire::enabled() { return acquire::capture_state(cache, slot); }
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let Some(active) = active.as_mut() else { return Ok(()) };
        if slot == active.target_slot {
            let result = match &mut active.transactions {
                Transactions::Disabled => Ok(()),
                Transactions::Recording(_) => Err(anyhow::anyhow!("required state stream lost its acquisition session")),
                Transactions::Finished => Err(anyhow::anyhow!("Incomplete: transaction captured after finish")),
                Transactions::Failed => Err(anyhow::anyhow!("Incomplete: transaction capture previously failed")),
            };
            if result.is_err() {
                active.transactions = Transactions::Failed;
            }
            result?;
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
    capture_words(store, at, bytemuck::cast_slice(values), values.as_ptr() as usize as u64, shape)
}
pub(super) fn capture_bits(store: &mut Store, at: Coordinate, values: &[u32], shape: Vec<usize>) -> Result<()> {
    capture_words(store, at, bytemuck::cast_slice(values), values.as_ptr() as usize as u64, shape)
}
fn capture_words(store: &mut Store, at: Coordinate, native: &[u8], parent: u64, shape: Vec<usize>) -> Result<()> {
    let converted;
    let bytes = if cfg!(target_endian = "little") { native } else {
        converted = native.chunks_exact(4).flat_map(|word| word.iter().rev().copied()).collect::<Vec<_>>();
        converted.as_slice()
    };
    store.capture(at, Scalar::F32, shape, parent, bytes.len() as u64, 0, None,
        "production forward returned after its checked native drain", bytes).map_err(anyhow::Error::msg)
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

#[test]
fn c23_linear_observer_matches_actual_contiguous_slot_addressing() {
    // Four slots, three channels and two taps; every raw word is distinct.
    let values: Vec<f32> = (0..24).map(|n| f32::from_bits(0x3f000000 + n)).collect();
    for slot in 0..4 {
        let bytes = linear_slot_bytes(&values, 4, SlotId(slot)).unwrap();
        let start = slot as usize * 6;
        let expected: Vec<u8> = values[start..start + 6].iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        assert_eq!(bytes, expected);
        assert_eq!(bytes.len(), 3 * 2 * 4);
        if slot > 0 { assert_ne!(bytes, linear_slot_bytes(&values, 4, SlotId(slot - 1)).unwrap()); }
    }
    assert!(linear_slot_bytes(&values, 4, SlotId(4)).is_err());
    assert!(linear_slot_bytes(&values[..23], 4, SlotId(0)).is_err());
}

#[test]
fn c2_transaction_scope_rejects_missing_writer_and_keeps_failure_sticky() {
    let scope = Scope::begin(None, 0).unwrap();
    assert!(scope.take().is_err(), "a missing writer must not publish observations");
    assert!(scope.finish_transactions().is_err());
    assert!(scope.take().is_err(), "ignored close error must not publish observations");
    assert!(scope.finish_transactions().is_err());
}
