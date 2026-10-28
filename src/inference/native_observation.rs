//! Explicit HF-side native observation. Never controls inference or scheduling.
//! A caller supplies actual logical input spans; constructors do not infer rows.
use mlx_native::submission_observer::{ObservationScope, ObservationStatus};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
mod witness;
pub(crate) use witness::{input, single_input, witness_command_rms_norm, witness_rms_norm, Input};

static INSTALLED: AtomicBool = AtomicBool::new(false);
static ARMED: AtomicBool = AtomicBool::new(false);
static USED: AtomicBool = AtomicBool::new(false);
static NEXT: AtomicU64 = AtomicU64::new(1);
thread_local! {
    static OWNERS: RefCell<Vec<crate::serve::phase_trace::SlotState>> = const { RefCell::new(Vec::new()) };
    static CURRENT: RefCell<Option<ObservationScope>> = const { RefCell::new(None) };
    static SELECTED: RefCell<Vec<crate::serve::phase_trace::SlotState>> = const { RefCell::new(Vec::new()) };
    static EPOCH: Cell<u64> = const { Cell::new(0) };
    static ROLE: Cell<&'static str> = const { Cell::new("target") };
}

pub(crate) struct Role {
    previous: Option<&'static str>,
    _thread_bound: PhantomData<Rc<()>>,
}
impl Drop for Role {
    fn drop(&mut self) {
        if let Some(previous) = self.previous {
            ROLE.with(|value| value.set(previous));
        }
    }
}
/// Called by the actual target verification branches, never inferred from M.
pub(crate) fn target_verify() -> Role {
    let previous = ARMED
        .load(Ordering::Acquire)
        .then(|| ROLE.with(|value| value.replace("target_verify")));
    Role {
        previous,
        _thread_bound: PhantomData,
    }
}

pub(crate) fn installed() {
    INSTALLED.store(true, Ordering::Release);
}

/// Runs only on the actual owning worker, before the listener is exposed.
/// An empty checked command on its actual queue joins all prior same-queue
/// commands. This is not a proof about an unrelated hidden queue/device.
pub(crate) fn arm_on_worker(
    device: &mlx_native::MlxDevice,
    nonce: &str,
    quiescent: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        quiescent,
        "native diagnostic arm requires no admitted/pending slot work"
    );
    anyhow::ensure!(
        crate::serve::phase_trace::native_control_nonce().as_deref() == Some(nonce),
        "native diagnostic nonce mismatch"
    );
    device.command_encoder()?.commit_and_wait()?;
    let baseline = arm().map_err(anyhow::Error::msg)?;
    crate::serve::phase_trace::publish(
        "native_arm_ack",
        serde_json::json!({
        "nonce":nonce,"clock_id":"CLOCK_MONOTONIC","worker_queue_checked":true,
        "quiescent":true,"baseline":baseline,"scope":"one_fresh_phase_diagnostic"}),
    );
    Ok(())
}

/// One diagnostic case per fresh process. Load/warmup counters are retained.
/// Caller must establish actual worker quiescence before arming; no counter reset.
pub(crate) fn arm() -> Result<ObservationStatus, &'static str> {
    if !INSTALLED.load(Ordering::Acquire) || USED.swap(true, Ordering::AcqRel) {
        return Err("native diagnostic requires installed observer and fresh process");
    }
    let baseline = mlx_native::submission_observer::submission_observation_status();
    if baseline.invalid
        || baseline.dropped_events != 0
        || baseline.unknown_scopes != 0
        || baseline.pending_commands != 0
        || baseline.pending_command_limit != 1024
    {
        return Err("invalid pre-arrival native baseline");
    }
    crate::serve::phase_trace::publish("native_baseline", serde_json::json!(baseline));
    ARMED.store(true, Ordering::Release);
    Ok(baseline)
}

/// Closing does not assert completion. The auditor joins every submitted ID to
/// an actual checked drain and rejects outstanding/unknown/drop/status deltas.
pub(crate) fn finish() -> ObservationStatus {
    ARMED.store(false, Ordering::Release);
    let status = mlx_native::submission_observer::submission_observation_status();
    crate::serve::phase_trace::publish("native_final", serde_json::json!(status));
    if status.pending_commands != 0 || status.pending_command_limit != 1024 {
        crate::serve::phase_trace::invalidate("native submitted command ownership not retired");
    }
    status
}

pub(crate) fn owner_forward_eligibility(
    handle: crate::serve::scheduler::SlotHandle,
) -> Option<bool> {
    if !INSTALLED.load(Ordering::Relaxed) {
        return None;
    }
    OWNERS.with(|owners| {
        owners
            .borrow()
            .iter()
            .find(|owner| {
                owner.slot_id == handle.slot_id.0 && owner.generation == handle.generation
            })
            .and_then(|owner| owner.forward_eligibility)
    })
}

pub(crate) fn update_owners(owners: &[crate::serve::phase_trace::SlotState]) {
    if !INSTALLED.load(Ordering::Acquire) {
        return;
    }
    if owners.len() > 256 {
        crate::serve::phase_trace::invalidate("native owner bound exceeded");
        return;
    }
    OWNERS.with(|value| *value.borrow_mut() = owners.to_vec());
}

/// Constructed from the actual selected input, never from a scheduler width.
#[derive(Clone, Copy)]
pub(crate) struct InputSpan {
    pub slot: u32,
    pub start: u64,
    pub count: u32,
}

pub(crate) fn qwen_input(
    source: &'static str,
    slot: u32,
    count: usize,
    positions: &[i32],
) -> Input {
    if !ARMED.load(Ordering::Acquire) {
        return input(source, &[]);
    }
    let valid = count > 0
        && count.checked_mul(4) == Some(positions.len())
        && positions.first().is_some_and(|value| *value >= 0)
        && positions[..count]
            .iter()
            .enumerate()
            .all(|(index, value)| i64::from(*value) == i64::from(positions[0]) + index as i64);
    match (valid, u32::try_from(count)) {
        (true, Ok(count)) => input(
            source,
            &[InputSpan {
                slot,
                start: positions[0] as u64,
                count,
            }],
        ),
        _ => input(source, &[]),
    }
}

pub(crate) fn ordered_input(
    source: &'static str,
    total: usize,
    values: impl Iterator<Item = (u32, usize, usize)>,
) -> Input {
    if !ARMED.load(Ordering::Acquire) {
        return input(source, &[]);
    }
    let spans: Option<Vec<_>> = values
        .take(257)
        .map(|(slot, start, count)| {
            Some(InputSpan {
                slot,
                start: u64::try_from(start).ok()?,
                count: u32::try_from(count).ok()?,
            })
        })
        .collect();
    let spans = spans.filter(|spans| {
        spans.len() <= 256
            && spans
                .iter()
                .try_fold(0usize, |sum, span| sum.checked_add(span.count as usize))
                == Some(total)
    });
    input(source, spans.as_deref().unwrap_or(&[]))
}

pub(crate) fn packed_input(
    source: &'static str,
    total: usize,
    slots: &[crate::serve::multi_seq_kv::SlotId],
    starts: &[usize],
    lengths: &[usize],
    offsets: &[usize],
) -> Input {
    if !ARMED.load(Ordering::Acquire) {
        return input(source, &[]);
    }
    let mut next = 0usize;
    let valid = slots.len() == starts.len()
        && slots.len() == lengths.len()
        && slots.len() == offsets.len()
        && slots.len() <= 256
        && offsets.iter().zip(lengths).all(|(offset, length)| {
            let valid = *offset == next && *length > 0;
            if let Some(end) = next.checked_add(*length) {
                next = end;
                valid
            } else {
                false
            }
        })
        && next == total;
    if !valid {
        return input(source, &[]);
    }
    ordered_input(
        source,
        total,
        slots
            .iter()
            .zip(starts)
            .zip(lengths)
            .map(|((slot, start), length)| (slot.0, *start, *length)),
    )
}

/// Selected DS decode source checks sorted unique slot_indices before building
/// its cache array in that same order. Only that explicitly proved route calls
/// this adapter; arbitrary model arrays cannot use a desired scheduler width.
pub(crate) fn sorted_decode_lane_input(lane: usize, start: usize, count: usize) -> Input {
    if !ARMED.load(Ordering::Acquire) {
        return input("deepseek4.sorted_cache_lane", &[]);
    }
    let slot = SELECTED.with(|selected| {
        let values = selected.borrow();
        (values.len() == 4
            && lane < 4
            && values
                .windows(2)
                .all(|pair| pair[0].slot_id < pair[1].slot_id))
        .then(|| values[lane].slot_id)
    });
    match (slot, u64::try_from(start), u32::try_from(count)) {
        (Some(slot), Ok(start), Ok(count)) => input(
            "deepseek4.sorted_cache_lane",
            &[InputSpan { slot, start, count }],
        ),
        _ => input("deepseek4.sorted_cache_lane", &[]),
    }
}

pub(crate) fn sorted_prefill_lane_input(
    lane: usize,
    lanes: usize,
    start: usize,
    count: usize,
) -> Input {
    let source = "deepseek4.prefill.actual_cooperative_lane";
    if !ARMED.load(Ordering::Acquire) {
        return input(source, &[]);
    }
    let slot = SELECTED.with(|selected| {
        let values = selected.borrow();
        ((2..=4).contains(&lanes)
            && values.len() == lanes
            && lane < lanes
            && values
                .windows(2)
                .all(|pair| pair[0].slot_id < pair[1].slot_id))
        .then(|| values[lane].slot_id)
    });
    match (slot, u64::try_from(start), u32::try_from(count)) {
        (Some(slot), Ok(start), Ok(count)) => input(source, &[InputSpan { slot, start, count }]),
        _ => input(source, &[]),
    }
}

pub(crate) fn sorted_prefill_pack_input(
    total: usize,
    spans: impl Iterator<Item = (usize, usize)>,
) -> Input {
    let source = "deepseek4.prefill.actual_cooperative_pack";
    if !ARMED.load(Ordering::Acquire) {
        return input(source, &[]);
    }
    let spans: Vec<_> = spans.take(5).collect();
    let slots = SELECTED.with(|selected| {
        let values = selected.borrow();
        ((2..=4).contains(&spans.len())
            && values.len() == spans.len()
            && values
                .windows(2)
                .all(|pair| pair[0].slot_id < pair[1].slot_id))
        .then(|| values.iter().map(|value| value.slot_id).collect::<Vec<_>>())
    });
    if let Some(slots) = slots {
        ordered_input(
            source,
            total,
            slots
                .into_iter()
                .zip(spans)
                .map(|(slot, (start, count))| (slot, start, count)),
        )
    } else {
        input(source, &[])
    }
}

pub(crate) struct Invocation {
    previous: Option<Vec<crate::serve::phase_trace::SlotState>>,
    previous_epoch: u64,
    _thread_bound: PhantomData<Rc<()>>,
}
impl Drop for Invocation {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            SELECTED.with(|value| *value.borrow_mut() = previous);
            EPOCH.with(|value| value.set(self.previous_epoch));
        }
    }
}
/// Logical ownership only. This does not define a physical operation scope.
pub(crate) fn invocation(owners: Vec<crate::serve::phase_trace::SlotState>) -> Invocation {
    if !INSTALLED.load(Ordering::Acquire) {
        return Invocation {
            previous: None,
            previous_epoch: 0,
            _thread_bound: PhantomData,
        };
    }
    let bounded = if owners.len() <= 256 {
        owners
    } else {
        Vec::new()
    };
    let previous = SELECTED.with(|value| value.replace(bounded));
    let epoch = if ARMED.load(Ordering::Acquire) {
        next_id().unwrap_or(0)
    } else {
        0
    };
    let previous_epoch = EPOCH.with(|value| value.replace(epoch));
    Invocation {
        previous: Some(previous),
        previous_epoch,
        _thread_bound: PhantomData,
    }
}

fn next_id() -> Option<u64> {
    NEXT.fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
        value.checked_add(1)
    })
    .ok()
}

/// Fully observed native work with deliberately opaque logical membership.
/// Auxiliary scopes cannot establish forward progress or a physical batch.
fn auxiliary(epoch: u64) -> Option<ObservationScope> {
    let scope = ObservationScope {
        epoch,
        id: next_id()?,
    };
    crate::serve::phase_trace::publish(
        "scope_definition",
        serde_json::json!({
        "scope":scope,"operation":"opaque_auxiliary","membership":"OpaqueAuxiliary","rows":[]}),
    );
    Some(scope)
}

pub(crate) struct Operation {
    previous: Option<ObservationScope>,
    active: bool,
    _thread_bound: PhantomData<Rc<()>>,
}
impl Drop for Operation {
    fn drop(&mut self) {
        if self.active {
            CURRENT.with(|value| *value.borrow_mut() = self.previous);
        }
    }
}
impl Operation {
    fn inactive() -> Self {
        Self {
            previous: None,
            active: false,
            _thread_bound: PhantomData,
        }
    }
}

/// Private exact-witness scope. Missing operand/position/owner stays unmapped.
/// Nested scopes restore the caller; no request or GPU state is changed.
fn operation(name: &'static str, input: &[InputSpan], role: &'static str) -> Operation {
    if !ARMED.load(Ordering::Acquire) {
        return Operation::inactive();
    }
    let mut scope = None;
    if ARMED.load(Ordering::Acquire) {
        let rows = SELECTED.with(|selected| {
            OWNERS.with(|owners| resolve_selected(input, &selected.borrow(), &owners.borrow()))
        });
        if let Some(mut rows) = rows {
            if role == "target_verify" {
                for row in &mut rows {
                    row.phase = "verify";
                }
            }
            if let Some(id) = next_id() {
                let epoch = EPOCH.with(Cell::get);
                if epoch == 0 {
                    crate::serve::phase_trace::invalidate("witness without current invocation");
                }
                scope = (epoch != 0).then_some(ObservationScope { epoch, id });
                crate::serve::phase_trace::publish(
                    "scope_definition",
                    serde_json::json!({
                    "scope":scope,"operation":name,"membership":"Known","rows":rows}),
                );
            } else {
                crate::serve::phase_trace::invalidate("native scope identity exhausted");
            }
        }
    }
    let previous = CURRENT.with(|value| value.replace(scope));
    Operation {
        previous,
        active: true,
        _thread_bound: PhantomData,
    }
}

fn resolve(
    input: &[InputSpan],
    owners: &[crate::serve::phase_trace::SlotState],
) -> Option<Vec<crate::serve::phase_trace::PhysicalRow>> {
    if input.is_empty() || input.len() > 256 {
        return None;
    }
    for (index, span) in input.iter().enumerate() {
        let end = span.start.checked_add(u64::from(span.count))?;
        if input[..index].iter().any(|other| {
            other.slot == span.slot
                && other.start < end
                && span.start < other.start + u64::from(other.count)
        }) {
            return None;
        }
    }
    input
        .iter()
        .map(|span| {
            if span.count == 0 || span.start.checked_add(u64::from(span.count)).is_none() {
                return None;
            }
            let mut matches = owners
                .iter()
                .filter(|owner| owner.active && owner.slot_id == span.slot);
            let owner = matches.next()?;
            if matches.next().is_some() {
                return None;
            }
            Some(crate::serve::phase_trace::PhysicalRow {
                request_id: owner.request_id,
                slot_id: span.slot,
                generation: owner.generation,
                token_position: span.start,
                token_count: span.count,
                phase: if owner.phase == "prefill" {
                    "prefill"
                } else {
                    "ordinary"
                },
            })
        })
        .collect()
}

fn resolve_selected(
    input: &[InputSpan],
    selected: &[crate::serve::phase_trace::SlotState],
    owners: &[crate::serve::phase_trace::SlotState],
) -> Option<Vec<crate::serve::phase_trace::PhysicalRow>> {
    if selected.iter().any(|old| {
        !owners.iter().any(|now| {
            now.slot_id == old.slot_id
                && now.generation == old.generation
                && now.request_id == old.request_id
                && now.active
        })
    }) {
        None
    } else {
        resolve(input, selected)
    }
}

/// Explicit constructor adapter: missing current scope stays Unknown. An
/// observation error never changes the successful native constructor result.
pub(crate) fn encoder(mut encoder: mlx_native::CommandEncoder) -> mlx_native::CommandEncoder {
    if ARMED.load(Ordering::Acquire) {
        let current = CURRENT.with(|value| *value.borrow());
        let epoch = EPOCH.with(Cell::get);
        let epoch = if epoch != 0 {
            epoch
        } else {
            next_id().unwrap_or(0)
        };
        let rotation_scope = auxiliary(epoch);
        let current = current.or(rotation_scope);
        if encoder.enable_submission_observation(epoch).is_err()
            || encoder
                .set_observation_rotation_scope(rotation_scope)
                .is_err()
        {
            crate::serve::phase_trace::invalidate("native constructor observation refused");
        } else {
            encoder.set_observation_scope(current);
        }
    }
    encoder
}

pub(crate) fn session<'a>(
    mut session: mlx_native::graph::GraphSession<'a>,
) -> mlx_native::graph::GraphSession<'a> {
    if ARMED.load(Ordering::Acquire) {
        attach(session.encoder_mut());
    }
    session
}

pub(crate) fn optional_session(
    mut session: Option<mlx_native::EncoderSession>,
) -> Option<mlx_native::EncoderSession> {
    if ARMED.load(Ordering::Acquire) {
        if let Some(value) = &mut session {
            attach(value.encoder());
        }
    }
    session
}

fn attach(encoder: &mut mlx_native::CommandEncoder) {
    let current = CURRENT.with(|value| *value.borrow());
    let epoch = EPOCH.with(Cell::get);
    let epoch = if epoch != 0 {
        epoch
    } else {
        next_id().unwrap_or(0)
    };
    let rotation_scope = auxiliary(epoch);
    let current = current.or(rotation_scope);
    if encoder.enable_submission_observation(epoch).is_err()
        || encoder
            .set_observation_rotation_scope(rotation_scope)
            .is_err()
    {
        crate::serve::phase_trace::invalidate("native session observation refused");
    } else {
        encoder.set_observation_scope(current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_row_resolution_refuses_ambiguous_stale_missing_or_overflow() {
        let owner = crate::serve::phase_trace::SlotState {
            request_id: 7,
            slot_id: 2,
            generation: 9,
            phase: "decode",
            active: true,
            forward_eligibility: Some(true),
            runnable_forward: None,
            runnable_emit: None,
            inflight: None,
            blocked: None,
        };
        let input = [InputSpan {
            slot: 2,
            start: 123,
            count: 4,
        }];
        let rows = resolve(&input, &[owner.clone()]).unwrap();
        assert_eq!(
            (rows[0].request_id, rows[0].generation, rows[0].token_count),
            (7, 9, 4)
        );
        assert!(resolve(&input, &[]).is_none());
        assert!(resolve(&input, &[owner.clone(), owner.clone()]).is_none());
        assert!(resolve(
            &[InputSpan {
                slot: 2,
                start: u64::MAX,
                count: 2
            }],
            &[owner.clone()]
        )
        .is_none());
        assert!(resolve(&[input[0], input[0]], &[owner.clone()]).is_none());
        let mut recycled = owner.clone();
        recycled.generation += 1;
        assert!(resolve_selected(&input, &[owner], &[recycled]).is_none());
    }
    #[test]
    fn nested_unknown_scope_restores_previous_without_cross_thread_transfer() {
        let scope = ObservationScope { epoch: 5, id: 8 };
        CURRENT.with(|value| value.replace(Some(scope)));
        {
            let previous = CURRENT.with(|value| value.replace(None));
            let _guard = Operation {
                previous,
                active: true,
                _thread_bound: PhantomData,
            };
            CURRENT.with(|value| assert_eq!(*value.borrow(), None));
        }
        CURRENT.with(|value| {
            assert_eq!(*value.borrow(), Some(scope));
            value.replace(None);
        });
    }
}
