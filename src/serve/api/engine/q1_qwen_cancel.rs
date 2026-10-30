//! Q1 test-only adapter around the REAL owning-engine advance/recovery path.
//! Closing the receiver occurs after a successful checked native advance,
//! before its production Pending/Ready cancellation branch. No reset shortcut.
use super::*;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
struct Hook {
    end: usize,
    close: Box<dyn FnOnce()>,
    fired: Rc<Cell<bool>>,
}
thread_local! {static AFTER_ADVANCE:RefCell<Option<Hook>>=const{RefCell::new(None)};}
struct Armed {
    fired: Rc<Cell<bool>>,
}
impl Drop for Armed {
    fn drop(&mut self) {
        AFTER_ADVANCE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}
fn arm(end: usize, close: impl FnOnce() + 'static) -> Result<Armed> {
    let fired = Rc::new(Cell::new(false));
    AFTER_ADVANCE.with(|slot| {
        let mut slot = slot.borrow_mut();
        anyhow::ensure!(slot.is_none(), "nested Q1 owning-worker cancel hook");
        *slot = Some(Hook {
            end,
            close: Box::new(close),
            fired: fired.clone(),
        });
        Ok(())
    })?;
    Ok(Armed { fired })
}
/// Called by advance_qwen35_prefill ONLY after the real state.advance returns.
/// An error is not a known drain and must never trigger a cancellation credit.
pub(super) fn after_actual_advance(success: bool, cursor: Option<usize>) {
    if !success {
        return;
    }
    AFTER_ADVANCE.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            assert_eq!(
                cursor,
                Some(hook.end),
                "Q1 cancellation injection reached wrong checked cursor"
            );
            hook.fired.set(true);
            (hook.close)();
        }
    });
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Phase {
    Checkpoint64,
    Recovered64,
}
#[derive(serde::Serialize)]
pub(crate) struct Evidence {
    pub source_route: &'static str,
    pub slot: u32,
    pub generation: u64,
    pub checked_prefix: usize,
    pub physical_tail_end_before_recovery: usize,
    pub recovered_cursor: usize,
    pub retained_tokens: usize,
    pub client_closed_after_actual_advance: bool,
    pub scheduler_in_flight_after: u32,
    pub supervisor_healthy: bool,
    pub full_144_token_trajectory: bool,
}
/// Source-owning caller supplies an already-loaded, freshly provisioned model.
/// Callback can stream the complete active cache at both known boundaries.
pub(crate) fn run_loaded(
    model: &mut super::super::engine_qwen35::Qwen35LoadedModel,
    prompt: &[u32],
    mut params: SamplingParams,
    mut observe: impl FnMut(
        Phase,
        &crate::inference::models::qwen35::kv_cache::HybridKvCache,
    ) -> Result<()>,
) -> Result<Evidence> {
    anyhow::ensure!(
        prompt.len() == 128,
        "Q1 cancellation prompt must be exactly128 tokens"
    );
    anyhow::ensure!(
        model.model.kv_graft.is_none(),
        "Q1 cancellation adapter has no graft scope"
    );
    anyhow::ensure!(
        model.tq_kv_active,
        "selected Q1 Qwen cache must be actual TQ"
    );
    anyhow::ensure!(
        params.max_tokens == 8192,
        "Q1 configured output allowance must remain8192"
    );
    params.stable_prompt_prefix_tokens = Some(64);
    let mut guard = Qwen35KvGuard::take(model)?;
    let kv = guard
        .kv
        .as_mut()
        .context("Q1 actual persistent KV absent")?;
    anyhow::ensure!(
        kv.n_seqs == 1,
        "Q1 cancellation localizer uses one independently owned physical slot"
    );
    anyhow::ensure!(
        kv.sequence_len_for_slot(SlotId(0))? == 0,
        "Q1 cancellation requires a fresh slot"
    );
    let state = super::super::engine_qwen35::Qwen35PrefillState::begin(
        prompt.to_vec(),
        params,
        None,
        kv,
        SlotId(0),
        0,
        None,
        None,
        None,
        guard.model.hidden_size,
        None,
    )?;
    let mut scheduler = InflightBatchedScheduler::new(1, 1);
    let admitted = scheduler
        .admit(AdmitRequest {
            prompt_tokens: 128,
            max_tokens: 8192,
            kv_bytes_needed: 0,
            prompt_kv_bytes: 0,
            preferred_slot: None,
        })
        .map_err(|error| anyhow::anyhow!("Q1 admission refused: {error:?}"))?;
    let handle = admitted.handle.context("Q1 slot was queued unexpectedly")?;
    let (send, receive) = oneshot::channel();
    let mut slots = vec![Some((
        Qwen35SlotWork::Prefill(state),
        SlotReply::Unary(send),
        handle,
    ))];
    let mut retained = vec![Qwen35RetainedPrefix::default()];
    let mut caches = vec![PromptCache::new()];
    let mut anchors = vec![None];
    let stats = Arc::new(Mutex::new(scheduler.stats()));
    let supervisor = EngineSupervisor::new();
    let first = advance_qwen35_prefill(
        &mut guard,
        &mut scheduler,
        &mut slots,
        &mut retained,
        &mut caches,
        &mut anchors,
        None,
        &stats,
        handle,
        64,
        0,
        &supervisor,
    );
    if let Some(fatal) = first {
        return Err(fatal.error.context("Q1 actual first engine advance failed"));
    }
    anyhow::ensure!(
        slots[0].is_some(),
        "Q1 first advance did not remain pending"
    );
    anyhow::ensure!(
        anchors[0]
            .as_ref()
            .is_some_and(|a| a.prompt_tokens == prompt[..64]),
        "real stable checkpoint64 missing"
    );
    anyhow::ensure!(
        guard
            .kv
            .as_ref()
            .unwrap()
            .sequence_len_for_slot(handle.slot_id)?
            == 64,
        "real first checked cursor"
    );
    observe(Phase::Checkpoint64, guard.kv.as_ref().unwrap())?;
    // Drop the ACTUAL response receiver only after the next checked advance.
    let armed = arm(128, move || drop(receive))?;
    let second = advance_qwen35_prefill(
        &mut guard,
        &mut scheduler,
        &mut slots,
        &mut retained,
        &mut caches,
        &mut anchors,
        None,
        &stats,
        handle,
        64,
        0,
        &supervisor,
    );
    if let Some(fatal) = second {
        return Err(fatal
            .error
            .context("Q1 cancellation recovery failed closed"));
    }
    anyhow::ensure!(
        armed.fired.get(),
        "actual post-advance cancellation hook never fired"
    );
    anyhow::ensure!(
        slots[0].is_none(),
        "cancelled engine work remained installed"
    );
    let recovered = guard
        .kv
        .as_ref()
        .unwrap()
        .sequence_len_for_slot(handle.slot_id)? as usize;
    anyhow::ensure!(
        recovered == 64 && retained[0].tokens == prompt[..64],
        "real recovery failed cursor/token-ledger join"
    );
    anyhow::ensure!(
        supervisor.is_healthy(),
        "Q1 cancellation is not GPU failure recovery"
    );
    observe(Phase::Recovered64, guard.kv.as_ref().unwrap())?;
    let in_flight = scheduler.stats().in_flight_slots;
    anyhow::ensure!(in_flight == 0, "cancelled reservation not released");
    Ok(Evidence{source_route:"advance_qwen35_prefill actual Ready/client_closed/recover_qwen35_slot_after_cancellation_for_reply",
        slot:handle.slot_id.0,generation:handle.generation,checked_prefix:64,physical_tail_end_before_recovery:128,
        recovered_cursor:recovered,retained_tokens:retained[0].len(),client_closed_after_actual_advance:armed.fired.get(),
        scheduler_in_flight_after:in_flight,supervisor_healthy:true,full_144_token_trajectory:false})
}
#[test]
fn q1_cancel_injection_requires_actual_success_and_fires_exactly_once() {
    let called = Rc::new(Cell::new(0));
    let counter = called.clone();
    let armed = arm(128, move || counter.set(counter.get() + 1)).unwrap();
    after_actual_advance(false, Some(64));
    assert_eq!(called.get(), 0);
    assert!(!armed.fired.get());
    after_actual_advance(true, Some(128));
    after_actual_advance(true, Some(128));
    assert_eq!(called.get(), 1);
    assert!(armed.fired.get());
}
#[test]
fn q1_cancel_hook_cleanup_removes_unfired_action() {
    let called = Rc::new(Cell::new(0));
    let counter = called.clone();
    {
        let _armed = arm(128, move || counter.set(1)).unwrap();
    }
    after_actual_advance(true, Some(128));
    assert_eq!(called.get(), 0);
}

#[path = "q1_qwen_cancel/actual.rs"]
mod actual;
#[test]
#[ignore = "root-owned actual Qwen model and resource packet; source-only until serialized gate"]
fn q1_qwen_actual_engine_cancel_restore_cache() {
    actual::run();
}

#[path = "q1_qwen_cancel/continuation.rs"]
mod continuation;
