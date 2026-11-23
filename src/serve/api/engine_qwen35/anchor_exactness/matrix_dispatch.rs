//! Test-owned serving driver. All partitioning, verified-output draining and
//! settlement belong to decode_batch_qwen35, never to a second dispatcher.
use super::*;
use std::{cell::RefCell, ops::{Deref, DerefMut}};
use crate::serve::{api::engine::{self, PromptCache, Qwen35KvGuard, Qwen35RetainedPrefix,
    Qwen35Slot, Qwen35SlotWork, SlotReply},
    scheduler::{AdmitRequest, InflightBatchedScheduler, Scheduler, SchedulerStep}};

pub(in super::super) struct Cache(Option<HybridKvCache>);
impl Cache { pub(super) fn new(cache: HybridKvCache) -> Self { Self(Some(cache)) } }
impl Deref for Cache {
    type Target = HybridKvCache;
    fn deref(&self) -> &Self::Target { self.0.as_ref().expect("matrix cache is held by serving guard") }
}
impl DerefMut for Cache {
    fn deref_mut(&mut self) -> &mut Self::Target { self.0.as_mut().expect("matrix cache is held by serving guard") }
}
thread_local! { static COMPLETED: RefCell<Option<Vec<(SlotId, Vec<u32>)>>> = const { RefCell::new(None) }; }
// Called at the real successful settlement boundary, before finish consumes state.
pub(crate) fn completed(slot: SlotId, tokens: &[u32]) {
    COMPLETED.with(|cell| { if let Some(rows) = cell.borrow_mut().as_mut() {
        rows.push((slot, tokens.to_vec()));
    }});
}
struct CompletionScope;
impl Drop for CompletionScope { fn drop(&mut self) { COMPLETED.with(|cell| { cell.borrow_mut().take(); }); } }
pub(super) struct Outcome {
    pub(super) tokens: Vec<(SlotId, Vec<u32>)>,
    pub(super) retained: Option<Qwen35RetainedPrefix>,
}
pub(super) struct Ready {
    pub(super) state: Qwen35DecodeState,
    pub(super) seed_logits: Option<Vec<f32>>,
}
type Receiver = tokio::sync::oneshot::Receiver<Result<engine::GenerationResult>>;
pub(super) struct Driver {
    scheduler: InflightBatchedScheduler,
    handles: Vec<Option<crate::serve::scheduler::SlotHandle>>,
    reservations: Vec<Option<(usize, usize)>>,
    slots: Vec<Option<Qwen35Slot>>,
    receivers: Vec<Option<Receiver>>,
    retained: Vec<Qwen35RetainedPrefix>,
    prompts: Vec<PromptCache>,
    anchors: Vec<Qwen35AnchorStore>,
}
impl Driver {
    pub(super) fn new() -> Result<Self> { Ok(Self {
        scheduler: InflightBatchedScheduler::new(SLOTS, SLOTS),
        handles: vec![None; SLOTS as usize],
        reservations: vec![None; SLOTS as usize],
        slots: (0..SLOTS).map(|_| None).collect(),
        receivers: (0..SLOTS).map(|_| None).collect(),
        retained: vec![Qwen35RetainedPrefix::default(); SLOTS as usize],
        prompts: vec![PromptCache::new(); SLOTS as usize],
        anchors: (0..SLOTS).map(|_| Qwen35AnchorStore::new(4)).collect::<Result<Vec<_>>>()?,
    }) }
    pub(super) fn admit(&mut self, slot: SlotId, prompt: usize, limit: usize) -> Result<u64> {
        let index = slot.0 as usize;
        let handle = if let Some(handle) = self.handles[index] {
            ensure!(self.slots[index].is_none(), "admitting over a live decode lane");
            ensure!(self.reservations[index] == Some((prompt, limit)),
                "prefill changed its already admitted request dimensions");
            handle
        } else {
            let handle = self.scheduler.admit(AdmitRequest { prompt_tokens: prompt as u32,
                max_tokens: limit as u32, preferred_slot: Some(slot), ..Default::default() })
                .map_err(|error| anyhow::anyhow!("matrix scheduler admission failed: {error:?}"))?
                .handle.context("matrix scheduler queued an available lane")?;
            ensure!(handle.slot_id == slot, "scheduler did not retain the requested physical slot");
            self.handles[index] = Some(handle);
            self.reservations[index] = Some((prompt, limit));
            handle
        };
        Ok(engine::test_qwen35_operator_request_id(handle))
    }
    pub(super) fn set_prefix(&mut self, slot: SlotId, tokens: &[u32],
        spec: Option<&Qwen35SpecPrefixBoundary>) {
        self.retained[slot.0 as usize] = Qwen35RetainedPrefix { tokens: tokens.to_vec(), spec: spec.cloned() };
    }
    pub(super) fn active(&self, slot: SlotId) -> bool { self.slots[slot.0 as usize].is_some() }
    pub(super) fn cancel(&mut self, slot: SlotId) {
        let index = slot.0 as usize;
        self.slots[index] = None;
        self.receivers[index] = None;
        self.reservations[index] = None;
        if let Some(handle) = self.handles[index].take() { self.scheduler.release(handle); }
    }
    fn collect_replies(&mut self) -> Result<()> {
        for index in 0..SLOTS as usize {
            if self.slots[index].is_none() {
                if let Some(mut receiver) = self.receivers[index].take() {
                    receiver.try_recv().context("settlement omitted its reply")??;
                    self.handles[index] = None;
                    self.reservations[index] = None;
                }
            }
        }
        Ok(())
    }
    pub(super) fn run(&mut self, fixture: &mut Fixture, cache: &mut Cache,
        states: Vec<Ready>, store: &mut Qwen35AnchorStore,
        turn_limit: Option<usize>) -> Result<Outcome> {
        COMPLETED.with(|cell| -> Result<()> {
            ensure!(cell.borrow().is_none(), "nested matrix serving completion scope");
            *cell.borrow_mut() = Some(Vec::new()); Ok(())
        })?;
        let _scope = CompletionScope;
        let supervisor = fixture.supervisor.clone();
        let joint = fixture.joint;
        let qwen = fixture.qwen();
        let target = states.iter().any(|ready| ready.state.slot_id == SLOT);
        let mut seeds = Vec::new();
        for Ready { state, seed_logits } in states {
            let index = state.slot_id.0 as usize;
            let handle = self.handles[index].context("decode state lacks its pre-capture admission")?;
            ensure!(self.slots[index].is_none(), "installing over live serving work");
            self.scheduler.advance_after_prefill(handle, state.prompt_cache_identity().0.len() as u32);
            let (sender, receiver) = tokio::sync::oneshot::channel();
            self.receivers[index] = Some(receiver);
            let reply = SlotReply::Unary(sender);
            if state.finished_at_seed() {
                seeds.push((handle, state, seed_logits.context("finished seed lacks its actual prefill logits")?, reply));
            } else {
                ensure!(seed_logits.is_none(), "ordinary decode unexpectedly retained a seed-only settlement");
                self.slots[index] = Some((Qwen35SlotWork::Decode(state), reply, handle));
            }
        }
        std::mem::swap(store, &mut self.anchors[SLOT.0 as usize]);
        let previous_model_cache = qwen.persistent_kv_cache.take();
        let mut guard = Qwen35KvGuard { model: qwen, kv: cache.0.take() };
        let result = (|| -> Result<Outcome> {
            let seeded = !seeds.is_empty();
            for (handle, state, logits, reply) in seeds {
                if let Some(fatal) = engine::test_finish_qwen35_prefill(&mut guard, &mut self.scheduler,
                    &mut self.slots, &mut self.retained, &mut self.prompts, &mut self.anchors, None,
                    handle, state, logits, reply, 0, &supervisor) { return Err(fatal.error); }
            }
            self.collect_replies()?;
            if seeded {
                trace::publish_ledger(SLOT, &self.retained[SLOT.0 as usize].tokens);
                trace::transaction(guard.kv.as_ref().expect("live serving KV"), SLOT)?;
            }
            for _ in 0..turn_limit.unwrap_or(DECODE_LIMIT + 1) {
                if target && self.slots[SLOT.0 as usize].is_none() { break; }
                if self.slots.iter().all(Option::is_none) { break; }
                let handles = match self.scheduler.step()
                    .map_err(|error| anyhow::anyhow!("matrix scheduler step failed: {error:?}"))? {
                    SchedulerStep::Decode { handles } => handles,
                    SchedulerStep::Mixed { decode_handles, .. } => decode_handles,
                    _ => anyhow::bail!("live matrix decoders were omitted from the scheduler turn"),
                };
                if let Some(fatal) = engine::decode_batch_qwen35(&mut guard, &mut self.scheduler,
                    &mut self.slots, &mut self.retained, &mut self.prompts, &mut self.anchors,
                    None, &handles, 0, joint, &supervisor) { return Err(fatal.error); }
                trace::publish_ledger(SLOT, &self.retained[SLOT.0 as usize].tokens);
                for slot in 0..SLOTS {
                    trace::transaction(guard.kv.as_ref().expect("live serving KV"), SlotId(slot))?;
                }
                self.collect_replies()?;
                if target && self.slots[SLOT.0 as usize].is_none() { break; }
            }
            if turn_limit.is_none() {
                ensure!(target && self.slots[SLOT.0 as usize].is_none(), "target request failed to complete");
            }
            let tokens = COMPLETED.with(|cell| cell.borrow_mut().as_mut()
                .expect("completion scope").drain(..).collect());
            let retained = (target && self.slots[SLOT.0 as usize].is_none())
                .then(|| self.retained[SLOT.0 as usize].clone());
            Ok(Outcome { tokens, retained })
        })();
        cache.0 = guard.kv.take();
        std::mem::swap(store, &mut self.anchors[SLOT.0 as usize]);
        drop(guard);
        fixture.qwen().persistent_kv_cache = previous_model_cache;
        result
    }
}
