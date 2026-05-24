//! Scheduler trait + FIFO adapter + InflightBatched signature stub
//! (ADR-040 Phase B iter-1 scaffolding).
//!
//! This module is the **pure data primitive** that ADR-040 Phase C (iter-2)
//! wires into `serve::api::engine::Engine`. At iter-1 it contains *no* engine
//! load, *no* GPU code, *no* `AppState` wiring — those land in Phase C. The
//! pattern mirrors `serve::multi_model` (W74 iter-206): a synthetic-fixture-
//! tested data structure that later iters glue into the live serve path.
//!
//! # What this module does (iter-1)
//!
//! - Declares the `Scheduler` trait surface (`admit`, `step`, `release`,
//!   `stats`, `policy`) — see ADR-040 §3.2 + AC-2.
//! - Ships `FifoSchedulerAdapter` — the **byte-equivalent** wrapper of the
//!   existing ADR-005 Phase 2 Decision #2 contract (one in-flight request,
//!   bounded queue, 429 on overflow). Iter-2 pins this contract with a
//!   regression test against `Engine::spawn`.
//! - Ships `InflightBatchedScheduler` with real `admit` + `release` + `stats`
//!   semantics; `step` returns `Err(StepError::NotImplemented)` as a typed
//!   iter-1 contract that Phase B iter-3 replaces with the admission-during-
//!   decode loop (mirrors llama.cpp `-cb`; see ADR-040 §3.3).
//!
//! # What this module does NOT do (iter-1)
//!
//! - Hold an `Engine` (or any `mlx_native` buffers) — `RequestSlot` is a
//!   pure descriptor. Phase C iter-2's `Engine::spawn` will accept an
//!   injected `Box<dyn Scheduler>` and dispatch against this trait.
//! - Touch `serve::api::engine`, `serve::mod::cmd_serve`, or any handler —
//!   wiring is Phase C.
//! - Build paged-KV blocks — ADR-040 §3.1 picks `SeparateSlots` first.
//! - Drive forward passes — Phase B iter-4 threads `slot_id` through
//!   `serve::forward_prefill` + `serve::forward_prefill_batched`.
//!
//! # Backward-compat contract (ADR-040 §3.6)
//!
//! `FifoSchedulerAdapter` MUST behave bit-equivalently to the pre-ADR-040
//! `Engine::spawn` channel + worker thread. Specifically:
//!
//! 1. At most one in-flight request (Decision #2 — `max_slots == 1`).
//! 2. Bounded queue with capacity = `queue_capacity` from `Engine::spawn`
//!    (Decision #19 — channel buffer at `queue_capacity.max(1)`).
//! 3. Overflow returns `AdmitError::QueueFull`, which the handler layer
//!    maps to HTTP 429 + `Retry-After: 1` (`schema::ApiError::queue_full`).
//! 4. FIFO ordering preserved: pop order == push order.
//!
//! `InflightBatchedScheduler` opts INTO admission-during-decode; both
//! schedulers are first-class production paths — `FifoSerial` is the
//! Phase E1 default until the benchmark gate (§3.4 + AC-4) flips it.
//!
//! # Reference lineage
//!
//! - Trait shape: ADR-040 §2.1 + AC-2 (`src/serve/scheduler.rs ~400 LOC`).
//! - FIFO contract: `src/serve/api/engine.rs:1-32` (module docstring) +
//!   `src/serve/api/engine.rs:2296` (`Engine::spawn(loaded, queue_capacity,
//!   kv_cache_budget_bytes)`).
//! - 429 mapping: `src/serve/api/schema.rs:108-120` (`ApiError::queue_full`).
//! - Pattern model: `src/serve/multi_model.rs` (W74 iter-206 — pure-data
//!   primitive with inline tests, no production callsite).
//!
//! # Tests
//!
//! Synthetic-fixture unit tests cover:
//!
//! - FIFO contract preservation (the load-bearing tests — these pin
//!   Decision #2 + Decision #19 byte-equivalence at the trait surface).
//! - InflightBatched signature gate (proves the type is wired but
//!   `step` is stubbed at iter-1; the explicit `NotImplemented` assert
//!   is removed when Phase B iter-3 lands the real implementation).
//! - Cross-cutting: monotone admit timestamps, `QueueFull` debug shape.

use std::collections::VecDeque;
use std::time::Instant;

// ---------------------------------------------------------------------------
// SlotId — re-exported from Phase A iter-1.
// ---------------------------------------------------------------------------
//
// Phase B iter-1 originally defined a local `SlotId` newtype because Phase A
// iter-1 was in flight in parallel; the orchestrator integration pass after
// both phases landed (2026-05-23) swapped the local definition for this
// re-export of `crate::serve::multi_seq_kv::SlotId`. The shape was identical
// (`pub struct SlotId(pub u32)`) so the swap was mechanical.

pub use crate::serve::multi_seq_kv::SlotId;

// ---------------------------------------------------------------------------
// Scheduler policy (ADR-040 §3.2)
// ---------------------------------------------------------------------------

/// Which scheduling discipline an `Engine` uses (ADR-040 §3.2).
///
/// `FifoSerial` is the Phase E1 default (byte-equivalent to pre-ADR-040
/// `Engine::spawn`). `InflightBatched` is the new admission-during-decode
/// policy that ADR-040 §3.4 ramps to default-on after the AC-4 benchmark gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerPolicy {
    /// One in-flight request, bounded queue, 429 on overflow.
    /// Mirrors `Engine::spawn`'s mpsc-channel + single-worker semantics.
    FifoSerial,
    /// Admission-during-decode with up to `max_slots` concurrent requests.
    /// Mirrors llama.cpp `-cb` (ADR-040 §3.3 reference choice).
    InflightBatched,
}

// ---------------------------------------------------------------------------
// Slot descriptor + admit request
// ---------------------------------------------------------------------------

/// An admitted request's slot handle.
///
/// `RequestSlot` is the **pure descriptor** the scheduler hands back from
/// `admit`. The forward path (Phase B iter-4) uses `slot_id` to index into
/// the per-model multi-seq KV cache; `admitted_at` is the wall-clock start
/// time for TTFT accounting.
#[derive(Debug, Clone)]
pub struct RequestSlot {
    /// Per-model multi-seq KV slot index (Phase A iter-1 newtype).
    pub slot_id: SlotId,
    /// Wall-clock instant `admit()` returned `Ok` for this slot.
    pub admitted_at: Instant,
    /// Prompt-token count as passed to `admit()` (post-template, pre-prefill).
    pub prompt_tokens: u32,
    /// Maximum new tokens the request may emit (sampler stop budget).
    pub max_tokens: u32,
}

/// Caller-supplied request bookkeeping handed to `Scheduler::admit`.
///
/// `AdmitRequest` is intentionally *not* the OpenAI-shaped chat request —
/// the handler layer translates the user-facing payload into this pure
/// descriptor before calling into the scheduler trait so the scheduler is
/// reusable across `/v1/chat/completions`, `/v1/completions`, and future
/// endpoint shapes.
#[derive(Debug, Clone)]
pub struct AdmitRequest {
    /// Tokenized prompt length after chat-template rendering.
    pub prompt_tokens: u32,
    /// Maximum new tokens the request may emit.
    pub max_tokens: u32,
}

// ---------------------------------------------------------------------------
// Scheduler step variant (ADR-040 AC-2)
// ---------------------------------------------------------------------------

/// What the scheduler decided the next forward pass should do.
///
/// `Mixed` is the variant the `InflightBatched` policy emits when a prefill
/// for a newly-admitted slot coexists with ongoing decode for one or more
/// already-prefilled slots — see ADR-040 AC-2 (Phase B iter-6 wires this
/// into the forward path). `FifoSerial` only ever returns `Idle`, `Prefill`,
/// or single-slot `Decode`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerStep {
    /// No work available; the engine loop should park.
    Idle,
    /// Run prefill for one slot for `n_tokens` tokens.
    Prefill { slot_id: SlotId, n_tokens: u32 },
    /// Run decode for the listed slots (one forward, batched in slot dim).
    Decode { slots: Vec<SlotId> },
    /// Run prefill for one slot AND decode for the listed slots in one forward.
    Mixed {
        prefill: SlotId,
        n_prefill_tokens: u32,
        decode_slots: Vec<SlotId>,
    },
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why `Scheduler::admit` rejected a request.
///
/// `QueueFull` is the load-bearing variant — it carries `capacity` and
/// `in_flight` so the handler layer can render an accurate diagnostic
/// alongside the 429 + `Retry-After: 1` response. `SchedulerStopped` is
/// the post-shutdown sentinel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmitError {
    /// Queue + in-flight slots are at the configured cap. Maps to HTTP 429.
    QueueFull { capacity: u32, in_flight: u32 },
    /// Scheduler is no longer accepting work (post-shutdown).
    SchedulerStopped,
}

/// Why `Scheduler::step` failed.
///
/// `NotImplemented` is the typed iter-1 contract that
/// `InflightBatchedScheduler::step` returns; Phase B iter-3 replaces it
/// with the real admission-during-decode loop. This is **not** a stub —
/// it's a discriminant the test suite explicitly pins (see
/// `inflight_batched_step_returns_not_implemented_at_iter_1` below).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepError {
    /// The scheduler's `step` impl has not landed yet (iter-gated).
    NotImplemented,
    /// The underlying engine forward-pass returned an error.
    EngineFailed(String),
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Lifetime counters + current resident state, for `/metrics` + ops view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerStats {
    pub policy: SchedulerPolicy,
    pub in_flight_slots: u32,
    pub admitted_total: u64,
    pub rejected_429_total: u64,
    pub completed_total: u64,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// The ADR-040 §2.1 scheduler surface.
///
/// `Send` is required because the production `Engine` worker is a
/// `std::thread::JoinHandle` (see `src/serve/api/engine.rs:1-32`).
pub trait Scheduler: Send {
    /// Which discipline this scheduler implements.
    fn policy(&self) -> SchedulerPolicy;
    /// Admit a new request. Returns the slot handle on success or an
    /// `AdmitError::QueueFull` when the queue + in-flight set is at cap.
    fn admit(&mut self, req: AdmitRequest) -> Result<RequestSlot, AdmitError>;
    /// Decide what the next forward pass should do.
    fn step(&mut self) -> Result<SchedulerStep, StepError>;
    /// Drop a slot (completion, error, or client-disconnect path).
    fn release(&mut self, slot: SlotId);
    /// Snapshot of lifetime counters + current resident state.
    fn stats(&self) -> SchedulerStats;
}

// ---------------------------------------------------------------------------
// FifoSchedulerAdapter — byte-equivalent wrap of pre-ADR-040 Engine
// ---------------------------------------------------------------------------

/// Byte-equivalent wrap of the existing `Engine::spawn` FIFO contract
/// (Decision #2 + Decision #19).
///
/// `max_slots` is **always 1** under this policy — that is the load-bearing
/// invariant the regression suite pins. Reads:
///
/// - `queue_capacity` maps 1:1 to the `Engine::spawn(_, queue_capacity, _)`
///   mpsc channel buffer.
/// - `in_flight: Option<RequestSlot>` is the single worker-thread slot.
/// - `queue: VecDeque<RequestSlot>` is the bounded pending queue.
///
/// On `admit`: if `in_flight.is_none()`, the new slot becomes in-flight
/// directly; otherwise it enqueues. Overflow returns `QueueFull` —
/// `capacity` is the queue cap, `in_flight` is `1` when occupied.
///
/// On `step`: if `in_flight` exists, return `Prefill` for it (the FIFO
/// model preserves single-request prefill+decode-in-one-forward). Phase B
/// iter-2's regression test will assert byte-equivalence against the
/// pre-ADR-040 channel-driven path.
pub struct FifoSchedulerAdapter {
    queue_capacity: u32,
    queue: VecDeque<RequestSlot>,
    in_flight: Option<RequestSlot>,
    next_slot_id: u32,
    /// True after the first `step` call on an in-flight slot — switches
    /// the next `step` from `Prefill` to `Decode` for the same slot.
    /// Mirrors the existing channel+thread model where one request runs
    /// prefill then decode in a single forward sequence before the next
    /// request is dequeued.
    in_flight_prefilled: bool,
    admitted_total: u64,
    rejected_429_total: u64,
    completed_total: u64,
}

impl FifoSchedulerAdapter {
    /// Build a FIFO scheduler with the given queue cap.
    ///
    /// `queue_capacity` mirrors `Engine::spawn`'s mpsc buffer size — the
    /// total admissible backlog is `queue_capacity` queued + 1 in-flight,
    /// matching the pre-ADR-040 contract (the mpsc channel holds
    /// `queue_capacity` pending requests while the worker drains one).
    pub fn new(queue_capacity: u32) -> Self {
        Self {
            queue_capacity,
            queue: VecDeque::new(),
            in_flight: None,
            next_slot_id: 0,
            in_flight_prefilled: false,
            admitted_total: 0,
            rejected_429_total: 0,
            completed_total: 0,
        }
    }

    fn alloc_slot_id(&mut self) -> SlotId {
        let id = SlotId(self.next_slot_id);
        self.next_slot_id = self.next_slot_id.wrapping_add(1);
        id
    }

    fn in_flight_count(&self) -> u32 {
        if self.in_flight.is_some() { 1 } else { 0 }
    }
}

impl Scheduler for FifoSchedulerAdapter {
    fn policy(&self) -> SchedulerPolicy {
        SchedulerPolicy::FifoSerial
    }

    fn admit(&mut self, req: AdmitRequest) -> Result<RequestSlot, AdmitError> {
        // Bound check: queue_capacity pending + 1 in-flight is the cap.
        if self.queue.len() as u32 >= self.queue_capacity && self.in_flight.is_some() {
            self.rejected_429_total = self.rejected_429_total.saturating_add(1);
            return Err(AdmitError::QueueFull {
                capacity: self.queue_capacity,
                in_flight: self.in_flight_count(),
            });
        }

        let slot = RequestSlot {
            slot_id: self.alloc_slot_id(),
            admitted_at: Instant::now(),
            prompt_tokens: req.prompt_tokens,
            max_tokens: req.max_tokens,
        };
        self.admitted_total = self.admitted_total.saturating_add(1);

        if self.in_flight.is_none() {
            self.in_flight = Some(slot.clone());
            self.in_flight_prefilled = false;
        } else {
            self.queue.push_back(slot.clone());
        }
        Ok(slot)
    }

    fn step(&mut self) -> Result<SchedulerStep, StepError> {
        match &self.in_flight {
            None => Ok(SchedulerStep::Idle),
            Some(slot) => {
                if !self.in_flight_prefilled {
                    let step = SchedulerStep::Prefill {
                        slot_id: slot.slot_id,
                        n_tokens: slot.prompt_tokens,
                    };
                    self.in_flight_prefilled = true;
                    Ok(step)
                } else {
                    Ok(SchedulerStep::Decode {
                        slots: vec![slot.slot_id],
                    })
                }
            }
        }
    }

    fn release(&mut self, slot: SlotId) {
        // Match-by-id; unknown slot is a no-op (mirrors LoadedPool::touch
        // unknown-key noop pattern at multi_model.rs:1513).
        match &self.in_flight {
            Some(s) if s.slot_id == slot => {
                self.in_flight = None;
                self.in_flight_prefilled = false;
                self.completed_total = self.completed_total.saturating_add(1);
                // FIFO promotion: next queued request takes the in-flight slot.
                if let Some(next) = self.queue.pop_front() {
                    self.in_flight = Some(next);
                    self.in_flight_prefilled = false;
                }
            }
            _ => { /* unknown slot — noop */ }
        }
    }

    fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            policy: SchedulerPolicy::FifoSerial,
            in_flight_slots: self.in_flight_count(),
            admitted_total: self.admitted_total,
            rejected_429_total: self.rejected_429_total,
            completed_total: self.completed_total,
        }
    }
}

// ---------------------------------------------------------------------------
// InflightBatchedScheduler — signature-only stub (Phase B iter-1)
// ---------------------------------------------------------------------------

/// Admission-during-decode scheduler — signature-only at Phase B iter-1.
///
/// `admit` + `release` + `stats` are **real** (the queue + slot accounting
/// is fully wired). `step` returns `Err(StepError::NotImplemented)` at this
/// iter — Phase B iter-3 replaces the body with the mirrored llama.cpp
/// `-cb` admission-during-decode loop (ADR-040 §3.3).
///
/// Capacity model: `queue_capacity` pending + `max_slots` concurrent
/// in-flight requests. Overflow returns `QueueFull` with the running
/// in-flight slot count for diagnostic accuracy.
pub struct InflightBatchedScheduler {
    queue_capacity: u32,
    max_slots: u32,
    in_flight: Vec<RequestSlot>,
    queue: VecDeque<RequestSlot>,
    admitted_total: u64,
    rejected_429_total: u64,
    completed_total: u64,
    next_slot_id: u32,
}

impl InflightBatchedScheduler {
    /// Build an inflight-batched scheduler with the given queue cap +
    /// concurrent-slot cap. `max_slots` defaults to ADR-040 §3.4's value
    /// of `4` at the CLI layer; this constructor takes it as a parameter
    /// so test fixtures can pin small values.
    pub fn new(queue_capacity: u32, max_slots: u32) -> Self {
        Self {
            queue_capacity,
            max_slots,
            in_flight: Vec::with_capacity(max_slots as usize),
            queue: VecDeque::new(),
            admitted_total: 0,
            rejected_429_total: 0,
            completed_total: 0,
            next_slot_id: 0,
        }
    }

    fn alloc_slot_id(&mut self) -> SlotId {
        let id = SlotId(self.next_slot_id);
        self.next_slot_id = self.next_slot_id.wrapping_add(1);
        id
    }
}

impl Scheduler for InflightBatchedScheduler {
    fn policy(&self) -> SchedulerPolicy {
        SchedulerPolicy::InflightBatched
    }

    fn admit(&mut self, req: AdmitRequest) -> Result<RequestSlot, AdmitError> {
        // Cap: queue_capacity pending + max_slots concurrent in-flight.
        let in_flight = self.in_flight.len() as u32;
        let queued = self.queue.len() as u32;
        if in_flight >= self.max_slots && queued >= self.queue_capacity {
            self.rejected_429_total = self.rejected_429_total.saturating_add(1);
            return Err(AdmitError::QueueFull {
                capacity: self.queue_capacity,
                in_flight,
            });
        }

        let slot = RequestSlot {
            slot_id: self.alloc_slot_id(),
            admitted_at: Instant::now(),
            prompt_tokens: req.prompt_tokens,
            max_tokens: req.max_tokens,
        };
        self.admitted_total = self.admitted_total.saturating_add(1);

        if in_flight < self.max_slots {
            self.in_flight.push(slot.clone());
        } else {
            self.queue.push_back(slot.clone());
        }
        Ok(slot)
    }

    fn step(&mut self) -> Result<SchedulerStep, StepError> {
        // Phase B iter-1 contract: signature-only. iter-3 lands the
        // admission-during-decode body. The test
        // `inflight_batched_step_returns_not_implemented_at_iter_1`
        // explicitly pins this discriminant.
        Err(StepError::NotImplemented)
    }

    fn release(&mut self, slot: SlotId) {
        let before = self.in_flight.len();
        self.in_flight.retain(|s| s.slot_id != slot);
        if self.in_flight.len() < before {
            self.completed_total = self.completed_total.saturating_add(1);
            // Promote queued request into the freed slot.
            if let Some(next) = self.queue.pop_front() {
                self.in_flight.push(next);
            }
        }
        // Unknown slot is a no-op (mirrors LoadedPool::touch unknown-key noop
        // pattern at multi_model.rs:1513).
    }

    fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            policy: SchedulerPolicy::InflightBatched,
            in_flight_slots: self.in_flight.len() as u32,
            admitted_total: self.admitted_total,
            rejected_429_total: self.rejected_429_total,
            completed_total: self.completed_total,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn req(prompt_tokens: u32, max_tokens: u32) -> AdmitRequest {
        AdmitRequest { prompt_tokens, max_tokens }
    }

    // -----------------------------------------------------------------------
    // FIFO contract preservation — load-bearing.
    // These tests pin ADR-005 Decision #2 + Decision #19 byte-equivalence
    // at the trait surface. Iter-2 adds a regression test against
    // Engine::spawn that asserts the same behaviour against the live mpsc
    // path.
    // -----------------------------------------------------------------------

    #[test]
    fn fifo_admit_then_step_returns_prefill_for_the_admitted_slot() {
        let mut s = FifoSchedulerAdapter::new(4);
        let slot = s.admit(req(11, 32)).expect("admit ok");
        match s.step().expect("step ok") {
            SchedulerStep::Prefill { slot_id, n_tokens } => {
                assert_eq!(slot_id, slot.slot_id);
                assert_eq!(n_tokens, 11);
            }
            other => panic!("expected Prefill, got {:?}", other),
        }
    }

    #[test]
    fn fifo_admit_twice_queues_second_until_first_releases() {
        let mut s = FifoSchedulerAdapter::new(4);
        let a = s.admit(req(10, 8)).expect("admit a");
        let b = s.admit(req(20, 16)).expect("admit b");
        assert_ne!(a.slot_id, b.slot_id);
        assert_eq!(s.stats().in_flight_slots, 1);

        // First step is prefill for slot a.
        match s.step().unwrap() {
            SchedulerStep::Prefill { slot_id, .. } => assert_eq!(slot_id, a.slot_id),
            other => panic!("expected Prefill for a, got {:?}", other),
        }

        // Release a — b promotes to in-flight.
        s.release(a.slot_id);
        assert_eq!(s.stats().in_flight_slots, 1);
        assert_eq!(s.stats().completed_total, 1);

        match s.step().unwrap() {
            SchedulerStep::Prefill { slot_id, n_tokens } => {
                assert_eq!(slot_id, b.slot_id);
                assert_eq!(n_tokens, 20);
            }
            other => panic!("expected Prefill for b, got {:?}", other),
        }
    }

    #[test]
    fn fifo_admit_at_capacity_returns_queue_full_with_both_fields() {
        let mut s = FifoSchedulerAdapter::new(2);
        let _a = s.admit(req(1, 1)).expect("a in-flight");
        let _b = s.admit(req(1, 1)).expect("b queued");
        let _c = s.admit(req(1, 1)).expect("c queued");
        match s.admit(req(1, 1)) {
            Err(AdmitError::QueueFull { capacity, in_flight }) => {
                assert_eq!(capacity, 2, "capacity field must echo queue_capacity");
                assert_eq!(in_flight, 1, "FIFO max in_flight is 1 by Decision #2");
            }
            other => panic!("expected QueueFull, got {:?}", other),
        }
        assert_eq!(s.stats().rejected_429_total, 1);
    }

    #[test]
    fn fifo_release_unknown_slot_is_noop() {
        let mut s = FifoSchedulerAdapter::new(4);
        let a = s.admit(req(1, 1)).expect("admit a");
        // Releasing a slot id the scheduler has never seen is a no-op.
        s.release(SlotId(9_999));
        assert_eq!(s.stats().completed_total, 0);
        assert_eq!(s.stats().in_flight_slots, 1);
        // a is still in-flight.
        match s.step().unwrap() {
            SchedulerStep::Prefill { slot_id, .. } => assert_eq!(slot_id, a.slot_id),
            other => panic!("expected Prefill, got {:?}", other),
        }
    }

    #[test]
    fn fifo_stats_admitted_rejected_completed_counters_advance() {
        let mut s = FifoSchedulerAdapter::new(1);
        let a = s.admit(req(1, 1)).expect("a");
        let _b = s.admit(req(1, 1)).expect("b queued");
        assert!(s.admit(req(1, 1)).is_err(), "c must be rejected");
        let stats = s.stats();
        assert_eq!(stats.admitted_total, 2);
        assert_eq!(stats.rejected_429_total, 1);
        assert_eq!(stats.completed_total, 0);

        s.release(a.slot_id);
        let stats = s.stats();
        assert_eq!(stats.admitted_total, 2);
        assert_eq!(stats.rejected_429_total, 1);
        assert_eq!(stats.completed_total, 1);
    }

    #[test]
    fn fifo_step_returns_idle_when_no_work() {
        let mut s = FifoSchedulerAdapter::new(4);
        assert_eq!(s.step().unwrap(), SchedulerStep::Idle);
    }

    #[test]
    fn fifo_policy_returns_fifoserial() {
        let s = FifoSchedulerAdapter::new(4);
        assert_eq!(s.policy(), SchedulerPolicy::FifoSerial);
        assert_eq!(s.stats().policy, SchedulerPolicy::FifoSerial);
    }

    #[test]
    fn fifo_decode_phase_returns_single_slot_in_decode_variant() {
        let mut s = FifoSchedulerAdapter::new(4);
        let a = s.admit(req(8, 4)).expect("admit a");
        // First step is prefill.
        match s.step().unwrap() {
            SchedulerStep::Prefill { slot_id, .. } => assert_eq!(slot_id, a.slot_id),
            other => panic!("expected Prefill, got {:?}", other),
        }
        // Second step is decode for the SAME slot (one-element vec,
        // NOT an empty vec).
        match s.step().unwrap() {
            SchedulerStep::Decode { slots } => {
                assert_eq!(slots.len(), 1, "decode vec must contain the in-flight slot");
                assert_eq!(slots[0], a.slot_id);
            }
            other => panic!("expected Decode, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // InflightBatched signature gate — proves type is wired but step is
    // stubbed at iter-1. The `_not_implemented_at_iter_1` test is REMOVED
    // (replaced with a real-step assertion) when Phase B iter-3 lands.
    // -----------------------------------------------------------------------

    #[test]
    fn inflight_batched_admit_succeeds_below_max_slots() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(1, 1)).expect("a");
        let b = s.admit(req(1, 1)).expect("b");
        assert_ne!(a.slot_id, b.slot_id);
        assert_eq!(s.stats().in_flight_slots, 2);
        assert_eq!(s.stats().admitted_total, 2);
    }

    #[test]
    fn inflight_batched_admit_returns_queue_full_at_capacity_plus_max_slots() {
        let mut s = InflightBatchedScheduler::new(2, 2);
        // Fill 2 in-flight + 2 queued.
        let _ = s.admit(req(1, 1)).expect("in-flight 0");
        let _ = s.admit(req(1, 1)).expect("in-flight 1");
        let _ = s.admit(req(1, 1)).expect("queued 0");
        let _ = s.admit(req(1, 1)).expect("queued 1");
        // Next admit rejects.
        match s.admit(req(1, 1)) {
            Err(AdmitError::QueueFull { capacity, in_flight }) => {
                assert_eq!(capacity, 2);
                assert_eq!(in_flight, 2);
            }
            other => panic!("expected QueueFull, got {:?}", other),
        }
        assert_eq!(s.stats().rejected_429_total, 1);
    }

    #[test]
    fn inflight_batched_step_returns_not_implemented_at_iter_1() {
        // Phase B iter-1 contract: step is signature-only. iter-3
        // replaces the body and this test gets rewritten to assert real
        // SchedulerStep variants. Until then the discriminant is pinned.
        let mut s = InflightBatchedScheduler::new(4, 2);
        let _ = s.admit(req(1, 1)).expect("admit ok");
        assert_eq!(s.step(), Err(StepError::NotImplemented));
    }

    #[test]
    fn inflight_batched_policy_returns_inflightbatched() {
        let s = InflightBatchedScheduler::new(4, 2);
        assert_eq!(s.policy(), SchedulerPolicy::InflightBatched);
        assert_eq!(s.stats().policy, SchedulerPolicy::InflightBatched);
    }

    #[test]
    fn inflight_batched_release_drops_slot_from_in_flight() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(1, 1)).expect("a");
        let b = s.admit(req(1, 1)).expect("b");
        assert_eq!(s.stats().in_flight_slots, 2);
        s.release(a.slot_id);
        assert_eq!(s.stats().in_flight_slots, 1);
        assert_eq!(s.stats().completed_total, 1);
        // b still in-flight.
        s.release(b.slot_id);
        assert_eq!(s.stats().in_flight_slots, 0);
        assert_eq!(s.stats().completed_total, 2);
        // Unknown slot is a no-op.
        s.release(SlotId(9_999));
        assert_eq!(s.stats().completed_total, 2);
    }

    #[test]
    fn inflight_batched_stats_counters_advance() {
        let mut s = InflightBatchedScheduler::new(1, 1);
        let a = s.admit(req(1, 1)).expect("a in-flight");
        let _b = s.admit(req(1, 1)).expect("b queued");
        // Cap reached: 1 in-flight + 1 queued = 2, and (max_slots=1, queue_cap=1).
        assert!(s.admit(req(1, 1)).is_err(), "c must be rejected");
        let stats = s.stats();
        assert_eq!(stats.admitted_total, 2);
        assert_eq!(stats.rejected_429_total, 1);
        assert_eq!(stats.completed_total, 0);
        assert_eq!(stats.in_flight_slots, 1);

        s.release(a.slot_id);
        let stats = s.stats();
        assert_eq!(stats.completed_total, 1);
        // b promoted to in-flight.
        assert_eq!(stats.in_flight_slots, 1);
    }

    // -----------------------------------------------------------------------
    // Cross-cutting
    // -----------------------------------------------------------------------

    #[test]
    fn request_slot_admitted_at_is_monotonic() {
        let mut s = FifoSchedulerAdapter::new(4);
        let a = s.admit(req(1, 1)).expect("a");
        // Sleep is unnecessary — Instant ticks are strictly monotonic on
        // all supported platforms, and the second admit happens after
        // the first returns. We only need >=.
        let b = s.admit(req(1, 1)).expect("b");
        assert!(b.admitted_at >= a.admitted_at,
            "admitted_at must be monotonically non-decreasing across consecutive admits");
    }

    #[test]
    fn admit_error_queue_full_names_capacity_and_in_flight() {
        let err = AdmitError::QueueFull { capacity: 7, in_flight: 3 };
        let dbg = format!("{:?}", err);
        assert!(dbg.contains("capacity"), "Debug must mention capacity: {}", dbg);
        assert!(dbg.contains("in_flight"), "Debug must mention in_flight: {}", dbg);
        assert!(dbg.contains('7'), "Debug must include capacity value: {}", dbg);
        assert!(dbg.contains('3'), "Debug must include in_flight value: {}", dbg);
    }
}
