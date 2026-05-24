//! Scheduler trait + FIFO adapter + InflightBatched FSM
//! (ADR-040 Phase B iter-1 + iter-1.5 + iter-B3 + iter-2.5 + iter-C2.5 +
//! **iter-A5**).
//!
//! # iter-A5 (this commit) — per-slot OOM + budget enforcement
//!
//! ADR-040 §3.5 — "`kv_cache_budget_bytes` (existing field on
//! `Engine::spawn`) divides equally across slots in SeparateSlots layout.
//! Per-slot budget = `total / max_slots`. Per-slot OOM returns 429 to the
//! admitting handler (Decision #19 contract preserved)."
//!
//! iter-A5 implements the **admit-time** half of this contract. The
//! complementary append-time half (defense-in-depth at the per-model
//! `MultiSeqKvCache::append_for_seq` site) is deliberately NOT shipped:
//! today's per-model impls (Qwen35 `HybridKvCache`, Gemma 4
//! `MultiSeqHbKvBuffers`) pre-allocate full `max_seq_len` K/V buffers per
//! slot at `new` / `alloc_hb_kv_for_layer` time, so a per-token append
//! cannot OOM at the buffer layer — the only place OOM can be
//! operator-actionable is before the work starts, at admission.
//!
//! Concretely iter-A5 adds:
//!
//! - [`AdmitRequest::kv_bytes_needed`] — caller-computed byte cost the
//!   request would put on the per-slot KV cache (typically
//!   `(prompt_tokens + max_tokens) * kv_bytes_per_token`).  `0` means
//!   "caller did not compute; do not enforce" — preserves byte-equivalence
//!   for callers that have not yet wired the byte-cost computation
//!   (today: every caller — Phase C2c will wire it).
//! - [`FifoSchedulerAdapter::new_with_kv_budget`] +
//!   [`InflightBatchedScheduler::new_with_kv_budget`] — opt-in
//!   constructors that take the per-slot byte budget. The 0-budget
//!   default reached via `new` (no env, no flag) preserves
//!   byte-equivalence.
//! - [`AdmitError::SlotBudgetExceeded`] — typed admit-time rejection
//!   mapped to HTTP 429 + `Retry-After: 1` upstream (parallel to
//!   `QueueFull`; ADR-040 §3.5 explicitly preserves the Decision #19
//!   wire-level contract).
//!
//! **Why bytes, not tokens** (Step 4 design seam — see ADR-040 §6.1.13
//! for the full closure block): the scheduler tracks BYTES directly,
//! and callers compute `kv_bytes_needed` per-arch.  Two alternatives
//! were considered:
//!
//! 1. Tokens-via-conversion: scheduler stores `per_slot_budget_tokens:
//!    u32`; the worker converts `kv_cache_budget_bytes →
//!    per_slot_budget_tokens` using a per-arch `kv_bytes_per_token`
//!    constant at engine-spawn time.  Rejected because: (a) it bakes
//!    per-arch math into the scheduler (a pure data primitive should
//!    stay arch-agnostic — same reason `SlotId` and `RequestId` are not
//!    arch-typed); (b) `kv_bytes_per_token` varies per layer for hybrid
//!    architectures (Gemma 4: full vs sliding layers have different
//!    head_dim × n_kv; Qwen3.5: full vs linear-attn layers diverge
//!    further) so a single scalar would either under-count (false
//!    accepts) or wildly over-count (false rejects) — neither is
//!    operator-honest.
//! 2. Bytes-direct (chosen): scheduler stores `per_slot_kv_budget_bytes:
//!    u64`; caller passes `kv_bytes_needed: u64` on each `AdmitRequest`.
//!    Caller (Phase C2c+ for per-arch SlotAware workers) computes the
//!    byte cost using the per-arch `KvSpillDescriptor` / per-layer
//!    `head_dim × n_kv × dtype_size × max_seq_len` math the existing
//!    `kv_spill_descriptor.rs` already does.  Scheduler stays arch-
//!    agnostic; the conversion lives at the per-arch seam where it
//!    belongs.
//!
//! Per-arch `kv_bytes_per_token` wiring is **deferred to Phase C2c+**
//! when the SlotAware runtime lands for each arch.  iter-A5 ships the
//! scheduler-side enforcement + the typed error + the schema-side 429
//! mapping; the actual per-arch byte-cost computation lands when the
//! worker arm that needs it (Qwen35 SlotAware iter-C2c, Gemma 4
//! SlotAware iter-C2d) wires `kv_bytes_needed` into the per-request
//! `AdmitRequest`.  Until then, `kv_bytes_needed: 0` ⇒ enforcement
//! disabled ⇒ byte-equivalent to pre-A5.
//!
//! (ADR-040 Phase B iter-1 + iter-1.5 + iter-B3 + iter-2.5 + iter-C2.5.)
//!
//! This module is the **pure data primitive** that ADR-040 Phase C (iter-2)
//! wires into `serve::api::engine::Engine`. It contains *no* engine load,
//! *no* GPU code, *no* `AppState` wiring — those land in Phase C. The
//! pattern mirrors `serve::multi_model` (W74 iter-206): a synthetic-fixture-
//! tested data structure that later iters glue into the live serve path.
//!
//! # What this module does (iter-1 + iter-1.5 + iter-B3 + iter-2.5)
//!
//! - Declares the `Scheduler` trait surface (`admit`, `step`, `release`,
//!   `stats`, `policy`) — see ADR-040 §3.2 + AC-2.
//! - Ships `FifoSchedulerAdapter` — the **byte-equivalent** wrapper of the
//!   existing ADR-005 Phase 2 Decision #2 contract (one in-flight request,
//!   bounded queue, 429 on overflow).
//! - Ships the production `InflightBatchedScheduler` with a real `step()`
//!   FSM body that mirrors llama.cpp's `-cb` (continuous-batching)
//!   admission-during-decode semantics per ADR-040 §3.3.
//! - **iter-2.5 (this commit)** closes 3 CRITICAL + 1 MAJOR adversarial
//!   findings (cfa-iter2.5 #C1/C2/C3/M2) by:
//!   - **C1 — SlotHandle (TOCTOU race fix)**: the driver-callback APIs
//!     (`advance_after_prefill`/`advance_after_decode`/`release`) no longer
//!     key off a raw `SlotId`. They take an opaque `SlotHandle` that carries
//!     both the physical `SlotId` AND a per-slot generation counter. The
//!     scheduler validates generation == current on every callback; a stale
//!     handle (the previous occupant of a recycled slot) is silently dropped
//!     as a no-op. Without this, an auto-release of `SlotId(0)` followed
//!     by an immediate promote of a queued request onto `SlotId(0)` left
//!     in-flight callbacks for the OLD occupant free to mutate the NEW
//!     occupant's state.
//!   - **C2 — RequestId (queued-sentinel collision fix)**: queued requests
//!     no longer share the sentinel `SlotId(u32::MAX)`. They carry a fresh
//!     `RequestId` (a monotonic u64) so `cancel_queued(request_id)` removes
//!     exactly ONE queued request (the previous `release(SlotId(u32::MAX))`
//!     used `retain != SlotId(u32::MAX)` which removed ALL queued requests
//!     — silent data loss).
//!   - **C3 — step() priority (older-prefilling-wins fix)**: after promoting
//!     one queued slot into `in_flight`, `step()` picks the FIRST
//!     `Prefilling` slot in insertion order (FIFO across in_flight). The
//!     previous code special-cased "if I just promoted, emit Mixed for
//!     the promoted slot" which starved an OLDER mid-chunk Prefilling slot.
//!     Now: Mixed/Prefill always uses `first_prefilling_idx()`; the promoted
//!     slot is just the slot that lands at the end of the `in_flight`
//!     vector and waits its turn.
//!   - **M2 — prompt_tokens=0 deadlock fix**: a request admitted with
//!     `prompt_tokens == 0` now transitions DIRECTLY to `Decoding` (skips
//!     `Prefilling` entirely — a zero-token prompt has nothing to prefill).
//!     Without this, `step()` would emit `Prefill { n_tokens: 0 }` and a
//!     driver that treated zero-token prefill as no-work would deadlock.
//!     A request admitted with `max_tokens == 0` is recognized at admit
//!     time as zero-budget: the scheduler returns
//!     `RequestSlot { handle: None, .. }` and bumps `completed_total`
//!     without creating an in-flight slot. (cfa-iter-C2.5 M1: closes the
//!     stuck-Decoding-slot risk Codex flagged — the prior implementation
//!     was a documentation lie: the comment claimed "auto-releases at
//!     admit time" but `initial_phase` still pushed a `Decoding{ tokens_
//!     produced:0, max_tokens:0 }` slot into in_flight, and only the
//!     Embed worker arm's explicit post-prefill `release` avoided the
//!     leak. Now the scheduler enforces it structurally: a zero-budget
//!     admit allocates no physical slot, and the caller observes
//!     `handle.is_none()` and skips the drive loop.)
//! - Pins the prefill chunk-size default at `DEFAULT_PREFILL_CHUNK_TOKENS
//!   = 512` — matches llama.cpp's `-ub` (ubatch) default.
//!
//! # What this module does NOT do (iter-2.5 scope)
//!
//! - Hold an `Engine` (or any `mlx_native` buffers) — `RequestSlot` is a
//!   pure descriptor. Phase C iter-2's `Engine::spawn` will accept an
//!   injected `Box<dyn Scheduler>` and dispatch against this trait.
//! - Touch `serve::api::engine`, `serve::mod::cmd_serve`, or any handler —
//!   wiring is Phase C.
//! - Build paged-KV blocks — ADR-040 §3.1 picks `SeparateSlots` first.
//! - Execute forward passes — `step()` returns a `SchedulerStep`
//!   discriminant describing what the driver loop SHOULD do; the driver
//!   loop (Phase C iter-2) actually calls `forward_prefill` /
//!   `forward_decode` and reports back via `advance_after_*`.
//! - Thread `slot_id` through `forward_prefill.rs` /
//!   `forward_prefill_batched.rs` — Phase B iter-4.
//! - Handle multi-prefill batching in `Mixed` — `Mixed` carries exactly
//!   ONE prefill (the first `Prefilling` slot in FIFO order) batched with
//!   N decode slots.
//!
//! # Backward-compat contract (ADR-040 §3.6)
//!
//! `FifoSchedulerAdapter` MUST behave bit-equivalently to the pre-ADR-040
//! `Engine::spawn` channel + worker thread. Specifically:
//!
//! 1. At most one in-flight request (Decision #2 — `max_slots == 1`).
//! 2. Bounded queue with capacity = `queue_capacity` from `Engine::spawn`
//!    (Decision #19 — channel buffer at `queue_capacity.max(1)`). The
//!    `.max(1)` normalization is mirrored in `FifoSchedulerAdapter::new`.
//! 3. Overflow returns `AdmitError::QueueFull`, which the handler layer
//!    maps to HTTP 429 + `Retry-After: 1`.
//! 4. FIFO ordering preserved: pop order == push order. Physical `SlotId`
//!    returned from `admit()` is **always `SlotId(0)`** for the FIFO
//!    policy (iter-1.5 F3b) — single physical slot reused as queued
//!    requests get promoted. Per-admit identity is now disambiguated via
//!    the new `RequestId` field carried inside `RequestSlot` (iter-2.5 C2).
//!
//! # iter-2.5 adversarial-review findings addressed
//!
//! - **C1 (Codex+Claude CRITICAL — SlotId TOCTOU race)**: see above.
//! - **C2 (Codex+Claude CRITICAL — queued-sentinel collision)**: see above.
//! - **C3 (Codex+Claude CRITICAL — step priority bug)**: see above.
//! - **M2 (Codex MAJOR — prompt_tokens=0 deadlock risk)**: see above.

use std::collections::VecDeque;
use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// SlotId — re-exported from Phase A iter-1.
// ---------------------------------------------------------------------------

pub use crate::serve::multi_seq_kv::SlotId;

// ---------------------------------------------------------------------------
// RequestId — per-admit logical identifier (iter-2.5 C2 fix).
// ---------------------------------------------------------------------------

/// Logical request identifier assigned at admit time, distinct from the
/// physical `SlotId`. Used to cancel queued requests before they get
/// promoted to a physical slot.
///
/// (cfa-iter2.5 C2: closes the `SlotId(u32::MAX)` collision among queued
/// requests. The prior implementation used `SlotId(u32::MAX)` as a
/// sentinel for ALL queued slots and `release(SlotId(u32::MAX))` used
/// `retain != SlotId(u32::MAX)` which removed ALL queued requests at
/// once — silent data loss for any concurrent client. Now: each admit
/// gets a unique monotonic `RequestId`, and `cancel_queued(request_id)`
/// removes exactly one.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

// ---------------------------------------------------------------------------
// SlotHandle — per-admit-or-promote handle (iter-2.5 C1 fix).
// ---------------------------------------------------------------------------

/// Per-admit-or-promote handle for an in-flight slot. Callers MUST pass
/// this back to [`InflightBatchedScheduler::advance_after_prefill`],
/// [`InflightBatchedScheduler::advance_after_decode`], and
/// [`InflightBatchedScheduler::release`] (and the equivalent FIFO entry
/// points) so the scheduler can validate the callback is for the
/// CURRENT occupant of that physical slot, not a recycled occupant.
///
/// (cfa-iter2.5 C1: closes the TOCTOU race where auto-release on
/// `max_tokens` + immediate promote of a queued request onto the same
/// `SlotId` let stale in-flight callbacks for the OLD occupant mutate
/// the NEW occupant's state. The generation counter is bumped on every
/// release; any callback with a generation that does not match the
/// scheduler's current generation for that slot is silently dropped as
/// a no-op.)
///
/// Stale callbacks are NOT an error condition — they happen legitimately
/// under any locking discipline that drops the scheduler lock between
/// `step()` and the driver-side `advance_after_*` callback. The scheduler
/// is defensive by construction; the driver does not need to know
/// whether its callback "won" or was superseded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotHandle {
    /// Physical slot index — always in `[0, max_slots)` for
    /// `InflightBatched`, always `SlotId(0)` for `FifoSerial`.
    pub slot_id: SlotId,
    /// Monotonic per-slot generation. Bumped on every release. A
    /// callback whose `generation` does not match the scheduler's
    /// current generation for `slot_id` is a no-op.
    pub generation: u64,
}

// ---------------------------------------------------------------------------
// Scheduler policy (ADR-040 §3.2)
// ---------------------------------------------------------------------------

/// Which scheduling discipline an `Engine` uses (ADR-040 §3.2).
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
/// `admit`.
///
/// iter-2.5 changes:
/// - `request_id` (new): always set, unique per-admit logical id (C2).
/// - `handle` (new): `Some` when the slot is in `in_flight` immediately
///   after admit; `None` when the slot was queued (the real `SlotHandle`
///   is allocated at promotion time and observed via
///   `SchedulerStep::Prefill { handle }`).
/// - `slot_id` (removed): callers now read `handle.slot_id` when present,
///   or treat the request as queued.
#[derive(Debug, Clone)]
pub struct RequestSlot {
    /// Per-admit logical id, unique for the lifetime of the scheduler
    /// (modulo u64 wraparound, which at 1 KHz takes 580M years).
    /// Always set — callers use this to cancel queued requests via
    /// `cancel_queued(request_id)` and to disambiguate two successive
    /// admits in `RequestSlot.handle == None` (queued) state.
    pub request_id: RequestId,
    /// `Some` when the request landed directly in `in_flight` at admit
    /// time. `None` when the request was queued (no physical slot yet);
    /// the handle is allocated at promotion time and observed via
    /// `SchedulerStep::Prefill { handle: SlotHandle, .. }`.
    pub handle: Option<SlotHandle>,
    /// Wall-clock instant `admit()` returned `Ok` for this slot.
    pub admitted_at: Instant,
    /// Prompt-token count as passed to `admit()` (post-template, pre-prefill).
    pub prompt_tokens: u32,
    /// Maximum new tokens the request may emit (sampler stop budget).
    pub max_tokens: u32,
}

/// Caller-supplied request bookkeeping handed to `Scheduler::admit`.
#[derive(Debug, Clone)]
pub struct AdmitRequest {
    /// Tokenized prompt length after chat-template rendering.
    pub prompt_tokens: u32,
    /// Maximum new tokens the request may emit.
    pub max_tokens: u32,
    /// **ADR-040 §3.5 iter-A5** — caller-computed byte cost the request
    /// would put on the per-slot KV cache, typically
    /// `(prompt_tokens + max_tokens) * per_arch_kv_bytes_per_token`.
    ///
    /// Set to `0` to opt out of per-slot KV-budget enforcement (the
    /// scheduler treats `0` as "caller did not compute; do not enforce").
    /// Today every caller in `worker_run` passes `0` — byte-equivalence
    /// with pre-A5 is preserved.  Phase C2c (Qwen35 SlotAware) and
    /// C2d (Gemma 4 SlotAware) wire the real per-arch byte cost so
    /// admission can fail-fast under aggregate budget pressure (per
    /// ADR-040 §3.5: "per-slot OOM returns 429 to the admitting
    /// handler").
    ///
    /// `0`-default also handles the per-iter back-compat surface for
    /// pre-A5 callsites that still construct `AdmitRequest { prompt_tokens,
    /// max_tokens }` literally — those callsites compile against the new
    /// field only if they switch to functional update (`..Default::default()`)
    /// or explicit `kv_bytes_needed: 0`; iter-A5's worker_run edits add
    /// the explicit field at each of the 4 admit sites.
    pub kv_bytes_needed: u64,
}

impl Default for AdmitRequest {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            max_tokens: 0,
            kv_bytes_needed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduler step variant (ADR-040 AC-2)
// ---------------------------------------------------------------------------

/// What the scheduler decided the next forward pass should do.
///
/// iter-2.5 shape change: all variant fields that previously carried
/// `SlotId` now carry `SlotHandle` (C1 — TOCTOU race). The driver
/// callbacks (`advance_after_prefill` / `advance_after_decode`) take
/// the handle directly off this discriminant and pass it back to the
/// scheduler; the scheduler validates generation and silently drops
/// stale callbacks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerStep {
    /// No work available; the engine loop should park.
    Idle,
    /// Run prefill for one slot for `n_tokens` tokens. The driver MUST
    /// call `Scheduler::advance_after_prefill(handle, n_consumed)` after
    /// running the forward pass.
    Prefill { handle: SlotHandle, n_tokens: u32 },
    /// Run decode for the listed slots (one forward, batched in slot dim).
    /// The driver MUST call `Scheduler::advance_after_decode(handle)` for
    /// each handle per emitted token.
    Decode { handles: Vec<SlotHandle> },
    /// Run prefill for one slot AND decode for the listed slots in one
    /// forward. The driver MUST call `advance_after_prefill` for the
    /// prefill handle AND `advance_after_decode` for each decode handle.
    ///
    /// iter-2.5 C3 fix: `prefill` is now the FIRST `Prefilling` slot in
    /// insertion (FIFO) order — NOT necessarily the just-promoted slot.
    /// An older mid-chunk prefill always wins over a freshly-promoted
    /// queued slot so chunk continuation cannot be starved.
    Mixed {
        prefill: SlotHandle,
        n_prefill_tokens: u32,
        decode_handles: Vec<SlotHandle>,
    },
}

// ---------------------------------------------------------------------------
// Prefill chunk size — (ADR-040 §3.3 llama.cpp -cb mirror)
// ---------------------------------------------------------------------------

/// Default number of prompt tokens consumed per `Prefill` step.
///
/// Pinned at 512 to mirror llama.cpp's `-ub` (ubatch) default.
pub const DEFAULT_PREFILL_CHUNK_TOKENS: u32 = 512;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why `Scheduler::admit` rejected a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmitError {
    /// Queue + in-flight slots are at the configured cap. Maps to HTTP 429.
    QueueFull {
        queue_capacity: u32,
        total_admissible: u32,
        in_flight: u32,
    },
    /// Scheduler is no longer accepting work (post-shutdown).
    SchedulerStopped,
    /// **ADR-040 §3.5 iter-A5** — the request's caller-computed KV-byte
    /// cost (`AdmitRequest::kv_bytes_needed`) exceeds the configured
    /// per-slot budget (`per_slot_kv_budget_bytes`).  Maps to HTTP 429 +
    /// `Retry-After: 1` upstream per Decision #19, parallel to
    /// `QueueFull`.  Distinct from `QueueFull` so observability +
    /// alerting can differentiate "too many concurrent requests"
    /// (transient — capacity will free) from "this single request asks
    /// for more KV than any single slot can hold" (operator-actionable
    /// — reduce `max_tokens` or use a shorter prompt).
    ///
    /// The fields are populated so the operator-facing 429 body can
    /// name what was attempted vs what was permitted — same shape as
    /// `MultiSeqError::SlotOom`'s `needed_bytes` / `budget_bytes` pair
    /// (`src/serve/multi_seq_kv.rs:265`).
    SlotBudgetExceeded {
        /// Bytes the admit attempt would need (from `AdmitRequest::kv_bytes_needed`).
        needed_bytes: u64,
        /// Per-slot KV budget configured on the scheduler.
        budget_bytes: u64,
    },
}

impl fmt::Display for AdmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdmitError::QueueFull { queue_capacity, total_admissible, in_flight } => write!(
                f,
                "queue full (queue_capacity={}, total_admissible={}, in_flight={})",
                queue_capacity, total_admissible, in_flight,
            ),
            AdmitError::SchedulerStopped => write!(f, "scheduler stopped"),
            AdmitError::SlotBudgetExceeded { needed_bytes, budget_bytes } => write!(
                f,
                "per-slot KV budget exceeded (needed_bytes={}, budget_bytes={}) \
                 — ADR-040 §3.5: reduce max_tokens or request a smaller prompt",
                needed_bytes, budget_bytes,
            ),
        }
    }
}

/// Why `Scheduler::step` failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepError {
    /// The underlying engine forward-pass returned an error. Reserved
    /// for Phase C iter-2 wiring; the scheduler itself is structurally
    /// infallible.
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
    pub queue_capacity: u32,
    pub admitted_total: u64,
    pub rejected_429_total: u64,
    pub completed_total: u64,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// The ADR-040 §2.1 scheduler surface.
///
/// iter-2.5 changes the callback identities on `release` from `SlotId`
/// to `SlotHandle` (C1 — TOCTOU race fix). The
/// `advance_after_prefill`/`advance_after_decode` driver callbacks live
/// on the concrete `InflightBatchedScheduler` / `FifoSchedulerAdapter`
/// types because their input types differ slightly (FIFO has no real
/// chunking) and the Phase C iter-2 engine loop dispatches against the
/// concrete type after the policy enum branch.
pub trait Scheduler: Send {
    /// Which discipline this scheduler implements.
    fn policy(&self) -> SchedulerPolicy;
    /// Admit a new request. Returns the slot handle on success or an
    /// `AdmitError::QueueFull` when the queue + in-flight set is at cap.
    fn admit(&mut self, req: AdmitRequest) -> Result<RequestSlot, AdmitError>;
    /// Decide what the next forward pass should do.
    fn step(&mut self) -> Result<SchedulerStep, StepError>;
    /// Drop an in-flight slot (completion, error, or client-disconnect).
    /// `handle` is the handle observed at admit (or at promotion via the
    /// `SchedulerStep::Prefill` discriminant). Stale handles are silently
    /// dropped as no-ops (iter-2.5 C1).
    fn release(&mut self, handle: SlotHandle);
    /// Snapshot of lifetime counters + current resident state.
    fn stats(&self) -> SchedulerStats;
}

// ---------------------------------------------------------------------------
// FifoSchedulerAdapter — byte-equivalent wrap of pre-ADR-040 Engine
// ---------------------------------------------------------------------------

/// Byte-equivalent wrap of the existing `Engine::spawn` FIFO contract
/// (Decision #2 + Decision #19).
///
/// `max_slots` is always 1 under this policy. The physical `SlotId` is
/// always `SlotId(0)`; the generation counter on `SlotId(0)` bumps on
/// every release so stale callbacks across recycled occupants are
/// rejected (iter-2.5 C1). Queued requests carry a `RequestId` (not a
/// sentinel SlotId — iter-2.5 C2) and are cancellable via
/// [`Self::cancel_queued`].
pub struct FifoSchedulerAdapter {
    queue_capacity: u32,
    /// Queued requests not yet promoted. Each carries a unique
    /// `RequestId` so cancellation removes exactly one. The
    /// `prompt_tokens` / `max_tokens` are captured at admit time.
    queue: VecDeque<QueuedFifoRequest>,
    /// Current in-flight slot (FIFO has max_slots=1).
    in_flight: Option<InFlightFifoSlot>,
    /// Generation counter for SlotId(0). Bumped on every release; the
    /// new occupant after a recycle observes a generation one higher
    /// than the prior occupant's.
    slot_generation: u64,
    next_request_id: u64,
    admitted_total: u64,
    rejected_429_total: u64,
    completed_total: u64,
    /// **ADR-040 §3.5 iter-A5** — per-slot KV budget in bytes.  `0`
    /// disables enforcement (preserves byte-equivalence for pre-A5
    /// callers).  When > 0, `admit` rejects requests whose
    /// `AdmitRequest::kv_bytes_needed` exceeds this value with
    /// [`AdmitError::SlotBudgetExceeded`] (HTTP 429 + Retry-After
    /// upstream per Decision #19).
    per_slot_kv_budget_bytes: u64,
}

/// Per-queued-request state for FIFO.
#[derive(Debug, Clone)]
struct QueuedFifoRequest {
    request_id: RequestId,
    admitted_at: Instant,
    prompt_tokens: u32,
    max_tokens: u32,
}

/// Per-in-flight-slot state for FIFO.
#[derive(Debug, Clone)]
struct InFlightFifoSlot {
    request_id: RequestId,
    handle: SlotHandle,
    admitted_at: Instant,
    prompt_tokens: u32,
    max_tokens: u32,
    /// `Some(remaining)` when prefill is incomplete (multi-chunk).
    /// `None` after prefill finishes (slot is Decoding).
    prefill_remaining: Option<u32>,
    /// Decode tokens emitted so far.
    tokens_produced: u32,
}

impl FifoSchedulerAdapter {
    /// Build a FIFO scheduler with the given queue cap and no per-slot
    /// KV budget enforcement (iter-A5: `per_slot_kv_budget_bytes = 0`
    /// preserves byte-equivalence with pre-A5 callers).
    pub fn new(queue_capacity: u32) -> Self {
        Self::new_with_kv_budget(queue_capacity, 0)
    }

    /// **ADR-040 §3.5 iter-A5** — build a FIFO scheduler with explicit
    /// per-slot KV byte budget.  `0` disables enforcement (equivalent
    /// to [`Self::new`]).  When > 0, `admit` rejects requests whose
    /// `AdmitRequest::kv_bytes_needed` exceeds `per_slot_kv_budget_bytes`
    /// with [`AdmitError::SlotBudgetExceeded`].
    pub fn new_with_kv_budget(queue_capacity: u32, per_slot_kv_budget_bytes: u64) -> Self {
        let queue_capacity = queue_capacity.max(1);
        Self {
            queue_capacity,
            queue: VecDeque::new(),
            in_flight: None,
            slot_generation: 0,
            next_request_id: 0,
            admitted_total: 0,
            rejected_429_total: 0,
            completed_total: 0,
            per_slot_kv_budget_bytes,
        }
    }

    /// **ADR-040 §3.5 iter-A5** — read the configured per-slot KV
    /// budget (bytes).  `0` means enforcement disabled.
    pub fn per_slot_kv_budget_bytes(&self) -> u64 {
        self.per_slot_kv_budget_bytes
    }

    fn in_flight_count(&self) -> u32 {
        if self.in_flight.is_some() { 1 } else { 0 }
    }

    fn total_admissible(&self) -> u32 {
        self.queue_capacity.saturating_add(1)
    }

    fn next_request_id(&mut self) -> RequestId {
        let id = RequestId(self.next_request_id);
        self.next_request_id = self.next_request_id.wrapping_add(1);
        id
    }

    /// Construct the in-flight slot for the given admit, using the current
    /// generation counter. Honours iter-2.5 M2: a `prompt_tokens == 0`
    /// request transitions directly to decoding (no prefill).
    ///
    /// Caller MUST have already short-circuited the iter-C2.5 M1
    /// `max_tokens == 0` case (see `classify_admit`) — this constructor
    /// always allocates a physical slot and assumes a non-zero decode
    /// budget. Passing `max_tokens == 0` would build a `Decoding {
    /// tokens_produced: 0, max_tokens: 0 }` slot that auto-releases on
    /// the very first `advance_after_decode`, i.e. the stuck-state bug
    /// the M1 short-circuit eliminates.
    fn build_in_flight(
        &self,
        request_id: RequestId,
        admitted_at: Instant,
        prompt_tokens: u32,
        max_tokens: u32,
    ) -> InFlightFifoSlot {
        InFlightFifoSlot {
            request_id,
            handle: SlotHandle { slot_id: SlotId(0), generation: self.slot_generation },
            admitted_at,
            prompt_tokens,
            max_tokens,
            // iter-2.5 M2: prompt_tokens==0 skips prefill entirely.
            prefill_remaining: if prompt_tokens == 0 { None } else { Some(prompt_tokens) },
            tokens_produced: 0,
        }
    }

    /// Cancel a queued request. Returns `true` if found, `false` otherwise.
    /// Does NOT affect the in-flight slot. (iter-2.5 C2 fix — the prior
    /// `release(SlotId(u32::MAX))` removed ALL queued requests at once.)
    pub fn cancel_queued(&mut self, request_id: RequestId) -> bool {
        let before = self.queue.len();
        self.queue.retain(|q| q.request_id != request_id);
        let removed = before > self.queue.len();
        if removed {
            self.completed_total = self.completed_total.saturating_add(1);
        }
        removed
    }

    /// Test-only generation accessor for SlotId(0). Always `SlotId(0)`
    /// for FIFO; ignores the argument's slot_id field.
    pub fn slot_generation(&self, _slot_id: SlotId) -> u64 {
        self.slot_generation
    }

    /// Promote the next queued request into the in-flight slot. Caller
    /// guarantees `self.in_flight.is_none()` (asserted in debug builds).
    ///
    /// iter-C2.5 M1: if the next queued request has `max_tokens == 0`,
    /// it completes-at-promote (bump `completed_total`, do NOT allocate
    /// a physical slot) and the loop continues to the NEXT queued
    /// request. This mirrors the admit-time short-circuit so a future
    /// caller that enqueues zero-budget work directly cannot leak a
    /// stuck slot. Under normal use (`admit` short-circuits FIRST), the
    /// branch is defensive.
    fn promote_one(&mut self) {
        debug_assert!(self.in_flight.is_none(), "promote_one called with in_flight occupied");
        while let Some(q) = self.queue.pop_front() {
            if classify_admit(q.prompt_tokens, q.max_tokens) == InitialAdmitOutcome::CompletedAtAdmit
            {
                self.completed_total = self.completed_total.saturating_add(1);
                continue;
            }
            self.in_flight = Some(self.build_in_flight(
                q.request_id, q.admitted_at, q.prompt_tokens, q.max_tokens,
            ));
            return;
        }
    }

    /// Driver callback: report that `n_consumed` tokens of prefill were
    /// just executed against the handle. Stale handles are no-ops.
    /// Transitions to decoding once `prefill_remaining` hits 0.
    pub fn advance_after_prefill(&mut self, handle: SlotHandle, n_consumed: u32) {
        let Some(slot) = self.in_flight.as_mut() else { return; };
        if slot.handle != handle { return; }
        let Some(remaining) = slot.prefill_remaining else { return; };
        let new_remaining = remaining.saturating_sub(n_consumed);
        slot.prefill_remaining = if new_remaining == 0 { None } else { Some(new_remaining) };
    }

    /// Driver callback: report that the slot just emitted one decode
    /// token. Auto-releases when `tokens_produced >= max_tokens`. Stale
    /// handles are no-ops.
    pub fn advance_after_decode(&mut self, handle: SlotHandle) {
        let should_release = {
            let Some(slot) = self.in_flight.as_mut() else { return; };
            if slot.handle != handle { return; }
            // Reject decode advance for a slot that is still prefilling.
            if slot.prefill_remaining.is_some() { return; }
            slot.tokens_produced = slot.tokens_produced.saturating_add(1);
            slot.tokens_produced >= slot.max_tokens
        };
        if should_release {
            self.release(handle);
        }
    }
}

impl Scheduler for FifoSchedulerAdapter {
    fn policy(&self) -> SchedulerPolicy {
        SchedulerPolicy::FifoSerial
    }

    fn admit(&mut self, req: AdmitRequest) -> Result<RequestSlot, AdmitError> {
        // ADR-040 §3.5 iter-A5 — per-slot KV budget check FIRST.  A
        // request that cannot fit in any single slot's KV budget
        // cannot be served regardless of queue state; rejecting before
        // the queue check is operator-honest (the 429 names the per-
        // request structural problem, not transient capacity).  Budget
        // == 0 disables the check entirely (pre-A5 byte-equivalence).
        if self.per_slot_kv_budget_bytes > 0
            && req.kv_bytes_needed > self.per_slot_kv_budget_bytes
        {
            self.rejected_429_total = self.rejected_429_total.saturating_add(1);
            return Err(AdmitError::SlotBudgetExceeded {
                needed_bytes: req.kv_bytes_needed,
                budget_bytes: self.per_slot_kv_budget_bytes,
            });
        }

        if self.queue.len() as u32 >= self.queue_capacity && self.in_flight.is_some() {
            self.rejected_429_total = self.rejected_429_total.saturating_add(1);
            return Err(AdmitError::QueueFull {
                queue_capacity: self.queue_capacity,
                total_admissible: self.total_admissible(),
                in_flight: self.in_flight_count(),
            });
        }

        let request_id = self.next_request_id();
        let admitted_at = Instant::now();
        self.admitted_total = self.admitted_total.saturating_add(1);

        // cfa-iter-C2.5 M1: zero-budget (`max_tokens == 0`) short-circuits.
        // No physical slot allocated, no queue entry, no generation bump
        // (no slot lifecycle to track). The caller observes
        // `handle.is_none()` and skips the drive loop.
        if classify_admit(req.prompt_tokens, req.max_tokens) == InitialAdmitOutcome::CompletedAtAdmit
        {
            self.completed_total = self.completed_total.saturating_add(1);
            return Ok(RequestSlot {
                request_id,
                handle: None,
                admitted_at,
                prompt_tokens: req.prompt_tokens,
                max_tokens: req.max_tokens,
            });
        }

        let public = if self.in_flight.is_none() {
            let slot = self.build_in_flight(
                request_id, admitted_at, req.prompt_tokens, req.max_tokens,
            );
            let public = RequestSlot {
                request_id,
                handle: Some(slot.handle),
                admitted_at,
                prompt_tokens: req.prompt_tokens,
                max_tokens: req.max_tokens,
            };
            self.in_flight = Some(slot);
            public
        } else {
            self.queue.push_back(QueuedFifoRequest {
                request_id,
                admitted_at,
                prompt_tokens: req.prompt_tokens,
                max_tokens: req.max_tokens,
            });
            RequestSlot {
                request_id,
                handle: None,
                admitted_at,
                prompt_tokens: req.prompt_tokens,
                max_tokens: req.max_tokens,
            }
        };
        Ok(public)
    }

    fn step(&mut self) -> Result<SchedulerStep, StepError> {
        let Some(slot) = self.in_flight.as_ref() else {
            return Ok(SchedulerStep::Idle);
        };
        match slot.prefill_remaining {
            Some(remaining) => Ok(SchedulerStep::Prefill {
                handle: slot.handle,
                n_tokens: remaining,
            }),
            None => Ok(SchedulerStep::Decode {
                handles: vec![slot.handle],
            }),
        }
    }

    fn release(&mut self, handle: SlotHandle) {
        // Stale or unknown handle — silent no-op (iter-2.5 C1).
        let Some(slot) = self.in_flight.as_ref() else { return; };
        if slot.handle != handle { return; }
        self.in_flight = None;
        // Bump generation so the next admit/promote sees a fresh value.
        self.slot_generation = self.slot_generation.saturating_add(1);
        self.completed_total = self.completed_total.saturating_add(1);
        // Promote next queued if any.
        self.promote_one();
    }

    fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            policy: SchedulerPolicy::FifoSerial,
            in_flight_slots: self.in_flight_count(),
            queue_capacity: self.queue_capacity,
            admitted_total: self.admitted_total,
            rejected_429_total: self.rejected_429_total,
            completed_total: self.completed_total,
        }
    }
}

// ---------------------------------------------------------------------------
// InflightBatchedScheduler — production FSM (ADR-040 §3.3)
// ---------------------------------------------------------------------------

/// Per-slot lifecycle state for the InflightBatched FSM.
///
/// iter-2.5 M2 change: a request with `prompt_tokens == 0` admits
/// directly to `Decoding` (no `Prefilling` phase). The Decoding variant
/// is reachable from admit/promote for that case in addition to the
/// normal Prefilling → Decoding transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotPhase {
    Prefilling { tokens_remaining: u32 },
    Decoding { tokens_produced: u32, max_tokens: u32 },
}

/// Outcome of evaluating a fresh admit request's prompt/budget shape.
///
/// (cfa-iter-C2.5 M1 fix.) `initial_admit_outcome` returns this so the
/// admit body can dispatch:
///
/// - `PhaseToPrefilling`: normal admit — slot transitions through
///   `SlotPhase::Prefilling` first.
/// - `PhaseToDecoding`: `prompt_tokens == 0` admit (iter-2.5 M2) — slot
///   skips prefill, transitions directly to `SlotPhase::Decoding`.
/// - `CompletedAtAdmit`: `max_tokens == 0` admit — no decode budget.
///   The scheduler does NOT allocate a physical slot; admit bumps
///   `admitted_total` + `completed_total` and returns
///   `RequestSlot { handle: None, .. }`. The caller observes
///   `handle.is_none()` and skips the drive loop entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitialAdmitOutcome {
    PhaseToPrefilling,
    PhaseToDecoding,
    CompletedAtAdmit,
}

/// Shared admit-shape classifier used by both `FifoSchedulerAdapter` +
/// `InflightBatchedScheduler` (cfa-iter-C2.5 M1). Pure function — no
/// scheduler state touched — so the two adapters share one classifier
/// without a cross-type method call indirection.
///
/// `max_tokens == 0` takes priority over `prompt_tokens == 0`: a
/// request with no decode budget has nothing to do regardless of
/// prompt length. See [`InitialAdmitOutcome`] variant docs.
fn classify_admit(prompt_tokens: u32, max_tokens: u32) -> InitialAdmitOutcome {
    if max_tokens == 0 {
        return InitialAdmitOutcome::CompletedAtAdmit;
    }
    if prompt_tokens == 0 {
        return InitialAdmitOutcome::PhaseToDecoding;
    }
    InitialAdmitOutcome::PhaseToPrefilling
}

/// In-flight slot bookkeeping.
#[derive(Debug, Clone)]
struct InflightSlot {
    request_id: RequestId,
    handle: SlotHandle,
    admitted_at: Instant,
    prompt_tokens: u32,
    max_tokens: u32,
    phase: SlotPhase,
}

/// Queued request bookkeeping (no physical slot yet — iter-2.5 C2).
#[derive(Debug, Clone)]
struct QueuedInflightRequest {
    request_id: RequestId,
    admitted_at: Instant,
    prompt_tokens: u32,
    max_tokens: u32,
}

/// Production continuous-batching scheduler (ADR-040 §3.3).
pub struct InflightBatchedScheduler {
    queue_capacity: u32,
    max_slots: u32,
    in_flight: Vec<InflightSlot>,
    queue: VecDeque<QueuedInflightRequest>,
    slot_id_free_list: Vec<SlotId>,
    next_fresh_slot_id: u32,
    /// Generation counter per physical slot index — bumped on every
    /// release. Length is always exactly `max_slots`. (iter-2.5 C1.)
    slot_generations: Vec<u64>,
    next_request_id: u64,
    admitted_total: u64,
    rejected_429_total: u64,
    completed_total: u64,
    /// **ADR-040 §3.5 iter-A5** — per-slot KV budget in bytes.  `0`
    /// disables enforcement (preserves byte-equivalence for pre-A5
    /// callers).  When > 0, `admit` rejects requests whose
    /// `AdmitRequest::kv_bytes_needed` exceeds this value with
    /// [`AdmitError::SlotBudgetExceeded`] (HTTP 429 + Retry-After
    /// upstream per Decision #19).
    ///
    /// Per ADR-040 §3.5: "per-slot budget = total / max_slots" — the
    /// engine spawn site is responsible for dividing the operator-
    /// supplied `kv_cache_budget_bytes` by `max_slots` before handing
    /// the per-slot value to this constructor.
    per_slot_kv_budget_bytes: u64,
}

impl InflightBatchedScheduler {
    /// Build an InflightBatched scheduler with the given queue cap +
    /// per-slot concurrency cap and no per-slot KV budget enforcement
    /// (iter-A5: `per_slot_kv_budget_bytes = 0` preserves
    /// byte-equivalence with pre-A5 callers).
    pub fn new(queue_capacity: u32, max_slots: u32) -> Self {
        Self::new_with_kv_budget(queue_capacity, max_slots, 0)
    }

    /// **ADR-040 §3.5 iter-A5** — build an InflightBatched scheduler
    /// with explicit per-slot KV byte budget.  `0` disables enforcement
    /// (equivalent to [`Self::new`]).  When > 0, `admit` rejects
    /// requests whose `AdmitRequest::kv_bytes_needed` exceeds
    /// `per_slot_kv_budget_bytes` with [`AdmitError::SlotBudgetExceeded`].
    ///
    /// Per ADR-040 §3.5 ("per-slot budget = total / max_slots") the
    /// caller — `Engine::spawn` / `spawn_with_mode` at the engine seam
    /// — is responsible for performing the division.  This constructor
    /// stores the value verbatim and enforces against it; it does NOT
    /// re-divide by `max_slots`.  Independence per-slot is structural:
    /// each admit checks against the same `per_slot_kv_budget_bytes`
    /// scalar (not an aggregate), so N admits each at exactly the
    /// per-slot budget all succeed (iter-A5 test
    /// `inflight_per_slot_budget_independent_per_slot`).
    pub fn new_with_kv_budget(
        queue_capacity: u32,
        max_slots: u32,
        per_slot_kv_budget_bytes: u64,
    ) -> Self {
        let queue_capacity = queue_capacity.max(1);
        let max_slots = max_slots.max(1);
        Self {
            queue_capacity,
            max_slots,
            in_flight: Vec::with_capacity(max_slots as usize),
            queue: VecDeque::new(),
            slot_id_free_list: Vec::new(),
            next_fresh_slot_id: 0,
            slot_generations: vec![0; max_slots as usize],
            next_request_id: 0,
            admitted_total: 0,
            rejected_429_total: 0,
            completed_total: 0,
            per_slot_kv_budget_bytes,
        }
    }

    /// **ADR-040 §3.5 iter-A5** — read the configured per-slot KV
    /// budget (bytes).  `0` means enforcement disabled.
    pub fn per_slot_kv_budget_bytes(&self) -> u64 {
        self.per_slot_kv_budget_bytes
    }

    /// Allocate a SlotId — prefer a recycled id, otherwise allocate fresh.
    /// Capped at `max_slots` distinct values.
    fn alloc_slot_id(&mut self) -> SlotId {
        if let Some(id) = self.slot_id_free_list.pop() {
            return id;
        }
        debug_assert!(
            self.next_fresh_slot_id < self.max_slots,
            "alloc_slot_id called past max_slots — caller violated in_flight cap invariant"
        );
        let id = SlotId(self.next_fresh_slot_id);
        self.next_fresh_slot_id += 1;
        id
    }

    fn total_admissible(&self) -> u32 {
        self.queue_capacity.saturating_add(self.max_slots)
    }

    fn next_request_id(&mut self) -> RequestId {
        let id = RequestId(self.next_request_id);
        self.next_request_id = self.next_request_id.wrapping_add(1);
        id
    }

    /// Test-only: read the current generation counter for a slot.
    pub fn slot_generation(&self, slot_id: SlotId) -> u64 {
        let idx = slot_id.0 as usize;
        debug_assert!(idx < self.slot_generations.len(),
            "slot_generation called with out-of-bounds slot_id");
        self.slot_generations[idx]
    }

    /// Compute the admit outcome for a request. Thin associated-fn
    /// wrapper around the module-level [`classify_admit`] free function
    /// — kept on the impl so call sites read symmetrically against the
    /// FIFO + InflightBatched adapters.
    fn initial_admit_outcome(prompt_tokens: u32, max_tokens: u32) -> InitialAdmitOutcome {
        classify_admit(prompt_tokens, max_tokens)
    }

    /// Try to promote one queued request into `in_flight`. Returns the
    /// new handle if a promotion happened.
    ///
    /// iter-2.5 C2: queued requests no longer carry sentinel slot ids.
    /// The real handle (slot_id + generation) is freshly allocated here.
    /// iter-2.5 M2: zero-prompt requests promote directly to decoding.
    /// iter-C2.5 M1: zero-budget (`max_tokens == 0`) queued requests
    /// complete-at-promote without allocating a physical slot. This
    /// path is reachable when admit was queued (in_flight was at cap)
    /// for a zero-budget request — `admit` short-circuits zero-budget
    /// FIRST so under normal use this branch is defensive, but the
    /// invariant is enforced here too so future call sites cannot leak
    /// a stuck slot by queuing zero-budget work directly.
    fn try_promote_one_queued(&mut self) -> Option<SlotHandle> {
        if (self.in_flight.len() as u32) >= self.max_slots {
            return None;
        }
        // iter-C2.5 M1: skip past zero-budget queued entries (they
        // complete-at-promote without consuming a physical slot) until
        // we find a real promotion candidate or the queue is empty.
        // Under normal use the admit-time short-circuit prevents zero-
        // budget entries from ever entering the queue, so this loop is
        // defensive; the test
        // `inflight_promote_queued_with_max_tokens_0_does_not_leak`
        // synthesizes the direct-push case to pin the invariant.
        while let Some(q) = self.queue.pop_front() {
            match classify_admit(q.prompt_tokens, q.max_tokens) {
                InitialAdmitOutcome::CompletedAtAdmit => {
                    self.completed_total = self.completed_total.saturating_add(1);
                    continue;
                }
                outcome @ (InitialAdmitOutcome::PhaseToPrefilling
                | InitialAdmitOutcome::PhaseToDecoding) => {
                    let slot_id = self.alloc_slot_id();
                    let generation = self.slot_generations[slot_id.0 as usize];
                    let handle = SlotHandle { slot_id, generation };
                    let phase = match outcome {
                        InitialAdmitOutcome::PhaseToPrefilling => SlotPhase::Prefilling {
                            tokens_remaining: q.prompt_tokens,
                        },
                        InitialAdmitOutcome::PhaseToDecoding => SlotPhase::Decoding {
                            tokens_produced: 0,
                            max_tokens: q.max_tokens,
                        },
                        InitialAdmitOutcome::CompletedAtAdmit => unreachable!(
                            "inner match already discriminated CompletedAtAdmit"
                        ),
                    };
                    self.in_flight.push(InflightSlot {
                        request_id: q.request_id,
                        handle,
                        admitted_at: q.admitted_at,
                        prompt_tokens: q.prompt_tokens,
                        max_tokens: q.max_tokens,
                        phase,
                    });
                    return Some(handle);
                }
            }
        }
        None
    }

    /// Find the index of the first Prefilling slot (insertion / FIFO order).
    fn first_prefilling_idx(&self) -> Option<usize> {
        self.in_flight
            .iter()
            .position(|s| matches!(s.phase, SlotPhase::Prefilling { .. }))
    }

    /// Collect handles for all currently-Decoding slots.
    fn collect_decoding_handles(&self) -> Vec<SlotHandle> {
        self.in_flight
            .iter()
            .filter(|s| matches!(s.phase, SlotPhase::Decoding { .. }))
            .map(|s| s.handle)
            .collect()
    }

    /// Driver callback: report `n_consumed` tokens of prefill were
    /// executed against `handle`. Stale handles, wrong-phase slots, and
    /// unknown handles are silent no-ops.
    pub fn advance_after_prefill(&mut self, handle: SlotHandle, n_consumed: u32) {
        let Some(idx) = self.in_flight.iter().position(|s| s.handle == handle) else {
            return;
        };
        let slot = &mut self.in_flight[idx];
        let SlotPhase::Prefilling { tokens_remaining } = slot.phase else {
            return;
        };
        let new_remaining = tokens_remaining.saturating_sub(n_consumed);
        slot.phase = if new_remaining == 0 {
            SlotPhase::Decoding { tokens_produced: 0, max_tokens: slot.max_tokens }
        } else {
            SlotPhase::Prefilling { tokens_remaining: new_remaining }
        };
    }

    /// Driver callback: report that `handle` just emitted one decode
    /// token. Auto-releases when `tokens_produced >= max_tokens`. Stale
    /// handles, wrong-phase slots, and unknown handles are silent no-ops.
    pub fn advance_after_decode(&mut self, handle: SlotHandle) {
        let Some(idx) = self.in_flight.iter().position(|s| s.handle == handle) else {
            return;
        };
        let SlotPhase::Decoding { tokens_produced, max_tokens } = self.in_flight[idx].phase else {
            return;
        };
        let new_produced = tokens_produced.saturating_add(1);
        if new_produced >= max_tokens {
            // Auto-release: bump generation, recycle id.
            let slot_id = handle.slot_id;
            self.in_flight.remove(idx);
            let gen_idx = slot_id.0 as usize;
            self.slot_generations[gen_idx] =
                self.slot_generations[gen_idx].saturating_add(1);
            self.slot_id_free_list.push(slot_id);
            self.completed_total = self.completed_total.saturating_add(1);
        } else {
            self.in_flight[idx].phase = SlotPhase::Decoding {
                tokens_produced: new_produced,
                max_tokens,
            };
        }
    }

    /// Cancel a queued (not-yet-promoted) request. Returns `true` if
    /// found, `false` otherwise. Does NOT affect in-flight slots.
    /// (iter-2.5 C2 fix.)
    pub fn cancel_queued(&mut self, request_id: RequestId) -> bool {
        let before = self.queue.len();
        self.queue.retain(|q| q.request_id != request_id);
        let removed = before > self.queue.len();
        if removed {
            self.completed_total = self.completed_total.saturating_add(1);
        }
        removed
    }
}

impl Scheduler for InflightBatchedScheduler {
    fn policy(&self) -> SchedulerPolicy {
        SchedulerPolicy::InflightBatched
    }

    fn admit(&mut self, req: AdmitRequest) -> Result<RequestSlot, AdmitError> {
        // ADR-040 §3.5 iter-A5 — per-slot KV budget check FIRST (same
        // ordering as the FIFO adapter; see that admit body for the
        // rationale).  Budget == 0 disables (pre-A5 byte-equivalence).
        if self.per_slot_kv_budget_bytes > 0
            && req.kv_bytes_needed > self.per_slot_kv_budget_bytes
        {
            self.rejected_429_total = self.rejected_429_total.saturating_add(1);
            return Err(AdmitError::SlotBudgetExceeded {
                needed_bytes: req.kv_bytes_needed,
                budget_bytes: self.per_slot_kv_budget_bytes,
            });
        }

        let in_flight = self.in_flight.len() as u32;
        let queued = self.queue.len() as u32;
        if in_flight >= self.max_slots && queued >= self.queue_capacity {
            self.rejected_429_total = self.rejected_429_total.saturating_add(1);
            return Err(AdmitError::QueueFull {
                queue_capacity: self.queue_capacity,
                total_admissible: self.total_admissible(),
                in_flight,
            });
        }

        let request_id = self.next_request_id();
        let admitted_at = Instant::now();
        self.admitted_total = self.admitted_total.saturating_add(1);

        // cfa-iter-C2.5 M1: dispatch on admit outcome. CompletedAtAdmit
        // (max_tokens == 0) short-circuits — no physical slot allocated,
        // no queue entry, handle: None. Other outcomes follow the
        // pre-existing in_flight-or-queue path.
        let outcome = Self::initial_admit_outcome(req.prompt_tokens, req.max_tokens);
        if matches!(outcome, InitialAdmitOutcome::CompletedAtAdmit) {
            self.completed_total = self.completed_total.saturating_add(1);
            return Ok(RequestSlot {
                request_id,
                handle: None,
                admitted_at,
                prompt_tokens: req.prompt_tokens,
                max_tokens: req.max_tokens,
            });
        }

        let public = if in_flight < self.max_slots {
            // Admit directly to in_flight: allocate handle, set phase
            // (honouring iter-2.5 M2 for zero-prompt requests).
            let slot_id = self.alloc_slot_id();
            let generation = self.slot_generations[slot_id.0 as usize];
            let handle = SlotHandle { slot_id, generation };
            let phase = match outcome {
                InitialAdmitOutcome::PhaseToPrefilling => SlotPhase::Prefilling {
                    tokens_remaining: req.prompt_tokens,
                },
                InitialAdmitOutcome::PhaseToDecoding => SlotPhase::Decoding {
                    tokens_produced: 0,
                    max_tokens: req.max_tokens,
                },
                InitialAdmitOutcome::CompletedAtAdmit => unreachable!(
                    "CompletedAtAdmit already handled by the early-return above"
                ),
            };
            self.in_flight.push(InflightSlot {
                request_id,
                handle,
                admitted_at,
                prompt_tokens: req.prompt_tokens,
                max_tokens: req.max_tokens,
                phase,
            });
            RequestSlot {
                request_id,
                handle: Some(handle),
                admitted_at,
                prompt_tokens: req.prompt_tokens,
                max_tokens: req.max_tokens,
            }
        } else {
            self.queue.push_back(QueuedInflightRequest {
                request_id,
                admitted_at,
                prompt_tokens: req.prompt_tokens,
                max_tokens: req.max_tokens,
            });
            RequestSlot {
                request_id,
                handle: None,
                admitted_at,
                prompt_tokens: req.prompt_tokens,
                max_tokens: req.max_tokens,
            }
        };
        Ok(public)
    }

    fn step(&mut self) -> Result<SchedulerStep, StepError> {
        // Priority 1: promote one queued request if room. Promotion does
        // NOT short-circuit — it just adds a slot to the back of
        // `in_flight`; selection below uses first_prefilling_idx() so an
        // older mid-chunk Prefilling slot wins (iter-2.5 C3 fix).
        let _ = self.try_promote_one_queued();

        // Priority 2: find FIRST Prefilling slot in FIFO order.
        let prefill_idx = self.first_prefilling_idx();

        // Collect all Decoding handles (always batched in one forward).
        let decode_handles = self.collect_decoding_handles();

        match (prefill_idx, decode_handles.is_empty()) {
            (Some(idx), false) => {
                // Mixed: oldest Prefilling slot batched with all Decoding.
                let SlotPhase::Prefilling { tokens_remaining } = self.in_flight[idx].phase
                else {
                    // first_prefilling_idx only returns Prefilling slots.
                    return Err(StepError::EngineFailed(
                        "first_prefilling_idx returned non-Prefilling index — invariant violated".to_string(),
                    ));
                };
                Ok(SchedulerStep::Mixed {
                    prefill: self.in_flight[idx].handle,
                    n_prefill_tokens: tokens_remaining.min(DEFAULT_PREFILL_CHUNK_TOKENS),
                    decode_handles,
                })
            }
            (Some(idx), true) => {
                let SlotPhase::Prefilling { tokens_remaining } = self.in_flight[idx].phase
                else {
                    return Err(StepError::EngineFailed(
                        "first_prefilling_idx returned non-Prefilling index — invariant violated".to_string(),
                    ));
                };
                Ok(SchedulerStep::Prefill {
                    handle: self.in_flight[idx].handle,
                    n_tokens: tokens_remaining.min(DEFAULT_PREFILL_CHUNK_TOKENS),
                })
            }
            (None, false) => Ok(SchedulerStep::Decode { handles: decode_handles }),
            (None, true) => Ok(SchedulerStep::Idle),
        }
    }

    fn release(&mut self, handle: SlotHandle) {
        // Stale or unknown handle — silent no-op (iter-2.5 C1).
        let Some(idx) = self.in_flight.iter().position(|s| s.handle == handle) else {
            return;
        };
        self.in_flight.remove(idx);
        let gen_idx = handle.slot_id.0 as usize;
        self.slot_generations[gen_idx] =
            self.slot_generations[gen_idx].saturating_add(1);
        self.slot_id_free_list.push(handle.slot_id);
        self.completed_total = self.completed_total.saturating_add(1);
        // Promotion is the responsibility of step(); not done here to
        // avoid release-vs-step double-promotion races.
    }

    fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            policy: SchedulerPolicy::InflightBatched,
            in_flight_slots: self.in_flight.len() as u32,
            queue_capacity: self.queue_capacity,
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
        AdmitRequest { prompt_tokens, max_tokens, kv_bytes_needed: 0 }
    }

    /// iter-A5 helper — admit request with explicit KV byte cost so
    /// per-slot KV budget tests can pin the SlotBudgetExceeded path.
    fn req_with_kv(prompt_tokens: u32, max_tokens: u32, kv_bytes_needed: u64) -> AdmitRequest {
        AdmitRequest { prompt_tokens, max_tokens, kv_bytes_needed }
    }

    /// Helper to extract the handle from a RequestSlot, panicking with a
    /// clear message if the slot was queued (handle == None).
    fn handle_of(slot: &RequestSlot) -> SlotHandle {
        slot.handle.expect("expected admitted-in-flight slot (handle == Some)")
    }

    // -----------------------------------------------------------------------
    // FIFO contract preservation — load-bearing.
    // -----------------------------------------------------------------------

    #[test]
    fn fifo_admit_then_step_returns_prefill_for_the_admitted_slot() {
        let mut s = FifoSchedulerAdapter::new(4);
        let slot = s.admit(req(11, 32)).expect("admit ok");
        match s.step().expect("step ok") {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, handle_of(&slot));
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
        assert!(a.handle.is_some(), "a admitted to in_flight has Some(handle)");
        assert!(b.handle.is_none(), "b queued has None handle until promoted");
        assert_ne!(a.request_id, b.request_id, "request ids are unique per admit");
        assert_eq!(s.stats().in_flight_slots, 1);

        let a_handle = handle_of(&a);
        assert_eq!(a_handle.slot_id, SlotId(0));

        // First step is prefill for slot a.
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, a_handle);
                assert_eq!(n_tokens, 10);
            }
            other => panic!("expected Prefill for a, got {:?}", other),
        }

        // Release a — b promotes to in-flight on the same physical slot
        // but with a bumped generation.
        s.release(a_handle);
        assert_eq!(s.stats().in_flight_slots, 1);
        assert_eq!(s.stats().completed_total, 1);

        // Step returns Prefill for b — same SlotId(0), bumped generation.
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle.slot_id, SlotId(0));
                assert_eq!(handle.generation, a_handle.generation + 1,
                    "post-promote generation bumped by release");
                assert_eq!(n_tokens, 20);
            }
            other => panic!("expected Prefill for b, got {:?}", other),
        }
    }

    #[test]
    fn fifo_admit_at_capacity_returns_queue_full_with_all_three_fields() {
        let mut s = FifoSchedulerAdapter::new(2);
        let _a = s.admit(req(1, 1)).expect("a in-flight");
        let _b = s.admit(req(1, 1)).expect("b queued");
        let _c = s.admit(req(1, 1)).expect("c queued");
        match s.admit(req(1, 1)) {
            Err(AdmitError::QueueFull { queue_capacity, total_admissible, in_flight }) => {
                assert_eq!(queue_capacity, 2);
                assert_eq!(total_admissible, 3, "FIFO total = queue_capacity (2) + 1 in-flight");
                assert_eq!(in_flight, 1, "FIFO max in_flight is 1");
            }
            other => panic!("expected QueueFull, got {:?}", other),
        }
        assert_eq!(s.stats().rejected_429_total, 1);
    }

    #[test]
    fn fifo_release_unknown_slot_is_noop() {
        let mut s = FifoSchedulerAdapter::new(4);
        let _a = s.admit(req(1, 1)).expect("admit a");
        // Bogus handle: SlotId(0) with future generation.
        s.release(SlotHandle { slot_id: SlotId(9_999), generation: 0 });
        assert_eq!(s.stats().completed_total, 0);
        assert_eq!(s.stats().in_flight_slots, 1);
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, .. } => assert_eq!(handle.slot_id, SlotId(0)),
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

        s.release(handle_of(&a));
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
        let a_handle = handle_of(&a);
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, .. } => assert_eq!(handle, a_handle),
            other => panic!("expected Prefill, got {:?}", other),
        }
        // Drive prefill to completion so step() can return Decode.
        s.advance_after_prefill(a_handle, 8);
        match s.step().unwrap() {
            SchedulerStep::Decode { handles } => {
                assert_eq!(handles.len(), 1);
                assert_eq!(handles[0], a_handle);
            }
            other => panic!("expected Decode, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // iter-1.5 FIFO tests for F3a/F3b/F3c.
    // -----------------------------------------------------------------------

    #[test]
    fn fifo_queue_capacity_zero_normalizes_to_one() {
        let s = FifoSchedulerAdapter::new(0);
        assert_eq!(s.stats().queue_capacity, 1);
    }

    #[test]
    fn fifo_serial_always_assigns_slot_id_0() {
        let mut s = FifoSchedulerAdapter::new(4);
        let a = s.admit(req(1, 1)).expect("admit a");
        let a_handle = handle_of(&a);
        assert_eq!(a_handle.slot_id, SlotId(0));
        s.release(a_handle);
        let b = s.admit(req(1, 1)).expect("admit b");
        let b_handle = handle_of(&b);
        assert_eq!(b_handle.slot_id, SlotId(0), "second admit reuses slot 0");
        assert!(b_handle.generation > a_handle.generation, "generation bumped on release");
    }

    #[test]
    fn fifo_concurrent_admits_under_mutex_match_429_boundary() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        let sched = Arc::new(Mutex::new(FifoSchedulerAdapter::new(2)));
        let mut handles = vec![];
        for i in 0..4 {
            let s = Arc::clone(&sched);
            handles.push(thread::spawn(move || {
                let mut g = s.lock().unwrap();
                g.admit(AdmitRequest { prompt_tokens: 1, max_tokens: 1, kv_bytes_needed: 0 })
                    .map(|slot| (i, slot.request_id))
            }));
        }
        let mut admitted = 0;
        let mut rejected = 0;
        for h in handles {
            match h.join().unwrap() {
                Ok(_) => admitted += 1,
                Err(AdmitError::QueueFull { .. }) => rejected += 1,
                Err(e) => panic!("unexpected error: {:?}", e),
            }
        }
        assert_eq!(admitted, 3);
        assert_eq!(rejected, 1);
    }

    // -----------------------------------------------------------------------
    // iter-2.5 C2 — FIFO RequestId + cancel_queued tests.
    // -----------------------------------------------------------------------

    #[test]
    fn fifo_admit_request_id_is_unique_and_monotonic() {
        let mut s = FifoSchedulerAdapter::new(200);
        let first = s.admit(req(1, 1)).expect("first admit");
        let mut last_id = first.request_id;
        // Release first so subsequent admits succeed.
        s.release(handle_of(&first));
        // Each admit yields a fresh RequestId. We capture a sequence
        // (using release between to keep the queue empty).
        let mut ids = vec![last_id];
        for _ in 0..99 {
            let r = s.admit(req(1, 1)).expect("admit ok");
            assert!(r.request_id.0 > last_id.0, "request id monotonic");
            last_id = r.request_id;
            ids.push(last_id);
            s.release(handle_of(&r));
        }
        // All ids unique.
        let mut sorted = ids.clone();
        sorted.sort_by_key(|id| id.0);
        sorted.dedup_by_key(|id| id.0);
        assert_eq!(sorted.len(), ids.len(), "all request ids distinct");
    }

    #[test]
    fn fifo_cancel_queued_removes_exactly_one_not_all() {
        // cfa-iter2.5 C2: previously `release(SlotId(u32::MAX))` removed
        // ALL queued requests. Now: cancel_queued(rid) removes exactly one.
        let mut s = FifoSchedulerAdapter::new(4);
        let _a = s.admit(req(1, 1)).expect("a in_flight");
        let b = s.admit(req(1, 1)).expect("b queued");
        let _c = s.admit(req(1, 1)).expect("c queued");
        let _d = s.admit(req(1, 1)).expect("d queued");
        assert_eq!(s.stats().in_flight_slots, 1);
        assert_eq!(s.queue.len(), 3, "3 queued");

        // Cancel b — exactly one removed.
        let removed = s.cancel_queued(b.request_id);
        assert!(removed, "cancel returns true for known request");
        assert_eq!(s.queue.len(), 2, "exactly one queued request removed (was 3)");
        assert_eq!(s.stats().in_flight_slots, 1, "in_flight unaffected");
        assert_eq!(s.stats().completed_total, 1, "cancellation bumps completed");
    }

    #[test]
    fn fifo_cancel_queued_unknown_request_id_returns_false() {
        let mut s = FifoSchedulerAdapter::new(4);
        let _a = s.admit(req(1, 1)).expect("a in_flight");
        assert!(!s.cancel_queued(RequestId(99_999)));
        assert_eq!(s.stats().completed_total, 0);
    }

    // -----------------------------------------------------------------------
    // iter-2.5 C1 — FIFO SlotHandle generation tests.
    // -----------------------------------------------------------------------

    #[test]
    fn fifo_slot_generation_bumps_on_release() {
        let mut s = FifoSchedulerAdapter::new(4);
        assert_eq!(s.slot_generation(SlotId(0)), 0);
        let a = s.admit(req(1, 1)).expect("a");
        let a_handle = handle_of(&a);
        assert_eq!(a_handle.generation, 0);
        s.release(a_handle);
        assert_eq!(s.slot_generation(SlotId(0)), 1, "generation bumped on release");
    }

    #[test]
    fn fifo_stale_handle_after_recycle_is_noop() {
        // cfa-iter2.5 C1: stale handle for an auto-released previous
        // occupant must NOT mutate the new occupant.
        let mut s = FifoSchedulerAdapter::new(4);
        let a = s.admit(req(1, 1)).expect("a");
        let a_handle = handle_of(&a);
        // Drive A to completion via advance_after_decode (max_tokens=1
        // auto-release on the very first decode advance after prefill).
        s.advance_after_prefill(a_handle, 1);
        s.advance_after_decode(a_handle); // auto-release
        assert_eq!(s.stats().in_flight_slots, 0);

        // Admit B — promoted onto SlotId(0) with generation 1.
        let b = s.admit(req(5, 3)).expect("b");
        let b_handle = handle_of(&b);
        assert_eq!(b_handle.slot_id, a_handle.slot_id);
        assert_eq!(b_handle.generation, a_handle.generation + 1);

        // Stale advance with a_handle MUST be no-op. If it weren't,
        // it would drop B's prefill_remaining or bump B's tokens_produced.
        s.advance_after_prefill(a_handle, 5);
        // Verify B's state intact.
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, b_handle);
                assert_eq!(n_tokens, 5, "B's prompt_tokens untouched by stale callback");
            }
            other => panic!("expected Prefill for B, got {:?}", other),
        }

        // Stale decode for A also a no-op.
        s.advance_after_decode(a_handle);
        assert_eq!(s.stats().completed_total, 1, "stale advance did not bump completed");
    }

    // -----------------------------------------------------------------------
    // InflightBatched — preserved admit/release/stats tests.
    // -----------------------------------------------------------------------

    #[test]
    fn inflight_admit_succeeds_below_max_slots() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(1, 1)).expect("a");
        let b = s.admit(req(1, 1)).expect("b");
        let a_h = handle_of(&a);
        let b_h = handle_of(&b);
        assert_ne!(a_h.slot_id, b_h.slot_id, "distinct physical slots");
        assert_eq!(s.stats().in_flight_slots, 2);
        assert_eq!(s.stats().admitted_total, 2);
    }

    #[test]
    fn inflight_admit_returns_queue_full_at_capacity_plus_max_slots() {
        let mut s = InflightBatchedScheduler::new(2, 2);
        let _ = s.admit(req(1, 1)).expect("in-flight 0");
        let _ = s.admit(req(1, 1)).expect("in-flight 1");
        let _ = s.admit(req(1, 1)).expect("queued 0");
        let _ = s.admit(req(1, 1)).expect("queued 1");
        match s.admit(req(1, 1)) {
            Err(AdmitError::QueueFull { queue_capacity, total_admissible, in_flight }) => {
                assert_eq!(queue_capacity, 2);
                assert_eq!(total_admissible, 4);
                assert_eq!(in_flight, 2);
            }
            other => panic!("expected QueueFull, got {:?}", other),
        }
        assert_eq!(s.stats().rejected_429_total, 1);
    }

    #[test]
    fn inflight_policy_returns_inflightbatched() {
        let s = InflightBatchedScheduler::new(4, 2);
        assert_eq!(s.policy(), SchedulerPolicy::InflightBatched);
        assert_eq!(s.stats().policy, SchedulerPolicy::InflightBatched);
    }

    #[test]
    fn inflight_release_drops_slot_from_in_flight() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(1, 1)).expect("a");
        let b = s.admit(req(1, 1)).expect("b");
        assert_eq!(s.stats().in_flight_slots, 2);
        s.release(handle_of(&a));
        assert_eq!(s.stats().in_flight_slots, 1);
        assert_eq!(s.stats().completed_total, 1);
        s.release(handle_of(&b));
        assert_eq!(s.stats().in_flight_slots, 0);
        assert_eq!(s.stats().completed_total, 2);
        // Stale handle is a no-op.
        s.release(SlotHandle { slot_id: SlotId(9_999), generation: 0 });
        assert_eq!(s.stats().completed_total, 2);
    }

    #[test]
    fn inflight_stats_counters_advance() {
        // Release does not auto-promote — step() must drive promotion.
        let mut s = InflightBatchedScheduler::new(1, 1);
        let a = s.admit(req(1, 1)).expect("a in-flight");
        let _b = s.admit(req(1, 1)).expect("b queued");
        assert!(s.admit(req(1, 1)).is_err(), "c must be rejected");
        let stats = s.stats();
        assert_eq!(stats.admitted_total, 2);
        assert_eq!(stats.rejected_429_total, 1);
        assert_eq!(stats.completed_total, 0);
        assert_eq!(stats.in_flight_slots, 1);

        s.release(handle_of(&a));
        assert_eq!(s.stats().completed_total, 1);
        assert_eq!(s.stats().in_flight_slots, 0);

        // step() promotes b.
        match s.step().unwrap() {
            SchedulerStep::Prefill { .. } => {}
            other => panic!("expected Prefill for promoted b, got {:?}", other),
        }
        let stats = s.stats();
        assert_eq!(stats.completed_total, 1);
        assert_eq!(stats.in_flight_slots, 1);
    }

    // -----------------------------------------------------------------------
    // InflightBatched — step() FSM tests.
    // -----------------------------------------------------------------------

    #[test]
    fn inflight_step_empty_returns_idle() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        assert_eq!(s.step().unwrap(), SchedulerStep::Idle);
    }

    #[test]
    fn inflight_step_admit_then_step_returns_prefill_for_admitted_slot() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(13, 32)).expect("admit a");
        let a_h = handle_of(&a);
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, a_h);
                assert_eq!(n_tokens, 13);
            }
            other => panic!("expected Prefill, got {:?}", other),
        }
    }

    #[test]
    fn inflight_step_after_prefill_completes_returns_decode() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(3, 8)).expect("admit a");
        let a_h = handle_of(&a);
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, a_h);
                assert_eq!(n_tokens, 3);
            }
            other => panic!("expected Prefill, got {:?}", other),
        }
        s.advance_after_prefill(a_h, 3);
        match s.step().unwrap() {
            SchedulerStep::Decode { handles } => {
                assert_eq!(handles.len(), 1);
                assert_eq!(handles[0], a_h);
            }
            other => panic!("expected Decode, got {:?}", other),
        }
    }

    #[test]
    fn inflight_step_decode_advances_per_token() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(2, 4)).expect("admit a");
        let a_h = handle_of(&a);

        // Prefill.
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, a_h);
                assert_eq!(n_tokens, 2);
            }
            other => panic!("expected Prefill, got {:?}", other),
        }
        s.advance_after_prefill(a_h, 2);

        // 4 decodes.
        for i in 1..=4u32 {
            match s.step().unwrap() {
                SchedulerStep::Decode { handles } => {
                    assert_eq!(handles, vec![a_h], "decode iter {}", i);
                }
                other => panic!("expected Decode at iter {}, got {:?}", i, other),
            }
            s.advance_after_decode(a_h);
        }
        assert_eq!(s.stats().in_flight_slots, 0);
        assert_eq!(s.stats().completed_total, 1);
        assert_eq!(s.step().unwrap(), SchedulerStep::Idle);
    }

    #[test]
    fn inflight_step_promotes_queued_when_slot_frees() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(5, 4)).expect("a in-flight");
        let _b = s.admit(req(7, 4)).expect("b in-flight");
        let _c = s.admit(req(11, 4)).expect("c queued");
        assert_eq!(s.stats().in_flight_slots, 2);

        s.release(handle_of(&a));
        assert_eq!(s.stats().in_flight_slots, 1);
        assert_eq!(s.stats().completed_total, 1);

        // step() — priority-1 promotes c. iter-2.5 C3 fix: the FIRST
        // Prefilling slot in FIFO order (b, since c was just promoted to
        // the back) is what step() emits.
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle: _, n_tokens } => {
                // FIFO across in_flight: b was admitted first, c was just
                // promoted to the END of in_flight, so step picks b (7).
                assert_eq!(n_tokens, 7, "first Prefilling slot in FIFO order is b");
            }
            other => panic!("expected Prefill after promotion, got {:?}", other),
        }
        assert_eq!(s.stats().in_flight_slots, 2, "c was promoted into the freed slot");
    }

    #[test]
    fn inflight_step_returns_mixed_when_prefill_and_decode_coexist() {
        // Mixed emits when ANY Prefilling slot + ANY Decoding slot coexist.
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(4, 8)).expect("admit a");
        let _b = s.admit(req(5, 8)).expect("admit b");
        let a_h = handle_of(&a);

        // Drive both to Decoding.
        s.step().unwrap();
        s.advance_after_prefill(a_h, 4);
        // Find b's handle.
        let b_h = s.in_flight.iter().find(|x| x.handle != a_h)
            .map(|x| x.handle)
            .expect("b in_flight");
        s.step().unwrap();
        s.advance_after_prefill(b_h, 5);

        // Both Decoding now.
        match s.step().unwrap() {
            SchedulerStep::Decode { handles } => assert_eq!(handles.len(), 2),
            other => panic!("expected Decode of 2 slots, got {:?}", other),
        }

        // Admit C queued.
        let c = s.admit(req(9, 8)).expect("admit c queued");
        assert!(c.handle.is_none(), "queued slot has None handle (no sentinel)");

        // Drain A.
        for _ in 0..8 {
            s.advance_after_decode(a_h);
        }
        assert_eq!(s.stats().in_flight_slots, 1, "A auto-released");

        // step() — promotes C; B Decoding; FIRST Prefilling slot in FIFO
        // order is C (only Prefilling slot). Emits Mixed.
        match s.step().unwrap() {
            SchedulerStep::Mixed { prefill, n_prefill_tokens, decode_handles } => {
                assert_eq!(n_prefill_tokens, 9);
                assert_eq!(decode_handles.len(), 1);
                assert_eq!(decode_handles[0], b_h);
                assert_eq!(prefill.slot_id, SlotId(0), "c got a's recycled slot id");
                assert!(prefill.generation > a_h.generation,
                    "c's generation is bumped past a's");
            }
            other => panic!("expected Mixed, got {:?}", other),
        }
    }

    #[test]
    fn inflight_step_chunks_prefill_at_default_size() {
        assert_eq!(DEFAULT_PREFILL_CHUNK_TOKENS, 512);
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(1500, 4)).expect("admit a");
        let a_h = handle_of(&a);

        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, a_h);
                assert_eq!(n_tokens, 512);
            }
            other => panic!("expected Prefill(512), got {:?}", other),
        }
        s.advance_after_prefill(a_h, 512);

        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, a_h);
                assert_eq!(n_tokens, 512);
            }
            other => panic!("expected Prefill(512), got {:?}", other),
        }
        s.advance_after_prefill(a_h, 512);

        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, a_h);
                assert_eq!(n_tokens, 476);
            }
            other => panic!("expected Prefill(476), got {:?}", other),
        }
        s.advance_after_prefill(a_h, 476);

        match s.step().unwrap() {
            SchedulerStep::Decode { handles } => {
                assert_eq!(handles, vec![a_h]);
            }
            other => panic!("expected Decode after chunked prefill, got {:?}", other),
        }
    }

    #[test]
    fn inflight_step_auto_releases_on_max_tokens() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(1, 2)).expect("admit a");
        let a_h = handle_of(&a);

        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, a_h);
                assert_eq!(n_tokens, 1);
            }
            other => panic!("expected Prefill, got {:?}", other),
        }
        s.advance_after_prefill(a_h, 1);

        match s.step().unwrap() {
            SchedulerStep::Decode { handles } => assert_eq!(handles, vec![a_h]),
            other => panic!("expected Decode iter 1, got {:?}", other),
        }
        s.advance_after_decode(a_h);
        assert_eq!(s.stats().in_flight_slots, 1);
        assert_eq!(s.stats().completed_total, 0);

        match s.step().unwrap() {
            SchedulerStep::Decode { handles } => assert_eq!(handles, vec![a_h]),
            other => panic!("expected Decode iter 2, got {:?}", other),
        }
        s.advance_after_decode(a_h);
        assert_eq!(s.stats().in_flight_slots, 0);
        assert_eq!(s.stats().completed_total, 1);

        assert_eq!(s.step().unwrap(), SchedulerStep::Idle);
    }

    #[test]
    fn inflight_advance_after_prefill_unknown_slot_is_noop() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(5, 8)).expect("admit a");
        let a_h = handle_of(&a);
        // Unknown handle.
        s.advance_after_prefill(SlotHandle { slot_id: SlotId(9_999), generation: 0 }, 100);
        // a's state unchanged.
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, a_h);
                assert_eq!(n_tokens, 5);
            }
            other => panic!("expected Prefill, got {:?}", other),
        }
        // Wrong-phase: advance after a is Decoding.
        s.advance_after_prefill(a_h, 5);
        s.advance_after_prefill(a_h, 1); // no-op
        match s.step().unwrap() {
            SchedulerStep::Decode { handles } => {
                assert_eq!(handles, vec![a_h]);
            }
            other => panic!("expected Decode, got {:?}", other),
        }
    }

    #[test]
    fn inflight_advance_after_decode_overflow_is_clamped_at_max_tokens() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(1, 3)).expect("admit a, max_tokens=3");
        let a_h = handle_of(&a);
        s.advance_after_prefill(a_h, 1);

        s.advance_after_decode(a_h);
        assert_eq!(s.stats().in_flight_slots, 1);
        s.advance_after_decode(a_h);
        assert_eq!(s.stats().in_flight_slots, 1);
        s.advance_after_decode(a_h);
        assert_eq!(s.stats().in_flight_slots, 0);
        assert_eq!(s.stats().completed_total, 1);

        // Stale advances (handle no longer in_flight) are no-ops.
        s.advance_after_decode(a_h);
        s.advance_after_decode(a_h);
        assert_eq!(s.stats().completed_total, 1);
    }

    #[test]
    fn inflight_max_slots_zero_normalizes_to_one() {
        let mut s = InflightBatchedScheduler::new(4, 0);
        let a = s.admit(req(3, 1)).expect("admit must succeed");
        let a_h = handle_of(&a);
        assert_eq!(s.stats().in_flight_slots, 1);
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, a_h);
                assert_eq!(n_tokens, 3);
            }
            other => panic!("expected Prefill, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // iter-2.5 C1 — InflightBatched SlotHandle generation tests.
    // -----------------------------------------------------------------------

    #[test]
    fn inflight_slot_generation_bumps_on_release() {
        let mut s = InflightBatchedScheduler::new(4, 2);
        assert_eq!(s.slot_generation(SlotId(0)), 0);
        let a = s.admit(req(1, 1)).expect("a");
        let a_h = handle_of(&a);
        assert_eq!(a_h.generation, 0);
        s.release(a_h);
        assert_eq!(s.slot_generation(SlotId(0)), 1, "release bumps generation");
    }

    #[test]
    fn inflight_slot_generation_bumps_on_auto_release() {
        // Auto-release (via advance_after_decode max-tokens path) also
        // bumps the generation counter.
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(1, 1)).expect("a");
        let a_h = handle_of(&a);
        s.advance_after_prefill(a_h, 1);
        s.advance_after_decode(a_h); // auto-release
        assert_eq!(s.slot_generation(a_h.slot_id), 1,
            "auto-release bumps generation just like explicit release");
    }

    #[test]
    fn inflight_stale_callback_after_recycle_is_noop_not_corrupt() {
        // cfa-iter2.5 C1: after auto-release recycles SlotId, the OLD
        // handle for that slot must NOT mutate the NEW occupant.
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(1, 1)).expect("a"); // prompt=1 max=1
        let a_h = handle_of(&a);
        // Drive A through to auto-release.
        s.step().unwrap();
        s.advance_after_prefill(a_h, 1);
        s.step().unwrap();
        s.advance_after_decode(a_h); // auto-release
        assert_eq!(s.stats().in_flight_slots, 0);

        // Admit B — gets recycled SlotId with bumped generation.
        let b = s.admit(req(8, 4)).expect("b");
        let b_h = handle_of(&b);
        assert_eq!(b_h.slot_id, a_h.slot_id, "B got A's recycled slot id");
        assert_eq!(b_h.generation, a_h.generation + 1,
            "B's generation is exactly one past A's");

        // Stale callback for A — MUST be no-op, MUST NOT touch B's state.
        s.advance_after_prefill(a_h, 5);
        // Verify B's tokens_remaining unchanged via step().
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, b_h);
                assert_eq!(n_tokens, 8, "B's prompt untouched by stale callback");
            }
            other => panic!("expected Prefill for B, got {:?}", other),
        }
        // Also a stale decode for A is a no-op.
        s.advance_after_decode(a_h);
        assert_eq!(s.stats().completed_total, 1,
            "stale decode did not double-complete");
    }

    #[test]
    fn inflight_handle_carries_correct_generation_at_promote() {
        // Queued requests carry None handle at admit; the handle observed
        // at promotion time (via SchedulerStep::Prefill { handle, .. })
        // MUST have the correct generation for the freshly-allocated slot.
        let mut s = InflightBatchedScheduler::new(4, 1);
        let a = s.admit(req(1, 1)).expect("a in_flight");
        let a_h = handle_of(&a);
        let b = s.admit(req(7, 1)).expect("b queued");
        assert!(b.handle.is_none(), "b queued has None handle");

        // Auto-release A.
        s.advance_after_prefill(a_h, 1);
        s.advance_after_decode(a_h);
        assert_eq!(s.stats().in_flight_slots, 0);

        // step() promotes B and emits Prefill with B's handle.
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle: b_h, n_tokens } => {
                assert_eq!(n_tokens, 7);
                assert_eq!(b_h.slot_id, a_h.slot_id, "B recycled A's slot id");
                assert_eq!(b_h.generation, a_h.generation + 1,
                    "B's promoted handle generation == prior slot generation + 1");
                // The slot_generations array now matches.
                assert_eq!(s.slot_generation(b_h.slot_id), b_h.generation);
            }
            other => panic!("expected Prefill for promoted B, got {:?}", other),
        }
    }

    #[test]
    fn inflight_concurrent_advance_under_mutex_with_handle_safe() {
        // Rewrite of the prior `inflight_concurrent_advance_pattern_under_mutex`
        // test to use SlotHandle. With the C1 generation discipline, even
        // a mutex-drop between step() and the driver callback cannot
        // corrupt state — stale callbacks no-op.
        use std::sync::{Arc, Mutex};
        use std::thread;

        let sched = Arc::new(Mutex::new(InflightBatchedScheduler::new(6, 3)));
        let mut handles = vec![];
        for thread_idx in 0..3u32 {
            let s = Arc::clone(&sched);
            handles.push(thread::spawn(move || {
                let _ = {
                    let mut g = s.lock().unwrap();
                    g.admit(AdmitRequest { prompt_tokens: 1, max_tokens: 4, kv_bytes_needed: 0 })
                        .expect("admit ok")
                };
                for _ in 0..5 {
                    let action: SchedulerStep = {
                        let mut g = s.lock().unwrap();
                        g.step().unwrap()
                    };
                    match action {
                        SchedulerStep::Prefill { handle, n_tokens } => {
                            let mut g = s.lock().unwrap();
                            g.advance_after_prefill(handle, n_tokens);
                        }
                        SchedulerStep::Decode { handles } => {
                            let mut g = s.lock().unwrap();
                            for h in handles {
                                g.advance_after_decode(h);
                            }
                        }
                        SchedulerStep::Mixed { prefill, n_prefill_tokens, decode_handles } => {
                            let mut g = s.lock().unwrap();
                            g.advance_after_prefill(prefill, n_prefill_tokens);
                            for h in decode_handles {
                                g.advance_after_decode(h);
                            }
                        }
                        SchedulerStep::Idle => {}
                    }
                }
                let _ = thread_idx;
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let stats = sched.lock().unwrap().stats();
        assert_eq!(stats.admitted_total, 3);
        assert_eq!(stats.completed_total, 3);
        assert_eq!(stats.in_flight_slots, 0);
        assert_eq!(stats.rejected_429_total, 0);
    }

    // -----------------------------------------------------------------------
    // iter-2.5 C2 — InflightBatched RequestId + cancel_queued tests.
    // -----------------------------------------------------------------------

    #[test]
    fn inflight_admit_returns_request_id_and_no_handle_when_queued() {
        let mut s = InflightBatchedScheduler::new(2, 1);
        let a = s.admit(req(1, 1)).expect("a");
        let b = s.admit(req(1, 1)).expect("b queued");
        assert!(a.handle.is_some(), "a in_flight has Some handle");
        assert!(b.handle.is_none(), "b queued has None handle");
        assert_ne!(a.request_id, b.request_id);
    }

    #[test]
    fn inflight_cancel_queued_removes_exactly_one_not_all() {
        let mut s = InflightBatchedScheduler::new(4, 1);
        let _a = s.admit(req(1, 1)).expect("a in_flight");
        let q1 = s.admit(req(1, 1)).expect("q1 queued");
        let q2 = s.admit(req(1, 1)).expect("q2 queued");
        let _q3 = s.admit(req(1, 1)).expect("q3 queued");
        assert_eq!(s.queue.len(), 3);

        // Cancel q2 — exactly one removed.
        let removed = s.cancel_queued(q2.request_id);
        assert!(removed);
        assert_eq!(s.queue.len(), 2,
            "exactly one queued request removed (regression: previously ALL queued removed)");
        assert_eq!(s.stats().in_flight_slots, 1, "in_flight unaffected by cancel_queued");

        // Verify q1 + q3 still there; q2 truly gone.
        let remaining_ids: Vec<_> = s.queue.iter().map(|q| q.request_id).collect();
        assert!(remaining_ids.contains(&q1.request_id), "q1 still queued");
        assert!(!remaining_ids.contains(&q2.request_id), "q2 removed");
    }

    #[test]
    fn inflight_cancel_queued_unknown_request_id_returns_false() {
        let mut s = InflightBatchedScheduler::new(4, 1);
        let _a = s.admit(req(1, 1)).expect("a in_flight");
        assert!(!s.cancel_queued(RequestId(99_999)));
        assert_eq!(s.stats().completed_total, 0);
    }

    #[test]
    fn inflight_admit_request_id_is_unique_and_monotonic() {
        let mut s = InflightBatchedScheduler::new(200, 1);
        // Admit 100 (1 in_flight + 99 queued; queue cap 200 covers it).
        let mut last = None;
        let mut ids = vec![];
        for _ in 0..100 {
            let r = s.admit(req(1, 1)).expect("admit ok");
            if let Some(l) = last {
                let prev: RequestId = l;
                assert!(r.request_id.0 > prev.0, "request_id monotonic");
            }
            last = Some(r.request_id);
            ids.push(r.request_id);
        }
        assert_eq!(ids.len(), 100);
        let mut sorted = ids.clone();
        sorted.sort_by_key(|id| id.0);
        sorted.dedup_by_key(|id| id.0);
        assert_eq!(sorted.len(), 100, "all 100 request ids distinct");
    }

    // -----------------------------------------------------------------------
    // iter-2.5 C3 — step() priority: older Prefilling wins over promoted.
    // -----------------------------------------------------------------------

    #[test]
    fn inflight_step_priority_older_prefilling_wins_with_promotion() {
        // cfa-iter2.5 C3: a freshly-promoted slot must NOT jump ahead of
        // an older mid-chunk Prefilling slot. The fix: step() picks the
        // FIRST Prefilling slot in insertion (FIFO) order.
        //
        // Construction:
        //   max_slots=3, queue_capacity=2
        //   Admit A (prompt=1000, max=4) → in_flight slot 0, Prefilling{1000}
        //   step() → Prefill(handle-A, 512); advance(512) → Prefilling{488}
        //   Admit B (prompt=10, max=4) → in_flight slot 1, Prefilling{10}
        //   step() picks A (older Prefilling) → Prefill(A, 488)
        //   Actually we drive B to Decoding to set up the scenario:
        //   Step → emits Prefill for A (FIFO order); advance(488) → A Decoding
        //   Now A Decoding, B Prefilling. Set up admit C also Prefilling.
        //
        // Simpler scenario aligning with the brief:
        //   Admit A (prompt=1000); step → Prefill(A, 512); advance(512)
        //     ⇒ A: Prefilling{488}
        //   Admit B (prompt=10); manually advance B's prefill (10) ⇒ B: Decoding
        //   Admit C (prompt=5) into a remaining in_flight slot ⇒ C: Prefilling{5}
        //   step() — no promotion (no queued); first Prefilling in FIFO = A
        //   ⇒ Mixed { prefill: A, n_prefill_tokens: 488, decode_handles: [B] }
        //   NOT Mixed { prefill: C, ... }.
        let mut s = InflightBatchedScheduler::new(4, 3);
        let a = s.admit(req(1000, 4)).expect("a");
        let a_h = handle_of(&a);
        match s.step().unwrap() {
            SchedulerStep::Prefill { handle, n_tokens } => {
                assert_eq!(handle, a_h);
                assert_eq!(n_tokens, 512);
            }
            other => panic!("expected Prefill(A, 512), got {:?}", other),
        }
        s.advance_after_prefill(a_h, 512);
        // A: Prefilling{488}.

        let b = s.admit(req(10, 4)).expect("b");
        let b_h = handle_of(&b);
        // Need to drive B's prefill via step() so the FSM tracks it.
        // step() will pick the FIRST Prefilling slot — which is A. We
        // want B to land in Decoding before our final assertion, so we
        // bypass step() and use the direct callback API to advance B
        // (representing a multi-step engine driver that interleaves).
        s.advance_after_prefill(b_h, 10);
        // B is now Decoding.

        let c = s.admit(req(5, 4)).expect("c");
        let c_h = handle_of(&c);
        // C: Prefilling{5}.

        // Now: A Prefilling{488}, B Decoding, C Prefilling{5}.
        // step() — no queued (everything in_flight); first Prefilling in
        // FIFO order is A (slot 0). decode = [B].
        match s.step().unwrap() {
            SchedulerStep::Mixed { prefill, n_prefill_tokens, decode_handles } => {
                assert_eq!(prefill, a_h,
                    "older Prefilling slot A wins over newer C (cfa-iter2.5 C3)");
                assert_eq!(n_prefill_tokens, 488,
                    "A's mid-chunk continuation must not be starved");
                assert_eq!(decode_handles, vec![b_h]);
                let _ = c_h; // C waits its turn (still Prefilling{5})
            }
            other => panic!("expected Mixed(A, 488, [B]), got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // iter-2.5 M2 — prompt_tokens=0 / max_tokens=0 edge cases.
    // -----------------------------------------------------------------------

    #[test]
    fn inflight_admit_with_prompt_tokens_0_skips_to_decoding() {
        // cfa-iter2.5 M2: prompt_tokens=0 admits directly to Decoding.
        // step() must emit Decode, NOT Prefill { n_tokens: 0 }.
        let mut s = InflightBatchedScheduler::new(4, 2);
        let a = s.admit(req(0, 3)).expect("admit a with empty prompt");
        let a_h = handle_of(&a);
        match s.step().unwrap() {
            SchedulerStep::Decode { handles } => {
                assert_eq!(handles, vec![a_h],
                    "zero-prompt admit transitions directly to Decoding");
            }
            other => panic!("expected Decode (M2), got {:?}", other),
        }
        // Decode 3 times → auto-release.
        s.advance_after_decode(a_h);
        s.advance_after_decode(a_h);
        s.advance_after_decode(a_h);
        assert_eq!(s.stats().in_flight_slots, 0);
        assert_eq!(s.stats().completed_total, 1);
    }

    #[test]
    fn inflight_promote_queued_with_prompt_tokens_0_skips_to_decoding() {
        // cfa-iter2.5 M2: a queued zero-prompt request, on promotion,
        // also transitions directly to Decoding.
        let mut s = InflightBatchedScheduler::new(4, 1);
        let a = s.admit(req(5, 1)).expect("a in_flight");
        let a_h = handle_of(&a);
        let _b = s.admit(req(0, 2)).expect("b queued with zero prompt");

        // Drain A.
        s.advance_after_prefill(a_h, 5);
        s.advance_after_decode(a_h); // auto-release

        // step() promotes B. B has prompt_tokens=0, so it goes straight
        // to Decoding and step() returns Decode (NOT Prefill).
        match s.step().unwrap() {
            SchedulerStep::Decode { handles } => {
                assert_eq!(handles.len(), 1,
                    "zero-prompt promoted slot skips Prefilling, emits Decode");
            }
            other => panic!("expected Decode for promoted zero-prompt B, got {:?}", other),
        }
    }

    #[test]
    fn fifo_admit_with_prompt_tokens_0_skips_to_decoding() {
        // cfa-iter2.5 M2: same M2 fix for FIFO adapter.
        let mut s = FifoSchedulerAdapter::new(4);
        let a = s.admit(req(0, 2)).expect("admit a");
        let a_h = handle_of(&a);
        match s.step().unwrap() {
            SchedulerStep::Decode { handles } => {
                assert_eq!(handles, vec![a_h],
                    "FIFO: zero-prompt admit skips Prefilling");
            }
            other => panic!("expected Decode (M2 FIFO), got {:?}", other),
        }
        s.advance_after_decode(a_h);
        s.advance_after_decode(a_h); // auto-release
        assert_eq!(s.stats().in_flight_slots, 0);
    }

    // -----------------------------------------------------------------------
    // iter-C2.5 M1 — `max_tokens == 0` admit short-circuits with
    // handle: None, does NOT leak an in-flight slot.
    //
    // Codex /cfa rev-1 finding M1: prior iter-2.5 inline comment claimed
    // "max_tokens == 0 auto-releases at admit time" but `initial_phase`
    // still pushed a `Decoding { tokens_produced: 0, max_tokens: 0 }`
    // slot into `in_flight` — only the Embed worker arm's explicit post-
    // prefill `release` avoided the stuck-slot bug, and that was
    // accidental: any future SlotAware / generate path that observed a
    // zero-budget admit as Decoding work would leak the slot.
    //
    // iter-C2.5 M1 fix (this iter):
    // - `classify_admit` returns `CompletedAtAdmit` for `max_tokens == 0`
    //   regardless of `prompt_tokens` (max_tokens takes priority — no
    //   decode budget means nothing to do).
    // - FIFO + Inflight `admit` short-circuit: bump `admitted_total` +
    //   `completed_total`, return `RequestSlot { handle: None, .. }`,
    //   NO slot allocated.
    // - `try_promote_one_queued` (Inflight) + `promote_one` (FIFO) also
    //   short-circuit zero-budget queued items so even a future caller
    //   that bypasses the admit-time short-circuit cannot leak a slot.
    // -----------------------------------------------------------------------

    #[test]
    fn fifo_admit_with_max_tokens_0_returns_handle_none() {
        // cfa-iter-C2.5 M1: max_tokens=0 admit → handle: None, no slot
        // allocated, completed_total bumped.
        let mut s = FifoSchedulerAdapter::new(4);
        let r = s.admit(req(8, 0)).expect("zero-budget admit must succeed");
        assert!(r.handle.is_none(),
            "max_tokens=0 admit must return handle: None (no slot allocated)");
        assert_eq!(r.prompt_tokens, 8, "RequestSlot still echoes input prompt_tokens");
        assert_eq!(r.max_tokens, 0, "RequestSlot still echoes input max_tokens");

        let stats = s.stats();
        assert_eq!(stats.admitted_total, 1, "admitted_total bumped");
        assert_eq!(stats.completed_total, 1,
            "completed_total bumped at admit time (zero-budget short-circuit)");
        assert_eq!(stats.in_flight_slots, 0,
            "no physical slot allocated for zero-budget admit");

        // step() returns Idle — the would-have-been slot doesn't exist.
        assert_eq!(s.step().unwrap(), SchedulerStep::Idle,
            "no in-flight slot, no queued slot → Idle");

        // A subsequent normal admit lands cleanly in the unoccupied slot.
        let b = s.admit(req(3, 5)).expect("normal admit after zero-budget");
        let b_h = handle_of(&b);
        assert_eq!(b_h.slot_id, SlotId(0),
            "next admit gets the never-allocated slot 0");
        assert_eq!(b_h.generation, 0,
            "no generation bump (no release happened — slot was never allocated)");
    }

    #[test]
    fn inflight_admit_with_max_tokens_0_does_not_leak_slot() {
        // cfa-iter-C2.5 M1: N admits with max_tokens=0 under max_slots=4
        // → in_flight_slots stays at 0; completed_total bumps each time.
        let mut s = InflightBatchedScheduler::new(8, 4);
        for i in 0..16 {
            let r = s.admit(req(3, 0)).expect("zero-budget admit must succeed");
            assert!(r.handle.is_none(),
                "iter {}: max_tokens=0 admit must return handle: None", i);
        }
        let stats = s.stats();
        assert_eq!(stats.in_flight_slots, 0,
            "16 zero-budget admits must leak ZERO in-flight slots (regression: \
             prior iter-2.5 pushed Decoding{{0,0}} into in_flight)");
        assert_eq!(stats.admitted_total, 16, "all 16 counted as admitted");
        assert_eq!(stats.completed_total, 16,
            "all 16 counted as completed-at-admit (no slot lifecycle)");
        assert_eq!(stats.rejected_429_total, 0,
            "no QueueFull — short-circuit happens after capacity check");

        // step() returns Idle — no slots ever allocated.
        assert_eq!(s.step().unwrap(), SchedulerStep::Idle);
    }

    #[test]
    fn fifo_admit_prompt_tokens_0_max_tokens_0_no_leak() {
        // cfa-iter-C2.5 M1: both zeros → CompletedAtAdmit (max_tokens
        // takes priority over prompt_tokens). No slot allocated.
        let mut s = FifoSchedulerAdapter::new(4);
        let r = s.admit(req(0, 0)).expect("both-zeros admit must succeed");
        assert!(r.handle.is_none(),
            "prompt_tokens=0 AND max_tokens=0 → handle: None (max_tokens wins)");
        let stats = s.stats();
        assert_eq!(stats.admitted_total, 1);
        assert_eq!(stats.completed_total, 1);
        assert_eq!(stats.in_flight_slots, 0);

        // Inflight side: same shape.
        let mut s2 = InflightBatchedScheduler::new(4, 2);
        let r2 = s2.admit(req(0, 0)).expect("both-zeros inflight admit ok");
        assert!(r2.handle.is_none(),
            "inflight prompt_tokens=0 AND max_tokens=0 → handle: None");
        let stats2 = s2.stats();
        assert_eq!(stats2.in_flight_slots, 0);
        assert_eq!(stats2.completed_total, 1);
    }

    #[test]
    fn inflight_promote_queued_with_max_tokens_0_does_not_leak() {
        // cfa-iter-C2.5 M1 defensive-promote pin: a zero-budget request
        // CANNOT enter the queue via the normal `admit` path (the admit-
        // time short-circuit fires FIRST regardless of in_flight/queue
        // occupancy). This test exercises the admit-time short-circuit
        // under the same shape that previously would have queued.
        //
        // Setup: fill max_slots with normal admits so the next admit
        // WOULD have been queued under the pre-M1 behaviour. Now the
        // zero-budget admit still short-circuits → handle: None,
        // in_flight unchanged, no queued entry.
        let mut s = InflightBatchedScheduler::new(4, 2);
        let _a = s.admit(req(5, 4)).expect("a normal admit, in_flight slot 0");
        let _b = s.admit(req(7, 4)).expect("b normal admit, in_flight slot 1");
        assert_eq!(s.stats().in_flight_slots, 2);
        assert_eq!(s.queue.len(), 0);

        // Now the would-be-queued admit is zero-budget; it short-circuits.
        let c = s.admit(req(11, 0)).expect("c zero-budget admit ok");
        assert!(c.handle.is_none(),
            "zero-budget admit at in_flight-cap still short-circuits — not queued");
        assert_eq!(s.queue.len(), 0,
            "zero-budget request MUST NOT enter the queue (cfa-iter-C2.5 M1)");
        assert_eq!(s.stats().in_flight_slots, 2, "in_flight unchanged");
        assert_eq!(s.stats().completed_total, 1, "c counted as completed");

        // Defensive layer: even if a future caller bypasses `admit` and
        // pushes a zero-budget request directly into the queue, the
        // `try_promote_one_queued` path also short-circuits. Synthesize
        // a queued zero-budget entry and drive promotion via release →
        // step.
        s.queue.push_back(QueuedInflightRequest {
            request_id: RequestId(99_999),
            admitted_at: Instant::now(),
            prompt_tokens: 5,
            max_tokens: 0,
        });
        s.queue.push_back(QueuedInflightRequest {
            request_id: RequestId(99_998),
            admitted_at: Instant::now(),
            prompt_tokens: 7,
            max_tokens: 4, // normal — should be the one promoted
        });
        let completed_before = s.stats().completed_total;
        let in_flight_before = s.stats().in_flight_slots;
        // Drop one normal in_flight slot so promotion has room.
        s.release(handle_of(&_a));
        // step() should promote PAST the zero-budget queued entry to
        // reach the normal one. Zero-budget queued entry completes-at-
        // promote (bumps completed_total) but consumes no slot.
        let _ = s.step().unwrap();
        let stats = s.stats();
        // a's release bumped completed_total by 1; zero-budget queued
        // promote-skip bumped it by 1 more.
        assert!(stats.completed_total >= completed_before + 2,
            "release + zero-budget-queued-skip both bump completed_total \
             (was {}, now {})", completed_before, stats.completed_total);
        assert_eq!(stats.in_flight_slots, in_flight_before,
            "in_flight unchanged: released a → promoted normal request \
             past skipped zero-budget queued entry");
    }

    // -----------------------------------------------------------------------
    // Cross-cutting
    // -----------------------------------------------------------------------

    #[test]
    fn request_slot_admitted_at_is_monotonic() {
        let mut s = FifoSchedulerAdapter::new(4);
        let a = s.admit(req(1, 1)).expect("a");
        let b = s.admit(req(1, 1)).expect("b");
        assert!(b.admitted_at >= a.admitted_at);
    }

    #[test]
    fn admit_error_queue_full_names_queue_capacity_and_total_admissible_and_in_flight() {
        let err = AdmitError::QueueFull {
            queue_capacity: 7,
            total_admissible: 8,
            in_flight: 3,
        };
        let dbg = format!("{:?}", err);
        assert!(dbg.contains("queue_capacity"));
        assert!(dbg.contains("total_admissible"));
        assert!(dbg.contains("in_flight"));
        assert!(dbg.contains('7'));
        assert!(dbg.contains('8'));
        assert!(dbg.contains('3'));

        let disp = format!("{}", err);
        assert!(disp.contains("queue_capacity=7"));
        assert!(disp.contains("total_admissible=8"));
        assert!(disp.contains("in_flight=3"));
    }

    // -----------------------------------------------------------------------
    // iter-A5 — per-slot OOM + budget enforcement (ADR-040 §3.5)
    //
    // The scheduler tracks the per-slot KV budget in BYTES; callers
    // compute `AdmitRequest::kv_bytes_needed` per-arch (Phase C2c+
    // wiring) and the scheduler enforces at admit time.  Budget == 0
    // ⇒ enforcement disabled ⇒ byte-equivalent to pre-A5.
    // -----------------------------------------------------------------------

    #[test]
    fn fifo_admit_below_per_slot_budget_succeeds() {
        // Budget 1 MiB; admit asks for 512 KiB ⇒ accepted.
        let mut s = FifoSchedulerAdapter::new_with_kv_budget(4, 1024 * 1024);
        let r = s
            .admit(req_with_kv(8, 16, 512 * 1024))
            .expect("admit below budget must succeed");
        assert!(r.handle.is_some(),
            "below-budget admit lands in_flight with Some(handle)");
        assert_eq!(s.stats().admitted_total, 1);
        assert_eq!(s.stats().rejected_429_total, 0);
        assert_eq!(s.per_slot_kv_budget_bytes(), 1024 * 1024);
    }

    #[test]
    fn fifo_admit_above_per_slot_budget_errors_with_named_fields() {
        // Budget 1 MiB; admit asks for 2 MiB ⇒ rejected.
        let mut s = FifoSchedulerAdapter::new_with_kv_budget(4, 1024 * 1024);
        let needed = 2 * 1024 * 1024;
        match s.admit(req_with_kv(1024, 64, needed)) {
            Err(AdmitError::SlotBudgetExceeded { needed_bytes, budget_bytes }) => {
                assert_eq!(needed_bytes, needed,
                    "error names the request's needed bytes");
                assert_eq!(budget_bytes, 1024 * 1024,
                    "error names the per-slot budget bytes");
            }
            other => panic!("expected SlotBudgetExceeded, got {:?}", other),
        }
        // Counters: rejected_429_total bumped; admitted_total NOT bumped
        // (admit returned Err before the admitted-counter increment).
        let stats = s.stats();
        assert_eq!(stats.rejected_429_total, 1,
            "SlotBudgetExceeded bumps rejected_429_total (maps to 429 upstream)");
        assert_eq!(stats.admitted_total, 0);
        assert_eq!(stats.in_flight_slots, 0,
            "no physical slot allocated for over-budget admit");
    }

    #[test]
    fn fifo_per_slot_budget_zero_means_unbounded() {
        // Default constructor (no budget arg) ⇒ enforcement disabled.
        // Even an astronomical kv_bytes_needed admits cleanly.
        let mut s = FifoSchedulerAdapter::new(4);
        assert_eq!(s.per_slot_kv_budget_bytes(), 0,
            "new() defaults to per_slot_kv_budget_bytes = 0 (unbounded)");
        let r = s
            .admit(req_with_kv(1, 1, u64::MAX))
            .expect("zero-budget scheduler must accept any kv_bytes_needed");
        assert!(r.handle.is_some());
        assert_eq!(s.stats().rejected_429_total, 0);
        // Explicit `new_with_kv_budget(.., 0)` equivalent.
        let mut s2 = FifoSchedulerAdapter::new_with_kv_budget(4, 0);
        assert_eq!(s2.per_slot_kv_budget_bytes(), 0);
        s2.admit(req_with_kv(1, 1, u64::MAX))
            .expect("explicit 0-budget also unbounded");
    }

    #[test]
    fn inflight_admit_above_per_slot_budget_errors() {
        // Budget 4 MiB per slot; request asks for 5 MiB ⇒ rejected.
        let mut s = InflightBatchedScheduler::new_with_kv_budget(8, 4, 4 * 1024 * 1024);
        let needed = 5 * 1024 * 1024;
        match s.admit(req_with_kv(2048, 128, needed)) {
            Err(AdmitError::SlotBudgetExceeded { needed_bytes, budget_bytes }) => {
                assert_eq!(needed_bytes, needed);
                assert_eq!(budget_bytes, 4 * 1024 * 1024);
            }
            other => panic!("expected SlotBudgetExceeded, got {:?}", other),
        }
        assert_eq!(s.stats().rejected_429_total, 1);
        assert_eq!(s.stats().admitted_total, 0);
        assert_eq!(s.stats().in_flight_slots, 0,
            "over-budget admit does not allocate a physical slot");
    }

    #[test]
    fn admit_error_slot_budget_exceeded_display_names_needed_and_budget() {
        let err = AdmitError::SlotBudgetExceeded {
            needed_bytes: 12_345_678,
            budget_bytes: 4_096_000,
        };
        let dbg = format!("{:?}", err);
        assert!(dbg.contains("SlotBudgetExceeded"));
        assert!(dbg.contains("needed_bytes"));
        assert!(dbg.contains("budget_bytes"));
        assert!(dbg.contains("12345678"));
        assert!(dbg.contains("4096000"));

        let disp = format!("{}", err);
        assert!(disp.contains("needed_bytes=12345678"),
            "Display names needed_bytes verbatim: {}", disp);
        assert!(disp.contains("budget_bytes=4096000"),
            "Display names budget_bytes verbatim: {}", disp);
        // Operator-actionable message MUST cite ADR-040 §3.5 + name the
        // remediation paths.
        assert!(disp.contains("ADR-040"),
            "Display cites ADR-040 §3.5: {}", disp);
        assert!(disp.contains("max_tokens") || disp.contains("prompt"),
            "Display names the actionable remediation: {}", disp);
    }

    #[test]
    fn inflight_per_slot_budget_independent_per_slot() {
        // ADR-040 §3.5: per-slot budget is independent per slot, NOT
        // aggregate.  Build a scheduler with max_slots=4, budget=1 MiB
        // per slot; admit 4 requests each asking for exactly 1 MiB.
        // All 4 must admit (each at the per-slot budget; the budget
        // does not sum across slots).
        let per_slot = 1024 * 1024;
        let mut s = InflightBatchedScheduler::new_with_kv_budget(8, 4, per_slot);
        let mut admitted = Vec::new();
        for i in 0..4 {
            let r = s
                .admit(req_with_kv(64, 16, per_slot))
                .unwrap_or_else(|e| panic!("slot {} at-budget admit must succeed; got {:?}", i, e));
            assert!(r.handle.is_some(),
                "slot {}: at-budget admit lands in_flight", i);
            admitted.push(r);
        }
        let stats = s.stats();
        assert_eq!(stats.admitted_total, 4,
            "all 4 at-budget admits counted; per-slot budget does not sum");
        assert_eq!(stats.in_flight_slots, 4,
            "all 4 physical slots occupied (max_slots=4)");
        assert_eq!(stats.rejected_429_total, 0,
            "ZERO 429s — each request fits its own per-slot budget");

        // The 5th admit AT budget queues (in_flight cap reached, queue
        // accepts) — still NOT a budget rejection because the request
        // itself fits the per-slot budget.
        let r5 = s
            .admit(req_with_kv(64, 16, per_slot))
            .expect("5th at-budget admit queues (not a budget violation)");
        assert!(r5.handle.is_none(),
            "5th admit is queued (in_flight at max_slots=4)");
        assert_eq!(s.stats().rejected_429_total, 0,
            "queueing is not a budget rejection");

        // But an OVER-budget admit IS rejected even when queue has room.
        match s.admit(req_with_kv(64, 16, per_slot + 1)) {
            Err(AdmitError::SlotBudgetExceeded { needed_bytes, budget_bytes }) => {
                assert_eq!(needed_bytes, per_slot + 1);
                assert_eq!(budget_bytes, per_slot);
            }
            other => panic!("expected SlotBudgetExceeded, got {:?}", other),
        }
    }

    #[test]
    fn admit_request_default_kv_bytes_needed_is_zero() {
        // Back-compat surface: AdmitRequest::default() leaves
        // kv_bytes_needed at 0 so enforcement is opt-in via the field.
        let r = AdmitRequest::default();
        assert_eq!(r.prompt_tokens, 0);
        assert_eq!(r.max_tokens, 0);
        assert_eq!(r.kv_bytes_needed, 0);
    }
}
