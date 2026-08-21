# ADR-040 Phase C iter-2 — Engine slot-aware wiring dossier

- **Date**: 2026-05-24 (research authored 2026-05-23 same-session)
- **Author**: research agent (grounding pass for Phase C iter-2 implementation)
- **Status**: pre-implementation; no code in this dossier
- **Mantra**: "Never guess. Multi-week structural work is in scope. Chesterton's fence: understand current fully before changing it."

## Inputs read

1. `/opt/hf2q/docs/adr/ADR-040-continuous-batching-reopen.md` §3.2, §3.3, §3.6, §6 (C1/C2/C3/C4 rows), §6.1.1–§6.1.6
2. `/opt/hf2q/docs/research/adr040-kv-cache-lift-dossier-2026-05-23.md` (A2/A3 dossier — shape + style reference)
3. `/opt/hf2q/src/serve/api/engine.rs` (12,791 LOC; key spans: 1–32 docstring; 460–541 Engine + LoadedArch + EngineMode; 555–576 EngineSpawnError; 847–939 EngineInner; 1224–1282 GemmaLoadedModel; 2410–2683 spawn + spawn_with_mode + mode accessor; 2832–3245 the 11 `tx.try_send` handler sites; 3263 shutdown; 3346–3700 worker_run + generate_once)
4. `/opt/hf2q/src/serve/scheduler.rs` (2,045 LOC; iter-2.5 surface — pages 1–1255 audited; concrete types: `Scheduler` trait, `FifoSchedulerAdapter`, `InflightBatchedScheduler`, `SlotHandle`, `RequestId`, `SchedulerStep`, `SchedulerPolicy`, `AdmitRequest`, `RequestSlot`, `AdmitError`, `StepError`, `SchedulerStats`)
5. `/opt/hf2q/src/serve/multi_seq_kv.rs` (full file; `MultiSeqKvCache` trait + `SlotId`/`SeqId` newtypes + `MultiSeqError` + `NoopMultiSeqKvCache` fixture)
6. `/opt/hf2q/src/inference/models/qwen35/forward_gpu.rs:1585-1617` (post-B4a `forward_gpu` signature with `slot_id: SlotId`); `:1624-1646` (`forward_gpu_last_logits` — still hard-coded `SlotId(0)` — B4b deferral)
7. `/opt/hf2q/src/serve/mod.rs:3679` (single production `Engine::spawn` callsite in cmd_serve)
8. `/opt/hf2q/src/serve/api/engine_qwen35.rs:1020-1049, 1232-1263` (`alloc_kv_cache_for_request` + `generate_qwen35_once`)
9. `/opt/hf2q/tests/continuous_batching_throughput.rs` (D iter-1 scaffold)

---

## §1 Executive summary

The C2 wiring decision narrows to **Shape A (scheduler-pulls-from-mpsc, worker-thread-owns-scheduler)** for two independent reasons. First, the existing `mpsc::channel` is load-bearing across 12 callsites that depend on tokio-aware `try_send`/`blocking_send` semantics (synchronous KV snapshot/restore + warmup + shutdown), and replacing it would multiply the change surface 10× without functional gain. Second, the `FifoSerial` byte-equivalence pledge (ADR-040 §3.6 + AC-3) is exactly what Shape A makes trivial: under `EngineMode::SerialFifo`, the worker loop's existing `while let Some(req) = rx.blocking_recv()` body is wrapped by a one-call `admit→step→drive→release` cycle whose observable outputs match pre-C2 byte-for-byte.

The **load-bearing falsifying test** is `engine_serial_fifo_byte_equivalent_to_pre_phase_c` (ADR-040 §5 AC-3): construct two engines (one via 3-arg `spawn`, one via the new `spawn_with_mode(.., EngineMode::SerialFifo)`), drive the same prompt through both via `engine.generate(...)`, assert byte-equality on the returned `GenerationResult.completion_tokens` and `usage` counters. A divergence here aborts C2.

The **top risk** is **R1 — KV cache lifecycle mismatch between Qwen35 and Gemma**: Qwen35's `HybridKvCache` is allocated per-request inside `generate_qwen35_once` (`engine_qwen35.rs:1263`) while Gemma's KV state lives inside `MlxModelWeights` (loaded once at `Engine::spawn` time). C2's `SlotAware` path requires both arches to expose a long-lived multi-seq KV cache. For Qwen35 the C2 work LIFTS allocation out of `generate_qwen35_once` into `EngineInner`; for Gemma it requires the Phase A3 lift to land first (currently NOT shipped — see `adr040-kv-cache-lift-dossier-2026-05-23.md` §2.2). **Mitigation**: ship C2 with Qwen35-only SlotAware support, gate Gemma SlotAware behind a Phase A3 follow-up iter — this is consistent with the §6 sequencing where B4c (Gemma forward path slot threading) is gated on A3.

The **iter-2 step-1 starting point** is: write `engine_serial_fifo_byte_equivalent_to_pre_phase_c` as a failing test FIRST (against the iter-1.5 codebase where `spawn_with_mode(SerialFifo)` already works — the test should pass at HEAD; this verifies the harness is correct), THEN refactor `worker_run` to thread a `Box<dyn Scheduler>` through the existing dispatch arms without changing the FifoSerial path's observable behaviour. The actual `SlotAware` runtime lands as iter-2b after iter-2a's byte-equivalence is locked.

**Confidence: medium**. The Shape A recommendation is high-confidence (Chesterton-aligned, smallest blast radius, preserves all 11 handler callsites unchanged at the signature level). The Qwen35-vs-Gemma KV lifecycle divergence is a real architectural wrinkle that iter-2 must address explicitly — this dossier proposes a Qwen35-only iter-2a scope and a Gemma-A3-gated iter-2b follow-up.

---

## §2 Per-question findings

### §2.1 Q1: Two C2 wiring shapes — which is right?

Three shapes were considered; Shape A is recommended. Detailed analysis below.

#### Shape A — Scheduler-pulls-from-mpsc (RECOMMENDED)

The existing `mpsc::Sender<Request>` + `mpsc::Receiver<Request>` stay exactly as-is. `worker_run` is extended to own a `Box<dyn Scheduler>` constructed at function entry from a `SchedulerPolicy` parameter (passed in via the `spawn_with_mode` path). The body wraps each `req` pulled from `rx.blocking_recv()` in a single `admit→step→drive→release` cycle:

```text
while let Some(req) = rx.blocking_recv() {
    match req {
        Generate { prompt_tokens, params, reply } => {
            let admit_req = AdmitRequest {
                prompt_tokens: prompt_tokens.len() as u32,
                max_tokens: params.max_tokens as u32,
            };
            let slot = scheduler.admit(admit_req)?; // returns Ok unconditionally
                                                    // under FifoSerial when in_flight==None
            let handle = slot.handle.expect("FifoSerial admit always returns Some(handle)");
            // existing generate_once logic, threaded with handle.slot_id
            let result = generate_once_for_slot(&mut loaded, &prompt_tokens, &params,
                                                handle.slot_id, registration.as_ref());
            scheduler.release(handle);
            let _ = reply.send(result);
        }
        // ... other 10 arms similarly wrapped
    }
}
```

**(a) Preserves Decision #2 byte-equivalence under `FifoSerial`**: yes. At `max_slots=1`, every `admit` returns `Ok(RequestSlot { handle: Some(SlotHandle { slot_id: SlotId(0), generation: G }) })` because the prior request's `release` bumped the generation (`scheduler.rs:587-596`). The `step()` call (if invoked from this synchronous shape) returns `Prefill { handle, n_tokens: prompt_tokens.len() }` then `Decode { handles: [handle] }` after `advance_after_prefill` — but in Shape A's tight per-request cycle we don't actually consult `step()`; we drive prefill+decode directly via the existing `generate_once` body and call `advance_after_prefill`/`advance_after_decode` post-hoc only as bookkeeping. This is byte-equivalent to today's `worker_run` because the `step()` discriminant is not on the data path.

**(b) 11 handler callsites change**: ZERO signature changes. Handlers still call `self.inner.tx.try_send(req)`. The scheduler lives entirely behind the channel, owned by the worker thread.

**(c) `worker_run` becomes**: same skeleton, with three additions: (1) accepts an extra `mode: EngineMode` parameter; (2) constructs `Box<dyn Scheduler>` at top-of-function based on `mode`; (3) wraps each request arm in `admit→...→release`.

**(d) Backward-compat risk for 3-arg `Engine::spawn`**: ZERO. The 3-arg constructor calls `spawn_with_mode(.., EngineMode::SerialFifo)` internally (already wired at `engine.rs:2644` in the iter-1.5 fix); `spawn_with_mode` constructs a `FifoSchedulerAdapter` and passes it to `worker_run`. Existing callers (cmd_serve at `mod.rs:3679`, `multi_model::DefaultModelLoader`) see no signature change.

**(e) Memory model**: scheduler lives on the worker thread, accessed via `&mut`. ZERO mutex/RwLock contention. The scheduler does NOT need `Send + Sync` for shared access (the trait is `: Send` per `scheduler.rs:338` for thread-handoff at construction; that's sufficient).

#### Shape B — Worker-replaced-by-scheduler-driven-loop (REJECTED)

Same mpsc channel for inbox; worker loop replaced by `loop { req = rx.try_recv(); if Some(req) { scheduler.admit(...); } while !idle { let step = scheduler.step(); dispatch(step); } }`.

**Why rejected**:
- Adds a busy-wait / select loop with no observable benefit at `max_slots=1` (where step() always returns Prefill-then-Decode-then-Idle and there's never anything to do during Idle except wait for the next mpsc recv).
- Doubles the complexity of the byte-equivalence proof: now the worker has TWO control flows (one for FifoSerial, one for InflightBatched). Shape A keeps one control flow.
- Doesn't gain anything for InflightBatched either, because admission is still synchronous per request (tokio handlers call `tx.try_send` → handler-side mutex around scheduler is NOT in the data path).

#### Shape C — mpsc replaced by direct scheduler.admit() in handlers (REJECTED)

Handlers call `engine.inner.scheduler.lock().admit(req)` directly; handlers block on a oneshot tied to slot completion; worker thread owns nothing scheduler-related and just executes `step()` results.

**Why rejected**:
- Requires `Mutex<Box<dyn Scheduler>>` on `EngineInner` — 11 handler callsites all contend on the lock per-admit. Under tokio, this means `tokio::sync::Mutex` (async-aware); under `std::sync::Mutex`, `block_in_place` warnings and starvation risk.
- Breaks the existing `try_send` non-blocking semantics. Today's `try_send` returns `TrySendError::Full` instantly when queue is full; with Shape C, admit + lock acquisition becomes the path. Async-await on the lock would change handler latency characteristics.
- Multiplies test surface: now we need axum-handler-side integration tests in addition to scheduler tests.
- Breaks the 12 synchronous KV snapshot/restore callsites (`request_kv_snapshot`, `request_kv_restore`, `tq_packed_v2_*`, `request_prompt_cache_*`) which assume a worker thread services them.

#### Recommendation

**Shape A**. Rationale: (1) preserves the entire ADR-005 Decision #2 control flow shape, which is the byte-equivalence target; (2) requires zero changes to the 11 handler callsites (Chesterton's fence — the channel pattern works for every kind of request, including the rare control-plane sync requests); (3) the scheduler is constructed once at worker-thread entry and accessed via `&mut`, eliminating all lock-contention questions; (4) iter-2's only `&mut` discipline question becomes "does `worker_run` drive prefill+decode against `&mut Box<dyn Scheduler>` correctly?" — a far smaller proof than Shape B's two-control-flow proof or Shape C's mutex-discipline proof.

Citations: ADR-040 §3.2 "FifoSchedulerAdapter wraps the existing mpsc-channel path with byte-equivalent behaviour"; §3.6 "every byte of Engine behaviour is bit-equivalent to pre-ADR-040"; `engine.rs:1-32` worker-thread + mpsc rationale ("forward passes are ~10-100ms of pure compute — holding a tokio mutex across that would starve keep-alive layers"); `scheduler.rs:9` "the pattern mirrors `serve::multi_model` (W74 iter-206): a synthetic-fixture-tested data structure that later iters glue into the live serve path".

---

### §2.2 Q2: EngineInner field additions

Existing `EngineInner` fields (`engine.rs:847-939`):
- `tx: mpsc::Sender<Request>`
- `worker_handle: Mutex<Option<JoinHandle<()>>>`
- `info: Arc<LoadInfo>`
- `arch: LoadedArch`
- `model_id: String`, `context_length: Option<usize>`, `quant_type: Option<String>`, `hidden_size: usize`, `vocab_size: usize`, `eos_token_ids: Vec<u32>`
- `tokenizer: Arc<Tokenizer>`, `chat_template: Arc<String>`
- `registration: Option<ModelRegistration>`
- `token_bytes: OnceLock<Arc<Vec<Vec<u8>>>>`
- `kv_spill_descriptor: Option<KvSpillDescriptor>`
- `tq_packed_descriptor: Option<TqPackedSpillDescriptor>`
- `mode: EngineMode` (confirmed present at `engine.rs:938`, added by iter-1.5 F1 fix)

**For C2 — proposed additions**:

| Field | Type | Purpose | Owner |
|---|---|---|---|
| `max_slots` | `u32` | The configured slot cap (1 for SerialFifo, N for SlotAware). Read by `/metrics`, `/v1/models` extensions, scheduler stats accessor. | EngineInner |
| `scheduler_stats_snapshot` | `Arc<std::sync::Mutex<SchedulerStats>>` | Worker-thread updates after each release; handler-side reads for `/metrics`. The actual `Scheduler` lives on the worker thread (Shape A). | EngineInner |

**NOT added to EngineInner**: the `Box<dyn Scheduler>` itself. Per Shape A, the scheduler lives on the worker thread (passed to `worker_run` at spawn). EngineInner holds a snapshot copy of `SchedulerStats` (cheap-to-clone struct of 6 u64+u32 fields per `scheduler.rs:315-323`), updated periodically by the worker via a `parking_lot::Mutex` or `std::sync::Mutex` write.

**NOT added (per-arch KV cache)**: this is the big design decision — see §2.4 below. The recommendation is to push the multi-seq KV cache LIVE-OWNERSHIP onto the per-arch `LoadedModel` variant (`Qwen35LoadedModel.persistent_kv_cache: HybridKvCache` for the SlotAware case), NOT onto `EngineInner`. This avoids the trait-object boxing problem (R3).

**Arc/Mutex pattern**: existing fields use a mix:
- `Arc<LoadInfo>`, `Arc<Tokenizer>`, `Arc<String>` — read-only, shared with handlers
- `Mutex<Option<JoinHandle<()>>>` — write-once, taken at shutdown
- `OnceLock<Arc<Vec<Vec<u8>>>>` — lazy init, then read-only
- Most metadata fields — plain owned (`String`, `usize`, etc.)

The proposed `scheduler_stats_snapshot: Arc<Mutex<SchedulerStats>>` matches the `worker_handle` pattern (Mutex used because mutation is rare and lock acquisition is cheap when uncontended). For `/metrics` reads (every Prometheus scrape) the lock is briefly held to clone the 32-byte struct.

**Re-verified**: `mode: EngineMode` IS already on `EngineInner` (`engine.rs:938`, populated at `:2594` from 3-arg spawn and at `:2650` from `spawn_with_mode`). Iter-2 does not need to add it.

---

### §2.3 Q3: Scheduler lifetime + ownership

**Where constructed**: at `worker_run` entry, inside the worker thread. The construction is:

```text
let mut scheduler: Box<dyn Scheduler> = match mode {
    EngineMode::SerialFifo => Box::new(FifoSchedulerAdapter::new(queue_capacity as u32)),
    EngineMode::SlotAware { max_slots } => {
        Box::new(InflightBatchedScheduler::new(queue_capacity as u32, max_slots))
    }
};
```

This is the simplest ownership model: scheduler is `&mut`-accessed on a single thread, never shared. No locking discipline at the scheduler level (the iter-2.5 SlotHandle generation counter handles only the post-release-stale-callback case, which is moot under Shape A's synchronous drive).

**Lock granularity**: N/A — the scheduler is not shared. Only the `scheduler_stats_snapshot: Arc<Mutex<SchedulerStats>>` is shared with handlers, and that's a per-scrape lock (sub-microsecond contention).

**Lifetime question**: scheduler lives exactly as long as `worker_run` runs (from `Engine::spawn`/`spawn_with_mode` until `Shutdown` request). Same lifetime as `loaded: LoadedModel`. Drops cleanly when the worker thread exits.

**llama.cpp `-cb` reference**: llama.cpp's `llama_decode` is single-threaded — the `llama_context` owns the equivalent scheduler state and is mutated by a single caller thread. Multi-threaded callers wrap their own external mutex. The Shape A recommendation mirrors this: hf2q's worker thread is the single mutator, equivalent to llama.cpp's user thread driving `llama_decode`. This is the closest behavioural mirror to the comparator ADR-040 §3.3 names.

---

### §2.4 Q4: MultiSeqKvCache ownership + lifetime

This is the single highest-leverage architectural question for C2. Three sub-questions:

#### 2.4.1 Where does the multi-seq KV cache live?

**Today** (pre-ADR-040 + iter-2a A2):
- **Gemma**: KV cache lives inside `MlxModelWeights` (specifically the per-layer `kv_caches: Vec<MlxKvCache | HbKvBuffers | DenseKvBuffers | HybridKvBuffers>` per `adr040-kv-cache-lift-dossier-2026-05-23.md` §2.2). Allocated once at `MlxModelWeights::load` time, lives for the lifetime of the engine.
- **Qwen35**: `HybridKvCache` is allocated **per-request** inside `engine_qwen35::alloc_kv_cache_for_request` (`engine_qwen35.rs:1027-1049`) and dropped at end of `generate_qwen35_once`. The cache lives ONLY for the request's lifetime.

The two arches have **opposite** KV lifecycles today.

**For C2 SlotAware path**: the cache MUST persist across multiple requests (slot 0's K/V outlives request A so request B can run concurrently in slot 1). Two implementation paths exist:

| Option | Description | Pros | Cons |
|---|---|---|---|
| **A** | Push a long-lived multi-seq KV cache onto each `LoadedModel` variant — `Qwen35LoadedModel.persistent_kv_cache: Option<HybridKvCache>`, allocated with `n_seqs=max_slots` at `LoadedModel::load` time (or first-request-time lazy). Replace `alloc_kv_cache_for_request` with a slot-acquire path. | Single source of truth per arch; matches Gemma's existing shape; no trait-object boxing. | Qwen35 needs a refactor of `generate_qwen35_once` to STOP allocating per-request when in SlotAware mode. |
| **B** | Add `multi_seq_kv: Arc<Mutex<Box<dyn MultiSeqKvCache>>>` to `EngineInner`. | Single field; trait-object polymorphic across arches. | Loses optimizer-friendly type info on the hot path; introduces a runtime cross-thread mutex when worker accesses; conflicts with existing per-arch shape. |

**Recommendation**: Option A. The Qwen35 refactor is bounded (`alloc_kv_cache_for_request` is called from exactly one site at `engine_qwen35.rs:1263`; the call becomes a slot-acquire from `qwen.persistent_kv_cache`). The Gemma path requires no change for iter-2a (SerialFifo preserves existing behaviour — Gemma's MlxModelWeights-owned KV cache continues to work). For iter-2b SlotAware, Gemma needs the Phase A3 lift first.

#### 2.4.2 Per-arch typed cache vs `Box<dyn MultiSeqKvCache>`?

**Per-arch typed**. The worker thread dispatches on `&mut loaded: LoadedModel` (an enum at `engine.rs:1284`); the per-arch arm in the match knows the concrete type. The Qwen35 arm calls `qwen.persistent_kv_cache.as_mut().expect(...).append_for_seq(slot, n)` directly on `HybridKvCache`. The trait `MultiSeqKvCache` is used for cross-cutting code (the eventual `serve::api::schema` HTTP 501 mapping for `MultiSeqError::CapabilityUnsupported`, per `multi_seq_kv.rs:282-306`) but NOT on the per-request forward path.

This preserves type info for the optimizer (LLVM can devirtualize `HybridKvCache::append_for_seq` calls), avoids the `dyn` overhead, and matches the existing per-arch dispatch pattern in `worker_run` (each Request arm matches on `&mut loaded` per `engine.rs:3380-3406`).

#### 2.4.3 Lifetime answer

| Scope | Today (Qwen35) | Today (Gemma) | C2 SlotAware (Qwen35) | C2 SlotAware (Gemma) |
|---|---|---|---|---|
| KV cache allocator | per-request | per-load | per-load (lift to `Qwen35LoadedModel`) | per-load (already there) |
| KV cache owner | `generate_qwen35_once` stack | `MlxModelWeights` | `Qwen35LoadedModel.persistent_kv_cache` | `MlxModelWeights` (unchanged) |
| Drop trigger | end of `generate_qwen35_once` | engine shutdown | engine shutdown | engine shutdown |
| `n_seqs` | 1 | 1 (no `n_seqs` field today — §2.2 of A2 dossier) | `max_slots` | `max_slots` (gated on Phase A3 lift) |

**Iter-2a scope**: Qwen35 only. Gemma SlotAware path is deferred to iter-2b (or to a Phase C iter-2c) per Phase A3 readiness.

---

### §2.5 Q5: Hot path preservation (Decision #2 byte-equivalence)

**Definition of byte-equivalence under FifoSerial**:
1. Every existing tests pass unmodified (95+ engine tests at iter-1.5; the existing 11 handler callsite tests).
2. `engine.generate(prompt_tokens, params)` produces a `GenerationResult` whose:
   - `completion_tokens: Vec<u32>` is byte-equal to pre-C2
   - `usage: GenerationUsage` counters (prompt_tokens, completion_tokens, total_tokens, cached_tokens) are byte-equal
   - `finish_reason: FinishReason` is byte-equal
3. `engine.generate_stream(...)` produces the same sequence of `GenerationEvent`s in the same order with the same token bytes.
4. Worker thread observability (tracing spans, metrics) is structurally unchanged (per-event names and field shapes preserve).

**Canonical proof test**: `engine_serial_fifo_byte_equivalent_to_pre_phase_c` — proposed shape:

```text
#[test]
fn engine_serial_fifo_byte_equivalent_to_pre_phase_c() {
    // SETUP: synthetic small Gemma fixture (uses make_synthetic_kv_engine_for_test pattern
    // at engine.rs:602 — or a tiny GGUF if available under HF2Q_BYTE_EQUIV_E2E env gate).
    let loaded_a = LoadedModel::Gemma(synthetic_gemma_for_test());
    let loaded_b = LoadedModel::Gemma(synthetic_gemma_for_test());  // identical seed
    let engine_a = Engine::spawn(loaded_a, 4, None);                // pre-C2 path
    let engine_b = Engine::spawn_with_mode(loaded_b, 4, None,
        EngineMode::SerialFifo).expect("SerialFifo always succeeds");
    let prompt = vec![1u32, 2, 3, 4, 5];
    let params = SamplingParams { temperature: 0.0, max_tokens: 16, ..Default::default() };
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let result_a = rt.block_on(engine_a.generate(prompt.clone(), params.clone())).unwrap();
    let result_b = rt.block_on(engine_b.generate(prompt, params)).unwrap();
    assert_eq!(result_a.completion_tokens, result_b.completion_tokens,
               "FifoSerial must produce byte-identical completion_tokens to pre-C2");
    assert_eq!(result_a.usage, result_b.usage,
               "FifoSerial must produce byte-identical usage counters");
    assert_eq!(result_a.finish_reason, result_b.finish_reason,
               "FifoSerial must produce byte-identical finish_reason");
}
```

**Catches**: any worker-loop wrapper that mutates state (e.g. an off-by-one in `advance_after_decode`, a token dropped because `step()` returned `Idle` prematurely, a sampler RNG seed shift from inserting a tokio-runtime layer between `try_send` and the worker).

**Doesn't catch**: timing/latency drift (test is correctness only); SSE keepalive ordering changes (separate test for that).

The test is feasible TODAY (the `make_synthetic_kv_engine_for_test` fixture already exists at `engine.rs:602` and is `pub(crate)`; a sibling test fixture for Gemma compute path is achievable). For a fully-real test, an env-gated `HF2Q_BYTE_EQUIV_E2E=1` variant with a tiny real GGUF (Qwen3.5-0.6B Q4_K_M is ~400 MB) is the iter-2 follow-up.

---

### §2.6 Q6: Backward-compat for 3-arg `Engine::spawn`

**The contract** (`engine.rs:2412-2417`):

```rust
pub fn spawn(
    loaded: LoadedModel,
    queue_capacity: usize,
    kv_cache_budget_bytes: Option<u64>,
) -> Self { ... }
```

**Production callers**:
- `src/serve/mod.rs:3679` (`cmd_serve`'s `load_engine` helper)
- `src/serve/multi_model.rs:103-119` (DefaultModelLoader — per ADR-005 Phase 4) — UNVERIFIED for exact callsite, needs grep at iter-2 start. The grep at this dossier authoring time found ONLY the cmd_serve callsite in `serve/mod.rs`; multi_model.rs only references it in doc comments. **UNVERIFIED — needs probe before implementation** to confirm there are no live `Engine::spawn` callers I missed.

**Regression pin in iter-1.5**: `engine_spawn_3_arg_signature_compile_pin` (renamed from iter-1's `engine_spawn_signature_unchanged_at_phase_c_iter_1` per ADR-040 §6.1.1 F4 fix). This test compile-fails if a future iter modifies the 3-arg signature.

**How 3-arg spawn gets its scheduler in C2**: the existing `spawn` body already calls into the `worker_run` thread (`engine.rs:2569`). In C2, we EXTEND `worker_run`'s signature with a `mode: EngineMode` parameter; the 3-arg `spawn` body hard-codes `EngineMode::SerialFifo` at its call site (it can't take a mode parameter without violating the signature pin). The flow becomes:

```text
Engine::spawn(loaded, qcap, kvbudget)  // signature unchanged
  -> [body]
     let mode = EngineMode::SerialFifo;
     ...
     spawn_worker_thread(... move || worker_run(loaded, rx, registration, queue_capacity as u32, mode));
     EngineInner { ..., mode, ... }

Engine::spawn_with_mode(loaded, qcap, kvbudget, mode)  // signature added in iter-1.5
  -> [body]
     ... constructs scheduler from mode ...
     spawn_worker_thread(... move || worker_run(loaded, rx, registration, queue_capacity as u32, mode));
     EngineInner { ..., mode, ... }
```

Both paths land in the same `worker_run` (signature extended). The 3-arg path always passes `SerialFifo`; the new path passes the caller's choice. This is the exact pattern that locks the 3-arg backward-compat contract while enabling the new path.

**Internal `spawn` change**: the body must drop the iter-1.5 fix's `Arc::get_mut` dance at `engine.rs:2649-2660` (or extend it: when 3-arg `spawn` is called, mode is already `SerialFifo` from the EngineInner construction at `:2594` — no mutation needed; iter-1.5 left the `get_mut` block as a no-op for the SerialFifo case). At C2 iter-2, the cleanest move is to factor the spawn body into a private helper that both entry points call with their respective mode value.

---

### §2.7 Q7: Handler-side admit pattern

Under Shape A (recommended), **handlers do NOT call admit**. Handlers continue to call `self.inner.tx.try_send(req)` (or `blocking_send` for control-plane sync requests). The scheduler's `admit` is called inside `worker_run` for each pulled `Request`.

This is the load-bearing simplification: the 11 handler callsites at `engine.rs:2832, 2859, 2895, 2949, 2994, 3031, 3077, 3155, 3174, 3235` are UNCHANGED. The `429 + Retry-After` path also unchanged — the mpsc channel's `TrySendError::Full` already maps to `anyhow::bail!("queue_full")` at every site, which the chat/embed/etc. handlers map to HTTP 429.

**What happens to `RequestSlot`?** It's an internal-to-worker concept under Shape A. The handler doesn't see it; the handler waits on its `oneshot::Receiver` per existing pattern. The slot bookkeeping (which slot ran which request) is worker-internal.

**What happens to `cancel_queued`?** Not yet needed under FifoSerial (the mpsc channel itself enforces queue ordering and the worker drains it FIFO). Becomes relevant under SlotAware when a tokio task is cancelled but the request has already been forwarded to the worker. Iter-2's recommendation: the C2 first cut DOES NOT wire `cancel_queued`. Client disconnect / cancellation is already handled by the existing `cancellation_counter: Arc<AtomicU64>` pattern (`engine.rs:3097-3115`). If a request is queued in the mpsc channel and the client disconnects, the channel's `oneshot::Receiver` drops; the worker's `let _ = reply.send(result);` silently fails; no slot-side cleanup needed because no slot was allocated for it yet.

**Wait — what about admission-during-decode for InflightBatched?** Under Shape A's tight per-request cycle, that's the limitation: SerialFifo runs requests serially, but InflightBatched (under Shape A) ALSO runs requests serially because the worker only processes one mpsc message at a time. **Therefore Shape A only works for FifoSerial**.

This is a real limitation. The fix is iter-2b/c: under `EngineMode::SlotAware`, the worker_run body becomes a `loop { select! { req = rx.recv() => admit; _ = compute_idle => step+dispatch } }` shape (the Shape B body — but only ENGAGED under SlotAware). Under SerialFifo we keep the simple Shape A body. This way:
- iter-2a ships Shape A + the FifoSerial byte-equivalence proof (the SlotAware path returns `EngineSpawnError::ModeNotYetWired` per iter-1.5).
- iter-2b ships the Shape B body INSIDE the `worker_run` `match mode { SlotAware { .. } => ..., SerialFifo => ... }`. SerialFifo path unchanged.

**Iter-2a recommendation revised**: ship ONLY the FifoSerial wrapping (Shape A inside the `worker_run` for `SerialFifo`). The `SlotAware` path lands at iter-2b after iter-2a's byte-equivalence is locked.

---

### §2.8 Q8: Forward dispatch — how does worker translate SchedulerStep → forward_gpu call?

**Under FifoSerial (iter-2a)**: the worker does NOT consult `step()` on the data path. It calls the existing `generate_once` body directly, then calls `advance_after_prefill` + `advance_after_decode` as post-hoc bookkeeping (mostly to keep `SchedulerStats` accurate for `/metrics`).

**Under InflightBatched (iter-2b/c)**: the worker dispatches `SchedulerStep` variants:

| Variant | Driver action | hf2q forward call |
|---|---|---|
| `Idle` | Park the worker on a notify/condvar or fall through to `rx.recv()`. | — |
| `Prefill { handle, n_tokens }` | Run prefill for `handle.slot_id` for `n_tokens` tokens; report back via `advance_after_prefill(handle, n_tokens)`. | `model.forward_gpu_prefill(prompt_tokens[off..off+n_tokens], positions, &mut kv_cache, handle.slot_id)` — for Qwen35, this is the existing `forward_gpu` (signature post-B4a accepts `slot_id: SlotId`). |
| `Decode { handles }` | One forward per handle in `handles` (per Phase B6 note: full batched mixed dispatch in one forward is Phase B6 scope). Report via `advance_after_decode(handle)`. | `model.forward_gpu_last_logits(decode_token, decode_position, &mut kv_cache, handle.slot_id)` — UNVERIFIED slot_id signature; today's `forward_gpu_last_logits` hard-codes `SlotId(0)` per `forward_gpu.rs:1644` (B4b deferral). **iter-2b is gated on B4b shipping first**. |
| `Mixed { prefill, n_prefill_tokens, decode_handles }` | Phase B6 scope — per ADR-040 §6 Phase B6, the Mixed variant requires kernel-level batching across slots. NOT in C2 scope. Iter-2 worker dispatches Mixed as a SEQUENCE: first the Prefill, then each Decode handle, sharing nothing. | — |

**For the FIFO path (max_slots=1)**, the question collapses: `step()` never returns `Mixed`; `Decode { handles }` always has `len==1`; the dispatch is identical to today's `worker_run` body.

**llama.cpp `-cb` reference**: llama.cpp's `llama_batch` API DOES batch prefill+decode across slots in a single `llama_decode` call — this is the kernel-level continuous-batching primitive. hf2q's `forward_gpu` (post-B4a) takes a single `tokens: &[u32]` slice + a single `slot_id: SlotId`; the analogue would be to extend it to `tokens: &[u32], slot_ids: &[SlotId]` with per-token-slot routing through the KV cache. **This is explicitly Phase B6 scope per ADR-040 §6 ("Mixed prefill+decode SchedulerStep::Mixed handling")**. C2 iter-2 does NOT need it.

**The clean punt for iter-2b**: when `step()` returns `Mixed { prefill, n_prefill_tokens, decode_handles }`, the worker breaks it into N+1 separate `forward_gpu*` calls. This is observably worse than llama.cpp's batched dispatch (one encoder per call vs one encoder for all), but it's CORRECT and ships the C2 scaffolding. Phase B6 then lands the real kernel-level batching.

---

### §2.9 Q9: advance_after_prefill / advance_after_decode driver discipline

**Who calls them**: the worker thread, AFTER each forward_gpu returns. This is the only correct timing — calling before would risk a state-machine advance for an as-yet-uncomputed token; calling at a different point would risk stale-handle behaviour.

**Race risk under Shape A**: NONE. The scheduler is `&mut`-owned on the worker thread; there are no concurrent accessors. `advance_after_*` is synchronous; the iter-2.5 SlotHandle generation counter is functionally inert under FifoSerial (the handle was just produced by `admit`; nothing else can have released it).

**Race risk under Shape C (rejected)**: substantial. If the scheduler is shared `Arc<Mutex<...>>`, the worker thread must take the mutex briefly to step()+get-work, RELEASE during the 10-100ms compute, then re-take to advance. The lock-release-during-compute pattern is exactly what the iter-2.5 SlotHandle generation counter (`scheduler.rs:127-159`) was designed for. But this race only exists under Shape C; Shape A sidesteps it.

**The lock-release-during-compute pattern, documented for completeness** (in case Shape A is later replaced by Shape C for InflightBatched scaling):

```text
loop {
    let step_decision = {
        let mut sched = scheduler.lock().unwrap();
        sched.step()
    }; // lock released
    match step_decision {
        SchedulerStep::Prefill { handle, n_tokens } => {
            // long compute outside the lock — handlers can admit concurrently
            let consumed = run_prefill(handle.slot_id, n_tokens);
            {
                let mut sched = scheduler.lock().unwrap();
                sched.advance_after_prefill(handle, consumed);
            } // lock released
        }
        // ...
    }
}
```

The SlotHandle's generation counter validates the callback: if a stale handle (a concurrent release happened during the compute) hits `advance_after_prefill`, the scheduler silently drops it as a no-op (`scheduler.rs:773-787`).

**Shape A note**: even though no race exists, calling `advance_after_*` is still correct hygiene under Shape A — it keeps `SchedulerStats` (queue depth, completion counters) accurate for `/metrics`. Without the calls, `admitted_total` would still increment (from `admit`) but `completed_total` would stay at 0 — operator-confusing.

---

### §2.10 Q10: Testability

**Minimum integration test for C2 iter-2a**: `engine_serial_fifo_byte_equivalent_to_pre_phase_c` (proposed in §2.5 above). Lives at `tests/multi_slot_engine.rs` (NEW file) OR as a sibling test in the existing `engine.rs` `#[cfg(test)] mod tests` block. Recommendation: NEW file `tests/engine_byte_equivalence.rs` because (a) it's an integration test (uses both `Engine::spawn` and `Engine::spawn_with_mode` from outside the crate's private surface), (b) it's structurally the C2 acceptance test and deserves its own file.

**Synthetic test for SlotAware (iter-2b+)**:
```text
#[test]
fn slot_aware_4_concurrent_requests_no_cross_contamination() {
    // ... constructs Engine with EngineMode::SlotAware { max_slots: 4 } ...
    // ... sends 4 concurrent requests with distinct seeded prompts ...
    // ... awaits all 4; asserts each completion is the expected per-prompt result ...
    // ... asserts SchedulerStats shows in_flight_slots peaked at 4 (or however high) ...
}
```

Lives at `tests/multi_slot_engine.rs` (NEW file at iter-2b).

**Can the test avoid loading a real model?** YES for the byte-equivalence test if a synthetic Gemma fixture is built (the `make_synthetic_kv_engine_for_test` at `engine.rs:602-606` is a no-op worker that drains the channel without running real inference — useful for the `EngineInner` lifecycle / handler-route tests but NOT for the byte-equivalence test which needs real `generate_once` execution). For real-inference byte-equivalence, an env-gated path using a tiny GGUF (Qwen3.5-0.6B-Q4_K_M is ~400 MB; loadable in ~1 second on M5 Max) is the iter-2 acceptance pattern. **UNVERIFIED — needs probe**: whether CI has access to a tiny GGUF for this purpose, or whether the test must be gated behind `HF2Q_BYTE_EQUIV_E2E=1`.

**For SlotAware tests at iter-2b+**: same constraint — needs a tiny real GGUF for end-to-end inference. The scheduler-only unit tests at `scheduler.rs:969+` are already comprehensive (30+ tests covering FSM transitions, stale-handle no-ops, race-under-mutex semantics); those serve as the scheduler-correctness pin. C2's tests are the ENGINE-INTEGRATION pin, which is fundamentally GGUF-dependent.

---

### §2.11 Q11: Concrete testable hypotheses

| # | Hypothesis | Falsifying test | Cost to falsify |
|---|---|---|---|
| H1 | Under `EngineMode::SerialFifo`, `engine.generate(prompt, params)` produces byte-identical `GenerationResult` to the pre-C2 path on a fixed-seed synthetic Gemma fixture. | `engine_serial_fifo_byte_equivalent_to_pre_phase_c` (proposed in §2.5) | 1 day (test scaffolding + GGUF fixture) |
| H2 | Under `EngineMode::SerialFifo`, calling `engine.generate(p1)` followed by `engine.generate(p2)` produces results byte-identical to calling them in the same order pre-C2 — i.e. the wrapper does NOT introduce inter-request state leakage. | `engine_serial_fifo_two_sequential_requests_no_state_leak` — same fixture as H1 but two sequential generates with assertions on both results | 2 days (additional `Qwen35LoadedModel.persistent_kv_cache` lifecycle correctness) |
| H3 | The 3-arg `Engine::spawn` signature remains compile-time stable. | `engine_spawn_3_arg_signature_compile_pin` (already exists at iter-1.5) — confirmed live | 0 (already passes; iter-2 must keep it passing) |
| H4 | `worker_run` accepting a `mode: EngineMode` parameter does not change the binary's `worker_run` symbol's behaviour for the `SerialFifo` arm — verifiable via `cargo asm` diff between iter-1.5 and iter-2 binaries on the `worker_run::SerialFifo` block. | Manual `cargo asm hf2q::serve::api::engine::worker_run --release` diff — out-of-band; not a unit test | 4 hours (not a routine CI gate) |
| H5 | The 11 existing handler callsites at `engine.rs:2832, 2859, 2895, 2949, 2994, 3031, 3077, 3155, 3174, 3235` remain UNCHANGED in iter-2 — verifiable via `git diff src/serve/api/engine.rs` showing zero hunks intersecting those line numbers (modulo iter-2's overall changes to the file). | Iter-2 PR review check (manual, not test) — or a hash-pin script that fingerprints each callsite's `tx.try_send(...)` block | 1 hour (PR-review checklist item) |

**Stakes if H1 falsifies**: C2 iter-2a is wrong; the wrapper introduces a behaviour change. Halt and investigate before any further work.

**Stakes if H2 falsifies**: the `Qwen35LoadedModel.persistent_kv_cache` lifecycle is incorrect — likely a missed `kv_cache.drop_seq(SlotId(0))` between requests in the FifoSerial arm. The fix is localized.

**Stakes if H4 falsifies (asm diff non-zero)**: not blocking — likely just a layout change from adding the `mode` parameter to `worker_run`. Verify the SerialFifo arm's emitted code is equivalent at semantic level (`cargo bench` baseline on a tiny model) before shipping.

**Stakes if H5 falsifies**: probably a Bad Idea — Chesterton's fence on the handler interface. If iter-2 finds it MUST change a handler callsite, that's a design-decision-revisit gate.

---

### §2.12 Q12: R1–R5 risk register specific to C2

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | KV cache lifecycle mismatch between Qwen35 (per-request alloc) and Gemma (per-engine alloc) creates an awkward iter-2 scope boundary. | **High** (verified: Qwen35 allocs at `engine_qwen35.rs:1263`; Gemma's KV lives in `MlxModelWeights`) | **High** (could derail iter-2 timeline if not addressed up-front) | iter-2a ships Qwen35 SlotAware only; Gemma SlotAware gated on Phase A3 lift landing. Document the scope split explicitly in the iter-2 PR. iter-2a's Qwen35 refactor lifts `alloc_kv_cache_for_request` out of the per-request path into `Qwen35LoadedModel.persistent_kv_cache` allocated at first-request-time with `n_seqs=max_slots`. |
| **R2** | Worker-thread-owned scheduler (Shape A) cannot service admission-during-decode under InflightBatched — the worker blocks on `rx.recv()` between requests, so no admit can race a decode. | **High under Shape A iter-2a** (this is BY DESIGN for iter-2a — SerialFifo only) | **Low for iter-2a** (the scope explicitly excludes InflightBatched runtime); **Medium for iter-2b** (Shape B body inside worker_run will be more complex than Shape A) | iter-2a ships ONLY SerialFifo wrapping. iter-2b extends `worker_run` body with a `match mode { SlotAware => select! { ... } }` arm. The `select!` uses `tokio::select!` on `rx.recv()` + a `compute_finished` notify; admit happens on `rx.recv()` arrival without blocking the worker on an in-flight forward. |
| **R3** | Per-arch KV cache (Gemma vs Qwen35) requires trait-object boxing on `EngineInner` that loses optimizer-friendly type info on the hot path. | **Low** (per §2.4.2 recommendation, the typed-cache-per-arch pattern avoids this entirely) | **Medium if it occurs** (decode hot path is sensitive — ~700µs/token sampling, 10-100ms forward) | Recommendation in §2.4: push the per-arch typed cache onto `Qwen35LoadedModel` / `GemmaLoadedModel` directly, NOT onto `EngineInner` as `Box<dyn MultiSeqKvCache>`. Worker thread dispatches on the existing `match &mut loaded: LoadedModel` arms; the per-arch arm calls the typed cache directly. The trait is used for cross-cutting code (error → HTTP mapping) but NOT on the data path. |
| **R4** | Spec-decode (EAGLE-3 / DFlash) + multi-slot interaction is undefined; Phase A4 deferred to research-quality (ADR-040 §6 + §4 question 5). C2 iter-2 might accidentally route a spec-decode request to a multi-slot path that doesn't support the drafter KV cache. | **Medium** (spec-decode is opt-in via `HF2Q_SPEC_EAGLE3=1` per ADR-038; if an operator sets BOTH spec-decode AND SlotAware, an undefined-behaviour combination is reachable) | **Medium** (likely failure mode is a typed error from the drafter KV cache, not silent corruption — the drafter's single-seq KV at `dflash/kv_cache.rs:48` would panic on `slot_id > 0`) | iter-2's spawn_with_mode validates: if `mode == SlotAware { max_slots > 1 }` AND the engine is constructed with a spec-decode flag set, reject with a typed `EngineSpawnError::SpecDecodeMultiSlotUnsupported`. Pin in test `slot_aware_with_spec_decode_returns_typed_error`. |
| **R5** | `SchedulerStep::Mixed` dispatch needs kernel-level batching that B4a doesn't ship; C2 iter-2's worker dispatches Mixed as N+1 sequential forward_gpu calls (correct but slow). | **Low for iter-2a** (FifoSerial never returns Mixed); **Certain for iter-2b** (Mixed is the standard step output once 2+ slots are in different phases) | **Low** (correct behaviour; just sub-optimal — closes the AC-3 contract; AC-4 throughput is the gate for kernel-level batching) | iter-2b dispatches Mixed as sequential forward calls per §2.8. Phase D iter-2 measures the actual throughput impact (per ADR-040 §6 Phase D). If aggregate throughput falls below the AC-4 1.5× bar, open Phase B6 — kernel-level batched dispatch — as a follow-up ADR. |

**Additional risk noted by the A2/A3 dossier (not specific to C2 but relevant)**:

- **R4-bis (from A2 dossier)**: Qwen35 hybrid persistor wire format supports `n_seqs > 1` (`qwen35_hybrid_persistor.rs:171-175`) but has never serialized a value > 1. C2 iter-2a's switch to `n_seqs=max_slots` exercises this for the first time; ADR-017 KV-spill could break under multi-slot. **Mitigation**: A2 dossier §2.4.1 recommends adding `qwen35_hybrid_persistor_roundtrip_n_seqs_4` test at iter-2; C2 should NOT be the discovery vehicle for a spiller bug.

---

## §3 Concrete hypothesis matrix

| ID | Hypothesis | Test name (proposed) | Falsifies what claim | Cost | Stakes if false |
|---|---|---|---|---|---|
| H1 | Under `EngineMode::SerialFifo`, `engine.generate(prompt, params)` produces byte-identical `GenerationResult` to the pre-C2 path. | `engine_serial_fifo_byte_equivalent_to_pre_phase_c` | ADR-040 §3.6 byte-equivalence claim — the load-bearing C2 contract | 1 day (test scaffolding + small GGUF fixture) | C2 iter-2a is wrong; halt + investigate before any further work. |
| H2 | Two sequential generates through `EngineMode::SerialFifo` produce byte-identical pairs to pre-C2 — i.e. wrapper does not leak inter-request state. | `engine_serial_fifo_two_sequential_requests_no_state_leak` | Sequence-level byte-equivalence (catches missed `drop_seq` between requests in the Qwen35 persistent-cache lift) | 2 days | Qwen35 persistent KV cache lifecycle bug; localized fix. |
| H3 | The 3-arg `Engine::spawn` signature remains compile-time stable. | `engine_spawn_3_arg_signature_compile_pin` (already live at iter-1.5) | ADR-040 §3.6 + AC-3 backward-compat contract | 0 (already passes) | C2 broke the contract — abort + revert. |
| H4 | The 11 existing handler callsites at `engine.rs:2832, 2859, 2895, 2949, 2994, 3031, 3077, 3155, 3174, 3235` remain UNCHANGED in iter-2 (modulo the `tq_packed_v2_*` block at :2994/:3031 which uses `try_send` only). | Manual PR-review checklist + a `git diff` invariant script | Chesterton's-fence on the handler interface | 1 hour | Indicates iter-2 design is leaking scope into handler layer; revisit. |
| H5 | Under `EngineMode::SlotAware { max_slots: 4 }` (iter-2b scope), 4 concurrent `engine.generate` calls all complete without cross-contamination of K/V state across slots. | `slot_aware_4_concurrent_requests_no_cross_contamination` (iter-2b) | ADR-040 §5 AC-3 SlotAware correctness | 3 days (iter-2b scope; requires real GGUF + 4 tokio tasks) | iter-2b SlotAware runtime is wrong; possibly the per-slot KV slot-id threading per B4a-cont/B4b is incomplete. |

---

## §4 Recommended sequencing for C2

### Iter-2a — FifoSerial wrapping (recommended start)

**Goal**: ship Shape A worker_run wrapping for FifoSerial path; lock the byte-equivalence test; defer all SlotAware runtime to iter-2b.

**Steps (in order)**:

1. **Write the byte-equivalence test FIRST** (`engine_serial_fifo_byte_equivalent_to_pre_phase_c`) and confirm it PASSES at HEAD (iter-1.5). This proves the test harness is correct before iter-2 changes the engine. Cost: 1 day.
2. **Extend `worker_run` signature** with `mode: EngineMode` + `queue_capacity: u32` parameters. Both spawn entry points pass them. Compile, verify no behavior change (test from step 1 still passes). Cost: 0.5 day.
3. **Construct `Box<dyn Scheduler>` at worker_run entry** based on `mode`. For SerialFifo path: `Box::new(FifoSchedulerAdapter::new(queue_capacity))`. For SlotAware: still rejected via `EngineSpawnError::ModeNotYetWired` at `spawn_with_mode` per iter-1.5 — so worker_run only ever sees `SerialFifo`. Cost: 0.5 day.
4. **Wrap each Request arm in admit→drive→release** for the FifoSerial path. Only the `Generate`, `GenerateStream`, `Embed`, `GenerateWithSoftTokens` arms need wrapping (the 4 generation paths); the control-plane arms (KvSnapshot/Restore, PromptCacheSnapshot/Restore, TqPackedKv*, Warmup, Shutdown) don't go through the scheduler. Cost: 1.5 days.
5. **Update `EngineInner`** with `max_slots: u32` + `scheduler_stats_snapshot: Arc<Mutex<SchedulerStats>>`. Wire a periodic stats update from worker thread (after each release). Add `engine.scheduler_stats()` accessor for `/metrics`. Cost: 0.5 day.
6. **Lift `alloc_kv_cache_for_request` for Qwen35**: add `Qwen35LoadedModel.persistent_kv_cache: Option<HybridKvCache>` (allocated at first-request-time with `n_seqs = max_slots`); modify `generate_qwen35_once` to acquire slot 0 from the persistent cache instead of allocating fresh. For iter-2a (SerialFifo only, max_slots=1) this is structurally a no-op — but it positions iter-2b for the SlotAware extension. Cost: 1 day. UNVERIFIED — needs careful review of `kv_cache.snapshot()` / `restore_from()` interaction (`engine_qwen35.rs:1503`) under the persistent-cache shape.
7. **Run H1+H2 tests**; resolve any divergence. Cost: 1 day.
8. **Acceptance**: iter-2a CFA pass + all existing 100+ engine tests pass + H1+H2 pass.

**Total estimated effort**: 5-7 days.

**Iter-2a out-of-scope (deferred to iter-2b)**:
- `SchedulerPolicy::InflightBatched` runtime
- Worker `select!` loop for admission-during-decode
- Mixed dispatch
- Gemma SlotAware (gated on Phase A3 lift)
- Spec-decode + multi-slot rejection (gated on R4 mitigation design)

### Iter-2b — SlotAware runtime for Qwen35

**Goal**: replace the `EngineSpawnError::ModeNotYetWired` rejection with the live `InflightBatchedScheduler` runtime for Qwen35.

**Steps**:

1. Extend `worker_run` body with a `match mode { SerialFifo => simple_path(...), SlotAware { .. } => slot_aware_path(...) }` branch. SerialFifo path is iter-2a's body unchanged. SlotAware path: `tokio::select!` between `rx.recv()` + `compute_finished_notify`.
2. Wire `step()` dispatch on the SlotAware arm — Prefill / Decode / Mixed (Mixed = sequential per §2.8).
3. Wire `advance_after_prefill` / `advance_after_decode` callbacks.
4. Update H5 test (`slot_aware_4_concurrent_requests_no_cross_contamination`) to pass.
5. Gate spec-decode incompatibility (R4) at `spawn_with_mode` validation.

**Estimated effort**: 5-8 days. Gated on B4b shipping first (decode-side slot_id threading per ADR-040 §6.1.4 row B4b).

### Iter-2c — Gemma SlotAware (gated on Phase A3)

**Goal**: extend SlotAware support to Gemma after the Phase A3 KV cache lift lands.

**Steps**: parallel to iter-2b but for `GemmaLoadedModel`. Gated on A3 + B4c.

**Estimated effort**: 3-5 days, gated on A3+B4c readiness (~8-10 days out).

### Recommended overall sequencing

```text
iter-2a (FifoSerial wrapping, Qwen35 cache lift) [5-7d]
  ├─→ iter-2b (Qwen35 SlotAware runtime) [5-8d] — gated on B4b
  └─→ iter-2c (Gemma SlotAware runtime) [3-5d] — gated on A3+B4c
```

**Do NOT** start iter-2b before iter-2a's H1+H2 are GREEN. The byte-equivalence proof is the load-bearing pin; without it, iter-2b's correctness has no reference.

---

## §5 Risk register (R1–R5)

| # | Risk | Likelihood | Impact | Mitigation summary | Gating decision |
|---|---|---|---|---|---|
| R1 | KV cache lifecycle mismatch (Qwen35 per-request vs Gemma per-engine) | High | High | Scope split: iter-2a Qwen35 only; iter-2c Gemma gated on A3. Lift `alloc_kv_cache_for_request` into `Qwen35LoadedModel.persistent_kv_cache`. | iter-2a must explicitly defer Gemma SlotAware; iter-2c blocked until A3 ships. |
| R2 | Shape A cannot service admission-during-decode under InflightBatched | High (intended for iter-2a) | Low for iter-2a; Medium for iter-2b | iter-2a ships only SerialFifo wrapping; iter-2b extends with select! loop for SlotAware path. | iter-2a explicitly out-of-scope for SlotAware runtime; iter-2b lands the SlotAware shape inside worker_run. |
| R3 | Per-arch KV cache requires trait-object boxing on EngineInner | Low | Medium | Push typed cache onto per-arch `LoadedModel` variant; trait used for error mapping only, NOT on data path. | Iter-2a design uses typed `HybridKvCache` directly in Qwen35 worker arm; no `Box<dyn MultiSeqKvCache>` in hot path. |
| R4 | Spec-decode + multi-slot undefined; HF2Q_SPEC_EAGLE3 + EngineMode::SlotAware combo unsupported | Medium | Medium | iter-2b's spawn_with_mode rejects the combination with typed error `EngineSpawnError::SpecDecodeMultiSlotUnsupported`. | iter-2b PR includes the rejection + test `slot_aware_with_spec_decode_returns_typed_error`. |
| R5 | SchedulerStep::Mixed dispatch needs kernel-level batching; B4a doesn't ship it | Low for iter-2a; Certain for iter-2b | Low | iter-2b dispatches Mixed as N+1 sequential forward_gpu calls (correct, slow). Phase D measures throughput impact; if below AC-4 1.5× bar, open Phase B6 as follow-up ADR. | iter-2b implementation note: Mixed = prefill_then_decode_loop, NOT a single forward. Phase D iter-2 measures the gap. |

**Additional ground-state risks identified**:

| # | Risk | Source | Mitigation |
|---|---|---|---|
| R6 | UNVERIFIED — multi_model.rs may have a live `Engine::spawn` callsite I missed at this dossier authoring time. | grep limitation | iter-2 step 0: `grep -rn "Engine::spawn\b" /opt/hf2q/src/` and verify ALL production callsites are 3-arg-compatible. |
| R7 | UNVERIFIED — `kv_cache.snapshot()` / `restore_from()` interaction at `engine_qwen35.rs:1503` under persistent-cache shape may need lifecycle changes. | LCP prompt cache replay (`PromptCache` lookup at `:1503`) restores into a fresh cache; under persistent-cache, restore needs to target slot 0 only. | iter-2a step 6: probe the prompt-cache restore path before lifting; may need a `HybridKvCache::restore_from_slot(snap, slot)` variant. |
| R8 | UNVERIFIED — the byte-equivalence test (H1) may need a real GGUF; CI may not have one available. | Synthetic Gemma fixture is no-op; real inference requires real weights. | iter-2a step 1 probe: check whether a tiny GGUF (~400 MB) is available for CI. If not, gate the test behind `HF2Q_BYTE_EQUIV_E2E=1` env. |

---

## §6 Closure note

The C2 wiring is a structural lift, not a kernel change. Shape A's correctness rests on a single observation: the existing `mpsc::channel` + `worker_run` body IS the byte-equivalence target, and the cleanest C2 path wraps that body in `admit→drive→release` without changing its essential structure. The SlotAware runtime is then a separable iter-2b concern that extends the worker body without disrupting iter-2a's locked-down SerialFifo path.

The single architectural surprise the dossier surfaces is the **Qwen35 per-request KV cache allocation pattern** (`engine_qwen35.rs:1027-1049, 1263`). This is not noted in ADR-040 §1.3's "existing footholds" framing. For C2 SlotAware to work, this must be lifted to engine-lifetime ownership — a refactor that is bounded (one callsite + one struct field + one allocator helper) but does change Qwen35's existing memory model materially. The iter-2a step 6 + R7 probe address this.

**Confidence in this dossier**: medium. Every claim above has a `file:line` citation. The four UNVERIFIED claims (R6, R7, R8, and the `multi_model.rs` grep finding) are noted and pinned to iter-2 step-0 probes; each is ≤4 hours to falsify before any production code lands.

The recommended next action: **start iter-2a with step 1 (write the H1 byte-equivalence test against HEAD and confirm PASS)**. If H1 doesn't pass at HEAD, the entire iter-2 plan needs re-grounding — the harness is wrong.
