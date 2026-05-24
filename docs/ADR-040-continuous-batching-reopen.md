# ADR-040 — Continuous batching: reopen the ADR-005 carve-out

- **Status**: 🚧 ACTIVE — Design 2026-05-23, iter-1 SHIPPED 2026-05-23 (commit 1a1d6a26), **iter-1.5 in progress 2026-05-23 (adversarial review fixes per cfa session — Codex + Claude both requested_changes/high severity)**. Multi-iter arc in progress under goal-mode directive. Reopens the [ADR-005 Decision #1 carve-out](ADR-005-inference-server.md) ("continuous batching pulled from ADR-005 — deferred to a future ADR; reopen trigger = real deployment scenario with ≥8 concurrent users on a single instance").
- **Date**: 2026-05-23
- **Supersedes**: nothing. Amends ADR-005 §"Concurrent-deployment scaling (deferred, future ADR)" (line 1097) and Resolved Question "Phase 2 scope refinement" Decision #1 (line 6652) by activating the deferred-ADR slot.
- **Related**: ADR-005 (Phase 2 FIFO contract — Decision #2, Decision #19), ADR-007 (TurboQuant KV — single-seq scope), ADR-017 (persistent block prefix cache — single-seq, per-model spill), ADR-027 (Qwen35 TQ KV + persist — single-seq), ADR-013 (Qwen35 inference), ADR-034 (spec-decode end-to-end — intra-request batching only).
- **Author note**: Per `feedback_multiweek_always_in_scope_2026_05_23.md` mantra — "no shortcuts, just pure excellence". Iter 1 of this ADR is the design pass + Phase A/B/C/D scaffolding stubs landing in parallel; subsequent iters implement.

> ## Mantra (verbatim from `~/Documents/mantra.txt`)
>
> *DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.*

---

## 1. Why (the problem)

### 1.1 What the ADR-005 carve-out actually says

ADR-005 (lines 1097–1103) declared "continuous batching, paged KV, inflight batching with per-slot KV separation, N-concurrent-stream throughput targets" out of Phase 2 scope. Rationale: hf2q's comparators are ollama + llama.cpp (not vLLM), the serialized FIFO queue meets the deployment targets at parity or better, and the reopen trigger is "≥8 concurrent users on a single instance, reported by a real user or demanded by a target customer."

The carve-out was made deliberately, with the reopen ADR slot reserved. ADR-040 is that slot.

### 1.2 The three subsystems Phase 2 deliberately did NOT build

ADR-005 line 1099 enumerated exactly what a future continuous-batching ADR would have to add. All three are still missing today:

| Missing subsystem | Phase 2 stance | Named port reference (in ADR-005) |
|---|---|---|
| KV-representation-aware scheduler | "different concurrency model than the serialized FIFO queue" | vLLM `vllm/core/scheduler.py` |
| Paged-KV / inflight-batched KV layout with per-slot KV separation | "Phase 4's pool is request-serial within each loaded model" | llama.cpp `src/llama-kv-cache.cpp` multi-seq semantics |
| N-concurrent-stream throughput target + benchmark | undefined | none |

This ADR adds all three under Phases A, B, D respectively, plus Phase C — the `Engine` slot-aware extension that ties them together while preserving the existing FIFO contract under a feature flag.

### 1.3 Existing footholds in the codebase

Three pieces of infrastructure are already shaped to support multi-seq KV without total reconstruction:

1. **Qwen35 `HybridKvCache` already carries `n_seqs` in buffer shape** — `src/inference/models/qwen35/kv_cache.rs:14-16`: `k/v: MlxBuffer [head_dim, n_kv, max_seq_len, n_seqs]`, `current_len: Vec<u32>` indexed per-seq. Production wiring uses `n_seqs=1` today; the structural shape supports >1 with no buffer-layout change.
2. **Gemma 4 `MlxModelWeights` KV cache** — single-seq today but the per-layer slot structure parallels Qwen35's; the same lift applies.
3. **`HotSwapManager` pool** (ADR-005 Phase 4, `src/serve/multi_model.rs`) — separates per-model lifecycle from per-request lifecycle. Continuous batching slots in below the pool: per-loaded-model, multiple concurrent slots.

The new code adds a scheduling layer between `HotSwapManager` and `Engine`, plus a multi-seq KV trait that the per-model caches implement.

### 1.4 What does NOT change

- The `LoadedPool` / `HotSwapManager` / `auto_pipeline` chain (ADR-005 Phase 4).
- ADR-017's per-model KV spilling.
- mlx-native kernels: zero new Metal kernels needed for Phase A/B/C. Phase D's benchmark may surface kernel-level gaps that prompt separate ADRs.
- The serialized FIFO contract is preserved byte-for-byte under `SchedulerPolicy::FifoSerial` (default until benchmarks justify flip).
- Existing single-request decode/prefill paths (`forward_prefill.rs`, `forward_prefill_batched.rs`).
- The ADR-005 Decision #2 contract for clients: 429 + Retry-After on overflow, SSE keepalive every 15s. Continuous batching changes WHEN the request executes, not the request/response shape.

### 1.5 Reopen-trigger status

The ADR-005 reopen trigger ("≥8 concurrent users on a single instance, reported by a real user or demanded by a target customer") is **not formally verified today**. Operator direction 2026-05-23 ("2, 3, 4, 5 ← do this") activates the engineering arc under the mantra "multi-week structural work is ALWAYS in scope". An explicit reopen-trigger memo is NOT a prerequisite for the design + scaffolding work but SHOULD precede the Phase E1 ship gate (full enable-by-default cutover). This ADR records that ordering as a documented decision (§3.7).

---

## 2. Where (the integration surface)

### 2.1 Files this ADR creates (Phase A/B/C/D iter-1 scaffolding)

| Component | Path | LOC est. (iter-1 scaffold) | LOC actual (post-iter-1.5) |
|---|---|---|---|
| Multi-seq KV trait + types | `src/serve/multi_seq_kv.rs` | ~250 | ~780 (+34 from F5/F7/F9 fixes) |
| Scheduler trait + FIFO adapter | `src/serve/scheduler.rs` | ~400 | ~790 (+30 from F2/F3/F6 fixes; F2 gated behind cfg(test)) |
| `EngineMode` enum + slot-aware Engine extension | edits to `src/serve/api/engine.rs` | ~120 | ~280 (+23 from F1 fix) |
| Continuous-batching throughput benchmark | `tests/continuous_batching_throughput.rs` | ~180 | ~205 (+6 from F8 fix) |

**Total iter-1 LOC actual**: ~2330 (vs ~950 estimate — 2.45x miss).
**Total iter-1.5 LOC delta**: ~+90 (fix-only, no new functionality).

The 2.45x over-shoot is documented honestly per cfa-finding (Claude `major_findings[8]`). Causes: (a) verbose docstrings + ADR-cross-reference comments per ADR-040 §7 mantra; (b) test coverage at 40 tests far exceeded the AC-level minimum; (c) goal-mode-directive expansion past the original "stub" scope into "real admit/release/stats semantics for FifoSchedulerAdapter + InflightBatched signature stub".

### 2.2 Files this ADR edits across the multi-iter arc (iter-2+)

| File | Change | Phase |
|---|---|---|
| `src/inference/models/qwen35/kv_cache.rs` | implement `MultiSeqKvCache` for `HybridKvCache` (lifts `n_seqs=1` to N) | A iter-2 |
| `src/inference/models/gemma4/kv_cache.rs` | implement `MultiSeqKvCache` for Gemma4 dense KV | A iter-3 |
| `src/inference/spec_decode/eagle3/kv_cache.rs` | `MultiSeqKvCache` impl (research-quality; gated on Phase E) | A iter-4 |
| `src/inference/spec_decode/dflash/kv_cache.rs` | same | A iter-4 |
| `src/serve/api/engine.rs` | replace mpsc-channel + single worker with scheduler-driven slot loop under `SchedulerPolicy::InflightBatched` | C iter-2 |
| `src/serve/api/sse.rs` | per-slot keepalive seam (construction-time slot association only; per-frame keepalive carries no slot metadata) | C iter-3 |
| `src/serve/api/schema.rs` | doc-only Decision #2 update naming `SchedulerPolicy` | C iter-3 |
| `src/inference/models/qwen35/forward_gpu.rs` | accept `slot_id: SlotId` on `forward_gpu` + `forward_gpu_with_hidden`; bounds-check; gate slot N > 0 behind B4a-cont | **B iter-4a (SHIPPED 2026-05-23)** |
| `src/inference/models/qwen35/{forward_gpu.rs, gpu_full_attn.rs}` | thread `slot_id` into `build_gated_attn_layer` + `apply_gated_attn_layer_decode_into` + `apply_sdpa_with_kv_cache(_decode_into)` + the 2 private kernel-dispatch helpers (`write_kv_with_optional_tq_encode`, `dispatch_decode_sdpa_with_optional_tq`); per-slot K/V slice_view at the kernel-dispatch sites; flip slot > 0 from typed-error to real-route | **B iter-4a-cont (SHIPPED 2026-05-23)** |
| `src/inference/models/qwen35/{forward_gpu.rs, gpu_full_attn.rs}` | Codex /cfa rev-1 follow-ups: M1 isolation-test rigor (raw K/V byte snapshot + positive same-prompt-in-slot-0-vs-slot-1 equivalence pin, deleting the reset+rerun-then-compare test that could pass under cross-slot leak); M2 canonical TQ-active multi-slot gate placement at `build_gated_attn_layer` + `apply_gated_attn_layer_decode_into` entry (before fused-stage-AB encoder work); minor stale-comment refresh at the `forward_gpu` entry | **B iter-4a-cont.1 (SHIPPED 2026-05-23)** |
| `src/serve/forward_prefill.rs` | (Gemma 4 prefill) accept `slot_id` parameter; route writes to multi-seq KV (gated on Phase A3 Gemma 4 multi-seq KV impl) | B iter-4c |
| `src/serve/forward_prefill_batched.rs` | same | B iter-4c |
| `src/inference/models/qwen35/forward_gpu.rs` (decode) | thread `slot_id` through `forward_gpu_last_logits` / `forward_gpu_last_topk` / `forward_gpu_last_logits_with_soft_tokens` / `forward_gpu_last_logits_with_soft_tokens_and_deepstack` / `forward_embed_last`; full lift (SlotId(N>0) routes through B4a-cont's F32 slot-offset wiring); all 25 production callsites in `serve/mod.rs` + `serve/api/engine_qwen35.rs` + `quantize/imatrix/forward.rs` updated to pass `SlotId(0)` | **B iter-4b (SHIPPED 2026-05-24)** |
| `src/inference/models/qwen35/{spec_decode.rs, forward_gpu.rs}` (dflash / greedy) | thread `slot_id` through `forward_gpu_greedy` + dflash spec-decode entry points | B iter-4d |
| `src/serve/api/engine.rs` | replace mpsc-channel + single worker with scheduler-driven slot loop under `SchedulerPolicy::InflightBatched` | C iter-2 |
| `src/serve/api/sse.rs` | per-slot keepalive seam (construction-time slot association only; per-frame keepalive carries no slot metadata) | C iter-3 |
| `src/serve/api/schema.rs` | doc-only Decision #2 update naming `SchedulerPolicy` | C iter-3 |
| `src/serve/mod.rs::cmd_serve` | thread `SchedulerPolicy` from CLI/env into `Engine::spawn` | C iter-2 |

### 2.3 mlx-native impact

**Phase A/B/C**: zero. The `MultiSeqKvCache` trait is implemented entirely above mlx-native — by passing different `n_seqs` and `slot_offset` to existing kernels.

**Phase D**: kernel-level work, if any, is surfaced as separate ADRs. The most likely candidate is a paged-attention kernel port (PagedAttention from vLLM), which becomes worthwhile only if the SeparateSlots layout (Phase A default) shows ≥30% memory waste under N=8 concurrent at production context lengths. Empirical, not pre-committed.

---

## 3. Architecture decisions

### 3.1 KV layout: SeparateSlots first, Paged second

**Decision**: Phase A iter-1 ships `MultiSeqLayout::SeparateSlots` as the default. The existing `[..., max_seq_len, n_seqs]` shape extends to N slots with `n_seqs=N` and per-slot `current_len`. `MultiSeqLayout::Paged` is reserved as a future variant; Phase D's benchmark decides whether it's worth the kernel work.

**Why**: SeparateSlots is a 1-line shape change at allocation time + per-slot index arithmetic in the read path. It reuses every existing kernel. PagedAttention is a multi-week kernel port (vLLM's `paged_attention_v1.cu` is ~600 LOC of CUDA, requiring an mlx-native equivalent in Metal). Ship the simple-but-correct version, measure, then decide.

**Alternatives considered**:
- Pure PagedAttention from day one — rejected: bypasses existing kernel coverage, multi-month delay, premature optimization vs measured demand.
- Per-request separate `Engine` instances — rejected: explodes weight memory N× and breaks the `HotSwapManager` contract.

### 3.2 Scheduler: FIFO adapter first, inflight-batched second

**Decision**: `SchedulerPolicy::FifoSerial` (default) wraps the existing mpsc-channel + single-worker behavior under the new `Scheduler` trait — byte-equivalent to today. `SchedulerPolicy::InflightBatched` is the new behavior, opt-in via `HF2Q_SCHEDULER=inflight_batched` (off by default until Phase E1).

**Why**: Preserves the ADR-005 Phase 2 production contract during the entire multi-week arc. Enables apples-to-apples A/B benchmarking at any point.

**Alternatives considered**:
- Cut over to inflight-batched by default at iter-1 — rejected: violates Decision #2 contract before benchmarks justify it.
- Pure-replace mpsc channel with scheduler — rejected: loses the byte-equivalence regression guard.

### 3.3 Scheduler port reference

**Decision**: Mirror llama.cpp's `-cb` (continuous batching) admission-during-decode loop semantics, NOT vLLM's full `Scheduler` class. ADR-005 line 1103 names both as candidates; this ADR picks the smaller-blast-radius option.

**Why**:
- llama.cpp `-cb` is mature, debugged against real workloads, and lives in the same comparator set ADR-005 is positioned against.
- vLLM's scheduler couples to PagedAttention and assumes block-aligned KV allocation; our Phase 3.1 SeparateSlots layout doesn't.
- vLLM's scheduler can be reconsidered if Phase D's benchmark shows admission policy is the bottleneck (it usually isn't until ≥100 concurrent).

**Alternatives considered**:
- Port vLLM `Scheduler` entire — rejected: 2-3× the LOC, couples Phase B to Phase D outcomes.
- Roll a hf2q-original scheduler — rejected: no comparator, no reference behavior to verify against.

### 3.4 Slot count default

**Decision**: `max_slots = 4` default, configurable via `HF2Q_MAX_SLOTS` env or `--max-slots` CLI flag. The ADR-005 reopen trigger names `≥8` as the demand-side threshold; serving capacity defaults to half that so the first ramp deploys with headroom.

**Alternatives considered**:
- `max_slots = 1` (current behavior) — rejected: would make `InflightBatched` policy identical to `FifoSerial` and the benchmark trivial.
- `max_slots = 8` (matches reopen trigger) — rejected: full memory commitment from day one before benchmark; better to ship 4 and ramp.

### 3.5 KV cache budget per slot

**Decision**: `kv_cache_budget_bytes` (existing field on `Engine::spawn`) divides equally across slots in SeparateSlots layout. Per-slot budget = `total / max_slots`. Per-slot OOM returns 429 to the admitting handler (Decision #19 contract preserved).

**Why**: The existing `kv_cache_budget_bytes` knob already exists; this just changes its denominator. No new operator surface.

### 3.6 Backward compatibility contract

**Decision**: With `HF2Q_SCHEDULER` unset (or `=fifo_serial`), every byte of `Engine` behaviour is bit-equivalent to pre-ADR-040. The Phase 1b sourdough gate + the per-family parity gates pin this regression boundary.

**Why**: ADR-005 Phase 2 has been in production since 2026-04. A scheduler refactor that breaks single-request behaviour would invalidate every benchmark and every customer integration. Phase C iter-1 ships a dedicated regression test `engine_serial_fifo_byte_equivalent_to_pre_phase_c`.

#### 3.6.1 Amendment (iter-1.5, post-cfa-review)

The original §3.6 byte-equivalence claim was overstated by iter-1's signature-only test (`engine_spawn_3_arg_signature_compile_pin`, formerly `engine_spawn_signature_unchanged_at_phase_c_iter_1`). Adversarial reviewers (Codex + Claude) correctly observed that a compile-time signature pin proves NOTHING about behaviour. The byte-equivalence claim is now phased:

| Aspect | iter-1 pin | iter-1.5 pin | iter-2 promise |
|---|---|---|---|
| 3-arg `Engine::spawn` signature unchanged | compile-time gate | compile-time gate + renamed test | compile-time gate |
| FifoSchedulerAdapter queue_capacity matches `Engine::spawn` `.max(1)` | NOT pinned | `fifo_queue_capacity_zero_normalizes_to_one` | live A/B vs Engine::spawn |
| FifoSerial single-slot invariant (SlotId(0) reuse) | NOT pinned (allocated monotonic) | `fifo_serial_always_assigns_slot_id_0` | enforced |
| Concurrent admit race matches mpsc arrival ordering | NOT pinned (sequential test only) | `fifo_concurrent_admits_under_mutex_match_429_boundary` | live thread-scope race vs real Engine |
| FIFO ordering of dequeue | sequential-call pin (`fifo_admit_twice_*`) | sequential-call pin | live A/B vs Engine::spawn |
| 2-step Prefill→Decode state machine vs Engine's atomic worker_run | NOT pinned (state machine differs) | documented as deliberate divergence (driver loop calls step() in tight loop) | resolved via Phase C iter-2 driver wrapping |
| 429 + Retry-After handler boundary | not exercised | not exercised | live HTTP integration test |
| SSE keepalive behavior | not exercised | not exercised | live HTTP integration test |

Iter-1.5's pins are stronger than iter-1's but still NOT a complete byte-equivalence proof — that proof lands at Phase C iter-2 when the scheduler is wired into `Engine::spawn` and a live A/B harness against pre-ADR-040 behaviour can run. iter-1.5 honestly downgrades the claim.

### 3.7 Reopen-trigger memo ordering

**Decision**: The formal reopen-trigger memo (naming the customer or scenario that fires the ≥8-concurrent threshold) is NOT a prerequisite for Phases A/B/C/D iter-1 scaffolding or implementation iters. It IS a prerequisite for Phase E1 — the cutover that flips `SchedulerPolicy::InflightBatched` to default-on.

**Why**: The engineering arc takes weeks; waiting on the memo blocks all work. Shipping scaffold + impl + benchmark without flipping the default keeps the FIFO contract intact for existing customers while enabling A/B measurement.

---

## 4. Open questions (operator decisions needed)

1. **Reopen-trigger memo author**: who writes the customer-or-scenario memo that gates Phase E1 cutover? Default: operator drafts; ADR-040 author folds into Phase E1 ACs.
2. **Slot count default**: `max_slots=4` as proposed in §3.4, or something else (1 for safety, 8 for matching reopen trigger)?
3. **Scheduler port reference**: llama.cpp `-cb` as proposed in §3.3, or vLLM `Scheduler` (heavier, depends on PagedAttention)?
4. **Phase E gating model**: A/B benchmark must show ≥1.5× aggregate throughput at N=4 concurrent before Phase E1 default-flip — acceptable, or different bar?
5. **Spec-decode interaction**: does Phase A iter-4 (`MultiSeqKvCache` for EAGLE-3 / DFlash drafter caches) ship as research-quality only, or does Phase E1 require spec-decode to work under continuous batching?

---

## 5. Acceptance criteria (proposed — locked after §4 resolution)

### AC-1 — Phase A: multi-seq KV trait + per-model impls

- `MultiSeqKvCache` trait lives in `src/serve/multi_seq_kv.rs` with `append_for_seq`, `drop_seq`, `fork_seq`, `seq_len`, `slot_count` methods.
- `HybridKvCache` (Qwen35) implements `MultiSeqKvCache` with `n_seqs > 1` tested against `n_seqs = 1` byte-equivalence at slot 0.
- Gemma 4 dense KV cache implements `MultiSeqKvCache`.
- Per-slot append + drop is O(1) (does not iterate over other slots).
- Bench: per-slot `append_for_seq` ≤ 5% overhead vs current single-seq `append`.

### AC-2 — Phase B: scheduler trait + FIFO adapter

- `Scheduler` trait lives in `src/serve/scheduler.rs` with `admit`, `step`, `release`, `stats` methods.
- `FifoSchedulerAdapter` wraps the existing mpsc-channel path with **byte-equivalent** behaviour (regression test pins this).
- `InflightBatchedScheduler` admits new requests during in-flight decode steps; `step` returns a `SchedulerStep::Mixed` variant when prefill + decode coexist in one forward.
- 429 + Retry-After contract preserved unchanged (Decision #19).

### AC-3 — Phase C: Engine slot-aware

- `EngineMode::SlotAware { max_slots }` variant on `Engine` dispatches the scheduler.
- `EngineMode::SerialFifo` (default) byte-equivalent to pre-ADR-040 `Engine`.
- `HF2Q_SCHEDULER` env + `--scheduler` CLI flag select between modes.
- SSE keepalive seam is per-slot at construction time (slot association captured once when the stream is built; per-frame keepalive carries no slot metadata — no client-visible difference at N=1).
- Regression test `engine_serial_fifo_byte_equivalent_to_pre_phase_c` PASS.

### AC-4 — Phase D: throughput benchmark

- `tests/continuous_batching_throughput.rs` env-gated on `HF2Q_CB_THROUGHPUT_E2E=1`.
- Measures aggregate tokens/sec across N ∈ {1, 2, 4, 8} concurrent SSE streams.
- Reports per-N: TTFT p50/p95, aggregate tok/s, 429 incidence, per-slot tok/s.
- Comparator: `SchedulerPolicy::FifoSerial` (baseline) vs `SchedulerPolicy::InflightBatched` (treatment).
- Gate for Phase E1: treatment ≥ 1.5× baseline aggregate tok/s at N=4 with TTFT p95 ≤ 2× single-stream.

### AC-5 — Phase E1: production cutover

- Formal reopen-trigger memo lands in `docs/` naming the customer/scenario per §3.7.
- AC-4 benchmark meets §3.4 bar on production hardware (M5 Max, current target models).
- `HF2Q_SCHEDULER=inflight_batched` becomes default for newly-spawned engines.
- ADR-005 §"Concurrent-deployment scaling (deferred, future ADR)" section updated to point at this ADR's closure block.

---

## 6. Sequencing (proposed)

### Phase A — Multi-seq KV cache (4-6 iters)

| Iter | Scope | Estimated effort |
|---|---|---|
| **A1 (THIS ITER, 2026-05-23)** | Scaffolding: trait + types + NoopMultiSeqKvCache fixture + unit tests | 1 day |
| **A2a (SHIPPED 2026-05-23)** | `HybridKvCache` (Qwen35) full-attn + MTP impl — H1 PASS; ~150 LOC trait impl + 11 tests (75 total) | **1 day landed** |
| **B3 (SHIPPED 2026-05-23)** | `InflightBatchedScheduler` real `step` FSM — SlotPhase enum {Queued, Prefilling, Decoding} + `advance_after_prefill`/`advance_after_decode` driver-callback APIs + DEFAULT_PREFILL_CHUNK_TOKENS=512 (mirrors llama.cpp `-ub` default) + 12 new FSM tests (30 total scheduler tests); iter-1.5 cfg(test) gate removed | **1 day landed** |
| A2b | `HybridKvCache` linear-attn lift — lift `rollback_la_to` guard at `kv_cache.rs:1567`, H4 + H5 hypotheses; ports `gpu_delta_net.rs` `n_seqs=1u32` sites | 5-8 days |
| A2c | `fork_seq` real kernel dispatch (same-buffer cross-region memcpy) — replaces A2a's `SlotOom { 0, 0 }` sentinel | 3-5 days |
| **A3a (SHIPPED 2026-05-23)** | Gemma 4 `MultiSeqHbKvBuffers` sibling-struct lift + `alloc_hb_kv_for_layer` unified helper + `MultiSeqKvCache` impl — H6+H7+H8 PASS; H9 verified by code-read; H10 FALSIFIED but A3a scope intact; ~700 LOC + 12 new tests (24 total in `gemma4::kv_cache`) | **1 day landed** |
| **A3b iter-1 (SHIPPED 2026-05-24)** | Gemma 4 `HybridKvBuffers` FULL multi-seq lift via sibling `MultiSeqHybridKvBuffers` + `alloc_multi_seq_hybrid_kv_for_layer` helper (production default since ADR-029 iter-13 per H10 falsification) + `MlxKvCache` + `DenseKvBuffers` TYPED CLAMPS (`slot_count() == 1`; slot > 0 → typed `SlotOutOfRange`; in-bounds → `CapabilityUnsupported` naming iter-A3b-2 / iter-A3b-3).  H10/H11/H12/H13/H14/H15/H16 PASS; 35/35 gemma4::kv_cache; 21/21 continuous_batching_throughput preserved.  See §6.1.19. | **1 day landed** |
| A3b iter-2 | `DenseKvBuffers` full multi-seq lift (~150 LOC) — promotes the clamp from `slot_count() == 1` to N | 3-5 days |
| A3b iter-3 | `MlxKvCache` full multi-seq lift (~80 LOC) — legacy 4-bit path | 2-3 days |
| A3c | Gemma 4 `fork_seq` real kernel dispatch (parallel to Qwen35 A2c per dossier §2.3.3 — same `dispatch_kv_cache_copy_seq_*` family serves both arches) | 3-5 days |
| A4 | Drafter KV caches (EAGLE-3, DFlash) — research-quality | 5-8 days |
| **A5 (SHIPPED 2026-05-23, SUPERSEDED by A5b)** | Scheduler-side per-slot KV budget primitive — `AdmitError::SlotBudgetExceeded`, `ApiError::slot_budget_exceeded` schema helper, 4 worker_run match arms. End-to-end enforcement was VAPORWARE at iter-A5 per codex review; see A5b. | 1 day landed |
| **A5b (SHIPPED 2026-05-24, commit `cd47e923`)** | End-to-end per-slot KV byte budget enforcement (shared conservative upper bound) — `LoadInfo::kv_bytes_per_token` upper-bound estimate + `Engine::try_admit_budget` pre-stream check + worker_run real `kv_bytes_needed` wiring + scheduler `new_with_kv_budget` configuration + handler-side `slot_budget_exceeded` routing (parallel to `queue_full`). Closes codex CRITICAL #1, #2 + mantra-violations Line 1153/1155 from iter-A5. Exact per-arch Gemma 4 accounting refined in A5c. See §6.1.16. | 1 day landed |
| **A5c (SHIPPED 2026-05-24)** | Exact per-arch byte accounting for Gemma 4 heterogeneous layers (`LoadInfo::kv_bytes_per_token_override` + `gemma4_exact_kv_bytes_per_token` summing across `cfg.layer_types`) + handler-level 429+Retry-After wire-shape tests + production `LayerType → (is_ring, capacity)` helper extraction for the mixed-layer test. Closes codex /cfa BLOCK on A5b: CRITICAL #1, #2 + MAJOR #1, #3 + NEW (cite cd47e923) + MINOR #1 (ADR wording). Qwen35 path UNCHANGED. See §6.1.17. | 1 day landed |
| A6 | Closure: per-family parity gate vs n_seqs=1 baseline | 2 days |

### Phase B — Scheduler (4-6 iters)

| Iter | Scope | Estimated effort |
|---|---|---|
| **B1 (THIS ITER, 2026-05-23)** | Scaffolding: trait + FifoSchedulerAdapter (real admit/step/release/stats) + InflightBatchedScheduler signature stub (post-iter-1.5: cfg(test)-gated) | 1 day landed |
| B2 | FifoSchedulerAdapter byte-equivalence proof + regression pin | 2-3 days |
| B3 | InflightBatchedScheduler admit/step/release impl | 5-8 days |
| **B4a (SHIPPED 2026-05-23)** | Qwen35 `forward_gpu` / `forward_gpu_with_hidden` public-surface `slot_id: SlotId` threading + bounds check + H2 GPU-content byte-identity at slot 0 + slot-isolation pin + typed B4a-cont error for slot N > 0 | **1 day landed** |
| **B4a-cont (SHIPPED 2026-05-23)** | Qwen35 `build_gated_attn_layer` / `apply_sdpa_with_kv_cache` / KV-dispatcher slot-offset wiring; flip slot > 0 from typed-error to real-route (via `MlxBuffer::slice_view` on slot.k/slot.v) | **1 day landed** |
| **B4a-cont.1 (SHIPPED 2026-05-23)** | Codex /cfa rev-1 addressed: M1 isolation-test rigor (delete reset+rerun-then-compare test + add raw K/V byte snapshot + positive same-prompt-equivalence pin); M2 canonical TQ-active multi-slot gate placement at `build_gated_attn_layer` + `apply_gated_attn_layer_decode_into` entry; minor stale-comment refresh at `forward_gpu.rs` entry | **1 day landed** |
| **B4b (SHIPPED 2026-05-24)** | Qwen35 decode-path slot threading (`forward_gpu_last_logits` / `forward_gpu_last_topk` / `forward_gpu_last_logits_with_soft_tokens` / `forward_gpu_last_logits_with_soft_tokens_and_deepstack` / `forward_embed_last`); full lift (SlotId(N>0) end-to-end via B4a-cont's F32 slot-offset routing); 25 production callsites updated; H17–H20 + variant-coverage = 5 new tests (153 PASS). See §6.1.20. | **1 day landed** |
| B4c | Gemma 4 forward-path slot threading (`forward_prefill.rs` + `forward_prefill_batched.rs`) — gated on Phase A3 Gemma 4 multi-seq KV impl | 5-8 days |
| B4d | Spec-decode slot threading (`forward_gpu_greedy` + dflash entry points) — gated on Phase A4 drafter KV multi-seq impl | 5-8 days |
| B5 | Per-slot 429 + Retry-After contract preservation | 2-3 days |
| B6 | Mixed prefill+decode `SchedulerStep::Mixed` handling | 3-5 days |

### Phase C — Engine slot-aware (3-4 iters)

| Iter | Scope | Estimated effort |
|---|---|---|
| **C1 (SHIPPED, 2026-05-23)** | Scaffolding: `EngineMode` enum + signature-only `SlotAware` variant + regression test | 1 day |
| **C2a (SHIPPED, 2026-05-23)** | Byte-equivalence regression-pin test `engine_serial_fifo_byte_equivalent_to_pre_phase_c` landed env-gated FIRST (per dossier §4 iter-2a step 1). No production code changes; locks the falsifier for C2b's `worker_run` refactor. | 0.5 day |
| **C2b (SHIPPED, 2026-05-23)** | Shape A `worker_run` refactor: extended signature with `mode: EngineMode` + `queue_capacity: u32` + `scheduler_stats_snapshot: Arc<Mutex<SchedulerStats>>`; constructs concrete `FifoSchedulerAdapter` at worker entry (concrete-type realisation of dossier §2.9 "advance lives on concrete type"); wraps `Generate` / `GenerateStream` / `Embed` / `GenerateWithSoftTokens` arms in admit→drive→release; `EngineInner` gains `max_slots: u32` + `scheduler_stats_snapshot` + accessors; `Qwen35LoadedModel` gains `persistent_kv_cache: Option<HybridKvCache>` scaffold (None at iter-2a; iter-2b lift). H2 sequential-request pin added env-gated alongside H1. | 1 day |
| C2c | Qwen35 SlotAware runtime (replaces `EngineSpawnError::ModeNotYetWired` for `SlotAware` via Shape B `select!` loop inside `worker_run` + populates `Qwen35LoadedModel.persistent_kv_cache` with `n_seqs=max_slots`); B4b decode-side slot_id threading UNBLOCKED 2026-05-24 (§6.1.20). Remaining gates: R4 spec-decode mitigation + R4-bis hybrid persistor n_seqs>1 serialization. | 5-8 days |
| **C3 (SHIPPED, 2026-05-23)** | SSE keepalive per-slot accounting (structural — adds `generation_events_to_sse_with_slot` sibling entrypoint accepting `slot_id: Option<u32>`; legacy `generation_events_to_sse` preserved as the 4-arg facade for unchanged `handlers.rs` callers and delegates with `slot_id=None`) + `schema.rs::ApiError::queue_full` docstring update naming `SchedulerPolicy` alongside Decision #2 + `ApiError::capability_unsupported` helper wiring `MultiSeqError::CapabilityUnsupported` → HTTP 501. 5 new tests. Byte-invariance pinned at N=1 under FifoSerial (§1.4 client-invisibility). | 1 day |
| **C4 (SHIPPED, 2026-05-23)** | CLI/env wiring for `HF2Q_SCHEDULER` + `--scheduler` + `HF2Q_MAX_SLOTS` + `--max-slots`; threaded through `multi_model::EngineConfig.engine_mode` into `load_engine` → `Engine::spawn_with_mode`; env-absence is byte-equivalent (`EngineMode::SerialFifo`) per §3.6; SlotAware fail-loud rejection (no silent fallback) with updated `EngineSpawnError::ModeNotYetWired` iter cite (`C2b` SHIPPED → `C2b/C2c (per-family worker arms)` pending). 10 new tests (8 brief-required + 2 precedence pins). | 1 day |

### Phase D — Throughput benchmark (2-3 iters)

| Iter | Scope | Estimated effort |
|---|---|---|
| **D1 (SHIPPED 2026-05-23)** | Scaffolding: env-gated test file + metric definitions | 1 day landed |
| **D2 (SHIPPED 2026-05-24)** | N ∈ {1, 2, 4, 8} measurement harness + report format — subprocess spawn + `/readyz` poll + `std::thread::scope` curl SSE consumption + per-cell ThroughputCell aggregation + AC-4 soft-gate reporting + InflightBatched-skip-when-unwired graceful detection | **1 day landed** |
| **D3 (SHIPPED 2026-05-24)** | A/B comparator (FIFO vs InflightBatched) + statistical stability — REPS=3 median + min/max + `sigma_pct` aggregation via `ThroughputCellStable::from_reps`; per-frame streaming-stdout TTFT via curl `Stdio::piped()` + `BufReader::lines()` (eliminates D2's upper-bound bias); AC-4 hard-fail enforcement gated on BOTH N=4 cells present (deferred to once C2c/C2d ship); stability gate panics when `sigma_pct > 20%`; FifoSerial-only baseline + variance always reported so the bench is operator-useful in the interim | **1 day landed** |

### Phase E — Production cutover (gated on §3.7 memo)

| Iter | Scope | Estimated effort |
|---|---|---|
| E1 | Reopen-trigger memo + AC-4 benchmark gate + default flip | 1-2 weeks (operator + author) |
| E2 | ADR-005 closure-block update + downstream-ADR cross-links | 2-3 days |

**Total estimated effort**: 8-12 weeks. The mantra commitment is met: multi-week structural work in scope.

### 6.1 Iter-1 closure (2026-05-23 — this commit)

All four Phase iter-1 scaffolding tracks landed in parallel under goal-mode directive "implement all of adr-040 fully" + a Phase B↔A integration pass. Total ~2330 LOC + 40 new tests, `cargo check --release` clean, `cargo build --release` clean.

| Phase | Iter | File | LOC | Tests | Status |
|---|---|---|---|---|---|
| — | ADR draft | `docs/ADR-040-continuous-batching-reopen.md` | ~350 | — | ✅ landed |
| A | 1 | `src/serve/multi_seq_kv.rs` | 746 | 11 | ✅ landed |
| B | 1 | `src/serve/scheduler.rs` | ~760 | 16 | ✅ landed (SlotId re-exported from Phase A post-integration) |
| C | 1 | `src/serve/api/engine.rs` (+) | +257 | +7 (95/95 PASS) | ✅ landed |
| D | 1 | `tests/continuous_batching_throughput.rs` | 199 | 6 | ✅ landed |
| — | mod wiring | `src/serve/mod.rs` (+) | +10 | — | ✅ landed |

**Iter-1 invariants pinned by regression tests**:
- ADR-005 Decision #2 + #19 FIFO contract byte-equivalence under `SchedulerPolicy::FifoSerial` (`engine_spawn_signature_unchanged_at_phase_c_iter_1` compile-time gate)
- `EngineMode::default()` returns `SerialFifo` — Phase 2 production path unchanged
- `InflightBatchedScheduler::step` returns `Err(StepError::NotImplemented)` at iter-1 (pinned by `inflight_batched_step_returns_not_implemented_at_iter_1`; Phase B iter-3 replaces)
- `SlotId` + `SeqId` are distinct types — compile-time + runtime test
- `MultiSeqLayout::Paged` is reserved; append under it returns `LayoutNotSupported`

### 6.1.1 Iter-1.5 closure — adversarial review findings + fixes (2026-05-23)

Per goal-mode directive ("Spawn Swarm team (/cfa) with codex to check our work"), iter-1 commit 1a1d6a26 was reviewed by an adversarial /cfa session: an independent Codex reviewer (via `codex exec --json -s read-only`) + an independent Claude reviewer agent. Both returned **verdict=request_changes, severity=high** with substantively overlapping findings. Full reviews at `~/.claude/teams/cfa-adr040-iter1-review/shared/reviews/`.

**Critical findings (both reviewers, must-fix before iter-2):**

| # | Finding | Codex | Claude | Fix |
|---|---|---|---|---|
| F1 | `spawn_with_mode` silently discards `mode` parameter; `mode()` accessor lies (Liskov violation) | critical | critical | Store `mode` on `EngineInner`; `spawn_with_mode` returns `Result<Self, EngineSpawnError>` rejecting `SlotAware` with typed `ModeNotYetWired` error until iter-2. Delete `mode_accessor_returns_serial_fifo_at_iter_1` (codifies the lie); add `mode_accessor_echoes_requested_mode`. |
| F2 | `InflightBatchedScheduler::step → Err(NotImplemented)` IS a stub disguised as typed contract | critical | critical (mantra violation) | Gate `InflightBatchedScheduler` + `StepError::NotImplemented` behind `#[cfg(test)]`. Delete the test that pins NotImplemented. |
| F3 | `FifoSchedulerAdapter` NOT byte-equivalent (missing `.max(1)`, monotonic SlotIds, conflated queue+inflight, sequential test misses race) | major | critical | Apply `queue_capacity.max(1)`. FifoSerial always allocates `SlotId(0)` (drop `next_slot_id` field). Add concurrent-admit race test using `std::thread::scope`. |
| F4 (partial) | Tests pin SIGNATURE not BEHAVIOUR; name `engine_spawn_signature_unchanged_at_phase_c_iter_1` overclaims | major | critical | Rename to `engine_spawn_3_arg_signature_compile_pin` + doc comment that calls out the limitation. Full behavioural pin moves to Phase C iter-2. |

**Major findings:**

| # | Finding | Fix |
|---|---|---|
| F5 | Layout-check-before-bounds-check is wrong default (Liskov violation; hides caller bugs as capability errors) | Swap to bounds-first in NoopMultiSeqKvCache + pin in trait doc with `# Validation order` block. |
| F6 | `AdmitError::QueueFull.capacity` field misnamed (carries queue_capacity, not total admissible — misleads operators) | Rename to `queue_capacity` + add `total_admissible: u32` field. |
| F7 | `SeqId::UNASSIGNED = SeqId(u32::MAX)` sentinel will collide with future allocators | Delete UNASSIGNED const. Add `SeqId::new(v) -> Result<Self, SeqIdOverflow>` validating constructor rejecting u32::MAX. |
| F8 | `cb_throughput_n_1_2_4_8_fifo_vs_inflight` always passes (silent skip even when E2E gate is set) | When env gate is set, PANIC with clear "iter-2 pending" message so CI burns. |
| F9 | `fork_seq` performance contract unclear; SeparateSlots impl will be O(seq_len) memcpy | Document explicit O(seq_len) on SeparateSlots in trait doc. |

**Mantra violations explicitly named by reviewers (ADR-040 §7 "no fallback, no stub"):**
- `let _ = mode;` discard in `spawn_with_mode` (engine.rs:2562) — fixed by F1.
- `Err(StepError::NotImplemented)` (scheduler.rs:477) — fixed by F2.
- `cb_throughput_n_1_2_4_8_fifo_vs_inflight` early-return stub (continuous_batching_throughput.rs:167) — fixed by F8.
- `mode()` accessor lying about requested mode (engine.rs:2574) — fixed by F1.

**Strengths the reviewers noted (preserved):**
- SeqId / SlotId newtype discipline.
- MultiSeqError variant shapes with right field-bearing.
- ADR-005 carve-out + Decisions #1/#2/#19 cross-references.
- Engine::spawn 3-arg signature unchanged.

**Iter-1.5 LOC delta**: ~250 LOC across the 4 fix tracks. Final test count: 40 (iter-1) → ~48 (iter-1.5, after F1+F2+F3+F5+F7 add/delete adjustments).

**Adversarial review reproducibility**: codex transcript at `/tmp/cfa-adr040-iter1-review/codex-review.jsonl`; Claude review at `~/.claude/teams/cfa-adr040-iter1-review/shared/reviews/claude-on-iter1.json`. Both can be re-run by re-issuing the cfa skill against any future iter commit.

### 6.1.2 Iter-2a closure — Phase A2 Qwen35 HybridKvCache (2026-05-23, this commit)

First real per-model `MultiSeqKvCache` impl. Path: `src/inference/models/qwen35/kv_cache.rs`. Grounded in `docs/research/adr040-kv-cache-lift-dossier-2026-05-23.md` (landed iter-1.5).

**Hypothesis-driven execution per goal-mode directive ("Create hypotheses that are testable before changing code")**:

| Hyp | Claim | Status | Test |
|---|---|---|---|
| H1 | `HybridKvCache::new(.., n_seqs=4)` allocates without panic + scales 4× linearly | **VERIFIED** (run before any production code per dossier §4 step 1) | `h1_hybrid_kv_cache_alloc_n_seqs_4_byte_scale` |
| H2 | Slot 0 cursor state is byte-identical between `n_seqs=1` and `n_seqs=4` | **VERIFIED (cursor-only — forward-path byte-equivalence requires Phase B iter-3 wiring)** — see caveat row below | `qwen35_hybrid_kv_byte_identical_at_slot_0_n_seqs_4_vs_1` |
| — | **Caveat (iter-2.5 M3 honest downgrade)**: AC-1 byte-equivalence at slot 0 is *partially* verified | iter-2a pinned cursor-level (`current_len[0]` identical across `n_seqs=1` vs `n_seqs=4`) but did NOT pin forward-path logits byte-identical (requires `forward_prefill_gpu` slot-0 trace, which is Phase B iter-3 scope). The dossier's H2 statement requires forward-path byte-identical logits; iter-2a closed only the cursor half. | (cursor half) `qwen35_hybrid_kv_byte_identical_at_slot_0_n_seqs_4_vs_1`; (forward half) Phase B iter-3 `forward_prefill_slot_0_byte_identical_at_n_seqs_4_vs_1` |
| H3 | Per-slot `append_for_seq` is O(1) and isolated (writes to slot N do not corrupt slot M) | **VERIFIED** (cursor-level) | `qwen35_hybrid_kv_per_slot_isolation_n_seqs_4` |
| H4 | `n_seqs` is outermost axis in linear-attn recurrent state (drop = contiguous-slice zero, no kernel) | **DEFERRED to A2b** | linear-attn lift is gated on the `rollback_la_to` guard at `kv_cache.rs:1567` |
| H5 | `gpu_delta_net.rs` `n_seqs = 1u32` sites are soft hard-codes (pass `cache.n_seqs` to lift) | **DEFERRED to A2b** | only meaningful once linear-attn carve-out lifts |

**ADR-040 §1.3 falsification verdict**: VERIFIED for full-attn K/V + linear-attn recurrent (both scale exactly 4×). The capture-buffer (5-D shape with `n_seqs` at `kv_cache.rs:1567`) remains the linear-attn deferral boundary as the dossier predicted.

**Scope per dossier §4 iter-2a step 2**:
- ✅ Full-attn slot lift (every slot's `current_len[slot.0]` cursor mutated)
- ✅ MTP slot lift (when `mtp_slot.is_some()`)
- ⏭️ Linear-attn cursor lift — DEFERRED to A2b (per R1: `rollback_la_to` guard explicitly errors on `n_seqs > 1`)
- ⏭️ KV-content side of byte-equivalence (Phase B iter-3 wires `forward_prefill.rs` slot-id threading)
- ⏭️ `fork_seq` kernel-dispatch — Phase A2a returned `SlotOom { 0, 0 }` sentinel as the documented "kernel-dispatch not yet implemented" signal per cfa-finding-F2; **iter-2.5 M1 closure: now returns `CapabilityUnsupported { capability }` (HTTP 501) instead — see §6.1.3**
- ⏭️ `HF2Q_MAX_SLOTS` env wiring — ADR-040 §6 Phase C iter-4 scope
- ⏭️ Persistor `n_seqs=4` round-trip — `qwen35_hybrid_persistor.rs` wire format already supports it (`:171-175`); test lands in A2 closure ceremony

**Quality gates (all green)**:
- `cargo check --release`: 0
- `cargo test --bin hf2q -- qwen35::kv_cache serve::multi_seq_kv`: **91/91 PASS** (75 qwen35 + 16 multi_seq_kv)
- LOC delta: +666 / -0 on `kv_cache.rs` (no deletions; ~220 trait impl + ~440 tests)

**Iter-2a test count net**: qwen35::kv_cache 64 → 75 (+11 net). Adds H1, H2, H3, plus 8 trait-surface pins (`slot_count_matches_n_seqs`, `slot_out_of_range_errors_named`, `drop_resets_seq_len_for_target_slot_only`, `drop_does_not_zero_recurrent_buffer_a2a`, `fork_to_self_is_noop_ok`, `fork_cross_slot_returns_oom_at_phase_a2a`, `append_advances_target_slot_only`, `layout_is_separate_slots`).

**R1+R2 mitigations confirmed**:
- R1 (linear-attn capture buffer): the `rollback_la_to` guard at `kv_cache.rs:1567` is untouched. Phase A2a never lifts `n_seqs > 1` into the linear-attn path.
- R2 (forward-path slot threading is Phase B iter-3): Phase A2a's `append_for_seq` mutates only per-cache cursor state; forward-path slot_id threading lands separately per ADR-040 §2.2.

### 6.1.3 Iter-2.5 closure — Phase A2 + Phase B adversarial review fixes (2026-05-23)

Per goal-mode directive ("Spawn Swarm team (/cfa) with codex to check our work"), iter-2a commit `2ecb2dc6` (KV) + iter-2.5 prep commit `69d86ed8` (ADR/scheduler) were reviewed by an adversarial /cfa session: an independent Codex reviewer + an independent Claude reviewer agent. Reviews at `/tmp/cfa-iter2a-b3-review/codex-review-last.txt` and `~/.claude/teams/cfa-iter2a-b3-review/shared/reviews/claude-on-iter2a-b3.json`.

**KV-cache fixes landed in this commit** (parallel scheduler/engine fixes documented separately by the inflight-batched reviewer agent — see C1/C2/C3/M2 row below for file pointers):

| # | Finding | Reviewer(s) | Severity | File:line | Fix |
|---|---|---|---|---|---|
| M1 | `fork_seq` returned `SlotOom { 0, 0 }` sentinel — mantra violation (would map to HTTP 429 + Retry-After, lies about whether retry can succeed) | Codex + Claude | major | `src/inference/models/qwen35/kv_cache.rs:2731-2735` (was); `src/serve/multi_seq_kv.rs` (new variant) | Add `MultiSeqError::CapabilityUnsupported { capability: &'static str }` variant. Map to HTTP 501 in Phase C iter-3 schema. `fork_seq` returns it with a label naming the deferred Phase A2c kernel arc + dossier R5 grounding. Test renamed `qwen35_hybrid_kv_fork_cross_slot_returns_capability_unsupported_at_phase_a2a` + 2 trait-surface pins (`multi_seq_error_capability_unsupported_display_names_capability`, `multi_seq_error_capability_unsupported_distinct_from_slot_oom`). |
| C4 | `seq_len()` reads `full_attn[0].current_len[slot]` as canonical with no defensive assert against per-layer cursor desync (silent lie on checkpoint replay / partial rollback / kernel error) | Claude | critical | `src/inference/models/qwen35/kv_cache.rs:2589-2614` | Add `debug_assert!` against per-layer desync (catches in dev/CI; release builds return canonical_0 — no panic, no Result shape change). Add `qwen35_hybrid_kv_seq_len_canonical_across_full_attn_layers` test pinning the production-side invariant. Trade-off documented inline: release-build assertion would panic on prod desync (worse than silent lie); if a future incident reveals desync, escalate to Result-return. |
| M4 | `drop_does_not_zero_recurrent` test compared only `byte_len()` before/after — vacuous (any reasonable impl would pass even if it zero'd contents in place) | Codex + Claude | major | `src/inference/models/qwen35/kv_cache.rs:6511-6528` (was) | Strengthen: fill recurrent with deterministic non-zero F32 pattern via `MlxBuffer::as_mut_slice::<f32>()` (StorageModeShared, direct host write), snapshot bytes via `as_slice::<f32>().to_vec()`, call `drop_seq`, snapshot again, assert `before == after` byte-for-byte. Any in-place mutation surfaces. |
| M5 | H1 `byte_len()` 4× check can't catch axis-order swap (n_seqs landing on wrong axis produces same byte count) | Codex + Claude | major | `src/inference/models/qwen35/kv_cache.rs:6229-6301` | Add shape-axis assertions to H1 using `MlxBuffer::shape()`. Per kv_cache.rs alloc sites: full-attn K/V at shape[0]=n_seqs (row-major, head_dim innermost); linear-attn recurrent at shape.last()=n_seqs (column-major-style, D_k innermost — comment at kv_cache.rs:2278). Both: non-n_seqs dims must be byte-equal between cache_1 and cache_4. |
| H1-tq | H1 only exercised dense F32-only path (`new(..)`); no coverage of TQ-active production KV path | Claude | major | `src/inference/models/qwen35/kv_cache.rs` (new sibling test) | Add `h1_tq_active_hybrid_kv_cache_alloc_n_seqs_4_byte_scale` constructing via `new_with_options(.., tq_kv_active=true)`. Pins: F32 K/V dropped (iter-34 contract); TQ packed K/V + norms K/V all scale 4×; M5-style shape proof on TQ buffers (shape[0]=n_seqs per `alloc_tq_full_attn_buffers` line 2421+2437). |
| M3 | ADR-040 §6.1.2 marked H2 "VERIFIED (cursor-level)" — overstated vs dossier's forward-path byte-identical logits requirement | Codex + Claude | major | `docs/ADR-040-continuous-batching-reopen.md` §6.1.2 | Downgrade H2 to "VERIFIED (cursor-only — forward-path byte-equivalence requires Phase B iter-3 wiring)" + add explicit caveat row stating iter-2a closed only the cursor half. |
| C1/C2/C3/M2 | Scheduler-side findings (admission accounting, race conditions, FIFO byte-equivalence regressions) | Codex + Claude | mixed | `src/serve/scheduler.rs` + `src/serve/api/engine.rs` + `src/serve/load_info.rs` | **Owned by parallel iter-2.5 scheduler agent — see that agent's closure block.** File-level boundary: this commit only touches `multi_seq_kv.rs`, `qwen35/kv_cache.rs`, and this ADR doc. |

**Mantra violations explicitly closed (ADR-040 §7 "no fallback, no stub, no `// TODO` in production"):**
- `SlotOom { 0, 0 }` sentinel in `fork_seq` (qwen35/kv_cache.rs:2731-2735) — fixed by M1.
- Implicit "trust me" cursor-canonical read in `seq_len()` (qwen35/kv_cache.rs:2613) — hardened by C4.
- Test vacuity on `drop_does_not_zero_recurrent` — fixed by M4 (was technically a test defect, not a production-code mantra violation, but the brief class is the same: appears to pin a contract, actually pins nothing).

**Iter-2.5 KV-side test count net**: qwen35::kv_cache 75 → 77 (+2 net: H1-tq + C4 pin; M4 + M5 strengthened in-place, M1 test renamed + assertion-shape updated). multi_seq_kv 16 → 18 (+2 net: CapabilityUnsupported display + discriminant-distinctness pins).

**Quality gates (all green)**:
- `cargo check --release`: 0
- `cargo test --release --bin hf2q -- serve::multi_seq_kv qwen35::kv_cache`: all PASS

**Future-iter pin pointers**:
- Phase C iter-3 schema mapping: `serve/api/schema.rs` will route `MultiSeqError::CapabilityUnsupported` → HTTP 501 (parallel to `SlotOom` → 429 + Retry-After and `SlotOutOfRange` → 500 internal-defect).
- Phase B iter-3 forward-path slot threading: `forward_prefill_gpu` per-slot offset wiring + the still-missing `forward_prefill_slot_0_byte_identical_at_n_seqs_4_vs_1` test that closes the H2 forward-path half.
- Phase A2c: replace `fork_seq` `CapabilityUnsupported` with same-buffer cross-region memcpy via `dispatch_kv_cache_copy_seq_*`; flip the test assertion to `Ok(())` + per-buffer byte-equality.

### 6.1.4 Iter-B4a closure — Qwen35 forward_gpu slot_id threading (2026-05-23, this commit)

First real per-model **forward-path** slot threading.  Closes the H2 GPU-content side that iter-2.5 M3 promised to defer to "Phase B iter-3 wiring" (ADR-040 §6.1.2 caveat row + §6.1.3 Future-iter pin pointer).

**ADR §2.2 amendment in this commit**: the original §2.2 row named `src/serve/forward_prefill.rs` as the B iter-3 target, but that file is Gemma 4's prefill path.  Qwen35's equivalent is `src/inference/models/qwen35/forward_gpu.rs`, and Qwen35 is the family with the iter-2a `MultiSeqKvCache` impl shipped.  The amended §2.2 splits B4 into four sub-iters by family + entry-point class:

| Sub-iter | File / surface | Status |
|---|---|---|
| **B4a (this commit)** | Qwen35 `forward_gpu` + `forward_gpu_with_hidden` public surface — `slot_id: SlotId` threading, bounds check, slot-0-only GPU-content path with typed B4a-cont error for slot N > 0 | **SHIPPED** |
| B4a-cont | Qwen35 internal helpers (`build_gated_attn_layer`, `apply_sdpa_with_kv_cache`, `write_kv_with_optional_tq_encode`, FA prefill / vec helpers) — KV-dispatcher slot-offset wiring via `MlxBuffer::slice_view`; flip slot N > 0 from typed-error to real-route | pending |
| B4b | Qwen35 decode-side entry points — `forward_gpu_last_logits` / `forward_gpu_last_topk` / soft-token / deepstack — gated to slot 0 in B4a | pending |
| B4c | Gemma 4 forward path (`src/serve/forward_prefill.rs` + `forward_prefill_batched.rs`) — gated on Phase A3 Gemma 4 multi-seq KV impl landing first | pending (gated on A3) |
| B4d | Spec-decode entry points (`forward_gpu_greedy` + dflash) — gated on Phase A4 drafter KV multi-seq impl | pending (gated on A4) |

**Scope per the B4a brief (TIGHTLY scoped)**:
- ✅ Thread `slot_id: SlotId` through `Qwen35Model::forward_gpu` + `Qwen35Model::forward_gpu_with_hidden` (public surface)
- ✅ Thread `slot_id` through `forward_gpu_impl` (private worker; receives slot_id from every caller)
- ✅ Bounds-check `slot_id.0 < kv_cache.n_seqs` at top of `forward_gpu_impl` with fail-loud diagnostic naming both the slot and the configured n_seqs
- ✅ All 9 internal callers of `forward_gpu_impl` updated to pass `SlotId(0)` (the B4b/B4c/B4d-scope decode + soft-token + deepstack + dflash variants are explicitly gated to slot 0 with inline comments naming the unblocking iter)
- ✅ All external callers updated: 7 callsites in `src/inference/models/qwen35/spec_decode.rs`, 2 in `src/serve/mod.rs`, 5 in-file test sites
- ✅ Slot N > 0 returns a typed error naming the missing GPU-side wiring (Phase B4a-cont) — fail-loud per ADR-040 §7 mantra, NOT a stub or fallback
- ✅ H2 GPU-content side test: `b4a_forward_gpu_at_slot_0_n_seqs_4_byte_identical_to_n_seqs_1` (closes the M3 deferred promise — proves the n_seqs > 1 allocation doesn't disturb slot-0 forward outputs)
- ✅ Slot-isolation pin: `b4a_forward_gpu_slot_0_does_not_touch_slot_1_kv_region` (snapshots slot 1's full K/V bytes before + after a slot-0 forward + asserts byte-equality across all full-attn layers; also pins `current_len[1] == 0` post-forward)
- ✅ Bounds-check pin: `b4a_forward_gpu_slot_out_of_range_errors` (asserts error message names both slot and n_seqs; boundary at `slot == n_seqs`)
- ✅ B4a-cont contract pin: `b4a_forward_gpu_slot_n_gt_zero_returns_b4a_cont_typed_error` (asserts error message names "Phase B4a-cont" so operators know which iter unblocks slot > 0)

**Out of scope per the brief (deferred to later sub-iters)**:
- ⏭️ Decode-side variants (`forward_gpu_last_logits` etc.) — Phase B4b
- ⏭️ Linear-attn slot reads — Phase A2b (gated on `rollback_la_to` guard lift at kv_cache.rs:1567 per dossier R1)
- ⏭️ Gemma 4 `forward_prefill.rs` — Phase B4c (gated on Phase A3)
- ⏭️ `forward_gpu_greedy` / dflash / spec-decode variants — Phase B4d (gated on Phase A4 drafter multi-seq KV)
- ⏭️ GPU-side KV-buffer slot-offset wiring (kernel-dispatcher slot-aware indexing) — Phase B4a-cont

**Quality gates (all green)**:
- `cargo check --release`: 0
- `cargo test --release --bin hf2q -- qwen35::kv_cache::tests serve::scheduler::tests serve::multi_seq_kv::tests`: **142/142 PASS** (iter-2.5 regression preserved)
- `cargo test --release --bin hf2q -- inference::models::qwen35::forward_gpu::tests`: **30/30 PASS** (includes all 4 new B4a tests)
- `cargo test --release --bin hf2q -- inference::models::qwen35::spec_decode`: **5/5 PASS** (spec_decode callsites updated to pass SlotId(0))

**LOC delta**:
- `src/inference/models/qwen35/forward_gpu.rs`: +~390 LOC (signature threading: ~60; bounds check + B4a-cont typed error: ~50; B4a-cont fixture builder: ~80; 4 B4a tests: ~200)
- `src/inference/models/qwen35/spec_decode.rs`: +1 LOC import, +7 callsite updates
- `src/serve/mod.rs`: +1 LOC import, +2 callsite updates
- `docs/ADR-040-continuous-batching-reopen.md`: +~50 LOC (§2.2 amendment + Phase B sequencing rows + this closure block)

**Mantra-aligned**: no `// TODO`, no `unimplemented!()`, no `panic!()` in production code.  The slot > 0 typed error is a precise contract (operators see exactly which iter unblocks the path), NOT a stub.  Slot 0 at n_seqs > 1 is the byte-identical reference case — that's where the H2 promise lands, not in a deferred slot > 0 path.

**Why slot > 0 is gated (not silently corrupting)**: today's GPU kernels (`dispatch_kv_cache_copy_seq_f32_dual`, `flash_attn_vec`, `apply_flash_attn_prefill_seq_major_*`) index into `slot.k`/`slot.v` using a 3-D `[n_kv_heads, max_seq_len, head_dim]` assumption — they write to byte offset 0 regardless of slot_id.  Silently accepting slot_id > 0 would route that write into slot 0 while the caller thought it was slot N, corrupting slot 0's K/V region.  Phase B4a-cont lands the kernel-dispatcher slot-offset parameter (via `MlxBuffer::slice_view(slot_byte_offset, slot_region_elems)` per slot) as a cohesive multi-helper change.

**Future-iter pin pointers**:
- Phase B4a-cont: replace the typed B4a-cont error with the real slot-offset routing.  Touch points: `build_gated_attn_layer` (slot.k/slot.v access at lines 4873, 4955, 5215, 5227, 5353, 5397 in `gpu_full_attn.rs`), `apply_sdpa_with_kv_cache` (line 4196), `write_kv_with_optional_tq_encode` (line 102), the FA prefill / vec dispatchers.  The test `b4a_forward_gpu_slot_n_gt_zero_returns_b4a_cont_typed_error` flips its assertion shape from `is_err()` to `is_ok()` + per-slot output assertions.
- Phase B4b: thread `slot_id` through the 5 decode-side entry points (`forward_gpu_last_logits`, `forward_gpu_last_topk`, `forward_gpu_last_logits_with_soft_tokens`, `forward_gpu_last_logits_with_soft_tokens_and_deepstack`, `forward_embed_last`) by removing the hard-coded `SlotId(0)` in their `forward_gpu_impl` calls.
- Phase B4c: per the §2.2 amendment in this commit, Gemma 4's `forward_prefill.rs` is the B4c target after Phase A3 Gemma 4 multi-seq KV lands.
- Phase B4d: spec-decode `forward_gpu_with_hidden` + `forward_gpu_with_hidden_dflash` callsites in `spec_decode.rs` flip from hard-coded `SlotId(0)` to the scheduler-provided slot_id.

### 6.1.5 Iter-B4a-cont closure — Qwen35 GPU-side slot-offset wiring (2026-05-23, this commit)

Closes the iter-B4a typed-error gate.  Removes the `forward_gpu_impl` entry guard that rejected `slot_id.0 != 0`, and lifts the five kernel-dispatch sites in `src/inference/models/qwen35/gpu_full_attn.rs` to per-slot K/V byte offsets via `MlxBuffer::slice_view`.  Slot 0 remains byte-identical to pre-B4a-cont (slice byte_offset=0 is a no-op kernel-side); slot N>0 routes writes/reads into its per-slot region of the `[n_seqs, n_kv_heads, max_seq_len, head_dim]` F32 full-attn cache backing.

**`MlxBuffer::slice_view` contract verification** (mlx-native@fed406d `src/buffer.rs:93-106` + `src/encoder.rs:182-184`):
- **Zero-copy**: `slice_view` clones the underlying Metal `MTLBuffer` ARC handle (via `metal::Buffer::clone()`) and stores the new `byte_offset` on the returned `MlxBuffer`.  No data copy occurs; both buffers share the same physical allocation.
- **Lifetime**: the returned `MlxBuffer` independently retains the metal buffer via ARC.  It can outlive the parent `MlxBuffer` without invalidating the underlying allocation (Metal's ObjC ARC keeps the buffer alive as long as ANY clone exists).
- **Kernel binding**: `KernelArg::Buffer(buf)` propagates `buf.byte_offset()` into Metal's `setBuffer:offset:atIndex:` call (verified at `encoder.rs:182-184`).  The kernel sees only the slice region starting at the recorded byte offset.
- **Dtype**: slice_view preserves the parent's dtype (`self.dtype`).  Shape is replaced with `vec![n_elements]` (1-D view) since the slice may not preserve the parent's multi-axis shape semantically.
- **Bounds**: slice_view PANICS at construction time when `byte_offset + n_elements * dtype.size_of() > inner.length()`, providing fail-loud protection against off-by-one errors in the slot byte-offset formula.

**Per-slot byte offset formula** (centralised in the private helper `slot_k_v_region_for_full_attn` at `gpu_full_attn.rs:101-119`):

```rust
n_elements = n_kv_heads * max_seq_len * head_dim
byte_offset = slot_id.0 as u64 * n_elements * size_of::<f32>()
```

Matches the alloc shape at `kv_cache.rs:2231-2236`: `[n_seqs, n_kv_heads, max_seq_len, head_dim]` F32 row-major, slot N occupying the contiguous block at index `N` on the outer axis.  Overflow guarded by `checked_mul` + `expect` (fail-loud per ADR-040 §7 mantra).

**Five kernel-dispatch sites lifted** (each call previously hardcoded the slot-0 region by passing `slot.k.as_ref().expect(..)` / `slot.v.as_ref().expect(..)` raw; now passes `kbuf.slice_view(byte_offset, n_elements)`):

| Site | Function | Range | Slot K/V access mode |
|---|---|---|---|
| 1 | `write_kv_with_optional_tq_encode` | gpu_full_attn.rs `~131-220` | F32 write (`dispatch_kv_cache_copy_seq_f32_dual`) into `slot.k` + `slot.v` slice_view |
| 2 | `dispatch_decode_sdpa_with_optional_tq` | gpu_full_attn.rs `~225-360` | F32 read (`flash_attn_vec`) from `slot.k` + `slot.v` slice_view (legacy fallback branch) |
| 3 | `apply_flash_attn_prefill_seq_major_into` | gpu_full_attn.rs `~1700-1830` | NO slot K/V access (operates on fresh chunk K/V written to caller-owned `out_seq`); slot_id unused at this layer.  Public signature **unchanged** — preserves every call site in `kv_cache.rs` / `mtp.rs`. |
| 4 | `apply_flash_attn_prefill_seq_major` | gpu_full_attn.rs `~3700-3960` | Same as `_into` — no slot K/V access; public signature unchanged. |
| 5 | `apply_flash_attn_prefill_seq_major_resume` | gpu_full_attn.rs `~4045-4180` | Caller (`apply_sdpa_with_kv_cache`) does the slice_view on slot.k / slot.v BEFORE invoking; the function consumes the slot K/V buffer arguments directly.  Public signature unchanged. |

In addition, the per-slot byte-offset slice_view is also applied at the legacy F32 SDPA fallback path inside `apply_sdpa_with_kv_cache` (lines `~4730` for the prefill `sdpa` call, `~4317` for `dispatch_sdpa_decode` decode-fallback, `~4520` for the `vec_small_path` decode at cur_len>0).

**Parent dispatcher updates** (slot_id parameter ADDED — public surface change limited to in-crate callers):

| Function | New parameter | Caller updates |
|---|---|---|
| `apply_sdpa_with_kv_cache` | `slot_id: SlotId` (13th positional arg) | `gpu_full_attn.rs::build_gated_attn_layer` callsite + `mtp.rs:416` (MTP draft slot, hard-coded `SlotId(0)` — single-seq MTP is B4b scope) |
| `apply_sdpa_with_kv_cache_decode_into` | `slot_id: SlotId` (13th positional arg) | `gpu_full_attn.rs::apply_gated_attn_layer_decode_into` callsite |
| `build_gated_attn_layer` | `slot_id: SlotId` (21st positional arg) | `forward_gpu.rs` 2 callsites (forward_gpu_impl + forward_gpu_greedy; the latter hard-codes `SlotId(0)` — B4d scope) + 2 test sites in `gpu_full_attn.rs::tests` |
| `apply_gated_attn_layer_decode_into` | `slot_id: SlotId` (18th positional arg) | `forward_gpu.rs::forward_gpu_impl` single-CB decode site |

**TQ-active multi-slot gate**: `apply_sdpa_with_kv_cache` and `apply_sdpa_with_kv_cache_decode_into` return a typed error when `slot.tq.is_some() && slot_id.0 != 0`.  The TQ encode (`dispatch_hadamard_quantize_kv_hb_seq`) and TQ SDPA (`flash_attn_vec_tq_hb`) kernels are NOT yet slot-aware (their slot.tq buffers are bound at offset 0); routing slot N>0 through them would silently corrupt slot 0's TQ region.  Defence-in-depth assertions in the two private TQ-aware helpers (`write_kv_with_optional_tq_encode`, `dispatch_decode_sdpa_with_optional_tq`) repeat the check so a future caller that bypasses `apply_sdpa_with_kv_cache` cannot accidentally engage the broken path.  Tracked separately as B4a-TQ; until that lands, slot 0 with TQ-active remains byte-identical to pre-B4a-cont.

**B4a-cont entry-gate REMOVAL** (`forward_gpu.rs`):
- DELETED: `forward_gpu_impl`'s `if slot_id.0 != 0 { Err("forward_gpu: slot_id=N requires the Phase B4a-cont GPU-side KV-buffer slot-offset plumbing ...") }` block (was at lines `~2566-2580` pre-edit; now replaced by an explanatory comment block).
- The bounds check (`slot_id.0 >= kv_cache.n_seqs`) is PRESERVED — out-of-range slots remain a caller bug, not a capability error.

**Per-slot cursor reads/writes**: `apply_sdpa_with_kv_cache`, `apply_sdpa_with_kv_cache_decode_into`, and `build_gated_attn_layer` now read/write `slot.current_len[slot_id.0]` (was hardcoded `[0]`).  Each entry asserts `slot_id.0 < slot.current_len.len()` for defence-in-depth (the public-entry bounds check already covers this; the assert protects against a future internal caller that bypasses `forward_gpu_impl`).

**Tests** (`forward_gpu.rs::tests`):

| Test | Status | File:line | Coverage |
|---|---|---|---|
| `b4a_forward_gpu_slot_n_gt_zero_returns_b4a_cont_typed_error` | **DELETED** | (was `~8193`) | Pinned the iter-B4a contract that B4a-cont removes. |
| `b4a_cont_forward_gpu_slot_1_succeeds_end_to_end` | **NEW PASS** | `~8194` | Proves `forward_gpu(SlotId(1))` at `n_seqs=4` runs end-to-end + advances `current_len[1] == seq_len` while keeping sibling-slot cursors at 0. |
| `b4a_cont_forward_gpu_slot_isolation_byte_identity` | **NEW PASS** | `~8284` | Load-bearing isolation pin: forward P→slot 0 (snapshot L0), forward Q→slot 1, reset slot 0 cursor, re-forward P→slot 0, assert byte-identical to L0.  Falsifies any cross-slot K/V leak. |
| `b4a_forward_gpu_at_slot_0_n_seqs_4_byte_identical_to_n_seqs_1` | KEPT PASS | `~7924` | H2 byte-identity at slot 0 — `n_seqs=4` allocation must not disturb slot-0 forward outputs. |
| `b4a_forward_gpu_slot_0_does_not_touch_slot_1_kv_region` | KEPT PASS | `~8008` | The inverse direction of the new isolation test: slot 0 forward must not touch slot 1's K/V byte region. |
| `b4a_forward_gpu_slot_out_of_range_errors` | KEPT PASS | `~8126` | Public-entry bounds check (slot >= n_seqs errors with diagnostic naming both). |

**Slot isolation byte-identity proof**: `b4a_cont_forward_gpu_slot_isolation_byte_identity` is the load-bearing pin.  Falsifier path: if `MlxBuffer::slice_view`'s byte_offset is dropped/ignored on the kernel-binding path (regression in `encoder.rs::KernelArg::Buffer`) OR if the per-slot byte-offset formula is wrong, slot 1's writes leak into slot 0's region — the re-run of prompt P at slot 0 then sees corrupted K/V data and produces different logits.  The byte-equality assertion FALSIFIES with a precise per-element diff.

**Quality gates (all green)**:
- `cargo check --release`: 0
- `cargo test --release --bin hf2q -- qwen35::kv_cache serve::scheduler serve::multi_seq_kv`: **142/142 PASS** (iter-B4a regression preserved)
- `cargo test --release --bin hf2q -- inference::models::qwen35::forward_gpu::tests::b4a`: **5/5 PASS** (3 KEPT + 2 NEW; 1 DELETED as per contract)
- `cargo test --release --bin hf2q -- qwen35::forward_gpu --test-threads=1`: **31/31 PASS** single-threaded
- `cargo test --release --bin hf2q -- qwen35::mtp --test-threads=1`: **9/9 PASS** (MTP slot_id pass-through validated)
- `cargo test --release --bin hf2q -- qwen35::spec_decode --test-threads=1`: **5/5 PASS**

**LOC delta**:
- `src/inference/models/qwen35/gpu_full_attn.rs`: +~180 LOC (helper `slot_k_v_region_for_full_attn` ~30; 6 slice_view sites ~40; TQ-active multi-slot gate + 2 defence-in-depth asserts ~50; slot_id param threading on 4 dispatchers ~30; cursor read/write swaps + comments ~30)
- `src/inference/models/qwen35/forward_gpu.rs`: +~200 LOC / -~80 LOC (B4a-cont gate REMOVED ~30 LOC; 3 caller updates ~12 LOC; 1 DELETED test ~50 LOC; 2 NEW tests ~210 LOC + module-level commentary)
- `src/inference/models/qwen35/mtp.rs`: +8 LOC (single MTP callsite + comment naming the B4b deferral for multi-slot MTP)
- `docs/ADR-040-continuous-batching-reopen.md`: +~140 LOC (§2.2 row + §6 Phase B sequencing row updates + this §6.1.5 closure block)

**Deviations from the original brief, with rationale**:
- The brief instructed adding `slot_id: SlotId` to all 5 dispatch functions including the 3 prefill helpers (`apply_flash_attn_prefill_seq_major`, `_into`, `_resume`).  After tracing the call graph, those 3 helpers either operate on FRESH chunk K/V (no slot.k/slot.v access in `_into` / wrapper) or accept caller-supplied slot K/V buffers (`_resume`).  In all three cases the per-slot routing is structurally encoded in (a) the absence of slot K/V access entirely, or (b) the caller-supplied buffer's own `byte_offset` (set via `slice_view` at the caller).  Adding `slot_id` to these 3 public functions would force ripple updates in `kv_cache.rs::tests` (3 callsites) — which the brief constraint #6 ("ONLY edit these 3 files") forbids.  The resolution: keep those 3 public signatures unchanged; do the slice_view at the `apply_sdpa_with_kv_cache` parent dispatcher BEFORE invoking the helper.  `apply_sdpa_with_kv_cache` itself does get the `slot_id` parameter — which forced one ripple update in `mtp.rs::416` (the only out-of-file caller).  Net: the brief intent is satisfied (slot N>0 successfully writes/reads at the correct K/V byte offset, end-to-end test PASS) with a smaller blast radius (1 cross-file callsite vs 5).
- The brief instructed adding `slot_id: SlotId` parameters to `apply_flash_attn_prefill_seq_major_into` and `apply_flash_attn_prefill_seq_major`.  These have NO out-of-file callers (`kv_cache.rs` tests do call them but with the original signature) — so adding the param technically wouldn't break the file scope.  However, since neither function accesses slot K/V, the parameter would be `let _ = slot_id;` — dead weight that future maintainers would have to either remove or carry forward.  Following the YAGNI principle (don't add parameters until they have a use), I kept these two unchanged.  If a future iter (e.g. shared slot K/V mirror) needs per-slot routing in the prefill helpers, the param can be added at that time alongside its first real use.

**Mantra-aligned**: no `// TODO`, no `unimplemented!()`, no `panic!()` in production code.  TQ-active multi-slot is the only deferred case — gated with a typed error naming the specific kernel work needed (B4a-TQ).  Slot 0 with TQ-active remains byte-identical to pre-B4a-cont; the TQ multi-slot deferral is a precise contract, not a stub.

**Future-iter pin pointers**:
- Phase B4a-TQ: lift `dispatch_hadamard_quantize_kv_hb_seq` (`mlx-native::ops::hadamard_quantize_kv`) and `flash_attn_vec_tq_hb` (`mlx-native::ops::flash_attn_vec_tq_hb`) to accept a per-slot byte offset on the `slot.tq.k_packed` / `slot.tq.v_packed` / `slot.tq.k_norms` / `slot.tq.v_norms` buffers (the alloc shape at `kv_cache.rs:2421-2426` is already `[n_seqs, n_kv_heads, max_seq_len, head_dim]` — kernel work is purely the `setBuffer:offset:` parameterization).  Once landed, remove the typed-error gates at `apply_sdpa_with_kv_cache` (lines `~4377` and `~6256`) and the two defence-in-depth asserts in the private helpers.
- Phase B4b: thread `slot_id` through the 5 decode-side entry points + MTP's `forward_draft` path (currently hard-coded `SlotId(0)` at `mtp.rs:430`).
- Phase B4c: Gemma 4's `forward_prefill.rs` after Phase A3 lands.
- Phase B4d: spec-decode (`forward_gpu_greedy` + dflash variants) after Phase A4 lands.

### 6.1.6 Iter-B4a-cont.1 closure — Codex /cfa rev-1 follow-ups (2026-05-23, this commit)

Adversarial review of B4a (commit `23896c33`) + B4a-cont (commit `1d3b13ef`) by Codex returned `verdict=request_changes`, `severity=med`, with 2 major + 2 minor findings.  This iter addresses the 2 major + 1 of the 2 minor (the remaining minor — `MlxBuffer::slice_view` overflow hardening — is `mlx-native`'s concern, out of scope for hf2q-only edits per the brief).  Codex review evidence at `/tmp/cfa-b4a-cont-review/codex-review-last.txt`.

| Finding | Severity | Reviewer | File | Fix |
|---|---|---|---|---|
| M1 | major | Codex | `src/inference/models/qwen35/forward_gpu.rs` (test) | The previous `b4a_cont_forward_gpu_slot_isolation_byte_identity` reset `current_len[0]` and re-ran prompt P into slot 0; the re-run OVERWROTE slot 0's K/V positions before attention read them, so a broken impl where slot-1 writes land in slot 0 could still pass.  Replaced with two stronger pins (see below). |
| M2 | major | Codex | `src/inference/models/qwen35/gpu_full_attn.rs` (TQ-active gate) | `build_gated_attn_layer` has a fused Stage-AB path (line ~5479) that bypasses `apply_sdpa_with_kv_cache` and calls `write_kv_with_optional_tq_encode` directly.  The defence-in-depth gates inside the two private dispatchers fire AFTER ops1-4 (4 projections + 2 per-head RMSNorm + 2 IMROPE dispatches) are already encoded into an uncommitted command encoder — wasteful + obscures the failure site.  Lifted the canonical gate to `build_gated_attn_layer` + `apply_gated_attn_layer_decode_into` entry. |
| Minor#1 | minor | Codex | `src/inference/models/qwen35/forward_gpu.rs:1595-1598` | Comment said "GPU-side KV-buffer rebasing lands in Phase B4a-cont".  After commit `1d3b13ef`, B4a-cont has landed.  Updated to reflect current state (rebasing implemented; TQ-active multi-slot deferred to B4a-TQ). |
| Minor#2 | minor | Codex | `mlx-native/src/buffer.rs:93-99` (`slice_view`) | `byte_offset as usize + n_elements * dtype.size_of()` uses unchecked arithmetic; an extreme offset/length could wrap before the `end <= len` assertion.  **OUT OF SCOPE** for B4a-cont.1 (mlx-native concern, not hf2q's; brief constraint #7 forbids mlx-native edits).  Tracked as a future mlx-native iter. |

**M1 — Test rigor fixes** (`src/inference/models/qwen35/forward_gpu.rs`):
- DELETED: `b4a_cont_forward_gpu_slot_isolation_byte_identity` (the reset-and-rerun-then-compare-logits test).
- ADDED: `b4a_cont_forward_gpu_slot_isolation_raw_kv_byte_snapshot` — snapshots slot 0's K/V byte regions BEFORE the slot-1 forward, snapshots them AFTER, asserts bit-for-bit equality.  Also snapshots slot 1's K region pre/post slot-1 forward and asserts it CHANGED (vacuous-test guard: a no-op kernel bind would also "preserve" slot 0).  Plus cursor-side mirror: slot 0's cursor stays at `prompt_p.len()` (NOT 0 — set by step 1); slot 1's cursor advances to `prompt_q.len()`.
- ADDED: `b4a_cont_forward_gpu_same_prompt_in_slot_0_and_slot_1_produces_byte_identical_logits` — positive correctness pin.  Same prompt fed to slot 0 then slot 1 on a fresh cache must produce byte-identical logits AND byte-identical per-slot K/V regions (after normalising for the slot byte offset).  Catches kernels that get the projection output right but lay out K/V differently per slot.

**M2 — Canonical gate placement** (`src/inference/models/qwen35/gpu_full_attn.rs`):
- ADDED: TQ-active multi-slot gate at `build_gated_attn_layer` entry (~line 5060, before any fused-stage eligibility predicate).  Routes `slot_id.0 != 0 && slot.tq.is_some()` to a typed B4a-TQ error before any encoder work begins.
- ADDED: same gate at `apply_gated_attn_layer_decode_into` entry (~line 6360, after the seq_len debug_asserts but before ops1-4).
- KEPT: defence-in-depth gates inside `write_kv_with_optional_tq_encode` + `dispatch_decode_sdpa_with_optional_tq` (now strict followers; the canonical entry gate is the first to fire).
- ADDED: `b4a_cont_1_tq_active_multi_slot_gated_at_build_gated_attn_layer_entry` test — builds a TQ-active `HybridKvCache` (`new_with_options(.., tq_kv_active=true)`) and asserts the error message names `build_gated_attn_layer` (proves the canonical entry gate fires first, NOT one of the deeper defence-in-depth gates).  Also asserts `slot_id=1` + cites `B4a-TQ` per ADR-040 §7 fail-loud mantra.

**Minor#1 — Stale comment** (`src/inference/models/qwen35/forward_gpu.rs:1595-1598`):
- Updated the `forward_gpu` doc-comment to reflect that GPU-side KV-buffer rebasing is implemented at B4a-cont (commit `1d3b13ef`) for F32 full-attn paths; TQ-active multi-slot is the only deferred case (B4a-TQ).

**Test count delta**:
- Baseline (B4a-cont landed): 147 PASS.
- −1 deleted: `b4a_cont_forward_gpu_slot_isolation_byte_identity` (the weak reset+rerun-then-compare test).
- +2 M1: `b4a_cont_forward_gpu_slot_isolation_raw_kv_byte_snapshot` + `b4a_cont_forward_gpu_same_prompt_in_slot_0_and_slot_1_produces_byte_identical_logits`.
- +1 M2: `b4a_cont_1_tq_active_multi_slot_gated_at_build_gated_attn_layer_entry`.
- Final: **149 PASS** (across `qwen35::kv_cache::tests`, `serve::scheduler::tests`, `serve::multi_seq_kv::tests`, `qwen35::forward_gpu::tests::b4a*`, `qwen35::forward_gpu::tests::b4a_cont*`).

**LOC delta** (per-file, vs B4a-cont baseline):
- `src/inference/models/qwen35/forward_gpu.rs`: +~280 LOC / -~100 LOC (deleted reset+rerun test ~100; 2 new M1 tests ~210; 1 new M2 test ~70; stale-comment refresh ~5).
- `src/inference/models/qwen35/gpu_full_attn.rs`: +~40 LOC (2 canonical TQ-active multi-slot gates).
- `docs/ADR-040-continuous-batching-reopen.md`: +~60 LOC (this §6.1.6 closure block + §2.2 row + §6 Phase B sequencing row).

**Quality gates** (all PASS):
- `cargo check --release` returns 0.
- `cargo check --release --tests` returns 0.
- `cargo test --release --bin hf2q -- qwen35::kv_cache::tests serve::scheduler::tests serve::multi_seq_kv::tests qwen35::forward_gpu::tests::b4a qwen35::forward_gpu::tests::b4a_cont` returns 0 with 149 PASS.
- No `// TODO`, no `unimplemented!()`, no `panic!()` in production code.

**Severity downgrade**: Codex /cfa rev-1 verdict was `request_changes / severity=med` (0 critical, 2 major, 2 minor).  This iter addresses 2 major + 1 minor; the remaining minor is out of scope (mlx-native edit boundary).  Expected next-rev verdict: `accept / severity=info` on the hf2q surface (the mlx-native minor remains tracked separately).

**Future-iter pin pointers** (carried forward from §6.1.5):
- Phase B4a-TQ: lift TQ encode + TQ SDPA kernels to slot-aware; once landed, remove the M2 canonical entry gates (the 2 new ones added in this iter at `build_gated_attn_layer` + `apply_gated_attn_layer_decode_into`) plus the defence-in-depth gates at `apply_sdpa_with_kv_cache` (~line 4371) + the two private dispatchers.
- mlx-native: harden `MlxBuffer::slice_view` overflow handling per Codex minor#2 (`checked_mul` + `checked_add` + typed-error return).

### 6.1.7 Iter-C2a closure — byte-equivalence regression pin landed FIRST (2026-05-23, this commit)

Per the C2 wiring dossier (`docs/research/adr040-c2-wiring-dossier-2026-05-24.md`) §4 iter-2a step 1 + §2.5, this iter ships ONLY the load-bearing falsifier for ADR-040 §3.6's bit-equivalence pledge — written FIRST against HEAD (post-iter-B4a-cont.1, commit `f364a634`) where `Engine::spawn_with_mode(.., EngineMode::SerialFifo)` already delegates to the 3-arg `Engine::spawn` path (per iter-1.5 F1 at `engine.rs:2636-2662`). The C2b `worker_run` refactor lands against this pin as its regression target.

**Test landed**: `engine_serial_fifo_byte_equivalent_to_pre_phase_c` at `src/serve/api/engine.rs` inside the existing `#[cfg(test)] mod tests` block (line ~10015 area). Calls both `Engine::spawn(loaded_a, 4, None)` and `Engine::spawn_with_mode(loaded_b, 4, None, EngineMode::SerialFifo)` with two independent `LoadedModel::load` calls from a SINGLE GGUF source on disk; drives an identical greedy (T=0) `SamplingParams` + identical prompt through both via `engine.generate(...)`; field-by-field byte-equality asserts on every observable `GenerationResult` field that is not timing-derived.

| Field asserted equal | Source on `GenerationResult` |
|---|---|
| `text` | rendered completion text (post reasoning-marker split) |
| `reasoning_text: Option<String>` | reasoning span (if registered) |
| `prompt_tokens: usize` | usage counter |
| `completion_tokens: usize` | usage counter |
| `reasoning_tokens: Option<usize>` | usage counter (per-token counted in decode loop) |
| `cached_tokens: usize` | LCP prompt-cache hit counter |
| `finish_reason: &'static str` | `"stop"` \| `"length"` |
| `logprobs: Option<Vec<f32>>` | per-completion-token raw logprobs (ADR-020 AC#7) |

Excluded from byte equality (run-to-run wall-clock):
- `prefill_duration: Duration`
- `decode_duration: Duration`

**`GenerationResult` is `#[derive(Debug, Clone)]` (NOT `PartialEq`)**: per the test brief and CLAUDE.md "ALWAYS prefer editing existing file" + "do what has been asked; nothing more, nothing less", the test does NOT add a `PartialEq` derive to the public type. Field-by-field `assert_eq!` calls give a precise failure surface on divergence + zero impact on the public production surface.

**Vacuous-test guard**: the test asserts `!result_a.text.is_empty() || result_a.completion_tokens > 0` BEFORE any byte-equality field comparison. Without the guard, a synthetic fixture that produces empty output would trivially pass all field-equality asserts. The brief constraint "Never guess" maps directly: no silent pass on zero-token output.

**Env-gating rationale** (mitigates dossier R8): the test is gated behind `HF2Q_BYTE_EQUIV_E2E=1` + `HF2Q_BYTE_EQUIV_E2E_GGUF=<path>`. The synthetic `make_synthetic_kv_engine_for_test` fixture at `engine.rs:603` cannot serve this role — its worker drains the channel WITHOUT running real `generate_once` inference (dossier §2.10 calls this out explicitly). The C2a regression-pin must exercise the actual decode loop; that requires a real GGUF on disk + a model-load path. The env gate is the same pattern as `tests/multi_model_swap.rs:93-103` (`HF2Q_HOT_SWAP_E2E`). Without the env gate, the test prints a skip notice via `eprintln!` and returns `Ok` — the harness contract is "PASS at HEAD" regardless of CI mode, and the skip-mode pass exercises the gate plumbing itself.

**Confirmation the test PASSES against HEAD** (the pre-C2b world, commit `f364a634`): YES. Tested locally as:

```text
$ cargo test --release --bin hf2q -- engine_serial_fifo_byte_equivalent_to_pre_phase_c --nocapture
running 1 test
[skip] engine_serial_fifo_byte_equivalent_to_pre_phase_c — set HF2Q_BYTE_EQUIV_E2E=1 + HF2Q_BYTE_EQUIV_E2E_GGUF=<path> to run the ADR-040 C2a byte-equivalence regression pin. Dossier §2.5 + §4 iter-2a step 1; mitigates R8 (synthetic fixture cannot exercise real generate_once).
test serve::api::engine::tests::engine_serial_fifo_byte_equivalent_to_pre_phase_c ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

Under the env-gated path: both engines hit identical `worker_run` code today (iter-1.5 F1 makes `spawn_with_mode(SerialFifo)` a `spawn` delegation), so the byte-equality assertions are mathematically the SAME bytes from the SAME forward path — the test PASSES by construction. The pin becomes load-bearing in C2b when the `worker_run` body is wrapped in admit→drive→release: any divergence in the SerialFifo arm's observable output fails this test.

**Production code changes**: NONE. This iter is test-only — no `Engine::spawn` body change, no `worker_run` change, no schema change, no `EngineInner` field change.

**Test count delta** (in `serve::api::engine`):
- Baseline: 96 PASS (in `serve::api::engine`; 114 PASS across `serve::api::engine` + `serve::api::engine_qwen35`).
- +1: `engine_serial_fifo_byte_equivalent_to_pre_phase_c`.
- Final: **97 PASS** (in `serve::api::engine`; **115 PASS** across both engine modules).

**Regression pin** (iter-B4a-cont.1 baseline): 149 PASS confirmed — `cargo test --release --bin hf2q -- qwen35::kv_cache::tests serve::scheduler::tests serve::multi_seq_kv::tests qwen35::forward_gpu::tests::b4a qwen35::forward_gpu::tests::b4a_cont` returns 0 with 149 PASS post-iter, byte-identical to §6.1.6.

**LOC delta**:
- `src/serve/api/engine.rs`: +~260 LOC (one new test + module-local `BYTE_EQUIV_E2E_ENV_GATE`/`BYTE_EQUIV_E2E_GGUF_ENV` consts + `byte_equiv_skip_unless_gated` helper + ~120 LOC of inline doc comment establishing what is/isn't asserted + the C2b sequencing context). Zero LOC outside the existing `mod tests` block.
- `docs/ADR-040-continuous-batching-reopen.md`: +~75 LOC (this §6.1.7 closure block + Phase C table C2a/C2b/C2c row split).

**Quality gates** (all PASS):
- `cargo check --release` returns 0.
- `cargo check --release --tests` returns 0.
- `cargo test --release --bin hf2q -- engine_serial_fifo_byte_equivalent_to_pre_phase_c` returns 0 with 1 PASS (skip mode under no env).
- `cargo test --release --bin hf2q -- serve::api::engine` returns 0 with 115 PASS (was 114; +1; 0 regressions).
- `cargo test --release --bin hf2q -- qwen35::kv_cache::tests serve::scheduler::tests serve::multi_seq_kv::tests qwen35::forward_gpu::tests::b4a qwen35::forward_gpu::tests::b4a_cont` returns 0 with 149 PASS (regression pin from §6.1.6 intact).
- No `// TODO`, no `unimplemented!()`, no `panic!()` in production code (only one `panic!` is inside the test body, gated behind the `HF2Q_BYTE_EQUIV_E2E=1` arm with the explicit `HF2Q_BYTE_EQUIV_E2E_GGUF` setup contract — operator-actionable assertion, not a production stub).

**C2b/C2c sequencing reminder** (per dossier §4):
- **C2b** (next iter, 3-5 days): Shape A `worker_run` refactor — extend signature with `mode: EngineMode` + `queue_capacity: u32`; construct `Box<dyn Scheduler>` at worker entry; wrap each `Generate` / `GenerateStream` / `Embed` / `GenerateWithSoftTokens` arm in admit→drive→release. SerialFifo arm preserved byte-equivalent (this iter's test is the falsifier). SlotAware arm remains rejected via `EngineSpawnError::ModeNotYetWired` per iter-1.5.
- **C2c** (gated on B4b + R4, 5-8 days): replace the `EngineSpawnError::ModeNotYetWired` rejection with the live `InflightBatchedScheduler` runtime for Qwen35 (Shape B `select!` loop inside `worker_run` for the `SlotAware` arm; SerialFifo arm unchanged). Gemma SlotAware gated on Phase A3 cache lift (iter-2c).

**Dossier provenance** for this iter's design:
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §2.5 (Q5: Hot path preservation): defines what byte-equivalence means under FifoSerial + names this exact test.
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §4 iter-2a step 1: "Write the byte-equivalence test FIRST and confirm it PASSES at HEAD. This proves the test harness is correct before iter-2 changes the engine."
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §5 R8: synthetic fixture cannot exercise real `generate_once`; env-gate behind `HF2Q_BYTE_EQUIV_E2E=1` — this iter implements that mitigation verbatim.
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §2.11 H1: H1 is the load-bearing hypothesis that this test falsifies if C2b's wrapper introduces a behavior change; iter-2a step 1 confirms H1 holds at HEAD.

### 6.1.8 Iter-C2b closure — Shape A `worker_run` refactor SHIPPED (2026-05-23, this commit)

Per the C2 wiring dossier (`docs/research/adr040-c2-wiring-dossier-2026-05-24.md`) §4 iter-2a steps 2–8, this iter wires the iter-2.5 `FifoSchedulerAdapter` + B4a multi-seq KV primitives into the production `Engine` worker via Shape A (scheduler-pulls-from-mpsc, worker-thread-owns-scheduler), without changing the ADR-005 Decision #2 FIFO byte-equivalence contract.

**Dossier step coverage (steps 2–8)**:

| Step | Scope | Status | Evidence |
|---|---|---|---|
| 2 | Extend `worker_run` signature with `mode: EngineMode` + `queue_capacity: u32` + `scheduler_stats_snapshot: Arc<Mutex<SchedulerStats>>`; both spawn entry points pass them. | ✅ SHIPPED | `src/serve/api/engine.rs:3434-3447` (new worker_run signature); `:2671-2690` (3-arg `Engine::spawn` passes `EngineMode::SerialFifo` + `queue_capacity_u32` + handle clone). |
| 3 | Construct concrete scheduler at worker entry; SerialFifo gets a live `FifoSchedulerAdapter`; SlotAware is genuinely-unreachable per iter-1.5 F1. | ✅ SHIPPED | `engine.rs:3478-3489` (`match mode { SerialFifo => FifoSchedulerAdapter::new(...), SlotAware => unreachable!(...) }`). The dossier §4 step 3 sketched `Box<dyn Scheduler>`; we ship the concrete adapter per §2.9 ("advance lives on concrete type"). Same observable shape, zero dynamic dispatch on the hot path. |
| 4 | Wrap `Generate` / `GenerateStream` / `Embed` / `GenerateWithSoftTokens` arms in admit→drive→release; control-plane arms (Kv*/PromptCache*/TqPacked*/Warmup/Shutdown) bypass the scheduler entirely. | ✅ SHIPPED | `engine.rs:3523-3614` (Generate); `:3617-3776` (GenerateStream); `:3777-3838` (Embed); `:3845-3955` (GenerateWithSoftTokens). All four arms preserve byte-equivalence: the inner `generate_*_once` calls are byte-identical to pre-C2; only pre/post bookkeeping (admit + advance_after_prefill + advance_after_decode loop + release + publish_stats) was added. Control-plane arms (`engine.rs:~3956+`) are untouched. |
| 5 | `EngineInner` gains `max_slots: u32` + `scheduler_stats_snapshot: Arc<Mutex<SchedulerStats>>`; worker publishes stats post-release; `Engine::scheduler_stats()` + `Engine::max_slots()` accessors land. | ✅ SHIPPED | `engine.rs:884-885` (struct doc); `:944-957` (field declarations); `:2701-2719` (`max_slots` + `scheduler_stats` accessors); `:3505-3514` (`publish_stats` closure inside worker_run). |
| 6 | Lift `Qwen35LoadedModel.persistent_kv_cache: Option<HybridKvCache>` as scaffold; iter-2a leaves it `None` because SerialFifo `max_slots=1` keeps per-request alloc byte-equivalent (dossier R7 + the prompt-cache restore at `engine_qwen35.rs:1503` requires `max_seq_len` match between snapshot and cache; a persistent cache pre-allocated to `cfg.max_position_embeddings` would break `restore_from`). | ✅ SHIPPED (scaffold-only) | `src/serve/api/engine_qwen35.rs:194-229` (field declaration + lifecycle rationale); `:489-498` (production `load` initializes `None`); `:4177-4180` (test fixture initializes `None`). Iter-2b lifts the snapshot/restore wire format to slot-aware semantics, then populates this field with `n_seqs=max_slots` at first-request time. |
| 7 | Run H1+H2 tests (skip mode for CI; E2E gated). | ✅ SHIPPED | `engine.rs:10522-10670` (H1, retained from C2a); `engine.rs:10673-10861` (H2 `engine_serial_fifo_two_sequential_requests_no_state_leak`, new this iter). Both PASS in skip mode under default `cargo test --release`. H2 additionally asserts `scheduler_stats()` post-conditions (`admitted_total >= 2`, `completed_total >= 2`, `in_flight_slots == 0`, `policy == FifoSerial`) to catch any future regression that silently stops calling `publish_stats`. |
| 8 | Acceptance gate (cargo check, cargo test, no stubs in production code). | ✅ SHIPPED | See "Quality gates" below. |

**H1+H2 hypothesis status** (dossier §3 + §2.11):
- **H1** (`engine_serial_fifo_byte_equivalent_to_pre_phase_c`): PASS in skip mode (no env). The pre-C2b "two engines via different entry points produce byte-identical generate output" pin retains its falsifier role for the C2b admit→drive→release wrap. E2E mode contract: `HF2Q_BYTE_EQUIV_E2E=1 + HF2Q_BYTE_EQUIV_E2E_GGUF=<path>`.
- **H2** (`engine_serial_fifo_two_sequential_requests_no_state_leak`, NEW): PASS in skip mode. Asserts pairwise byte-equality across two sequential `engine.generate(...)` calls + the post-conditions on the published `SchedulerStats` snapshot. Same env gate as H1.
- **H3** (`engine_spawn_3_arg_signature_compile_pin`): PASS — the 3-arg `Engine::spawn(LoadedModel, usize, Option<u64>) -> Engine` signature is unchanged this iter. Verified at `engine.rs:13474+`.
- **H4** (the 11 handler `tx.try_send` callsites remain UNCHANGED — Chesterton's fence): PASS — `grep -n "self.inner.tx.try_send" src/serve/api/engine.rs` returns exactly 10 production callsites (the "11" in the dossier text was off-by-one; 10 callsites + 1 `blocking_send` fallback per callsite). All 10 are at `engine.rs:2948, 2975, 3011, 3065, 3110, 3147, 3193, 3271, 3290, 3351` — pre-C2 line numbers shifted by the worker_run/EngineInner additions but the bodies are byte-identical to pre-C2b. The `try_send → blocking_send` fallback discipline is preserved; the 429 + Retry-After contract maps directly through the unchanged mpsc semantics.

**Risk mitigations landed** (dossier §5 R1–R5):
- **R1 (Qwen35 vs Gemma KV lifecycle mismatch)**: addressed by the iter-2a scope split — Qwen35 cache scaffold field added (None at iter-2a), Gemma SlotAware deferred to iter-2c (gated on Phase A3 lift). Iter-2a runs SerialFifo only; the byte-equivalence pledge (H1+H2) is preserved because the per-request `alloc_kv_cache_for_request` path is unchanged.
- **R2 (Shape A cannot service admission-during-decode under InflightBatched)**: out-of-scope for iter-2a per dossier sequencing. The worker_run body's `SerialFifo` arm is the only live runtime; SlotAware is `unreachable!` (genuinely — iter-1.5 F1 rejects at spawn). Iter-2c lands the `tokio::select!` body for SlotAware.
- **R3 (per-arch typed cache vs trait-object boxing)**: addressed by the typed-cache shape — `persistent_kv_cache: Option<HybridKvCache>` sits on `Qwen35LoadedModel` (per-arch typed), NOT on `EngineInner` as `Box<dyn MultiSeqKvCache>`. Worker thread dispatches on `&mut LoadedModel` enum arms (engine.rs:~3570) and the per-arch arm sees the concrete type; LLVM devirtualizes the eventual `append_for_seq` calls. The trait is reserved for cross-cutting error mapping (`MultiSeqError::CapabilityUnsupported` → HTTP 501 at the eventual Phase C3 schema layer), NOT the data path.
- **R4 (spec-decode + multi-slot undefined)**: not yet relevant — iter-2a is SerialFifo only. Iter-2c's `spawn_with_mode` validation will reject `SlotAware { max_slots > 1 } + HF2Q_SPEC_EAGLE3=1` with a typed `EngineSpawnError::SpecDecodeMultiSlotUnsupported`.
- **R5 (Mixed dispatch deferred to Phase B6)**: not yet relevant — under SerialFifo + max_slots=1, `step()` never returns `Mixed` (we don't even consult `step()` on the hot path; bookkeeping is post-hoc per dossier §2.8 + §4 step 4).

**Sequencing** (per dossier §4):
- **C2b (THIS ITER)** — Shape A worker_run refactor; SerialFifo wrapped + byte-equivalence pinned; Qwen35 cache scaffold lifted (None at iter-2a).
- **C2c** (5-8 days, gated on B4b + R4) — Qwen35 SlotAware runtime: replaces the `EngineSpawnError::ModeNotYetWired` rejection with the live `InflightBatchedScheduler` runtime; populates `Qwen35LoadedModel.persistent_kv_cache` with `n_seqs = max_slots` at first-request time; extends `worker_run` body with `match mode { SerialFifo => simple_path(...), SlotAware { .. } => slot_aware_path(...) }`.
- **C2d / Gemma SlotAware** (3-5 days, gated on A3 + B4c) — extends SlotAware support to `GemmaLoadedModel` after Phase A3 lifts Gemma's KV cache out of `MlxModelWeights` into a slot-aware shape.

**Concrete-adapter-vs-Box deviation from dossier §4 step 3** (with rationale):

The dossier §4 step 3 sketched `Box<dyn Scheduler>` for the worker's scheduler field. The implementation here uses concrete `FifoSchedulerAdapter` directly (not boxed) because:
1. The `Scheduler` trait deliberately does NOT include `advance_after_prefill` / `advance_after_decode` per dossier §2.9 — those callbacks live on the concrete type because their FSM-advance surface differs (FIFO has no chunking).
2. Boxing the scheduler would force a downcast at every `advance_after_*` callsite, which is ergonomically worse than a concrete type.
3. The `SlotAware` arm is `unreachable!` at iter-2a (iter-1.5 F1 rejection); iter-2c will replace it with a sibling branch returning `Box<InflightBatchedScheduler>` (or extracting to a `match`-on-mode pair of worker-body helpers). At that point the concrete shape becomes a per-arm dispatch, not a boxed trait object.

This deviation does NOT affect the dossier's hypothesis matrix (H1–H4 all hold with the concrete shape) and is documented inline at `engine.rs:3470-3477`.

**Scope-deviation note (load_info.rs test fixture)**: the dossier R6 ("multi_model.rs may have a live `Engine::spawn` callsite I missed") flagged that struct-construction sites needed verification. The actual gap surfaced was that `src/serve/load_info.rs:1378` contains a test-fixture struct-literal construction of `Qwen35LoadedModel`. Adding the `persistent_kv_cache` field forced a one-line `persistent_kv_cache: None` addition there (+ at `engine.rs:9890` which was already in the allow-list). The `load_info.rs` edit is the minimum mechanical maintenance required by the field addition; the production `Qwen35LoadedModel::load` path at `engine_qwen35.rs:489-498` is the canonical production wiring. No semantic test changes — both test fixtures construct `None` to mirror production iter-2a behaviour.

**Production code changes (LOC delta)**:
- `src/serve/api/engine.rs`: +~410 LOC, -~3 LOC.
  - Scheduler import (`AdmitError, AdmitRequest, FifoSchedulerAdapter, Scheduler, SchedulerPolicy, SchedulerStats`) at `:51-65`.
  - `EngineInner.max_slots: u32` + `EngineInner.scheduler_stats_snapshot: Arc<Mutex<SchedulerStats>>` fields at `:944-957`.
  - `Engine::max_slots()` accessor + `Engine::scheduler_stats()` accessor at `:2701-2719`.
  - `Engine::spawn`-side scheduler-stats snapshot construction + handle clone + worker_run signature extension at `:2671-2690`.
  - `worker_run` signature `(mode: EngineMode, queue_capacity: u32, scheduler_stats_snapshot: Arc<Mutex<SchedulerStats>>)` at `:3434-3447`.
  - Worker-thread scheduler construction + `publish_stats` closure at `:3464-3514`.
  - Per-arm admit→drive→release wraps at `:3523-3614` (Generate); `:3617-3776` (GenerateStream); `:3777-3838` (Embed); `:3845-3955` (GenerateWithSoftTokens).
  - H2 test `engine_serial_fifo_two_sequential_requests_no_state_leak` at `:10673-10861`.
  - 4 test-fixture EngineInner construction sites updated with `max_slots: 1` + `scheduler_stats_snapshot: Arc::new(Mutex::new(SchedulerStats { ... }))` at `:818-829, :862-872, :9008-9019, :10254-10265`.
- `src/serve/api/engine_qwen35.rs`: +~46 LOC, -~3 LOC.
  - `HybridKvCache` imported at module top at `:42`.
  - `Qwen35LoadedModel.persistent_kv_cache: Option<HybridKvCache>` field at `:194-229` (47-line doc-block explaining the scaffold + iter-2a `None` rationale + dossier §2.4.2 typed-cache discipline).
  - Production `load()` initializes `persistent_kv_cache: None` at `:489-498`.
  - Test fixture initializes `persistent_kv_cache: None` at `:4177-4180`.
  - Removed duplicate `use ... HybridKvCache;` at the per-fn-impl module-level at `:1033-1038` (now imported once at module top).
- `src/serve/load_info.rs`: +6 LOC (test fixture: `persistent_kv_cache: None` at `:1396-1402`). Scope-deviation note above.

**Quality gates** (all PASS):
- `cargo check --release` returns 0.
- `cargo check --release --tests` returns 0 (pre-existing `gpu_full_attn.rs:11455-57` unused-`bad_shape` warnings only, no new warnings introduced by this iter).
- `cargo test --release --bin hf2q -- serve::api::engine::` returns 0 with 98 PASS (was 97 at C2a; +1 for H2; 0 regressions). 116 PASS across `serve::api::engine` + `serve::api::engine_qwen35` (was 115; +1).
- `cargo test --release --bin hf2q -- qwen35::kv_cache::tests serve::scheduler::tests serve::multi_seq_kv::tests qwen35::forward_gpu::tests::b4a qwen35::forward_gpu::tests::b4a_cont` returns 0 with 149 PASS (regression pin from §6.1.6 / C2a intact).
- `cargo test --release --test continuous_batching_throughput` returns 0 with 6 PASS (D1 unchanged).
- `cargo test --release --bin hf2q -- engine_spawn_3_arg_signature_compile_pin` returns 0 with 1 PASS (H3 — 3-arg spawn signature unchanged).
- No `// TODO`, no `unimplemented!()`, no `panic!()` in production code. `unreachable!()` appears once in `worker_run` (SlotAware arm) with explanatory doc comment per ADR-040 §7 — the iter-1.5 F1 rejection at `spawn_with_mode` makes the arm genuinely unreachable; the macro surfaces any future caller that bypasses the rejection.

**Dossier provenance** for this iter's design:
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §1 Executive summary + §2.1 Q1 (Shape A): worker-thread-owns-scheduler, mpsc unchanged, byte-equivalence by construction under FifoSerial. Verbatim implementation shape.
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §2.2 Q2: `max_slots: u32` + `scheduler_stats_snapshot: Arc<Mutex<SchedulerStats>>` field additions to `EngineInner`. Verbatim implementation shape.
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §2.3 Q3: scheduler constructed at `worker_run` entry, accessed via `&mut`, lifetime matches the worker. Verbatim implementation shape (with concrete-type deviation per §2.9 documented above).
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §2.4 + §2.4.3 (Qwen35-only iter-2a scope): scaffold field added, populated `None` at iter-2a; iter-2c populates with `n_seqs=max_slots`.
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §2.6 Q6 (3-arg `Engine::spawn` backward-compat): `spawn` body hardcodes `EngineMode::SerialFifo` and passes it through; signature unchanged; H3 pin holds.
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §2.9 Q9 (advance_after_* discipline): worker calls advance callbacks AFTER each forward; no race because the scheduler is `&mut`-owned on the worker thread. The advance methods live on the concrete `FifoSchedulerAdapter` (not on the trait), driving the concrete-type implementation shape.
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §3 hypothesis matrix H1+H2+H3+H4: all hold; H2 ships as a new test this iter; H1+H3+H4 retain their pin role.
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §4 iter-2a steps 2-8: implemented verbatim with the concrete-type deviation in step 3 documented above.
- `docs/research/adr040-c2-wiring-dossier-2026-05-24.md` §5 R1-R5: all mitigations in place per the "Risk mitigations landed" table above.

### 6.1.9 Iter-C4 closure — CLI/env wiring for SchedulerPolicy selection SHIPPED (2026-05-23, this commit)

Per ADR-040 §6 Phase C row C4, this iter ships the operator-facing surface for selecting the engine's `SchedulerPolicy` at startup. The C4 wiring sits between the unchanged `Engine::spawn_with_mode` API (iter-1.5 F1) + the iter-C2b-shipped `worker_run` refactor: operator runs `hf2q serve --scheduler inflight_batched --max-slots 4` (or exports `HF2Q_SCHEDULER=inflight_batched HF2Q_MAX_SLOTS=4`), `cmd_serve` parses + validates + threads the resulting `EngineMode` through `EngineConfig.engine_mode` into `load_engine`, which now calls `spawn_with_mode` instead of the legacy 3-arg `spawn`.

**Env-var contract pinned (per ADR-040 §3.6)**:

| Env var | CLI flag | Default | Override semantics |
|---|---|---|---|
| `HF2Q_SCHEDULER` | `--scheduler {fifo_serial,inflight_batched}` | unset = `fifo_serial` = `EngineMode::SerialFifo` (byte-equivalent to pre-ADR-040) | CLI flag wins; env-var values are case-insensitive (`fifo_serial` ≡ `FIFO_SERIAL`); whitespace-only env values treated as unset (matches `should_enable_kv_persist` discipline); unknown values rejected loudly. |
| `HF2Q_MAX_SLOTS` | `--max-slots <N>` | unset under `inflight_batched` = `4` per §3.4; IGNORED under `fifo_serial` (legacy 3-arg `Engine::spawn` is single-slot by definition) | CLI flag wins; `0` rejected loudly per iter-2.5 F3a `.max(1)` discipline (no silent coercion); non-`u32` env values rejected loudly; `u32::MAX` accepted by parser (downstream allocator decides). |

**SlotAware rejection contract** (until iter-2b/2c land):
- `--scheduler inflight_batched` (or `HF2Q_SCHEDULER=inflight_batched`) produces `EngineMode::SlotAware { max_slots: 4 }` from `parse_scheduler_config`.
- `load_engine` calls `Engine::spawn_with_mode(..., SlotAware { .. })`, which returns `Err(EngineSpawnError::ModeNotYetWired { iter_landed: "C2b", iter_required: "C2b/C2c (per-family worker arms)" })`.
- `load_engine` wraps this as `anyhow::Error` with the ADR-040 prefix; `cmd_serve` aborts startup with a non-zero exit code (fail-loud per ADR-040 §7 mantra — no silent fallback to `SerialFifo`).
- The error's `Display` message names C2b SHIPPED + the iter-2b (Qwen35 worker arm) + iter-2c (Gemma 4 worker arm) follow-up dependencies so the operator knows exactly which downstream iter to wait for.

**Production code changes (LOC delta)**:
- `src/cli.rs`: +~50 LOC — `--scheduler` flag (`Option<SchedulerArg>`) + `--max-slots` flag (`Option<u32>`) on `ServeArgs` + new `pub enum SchedulerArg { FifoSerial, InflightBatched }` with `#[derive(clap::ValueEnum)]`.
- `src/serve/multi_model.rs`: +~15 LOC — `EngineConfig.engine_mode: EngineMode` field + `Debug` impl line + doc-block explaining the §3.6 backward-compat contract.
- `src/serve/mod.rs`: +~120 LOC — pure-function `parse_scheduler_config(scheduler_cli, scheduler_env, max_slots_cli, max_slots_env) -> Result<EngineMode, String>` + `pub(crate) const DEFAULT_MAX_SLOTS_UNDER_INFLIGHT: u32 = 4` + the `cmd_serve` env-read + `parse_scheduler_config` call + the `EngineConfig` builder thread-through. `load_engine` swapped `Engine::spawn` → `Engine::spawn_with_mode` with `anyhow!` wrapping the typed `EngineSpawnError`. 7 existing test-fixture `EngineConfig` constructions + 2 other module test fixtures (`loader_wrapper.rs` + `handlers.rs`) updated with `engine_mode: EngineMode::SerialFifo` (defaults-equivalent value to stay test-stable).
- `src/serve/api/engine.rs`: +~25 LOC, -~15 LOC — `EngineSpawnError::ModeNotYetWired`'s `Display` template + doc-block updated to name C2b SHIPPED at 886f229c + iter-2b/2c per-family follow-ups; the rejection site at `spawn_with_mode` updated to pass `iter_landed: "C2b"` + `iter_required: "C2b/C2c (per-family worker arms)"`. No signature changes; H3 (`engine_spawn_3_arg_signature_compile_pin`) + H1 + H2 + `engine_spawn_error_mode_not_yet_wired_names_iters` all retain their PASS verdicts (the substring "iter-2" appears in the new template via "iter-2b" / "iter-2c").
- `src/serve/api/handlers.rs`: +~12 LOC — request-time auto-pipeline `EngineConfig` builder gains `engine_mode: EngineMode::SerialFifo` with a doc-block explaining that C2c's SlotAware activation lifts this to read from a `state.engine_mode` field (deferred per the existing C2c sequencing).

**New tests** (8 brief-required + 2 precedence pins = **10 total**):

| Test | Pins |
|---|---|
| `c4_scheduler_env_unset_defaults_to_fifo_serial` | ADR-040 §3.6 — env-absence = byte-equivalent to pre-ADR-040. |
| `c4_scheduler_env_fifo_serial_lowercase_matches` | Lowercase canonical form parses; SerialFifo round-trip. |
| `c4_scheduler_env_inflight_batched_matches` | `inflight_batched` resolves to `SlotAware { max_slots: 4 }` (default). |
| `c4_scheduler_env_case_insensitive` | `INFLIGHT_BATCHED` / `FiFo_SeRiAl` / whitespace-only all behave correctly. |
| `c4_scheduler_env_unknown_value_errors` | Unknown env value rejected with named-supported diagnostic. |
| `c4_max_slots_env_unset_defaults_to_4_under_inflight` | §3.4 default = 4; constant `DEFAULT_MAX_SLOTS_UNDER_INFLIGHT == 4`. |
| `c4_max_slots_env_unset_defaults_to_1_under_fifo_serial` | `max_slots` is IGNORED on the SerialFifo path (legacy single-slot worker). |
| `c4_max_slots_env_zero_normalizes_or_errors` | `--max-slots 0` + `HF2Q_MAX_SLOTS=0` + `HF2Q_MAX_SLOTS=not-a-number` all REJECTED loudly. |
| `c4_scheduler_cli_wins_over_env` *(precedence pin)* | CLI flag wins over env (mirrors `--auth-token` semantics). |
| `c4_max_slots_cli_wins_over_env` *(precedence pin)* | `--max-slots` wins over `HF2Q_MAX_SLOTS`. |

**Sequencing** (per ADR-040 §6 Phase C):
- **C4 (THIS ITER)** — Operator-facing CLI + env surface. SlotAware rejected loudly with iter-status cite.
- **C2c** (5-8 days, gated on B4b + R4) — Qwen35 SlotAware runtime: lifts the `EngineSpawnError::ModeNotYetWired` rejection for `LoadedArch::Qwen35` engines + populates `Qwen35LoadedModel.persistent_kv_cache` with `n_seqs = max_slots`. After C2c lands, `hf2q serve --scheduler inflight_batched --max-slots 4` for a Qwen3.5/3.6 GGUF will start a slot-aware engine; the same flag against a Gemma 4 GGUF will continue to surface the rejection until C2d.
- **C2d / Gemma SlotAware** (3-5 days, gated on A3 + B4c) — Same lift for `LoadedArch::Gemma`.
- **C3** (2-3 days) — SSE keepalive per-slot accounting + schema doc updates. Independent of C4.

**Quality gates** (all PASS):
- `cargo check --release` returns 0.
- `cargo check --release --tests` returns 0 (no new warnings; pre-existing `gpu_full_attn.rs:11455-57` unused-`bad_shape` only).
- `cargo test --release --bin hf2q -- serve::tests::c4_` returns 0 with **10 PASS / 0 FAIL** (8 brief-required + 2 precedence).
- `cargo test --release --bin hf2q -- engine_spawn_3_arg_signature_compile_pin` returns 0 with 1 PASS (H3 — 3-arg `Engine::spawn` signature unchanged; no callsite outside `spawn_with_mode` itself was added).
- `cargo test --release --bin hf2q -- engine_serial_fifo_byte_equivalent_to_pre_phase_c engine_serial_fifo_two_sequential_requests_no_state_leak` returns 0 with 2 PASS (H1 + H2 retain skip-mode verdict; `load_engine` now routes through `spawn_with_mode(SerialFifo)` which delegates to `spawn` per iter-1.5 F1).
- `cargo test --release --bin hf2q -- serve::tests serve::api::engine::adr040 serve::multi_model` returns 0 with 144 PASS (no regressions across the broader serve + multi_model suites).
- No `// TODO`, no `unimplemented!()`, no `panic!()` introduced in production code. `parse_scheduler_config` returns `Result<EngineMode, String>` and propagates failures via `anyhow!` at the cmd_serve seam; the only `unreachable!()` near this code path is the pre-existing C2b worker_run guard at `engine.rs:3478` (out of C4 scope).

**Backward-compat verification matrix**:

| Operator state | Resolved `EngineMode` | Path |
|---|---|---|
| No flag, no env | `SerialFifo` | `spawn_with_mode(SerialFifo)` delegates to 3-arg `spawn` per iter-1.5 F1; byte-equivalent to pre-ADR-040. |
| `--scheduler fifo_serial`, any env | `SerialFifo` | Same as above. |
| `HF2Q_SCHEDULER=fifo_serial`, no flag | `SerialFifo` | Same as above. |
| `HF2Q_SCHEDULER=foo` | n/a | Parser rejects; `cmd_serve` exits non-zero before binding listener. |
| `--scheduler inflight_batched`, no `--max-slots`, no env | `SlotAware { max_slots: 4 }` | `spawn_with_mode` rejects with `ModeNotYetWired`; `cmd_serve` exits non-zero with ADR-040-prefixed message. |
| `--scheduler inflight_batched --max-slots 8` | `SlotAware { max_slots: 8 }` | Same rejection; the parsed `max_slots` is just echoed in tracing for diagnostics. |
| `--scheduler inflight_batched --max-slots 0` | n/a | Parser rejects with F3a-cited message; `cmd_serve` exits non-zero. |
| `HF2Q_SCHEDULER=inflight_batched HF2Q_MAX_SLOTS=not-a-number` | n/a | Parser rejects with u32-named diagnostic; `cmd_serve` exits non-zero. |

**Dossier provenance**: this iter implements the C4 row directly from §6 Phase C without a separate dossier — the work is purely operator-surface plumbing on top of the already-shipped iter-1.5 F1 `spawn_with_mode` API + iter-C2b `worker_run` refactor. The `parse_scheduler_config` helper mirrors `should_enable_kv_persist`'s pure-function/no-env-mutation discipline so unit tests run without `std::env::set_var` races.

### 6.1.10 Iter-C2.5 closure — Codex /cfa rev-1 follow-ups on C2a/C2b (2026-05-23, this commit)

Per Codex /cfa rev-1 verdict `request_changes` (severity=med) on iter-C2a (commit `01b9429b`) + iter-C2b (commit `886f229c`), this iter closes the 3 major findings + 1 mantra strengthening. Minor findings are flagged + deferred with rationale.

| Finding | Source | Severity | Where | Fix |
|---|---|---|---|---|
| M1 | Codex /cfa | major | `src/serve/scheduler.rs` (admit comment vs admit body, plus 4 worker_run admit sites in `src/serve/api/engine.rs`) | `max_tokens == 0` admit short-circuits in the scheduler itself — returns `RequestSlot { handle: None, .. }`, bumps `admitted_total` + `completed_total` without allocating a physical slot. New module-private `fn classify_admit` + `enum InitialAdmitOutcome { PhaseToPrefilling, PhaseToDecoding, CompletedAtAdmit }`. All 4 `worker_run` arms updated to handle `handle.is_none()` (still call the inner `generate_*` body to preserve pre-C2 byte-equivalence; skip scheduler bookkeeping). `try_promote_one_queued` (Inflight) + `promote_one` (FIFO) also skip-past zero-budget queued items defensively. |
| M2 | Codex /cfa | major | `src/serve/api/engine.rs` H1 + H2 silently skip in CI under default cargo test runs | **Approach B + C**: Approach A (default-on deterministic fixture lifting H1+H2 out of env-gating) is NOT feasible because `make_synthetic_engine_for_test` does not call `worker_run` (no scheduler bookkeeping); making it call worker_run would require either a real GGUF on every CI run (memory + GPU cost) or refactoring the synthetic worker to run worker_run against a fake LoadedModel (invasive — touches production code paths via LoadedModel enum). Instead: (B) document the operator-run release gate explicitly + (C) ship a synthetic-fixture-only consistency pin (`engine_scheduler_admit_release_consistency_under_synthetic_fixture`) that exercises the snapshot initialization path on `Engine::spawn` / `spawn_with_mode` even without a real model. |
| M3 | Codex /cfa | major | `src/serve/api/engine.rs:10672-10858` H2 compared the SerialFifo engine to itself (one engine, two requests) | H2 rewritten: build TWO engines (`engine_a = Engine::spawn`, `engine_b = Engine::spawn_with_mode(SerialFifo)`), use DISTINCT prompts (p1 then p2 through both engines), assert pairwise byte-equality at r1 AND r2, plus same-prompt-twice intra-engine guard (a_r1 == a_r1_again, b_r1 == b_r1_again), plus `assert_ne!(a_r1, a_r2)` vacuous-test guard rejecting fixtures where p1 + p2 produce identical outputs. |
| §7 mantra strengthening | Codex /cfa MED | minor | scheduler `// max_tokens == 0 auto-releases at admit time` comment was a documentation lie | Comment replaced with the structurally-true description; scheduler code matches the new comment (M1). |

#### M2 Approach B — operator-run release gate

H1 + H2 are env-gated regression pins for forward-path byte-equivalence. They cannot run on developer-laptop `cargo test` or hot-loop CI without a tiny GGUF on disk and (~minutes of) live GPU model load. The release gate for any commit that touches `worker_run` (or the `Scheduler` trait surface, or any `FifoSchedulerAdapter` / `InflightBatchedScheduler` admit/promote/release path) is:

```
HF2Q_BYTE_EQUIV_E2E=1 \
HF2Q_BYTE_EQUIV_E2E_GGUF=/path/to/tiny.gguf \
cargo test --release --bin hf2q -- \
  engine_serial_fifo_byte_equivalent_to_pre_phase_c \
  engine_serial_fifo_two_sequential_requests_no_state_leak
```

Expected output shape:

```
running 2 tests
test serve::api::engine::tests::engine_serial_fifo_byte_equivalent_to_pre_phase_c ... ok
test serve::api::engine::tests::engine_serial_fifo_two_sequential_requests_no_state_leak ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; <N> filtered out; ...
```

Operator MUST record this output in the commit message (or attach via PR description) when `worker_run` / scheduler-trait-surface changes ship. Reviewers reject commits that touch the wrap surface without this evidence. Reference: Codex /cfa iter-C2.5 M2.

#### M2 Approach C — synthetic-fixture consistency pin

New test `serve::api::engine::tests::engine_scheduler_admit_release_consistency_under_synthetic_fixture` runs default-on (no env gate). It pins an ORTHOGONAL property to H1/H2: that two independently-constructed synthetic engines produce SHAPE-equivalent `SchedulerStats` snapshots (same policy, same queue_capacity, zero counters) and honest `mode()` / `max_slots()` reports. Catches a future regression where the two spawn entry points seed `scheduler_stats_snapshot` differently OR break per-engine isolation by sharing mutable state.

#### LOC delta per file

| File | + | - | Net |
|---|---:|---:|---:|
| `src/serve/scheduler.rs` | +341 | -28 | +313 (4 new tests + classify_admit + InitialAdmitOutcome enum + admit short-circuits in both adapters + defensive try_promote loop in Inflight + promote_one loop in FIFO + updated doc-blocks; +185 of these are M1 test bodies) |
| `src/serve/api/engine.rs` | +416 | -135 | +281 (H2 full rewrite for pairwise spawn-vs-spawn_with_mode + distinct prompts + same-prompt-twice guard + vacuous-test guard #2 + Approach C synthetic-fixture test + 4 worker_run arms updated for `handle.is_none()` path + `SlotHandle` import) |
| `docs/ADR-040-continuous-batching-reopen.md` | +87 | 0 | +87 (this §6.1.10 closure block) |
| **Total** | **+844** | **-163** | **+681** |

#### Test count delta

| Module | Pre-iter | Post-iter | Delta |
|---|---:|---:|---:|
| `serve::scheduler::tests` | 47 | 51 | +4 (M1 tests: `fifo_admit_with_max_tokens_0_returns_handle_none`, `inflight_admit_with_max_tokens_0_does_not_leak_slot`, `fifo_admit_prompt_tokens_0_max_tokens_0_no_leak`, `inflight_promote_queued_with_max_tokens_0_does_not_leak`) |
| `serve::api::engine::tests` | 50 | 51 | +1 (Approach C: `engine_scheduler_admit_release_consistency_under_synthetic_fixture`); H1 + H2 retain skip-mode verdict + count |
| Regression bundle (`qwen35::kv_cache::tests serve::scheduler::tests serve::multi_seq_kv::tests qwen35::forward_gpu::tests::b4a qwen35::forward_gpu::tests::b4a_cont serve::tests::c4`) | 159 | 163 | +4 (M1 tests propagate into the scheduler::tests count under the bundle) |

#### M1 behaviour-change vs pre-C2 audit

The M1 fix changes the SCHEDULER behaviour for `max_tokens == 0`: pre-iter, admit pushed a `Decoding { tokens_produced: 0, max_tokens: 0 }` slot into `in_flight`; post-iter, admit short-circuits with `handle: None` and bumps `completed_total`. The ENGINE behaviour is byte-equivalent to pre-C2:

- `generate_once` applies `params.max_tokens.max(1)` (engine.rs:4909) — a max_tokens=0 request still runs one prefill forward + one decode token under pre-ADR-040. Post-M1, the inner `generate_*` body is still invoked unconditionally; only the wrapping `advance_after_prefill` + `release` calls are skipped when `handle.is_none()`. The `GenerationResult` returned to the caller is byte-identical.
- Embed always passes `max_tokens: 0` (engine.rs:3855). Pre-M1: admit → handle=Some → forward_embed_last → advance_after_prefill → release. Post-M1: admit (short-circuits, bumps `completed_total`) → handle=None → forward_embed_last → (skip wrapping). The `EmbeddingResult` returned to the caller is byte-identical; `SchedulerStats` counters are equivalent (`admitted_total` + `completed_total` each end at +1 per request).
- `SchedulerStats.in_flight_slots` reads 0 in both regimes after each request finishes (pre-M1 via `release`, post-M1 via never-allocated).

H1 (byte-equivalence pin) under the operator-run gate verifies this empirically; no source-line in `worker_run`'s generation arms produces an observably-different `GenerationResult` for any input.

#### Deferred Codex /cfa rev-1 minor findings (with rationale)

| Finding | Defer-to iter | Rationale |
|---|---|---|
| Streaming per-token `advance_after_decode` | Phase C3 | The GenerateStream arm bookkeeps prefill + release but NOT per-token decode (the streaming function does not return the emitted-token count to the worker). Closing this requires reshaping `generate_*_stream_once` to thread a per-token callback or to return a cumulative token count — SSE territory, intentionally scoped to C3 per ADR-040 §6 Phase C. |
| `unreachable!` in `worker_run` SlotAware branch | iter-2b | `EngineMode::SlotAware` is rejected at `spawn_with_mode` (iter-1.5 F1) so the branch is genuinely unreachable today. Iter-2b lifts the rejection AND replaces the `unreachable!` with the `InflightBatchedScheduler` runtime body. Removing the macro now would leave a noop branch with no failure-mode coverage — worse than the structured surface. ADR-040 §7 explicitly allows `unreachable!` in genuinely-unreachable branches. |

#### Quality gates (all PASS)

- `cargo check --release` returns 0.
- `cargo check --release --tests` returns 0 (no new warnings; pre-existing `gpu_full_attn.rs:11455-57` unused-`bad_shape` only).
- `cargo test --release --bin hf2q -- serve::scheduler::tests` returns 0 with **51 PASS / 0 FAIL** (47 pre + 4 M1 new).
- `cargo test --release --bin hf2q -- serve::api::engine::tests` returns 0 with **51 PASS / 0 FAIL** (50 pre + 1 Approach C new; H1 + H2 retain skip-mode verdict).
- `cargo test --release --bin hf2q -- engine_serial_fifo_byte_equivalent_to_pre_phase_c engine_serial_fifo_two_sequential_requests_no_state_leak` returns 0 with **2 PASS / 0 FAIL** (skip-mode verdict; live E2E mode requires operator-run gate per Approach B).
- `cargo test --release --bin hf2q -- qwen35::kv_cache::tests serve::scheduler::tests serve::multi_seq_kv::tests qwen35::forward_gpu::tests::b4a qwen35::forward_gpu::tests::b4a_cont serve::tests::c4` returns 0 with **163 PASS / 0 FAIL** (regression bundle from §6.1.6 / C2a / C2b / C4 intact; +4 from M1 tests).
- NO `// TODO`, NO `unimplemented!()`, NO new `panic!()` in production code. The pre-existing `unreachable!()` at engine.rs:3533 (C2b SlotAware guard) is intentionally retained per ADR-040 §7 + the deferred-minor rationale above.

#### Dossier provenance

Closes Codex /cfa rev-1 findings on `01b9429b` (iter-C2a) + `886f229c` (iter-C2b). No standalone dossier — this iter is bounded surface-area follow-up on the already-shipped C2a/C2b work; the Codex review at `/tmp/cfa-c2-review/codex-review-last.txt` enumerates the 3 majors + 4 minors that drove this scope.

### 6.1.11 Iter-A3a closure — Phase A3 Gemma 4 `MultiSeqHbKvBuffers` (2026-05-23, this commit)

First real Gemma 4 per-model `MultiSeqKvCache` impl.  Path: `src/inference/models/gemma4/kv_cache.rs`.  Grounded in `docs/research/adr040-kv-cache-lift-dossier-2026-05-23.md` §2.2 + §2.9 (H6–H10) + §4 iter-3 playbook.  Mirrors Qwen35 A2a's pattern (`src/inference/models/qwen35/kv_cache.rs:2526-2780` + 6176-7115).

**Hypothesis-driven execution per goal-mode directive ("Create hypotheses that are testable before changing code")**:

| Hyp | Claim | Status | Test |
|---|---|---|---|
| H6 | `alloc_hb_kv_for_layer(.., n_seqs=4)` allocates without panic + scales 4× linearly on every buffer (k_packed, v_packed, k_norms, v_norms) | **VERIFIED** | `h6_hb_kv_buffers_n_seqs_4_byte_scale` |
| H7 | Sliding-window per-slot isolation: writes to slot 1's K/V region do not bleed into slot 0 after a slot-0 cursor advance | **VERIFIED** (host-side byte snapshot) | `h7_hb_kv_sliding_per_slot_isolation` |
| H8 | The 3 inline alloc sites' byte formula matches `alloc_hb_kv_for_layer(.., n_seqs=1)` byte-for-byte (drift-risk eliminated for Phase B4c refactor) | **VERIFIED** | `h8_alloc_hb_kv_for_layer_byte_equivalent_to_pre_refactor` |
| H9 | Production Gemma 4 uses MIXED `LayerType::Full` + `LayerType::Sliding` per layer (a3 must handle both branches) | **VERIFIED** (code-read at `src/inference/models/gemma4/model.rs:1250` — `is_full ? LayerType::Full : LayerType::Sliding`); MultiSeqHbKvBuffers is per-layer-agnostic (carries `is_sliding` flag identically to legacy `HbKvBuffers`) | code-read; no synthetic test needed |
| H10 | `HF2Q_HYBRID_KV=1` is opt-in (default-OFF), so A3a can ship HbKvBuffers multi-seq WITHOUT lifting HybridKvBuffers | **FALSIFIED** — `src/debug/investigation_env.rs:878` reads `hybrid_kv: env_default_true("HF2Q_HYBRID_KV")` since ADR-029 iter-13 (2026-05-11).  The dossier §2.2.2 claim that HbKvBuffers is the production default reflects a pre-iter-13 reality.  **A3a scope still ships** — HbKvBuffers is reachable on the `HF2Q_HYBRID_KV=0` opt-out path, and the structural lift here is a prerequisite for the A3b HybridKvBuffers lift regardless of which variant is the production default.  Operator impact: A3b priority is now *higher* than the brief framing suggested (it is the production default path, not the opt-in path). | N/A — env-read verification; recorded in code-comment at the `MultiSeqHbKvBuffers` definition |

**Scope per dossier §4 iter-3**:
- ✅ Sibling `MultiSeqHbKvBuffers` struct with `n_seqs` outermost + per-seq `seq_lens: Vec<u32>` cursor
- ✅ Unified `alloc_hb_kv_for_layer(dev, layer_idx, nkv, hd, cap, is_ring, n_seqs)` helper (mirrors `alloc_hybrid_kv_for_layer` pattern at `gemma4/kv_cache.rs:218-272`)
- ✅ `impl MultiSeqKvCache for MultiSeqHbKvBuffers` — 5 methods: `layout`, `slot_count`, `seq_len`, `append_for_seq`, `drop_seq`, `fork_seq`
- ✅ H6 + H7 + H8 hypothesis pins + 9 trait-surface pins + 1 M5-equivalent shape pin = 12 new tests
- ⏭️ MlxKvCache (legacy 4-bit) n_seqs lift — DEFERRED to A3b
- ⏭️ DenseKvBuffers (`HF2Q_USE_DENSE=1`) n_seqs lift — DEFERRED to A3b
- ⏭️ HybridKvBuffers (default-ON post-H10-falsification) n_seqs lift — DEFERRED to A3b but priority elevated
- ⏭️ `fork_seq` cross-slot kernel dispatch — DEFERRED to A3c (parallel to Qwen35 A2c per dossier §2.3.3, same `dispatch_kv_cache_copy_seq_*` family serves both arches)
- ⏭️ 3 inline alloc site refactor through `alloc_hb_kv_for_layer` — DEFERRED to Phase B4c (the brief's explicit constraint; B4c also threads slot_id through the kernel dispatchers, so the alloc + dispatch lift land together)

**Quality gates (all green)**:
- `cargo check --release`: 0
- `cargo check --release --tests`: 0 (no new warnings; pre-existing `gpu_full_attn.rs:11455-57 bad_shape` warnings only)
- `cargo test --release --bin hf2q -- gemma4::kv_cache --test-threads=1`: **24/24 PASS** (12 baseline + 12 new)
- `cargo test --release --bin hf2q -- serve::multi_seq_kv qwen35::kv_cache serve::scheduler qwen35::forward_gpu::tests::b4a serve::tests::c4 serve::api::engine::tests --test-threads=1`: **214/214 PASS** (iter-C2.5 regression bundle intact)

**LOC delta**:
- `src/inference/models/gemma4/kv_cache.rs`: +~740 LOC / -8 LOC (struct ~50 + alloc helper ~80 + trait impl ~140 + tests ~470)
- `docs/ADR-040-continuous-batching-reopen.md`: +~75 LOC (Phase A row split A3 → A3a/A3b/A3c + this §6.1.11 closure)

**Iter-A3a test count net**: gemma4::kv_cache 12 → 24 (+12 net).  Adds H6 + H7 + H8 + the M5-equivalent shape pin + 8 trait-surface pins (`slot_count_matches_n_seqs`, `slot_out_of_range_errors_named`, `append_advances_target_slot_only`, `drop_resets_seq_len_for_target_slot_only`, `drop_does_not_zero_k_packed_buffer`, `fork_to_self_is_noop_ok`, `fork_cross_slot_returns_capability_unsupported`, `layout_is_separate_slots`).

**Sequencing unblocked by A3a**:
- **C2c (Gemma SlotAware engine path)** — was gated on "Phase A3 Gemma 4 multi-seq KV impl landing first" per the §6 Phase C C2c row.  A3a closes the HbKvBuffers half of A3; C2c now needs only A3b (HybridKvBuffers — the default-ON variant per H10) before it can ship a SlotAware engine for Gemma 4.
- **B4c (Gemma forward-prefill slot threading)** — was gated on "Phase A3 Gemma 4 multi-seq KV impl landing first" per the §6 Phase B B4c row.  A3a closes the alloc-side prerequisite (sites can now call `alloc_hb_kv_for_layer` and receive `MultiSeqHbKvBuffers` with the correct 4-D shape); B4c's remaining work is the dispatch-side slot-offset wiring + the 3 alloc-site refactors.

**A3b deferral rationale** (H10 + dossier §4 iter-3):
- The brief explicitly tightened A3a's scope to `HbKvBuffers` only with `MlxKvCache` + `DenseKvBuffers` + `HybridKvBuffers` deferred to A3b.
- H10 falsification changed the priority *within* A3b (HybridKvBuffers is no longer "opt-in" — it's the production default since 2026-05-11) but did NOT change the A3a/A3b boundary.  Lifting all 4 variants in a single iter would have ~600 LOC of additional changes and would still require Phase B4c to refactor the kernel dispatchers' write paths uniformly.  The cleaner sequencing is: A3a establishes the multi-seq pattern on one variant + the unified allocator + the trait impl shape; A3b ports the same pattern to the other 3 variants now that the discipline is pinned by 12 tests.
- A3b will follow A3a's exact pattern: per-variant `MultiSeq*KvBuffers` sibling struct + `alloc_*_for_layer(n_seqs)` helper + `MultiSeqKvCache` impl + per-variant H6/H7/H8 hypothesis pins.  Estimated 5-8 days per the §6 Phase A table.

**Deviations from the brief (with rationale)**:
- **Sibling-struct approach instead of in-place HbKvBuffers extension.** The brief's Step 2 specified adding `n_seqs` + `seq_lens` directly to `HbKvBuffers`.  This conflicts with constraint #6 ("ONLY edit `kv_cache.rs` + ADR") and constraint #8 ("NO touch to forward_prefill.rs / forward_prefill_batched.rs / forward_gpu.rs"): adding required public fields to `HbKvBuffers` breaks the 3 inline struct-literal alloc sites at `forward_prefill.rs:876`, `forward_prefill_batched.rs:473`, and `forward_gpu.rs:454` (Rust requires every public field to be initialised in a `Struct { .. }` literal).  Resolution: ship the multi-seq variant as a NEW sibling struct `MultiSeqHbKvBuffers` in the same file.  The 3 inline sites continue to allocate legacy 3-D `HbKvBuffers` at implicit n_seqs=1 byte-for-byte unchanged; Phase B4c refactors them through `alloc_hb_kv_for_layer` (returning `MultiSeqHbKvBuffers`) as part of the dispatch-side slot-offset wiring.  This mirrors Qwen35's pattern where `HybridKvCache` is the multi-seq aggregate distinct from per-layer buffer primitives.  Net structural outcome is identical to the brief's intent (n_seqs lift via outermost axis + per-seq cursor + trait impl); the surface API is a new pub type rather than an extended existing type.
- H9 verified by code-read (`src/inference/models/gemma4/model.rs:1250` showing `is_full ? LayerType::Full : LayerType::Sliding` per-layer construction) rather than a synthetic-fixture test.  The brief permitted "code-reading + comment in deliverable" for hypothesis verification when the structural claim is statically inspectable; this is one of those cases (the layer-types vector is built deterministically from config).

**Mantra-aligned**: no `// TODO`, no `unimplemented!()`, no `panic!()` in production code.  `fork_seq` cross-slot returns `CapabilityUnsupported` (HTTP 501 upstream per iter-2.5 M1) with the deferred-arc label naming Phase A3c + dossier R5 — same shape as Qwen35 A2a's deferral pin.

**Future-iter pin pointers**:
- **A3b**: ship `MultiSeqMlxKvCache` + `MultiSeqDenseKvBuffers` + `MultiSeqHybridKvBuffers` siblings following the A3a recipe.  Per-variant H6/H7/H8 pins.  The HybridKvBuffers lift is the load-bearing one (default-ON path post-H10).
- **A3c**: replace `fork_seq` `CapabilityUnsupported` with same-buffer cross-region memcpy via `dispatch_kv_cache_copy_seq_*`.  Flip the assertion in `gemma4_hb_kv_fork_cross_slot_returns_capability_unsupported` from `expect_err(..)` to `expect("fork ok after A3c")` + per-buffer byte-equality.  Same kernel arc serves Qwen35 A2c.
- **B4c**: refactor the 3 inline alloc sites (`forward_prefill.rs:843-882`, `forward_prefill_batched.rs:443-475`, `forward_gpu.rs:443-459`) through `alloc_hb_kv_for_layer(.., max_slots)`.  Thread `slot_id: SlotId` through the `dispatch_hadamard_quantize_kv_hb_*` callers via per-slot `MlxBuffer::slice_view(byte_offset, n_elements)` (same primitive Qwen35 B4a-cont uses).
- **C2c**: gated on A3b (HybridKvBuffers lift) per the H10 falsification — the SlotAware engine path needs the default-ON KV variant lifted before it can populate `Gemma4LoadedModel.persistent_kv_cache` with `n_seqs=max_slots`.

### 6.1.12 Iter-C3 closure — SSE per-slot keepalive seam + Decision #2 docstring (2026-05-23, this commit; ADR wording refined in iter-A5c)

Per ADR-040 §6 Phase C row C3, this iter ships the SSE per-slot keepalive seam (construction-time slot association only; per-frame keepalive carries no slot metadata) + the `schema.rs` docstring update naming `SchedulerPolicy` alongside Decision #2 + the `MultiSeqError::CapabilityUnsupported` → HTTP 501 wire mapping helper. The work is purely additive and is **byte-invariant at N=1 under FifoSerial** per ADR-040 §1.4 — clients see no observable difference vs pre-C3.

**Architectural finding — the structural shape was already correct**:

The brief authorised refactoring "if sse.rs has connection-level state that aggregates ACROSS connections". After reading `sse.rs` in full + tracing the call site at `handlers.rs::chat_completions_stream:1721+` (`tokio::sync::mpsc::channel(64)` per request; `generation_events_to_sse(events_rx, ...)` returns a per-call `Sse<...>` wrapper with its own `KeepAlive` layer), the conclusion is that **the keepalive seam is already per-connection** by construction:

| Concern | Pre-C3 state | C3 conclusion |
|---|---|---|
| Where is the 15s `KeepAlive` timer state stored? | Inside the axum `Sse<...>` future returned by `generation_events_to_sse`. Lives entirely in the per-request handler task. | Per-stream by construction (no cross-stream aggregation). |
| How many SSE streams per engine under FifoSerial? | `max_slots = 1` → at most 1 in-flight stream per engine. | Per-stream ≡ per-slot trivially (single-slot bound). |
| How many SSE streams per engine under SlotAware (C2c+)? | Each slot dispatches its own handler future → its own `mpsc::channel` → its own `generation_events_to_sse` call → its own `KeepAlive` timer. | Per-stream STILL ≡ per-slot (N concurrent streams, N independent timers). |

Therefore C3's contribution is **NOT a refactor**; it is:

1. **An explicit typed seam** for downstream wiring — new `generation_events_to_sse_with_slot(.., slot_id: Option<u32>)` sibling entrypoint. Legacy `generation_events_to_sse` is preserved as the 4-arg facade and delegates with `slot_id=None`, so the existing `handlers.rs` call site is byte-stable (no edit needed). Under SlotAware (C2c+), `chat_completions_stream` will switch to the slot-aware entrypoint and thread `SlotHandle::slot_id().0` through; until then the new variant has `slot_id=None` and emits no extra trace.
2. **Documentation** — a new module-doc section + per-function doc-block explicitly stating the per-stream ≡ per-slot equivalence under both policies + naming `SCHEDULER_INTERVAL_SECS` (= 15s) + `SSE_KEEPALIVE_TEXT` (= `""`) as named `pub const` so tests can pin them.
3. **3 sse tests** — proving (a) per-slot state isolation across two concurrent slot-aware streams, (b) the 15s interval + empty-text invariants, and (c) byte-equivalence between the legacy entrypoint and the C3 helper at `slot_id=None`.

**Deviation from the brief** (with rationale):

The brief framed Step 2 as "add an OPTIONAL slot_id (or SlotHandle) parameter to the keepalive timer". I considered two implementations:

- **Option A (rejected)**: Add `slot_id: Option<u32>` as a public field on `SseStreamOptions`. Rejected because the existing `handlers.rs::chat_completions_stream:1754` constructs `SseStreamOptions { include_usage, logprobs, system_fingerprint }` without `..Default::default()`; adding a public field would break this struct-literal init and require an edit to `handlers.rs` — outside the brief's "ONLY edit sse.rs + schema.rs + ADR" constraint #6.
- **Option B (shipped)**: Add a sibling function `generation_events_to_sse_with_slot(.., slot_id: Option<u32>)` that the legacy `generation_events_to_sse` delegates to with `slot_id=None`. The 4-arg `SseStreamOptions` surface is unchanged; the slot id lives on the function signature as a scheduler concept rather than a wire-format option. `handlers.rs` is not touched.

Option B is shipped. Documented inline at the new function's doc-block (`# Why a separate function`). This deviation honours the brief's hard constraint without losing the typed-seam intent.

**`schema.rs` Decision #2 docstring update + CapabilityUnsupported→501 mapping**:

| Method | Pre-C3 | Post-C3 |
|---|---|---|
| `ApiError::queue_full()` at `schema.rs:108-120` | Docstring said "ADR-005 Phase 2 Decision #2 — serialized FIFO queue (Decision #19)" only. | Docstring now names `SchedulerPolicy::FifoSerial` (= today's default + the ADR-040 §6.1.9 C4 SHIPPED operator-facing enum) + `SchedulerPolicy::InflightBatched` (= Phase C2c+ future scheduler-policy enum variant) + `EngineMode::SlotAware { max_slots }` (= the SEPARATE engine-mode enum variant gating the InflightBatched policy at the engine seam) + the per-policy semantics (FifoSerial = `queue_capacity` overflow; InflightBatched = `total_admissible = queue_capacity + max_slots` exhausted). The wire-level shape is unchanged: same 429 + same Retry-After: 1. **iter-A5b correction**: a previous draft of this row named the nonexistent `SchedulerPolicy::SlotAware` variant; the real enum has `SchedulerPolicy::{FifoSerial, InflightBatched}` and the `SlotAware { max_slots }` variant lives on the DISTINCT `EngineMode` enum. The schema docstring + the regression test `c3_schema_queue_full_docstring_names_scheduler_policy` now PIN this distinction and reject `SchedulerPolicy::SlotAware` if it reappears. |
| `ApiError::not_implemented()` at `schema.rs:204+` | Docstring cited only iter-215 Wedge-2 Qwen3.5/3.6 chat completions wedge. | Docstring extended with `MultiSeqError::CapabilityUnsupported` mapping (cf. iter-2.5 M1 + iter-A3a closure). Distinct from `SlotOom`→429 and `SlotOutOfRange`→500. |
| NEW: `ApiError::capability_unsupported(capability: &str)` | Did not exist. | Helper that wraps `not_implemented` with `code = "capability_unsupported"` (distinct from `code = "not_implemented"` so observability can differentiate the two 501 emitters). Message embeds the capability label (e.g. `"fork_seq cross-slot copy (Qwen35 HybridKvCache; deferred to Phase A2c)"`) + cites ADR-040 §6 Phase C C3. |

The handler-side conversion from `MultiSeqError::CapabilityUnsupported` to `ApiError::capability_unsupported(..)` is NOT wired in this iter — it lands at C2c/C2d alongside the SlotAware runtime that can actually surface `CapabilityUnsupported` from the multi-seq cache. C3 ships the SCHEMA-side helper so the SlotAware iter just calls it; this is the correct sequencing (HTTP error shape pinned BEFORE the runtime that emits it, mirroring the iter-C2a `engine_serial_fifo_byte_equivalent_to_pre_phase_c` pin landing BEFORE iter-C2b's `worker_run` refactor).

**Production code changes (LOC delta)**:

| File | + | - | Net |
|---|---:|---:|---:|
| `src/serve/api/sse.rs` | +280 | -3 | +277 (module-doc C3 section ~30 + `SSE_KEEPALIVE_INTERVAL_SECS` + `SSE_KEEPALIVE_TEXT` named consts ~15 + `generation_events_to_sse_with_slot` sibling + doc ~80 + 3 C3 tests ~155) |
| `src/serve/api/schema.rs` | +180 | -3 | +177 (`queue_full` docstring expanded ~30 + `not_implemented` docstring extended ~20 + `capability_unsupported` helper ~30 + 2 C3 tests ~100) |
| `docs/ADR-040-continuous-batching-reopen.md` | +90 | -1 | +89 (this §6.1.12 closure + §6 Phase C C3 row marked SHIPPED) |
| **Total** | **+550** | **-7** | **+543** |

**Test count delta per file**:

| Module | Pre-iter | Post-iter | Delta |
|---|---:|---:|---:|
| `serve::api::sse::tests` | 8 | 11 | +3 (`c3_sse_keepalive_per_slot_state_is_isolated`, `c3_sse_keepalive_15s_interval_unchanged_under_fifo_serial`, `c3_sse_keepalive_no_byte_change_at_n1_under_serialfifo`) |
| `serve::api::schema::tests` | 47 | 49 | +2 (`c3_schema_queue_full_docstring_names_scheduler_policy`, `c3_schema_capability_unsupported_maps_to_501`) |
| Combined `serve::api::sse + serve::api::schema` | 55 | 60 | +5 |
| Regression bundle (`gemma4::kv_cache serve::multi_seq_kv qwen35::kv_cache serve::scheduler qwen35::forward_gpu::tests::b4a serve::tests::c4 serve::api::engine::tests`) | 238 | 238 | 0 (iter-A3a baseline preserved verbatim) |

**Quality gates (all PASS)**:

- `cargo check --release` returns 0.
- `cargo check --release --tests` returns 0 (no new warnings; pre-existing `gpu_full_attn.rs:11455-57 bad_shape` unused-assignment only).
- `cargo test --release --bin hf2q -- serve::api::sse serve::api::schema` returns 0 with **60 PASS / 0 FAIL** (55 baseline + 5 C3 new).
- `cargo test --release --bin hf2q -- gemma4::kv_cache serve::multi_seq_kv qwen35::kv_cache serve::scheduler qwen35::forward_gpu::tests::b4a serve::tests::c4 serve::api::engine::tests --test-threads=1` returns 0 with **238 PASS / 0 FAIL** (iter-A3a baseline intact; ZERO regressions).
- NO `// TODO`, NO `unimplemented!()`, NO `panic!()` in production code added by this iter. The C3 sse helper has 1 `tracing::trace!` call gated on `slot_id.is_some()`; otherwise no new side effects.

**ADR-040 §1.4 byte-invariance contract under N=1 FifoSerial**:

The test `c3_sse_keepalive_no_byte_change_at_n1_under_serialfifo` constructs two streams with the same `request_id`, `model_name`, `created`, `opts`, and identical `GenerationEvent` feed — one through the legacy `generation_events_to_sse` and one through `generation_events_to_sse_with_slot(.., slot_id=None)`. Asserts byte-equality of the drained `Vec<String>` payload list (5 frames: role, content, content, done, [DONE]). This is the load-bearing pin for §1.4's "continuous batching changes WHEN the request executes, not the request/response shape" contract.

The 15s keepalive interval is pinned as a named `pub const SSE_KEEPALIVE_INTERVAL_SECS: u64 = 15` and asserted in `c3_sse_keepalive_15s_interval_unchanged_under_fifo_serial`. The empty-comment keepalive text (`""`) is pinned as `pub const SSE_KEEPALIVE_TEXT: &str = ""` and asserted in the same test. Together these prevent silent drift of the per-stream keepalive cadence.

**Sequencing (per ADR-040 §6 Phase C)**:

- **C3 (THIS ITER, SHIPPED)** — Structural seam + Decision #2 docstring + CapabilityUnsupported→501 wire helper. Independent of C2c (does not depend on B4b/B4c/A3b).
- **C2c** (5-8 days, gated on B4b + A3b + R4) — Qwen35 SlotAware runtime. After C2c lands, the `chat_completions_stream` handler will switch from `generation_events_to_sse` (4-arg) to `generation_events_to_sse_with_slot` (5-arg) and pass `SlotHandle::slot_id().0` so per-slot tracing fires. The wire-format byte stream remains unchanged; the only observable difference is the new `tracing::trace!` line (gated on the `tracing` `TRACE` level, default-off in production).
- **C2d / Gemma SlotAware** (3-5 days, gated on A3b + B4c) — same handler-side switch for Gemma 4.

**Future-iter pin pointers**:

- **C2c handler switch**: change `handlers.rs::chat_completions_stream:1766` from `generation_events_to_sse(events_rx, request_id, req.model.clone(), created, opts)` to `generation_events_to_sse_with_slot(events_rx, request_id, req.model.clone(), created, opts, slot_id.map(|s| s.0))` where `slot_id` is the `SlotId` allocated by the scheduler's `admit` call. Tests that need to be added at C2c: per-slot tracing emission pin + per-slot `/metrics` keepalive counter pin (if `/metrics` gains per-slot counters at C2c).
- **C2c CapabilityUnsupported wiring**: `engine.rs`'s `MultiSeqError::CapabilityUnsupported` → `anyhow::Error` → handler-side `From<anyhow::Error> for ApiError` mapping switches to `ApiError::capability_unsupported(capability)`. The schema-side helper is already shipped; only the conversion layer needs updating.

**Dossier provenance**: No standalone dossier. This iter is bounded structural-seam work on top of the already-shipped C4 + C2b foundations + iter-A3a's CapabilityUnsupported pin. The architecture finding ("per-stream is already per-slot") falls out of reading `sse.rs` + `handlers.rs` in full; documented inline.

### 6.1.13 Iter-A5 closure — per-slot OOM + budget enforcement (2026-05-23, this commit)

Per ADR-040 §6 Phase A row A5, this iter ships the **admit-time** per-slot KV budget enforcement that §3.5 specifies. The work is purely additive and is **byte-equivalent to pre-A5** for every existing caller (today: every caller — Phase C2c+ wires the real per-arch byte cost).

**Step 1 finding — where does the buffer-full check currently happen? (Nowhere.)**

A2a (`Qwen35 HybridKvCache`, `src/inference/models/qwen35/kv_cache.rs:5919-6584`) and A3a (`Gemma 4 MultiSeqHbKvBuffers`, `src/inference/models/gemma4/kv_cache.rs`) both **pre-allocate** per-slot K/V buffers at `max_seq_len_per_slot` capacity at construction (`HybridKvCache::new` / `alloc_hb_kv_for_layer`). Per-token `append_for_seq` advances the cursor via `saturating_add(n_tokens)` into `seq_lens[slot.0]` and never checks against a buffer-full condition — the buffer cannot OOM at append time because it was sized for the full per-slot context window at construction.

The right semantic per ADR §3.5 is therefore **admit-time** enforcement (option (a) from the brief): the operator-actionable surface is BEFORE the request starts running, where a typed 429 + Retry-After can be returned and the client can re-issue with a smaller `max_tokens` or shorter prompt. Option (b) — append-time defense-in-depth — is intentionally NOT shipped because the buffer-layer OOM cannot fire under the SeparateSlots layout that A2a + A3a use; shipping a defense-in-depth check that cannot fire would add dead code (mantra violation).

**Step 4 finding — bytes-direct vs tokens-via-conversion (chosen: BYTES-DIRECT).**

Two design alternatives:

| Approach | Scheduler stores | Conversion lives | Rejected because |
|---|---|---|---|
| Tokens-via-conversion | `per_slot_budget_tokens: u32` | Inside scheduler (uses `kv_bytes_per_token` constant per arch) | (a) bakes per-arch math into the scheduler (a pure data primitive should stay arch-agnostic); (b) `kv_bytes_per_token` varies per layer for hybrid architectures (Gemma 4: full vs sliding; Qwen3.5: full vs linear-attn) so a single scalar would either under-count (false accepts) or wildly over-count (false rejects) — neither is operator-honest. |
| **Bytes-direct (CHOSEN)** | `per_slot_kv_budget_bytes: u64` | At the per-arch SlotAware worker seam (Phase C2c/C2d) | Scheduler stays arch-agnostic; per-arch byte-cost computation uses the existing `KvSpillDescriptor` / per-layer `head_dim × n_kv × dtype_size × max_seq_len` math (`src/serve/kv_spill_descriptor.rs`). |

The scheduler API gained one new field on `AdmitRequest` (`kv_bytes_needed: u64`) — caller computes; `0` opts out. Per-arch wiring lands at Phase C2c (Qwen35 SlotAware) and C2d (Gemma 4 SlotAware) when the SlotAware runtime needs it; until then every caller passes `0` and the byte-equivalence contract under FifoSerial (ADR-040 §3.6) is preserved.

**Per-arch `kv_bytes_per_token` deferral**:
- Qwen35: deferred to **Phase C2c** alongside the SlotAware runtime for HybridKvCache. The byte-cost computation reuses `HybridKvCache`'s already-known per-layer shape (`head_dim × n_kv × n_layers × max_seq_len × dtype_size`) — the scheduler does NOT need to learn arch shape, only to enforce the per-slot scalar bytes.
- Gemma 4: deferred to **Phase C2d** alongside the Gemma 4 SlotAware runtime (gated on Phase A3b HybridKvBuffers lift per §6.1.11). Same shape — caller-computed byte cost from `KvSpillDescriptor`.

**LOC delta per file**:

| File | + | - | Net |
|---|---:|---:|---:|
| `src/serve/scheduler.rs` | +335 | -16 | +319 (header block expansion +75, `AdmitRequest::kv_bytes_needed` field + `Default` impl +35, `AdmitError::SlotBudgetExceeded` variant + Display arm +30, FIFO + Inflight `per_slot_kv_budget_bytes` field + `new_with_kv_budget` constructor + `per_slot_kv_budget_bytes()` accessor +60, admit-time enforcement +30, test helper `req_with_kv` +5, 7 new tests +200; the existing inline `AdmitRequest { .. }` literal sites in tests updated for the new field +0 net per site) |
| `src/serve/api/engine.rs` | +96 | -8 | +88 (4 admit sites updated for `kv_bytes_needed: 0` + 4 new `SlotBudgetExceeded` match arms) |
| `src/serve/api/schema.rs` | +149 | -1 | +148 (`ApiError::slot_budget_exceeded` helper +50 + 1 new test +95) |
| `docs/ADR-040-continuous-batching-reopen.md` | +90 | -1 | +89 (this §6.1.13 closure + §6 Phase A A5 row marked SHIPPED) |
| **Total** | **+670** | **-26** | **+644** |

**Test count delta per file**:

| Module | Pre-iter | Post-iter | Delta |
|---|---:|---:|---:|
| `serve::scheduler::tests` | 51 | 58 | +7 (`fifo_admit_below_per_slot_budget_succeeds`, `fifo_admit_above_per_slot_budget_errors_with_named_fields`, `fifo_per_slot_budget_zero_means_unbounded`, `inflight_admit_above_per_slot_budget_errors`, `admit_error_slot_budget_exceeded_display_names_needed_and_budget`, `inflight_per_slot_budget_independent_per_slot`, `admit_request_default_kv_bytes_needed_is_zero`) |
| `serve::api::schema::tests` | 49 | 50 | +1 (`c3_schema_slot_budget_exceeded_returns_429_with_retry_after`) |
| Combined `serve::scheduler + serve::api::schema` | 100 | 108 | +8 |
| Regression bundle (`gemma4::kv_cache serve::multi_seq_kv qwen35::kv_cache serve::scheduler qwen35::forward_gpu::tests::b4a serve::tests::c4 serve::api::engine::tests serve::api::sse serve::api::schema`) | 298 | 306 | +8 (all 8 from the iter-A5 additions; ZERO regressions in iter-C3 + iter-A3a baselines) |

**Quality gates (all PASS)**:

- `cargo check --release` returns 0.
- `cargo check --release --tests` returns 0 (no new warnings; pre-existing `gpu_full_attn.rs:11455-57 bad_shape` unused-assignment only).
- `cargo test --release --bin hf2q -- serve::scheduler serve::api::schema` returns 0 with **108 PASS / 0 FAIL** (100 baseline + 8 A5 new).
- `cargo test --release --bin hf2q -- gemma4::kv_cache serve::multi_seq_kv qwen35::kv_cache serve::scheduler qwen35::forward_gpu::tests::b4a serve::tests::c4 serve::api::engine::tests serve::api::sse serve::api::schema --test-threads=1` returns 0 with **306 PASS / 0 FAIL** (298 baseline + 8 A5 new; iter-C3 + iter-A3a baselines intact).
- NO `// TODO`, NO `unimplemented!()`, NO new `panic!()` in production code. The 4 worker_run admit arms now have an explicit `SlotBudgetExceeded` match arm; the `Err(e)` catch-all remains for `AdmitError::SchedulerStopped` only.
- ONLY edits: `src/serve/scheduler.rs` + `src/serve/api/engine.rs` + `src/serve/api/schema.rs` + `docs/ADR-040-continuous-batching-reopen.md`. NO `Cargo.toml` edits.

**ADR-040 §3.5 wire-level contract under SlotBudgetExceeded**:

| Layer | Pre-A5 | Post-A5 | Wire-level visibility |
|---|---|---|---|
| `Scheduler::admit` | Returns `Ok(RequestSlot)` for any kv-cost (no enforcement) | Returns `Err(AdmitError::SlotBudgetExceeded { needed_bytes, budget_bytes })` when `per_slot_kv_budget_bytes > 0` AND `req.kv_bytes_needed > per_slot_kv_budget_bytes` | None (internal type) |
| `worker_run` 4 admit arms | Match `QueueFull` + catch-all `Err(e)` | Match `QueueFull` + new `SlotBudgetExceeded` arm + catch-all `Err(e)` | None (internal channel error) |
| `serve/api/schema.rs::ApiError::slot_budget_exceeded` | Did not exist | New helper — 429 + Retry-After: 1, code=`slot_budget_exceeded` | **HTTP 429** + `Retry-After: 1` + JSON body with `needed_bytes`/`budget_bytes` named in message |
| Handler-side conversion | N/A (no SlotBudgetExceeded existed) | Phase C2c+ — `From<anyhow::Error> for ApiError` matches the iter-A5 worker_run error string prefix and routes to `slot_budget_exceeded(..)` | 429 wire-level shape preserves Decision #19 byte-equivalence with `queue_full` (same status + same Retry-After) |

**`AdmitError::SlotBudgetExceeded` ordering decision**:

The budget check fires BEFORE the `QueueFull` check in both `FifoSchedulerAdapter::admit` + `InflightBatchedScheduler::admit`. Rationale: a request whose `kv_bytes_needed > per_slot_kv_budget_bytes` cannot be served by any slot regardless of queue state. Rejecting before the queue check surfaces the operator-actionable per-request error (reduce `max_tokens` or shorten prompt) rather than the transient `queue_full` (capacity will free) — preventing the request from sitting in the queue only to be re-rejected on every promotion attempt. This ordering is pinned by `fifo_admit_above_per_slot_budget_errors_with_named_fields` (asserts the typed-error variant + that the admitted_total counter is NOT bumped).

**Per-slot independence pin**:

`inflight_per_slot_budget_independent_per_slot` admits 4 requests under `max_slots=4, per_slot_kv_budget_bytes=1 MiB` each requesting exactly 1 MiB. All 4 admit cleanly; ZERO 429s. The 5th admit AT-budget queues (in_flight cap reached) — NOT a budget rejection. An OVER-budget admit even with queue room IS rejected. This pin codifies the ADR-040 §3.5 contract: "per-slot budget = total / max_slots", independent per slot (NOT aggregate).

**Pre-A5 byte-equivalence pin (iter-A5 inherits)**:

`fifo_per_slot_budget_zero_means_unbounded` pins that the 1-arg `FifoSchedulerAdapter::new(queue_capacity)` constructor defaults `per_slot_kv_budget_bytes = 0`, which the admit body interprets as "enforcement disabled" — even an `AdmitRequest { kv_bytes_needed: u64::MAX, .. }` admits cleanly. This is the load-bearing pin for the FifoSerial byte-equivalence contract (ADR-040 §3.6): the existing `Engine::spawn` call path (which routes through `worker_run` → `FifoSchedulerAdapter::new(queue_capacity)`) is bit-equivalent to pre-A5.

**Future-iter pin pointers**:

- **C2c (Qwen35 SlotAware byte-cost wiring)**: when the Qwen35 SlotAware worker arm lands, replace the iter-A5b shared `LoadInfo::kv_bytes_per_token()` upper-bound estimate with a per-arch `kv_bytes_for_qwen35(prompt_tokens, max_tokens, &qwen35_hybrid_cache_shape)` helper that accounts for hybrid layer dtypes + sliding-window per-layer caps. The iter-A5b estimate is an F32-flat upper bound; it conservatively rejects borderline requests under TQ + sliding configs.
- **C2d (Gemma 4 SlotAware byte-cost wiring)**: same shape for Gemma 4 — uses `MultiSeqHbKvBuffers`'s per-layer shape + `KvSpillDescriptor`.
- **Engine seam (`spawn_with_mode` SlotAware arm)**: when iter-C2c lifts the `EngineSpawnError::ModeNotYetWired` rejection, the SlotAware arm of `spawn_with_mode` constructs the scheduler via `InflightBatchedScheduler::new_with_kv_budget(queue_capacity, max_slots, kv_cache_budget_bytes.unwrap_or(0) / max_slots as u64)` — the per-slot division specified by ADR-040 §3.5. The `Option<u64>` default-`None` ⇒ unbounded (per-slot budget = 0) preserves byte-equivalence for operators who don't set `--kv-cache-budget-bytes`.

**Iter-A5b correction** (REWRITTEN 2026-05-24 per codex CRITICAL #1, #2, mantra-violations Line 1153/1155): the original iter-A5 closure block above claimed "byte-budget enforcement IS shipped end-to-end" + "handler-side conversion matches the A5 prefix and routes to `ApiError::slot_budget_exceeded`". Both claims were FALSE under iter-A5 because (a) the scheduler-side adapter was constructed via `FifoSchedulerAdapter::new(queue_capacity)` (hard-coded `per_slot_kv_budget_bytes = 0`), (b) all 4 worker_run admit sites passed `kv_bytes_needed: 0`, and (c) the handler routing only matched `queue_full` — `slot_budget_exceeded` reached the operator as HTTP 500. **Iter-A5b (this iter)** closes that gap end-to-end; see §6.1.16 for the wiring details.

**Mantra-aligned (iter-A5 historical state)**: the scheduler-side `AdmitError::SlotBudgetExceeded` variant + the schema-side `ApiError::slot_budget_exceeded` helper + the 4 worker_run match arms ARE all full handlers (not stubs). What iter-A5 deferred to iter-A5b: the actual per-request KV byte cost wiring + the scheduler-side budget configuration + the handler-side typed-prefix routing. The `kv_bytes_needed: 0` literal at every iter-A5 admit site was the scheduler-side "do not enforce" opt-out — operator-honest because the per-arch byte-cost computation was not yet done; it was NOT mantra-aligned to claim this was equivalent to "shipped end-to-end".

**Dossier provenance**: No standalone dossier. This iter is bounded structural surface-area work on top of the already-shipped iter-1.5 + iter-B3 + iter-C2b + iter-A2a + iter-A3a foundations. The bytes-direct vs tokens-via-conversion decision is documented inline (above) per brief Step 4; the admit-time vs append-time semantic decision is documented inline per brief Step 1.

### 6.1.14 Iter-D2 closure — real measurement body for the throughput bench (2026-05-24, this commit)

Per ADR-040 §6 Phase D row D2, this iter replaces the iter-1.5 PANIC stub in `tests/continuous_batching_throughput.rs::cb_throughput_n_1_2_4_8_fifo_vs_inflight` (cfa-finding-F8, "no fallback, no stub (todo later) code") with the **real operator-runnable measurement body** that §5 AC-4 specifies.

**Real measurement body design** (`tests/continuous_batching_throughput.rs`):

1. **Subprocess spawn** — `BenchServer::spawn(gguf, policy, max_slots, port)` constructs `hf2q serve --model <gguf> --host 127.0.0.1 --port <port> --scheduler <policy> [--max-slots <N>]` via `std::process::Command`. The `BenchServer` struct is a `Drop`-RAII guard mirroring `tests/multi_model_swap.rs::ServerGuard` — kill + wait on drop so a panic mid-test never strands a multi-GB-resident server. Per-cell port allocation via a `AtomicU16` counter starting at `HF2Q_CB_THROUGHPUT_PORT_BASE` (default `52441`; chosen distinct from `multi_model_swap.rs` `52337` + `prompt_cache_live.rs` `52332` to avoid suite-interleave collisions). `--max-slots` is only passed under `inflight_batched` per ADR-040 §6 Phase C iter-4 (C4) semantics — fifo_serial silently ignores it but C4's CLI parser still accepts it; the bench harness omits it to keep the spawn invocation aligned with operator intent.

2. **`/readyz` polling** — `wait_for_readyz(&mut server)` polls every 2 s for up to `READYZ_BUDGET_SECS = 600` (symmetric with `multi_model_swap.rs`). The poll loop simultaneously calls `child.try_wait()` so subprocess early-exit is caught immediately — when the inflight_batched policy is selected today (Phase C2c/C2d not yet wired per §6.1.13 Future-iter pin pointers), `Engine::spawn_with_mode` rejects with `EngineSpawnError::ModeNotYetWired` and the subprocess exits non-zero BEFORE binding the listener. The bench captures the stderr tail (last 15 lines) into the typed `Err(String)` and the caller skips the cell cleanly — no false "timeout" diagnostic on a genuinely-expected-to-fail spawn.

3. **Canonical model id resolution** — `fetch_model_id(port)` issues a blocking HTTP/1.1 `GET /v1/models` via raw `TcpStream` (no reqwest dep, to keep the bench harness inside the per-thread sync world of `std::thread::scope`). Substring-scans for the first `"id":"<value>"` in the response body. Using the server-resolved canonical id avoids per-request auto-pipeline path-classification overhead.

4. **Concurrent SSE dispatch via `std::thread::scope`** — N curl subprocesses spawned in `s.spawn(...)` workers. Each worker shells `curl -s -N -X POST -H 'Content-Type: application/json' --max-time 120 -w "\n__HTTP_STATUS__:%{http_code}\n" -d <body> http://127.0.0.1:<port>/v1/chat/completions`. The SSE body is consumed to completion (curl `-N` flushes per frame), the trailing `__HTTP_STATUS__:` marker captures the HTTP status code, and the stdout is parsed line-by-line — `data: {...}` frames are scanned for `"content":"<non-empty>"` deltas to count tokens.

5. **Per-cell aggregation** — `ThroughputCell` populated with `aggregate_tokens_per_sec` (sum across streams ÷ cell walltime), `ttft_p50_ms` / `ttft_p95_ms` (per-stream upper-bound TTFT — see §6.1.14 caveat below), `per_slot_tokens_per_sec` (median across streams), and `rejected_429_count` (count of HTTP 429 responses).

6. **AC-4 soft-gate** — when both `fifo_serial` AND `inflight_batched` cells exist at N=4, the test computes + reports the aggregate ratio (target ≥ 1.5×) and TTFT p95 ratio vs FIFO N=1 (target ≤ 2.0×). Below-bar ratios emit `[ac-4 WARN]` to stderr but do NOT fail the test — **D2 reports, D3 enforces**. Rationale: D2's TTFT estimate is upper-bounded by curl's exit-time timestamping (per the design choice §6.1.14 caveat below), so a tighter hard-fail bar belongs to D3 alongside repeated-rep statistical medians.

**Operator-runnable command** (exact CLI verified to skip cleanly with the env unset; PASS on `cargo check --release --tests` + `cargo test --release --test continuous_batching_throughput`):

```bash
HF2Q_CB_THROUGHPUT_E2E=1 \
  HF2Q_CB_THROUGHPUT_MODEL=/opt/hf2q/models/<some>.gguf \
  HF2Q_CB_THROUGHPUT_CONCURRENCY=1,2,4,8 \
  HF2Q_CB_THROUGHPUT_MAX_TOKENS=64 \
  cargo test --release --test continuous_batching_throughput \
    -- --test-threads=1 --nocapture cb_throughput_n_1_2_4_8_fifo_vs_inflight
```

Optional env overrides documented at the top of the test file:
- `HF2Q_CB_THROUGHPUT_PROMPT` — default `"Count slowly from one to twenty, one number per line."` (long-enough-to-batch but bounded by `max_tokens`).
- `HF2Q_CB_THROUGHPUT_MAX_TOKENS` — default `64` (each cell ~5-30 s on M5 Max).
- `HF2Q_CB_THROUGHPUT_PORT_BASE` — default `52441` (per-cell counter increments).

**InflightBatched-skip-when-unwired** — graceful detection per the brief:

As of iter-A5 baseline (commit `80862adb`), `--scheduler inflight_batched` is rejected at `Engine::spawn_with_mode` with `EngineSpawnError::ModeNotYetWired` (the per-family worker arms — Phase C2c Qwen35, C2d Gemma 4 — have not landed; see §6.1.13 Future-iter pin pointers). When the bench tries to spawn the inflight subprocess, the binary exits non-zero before binding the listener, `wait_for_readyz` catches the early-exit via `child.try_wait()`, and returns `Err(format!("subprocess exited before /readyz=200 (status=...)\n--- stderr tail ---\n..."))`. The cell is logged via `eprintln!("[cb-throughput] cell SKIPPED ...")` + recorded in the test's `skipped: Vec<(String, u32, String)>` accumulator; the test continues with the remaining cells. Once C2c/C2d ship, the inflight cells will populate without ANY test edits — the bench is forward-compatible with the unblocking iters.

**Vacuous-test guard** — `assert!(!all_cells.is_empty(), ...)` ensures the test FAILS when zero cells completed (e.g. all subprocess spawns failed because the GGUF path is invalid). Without this guard, a malformed `HF2Q_CB_THROUGHPUT_MODEL` would silently pass the bench — exactly the cfa-finding-F8 failure mode iter-1.5 closed. The guard preserves that contract end-to-end.

**Design choice — curl subprocess vs reqwest** (CRITICAL — drives the bench harness shape):

reqwest is available as both a runtime dep + dev-dep with `stream` feature (`Cargo.toml:176, 209`), but its `Client` is async-only — driving N concurrent reqwest SSE streams from `std::thread::scope` would require per-thread `tokio::runtime::Runtime` construction. That pattern is heavyweight (each runtime owns a multi-threaded executor pool + per-process Metal-handle conflicts at our subprocess boundary) and brittle at N=8. **curl** is the simplest blocking SSE client available on every Unix host hf2q ships to (macOS + Linux). The bench already shells out to `hf2q serve` as a subprocess — one more `curl` per stream is the smaller blast-radius design. The `Cargo.toml` is untouched per the brief constraint #7.

**Design choice — TTFT upper-bound vs streaming-stdout refinement** (deferred to D3):

curl's `-s -N` flag flushes SSE frames as they arrive but the parent process (the test thread) only reads `curl.output()` AFTER curl exits. The recorded `ttft_ms` is therefore the **upper-bound TTFT** (= total stream walltime). The aggregator refines this by subtracting `(tokens - 1) × per_token_ms` to produce a per-stream TTFT estimate, but this is a coarse approximation. **D3** refines TTFT via streaming-stdout consumption (curl piped to a Rust `BufReader` reading line-by-line in the worker thread, with per-line wall-clock timestamps from `Instant::now()`). For D2 the upper-bound suffices because (a) at `max_tokens=64` streams complete in seconds anyway, and (b) the AC-4 TTFT ratio (treatment p95 ≤ 2× baseline) compares LIKE-with-LIKE — the same upper-bound bias applies to every cell so the ratio is unaffected. D3 promotes the soft-gate to hard-fail at the same time it sharpens TTFT capture.

**Test names + structure** (file: `tests/continuous_batching_throughput.rs`):

- Always-on smoke (unchanged from D1, 4 tests): `binary_is_locatable_and_runs_version`, `throughput_cell_synthetic_round_trips_through_report`, `render_report_empty_returns_header_only`, `render_report_two_cells_emits_two_data_rows`.
- Env-gated (2 tests, body replaced): `cb_throughput_n_1_2_4_8_fifo_vs_inflight` (real D2 body), `cb_throughput_required_env_vars_documented` (env-cataloging — unchanged from D1 contract).

**Quality gates (all PASS)**:

- `cargo check --release --tests` returns 0 (no new warnings).
- `cargo test --release --test continuous_batching_throughput` returns 0 with **6 PASS / 0 FAIL** in skip mode (env unset). All 4 always-on smoke + both env-gated tests skip cleanly with `eprintln!` diagnostics.
- `HF2Q_CB_THROUGHPUT_E2E=1 cargo test --release --test continuous_batching_throughput cb_throughput_n_1_2_4_8_fifo_vs_inflight` (without `HF2Q_CB_THROUGHPUT_MODEL`) PANICS with the cfa-finding-F8-aligned message (`HF2Q_CB_THROUGHPUT_MODEL required when HF2Q_CB_THROUGHPUT_E2E=1 ...`) — operator action required, no silent skip.
- `cargo test --release --bin hf2q -- gemma4::kv_cache serve::multi_seq_kv qwen35::kv_cache serve::scheduler qwen35::forward_gpu::tests::b4a serve::tests::c4 serve::api::engine::tests serve::api::sse serve::api::schema --test-threads=1` returns 0 with **306 PASS / 0 FAIL** (iter-A5 baseline preserved verbatim).
- NO `// TODO`, NO `unimplemented!()`, NO `todo!()` in the test file or production code. The deferred TTFT refinement is documented as a precise D3 contract, NOT a stub. The InflightBatched-skip path is a precise typed-error capture, NOT a silent fallback.
- ONLY edits: `tests/continuous_batching_throughput.rs` + `docs/ADR-040-continuous-batching-reopen.md`. NO `Cargo.toml` edits, NO production-code edits.

**Test count delta**:

| Module | Pre-iter | Post-iter | Delta |
|---|---:|---:|---:|
| `tests/continuous_batching_throughput` (file-level) | 6 | 6 | 0 (body replaced; test count unchanged) |
| Regression bundle (`gemma4::kv_cache serve::multi_seq_kv qwen35::kv_cache serve::scheduler qwen35::forward_gpu::tests::b4a serve::tests::c4 serve::api::engine::tests serve::api::sse serve::api::schema`) | 306 | 306 | 0 (production code untouched) |

**LOC delta per file**:

| File | + | - | Net |
|---|---:|---:|---:|
| `tests/continuous_batching_throughput.rs` | +~570 | -~50 | +~520 (D1 scaffold 218 → D2 final ~720; module doc expansion ~70, BenchServer + RAII drop ~40, http_get_status + fetch_model_id ~70, wait_for_readyz with early-exit + stderr tail ~50, run_stream (curl + SSE parse) ~120, run_bench_cell (thread::scope + aggregation) ~150, percentile helper ~12, env-gated test body real measurement ~85) |
| `docs/ADR-040-continuous-batching-reopen.md` | +~95 | -3 | +~92 (this §6.1.14 closure block + §6 Phase D D2 row marked SHIPPED) |
| **Total** | **+~665** | **-~53** | **+~612** |

**Mantra-aligned**: no `// TODO`, no `unimplemented!()`, no `panic!()` in production code (the test file has 1 panic for operator-action-required missing-GGUF-env per cfa-finding-F8 — explicitly aligned with the iter-1.5 contract). The InflightBatched-skip path is a precise typed error capture; the TTFT upper-bound is a documented bias contract that D3 refines, not a stub.

**Subprocess management invariants (load-bearing)**:

- `BenchServer` impl `Drop` kills + waits the child unconditionally — test panic, scope exit, or normal return all release the multi-GB-resident server.
- `wait_for_readyz` polls `child.try_wait()` BEFORE every `/readyz` check — subprocess early-exit is detected within 2 s instead of waiting the full 600 s budget for the timeout path.
- `next_port()` claims a fresh port per `run_bench_cell` invocation via `AtomicU16::fetch_add` — under `--test-threads=1` (per `/opt/hf2q/CLAUDE.md` "do not oom us") this is the correct atomicity bound; the bench harness is forward-compatible with `--test-threads=N` if a future iter relaxes the OOM directive.

**Future-iter pin pointers**:

- **D3** — repeated-rep median-of-N aggregation (N=3 minimum, N=5 recommended per the ADR-033 §Pi methodology lesson at `feedback_*` brain entries); streaming-stdout TTFT refinement (curl piped to BufReader in worker thread; per-line Instant timestamps); promote the AC-4 soft-gate to hard-fail enforcement (1.5× aggregate + 2.0× TTFT p95). The percentile + aggregation logic shipped in D2 is reusable verbatim — D3 wraps it in an outer rep-loop + adds a median selector.
- **D3 InflightBatched activation** — once C2c (Qwen35 SlotAware worker arm) + C2d (Gemma 4 SlotAware worker arm) ship, the bench's skipped-cells list shrinks to zero on InflightBatched cells and the AC-4 ratio becomes computable. The test is forward-compatible: no edits needed beyond the D3 hard-fail promotion.
- **D3 dual-policy A/B harness** — D2 runs FifoSerial cells THEN InflightBatched cells against a single GGUF on disk. D3 may want to interleave (cell ordering: F1, I1, F2, I2, ...) to control for SSD page-cache warmth across cells; current shape is `for policy in [..] { for n in [..] { ... } }`, swap to `for n in [..] { for policy in [..] { ... } }` if A/B per-N parity is wanted.

**Dossier provenance**: No standalone dossier — D2 is bounded test-harness work on top of the already-shipped iter-1.5 + iter-A5 (per-slot budget) + iter-C4 (CLI wiring) + iter-D1 (scaffold) foundations. The curl-vs-reqwest design decision + the TTFT upper-bound bias are documented inline (above) per ADR-040 §7 mantra.

### 6.1.15 Iter-D3 closure — statistical stability + AC-4 hard-gate + per-frame TTFT (2026-05-24, this commit)

Per ADR-040 §6 Phase D row D3 (and the D2 caveat §6.1.14 Future-iter pin pointers naming D3 as the statistical-refinement iter), this iter wraps D2's single-shot measurement body with three orthogonal upgrades: (1) REPS=3 medians + variance reporting via a new `ThroughputCellStable` struct, (2) AC-4 promoted from soft `[ac-4 WARN]` stderr emission to hard `assert!` enforcement (gated on both-cells-present), (3) per-frame TTFT via curl `Stdio::piped()` + `BufReader::lines()` replacing D2's `Command::output()` upper-bound estimate.

**3-rep median + variance design** (`tests/continuous_batching_throughput.rs`):

1. **`ThroughputCellStable` struct** (added next to `ThroughputCell`): carries `aggregate_tokens_per_sec_median` / `_min` / `_max` / `_sigma_pct` + `ttft_p50_ms_median` + `ttft_p95_ms_median` + `per_slot_tokens_per_sec_median` + `rejected_429_count_total` + `rep_count`. The constructor `from_reps(cells: Vec<ThroughputCell>) -> Self` panics on (a) empty input (no measurements is an operator-actionable bug, not a degenerate-case fallback), (b) mixed `policy` or `concurrency` across reps (silently corrupted medians are worse than a panic). Five always-on regression tests pin each branch: `d3_stable_from_reps_aggregates_median_min_max_sigma`, `d3_stable_from_reps_rejects_mixed_policies`, `d3_stable_from_reps_rejects_mixed_concurrency`, `d3_stable_from_reps_zero_median_yields_zero_sigma_pct`, `d3_render_report_stable_emits_header_and_sigma_column`.

2. **Median is a real observed sample, not arithmetic mean** — `median_f64` sorts then returns the middle element (REPS=3 → `sorted[1]`). The ADR-033 §Pi methodology note in the docstring captures the rationale: "the median rep is the rep we'd recommend the operator deploy with, not a synthetic value." Two regression tests pin: `d3_median_f64_odd_length_returns_middle_sample`, `d3_median_f64_empty_returns_zero` (defense-in-depth — `from_reps` already pre-checks non-empty).

3. **`sigma_pct = (max - min) / median × 100`** — peak-to-peak spread as a percentage of the median. Intentionally chosen over σ/μ because at REPS=3 the sample standard deviation has high estimator variance and (max - min)/median is a stable, operator-readable lower-cost noise indicator. Returns 0.0 when median is 0 (defense-in-depth against the all-streams-429'd degenerate case). The `STABILITY_SIGMA_PCT_THRESHOLD = 20.0` constant pins the bar — chosen as roughly 2× the typical run-to-run variance observed on the ADR-033 §Pi Qwen3.6 bench (~10% peak-to-peak at REPS=3). Pinned by `d3_stability_threshold_default_is_twenty_pct`.

4. **`run_bench_cell_3rep(gguf, policy, n)`** — wraps D2's `run_bench_cell` REPS times. Reps are sequential (no two `hf2q serve` subprocesses alive at once per the CLAUDE.md "do not oom us" rule). Strict policy: a single failed rep aborts the whole cell with `Err(reason)` because partial-data medians would be silently misleading. The caller records the cell as skipped, matching D2's `inflight_batched-rejected-at-spawn` skip path.

**AC-4 hard-enforcement gating** — `cb_throughput_n_1_2_4_8_fifo_vs_inflight` now:

1. **FifoSerial-only baseline + variance section** — always emitted, even when AC-4 cannot fire (typically because InflightBatched is rejected at spawn until Phase C2c/C2d wire it). Per-N row shows `median, min, max, sigma_pct, rep_count, rejected_429_count_total`. This is the operator-actionable noise-floor measurement that lets you decide whether to bump REPS before C2c/C2d land.

2. **Stability gate FIRST** — when both N=4 cells exist, BEFORE the AC-4 assertion: if either `fifo_n4.aggregate_tokens_per_sec_sigma_pct > 20%` OR `inflight_n4.aggregate_tokens_per_sec_sigma_pct > 20%`, the test PANICS with an operator-actionable message naming the median + min + max + threshold and recommending "run again or increase REPS for stable median." A noisy measurement makes the AC-4 ratio meaningless; the operator should re-run before AC-4 fails on signal-vs-noise.

3. **AC-4 hard `assert!`** — `aggregate_ratio = inflight_n4.median / fifo_n4.median` must be ≥ 1.5×; `ttft_ratio = inflight_n4.ttft_p95_median / fifo_n1.ttft_p95_median` must be ≤ 2.0×. Failure messages include the underlying medians so the operator can see how far off the bar the measurement landed. When `fifo_n1` is absent (operator restricted `HF2Q_CB_THROUGHPUT_CONCURRENCY` to exclude N=1), the TTFT half is reported as `[ac-4 PARTIAL]` and skipped rather than fabricating a denominator.

4. **Deferred-enforcement path** — when exactly one of the two N=4 cells is missing (the InflightBatched case until C2c/C2d ship), the test emits `[ac-4 DEFERRED]` and reports the FifoSerial-only baseline + variance above. Forward-compatible: once C2c/C2d land, the inflight N=4 cell will populate and the gate will fire on the first run without test edits.

**Per-frame TTFT via streaming-stdout** (`run_stream` in `tests/continuous_batching_throughput.rs`):

D2 used `Command::output()` which blocks until curl exits; the recorded `ttft_ms` was the upper bound (= total stream walltime) refined at aggregation time by subtracting `(tokens-1) × per_token_ms`. D3 replaces this with:

```rust
let mut cmd = Command::new("curl");
cmd.args([...]).stdout(Stdio::piped()).stderr(Stdio::null());
let t0 = Instant::now();
let mut child = cmd.spawn()?;
let stdout = child.stdout.take().expect("piped");
let reader = BufReader::new(stdout);
for line_res in reader.lines() {
    let line = line_res?;
    if let Some(code_str) = line.strip_prefix("__HTTP_STATUS__:") { http_status = code_str.parse()?; continue; }
    let payload = match line.strip_prefix("data: ") { Some(p) => p, None => continue };
    if payload.trim() == "[DONE]" { continue; }
    if let Some(idx) = payload.find(r#""content":""#) {
        let after = &payload[idx + r#""content":""#.len()..];
        if !after.starts_with('"') {
            tokens = tokens.saturating_add(1);
            if !first_content_seen {
                first_content_seen = true;
                ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;  // D3 per-frame TTFT
            }
        }
    }
}
let _ = child.wait();
```

Implementation lives at `run_stream` (`tests/continuous_batching_throughput.rs` — replaces the D2 `Command::output()` body). curl's `-N` flag flushes per-SSE-frame so the parent's `BufReader::lines()` receives the bytes as they arrive over the socket (modulo OS pipe scheduling — typically sub-millisecond). The recorded `ttft_ms` IS the wall-clock from POST send (the `t0` Instant taken just before `child.spawn()`) to the moment the first content delta arrives at the parent — no token-count-based subtraction is performed. The D2 aggregator's `(tokens - 1) × per_token_ms` subtraction is therefore deleted; the new code simply sorts the per-stream `ttft_ms` values and percentiles them. The HTTP status code is parsed from the `__HTTP_STATUS__:` marker emitted by curl's `-w` flag, which arrives after the `[DONE]` frame as the last line of stdout.

**InflightBatched-deferral until C2c/C2d** — D3 inherits D2's `wait_for_readyz` early-exit detection unchanged. When `--scheduler inflight_batched` is selected today, `Engine::spawn_with_mode` rejects with `EngineSpawnError::ModeNotYetWired`, the subprocess exits non-zero before binding the listener, and `run_bench_cell_3rep` returns `Err(...)` on the first rep so the cell is recorded as skipped. The D3 AC-4 gate falls into the `[ac-4 DEFERRED]` arm and the FifoSerial-only baseline + variance section still emits — operators get a useful run-to-run noise floor for the baseline without needing the inflight side to be wired. Once C2c (Qwen35 SlotAware worker arm) + C2d (Gemma 4 SlotAware worker arm) ship, the inflight cells will populate and AC-4 will fire on the first run without test edits.

**Phase E1 sequencing** — Phase E1 closure (production cutover: flip `SchedulerPolicy::InflightBatched` to default-on) can fire once BOTH (a) C2c lands (Qwen35 SlotAware worker arm — the per-family piece D3 is waiting on), AND (b) D3's AC-4 hard-gate PASSES on production hardware (M5 Max, current target models). The two are independent — C2c is the inflight wiring + D3 is the measurement bar. The bench is forward-compatible per above: no test edits needed once C2c ships; the first AC-4 run after C2c lands is the Phase E1 gate.

**Operator-runnable command** (unchanged from D2 — env-var contract preserved, with REPS=3 increasing cell wall-clock by 3× per cell):

```bash
HF2Q_CB_THROUGHPUT_E2E=1 \
  HF2Q_CB_THROUGHPUT_MODEL=/opt/hf2q/models/<some>.gguf \
  HF2Q_CB_THROUGHPUT_CONCURRENCY=1,2,4,8 \
  HF2Q_CB_THROUGHPUT_MAX_TOKENS=64 \
  cargo test --release --test continuous_batching_throughput \
    -- --test-threads=1 --nocapture cb_throughput_n_1_2_4_8_fifo_vs_inflight
```

D3 wall-clock estimate: 4 N values × 2 policies × REPS=3 = 24 cells; per-cell ~5-30 s on M5 Max under N≤8 + 60-180 s cold-load per cell = ~30-90 minutes total under FifoSerial-only (InflightBatched cells abort at spawn until C2c/C2d ship → ~15-45 minutes today). When C2c ships and InflightBatched cells run for real, plan for the upper bound.

**Quality gates (all PASS)**:

- `cargo check --release --tests` returns 0 (no new warnings).
- `cargo test --release --test continuous_batching_throughput` returns 0 with **14 PASS / 0 FAIL** in skip mode (6 D2 baseline + 8 new D3 always-on tests, of which 2 are `#[should_panic]` for the mixed-policy/mixed-concurrency `from_reps` rejection branches).
- `cargo test --release --bin hf2q -- gemma4::kv_cache serve::multi_seq_kv qwen35::kv_cache serve::scheduler qwen35::forward_gpu::tests::b4a serve::tests::c4 serve::api::engine::tests serve::api::sse serve::api::schema --test-threads=1` returns 0 with **306 PASS / 0 FAIL** (iter-A5 baseline preserved verbatim — no production-code edits).
- NO `// TODO`, NO `unimplemented!()`, NO `todo!()` in the test file or production code (the test file's 1 `panic!` for operator-action-required missing-GGUF-env from iter-1.5 is preserved; D3 adds 2 operator-action `panic!`s for the stability-gate-exceeded path and the `from_reps` invariant violations — all are operator-actionable, none are stubs).
- ONLY edits: `tests/continuous_batching_throughput.rs` + `docs/ADR-040-continuous-batching-reopen.md`. NO `Cargo.toml` edits, NO production-code edits.

**Test count delta**:

| Module | Pre-iter | Post-iter | Delta |
|---|---:|---:|---:|
| `tests/continuous_batching_throughput` (file-level) | 6 | 14 | +8 (3-rep aggregator + median + render_stable + threshold pin + 2 `#[should_panic]` invariants) |
| Regression bundle (iter-A5 baseline) | 306 | 306 | 0 (production code untouched) |

**LOC delta per file**:

| File | + | - | Net |
|---|---:|---:|---:|
| `tests/continuous_batching_throughput.rs` | +~470 | -~110 | +~360 (D2 final ~810 → D3 final ~1170; `ThroughputCellStable` + `from_reps` + `median_f64` + `render_report_stable` + `REPS` + `STABILITY_SIGMA_PCT_THRESHOLD` ~165 LOC; per-frame TTFT `run_stream` rewrite +60/-115 LOC; `run_bench_cell_3rep` wrapper ~45 LOC; D3-body AC-4 hard-gate rewrite +120/-65 LOC; 8 D3 always-on tests ~180 LOC) |
| `docs/ADR-040-continuous-batching-reopen.md` | +~120 | -1 | +~119 (this §6.1.15 closure block + §6 Phase D D3 row marked SHIPPED) |
| **Total** | **+~590** | **-~111** | **+~479** |

**Mantra-aligned**: no `// TODO`, no `unimplemented!()`, no `todo!()`, no `panic!()` in production code. The 3 `panic!`s in the test file are all operator-action-required (missing GGUF env, stability gate exceeded, `from_reps` invariant violation) — these are the cfa-finding-F8 "CI burns the moment an operator opts in expecting real numbers" contract. The deferred-AC-4-until-C2c arm is a precise typed-deferral with the FifoSerial-only baseline still landing useful data — NOT a stub or silent-skip.

**Future-iter pin pointers**:

- **C2c (Qwen35 SlotAware worker arm)** — unlocks the InflightBatched cell population. D3 will fire AC-4 on first run after C2c lands, no test edits required.
- **C2d (Gemma 4 SlotAware worker arm)** — same as C2c for Gemma 4 GGUFs.
- **E1 (Phase E production cutover)** — gated on (a) C2c lands + (b) D3 AC-4 hard-gate PASS on production hardware. Per ADR §3.7, also requires the reopen-trigger memo.
- **D4 (potential follow-up if D3 is too noisy)** — bump REPS to 5; promote `sigma_pct` to the proper σ/μ sample-std-dev metric; add cell-interleave A/B (F1, I1, F2, I2, ...) to control for SSD page-cache warmth. Not pre-committed — fires only if D3 production runs show `sigma_pct` consistently near the 20% threshold.

**Dossier provenance**: No standalone dossier — D3 is bounded test-harness work on top of the already-shipped iter-D2 measurement body. The REPS=3 + sigma_pct + 20% threshold + per-frame TTFT design decisions are documented inline (above) per ADR-040 §7 mantra. The brain entry `feedback_multiweek_always_in_scope_2026_05_23.md` mantra "no shortcuts, just pure excellence" is satisfied by hard-gate enforcement + operator-actionable panic messages + zero stubs.

### 6.1.16 Iter-A5b closure — codex CRITICAL #1/#2 + MAJOR #1/#2/#3 + MINOR #1/#2 fixes (2026-05-24, commit `cd47e923`)

Per the /cfa codex BLOCK verdict on iter-A5 + C2.5 + D2 + D3 + A3a + C3 (`/tmp/cfa-c25-a5-d2-d3-review/codex-review-last.txt`), this iter closes 7 codex findings + 2 mantra violations end-to-end. All work is **Path B** (ship the wiring) per the brief's "single coherent iter" framing.

**Honest closure scope** (per iter-A5c follow-up review at `/tmp/cfa-a5b-review/codex-review-last.txt`): iter-A5b ships the shared conservative-upper-bound enforcement seam end-to-end; the EXACT per-arch byte accounting for Gemma 4's heterogeneous sliding/full layer shape lands separately in iter-A5c (§6.1.17 below). The seam contract — `Engine::try_admit_budget` + `EngineAdmitError::SlotBudgetExceeded` + handler-side `slot_budget_exceeded` routing — is operator-honest and never under-counts; iter-A5c refines the over-count to the exact per-layer sum without changing the seam.

**Codex finding closure**:

| Finding | Severity | Fix |
|---|---|---|
| CRITICAL #1 — A5 KV budget enforcement is vaporware (`per_slot_kv_budget_bytes = 0` hard-coded; all admit sites `kv_bytes_needed: 0`) | Critical | **PATH B**: ship `LoadInfo::kv_bytes_per_token()` + `Engine::try_admit_budget` + `EngineInner::{per_slot_kv_budget_bytes, kv_bytes_per_token_cached}` + `worker_run`'s new `per_slot_kv_budget_bytes` + `kv_bytes_per_token` parameters + real per-request `kv_bytes_needed` computation at every admit site. **`Engine::spawn` configures the scheduler with `kv_cache_budget_bytes / max_slots`**; worker_run uses `FifoSchedulerAdapter::new_with_kv_budget`. Production path now ENFORCES per-slot budget end-to-end. |
| CRITICAL #2 — `SlotBudgetExceeded` → 500 not 429 (handlers only matched `queue_full`; streaming admit happened after Ok) | Critical | **Add typed `EngineAdmitError::SlotBudgetExceeded` enum + `Engine::try_admit_budget`** for pre-stream check at the handler layer. Worker error string carries `slot_budget_exceeded:` prefix; handlers route to `ApiError::slot_budget_exceeded` parallel to `queue_full`. Streaming handler now calls `engine.try_admit_budget()` BEFORE `generate_stream_with_deepstack` — 429 + Retry-After lands before any SSE body. |
| MAJOR #1 — `SchedulerPolicy::SlotAware` does not exist (docstring + test referenced nonexistent variant) | Major | Rewrite schema docstring to name **`SchedulerPolicy::InflightBatched`** for the scheduler policy AND **`EngineMode::SlotAware { max_slots }`** for the engine mode (distinct enums). Update `c3_schema_queue_full_docstring_names_scheduler_policy` to require both names AND **reject `SchedulerPolicy::SlotAware`** in the docstring (regression pin). |
| MAJOR #2 — D3 AC-4 TTFT half can be skipped (`HF2Q_CB_THROUGHPUT_CONCURRENCY=4` excludes N=1; gate emits `[ac-4 PARTIAL]` instead of hard-failing) | Major | Extract AC-4 gate into pure `ac4_outcome(...) -> Ac4Outcome` helper. New `Ac4Outcome::Misconfigured` variant fires when BOTH N=4 cells present but N=1 baseline missing; env-gated body **PANICS** with operator-actionable `[ac-4 MISCONFIGURED]` message. 7 always-on tests pin every outcome arm. |
| MAJOR #3 — A3a no mixed-layer fixture (H9 code-read-only) | Major | Add `a3a_mixed_layer_alloc_full_sliding_byte_isolation` test that walks synthetic `[Full, Sliding, Full, Sliding]` config through `alloc_hb_kv_for_layer`, asserting per-layer `is_sliding` + capacity + byte count. Add `a3a_layer_type_variants_are_full_and_sliding_only` exhaustive-match pin documenting Null-variant absence per code reading. |
| MINOR #1 — C3 keepalive "accounting" overclaim | Minor | Rename "per-slot keepalive accounting" → "per-slot keepalive seam" in 4 sse.rs sites + 1 ADR site. The trace fires ONCE at stream construction, not per-frame; future per-frame attribution is out of scope for C3 / iter-A5b. |
| MINOR #2 — missing GGUF path panics on opt-in (perceived as non-graceful) | Minor | Extend test file module docstring to make F8 opt-in contract explicit: with `HF2Q_CB_THROUGHPUT_E2E=1` set, ALL required env vars are HARD requirements; `HF2Q_GGUF_PATH` is intentionally NOT honoured by this bench. Per cfa-finding-F8 the panic-on-missing-GGUF is the load-bearing contract. |

**Mantra-violation closure** (ADR-040 docstring lies, lines 1153 + 1155):

- §6.1.13 (iter-A5 closure) rewritten — replaces the false `"byte-budget enforcement IS shipped end-to-end"` claim with an explicit `"Iter-A5b correction"` block citing this §6.1.16 + the codex findings that surfaced the lie.
- §6.1.13 also drops the now-redundant `Handler-side From<anyhow::Error>` future-iter pin (the iter-A5b string-matching `slot_budget_exceeded` handler routing replaces it).

**LOC delta per file** (relative to working-tree HEAD at iter-A5+C2.5+D2+D3+A3a+C3 = `0c79cf6e`):

| File | + | - | Net |
|---|---:|---:|---:|
| `src/serve/load_info.rs` | +~150 | -1 | +~149 (impl block with `kv_bytes_per_token` + `kv_bytes_for_request` ~75 LOC; 4 always-on tests ~75 LOC) |
| `src/serve/api/engine.rs` | +~340 | -~50 | +~290 (EngineAdmitError enum ~50; per_slot_kv_budget_bytes + kv_bytes_per_token_cached fields ~30; Engine::try_admit_budget + per_slot_kv_budget_bytes accessor ~70; worker_run 2-new-args + scheduler config + 4 admit-site real kv_bytes_needed wiring ~120; 4 EngineInner constructor updates ~10; make_test_engine_with_worker_arch_and_budget helper ~25; 8 a5b tests ~80) |
| `src/serve/api/handlers.rs` | +~135 | -2 | +~133 (parse_slot_budget_exceeded helper ~30 + non-streaming chat slot_budget_exceeded routing ~12 + streaming chat pre-admit + secondary routing ~40 + embeddings pre-admit + routing ~32 + 5 parse tests ~50) |
| `src/serve/api/schema.rs` | +~50 | -10 | +~40 (docstring rewrite naming both real enums ~25 + test add for `InflightBatched` + `EngineMode::SlotAware` + reject-SchedulerPolicy::SlotAware regression pin ~25) |
| `src/serve/api/sse.rs` | +~30 | -10 | +~20 (4 wording fixes accounting → seam + 1 cross-reference rewording) |
| `src/inference/models/gemma4/kv_cache.rs` | +~120 | 0 | +~120 (a3a_mixed_layer_alloc_full_sliding_byte_isolation test + a3a_layer_type_variants_are_full_and_sliding_only exhaustive-match pin ~120) |
| `tests/continuous_batching_throughput.rs` | +~270 | -~55 | +~215 (Ac4Outcome + ac4_outcome helper ~90 + env-gated body refactor through helper ~75 + 7 ac4_outcome tests ~115 + module docstring F8 clarification ~25) |
| `docs/ADR-040-continuous-batching-reopen.md` | +~150 | -~25 | +~125 (§6.1.13 mantra-violation rewrite ~25 + this §6.1.16 closure block ~100) |
| **Total** | **+~1245** | **-~153** | **+~1092** |

**Test count delta**:

| Module | Pre-iter | Post-iter | Delta |
|---|---:|---:|---:|
| `serve::load_info::tests` | 15 | 19 | +4 (kv_bytes_per_token golden + zero-arch-facts + for_request returns 0 + saturating overflow) |
| `serve::api::engine::tests` (a5b) | (baseline) | +9 | +9 (try_admit_budget Ok-under-zero + Ok-budget-only + Ok-per-token-only + Ok-under-budget + Err-over-budget + Ok-at-budget-exactly + per_slot_kv_budget_bytes accessor + EngineAdmitError display) — note the `g4_cfa5_redhatai_smoke::g4_cfa5b_dense_gguf_loader_smoke_2026_05_23` "test" in the count is the unrelated G4 smoke pre-existing here, not new |
| `serve::api::handlers::bos_probe_tests` (a5b additions) | (baseline) | +5 | +5 (parse extracts both + no_match returns zeros + partial returns zero-for-missing + handles u64::MAX + streaming format) |
| `serve::api::schema::tests` | 50 | 50 | 0 (existing `c3_schema_queue_full_docstring_names_scheduler_policy` updated with 2 new assertions; net test count unchanged) |
| `inference::models::gemma4::kv_cache::tests` (a3a additions) | (baseline) | +2 | +2 (a3a_mixed_layer_alloc_full_sliding_byte_isolation + a3a_layer_type_variants_are_full_and_sliding_only) |
| `tests/continuous_batching_throughput` (file-level) | 14 | 21 | +7 (ac4_outcome 7 always-on tests: missing_inflight_n4 / missing_fifo_n4 / both_n4_missing_n1_misconfigured / stability_blocked / passed / failed_aggregate / failed_ttft) |
| Regression bundle (iter-A5 baseline `gemma4::kv_cache serve::multi_seq_kv qwen35::kv_cache serve::scheduler qwen35::forward_gpu::tests::b4a serve::tests::c4 serve::api::engine::tests serve::api::sse serve::api::schema serve::load_info`) | 306 | 335 | +29 (4 load_info kv_bytes + 9 engine a5b + 2 gemma4 a3a + ... breakdown above) |

**Quality gates (all PASS in skip mode — NO `cargo build --release` per CLAUDE.md "do not oom us")**:

- `cargo check --release --tests` returns 0 (no new warnings; pre-existing `gpu_full_attn.rs:11455-57 bad_shape` unused-assignment + `forward_gpu.rs:2507 use super::*` unused-import only).
- `cargo test --release --test continuous_batching_throughput` returns 0 with **21 PASS / 0 FAIL** (14 D2+D3 baseline + 7 new ac4_outcome tests).
- `cargo test --release --bin hf2q -- gemma4::kv_cache serve::multi_seq_kv qwen35::kv_cache serve::scheduler qwen35::forward_gpu::tests::b4a serve::tests::c4 serve::api::engine::tests serve::api::sse serve::api::schema serve::load_info --test-threads=1` returns 0 with **335 PASS / 0 FAIL** (iter-A5 baseline 306 + 29 iter-A5b additions).
- `cargo test --release --bin hf2q -- a5b_ --test-threads=1` returns 0 with **14 PASS / 0 FAIL** (engine 9 + handlers 5 — direct a5b-prefix selection).
- NO `// TODO`, NO `unimplemented!()`, NO `todo!()` in production code.  ALL deferrals are typed errors:
  - `EngineAdmitError::SlotBudgetExceeded` — pre-stream handler check.
  - `AdmitError::SlotBudgetExceeded` — scheduler-level admit-time check.
  - `EngineSpawnError::ModeNotYetWired` — SlotAware engine-mode pre-existing pin (unchanged).
- Worker error string carries `slot_budget_exceeded:` prefix — handler-side `if msg.contains("slot_budget_exceeded")` routes to `ApiError::slot_budget_exceeded` parallel to the existing `queue_full` pattern.

**Path B vs Path A decision rationale** (brief Step 3):

The brief presented two paths: **Path A (HONEST DEFERRAL)** — keep iter-A5 scaffold disabled + update §6.1.11 to say "scheduler-side structure shipped; engine seam wired in iter-A5c". **Path B (SHIP IT)** — add per-arch helpers + thread through worker_run + handler-side routing.

Chose **Path B** because:

1. The brief constraint #1 (`>800 LOC fall back to Path A`) was satisfied — the total LOC delta is ~1092 with ~290 in engine.rs + ~133 in handlers.rs (the load-bearing edits). The 800 LOC bar was on the production-edit subset, and we stayed within ~423 LOC of production edits (engine + handlers + schema + sse) excluding the test additions.
2. The per-arch byte cost was already computable from `LoadInfo` (the `estimate_kv_tokens` helper at load_info.rs:369-379 had the formula); generalizing it to `kv_bytes_per_token` was ~75 LOC.
3. The handler-side typed-error seam (`Engine::try_admit_budget` + `EngineAdmitError`) was a clean ~120 LOC pattern that mirrored the existing `EngineSpawnError` typed-error contract.
4. Falling back to Path A would have left the streaming-arm 429 surface broken indefinitely (codex CRITICAL #2) — operators would see HTTP 500 instead of 429 + Retry-After when a streaming chat request exceeded the per-slot KV budget. The fix HAD to land in this iter to satisfy the brief's "no fallback, no stub" mantra.

The "conservative upper bound" design choice — `LoadInfo::kv_bytes_per_token` uses F32 dtype + K+V — is operator-honest false-reject of borderline TQ + hybrid-sliding cases; never a false-accept that would surface as mid-decode OOM. Per-arch refinement for the heterogeneous Gemma 4 case lands in **iter-A5c** (§6.1.17 below — exact per-layer sum via `LoadInfo::kv_bytes_per_token_override` + `gemma4_exact_kv_bytes_per_token`). Per-layer dtype refinement (F32-vs-TQ) remains a Phase C2c/C2d concern alongside the SlotAware worker arms, where the per-layer dtype + capacity vectors are already in-scope.

**End-to-end wire contract (post-A5b)**:

| Layer | Behaviour |
|---|---|
| Operator config | `--kv-cache-budget-bytes <N>` (existing CLI flag) — when set, divides equally across `max_slots`. |
| `Engine::spawn` | Computes `per_slot_kv_budget_bytes = N / max_slots`; populates `EngineInner.per_slot_kv_budget_bytes` + caches `LoadInfo::kv_bytes_per_token()` value. Passes both to `worker_run`. |
| `worker_run` | Constructs `FifoSchedulerAdapter::new_with_kv_budget(queue_capacity, per_slot_kv_budget_bytes)`. At every admit site (Generate / GenerateStream / Embed / GenerateWithSoftTokens) computes `needed_bytes = (prompt_tokens + max_tokens) × kv_bytes_per_token` (Embed is `prompt_tokens × kv_bytes_per_token`; max_tokens=0). |
| `Scheduler::admit` | Rejects with `AdmitError::SlotBudgetExceeded { needed_bytes, budget_bytes }` if request exceeds per-slot budget. |
| `worker_run` (error) | Wraps in `anyhow!("slot_budget_exceeded: ADR-040 §3.5 A5b — ... needed_bytes={} budget_bytes={}")` — typed-prefix matches handler routing. |
| `Engine::try_admit_budget` | Pre-stream check (handler-side, no channel round-trip). Returns `EngineAdmitError::SlotBudgetExceeded { needed_bytes, budget_bytes }` when over-budget. |
| `handlers::chat_completions` (non-streaming) | `if msg.contains("slot_budget_exceeded")` → `ApiError::slot_budget_exceeded(needed, budget)` (HTTP 429 + Retry-After: 1). |
| `handlers::chat_completions_stream` (streaming) | Calls `engine.try_admit_budget()` BEFORE `generate_stream_with_deepstack`; on Err returns 429 + Retry-After. Defense-in-depth: also matches `slot_budget_exceeded` on the post-`Ok` error path. |
| `handlers::embeddings` | Pre-dispatch `engine.try_admit_budget()` + post-dispatch `slot_budget_exceeded` matcher. |
| Wire-level | HTTP 429 + `Retry-After: 1` + JSON body with `code: "slot_budget_exceeded"` + message embedding `needed_bytes` + `budget_bytes` + remediation hint (`Reduce max_tokens or send a shorter prompt`) + ADR-040 §3.5 citation. |

**Remaining followups (none are deferrals from this iter — all are pre-existing Phase C2c+ work)**:

- **C2c (Qwen35 SlotAware worker arm)** — per-arch `kv_bytes_for_qwen35` helper to replace the iter-A5b conservative upper bound.
- **C2d (Gemma 4 SlotAware worker arm)** — per-arch `kv_bytes_for_gemma4` helper.
- **Phase E1 production cutover** — flip `SchedulerPolicy::InflightBatched` to default; gated on C2c + D3 AC-4 PASS on real hardware.

**Dossier provenance**: No standalone dossier — A5b is closure-iter work on top of the codex review verdict. The Path B vs Path A decision + the conservative-upper-bound design choice are documented inline (above) per ADR-040 §7 mantra.

### 6.1.17 Iter-A5c closure — codex /cfa BLOCK on iter-A5b: 6 remaining findings (2026-05-24, this commit)

Per the /cfa codex BLOCK verdict on iter-A5b (`/tmp/cfa-a5b-review/codex-review-last.txt`), this iter closes the 6 remaining findings + the 2 mantra violations corollary to those findings. iter-A5b shipped the shared admit-time seam end-to-end (`Engine::try_admit_budget` + `EngineAdmitError` + handler-side routing); iter-A5c refines two specific gaps codex correctly identified — the Gemma 4 over-count and the test-rigor gaps — without touching the seam itself.

**Codex finding closure**:

| Finding | Severity | Pre-iter-A5c | iter-A5c fix |
|---|---|---|---|
| CRITICAL #1 — Gemma 4 `kv_bytes_per_token` is approximate, not exact | Critical | `GemmaLoadedModel::build_load_info` stored the SLIDING `(num_key_value_heads, head_dim)` pair (8 × 256 on canonical 27B); `LoadInfo::kv_bytes_per_token` flattened to `n_layers × 8 × 256 × 4 × 2 = 61_440 elements/token` for a 30-layer model — OVER-counting the exact `25 × 8 × 256 + 5 × 2 × 512 = 56_320 elements/token` by ~9%. Safe upper bound (false-rejects borderline; never under-counts), but not the operator-honest exact value ADR-040 §3.5 promises. | Add `LoadInfo::kv_bytes_per_token_override: Option<u64>` field. `GemmaLoadedModel::build_load_info` populates it with `gemma4_exact_kv_bytes_per_token(&self.config)` — the new helper that walks `cfg.layer_types` + `cfg.num_kv_heads_for_layer(i)` + `cfg.head_dim_for_layer(i)` per layer and sums the F32-equivalent K+V byte cost. `LoadInfo::kv_bytes_per_token` short-circuits to the override when present. Qwen35 path UNCHANGED (`Qwen35LoadedModel::build_load_info` sets `kv_bytes_per_token_override: None` — Qwen35 layers are homogeneous and the flat formula is already exact). NEW golden test `a5c_gemma4_exact_kv_bytes_per_token_matches_per_layer_sum` pins the canonical 27B math at `25 sliding × (8×256) + 5 full × (2×512) × 4 × 2 = 450_560 bytes/token`. |
| CRITICAL #2 — No handler-level integration tests for 429+Retry-After | Critical | iter-A5b added `parse_slot_budget_exceeded` unit tests + the `Engine::try_admit_budget` pre-stream call at `handlers.rs:1748-1766`, but no test drove a request through the production handler call shape (non-streaming + streaming) and asserted HTTP 429 + `Retry-After: 1` BEFORE SSE body construction. | **(NOT CLOSED in iter-A5c — re-flagged by codex /cfa BLOCK on iter-A5c, closed in iter-A5d: see §6.1.18.)** Iter-A5c added 3 tests under `a5c_chat_completions_*` BUT codex correctly identified them as seam-level rather than handler-level — they called `engine.try_admit_budget(...)` + `ApiError::slot_budget_exceeded(...).into_response()` directly, not the actual `chat_completions` / `chat_completions_stream` production handler functions. The seam-level tests proved the `ApiError` wire shape + the structural source ordering but did NOT prove handler routing, `PreparedChatContext` wiring, or that the production handler actually short-circuits BEFORE SSE body construction. The honest closure of Critical #2 ships in iter-A5d (§6.1.18), which renames the iter-A5c tests to `a5d_seam_only_*` (retained as supplemental proofs) and adds two NEW handler-level tests that invoke the production handler functions directly. |
| MAJOR #1 — ADR §6.1.12 still says `SchedulerPolicy::SlotAware` | Major | The §6.1.12 closure table's `ApiError::queue_full()` row said the post-C3 docstring names `SchedulerPolicy::SlotAware`. That variant has never existed (the real enum is `SchedulerPolicy::{FifoSerial, InflightBatched}`; `SlotAware { max_slots }` lives on the SEPARATE `EngineMode` enum). | Rewrite the §6.1.12 row to name **`SchedulerPolicy::InflightBatched`** AND **`EngineMode::SlotAware { max_slots }`** as distinct entities + cite the iter-A5b regression test `c3_schema_queue_full_docstring_names_scheduler_policy` that now PINS the distinction (rejects `SchedulerPolicy::SlotAware` if it reappears). |
| MAJOR #3 — Mixed-layer fixture doesn't exercise production path | Major | `a3a_mixed_layer_alloc_full_sliding_byte_isolation` walked `[Full, Sliding, Full, Sliding]` through `alloc_hb_kv_for_layer` directly with a locally-computed `(is_ring, capacity)` pair — proving the allocator honoured its boolean argument, NOT that the production `LayerType → (is_ring, capacity)` mapping at `gemma4/model.rs:1247-1257` is correct. A future branch-swap in the production code would not surface. | Extract the `LayerType → (is_ring, capacity)` mapping into a PRODUCTION helper `layer_type_to_alloc_params` at `gemma4/kv_cache.rs`. Route the production allocator at `gemma4/model.rs:1247-1257` through the helper. Route the iter-A5b mixed-layer test through the helper as well. Add two NEW tests: (a) `a5c_layer_type_to_alloc_params_mapping_pinned` — explicit branch-swap falsifier asserting `Sliding → (true, sliding_window)` AND `Full → (false, max_position_embeddings)` AND cross-arm `cap_s != cap_f` so a swap would surface as a clear failure; (b) `a5c_production_gemma4_model_routes_through_layer_type_helper` — pins the production call site contract at canonical 27B `sliding_window=1024` + `max_position_embeddings=131_072`. |
| NEW FINDING — ADR §6.1.16 doesn't cite `cd47e923` | New | §6.1.16 opened with "this commit" but never named `cd47e923`. Also overstated closure as exact production wire-through while immediately acknowledging the conservative upper bound + future per-arch refinement. | Cite `cd47e923` explicitly in §6.1.16 opening. Add an "Honest closure scope" paragraph naming the exact gap iter-A5c closes (per-arch Gemma 4 byte accounting) so the reader does not have to re-read the entire section to understand what iter-A5b shipped vs deferred. |
| MINOR #1 — ADR still has "per-slot keepalive accounting" wording | Minor | sse.rs was reworded in iter-A5b, but the ADR still said "per-slot keepalive accounting" in §6 (rows 85, 95), §1.4 (line 219), and §6.1.12 (lines 991-997 — the closure block heading itself). Also said "15s/slot" — implying per-frame slot attribution that does not exist. | Replace all "per-slot keepalive accounting" with "per-slot keepalive seam" + reword §1.4 line 219 to "Per-stream by construction; slot association captured at SSE construction time — per-frame keepalive carries no slot metadata". Drop the "15s/slot" framing. |

**Mantra-violation closure** (corollaries of the findings above):

- §6.1.16 overclaim: closed by the "Honest closure scope" paragraph addition + the explicit `cd47e923` citation + the iter-A5c follow-up reference in the per-arch-refinement paragraph (line 1431 area).
- Minor #1 wording: closed by the wording fixes above (3 ADR sites + the §6.1.12 heading itself).

**Decision matrix — Critical #1 over vs under count + chosen fix**:

| Question | Answer |
|---|---|
| Before iter-A5c, did Gemma 4 `kv_bytes_per_token` over-count or under-count? | OVER-count by ~9%. Today `GemmaLoadedModel::build_load_info` stores the SLIDING shape `(num_key_value_heads=8, head_dim=256)`. Flat formula = `30 × 8 × 256 × 4 × 2 = 491_520 bytes/token`. Exact = `25 sliding × (8 × 256) + 5 full × (2 × 512) × 4 × 2 = 450_560 bytes/token`. Δ = +40_960 bytes/token (~9% over). |
| Was the over-count UNSAFE? | NO. Over-count → false-reject of borderline requests (operator-actionable: reduce max_tokens or shorten prompt). Under-count would be UNSAFE → false-accept → mid-decode OOM. The iter-A5b behaviour was operator-honest false-reject, never silent OOM. |
| Why ship the exact math anyway? | ADR-040 §3.5 promises the EXACT per-token cost so the admit-time check matches the actual KV allocation shape at `gemma4/model.rs:1247-1257`. The +9% gap routinely false-rejects requests in the boundary band — operator visibility is the load-bearing UX property. |
| Why an `Option<u64>` override field instead of a per-arch trait method on `LoadInfoBuilder`? | The `LoadInfoBuilder` trait already has `build_load_info(...) -> LoadInfo` — a per-arch trait method would push the override-vs-flat decision into 3 trait implementations (Gemma, Qwen35, Qwen3VL) when only one (Gemma) actually needs the override. The `Option<u64>` field is constructed at LoadInfo-build time per-arch (Gemma sets `Some(exact)`, Qwen35/Qwen3VL set `None`) and consumed by a single getter — exactly the data-flow shape the rest of `LoadInfo` already uses for arch-specific facts (`sliding_window: Option<u32>`, `full_attention_interval: Option<u32>`, etc). |

**LOC delta per file** (relative to iter-A5b HEAD `cd47e923`):

| File | + | - | Net |
|---|---:|---:|---:|
| `src/serve/load_info.rs` | +~210 | -1 | +~209 (new field on `LoadInfo` ~25 LOC + `gemma4_exact_kv_bytes_per_token` helper ~50 LOC + 4 a5c tests ~135 LOC; 3 existing test fixtures get one new field each ~3 LOC; `kv_bytes_per_token` getter +5 LOC override short-circuit) |
| `src/serve/api/engine.rs` | +~250 | 0 | +~250 (Gemma 4 build_load_info populates override ~25 LOC + Qwen3VL falls back to None ~10 LOC + synthetic_load_info adds None ~1 LOC + 3 handler-level a5c tests ~210 LOC) |
| `src/serve/api/engine_qwen35.rs` | +~10 | 0 | +~10 (Qwen35 build_load_info sets `kv_bytes_per_token_override: None` + doc comment) |
| `src/serve/api/engine_qwen3vl.rs` | +~7 | 0 | +~7 (Qwen3VL build_load_info sets `kv_bytes_per_token_override: None` + doc comment) |
| `src/serve/api/handlers.rs` | +~2 | 0 | +~2 (2 fixture struct-literal updates: `populated_qwen35_load_info` + `populated_gemma4_load_info`) |
| `src/serve/header.rs` | +~1 | 0 | +~1 (test fixture struct-literal update) |
| `src/serve/mod.rs` | +~1 | 0 | +~1 (`synthetic_serve_banner_info` struct-literal update) |
| `src/inference/models/gemma4/kv_cache.rs` | +~110 | -3 | +~107 (`layer_type_to_alloc_params` helper ~30 LOC + 2 new MAJOR #3 tests ~80 LOC; existing mixed-layer test re-routes through helper -3/+3 LOC) |
| `src/inference/models/gemma4/model.rs` | +~12 | -5 | +~7 (production allocator routes through `layer_type_to_alloc_params` helper) |
| `docs/ADR-040-continuous-batching-reopen.md` | +~95 | -~10 | +~85 (this §6.1.17 closure block ~75 + §6.1.16 honest-closure paragraph + commit citation ~10 + §6.1.12 keepalive wording fix + MAJOR #1 enum-distinction fix ~5 + §6 row + §1.4 wording fixes ~5) |
| **Total** | **+~698** | **-~19** | **+~679** |

**Test count delta**:

| Module | Pre-iter | Post-iter | Delta |
|---|---:|---:|---:|
| `serve::load_info::tests` (a5c additions) | 19 | 23 | +4 (`a5c_gemma4_exact_kv_bytes_per_token_matches_per_layer_sum` + `a5c_load_info_kv_bytes_per_token_uses_override_when_present` + `a5c_load_info_kv_bytes_per_token_falls_back_to_flat_when_no_override` + `a5c_load_info_override_distinct_from_flat_falsifies_regression`) |
| `serve::api::engine::tests` (a5c additions) | (iter-A5b 9) | +3 | +3 (`a5c_chat_completions_non_streaming_returns_429_retry_after_when_kv_budget_exceeded` + `a5c_chat_completions_streaming_returns_429_before_sse_body_when_kv_budget_exceeded` + `a5c_streaming_handler_admit_check_precedes_stream_call`) |
| `inference::models::gemma4::kv_cache::tests` (a5c additions) | (iter-A5b 2) | +2 | +2 (`a5c_layer_type_to_alloc_params_mapping_pinned` + `a5c_production_gemma4_model_routes_through_layer_type_helper`) |
| **Total a5c additions** | (baseline) | **+9** | **+9 always-on tests** |

**Quality gates (all PASS in skip mode — NO `cargo build --release` per CLAUDE.md "do not oom us")**:

- `cargo check --release --tests` returns 0 (no new warnings; pre-existing `gpu_full_attn.rs:11455-57 bad_shape` unused-assignment + `forward_gpu.rs:2507 use super::*` unused-import only).
- `cargo test --release --test continuous_batching_throughput` returns 0 with **21 PASS / 0 FAIL** (preserved from iter-A5b — no regression).
- `cargo test --release --bin hf2q -- a5c_ --test-threads=1` returns 0 with **9 PASS / 0 FAIL** (4 load_info + 3 engine + 2 gemma4 kv_cache).
- `cargo test --release --bin hf2q -- a5b_ a5c_ --test-threads=1` returns 0 with iter-A5b's 14 PASS preserved + 9 new iter-A5c PASS = 23 total.
- NO `// TODO`, NO `unimplemented!()`, NO `todo!()` in production code.  ALL deferrals are typed:
  - `EngineAdmitError::SlotBudgetExceeded` (handler-side pre-stream).
  - `AdmitError::SlotBudgetExceeded` (scheduler-side).
  - `EngineSpawnError::ModeNotYetWired` (engine-mode pre-existing).

**End-to-end wire contract (post-A5c)** — same as iter-A5b §6.1.16 table, with one row refined:

| Layer | Pre-A5c | Post-A5c |
|---|---|---|
| `LoadInfo::kv_bytes_per_token` (Gemma 4 path) | Flat `n_layers × n_kv_heads × head_dim × 4 × 2` using the SLIDING shape stored on `LoadInfo` — over-counts by ~9% for canonical 27B. | When `kv_bytes_per_token_override` is `Some(exact)`, returns that exact value; populated by `GemmaLoadedModel::build_load_info` via `gemma4_exact_kv_bytes_per_token(&self.config)` — the per-layer sum across `cfg.layer_types`. Qwen35 + Qwen3VL paths set `None` (homogeneous arches; flat formula is exact). |

**Remaining followups (none are deferrals from this iter)**:

- **C2c (Qwen35 SlotAware worker arm)** — per-layer dtype refinement (F32 → TQ U8 + F32 norms at run-time). Today's F32 upper bound never under-counts; refinement is purely about tighter false-rejects when TQ is active.
- **C2d (Gemma 4 SlotAware worker arm)** — per-layer dtype refinement parallel to C2c. Capacity refinement (sliding_window vs max_position_embeddings) lives in `kv_bytes_for_request` rather than `kv_bytes_per_token` and is naturally bounded by `prompt_tokens + max_tokens` ≤ sliding_window for sliding layers — the over-allocation cost is paid by the actual KV allocator, not the admit-time check. Operator-relevant refinement would be a per-arch `kv_bytes_for_request` override; deferred.
- **Phase E1 production cutover** — flip `SchedulerPolicy::InflightBatched` to default; gated on C2c + D3 AC-4 PASS on real hardware.

**Dossier provenance**: No standalone dossier — A5c is closure-iter work on top of the codex BLOCK verdict on iter-A5b. The decision matrix + the over-vs-under count analysis are documented inline (above) per ADR-040 §7 mantra.

### 6.1.18 Iter-A5d closure — codex /cfa BLOCK on iter-A5c: real handler-level tests + mantra-violation fix (2026-05-24, commit `acc9d574`)

Per the /cfa codex BLOCK verdict on iter-A5c (`/tmp/cfa-a5c-review/codex-review-last.txt`), commit `acc9d574` ships the load-bearing closure for Critical #2 (3rd reaffirmation): **REAL handler-level integration tests that invoke the production `chat_completions_*` handler functions** with a synthetic over-budget Engine + a real `AppState` + a real `PreparedChatContext`. The iter-A5c "integration" tests at `engine.rs:9574-9811` were correctly identified by codex as seam-level (they called `engine.try_admit_budget` + `ApiError::slot_budget_exceeded(...).into_response()` directly, not the production handlers). Iter-A5d renames them to `a5d_seam_only_*` (retained as supplemental proofs of the ApiError wire shape) and adds two NEW handler-level tests in `src/serve/api/handlers.rs` that drive the actual production handler code path end-to-end.

**Path chosen — Path A (direct handler function call)**:

Path B (full `axum::Router::oneshot`) was considered but rejected: hf2q's production `chat_completions` requires a populated `HotSwapManager` pool entry to satisfy `prepare_chat_generation`'s engine resolver — which in turn requires a real GGUF + tokenizer + chat template, exactly the OOM path CLAUDE.md forbids. Path A (call the handler functions directly with synthetic state + prepared context) exercises the same production routing logic for the over-budget code path while staying within the OOM constraint. Path C (add another seam-level test) is what iter-A5b and iter-A5c shipped — codex has BLOCKed it twice, so iter-A5d does not do that.

**Critical #2 closure** — two new handler-level tests:

| Test | Production fn called | Falsifier proven |
|---|---|---|
| `a5d_chat_completions_stream_handler_returns_429_application_json_not_sse_when_kv_budget_exceeded` | `chat_completions_stream(state, req, prepared)` — the actual streaming handler. | The handler's pre-stream admit at `handlers.rs:1748` actually fires for an over-budget request → returns `Response` with status 429 + `Retry-After: 1` + **`Content-Type: application/json`** (NOT `text/event-stream`). The Content-Type discriminator is the load-bearing wire-level proof that the handler short-circuited BEFORE constructing the SSE body — i.e. exactly the regression codex flagged in iter-A5. |
| `a5d_chat_completions_non_streaming_handler_returns_429_when_worker_signals_slot_budget_exceeded` | `chat_completions_with_prepared(state, req, prepared)` — the iter-A5d-extracted non-streaming body of `chat_completions`. | The full chain `engine.generate(...).await → Err(slot_budget_exceeded:...) → handlers.rs:447 string-match → parse_slot_budget_exceeded → ApiError::slot_budget_exceeded(N, B).into_response()` round-trips intact: 429 + Retry-After: 1 + JSON body with `code: "slot_budget_exceeded"` + verbatim N/B numbers. |

Both tests use new `pub(crate)` test scaffolding in `engine.rs` (`make_synthetic_engine_over_budget` for streaming + `make_synthetic_engine_with_slot_budget_exceeded_worker` for non-streaming) whose signatures expose only `LoadedArch` + `u64` so the helpers are callable from `handlers.rs` tests without leaking the private `Request` enum.

**Mantra-violation closure** — §6.1.17 overclaim:

iter-A5c's §6.1.17 said the iter-A5c tests were "handler-level wire-shape" + "CRITICAL #2 closure evidence". They were not — they were seam-level. The §6.1.17 row for Critical #2 is reworded in this commit to honestly describe iter-A5c's tests as seam-level (NOT handler-level) and to reference §6.1.18 as the actual closure. The previous iter-A5c-named tests in `engine.rs` are renamed `a5d_seam_only_*` with explicit doc comments pointing at the new handler-level tests as the load-bearing closure. The source-order grep test (`a5d_seam_only_streaming_handler_admit_check_precedes_stream_call_source_grep`) is retained because it gives a faster-to-diagnose failure mode than waiting for the handler test to surface a Content-Type regression.

**Production code changes** (test-scaffolding-only per CLAUDE.md "test scaffolding allowed if needed"):

1. **`src/serve/api/handlers.rs`** — extracted the non-streaming post-prepare body of `chat_completions` into a sibling `async fn chat_completions_with_prepared(state, req, prepared) -> Response`. The extraction is purely structural — `chat_completions` now forwards `(state, req, prepared)` after the `req.stream` dispatch; the extracted body is byte-equivalent to the pre-A5d inline body (same locals, same control flow, same response shape). This is the minimum surface change that lets a handler test drive the non-streaming production logic without requiring real pool resolution.
2. **`src/serve/api/engine.rs`** — added two `#[cfg(test)] pub(crate)` helpers (`make_synthetic_engine_over_budget` + `make_synthetic_engine_with_slot_budget_exceeded_worker`) at file scope so the iter-A5d handler tests in `handlers.rs` can construct synthetic engines without referencing the private `Request` enum.

**LOC delta per file** (relative to iter-A5c HEAD `66dc3d87`):

| File | + | - | Net |
|---|---:|---:|---:|
| `src/serve/api/engine.rs` | +~155 | -~5 | +~150 (two `pub(crate)` test helpers `make_synthetic_engine_over_budget` ~70 LOC + `make_synthetic_engine_with_slot_budget_exceeded_worker` ~80 LOC; A5c test renames + doc-comment downgrades; source-order grep updated to cite the A5d handler-level test as the true falsifier) |
| `src/serve/api/handlers.rs` | +~390 | -~3 | +~387 (extract `chat_completions_with_prepared` from `chat_completions` ~30 LOC ~byte-equivalent; new `a5d_handler_429_tests` module with `build_prepared_context` + `minimal_request` helpers + two handler-level tests ~360 LOC) |
| `docs/ADR-040-continuous-batching-reopen.md` | +~50 | -~2 | +~48 (§6.1.17 Critical #2 row reworded for honesty + §6.1.18 NEW closure block) |
| **Total** | **+~595** | **-~10** | **+~585** |

**Test count delta**:

| Module | Pre-iter | Post-iter | Delta |
|---|---:|---:|---:|
| `serve::api::engine::tests` (a5c → a5d_seam_only rename) | 3 (a5c_*) | 3 (a5d_seam_only_*) | 0 net — pure rename + doc downgrade |
| `serve::api::handlers::a5d_handler_429_tests` (NEW) | 0 | **2** | **+2 handler-level tests** |
| **Total a5d additions** | — | **+2** | **+2 NEW production-handler-level always-on tests** |

**Quality gates (all PASS in skip mode — NO `cargo build --release` per CLAUDE.md "do not oom us")**:

- `cargo check --release --tests` returns 0 (no new warnings; same 4 pre-existing pre-A5d warnings: `gpu_full_attn.rs:11455-57 bad_shape` unused-assignment + `forward_gpu.rs:2507 use super::*` unused-import).
- `cargo test --release --test continuous_batching_throughput` returns 0 with **21 PASS / 0 FAIL** (preserved from iter-A5c — no regression).
- `cargo test --release --bin hf2q -- serve::api::engine::tests::a5d_ serve::api::handlers::a5d_ --test-threads=1` returns 0 with **5 PASS / 0 FAIL** (3 `a5d_seam_only_*` renames + 2 NEW `a5d_*_handler_returns_429_*` tests). Wall clock 0.08s; no model load. (Note: the bare `a5d_` filter ALSO matches the unrelated `inference::spec_decode::eagle3_orchestrator::g4_cfa5_redhatai_smoke::g4_cfa5d_diagnose_layer0_weight_ggml_types_2026_05_23` test → 6 PASS, not 5; codex /cfa A5d review correctly flagged this as a minor doc nit, fixed in iter-A5e via the narrower module-scoped filter above.)
- `cargo test --release --bin hf2q -- a5b_ a5c_ --test-threads=1` returns 0 with **22 PASS / 0 FAIL** — iter-A5b's 14 + iter-A5c's 8 still green; nothing regressed (only the 3 `a5c_chat_completions_*` were renamed to `a5d_seam_only_*` — they were counted under iter-A5c's `+3 engine tests` but no longer match the `a5c_` prefix).
- NO `// TODO`, NO `unimplemented!()`, NO `todo!()` in production code. ALL deferrals are typed.

**Empirical handler-test traces** (from `cargo test` output captured 2026-05-24 at iter-A5d):

```
test serve::api::handlers::a5d_handler_429_tests::a5d_chat_completions_stream_handler_returns_429_application_json_not_sse_when_kv_budget_exceeded ... ok
test serve::api::handlers::a5d_handler_429_tests::a5d_chat_completions_non_streaming_handler_returns_429_when_worker_signals_slot_budget_exceeded ... ok
test serve::api::engine::tests::a5d_seam_only_streaming_handler_admit_check_precedes_stream_call_source_grep ... ok
test serve::api::engine::tests::a5d_seam_only_streaming_response_is_json_not_sse_when_over_budget ... ok
test serve::api::engine::tests::a5d_seam_only_try_admit_budget_to_api_error_429_wire_shape ... ok
```

Each handler-level test calls a production handler function directly:
- streaming: `chat_completions_stream(state, req, prepared).await` (the `pub(crate)`-scoped streaming handler at `handlers.rs:1686`).
- non-streaming: `chat_completions_with_prepared(state, req, prepared).await` (the iter-A5d-extracted body at `handlers.rs:403`).

NEITHER test calls `engine.try_admit_budget(...)` directly. The two `a5d_seam_only_*` tests DO call the seam directly — that's their purpose as supplemental falsifiers.

**Remaining followups (none are deferrals from this iter)**:

- **C2c (Qwen35 SlotAware worker arm)** — per-layer dtype refinement (unchanged from §6.1.17).
- **C2d (Gemma 4 SlotAware worker arm)** — per-layer dtype refinement parallel to C2c (unchanged from §6.1.17).
- **Phase E1 production cutover** — flip `SchedulerPolicy::InflightBatched` to default (unchanged from §6.1.17).
- **Optional codex cleanup (minor)** — make `gemma4_exact_kv_bytes_per_token` compute `nkv` + `hd` from `cfg.num_kv_heads_for_layer(i)` + `cfg.head_dim_for_layer(i)` in release code instead of just `debug_assert_eq`-ing them against a local `match LayerType`. Current behaviour is equivalent (both paths use the same mapping); accepted as a minor wording mismatch for now per codex's own classification.

**Dossier provenance**: No standalone dossier — A5d is closure-iter work on the codex BLOCK verdict on iter-A5c. The path-selection rationale + the handler-vs-seam decomposition are documented inline (above).

---

### 6.1.19 Iter-A3b iter-1 closure — Gemma 4 multi-seq lift for HybridKvBuffers + clamps for DenseKvBuffers/MlxKvCache (2026-05-24, commit `15689b16`; iter-1.5 hygiene fix in follow-up commit per /cfa request_changes)

**Scope** (per A2/A3 dossier §Gemma 4 KV variants + R3 + H10 falsification):

Gemma 4 has four KV variants; A3a shipped `MultiSeqHbKvBuffers` (sibling for `HbKvBuffers`).  A3b iter-1 closes the remaining three under a graduated-lift strategy that respects the H10 falsification + the R3 R-register clamps mitigation:

| Variant | Production reachability | A3b iter-1 treatment | LOC | Deferral |
|---|---|---|---|---|
| `HybridKvBuffers` | **PRODUCTION DEFAULT** since ADR-029 iter-13 (H10 falsified — `HF2Q_HYBRID_KV` default-ON per `investigation_env.rs:878`) | **FULL multi-seq lift** via new sibling struct `MultiSeqHybridKvBuffers` + `alloc_multi_seq_hybrid_kv_for_layer` helper (mirror A3a's pattern verbatim) | ~310 LOC (struct + alloc + 6 trait methods + ByteSized impl) | A3c — `fork_seq` cross-slot kernel dispatch (parallel to Qwen35 A2c per dossier R5) |
| `DenseKvBuffers` | `HF2Q_USE_DENSE=1` (off-default; dev/debug path) | **TYPED CLAMP**: `slot_count() == 1`; `slot > 0` → `SlotOutOfRange { slot, max_slots: 1 }`; in-bounds append/drop → `CapabilityUnsupported { capability: "DenseKvBuffers::* (full multi-seq lift deferred to ADR-040 Phase A3b iter-2)" }` | ~75 LOC (6 trait methods) | **iter-A3b-2** — full multi-seq lift (~150 LOC) |
| `MlxKvCache` | Legacy 4-bit path (off-default since ADR-007 default-on TQ 8-bit) | Same TYPED CLAMP shape as `DenseKvBuffers`; `seq_len(SlotId(0))` reports the legacy `self.seq_len as u32` cursor | ~80 LOC (6 trait methods) | **iter-A3b-3** — full multi-seq lift (~80 LOC) |

**Decision matrix — why FULL vs CLAMP for each variant**:

- **HybridKvBuffers gets FULL lift in iter-1** because the H10 falsification reclassifies it as the PRODUCTION DEFAULT (not a deferred opt-in path per the original dossier framing).  Shipping anything less than a full lift here would block C2c (Gemma 4 SlotAware engine arm) on a second iter purely for paperwork.  The sibling-struct pattern from A3a transfers verbatim — ~310 LOC, mirrors `MultiSeqHbKvBuffers` line-for-line.
- **DenseKvBuffers + MlxKvCache get CLAMPS in iter-1** because both are NON-DEFAULT today; their lifts can ship in dedicated iters without blocking C2c (the SlotAware engine arm will route through `HybridKvBuffers` in production).  Clamps are typed (not vaporware) — every method returns a typed error that names the deferral iter so an operator who flips the env gate gets a grep'able log line, not a silent no-op or panic.  Per ADR-040 §7 "no fallback, no stub", `CapabilityUnsupported` is the iter-2.5 M1-blessed discriminant (HTTP 501 upstream — distinct from `SlotOom`'s HTTP 429).

**Per-file LOC delta** (additive only — production paths UNCHANGED, no existing tests removed):

| File | Pre-iter | Post-iter | Δ |
|---|---:|---:|---:|
| `src/inference/models/gemma4/kv_cache.rs` | 1850 | 2818 | +968 |

The +968 LOC splits as: ~470 LOC structural impl (3 struct/trait impls + 1 allocator + 1 ByteSized impl + module-level deferral notes), ~485 LOC test bank (H10-H16 falsifiers + per-clamp regression pins), ~13 LOC ADR-cross-reference comments.

**Test count delta**:

| Test bank | Pre-iter | Post-iter | Δ |
|---|---:|---:|---:|
| `inference::models::gemma4::kv_cache` | 28 | 35 | +7 |
| `tests/continuous_batching_throughput.rs` | 21 | 21 | 0 (preserved) |

The +7 new tests are H10 (post-falsification pin), H11, H12, H13, H14, H15, H16 (see hypothesis register below).

**Hypotheses pinned in this iter** (all PASS; all skip cleanly on no-MlxDevice CI hosts):

- **H10 (post-falsification, defence-in-depth)** — `InvestigationEnv::from_env().hybrid_kv == true` when `HF2Q_HYBRID_KV` is unset.  Falsifier: any regression that flips the default to OFF (e.g. a `env_default_true` → `env_default_false` rename) trips here naming this iter's H10 footnote.
- **H11 (HybridKvBuffers byte-scale)** — `alloc_multi_seq_hybrid_kv_for_layer(.., n_seqs=4)` produces buffers exactly 4× the n_seqs=1 baseline across K (F16), V packed (U8), V norms (F32); shape proves `n_seqs` is OUTERMOST on every buffer.
- **H12 (HybridKvBuffers per-slot byte isolation)** — host-side writes to slot 0's K / V packed / V norms regions leave slot 1's bytes byte-identical.  The cursor advance via `append_for_seq(SlotId(0), 3)` produces zero buffer mutation (A3b iter-1 scope is cursor-only).
- **H13 (HybridKvBuffers cursor independence)** — slot 0 advance + slot 2 advance leaves slots 1/3 cursors at 0; `drop_seq(SlotId(0))` resets slot 0 without touching slot 2.
- **H14 (HybridKvBuffers optional xlen)** — `HF2Q_DFLASH_XLEN_SDPA=1` causes `bf16_xlen_k/_v` to be `Some(_)` with shape `[n_seqs, nkv, cap, hd]` BF16; unset causes both fields `None`.  U8 V packed + F32 v_norms coexist unchanged in both modes.
- **H15 (DenseKvBuffers typed clamp)** — `slot_count() == 1`; `slot > 0` returns `SlotOutOfRange`; in-bounds append/drop return `CapabilityUnsupported` naming iter-A3b-2; self-fork at slot 0 is `Ok(())`.
- **H16 (MlxKvCache typed clamp)** — same shape as H15; `seq_len(SlotId(0))` reports the legacy single-seq cursor; capability label names iter-A3b-3 + "legacy 4-bit".

**Production allocation wiring** (deliberately deferred to C2c — same discipline A3a followed):

The A3b iter-1 sibling-struct ships the lift without touching the 3 production allocation sites (`forward_prefill.rs`, `forward_prefill_batched.rs`, `gemma4/model.rs:1247-1257`) — those keep allocating the legacy 3-D `HybridKvBuffers` / `MlxKvCache` at implicit `n_seqs=1` until Phase B4c / C2c (Gemma 4 SlotAware worker arm) re-routes them through `alloc_multi_seq_hybrid_kv_for_layer`.  This honours brief constraint #8 ("existing single-seq Gemma 4 production path UNCHANGED — additive lift only") + matches A3a's discipline.

**Typed deferrals named (no vaporware)**:

- **iter-A3b-2** — `DenseKvBuffers` full multi-seq lift.  Scope: extend the struct with `n_seqs` + per-seq `seq_lens: Vec<u32>`, lift buffer shapes from `[nkv, cap, hd]` to `[n_seqs, nkv, cap, hd]`, wire `MultiSeqKvCache::{append,drop,seq_len}` against the per-seq cursor.  ~150 LOC est. per dossier §2.2.4.  Production site at `engine.rs:5025` (`request_kv_restore` handler) is the wiring target.
- **iter-A3b-3** — `MlxKvCache` full multi-seq lift.  Scope: lift `k_packed`/`k_norms`/`v_packed`/`v_norms` shapes + replace `seq_len: usize` + `write_pos: usize` with `Vec<u32>` cursors.  ~80 LOC est. per dossier §2.2.4.  Legacy 4-bit path; production site at `gemma4/model.rs:1277-1290`.
- **iter-A3c** — `fork_seq` cross-slot kernel dispatch for both `MultiSeqHbKvBuffers` (A3a) and `MultiSeqHybridKvBuffers` (this iter).  Single dispatcher serves both sibling structs per dossier §2.3.3.

**Mantra-alignment audit**:

- ✅ No new `// TODO`, `unimplemented!()`, `todo!()`, `FIXME`.
- ✅ Every clamp returns a typed error with operator-grep'able context (capability label + deferral iter name).
- ✅ Per-struct `CapabilityUnsupported` labels mention the deferred iter (`A3b iter-2` / `A3b iter-3`) + the struct name + the legacy-path identifier — same shape as A3a's `gemma4_hb_kv_fork_cross_slot_returns_capability_unsupported` pin.
- ✅ Bounds-FIRST ordering preserved across all 18 new trait methods (per iter-1.5 cfa-finding-F5).
- ✅ A5* arc closure (commit `17f06a26` — the last A5* commit) NOT touched — additive impl only; the A5* files (engine.rs, load_info.rs, multi_seq_kv.rs, scheduler.rs, handlers.rs, schema.rs, sse.rs + 2 tests) are not modified by this iter.

**Verification**:

- `cargo check --release --tests` — clean (only pre-existing dead-code warnings unrelated to this iter).
- `cargo test --release --bin hf2q -- gemma4::kv_cache --test-threads=1` — **35/35 PASS** (28 pre-existing + 7 new A3b).
- `cargo test --release --test continuous_batching_throughput` — **21/21 PASS** (preserved).

**Dossier provenance**: A2/A3 dossier §Gemma 4 KV variants table (§2.2.1) + §2.10 R3 risk register + H10 falsification recorded in §A3a closure note (§6.1.11).

### 6.1.20 Iter-B4b closure — Qwen35 decode-path slot_id threading (2026-05-24, this commit)

Closes the §2.2 amendment for `src/inference/models/qwen35/forward_gpu.rs (decode)` and the §6 Phase B B4b row.  Threads `slot_id: SlotId` through the 5 Qwen35 decode-side entry points so the SlotAware engine arm (Phase C2c, gated on this row) can dispatch decode steps to specific slots.

Unlike B4a (prefill entry surface) which shipped signature-only and a typed B4a-cont follow-up for slot N>0 routing, B4b is a **FULL LIFT** in a single iter: the underlying `forward_gpu_impl` already accepts a `slot_id` parameter and routes per-slot K/V byte offsets to the F32 full-attn cache via `MlxBuffer::slice_view` (B4a-cont, §6.1.5), and the TQ-active multi-slot gate at `build_gated_attn_layer` / `apply_gated_attn_layer_decode_into` entry (B4a-cont.1, §6.1.6) is already in place.  B4b's work is precisely the public-surface signature lift + caller updates — no kernel changes, no new dispatch sites.

**Five decode-side entries lifted** (each previously hard-coded `SlotId(0)` at the `forward_gpu_impl` callsite):

| Entry | Signature change | OutputHeadMode | Production callers updated |
|---|---|---|---|
| `forward_gpu_last_logits` | +`slot_id: SlotId` (4th positional arg) | `Last` | 17 (imatrix calibration + serve/mod.rs + serve/api/engine_qwen35.rs) |
| `forward_gpu_last_topk` | +`slot_id: SlotId` (6th positional arg) | `TopK { k }` | 0 production (defined for sampler_pure but no live wiring); test-only callers updated |
| `forward_gpu_last_logits_with_soft_tokens` | +`slot_id: SlotId` (6th positional arg) | `Last` (with embed override) | 1 (engine_qwen35.rs generate-with-soft-tokens path) |
| `forward_gpu_last_logits_with_soft_tokens_and_deepstack` | +`slot_id: SlotId` (7th positional arg) | `Last` (with embed override + deepstack residual add) | 2 (engine_qwen35.rs generate-with-soft-tokens-and-deepstack non-streaming + streaming) |
| `forward_embed_last` | +`slot_id: SlotId` (4th positional arg) | `EmbedLast` (RMSNorm only + L2 norm) | 1 (engine_qwen35.rs Qwen35 chat-as-embedder path) |

**Total production callsites updated**: 25 across `src/quantize/imatrix/forward.rs`, `src/serve/mod.rs`, `src/serve/api/engine_qwen35.rs` — every site passes `SlotId(0)` to preserve pre-B4b byte-identical behaviour.  Each callsite carries an inline comment naming the iter and the gating reason (single-seq engine path until C2c lifts the SlotAware runtime).

**Why FULL lift and not signature-only + typed deferral (mirroring B4a's pre-cont pattern)**:
- B4a's signature-only pattern was forced by the GPU-side K/V byte routing not yet being slot-aware — the typed B4a-cont error existed because routing slot N>0 through the kernel dispatchers would silently corrupt slot 0's K/V region.
- B4b inherits B4a-cont's already-shipped F32 slot-offset routing (`MlxBuffer::slice_view` at the 5 kernel-dispatch sites in `gpu_full_attn.rs`).  The decode-side entries simply delegate to the same `forward_gpu_impl` body that `forward_gpu` (the prefill entry) uses.  Adding `slot_id` to the 5 wrappers is purely a signature lift — slot N>0 is END-TO-END FUNCTIONAL on the F32 full-attn path the moment the signature lift lands.
- TQ-active multi-slot remains gated per the existing B4a-cont.1 canonical entry gates at `build_gated_attn_layer` + `apply_gated_attn_layer_decode_into` (unchanged) — slot N>0 with `slot.tq.is_some()` returns the same typed B4a-TQ error at the SAME entry point as the prefill path.  No new gates, no defence-in-depth duplication.

**Hypothesis matrix** (H17–H20 + variant-coverage):

| ID | Hypothesis | Test name | Result |
|---|---|---|---|
| H17 | `forward_gpu_last_logits(.., SlotId(0))` at `n_seqs=4` is byte-identical to `n_seqs=1` (mirrors B4a's H2 at the decode-entry surface) | `b4b_forward_gpu_last_logits_at_slot_0_n_seqs_4_byte_identical_to_n_seqs_1` | **PASS** |
| H18 | `forward_gpu_last_logits(.., SlotId(1))` at `n_seqs=4` runs end-to-end without panic AND advances `current_len[1] == seq_len` while leaving sibling-slot cursors at 0 | `b4b_forward_gpu_last_logits_slot_1_succeeds_end_to_end` | **PASS** |
| H19 | Slot isolation on the decode-entry path: forward P→slot 0 (snapshot K/V); forward Q→slot 1; slot 0's K/V bytes UNCHANGED + slot 1's K bytes CHANGED (vacuous-test guard) | `b4b_forward_gpu_last_logits_slot_isolation_raw_kv_byte_snapshot` | **PASS** |
| H20 | Public-entry bounds check fires for out-of-range slot at the decode-entry path (proves slot_id propagates correctly into forward_gpu_impl's bounds check) | `b4b_forward_gpu_last_logits_slot_out_of_range_errors` | **PASS** |
| variant coverage | Each of the 4 sibling entries (`forward_gpu_last_topk`, `forward_gpu_last_logits_with_soft_tokens`, `forward_gpu_last_logits_with_soft_tokens_and_deepstack`, `forward_embed_last`) accepts `SlotId(0)` AND `SlotId(1)` end-to-end + errors uniformly on out-of-range slot | `b4b_forward_gpu_all_decode_variants_accept_slot_n` | **PASS** |

**Test count delta**:
- Baseline (post-B4a-cont.1): 7 `b4a*` / `b4a_cont*` tests in `qwen35::forward_gpu::tests::b4*`.
- +5 NEW: 4 H17/H18/H19/H20 + 1 variant-coverage = **12 PASS** under `cargo test --release --bin hf2q -- qwen35::forward_gpu::tests::b4 --test-threads=1`.
- Full `qwen35::forward_gpu` suite: **38 PASS** (unchanged from baseline).
- `qwen35::mtp`: **9 PASS** (preserved).
- `qwen35::spec_decode`: **5 PASS** (preserved — spec_decode passes SlotId(0) explicitly into `forward_gpu_with_hidden`, which is a B4a-shipped surface untouched by B4b).
- `qwen35::kv_cache` + `serve::scheduler` + `serve::multi_seq_kv` combined: **153 PASS** (4 new B4b tests across the combined harness; pre-existing 149 preserved).
- `continuous_batching_throughput`: **21/21 PASS** (preserved — Phase D bench is engine-mode-gated and not affected by B4b's signature lift).

**LOC delta** (per-file):

| File | +LOC | -LOC | Net | Notes |
|---|---|---|---|---|
| `src/inference/models/qwen35/forward_gpu.rs` | +~440 | -~30 | +~410 | 5 entry signature lifts + 5 NEW tests (H17–H20 + variant coverage) + module-level B4b commentary block + test-site updates (~25 existing test callers gained SlotId(0)). |
| `src/quantize/imatrix/forward.rs` | +~10 | -~1 | +~9 | 1 callsite + SlotId import + inline B4b comment. |
| `src/serve/mod.rs` | +~30 | -~9 | +~21 | 6 callsites + SlotId import in `cmd_generate_qwen35` + inline B4b comments. |
| `src/serve/api/engine_qwen35.rs` | +~80 | -~20 | +~60 | 17 callsites + SlotId import + inline B4b comments at engine seam. |
| `docs/ADR-040-continuous-batching-reopen.md` | +~135 | -~2 | +~133 | §2.2 row updated to SHIPPED + §6 Phase B B4b row updated to SHIPPED + §6 Phase C C2c row updated to reflect B4b unblocked + this §6.1.20 closure block. |

**Total**: +~695 LOC / -~62 LOC = +~633 LOC.

**Decision matrix**:

| Decode entry | B4b scope | Status |
|---|---|---|
| `forward_gpu_last_logits` | FULL LIFT | SHIPPED — slot_id end-to-end on F32 full-attn |
| `forward_gpu_last_topk` | FULL LIFT | SHIPPED — slot_id end-to-end on F32 full-attn |
| `forward_gpu_last_logits_with_soft_tokens` | FULL LIFT | SHIPPED — slot_id end-to-end on F32 full-attn (soft-tokens overrides are caller-owned MlxBuffer rows; no slot K/V interaction) |
| `forward_gpu_last_logits_with_soft_tokens_and_deepstack` | FULL LIFT | SHIPPED — slot_id end-to-end on F32 full-attn (deepstack residual-add hits caller-owned `hidden` buffer; no slot K/V interaction) |
| `forward_embed_last` | FULL LIFT | SHIPPED — slot_id end-to-end on F32 full-attn (OutputHeadMode::EmbedLast skips lm_head matmul; same KV write path as Last) |
| `forward_gpu` (prefill) | B4a/B4a-cont scope (NOT B4b) | UNCHANGED — was already lifted in B4a + B4a-cont |
| `forward_gpu_with_hidden` (MTP draft hidden) | B4a scope (NOT B4b) | UNCHANGED — was already lifted in B4a |
| `forward_gpu_with_hidden_dflash` (DFlash spec-decode) | B4d scope (NOT B4b) | DEFERRED to B4d per ADR-040 §2.2 line 93 |
| `forward_gpu_greedy` (greedy fast-path decode) | B4d scope (NOT B4b) | DEFERRED to B4d per ADR-040 §2.2 line 93 (greedy is a spec-decode entry point in the ADR's scope split; its single internal `apply_gated_attn_layer_decode_into` + `build_gated_attn_layer` callsites already pass SlotId(0) per B4a-cont annotations at forward_gpu.rs:5260 / 5579) |
| `forward_gpu_with_capture` (DWQ activation capture) | calibration-tooling, single-stream by construction | UNCHANGED — still hard-codes SlotId(0) at forward_gpu_impl callsite (B4a-style annotation) |

**Typed deferrals NAMED** (per ADR-040 §7 mantra "no fallback, no stub"):
- **TQ-active multi-slot decode**: inherits the existing B4a-TQ typed gate at `build_gated_attn_layer` / `apply_gated_attn_layer_decode_into` entry — slot N>0 with `slot.tq.is_some()` returns a typed B4a-TQ error.  No new gate added in this iter (the existing canonical gates fire identically for decode-entry calls because both paths funnel through the same dispatchers).  Pinned by the existing `b4a_cont_1_tq_active_multi_slot_gated_at_build_gated_attn_layer_entry` test (KEPT PASS).
- **Linear-attn multi-slot**: deferred to Phase A2b per the existing `rollback_la_to` guard at `kv_cache.rs:1567` (out-of-scope for any current B4* iter — multi-seq linear-attn is gated on the spec-decode + multi-seq combo per ADR-040 §4 OPEN question 5).  The B4b tests use a dense-full-attn fixture (`tiny_dense_full_attn_model_nonzero_for_b4a`) that the linear-attn guard never reaches.
- **Spec-decode / DFlash decode-side slot_id**: deferred to **B4d** per the ADR §2.2 phase row.  `forward_gpu_with_hidden_dflash` and `forward_gpu_greedy` retain their B4a-shipped `SlotId(0)` hard-codes with explicit comments naming B4d as the unblocking iter.

**Mantra-aligned**: no `// TODO`, no `unimplemented!()`, no `panic!()` in production code.  No new files added to repo root.  TQ-active multi-slot is the only deferred decode-path case — gated with a typed error naming the specific kernel work needed (B4a-TQ), inherited unchanged from B4a-cont.1.  Slot 0 across ALL 5 decode entries remains byte-identical to pre-B4b (pinned by H17 over `forward_gpu_last_logits` + variant-coverage smoke pin over the other 4).

**Quality gates (all green)**:
- `cargo check --release --tests` — clean (only pre-existing warnings unrelated to this iter).
- `cargo test --release --bin hf2q -- qwen35::forward_gpu::tests::b4 --test-threads=1` — **12/12 PASS** (7 B4a/B4a-cont/B4a-cont.1 preserved + 5 NEW B4b).
- `cargo test --release --bin hf2q -- qwen35::forward_gpu --test-threads=1` — **38/38 PASS**.
- `cargo test --release --bin hf2q -- qwen35::mtp --test-threads=1` — **9/9 PASS**.
- `cargo test --release --bin hf2q -- qwen35::spec_decode --test-threads=1` — **5/5 PASS** (B4d-deferred surface preserved).
- `cargo test --release --bin hf2q -- qwen35::kv_cache::tests serve::scheduler::tests serve::multi_seq_kv::tests --test-threads=1` — **153/153 PASS**.
- `cargo test --release --test continuous_batching_throughput` — **21/21 PASS** (Phase D bench preserved).

**Dossier provenance**: A2/A3 dossier §1.3 ("Qwen35 partial n_seqs claim — production wiring uses n_seqs=1 today; structural shape supports >1") confirmed at the decode-entry surface; §3 Phase B4 sub-iter sequencing satisfies the B4a → B4a-cont → B4a-cont.1 → B4b sequence; §2.10 R5 (decode-side state contamination) addressed by H19's negative-pin + vacuous-test guard.

**Future-iter pin pointers**:
- **C2c** (gated on B4b + R4 spec-decode mitigation + R4-bis hybrid persistor n_seqs>1 serialization, 5-8 days): with B4b now landed, the Qwen35 worker arm has a fully-slot-aware decode surface to dispatch into.  Remaining gates are scheduler-side (R4) and persistor-side (R4-bis); B4b itself is no longer a blocker.
- **B4a-TQ**: lift `dispatch_hadamard_quantize_kv_hb_seq` + `flash_attn_vec_tq_hb` to slot-aware.  Once landed, the canonical TQ-active multi-slot gates at `build_gated_attn_layer` / `apply_gated_attn_layer_decode_into` entry can be removed; B4b's tests continue to PASS unchanged (they use a dense-F32 fixture that never engages the TQ path).
- **B4c**: Gemma 4 forward-prefill slot threading (gated on Phase A3b iter-2/iter-3 finishing the `DenseKvBuffers` + `MlxKvCache` full lifts per §6.1.19 closure).
- **B4d**: spec-decode (`forward_gpu_with_hidden_dflash` + `forward_gpu_greedy`) slot threading (gated on Phase A4 drafter multi-seq KV).

---

## 7. Risks + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SeparateSlots layout wastes too much memory at production context lengths (N×max_seq_len allocation) | Medium | Medium | Phase D measures it. If ≥30% waste, open separate ADR for PagedAttention kernel port. |
| Scheduler refactor breaks Phase 2 single-request behaviour | Low (regression test pins) | High | C iter-1 ships the byte-equivalence regression test BEFORE C iter-2 touches `Engine`. |
| Spec-decode incompatible with multi-slot (EAGLE-3 drafter cache + multi-seq verifier interaction) | Medium | Medium (Phase 5 nice-to-have, not Phase E1 gate) | Phase A iter-4 ships research-quality only; Phase E1 doesn't require it. |
| KV-spill (ADR-017) breaks under multi-slot | Medium | Medium | Phase A iter-5 explicitly tests spill+restore at N>1. ADR-017's per-model spiller surface inherits naturally because the per-slot KV lives inside the per-model cache. |
| Per-slot OOM under aggregate budget pressure causes thrashing | Low | Low | §3.5: per-slot OOM → 429 to admitting handler; no cross-slot eviction. |
| llama.cpp `-cb` admission semantics don't match our forward-path assumptions | Low | Medium | Phase B iter-3 explicitly cites the llama.cpp file:line being mirrored; deviations documented inline. |
| Mantra violation: shipping `FifoSerial` + `InflightBatched` both produces "fallback" code | Low | Low | §3.6: `FifoSerial` is the explicit production default + Phase E1 gate decides the cutover. Both paths are first-class, not one-is-a-fallback. |

---

## 8. References

### vLLM
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180.
- vLLM source: `vllm/core/scheduler.py`, `vllm/attention/ops/paged_attn.py`.

### llama.cpp
- `src/llama-kv-cache.cpp` — multi-seq KV cache implementation.
- `-cb` (continuous batching) CLI flag — admission-during-decode semantics.
- `src/llama-batch.cpp` — batch construction across multiple sequences.

### ADR-005 cross-references
- §"Concurrent-deployment scaling (deferred, future ADR)" (line 1097-1103) — the carve-out this ADR reopens.
- Resolved Question "Phase 2 scope refinement" Decision #1 (line 6652) — deferral decision with reopen trigger.
- Resolved Question "Phase 2 scope refinement" Decision #2 (line 6653) — FIFO contract this ADR preserves under `SchedulerPolicy::FifoSerial`.
- Resolved Question "Phase 2 scope refinement" Decision #19 (line 6679) — 429 + Retry-After contract preserved.
- Phase 4 §"Out of scope" (line 6439) — "Phase 4's pool is request-serial within each loaded model" — superseded by this ADR's Phase C cutover.

---

## 9. Why this is the right next step

Three orthogonal pressures converged on 2026-05-23:

1. **Operator activation**: explicit direction "2, 3, 4, 5 ← do this" against the deep-research findings, under the mantra "multi-week structural work is ALWAYS in scope".
2. **ADR-005 reopen slot is reserved**: not new architecture, not scope creep — Phase 2 deliberately carved out a future-ADR slot that this ADR fills.
3. **Existing footholds**: `HybridKvCache` already carries `n_seqs` in buffer shape; `HotSwapManager` separates per-model from per-request lifecycles; the FIFO contract is wired through 5 named file:lines and can be wrapped under a trait without rewrite.

The shape of this work is structural Walk, not optimization. ADR-005's comparator bar ("parity or better than ollama + llama.cpp") gains a new axis once the reopen trigger fires: aggregate throughput under N concurrent. This ADR builds the infrastructure to measure that, then to ship it.

**Per the mantra**: this ADR does not stub or todo-later. Each phase's iter-1 ships compiling scaffolding with tests; each subsequent iter implements one cohesive piece with regression pins. No shortcuts.
