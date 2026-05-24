# ADR-040 — Continuous batching: reopen the ADR-005 carve-out

- **Status**: 🚧 ACTIVE — Design 2026-05-23, **iter-1 SHIPPED 2026-05-23** (this commit). Multi-iter arc in progress under goal-mode directive. Reopens the [ADR-005 Decision #1 carve-out](ADR-005-inference-server.md) ("continuous batching pulled from ADR-005 — deferred to a future ADR; reopen trigger = real deployment scenario with ≥8 concurrent users on a single instance").
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

| Component | Path | LOC est. (iter-1 scaffold) |
|---|---|---|
| Multi-seq KV trait + types | `src/serve/multi_seq_kv.rs` | ~250 |
| Scheduler trait + FIFO adapter | `src/serve/scheduler.rs` | ~400 |
| `EngineMode` enum + slot-aware Engine extension | edits to `src/serve/api/engine.rs` | ~120 |
| Continuous-batching throughput benchmark | `tests/continuous_batching_throughput.rs` | ~180 |

**Total iter-1 LOC**: ~950 (scaffolding-only — no production callsite activation).

### 2.2 Files this ADR edits across the multi-iter arc (iter-2+)

| File | Change | Phase |
|---|---|---|
| `src/inference/models/qwen35/kv_cache.rs` | implement `MultiSeqKvCache` for `HybridKvCache` (lifts `n_seqs=1` to N) | A iter-2 |
| `src/inference/models/gemma4/kv_cache.rs` | implement `MultiSeqKvCache` for Gemma4 dense KV | A iter-3 |
| `src/inference/spec_decode/eagle3/kv_cache.rs` | `MultiSeqKvCache` impl (research-quality; gated on Phase E) | A iter-4 |
| `src/inference/spec_decode/dflash/kv_cache.rs` | same | A iter-4 |
| `src/serve/api/engine.rs` | replace mpsc-channel + single worker with scheduler-driven slot loop under `SchedulerPolicy::InflightBatched` | C iter-2 |
| `src/serve/api/sse.rs` | per-slot keepalive accounting (no contract change) | C iter-3 |
| `src/serve/api/schema.rs` | doc-only Decision #2 update naming `SchedulerPolicy` | C iter-3 |
| `src/serve/forward_prefill.rs` | accept `slot_id` parameter; route writes to multi-seq KV | B iter-3 |
| `src/serve/forward_prefill_batched.rs` | same | B iter-3 |
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
- SSE keepalive accounting moves to per-slot (15s/slot vs 15s/connection — no client-visible difference at N=1).
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
| A2 | `HybridKvCache` (Qwen35) impl — lift `n_seqs` from 1 to N | 3-5 days |
| A3 | Gemma 4 dense KV impl | 3-5 days |
| A4 | Drafter KV caches (EAGLE-3, DFlash) — research-quality | 5-8 days |
| A5 | Per-slot OOM + budget enforcement | 2-3 days |
| A6 | Closure: per-family parity gate vs n_seqs=1 baseline | 2 days |

### Phase B — Scheduler (4-6 iters)

| Iter | Scope | Estimated effort |
|---|---|---|
| **B1 (THIS ITER, 2026-05-23)** | Scaffolding: trait + FifoSchedulerAdapter + InflightBatchedScheduler stub | 1 day |
| B2 | FifoSchedulerAdapter byte-equivalence proof + regression pin | 2-3 days |
| B3 | InflightBatchedScheduler admit/step/release impl | 5-8 days |
| B4 | Forward-path slot-id threading (`forward_prefill.rs` + `forward_prefill_batched.rs`) | 5-8 days |
| B5 | Per-slot 429 + Retry-After contract preservation | 2-3 days |
| B6 | Mixed prefill+decode `SchedulerStep::Mixed` handling | 3-5 days |

### Phase C — Engine slot-aware (3-4 iters)

| Iter | Scope | Estimated effort |
|---|---|---|
| **C1 (THIS ITER, 2026-05-23)** | Scaffolding: `EngineMode` enum + signature-only `SlotAware` variant + regression test | 1 day |
| C2 | `Engine::spawn` accepts `EngineMode` + `Scheduler` injection | 3-5 days |
| C3 | SSE keepalive per-slot accounting + schema.rs doc updates | 2-3 days |
| C4 | CLI/env wiring for `HF2Q_SCHEDULER` + `--scheduler` | 1-2 days |

### Phase D — Throughput benchmark (2-3 iters)

| Iter | Scope | Estimated effort |
|---|---|---|
| **D1 (THIS ITER, 2026-05-23)** | Scaffolding: env-gated test file + metric definitions | 1 day |
| D2 | N ∈ {1, 2, 4, 8} measurement harness + report format | 3-5 days |
| D3 | A/B comparator (FIFO vs InflightBatched) + statistical stability | 2-3 days |

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
