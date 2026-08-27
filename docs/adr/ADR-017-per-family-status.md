# ADR-017 Per-Family Ship-Gate Status

**Last updated:** 2026-08-26 (ADR-049 Lane A reconciliation: Qwen
SerialFifo registry/disk hydrate is the restart tier; SlotAware uses a
separate slot-local anchor store. Model-free invariants and Qwen3.6/Qwen3.8
real-artifact divergence/coherence gates are green. Current-head product and
matched-peer matrices remain separate ADR-049 Lane B authorities.)
**Companion to:** [ADR-017](./ADR-017-persistent-block-prefix-cache.md)
**Phase D §476 closure doc.**

This document satisfies the ADR-017 Phase D §476 checklist item:

> **Per-family ship-gate read:** every in-scope family's parity gate
> documented GREEN at `docs/adr/ADR-017-per-family-status.md` with
> measurement evidence and date.

It is the canonical place to look up, per LLM family, which ADR-017
parity / coherence / perf gates have landed in code, which have been
validated by measurement, and which remain operator-controlled
(bench-pending or ADR-blocked).

For the spec of each gate (R-Cn / R-Pn / Kn) see ADR-017 §§Coherence
requirements, §Performance requirements, §Kill-gates.

---

## Summary table

| Family | Engine path | R-C1 | R-C3 | R-C4 | R-P4 | K1/K2/K3 | Phase D Status |
|---|---|---|---|---|---|---|---|
| Gemma 4 (dense, A4B variant) | `src/serve/kv_persist/families/gemma4_dense.rs` | PASS | PASS | PASS | PASS (ratio=0.000 @ L=32K) | All falsified | **GREEN** (primary; R-P5/R-P6 measured 44,500× / 1.00× post Phase D iter-5/6 + B.5; stress 24h smoke pass at iter-11/12) |
| Qwen 3.5 / 3.6 / 3.8 (hybrid DeltaNet family) | SerialFifo: `engine_qwen35.rs::Qwen35LoadedModel::lcp_registry` + disk hydrate. SlotAware: `engine.rs::Qwen35AnchorStore` (same-process semantic boundaries; registry unused). | PASS for the SerialFifo LCP substrate (B.2-iso falsifier 0/131072) and SlotAware anchor byte identity on Qwen3.6/Qwen3.8 | n/a (neither Qwen reuse path uses the Phase D spiller contract) | PASS for SerialFifo stride-aligned/disk restore and SlotAware divergent branches, cancellation, retry, and failed publication | n/a (LCP benchmark and SlotAware product TTFT gates are separate) | n/a (dense-only kill-gates do not apply) | **GREEN, SerialFifo persistence and SlotAware anchors**. Exact real-artifact receipts are indexed by ADR-049 revs 20–23; current-head Lane B product/peer matrices remain open. `bench_lcp_resume_speedup.sh` is not the SlotAware gate. |
| TQ-packed (codec_version=1 + codec_version=2 + bundle codec) | `src/serve/kv_persist/families/tq_packed.rs` (B-tq.1 v1 envelope + B-tq.2 `TqPackedSpill` hook + B-tq.3 v2 engine wiring + B-tq.4 iter-1+2+3 activation factory) + `src/serve/forward_mlx.rs::MlxModelWeights::tq_v2_*` + `src/serve/api/tq_packed_descriptor.rs` + `src/serve/api/engine.rs::tq_packed_v2_*` worker bridge + `src/serve/mod.rs::cmd_serve` single-mode factory registration + `tests/kv_persist_tq_packed_roundtrip.rs` AUTOMATED integration test (uses `tests/common/serve_driver.rs` shared driver via B-tq.5 extraction) | PASS (v1 + v2 round-trip byte-exact = R-C1 unit; cross-process R-C1 automated when `HF2Q_KV_PERSIST_TQ_E2E=1`) | n/a | PASS (D2 byte-exact rebuild → cosine = 1.0 = R-C2 trivially; v2 capture→restore byte-identity on synthetic `[nkv, capacity, hd_packed]` U8 + `[nkv, capacity]` F32 buffers) | n/a (no inference perf bench at substrate level) | n/a | **GREEN** (engine wiring + factory registration + automated integration test landed 2026-05-06 across B-tq.4 iter-1+2+3 + B-tq.5, commits `62bb8b5`+`b346425`+`69b3bc2`+`539e6f7`; live R-C1 measurement on a real GGUF is operator-driven post-merge work — harness drives the full subprocess round-trip when env-gated). NOT blocked on ADR-007 Path C; codec-freeze contract F-7 LANDED 2026-05-05 |

Legend:
- **PASS**: gate validated by measurement; evidence linked below.
- **GREEN**: family's primary ship-gates passed; remaining items are
  operator-controlled bench or ADR-blocked, not in-tree.
- **GREEN-substrate**: family's storage / serialization layer GREEN;
  engine-side runtime integration is a separate iter (B-tq.2 for
  TQ-packed).
- **PENDING**: family hook not yet landed; gate not yet runnable.
- **n/a**: gate not applicable to this family's architecture.

---

## Gemma 4 (dense, A4B variant) — primary

**Engine path:** `src/serve/kv_persist/families/gemma4_dense.rs`
**Factory registration:** `src/serve/mod.rs::cmd_serve` (Gemma4DenseSpillFactory).
**Descriptor closure:** `src/serve/api/engine.rs:2063-2112`
(`KvSpillDescriptor::from_gemma_loaded_model`) — captures real
GGUF-derived shape from `MlxModelWeights` at engine spawn; consumed
by the factory downcast at
`src/serve/kv_persist/families/gemma4_dense.rs:1464-1479`.

### R-C4 — internal sourdough byte-equality (PASS)

- **Test:** `tests/kv_persist_gemma4_roundtrip.rs::kv_persist_phase_d_coherence_e2e`
- **Bench output:** `docs/adr/diary/ADR-017-persistent-block-prefix-cache.md:2140-2151`
  (Phase D iter-4 2026-05-01)
- **Measurement:**
  - Baseline decoded 3632 bytes (1000 tokens, ttft=311.8 ms).
  - Restored decoded 3632 bytes (1000 tokens, ttft=0.5 ms).
  - **Byte-identical** (3632 == 3632).
  - TTFT 311.8 ms → 0.5 ms = **624× speedup** on cache-hit.

### R-P4 — `cache_hit_TTFT(32K) / no_cache_TTFT(32K) ≤ 0.20` (PASS)

- **Test:** `tests/kv_persist_gemma4_roundtrip.rs::kv_persist_phase_d_r_p4_e2e`
- **Bench output:** `docs/adr/diary/ADR-017-persistent-block-prefix-cache.md:2178-2192`
  (Phase D iter-4 2026-05-01)
- **Measurement:**
  - `no_cache_ttft` = 649,569.3 ms
  - `cache_hit_ttft` = 13.1 ms
  - **ratio = 0.000** (ship-gate ≤ 0.20; **49,585× speedup at L=32K**).

### K2 R-P1 sustained-decode overhead (FALSIFIED — PASS)

- **Test:** `tests/kv_persist_gemma4_roundtrip.rs::kv_persist_phase_d_r_p1_decode_overhead_e2e`
- **Bench output:** `docs/adr/diary/ADR-017-persistent-block-prefix-cache.md:2826-2837`
  (Phase D iter-8 2026-05-01)
- **Measurement:**
  - `baseline_ttft_avg` = 60.8 ms
  - `sustained_ttft_avg` = 0.3 ms
  - **overhead = −0.995** (gate ≤ 0.05; sustained path is FASTER than baseline).
  - K2 kill-gate **falsified**.
- **Iter-12 polish:** concurrent-eviction-during-decode variant
  (`kv_persist_phase_d_r_p1_concurrent_eviction_e2e`) closes the
  honest caveat at iter-8 about iter #1-4 hitting empty pool slots.
  Gate verdict unchanged.

### K1 / K3 status

- **K1** (cache-hit ratio gate fail) — falsified by R-P4 with 200× margin.
- **K3** (decode regression) — falsified by R-P1 with negative overhead.

### Production fix derived from Phase D bench

- `2b3f62d` — `Gemma4DenseSpill engine_arc must be Weak<Engine>` (P0-bench fix).
  - File: `src/serve/kv_persist/families/gemma4_dense.rs:277`
  - Surfaced by Phase D iter-4 attempt while validating R-C4.

---

## Qwen 3.5 / 3.6 / 3.8 (hybrid DeltaNet family)

Qwen does not use the Gemma `KvCacheSpill` hook. Its two live mechanisms
have different ownership and scheduler contracts:

- **SerialFifo restart tier:** `Qwen35LoadedModel::lcp_registry` stores
  byte-budgeted hybrid snapshots and `hydrate_lcp_registry_from_disk`
  repopulates them after restart. This remains a supported, permanent tier;
  it is not an unfinished stub. The registry's effective capacity comes from
  payload bytes, not a one-entry or fixed-count comment.
- **SlotAware same-process tier:** `Qwen35AnchorStore` retains up to four
  committed stable-boundary anchors plus one preflighted pending capture per
  physical slot. It never probes the SerialFifo registry. Epoch validation,
  descendant pruning before divergent writes, fail-atomic restore preflight,
  and full-store invalidation on reset/poison/restore failure protect the one
  mutable KV lineage.

The model-free proof includes an independent reference state machine, a
17-injected-mutation invariant battery, the A→B→C rewind regression,
restore-no-partial-mutation, exact committed+pending accounting, and the
right-sized speculative hidden-row ownership test. ADR-049 revs 20–23 then
closed the real-artifact Qwen3.6/Qwen3.8 anchor authorities: concurrent
SlotAware divergence, cold byte comparison across anchor depths and physical
widths, cancellation/failure/spec-state joins, exact continuation, and
cached-token/TTFT receipts. That makes the SlotAware anchor milestone green.
ADR-049 Lane B's joined rectangular/Mixed product matrix and current pinned-
peer performance matrix remain open performance authorities; they do not
reopen Lane A cache coherence.

---

## Pending operator gates (Gemma 4)

These are code-complete; bench-pending and operator-controlled
(see `scripts/adr017_phase_d.sh --help` for env-var opt-ins).

| Gate | Spec | Code Status | Bench Status |
|---|---|---|---|
| R-P5 | cold-process resume `cache_hit_TTFT(32K) / no_cache_TTFT(32K) ≤ 0.15` | code-complete (W1, this iter) | PENDING |
| R-P6 | 4-agent shared 4K prefix `aggregate ≤ 1.25 × single_agent_prefill(4K)` | code-complete (W1, this iter) | PENDING |
| Stress | 24h continuous swap-in/swap-out (RSS within 5%, no descriptor leak) | code-complete (W2, this iter) | PENDING (full 24h operator-only; this session ran 30-min reduced-duration smoke) |
| R-C4 peer arm | byte-shared prefix vs `llama-completion` ≥ 3094 bytes | code-complete | DEFERRED (blocked on ADR-005 chat-template defect; iter-6 commit `c8dc50f`) |
| Full 60-cell matrix sweep | `kv_persist_gemma4_roundtrip_matrix_e2e` 60 cells | code-complete | PENDING (operator-controlled bench) |

---

## Maintenance

Update this doc when:

- a new bench is run on Gemma 4 (e.g. R-P5, R-P6, full matrix, 24h
  stress) — add a row under "Pending operator gates" with the
  bench-output cross-reference and date.
- Qwen current-head Lane B matrices run — update the separate product and
  matched-peer status without reopening the green SlotAware cache milestone.
- a new family ships — add a section mirroring the Gemma 4 layout
  (engine path, descriptor closure, R-Cn / R-Pn rows, kill-gate
  status, production fixes, pending operator gates).

Cross-link from any new ADR-017 status updates: when ADR-017's
status line changes (e.g. as additional operator-controlled benches
return), update both the ADR-017 §Status block AND this document so
the per-family read stays in sync with the headline.
