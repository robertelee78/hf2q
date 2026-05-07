# ADR-017 Per-Family Ship-Gate Status

**Last updated:** 2026-05-06 (post Phase B-tq.4 iter-1+2+3 — substrate + cmd_serve registration + integration test harness; live E2E automation deferred to B-tq.5 driver extraction)
**Companion to:** [ADR-017](./ADR-017-persistent-block-prefix-cache.md)
**Phase D §476 closure doc.**

This document satisfies the ADR-017 Phase D §476 checklist item:

> **Per-family ship-gate read:** every in-scope family's parity gate
> documented GREEN at `docs/ADR-017-per-family-status.md` with
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
| Qwen 3.5 / 3.6 (hybrid, MoE+DeltaNet) | `src/serve/api/engine_qwen35.rs::Qwen35LoadedModel::lcp_registry` (Phase E.a B.2-B.5 substrate; no sibling `KvCacheSpill` family hook needed) | PASS (B.2-iso falsifier 0/131072) | n/a (LCP path doesn't use Phase D spiller) | PASS (B.3 stride-aligned + B.5 byte-identity end-to-end) | n/a (LCP-resume measured separately at R-P6 0.79× of 4×cold; bench at `scripts/bench_lcp_resume_speedup.sh`) | n/a (sourdough/dense-only kill-gates don't apply to LCP path) | **GREEN** (Phase E.a B.5 closed 2026-05-05; HF2Q_KV_LCP_RESUME=1 default-on flip operator-controlled post 24h soak) |
| TQ-packed (codec_version=1 + codec_version=2 + bundle codec) | `src/serve/kv_persist/families/tq_packed.rs` (B-tq.1 v1 envelope + B-tq.2 `TqPackedSpill` hook + B-tq.3 v2 engine wiring + B-tq.4 iter-1+2+3 activation factory) + `src/serve/forward_mlx.rs::MlxModelWeights::tq_v2_*` + `src/serve/api/tq_packed_descriptor.rs` + `src/serve/api/engine.rs::tq_packed_v2_*` worker bridge + `src/serve/mod.rs::cmd_serve` single-mode factory registration + `tests/kv_persist_tq_packed_roundtrip.rs` integration harness | PASS (v1 + v2 round-trip byte-exact = R-C1; bundle round-trip byte-exact) | n/a | PASS (D2 byte-exact rebuild → cosine = 1.0 = R-C2 trivially; v2 capture→restore byte-identity on synthetic `[nkv, capacity, hd_packed]` U8 + `[nkv, capacity]` F32 buffers) | n/a (no inference perf bench at substrate level) | n/a | **GREEN-substrate** (engine wiring + factory registration + integration harness landed 2026-05-06 across B-tq.4 iter-1+2+3, commits `62bb8b5`+`b346425`+`69b3bc2`; live automated R-C1 round-trip test deferred to B-tq.5 driver extraction — operator runbook in `tests/kv_persist_tq_packed_roundtrip.rs` documents manual validation sequence). NOT blocked on ADR-007 Path C; codec-freeze contract F-7 LANDED 2026-05-05 per `docs/adr007-pathC/PATHC_CLOSURE.md` |

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
- **Bench output:** `docs/ADR-017-persistent-block-prefix-cache.md:2140-2151`
  (Phase D iter-4 2026-05-01)
- **Measurement:**
  - Baseline decoded 3632 bytes (1000 tokens, ttft=311.8 ms).
  - Restored decoded 3632 bytes (1000 tokens, ttft=0.5 ms).
  - **Byte-identical** (3632 == 3632).
  - TTFT 311.8 ms → 0.5 ms = **624× speedup** on cache-hit.

### R-P4 — `cache_hit_TTFT(32K) / no_cache_TTFT(32K) ≤ 0.20` (PASS)

- **Test:** `tests/kv_persist_gemma4_roundtrip.rs::kv_persist_phase_d_r_p4_e2e`
- **Bench output:** `docs/ADR-017-persistent-block-prefix-cache.md:2178-2192`
  (Phase D iter-4 2026-05-01)
- **Measurement:**
  - `no_cache_ttft` = 649,569.3 ms
  - `cache_hit_ttft` = 13.1 ms
  - **ratio = 0.000** (ship-gate ≤ 0.20; **49,585× speedup at L=32K**).

### K2 R-P1 sustained-decode overhead (FALSIFIED — PASS)

- **Test:** `tests/kv_persist_gemma4_roundtrip.rs::kv_persist_phase_d_r_p1_decode_overhead_e2e`
- **Bench output:** `docs/ADR-017-persistent-block-prefix-cache.md:2826-2837`
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

## Qwen 3.5 / 3.6 (hybrid: MoE + DeltaNet) — pending

**Status:** family hook NOT YET LANDED.

ADR-017 §B-hybrid sequencing depends on serve-side qwen35 load
landing under ADR-013 — the trait surface ships family-agnostic, but
the impl rides the family's serve-side enablement, and the
DeltaNet-boundary-snapshot integration adds requirements beyond the
dense Gemma 4 pattern.

**Tracking:** ADR-013 (hybrid-architecture serve-side load).
B-hybrid family hook + per-family parity gate verification will be
recorded here when it lands.

Until then `src/serve/api/engine.rs:2114` returns `None` for the
`LoadedModel::Qwen35(_)` arm of the descriptor closure — i.e. the
KV-spill descriptor is not yet emitted for this family, and the
spiller short-circuits to the `NoopKvSpiller` path on Qwen35 hot
sessions. This is the explicit "no stub" semantic per ADR-017's
mantra: hybrid Qwen3.5 ships when its parity gate is GREEN by
measurement, not before.

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
- B-hybrid lands — flip the Qwen 3.5/3.6 row from PENDING to PASS
  and link the bench evidence.
- a new family ships — add a section mirroring the Gemma 4 layout
  (engine path, descriptor closure, R-Cn / R-Pn rows, kill-gate
  status, production fixes, pending operator gates).

Cross-link from any new ADR-017 status updates: when ADR-017's
status line changes (e.g. as additional operator-controlled benches
return), update both the ADR-017 §Status block AND this document so
the per-family read stays in sync with the headline.
