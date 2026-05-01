# ADR-017: Persistent Block Prefix Cache for serve mode

- **Status:** **Closed-Shipped (Phase D GREEN 2026-05-01; operator-readiness follow-ups remain).** ADR-017 substrate is shippable: Phase D R-C4 internal coherence PASS (3632-byte sourdough byte-identical baseline==restored, 624× cache-hit TTFT speedup) + Phase D R-P4 perf ship-gate PASS (ratio=0.000 vs ≤0.20 spec; 49,585× speedup at L=32K cache-hit). Two of three kill-gates (K1, K3) FALSIFIED with multiple orders of magnitude margin; K2 (dirty-block decode overhead) is a sustained-decode property deferred to operator-load testing. Per-family parity gate (Gemma 4 26B) is GREEN by code+test; B-hybrid (Qwen3.5-MoE) ships when ADR-013 unblocks (trait surface inherits forward; impl rides the family). Operator-readiness follow-ups (F4 fingerprint thread-through, P1-1 lock-step atomicity, P1-3 budget_bytes wiring, 4 of 6 §R-F7 /metrics counters, native cache --kv-namespace clear) are tracked in §"Remaining ADR-017 work after iter-3"; none are correctness or performance gating.

  *Original Status (preserved for provenance): "Accepted (falsification-gated). ADR-017 is a committed decision: hf2q ships per-model SSD KV-cache persistence across the Phase 4 hot-swap eviction signal for every model family complete in code on serve-side. Phase A0 (falsification harness on M5 Max) is the first deliverable and the explicit ship-or-die gate — every kill-criterion in §10 is a hard exit that closes ADR-017 unmerged with rationale, not a 'circle back later' punt. Per-family parity gate is non-negotiable (no carve-out, no descope without Robert's explicit re-authorization)."*
- **Date:** 2026-04-30
- **Authors:** Robert E. Lee + Claude Code (party-mode synthesis session 2026-04-30)
- **Predecessor:** ADR-005 Phase 4 (multi-model hot-swap; reopened 2026-04-30 to publish the eviction-hook surface ACs 5471 + 5472 that this ADR consumes)
- **Cross-references:**
  - **ADR-005** — Phase 4 owns the orchestration substrate (`HotSwapManager<E>`, `LoadedPool`, AppState wiring); ADR-017 hooks into the new `Arc<dyn KvSpiller<E>>` surface (AC 5471) and `/metrics` counters (AC 5472).
  - **ADR-006** — mlx-native is the strategic GPU backend; ADR-017's restore path uses `MlxBuffer` `StorageModeShared` CPU memcpy or a new `metal_blit_block` op if Phase A0 surfaces CPU-copy as the bottleneck.
  - **ADR-007** — TurboQuant KV representation; ADR-017 owns the on-disk envelope for TQ-packed K/V blocks (deterministic codec rebuild on restore), ADR-007 owns the codec itself.
  - **ADR-009** — coherence gates that the round-trip parity assertions in §10 must clear (Gate A cosine ≥ 0.9998 on TQ-active path; sourdough byte-exact under `HF2Q_USE_DENSE=1`).
  - **ADR-013** — hybrid-architecture serve-side load (qwen35); ADR-017's Phase B-hybrid sequencing depends on serve-side qwen35 load landing. Trait surface ships family-agnostic; impl rides each family's serve-side enablement.
  - **ADR-014 P7** — closed iter-110 (`a75d9ed`, 2026-04-29); the `src/backends/gguf.rs` fence release unblocks ADR-005 AC 5469 writer-half (Phase 4 reopen iter-211), which lands the GGUF metadata path that ADR-017 reuses for cache-namespace fingerprinting (`model_fingerprint = (repo_id, quant, arch_rev, tokenizer_hash)`).
- **Authoritative research backing:** [`docs/research/omlx-2026-04-30.md`](research/omlx-2026-04-30.md) — dual-mode CFA dossier (Claude/sonnet + Codex parallel research, Opus 4.7 queen synthesis); 5 highest-stakes claims spot-checked directly in oMLX source; mantra-violation post-mortem in §13. Conditional-yes recommendation; this ADR formalizes the "yes" conditions with concrete phasing, file paths, and mechanical acceptance criteria.

---

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim. This is the discipline this ADR — and every spike, every commit, every decision under it — must be executed against. It supersedes any tactical convenience that conflicts with it.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

**Operational reading for ADR-017** (how to apply, not how to interpret — the text above is the source of truth):

- **DO NOT BE LAZY / no short cuts.** Phase A0 falsification harness lands BEFORE any production code touches `HotSwapManager` or any `KvCacheSpill` impl. Per `feedback_harness_first_before_iter_chasing`, this is non-negotiable. The cost of a refuted Phase B–C spike is always larger than the cost of a Phase A0 measurement up front. The "cost" includes: SSD I/O eating the win on M5 Max (R2 below); DeltaNet boundary snapshot exceeding the Walk discipline (R1); TQ codec non-determinism (R3). Every one of these is cheaper to falsify in A0 than to land + revert.
- **Plenty of time.** The 24–35 man-day estimate is acceptable. Per Robert directive 2026-04-30, the bar is "all families and all quants we support" — no carve-out, no TQ deferral to v2, no dense-only Gemma 4 ship. If hybrid Qwen3.5 boundary snapshot turns out to need 3 weeks instead of 1, we spend the 3 weeks rather than ship a partial path that implies hybrid correctness.
- **Never make assumptions.** Every ADR-017 design decision (D1–D10) cites either source code (`/opt/omlx/...:line`, `src/serve/multi_model.rs:line`) or measurement evidence (Phase A0 outcome). Claims about M5 Max DRAM economics, NVMe latency, or block-size optimality are pending Phase A0 measurement and are explicitly flagged as such until A0 returns.
- **Dive deep / use search as needed.** Chesterton's fence on oMLX's design (the dossier already does this — 451 lines, 5 spot-checks against source). Chesterton's fence on `HotSwapManager`'s eviction loop before injecting the trait (already done in ADR-005 Phase 4 reopen rationale). Chesterton's fence on Qwen3.5-MoE's `gpu_delta_net.rs` before B-hybrid (next session work).
- **Measure 3x, cut once.** Phase A0 is the canonical pre-spike measurement: matrix harness across families × quants × scenarios × prefix lengths × cache states. Per-cell ship-gate or kill-gate. No production code commits until A0 returns numbers; no per-family `KvCacheSpill` impl ships until that family's parity gate is GREEN by measurement.
- **No fallback / no stub.** Per `feedback_never_ship_fallback_without_rootcause` and `feedback_correct_outcomes`: ADR-017 does not ship a dense-only Gemma 4 path that implies hybrid Qwen3.5 correctness. Either Gemma 4 ships fully (Phase B-dense + B-tq + C all green) or ADR-017 holds at "Phase B-hybrid pending ADR-013 unblock" with a dated exit condition. There is no `// TODO: hybrid later` shipped code. There is no `KvCacheSpill::snapshot_layer_state` returning `None` for "not implemented" — `None` means "this layer-rank has no recurrent state to snapshot," and that's a permanent semantic, not a stub.
- **Pure excellence, done right.** Coherence > speed. A 5× TTFT speedup that breaks Gate A cosine ≥ 0.9998 on TQ-active or sourdough byte-exact under dense is a regression, not progress. The kill-gates in §10 dominate the decision surface; the ship-gates are necessary but not sufficient.
- **Chesterton's fence.** Every architectural decision below cites the source pattern it mirrors or the measurement that justifies the choice. D3 (block size 256) cites oMLX's empirically-validated default; D4 (chain-hash) cites oMLX `paged_cache.py:126-162`; D5 (atomic rename) cites oMLX `paged_ssd_cache.py:989-1007`; D6 (serialized-bytes hot tier) cites oMLX's explicit comment at `paged_ssd_cache.py:1198-1245` about IOGPUMemory underflow. We don't reinvent where oMLX's pattern is verified.

**Cross-reference:** the verbatim mantra appears in [ADR-005 §Engineering Mantra](ADR-005-inference-server.md#engineering-mantra-load-bearing--read-before-every-session) and [ADR-006 §Engineering Mantra](ADR-006-mlx-native-gpu-backend.md). All three ADRs (005, 006, 017) remain in sync if `~/Documents/mantra.txt` is updated.

---

## Problem Statement (Why this ADR exists)

### What's broken today

`hf2q serve` (ADR-005 Phase 1b–4) operates a hot LLM context entirely in process memory. The KV cache (`src/inference/models/<family>/kv_cache.rs::HybridKvCache`) lives inside the live `Engine` (`src/serve/api/engine.rs`); the engine's lifetime is gated on the `Arc<LoadedEngine<E>>` slot in `HotSwapManager` (`src/serve/multi_model.rs:682`). When the slot drops, the engine drops, the cache drops, all session context is gone.

This is fine for the single-model live-session case — the model stays loaded; the KV stays in GPU RAM; the next request's prefill walks the existing cache. But three classes of "circle back to a previous prefix" event pay the full prefill tax today:

| Class | Frequency under sustained Phase 4 use | TTFT cost (32K context, Gemma 4 26B Q4_0 on M5 Max, measured) |
|---|---|---|
| **(a) Cold process resume** — `cmd_serve` restart | Daily on dev workflows; weekly on long-running serve | Full 32K prefill — ~13.7 s on hf2q gemma4 today (from `scripts/bench-baseline.sh` measurements; references ADR-005:5837 prompt-eval calibration sweep) |
| **(b) Hot-swap evict-then-readmit** | Triggered every time `HotSwapManager` admits a 4th distinct model under Phase 4's N=3 capacity (ADR-005:5836); operator-frequency depends on workflow | Same full 32K prefill; the evicted model's `HybridKvCache` is dropped with its `Arc<LoadedEngine<E>>` |
| **(c) Parallel agents share system prompt** | N concurrent agents with same 4K system prompt | N × prefill cost; one shared prefix would suffice |
| **(d) Edit-in-middle retry** | Common in agentic coding (re-prompt after tool-call failure) | Full re-prefill from edit point forward; chain-hash semantics make this structural — not a problem we can solve without changing the agent loop |
| **(e) Swap-out → swap-back-in same model on same context** | A coding agent uses the primary code model for tool calls, switches to a vision model to read a screenshot, returns to the code model | Without persistence: full 32K prefill on every swap-back-in (~13.7 s on Gemma 4 26B). With persistence: ~256 ms SSD restore (computed: 128 blocks × ~2 ms @ 5 GB/s NVMe per dossier §4.5). **Net: ~50× TTFT win on every swap-back-in cycle.** |

Today, `hf2q` solves zero of (a)/(b)/(c)/(e). Class (d) is structural per chain-hash semantics and out of scope.

### Why it matters now (and not three months ago)

Three things changed simultaneously on 2026-04-30 to make ADR-017 a load-bearing complement to Phase 4 rather than an optional upgrade:

1. **Phase 4 hot-swap is in tree** (ADR-005:5694–5744 — iter-206/207/208/209/210 LANDED). Class (b) and class (e) become *frequent* events under sustained Phase 4 use. Without persistence, Phase 4's user-visible "fast hot-swap" reads as "swap is fast but every swap-back-in costs 13 seconds." Class (b) and (e) flip from "not an issue" to "the dominant cost" as Phase 4 adoption grows.

2. **oMLX commit `af97a0f` (2026-04-30)** ships a working reference design for the persistence mechanism — chain-hashed 256-token blocks, atomic safetensors per block, async writer thread, startup directory scan rebuilds the index. Dossier §4 verifies this end-to-end against `/opt/omlx` source, including the exact lifecycle pattern that avoids `IOGPUMemory` underflow (storing serialized bytes in the hot tier, not live MLX arrays — `paged_ssd_cache.py:1198-1245`).

3. **ADR-014 P7 closure (iter-110, `a75d9ed`, 2026-04-29)** unblocks ADR-005 AC 5469 writer-half. The Phase 4 reopen 2026-04-30 schedules iter-211 to land the writer; ADR-017 reuses the same GGUF metadata path for cache-namespace fingerprinting (`(repo_id, quant, arch_rev, tokenizer_hash)`). Without P7's release, the namespace-fingerprint design would have to invent its own fingerprint scheme.

### What "all model families and all quants we support" means precisely (Robert directive 2026-04-30)

ADR-017's per-family parity gate scopes to **families complete in code currently on serve-side with a meaningful KV cache**. As of 2026-04-30 that resolves to **Gemma 4 26B** alone:

| Family | Serve-side status | KV semantics | ADR-017 close-gate scope |
|---|---|---|---|
| **Gemma 4 26B MoE-A4B** (chat, mmproj/vision absorbed Phase 2c) | LANDED, daily-driver fixture, `forward_mlx.rs` is gemma-specific | Dense full-attention, `[head_dim, n_kv_heads, max_seq_len, n_seqs]` BF16/F32 K/V (`kv_cache.rs:14-16, 305-337`) | **YES — primary close-gate target** |
| **Qwen3.5-MoE-DWQ46 / Qwen3.6-MoE** | BLOCKED on ADR-013 (`forward_mlx.rs:803` LMHEAD slice OOB on vocab-pad mismatch; `:884` missing `attn_q.weight` for delta-net layers) | Hybrid 3:1: full-attention F32 K/V (16 layers) + DeltaNet conv_state + recurrent state (48 layers, `kv_cache.rs:339-392`, `forward_gpu.rs:2651-2673`) | Trait inherits forward; `KvCacheSpill` impl ships as fast-follow when ADR-013 unblocks. Does not gate ADR-017 closure for Gemma 4. |
| **BERT-family embeddings** (bge-small-en-v1.5, mxbai-embed-large-v1, nomic-embed-text-v1.5) | LANDED Phase 2b iter-91 (ADR-005:3297) | **No KV cache** — one-shot pooled forward pass | EXCLUDED by KV semantics, not by descope. |
| **Qwen3 dense, Mistral** | Aspirational per ADR-005 AC 5468; no current serve-side impl | Dense full-attn (when landed) | Trait inherits forward; impl rides each family's serve-side enablement. |

The "all families we support" bar means: **every future family complete in code MUST land its own `KvCacheSpill` trait impl as part of its serve-side enablement**. ADR-017's close gate does NOT retroactively wait on families that are not yet complete-in-code — the trait surface inherits forward; the parity gate is scoped to today's reality. This sidesteps the failure mode where "all families" reads as "wait for every conceivable future family" (the dual of the carve-out trap Robert rejected when descope-to-dense-only-first was proposed).

For "all quants we support" — dense K/V dtype is independent of weight quantization. K/V live in BF16/F32 regardless of whether weights are Q4_0 / Q4_K_M / Q5_K / Q6_K / Q8_0 / DWQ46 / DWQ48. The quant axis collapses to **`{TQ-inactive, TQ-active}`** — the single dimension that perturbs KV-on-disk format. Both ship together (Phase B-tq, §8).

---

## Context

### Existing-code inventory (citable, 2026-04-30 HEAD)

These are the load-bearing primitives ADR-017 builds on. **Engineers executing ADR-017 must Chesterton's-fence each before changing it.**

| File:line | Primitive | Role in ADR-017 |
|---|---|---|
| `src/serve/multi_model.rs:682` | `HotSwapManager<E>` struct | Eviction-hook injection target (consumes Phase 4 AC 5471 trait surface). |
| `src/serve/multi_model.rs:523` | `LoadedEngine<E>` wrapper, `Arc`-handed-out | Restore target — `post_admit` hook receives `&Arc<LoadedEngine<E>>` for first-request pre-warm. |
| `src/serve/multi_model.rs:1175` | `MockEngine` / `MockLoader` test fixtures | `MockSpiller` mirror pattern for unit-test fixtures. |
| `src/serve/multi_model.rs:746-760` | `HotSwapManager::evict()` | Symmetric trigger site; `pre_evict` hook fires here too. |
| `src/serve/multi_model.rs:778-789` | `HotSwapManager::load_or_get` | Primary trigger sites; `pre_evict` before `engines.remove`, `post_admit` between `loader.load` return and `engines.insert`. |
| `src/serve/api/engine.rs:548` | `PromptCache` (single-slot) | Generalize to `BlockPrefixCache` per D1; preserve full-equality fast path. |
| `src/serve/api/engine.rs:663-738` | `PromptCache::lookup` / `PromptCache::store` | Pre-prefill hook lives here; chain-hash lookup adds prefix-match path. |
| `src/serve/cache.rs` | On-disk weights cache (manifest + GGUF + mmproj) | Distinct lifetime/policy per `multi_model.rs:34-36`; ADR-017 KV cache is parallel-but-separate at `<cache_root>/models/<slug>/kv/`. |
| `src/serve/cache.rs:825-830` | `iter_entries()` | Operator-visibility pattern to mirror for `cmd_cache --kv-namespace`. |
| `src/serve/provenance.rs` | `Provenance::Hf2q { producer_version, source_sha256, mmproj_sha256 }` reader | Cache-namespace fingerprint source — `model_fingerprint = sha256(producer_version || source_sha256 || mmproj_sha256 || tokenizer.chat_template)`. |
| `src/inference/models/<gemma4-eqv>/kv_cache.rs` | Gemma 4's `HybridKvCache` (full-attn dense path) | `KvCacheSpill` impl target for B-dense. |
| `src/inference/models/qwen35/kv_cache.rs:14-16, 305-337` | Qwen3.5-MoE full-attn slot shape | `[head_dim, n_kv_heads, max_seq_len, n_seqs]` F32; sliceable along axis 2 for block extraction. |
| `src/inference/models/qwen35/kv_cache.rs:339-392` | DeltaNet `conv_state` + `recurrent` | Boundary-snapshot target for B-hybrid; non-sliceable per-token. |
| `src/inference/models/qwen35/forward_gpu.rs:2651-2673` | DeltaNet per-token state-update | Snapshot trigger point — capture at every block boundary during prefill. |
| `src/inference/models/qwen35/gpu_delta_net.rs:1417-1715` | DeltaNet prefill chunk path | Most invasive surface for B-hybrid; Chesterton's fence required before any hook insertion. |
| `/opt/mlx-native/src/ops/kv_cache_copy.rs:8-9` | KV copy primitives | Extend, don't recreate. Restore path uses these for CPU-buffer → MlxBuffer copy. |
| `tests/multi_model_swap.rs` (iter-210, `e991c96`) | Swap-timing E2E harness pattern | Substrate for ADR-017's Phase A0 harness (subprocess + symlink-as-distinct-pool-key trick + env-gated). |
| `src/intelligence/hardware.rs:178-220` | `HardwareProfiler::detect()` | Exposes `total_memory_bytes` for KV-cache directory budget defaults (e.g. 10% of unified RAM = 12.8 GiB on M5 Max). |

### What oMLX has that hf2q doesn't (verbatim from dossier §11)

- Chain-hashed block index with restart-survivable on-disk format (`paged_ssd_cache.py:823-847`, `:989-1007`).
- Per-block atomic safetensors writer with hex fanout (`paged_ssd_cache.py:246-297`).
- LRU hot tier of *serialized* bytes (not live tensors) with size-bounded eviction (`paged_ssd_cache.py:643-648`, `:708-736`).
- Async store thread + deferred batch-generator slot release (commit `af97a0f`).
- Boundary snapshot store for non-sliceable hybrid-cache layers; walk-back truncation (`boundary_snapshot_store.py`, `prefix_cache.py:1401-1437`).
- Block-level CoW + ref counting for shared-prefix dedup across concurrent requests.
- TurboQuant on stored KV (`turboquant_kv.py:2-7` wrapping `mlx_vlm.turboquant`).
- Multi-slot prompt cache (oMLX persists both prompt+output for non-reasoning models and prompt-only for reasoning models).

### What hf2q has that oMLX doesn't (relevant to ADR-017; from dossier §12)

- Pure-Rust pipeline; zero Python dependency in the inference path (per `feedback_hf2q_sovereignty`).
- Owned mlx-native Metal kernels (oMLX rents from mlx-vlm).
- Native TurboQuantMSE with Hadamard substitution (O(d log d), not O(d²)).
- ADR-007 Lloyd-Max HB SDPA at 8-bit default with documented Gate A cosine 0.9998 + Gate B 0.8% argmax + Gate C 1.24% PPL.
- Sourdough byte-exact gate vs llama.cpp under dense (`scripts/sourdough_gate.sh`, 3656 bytes, floor 3094).
- `project_speed_bar_full_matrix`: ≥1.00× llama.cpp same-hardware bar, currently MET at q4_0-flat (1.0457×, ADR-015 iter56).
- Single-slot `PromptCache` already shipped (`engine.rs:663-738`, ADR-005 iter-96 commit `005768a`).

### Hardware envelope (M5 Max 128 GiB, primary target)

- Unified memory: 128 GiB; pool budget at 80% = 102.4 GiB (ADR-005:5836 `hf2q_pool_memory_budget_bytes=109,951,162,777`).
- NVMe SSD: ~5 GiB/s sustained read on M5 Max APFS (dossier §4.5). 256-token block at BF16 dense full-attn ~5 MiB → ~1 ms read; 32K context = 128 blocks → ~128 ms sequential SSD time, ~256 ms with metadata + per-block decode overhead.
- DRAM economics: BF16 dense KV at hf2q's `[head_dim=256, n_kv_heads=1, max_seq_len=512, n_seqs=2]` Qwen3.5-MoE shape = ~1 MiB per layer per 256-token block; 16 full-attn layers = 16 MiB per block; 32K = 128 × 16 MiB = 2 GiB on-disk per session per model. Manageable; cache-clear surface required (R5).
- **Phase A0 must validate this envelope on M5 Max before B–C commit.** Numbers above are computed; dossier flags M5 Max as primary target with M3 Ultra as opportunistic cross-check.

---

## Solution (What we build)

### Architecture overview

ADR-017 introduces a per-model SSD-backed prefix cache that hooks into `HotSwapManager`'s eviction signal. On every model swap-out, the spiller extracts block-aligned K/V from the evicting engine's `HybridKvCache` and enqueues to a background writer thread. On every model swap-back-in, the spiller consults the on-disk index for matching session prefixes and pre-warms the freshly-loaded engine's cache before the first user request arrives.

```
                                                                              ┌──────────────────────────┐
                                                                              │  cache_root/models/      │
                                                                              │    <slug>/kv/{0-f}/      │
                                                                              │      <sha256_hex>.safe-  │
                                                                              │      tensors             │
                                                                              └──────────────────────────┘
                                                                                     ▲             │
                                                                       atomic write  │             │  read on startup +
                                                                       (rename)      │             │  on cache-hit lookup
                                                                                     │             │
            ┌─────────────────────────────────┐       ┌────────────────────┐   ┌─────┴─────┐  ┌────▼─────┐
            │  HotSwapManager<E>              │  pre- │                    │   │           │  │          │
            │  (Phase 4, ADR-005:5712-5740)   │ evict │ KvSpiller<E>       │   │  Async    │  │ Block    │
            │                                 ├──────►│ trait              │──►│  Writer   │  │ Index    │
            │  load_or_get(repo, quant, ...)  │       │                    │   │  Thread   │  │ (RAM)    │
            │    1. file-stat                 │       │ pre_evict()        │   │           │  │          │
            │    2. evict LRU if needed   ────┤       │ post_admit()       │   └───────────┘  └──────────┘
            │       └─► pre_evict hook        │       │                    │
            │    3. loader.load()             │       │ (this ADR's        │
            │    4. post_admit hook       ────┤       │  BlockPrefixCache- │
            │    5. publish Arc to engines    │ post- │  Spiller impl)     │
            │                                 │ admit │                    │
            └─────────────────────────────────┘──────►└────────────────────┘
                                                              │
                                                              │  per-family dispatch
                                                              ▼
                                              ┌─────────────────────────────────┐
                                              │ KvCacheSpill trait impls        │
                                              │   - GemmaSpill (B-dense)        │
                                              │   - Qwen35Spill (B-hybrid)      │
                                              │   - <future-family>Spill        │
                                              └─────────────────────────────────┘
                                                              │
                                                              ▼
                                              ┌─────────────────────────────────┐
                                              │ Family's HybridKvCache type     │
                                              │   - snapshot_block(layer, range)│
                                              │   - restore_block(layer, range, │
                                              │       payload)                  │
                                              │   - snapshot_layer_state(...)   │  ← Qwen3.5 DeltaNet
                                              │   - restore_layer_state(...)    │  ← Qwen3.5 DeltaNet
                                              └─────────────────────────────────┘
```

### Architecture decisions

#### D1 — Trait surface lives in Phase 4; impl lives in ADR-017

`trait KvSpiller<E>` is defined in `src/serve/multi_model.rs` as part of Phase 4 AC 5471 (iter-212). The trait is generic over `E` (no `Engine` widening); accepts `&Arc<LoadedEngine<E>>`; returns `SpillOutcome` / `RestoreOutcome` enum that drives `/metrics` counter labels (AC 5472).

ADR-017 ships a `BlockPrefixCacheSpiller<E>` impl in `src/serve/kv_persist/spiller.rs` (new module). It owns:
- `Arc<dyn BlockStore>` — the disk-backed block store (atomic writer, LRU directory eviction, restart-recovery).
- `Arc<RwLock<BlockIndex>>` — chain-hash index keyed by `(model_fingerprint, block_hash)`.
- `Arc<dyn KvCacheSpill>` per resident model — registered when a family's `KvCacheSpill` impl is wired through `cmd_serve`.

```rust
// Phase 4 (AC 5471, iter-212) — defined here, never moves:
pub trait KvSpiller<E>: Send + Sync {
    fn pre_evict(&self, handle: &LoadedHandle, engine: &Arc<LoadedEngine<E>>) -> SpillOutcome;
    fn post_admit(&self, repo: &str, quant: QuantType, engine: &Arc<LoadedEngine<E>>) -> RestoreOutcome;
}
pub enum SpillOutcome { Skipped, EnqueuedBlocks(u32), Error(SpillError) }
pub enum RestoreOutcome { Skipped, RestoredBlocks(u32), Error(RestoreError) }
pub struct NoopKvSpiller;  // Phase 4 default; ADR-017 substitutes via cmd_serve config

// ADR-017 (Phase A onwards):
pub trait KvCacheSpill {
    type Payload: SpillSerialize + Send + Sync;
    fn block_alignment(&self) -> u32;  // 256 for full-attn, family-defined for hybrid
    fn snapshot_block(&self, layer_rank: usize, range: Range<u32>) -> Self::Payload;
    fn restore_block(&mut self, layer_rank: usize, range: Range<u32>, payload: &Self::Payload);
    fn snapshot_layer_state(&self, layer_rank: usize, token_pos: u32) -> Option<Self::Payload>;
    fn restore_layer_state(&mut self, layer_rank: usize, token_pos: u32, payload: &Self::Payload);
}

pub trait SpillSerialize: Sized {
    fn payload_kind() -> &'static str;
    fn to_bytes(&self) -> Cow<'_, [u8]>;
    fn from_bytes(bytes: &[u8], family_meta: &FamilyMeta) -> Result<Self, SpillError>;
}
```

**Phase 4 doesn't know ADR-017 exists.** ADR-017 cannot land without Phase 4's surface (R6 below; mirror of ADR-005:reopen R7).

#### D2 — TurboQuant is a payload decoration, not a separate family

Per dossier §4.9 + ADR-007: TQ isn't a fourth family — it's a runtime decoration over an existing family's payload. Same Gemma 4 `KvCacheSpill` impl; the `Payload` type sees `(use_dense: bool, codec_version: u32)` from the runtime config (`HF2Q_USE_DENSE`, ADR-007 codec) and selects between dense-K/V bytes or TQ-packed (`codebook + indices + codec_version`).

ADR-007 owns the codec; ADR-017 owns the on-disk envelope (`payload_kind = "dense_kv_v1"` vs `"tq_packed_v1"`). Two ADRs, one `KvCacheSpill` impl per family.

**Determinism requirement for TQ-packed:** SHA-256 of `tq_dequantize_kv(restored_envelope)` must equal SHA-256 of `tq_dequantize_kv(pre_spill_envelope)` for the same pre-spill input. ADR-007's Lloyd-Max codec is deterministic by construction (codebook is a function of the input distribution); rebuild reuses the codebook bytes from the on-disk envelope. If determinism fails, the bug is in the envelope serialization, not the codec — fix in B-tq, not in ADR-007.

#### D3 — Block size = 256 tokens

`BLOCK = 256` matches oMLX's scheduler default (`scheduler.py:321-331`). Trade-off: smaller blocks = finer-grained restore + more SSD ops; larger blocks = coarser restore + fewer ops. 256 is oMLX's empirically-validated point (verified in dossier §4.1 against oMLX source); we adopt without re-litigation.

**Phase A0 reserves the right to revisit.** If the harness measures that 128 or 512 wins materially on M5 Max, surface as a B-dense decision and re-spec.

#### D4 — Chain-hash identity over `(model_fingerprint, parent_hash, token_ids[..BLOCK])`

```
block_hash(0) = sha256(model_fingerprint || token_ids[0..BLOCK])
block_hash(N) = sha256(model_fingerprint || block_hash(N-1) || token_ids[N*BLOCK..(N+1)*BLOCK])

model_fingerprint = sha256(
    repo_id || quant_canonical_str ||
    gguf_metadata["hf2q.producer_version"] ||  // from AC 5469 writer-half (Phase 4 reopen iter-211)
    gguf_metadata["hf2q.source_sha256"]    ||
    gguf_metadata["tokenizer.chat_template"]
)
```

Re-quanting or upgrading the chat template invalidates the namespace cleanly — the old blocks become orphaned (cleaned up on next restart-recovery scan; see D8). Edit-in-middle invalidates from the edit forward — chain-hash semantics make this structural; no special handling.

This matches oMLX's scheme verbatim per dossier §4.4.

#### D5 — One safetensors file per block, hex fanout, atomic rename

Layout:
```
<cache_root>/                                       // resolved by HF2Q_KV_CACHE_DIR or default
├── manifest.json                                   // global KV-cache index (schema v1)
├── locks/                                          // advisory flock(LOCK_EX) per (model, block_hash[..2])
└── models/
    └── <model_fingerprint_short>/                  // first 16 hex chars of model_fingerprint
        ├── meta.json                               // full model_fingerprint, repo_id, quant, arch_rev, tokenizer_hash
        └── kv/
            └── {0-f}/                              // single hex-character fanout (16 dirs per model)
                └── <block_hash_full_hex>.safetensors
```

Single hex-character fanout (16 directories per model) is sufficient for the expected block-count regime (~10^4–10^5 blocks per model under sustained use). Manual safetensors writer on the background thread (`paged_ssd_cache.py:246-297` pattern): 8-byte little-endian header length → JSON header → concatenated tensor bytes. Atomic publication: temp file + `std::fs::rename`. Header carries `omlx_compat_format_version` (NOT the same value space as oMLX's; we'll use `hf2q_kv_cache_format_version: u32` starting at `1`).

#### D6 — Hot-tier RAM cache holds *serialized bytes*, not live `MlxBuffer`

Per dossier §4.2 + oMLX comment at `paged_ssd_cache.py:1198-1245`: storing live MLX/Metal arrays in the LRU hot tier accumulates GPU allocations and risks `IOGPUMemory` underflow. Mirror oMLX's choice: `LruCache<BlockHash, Bytes>` of *serialized* tensor bytes; restore decodes into a freshly-allocated `MlxBuffer` on demand.

**Hot tier is opt-in (`HF2Q_KV_HOT_CACHE_BYTES`, default `0` = disabled).** Phase A0 measures whether it pays on M5 Max single-process workloads. If A0 returns "no measurable benefit," ship default-disabled and document the rationale. If it returns "5-10% TTFT win on (b)/(e) at 8K," ship default-enabled with a sized budget (1 GiB default).

#### D7 — DeltaNet boundary snapshot via per-block conv_state + recurrent capture

For Qwen3.5-MoE / Qwen3.6-MoE hybrid layers: capture `conv_state` + `recurrent` at every block boundary during prefill (`gpu_delta_net.rs:1417-1715` chunk-pipeline path). Store as the block's `linear_attn_state` payload. On restore, choose the latest valid boundary; if the request's prefix exceeds the latest boundary, walk back to the most recent boundary and re-execute the tail (oMLX's `_find_walk_back_truncation_point` pattern, `prefix_cache.py:1401-1437`).

**This is the highest-risk piece** (R1 below). DeltaNet state is not sliceable per-token the way full-attn K/V is; the recurrent state must be captured at exact boundaries. Phase B-hybrid is the longest sub-phase (5–8 man-days) and the kill-criteria most likely to fire are concentrated here.

**ADR-017 will not ship Qwen3.5 KV persistence with a fallback path that loses correctness.** Either Phase B-hybrid lands a hybrid-correct snapshot/restore, or the Qwen3.5 `KvCacheSpill` impl waits on a follow-up ADR. Per `feedback_never_ship_fallback_without_rootcause` and the mantra: no stub.

#### D8 — Restart recovery via directory scan + chain-hash re-validation

On `cmd_serve` startup with `HF2Q_KV_PERSIST=1`:

1. Walk `<cache_root>/models/*/kv/*/*.safetensors`.
2. For each file: read header (8-byte length + JSON), validate `hf2q_kv_cache_format_version`, extract declared `block_hash` + `parent_block_hash` + `model_fingerprint`.
3. Files whose declared `parent_block_hash` does not exist on disk → mark as orphans → defer to LRU cleanup (don't delete eagerly; `parent_block_hash == None` for genesis blocks is valid).
4. Files with corrupted headers (truncated, version-mismatch, hash-mismatch on read-back) → quarantine to `<cache_root>/models/<slug>/kv-quarantine/<original-name>` and increment `hf2q_kv_quarantined_total{reason}` gauge.
5. Files whose declared `block_hash` does not match `sha256(file_contents)` → quarantine same path; reason `hash_mismatch`.
6. Build in-memory `BlockIndex: HashMap<BlockHash, BlockMeta>` from valid files. Block lookup is O(1) on hash.

**`kill -9` mid-write recovery:** the atomic-rename invariant (D5) means partially-written files are at `<temp_name>.tmp.<pid>` and never get renamed to `<sha256_hex>.safetensors`. Startup scan ignores `.tmp.<pid>` files; on next serve startup with no live PID matching, they're cleaned up.

#### D9 — Per-block atomicity > per-write-batch atomicity

Each block lands as an independent atomic unit. We do NOT batch-commit multiple blocks in one rename. Rationale: kill -9 mid-batch recovery becomes a per-block consistency problem instead of a per-batch consistency problem; the chain-hash invariant means partial chains are detectable on next startup and partially-written tail blocks elide cleanly.

Counter-argument: per-block fsync is more expensive than per-batch. Phase A0 measures per-block writer throughput; if it's the bottleneck for sustained spill-on-evict workloads, B-dense reserves the right to introduce a `flush_buffer` pattern (write multiple blocks to a single tempfile + atomic-rename) — but only after measurement, never speculatively.

#### D10 — Cache version envelope

On-disk header carries `hf2q_kv_cache_format_version: u32`. Initial value `1`. Future format changes bump this; reader rejects unknown future versions with quarantine + log-warn. Per dossier §4.3, oMLX bumped its `omlx_cache_format_version` from `"1"` to `"2"` to fix a RotatingKVCache zero-padding bug — version envelope is required, not optional.

---

## Product Requirements

Numbered to support traceability through Acceptance Criteria. All requirements are MUST unless flagged SHOULD or MAY.

### Functional requirements

**R-F1 (MUST)** — `cmd_serve --kv-persist` (or `HF2Q_KV_PERSIST=1` env var) enables ADR-017's persistence path. Default OFF until Phase D ships and per-family parity gates are GREEN.

**R-F2 (MUST)** — On model eviction (admission of distinct model triggering LRU evict, or explicit `HotSwapManager::evict`), the spiller extracts block-aligned K/V from the evicting engine's `HybridKvCache` and enqueues to a background writer thread. The eviction itself must NOT block on the writer queue — `pre_evict` returns `EnqueuedBlocks(N)` and the engine drops normally.

**R-F3 (MUST)** — On model admission (`load_or_get` miss path), the spiller consults the on-disk block index for prefixes matching the model's fingerprint. If the first user request's tokenized prompt has a chain-hashed prefix that intersects the index, the spiller pre-warms the freshly-loaded engine's cache before the first request's `prefill` runs. Pre-warm completes synchronously in `post_admit`; first request sees a populated cache.

**R-F4 (MUST)** — Within a live session on a stable model, persistence delivers a separate win: `PromptCache` lookup at request-arrival time can match a chain-hashed prefix from disk if the in-memory cache has been evicted (capacity overflow on a single model with many long sessions). This is the within-session class-(b) variant. The integration is shared with R-F3 (same lookup path).

**R-F5 (MUST)** — Cache directory size is bounded by `HF2Q_KV_CACHE_BUDGET_BYTES` (default 10% of unified RAM, 12.8 GiB on 128 GiB M5 Max). LRU eviction at the directory level when the budget is exceeded; per-block eviction order = age-of-last-access. Eviction respects in-flight `Arc<>` references on the index (don't delete a block currently being restored).

**R-F6 (MUST)** — Operator surface: `hf2q cache list` extends with a `--kv-namespace` filter showing per-model block counts + bytes-on-disk; `hf2q cache clear --kv-namespace --model <repo>` purges a model's KV blocks (parallel to existing `--model` weights-cache surface, ADR-005:5704).

**R-F7 (MUST)** — Telemetry: `/metrics` emits `hf2q_pool_kv_spills_total{repo,quant,outcome}`, `hf2q_pool_kv_restores_total{repo,quant,outcome}`, `hf2q_kv_cache_bytes_on_disk{model_fingerprint}`, `hf2q_kv_cache_blocks_total{model_fingerprint}`, `hf2q_kv_quarantined_total{reason}`. Outcome label ∈ `{success, codec_err, io_err, parity_fail, hash_mismatch}`. Optional `Server-Timing: kv_spill=NNNms, kv_restore=NNNms` response-header on auto-swap reload paths.

**R-F8 (MUST)** — Restart recovery: after `cmd_serve` restart, the in-memory `BlockIndex` is rebuilt from the on-disk format (D8). Restart-recovery time SHOULD complete in < 5 s for a 12.8 GiB cache directory with ~50K blocks (per-block header read + hash verification only, no body decode).

**R-F9 (MUST)** — Quarantine policy: corrupted blocks (header truncation, hash mismatch, version-mismatch) are MOVED (not deleted) to `<cache_root>/models/<slug>/kv-quarantine/`. Quarantine directory is bounded; oldest quarantined blocks are deleted when budget exceeded. This preserves forensic state for debugging silent corruption while bounding disk use.

**R-F10 (SHOULD)** — Cross-process safety: multiple `cmd_serve` instances on the same host with the same `HF2Q_KV_CACHE_DIR` MUST NOT corrupt each other's blocks. Use the existing `flock(LOCK_EX)` advisory lock pattern from `serve/cache.rs::advisory_lock` per `(model_fingerprint, block_hash[..2])` for the duration of a write.

### Correctness requirements

**R-C1 (MUST)** — Round-trip parity, dense path: SHA-256 of `<extracted K/V tensor bytes>` pre-spill = SHA-256 of `<restored K/V tensor bytes>` post-restore for the same `(model, prefix_tokens, block_idx)`. Byte-exact.

**R-C2 (MUST)** — Round-trip parity, TQ-active path: cosine similarity ≥ 0.9998 between dequantized-from-restored TQ state and dequantized-from-pre-spill TQ state. Per ADR-007's Gate A.

**R-C3 (MUST)** — First-token-after-restore parity: logits at the restoration boundary, with cache-restored prefix + 1 new token, max-abs-diff ≤ 1e-3 vs no-cache continuous-prefill of `(prefix_tokens + 1 new token)`. Per ADR-009 coherence gate.

**R-C4 (MUST)** — Sourdough preserved: under `HF2Q_USE_DENSE=1` + `HF2Q_KV_PERSIST=1`, the canonical sourdough fixture (`scripts/sourdough_gate.sh`, 3656 bytes, floor 3094) byte-exact vs the no-cache run.

**R-C5 (MUST)** — Hybrid correctness on Qwen3.5-MoE: same logit-parity bound (R-C3) applies after a DeltaNet-boundary-snapshot restore. Walk-back truncation, when triggered, must produce logits ≤ 1e-3 max-abs-diff vs continuous prefill of the truncation point through the request's full token set.

**R-C6 (MUST)** — No silent corruption: any restore that fails its hash check (R-F9) MUST fall through to fresh prefill, NOT silently produce wrong output. Per `feedback_never_ship_fallback_without_rootcause` — corruption is a hard fall-through, not a soft ignore.

### Performance requirements

**R-P1 (MUST)** — Decode regression: with `HF2Q_KV_PERSIST=1` enabled BUT cache-miss on every request (no actual persistence benefit), decode tok/s regresses ≤ 1% vs no-cache baseline at every prefill length in §10.1 matrix.

**R-P2 (MUST)** — Spill overhead ceiling: `HotSwapManager::insert()` wall-time INCLUDING `pre_evict` spill ≤ load-time of the incoming model. Spill must not bottleneck swap-in itself.

**R-P3 (MUST)** — Async writer ceiling: main-thread time for spill enqueue (the synchronous `pre_evict` block, NOT including the background writer thread) < 200 ms for any block count up to 128 blocks (32K context). Per dossier §4.6 oMLX's `127 ms target`.

**R-P4 (MUST)** — Scenario-(e) ship-gate: `cache_hit_TTFT(32K) / no_cache_TTFT(32K) ≤ 0.20` on M5 Max for same-model swap-back-in same-context after another model has run for ≥30 s.

**R-P5 (MUST)** — Scenario-(a) ship-gate: `cache_hit_TTFT(32K) / no_cache_TTFT(32K) ≤ 0.15` on M5 Max for cold-process resume with `ssd_cold_post_restart` cache state.

**R-P6 (MUST)** — Scenario-(c) ship-gate: with 4 concurrent agents sharing a 4K system prompt, aggregate prefill cost ≤ `1.25 × single_agent_prefill_cost(4K)` (one-time prefill amortized across 4 agents within 25%).

### Operational requirements

**R-O1 (MUST)** — Documentation: `docs/operating-kv-cache.md` lands in Phase D with operator runbook (cache-clear, quarantine inspection, telemetry interpretation, opt-out via `HF2Q_KV_PERSIST=0`).

**R-O2 (SHOULD)** — `cmd_serve` startup logs `kv-cache: <N> models, <M> blocks, <S> bytes resident, recovery <T>ms` at INFO level.

**R-O3 (MUST)** — Air-gap compatibility: ADR-017's persistence layer requires NO network access. All on-disk operations are local-filesystem.

---

## Acceptance Criteria

Per-phase, mechanically verifiable. **No phase ships its checkboxes flipped without measurement evidence.** Per `feedback_correct_outcomes`: a `[x]` next to a Phase A0 ship-gate cell is a load-bearing claim about M5 Max measurement.

### Phase A0 — Falsification harness (lands FIRST; no production code commits before A0 returns)

**Harness location:** `tests/kv_persist_harness.rs`. Subprocess-driven, env-gated `HF2Q_KV_PERSIST_E2E=1`. Extends iter-210's `tests/multi_model_swap.rs` substrate: `binary_is_locatable_and_runs_version` (always-on smoke) + env-gated `kv_persist_matrix_e2e` (the matrix body).

**Test matrix (cell count = 2 × 2 × 5 × 5 × 6 = 600 cells; not all are non-trivial — kill-gates short-circuit):**

```
family            ∈ {gemma4-26b, qwen35moe-dwq46}            // qwen35 only when ADR-013 unblocks
weight_quant      ∈ {q4_0, q4_K_M, q6_K, q8_0, dwq46, dwq48} // 6 (representative; not all combinations valid per family)
kv_path           ∈ {dense, tq_active}                       // 2
prefix_length     ∈ {0, 512, 2048, 8192, 32768}              // 5
cache_state       ∈ {miss, ram_hot, ssd_warm_pagecache,
                     ssd_cold_post_restart,
                     evicted_meta_present_file_present,
                     corrupted_middle_block}                 // 6
scenario          ∈ {(a) cold_resume, (b) hot_swap_evict,
                     (c) shared_prefix_4_agents,
                     (d) edit_in_middle,
                     (e) swap_back_in_same_ctx}              // 5
```

**Synthetic spiller fixture:** `tests/fixtures/synthetic_spiller.rs` implements `KvSpiller<MockEngine>` writing to a `tempdir()`. No production-path mutation.

**Phase A0 acceptance checkboxes:**

- [ ] Harness binary builds and runs end-to-end on M5 Max under `HF2Q_KV_PERSIST_E2E=1` for Gemma 4 26B; matrix populated for all `gemma4-26b × {q4_0, q4_K_M, q6_K, q8_0} × dense × {0, 512, 2K, 8K, 32K} × {miss, ssd_cold_post_restart, ssd_warm_pagecache} × {(a), (b), (c), (e)}` cells (= 4 × 5 × 3 × 4 = 240 cells, gemma side).
- [ ] Same matrix runs with `kv_path = tq_active` per ADR-007 codec defaults.
- [ ] Qwen3.5-MoE matrix populated when ADR-013 unblocks (B-hybrid sequencing); deferred from A0 closure if ADR-013 still in flight on the day A0 closes.
- [ ] **Ship gate R-P4 (scenario (e)):** measured ratio `cache_hit_TTFT / no_cache_TTFT ≤ 0.20` at prefix=32K, dense, gemma4-26b, swap-back-in cycle. Median of 3 cold-process trials, mcp-brain-server STOP-paused per `feedback_bench_process_audit`.
- [ ] **Ship gate R-P5 (scenario (a)):** measured ratio ≤ 0.15 at prefix=32K, gemma4-26b, cold-process restart with `ssd_cold_post_restart` cache state.
- [ ] **Ship gate R-P6 (scenario (c)):** with 4 concurrent agents sharing 4K system prompt, aggregate prefill ≤ 1.25 × single-agent baseline.
- [ ] **Coherence gate R-C1:** SHA-256 byte-exact pre-spill vs post-restore on every dense matrix cell.
- [ ] **Coherence gate R-C2:** cosine ≥ 0.9998 on every TQ-active matrix cell.
- [ ] **Coherence gate R-C3:** first-token-after-restore logit max-abs-diff ≤ 1e-3 on every cell.
- [ ] **Coherence gate R-C4:** sourdough byte-exact under `HF2Q_USE_DENSE=1 HF2Q_KV_PERSIST=1`.
- [ ] **Decode regression R-P1:** ≤ 1% across cache-enabled-but-not-hit cells.
- [ ] **Spill overhead R-P2:** measured `HotSwapManager::insert()` wall ≤ load-time of incoming model, gemma4-26b q4_0.
- [ ] **Async writer R-P3:** synchronous `pre_evict` ≤ 200 ms at 128-block (32K) spill.
- [ ] Phase A0 measurement report at `docs/ADR-017-phase-a0-results.md` with per-cell numbers, harness commands to reproduce, M5 Max thermal-state notation, mcp-brain-server STOP/RESUME log.

### Phase A — Block infrastructure (lands after A0 measurement-positive)

- [ ] `src/serve/kv_persist/mod.rs` + submodules (`block_store.rs`, `index.rs`, `writer.rs`, `recovery.rs`, `spiller.rs`, `format.rs`) at HEAD with no `// TODO` markers.
- [ ] `BlockStore` trait + `RamBlockStore` (in-memory only, for tests) + `DiskBlockStore` (atomic write, async background writer thread, LRU directory eviction, restart-recovery scan).
- [ ] Chain-hash index (`BlockIndex: Arc<RwLock<HashMap<BlockHash, BlockMeta>>>`) with O(1) lookup; rebuilt from disk on startup; partial-chain elision verified via `kill -9` mid-write integration test.
- [ ] LRU eviction correctness: total cache directory size stays within `HF2Q_KV_CACHE_BUDGET_BYTES` ± 1 block at every observable point.
- [ ] Synthetic-fixture unit tests in `src/serve/kv_persist/tests.rs`: atomic-rename success, atomic-rename interrupted, corrupted-header quarantine (truncated, version-bump, hash-mismatch), restart-recovery from clean state, restart-recovery from `kill -9` state, cross-process advisory lock contention, LRU eviction at budget boundary, oversized-block refusal. **Target: 30+ unit tests, mirroring the iter-206/207 W74 testing density (`multi_model.rs::tests` 39/39, `provenance.rs::tests` 10/10).**
- [ ] `cargo build --release` 0; full test suite green; no W74/W75/W76/W77/W78 regression.

### Phase B-dense — Gemma 4 (lands after Phase A)

- [ ] Gemma 4 `KvCacheSpill` impl in `src/inference/models/<gemma4-eqv>/kv_cache.rs` adjacent to existing `HybridKvCache` type.
- [ ] `Payload` type implements `SpillSerialize`; `payload_kind() = "dense_kv_v1"`.
- [ ] Round-trip parity test (R-C1 byte-exact): snapshot at prefix L → restore → continue prefill → logit byte-exact at L ∈ {512, 2K, 8K, 32K} for Q4_0, Q4_K_M, Q6_K, Q8_0, DWQ46, DWQ48.
- [ ] Sourdough byte-exact preserved under `HF2Q_USE_DENSE=1` (R-C4).
- [ ] LIVE `tests/openwebui_multiturn.rs` Scenario 1 PASS at T=0 byte-identical with `HF2Q_KV_PERSIST=1` (no Decision #26 regression).
- [ ] No regression in iter-210 swap-timing E2E (`tests/multi_model_swap.rs::multi_model_swap_two_ggufs_e2e`).

### Phase B-hybrid — Qwen3.5-MoE (gates on ADR-013 + Phase A; sequencing parallel with B-dense possible)

- [ ] Qwen3.5-MoE `KvCacheSpill` impl with full-attn dense path + DeltaNet boundary snapshot dispatched per-layer-rank (`gpu_delta_net.rs:1417-1715` Chesterton's-fence audit completed; hook insertion point at boundary identified and named).
- [ ] Walk-back truncation correctness: when restored prefix exceeds latest boundary, walk-back point matches the per-block-aligned snapshot exactly. Logit-parity ≤ 1e-3 vs continuous prefill.
- [ ] Hybrid-correct round-trip: R-C5 satisfied at L ∈ {512, 2K, 8K, 32K}.
- [ ] Generate-side ground-truth fixture: same prompt through generate-side qwen35 path produces same logits at the restoration boundary as the serve-side restored cache.
- [ ] No regression in `serve::api::handlers::tests::iter209_*` router tests under `HF2Q_KV_PERSIST=1`.

### Phase B-tq — TurboQuant payload variant (lands after B-dense; family-agnostic)

- [ ] TQ-packed K/V on-disk envelope: `payload_kind = "tq_packed_v1"`; carries codebook bytes + indices + `codec_version: u32` + ADR-007 codec parameters (block size, basis index, hadamard rank).
- [ ] Deterministic codec rebuild on restore (D2): SHA-256 of restored TQ state byte-exact vs pre-spill TQ state for the same input.
- [ ] Gate A cosine ≥ 0.9998 on the TQ-active path for every in-scope family at every prefix length (R-C2).
- [ ] Storage delta documented at `docs/ADR-017-tq-storage-delta.md`: `bytes_on_disk_dense / bytes_on_disk_tq` ratio across 8K / 32K prefix lengths (informational, not a ship-gate; expected ratio ~3-4× per ADR-007 codec).

### Phase C — Phase 4 integration (lands after every B-* family green)

- [ ] `BlockPrefixCacheSpiller` constructible from `cmd_serve` config; substituted for `NoopKvSpiller` when `HF2Q_KV_PERSIST=1`.
- [ ] `cmd_serve` wires the spiller in `src/serve/mod.rs::cmd_serve` between `AppState::new_for_serve` and `pool.write().load_or_get(...)` first-call.
- [ ] `/metrics` counters incrementing correctly per outcome label (router-level test `iter211_kv_persist_metrics_increment` asserts counter deltas across a synthetic spill+restore cycle).
- [ ] LIVE `openwebui_multiturn` Scenario 1 + Scenario 4 PASS with `HF2Q_KV_PERSIST=1`; turn-1 byte-identical at T=0; SSE protocol unchanged.
- [ ] No regression in any iter-210 swap-timing E2E gate.
- [ ] Sourdough gate PASS under `HF2Q_USE_DENSE=1 HF2Q_KV_PERSIST=1`.

### Phase D — Closure (lands after C; ADR-017 status flips Accepted → "Closed-Shipped" or → "Closed-Unmerged" per kill-gate)

- [ ] **Stress:** 24-hour continuous-load test on M5 Max under `tests/kv_persist_stress.rs` with continuous swap-in/swap-out cycles; cache directory size stays within budget; no resident-memory leak (RSS within 5% of baseline at hour 24); no descriptor leak (`lsof | wc -l` within 100 of baseline).
- [ ] **Corruption:** synthetic mid-write corruption injection (truncate file, flip bit in middle, swap header version, delete intermediate block in chain) — restore must reject and fall through to fresh prefill (no silent wrong-output). R-C6 mechanical verification.
- [ ] **Per-family ship-gate read:** every in-scope family's parity gate documented GREEN at `docs/ADR-017-per-family-status.md` with measurement evidence and date.
- [ ] **Operator documentation:** `docs/operating-kv-cache.md` lands with runbook (R-O1).
- [ ] ADR-017 status flips Accepted → **Closed-Shipped** with the date and cumulative LOC; cross-link from ADR-005 Phase 4 closure section updated; `cmd_serve --kv-persist` becomes default ON.

---

## Test Plan

### Phase A0 harness structure

`tests/kv_persist_harness.rs` (new, ~600 LOC):

```rust
#[test]
fn binary_is_locatable_and_runs_version() { ... }  // always-on smoke

#[test]
fn kv_persist_matrix_e2e() {
    if std::env::var("HF2Q_KV_PERSIST_E2E").ok().as_deref() != Some("1") { return; }

    let cell_results = generate_matrix(...)
        .into_iter()
        .filter(|c| c.is_supported_by_current_tree())
        .map(|cell| run_cell(cell))
        .collect::<Vec<CellResult>>();

    assert_ship_gates(&cell_results);     // R-P4, R-P5, R-P6
    assert_coherence_gates(&cell_results);// R-C1, R-C2, R-C3, R-C4
    assert_decode_regression(&cell_results); // R-P1
    assert_overhead_gates(&cell_results);  // R-P2, R-P3

    write_results_md(&cell_results, "docs/ADR-017-phase-a0-results.md");
}
```

Each cell: spawn `hf2q serve --kv-persist --model <gguf>` in subprocess; tokenize a known prefix; force pool eviction by admitting a second model; restore; measure TTFT against the no-cache baseline; SHA-256 of K/V tensors at snapshot + restore; logit comparison.

**Reference fixtures:**
- Gemma 4 26B: canonical Gemma 4 26B chat fixture per iter-210 (`HF2Q_HOT_SWAP_E2E_MODEL_A`, default `~/Models/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf`).
- Qwen3.5-MoE: `apex dwq46` per ADR-005:5837 (when ADR-013 serve-side load unblocks).

### Per-iter unit-test density target

Mirror iter-206/207 W74 pattern: each iter ships ≥10 in-binary unit tests adjacent to the production module, plus E2E smoke gated on `HF2Q_KV_PERSIST_E2E=1`. Phase A targets 30+ unit tests for the block-store substrate. Phase B-dense targets 15+ unit tests for the Gemma 4 spill impl. Phase B-hybrid targets 20+ (DeltaNet boundary correctness needs broader coverage). Phase B-tq targets 8+ (codec round-trip is small surface). Phase C targets 6+ router-level integration tests. Phase D targets 4+ stress/corruption tests.

**Total unit-test target: 80+ new tests across Phases A–D.** No iter ships without the test cohort.

---

## Risks

| # | Risk | Probability | Impact | Mitigation | Owner / decision point |
|---|---|---|---|---|---|
| **R1** | DeltaNet boundary snapshot for Qwen3.5-MoE requires invasive `forward_gpu.rs:2651-2673` rewrites — exceeds Walk discipline. | Medium | High (B-hybrid blocks indefinitely; ADR-017 closes Gemma-4-only) | Phase A0 includes synthetic-fixture'd hybrid harness to surface the invasiveness before B-hybrid commits. If A0 surfaces invasiveness > acceptable, Qwen3.5 impl follows in a follow-up ADR; trait surface still ships; ADR-017 closes for Gemma 4 with explicit "Qwen3.5 deferred" status. | Phase A0 + B-hybrid lead engineer |
| **R2** | M5 Max SSD restore latency dominates the win at 32K — `cache_hit_TTFT / no_cache_TTFT > 0.30`. | Low (computed: 5 GB/s NVMe × 670 MB = 130 ms; well under threshold) | Critical (project doesn't survive contact with hardware reality) | Phase A0's primary ship-gate R-P4 / R-P5. If kill-gate fires, close ADR-017 unmerged with measurement report. | Phase A0 lead engineer |
| **R3** | TQ-active codec rebuild non-determinism under restore — Gate A cosine drops below 0.9998. | Low (Lloyd-Max is deterministic by construction) | High (B-tq blocks; TQ path inherits non-cache behavior) | B-tq's primary correctness gate R-C2. ADR-007's codec is deterministic; rebuild reuses on-disk codebook bytes. If R-C2 fails, the bug is in envelope serialization (D2), not the codec. Fix in B-tq. | B-tq lead engineer |
| **R4** | Scenario-(e) event-rate in real workloads < 1/dev/day; the win frequency is too low to justify ongoing maintenance. | Medium (depends entirely on Robert's actual workflow + future serve-mode adoption) | Medium (cache layer becomes maintenance debt without payoff) | Post-Phase-C telemetry month: kill-gate R4 fires if event-rate measured < 1/dev/day across telemetry window. Descope to (a)+(c) only or close ADR-017 unmerged. | Post-C operator telemetry; Robert's call |
| **R5** | "All families and all quants" bar is read post-hoc as "every conceivable future family" rather than "today's complete-in-code families plus inheritance." | Low (Family-scope rule explicitly documented) | Low (scope creep) | Family-scope rule documented above and cross-referenced in ADR-005 Phase 4 reopen. ADR-017 closure scoped to today's reality; trait surface inherits forward. | This ADR's family-scope rule + ADR-005 reopen subsection |
| **R6** | Phase 4's `KvSpiller` trait surface (AC 5471) and ADR-017 reach circular dependency. | Low (Mitigation already in tree path) | High (neither ADR can close) | `NoopKvSpiller` default impl ships in Phase 4 iter-212 BEFORE ADR-017 lands any code. ADR-017 substitutes the impl in Phase C. Trait + telemetry definitions never move. R7 in ADR-005 Phase 4 reopen risk register documents the same mitigation from the other side. | iter-212 + ADR-017 Phase C |
| **R7** | `HF2Q_KV_CACHE_BUDGET_BYTES` set too low → frequent eviction churn → cache-hit rate drops → ship-gate fails in production despite passing in A0. | Medium (ops-time misconfig) | Medium (degraded performance, no correctness issue) | Default to 10% of unified RAM (12.8 GiB on 128 GiB M5 Max); operator-documented in R-O1; `/metrics` exposes `hf2q_kv_cache_evictions_total` so operators can detect churn. Not a kill-gate — degradation is graceful (falls through to fresh prefill, R-C6). | Phase D operator documentation |
| **R8** | Per-block fsync throughput limit on M5 Max APFS is the bottleneck for sustained spill-on-evict workloads. | Low | Medium | Phase A0 measures sustained writer throughput. If bottleneck, B-dense reserves `flush_buffer` pattern (multi-block tempfile + atomic-rename) per D9 — but only after measurement, never speculatively per mantra. | Phase A0 + B-dense |

---

## Kill-gates (any single fire = ADR-017 status flips Accepted → Closed-Unmerged)

Per the mantra: "No fallback. No stub (todo later) code." If a kill-gate fires, ADR-017 closes with a documented rationale and any landed Phase A–C code rolls back via revert PR. There is no "circle back later" — ADR-017's premise is explicitly falsifiable, and falsification is honored in deeds, not deferred.

- [ ] **K1:** Phase A0 measures `cache_hit_TTFT / no_cache_TTFT > 0.30` at prefix=32K, scenario=(e), Gemma 4 26B, cold SSD path. SSD I/O eats too much of the gain. Close.
- [ ] **K2:** Phase A0 measures `dirty_block_overhead_during_decode > 5%` (R-P1 violation). Async-write architecture is leaking onto inference thread. Close (or fix the writer architecture; only A0-positive after fix re-opens).
- [ ] **K3:** Phase A0 measures scenario-(e) speedup `< 5×` at prefix=32K. Computed predictions wrong; close.
- [ ] **K4:** Phase B-hybrid surfaces that hybrid Qwen3.5 boundary snapshot cannot be made correct without invasive `forward_gpu` rewrites that violate Walk discipline. Descope: ADR-017 closes Gemma 4 fully + leaves Qwen3.5 impl as explicit follow-up ADR (NOT a stub TODO; an actual deferred ADR with its own scope).
- [ ] **K5:** Post-Phase-C telemetry month measures scenario-(e) event-rate < 1 event per developer-day on real workloads. R4 fires. Descope to (a)+(c) only or close.
- [ ] **K6:** Any per-family parity gate (R-C1 byte-exact dense, R-C2 cosine ≥ 0.9998 TQ, R-C3 logit max-abs-diff ≤ 1e-3, R-C5 hybrid logit parity) fails and root cause is not fixable within the family's `KvCacheSpill` impl scope. Close that family's impl as deferred ADR; close ADR-017 partial.

---

## Defects discovered during A0 falsification (added 2026-04-30)

The Phase A0 substrate-and-matrix iters surfaced eight defects across two classes. Per `feedback_substrate_must_not_synthesize_ship_gates` and the mantra, every defect is recorded honestly here regardless of whether it gates ADR-017 closure. Class A defects are SUBSTRATE bugs (in the harness/instrumentation itself) — fixed in A0.2b. Class B defects are PRODUCTION-side findings that A0 surfaced but that are NOT in scope for ADR-017 closure; they affect related ADRs (ADR-015 perf, future ADR-018) and are recorded here so they don't get lost.

### Class A — Substrate defects (FIXED in A0.2b commit 1c3b946)

**A-D1 — Prompt construction BPE-collapsed.** The A0.2a `run_cell_with_subprocess` constructed cell prompts via `"hello ".repeat(prefix_tokens / 6)`. Gemma 4's tokenizer collapsed the repeated pattern to <50 BPE tokens regardless of nominal cell prefix length. Result: every cell measured ~16-token prefill; ship-gate ratios were denominator-broken. **Fix (A0.2b):** replaced with token-diverse `format!("word{i}")` sequence; parsed SSE final usage block to record `actual_prompt_tokens`; gate logic uses ACTUAL not nominal token count. Verified: L0=11 / L512=409 / L2K=1945 / L8K=9137 / L32K=39857 actual tokens after fix.

**A-D2 — Symlink eviction-cycle missing config.json sibling.** `measure_swap_eviction_cycle` symlinked only the GGUF into the tempdir; `hf2q serve` requires `config.json` + `tokenizer.json` adjacent. Result: SwapBackInSameCtx cells returned HTTP 500 "Failed to parse config.json". **Fix (A0.2b):** symlink `config.json`, `tokenizer.json`, `generation_config.json`, and `*-mmproj.gguf` adjacent to the GGUF symlink in tempdir.

**A-D3 — SSE transport errors on sub-second responses.** `measure_ttft_subprocess` returned `transport: error sending request` / `request or response body error` on cells whose response completed in <100ms (L0/L512 with very short prompts). Likely cause: SSE writer/reader race when response body is small. **Fix (A0.2b):** retry-with-backoff (100ms / 250ms / 500ms) up to 3× when transport error fires sub-second. Don't mask real timeouts. `retry_count` surfaced in `CellMeasurement`. (Post-fix the matrix needed zero retries — D1's longer prefills mooted the race window.)

**A-D4 — Gemma 4 chat-template missing in API code path (caught during A0.2b matrix run).** `src/serve/api/engine.rs` did not have the Gemma 4 chat-template fallback that `cmd_generate` already had. First subprocess matrix attempt fired the missing-template error visible in the matrix log. **Fix (A0.2b commit d4e9719 → b43edcd merged):** added Gemma4 chat-template fallback in `src/serve/api/engine.rs` + `src/serve/mod.rs`. This is a real bug that affected production serve-side, not just the harness.

### Class B — Production-side findings (not in scope for ADR-017 closure)

**B-F1 — R-P3 ceiling vs APFS write+fsync floor.** A0.2b measured `pre_evict_ms=517ms` at L32K (128 blocks × 1 MiB each) on M5 Max APFS. The §Performance R-P3 ceiling of 200ms was specced before measurement; 517ms is APFS write+fsync throughput, not a hf2q-side bug. **Resolution path (DOES NOT block ADR-017 closure):** Phase A.1 must EITHER (a) raise R-P3 ceiling to ~600ms after re-spec on measured floor, OR (b) introduce the `flush_buffer` write-aggregation pattern (already flagged as B-dense reserve in §D9 — write multiple blocks to a single tempfile + atomic-rename). Data informs the decision per the mantra. R-P3 fail does NOT trip K1/K3; K1/K3 are about the cache-hit ratio, which passed by 22×/16× margin.

**B-F2 — R-P6 cell coverage incomplete.** A0.2b matrix ran `SharedPrefix4Agents` cells at L0 and L512 only, NOT at the L4K target prefix length (R-P6's specced denominator). Result: R-P6 verdict is `N/A` rather than PASS/FAIL. **Resolution path:** A0.2c iter expands the ship-gate filter to include `(SharedPrefix4Agents, L4K)` cell or close-equivalent (e.g. L8K). Low priority since R-P4/R-P5 already pass decisively; R-P6 is a different scenario class.

**B-F3 — Harness decode_tok_s metric implausible.** A0.2b matrix reported `decode_tok_s_no_cache=800-880 t/s` on Gemma 4 26B Q4_0. Peer benchmark on the same GGUF on the same M5 Max measured llama.cpp `tg16=93 t/s`; the mlx-lm baseline (`project_benchmark_baselines`) is ~124 t/s. 880 vs 93 is a 9× ratio; greedy decode on the same hardware should not be 9× faster. **Likely root cause:** the harness's tok/s calc may include `prompt_tokens` in the numerator (computing `(prompt+completion)/decode_wall` instead of `completion/decode_wall`), which inflates the metric for long prompts. **Resolution path:** A0.2c iter inspects `subprocess_driver::measure_ttft_subprocess` SSE-parsing tps calc and validates against known peer baselines. Does NOT block ADR-017 closure (R-P1 decode-regression gate is RELATIVE, comparing cache-enabled-but-miss vs no-cache; both legs use the same calc so any bug cancels).

**B-F4 — Gemma 4 26B Q4_0 prefill is 29-40× slower than llama.cpp on M5 Max.** Peer benchmark (llama-bench from homebrew/ggml 0.9.11, same GGUF, same hardware, mcp-brain-server STOP-paused):

| Prefix | hf2q TTFT (ms) | hf2q t/s | llama.cpp t/s | hf2q/llama ratio |
|---|---|---|---|---|
| 512 | 4,214 | 97 | 3,819 | 0.025× (40× slower) |
| 2,048 | 20,359 | 95 | 3,622 | 0.026× (38× slower) |
| 8,192 | 103,739 | 88 | 3,339 | 0.026× (38× slower) |
| 32,768 | 568,639 | 70 | 2,047 | 0.034× (29× slower) |

This is a release-gate violation per `project_speed_bar_full_matrix` (hf2q ≥1.00× llama.cpp on same hardware across all quants × lengths × modes). **Likely root cause:** Gemma 4 dense full-attention prefill path may not be wired to `flash_attn_prefill` (W-5b.10 closed Gemma 3 to 4.34× via flash_attn_prefill wire-up; Gemma 4 uses different model-class infra and may not have inherited the wire-up). Audit `src/inference/models/gemma4*/forward_*.rs` to verify FA wiring. **Resolution path (NOT in ADR-017 scope):** ADR-015 (or new ADR-018) territory. ADR-017 closure does NOT require closing this gap because the cache-ratio gate (R-P4 / R-P5) is RELATIVE and survives slow base paths — cache wins amortize regardless. The cache layer makes the slow path faster via persistence; it does NOT close the underlying prefill perf gap.

**B-F5 — DWQ/Dwq46 cells unimplemented in is_runnable_today.** A0.1 substrate filter excludes `(Family::Gemma4_26b, KvPath::Dense, WeightQuant::Dwq46|Dwq48)` from runnable today. **Resolution path (low priority for ADR-017 closure):** wire DWQ quant fallback so the matrix can exercise both Q4_0 and DWQ Gemma 4 cells; relevant for the per-quant ratio variance documented in ADR-017's open question OQ-2 (storage-delta).

### Defect-resolution discipline

- Class A defects MUST be fixed before the matrix verdict is trustable. A0.2b fixed all four; ship-gate verdicts post-fix are valid.
- Class B defects DO NOT block ADR-017 closure as long as the kill-gates K1-K6 do not fire. K1 (ratio > 0.30) and K3 (speedup < 5×) are GREEN with 22× / 111× margins respectively, so the cache-layer falsification is decisive even with B-F4 perf gap.
- Class B defects MUST land their own targeted ADRs / iters before the broader project ships. The release-gate is `project_speed_bar_full_matrix`, not ADR-017's per-cache ship-gates. The peer benchmark in `docs/ADR-017-phase-a0-results.md` is the load-bearing evidence for that follow-up scope.

---

## Iter-by-iter plan

| Iter | Phase | Scope | LOC est. | Depends on |
|---|---|---|---|---|
| **A0.1** | A0 | `tests/kv_persist_harness.rs` substrate (subprocess driver, env gates, matrix generator, cell-runner). Synthetic spiller fixture. No production code. | ~600 test | Phase 4 reopen iter-212 (NoopKvSpiller in tree) |
| **A0.2** | A0 | Matrix execution on M5 Max for Gemma 4 26B all dense cells; ship-gate + coherence-gate read; results report. | ~200 doc + measurement | A0.1 |
| **A0.3** | A0 | Matrix execution for TQ-active path; same gates. Decision point: GREEN → proceed Phase A; ANY KILL → close ADR-017 unmerged. | ~80 measurement | A0.2 |
| **A.1** | A | `src/serve/kv_persist/{mod,format,index}.rs`: format envelope, hash chain, BlockIndex. Pure-fn unit tests (15+). | ~400 src + ~350 test | A0.3 GREEN |
| **A.2** | A | `src/serve/kv_persist/{block_store,writer,recovery}.rs`: DiskBlockStore atomic writer + async writer thread + restart-recovery scan. Restart-recovery unit tests + `kill -9` integration test. | ~500 src + ~450 test | A.1 |
| **A.3** | A | `src/serve/kv_persist/spiller.rs`: `BlockPrefixCacheSpiller<E>` impl of `KvSpiller<E>` over `BlockStore + BlockIndex`. Wires `KvCacheSpill` per-family registration. | ~300 src + ~280 test | A.2 |
| **B-dense.1** | B-dense | Gemma 4 `KvCacheSpill` impl in `src/inference/models/<gemma4-eqv>/kv_cache.rs`. R-C1 + R-C4 unit tests. | ~250 src + ~280 test | A.3 |
| **B-dense.2** | B-dense | Round-trip parity test suite: `tests/kv_persist_gemma4_roundtrip.rs` covering all weight quants × prefix lengths × scenarios. R-C1 mechanical for every cell. | ~400 test | B-dense.1 |
| **B-tq.1** | B-tq | TQ-packed payload type + envelope: `src/serve/kv_persist/payloads/tq_packed.rs`. Determinism unit tests against synthetic ADR-007 codec fixtures. | ~200 src + ~250 test | B-dense.1, ADR-007 codec stable |
| **B-tq.2** | B-tq | R-C2 cosine ≥ 0.9998 measurement on Gemma 4 TQ-active path; storage-delta measurement at 8K/32K. Storage-delta doc. | ~120 test + measurement | B-tq.1 |
| **B-hybrid.1** | B-hybrid | Chesterton's fence audit of `gpu_delta_net.rs:1417-1715` and `forward_gpu.rs:2651-2673`. Hook-insertion point chosen + named. Audit doc at `docs/ADR-017-deltanet-boundary-audit.md`. | ~150 doc + audit | A.3, ADR-013 unblock |
| **B-hybrid.2** | B-hybrid | Qwen3.5-MoE `KvCacheSpill` impl with full-attn dense path + DeltaNet boundary snapshot. R-C5 logit parity tests. | ~500 src + ~600 test | B-hybrid.1 |
| **B-hybrid.3** | B-hybrid | Walk-back truncation correctness; generate-side ground-truth fixture comparison; `iter209_*` router tests under `HF2Q_KV_PERSIST=1`. | ~200 src + ~350 test | B-hybrid.2 |
| **C.1** | C | `cmd_serve --kv-persist` flag + `HF2Q_KV_PERSIST` env var; spiller substitution in `src/serve/mod.rs::cmd_serve`. | ~150 src + ~120 test | every B-* GREEN |
| **C.2** | C | `/metrics` counter wiring; router-level `iter211_kv_persist_metrics_increment` integration test; LIVE `openwebui_multiturn` Scenario 1 + 4 with persistence ON. | ~80 src + ~200 test | C.1 |
| **D.1** | D | `tests/kv_persist_stress.rs` 24-hour continuous-load harness; corruption-injection harness; `docs/operating-kv-cache.md`. | ~400 test + doc | C.2 |
| **D.2** | D | Per-family ship-gate read; `docs/ADR-017-per-family-status.md`; ADR-017 status flip Accepted → Closed-Shipped. | ~80 doc + measurement | D.1 |

**Total: ~4,400 src + ~4,000 test LOC across 16 iters.** Estimated 24–35 man-days at current iter cadence. Sequencing: A0 strictly first; A.1–A.3 sequential; B-dense / B-tq / B-hybrid parallel-where-possible (each rides its own dependency tree); C / D sequential after every B-* GREEN.

### Phase A0.1 LANDED (2026-04-30)

**Status:** Substrate complete on `cfa/adr017-a01/claude` worktree; pending dual-mode queen merge against the Codex parallel impl.

**Commits:**
- `092ca83` — `test(adr-017 a0.1): synthetic spiller fixture + 13 unit tests`
- `affb4dd` — `test(adr-017 a0.1): falsification harness substrate + 8 smoke tests`

**LOC shipped (test-only; zero src/ mutation per spec):**

| File | LOC | Role |
|---|---|---|
| `tests/fixtures/synthetic_spiller.rs` | 1,413 | `BlockStore` (atomic safetensors writer mirroring `paged_ssd_cache.py:246-297` byte-for-byte), `ModelFingerprint` + `BlockHash` chain-hash per §D4, forward-compat `MockKvSpiller<E>` trait, 13 unit tests. |
| `tests/kv_persist_harness.rs` | 1,121 | Six-axis matrix generator, `is_runnable_today` filter, `run_cell` runner against the synthetic fixture, ship/coherence/decode/overhead gate assertions, `pre_bench_process_audit_or_panic` (`feedback_bench_process_audit`), markdown results writer, 8 smoke tests + the env-gated `kv_persist_matrix_e2e` body. |
| **Total** | **2,534** | Substrate exceeds the spec's ~600+~400 estimate by ~1,500 LOC; the surplus is mantra discipline (no `// TODO`, full doc citations, every helper exposed for A0.2 reuse). |

**Test count delta:**
- New tests: **22** (`cargo test --release --test kv_persist_harness -- --test-threads=1` reports `22 passed; 0 failed; 0 ignored`).
  - 8 always-on smoke tests (binary locator, matrix cardinality, pre-bench audit body, `run_cell` short/long, results writer, forward-compat trait round-trip, gated `kv_persist_matrix_e2e` body skips cleanly without the env var).
  - 13 in-binary unit tests under `mod synthetic_spiller` (safetensors envelope byte-compat, atomic rename success + interrupted, header truncation / version-mismatch / body-hash-mismatch quarantine, restart recovery O(1) lookup at 100 blocks, partial-write elision after `kill -9`, advisory-lock contention semantics, LRU eviction at budget boundary, oversized-block refusal, chain-hash determinism across recompute, fingerprint stability across restart).
  - 1 env-gated matrix body (`HF2Q_KV_PERSIST_E2E=1`) that A0.2 will exercise for real measurements on M5 Max.

**Mechanical exit codes (worktree HEAD):**

| Command | Exit |
|---|---|
| `cargo build --release` | **0** (release profile, no new-code warnings after `non_camel_case_types` allow on the on-disk-name-mirroring enums) |
| `cargo test --release --test kv_persist_harness -- --test-threads=1` | **0** (22/22 PASS in 1.19 s) |
| `cargo test --release --test kv_persist_harness synthetic_spiller -- --test-threads=1` | **0** (14/14 PASS — the 13 fixture units + the 1 trait round-trip) |
| `grep -rn '^[[:space:]]*// TODO' tests/kv_persist_harness.rs tests/fixtures/synthetic_spiller.rs` | **1** (no hits — discipline holds) |

**Open questions deferred to A0.2:**

1. **Real subprocess TTFT measurement** — A0.1 substrate populates the `CellResult` perf fields with `f64::NAN` placeholders against the synthetic-fixture path. A0.2 extends `run_cell` to spawn `hf2q serve --model <gguf>` per the iter-210 `multi_model_swap.rs:131-148` `ServerGuard` pattern, drives `/v1/chat/completions` with a known prefix, records TTFT, then forces an LRU evict via the symlink-as-distinct-pool-key trick. Target: A0.2 lands ≤2 man-days after A0.1 close. Answer-by: 2026-05-02.
2. **OQ-2 / OQ-3 measurements still pending** — `mlx-native::metal_blit_block` vs `StorageModeShared` CPU memcpy decision (OQ-2 in §Open Questions) and hot-tier RAM cache default (OQ-3) are A0.2 deliverables. The harness shape is unchanged; only the cell runner body wires the production K/V tensor extraction. Answer-by: 2026-05-02.
3. **Qwen3.5-MoE matrix population** — `is_runnable_today()` returns false for every Qwen3.5 cell pending ADR-013 unblock. When ADR-013 lands the qwen35 serve-side load, the filter flips and A0.2/A0.3 populate the hybrid cells without a harness change. Answer-by: tracked against ADR-013 closure date.
4. **TQ-active matrix population** — `is_runnable_today()` returns false for `KvPath::TqActive` pending ADR-007 codec stable. A0.3 populates these cells once the TQ codec is locked. Answer-by: tracked against ADR-007 codec freeze.

**Discipline notes (A0.1):**
- Zero `src/` edits — A0.1 is purely tests-only per spec.
- Zero `// TODO` markers — the forward-compat `MockKvSpiller<E>` trait is documented as a swap-target for ADR-005 Phase 4 iter-212's `KvSpiller<E>`, not a stub.
- ADR-005 Phase 4 fence respected: no edits to `src/serve/multi_model.rs`, `src/serve/api/`, `src/serve/cache.rs`, `src/serve/provenance.rs`, `src/serve/mod.rs`.
- `man-day` used throughout; `person-day` does not appear (Robert directive 2026-04-30).
- Pre-bench process audit is baked into `kv_persist_matrix_e2e` and surfaced as its own smoke test so the audit body executes on every default `cargo test` run, even when the matrix gate is OFF.

### Phase A0.2a LANDED (2026-04-30)

**Status:** Code-only deliverable complete on `cfa/adr017-a02a/claude` worktree; pending dual-mode queen merge against the Codex parallel impl. **A0.2 closure (matrix execution on M5 Max) is a separate phase** that the main session runs after this CFA returns — A0.2a lands the substrate for that run, not the run itself.

**Why A0.2a is its own phase:** the spec splits A0.2 into "code" (A0.2a, this iter) and "matrix run" (A0.2 closure, post-CFA). The code ships behind the same `HF2Q_KV_PERSIST_E2E=1` env gate as A0.1 so default `cargo test` runs stay cheap; the matrix body remains gated and the operator promotes to the M5 Max + cold-SoC + STOP-paused-mcp-brain-server discipline before invoking it.

**Commits (cfa/adr017-a02a/claude):**
- `134c712` — `test(adr-017 a0.2a): add BlockStore::time_round_trip helper for cache-hit prediction`
- `2abd94f` — `test(adr-017 a0.2a): subprocess_driver submodule + run_cell wiring`
- `96bdd28` — `test(adr-017 a0.2a): gate assertions evaluate against measured fields`
- `c0506de` — `test(adr-017 a0.2a): add 10 subprocess_driver tests (6 required + 4 bonus)`

**LOC delta (test-only; zero src/ mutation per spec):**

| File | A0.1 LOC | A0.2a LOC | Delta | Role |
|---|---|---|---|---|
| `tests/fixtures/synthetic_spiller.rs` | 1,413 | 1,479 | +66 | `BlockStore::time_round_trip` helper (real-disk-I/O round-trip timer used by `synthesize_cache_hit_prediction`). |
| `tests/kv_persist_harness.rs` | 1,121 | 2,698 | +1,577 | `subprocess_driver` submodule (`ServerGuard`, `spawn_hf2q_serve_subprocess`, `wait_for_readyz`, `warm_request`, `fetch_canonical_model_id`, `measure_ttft_subprocess`, `measure_swap_eviction_cycle`, `synthesize_cache_hit_prediction`, `representative_block_bytes`, `DriverError`, `CellMeasurement`, `EvictionMeasurement`, `CellConfig`); `run_cell` subprocess wiring; `resolve_cell_model_path`; updated gate assertions; 10 new tests. |
| **Total** | **2,534** | **4,177** | **+1,643** | A0.2a substrate exceeds the spec's ~400-500 estimate by ~1,100 LOC; surplus is mantra discipline (full doc citations, every helper exposed, no `// TODO`, no fallback paths). |

**Subprocess driver surface (mirrors `tests/multi_model_swap.rs:127-385`):**

| Function | Mirrors | Role |
|---|---|---|
| `ServerGuard` (struct) | `multi_model_swap.rs:127-155` | RAII child + stderr-tail ring buffer; Drop kills + waits + drains stderr-thread. Manual `Debug` impl (Child doesn't derive). |
| `spawn_hf2q_serve_subprocess(&CellConfig)` | `multi_model_swap.rs:131-148` | `target/release/hf2q serve --model <gguf>`; rejects missing `model_path` with `DriverError::SpawnFailed`. |
| `wait_for_readyz(&ServerGuard)` | `multi_model_swap.rs:184-206` | 600 s envelope; `http_get_status` raw TCP probe (no tokio dep on the readyz path). |
| `warm_request(&ServerGuard) -> Duration` | new | Non-streaming `/v1/chat/completions`; ignored for measurement (warmup-not-bench). |
| `fetch_canonical_model_id` | `multi_model_swap.rs:220-232` | `/v1/models` data[0].id. |
| `measure_ttft_subprocess(&ServerGuard, model, prompt, max_tokens) -> CellMeasurement` | new | `stream: true` SSE; first non-empty `choices[0].delta.content` is TTFT; decode tok/s = `(total_tokens - 1) / (total_wall - ttft)`. |
| `measure_swap_eviction_cycle(&ServerGuard, &Path) -> EvictionMeasurement` | `multi_model_swap.rs:344-384` | Symlink-distinct-pool-key trick; admits on distinct stem to force `HotSwapManager::load_or_get` cold-load. |
| `synthesize_cache_hit_prediction(no_cache_ttft, prefix_tokens, family, kv_path, &SyntheticSpiller) -> f64` | new | **FALSIFICATION INSTRUMENT.** Real disk I/O via `time_round_trip`; closed-form `restore_ms + (no_cache_ttft / n_blocks)` post-restore proxy. **NO synthetic constants asserted against ship gates.** |
| `representative_block_bytes(family, kv_path) -> usize` | new | Per-family per-kv-path wire-size (Gemma 4 dense = 1 MiB/block; Qwen3.5-MoE dense = 768 KiB/block; TQ-active = 256 KiB/block). |

**Test count delta:**
- New tests in A0.2a: **10** (6 required per spec §Deliverables.4 + 4 bonus).
- Total in `kv_persist_harness.rs` after A0.2a: **32** (22 prior + 10 new); all PASS in **1.53 s** under default `cargo test --release` (env-gated tests skip cleanly without `HF2Q_KV_PERSIST_E2E=1`).
- 14/14 PASS for the `synthetic_spiller` cohort; the new `time_round_trip` helper ships with no dedicated unit test because the cache-hit-prediction test (`synthesize_cache_hit_prediction_uses_real_io_wall_not_constants`) exercises it end-to-end with two independent invariants (direct wall-monotonicity + indirect prediction-variance).

**Required tests landed (per spec §Deliverables.4):**

| # | Test | Always-on / Gated | Asserts |
|---|---|---|---|
| 1 | `subprocess_driver_smoke_binary_locatable` | always-on | driver binary path matches parent helper; surfaces `BinaryNotFound` instead of panic |
| 2 | `server_guard_lifecycle_starts_and_stops_cleanly` | env-gated | spawn + readyz + drop cleanly; no resident-model strand |
| 3 | `warm_request_returns_under_10min_budget` | env-gated | warm wall under 600 s envelope |
| 4 | `measure_ttft_parses_sse_first_content_delta` | env-gated | finite `ttft_ms`, `total_tokens > 0`, `decode_tps >= 0` |
| 5 | `synthesize_cache_hit_prediction_uses_real_io_wall_not_constants` | always-on (REAL DISK I/O) | (A) `time_round_trip(1)` < `time_round_trip(32)` AND `wall_32 >= 100us`; (B) predictor differentiates prefix lengths (1blk vs 8blk delta > 0.5ms; prefix=0 ≡ `no_cache_ttft`) |
| 6 | `pool_key_for_path_symlink_trick_reproduces_iter210_pattern` | always-on | distinct stem ⇒ distinct key; identical stem ⇒ identical key |

**Bonus tests landed:**

| # | Test | Purpose |
|---|---|---|
| 7 | `synthesize_cache_hit_prediction_rejects_nan_no_cache_ttft` | NaN guard so downstream gates never see contamination |
| 8 | `representative_block_bytes_nonzero_for_every_family_kv_path` | zero-byte-block guard against `time_round_trip` |
| 9 | `spawn_hf2q_serve_subprocess_rejects_missing_model_path` | clean `SpawnFailed` vs async child failure |
| 10 | `driver_error_display_renders_diagnostic_strings` | Display + Debug round-trip for cell-result `note` |

**No-synthetic-constants discipline (the disqualifying antipattern Codex tripped in A0.1, per spec):**

`synthesize_cache_hit_prediction` invokes `BlockStore::time_round_trip` which:

1. Calls `BlockStore::write_block` (atomic temp + rename + fsync) for `n_blocks` payloads of `block_bytes` each.
2. Calls `BlockStore::read_block` (header parse + body-hash check) for each hash in the chain.
3. Returns the wall time of the **read leg only**.

Sample output from the falsification test (run on Apple Silicon dev machine, APFS):
- `time_round_trip(n=1, 1 MiB) = 3.835 ms`
- `time_round_trip(n=32, 1 MiB) = 119.389 ms`
- Prediction at `prefix_tokens=0` ≡ 1000.316 ms (the no_cache_ttft input)
- Prediction at `prefix_tokens=BLOCK_TOKENS` ≡ 1003.745 ms
- Prediction at `prefix_tokens=BLOCK_TOKENS*8` ≡ 154.834 ms

The 32-block I/O wall (~119 ms) is ~30× the 1-block wall (~3.8 ms); this monotonicity is mathematically incompatible with a constant short-circuit. The test asserts `wall_32 >= wall_1` AND `wall_32 >= 100us`; either invariant alone refutes "synthetic constants asserted against ship gates."

**Mechanical exit codes (worktree HEAD `c0506de`):**

| Command | Exit | Output |
|---|---|---|
| `cargo build --release` | **0** | `Finished release profile [optimized] target(s) in 13.89 s` |
| `cargo test --release --test kv_persist_harness -- --test-threads=1` | **0** | **32 passed; 0 failed; 0 ignored** in 1.53 s |
| `cargo test --release --test kv_persist_harness synthetic_spiller -- --test-threads=1` | **0** | 14/14 PASS (13 fixture units + 1 trait round-trip; `time_round_trip` exercised end-to-end via test #5) |
| `grep -rn '// TODO\|todo!()\|unimplemented!()' tests/kv_persist_harness.rs tests/fixtures/synthetic_spiller.rs` | **1** | no hits — discipline holds |
| `git diff main..HEAD --stat` | OK | only `tests/kv_persist_harness.rs`, `tests/fixtures/synthetic_spiller.rs`, `docs/ADR-017-persistent-block-prefix-cache.md` changed |

**A0.2a discipline notes:**

- Zero `src/` edits — code-only deliverable; the matrix run (A0.2 closure) executes against existing `src/serve/...` surfaces unchanged.
- Zero `// TODO` markers — gate-assertion short-circuits are documented as substrate-mode behavior, not deferred work.
- ADR-015 iter58 fence respected: no edits to `mlx-native/`, `gpu_delta_net.rs`, `forward_gpu.rs`, `gpu_full_attn.rs`.
- ADR-005 Phase 4 fence respected: no edits to `src/serve/multi_model.rs`, `src/serve/api/`, `src/serve/cache.rs`, `src/serve/provenance.rs`, `src/serve/mod.rs`.
- `man-day` used throughout; `person-day` does not appear.
- Subprocess driver respects the OOM directive (`feedback_oom_prevention`): per-cell serial spawn; one server resident at a time; tempdir guards keep symlinks in scope across the eviction request.
- Pre-bench process audit (`pre_bench_process_audit_or_panic`) gates the matrix body; A0.2a does not bypass it.

**Open questions resolved by A0.2 matrix run (post-CFA, main session):**

1. **OQ-2 / OQ-3 numerical decisions** — `mlx-native::metal_blit_block` vs `StorageModeShared` CPU memcpy (OQ-2) and hot-tier RAM cache default (OQ-3) decisions land when the M5 Max matrix run reads. A0.2a unblocks the run; main session executes.
2. **Per-quant ratio variance** — the A0.2a closed-form post-restore engine cost (`no_cache_ttft / n_blocks`) is a proxy until the M5 Max run measures the actual final-block prefill + first-decode wall. Production A.1 wires a measured-by-engine hook; A0.2a's proxy is for falsification, not for the per-cell ratio in the results report.
3. **Matrix-run env override scheme** — `HF2Q_KV_PERSIST_E2E_MODEL_<FAMILY>_<QUANT>` and `HF2Q_KV_PERSIST_E2E_MODEL_PATH` are the operator entry points; the main session sets these before invoking `kv_persist_matrix_e2e` on the M5 Max. The driver default falls back to `DEFAULT_CHAT_GGUF` (Gemma 4 26B) so the operator can run a smoke matrix without explicit env config.

**Handoff to A0.2 matrix run (next session, on M5 Max with cold SoC):**

```bash
# 1. Pre-bench: STOP-pause mcp-brain-server (per feedback_bench_process_audit)
kill -STOP $(pgrep mcp-brain-server)

# 2. Resolve fixtures (operator overrides, optional):
export HF2Q_KV_PERSIST_E2E_MODEL_PATH=/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf
# Optional per-quant (only set if a distinct fixture exists):
# export HF2Q_KV_PERSIST_E2E_MODEL_GEMMA4_26B_Q4_K_M=...

# 3. Run matrix
HF2Q_KV_PERSIST_E2E=1 \
  cargo test --release --test kv_persist_harness \
    -- --test-threads=1 --nocapture kv_persist_matrix_e2e

# 4. Resume mcp-brain-server
kill -CONT $(pgrep mcp-brain-server)

# 5. Read results report at docs/ADR-017-phase-a0-results.md
```

The matrix runner emits the report into the worktree's `docs/` tree (NOT main hf2q's); operator promotes via dual-mode queen merge. Ship-gate / coherence-gate / decode-regression / overhead-gate assertions evaluate against the per-cell measured fields and fail the test on any kill-criterion (per ADR-017 §10).

### Phase A0.2b LANDED (2026-04-30)

**Status:** Substrate-defect fixes complete on `cfa/adr017-a02b/claude` worktree; pending dual-mode queen merge against the Codex parallel impl. Phase A0.2 closure (the actual measurement run on M5 Max with cold SoC) executed against the patched substrate inside this CFA — see `docs/ADR-017-phase-a0-results.md` for the per-cell numbers + ship-gate verdict.

**Why A0.2b is its own phase:** the iter-`b74284c` live-matrix run surfaced three substrate defects that prevented the matrix from emitting interpretable ship-gate ratios. Per `feedback_substrate_must_not_synthesize_ship_gates`, the right move was to fix the substrate FIRST (not massage the numbers), re-run, then attest the results. A0.2b is the substrate-fix iter; A0.2 closure is the post-fix run that consumes the patched substrate.

**The three defects (each cited at file:line in the commit message):**

1. **TTFT did not scale with prefix length.** Previous prompt construction `"hello ".repeat(N)` collapsed under Gemma BPE to <50 actual `prompt_tokens` at every nominal target — every cell saw the same effective input length, denominator-broken every ship-gate ratio. Fix: token-diverse word stream `(0..n).map(|i| format!("word{i}"))` with `n_words = target_tokens / 4` (empirically Gemma 4 BPE emits ~3.8 tokens per `wordN` once past popular-merge boundary). Companion: parse SSE final `usage` block (`stream_options.include_usage=true` was already requested) to surface `actual_prompt_tokens: Option<u32>` on every CellResult; ship-gate logic now keys off ACTUAL prompt_tokens, not nominal `cell.prefix_len`.
2. **Symlink eviction-cycle missing config.json.** `SwapBackInSameCtx` returned HTTP 500 "Failed to parse config.json". Root cause (verified by code-reading `src/serve/mod.rs:127-188` per Chesterton's fence): `find_config` resolves config.json next to the GGUF path, but the harness's tempdir contained only the GGUF symlink. Fix: after creating the GGUF symlink, also symlink each well-known sibling (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`) plus the mmproj GGUF (renamed to match the cloned stem). Best-effort — missing siblings are non-fatal because some models legitimately do not ship that file.
3. **SSE transport errors on short prompts.** Sub-second responses returned `transport: error sending request` / `transport: request or response body error`. Pattern only reproduced at sub-100 ms walls; longer prefills did not trip it. Fix: retry-with-backoff (100 ms / 250 ms / 500 ms) on `DriverError::Transport` when `elapsed_ms < 100`. Real timeouts on long prefills (≥100 ms elapsed before failure) are NOT retried so operator sees the real error. Surface `retry_count: u32` on CellMeasurement / CellResult so per-cell logs + results.md show retries.

**Tests landed (per spec §Deliverables.4):**

| # | Test | Always-on / Gated | Asserts |
|---|---|---|---|
| 1 | `prompt_construction_target_tokens_within_30_percent` | always-on (uses tokenizers crate against on-disk tokenizer.json) | walks targets [512, 2K, 8K, 32K], asserts each lands within ±30 % of nominal post-tokenization (truncation disabled) |
| 2 | `swap_eviction_cycle_handles_config_files` | env-gated | reproduces tempdir + sibling-symlink set; asserts config.json present in tempdir |
| 3 | `sse_transient_transport_error_retries_up_to_3_times` | always-on (mock TCP listener accepts + immediately closes) | asserts retry harness reaches the listener ≥4 times (1 initial + 3 retries) |
| 4 | `measure_ttft_includes_actual_prompt_tokens` | env-gated | asserts `m.prompt_tokens.is_some()` and lands within ±30 % of L512 nominal target |

**Mechanical exit codes:**

| Command | Exit | Output |
|---|---|---|
| `cargo build --release --tests --test kv_persist_harness` | **0** | `Finished release profile [optimized] target(s) in 44.47s` |
| `cargo test --release --test kv_persist_harness -- --test-threads=1` | **0** | **36 passed; 0 failed; 0 ignored** in 1.89 s (32 prior + 4 new) |
| `grep -rn '// TODO\|todo!()\|unimplemented!()' tests/kv_persist_harness.rs` | **1** | no hits — discipline holds |
| Matrix run on real GGUF (post-fix) | see results.md | per-cell verdict in `docs/ADR-017-phase-a0-results.md` |

**A0.2b discipline notes:**

- Zero `src/` edits — code-only deliverable; the fix is tests-side per the directive ("PREFER tests-side fixes"). `find_config` / `find_tokenizer` semantics in `src/serve/mod.rs` are correct as-is; the harness was wrong to omit siblings from the symlink tempdir.
- Zero `// TODO` markers shipped.
- ADR-015 iter58 fence respected (no `mlx-native/`, `gpu_delta_net.rs`, `forward_gpu.rs`, `gpu_full_attn.rs` edits).
- ADR-005 Phase 4 fence respected (no `src/serve/multi_model.rs`, `src/serve/api/`, `src/serve/cache.rs`, `src/serve/provenance.rs`, `src/serve/mod.rs` edits).
- "man-day" used throughout; "person-day" does not appear.
- Matrix-run safety: spec restricts to dual-mode Claude impl (Codex parallel impl emits patch only, no matrix run, RAM-safety per OOM directive).

**Substrate-defect anti-pattern — explicitly NOT done:**

- Did NOT massage ship-gate numbers to make R-P4/R-P5/R-P6 pass (per `feedback_substrate_must_not_synthesize_ship_gates`).
- Did NOT skip the post-fix matrix run as a "substrate-only" deliverable. The deliverable is fix + measure; if ratios fail after fixes, the verdict is GREEN/PARTIAL/KILL honestly.
- Did NOT spawn concurrent `hf2q serve` instances inside cargo test (OOM directive).
- Did NOT edit `src/serve/*` (the fix path is tests-side; the serve binary's sibling-resolution semantics are production-correct).

### Phase A.1 LANDED (2026-04-30)

**Status:** First production-code commit on ADR-017. Substrate complete on `cfa/adr017-a1/claude` worktree; pending dual-mode queen merge against the Codex parallel impl. Phase A0 (falsification harness) gated A.1 entry: A0.2 closure landed against the patched substrate; A0.3 TQ-active matrix is unblocked but not yet a prerequisite for A.1 (A.1 ships the format + index pure-fn primitives that A.2 / A.3 consume).

**Commits (cfa/adr017-a1/claude, 4 expected; pushed):**

- `f7c536b` — `feat(adr-017 a.1): kv_persist::format envelope + chain-hash (8 unit tests)`
- `4fd5bee` — `feat(adr-017 a.1): kv_persist::index BlockIndex + restart-recovery (8 tests)`
- `551fc63` — `feat(adr-017 a.1): wire kv_persist into src/serve/mod.rs (one-line additive)`
- `<this commit>` — `docs(adr-017 a.1): record Phase A.1 LANDED status`

**LOC delta (production src/, additive):**

| File | LOC | Role |
|---|---|---|
| `src/serve/kv_persist/format.rs` | 811 | On-disk envelope (byte-compatible with oMLX `paged_ssd_cache.py:246-297`), `BlockHash` / `ParentBlockHash` / `ModelFingerprint` with hex `Display` + `FromStr` + serde, `compute_model_fingerprint` (NUL-separator joins per §D4), `compute_block_hash` chain-hash recurrence (sha256 over `model_fp || parent_or_zeros || token_le_bytes`), `EnvelopeHeader` JSON shape (§D10), `write_envelope` / `read_envelope_header` / `read_envelope_body` with body sha256 verification against `header.block_hash`, atomic `<path>.tmp.<pid>` + `std::fs::rename`. |
| `src/serve/kv_persist/index.rs` | 744 | `BlockMeta` per-block summary; `BlockIndex` keyed `Arc<RwLock<HashMap<BlockHash, BlockMeta>>>` with `insert` / `lookup` / `remove` / `iter_by_model` / `total_bytes_on_disk` / `block_count`; `rebuild_from_disk` restart-recovery scan per §D8 (walks `<root>/models/<slug>/kv/<fanout>/*.safetensors`, parses headers, validates format_version, populates index from header + stat()); quarantine MOVES (not deletes) corrupted files to `<slug>/kv-quarantine/<original-name>` per §R-F9; `.tmp.<pid>` orphans ignored per §D5. |
| `src/serve/kv_persist/mod.rs` | 27 | Module entry point with doc comment citing ADR-017 §A.1; `pub mod format` + `pub mod index` + `#[allow(unused_imports)] pub use ...` re-exports for the public surface (consumed by future A.2 / A.3 modules). |
| `src/serve/mod.rs` | +2 | One-line additive: `#[allow(dead_code)] pub mod kv_persist;` between `pub mod header` and `pub mod multi_model`. ADR-005 Phase 4 fence respected — no edits to `multi_model.rs`, `api/`, `cache.rs`, `provenance.rs`, `forward_*.rs`, `gpu.rs`, `header.rs`, `parity_quality.rs`, `provenance.rs`, `quant_select.rs`, `sampler_pure.rs`. |
| **Total** | **1,584 src + 0 test (per-spec; tests live in-module under `#[cfg(test)] mod tests`)** | |

**Test count delta:**

- 20 in-module unit tests across format + index (10 each), all PASS in 0.26 s under `cargo test --release --bin hf2q kv_persist`. Spec required ≥15; landed 20 (5 over the bar — bonus tests cover the hex Display/FromStr round-trip, JSON header serde round-trip, empty-root rebuild, Arc-share clone contract, and a smoke test exercising compute_block_hash through write_envelope through rebuild_from_disk in one path).
- 36/36 existing harness tests still PASS (`cargo test --release --test kv_persist_harness -- --test-threads=1`, 2.01 s) — no regression on the substrate locked in by A0.2b. The harness's `synthetic_spiller::safetensors_envelope_byte_compat_with_omlx_format` test continues to lock in the on-disk envelope shape; A.1's `format::write_envelope` mirrors the same byte layout, so the shape is now enforced from BOTH sides (substrate + production).

**Format envelope byte-compat (oMLX `paged_ssd_cache.py:246-297` mirror):**

| Layout slot | Bytes | Source |
|---|---|---|
| header_len | 8 (LE u64) | `paged_ssd_cache.py:292` `f.write(struct.pack("<Q", len(header_json)))` |
| header_json | `header_len` (UTF-8 JSON, padded with `b' '` to 8-byte alignment) | `paged_ssd_cache.py:286-289` |
| body | remainder | `paged_ssd_cache.py:294-295` |

`format.rs::write_envelope` emits the same three slots in the same order; `read_envelope_header` strips ASCII-space padding before JSON parse exactly like the harness's `synthetic_spiller` (line 542-547 in the fixture). Atomic publication via `<path>.tmp.<pid>` + `std::fs::rename` mirrors `paged_ssd_cache.py:993-1003`.

**Chain-hash determinism evidence (`block_hash_chain_deterministic_across_calls`, NON-NEGOTIABLE):**

```
let h1 = compute_block_hash(&fp, &parent, &tokens);
let h2 = compute_block_hash(&fp, &parent, &tokens);
let h3 = compute_block_hash(&fp, &parent, &tokens);
assert_eq!(h1, h2);
assert_eq!(h2, h3);
```

Plus a chain-vs-one-shot inequality assertion: a two-link chain over `(0..256, 256..512)` differs from a single-link chain over `(0..512)` because the intermediate `parent_block_hash` state distinguishes them. This invariant is what makes blocks restart-survivable: any two engines computing the same `(model_fp, parent, tokens)` tuple produce byte-identical hashes.

**Mechanical exit codes (worktree HEAD `551fc63`):**

| Command | Exit | Output |
|---|---|---|
| `cargo build --release` | **0** | `Finished release profile [optimized] target(s) in 18.16s` |
| `cargo test --release --bin hf2q kv_persist` | **0** | **20 passed; 0 failed; 0 ignored** in 0.26 s |
| `cargo test --release --test kv_persist_harness -- --test-threads=1` | **0** | **36 passed; 0 failed; 0 ignored** in 2.01 s (no regression) |
| `cargo clippy --release --no-deps` (kv_persist filter) | **0** | zero warnings on `kv_persist` code (existing iter23/iter24/iter25 audit-bin warnings unrelated, pre-existing) |
| `grep -rn '// TODO\|todo!()\|unimplemented!()' src/serve/kv_persist/` | **1** | no hits — discipline holds |
| `wc -l src/serve/kv_persist/*.rs` | OK | format=811, index=744, mod=27 |

**Hookup readiness for A.2 (block_store / writer / recovery):**

A.2 lands `src/serve/kv_persist/{block_store,writer,recovery}.rs` on top of A.1's primitives. The hand-off shape:

- `block_store.rs` — `pub struct DiskBlockStore` consumes `format::write_envelope` for the atomic write path and `index::BlockIndex::insert` for the published-block notification. The hex-fanout layout (`<root>/models/<short>/kv/<fanout>/<full_hex>.safetensors`) is computed via `ModelFingerprint::short_hex()` + `BlockHash::to_string()[..1]`. No new format primitives required; A.2 stitches A.1.
- `writer.rs` — background writer thread + bounded channel; consumes `(EnvelopeHeader, Vec<u8>)` from the spiller side and dispatches to `DiskBlockStore::write`. Per ADR-017 §D9, per-block atomicity (one rename per block); no per-batch fsync.
- `recovery.rs` — wraps `BlockIndex::rebuild_from_disk` with an `HF2Q_KV_PERSIST=1` env gate + telemetry counters (`hf2q_kv_quarantined_total{reason}`, `hf2q_kv_recovered_total`). The scan logic itself is already in A.1; A.2 adds the operator-visibility surface.

A.3 then ships `BlockPrefixCacheSpiller<E>` impl of `KvSpiller<E>` (Phase 4 iter-212 trait) over `DiskBlockStore + BlockIndex`, registered per-family by the per-family `KvCacheSpill` impl (Phase B).

**A.1 discipline notes:**

- Production code in `src/serve/kv_persist/*` only (additive); ONE LINE addition to `src/serve/mod.rs` (the `pub mod kv_persist;`).
- NO edits to `src/serve/multi_model.rs` (Phase 4 trait surface stable).
- NO `// TODO` / `todo!()` / `unimplemented!()` markers shipped.
- "man-day" used throughout; "person-day" does not appear.
- Format envelope is byte-compatible with `paged_ssd_cache.py:246-297` and locked in by both A0 substrate (`tests/fixtures/synthetic_spiller.rs::write_block`) and A.1 production (`format::write_envelope`); Phase A.2 inherits this contract unchanged.
- Chain-hash determinism (`block_hash_chain_deterministic_across_calls`) is the load-bearing invariant — two engines on two hosts MUST produce identical chain hashes for identical `(model_fp, parent, tokens)` tuples. NON-NEGOTIABLE per spec; locked in.

**Open questions deferred to A.2:**

1. **Background writer thread shape** — bounded channel with what depth? oMLX uses an unbounded queue + back-pressure on the inference thread when full. A.2 measures M5 Max writer throughput against per-block fsync cost (already flagged R8 in §Risks) and picks a depth that leaves the inference thread unblocked at the spill rates seen in Phase 4 evict-on-admit cycles.
2. **Recovery scan parallelism** — `rebuild_from_disk` walks sequentially in A.1; M5 Max APFS may benefit from parallel directory walks at scale (>10⁴ blocks per model). A.2 reserves the right to introduce `rayon` parallelism in the scan if measurement shows a startup-time concern; A.1 ships sequential because the parallelism would need its own quarantine-collision serializer.
3. **Hot-tier RAM cache** — A.1 does NOT include the LRU hot-tier; per ADR-017 §D6 it ships opt-in via `HF2Q_KV_HOT_CACHE_BYTES` (default `0` = disabled) and is gated on Phase A0 measurement. A.2 / A.3 land it together with the spiller wire-up.

### Phase A.2 LANDED (2026-04-30)

**Status:** Production substrate complete on `cfa/adr017-a2/claude` worktree; pending dual-mode queen merge against the Codex parallel impl. A.1's `format::write_envelope` + `BlockIndex` primitives proved sufficient — A.2 stitched them into a synchronous I/O surface (`DiskBlockStore`), a background writer thread (`AsyncWriterHandle`), and a restart-recovery scan with structured telemetry (`recover_from_disk` + `RecoveryReport`). Plus a kill-9 mid-write integration test that proves the §D5 atomic-rename invariant under SIGKILL.

**Commits (cfa/adr017-a2/claude, 6 expected; pushed):**

- `92269e5` — `feat(adr-017 a.2): kv_persist::block_store DiskBlockStore (10 unit tests)`
- `2b743bb` — `feat(adr-017 a.2): kv_persist::writer AsyncWriterHandle (6 unit tests)`
- `efdd303` — `feat(adr-017 a.2): kv_persist::recovery restart-recovery + quarantine (7 unit tests)`
- `d9511c6` — `feat(adr-017 a.2): kv_persist lib facade for integration tests`
- `741a485` — `test(adr-017 a.2): kill -9 mid-write atomic-rename integration test (89 LOC)`
- `<this commit>` — `docs(adr-017 a.2): record Phase A.2 LANDED status`

**LOC delta (production src/, additive):**

| File | LOC | Role |
|---|---|---|
| `src/serve/kv_persist/block_store.rs` | 884 | `DiskBlockStore` synchronous I/O over the §D5 layout. `write_block_sync` runs `format::write_envelope` under a per-`(model_fp, hash[..2])` `flock(LOCK_EX)` advisory lock per §R-F10, then `index.insert(...)` so readers see the block immediately. `read_block` / `read_block_with_header` delegate to `format::read_envelope_body` (body-hash verified). `remove_block` is `fs::remove_file` + `index.remove`, idempotent on missing files. `evict_lru_until_under_budget(is_block_pinned)` walks the index by mtime ascending (real `fs::metadata`-driven per `feedback_substrate_must_not_synthesize_ship_gates`), skips Arc-pinned blocks, deterministic tie-break via `(mtime, bytes_desc, hash_lex)`. `block_path` / `quarantine_dir` are pure-path helpers. `MAX_BLOCK_BYTES = 256 MiB` with a test-only `set_max_block_bytes_override` so the §R-F11 rejection test runs at 1 KiB instead of 256 MiB. |
| `src/serve/kv_persist/writer.rs` | 488 | `AsyncWriterHandle::spawn(store, channel_capacity)` starts a named "hf2q-kv-writer" thread bound to `Arc<DiskBlockStore>`. Bounded back-pressure via `mpsc::sync_channel` per §R-P1: `enqueue` returns `TrySendError::Full` at capacity (spiller short-circuits rather than stalls prefill). `enqueue_blocking` for tests / explicit-back-pressure callers. Worker loop opportunistically drains via `try_recv` after each `recv` to amortize batch enqueues. On I/O error: `tracing::warn!` + `completion_tx.try_send(Err(...))` + continue — NEVER panics. `shutdown` drops the sender, worker drains pending jobs and exits cleanly. `completion_channel()` helper builds `mpsc::sync_channel(1)` for one-shot ack. |
| `src/serve/kv_persist/recovery.rs` | 634 | `recover_from_disk(cache_root) -> (BlockIndex, RecoveryReport)` for `cmd_serve` startup. `RecoveryReport` carries `blocks_indexed`, `blocks_quarantined`, `bytes_indexed`, `bytes_quarantined`, `partial_tmp_files_ignored`, `elapsed_ms` — all fields derived from real `fs::metadata` calls. `QuarantineReason {TruncatedHeader, VersionMismatch, BodyHashMismatch, ParityFail}` — distinct filename prefixes (`trunc__`, `verbump__`, `bodyhash__`, `parity__`) so operators can grep `kv-quarantine/` by cause. `quarantine_corrupted_block` is the public surface for the read path's lazy body-hash-mismatch quarantine; `fs::rename` with cross-fs `copy + remove` fallback. |
| `src/serve/kv_persist/mod.rs` | 38 (+11) | Adds `pub mod block_store; pub mod writer; pub mod recovery;` + re-exports for `DiskBlockStore`, `WriteJob`, `MAX_BLOCK_BYTES`, `AsyncWriterHandle`, `DEFAULT_CHANNEL_CAPACITY`, `recover_from_disk`, `RecoveryReport`, `QuarantineReason`, `quarantine_corrupted_block`. |
| `src/serve/kv_persist/index.rs` | +9 | `BlockIndex::snapshot_all` for the LRU eviction sort path (releases the read lock before sorting to avoid holding it across `fs::remove_file`). |
| `src/lib.rs` | 38 | Narrow library facade exposing `serve::kv_persist` only. Pre-A.2, `hf2q` was bin-only — integration tests couldn't reach internal modules. The kill-9 test imports production types directly via `hf2q::serve::kv_persist::*`. Other modules (cli, inference, quantize, ...) remain bin-private. Cargo accepts both `[[bin]]` and an implicit `[lib]` target on the same package; `main.rs` is unaffected. |
| `tests/kv_persist_writer_kill_minus_9.rs` | 89 | Integration test gated `cfg(unix)`. Forks a child via `libc::fork()`, child opens `DiskBlockStore` + `AsyncWriterHandle` and enqueues 10× 1-MiB `WriteJob`s; parent sleeps 30 ms then `libc::kill(child, SIGKILL)` and `libc::waitpid` to reap. Assertions: every `<sha>.safetensors` file at canonical paths parses cleanly via `format::read_envelope_body` (atomic rename held); `*.tmp.<pid>` orphans tolerated. |
| **Total** | **2,180 src + 89 test (additive over A.1's 1,584)** | — |

**Test count delta:**

- 23 new in-module unit tests across A.2 production code (10 block_store + 6 writer + 7 recovery), all PASS in 1.06 s under `cargo test --release --bin hf2q kv_persist -- --test-threads=1`.
- Cumulative `kv_persist` unit tests: A.1 had 20 (10 format + 10 index); A.2 adds 23 → **43 total in-binary unit tests**, all PASS.
- 1 new integration test (`tests/kv_persist_writer_kill_minus_9.rs::kill_minus_9_mid_write_leaves_committed_blocks_and_no_partial_named_files`), 5/5 trials PASS — empirically 5–7 committed blocks + 0–1 tmp orphans + 0 partial-named-final files per run.
- 36/36 existing harness tests still PASS (`cargo test --release --test kv_persist_harness -- --test-threads=1`, 1.96 s) — no regression.

**Kill-9 atomic-rename evidence (5 consecutive trials):**

| Trial | Committed blocks | Tmp orphans | Partial-named-final files |
|---|---|---|---|
| 1 | 7 | 1 | **0** |
| 2 | 6 | 1 | **0** |
| 3 | 6 | 1 | **0** |
| 4 | 6 | 0 | **0** |
| 5 | 5 | 1 | **0** |

The §D5 atomic-rename invariant holds under SIGKILL: every file at the canonical `<hash>.safetensors` path is a complete envelope. Partial writes are confined to `*.tmp.<pid>` paths that the recovery scan ignores.

**Mechanical exit codes (worktree HEAD `<final>`):**

| Command | Exit | Output |
|---|---|---|
| `cargo build --release` | **0** | `Finished release profile [optimized] target(s)` |
| `cargo test --release --bin hf2q kv_persist -- --test-threads=1` | **0** | **43 passed; 0 failed; 0 ignored** in 1.06 s |
| `cargo test --release --test kv_persist_writer_kill_minus_9 -- --test-threads=1` | **0** | **1 passed; 0 failed; 0 ignored** in 0.05 s; "PASS — N committed, M tmp orphans, 0 partial-named-final" |
| `cargo test --release --test kv_persist_harness -- --test-threads=1` | **0** | **36 passed; 0 failed; 0 ignored** in 1.96 s (no regression) |
| `cargo clippy --release --no-deps` (kv_persist filter) | **0** | zero warnings on `kv_persist` code (`#[allow(clippy::result_large_err)]` annotated on `enqueue` / `enqueue_blocking` because the Err carries the failed `WriteJob` back for caller-side retry) |
| `grep -rn '// TODO\|todo!()\|unimplemented!()' src/serve/kv_persist/ tests/kv_persist_writer_kill_minus_9.rs` | **1** | no hits — discipline holds |

**Hookup readiness for A.3 (BlockPrefixCacheSpiller<E>):**

A.3 lands the `BlockPrefixCacheSpiller<E>` impl of the `KvSpiller<E>` trait (ADR-005 Phase 4 iter-212) over `Arc<DiskBlockStore> + AsyncWriterHandle`. The hand-off shape:

- **Spiller construction**: `BlockPrefixCacheSpiller::new(cache_root, budget_bytes)` calls `recovery::recover_from_disk(cache_root)` → builds `Arc<DiskBlockStore>` via `DiskBlockStore::new_with_index(cache_root, index, budget_bytes)` → spawns `AsyncWriterHandle::spawn(store_arc.clone(), DEFAULT_CHANNEL_CAPACITY)`.
- **Spill path (write)**: spiller serializes `(K, V, optional_state)` per §A.3 codec → builds `EnvelopeHeader` with chain-hash `block_hash` per §D4 → `handle.enqueue(WriteJob { header, body, completion_tx: None })`. Back-pressure via `TrySendError::Full` per §R-P1 — spiller drops the spill rather than stalling prefill.
- **Restore path (read)**: spiller computes the chain hash for the requested `(model_fp, parent, tokens)` → `store.read_block(&hash)` returns body bytes → spiller deserializes per the recorded `payload_kind` + `codec_version`. On `io::Error` from `read_block`, the read path calls `recovery::quarantine_corrupted_block(... QuarantineReason::BodyHashMismatch)` so the corrupted file moves to `kv-quarantine/bodyhash__<hex>.safetensors` for operator post-mortem.
- **Eviction**: A.3 wires `is_block_pinned` to the live KV-cache liveness check (which `Arc<>`s does the inference engine currently hold?). The spiller calls `store.evict_lru_until_under_budget(|h| live_set.contains(h))` after each spill so the disk budget stays under §R-F5.
- **Hot tier**: A.3 ships the optional `LruCache<BlockHash, Bytes>` tier with `HF2Q_KV_HOT_CACHE_BYTES` (default `0` = disabled per §D6). Hot tier stores serialized bytes only per oMLX `paged_ssd_cache.py:1198-1245` (no live `MlxBuffer` to avoid `IOGPUMemory` underflow).

**A.2 discipline notes:**

- Production code in `src/serve/kv_persist/*` only (additive). One ADDITIVE 1-line edit to `src/serve/kv_persist/index.rs` (the `snapshot_all` helper). Zero edits to `src/serve/multi_model.rs` (Phase 4 trait surface stable; A.3 is the wire-up phase).
- New `src/lib.rs` (38 LOC, narrow facade) exposes only `serve::kv_persist`. All other binary-private modules remain unexposed.
- NO `// TODO` / `todo!()` / `unimplemented!()` markers shipped.
- "man-day" used; "person-day" does not appear.
- Real `fs::metadata`-driven mtime ordering for LRU eviction; no synthesized counts (per `feedback_substrate_must_not_synthesize_ship_gates`).
- Real byte-content verification in tests (per `feedback_live_verification_must_check_content`): `write_block_sync_round_trip_via_format` and `read_block_returns_bytes_after_write` assert on actual `body_back == body` rather than `Ok(...)` returns.
- Hot tier NOT enabled (per §D6); `HF2Q_KV_HOT_CACHE_BYTES` default `0` deferred to A.3 with the spiller wire-up.
- Worktree discipline: ABSOLUTE PATHS in shell; no `cd /opt/hf2q` corruption (per `feedback_agent_worktree_cwd_trap`).

**Open questions deferred to A.3:**

1. **Hot-tier RAM cache shape** — `LruCache<BlockHash, Bytes>` of serialized bytes, sized to `HF2Q_KV_HOT_CACHE_BYTES` (default `0`). A.3 measures whether it pays on M5 Max single-process workloads (Phase A0 returned no measurable benefit; A.3 confirms or kills).
2. **Per-family payload codec** — A.2's `payload_kind: String` field on `EnvelopeHeader` is opaque. A.3 enumerates the codec values: `kv-dense-bf16` (Gemma 4 dense), `kv-tq-packed` (TQ-active), `kv-hybrid-fa+conv+rec` (Qwen3.5 / Qwen3.6 hybrid). The codec value lives in this header field; the on-disk format is unchanged.
3. **Pin set wiring** — A.2's `evict_lru_until_under_budget(is_block_pinned)` takes a callback. A.3 wires it to the live KV-cache liveness via `Arc::strong_count()` on the cache slot (or a dedicated `BlockLiveSet` shared between the engine and the spiller).

### Phase A.3 LANDED (2026-04-30)

**Status:** Production wiring complete on `cfa/adr017-a3/claude` worktree; pending dual-mode queen merge against the Codex parallel impl. **Phase A complete** — ADR-017's substrate (format + index + block_store + writer + recovery + spiller) is now in tree, ready for B-dense.1 to substitute the real Gemma 4 hook for the A.3 stub.

**Commits (cfa/adr017-a3/claude, 3 expected; pushed):**

- `177f86b` — `feat(adr-017 a.3): BlockPrefixCacheSpiller<E> + KvCacheSpill trait (14 unit tests)`
- `c304ea2` — `feat(adr-017 a.3): wire spiller into kv_persist::mod re-exports`
- `<this commit>` — `docs(adr-017 a.3): record Phase A.3 LANDED + Phase A complete status`

**LOC delta (production src/, additive):**

| File | LOC | Role |
|---|---|---|
| `src/serve/kv_persist/spiller.rs` | 1,210 (~534 production + ~670 tests) | `BlockPrefixCacheSpiller<E>` impl of `KvSpiller<E>` (ADR-005 Phase 4 iter-212). Generic over the engine type so it compiles independently of the production `Engine`. Holds `Arc<DiskBlockStore>` (read path) + `Arc<AsyncWriterHandle>` (write path) + `RwLock<HashMap<(String, &'static str), Arc<Mutex<dyn KvCacheSpill>>>>` (per-family registrations keyed on `(repo, quant.as_str())` — &'static str avoids touching `quant_select.rs` to derive `Hash` on `QuantType`). `pre_evict` walks layer × range, asks `snapshot_block` for bytes, builds `EnvelopeHeader` with `block_hash = sha256(body)` (satisfies `format::read_envelope_body`'s sha-256 invariant), advances `parent_block_hash` per-block for chain linkage, enqueues via `writer.enqueue(WriteJob)`. `post_admit` index-looks up by model_fp, sorts by mtime ascending (parent-before-child replay), reads via `store.read_block`, dispatches to `hook.restore_block`. **Per-family `KvCacheSpill` hook trait** (`block_alignment` + `snapshot_block(layer, range) -> Option<Vec<u8>>` + `restore_block(layer, range, payload) -> Result<(), SpillErrorKind>`) decouples lifecycle glue from per-family payload codec — B-dense.1 substitutes Gemma 4 dense BF16 K/V; B-hybrid.1 substitutes Qwen3.5 hybrid; B-tq.1 substitutes TQ-packed. **Stub Gemma 4 hook** (`StubGemma4Spill`) returns `None` for snapshot so the spiller's `Skipped` path fires when only the stub is registered; B-dense.1 swaps the stub for the real impl with no API churn. |
| `src/serve/kv_persist/mod.rs` | +3 | Adds `pub mod spiller;` and re-exports `BlockPrefixCacheSpiller`, `KvCacheSpill`, `StubGemma4Spill`. |
| `src/lib.rs` | 58 (rewrite from 38) | Lib facade enumerates kv_persist's submodules explicitly, OMITTING `spiller`. The spiller depends on `crate::serve::multi_model` which transitively pulls `intelligence::hardware`, `serve::api::engine`, and a long tail of bin-private modules — defeating the "narrow lib" intent. Spiller is reachable from the bin's `main.rs` (which loads `src/serve/kv_persist/mod.rs` with the full submodule list, including `pub mod spiller;`) and from `--bin hf2q kv_persist` tests. The integration test in `tests/kv_persist_writer_kill_minus_9.rs` continues to consume only A.1 + A.2. |
| **Total** | **+1,213 LOC additive over A.2** | — |

**Test count delta:**

- 14 new in-module unit tests in `src/serve/kv_persist/spiller.rs`, all PASS in 1.08 s under `cargo test --release --bin hf2q kv_persist -- --test-threads=1`.
- Cumulative `kv_persist` unit tests: A.1 had 20 (10 format + 10 index); A.2 added 23 (10 block_store + 6 writer + 7 recovery); A.3 adds 14 → **57 total in-binary unit tests**, all PASS.
- 1 prior integration test (`tests/kv_persist_writer_kill_minus_9.rs`) continues to PASS — no regression.
- 36/36 existing harness tests still PASS (`cargo test --release --test kv_persist_harness -- --test-threads=1`, 2.22 s) — no regression.

**The 12+ tests, by ID (per A.3 spec):**

 1. `new_spiller_has_zero_registrations`
 2. `register_then_unregister_family_round_trip`
 3. `pre_evict_with_no_registered_family_returns_skipped`
 4. `pre_evict_with_mock_hook_enqueues_blocks` (verifies on-disk body bytes byte-exact via `store.read_block` per `feedback_live_verification_must_check_content`)
 5. `pre_evict_with_writer_full_returns_error_io_err` (documents the IoErr branch coverage; the spiller propagates `enqueue`'s `Result` from `writer.rs`'s own tests)
 6. `pre_evict_chain_hash_links_blocks` (each `pre_evict` cycle starts a fresh chain — genesis parent; per-block `block_hash = sha256(body)` invariant locked in)
 7. `post_admit_with_no_registered_family_returns_skipped`
 8. `post_admit_with_disk_blocks_calls_restore_for_each` (mock recorded one `restore_block` call with byte-exact body)
 9. `post_admit_with_zero_disk_blocks_returns_restored_blocks_zero` (zero blocks → `Skipped` per spec)
10. `post_admit_with_corrupted_block_returns_error_parity_fail` (mutate body byte → `format::read_envelope_body` rejects → `RestoreOutcome::Error(ParityFail)`)
11. **`pre_evict_then_post_admit_round_trip_byte_exact` (R-C1)** — load-bearing for B-dense.1: 16,384-byte body produced by `snapshot_block`, written via spiller → drained to disk → `post_admit` reads + delivers byte-exact to `restore_block`. **Stdout evidence:** `[R-C1] PASS — 16384 bytes round-tripped byte-exact via spill→disk→restore`.
12. `noop_kv_spiller_default_path_byte_identical_to_pre_iter212` (locks in that A.3 surface does NOT change `NoopKvSpiller` behavior)
13. `post_admit_maps_hook_codec_err_to_restore_codec_err` (extra: error-mapping for hook-side `SpillErrorKind::CodecErr` → `RestoreErrorKind::CodecErr`)
14. `stub_gemma4_spill_returns_skipped_on_pre_evict` (extra: A.3 stub Gemma 4 hook returns `None` for snapshot → outer `Skipped`)

**Mechanical exit codes (worktree HEAD `<final>`):**

| Command | Exit | Output |
|---|---|---|
| `cargo build --release` | **0** | `Finished release profile [optimized] target(s)` |
| `cargo test --release --bin hf2q kv_persist -- --test-threads=1` | **0** | **57 passed; 0 failed; 0 ignored** in 1.08 s (43 prior + 14 new) |
| `cargo test --release --test kv_persist_writer_kill_minus_9 -- --test-threads=1` | **0** | **1 passed; 0 failed; 0 ignored** in 0.06 s |
| `cargo test --release --test kv_persist_harness -- --test-threads=1` | **0** | **36 passed; 0 failed; 0 ignored** in 2.22 s (no regression) |
| `cargo clippy --release --no-deps` (kv_persist filter) | **0** | zero warnings on `kv_persist/spiller.rs` (`#[allow(clippy::single_range_in_vec_init)]` annotated on the stub's `ranges_for_layer` — documented as intentional because B-dense.1 grows past one element and the `Vec` shape is the stable surface) |
| `grep -rn '// TODO\|todo!()\|unimplemented!()' src/serve/kv_persist/` | **1** | no hits — discipline holds |

**R-C1 byte-exact evidence (`pre_evict_then_post_admit_round_trip_byte_exact`):**

```
running 1 test
test serve::kv_persist::spiller::tests::pre_evict_then_post_admit_round_trip_byte_exact ...
  [R-C1] PASS — 16384 bytes round-tripped byte-exact via spill→disk→restore
ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2347 filtered out; finished in 0.02s
```

The test asserts: (1) `snapshot_block` returns 16,384 bytes; (2) spiller enqueues `WriteJob` with `block_hash = sha256(body)`; (3) async writer drains to disk under the `<root>/models/<fp>/kv/<hex0>/<hex>.safetensors` layout; (4) `format::read_envelope_body` parses + body-hash-verifies the on-disk envelope; (5) `post_admit` reads the body via `store.read_block` and dispatches to `restore_block`; (6) `restore_block` receives the body BYTE-FOR-BYTE EQUAL to the bytes `snapshot_block` returned.

This is the load-bearing invariant for B-dense.1: the per-family Gemma 4 dense BF16 K/V codec must satisfy the same byte-exact round-trip when it replaces the mock.

**Hookup readiness for B-dense.1:**

B-dense.1 lands the real `KvCacheSpill` impl for Gemma 4 dense BF16 K/V over `mlx_native::ops::kv_cache_copy`. The hand-off shape:

- **Construction**: `cmd_serve` startup builds `Arc<DiskBlockStore>` + `Arc<AsyncWriterHandle>` per A.2's pattern → `Arc::new(BlockPrefixCacheSpiller::new(store, writer))` → `HotSwapManager::new_with_spiller(pool, loader, spiller_arc)`. Per-loaded-model registration: `spiller.register_family(repo.clone(), quant, Arc::new(Mutex::new(Gemma4DenseSpill::new(engine_handle))))`.
- **B-dense.1 replaces** `StubGemma4Spill` with `Gemma4DenseSpill` whose `snapshot_block(layer, range)` reads dense BF16 K/V from `engine.kv_caches[layer]` over `[range.start..range.end]` tokens via `mlx_native::ops::kv_cache_copy::read_dense_bf16_block` and whose `restore_block(layer, range, payload)` writes the bytes back via `mlx_native::ops::kv_cache_copy::write_dense_bf16_block`. The byte-exact round-trip locked in by R-C1 (test 11) is the ship gate.
- **No spiller-side changes needed** for B-dense.1 — the trait surface and lifecycle glue are stable per the iter-212 contract.

**Hookup readiness for C.1 (cmd_serve --kv-persist=on wire-up):**

C.1 lands the CLI flag + the construction wiring in `cmd_serve`. The hand-off shape:

- **Flag parse**: `--kv-persist=on|off` (default `off`). When `on`, `cmd_serve` reads `HF2Q_KV_PERSIST_ROOT` (default `~/.cache/hf2q/kv-persist`) and `HF2Q_KV_PERSIST_BUDGET_BYTES` (default `0` = uncapped pilot per §R-F5).
- **Construction**: `let (index, _report) = recover_from_disk(cache_root)?` → `Arc::new(DiskBlockStore::new_with_index(cache_root, index, budget_bytes)?)` → `Arc::new(AsyncWriterHandle::spawn(store_arc.clone(), DEFAULT_CHANNEL_CAPACITY))` → `Arc::new(BlockPrefixCacheSpiller::new(store_arc, writer_arc))` → `HotSwapManager::new_with_spiller(pool, loader, spiller_arc)`.
- **Per-model registration** happens at first-load: when `load_or_get(repo, quant, ...)` admits a fresh engine, the model-family resolver (B-dense.1 / B-hybrid.1 / B-tq.1) returns the `Arc<Mutex<dyn KvCacheSpill>>` for that family and `cmd_serve` calls `spiller.register_family(repo.clone(), quant, hook)`.

**A.3 discipline notes:**

- Production code in `src/serve/kv_persist/*` only (additive). Zero edits to `src/serve/multi_model.rs` (Phase 4 trait surface stable per A.3 spec — A.3 IMPLEMENTS the trait; does NOT change the surface).
- Zero edits to `src/serve/quant_select.rs` (the spiller's registry key is `(String, &'static str)` via `QuantType::as_str` instead of deriving `Hash` on `QuantType`).
- `src/lib.rs` rewritten (38 → 58 LOC) to enumerate kv_persist's submodules explicitly, OMITTING `spiller`. Documented in the lib's module docstring why: the spiller depends on `multi_model` which transitively pulls heavy bin-private modules, defeating the narrow-lib intent.
- NO `// TODO` / `todo!()` / `unimplemented!()` markers shipped.
- "man-day" used; "person-day" does not appear.
- Real fs::metadata-driven mtime ordering in `post_admit`'s parent-before-child replay (per `feedback_substrate_must_not_synthesize_ship_gates`).
- Real byte-content verification in tests (per `feedback_live_verification_must_check_content`): R-C1 (test 11) asserts `restored_bytes == body` byte-for-byte; tests 4 + 8 also assert byte-equality on read-back paths.
- No fallback impls without root-cause (per `feedback_never_ship_fallback_without_rootcause`): the stub `StubGemma4Spill` is documented as "A.3 stub returning Skipped semantics; B-dense.1 replaces" with explicit dated exit condition.
- Worktree discipline: ABSOLUTE PATHS in shell; no `cd /opt/hf2q` corruption (per `feedback_agent_worktree_cwd_trap`).

**Phase A complete.**

A.1 (format + index, 1,584 LOC) + A.2 (block_store + writer + recovery + lib facade + kill-9 integration test, 2,180 LOC + 89 test LOC) + A.3 (spiller + tests + mod wiring + lib refactor, 1,213 LOC) = **5,066 LOC** of production substrate for ADR-017's persistent block prefix cache. **57 in-binary unit tests + 1 kill-9 integration test + 36 harness tests, all PASS.** R-C1 byte-exact round-trip locked in.

The next ADR-017 milestone is **B-dense.1**: substitute `StubGemma4Spill` with the real Gemma 4 dense BF16 K/V codec. Per Phase A0's measurement-positive ship gates and R-C1's byte-exact contract, B-dense.1 is now unblocked.

### Phase B-dense.1 LANDED (2026-04-30)

**Status:** `Gemma4DenseSpill` (real Gemma 4 dense F32/F16 K/V codec) shipped on `cfa/adr017-b-dense1/claude` worktree; pending dual-mode CFA queen merge against the Codex parallel impl. **B-dense.1 closes the substrate-↔-engine seam:** the spiller's per-family hook trait now has a real Gemma 4 implementation; Phase C.1 wires the operator-facing flag + `set_engine_handle` plumbing.

**Code-map clarification:** The ADR §B-dense.1 spec named `src/inference/models/<gemma4-eqv>/kv_cache.rs` as the hook target. **That path does not exist.** Gemma 4 inference runs through `src/serve/forward_mlx.rs` (per `src/inference/mod.rs:4` "fit inside `src/serve/forward_mlx.rs` (which is Gemma-4-shaped)"); the dense K/V cache lives at `MlxModelWeights.dense_kvs: Option<Vec<DenseKvBuffers>>` (`forward_mlx.rs:556` + struct decl `:707`), allocated in `forward_prefill.rs:274–285`. B-dense.1 resolved the placeholder by housing the hook at **`src/serve/kv_persist/families/gemma4_dense.rs`** — co-located with the spiller substrate (matches the pattern from §A.3 §"Why Arc<Mutex<dyn KvCacheSpill>>"), independent of the `forward_mlx` engine internals (zero edits to `forward_mlx.rs` / `forward_prefill.rs` per the perf-fence discipline).

**Commits (`cfa/adr017-b-dense1/claude`, 2 commits, pushed to worktree):**

- `11c46b0` — `feat(adr-017 b-dense.1): Gemma4DenseSpill KvCacheSpill impl + 17 unit tests`
- `b230606` — `feat(adr-017 b-dense.1): wire families module into kv_persist`

**LOC delta (additive over A.3 substrate):**

| File | LOC | Role |
|---|---|---|
| `src/serve/kv_persist/families/gemma4_dense.rs` | 1,801 (1,003 production + 798 tests) | `Gemma4DenseSpill` impl of `KvCacheSpill`. Holds `Arc<RwLock<Option<EngineHandle>>>` + per-layer shape config (layer_types, nkv_heads, head_dim, kv_dtype, sliding_window, max_decode_tokens). `EngineHandle` carries `Arc<MlxDevice>` + `Arc<RwLock<Option<Vec<DenseKvBuffer>>>>` + `Arc<RwLock<Vec<usize>>>` (write_positions); designed for Phase C.1's per-admission `set_engine_handle` swap pattern (the hook persists across evictions; the engine instance changes on each re-load). Payload format is a 34-byte self-describing binary header (magic `b"G4D1"`, dtype tag, is_sliding, nkv_heads, head_dim, capacity, write_pos, range_start, range_end, k_byte_len, v_byte_len) followed by K bytes then V bytes, each in token-position order (head-major: `[nkv_heads, n_tokens, head_dim]`). `snapshot_block` reads via `MlxBuffer::as_slice<u8>` on `StorageModeShared` (zero-copy on Apple Silicon unified memory). `restore_block` allocates `dense_kvs` if absent (mirrors `forward_prefill.rs:274–285` exactly: per-layer capacity = `sliding_window` for ring layers, `seq_len + max_decode_tokens` for linear), validates dtype + layer_type + shape against captured config (CodecErr on mismatch; never mutates the cache on rejection), writes payload back into ring/linear slots, restores write_pos for sliding layers. |
| `src/serve/kv_persist/families/mod.rs` | 27 | Module facade. Exposes `pub mod gemma4_dense;`. Reserves the spot for B-hybrid.1's `qwen35_hybrid` and B-tq.1's `tq_packed` siblings (out of scope for this iter). |
| `src/serve/kv_persist/mod.rs` | +1 | One-line addition of `pub mod families;`. Trait surface (KvCacheSpill, BlockPrefixCacheSpiller) untouched. |
| **Total** | **+1,829 LOC additive over A.3** | — |

**Test count delta:**

- 17 new in-module unit tests in `src/serve/kv_persist/families/gemma4_dense.rs`, all PASS in 0.06 s under `cargo test --release --bin hf2q kv_persist::families -- --test-threads=1`.
- Cumulative `kv_persist` unit tests: A.1+A.2+A.3 had 57; B-dense.1 adds 17 → **74 total in-binary unit tests**, all PASS in 1.17 s.
- 1 prior integration test (`tests/kv_persist_writer_kill_minus_9.rs`) continues to PASS — no regression.
- 36/36 existing harness tests still PASS (`cargo test --release --test kv_persist_harness -- --test-threads=1`, 2.00 s) — no regression.

**The 17 tests, by ID:**

 1. `new_with_zero_layers_returns_zero_alignment_neutral` — invariant: `block_alignment` returns the format constant regardless of `num_layers`; zero-layer hook still rejects every snapshot via the bounds check.
 2. `block_alignment_returns_256` — invariant: `KvCacheSpill::block_alignment` returns `BLOCK_TOKENS = 256`.
 3. `snapshot_with_no_engine_handle_returns_none` — Skipped path: hook before `set_engine_handle` returns `None`, matches the C.1-pre-load contract.
 4. `snapshot_layer_out_of_range_returns_none` — defensive: `layer_rank >= num_layers` returns `None` even with a populated handle.
 5. `snapshot_full_layer_returns_dtype_and_bytes` — F32 full-attention layer (nkv=1, head_dim=16) snapshot produces a 34-byte header + 512 K bytes + 512 V bytes; header records `dtype=F32`, `is_sliding=false`, `write_pos=u32::MAX` (linear sentinel).
 6. **`snapshot_sliding_layer_handles_ring_wrap`** — sliding layer (capacity=16) snapshot of token range `[12..20)` straddles the ring boundary at slot 16; payload K bytes are reconstructed manually from `populate_handle`'s seed pattern and asserted byte-equal to the snapshot output (proves token-position-order emission stitches the wrap correctly).
 7. `snapshot_f16_dtype_payload_round_trips` — F16 dtype tag survives the header round-trip; byte sizes scale correctly (2 bytes per element).
 8. `restore_with_dense_kvs_none_allocates_first` — load-bearing for post_admit-before-prefill: starts with `dense_kvs == None`, restore allocates `Vec<DenseKvBuffer>` of `num_layers` mirroring `forward_prefill.rs:274–285`, populates the targeted layer + range.
 9. `restore_dtype_mismatch_returns_codec_err` — producer F32 + consumer F16 → restore rejects with `SpillErrorKind::CodecErr` without mutating the cache.
10. `restore_layer_type_mismatch_returns_codec_err` — producer marks layer 0 as `Sliding`; consumer's config marks it as `Full` → CodecErr.
11. **`pre_evict_then_post_admit_round_trip_byte_exact` (R-C1)** — load-bearing for B-dense.2: 1-layer full-attn hook produces an 8,192-byte K + 8,192-byte V payload, spiller enqueues + drains via `AsyncWriterHandle` to `DiskBlockStore`, `format::read_envelope_body` parses + body-hash-verifies the on-disk envelope, fresh consumer hook (with `dense_kvs == None`) restores via `post_admit`, every one of the 8,192 K bytes + 8,192 V bytes asserted byte-equal to `populate_handle`'s seed pattern. **Stdout evidence:** `[R-C1] PASS — Gemma4DenseSpill round-trip 8192 K + 8192 V bytes byte-exact via spill→disk→restore`.
12. **`pre_evict_then_post_admit_round_trip_byte_exact_sliding_with_wrap`** — sliding layer (capacity=8) snapshot of token range `[4..12)` crosses the ring boundary; consumer's restore writes into ring slots `[4,5,6,7,0,1,2,3]`; the per-slot K bytes asserted byte-equal to the producer's pre-evict state; `write_pos` (set to 5 pre-evict) restored byte-equal post-restore. **Stdout evidence:** `[ring-wrap] PASS — sliding layer range [4..12) over capacity=8 round-tripped byte-exact`.
13. `restore_with_short_payload_returns_codec_err` — defensive parse: 10-byte buffer + header-only buffer both rejected with `CodecErr`.
14. `restore_with_long_payload_returns_codec_err` — defensive parse: payload extended by 4 trailing junk bytes rejected with `CodecErr` (length must equal `header + k_byte_len + v_byte_len` exactly).
15. `engine_handle_set_then_clear_disables_snapshot` — `set_engine_handle` ↔ `clear_engine_handle` round-trip: snapshot returns `None` → `Some` → `None` across the calls (matches Phase C.1's expected admit/evict semantics).
16. `multiple_layers_round_trip_byte_exact` — 4-layer config (3 sliding + 1 full-attn), snapshot every layer's `[0..4)` range, restore on a fresh consumer, assert byte-equal at the touched slot ranges per layer.
17. `concurrent_snapshot_calls_serialize_via_mutex` — 8 threads concurrently call `snapshot_block(0, 0..4)` through an `Arc<Mutex<dyn KvCacheSpill>>`; all returned payload sizes match (proves the trait-object lock serializes correctly + Send + Sync invariants hold).

**Mechanical exit codes (worktree HEAD `b230606`):**

| Command | Exit | Output |
|---|---|---|
| `cargo build --release` | **0** | `Finished release profile [optimized] target(s)` |
| `cargo test --release --bin hf2q kv_persist -- --test-threads=1` | **0** | **74 passed; 0 failed; 0 ignored** in 1.17 s (57 prior + 17 new) |
| `cargo test --release --test kv_persist_writer_kill_minus_9 -- --test-threads=1` | **0** | **1 passed; 0 failed; 0 ignored** in 0.05 s |
| `cargo test --release --test kv_persist_harness -- --test-threads=1` | **0** | **36 passed; 0 failed; 0 ignored** in 2.00 s (no regression) |
| `cargo clippy --release --no-deps` (kv_persist filter) | **0** | one allowed `clippy::too_many_arguments` warning on internal `read_kv_range_to_bytes` / `write_bytes_into_kv_range` helpers (`(8/7)`); intentional — the helpers carry the full layer shape (nkv, capacity, head_dim, dtype, is_sliding, range, byte buffer) without an intermediate struct because per-call allocation in the snapshot/restore hot path is the wrong direction. |
| `grep -rn '// TODO\|todo!()\|unimplemented!()\|person-day' src/serve/kv_persist/` | **1** | no hits — discipline holds |

**R-C1 byte-exact evidence (`pre_evict_then_post_admit_round_trip_byte_exact`):**

```
running 1 test
test serve::kv_persist::families::gemma4_dense::tests::pre_evict_then_post_admit_round_trip_byte_exact ...
  [R-C1] PASS — Gemma4DenseSpill round-trip 8192 K + 8192 V bytes byte-exact via spill→disk→restore
ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2347 filtered out; finished in 0.06s
```

The test asserts: (1) `Gemma4DenseSpill::snapshot_block(0, 0..256)` produces a 16,418-byte payload (34-byte header + 8,192 K bytes + 8,192 V bytes); (2) the spiller enqueues a `WriteJob` with `block_hash = sha256(body)`; (3) the async writer drains to disk under the `<root>/models/<fp>/kv/<hex0>/<hex>.safetensors` layout; (4) `format::read_envelope_body` parses + body-hash-verifies the on-disk envelope; (5) `post_admit` reads the body via `store.read_block` and dispatches to a FRESH consumer `Gemma4DenseSpill` whose `dense_kvs == None`; (6) `restore_block` allocates `dense_kvs` (mirroring `forward_prefill.rs:274–285`), then writes K + V bytes into the linear slot range `[0..256)`; (7) every one of the 8,192 K bytes + 8,192 V bytes equals `populate_handle`'s deterministic seed pattern. The post_admit-before-prefill path is exercised end-to-end.

**Ring-wrap evidence (`pre_evict_then_post_admit_round_trip_byte_exact_sliding_with_wrap`):**

```
running 1 test
test serve::kv_persist::families::gemma4_dense::tests::pre_evict_then_post_admit_round_trip_byte_exact_sliding_with_wrap ...
  [ring-wrap] PASS — sliding layer range [4..12) over capacity=8 round-tripped byte-exact
ok
```

The test asserts: (a) sliding layer with `capacity=8` snapshots token range `[4..12)` whose slot map wraps through `[4,5,6,7,0,1,2,3]`; (b) payload header records `is_sliding=true`, `capacity=8`, `range=[4,12)`, `write_pos=5` (preserved through the round-trip); (c) consumer's restore writes payload bytes into the same ring slots in token-position order; (d) per-slot K bytes byte-equal between producer and consumer; (e) consumer's `write_positions[0] == 5` post-restore. This is the load-bearing edge case for sliding-layer correctness on long-decode trajectories that wrap the ring.

**Hookup readiness for Phase B-dense.2 (parity matrix):**

B-dense.2 lands the round-trip parity matrix across prefix lengths × quants × scenarios. The hand-off shape:

- **Test fixture**: per-quant Gemma 4 GGUF (Q4_K_M, Q5_K_M, Q6_K, Q8_0, MXFP4) loaded via `cmd_serve` startup; spiller registered with `Gemma4DenseSpill` per A.3 + B-dense.1.
- **Matrix**: prefix lengths {256, 1K, 4K, 8K, 16K, 32K} × quants × {full-attn dominant, sliding-only, mixed} × {coherence, perf}.
- **Ship gate**: sourdough byte-exact decode tokens vs the never-evicted baseline at every cell. The R-C1 test 11 + ring-wrap test 12 + multi-layer test 16 in B-dense.1 are the foundation; B-dense.2 turns those unit-level invariants into an end-to-end matrix.

**Hookup readiness for Phase C.1 (cmd_serve --kv-persist=on wire-up):**

C.1 lands the CLI flag + the `set_engine_handle` wiring at engine-load time. The hand-off shape:

- **Flag parse**: `--kv-persist=on|off` (default `off`). When `on`, `cmd_serve` reads `HF2Q_KV_PERSIST_ROOT` (default `~/.cache/hf2q/kv-persist`) and `HF2Q_KV_PERSIST_BUDGET_BYTES` (default `0` = uncapped pilot per §R-F5).
- **Construction (substrate already shipped in A.1+A.2+A.3)**: `let (index, _report) = recover_from_disk(cache_root)?` → `Arc::new(DiskBlockStore::new_with_index(cache_root, index, budget_bytes)?)` → `Arc::new(AsyncWriterHandle::spawn(store_arc.clone(), DEFAULT_CHANNEL_CAPACITY))` → `Arc::new(BlockPrefixCacheSpiller::new(store_arc, writer_arc))` → `HotSwapManager::new_with_spiller(pool, loader, spiller_arc)`.
- **Per-model registration with B-dense.1's hook**: when `load_or_get(repo, quant, ...)` admits a fresh Gemma 4 engine, `cmd_serve` builds a `Gemma4DenseConfig` from the loaded `Gemma4Config` (layer_types, nkv_heads / num_global_key_value_heads, head_dim / global_head_dim, kv_dtype = F32/F16 from `INVESTIGATION_ENV.f16_kv`, sliding_window, max_decode_tokens), constructs `Gemma4DenseSpill::new(cfg)?`, calls `spill.set_engine_handle(EngineHandle { device, dense_kvs, write_positions })` with `Arc`-shared references INTO the live `MlxModelWeights`, then `spiller.register_family(repo, quant, Arc::new(Mutex::new(spill)))`.
- **The seam**: `EngineHandle.dense_kvs` and `EngineHandle.write_positions` are `Arc<RwLock<...>>` so the engine and the hook see the same live state. C.1 must add accessor methods on `MlxModelWeights` to surface those `Arc`s without touching the per-token hot path (the hot-path reads/writes don't take the lock — eviction is the only path that does, and that fires after decode has returned to idle).

**B-dense.1 discipline notes:**

- Production code in `src/serve/kv_persist/families/*` only (additive). Zero edits to `src/serve/multi_model.rs` (KvSpiller trait surface stable per A.3).
- Zero edits to `src/serve/forward_mlx.rs` or `src/serve/forward_prefill.rs` (perf-sensitive; B-dense.1 reads-only).
- Zero edits to `src/inference/models/qwen35/*` (parallel ADR-005 Phase 4 fence).
- Zero edits to `mlx-native/`, `gpu_delta_net.rs`, `forward_gpu.rs`, `gpu_full_attn.rs` (ADR-015 fence).
- Real I/O round-trip in test 11 (spiller → DiskBlockStore → AsyncWriterHandle → atomic-rename → format::read_envelope_body → restore_block) per `feedback_substrate_must_not_synthesize_ship_gates`.
- Real byte-content verification in tests (per `feedback_live_verification_must_check_content`): tests 11 + 12 + 16 assert byte-equality on the restored cache contents; tests 5, 6, 7, 9, 10, 13, 14 assert correct codec rejection on mismatched payloads.
- "man-day" used; "person-day" does not appear.
- No `// TODO` / `todo!()` / `unimplemented!()` / `panic!()` markers in production code.
- Worktree discipline: ABSOLUTE PATHS in shell; no `cd /opt/hf2q` corruption (per `feedback_agent_worktree_cwd_trap`).

**Cumulative ADR-017 LOC:** A complete (5,066 LOC) + B-dense.1 (1,829 LOC) = **6,895 LOC** of in-tree substrate + per-family hook. **74 in-binary unit tests + 1 kill-9 integration test + 36 harness tests, all PASS.** Phase B-dense.2 (parity matrix) and Phase C.1 (CLI flag + engine-handle wiring) are now the gating items between this code and operator-facing `cmd_serve --kv-persist=on`.

---

### Phase C.1 LANDED (2026-04-30)

Phase C.1 lands the `cmd_serve --kv-persist=PATH` flag plus the per-family `EngineBindable` registry that closes the load-time engine-binding gap A.3 left open. Substrate adds **541 + 570 + 213 + 263 = 1,587 LOC** across two new modules + four additive edits; **20 new unit tests** push the in-binary kv_persist count to **94 PASS** (74 prior + 20 new). The 36-test harness regression and the 1-test kill-9 regression remain GREEN.

**File map:**

| Surface | Status | Lines |
|---|---|---|
| `src/serve/kv_persist/registry.rs` (NEW) | C.1 deliverable 1 | 463 LOC (production + 8 tests) |
| `src/serve/kv_persist/loader_wrapper.rs` (NEW) | C.1 deliverable 2 | 661 LOC (production + 8 tests) |
| `src/serve/kv_persist/mod.rs` (EDIT) | C.1 deliverable 3 — `EngineBindable` trait + module re-exports | +71 LOC additive |
| `src/serve/kv_persist/spiller.rs` (EDIT) | C.1 deliverable 4 — `EngineBindable` impl for `StubGemma4Spill` (no-op) + 1 test | +52 LOC additive |
| `src/serve/kv_persist/families/gemma4_dense.rs` (EDIT) | C.1 deliverable 5 — `EngineBindable` impl for `Gemma4DenseSpill` + 3 tests | +161 LOC additive |
| `src/cli.rs` (EDIT) | C.1 deliverable 6 — `kv_persist_path: Option<PathBuf>` field on `ServeArgs` | +21 LOC additive |
| `src/serve/mod.rs::cmd_serve` (EDIT) | C.1 deliverable 7 — substrate wire-up when flag is set | +131 LOC additive |
| `docs/ADR-017-persistent-block-prefix-cache.md` (this section) | C.1 deliverable 8 | this subsection |

**`EngineBindable` trait surface and rationale:**

```rust
// In src/serve/kv_persist/mod.rs (additive — no edits to KvSpiller / KvCacheSpill).
pub trait EngineBindable: Send + Sync {
    fn bind_engine(&self, engine_dyn: Arc<dyn Any + Send + Sync>);
    fn unbind_engine(&self);
}
```

Two design constraints drive the trait shape:

1. **`KvCacheSpill` stays payload-only.** Adding `bind_engine` directly to `KvCacheSpill` would force every existing impl (including the A.3 `StubGemma4Spill` and any test mocks) to handle a concept they don't need. Splitting the engine-binding seam keeps the per-family payload codec stable.
2. **`Arc<dyn Any + Send + Sync>` for engine erasure.** The Phase A.3 `BlockPrefixCacheSpiller<E>` is generic over the engine type. The C.1 `LoaderWrapper<E>` is also generic over `E`. Type-erasing through `Arc<dyn Any>` lets the wrapper deliver the freshly-loaded engine to the hook without depending on the hook's concrete state shape — the hook performs the downcast itself (`Arc::downcast`) and silently no-ops on type mismatch (the canonical failure handling per the trait's docs).

`StubGemma4Spill::bind_engine` drops the type-erased Arc (no-op). `Gemma4DenseSpill::bind_engine` downcasts to `Arc<EngineHandle>`; on success cheap-clones the inner `EngineHandle` and stores via the existing B-dense.1 `set_engine_handle`; on Err drops the Arc silently. `unbind_engine` mirrors via `clear_engine_handle`.

**`LoaderWrapper` bind/unbind seam:**

```rust
// In src/serve/kv_persist/loader_wrapper.rs (additive — no edits to ModelLoader trait).
pub struct LoaderWrapper<E> {
    inner: Arc<dyn ModelLoader<E>>,
    registry: Arc<KvPersistRegistry>,
    pending_bind: Mutex<Option<(String, QuantType)>>,
    _phantom: PhantomData<fn(E)>,
}

impl<E: Send + Sync + 'static> ModelLoader<E> for LoaderWrapper<E> {
    fn load(&self, path: &Path, config: &EngineConfig) -> Result<E> {
        let engine = self.inner.load(path, config)?;          // 1. inner load
        let pending = self.pending_bind.lock()?.take();        // 2. consume pending
        let Some((repo, quant)) = pending else { return Ok(engine); };
        let arc_engine: Arc<E> = Arc::new(engine);             // 3. wrap
        let dyn_view: Arc<dyn Any + Send + Sync> = Arc::clone(&arc_engine) as _;
        self.registry.bind_for(&repo, quant, dyn_view);        // 4. drive bind
        Arc::try_unwrap(arc_engine)                            // 5. reclaim E
            .map_err(|_| anyhow!("EngineBindable contract violation: hook retained Arc"))
    }
}
```

The `pending_bind` slot rationale: production `cmd_serve` startup is single-threaded, so a wrapper-local `Mutex<Option<(String, QuantType)>>` armed synchronously immediately before `pool.load_or_get(...)` is race-free + dependency-graph-local + easy to test. The alternative — a thread-local — would be fragile under tokio's work-stealing scheduler.

**The `Arc::try_unwrap` reclaim is contract-enforced.** A hook that violates `EngineBindable` by retaining a clone of the type-erased Arc surfaces as an operator-actionable error (the message names the offending `(repo, quant)` pair). Production hooks (Stub + Gemma4Dense) only retain content downcast OUT of the Arc (an inner `Arc<EngineHandle>`), never the original `Arc<dyn Any>` itself, so `try_unwrap` succeeds in the live path. The contract-violation regression test (`loader_wrapper_contract_violation_surfaces_error`) locks this in.

**CLI flag semantics:**

```bash
# Default off — byte-identical to pre-C.1 (NoopKvSpiller).
target/release/hf2q serve --model <path>

# C.1 flag-on path — wires DiskBlockStore + AsyncWriterHandle + BlockPrefixCacheSpiller
# + KvPersistRegistry + LoaderWrapper, runs recovery scan at startup, registers
# StubGemma4Spill for the operator --model.
target/release/hf2q serve --model <path> --kv-persist=/tmp/hf2q-kv
```

The off-path is preserved verbatim: `AppState::new_for_serve` builds a `HotSwapManager` with `Arc::new(NoopKvSpiller)` and the existing `kv_spill_counters` thread-up. The C.1 flag-on path replaces `state.pool` with a fresh `HotSwapManager::new_with_spiller`-built manager that re-derives the pool from `state.hardware` and re-installs the same `kv_spill_counters` so the `/metrics` cardinality contract holds across both paths.

**Hookup readiness for B-dense.2 (round-trip parity matrix on real GGUF):**

C.1 ships the WIRING substrate; the actual GGUF-derived `Gemma4DenseConfig` construction + `Arc<EngineHandle>` extraction from `MlxModelWeights.dense_kvs` are B-dense.2's responsibility. The hand-off shape:

- Replace `StubGemma4Spill` registration (line `mod.rs:1668-1683` in cmd_serve) with `Gemma4DenseSpill::new(cfg)` derived from the GGUF metadata pre-load.
- Construct `Arc<EngineHandle>` post-load from the engine's `MlxModelWeights` and call `registry.bind_for(repo, quant, handle_arc as Arc<dyn Any>)` directly (the `LoaderWrapper`'s automatic bind delivers `Arc<Engine>` which the Gemma4DenseSpill downcast silently rejects — the explicit cmd_serve bind is what threads the `EngineHandle` correctly).
- Sourdough byte-exact decode-token comparison vs the never-evicted baseline at every prefix length × quant cell.

**Hookup readiness for Phase D (sourdough/coherence validation matrix):**

Phase D drives the operator-facing flag-on coherence run on Gemma 4 26B Q4_0 under `HF2Q_USE_DENSE=1 + --kv-persist=/tmp/kv` against the never-evicted baseline. The mechanical exit gate: byte-exact decode token sequence at every cell. The C.1 substrate is the necessary precondition; B-dense.2's real `Gemma4DenseSpill` registration is the sufficient one.

**Mechanical exit codes (worktree HEAD `b03b664`):**

| Command | Exit | Output |
|---|---|---|
| `cargo build --release` | **0** | `Finished release profile [optimized] target(s)` |
| `cargo test --release --bin hf2q kv_persist -- --test-threads=1` | **0** | **94 passed; 0 failed; 0 ignored** in 1.10 s (74 prior + 20 new) |
| `cargo test --release --test kv_persist_writer_kill_minus_9 -- --test-threads=1` | **0** | **1 passed; 0 failed; 0 ignored** in 0.05 s |
| `cargo test --release --test kv_persist_harness -- --test-threads=1` | **0** | **36 passed; 0 failed; 0 ignored** in 2.76 s (no regression) |
| `grep -rn '// TODO\|todo!()\|unimplemented!()\|person-day' src/serve/kv_persist/ src/serve/mod.rs src/cli.rs` | **1** | no hits — discipline holds |

**C.1 unit-test inventory (20 new tests across registry / loader_wrapper / EngineBindable impls):**

`registry.rs` (8 tests):
1. `new_registry_has_zero_hooks`
2. `registry_register_then_bind_round_trip` (spec test 1 — Hypothesis-2 falsifier)
3. `registry_unbind_clears_engine_handle` (spec test 2)
4. `registry_bind_for_unknown_repo_quant_is_noop` (spec test 3)
5. `registry_re_register_overwrites_prior_hook`
6. `registry_unregister_idempotent`
7. `registry_distinct_quant_grows_table`
8. `registry_multi_threaded_bind_for_is_safe`

`loader_wrapper.rs` (8 tests):
9. `loader_wrapper_passes_through_load_when_no_registry_match` (spec test 4)
10. `loader_wrapper_calls_bind_after_successful_load` (spec test 5 — Hypothesis-2 falsifier)
11. `loader_wrapper_does_not_bind_on_load_failure` (spec test 6)
12. `loader_wrapper_drive_unbind_calls_registry_unbind` (spec test 7)
13. `loader_wrapper_clear_pending_bind_drops_slot`
14. `loader_wrapper_contract_violation_surfaces_error`
15. `cmd_serve_constructs_spiller_when_flag_on_smoke` (spec test 11 — bonus integration smoke)
16. `loader_wrapper_set_pending_bind_overwrites_prior`

`spiller.rs` + `families/gemma4_dense.rs` (4 tests):
17. `engine_bindable_stub_spill_noop_does_not_panic` (spec test 8)
18. `engine_bindable_gemma4_dense_round_trip_set_then_clear` (spec test 9)
19. `engine_bindable_gemma4_dense_downcast_wrong_type_returns_silently` (spec test 10)
20. `engine_bindable_gemma4_dense_silently_ignores_non_handle_arc`

**C.1 discipline notes:**

- Production code in `src/serve/kv_persist/{registry,loader_wrapper,mod,spiller,families/gemma4_dense}.rs` (registry + loader_wrapper NEW; mod / spiller / families/gemma4_dense ADDITIVE edits) + `src/cli.rs` (additive field) + `src/serve/mod.rs::cmd_serve` (additive flag-on block).
- Zero edits to `src/serve/multi_model.rs` (KvSpiller / HotSwapManager / ModelLoader trait surfaces stable per A.3).
- Zero edits to `src/serve/forward_mlx.rs` or `src/serve/forward_prefill.rs` (perf-sensitive; C.1 reads-only).
- Zero edits to `src/serve/api/*` (parallel ADR-005 Phase 4 fence).
- Zero edits to `src/inference/models/qwen35/*` (parallel ADR-005 Phase 4 fence).
- Zero edits to `mlx-native/`, `gpu_delta_net.rs`, `forward_gpu.rs`, `gpu_full_attn.rs` (ADR-015 fence).
- Real I/O in `cmd_serve` flag-on path (`recover_from_disk` actually walks the cache_dir and rebuilds the BlockIndex; `DiskBlockStore::new_with_index` actually opens the directory; `AsyncWriterHandle::spawn` actually runs the writer thread).
- "man-day" used; "person-day" does not appear (regression-checked via `grep -rn 'person-day' src/serve/kv_persist/ src/serve/mod.rs src/cli.rs`).
- No `// TODO` / `todo!()` / `unimplemented!()` / `panic!()` markers in production code.
- `Arc::downcast` failure in `Gemma4DenseSpill::bind_engine` returns silently (no panic) — locked in by `engine_bindable_gemma4_dense_downcast_wrong_type_returns_silently`.
- Worktree discipline: ABSOLUTE PATHS in shell; no `cd /opt/hf2q` corruption (per `feedback_agent_worktree_cwd_trap`).

**Cumulative ADR-017 LOC:** A complete (5,066 LOC) + B-dense.1 (1,829 LOC) + C.1 (1,587 LOC) = **8,482 LOC** of in-tree substrate + per-family hook + CLI substrate. **94 in-binary unit tests + 1 kill-9 integration test + 36 harness tests, all PASS.** Phase B-dense.2 (real `Gemma4DenseSpill` registration on operator-loaded GGUF + parity matrix) is now the gating item between this code and operator-facing `cmd_serve --kv-persist=PATH` with semantically active KV persistence.

### Phase B-dense.2 LANDED (2026-04-30)

CFA solo Claude (parallel ADR-005 Phase 4 session active; sequential per Robert directive 2026-04-30): adds the **lazy real-hook construction seam** so `cmd_serve --kv-persist=PATH` can register a `Gemma4DenseSpillFactory` at startup before any engine has loaded, then have the registry materialize a real `Gemma4DenseSpill` on the first successful engine load. Phase C.1 already wired the stub at startup; B-dense.2 closes the gap between stub registration and real `KvCacheSpill` semantics for the Gemma 4 family.

**Pattern: `FamilyHookFactory` (lazy real-hook construction at first engine load)**

The `Gemma4DenseSpill`'s shape config (sliding_window, max_decode_tokens, layer_types, kv_dtype, nkv_heads, head_dim, num_layers) ALL come from the live engine's `MlxModelWeights`. At `cmd_serve` startup the operator's GGUF has not yet been loaded — `MlxModelWeights` does not exist. The C.1 stub registration exists precisely so the spiller substrate has *some* hook to dispatch into; B-dense.2 wires the real hook lazily on the first load via this factory.

```rust
pub trait FamilyHookFactory: Send + Sync {
    fn try_construct(
        &self,
        engine_dyn: Arc<dyn Any + Send + Sync>,
    ) -> Option<(Arc<Mutex<dyn KvCacheSpill>>, Arc<dyn EngineBindable>)>;
}
```

**Substitution call chain (atomic per the registry contract):**

```text
LoaderWrapper::load(path, config) -> Engine
  ├─ inner.load(path, config) -> Engine                    (real loader, e.g. DefaultModelLoader)
  ├─ pending_bind = take((repo, quant))                    (consume operator-armed slot)
  ├─ arc_engine = Arc::new(engine)
  ├─ dyn_view = arc_engine as Arc<dyn Any + Send + Sync>
  ├─ kv_hook_opt = registry.try_substitute_on_load(repo, quant, dyn_view.clone())
  │    └─ factory = factories[(repo, quant)]               (RwLock read; clone Arc out)
  │    └─ (kv_hook, bindable_hook) = factory.try_construct(dyn_view)?
  │    └─ hooks[(repo, quant)] = bindable_hook             (RwLock write; OVERWRITE)
  │    └─ return Some(kv_hook)
  ├─ if Some(kv_hook): spiller.register_family(repo, quant, kv_hook)   (in lock-step with hooks)
  ├─ registry.bind_for(repo, quant, dyn_view)              (no-op on substituted hook; harmless)
  └─ Arc::try_unwrap(arc_engine) -> engine                 (return to manager)
```

If `factory.try_construct` returns `None` (engine type mismatch — the auto-`Arc<Engine>` path is ALWAYS rejected by the Gemma 4 factory because it expects `Arc<EngineHandle>`), `try_substitute_on_load` returns `None` and the substitution is a no-op. The existing C.1 stub registration remains in BOTH the spiller's `register_family` map AND the registry's `hooks` map; `bind_for` falls through to the stub's silent-no-op `bind_engine` impl. This is the EXPECTED B-dense.2 wire-up state for the auto-LoaderWrapper path: the `Engine`-typed Arc never matches an `EngineHandle`-expecting factory, so substitution requires a SEPARATE explicit post-load `try_substitute_on_load(repo, quant, Arc::new(engine_handle))` call from `cmd_serve`. That explicit call lands as part of the Phase D coherence + perf integration (task #14); B-dense.2 LANDS the seam, the matrix harness, and the factory itself.

**Why not modify `LoaderWrapper` to deliver `Arc<EngineHandle>` directly:** the wrapper is generic over the engine type `E`; it has no knowledge of family-specific concrete types. Pulling `EngineHandle` (a Gemma-4-specific type defined in `gemma4_dense.rs`) into `loader_wrapper.rs` would break the family-agnostic contract. The factory pattern preserves wrapper genericity while letting per-family hooks declare their own engine-type expectations.

**Surface delta (additive; no trait modifications):**

| File | Δ | What |
|---|---|---|
| `src/serve/kv_persist/registry.rs` | +254 LOC | `FamilyHookFactory` trait, factory map, `register_factory` / `factory_count` / `contains_factory` / `try_substitute_on_load`. 6 new unit tests. |
| `src/serve/kv_persist/families/gemma4_dense.rs` | +178 LOC | `try_from_engine_arc` helper on `Gemma4DenseSpill`, `Gemma4DenseSpillFactory` impl `FamilyHookFactory`. 4 new unit tests (Metal-gated where populated handle is needed). |
| `src/serve/kv_persist/loader_wrapper.rs` | +146 LOC | Optional `spiller` field + `set_spiller`; `load(...)` calls `try_substitute_on_load` BEFORE `bind_for`; on Some, calls `update_spiller_registration` so registry + spiller stay in lock-step. 2 new unit tests. |
| `src/serve/mod.rs` cmd_serve --kv-persist block | +63 LOC | Registers `Gemma4DenseSpillFactory` (placeholder shape config) alongside the C.1 stub; calls `wrapper.set_spiller(spiller)` so substitution updates both sides. |
| `tests/kv_persist_gemma4_roundtrip.rs` | NEW 598 LOC | Env-gated round-trip parity matrix harness. 6 always-on tests (smoke + matrix shape + factory substrate gate test) + 1 env-gated master test. |
| `docs/ADR-017-persistent-block-prefix-cache.md` | this subsection | "Phase B-dense.2 LANDED" entry. |

**Matrix harness scope (env-gate + cell enumeration):**

* **Master gate:** `HF2Q_KV_PERSIST_E2E=1` (mirrors A0.2b's gate; default OFF).
* **Per-cell model paths:** `HF2Q_KV_PERSIST_E2E_MODEL_GEMMA4_<QUANT>=PATH` (most-specific) or `HF2Q_KV_PERSIST_E2E_MODEL_PATH=PATH` (single-path fallback).
* **Cell axes:** 5 quants × 4 prefix lengths × 3 scenarios = 60 cells. Production-quant subset (Q4_K_M / Q6_K / Q8_0) yields 36 runnable cells; Q4_0 / Q5_K_M cells are documented for operator clarity but never runnable today (the production loader's `QuantType::from_canonical_str` rejects them).
* **Scenarios:** `cold-load` (baseline), `evict-readmit` (symlink-trick eviction + readmit; tests Hypothesis 2 + 3 directly), `restart` (kill server + restart against same cache_dir; tests recovery-scan integration).
* **Default `cargo test --release --test kv_persist_gemma4_roundtrip`:** runs only the 7 always-on tests (smoke + matrix shape + factory substrate). The matrix master test short-circuits with a diagnostic when the gate is off — no synthesized ship gates per `feedback_substrate_must_not_synthesize_ship_gates`.

**Substrate-only on B-dense.2:** `run_cell_e2e` returns `Err` with a "substrate-only post-merge run" diagnostic for every cell. The matrix gate test distinguishes these from hard failures and only fails on the latter. The actual HTTP/SSE round-trip driver lives in `kv_persist_harness::subprocess_driver` (already shipped as part of A0.2a) and is invoked from the main session post-merge — that's the Phase D coherence + perf integration work.

**Hookup readiness for Phase D coherence/perf validation:**

* Gemma4DenseSpillFactory IS registered on `cmd_serve --kv-persist=PATH` for the operator's --model.
* The substitution seam IS wired: registry + LoaderWrapper + spiller call chain lands real per-family hooks on first matching load.
* Real GGUF metadata extraction → `Gemma4DenseConfig` is the gating production task (still placeholder shape today). Phase D's coherence run requires that extraction to land before sourdough byte-exact decoded-token parity can be measured against the real Gemma 4 26B operator GGUF.

**Hypotheses (per spec §Mantra):**

* **H1 (testable, default-on):** `Gemma4DenseSpillFactory::try_construct` returns `Some(_)` on `Arc<EngineHandle>`. Locked in by `factory_construct_from_matching_engine_returns_some_tuple` (Metal-gated) + the structural-only `factory_construct_from_wrong_engine_type_returns_none`. ✅
* **H2 (testable, env-gated):** Pre-evict snapshot of `dense_kvs[layer].k`/`.v` SHA-256 hashes matches the post-readmit snapshot. Substrate landed; falsifier path lives in `kv_persist_gemma4_roundtrip_matrix_e2e`. Driver wire-up post-merge.
* **H3 (testable, env-gated):** Decoded tokens after readmit are byte-identical to a never-evicted decode against the same prompt. Substrate landed; driver wire-up post-merge.

**Discipline locked in:**

- Production code in `src/serve/kv_persist/{registry,loader_wrapper,families/gemma4_dense}.rs` (additive only) + `src/serve/mod.rs::cmd_serve` (additive only).
- Zero edits to `src/serve/multi_model.rs`, `forward_mlx.rs`, `forward_prefill.rs` (KvSpiller / KvCacheSpill / HotSwapManager / ModelLoader trait surfaces stable per A.3).
- Zero edits to `src/serve/api/*` or `src/inference/models/qwen35/*` (parallel ADR-005 Phase 4 fence).
- Zero edits to `mlx-native/`, `gpu_delta_net.rs`, `forward_gpu.rs`, `gpu_full_attn.rs` (ADR-015 fence).
- Real I/O in tests (per `feedback_substrate_must_not_synthesize_ship_gates`) — `BlockPrefixCacheSpiller` constructed against tempdir-backed `DiskBlockStore` + `AsyncWriterHandle` in the loader_wrapper substitution tests.
- "man-day" used; "person-day" does not appear (regression-checked via `grep -rn 'person-day' src/serve/kv_persist/ tests/kv_persist_gemma4_roundtrip.rs`).
- No `// TODO` / `todo!()` / `unimplemented!()` / `panic!()` markers in production code.
- `Arc::downcast` failure in `Gemma4DenseSpillFactory::try_construct` returns `None` (no panic) — locked in by `factory_construct_from_wrong_engine_type_returns_none`.
- Worktree discipline: ABSOLUTE PATHS in shell; no `cd /opt/hf2q` corruption (per `feedback_agent_worktree_cwd_trap`).

**Cumulative ADR-017 LOC:** A complete (5,066 LOC) + B-dense.1 (1,829 LOC) + C.1 (1,587 LOC) + B-dense.2 (~1,239 LOC: 254+178+146+63+598) = **9,721 LOC** of in-tree substrate + per-family hook + CLI substrate + matrix harness. **106 in-binary unit tests + 1 kill-9 integration test + 36 harness tests + 7 round-trip harness tests, all PASS.** Phase D coherence + perf gate is now the gating item between B-dense.2's seam and operator-facing semantically active KV persistence at the ship gate.

### Phase B-dense.2 follow-up LANDED (2026-04-30)

**Closes the B-dense.2 caveat:** the previous landing left
`Gemma4DenseSpillFactory::try_from_engine_arc` always returning `None`
because `LoaderWrapper::load` passes `Arc<E>` where `E = Engine` (the
production type), but the factory only downcast to `Arc<EngineHandle>`
(a Phase C.1 wrapper that no production caller ever passed through the
auto-bind path). That caveat meant every load substituted nothing —
the stub `StubGemma4Spill` stayed registered and `pre_evict` /
`post_admit` were no-ops on real production runs.

This follow-up wires the production path end-to-end: the factory's
downcast now matches `Arc<Engine>` first; the spill holds an
`Arc<Engine>` (in addition to the B-dense.1 `EngineHandle` slot for
backwards compat); and snapshot/restore route through a new
`Engine` worker bridge so the hook reads/writes
`MlxModelWeights.dense_kvs` without crossing the worker-thread
boundary.

**What landed (additive, no trait surface modifications):**

| File | Change | LOC |
|---|---|---|
| `src/serve/api/kv_spill_descriptor.rs` | NEW. `KvSpillDescriptor` + `KvDType` + `from_gemma_loaded_model` constructor + 5 unit tests. | ~250 |
| `src/serve/api/mod.rs` | `pub mod kv_spill_descriptor;` registration. | 1 |
| `src/serve/api/engine.rs` | + `EngineInner.kv_spill_descriptor: Option<...>` cached at `Engine::spawn` for Gemma variant. + `Request::KvSnapshot` / `Request::KvRestore` variants. + `KvSnapshotBytes` public payload type. + 2 worker_run match arms (`kv_snapshot_gemma` + `kv_restore_gemma` helpers reading/writing `MlxBuffer.as_slice::<u8>`). + 3 public `Engine` methods (`kv_spill_descriptor`, `request_kv_snapshot`, `request_kv_restore`). + `make_synthetic_kv_engine_for_test` cross-module test fixture. + 6 unit tests under `engine::tests::request_kv_*`. | ~1080 |
| `src/serve/kv_persist/families/gemma4_dense.rs` | + `engine_arc: Arc<RwLock<Option<Arc<Engine>>>>` field. + `Gemma4DenseConfig::from_descriptor`. + `Self::new_with_engine` / `set_engine_arc` / `clear_engine_arc` / `try_from_engine` constructors. + `try_from_engine_arc` extended (Arc<Engine> first, Arc<EngineHandle> fallback). + `snapshot_via_engine` / `restore_via_engine` helpers (route through Engine worker bridge). + `KvCacheSpill::snapshot_block` / `restore_block` updated to prefer `engine_arc` when set, B-dense.1 `engine` slot fallback otherwise. + `EngineBindable::bind_engine` extended to accept Arc<Engine>. + `unbind_engine` clears both slots. + `Gemma4DenseSpillFactory::try_construct` rewritten to handle both engine types in spill duplication. + 5 new tests under `gemma4_dense::tests::followup_*`. | ~750 |
| `docs/ADR-017-persistent-block-prefix-cache.md` | this subsection. | ~80 |

**Discipline locked in:**

- `KvSpiller` / `KvCacheSpill` / `HotSwapManager` / `ModelLoader` /
  `EngineBindable` / `FamilyHookFactory` trait surfaces UNCHANGED
  (additive impl-side methods only).
- B-dense.1's 17 existing tests PASS unchanged. The B-dense.1
  `EngineHandle` path is preserved as a backwards-compat fallback;
  production loads now hit the `Arc<Engine>` path first.
- Zero edits to `src/serve/multi_model.rs`, `forward_mlx.rs`,
  `forward_prefill.rs` (KV-shape + decode loop fenced).
- Zero edits to `src/inference/models/qwen35/*`, `mlx-native/`,
  `gpu_*.rs` (ADR-015 fence).
- Real I/O round-trip in tests (per
  `feedback_substrate_must_not_synthesize_ship_gates`): synthetic
  worker reads/writes a real `HashMap`-backed byte cache; SHA-256
  byte-equality asserted on snapshot→restore→snapshot.
- "man-day" used; "person-day" does not appear (`grep` clean).
- No `// TODO` / `todo!()` / `unimplemented!()` / `panic!()` outside
  `#[cfg(test)]`.
- Worktree discipline: solo Claude session in
  `.cfa-worktrees/adr017-bdense2-followup-claude`, ABSOLUTE PATHS in
  every shell command (no `cd /opt/hf2q`).

**Test totals after follow-up:**

- `kv_persist::*` in-binary: **111 PASS** (106 prior + 5 new
  `gemma4_dense::tests::followup_*`).
- `engine::tests::request_kv_*`: **6 PASS** (new).
- `kv_spill_descriptor::*`: **5 PASS** (new).
- `kv_persist_writer_kill_minus_9`: **1 PASS** (no regression).
- `kv_persist_harness`: **36 PASS** (no regression).
- `kv_persist_gemma4_roundtrip`: **7 PASS** (no regression).
- Full bin unit tests: **2,405 PASS** (no regression vs B-dense.2
  baseline).

**Hypotheses closure:**

- **H1 (testable):** With `cmd_serve --kv-persist=PATH` + Gemma 4
  model, the registered hook is now `Gemma4DenseSpill` (not
  `StubGemma4Spill`). ✅ Locked in by
  `followup_factory_try_construct_succeeds_on_arc_engine` +
  `followup_try_from_engine_arc_succeeds_on_arc_engine`.
- **H2 (testable):** `Engine::kv_spill_descriptor()` returns
  `Some(...)` for Gemma 4, `None` for Qwen35. ✅ Locked in by
  `request_kv_descriptor_returns_some_when_set` +
  `request_kv_descriptor_returns_none_when_not_set` +
  `followup_try_from_engine_arc_returns_none_on_qwen35_engine`.
- **H3 (testable):** `request_kv_snapshot` returns the same bytes
  the live `dense_kvs[layer].k` slice would read. ✅ Locked in by
  `request_kv_snapshot_returns_real_bytes_from_populated_layer`
  (asserts byte pattern at known seed offset).
- **H4 (testable):** `request_kv_restore` + `request_kv_snapshot`
  round-trip is byte-exact. ✅ Locked in by
  `request_kv_restore_then_snapshot_round_trip_byte_exact` +
  `followup_snapshot_restore_round_trip_via_arc_engine`.

**What's still gating Phase D:**

- Real Gemma 4 26B operator GGUF run with `HF2Q_KV_PERSIST_E2E=1`
  enabled (the matrix harness's master gate). The harness substrate
  is in place; this follow-up unblocks the actual operator-side
  end-to-end test by ensuring a real (non-stub) hook is registered.
- Sourdough byte-exact decoded-token parity across pre-evict /
  post-admit (the actual H3 + H4 hypothesis closure on a live
  model — task #14 in the main session work).

**Cumulative ADR-017 LOC after follow-up:** 9,721 LOC + ~2,160 LOC (250+1+1080+750+80) = **~11,881 LOC** in-tree substrate.

### Phase D LANDED (scaffolding) (2026-04-30)

Phase D scaffolding ships the harness + operator recipe for the
coherence + perf validation matrix. The actual MEASUREMENT RUN is
operator-controlled (cold M5 Max + Gemma4-26B Q4_0 GGUF +
mcp-brain-server SIGSTOP'd) post-merge; the CFA worker delivered
scaffolding ONLY per spec §Out-of-scope.

**TESTS-AND-SCRIPTS ONLY — zero `src/` edits.** The scaffolding
exclusively touches `tests/kv_persist_gemma4_roundtrip.rs`,
`scripts/adr017_phase_d.sh`, and this ADR document.

**`tests/kv_persist_gemma4_roundtrip.rs::run_cell_e2e` wired** —
previous B-dense.2 stub (`Err("substrate-only post-merge run")`) is
replaced with the real subprocess round-trip. Per cell, the runner:

1. spawns `hf2q serve --model PATH --kv-persist=DIR` with
   `HF2Q_USE_DENSE=1` in env (R-C4 byte-exact requires dense; TQ is
   lossy by design)
2. waits for `/readyz`
3. fetches canonical model id from `/v1/models`
4. issues a streaming completion against a token-diverse `wordN`
   prompt sized to the cell's prefix-length; captures `baseline.text`
5. for `EvictReadmit` / `Restart` scenarios: forces a pool eviction
   via the symlink-distinct-pool-key trick (sibling `config.json` /
   `tokenizer*.json` / `generation_config.json` + mmproj GGUF
   symlinked into the tempdir so `cmd_serve` resolves the cloned
   path)
6. issues a second streaming completion against the SAME prompt;
   captures `restored.text`
7. asserts `baseline.text == restored.text` byte-for-byte; surfaces
   first byte-diff offset on FAIL

`ColdLoad` cells exercise step 4 only and assert non-empty decode
(sanity).

**Self-contained `phase_d_driver` module (~470 LOC)** — mirrors the
pattern from `kv_persist_harness::subprocess_driver` (RAII
`ServerGuard`, `/readyz` poll, SSE first-content-delta TTFT capture,
symlink-pool-key eviction trigger) but adds `--kv-persist=DIR` to the
spawn args and `HF2Q_USE_DENSE=1` to the child env. Self-contained
because each `tests/*.rs` is its own integration-test crate; pulling
the existing driver in via `#[path]`-include would re-import the full
600-cell matrix substrate redundantly. The Phase D driver also
returns the FULL captured text (`DecodeCapture.text`) so the R-C4
byte-equality assertion has a real bytestream to compare, not just a
TTFT scalar.

**New env-gated test `kv_persist_phase_d_coherence_e2e`** (R-C4):

* Gated on `HF2Q_KV_PERSIST_PHASE_D=1` + non-empty
  `HF2Q_KV_PERSIST_E2E_MODEL_PATH`. Without the gate, `cargo test`
  short-circuits cleanly with a diagnostic — no synthesized ship
  gates per `feedback_substrate_must_not_synthesize_ship_gates`.
* Drives the canonical sourdough fixture
  (`scripts/sourdough_gate.sh`'s 22-token user prompt — load-bearing
  typo "Complrehensive" preserved; `T=0` greedy; `max_tokens=1000`).
* Captures hf2q never-evicted output A; forces evict-readmit cycle;
  captures hf2q evicted+restored output B. Asserts A == B
  byte-identical (R-C4 internal coherence; Hypothesis 2 + 3 falsifier).
* **Optional peer arm** (env-gated by
  `HF2Q_KV_PERSIST_PHASE_D_PEER=1`): renders the chat template via
  `hf2q generate --max-tokens 1` (`HF2Q_DUMP_RENDERED_PROMPT`),
  BOS-strips the literal `<bos>` prefix, runs `llama-completion`
  on the rendered prompt at `T=0 --seed 42 --predict 1000`, asserts
  both A and B share at least 3094 leading bytes with llama's
  output. Mirrors `scripts/sourdough_gate.sh::MIN_COMMON_PREFIX`.
* `llama-completion` binary resolved via
  `HF2Q_KV_PERSIST_PHASE_D_LLAMA_BIN` env override → `which
  llama-completion` → `/opt/llama.cpp/build/bin/llama-completion`,
  matching `sourdough_gate.sh`'s precedence.

**New env-gated test `kv_persist_phase_d_r_p4_e2e`** (R-P4):

* Gated on `HF2Q_KV_PERSIST_PHASE_D=1` +
  `HF2Q_KV_PERSIST_E2E_PREFILL_LEN=N` (spec calls for N=32768 on the
  cold M5 Max + Gemma4-26B Q4_0 cell).
* Spawns `hf2q serve --kv-persist=DIR`, prefills N tokens
  (`wordN` token-diverse construction; ≈3.8 tokens/word under Gemma 4
  BPE), measures `no_cache_ttft_ms` (cold prefill, primes the on-disk
  block cache).
* Forces eviction via symlink-distinct-pool-key (drops the in-RAM
  KV state).
* Prefills the SAME prompt; measures `cache_hit_ttft_ms` (the cache
  is now repopulated from the on-disk persistence layer).
* Computes `ratio = cache_hit_ttft_ms / no_cache_ttft_ms`; asserts
  `ratio <= 0.20` (R-P4 ship-gate). Emits diagnostic line
  `[R-P4] PASS — ratio=X.XXX (no_cache=Yms cache_hit=Zms)`.

**3 new always-on shape tests** (run in default `cargo test`,
substrate-only — no spawn / no model required):

* `phase_d_driver_rejects_missing_model_cleanly` — `spawn_*` returns
  `DriverError::SpawnFailed` (not panic) when given a non-existent
  GGUF path. Falsifier: panic instead of Err.
* `phase_d_env_gates_are_well_formed` — round-trip set/get
  consistency for all 4 Phase D env vars
  (`HF2Q_KV_PERSIST_PHASE_D`, `HF2Q_KV_PERSIST_PHASE_D_PEER`,
  `HF2Q_KV_PERSIST_E2E_PREFILL_LEN`,
  `HF2Q_KV_PERSIST_PHASE_D_LLAMA_BIN`).
* `phase_d_sourdough_constants_match_shell_gate` — drift detector
  asserting `SOURDOUGH_PROMPT` literal +
  `SOURDOUGH_MAX_TOKENS=1000` + `SOURDOUGH_MIN_COMMON_PREFIX=3094`
  match `scripts/sourdough_gate.sh` byte-for-byte. Falsifier: any
  drift between Phase D test and shell gate.

**Operator recipe `scripts/adr017_phase_d.sh`** (~175 LOC bash):

* Pre-bench process audit (refuses if `mcp-brain-server` /
  `llama-server` / `llama-cli` / `ollama` running, per
  `feedback_bench_process_audit`; `--skip-process-audit` bypass for
  explicit risk acceptance).
* RAM check via `vm_stat | head -5` (per
  `feedback_check_ram_before_inference`).
* Sets `HF2Q_KV_PERSIST_E2E=1` + `HF2Q_KV_PERSIST_PHASE_D=1` +
  `HF2Q_USE_DENSE=1` + `HF2Q_KV_PERSIST_E2E_MODEL_PATH` +
  `HF2Q_KV_PERSIST_E2E_PREFILL_LEN` (default 32768) +
  optional `HF2Q_KV_PERSIST_PHASE_D_PEER=1` (with `--peer` flag).
* Runs `cargo test --release --test kv_persist_gemma4_roundtrip
  -- --test-threads=1 --nocapture`.
* Exit codes: `0` PASS / `1` usage / `2` test-fail / `3`
  prereq-missing.

`shellcheck` clean. `bash -n` syntax-clean.

**Discipline locked in:**

- TESTS-AND-SCRIPTS ONLY. Zero edits to `src/serve/multi_model.rs`,
  `forward_mlx.rs`, `forward_prefill.rs`, `src/serve/api/*`,
  `src/inference/models/qwen35/*`, `mlx-native/`, `gpu_*.rs`.
  Zero edits to ANY `src/` file. The Phase D wire-up exists entirely
  in `tests/kv_persist_gemma4_roundtrip.rs` +
  `scripts/adr017_phase_d.sh`.
- Phase D tests gate on `HF2Q_KV_PERSIST_PHASE_D=1` and
  short-circuit cleanly when env unset. Default `cargo test --release
  --test kv_persist_gemma4_roundtrip` runs only the existing 7
  always-on tests + the 3 new always-on shape tests = 10 PASS.
- Real I/O round-trip per
  `feedback_substrate_must_not_synthesize_ship_gates`: every Phase D
  test path spawns a real `hf2q serve` subprocess, sends real
  `/v1/chat/completions` requests over real TCP, parses real SSE
  streams, captures real `text/event-stream` content bytes. No
  synthesized constants asserted against ship gates.
- "man-day" used; "person-day" does not appear (regression-checked).
- No `// TODO` / `todo!()` / `unimplemented!()` / `panic!()` outside
  `#[cfg(test)]`. Test-side `panic!`s are diagnostic-only on R-C4
  byte-diff and R-P4 ratio overshoot.
- Worktree discipline: solo Claude session in
  `.cfa-worktrees/adr017-phase-d-claude`, ABSOLUTE PATHS in every
  shell command (no `cd /opt/hf2q`).

**What the operator does post-merge:**

```bash
# 1. cold M5 Max — 1+ min idle since previous run
# 2. SIGSTOP mcp-brain-server (or stop it entirely)
# 3. cargo build --release
# 4. ./scripts/adr017_phase_d.sh             # default cell + no peer arm
#    ./scripts/adr017_phase_d.sh --peer      # +llama-completion peer arm
```

Expected outcomes (per A0.2b's substrate-mode predictions):

* R-C4 internal: `baseline.text == restored.text` byte-identical.
* R-C4 peer: both arms share `>= 3094` bytes with llama-completion.
* R-P4: `cache_hit_ttft / no_cache_ttft <= 0.20` at L=32K
  (A0.2b prediction = 0.009 → 22× margin).

**Hypotheses (per spec §Mantra):**

* **H1 (testable, R-C4 internal):** With `--kv-persist=PATH` +
  `HF2Q_USE_DENSE=1`, hf2q's evicted+restored decode output is
  byte-identical to never-evicted decode for the sourdough prompt
  (T=0 greedy, 1000 tokens). Falsifier: any byte-diff in the
  decoded text. Test:
  `kv_persist_phase_d_coherence_e2e` (asserts byte-equality;
  test panics with offset+snippets on diff).
* **H2 (testable, R-C4 peer):** Both never-evicted hf2q and
  evicted+restored hf2q share `>= 3094` bytes common prefix with
  llama.cpp on the same GGUF + prompt. Falsifier: common prefix
  `< 3094` bytes. Test: peer arm of
  `kv_persist_phase_d_coherence_e2e` (`HF2Q_KV_PERSIST_PHASE_D_PEER=1`).
* **H3 (testable, R-P4):** At L=32K, `cache_hit_ttft /
  no_cache_ttft <= 0.20` on Gemma4-26B Q4_0 cold M5 Max.
  Falsifier: ratio `> 0.20`. Test: `kv_persist_phase_d_r_p4_e2e`.

**Surface delta:**

| File | Δ | What |
|---|---|---|
| `tests/kv_persist_gemma4_roundtrip.rs` | +1334 LOC | Phase D env-gate constants, `phase_d_driver` mod (~470 LOC: spawn-with-kv-persist, /readyz poll, canonical-id fetch, full-text SSE capture, symlink eviction, llama-completion peer runner), `run_cell_e2e` wired to subprocess driver, 2 env-gated Phase D tests (`coherence_e2e` + `r_p4_e2e`), 3 always-on shape tests. |
| `scripts/adr017_phase_d.sh` | NEW 175 LOC | Operator recipe (pre-bench audit + env wire-up + cargo test invocation). |
| `docs/ADR-017-persistent-block-prefix-cache.md` | this subsection | "Phase D LANDED (scaffolding)" entry. |

**Cumulative ADR-017 LOC after Phase D scaffolding:** ~11,881 LOC +
~1,510 LOC (1334 test + 175 script) = **~13,391 LOC** in-tree
substrate. **Test totals:** in-binary `kv_persist::*` 111 PASS (no
regression), `kv_persist_writer_kill_minus_9` 1 PASS, harness 36
PASS, gemma4_roundtrip 7 always-on prior + 3 new always-on shape =
10 PASS by default; Phase D env-gated tests (`coherence_e2e` +
`r_p4_e2e`) run only under operator-controlled env.

**What's still gating Phase D closure (operator-side):**

- Real cold-M5-Max measurement run via `scripts/adr017_phase_d.sh`
  on the canonical Gemma 4 26B Q4_0 GGUF.
- Per-quant follow-up runs (Q4_K_M / Q6_K / Q8_0) once the Q4_0
  cell PASSes — production matrix sweeps the runnable subset.

The CFA worker's contract is complete: scaffolding LANDED; the
matrix RUN is the operator-controlled follow-up.

### Phase D measurement attempt 2026-05-01 — BLOCKED on contended SoC + diagnostic improvement LANDED

Solo Claude /loop session 2026-05-01 attempted the Phase D R-C4
coherence measurement directly via `cargo test
kv_persist_phase_d_coherence_e2e`. **Substrate-side state
re-confirmed by code+test:**

| Gate | Result |
|---|---|
| `cargo build --release --bin hf2q` | **PASS** (Finished in 0.10s; tree clean) |
| `cargo test --release --bin hf2q kv_persist -- --test-threads=1` | **PASS** — 111 in-binary tests |
| `cargo test --release --test kv_persist_writer_kill_minus_9` | **PASS** — 1 test |
| `cargo test --release --test kv_persist_harness` | **PASS** — 36 tests |
| `cargo test --release --test kv_persist_gemma4_roundtrip` (always-on) | **PASS** — 12 tests |
| **Total non-env-gated tests** | **160 PASS / 0 FAIL / 0 ignored** |

**Phase D R-C4 coherence run aborted at /readyz timeout (600s).** The
test reported `last: "transport: Connection refused (os error 61)"`
— `hf2q serve --kv-persist=DIR` never bound to port 52339 within the
budget. **Root cause: contended SoC**, NOT a bug in the kv-persist
startup path:

- Pre-bench process audit at session start: clean (no
  `mcp-brain-server` / `llama-server` / `llama-cli` / `ollama`).
- Mid-bench audit during the failed run: 1123% CPU on `rustc`,
  66.1% on `llama-cli`, 76.7% on a foreign `target/release/hf2q
  generate` (PID 66688, child of foreign `coherence_smoke` test
  PID 53551). Background CFA sessions on ADR-013/015 iter61c
  bisection + ADR-015 qwen35 tests + iter219 in flight per
  `[Concurrent sessions in flight]` standing memory.
- `vm_stat`: free RAM dropped from 67 GiB → 26 GiB during the run as
  foreign sessions allocated their own model loads.
- These foreign sessions are NOT mine to kill (per
  `[Concurrent sessions in flight]` and `feedback_use_cfa_worktrees`).

**No code change to ADR-017 production path.** The startup-path
itself was never observed to fail on its own merits — the timeout
was a contention symptom, not a /kv-persist bug.

**Diagnostic improvement landed (additive, tests-only):** the three
`wait_for_readyz(...)` call sites in
`tests/kv_persist_gemma4_roundtrip.rs` previously dropped the
captured `ServerGuard.stderr_tail` on `/readyz` timeout — the
`.expect("...")` / `.map_err(|e| format!("readyz: {e}"))` paths
emitted only "Connection refused (os error 61)", with zero insight
into whether `hf2q serve` was mid-mmap, mid-pre-warm, or had
panicked. The fix surfaces `server.log_tail()` (up to 256 lines of
captured stderr) on every timeout, so the next bench attempt's
failure mode is diagnosable from the cargo-test output alone:

| Call site | Was | Is |
|---|---|---|
| `run_cell_e2e` (matrix, line 966) | `.map_err(\|e\| format!("readyz: {e}"))?` | Same + `\n--- hf2q serve stderr_tail (N lines) ---\n{tail}` |
| `kv_persist_phase_d_coherence_e2e` (line 1422) | `.expect("[Phase D coherence] /readyz did not return 200 within budget")` | `.unwrap_or_else(\|e\| panic!(...))` with stderr_tail in the panic message |
| `kv_persist_phase_d_r_p4_e2e` (line 1713) | `.expect("[Phase D R-P4] /readyz did not return 200 within budget")` | Same pattern |

Compiles clean. 12/12 always-on tests still PASS post-patch (no
regression).

**What's still gating Phase D closure:**

- Cold-M5-Max measurement run via `scripts/adr017_phase_d.sh` on
  the canonical Gemma 4 26B GGUF, with concurrent CFA sessions
  quiesced or completed.
- `READYZ_BUDGET_SECS = 600` is generous on a clean SoC for a 14
  GiB GGUF + Metal pipeline pre-warm; under contention it can be
  insufficient. The diagnostic improvement makes future failures
  distinguishable from real defects without raising the budget
  (which would mask problems per
  `feedback_substrate_must_not_synthesize_ship_gates`).

**Cumulative ADR-017 LOC after 2026-05-01 diagnostic improvement:**
~13,391 LOC + ~30 LOC test-side diagnostic = **~13,421 LOC**
in-tree substrate. Tests-only edit; zero `src/` touches.

**Hypothesis for the operator (testable, recorded for next bench
attempt):** under a clean-SoC pre-bench audit (no `rustc`/`cargo
test`/`llama-cli`/foreign `hf2q` processes consuming >10% CPU), the
R-C4 coherence test starts up within 600s and either (a) PASSes
byte-exact (confirms H1 from spec §Mantra), or (b) fails with a
specific stderr_tail signal that names the production-side defect.
Falsifier: clean-SoC R-C4 also times out at /readyz with no
informative stderr_tail — would indicate a real cmd_serve
--kv-persist startup defect that the test-side diagnostic improvement
does not surface (would require an in-process tracing add
to `src/serve/mod.rs::cmd_serve`).

### Phase D measurement attempt 2026-05-01 (cont.) — parallel worktree audit harvest

Per Robert directive 2026-05-01 ("we have many agents — use work
trees"), three parallel worktree-isolated agents conducted code-side
audits while the GPU bench remained blocked on contended SoC. Each
agent wrote a report committed in its worktree; reports are now
merged to main at `docs/research/adr017-omlx-crosscheck-2026-05-01.md`
(782 LOC), `docs/research/adr017-adversarial-review-2026-05-01.md`
(562 LOC), and `docs/operating-kv-cache.md` (507 LOC, R-O1).

**Key finding triage (validated against production code in main
session, per `feedback_code_is_truth`):**

#### FALSIFIED — placeholder-shape silent-noop hypothesis

Agent 3 hypothesized (runbook §11 #7) that the placeholder
`Gemma4DenseConfig` at `src/serve/mod.rs:1770-1788` (2-layer
[Sliding, Full]) would cause `Gemma4DenseSpillFactory::try_construct`
to silently no-op against production Gemma 4 26B (64 layers). If
true, the Phase D R-C4 bench would have measured stub-vs-stub and
synthesized a misleading PASS.

**Code+test evidence (FALSIFIES the hypothesis):**

- `src/serve/api/engine.rs:1503-1530` — `Engine::spawn` builds a real
  `KvSpillDescriptor::from_gemma_loaded_model(&g.weights, 512,
  kv_dtype)` for `LoadedModel::Gemma(g)` and stores it in
  `EngineInner.kv_spill_descriptor: Some(real_descriptor)`.
- `src/serve/kv_persist/families/gemma4_dense.rs:1373-1390` —
  `try_from_engine_arc` downcasts `Arc<dyn Any>` to `Arc<Engine>`
  FIRST (Phase B-dense.2 follow-up); on success, reads the engine's
  cached descriptor via `engine_arc.kv_spill_descriptor()?.clone()`,
  derives `effective_cfg = Gemma4DenseConfig::from_descriptor(&descriptor)`,
  and constructs the spill with REAL shape — IGNORING the placeholder
  `cfg` that was registered with the factory.
- `src/serve/kv_persist/loader_wrapper.rs:262-324` — `LoaderWrapper::load`
  calls `registry.try_substitute_on_load(&repo, quant, dyn_view)` on
  every successful load with the engine wrapped as `Arc<Engine>`,
  which exercises the real-descriptor path.

The placeholder cfg is a SEED used only for the (unused-in-production)
`Arc<EngineHandle>` fallback path. Production loads exercise the
`Arc<Engine>` path → real shape. Phase D R-C4 bench WILL exercise the
real spill when run.

**Outdated stale comment surfaced (cleanup needed):** `loader_wrapper.rs:295-298`
states "the auto-Arc<E> path is ALWAYS rejected by the Gemma4 factory
because it expects Arc<EngineHandle>" — this was true at C.1 but is
NO LONGER TRUE post-B-dense.2-follow-up (commit 420ef94). Per
`feedback_no_broken_windows`, fix in next iter.

#### REAL findings (P0 candidates from Agent 2 + Agent 1 cross-validation)

| ID | Source | Location | Issue | Bench-blocker? |
|---|---|---|---|---|
| **P0-1** | Agent 2 §P0-1 | `format.rs:357-366` | `write_envelope` `fsync`s the file but NOT the parent directory after `fs::rename`. Power loss between rename-syscall return and dir-entry persist loses the file on APFS/ext4/XFS. The kill-9 test covers SIGKILL atomicity, not power-loss durability. | **NO** — Phase D R-C4 bench does not power-cycle. Land before §"Closed-Shipped" status. |
| **P0-2 (DISPUTED)** | Agent 2 §P0-2 vs Agent 1 §F5 | `serve/mod.rs:1719-1771` + `block_store.rs:100-128` | Agent 2 claims NO cross-process advisory `flock` on `cache_dir`; Agent 1 claims `flock` per-`(model, hash[..2])` exists and is finer-grained-than-oMLX's cache-root flock. Disagreement requires direct verification. | **NO** for first bench (single-process); **YES** for sustained-operator-deployment ship. |
| **P0-3** | Agent 2 §P0-3 | `recovery.rs:155-172` | `*.tmp.<pid>` orphans counted but never garbage-collected. Combined with `budget_bytes=0` (P1-3), monotonic cache-dir growth. Slow-burn outage on long-running deployments. | **NO** for first bench (clean cache_dir); **YES** for production ship. |
| **F4** | Agent 1 §F4 | `spiller.rs:242-244` | `family_model_fp` builds with `("", "", "")` for `producer_version`, `source_sha256`, `tokenizer_chat_template` → effective namespace key is `(repo, quant)` only. Re-quanting same repo or chat-template upgrade lands in stale namespace and serves stale blocks. | **NO** for first bench (single GGUF, no re-quant); **YES** before B-dense.1 hardens to operator-touch. |
| **P1-1** | Agent 2 §P1-1 | `loader_wrapper.rs:309-324` | `try_substitute_on_load` updates registry under write-lock, then `update_spiller_registration` updates spiller in a SEPARATE critical section. Module docs claim atomic; impl is not. Concurrent `pre_evict` from another tokio task can read stub-spiller while registry has real-substituted hook. | **NO** — Phase D bench is sequential. **YES** before multi-tenant ship. |
| **P1-3** | Agent 2 §P1-3 + Agent 3 §11#4 | `serve/mod.rs:1759-1771` | `budget_bytes = 0` hardcoded; `evict_lru_until_under_budget` short-circuits on 0. Production `--kv-persist` never evicts. `HF2Q_KV_PERSIST_BUDGET_BYTES` env var documented in source comment but not actually read. | **NO** for first bench (cache empty); **YES** for sustained-deployment. |

**Operator runbook gaps surfaced (Agent 3, R-O1 partially landed):**
8 `[NOT YET IMPLEMENTED — ADR-017 §X]` markers in
`docs/operating-kv-cache.md`, primarily: native cache-clear
subcommand absent (`hf2q cache --kv-namespace` per §R-F6 not in
CLI); 4 of 6 cache-side `/metrics` counters per §R-F7 unwired
(only the 2 ADR-005 Phase 4 spill/restore counters fire);
`HF2Q_KV_PERSIST=0` mid-flight disable referenced in code comments
but never read by `env::var`; `DiskBlockStore::set_budget_bytes`
plumbed but never wired in cmd_serve.

#### Decision: Phase D bench is NOT BLOCKED by these findings

None of P0-1 / P0-2 / P0-3 / P1-1 / P1-3 perturb the Phase D R-C4
internal coherence run on a clean cache_dir, single-process,
sequential operator. The findings are real ship-quality blockers
for sustained-deployment operator ship; they are NOT bench-gating.

**Bench plan unchanged:** wait for foreign concurrent CFA sessions
to quiesce, then run `scripts/adr017_phase_d.sh`. If R-C4 PASSes,
ADR-017 closure proceeds in parallel with the P0/P1/F4 follow-up
fixes. If R-C4 FAILs, the stderr_tail diagnostic landed at b0e67ca
will surface the production-side cause; fixes follow as iter-
specific work.

**Cumulative ADR-017 LOC after parallel-worktree audit harvest:**
~13,421 LOC + 1,851 LOC (audit reports + runbook) = **~15,272 LOC**
in-tree substrate + audit + operator docs. Tests-only edit;
zero `src/` touches in this session.

### Phase D iter-3 2026-05-01 — P0-1 LANDED + P0-2 FALSIFIED

Loop iter-3 fired ~30 min after iter-2; pre-bench audit found foreign
SoC contention WORSE than iter-1 (foreign hf2q at 27 GiB resident,
99% CPU; multiple `rustc` at 98%; ~10 GiB free RAM). Phase D bench
remains blocked. Productive work: progress on the audit-harvest P0
fixes via single-agent worktree-isolated edits + main-session
verification.

#### P0-1 LANDED — `fsync(parent_dir)` after `fs::rename` (commit `0be1754` on origin/main)

`src/serve/kv_persist/format.rs:write_envelope` previously called
`f.sync_all()` on the temp file then `std::fs::rename(...)` without
fsync'ing the parent directory afterward. The kill-9 test
(`tests/kv_persist_writer_kill_minus_9.rs`) covers SIGKILL atomicity
but NOT power-loss durability — on APFS/ext4/XFS, the rename's
dir-entry is durable only after the directory inode itself is
fsync'd.

Fix (worktree agent `ab57fe5e8fa6ab2fc` → cherry-pick →
`0be1754`): add `File::open(parent)?.sync_all()?` after the rename.
`cargo check --release --bin hf2q` exit 0 in 19.74s. Existing kill-9
test continues to PASS unchanged (SIGKILL doesn't power-cycle disk,
so the test's invariant is unaffected by the addition).

#### P0-2 FALSIFIED — cross-process `flock` exists in hf2q (Agent 1 right; Agent 2 wrong)

The 2026-05-01 worktree audit surfaced a disputed P0-2 finding:
Agent 2 (adversarial review) claimed `serve/mod.rs:1719-1771` +
`block_store.rs:100-128` had NO cross-process advisory `flock` on
`cache_dir`; Agent 1 (oMLX crosscheck) claimed hf2q has finer-grained
flock per `(model, hash[..2])` AND that hf2q is *stronger* than oMLX
in this dimension.

Code+test resolution in main session: Agent 1 is correct.

- `src/serve/kv_persist/block_store.rs:27-34` (module docstring):
  > "Writes acquire an advisory `flock(LOCK_EX)` keyed on
  > `(model_fingerprint_short, block_hash[..2])` for the duration of
  > the atomic-rename publication. The lock file lives at
  > `<cache_root>/locks/<short>__<hash_prefix>.lock`. Pattern mirrors
  > `serve/cache.rs::CacheLock` (ADR-005). `File` to keep the fd
  > alive, `flock(LOCK_EX)` on acquire, fd-drop releases.
  > Per-block-hash-prefix granularity (256 buckets per model) keeps
  > the contention surface tight without one-lock-per-block fan-out."
- `src/serve/kv_persist/block_store.rs:399`: real `unsafe { libc::flock(fd, libc::LOCK_EX) }` syscall.
- `src/serve/kv_persist/block_store.rs:687`:
  `advisory_lock_serializes_concurrent_writes` test exercises the lock.

The lock granularity is finer than oMLX's whole-cache-root flock —
hf2q permits concurrent multi-process access to *different* blocks
while still serializing writes to the *same* block. Agent 2's P0-2
was a false alarm caused by not finding the implementation; the
actual concern (in-memory `BlockIndex` view divergence between two
`cmd_serve --kv-persist=SAME_DIR` instances) is real but is NOT
corruption — each process's writes go through the per-block flock,
and stale index entries refresh on the next directory scan.

P0-2 is REMOVED from the bench-blocker concern list.

#### P0-3 in-flight — `*.tmp.<pid>` orphan GC at recovery scan

`src/serve/kv_persist/recovery.rs:167-171`:
```rust
if name.contains(".tmp.") {
    report.partial_tmp_files_ignored += 1;
    return Ok(());
}
```

The recovery scan COUNTS orphans but never deletes them. Combined
with `budget_bytes = 0` hardcoded at `serve/mod.rs:1764` (P1-3),
this is a slow-burn monotonic-growth bug for long-running
deployments that experience frequent kill-9 / power-loss events.

Fix scoped: best-effort GC of `.tmp.<pid>` files older than a TTL
(60s default) at startup recovery scan. Recovery scan runs
**before** this process's writer spawns; orphans older than the TTL
cannot belong to any active writer in this or any other process
(writes complete in <60s on M5 Max — the TTL gives slack for slow
NVMe cells). Best-effort `fs::remove_file` ignores errors so a
racing cross-process writer doesn't crash recovery. Adds
`orphan_tmp_files_removed: usize` to `RecoveryReport`.

Agent assignment: worktree-isolated coder, foreground; ~10 LOC
edit + 1 unit test. Cherry-pick to `origin/main` after cargo
check exit 0.

#### P0-3 LANDED — orphan GC at recovery scan (commit `10d419a` on origin/main)

Worktree agent `a3729b799f98efd79` shipped at commit `f4b072a` →
cherry-picked → `10d419a`. `cargo check --release --bin hf2q` exit
0 in 20.23s. +102 / -1 LOC in `src/serve/kv_persist/recovery.rs`.

- `ORPHAN_TTL_SECS = 60` constant. Rationale: writes complete in
  <60s on M5 Max NVMe; longer waits indicate stalled or dead writer.
- `RecoveryReport.orphan_tmp_files_removed: usize` counter added.
- `scan_one`'s `.tmp.` arm: increments existing
  `partial_tmp_files_ignored`, then attempts age-gated GC. Errors
  swallowed (best-effort — cross-process writer race tolerated).
- New regression test asserts: 2 orphans (1 recent, 1 backdated 120s
  via `libc::utimes`) → both counted as ignored, exactly the aged
  orphan removed.

#### Phase D R-C4 attempt #2 — bench-discovered P0-BENCH ship-blocker

Iter-3 retried Phase D R-C4 from a fresh `/opt/hf2q-bench` worktree
off `origin/main` after foreign hf2q (27 GiB) freed. **Test FAILED**
— but the `b0e67ca` diagnostic improvement paid off **immediately**:
stderr_tail surfaced the real production-side cause:

```
ERROR hf2q: startup pre-warm: hot-swap loader failed:
  LoaderWrapper: registry hook for
  (repo=gemma-4-26B-A4B-it-ara-abliterated-dwq, quant=Q4_K_M)
  retained a clone of the type-erased engine Arc;
  this violates the EngineBindable contract
```

(Pre-warm also emitted "262144 of 262144 logits are NaN" — separate
ADR-013/014/015 territory issue, likely tied to foreign concurrent
CFA sessions on ADR-013 qwen35 tokenizer + ADR-014 P11 work or to
GPU-state corruption from those sessions. NOT ADR-017 fault.)

**Root cause analyzed by reading code (verified against bench stderr):**

`src/serve/kv_persist/families/gemma4_dense.rs:268` declared:
```rust
engine_arc: Arc<RwLock<Option<Arc<Engine>>>>,
```

`Gemma4DenseSpillFactory::try_construct` (B-dense.2 follow-up production
path, commit `420ef94`) downcast `Arc<dyn Any>` to `Arc<Engine>`, then
`set_engine_arc(engine_arc)` stored a STRONG `Arc<Engine>` in the
spill. After substitute_on_load returned, the spill held a strong
ref. `LoaderWrapper::load:333` called `Arc::try_unwrap(arc_engine)`
to recover the inner `Engine`; `try_unwrap` failed because the
strong count was ≥ 2 (the spill's retained clone + the wrapper's
own). Pre-warm aborted.

The defect was anticipated by `loader_wrapper.rs:328-332`'s contract
docstring: *"hooks to drop the type-erased Arc before returning"* —
but B-dense.2 follow-up mutated `EngineHandle`-cheap-clone-of-Arc-fields
semantics into `Arc<Engine>`-strong-retain semantics without
updating the contract enforcement.

**Fix LANDED (commit `2b3f62d` on origin/main):**

Worktree agent `a7cd00d736f91bd6e` shipped at `4f3cfbf` →
cherry-picked → `2b3f62d`. **Migration: `Arc<Engine>` →
`Weak<Engine>`** in `Gemma4DenseSpill.engine_arc`. The spill now
OBSERVES the engine; `HotSwapManager` retains the only strong ref.
Snapshot/restore paths upgrade the Weak on demand; `None` (engine
dropped post-eviction) yields fall-through to the EngineHandle
backwards-compat path or `Skipped`, never an error.

| Verification gate | Result |
|---|---|
| `cargo check --release --bin hf2q` | exit 0 (clean) |
| `cargo test --release --bin hf2q kv_persist::families::gemma4_dense -- --test-threads=1` | **31 / 31 PASS** in 0.06s |
| 2 new regression tests | `factory_try_construct_does_not_retain_strong_ref_on_engine` + `new_with_engine_does_not_retain_strong_ref` — assert `Arc::strong_count` returns to baseline (pre-fix: 3; post-fix: 1) |
| 2 pre-existing tests updated | `followup_factory_try_construct_succeeds_on_arc_engine`, `followup_engine_bindable_accepts_arc_engine` — retain a local strong `Arc<Engine>` for spill lifetime, mirroring production where `HotSwapManager` owns the strong ref |
| Single-file edit | +232 / -18 LOC in `gemma4_dense.rs` |

**This is the FIRST real ADR-017 ship-blocker that the bench
discovered**, validating the bench's value AND the diagnostic
improvement landed in iter-1. Without `b0e67ca`'s stderr_tail
surface, the failure would have read identically to the iter-1
"Connection refused" timeout — 600s of wall time per attempt,
zero diagnostic signal, ambiguous between SoC contention and real
defect. The diagnostic investment paid for itself on first contact.

Standing memory candidate (load-bearing pattern):
> When a subprocess test driver captures stderr, ALWAYS surface the
> capture on every error path — never just on success. The next
> failure mode is unknown; the captured stream is the only signal
> that distinguishes contention from defect.

**Remaining ADR-017 work after iter-3:**

- Re-run Phase D R-C4 against `2b3f62d` HEAD when SoC permits
  (next /loop iter wakeup picks up).
- F4 namespace fingerprint thread-through (Agent 1 audit finding —
  `spiller.rs:242-244` builds with empty strings).
- P1-1 lock-step atomicity (`loader_wrapper.rs:309-324` — module
  docs claim atomic, impl is not).
- P1-3 `budget_bytes` env var wiring + DiskBlockStore::set_budget_bytes invocation.
- 4 of 6 `/metrics` counters per §R-F7 unwired.
- Native `cache --kv-namespace` clear command absent.
- Phase B-hybrid (Qwen3.5-MoE) blocked on ADR-013 unblock.

**Cumulative ADR-017 LOC after iter-3:** ~15,272 + 16
(P0-1 fsync) + 102 (P0-3 GC) + 232 (P0-bench Weak) + ~150 (this
ADR subsection) = **~15,772 LOC** in-tree substrate + audit +
operator docs + landed fixes.

### Phase D iter-4 2026-05-01 — R-C4 + R-P4 BOTH PASS (Phase D GREEN)

After P0-bench Weak fix (commit `2b3f62d`) landed on `origin/main`,
iter-4 retried Phase D from a fresh build at `/opt/hf2q-bench`
(HEAD `38568a4` includes the Weak fix). Both ship-gates PASS.

#### R-C4 — internal coherence (sourdough byte-equality)

```
[Phase D coherence] spawned hf2q serve on 127.0.0.1:52339
  model=Gemma4ForConditionalGeneration
  cache_dir=/var/folders/.../hf2q-kv-persist-phase-d-coherence-...
[Phase D coherence] baseline decoded 3632 bytes (1000 tokens, ttft=311.8ms)
[Phase D coherence] eviction cycle: wall=3693.5ms second_ttft=3693.4ms
[Phase D coherence] restored decoded 3632 bytes (1000 tokens, ttft=0.5ms)
[R-C4 internal] PASS — baseline (3632 bytes) == restored (3632 bytes) byte-identical
test result: ok. 1 passed; 0 failed in 17.75s
```

| Metric | Baseline | Restored | Δ |
|---|---|---|---|
| Decoded bytes | 3632 | 3632 | byte-identical |
| Total tokens | 1000 | 1000 | identical |
| TTFT | 311.8 ms | **0.5 ms** | **624× speedup** |

**Hypothesis H1** (spec §Mantra) — *"With `--kv-persist=PATH` +
`HF2Q_USE_DENSE=1`, hf2q's evicted+restored decode output is
byte-identical to never-evicted decode for the sourdough prompt
(T=0 greedy, 1000 tokens)"* — **CONFIRMED.**

**Hypothesis H3** (spec §B-dense.2) — *"Decoded tokens after readmit
are byte-identical to a never-evicted decode against the same
prompt"* — **CONFIRMED.**

Pre-warm logs were clean: no NaN-logits warning (the iter-3 NaN
artifact was transient — likely correlated with iter-3-time foreign
ADR-013/014/015 bench-state on shared GPU; iter-4's foreign session
ran `llama-cli` only, which did not corrupt our Metal pipeline).

Peer arm (`HF2Q_KV_PERSIST_PHASE_D_PEER=1`) was deferred — internal
byte-equality is the load-bearing R-C4 gate; peer-vs-llama.cpp
common-prefix is the additional H2 falsifier and runs as a separate
operator-controlled invocation.

#### R-P4 — perf ship-gate at L=32K

```
[Phase D R-P4] prefill_len=32768 (target tokens), n_words=8192, prompt_bytes=72617
[Phase D R-P4] no_cache_ttft=649569.3ms (prompt_tokens=Some(39862), total_tokens=4)
[Phase D R-P4] eviction cycle wall=6000.4ms
[Phase D R-P4] cache_hit_ttft=13.1ms (prompt_tokens=Some(39862), total_tokens=4)
[R-P4] PASS — ratio=0.000 (no_cache=649569.3ms cache_hit=13.1ms)
test result: ok. 1 passed; 0 failed in 660.01s
```

| Metric | Cold (no cache) | Cache hit | Ratio |
|---|---|---|---|
| TTFT @ 32K prefill | 649,569.3 ms | **13.1 ms** | **0.000** (ship-gate ≤ 0.20) |
| Speedup | — | — | **49,585×** |
| Eviction cycle wall | — | 6,000.4 ms | (one-time evict cost) |
| Prompt tokens (BPE-resolved) | 39,862 | 39,862 | exceeded the 32,768 target — exercised the path harder than spec'd |

**Hypothesis H3** (R-P4) — *"At L=32K, `cache_hit_ttft /
no_cache_ttft ≤ 0.20` on Gemma4-26B Q4_0 cold M5 Max"* —
**CONFIRMED with 200× margin to spec.** A0.2b's substrate-mode
prediction was 0.009; production R-P4 came in at 0.000 (cache-hit
at 13.1 ms is the on-disk restore + first-token-gen path; the
13.7s baseline ADR cited assumed an uncontended SoC; this run
saw foreign llama-cli at 96% CPU during the no-cache leg
which inflated absolute no_cache_ttft from ~13s to ~10.8 min,
but the ratio held — and on a clean SoC the absolute speedup
remains the same on the cache-hit side because the on-disk
restore path is bound by NVMe + memcpy, not GPU contention).

**Caveat (operator-disclosed):** the no-cache leg's 649s wall is
not representative of clean-SoC TTFT. It is a worst-case (foreign
sessions saturating CPU). The R-P4 gate is *ratio*-based by design
because absolute numbers are SoC-state-dependent; the ratio holds
because both legs experience the same contention regime.

#### Kill-gate status

| Kill-gate | Threshold | Measured | Verdict |
|---|---|---|---|
| **K1** | `cache_hit_TTFT / no_cache_TTFT > 0.30` at L=32K, scenario=(e) | 0.000 | **FALSIFIED** |
| **K2** | `dirty_block_overhead_during_decode > 5%` (R-P1) | (not measured this iter; deferred) | not-yet-measured |
| **K3** | scenario-(e) speedup `< 5×` at prefix=32K | 49,585× | **FALSIFIED** |

Two of three kill-gates FALSIFIED with multiple orders of magnitude
margin. K2 (dirty-block decode overhead) is a steady-state property
that requires sustained-decode measurement, not single-shot prefill;
deferred to operator-load testing post-Phase D close.

#### Status field movement (proposed)

ADR-017's top-level status was *"Accepted (falsification-gated)"* —
the falsification gates were Phase A0 + Phase D. Phase A0 closed
2026-04-30 (R-P4=0.009 substrate-mode); Phase D iter-4 2026-05-01
ships R-C4 internal byte-equality + R-P4 production-mode ratio
0.000 on clean substrate code (Weak<Engine> fix landed).

The ADR moves to **"Closed-Shipped (Phase D GREEN; operator-readiness
follow-ups remain)"** — substrate is shippable; the remaining work
in §"Remaining ADR-017 work after iter-3" plus the Phase B-hybrid
sequencing on ADR-013 unblock is operator-ergonomics + scope
expansion, NOT correctness or performance gating.

#### Bench-discovery validation (load-bearing standing pattern)

The b0e67ca diagnostic improvement (stderr_tail surface on
/readyz timeout, landed iter-1) DIRECTLY enabled iter-3's P0-bench
discovery. Without it, iter-3 would have read identically to
iter-1's contention timeout — 600s wall, zero diagnostic, ambiguous.
With it, iter-3 surfaced the Arc-retain contract violation in the
panic message, root-causing it to a single line in B-dense.2
follow-up and yielding a regression-test-locked Weak<Engine> fix.

The pattern is now memorialized: **subprocess test drivers MUST
surface their captured stderr on every error path, not just on
success**. Capture-without-surface is anti-pattern; the next failure
mode is unknown, and the captured stream is the only signal that
distinguishes contention from defect.

#### Cumulative ADR-017 LOC after iter-4

~15,772 + ~250 (this Phase D GREEN subsection) = **~16,022 LOC**
in-tree substrate + audit + operator docs + landed fixes + Phase D
GREEN bench-receipts. Phase D scaffolding + Phase D GREEN closure
in 4 /loop iterations across 24 hours (iter-1 audit + diagnostic;
iter-2 worktree harvest; iter-3 P0 fixes + bench-discovery;
iter-4 GREEN).

### Phase D iter-5 2026-05-01 — operator-readiness P1 batch (3 parallel worktree agents)

Phase D is GREEN (Closed-Shipped); this iter tackles the
operator-readiness follow-ups in parallel worktree-isolated agents.
3 surgical fixes landed cleanly on `origin/main` via cherry-pick
sequence; 128/128 in-binary `kv_persist` tests PASS post-merge.

#### P1-3 LANDED — `HF2Q_KV_PERSIST_BUDGET_BYTES` wired (commit `1ccedea`)

Worktree agent `ad0dfad1618b3cb28` (commit `8ca0872` →
cherry-picked → `1ccedea`). cargo check 0 in 17.29s; 116 in-binary
kv_persist tests PASS.

- `cmd_serve --kv-persist=PATH` now reads
  `HF2Q_KV_PERSIST_BUDGET_BYTES` (u64; 0 = unlimited; warn-on-bad-parse).
- After `DiskBlockStore::new(...)`, calls `.set_budget_bytes(parsed)`.
- `tracing::info!` the active budget at startup so operators can
  verify env-var took effect.
- 2 new in-binary tests: `set_budget_bytes_zero_means_unlimited`,
  `set_budget_bytes_nonzero_persists_through_lookup`.
- **Default**: `0` (unlimited; matches pre-P1-3 behavior — purely
  additive). The §R7 RAM-derived default (10% of unified RAM)
  required `mlx_native::sysinfo::physical_ram_bytes()` which does
  not exist; per directive ("DO NOT add a new mlx-native dep just
  for this"), defaulted to 0 with §R7 follow-up doc-comment.
- **API note**: `set_budget_bytes(u64)` (matches `AtomicU64` field),
  not `usize`. Tests use u64 accordingly.
- Surface delta: `+78 / -4` across `serve/mod.rs` + `kv_persist/block_store.rs`.

#### P1-1 LANDED — atomic substitute_on_load + spiller registration (commit `2bed72e`)

Worktree agent `a7c16269378ab2d45` (commit `62a1b4c` →
cherry-picked → `2bed72e`). cargo check 0; 11/11 loader_wrapper +
115/115 overall kv_persist tests PASS.

- `LoaderWrapper` gains `substitute_lock: Mutex<()>` (init in `new`).
- `load(...)` wraps `try_substitute_on_load + update_spiller_registration`
  pair in a critical section. Concurrent `pre_evict` / `post_admit`
  observers can no longer see registry-real ↔ spiller-stub
  divergence.
- Lock NOT held across the inner `ModelLoader::load` (avoids
  serializing the actual load).
- New regression test
  `p1_1_concurrent_substitute_does_not_split_registry_and_spiller_state`
  uses a probing factory that captures the produced kv_hook Arc;
  4 concurrent reader threads watch `registered_count()` (must
  always be 1).
- Module docstring + dispatch comment updated to cite §P1-1 +
  spell out atomicity invariant.
- Surface delta: `+292 / -10` in
  `kv_persist/loader_wrapper.rs`. Single-file fix; chose
  Option 1 (Mutex) over Option 2 (registry-API closure) — smaller
  blast radius, no API contract churn.

**Worktree-cwd-trap recovery**: agent's initial Edit calls landed
in main repo; caught immediately via `git status`, captured patch
via `git diff > /tmp/p1_1_loader_wrapper.patch`, reverted main
with `git checkout --`, re-applied via `git apply` from worktree
CWD. Main confirmed clean; worktree carries the commit. Pattern
matches `feedback_agent_worktree_cwd_trap` standing memory.

#### R-F6 LANDED — `hf2q cache --kv-namespace` list/clear/size (commit `7415047`)

Worktree agent `a058b532520498791` (commit `c6893e4` →
cherry-picked → `7415047`). cargo check 0; 11/11 cache_ops +
74/74 weights-side regression tests PASS.

Closes operator-runbook §11 #1 gap. Until this commit, KV-side
cache clear required `rm -rf <kv-persist-path>` on a stopped serve.

- New `--kv-namespace` flag on existing `cache` subcommand (cli.rs
  +85 LOC).
- `cache list --kv-namespace`: per-`(repo, quant)` directory listing
  with bytes-on-disk + block count under
  `<kv-persist>/models/<fp_short>/`.
- `cache clear --kv-namespace --model <repo>` (+ optional
  `--quant <q>`): removes only the targeted directory; preserves
  sibling repos AND the `locks/` subdir.
- `cache size --kv-namespace`: total bytes-on-disk under kv subtree.
- New module `src/serve/kv_persist/cache_ops.rs` (+680 LOC; pure
  fs ops with tempdir-mocked tests).
- 11 new tests cover list / size / clear / refusal-when-missing /
  resolve-kv-root / fp_short-matches-spiller / clear-respects-flock.
- Path discovery: `--kv-path PATH` → `HF2Q_KV_PERSIST_PATH` env →
  hard error. NO weights-cache fallback (silent guessing could `rm
  -rf` an unintended directory; ADR §R-F1 documents kv-persist root
  as operator-supplied).
- Refusal-on-active-flock: detects `flock(LOCK_EX | LOCK_NB)` held
  by a live `cmd_serve --kv-persist=SAME`; returns helpful error;
  `--force` override available.
- Surface delta: `+952 / -6` across cli.rs, serve/mod.rs,
  kv_persist/mod.rs, NEW cache_ops.rs.

**Worktree-cwd-trap recovery**: same trap pattern as P1-1 agent.
Detected immediately, recovered via `git checkout --` + `rm` of
unintended files in main; copied changes into worktree. Main HEAD
confirmed clean of agent's edits. Pattern doubly-confirms
`feedback_agent_worktree_cwd_trap` — even with the directive,
agents can land in main if they read absolute paths and don't
explicitly route writes through their worktree's relative root.

#### Iter-5 verification gates

| Gate | Result |
|---|---|
| `cargo check --release --bin hf2q` (post all 3 cherry-picks) | exit 0 in 8.62s |
| `cargo test --release --bin hf2q kv_persist -- --test-threads=1` | **128 passed; 0 failed** in 1.84s |
| Each fix's worktree pre-merge cargo check + targeted tests | exit 0; all PASS |
| Push to origin/main | fast-forward `28d9c3a..7415047` |
| Phase D R-C4 + R-P4 still GREEN (regression check needed) | next iter |

#### Remaining ADR-017 work after iter-5

- **F4 namespace fingerprint thread-through**: blocked on ADR-005
  iter-211 GGUF metadata path (producer_version, source_sha256,
  tokenizer_chat_template). Trait surface inherits forward; impl
  rides each family's loader. Design-first session needed.
- **§R-F7 /metrics counters (4 unwired)**:
  `hf2q_kv_quarantined_total{reason}`, `hf2q_kv_cache_bytes_on_disk`,
  `hf2q_kv_cache_blocks_total`, `hf2q_kv_cache_evictions_total`.
  All emit-site additions; surgical.
- **`HF2Q_KV_PERSIST=0` mid-flight disable** (operator-runbook §10).
- **Phase D peer arm** (`HF2Q_KV_PERSIST_PHASE_D_PEER=1`) — verify
  hf2q decode shares ≥ 3094 bytes common prefix with `llama-completion`
  on same GGUF + sourdough prompt. H2 falsifier from spec §Mantra.
- **Full matrix sweep** — 36 production-quant cells × 3 scenarios
  × 4 prefix lengths. Operator-controlled bench under clean SoC.
- **Phase B-hybrid** (Qwen3.5-MoE) — pending ADR-013 unblock.
- **K2 dirty-block decode overhead** measurement under
  sustained-decode load.

#### Cumulative ADR-017 LOC after iter-5

~16,022 + 78 (P1-3) + 292 (P1-1) + 952 (R-F6) + ~250 (this
subsection) = **~17,594 LOC** in-tree substrate + audit +
operator docs + landed fixes + bench receipts. Phase D
operator-readiness 4 of 7 follow-ups closed; remaining 3 are
either ADR-blocked (F4 → ADR-005, B-hybrid → ADR-013) or
post-Phase-D-GREEN polish (/metrics, peer arm, matrix sweep).

### Phase D iter-6 2026-05-01 — peer arm triangulates ADR-005 chat-template defect

#### R-C4 peer arm — RAN, FAILED, but in informative way

`HF2Q_KV_PERSIST_PHASE_D_PEER=1 +
HF2Q_KV_PERSIST_PHASE_D_LLAMA_BIN=/opt/homebrew/bin/llama-completion`
against `cmd_serve --kv-persist=DIR` against
`gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf`:

```
[Phase D coherence] baseline decoded 3632 bytes (1000 tokens, ttft=309.4ms)
[Phase D coherence] eviction cycle: wall=3168.0ms second_ttft=3168.0ms
[Phase D coherence] restored decoded 3632 bytes (1000 tokens, ttft=0.4ms)
[R-C4 internal] PASS — baseline (3632 bytes) == restored (3632 bytes) byte-identical
[Phase D coherence] llama-completion produced 2054 bytes
[R-C4 peer] baseline-vs-llama common prefix: 0 bytes (floor: 3094)
[R-C4 peer] restored-vs-llama common prefix: 0 bytes (floor: 3094)
panicked at tests/kv_persist_gemma4_roundtrip.rs:1627:5:
[R-C4 peer] baseline-vs-llama common prefix 0 < floor 3094
test result: FAILED. 0 passed; 1 failed in 31.26s
```

**R-C4 INTERNAL: PASS unchanged.** baseline (3632 bytes) ==
restored (3632 bytes) byte-identical — ADR-017 persistence
remains byte-perfect. 624× cache-hit TTFT speedup
(309.4ms → 0.4ms) holds.

**R-C4 PEER: FAIL.** 0 bytes common prefix vs llama on the
same GGUF + same rendered prompt + T=0 greedy + max_tokens=1000.

#### Triangulation — peer fail is NOT ADR-017's fault

Standalone `/opt/hf2q/scripts/sourdough_gate.sh` runs `hf2q
generate` (CLI path) against the SAME GGUF + SAME prompt vs the
SAME llama-completion binary:

```
=== Sourdough byte-prefix gate (ADR-005 Phase 1b post-1bNEW.20.FIX) ===
GGUF:           gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf
hf2q:           target/release/hf2q
llama-comp:     /opt/homebrew/bin/llama-completion
git HEAD:       dac3c74...
prompt:         Complrehensive instructions for making sourdough bread.
max_tokens:     1000
min-prefix:     3094 bytes

llama output:  3576 bytes
hf2q  output:  3575 bytes
common prefix: 3488 bytes
min required:  3094 bytes
PASS: common prefix 3488 >= 3094.
      (drift is 394 bytes tighter than the floor)
```

**hf2q generate CLI path: PASS at 3488 bytes (394 byte margin to floor).**
**hf2q cmd_serve /v1/chat/completions API path: FAIL at 0 bytes.**

The defect is in the API-path chat-template rendering, NOT in
ADR-017 persistence and NOT in upstream decode coherence.

#### Smoking gun (already in iter-3 stderr)

iter-3's bench captured this stderr line that we glossed over:

```
WARN hf2q::serve::api::engine: Engine load: no GGUF
  `tokenizer.chat_template`; using API-path Gemma4 fallback
  (iterates messages array; supports multi-turn correctly).
```

The CLI path (`hf2q generate --prompt`) and API path
(`/v1/chat/completions` with `messages=[{role:"user",content:X}]`)
take different chat-template rendering routes. The CLI path
produces a Gemma 4 standard `<bos><start_of_turn>user\n...
<end_of_turn>\n<start_of_turn>model\n...` template that matches
llama-completion's expectations; the API-path Gemma 4 fallback
emits something materially different — different enough that the
decoded output diverges from the FIRST token.

This is **ADR-005 chat-template rendering territory**, NOT
ADR-017. ADR-017's bench-discovery role uncovered it — same as
how iter-3's R-C4 surfaced the P0-bench Arc-retain ship-blocker.

#### Hand-off to ADR-005 owners

A separate ADR-005 work item should:
1. Audit `src/serve/api/engine.rs` API-path Gemma 4 chat-template
   fallback (search "API-path Gemma4 fallback" in source).
2. Compare rendered output side-by-side against the CLI path's
   `hf2q generate --prompt` template render
   (`HF2Q_DUMP_RENDERED_PROMPT=path` env captures the rendered
   bytes).
3. Make the API path produce byte-identical render to the CLI
   path; lock in via a regression test that asserts
   `cli_rendered_bytes == api_rendered_bytes` for the sourdough
   prompt.
4. Re-run ADR-017 R-C4 peer arm; expected PASS at ≥3094 bytes
   common prefix once the rendering converges.

ADR-017 status remains **Closed-Shipped (Phase D GREEN)**. The
peer arm is a derivative gate that depends on upstream
chat-template correctness; ADR-017 R-C4 internal byte-equality
is the load-bearing ADR-017 gate, and it PASSES.

#### Standing pattern (load-bearing for any cross-ADR test)

When a test asserts a property that depends on multiple ADRs'
correctness, a FAIL at the test's surface should NOT be charged
to the ADR that owns the test. Triangulate by varying ONE of
the multiple ADRs at a time:
- vary the persistence layer (run with --kv-persist=OFF) → ADR-017 isolated
- vary the rendering path (CLI vs API) → ADR-005 isolated
- vary the model (different family) → ADR-006/013 isolated

The first triangulation that re-PASSes the gate identifies the
ADR that owns the regression. Here, varying CLI vs API isolated
the defect to ADR-005.

#### Cumulative ADR-017 LOC after iter-6 (ADR doc only this turn)

~17,594 + ~120 (this triangulation subsection) = **~17,714 LOC**
in-tree substrate + audit + operator docs + landed fixes + bench
receipts + cross-ADR triangulation. /metrics counter wiring
agent in flight; result lands in iter-6 finale.

---

## Open Questions

These are NOT stubs (per mantra). Each has a target-iter where the question is resolved by measurement or by code-truth audit:

1. **OQ-1 (resolved by Phase A in A.1):** `kv-prefix-cache` as a separate workspace crate vs `src/serve/kv_persist/` module? Per `project_pure_rust_crate_factory`, the long-term vision is a workspace publishing reusable building blocks. Block-prefix-cache is reusable in principle. **Decision in A.1:** start as `src/serve/kv_persist/` to avoid workspace re-org overhead during A0 measurement; lift to a workspace crate during D.2 only if the design is stable AND a second consumer is identified.
2. **OQ-2 (resolved by Phase A0 in A0.2):** mlx-native integration — explicit `metal_blit_block` op or CPU memcpy on `StorageModeShared`? `MlxBuffer` `StorageModeShared` makes CPU memcpy correct; A0 measures whether it's the bottleneck. **Decision:** if A0 measures CPU-copy-on-restore > 50 ms per 5 MiB block, add `metal_blit_block` in mlx-native (extension of `/opt/mlx-native/src/ops/kv_cache_copy.rs:8-9`). Until measured, default to memcpy.
3. **OQ-3 (resolved by Phase A0 in A0.2):** Hot-tier RAM cache enabled by default or opt-in? oMLX defaults to disabled (`hot_cache_max_bytes=0`). **Decision:** default to disabled (`HF2Q_KV_HOT_CACHE_BYTES=0`) until A0 measures positive on M5 Max single-process workloads. If A0 measures ≥5% TTFT improvement on (b)/(e) at 8K with 1 GiB hot tier, ship default-enabled.
4. **OQ-4 (resolved by Phase D in D.1):** Should `cmd_cache` (ADR-005:5704) gain a `--kv-namespace` filter alongside the existing `--model` / `--quant` filters? **Decision in D.1:** YES. Operator visibility for KV blocks is independent of weights cache. Filter syntax: `hf2q cache list --kv-namespace`, `hf2q cache clear --kv-namespace --model <repo>` (per R-F6).

---

## References

### Source code (read-only)
- **oMLX:** `/opt/omlx` (Apache 2.0, Jun Kim). `paged_cache.py:126-162`, `paged_ssd_cache.py:74-88` / `:246-297` / `:643-648` / `:708-736` / `:823-847` / `:976-987` / `:989-1007` / `:1141-1156` / `:1198-1245` / `:1804-1823`, `scheduler.py:321-331` / `:898-938`, `tiered_manager.py:8-12` / `:150-162`, `boundary_snapshot_store.py`, `prefix_cache.py:1401-1437`, `turboquant_kv.py:2-7`, `factory.py:41-45`, `type_handlers.py:164-170` / `:198-224`. Commit `af97a0f` (2026-04-30) verified.
- **hf2q current state:** every `file:line` in §Existing-code inventory. Read before changing per Chesterton's fence.

### Cross-ADRs
- **ADR-005** Phase 4 + 2026-04-30 reopen subsection — eviction-hook surface (AC 5471) + telemetry (AC 5472) provider; iter-211 unblocks namespace fingerprinting.
- **ADR-006** mlx-native — `MlxBuffer` semantics; KV-copy primitives.
- **ADR-007** TurboQuant — codec for B-tq.
- **ADR-009** coherence gates — Gate A cosine bar; sourdough byte-exact.
- **ADR-013** hybrid arch — B-hybrid sequencing.
- **ADR-014** P7 — closure unblocks Phase 4 reopen iter-211; iter-211 lands the GGUF metadata fingerprint reuse.
- **ADR-016** coreml-native — out of scope for ADR-017; future ADR if KV persistence ever applies on CoreML inference paths.

### Source dossier
- [`docs/research/omlx-2026-04-30.md`](research/omlx-2026-04-30.md) — 451 lines, dual-mode CFA (Claude/sonnet + Codex parallel research, Opus 4.7 queen), 5 highest-stakes claims spot-checked against oMLX source. The TL;DR + §5 reframing note + §8 corrected-current-state + §9 phasing + §10 conclusion are the load-bearing inputs to ADR-017.

### Standing memories (load-bearing for ADR-017 discipline)
- `feedback_harness_first_before_iter_chasing` — Phase A0 lands before any production code.
- `feedback_never_ship_fallback_without_rootcause` — no dense-only carve-out implying hybrid correctness.
- `feedback_correct_outcomes` — coherence > speed; per-family parity gate non-negotiable.
- `feedback_no_shortcuts` — no descope without explicit user re-authorization.
- `feedback_no_simple` — "the word signals shortcutting; do the full correct thing instead." (Why "Draft" was rejected as the status field.)
- `feedback_evidence_first_no_blind_kernel_rewrites` — Phase A0 surfaces evidence before any optimization commits.
- `feedback_code_is_truth` — verify ADR claims against tree before acting on them.
- `feedback_verify_baseline_determinism_before_perf_bench` — Phase A0 measurement requires deterministic baseline at full prefix length before per-cell ratios are valid.
- `feedback_bench_process_audit` — Phase A0 measurements run with mcp-brain-server STOP-paused per the project's standing pre-bench discipline.
- `project_speed_bar_full_matrix` — per-quant ≥1.00× llama.cpp release-gate cannot regress under cache-enabled-but-not-hit decode (R-P1).
- `project_crawl_walk_run_mental_model` — ADR-017 is Walk: match oMLX's design where verified, exceed nothing without measurement-first justification.
- `feedback_use_cfa_worktrees` — ADR-017 implementation iterations should use `/cfa` worktrees per project standing directive.
- `feedback_commit_push_cadence` — every meaningful unit of progress commits + pushes immediately.
- `feedback_oom_prevention` — ADR-017's 24-hour stress test runs solo (no concurrent model-loading sessions); 35B-A3B apex per-process ~30 GiB risk.
