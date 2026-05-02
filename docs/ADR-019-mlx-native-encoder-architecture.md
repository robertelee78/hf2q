# ADR-019: mlx-native Encoder Architecture — Per-Stage Fence Design (D3 PRIMARY, D4 deferred-pending-Phase-0a-microprototype)

**Status:** Proposed v1.1 (post-Codex Phase-2b review, 2026-05-03). D3 is the PRIMARY architectural target subject to Phase 0a evidence-based gates (xctrace residual attribution + D4 microprototype falsification). D2 (CPU dispatch_apply) REJECTED. D4 (MLX-style watermark) DEFERRED until Phase 0a.2 microprototype evidence; if D4 microprototype beats D3-projection on FA-path slice, ADR-019 is rewritten with D4 as primary.



- **Status:** Proposed (2026-05-03)
- **Authors:** Robert E. Lee + Claude Code (CFA session `adr019-parallel-encode`, opus 4.7 1M-context architect over 3 parallel research artifacts: researcher-llama, researcher-mlx, deep-researcher)
- **Predecessor:** ADR-015 §"Resume Here 2026-05-03 ~00:00Z" — gap mechanism localized to command-buffer-count asymmetry; iter89e committed (in flight) to prepare ADR-019 design substrate
- **Companion ADRs:**
  - **ADR-013** — Qwen3.5/3.6 inference; owns the `forward_gpu.rs` / `gpu_full_attn.rs` / `gpu_delta_net.rs` callsites that ADR-019 retargets. P21 K=8 batching (sync_count 161 → 6) is the substrate this ADR builds on.
  - **ADR-015** — mlx-native general decode/prefill perf; iter88a-89e measurement chain produced the evidence that authorizes this ADR.
  - **ADR-006** — mlx-native crate; ADR-019 modifies `encoder.rs` / `device.rs` / `graph.rs` indirectly via the `EncoderSession` abstraction, but mlx-native production semantics stay backwards-compatible.
  - **ADR-017** — KV-persistence; uses ADR-005 hot-swap surface but is otherwise orthogonal. ADR-019 changes nothing in ADR-017's path.
- **Standing requirement (load-bearing):** "as fast as our peers" applies to **every shipped model family** at every prefill/decode regime. Same-day same-hardware llama-bench measures peer at 3322 t/s pp4096 / 3249 t/s pp4127; current hf2q 1849 ms vs peer 1233 ms = **0.667× chunk-engaged ratio (+50% gap)**. ADR-019 is the architectural lever for closing the encoder/orchestration component of that gap. (Robert directive 2026-04-30: *"Rust is not slower than C++ on the same hardware; any gap is structural"*.)

---

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim. This is the discipline this ADR — and every iteration, every commit, every decision under it — must be executed against. It supersedes any tactical convenience that conflicts with it.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

**Operational reading for ADR-019:**

- **DO NOT BE LAZY / no short cuts.** D3 is a multi-week implementation effort. Each phase has its own parity gate; no phase ships without 4-fixture decode parity green and per-CB sync_count receipts.
- **Plenty of time.** The estimate is 4-7 weeks single-developer (or 2-4 weeks /cfa dual-mode). If chunk-internal MTLEvent conversion proves harder than Phase 4 anticipates, we extend rather than ship a partial path that re-opens the iter58b residency-rescission class.
- **Never make assumptions.** Every Chesterton's fence below cites file:line. Every acceptance threshold is concrete. Every rejected option (D2, D4) cites empirical or peer-measurement counter-evidence, not aesthetic preference.
- **Dive deep / use search as needed.** Researcher-llama walked `ggml-metal-context.m:438-615` line-by-line; researcher-mlx enumerated all 96+ commit sites in qwen35 by file:line; deep-researcher cross-checked Apple's verbatim Best Practices Guide + MLX PR #1864 + Metal 4 WWDC 2025 against hf2q's own iter88a-89e measurements. ADR-019 leans on that work; the architect did not re-do it.
- **Measure 3x, cut once.** ADR-013 P21 was 6 stages × measurements per stage. ADR-019 inherits the same discipline: per-phase microbench (iter88a regression check), per-fixture decode parity, per-CB sync_count + cmd_buf_count receipts. No production code lands without numbers.
- **No fallback / no stub.** No `// TODO: chunk-internal MTLEvent later`. Either Phase 3 lands the chunk-internal conversion green (with a parity audit covering the iter58b residency-rescission contract), or Phase 3 holds at "deferred to follow-up CFA cycle pending arena-lifetime work" with a dated exit condition. There is no env-gated half-shipped state-machine.
- **Pure excellence, done right.** Coherence > speed. A 100 ms wall-time win that flips one of 4 fixtures off-IDENTICAL at temp=0 is a regression, not progress. Decode parity fixtures (currently 1.07× peer) cannot regress; if D3 prefill ships at the cost of decode parity, D3 holds.
- **Chesterton's fence.** Every commit site that **stays** in D3 is documented in §6 with WHY (residency-rescission class / arena-reset-quota class / CPU-read-hazard class). Every commit site that **moves** in D3 is documented with what hazard moves with it (MTLEvent intra-CB or stage-fence cross-CB).

---

## Status of Predecessor Work (post-ADR-013 P21, 2026-05-02)

This ADR is authorized by the convergent finding of three independent measurement chains, all completed within the 72 hours preceding 2026-05-03.

### ADR-013 P21 closed the easy half of the gap (sync-count batching)

| Stage | Commit | Mechanism | sync_count | pp80 t/s |
|---|---|---|---:|---:|
| baseline (P19 H9) | — | per-layer commit_and_wait | 161 | 199 |
| Stage 1 | (FaPrefillArena terminal) | FA bridge → commit_labeled | 141 | — |
| Stage 3a | `c3f35d4` | DN ops1-3 / qkv_split / ops5-9 → commit_labeled | 51 | — |
| Stage 3b | `1ecfa7b` | K=4 FFN-terminal batching | 21 | — |
| Stage 2 | `2ee0ffc` | GPU-side KV cache write + ops1-4 commit_labeled | 11 | — |
| Stage 2c | `9cfca06` | K=8 FFN-terminal batching | 6 | — |
| Stage 4 | `0847f56` | Autoregressive GDN simd_sum kernel | 6 | 582 |
| **Merged** | `524af1e` | All stages on `main` | **6** | **582** |

**Wall-time arc:** pp80 199 → 582 t/s = **2.92× speedup**; pp726 ~200 → 1968 t/s = **~10×**; decode 105 → 116 t/s = **1.07× peer parity** (4/4 fixtures byte-identical). 96% sync-count reduction. Source: `project_adr013_p21_stage4_canonical_2026_05_02.md` MEMORY note + `docs/ADR-013-qwen35-inference.md:2918-2932`.

### ADR-015 iter88a-89e localized the residual gap to command-buffer count

iter88a (`/tmp/cfa-iter88a/COMPARISON.md`) ran a per-kernel peer comparison harness and found:
- **mlx-native MoE FFN gate/up: 1.16-1.20× FASTER than llama.cpp at production batches.**
- **Only MoE FFN-down lags 0.76-0.86× of peer = 86 ms gap** (14% of the 616 ms wall gap).
- **The OTHER 530 ms (86%) is in encoder/orchestration/CB residency, NOT kernel speed.**

Subsequent iters falsified four candidate mechanisms for the 530 ms residual:
| iter | Hypothesis | Verdict |
|---|---|---|
| iter89a | mm_id routing threshold | mm_id route is correct |
| iter89b | `kernel_mul_mm_id_q4_0_f32` audit at production shape | byte-equivalent ISA per peer |
| iter89c | dense-q microbench mirror | confirms kernel parity |
| iter89d | xctrace 577 ms residual | byte-identical metallib + byte-equivalent ISA |
| iter89e | GDN microbench | kernel speed is **not** the mechanism |

**Convergent finding (load-bearing):** the +50% chunk-engaged peer gap is COMMAND BUFFER COUNT ASYMMETRY. mlx-native uses 100+ CBs per chunk-engaged prefill via per-layer `commit_labeled` / `commit_and_wait_labeled` sites; llama.cpp uses `dispatch_apply(n_cb, encode_async)` to encode the entire graph into n_cb+1 ∈ {2, 3, 5, 9} CBs. Each MTLCommandBuffer creation+commit has measurable CPU + driver-round-trip overhead; 100× more CBs = order-of-magnitude more orchestration cost, plus GPU pipeline stalls at every `commit_and_wait_labeled` boundary.

### Operator decision authorized 2026-05-03

ADR-015 explicitly tagged its iter89e Resume-Here entry with: *"multi-week architectural changes. Operator decision required: continue ADR-015 architectural OR pivot to ADR-013 P14 / ADR-017 / ADR-005 amortization-class wins."*

The operator authorized continuation: ADR-019 is that continuation, formalized as a discrete ADR rather than as further ADR-015 iterations because (a) the work is large enough to need its own decision record, (b) the design space spans 4 candidates that need to be evaluated jointly rather than serially, and (c) the implementation will touch encoder.rs / device.rs / forward_gpu.rs / gpu_delta_net.rs / gpu_full_attn.rs and needs to be sequenced as a coherent unit rather than as iter-by-iter accretion.

---

## Context

### What "ADR-019 parallel encode" actually means

The framing "parallel encode" is borrowed from llama.cpp's `dispatch_apply(n_cb, encode_async)` mechanism (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:550`). Researcher-llama's literal walkthrough (research artifact §C, lines 152-217) shows that the "parallelism" llama.cpp achieves is **CPU-side encode parallelism on GCD worker threads**, not GPU execution parallelism. The CBs themselves still execute serially on the Metal command queue (per `[cmd_buf enqueue]` ordering at `ggml-metal-context.m:541-547`). What `dispatch_apply` saves is the CPU-side time spent calling `setComputePipelineState` / `setBuffer:offset:atIndex:` / `dispatchThreadgroups` ObjC msg_send chains across thousands of nodes — that work is amortized across GCD worker threads while the GPU pipelines previous CBs.

The deeper truth (research synthesis §3.1, §7.2): llama.cpp's own benchmark comment at `ggml-metal-context.m:458` reads *"tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2"*. **The encode-parallelism win is small on Apple Silicon because unified memory eliminates the PCIe amortization that justifies n_cb=4-8 on AMD dGPUs.** llama.cpp ships `n_cb=1` default on dGPU paths (`ggml-metal.cpp:608, 702`) and only escalates to `n_cb=2` when the graph is unusually large.

What llama.cpp **does** do that hf2q doesn't is consolidate the entire transformer forward pass into 2 CBs (main-thread prefix of `MAX(64, 0.1·n_nodes)`, plus n_cb=1 worker CB for the rest). hf2q's qwen35 chunk-engaged prefill uses **~100 CBs**. The 50× CB-count delta is the load-bearing asymmetry, and it persists whether the encoding happens on one thread or many.

**ADR-019 is therefore not primarily a parallel-encode ADR. It is a CB-count-consolidation ADR, with parallel-encode (D2) explicitly evaluated as one of four design candidates and rejected on the evidence below.**

### The +50% chunk-engaged gap (load-bearing measurement)

Same-day same-hardware llama-bench measurements (orchestrator session 2026-05-02 18:30Z, ADR-015 §"Resume Here ~18:30Z"):

| Fixture | hf2q | llama.cpp | Ratio | Gap | Source |
|---|---:|---:|---:|---:|---|
| pp4096 chunk-engaged | 1849 ms | 1233 ms | **0.667×** | +50% | iter88a / orchestrator llama-bench |
| pp4127 default-axis | 1510 ms | 1271 ms | **0.842×** | +18.8% | same |
| Decode (4 fixtures, K=8) | 116 t/s | 108 t/s | **1.07×** | parity ✓ | iter61c re-validation 2026-04-30 |

**Chunk-engaged gap decomposition (post-iter88a-89e attribution):**

| Component | Time | % of gap | Owner | Closable by ADR-019? |
|---|---:|---:|---|---|
| MoE FFN-down kernel speed (mlx-native) | 86 ms | 14.0% | mlx-native (out of scope for this ADR) | NO |
| Encoder/orchestration/CB-count residual | 530 ms | 86.0% | hf2q forward path + mlx-native CommandEncoder | YES (this ADR) |
| **Total wall gap** | **616 ms** | 100% | | |

**ADR-019's scope ceiling:** the 530 ms encoder/orchestration residual. ADR-019 does NOT address the 86 ms MoE FFN-down kernel-speed gap; that lives in mlx-native kernel-internal work (separate ADR territory; out of scope here).

### Why the current architecture is shaped this way (Chesterton's fence summary)

Per researcher-mlx §1: *"the architecture is an artifact of incremental ADR work — ADR-013 P11 (Qwen3.5 hot path), ADR-015 iter8e (residency-set flush hooks), ADR-015 iter13 (`MLX_UNRETAINED_REFS`), ADR-015 iter16 (label propagation), ADR-015 iter63 (per-dispatch sampling) — each iter added a feature to the CB-boundary lifecycle without revisiting whether more, smaller CBs was the right shape. There is no design ADR documenting 'we picked one-CB-per-component because X.'"*

The current shape has 12 explicit Chesterton's fences (researcher-mlx §5; reproduced in §6 below). The most load-bearing:
1. **F2 — iter58b residency-rescission** (commits `b6f416b → 2565eab`, 2026-04-30): wrapper-internal `device.alloc_buffer` scratches + bare `commit_labeled` + downstream `commit*` flushes the wrapper's deferred `removeAllocation:` while the wrapper's CB is mid-flight → garbage tensor values. This is the single most expensive class of regression in mlx-native history. Any encoder-architecture redesign must respect it.
2. **F3 — M5 Max residency-quota architectural ceiling**: dense-Q FFN's 5 pooled scratches × 33 dense layers ≈ 33 GB cumulative > Metal residency quota. K=8 is the conservative production floor; K=40 = "single arena reset at end of prefill" is INFEASIBLE.
3. **F7 — K-boundary scratch lifetime contract**: every K-boundary FFN-terminal commit MUST be `commit_and_wait`, NOT `commit_labeled`. This is the structural floor for sync_count at non-chunk pp80 = 6.

These three fences alone bound the design space: D1 (single-CB-everywhere) requires solving F2 end-to-end and partially relaxing F3 + F7; D3 (per-stage-fence) preserves all three with smaller per-stage windows.

---

## Current State (post-ADR-013 P21)

### CB lifecycle topology

Per researcher-mlx §F (lines 188-196):

> One `mlx_native::CommandEncoder` owns exactly one `MTLCommandBuffer`. The struct has `cmd_buf: CommandBuffer` allocated in `new_with_residency` (encoder.rs:660-684) via either `queue.new_command_buffer()` or `queue.new_command_buffer_with_unretained_references()` (env-gated by `MLX_UNRETAINED_REFS`). The `CMD_BUF_COUNT` atomic increments once per `CommandEncoder::new` (encoder.rs:669).
>
> Within one CommandEncoder, ONE persistent `MTLComputeCommandEncoder` is reused across all dispatches. `get_or_create_encoder` (encoder.rs:815-843) is the lazy creation site; it sets `MTLDispatchType::Concurrent` (or Serial under `HF2Q_FORCE_SERIAL_DISPATCH=1` iter61a-2 probe) and stashes a raw pointer. Subsequent dispatches see a non-null `active_encoder` and reuse it. `end_active_encoder` (encoder.rs:847-854) calls `[encoder endEncoding]` on commit.

### Sync count and CB count at HEAD

| Regime | sync_count | cmd_buf_count (approx) | Notes |
|---|---:|---:|---|
| Decode (autoregressive, seq=1) | 5-10 | 5-10 | gemma at 1-2 (the canonical correct shape) |
| Prefill pp80 (chunk path NOT engaged, K=8) | **6** | ~30-40 | 5 K-boundary FFN + 1 output-head |
| Prefill pp4096 (chunk-engaged) | **~96** | **~100+** | 30 chunk-prep + 30 chunk-attn + 30 chunk-ops8-9 + 5 K-boundary + 1 output-head |

**ADR-019's primary target is the chunk-engaged regime.** Decode is already at peer parity (1.07×); pp80 non-chunk is already at 6 syncs / structural floor.

### Production blocking site enumeration (chunk-engaged pp4096)

Per researcher-mlx §C, table at lines 99-117. Each is a `commit_and_wait_labeled` (BLOCKING; host waits for GPU):

| File:Line | Label | Hazard class | Removable in D3? |
|---|---|---|---|
| `gpu_delta_net.rs:2297` | (no label) chunk-prep prefill, non-arena | Cross-encoder boundary; chunk wrapper opens own enc | YES — convert to MTLEvent intra-stage |
| `gpu_delta_net.rs:2808` | (no label) chunk-prep prefill, arena | Cross-encoder boundary; identical | YES — convert to MTLEvent intra-stage |
| `gpu_delta_net.rs:1431` | `layer.gdn.chunk_attn` (non-arena) | iter58b residency-rescission | NO (non-prod fallback; arena variant at 1757 is prod) |
| `gpu_delta_net.rs:1757` | `layer.gdn.chunk_attn` (arena) | iter58b: chunk-pipeline kernel internal scratches | CONDITIONALLY — Phase 3 chunk-internal MTLEvent |
| `gpu_delta_net.rs:2904` | `layer.gdn.ops8-9` (chunk path, arena) | Residual handoff; conservative wait | YES — Phase 3 (mirror autoreg ops5-9 at line 3052 which is `commit_labeled`) |
| `forward_gpu.rs:2382` | `layer.dense_ffn` (K-boundary at K=8) | F7: arena-reset-quota | NO (preserved; D3 keeps K=8 boundary) |
| `forward_gpu.rs:2489` | `layer.moe_ffn` (K-boundary at K=8) | F7: arena-reset-quota | NO (preserved) |
| `forward_gpu.rs:974` | `output_head.fused_norm_lm_argmax` | F6: argmax 4-byte CPU read | NO (terminal sync) |

**Score after D3 (chunk-engaged pp4096):** ~96 → ~12 commit_and_wait sites (4 stage-end + 1 chunk-attn × 2 unavoidable per layer until Phase 3 + 5 K-boundary + 1 output-head = ~16). After Phase 3 (chunk-internal MTLEvent): ~96 → ~6.

### Existing infrastructure (already in place; not new for ADR-019)

- **`MlxDispatchTypeConcurrent`**: encoder.rs:826-837. Dispatches within a single CB are non-serial unless `HF2Q_FORCE_SERIAL_DISPATCH=1`. Conceptual peer of llama.cpp's `MTLDispatchTypeConcurrent` at `ggml-metal-device.m:467-470`.
- **`MemRanges` auto-barrier**: encoder.rs:391-399, 1217-1243; mem_ranges.rs. Conceptual peer of llama.cpp's `ggml_mem_ranges`. Currently env-gated `HF2Q_AUTO_BARRIER=1` (default OFF); production uses hand-placed `enc.memory_barrier()` calls.
- **Single global `ResidencySet`**: residency.rs:75. ALL allocations attach to one device-level set, registered with the queue at construction (residency.rs:243-247). Topologically opposite to llama.cpp's "one set per allocation" pattern (`ggml-metal-device.m:1354-1390`) but functionally equivalent.
- **Seven caller-owned arenas (ADR-013 P21 Stage 1-3 + ADR-015 iter72-83)**: `FaPrefillArena`, `DnPrefillArena`, `ChunkAllocsArena`, `ChunkInternalArena` (mlx-native), `DenseFfnArena`, `MoeFfnArena`, `FaProjectionsArena`. These shipped specifically to defeat F2 (iter58b residency-rescission) at the per-layer scope.
- **Six commit variants**: `commit`, `commit_and_wait`, `commit_labeled`, `commit_and_wait_labeled`, `commit_wait_with_gpu_time`, `commit_dual_buffer` (encoder.rs:1832-2008; graph.rs:308-327).
- **`unsafe impl Send for CommandEncoder`** (encoder.rs:580-588): documented to support llama.cpp-style GCD `dispatch_apply` worker-thread encoding — **dead infrastructure today** (never exercised in production).
- **Per-CB profiling (`MLX_PROFILE_CB=1`)**: kernel_profile.rs uses `GPUStartTime` / `GPUEndTime` properties; fully functional on M5 Max. Per-dispatch profiling (`MLX_PROFILE_DISPATCH=1`) is silent-no-op on M5 Max per F9 (Apple Silicon AtStageBoundary-only sampling).

### Sequence diagram — current state, chunk-engaged DN layer

```
┌────────────── single transformer DN layer (chunk-engaged) ──────────────┐
│                                                                          │
│  CB 1: layer.gdn.ops1-3                                                 │
│     dispatch: pre_norm → barrier → qkv_proj → z_proj → barrier          │
│              → ssm_conv                                                 │
│     commit_labeled  ◄── NON-BLOCKING (Stage 3a)                         │
│                                                                          │
│  CB 2: layer.gdn.qkv_split                                              │
│     dispatch: qkv_split (W-5b.18 GPU kernel)                            │
│     commit_labeled  ◄── NON-BLOCKING (Stage 3a)                         │
│                                                                          │
│  CB 3: chunk-prep (no label)                                            │
│     dispatch: l2_norm_q → l2_norm_k → alpha_proj → beta_proj            │
│              → barrier → scalar_mul → compute_g_beta                    │
│     commit_and_wait  ◄── BLOCKING (cross-encoder boundary)              │
│                                                                          │
│  CB 4: layer.gdn.chunk_attn (arena)                                     │
│     dispatch_chunk_gated_delta_rule_fwd_with_arena                      │
│     commit_and_wait_labeled  ◄── BLOCKING (iter58b)                     │
│                                                                          │
│  CB 5: layer.gdn.ops8-9                                                 │
│     dispatch: ssm_norm_gate → barrier → out_proj                        │
│     commit_and_wait_labeled  ◄── BLOCKING (residual handoff)            │
│                                                                          │
│  CB 6: fused_residual_norm + DenseQ/MoeQ FFN (SHARED ENCODER)           │
│     dispatch: dispatch_fused_residual_norm_f32                          │
│     barrier                                                              │
│     dispatch_dense_q (or dispatch_moe_q): full FFN body                 │
│     IF is_k_boundary: commit_and_wait_labeled  ◄── BLOCKING (F7)        │
│     ELSE:             commit_labeled            ◄── NON-BLOCKING        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Per layer:** 6 CBs, of which 3 are blocking. Across 30 chunk-engaged DN layers + 10 FA layers + output head: ~100 CBs total, ~96 blocking syncs.

---

## Peer Reference: llama.cpp ggml-metal Encoder Architecture

Per researcher-llama §1, §3 (lines 17-69, 152-217). Reproduced in compressed form for ADR-019 decision context. Full details at `/tmp/cfa-20260503-adr019-parallel-encode/research/llama-encoder-architecture.md`.

### Topology (the load-bearing facts)

| Property | llama.cpp value | hf2q value | Source |
|---|---|---|---|
| MTLCommandQueue per device | 1 (shared across all backends) | 1 (shared) | `ggml-metal-device.m:639` / `device.rs:50-63` |
| Default `n_cb` | 1 (so 2 CBs per `graph_compute`) | n/a (no n_cb knob) | `ggml-metal.cpp:608, 702` |
| `n_cb` ceiling | 8 (`GGML_METAL_MAX_COMMAND_BUFFERS`) | n/a | `ggml-metal-context.m:20` |
| Apple-measured optimal n_cb on M1 Pro / M2 Ultra | **1-2** | n/a | `ggml-metal-context.m:458` (verbatim comment) |
| Encoder-fanout mechanism | `dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async)` GCD worker pool | none (single-thread encoding) | `ggml-metal-context.m:550` |
| Cross-CB sync primitive | `[cmd_buf enqueue]` order on shared queue (NO MTLEvent/MTLFence) | Same: serial-queue submission order (no MTLEvent/MTLFence on graph path) | §3.1 |
| Intra-CB hazard mechanism | `MTLDispatchTypeConcurrent` + `ggml_mem_ranges` scoreboard auto-barriers | `MTLDispatchTypeConcurrent` + `MemRanges` (env-gated) + hand-placed `enc.memory_barrier()` | §3.1 / encoder.rs |
| Graph-split partition unit | Node count (`n_nodes_per_cb = ceil((n_nodes - n_main) / n_cb)`) | n/a (no graph IR on production path) | `ggml-metal-context.m:466` |
| Main-thread prefix | `MAX(64, 0.1 * n_nodes)` | n/a | `ggml-metal-context.m:445` |
| Capture mode (debug) | Force per-CB `waitUntilCompleted` (`GGML_METAL_CAPTURE_COMPUTE`) | `MLX_PROFILE_CB=1` forces synchronous commits | encoder.rs:1923 |
| MTLEvent usage on graph path | **NONE.** Only used for `ev_cpy` async tensor copies | NONE (matches) | `ggml-metal-device.m:944-988` |

**Key invariant llama.cpp relies on (Apple's documented contract):** `MTLCommandQueue` executes CBs in **submission order**, where submission order is fixed by the order of `[cmd_buf enqueue]` calls (NOT `[cmd_buf commit]`). Apple ref: [MTLCommandBuffer.enqueue](https://developer.apple.com/documentation/metal/mtlcommandbuffer/1442997-enqueue). Worker order in `dispatch_apply` is irrelevant for correctness: worker 5 may finish encoding before worker 0, but the GPU still executes CB 0's work first because of the prior `enqueue` order.

### What this means for ADR-019

1. **Cross-CB hazard policy is already aligned.** mlx-native already relies on serial-queue submission order; no MTLEvent/MTLFence wrapping is required to preserve graph correctness. The CB-count consolidation is what's missing, not the sync primitive.
2. **`dispatch_apply` is not the lever.** llama.cpp's own measurement says n_cb=1-2 optimal on Apple Silicon. The CPU-side encode parallelism win is small once the CB count drops to ~2.
3. **gemma already does this correctly in mlx-native** (forward_mlx.rs:1488 = 1 CB / decode default; 2 CBs under `HF2Q_DUAL_BUFFER`). The question for ADR-019 is whether to retarget qwen35's chunk-engaged path to the gemma-shape, and how aggressively.
4. **Apple Best Practices Guide says "preferably one CB per frame"** ([archive link](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html), researcher-deep §2.1). gemma's 1-CB/decode shape is the verbatim Apple-recommended pattern. qwen35's ~100 CBs/chunk-engaged-prefill is the verbatim Apple-anti-pattern.

---

## Counter-Evidence Against Naïve Parallel-Encode (D2 — REJECTED)

Per researcher-deep §3.1, §3.2, §4 (D2), §7.2. The framing "ADR-019 = port llama.cpp's `dispatch_apply`" is intuitive but empirically wrong for hf2q's workload + hardware. Five independent counter-evidence streams, all measured or cited:

### 1. Apple's Metal Best Practices Guide (verbatim)

> "Submit the fewest possible command buffers per frame without underutilizing the GPU."
>
> "There is usually sufficient CPU work queued up to keep the GPU busy by submitting only one or two command buffers per frame **(preferably one)**."
>
> ([archive link](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html))

The Apple-recommended shape is one CB per inference frame. n_cb=4-8 (D2) is anti-shape. n_cb=1 (which is llama.cpp's default) IS the recommended shape and is what D3 + D1 converge to. D2 over-shoots.

### 2. Apple's MLX framework chose ops-per-buffer batching, NOT n_cb pool

PR [ml-explore/mlx#1864](https://github.com/ml-explore/mlx/pull/1864) (awni 2025) — Apple's own ML framework. The PR implements **dynamic ops-per-buffer batching** (20 ops/iPhone, 40 ops/M-series base/pro, 50 ops/M-series Max/Ultra) and posts gains:
- M2 Ultra Qwen2.5-0.5B: 292.6 → 368.2 t/s (+25.8%)
- M4 Max Qwen2.5-0.5B: 479.0 → 532.7 t/s (+11.2%)
- iPhone 16 Pro Qwen2.5-0.5B: 131.6 → 155.8 t/s (+18.4%)

**MLX framework does NOT parallel-encode** (single global RLock per backend, confirmed via `MLXLockContext` reference). The Apple ML team had every option open and **chose D-shape (consolidate-larger-CBs, per researcher-deep §3.2) over D2-shape (parallel-encode)**. This is the most direct counter-evidence available — the team that owns Metal chose against D2 for inference workloads.

### 3. llama.cpp's own measurement says n_cb=1-2 optimal on Apple Silicon

`ggml-metal-context.m:458` (verbatim): *"tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2"*. The benchmark range explored is 1..8; the answer is 1-2. PR comments cite "n_cb=2 is a good default on AMD dGPU" — the win is on AMD discrete GPUs (PCIe amortization), not on Apple Silicon (unified memory). The architectural primitive for D2 (`dispatch_apply` + `[cmd_buf enqueue]`) is lifted directly from llama.cpp; lifting only the primitive without the n_cb=1-2 conclusion would be cargo-culting.

### 4. ADR-013 P21 K=8 already captured most of llama.cpp's win shape

ADR-013 P21 dropped sync_count from 161 → 6 via FFN-terminal batching. That **is** the consolidation win. The remaining gap (chunk-engaged pp4096 0.667× peer) is in the chunk path — 30 layers × 3 internal commits each = 90 commits that K-batching cannot reach (chunk wrapper internal lifecycle). D2 (parallel-encode) does not address chunk-path commits at all; it merely encodes the existing 90 commits on multiple threads. **D3 (per-stage-fence) does address them** by collapsing to MTLEvent intra-stage boundaries.

### 5. D2 introduces material OOM risk on apex MoE

Per researcher-deep §5.5: parallel CBs hold more buffers in flight simultaneously. The Qwen3.6-35B-A3B working set is ~24 GB on M5 Max (5.2 GB context + 23.9 GB model). Current 1-CB-at-a-time keeps the in-flight working set bounded. n_cb=4 (D2) inflates this 4× — peak memory may overrun on apex MoE under long context. **D2 is the only candidate of the four with material OOM risk.**

### Additional D2-specific weaknesses (research synthesis §4 D2)

- **Concurrency surface area worsens iter58b lifetime invariants.** GCD timing is unpredictable; mlx-native's `ResidencySet`'s `pending` queue is not lock-free; concurrent `commit_labeled` from multiple worker threads opens new race classes.
- **mlx-native's `KernelRegistry` is `&mut self` for `get_pipeline`** (kernel_registry.rs:626). With N workers, this becomes a contention point requiring `parking_lot::Mutex` wrapping — not a hard problem but additional surface area.
- **Determinism gates may flake.** ADR-013 ships a 4-fixture decode parity gate; concurrent encoding may produce subtly different ordering across fixtures. D1, D3, D4 are deterministic (single-threaded); D2 is the only candidate where concurrency-induced non-determinism is a real risk.

### D2 verdict: REJECTED with prejudice

Citing five convergent streams: Apple BPG verbatim + Apple ML team's choice + llama.cpp's own measurement + ADR-013 P21 already captured the win shape + OOM risk on apex MoE + concurrency surface area worsens existing fences. Implementation cost (600-1000 LOC, multi-week) is high; expected wall savings are low (~150 µs CPU encode amortization once CB count drops to ~10 anyway). **No D2-shaped option will be shipped in the ADR-019 implementation arc.** The `unsafe impl Send for CommandEncoder` infrastructure (encoder.rs:580-588) is preserved as dead infrastructure with a docstring update redirecting future readers to ADR-019 for the rationale.

---

## Decision

ADR-019 ships **D3 (per-stage-fence) as the primary architectural target**, with **D1 (single-CB-everywhere) as the second-stage convergence target** after the chunk-prefill arena lifetime work clears the iter58b residency-rescission fence in its long-window form. **D2 (n_cb-pool / parallel-encode) and D4 (streaming watermark) are explicitly rejected** for the reasons enumerated above and below.

### Design space summary table

| Axis | D1 (single-CB) | D2 (n_cb pool) | **D3 (per-stage-fence)** | D4 (watermark) |
|---|---|---|---|---|
| Apple BPG alignment | verbatim ✓ | counter ✗ | near-aligned ✓ | MLX-aligned ✓ |
| Sourdough risk | High (whole-pass iter58b) | High (concurrency races) | **Medium (per-stage)** | Low-medium |
| LoC cost | 800-1500 | 600-1000 | **300-600** | 400-700 |
| Decode wall savings | small | none | small | small |
| Prefill wall savings (default) | medium | small | medium | medium |
| Prefill wall savings (chunk-engaged) | large (~80 ms via D3+arena) | small | **medium (~30-60 ms standalone; 80 ms via Phase 3a+3b combined)** | medium-large (Apple PR #1864 +11-26%) |
| Profiling continuity | new (encoder labels) | preserves per-CB | **preserves per-stage** | new (auto labels) |
| Unblocks `MLX_UNRETAINED_REFS` | Yes | No | Partially | Partially |
| Apple-precedent | gemma (already ships) | llama.cpp (n_cb=1) | none (novel) | MLX framework |
| OOM risk on apex MoE | none | **yes** | none | bounded |
| **Verdict** | **second-stage** | **REJECTED** | **PRIMARY (subject to Phase 0 D4-falsification gate)** | **DEFERRED (Phase 0 microprototype)** |

**Codex-review note (2026-05-03):** AC-P1's 80 ms wall reduction gate is **contingent on Phase 3a + 3b combined**. Phase 3a alone (per-stage fence, 4 CBs) projects ~30-60 ms savings; the residual ~20-50 ms requires Phase 3b chunk-prefill arena lifetime work (deferred from ADR-013 P21 Stage 3). If D3 Phase 3b stalls, D4 microprototype is the documented fallback.

### D3 (PRIMARY) — Per-Stage-Fence

**Shape:** 8-9 MTLCommandBuffers per chunk-engaged prefill (revised post-Codex-review for accurate accounting), partitioned by high-level stage:

| Stage | CB count | Dispatches | Commit kind |
|---|---:|---|---|
| 1 | 1 | embedding + first-layer pre-norm | `commit_labeled` (non-blocking) |
| 2 | 1 | attention tower (10 FA layers + 30 DN layers attn body for Qwen3.6-35B-A3B-MoE) | `commit_labeled` (non-blocking) — see §6 chunk-internal MTLEvent caveat |
| 3 | 5-6 | FFN tower (dense or MoE expert dispatch); K=8 batching produces 5 K-boundary commits + 1 stage-end commit | `commit_labeled` × (5+1) (within-stage; iter58b residency-fence-mandated) |
| 4 | 1 | output norm + LM head + argmax | `commit_and_wait_labeled` (terminal — F6 unavoidable) |
| **Total** | **8-9 CBs** | ~8000+ dispatches across stages | 1 host-wait at the end |

**CB-count reconciliation (post-Codex):** the ADR's earlier "4-6 CB target" was the IDEAL (collapsed FFN K-boundaries), not the achievable figure for ADR-013 P21's K=8 batching invariant. AC-P6 is set to `cmd_buf_count ≤ 8` to match this revised accounting. The 4-6 ideal is reachable in Phase 3b once the K-boundary fence (ADR-013 P21 Stage 3 chunk-prefill arena) clears; Phase 3a-only ships at 8-9 CBs.

**Boundary primitive:** MTLEvent fences between stages (intra-stage hazards still use `MTLDispatchTypeConcurrent` + `enc.memory_barrier()` + `MemRanges`). Single host-side `commit_and_wait` at stage 4 (output head). Total: 4-6 CBs / chunk-engaged prefill, 1 host wait.

**Migration scope (research synthesis §6.3):** ~300-600 LoC. Lift the 4 stage boundaries in `forward_gpu.rs`; replace 4-5 of the K-boundary `commit_and_wait_labeled` calls with `commit_labeled` + MTLEvent.signal; the next stage opens its CB with MTLEvent.wait. Single host-side `commit_and_wait` at the very end (output head). Keeps decode hot path on its current shape until phase 5.

**Why this is the primary choice (research synthesis §7.1):**
1. **Smallest LoC for measurable win.** Closes the easy 60% of remaining sync_count overhead with minimal new abstraction.
2. **Sourdough-safe.** Per-stage iter58b windows are smaller than D1's whole-pass window. The chunk-prefill stage (the iter58b-touchy zone) is one of four stages; the other three (embedding, FFN tower, output head) have already-clean lifetimes.
3. **Profiling continuity.** xctrace MST attribution stays per-stage with minimal redesign. Operators read "embedding | attn | ffn | output" rows in the trace — far easier than D4's auto-named CB1/CB2/...
4. **Buys time for chunk-prefill arena work.** The iter58b-DANGEROUS class (30 chunk-internal sync sites in `apply_gated_delta_net_chunk`) is not on D3's critical path; it can be addressed in Phase 3 or deferred. When ChunkInternalArena (mlx-native iter83 commit `62298b4`) is verified end-to-end, D1 becomes a 1-week refactor of D3's stage-fences into a single CB.
5. **Decode is already there.** D3's prefill focus does not regress decode parity (currently 1.07× peer). Phase 5 (D1 convergence) deals with decode separately.

### D1 (SECOND-STAGE) — Single-CB-Everywhere

**Shape:** ONE MTLCommandBuffer per entire prefill (or per decode token). Encoder-switching via `endEncoding` / `beginCompute*` pairs *within* the same CB. Hazards via `memoryBarrierWithScope:` (already shipping). Wait at the END only via single `commit_and_wait`.

**Why second-stage:**
- Prerequisite is **chunk-prefill arena lifetime work** — the ADR-013 P21 Stage 3 deferred scope (~30 internal scratches in `apply_gated_delta_net_chunk` + `chunk_gated_delta_rule.rs`). Once that lands, D1 is a mechanical refactor on top of D3.
- Unblocks `MLX_UNRETAINED_REFS=1` (~3-5% additional perf via skipped per-buffer ARC retain on submit).
- Matches Apple's verbatim guidance and gemma's already-shipping shape.

**Why not primary:**
- D1 directly is high-risk (multi-week arena refactor + simultaneous CB consolidation = simultaneous failure modes).
- Decode regression risk: D1 single-CB-decode encloses all 64 layers + output head. If any kernel hits an iter58b lifetime corner, the whole token panics. Currently a per-layer panic localizes to one layer. Mitigation: ship D1 prefill-only first; keep decode on its current shape until the arena work is comprehensively proven on prefill.

### D2 (REJECTED) — n_cb-pool

See §"Counter-Evidence Against Naïve Parallel-Encode" above. Five convergent streams of counter-evidence. Not pursued.

### D4 (DEFERRED — pending Phase 0 microprototype) — Streaming Watermark

**Shape:** continuous encoding into a "currently-active" CB. Commit + start-new-CB when the active CB hits a threshold (N dispatches encoded, M MB of buffer references retained, T µs since open). MLX framework's actual production scheduler shape (per PR #1864).

**Status revised post-Codex review (2026-05-03):** Codex Phase-2b cross-review flagged D4's outright rejection as "too categorical relative to the Apple/MLX evidence" (the apple-mlx-metal-2026 dossier identifies PR #1864's 50-op/50-MB scheduler as the closest Apple-blessed M-series reference). **D4 is NOT rejected — it is DEFERRED pending a Phase 0 microprototype that benchmarks D4-shape against D3-shape on the FA-path slice.**

**Why D3 remains the likely primary (subject to Phase 0 falsification):**
- **Profile redesign cost.** D4 loses the *semantic* per-CB labels that hf2q's profiling story is built on (`layer.attn_moe_ffn`, `output_head.fused_norm_lm_argmax`). xctrace MST shows generic CB1/CB2/... rows; correlating to model components requires a separate label scheme (encoder-level `setLabel`).
- **Workload-tunable knob.** D4 introduces an `N` parameter (20-50) that affects perf and behaves differently per chip generation. D3's per-stage boundaries are intrinsic to transformer structure.
- **Conditional advantage.** D4's wins likely shine for non-transformer workloads where stage boundaries are unclear; hf2q's transformer forward has crisp 4-stage structure. Phase 0 microprototype falsifies-or-confirms.

**Phase 0 D4 microprototype gate (NEW per Codex-review):**

If a 1-week branch-local D4 prototype on the FA-path slice (~10 layers, 10% of the workload) shows:
- Wall reduction ≥ 60 ms above D3-projected savings (AC-P1 / 80 ms target × 75% to account for FA-only scope), AND
- No new parity violations introduced (4-fixture decode no-regression on FA-only canary),

then D4 BECOMES the primary path and D3 is dropped. Otherwise, D4 stays deferred and D3 ships as planned. **This Phase 0 evidence-based gate replaces the original "rejected with prejudice" framing.**

**Revisit triggers post-D3 ship:** if hf2q starts shipping non-transformer workloads (encoder-decoder, vision-language, agentic tool-use scheduling) where the watermark heuristic earns its keep, OR if D3 Phase 3b stalls on the chunk-prefill arena work, D4 is the queued fallback architecture.

---

## Acceptance Criteria

Concrete, testable, with thresholds. Pre-registered before implementation begins. Per-phase gates in §"Migration Plan" below; this section is the ADR-019 ship gate (cumulative across all D3 phases).

### Performance gates

| ID | Metric | Threshold | Source |
|---|---|---|---|
| **AC-P1** | Wall-time, pp4096 chunk-engaged | hf2q ≤ 1769 ms (vs current 1849 ms; ≥ 80 ms reduction = 16% of 530 ms encoder/orchestration residual) | iter88a fixture, 5-trial trimmed median, cold start |
| **AC-P2** | Wall-time, pp4127 default-axis | hf2q ≤ 1480 ms (vs current 1510 ms; no regression + ≥ 30 ms reduction) | same harness |
| **AC-P3** | Wall-time, decode (4 fixtures) | All 4 ≥ 1.00× peer (no regression from current 1.07× decode parity) | iter61c re-validation harness |
| **AC-P4** | Sync count, chunk-engaged pp4096 | `sync_count ≤ 12` (down from current ~96 estimated via §"Production blocking site enumeration") | `mlx_native::sync_count()` runtime telemetry |
| **AC-P5** | Sync count, pp80 non-chunk | `sync_count == 6` (unchanged from P21 K=8 floor) | same |
| **AC-P6** | CB count, chunk-engaged pp4096 | `cmd_buf_count ≤ 8` (down from current ~100) | `mlx_native::cmd_buf_count()` |
| **AC-P7** | iter88a microbench, gate/up MoE FFN | ≥ 1.0× peer (no regression from current 1.16-1.20×) | iter88a per-kernel comparison harness |

### Parity gates

| ID | Metric | Threshold | Source |
|---|---|---|---|
| **AC-PA1** | 4-fixture decode byte-parity at temp=0 / top-k=1 | All 4 IDENTICAL: 27B-dwq46, 35B-A3B-apex Q5_K, 35B-A3B-apex q4_0-flat, gemma-26B-bf16-dwq | iter61c capture method (SHA256 over UTF-8 decoded text) |
| **AC-PA2** | Heisenbug 5× cold guard, chunk-prefill | 5 cold-cache trials produce byte-identical output | researcher-mlx F2 / iter58b methodology |
| **AC-PA3** | Sourdough byte-exact gate | 3656-byte output unchanged under `HF2Q_USE_DENSE=1` | `scripts/sourdough_gate.sh` |
| **AC-PA4** | Test suite | 2615/2615 PASS (current) | `cargo test --release` |
| **AC-PA5** | mlx-native test suite | 137/0/0 (current) + new MTLEvent ordering tests | `cargo test --release` from `/opt/mlx-native` |

### Operational gates

| ID | Metric | Threshold | Source |
|---|---|---|---|
| **AC-O1** | Memory peak, apex MoE long-context | No regression vs current (≤ 24 GB working set) | `vm_stat` / Activity Monitor |
| **AC-O2** | xctrace MST attribution | 4-6 named stage rows visible per prefill (embedding, attn, ffn, output) | xctrace `metal-application-encoders-list` join |
| **AC-O3** | env-gate behavior | `HF2Q_PER_STAGE_FENCE=0` reverts to current per-component CBs (kill switch) | runtime env check |
| **AC-O4** | Capture mode interaction | `MLX_PROFILE_CB=1` + `MLX_METAL_CAPTURE=path.gputrace` produce coherent traces | iter63 profiling kit recipe |

### Cumulative ship-gate

ADR-019 closes when **all 16 ACs** pass, plus operator review of the full xctrace MST profile against pre-D3 baseline.

**Partial-closure clarification (post-Codex Phase-2b review, 2026-05-03):** Phase 3a alone (per-stage fence at 4-stage boundaries) projects ~30-60 ms savings = AC-P1 partial. Phase 3a CAN merge to main behind `HF2Q_PER_STAGE_FENCE_PHASE3A=1` env-gate **without ADR-019 closure** if it shows measurable wall improvement and all parity ACs pass; it is documented as "intermediate progress, not final closure." ADR-019 closure (i.e. the env gate flips to default-on) requires Phase 3b chunk-prefill arena work landing AND AC-P1's 80 ms target met OR D4 microprototype evidence (Phase 0a.2) re-routing the architecture.

This split addresses Codex issue 4: Phase 3a-only ships behind env gate (preserving kill-switch) but doesn't auto-close ADR-019 until the cumulative target is met.

---

## Migration Plan

Phased, gated, with parity per increment. Each phase ships behind its own env gate (`HF2Q_PER_STAGE_FENCE_<phase>=1`) and lands as a discrete CFA cycle (Codex Phase-2b review before merge to main).

### Phase 0a — Pre-implementation measurement gates (NEW post-Codex review)

**Promoted from OQ-1 / OQ-2 per Codex Phase-2b review (2026-05-03)** — these measurements run BEFORE Phase 0b (EncoderSession) abstraction work to validate the assumptions D3 rests on.

**Phase 0a.1 — xctrace residual attribution.** Operator-driven Xcode Instruments capture of one chunk-engaged generate run on apex q4_0-flat pp4096. Bin the 530 ms `wall - GPU` residual by:
- CPU encoder-build time (per-CB)
- Driver commit overhead (per-CB)
- Inter-CB GPU pipeline-bubble time
- Residency-set add/remove churn

**Acceptance Phase 0a.1:** if any single class accounts for ≥ 70% of the 530 ms residual, that class is the primary D3 lever. If the residual is uniformly distributed, D4 watermark may be a better fit and Phase 0a.2 microprototype becomes load-bearing.

**Phase 0a.2 — D4 microprototype falsification gate.** 1-week branch-local prototype implementing MLX PR #1864-style watermark on the FA-path slice (10 FA layers only, not the whole forward). Compare wall-time + per-CB profile against current per-component shape and against D3-projected shape (calculated, not yet built).

**Acceptance Phase 0a.2:** D4 microprototype must show wall reduction ≥ 60 ms above D3-projected on the FA-path slice (= 75% of AC-P1's 80 ms target × FA-path's ~16% of total wall = 9.6 ms minimum; using 60 ms as conservative ceiling) AND no parity regressions on FA-path canary fixture. If D4 fails this gate, D3 ships as planned. If D4 passes, D3 is dropped and D4 becomes new PRIMARY.

**Phase 0a.3 — MTLEvent cost calibration.** Microbench MTLEvent.signal/wait roundtrip on M5 Max. iter89b audit cited 100-500 ns / commit; iter63 inter-CB GPU pipeline-bubble cost is unmeasured. Need: per-CB MTLEvent cost, per-CB residency add/remove, per-CB driver commit overhead.

**Acceptance Phase 0a.3:** all three microbench numbers documented with ± noise floor; AC-P4 sync_count target validated against measured event cost.

**Measured 2026-05-02 (M5 Max, mlx-native branch `adr019-phase0a3-mtl-event-microbench`):**
- **H1 = 1.25 µs ± 0.25 µs** per MTLSharedEvent signal+wait CB-pair (= ~620 ns per single signal-or-wait encode at the Metal driver surface). iter89b's 100-500 ns estimate was the encode-call CPU cost in isolation; the cross-CB GPU sync the event enforces costs an additional ~500 ns. Combined per-encode is at the upper end of the iter89b range (revised upward by ~6×), still negligible vs. driver-commit floor.
- **H2 = 2.79 µs ± 0.4 µs** per residency add+remove cycle (= ~1.4 µs / staged-flush). PASS by ~36× vs. the < 100 µs prediction. iter8e defer-and-flush design (one `[set commit]` per CB instead of per-allocation) is doing exactly the work it was designed to do.
- **H3 = 13.3 µs ± 0.1 µs** per single empty CB with synchronous wait; ~8 µs per CB when two CBs pipeline behind one `wait_until_completed`. PASS by ~38× vs. the < 500 µs prediction.
- **AC-P4 validation: PASS.** At 1.25 µs / CB-pair, D3's projected ~80 intra-stage MTLEvent fences cost ~100 µs total — negligible against the 530 ms residual AC-P1 targets (80 ms). Per-component `commit_and_wait` → chunk-internal MTLEvent fence saves ~6.5 µs per converted boundary at the M5 Max driver level. Event cost cannot block AC-P4. The 530 ms residual is pipeline-bubble + encoder-build dominated, NOT driver-commit, so D3's per-stage-fence consolidation must address those (Phase 0a.1 xctrace bins), confirming the existing migration plan.
- Methodology: criterion 0.5, 5-trial protocol (3 contaminated by rust-analyzer at 96.8% CPU; 4 + 5 canonical, ≤ 1.5% drift). Full report: `/tmp/cfa-adr019-phase0a3/results.md`.

**ETA Phase 0a:** 5-10 man-days operator + 3-5 man-days dev for microprototype = ~2 calendar weeks.

**Phase 0a is the gate that authorizes Phase 0b.** If Phase 0a falsifies D3 in favor of D4, the Migration Plan below is rewritten before any structural code lands.

### Phase 0b — EncoderSession abstraction (PREREQUISITE)

**Scope:** Lift the existing `CommandEncoder` to a `CommandEncoder + EncoderSession` pair. `EncoderSession` owns:
- The current MTLCommandBuffer (or pool, for D1 future).
- An open MTLComputeCommandEncoder (the current persistent compute encoder).
- An optional MTLEvent for stage-fence semantics.
- The residency-set membership (delegated to MlxDevice's existing single-set).
- The K-batch state (already partially in P21).
- The labeled phase context (for xctrace).

**LoC:** ~200 in `/opt/mlx-native/src/encoder.rs` + new `/opt/mlx-native/src/encoder_session.rs`. No behavior change. Feature-flagged behind `HF2Q_ENCODER_SESSION=1`.

**Acceptance:**
- AC-PA4 (test suite green) at every commit.
- AC-PA1 (4-fixture decode byte-parity) at end of phase.
- New tests: `encoder_session_lifecycle.rs`, `encoder_session_label_propagation.rs`.
- Code review: Codex Phase-2b verifies no regression to F1 (persistent compute encoder amortization), F2 (residency lifetime), F11 (zero-init alloc).

**ETA:** 3-5 man-days single-developer. **Iter labels:** iter89e2-A through iter89e2-C.

### Phase 1 — Output-head + last-layer fusion (lowest-risk first)

**Scope:** Per researcher-mlx Q-5: fuse the last-layer FFN-terminal commit_and_wait into the same CB as `output_head.fused_norm_lm_argmax`. The two are already adjacent in `forward_gpu.rs:974` (output head) and `forward_gpu.rs:2382 / 2489` (last-layer FFN K-boundary). Fusing drops 1 commit per prefill (small but free; the win is the path-finding for D3 boundaries).

**LoC:** ~50 in `forward_gpu.rs`. No new abstraction.

**Acceptance:**
- AC-PA1, AC-PA4 green.
- AC-P5: pp80 sync_count drops from 6 → 5 (or stays at 6 if the test fixture doesn't trigger the fusion path; document either way).

**ETA:** 1-2 man-days.

### Phase 2 — FA-path D3 stage migration (smallest hazard graph; lowest risk)

**Scope:** 10 FA layers; ops1-4 + kv_cache_write + fa.prefill_bridge + ops6-7 form a clear stage. Replace per-component `commit_labeled` with stage-end `commit_labeled` + intra-stage `enc.memory_barrier()`. The four ops already use `FaPrefillArena` (Stage 1) and `FaProjectionsArena` (iter86) — F2 lifetime is structurally clean.

**LoC:** ~150 in `gpu_full_attn.rs`. No new mlx-native code.

**Acceptance:**
- AC-PA1 (full 4-fixture parity).
- AC-PA2 (Heisenbug 5×) — FA path is not the iter58b zone but exercise it anyway.
- AC-P3 (decode no-regression).
- AC-P4: chunk-engaged sync_count drops by ~10 (10 FA layers × 1 commit consolidated).

**ETA:** 5-8 man-days.

### Phase 3 — DN-path D3 stage migration (chunk-engaged + autoreg variants)

**Scope:** 30 DN layers across two regimes:
- **Autoregressive prefill (pp80):** ops1-3 + qkv_split + ops5-9. Already largely `commit_labeled` per Stage 3a; consolidate into one stage-end commit with intra-stage MTLEvent fence for ops5-9 producer→consumer.
- **Chunk-engaged prefill (pp4096):** the iter58b-DANGEROUS class. ops1-3 + qkv_split + chunk-prep + chunk-attn + ops8-9. Two sub-options:
  - **Phase 3a (conservative):** Consolidate ops1-3 + qkv_split into one stage CB; keep chunk-prep + chunk-attn + ops8-9 as separate CBs (preserves existing iter58b fences). Sync count drops by ~30 (30 DN × 1 site).
  - **Phase 3b (aggressive):** Convert chunk-internal commit_and_wait to MTLEvent intra-stage. Requires verifying ChunkInternalArena (mlx-native iter83 commit `62298b4`) covers ALL chunk-pipeline-internal scratches end-to-end. Sync count drops by ~60 (30 DN × 2 additional sites).

**LoC:** Phase 3a ~200 in `gpu_delta_net.rs`. Phase 3b additional ~150 + mlx-native chunk-pipeline scratch lifetime audit (~100 LoC of test code).

**Acceptance (Phase 3a):**
- AC-PA1, AC-PA2 (Heisenbug 5× MANDATORY — chunk-prefill is iter58b zone).
- AC-P4: chunk-engaged sync_count drops to ~60.
- Chunk-engaged pp4096 wall improvement ≥ 40 ms (half of AC-P1's 80 ms).

**Acceptance (Phase 3b):**
- All Phase 3a gates +
- AC-P4: chunk-engaged sync_count drops to ≤ 12.
- Chunk-engaged pp4096 wall improvement ≥ 80 ms (full AC-P1).
- iter58b adversarial test: deliberately drop a chunk-pipeline scratch mid-flight under instrumented arena; verify no rescission via `residency_count_for_test()`.

**ETA:** Phase 3a: 5-8 man-days. Phase 3b: 8-12 man-days (the arena-lifetime audit is the load-bearing scope).

**Risk gate:** if Phase 3b chunk-internal MTLEvent conversion proves materially harder than the 8-12 day estimate, ship Phase 3a alone (still ≥ 40 ms win toward AC-P1; Phase 3b deferred to follow-up CFA cycle).

### Phase 4 — MoE-FFN / Dense-FFN tower D3 stage migration

**Scope:** The FFN tower already has K=8 batching from P21 (5 K-boundary commit_and_wait + 35 K-window-interior commit_labeled per 40-layer prefill). D3 consolidates the 35 K-window-interior commits into 5 stage-CBs (one per K-window), with MTLEvent fence between K-windows and K-boundary commit_and_wait preserved at K-window end.

**LoC:** ~150 in `forward_gpu.rs` + `gpu_ffn.rs`. The K-batching loop body is the consolidation site.

**Acceptance:**
- AC-PA1 (4-fixture parity — iter88a microbench MUST stay ≥ 1.0× peer).
- AC-P4: pp80 sync_count stays at 6 (K-boundaries preserved); chunk-engaged stays at Phase 3 outcome.
- AC-P7: iter88a gate/up not regressed.

**ETA:** 5-8 man-days.

### Phase 5 — End-to-end + parity gate

**Scope:** Wire all four phases together. Run the full ADR-019 acceptance suite (16 ACs). Operator review of xctrace MST against pre-D3 baseline.

**LoC:** ~50 (config wiring + env-gate audit). Mostly verification work.

**Acceptance:** All 16 ACs in §"Acceptance Criteria" pass.

**ETA:** 3-5 man-days.

### Phase 6 — D1 convergence (FUTURE; deferred to follow-up CFA cycle)

**Scope:** After D3 lands and the chunk-prefill arena work (ADR-013 P21 Stage 3 deferred scope) is comprehensively in place, converge to ONE CB per prefill. Mirrors gemma's already-correct shape. ~100 additional LoC. Unblocks `MLX_UNRETAINED_REFS=1` (additional ~3-5% perf).

**Acceptance:**
- AC-P6 retained at chunk-engaged cmd_buf_count = 1.
- `MLX_UNRETAINED_REFS=1` runs cleanly under all 4 fixtures.
- Wall-time: chunk-engaged ratio ≥ 0.85× peer (vs Phase 5 outcome of ~0.80×).

**ETA:** Out of scope for ADR-019 ship gate. Tracked as ADR-019 follow-up.

### Phase Schedule Summary

| Phase | Scope | LoC | ETA (man-days) | Iter labels | Critical-path? |
|---|---|---:|---:|---|:---:|
| 0 | EncoderSession abstraction | 200 | 3-5 | iter89e2-A..C | YES |
| 1 | Output-head + last-layer fusion | 50 | 1-2 | iter89e2-D | NO (lowest risk first) |
| 2 | FA-path D3 | 150 | 5-8 | iter89e2-E..G | YES |
| 3a | DN-path D3 conservative | 200 | 5-8 | iter89e2-H..J | YES |
| 3b | DN-path D3 aggressive (chunk-internal MTLEvent) | 150 + 100 | 8-12 | iter89e2-K..N | YES (or hold for AC-P1 partial credit) |
| 4 | MoE/Dense FFN tower D3 | 150 | 5-8 | iter89e2-O..Q | YES |
| 5 | End-to-end + parity gate | 50 | 3-5 | iter89e2-R | YES |
| **Total** | | **~1050** | **30-48** | | |
| 6 (future) | D1 convergence | 100 | 5-7 | (separate ADR follow-up) | NO |

**Single-developer ETA: 30-48 man-days = 6-10 calendar weeks** (allowing for bench / parity / review cycles).
**/cfa dual-mode ETA: 15-24 man-days = 3-5 calendar weeks** (claude+codex parallel implementation per phase; queen synthesis at phase boundaries).

---

## Parity Strategy

Per-phase parity testing is mandatory; D3 will not ship without all gates green at each phase.

### Per-phase mandatory tests

| Test | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `cargo test --release` (hf2q) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cargo test --release` (mlx-native) | ✓ | — | — | ✓ (Phase 3b) | — | ✓ |
| 4-fixture decode byte-parity | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Heisenbug 5× cold guard | — | — | ✓ | ✓ MANDATORY | — | ✓ |
| Sourdough byte-exact | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| iter88a microbench (gate/up + down) | — | — | ✓ | ✓ | ✓ MANDATORY | ✓ |
| pp4096 chunk-engaged wall (5-trial trimmed median) | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| pp80 sync_count receipt | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| pp4096 cmd_buf_count receipt | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| xctrace MST capture (operator-driven) | — | — | — | — | — | ✓ |
| Codex Phase-2b review | ✓ | ✓ | ✓ | ✓ MANDATORY | ✓ | ✓ |

### Byte-exact F32 unit tests at seq_len=128

For each phase that touches a kernel boundary or commit semantics, add a unit test that:
1. Runs the affected layer-component at seq_len=128 with deterministic input (fixed seed).
2. Captures full F32 output buffer via `download_f32`.
3. Compares to a baseline captured pre-phase via SHA256 over the F32 byte array.

Pattern reference: `gpu_delta_net.rs` already has 7 unit tests in this style (lines 3646 / 3868 / 4064 / 4114 / 4275 / 4335 / 4697 per researcher-mlx §3 table). New tests added per phase follow the same pattern.

### 4-fixture decode gate

Pattern: ADR-013 iter61c re-validation methodology. Each fixture runs:
- 16 prompts × 32-tok smoke
- 1 prompt × 200-tok sourdough
- temp=0, top-k=1, seed=0
- SHA256 over UTF-8 decoded text bytes
- Expected: byte-identical to baseline at HEAD pre-phase

Captured to `/tmp/cfa-iter89e2-{phase}/parity-receipts.json`.

### iter88a microbench regression check

Pattern: `/tmp/cfa-iter88a/COMPARISON.md` per-kernel peer comparison. Each phase that touches the FFN tower (Phase 4 explicitly; Phases 2-3 indirectly via stage-fence overhead) MUST verify:
- mlx-native MoE FFN gate/up: ≥ 1.0× peer (current 1.16-1.20×)
- mlx-native MoE FFN down: no further regression below current 0.76-0.86×

If iter88a regresses below 1.0× on gate/up, ADR-019 phase HOLDS pending root-cause.

---

## Risk Register

Each Chesterton's fence (F1-F12 from researcher-mlx §5) → which D3 design element addresses it / what risk remains.

### F1. Persistent compute encoder per CB

- **Where:** encoder.rs:815-843 `get_or_create_encoder`.
- **Why:** ~800 encoder create/end cycles per forward pass amortized to 1 per CB.
- **D3 impact:** PRESERVED. Each stage CB still has one persistent compute encoder; D3 reduces CB count, not encoder count per CB.
- **Residual risk:** none. F1 is a net win for D3 (fewer CBs × same persistent-encoder amortization).

### F2. iter58b residency-rescission (THE BIG FENCE)

- **Where:** buffer.rs:68-77 (`MlxBufferStorage::drop` stages remove); encoder.rs flush_residency_pending in commit_*.
- **Why:** wrapper allocates → wrapper commits non-blocking → wrapper returns → scratches drop & stage remove → caller commits & flushes remove BEFORE wrapper's CB has run → Metal demotes pages mid-flight → garbage values.
- **D3 impact:** PARTIALLY MITIGATED. D3 stage CBs are larger windows (longer in-flight time) than per-component CBs, which would *increase* F2 exposure if applied naively. **D3's mitigation:** (a) every stage uses caller-owned arenas (FaPrefillArena, DnPrefillArena, ChunkAllocsArena, FaProjectionsArena, DenseFfnArena, MoeFfnArena) that outlive the stage CB; (b) Phase 3b chunk-internal MTLEvent conversion is gated on verifying ChunkInternalArena coverage end-to-end.
- **Residual risk:** Phase 3b is the highest-risk phase. Mitigation: AC-PA2 Heisenbug 5× MANDATORY at Phase 3b; iter58b adversarial test (deliberately drop scratch under instrumented arena) added to test suite; Codex Phase-2b review MANDATORY at Phase 3b.

### F3. M5 Max residency-quota architectural ceiling (33 GB)

- **Where:** decode_pool.rs:119-127.
- **Why:** dense-Q FFN's 5 pooled scratches × 33 dense layers ≈ 33 GB > residency quota.
- **D3 impact:** PRESERVED. K=8 boundary remains the structural floor for FFN tower; D3 keeps the K-boundary commit_and_wait for arena reset. Phase 4 consolidates K-window-interior commits but does not change K-boundary policy.
- **Residual risk:** none for D3. D1 (Phase 6 future) requires arena-reset cadence redesign to push K toward 40; explicitly out of scope here.

### F4. flash_attn_prefill `cur_len == 0` eligibility

- **Where:** gpu_full_attn.rs:1607.
- **D3 impact:** PRESERVED. The legacy SDPA fallback at gpu_full_attn.rs:1660 retains its commit_and_wait. Production prefill-from-zero (Qwen3.5/3.6) takes the new path; not exercised in chunk-engaged regime.
- **Residual risk:** none. D3 does not touch the eligibility predicate.

### F5. iter61a-4 RAW barrier (gpu_full_attn.rs:2151-2182)

- **Where:** explicit `enc.memory_barrier()` between Op 6 (sigmoid_gate writes `gated`) and Op 7 (linear_projection reads `gated`).
- **D3 impact:** PRESERVED. Phase 2 FA-path migration explicitly retains this barrier; intra-stage hazard handling remains hand-placed (until `HF2Q_AUTO_BARRIER=1` migration in a future ADR).
- **Residual risk:** every NEW producer→consumer edge introduced by D3 stage consolidation MUST add a memory_barrier. Mitigation: per-phase code review checklist explicitly includes "every dispatch added in this phase has explicit producer→consumer barriers."

### F6. Output-head argmax CPU read

- **Where:** forward_gpu.rs:974, 1049.
- **D3 impact:** PRESERVED. AC-P5 explicitly keeps 1 commit_and_wait per prefill (the output-head terminal). Phase 1 fuses last-layer-FFN-terminal into the same CB but still ends with commit_and_wait.
- **Residual risk:** none. Structurally unavoidable.

### F7. K-boundary scratch lifetime contract

- **Where:** forward_gpu.rs:2671-2676 `decode_pool::reset_for_prefill_chunk()`.
- **D3 impact:** PRESERVED. Phase 4 keeps K=8 boundary commit_and_wait; D3 only consolidates K-window-interior commits.
- **Residual risk:** none. AC-P5 explicitly verifies pp80 sync_count == 6.

### F8. ChunkAllocsArena vs ChunkInternalArena

- **Where:** chunk_allocs_arena.rs (caller-owned outer scratches) + mlx-native iter83 ChunkInternalArena (chunk-pipeline kernel internal scratches).
- **D3 impact:** Phase 3b MUST verify both arenas cover ALL transient lifetimes end-to-end before converting chunk-internal commit_and_wait to MTLEvent.
- **Residual risk:** Phase 3b is gated on this verification. If verification surfaces uncovered scratches, Phase 3b HOLDS pending arena audit; ship Phase 3a alone (partial AC-P1 credit).

### F9. `MLX_PROFILE_DISPATCH=1` is silent-no-op on M5 Max

- **Where:** encoder.rs:1601-1639 (Apple Silicon AtStageBoundary-only sampling).
- **D3 impact:** PRESERVED. ADR-019 cannot rely on per-dispatch GPU sampling for measurement; uses per-CB sampling (`MLX_PROFILE_CB=1`) which forces synchronous commits and defeats pipelining.
- **Residual risk:** D3 measurement uses (a) per-CB GPU profile in MLX_PROFILE_CB=1 mode + (b) wall-time bench in normal mode. Both are pre-registered in §"Acceptance Criteria".

### F10. `MLX_UNRETAINED_REFS=1` caller contract

- **Where:** encoder.rs:660-684.
- **D3 impact:** D3 does NOT enable unretained refs. Phase 6 (D1 convergence) is the unblocking phase.
- **Residual risk:** none for D3. D1 future phase explicitly gates on arena-lifetime audit.

### F11. iter61a-1 zero-init alloc_buffer

- **Where:** mlx-native commit `3f21443`.
- **D3 impact:** PRESERVED. Every new buffer allocation in Phase 0-5 respects zero-init.
- **Residual risk:** none.

### F12. iter61a-2 `HF2Q_FORCE_SERIAL_DISPATCH` falsification probe

- **Where:** encoder.rs:821-834.
- **D3 impact:** PRESERVED as debug knob.
- **Residual risk:** none. ADR-019 production does not flip it.

### Decode hot-path regression risk (cross-cutting)

- **Concern:** D3 prefill-only landing in Phases 2-4 leaves decode unchanged; Phase 5 end-to-end may inadvertently regress decode.
- **Mitigation:** AC-P3 (decode no-regression) tested at every phase boundary; AC-PA1 (4-fixture decode parity) MANDATORY at every phase. If decode regresses ≥ 5%, the offending phase HOLDS.

### Multi-fixture parity flake risk (cross-cutting)

- **Concern:** D3 stage consolidation may produce ordering changes that flip one of 4 fixtures off-IDENTICAL while the other 3 stay byte-identical (the worst-case parity failure mode — masks F5 RAW barrier omissions).
- **Mitigation:** AC-PA1 explicitly requires ALL 4 fixtures byte-identical, not 3 of 4. iter61c capture method (SHA256 over UTF-8 decoded text) catches single-token divergence cleanly.

### Test-surface gap risk

- **Concern:** Current mlx-native test suite is 137 tests; hf2q test suite 2615 tests. Neither has an MTLEvent-ordering test (no infrastructure). D3 introduces MTLEvent fences as a new sync primitive with a new ordering invariant.
- **Mitigation:** Phase 0 deliverable explicitly adds:
  - `mlx_native::tests::mtl_event_signal_wait_ordering` — verifies signal-then-wait produces deterministic ordering.
  - `mlx_native::tests::mtl_event_concurrent_signal_one_wait` — verifies multiple signals fold correctly.
  - `hf2q::tests::stage_fence_byte_parity_seq128` — F32 unit test at seq_len=128 with stage fence enabled vs disabled, byte-identical.

---

## Implementation Tasks (iter89e2 series)

Concrete iter89e2-A through iter89e2-R list with effort estimates + ownership + dependencies. Each task lands as one CFA cycle (claude+codex dual-mode where complexity warrants).

| Iter | Task | Phase | LoC | ETA (man-days) | Dependencies | Mode |
|---|---|---|---:|---:|---|---|
| iter89e2-A | EncoderSession struct skeleton | 0 | 80 | 1-2 | — | solo |
| iter89e2-B | EncoderSession + MTLEvent integration | 0 | 100 | 1-2 | A | solo |
| iter89e2-C | EncoderSession parity tests + Phase 0 ship | 0 | 50 | 1 | B | solo |
| iter89e2-D | Output-head + last-layer fusion | 1 | 50 | 1-2 | C | solo |
| iter89e2-E | FA-path D3 stage migration (research) | 2 | — | 1 | D | /cfa research |
| iter89e2-F | FA-path D3 stage migration (impl) | 2 | 150 | 3-5 | E | /cfa dual |
| iter89e2-G | FA-path D3 parity + bench gate | 2 | 30 | 1-2 | F | solo |
| iter89e2-H | DN-path D3 conservative (research) | 3a | — | 1 | G | /cfa research |
| iter89e2-I | DN-path D3 conservative (impl) | 3a | 200 | 3-5 | H | /cfa dual |
| iter89e2-J | DN-path D3a parity + bench gate | 3a | 30 | 1-2 | I | solo |
| iter89e2-K | ChunkInternalArena audit (mlx-native) | 3b | 100 (test) | 2-3 | J | /cfa research |
| iter89e2-L | DN-path D3 aggressive chunk-internal MTLEvent (research) | 3b | — | 1-2 | K | /cfa research |
| iter89e2-M | DN-path D3 aggressive (impl) | 3b | 150 | 3-5 | L | /cfa dual MANDATORY |
| iter89e2-N | DN-path D3b parity + Heisenbug 5× + bench gate | 3b | 50 | 2-3 | M | /cfa parity |
| iter89e2-O | MoE/Dense FFN tower D3 (research) | 4 | — | 1 | N | /cfa research |
| iter89e2-P | MoE/Dense FFN tower D3 (impl) | 4 | 150 | 3-5 | O | /cfa dual |
| iter89e2-Q | FFN tower D3 parity + iter88a microbench | 4 | 30 | 1-2 | P | solo |
| iter89e2-R | End-to-end ADR-019 acceptance suite | 5 | 50 | 3-5 | Q | /cfa parity |
| **Total** | | | **1170** | **30-46** | | |

**/cfa dual-mode notes:**
- iter89e2-F, I, M, P are dual-mode (claude+codex parallel impl, queen synthesis). Apparent rate-of-progress doubles per CFA mantra `feedback_use_cfa_worktrees`.
- iter89e2-M is MANDATORY dual-mode per CFA standing directive on iter58b-class hazards: `feedback_codex_review_catches_unified_memory_races`.
- iter89e2-K is research-only (chunk-pipeline scratch lifetime audit). Its output gates iter89e2-L scope.

**Critical path:** A → B → C → D → E → F → G → H → I → J → K → L → M → N → O → P → Q → R. Phases 0+1 (5 iters) unblock Phase 2; Phase 2 unblocks Phase 3a; Phase 3a + ChunkInternalArena audit unblock Phase 3b; Phase 3b unblocks Phase 4 (no compelling parallel-path); Phase 4 unblocks Phase 5. Some Phase 1 / 2 parallelism is possible after Phase 0 ships.

---

## Open Questions

Per researcher-deep §8 (8.1-8.7) + new questions surfaced during synthesis. Each must be resolved (or explicitly deferred) before final design closes.

### OQ-1. What's actually in the 530 ms encoder/orchestration residual?

**Source:** researcher-deep §8.1.

ADR-015 iter88a found that the chunk-engaged 616 ms gap to peer is dominated by:
- ~86 ms in MoE FFN-down (mlx-native kernel speed, hf2q 0.76-0.86× of peer)
- ~530 ms in "encoder/orchestration/CB residency" — *not yet attributed below this rollup*

ADR-019 (CB-architecture) only addresses the encoder/orchestration piece. If the 530 ms is mostly something else (e.g. mlx-native kernel speed in chunk_gated_delta_rule, page-fault residency overhead), then even perfect D3 won't close the wall-clock gap meaningfully.

**Action:** before committing to Phase 3b (the highest-risk piece): capture xctrace TimeProfiler trace at 0.1 ms sampling on chunk-engaged pp4096 *post-P21 K=8* and bin the 530 ms into (CB-overhead, kernel-time, page-fault, queue-wait). This is a half-day of work; the answer determines whether Phase 3b's 80 ms target is actually achievable.

**Resolution gate:** Phase 3a end. If post-Phase-3a measurement shows ≤ 10% of the 530 ms is in CB-overhead, ADR-019 HOLDS pending pivot to ADR-005 / ADR-013 P14 / mlx-native kernel work.

### OQ-2. Does MTLEvent actually cost what we think on M5 Max?

**Source:** researcher-deep §8.2.

We have measurements for `commit()` (1.6 µs steady-state) and `commit_and_wait()` (~13-17 µs floor; 1.32 ms on real work). We do **not** have measurements for `MTLEvent.encodeSignal` / `encodeWait`. Apple docs imply µs-class; in practice on Apple Silicon the cost may be lower (no PCIe). If MTLEvent costs ~10 µs each, D3 with 4-6 events is fine. If it's 100 µs (worst case), D3's wins are eaten by the events themselves.

**Action:** Phase 0 deliverable extends `mlx-native/examples/cb_cost_calibration.rs` with an MTLEvent microbench. Pre-register threshold: if MTLEvent.encodeSignal+encodeWait > 50 µs end-to-end, Phase 2-4 design pivots to "explicit `commit_and_wait`-then-new-CB" boundary semantics (still 4-6 CBs but with host-side micro-waits between stages instead of MTLEvent).

**Resolution gate:** Phase 0 end.

### OQ-3. Is Metal 4 (`MTL4CommandBuffer`) actually available on production macOS / Rust?

**Source:** researcher-deep §8.3.

WWDC 2025 announced Metal 4. macOS 26 ships it. But Metal 4 APIs in Rust via the `metal` crate may lag — the wrapper may not expose `MTL4CommandBuffer` yet. If we'd want to ship D1 (Phase 6 future) against Metal 4 long-lived CBs, that's a `metal-rs` crate update (or hand-rolled `objc::msg_send!`).

**Action:** check `metal-rs` crate version pin in `mlx-native/Cargo.toml`; if it doesn't expose MTL4, pure-objc bindings may be needed.

**Resolution gate:** Phase 6 (FUTURE; out of ADR-019 scope). Documented here for tracking.

### OQ-4. What is MLX framework's MTL4 strategy?

**Source:** researcher-deep §8.4.

MLX is Apple's own framework. They will be first to MTL4 patterns. Worth tracking `ml-explore/mlx` PRs in 2026 H1 for MTL4 adoption — that's our reference for what the MTL4-shape ADR-019 follow-up should look like.

**Action:** none for ADR-019 scope. Flag for Phase 6 follow-up ADR.

### OQ-5. What's the actual cost of the existing `commit_labeled` infrastructure?

**Source:** researcher-deep §8.5.

The labeled-commit pattern is shared by ADR-015 iter16 (xctrace MST), iter63 (per-dispatch sampling), residency-set flush. Removing/replacing per-layer commits in D3 may interact with each of these. iter63's per-dispatch sampling, in particular, has a 4096-samples-per-CB ceiling (encoder.rs:442) — at decode 120 dispatches/CB it's fine, but at single-CB-prefill (Phase 6 D1, ~6000 dispatches), it'd truncate.

**Action:** Phase 0 EncoderSession design includes a sample-buffer-chunking story (when CB grows beyond 4096-sample ceiling, chain into a fresh sample buffer transparently to the caller).

**Resolution gate:** Phase 0 end.

### OQ-6. What does MLC-LLM do?

**Source:** researcher-deep §8.6.

This synthesis did not research MLC-LLM's Metal backend in depth. MLC-LLM ships production. Their CB architecture would be a useful third reference point alongside MLX and llama.cpp.

**Action:** if /cfa cycle has bandwidth, spend 30-60 min on MLC-LLM source — `mlc-ai/mlc-llm/cpp/serve/` and Metal-specific paths. If MLC-LLM has converged on a different shape than D1/D3, revisit.

**Resolution gate:** opportunistic; not blocking.

### OQ-7. What about the chunk-prefill kernel itself?

**Source:** researcher-deep §8.7 + researcher-mlx Q-6.

`mlx-native/src/ops/chunk_gated_delta_rule.rs` has internal `commit_and_wait` calls (per ADR-013 P19 audit, "dangerous" class). These are inside the kernel wrapper, not in hf2q's hot path. ADR-019 needs to coordinate with mlx-native ownership: are these moving to MTLEvent in mlx-native, or do we keep them and tolerate the residual?

**Action:** iter89e2-K (ChunkInternalArena audit) is the canonical resolution work. Phase 3b explicitly gates on this question.

**Resolution gate:** Phase 3a end → iter89e2-K → Phase 3b authorization.

### OQ-8. Is sequential-encode-faster the actual lever, not parallel-encode?

**Source:** researcher-mlx Q-7.

The P21 EOD residual at pp80 is 262 ms host overhead (between commits) and ~37 ms per-commit floor. With 6 commits at K=8 the residual is ~6 ms × 6 = 36 ms which is consistent with the wall − GPU residual. So the host overhead is NOT in the commit boundaries themselves but in the CPU-side encoding work (apply_bindings, apply_pipeline, dispatch_threadgroups ObjC calls, allocator churn for params buffers, etc.). Moving to worker-thread encoding (D2) addresses this directly. Per-CB-fusion (D3) addresses commit count which has a 1.32 ms × commits floor that's already small.

**Resolution:** ADR-019 chooses D3 for the chunk-engaged regime where CB-count is the dominant cost (96 commits × 1.32 ms ≈ 127 ms). For pp80 where CB-count is already at floor (6), per-commit-cost optimization is the lever — but per-commit cost is already at ~6 ms / commit and further reduction is ADR-005 / ADR-015 territory, not ADR-019. **ADR-019's win is the chunk-engaged regime.** This is documented in §"Acceptance Criteria" — AC-P1 targets chunk-engaged; AC-P2 targets default-axis with a smaller threshold.

### OQ-9. Should `HF2Q_AUTO_BARRIER` flip default ON as part of D3?

**Source:** researcher-llama §6.7.

Today `HF2Q_AUTO_BARRIER=1` is off; hand-placed `enc.memory_barrier()` is the production path. With D3 stage consolidation, every barrier site needs to participate in the per-CB scoreboard so cross-stage-boundary hazards (a stage's last dispatch writes a buffer that the next stage's first dispatch reads) get the benefit of `MTLDispatchTypeConcurrent` plus the right barriers.

**Resolution:** ADR-019 holds `HF2Q_AUTO_BARRIER=1` default OFF for now; flipping it on is a separate ADR (ADR-020 candidate). D3 stage boundaries use explicit MTLEvent fences (no MemRanges interaction). Intra-stage hazards remain hand-placed. If Phase 5 measurement surfaces a missing barrier (caught by AC-PA2 Heisenbug 5×), the fix is hand-placement, not auto-barrier.

### OQ-10. What about the gemma fast path?

**New question surfaced during synthesis.**

gemma's `forward_mlx.rs:1488` is already at 1 CB / decode (the canonical correct shape). Does D3 retroactively apply to gemma? Or stay qwen35-only?

**Resolution:** D3 is qwen35-specific. gemma is already at the D1 end-state (post-Phase-6 future). ADR-019 explicitly does not touch `forward_mlx.rs` (the gemma forward path). The encoder.rs / device.rs changes (Phase 0 EncoderSession) are mlx-native-side and benefit gemma indirectly (e.g. EncoderSession can host gemma's `HF2Q_DUAL_BUFFER` mode), but no behavior change to gemma's production path.

---

## Receipts

Pointers to all 3 research artifacts, the iter88a-89e bench/measurement evidence, and supporting documents.

### Primary research artifacts (CFA session adr019-parallel-encode)

- `/tmp/cfa-20260503-adr019-parallel-encode/research/llama-encoder-architecture.md` (910 lines, researcher-llama, 2026-05-02) — literal walkthrough of llama.cpp ggml-metal encoder lifecycle with file:line citations + per-question delta against mlx-native current state (sections A-H).
- `/tmp/cfa-20260503-adr019-parallel-encode/research/mlx-native-current-state.md` (531 lines, researcher-mlx, 2026-05-03) — full enumeration of every commit_and_wait_labeled / commit_labeled site in qwen35 production prefill path; 12 Chesterton's fences (F1-F12); 7 open questions for architect.
- `/tmp/cfa-20260503-adr019-parallel-encode/research/multi-source-synthesis.md` (605 lines, deep-researcher, 2026-05-03) — Apple BPG verbatim citations + MLX framework PR #1864 + Metal 4 WWDC 2025 + cross-impl comparison (llama.cpp vs MLX vs PyTorch MPS vs tinygrad) + 4-candidate design space evaluation + 7 open questions.

### Supporting measurement evidence

- iter88a per-kernel peer comparison: `/tmp/cfa-iter88a/COMPARISON.md` (206 lines, 2026-05-02 ~22:00Z) — establishes the load-bearing finding that mlx-native MoE FFN gate/up is FASTER than peer; only FFN-down lags by 86 ms; remaining 530 ms is encoder/orchestration.
- iter89b kernel audit: `/tmp/cfa-iter89b/research/AUDIT.md` — `kernel_mul_mm_id_q4_0_f32` byte-equivalent ISA at production shape (m=2048, k=512).
- iter89d MSL diff: `/tmp/cfa-iter89d/research/MSL-DIFF.md` — byte-identical metallib + xctrace 577 ms residual.
- ADR-013 P21 closure: `project_adr013_p21_stage4_canonical_2026_05_02.md` (MEMORY) + `docs/ADR-013-qwen35-inference.md:2912-3011` — sync_count 161→6, pp80 199→582 t/s.

### Apple primary sources

- Metal Best Practices Guide — Command Buffers ([archive](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html)) — verbatim "preferably one CB per frame" guidance.
- Metal Programming Guide — Command Submission ([archive](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Cmd-Submiss/Cmd-Submiss.html)).
- MTLCommandBuffer ([dev portal](https://developer.apple.com/documentation/metal/mtlcommandbuffer)) — documented `enqueue` / `commit` ordering contract.
- About synchronization events ([dev portal](https://developer.apple.com/documentation/metal/about-synchronization-events)) — MTLEvent semantics.
- MTLResidencySet ([dev portal](https://developer.apple.com/documentation/metal/mtlresidencyset)).
- WWDC 2025 Session 205 (Discover Metal 4) — `MTL4CommandBuffer` decoupling.
- WWDC 2025 Session 298 (Explore LLM on Apple silicon with MLX).

### Peer implementation sources

- llama.cpp ggml-metal: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m`, `ggml-metal-ops.cpp`, `ggml-metal-device.m`, `ggml-metal-common.cpp`, `ggml-metal.cpp` (verified at master 2026-05-02 by researcher-llama).
- llama.cpp PR [#11427](https://github.com/ggml-org/llama.cpp/pull/11427) (residency sets — 250 ms faster requests on M2 Ultra 7B Q8_0).
- llama.cpp commit `0320ac526` (PR #15995, ggerganov 2025-09-17) — `ggml_mem_ranges_t` (inspiration for mlx-native `mem_ranges.rs`).
- llama.cpp PR [#16634](https://github.com/ggml-org/llama.cpp/pull/16634) — initial Metal 4 tensor API support.
- MLX PR [ml-explore/mlx#1864](https://github.com/ml-explore/mlx/pull/1864) (awni 2025) — dynamic ops-per-buffer batching, 20/40/50 by chip.
- MLX `MLXLockContext` (DeepWiki) — global RLock per backend confirms NO parallel-encode in MLX.

### hf2q internal sources

- ADR-013: `/opt/hf2q/docs/ADR-013-qwen35-inference.md` — P19 H9 (1.32 ms commit floor); P21 stages 1-4; iter58b regression entry.
- ADR-015: `/opt/hf2q/docs/ADR-015-mlx-native-single-cb-decode.md` — iter88a-89e measurement chain.
- mlx-native: `/opt/mlx-native/src/encoder.rs` (2046 LoC), `buffer.rs` (367), `residency.rs` (344), `device.rs` (204), `kernel_registry.rs` (1069), `graph.rs` (1970).
- hf2q forward path: `/opt/hf2q/src/inference/models/qwen35/forward_gpu.rs` (4884), `gpu_full_attn.rs` (3416), `gpu_delta_net.rs` (4781), `gpu_ffn.rs` (3855).
- 7 caller-owned arenas: `dense_ffn_arena.rs` (656), `moe.rs`, `dn_prefill_arena.rs` (691), `chunk_allocs_arena.rs` (471), `fa_projections_arena.rs` (594), `fa_prefill_arena.rs` (426), and mlx-native iter83 `ChunkInternalArena`.
- Memory-bank entries: `solution_mlx_native_residency_lifetime_race.md`, `feedback_codex_review_catches_unified_memory_races`, `project_w5b22_hf2q_exhausted_remaining_in_mul_mm_id`, `project_w5b23_audit_falsifies_w5b22_kernel_attribution`, `project_qwen36_perf_gap_is_full_attention`, `project_adr013_p21_stage4_canonical_2026_05_02`.

### Cross-references

- ARC overhead: [floooh.github.io/2016/01/14/metal-arc.html](https://floooh.github.io/2016/01/14/metal-arc.html) — 20-40% CPU time on iPad Mini 4 in retain/release without unretained CBs.
- PyTorch MPS backend: [docs.pytorch.org/serve/hardware_support/apple_silicon_support](https://docs.pytorch.org/serve/hardware_support/apple_silicon_support.html).
- tinygrad scheduler: [docs.tinygrad.org/developer/developer/](https://docs.tinygrad.org/developer/developer/) — naive 1-CB-per-kernel; not a perf reference.

### Assumptions explicitly flagged "from training data, not verified at this session"

- MTLEvent / MTLFence cost numbers ("µs-class") — Apple does not publish per-call costs. Stable ML-systems community consensus, but not measured on M5 Max. **Action item:** Phase 0 microbench validates (OQ-2).
- MTL4CommandAllocator / long-lived CB ergonomics — based on WWDC 25 session description; we have not run M5 Max code against MTL4 APIs at this session.
- 20-40% ARC overhead figure: 2016 iPad Mini 4 measurement; recent M-series likely 3-10%.
- MLX team's design rationale ("they chose batch-larger over encode-parallel") — inferred from PR shape and global-RLock evidence; not from a published design doc.

---

## Changelog

### 2026-05-03 — ADR-019 v1.1 (post-Codex Phase-2b review)

**Codex Phase-2b review verdict:** `request_changes / med severity / either_defensible on D3-vs-D4 / acceptance_criteria=concrete / fence_coverage=comprehensive / d2_rejection=consistent_with_user_directive`. Receipt: `/tmp/cfa-20260503-adr019-parallel-encode/codex-review-last.txt`.

Revisions applied per Codex review:
1. **D4 verdict softened** from "REJECTED" to "DEFERRED — pending Phase 0a.2 microprototype falsification gate" (Codex issue 1)
2. **Performance claim reconciled**: D3 standalone projects ~30-60 ms; AC-P1's 80 ms target requires Phase 3a + 3b combined. Table updated accordingly. (Codex issue 2)
3. **CB-count reconciled**: D3 actual = 8-9 CBs (1 embedding + 1 attn + 5-6 FFN K-boundaries + 1 output); not 4-6 as initially stated. AC-P6 set to ≤ 8. (Codex issue 3)
4. **Phase 3a partial-ship clarified**: Phase 3a CAN merge behind env-gate without ADR closure; full closure requires Phase 3b OR D4 microprototype re-route. (Codex issue 4)
5. **Phase 0a NEW** — promoted from OQ-1/OQ-2 to pre-implementation gate: xctrace residual attribution + D4 microprototype falsification + MTLEvent cost calibration. ~2 calendar weeks before any structural code lands. (Codex issue 5)

### 2026-05-03 — ADR-019 v1.0 (initial CFA synthesis)

- Initial draft synthesizing 3 research artifacts (researcher-llama, researcher-mlx, deep-researcher) + 2 extended-research artifacts (apple-mlx-metal-2026, arxiv-2026-papers).
- Status: Proposed.
- Decision: D3 (per-stage-fence) primary + D1 (single-CB) second-stage; D2 + D4 explicitly rejected.
- 16 Acceptance Criteria pre-registered.
- 6-phase migration plan (Phase 0-5 in scope; Phase 6 D1 future).
- 18 implementation tasks (iter89e2-A through iter89e2-R).
- 10 Open Questions (8 from research synthesis + 2 surfaced during architect synthesis).
- ETA: 30-48 man-days single-developer; 15-24 man-days /cfa dual-mode.

**Pending operator approval before iter89e2-A starts.** Codex Phase-2b review of this ADR is the gating cycle; ADR-019 status flips Proposed → Accepted on Codex sign-off + operator authorization.

---

**End of ADR-019.**
