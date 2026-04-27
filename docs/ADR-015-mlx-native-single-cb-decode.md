# ADR-015: mlx-native — general decode-path speed improvements (qwen35 + gemma)

- **Status:** Proposed (Phase 1 single-CB hypothesis revised by P2 calibration — see Diagnosis update §2026-04-26)
- **Date:** 2026-04-26 (initial); revised same-day after P2 empirical data
- **Authors:** Robert E. Lee + Claude Code
- **Successor of:** ADR-012 §Optimize / Task #15 (closed at 0.94× of llama.cpp; cited the need for a "new mlx-native perf ADR")
- **Siblings of:** ADR-013 (qwen35 inference — owns the qwen35 forward path being rewritten); ADR-006 (mlx-native GPU backend — owns the Gemma `forward_decode` path being rewritten)
- **Standing requirement:** "as fast as our peers" applies to **every shipped model family** — `feedback_shippability_standing_directive`, restated 2026-04-26: *"we need this coherence and speed for qwen and gemma families of models"*. ADR-015 covers both.

## Context

ADR-012 closed the qwen35moe **conversion** scope at 0.94× of llama.cpp on
`dwq46` `n=256` decode (109.1 vs 116.0 t/s, M5 Max, same-day median per
`project_end_gate_reality_check`). Per-CB GPU time matches llama.cpp; the
gap is **CPU-side encoding/scheduling overhead**.

### The structural diagnosis

| Path | CBs / decode token | CPU encode overhead at ~5µs/CB |
|---|---:|---:|
| **llama.cpp** (`ggml-metal-context.m:458`) | 1–2 | ~5–10 µs |
| **hf2q Gemma** (`forward_mlx.rs:1309` — "one GraphSession per layer (30/pass)") | ~30 | ~150 µs |
| **hf2q qwen35** (95 `device.command_encoder()` call-sites in `inference/models/qwen35/`) | ~102 (measured `fa3f9d6`) | ~510 µs |

llama.cpp's own measurement: *"tests on M1 Pro and M2 Ultra using LLaMA
models, show that optimal values for n_cb are 1 or 2"* (`ggml-metal-context.m:458`).
Their `dispatch_apply` parallel-encoding mechanism (`:550`) targets
**prefill**, where `n_cb` may be larger; for decode, `n_cb=1` is fastest.

### Why prior CB-fusion attempts moved the needle by noise

`fa3f9d6` (small-scope CB fusion, FullAttn `sdpa+ops6-7` → 1 CB) recovered
~40 µs/token at +0.5% within noise — exactly the slope predicted by ~5 µs/CB:
the experiment removed 10 CBs / token. That data point **confirms the
theory** at one decimal place; it just couldn't statistically distinguish
+0.5% from zero at n=256 with 3 cold runs. Removing all ~100 CBs (going
to 1–2) projects to ~500 µs / token saved ≈ the entire 6% deficit.

### Why dispatch_apply alone isn't the lever for decode

`dispatch_apply` parallelizes encoding **across CPU threads**; CB count
**stays the same**. To recover 0.5 ms / token of encode overhead the CB
count itself has to drop by ~50× — not the encode wall-clock per CB.
The lever for decode is **single-CB forward pass**, not parallel encoding.

(Prefill is a separate axis — see Non-Goals.)

### 2026-04-26 — DIAGNOSIS UPDATE FROM P2 CALIBRATION

The working assumption above (~5 µs/CB) was a citation, not a measurement.
P2 (mlx-native `examples/cb_cost_calibration.rs`) measured it directly on
M5 Max:

| N    | regime     | wall_ms (median of 5) | µs/CB |
|-----:|:-----------|----------------------:|------:|
|   10 | async      |                 0.056 |  5.61 |
|   10 | sync       |                 0.154 | 15.39 |
|   10 | alloc+cmit |                 0.041 |  4.05 |
|  100 | async      |                 0.174 |  1.74 |
|  100 | sync       |                 1.694 | 16.94 |
|  100 | alloc+cmit |                 0.179 |  1.79 |
|  500 | async      |                 0.858 |  1.72 |
|  500 | sync       |                 6.688 | 13.38 |
|  500 | alloc+cmit |                 0.826 |  1.65 |
| 1000 | async      |                 1.583 |  1.58 |
| 1000 | sync       |                13.484 | 13.48 |
| 1000 | alloc+cmit |                 1.617 |  1.62 |

The **production hot path is `async`** (`enc.commit()` per layer, single
terminal `commit_and_wait` for argmax download).  Steady-state
**async µs/CB ≈ 1.6 µs** at N≥100 — **3.1× lower than the working
assumption**.

**Re-derived budget for hf2q qwen35 decode:**

| Component | CBs | µs/CB | total |
|---|---:|---:|---:|
| hf2q qwen35 (102 CBs) | 102 | 1.6 | **163 µs** |
| llama.cpp (1–2 CBs) | 1.5 | 1.6 | **2 µs** |
| **CB-overhead gap** | | | **~160 µs** |

Measured hf2q→llama.cpp gap = **~500 µs / token** (`fa3f9d6` /
ADR-012 §Optimize).  CB-count alone explains **~32%** of the gap, not the
~100% the original 5 µs/CB working assumption implied.

**Implication: single-CB rewrite is necessary but insufficient.**  Even a
perfect 102→1 CB collapse leaves ~340 µs / token (~68% of the gap)
unexplained.  The remaining cost likely lives in **per-dispatch
encoding** (hf2q decode issues ~1070 dispatches/token vs llama.cpp's
similar count, but with a different per-dispatch cost profile) and/or
**Rust-side orchestration overhead** (helper-function indirection,
buffer-pool acquisition, barrier bookkeeping).

ADR-015 phasing is updated below: **P3a** measures per-dispatch cost
directly before P3 commits to the single-CB rewrite, so the upper bound
on the rewrite's win is established empirically before multi-day
implementation work is spent.

### 2026-04-26 — P3a CALIBRATION DATA

`mlx-native/examples/dispatch_cost_calibration.rs` measures per-dispatch
encode cost using `scalar_mul_bf16` on a 1-element BF16 buffer (minimal
GPU work; CPU encoding cost dominates):

| N        | encode_ms (med of 5) | commit_ms (med of 5) | µs/dispatch |
|---------:|---------------------:|---------------------:|------------:|
|       10 |                0.006 |                0.189 |       0.629 |
|       50 |                0.012 |                0.589 |       0.249 |
|      100 |                0.018 |                0.198 |       0.185 |
|      500 |                0.084 |                0.345 |       0.168 |
|     1000 |                0.163 |                0.515 |       0.163 |
|     5000 |                0.784 |                1.575 |       0.157 |

**Steady-state µs/dispatch ≈ 0.16 µs** at N≥500 — within ±15% of
llama.cpp's implied ~0.14 µs / dispatch (per ADR-012 §Optimize ~150 µs
CPU encode for ~1070 dispatches).  hf2q's shader-launch path is **not
materially slower** than llama.cpp's.

### Budget reconciliation (P2 + P3a synthesized)

| Component | hf2q | llama.cpp | Δ |
|---|---:|---:|---:|
| **GPU work** (per ADR-012 §Optimize) | 8.45 ms | 8.38 ms | **+70 µs** |
| **CPU CB encode** (102 vs 1.5 CBs × 1.6 µs/CB) | 163 µs | 2 µs | **+161 µs** |
| **CPU dispatch encode** (~1070 dispatches × 0.16 vs 0.14 µs) | 171 µs | 150 µs | **+21 µs** |
| **Sub-total accountable** | | | **+252 µs** |
| **Measured wall delta** (1/109.1 − 1/116.0 t/s) | 9.16 ms | 8.62 ms | **+540 µs** |
| **Residual unaccounted** | | | **+288 µs** |

**The biggest single lever is the residual ~288 µs (~53% of gap)**, not
the CB-encode delta (~30%).  The residual cannot be CB or dispatch
encoding — both are now measured.  Suspects (in priority order):

1. **Helper-function indirection** — every `apply_*` helper in
   `inference/models/qwen35/` does multiple short-lived allocations
   (`MlxBuffer` for params, scratch shapes) and Result wrapping.  ~95
   helpers per token × Rust function-prologue cost is plausibly
   double-digit µs each.
2. **Buffer pool acquisition** — `pooled_alloc_buffer` calls walk the
   in-use list and bucket size on every call.
3. **Memory barrier bookkeeping** — `enc.memory_barrier()` mutates
   `pending_reads`/`pending_writes` Vecs (capture mode + RAW/WAR
   tracking).
4. **KV-cache cursor updates** — `slot.current_len[0] += 1` per
   FullAttn / DeltaNet layer with the associated buffer fence.

The **lever for closing the residual** is a Rust-orchestration sweep
(profile guided), not architecture.  Single-CB still helps because it
removes 100 of the encoder lifetimes plus barrier-bookkeeping
state-machine transitions, but the orchestration sweep is the bigger
chunk.

### Phase plan implication

Original P3 (qwen35 single-CB rewrite) recovers ~30% of the gap.
**A new P3b (Rust-orchestration profiling sweep) is added** that targets
the ~53% residual.  P3 and P3b can run in parallel; the P5 bench gate
requires both wins to clear ≥1.00× of llama.cpp.

## Decision

**D1.** Migrate **both** `qwen35` and `gemma` decode forward paths to a
**single `GraphSession`** (1 CB per forward pass), mirroring llama.cpp's
`encode_async` (`ggml-metal-context.m:676–722`) — one logical command
buffer encoded with explicit `encoder.memory_barrier()` between dependent
op stages.  Order: `qwen35` first (proves the pattern on the harder MoE
+ DeltaNet shape), `gemma` second (port the established pattern to the
dense-only path).

**D2.** Keep the existing per-layer-encoder paths as the legacy
implementation behind a single env var `HF2Q_LEGACY_PER_LAYER_CB=1` (covers
both families).  After a 7-day soak + sourdough-pass on each family's new
path, delete the legacy path for that family entirely (per
`feedback_no_broken_windows`).

**D3.** Bit-exact parity gate vs the current decode entry points is
mandatory **per family** (within 1e-5 max-abs logit error on each
family's smoke set).  No fallback path is shipped — if parity fails, the
lever is wrong; fix it, don't gate the primary path off (per
`feedback_never_ship_fallback_without_rootcause`).

**D4.** Exit criteria — **both** families must clear the bar:
  - **qwen35:** `dwq46` `n=256` decode median **≥ 1.00× of llama.cpp**
    same-day median (3 cold runs each, M5 Max, cold SoC per
    `feedback_perf_gate_thermal_methodology`).  Reference baseline from
    ADR-012 closure: hf2q 109.1 t/s vs llama.cpp 116.0 t/s = **0.94×**.
  - **gemma:** `gemma-4-26B-A4B-it-ara-abliterated-dwq` `n=256` decode
    median **≥ 1.00× of llama.cpp** same-day median (same methodology).
    Reference baseline must be re-measured at P0 (no committed value
    on file as of 2026-04-26 — see Open Question Q1).

## Phasing

| Phase | Deliverable | Definition of done |
|---|---|---|
| **P0 — Gemma baseline** | Same-day `gemma-4-26B-A4B-it-ara-abliterated-dwq` `n=256` decode median: 3 cold runs of `hf2q` + 3 of `llama-bench`. Establishes the gemma-side ratio. | Numbers recorded in this ADR as the baseline gemma-family bar. |
| **P1 — Audit** | Map every cross-CB synchronization point in `forward_gpu` (qwen35) and `forward_decode` (Gemma) paths. List of `(commit/commit_and_wait → next encoder)` transitions and the buffer dependencies that justify each one. Output: a markdown table per family in this ADR. | Every CB boundary in each family's live path is annotated with the data dependency it protects, OR documented as legacy / removable. |
| **P2 — Empty-CB cost calibration** | ✅ **Done 2026-04-26** — `mlx-native/examples/cb_cost_calibration.rs` measures async/sync/alloc-only µs/CB on M5 Max.  Result: **async µs/CB ≈ 1.6 µs at N≥100, NOT ~5 µs**.  Single-CB upper bound = ~160 µs / token of the ~500 µs gap (~32%). | Numbers published in §Diagnosis update above. |
| **P3a — Per-dispatch cost calibration** | ✅ **Done 2026-04-26** — `mlx-native/examples/dispatch_cost_calibration.rs` measures `scalar_mul_bf16` encode cost.  Result: **µs/dispatch ≈ 0.16 µs** at N≥500, within ±15% of llama.cpp's implied 0.14 µs/dispatch.  Shader-launch path is competitive; lever is Rust orchestration. | Numbers published in §P3a calibration data above. |
| **P3 — Single-CB rewrite** (qwen35 first, gemma second) | New `forward_gpu_single_cb` in `inference/models/qwen35/forward_gpu.rs` and `forward_decode_single_cb` in `serve/forward_mlx.rs`.  Each: one `GraphSession`, all dispatches encoded, explicit `memory_barrier()` for cross-stage dependencies, single `commit()` at end. | Builds in release.  No new `unsafe`.  Recovers ~30% of the 540 µs gap (~161 µs). |
| **P3b — Rust orchestration sweep** ⬅ **NEW (parallel to P3)** | Profile-guided reduction of the ~288 µs residual from helper-function indirection, buffer pool acquisition, barrier bookkeeping, KV-cursor updates.  Use `cargo flamegraph` or `instruments -t TimeProfiler` on a hot decode loop.  Target: shrink the residual by ≥50% (~140 µs). | At least 5 profile-driven specific reductions landed (each cited with profile evidence in the commit message).  Combined with P3, restores ≥0.99× ratio (one cold-run iteration). |
| **P5 — Parity gate (per family)** | 16-prompt smoke on each family's new path produces logits within 1e-5 max-abs of legacy path. | Logged in commit message. |
| **P6 — Bench gate (per family)** | 3 cold runs of `hf2q --benchmark` on `dwq46` and `gemma-4-26B-A4B-it-ara-abliterated-dwq` at `n=256` (cold SoC).  Same-day `llama-bench -p 0 -n 256 -r 3` on the same model.  Ratio computed per family. | Ratio ≥ 1.00× recorded for **both** families in ADR + commit message. |
| **P7 — Default cut-over** | `HF2Q_LEGACY_PER_LAYER_CB=1` is the only way back to legacy.  Default = single-CB on both families. | Smoke + bench gate green on default-default settings. |
| **P8 — Soak + delete legacy** | 7-day window, no regressions.  Then delete `forward_gpu_greedy` (qwen35 legacy) and the per-layer-GraphSession Gemma legacy entirely. | Diff-stat ≥ 1000 LOC removed across both families. |

## Open questions / risks

| ID | Risk / open question | Mitigation |
|---|---|---|
| **R1** | Barrier correctness: single-CB requires every cross-stage dependency expressed as `enc.memory_barrier()`. A missed barrier = silent corruption. | P4 bit-exact parity gate is mandatory. Spot-check with `mlx_native::ops::memory_barrier` debug logging during P3. |
| **R2** | Encoder lifetime + buffer pool: a long-lived encoder may hold buffer refs across the full forward pass, blocking pool reuse and inflating peak working set. | P3 measures pool stats; if `pool.alloc_count` > pre-ADR + 10%, size the pool up or refactor to release. |
| **R3** | `MLX_PROFILE_CB` per-label timing relies on CB granularity. With 1 CB / token, per-stage attribution is gone. | Switch to per-dispatch labels (`commit_labeled` becomes `dispatch_labeled` — already partially supported). Document the change in the P3 commit. |
| **R4** | The 6% gap may not be 100% encode overhead. If P2 shows µs/CB << 5, single-CB wins shrink. | P2 is the cost calibration. If the upper bound drops below ~3% headroom, re-scope to a different lever (Option B below) before P3. |
| **R5** | MoE expert dispatch is the largest contributor to dispatch count (256 experts × N selected per layer). Single-CB doesn't compress dispatch count, only CB count. If dispatch overhead dominates encode cost, single-CB still wins (one `commit` not 100), but the CPU encode cost itself doesn't shrink. | P2 also measures per-dispatch encode cost (encode N noop dispatches into 1 CB). If per-dispatch dominates, the lever is fewer dispatches, not fewer CBs — branch to a separate ADR (e.g., persistent-MoE-router or shader-side expert loop). |
| **R6** | Decode-only pivot: prefill is unchanged. Long-prefill parity inversion (`project_long_prefill_parity_inverts`) remains open. | Out of scope for ADR-015 — covered by Non-Goal 1. |

## Non-goals

1. **Prefill parity** — `pp≥1024` parity inversion at `-ub 512` matched batching is a separate axis, owned by a future ADR. ADR-015 measures decode only.
2. **Speculative / MTP decode** — ADR-013 P14 territory.
3. **Sampler upgrades beyond greedy** — ADR-008 placement decision stands; sampling lives in hf2q `sampler_pure.rs`.

## Alternatives considered

- **Option A: dispatch_apply parallel encoding (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:550`).** *Rejected for decode* — llama.cpp's own data shows `n_cb=1` is optimal for decode (`:458`). dispatch_apply is a prefill mechanism. Would still help our prefill — but per Non-Goal 1, that's a separate ADR. Not committed in this iteration.
- **Option B: cross-token speculative pipelining (draft + verify decode).** *Deferred* — substantial program-structure change. Becomes attractive only if Option A (single-CB) hits the wall before 1.00×. Land in a successor ADR if needed.
- **Option C: source-level kernel fusion (combine adjacent compute kernels into one shader).** *Falsified 9× in the M5 Max static-evidence kernel-hypothesis register* — most recent: `swiglu` Q4_0 `mv_id` (mlx-native@`4efeec0`, hf2q@`502364d`, −1.5%). The Metal compiler hoists what llama.cpp hand-tunes (`project_metal_compiler_auto_optimizes_static_levers`). Not a productive lever on M5 Max for current workloads.
- **Option D: keep status quo at 0.94×.** *Rejected.* Per `feedback_shippability_standing_directive` and `feedback_no_shortcuts`, "as fast as our peers" is the standing exit bar. 0.94× is not shippable.

## Literature foundations

The architectural choices below draw from the following primary
sources, each fetched and read directly via WebFetch against
arxiv.org.  Each citation is verified pre-training-cutoff
(no Perplexity-summary or post-cutoff claims included).  Most LLM
inference research is CUDA-centric; the §Apple-Silicon-applicability
note after each grouping calls out where Metal / unified-memory
porting changes the equation.

### Kernel-level performance — informs P3 inner loop kernel correctness + adjacent fusion candidates

- **Dao et al. (2022).** *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* arXiv:2205.14135.  Establishes IO-awareness — accounting for reads/writes between GPU memory tiers — as the canonical principle behind fast attention.  Tiling reduces HBM↔SRAM traffic, achieving up to 3× speedup on GPT-2.  Apple Silicon applicability: principle is universal; M-series GPU's unified memory removes the HBM/SRAM dichotomy but preserves the L2/threadgroup-memory hierarchy.  mlx-native already implements `flash_attn_prefill` + `flash_attn_vec` analogues with M5-tuned block sizes.
- **Dao (2023).** *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv:2307.08691.  Refines work partitioning across thread blocks and warps; reaches 50–73 % of A100 theoretical FLOPs/s.  Apple Silicon applicability: NVIDIA-warp-specific tactics (e.g. `swizzle` patterns) translate imperfectly to Metal SIMD-groups; the *parallel-decomposition* insight (single-head splits across thread blocks for occupancy) maps directly and is already echoed in our MoE-routed mat-vec kernels.

### System architecture — informs P3b orchestration sweep, KV-cache management

- **Kwon et al. (2023).** *Efficient Memory Management for Large Language Model Serving with PagedAttention.* arXiv:2309.06180 (SOSP 2023, **vLLM**).  Treats KV cache as paged virtual memory; achieves "near-zero waste" + 2–4× throughput vs FasterTransformer / Orca.  Apple Silicon applicability: portable.  `mlx_native::ops::kv_cache_copy` + the per-slot abstraction in `inference/models/qwen35/kv_cache.rs` is a related but coarser allocation strategy.  PagedAttention-style block reuse remains an open lever for our serve path (orthogonal to ADR-015 decode-perf scope).
- **Zheng et al. (2023).** *SGLang: Efficient Execution of Structured Language Model Programs.* arXiv:2312.07104.  Introduces `RadixAttention` for KV reuse across requests + compressed FSMs for structured output; up to 6.4× throughput vs prior systems.  Applicability: cross-request KV reuse is a serve-side concern (ADR-005 territory), not single-request decode latency.
- **Agrawal et al. (2024).** *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* arXiv:2403.02310.  Chunked-prefill scheduling: splits prefill into near-equal chunks so the queue admits new requests without pausing ongoing decodes; 2.6× higher serving capacity vs vLLM on Mistral-7B.  Applicability: directly relevant to our `project_long_prefill_parity_inverts` open question (Non-Goal 1 today, future ADR).  Confirms prefill chunking is a multiplier orthogonal to single-CB decode.

### Decode acceleration alternatives — informs §Alternatives Option B (speculative pipelining)

- **Leviathan et al. (2022).** *Fast Inference from Transformers via Speculative Decoding.* arXiv:2211.17192 (ICML 2023 Oral).  Foundational paper: small draft model proposes K tokens; large model verifies in parallel.  2–3× speedup with identical output distribution.  Applicability: portable; the verification-step parallelism is what mlx-native would need to support batched K-token forward.
- **Cai et al. (2024).** *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* arXiv:2401.10774.  Avoids the separate-draft-model burden by adding multiple decoding heads to the existing model + tree-attention verification.  2.2–3.6× speedup.  Applicability: `Qwen3.5-MoE`'s **MTP** (multi-token prediction) head per `convert_hf_to_gguf.py:Qwen3NextModel` (ADR-012 Decision 11) is precisely a Medusa-shaped feature; ADR-013 P14 owns the inference-side execution.  ADR-015 P3 single-CB rewrite is therefore *complementary* to Medusa-style decode — they compose multiplicatively.
- **Li et al. (2024).** *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* arXiv:2401.15077.  Operates at the second-to-top-layer feature level rather than token level; 2.7–3.5× speedup on LLaMA2-Chat 70B.  Applicability: portable; informs the design choice of *what* to predict in our future MTP path.
- **Fu et al. (2024).** *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.* arXiv:2402.02057.  *Exact, parallel* decoding without any auxiliary draft model — Jacobi-iteration style.  Up to 1.8× MT-Bench, 4× with multi-GPU.  Applicability: M5 Max single-device, so the multi-GPU number doesn't apply, but the no-draft-model property is attractive given how much surface a draft-model path adds.

### Graph compilation — informs the single-CB rewrite framing (P3 architecture)

- **Chen et al. (2018).** *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.* arXiv:1802.04799 (OSDI 2018).  Establishes the graph-level + operator-level optimization layering that lets a single high-level forward pass compile to a hardware-specific dispatch sequence.  Applicability: TVM has limited Metal support; the *layering principle* informs whether mlx-native's `GraphSession` should grow toward graph-compile semantics or stay eager-with-explicit-barriers.  ADR-015 P3 takes the latter (explicit barriers within one encoder); a successor ADR could revisit the graph-compile path.
- **Ding et al. (2023).** *Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs.* arXiv:2210.09603 (ASPLOS 2023).  Embeds scheduling directly into tensor programs via "task mappings"; introduces post-scheduling fusion that automates operator fusion after individual scheduling.  1.48× over ONNX Runtime / TVM-AutoTVM.  Applicability: post-scheduling fusion is the formal version of what our 9× falsified manual-fusion experiments were trying to discover empirically.  The data point that automated fusion still struggles to beat hand-tuned BLAS suggests that *simple* compile-time fusion isn't the lever for our 0.94×→1.00× workload — it would need to capture the *runtime* memory pattern that the M5 Max compiler already auto-hoists (per `project_metal_compiler_auto_optimizes_static_levers`).

### Quantization lineage — DWQ-aware framing (informs which kernel-level levers move the perf needle on our actual workload)

We don't ship uniform-precision Q4 / Q8 / FP16; we ship MLX-flavored
**DWQ** (Dynamic Weight Quantization) — *activation-calibrated mixed
precision* with sensitive layers promoted to higher bits per
`docs/ADR-012-qwen35moe-conversion.md` (default mix: 4-bit base / 6-bit
sensitive = `dwq46`).  The DWQ algorithm itself has no formal arxiv
paper that I could verify pre-cutoff; the lineage of techniques DWQ
synthesizes is well-documented:

- **Dettmers et al. (2022).** *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* arXiv:2208.07339 (NeurIPS 2022).  Identifies *emergent activation outliers* dominating transformer predictive performance; introduces vector-wise W8A8 + a mixed-precision decomposition isolating outlier dimensions into 16-bit while >99.9% of values run in 8-bit.  This is the **conceptual ancestor** of DWQ's "promote sensitive tensors to higher bits" rule — the difference is granularity (LLM.int8 is per-dimension; DWQ is per-tensor / per-expert).
- **Frantar et al. (2022).** *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323 (ICLR 2023).  Second-order one-shot PTQ; 175B → 3–4 bits in ~4 GPU-hours, negligible accuracy loss.  Foundational for "calibration-driven post-training quant" — DWQ inherits the calibration-set + per-tensor-error-measurement methodology.
- **Xiao et al. (2022).** *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* arXiv:2211.10438 (ICML 2023).  "Weights are easy to quantize while activations are not" → migrate quantization difficulty from activations to weights via mathematically equivalent transformation.  Informs why DWQ does **not** quantize activations: the math says you don't have to.  Confirms our W-only quantization path.
- **Dettmers et al. (2023).** *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314.  Introduces NF4 (4-bit NormalFloat) data type — "information-theoretically optimal for normally-distributed weights" — plus double-quantization of quantization constants.  Applicability: NF4's data-type choice is *adjacent* to but not the same as the GGUF Q4 / Q8 nibble layouts we ship.  An NF4-style data type shipped through GGUF + DWQ-mixed semantics is a future quant direction (out of ADR-015 scope; would belong in an ADR-014 successor).
- **Lin et al. (2023).** *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* arXiv:2306.00978.  *Channel-wise* salience: protecting just 1 % of salient weight channels (identified via activation distribution, not weight magnitude) substantially reduces quantization error.  AWQ avoids hardware-inefficient mixed precision by *scaling* salient channels.  Compare to DWQ which *is* mixed precision — AWQ takes the opposite tradeoff.  Both rely on activations identifying the salient set.  Tactical: AWQ-style channel-scaling could be a complementary post-DWQ layer for the 6-bit experts we already promote; out of P3 scope.
- **Kim et al. (2023).** *SqueezeLLM: Dense-and-Sparse Quantization.* arXiv:2306.07629 (ICML 2024).  Most directly DWQ-shaped of the lineage — *"sensitivity-based non-uniform quantization, which searches for the optimal bit precision assignment based on second-order information"*.  This is functionally what DWQ-mixed-4-6 does at a coarser granularity (per-tensor, not per-weight-cluster).  **Key framing for ADR-015**: the SqueezeLLM abstract declares *"the main bottleneck for generative inference with LLMs is memory bandwidth, rather than compute, specifically for single batch inference."*  This is the canonical statement of why decode is bandwidth-bound — and why per-CB / per-dispatch overhead (P2 + P3a) is the *secondary* axis after the kernel-level bandwidth efficiency that mlx-native's `quantized_matmul_*` family already targets.

### What the quantization literature says — and doesn't say — about our perf gap

**What it says:**
1. Decode is bandwidth-bound (SqueezeLLM, Lookahead Decoding, Medusa all converge on this).  Therefore the per-CB / per-dispatch CPU encoding overhead is not the *primary* axis — kernel bandwidth efficiency is.  ADR-015's measured numbers (P2: 1.6 µs/CB, P3a: 0.16 µs/dispatch) are consistent with this: even all of the encoding overhead removed leaves ~14 % of the gap as GPU work delta.
2. DWQ's per-tensor mixed-precision is the right scheme for quality (SqueezeLLM-style sensitivity-based assignment).  Switching schemes wouldn't close the perf gap — both hf2q and llama.cpp consume identical `dwq46` tensor data.

**What it does NOT say:**
1. The gap **cannot** come from DWQ-the-algorithm.  Both sides see the same Q4_K + Q6_K + Q8_0 tensor layouts in the same GGUF — only the kernel that decodes + multiplies differs.  Therefore: per-tensor-format kernel design (mlx-native's vs llama.cpp's `quantized_matmul_*` for each format involved in `dwq46`) is the relevant axis, not the quant scheme.
2. A DWQ-aware perf investigation should attribute per-CB GPU time to *which tensor format* dominates that CB — already partially recorded in ADR-012's per-CB profile (`f11cef9`).  An ADR-015 P3a' (DWQ-format-attribution micro-bench) is a candidate addition if P3 + P3b underdeliver.

### Apple Silicon literature gap — honest acknowledgement

Searches for arxiv work specifically on Apple Silicon LLM inference perf
returned nothing pre-cutoff that I could verify by direct fetch.  The
M-series GPU is academically under-studied compared to NVIDIA datacenter
parts.  This is itself a data point: bleeding-edge M5 Max optimization
has no published roadmap to copy from; the lever space we explore in
ADR-015 is one we have to discover empirically.  See P2 + P3a measured
data above.

## Why ADR-015 stays in mlx-native (and what coreml-native would be for)

We also own [`/opt/coreml-native`](https://github.com/robertelee78/coreml-native) — a sibling pure-Rust crate exposing CoreML inference (CPU/GPU/ANE).  ADR-015 stays in `mlx-native` rather than pivoting to CoreML for three concrete reasons grounded in our actual workload, not in framework preference:

1. **DWQ-mixed-4-6 tensor data is not a CoreML quant scheme.**  Both `hf2q` and `llama.cpp` consume identical `dwq46` GGUF tensors via their own Metal-GPU `quantized_matmul_*` kernels.  Routing through CoreML would force re-quantization through `coremltools`' API, losing our DWQ calibration and breaking the apples-to-apples bench against `llama.cpp`.
2. **MoE expert routing + Gated DeltaNet are not standard CoreML ops.**  CoreML's compiler maps a fixed op set to ANE / GPU / CPU; novel architectures (Qwen3.5-MoE's hybrid 3:1 DeltaNet:FullAttn, our 256-expert + custom router) fall back to CPU.  ANE acceleration evaporates exactly where our hot path lives.
3. **The 0.94× gap is not a runtime-choice problem.**  Per ADR-012 §Optimize, hf2q's GPU compute (8.45 ms) ≈ llama.cpp's (8.38 ms) — the gap is in CPU-side encoding/orchestration on **the same Metal substrate**.  Switching to CoreML wouldn't address that; it would replace the lever space with a different one whose ceilings on our model class are unknown and uncontrolled.

CoreML/ANE *is* the right tool for adjacent workloads where ADR-015 doesn't apply: standard ViT prefill (mmproj on Gemma-4V / Qwen vision), BERT-style embeddings (the `bert-test` target), Whisper if we ever pick that up.  The mlx-vs-CoreML strategic comparison for a possible Qwen3.5-MoE-on-ANE hybrid is its own ADR worth of work — see [ADR-016](./ADR-016-mlx-vs-coreml-strategic-comparison.md) (stub, 2026-04-26).

## Architectural implications given full-stack ownership

We own both `/opt/mlx-native` (kernel + dispatch substrate) and
`/opt/hf2q` (forward orchestration + serving).  That gives ADR-015
a strictly larger option space than any consumer of an upstream
inference framework:

1. **No API stability constraint.**  Single-CB rewrite can change
   `mlx_native::CommandEncoder` signatures freely; consumers are
   only us.
2. **Persistent / megakernel options open.**  An mlx-native ADR
   succeeding ADR-015 can investigate a persistent-thread Metal
   kernel that loops *internally* over decode tokens — the literature
   above (FlashAttention's tiling principle + Hidet's task-mapping
   compositionality) sketches the design space.  Single-CB is the
   conservative first lever; megakernel is the next-octave lever.
3. **Shader-level cross-token pipelining.**  Lookahead Decoding
   (2402.02057) and Medusa-style MTP (2401.10774) both decompose to
   "more dispatches per wall-clock token" — a regime where the per-
   dispatch overhead measured in P3a starts mattering more, but the
   speedup more than compensates.  ADR-013 P14 is the consumer side;
   mlx-native's role would be to expose efficient batched-K forward.
4. **The ground truth is what we measure.**  Per `feedback_ground_truth_is_what_we_can_measure_now`,
   any literature claim that doesn't survive a P2-style cold-SoC bench
   on M5 Max is irrelevant to shippability.  ADR-015 commits to
   measure-3x discipline at every phase gate.

## Capturing M5's GPU Neural Accelerators (Metal 4 TensorOps + NAX)

Source for this section: deep technical brief at `/tmp/cfa-adr016/research-metal4-tensorops.md` (also stored at memory key `swarm-cfa-adr016/deep-research/metal4-tensorops`, 2026-04-26).  All API signatures and file:line citations were verified against Apple developer docs, MLX source (`github.com/ml-explore/mlx`) and `/opt/mlx-native` source — no training-data recall used.

### The path Apple is publicly accelerating

Apple's M5 generation publishes a measured **3.33–4.06× faster TTFT vs M4** and **1.19–1.27× decode** across Qwen 1.7B / 8B / 14B / 30B-A3B-MoE / GPT OSS 20B at prompt=4096 / generation=128 ([Apple ML Research M5/MLX post](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)).  The 4× lives in **GPU-integrated Neural Accelerators (NAX)** — per-GPU-core matrix-multiplication units accessed via **Metal Performance Primitives (MPP) TensorOps** at the `metal4.0` shader standard.  M5 Max (40-core GPU) = 40 × 1,024 FP16 FMA / cycle.  This is the substrate `mlx-native` already targets — but only partially.  The 1.19–1.27× decode boost is bandwidth-bound (460 → 614 GB/s); ADR-015 P2 + P3 + P3b own that axis.  The TTFT axis is compute-bound and is the new lever introduced here.

The standalone 16-core ANE is **not** the path that captures Apple's headline numbers.  CoreML / ANE work is opportunistic encoder offload (ADR-016 territory), not LLM-body acceleration.

### mlx-native current state — gap analysis

`mlx-native` already uses `mpp::tensor_ops::matmul2d` in four `*_tensor.metal` GEMM kernels, with correct `get_destination_cooperative_tensor` output stores.  That part is shipping correctly.  Verified gaps vs MLX's NAX path:

| Lever | mlx-native today | MLX NAX path | Source |
|---|---|---|---|
| Inner MMA descriptor | `matmul2d_descriptor(NR1=32, NR0=64, NK=32, …)` | `matmul2d_descriptor(16, 32, 16, …)` | `quantized_matmul_mm_tensor.metal:185-231` vs `mlx/.../steel/gemm/nax.h` |
| MMA scope | `execution_simdgroups<4>` | `execution_simdgroup` (single, 32 lanes) | same |
| `reduced_precision` flag | `false` everywhere | not explicitly false in NAX path | `*_tensor.metal` files |
| Outer tile sizes (M5 Max) | NR0=64, NR1=32 | bm=64, bn=128 or 256, swizzle=2 | MLX `matmul.cpp` commit `0879a6a` |
| Morton/Hilbert dispatch order | absent (raster order) | swizzle=2 / Morton for large GEMM | grep mlx-native: zero matches |
| macOS 26.2 / arch-gen ≥ 17 runtime gate | absent (`probe_tensor_mm()` is M3+ compile probe only) | `is_nax_available()` at runtime | `quantized_matmul_ggml.rs:155-169` vs MLX `device.cpp` |
| Attention kernel TensorOps | **absent** — `flash_attn_prefill.metal:621-754` uses M1-era `simdgroup_matrix` + `simdgroup_multiply_accumulate` | `steel_attention_nax` (commit `b41b349`) | `flash_attn_prefill.metal:621-754` |
| Split-K NAX | absent | `steel_gemm_splitk_nax` for large K | grep mlx-native: zero matches |

Per ADR-015 §"Prefill kernel breakdown (measured)" memory pin: MoE 39 %, MM 25 %, FA 16 %.  The single biggest unrouted lever is **flash-attention prefill** (16 % of compute) running on the M1-era simdgroup path.  GEMM kernels are partially routed through TensorOps but with non-NAX-tuned tile sizes and no Morton-order dispatch.

### P3c — three concrete actions to capture the 3.3–4.1× TTFT speedup

**P3c is a new sub-phase orthogonal to P3 (single-CB rewrite) and P3b (Rust-orchestration sweep).**  All three are shader-internal changes; none affect command-buffer lifecycle, encoder lifetime, barrier bookkeeping, or Rust orchestration cost.

| Action | File | Today | Change | Expected lever | Risk |
|---|---|---|---|---|---|
| **P3c.1 — Morton/Z-order dispatch for large prefill GEMM** | `quantized_matmul_mm_tensor.metal`, `dense_mm_bf16_tensor.metal`, `dense_mm_f16_tensor.metal` | Standard 2D raster grid (`tgpig`) | Map flat dispatch index to Morton coordinates before tile-offset calc | **1.5–2× on large-K GEMM** (seq ≥ 1024); per Tech Talk 111432, alone is the difference between ~50 % and ~100 % NAX utilization on a 4K×4K matmul (~6× speedup demo) | Tile-boundary tail handling for non-power-of-2 `ne0`/`ne1`; ~20 lines per kernel, shader-only |
| **P3c.2 — NAX-tuned outer tile sizes + macOS 26.2 / arch-gen ≥ 17 runtime gate** | `quantized_matmul_ggml.rs` (gate logic), new `quantized_matmul_mm_nax.metal` instantiation | NR0=64, NR1=32 always; M3+ compile probe only | Add `is_nax_available_m5()` Rust gate (mirror MLX `device.cpp`); on M5+macOS 26.2, dispatch new variant with bm=64, bn=128, swizzle=2 | **10–20 % on prefill GEMM** | Existing tensor path stays as M3/M4 fallback; no breakage |
| **P3c.3 — Port `flash_attn_prefill` inner matmul from `simdgroup_matrix` to `mpp::tensor_ops::matmul2d` + Morton dispatch** | New `flash_attn_prefill_nax.metal` (gate behind `is_nax_available_m5()`); existing kept as M3/M4 fallback | Lines 621–754 use `metal::simdgroup_matrix<T>` + `simdgroup_multiply_accumulate` | Replace inner QK^T and scores×V matmul loops with `mpp::tensor_ops::matmul2d` (16×32×16 fragments per MLX NAX); preserve online-softmax tile state | **2–3× on TTFT for long prefill** (seq ≥ 2048); attention is 16 % of prefill compute — this is the largest single lever for the TTFT headline | Non-trivial port — `flash_attn_prefill` is a fully fused SDPA with online softmax, sliding-window masking, and partial-output reduction.  MLX's `steel_attention_nax` (`b41b349`) is the reference port. |

The 3.33–4.06× Apple-measured TTFT speedup is achievable only if **all three** actions land — GEMM Morton + tile-sizing alone covers ~64 % of prefill compute (MoE 39 % + MM 25 %), but the remaining 16 % attention prefill is the bottleneck if it stays on `simdgroup_matrix`.  P3c.3 is the highest-impact and highest-effort of the three.

**Optional P3c.4 — `reduced_precision = true` for BF16 prefill attention scores** (5–15 % additional lever on attention GEMM): permits internal FP16 accumulation on Neural Accelerators.  *Numerically risky* per memory pin `project_vit_attention_bf16_softmax_drift` (BF16 cast in attention K already causes ~0.68 logit perturbation on saturated-softmax winners).  Gate behind sourdough parity check (ADR-015 D3).  Do not enable for weight-dequant matmuls (Q4_0, Q6_K) — accumulation precision directly impacts model quality.

### OS / version constraints

| Requirement | Value | Source |
|---|---|---|
| Minimum macOS for NAX hardware activation | **26.2** | MLX `device.cpp` `is_nax_available()` |
| Metal shader standard | `-std=metal4.0` | MLX CMake, commit `54f1cc6` |
| GPU architecture generation | **≥ 17** (non-phone) / ≥ 18 (phone) | MLX `device.cpp` |
| M5 Max architecture char | `'s'` (Max) / `'c'` (Pro) / `'d'` (Ultra) | MLX `matmul.cpp` commit `0879a6a` |
| BF16 tensor support | macOS 26.1 | Apple Tech Talk 111432 |
| 4-bit / 8-bit integer tensors | macOS 26.4 | Apple Tech Talk 111432 |
| Cooperative tensors as matmul inputs | macOS 26.3 | Apple Tech Talk 111432 |
| `MLX_METAL_NO_NAX` escape hatch | env var, runtime force-fallback | MLX `device.cpp` |

**Hard implication:** M4 (gen 16) does NOT have NAX hardware.  The 3.3–4.1× TTFT speedup is exclusively M5+ + macOS 26.2+.  P3c gates must bottom out to the existing `*_tensor.metal` simdgroup-matrix path on M3/M4 and on M5 with macOS < 26.2.  ADR-015's bench methodology (`feedback_perf_gate_thermal_methodology`) already requires documenting the macOS version for every bench run; that documentation now becomes load-bearing — same kernels, different hardware activation.

### Composition with P3 + P3b — strict orthogonality

| Aspect | P3 (single-CB) | P3b (Rust orchestration sweep) | P3c (NAX kernels) |
|---|---|---|---|
| What changes | Command-buffer lifecycle (102 → 1 CB / token) | Helper-function indirection, buffer pool, barriers, KV cursor | Shader-internal MMA descriptor, Morton order, attention port |
| Where | Rust host-side `forward_gpu` / `forward_decode` | Rust host-side `apply_*` helpers | Metal shader `*_tensor.metal`, `flash_attn_prefill*.metal` |
| Affects | CB encode time, encoder lifetime | CPU-side per-dispatch overhead | GPU-side compute throughput |
| Composes with others how | Removes 100 of the 102 encoder lifetimes | Removes ~140 µs of the ~288 µs/token Rust residual | Multiplies prefill TTFT independently |

P3c is **fully orthogonal** to P3 and P3b.  Single-CB removes encoder boundaries; orchestration sweep removes CPU overhead; NAX kernels increase per-dispatch GPU throughput.  Order-of-operations recommendation: **land P3 + P3b first** (they target the existing 540 µs/token decode gap and clear ADR-015 D4 ≥ 1.00× of llama.cpp); **open P3c only after ADR-015 P6 bench gate clears**.  Reasoning: avoid stacking shader-level complexity on top of an unmeasured decode-path baseline.  Once decode is at 1.00×, P3c becomes a TTFT-only lever measured against Apple's own MLX numbers, not against llama.cpp.

P3c does NOT change the ~288 µs/token Rust-orchestration residual identified in ADR-015 §"Budget reconciliation".  However once P3c lands, faster TTFT compounds with faster decode multiplicatively at the user-perceived end-gate (time to complete a response): TTFT improvement + decode improvement = total response-time improvement.

### Open questions — to refine before P3c opens

| ID | Question | Resolution |
|---|---|---|
| **Q-NAX-1** | Does mlx-native's existing `matmul2d_descriptor(32, 64, 32, …)` already activate NAX hardware on M5 Max, or stay on simdgroup-matrix? | Profile via Instruments "Metal System Trace" + GPU counter `gpu.counters.neuralAcceleratorUtilization` (if exposed) on M5 Max with a prefill workload.  If activation is already happening, P3c.2 inner-tile changes are moot; P3c.1 (Morton) and P3c.3 (attention port) remain. |
| **Q-NAX-2** | What is the actual TTFT baseline for mlx-native vs mlx-lm on M5 Max at pp=4096?  ADR-015 D4 baseline is decode-only (n=256). | Measure before P3c sized.  Add TTFT bar to D4 exit criteria when P3c opens.  Reference: Apple's MLX/M5 post = 3.3–4.1× over M4 across Qwen 1.7B–30B-MoE. |
| **Q-NAX-3** | What macOS version is the M5 Max benchmark machine running? | Document explicitly in every cold-SoC bench log; NAX requires 26.2+.  If on 26.1 or 15, P3c lever is unavailable. |
| **Q-NAX-4** | Is `execution_simdgroups<4>` causing NAX stalls on long K-loops? | Profile with Instruments' Metal GPU Counter `ShaderOccupancy` and (if exposed) `NeuralAcceleratorUtil` on a 4096-token prefill.  If stalls are present, restructure to `execution_simdgroup` with 4× more threadgroups per launch. |
| **Q-NAX-5** | Does `swizzle=2` in MLX's M5 tuning mean Morton/Z-order, column-first, or something device-specific? | Read MLX `matmul.cpp` swizzle implementation directly before implementing P3c.1; `swizzle=0` is raster, `=1` is column-first per common convention but verify. |
| **Q-NAX-6** | Does `reduced_precision = false` affect decode-path throughput on M5? | Decode is bandwidth-bound (SqueezeLLM canonical statement, confirmed by ADR-015 P2 / P3a data).  For decode-path weight-dequant matmuls (m=1), compute is not the bottleneck.  The flag has no effect on bandwidth-bound ops — confirm empirically before spending time on precision-flag experiments for decode. |
| **Q-NAX-7** | Does porting `flash_attn_prefill` require a new file or modify the existing one? | Recommend: new `flash_attn_prefill_nax.metal` gated behind `is_nax_available_m5()`, keeping existing simdgroup path as M3/M4 fallback.  Two-variant pattern matches MLX's own approach. |

## References

- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:448-614` — main encode loop; `:550` `dispatch_apply`; `:458` "optimal n_cb is 1 or 2"
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:670-722` — `encode_async` block, the per-CB encoding worker
- `/opt/hf2q/docs/ADR-012-qwen35moe-conversion.md:215-256` — diagnosis + closure
- `/opt/hf2q/src/inference/models/qwen35/forward_gpu.rs` — current per-layer-encoder qwen35 forward (target of P3 rewrite)
- `/opt/hf2q/src/serve/forward_mlx.rs:1309` — Gemma's "one GraphSession per layer" baseline (30 CBs / pass)
- mlx-native: `mlx_native::GraphSession`, `mlx_native::CommandEncoder::memory_barrier()`
- mlx-native examples: `cb_cost_calibration.rs`, `dispatch_cost_calibration.rs` (P2 + P3a empirical benches)
- Memory pins: `feedback_perf_gate_thermal_methodology`, `feedback_shippability_standing_directive`, `feedback_never_ship_fallback_without_rootcause`, `feedback_no_broken_windows`, `project_metal_compiler_auto_optimizes_static_levers`, `project_end_gate_reality_check`, `feedback_ground_truth_is_what_we_can_measure_now`

## Changelog

- **2026-04-26 — Proposed (initial).** Diagnosis pivot from ADR-012 §Optimize: gap is CB-count, not CB-encode-time. Single-CB forward pass selected over `dispatch_apply` based on llama.cpp's own n_cb data.
- **2026-04-26 — Title broadened to "general decode-path speed improvements" + scope extended to gemma family** per standing directive *"we need this coherence and speed for qwen and gemma families of models"*.
- **2026-04-26 — P2 calibration empirical:** `async µs/CB ≈ 1.6 µs` (3.1× lower than working assumption).  Single-CB recovers ~32% of the 500 µs gap, not ~100%.  Plan revised: P3a per-dispatch cost calibration inserted before P3; P4 added as a second-lever phase whose target is determined by P3a.  Title kept "single-CB" because that's still the first lever; scope is honestly "general decode speed improvements" with the single-CB rewrite as Phase 1 of N.
- **2026-04-26 — P3a calibration empirical:** `µs/dispatch ≈ 0.16 µs` at N≥500 — within ±15% of llama.cpp's ~0.14 µs/dispatch.  Shader-launch path is **not** materially slower.  Budget reconciliation: ~252 µs of the 540 µs gap is encode-time (CB + dispatch); ~288 µs (~53%) is **residual unaccounted** that lives in Rust-side orchestration.  P3b (Rust orchestration sweep) added as parallel phase to P3 single-CB rewrite.  Together P3 + P3b target ≥1.00× exit criteria.
- **2026-04-26 — Literature foundations infused.**  11 arxiv papers fetched + verified pre-training-cutoff, organized into Kernel-level (FlashAttention 1+2), System architecture (vLLM, SGLang, Sarathi-Serve), Decode acceleration (Speculative Decoding, Medusa, EAGLE, Lookahead), and Graph compilation (TVM, Hidet) categories.  Each entry calls out Apple-Silicon applicability honestly.  Apple-Silicon-specific gap acknowledged: no pre-cutoff arxiv work on M-series LLM inference perf that I could verify by direct fetch.  Added §Architectural implications given full-stack ownership — single-CB is the conservative first lever; megakernel + cross-token pipelining open as next-octave levers given we own both layers.
- **2026-04-26 — §"Capturing M5's GPU Neural Accelerators (Metal 4 TensorOps + NAX)" infused.**  Per ADR-016 research dossier graduation: Apple's measured 3.33–4.06× TTFT vs M4 (1.19–1.27× decode bandwidth-bound) lives in per-GPU-core Neural Accelerators via Metal 4 TensorOps, **not** standalone ANE.  mlx-native already partially routes through `mpp::tensor_ops::matmul2d` in four `*_tensor.metal` kernels but is missing: (a) Morton/Z-order dispatch for large prefill GEMM (Tech Talk 111432: ~50 → ~100 % NAX utilization on 4K×4K matmul); (b) NAX-tuned outer tile sizes + macOS 26.2 / arch-gen ≥ 17 runtime gate (mirroring MLX's `is_nax_available()`); (c) `flash_attn_prefill` inner matmul still on M1-era `simdgroup_matrix` / `simdgroup_multiply_accumulate` — biggest unrouted lever (16 % of prefill compute).  Added new sub-phase **P3c — M5 Neural Accelerator prefill kernels** with three actions; P3c is shader-internal only and **fully orthogonal** to P3 (single-CB) and P3b (orchestration sweep) — opens only after P6 bench gate clears.  All API claims verified against Apple developer docs, MLX commits (`54f1cc6` 2025-11-19 NAX support, `b41b349` 2026-03-18 NAX refactor, `0879a6a` 2026-03-11 M5 tuning), and `/opt/mlx-native` source by direct file:line read — no training-data recall used.  Source brief: `/tmp/cfa-adr016/research-metal4-tensorops.md` + memory key `swarm-cfa-adr016/deep-research/metal4-tensorops`.
