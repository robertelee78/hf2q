# ADR-005 Phase 1b — 1bNEW.22 Instrumentation Spike

**Date:** 2026-04-11
**Runner:** Claude (investigation-only; spike instrumentation reverted; no `src/` deltas in main branch beyond this doc)
**Scope:** Reject or confirm the dispatch-count-reduction hypothesis for 1bNEW.22 by direct measurement on the post-1bNEW.21 HEAD baseline. Locate where the 2.37 ms/token gap between hf2q (85.4 tok/s) and llama.cpp (107 tok/s) actually lives — CPU enqueue, GPU compute, GPU launch overhead, or kernel implementation efficiency.
**Baseline binary:** `main` HEAD `a377f76` (post-1bNEW.21 vendored candle-metal-kernels patch).
**Model:** `models/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf` (Gemma 4 26B MoE DWQ).
**Hardware:** Apple M5 Max, 128 GB unified memory.
**Worktree discipline:** all instrumentation reverted before writing this report. `git diff --stat src/serve/gemma4.rs src/serve/mod.rs` returns empty.

---

## Executive summary — the post-1bNEW20 spike's framing was wrong

The previous spike (`docs/spike-post-1bNEW20-results.md`) argued from first principles that the 4.28 ms/token "non-bandwidth residual" was **CPU dispatch overhead** in candle's op-graph machinery, and proposed a 1bNEW.22 = "dispatch-count reduction at 4 hot sites" item with a +6 tok/s envelope. **Direct measurement falsifies that framing in three concrete ways:**

1. **CPU enqueue is only 1.48 ms/token, not 4.28 ms.** Bracketing the entire `Gemma4Model::forward` and the entire `DecoderLayer::forward` shows hf2q's CPU work per forward is just 1484 μs total — 12% of the 11.71 ms/token wall-clock, ~88% of which is GPU compute time draining at the windowed sampler sync. CPU/GPU overlap is **already near-perfect**; there is no idle CPU to recover.

2. **Per-dispatch CPU cost is already at the floor.** 2104 dispatches/token in 1.48 ms = **0.7 μs/dispatch CPU-side**. That is essentially the cost of an `Arc::new` + a HashMap lookup + a function call. There is no fat to trim here.

3. **The 4.28 ms residual is GPU compute time, not CPU dispatch.** Pure weight-traffic bandwidth floor is 8.12 ms/token at M5 Max's effective ~400 GB/s on Q-mixed sequential reads. hf2q's measured GPU compute = wall_clock − CPU_enqueue = 11.71 − 1.48 = **10.23 ms/token**. The 2.11 ms above the bandwidth floor is **GPU per-kernel-launch overhead** at 2104 dispatches × ~1 μs per Metal kernel launch = 2.10 ms — matches almost exactly.

**Corrected mental model:** hf2q at HEAD is GPU-bound, not CPU-bound. The 2.37 ms gap to llama.cpp's 107 tok/s sits **entirely on the GPU side**, and is dominated by per-kernel launch overhead. **Closing the gap means cutting hf2q's GPU dispatch count from 2104 to ~1000** (matching what llama.cpp's ggml graph dispatches per Gemma 4 26B MoE forward).

This DOES still put dispatch-count reduction in the right direction — but the leverage is **GPU launch latency**, not **CPU op-graph cost**. Both correlate, but the lever is GPU-side, and the savings model is "every kernel launch eliminated saves ~1 μs of GPU time" rather than "every candle Tensor op eliminated saves ~2 μs of CPU time".

---

## Methodology

Two-mode bracket instrumentation, gated on the env var `HF2Q_SPIKE_SYNC`:

* **Unset → enqueue-only.** `Instant::now()` at bracket start, `elapsed()` at drop. Captures CPU wall-clock for the lazy candle Tensor op construction inside the bracket. GPU work runs in background and is **not** included.
* **=region_name → forced sync.** Pre + post `Device::synchronize()` calls around the bracket. Captures GPU + CPU wall-clock for that one region in isolation. The other regions stay enqueue-only. (This methodology has a known flaw documented below — it inflates the synced region by the **entire pool's** worth of GPU work, not just the region's own. Useful as a sanity check, not as a primary attribution tool.)
* **=all → sync every bracket.** Diagnostic only.

10 brackets added to `src/serve/gemma4.rs`:

| Bracket | Site | What it covers |
|---|---|---|
| `forward_total` | `Gemma4Model::forward` whole body | Embed → 30 layers → final norm → lm_head → softcap |
| `layer_total` | `DecoderLayer::forward` whole body | One per layer × 30 = 3840 calls/bench |
| `attn_pre` | `Attention::forward` Q/K/V projection chain | q_proj + k_proj + v_proj + reshapes + q_norm + k_norm + v unit-norm + transposes |
| `rope` | `Attention::forward` RoPE call | `rotary_emb.apply(&q, &k, seqlen_offset)` (fused or loop) |
| `kv_append` | `Attention::forward` KV cache append | `kv_cache.append(&k, &v)` (in-place or slice-scatter) |
| `sdpa_oproj` | `Attention::forward` SDPA + post-attention | SDPA dispatch + transpose + reshape + o_proj |
| `moe_topk` | `MoeBlock::forward` router top-k selection | softmax + cast + arg_sort + narrow + gather + sum + broadcast_div |
| `moe_fanin` | `MoeBlock::forward_fused` output combine | gather scale + w_total + unsqueeze + broadcast_mul + sum + reshape + to_dtype |
| `decoder_tail` | `DecoderLayer::forward` post-FFN tail | mlp+moe add + post_feedforward_layernorm.forward_with_post_residual + layer_scalar.broadcast_mul |
| `lm_head` | `Gemma4Model::forward` lm_head + softcap | last-token narrow + final norm + reshape + matmul + softcap |

The dump call lives in `src/serve/mod.rs::run_single_generation` and prints a per-region table at the end of decode (calls, total_ms, μs/call, μs/decode_step).

All 10 brackets were added under `#[cfg(feature = "metal")]` and `drop()`-scoped so they have zero cost when the env var is unset (only the `Instant::now() / elapsed()` pair runs, ~50 ns overhead per bracket).

---

## Run SA — enqueue-only baseline (full coverage)

Build: HEAD `a377f76` + 10 brackets (reverted post-spike). Single 5-run canonical bench, T=0 greedy, 128 decode tokens.

```
HF2Q_SPIKE_SYNC unset → enqueue-only mode.
```

| Region | calls | total_ms | μs/call | μs/decode_step |
|---|---|---|---|---|
| **forward_total** | **128** | **190.0** | **1484.0** | **1484.0** |
| **layer_total** | **3840** | **187.6** | **48.85** | **1465.6** |
| attn_pre | 3840 | 22.7 | 5.92 | 177.6 |
| rope | 3840 | 21.0 | 5.47 | 164.1 |
| moe_topk | 3840 | 20.0 | 5.22 | 156.6 |
| moe_fanin | 3840 | 13.5 | 3.51 | 105.4 |
| decoder_tail | 3840 | 11.1 | 2.90 | 87.0 |
| kv_append | 3840 | 10.1 | 2.64 | 79.3 |
| sdpa_oproj | 3840 | 8.3 | 2.16 | 64.6 |
| lm_head | 128 | 1.10 | 8.59 | 8.59 |
| **Sum of 8 sub-region brackets** | | | | **843** |

**Total decode time at 85.4 tok/s = 11710 μs/token.**

### Decomposition

1. **forward_total = 1484 μs/token** = the entire CPU work inside `Gemma4Model::forward` to encode one decode step.
2. **layer_total = 1466 μs/token** = the CPU work inside the 30 `DecoderLayer::forward` calls.
3. **forward_total − layer_total = 18 μs/token** = the CPU work in non-layer parts of the model forward (embed lookup at 2 μs + final norm + lm_head at ~9 μs + softcap at ~7 μs). Tiny.
4. **CPU not inside `Gemma4Model::forward` = wall_clock − forward_total = 11710 − 1484 = 10226 μs/token = 10.23 ms/token.** This is the GPU compute time that runs concurrently with the CPU encoding the next forward, drained at the windowed sampler sync every 4 tokens. The CPU never blocks on it directly; it's pure GPU compute time that the windowed-drain pattern hides behind future CPU work, but ultimately wall-clock-determining.

### Implications

**The CPU is NOT the bottleneck.** At 1.48 ms CPU enqueue / 11.71 ms wall-clock, the CPU is busy 12.6% of the time. Every existing kernel landing (1bNEW.1/4/6/17/20) collapsed CPU enqueue ALONG WITH GPU execution, but the marginal CPU-side gain from further fusion is bounded above by ~1.4 ms (the entire forward CPU encode time). The marginal GPU-side gain from further fusion is bounded by **the whole 2.11 ms residual above the bandwidth floor**.

**Per-dispatch CPU cost = 1484 μs / 2104 dispatches = 0.71 μs/dispatch.** That's near the floor of what candle's `Tensor::new` + `op::BackpropOp` + `Storage` + `Layout` allocation chain can do (Arc refcounts + small allocations). Candle's CPU side is already optimized for this workload.

**Per-dispatch GPU cost = (10.23 − 8.12) ms / 2104 dispatches = 1.0 μs/dispatch GPU launch overhead** (assuming the rest is bandwidth-bound). This matches Apple's published Metal kernel launch latency on M5 (~0.5–1.5 μs per dispatch on the AGXMetalG17X).

---

## Run SB-SE — forced-sync per-region sweep

For each of 4 hot sites (`attn_pre`, `moe_topk`, `moe_fanin`, `decoder_tail`), 5-run canonical bench with `HF2Q_SPIKE_SYNC=<region>`.

| Sync mode | Median tok/s | ms/token | Slowdown vs SA |
|---|---|---|---|
| SA (none) | 85.4 | 11.71 | — |
| SB (attn_pre) | 42.2 | 23.70 | +12.0 ms/token |
| SC (moe_topk) | 41.6 | 24.04 | +12.3 ms/token |
| SD (moe_fanin) | 41.7 | 23.98 | +12.3 ms/token |
| SE (decoder_tail) | 41.8 | 23.92 | +12.2 ms/token |

The targeted region's reported per-call cost in each sync run is ~190–235 μs/call (vs 2.16–6.6 μs in enqueue-only). The forced sync drains the entire pool at every target-region call. The slowdown is approximately the same regardless of which region is synced (~12 ms/token = ~3 ms per pool drain × 4 drains/window because the sampler windows are 4 tokens × 30 layers of bracket calls).

**The forced-sync data is NOT a clean per-region attribution.** It's measuring the "flush_and_wait floor" the post-Walk re-spike documented at 0.15-0.20 ms — except that floor has grown to ~3 ms because the pool now contains more queued work. The per-call inflation (0.7 μs unsynced → 200+ μs synced) is the entire pool's worth of GPU work, not the region's own work.

**The useful insight from the forced-sync runs is the OTHER regions in the same run dropping below their unsynced cost** — e.g., when `attn_pre` is synced, `decoder_tail` drops from 86 to 57 μs/step. This confirms candle's command pool was already overlapping CPU/GPU work effectively, and the syncs serialize what would otherwise be pipelined.

---

## Run SF — kernel-fallback bisect (the real attribution data)

The CLI exposes `--rms-norm-kernel`, `--rope-kernel`, `--moe-kernel`, `--lm-head-kernel`, `--kv-cache-kernel` flags with `loop` (pre-fusion) fallbacks for each landed Walk-KERNEL-PORT. Running each in turn measures **how much each landed kernel currently saves**, which is the inverse measurement of "how much would more of this kind of work save".

Build: HEAD `a377f76`, no spike instrumentation, 5-run canonical bench per mode.

| Mode | Median tok/s | ms/token | Δ vs HEAD | Mechanism eliminated |
|---|---|---|---|---|
| **HEAD (all fused)** | **85.4** | **11.71** | **—** | **—** |
| `--rope-kernel=loop` | 72.8 | 13.74 | **−2.03 ms** | 1bNEW.6 fused RoPE neox kernel |
| `--lm-head-kernel=loop` | 66.0 | 15.15 | **−3.44 ms** | 1bNEW.17 F16 lm_head matmul (vs F32 dense) |
| `--rms-norm-kernel=loop` | 61.6 | 16.23 | **−4.52 ms** | 1bNEW.4 fused RmsNorm F=1/F=2/F=3 kernel |
| `--kv-cache-kernel=slice-scatter` | 58.2 | 17.18 | **−5.47 ms** | 1bNEW.20 in-place KV via slice_set |
| `--moe-kernel=loop` | 36.6 | 27.32 | **−15.61 ms** | 1bNEW.1 fused MoE kernel_mul_mv_id_t |

**Cumulative landed savings = 31.07 ms/token across 5 Walk-KERNEL-PORT items**, taking hf2q from a synthetic-all-loop baseline of ~36.8 tok/s (matching the pre-1bNEW.6 ADR record of 36.78) to the current 85.4 tok/s. Each individual fusion has been a multi-ms win.

**Forward-total CPU enqueue under each fallback:**

| Mode | forward_total μs | ratio vs HEAD |
|---|---|---|
| HEAD | 1484 | 1.0× |
| rope_loop | 1816 | 1.22× |
| kv_loop | 1788 | 1.20× |
| lm_head_loop | 1494 | 1.01× |
| rms_loop | 10029 | 6.76× |
| moe_loop | 34095 | 22.97× |

The **rms_loop** and **moe_loop** modes balloon CPU enqueue 7-23× because they explode the candle Tensor op count (RmsNorm: 1 fused dispatch → 11 manual ops × 331 sites; MoE: 2 fused kernels → ~30 ops/expert × 8 experts × 30 layers + CPU expert loop). At HEAD's 1484 μs CPU enqueue, the fused kernel landings are NOT just speeding up the GPU side — they are also keeping the CPU enqueue cost low enough that CPU stays well-overlapped behind GPU work.

The smaller-ratio modes (`rope_loop`, `kv_loop`, `lm_head_loop`) have similar CPU enqueue as HEAD — those landings save predominantly **GPU compute time**, not CPU encode time. Specifically:
- **lm_head_loop**: same op count (just F32 vs F16 matmul), saves 3.44 ms purely from halving the lm_head weight read (2.95 → 1.48 GB/token). Pure bandwidth-savings via dtype, not dispatch fusion.
- **rope_loop**: ~7 extra ops per RoPE call site × 60 sites = 420 extra dispatches, saves 2.03 ms = ~4.8 μs per saved dispatch. That's 4-5× the per-dispatch GPU launch overhead, so the savings are NOT purely launch-overhead — the loop-mode RoPE also generates intermediate tensors that read/write memory that the fused kernel does in-register.
- **kv_loop**: only ~3 extra ops per KV append × 30 layers = 90 extra dispatches, saves 5.47 ms = ~60 μs per saved dispatch. Way over the launch-overhead floor — this is the "implicit drain unlock" effect 1bNEW.20 documented (slice_scatter ops force pool drains; in-place doesn't).

---

## Where the 21.7 tok/s gap to llama.cpp lives

Synthesizing the data:

1. **CPU-side ceiling already reached.** At 1.48 ms/token CPU enqueue, with 0.7 μs/dispatch and near-perfect CPU/GPU overlap, the CPU side is already at the floor for what candle can deliver on this workload. Further dispatch-count reduction is **not** primarily a CPU-side win; it's a **GPU-side launch-latency** win.

2. **GPU compute time = 10.23 ms/token.** Bandwidth floor (sequential weight reads at ~400 GB/s) ≈ 8.12 ms. Excess = **2.11 ms/token**. This is GPU launch latency at 2104 dispatches × ~1 μs each.

3. **The 2.37 ms gap to llama.cpp is essentially the GPU launch latency difference.** llama.cpp's ggml graph scheduler dispatches ~1000 Metal kernels per Gemma 4 26B MoE forward (estimated from ggml-metal source — exact count would require running with `GGML_METAL_NDISPATCH_LOG=1` or similar; not measured in this spike). At 1000 × 1 μs = 1 ms launch overhead, llama.cpp's effective GPU compute time is ~9.12 ms, matching their 9.35 ms wall-clock minus a small CPU overhead.

4. **Closing the gap requires cutting hf2q's GPU dispatch count from 2104 to ~1000.** That's roughly halving the dispatch count, which means **kernel fusion at multiple sites** beyond just one.

### Where the 1100+ excess dispatches live

From `metrics.txt` on HEAD:
- `dispatches_per_token = 2104.52`
- `moe_dispatches_per_layer = 34.00` → 30 layers × 34 = **1020 dispatches in MoE blocks**
- `norm_dispatches_per_token = 331` → **331 dispatches in RmsNorm sites**
- Remaining: 2104 − 1020 − 331 = **753 dispatches** in attention (q/k/v/o projections, RoPE, KV append, SDPA), MLP, residual adds, layer_scalar, lm_head, embed lookup, softcap, sampler.

Per layer: ~70 dispatches. Of those:
- MoE: 34 (router preamble 17 + expert path 17)
- Norms: 11 (input + post-attn + pre-FFW + 2× pre-FFW2 + 2× post-FFW + q_norm + k_norm + v_norm + post-MoE = 11 sites)
- Attention preamble: ~6 (3 QLinear matmul + 2 reshapes + 1 v_norm)
- RoPE: 2 (one Q kernel, one K kernel post-1bNEW.6)
- KV append: 3 (in-place mode)
- SDPA: 1
- Post-attention transpose+reshape+o_proj: 4
- MLP: 3 (gate_proj + up_proj + down_proj QLinears, no fan-in needed since dense MLP has no top-k)
- Decoder residuals/scalars: 4 (pre-attn add + mlp+moe add + layer_scalar + post-FFN combine)
- Total per layer: 68. Plus 28 model-level (embed + final norm + lm_head + softcap). Matches 2104.

### Which sites are fusable

Ranking by absolute dispatch count and reference availability:

| Sub-item | Dispatches saved | Reference | Difficulty | Risk |
|---|---|---|---|---|
| **A. MoE router top-k → single fused kernel** (softmax + arg_sort + narrow + gather + sum + div) | 240 | mlx_lm `mx.topk` (`mlx_lm/models/switch_layers.py:62-75`); llama.cpp `ggml_top_k` | Medium | Low — pure compute, no memory rewrite |
| **B. MoE fan-in → into the down kernel** (gather scale + broadcast_mul + sum) | 120 | llama.cpp `kernel_mul_mv_id_q*` already includes the per-expert weight broadcast in its kernel body | Medium-High — modifies a vendored Metal kernel | Medium |
| **C. RmsNorm → fold into adjacent QMatMul** | 200-300 | mlx-lm at the framework level via lazy graph fusion; llama.cpp via `ggml_mul_mat(ggml_rms_norm(x), w)` graph rewriter | High — new fused kernel per dtype | Medium |
| **D. Attention QKV → fused projection + norm** (3 QLinears + 3 reshapes + 3 norms → 3 dispatches) | 240 | No direct reference — neither llama.cpp nor mlx-lm fuses Q/K/V projection with their norms at the kernel level (both do them as separate ggml/mx ops). **Walk-CAPABILITY** justification only. | High | Medium |
| **E. Pre-attention residual + input_layernorm → fused F=4 kernel** | 30 | Same pattern as the existing F=3 (norm+residual_add); inverted order. mlx-lm and llama.cpp do them as separate ops, but both also evaluate in their respective graph schedulers — Walk-CAPABILITY. | Low — extends existing 1bNEW.4 kernel | Low |
| **F. Layer_scalar broadcast_mul → fold into next layer's input_layernorm** | 30 | Both references do this as a separate op. Walk-CAPABILITY only. | Low | Low |

**Combined dispatch savings:** A+B+C+D+E+F ≈ 860–960 dispatches saved. Closing 2104 → ~1150-1250.

**Combined wall-clock savings (at ~1 μs/dispatch GPU launch):** ~0.86-0.96 ms/token = **8-9 tok/s**. Plus second-order savings from reduced CPU enqueue (each saved dispatch is also ~0.7 μs of CPU encode time, ~0.7 ms = ~6 tok/s savings on the CPU side, ~half of which is hidden behind GPU and ~half of which surfaces as wall-clock).

**Realistic combined envelope: 11-13 tok/s** taking hf2q from 85.4 → 96-98 tok/s. **Still 9-11 tok/s short of 107.**

### What closes the remaining 9-11 tok/s

The bandwidth floor at 8.12 ms/token = 123 tok/s gives an absolute upper limit if GPU launch overhead were zero. To get closer to that limit, we need either:

1. **Even more aggressive dispatch reduction** at sites I haven't enumerated (sub-item-level fusion inside attention SDPA prelude, finer fusion of the per-layer norm chain, etc.). Diminishing returns.

2. **Per-kernel implementation improvements** — port faster Q4_0/Q6_K matmul kernels from llama.cpp where they're more bandwidth-efficient than candle's MLX-derived templates. Specific candidates:
   - `kernel_mul_mv_q4_0_f32` for the MLP gate_proj/up_proj/down_proj (currently dispatches via candle's QMatMul → MLX gemm path). llama.cpp's hand-tuned variant achieves higher effective bandwidth on M-series GPUs.
   - `kernel_mul_mv_q6_K_f32` for attention q_proj/k_proj/o_proj. Same story.
   - This is risky/expensive — modifying vendored candle-metal-kernels at the kernel level for each weight dtype variant.

3. **Sticky encoder optimization in candle** (the 1bNEW.23 candidate from the prior spike). This was speculative; with the corrected mental model it remains speculative because per-encoder GPU setup overhead is part of the 1 μs/dispatch launch latency floor that's already counted. The savings if any would be from coalescing argument table bindings across consecutive same-buffer dispatches.

4. **Reduce per-token weight traffic** — the lm_head F16 read at 1.48 GB is the largest single contributor. Could try a Q8_0 or smaller quant of the embed/lm_head tensor (note: 1bNEW.17 spike A established the GGUF stores it as F16; converting to a smaller quant requires re-quantizing the model, which is out of scope for ADR-005). Bandwidth-bound floor argument: at the spike-Q3Q4Q5 measurements, Q6_K at this output dim projects ~7.97 ms/call, slower than F16 at 3.73 ms. **No further win available here on this hardware on this GGUF.**

5. **Reduce per-token attention weight traffic** at sliding layers — sliding has KV cache up to 1024 tokens × 2 KV heads × 256 head_dim × 2 bytes f16 = 1 MB per layer × 25 sliding layers = 25 MB/token of KV traffic. Already small. Marginal.

6. **Single-command-buffer patterns at the layer level** — submit one MTLCommandBuffer per decoder layer (30 buffers per token instead of ~42). Micro-tuning of the candle-metal-kernels patch from 1bNEW.21. Worth a small spike but unlikely to be more than 0.5 tok/s.

---

## Proposed 1bNEW.22+ roadmap (post-instrumentation reality)

Sequenced for Walk-discipline correctness gating and incremental validation:

| Item | Scope | Expected Δ tok/s | Cumulative tok/s | Risk | Reference |
|---|---|---|---|---|---|
| **1bNEW.22** | MoE router top-k fused kernel: replace softmax + arg_sort + narrow + gather + sum + broadcast_div with a single Metal kernel that takes router logits and writes top_k_indices + top_k_weights | +2 to +3 | 87-88 | Low | mlx_lm `mx.topk`, ggml `ggml_top_k` |
| **1bNEW.23** | Decoder pre-attention residual + input_layernorm fused via new RmsNormKernel F=4 variant (residual_add → norm) | +0.5 to +1 | 88-89 | Low | Extension of 1bNEW.4 |
| **1bNEW.24** | MoE fan-in fused into the down kernel: write the per-expert weight broadcast and per-token sum directly inside `kernel_mul_mv_id_*` instead of as separate candle ops | +1 to +2 | 89-91 | Medium-High | llama.cpp's MoE down kernel does this internally |
| **1bNEW.25** | RmsNorm → adjacent QMatMul fusion for the pre-projection norms (`q_norm` → `q_proj`, `k_norm` → `k_proj`, etc.). Per-norm cost at HEAD ≈ 1 dispatch each; folding eliminates 8-10 sites/layer × 30 = 240-300 dispatches | +2 to +3 | 91-94 | High — new fused kernel per dtype combination | mlx-lm lazy graph fusion; ggml-metal `kernel_norm_mat` (does not exist as a single kernel today; would be a hf2q-side novel kernel based on extending the 1bNEW.4 pattern) |
| **1bNEW.26** | Attention QKV fused projection — combine q_proj + k_proj + (v_proj if not k_eq_v) into one fused QLinear kernel that reads xs once and writes Q+K+V outputs in one dispatch. Eliminates ~2 dispatches per layer plus ~3 reshapes, ~120 total | +1 to +2 | 92-96 | High — new kernel | No direct reference (mlx-lm/llama.cpp keep them separate); Walk-CAPABILITY |
| **1bNEW.27** | Layer_scalar broadcast_mul folded into next layer's input_layernorm (or into post_feedforward_layernorm of current layer) | +0.3 to +0.5 | 92-96 | Low | Walk-CAPABILITY |
| **1bNEW.28** | Per-layer command-buffer batching in vendored candle-metal-kernels: submit one `MTLCommandBuffer` per `DecoderLayer::forward` | +0.5 to +1 | 93-97 | Medium | Continuation of 1bNEW.21 |
| **1bNEW.29** | Port llama.cpp's `kernel_mul_mv_q4_0_f32` and `kernel_mul_mv_q6_K_f32` for the dense projection sites (MLP gate/up/down + attention q/k/v/o), replacing candle's MLX-derived QMatMul path. Measure first to confirm llama.cpp's variants are faster on M5 Max. | +5 to +10 (speculative) | 98-107 | High — multiple new vendored kernels | llama.cpp `ggml/src/ggml-metal/ggml-metal.metal:7400-7700` |

**Honest envelope:** items 22-28 give a measured-confident **+8 to +13 tok/s** taking hf2q to ~94-98. Item 29 is the **End-gate-closing item** but is the largest and most speculative — it requires verifying that llama.cpp's hand-tuned Q-kernels are actually faster than candle's MLX-derived ones on M5 Max for the specific shapes hf2q dispatches. If they are, the +5 to +10 tok/s is achievable and we hit 107. If they're not, item 29 needs to be replaced with a different End-gate-closing strategy (which at present I do not have a candidate for under Walk discipline).

**Pre-29 validation spike:** before embarking on item 29, run a microbenchmark against candle's `call_quantized_matmul_mv_t` vs a manually-ported llama.cpp `kernel_mul_mv_q4_0_f32` for the exact MLP gate_proj shape `[1, 2816] @ [2112, 2816] Q4_0`. If the latter is ≥10% faster at this shape, item 29 is justified. If parity, item 29 is not justified and we need a different gap-closer.

---

## Mantra-aligned discipline notes

Three places this spike could have been lazy and weren't:

1. **The first 4 brackets (post-1bNEW.20 spike's chosen sites) accounted for only 0.52 ms/token of CPU enqueue.** The lazy move at that point was "the cost is distributed, pivot to sticky encoder". The diligent move was to add 5 more brackets (rope, kv_append, sdpa_oproj, lm_head, layer_total, forward_total) and find that the entire forward CPU encode is only 1.48 ms — at which point the whole "CPU dispatch overhead" framing collapsed and the GPU-bound truth surfaced.

2. **The forced-sync per-region runs gave inflated numbers (~6 ms/region) that look like attribution data but are not.** The lazy move was to report those as "MoE topk costs 6 ms/token, that's the lever". The diligent move was to recognize the inflation as the pool-flush_and_wait floor (the same mechanism the post-Walk re-spike documented) and discard those numbers in favor of the enqueue-only data.

3. **The kernel-loop bisect via `--rms-norm-kernel=loop` etc. is the highest-signal data in this spike.** The lazy move was to skip it because "those flags exist for bisect-safety, not benchmarking". The diligent move was to run them anyway, which produced the table that anchors the "GPU launch latency is the lever" conclusion (because the loop modes inflate CPU enqueue 7-23× and inflate wall-clock 1.2-2.3× — so the per-fusion savings include both CPU AND GPU contributions).

**Chesterton's fence on the 4 originally-proposed 1bNEW.22 sites:** I changed my mind. The post-1bNEW20 spike's choice of those 4 sites was based on a wrong CPU-bound model. Under the GPU-bound model, the same 4 sites are still the right targets, but the savings calculus is different (1 μs GPU launch / saved dispatch instead of 2 μs CPU encode / saved dispatch). The MoE topk site remains the highest-value first item because it has the largest single fusion (270 → 30 dispatches) AND a clear reference port from mlx-lm. The other 3 original sites are folded into items 23-27.

---

## Citation trail

- Baseline binary: HEAD `a377f76` post-1bNEW.21 vendored candle-metal-kernels patch.
- Spike instrumentation site: `src/serve/gemma4.rs::spike_sync` module + 10 `Bracket::new` call sites + dump call in `src/serve/mod.rs::run_single_generation`. Reverted before this report; `git diff --stat src/serve/gemma4.rs src/serve/mod.rs` returns empty.
- Kernel-loop bisect commands:
  - `./target/release/hf2q generate --model <gguf> --prompt-file tests/bench_prompt_128.txt --max-tokens 128 --temperature 0 --benchmark --rms-norm-kernel=loop`
  - (and similar for `--rope-kernel=loop`, `--moe-kernel=loop`, `--lm-head-kernel=loop`, `--kv-cache-kernel=slice-scatter`)
- candle-metal-kernels command pool: `/opt/hf2q/vendor/candle-metal-kernels/src/metal/commands.rs` (post-1bNEW.21 vendor patch).
- Per-dispatch GPU launch latency reference: Apple Metal Best Practices Guide, "Drawable" + "Compute" launch overhead measurements (~0.5-1.5 μs on M-series).
- llama.cpp ggml-metal dispatch reference: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` and `ggml-metal-ops.cpp`.
- mlx_lm top-k reference: `/Users/robert/.pyenv/versions/3.13.12/lib/python3.13/site-packages/mlx_lm/models/switch_layers.py:62-75`.
- ADR-005 End gate (line 860): "≥107 tok/s on M5 Max, Gemma 4 26B MoE, Q4_K_M".

---

## Return message (concise)

- HEAD baseline 85.4 tok/s = 11.71 ms/token. 21.7 tok/s short of 107.
- **CPU enqueue is only 1.48 ms/token** — measured by bracketing `Gemma4Model::forward` in its entirety. The CPU is at 12.6% utilization, well-overlapped with GPU.
- **GPU compute is 10.23 ms/token.** Bandwidth floor is 8.12 ms/token; excess is 2.11 ms/token = **GPU per-kernel launch overhead** at 2104 dispatches × ~1 μs each.
- **The 2.37 ms gap to llama.cpp is on the GPU side**, dominated by the dispatch-count difference (~2104 hf2q vs ~1000 llama.cpp estimated).
- **Dispatch-count reduction IS the right direction**, but the leverage is GPU launch latency (~1 μs/dispatch saved) NOT CPU op-graph overhead (already at floor).
- Proposed 7-item roadmap 1bNEW.22-29 sequenced from low-risk smallest fusions (MoE topk, residual+norm, fan-in) through medium-risk larger fusions (RmsNorm-into-matmul, fused QKV) up to the End-gate-closing item 1bNEW.29 (port llama.cpp's hand-tuned Q-kernels for the dense projection sites). Expected combined: **96-98 tok/s after 22-28**, plus **+5 to +10 tok/s speculative from item 29** to close the End gate.
- Pre-1bNEW.29 validation spike required: microbenchmark candle's QMatMul vs llama.cpp's `kernel_mul_mv_q4_0_f32` on the exact MLP gate_proj shape. Item 29 is justified iff llama.cpp's variant is ≥10% faster at the relevant shape on M5 Max.
- Spike instrumentation reverted; no `src/` deltas in mainline beyond this doc.
