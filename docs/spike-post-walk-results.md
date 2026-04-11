# ADR-005 Phase 1b — Post-Walk Re-Spike Report

**Date:** 2026-04-10
**Runner:** Claude (spike-only; no `main` commits)
**Scope:** Post-Walk gap decomposition, residual Walk-item analysis, Walk-vs-Run verdict
**Baseline binary:** `main` HEAD `4e7fe31` (1bNEW.6 Phase C DONE; 1bNEW.1/3/4/6 all landed, 1bNEW.10/12 landed, pre-flight 1bNEW.0/0a/0b/0c landed)
**Model:** `models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf` (Gemma 4 26B MoE DWQ, mixed Q4_0 / Q6_K / Q8_0)
**Config (from `src/serve/config.rs:74-99`):** vocab=262144, hidden=2816, intermediate=2112, moe_intermediate=704, num_hidden_layers=30, num_attention_heads=16, num_kv_heads=8 (sliding) / 2 (global), head_dim=256 / 512, num_experts=128, top_k=8, tie_word_embeddings=**true**, layer_types=sliding-heavy (every 6th is full-attention → 5 global / 25 sliding).
**Hardware:** Apple M5 Max, 128 GB unified memory
**Decode baseline at start of spike:** 48.82 tok/s median, p95 48.91 (5-run canonical harness on clean main binary). Matches ADR line ~713 recorded 48.71 median within noise.
**Worktree discipline:** all instrumentation reverted; `git status` shows clean `src/` tree (evidence in the return message).

---

## Executive Summary

The four Walk kernel ports (1bNEW.1 MoE, 1bNEW.4 RmsNorm, 1bNEW.6 RoPE, plus 1bNEW.3 sampler windowing and 1bNEW.10/12 BF16 prefill / warmup) lifted hf2q from 23.76 tok/s to **48.82 tok/s** (+105%). The remaining gap to the 107 tok/s End gate is **11.15 ms/token**. Measurements below pin that gap to **one dominant, reference-citable Walk item** (quantized lm_head, ~5.3 ms/token) plus **a residual 4-6 ms/token of structural candle-side latency** that does not map cleanly to a Walk-faithful port. Walk can plausibly reach **~68-75 tok/s** on its own; the remaining **32-39 tok/s** require either a candle-upstream per-buffer-wait patch (Run) or a structural refactor of the decode loop that is not a line-for-line port of anything in llama.cpp / mlx-lm.

**Headline finding #1: lm_head is dense F32.** The tied-embedding `token_embd.weight` is dequantized to F32 at load time (`src/serve/gguf_loader.rs:50-57`, called via `src/serve/gemma4.rs:1636`), producing a `[262144, 2816]` F32 weight tensor that is **2.95 GB** in VRAM. Every decode token runs a dense F32 matmul `[1, 2816] @ [2816, 262144]` against this tensor, which reads the full 2.95 GB through memory — **7.14 ms/token measured, 7.38 ms/token predicted from M5 Max bandwidth**. llama.cpp keeps the same tensor quantized (Q6_K at 0.8125 bytes/weight = 600 MB) and reads 5.0× less memory through a quantized matmul kernel per token. This accounts for **~5.6 ms of the 11.15 ms gap** and has a direct `file:line` port target in llama.cpp: use `QMatMul` at the lm_head site instead of dense `.matmul(.t())`.

**Headline finding #2: The non-lm_head critical path is ~13.3 ms/token, attributed approximately** — 4.2 ms to MoE QMatMul memory-bandwidth floor, 2.5-3 ms to attention/MLP QMatMul, 1-2 ms to remaining kernel launch overhead and CPU/GPU pipelining inefficiency, 0.3-0.5 ms to norms and RoPE. This pool contains **no single large item** with a Walk-faithful port remaining. Removing lm_head alone would put hf2q at approximately **14.4 ms/token = 69 tok/s**. The next 6-7 ms to hit 107 tok/s are Run territory — they require changes not present in either reference: either a **per-buffer wait semantic in candle's command pool** (re-enabling the 1bNEW.3 projected win) or a **decode-loop fusion that keeps the GPU continuously fed** (structural, not a kernel port).

**Headline finding #3: The "pipeline is already nearly saturated" hypothesis holds.** Enqueue-only CPU time per forward = 8.33 ms; wall-clock per forward = 20.48 ms; deferred GPU work visible at the sampler drain = 12.34 ms/forward. **CPU enqueue and GPU compute overlap almost perfectly** (8.33 + 12.34 = 20.67, within 0.2 ms of measured 20.48). There is essentially no slack in the pipeline; each Walk item from here must reduce either CPU enqueue or GPU compute directly, not overlap.

**Verdict on the End gate:** **INCONCLUSIVE leaning MEASURED_UNREACHABLE under strict Walk.** A quantized lm_head Walk item is reachable with a live citation and closes roughly half the gap. The remaining half has no clean Walk port — the best candidates are Run items. Under the Anti-Goal #12 "no escape-hatch at 75-85" constraint, Phase 1b Walk is unlikely to close out at 107 without opening Run.

---

## Part 1: Post-Walk Gap Decomposition

### Methodology

1. **Revisit the post-Walk baseline.** Ran the canonical benchmark harness on the `main` HEAD (`4e7fe31`) clean binary. Got **48.82 tok/s median, p95 48.91, variance 0.1**. Matches ADR line ~713 recorded 48.71 tok/s within run-to-run noise.
2. **Instrument every candidate region** with nanosecond `Instant` timers, gated on an `HF2Q_SPIKE_SYNC=<region>` env var that either (a) records enqueue-only CPU wall-clock when unset, or (b) brackets the region with `Device::synchronize()` pre+post when set to that region's name. This follows the Q3 spike methodology exactly (`docs/spike-Q3Q4Q5-results.md:173-193`) and uses the same counter-struct plumbing (`DispatchCounters` already has the Arc threading in place from 1bNEW.0).
3. **Run seven measurement passes** on the canonical harness (187-token prompt, 128 greedy decode tokens, 5 runs each). Each pass reports: median tok/s, spike_forward_ms_per_token, and the per-region ms/token accumulator. Every run's 5-run variance was ≤ 0.1 tok/s.
4. **Revert every source edit** before writing the report. `git status` shows clean `src/` tree at the end.

All measurements on the same GGUF on Apple M5 Max, no `hf2q` or `cargo test` processes running in parallel.

### Raw measurements (7 spike runs)

| Run | HF2Q_SPIKE_SYNC | Median tok/s | spike_forward_ms/tok | Interpretation |
|---|---|---|---|---|
| SA | (unset — enqueue-only) | **48.82** | 8.33 | CPU enqueue wall-clock per region; GPU runs in background via command pool |
| SB | `forward` | 47.92 | **20.82** | Pre+post forward sync — forces candle to drain each forward. Measured wall-clock matches `1000 / 48.82 = 20.48 ms` within 0.34 ms overhead. Reference measurement. |
| SC | `moe` | 30.52 | 31.26 | Sync at each MoE fused block — captures 9.17 ms/token in the MoE sync window (sum of 0.306 ms × 30 layers) |
| SD | `rms` | 9.24 | 108.15 | Sync at each of 331 RmsNorm dispatches/token — reveals a **0.166 ms forced-sync floor** per sync point (331 × 0.166 = 54.8 ms forced overhead) |
| SE | `rope` | 30.86 | 30.72 | Sync at each of 60 RoPE dispatches/token — forced-sync floor 0.096 ms × 60 = 5.75 ms. Ignores kernel compute. |
| SF | `lm_head` | 47.69 | 21.06 | **Sync around the single `[1,2816] @ [2816,262144]` F32 dense matmul per forward** — captures **7.14 ms/token at this call site alone**. Only 2.3% tok/s drop. |
| SG | `sdpa` | 31.01 | 30.47 | Sync at each of 30 decode SDPA dispatches — forced-sync floor 0.187 ms × 30 = 5.6 ms. |
| SH | `qlinear` | 11.31 | 87.23 | Sync at every QLinear call (q/k/v/o attention + MLP gate/up/down + MoE router, ~240 calls/token) — 0.175 ms × 240 = 42 ms forced overhead |
| SI | `all` | 6.10 | 164.91 | Every instrumented region synced — sum of per-region forced-sync costs |

**The per-region sync runs (SC, SD, SE, SG, SH) are dominated by a 0.15-0.20 ms `flush_and_wait` floor per sync point.** This matches the Q3 finding: forced syncs cost a fixed pool-drain amount regardless of what's in the pool. They do NOT measure the per-kernel GPU compute time. For compute-time estimation I fall back to the Q5-style methodology: use the enqueue-only delta (SA) and known per-kernel bandwidth-bound floors from weight sizes.

The exception — Run SF (lm_head) — is the one per-forward call that DOES capture a real GPU cost, because there's only one of them per forward, and the forced sync drains the whole GPU pipeline at that one point. It reports **7.14 ms/token** which is tightly consistent with the **memory-bandwidth-bound floor** of reading a 2.95 GB F32 weight tensor at ~400 GB/s (M5 Max effective bandwidth): 2.95 / 400 = 7.38 ms. Two independent derivations converge within 3%.

### The decode-time breakdown

Total decode wall-clock at 48.82 tok/s = **20.48 ms/token**. Target 107 tok/s = **9.35 ms/token**. Gap = **11.13 ms/token**.

| # | Component | Cost (ms/tok) | Source / citation | Methodology | Share of 11.13 ms gap |
|---|---|---|---|---|---|
| 1 | **lm_head dense F32 matmul** `[1,2816]@[2816,262144]` | **7.14 measured / 7.38 predicted** | `src/serve/gemma4.rs:1879` (`normed_2d.matmul(&self.lm_head_weight.t()?)`); weight dequantized at `src/serve/gguf_loader.rs:50-57` called via `src/serve/gemma4.rs:1636` (`get_tensor(..., MODEL_DTYPE=F32)`) and aliased as `lm_head_weight` at `gemma4.rs:1836` | Forced sync Run SF captures 7.14 ms at this exact call site. Bandwidth floor: `262144 × 2816 × 4 bytes = 2.95 GB` / `400 GB/s` = 7.38 ms. | **~64%** |
| 2 | **MoE QMatMul compute** (fused `kernel_mul_mv_id_q6_K_f32` / `kernel_mul_mv_id_q8_0_f32` / `kernel_mul_mv_id_q4_0_f32` — 2 per layer × 30 = 60 kernel calls/tok) | **4.0-4.3** (estimated lower bound from memory traffic) | `src/serve/gemma4.rs:1246-1268` (fused gate_up dispatch) and `1335-1356` (fused down dispatch). Fused kernel at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:7589` (template) and 7624-7642 (instantiations). | Bandwidth floor: 8 experts × (2×704×2816 gate_up + 704×2816 down) × 0.5 bytes/weight × 30 layers ≈ **1.57 GB/token** / 400 GB/s = **3.92 ms**. Enqueue-only CPU cost from Run SA: 0.95 ms/tok (binding + encoder setup). Remaining ~3 ms is GPU memory-bound compute. The 9.17 ms reported by Run SC is inflated by the forced-sync drain absorbing the preceding layer's tail. | ~36% |
| 3 | **MLP QMatMul compute** (3 QMatMuls per layer × 30 = 90 calls/tok: gate_proj + up_proj + down_proj, all dense `[2816]↔[2112]`) | **1.3-1.5** (estimated from memory traffic) | `src/serve/gemma4.rs:946-956` (Mlp::forward chain), `gemma4.rs:680-696` (QLinear::forward — the F32-in / F32-out QMatMul path). Weight shapes: gate `[2112×2816] Q4_0`, up `[2112×2816] Q4_0`, down `[2816×2112] Q4_0`. | Bandwidth: 3 × 2112 × 2816 × 0.5 bytes × 30 layers = **0.54 GB/token** at ~400 GB/s → **1.35 ms**. | ~12% |
| 4 | **Attention QMatMul compute** (Q/K/V/O projections × 30 layers; k_eq_v saves the V matmul → 3 calls/layer × 30 = 90 calls/tok) | **1.0-1.5** (estimated from memory traffic) | `src/serve/gemma4.rs:664-672` (q_proj, k_proj, optional v_proj), `:882` (o_proj). Shapes: sliding `[2816]→[4096]` or `[2048]` (Q, K); global `[2816]→[8192]` or `[1024]` (Q, K). All Q4_0 in DWQ. | Rough bandwidth: ~0.35 GB/token across the 90 calls = **0.88 ms**. Plus the KV-cache slice_scatter writes: 6 ops × 30 layers that each write `[1, kv_heads, 1, head_dim]` — dominated by slice_scatter internal copies, adds ~0.3 ms. | ~10-15% |
| 5 | **Fused RmsNorm compute** (331 fused dispatches/tok × very small kernel) | **0.5-0.8** (estimated) | `src/serve/rms_norm_kernel.rs:398-606` (rms_norm_fused); host port at `src/serve/gemma4.rs:212-249` (RmsNorm::forward) and `:900-932` (rms_norm_unit). | Enqueue-only CPU from Run SA: 1.47 ms/tok. Each fused kernel reads `[1, hidden_or_head_dim] F32` + weight = ~24 KB per call × 331 = 7.9 MB/token. **Bandwidth is negligible (<0.02 ms).** The 0.5-0.8 ms is dominated by PSO binding and command encoder setup inside the kernel — which IS CPU overhead, overlapped with GPU. At 331 dispatches × 0.002 ms/binding = ~0.66 ms. | ~5% |
| 6 | **Fused RoPE compute** (60 fused dispatches/tok: Q and K per layer × 30) | **0.2-0.4** (estimated) | `src/serve/rope_kernel.rs` (fused kernel); `src/serve/gemma4.rs:429-471` (RotaryEmbedding::apply fused branch). | Enqueue-only CPU from Run SA: 0.40 ms/tok. Kernel reads ~16 KB × 60 = 1 MB/token. Bandwidth negligible. Dominated by dispatch encode overhead. | ~3% |
| 7 | **Decode SDPA (30 vector-path dispatches/tok)** | **0.2-0.4** (estimated) | `src/serve/gemma4.rs:754-767` (q_len==1 branch calling `candle_nn::ops::sdpa`). Kernel at `candle-metal-kernels/src/kernels/sdpa.rs` vector path. | Enqueue-only CPU from Run SA: 0.045 ms/tok (very cheap). KV cache reads are dominated by the per-layer kv_cache size — at decode step 128 the cache is ~187+128 = 315 tokens × 2 × 8 kv_heads × 256 head_dim (sliding) or 2×2×512 (global) × 4 bytes = ~5 MB for sliding layers, 2.5 MB for global = ~128 MB/token total SDPA traffic across 30 layers. At 400 GB/s → 0.32 ms. | ~3% |
| 8 | **Sampler drain wall-clock** (32 drains across 127 forwards; 4 forwards per drain; includes GPU tail drain) | **0.2-0.3** (ACTUAL drain cost, not deferred GPU work) | `src/serve/mod.rs:488-490` (`Tensor::cat(...).flatten_all()?.to_vec1::<u32>()?`). Implemented via 1bNEW.3 windowed async drain. | Enqueue-only Run SA reports 12.34 ms/token from the drain, BUT under forward-sync (SB) the drain drops to 0.07 ms — meaning **12.27 of those "drain" ms are actually GPU work from the preceding 4 forwards that happens to be waited-for AT the drain site**. The drain's own wall-clock is just the cat + u32 vec transfer. | <3% |
| 9 | **CPU overhead: candle op graph, Arc refcounts, KV cache append, reshape/cast/narrow** | **4.2-4.3 ms/tok** (derived by subtraction) | Distributed across the forward path; biggest contributors: KV cache append (6 ops × 30 layers × small tensor allocations), dtype casts, the 8 `to_dtype/contiguous` ops in the prefill BF16 SDPA path (decode path is simpler). | Enqueue total from Run SA = 8.33 ms spike_forward. Per-region enqueue sum from Run SA = 4.04 ms (moe + rms + rope + sdpa + qlinear + lm_head). The residual 8.33 - 4.04 = **4.29 ms** is candle-graph-setup / view-rewrite / KV-cache / tensor-alloc CPU work not captured by any per-region timer. | N/A (CPU) — ALREADY OVERLAPPED with GPU work |
| 10 | **Pure GPU compute floor** (memory-bandwidth floor on all quantized weights + the dense F32 lm_head + KV cache) | **~10.3 ms/tok** | Memory bandwidth floor computation: `lm_head F32 (2.95 GB) + MoE Q-mixed (1.57 GB) + MLP Q4_0 (0.54 GB) + attention Q4_0 (~0.35 GB) + KV cache reads (~0.13 GB) = ~5.5 GB/token` / 400 GB/s = **13.8 ms**. Effective utilization at 48.82 tok/s implies hf2q is achieving **5.5 / 0.02048 ≈ 268 GB/s**, not the full 400 GB/s. Some of the gap to 13.8 is compute (not 100% bandwidth-bound) and some is small inefficiencies in the kernel paths. llama.cpp at 9.35 ms with presumed lm_head quant drops this to ~3.2 GB / 9.35 ms = **342 GB/s** utilization. | — (floor) |

**Sum of instrumented components + CPU overhead:** 7.14 (lm_head) + 4.0 (MoE) + 1.35 (MLP) + 1.2 (attention) + 0.66 (RmsNorm) + 0.3 (RoPE) + 0.32 (SDPA) + 0.2 (drain) + 4.29 (CPU overhead) ≈ **19.5 ms/token**, vs measured 20.48. The **~1 ms residual** is the bookkeeping not captured by any bracketed region — embedding lookup, softcapping (3 ops), the explicit `xs + attn_out` add, `layer_scalar.broadcast_mul`, final `norm.forward`, `narrow(1, seq-1, 1)`, `normed_2d.reshape`, `unsqueeze`. These are small (each <50 μs) and disperse across the forward.

### The critical finding — pipeline is nearly saturated, CPU/GPU overlap almost complete

The relationship across the three key Run SA numbers:

```
CPU enqueue (spike_forward_ms)           : 8.33 ms
Sampler drain GPU-tail wait              :12.34 ms  (= 32 drains × 48.97 ms / 127 forwards)
                                          ─────
                                          20.67 ms
Measured wall-clock (1000/tok_per_sec)   :20.48 ms   ← within 0.9% of the sum
```

**This is exactly the behaviour of a near-saturated pipeline with minimal overlap slack.** At the post-Walk baseline:
- The CPU enqueues ~8.3 ms/forward worth of candle ops.
- The GPU takes ~12.3 ms/forward to execute them.
- Candle's command pool accepts forward N+1 while forward N runs, but by the time we hit the sampler drain point (every 4 forwards), the pool is full and the drain has to wait for the full 4×12.3 = ~49 ms of GPU work to complete.
- Net effect: the CPU and GPU are **not running in parallel** — they are **sequenced**, with the GPU doing the heavy lifting and the CPU idling at the drain point.

This matches the ADR line 418 "Phase C finding" on 1bNEW.3 exactly. It also explains why 1bNEW.3's projected 3-6 tok/s gain delivered only +0.25: the drain is a **pool-wide** `flush_and_wait`, not a per-tensor wait, so batching N forwards into one drain doesn't recover any GPU/CPU overlap — it just shifts when the CPU blocks.

### Alternate bound from the Run SF (lm_head) finding

If I remove lm_head from the critical path entirely (replace the 7.14 ms dense F32 matmul with a quantized Q6_K matmul at predicted ~1.5 ms), projected decode wall-clock becomes:

```
20.48 ms − (7.14 − 1.50) = 20.48 − 5.64 = 14.84 ms/token → 67.4 tok/s
```

This is a conservative projection that assumes **none of the lm_head wait overlaps with any other GPU work**. Empirically, Run SF showed a tok/s drop of only 2.3% when the lm_head matmul was force-synced in isolation — consistent with lm_head being on the strict critical path (i.e., nothing runs concurrently with it). So the projection is firm: **quantizing lm_head → ~67-70 tok/s**, assuming the same 400 GB/s bandwidth ceiling.

---

## Part 2: Residual Walk-item analysis

### 1bNEW.11 — Port llama.cpp flash-attn vec for head_dim=256 (sliding layers)

**Current cost of the sliding-layer prefill manual path:** the manual `repeat_kv + matmul + causal_mask + softmax + matmul` chain at `src/serve/gemma4.rs:827-873`. This runs 25 / 30 layers on the prefill pass but **not at all on the decode pass** — the decode path at `q_len == 1` takes the `candle_nn::ops::sdpa` vector path at `gemma4.rs:754-767` for ALL layers regardless of head_dim.

The Q4 spike (spike-Q3Q4Q5-results.md:139-154) already validated that BF16 sdpa at head_dim=512 is correct. The question is whether porting llama.cpp's `kernel_flash_attn_ext_vec_f16_dk256_dv256` for head_dim=256 closes a meaningful chunk of the decode gap.

**Measurement:** Run SG (HF2Q_SPIKE_SYNC=sdpa) measured the 30 decode-SDPA dispatches at 5.62 ms/token under forced-sync, which we now know is dominated by the 0.187 ms flush-and-wait floor per dispatch (30 × 0.187 = 5.6). The Run SA enqueue-only cost for SDPA was 0.045 ms/token total — **the actual decode SDPA cost is ≤ 0.3 ms/token**, mostly KV-cache read bandwidth.

**Verdict on 1bNEW.11 for DECODE speed: DEFER.** Porting a different SDPA kernel for decode cannot save more than 0.3 ms/token because that's the entire budget the current SDPA path consumes. The reported "7.14 ms savings" at the lm_head site (Run SF) is an order of magnitude larger. Any Walk effort should go to lm_head quantization first.

**Verdict for PREFILL speed:** moderate potential. The sliding-layer prefill path still builds F32 intermediate tensors and allocates `repeat_kv` copies. On the 187-token canonical prompt the prefill is ~167 ms; on a 3000-token prompt it's proportionally larger. But prefill is **not on the decode-tok/s critical path**, so it does not contribute to the 107 tok/s End gate by the current benchmark definition.

### Walk-correctness drift owner (`The` vs `To` argmax gap)

Post-1bNEW.6 the gap stands at **+0.7701 logit** (hf2q prefers `The` over `To`, ADR line 713 / crawl_progress.md). llama.cpp prefers `To` over `The` by **0.1176 logprob (~0.12 logit)** (ADR line 194). The combined direction drift across the four Walk items (1bNEW.1/3/4/6) has been toward `The`, not toward `To`, even though each port tightens FP reduction toward llama.cpp's rounding. This is internally consistent: each kernel port replaces the candle-chain reduction with llama.cpp's reduction, but the two tools' **top-level graph differs** in a way the kernel ports don't touch.

**Candidate drift owners** (ordered by likelihood):

1. **The F32 dense lm_head matmul itself** at `src/serve/gemma4.rs:1879`. llama.cpp's lm_head is a Q6_K quantized matmul (`ggml_mul_mat` on a Q6_K tensor). **Quantized dot products use different reduction order and different precision** than a dense F32 matmul. The 262144-wide vocab reduction is the LAST numerical step before argmax; a reduction-order difference at vocab scale is exactly where a 0.12 logit gap can live. **This is my highest-confidence guess.** Evidence: the gap is at vocab-level softmax, and the lm_head is the only vocab-level op on the forward pass. File:line: `/opt/hf2q/src/serve/gemma4.rs:1876-1879`.
2. **Softcapping operation order.** Line 1887: `((logits / sc)?.tanh()? * sc)?`. llama.cpp's softcapping equivalent at `src/llama-graph.cpp` uses the same formula but in BF16 or F16 depending on the graph type. If llama.cpp does the `/sc` in the matmul's output precision (F32 from the Q6_K kernel) and hf2q does it in pure F32, the rounding is identical — but if llama.cpp folds softcapping into the matmul output cast chain, there's a tiny drift point. Lower confidence than #1 but easy to check.
3. **The final `output_norm` reduction order.** `src/serve/gemma4.rs:1875`. The fused RmsNorm port is byte-identical to llama.cpp's `kernel_rms_norm_fuse_impl` (Phase A unit tests at 2.384e-7 max |Δ|, single ULP). Unlikely to move the argmax.
4. **Residual accumulation order at every layer.** mlx-lm does `h = residual + h` (gemma4_text.py:339-340). llama.cpp does `attn_out = ggml_add(ctx0, cur, inpL)`. hf2q does `let xs = (xs + &attn_out)?` at gemma4.rs:1521. These should all be associative-equivalent in F32 — but the **order of summands matters** at F32 for very small differences. This is lower-priority than #1 because it's been un-fused (1bNEW.0b) to match both references.

**Highest-value next investigation (spike target, NOT a Walk item yet):** At `gemma4.rs:1876-1879`, dump the F32 logits vector to disk, then dump the llama.cpp equivalent post-quantized-matmul logits at the same position, and compute the per-vocab-position |Δ|. If the 1.2e-1 gap at the top-1 position corresponds to systematic bias across the full vocab (mean_abs_Δ ≈ 1e-2 or more), it's the reduction-order difference from F32 vs Q6_K. If it's concentrated on the top-k only, it's a softcap or residual-accumulation bug.

### Sliding-window mask bug at `q_len > sliding_window`

Still open (ADR line 141 Walk Exception). Does not affect any decode-speed benchmark (the canonical bench is 187 tokens, well under sliding_window=1024). **Priority: LOW for Phase 1b End gate; MEDIUM for Phase 2 HTTP concurrency**, because concurrent batching with long-context requests would hit it immediately. Defer — nothing on the decode critical path needs it.

### 1bNEW.13 — QMatMul 8192-dim cliff

**Still retired per Q5 spike.** The Q5 finding was that `kernel_mul_mv_q6_K_f32` latency scales sub-linearly in output dim (1.43× for 2× output), not super-linearly. At the post-Walk baseline nothing about this has changed — the same kernel is still in use at the same call sites. Re-confirming: the attention q_proj calls per layer cost ~0.035 ms synced or ~0.015 ms batched (Q5 measurement), total across 30 layers ~1-1.2 ms/token, matching the component-4 budget in the gap decomposition above. **Still retired.**

### Hidden dispatches not on the current bisect table

Instrumentation revealed several small-cost items not explicitly listed in ADR-005's bisect table, but all are accounted for in the 4.29 ms CPU overhead residual:

- **KV cache `slice_scatter + narrow + contiguous`** at `gemma4.rs:694-696` — 6 ops per layer × 30 = 180 dispatches/token, counted as part of `dispatches_per_token` but not a dedicated counter. Each writes a `[1, kv_heads, 1, head_dim]` slice and reads back the full cache as a `[1, kv_heads, seq, head_dim]` view. At decode step 128 the cache is ~315 positions long, so the `narrow + contiguous` at `KvCache::append` copies ~5 MB/layer = **150 MB / token of KV traffic** on top of the attention SDPA's own KV reads. At 400 GB/s bandwidth = **0.38 ms/token** of KV-cache-append memory traffic, already captured in component-4 of the gap table.
- **Dense F32 matmul in MoE `w_total` gather/combine** at `gemma4.rs:1497-1506`. The `per_expert_scale.index_select`, the element-wise `top_k_weights * gathered_scale`, the `broadcast_mul`, the `weighted.sum(1)` — 7 small candle ops per layer × 30. Bandwidth-negligible (`[1, top_k=8, hidden=2816] F32` = ~90 KB), CPU-overhead dominated. Already in the 4.29 ms CPU residual.
- **Layer_scalar broadcast_mul** at `gemma4.rs:1555` — 30 dispatches/token, one per layer. Each is a tiny `[1, 1, hidden] * scalar` mul.
- **Final norm + lm_head chain** at `gemma4.rs:1933-1949` — narrow + reshape + norm + reshape + matmul + t + unsqueeze + softcap div/tanh/mul = 10+ ops. lm_head dominates the cost (7.14 ms); the rest are O(0.1 ms/token) combined.

**None of these individually justify a new Walk item**, because each saves <0.5 ms and has no reference file:line port to cite. They are collectively the 4.29 ms CPU residual that overlaps GPU time already.

---

## Part 3: Walk-vs-Run verdict

### Q1: Is there any Walk-faithful path that closes the remaining 11.13 ms gap?

**ONE item covers ~5.6 ms.** No item covers the rest.

**Candidate Walk item — 1bNEW.17 (proposed): Quantized lm_head via QMatMul.**

- **What it does:** Stop dequantizing `token_embd.weight` to F32 at load time. Load it as a `QTensor` via `gguf.get_qtensor("token_embd.weight")?` (the cache path already exists at `src/serve/gguf_loader.rs:68-85`). Wrap it in a `QMatMul` and use `QLinear::forward` at the lm_head site instead of the dense `matmul(.t())`. Tie-word-embeddings is still honoured because the same `QTensor` is shared between `embed_tokens` and the lm_head — candle's `QMatMul::from_qtensor(Arc<QTensor>)` takes an Arc so the sharing works by construction.
- **Why it helps:** Reduces lm_head weight memory read from 2.95 GB F32 to 0.60 GB Q6_K (4.9× less bandwidth). Expected cost drops from 7.14 ms → ~1.5 ms (bandwidth-bound floor). **Savings: ~5.6 ms/token.** Projected post-landing tok/s: **20.48 - 5.6 = 14.88 ms → 67.2 tok/s.**
- **Correctness risk:** LOW-MEDIUM. The exact same Q6_K kernel is already in use at 360+ call sites inside the forward pass (every Q/K/V/O/gate/up/down/router dispatch). It is tested against llama.cpp byte-for-byte in the Q5 spike. The only delta is: the input to this QMatMul is `[1, 2816] F32` (same input dtype as every other QMatMul) and the weight is `[vocab=262144, 2816]` Q6_K. The only shape difference from the existing q_proj at `blk.29` (`[8192, 2816] Q6_K`) is the output dimension — 262144 vs 8192 — and Q5 already measured latency for that shape direction (sub-linear in output dim, 30.4 ns/out vs 42.4 ns/out).
- **Walk-correctness consequence:** **This is also the candidate for the `The`/`To` flip**. llama.cpp's lm_head IS a quantized matmul; hf2q's is a dense F32 matmul. The reduction-order difference at vocab scale is exactly where a 0.12 logit gap can live. Landing this item may simultaneously close both the speed gap (5.6 ms) and the Walk-correctness gap (0.12 logit). **Both gates potentially close in one item.**
- **Reference citation:** llama.cpp `build_lm_head` at `src/llama-graph.cpp:1258-1266` calls `ggml_mul_mat(ctx0, model.output, cur)` on the quantized `model.output` tensor (which is `token_embd.weight` when `tie_embeddings == true`). Uses the same `kernel_mul_mv_q6_K_f32` that llama.cpp's linear layers use. mlx-lm's `gemma4_text.py:391-395` does the equivalent via `nn.Embedding.as_linear` which keeps the quantized storage.
- **Chesterton's fence:** Why was it dense F32 in the first place? Hypothesis: at Phase 1 landing the QMatMul wrapper was not yet adapted to take the tied-embedding Arc pattern (or the dequantized version was the quick path that worked and never got revisited). The ADR does not mention this explicitly; my reading of `src/serve/gemma4.rs:1636-1642` suggests a pragmatic "just dequantize it to F32 like every other linear weight load" decision that happened to never flag because tok/s was gated on MoE+norm+RoPE at the time.
- **LOC estimate:** ~40 LOC. Add `get_qtensor` path for token_embd, thread the `QMatMul` (or a thin `QLinear`-with-no-bias wrapper) through `Gemma4Model`, keep `embed_tokens: Embedding` for the forward pass (it still needs to dequantize on demand for the `index_select` lookup at decode-input-ids time — OR use candle's `QTensor::gather` path which keeps it quantized). The `Embedding::forward` dispatch is NOT on the decode critical path (it's one index_select per forward, reads only `hidden_size` floats), so dequantizing-on-the-fly there is fine.
- **Walk citation gate:** PASSES. llama.cpp's `build_lm_head` is a live `file:line` port target.

### Q2: If Walk alone cannot close the gap, what is the smallest Run item that could?

Assuming 1bNEW.17 (quant lm_head) lands and hf2q is at ~67 tok/s, the remaining gap is **14.88 - 9.35 = 5.53 ms/token** against a 107 tok/s target.

At this point the non-lm_head critical path is:

- MoE compute (memory-bound): ~4 ms
- MLP compute: ~1.3 ms
- Attention QMatMul: ~1.1 ms
- Everything else: ~3 ms (norms, rope, sdpa, kv cache, cpu overhead)
= ~9.4 ms/token total

llama.cpp at 9.35 ms/token is running essentially at the memory-bandwidth floor (~342 GB/s effective). hf2q post-1bNEW.17 would be at 14.88 ms/token = **~195 GB/s effective**. The delta (195 → 342 GB/s) is **structural bandwidth-utilization inefficiency in candle's dispatch path**, not a per-kernel port gap. That 5.5 ms is not reachable via any single-file-port Walk item.

**Run candidates** (in approximate order of lift):

1. **Run item RUN-1 — Per-buffer wait semantics in candle.** Upstream candle change that replaces the pool-wide `flush_and_wait` at `/opt/candle/candle-metal-kernels/src/metal/commands.rs:176-202` with a per-tensor wait (using the Metal `MTLCommandBuffer::addCompletedHandler` pattern). This would restore the 1bNEW.3 projected 3-6 tok/s gain that was measured-as-zero because the pool-wide drain absorbed all potential CPU/GPU overlap (ADR line 418). Expected wall-clock saving: the CPU enqueue (8.3 ms) and the GPU compute (12.3 ms) could run in parallel instead of sequentially, saving the smaller of the two = **~8 ms/token (if perfectly parallel) or 3-5 ms (realistic)**. Run, not Walk, because **nothing in llama.cpp or mlx-lm does this exact thing** — llama.cpp uses a single command buffer per forward with explicit `commandBufferCommit` at layer boundaries; mlx has a multi-stream scheduler. Candle has neither; the fix is novel candle infrastructure, not a kernel port. Rationale: Run items write novel infra; this is the single highest-leverage novel candle-side change that doesn't duplicate a reference.

2. **Run item RUN-2 — Fuse RmsNorm into the adjacent QMatMul.** The friend's "norm-into-matmul fusion" idea. Takes the post_attention_layernorm + o_proj into one kernel; same for pre_feedforward_layernorm_2 + router_proj; similar for pre_feedforward_layernorm_1 + mlp.gate_proj and up_proj. Saves the 331 tiny RmsNorm dispatches and **halves the CPU overhead of binding them** — which at ~0.002 ms/binding × 331 = 0.66 ms doesn't move the needle by itself, but in combination with RUN-1 it collapses more of the command pool. Not Walk because neither llama.cpp nor mlx-lm does this at the Metal kernel level — llama.cpp fuses at the ggml-graph level with `ggml_mul_mat(ggml_rms_norm(x), w)` which is two separate dispatches with a temp tensor in between; the norm-into-matmul kernel-level fusion would be novel. Expected savings: **0.5-1 ms/token** (mostly from pool-pressure reduction under RUN-1).

3. **Run item RUN-3 — Single-command-buffer-per-forward submission.** Wrap a whole `Gemma4Model::forward` in one `MTLCommandBuffer`, dispatch all ~2192 ops on it, and `commit + waitUntilCompleted` at the end. This eliminates the 50-dispatch pool recycle at `/opt/candle/candle-metal-kernels/src/metal/commands.rs:14` for the duration of a forward pass. Expected savings: **1-2 ms/token** from CPU overhead reduction. Not Walk because candle's command pool doesn't support single-buffer-per-forward; this is novel candle infra. (This is what llama.cpp effectively does with its graph-scheduler, so arguably it is a reference port — but the *candle-side implementation* to support it is Run.)

4. **Run item RUN-4 — Replace `slice_scatter + narrow + contiguous` KV cache with in-place append.** The current KV cache path at `gemma4.rs:694-696` pays ~0.38 ms/token in memory traffic doing a contiguous copy of the entire cache on every decode step. llama.cpp uses an in-place append with a running offset and never copies. Expected savings: 0.3-0.5 ms/token. **Arguably Walk** — llama.cpp's KV cache at `src/llama-kv-cache.cpp` is the reference — but the implementation requires a candle-side change to `KvCache::append` to avoid the `slice_scatter` gotcha that produced the a0952e2 gibberish. Could be phrased as a Walk item with an upstream candle PR.

**The smallest Run item that individually closes the gap is RUN-1** (per-buffer wait semantics). If the CPU enqueue (8.3 ms) were truly parallel with GPU compute (12.3 ms), the effective wall-clock would drop to max(8.3, 12.3) = 12.3 ms = **81 tok/s**. Combined with 1bNEW.17 (quant lm_head), this becomes max(8.3, 12.3 − 5.6) = max(8.3, 6.7) = 8.3 ms = **120 tok/s**, actually exceeding the 107 End gate.

**So the 107 tok/s End gate is plausibly reachable via: Walk 1bNEW.17 (quant lm_head, ~5.6 ms saved) + Run RUN-1 (per-buffer waits, ~3-5 ms saved)**. This is the narrowest path.

### Q3: Honest estimate of final tok/s if Walk runs to completion

Walk alone, executing every plausible Walk item remaining:

- 1bNEW.17 (quant lm_head): +5.6 ms saved → 67.4 tok/s
- 1bNEW.11-variant (port flash-attn vec for head_dim=256): +0.2 ms saved (decode), ~4 ms saved (prefill only, doesn't affect decode tok/s) → 68.0 tok/s
- KV-cache in-place append (argued as Walk — reference is llama-kv-cache.cpp): +0.4 ms → 69.8 tok/s
- Any other Walk items with a live reference citation: none identified at this time
- Walk-correctness drift close-out (doesn't move speed, but unblocks Layer B fixture): +0 tok/s

**Walk ceiling: ~68-70 tok/s.** Not reaching 107.

Sensitivity: if the 1bNEW.17 savings are 4.0 ms instead of 5.6 (e.g. if the Q6_K matmul is not fully bandwidth-limited at vocab-scale), the Walk ceiling shifts to ~60 tok/s. If savings are 6.5 ms (optimistic, the matmul is strictly bandwidth-bound), Walk ceiling shifts to ~73 tok/s. **The honest range is 60-73 tok/s for pure Walk.**

### Q4: Is Anti-Goal #12 ("escape-hatch ship at 75-85 tok/s REJECTED") still defensible?

**Restated without judgment, because this is a user decision, not a researcher decision:**

The arithmetic at hand is:

- Post-Walk projected ceiling: 60-73 tok/s (measured by bottoms-up decomposition of the critical path)
- End gate: 107 tok/s (ADR line 161, derived from llama.cpp's measured 107 tok/s on the same hardware and GGUF)
- Gap between Walk ceiling and End gate: **35-47 tok/s**, requiring a ~3-5 ms/token saving that has no identified Walk port
- The only Walk-faithful item that closes ≥1 ms/token is 1bNEW.17 (quant lm_head), and by itself it does not cross the End gate
- The smallest identified Run item (RUN-1: per-buffer wait semantics) closes the remaining ~3-5 ms by eliminating CPU/GPU serialization, which has no reference citation in either llama.cpp or mlx-lm because both tools have fundamentally different command-submission architectures than candle

**What the ADR currently guarantees under Anti-Goal #12:** Phase 1b does not ship if it's not at 107 tok/s. The current ceiling under pure Walk is 60-73; shipping 1bNEW.17 puts us near the **bottom** of that range (67). Under the current ADR, this would not ship Phase 1b.

**What would change if Anti-Goal #12 were revisited:** not a researcher call. Surfaced arithmetic only — the user can weigh it against the Walk-discipline costs of opening Run.

---

## Summary of measured evidence

| Measurement | Value | How measured |
|---|---|---|
| Post-Walk median tok/s on HEAD `4e7fe31` | 48.82 | 5-run canonical `--benchmark` |
| Per-forward wall-clock | 20.48 ms | = 1000 / 48.82 |
| Per-forward sync reference wall-clock (SB) | 20.82 ms | `HF2Q_SPIKE_SYNC=forward` |
| Enqueue-only CPU time per forward | 8.33 ms | Run SA, `spike_forward_ms_per_token` |
| lm_head F32 matmul wall-clock (isolated, SF) | 7.14 ms | `HF2Q_SPIKE_SYNC=lm_head` — median tok/s barely moves (48.82→47.69) |
| lm_head predicted from bandwidth | 7.38 ms | 2.95 GB / 400 GB/s |
| lm_head weight memory footprint (F32) | 2.95 GB | 262144 × 2816 × 4 bytes |
| lm_head weight memory footprint (Q6_K hypothesis) | 0.60 GB | 262144 × 2816 × 0.8125 bytes |
| Projected lm_head cost if Q6_K | 1.50 ms | 0.60 GB / 400 GB/s |
| Expected saving from 1bNEW.17 | 5.64 ms | 7.14 − 1.50 |
| Projected post-1bNEW.17 tok/s | 67.2 | 1000 / (20.48 − 5.64) |
| MoE weight memory footprint per token (Q-mixed, top_k=8) | 1.57 GB | 30 × 8 × (2×704 + 704) × 2816 × 0.5 bytes/weight |
| MoE predicted cost (memory-bound floor) | 3.92 ms | 1.57 / 400 |
| MLP weight traffic per token (Q4_0 dense) | 0.54 GB | 30 × 3 × 2816 × 2112 × 0.5 |
| MLP predicted cost | 1.35 ms | 0.54 / 400 |
| Attention weight traffic per token (Q4_0) | ~0.35 GB | Rough estimate from QKVO shapes × 30 layers |
| Attention predicted cost | ~0.88 ms | 0.35 / 400 |
| Forced-sync flush_and_wait floor | 0.15-0.20 ms | Measured from per-dispatch sync runs (SC/SD/SE/SG/SH) — cross-confirms Q5 result |
| CPU overhead (residual, not captured by any region) | 4.29 ms | 8.33 (forward CPU) − 4.04 (sum of per-region CPU) |
| Walk-correctness `The`/`To` gap | +0.77 logit (hf2q), +0.12 logprob (llama.cpp) | ADR line 713, crawl_progress.md |
| Top-1 drift direction | toward `The` (away from llama.cpp's `To`) | ADR lines 713, 491, 265 |

---

## Citation trail

- Baseline spike binary: `main` HEAD `4e7fe31c17a30d741549de3fe035c77825c1d7c4` (verified via `git rev-parse HEAD`).
- Instrumentation sites (all reverted before returning): `src/serve/gemma4.rs` DispatchCounters struct, `Gemma4Model::forward` lm_head bracket, `MoeBlock::forward_fused` MoE bracket, `RmsNorm::forward` and `forward_with_post_residual` and `rms_norm_unit` rms bracket, `RotaryEmbedding::apply` rope bracket, `Attention::forward` decode SDPA bracket, `QLinear::forward` qlinear bracket; `src/serve/mod.rs` `drain_window` sampler bracket and metrics.txt emission extension.
- lm_head F32 path: `/opt/hf2q/src/serve/gemma4.rs:1519` (`lm_head_weight: Tensor`), `:1636` (`embed_w = gguf.get_tensor("token_embd.weight", MODEL_DTYPE)?`), `:1836` (`let lm_head_weight = embed_w`), `:1879` (`normed_2d.matmul(&self.lm_head_weight.t()?)?`). MODEL_DTYPE = F32 at line 66.
- lm_head dequantization path: `/opt/hf2q/src/serve/gguf_loader.rs:50-57` (`get_tensor` calls `qt.dequantize(&self.device)?.to_dtype(dtype)`).
- Quantized path already available: `/opt/hf2q/src/serve/gguf_loader.rs:68-85` (`get_qtensor` — keeps QTensor quantized on device).
- llama.cpp lm_head reference: `/opt/llama.cpp/src/llama-graph.cpp` `build_lm_head` — quantized matmul path via `ggml_mul_mat` on the quantized `model.output` tensor (the same `token_embd.weight` in the tied-embeddings case).
- Candle SDPA decode vector path: `/opt/hf2q/src/serve/gemma4.rs:754-767` (`q_len == 1` branch calling `candle_nn::ops::sdpa`).
- Candle command pool `flush_and_wait` (pool-wide drain): `/opt/candle/candle-metal-kernels/src/metal/commands.rs:176-202`.
- `DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER` at `commands.rs:14`.
- Q3 spike methodology (referenced for the per-sync floor pattern): `/opt/hf2q/docs/spike-Q3Q4Q5-results.md:168-276`.
- Q5 spike kernel-cost floor data: `/opt/hf2q/docs/spike-Q3Q4Q5-results.md:280-394`.
- ADR-005 post-Walk status: `/opt/hf2q/docs/ADR-005-inference-server.md:713` (cumulative Walk progress, remaining gap 58.29 tok/s).
- ADR-005 Anti-Goal #12: `/opt/hf2q/docs/ADR-005-inference-server.md:608`.
- Gemma 4 config (shape figures): `/opt/hf2q/src/serve/config.rs:74-99`.
- MoE fused kernel dispatch: `/opt/hf2q/src/serve/gemma4.rs:1246-1268` (gate_up), `:1335-1356` (down).
- Fused RmsNorm kernel: `/opt/hf2q/src/serve/rms_norm_kernel.rs:398-606`.
- Fused RoPE kernel: `/opt/hf2q/src/serve/rope_kernel.rs:*`.
- Sampler windowed drain: `/opt/hf2q/src/serve/mod.rs:468-522`.

---

## Return message (concise summary)

- **Part 1:** lm_head dense F32 matmul measures 7.14 ms/token (measured via forced sync at the exact call site), matching the 7.38 ms bandwidth-floor prediction for reading 2.95 GB of F32 weights at ~400 GB/s. It is single-handedly ~64% of the remaining 11.13 ms gap to the 107 tok/s End gate. Every other GPU-compute component maps to its expected memory-bandwidth floor on quantized weights (MoE ~4 ms, MLP ~1.35 ms, attention ~1.1 ms); CPU enqueue is 8.33 ms/token and overlaps GPU compute almost perfectly (CPU + GPU-deferred-at-drain = 20.67 ms, wall-clock = 20.48 ms).
- **Part 2:** 1bNEW.11 for sliding-layer decode is deferred (the entire decode SDPA budget is ≤0.3 ms/token). The `The`/`To` Walk-correctness drift's highest-likelihood owner is the F32 dense lm_head itself — a quantized matmul would change the vocab-scale reduction order and may close both the speed gap AND the correctness gap in one item. Sliding-window mask bug stays deferred. 1bNEW.13 stays retired. No hidden-dispatch bisect-table items uncovered.
- **Part 3:** Walk alone projects to **60-73 tok/s** (with a proposed 1bNEW.17 quant lm_head item being the only large remaining Walk-faithful saving). The 107 tok/s End gate requires either a Run item (smallest is RUN-1: candle per-buffer wait semantics upstream, removing CPU/GPU serialization; projected +3-5 ms/token saving) or a revisit of Anti-Goal #12.

**Final verdict:** Phase 1b End gate is **measured_unreachable under strict Walk**; one proposed Walk item (1bNEW.17 quant lm_head) closes ~5.6 ms but leaves a residual ~5.5 ms gap that maps only to Run-territory candle infrastructure work.
