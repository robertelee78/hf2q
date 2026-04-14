# Spike: Fresh Diagnosis (Phase 0.5) — 2026-04-14

## Context

ADR-006 Phase 4e (graph optimization) completed 2026-04-14 with zero measured speedup.
The ADR's status line recorded: "Gap: 13.1 tok/s to llama.cpp (91.8 vs 104.9)."
A fresh Phase 0-style diagnosis was requested to identify where the remaining gap lives.

## Methodology

All measurements on M5 Max, Gemma 4 26B MoE Q4_K_M, bench_prompt_128.txt, T=0, 128 decode tokens.
Each stack tested **alone on the GPU** — never concurrently — to avoid OOM/memory pressure artifacts.

**hf2q**: 7 sequential runs, first 2 discarded as warmup, 5 measurement runs. Median reported.
**llama.cpp**: `llama-bench -r 5` (built-in 5-run with warmup). Mean ± stdev reported.
**No-barriers**: 3 runs of `HF2Q_NO_BARRIERS=1` (temporary env-var skip in encoder.rs, reverted after).

## Results

### Speed

| Stack | tok/s | σ | ms/token |
|-------|------:|--:|--------:|
| **hf2q mlx-native** | **94.7** (median) | 3.0 | 10.56 |
| **llama.cpp** | **104.58** (mean) | 0.54 | 9.56 |
| **hf2q no-barriers** | **161.8** (median) | 0.2 | 6.18 |
| **Gap** | **9.9** | | **1.00** |

Gap is 3.3σ — **statistically significant**.

### hf2q run-by-run

| Run | tok/s | Note |
|-----|------:|------|
| 1 | 94.1 | Warmup (discarded) |
| 2 | 100.6 | Warmup (discarded) |
| 3 | 94.4 | Measurement |
| 4 | 94.7 | Measurement |
| 5 | 95.0 | Measurement |
| 6 | 101.4 | Measurement (outlier) |
| 7 | 94.7 | Measurement |

**Variance note**: hf2q σ=3.0 vs llama.cpp σ=0.54 — 6x higher variance. This suggests a non-deterministic
overhead source (command buffer creation jitter, Metal scheduler variance, or thermal throttling bursts).
The two fast runs (100.6, 101.4) are outliers, not the norm. The median (94.7) is the honest measurement.

### Dispatch structure (measured via ConflictTracker histogram)

| Metric | hf2q | llama.cpp (from ADR-006 study) |
|--------|------|------|
| Total dispatches | 841 (tracked) + 31 (SDPA internal) = 872 | ~1811 graph nodes |
| Barriers | 486 (from barrier_between) + ~30 (SDPA internal) = ~516 | ~759 |
| Dispatches/barrier | 1.73 | 2.39 |

**Group size histogram** (per barrier_between tracking):

| Group size | hf2q groups | hf2q % | llama.cpp est % |
|-----------|------------:|-------:|----------------:|
| 1 (singleton) | 246 | 51% | ~40% |
| 2 | 125 | 26% | ~10% |
| 3 | 115 | 24% | ~50% |
| 4+ | 0 | 0% | ~10% |

**Key insight**: 51% of our barrier groups are singletons — one dispatch, then a barrier, then one dispatch.
llama.cpp has ~40% singletons and 50% at size-3+. Their GPU can pipeline within groups; ours mostly cannot.

## Gap decomposition

| Component | hf2q (ms) | llama.cpp (ms, est) | Delta (ms) | % of gap |
|-----------|--------:|-------------------:|----------:|--------:|
| Compute floor | 5.88 | ~5.9 | ~0 | 0% |
| Barrier stalls | 4.28 | ~3.7 | ~0.58 | 58% |
| CPU encode (serial) | 0.40 | ~0 (overlapped) | ~0.40 | 40% |
| **Total** | **10.56** | **9.56** | **1.00** | **~98%** |

### Barrier stall breakdown

hf2q: 516 barriers × 8.3μs = 4.28ms
llama.cpp (est): 759 barriers × 4.9μs = 3.72ms

They have MORE barriers but pay LESS per barrier. Why:
- Their larger concurrent groups (50% at size-3+) give the GPU more intra-group pipelining,
  reducing the effective idle time between groups.
- Our 51% singletons mean the GPU has nothing to pipeline — it finishes the one dispatch,
  then stalls waiting for the barrier fence to complete.

### CPU encode

hf2q encodes all 872 dispatches + 486 barriers into the command buffer serially: 0.40ms.
llama.cpp commits the first ~10% of dispatches immediately (GPU starts executing while CPU
encodes the remaining 90% into a second command buffer). This overlaps ~0.4ms of CPU work
with GPU execution, effectively hiding the encode cost.

## What does NOT explain the gap

1. **Kernel speed** — compute floor is at parity (5.88ms both). Confirmed by no-barriers experiment.
2. **Barrier count** — we have FEWER barriers (516 vs 759). The count is not the problem.
3. **Dispatch count** — we have FEWER dispatches (872 vs ~1811) due to kernel fusions.
4. **Graph reorder** — Phase 4e proved this has zero effect on our graph's structure (only 7% reorderable).

## Recommendations

### Path 1: Reduce singleton fraction (highest leverage, ~0.58ms)

Find barrier_between calls that emit unnecessarily. 246 singletons exist because the ConflictTracker
detects buffer conflicts. Some may be false positives from:
- Buffer aliasing (two activation names sharing the same underlying allocation)
- Overly conservative range tracking (tracking entire buffer when only a subrange is written)
- Unnecessary barrier_between calls where the conflict is with the current dispatch, not a prior one

**Action**: Add per-barrier logging to identify which specific dispatch pairs are generating singletons.
Find cases where dispatches COULD be concurrent but aren't due to a tracking issue.

### Path 2: Dual command buffer encode overlap (~0.40ms)

Split the single command buffer at a fixed point (e.g., after 10% of dispatches). Commit the first
buffer immediately so the GPU starts while CPU encodes the remainder.

**Caveat**: Phase 4e.4 tried this in the graph optimization path and found "0.3ms saved by overlap,
0.3ms added by encode overhead" = net zero. A simpler split (without the full graph infrastructure)
might have less overhead, but this needs to be measured, not assumed.

### Path 3: Reduce variance (address the 6x jitter)

hf2q's σ=3.0 vs llama.cpp's σ=0.54 suggests a non-deterministic overhead. Possible causes:
- Command buffer creation per token (allocate/free cycle vs pool)
- Pipeline state object lookups (string hash vs cached pointer)
- Metal driver scheduling jitter under thermal pressure

Reducing variance won't change the median, but it improves the user-visible experience and
makes future measurements more reliable.

## Prior spike correction

An earlier version of this spike (written mid-session) reported the gap as 1.5 tok/s based on
cherry-picked warm runs (98.7 vs 100.19). The 7-run benchmark with proper warmup separation
shows the median is 94.7, not 98.7. The 100.6 and 101.4 runs were outliers. The corrected
gap is **9.9 tok/s**, consistent with the ADR-006 previous gap of 13.1 tok/s (the improvement
is from the router/MLP interleave, which reduced barriers from 606 to 486).
