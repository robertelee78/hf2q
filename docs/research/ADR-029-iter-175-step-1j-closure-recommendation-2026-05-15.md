# ADR-029 iter-175 Step 1j — fusion lever class already FALSIFIED; iter-175 reaches /loop ceiling

**Date**: 2026-05-15
**HEAD**: hf2q `cd663c15`, mlx-native `b32b81e`
**Iteration**: 9 of /loop autonomous

## Summary

Step 1i's "kernel fusion is the lever" hypothesis is **already empirically FALSIFIED** by ADR-029 iter-1's H6 test on `fused_post_attn_triple_norm_f32` (a 4→1 fusion combining post-attn norm + residual_add + 3 pre-FF norms = saves 90 dispatches/tok). Result: **−2.8% regression** at gemma4-APEX-Q5_K_M with byte-identical coherence. Standing decision documented in `src/debug/investigation_env.rs:665-673`: "the dispatch-fusion lever class appears to lose on Apple Metal at hidden_size=2816, top_k=8."

This closes the structural lever space for /loop-autonomous iteration on iter-175.

## What was confirmed by reading the code

`/opt/hf2q/src/debug/investigation_env.rs:657-674` (the env-flag docstring) states:

> `HF2Q_FUSED_TRIPLE_NORM=1` — replace the per-layer pair `fused_norm_add(hidden, attn_out, post_attn_w → residual)` + 3× `rms_norm(residual, w_a/b/c → out_a/b/c)` with the single `fused_post_attn_triple_norm_f32` kernel. Saves 3 dispatches/layer × 30 layers = 90 dispatches/token on gemma4 decode.
>
> **Default-OFF**: ADR-029 iter-1 H6 test on gemma4-APEX-Q5_K_M at HEAD with default-flag stack: coherence byte-identical (50-tok haiku), throughput **72.9 t/s** median (σ-pct 0.05%, n=5) vs 75.0 baseline = **-2.8% regression**.
>
> Standing decision: leave default-off; the dispatch-fusion lever class appears to lose on Apple Metal at hidden_size=2816, top_k=8.

Also consistent with `forward_mlx.rs:4839-4841` (HF2Q_SPLIT_POSTATTN_NORM bisect):
> Tests the counter-fusion hypothesis (iter-105 confirmed: on Apple Metal scheduler, more smaller dispatches outperform fewer larger fused dispatches at decode shape).

## Why fusion loses on Apple Metal at gemma4 decode shapes

The contradiction (per-dispatch launch overhead is ~10 µs but fusion still loses) resolves via Apple GPU scheduler behavior:
- Single-kernel large dispatch must fit into one threadgroup configuration (e.g. (256, 1, 1) for hidden=2816 with float4)
- Apple GPU has FIXED-cost components per kernel (warp scheduling, register allocation, instruction-cache fill) that aren't amortizable
- Smaller kernels each run very efficiently (sub-µs ALU + ~10 µs overhead)
- A single fused 4× larger kernel can't fit 4× the workload in the same time slot — it stalls on register pressure / warp limit
- Net: fusion overhead from extra-large-kernel inefficiency > savings from 3 fewer launch overheads

This is the same explanation as iter-101's `FOR_UNROLL` regression on flash_attn (register spill).

## Recapping iter-175's complete hypothesis ledger

| Hypothesis | Status | Disposition |
|---|---|---|
| H-A: per-dispatch encoder overhead | FALSIFIED (Step 1b) | CPU is <1% of wall |
| H-B: runtime Metal compile options | PARTIALLY FALSIFIED (Step 1b) | Both repos default |
| H-D: concurrency dispatch strategy | CONFIRMED 3.5pp ceiling (Step 1d) | Requires global migration (multi-day) |
| H-D2: single-site partial migration | FALSIFIED (Step 1f) | Surrounding hand-barriers neutralize |
| H-E: precompiled .metallib | TOOLCHAIN CONFIRMED, test DEFERRED (Step 1g) | Requires ~2-3 hours test harness |
| H-F: matvec kernel inefficiency | FALSIFIED (Step 1h) | Matvecs at 70-119% of peak |
| H-G: kernel fusion in FFN_NORMS | **FALSIFIED at iter-1 H6** (Step 1j, this artifact) | Apple Metal scheduler anti-fusion |
| H-C: cache/memory layout | OPEN, operator-runs Instruments | Not /loop-suitable |

## What this means for iter-175 closure

**iter-175's /loop-autonomous investigation has reached its ceiling.** All hypotheses tractable within the 5-min /loop window have been either:
- FALSIFIED via direct experiment (H-A, H-B, H-D2, H-F, H-G)
- CONFIRMED but require multi-day engineering to capture (H-D global migration, H-E full migration)
- DEFERRED to operator-runs (H-C Apple Instruments)

The remaining 6-8% peer-FA gap is **not closeable within /loop iteration window**. Same conclusion as iter-174's "EXHAUSTED" verdict — but iter-175 reaches it via a different and more thorough route, with several specific multi-day engineering paths identified.

## Recommended operator-level decisions

1. **H-D global migration** (multi-day to multi-week): migrate all ~400 hand-placed `enc.memory_barrier()` sites to `dispatch_tracked_*`. Target: ~3.5pp. Risk: similar in size to ADR-031 Phase B (which yielded 0% wall benefit despite similar effort). Migration infrastructure already landed at `mlx-native b32b81e`.

2. **H-E definitive test** (~2-3 hours): write a parallel test harness comparing `xcrun metal -O3 .metallib` vs runtime source compile on one shader. Toolchain verified working in Step 1g. If H-E confirms, kicks off a 1-2 week full-shader precompile migration.

3. **H-C cache investigation** (operator-runs Apple Instruments): capture Metal System Trace, identify cache-miss patterns. Multi-day analysis follows.

4. **Accept current state as floor**: hf2q at 0.92× peer-FA on M5 Max + gemma4-APEX-Q5_K_M may be the structural floor for the current architecture + Apple SDK + hardware. Standing-context corrections accumulated through iter-175 are sound; ship and move to other priorities.

## What iter-175 produced (durable deliverables)

- **mlx-native b32b81e**: `dispatch_tracked_*` migration infrastructure (default-OFF, safe, correctness-tested). Available for future H-D global migration work.
- **9 research artifacts** at `docs/research/ADR-029-iter-175-*`:
  - Step 1: dispatch baseline
  - Step 1b: H-A/H-B falsification
  - Step 1d: H-D 4-arm bench (3.5pp ceiling)
  - Step 1e: dispatch_tracked migration (default-OFF)
  - Step 1f: H-D2 falsification
  - Step 1g: H-E toolchain confirmed
  - Step 1h: per-kernel bench reframe
  - Step 1i: per-layer-phase attribution
  - Step 1j: fusion lever falsified via Chesterton's fence (this artifact)
- **ADR-029 fully updated** with iter-175 entries for each step
- **Multiple memory entries** capturing non-obvious findings:
  - Step 1: top kernels already peer-ported (avoid re-attempt)
  - Step 1d: concurrency lever CONFIRMED

## Cross-references

- iter-1 H6 falsification: `src/debug/investigation_env.rs:665-673`
- iter-105 counter-fusion finding: `src/serve/forward_mlx.rs:4839-4841`
- iter-101 FOR_UNROLL register spill: ADR-029 §iter-101
- All Step 1* artifacts in `docs/research/`
