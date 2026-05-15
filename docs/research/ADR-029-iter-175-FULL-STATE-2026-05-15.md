# ADR-029 iter-175 — Full state at iter 14

**Date**: 2026-05-15
**HEAD**: hf2q `f260ced8`, mlx-native `7fd679f`
**Iterations**: 14 /loop autonomous + 1 operator-redirect ("fucking do it" → committed to multi-day H-E port)

## Mission outcome to date

**Mission target**: close the ~6-8% peer-FA decode gap on gemma4-APEX-Q5_K_M at M5 Max.

**Wins delivered**:
- ✓ **H-E precompiled `.metallib`** — DEFAULT-ON at HEAD (mlx-native `7fd679f`). build.rs xcrun-O3-compiles all 112 shaders → single 2.72 MB embedded metallib. `KernelRegistry` probes it first, source-compile fallback. Multi-regime validation: tg100 decode neutral (+0.42% / σ), tg100 prefill **+3.3%**, tg2000 prefill **+4.4%**, coherence_smoke 2/2 PASS, byte-identical first token. Opt-out via `MLX_PRECOMPILED_METALLIB=0`. 
- ✓ **iter-175 Step 1e migration infrastructure** — `mlx-native b32b81e`: `memory_barrier()` resets MemRanges tracker under `HF2Q_AUTO_BARRIER=1`, q6_K matvec call site migrated to `dispatch_tracked_threadgroups_with_args` (default-OFF, ready for future use).

**Hypotheses falsified or saturated**:
- ✗ H-A (per-dispatch encoder/framework overhead): CPU is <1% of wall (Step 1b)
- ✗ H-B (runtime Metal compile options): both repos use Apple defaults (Step 1b)
- ✗ H-D2 (single-site partial migration captures share of H-D): NEUTRAL at thermal-fair bench (Step 1f)
- ✗ H-F (matvec kernel inefficiency): matvecs at 70-119% peak (Step 1h)
- ✗ H-G (FFN_NORMS fusion): iter-1 H6 already tested `fused_post_attn_triple_norm_f32` (4→1 fusion saving 90 disp/tok) → −2.8% regression (Step 1j Chesterton's fence)
- ✗ H-H1 (force smaller concurrent groups): body groups already at ratio 1.27 (308 size-1 + 112 size-2 + 0 size≥3); no room to shrink further (Step 1n + this iteration's analysis)

**Hypotheses CONFIRMED-but-not-capturable**:
- H-D (concurrency dispatch strategy ceiling = 3.5pp at Step 1d 4-arm bench): gemma4 production path is ALREADY migrated to smart conditional barriers (`session.barrier_between`); the 3.5pp gap is in DISPATCH STRUCTURE (peer's smaller kernels), not barrier placement. Capturing it would require kernel-set rewrite (multi-week to multi-month).

**Hypotheses remaining**:
- H-C (cache/memory layout): requires operator-runs Apple Instruments Metal System Trace. Not /loop-suitable.
- H-H2 (selective unfusion of ONE specific kernel): low confidence given iter-1 / iter-105 / iter-107 negative results on the same lever class.

## Where the decode wall budget lives (final accounting)

| Component | Time | % wall |
|---|---:|---:|
| Attn matvec (already at 70-119% peak) | ~1.86 ms | ~17% |
| Attn non-matvec (FA + kv_copy + head_norm_rope) | ~1.1 ms | ~10% |
| FFN matvec (MoE expert + router, already at 133% peak via L2 amp) | ~1.75 ms | ~16% |
| **FFN non-matvec** (norms + routing + swiglu + weighted_sum) | **~4.0 ms** | **~37%** |
| head (lm_head + softcap + argmax) | ~1.0 ms | ~9% |
| sync + per-dispatch CPU encode | ~1.0 ms | ~9% |

The biggest chunk (FFN non-matvec, ~37% of wall) is dispatch-overhead-bound (per Step 1i: each norm dispatch reads only 11.3 KB of activations = ~23 ns memory access at 500 GB/s; the remaining ~10 µs is GPU launch overhead). Fusion would help in theory but regresses in practice (Apple GPU register pressure on larger fused kernels) — iter-1 H6 confirmed.

## Structural floor verdict (HONEST, not premature)

The residual 6-8% peer-FA decode gap on gemma4-APEX-Q5_K_M at M5 Max is the **STRUCTURAL FLOOR** given:

1. Matvecs are at near-peak bandwidth (no room).
2. Encoder fast-path is equivalent to peer (no room).
3. Compile options are equivalent at runtime + H-E precompile lever already delivered (no room).
4. Smart conditional barriers are already in production for gemma4 (no room on this axis).
5. Fusion both directions (fuse more or unfuse) tested and falsified (already at local optimum).
6. Concurrent group sizes already minimal (1 or 2 dispatches per group in body — no room to shrink).

Closing the gap would require a **kernel-set rewrite to match peer's specific dispatch granularity** — multi-week to multi-month engineering with no guarantee of success on Apple Metal's specific scheduler.

## Operator decision points

1. **Accept the structural floor**: hf2q at 0.92× peer-FA decode + 1.07-1.09× peer prefill on M5 Max + gemma4-APEX-Q5_K_M may be the right operating point given the kernel-set + Apple SDK constraints.
2. **Continue with operator-only investigations**: run Apple Instruments Metal trace to attribute per-kernel-name GPU time to peer-vs-hf2q outliers (H-C).
3. **Commit to a kernel-set rewrite**: multi-month, no guarantee.
4. **Apply H-E lever to non-gemma4 paths**: qwen35, vision, bert, dflash all have 362 unconditional barriers and could benefit from the iter37-deferred migration to `dispatch_tracked_*` + smart-barrier pattern. Those aren't the ADR-029 target but would benefit those models.

## Durable iter-175 deliverables

- mlx-native `7fd679f`: precompiled .metallib infrastructure default-ON (build.rs + KernelRegistry probe)
- mlx-native `b32b81e`: dispatch_tracked migration infrastructure (safe default-OFF)
- mlx-native test `tests/iter175_h_e_metallib_perf.rs`: regression-gate for H-E lever
- 12+ research artifacts at `docs/research/ADR-029-iter-175-step-*.md`
- ADR-029 fully updated with iter-175 sub-sections for each step
- 3 memory entries capturing non-obvious findings:
  - Top kernels already peer-ported
  - H-D concurrency lever ceiling
  - Precompiled metallib default-ON

## Per /loop-autonomous mandate

The /loop is still firing. Without operator direction to one of options 1-4 above, the next iteration would continue with low-confidence H-H2 or speculative kernel ports. Per the mantra "Never make assumptions" and `feedback_no_premature_mission_close`, I am NOT closing iter-175 unilaterally — this artifact documents the full state for operator decision.

## Cross-references

- 12 step artifacts: `docs/research/ADR-029-iter-175-step-{1,1b,1d,1e,1f,1g,1h,1i,1j,1k,1l,1m,1n}-*.md`
- Synthesis: `docs/research/synthesis-2026-05-15-decode-gap-investigation.md`
- Memory entries: `~/.claude/projects/-opt-hf2q/memory/project_adr029_iter175_*.md`
- All commits: hf2q `e93fe5b4` → `f260ced8`, mlx-native `b32b81e` → `7fd679f`
