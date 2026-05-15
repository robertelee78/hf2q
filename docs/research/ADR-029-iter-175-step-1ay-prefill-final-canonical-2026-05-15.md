# ADR-029 iter-175 Step 1ay — fresh prefill canonical 1.0396× AHEAD

**Date**: 2026-05-15
**HEAD**: hf2q `8ee4b056`, mlx-native `ff9a0ff`
**Iteration**: 57 of /loop autonomous

## Result

3-cycle alt-pair at HEAD after all env-cache wins landed:

```
hf2q    : mean 3047.43 t/s  σ 1.93 (0.06%)  samples: [3044.7, 3048.9, 3048.7]
peer-FA : mean 2931.22 t/s  σ 3.91 (0.13%)  samples: [2935.21, 2925.91, 2932.54]
ratio   : 1.0396× AHEAD (+3.96%)
```

Both σ < 0.15% — tight bench.

## Progression

| Step | hf2q pp2013 | peer-FA | ratio |
|---|---|---|---|
| Step 1ac (2026-05-15 mid-iter) | 3030 ± 5 | 2922 ± 11 | 1.0370× |
| **Step 1ay (this)** | **3047 ± 2** | **2931 ± 4** | **1.0396×** |

Net iter-175 prefill improvement: **+0.26pp** (likely Step 1at TQ_NSG+NWG
env-cache spillover — FA-vec-tq-hb is shared between decode and prefill paths).

## Final iter-175 canonical baselines (post-validation)

| Phase | hf2q | peer-FA | ratio |
|---|---|---|---|
| **decode tg200** | **95.80 ± 0.08** | **101.58 ± 0.12** | **0.9431×** (5.69% gap) |
| **prefill pp2013** | **3047 ± 2** | **2931 ± 4** | **1.0396× AHEAD** |

Coherence: PASS 2/2 (Step 1ax).

Iter-175 work is fully validated end-to-end:
- decode: +1.91pp peer ratio vs iter-100 (0.924× → 0.9431×)
- prefill: maintained AHEAD ratio (1.037-1.040×)
- coherence: byte-identical
