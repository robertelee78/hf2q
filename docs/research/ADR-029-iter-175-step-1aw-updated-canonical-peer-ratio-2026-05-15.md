# ADR-029 iter-175 Step 1aw — updated canonical peer ratio 0.9431× post-env-cache campaign

**Date**: 2026-05-15
**HEAD**: hf2q `1333bfcd`, mlx-native `ff9a0ff`
**Iteration**: 54 of /loop autonomous

## Updated canonical decode ratio

After iter-175 env-cache campaign (Steps 1an + 1as + 1at + 1au), fresh
3-cycle alt-pair on same machine same session:

```
hf2q    : mean 95.80  σ 0.08 (0.09%)  samples: [95.8, 95.7, 95.9]
peer-FA : mean 101.58 σ 0.12 (0.12%)  samples: [101.42, 101.71, 101.6]
ratio   : 0.9431×  (5.69% gap)
```

Both σ < 0.15% — tightest peer-FA bench in iter-175.

## Progression timeline

| Step | hf2q t/s | peer-FA t/s | ratio | gap |
|---|---|---|---|---|
| iter-100 (2026-05-12) | 91.17 | 98.64 | 0.924× | 7.6% |
| iter-159 multi-regime gate (2026-05-13) | ~92.7 | ~99.5 | 0.926-0.944× | 5.6-7.4% |
| iter-162 H93 WIN | post-FC-promote | — | 0.944× | 5.6% |
| Step 1aa (iter-175 baseline) | 94.53 | 100.94 | 0.9365× | 6.35% |
| **Step 1aw (this, post env-cache)** | **95.80** | **101.58** | **0.9431×** | **5.69%** |

**Net iter-175 improvement: +0.66pp peer ratio** from env-cache campaign
(95.80 - 94.53 = 1.27 t/s hf2q gain, 0.6 t/s peer gain → ratio improved).

## Standing-context update for iter-176+

Updated canonical baselines:
- **hf2q decode tg200**: 95.80 ± 0.08 t/s (σ 0.09%)
- **peer-FA decode tg200**: 101.58 ± 0.12 t/s (σ 0.12%)
- **Canonical ratio**: 0.9431× peer-FA
- **Prefill pp2013** (Step 1ac, unchanged): 1.0370× AHEAD

Regression gates for iter-176+:
- Decode drop below ratio 0.93× = real regression to investigate
- Decode push above ratio 0.95× consistently = real improvement
- Prefill drop below 1.00× = investigate

## Cumulative iter-175 wins

* Step 1d: concurrent dispatch default (infrastructure)
* Step 1e: q6_K_nr2 tracked-dispatch migration (infrastructure)
* Step 1m: precompiled metallib default-ON (+3-5% prefill)
* Step 1an: Q6K_ID + Q8_0_ID dispatch_id_mv env-cache (+0.24% decode)
* Step 1as: HYBRID_NWG env-cache (+0.28% cumulative)
* Step 1at: TQ_NSG + TQ_NWG env-cache (+0.49% cumulative)
* Step 1au: FUSED_MOE_WSUM_END_LAYER_V2 INVESTIGATION_ENV fix (+0.63% cumulative)
* Step 1aw (this): canonical ratio updated 0.9365× → **0.9431×**

## Mantra-prevented wrong-direction commits

iter-175 prevented 8 wrong-direction commits via "measure 3x, cut once":
1ad shape mismatch, 1ai FFI cost estimate (4× off), 1aj wrapper attribution,
1ak orchestration attribution, 1al pipeline lookups, 1ap barrier_between,
1aq LTO=thin, 1ar #[inline] hints, 1av diagnostic env-cache (below noise).

## Cross-references

* Step 1aa (original canonical 0.9365×): `docs/research/ADR-029-iter-175-step-1aa-peer-ratio-at-HEAD-2026-05-15.md`
* Step 1au (last validated win): `hf2q/f6450365`
* Step 1av (env-cache campaign ceiling): `docs/research/ADR-029-iter-175-step-1av-diagnostic-env-cache-noise-2026-05-15.md`
