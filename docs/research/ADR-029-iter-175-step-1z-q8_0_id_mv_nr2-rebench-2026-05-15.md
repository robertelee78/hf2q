# ADR-029 iter-175 Step 1z — `HF2Q_Q8_0_ID_MV_NR2` re-bench at HEAD: WIDENED to -1.96%

**Date**: 2026-05-15
**HEAD**: hf2q `16582858`, mlx-native `22dc55b`
**Iteration**: 30 of /loop autonomous

## Background

iter-6 (commit `7acd4d4` on mlx-native, 2026-05-11) ported peer's
`kernel_mul_mv_id_q8_0_f32_nr2` (NR0=2, NSG=4) for gemma4-APEX-Q5_K_M's
`ffn_down_exps` (Q8_0 quant).  Initial bench at -1.3% (FALSIFIED for
throughput; coherence byte-identical).  Kept default-OFF behind
`HF2Q_Q8_0_ID_MV_NR2=1`.

Since iter-6:
* Step 1d/1e: tracked-dispatch infrastructure + Q6_K_nr2 migration
* Step 1m: precompiled metallib DEFAULT-ON (production HEAD)
* Step 1y: per-layer-phase profile measured (FFN dominates ATTN 2:1)

Per the standing rule
`feedback_levers_can_widen_opposing_regressions_2026_05_15.md`: when
opposing levers land, previously-falsified levers may WIDEN their regression
(rather than stay fixed).  Test that prediction directly.

## Hypothesis

`HF2Q_Q8_0_ID_MV_NR2=1` regresses more at HEAD than iter-6's -1.3%, because:
- Step 1m makes the baseline Q8_0 _id kernel faster (precompiled -O3)
- Step 1e migrated Q6_K nr2 to tracked-dispatch but NOT Q8_0
- The opposing-direction lever (more SGs at this kernel) becomes worse-fitting

## Method

3-cycle alt-pair thermal-fair, tg200, gemma4-APEX-Q5_K_M, M5 Max:

```
[iter-175 Step 1z] 3-cycle alt-pair re-bench HF2Q_Q8_0_ID_MV_NR2
[iter-175 Step 1z] tg200, M5 Max, gemma4-APEX-Q5_K_M

--- cycle 0 ---
  baseline      : 95.1 tok/s
  Q8_0_ID_MV_NR2: 94.0 tok/s

--- cycle 1 ---
  Q8_0_ID_MV_NR2: 93.2 tok/s
  baseline      : 95.2 tok/s

--- cycle 2 ---
  baseline      : 95.6 tok/s
  Q8_0_ID_MV_NR2: 93.1 tok/s

=== aggregate ===
baseline mean   : 95.30 t/s  samples: [95.1, 95.2, 95.6]
treatment mean  : 93.43 t/s  samples: [94.0, 93.2, 93.1]
delta           : -1.96% — REFUTED ≤-1%
```

Within-arm σ:
- baseline: 0.21 t/s (0.22%) — excellent thermal stability
- treatment: 0.40 t/s (0.43%) — well within bench protocol

## Result

**REFUTED at HEAD with -1.96% delta** — WIDER than iter-6's -1.3%.

This is the **first quantitative confirmation** of the standing rule
`feedback_levers_can_widen_opposing_regressions_2026_05_15.md`: a
previously-falsified lever (iter-6 H-Q8_0_NR2, -1.3%) measured today at
-1.96% — the regression grew by **0.66pp** as opposing levers (Step 1m
precompiled + Step 1e tracked-dispatch + general infra) landed.

`HF2Q_Q8_0_ID_MV_NR2` remains rightfully default-OFF.  Do not re-test
without first checking what new opposing levers have landed since 2026-05-15.

## Standing-context baseline update

3-cycle alt-pair baseline at HEAD: **95.30 ± 0.21 t/s** (σ-pct 0.22%).
This supersedes iter-158's 92.60 baseline (different session/build state)
as the canonical iter-175 reference.  ~+2.9% above iter-158 reflects
the cumulative effect of Steps 1d/1e/1m default-ons since iter-158.

Peer-FA at same machine same session unknown for this run — should be
benched in a future iter for an absolute ratio update.

## Cross-references

* Original iter-6 falsification commit: `mlx-native 7acd4d4`
* Standing rule confirmed: `feedback_levers_can_widen_opposing_regressions_2026_05_15.md`
* Step 1m landed (most likely opposing lever): `project_adr029_iter175_step1m_precompiled_metallib_DEFAULT_ON_2026_05_15.md`
* Step 1e tracked-dispatch landed: see `mlx-native b32b81e`
* Falsification ledger: `feedback_class_AB_lever_falsification_ledger_2026_05_12.md`
