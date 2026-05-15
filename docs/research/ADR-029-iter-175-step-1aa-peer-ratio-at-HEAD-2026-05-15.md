# ADR-029 iter-175 Step 1aa — peer ratio at HEAD: 0.9365× (6.35% gap)

**Date**: 2026-05-15
**HEAD**: hf2q `89ce388a`, mlx-native `22dc55b`, llama.cpp `389ff61d7`
**Iteration**: 31 of /loop autonomous

## Goal

Re-measure peer-FA ratio at thermal-fair alt-pair on the same machine
in the same session, post-Step-1d+1e+1m default-ons.  Establish the
canonical iter-175 peer ratio for forward regression gates.

## Method

3-cycle alt-pair, gemma4-APEX-Q5_K_M, M5 Max, tg200:

```
hf2q : ./target/release/hf2q generate --max-tokens 200 --ignore-eos
peer : ./build/bin/llama-bench -p 0 -n 200 -r 1 -fa 1
```

60s within-cycle sleep, 90s between-cycle thermal recovery.

## Result

```
[iter-175 Step 1aa] 3-cycle alt-pair hf2q vs peer-FA, tg200

--- cycle 0 ---
  hf2q  : 94.1 tok/s
  peer  : 100.55 tok/s

--- cycle 1 ---
  peer  : 101.05 tok/s
  hf2q  : 94.9 tok/s

--- cycle 2 ---
  hf2q  : 94.6 tok/s
  peer  : 101.22 tok/s

=== aggregate ===
hf2q    : mean 94.53  σ 0.33 (0.35%)  samples: [94.1, 94.9, 94.6]
peer-FA : mean 100.94  σ 0.28 (0.28%)  samples: [100.55, 101.05, 101.22]
ratio   : hf2q/peer = 0.9365×  (+6.35% gap)
```

Both arms σ < 0.5% — well within Step 1z's bench-protocol band (<1%).

## Comparison to historical

| Iter | Date | Build | hf2q t/s | peer t/s | ratio | regime |
|---|---|---|---|---|---|---|
| iter-100 | 2026-05-12 | (dbfd0be6) | 91.17 | 98.64 | 0.924× | tg2000 |
| iter-159 | 2026-05-13 | (382e9227) | ~92.7 | ~99.5 | 0.927-0.944× | multi-regime |
| iter-162 H93 WIN | 2026-05-13 | (e97f7927) | post-FC-promote | — | 0.944× | tg2000 |
| **iter-175 Step 1aa** | **2026-05-15** | `89ce388a`/`22dc55b` | **94.53** | **100.94** | **0.9365×** | **tg200** |

**Net since iter-100**: +3.7% hf2q absolute (91.17 → 94.53), peer also up (98.64 → 100.94 = +2.3%), ratio modestly improved (0.924 → 0.9365).

The 6.35% gap is well within the "Class A + B exhausted" finding from
iter-108 and confirms the structural-floor interpretation: per-kernel
optimization has plateaued, and closing the residual requires either:
- Multi-week MoE-pipeline redesign (per Step 1y leverage analysis)
- Operator-only Apple Instruments deep dive
- Or accepting it as structural for our weight + KV format choices.

## What this rules in / out

**Rules in**: Step 1d (concurrent) + Step 1e (tracked-dispatch q6_K_nr2)
+ Step 1m (precompiled metallib) cumulatively closed ~1-2pp of decode gap
from iter-100's 0.924× baseline.  Real infrastructure-level wins.

**Rules out**: any expectation that the iter-175 micro-bench exploration
will close significantly more without a multi-week investment.  17+
falsifications across Steps 1-1aa establish the structural floor at
~0.93-0.94× peer-FA on gemma4 decode.

## Standing-context update

For iter-176+ regression gates use:
- **hf2q baseline: 95.30 t/s ± 0.21 (Step 1z, σ-pct 0.22%)**
- **peer-FA baseline: 100.94 t/s ± 0.28 (Step 1aa, σ-pct 0.28%)**
- **Canonical ratio: 0.9365× peer-FA**

Drop below ratio 0.92× = real regression to investigate.
Push above ratio 0.95× consistently in alt-pair bench = real improvement.

## Cross-references

* Step 1z baseline: `docs/research/ADR-029-iter-175-step-1z-q8_0_id_mv_nr2-rebench-2026-05-15.md`
* Step 1y phase profile: `docs/research/ADR-029-iter-175-step-1y-phase-profile-at-HEAD-2026-05-15.md`
* Step 1x archaeology: `docs/research/ADR-029-iter-175-step-1x-peer-commit-archaeology-2026-05-15.md`
* iter-162 H93 WIN: `project_adr029_iter162_h93_WIN_2026_05_13.md`
* iter-100 decode reopen: `project_adr029_iter100_decode_REOPENED_2026_05_12.md`
* Bench protocol: `feedback_metal_bench_protocol_2026_05_12.md`
