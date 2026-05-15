# ADR-031 Phase B iteration 2 — B8 perf re-bench results

**Date**: 2026-05-15
**Bench operator**: launcher (claude-impl bench stalled at cycle 1; launcher re-ran from scratch)
**Hardware**: M5 Max, thermal-fair alt-pair protocol per `feedback_metal_bench_protocol_2026_05_12`
**HF2Q_B7**: `ce19198d` (FIX-1 INV-4 unconditional recv + FIX-2 single unsafe + FIX-3 registry on Err + FIX-4 HF2Q_PARALLEL_ENCODE_KV_THRESHOLD default 512)
**HF2Q_MAIN**: `e86831ab` (Phase B base, pre-Phase-B)
**GGUF**: `gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf`
**Protocol**: 75s cool-downs, one instance at a time, `--prompt "Q." --ignore-eos --temperature 0`

## tg100 alt-pair (5 cycles)

| Arm | Samples | Mean | σ | σ% |
|---|---|---|---|---|
| MAIN (e86831ab) | 95.3, 95.8, 96.1, 96.2, 95.9 | 95.86 | 0.34 | 0.36% |
| PARALLEL=0 (B7) | 96.0, 96.0, 96.2, 93.5, 95.9 | 95.52 | 1.11 | 1.16% (cycle 4 outlier 93.5) |
| **PARALLEL=1 (B7)** | 95.9, 94.6, 96.2, 95.9, 95.3 | **95.58** | 0.62 | 0.65% |

**Deltas**:
- PARALLEL=1 vs PARALLEL=0: **+0.06 t/s** — within spec ±0.3 gate ✓
- PARALLEL=1 vs MAIN: **−0.28 t/s** — within spec ±0.3 gate ✓

**Interpretation**: FIX-4 kv-depth threshold (default 512) works as designed at tg100.  seq_pos peaks at ~100, below threshold → parallel mode falls back to serial → PARALLEL=1 ≈ PARALLEL=0.  Original Phase B v1 −1.17% regression at tg100 (commit `7468cdaf`) is **eliminated**.

## tg2000 alt-pair (3 cycles)

| Arm | Samples | Mean | σ | σ% |
|---|---|---|---|---|
| MAIN | 93.0, 92.1, 92.8 | 92.63 | 0.47 | 0.51% |
| PARALLEL=0 | 93.3, 92.8, 92.7 | 92.93 | 0.32 | 0.35% |
| **PARALLEL=1** | 91.9, 92.8, 92.6 | **92.43** | 0.47 | 0.51% |

**Deltas**:
- PARALLEL=1 vs PARALLEL=0: **−0.50 t/s** — just outside spec ±0.3 gate ⚠
- PARALLEL=1 vs MAIN: **−0.20 t/s** — within spec ±0.3 gate ✓

**Interpretation**: At tg2000 (seq_pos crosses 512 threshold around token ~412), parallel path actively engages.  PARALLEL=1 shows a small regression vs PARALLEL=0 (−0.50 t/s, −0.54%) — slightly outside the spec ±0.3 gate but within measurement noise floor (σ 0.47 on both arms).  This is the same R-B7 ceiling Queen flagged in Phase 1: "Phase B v1 deliberately serializes COMMITS via mpsc::recv ordering; encoding is parallel but commit overhead is serial.  Realistic gain ceiling for v1 is small."

The cycle-1 PARALLEL=1 sample at 91.9 t/s is the lowest of the 3.  Dropping it as a warm-up artifact (cycles 2-3 show 92.8 and 92.6) yields a warmed-mean of 92.70, vs PARALLEL=0 warmed (92.8 + 92.7)/2 = 92.75 — **Δ −0.05 t/s, essentially zero**.  But cherry-picking warm-up cycles is not a sound statistical practice; the honest reading is "PARALLEL=1 is statistically indistinguishable from PARALLEL=0 within measurement noise (Δ ≈ σ)".

## Comparison to Phase B v1 (commit 7468cdaf, pre-iteration-2)

Phase B v1 at HEAD `7468cdaf` reported tg100 PARALLEL=1 = 94.56 ± 0.69, vs PARALLEL=0 = 95.68 ± 0.51 = **Δ −1.12 t/s (−1.17%)** — the original regression that triggered operator's "fix perf in Phase B" requirement.

Phase B v2 at HEAD `ce19198d` (after FIX-4 threshold) shows tg100 PARALLEL=1 = 95.58 ± 0.62 vs PARALLEL=0 = 95.52 ± 1.11 = **Δ +0.06 t/s** — within noise.  Net improvement: **+1.18 t/s closure of the tg100 gap**.

The tg2000 regime (where parallel actually engages) is the new perf surface.  At v1 there was no tg2000 measurement (not in the spec); at v2 we measure ~Δ −0.5 t/s, borderline outside spec gate but inside noise.

## Recommendation to Queen Phase 3

**Phase B v2 (HEAD ce19198d) is structurally complete and approves merge consideration:**

- Correctness gates ALL PASS (sourdough 6/6 byte-identical both arms, coherence_smoke 2/2, build clean)
- INV-3, INV-4, INV-5, INV-6, INV-7 all satisfied
- tg100 perf gate PASS (Δ +0.06 t/s vs PARALLEL=0; within spec ±0.3)
- tg2000 perf gate MARGINAL (Δ −0.50 t/s vs PARALLEL=0; ~σ; outside spec ±0.3 letter but inside noise)
- Default-OFF means production tg100 + tg2000 are unaffected

**Operator's "fix perf in Phase B" requirement at tg100 is met.**  The tg2000 marginal regression is a known consequence of Phase B v1's commit-order serialization (R-B7); Phase C must add `GraphSession::enqueue()` to mlx-native to overlap commits, which is multi-day work and out of scope here.

**Operator decision needed**: accept tg2000 −0.50 t/s as within-noise non-regression (honest reading: Δ ≈ σ), OR require further tuning before merge.  Default recommendation: ACCEPT and merge.  Phase C will tune.

## Raw log
`/tmp/cfa-adr031-phaseB/b8-bench-v2.log`
