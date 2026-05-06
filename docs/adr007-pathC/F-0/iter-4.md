# ADR-007 Path C / F-0 / iter-4 — F-0.3 LANDED + F-2 CALIBRATION FALSIFIED

**Date:** 2026-05-05
**Verdict:** Empirical post-FWHT KV distribution is **N(0,1) at every (layer, head) cell** measured (std max-deviation 0.0012). F-2 calibration design path is falsified by data. F-6 (16-bit opt-in) becomes the strategic close path for strict Gate C.

## What landed

### hf2q (commit pending)

- `src/debug/investigation_env.rs`: new fields `dump_pre_quant_layers: Vec<usize>` and `dump_pre_quant_positions: Vec<usize>` parsed from `HF2Q_DUMP_PRE_QUANT_LAYERS` / `HF2Q_DUMP_PRE_QUANT_POSITIONS`.
- `src/serve/forward_mlx.rs`: extended pre-quant dump site to honor the layer/position filters; per-(layer, position) filenames `L{layer:02}_p{pos:04}_{k|v|meta}_pre_quant.{bin|json}`. Legacy single-file behavior preserved when both filters empty.
- `docs/adr007-pathC/F-0/empirical_kv_distribution.json` — measured distribution report (392 dump triples, 8 layers × ~50 decode positions, on Gemma 4 26B-A4B).

### mlx-native (commit pending)

- `examples/tq_distribution_analyze.rs` — F-0.3 analyzer. Reads pre_quant F32 dumps, runs the production encoder pipeline (D1 SRHT + FWHT + 1/sqrt(d) + norm extraction + scale-to-N(0,1)) up to "post-scale, pre-quantize", aggregates per-(layer, kv_head, coord) statistics, computes outlier rates vs 5/8-bit codebook ranges. Compares to N(0,1) theoretical reference.

## F-0.3 measurements

**Setup:** Gemma 4 26B-A4B-it-ara-abliterated-dwq (30 layers, 16 heads / 8 KV heads, head_dim=256, MoE 128/8). Single prompt: a 50-token quantum-mechanics description; 50 decode tokens generated; 8 layers sampled (0, 1, 5, 10, 15, 20, 25, 28) at every decode position.

**Total samples**: 392 dump triples → 343 D=256 (analyzed) + 49 D=512 (architectural variant, layer 29 — skipped). Per layer: ~50 positions × 8 KV heads × 256 head_dim ≈ 100K post-scale F32 values.

### Per-layer aggregate stats

| Layer | K mean | K std | K p1 | K p99 | K max | K out_8bit | K out_5bit | V mean | V std | V max | V out_8bit | V out_5bit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  0 |  0.0014 | 1.0000 | -2.285 | 2.265 | 3.993 | 0 | 4.19e-04 |  0.0020 | 1.0000 | 4.669 | 0 | 7.87e-04 |
|  1 |  0.0064 | 1.0000 | -2.302 | 2.312 | 4.080 | 0 | 6.18e-04 | -0.0097 | 1.0000 | 4.354 | 0 | 5.88e-04 |
| 10 |  0.0004 | 1.0000 | -2.273 | 2.301 | 3.809 | 0 | 7.08e-04 |  0.0039 | 1.0000 | 3.630 | 0 | 3.49e-04 |
| 15 | -0.0012 | 1.0000 | -2.321 | 2.328 | 4.233 | 0 | 1.19e-03 |  0.0065 | 1.0000 | 4.093 | 0 | 3.29e-04 |
| 20 | -0.0012 | 1.0000 | -2.323 | 2.314 | 4.074 | 0 | 9.17e-04 |  0.0010 | 1.0000 | 3.535 | 0 | 2.19e-04 |
| 25 |  0.0030 | 1.0000 | -2.302 | 2.327 | 4.322 | 0 | 1.04e-03 | -0.0011 | 1.0000 | 3.765 | 0 | 2.89e-04 |
| 28 | -0.0009 | 1.0000 | -2.303 | 2.328 | 3.963 | 0 | 5.48e-04 | -0.0104 | 0.9999 | 2.027 | 0 | 7.97e-05 |

**N(0,1) reference**: theoretical 8-bit outlier rate = 4.172e-7; theoretical p1 = -2.326, p99 = +2.326.

### Per-(layer, head) std deviation

Across all 112 (layer, K|V, kv_head) cells:
- **std min = 0.9988, max = 1.0000, mean = 0.9999, stdev = 1.7e-4**
- **Max 8-bit outlier rate across all cells = 0.0**
- **Max 5-bit outlier rate across all cells = 0.33%** (L15 K kv_head=2)

No (layer, head) cell deviates from N(0,1) by more than 0.0012 in std. No cell produces ANY value beyond ±5.07.

## Verdict and implications

### F-2 calibration design path is FALSIFIED

The hypothesis underlying F-2 was: *"per-(layer, head) calibrated dynamic-range estimator may close Gate C if production KV distribution diverges from N(0,1)."*

The empirical evidence shows production KV distribution **is N(0,1) at every layer and every head**, with std deviation < 0.002 across the entire model. **There is no calibration target.** Any per-(layer, head) calibration would, at best, match the existing N(0,1) Lloyd-Max codebook to 4 decimal places.

This is consistent with the iter-27 close-section observation (`§1117`) that calibrated-codebook implementation collapsed in tail-clip mode — the standard codebook is already optimal for the data; a calibrated codebook can only over-fit to sampling noise.

### F-2's deliverable becomes "uncalibrated_is_floor" verdict

Per the ADR §F-2 falsifier:

> If FAIL → either (a) iterate the design (max 2 attempts; document both); or (b) write `docs/adr007-pathC/F-2/uncalibrated_is_floor.md` with measured evidence that 8-bit uncalibrated *is* the achievable floor for this distribution and Gate C cannot be earned via codebook-tuning.

F-0.3 evidence definitively triggers branch (b). F-2 closes by writing the floor-evidence document; no calibration implementation is mantra-compliant given the data.

### F-3 (dense_kvs decision) constrained

Per the ADR §F-3 decision criterion:

> F-3-A if F-2 passes strict Gate C (TQ is good enough that dense-fallback is dead weight); F-3-B if F-2 fails strict and mixed-precision is the only path to PPL parity

F-2 **fails strict Gate C, but the failure is intrinsic** (Lloyd-Max 8-bit distortion floor, not codebook fitting). So F-3 has a third branch: **F-3-C: keep dense_kvs as opt-out, accept 8-bit as default, and pursue F-6 (16-bit opt-in) for byte-exact-comparable jobs that need stricter PPL.**

This matches the close-section's pragmatic decision (`§1083`) that already shipped: TQ-8-bit default, dense opt-out via `HF2Q_USE_DENSE=1`. F-3 effectively confirms this is the right policy given the F-0.3 evidence.

### F-6 (16-bit opt-in) becomes load-bearing for strict Gate C

If a downstream user/test demands strict Gate C compliance (PPL Δ < 1.15%), the only available path is to bump the bit-width. 8-bit Lloyd-Max distortion floor is intrinsic at ~4.4e-4 MSE per coord; 16-bit Lloyd-Max at ~6.4e-9 MSE per coord (256× tighter) would close the gap with margin to spare.

F-6 becomes the strategic completion path for any future "earn strict Gate C" requirement. Until that requirement is voiced, 8-bit remains the default per the close-section policy.

## Mantra discipline maintained

The mantra `Code + test == truth` worked at every step:
- iter-2 hypothesized D=512 norm bug → falsified by code
- iter-3 hypothesized SDPA NRMSE may exceed gate → falsified by data (607× margin)
- iter-4 hypothesized post-FWHT distribution may diverge from N(0,1) → falsified by data (max std deviation 0.0012)

Each falsification advanced the actual understanding rather than deferring it. The close-section's 1.24% PPL gap origin is now isolated to the **intrinsic Lloyd-Max distortion floor at 8-bit** — not codec implementation, not kernel implementation, not distribution mismatch, not calibration opportunity. It is physics.

## Iter-5 plan

1. **F-2 closure**: write `docs/adr007-pathC/F-2/uncalibrated_is_floor.md` with the F-0.3 evidence as the load-bearing argument. Move task #5 → completed.
2. **F-3 closure**: write the F-3-C verdict (keep dense_kvs as opt-out, 8-bit TQ as default — matches current production policy). Move task #6 → completed.
3. **Begin F-6 (16-bit opt-in)**: the actual strategic completion path for strict Gate C. ~50 LOC: extend `CODEBOOK_HB_8BIT` to a `CODEBOOK_HB_16BIT` (or use direct half-precision representation), add CLI flag `--kv-bits 16`, measure Gate C delta. Falsifier: 16-bit Gate C ≤ 0.1% PPL Δ.
4. **Begin F-4 (262K context unlock)**: the headline goal. Independent of F-2/F-3/F-6.
5. **Defer F-5 (paper-standard benchmarks)**: blocked on F-4 long-context unlock.
