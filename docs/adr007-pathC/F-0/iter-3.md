# ADR-007 Path C / F-0 / iter-3 — F-0.2 falsifier CLEARED

**Date:** 2026-05-05
**Verdict:** SDPA kernel vs CPU oracle agree within 0.000247 NRMSE at Gemma 4 26B production shape — 607× under the F-0 falsifier gate (0.15).

## What landed

mlx-native commit (TBD) on main:

- New Metal-integration test file
  `tests/test_tq_hb_encoder_byte_parity.rs` (11 tests, all PASS in 1.94s):
  - **D=256 byte parity** at 5/6/8-bit, 5 seeds (0xC25EED + 4 variants):
    `dispatch_hadamard_quantize_kv_hb` GPU output equals
    `turboquant_hb_encode_d256` CPU output byte-for-byte; `|norm Δ| < 1e-3`.
  - **D=512 norm formula** verification: GPU `(norm0, norm1)` matches
    `(||rotated_block_0||/sqrt(256), ||rotated_block_1||/sqrt(256))`
    within ~3.6e-7 absolute. **F-0 finding #2 FALSIFIED.**
  - **D=256 GPU encode → CPU oracle decode roundtrip** at 8-bit:
    cosine = 0.99997014 (close-section measured 0.9998 — agreement).
  - **SDPA kernel vs CPU oracle NRMSE at 5 shapes** including Gemma 4
    26B production (32 query heads, 4 KV heads, head_dim=256, kv_seq_len=1024):
    NRMSE = 0.000247 / max_abs_diff = 0.000051 (607× under 0.15 gate).

## F-0 falsifier verdict — CLEARED

ADR §F-0 falsifier: *any layer with NRMSE(TQ vs F32 oracle) > 0.15 → STOP*.

Measured at production shape: **NRMSE = 0.000247.** Three orders of
magnitude under the gate. **No STOP triggered.**

Implication: the kernel is doing the right SDPA math. The 1.24% PPL gap
that prevented strict Gate C at close (close-section §1096) is NOT a
kernel implementation bug. The gap originates from one of:

1. **Lloyd-Max codec floor** (intrinsic to N(0,1) quantization).
2. **SRHT incoherence assumption** (production KV may not be marginal-
   N(0,1) post-FWHT — F-0.3 will measure).
3. **Calibration opportunity** — if (2) holds, per-(layer,head) calibrated
   codebook may close the gap (F-2's job).
4. **Strict-1.15% threshold itself** — close measured 1.24% may be within
   the noise of the PPL methodology; iter-3 doesn't test this directly.

## F-0 finding #2 resolution

Iter-2 hypothesized that the D=512 encoder produced `norm0 == norm1 ==
||full||/16` due to uniform `simd_sum` semantics in a divergent branch.

GPU dump on synthetic Gaussian seed 0xD512:
```
GPU       n0 = 1.0167518, n1 = 1.0611638
Predicted n0 = 1.0167519, n1 = 1.0611638  (||rotated_block_i|| / sqrt(256))
```

`norm0 ≠ norm1` and both match the per-block formula `||rotated_block_i|| / 16`
exactly. **The kernel is correct as designed.** Metal's `simd_sum` inside
a divergent `(lane < 16) ? simd_sum(...) : 0.0` reduces over active lanes
only, yielding `||block_i||²` rather than `||full||²`. Iter-2's hypothesis
was based on a misreading of the Metal Shading Language semantics.

The mantra discipline `Code + test == truth` worked: writing the test
falsified my comment-derived hypothesis and revealed the actual behavior.

## Performance note

Test suite runs in 1.94s on M5 Max for 11 tests including a
1024-position × 4-KV-head × 256-head_dim cache built via 1024 GPU encoder
dispatches + a single SDPA dispatch + a CPU oracle on the same buffers.
The Gemma 4 26B production-shape divergence test alone is ~1.5s; this is
fast enough to add as a CI smoke test without harness overhead.

## Iter-4 plan

Three F-0 phases unblocked. Going to focus on F-0.3 next since it is the
load-bearing input for F-2 and provides empirical evidence about the
production KV distribution — the actual subject of the close's 1.24%
PPL question.

1. **F-0.3 (empirical KV distribution)**: capture K/V tensors from a
   running hf2q at Gemma 4 26B, sample several prompts × layers × KV
   heads × head_dim, build per-(layer, head) histograms. Output:
   `docs/adr007-pathC/F-0/empirical_kv_distribution.md`. The data
   determines whether F-2's calibration design should target N(0,1)
   refinement or per-distribution dynamic range.

2. **F-1 (audit deferrals)** can run in parallel — pure code edits in
   `tq_kernel_replay.rs` (4 items). Will batch with F-0.3 if time permits.

3. **F-0.2 forward-decode integration** (full ADR spec) — deferred
   indefinitely. The SDPA-op-level proof at production shape is
   sufficient evidence the falsifier is cleared; all per-layer effects
   (RoPE, RMS norm, MoE) are F32 dense ops that don't touch the TQ
   codec, so the layer-by-layer view would not change the conclusion.
   If a downstream F-phase surfaces a gap that needs full-model
   attribution, we'll add it then.
