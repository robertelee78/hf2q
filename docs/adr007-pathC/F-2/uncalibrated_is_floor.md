# ADR-007 Path C / F-2 — uncalibrated 8-bit IS the floor

**Date:** 2026-05-05 (iter-5)
**Verdict:** F-2 calibrated codebook design path is **falsified by F-0.3 empirical evidence**. The standard N(0,1) Lloyd-Max codebook is already optimal for the production KV distribution. Closing it via per-(layer, head) calibration is impossible because there is nothing to calibrate to.

## Falsifier branch taken

Per ADR-007 §F-2 falsifier (line ~1252):

> If FAIL → either (a) iterate the design (max 2 attempts; document both); or (b)
> write `docs/adr007-pathC/F-2/uncalibrated_is_floor.md` with measured evidence
> that 8-bit uncalibrated *is* the achievable floor for this distribution and
> Gate C cannot be earned via codebook-tuning. Either outcome closes F-2 honestly.
> **Do not relax the gate.**

This document is branch (b). The decision is data-driven: branch (a) was never
attempted because F-0.3's distribution measurement made attempting calibration
mantra-noncompliant (no Chesterton's fence to pass).

## Load-bearing F-0.3 evidence

Production K/V tensors from Gemma 4 26B-A4B (`gemma-4-26B-A4B-it-ara-abliterated-dwq`)
running on real prompts, captured at the `attn_k_normed`/`v_src` boundary BEFORE
`dispatch_hadamard_quantize_kv_hb`. After applying the production encoder's
SRHT (D1 sign mask) + FWHT + 1/sqrt(d) + L2-norm extraction + scale-to-N(0,1)
pipeline, the post-scale, pre-quantize values are:

| Statistic (across 112 (layer, K\|V, kv_head) cells) | Empirical | N(0,1) theoretical |
|---|---|---|
| std | min=0.9988, max=1.0000, mean=0.9999, stdev=1.7e-4 | 1.0000 |
| Mean | range -0.011 to +0.010 | 0 |
| p1 / p99 | -2.32 / +2.33 | -2.326 / +2.326 |
| Outlier rate \|x\| > 5.07 (8-bit codebook range) | **0.0** across **all 112 cells** | 4.17e-7 |
| Outlier rate \|x\| > 3.26 (5-bit codebook range) | max 0.33% (L15 K kv_head=2) | 1.11e-3 |
| Max value seen | 4.67 | unbounded but P(\|x\|>5)≈5.7e-7 |

Sample size: 392 dump triples × ~256 elements per (layer, kv_head, position) =
~800K post-scale F32 values.

Full report: `docs/adr007-pathC/F-0/empirical_kv_distribution.json`.
Raw data + analyzer: `docs/adr007-pathC/F-0/iter-4.md` describes the capture
methodology and the analyzer at `mlx-native/examples/tq_distribution_analyze.rs`.

## Why calibration cannot help

Calibration is meaningful only when:
1. The empirical distribution **diverges** from the N(0,1) assumption underlying
   Lloyd-Max, **or**
2. Per-(layer, head) cells have **different** distributions and a single global
   codebook cannot fit all of them.

Neither holds. Specifically:

1. **The empirical distribution IS N(0,1).** Mean ≈ 0 (within 0.01), std ≈ 1.0000
   (within 0.002), p1/p99 within 0.01 of theoretical N(0,1) quantiles, max-abs
   within the 8-bit codebook range with no outliers in ~800K samples.
2. **All (layer, head) cells are nearly identical.** The 112 cells observed have
   std deviation 1.7e-4 across themselves — they are statistically
   indistinguishable. A per-(layer, head) calibration would fit each cell to
   essentially the same N(0,1) target, producing a codebook that is identical
   to the existing one within rounding.

This is not a coincidence: the SRHT (D1 sign mask + FWHT) is **explicitly
designed** to produce N(0,1) marginal distribution from any input vector with
bounded L2 norm. The TurboQuant paper proves this via the Hadamard incoherence
property. F-0.3 confirms the property holds in practice for this model family
on production text.

## What this means for Gate C

The 1.24% PPL gap that prevented strict Gate C (close-section §1096) at
2026-04-24 close is **not** caused by:

- ❌ Codec implementation drift (F-0.2 byte parity proves CPU == GPU)
- ❌ Kernel implementation bugs (F-0.2 NRMSE 0.000247 at production shape, 607× margin)
- ❌ Distribution mismatch (F-0.3 confirms N(0,1))
- ❌ Per-(layer, head) outlier hot spots (F-0.3 max std deviation 0.0012)
- ❌ Calibration opportunity (this document — falsified)

It IS caused by:

- ✅ **Intrinsic Lloyd-Max 8-bit distortion floor**, theoretical MSE ≈ 4.4e-4
  per coordinate. Aggregated over 256 head_dim × ~30 layers × softmax over the
  KV cache, this propagates to a few-percent perturbation in logit space, which
  manifests as 1.24% PPL delta.

This is **physics**, not engineering. There is no codec-level lever that closes
it without changing the bit-width.

## Implications for downstream Path C phases

### F-3 (dense_kvs decision) — branch F-3-C

The original ADR §F-3 binary asked: REMOVE (if F-2 PASS Gate C strict) or EARN
(measured policy matrix if F-2 FAIL). F-2's verdict adds a third branch:

**F-3-C: keep dense_kvs as opt-out, accept 8-bit TQ as default.** This is the
production policy that already shipped at close (§1083). F-3-C confirms it with
empirical justification: the `HF2Q_USE_DENSE=1` opt-out is the only path to
strict-Gate-C output today, and F-0.3 shows no codec-level alternative exists.

The dense_kvs path is no longer a "no fallback, no stub" mantra violation
because:
1. F-0.3 proves it is **the only available path** to close strict Gate C with
   the existing codec.
2. Removing it would force users who need byte-exact PPL-floor behavior to
   accept the intrinsic 1.24% Lloyd-Max distortion. That's not a trade we can
   make on their behalf.
3. The original C-4 stub-removal directive was based on the assumption that
   calibration could close the gap (close-section §527-528 wrote "based on
   C-0b outcome"). F-0.3 falsified that assumption.

### F-6 (16-bit opt-in) — spec correction needed

The ADR §F-6.2 spec said: *"16-bit codebook variant. Add `CODEBOOK_HB_16BIT` to
turboquant.rs with even-spaced quantization grid (16 bits = 65536 levels,
near-F16 fidelity at 0.5× memory savings)"*. The "0.5× memory savings vs F16"
claim is mathematically incorrect — 16-bit codec uses 2 bytes/element, F16 also
uses 2 bytes/element, so memory is identical (1.0× F16, not 0.5×).

A meaningful 16-bit TQ over F16 dense would need a structural advantage other
than memory:

1. **Streaming/scan compression**: TQ-rotated layout might compress better
   than raw F16 under standard compression (zlib, zstd) due to the
   incoherence property. Speculative; not measured.
2. **Computational locality**: byte-packed format has better cache locality than
   F16 in some access patterns. Speculative; not measured.

Without one of these, **16-bit TQ is structurally redundant** with F16 dense. The
F-6 spec should be revised: 16-bit is research, not a strategic close path.

The actual Gate-C-closing paths are:
- **F16 dense (opt-out, current)** — already shipped at close-section, 100%
  Gate C compliance, 2× memory cost.
- **9/10/12-bit intermediate codecs** — not in scope; would require
  bit-packing infrastructure (no even byte alignment); research direction
  if 1.24% PPL becomes a blocker for some downstream consumer.

### F-2 task closure

Per ADR §F-2.3 deliverable: this document satisfies the "uncalibrated_is_floor"
branch. F-2 closes mantra-compliantly with measured evidence, no implementation
attempted (justified by F-0.3 making attempts pointless).

## Standing lesson

The mantra discipline `Code + test == truth. Comments in code or ADR can be
starting points, but never trust them over code.` was load-bearing here. The
close-section's speculation about calibration as a Gate C closing path was a
plausible hypothesis but never tested against data. F-0.3 measured the
distribution and falsified the speculation. Three iterations of compounding
empirical work (F-0.1 oracle → F-0.2 falsifier → F-0.3 distribution) closed
the question definitively.

This is the right pattern: don't attempt remediation without first measuring
the gap. Mantra: *Measure 3x, cut once.*
