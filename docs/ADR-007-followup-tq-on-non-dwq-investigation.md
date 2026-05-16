# ADR-007 follow-up: TQ KV codec behavior on non-DWQ models — investigation findings

**Status:** Investigation complete with measured data. Initial fix
hypothesis was wrong. Updated root-cause understanding below.

**Date:** 2026-05-16
**Authors:** Robert (operator) + claude-flow (investigation)
**Related:** ADR-007 (TurboQuant KV cache), commits `a035a2aa` (per-layer
cosine diag), this session's `HF2Q_DEBUG_TQ_RMS` extended probe.

---

## Problem statement

`scripts/release-check.sh` Gate H (TQ-active quality envelope) fails on
`gemma4-ara-2pass-APEX-Q5_K_M.gguf` with:

| Metric | Floor (from DWQ-captured fixture) | APEX measurement |
|---|---|---|
| `cosine_mean` (TQ vs dense SDPA output) | ≥ 0.999 | **0.865** |
| `cosine_p1`   | ≥ 0.990 | **0.628** |
| `argmax_flip_rate` | ≤ 0.015 | **0.148** (14.8%) |
| `ppl_delta_pct` | ≤ 0.020 | **0.673** (67%) |

Operator framing: TQ is a KV-cache codec — should be model-agnostic.

## Investigation chain

### Step 1 — per-layer cosine attribution (commit `a035a2aa`)

Added per-layer cosine_mean output to `synthesize_cosine` in
`src/serve/parity_quality.rs`. Re-ran Gate H capture on sourdough
(1000 tokens, 30 layers). Result:

```
ALL 30 layers in 0.82-0.93 range.  None reaches 0.99.
Layer 0 highest (0.930, shallowest).  Mid-late clustered 0.82-0.85.
Min values per layer reach as far as -0.45 (catastrophic single pairs).
```

**Conclusion: uniform degradation, not a layer-specific bug.**

### Step 2 — original hypothesis: heavier-tail distribution outside codebook range

Read TQ codec sources:
- `/opt/mlx-native/src/shaders/hadamard_quantize_kv_fast.metal`
- `/opt/mlx-native/src/shaders/flash_attn_vec_tq_hb.metal`

Codec normalizes post-FWHT, scales by `sqrt(d) / norm` so output is
"approximately N(0,1)", then quantizes to Lloyd-Max-for-N(0,1)
codebook (256 centroids spanning ±5.07σ for 8-bit).

Initial hypothesis: APEX activations have heavier tails than DWQ
(K-quant doesn't shape weights to produce Gaussian activations like
DWQ training does); outliers get clipped at ±5σ codebook boundary;
loss of magnitude information → cosine drops.

### Step 3 — measured the actual post-scale distribution (this session)

Extended the existing `HF2Q_DEBUG_TQ_RMS` probe (in
`src/serve/forward_mlx.rs:3897-3903`) to also emit `max_abs`, 50/90/99
percentiles, **excess kurtosis**, and **codebook-clip count**.

Ran on APEX-Q5_K_M with `HF2Q_DEBUG_TQ_RMS=1 hf2q generate --prompt
"What is 2+2?"`. All 30 layers report:

```
rms ≈ 1.000        (perfect — codec scaling does its job)
max_abs ≈ 2.4-3.7  (well within ±5.07 codebook range)
p99_abs ≈ 2.3-3.0  (matches Gaussian where p99|x| ≈ 2.33σ)
p50_abs ≈ 0.60-0.73 (matches Gaussian where p50|x| ≈ 0.674σ)
excess_kurt ∈ [-0.55, +0.41]  (essentially Gaussian; +3 for Laplace)
clipped = 0/256    (zero outlier clipping)
```

**Conclusion: distribution IS Gaussian-N(0,1). Initial hypothesis
REFUTED. There is NO heavy-tail / no outlier clipping.**

### Step 4 — alternative hypothesis: code regression since fixture capture

The DWQ fixture (`tests/evals/reference/sourdough_tq_quality.json`)
was captured at git HEAD `4fbeb05` (2026-04-26). Between then and now,
~50 commits of ADR-029 perf work landed. Maybe the TQ codec output
drifted in a way that makes APEX worse.

To test: cherry-pick the `from_gguf` migration onto pre-perf commit
`7ffa01de` (right before ADR-029 Step 1c) and re-run Gate H capture
on APEX at that pre-perf code state.

Result:

```
Pre-perf code, APEX-Q5_K_M:
  cosine_mean=0.865037  cosine_p1=0.628370  argmax_div=0.1480  ppl_delta=0.6734

HEAD code, APEX-Q5_K_M (current session):
  cosine_mean=0.865037  cosine_p1=0.628370  argmax_div=0.1480  ppl_delta=0.6734
```

**Bit-for-bit identical.** Conclusion: NOT a code regression. The TQ
codec produces the same result before and after all ADR-029 perf work.
**The 0.865 cosine is a stable property of TQ-on-APEX-Q5_K_M, not a
regression.**

## Corrected understanding

Both the codec and the code are working correctly. The cosine
difference between DWQ (0.9996) and APEX (0.865) reflects a real
difference in how the two models interact with TQ.

What's actually going on:

1. **TQ codec input distribution is Gaussian on both models** (measured
   for APEX; assumed-similar for DWQ since the codec scaling produces
   RMS=1 by construction for any moderate distribution).
2. **Per-element K/V reconstruction error is tiny on both models** —
   bounded by Lloyd-Max codebook spacing, ≈ 0.04σ per element for
   8-bit on N(0,1). RMS reconstruction error is ≈ 0.02σ.
3. **SDPA amplification differs by model.** Attention is `softmax(Q·Kᵀ
   / sqrt(d)) · V`. If two K vectors give near-tied attention scores,
   a small reconstruction perturbation can flip which one wins → large
   shift in attention weight → large shift in `softmax · V` output.
4. **DWQ training shapes the model to be quantization-robust**, meaning
   attention scores are NOT near-tied at quantization-noise scale, so
   TQ noise stays within softmax's flat zone. APEX K-quant has no
   such training pressure; attention scores have more near-ties; TQ
   noise flips them; output cosine drops.

This is consistent with the per-layer pattern:
- Layer 0 (least context, simplest attention pattern) least affected
- Mid-late layers (composition of many softmaxes, more accumulated
  near-tie sensitivities) most affected
- Some individual (layer, position) pairs catastrophically affected
  (cosine -0.45) — these are tokens where attention is bimodal and TQ
  noise tipped the bimodality to the other mode

## Why this is NOT fixable at the codec level

A per-block adaptive scaling change (my original ADR proposal) would
NOT help because:

- The codec input is already Gaussian, so per-block adaptive scaling
  would produce ≈ identical results (the data fits the assumption
  already).
- Even a hypothetically perfect lossless codec (cosine ≈ 1.0 on K/V
  reconstruction) would still have non-trivial SDPA-output divergence
  on APEX because the issue is attention-score near-ties, not
  reconstruction magnitude. The next bit of quantization noise (or
  even the next FLOP of FP arithmetic) tips the same balance.

The proper "fix" lives outside the codec:

- **Use DWQ-trained models for TQ-enabled deployments** (the design
  intent; cosine 0.9996 on DWQ proves the codec is doing its job).
- **For non-DWQ models, recommend `HF2Q_USE_DENSE=1`** to bypass TQ
  and accept higher KV memory cost in exchange for byte-exact
  attention.
- (Possible engineering direction) Investigate whether different
  attention-time strategies — e.g., compute attention scores at
  higher precision than the K vectors, or detect near-ties at runtime
  and fall back to dense for those tokens — could reduce SDPA
  amplification without retraining the model. This is a research
  direction, not a known-good fix.

## What this session produced

1. **commit `a035a2aa`** — per-layer cosine_mean diagnostic in
   `synthesize_cosine`. Useful for any future Gate H investigation.
2. **forward_mlx.rs extended `HF2Q_DEBUG_TQ_RMS` probe** (uncommitted
   at the time of this draft) — emits max_abs, percentiles, excess
   kurtosis, codebook-clip count per (layer, block) per decode step.
   Confirms the distribution assumption is met on APEX.
3. **Bisect-style experiment** — pre-perf vs HEAD bit-identical
   confirms not a code regression.
4. **This ADR** — corrected understanding of why Gate H fails on APEX.

## Implications for the v0.1.0 release

- Gate H failing on APEX is **expected and load-bearing** — it
  signals "this model wasn't trained to be TQ-robust." Threshold
  relaxation would mask a real model-quality signal.
- The shipping recommendation for v0.1 should be: **for production
  use with TQ enabled, use DWQ-trained models.** For other models,
  set `HF2Q_USE_DENSE=1` if attention parity with dense matters.
- The DWQ-paired TQ default is not a bug — it's the design.
- Gate H thresholds in `scripts/release-check.sh` stay at DWQ levels
  (0.999/0.99/0.015/0.02). The fixture in `tests/evals/reference/
  sourdough_tq_quality.json` stays the DWQ-captured one. Operators
  running release-check on non-DWQ models see Gate H fail loud,
  prompting them to choose: use DWQ, or accept the model isn't a
  TQ-paired production candidate.

## Open questions left for a future mission

- Does the softmax-amplification hypothesis hold quantitatively?
  Direct measurement: compute cosine(dense_K, tq_K) and
  cosine(dense_V, tq_V) at each layer. If K/V reconstruction is
  ≈ 0.999 but SDPA output is 0.865, the gap is exactly
  softmax amplification and the hypothesis is confirmed.
- Could a runtime "attention-score near-tie detector" gracefully
  fall back to dense for the affected tokens, retaining most of TQ's
  memory savings while preserving quality? Open research direction.
- Cross-model verification: does Qwen3.6 APEX show similar Gate H
  numbers? Need the Qwen-arch parity gate extension (task #21) to
  test. If Qwen APEX also degrades to 0.85-ish cosine, that confirms
  the issue is "non-DWQ models", not "this particular gemma APEX
  build."

## Honest note on the original ADR draft

The first draft of this ADR proposed "per-block adaptive scaling" as
the fix. The post-scale distribution measurement (Step 3 above) refuted
that hypothesis — the distribution is already Gaussian, so adaptive
scaling has nothing to fix. **The original fix proposal would have
been wasted engineering.** Measurement saved the project from
implementing the wrong fix. (Mantra: "measure 3× cut once" — this is
exactly why.)
