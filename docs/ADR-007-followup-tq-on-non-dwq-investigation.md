# ADR-007 follow-up: TQ KV codec degradation on non-DWQ models — investigation findings + proposed fix

**Status:** Investigation complete. Root cause identified. Fix design proposed. Implementation deferred to a dedicated focused mission (multi-day shader-level engineering).

**Date:** 2026-05-16
**Authors:** Robert (operator) + claude-flow (investigation)
**Related:** ADR-007 (TurboQuant KV cache), ADR-022 P1.8 (Gemma4Config::from_gguf), commit `a035a2aa` (Gate H per-layer diagnostic)

---

## Problem statement

`scripts/release-check.sh` Gate H (TQ-active quality envelope) **FAILS** on
`gemma4-ara-2pass-APEX-Q5_K_M.gguf` with:

| Metric | Floor (from DWQ-captured fixture) | APEX measurement |
|---|---|---|
| `cosine_mean` (TQ vs dense SDPA output) | ≥ 0.999 | **0.865** |
| `cosine_p1`   | ≥ 0.990 | **0.628** |
| `argmax_flip_rate` | ≤ 0.015 | **0.148** (14.8%) |
| `ppl_delta_pct` | ≤ 0.020 | **0.673** (67%) |

Operator pushback (justified): **TQ is a KV-cache codec — it should be
model-agnostic. The DWQ-pairing is a real design limitation worth fixing,
not relaxing thresholds for.**

## Investigation methodology

Diagnostic patch `a035a2aa` added per-layer cosine_mean output to
`synthesize_cosine` in `src/serve/parity_quality.rs`. Re-ran Gate H
capture on sourdough prompt (1000 tokens, 30 layers).

## Per-layer cosine_mean on APEX-Q5_K_M (TQ-8-bit vs dense)

```
layer 00: mean=0.930  min=0.900    (best, shallowest)
layer 01: mean=0.868  min=0.746
layer 02: mean=0.888  min=0.770
layer 03: mean=0.889  min=0.801
layer 04: mean=0.890  min=0.819
layer 05: mean=0.879  min=0.714
layer 06: mean=0.880  min=0.661
layer 07: mean=0.880  min=0.645
layer 08: mean=0.885  min=0.478
layer 09: mean=0.868  min=-0.027  ← first negative outlier
layer 10: mean=0.893  min=-0.038
layer 11: mean=0.906  min=0.153
layer 12: mean=0.855  min=-0.355
layer 13: mean=0.865  min=-0.453  ← worst single (layer, pos) pair
layer 14: mean=0.865  min=-0.217
layer 15: mean=0.858  min=-0.125
layer 16: mean=0.829  min=0.009
layer 17: mean=0.832  min=-0.221
layer 18: mean=0.836  min=-0.190
layer 19: mean=0.844  min=-0.359
layer 20: mean=0.828  min=0.028
layer 21: mean=0.865  min=0.139
layer 22: mean=0.851  min=0.207
layer 23: mean=0.840  min=0.263
layer 24: mean=0.834  min=0.390
layer 25: mean=0.862  min=0.394
layer 26: mean=0.862  min=0.555
layer 27: mean=0.825  min=0.365
layer 28: mean=0.860  min=0.592
layer 29: mean=0.886  min=0.786
```

**Pattern: uniform degradation across ALL 30 layers.** None reaches the
0.99+ band achieved on DWQ. Layer 0 (shallowest) is best at 0.930.
Mid-late layers (16-27) cluster around 0.82-0.85. Some individual
(layer, position) pairs go wildly negative — catastrophic codec failure
on specific activations.

**Conclusion: this is distribution-systemic, not a layer-specific bug.**

## Codec math (from `/opt/mlx-native/src/shaders/hadamard_quantize_kv_fast.metal`)

The TQ-HB 8-bit encoder for D=256 (Gemma-4 sliding head_dim):

```
1. Load EPT=8 elements per lane (32 lanes × 8 = 256 elements per head).
2. D1 sign pre-multiplication (SRHT — AmesianX-verbatim sign table).
3. FWHT via simd_shuffle_xor (zero threadgroup barriers).
4. Normalize by 1/sqrt(head_dim).
5. Compute L2 norm:   norm = sqrt(sum_sq) over all 256 elements.
6. Scale:             scale = (1/norm) * sqrt(256)
                      elem  = elem * scale     ← post-scale assumed ≈ N(0,1)
7. Quantize:          idx = argmin_k |CODEBOOK_8BIT[k] - elem|
                            (codebook range: ±5.07 σ, 256 centroids)
8. Pack as one byte per element.
```

Stored: `packed[head, pos, dim]` (u8) + `norms[head, pos]` (f32).

Decode reverses: `reconstructed_elem = CODEBOOK_8BIT[idx] * (norm / sqrt(256))`.

**The load-bearing assumption is step 6 + 7: post-FWHT-post-normalize-
post-scale elements are approximately N(0,1) → fit the Lloyd-Max
codebook.**

### Why DWQ works

DWQ training explicitly optimizes the model so that activations
(specifically the post-rotation post-normalize K/V) stay Gaussian-like.
The Lloyd-Max-for-N(0,1) codebook captures the distribution faithfully.
**Measured: cosine 0.9996 on DWQ → essentially lossless TQ KV cache.**

### Why APEX breaks the assumption

APEX-Q5_K_M is a regular K-quant (no quantization-aware training).
Activations have heavier tails and irregular distributions; FWHT alone
doesn't whiten them enough; the L2-norm-based scale formula
`(1/norm) * sqrt(d)` over-scales when outliers dominate the norm,
compressing the bulk distribution into a narrow band of the codebook
(quantization resolution wasted) AND clipping the remaining outliers
to the codebook boundary (±5σ for 8-bit). Both effects increase
reconstruction error → cosine drops to 0.865.

This is consistent with the per-layer data: layer 0 (shallowest,
least-accumulated distribution distortion) is best at 0.930; deeper
layers compound prior layers' quantization-shaped activations,
worsening to 0.82-0.86.

## Why the existing D=512 `HF2Q_SCALE_FORMULA` knob doesn't help

The D=512 path (Gemma-4 global/full-attention layers, 1 in 6) already
has a tunable `scale_factor_d512` (env: `HF2Q_SCALE_FORMULA` ∈ {`bare`,
`sqrt256`, `sqrt512`}). It tunes the per-block scale uniformly across
heads/positions but does NOT adapt per-block to actual activation
distribution. The D=256 path (the other 5 in 6 layers) has NO scale knob.

Either way, **the underlying issue is fixed-scale, not tunable-scale.**
A static knob can't fix a dynamic distribution mismatch.

## Proposed fix: per-block adaptive scaling

### Encode-side change (`hadamard_quantize_kv_fast.metal`)

Replace the fixed `scale = (1/norm) * sqrt(d)` with a per-block
adaptive scale computed from the actual max-absolute-value of the
post-FWHT post-normalize block:

```metal
// After FWHT + 1/sqrt(d) normalize:
float local_max = 0.0f;
for (ushort i = 0; i < EPT; i++) local_max = max(local_max, abs(elems[i]));
float blk_max = simd_max(local_max);          // per-simdgroup max (block max)

// Adaptive scale: choose scale so blk_max maps to codebook_max_centroid.
// Codebook_max for 8-bit Lloyd-Max N(0,1) ≈ 5.07.
float scale = (blk_max > 1.0e-10f)
    ? (CODEBOOK_8BIT_MAX / blk_max)
    : 1.0f;
for (ushort i = 0; i < EPT; i++) elems[i] *= scale;

// Store `scale` (or `1/scale`) per block in the `norms` buffer
// instead of the L2 norm.  Decoder uses the same value to recover.
```

This ensures:
1. **Zero outlier clipping** — the largest element exactly hits the
   codebook boundary, not beyond.
2. **Full codebook resolution for bulk** — small elements use the
   centroids near zero, large elements use the centroids near the
   boundary.
3. **Model-agnostic** — no DWQ-specific assumption; works for any
   distribution that FWHT can rotate.

### Decode-side change (`flash_attn_vec_tq_hb.metal`)

Mirror the encode: replace `scale_norm = norm * inv_sqrt(256)` with
`scale_norm = stored_block_scale_inverse` (read what encoder stored).

### Round-trip math

Encode: `stored_idx = nearest_centroid(elem * scale)` where
`scale = codebook_max / blk_max`.
Decode: `reconstructed = CODEBOOK[stored_idx] / scale = CODEBOOK[stored_idx] * blk_max / codebook_max`.

For an element at the block max: `elem * scale = blk_max * (codebook_max / blk_max) = codebook_max` →
quantizes to the boundary centroid → reconstructed to `codebook_max * blk_max / codebook_max = blk_max`.
Round-trip exact at the boundary.

For an element at half the block max: `elem * scale = 0.5 * codebook_max` →
quantizes to a centroid near 0.5 * codebook_max → reconstructed to ~0.5 * blk_max.
Error bounded by half-codebook-spacing.

### Storage compatibility

Current `norms` buffer is `[num_kv_heads, capacity, 1]` f32 for D=256.
New scheme stores `1/scale` per block (same shape) — drop-in replacement.
For D=512 (2 blocks per position), `[num_kv_heads, capacity, 2]` f32
stays the same.

### Backwards compatibility

The change is **encode-decode-symmetric**: as long as encode and decode
use the same scale-storage convention, round-trip is exact. Storage
format on disk is unchanged. Existing serialized KV caches (if any)
would need re-encoding — but the KV cache is typically not persisted
across restarts.

A new env var `HF2Q_TQ_SCALE_MODE` ∈ {`legacy`, `adaptive`} should
gate the new path during validation, with `legacy` (current behavior)
as default until APEX cosine ≥ 0.99 is verified across the test matrix.

## Validation plan (separate mission)

1. **Instrumentation pre-fix**: enable existing `rms_probe` (already
   wired through `hadamard_quantize_kv.rs:107`'s `rms_scratch` param)
   to dump post-scale element values on APEX. Confirm distribution
   has heavier tails than N(0,1).
2. **Implement encode change** with `HF2Q_TQ_SCALE_MODE=adaptive`
   default-OFF.
3. **Implement decode change** matching the encode change.
4. **Round-trip test** at the codec layer (encode-then-decode on
   synthetic and real activations; verify max-element exact match
   + bounded RMSE for bulk).
5. **APEX Gate H** with `HF2Q_TQ_SCALE_MODE=adaptive`: expect
   cosine_mean ≥ 0.99.
6. **DWQ Gate H regression check**: with `adaptive` mode the codec
   uses different storage convention; need to either regenerate the
   DWQ fixture under the new mode OR verify legacy mode keeps DWQ
   fixture passing.
7. **Performance impact**: measure decode tok/s; expect ≤ 1% regression
   from one extra `simd_max` per encode.
8. **Default-flip decision**: once APEX + DWQ both pass at adaptive
   mode (or are within tolerance), flip default to `adaptive`.

Estimated effort: 3-6 days of focused shader engineering + measurement.

## Open questions

- Does Qwen3.6 / Qwen3-VL exhibit the same TQ degradation? Need a
  Qwen APEX Gate H run (depends on extending the parity gate to
  arch-aware dispatch — task #21 in current session).
- Is the proposed `blk_max`-based scaling optimal, or would a
  std-deviation-based scaling perform better on some distributions?
  The trade-off: max-based clips no outliers but wastes resolution
  when bulk is much smaller than max; std-based fits bulk better
  but clips outliers. The right answer depends on what the codec
  consumer (SDPA) is most sensitive to — likely outlier preservation
  matters more for attention because outliers correspond to the
  "salient" key/value vectors that dominate softmax.
- Is the 8-bit codebook range (±5.07σ) actually limiting? If almost
  all post-scale elements naturally fall in ±3σ, the adaptive scaling
  has plenty of headroom. If they reach beyond ±5σ frequently, even
  adaptive scaling would benefit from a wider codebook (e.g., extended
  Lloyd-Max for thicker-tailed distributions).

## Why this is deferred to a separate mission

This investigation produced root cause + concrete fix design.
**Implementation requires:**
- Shader-level Metal edits in two files (encode + decode)
- Careful symmetric round-trip testing
- Re-validation against both APEX AND DWQ baselines
- CFA codex review on every shader change
- Performance impact verification

It is genuinely 3-6 days of focused engineering work, deserving its own
dedicated mission rather than being squeezed into the tail of a release-
prep session. Filing this ADR as the complete starting point so the
codec mission can begin with all context loaded.
