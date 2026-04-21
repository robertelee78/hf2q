# TQ C-0 Per-Head Heatmap — sdpa_out

Session: cfa-20260421-C0-audit | Worker 2 analyst-differ

## Drilldown Trigger

Classification is bug-candidate with ALL 4 ops breaching threshold.
Drilldown fires per queen spec: multiple ops breach rule.

## KEY FINDING: Layer 0, Pos 1 — Byte-Identical Inputs, Divergent SDPA Output

At layer=0, pos=1 (first decode step), Q/K/V dumps are **byte-identical** between dense and TQ.
sdpa_out max_abs_diff = 8.441916e-01. The bug is INSIDE the TQ-SDPA kernel, NOT in KV representation.

### Per-Head Analysis at layer=0, pos=1 (sdpa_out)

| Head | max_abs_diff | mean_abs_diff | Outlier (>5x mean)? |
|------|-------------|---------------|---------------------|
|  0 | 3.251911e-01 | 9.111434e-02 | no |
|  1 | 1.796662e-01 | 5.951498e-02 | no |
|  2 | 3.149224e-01 | 7.416060e-02 | no |
|  3 | 2.041563e-01 | 4.979555e-02 | no |
|  4 | 1.616058e-01 | 4.395083e-02 | no |
|  5 | 1.940693e-01 | 5.336425e-02 | no |
|  6 | 1.202304e-01 | 3.237043e-02 | no |
|  7 | 2.373320e-01 | 5.751056e-02 | no |
|  8 | 1.531802e-01 | 4.003736e-02 | no |
|  9 | 1.462643e-01 | 4.135962e-02 | no |
| 10 | 1.822425e-01 | 4.477265e-02 | no |
| 11 | 2.789050e-01 | 6.213696e-02 | no |
| 12 | 8.441916e-01 | 5.359153e-02 | no |
| 13 | 8.350761e-01 | 8.413579e-02 | no |
| 14 | 3.366652e-01 | 7.767236e-02 | no |
| 15 | 5.669622e-01 | 6.203232e-02 | no |

Mean of per-head max_abs_diff: 3.175413e-01  |  5x threshold: 1.587707e+00

**No head is a 5x outlier** — error is broadly distributed across all 16 heads (floor-like within-layer pattern),
but heads 12, 13, 15 carry the largest spikes (0.844, 0.835, 0.567).

### Per-Dimension Pattern (Worst 3 Heads, layer=0 pos=1)

| Head | max_diff | dims>0.1 | dims>0.5 | dims>0.8 | Pattern |
|------|----------|----------|----------|----------|---------|
| 12 | 8.441916e-01 | 29/256 | 1/256 | 1/256 | peaked (1-2 extreme dims) |
| 13 | 8.350761e-01 | 71/256 | 1/256 | 1/256 | peaked (1-2 extreme dims) |
| 15 | 5.669622e-01 | 47/256 | 1/256 | 0/256 | spread |

### Interpretation

The per-head error distribution is NOT uniform noise:

- All 16 heads show non-zero error (systemic, not a missing-head bug)
- Heads 12 and 13 each have 1 dimension exceeding 0.5 (and 0.8) — extreme single-dim spikes
- The spike-per-dim pattern in heads 12/13 points to a specific numerical issue in the
  TQ SDPA softmax or attention-weight accumulation at those head positions.
- This is NOT quantization floor noise (floor noise would be flat, small, and consistent).
- Verdict: **concentrated bug** — TQ-SDPA kernel produces wrong output with identical Q/K/V inputs.

## Summary Across All Layers at pos=5

| Layer | max head_max | min head_max | std head_max | any_head_>5x_mean |
|-------|-------------|-------------|-------------|-------------------|
|  0 | 7.7335e-01 | 1.3595e-01 | 1.7930e-01 | no |
|  1 | 6.9104e-01 | 1.1871e-01 | 2.0308e-01 | no |
|  2 | 1.5795e+00 | 1.2287e-01 | 4.1051e-01 | no |
|  3 | 1.7435e+00 | 1.2126e-01 | 4.3528e-01 | no |
|  4 | 2.8556e+00 | 1.3014e-01 | 6.7206e-01 | no |
|  5 | 9.4898e-01 | 8.3291e-02 | 2.0695e-01 | no |
|  6 | 2.1287e+00 | 1.4036e-01 | 5.6618e-01 | no |
|  7 | 2.3453e+00 | 1.4943e-01 | 6.0332e-01 | no |
|  8 | 3.2632e+00 | 1.3810e-01 | 8.6181e-01 | no |
|  9 | 1.9120e+00 | 1.8777e-01 | 5.4884e-01 | no |
| 10 | 3.5675e+00 | 1.5706e-01 | 1.0759e+00 | no |
| 11 | 1.1473e+00 | 9.3297e-02 | 2.6518e-01 | no |
| 12 | 3.0667e+00 | 1.2368e-01 | 9.0616e-01 | no |
| 13 | 3.2378e+00 | 1.3983e-01 | 7.3543e-01 | no |
| 14 | 2.8613e+00 | 1.3360e-01 | 8.3873e-01 | no |
| 15 | 2.2903e+00 | 1.3808e-01 | 7.9364e-01 | no |
| 16 | 3.0131e+00 | 9.7865e-02 | 8.2394e-01 | no |
| 17 | 1.7502e+00 | 1.2242e-01 | 4.3418e-01 | no |
| 18 | 3.1732e+00 | 1.3616e-01 | 7.1864e-01 | no |
| 19 | 1.3090e+00 | 2.0430e-01 | 3.1809e-01 | no |
| 20 | 2.6768e+00 | 1.4413e-01 | 6.6545e-01 | no |
| 21 | 1.6419e+00 | 1.0169e-01 | 4.0512e-01 | no |
| 22 | 4.0965e+00 | 1.1470e-01 | 9.7232e-01 | no |
| 23 | 1.3729e+00 | 1.2592e-01 | 3.2267e-01 | no |
| 24 | 2.2656e+00 | 1.4969e-01 | 7.2194e-01 | no |
| 25 | 3.7949e+00 | 1.4998e-01 | 1.1782e+00 | no |
| 26 | 2.3545e+00 | 3.2983e-01 | 5.8617e-01 | no |
| 27 | 3.7738e+00 | 1.8911e-01 | 9.1845e-01 | no |
| 28 | 2.0729e+00 | 1.6721e-01 | 5.6183e-01 | no |
| 29 | 9.5693e-01 | 1.0827e-01 | 1.9410e-01 | no |

## Drilldown Conclusion

- Drilldown triggered: YES (all 4 ops breach; multiple-op rule)
- Outlier heads (>5x mean): **NONE at layer 0**
- Error type: **systemic kernel error** — all heads affected, with concentration in heads 12/13
- The byte-identical Q/K/V + divergent sdpa_out at layer 0 pos 1 PROVES the bug is in the TQ-SDPA kernel itself
- Q/K/V diverge at layers 1+ only because layer 0 sdpa_out error propagates through the residual stream
- This is NOT quantization noise — it is a functional error in the TQ attention computation
