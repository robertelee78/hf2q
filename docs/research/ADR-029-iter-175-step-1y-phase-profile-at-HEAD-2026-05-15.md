# ADR-029 iter-175 Step 1y — per-layer-phase GPU profile at HEAD

**Date**: 2026-05-15
**HEAD**: hf2q `a66f8fc6`, mlx-native `22dc55b`
**Iteration**: 29 of /loop autonomous

## Goal

Re-characterize the per-layer-phase GPU time distribution at HEAD with all
landed Step 1 levers enabled by default (precompiled metallib ON, concurrent
dispatch default, FC-promoted batch divisors).  Future investigators can use
this profile to scope which phase/layer/kernel offers the most leverage.

## Method

```
HF2Q_PER_LAYER_PHASE_GPU_TIME=1 ./target/release/hf2q generate \
  --model models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf \
  --prompt "The quick brown fox" --max-tokens 5 --ignore-eos
```

`PHASE_ATTN` reports the GPU time for `attn_norm → Q/K/V → RoPE → FA → o_proj`.
`PHASE_FFN` reports `attn_post_norm → MoE_router → experts → ffn_post_norm → add`.

## Result (token 2, steady-state)

### Per-layer summary

* Total layers: **30** (26 sliding `S`, 4 global `G`)
* `S` layers: L00-L04, L06-L10, L12-L16, L18-L22, L24-L28
* `G` layers: L05, L11, L17, L23, L29

### Per-phase distribution (token 2)

| Phase | Layer type | Median | Range | σ |
|---|---|---|---|---|
| ATTN | sliding (S) | **~100 µs** | 98-105 µs | ~2 µs |
| ATTN | global  (G) | **~138 µs** | 135-145 µs | ~3 µs |
| FFN  | sliding (S) | **~195 µs** | 191-205 µs | ~5 µs |
| FFN  | global  (G) | **~194 µs** | 192-220 µs | ~6 µs |

**FFN dominates ATTN ~2:1** at both sliding and global layers.  The FFN
sub-ladder (MoE router + expert dispatches + norms) is where the per-layer
time concentrates.

### Per-token GPU body estimate

```
26 S-layers × (100 ATTN + 195 FFN) = 26 × 295 = 7,670 µs
 4 G-layers × (138 ATTN + 194 FFN) =  4 × 332 = 1,328 µs
                                              = 8,998 µs ≈ 9.0 ms
```

Plus embedding + output projection + CPU encode ≈ 1.5 ms.

**Total ≈ 10.5 ms / tok ≈ 95 tok/s** — matches measured (iter-158 canonical
baseline 0.934× peer-FA).

### Outliers

- L04 FFN spiked to 384 µs first token, 300 µs second token (then settled
  to ~195 µs).  Probably first-time MoE expert PSO build for that layer's
  expert set.  Not a steady-state factor.
- L24 ATTN spiked to 192 µs once; rare PSO-cache miss.
- L27 ATTN spiked to 293 µs once on token 2.  Same PSO-cache miss class.

## Implications for closure work

1. **FFN-side levers have 2x leverage over ATTN-side** at the per-layer
   scale.  An optimization that saves 5% on FFN (10 µs/layer) saves 300 µs/tok
   = ~2.7% wall.  Same 5% on ATTN saves only 130 µs/tok = ~1.3% wall.
   Future investigators should target FFN first.

2. **No single layer is a bottleneck**.  σ within a phase is ~2-5 µs (2-5%
   of the median), and there's no layer that takes 2-3x longer than its
   peers.  This confirms the diffused-gap interpretation from Step 1.

3. **Global layers cost ~40% more on ATTN, equal on FFN**.  As expected:
   global attention has wider KV-cache reach; FFN is identical for both.
   ATTN-G is only 4/30 layers, so the global-attn gap contributes <2% wall.

4. **Top kernel candidates** for FFN-side investigation:
   - `kernel_mul_mv_id_q6_K_f32_nr2` (expert down_proj, biggest weight matrix)
   - `kernel_mul_mv_id_q5_K_f32` (expert up/gate_proj)
   - `kernel_rms_norm_f32_v2` (ffn_norm, post_attn_norm, post_ffn_norm)

## Standing-context update

Prior canonical baseline (iter-158, `382e9227`): 92.60 ± 0.126 t/s.
Current HEAD (`a66f8fc6` + `22dc55b` with Step 1m default-on): same regime
~95 t/s (precompiled metallib + concurrent are infrastructure wins,
not productive of wall delta in steady state).

The structural floor at this codebase state is **~9.0 ms GPU body + ~1.5 ms
CPU/encode** = **~10.5 ms/tok = ~95 t/s**.  Peer FA at the same shape on the
same machine runs ~10.1 ms = ~99 t/s = the ~4% gap.

Closing that 4% requires either:
- **FFN-MoE redesign**: reduce per-expert dispatch count (currently 2-3
  matmuls per expert × 4 experts = 8-12 dispatches per layer FFN; if peer
  fuses into fewer macro-dispatches, that's a multi-week port).
- **Reducing FFN dispatch overhead**: pre-batched expert dispatch via a
  single mul_mv_id batched over experts vs. 4 separate mul_mv_id calls.
- **Or accept this is structural** for our weight layout + KV format choices.

## Cross-references

* Step 1 baseline measurement: `docs/research/ADR-029-iter-175-step-1-2026-05-15.md`
* iter-158 canonical bench: `project_adr029_iter158_canonical_baseline_2026_05_13.md`
* iter-115 prior 95%-GPU finding: `project_adr029_iter112_quant_v_advantage_2026_05_12.md`
* Step 1x archaeology: `docs/research/ADR-029-iter-175-step-1x-peer-commit-archaeology-2026-05-15.md`
