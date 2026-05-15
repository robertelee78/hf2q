# ADR-029 iter-175 Step 1ae — peer vs hf2q `_id` kernel: hf2q WINS 3.89%

**Date**: 2026-05-15
**HEAD**: hf2q `fd99306d`, mlx-native `4905f09`
**Iteration**: 36 of /loop autonomous

## Hypothesis (H-I)

After Step 1ad showed our `_id` indirection isn't slow, the remaining
hypothesis is: peer's `kernel_mul_mv_id_q6_K_f32` may be algorithmically
faster than hf2q's `kernel_mul_mv_id_q6_K_f32_nr2` at the gemma4 down_exps
shape — and that difference is where some of the 6.35% wall gap lives.

**Testable**: load BOTH kernels from precompiled -O3 metallibs and bench
at IDENTICAL shape and buffers; measure delta.

## Method

`tests/iter175_h_i_peer_id_kernel_compare.rs`:

1. Compile peer's `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` to a
   metallib via `xcrun metal -O3` (same as hf2q's Step 1m methodology).
2. Compile hf2q's `src/shaders/quantized_matmul_id_ggml.metal` likewise.
3. Build both pipelines with their respective Function Constants:
   - peer: `FC_mul_mv_nsg=2`, `FC_mul_mv_nxpsg=1`, `FC_mul_mv_ne12=1`, `FC_mul_mv_r2=1`, `FC_mul_mv_r3=1`
   - hf2q: slots 700/701/702 = 1
4. Allocate same buffers (Q6_K weights 128 experts × 2816 × 8192/256 × 210 bytes,
   F32 input/output, U32 ids).
5. Configure dispatch geometries matching each repo's production:
   - hf2q: `tg=(704, 8, 1)`, threads=(2, 32, 1) — y-routed
   - peer: `tg=(704, 1, 8)`, threads=(32, 2, 1) — z-routed
6. 3 alt-paired cycles, BATCH=32, MEASURE=80, WARMUP=20.

## Result

```
[H-I] gemma4 MoE down_exps: N=2816 K=8192 top_k=8 (5632 TGs each)

--- cycle 0 ---
  hf2q _id_nr2                     median=  154.38us  p10= 153.39
  peer _id                         median=  160.21us  p10= 158.81

--- cycle 1 ---
  peer _id                         median=  159.69us  p10= 158.77
  hf2q _id_nr2                     median=  154.12us  p10= 152.81

--- cycle 2 ---
  hf2q _id_nr2                     median=  153.52us  p10= 152.46
  peer _id                         median=  160.09us  p10= 158.98

[H-I] aggregate (3 cycles):
  hf2q : mean 154.01us  samples [154.38, 154.12, 153.52]  σ range 0.86 µs
  peer : mean 159.99us  samples [160.21, 159.69, 160.09]  σ range 0.52 µs
  delta: peer is +3.89% vs hf2q
  verdict: HF2Q FASTER — kernel quality not the bottleneck
```

Both σ < 1 µs out of ~155 µs (~0.5-0.6%) — tightest peer-vs-hf2q measurement
in iter-175.

## Critical implication

hf2q's `kernel_mul_mv_id_q6_K_f32_nr2` is **3.89% FASTER** than peer's
`kernel_mul_mv_id_q6_K_f32` at gemma4 MoE down_exps shape.  This is the
single biggest kernel call at decode (~80% of FFN per Step 1y) per layer.

**Yet peer's overall decode wall is +5.9% faster** (Step 1aa: peer 100.94
vs hf2q 95.30 t/s).

The arithmetic doesn't work unless the wall gap is **dispersed across
other kernels and/or non-kernel overhead**:

- **Other kernels**: up_exps, gate_exps, attention path (Q/K/V/o_proj, FA-vec),
  norms (rms_norm_f32_v2), softmax, GELU, etc.  If peer's combined "rest"
  is 5-10% faster than hf2q's combined "rest", that overcomes our 3.89%
  win on down_exps.
- **Non-kernel overhead**: CPU encode time (Step 1y measured ~1.5 ms /
  10.5 ms wall = ~14% of decode), Metal scheduler latency between
  dispatches, barrier placement efficiency, command buffer commit overhead.
  Peer has 1339 dispatches/tok vs hf2q's 866; if peer's per-dispatch
  encode is faster, the wall gap concentrates there.

## What this rules in / out

**RULES OUT**:
- "Our top FFN kernel is slow vs peer" → REFUTED.  hf2q wins by 3.89%.
- "MoE indirection is slow" → REFUTED in Step 1ad; reconfirmed here.
- "Peer's z-routing is faster than y-routing" → REFUTED.  Peer's z-routed
  geometry runs 3.89% SLOWER than our y-routed.  iter-321's prior
  falsification of z-routing on dwq46 holds; today's confirms for gemma4.

**RULES IN** for future investigation:
- Other kernel-level deltas (up_exps Q5_K kernel might be where peer wins)
- Non-kernel overhead (Apple Instruments dispatch-latency profiling)
- Acceptance of structural floor at this codebase state

## What stays valuable

This is the **first head-to-head peer-vs-hf2q kernel bench at apples-to-apples
compilation** (both -O3 metallib).  Test artifact retained for:
- Rebench when hf2q or peer changes the kernel
- Pattern for benching other kernels (up_exps, attn) in future iters
- Definitive answer to "is OUR top kernel slower than peer's?" — NO.

## Cross-references

* Test artifact: `/opt/mlx-native/tests/iter175_h_i_peer_id_kernel_compare.rs`
* Step 1ad (id vs non-id): `docs/research/ADR-029-iter-175-step-1ad-id-kernel-perdispatch-2026-05-15.md`
* Step 1aa (canonical wall gap 6.35%): `docs/research/ADR-029-iter-175-step-1aa-peer-ratio-at-HEAD-2026-05-15.md`
* Step 1y profile: `docs/research/ADR-029-iter-175-step-1y-phase-profile-at-HEAD-2026-05-15.md`
* iter-321 z-routing falsification: see comment in `src/ops/quantized_matmul_id_ggml.rs:565-571`
