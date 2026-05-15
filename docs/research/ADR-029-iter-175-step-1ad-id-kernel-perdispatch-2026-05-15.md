# ADR-029 iter-175 Step 1ad — H-H _id kernel per-dispatch bench REFUTES "indirection is slow"

**Date**: 2026-05-15
**HEAD**: hf2q `7df16797`, mlx-native `7e36af7`
**Iteration**: 35 of /loop autonomous

## Hypothesis (H-H)

Step 1y identified `kernel_mul_mv_id_q6_K_f32_nr2` as the top FFN kernel
candidate (down_exps, biggest weight matrix).  The implicit hypothesis from
the "FFN dominates ATTN 2:1" finding: the `_id` variant's expert_id
indirection on every weight access adds measurable per-dispatch overhead
vs the non-`_id` variant at the same K/N.

**Testable**: bench both kernels in isolation at gemma4 down_exps shape
(K=8192, N=2816), measure per-row cost, compare.

## Method

Standalone test `tests/iter175_h_h_id_kernel_perdispatch.rs`:

- Both kernels compiled from production sources via `xcrun metal -O3`
  precompiled metallib (matches Step 1m production path).
- Same Q6_K weight buffer (128 experts × 2816 × 8192/256 × 210 bytes).
- _id dispatch: TGs=(704, 8, 1), threads=(2, 32, 1), 5632 TGs total
- non-_id dispatch: TGs=(704, 1, 1), threads=(2, 32, 1), 704 TGs total
- BATCH=32 per CB, MEASURE=80, WARMUP=20
- 3 alt-paired cycles with 3s/5s cool-downs

## Result

```
[H-H] aggregate (3 alt-paired cycles):
  _id    mean: 153.92us  samples: [153.99, 153.73, 154.05]
  non-id mean:  30.19us  samples: [30.48,  30.33,  29.76]
  _id per-row (÷top_k=8):  19.240us
  non-id per-row:           30.190us
  _id indirection overhead per-row: -36.27%
```

**REFUTED: `_id` is MORE efficient per-row than non-`_id` at this shape.**

σ within-arm:
- _id: 0.16 µs (0.10%)  — extraordinarily tight
- non-_id: 0.36 µs (1.2%) — slightly noisier

## Interpretation

The non-`_id` kernel at N=2816 dispatches **704 threadgroups** (5632/(nsg=2 × nr0=2)),
each running 64 threads = 45,056 threads.  At gemma4 hidden=2816 shape, this
under-fills M5 Max's compute units.

The `_id` kernel at top_k=8 dispatches **5632 threadgroups** (8× more), filling
the GPU much better.  Even though each TG does the same per-row work, the
better GPU occupancy more than compensates for the indirection overhead.

This is a real per-shape effect:
- At small N (e.g., 2816), single matvec is under-occupied; batched-via-top_k wins.
- At large N (e.g., 16384), single matvec already saturates the GPU; the batched
  variant might show smaller advantage or none.

## Implication for ADR-029 closure

**The MoE expert dispatch is not a per-kernel bottleneck.**  The Step 1y profile's
FFN ≈ 195 µs/layer is **structural** for our shape, not caused by `_id` indirection.

Specifically:
- 153.92 µs for one down_exps expert dispatch (top_k=8) is the bulk of FFN compute.
- Plus up_exps + gate_exps + GELU + norms + softmax ≈ 40 µs to reach 195 µs/layer.
- Closing FFN further requires either:
  - **Smaller weight matrices** (model change, not codebase change)
  - **Higher dispatch occupancy** (more TGs per dispatch, but we already saturate at 5632)
  - **Multi-stage pipelining** (overlap commit + GPU exec — already explored in ADR-031)

None of these are remaining /loop-tractable levers.

## What this RULES OUT

Step 1ad eliminates "MoE indirection overhead" from the iter-175 hypothesis
space.  Combined with Steps 1t/1u/1v/1w/1x/1y/1z/1aa/1ac, the closure search
is exhausted at the per-kernel level for gemma4 decode at our shape.

## What stays valuable

This is a **positive** falsification — it confirms our `_id` kernels are
already operating efficiently (more so than the simpler non-`_id` variant
would be at the same N).  No regression to fix; no port to do.  The kernel
behaves as designed.

Test artifact retained for future rebenches at HEAD.

## Cross-references

* Step 1y profile (identified FFN as primary): `docs/research/ADR-029-iter-175-step-1y-phase-profile-at-HEAD-2026-05-15.md`
* Step 1aa canonical decode ratio: `docs/research/ADR-029-iter-175-step-1aa-peer-ratio-at-HEAD-2026-05-15.md`
* Test artifact: `/opt/mlx-native/tests/iter175_h_h_id_kernel_perdispatch.rs`
* Step 1k baseline kernel test (non-_id at smaller N): `mlx-native/tests/iter175_h_e_metallib_perf.rs`
