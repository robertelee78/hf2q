# ADR-029 iter-175 Step 1av — diagnostic env-cache below measurement floor

**Date**: 2026-05-15
**HEAD**: hf2q `f6450365`, mlx-native `ff9a0ff`
**Iteration**: 53 of /loop autonomous

## Hypothesis

Steps 1an/1ao/1as/1at/1au all delivered measurable wins via env-cache
rollout.  Continue the pattern at 5 more diagnostic env-read sites in
`forward_mlx.rs` hot path:
- `HF2Q_FFN_SPLIT` (3 call sites × 30 layers = 90 reads/tok)
- `HF2Q_PER_LAYER_PHASE_GPU_TIME` (2 call sites × 30 layers = 60 reads/tok)

Estimated savings: ~150 reads/tok × 70 ns = ~10 µs/tok = ~0.1% wall.

## Method

Replaced each `std::env::var("HF2Q_X").as_deref() == Ok("1")` with a
function-local `static AtomicI8` cache (uninit/off/on tristate).
5 sites total.  Built + ran 3-cycle alt-pair.

## Result

```
[Step 1av] 3-cycle: + 5 diagnostic env reads cached
  cycle 0: 95.7 tok/s
  cycle 1: 95.8 tok/s
  cycle 2: 95.7 tok/s

hf2q : mean 95.73 t/s  σ 0.05 (0.05%)  samples: [95.7, 95.8, 95.7]
Step 1z baseline: 95.30 ± 0.21
Step 1au        : 95.90 (+0.63%)

delta vs Step 1z : +0.45%
delta vs Step 1au: -0.17%
```

**Verdict: within-noise non-win.**

The 0.17% delta vs Step 1au is consistent across all 3 cycles (σ 0.05%,
very tight), but it's slight regression rather than improvement.

## Why this is below the measurement floor

These diagnostic env vars (FFN_SPLIT, PER_LAYER_PHASE_GPU_TIME) take
the **cheap unset-env path** — `std::env::var()` returns `Err`
immediately when the var is unset, without any allocation or parsing.

Per H-N microbench (Step 1am): unset env::var was 82 ns/call.  But this
includes the case where the var IS set; the unset path may be ~50-60 ns.

Cumulative: 5 sites × 30 layers × ~60 ns = 9 µs/tok = 0.09% wall.

Cross-session thermal/PSO variance for our 3-cycle bench is ~0.1-0.2%.
The expected gain is at the noise floor.

## What this reveals about the env-cache campaign

The earlier wins (Steps 1an: +0.24%, 1at: +0.49%, 1au: +0.63%) targeted env
vars in inner loops with HIGHER call counts:
- 1an: dispatch_id_mv ~150 calls/tok × 2 reads = 300 reads
- 1at: TQ-HB compute_nsg + compute_nwg ~26 calls × 2 reads = 52 reads
- 1au: HF2Q_FUSED_MOE_WSUM_END_LAYER_V2 30 calls

Each delivered ~0.14-0.28% incremental, just above noise.

Step 1av's call count (150 reads from diagnostic envs) seems comparable
but the per-read time is lower (Err path), so total savings drop below
noise floor.

**The env-cache campaign has reached diminishing returns.**  Remaining
uncached env reads in forward_mlx.rs fire per-layer in diagnostic-only
paths with Err semantics — caching them is correct code but
sub-measurable.

## Action

Reverted the 5 cache additions to forward_mlx.rs.  HEAD remains at
`f6450365` (Step 1au).

## Standing-context update

iter-175 env-cache campaign FINAL: **+0.63% cumulative wall** (95.30 → 95.90 t/s):
- Step 1an (Q6K_ID + Q8_0_ID): +0.24%
- Step 1ao (4 more, mostly within noise)
- Step 1as (HYBRID_NWG): +0.28%
- Step 1at (TQ_NSG + TQ_NWG): +0.49%
- Step 1au (FUSED_MOE_WSUM_END_LAYER_V2): +0.63%
- Step 1av (5 diagnostic envs): -0.17% (noise-floor; reverted)

Pattern is correct and validated; further sites would be incorrect to apply
without re-verifying each yields above-noise improvement.

## Cross-references

* Step 1au commit: `hf2q/f6450365`
* Step 1at commit: `mlx-native/ff9a0ff`
* H-N env::var microbench: `mlx-native/tests/iter175_h_n_env_var_cost.rs`
* Mantra: 'Measure 3x, cut once' — bench at 0.05% σ caught the within-noise
  regression and prevented committing a non-win.
