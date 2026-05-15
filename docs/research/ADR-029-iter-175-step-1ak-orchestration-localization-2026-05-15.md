# ADR-029 iter-175 Step 1ak — 2nd correction: overhead is in forward_mlx.rs orchestration, NOT encoder.rs

**Date**: 2026-05-15
**HEAD**: hf2q `45b93cc8`, mlx-native `3bb96f2`
**Iteration**: 42 of /loop autonomous

## What changed (again)

Step 1ai attributed per-dispatch overhead to Metal FFI (~94%).  Step 1aj
corrected: FFI is only 24%; "76% is in encoder.rs wrapper".

**Step 1ak corrects AGAIN: encoder.rs wrapper is only 5%.  The real
overhead (67%) is in forward_mlx.rs orchestration.**

## Method

`tests/iter175_h_l_wrapper_overhead.rs` benchmarks the SAME trivial
kernel through 3 progressively-deeper API layers:

- **ARM A**: Direct metal-rs (no hf2q wrapper at all)
- **ARM B**: `CommandEncoder::encode_threadgroups_with_args` (the encoder
  wrapper, but NOT through `dispatch_tracked_*` tracker)
- **ARM C**: `CommandEncoder::dispatch_tracked_threadgroups_with_args`
  (the exact production hot path for Step 1e q6_K_nr2)

All three use the same noop kernel, same 4 bindings, same threadgroup
geometry, same N=1000 dispatches per CB, MEASURE=50.

## Result

```
[H-L] N=1000 dispatches/CB, MEASURE=50
  ARM A raw FFI            : 191.5 us/CB = 191.5 ns/dispatch
  ARM B CommandEncoder     : 205.0 us/CB = 205.0 ns/dispatch
  ARM C dispatch_tracked   : 228.1 us/CB = 228.1 ns/dispatch

[H-L] Per-dispatch breakdown:
  Raw FFI                : 191.5 ns
  + CommandEncoder wrap   : 205.0 ns  (Δ +13.5 ns)
  + dispatch_tracked      : 228.1 ns  (Δ +23.0 ns)

[H-L] Step 1ah production : ~680 ns/dispatch
```

## Updated attribution

| Layer | Per-dispatch cost | % of total | Cumulative |
|---|---|---|---|
| Raw Metal FFI | 192 ns | 28% | 192 ns |
| encoder.rs `encode_threadgroups_with_args` wrap | +14 ns | 2% | 205 ns |
| encoder.rs `dispatch_tracked_*` tracker | +23 ns | 3% | 228 ns |
| **forward_mlx.rs orchestration (the gap)** | **+452 ns** | **67%** | **680 ns** |

The 452 ns "missing" between the deepest encoder.rs path and production
must be in the code that CALLS the encoder — i.e., the orchestration
in `forward_mlx.rs` (and supporting modules).

## What's in the 452 ns of orchestration overhead?

Speculative — needs further bisect:

1. **Pipeline registry HashMap lookups** — each dispatch calls
   `registry.get_pipeline(name, device)` which does a HashMap lookup +
   optional pipeline creation.  Cached after first call but still a
   HashMap probe = ~50 ns.
2. **Param struct construction** — e.g., `GgmlMatvecGpuParams { ne00, ... }`
   is built fresh on each call from the Rust caller's params, then
   `as_bytes()` is called.  Possibly ~50-100 ns.
3. **GraphSession::barrier_between** logic — checks if any of the read/write
   buffers need a barrier; runs a HashMap intersection or similar.
4. **Buffer offset arithmetic / view computation** — for batched cases.
5. **Nested method-call dispatch** — `forward_decode` → `encode_one_layer`
   → `attn_norm_qkv_rope` → `dispatch_id_mv` → `encoder.dispatch_tracked_*`
   is 5 layers of function calls per dispatch, each with prologue/epilogue.
6. **Conditional checks** — env-var reads (HF2Q_SPLIT_TIMING, HF2Q_AUTO_BARRIER,
   etc.) — should be cached but might not all be.

## Implication for closure

The orchestration layer is THE concrete closure target.  Each ~50 ns saving
per dispatch × 866 dispatches × 30 layers = 0.45 ms / token = ~4.5% wall.
Even halving the 452 ns to 226 ns saves ~2.3% wall.

**Estimated effort**: ~1 week of careful profiling + inlining + caching in
`forward_mlx.rs` and friends.  Much smaller than the multi-week Metal
infrastructure rewrites that Step 1ai recommended.

## Mantra in action

Iteration sequence:
- Step 1ai: estimated FFI dominates → recommended Metal-layer levers (~4-6 weeks)
- Step 1aj: empirically measured raw FFI = 162 ns, 76% in "wrapper" → recommended encoder.rs levers (~1-2 weeks)
- Step 1ak: bisected wrapper = only +37 ns; 67% in orchestration → recommended forward_mlx.rs levers (~1 week)

**Three corrections in three iterations.**  Each saved progressively more
mistaken engineering work.  "Measure 3x, cut once" applied recursively:
each measurement refined the target.

## Cross-references

* Step 1ai (overestimate #1, FFI): `docs/research/ADR-029-iter-175-step-1ai-encode-hot-path-trace-2026-05-15.md`
* Step 1aj (correction #1, wrapper): `docs/research/ADR-029-iter-175-step-1aj-raw-ffi-microbench-2026-05-15.md`
* Step 1ah (production wall): `docs/research/ADR-029-iter-175-step-1ah-cpu-encode-localization-2026-05-15.md`
* Step 1ag (kernels not the bottleneck): `docs/research/ADR-029-iter-175-step-1ag-down-exps-q8_0-corrected-shape-2026-05-15.md`
* Test artifact: `/opt/mlx-native/tests/iter175_h_l_wrapper_overhead.rs`
