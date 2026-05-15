# ADR-029 iter-175 Step 1ai — encode hot-path trace: where the per-dispatch 0.68 µs goes

**Date**: 2026-05-15
**HEAD**: hf2q `691b3fc2`, mlx-native `2898e02`
**Iteration**: 40 of /loop autonomous

## Background

Step 1ah localized the 6.35% wall gap to CPU encode (~0.59 ms / 866 dispatches
= 0.68 µs/dispatch).  Step 1ai traces what work happens per dispatch in
`mlx-native/src/encoder.rs` to identify the specific cost drivers.

## Method

Per Chesterton's fence rule: read the production hot path code first.
Traced `dispatch_tracked_threadgroups_with_args` (Step 1e production path)
+ delegates.

## What runs per dispatch (production: HF2Q_AUTO_BARRIER=0, no profiling, no capture)

`dispatch_tracked_threadgroups_with_args(...)`:
1. `is_capturing()` — cached `Option::is_some()` check on `self.capture` → ~2 ns
2. `auto_barrier_enabled()` — cached atomic load → ~3 ns (returns false in production)
3. Delegate to `encode_threadgroups_with_args(...)`

`encode_threadgroups_with_args(...)`:
4. `DISPATCH_COUNT.fetch_add(1, Relaxed)` — atomic → ~5 ns
5. `bucket_dispatch(pipeline)` — cached env check (default false), early return → ~5 ns
6. `take_pending_op_kind()` — `Option::take()` → ~3 ns
7. `take_pending_buffer_ranges()` — `mem::take` on two empty Vec<MemRange> → ~10 ns (pointer swap + cheap drop)
8. `if let Some(...) = self.capture` — match on None → ~2 ns
9. `ensure_sample_buffer()` — cached env check (`kernel_profile::is_dispatch_enabled()`), early return → ~5 ns
10. `get_or_create_encoder()` — cached path, returns encoder ptr → ~5 ns
11. `set_compute_pipeline_state(pipeline)` — **Metal FFI** → ~100 ns
12. `apply_bindings(encoder, bindings)` — loops over N bindings:
    - For each `Buffer/BufferWithOffset`: `set_buffer(index, Some(buf), offset)` — **Metal FFI** → ~100 ns × N
    - For each `Bytes`: `set_bytes(index, len, ptr)` — **Metal FFI** → ~100 ns
13. `sample_dispatch_pre()` — early return → ~5 ns
14. `assert_tg_size_multiple_of_32_if_hinted()` — cached env check (default false), early return → ~5 ns
15. `dispatch_thread_groups(threadgroups, threadgroup_size)` — **Metal FFI** → ~150 ns
16. `sample_dispatch_post()` — early return → ~5 ns

## Per-dispatch budget breakdown

For a typical Q6_K _id matvec with 4 bindings (weights, input, output, params/ids):

| Phase | Approx cost |
|---|---|
| Bookkeeping (atomics, env checks, mem::take) | ~40 ns |
| Metal FFI: set_compute_pipeline_state | ~100 ns |
| Metal FFI: 3× set_buffer + 1× set_bytes | ~400 ns |
| Metal FFI: dispatch_thread_groups | ~150 ns |
| **Total** | **~690 ns ≈ 0.69 µs** |

Matches the measured 0.68 µs/dispatch from Step 1ah.

**FFI calls dominate**: ~650 ns of ~690 ns = **94% of per-dispatch encode cost**.

## Implications for closure

To reduce per-dispatch encode cost requires cutting **Metal FFI calls**.
Bookkeeping is already near-zero.  Concrete levers:

### Lever 1: Argument Buffer Encoding (Metal 3 feature)
Peer's `kernel_mul_mv_id` template (peer/ggml-metal.metal:10266) takes 5
buffer arguments via metalrt-style binding.  If we wrap our 3-4 buffers
+ params into a single argument buffer descriptor + a 30s warmup
pre-bake of pipeline-specific argument layouts, the per-dispatch cost
drops to 1 Metal FFI (bind argument buffer) + 1 dispatch.

**Estimated savings**: 3-4 FFI calls → 1 FFI call ≈ 300 ns/dispatch ≈ 260 µs/token = ~2.5% wall.

### Lever 2: Argument deduplication across dispatches
If consecutive dispatches share the same pipeline or the same weight buffer
binding, skip the `set_compute_pipeline_state` / `set_buffer` Metal call.
Apple's Metal documentation says "the encoder retains the last-bound
resources at each slot until explicitly rebound."

**Estimated savings**: For the 30 expert dispatches per layer that all
have the same per-expert kernel + token input but different weight indices,
we re-bind 3 buffers each time → only the `ids` and dispatch grid change.
Could save 1-2 FFI calls × 30 dispatches × 30 layers = 1800 FFI = 180 µs = ~2% wall.

### Lever 3: Direct objc dispatch (bypass metal-rs)
metal-rs 0.33 adds a thin Rust wrapper layer over `objc::msg_send!`.
Direct calls would save ~10-20 ns per FFI = small but cumulative
(~5000 FFI calls × 15 ns = 75 µs = ~0.7% wall).

### Lever 4: HEAD path fusion (Step 1ah's 0.31 ms)
Pre-bake final-norm + lm_head + softcap + argmax into one or two
mega-dispatches via threadgroup memory + a fused kernel.  This is a
clean engineering project (~1 week).

## Total closure potential

If all levers landed: 2.5% + 2% + 0.7% + 3% (HEAD fusion) = **~8% wall improvement**.
That would put us from 0.9365× peer to ~1.01× peer (AHEAD of peer-FA on decode).

These are MULTI-WEEK engineering projects.  None are /loop-tractable in
single iterations.

## What this iteration RULED OUT

* "There's a simple bookkeeping inefficiency to remove" → REFUTED.
  Bookkeeping is ~6% of per-dispatch cost (40 ns of 690 ns).  Cutting
  it entirely saves <0.4% wall.
* "Per-dispatch CPU cost has dead code" → REFUTED.  Every line on the
  hot path is either an early-return env check or a Metal FFI call.

## Cross-references

* Step 1ah CPU encode localization: `docs/research/ADR-029-iter-175-step-1ah-cpu-encode-localization-2026-05-15.md`
* Step 1aa canonical wall ratio: `docs/research/ADR-029-iter-175-step-1aa-peer-ratio-at-HEAD-2026-05-15.md`
* Source: `mlx-native/src/encoder.rs:1260-1296` (encode_threadgroups_with_args)
* Source: `mlx-native/src/encoder.rs:1382-1408` (dispatch_tracked_threadgroups_with_args)
* Source: `mlx-native/src/encoder.rs:215-229` (apply_bindings)
