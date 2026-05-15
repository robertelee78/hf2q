# ADR-029 iter-175 Step 1aj — raw Metal FFI is only 162 ns/dispatch (Step 1ai overestimated 4×)

**Date**: 2026-05-15
**HEAD**: hf2q `9b0df771`, mlx-native `5627e0a`
**Iteration**: 41 of /loop autonomous

## What changed

Step 1ai estimated per-dispatch cost as ~690 ns dominated by Metal FFI
(~94% of cost).  Step 1aj microbenched the raw metal-rs FFI calls in
isolation and found:

**Raw FFI cost = 162 ns/dispatch**, not 690 ns.

## Method

`tests/iter175_h_k_dispatch_overhead.rs`:

1. Compiled a trivial noop kernel via `xcrun metal -O3`.
2. Created a metal-rs command buffer + encoder DIRECTLY (bypassing hf2q's
   `CommandEncoder` wrapper).
3. Called the SAME 6 FFI methods used in production:
   - `set_compute_pipeline_state(&pipe)`
   - `set_buffer(0, ...)`, `set_buffer(1, ...)`, `set_buffer(2, ...)`
   - `set_bytes(3, ...)`
   - `dispatch_thread_groups((1,1,1), (1,1,1))`
4. Repeated N=1000 times per command buffer, measured CPU time only
   (didn't include `wait_until_completed` in the timing).
5. 50 sample runs after 10 warmup runs.

## Result

```
[H-K] Dispatch overhead microbench (production-equivalent path)
[H-K] N=1000 dispatches per CB, MEASURE=50
[H-K] Per-CB CPU encode time (excludes GPU wait):
  median=161.7us  p10=148.0  p90=175.7
  per-dispatch CPU: 161.7 ns
  Step 1ai estimate: ~690 ns/dispatch (with 4 ffi calls)
  delta vs Step 1ai: -76.6% (likely lower b/c no tracker overhead)
```

## Where did 528 ns/dispatch go?

| Component | Cost | % of total |
|---|---|---|
| Step 1ah measured production | **0.68 µs/dispatch** | 100% |
| Step 1aj raw FFI (this) | 0.16 µs/dispatch | 24% |
| **hf2q wrapper layer overhead** | **0.52 µs/dispatch** | **76%** |

The bulk of per-dispatch cost is in hf2q's `encoder.rs` wrapper layer:
the `CommandEncoder::encode_threadgroups_with_args` function (and friends)
adds ~520 ns of bookkeeping over what raw metal-rs charges.

## What's in the 520 ns of wrapper overhead?

Speculative — Step 1ai listed:
- atomic increment (DISPATCH_COUNT)
- bucket_dispatch (cached env check, early return)
- take_pending_op_kind (Option::take)
- take_pending_buffer_ranges (mem::take on 2 Vec<MemRange>)
- match on self.capture (None branch in production)
- ensure_sample_buffer (cached env check, early return)
- get_or_create_encoder (cached path)
- sample_dispatch_pre (early return)
- assert_tg_size_multiple_of_32_if_hinted (cached env check, early return)
- sample_dispatch_post (early return)

Each operation should be ~5-10 ns.  10 operations = 50-100 ns.  But the
measured overhead is 520 ns.  That's a 5× discrepancy.

Possible explanations to investigate:
1. **Cache miss patterns** — wrapper layer touches `&mut self` fields
   triggering cache reloads on every call.
2. **Function call overhead** — multi-method-call indirection (Rust
   dispatch → wrapper → raw FFI) adds prologue/epilogue cost.
3. **Hidden allocations** — `mem::take` on Vec<MemRange> may allocate
   a new Vec if `pending_reads` was non-empty (e.g., the prior dispatch
   left state).
4. **The bookkeeping work is genuinely more than I estimated** —
   each "10 ns operation" might actually be 50 ns.
5. **`get_or_create_encoder` check** — even on cached path, the
   `Option::as_ref()` + cast might do more than expected.

This requires further micro-bench investigation — wrapping each phase of
`encode_threadgroups_with_args` in a Timer and bisecting cost.

## Implication for closure

**Step 1ai's 4-lever roadmap is INVALIDATED.**

The Step 1ai levers (Argument Buffer Encoding, Argument dedup, Direct objc,
HEAD fusion) all assumed FFI was the bottleneck.  FFI is actually fine.

The real lever is **wrapper-layer optimization**: reduce the ~520 ns of
overhead per dispatch in `CommandEncoder::encode_threadgroups_with_args`.

Closure potential:
- If we could match raw FFI cost (162 ns), per-dispatch saving = 520 ns.
- Per-token saving = 520 ns × 866 = 450 µs = **4.6% wall**.
- That alone (plus HEAD fusion ~3%) would put us at ~1.01× peer-FA.

**This is a much smaller engineering project** than the Step 1ai levers
(maybe 1-2 weeks of inlining + careful optimization of `encoder.rs`)
because we're optimizing OUR code, not designing new Metal infrastructure.

## What this iteration RULED OUT

* "Metal FFI is the bottleneck" → REFUTED.  FFI is 162 ns/dispatch
  (24% of production cost).
* "Step 1ai's lever roadmap is the right direction" → REFUTED.
  Argument Buffer Encoding etc target a non-bottleneck.

## What this iteration RULES IN

* The hf2q `encoder.rs` wrapper adds ~520 ns/dispatch over raw FFI.
* This wrapper-layer overhead is the actual closure target.
* Engineering effort to close is potentially much smaller than Step 1ai
  estimated (1-2 weeks of internal optimization, not 4-6 weeks of new
  Metal infrastructure).

## Caveats

- The 162 ns is in a maximally hot microbench with simple buffer layout.
  Real production may be slightly higher due to larger kargs payloads
  (e.g., 12-byte vs 4-byte set_bytes) or buffer cache effects.
- The 0.68 µs production number includes calls from forward_mlx.rs
  through `dispatch_tracked_*` which adds a thin tracker layer; the
  raw `CommandEncoder::encode_*` is likely closer to ~0.6 µs.
- BATCH=32 in production benches may amortize encoder setup differently
  than N=1000 here.

## Cross-references

* Step 1ai (overestimate): `docs/research/ADR-029-iter-175-step-1ai-encode-hot-path-trace-2026-05-15.md`
* Step 1ah (production wall measurement): `docs/research/ADR-029-iter-175-step-1ah-cpu-encode-localization-2026-05-15.md`
* Test artifact: `/opt/mlx-native/tests/iter175_h_k_dispatch_overhead.rs`
* mantra: "Measure 3x, cut once" — Step 1ai estimated, Step 1aj measured, found 4× discrepancy.
