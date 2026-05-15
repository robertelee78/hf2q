# ADR-029 iter-175 Step 1ar — `#[inline]` hints on hot dispatch methods: NEUTRAL

**Date**: 2026-05-15
**HEAD**: hf2q `1391e4a0`, mlx-native `6edc809` (reverted)
**Iteration**: 48 of /loop autonomous

## Hypothesis

Step 1aq tested global `lto = "thin"` and saw -0.28% regression.  Step 1ar
tries a more targeted approach: `#[inline]` attributes on the specific
hot dispatch methods.  Rust's `#[inline]` lets the compiler inline across
crate boundaries without the global compilation cost of LTO.

Methods targeted:
- `mlx_native::CommandEncoder::encode_threadgroups_with_args`
- `mlx_native::CommandEncoder::dispatch_tracked_threadgroups_with_args`

These are called from `hf2q::serve::forward_mlx` ~150-300 times per token
through the 5-layer nested call chain identified in Step 1ak.

## Method

1. Added `#[inline]` (regular, not `inline(always)`) to the two methods.
2. Rebuilt mlx-native + hf2q.
3. 3-cycle alt-pair, gemma4-APEX-Q5_K_M tg200, M5 Max.

## Result

```
[Step 1ar] 3-cycle bench with #[inline] on 2 hot encoder methods
  cycle 0: 95.5 tok/s
  cycle 1: 95.0 tok/s
  cycle 2: 95.4 tok/s

hf2q : mean 95.30 t/s  σ 0.22 (0.23%)  samples: [95.5, 95.0, 95.4]
Step 1z baseline    : 95.30 ± 0.21 t/s
Step 1ao (6 caches) : 95.47 t/s

delta vs Step 1z    : +0.00%
delta vs Step 1ao   : -0.18%
```

**Verdict: NEUTRAL — exactly matches baseline; 0.00% change.**

## Why this is consistent with Step 1aq's LTO finding

Two possible interpretations:
1. **The compiler isn't actually inlining** even with the hint, because
   the function bodies are larger than its heuristic threshold for
   cross-crate inlining.
2. **The compiler IS inlining, but it doesn't help** — the per-call
   prologue/epilogue overhead is smaller than I estimated, OR the
   resulting larger hot-loop offsets the inlining benefit (ICache pressure
   from Step 1aq's likely cause).

Either way, the conclusion is the same: targeted `#[inline]` hints on
these specific methods don't move the wall.

## Action

Reverted the `#[inline]` additions to `mlx-native/src/encoder.rs`.  HEAD
state restored to `6edc809`.

## What's left for the unaccounted 3.1%

Step 1ak's 3.1% unaccounted orchestration overhead is now both:
- Empirically robust (multiple measurement passes confirm it)
- **Resistant to standard inlining levers** (Steps 1aq + 1ar)

Possible explanations:
- The overhead is at a finer grain than the Rust compiler can optimize
  through its standard heuristics (struct construction inside inline'd
  functions, MTLSize::new constructions, etc.).
- The overhead is in code paths that hf2q's specific call graph hides
  from cross-crate analysis (e.g., trait dispatch through MlxBuffer
  methods).
- The 3.1% may include irreducible overhead like CPU pipeline stalls
  on cache misses, branch mispredicts, and memory access latency that
  no software inlining can fix.

## Conclusion for /loop closure

Steps 1ai through 1ar have empirically narrowed the closure target multiple
times, each time invalidating the prior hypothesis with measurement.  The
remaining ~3.1% wall in dispersed orchestration overhead is **resistant to
standard compiler-level optimizations** (LTO, inline hints).

To close further would require either:
1. **Algorithmic restructuring** of the dispatch chain (collapse the 5-layer
   call chain into fewer functions; pre-bake per-layer dispatch records at
   model-load).  Multi-week project.
2. **Operator-only Apple Instruments profiling** to identify the actual
   cycle-level bottleneck.

Or accept the current state (0.9365× peer-FA decode, 1.037× peer-FA prefill)
as the practical floor at this codebase shape.

## Cross-references

* Step 1aq (LTO=thin -0.28%): `docs/research/ADR-029-iter-175-step-1aq-lto-thin-falsified-2026-05-15.md`
* Step 1ak (orchestration localization): `docs/research/ADR-029-iter-175-step-1ak-orchestration-localization-2026-05-15.md`
* Step 1an/1ao (validated wins +0.17-0.24%): mlx-native commits `f273294` and `2e381b0`
