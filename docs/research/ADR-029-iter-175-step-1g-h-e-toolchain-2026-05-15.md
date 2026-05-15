# ADR-029 iter-175 Step 1g — H-E (precompiled metallib) toolchain confirmed; empirical test deferred

**Date**: 2026-05-15
**HEAD**: hf2q `c3a3b2f6`, mlx-native `b32b81e`
**Iteration**: 6 of /loop autonomous

## Summary

H-E (precompiled `.metallib` via `xcrun metal -O3` vs runtime `newLibraryWithSource`) toolchain is verified working on our `quantized_matmul_ggml.metal` shader: `.air` and `.metallib` artifacts produced cleanly. AIR sizes vary by optimization level (-O0=31264, default=22640, -O3=21984 bytes), confirming there IS a real difference at the AIR level. **The DEFINITIVE empirical test (does kernel execution speed differ?) requires either (a) a parallel `KernelRegistry::from_metallib` path (~2-3 hour test harness + bench) or (b) a multi-day full migration of mlx-native's shader-load pipeline.** Neither fits a /loop-iteration window.

## Toolchain verification

```bash
# All commands succeed on our shader; sizes shown
xcrun -sdk macosx metal -O0      -c quantized_matmul_ggml.metal -o quantized_matmul_ggml_O0.air        #  31264 B
xcrun -sdk macosx metal          -c quantized_matmul_ggml.metal -o quantized_matmul_ggml_default.air   #  22640 B
xcrun -sdk macosx metal -O3      -c quantized_matmul_ggml.metal -o quantized_matmul_ggml.air           #  21984 B
xcrun -sdk macosx metallib              quantized_matmul_ggml.air -o quantized_matmul_ggml.metallib    #  71542 B
```

The shader compiles cleanly without external includes (`<metal_stdlib>` only) — same as llama.cpp's setup. Building a custom `.metallib` for hf2q is mechanically straightforward.

## What this DOES tell us

- **AIR optimization-level granularity is real.** Default vs -O3 differs by ~3% in AIR bytes (22640 vs 21984). Smaller AIR typically = more inlining + constant folding + dead-code elimination, which usually correlates with faster execution.
- **The `MTLLibraryOptimizationLevelDefault` (Apple's runtime default) ≠ `-O3` at the AIR-byte level on at least this shader.** Apple documentation calls Default "optimize for runtime performance" but the actual AIR output is not byte-identical to -O3.

## What this does NOT tell us

- **AIR-byte-size difference doesn't directly predict execution speed difference.** Apple's GPU machine-code generator may compile both AIRs to the same final binary. The empirical kernel-execution timing is the only definitive answer.
- **Default may already be functionally identical to -O3 at execution time.** This is the most-likely hypothesis based on Apple's documentation; needs verification.

## Why definitive testing is non-trivial

To get an apples-to-apples kernel timing comparison:

1. Need a Rust test that loads a single shader BOTH ways (`new_library_with_source` AND `new_library_with_file`)
2. Need to construct realistic input buffers (Q6_K quantized weight, F32 input, F32 output) at decode shapes (e.g. K=2816, N=4096)
3. Need to run N=1000+ dispatches each, measure GPU wall-clock per dispatch
4. Need to gate spurious confounders (warmup, thermal, sample-buffer overhead, CB count)

The existing `bench_decode_qmatmul_shapes.rs` does step (2)+(3) for the runtime-compile path. Extending it for the precompiled path requires either:
- Modify `KernelRegistry` to accept a precompiled `Library` (clean refactor, ~1-2 hours)
- OR write a parallel test that bypasses `KernelRegistry` entirely (~2-3 hours, brittle)

Neither fits within a /loop iteration.

## Decision recommendation

**Defer H-E empirical test to a dedicated session** (not /loop autonomous) where the multi-hour test-harness work is acceptable. The expected outcomes are bounded:

- **If H-E confirms** (precompile is materially faster): close iter-175 with H-E as the lever; estimate multi-day full-shader-migration cost (build system change in mlx-native to xcrun-compile all .metal files at build time, plus kernel_registry.rs refactor to load from embedded .metallib, plus rebuild integration tests).
- **If H-E falsifies** (no measurable kernel-speed difference): close iter-175 at "structural parity reached; remaining ~6-8% is the floor on M5 Max + current hardware/Apple SDK".

Both outcomes are mission-advancing.

## iter-175 cumulative status

| Step | Hypothesis | Status |
|---|---|---|
| 1 | Dispatch-count baseline | DONE — top kernels already peer-ported |
| 1b | H-A: encoder overhead | FALSIFIED — CPU <1% of wall |
| 1b | H-B: Metal compile options (runtime) | PARTIALLY FALSIFIED — both default |
| 1d | H-D: concurrency strategy | CONFIRMED — 3.5pp ceiling |
| 1e | H-D2 enabling code | LANDED — default-OFF infrastructure |
| 1f | H-D2: single-site migration | FALSIFIED — neutral |
| 1g | H-E: precompiled .metallib | TOOLCHAIN CONFIRMED, empirical test DEFERRED |

**Three open paths beyond /loop scope**:
1. **H-D global migration** (multi-day to multi-week): migrate all ~400 hand-placed barriers to `dispatch_tracked_*`, remove redundant barriers, validate byte-identity + coherence. Targets ~3.5pp.
2. **H-E full migration** (multi-day): precompile all 30+ shaders to bundled `.metallib`, modify kernel_registry. Targets unknown (need empirical test first).
3. **H-C cache/memory layout** (operator-runs Apple Instruments + multi-day analysis): identify cache-miss patterns, restructure memory access. Targets unknown.

## Cross-references

- Step 1 (dispatch baseline): `docs/research/ADR-029-iter-175-step-1-dispatch-distribution-2026-05-15.md`
- Step 1b (H-A/H-B): `docs/research/ADR-029-iter-175-step-1b-encoder-and-compile-options-2026-05-15.md`
- Step 1d (H-D 4-arm): `docs/research/ADR-029-iter-175-step-1d-concurrency-lever-2026-05-15.md`
- Step 1f (H-D2 falsification): `docs/research/ADR-029-iter-175-step-1f-thermal-fair-bench-2026-05-15.md`
- iter-174 final outcome (parallel pattern): memory `project_adr029_iter174_FINAL_session_outcome_2026_05_13`
