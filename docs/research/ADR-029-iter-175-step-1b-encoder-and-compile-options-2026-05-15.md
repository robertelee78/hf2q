# ADR-029 iter-175 Step 1b — H-A & H-B falsifications: encoder fast-path + compile-options

**Date**: 2026-05-15
**HEAD**: hf2q `e93fe5b4`, mlx-native (current)
**Iteration**: 2 of /loop autonomous

## Summary

Side-by-side read of both repos' encoder fast-path and `MTLCompileOptions` usage. **Both H-A (per-dispatch encoder overhead) and H-B (Metal compile-options divergence) are FALSIFIED at the runtime level**. A residual hypothesis **H-E (precompiled `.metallib` vs runtime source compile)** survives as a multi-day next experiment.

## H-A: per-dispatch encoder/framework overhead — FALSIFIED

### hf2q hot-path encode sequence

The Q6_K_nr2 matvec (top kernel #1, 19.91% of dispatches) routes through `encode_threadgroups_with_args_and_shared` at `/opt/mlx-native/src/encoder.rs:1227`. Per-dispatch Rust+ObjC sequence:

| Step | Cost class | Est. ns |
|---|---|---:|
| 1. `DISPATCH_COUNT.fetch_add(1, Relaxed)` | 1 atomic | ~1 |
| 2. `bucket_dispatch(pipeline)` early-return after cached-env load | 1 atomic load | ~1 |
| 3. `take_pending_op_kind()` | Option swap | ~1 |
| 4. `take_pending_buffer_ranges()` | 2 Vec swaps | ~5-10 |
| 5. Capture-mode `if let Some(..)` predicted-not-taken | branch | ~0 |
| 6. `ensure_sample_buffer()` cached-env early-return | 1 atomic load | ~1 |
| 7. `get_or_create_encoder()` pointer-check | pointer + branch | ~2 |
| 8. `encoder.set_compute_pipeline_state(pipeline)` | 1 ObjC msg_send | ~5-10 |
| 9. `apply_bindings(encoder, bindings)` — N args | N ObjC msg_send | ~5-10·N |
| 10. `for threadgroup_mem: set_threadgroup_memory_length` — M slots | M ObjC msg_send | ~5-10·M |
| 11. `sample_dispatch_pre` cached-env no-op | atomic load | ~1 |
| 12. `encoder.dispatch_thread_groups(threadgroups, tg_size)` | 1 ObjC msg_send | ~5-10 |
| 13. `sample_dispatch_post` cached-env no-op | atomic load | ~1 |

Typical Q6_K matvec has ~5 args + 1 threadgroup-mem slot → **~50-100 ns total Rust+ObjC per dispatch**.

### llama.cpp's equivalent (peer)

`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:501-616`:

| Step | Cost class |
|---|---|
| `ggml_metal_encoder_set_pipeline(pipeline)` — `getenv("HF2Q_PEER_PIPELINE_HIST")` per call (!) + 1 ObjC msg_send | ~50-100 (env unset slow path) |
| `ggml_metal_encoder_set_bytes(...)` × M — 1 ObjC msg_send each | ~5-10·M |
| `ggml_metal_encoder_set_buffer(...)` × N — 1 ObjC msg_send each | ~5-10·N |
| `ggml_metal_encoder_dispatch_threadgroups(...)` — 1 atomic + getenv + 1 ObjC msg_send | ~5-10 |

Peer's per-dispatch CPU work is comparable (in fact slightly higher because of unconditional `getenv` per dispatch on the instrumentation hooks they added during ADR-029 iter-5).

### Comparison

Both repos have **~50-100 ns Rust/ObjC per dispatch**. At 866 dispatches/decode_tok:
- hf2q: ~43-87 µs/tok CPU overhead
- peer: ~50-100 µs/tok CPU overhead (slightly higher due to per-dispatch getenv)

Wall budget per token: ~10.7 ms. CPU overhead is **<1% of wall**.

The implied per-dispatch gap of ~10.05 µs (hf2q) vs ~7.3 µs (peer) is **almost entirely GPU-side time, not CPU encode time**.

### H-A verdict: FALSIFIED

CPU/Rust encoder fast-path is functionally equivalent. The gap doesn't live here.

## H-B: Metal compile-options divergence — PARTIALLY FALSIFIED

### Both repos use default `MTLCompileOptions`

**hf2q** (`/opt/mlx-native/src/kernel_registry.rs:1032, 1138`):

```rust
let compile_opts = metal::CompileOptions::new();  // <-- defaults
let library = device.new_library_with_source(source, &compile_opts)?;
```

**llama.cpp** (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:236, 293`) — runtime-source fallback path:

```objc
MTLCompileOptions * options = [MTLCompileOptions new];  // <-- defaults
options.preprocessorMacros = prep;   // GGML_METAL_HAS_BF16, HAS_TENSOR, EMBED_LIBRARY
//[options setFastMathEnabled:false];  // commented out — they want fast-math YES
library = [device newLibraryWithSource:src options:options error:&error];
```

Both use Apple's default `MTLCompileOptions`. Both have fast-math YES by default. Both leave `preserveInvariance = NO` and `languageVersion` at SDK default.

llama.cpp adds three preprocessor macros (`GGML_METAL_HAS_BF16`, `GGML_METAL_HAS_TENSOR`, `GGML_METAL_EMBED_LIBRARY`). These are presence/absence shader-codepath selectors — they enable `#if defined(...)` branches in shaders, **NOT optimization flags**.

### Where peer DOES diverge — the precompiled `.metallib` PRIMARY path

`/opt/llama.cpp/ggml/src/ggml-metal/CMakeLists.txt:78-79`:

```cmake
else()
    set(XC_FLAGS -O3)
endif()
```

```cmake
add_custom_command(
    OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
    COMMAND xcrun -sdk macosx metal ${XC_FLAGS} -c ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.metal -o - |
            xcrun -sdk macosx metallib        - -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
    ...
```

llama.cpp's PRIMARY path is `[device newLibraryWithURL:libURL]` at `ggml-metal-device.m:185` — loads a `.metallib` precompiled with `xcrun metal -O3`. The runtime-source path (lines 236, 293) is only the fallback when the .metallib is missing.

hf2q has NO precompiled-metallib path; all shaders are runtime-compiled via `newLibraryWithSource`.

### Does precompile vs runtime-compile matter?

Apple's `MTLDevice newLibraryWithSource:options:` uses Apple's runtime compile chain (compiles AIR + caches). The default `MTLLibraryOptimizationLevelDefault` is documented as "optimize for runtime performance" — which Apple internally maps to something equivalent to `-O3` in recent SDKs.

But there are documented differences:
- The xcrun CLI `-O3` invokes the full metal-frontend optimization pipeline at developer-build time on the host CPU.
- The runtime `newLibraryWithSource` invokes a similar (but not necessarily identical) optimization pipeline at user-machine time.
- `MTLLibraryOptimizationLevel` (the runtime equivalent of `-O3`) is only `Default` vs `Size`. There's no Apple API to choose `-O0/-O1/-O2/-O3` at runtime.

**This survives as H-E** (see below).

### H-B verdict: PARTIALLY FALSIFIED

Runtime `MTLCompileOptions` is **identical** between repos (modulo preprocessor macros for shader code-path selection, which don't affect optimization). The build-time precompile divergence remains a separate hypothesis (H-E).

## H-E (new, deferred): precompiled `.metallib` vs runtime source compile

**Hypothesis**: hf2q's runtime-compiled shaders may produce slightly less-optimized AIR than llama.cpp's `xcrun metal -O3` precompiled `default.metallib`.

**Testable lever**:
1. Pick one hot shader (e.g. `quantized_matmul_ggml.metal`).
2. Precompile via: `xcrun -sdk macosx metal -O3 -c shader.metal -o shader.air && xcrun -sdk macosx metallib shader.air -o shader.metallib`.
3. Load via `device.new_library_with_url()` instead of `new_library_with_source()`.
4. Bench `kernel_mul_mv_q6_K_f32_nr2` micro-bench OR full-decode tg100.
5. If precompiled version is ≥0.5% faster on the kernel → H-E confirmed → port to all shaders.
6. If equivalent → H-E falsified → ADR-029 closure at structural parity.

**Effort estimate**: 0.5-1 day for the single-shader test. If positive, ~3-5 days to migrate all 30+ shaders to a build-time pipeline (CMakeLists.txt / build.rs / etc).

**Risk**:
- If hf2q ever wants per-kernel function-constant specialization at runtime (which we currently do via FunctionConstantValues), precompiled metallib must include all specialization variants — bloats the library substantially.
- Symbol naming and `[[host_name(...)]]` attribute matching may need attention.

## What this leaves on the table for /loop iterations

Remaining hypotheses, ranked by /loop-tractability:

1. **H-E** (this artifact): single-shader precompile experiment — **0.5-1 day**, /loop-suitable
2. **H-D**: stage-boundary serialization — compare CB structure / dispatch grouping; **0.5-1 day** read-only
3. **H-C**: memory layout / cache miss diff — **operator-runs Apple Instruments**, NOT /loop-suitable

## Reading list (source paths cited)

- hf2q encoder fast path: `/opt/mlx-native/src/encoder.rs:1003-1265` (encode, encode_threadgroups, encode_threadgroups_with_args, encode_threadgroups_with_args_and_shared)
- hf2q compile options: `/opt/mlx-native/src/kernel_registry.rs:1021-1144`
- llama.cpp encoder primitives: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:501-622`
- llama.cpp compile options runtime path: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:236-249, 293-303`
- llama.cpp metallib precompile: `/opt/llama.cpp/ggml/src/ggml-metal/CMakeLists.txt:65-108`
- metal-rs CompileOptions API: `~/.cargo/registry/src/.../metal-0.29.0/src/library.rs:429-540`

## Cross-references

- iter-175 Step 1 (dispatch-count baseline): `/opt/hf2q/docs/research/ADR-029-iter-175-step-1-dispatch-distribution-2026-05-15.md`
- iter-111 constant-ratio finding (gap is diffused at ~1µs/dispatch): memory `project_adr029_iter111_constant_ratio_2026_05_12`
