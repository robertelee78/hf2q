# Current Inference Path — Architecture Map

> As-built map of the hf2q inference pipeline. Read from the source at
> commit ba5f446 (2026-04-11). Documents what IS, not what should be.

---

## 1. Entry Point and CLI

**File:** `src/main.rs` -> `src/cli.rs` -> `src/serve/mod.rs`

The binary dispatches on `Command::Generate(args)`, which calls
`serve::cmd_generate(args)` at `main.rs:123`. The `Serve` subcommand
exists in the CLI enum but is unimplemented (prints a stub message and
returns `Ok(())`).

### Mode Flags (all default to "fused" or "in-place")

| Flag | Default | Values | What it controls |
|------|---------|--------|------------------|
| `--moe-kernel` | `fused` | `loop`, `fused` | MoE expert dispatch: per-expert QMatMul loop vs fused `kernel_mul_mv_id_*` |
| `--rms-norm-kernel` | `fused` | `loop`, `fused` | RmsNorm: 11-op candle chain vs single Metal dispatch |
| `--rope-kernel` | `fused` | `loop`, `fused` | RoPE: 9-op `rope_apply` chain vs single Metal dispatch |
| `--lm-head-kernel` | `fused` | `loop`, `fused` | lm_head: F32 dense matmul vs F16 gemm |
| `--kv-cache-kernel` | `in-place` | `slice-scatter`, `in-place` | KV cache append: slice_scatter+contiguous vs slice_set |

All defaults are set to their optimized variants. The `loop`/`slice-scatter`
modes exist as bisect-safety fallbacks.

### Device Selection

`select_device()` at `mod.rs:1001`: `#[cfg(feature = "metal")]` returns
`Device::new_metal(0)`; `#[cfg(feature = "cuda")]` returns
`Device::new_cuda(0)`; fallback returns `Device::Cpu`. In practice, the
entire inference path requires Metal -- see Section 8.

---

## 2. Model Loading

**Flow:** `cmd_generate` -> `Gemma4Config::from_config_json` -> `GgufModel::load` -> `Gemma4Model::load_with_modes`

### Step 1: Config parse (`serve/config.rs`)

Parses `config.json` into `Gemma4Config`. Model is Gemma 4 A4B:
30 layers, 16 attention heads, sliding window 1024. Layer types alternate
between sliding (head_dim=256, 8 KV heads) and full/global (head_dim=512,
2 KV heads), with every 6th layer being full attention.

### Step 2: GGUF load (`serve/gguf_loader.rs`)

`GgufModel::load(path, device)` opens the GGUF file, parses the header
via `candle_core::quantized::gguf_file::Content::read`. Tensors are loaded
on demand via `get_tensor` (dequantizes to target dtype on the device) or
`get_qtensor` (keeps quantized form on device). The `device` field stores
the target device; all tensors land there.

### Step 3: Weight loading (`serve/gemma4.rs:load_with_modes`)

Called at `mod.rs:722`. The function:

1. **Creates shared dispatch counters** (`DispatchCounters::new()`)
2. **Compiles fused kernel pipelines** at load time (if modes are `Fused`):
   - `RmsNormPipelines` via `rms_norm_kernel::RmsNormKernel::fused_mode(md)` -- compiles MSL source
   - `RopePipelines` via `rope_kernel::RopeKernel::fused_mode(md)` -- compiles MSL source
   - Both bail if device is not Metal
3. **Loads embedding** from `token_embd.weight`, transposes if needed
4. **Loads `rope_freqs.weight`** (global-layer frequency mask, shape `[256]`)
5. **Creates two `RotaryEmbedding` instances** (sliding + global), shared via `Arc`
6. **Loads 30 decoder layers**, each containing:
   - `Attention`: Q/K/V/O projections (QLinear), Q/K norms, RoPE ref, KV cache
   - `Mlp`: gate/up/down projections (QLinear) -- dense SwiGLU
   - `MoeBlock`: router, 128 experts (per-expert QMatMul + optional 3D fused QStorage), scales
   - 7 RmsNorm instances per layer
   - `layer_scalar` weight
7. **Loads final norm** (`output_norm`)
8. **Ties lm_head to embedding** (`lm_head_weight = embed_w`)
9. **Optionally creates F16 lm_head copy** (if `--lm-head-kernel=fused`)

### Warmup

After loading, `cmd_generate` runs two warmup forward passes
(`mod.rs:763-803`): a 1-token decode pass and a 10-token prefill pass.
Both are followed by `clear_kv_cache()`. The prefill warmup forces a
GPU sync to drain pending commands before real inference begins.

---

## 3. Forward Pass (the hot path)

**Function:** `Gemma4Model::forward(input_ids, seqlen_offset)` at `gemma4.rs:2274`

### Per-token decode sequence (seq_len=1):

```
1. Embedding lookup + scale by sqrt(hidden_size)           [2 ops]
2. For each of 30 layers (DecoderLayer::forward):
   a. input_layernorm (RmsNorm)                            [1 or 11 ops]
   b. Attention::forward:
      - Q/K/V projections (QLinear::forward x3)            [3 ops]
      - Reshape Q, K, V                                    [2-3 ops]
      - Q norm, K norm (RmsNorm x2)                        [2 or 22 ops]
      - V unit norm (rms_norm_unit)                        [1 or 9 ops]
      - Transpose Q, K, V                                  [3 ops]
      - RoPE apply (RotaryEmbedding::apply)                [2 or 10 ops]
      - KV cache append                                    [3 or 6 ops]
      - SDPA (candle_nn::ops::sdpa, vector kernel)         [1 op]
      - Transpose + reshape output                         [2 ops]
      - O projection (QLinear::forward)                    [1 op]
   c. post_attention_layernorm (RmsNorm)                   [1 or 11 ops]
   d. Residual add (xs + attn_out)                         [1 op]
   e. pre_feedforward_layernorm (RmsNorm)                  [1 or 11 ops]
   f. Dense MLP (Mlp::forward):
      - gate_proj, gelu, up_proj, mul, down_proj           [5 ops]
   g. post_feedforward_layernorm_1 (RmsNorm)               [1 or 11 ops]
   h. pre_feedforward_layernorm_2 (RmsNorm)                [1 or 11 ops]
   i. MoE (MoeBlock::forward):
      - Router norm + scale + projection + softmax         [~14 ops]
      - Top-k sort/gather                                  [~9 ops]
      - Expert dispatch (fused: 2 kernel dispatches        [~18 ops fused
        + SwiGLU + scale gather + reduction)                or ~76 ops loop]
   j. post_feedforward_layernorm_2 (RmsNorm)               [1 or 11 ops]
   k. mlp_normed + moe_normed sum                          [1 op]
   l. post_feedforward_layernorm (RmsNorm + residual add)  [1-2 or 12 ops]
   m. Layer scalar multiply                                [1 op]
3. Narrow to last token                                    [1 op]
4. Final norm (RmsNorm)                                    [1 or 11 ops]
5. lm_head: normed @ lm_head_weight.T                      [2 or 4 ops]
6. Unsqueeze                                                [1 op]
7. Softcapping (div + tanh + mul)                          [3 ops]
```

**Decode loop** (`mod.rs:run_single_generation`):
- Greedy (T=0, rep_penalty=1.0): routes to `run_decode_greedy_batched`
  which chains `GREEDY_WINDOW=4` forward passes before draining via
  a single `to_vec1` sync.
- Non-greedy: per-token sync via `sample_token` -> `argmax().to_scalar()`.

---

## 4. The Kernel Modules

### `moe_kernel` (`serve/moe_kernel.rs`, 628 lines)

- **What:** Wraps candle's pre-compiled `kernel_mul_mv_id_*` Metal kernels
  (Q6_K, Q8_0, etc.) for batched index-driven quantized matmul.
- **Provides:** `call_quantized_matmul_mv_id_t()` function, `MoeDispatchShape` struct.
- **Depends on:** `candle_metal_kernels` (pre-compiled pipeline lookup via `Kernels`).
- **Compilation:** `#[cfg(feature = "metal")]` gated in `serve/mod.rs:7`. Does NOT
  compile without Metal. This is correct -- it uses Metal-only APIs.
- **Called from:** `MoeBlock::forward_fused` (also `#[cfg(feature = "metal")]` gated).

### `rms_norm_kernel` (`serve/rms_norm_kernel.rs`, 871 lines)

- **What:** Runtime-compiles a custom Metal kernel (`kernel_rms_norm_fuse_impl`)
  ported byte-for-byte from llama.cpp. Supports F=1 (unit), F=2 (weighted),
  F=3 (weighted + residual add).
- **Provides:** `RmsNormPipelines`, `RmsNormKernel` (mode+optional pipelines),
  `rms_norm_fused()` dispatch function, `RmsNormKernelMode` enum.
- **Depends on:** `candle_metal_kernels::metal::*`, `objc2_metal`.
- **Compilation:** Always compiled (NOT cfg-gated in `mod.rs:8`). This is a
  build problem -- imports `candle_metal_kernels` and `objc2_metal` which are
  optional deps. **Without `--features metal`, this module fails to compile.**
- **Called from:** `RmsNorm::forward`, `rms_norm_unit`, `RmsNorm::forward_with_post_residual`.

### `rope_kernel` (`serve/rope_kernel.rs`, 1351 lines)

- **What:** Runtime-compiles custom Metal kernels (`kernel_rope_norm`,
  `kernel_rope_neox`) ported from llama.cpp. Stride-aware -- no `.contiguous()`
  bounce needed on Q/K inputs.
- **Provides:** `RopePipelines`, `RopeKernel`, `rope_fused()`, `RopeKernelMode`,
  `RopeVariant` (Norm and Neox).
- **Depends on:** `candle_metal_kernels::metal::*`, `objc2_metal`.
- **Compilation:** Always compiled (NOT cfg-gated in `mod.rs:9`). Same build
  problem as `rms_norm_kernel`.
- **Called from:** `RotaryEmbedding::apply`.

### `lm_head_kernel` (`serve/lm_head_kernel.rs`, 372 lines)

- **What:** Dispatches the final vocab projection via candle's native F16 gemm
  (`call_mlx_gemm`) instead of the F32 dense matmul. No custom Metal kernel.
- **Provides:** `LmHeadKernelMode` enum, `lm_head_forward_fused()` function.
- **Depends on:** Only `candle_core` (DType, Tensor). No Metal-specific imports.
- **Compilation:** Always compiled. Builds without Metal (F16 matmul via candle's
  standard `Tensor::matmul` path). However, candle's CPU backend does not
  implement F16 matmul, so it would fail at runtime on CPU.
- **Called from:** `Gemma4Model::forward`.

---

## 5. Metal Dispatch Pattern

### How a custom kernel dispatch works (rms_norm, rope):

1. At model load: `Device::new_library_with_source(msl_source)` compiles MSL
   to a `Library`. `new_compute_pipeline_state(library, function_name)` creates
   a `ComputePipeline`. These are cached in `Arc<RmsNormPipelines>` /
   `Arc<RopePipelines>`.

2. At dispatch time:
   ```
   metal_device.command_encoder()?  // borrows candle's shared encoder
   encoder.set_compute_pipeline_state(pipeline)
   encoder.set_bytes(kargs, 0)      // argument struct at binding 0
   encoder.use_resource(src_buf)    // input buffer
   encoder.use_resource(dst_buf)    // output buffer (freshly allocated)
   encoder.dispatch_thread_groups(grid, threads_per_group)
   // encoder drops -> dispatch is enqueued in candle's command pool
   ```

3. The dispatch is in-order with all other candle ops on the same queue.
   No explicit sync. The 50-dispatch (now bumped to 100 via vendor patch)
   recycle threshold in `commands.rs` cycles command buffers automatically.

### How a candle QMatMul dispatch works (MoE fused):

1. Uses candle's pre-compiled `Source::Quantized` pipeline library.
2. `Kernels::load_pipeline(device, source, kernel_name)` looks up the
   cached pipeline by symbol name (e.g. `kernel_mul_mv_id_q6_K_f32`).
3. Same shared encoder pattern as custom kernels.

### How candle's built-in ops dispatch (matmul, softmax, etc.):

Each `Tensor::matmul`, `Tensor::sqr`, etc. call goes through
`candle_core::metal_backend::MetalStorage` methods, which internally
call `metal_device.command_encoder()`, set up the kernel, dispatch, and
drop the encoder. Every candle op is one encoder acquire+dispatch cycle.

### Syncs per forward pass

The only forced GPU-to-CPU syncs in the decode hot path are:
- `MoeBlock::forward_loop`: 2x `to_vec2()` per layer (60 total) -- eliminated by fused mode
- `sampler`: 1x `argmax().to_scalar()` or `to_vec1()` per token (greedy batched: 1 per 4 tokens)

All other operations stay lazy in the command pool until a sync is needed.

---

## 6. Dead Code and Vestigial Artifacts

### `device` field on `Gemma4Model` -- DEAD

The struct field `device: Device` at `gemma4.rs:1830` is populated at load
time (`gemma4.rs:2268`) but `self.device` is never referenced anywhere in
the file. All device access is done via `tensor.device()` inline. The field
is dead weight.

### `RopeVariant::Norm` -- NEARLY DEAD

Defined in `rope_kernel.rs:498`. The pipeline for it (`pipe_norm_f32`) is
compiled at load time and dispatch code exists at `rope_kernel.rs:803`. A
test uses it at `rope_kernel.rs:1344`. But Gemma 4 always uses
`RopeVariant::Neox` (`gemma4.rs:532`). The `Norm` variant exists for
future model families using GPT-J interleaved rotation. It is compiled and
tested but never exercised in production inference.

### `use super::moe_kernel` -- correctly cfg-gated

The import at `gemma4.rs:17` is `#[cfg(feature = "metal")]`. Without Metal,
the `moe_kernel` module is not compiled (`mod.rs:7` has
`#[cfg(feature = "metal")] pub mod moe_kernel`), so the import is also
correctly gated. There is no unused-import warning here.

### `KvCache::reset()` -- DEAD

At `gemma4.rs:878`, marked `#[allow(dead_code)]`. Called from
`Attention::clear_cache()` (also dead_code) and `DecoderLayer::clear_cache()`
(also dead_code). The actual cache clearing goes through
`Gemma4Model::clear_kv_cache()` which calls `layer.clear_cache()`, but that
method IS reachable from `cmd_generate` (warmup and benchmark loops), so the
`#[allow(dead_code)]` annotations on `clear_cache` are probably stale lints
rather than truly dead code. `reset()` on `KvCache` is reachable.

### `Gemma4Model::load` and `load_with_moe_mode` -- DEAD

Both at `gemma4.rs:1855` and `1872`, marked `#[allow(dead_code)]`. They are
backward-compat wrappers that default all modes to Loop. The live path uses
`load_with_modes` directly. Kept for tests that predate the mode flags.

### `has_nan` / `nan_check` -- DEAD

Debug helpers at `gemma4.rs:2366-2383`, marked `#[allow(dead_code)]`.

### `GgufModel::has_tensor` / `device()` -- DEAD

In `gguf_loader.rs:89,95`, marked `#[allow(dead_code)]`.

### Why the non-metal build does not work

The `rms_norm_kernel` and `rope_kernel` modules are NOT behind
`#[cfg(feature = "metal")]` in `serve/mod.rs` (lines 8-9). Both
unconditionally import `candle_metal_kernels` and `objc2_metal`, which are
optional deps available only under `--features metal`. Without that feature,
compilation fails with unresolved import errors. The `lm_head_kernel` module
compiles without Metal but would fail at runtime (candle CPU does not support
F16 matmul).

**The non-metal build is effectively broken.** The only way to compile and
run inference is `cargo build --features metal`.

---

## 7. Dependency Map

### Crate relationships

```
hf2q (binary)
  |
  +-- candle-core 0.10 (tensor ops, Metal backend)
  |     \-- [with "metal" feature: Metal device, MetalStorage, command pool]
  |
  +-- candle-nn 0.10 (VENDORED at vendor/candle-nn/)
  |     +-- Embedding, ops::sdpa, ops::softmax_last_dim, Activation
  |     \-- Vendor patch: SDPA Metal offset fix (element count -> bytes)
  |
  +-- candle-metal-kernels 0.10 (VENDORED at vendor/candle-metal-kernels/)
  |     +-- Pre-compiled Metal kernels: quantized matmul, SDPA, softmax
  |     +-- Rust-side dispatch wrappers
  |     +-- Commands pool (shared encoder management)
  |     \-- Vendor patch: DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER 50 -> 100
  |
  +-- objc2-metal 0.3 (optional, Metal framework bindings)
  +-- tokenizers 0.22 (HuggingFace tokenizer)
  \-- minijinja 2 (Jinja2 chat template rendering)
```

### What is vendored and why

**`vendor/candle-nn/`:** Patches the SDPA Metal dispatch to multiply
`start_offset` by `dtype.size_in_bytes()`. Upstream candle passes an element
count where Metal's `setBuffer:offset:atIndex:` expects bytes. The bug is
latent when `start_offset == 0` (all other candle consumers). hf2q's
in-place KV cache (1bNEW.20) creates views with non-zero `start_offset`
past `current_len > sliding_window`, exposing the bug as coherence failure
at ~token 1024.

**`vendor/candle-metal-kernels/`:** Bumps the command buffer recycle threshold
from 50 to 100 dispatches per buffer. Empirically measured +0.9 tok/s on
M5 Max. The upstream default of 50 has no empirical rationale.

### Runtime-compiled vs pre-compiled kernels

| Kernel | Source | Compiled when |
|--------|--------|---------------|
| `kernel_mul_mv_id_q6_K_f32` etc. | Pre-compiled in candle-metal-kernels | At candle crate build |
| `kernel_rms_norm_*` | Runtime MSL in `rms_norm_kernel.rs` | At model load (`Fused` mode) |
| `kernel_rope_neox_f32` / `kernel_rope_norm_f32` | Runtime MSL in `rope_kernel.rs` | At model load (`Fused` mode) |
| `gemm_*_hgemm` (MLX GEMM) | Pre-compiled in candle-metal-kernels | At candle crate build |
| SDPA vector/full kernels | Pre-compiled in candle-metal-kernels | At candle crate build |

---

## 8. Build Configurations

### What builds and runs

| Configuration | Compiles? | Runs inference? |
|---------------|-----------|-----------------|
| `cargo build --features metal` | Yes | Yes |
| `cargo build` (no features) | **No** | N/A |
| `cargo build --features cuda` | **No** | N/A |
| `cargo build --features metal,cuda` | Untested | UNCLEAR |

The default featureless build fails because `rms_norm_kernel.rs` and
`rope_kernel.rs` unconditionally import Metal-only optional deps. CUDA is
declared in Cargo.toml features but has no implementation -- `select_device`
would return a CUDA device, but every fused kernel module assumes Metal APIs.

### The only working build command

```
cargo build --release --features metal
```

This is the only configuration that has ever been tested or is expected to
work. The codebase is de facto Metal-only despite the presence of `cuda` and
CPU feature flags.
