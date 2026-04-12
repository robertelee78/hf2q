# Target Inference Architecture: hf2q on mlx-native

**Date:** 2026-04-11
**Governing ADR:** ADR-006 (mlx-native as hf2q's GPU Compute Backend)
**Phase 0 Verdict:** Framework-overhead-dominated (86 vs 105 tok/s gap is GPU pipelining, not kernel speed)
**Target:** >= 102 tok/s decode, Gemma 4 26B MoE DWQ, M5 Max, coherence preserved

---

## 1. Design Principles

These are load-bearing constraints. Every design decision below must satisfy all of them simultaneously.

**P1. Coherence preservation.** The migration must not regress the sourdough gate (common-byte-prefix >= 3094) or the byte-identical 16-token greedy gen vs llama.cpp at T=0. Coherence > speed always. Every op cutover gets a sourdough gate run before the next op starts.

**P2. Single commit_and_wait per forward pass.** The Phase 0 diagnosis proved the speed gap is GPU pipelining loss from candle's per-dispatch encoder model. The target architecture must encode all ops in a forward pass into shared command buffer(s) and commit once at the end. This is the primary speed lever (~1500-2000 us/token recovery).

**P3. No candle dependency post-migration.** After Phase 5 completes, `candle-core`, `candle-nn`, and `candle-metal-kernels` are removed from `Cargo.toml`. The `vendor/candle-nn/` and `vendor/candle-metal-kernels/` patches are deleted. Zero vendor patch maintenance going forward.

**P4. Weight buffers stay GPU-resident.** Quantized weight `MlxBuffer`s are allocated once at model load and never copied or reallocated during inference. Activation buffers are transient (arena-allocated per forward pass, released after commit).

**P5. Owned infrastructure.** mlx-native is Robert's repo. Every kernel, every framework pattern, every dispatch path lives in code we control. No upstream coordination for fixes.

**P6. Graph-scheduled dispatch.** The forward pass is expressed as a sequence of ops that mlx-native's `CommandEncoder` batches into a single (or small number of) Metal command buffer(s). The caller does not manage individual compute encoders.

---

## 2. Module Structure

### What gets deleted

| Current module | Why it goes away |
|---|---|
| `vendor/candle-nn/` (SDPA byte-offset patch) | mlx-native owns its own SDPA kernel; no upstream to patch |
| `vendor/candle-metal-kernels/` (compute_per_buffer patch) | mlx-native's `CommandEncoder` owns its own lifecycle |
| `src/serve/rms_norm_kernel.rs` (runtime MSL compile + candle encoder dispatch) | Replaced by mlx-native's `rms_norm_*` kernels dispatched through `CommandEncoder::encode_threadgroups_with_shared` |
| `src/serve/rope_kernel.rs` (runtime MSL compile + candle encoder dispatch) | Replaced by mlx-native's `rope_*` kernels |
| `src/serve/moe_kernel.rs` (candle `call_quantized_matmul_mv_id_t`) | Replaced by `mlx_native::quantized_matmul_id` |
| `src/serve/lm_head_kernel.rs` (candle F16 gemm path) | Replaced by mlx-native F16 GEMM op |
| `Cargo.toml` entries for `candle-core`, `candle-nn`, `candle-metal-kernels`, `objc2-metal` | Replaced by single `mlx-native = { path = "/opt/mlx-native" }` dependency |
| `[patch.crates-io]` section in `Cargo.toml` | No vendor patches needed |
| All `#[cfg(feature = "metal")]` guards around kernel mode dispatch | mlx-native is always-Metal; no dual-backend gates |
| `DispatchCounters` atomic increment sites throughout forward path | Replaced by mlx-native's global `dispatch_count()` / `sync_count()` counters |

### What gets added

| New module | Purpose |
|---|---|
| `src/serve/gpu.rs` | Integration layer: `GpuContext` struct holding `MlxDevice`, `KernelRegistry`, `MlxBufferPool`, and weight buffers. Single owner of all GPU state. |
| `src/serve/graph.rs` | Forward pass graph builder: `ForwardGraph` that records ops as a sequence of `GpuOp` enums, then executes them through a single `CommandEncoder`. |
| `src/serve/weight_loader.rs` | GGUF-to-MlxBuffer loader: reads quantized weights from the existing `GgufModel` and populates `MlxBuffer` instances with the correct quantization metadata. |

### What stays the same

| Module | Why unchanged |
|---|---|
| `src/serve/config.rs` | Model config parsing is backend-agnostic |
| `src/serve/gguf_loader.rs` | GGUF file reading is backend-agnostic (weight bytes are the same; only the destination changes from candle `QTensor` to `MlxBuffer`) |
| `src/serve/sampler.rs` | Sampling logic stays the same; input changes from `candle::Tensor` to an `MlxBuffer` read via `as_slice::<f32>()` |
| `src/cli.rs` | CLI flags stay (the `--moe-kernel`, `--rms-norm-kernel` etc. flags become no-ops or are removed; the `--backend mlx-native` flag already exists from Phase 5) |
| `src/serve/mod.rs` | HTTP server, generate loop, token handling |
| `scripts/sourdough_gate.sh` | Coherence validation |

---

## 3. Forward Pass (target hot path)

### 3.1 Current architecture (candle, what we are leaving)

```
Gemma4Model::forward(input_ids, seqlen_offset)
  for each of ~120 candle ops per layer x 26 layers:
    candle op -> MetalDevice::command_encoder() -> new compute encoder
    -> set pipeline, set buffers, dispatch, end encoding
    -> (every 50-100 dispatches: implicit commit + waitUntilCompleted)
  Total: ~2104 dispatches, ~120 GPU sync points per forward pass
```

Each candle op creates its own compute command encoder, dispatches one kernel, ends the encoder. Candle's `Commands` pool commits the command buffer after every `DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER` dispatches (currently 100 via vendor patch, upstream 50). This prevents GPU pipelining -- the GPU cannot overlap execution of consecutive kernels because each encoder boundary is a serialization point.

### 3.2 Target architecture (mlx-native)

```
Gemma4Model::forward(input_ids, seqlen_offset)
  let mut encoder = gpu_ctx.device.command_encoder()?;
  // Embed
  gpu_ops::embedding_gather(&mut encoder, &registry, &embed_buf, input_ids, &mut act_pool);
  gpu_ops::scalar_mul(&mut encoder, &registry, &hidden_buf, scale);
  // For each layer:
    gpu_ops::rms_norm(&mut encoder, ...);        // input_layernorm
    gpu_ops::qmatmul(&mut encoder, ...);         // q_proj
    gpu_ops::qmatmul(&mut encoder, ...);         // k_proj
    gpu_ops::qmatmul(&mut encoder, ...);         // v_proj
    gpu_ops::rms_norm(&mut encoder, ...);         // q_norm
    gpu_ops::rms_norm(&mut encoder, ...);         // k_norm
    gpu_ops::rms_norm_unit(&mut encoder, ...);    // v_norm
    gpu_ops::rope_neox(&mut encoder, ...);        // q RoPE
    gpu_ops::rope_neox(&mut encoder, ...);        // k RoPE
    gpu_ops::kv_cache_write(&mut encoder, ...);   // in-place KV append
    gpu_ops::sdpa_vector(&mut encoder, ...);      // attention (decode)
    gpu_ops::qmatmul(&mut encoder, ...);         // o_proj
    gpu_ops::rms_norm(&mut encoder, ...);         // post_attention_layernorm
    gpu_ops::add(&mut encoder, ...);              // residual
    gpu_ops::rms_norm(&mut encoder, ...);         // pre_feedforward_layernorm
    // Dense MLP branch
    gpu_ops::qmatmul(&mut encoder, ...);         // gate_proj
    gpu_ops::gelu(&mut encoder, ...);
    gpu_ops::qmatmul(&mut encoder, ...);         // up_proj
    gpu_ops::mul(&mut encoder, ...);              // gate * up
    gpu_ops::qmatmul(&mut encoder, ...);         // down_proj
    gpu_ops::rms_norm(&mut encoder, ...);         // post_feedforward_layernorm_1
    // MoE branch
    gpu_ops::rms_norm(&mut encoder, ...);         // pre_feedforward_layernorm_2
    gpu_ops::rms_norm_unit(&mut encoder, ...);    // router norm
    gpu_ops::qmatmul(&mut encoder, ...);         // router_proj
    gpu_ops::softmax(&mut encoder, ...);
    gpu_ops::argsort(&mut encoder, ...);
    gpu_ops::gather(&mut encoder, ...);           // top_k_probs
    gpu_ops::qmatmul_id(&mut encoder, ...);      // fused gate_up
    gpu_ops::gelu(&mut encoder, ...);
    gpu_ops::mul(&mut encoder, ...);              // SwiGLU
    gpu_ops::qmatmul_id(&mut encoder, ...);      // fused down
    gpu_ops::gather(&mut encoder, ...);           // per_expert_scale
    gpu_ops::mul(&mut encoder, ...);              // weight * scale
    gpu_ops::weighted_sum(&mut encoder, ...);     // sum over top_k
    gpu_ops::rms_norm(&mut encoder, ...);         // post_feedforward_layernorm_2
    // Combine
    gpu_ops::add(&mut encoder, ...);              // mlp + moe
    gpu_ops::rms_norm_add(&mut encoder, ...);     // post_feedforward_layernorm + residual
    gpu_ops::mul(&mut encoder, ...);              // layer_scalar
  // Final
  gpu_ops::rms_norm(&mut encoder, ...);           // final norm
  gpu_ops::f16_gemm(&mut encoder, ...);           // lm_head
  gpu_ops::softcap(&mut encoder, ...);            // logit softcapping
  // SINGLE SYNC POINT
  encoder.commit_and_wait()?;
  // Read logits from output MlxBuffer
  let logits: &[f32] = output_buf.as_slice()?;
```

Key differences from the current architecture:

1. **One `CommandEncoder` per forward pass.** All ops encode into the same command buffer. Metal's GPU can pipeline execution across all dispatches within the buffer.

2. **No intermediate GPU syncs.** The current MoE Loop path's `to_vec2()` calls (which force `waitUntilCompleted`) are eliminated -- the fused `qmatmul_id` kernel keeps routing indices on the GPU.

3. **Activation buffers from `MlxBufferPool`.** Each forward pass calls `pool.alloc()` for intermediate results and `pool.release()` after the commit. The pool's power-of-two bucketing reuses Metal allocations across tokens without per-token `newBuffer` calls.

4. **Weight buffers are pre-loaded `MlxBuffer` instances.** No per-dispatch buffer extraction from `candle::Tensor::storage_and_layout()`.

### 3.3 Buffer flow

```
Weight buffers (persistent, loaded once):
  embed_weight: MlxBuffer [vocab, hidden] quantized
  per-layer:
    q/k/v/o_proj weights: MlxBuffer [out, in] quantized
    gate/up/down_proj weights: MlxBuffer [out, in] quantized
    expert_gate_up_3d: MlxBuffer [num_experts, 2*intermediate, hidden] quantized
    expert_down_3d: MlxBuffer [num_experts, hidden, intermediate] quantized
    norm weights: MlxBuffer [hidden] f32
    layer_scalar: MlxBuffer [1] f32
  lm_head_f16: MlxBuffer [vocab, hidden] f16

Activation buffers (transient, from pool):
  hidden_state: MlxBuffer [1, seq_len, hidden] f32
  q/k/v projections: MlxBuffer [1, seq_len, heads*head_dim] f32
  attention output: MlxBuffer [1, seq_len, hidden] f32
  mlp intermediates: MlxBuffer [1, seq_len, intermediate] f32
  moe intermediates: MlxBuffer [tokens, top_k, 2*intermediate] f32
  logits: MlxBuffer [1, vocab] f32

KV cache buffers (persistent, pre-allocated):
  per-layer k_cache: MlxBuffer [1, kv_heads, max_cache, head_dim] f32
  per-layer v_cache: MlxBuffer [1, kv_heads, max_cache, head_dim] f32
```

---

## 4. Integration Layer

### 4.1 GpuContext

A single struct that owns all GPU state. Lives on `Gemma4Model`.

```rust
// src/serve/gpu.rs
pub struct GpuContext {
    pub device: MlxDevice,
    pub registry: KernelRegistry,  // pre-warmed at model load
    pub pool: MlxBufferPool<'static>,  // activation arena (lifetime tied to device)
}
```

`GpuContext` is created once at model load. The `KernelRegistry` pre-compiles all shader pipelines at init (not lazy) so no first-dispatch compile stall occurs during inference.

### 4.2 Type mapping

| Current (candle) | Target (mlx-native) | Notes |
|---|---|---|
| `candle_core::Tensor` | `MlxBuffer` | No shape/stride metadata on MlxBuffer; shapes tracked by the caller (Gemma4Model fields or local variables) |
| `candle_core::quantized::QMatMul` | `MlxBuffer` (raw quantized bytes) + `QuantizedMatmulParams` | Params struct carries M/K/N/group_size/bits |
| `candle_core::quantized::QStorage` | `MlxBuffer` | Direct 1:1 mapping for the raw quantized byte buffer |
| `candle_core::MetalDevice` | `MlxDevice` | 1:1 |
| `candle_core::DType` | `mlx_native::DType` | Enum mapping; both have F32/F16/BF16/U8/U32/I32 |
| `candle_metal_kernels::Kernels` | `mlx_native::KernelRegistry` | Same role: pipeline cache. KernelRegistry also holds MSL source. |

### 4.3 Does hf2q call mlx-native ops directly?

**Yes.** There is no additional wrapper crate between `gemma4.rs` and `mlx_native`. The integration layer is a module (`src/serve/gpu.rs`) containing:

- `GpuContext` (device + registry + pool)
- Free functions like `gpu_ops::qmatmul(encoder, registry, device, weight, input, output, params)` that are thin wrappers around `mlx_native::quantized_matmul()` with hf2q-specific shape conventions.

The reason for thin wrappers rather than raw `mlx_native` calls: hf2q's op call sites need to handle buffer pool allocation for outputs, shape validation, and the specific parameter conventions (e.g., Gemma 4's quantization types, MoE routing shapes). These are hf2q-specific concerns that do not belong in mlx-native itself.

### 4.4 Removing the candle dependency

The candle dependency is removed in Phase 5 step 5 (after all ops are validated on mlx-native). Concretely:

1. Delete `candle-core`, `candle-nn` from `[dependencies]` in `Cargo.toml`
2. Delete `candle-metal-kernels`, `objc2-metal` from `[dependencies]`
3. Delete `[patch.crates-io]` section entirely
4. Delete `vendor/candle-nn/` directory
5. Delete `vendor/candle-metal-kernels/` directory
6. Delete `src/serve/rms_norm_kernel.rs`, `rope_kernel.rs`, `moe_kernel.rs`, `lm_head_kernel.rs`
7. Remove all `use candle_core::*` and `use candle_nn::*` imports from `gemma4.rs`
8. Remove `candle_core::quantized::QMatMul` from `QLinear` (replace with `MlxBuffer` + params)
9. Remove `candle_nn::Embedding` (replace with a GPU embedding gather kernel)
10. Remove the `metal` feature flag (mlx-native is always Metal; non-Metal builds are not supported post-migration)

The `Embedding` struct is currently candle's `candle_nn::Embedding` which does a gather lookup. mlx-native already has `embedding_gather_4bit` and `embedding_gather_6bit` kernels registered in its `KernelRegistry`. For the F16 embedding (Gemma 4's `token_embd.weight`), an `embedding_gather_f16` kernel is needed, or the gather can be implemented as a memcpy slice from the weight buffer.

---

## 5. The Graph Scheduler

### 5.1 Compile-time vs runtime graph

**Runtime, rebuilt each forward pass -- but only the buffer binding changes, not the op sequence.**

For decode (seq_len=1, the hot path), the op sequence is identical every token. The structural question is whether to formalize this as a compiled graph (build once at model load, replay with different buffer pointers) or simply call the ops in sequence.

**Decision: start with sequential op calls through a shared `CommandEncoder`, not a formal graph IR.** Rationale:

- The Phase 0 data shows the speed gap is GPU pipelining, not CPU-side graph compilation overhead. Sequential calls through a shared encoder already capture the pipelining win.
- ggml-metal's `ggml_metal_graph_compute` does NOT use a compiled graph either -- it walks the `ggml_cgraph` node array and calls `ggml_metal_op_encode` for each node in sequence. The "graph" is just an array of tensor ops with dependency edges implicit in the tensor pointers.
- A formal `MlxGraph` / `MlxNode` IR (as sketched in the PRD) is a Phase 4 optimization if the sequential-call approach proves to have measurable CPU overhead. The PRD's graph scheduler can be added later without changing the op-call API.

### 5.2 What ggml-metal actually does (from reading the code)

`ggml_metal_graph_compute` in `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:441-750`:

1. Takes a `ggml_cgraph` (a flat array of `n_nodes` tensor operation nodes).
2. Splits nodes into `n_nodes_0` (main thread, ~max(64, 10% of nodes)) and `n_nodes_1` (remaining, split across `n_cb` extra command buffers, typically 1-2).
3. The main thread creates one command buffer, enqueues it, and calls `encode_async(n_cb)` which iterates its assigned node range calling `ggml_metal_op_encode(ctx_op, idx)` for each node. Each `ggml_metal_op_encode` creates a compute encoder within the same command buffer, dispatches the kernel, and ends the encoder.
4. The extra threads (if any) each get their own command buffer and encode their assigned node ranges in parallel via `dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async)`.
5. Each command buffer is committed after encoding all its nodes.
6. The function returns WITHOUT waiting -- the GPU runs asynchronously. The next `graph_compute` call for the next token implicitly waits on the previous command buffer via Metal's in-order queue guarantee.

Key insight: ggml-metal uses **multiple compute encoders within a single command buffer** (one per op), but commits the entire command buffer as one unit. Metal pipelines execution across encoders within the same command buffer. This is what candle cannot do -- candle creates a new command buffer every 50-100 dispatches and commits each one.

### 5.3 Target encoder lifecycle for mlx-native

```
Per forward pass:
  1. Create 1 CommandEncoder (wraps 1 MTLCommandBuffer)
  2. For each op in the forward pass:
     a. encoder.encode(pipeline, buffers, grid, threadgroup_size)
        -- internally: new_compute_command_encoder, set pipeline,
           set buffers, dispatch, end_encoding
  3. encoder.commit_and_wait()  -- single CPU-GPU sync point
```

mlx-native's `CommandEncoder::encode()` already follows this pattern: it creates a compute encoder per call, dispatches, and ends it. The key is that all `encode()` calls share the same `cmd_buf` (command buffer), and `commit_and_wait()` is called once at the end.

For the Gemma 4 forward pass (~2100 dispatches at decode), this means one command buffer with ~2100 compute encoder create/end cycles. Each cycle is a few hundred nanoseconds of CPU work, totaling ~0.5 ms -- well within budget.

### 5.4 Buffer allocation within the forward pass

Activation buffers are allocated from `MlxBufferPool` at the start of each op and released after the forward pass completes (after `commit_and_wait`). The pool's power-of-two bucketing means most allocations hit the free list after the first token.

Weight buffers and KV cache buffers are pre-allocated at model load and never touch the pool.

The total activation memory high-water mark for decode is small (~50 MB for Gemma 4's hidden_size=2816, intermediate=16384, vocab=262144). The pool pre-allocates this on the first token and reuses it for every subsequent token.

---

## 6. Migration Path

### 6.1 Per-op cutover order (from ADR-006 Phase 5)

The order follows ADR-006 Phase 3's sub-phase sequence. Each step adds a per-op CLI flag (`--<op>-backend candle|mlx-native`), validates with the sourdough gate, and records tok/s.

| Step | Op | Current module | mlx-native op | Risk |
|---|---|---|---|---|
| 1 | Quantized matmul (QLinear) | `candle_core::quantized::QMatMul` | `mlx_native::quantized_matmul` | Medium -- highest call count (~190 calls/token), must match candle's GGML block layout exactly |
| 2 | SDPA vector (decode attention) | `candle_nn::ops::sdpa` | `mlx_native` SDPA kernel | High -- the vendored byte-offset fix must be replicated; non-zero start_offset KV cache views |
| 3 | RoPE neox | `src/serve/rope_kernel.rs` | `mlx_native` rope kernels | Low -- already ported from llama.cpp; same MSL source |
| 4 | RmsNorm (F=1/2/3) | `src/serve/rms_norm_kernel.rs` | `mlx_native` rms_norm kernels | Low -- already ported from llama.cpp; same MSL source |
| 5 | F16 lm_head GEMM | `src/serve/lm_head_kernel.rs` | mlx-native F16 GEMM | Low -- uses candle's mlx-gemm internally anyway |
| 6 | KV cache in-place append | `candle_core::Tensor::slice_set` | `mlx_native` buffer copy op | Medium -- stride arithmetic must match exactly |
| 7 | Embedding, softmax, argsort, gelu, elementwise, softcap, transpose | Various candle ops | `mlx_native` elementwise/utility kernels | Low per-op, many ops |
| 8 | Fused MoE dispatch (`kernel_mul_mv_id`) | `src/serve/moe_kernel.rs` | `mlx_native::quantized_matmul_id` | Medium -- complex buffer layout, must match candle's vendored template |

### 6.2 Incremental vs big-bang

**Incremental per-op cutover** (ADR-006 Phase 5's design). Each op is swapped independently. During the transition, the forward pass mixes candle ops and mlx-native ops in the same `forward()` call. This requires a bridge: candle tensors and mlx-native buffers share the same underlying Metal buffer via `StorageModeShared` unified memory, so a candle `MetalStorage::buffer()` pointer and an `MlxBuffer::metal_buffer()` pointer can point to the same GPU memory without copies.

The bridge works because:
- Both candle and mlx-native use `StorageModeShared` Metal buffers
- Apple Silicon unified memory means the same pointer is valid for both frameworks
- No explicit host-device transfer is needed

**What requires big-bang:** the final removal of candle (Phase 5 step 5). Once the per-op flags all default to `mlx-native` and the sourdough gate passes, the candle code paths and dependency are deleted in a single commit. This is irreversible but safe because every op has been individually validated.

### 6.3 Risk points

1. **SDPA byte-offset bug.** The candle SDPA vendor patch (`vendor/candle-nn/src/ops.rs`) fixes a start_offset-in-elements vs bytes confusion. mlx-native's SDPA kernel must get this right from day one. The unit test MUST exercise a non-zero start_offset case (KV cache at current_len > sliding_window).

2. **Quantized weight layout.** candle's `QStorage::Metal` stores quantized blocks in GGML block order. mlx-native's `quantized_matmul` uses MLX's affine quantization (4-bit packed uint32 with bf16 scales/biases). These are DIFFERENT formats. The GGUF loader must produce `MlxBuffer` in whichever format the target kernel expects. For Phase 3 borrowed kernels (candle's vendored GGML kernels), the GGML block format is correct. For mlx-native's own kernels, a requantization step may be needed.

3. **MoE 3D weight buffer layout.** The fused `kernel_mul_mv_id` kernel expects a specific `[num_experts, n, k]` byte-contiguous layout. The current code builds this by concatenating per-expert quantized bytes in `MoeBlock` load. The same construction must produce an `MlxBuffer` with identical byte layout.

4. **Command buffer lifetime.** mlx-native's `CommandEncoder` wraps a single `CommandBuffer`. If the forward pass encodes ~2100 dispatches into one command buffer, Metal must handle this without hitting internal limits. ggml-metal uses 2-3 command buffers for ~2650 nodes. If 2100 dispatches in one buffer proves problematic, split into 2 buffers at a layer boundary (e.g., after layer 13).

---

## 7. What Gets Simpler

### Vendor patches eliminated

| Patch | Lines | Why it existed | Why it disappears |
|---|---|---|---|
| `vendor/candle-nn/src/ops.rs` SDPA byte-offset fix | ~50 LOC patch | candle upstream passes element offset where Metal expects byte offset | mlx-native's SDPA is written correct from the start |
| `vendor/candle-metal-kernels/src/metal/commands.rs` compute_per_buffer bump | ~5 LOC patch | candle's default 50 dispatches/buffer was suboptimal | mlx-native uses 1 command buffer per forward pass; no per-buffer threshold |

### Kernel mode flags eliminated

The current architecture has 5 kernel mode CLI flags, each with `Loop` / `Fused` variants:
- `--moe-kernel`
- `--rms-norm-kernel`
- `--rope-kernel`
- `--lm-head-kernel`
- `--kv-cache-kernel`

Post-migration, there is exactly one dispatch path per op. The flags are deleted. The code that checks `self.kernel.is_fused()` and branches between an 11-op manual chain and a single fused dispatch is replaced by a single mlx-native call.

### Per-dispatch encoder overhead eliminated

The current forward pass creates ~120 GPU sync points per token (candle's `Commands` pool commits every 100 dispatches). Post-migration: 1 sync point per token.

### DispatchCounters simplified

The current `DispatchCounters` struct tracks 7 different atomic counters with manual `fetch_add` calls at ~40 sites in `gemma4.rs`. Post-migration, mlx-native's global `dispatch_count()` and `sync_count()` provide the same observability with zero instrumentation code in hf2q.

### `#[cfg(feature = "metal")]` gates eliminated

The current code has `#[cfg(feature = "metal")]` around MoE fused state fields, the fused forward path, and kernel mode types. Post-migration, mlx-native is always-Metal. The `metal` feature flag in `Cargo.toml` is deleted. Non-Metal builds are not a supported configuration.

### Dependency tree shrinks

Current: `candle-core` (large, pulls accelerate/BLAS/CUDA stubs) + `candle-nn` + `candle-metal-kernels` + `objc2-metal`.
Target: `mlx-native` (one crate, ~2000 LOC, depends only on `metal` and `bytemuck`).

---

## 8. Open Questions

**Q1. GGML block format vs MLX affine format.** mlx-native's existing `quantized_matmul` uses MLX-style affine quantization (packed uint32 with bf16 scales/biases per group). candle's borrowed kernels (the `kernel_mul_mv_q*_f32` family) use GGML block format (block_q4_0, block_q6_K, etc.). Which format does the target architecture standardize on? Options:
- (a) Keep GGML block format, borrow candle's kernel MSL source into mlx-native. Simpler migration, but the kernels are an older llama.cpp snapshot.
- (b) Requantize GGUF weights to MLX affine format at load time. Uses mlx-native's own kernels, but adds a one-time conversion cost and must prove bitwise-equivalent dequantization.
- Decision needed before Phase 3 starts.

**Q2. Prefill path.** This document focuses on decode (seq_len=1, the speed-critical path). The prefill path (seq_len > 1) uses different SDPA kernels (full attention, not vector), different MoE dispatch shapes, and may benefit from batched matmul. Does the same single-encoder approach work for prefill, or does prefill need a different dispatch strategy? The Phase 0 data is decode-only.

**Q3. Multiple command buffers.** ggml-metal uses up to 3 command buffers dispatched across 1-2 threads via `dispatch_apply`. Is this pipelining beneficial on M5 Max, or does a single command buffer suffice? A microbench is needed: encode all 2100 dispatches into 1 vs 2 vs 3 command buffers and measure tok/s. Start with 1 (simplest); add multi-CB only if measured to help.

**Q4. Activation dtype.** The current forward pass is end-to-end F32 (`MODEL_DTYPE = DType::F32`). mlx-native's kernels support BF16 for several ops (rms_norm, elementwise, SDPA). Should the target architecture use BF16 activations for speed? Phase 0 data shows kernel time is not the bottleneck, so dtype changes would help only if they reduce memory bandwidth pressure. Defer to post-Walk (Run phase).

**Q5. KV cache buffer sharing.** Can the KV cache `MlxBuffer` be a view into a single large pre-allocated buffer (one per layer, or one for all layers)? Or does each layer need its own independent allocation? ggml-metal uses a single large buffer with computed offsets. The answer affects memory fragmentation and the `MlxBufferPool` design.

**Q6. `MlxBufferPool` lifetime.** The pool borrows `&'d MlxDevice`. For the activation arena pattern (pool lives on `Gemma4Model`, device also lives on `Gemma4Model`), Rust's borrow checker requires either self-referential struct tricks or an `Arc<MlxDevice>` so the pool can hold a shared reference. The current `MlxBufferPool<'d>` lifetime parameter may need to change to `MlxBufferPool` with an internal `Arc<MlxDevice>`.

**Q7. Fused kernel opportunities.** Post-migration, are there op fusion opportunities beyond what the current fused kernels provide? Candidates: fused `norm + qmatmul` (saves one buffer write), fused `gelu + mul` (SwiGLU), fused `softmax + argsort` (MoE routing). These are Run-phase optimizations, not Walk-phase, but the target architecture should not preclude them.

**Q8. Profiling infrastructure.** mlx-native's `dispatch_count()` and `sync_count()` are coarse. For future optimization, should mlx-native support per-kernel-name timing (via `MTLCounterSampleBuffer` or CPU-side `Instant::now()` around each `encode()` call)? This is observability infrastructure, not correctness-critical, but the decision affects the `CommandEncoder` API surface.
