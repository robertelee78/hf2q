# Spike: llama.cpp Metal Dispatch Pattern

Source: `/opt/llama.cpp/ggml/src/ggml-metal/` (local checkout, 2026-04-11)

## 1. The llama.cpp dispatch pipeline (step by step)

What happens from "forward pass starts" to "forward pass ends":

1. **`ggml_metal_graph_compute(ctx, gf)`** is called with a compute graph `gf` containing all ops for one forward pass.
   (`ggml-metal-context.m:441`)

2. **Decide node split.** The first `n_nodes_0 = MIN(MAX(64, 0.1*gf->n_nodes), gf->n_nodes)` nodes are assigned to the main thread. The remaining `n_nodes_1` are split across `n_cb` extra threads (typically 1-2).
   (`ggml-metal-context.m:467-488`)

3. **Get the single global command queue** from the device (one queue per device, shared across all backends).
   (`ggml-metal-context.m:529`, device struct at `ggml-metal-device.m:964`)

4. **Main thread: create command buffer, enqueue, encode, commit.**
   - Create one `MTLCommandBuffer` via `commandBufferWithUnretainedReferences`.
   - Call `[cmd_buf enqueue]` to reserve ordering in the queue.
   - Call `ctx->encode_async(n_cb)` which does all encoding for nodes `[0, n_nodes_0)`.
   - The encode_async block creates a **single `ggml_metal_encoder_t`** (wrapping one `MTLComputeCommandEncoder`) for all nodes in its range, then loops calling `ggml_metal_op_encode()` for each node.
   - After all nodes are encoded, `[cmd_buf commit]` is called.
   (`ggml-metal-context.m:533-545, 705-750`)

5. **Worker threads: create their own command buffers, enqueue, encode, commit.**
   - For each of `n_cb` threads (typically 1-2), a separate `MTLCommandBuffer` is created and enqueued.
   - `dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async)` runs the encoding concurrently on a GCD concurrent queue.
   - Each thread gets its own `ggml_metal_op_t` with its own encoder wrapping its own command buffer.
   - Each thread commits its command buffer when done.
   (`ggml-metal-context.m:552-573`)

6. **Return immediately.** `ggml_metal_graph_compute` does NOT wait for GPU completion. It returns `GGML_STATUS_SUCCESS` as soon as encoding is done. The GPU executes asynchronously.
   (`ggml-metal-context.m:574-575` — the `waitUntilCompleted` line is commented out)

7. **Synchronize later.** `ggml_metal_synchronize()` is called only when the CPU needs results (e.g., reading back logits). It calls `[cmd_buf_last waitUntilCompleted]` — a single wait on the last committed command buffer.
   (`ggml-metal-context.m:242-298`)

**Total sync points per forward pass: 0 during encoding, 1 when results are needed.**

## 2. Buffer allocation pattern

### Weight buffers
- Allocated once at model load via `ggml_metal_buffer_init()` or `ggml_metal_buffer_map()`.
- Weights are stored in `MTLResourceStorageModeShared` (host-visible) buffers when mmap'd, or `MTLResourceStorageModePrivate` (GPU-only) when not shared.
- Multiple MTLBuffers may back a single logical buffer (to work around the per-buffer size limit when using mmap).
  (`ggml-metal-device.m:1728-1738, 1848-1916, 1918+`)

### Activation / scratch buffers
- Same allocation path. ggml's graph allocator pre-plans all activation tensors into a contiguous buffer. The Metal backend allocates this once.
- There is NO per-op buffer allocation. All tensors in a graph are views into pre-allocated MTLBuffers.

### KV cache buffers
- Same as weight buffers — allocated once, persisted across forward passes.

### Buffer binding at dispatch time
- Each op calls `ggml_metal_get_buffer_id(tensor)` which returns `{id<MTLBuffer>, offset}` by looking up the tensor's backing buffer.
- The encoder sets the buffer+offset on the pipeline: `[encoder->obj setBuffer:buffer.metal offset:buffer.offs atIndex:idx]`.
- This is a trivial pointer lookup, not an allocation.
  (`ggml-metal-device.m:930-931, 2185+`, `ggml-metal-ops.cpp:16-26`)

### Residency sets
- On macOS 15+, all buffers are grouped into `MTLResidencySet`s. A background thread periodically calls `requestResidency` to keep buffers GPU-resident.
- `ggml_metal_device_rsets_keep_alive()` is called at the start of each `graph_compute` to reset the keep-alive timer.
  (`ggml-metal-device.m:980-1048, ggml-metal-context.m:473`)

## 3. Encoder / command buffer lifecycle

### Per forward pass (typical n_cb=2):
- **3 command buffers** total: 1 for main thread (cmd_bufs[n_cb]), 2 for worker threads (cmd_bufs[0..n_cb)).
- **3 compute command encoders** total: 1 per command buffer.
- Each encoder is created once at the start of the encode_async block via `ggml_metal_encoder_init()`, which calls `[cmd_buf computeCommandEncoder]` or `[cmd_buf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]`.
  (`ggml-metal-device.m:820-858`)
- The encoder is reused for ALL ops in that command buffer's range. The per-op function simply calls `ggml_metal_encoder_set_pipeline`, `set_bytes`, `set_buffer`, `dispatch_threadgroups` — all thin wrappers around the retained encoder object.
  (`ggml-metal-ops.cpp:2135-2141` for a mul_mat example)
- `endEncoding` is called once, at `ggml_metal_op_free()` time (the `~ggml_metal_op` destructor calls `ggml_metal_encoder_end_encoding`).
  (`ggml-metal-ops.cpp:66-68`)
- Each command buffer is committed once, immediately after encoding all its nodes.
  (`ggml-metal-context.m:748`)

### Concurrency within an encoder
- When `use_concurrency` is true, the encoder is created with `MTLDispatchTypeConcurrent`. This lets Metal overlap independent dispatches within the same encoder.
- Memory barriers (`memoryBarrierWithScope:MTLBarrierScopeBuffers`) are inserted between ops that have overlapping read/write ranges.
  (`ggml-metal-device.m:942-943`, `ggml-metal-ops.cpp:147-157, 210-226`)

### Timeline:
```
Main thread:  [create CB] [enqueue] [encode N nodes] [commit]
Worker 0:     [create CB] [enqueue] [encode M nodes] [commit]
Worker 1:     [create CB] [enqueue] [encode K nodes] [commit]
GPU:          [execute CB_main] [execute CB_0] [execute CB_1]
                                                        ^-- one waitUntilCompleted here, only when CPU reads results
```

## 4. Pipeline (PSO) management

### Compilation: lazy, cached in a hashmap
- Pipelines are compiled on first use via `ggml_metal_library_compile_pipeline()`.
  (`ggml-metal-device.m:375-462`)
- The first call for a given kernel name:
  1. Looks up the name in `lib->pipelines` (an `std::unordered_map<std::string, ggml_metal_pipeline_t>`). Cache miss.
  2. Gets the MTLFunction from the library: `[lib->obj newFunctionWithName:...]` (with optional constant values for specialization).
  3. Compiles the PSO: `[lib->device newComputePipelineStateWithFunction:mtl_function error:&error]`.
  4. Stores the PSO in the hashmap: `ggml_metal_pipelines_add(lib->pipelines, name, res.pipeline)`.
  (`ggml-metal-device.cpp:28-30` for the hashmap structure)
- Subsequent calls hit the cache and return immediately.
  (`ggml-metal-device.m:386-393`)
- The compile is behind a lock (`lib->lock`) for thread safety.

### At dispatch time
- The per-op encoder functions call `ggml_metal_library_get_pipeline_*()` which returns the cached PSO.
- Then `ggml_metal_encoder_set_pipeline(enc, pipeline)` calls `[encoder->obj setComputePipelineState:pipeline.pipeline->obj]`.
- This is a single ObjC message send per op — negligible cost.

## 5. What candle does differently (the ~120 sync points)

### Candle's architecture: per-op encoder creation
- Candle's `Commands` struct (`candle-metal-kernels/src/metal/commands.rs`) maintains a pool of command buffers (default 5).
- Every time an op needs a compute encoder, it calls `commands.command_encoder()` which:
  1. Selects a pool entry (a command buffer with a semaphore).
  2. Increments `compute_count` on that entry.
  3. If `compute_count >= compute_per_buffer` (default 100, was 50), it **commits the current buffer** and **creates a new one** (`commit_swap_locked`, `commands.rs:277-289`).
  4. Creates a NEW `MTLComputeCommandEncoder` from the (possibly new) command buffer.
- Every op creates and destroys its own encoder. The `ComputeCommandEncoder::drop()` calls `end_encoding()`.
  (`encoder.rs:183-187`)

### Where the sync points come from
- A Gemma-3 forward pass has ~120 GPU ops. With `compute_per_buffer=100`, that is ~1-2 command buffer commits during the forward pass itself.
- But the critical sync happens at **`flush_and_wait()`** (`commands.rs:220-255`):
  - Called every time the CPU needs to read a tensor (e.g., `Tensor::to_scalar`, `to_vec`).
  - Iterates ALL pool entries, commits any with pending work, then calls `waitUntilCompleted` on every in-flight buffer.
  - This is a **full GPU pipeline drain** — not selective.

### The per-op encoder overhead
- Even within a single command buffer, candle creates and destroys ~100 encoders. Each `computeCommandEncoder()` / `endEncoding()` pair has non-trivial ObjC overhead (autorelease pool churn, state validation).
- llama.cpp creates 1 encoder per command buffer (3 total). Candle creates ~120.

### Structural comparison:

| Aspect | llama.cpp | candle/hf2q |
|--------|-----------|-------------|
| Encoders per forward | 3 (1 per CB) | ~120 (1 per op) |
| Command buffers per forward | 3 | 1-2 (pool rotation) |
| Commits per forward | 3 (fire-and-forget) | 1-2 |
| GPU waits per forward | 0 (deferred to sync) | 0 during forward, 1 full drain on read |
| Encoder creation cost | 3 ObjC calls | ~120 ObjC calls |
| Concurrency within encoder | MTLDispatchTypeConcurrent + barriers | None (serial encoders) |
| Buffer allocation per op | None (pre-allocated views) | None (pre-allocated views) |

### Why candle's pattern is slower
1. **Encoder churn**: 120 create/destroy cycles vs 3. Each cycle is an ObjC message send pair plus Metal driver state management.
2. **No concurrent dispatch**: llama.cpp marks its encoder as concurrent and uses memory barriers to let independent ops overlap. Candle uses serial encoders — each op must complete before the next starts.
3. **Full-drain sync**: When the CPU reads results, candle drains the entire pool. llama.cpp waits on only the last command buffer.

## 6. The minimal set of things to port

To replicate llama.cpp's dispatch pattern in mlx-native (or as a replacement for candle's Metal dispatch layer):

### Essential (must have)

1. **Graph-level compute function.** Accept a pre-built list of ops with their tensor arguments. This is the entry point that replaces per-op encoder creation.
   - Signature: `fn graph_compute(graph: &ComputeGraph) -> Result<()>`

2. **Single encoder per command buffer.** Create ONE `MTLComputeCommandEncoder` per command buffer. Reuse it for all ops in that buffer's range. Call `endEncoding` only when done with all ops.
   - This eliminates ~117 encoder create/destroy cycles.

3. **Pre-allocated buffer views.** Keep the current model: tensors are views into pre-allocated MTLBuffers. The per-op dispatch just looks up `(buffer, offset)`. No change needed here — candle already does this correctly.

4. **Deferred sync / fire-and-forget commit.** `commit` the command buffer immediately after encoding is done. Do NOT call `waitUntilCompleted`. Only wait when the CPU actually needs to read results.
   - Requires a "last committed command buffer" reference to wait on later.

5. **Pipeline (PSO) cache.** A hashmap of `String -> MTLComputePipelineState`. Compile on first use, cache forever. llama.cpp uses `std::unordered_map`; Rust equivalent is `HashMap<String, ComputePipeline>`.
   - Candle likely already has this; verify it is not recompiling.

### Important (significant speedup)

6. **Concurrent dispatch type with memory barriers.** Create the encoder with `MTLDispatchTypeConcurrent`. Before each op, check if its read/write ranges overlap with any previously encoded op. If they do, insert `memoryBarrierWithScope:MTLBarrierScopeBuffers` and reset the tracking. Otherwise, encode the op without a barrier — Metal will overlap it with previous ops.
   - This is llama.cpp's `ggml_mem_ranges` system (`ggml-metal-ops.cpp:147-173, 210-226`).
   - For a LLM decoder pass, many elementwise ops (RMSNorm, RoPE, SiLU) are independent and can overlap.

7. **Multi-CB parallel encoding (optional).** Split the graph across 2-3 command buffers encoded on separate threads. The main thread encodes the first ~10% of nodes and commits immediately so the GPU can start while workers encode the rest.
   - This is llama.cpp's `n_cb` system. On M-series, optimal `n_cb` is 1-2.
   - Less important than items 1-5 because the GPU is the bottleneck during decode, not the CPU encoder.

### Can skip

- **Residency sets.** Only relevant on macOS 15+ for very large models. Not needed for initial parity.
- **Op fusion.** llama.cpp's `use_fusion` flag and `ggml_can_fuse_ext` system. This is an optimization on top of the dispatch pattern, not the pattern itself.
- **Graph optimization.** `ggml_graph_optimize()` reshuffles ops. Candle's graph is already in a reasonable order.
- **Abort callback.** Only needed for interactive cancellation.

### Concrete types/functions to implement

```rust
/// The graph-level dispatch controller.
struct MetalDispatch {
    device: Device,
    queue: CommandQueue,
    pipeline_cache: HashMap<String, ComputePipeline>,
    last_committed: Option<CommandBuffer>,
}

impl MetalDispatch {
    /// Encode and commit an entire forward pass. Does NOT wait.
    fn graph_compute(&mut self, ops: &[GraphOp]) -> Result<()>;

    /// Wait for the last committed command buffer to finish.
    fn synchronize(&mut self) -> Result<()>;
}

/// Per-command-buffer encoding context. Created at the start of
/// graph_compute, destroyed (with endEncoding) at the end.
struct EncoderContext {
    cmd_buf: CommandBuffer,
    encoder: ComputeCommandEncoder,  // ONE for all ops
    mem_ranges: MemoryRangeTracker,  // for concurrent dispatch barriers
}

/// Memory range tracker for concurrent dispatch.
/// Tracks read/write ranges of all ops encoded so far.
/// Returns true if a new op can run concurrently, false if a barrier is needed.
struct MemoryRangeTracker { ... }
```

### Migration path

The cleanest approach: build `MetalDispatch` as a new entry point in mlx-native. The existing candle `Commands` + per-op encoder path stays for correctness verification. Once `MetalDispatch::graph_compute` matches candle's output, switch inference to use it.

This avoids modifying candle's internals (which we don't own) and puts all new GPU dispatch code in mlx-native (which we do own).
