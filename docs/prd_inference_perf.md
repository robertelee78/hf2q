# PRD: Inference Performance Optimization

**Status:** Draft
**Date:** 2026-04-07
**Goal:** Match or beat Ollama's MLX inference throughput for Gemma 4 on Apple Silicon

## 1. Problem Statement

Our pure-Rust Metal inference engine (hf2q + mlx-native) has significant performance gaps compared to Ollama's MLX runner. Profiling reveals that **kernel launch overhead (~5ms) exceeds actual GPU compute time (~3-4ms)** per decode step, meaning we spend more time dispatching work than doing work.

### Root Causes (ordered by impact)

| # | Bottleneck | Current State | Impact |
|---|-----------|---------------|--------|
| 1 | Excessive GPU sync points | ~305 `commit_and_wait()` per forward pass | ~2.4ms of pure sync overhead |
| 2 | MoE routing on CPU | 1.44 GB CPU memcpy + 10.8M CPU FLOPs per step | Forces GPU stall at every MoE layer |
| 3 | Kernel dispatch overhead | ~2,278 kernel dispatches per forward pass | ~4.6ms encoding overhead |
| 4 | Synchronous decode loop | CPU blocks GPU; GPU blocks CPU; no overlap | 0% CPU/GPU overlap |
| 5 | CPU-side sampling | Full vocab sort O(V log V) + 1MB logits readback | ~2ms CPU + 1MB transfer per token |
| 6 | Missing kernel fusions | bf16<->f32 casts, separate norm+proj, separate GELU+mul | ~500 unnecessary kernel launches |
| 7 | Embedding CPU round-trip | GPU->CPU->GPU for f32->bf16 cast | 1 unnecessary sync + 2 data copies |

### Reference: Ollama's Approach

Ollama achieves high throughput through:
- **Lazy evaluation + graph compilation** -- MLX JIT fuses entire op graphs into optimized Metal kernels
- **Fused fast API** -- SDPA, RMSNorm, RoPE each as single fused kernels
- **Async pipelined generation** -- GPU computes token N+1 while CPU processes token N
- **Fused quantized matmul** -- dequantize-on-the-fly, never materializes full-precision weights
- **Single default stream** -- all work serialized via lazy eval graph; zero explicit syncs during generation
- **Pin/Sweep memory** -- bulk free of intermediates instead of per-tensor deallocation

## 2. Phases

### Phase 1: Eliminate GPU Sync Points

**Objective:** Reduce `commit_and_wait()` calls from ~305 to ~93 per forward pass (3 per layer + overhead).

#### Current State (per MoE layer, GPU fast path)

| Sync | Line | Batched Ops | Eliminable? |
|------|------|-------------|-------------|
| L1 | 1097 | norm + QKV proj + head norms + RoPE (13 dispatches) | No -- KV cache needs results |
| L2 | 1287 | KV cache append (2 dispatches) | **Yes** -- merge with L3 |
| L3 | 2462 | Q/K/V permute + SDPA (4 dispatches) | **Yes** -- merge with L4 |
| L4 | 1388 | O proj + post-attn norm + residual (4 dispatches) | No -- FFN needs residual |
| L5a | 1428 | 2x pre-FFN norms (2 dispatches) | **Forced** by CPU MoE router |
| L6 | 3525 | dense FFN (8 dispatches) | **Yes** -- merge with L7 |
| L7-disp | 2862 | MoE expert dispatch (40 dispatches) | **Yes** -- merge with L7-acc |
| L7-acc | 2922 | MoE accumulation (9 dispatches) | No -- post-FFN needs results |
| L8 | 1465 | post-FFN norms + combine (4 dispatches) | **Yes** -- merge with L9 |
| L9 | 1564 | residual + scalar (2 dispatches) | **Yes** -- merge with next L1 |

#### Target: 3 syncs per layer

1. **Mega-encoder A**: norm + QKV proj + head norms + RoPE + KV append + SDPA + O proj + post-attn norm + residual (~23 dispatches)
2. **Mega-encoder B**: pre-FFN norms + dense FFN + MoE dispatch + MoE accum (~59 dispatches)
3. **Mega-encoder C**: post-FFN norms + combine + residual + scalar (~6 dispatches, merge with next layer's start)

#### Changes Required

| File | Change |
|------|--------|
| `gemma4.rs:forward_layer()` | Restructure to use 3 command encoders instead of 10 |
| `gemma4.rs:compute_attention_with_encoder()` | Accept external encoder instead of creating its own |
| `gemma4.rs:dense_ffn_with_encoder()` | Accept external encoder, remove internal commit |
| `encoder.rs` | Add `commit()` (non-blocking) alongside `commit_and_wait()` |

#### Acceptance Criteria

- [ ] **AC-1.1**: `commit_and_wait()` count per forward pass (decode, seq_len=1) <= 100, measured by atomic counter in CommandEncoder
- [ ] **AC-1.2**: `commit_and_wait()` count per forward pass (decode, seq_len=1) <= 93 (stretch goal: 3 per layer x 30 + 3 overhead)
- [ ] **AC-1.3**: No correctness regression -- output logits match pre-optimization baseline within bf16 epsilon (< 1e-3 max absolute diff)
- [ ] **AC-1.4**: All existing tests pass (`cargo test --features mlx-native`)

---

### Phase 2: GPU-Native MoE Routing

**Objective:** Eliminate all CPU computation and GPU<->CPU transfers from the MoE routing hot path.

#### Current State (per MoE layer, decode)

| Step | Location | CPU/GPU | Data Size | Problem |
|------|----------|---------|-----------|---------|
| Read router_input | gemma4.rs:2710 | GPU->CPU | 11.3 KB | Blocks GPU |
| Read expert_input | gemma4.rs:2711 | GPU->CPU | 11.3 KB | Blocks GPU |
| CPU RMS norm | gemma4.rs:2775 | CPU | 2816 elements | Should be GPU |
| CPU router matmul | gemma4.rs:2778-2784 | CPU | 128x2816 dot products | 360K FLOPs on CPU |
| CPU top-K + softmax | gemma4.rs:2799 | CPU | 128-element sort | Should be GPU |
| Expert weight extraction | gemma4.rs:3379-3394 | CPU memcpy | ~6 MB per expert | 48 MB/layer total |
| Input upload per expert | gemma4.rs:3269 | CPU->GPU | 11.3 KB x 8 | Avoidable |

**Per step totals across 30 layers:** 660 KB GPU->CPU readback, 10.8M CPU FLOPs, 1.44 GB CPU memcpy, 1.44 GB GPU buffer allocations.

#### Target Architecture

1. **Router projection cache on GPU**: Convert `router_proj_cache` from `HashMap<usize, Vec<f32>>` to `HashMap<usize, MlxBuffer>` holding [128, 2816] f32 GPU buffers
2. **GPU router kernel**: RMS norm + matvec against cached router weights -> [seq_len, 128] logits (compose existing `rms_norm` + `quantized_matmul_simd`)
3. **GPU top-K + softmax kernel**: Upgrade `moe_gate.metal` from single-threaded (grid 1,1,1) to parallel, with `per_expert_scale` multiplication
4. **Expert offset indexing**: Add `expert_offset` parameter to `quantized_matmul_simd` to index into 3D packed weight tensor [128, rows, packed_cols] without CPU memcpy
5. **Keep expert_input on GPU**: Remove line 2711 readback entirely

#### New/Modified Metal Kernels

| Kernel | File | Status | Description |
|--------|------|--------|-------------|
| `moe_gate` | `moe_gate.metal` | **Modify** | Parallelize (currently single-threaded), add per_expert_scale, bf16 input support |
| `quantized_matmul_simd` | `quantized_matmul.metal` | **Modify** | Add expert_offset param for 3D weight indexing |
| `moe_router_norm_matmul` | New | **Create** | Fused RMS norm + router matvec (optional, can compose existing kernels) |

#### Changes Required

| File | Change |
|------|--------|
| `mlx-native/src/shaders/moe_gate.metal` | Parallelize kernel, add per_expert_scale, bf16 input |
| `mlx-native/src/ops/moe_gate.rs` | Update dispatch to use proper threadgroup size |
| `mlx-native/src/shaders/quantized_matmul.metal` | Add expert_offset indexing for 3D packed tensors |
| `mlx-native/src/ops/quantized_matmul.rs` | Add expert_offset parameter to dispatch functions |
| `gemma4.rs:moe_ffn()` | Rewrite to use GPU routing, remove CPU readbacks |
| `gemma4.rs` | Convert router_proj_cache to GPU buffers |

#### Acceptance Criteria

- [ ] **AC-2.1**: Zero `read_buffer_f32()` calls in the MoE hot path (lines 2710-2711 eliminated)
- [ ] **AC-2.2**: Zero CPU memcpy in `extract_expert_quant_buffers` -- expert weights accessed via GPU offset
- [ ] **AC-2.3**: MoE routing produces identical expert selections and weights as CPU baseline (test with 100 random inputs)
- [ ] **AC-2.4**: GPU buffer allocations per MoE layer <= 5 (down from ~48 MB of per-dispatch allocations), using buffer pooling
- [ ] **AC-2.5**: `moe_gate.metal` kernel uses parallel threadgroups (grid size > (1,1,1))

---

### Phase 3: Async Pipelined Generation

**Objective:** Overlap GPU computation with CPU sampling/streaming using double-buffered command execution.

#### Current Decode Timeline (sequential)

```
Token N:  [===GPU forward (8ms)===][CPU read logits (0.1ms)][CPU sample (2ms)][stream]
Token N+1:                                                                     [===GPU forward===]...
```

#### Target Decode Timeline (pipelined)

```
Token N:   [===GPU forward===][commit non-blocking]
Token N:                      [CPU: read logits N-1][sample N-1][stream N-1]
Token N+1:  GPU completed -->  [===GPU forward===][commit non-blocking]
Token N+1:                     [CPU: read logits N][sample N][stream N]
```

#### Changes Required

| File | Change |
|------|--------|
| `mlx-native/src/encoder.rs` | Add `commit()` (non-blocking) and `wait_until_completed()` as separate methods |
| `gemma4.rs:forward()` | Return logits buffer handle instead of `Vec<f32>` -- defer readback |
| `engine.rs` | Restructure decode loop: submit forward N+1, then read+sample N, then wait for N+1 |
| `engine.rs` | Pre-allocate two logits buffers for double-buffering |

#### Acceptance Criteria

- [ ] **AC-3.1**: GPU forward pass N+1 begins before CPU sampling of step N completes (verified by Metal GPU profiler timestamps)
- [ ] **AC-3.2**: Time-to-first-token (TTFT) unchanged or improved (no regression from pipeline setup)
- [ ] **AC-3.3**: Token streaming latency unchanged (tokens still delivered as soon as sampled)
- [ ] **AC-3.4**: Decode tok/s improves by >= 15% over Phase 2 baseline (CPU sampling hidden behind GPU compute)

---

### Phase 4: GPU-Side Sampling

**Objective:** Eliminate the 1MB logits readback and 2ms CPU sampling time per decode step.

#### Current Sampling Pipeline (CPU)

| Stage | Complexity | Allocations | Problem |
|-------|-----------|-------------|---------|
| Clone logits | O(V) | 1MB | Unnecessary copy |
| Repetition penalty | O(G) | None | Small, keep on CPU |
| Temperature scaling | O(V) | None | Trivial GPU kernel |
| Top-k filtering | **O(V log V)** | 2MB (indexed vec) | **Full sort of 262K elements** |
| Top-p filtering | **O(V log V)** | 3MB (softmax + sort + set) | **Another full sort** |
| Softmax | O(V) | 1MB | Already have GPU kernel |
| Categorical sample | O(V) | 1MB (cumulative dist) | Sequential scan |

**Total per token:** ~9.4M comparisons, ~8MB allocations, ~2ms wall time.

#### Target: GPU Sampling Pipeline

1. **GPU temperature kernel**: Elementwise divide (trivial, reuse `scalar_mul`)
2. **GPU top-K kernel**: Parallel partial sort using radix select or bitonic sort -- O(V) instead of O(V log V)
3. **GPU softmax**: Already exists (`softmax.metal`)
4. **GPU categorical sample**: Prefix sum + binary search on GPU, or read top-1 with `argmax` kernel for greedy
5. **CPU receives**: Single `u32` token ID (4 bytes vs 1MB)

#### New Metal Kernels

| Kernel | Description |
|--------|-------------|
| `top_k_filter` | Parallel partial sort: find k-th largest, mask below threshold |
| `argmax_f32` | Single-pass maximum with index (for greedy decoding) |
| `sample_categorical` | Prefix sum + uniform random threshold search |

#### Acceptance Criteria

- [ ] **AC-4.1**: GPU->CPU transfer per decode step <= 64 bytes (token ID + optional logprob)
- [ ] **AC-4.2**: Greedy decoding (temperature=0) produces identical output to CPU baseline
- [ ] **AC-4.3**: Stochastic sampling distribution matches CPU baseline within statistical tolerance (chi-squared test, p > 0.01, over 10K samples)
- [ ] **AC-4.4**: Sampling time per token < 0.1ms (down from ~2ms)

---

### Phase 5: Kernel Fusion

**Objective:** Reduce kernel dispatches from ~2,278 to ~800 per forward pass by fusing adjacent operations.

#### Fusion Targets (ordered by dispatch count reduction)

| Fusion | Current Dispatches | Fused | Saved/Layer | Saved/Forward | Priority |
|--------|-------------------|-------|-------------|---------------|----------|
| bf16 I/O qmatmul | 2 casts + 1 matmul = 3 | 1 | 8 | 272 | **P0** |
| RMSNorm + qmatmul | 1 norm + 1 matmul = 2 | 1 | 4 | 136 | **P1** |
| Head norm + RoPE | 3 norms + 2 ropes = 5 | 2 | 3 | 102 | **P1** |
| GELU + mul + cast | 2+1 = 3 | 1 | 1-2 | 34-68 | P2 |
| Residual + norm | 1 add + 1 norm = 2 | 1 | 4 | 136 | P2 |
| SDPA + output permute | 1 sdpa + 1 permute = 2 | 1 | 1 | 34 | P3 |

#### New/Modified Metal Kernels

| Kernel | Description | Saves |
|--------|-------------|-------|
| `quantized_matmul_simd_bf16` | bf16 input, bf16 output (no f32 casts) | 272 dispatches |
| `fused_head_norm_rope_bf16` | Per-head RMSNorm + NeoX RoPE in one pass | 102 dispatches |
| `fused_residual_norm_bf16` | elementwise_add + rms_norm | 136 dispatches |

#### Acceptance Criteria

- [ ] **AC-5.1**: Total kernel dispatches per forward pass (decode) <= 1,200
- [ ] **AC-5.2**: Fused `quantized_matmul_simd_bf16` produces output within bf16 epsilon of cast+matmul+cast chain
- [ ] **AC-5.3**: Fused `fused_head_norm_rope_bf16` matches separate norm+rope output within bf16 epsilon
- [ ] **AC-5.4**: Kernel launch overhead < 2ms per forward pass (down from ~5ms)

---

### Phase 6: Fix Remaining Inefficiencies

**Objective:** Eliminate remaining low-hanging fruit identified in the audit.

#### 6a. Embedding CPU Round-Trip

**Current** (gemma4.rs:417-420): `embedding_gather` -> `read_buffer_f32` -> CPU bf16 convert -> `f32_vec_to_bf16_buffer`
**Fix**: GPU f32->bf16 cast kernel directly on embedding output. Eliminates 1 sync + 2 copies.

#### 6b. Prefill Last-Token-Only LM Head

**Current**: During prefill, `lm_head` projects `[seq_len, hidden_size] @ [hidden_size, vocab_size]` producing `[seq_len, vocab_size]` logits, but only the last row is used.
**Fix**: Extract last hidden state position before lm_head. Saves `(seq_len-1)/seq_len` of compute and memory.

#### 6c. Layer Scalar Caching

**Current** (gemma4.rs:1529-1538): Reads 4-byte scalar weight from GPU buffer every layer.
**Fix**: Cache scalar values on CPU at model init (30 x 4 bytes = 120 bytes).

#### 6d. Top-K Partial Sort (Quick Win)

**Current** (sampler.rs:126): `sort_unstable_by` O(V log V) on 262K elements.
**Fix**: `select_nth_unstable_by` O(V) to find k-th element, then single pass to collect top-k. 3-4x speedup even before GPU sampling is ready.

#### 6e. Deduplicate Softmax

**Current**: Sampler calls softmax up to 3 times per sample (top-p, categorical, implicit in top-k).
**Fix**: Compute once, reuse.

#### 6f. Buffer Pooling for MoE

**Current**: ~48 MB of GPU buffer allocations per MoE layer (fresh alloc every call).
**Fix**: Pre-allocate scratch buffers at model init, reuse across layers and steps.

#### Acceptance Criteria

- [ ] **AC-6.1**: Zero `read_buffer_f32` calls in `forward()` for embedding (line 417 eliminated)
- [ ] **AC-6.2**: During prefill, lm_head processes only 1 hidden state position (last), not seq_len
- [ ] **AC-6.3**: Sampler calls softmax at most once per `sample_next()` invocation
- [ ] **AC-6.4**: Top-K uses O(V) selection algorithm, not O(V log V) sort
- [ ] **AC-6.5**: MoE scratch buffer allocations per decode step = 0 (all pre-allocated)

---

## 3. Metrics and Measurement

### Metrics to Track

| Metric | Unit | How Measured | Current Infra |
|--------|------|-------------|---------------|
| Decode tok/s | tokens/sec | `GenerationStats.decode_tokens_per_sec()` | Exists in engine.rs |
| Prefill tok/s | tokens/sec | `GenerationStats.prefill_tokens_per_sec()` | Exists in engine.rs |
| TTFT | ms | Needs explicit field in `GenerationStats` | **Must build** |
| Sync points/forward | count | Atomic counter in `CommandEncoder` | **Must build** |
| Kernel dispatches/forward | count | Counter in encoder `encode()` calls | **Must build** |
| Peak GPU memory | bytes | Metal `device.currentAllocatedSize` or equivalent | **Must build** |
| GPU->CPU transfers/step | count + bytes | Counter in `read_buffer_f32` and `as_slice` | **Must build** |

### Required Instrumentation (build before Phase 1)

1. **Sync counter**: Add `static SYNC_COUNT: AtomicU64` to `encoder.rs`, increment on every `commit_and_wait()`. Reset at forward pass start, read at end. Expose in `GenerationStats`.
2. **Dispatch counter**: Same pattern for kernel encodes.
3. **TTFT field**: Add `time_to_first_token_ms: f64` to `GenerationStats`. Measure as prefill_time + first sample time.
4. **API exposure**: Add `x_hf2q_timing` extension object to OpenAI-compatible API responses containing all timing fields.

### Benchmark Protocol

Use existing `scripts/benchmark.sh` infrastructure with additions:

- **Model**: `mlx-community/gemma-4-12b-a4b-it-4bit` (or 27B if available)
- **Hardware**: Apple Silicon (specify exact chip in results)
- **Prompt lengths**: 20, 256, 1024 tokens
- **Max tokens**: 128
- **Temperature**: 0 (deterministic for correctness verification)
- **Warm-up**: 2 runs discarded
- **Measurement**: 5 runs, report median with min/max
- **Baselines**: mlx-lm (via `scripts/bench_mlx_lm.py`), Ollama (via new `scripts/bench_ollama.py`)

### Success Criteria (End State)

| Metric | Baseline (est.) | Phase 1 Target | Phase 3 Target | Final Target |
|--------|-----------------|----------------|----------------|--------------|
| Decode tok/s | ~15-25 | >= 30 | >= 40 | >= 50 (match Ollama) |
| Prefill tok/s | ~100-200 | >= 200 | >= 300 | >= 500 |
| TTFT (256 tok prompt) | ~500ms | <= 400ms | <= 300ms | <= 200ms |
| Sync points/forward | ~305 | <= 100 | <= 93 | <= 93 |
| Kernel dispatches/forward | ~2,278 | ~2,278 | ~2,278 | <= 1,200 |
| GPU->CPU bytes/decode step | ~1.7 MB | ~1.0 MB | ~1.0 MB | <= 64 B |

## 4. Implementation Order

```
Phase 0: Instrumentation (sync counter, dispatch counter, TTFT, API exposure)
    |
Phase 1: Eliminate GPU Sync Points (gemma4.rs encoder restructuring)
    |
Phase 2: GPU-Native MoE Routing (moe_gate.metal + expert offset indexing)
    |
Phase 3: Async Pipelined Generation (double-buffered decode loop)
    |
Phase 4: GPU-Side Sampling (top-k kernel + argmax + categorical)
    |
Phase 5: Kernel Fusion (bf16 qmatmul, head norm+rope, residual+norm)
    |
Phase 6: Remaining Fixes (embedding round-trip, prefill lm_head, buffer pool)
```

Phases 1-3 are expected to deliver the majority of throughput improvement. Phases 4-6 are polish that close the remaining gap.

## 5. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Merging encoders introduces Metal command buffer ordering bugs | Medium | High -- silent correctness errors | Test with bit-exact logit comparison before/after each merge |
| GPU MoE routing produces different expert selection due to floating point | Medium | Medium -- quality regression | Compare expert IDs across 1000 inputs, allow zero divergence |
| Async pipeline introduces race conditions | Medium | High -- crashes or corruption | Use Metal `MTLEvent` for synchronization, no raw pointer sharing |
| bf16 qmatmul kernel has different rounding than cast+f32+cast chain | Low | Low -- minor quality diff | Validate within bf16 epsilon (1/256) |
| Buffer pooling logic complexity | Low | Medium -- memory leaks | Track allocations with debug counter, assert zero leak at shutdown |

## 6. Non-Goals

- Multi-sequence batching / continuous batching (separate future PRD)
- Speculative decoding (requires draft model infrastructure)
- Tensor parallelism / multi-GPU (single-device only for now)
- Disk-backed KV cache / prefix trie (Ollama feature, not needed yet)
- Support for non-Gemma4 model architectures
- Graph compilation / JIT (would require a mini compiler in mlx-native -- too large for this PRD)

## 7. Key Files

### hf2q (this repo)

| File | Lines | Role |
|------|-------|------|
| `src/inference/models/gemma4.rs` | 4657 | Forward pass, all sync points, MoE routing |
| `src/inference/engine.rs` | 927 | Decode loop, prefill, prompt cache, GenerationStats |
| `src/inference/kv_cache.rs` | 721 | KV cache with sliding/global dual layout |
| `src/inference/sampler.rs` | 455 | CPU sampling pipeline |
| `src/inference/prompt_cache.rs` | 686 | Prefix matching and cache management |

### mlx-native (/opt/mlx-native)

| File | Role |
|------|------|
| `src/encoder.rs` | CommandEncoder with commit_and_wait sync model |
| `src/shaders/quantized_matmul.metal` | SIMD quantized matmul (most-called kernel) |
| `src/shaders/rms_norm.metal` | RMS normalization (most-frequent kernel) |
| `src/shaders/sdpa.metal` | Scaled dot-product attention |
| `src/shaders/sdpa_sliding.metal` | Sliding window SDPA |
| `src/shaders/moe_gate.metal` | MoE gating (exists but single-threaded, unused) |
| `src/shaders/moe_dispatch.metal` | MoE dispatch helpers (partially unused) |
| `src/ops/moe_gate.rs` | MoE gate Rust dispatch |
| `src/ops/quantized_matmul.rs` | Quantized matmul Rust dispatch |

### Ollama Reference (/opt/ollama)

| File | What to Study |
|------|---------------|
| `x/mlxrunner/pipeline.go` | Async pipelined generation with double-buffering |
| `x/mlxrunner/mlx/fast.go` | Fused SDPA, RMSNorm, RoPE, LayerNorm |
| `x/mlxrunner/cache_trie.go` | Prefix trie for KV cache sharing |
| `x/mlxrunner/mlx/array.go` | Pin/Sweep memory management |
| `model/models/gemma4/model_text.go` | Gemma4 attention with KV sharing, K=V mode |
