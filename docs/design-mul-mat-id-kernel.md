# Design: SIMD-Optimized mul_mat_id Metal Kernel for MoE

## Reference: llama.cpp Architecture

### Two dispatch paths based on batch size

**Decision point** (ggml-metal-ops.cpp:2297):
- `ne21 >= 32` (batch >= 32 tokens) → **mm_id** path (matrix-matrix, simdgroup 8x8 ops)
- `ne21 < 32` (batch < 32 tokens, includes decode) → **mv_id** path (matrix-vector, dot product per row)

For single-token decode (our primary benchmark), the **mv_id** path is used.

### mv_id path (decode, batch=1)

**Architecture** (ggml-metal.metal:10179-10240):
- `kernel_mul_mv_id` is a thin wrapper that extracts expert-specific pointers from the routing indices, then delegates to the existing `kernel_mul_mv_*` implementation
- Grid Z = `ne20 * ne21` = `n_expert_used * n_tokens` (for decode: `top_k * 1 = 8`)
- Each threadgroup computes one (expert, token) pair
- The inner kernel is the same SIMD-optimized mul_mv used for dense layers — just indexed differently

**Key insight**: llama.cpp reuses the SAME optimized mul_mv kernel for both dense and MoE dispatch. The _id wrapper just redirects pointers.

**Threadgroup dispatch** (ggml-metal-ops.cpp:2436-2438):
- Q6_K: `(ne01 + nr0*nsg - 1)/(nr0*nsg)` × 1 × `ne123`
- Where `nr0=2, nsg=2` → `(ne01 + 3)/4` threadgroups per output row batch
- 32 threads × nsg simdgroups per threadgroup
- ne123 = top_k * n_tokens = 8 for decode

### mm_id path (prefill, batch >= 32)

**Two-phase dispatch:**

Phase 1: `kernel_mul_mm_id_map0` (ggml-metal.metal:9575-9626)
- One threadgroup with `ne02` threads (one per expert)
- Builds per-expert token lists: `ids_i32[expert][n]` = token indices, `tpe_u32[expert]` = count
- Templated on `ne20` (n_expert_used, typically 8)
- Uses threadgroup memory for broadcast of expert routing data

Phase 2: `kernel_mul_mm_id` (ggml-metal.metal:9641-9943)
- Grid: `(ceil(ne21/32), ceil(ne01/64), ne02)` = `(tokens/32, out_rows/64, n_experts)`
- 128 threads per threadgroup (4 simdgroups)
- Tile size: NR0=64 (output rows) × NR1=32 (tokens) × NK=32 (K-dim tile)
- Uses simdgroup 8x8 matrix multiply-accumulate
- Threadgroup memory: 8192 bytes for tile staging
- Early exit when `r1 >= neh1` (no tokens for this expert in this tile)
- Dequantization happens during load to threadgroup memory via `dequantize_func`

Memory barrier between phases 1 and 2 (`ggml_metal_op_concurrency_reset`).

### Q6_K dequantization function

Reference: `dequantize_q6_K` in ggml-metal.metal
- 210 bytes per 256 elements
- ql[128]: lower 4 bits, qh[64]: upper 2 bits, scales[16]: 8-bit per sub-block
- Produces a 4×4 register of 16 float values per call
- Called with `il` parameter (0..15) to select which 16-element chunk

### Q8_0 dequantization function

Reference: `dequantize_q8_0` in ggml-metal.metal
- 34 bytes per 32 elements
- QK8_0 = 32 (NOT 256 like K-quants)
- Simple: `weight = d * qs[i]`

## Design for hf2q

### Approach: Reuse candle's existing QMatMul Metal kernels via mul_mv_id wrapper

Instead of writing our own SIMD dequant kernels (duplicating llama.cpp's work), we should:

1. **For decode (single token)**: Write a thin `mul_mv_id`-style wrapper that:
   - Takes the expert routing indices and the merged 3D expert weight QTensor
   - For each of the top_k experts, computes the byte offset into the merged weight buffer
   - Dispatches candle's existing `kernel_mul_mv_q6_K_f32` / `kernel_mul_mv_q8_0_f32` with the offset pointer
   - This reuses candle's already-optimized SIMD kernels

2. **For prefill (multi-token)**: Write the two-phase approach:
   - Phase 1: map kernel (simple, builds per-expert token lists)
   - Phase 2: dispatch candle's `kernel_mul_mm_q6_K_f32` / `kernel_mul_mm_q8_0_f32` per expert
   - Or: use candle's QMatMul directly with per-expert index_select (batched matmul)

### Alternative: Direct integration with candle's QMatMul

Instead of custom Metal kernels, leverage candle's existing QMatMul:
- Store expert weights as individual QMatMul objects (128 per layer)
- For decode: call `qmatmul.forward()` for each of the top_k=8 selected experts
- For prefill: batch tokens per expert, call `qmatmul.forward()` on the batch
- This uses candle's fully-optimized Metal kernels without any custom shader code

**Pros**: Zero custom Metal code. Proven correct. SIMD-optimized already.
**Cons**: 8 separate kernel launches per layer for decode (vs llama.cpp's 1). Per-token CPU routing.

### Hybrid approach (recommended)

1. Store expert weights as 128 QMatMul objects per layer (loaded from QTensors)
2. GPU-side routing (1b.6 fix) to avoid CPU sync
3. For decode: dispatch 8 QMatMul calls with GPU routing indices
4. For prefill: batch per expert, dispatch N QMatMul calls (N = active experts)
5. Custom mul_mv_id kernel as a future optimization to merge the 8 launches into 1

## Key Numbers (Gemma 4 26B MoE)

- num_experts = 128, top_k = 8
- hidden_size = 2816
- moe_intermediate_size = 704
- gate_up weight per expert: [1408, 2816] Q6_K
- down weight per expert: [2816, 704] Q8_0
- Expert weights total: ~20 GB quantized on GPU

## Implementation Plan

1. Load expert weights as Vec<QMatMul> per layer (128 entries)
2. Keep GPU-side routing (index_select + gather for combined weights)
3. For decode: loop over top_k, call qmatmul.forward() for each
4. For prefill: batch tokens per expert, call qmatmul.forward()
5. Eliminate per-token CPU sync via GPU tensor routing
6. Benchmark against llama.cpp
7. If still slow, write proper mul_mv_id wrapper kernel
