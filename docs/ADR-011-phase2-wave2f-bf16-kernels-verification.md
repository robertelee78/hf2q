# ADR-011 Phase 2 Wave 2F — bf16 Kernel Verification

**Agent**: 2F (bf16-kernels)  
**Swarm**: swarm-1776516482254-ft5mwj  
**Date**: 2026-04-17  
**Status**: COMPLETE — all 5 kernels ported, 7 tests passing, clippy below baseline

---

## Summary

Ported 5 missing bfloat16 Metal GPU kernel variants into `/opt/mlx-native` required for
the Phase 2 bf16 activation conversion of `forward_prefill_batched.rs`. All kernels
follow the MLX-LM convention: accumulate in f32, read/write in bf16.

---

## Kernels Ported

### 1. `fused_head_norm_rope_batch_bf16`

**File**: `src/shaders/fused_head_norm_rope_bf16.metal` (appended)  
**Rust dispatch**: `src/ops/fused_head_norm_rope.rs::dispatch_fused_head_norm_rope_batch_bf16`

Batched Q/K head normalization + NeoX RoPE for seq_len tokens in a single dispatch.

- Grid: `(seq_len * n_heads, 1, 1)` — one threadgroup per (token, head) pair
- Phase 1: parallel sum-of-squares reduction in shared f32 memory
- Phase 2: normalize + optional weight scale → store in shared f32 scratch
- Phase 3: NeoX rotation on pairs `(shared[i], shared[i + half_dim])`, sin/cos computed
  on-the-fly from `positions_buf` and `theta` (ProportionalRoPE, head_dim as denominator)
- Supports `freq_factors` for Gemma4-style per-pair frequency divisors
- Full barrier between Phase 1 and Phase 2 to prevent shared memory race
- Shared memory: `max(tg_size, head_dim) * 4` bytes

**f32 equivalent verified**: `fused_head_norm_rope_f32` in `fused_head_norm_rope_f32.metal`

### 2. `fused_gelu_mul_bf16`

**File**: `src/shaders/moe_dispatch.metal` (inserted before multi-token SwiGLU)  
**Rust dispatch**: `src/ops/moe_dispatch.rs::fused_gelu_mul_bf16_encode`

Element-wise GELU(gate) * up with bf16 I/O.

- Grid: 1D flat, one thread per element
- Reads gate and up bf16 → promotes to f32 → applies `tanh`-approximated GELU → multiplies → casts back to bf16
- Uses the same `gelu_approx` helper already defined in the file
- Validates `gate_out` and `up_out` byte sizes against `n_elements * 2` (bf16)

**f32 equivalent verified**: `fused_gelu_mul` in the same file (line 1 of moe_dispatch.metal)

### 3. `moe_swiglu_seq_bf16`

**File**: `src/shaders/moe_dispatch.metal` (appended)  
**Rust dispatch**: `src/ops/moe_dispatch.rs::moe_swiglu_seq_bf16_encode`

Batched MoE SwiGLU: GELU(gate_half) * up_half with bf16 I/O, for seq_len tokens × top_k experts.

- Input: `gate_up_buf` as bf16 `[n_tokens, top_k, 2 * intermediate]` (gate and up concatenated)
- Output: bf16 `[n_tokens, top_k, intermediate]`
- Grid: `(intermediate, top_k, n_tokens)` — one thread per output element
- Reads gate/up in bf16, accumulates GELU in f32, stores result in bf16

**f32 equivalent verified**: `moe_swiglu_seq` in the same file

### 4. `kv_cache_copy_seq_bf16`

**File**: `src/shaders/kv_cache_copy.metal` (inserted before `kv_cache_copy_seq_f32_to_f16`)  
**Rust dispatch**: `src/ops/kv_cache_copy.rs::dispatch_kv_cache_copy_seq_bf16`

Copies bf16 K/V normed output to f32 KV cache, with src_tok_offset slicing and ring-buffer support.

- Source: bf16 `[n_src_tokens, n_heads, head_dim]` (token-major)
- Cache: f32 `[n_heads, capacity, head_dim]` (head-major)
- Grid: `(head_dim, n_heads, n_tokens)` — 3D dispatch
- Promotes bf16 → f32 on write to preserve cache precision
- Supports `src_tok_offset` for sliding-window partial copies
- `dst_pos % capacity` for ring-buffer wrapping (consistent with `ring_start` decode path)
- Validation: `src.byte_len() >= total_src * 2` (bf16 = 2 bytes/element)

**f32 equivalent verified**: `kv_cache_copy_seq_f32` in the same file

### 5. `moe_weighted_sum_seq_bf16_input`

**File**: `src/shaders/moe_dispatch.metal` (appended)  
**Rust dispatch**: `src/ops/moe_dispatch.rs::moe_weighted_sum_seq_bf16_input_encode`

MoE expert-output weighted combiner: bf16 expert outputs × f32 routing weights → f32 result.

- Expert outputs: bf16 `[n_tokens, top_k, hidden_size]`
- Weights: f32 `[n_tokens, top_k]`
- Output: f32 `[n_tokens, hidden_size]`
- Grid: `(hidden_size, n_tokens)` — one thread per output element
- Accumulates in f32 across top_k experts; promotes bf16 expert data on read
- Validation: expert_outputs byte size against `total_expert * 2`, weights f32 size

**f32 equivalent verified**: `moe_weighted_sum_seq` in the same file

---

## Kernel Registry Changes

`src/kernel_registry.rs` additions in `KernelRegistry::new()`:

```
"kv_cache_copy_seq_bf16"              → kv_cache_src
"fused_gelu_mul_bf16"                  → moe_dispatch_src
"moe_swiglu_seq_bf16"                  → moe_dispatch_src
"moe_weighted_sum_seq_bf16_input"      → moe_dispatch_src
"fused_head_norm_rope_batch_bf16"      → fused_hnr_bf16_src (new block)
```

---

## Test Results

```
test test_fused_head_norm_rope_batch_bf16_with_weight ... ok
test test_fused_head_norm_rope_batch_bf16_no_weight   ... ok
test test_fused_gelu_mul_bf16_basic                   ... ok
test test_moe_swiglu_seq_bf16_basic                   ... ok
test test_kv_cache_copy_seq_bf16_linear               ... ok
test test_kv_cache_copy_seq_bf16_with_offset          ... ok
test test_moe_weighted_sum_seq_bf16_input_basic       ... ok

test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Test file: `/opt/mlx-native/tests/test_bf16_kernels.rs`  
All tests run on Apple Silicon GPU (Metal); `#![cfg(target_vendor = "apple")]` guard.

---

## Build Verification

```
cargo check --lib          → Finished cleanly (0 errors)
cargo clippy --lib --no-deps → 81 warnings (baseline was 84 — net -3)
cargo test --test test_bf16_kernels → 7 passed, 0 failed
```

Zero net-new clippy warnings. Fixed 3 doc-indent warnings in `fused_head_norm_rope.rs`
and 2 `manual_div_ceil` warnings in flash_attn files that were included in the diff.

---

## Design Decisions

### bf16 + f32 accumulator pattern
All kernels follow the MLX-LM convention: `static_cast<float>()` on read, accumulate in
f32, `bfloat()` cast on store. This preserves numerical stability in GELU and weighted sum.

### On-the-fly RoPE in batch kernel
The single-token `fused_head_norm_rope_bf16` uses precomputed cos/sin caches (position=0
only). The batch variant must handle arbitrary positions per token, so it computes sin/cos
on-the-fly from `positions_buf[seq_idx]` and `theta`, matching the f32 batch kernel.

### Shared memory race fix
A barrier is placed after reading `rms_inv` from `shared[0]` and before any thread writes
to `shared[i]` in Phase 2. This prevents a race where thread 0 (which completes the
reduction) could begin writing normalized values before other threads have read `rms_inv`.
Ref: `docs/spike-batched-prefill-race-rootcause.md`.

### `src_tok_offset` in kv_cache_copy_seq_bf16
Mirrors the f32 variant exactly. For sliding-window layers, caller passes
`src_tok_offset = seq_len - n_tokens` to select the tail of the source buffer.
