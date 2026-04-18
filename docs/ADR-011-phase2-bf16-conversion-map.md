# ADR-011-phase2-bf16-conversion-map

**Status**: Research / Pre-implementation  
**Date**: 2026-04-17  
**Author**: Agent #5 (research-bf16-surface), CFA swarm swarm-1776516482254-ft5mwj  
**Scope**: Phase 2 bf16 conversion plan for `forward_prefill_batched.rs`

---

## 1. Objective

Convert hf2q's batched prefill path from an all-f32 activation pipeline to a
bf16-for-intermediates / f32-for-residual pipeline matching the MLX-LM and
llama.cpp convention, so that `flash_attn_prefill` (which only has bf16/f16
kernel variants — f32 is not instantiated due to Apple Silicon threadgroup
memory limits) can be wired into the batched prefill path.

---

## 2. Background: current state of both prefill paths

### 2.1 `forward_prefill.rs` (single-token / per-token prefill)

This path is **also fully f32**. It allocates activations as f32 and dispatches
all f32-variant kernels:

- `embedding_gather_scale_f32` (elementwise.rs)
- `dispatch_fused_head_norm_rope_f32` (fused_head_norm_rope.rs)
- `dispatch_fused_norm_add_f32` (fused_norm_add.rs)
- `dispatch_fused_norm_add_scalar_f32` (fused_norm_add.rs)
- `flash_attn_vec` (flash_attn_vec.rs) — the decode SDPA, operates on f32 KV

So both prefill paths are f32 end-to-end. **Phase 2 scope is limited to
`forward_prefill_batched.rs`**; the per-token path can remain f32 as it uses
`flash_attn_vec` (decode kernel, not prefill kernel).

### 2.2 MLX-LM dtype convention for prefill

Based on existing ADR analysis and the `flash_attn_prefill.rs` module doc:

| Tensor | MLX-LM dtype | Notes |
|---|---|---|
| Embedding output (residual start) | f32 | Gemma-4 uses f32 residual for numerical stability |
| Residual stream throughout | f32 | Accumulated in f32 |
| QKV projections output | bf16 | Dequant matmul output is bf16 |
| Q/K after head norm + RoPE | bf16 | Input to SDPA |
| V after head norm | bf16 | Input to SDPA |
| SDPA output | bf16 | flash_attn_prefill outputs bf16 |
| O-proj output | bf16 | matmul output |
| attn_out added to residual (cast) | f32 | bf16 -> f32 before residual add |
| MLP gate/up outputs | bf16 | matmul outputs |
| MLP fused (GELU*up) | bf16 | stays bf16 |
| MLP down output | bf16 | matmul output |
| MoE gate_up output | bf16 | quantized_matmul_id_ggml output |
| MoE swiglu output | bf16 | stays bf16 |
| MoE down output | bf16 | quantized_matmul_id_ggml output |
| MoE accum (weighted sum) | f32 | gathered into f32 for residual |
| Final norm output | f32 | feeds lm_head |
| Logits | f32 | argmax in f32 |

**Key convention**: residual = f32 everywhere; sublayer outputs (attn, MLP, MoE)
= bf16; cast bf16 → f32 before adding to residual.

---

## 3. Full f32-site inventory in `forward_prefill_batched.rs`

Every `alloc_f32` and every f32-variant kernel call:

### 3.1 Buffer allocations (lines 105-149)

| File:Line | Variable | Current dtype | Target dtype | Category |
|---|---|---|---|---|
| forward_prefill_batched.rs:105-108 | `alloc_f32` closure | f32 | (helper) | Helper |
| forward_prefill_batched.rs:115 | `pf_hidden` | f32 | **f32** | residual stream |
| forward_prefill_batched.rs:116 | `pf_residual` | f32 | **f32** | residual stream |
| forward_prefill_batched.rs:117 | `pf_norm_out` | f32 | **bf16** | pre-attn norm output (intermediate) |
| forward_prefill_batched.rs:118 | `pf_moe_norm_out` | f32 | **bf16** | pre-FF norm2 output (intermediate) |
| forward_prefill_batched.rs:119 | `pf_router_norm_out` | f32 | **bf16** | router norm output (intermediate) |
| forward_prefill_batched.rs:120 | `pf_attn_out` | f32 | **bf16** | O-proj output (intermediate) |
| forward_prefill_batched.rs:121 | `pf_mlp_down_out` | f32 | **bf16** | post-FF norm1 output (intermediate) |
| forward_prefill_batched.rs:123 | `pf_q` | f32 | **bf16** | Q projection output (intermediate) |
| forward_prefill_batched.rs:124 | `pf_k` | f32 | **bf16** | K projection output (intermediate) |
| forward_prefill_batched.rs:125 | `pf_v` | f32 | **bf16** | V projection output (intermediate) |
| forward_prefill_batched.rs:126 | `pf_q_normed` | f32 | **bf16** | Q after head norm+RoPE (SDPA input) |
| forward_prefill_batched.rs:127 | `pf_k_normed` | f32 | **bf16** | K after head norm+RoPE (SDPA input, cache source) |
| forward_prefill_batched.rs:128 | `pf_v_normed` | f32 | **bf16** | V after head norm (SDPA input, cache source) |
| forward_prefill_batched.rs:130 | `pf_q_perm` | f32 | **bf16** | Q permuted [nh, seq, hd] |
| forward_prefill_batched.rs:131 | `pf_k_perm` | f32 | **bf16** | K permuted [nkv, seq, hd] |
| forward_prefill_batched.rs:132 | `pf_v_perm` | f32 | **bf16** | V permuted [nkv, seq, hd] |
| forward_prefill_batched.rs:133 | `pf_sdpa_out_perm` | f32 | **bf16** | SDPA output permuted [nh, seq, hd] |
| forward_prefill_batched.rs:134 | `pf_sdpa_out` | f32 | **bf16** | SDPA output [seq, nh, hd] |
| forward_prefill_batched.rs:136 | `pf_mlp_gate` | f32 | **bf16** | MLP gate projection output |
| forward_prefill_batched.rs:137 | `pf_mlp_up` | f32 | **bf16** | MLP up projection output |
| forward_prefill_batched.rs:138 | `pf_mlp_fused` | f32 | **bf16** | GELU(gate)*up (intermediate) |
| forward_prefill_batched.rs:139 | `pf_mlp_down` | f32 | **bf16** | MLP down projection output (re-used as combined output below) |
| forward_prefill_batched.rs:143 | `pf_router_logits` | f32 | **f32** | Router logits (softmax internal, stays f32) |
| forward_prefill_batched.rs:145 | `pf_routing_weights` | f32 | **f32** | Routing weights (used in weighted sum, stays f32) |
| forward_prefill_batched.rs:146 | `pf_moe_gate_up` | f32 | **bf16** | MoE gate_up expert output |
| forward_prefill_batched.rs:147 | `pf_moe_swiglu` | f32 | **bf16** | MoE SwiGLU output |
| forward_prefill_batched.rs:148 | `pf_moe_down` | f32 | **bf16** | MoE down expert output |
| forward_prefill_batched.rs:149 | `pf_moe_accum` | f32 | **f32** | MoE weighted sum result (before residual add, stays f32) |

**Note on `pf_router_logits` and `pf_routing_weights`**: the router logits go
through softmax internally in `fused_moe_routing_batch_f32`; these are scalar
probability tensors used only for the weighted sum weighting and do not feed
into the flash_attn_prefill pipeline, so staying f32 is acceptable and avoids
precision issues in the softmax.

**Note on `pf_mlp_down` dual use (lines 139, 671, 688)**: `pf_mlp_down` is
allocated at line 139, used as the dense MLP down projection output (line 574),
then overwritten by `dispatch_fused_norm_add_f32` (post-FF norm2+combine at
line 673) and again by `dispatch_fused_norm_add_scalar_f32` (end-of-layer at
line 688). After the end-of-layer step it feeds `pf_hidden` (the next layer's
residual input). This dual-role means `pf_mlp_down` should become **bf16**
for its first use (down projection output), but must be f32 by the time it
feeds `pf_hidden`. Either: (a) use a separate buffer for the combined MLP+MoE
f32 result, or (b) apply the cast inside a new bf16-capable fused kernel.

### 3.2 Kernel call sites (f32-typed calls)

| File:Line | Function called | Inputs | Output | Notes |
|---|---|---|---|---|
| :175 | `embedding_gather_scale_batch_f32` | embed_weight, pf_token_ids | pf_hidden | embeds into f32 residual — stays f32 |
| :257-264 | `s.rms_norm` (pre-attn norm) | pf_hidden (f32) | pf_norm_out | needs bf16 output |
| :309-330 | `dispatch_fused_head_norm_rope_batch_f32` (Q) | pf_q (f32) | pf_q_normed | all-f32 kernel |
| :320-330 | `dispatch_fused_head_norm_rope_batch_f32` (K) | pf_k (f32) | pf_k_normed | all-f32 kernel |
| :339-348 | `dispatch_rms_norm_unit_perhead` (V, v_is_k path) | pf_k | pf_v_normed | f32 path |
| :354-363 | `dispatch_rms_norm_unit_perhead` (V, else path) | pf_v | pf_v_normed | f32 path |
| :371-375 | `permute_021_f32` (Q) | pf_q_normed | pf_q_perm | f32 permute |
| :376-380 | `permute_021_f32` (K) | pf_k_normed | pf_k_perm | f32 permute |
| :381-385 | `permute_021_f32` (V) | pf_v_normed | pf_v_perm | f32 permute |
| :420-426 | `sdpa_sliding` (is_sliding path) | pf_q_perm, pf_k_perm, pf_v_perm (f32) | pf_sdpa_out_perm | will dispatch sdpa_sliding_bf16 if buf dtype is bf16 |
| :437-443 | `s.sdpa` (global path) | pf_q_perm, pf_k_perm, pf_v_perm (f32) | pf_sdpa_out_perm | will dispatch sdpa_bf16 if buf dtype is bf16 |
| :451-455 | `permute_021_f32` (SDPA back) | pf_sdpa_out_perm | pf_sdpa_out | f32 permute |
| :462-464 | `dispatch_qmatmul` O-proj | pf_sdpa_out (f32) | pf_attn_out | quantized weight matmul; output dtype follows weight convention |
| :472-479 | `dispatch_fused_norm_add_f32` (post-attn) | pf_hidden (f32), pf_attn_out (f32) | pf_residual | residual add — must remain f32 |
| :497-517 | `s.rms_norm` x3 (pre-FF norms) | pf_residual (f32) | pf_norm_out, pf_moe_norm_out, pf_router_norm_out | all f32 → need bf16 outputs |
| :524-532 | `dispatch_qmatmul` gate+up+router | pf_norm_out / pf_router_norm_out | pf_mlp_gate, pf_mlp_up, pf_router_logits | output dtype |
| :541-557 | `fused_gelu_mul` pipeline | pf_mlp_gate, pf_mlp_up | pf_mlp_fused | f32 kernel currently |
| :558-565 | `dispatch_fused_moe_routing_batch_f32` | pf_router_logits | pf_expert_ids, pf_routing_weights | stays f32 |
| :572-574 | `dispatch_qmatmul` MLP down | pf_mlp_fused | pf_mlp_down | output dtype |
| :588-603 | `s.quantized_matmul_id_ggml` (gate_up_id) | pf_moe_norm_out | pf_moe_gate_up | output dtype |
| :610-615 | `moe_swiglu_seq_encode` | pf_moe_gate_up | pf_moe_swiglu | kernel dtype |
| :624-639 | `s.quantized_matmul_id_ggml` (down_id) | pf_moe_swiglu | pf_moe_down | output dtype |
| :646-652 | `s.rms_norm` (post-FF norm1) | pf_mlp_down | pf_mlp_down_out | f32 → need bf16 output |
| :660-666 | `moe_weighted_sum_seq_encode` | pf_moe_down, pf_routing_weights | pf_moe_accum | kernel dtype |
| :673-680 | `dispatch_fused_norm_add_f32` (post-FF norm2+combine) | pf_mlp_down_out, pf_moe_accum | pf_mlp_down | mixed dtypes; result feeds end-of-layer |
| :688-697 | `dispatch_fused_norm_add_scalar_f32` (end-of-layer) | pf_residual, pf_mlp_down | pf_hidden | residual update — output must be f32 |
| :924-931 | `dispatch_copy_f32` (last-row copy) | pf_hidden | activations.hidden | f32 copy |

**Total distinct f32 sites**: 28 buffer allocations + 24 kernel call sites = **52 f32 sites** to touch.

---

## 4. Dependent mlx-native kernels

### 4.1 Kernels that ALREADY have bf16 variants (switch is free)

| Kernel (f32 name) | bf16 variant | Status | Notes |
|---|---|---|---|
| `sdpa` | `sdpa_bf16` | Registered in kernel_registry.rs:141 | Dispatch selects by buffer dtype automatically |
| `sdpa_sliding` | `sdpa_sliding_bf16` | Registered in kernel_registry.rs:144 | Same auto-dispatch |
| `permute_021_f32` | `permute_021_bf16` | Registered in kernel_registry.rs:133; `transpose.rs` exports both | Caller must change function name + buffer alloc dtype |
| `rms_norm_f32` | `rms_norm_bf16` | Registered in kernel_registry.rs:219 | `dispatch_rms_norm` selects by buffer dtype |
| `fused_norm_add_bf16` | Already bf16 | Registered; but current batched code calls `dispatch_fused_norm_add_f32` | Must switch to bf16 variant where output is bf16 |
| `cast_f32_to_bf16` | (is the bf16 variant) | Registered in kernel_registry.rs:128 | `dispatch_cast_f32_to_bf16_with_encoder` available in elementwise.rs:486 |
| `cast_bf16_to_f32` | (is the f32 variant) | Registered in kernel_registry.rs:127 | `dispatch_cast_bf16_to_f32_with_encoder` available in elementwise.rs:524 |

### 4.2 Kernels that NEED a new bf16 variant

| Kernel (f32 name) | Where called | What bf16 variant is needed |
|---|---|---|
| `embedding_gather_scale_batch_f32` | :175 | `embedding_gather_scale_batch_bf16` — **NOT NEEDED**: embedding output goes to residual (stays f32) |
| `fused_head_norm_rope_batch_f32` | :309, :320 | `fused_head_norm_rope_batch_bf16` — **NEW kernel required in mlx-native** |
| `dispatch_rms_norm_unit_perhead` (f32) | :339, :354 | bf16 variant of unit-perhead rms norm — **NEW kernel required** |
| `fused_gelu_mul` | :544 | `fused_gelu_mul_bf16` — **NEW kernel required in mlx-native** |
| `moe_swiglu_seq_encode` | :610 | bf16 variant of `moe_swiglu_seq` — **NEW kernel required** |
| `moe_weighted_sum_seq_encode` | :660 | bf16 variant (but output is pf_moe_accum which stays f32) — stays f32 |
| `dispatch_fused_norm_add_f32` (post-attn) | :472 | Residual add — residual stays f32, **stays f32** |
| `dispatch_fused_norm_add_f32` (post-FF norm2+combine) | :673 | Mixed: mlp_down_out (bf16) + moe_accum (f32) → f32 — needs **new mixed-dtype fused norm add** or separate casts |
| `dispatch_fused_norm_add_scalar_f32` (end-of-layer) | :688 | Residual + mlp_down (bf16 intermediate) → f32 — needs **bf16-input variant** or cast before |
| `dispatch_copy_f32` (last-row copy) | :924 | Copy from f32 pf_hidden → f32 activations.hidden — **stays f32**, no change |
| `kv_cache_copy_seq_f32` | :758, :767 | Copies f32 pf_k_normed/pf_v_normed to f32 KV cache — becomes `kv_cache_copy_seq_bf16` — **NEW kernel required** |
| `kv_cache_copy_seq_f32_to_f16` | :739, :748 | bf16 pf_k_normed/pf_v_normed → f16 cache — needs `kv_cache_copy_seq_bf16_to_f16` — **NEW kernel required** if use_f16_kv |

### 4.3 Summary: new mlx-native kernels needed

1. `fused_head_norm_rope_batch_bf16` — batched Q/K head norm + RoPE with bf16 I/O
2. `rms_norm_unit_perhead_bf16` — unit RMS norm per head with bf16 I/O (V norm)
3. `fused_gelu_mul_bf16` — GELU(gate) * up with bf16 I/O
4. `moe_swiglu_seq_bf16` — SwiGLU for batched MoE with bf16 I/O
5. `kv_cache_copy_seq_bf16` — copy bf16 K/V normed to bf16 KV cache (f32 kv path)

Optionally (for use_f16_kv path):

6. `kv_cache_copy_seq_bf16_to_f16` — copy bf16 K/V normed to f16 KV cache

**Important design note on `dispatch_fused_norm_add_f32` for post-FF combine
(line 673)**: This call takes `pf_mlp_down_out` (bf16 target) and `pf_moe_accum`
(f32 target) and produces `pf_mlp_down` as the combined result. The cleanest
approach is to:
- Cast `pf_mlp_down_out` bf16 → f32 before this call (using existing
  `dispatch_cast_bf16_to_f32_with_encoder`), OR
- Write a new `dispatch_fused_norm_add_mixed_f32` that takes bf16 input1 and
  f32 input2, OR
- Keep `pf_mlp_down_out` as f32 (it is the output of a rms_norm, so it can
  stay f32 without breaking the bf16 chain). This is the pragmatic path.

The same analysis applies to `dispatch_fused_norm_add_scalar_f32` at line 688:
`pf_mlp_down` (the combined MLP+MoE result) feeds as the second input; if it is
f32, the existing `dispatch_fused_norm_add_scalar_f32` works without change.

**Pragmatic resolution**: Keep `pf_mlp_down_out` and `pf_mlp_down` (in their
role as the post-FF combined output) as f32. Only convert the intermediate
sublayer outputs (Q, K, V, SDPA, MLP gate/up/fused, MoE gate_up/swiglu/down)
to bf16.

---

## 5. Conversion ordering plan (topological)

The data flow graph has these dependencies:

```
pf_hidden (f32) → [rms_norm] → pf_norm_out → [QKV matmul] → pf_q, pf_k, pf_v
pf_q → [head_norm_rope_batch] → pf_q_normed → [permute] → pf_q_perm
pf_k → [head_norm_rope_batch] → pf_k_normed → [permute] → pf_k_perm → [SDPA]
pf_v → [rms_norm_unit_perhead] → pf_v_normed → [permute] → pf_v_perm → [SDPA]
[SDPA] → pf_sdpa_out_perm → [permute] → pf_sdpa_out → [O-proj] → pf_attn_out
pf_attn_out → [fused_norm_add] → pf_residual (f32)
pf_residual → [rms_norm x3] → pf_norm_out, pf_moe_norm_out, pf_router_norm_out
pf_norm_out → [QKV matmul] → pf_mlp_gate, pf_mlp_up
pf_mlp_gate + pf_mlp_up → [fused_gelu_mul] → pf_mlp_fused → [down_proj] → pf_mlp_down
pf_moe_norm_out → [gate_up_id] → pf_moe_gate_up → [swiglu_seq] → pf_moe_swiglu
pf_moe_swiglu → [down_id] → pf_moe_down → [moe_weighted_sum] → pf_moe_accum
pf_mlp_down → [rms_norm] → pf_mlp_down_out (can stay f32 for combine)
pf_mlp_down_out + pf_moe_accum → [fused_norm_add] → pf_mlp_down (combined, f32)
pf_residual + pf_mlp_down → [fused_norm_add_scalar] → pf_hidden (f32)
```

### Staged commit ordering (topological, each stage compile-and-test safe)

**Stage 1: QKV projections output → bf16**
- Change `pf_q`, `pf_k`, `pf_v` allocations to bf16
- `dispatch_qmatmul` for Q/K/V already outputs to the buffer dtype it's given;
  verify `dispatch_qmatmul` in `forward_mlx.rs` selects bf16 output variant
- Test: check pf_q dtype in dump path (disable head_norm_rope for now)

**Stage 2: Head norm + RoPE → bf16 (requires new mlx-native kernel)**
- Add `fused_head_norm_rope_batch_bf16` to mlx-native
- Change `pf_q_normed`, `pf_k_normed`, `pf_v_normed` to bf16
- Change `dispatch_fused_head_norm_rope_batch_f32` → `dispatch_fused_head_norm_rope_batch_bf16`
- Change `dispatch_rms_norm_unit_perhead` to bf16 path for V norm

**Stage 3: Permute Q/K/V → bf16**
- Change `pf_q_perm`, `pf_k_perm`, `pf_v_perm`, `pf_sdpa_out_perm`, `pf_sdpa_out` to bf16
- Change `permute_021_f32` → `permute_021_bf16` for all 4 permute calls
- The sdpa/sdpa_sliding dispatch auto-selects bf16 kernel by buffer dtype

**Stage 4: SDPA output chain → bf16 + switch to flash_attn_prefill**
- Replace `s.sdpa` + `s.sdpa_sliding` with `dispatch_flash_attn_prefill_bf16_d256`
  (for head_dim=256) or bf16_d512 (for head_dim=512)
- `pf_sdpa_out` already bf16 from Stage 3
- Note: `flash_attn_prefill` layout is `[B, H, qL, D]`; current layout is
  `[nh, seq_len, hd]` (after permute); batch=1 so `[1, nh, seq_len, hd]` matches

**Stage 5: O-proj → bf16 + bf16 input to fused_norm_add (with cast)**
- Change `pf_attn_out` to bf16
- `dispatch_qmatmul` for O-proj takes bf16 input (pf_sdpa_out), outputs bf16
- Post-attn `dispatch_fused_norm_add_f32` takes f32 residual (pf_hidden) and bf16 input
  (pf_attn_out); cast pf_attn_out bf16→f32 before the fused_norm_add call using
  `dispatch_cast_bf16_to_f32_with_encoder`, OR write a mixed variant

**Stage 6: Pre-FF norms → bf16 outputs**
- Change `pf_norm_out`, `pf_moe_norm_out`, `pf_router_norm_out` to bf16
- `s.rms_norm` selects kernel by buffer dtype; no code change beyond dtype
- `dispatch_qmatmul` for gate/up/router takes bf16 norm_out as input
- `pf_router_logits` stays f32 (feeds softmax)

**Stage 7: MLP gate/up/fused → bf16 + fused_gelu_mul_bf16 (requires new kernel)**
- Change `pf_mlp_gate`, `pf_mlp_up`, `pf_mlp_fused` to bf16
- Add `fused_gelu_mul_bf16` to mlx-native moe_dispatch.metal
- MLP down proj output (`pf_mlp_down` first use) → bf16 or keep f32

**Stage 8: MoE expert outputs → bf16**
- Change `pf_moe_gate_up`, `pf_moe_swiglu`, `pf_moe_down` to bf16
- Add `moe_swiglu_seq_bf16` to mlx-native
- `moe_weighted_sum_seq_encode` input `pf_moe_down` bf16 → output `pf_moe_accum` f32
  (this kernel needs to accept bf16 input + f32 output; existing f32 version only
  handles f32; new bf16-input variant needed or cast pf_moe_down before)
- `pf_mlp_down_out` (post-FF norm1 output): keep as f32 for simplicity of combine step

**Stage 9: KV cache copy → bf16-aware**
- `pf_k_normed` and `pf_v_normed` are now bf16 (from Stage 2)
- Switch `dispatch_kv_cache_copy_seq_f32` → new `dispatch_kv_cache_copy_seq_bf16`
- Add kernel to mlx-native kv_cache_copy module

**Stage 10: Dump path updates**
- All `pf_*.as_slice::<f32>()` calls in the dump path (lines 219-895) must be
  updated to match the new dtype of each buffer. Buffers that become bf16 need
  bf16 reads. This is mechanical but large.

---

## 6. Residual stream dtype determination

**Decision**: residual stream STAYS f32. The following buffers are residual:

- `pf_hidden` (line 115) — **stays f32** — embeds into this, iterates per layer
- `pf_residual` (line 116) — **stays f32** — post-attention residual accumulation

All other buffers that feed through the post-attention and post-FF fused_norm_add
paths write their output (the new hidden / residual) into these f32 buffers.

**Intermediate buffers that BECOME bf16**:

```
pf_norm_out, pf_moe_norm_out, pf_router_norm_out   (norm outputs)
pf_q, pf_k, pf_v                                    (QKV projections)
pf_q_normed, pf_k_normed, pf_v_normed               (after head norm + RoPE)
pf_q_perm, pf_k_perm, pf_v_perm                     (permuted for SDPA)
pf_sdpa_out_perm, pf_sdpa_out                        (SDPA output)
pf_attn_out                                           (O-proj output)
pf_mlp_gate, pf_mlp_up, pf_mlp_fused                (dense MLP intermediates)
pf_moe_gate_up, pf_moe_swiglu, pf_moe_down          (MoE expert intermediates)
```

**Cast sites (bf16 → f32 before residual add)**:

The `dispatch_fused_norm_add_f32` function at line 472 (post-attn residual) takes
`pf_attn_out` as the "input to normalize" argument. This function currently expects
f32. Two options:

- **Option A (recommended)**: Cast `pf_attn_out` bf16 → f32 into a temporary f32
  buffer immediately before the fused_norm_add call. One extra buffer + one cast
  dispatch per layer.
- **Option B**: Write `dispatch_fused_norm_add_bf16_input_f32_residual` — takes bf16
  `input`, f32 `residual`, writes f32 `output`. This fuses the cast into the norm
  kernel (saves one dispatch per layer but requires new Metal kernel).

Option A (cast + existing kernel) is faster to land for Phase 2.

---

## 7. Session-level / KV cache analysis

### Current KV cache dtype

Looking at `forward_prefill_batched.rs` lines 65-67:
```rust
let use_f16_kv = INVESTIGATION_ENV.f16_kv;
let kv_dtype = if use_f16_kv { DType::F16 } else { DType::F32 };
```

The KV cache allocated at lines 87-94 is **f32 by default** (f16 opt-in via
`HF2Q_F16_KV=1`). This is noted in `forward_prefill.rs` comments
(lines 89-95): f16 KV was tested and regressed parity; f32 is the default.

The `flash_attn_vec` (decode SDPA) uses the dense KV buffers from the prefill
path directly. These are currently allocated as f32/f16 via `kv_dtype`.

**For Phase 2**: the bf16 conversion of activations (pf_q_normed etc.) does NOT
require changing the KV cache dtype. The KV cache copy step just needs to accept
bf16 source (pf_k_normed, pf_v_normed) and write to f32/f16 destination as before.
This is a new kernel (`kv_cache_copy_seq_bf16`) not a dtype change to the cache itself.

**Recommendation**: Keep KV cache as f32 for Phase 2. The bf16 KV regression from
`forward_prefill.rs` has not been root-caused; do not conflate with the bf16
activation conversion.

---

## 8. Test coverage

Current gate tests:
- **sourdough_gate.sh** — 22-token prompt, checks common byte prefix ≥ 3094 vs llama.cpp
- **release-check** — Gate A: hf2q batched prefill tok/s ≥ 130 tok/s (thermal)

Verification approach for each stage:
1. Run sourdough_gate with batched prefill enabled
2. Compare output numerically against f32 baseline (allow bf16 tolerance: max
   element difference ≤ 0.01 in logit space, or same argmax token sequence)
3. For the final stage: compare against MLX-LM reference output to confirm parity

The dump infrastructure (HF2Q_BATCHED_DUMP="layer,tok") already exists for
per-layer, per-token tensor inspection and can be used to verify each stage does
not regress hidden-state values beyond bf16 rounding.

---

## 9. Scope estimation

### LOC changes in hf2q (`forward_prefill_batched.rs`)

- Buffer dtype changes: ~30 lines (change `alloc_f32` to `alloc_bf16` calls)
- Kernel name changes: ~15 lines
- New cast dispatches inserted: ~20 lines (one before each fused_norm_add that
  takes a bf16 intermediate)
- Dump path dtype fixes: ~40 lines (every `as_slice::<f32>` for bf16 buffers)
- New `alloc_bf16` closure alongside `alloc_f32`: ~5 lines

**Estimate: ~110 LOC changes in forward_prefill_batched.rs**

### New kernels in mlx-native

| Kernel | Effort | Notes |
|---|---|---|
| `fused_head_norm_rope_batch_bf16` | 1 day | Port f32 shader to bf16; add registration |
| `rms_norm_unit_perhead_bf16` | 0.5 day | Existing rms_norm_bf16 may handle this already |
| `fused_gelu_mul_bf16` | 0.5 day | Copy f32 shader, change types |
| `moe_swiglu_seq_bf16` | 0.5 day | Copy f32 shader, change types |
| `kv_cache_copy_seq_bf16` | 0.5 day | Copy f32 shader, change input type |
| Registration + dispatch stubs | 0.5 day | Adding to kernel_registry.rs and ops/*.rs |
| **Total mlx-native** | **3.5 days** | |

### Session/KV-cache changes

None required for Phase 2 (KV cache stays f32; only the copy source dtype changes).

### Test changes

~20 LOC (update dump comparisons; add bf16 tolerance assertions).

### Total estimate

| Component | Days |
|---|---|
| mlx-native new kernels (5 kernels) | 3.5 |
| hf2q forward_prefill_batched.rs changes | 1.0 |
| Integration testing + gate validation | 0.5 |
| **Total** | **~5 engineering days** |

---

## 10. Actionable checklist (topological order)

### mlx-native work (must land first — hf2q depends on it)

1. **`/opt/mlx-native/src/shaders/fused_head_norm_rope_f32.metal`** — port to
   `fused_head_norm_rope_bf16.metal` with bf16 I/O, f32 accumulator.
   Register as `fused_head_norm_rope_batch_bf16` in kernel_registry.rs.
   Add `dispatch_fused_head_norm_rope_batch_bf16` to
   `/opt/mlx-native/src/ops/fused_head_norm_rope.rs`.

2. **`/opt/mlx-native/src/ops/rms_norm.rs`** — verify `dispatch_rms_norm_unit_perhead`
   selects `rms_norm_bf16` when output buffer is bf16 (it calls `dispatch_rms_norm`
   which auto-selects by dtype). If so, no new kernel needed — just change buffer dtype.

3. **`/opt/mlx-native/src/shaders/moe_dispatch.metal`** — add `fused_gelu_mul_bf16`
   kernel. Register in kernel_registry.rs. Add dispatch function in
   `/opt/mlx-native/src/ops/moe_dispatch.rs`.

4. **`/opt/mlx-native/src/shaders/moe_dispatch.metal`** — add `moe_swiglu_seq_bf16`
   kernel (copy `moe_swiglu_seq`, change half/bfloat types). Register + dispatch.

5. **`/opt/mlx-native/src/shaders/kv_cache_copy.metal`** — add
   `kv_cache_copy_seq_bf16` kernel (bf16 src → f32 dst, same geometry as f32→f32).
   Register in kernel_registry.rs. Add dispatch in
   `/opt/mlx-native/src/ops/kv_cache_copy.rs`.

### hf2q work (after mlx-native kernels land)

6. **`/opt/hf2q/src/serve/forward_prefill_batched.rs:105-108`** — add `alloc_bf16`
   closure alongside `alloc_f32`.

7. **`:115-119, 120-121`** — change `pf_hidden`, `pf_residual` to stay f32;
   change `pf_norm_out`, `pf_moe_norm_out`, `pf_router_norm_out`, `pf_attn_out`,
   `pf_mlp_down_out` to `alloc_bf16`.

8. **`:123-128`** — change `pf_q`, `pf_k`, `pf_v`, `pf_q_normed`, `pf_k_normed`,
   `pf_v_normed` to `alloc_bf16`.

9. **`:130-134`** — change `pf_q_perm`, `pf_k_perm`, `pf_v_perm`,
   `pf_sdpa_out_perm`, `pf_sdpa_out` to `alloc_bf16`.

10. **`:136-139`** — change `pf_mlp_gate`, `pf_mlp_up`, `pf_mlp_fused` to `alloc_bf16`;
    keep `pf_mlp_down` as f32 (it's the combined MLP+MoE result feeding the residual).

11. **`:146-148`** — change `pf_moe_gate_up`, `pf_moe_swiglu`, `pf_moe_down`
    to `alloc_bf16`.

12. **`:309, :320`** — change `dispatch_fused_head_norm_rope_batch_f32` →
    `dispatch_fused_head_norm_rope_batch_bf16` for Q and K.

13. **`:371-385`** — change all four `permute_021_f32` calls → `permute_021_bf16`.

14. **`:420-443`** — replace `sdpa_sliding` + `s.sdpa` calls with
    `dispatch_flash_attn_prefill_bf16_d256` (head_dim=256) or `_d512`; adjust
    buffer layout from `[nh, seq, hd]` to `[1, nh, seq, hd]` (add batch dim).

15. **`:451-455`** — change back-permute `permute_021_f32` → `permute_021_bf16`.

16. **`:462-464`** — O-proj `dispatch_qmatmul`: input is now bf16 `pf_sdpa_out`;
    verify `dispatch_qmatmul` in `forward_mlx.rs` handles bf16 input correctly.
    Output into `pf_attn_out` (bf16).

17. **`:466-479`** — Insert `dispatch_cast_bf16_to_f32_with_encoder` to cast
    `pf_attn_out` (bf16) → a temporary f32 buffer, then call
    `dispatch_fused_norm_add_f32` with the f32 temporary. Alternatively, if
    `dispatch_fused_norm_add_bf16` already takes the `input` arg as bf16 and
    `residual` as bf16, restructure to use the bf16 variant. But the residual
    MUST be f32. **Cast is the clean path**.

18. **`:544-557`** — replace `fused_gelu_mul` pipeline call with `fused_gelu_mul_bf16`.
    Inputs `pf_mlp_gate`, `pf_mlp_up` and output `pf_mlp_fused` are all bf16.

19. **`:572-574`** — MLP down proj: input `pf_mlp_fused` (bf16), output
    `pf_mlp_down` (f32 — feeds the fused_norm_add_scalar residual update);
    verify `dispatch_qmatmul` can take bf16 input + f32 output.

20. **`:588-603`** — gate_up_id: input `pf_moe_norm_out` (bf16), output
    `pf_moe_gate_up` (bf16); verify `quantized_matmul_id_ggml` outputs bf16.

21. **`:610-615`** — change `moe_swiglu_seq_encode` → `moe_swiglu_seq_bf16_encode`.

22. **`:624-639`** — down_id: input `pf_moe_swiglu` (bf16), output `pf_moe_down`
    (bf16); verify.

23. **`:646-652`** — post-FF norm1 on `pf_mlp_down` (f32) → output `pf_mlp_down_out`
    (bf16); `s.rms_norm` auto-selects by output dtype — change `pf_mlp_down_out`
    alloc to bf16, done.

24. **`:660-666`** — `moe_weighted_sum_seq_encode`: input `pf_moe_down` (bf16),
    weights `pf_routing_weights` (f32), output `pf_moe_accum` (f32); new kernel
    variant needs bf16 expert_outputs input.

25. **`:673-680`** — `dispatch_fused_norm_add_f32`: input `pf_mlp_down_out` (bf16)
    + `pf_moe_accum` (f32) → `pf_mlp_down` (f32). Cast `pf_mlp_down_out` bf16→f32
    before this call.

26. **`:688-697`** — `dispatch_fused_norm_add_scalar_f32`: inputs `pf_residual` (f32)
    + `pf_mlp_down` (f32) → `pf_hidden` (f32). No change needed here.

27. **`:734-775`** — KV cache copy: `pf_k_normed` and `pf_v_normed` are now bf16.
    Change `dispatch_kv_cache_copy_seq_f32` → `dispatch_kv_cache_copy_seq_bf16`
    and `dispatch_kv_cache_copy_seq_f32_to_f16` → `dispatch_kv_cache_copy_seq_bf16_to_f16`.

28. **`:786-908`** — dump path: update all `pf_*.as_slice::<f32>()` for buffers
    that became bf16 to use `as_slice::<half::bf16>()` (or the mlx-native bf16
    read API). This is diagnostic code only; can be done in a separate cleanup commit.

---

## 11. Open questions / blockers

1. **`dispatch_qmatmul` bf16 output**: In `forward_mlx.rs`, `dispatch_qmatmul`
   currently always produces f32 output. Needs verification. If it hardcodes f32,
   it must be updated to check the output buffer dtype.

2. **`quantized_matmul_id_ggml` bf16 output**: The GGML matmul-id kernel output
   dtype may be hardcoded to f32. Needs verification against
   `/opt/mlx-native/src/ops/quantized_matmul_id_ggml.rs`.

3. **`flash_attn_prefill` layout**: The kernel expects `[B, H, qL, D]` layout.
   The current prefill path uses a permute to produce `[nh, seq, hd]`. The
   flash_attn_prefill kernel takes this directly as `[1, nh, seq, hd]` (batch=1).
   The intermediate permute steps (items 13, 15 in checklist) may become
   **unnecessary** once flash_attn_prefill is used — the kernel accepts
   `[seq, nh, hd]` if strides are set correctly. Verify stride parameters.

4. **`moe_weighted_sum_seq_encode` bf16 input**: No bf16-input variant exists.
   Either add one or cast `pf_moe_down` to f32 before this call.

5. **`pf_mlp_down` dual role**: The buffer is used as down projection output
   (line 574, should be bf16) AND as the end-of-layer combined MLP+MoE result
   (lines 671, 688, must be f32). Either use two separate buffers or accept that
   `pf_mlp_down` stays f32 throughout and the MLP down projection output must
   be cast to f32 immediately.

---

## 12. Confidence notes

- **f32 site inventory**: high confidence (direct line-by-line read of 1027-line file)
- **mlx-native bf16 kernel existence**: high confidence (grep of kernel_registry.rs)
- **Missing kernel list**: high confidence (no bf16 variants found for listed kernels)
- **dispatch_qmatmul bf16 output**: unverified — forward_mlx.rs dispatch_qmatmul
  code was not fully read in this session; treat as a blocker until confirmed
- **KV cache current dtype**: confirmed f32 default, f16 opt-in, at lines 65-67
- **Residual stream stays f32**: confirmed by both forward_prefill.rs and the
  MLX-LM convention documented in ADR-011

---

*Research-only document. No code changes made.*
