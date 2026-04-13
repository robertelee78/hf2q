# Spike: Barrier + Buffer Alias Audit (2026-04-13)

**Goal:** Verify no cross-layer or intra-layer buffer aliasing hazards exist before any barrier structural changes.

---

## Buffer Allocation Model

Every activation buffer in `MlxActivationBuffers` is a **unique `MTLBuffer`** allocated via `device.new_buffer()` (`MTLResourceOptions::StorageModeShared`). There is:
- No buffer pool
- No sub-allocation from a shared arena
- No power-of-two bucketing
- No offset-based aliasing

Each field name (`hidden`, `norm_out`, `attn_out`, etc.) maps 1:1 to a distinct physical memory range. Two buffers with different field names cannot alias.

Source: `alloc_f32` closure at `forward_mlx.rs:3626` calls `device.alloc_buffer()` which calls `self.device.new_buffer()` at `mlx-native/src/device.rs:71`. `MlxBuffer::from_raw` at `mlx-native/src/buffer.rs:49` stores the raw `MetalBuffer` with no offset.

---

## Dispatch Model

Single `CommandEncoder` with `MTLDispatchTypeConcurrent` for the entire forward pass (all 30 layers + embedding + head). One `begin()` at top, one `finish_with_timing()` at bottom. Barriers use `memoryBarrierWithScope:MTLBarrierScopeBuffers`.

---

## Per-Layer Buffer Flow (20 barriers)

Each row: barrier number, dispatches before it, buffers written, buffers read.

| B# | Line | Dispatches since prev barrier | Writes to | Reads from | Required? |
|----|------|-----|-----------|------------|-----------|
| 0  | 1020 | 1 (embedding) | `hidden` | `embed_weight` | Yes: L0 rms_norm reads `hidden` |
| 1  | 1043 | 1 (rms_norm) | `norm_out` | `hidden` | Yes: QKV read `norm_out` |
| 2  | 1061 | 2-3 (Q/K/V proj) | `attn_q`, `attn_k`, `attn_v` | `norm_out` | Yes: head norms read Q/K/V |
| 3  | 1138 | 3 (Q norm+rope, K norm+rope, V norm) | `attn_q_normed`, `attn_k_normed`, `attn_v`/`moe_expert_out` | `attn_q`, `attn_k` | Yes: KV cache reads normed K/V |
| 4  | 1165 | 2 (KV cache K, KV cache V) | `kv_caches[L].k`, `kv_caches[L].v` | `attn_k_normed`, v_src | Yes: SDPA reads KV cache |
| 5  | 1190 | 2 (flash_attn main + reduce) | `sdpa_out`, `flash_attn_tmp` | `attn_q_normed`, `kv_caches[L].k`, `kv_caches[L].v` | Yes: O-proj reads `sdpa_out` |
| 6  | 1198 | 1 (O-proj) | `attn_out` | `sdpa_out` | Yes: fused_norm_add reads `attn_out` |
| 7  | 1214 | 1 (fused_norm_add) | `residual` | `hidden`, `attn_out` | Yes: pre-FF norm reads `residual` |
| 8  | 1228 | 1 (pre-FF norm) | `norm_out` | `residual` | Yes: gate/up read `norm_out` |
| 9  | 1239 | 2 (gate proj, up proj) | `mlp_gate`, `mlp_up` | `norm_out` | Yes: fused_gelu reads gate+up |
| 10 | 1261 | 1 (fused_gelu_mul) | `mlp_fused` | `mlp_gate`, `mlp_up` | Yes: down_proj reads `mlp_fused` |
| 11 | 1268 | 1 (down_proj) | `mlp_down` | `mlp_fused` | Yes: post-FF norm reads `mlp_down` |
| -- | -- | 3 (post-FF norm1 + pre-FF norm2 + router norm) | `attn_out`, `moe_norm_out`, `norm_out` | `mlp_down`, `residual` | No barrier needed: disjoint write sets |
| 12 | 1310 | 0 (logically after the 3 barrier-free norms) | -- | -- | Yes: router_proj reads `norm_out` written by router norm |
| 13 | 1318 | 1 (router_proj) | `moe_router_logits` | `norm_out` | Yes: fused routing reads `moe_router_logits` |
| 14 | 1337 | 1 (fused_moe_routing) | `moe_expert_ids`, `moe_routing_weights_gpu` | `moe_router_logits` | Yes: gate_up_id reads expert_ids |
| 15 | 1373 | 1 (gate_up_id) | `moe_gate_up_id_out` | `moe_norm_out`, `moe_expert_ids` | Yes: SwiGLU reads gate_up_id_out |
| 16 | 1385 | 1 (moe_swiglu_batch) | `moe_swiglu_id_out` | `moe_gate_up_id_out` | Yes: down_id reads swiglu_out |
| 17 | 1408 | 1 (down_id) | `moe_down_id_out` | `moe_swiglu_id_out`, `moe_expert_ids` | Yes: weighted_sum reads down_id_out |
| 18 | 1438 | 1 (weighted_sum) | `moe_accum` | `moe_down_id_out`, `moe_routing_weights_gpu` | Yes: fused norm2+combine reads `moe_accum` |
| 19 | 1458 | 1 (fused_norm_add post-MoE) | `mlp_down` | `attn_out`, `moe_accum` | Yes: end-of-layer reads `mlp_down` |
| 20 | 1478 | 1 (fused_norm_add_scalar) | `hidden` | `residual`, `mlp_down` | Yes: next layer rms_norm reads `hidden` |

---

## Cross-Layer Analysis

### Layer N → Layer N+1 boundary

**Last write of layer N:** `hidden` (at barrier 20, line 1478)
**First read of layer N+1:** `hidden` (rms_norm at line 1034, before barrier 1)

Barrier 20 at the end of layer N ensures `hidden` is fully written before the next iteration's rms_norm reads it. **Correct RAW ordering.**

### Buffer reuse across layers

These buffers are written and then re-read/re-written every layer:

| Buffer | First write in layer | Last read in layer | Cross-layer hazard? |
|--------|---------------------|-------------------|-------------------|
| `norm_out` | B1 (rms_norm) | B8 (pre-FF norm) and B12 (router proj) | No: B20 serializes all layer N work before N+1 starts |
| `attn_out` | B6 (O-proj) | B19 (fused_norm_add post-MoE reads it) | No: same reasoning |
| `residual` | B7 (fused_norm_add) | B20 (end-of-layer reads it) | No: same reasoning |
| `mlp_down` | B11 (down_proj) | B20 (end-of-layer reads it) | No: same reasoning |
| `hidden` | B20 (end-of-layer) | B1 (next layer rms_norm) | No: B20 serializes |

**Key invariant:** Barrier 20 at the end of every layer iteration is a full serialization point. All dispatches from layer N are ordered before any dispatch of layer N+1 begins (because barrier 20 → next iteration's rms_norm → barrier 1 is a strict chain with no concurrent dispatches crossing the loop boundary).

### Intra-layer reuse (same buffer written twice)

| Buffer | First write | Second write | Hazard? |
|--------|------------|-------------|---------|
| `norm_out` | B1 (pre-attn norm) | B8 (pre-FF norm), then router norm (no barrier) | No: first write consumed by B2's QKV (barrier 1 ensures). Second write consumed by B9's gate/up (barrier 8 ensures). Router norm's write consumed by B13's router proj (barrier 12 ensures). Each write is fenced by a barrier before the next write occurs. |
| `attn_out` | B6 (O-proj) | post-FF norm 1 (after B11, no barrier before) | No: O-proj's `attn_out` consumed by B7's fused_norm_add (barrier 6 ensures). Post-FF norm 1's `attn_out` consumed by B19's fused_norm_add (barrier 18 ensures). The intervening barriers serialize the writes. |
| `mlp_down` | B11 (down_proj) | B19 (fused_norm_add post-MoE) | No: B11's `mlp_down` consumed by post-FF norm 1 (barrier 11 ensures). B19's `mlp_down` consumed by B20's end-of-layer (barrier 19 ensures). |

---

## Verdict

**No aliasing hazards exist.** All buffers are physically distinct. All cross-layer dependencies are correctly serialized by barrier 20. All intra-layer buffer reuse is correctly fenced by intervening barriers.

The existing barrier placement is both necessary AND sufficient for correctness.

---

## Barrier Profile Comparison (measured 2026-04-13)

| Metric | hf2q (mlx-native) | llama.cpp |
|--------|-------------------|-----------|
| Dispatches/token | 872 | ~1,811 (181 + 1630) |
| Barriers/token | 606 (20/layer x 30 + 6) | 759 (74 + 685) |
| Dispatches/barrier ratio | 1.44 | 2.38 |
| GPU wait/token | 10.65ms | ~9.5ms (estimated) |
| Total/token | 11.15ms | ~9.8ms |
| tok/s (128 tokens) | 90.1 | 102 |

**The barrier count is NOT the bottleneck.** llama.cpp uses MORE barriers (759 vs 606) and is still faster. The gap is in GPU utilization — llama.cpp groups 2.38 dispatches between barriers on average vs our 1.44, giving more GPU pipelining opportunities.

---

## Recommendations (safe structural changes)

1. **Kernel fusion** to eliminate 1-dispatch-per-barrier chains. Each fusion removes both a dispatch and a barrier, improving the dispatches/barrier ratio. No correctness risk because the barrier is eliminated along with the separate dispatch.

2. **Do NOT remove barriers without fusing the surrounding kernels.** Every barrier protects a true RAW dependency.
