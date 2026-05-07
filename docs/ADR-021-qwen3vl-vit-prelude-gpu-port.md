# ADR-021: Qwen3-VL ViT Prelude → GPU Port

**Status**: In Progress — user pivoted 2026-05-07 to begin ADR-021 ahead of ADR-020 iters 10–19 closure. Iter-1a baseline first.

**Driver**: User mission — "fall back is a forbidden term"; "GPU EVERYWHERE we can; CPU == poop slow"; "we'd never want a user to use CPU for something that should be GPU." `compute_vision_embeddings_gpu_qwen3vl` (the production GPU vision path for Qwen3-VL) currently calls 5 CPU functions in the middle of an otherwise-GPU pipeline. Pre-dates ADR-020 §11 "no external tools / no CPU fallback." Misaligned architecturally.

**Predecessors**: ADR-020 §11 "No external tools" (the canonical statement of the principle this ADR enforces); memory `feedback_no_external_tools.md`.

**Note on numbering**: The H2O probe (rejected 2026-05-07) was informally labeled "ADR-021" in memory file `project_adr021_h2o_REJECTED_2026_05_07.md` but no `docs/ADR-021-*.md` was ever created. This ADR claims the ADR-021 doc number; the H2O memory's "ADR-021" label was always a placeholder.

---

## 1. Why (problem)

`compute_vision_embeddings_gpu_qwen3vl` at `src/inference/vision/vit_gpu_qwen3vl.rs:2086` is named "GPU" but executes a non-trivial CPU prelude before each ViT block forward. Today's flow:

```
[3, H, W] f32 pixels (CPU)
   ↓ qwen3vl_dual_conv_patch_embed_cpu_hw                      ← CPU heavy
   ↓ qwen3vl_resize_position_embeddings_bilinear               ← CPU
   ↓ inline elementwise add of resized pos-embd into patches   ← CPU
   ↓ qwen3vl_2x2_block_merge_reshape                           ← CPU
   ↓ Vec<f32> → upload_f32_to_gpu (line 2320-2325)             ← CPU→GPU readback round-trip
   ↓ Stage B: per-block ViT forward                            ← GPU
   ↓ post-LN + main mm.0/mm.2 GELU projector                   ← GPU
   ↓ readback main + deepstack heads to Vec<f32>               ← GPU→CPU
   ↓ qwen3vl_concat_augmented_embed_cpu                        ← CPU
   ↓ Vec<Vec<f32>> return to engine seam                       ← CPU
```

The Vec<f32> round-trips at lines 2325 + 2522-2552 break the otherwise-contiguous GPU chain. The dual conv at line 2241 is the heavy one — O(num_patches × hidden × 3 × patch²) per stem × 2 stems. For a 1568×1568 (Qwen3-VL canonical) image at patch_size=16, that's:

- num_patches = (1568/16)² = 9604
- hidden = 1280 (Qwen3-VL n_embd)
- 3 × 16² = 768

→ 9604 × 1280 × 768 × 2 = **18.9 G multiply-adds per image, on a single CPU thread**. That dominates Stage A wall.

### CPU surface (verified 2026-05-07)

| # | Line | Function | Compute weight | Ports to |
|---|---|---|---|---|
| 1 | 2241 | `qwen3vl_dual_conv_patch_embed_cpu_hw` (calls `vit::patch_embed_forward_hw` 2× + sum + bias) | **Heavy** — O(num_patches × hidden × 3 × patch²) | im2col + matmul + bias_add |
| 2 | 2261 | `qwen3vl_resize_position_embeddings_bilinear` | Light — O(n_pos_pre × hidden) | bilinear-resize-2D Metal kernel |
| 3 | 2277-2280 | inline pos-embd add | Trivial — element-wise | reuse `dispatch_elementwise_add` |
| 4 | 2288 | `qwen3vl_2x2_block_merge_reshape` | Trivial — O(n_pos × hidden) memcpy with stride permute | block-merge Metal kernel |
| 5 | 2555 | `qwen3vl_concat_augmented_embed_cpu` | Trivial — O(n_image_tokens × augmented_dim) memcpy | feature-concat Metal kernel |

---

## 2. What (proposed solution)

Replace the 5 CPU steps with 4 new Metal kernels (#3 reuses an existing one) so that `compute_vision_embeddings_gpu_qwen3vl` becomes a single-encoder GPU graph from `[3, H, W]` pixel upload through deepstack-augmented embedding readback. The existing 5 CPU functions migrate to `#[cfg(test)]` only — they remain as byte-identical oracles for the new GPU kernels (the same pattern ADR-020's `src/calibrate/autograd.rs` uses).

### New kernels (mlx-native)

**K1 — `im2col_2d_3ch_f32`**: input `[3, H, W]` row-major fp32 → output `[num_patches, 3*p²]` row-major fp32 with each row being one unfolded patch in (channel-major, dy-major, dx-major) order matching `patch_embed_forward_hw`'s inner kernel iteration. One threadgroup per output row; each thread copies one element.

After im2col, the dual conv reduces to:
```
patches_pre[m, n] = bias[n]
                  + Σ_k im2col[m, k] · weight_0[n, k]
                  + Σ_k im2col[m, k] · weight_1[n, k]
```
which is **two `dense_matmul_f32_f32_tensor` dispatches into the same accumulator, then a single elementwise bias broadcast**. K = 3·p² = 768 (Qwen3-VL patch_size=16) clears the K≥32 constraint.

**K2 — `bilinear_resize_2d_f32`**: input `[H_src, W_src, C]` row-major fp32 → output `[H_dst, W_dst, C]` row-major fp32 via 4-tap bilinear filter. One threadgroup per (dst_y, dst_x) tile; each thread handles one channel. Mirrors `qwen3vl_resize_position_embeddings_bilinear` semantics exactly (corner alignment + sample weights match qwen3vl.cpp:47).

**K3 — reuse `dispatch_elementwise_add`** (already in mlx-native).

**K4 — `block_merge_2x2_f32`**: input `[ny, nx, n_embd]` row-major fp32 → output `[ny*nx, n_embd]` row-major in 2×2-block-major-then-row-major order. The exact permutation in `qwen3vl_2x2_block_merge_reshape` (line 488-560 of vit_gpu_qwen3vl.rs). One threadgroup per output patch; each thread copies one element of `n_embd`.

**K5 — `feature_concat_f32`**: variable-arity feature-axis concat. Input is a list of `[T, D]` slabs; output is `[T, D_total]` row-major where `D_total = Σ D_i`. One threadgroup per (token, slab); within a slab each thread handles one feature element.

### Wiring change in `compute_vision_embeddings_gpu_qwen3vl`

The function gains a single `MlxBuffer pixel_values_gpu` upload at function entry (replacing the late upload at line 2320-2325) and threads the encoder through Stage A. The 5 CPU calls are replaced 1:1 by their GPU dispatch counterparts, all sharing `session.encoder_mut()`. No memory_barrier() additions beyond what ADR-019 already requires between RAW dependencies.

The `Vec<f32>` readback at 2522-2552 collapses to a single `feature_concat_f32` dispatch + one final `as_slice::<f32>()` at function return.

---

## 3. Acceptance Criteria

- **AC-1 (parity)**: For each new kernel K1–K5, a `#[cfg(test)]` test in mlx-native asserts `gpu_output == cpu_oracle_output` byte-identically (or fp-identical to within 1 ULP for K2 bilinear, which has float-sum-order ambiguity). Oracles call directly into the existing `qwen3vl_*_cpu*` functions in vit_gpu_qwen3vl.rs.
- **AC-2 (e2e parity)**: A `#[cfg(test)]` test exercises `compute_vision_embeddings_gpu_qwen3vl` against a fixture image and asserts byte-identity vs the pre-port output captured at the head of this ADR's branch (golden f32 baseline).
- **AC-3 (live-check)**: Three multimodal Qwen3-VL inference requests (different image sizes — square, wide, tall — chosen so each exercises rectangular `pixel_h × pixel_w` paths) produce coherent generated text. Quote literal output bytes per `feedback_live_verification_must_check_content`.
- **AC-4 (no-fallback)**: After the port lands, `grep -n "_cpu\|_cpu_hw" src/inference/vision/vit_gpu_qwen3vl.rs` returns ONLY `#[cfg(test)]`-gated definitions and call sites. No production codepath reaches a CPU function.
- **AC-5 (cleanup)**: `vit::patch_embed_forward_hw` stays public for the Gemma-4v vision path (its own concern); only the qwen3vl entry points get the `#[cfg(test)]` lift.
- **AC-6 (perf)**: Stage A wall on a 1568×1568 image drops from current ~CPU-bound (TBD measure) to GPU-bound. No regression on Stage B per-block forward (the work the port does NOT touch).

---

## 4. Iteration Plan

| # | Sub | Description | Status |
|---|---|---|---|
| 1 | a | Capture pre-port byte-identical golden via test on current `main`; commit baseline fixture | LANDED on branch `adr-021/impl` worktree `/tmp/hf2q-adr-021`. Test `adr021_iter1a_e2e_byte_pinned_baseline_2026_05_07` pins fnv1a64=`0xf1a71d67_3b0b5891`, len=1024, first8 + last8 f32 bit patterns. Captured from CPU-prelude path on commit 5f2ba02. |
| 1 | b | Add K1 `im2col_2d_3ch_f32` Metal kernel + Rust dispatch + parity test | pending |
| 2 | a | Wire K1 + 2× existing `dense_matmul_f32_f32_tensor` + bias_add to replace `qwen3vl_dual_conv_patch_embed_cpu_hw` at line 2241; AC-1 + AC-2 byte-identical | pending |
| 3 | a | Add K2 `bilinear_resize_2d_f32` + parity test; replace `qwen3vl_resize_position_embeddings_bilinear` at 2261 | pending |
| 3 | b | Replace inline pos-embd add at 2277-2280 with `dispatch_elementwise_add` | pending |
| 4 | a | Add K4 `block_merge_2x2_f32` + parity test; replace `qwen3vl_2x2_block_merge_reshape` at 2288 | pending |
| 4 | b | Add K5 `feature_concat_f32` + parity test; replace `qwen3vl_concat_augmented_embed_cpu` at 2555 + collapse Vec<f32> readback loop | pending |
| 5 | a | Lift `qwen3vl_dual_conv_patch_embed_cpu` + `_cpu_hw` + `qwen3vl_2x2_block_merge_reshape` + `qwen3vl_resize_position_embeddings_bilinear` + `qwen3vl_concat_augmented_embed_cpu` to `#[cfg(test)]` only | pending |
| 5 | b | AC-3 live-check: square + wide + tall multimodal Qwen3-VL inference; quote literal output | pending |
| 5 | c | AC-6 perf measurement; ADR closure | pending |

---

## 5. Out of Scope

- **Gemma-4v vision path** (`vit_gpu_gemma4v.rs` and `vit::patch_embed_forward_hw`'s Gemma-4v call sites). Different family — its own ADR if needed.
- **Generic conv2d kernel.** We exploit the patch-conv structure (3 input channels, square kernel = patch_size, stride = patch_size, no padding, no dilation) to avoid building a general conv2d. A future generic conv2d is a separate effort.
- **F16/BF16 paths.** The vision-prelude tensors are fp32 today; we keep that. Quantizing the ViT itself is its own concern.

---

## 6. References

- `/opt/llama.cpp/tools/mtmd/models/qwen3vl.cpp` — graph builder (lines 16-186 are the prelude + per-block + projector this ADR mirrors).
- `src/inference/vision/vit_gpu_qwen3vl.rs:2086-2567` — the function we're porting.
- `src/inference/vision/vit.rs:598-687` — `patch_embed_forward_hw` (the kernel we replace with im2col + matmul).
- ADR-020 §11 — "No external tools" — the principle this ADR enforces.
- `feedback_no_external_tools.md` — memory pin for the principle.
