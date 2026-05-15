# ADR-029 iter-175 — Step 1: per-kernel dispatch-count baseline at HEAD

**Date**: 2026-05-15
**HEAD**: `f3aa12ea` (post-ADR-031 close + synthesis)
**Hardware**: M5 Max (Apple Silicon, AGXG17XFamilyComputeContext)
**Model**: `gemma4-ara-2pass-APEX-Q5_K_M.gguf`
**Bench**: `MLX_DISP_BUCKET=1 HF2Q_DUMP_COUNTERS=1 hf2q generate --prompt "Q." --max-tokens 100 --temperature 0 --ignore-eos`

## Summary

Per-kernel dispatch distribution at HEAD is **highly concentrated** (top 2 = 37%, top 14 = 90%) across **53 unique pipelines / 861 dispatches per decode token**. Decode throughput on this run: 94.5 t/s (~0.99× of the 95.86 standing baseline; small thermal-warm-up drag).

**Key finding for iter-175 planning**: the top two hottest kernels (`kernel_mul_mv_q6_K_f32_nr2`, `rms_norm_f32_v2`) are **already functionally peer-equivalent** to llama.cpp's corresponding implementations. The classical "port peer pattern" lever (H93 FC-promote template) is largely tapped on the hottest kernels — additional gains likely require investigating ENCODER overhead, compile flags, or memory layout, not kernel source ports.

## Per-pipeline breakdown

Aggregate (15 prompt + 100 decode tokens → ~99% decode):

| Rank | Count | %     | Pipeline                                                                          | Notes                                                          |
|------|-------|-------|-----------------------------------------------------------------------------------|----------------------------------------------------------------|
| 1    | 17425 | 19.91 | `kernel_mul_mv_q6_K_f32_nr2` (FC: `700:i1 701:i1 702:i1`)                          | Already iter-308/352/401 peer-ported (nr0=2, short indexing)   |
| 2    | 14980 | 17.11 | `rms_norm_f32_v2`                                                                 | Already peer-pattern (float4 + simd_sum + fused weight mul)    |
| 3    | 5940  | 6.79  | `fused_head_norm_rope_f32_v2`                                                     | hf2q-specific fusion (Q,K norm + RoPE in 1 kernel)             |
| 4    | 4950  | 5.65  | `hadamard_quantize_kv_fast_d256`                                                  | hf2q-specific (FA hybrid V-quant path, no peer equivalent)     |
| 5    | 3000  | 3.43  | `kernel_mul_mv_id_q6_K_f32_nr2`                                                   | MoE-id Q6_K — same algo as #1, gather-routed                   |
| 6    | 3000  | 3.43  | `fused_gelu_mul`                                                                  | hf2q-specific (gelu(x) * y in 1 kernel)                        |
| 7    | 3000  | 3.43  | `fused_norm_add_f32_v2`                                                           | hf2q-specific (norm + add fusion)                              |
| 8    | 2970  | 3.39  | `fused_moe_routing_f32_v2`                                                        | hf2q-specific (MoE softmax + top-K)                            |
| 9    | 2970  | 3.39  | `moe_weighted_sum`                                                                | hf2q-specific (expert combine)                                 |
| 10   | 2970  | 3.39  | `rms_norm_no_scale_f32_v2`                                                        | Same as #2 minus the weight mul                                |
| 11   | 2970  | 3.39  | `moe_swiglu_batch`                                                                | hf2q-specific (MoE SwiGLU)                                     |
| 12   | 2970  | 3.39  | `dense_matvec_f32`                                                                | (lm_head residual path)                                        |
| 13   | 2970  | 3.39  | `kernel_mul_mv_q8_0_f32` (FC: `700:i1 701:i1 702:i1`)                              | Q8_0 dense matvec                                              |
| 14   | 2970  | 3.39  | `fused_post_ff_norm2_endlayer_f32_v2`                                             | hf2q-specific 3-op end-of-layer fusion                         |
| 15   | 2475  | 2.83  | `kv_copy_kf16_quantize_v_no_fwht_d256`                                            | hf2q-specific (KV append + V-quant)                            |
| 16   | 2475  | 2.83  | `flash_attn_vec_hybrid_dk256` (FC: `50:i8 51:i0`)                                  | hf2q-specific FA-vec hybrid (V quantized, K f16)               |
| 17   | 2475  | 2.83  | `flash_attn_vec_reduce_dk256`                                                     | hf2q FA-vec reduce phase                                       |

**Long tail**: 36 more kernels, each ≤1.1%, totalling ~10% of dispatches. Mostly id-variants of matvecs and prefill kernels.

## Per-token derived rates

Total: 861.3 dispatches / decode_tok, 478.2 barriers / decode_tok (slight drift from iter-115's 866/420 — different bench prompt + system state). 99% decode purity (15 prefill tokens dilute by ~1.7%).

By kernel-class:
- Quantized matvecs: 174 q6_K_nr2 + 30 q8_0 + 30 id-q6_K + 10 id-q8_0 + 10 id-q5_1 + 10 id-iq4_nl + 2 mm dense = ~266/tok
- RMS-norm family: 150 rms_norm_v2 + 30 rms_norm_no_scale_v2 = ~180/tok (~5 per gemma4 layer)
- FA family: 25 hybrid_dk256 + 25 reduce_dk256 + 5 hybrid_dk512 + 5 reduce_dk512 = ~60/tok
- KV append/quant: 50 hadamard_d256 + 10 hadamard_d512 + 25 kv_copy = ~85/tok
- MoE: 30 routing + 30 weighted_sum + 30 swiglu = ~90/tok
- Misc fused: head_norm_rope, gelu_mul, norm_add, post_ff = ~180/tok

## Side-by-side: #1 hottest kernel (already peer-ported)

`kernel_mul_mv_q6_K_f32_nr2` (`/opt/mlx-native/src/shaders/quantized_matmul_ggml.metal:713`) vs llama.cpp `kernel_mul_mv_q6_K_f32_impl` (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:7970`):

| Aspect                                  | hf2q                                                            | llama.cpp                                                  |
|-----------------------------------------|------------------------------------------------------------------|------------------------------------------------------------|
| NSG (simdgroups/tg)                     | constexpr `2`                                                    | `FC_mul_mv_nsg` (FC-baked = 2 at build)                    |
| nr0 (rows/SG)                           | constexpr `2`                                                    | template `N_R0_Q6_K` (= 2)                                 |
| Y vector cache (`yl[16]`)               | YES — load once per block, reuse across rows                     | YES — same pattern                                         |
| Inner block-unpack loop                 | plain `for (l=0;l<4;l++)`                                        | `FOR_UNROLL` macro                                         |
| Row stride access pattern               | `xr = x_base + row * nb` (recompute per row)                     | `q1 += args.nb01; q2 += ...; dh += args.nb01/2` (increment)|
| Indexing types                          | `short` for sub-indices (iter-401)                               | `short` for sub-indices                                    |

Differences are micro-optimizations. iter-352 tested `FOR_UNROLL` equivalent on hf2q and found it **FALSIFIED** (Apple Metal auto-unroll did better). Pointer-increment vs row-mul is unlikely to move the needle — Metal compilers fold address arithmetic.

## Side-by-side: #2 hottest kernel (already peer-ported)

`rms_norm_f32_v2` (`/opt/mlx-native/src/shaders/rms_norm.metal:81`) vs llama.cpp `kernel_rms_norm_fuse_impl<float4, 2>` (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:2989`):

| Aspect                  | hf2q rms_norm_f32_v2          | llama.cpp F=2 (rms_norm + mul)   |
|-------------------------|--------------------------------|-----------------------------------|
| Vector type             | `float4`                       | `T = float4` template inst.       |
| Reduce inputs           | `simd_sum` per SG              | `simd_sum` per SG                 |
| Cross-SG reduce         | shared mem + simd_sum          | shared mem + simd_sum             |
| Weight fusion           | YES — second loop: `*weight[i]`| YES — F=2 path                    |
| Add fusion (residual)   | NO — separate kernel           | F=3 path: `*scale * f0[i] + f1[i]`|

hf2q's `rms_norm_f32_v2` ≈ peer's F=2. The remaining fusion opportunity is **F=3 equivalent** (rms_norm + mul + residual_add), which would consume `fused_norm_add_f32_v2` dispatches (30/tok). But this is only 3.4% of dispatches, not the structural lever.

## What this tells us about the gap

iter-308 (q6_K_nr2 port) and iter-162 (FC-promote port) both landed real wins by porting peer patterns to hottest kernels. **Those wins are still in HEAD.** The current top kernels are not the next easy win.

The remaining 6-8% gap to peer-FA is therefore NOT explained by "kernel X has a different algorithm." It must be one of:

**H94 (next iteration's hypothesis candidates)**:
- **H-A**: Per-dispatch encoder/framework overhead (argument-buffer binding, residency, validation). Peer has 1339 dispatches/tok yet runs faster → per-dispatch overhead is lower in peer. Lever: profile encoder fast-path; compare to peer's command-encoder pattern.
- **H-B**: Metal compile-option divergence. We use `default MTLCompileOptions`; peer may use `-O3` / specific math flags. Lever: dump `MTLCompileOptions` in both at runtime and compare.
- **H-C**: Memory-layout / cache miss diffs between equivalent kernels. Same algo + different KV layout / arg ordering → different L1/L2 behavior. Lever: Apple Instruments Metal trace cache-miss counters (operator-runnable).
- **H-D**: Stage-boundary serialization. M5 Max only supports `MTLCounterSamplingPointAtStageBoundary` (not `AtDispatchBoundary`) — there's HW-enforced serialization at stage boundaries that peer's command sequencing may exploit better than ours. Lever: capture & diff Metal command-buffer structure (CB count, dispatch grouping per CB).

## What this rules out

- ✗ Porting #1-#2 hottest kernels from peer (already done in iter-308/352/401 and matching peer at F=2)
- ✗ Single-site optimization on the top 2 kernels (gap is uniformly ~1 µs/dispatch per iter-111; no single kernel concentrates >1ms of gap)

## Limitations of this run

- Per-CB / per-dispatch GPU TIME not collected: `MLX_PROFILE_DISPATCH=1` is no-op'd on M5 Max (AGXG17X only supports AtStageBoundary). `MLX_PROFILE_CB=1` requires `commit_and_wait_labeled` which is not wired in gemma4 path (`/opt/hf2q/src/serve/forward_mlx.rs` uses unlabeled `commit_and_wait`). So we have **dispatch counts but not GPU time per kernel**.
- Per-kernel-class time attribution would need either:
  - (a) Wire `commit_and_wait_labeled` into gemma4 decode path (medium effort, durable)
  - (b) Apple Instruments Metal System Trace (operator-runnable, one-shot per investigation)

## Files

- Raw stderr: `/tmp/iter175-step1-stderr.txt` (90 lines + 53-pipeline dump)
- Source refs: `quantized_matmul_ggml.metal:713`, `rms_norm.metal:81`, `ggml-metal.metal:7970,2989`

## Recommended next iteration

**Step 1b**: Investigate **H-B (compile-options divergence)** — fastest to falsify. Dump both repos' `MTLCompileOptions` at pipeline-creation time and diff. If divergent, port peer's options; if equivalent, H-B falsified.

**Step 1c**: Investigate **H-A (per-dispatch encoder overhead)** by reading `/opt/mlx-native/src/encoder.rs` encode* fast-path against `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m` `ggml_metal_op_*` per-op encode pattern.

Both are read-only / data-collection, no code changes, fit a 5-min /loop iteration.

## Sources

- iter-308 kernel port: `project_adr029_iter112_quant_v_advantage_2026_05_12` (memory) — q6_K_nr2 origin
- iter-352 FOR_UNROLL falsification: inline kernel comment at `quantized_matmul_ggml.metal:764-769`
- iter-401 short-indexing port: inline kernel comment at `quantized_matmul_ggml.metal:748-750`
- iter-115 GPU 95% body decode: memory `project_adr029_iter112_quant_v_advantage_2026_05_12`
- iter-111 constant ratio (gap is ~1 µs/dispatch): memory `project_adr029_iter111_constant_ratio_2026_05_12`
