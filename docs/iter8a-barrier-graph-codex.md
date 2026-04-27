# iter8a 8a-A: qwen35 single-CB greedy decode barrier graph

Scope: design only. No Rust changes in this subtask. This document maps the current `forward_gpu_greedy` hot path to a new `forward_gpu_single_cb` entry that opens one `mlx_native::CommandEncoder`, sends every decode dispatch into that encoder, replaces every ADR-015 fuse-safe command-buffer boundary with `enc.memory_barrier()`, and ends with one terminal `commit_and_wait_labeled`.

Authoritative inputs:

- `/tmp/cfa-cfa-20260427-adr015-iter8a/spec.json`
- `/opt/hf2q/docs/ADR-015-mlx-native-single-cb-decode.md`, section "P1 audit (merged) - qwen35 decode CB boundaries"
- `src/inference/models/qwen35/forward_gpu.rs`
- `src/inference/models/qwen35/gpu_full_attn.rs`
- `src/inference/models/qwen35/gpu_ffn.rs`
- `src/inference/models/qwen35/gpu_delta_net.rs`
- `/opt/mlx-native/src/encoder.rs`
- `/opt/mlx-native/src/graph.rs`

## Core invariants

- New default decode entry: `forward_gpu_single_cb`.
- Legacy escape hatch only: `HF2Q_LEGACY_PER_LAYER_CB=1`.
- No parity-failure fallback, no runtime fallback, no catch-and-rerun legacy.
- One `device.command_encoder()` in the single-CB path, after cache/buffer setup and before the first GPU dispatch.
- Zero `device.command_encoder()` calls inside the per-layer loop.
- Zero `commit`, `commit_labeled`, `commit_and_wait`, or `commit_and_wait_labeled` calls inside helpers when they are reached from the single-CB path.
- Exactly one terminal `enc.commit_and_wait_labeled(...)`, after argmax has been encoded and before `out_index.as_slice::<u32>()`.
- The implementation should use the existing `CommandEncoder` substrate directly. `GraphSession` in `/opt/mlx-native/src/graph.rs` is a wrapper around the same single-encoder model and exposes `encoder_mut()`, but current qwen35 helper APIs already take `&mut CommandEncoder`.

## Pseudocode

Use the current source signature, not the stale two-argument wording in one spec line:

```rust
pub fn forward_gpu_single_cb(
    &self,
    tokens: &[u32],
    positions_flat: &[i32],
    kv_cache: &mut HybridKvCache,
) -> Result<u32> {
    if std::env::var("HF2Q_LEGACY_PER_LAYER_CB").as_deref() == Ok("1") {
        return self.forward_gpu_greedy(tokens, positions_flat, kv_cache);
    }

    debug_assert_eq!(tokens.len(), 1);
    reject empty tokens;
    super::decode_pool::reset_decode_pool();
    validate positions_flat.len() == 4 * seq_len;

    // Same cache population and lazy DecodeBuffers initialization as
    // forward_gpu_greedy. Keep the one-time lm_head BF16 cast commit in the
    // cold cache initialization path; it is not per-token decode work.
    populate GPU_CACHE if model_ptr changed;
    lazily initialize DecodeBuffers;

    // Same raw-pointer extraction pattern as forward_gpu_greedy.
    // pos_buf and embed_buf are CPU-written before the encoder is opened.
    let (pos_buf, layer_weights_gpu, device, registry, output_head, decode_bufs) =
        borrow cached GPU state;

    let mut hidden = embed one token into decode_bufs.embed_buf;

    let mut enc = device.command_encoder()
        .context("enc qwen35 single-cb greedy decode")?;

    for (layer_idx, layer_gpu) in layer_weights_gpu.iter().enumerate() {
        let attn_out = match layer_gpu {
            LayerWeightsGpu::FullAttn { attn, .. } => {
                let slot = full-attn kv slot for layer_idx;
                let shape = FullAttnShape::from_config(cfg);

                // Stage FA-A, new helper. Uses the shared encoder and preserves
                // existing internal barriers:
                //   pre_attn_norm -> Q/K/V/G projections
                //   projections -> Q/K per-head norms
                //   Q/K norms -> IMROPE
                let full = build_gated_attn_ops1_4_into(
                    &mut enc, device, registry, &hidden, &pos_buf, attn, shape, seq_len,
                )?;

                // P3 barrier FA-1: ops1-4 outputs -> SDPA input.
                enc.memory_barrier();

                // Stage FA-B, new helper. Decode branch only in single-CB:
                // KV cache copy, optional internal barrier, SDPA decode.
                // Updates slot.current_len on CPU after encode, as today.
                let attn_out = apply_sdpa_with_kv_cache_into(
                    &mut enc, device, registry,
                    &full.q_rope, &full.k_rope, &full.v_flat,
                    slot, seq_len, shape.n_head, shape.n_kv, shape.head_dim, max_seq,
                )?;

                // P3 barrier FA-2: SDPA output -> gated output projection.
                enc.memory_barrier();

                // Stage FA-C, new helper. Encodes sigmoid gate multiply and
                // output projection into the same shared encoder. Add/preserve
                // an internal RAW barrier between gated write and wo read.
                let out = build_gated_attn_ops6_7_into(
                    &mut enc, device, registry,
                    &attn_out, &full.gate_flat, attn, shape, seq_len,
                )?;

                // P3 barrier FA-3: full-attn contribution -> fused residual norm.
                enc.memory_barrier();
                out
            }

            LayerWeightsGpu::LinearAttn { attn, .. } => {
                let slot = linear-attn kv slot for layer_idx;
                let shape = DeltaNetLayerShape::from_config(cfg);
                let (conv_in, conv_out, state_in, state_out) =
                    current ping-pong buffers for this layer;

                // New helper. It is the current seq_len == 1 ops1-9 branch of
                // build_delta_net_layer, but takes &mut enc and does not commit.
                // Preserve its seven internal barriers.
                let out = build_delta_net_layer_into(
                    &mut enc, device, registry, &hidden, attn,
                    conv_in, conv_out, state_in, state_out,
                    seq_len, shape.hidden_size, shape.n_k_heads, shape.n_v_heads,
                    shape.d_k, shape.d_v, shape.conv_kernel, shape.rms_norm_eps,
                )?;

                // Safe after encode: Metal retained the bound buffers, and the
                // next token cannot start until the terminal commit_and_wait.
                slot.swap_conv_state();
                slot.swap_recurrent();

                // P3 barrier DN-1: DeltaNet output -> fused residual norm.
                enc.memory_barrier();
                out
            }
        };

        let (ffn_input, ffn_residual) = &decode_bufs.layer_scratch[layer_idx];
        let post_norm_w = layer_gpu.post_attn_norm();

        dispatch_fused_residual_norm_f32(
            &mut enc, registry, device.metal_device(),
            &hidden, &attn_out, post_norm_w,
            ffn_input, Some(ffn_residual), seq_len, h, eps,
        )?;

        // Existing required RAW barrier: fused residual norm writes ffn_input
        // and ffn_residual; build_moe_ffn_layer_gpu_q_into reads them.
        enc.memory_barrier();

        let ffn_out = match ffn_weights_gpu {
            FfnWeightsGpu::MoeQ(w_gpu) => {
                build_moe_ffn_layer_gpu_q_into(
                    &mut enc, device, registry, ffn_input, w_gpu, moe_shape,
                    Some(ffn_residual),
                )?
            }
            _ => return Err(anyhow!(
                "forward_gpu_single_cb iter8a supports the MoeQ decode hot path only"
            )),
        };

        hidden = ffn_out;

        // P3 barrier MOE-1: FFN output becomes next layer hidden, or final
        // output-head input on the last layer.
        enc.memory_barrier();
    }

    // Output head, encoded directly into the same encoder instead of calling
    // apply_output_head_gpu_greedy, because that helper owns three encoders.
    let normed = &decode_bufs.norm_out_buf;
    let logits = &mut decode_bufs.logits_buf;
    let out_index = &decode_bufs.argmax_index_buf;
    let out_value = &decode_bufs.argmax_value_buf;

    rms_norm::dispatch_rms_norm(
        &mut enc, registry, device.metal_device(),
        &hidden, &output_head.norm_w, normed, &decode_bufs.norm_params_buf,
        1, h,
    )?;

    // P3 barrier HEAD-1: output norm -> lm_head.
    enc.memory_barrier();

    apply_linear_projection_f32_into(
        &mut enc, registry, device, normed, &output_head.lm_head_q4,
        logits, 1, h, cfg.vocab_size,
    )?;

    // P3 barrier HEAD-2: lm_head logits -> argmax.
    enc.memory_barrier();

    dispatch_argmax_f32(
        &mut enc, registry, device.metal_device(),
        logits, out_index, out_value, &decode_bufs.argmax_params_buf,
        cfg.vocab_size,
    )?;

    enc.commit_and_wait_labeled("qwen35.single_cb.decode")
        .context("commit qwen35 single-cb greedy decode")?;

    Ok(out_index.as_slice::<u32>()
        .map_err(|e| anyhow!("out_index as_slice: {e}"))?[0])
}
```

## P3 barrier placement table

This table is the implementation contract for the 102 ADR fuse-safe command-buffer boundaries. Multiplicity sums to 102. The terminal CPU read is listed separately and is not a memory-barrier row.

| ID | Multiplicity | Site | Producer | Consumer | Race kind | Scope | Placement |
|---|---:|---|---|---|---|---|---|
| FA-1 | 10 | `gpu_full_attn.rs:1115/:1180 -> :930/:971` | Full-attn ops1-4 write `q_rope`, `k_rope`, `v_flat`, `gate_flat` | `apply_sdpa_with_kv_cache_into` reads `q_rope`, `k_rope`, `v_flat` | RAW | Inter-helper, same FullAttn layer | After ops1-4 final IMROPE dispatches, before KV-cache SDPA encode |
| FA-2 | 10 | `gpu_full_attn.rs:959/:983 -> :1211/:1229` | `dispatch_sdpa_decode` writes `attn_out`/`out_buf` | ops6-7 `apply_sigmoid_gate_multiply` reads `attn_out` | RAW | Inter-helper, same FullAttn layer | After SDPA decode dispatch, before sigmoid gate multiply |
| FA-3 | 10 | `gpu_full_attn.rs:1211/:1229 -> forward_gpu.rs:1422` | Full-attn output projection writes attention contribution `out` | `dispatch_fused_residual_norm_f32` reads `attn_out` | RAW | Inter-helper, FullAttn -> common post-attn block | After output projection, before fused residual norm |
| DN-1 | 30 | `gpu_delta_net.rs:893/:970 -> forward_gpu.rs:1422` | DeltaNet op9 output projection writes attention contribution `output` | `dispatch_fused_residual_norm_f32` reads `attn_out` | RAW | Inter-helper, DeltaNet -> common post-attn block | After `build_delta_net_layer_into` terminal op9, before fused residual norm |
| MOE-1 | 40 | `forward_gpu.rs:1422/:1445 -> next layer or output head` | `build_moe_ffn_layer_gpu_q_into` writes `ffn_out` | Next layer pre-attn norm/pre-norm reads `hidden`, or output norm reads final `hidden` | RAW | Inter-helper; inter-layer except final layer | Immediately after MoE helper returns, before next layer attention stage or final output norm |
| HEAD-1 | 1 | `forward_gpu.rs:393/:399 -> :403` | output RMSNorm writes `decode_bufs.norm_out_buf` | lm_head Q4 projection reads `norm_out_buf` | RAW | Inter-helper, output head | After `dispatch_rms_norm`, before `apply_linear_projection_f32_into`/`quantized_matmul_ggml` |
| HEAD-2 | 1 | `forward_gpu.rs:403/:409 -> :412` | lm_head Q4 projection writes `decode_bufs.logits_buf` | `dispatch_argmax_f32` reads `logits_buf` | RAW | Inter-helper, output head | After lm_head projection, before argmax |
| TERM | 1 | `forward_gpu.rs:412/:417 -> :421` | argmax writes `decode_bufs.argmax_index_buf` | CPU `out_index.as_slice::<u32>()` reads token id | GPU/CPU RAW | Terminal host sync | No `memory_barrier`; this is the single `commit_and_wait_labeled` at the end |

Count: `10 + 10 + 10 + 30 + 40 + 1 + 1 = 102` memory barriers replacing fuse-safe CB boundaries, plus one terminal commit.

## Existing intra-helper barriers to preserve

These are not part of the 102 cross-CB replacement count, but the `_into` variants must retain them because `/opt/mlx-native/src/encoder.rs` uses `MTLDispatchTypeConcurrent` and documents that dependent dispatches require `memory_barrier()`.

| Site | Producer | Consumer | Race kind | Scope | Required action |
|---|---|---|---|---|---|
| `gpu_full_attn.rs:1134` | pre-attn RMSNorm writes `x_norm` | Q/K/V/G projections read `x_norm` | RAW | Intra-helper ops1-4 | Preserve in `build_gated_attn_ops1_4_into` |
| `gpu_full_attn.rs:1157` | projections write `q_flat`, `k_flat` | Q/K per-head norms read `q_flat`, `k_flat` | RAW | Intra-helper ops1-4 | Preserve in `build_gated_attn_ops1_4_into` |
| `gpu_full_attn.rs:1169` | Q/K norms write `q_normed`, `k_normed` | IMROPE reads `q_normed`, `k_normed` | RAW | Intra-helper ops1-4 | Preserve in `build_gated_attn_ops1_4_into` |
| `gpu_full_attn.rs:981` | KV cache copy writes `slot.k`, `slot.v` | SDPA decode reads `slot.k`, `slot.v` | RAW | Intra-helper SDPA | Preserve conditionally when `kv_write_tokens > 0` in `apply_sdpa_with_kv_cache_into` |
| `gpu_full_attn.rs:1223 -> :1226` | sigmoid gate multiply writes `gated` | output projection reads `gated` | RAW | Intra-helper ops6-7 | Add/preserve an explicit barrier in `build_gated_attn_ops6_7_into`; current helper has no visible barrier here |
| `gpu_delta_net.rs:899/:909/:916/:932/:945/:953/:961` | DeltaNet ops1 through op8 stage outputs | Next dependent DeltaNet stage reads those outputs | RAW | Intra-helper DeltaNet ops1-9 | Preserve all seven barriers in `build_delta_net_layer_into` |
| `forward_gpu.rs:1438` | fused residual norm writes `ffn_input` and `ffn_residual` | `build_moe_ffn_layer_gpu_q_into` reads both | RAW | Same layer, direct dispatch -> MoE helper | Preserve in the common post-attn block |
| `gpu_ffn.rs:1255/:1269/:1303/:1325/:1343` | MoE phases A through E write router/shared/expert intermediates | Next MoE phase reads those intermediates | RAW | Intra-helper MoE | Already inside `build_moe_ffn_layer_gpu_q_into`; preserve unchanged |

No WAR or WAW barrier is required in the P3 cross-CB table. Current hot-path reuse writes to distinct per-layer scratch buffers (`decode_bufs.layer_scratch[layer_idx]`) and ping-pong state buffers. The races are RAW producer/consumer ordering hazards.

## Helper-signature audit

| Helper or op | Current signature/ownership | Called by current greedy path | Single-CB action |
|---|---|---:|---|
| `build_gated_attn_layer` (`gpu_full_attn.rs:1088`) | Owns encoders internally: ops1-4 encoder, `apply_sdpa_with_kv_cache` encoder, ops6-7 encoder | Yes, FullAttn layers | Do not call from `forward_gpu_single_cb`; split into stage `_into` helpers or make it a thin legacy wrapper over those stages |
| `build_gated_attn_ops1_4_into` | Does not exist | Proposed for FullAttn layers | Add. Takes `&mut CommandEncoder`; returns a small stage struct with `q_rope`, `k_rope`, `v_flat`, `gate_flat` and keeps temporary buffers alive long enough for encoded resources |
| `apply_sdpa_with_kv_cache` (`gpu_full_attn.rs:930`) | Owns encoder in decode and prefill branches; commits internally | Yes, through `build_gated_attn_layer` | Add `apply_sdpa_with_kv_cache_into(&mut CommandEncoder, ...)` for decode path. Legacy wrapper can call it for `seq == 1 && head_dim % 32 == 0`, then commit as today; prefill branch can remain in the existing function |
| `build_gated_attn_ops6_7_into` | Does not exist; current logic is an inline block in `build_gated_attn_layer` | Proposed for FullAttn layers | Add. Takes `&mut CommandEncoder`, `attn_out`, `gate_flat`; dispatches sigmoid multiply, barrier, output projection; no commit |
| `apply_pre_attn_rms_norm` (`gpu_full_attn.rs:478`) | Already takes `&mut CommandEncoder` | Yes, inside FullAttn ops1-4 | Reuse unchanged |
| `apply_linear_projection_f32_pooled` (`gpu_full_attn.rs:733`) | Already takes `&mut CommandEncoder` and calls `_into` | Yes, FullAttn projections | Reuse unchanged |
| `apply_linear_projection_f32_into` (`gpu_full_attn.rs:644`) | Already takes `&mut CommandEncoder` and caller-supplied dst | Yes, lm_head greedy | Reuse unchanged for lm_head |
| `apply_q_or_k_per_head_rms_norm` (`gpu_full_attn.rs:309`) | Already takes `&mut CommandEncoder` | Yes, FullAttn ops1-4 | Reuse unchanged |
| `apply_imrope` (`gpu_full_attn.rs:361`) | Already takes `&mut CommandEncoder` | Yes, FullAttn ops1-4 | Reuse unchanged |
| `apply_sigmoid_gate_multiply` (`gpu_full_attn.rs:431`) | Already takes `&mut CommandEncoder` | Yes, FullAttn ops6 | Reuse unchanged, but put a barrier before the projection that reads its output |
| `build_delta_net_layer` (`gpu_delta_net.rs:776`) | Owns encoder in decode branch and two encoders in prefill branch; commits internally | Yes, LinearAttn layers | Add `build_delta_net_layer_into(&mut CommandEncoder, ...)` for `seq_len == 1`; legacy wrapper calls into then commits for decode, while prefill remains unchanged |
| `apply_pre_norm` (`gpu_delta_net.rs:265`) | Already takes `&mut CommandEncoder` | Yes, inside DeltaNet | Reuse unchanged |
| `apply_proj` (`gpu_delta_net.rs:315`) | Already takes `&mut CommandEncoder` | Yes, inside DeltaNet | Reuse unchanged |
| `apply_l2_norm_per_head` (`gpu_delta_net.rs:509`) | Already takes `&mut CommandEncoder` | Yes, inside DeltaNet | Reuse unchanged |
| `dispatch_ssm_conv`, `dispatch_compute_g_beta`, `dispatch_gated_delta_net`, `dispatch_ssm_norm_gate` | All take `&mut CommandEncoder` at op level | Yes, inside DeltaNet | Reuse unchanged inside `build_delta_net_layer_into` |
| `dispatch_fused_residual_norm_f32` (`/opt/mlx-native/src/ops/fused_norm_add.rs:446`) | Already takes `&mut CommandEncoder` | Yes, common post-attn block | Reuse unchanged |
| `build_moe_ffn_layer_gpu_q_into` (`gpu_ffn.rs:1145`) | Already takes `&mut CommandEncoder`; no commit | Yes, MoeQ FFN path | Reuse unchanged |
| `build_moe_ffn_layer_gpu_q` (`gpu_ffn.rs:1086`) | Owns encoder and commits | Not on current greedy MoeQ hot path, because `forward_gpu.rs` calls `_into` directly | No action for single-CB path |
| `rms_norm::dispatch_rms_norm` (`/opt/mlx-native/src/ops/rms_norm.rs:64`) | Already takes `&mut CommandEncoder` | Yes, output head helper | Call directly from `forward_gpu_single_cb` |
| `quantized_matmul_ggml` (`/opt/mlx-native/src/ops/quantized_matmul_ggml.rs:264`) | Already takes `&mut CommandEncoder` | Yes, through `apply_linear_projection_f32_into` for Q4 lm_head | Reuse through existing `apply_linear_projection_f32_into` |
| `dispatch_argmax_f32` (`/opt/mlx-native/src/ops/argmax.rs:47`) | Already takes `&mut CommandEncoder` | Yes, output head helper | Call directly from `forward_gpu_single_cb` |
| `apply_output_head_gpu_greedy` (`forward_gpu.rs:363`) | Owns three encoders and commits internally | Yes, after layer loop | Do not call from `forward_gpu_single_cb`; either inline direct dispatches as above or add a private `apply_output_head_gpu_greedy_into` helper |

Required new `_into` helpers for iter8a: 4.

1. `build_gated_attn_ops1_4_into`
2. `apply_sdpa_with_kv_cache_into`
3. `build_gated_attn_ops6_7_into`
4. `build_delta_net_layer_into`

Optional factoring helper, not required by the barrier graph: `apply_output_head_gpu_greedy_into`.

## Env-gate dispatch design

Call sites should route to `forward_gpu_single_cb` by default. The only legacy route is an explicit env var value:

```rust
if std::env::var("HF2Q_LEGACY_PER_LAYER_CB").as_deref() == Ok("1") {
    self.forward_gpu_greedy(tokens, positions_flat, kv_cache)
} else {
    self.forward_gpu_single_cb(tokens, positions_flat, kv_cache)
}
```

Recommended implementation shape: put the guard at the top of `forward_gpu_single_cb`, then update external greedy decode callers to call `forward_gpu_single_cb`. The known current call site is `src/serve/mod.rs:960`.

Semantics:

- `HF2Q_LEGACY_PER_LAYER_CB=1`: run existing `forward_gpu_greedy` unchanged.
- Env unset, empty, `0`, `true`, or any value other than exact string `1`: run `forward_gpu_single_cb`.
- No fallback: if the new path returns an error, return that error. Do not rerun `forward_gpu_greedy`.
- Telemetry can use the existing label/profile channel: new path should emit one terminal label such as `qwen35.single_cb.decode`; legacy retains its per-layer labels. If `HF2Q_PROFILE_GPU_TS=1` support is wired in 8a-C, report `single_cb_count=1` on the new path and `single_cb_count=103` on the legacy path.

## Risk register and concrete spikes

| Risk | Why it matters | Spike / mitigation |
|---|---|---|
| `apply_sdpa_with_kv_cache` encoder split | The current helper owns and commits a decode encoder at `gpu_full_attn.rs:971/:995`, and has a prefill branch with CPU downloads. Accidentally threading prefill through single-CB would race CPU reads. | Add `_into` only for the decode branch (`seq == 1 && head_dim % 32 == 0`). Preserve the conditional KV-copy -> SDPA barrier when `kv_write_tokens > 0`. Keep prefill in the legacy helper. Update `slot.current_len` after encoding, as today. |
| DeltaNet encoder ownership | `build_delta_net_layer` owns the ops1-9 decode encoder and commits at `gpu_delta_net.rs:893/:970`. It also swaps ping-pong state in the caller after return. | Add `build_delta_net_layer_into` by extracting only the `seq == 1` branch. Preserve seven internal barriers. Keep state swaps in `forward_gpu_single_cb` after the dispatches are encoded; Metal retains buffer resources, and the next token cannot start before terminal wait. |
| Decode pool reset lifecycle | The pool is reset at the top of `forward_gpu_greedy`, while single-CB extends every scratch buffer lifetime until the terminal commit. A reset before commit would free/reuse buffers still referenced by the command buffer. | Keep exactly one `reset_decode_pool()` at the start of the token, before any pooled allocation. Do not reset inside helpers. Do not CPU-read pooled scratch buffers before terminal commit. Existing regular command buffers retain resources, and the pool keeps ARC clones in its in-use list. |
| Encoder lifetime and Rust borrow-check | One `&mut CommandEncoder` must cross FullAttn, DeltaNet, MoE, and output-head calls while the function also borrows `&mut KernelRegistry`, `&mut HybridKvCache`, and cached buffers from a thread-local raw-pointer pattern. | Keep borrows scoped tightly. Do not hold `&mut FullAttnKvSlot` or `&mut LinearAttnKvSlot` across the common post-attn/FFN block. Return buffers from helpers by value. If borrow-check pressure grows, factor a private `encode_layer_single_cb(&mut CommandEncoder, ...)` with explicit inputs instead of widening long-lived borrows. |
| FullAttn ops6-7 internal RAW barrier | Current code has no visible `memory_barrier()` between `apply_sigmoid_gate_multiply` writing `gated` and the output projection reading it in the same concurrent encoder. | In the new `build_gated_attn_ops6_7_into`, insert `enc.memory_barrier()` between sigmoid gate multiply and output projection. This is an intra-helper barrier, not one of the 102 ADR cross-CB replacement barriers. Parity tests should decide whether legacy accidentally relied on implicit ordering. |
| Non-MoeQ FFN branch | Iter8a objective is the MoeQ 35B-A3B dwq46 hot path. Current greedy has Dense/DenseQ/F32-MoE legacy branches with owned encoders and commits. | Do not silently fallback to legacy inside the single-CB path. Either return an explicit unsupported error for non-MoeQ in iter8a, or separately design dense `_into` coverage in a later task. |

## Implementation checklist for coder

- Add the four required `_into` helpers.
- Keep old helpers as compatibility wrappers; wrappers may own encoders and commit as they do today.
- Implement `forward_gpu_single_cb` by copying the setup from `forward_gpu_greedy`, then replacing the layer loop and output head with the shared-encoder flow above.
- Place the seven P3 table barriers in order, with their multiplicities naturally produced by the layer loop.
- Preserve helper-internal barriers from the second table.
- End with exactly one `commit_and_wait_labeled`.
- Update external greedy call sites to call `forward_gpu_single_cb`; rely on the env gate for legacy.
