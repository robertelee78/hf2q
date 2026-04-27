# ADR-015 iter8a 8a-A — qwen35 single-CB decode barrier graph (Claude architect)

Session: `cfa-20260427-adr015-iter8a` — subtask `8a-A` (architect, design only).

This doc is the implementation contract for `8a-B` (coder). It defines the
pseudocode skeleton, the cross-stage barrier table, the helper-signature
audit, the env-gate dispatch, and the risk register for collapsing the
`qwen3.5-MoE-35B-A3B dwq46` greedy decode path from **103 CBs/decode-token**
to **1 CB/decode-token**.

Authoritative inputs (verbatim):

- `/tmp/cfa-cfa-20260427-adr015-iter8a/spec.json` — subtasks, off-limits
  fences, barrier_placement_table, invariants, AC, judging rubric.
- `/opt/hf2q/docs/ADR-015-mlx-native-single-cb-decode.md` §"P1 audit
  (merged) — qwen35 decode CB boundaries" (lines 739-926).
- `src/inference/models/qwen35/{forward_gpu,gpu_full_attn,gpu_ffn,gpu_delta_net}.rs`
  at branch `cfa/cfa-20260427-adr015-iter8a/claude`.
- `/opt/mlx-native/src/{encoder,graph}.rs` (read-only substrate).

**Critical correction vs prior scaffold draft.** A scaffold version of this
doc previously claimed four new `_into` variants must be authored
(`build_gated_attn_layer_ops1to4_into`, `apply_sdpa_with_kv_cache_into`,
`build_gated_attn_layer_ops6to7_into`, `build_delta_net_layer_into`).
Live audit of branch HEAD shows that the relevant `_into` helpers already
exist — see §3. The 8a-B coder needs ZERO new helper extractions; the
work is the top-level `forward_gpu_single_cb` body plus a one-line caller
swap in `src/serve/mod.rs:960`. Per the "code is truth, doc may lag"
memory pin, this doc supersedes the scaffold.

---

## 1. Pseudocode for `forward_gpu_single_cb`

The new function lives in `src/inference/models/qwen35/forward_gpu.rs`,
replacing the existing stub at lines 1610–1626. The signature MUST match
the existing stub (and `forward_gpu_greedy`) — the launcher's spec text
calling out `(tokens, positions)` is stale and the code-of-record carries
`positions_flat` and `kv_cache`.

```rust
pub fn forward_gpu_single_cb(
    &self,
    tokens: &[u32],
    positions_flat: &[i32],
    kv_cache: &mut HybridKvCache,
) -> Result<u32> {
    // ---- Env gate: legacy escape hatch ----
    // Per spec invariant "HF2Q_LEGACY_PER_LAYER_CB=1 routes to existing
    // forward_gpu_greedy unchanged". This is the ONLY caller of
    // forward_gpu_greedy that must remain after coder-pass; production
    // call site in src/serve/mod.rs:960 pivots to forward_gpu_single_cb.
    if std::env::var("HF2Q_LEGACY_PER_LAYER_CB").as_deref() == Ok("1") {
        return self.forward_gpu_greedy(tokens, positions_flat, kv_cache);
    }

    // ---- Same precondition + decode-pool reset as greedy ----
    debug_assert_eq!(tokens.len(), 1, "forward_gpu_single_cb: tokens length 1");
    if tokens.is_empty() {
        return Err(anyhow!("forward_gpu_single_cb: tokens must be non-empty"));
    }
    super::decode_pool::reset_decode_pool();
    let seq_len = tokens.len() as u32;
    let expected_pos_len = 4 * seq_len as usize;
    if positions_flat.len() != expected_pos_len {
        return Err(anyhow!(
            "forward_gpu_single_cb: positions_flat.len() = {} != 4 * seq_len = {}",
            positions_flat.len(), expected_pos_len
        ));
    }

    let cfg = &self.cfg;
    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let self_ptr = self as *const _ as *const ();

    // ---- Cache + decode-buffer setup (BEFORE the encoder is opened) ----
    // Identical to forward_gpu_greedy lines 1108–1218. The lm_head BF16 cast
    // at greedy line 1121 (model-load-only, fires once per model) commits its
    // own tiny encoder; that's NOT on the per-decode-token hot path so it stays.
    populate_gpu_cache_if_needed(self, self_ptr)?;     // greedy 1108–1155
    lazy_init_decode_buffers(self, cfg)?;              // greedy 1162–1218

    let (pos_buf, layer_weights_gpu, device, registry, output_head, decode_bufs) =
        borrow_cached_state_with_positions(positions_flat)?;  // greedy 1220–1246

    // ---- Embedding (CPU gather → embed_buf, BEFORE encoder opens) ----
    // Reuses greedy lines 1253–1266. CPU-side write of decode_bufs.embed_buf
    // is a host store; no GPU dispatch yet, no encoder needed.
    let mut hidden = embed_token_into_buf(tokens, decode_bufs)?;

    // ================================================================
    // ONE encoder, opened ONCE, used for ALL 103 dispatches of this token.
    // ================================================================
    let mut enc = device.command_encoder()
        .context("enc qwen35 single-cb decode")?;

    for (layer_idx, layer_gpu) in layer_weights_gpu.iter().enumerate() {
        // ---- Stage A: attention block (FullAttn or DeltaNet) ----
        let attn_out = match layer_gpu {
            LayerWeightsGpu::FullAttn { attn, .. } => {
                let shape = FullAttnShape::from_config(cfg);
                let full_attn_rank = match kv_cache.slot_index_for_layer(layer_idx as u32) {
                    Some(super::kv_cache::LayerSlot::Full(rank)) => rank as usize,
                    other => return Err(anyhow!(
                        "layer {layer_idx}: expected FullAttn slot, got {:?}", other
                    )),
                };
                let max_seq = kv_cache.max_seq_len;
                let slot = &mut kv_cache.full_attn[full_attn_rank];

                // build_gated_attn_layer_into already exists (gpu_full_attn.rs:1278).
                // It encodes ops1-4 + KV-cache-copy + sdpa_decode + ops6-7 into
                // `enc`, with intra-block memory_barriers at: ops1→ops2,
                // ops2→ops3, ops3→ops4, ops4→sdpa, kv_copy→sdpa, sdpa→ops6-7.
                // Each is a P1-audit fuse_safe=YES boundary that today fires
                // as a per-stage commit_labeled and becomes intra-encoder
                // memory_barriers under single-CB.
                build_gated_attn_layer_into(
                    &mut enc, device, registry, &hidden, &pos_buf, attn,
                    slot, max_seq, seq_len,
                    shape.hidden_size, shape.n_head, shape.n_kv,
                    shape.head_dim, shape.rotary_dim, shape.rope_theta,
                    shape.mrope_section, shape.rms_norm_eps,
                ).with_context(|| format!("full_attn_into single-cb layer {layer_idx}"))?
            }
            LayerWeightsGpu::LinearAttn { attn, .. } => {
                let shape = DeltaNetLayerShape::from_config(cfg);
                // KV-state plumbing (preserves the synthetic-fixture zero-state
                // fallback exactly as in greedy 1313–1349; production decode
                // always carries a slot).
                let (conv_in, conv_out, state_in, state_out, _holdouts) =
                    resolve_delta_state(kv_cache, layer_idx, cfg, &device)?;

                // build_delta_net_layer_into already exists (gpu_delta_net.rs:1097).
                // It encodes ops1-9 with internal memory_barriers at lines
                // 1189, 1199, 1206, 1222, 1235, 1243, 1251 — exactly the 7
                // boundaries §P1 audit cites at ":899/:909/:916/:932/:945/
                // :953/:961" (op-numbering preserved across the refactor).
                let out = build_delta_net_layer_into(
                    &mut enc, device, registry, &hidden, attn,
                    conv_in, conv_out, state_in, state_out,
                    seq_len, shape.hidden_size,
                    shape.n_k_heads, shape.n_v_heads,
                    shape.d_k, shape.d_v, shape.conv_kernel,
                    shape.rms_norm_eps,
                ).with_context(|| format!("delta_net_into single-cb layer {layer_idx}"))?;

                // CPU-side cursor advance (no GPU work). Safe to fire while
                // `enc` is still open — `swap_*` are pointer swaps on metadata,
                // not buffer-content mutations.
                if let Some(super::kv_cache::LayerSlot::Linear(rank)) =
                    kv_cache.slot_index_for_layer(layer_idx as u32)
                {
                    let slot = &mut kv_cache.linear_attn[rank as usize];
                    slot.swap_conv_state();
                    slot.swap_recurrent();
                }
                out
            }
        };

        // ---- Barrier IL-1 (inter-helper): attention out → fused_residual_norm.
        // ADR-015 §P1 rows: gpu_full_attn.rs:1211/:1229 (FullAttn ops6-7 →
        // forward_gpu.rs:1422 fused_res_norm) AND gpu_delta_net.rs:893/:970
        // (DeltaNet op9 → forward_gpu.rs:1422 fused_res_norm).
        // Producer: attn_out (terminal write of attention block).
        // Consumer: dispatch_fused_residual_norm_f32 reads attn_out + hidden.
        enc.memory_barrier();

        // ---- Stage B: fused residual + post-attn RMSNorm + MoE FFN ----
        let post_norm_w = match layer_gpu {
            LayerWeightsGpu::FullAttn  { attn, .. } => &attn.post_attn_norm,
            LayerWeightsGpu::LinearAttn{ attn, .. } => &attn.post_attn_norm,
        };
        let (ffn_input_buf_ref, ffn_residual_buf_ref) =
            &decode_bufs.layer_scratch[layer_idx];
        let ffn_weights_gpu = match layer_gpu {
            LayerWeightsGpu::FullAttn  { ffn, .. } => ffn,
            LayerWeightsGpu::LinearAttn{ ffn, .. } => ffn,
        };

        // Single-CB on dwq46 only supports the MoeQ FFN path. Spec scope
        // (Non-Goals 3 + 4) explicitly excludes Dense / DenseQ / F32-MoE
        // legacy paths from iter8a; they remain in forward_gpu_greedy under
        // HF2Q_LEGACY_PER_LAYER_CB=1. Bail loudly on non-MoeQ — better than
        // silent dual-encoder mixing.
        let ffn_out = match ffn_weights_gpu {
            FfnWeightsGpu::MoeQ(w_gpu) => {
                let moe = cfg.moe.as_ref().ok_or_else(|| {
                    anyhow!("MoeQ FFN missing moe config single-cb (layer {layer_idx})")
                })?;
                let shape = MoeFfnShape {
                    hidden_size: h,
                    num_experts: moe.num_experts,
                    num_experts_per_tok: moe.num_experts_per_tok,
                    moe_intermediate_size: moe.moe_intermediate_size,
                    shared_intermediate_size: moe.shared_expert_intermediate_size,
                };
                // Same dispatch_fused_residual_norm_f32 + memory_barrier +
                // build_moe_ffn_layer_gpu_q_into pattern as greedy 1425–1444,
                // but threaded through the SHARED `enc` instead of opening
                // a new device.command_encoder().
                dispatch_fused_residual_norm_f32(
                    &mut enc, registry, device.metal_device(),
                    &hidden, &attn_out, post_norm_w,
                    ffn_input_buf_ref, Some(ffn_residual_buf_ref),
                    seq_len, h, eps,
                ).with_context(|| format!(
                    "fused_residual_norm single-cb layer {layer_idx}"
                ))?;

                // ---- Barrier MOE-1 (intra-block, RAW): fused_res_norm
                // writes ffn_input_buf + ffn_residual_buf; build_moe_ffn_*_into
                // reads them. This is the existing intra-encoder barrier from
                // greedy line 1439, preserved verbatim.
                enc.memory_barrier();

                build_moe_ffn_layer_gpu_q_into(
                    &mut enc, &device, registry, ffn_input_buf_ref, w_gpu, shape,
                    Some(ffn_residual_buf_ref),
                ).with_context(|| format!(
                    "build_moe_ffn_q_into single-cb layer {layer_idx}"
                ))?
            }
            FfnWeightsGpu::DenseQ(_) | FfnWeightsGpu::Dense(_) | FfnWeightsGpu::Moe(_) => {
                // dwq46 production has zero non-MoeQ layers. If we land here
                // we are running an unsupported model on the new path and
                // should fail loudly. Soak window flips dense-Q to single-CB
                // in iter8c+; iter8a is dwq46-only per spec acceptance.
                return Err(anyhow!(
                    "forward_gpu_single_cb: only MoeQ FFN supported in iter8a; \
                     layer {layer_idx} ffn = {:?}; set HF2Q_LEGACY_PER_LAYER_CB=1 \
                     to route via forward_gpu_greedy",
                    std::any::type_name_of_val(ffn_weights_gpu),
                ));
            }
        };

        // ---- Barrier IL-2 (inter-layer): MoE FFN out → next layer's pre-attn
        // norm input (or, on the LAST layer, → output rms_norm read). ADR-015
        // §P1 row: forward_gpu.rs:1422/:1445 → next layer or output norm.
        // Producer: ffn_out (terminal write inside build_moe_ffn_layer_gpu_q_into
        // via add_residual = Some(ffn_residual_buf_ref)).
        enc.memory_barrier();

        // Residual is folded inside build_moe_ffn_layer_gpu_q_into already
        // (greedy MoeQ branch lines 1522–1526 confirms — no separate
        // residual_add_gpu call).
        hidden = ffn_out;
    }

    // ---- Stage C: output head (norm → lm_head_q4 → argmax) ----
    // Replaces apply_output_head_gpu_greedy (forward_gpu.rs:364–425) which
    // owns 3 separate encoders + 3 commits. Inlined here on the shared
    // `enc` so all 3 ops join the single CB.
    let normed        = &decode_bufs.norm_out_buf;
    let norm_params   = &decode_bufs.norm_params_buf;
    let out_index     = &decode_bufs.argmax_index_buf;
    let out_value     = &decode_bufs.argmax_value_buf;
    let argmax_params = &decode_bufs.argmax_params_buf;
    // SAFETY: same exclusive-access argument as greedy line 385–387 — single
    // forward call, no concurrent access; logits_buf is needed &mut for the
    // apply_linear_projection_f32_into write.
    let logits_buf = unsafe {
        &mut (*(decode_bufs as *const DecodeBuffers as *mut DecodeBuffers)).logits_buf
    };

    // C.1: output RMSNorm (replaces greedy line 395–399).
    rms_norm::dispatch_rms_norm(
        &mut enc, registry, device.metal_device(),
        &hidden, &output_head.norm_w, normed, norm_params,
        seq_len, h,
    ).context("dispatch_rms_norm output single-cb")?;

    // ---- Barrier OH-1: output norm → lm_head_q4. ADR-015 §P1 row
    // forward_gpu.rs:393/:399 → :403. Producer: dispatch_rms_norm writes
    // norm_out_buf. Consumer: apply_linear_projection_f32_into reads it.
    enc.memory_barrier();

    // C.2: lm_head_q4 (Q4_0 quantized matmul into pre-allocated logits_buf).
    apply_linear_projection_f32_into(
        &mut enc, registry, device, normed,
        &output_head.lm_head_q4, logits_buf,
        seq_len, h, cfg.vocab_size,
    ).context("lm_head_q4 single-cb")?;

    // ---- Barrier OH-2: lm_head_q4 → argmax. ADR-015 §P1 row
    // forward_gpu.rs:403/:409 → :412. Producer: apply_linear_projection writes
    // logits_buf (full vocab). Consumer: dispatch_argmax_f32 reads it.
    enc.memory_barrier();

    // C.3: GPU argmax over [vocab_size] logits → 4-byte u32 token id.
    dispatch_argmax_f32(
        &mut enc, registry, device.metal_device(),
        logits_buf, out_index, out_value, argmax_params, cfg.vocab_size,
    ).context("dispatch_argmax_f32 single-cb")?;

    // ---- Terminal commit: the ONLY commit_and_wait* in the new path.
    // ADR-015 §P1 row forward_gpu.rs:412/:417, fuse_safe=NO (terminal CPU
    // host read of out_index 4 bytes). Drains the GPU before the host
    // load below; equivalent to the existing greedy commit_and_wait_labeled
    // at line 418 but now consolidating ALL 103 dispatches into 1 CB.
    enc.commit_and_wait_labeled("decode-token")
        .context("commit qwen35 single-cb decode")?;

    // 4-byte host read of the winning token id.
    let token_id = out_index.as_slice::<u32>()
        .map_err(|e| anyhow!("out_index as_slice: {e}"))?[0];
    Ok(token_id)
}
```

**CB invariant check.** This pseudocode satisfies every invariant from
spec.json line 173–188:

- Exactly 1 `device.command_encoder()` (line "let mut enc =").
- Exactly 1 `enc.commit_and_wait_labeled("decode-token")` at the terminal
  argmax-host-read.
- Zero `device.command_encoder()` / `enc.commit*()` calls inside the
  per-layer loop (every helper is `_into` and threads `&mut enc`).
- `decode_pool::reset_decode_pool()` fires at function entry (matches
  greedy line 1091). Encoder-independent — manages MlxBuffer Arc clones,
  not Metal CommandBuffers.
- ForwardGpuCache + DecodeBuffers reused unchanged (no layout changes).
- No new `unsafe`. The single existing `decode_bufs` cast at the
  greedy lm_head site (forward_gpu.rs:385–387) is the same one we
  reuse for `logits_buf`; net unsafe count is unchanged.
- No new env vars beyond `HF2Q_LEGACY_PER_LAYER_CB`.

**Function size check.** Body lines (excluding comments + helper
extractions to private functions): ~150. Well under the 300-line cap from
AC8. The four extraction-helpers
(`populate_gpu_cache_if_needed`, `lazy_init_decode_buffers`,
`borrow_cached_state_with_positions`, `embed_token_into_buf`,
`resolve_delta_state`) are NOT new code — they exist as inlined blocks in
`forward_gpu_greedy` today. The coder may copy-paste those blocks into
`forward_gpu_single_cb` as-is, OR factor them into private fns first;
either approach satisfies AC8 since the coder's choice is mechanical.

---

## 2. Barrier placement table

ADR-015 §P1 audits 103 CBs/decode-token = 102 fuse_safe=YES + 1 fuse_safe=NO.
In single-CB form, 102 CBs collapse to inline `enc.memory_barrier()` calls
(intra-helper barriers stay where they are; inter-helper boundaries get
a NEW explicit barrier between the helper return and the next consumer).
The 1 fuse_safe=NO row becomes the terminal `commit_and_wait_labeled`.

Layer counts (×40 / ×10 / ×30) collapse to a single ROW each in this
table — the per-layer instance count is the "Layers ×N" column.

| # | Site (file:line at branch HEAD) | Producer (writes) | Consumer (reads) | Race | Scope | Layers ×N | ADR-015 §P1 row |
|---|---|---|---|---|---|---:|---|
| **FA-1** | gpu_full_attn.rs:1308–1309 (in `_into`) | `apply_pre_attn_rms_norm` writes `x_norm` | Q/K/V/G `apply_linear_projection_f32_pooled` (×4) read `x_norm` | RAW | intra-helper | 10 | gpu_full_attn.rs:1115/:1180 (ops1-4) |
| **FA-2** | gpu_full_attn.rs:1328–1329 (in `_into`) | Q/K/V/G projections write `q_flat`, `k_flat`, `v_flat`, `gate_flat` | `apply_q_or_k_per_head_rms_norm` reads `q_flat`/`k_flat` | RAW | intra-helper | 10 | gpu_full_attn.rs:1115/:1180 (ops1-4) |
| **FA-3** | gpu_full_attn.rs:1340–1341 (in `_into`) | per-head Q/K norm writes `q_normed`, `k_normed` | `apply_imrope` ×2 reads `q_normed`/`k_normed` | RAW | intra-helper | 10 | gpu_full_attn.rs:1115/:1180 (ops1-4) |
| **FA-4** | gpu_full_attn.rs:1352–1354 (in `_into`) | `apply_imrope` writes `q_rope`, `k_rope` | `dispatch_kv_cache_copy_seq_f32_dual` reads `k_rope`; `dispatch_sdpa_decode` reads `q_rope` | RAW | inter-stage (ops1-4 → SDPA) | 10 | gpu_full_attn.rs:1115/:1180 → :959/:983 |
| **FA-5** | gpu_full_attn.rs:1377–1379 (in `_into`) | `dispatch_kv_cache_copy_seq_f32_dual` writes `slot.k`, `slot.v` | `dispatch_sdpa_decode` reads `slot.k`, `slot.v` | RAW | intra-helper | 10 | gpu_full_attn.rs:959/:983 |
| **FA-6** | gpu_full_attn.rs:1389–1391 (in `_into`) | `dispatch_sdpa_decode` writes `out_buf` | `apply_sigmoid_gate_multiply` reads `out_buf` | RAW | inter-stage (SDPA → ops6-7) | 10 | gpu_full_attn.rs:959/:983 → :1211 |
| **DN-1** | gpu_delta_net.rs:1188–1189 (in `_into`) | `apply_pre_norm` writes `x_norm` | qkv_proj + z_proj read `x_norm` | RAW | intra-helper | 30 | gpu_delta_net.rs:893/:970 (op1→ops2) |
| **DN-2** | gpu_delta_net.rs:1198–1199 (in `_into`) | `apply_proj` writes `qkv_raw` | `dispatch_ssm_conv` reads `qkv_raw` | RAW | intra-helper | 30 | gpu_delta_net.rs:893/:970 (ops2→op3) |
| **DN-3** | gpu_delta_net.rs:1205–1206 (in `_into`) | `dispatch_ssm_conv` writes `qkv_conv` (Q/K/V slice views) + `conv_state_out` | `apply_l2_norm_per_head` (×2) + `apply_proj` (alpha/beta) read | RAW | intra-helper | 30 | gpu_delta_net.rs:893/:970 (op3→ops5/6) |
| **DN-4** | gpu_delta_net.rs:1221–1222 (in `_into`) | `apply_l2_norm_per_head` + `apply_proj` write `q_l2`/`k_normed`/`alpha_logit`/`beta_logit` | `scalar_mul_f32` reads `q_l2`; `dispatch_compute_g_beta` reads alpha/beta | RAW | intra-helper | 30 | gpu_delta_net.rs:893/:970 (ops5/6→q_scale+g_beta) |
| **DN-5** | gpu_delta_net.rs:1234–1235 (in `_into`) | `scalar_mul_f32` writes `q_scaled`; `dispatch_compute_g_beta` writes `g_buf`/`beta_buf` | `dispatch_gated_delta_net` reads `q_scaled`/`k_normed`/`v_gpu`/`g_buf`/`beta_buf` | RAW | intra-helper | 30 | gpu_delta_net.rs:893/:970 (q_scale+g_beta→op7) |
| **DN-6** | gpu_delta_net.rs:1242–1243 (in `_into`) | `dispatch_gated_delta_net` writes `attn_out_buf` + `state_out` | `dispatch_ssm_norm_gate` reads `attn_out_buf`, `z` | RAW | intra-helper | 30 | gpu_delta_net.rs:893/:970 (op7→op8) |
| **DN-7** | gpu_delta_net.rs:1250–1251 (in `_into`) | `dispatch_ssm_norm_gate` writes `gated_buf` | `apply_proj` (out) reads `gated_buf` | RAW | intra-helper | 30 | gpu_delta_net.rs:893/:970 (op8→op9) |
| **IL-1** | NEW barrier in `forward_gpu_single_cb` between attn helper return and `dispatch_fused_residual_norm_f32` | FullAttn `out` (proj `wo`) OR DeltaNet `output` (proj `ssm_out`) | `dispatch_fused_residual_norm_f32` reads `attn_out` + `hidden` | RAW | inter-helper | 40 | gpu_full_attn.rs:1211/:1229 → forward_gpu.rs:1422  AND  gpu_delta_net.rs:893/:970 → forward_gpu.rs:1422 |
| **MOE-1** | forward_gpu.rs:1439 (already in greedy MoeQ branch, preserved) | `dispatch_fused_residual_norm_f32` writes `ffn_input_buf` + `ffn_residual_buf` | `build_moe_ffn_layer_gpu_q_into` reads `ffn_input_buf`, accumulates onto `ffn_residual_buf` | RAW | intra-block | 40 | forward_gpu.rs:1422/:1445 (intra) |
| **MOE-internal** | preserved inside `build_moe_ffn_layer_gpu_q_into` (gpu_ffn.rs A→F pipeline) | 5 internal barriers between dispatch groups | (same helper) | RAW | intra-helper | 40 | gpu_ffn.rs:997/:1001 (wrapper) — rows already inside `_into` |
| **IL-2** | NEW barrier in `forward_gpu_single_cb` between MoE return and next layer's attn input (or layer-39 → output-norm) | `build_moe_ffn_layer_gpu_q_into` writes `ffn_out` (residual folded) | next layer's `apply_pre_attn_rms_norm` (FA) or `apply_pre_norm` (DN) reads `hidden`; OR for last layer: output `dispatch_rms_norm` reads `hidden` | RAW | inter-layer (or layer-39→output-head) | 40 | forward_gpu.rs:1422/:1445 → next layer or output norm |
| **OH-1** | NEW barrier in `forward_gpu_single_cb` between output rms_norm and lm_head_q4 | `dispatch_rms_norm` (output) writes `norm_out_buf` | `apply_linear_projection_f32_into` (lm_head_q4) reads `norm_out_buf` | RAW | inter-stage (output head) | 1 | forward_gpu.rs:393/:399 → :403 |
| **OH-2** | NEW barrier in `forward_gpu_single_cb` between lm_head_q4 and argmax | `apply_linear_projection_f32_into` writes `logits_buf` (vocab F32) | `dispatch_argmax_f32` reads `logits_buf` | RAW | inter-stage (output head) | 1 | forward_gpu.rs:403/:409 → :412 |
| **TERM** | terminal `enc.commit_and_wait_labeled("decode-token")` at end of `forward_gpu_single_cb` | `dispatch_argmax_f32` writes `out_index` (4 bytes) | CPU `out_index.as_slice::<u32>()` reads token id | host-read | terminal | 1 | forward_gpu.rs:412/:417 (fuse_safe=NO) |

### 2.1 Per-token barrier dispatch totals

| Bucket | Per-layer count | Layer count | Per-decode-token total |
|---|---:|---:|---:|
| FullAttn intra-helper (FA-1, FA-2, FA-3, FA-5) | 4 | 10 | 40 |
| FullAttn inter-stage (FA-4, FA-6) | 2 | 10 | 20 |
| DeltaNet intra-helper (DN-1..DN-7) | 7 | 30 | 210 |
| Attn → MoE inter-helper (IL-1) | 1 | 40 | 40 |
| MoE intra-block (MOE-1) | 1 | 40 | 40 |
| MoE internal (preserved inside `_into`) | 5 | 40 | 200 |
| MoE → next-layer or → output-norm (IL-2) | 1 | 40 | 40 |
| Output head (OH-1, OH-2) | 2 | 1 | 2 |
| Terminal commit_and_wait (TERM) | — | 1 | 1 |

**Aggregate per-decode-token barriers:** 40 + 20 + 210 + 40 + 40 + 200 + 40 + 2 = **592 `enc.memory_barrier()` calls** + 1 terminal `commit_and_wait_labeled`. The 102 fuse_safe=YES P1-audit rows from ADR-015 are exactly the 19 unique table rows above (excluding TERM); the 592 figure is each row's per-layer instance count summed across all layers.

**NEW barriers introduced by 8a-A vs current `_into` helpers + greedy:**
- IL-1: ×40 (one per layer; new between attn out and fused_res_norm)
- IL-2: ×40 (one per layer; new between MoE FFN out and next consumer)
- OH-1: ×1 (new between output norm and lm_head_q4)
- OH-2: ×1 (new between lm_head_q4 and argmax)

= **82 NEW per-decode-token barriers**. The other 510 dispatches already
exist inside `build_gated_attn_layer_into`, `build_delta_net_layer_into`,
and `build_moe_ffn_layer_gpu_q_into` (preserved verbatim).

**Removed under single-CB:** every `enc.commit_labeled(...)` /
`enc.commit_and_wait*(...)` inside the per-layer loop (102 of them) plus
the 3 commits in `apply_output_head_gpu_greedy` — total **105 commit
calls eliminated** per decode token. Net effect: 103 CBs → 1 CB
(+ 82 cheap `memoryBarrierWithScope` IPC-free GPU-side barriers).

---

## 3. Helper-signature audit

Status as of branch HEAD `cfa/cfa-20260427-adr015-iter8a/claude` (built
from `f1ae8dc`-era). Live audit, not historical claim.

| Helper | Current signature | Encoder behavior | `_into` variant exists? | 8a-B coder action |
|---|---|---|---|---|
| `apply_pre_attn_rms_norm` (gpu_full_attn.rs:478) | `(&mut Encoder, registry, device, …) → MlxBuffer` | does NOT commit | YES (already by-ref) | none |
| `apply_linear_projection_f32_pooled` (gpu_full_attn.rs:733) | `(&mut Encoder, registry, device, …) → MlxBuffer` | does NOT commit | YES (already by-ref) | none |
| `apply_linear_projection_f32_into` (gpu_full_attn.rs:644) | `(&mut Encoder, registry, device, dst: &mut MlxBuffer, …) → ()` | does NOT commit | YES (already exists) | none — used by lm_head |
| `apply_q_or_k_per_head_rms_norm` (gpu_full_attn.rs:309) | `(&mut Encoder, registry, device, …) → MlxBuffer` | does NOT commit | YES (already by-ref) | none |
| `apply_imrope` (gpu_full_attn.rs:361) | `(&mut Encoder, registry, device, …) → MlxBuffer` | does NOT commit | YES (already by-ref) | none |
| `apply_sigmoid_gate_multiply` (gpu_full_attn.rs:431) | `(&mut Encoder, registry, device, …) → MlxBuffer` | does NOT commit | YES (already by-ref) | none |
| `dispatch_kv_cache_copy_seq_f32_dual` | `(&mut Encoder, registry, metal_device, …) → ()` | does NOT commit | YES (already by-ref) | none |
| `dispatch_sdpa_decode` | `(&mut Encoder, registry, device, …) → ()` | does NOT commit | YES (already by-ref) | none |
| `apply_sdpa_with_kv_cache` (gpu_full_attn.rs:930) | `(device, registry, q, k, v, slot, …) → MlxBuffer` opens own encoder + commits | NO (legacy wrapper) | NOT EXTRACTED — see §3.1 below. **NOT CALLED FROM SINGLE-CB.** Its decode-path body (lines 966–995) is inlined directly into `build_gated_attn_layer_into`. |
| `build_gated_attn_layer` (gpu_full_attn.rs:1088) | `(device, registry, …) → MlxBuffer` opens 3 encoders + 3 commits | NO (legacy wrapper) | **YES — `build_gated_attn_layer_into` exists at gpu_full_attn.rs:1278–1410** | use the `_into` variant; legacy stays for tests + prefill |
| `build_delta_net_layer` (gpu_delta_net.rs:776) | `(device, registry, …) → MlxBuffer` opens 1 encoder + 1 commit | NO (legacy wrapper) | **YES — `build_delta_net_layer_into` exists at gpu_delta_net.rs:1097–1262** | use the `_into` variant; legacy stays for tests + prefill |
| `dispatch_fused_residual_norm_f32` | `(&mut Encoder, registry, metal_device, …) → ()` | does NOT commit | YES (already by-ref) | none |
| `build_moe_ffn_layer_gpu_q_into` (gpu_ffn.rs:1145) | `(&mut Encoder, device, registry, x, w, shape, residual) → MlxBuffer` | does NOT commit | YES (already exists) | none — used by greedy MoeQ branch |
| `build_moe_ffn_layer_gpu_q` (gpu_ffn.rs:1086) | legacy wrapper | NO (legacy) | wrapper around `_into` | not used by single-CB |
| `dispatch_rms_norm` (used by output head) | `(&mut Encoder, registry, metal_device, …) → ()` | does NOT commit | YES (already by-ref) | none — used in single-CB output head C.1 |
| `dispatch_argmax_f32` (used by output head) | `(&mut Encoder, registry, metal_device, …) → ()` | does NOT commit | YES (already by-ref) | none — used in single-CB output head C.3 |
| `apply_output_head_gpu_greedy` (forward_gpu.rs:363) | `(device, registry, …) → u32` opens 3 encoders, commits each | NO (legacy wrapper) | NOT EXTRACTED — bypassed; the 3 dispatches are inlined into `forward_gpu_single_cb` body §1 stage C |
| `residual_add_gpu` (forward_gpu.rs:436) | `(dst, src, device, registry) → MlxBuffer` opens own encoder + `commit_and_wait()` | NO (legacy) | NOT REQUIRED — only reached on F32-MoE path which is hard-errored in single-CB |

### 3.1 Why `apply_sdpa_with_kv_cache_into` is NOT a separate helper

The §P1 audit row for `gpu_full_attn.rs:959/:983` (kv-cache + SDPA decode)
sits inside `apply_sdpa_with_kv_cache`. In branch HEAD, the existing
`build_gated_attn_layer_into` (lines 1278–1410) inlines the seq=1
decode-path body directly:

- lines 1369–1376: `dispatch_kv_cache_copy_seq_f32_dual` (replaces the
  helper's :972–:980).
- line 1379: `enc.memory_barrier()` (replaces the helper's :981 — FA-5
  in §2 above).
- lines 1381–1387: `dispatch_sdpa_decode` (replaces the helper's
  :983–:989).
- line 1394: `slot.current_len[0] = kv_seq_len` (CPU counter — replaces
  the helper's :1056).

This is the correct factoring: `apply_sdpa_with_kv_cache` carries a
prefill (seq>1) branch that requires CPU permutes + downloads
(:996–:1052) and is fundamentally non-fusable. Splitting the helper
would force every `_into` caller to duplicate either the prefill
fallback OR the decode body. Inlining the decode body directly into
`build_gated_attn_layer_into` (which already debug-asserts seq_len==1)
keeps the prefill code path intact in the legacy wrapper without
introducing a half-fused helper.

### 3.2 Net helper changes for 8a-B

**ZERO NEW `_into` variants needed.** All three originally-flagged risks
(`build_delta_net_layer_into` and the implicit FullAttn split) are
already resolved upstream by an earlier ADR-015 sub-iter that landed
`build_gated_attn_layer_into` (gpu_full_attn.rs:1278) and
`build_delta_net_layer_into` (gpu_delta_net.rs:1097). The coder needs
only to:

1. write the new top-level `forward_gpu_single_cb` body (§1 pseudocode);
2. call the existing `_into` helpers for FullAttn, DeltaNet, MoE FFN;
3. inline the 3-commit output head from `apply_output_head_gpu_greedy`
   directly into the new function (3 dispatches + 2 barriers + 1
   terminal commit);
4. swap the production caller in `src/serve/mod.rs:960` from
   `forward_gpu_greedy` → `forward_gpu_single_cb`.

This downgrades spec R2 from "AMBIGUOUS-NEEDS-SPIKE" to "CLOSED — verified
against branch HEAD". The 8a-A → 8a-B contract is mechanical.

---

## 4. Env-gate dispatch design

Single env var: `HF2Q_LEGACY_PER_LAYER_CB`.

Routing rules (per spec invariants 6–7):

| `HF2Q_LEGACY_PER_LAYER_CB` value | `forward_gpu_single_cb` behavior |
|---|---|
| unset (default) | execute single-CB body (1 encoder, 1 terminal commit) |
| `"1"` | early-return `self.forward_gpu_greedy(tokens, positions_flat, kv_cache)` |
| any other value (e.g. `"0"`, `"true"`, empty string) | execute single-CB body (default) |

The check is the FIRST statement in `forward_gpu_single_cb`, before any
allocation or cache-population work. This satisfies spec line 32:
`if std::env::var("HF2Q_LEGACY_PER_LAYER_CB").as_deref() == Ok("1") { ... }`.

**Caller cutover.** The single production caller of `forward_gpu_greedy`
in qwen35 territory is `src/serve/mod.rs:960`. The 8a-B coder pivots that
call site to `forward_gpu_single_cb`. With the env-gate above, this is
the **only** code change needed for default-on routing — when
`HF2Q_LEGACY_PER_LAYER_CB=1` is set, the same call site transparently
hops back to the legacy path via the gate. No fallback code, no
double-routing, no parity-failure auto-revert (per
`feedback_never_ship_fallback_without_rootcause`).

`forward_gpu_greedy` is NOT deleted in iter8a (spec Non-Goal 6: "Removing
forward_gpu_greedy or any legacy code (P8 task; ≥7 day soak window)").
It remains the legacy escape hatch through P8.

**Telemetry (8a-C scope).** When `HF2Q_PROFILE_GPU_TS=1` is set, the
existing per-bucket timestamp instrumentation in `mlx-native` records
each `commit_and_wait_labeled` label. Single-CB emits exactly one label
`"decode-token"` per token; legacy path emits 103 distinct labels. The
AC5 reviewer counts label occurrences via `grep -c 'graph_session'`.
Detailed instrumentation wiring is the 8a-C subtask, not 8a-A.

**No new env vars.** Spec line 33 explicitly forbids new env vars.
`HF2Q_DECODE_PROFILE` and `MLX_PROFILE_CB` (already present in
`forward_gpu_greedy` at lines 1269 + 1564) are NOT plumbed into
`forward_gpu_single_cb` for iter8a — they belong to the legacy path
where per-CB attribution makes sense; on single-CB the per-CB dimension
collapses to 1 by construction.

---

## 5. Risk register

R1 — `apply_sdpa_with_kv_cache` encoder lifetime.
- **Spike.** Does inlining `dispatch_kv_cache_copy_seq_f32_dual` +
  `dispatch_sdpa_decode` directly into `build_gated_attn_layer_into`
  trip a `slot.k`/`slot.v` borrow conflict?
- **Verdict: CLOSED.** The existing `build_gated_attn_layer_into`
  (gpu_full_attn.rs:1278–1410) already does exactly this inline today;
  the `_into` helper takes `kv_cache_slot: &mut FullAttnKvSlot` and
  writes/reads the same buffer pair via the shared encoder. Borrow
  checker passes (file builds at branch HEAD). 8a-B reuses verbatim.

R2 — DeltaNet encoder lifetime (per spec R2).
- **Spike.** Does `build_delta_net_layer_into` correctly preserve the 7
  internal barriers (DN-1..DN-7)? Are conv_state_in/out + state_in/out
  buffer aliasing rules satisfied with a shared encoder?
- **Verdict: CLOSED.** The `_into` impl at gpu_delta_net.rs:1097–1262
  fires `enc.memory_barrier()` at lines 1189, 1199, 1206, 1222, 1235,
  1243, 1251 — exactly 7, matching the §P1 audit
  ":899/:909/:916/:932/:945/:953/:961" enumeration (op-numbering
  preserved across the refactor). State-buffer plumbing matches the
  greedy path. The kv_cache slot pointer-swap (`swap_conv_state` /
  `swap_recurrent`) is CPU metadata (no buffer mutation) so it can fire
  while `enc` is still open.

R3 — `decode_pool::reset_decode_pool` + arena lifetime across 40 layers.
- **Spike.** Does the thread-local arena's `pooled_alloc_buffer` produce
  buffers that survive the entire 1-CB lifetime (from `enc` open through
  terminal commit)?
- **Verdict: EXPECTED-PASS.** Per `decode_pool.rs:30` doc:
  "ARC clones [...] move back to the free list for the next token's
  reuse" — the reset-at-top + free-on-token-end pattern is unchanged.
  In single-CB form the arena allocations span the full encoder
  lifetime (longer than per-CB form) but the buffers are not freed
  until the next `reset_decode_pool` call (next token), so they are
  GPU-live for the whole CB. Verified by inspection of
  `pooled_alloc_buffer` (returns `MlxBuffer` clones; no premature drop).
- **Mitigation if it fails.** Parity tests (`single_cb_parity_*`) catch
  use-after-free as garbage logits. No production-ship risk.

R4 — borrow-check across 40 `&mut enc` threadings.
- **Spike.** Does Rust borrow-check accept 40 sequential `&mut enc`
  passes through `build_gated_attn_layer_into` /
  `build_delta_net_layer_into` / `build_moe_ffn_layer_gpu_q_into`
  without conflicting with the `&mut registry` and
  `&mut kv_cache.full_attn[r]` borrows?
- **Verdict: EXPECTED-PASS.** Each helper takes `&mut Encoder` and
  `&mut KernelRegistry` by reference and returns before the next
  helper claims them — sequential, non-overlapping mutable borrows.
  The kv_cache slot borrow via `&mut kv_cache.full_attn[rank]` is
  scoped to a single FullAttn iteration (via match arm); subsequent
  iterations re-borrow at the new index. Identical pattern to greedy
  forward at lines 1290–1311 which already builds.
- **Failure shape.** If it fails, it fails at compile time, not
  runtime — coder sees a clear E0499/E0502 and the fix is local.

R5 — FullAttn `_into` only handles seq_len=1 + head_dim%32==0.
- **Spike.** What if a future model has head_dim%32!=0?
- **Verdict: ACCEPTABLE for iter8a.** Spec scope is `dwq46`; qwen3.5/3.6
  ship head_dim=128 (% 32 == 0). The `_into` helper debug-asserts the
  constraint at lines 1297–1298. If a future model violates it, the
  caller falls back via `HF2Q_LEGACY_PER_LAYER_CB=1`. iter8b/c
  generalizes if/when needed — tracked separately, not iter8a scope.

R6 — single-CB encodes >GPU command-buffer command limit.
- **Spike.** Does Apple Metal's per-CB command count hard cap (~4M
  commands on M-series; un-published) get exceeded by 40 layers ×
  ~25 dispatches/layer + output head?
- **Verdict: WELL-UNDER-LIMIT.** Total dispatches: ~1005, plus ~600
  memory_barriers = ~1600 encoded commands. llama.cpp routinely fits
  1-2 CBs per decode token on the same hardware at ~10× this density —
  confirmed by `project_decode_parity_achieved` memory. Not a real
  risk.

R7 — Apex bench RAM contention with parallel ADR-014 worktrees (spec R4).
- **Verdict: OUT-OF-DESIGN-SCOPE.** Tester (8a-E) acquires `bench-lock`
  and aborts on thermal warning per `ram_discipline`. Architect (8a-A)
  has no surface area here.

R8 — Negative-result: single-CB latency vs throughput trade-off (spec R5).
- **Spike.** If encoder-lifetime bookkeeping cost exceeds per-CB
  overhead, we could land below the 110.0 t/s floor.
- **Verdict: PLAUSIBLE BUT BOUNDED.** ADR-015 §P3a''' apex H3 bucket
  measured 125 µs/token = ~1.5 t/s ceiling. Even a pessimistic case
  recovers 50% (62 µs/token = +0.5 t/s). Spec AC4 fails only below
  109.0 (5% regression); single-CB structural cleanup well within
  bounds. AC4 stretch (110.6) requires full 125 µs recovery — likely
  but not guaranteed. If we land at 110.1–110.5 the implementation is
  still a structural win that unlocks iter8b dual-buffer work.
- **Decision rule.** Report raw bench numbers; let the launcher decide
  per the spec rubric.

R9 — Spec R3 — encoder-thread parameter explosion (architect decision).
- **Decision: option (a), explicit `&mut Encoder` threading.**
  Rationale: existing helper family (`apply_pre_attn_rms_norm`,
  `apply_linear_projection_f32_pooled`, `apply_q_or_k_per_head_rms_norm`,
  `apply_imrope`, `apply_sigmoid_gate_multiply`, `dispatch_*`) already
  takes `&mut enc, &mut registry, &device` as its first three params —
  the established convention. A new `EncoderCtx` struct would require
  rewriting every existing helper signature (off-limits per
  files_off_limits — `/opt/mlx-native/**` is substrate-stable; the
  helper files inside qwen35 are fence-free but rewriting all 16 of
  them adds noise without benefit) OR introducing call-site adapter
  glue. Spec invariants "No new compiler warnings" + "single function
  under 300 lines" both push toward minimal new abstraction. Threading
  `&mut enc` is 0 new types.

---

## 6. What 8a-B must NOT do

Hard fences from spec.json + this design:

- Do NOT touch `/opt/mlx-native/**` (substrate stable, off-limits).
- Do NOT touch `src/serve/forward_mlx.rs` (gemma — iter8b scope).
- Do NOT touch `src/quantize/**`, `src/calibrate/**`, `src/convert/**`,
  `src/quality/**`, `src/backends/gguf.rs`, `src/ir/{mod,lazy}.rs`,
  `src/serve/api/grammar/**`.
- Do NOT remove `forward_gpu_greedy` (P8 task, soak window required).
- Do NOT add new env vars beyond `HF2Q_LEGACY_PER_LAYER_CB`.
- Do NOT add new `unsafe` blocks beyond the existing `decode_bufs` cast.
- Do NOT introduce a parity-failure-triggered fallback to legacy
  (cf. `feedback_never_ship_fallback_without_rootcause`); a test
  failure means fix the bug, not gate the path off.
- Do NOT add `#[allow(...)]` attributes (judging rubric −10/each).
- Do NOT call `apply_output_head_gpu_greedy` from
  `forward_gpu_single_cb`; inline its 3 dispatches directly so they
  share `enc`.
- Do NOT call `build_gated_attn_layer` (the legacy 3-encoder version);
  always use `build_gated_attn_layer_into`.
- Do NOT call `build_delta_net_layer` (the 1-encoder version); always
  use `build_delta_net_layer_into`.

---

## 7. Coder execution order (suggested for 8a-B)

1. Read this doc + the spec.
2. Replace `forward_gpu_single_cb` body (forward_gpu.rs:1610–1626) with
   the §1 pseudocode, copying the cache/decode-buf setup blocks verbatim
   from `forward_gpu_greedy` lines 1108–1246.
3. Inline the output head (greedy lines 364–425 minus its 3
   `commit_*()` calls) as §1 stage C, replacing the
   `apply_output_head_gpu_greedy` call.
4. Pivot `src/serve/mod.rs:960` from `forward_gpu_greedy(...)` to
   `forward_gpu_single_cb(...)`.
5. `cargo build --release --bin hf2q` — must succeed with the same
   warning baseline as HEAD.
6. Hand off to 8a-C (telemetry hookup) and 8a-D (parity tests).

---

## 8. Cross-reference

- Codex sibling design (independent draft):
  `/opt/hf2q/.cfa-worktrees/cfa-20260427-adr015-iter8a-codex/docs/iter8a-barrier-graph-codex.md`.
- Both designs converge on the same barrier topology. The codex
  variant proposes splitting `build_gated_attn_layer_into` into three
  finer-grained helpers (`build_gated_attn_layer_ops1to4_into`,
  `apply_sdpa_with_kv_cache_into`,
  `build_gated_attn_layer_ops6to7_into`); this Claude variant uses the
  existing single `build_gated_attn_layer_into` as-is on the grounds
  that the per-stage commit boundaries it replaces are already
  collapsed into `enc.memory_barrier()` calls at lines 1352–1354 and
  1389–1391, leaving no work for a finer split. Mechanical correctness
  is identical between the two designs; this variant has a smaller
  diff-stat (judging rubric tiebreaker).
