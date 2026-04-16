//! True batched prefill forward pass — ADR-009 Phase 3A.
//!
//! Unlike `forward_prefill` (which loops per-token), this processes the
//! entire prompt through each transformer layer in ONE batched session per
//! layer, matching llama.cpp's default batched prefill kernel dispatch.
//!
//! Key differences from per-token prefill:
//! - Embedding: single dispatch gathers all seq_len rows
//! - QKV projections: single `quantized_matmul_ggml` with `m = seq_len`
//! - Head norm + RoPE: single `fused_head_norm_rope_batch_f32` dispatch
//!   with `n_heads * seq_len` threadgroups (kernel already supports
//!   seq_idx = head_id / n_heads)
//! - SDPA: ONE call to the tiled `sdpa` kernel with `seq_len > 1` and
//!   causal mask covering all positions at once
//! - O-proj / MLP: batched via `m = seq_len`
//! - MoE: fused_moe_routing_batch_f32 + quantized_matmul_id_ggml with
//!   n_tokens = seq_len, then moe_swiglu_seq + quantized_matmul_id with
//!   n_tokens = seq_len*top_k, then moe_weighted_sum_seq
//! - End-of-layer: batched fused_norm_add_scalar with rows = seq_len
//!
//! Gated by `HF2Q_BATCHED_PREFILL=1`.

use anyhow::Result;
use mlx_native::{DType, MlxBuffer};
use mlx_native::ops::dense_gemm::DenseGemmF16Params;
use std::time::Instant;

use crate::debug::INVESTIGATION_ENV;
use super::config::LayerType;
use super::forward_mlx::{
    DenseKvBuffers, MlxModelWeights, dispatch_qmatmul, dispatch_rms_norm_unit_perhead,
};
use super::gpu::GpuContext;

impl MlxModelWeights {
    /// True batched prefill with single-shot dense SDPA over the whole prompt.
    ///
    /// Returns the first decode token (greedy argmax of last-row logits).
    pub fn forward_prefill_batched(
        &mut self,
        prompt_tokens: &[u32],
        max_decode_tokens: usize,
        gpu: &mut GpuContext,
    ) -> Result<u32> {
        let seq_len = prompt_tokens.len();
        if seq_len == 0 {
            anyhow::bail!("forward_prefill_batched: empty prompt");
        }
        let hs = self.hidden_size;
        let num_layers = self.layers.len();
        let vocab_size = self.vocab_size;
        let eps = self.rms_norm_eps;
        let nh = self.num_attention_heads;
        let intermediate = self.intermediate_size;
        let num_experts = self.num_experts;

        let f32_sz = std::mem::size_of::<f32>();
        let u32_sz = std::mem::size_of::<u32>();

        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let metal_dev = dev.metal_device();

        let use_f16_kv = INVESTIGATION_ENV.f16_kv;
        let kv_dtype = if use_f16_kv { DType::F16 } else { DType::F32 };
        let kv_elem_bytes = if use_f16_kv { 2 } else { 4 };
        eprintln!("Batched prefill: KV={:?}, seq_len={}", kv_dtype, seq_len);

        // -------------------------------------------------------------------
        // Per-layer dense KV buffers [n_kv_heads, capacity, head_dim]
        // Sliding layers use ring buffer (capacity = sliding_window) and
        // dense flash_attn_vec uses mask_type=1 (causal); the ring itself
        // applies the sliding-window constraint. Attention is permutation-
        // invariant over cached K,V (RoPE is baked in pre-cache), so ring
        // slot order doesn't affect correctness.
        // -------------------------------------------------------------------
        let linear_capacity = seq_len + max_decode_tokens;
        let sw = self.sliding_window;
        let mut dense_kvs_vec: Vec<DenseKvBuffers> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let nkv = self.num_kv_heads[layer_idx];
            let hd = self.head_dims[layer_idx];
            let layer_is_ring = self.layer_types[layer_idx] == LayerType::Sliding;
            let capacity = if layer_is_ring { sw } else { linear_capacity };
            let n = nkv * capacity * hd;
            let k = dev.alloc_buffer(n * kv_elem_bytes, kv_dtype,
                                      vec![nkv, capacity, hd])
                .map_err(|e| anyhow::anyhow!("batched dense K L{layer_idx}: {e}"))?;
            let v = dev.alloc_buffer(n * kv_elem_bytes, kv_dtype,
                                      vec![nkv, capacity, hd])
                .map_err(|e| anyhow::anyhow!("batched dense V L{layer_idx}: {e}"))?;
            dense_kvs_vec.push(DenseKvBuffers { k, v, capacity, is_sliding: layer_is_ring });
        }
        let max_nh = nh;
        let max_hd = self.head_dims.iter().copied().max().unwrap_or(512);
        let tmp_bytes = mlx_native::ops::flash_attn_vec::tmp_buffer_bytes(
            max_nh as u32, max_hd as u32);
        let sdpa_tmp = dev.alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
            .map_err(|e| anyhow::anyhow!("batched sdpa_tmp: {e}"))?;

        // -------------------------------------------------------------------
        // Batched activation buffers (seq_len × ...)
        // -------------------------------------------------------------------
        let alloc_f32 = |n: usize, name: &str| -> Result<MlxBuffer> {
            dev.alloc_buffer(n * f32_sz, DType::F32, vec![n])
                .map_err(|e| anyhow::anyhow!("batched alloc {name}: {e}"))
        };
        let alloc_u32 = |n: usize, name: &str| -> Result<MlxBuffer> {
            dev.alloc_buffer(n * u32_sz, DType::U32, vec![n])
                .map_err(|e| anyhow::anyhow!("batched alloc {name}: {e}"))
        };

        let max_nkv = self.num_kv_heads.iter().copied().max().unwrap_or(8);
        let pf_hidden = alloc_f32(seq_len * hs, "pf_hidden")?;
        let pf_residual = alloc_f32(seq_len * hs, "pf_residual")?;
        let pf_norm_out = alloc_f32(seq_len * hs, "pf_norm_out")?;
        let pf_moe_norm_out = alloc_f32(seq_len * hs, "pf_moe_norm_out")?;
        let pf_router_norm_out = alloc_f32(seq_len * hs, "pf_router_norm_out")?;
        let mut pf_attn_out = alloc_f32(seq_len * hs, "pf_attn_out")?;
        let pf_mlp_down_out = alloc_f32(seq_len * hs, "pf_mlp_down_out")?;

        let mut pf_q = alloc_f32(seq_len * nh * max_hd, "pf_q")?;
        let mut pf_k = alloc_f32(seq_len * max_nkv * max_hd, "pf_k")?;
        let mut pf_v = alloc_f32(seq_len * max_nkv * max_hd, "pf_v")?;
        let pf_q_normed = alloc_f32(seq_len * nh * max_hd, "pf_q_normed")?;
        let pf_k_normed = alloc_f32(seq_len * max_nkv * max_hd, "pf_k_normed")?;
        let pf_v_normed = alloc_f32(seq_len * max_nkv * max_hd, "pf_v_normed")?;

        let pf_q_perm = alloc_f32(nh * seq_len * max_hd, "pf_q_perm")?;
        let pf_k_perm = alloc_f32(max_nkv * seq_len * max_hd, "pf_k_perm")?;
        let pf_v_perm = alloc_f32(max_nkv * seq_len * max_hd, "pf_v_perm")?;
        let pf_sdpa_out_perm = alloc_f32(nh * seq_len * max_hd, "pf_sdpa_out_perm")?;
        let pf_sdpa_out = alloc_f32(seq_len * nh * max_hd, "pf_sdpa_out")?;

        let mut pf_mlp_gate = alloc_f32(seq_len * intermediate, "pf_mlp_gate")?;
        let mut pf_mlp_up = alloc_f32(seq_len * intermediate, "pf_mlp_up")?;
        let pf_mlp_fused = alloc_f32(seq_len * intermediate, "pf_mlp_fused")?;
        let mut pf_mlp_down = alloc_f32(seq_len * hs, "pf_mlp_down")?;

        let top_k_max = self.layers.iter().map(|l| l.moe.top_k).max().unwrap_or(2);
        let moe_int_max = self.layers.iter().map(|l| l.moe.moe_intermediate_size).max().unwrap_or(0);
        let mut pf_router_logits = alloc_f32(seq_len * num_experts, "pf_router_logits")?;
        let pf_expert_ids = alloc_u32(seq_len * top_k_max, "pf_expert_ids")?;
        let pf_routing_weights = alloc_f32(seq_len * top_k_max, "pf_routing_weights")?;
        let mut pf_moe_gate_up = alloc_f32(seq_len * top_k_max * 2 * moe_int_max, "pf_moe_gate_up")?;
        let pf_moe_swiglu = alloc_f32(seq_len * top_k_max * moe_int_max, "pf_moe_swiglu")?;
        let mut pf_moe_down = alloc_f32(seq_len * top_k_max * hs, "pf_moe_down")?;
        let pf_moe_accum = alloc_f32(seq_len * hs, "pf_moe_accum")?;

        let mut pf_positions = alloc_u32(seq_len, "pf_positions")?;
        {
            let p: &mut [u32] = pf_positions.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("positions write: {e}"))?;
            for i in 0..seq_len { p[i] = i as u32; }
        }
        let mut pf_token_ids = alloc_u32(seq_len, "pf_token_ids")?;
        {
            let t: &mut [u32] = pf_token_ids.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("token_ids write: {e}"))?;
            for (i, &tok) in prompt_tokens.iter().enumerate() { t[i] = tok; }
        }

        // -------------------------------------------------------------------
        // SESSION 1: batched embedding
        // -------------------------------------------------------------------
        let prefill_start = Instant::now();
        {
            let mut s = exec.begin()
                .map_err(|e| anyhow::anyhow!("batched embed session: {e}"))?;
            s.track_dispatch(&[&self.embed_weight, &pf_token_ids], &[&pf_hidden]);
            mlx_native::ops::elementwise::embedding_gather_scale_batch_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.embed_weight,
                &pf_token_ids,
                &pf_hidden,
                hs, seq_len,
                (hs as f32).sqrt(),
            ).map_err(|e| anyhow::anyhow!("batched embed: {e}"))?;
            s.finish()
                .map_err(|e| anyhow::anyhow!("batched embed finish: {e}"))?;
        }

        // -------------------------------------------------------------------
        // Per-layer forward pass
        // -------------------------------------------------------------------
        // ADR-010 batched sub-stage dump anchor. HF2Q_BATCHED_DUMP="layer,tok"
        // (e.g. "7,34"). When set and the target layer finishes its batched
        // forward pass, dump Q_normed row, K/V_normed row, dense K/V cache
        // slice [nkv, tok+1, hd], and sdpa_out row at the target token.
        let batched_dump: Option<(usize, usize)> = INVESTIGATION_ENV.batched_dump;
        let batched_dump_dir: &str = &INVESTIGATION_ENV.dump_dir;

        for layer_idx in 0..num_layers {
            let hd = self.head_dims[layer_idx];
            let nkv = self.num_kv_heads[layer_idx];
            let is_sliding = self.layer_types[layer_idx] == LayerType::Sliding;
            let top_k = self.layers[layer_idx].moe.top_k;
            let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;

            // ADR-010 early dump: capture layer INPUT (= previous layer's output)
            // before any modification. pf_hidden at end of layer holds the NEXT
            // layer's input, so we must grab it here, at start of target layer.
            // Two modes:
            //   HF2Q_BATCHED_DUMP="layer,tok" — dump only for that target layer
            //   HF2Q_BATCHED_LAYER_SCAN="tok" — dump pf_hidden row `tok` for
            //     EVERY layer (per-layer l_out scan for cross-layer drift bisection)
            let layer_scan_tok: Option<usize> = INVESTIGATION_ENV.batched_layer_scan;
            let should_dump_input = match (batched_dump, layer_scan_tok) {
                (Some((dump_layer, tok)), _) if dump_layer == layer_idx => Some(tok),
                (_, Some(tok)) => Some(tok),
                _ => None,
            };
            if let Some(target_tok) = should_dump_input {
                if target_tok < seq_len && !use_f16_kv {
                    let h: &[f32] = pf_hidden.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_hidden L{layer_idx}: {e}"))?;
                    let off = target_tok * hs;
                    let row = &h[off..off + hs];
                    let path = format!(
                        "{batched_dump_dir}/hf2q_batched_pre_layer_hidden_row_layer{layer_idx:02}_tok{target_tok:03}.bin");
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len() * 4) };
                    std::fs::write(&path, bytes)
                        .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
                    eprintln!("[BATCHED DUMP] pre_layer_hidden_row L{layer_idx:02} [{}] f32 -> {path}", hs);
                }
            }

            let ff_gpu = if is_sliding { None }
                else { Some(&self.activations.rope_freq_factors_gpu) };
            let theta = if is_sliding { self.rope_theta_sliding }
                else { self.rope_theta_global };
            let half_rope = (hd / 2) as u32;

            let hd_norm_params = if is_sliding {
                &self.activations.norm_params_sliding_hd
            } else {
                &self.activations.norm_params_global_hd
            };

            // ================================================================
            // SESSION A: norm → QKV → head_norm+RoPE → permute → SDPA →
            //            permute_back → O-proj → post-attn norm+residual
            // ================================================================
            {
                let mut s = exec.begin()
                    .map_err(|e| anyhow::anyhow!("batched attn session L{layer_idx}: {e}"))?;

                // 1. Pre-attention norm over [seq_len, hs]
                s.barrier_between(
                    &[&pf_hidden, &self.layers[layer_idx].norms.input_layernorm],
                    &[&pf_norm_out],
                );
                s.rms_norm(
                    reg, metal_dev,
                    &pf_hidden,
                    &self.layers[layer_idx].norms.input_layernorm,
                    &pf_norm_out,
                    &self.activations.norm_params,
                    seq_len as u32, hs as u32,
                ).map_err(|e| anyhow::anyhow!("batched pre-attn norm L{layer_idx}: {e}"))?;

                // ADR-010 sub-stage dump: pf_norm_out is reused in session B
                // for the pre-feedforward norm, so the end-of-layer dump hook
                // reads the WRONG tensor. Snapshot it HERE, right after the
                // pre-attention RMS norm is written.
                if let Some((dump_layer, target_tok)) = batched_dump {
                    if dump_layer == layer_idx && target_tok < seq_len && !use_f16_kv {
                        s.finish().map_err(|e| anyhow::anyhow!("dump norm finish L{layer_idx}: {e}"))?;
                        let nrm: &[f32] = pf_norm_out.as_slice()
                            .map_err(|e| anyhow::anyhow!("dump pf_norm_out early L{layer_idx}: {e}"))?;
                        let off = target_tok * hs;
                        let row = &nrm[off..off + hs];
                        let path = format!(
                            "{batched_dump_dir}/hf2q_batched_post_input_norm_row_layer{layer_idx:02}_tok{target_tok:03}.bin");
                        let bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len() * 4) };
                        std::fs::write(&path, bytes)
                            .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
                        eprintln!("[BATCHED DUMP] post_input_norm_row (inline) [{hs}] f32 -> {path}");
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump norm restart L{layer_idx}: {e}"))?;
                    }
                }

                // 2. QKV projections (m = seq_len) — concurrent
                s.barrier_between(
                    &[&pf_norm_out],
                    &[&pf_q, &pf_k, &pf_v],
                );
                dispatch_qmatmul(&mut s, reg, dev, &pf_norm_out,
                    &self.layers[layer_idx].attn.q_proj, &mut pf_q, seq_len as u32)?;
                dispatch_qmatmul(&mut s, reg, dev, &pf_norm_out,
                    &self.layers[layer_idx].attn.k_proj, &mut pf_k, seq_len as u32)?;
                let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
                if !v_is_k {
                    dispatch_qmatmul(&mut s, reg, dev, &pf_norm_out,
                        self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                        &mut pf_v, seq_len as u32)?;
                }

                // 3. Batched fused head norm + RoPE on Q (with weight) and K (with weight)
                s.barrier_between(
                    &[&pf_q, &pf_k],
                    &[&pf_q_normed, &pf_k_normed],
                );
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_batch_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_q,
                    &pf_q_normed,
                    Some(&self.layers[layer_idx].attn.q_norm_weight),
                    &pf_positions,
                    ff_gpu,
                    nh as u32, hd as u32, half_rope,
                    seq_len as u32,
                    eps, theta,
                ).map_err(|e| anyhow::anyhow!("batched Q norm+RoPE L{layer_idx}: {e}"))?;
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_batch_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_k,
                    &pf_k_normed,
                    Some(&self.layers[layer_idx].attn.k_norm_weight),
                    &pf_positions,
                    ff_gpu,
                    nkv as u32, hd as u32, half_rope,
                    seq_len as u32,
                    eps, theta,
                ).map_err(|e| anyhow::anyhow!("batched K norm+RoPE L{layer_idx}: {e}"))?;

                // 4. V norm (unit RMS, no RoPE, per-head across seq_len)
                //    Layout: [seq_len * nkv, hd] — treat all positions' heads as rows
                if v_is_k {
                    s.barrier_between(
                        &[&pf_k],
                        &[&pf_v_normed],
                    );
                    dispatch_rms_norm_unit_perhead(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_k,
                        &pf_v_normed,
                        hd_norm_params,
                        (seq_len * nkv) as u32, hd as u32,
                    )?;
                } else {
                    s.barrier_between(
                        &[&pf_v],
                        &[&pf_v_normed],
                    );
                    dispatch_rms_norm_unit_perhead(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_v,
                        &pf_v_normed,
                        hd_norm_params,
                        (seq_len * nkv) as u32, hd as u32,
                    )?;
                }

                // 5. Permute [seq_len, n_heads, hd] → [n_heads, seq_len, hd]
                s.barrier_between(
                    &[&pf_q_normed, &pf_k_normed, &pf_v_normed],
                    &[&pf_q_perm, &pf_k_perm, &pf_v_perm],
                );
                mlx_native::ops::transpose::permute_021_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_q_normed, &pf_q_perm,
                    seq_len, nh, hd,
                ).map_err(|e| anyhow::anyhow!("batched permute Q L{layer_idx}: {e}"))?;
                mlx_native::ops::transpose::permute_021_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_k_normed, &pf_k_perm,
                    seq_len, nkv, hd,
                ).map_err(|e| anyhow::anyhow!("batched permute K L{layer_idx}: {e}"))?;
                mlx_native::ops::transpose::permute_021_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_v_normed, &pf_v_perm,
                    seq_len, nkv, hd,
                ).map_err(|e| anyhow::anyhow!("batched permute V L{layer_idx}: {e}"))?;

                // 6. Dense batched SDPA (tiled kernel):
                //    Q: [1, nh, seq_len, hd], K: [1, nkv, seq_len, hd], V: same
                //    kv_capacity = seq_len (tight packing, no pre-allocated capacity)
                //    scale = 1.0 for Gemma4 (per llama.cpp oracle)
                //    Causal mask: kernel uses abs_pos = kv_seq_len - seq_len + q_pos
                //                 which gives the correct triangular causal pattern
                //                 when seq_len == kv_seq_len.
                let sdpa_params = mlx_native::ops::sdpa::SdpaParams {
                    n_heads: nh as u32,
                    n_kv_heads: nkv as u32,
                    head_dim: hd as u32,
                    seq_len: seq_len as u32,
                    kv_seq_len: seq_len as u32,
                    scale: 1.0,
                    kv_capacity: seq_len as u32,
                };
                s.barrier_between(
                    &[&pf_q_perm, &pf_k_perm, &pf_v_perm],
                    &[&pf_sdpa_out_perm],
                );
                s.sdpa(
                    reg, dev,
                    &pf_q_perm, &pf_k_perm, &pf_v_perm,
                    &pf_sdpa_out_perm,
                    &sdpa_params,
                    1,
                ).map_err(|e| anyhow::anyhow!("batched dense SDPA L{layer_idx}: {e}"))?;

                // 7. Permute sdpa_out [n_heads, seq_len, hd] → [seq_len, n_heads, hd]
                s.barrier_between(
                    &[&pf_sdpa_out_perm],
                    &[&pf_sdpa_out],
                );
                mlx_native::ops::transpose::permute_021_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_sdpa_out_perm, &pf_sdpa_out,
                    nh, seq_len, hd,
                ).map_err(|e| anyhow::anyhow!("batched permute SDPA L{layer_idx}: {e}"))?;

                // 8. O-proj (m = seq_len): [seq_len, nh*hd] → [seq_len, hs]
                s.barrier_between(
                    &[&pf_sdpa_out, &self.layers[layer_idx].attn.o_proj.buffer],
                    &[&pf_attn_out],
                );
                dispatch_qmatmul(&mut s, reg, dev, &pf_sdpa_out,
                    &self.layers[layer_idx].attn.o_proj,
                    &mut pf_attn_out, seq_len as u32)?;

                // 9. Post-attn fused norm + residual add (rows = seq_len)
                //    residual = (pre-attn hidden) + norm(attn_out, post_attn_norm)
                s.barrier_between(
                    &[&pf_hidden, &pf_attn_out],
                    &[&pf_residual],
                );
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_hidden,
                    &pf_attn_out,
                    &self.layers[layer_idx].norms.post_attention_layernorm,
                    &pf_residual,
                    hs as u32, seq_len as u32, eps,
                ).map_err(|e| anyhow::anyhow!("batched post-attn L{layer_idx}: {e}"))?;

                s.finish()
                    .map_err(|e| anyhow::anyhow!("batched attn finish L{layer_idx}: {e}"))?;
            }

            // ================================================================
            // SESSION B: batched MLP + MoE
            // ================================================================
            {
                let mut s = exec.begin()
                    .map_err(|e| anyhow::anyhow!("batched mlp session L{layer_idx}: {e}"))?;

                // Pre-FF norm (for MLP), pre-FF norm 2 (for MoE input), router norm
                s.barrier_between(
                    &[&pf_residual],
                    &[&pf_norm_out, &pf_moe_norm_out, &pf_router_norm_out],
                );
                s.rms_norm(reg, metal_dev,
                    &pf_residual,
                    &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                    &pf_norm_out,
                    &self.activations.norm_params,
                    seq_len as u32, hs as u32,
                ).map_err(|e| anyhow::anyhow!("batched pre-FF norm L{layer_idx}: {e}"))?;
                s.rms_norm(reg, metal_dev,
                    &pf_residual,
                    &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                    &pf_moe_norm_out,
                    &self.activations.norm_params,
                    seq_len as u32, hs as u32,
                ).map_err(|e| anyhow::anyhow!("batched pre-FF norm2 L{layer_idx}: {e}"))?;
                s.rms_norm(reg, metal_dev,
                    &pf_residual,
                    &self.layers[layer_idx].moe.router_combined_weight,
                    &pf_router_norm_out,
                    &self.activations.norm_params,
                    seq_len as u32, hs as u32,
                ).map_err(|e| anyhow::anyhow!("batched router norm L{layer_idx}: {e}"))?;

                // Dense MLP gate / up (m = seq_len); router proj (m = seq_len)
                s.barrier_between(
                    &[&pf_norm_out, &pf_router_norm_out],
                    &[&pf_mlp_gate, &pf_mlp_up, &pf_router_logits],
                );
                dispatch_qmatmul(&mut s, reg, dev, &pf_norm_out,
                    &self.layers[layer_idx].mlp.gate_proj,
                    &mut pf_mlp_gate, seq_len as u32)?;
                dispatch_qmatmul(&mut s, reg, dev, &pf_norm_out,
                    &self.layers[layer_idx].mlp.up_proj,
                    &mut pf_mlp_up, seq_len as u32)?;
                dispatch_qmatmul(&mut s, reg, dev, &pf_router_norm_out,
                    &self.layers[layer_idx].moe.router_proj,
                    &mut pf_router_logits, seq_len as u32)?;

                // Fused GELU(gate) * up over [seq_len, intermediate]
                // + batched MoE routing over [seq_len, num_experts]
                s.barrier_between(
                    &[&pf_mlp_gate, &pf_mlp_up, &pf_router_logits],
                    &[&pf_mlp_fused, &pf_expert_ids, &pf_routing_weights],
                );
                {
                    use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};
                    // fused_gelu_mul: operates on flat buffers, n_elements = seq_len * intermediate
                    let n_elements_bytes = ((seq_len * intermediate) as u32).to_ne_bytes();
                    let pipeline = reg.get_pipeline("fused_gelu_mul", metal_dev)?;
                    encode_with_args(
                        s.encoder_mut(), pipeline,
                        &[
                            (0, KernelArg::Buffer(&pf_mlp_gate)),
                            (1, KernelArg::Buffer(&pf_mlp_up)),
                            (2, KernelArg::Buffer(&pf_mlp_fused)),
                            (3, KernelArg::Bytes(&n_elements_bytes)),
                        ],
                        mlx_native::MTLSize::new((seq_len * intermediate) as u64, 1, 1),
                        mlx_native::MTLSize::new(
                            std::cmp::min(256, (seq_len * intermediate) as u64), 1, 1),
                    );
                }
                mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_batch_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_router_logits,
                    &pf_expert_ids,
                    &pf_routing_weights,
                    &self.layers[layer_idx].moe.per_expert_scale,
                    num_experts as u32, top_k as u32, seq_len as u32,
                ).map_err(|e| anyhow::anyhow!("batched MoE routing L{layer_idx}: {e}"))?;

                // Dense MLP down
                s.barrier_between(
                    &[&pf_mlp_fused, &self.layers[layer_idx].mlp.down_proj.buffer],
                    &[&pf_mlp_down],
                );
                dispatch_qmatmul(&mut s, reg, dev, &pf_mlp_fused,
                    &self.layers[layer_idx].mlp.down_proj,
                    &mut pf_mlp_down, seq_len as u32)?;

                // MoE gate_up experts: quantized_matmul_id_ggml with n_tokens = seq_len
                if self.layers[layer_idx].moe.stacked_gate_up.is_none()
                    || self.layers[layer_idx].moe.stacked_down.is_none()
                {
                    anyhow::bail!("batched prefill requires fused MoE _id path at L{layer_idx}");
                }
                let ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                s.barrier_between(
                    &[&pf_moe_norm_out, &pf_expert_ids,
                      self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap()],
                    &[&pf_moe_gate_up],
                );
                s.quantized_matmul_id_ggml(
                    reg, dev,
                    &pf_moe_norm_out,
                    self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap(),
                    &pf_expert_ids,
                    &mut pf_moe_gate_up,
                    &mlx_native::GgmlQuantizedMatmulIdParams {
                        n_tokens: seq_len as u32,
                        top_k: top_k as u32,
                        n: (2 * moe_int) as u32,
                        k: hs as u32,
                        n_experts: num_experts as u32,
                        expert_stride: self.layers[layer_idx].moe.gate_up_expert_stride,
                        ggml_type: ggml_type_gu,
                    },
                ).map_err(|e| anyhow::anyhow!("batched gate_up_id L{layer_idx}: {e}"))?;

                // Batched SwiGLU over [seq_len, top_k, 2*moe_int] → [seq_len, top_k, moe_int]
                s.barrier_between(
                    &[&pf_moe_gate_up],
                    &[&pf_moe_swiglu],
                );
                mlx_native::ops::moe_dispatch::moe_swiglu_seq_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_moe_gate_up,
                    &pf_moe_swiglu,
                    moe_int, top_k, seq_len,
                ).map_err(|e| anyhow::anyhow!("batched MoE swiglu L{layer_idx}: {e}"))?;

                // MoE down experts: quantized_matmul_id_ggml with n_tokens = seq_len*top_k, top_k=1
                let ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;
                s.barrier_between(
                    &[&pf_moe_swiglu, &pf_expert_ids,
                      self.layers[layer_idx].moe.stacked_down.as_ref().unwrap()],
                    &[&pf_moe_down],
                );
                s.quantized_matmul_id_ggml(
                    reg, dev,
                    &pf_moe_swiglu,
                    self.layers[layer_idx].moe.stacked_down.as_ref().unwrap(),
                    &pf_expert_ids,
                    &mut pf_moe_down,
                    &mlx_native::GgmlQuantizedMatmulIdParams {
                        n_tokens: (seq_len * top_k) as u32,
                        top_k: 1,
                        n: hs as u32,
                        k: moe_int as u32,
                        n_experts: num_experts as u32,
                        expert_stride: self.layers[layer_idx].moe.down_expert_stride,
                        ggml_type: ggml_type_dn,
                    },
                ).map_err(|e| anyhow::anyhow!("batched down_id L{layer_idx}: {e}"))?;

                // post-FF norm 1 on mlp_down: [seq_len, hs]
                s.barrier_between(
                    &[&pf_mlp_down],
                    &[&pf_mlp_down_out],
                );
                s.rms_norm(reg, metal_dev,
                    &pf_mlp_down,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                    &pf_mlp_down_out,
                    &self.activations.norm_params,
                    seq_len as u32, hs as u32,
                ).map_err(|e| anyhow::anyhow!("batched post-FF norm1 L{layer_idx}: {e}"))?;

                // Batched MoE weighted sum: [seq_len, top_k, hs] with weights [seq_len, top_k]
                //   → [seq_len, hs]
                s.barrier_between(
                    &[&pf_moe_down, &pf_routing_weights],
                    &[&pf_moe_accum],
                );
                mlx_native::ops::moe_dispatch::moe_weighted_sum_seq_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_moe_down,
                    &pf_routing_weights,
                    &pf_moe_accum,
                    hs, top_k, seq_len,
                ).map_err(|e| anyhow::anyhow!("batched MoE weighted_sum L{layer_idx}: {e}"))?;

                // post-FF norm 2 + combine MLP + MoE: output = mlp_down_out + norm(moe_accum)
                s.barrier_between(
                    &[&pf_mlp_down_out, &pf_moe_accum],
                    &[&pf_mlp_down],
                );
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_mlp_down_out,
                    &pf_moe_accum,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                    &pf_mlp_down,
                    hs as u32, seq_len as u32, eps,
                ).map_err(|e| anyhow::anyhow!("batched post-FF norm2+combine L{layer_idx}: {e}"))?;

                // End-of-layer: output = (residual + norm(mlp_down, post_feedforward_layernorm)) * scalar
                let scalar_is_vector = self.layers[layer_idx].layer_scalar.element_count() > 1;
                s.barrier_between(
                    &[&pf_residual, &pf_mlp_down],
                    &[&pf_hidden],
                );
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_scalar_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_residual,
                    &pf_mlp_down,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm,
                    &pf_hidden,
                    &self.layers[layer_idx].layer_scalar,
                    seq_len as u32, hs as u32, eps,
                    scalar_is_vector,
                ).map_err(|e| anyhow::anyhow!("batched end-layer L{layer_idx}: {e}"))?;

                s.finish()
                    .map_err(|e| anyhow::anyhow!("batched mlp finish L{layer_idx}: {e}"))?;
            }

            // ================================================================
            // SESSION C: write K,V to dense cache at positions [0, seq_len)
            //            Layout conversion: [seq_len, nkv, hd] → [nkv, cap, hd]
            // ================================================================
            {
                let mut s = exec.begin()
                    .map_err(|e| anyhow::anyhow!("batched cache write L{layer_idx}: {e}"))?;
                // Ring mode needs wrap support in the seq kernel — not
                // implemented yet. Guard: batched prefill requires seq_len
                // <= per-layer capacity.
                let layer_cap = dense_kvs_vec[layer_idx].capacity;
                if seq_len > layer_cap {
                    anyhow::bail!(
                        "batched prefill L{}: seq_len={} exceeds dense cap={} (sliding layer). \
                         Ring-wrap in the seq kernel is a follow-up; use HF2Q_BATCHED_PREFILL=0 \
                         (default) for now.",
                        layer_idx, seq_len, layer_cap);
                }
                s.barrier_between(
                    &[&pf_k_normed, &pf_v_normed],
                    &[&dense_kvs_vec[layer_idx].k, &dense_kvs_vec[layer_idx].v],
                );
                if use_f16_kv {
                    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_k_normed,
                        &dense_kvs_vec[layer_idx].k,
                        nkv as u32, hd as u32,
                        layer_cap as u32, 0, seq_len as u32,
                    ).map_err(|e| anyhow::anyhow!("batched K cache copy L{layer_idx}: {e}"))?;
                    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_v_normed,
                        &dense_kvs_vec[layer_idx].v,
                        nkv as u32, hd as u32,
                        layer_cap as u32, 0, seq_len as u32,
                    ).map_err(|e| anyhow::anyhow!("batched V cache copy L{layer_idx}: {e}"))?;
                } else {
                    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_k_normed,
                        &dense_kvs_vec[layer_idx].k,
                        nkv as u32, hd as u32,
                        layer_cap as u32, 0, seq_len as u32,
                    ).map_err(|e| anyhow::anyhow!("batched K cache copy L{layer_idx}: {e}"))?;
                    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_v_normed,
                        &dense_kvs_vec[layer_idx].v,
                        nkv as u32, hd as u32,
                        layer_cap as u32, 0, seq_len as u32,
                    ).map_err(|e| anyhow::anyhow!("batched V cache copy L{layer_idx}: {e}"))?;
                }
                s.finish()
                    .map_err(|e| anyhow::anyhow!("batched cache finish L{layer_idx}: {e}"))?;

                // Update KV cache metadata for subsequent decode
                self.kv_caches[layer_idx].write_pos = seq_len;
                self.kv_caches[layer_idx].seq_len = seq_len.min(self.kv_caches[layer_idx].capacity);
            }

            // ADR-010 batched sub-stage dump at (layer_idx, target_tok)
            if let Some((dump_layer, target_tok)) = batched_dump {
                if dump_layer == layer_idx && target_tok < seq_len && !use_f16_kv {
                    // Row slices from [seq_len, *, hd] row-major buffers
                    // pf_norm_out is dumped inline after the pre-attn RMS norm
                    // (see session A); the buffer gets overwritten in session B.
                    let qpre_full: &[f32] = pf_q.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_q: {e}"))?;
                    let kpre_full: &[f32] = pf_k.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_k: {e}"))?;
                    let vpre_full: &[f32] = pf_v.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_v: {e}"))?;
                    let q_full: &[f32] = pf_q_normed.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_q_normed: {e}"))?;
                    let k_full: &[f32] = pf_k_normed.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_k_normed: {e}"))?;
                    let v_full: &[f32] = pf_v_normed.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_v_normed: {e}"))?;
                    let sdpa_full: &[f32] = pf_sdpa_out.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_sdpa_out: {e}"))?;

                    let q_off = target_tok * nh * hd;
                    let k_off = target_tok * nkv * hd;
                    let v_off = target_tok * nkv * hd;
                    let s_off = target_tok * nh * hd;

                    let qpre_row = &qpre_full[q_off..q_off + nh * hd];
                    let kpre_row = &kpre_full[k_off..k_off + nkv * hd];
                    let vpre_row = &vpre_full[v_off..v_off + nkv * hd];
                    let q_row = &q_full[q_off..q_off + nh * hd];
                    let k_row = &k_full[k_off..k_off + nkv * hd];
                    let v_row = &v_full[v_off..v_off + nkv * hd];
                    let sdpa_row = &sdpa_full[s_off..s_off + nh * hd];

                    // Cache slice positions 0..=target_tok in [nkv, tok+1, hd] logical layout
                    let cap = dense_kvs_vec[layer_idx].capacity;
                    let n_valid = target_tok + 1;
                    let k_cache: &[f32] = dense_kvs_vec[layer_idx].k.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump dense K L{layer_idx}: {e}"))?;
                    let v_cache: &[f32] = dense_kvs_vec[layer_idx].v.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump dense V L{layer_idx}: {e}"))?;
                    let mut k_valid = Vec::<f32>::with_capacity(nkv * n_valid * hd);
                    let mut v_valid = Vec::<f32>::with_capacity(nkv * n_valid * hd);
                    for h in 0..nkv {
                        for p in 0..n_valid {
                            let off = h * cap * hd + p * hd;
                            k_valid.extend_from_slice(&k_cache[off..off + hd]);
                            v_valid.extend_from_slice(&v_cache[off..off + hd]);
                        }
                    }

                    let write_slice = |name: &str, data: &[f32], tag_shape: &str| -> anyhow::Result<()> {
                        let path = format!(
                            "{batched_dump_dir}/hf2q_batched_{name}_layer{layer_idx:02}_tok{target_tok:03}.bin");
                        let bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                        };
                        std::fs::write(&path, bytes)
                            .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
                        eprintln!("[BATCHED DUMP] {} {} f32 -> {}", name, tag_shape, path);
                        Ok(())
                    };
                    write_slice("q_pre_normed_row", qpre_row, &format!("[{nh},{hd}]"))?;
                    write_slice("k_pre_normed_row", kpre_row, &format!("[{nkv},{hd}]"))?;
                    write_slice("v_pre_normed_row", vpre_row, &format!("[{nkv},{hd}]"))?;
                    write_slice("q_normed_row", q_row, &format!("[{nh},{hd}]"))?;
                    write_slice("k_normed_row", k_row, &format!("[{nkv},{hd}]"))?;
                    write_slice("v_normed_row", v_row, &format!("[{nkv},{hd}]"))?;
                    write_slice("sdpa_out_row", sdpa_row, &format!("[{nh},{hd}]"))?;
                    write_slice("k_cache_upto", &k_valid, &format!("[{nkv},{n_valid},{hd}]"))?;
                    write_slice("v_cache_upto", &v_valid, &format!("[{nkv},{n_valid},{hd}]"))?;

                    // ADR-010 L6 post-attention bisection: dump the rest of
                    // the post-SDPA pipeline for this token. All target
                    // buffers are distinct per-role and not reused within
                    // a layer, so end-of-layer reads are safe.
                    let attn_out_full: &[f32] = pf_attn_out.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_attn_out: {e}"))?;
                    let residual_full: &[f32] = pf_residual.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_residual: {e}"))?;
                    let rlogits_full: &[f32] = pf_router_logits.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_router_logits: {e}"))?;
                    let rweights_full: &[f32] = pf_routing_weights.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_routing_weights: {e}"))?;
                    let eids_full: &[u32] = pf_expert_ids.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_expert_ids: {e}"))?;
                    let mlp_down_full: &[f32] = pf_mlp_down.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_mlp_down: {e}"))?;
                    let moe_accum_full: &[f32] = pf_moe_accum.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_moe_accum: {e}"))?;
                    let hidden_full: &[f32] = pf_hidden.as_slice()
                        .map_err(|e| anyhow::anyhow!("dump pf_hidden end: {e}"))?;

                    let hs_off = target_tok * hs;
                    let exp_off = target_tok * top_k;
                    let rl_off = target_tok * num_experts;

                    write_slice("attn_out_row",
                        &attn_out_full[hs_off..hs_off + hs], &format!("[{hs}]"))?;
                    write_slice("residual_row",
                        &residual_full[hs_off..hs_off + hs], &format!("[{hs}]"))?;
                    write_slice("router_logits_row",
                        &rlogits_full[rl_off..rl_off + num_experts], &format!("[{num_experts}]"))?;
                    write_slice("routing_weights_row",
                        &rweights_full[exp_off..exp_off + top_k], &format!("[{top_k}]"))?;
                    write_slice("mlp_down_row",
                        &mlp_down_full[hs_off..hs_off + hs], &format!("[{hs}]"))?;
                    write_slice("moe_accum_row",
                        &moe_accum_full[hs_off..hs_off + hs], &format!("[{hs}]"))?;
                    write_slice("l_out_row",
                        &hidden_full[hs_off..hs_off + hs], &format!("[{hs}]"))?;

                    // u32 expert IDs — separate byte format
                    let eid_slice = &eids_full[exp_off..exp_off + top_k];
                    let path_eid = format!(
                        "{batched_dump_dir}/hf2q_batched_expert_ids_row_layer{layer_idx:02}_tok{target_tok:03}.bin");
                    let eid_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(eid_slice.as_ptr() as *const u8,
                                                    eid_slice.len() * 4) };
                    std::fs::write(&path_eid, eid_bytes)
                        .map_err(|e| anyhow::anyhow!("write {path_eid}: {e}"))?;
                    eprintln!("[BATCHED DUMP] expert_ids_row [{top_k}] u32 -> {path_eid}");
                }
            }
        }

        // -------------------------------------------------------------------
        // FINAL: last-row → final_norm → lm_head → softcap → argmax
        // -------------------------------------------------------------------
        let first_token: u32;
        {
            let mut s = exec.begin()
                .map_err(|e| anyhow::anyhow!("batched head session: {e}"))?;

            // Copy last row of pf_hidden ([seq_len, hs]) into activations.hidden ([hs])
            s.barrier_between(
                &[&pf_hidden],
                &[&self.activations.hidden],
            );
            mlx_native::ops::copy::dispatch_copy_f32(
                s.encoder_mut(), reg, metal_dev,
                &pf_hidden,
                &self.activations.hidden,
                (seq_len - 1) * hs,
                0,
                hs,
            ).map_err(|e| anyhow::anyhow!("batched last-row copy: {e}"))?;

            // Final norm
            s.barrier_between(
                &[&self.activations.hidden, &self.final_norm],
                &[&self.activations.norm_out],
            );
            s.rms_norm(reg, metal_dev,
                &self.activations.hidden,
                &self.final_norm,
                &self.activations.norm_out,
                &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("batched final norm: {e}"))?;

            // lm_head: whichever weight was loaded (Q8 for large vocab×hs, F16 otherwise).
            if let Some(ref q8) = self.lm_head_q8 {
                s.barrier_between(
                    &[&self.activations.norm_out, &q8.buffer],
                    &[&self.activations.logits],
                );
                super::forward_mlx::dispatch_qmatmul(
                    &mut s, reg, dev,
                    &self.activations.norm_out,
                    q8,
                    &mut self.activations.logits,
                    1,
                ).map_err(|e| anyhow::anyhow!("batched lm_head Q8: {e}"))?;
            } else if let Some(ref lm_head_f16) = self.lm_head_f16 {
                s.barrier_between(
                    &[&self.activations.norm_out, lm_head_f16],
                    &[&self.activations.logits],
                );
                mlx_native::ops::dense_gemm::dispatch_dense_matvec_f16w_f32io(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.norm_out,
                    lm_head_f16,
                    &self.activations.logits,
                    &DenseGemmF16Params { m: 1, n: vocab_size as u32, k: hs as u32 },
                ).map_err(|e| anyhow::anyhow!("batched lm_head: {e}"))?;
            } else {
                anyhow::bail!("batched prefill requires GPU lm_head (F16 or Q8 weight)");
            }

            if let Some(cap) = self.final_logit_softcapping {
                s.barrier_between(
                    &[&self.activations.logits],
                    &[&self.activations.logits],
                );
                mlx_native::ops::softcap::dispatch_softcap(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.logits,
                    &self.activations.logits,
                    &self.activations.softcap_params,
                    cap,
                ).map_err(|e| anyhow::anyhow!("batched softcap: {e}"))?;
            }

            s.barrier_between(
                &[&self.activations.logits],
                &[&self.activations.argmax_index, &self.activations.argmax_value],
            );
            mlx_native::ops::argmax::dispatch_argmax_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.logits,
                &self.activations.argmax_index,
                &self.activations.argmax_value,
                &self.activations.argmax_params,
                vocab_size as u32,
            ).map_err(|e| anyhow::anyhow!("batched argmax: {e}"))?;

            s.finish()
                .map_err(|e| anyhow::anyhow!("batched head finish: {e}"))?;

            first_token = {
                let idx: &[u32] = self.activations.argmax_index.as_slice()
                    .map_err(|e| anyhow::anyhow!("argmax read: {e}"))?;
                idx[0]
            };
        }

        let elapsed = prefill_start.elapsed();
        eprintln!(
            "Batched prefill complete: {} tokens in {:.1} ms ({:.1} tok/s), first decode token = {}",
            seq_len,
            elapsed.as_secs_f64() * 1000.0,
            seq_len as f64 / elapsed.as_secs_f64(),
            first_token,
        );

        // Store dense KV buffers so forward_decode can use them
        self.dense_kvs = Some(dense_kvs_vec);
        self.dense_sdpa_tmp = Some(sdpa_tmp);

        Ok(first_token)
    }
}
