//! Dense prefill forward pass — ADR-009 Track 1.
//!
//! This module contains `forward_prefill()`, which processes the entire prompt
//! through the transformer layers using dense F32 attention instead of
//! TQ-packed attention. The rest of the layer pipeline (norms, QKV, MLP, MoE)
//! reuses the same ops as `forward_decode`.
//!
//! Architecture:
//! - Tokens are processed one at a time through all layers (same as decode)
//! - For each token, Q/K/V are computed identically to decode
//! - K,V are accumulated as dense F32 in head-major layout per layer
//! - Attention uses `flash_attn_vec` (dense F32 SDPA) instead of `flash_attn_vec_tq`
//! - K,V are also TQ-encoded into the packed cache for subsequent decode
//! - After all tokens: extract last-row logits, argmax → first decode token

use anyhow::Result;
use mlx_native::MlxBuffer;
use mlx_native::ops::flash_attn_vec::FlashAttnVecParams;
use mlx_native::ops::dense_gemm::DenseGemmF16Params;
use std::time::Instant;

use super::forward_mlx::{MlxModelWeights, DenseKvBuffers, dispatch_qmatmul, dispatch_rms_norm_unit_perhead};
use super::config::LayerType;
use super::gpu::GpuContext;

/// Helper: dump an F32 MlxBuffer's first `n_elems` to a file at dump_dir.
fn write_dump_f32(
    dump_dir: &str,
    name: &str,
    layer: usize,
    tok: usize,
    buf: &MlxBuffer,
    n_elems: usize,
) -> Result<()> {
    let data: &[f32] = buf.as_slice()
        .map_err(|e| anyhow::anyhow!("dump {name} L{layer} T{tok}: {e}"))?;
    let path = format!("{dump_dir}/hf2q_prefill_{name}_layer{layer:02}_tok{tok:03}.bin");
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8,
            n_elems * std::mem::size_of::<f32>())
    };
    std::fs::write(&path, bytes)
        .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
    eprintln!("[PREFILL DUMP] {} L{} T{} ({} f32) -> {}", name, layer, tok, n_elems, path);
    Ok(())
}

impl MlxModelWeights {
    /// True batched prefill with dense attention (ADR-009 Track 1).
    ///
    /// Processes the prompt token-by-token through all layers but replaces
    /// TQ-packed attention with dense F32 SDPA. This eliminates compounding
    /// TQ quantization noise during prompt ingestion.
    ///
    /// Returns the first decode token (greedy argmax of last-row logits).
    pub fn forward_prefill(
        &mut self,
        prompt_tokens: &[u32],
        max_decode_tokens: usize,
        gpu: &mut GpuContext,
    ) -> Result<u32> {
        let seq_len = prompt_tokens.len();
        if seq_len == 0 {
            anyhow::bail!("forward_prefill: empty prompt");
        }
        let hs = self.hidden_size;
        let num_layers = self.layers.len();
        let vocab_size = self.vocab_size;
        let eps = self.rms_norm_eps;

        let f32_sz = std::mem::size_of::<f32>();

        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let metal_dev = dev.metal_device();

        // ===================================================================
        // Allocate per-layer dense K,V buffers in head-major layout:
        //   [n_kv_heads, capacity, head_dim]
        // This layout matches flash_attn_vec's K,V input format.
        //
        // Capacity = seq_len + max_decode_tokens. dense flash_attn_vec
        // requires a linear (non-ring-buffer) cache, so capacity must
        // cover the full generation budget. A ring-buffer path for
        // sliding layers is deferred (ADR-010).
        // ===================================================================
        // ADR-009 Phase 3A finding: matching llama.cpp's F16 KV cache
        // REGRESSED our parity (sourdough 3656→3095, sliding_wrap 752→627).
        // llama.cpp itself is insensitive to KV dtype (its F16 and F32 outputs
        // are byte-identical). Our F16 path has a separate bug worse than F32.
        // F32 remains the default; F16 is opt-in via HF2Q_F16_KV=1 for the
        // follow-up investigation into the F16-specific regression.
        let use_f16_kv = std::env::var("HF2Q_F16_KV").map_or(false, |v| v == "1");
        let kv_dtype = if use_f16_kv { mlx_native::DType::F16 } else { mlx_native::DType::F32 };
        let kv_elem_bytes = if use_f16_kv { 2 } else { 4 };
        eprintln!("Prefill: KV cache dtype = {:?}", kv_dtype);

        // Per-layer capacity:
        //   - Sliding (ring): sliding_window. Writes wrap at seq_pos % capacity.
        //     Attention is permutation-invariant over cached K,V, so slot
        //     order doesn't affect correctness. Dense flash_attn_vec reads
        //     the populated slots with a pure causal mask.
        //   - Global (linear): seq_len + max_decode_tokens. Writes are monotonic.
        // Ring buffer for sliding drops ~5 GB of dense KV at 20k decode on
        // Gemma-4 26B (8×1024×256 per layer vs 8×20022×256).
        let linear_capacity = seq_len + max_decode_tokens;
        let sw = self.sliding_window;
        let mut dense_kvs_vec: Vec<DenseKvBuffers> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let nkv = self.num_kv_heads[layer_idx];
            let hd = self.head_dims[layer_idx];
            let layer_is_ring = self.layer_types[layer_idx] == LayerType::Sliding;
            let capacity = if layer_is_ring { sw } else { linear_capacity };
            let n = nkv * capacity * hd;
            let k = dev.alloc_buffer(n * kv_elem_bytes, kv_dtype, vec![nkv, capacity, hd])
                .map_err(|e| anyhow::anyhow!("prefill dense K L{layer_idx}: {e}"))?;
            let v = dev.alloc_buffer(n * kv_elem_bytes, kv_dtype, vec![nkv, capacity, hd])
                .map_err(|e| anyhow::anyhow!("prefill dense V L{layer_idx}: {e}"))?;
            dense_kvs_vec.push(DenseKvBuffers { k, v, capacity, is_sliding: layer_is_ring });
        }
        let _ = f32_sz; // kept for clarity; unused after dtype switch

        // Tmp buffer for flash_attn_vec (sized for largest layer config)
        let max_nh = self.num_attention_heads;
        let max_hd = self.head_dims.iter().copied().max().unwrap_or(512);
        let tmp_bytes = mlx_native::ops::flash_attn_vec::tmp_buffer_bytes(
            max_nh as u32, max_hd as u32);
        let sdpa_tmp = dev.alloc_buffer(tmp_bytes, mlx_native::DType::F32,
            vec![tmp_bytes / 4])
            .map_err(|e| anyhow::anyhow!("prefill sdpa_tmp: {e}"))?;

        eprintln!("Prefill: {} tokens × {} layers (dense SDPA)", seq_len, num_layers);

        // ADR-009 Phase 3A: prefill boundary dumps at (target_layer, target_tok).
        // Controlled by HF2Q_PREFILL_DUMP="layer,tok" e.g. "7,34".
        let prefill_dump: Option<(usize, usize)> = std::env::var("HF2Q_PREFILL_DUMP")
            .ok()
            .and_then(|v| {
                let parts: Vec<&str> = v.split(',').collect();
                if parts.len() == 2 {
                    Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
                } else {
                    None
                }
            });
        let dump_dir = std::env::var("HF2Q_DUMP_DIR").unwrap_or_else(|_| "/tmp".into());

        // ===================================================================
        // Process each prompt token through all layers
        // ===================================================================
        let prefill_start = Instant::now();
        let mut last_token = 0u32;

        for (tok_i, &tok) in prompt_tokens.iter().enumerate() {
            let seq_pos = tok_i;

            // Write position buffer
            {
                let pos_dst: &mut [u32] = self.activations.position.as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("position write: {e}"))?;
                pos_dst[0] = seq_pos as u32;
            }

            // KV cache bookkeeping (same as decode: advance write_pos, seq_len)
            let mut kv_info: Vec<(bool, usize, usize, usize)> = Vec::with_capacity(num_layers);
            for layer_idx in 0..num_layers {
                let is_sliding = self.kv_caches[layer_idx].is_sliding;
                let write_pos = self.kv_caches[layer_idx].write_pos;
                let capacity = self.kv_caches[layer_idx].capacity;
                self.kv_caches[layer_idx].write_pos += 1;
                self.kv_caches[layer_idx].seq_len = self.kv_caches[layer_idx].seq_len
                    .saturating_add(1).min(capacity);
                let kv_seq_len = self.kv_caches[layer_idx].seq_len;
                kv_info.push((is_sliding, write_pos, capacity, kv_seq_len));
            }

            // ===============================================================
            // Single GPU session per token (same structure as forward_decode)
            // ===============================================================
            {
                let mut s = exec.begin()
                    .map_err(|e| anyhow::anyhow!("prefill session T{tok_i}: {e}"))?;

                // --- 1. Embedding ---
                mlx_native::ops::elementwise::embedding_gather_scale_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.embed_weight,
                    &self.activations.hidden,
                    tok, hs,
                    (hs as f32).sqrt(),
                ).map_err(|e| anyhow::anyhow!("prefill embed T{tok_i}: {e}"))?;
                s.track_dispatch(&[&self.embed_weight], &[&self.activations.hidden]);

                // --- 2. Transformer layers ---
                for layer_idx in 0..num_layers {
                    let hd = self.head_dims[layer_idx];
                    let nkv = self.num_kv_heads[layer_idx];
                    let nh = self.num_attention_heads;
                    let is_sliding = self.layer_types[layer_idx] == LayerType::Sliding;
                    let (kv_is_sliding, kv_write_pos, kv_capacity, _kv_seq_len) = kv_info[layer_idx];

                    // Active dump flag for this iteration
                    let dump_here = prefill_dump == Some((layer_idx, tok_i));
                    // Dump at layer-start: hidden = L(layer_idx-1) l_out (or embed for L0)
                    if dump_here {
                        s.finish()
                            .map_err(|e| anyhow::anyhow!("prefill dump L{layer_idx} T{tok_i} start finish: {e}"))?;
                        write_dump_f32(&dump_dir, "pre_layer_hidden", layer_idx, tok_i,
                                        &self.activations.hidden, hs)?;
                        s = exec.begin()
                            .map_err(|e| anyhow::anyhow!("prefill dump restart: {e}"))?;
                    }

                    // -- Pre-attention norm --
                    s.barrier_between(
                        &[&self.activations.hidden, &self.layers[layer_idx].norms.input_layernorm],
                        &[&self.activations.norm_out],
                    );
                    s.rms_norm(
                        reg, metal_dev,
                        &self.activations.hidden,
                        &self.layers[layer_idx].norms.input_layernorm,
                        &self.activations.norm_out,
                        &self.activations.norm_params,
                        1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill norm L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(&dump_dir, "post_input_norm", layer_idx, tok_i,
                                        &self.activations.norm_out, hs)?;
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- QKV projections (concurrent) --
                    s.barrier_between(
                        &[&self.activations.norm_out],
                        &[&self.activations.attn_q, &self.activations.attn_k, &self.activations.attn_v],
                    );
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].attn.q_proj, &mut self.activations.attn_q, 1)?;
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].attn.k_proj, &mut self.activations.attn_k, 1)?;
                    let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
                    if !v_is_k {
                        dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                            self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                            &mut self.activations.attn_v, 1)?;
                    }

                    // -- Fused per-head RMS norm + RoPE on Q and K --
                    let ff_gpu = if is_sliding {
                        None
                    } else {
                        Some(&self.activations.rope_freq_factors_gpu)
                    };
                    let theta = if is_sliding {
                        self.rope_theta_sliding
                    } else {
                        self.rope_theta_global
                    };
                    let half_rope = (hd / 2) as u32;

                    s.barrier_between(
                        &[&self.activations.attn_q, &self.activations.attn_k],
                        &[&self.activations.attn_q_normed, &self.activations.attn_k_normed],
                    );
                    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_q,
                        &self.activations.attn_q_normed,
                        Some(&self.layers[layer_idx].attn.q_norm_weight),
                        &self.activations.position,
                        ff_gpu,
                        nh as u32, hd as u32, half_rope,
                        eps, theta,
                    ).map_err(|e| anyhow::anyhow!("prefill Q norm+RoPE L{layer_idx} T{tok_i}: {e}"))?;
                    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_k,
                        &self.activations.attn_k_normed,
                        Some(&self.layers[layer_idx].attn.k_norm_weight),
                        &self.activations.position,
                        ff_gpu,
                        nkv as u32, hd as u32, half_rope,
                        eps, theta,
                    ).map_err(|e| anyhow::anyhow!("prefill K norm+RoPE L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(&dump_dir, "q_pre_normed", layer_idx, tok_i,
                                        &self.activations.attn_q, nh * hd)?;
                        write_dump_f32(&dump_dir, "k_pre_normed", layer_idx, tok_i,
                                        &self.activations.attn_k, nkv * hd)?;
                        write_dump_f32(&dump_dir, "q_normed", layer_idx, tok_i,
                                        &self.activations.attn_q_normed, nh * hd)?;
                        write_dump_f32(&dump_dir, "k_normed", layer_idx, tok_i,
                                        &self.activations.attn_k_normed, nkv * hd)?;
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- V norm --
                    let hd_norm_params = if is_sliding {
                        &self.activations.norm_params_sliding_hd
                    } else {
                        &self.activations.norm_params_global_hd
                    };
                    if v_is_k {
                        s.barrier_between(
                            &[&self.activations.attn_k],
                            &[&self.activations.attn_v],
                        );
                        dispatch_rms_norm_unit_perhead(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k,
                            &self.activations.attn_v,
                            hd_norm_params,
                            nkv as u32, hd as u32,
                        )?;
                    } else {
                        s.barrier_between(
                            &[&self.activations.attn_v],
                            &[&self.activations.moe_expert_out],
                        );
                        dispatch_rms_norm_unit_perhead(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_v,
                            &self.activations.moe_expert_out,
                            hd_norm_params,
                            nkv as u32, hd as u32,
                        )?;
                    }

                    let v_src = if v_is_k {
                        &self.activations.attn_v
                    } else {
                        &self.activations.moe_expert_out
                    };

                    // ====================================================
                    // DENSE K,V ACCUMULATION (ADR-009 Track 1 key change)
                    //
                    // Copy this position's K,V into head-major dense buffers:
                    //   dense_k[head, pos, :] = attn_k_normed[head, :]
                    //   dense_v[head, pos, :] = v_src[head, :]
                    //
                    // Layout: [nkv, seq_len, hd], writing at pos = tok_i
                    // ====================================================
                    // Per-layer dense cap + ring-wrap write for sliding layers.
                    let layer_dense_cap = dense_kvs_vec[layer_idx].capacity;
                    let layer_is_ring = dense_kvs_vec[layer_idx].is_sliding;
                    let write_slot = if layer_is_ring {
                        (tok_i % layer_dense_cap) as u32
                    } else {
                        tok_i as u32
                    };
                    s.barrier_between(
                        &[&self.activations.attn_k_normed, v_src],
                        &[&dense_kvs_vec[layer_idx].k, &dense_kvs_vec[layer_idx].v],
                    );
                    if use_f16_kv {
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &dense_kvs_vec[layer_idx].k,
                            nkv as u32, hd as u32,
                            layer_dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("prefill F16 K copy L{layer_idx} T{tok_i}: {e}"))?;
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            v_src,
                            &dense_kvs_vec[layer_idx].v,
                            nkv as u32, hd as u32,
                            layer_dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("prefill F16 V copy L{layer_idx} T{tok_i}: {e}"))?;
                    } else {
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &dense_kvs_vec[layer_idx].k,
                            nkv as u32, hd as u32,
                            layer_dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("prefill F32 K batch copy L{layer_idx} T{tok_i}: {e}"))?;
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(), reg, metal_dev,
                            v_src,
                            &dense_kvs_vec[layer_idx].v,
                            nkv as u32, hd as u32,
                            layer_dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("prefill F32 V batch copy L{layer_idx} T{tok_i}: {e}"))?;
                    }

                    // Also TQ-encode into packed cache (for subsequent decode)
                    if !std::env::var("HF2Q_SKIP_TQ_ENCODE").map_or(false, |v| v == "1") {
                        let cache_pos_val = if kv_is_sliding {
                            (kv_write_pos % kv_capacity) as u32
                        } else {
                            kv_write_pos as u32
                        };
                        s.barrier_between(
                            &[&self.activations.attn_k_normed, v_src],
                            &[&self.kv_caches[layer_idx].k_packed, &self.kv_caches[layer_idx].k_norms,
                              &self.kv_caches[layer_idx].v_packed, &self.kv_caches[layer_idx].v_norms],
                        );
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &self.kv_caches[layer_idx].k_packed,
                            &self.kv_caches[layer_idx].k_norms,
                            nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                            kv_is_sliding,
                        ).map_err(|e| anyhow::anyhow!("prefill TQ K L{layer_idx} T{tok_i}: {e}"))?;
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                            s.encoder_mut(), reg, metal_dev,
                            v_src,
                            &self.kv_caches[layer_idx].v_packed,
                            &self.kv_caches[layer_idx].v_norms,
                            nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                            kv_is_sliding,
                        ).map_err(|e| anyhow::anyhow!("prefill TQ V L{layer_idx} T{tok_i}: {e}"))?;
                    }

                    // ====================================================
                    // DENSE SDPA (ADR-009 Track 1 key change)
                    //
                    // Use flash_attn_vec with dense F32 K,V instead of
                    // flash_attn_vec_tq with packed TQ K,V.
                    //
                    // Q: attn_q_normed [nh, 1, hd] (already in head-major)
                    // K: dense_kvs[layer].k [nkv, seq_len, hd]
                    // V: dense_kvs[layer].v [nkv, seq_len, hd]
                    //
                    // No FWHT rotation needed — pure model-space attention.
                    // ====================================================
                    // kv_seq_len: ring clamps to capacity (== sliding_window).
                    // Ring mode uses mask_type=1 (causal only) — the ring
                    // already applies the sliding-window constraint.
                    let dense_kv_seq_len = if layer_is_ring {
                        ((tok_i + 1).min(layer_dense_cap)) as u32
                    } else {
                        (tok_i + 1) as u32
                    };
                    s.barrier_between(
                        &[&self.activations.attn_q_normed,
                          &dense_kvs_vec[layer_idx].k, &dense_kvs_vec[layer_idx].v],
                        &[&self.activations.sdpa_out],
                    );
                    let p = FlashAttnVecParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: dense_kv_seq_len,
                        kv_capacity: layer_dense_cap as u32,
                        scale: 1.0, // Gemma4: scale = 1.0 (llama.cpp oracle)
                        mask_type: 1, // causal; ring applies the sliding window
                        sliding_window: 0,
                        softcap: 0.0,
                    };
                    mlx_native::ops::flash_attn_vec::flash_attn_vec(
                        s.encoder_mut(), reg, dev,
                        &self.activations.attn_q_normed,
                        &dense_kvs_vec[layer_idx].k,
                        &dense_kvs_vec[layer_idx].v,
                        &self.activations.sdpa_out,
                        &sdpa_tmp,
                        &p,
                    ).map_err(|e| anyhow::anyhow!("prefill dense SDPA L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(&dump_dir, "sdpa_out", layer_idx, tok_i,
                                        &self.activations.sdpa_out, nh * hd)?;
                        // ADR-010 sub-stage dump: full dense K,V cache up to
                        // (and including) the target token, packed as
                        // [nkv, tok_i+1, hd] for comparison with llama's
                        // cache_k_l*/cache_v_l* at pos tok_i. Only F32 path.
                        if !use_f16_kv {
                            let cap = dense_kvs_vec[layer_idx].capacity;
                            let n_valid = tok_i + 1;
                            let k_full: &[f32] = dense_kvs_vec[layer_idx].k.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump K cache L{layer_idx}: {e}"))?;
                            let v_full: &[f32] = dense_kvs_vec[layer_idx].v.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump V cache L{layer_idx}: {e}"))?;
                            let mut k_valid = Vec::<f32>::with_capacity(nkv * n_valid * hd);
                            let mut v_valid = Vec::<f32>::with_capacity(nkv * n_valid * hd);
                            for h in 0..nkv {
                                for p in 0..n_valid {
                                    let off = h * cap * hd + p * hd;
                                    k_valid.extend_from_slice(&k_full[off..off+hd]);
                                    v_valid.extend_from_slice(&v_full[off..off+hd]);
                                }
                            }
                            for (name, buf) in [("k_cache_upto", &k_valid), ("v_cache_upto", &v_valid)] {
                                let path = format!(
                                    "{dump_dir}/hf2q_prefill_{name}_layer{layer_idx:02}_tok{tok_i:03}.bin");
                                let bytes: &[u8] = unsafe {
                                    std::slice::from_raw_parts(
                                        buf.as_ptr() as *const u8, buf.len() * 4) };
                                std::fs::write(&path, bytes)
                                    .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
                                eprintln!(
                                    "[PREFILL DUMP] {} [{},{},{}] f32 -> {}",
                                    name, nkv, n_valid, hd, path);
                            }
                        }
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- O-proj (same as decode) --
                    s.barrier_between(
                        &[&self.activations.sdpa_out, &self.layers[layer_idx].attn.o_proj.buffer],
                        &[&self.activations.attn_out],
                    );
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.sdpa_out,
                        &self.layers[layer_idx].attn.o_proj, &mut self.activations.attn_out, 1)?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(&dump_dir, "attn_out_pre_resid", layer_idx, tok_i,
                                        &self.activations.attn_out, hs)?;
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- Fused post-attention norm + residual add --
                    s.barrier_between(
                        &[&self.activations.hidden, &self.activations.attn_out],
                        &[&self.activations.residual],
                    );
                    mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.hidden,
                        &self.activations.attn_out,
                        &self.layers[layer_idx].norms.post_attention_layernorm,
                        &self.activations.residual,
                        hs as u32, 1, eps,
                    ).map_err(|e| anyhow::anyhow!("prefill post-attn L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(&dump_dir, "residual", layer_idx, tok_i,
                                        &self.activations.residual, hs)?;
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // ============================================================
                    // Dense MLP + MoE (identical to forward_decode)
                    // ============================================================
                    let num_experts = self.num_experts;
                    let top_k = self.layers[layer_idx].moe.top_k;

                    // B8: pre-FF norms [3 concurrent]
                    s.barrier_between(
                        &[&self.activations.residual],
                        &[&self.activations.norm_out, &self.activations.moe_norm_out,
                          &self.activations.router_norm_out],
                    );
                    s.rms_norm(reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                        &self.activations.norm_out,
                        &self.activations.norm_params, 1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill pre-FF1 L{layer_idx} T{tok_i}: {e}"))?;
                    s.rms_norm(reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                        &self.activations.moe_norm_out,
                        &self.activations.norm_params, 1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill pre-FF2 L{layer_idx} T{tok_i}: {e}"))?;
                    s.rms_norm(reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].moe.router_combined_weight,
                        &self.activations.router_norm_out,
                        &self.activations.norm_params, 1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill router norm L{layer_idx} T{tok_i}: {e}"))?;

                    // B9: gate + up + router [3 concurrent]
                    s.barrier_between(
                        &[&self.activations.norm_out, &self.activations.router_norm_out],
                        &[&self.activations.mlp_gate, &self.activations.mlp_up,
                          &self.activations.moe_router_logits],
                    );
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].mlp.gate_proj, &mut self.activations.mlp_gate, 1)?;
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].mlp.up_proj, &mut self.activations.mlp_up, 1)?;
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.router_norm_out,
                        &self.layers[layer_idx].moe.router_proj,
                        &mut self.activations.moe_router_logits, 1)?;

                    // B10: gelu_mul + moe_routing [2 concurrent]
                    s.barrier_between(
                        &[&self.activations.mlp_gate, &self.activations.mlp_up,
                          &self.activations.moe_router_logits],
                        &[&self.activations.mlp_fused,
                          &self.activations.moe_expert_ids, &self.activations.moe_routing_weights_gpu],
                    );
                    {
                        use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};
                        let n_elements_bytes = (self.intermediate_size as u32).to_ne_bytes();
                        let pipeline = reg.get_pipeline("fused_gelu_mul", metal_dev)?;
                        encode_with_args(
                            s.encoder_mut(), pipeline,
                            &[
                                (0, KernelArg::Buffer(&self.activations.mlp_gate)),
                                (1, KernelArg::Buffer(&self.activations.mlp_up)),
                                (2, KernelArg::Buffer(&self.activations.mlp_fused)),
                                (3, KernelArg::Bytes(&n_elements_bytes)),
                            ],
                            mlx_native::MTLSize::new(self.intermediate_size as u64, 1, 1),
                            mlx_native::MTLSize::new(
                                std::cmp::min(256, self.intermediate_size as u64), 1, 1),
                        );
                    }
                    mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_router_logits,
                        &self.activations.moe_expert_ids,
                        &self.activations.moe_routing_weights_gpu,
                        &self.layers[layer_idx].moe.per_expert_scale,
                        num_experts as u32, top_k as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill MoE routing L{layer_idx} T{tok_i}: {e}"))?;

                    // MoE expert dispatch (fused _id path)
                    let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;
                    if self.layers[layer_idx].moe.stacked_gate_up.is_none()
                        || self.layers[layer_idx].moe.stacked_down.is_none()
                    {
                        anyhow::bail!("Prefill requires fused _id path (stacked weights) at L{layer_idx}");
                    }

                    // B11: dense down + gate_up_id
                    s.barrier_between(
                        &[&self.activations.mlp_fused, &self.layers[layer_idx].mlp.down_proj.buffer],
                        &[&self.activations.mlp_down],
                    );
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.mlp_fused,
                        &self.layers[layer_idx].mlp.down_proj, &mut self.activations.mlp_down, 1)?;

                    let ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                    s.barrier_between(
                        &[&self.activations.moe_norm_out, &self.activations.moe_expert_ids,
                          self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap()],
                        &[&self.activations.moe_gate_up_id_out],
                    );
                    s.quantized_matmul_id_ggml(
                        reg, dev,
                        &self.activations.moe_norm_out,
                        self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap(),
                        &self.activations.moe_expert_ids,
                        &mut self.activations.moe_gate_up_id_out,
                        &mlx_native::GgmlQuantizedMatmulIdParams {
                            n_tokens: 1,
                            top_k: top_k as u32,
                            n: (2 * moe_int) as u32,
                            k: hs as u32,
                            n_experts: num_experts as u32,
                            expert_stride: self.layers[layer_idx].moe.gate_up_expert_stride,
                            ggml_type: ggml_type_gu,
                        },
                    ).map_err(|e| anyhow::anyhow!("prefill gate_up_id L{layer_idx} T{tok_i}: {e}"))?;

                    // B12: swiglu
                    s.barrier_between(
                        &[&self.activations.moe_gate_up_id_out],
                        &[&self.activations.moe_swiglu_id_out],
                    );
                    mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_gate_up_id_out,
                        &self.activations.moe_swiglu_id_out,
                        moe_int, top_k,
                    ).map_err(|e| anyhow::anyhow!("prefill swiglu L{layer_idx} T{tok_i}: {e}"))?;

                    // B13: down_id + post-FF norm1
                    let ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;
                    s.barrier_between(
                        &[&self.activations.moe_swiglu_id_out, &self.activations.moe_expert_ids,
                          self.layers[layer_idx].moe.stacked_down.as_ref().unwrap()],
                        &[&self.activations.moe_down_id_out],
                    );
                    s.quantized_matmul_id_ggml(
                        reg, dev,
                        &self.activations.moe_swiglu_id_out,
                        self.layers[layer_idx].moe.stacked_down.as_ref().unwrap(),
                        &self.activations.moe_expert_ids,
                        &mut self.activations.moe_down_id_out,
                        &mlx_native::GgmlQuantizedMatmulIdParams {
                            n_tokens: top_k as u32,
                            top_k: 1,
                            n: hs as u32,
                            k: moe_int as u32,
                            n_experts: num_experts as u32,
                            expert_stride: self.layers[layer_idx].moe.down_expert_stride,
                            ggml_type: ggml_type_dn,
                        },
                    ).map_err(|e| anyhow::anyhow!("prefill down_id L{layer_idx} T{tok_i}: {e}"))?;

                    s.barrier_between(
                        &[&self.activations.mlp_down],
                        &[&self.activations.attn_out],
                    );
                    s.rms_norm(reg, metal_dev,
                        &self.activations.mlp_down,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                        &self.activations.attn_out,
                        &self.activations.norm_params, 1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill post-FF1 L{layer_idx} T{tok_i}: {e}"))?;

                    // B14: weighted_sum
                    s.barrier_between(
                        &[&self.activations.moe_down_id_out, &self.activations.moe_routing_weights_gpu],
                        &[&self.activations.moe_accum],
                    );
                    mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_down_id_out,
                        &self.activations.moe_routing_weights_gpu,
                        &self.activations.moe_accum,
                        hs, top_k,
                    ).map_err(|e| anyhow::anyhow!("prefill weighted_sum L{layer_idx} T{tok_i}: {e}"))?;

                    // Post-FF norm2 + combine
                    s.barrier_between(
                        &[&self.activations.attn_out, &self.activations.moe_accum],
                        &[&self.activations.mlp_down],
                    );
                    mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_out,
                        &self.activations.moe_accum,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                        &self.activations.mlp_down,
                        hs as u32, 1, eps,
                    ).map_err(|e| anyhow::anyhow!("prefill post-FF2 L{layer_idx} T{tok_i}: {e}"))?;

                    // End-of-layer: norm + residual + scalar
                    let scalar_is_vector = self.layers[layer_idx].layer_scalar.element_count() > 1;
                    s.barrier_between(
                        &[&self.activations.residual, &self.activations.mlp_down],
                        &[&self.activations.hidden],
                    );
                    mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_scalar_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.residual,
                        &self.activations.mlp_down,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm,
                        &self.activations.hidden,
                        &self.layers[layer_idx].layer_scalar,
                        1, hs as u32, eps,
                        scalar_is_vector,
                    ).map_err(|e| anyhow::anyhow!("prefill end-layer L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(&dump_dir, "l_out", layer_idx, tok_i,
                                        &self.activations.hidden, hs)?;
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }
                }

                // --- 3. Final norm + lm_head + softcap + argmax ---
                s.barrier_between(
                    &[&self.activations.hidden, &self.final_norm],
                    &[&self.activations.norm_out],
                );
                s.rms_norm(reg, metal_dev,
                    &self.activations.hidden,
                    &self.final_norm,
                    &self.activations.norm_out,
                    &self.activations.norm_params, 1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("prefill final norm T{tok_i}: {e}"))?;

                if let Some(ref lm_head_f16) = self.lm_head_f16 {
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
                    ).map_err(|e| anyhow::anyhow!("prefill lm_head T{tok_i}: {e}"))?;
                } else {
                    anyhow::bail!("Prefill requires GPU lm_head (F16 weight)");
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
                    ).map_err(|e| anyhow::anyhow!("prefill softcap T{tok_i}: {e}"))?;
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
                ).map_err(|e| anyhow::anyhow!("prefill argmax T{tok_i}: {e}"))?;

                s.finish()
                    .map_err(|e| anyhow::anyhow!("prefill finish T{tok_i}: {e}"))?;

                last_token = {
                    let idx: &[u32] = self.activations.argmax_index.as_slice()
                        .map_err(|e| anyhow::anyhow!("prefill argmax read T{tok_i}: {e}"))?;
                    idx[0]
                };
            }
        }

        let prefill_elapsed = prefill_start.elapsed();
        eprintln!(
            "Prefill complete (dense SDPA): {} tokens in {:.1} ms ({:.1} tok/s), first decode token = {}",
            seq_len,
            prefill_elapsed.as_secs_f64() * 1000.0,
            seq_len as f64 / prefill_elapsed.as_secs_f64(),
            last_token,
        );

        // Store dense KV buffers on self so forward_decode can use them
        // for dense attention during the decode phase (ADR-009 Track 3).
        self.dense_kvs = Some(dense_kvs_vec);
        self.dense_sdpa_tmp = Some(sdpa_tmp);

        Ok(last_token)
    }
}
