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

use crate::debug::INVESTIGATION_ENV;
use super::forward_mlx::{
    MlxModelWeights, DenseKvBuffers, HbKvBuffers, dispatch_qmatmul,
    dispatch_rms_norm_unit_perhead, RmsNormPerHeadArgs,
};
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
        let use_f16_kv = INVESTIGATION_ENV.f16_kv;
        let kv_dtype = if use_f16_kv { mlx_native::DType::F16 } else { mlx_native::DType::F32 };
        let kv_elem_bytes = if use_f16_kv { 2 } else { 4 };
        tracing::debug!("Prefill: KV cache dtype = {:?}", kv_dtype);

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
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let nkv = layer.num_kv_heads;
            let hd = layer.head_dim;
            let layer_is_ring = layer.layer_type == LayerType::Sliding;
            let capacity = if layer_is_ring { sw } else { linear_capacity };
            let n = nkv * capacity * hd;
            let k = dev.alloc_buffer(n * kv_elem_bytes, kv_dtype, vec![nkv, capacity, hd])
                .map_err(|e| anyhow::anyhow!("prefill dense K L{layer_idx}: {e}"))?;
            let v = dev.alloc_buffer(n * kv_elem_bytes, kv_dtype, vec![nkv, capacity, hd])
                .map_err(|e| anyhow::anyhow!("prefill dense V L{layer_idx}: {e}"))?;
            dense_kvs_vec.push(DenseKvBuffers { k, v, capacity, is_sliding: layer_is_ring });
        }

        // Tmp buffer for flash_attn_vec (sized for largest layer config)
        let max_nh = self.num_attention_heads;
        let max_hd = self.layers.iter().map(|l| l.head_dim).max().unwrap_or(512);
        let tmp_bytes = mlx_native::ops::flash_attn_vec::tmp_buffer_bytes(
            max_nh as u32, max_hd as u32);
        let sdpa_tmp = dev.alloc_buffer(tmp_bytes, mlx_native::DType::F32,
            vec![tmp_bytes / 4])
            .map_err(|e| anyhow::anyhow!("prefill sdpa_tmp: {e}"))?;

        tracing::debug!("Prefill: {} tokens × {} layers (dense SDPA)", seq_len, num_layers);

        // Track A fix (iter-21): allocate leg_f_kvs shadow cache BEFORE the token
        // loop so the per-token populate code (below) can write into it.
        // In iter-20 the allocation appeared at the END of this function (after
        // the loop), so self.leg_f_kvs was always None when the populate block ran.
        {
            let force_dense_on_tq =
                std::env::var("HF2Q_FORCE_DENSE_SDPA_ON_TQ_KV").ok().as_deref() == Some("1");
            if force_dense_on_tq {
                eprintln!("[iter-21 Track A] Allocating Leg F shadow KV cache before prefill loop");
                let mut leg_f_kvs_vec: Vec<DenseKvBuffers> = Vec::with_capacity(num_layers);
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let nkv = layer.num_kv_heads;
                    let hd = layer.head_dim;
                    let layer_is_ring = layer.layer_type == LayerType::Sliding;
                    let capacity = if layer_is_ring { sw } else { linear_capacity };
                    let n = nkv * capacity * hd;
                    let f32_bytes = std::mem::size_of::<f32>();
                    let k = dev.alloc_buffer(n * f32_bytes, mlx_native::DType::F32, vec![nkv, capacity, hd])
                        .map_err(|e| anyhow::anyhow!("leg_f early K L{layer_idx}: {e}"))?;
                    let v = dev.alloc_buffer(n * f32_bytes, mlx_native::DType::F32, vec![nkv, capacity, hd])
                        .map_err(|e| anyhow::anyhow!("leg_f early V L{layer_idx}: {e}"))?;
                    leg_f_kvs_vec.push(DenseKvBuffers { k, v, capacity, is_sliding: layer_is_ring });
                }
                let leg_f_tmp_bytes = mlx_native::ops::flash_attn_vec::tmp_buffer_bytes(
                    max_nh as u32, max_hd as u32);
                let leg_f_sdpa_tmp = dev.alloc_buffer(leg_f_tmp_bytes, mlx_native::DType::F32,
                    vec![leg_f_tmp_bytes / 4])
                    .map_err(|e| anyhow::anyhow!("leg_f early sdpa_tmp: {e}"))?;
                self.leg_f_kvs = Some(leg_f_kvs_vec);
                self.leg_f_sdpa_tmp = Some(leg_f_sdpa_tmp);
                eprintln!("[iter-21 Track A] Leg F shadow KV cache ready ({} layers)", num_layers);
            }
        }

        // iter-21 Track B + 2026-04-24 post-close default correction.
        // HF2Q_TQ_CODEBOOK_BITS=5|6|8 (or unset) → allocate per-layer byte-packed HB buffers.
        // Also allocates leg_f_kvs (F32 shadow) if not already allocated by Leg F above.
        // MUST stay in lockstep with forward_mlx.rs::tq_codebook_bits and cb_bits gates.
        //   unset (DEFAULT) = 8-bit native HB SDPA
        //   "4"             = legacy 4-bit (no HB buffers)
        //   "5" | "6" | "8" = corresponding HB bits
        let tq_codebook_bits_prefill: u32 = match std::env::var("HF2Q_TQ_CODEBOOK_BITS").as_deref() {
            Ok("4") => 0,
            Ok("5") => 5, Ok("6") => 6, Ok("8") => 8,
            _ => 8,  // DEFAULT: 8-bit
        };
        if tq_codebook_bits_prefill >= 5 {
            eprintln!("[iter-21 Track B] Allocating leg_hb_encoded ({}-bit, {} layers)",
                      tq_codebook_bits_prefill, num_layers);
            let mut leg_hb_vec: Vec<HbKvBuffers> = Vec::with_capacity(num_layers);
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let nkv = layer.num_kv_heads;
                let hd = layer.head_dim;
                let layer_is_ring = layer.layer_type == LayerType::Sliding;
                let capacity = if layer_is_ring { sw } else { linear_capacity };
                let norms_per_pos = (hd / 256).max(1);
                let norms_n = nkv * capacity * norms_per_pos;
                let k_packed = dev.alloc_buffer(nkv * capacity * hd, mlx_native::DType::U8,
                    vec![nkv, capacity, hd])
                    .map_err(|e| anyhow::anyhow!("leg_hb prefill K packed L{layer_idx}: {e}"))?;
                let k_norms = dev.alloc_buffer(norms_n * 4, mlx_native::DType::F32,
                    if norms_per_pos == 1 { vec![nkv, capacity] } else { vec![nkv, capacity, norms_per_pos] })
                    .map_err(|e| anyhow::anyhow!("leg_hb prefill K norms L{layer_idx}: {e}"))?;
                let v_packed = dev.alloc_buffer(nkv * capacity * hd, mlx_native::DType::U8,
                    vec![nkv, capacity, hd])
                    .map_err(|e| anyhow::anyhow!("leg_hb prefill V packed L{layer_idx}: {e}"))?;
                let v_norms = dev.alloc_buffer(norms_n * 4, mlx_native::DType::F32,
                    if norms_per_pos == 1 { vec![nkv, capacity] } else { vec![nkv, capacity, norms_per_pos] })
                    .map_err(|e| anyhow::anyhow!("leg_hb prefill V norms L{layer_idx}: {e}"))?;
                leg_hb_vec.push(HbKvBuffers {
                    k_packed, k_norms, v_packed, v_norms,
                    capacity, is_sliding: layer_is_ring, norms_per_pos,
                });
            }
            self.leg_hb_encoded = Some(leg_hb_vec);

            // Also ensure leg_f_kvs shadow (F32) is allocated for the dequant→SDPA step.
            if self.leg_f_kvs.is_none() {
                let mut leg_f_vec: Vec<DenseKvBuffers> = Vec::with_capacity(num_layers);
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let nkv = layer.num_kv_heads;
                    let hd = layer.head_dim;
                    let layer_is_ring = layer.layer_type == LayerType::Sliding;
                    let capacity = if layer_is_ring { sw } else { linear_capacity };
                    let n = nkv * capacity * hd;
                    let k = dev.alloc_buffer(n * 4, mlx_native::DType::F32, vec![nkv, capacity, hd])
                        .map_err(|e| anyhow::anyhow!("leg_f hb-path K L{layer_idx}: {e}"))?;
                    let v = dev.alloc_buffer(n * 4, mlx_native::DType::F32, vec![nkv, capacity, hd])
                        .map_err(|e| anyhow::anyhow!("leg_f hb-path V L{layer_idx}: {e}"))?;
                    leg_f_vec.push(DenseKvBuffers { k, v, capacity, is_sliding: layer_is_ring });
                }
                let tmp_bytes = mlx_native::ops::flash_attn_vec::tmp_buffer_bytes(
                    max_nh as u32, max_hd as u32);
                let leg_f_sdpa_tmp = dev.alloc_buffer(tmp_bytes, mlx_native::DType::F32,
                    vec![tmp_bytes / 4])
                    .map_err(|e| anyhow::anyhow!("leg_f hb-path sdpa_tmp: {e}"))?;
                self.leg_f_kvs = Some(leg_f_vec);
                self.leg_f_sdpa_tmp = Some(leg_f_sdpa_tmp);
            }
            eprintln!("[iter-21 Track B] leg_hb_encoded + leg_f_kvs ready ({} layers)", num_layers);
        }

        // ADR-010 one-shot norm weight dump: read self.layers[L].norms.input_layernorm
        // as the hf2q kernel sees it, compare against the raw GGUF tensor.
        // Gated on HF2Q_DUMP_NORM_WEIGHT="layer" (e.g. "7"). Writes to HF2Q_DUMP_DIR.
        if let Some(target_l) = INVESTIGATION_ENV.dump_norm_weight {
            if target_l < num_layers {
                let w: &[f32] = self.layers[target_l].norms.input_layernorm.as_slice()
                    .map_err(|e| anyhow::anyhow!("norm weight read L{target_l}: {e}"))?;
                let dir = &INVESTIGATION_ENV.dump_dir;
                let path = format!("{dir}/hf2q_input_layernorm_weight_layer{target_l:02}.bin");
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) };
                std::fs::write(&path, bytes)
                    .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
                eprintln!("[DUMP] input_layernorm weight L{target_l} [{}] f32 -> {}",
                          w.len(), path);
            }
        }

        // ADR-009 Phase 3A: prefill boundary dumps at (target_layer, target_tok).
        // Controlled by HF2Q_PREFILL_DUMP="layer,tok" e.g. "7,34".
        let prefill_dump: Option<(usize, usize)> = INVESTIGATION_ENV.prefill_dump;
        let dump_dir: &str = &INVESTIGATION_ENV.dump_dir;

        // Track A fix (iter-21): Leg F shadow-cache prefill population.
        // tq_scale_factor_d512 matches the decode-path value so prefill and
        // decode dequant use the same scale, keeping the shadow KV cache
        // byte-compatible across the prefill→decode boundary.
        let tq_scale_factor_d512: f32 = {
            match std::env::var("HF2Q_SCALE_FORMULA").as_deref() {
                Ok("sqrt256") => 16.0_f32,
                Ok("sqrt512") => 512.0_f32.sqrt(),
                _ => 1.0_f32, // bare (iter-16 default)
            }
        };

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
                    let layer = &self.layers[layer_idx];
                    let hd = layer.head_dim;
                    let nkv = layer.num_kv_heads;
                    let nh = self.num_attention_heads;
                    let is_sliding = layer.layer_type == LayerType::Sliding;
                    let (kv_is_sliding, kv_write_pos, kv_capacity, _kv_seq_len) = kv_info[layer_idx];

                    // Active dump flag for this iteration
                    let dump_here = prefill_dump == Some((layer_idx, tok_i));
                    // Dump at layer-start: hidden = L(layer_idx-1) l_out (or embed for L0)
                    if dump_here {
                        s.finish()
                            .map_err(|e| anyhow::anyhow!("prefill dump L{layer_idx} T{tok_i} start finish: {e}"))?;
                        write_dump_f32(dump_dir, "pre_layer_hidden", layer_idx, tok_i,
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
                        write_dump_f32(dump_dir, "post_input_norm", layer_idx, tok_i,
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
                        write_dump_f32(dump_dir, "q_pre_normed", layer_idx, tok_i,
                                        &self.activations.attn_q, nh * hd)?;
                        write_dump_f32(dump_dir, "k_pre_normed", layer_idx, tok_i,
                                        &self.activations.attn_k, nkv * hd)?;
                        write_dump_f32(dump_dir, "q_normed", layer_idx, tok_i,
                                        &self.activations.attn_q_normed, nh * hd)?;
                        write_dump_f32(dump_dir, "k_normed", layer_idx, tok_i,
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
                            &RmsNormPerHeadArgs {
                                input: &self.activations.attn_k,
                                output: &self.activations.attn_v,
                                params_buf: hd_norm_params,
                                rows: nkv as u32,
                                dim: hd as u32,
                            },
                        )?;
                    } else {
                        s.barrier_between(
                            &[&self.activations.attn_v],
                            &[&self.activations.moe_expert_out],
                        );
                        dispatch_rms_norm_unit_perhead(
                            s.encoder_mut(), reg, metal_dev,
                            &RmsNormPerHeadArgs {
                                input: &self.activations.attn_v,
                                output: &self.activations.moe_expert_out,
                                params_buf: hd_norm_params,
                                rows: nkv as u32,
                                dim: hd as u32,
                            },
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
                    if !INVESTIGATION_ENV.skip_tq_encode {
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
                            None, // scale_factor_d512: bare=1.0 for prefill
                            None, // rms_scratch: probe not used during prefill
                        ).map_err(|e| anyhow::anyhow!("prefill TQ K L{layer_idx} T{tok_i}: {e}"))?;
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                            s.encoder_mut(), reg, metal_dev,
                            v_src,
                            &self.kv_caches[layer_idx].v_packed,
                            &self.kv_caches[layer_idx].v_norms,
                            nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                            kv_is_sliding,
                            None, // scale_factor_d512: bare=1.0 for prefill
                            None, // rms_scratch: probe not used during prefill
                        ).map_err(|e| anyhow::anyhow!("prefill TQ V L{layer_idx} T{tok_i}: {e}"))?;
                    }

                    // Track A fix (iter-21): populate leg_f_kvs shadow cache
                    // during prefill so decode Leg F can attend over all prompt
                    // positions.  Without this, the shadow cache was all-zeros
                    // for positions 0..seq_len-1, making Q*K^T garbage (6 bytes).
                    //
                    // Pipeline mirrors the decode path (forward_mlx.rs:1869-1945):
                    //   dequant K → copy to leg_f_kvs.k, dequant V → copy to leg_f_kvs.v.
                    // Both K and V are dequantized from the just-written TQ cache position.
                    // Guard: only when TQ encode ran (skip_tq_encode == false).
                    // Guard: skip if Track B (tq_codebook_bits >= 5) — Track B populates
                    //   leg_f_kvs from HB-encoded data below, not 4-bit TQ dequant.
                    //   Running both would corrupt leg_f_kvs with 4-bit values before
                    //   Track B overwrites them, and more importantly the Leg F dequant
                    //   would overwrite attn_k_normed before the HB encode reads it.
                    if !INVESTIGATION_ENV.skip_tq_encode && tq_codebook_bits_prefill == 0 {
                    if let Some(ref leg_f_kvs) = self.leg_f_kvs {
                        let leg_f_cap = leg_f_kvs[layer_idx].capacity;
                        let leg_f_is_ring = leg_f_kvs[layer_idx].is_sliding;
                        let lf_write_slot = if leg_f_is_ring {
                            (tok_i % leg_f_cap) as u32
                        } else {
                            tok_i as u32
                        };
                        let cache_pos_read = if kv_is_sliding {
                            (kv_write_pos % kv_capacity) as u32
                        } else {
                            kv_write_pos as u32
                        };

                        // Dequantize K into attn_k_normed scratch.
                        s.barrier_between(
                            &[&self.kv_caches[layer_idx].k_packed,
                              &self.kv_caches[layer_idx].k_norms],
                            &[&self.activations.attn_k_normed],
                        );
                        mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_kv(
                            s.encoder_mut(), reg, metal_dev,
                            &self.kv_caches[layer_idx].k_packed,
                            &self.kv_caches[layer_idx].k_norms,
                            &self.activations.attn_k_normed,
                            nkv as u32, hd as u32, kv_capacity as u32,
                            cache_pos_read, tq_scale_factor_d512,
                        ).map_err(|e| anyhow::anyhow!("legF prefill dequant_k L{layer_idx} T{tok_i}: {e}"))?;

                        // Copy dequantized K into shadow cache at lf_write_slot.
                        s.barrier_between(
                            &[&self.activations.attn_k_normed],
                            &[&leg_f_kvs[layer_idx].k],
                        );
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &leg_f_kvs[layer_idx].k,
                            nkv as u32, hd as u32, leg_f_cap as u32, lf_write_slot,
                        ).map_err(|e| anyhow::anyhow!("legF prefill K cache copy L{layer_idx} T{tok_i}: {e}"))?;

                        // Dequantize V into attn_k_normed scratch (K copy is committed).
                        s.barrier_between(
                            &[&self.kv_caches[layer_idx].v_packed,
                              &self.kv_caches[layer_idx].v_norms],
                            &[&self.activations.attn_k_normed],
                        );
                        mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_kv(
                            s.encoder_mut(), reg, metal_dev,
                            &self.kv_caches[layer_idx].v_packed,
                            &self.kv_caches[layer_idx].v_norms,
                            &self.activations.attn_k_normed,
                            nkv as u32, hd as u32, kv_capacity as u32,
                            cache_pos_read, tq_scale_factor_d512,
                        ).map_err(|e| anyhow::anyhow!("legF prefill dequant_v L{layer_idx} T{tok_i}: {e}"))?;

                        // Copy dequantized V into shadow cache at lf_write_slot.
                        s.barrier_between(
                            &[&self.activations.attn_k_normed],
                            &[&leg_f_kvs[layer_idx].v],
                        );
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &leg_f_kvs[layer_idx].v,
                            nkv as u32, hd as u32, leg_f_cap as u32, lf_write_slot,
                        ).map_err(|e| anyhow::anyhow!("legF prefill V cache copy L{layer_idx} T{tok_i}: {e}"))?;
                    } // end if let Some(leg_f_kvs)
                    } // end if !skip_tq_encode

                    // iter-21 Track B: HB encode + dequant → leg_f_kvs shadow during prefill.
                    // When HF2Q_TQ_CODEBOOK_BITS=5|6, also encode K/V to byte-packed HB,
                    // then dequantize into leg_f_kvs so decode Track B SDPA sees all positions.
                    if tq_codebook_bits_prefill >= 5 && !INVESTIGATION_ENV.skip_tq_encode {
                    if let (Some(ref leg_hb_enc), Some(ref leg_f_kvs)) =
                        (&self.leg_hb_encoded, &self.leg_f_kvs)
                    {
                        let hb_cap = leg_hb_enc[layer_idx].capacity;
                        let hb_is_ring = leg_hb_enc[layer_idx].is_sliding;
                        let hb_write_slot = if hb_is_ring {
                            (tok_i % hb_cap) as u32
                        } else {
                            tok_i as u32
                        };
                        let lf_cap = leg_f_kvs[layer_idx].capacity;
                        let lf_is_ring = leg_f_kvs[layer_idx].is_sliding;
                        let lf_write_slot = if lf_is_ring {
                            (tok_i % lf_cap) as u32
                        } else {
                            tok_i as u32
                        };

                        // HB encode K → leg_hb_enc.k_packed
                        s.barrier_between(
                            &[&self.activations.attn_k_normed, v_src],
                            &[&leg_hb_enc[layer_idx].k_packed, &leg_hb_enc[layer_idx].k_norms],
                        );
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &leg_hb_enc[layer_idx].k_packed,
                            &leg_hb_enc[layer_idx].k_norms,
                            nkv as u32, hd as u32, hb_cap as u32, hb_write_slot,
                            hb_is_ring, tq_scale_factor_d512, tq_codebook_bits_prefill,
                        ).map_err(|e| anyhow::anyhow!("prefill hb_encode K L{layer_idx} T{tok_i}: {e}"))?;

                        // HB encode V → leg_hb_enc.v_packed
                        s.barrier_between(
                            &[v_src],
                            &[&leg_hb_enc[layer_idx].v_packed, &leg_hb_enc[layer_idx].v_norms],
                        );
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                            s.encoder_mut(), reg, metal_dev,
                            v_src,
                            &leg_hb_enc[layer_idx].v_packed,
                            &leg_hb_enc[layer_idx].v_norms,
                            nkv as u32, hd as u32, hb_cap as u32, hb_write_slot,
                            hb_is_ring, tq_scale_factor_d512, tq_codebook_bits_prefill,
                        ).map_err(|e| anyhow::anyhow!("prefill hb_encode V L{layer_idx} T{tok_i}: {e}"))?;

                        // Dequantize HB K → attn_k_normed scratch → leg_f_kvs.k
                        s.barrier_between(
                            &[&leg_hb_enc[layer_idx].k_packed, &leg_hb_enc[layer_idx].k_norms],
                            &[&self.activations.attn_k_normed],
                        );
                        mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv(
                            s.encoder_mut(), reg, metal_dev,
                            &leg_hb_enc[layer_idx].k_packed,
                            &leg_hb_enc[layer_idx].k_norms,
                            &self.activations.attn_k_normed,
                            nkv as u32, hd as u32, hb_cap as u32,
                            hb_write_slot, tq_scale_factor_d512, tq_codebook_bits_prefill,
                        ).map_err(|e| anyhow::anyhow!("prefill hb_dequant_k L{layer_idx} T{tok_i}: {e}"))?;

                        s.barrier_between(
                            &[&self.activations.attn_k_normed],
                            &[&leg_f_kvs[layer_idx].k],
                        );
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &leg_f_kvs[layer_idx].k,
                            nkv as u32, hd as u32, lf_cap as u32, lf_write_slot,
                        ).map_err(|e| anyhow::anyhow!("prefill hb K→lf copy L{layer_idx} T{tok_i}: {e}"))?;

                        // Dequantize HB V → attn_k_normed scratch → leg_f_kvs.v
                        s.barrier_between(
                            &[&leg_hb_enc[layer_idx].v_packed, &leg_hb_enc[layer_idx].v_norms],
                            &[&self.activations.attn_k_normed],
                        );
                        mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv(
                            s.encoder_mut(), reg, metal_dev,
                            &leg_hb_enc[layer_idx].v_packed,
                            &leg_hb_enc[layer_idx].v_norms,
                            &self.activations.attn_k_normed,
                            nkv as u32, hd as u32, hb_cap as u32,
                            hb_write_slot, tq_scale_factor_d512, tq_codebook_bits_prefill,
                        ).map_err(|e| anyhow::anyhow!("prefill hb_dequant_v L{layer_idx} T{tok_i}: {e}"))?;

                        s.barrier_between(
                            &[&self.activations.attn_k_normed],
                            &[&leg_f_kvs[layer_idx].v],
                        );
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &leg_f_kvs[layer_idx].v,
                            nkv as u32, hd as u32, lf_cap as u32, lf_write_slot,
                        ).map_err(|e| anyhow::anyhow!("prefill hb V→lf copy L{layer_idx} T{tok_i}: {e}"))?;
                    } // end if let (leg_hb_enc, leg_f_kvs)
                    } // end if tq_codebook_bits_prefill >= 5

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
                        write_dump_f32(dump_dir, "sdpa_out", layer_idx, tok_i,
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
                        write_dump_f32(dump_dir, "attn_out_pre_resid", layer_idx, tok_i,
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
                        write_dump_f32(dump_dir, "residual", layer_idx, tok_i,
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
                        write_dump_f32(dump_dir, "l_out", layer_idx, tok_i,
                                        &self.activations.hidden, hs)?;
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }
                }

                // C-0b: HF2Q_DUMP_TQ_STATE — dump packed KV cache at end-of-prefill
                // (last token only) for ADR-007 layer-0 localization audit.
                if INVESTIGATION_ENV.dump_tq_state && tok_i + 1 == seq_len {
                    let dump_layers_list = &INVESTIGATION_ENV.dump_tq_layers_list;
                    s.finish()
                        .map_err(|e| anyhow::anyhow!("tq_dump nonbatched finish T{tok_i}: {e}"))?;
                    for li in 0..num_layers {
                        if !dump_layers_list.is_empty() && !dump_layers_list.contains(&li) {
                            continue;
                        }
                        let layer = &self.layers[li];
                        let hd = layer.head_dim;
                        let nkv = layer.num_kv_heads;
                        let (kv_is_sliding, _kv_write_pos, kv_capacity, kv_seq_len) = kv_info[li];
                        let hd_half = hd / 2;
                        let k_raw: &[u8] = self.kv_caches[li].k_packed.as_slice()
                            .map_err(|e| anyhow::anyhow!("tq_dump nb k_packed L{li}: {e}"))?;
                        let v_raw: &[u8] = self.kv_caches[li].v_packed.as_slice()
                            .map_err(|e| anyhow::anyhow!("tq_dump nb v_packed L{li}: {e}"))?;
                        let k_norms_raw: &[f32] = self.kv_caches[li].k_norms.as_slice()
                            .map_err(|e| anyhow::anyhow!("tq_dump nb k_norms L{li}: {e}"))?;
                        let v_norms_raw: &[f32] = self.kv_caches[li].v_norms.as_slice()
                            .map_err(|e| anyhow::anyhow!("tq_dump nb v_norms L{li}: {e}"))?;
                        let mut k_tight = vec![0u8; nkv * kv_seq_len * hd_half];
                        let mut v_tight = vec![0u8; nkv * kv_seq_len * hd_half];
                        let mut kn_tight = vec![0.0f32; nkv * kv_seq_len];
                        let mut vn_tight = vec![0.0f32; nkv * kv_seq_len];
                        for h in 0..nkv {
                            for p in 0..kv_seq_len {
                                let src_packed = h * kv_capacity * hd_half + p * hd_half;
                                let dst_packed = h * kv_seq_len * hd_half + p * hd_half;
                                k_tight[dst_packed..dst_packed + hd_half]
                                    .copy_from_slice(&k_raw[src_packed..src_packed + hd_half]);
                                v_tight[dst_packed..dst_packed + hd_half]
                                    .copy_from_slice(&v_raw[src_packed..src_packed + hd_half]);
                                let src_norm = h * kv_capacity + p;
                                let dst_norm = h * kv_seq_len + p;
                                kn_tight[dst_norm] = k_norms_raw[src_norm];
                                vn_tight[dst_norm] = v_norms_raw[src_norm];
                            }
                        }
                        let dir = &INVESTIGATION_ENV.dump_dir;
                        std::fs::create_dir_all(dir.as_str())
                            .map_err(|e| anyhow::anyhow!("tq_dump nb mkdir {dir}: {e}"))?;
                        let kp = format!("{dir}/hf2q_k_packed_layer{li:02}_pos{kv_seq_len}.u8.bin");
                        let vp = format!("{dir}/hf2q_v_packed_layer{li:02}_pos{kv_seq_len}.u8.bin");
                        std::fs::write(&kp, &k_tight)
                            .map_err(|e| anyhow::anyhow!("write {kp}: {e}"))?;
                        std::fs::write(&vp, &v_tight)
                            .map_err(|e| anyhow::anyhow!("write {vp}: {e}"))?;
                        eprintln!("[TQ_DUMP] k_packed L{li:02} [{nkv},{kv_seq_len},{hd_half}] u8 -> {kp}");
                        eprintln!("[TQ_DUMP] v_packed L{li:02} [{nkv},{kv_seq_len},{hd_half}] u8 -> {vp}");
                        let kn = format!("{dir}/hf2q_k_norms_layer{li:02}_pos{kv_seq_len}.f32.bin");
                        let vn = format!("{dir}/hf2q_v_norms_layer{li:02}_pos{kv_seq_len}.f32.bin");
                        let kn_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                kn_tight.as_ptr() as *const u8, kn_tight.len() * 4)
                        };
                        let vn_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                vn_tight.as_ptr() as *const u8, vn_tight.len() * 4)
                        };
                        std::fs::write(&kn, kn_bytes)
                            .map_err(|e| anyhow::anyhow!("write {kn}: {e}"))?;
                        std::fs::write(&vn, vn_bytes)
                            .map_err(|e| anyhow::anyhow!("write {vn}: {e}"))?;
                        eprintln!("[TQ_DUMP] k_norms L{li:02} [{nkv},{kv_seq_len}] f32 -> {kn}");
                        eprintln!("[TQ_DUMP] v_norms L{li:02} [{nkv},{kv_seq_len}] f32 -> {vn}");
                        let layer_type_str = if kv_is_sliding { "sliding" } else { "global" };
                        let kv_write_pos_final = self.kv_caches[li].write_pos;
                        let meta = serde_json::json!({
                            "nkv": nkv, "nh": max_nh, "hd": hd,
                            "kv_seq_len": kv_seq_len,
                            "kv_capacity": kv_capacity,
                            "kv_write_pos": kv_write_pos_final,
                            "kv_is_sliding": kv_is_sliding,
                            "ring_start": 0,
                            "sliding_window": sw,
                            "mask_type": 1,
                            "layer_type": layer_type_str,
                            "path": "nonbatched"
                        });
                        let meta_str = serde_json::to_string_pretty(&meta)
                            .map_err(|e| anyhow::anyhow!("meta json nb L{li}: {e}"))?;
                        let mp = format!("{dir}/hf2q_tq_meta_layer{li:02}_pos{kv_seq_len}.json");
                        std::fs::write(&mp, meta_str.as_bytes())
                            .map_err(|e| anyhow::anyhow!("write {mp}: {e}"))?;
                        eprintln!("[TQ_DUMP] meta L{li:02} -> {mp}");
                    }
                    s = exec.begin()
                        .map_err(|e| anyhow::anyhow!("tq_dump nonbatched re-begin: {e}"))?;
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
                    ).map_err(|e| anyhow::anyhow!("prefill lm_head Q8 T{tok_i}: {e}"))?;
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
                    ).map_err(|e| anyhow::anyhow!("prefill lm_head T{tok_i}: {e}"))?;
                } else {
                    anyhow::bail!("Prefill requires GPU lm_head (F16 or Q8 weight)");
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
        tracing::debug!(
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

        // Note: iter-20 had leg_f_kvs allocation here (after the loop).
        // iter-21 Track A moved it to BEFORE the loop (see above) so the
        // per-token populate block can write into it during prefill.

        Ok(last_token)
    }
}
