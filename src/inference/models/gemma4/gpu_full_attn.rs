//! Per-layer attention (and interleaved FFN) encoding for the Gemma 4 forward pass.
//!
//! Path A: `encode_one_layer` keeps attention + FFN interleaved (as in the monolith).
//! Path B follow-up will extract `encode_attention_block` + `encode_ffn_block` separately.
//!
//! Moved from `src/serve/forward_mlx.rs` by ADR-038 Step 3.

use anyhow::Result;
use mlx_native::{KernelRegistry, MlxBuffer, MlxDevice};
use mlx_native::ops::flash_attn_vec_tq::FlashAttnVecTqParams;
use mlx_native::ops::dense_gemm::DenseGemmF16Params;
use mlx_native::ops::elementwise::CastDirection;

use anyhow::Context as _;
use crate::debug::{dumps, INVESTIGATION_ENV};
use crate::serve::config::LayerType;
use crate::serve::gpu::GpuContext;
use crate::serve::layer_ctx::LayerCtx;
use crate::serve::forward_mlx_shared::{
    dispatch_qmatmul, dispatch_rms_norm_unit_perhead, rms_norm_f32_hs_cached,
    RmsNormPerHeadArgs,
};
use super::profile::TokenProfile;
use super::model::{MlxModelWeights, MlxDecoderLayerWeights};
use super::kv_cache::DecodeRegime;

impl MlxModelWeights {
    /// Run one decode step through mlx-native's GraphExecutor.
    ///
    /// MVP: one GraphSession per layer (30 sessions per forward pass).
    /// Each session encodes all ops for that layer, then commits.
    /// The final session handles the lm_head + argmax.
    ///
    /// Arguments:
    ///   - input_token: the token ID to embed
    ///   - seq_pos: position in the sequence (for RoPE and KV cache)
    ///   - gpu: the GpuContext holding the executor and registry
    ///   - profile: optional per-token profile accumulator
    ///
    /// ADR-028 iter-391: stub method for the upcoming layer-body extraction.
    /// Currently UNUSED (forward_decode keeps inline layer loop).  iter-392+
    /// will incrementally populate this method with code from the existing
    /// layer loop body, then switch forward_decode to call it instead.
    ///
    /// When complete, this method will be the unit of work that the
    /// EncoderWorker thread runs in parallel with the main thread for the
    /// second-half of layer encoding.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_one_layer<'sess>(
        &self,
        layer_idx: usize,
        ctx: &LayerCtx<'_>,
        session: &mut mlx_native::graph::GraphSession<'sess>,
        exec: &'sess mlx_native::GraphExecutor,
        reg: &mut mlx_native::KernelRegistry,
        profile: &mut Option<TokenProfile>,
        per_layer_disp_log: &mut Vec<(usize, bool, u64)>,
        total_dispatches: &mut usize,
    ) -> Result<()> {
        let dev = exec.device();
        let metal_dev = dev.metal_device();
        let hs = ctx.hidden_size;
        let seq_pos = ctx.seq_pos;
        let dump_layers = ctx.dump_layers;
        let dump_detail_layer = ctx.dump_detail_layer;
        let dump_sliding_l0 = ctx.dump_sliding_l0;
        let dump_run_name = ctx.dump_run_name;
        let dual_buffer_splits = ctx.dual_buffer_splits;
        let per_layer_disp_enabled = ctx.per_layer_disp_enabled;
                let layer_disp_start = if per_layer_disp_enabled {
                    mlx_native::dispatch_count()
                } else { 0 };
                let hd = self.layers[layer_idx].head_dim;
                let nkv = self.layers[layer_idx].num_kv_heads;
                let nh = self.num_attention_heads;
                let is_sliding = self.layers[layer_idx].layer_type == LayerType::Sliding;
                let eps = self.rms_norm_eps;
                let (kv_is_sliding, kv_write_pos, kv_capacity, kv_seq_len) = ctx.kv_info[layer_idx];

                // -- Pre-attention norm (GPU) --
                session.barrier_between(
                    &[&self.activations.hidden, &self.layers[layer_idx].norms.input_layernorm],
                    &[&self.activations.norm_out],
                );
                session.rms_norm(
                    reg, metal_dev,
                    &self.activations.hidden,
                    &self.layers[layer_idx].norms.input_layernorm,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("GPU pre-attn norm L{layer_idx}: {e}"))?;
                *total_dispatches += 1;

                // -- QKV projections (CONCURRENT: all read norm_out, write separate buffers) --
                // ONE barrier after norm (which wrote norm_out), then all 3 projections
                // dispatch without barriers between them — they share reads and have disjoint writes.
                session.barrier_between(
                    &[&self.activations.norm_out],
                    &[&self.activations.attn_q, &self.activations.attn_k, &self.activations.attn_v],
                );
                // ADR-028 iter-210: SKIP_ATTN_QKV bisect — skip Q/K/V
                // qmatmul dispatches.  Concurrent ops; their max time
                // is the sequential cost on critical path.  Garbage
                // attention output downstream.
                if !INVESTIGATION_ENV.skip_attn_qkv {
                    dispatch_qmatmul(session, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].attn.q_proj, &self.activations.attn_q, 1,
                        crate::quantize::imatrix::ImatrixHint::Layered { tag: "attn_q", layer: layer_idx })?;
                    *total_dispatches += 1;
                    // Per-dispatch range annotation for the reorder pass. The
                    // single barrier_between above only annotates the first
                    // dispatch; concurrent K and V need their own ranges.
                    dispatch_qmatmul(session, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].attn.k_proj, &self.activations.attn_k, 1,
                        crate::quantize::imatrix::ImatrixHint::Layered { tag: "attn_k", layer: layer_idx })?;
                    session.track_dispatch(&[&self.activations.norm_out], &[&self.activations.attn_k]);
                    *total_dispatches += 1;
                }
                let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
                if !v_is_k && !INVESTIGATION_ENV.skip_attn_qkv {
                    dispatch_qmatmul(session, reg, dev, &self.activations.norm_out,
                        self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                        &self.activations.attn_v, 1,
                        crate::quantize::imatrix::ImatrixHint::Layered { tag: "attn_v", layer: layer_idx })?;
                    session.track_dispatch(&[&self.activations.norm_out], &[&self.activations.attn_v]);
                    *total_dispatches += 1;
                }

                // -- Fused per-head RMS norm + RoPE on Q and K (CONCURRENT) --
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

                // Fused Q + K norm+RoPE (CONCURRENT: read attn_q/attn_k from QKV proj,
                // write to disjoint attn_q_normed/attn_k_normed). ONE barrier for both.
                session.barrier_between(
                    &[&self.activations.attn_q, &self.activations.attn_k],
                    &[&self.activations.attn_q_normed, &self.activations.attn_k_normed],
                );
                // ADR-028 iter-204: SKIP_HEAD_NORM_ROPE bisect — skip
                // both Q-norm-rope and K-norm-rope dispatches.  Produces
                // garbage SDPA (attn_q_normed/attn_k_normed stale).
                if !INVESTIGATION_ENV.skip_head_norm_rope {
                    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_q,
                        &self.activations.attn_q_normed,
                        Some(&self.layers[layer_idx].attn.q_norm_weight),
                        &self.activations.position,
                        ff_gpu,
                        nh as u32, hd as u32, half_rope,
                        eps, theta,
                    ).map_err(|e| anyhow::anyhow!("fused Q norm+RoPE L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_k,
                        &self.activations.attn_k_normed,
                        Some(&self.layers[layer_idx].attn.k_norm_weight),
                        &self.activations.position,
                        ff_gpu,
                        nkv as u32, hd as u32, half_rope,
                        eps, theta,
                    ).map_err(|e| anyhow::anyhow!("fused K norm+RoPE L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                }

                // GPU V norm
                let hd_norm_params = if is_sliding {
                    &self.activations.norm_params_sliding_hd
                } else {
                    &self.activations.norm_params_global_hd
                };
                // ADR-028 iter-214: SKIP_V_NORM bisect.  V-norm output
                // is consumed passively by KV-copy + SDPA — no control
                // signal confound.
                if v_is_k && !INVESTIGATION_ENV.skip_v_norm {
                    session.barrier_between(
                        &[&self.activations.attn_k],
                        &[&self.activations.attn_v],
                    );
                    dispatch_rms_norm_unit_perhead(
                        session.encoder_mut(), reg, metal_dev,
                        &RmsNormPerHeadArgs {
                            input: &self.activations.attn_k,
                            output: &self.activations.attn_v,
                            params_buf: hd_norm_params,
                            rows: nkv as u32,
                            dim: hd as u32,
                        },
                    )?;
                    *total_dispatches += 1;
                } else if !v_is_k && !INVESTIGATION_ENV.skip_v_norm {
                    session.barrier_between(
                        &[&self.activations.attn_v],
                        &[&self.activations.moe_expert_out],
                    );
                    dispatch_rms_norm_unit_perhead(
                        session.encoder_mut(), reg, metal_dev,
                        &RmsNormPerHeadArgs {
                            input: &self.activations.attn_v,
                            output: &self.activations.moe_expert_out,
                            params_buf: hd_norm_params,
                            rows: nkv as u32,
                            dim: hd as u32,
                        },
                    )?;
                    *total_dispatches += 1;
                }

                let v_src = if v_is_k {
                    &self.activations.attn_v
                } else {
                    &self.activations.moe_expert_out
                };

                // ADR-007 C-2: pre-hadamard_quantize K/V dump (independent-floor oracle inputs).
                // Gate: dump_pre_quant && layer_idx == 0 && kv_seq_len == 23.
                // Fires BEFORE dispatch_hadamard_quantize_kv — captures raw F32 K (attn_k_normed)
                // and V (attn_v or moe_expert_out) at the exact moment before TQ encode.
                // Category-4 read-only diagnostic; no HF2Q_UNSAFE_EXPERIMENTS ack required.
                // Path C F-0.3 generalization: if HF2Q_DUMP_PRE_QUANT_LAYERS or
                // HF2Q_DUMP_PRE_QUANT_POSITIONS is set, fire at every matching
                // (layer, kv_seq_len) pair and write per-(layer, position) files
                // named L{layer:02}_p{pos:04}_{k,v}_pre_quant.f32.bin. Otherwise
                // preserve legacy single-file behavior at L0 / kv_seq_len=23.
                let pre_quant_layers_filter = &INVESTIGATION_ENV.dump_pre_quant_layers;
                let pre_quant_positions_filter = &INVESTIGATION_ENV.dump_pre_quant_positions;
                let pre_quant_extended = !pre_quant_layers_filter.is_empty()
                    || !pre_quant_positions_filter.is_empty();
                let layer_match = if pre_quant_extended {
                    pre_quant_layers_filter.is_empty()
                        || pre_quant_layers_filter.contains(&layer_idx)
                } else {
                    layer_idx == 0
                };
                let pos_match = if pre_quant_extended {
                    pre_quant_positions_filter.is_empty()
                        || pre_quant_positions_filter.contains(&kv_seq_len)
                } else {
                    kv_seq_len == 23
                };
                if INVESTIGATION_ENV.dump_pre_quant && layer_match && pos_match {
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("pre_quant dump re-begin: {e}"))?).finish()
                        .map_err(|e| anyhow::anyhow!("pre_quant dump finish L{layer_idx}: {e}"))?;
                    let dump_dir = &INVESTIGATION_ENV.dump_dir;
                    let pre_quant_dir = format!("{dump_dir}/pre_quant");
                    std::fs::create_dir_all(&pre_quant_dir)
                        .map_err(|e| anyhow::anyhow!("pre_quant mkdir: {e}"))?;

                    // Filenames: legacy = `k_pre_quant.f32.bin`; extended =
                    // `L{layer:02}_p{kv_seq_len:04}_k_pre_quant.f32.bin`.
                    let (k_fname, v_fname, meta_fname) = if pre_quant_extended {
                        (
                            format!("L{:02}_p{:04}_k_pre_quant.f32.bin", layer_idx, kv_seq_len),
                            format!("L{:02}_p{:04}_v_pre_quant.f32.bin", layer_idx, kv_seq_len),
                            format!("L{:02}_p{:04}_meta.json", layer_idx, kv_seq_len),
                        )
                    } else {
                        (
                            "k_pre_quant.f32.bin".to_string(),
                            "v_pre_quant.f32.bin".to_string(),
                            "meta.json".to_string(),
                        )
                    };

                    // K pre-quant [nkv, hd] F32 little-endian
                    {
                        let k_raw: &[f32] = self.activations.attn_k_normed.as_slice()
                            .map_err(|e| anyhow::anyhow!("pre_quant k_normed read: {e}"))?;
                        let n_elems = nkv * hd;
                        let k_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                k_raw.as_ptr() as *const u8,
                                n_elems * std::mem::size_of::<f32>(),
                            )
                        };
                        let kp = format!("{pre_quant_dir}/{k_fname}");
                        std::fs::write(&kp, k_bytes)
                            .map_err(|e| anyhow::anyhow!("write {kp}: {e}"))?;
                        eprintln!("[PRE_QUANT_DUMP] L{layer_idx} p{kv_seq_len} k_pre_quant [{nkv},{hd}] f32 -> {kp}");
                    }

                    // V pre-quant [nkv, hd] F32 little-endian
                    {
                        let v_raw: &[f32] = v_src.as_slice()
                            .map_err(|e| anyhow::anyhow!("pre_quant v_src read: {e}"))?;
                        let n_elems = nkv * hd;
                        let v_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                v_raw.as_ptr() as *const u8,
                                n_elems * std::mem::size_of::<f32>(),
                            )
                        };
                        let vp = format!("{pre_quant_dir}/{v_fname}");
                        std::fs::write(&vp, v_bytes)
                            .map_err(|e| anyhow::anyhow!("write {vp}: {e}"))?;
                        eprintln!("[PRE_QUANT_DUMP] L{layer_idx} p{kv_seq_len} v_pre_quant [{nkv},{hd}] f32 -> {vp}");
                    }

                    // meta.json sidecar with provenance
                    {
                        let cache_pos_at_dump = if kv_is_sliding {
                            (kv_write_pos % kv_capacity) as u32
                        } else {
                            kv_write_pos as u32
                        };
                        let meta = serde_json::json!({
                            "site": "pre_hadamard_quantize_kv",
                            "layer_idx": layer_idx,
                            "kv_seq_len": kv_seq_len,
                            "cache_pos_val": cache_pos_at_dump,
                            "nkv": nkv,
                            "hd": hd,
                            "kv_is_sliding": kv_is_sliding,
                            "k_pre_quant_shape": [nkv, hd],
                            "v_pre_quant_shape": [nkv, hd],
                        });
                        let meta_str = serde_json::to_string_pretty(&meta)
                            .map_err(|e| anyhow::anyhow!("pre_quant meta json: {e}"))?;
                        let mp = format!("{pre_quant_dir}/{meta_fname}");
                        std::fs::write(&mp, meta_str.as_bytes())
                            .map_err(|e| anyhow::anyhow!("write {mp}: {e}"))?;
                        eprintln!("[PRE_QUANT_DUMP] meta -> {mp}");
                    }

                }

                // -- GPU KV cache update: Hadamard-quantize into TQ packed cache (ADR-007) --
                // HF2Q_SKIP_TQ_ENCODE=1: skip for timing bisection (output garbage).
                //
                // ADR-028 iter-485 (Phase 7d / H4): when HF2Q_TQ_FAST_FUSED_KV=1
                // collapse the two consecutive dispatches into one via the
                // Z-dim-split `dispatch_hadamard_quantize_kv_fast_dual`.
                // Byte-identical to the 2-dispatch reference; HF2Q_DEBUG_TQ_RMS
                // path forces the legacy split (probe is single-stream only).
                if !INVESTIGATION_ENV.skip_tq_encode {
                    let cache_pos_val = if kv_is_sliding {
                        (kv_write_pos % kv_capacity) as u32
                    } else {
                        kv_write_pos as u32
                    };
                    session.barrier_between(
                        &[&self.activations.attn_k_normed, v_src],
                        &[&self.kv_caches[layer_idx].k_packed, &self.kv_caches[layer_idx].k_norms,
                          &self.kv_caches[layer_idx].v_packed, &self.kv_caches[layer_idx].v_norms],
                    );
                    if INVESTIGATION_ENV.tq_fast_fused_kv && !INVESTIGATION_ENV.debug_tq_rms {
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_fast_dual(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            v_src,
                            &self.kv_caches[layer_idx].k_packed,
                            &self.kv_caches[layer_idx].v_packed,
                            &self.kv_caches[layer_idx].k_norms,
                            &self.kv_caches[layer_idx].v_norms,
                            nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                            kv_is_sliding,
                            Some(ctx.tq_scale_factor_d512),
                        ).map_err(|e| anyhow::anyhow!("hadamard_quantize KV dual L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                    } else {
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &self.kv_caches[layer_idx].k_packed,
                            &self.kv_caches[layer_idx].k_norms,
                            nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                            kv_is_sliding,
                            Some(ctx.tq_scale_factor_d512),
                            None, // rms_scratch: handled below by HF2Q_DEBUG_TQ_RMS path
                        ).map_err(|e| anyhow::anyhow!("hadamard_quantize K L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                            session.encoder_mut(), reg, metal_dev,
                            v_src,
                            &self.kv_caches[layer_idx].v_packed,
                            &self.kv_caches[layer_idx].v_norms,
                            nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                            kv_is_sliding,
                            Some(ctx.tq_scale_factor_d512),
                            None, // rms_scratch: probe not wired here
                        ).map_err(|e| anyhow::anyhow!("hadamard_quantize V L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                    }
                }

                // iter-24: higher-bit (5/6/8-bit) KV encode into leg_hb_encoded.
                // When HF2Q_TQ_CODEBOOK_BITS=5|6|8, encode K/V to byte-packed HB format
                // for native HB SDPA dispatch via `flash_attn_vec_tq_hb` (which reads
                // `leg_hb_encoded` directly — no F32 shadow-cache round-trip).
                // iter-222 (2026-05-01): the `&& !force_dense_sdpa_on_tq_kv` gate
                // that suppressed this block under iter-34's dense-on-shadow
                // default was deleted along with the iter-34 Leg F branch.
                if ctx.use_native_hb_sdpa && !INVESTIGATION_ENV.skip_tq_encode {
                    // ADR-028 Phase 10c (iter-348): hybrid F16-K + TQ-HB-V
                    // encode path. K is written F32→F16 via the existing
                    // `kv_cache_copy_batch_f32_to_f16` (no Hadamard, no
                    // codebook lookup); V is encoded via the existing
                    // single-buffer `dispatch_hadamard_quantize_kv_hb` path
                    // (legacy 2-dispatch arm reused).
                    //
                    // 2 dispatches/layer/token vs 1 in the dual legacy path
                    // (+30 dispatches/decode-token at gemma4 30L).  Trade-off
                    // documented in ADR-028 §iter-348: the K-side SDPA
                    // throughput gain (Phase 10d) outweighs the encode
                    // overhead; if not, follow-up adds a fused
                    // `kv_copy_f16_quantize_v_dual` kernel.
                    if INVESTIGATION_ENV.hybrid_kv {
                        if let Some(ref hybrid_kv) = self.hybrid_kv {
                            let cache_pos_val = if kv_is_sliding {
                                (kv_write_pos % kv_capacity) as u32
                            } else {
                                kv_write_pos as u32
                            };
                            session.barrier_between(
                                &[&self.activations.attn_k_normed, v_src],
                                &[&hybrid_kv[layer_idx].k,
                                  &hybrid_kv[layer_idx].v_packed, &hybrid_kv[layer_idx].v_norms],
                            );
                            // ADR-029 iter-20 H27: when V is allocated as F16
                            // (HF2Q_FULL_F16_KV=1), write both K and V via a
                            // plain F32→F16 cast — no TQ-HB quantize, no FWHT.
                            // Detect via v_packed dtype (single source of truth
                            // matching the alloc-time selection).
                            if hybrid_kv[layer_idx].v_packed.dtype() == mlx_native::DType::F16 {
                                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16_kv_dual(
                                    session.encoder_mut(), reg, metal_dev,
                                    &self.activations.attn_k_normed,
                                    v_src,
                                    &hybrid_kv[layer_idx].k,
                                    &hybrid_kv[layer_idx].v_packed,
                                    nkv as u32, hd as u32,
                                    hybrid_kv[layer_idx].capacity as u32,
                                    cache_pos_val,
                                ).map_err(|e| anyhow::anyhow!("full F16 KV write L{layer_idx}: {e}"))?;
                                *total_dispatches += 1;
                            } else {
                            // BUG-coherence fix (supersedes ADR-028 Phase 10c.5 / 10e.5):
                            //
                            // Phase 10e.5 (iter-351) switched V quantize to a no-FWHT
                            // variant on the parity hypothesis that V is "approximately
                            // N(0,1) per head" after RMS-norm.  Empirical dump of real
                            // gemma4-APEX-Q5_K_M V activations shows kurtosis up to
                            // 72.88 with max|v| up to 14.63 (4x exceeding the 8-bit
                            // Lloyd-Max codebook range of ±5.07).  20/24 sampled
                            // positions clip — outlier-bearing V channels lose their
                            // magnitude through quantization, attention output drifts,
                            // and greedy decode lands in fixed-point loops on
                            // enumeration prompts.
                            //
                            // Fix: restore Hadamard rotation on V.  FWHT spreads any
                            // outlier across all 256 dims (each becomes ≈outlier/√D),
                            // bringing the post-FWHT distribution well within codebook
                            // range.  K stays F16 raw (Phase 10c speedup retained).
                            // Dispatch shape changes:
                            //   * The fused `kv_copy_kf16_quantize_v_no_fwht` is split
                            //     into a separate F16-K copy + FWHT-V quantize (one
                            //     extra dispatch per layer).
                            //   * SDPA output is now in the FWHT domain → a single
                            //     `fwht_sign_undo` dispatch is added after SDPA
                            //     (also one per layer).
                            // Net: +2 dispatches/layer vs broken Phase 10e.5 path
                            // (~0.5% throughput at gemma4 30L); Phase 10c F16-K and
                            // Q-stays-raw savings are preserved.
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_k_normed,
                                &hybrid_kv[layer_idx].k,
                                nkv as u32, hd as u32,
                                hybrid_kv[layer_idx].capacity as u32,
                                cache_pos_val,
                            ).map_err(|e| anyhow::anyhow!("hybrid F16 K write L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                                session.encoder_mut(), reg, metal_dev,
                                v_src,
                                &hybrid_kv[layer_idx].v_packed,
                                &hybrid_kv[layer_idx].v_norms,
                                nkv as u32, hd as u32,
                                hybrid_kv[layer_idx].capacity as u32,
                                cache_pos_val,
                                hybrid_kv[layer_idx].is_sliding,
                                ctx.tq_scale_factor_d512,
                                ctx.tq_codebook_bits,
                            ).map_err(|e| anyhow::anyhow!("hybrid V FWHT quant L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                            } // closes else-block (legacy TQ-HB V path under hybrid)
                        }
                    } else if let Some(ref leg_hb_enc) = self.leg_hb_encoded {
                        let cache_pos_val = if kv_is_sliding {
                            (kv_write_pos % kv_capacity) as u32
                        } else {
                            kv_write_pos as u32
                        };
                        session.barrier_between(
                            &[&self.activations.attn_k_normed, v_src],
                            &[&leg_hb_enc[layer_idx].k_packed, &leg_hb_enc[layer_idx].k_norms,
                              &leg_hb_enc[layer_idx].v_packed, &leg_hb_enc[layer_idx].v_norms],
                        );
                        // ADR-028 iter-149: fused K+V HB encoder (default-on).
                        // HF2Q_HB_DUAL_LEGACY=1 forces 2-dispatch reference path
                        // for forensic A/B parity audit. Both paths byte-identical
                        // by mlx-native unit test
                        // (`test_hadamard_quantize_kv_hb_dual_byte_identity_d256`).
                        if INVESTIGATION_ENV.hb_dual_legacy {
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_k_normed,
                                &leg_hb_enc[layer_idx].k_packed,
                                &leg_hb_enc[layer_idx].k_norms,
                                nkv as u32, hd as u32,
                                leg_hb_enc[layer_idx].capacity as u32,
                                cache_pos_val,
                                leg_hb_enc[layer_idx].is_sliding,
                                ctx.tq_scale_factor_d512,
                                ctx.tq_codebook_bits,
                            ).map_err(|e| anyhow::anyhow!("hb_quantize K L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                                session.encoder_mut(), reg, metal_dev,
                                v_src,
                                &leg_hb_enc[layer_idx].v_packed,
                                &leg_hb_enc[layer_idx].v_norms,
                                nkv as u32, hd as u32,
                                leg_hb_enc[layer_idx].capacity as u32,
                                cache_pos_val,
                                leg_hb_enc[layer_idx].is_sliding,
                                ctx.tq_scale_factor_d512,
                                ctx.tq_codebook_bits,
                            ).map_err(|e| anyhow::anyhow!("hb_quantize V L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        } else {
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_dual(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_k_normed, v_src,
                                &leg_hb_enc[layer_idx].k_packed, &leg_hb_enc[layer_idx].v_packed,
                                &leg_hb_enc[layer_idx].k_norms,  &leg_hb_enc[layer_idx].v_norms,
                                nkv as u32, hd as u32,
                                leg_hb_enc[layer_idx].capacity as u32,
                                cache_pos_val,
                                leg_hb_enc[layer_idx].is_sliding,
                                ctx.tq_scale_factor_d512,
                                ctx.tq_codebook_bits,
                            ).map_err(|e| anyhow::anyhow!("hb_quantize KV dual L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        }
                    }
                }

                // iter-18 S2A: HF2Q_DEBUG_TQ_RMS — POST-SCALE RMS probe (Codex HIGH-1 fix).
                // Previous iter reported stored blk_norm (pre-scale ~0.06), which was WRONG.
                // This iter reports actual post-scale quantizer-input RMS by:
                //   1. Committing the encode command buffer.
                //   2. Reading back the stored norm from k_norms (blk_norm).
                //   3. Computing the post-scale RMS analytically:
                //      post_scale_rms = scale_factor * blk_norm / blk_norm = scale_factor
                //      (exact: after FWHT_norm, block RMS = blk_norm; scale = inv_blk_norm * sf;
                //       → post_scale_elem_rms = sqrt(mean(e^2)) = sf).
                //   4. Probing via scratch buffer (16 samples/block) for empirical verification.
                //
                // Reports both SLIDING (hd=256) and GLOBAL (hd=512) — spec AC-1 requires both.
                // ADR-005 wave-1 T1.2: read from INVESTIGATION_ENV LazyLock.
                if INVESTIGATION_ENV.debug_tq_rms {
                    // iter-19 A1: Fixed RMS probe (catalog #21 — write ALL EPT samples per lane).
                    // Previous iter-18 bug: scratch=[nkv, norms_per_pos, 16], only 8 values written
                    // for D=256 (EPT=8), rest zeros; host divided by 16 → RMS ≈ sqrt(0.5) * true_RMS.
                    // Fix: scratch=[1_head, head_dim] = 256 elements for D=256 (32 lanes × EPT=8).
                    //      For D=512: 512 elements (32 lanes × EPT=16); blk0=[0..255], blk1=[256..511].
                    //      Host divisor = 256 per block.
                    //
                    // iter-19 A2: RMS band LOCKED at [0.8, 1.2] (catalog #11).
                    // No expected*0.5/expected*2.0 arithmetic; constants are literal.
                    const RMS_BAND_LOW: f32 = 0.8;
                    const RMS_BAND_HIGH: f32 = 1.2;

                    // Commit the encode command buffer.
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS re-begin: {e}"))?).finish()
                        .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS finish L{layer_idx}: {e}"))?;

                    let norms_per_pos = (hd / 256).max(1);
                    // Allocate scratch buffer: [1_head, head_dim] f32 = head_dim elements.
                    // All 32 lanes × EPT elements each = head_dim total samples per block (D=256)
                    // or head_dim total samples covering both blocks (D=512: blk0=[0..255], blk1=[256..511]).
                    let scratch_n = hd; // 256 for D=256, 512 for D=512
                    let mut scratch_buf = dev.alloc_buffer(
                        scratch_n * 4, mlx_native::DType::F32,
                        vec![1, hd],
                    ).map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS alloc scratch L{layer_idx}: {e}"))?;
                    // Zero-initialize scratch.
                    {
                        let scratch_slice: &mut [f32] = scratch_buf.as_mut_slice()
                            .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS scratch zero L{layer_idx}: {e}"))?;
                        scratch_slice.iter_mut().for_each(|v| *v = 0.0);
                    }

                    // Compute actual write position for this token.
                    let actual_pos = if kv_is_sliding {
                        kv_write_pos % kv_capacity
                    } else {
                        kv_write_pos.min(kv_capacity - 1)
                    };
                    // Re-dispatch probe for head=0 only using a fresh command buffer.
                    let probe_kind = if kv_is_sliding { "sliding" } else { "global" };
                    let mut sp = exec.begin()
                        .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS probe begin L{layer_idx}: {e}"))?;
                    mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                        sp.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_k_normed,
                        &self.kv_caches[layer_idx].k_packed,
                        &self.kv_caches[layer_idx].k_norms,
                        1u32, // probe head=0 only
                        hd as u32, kv_capacity as u32, actual_pos as u32,
                        kv_is_sliding,
                        Some(ctx.tq_scale_factor_d512),
                        Some(&scratch_buf),
                    ).map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS probe dispatch L{layer_idx}: {e}"))?;
                    sp.finish()
                        .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS probe finish L{layer_idx}: {e}"))?;

                    // Read back scratch: [1 head, head_dim] f32 — ALL samples written.
                    // D=256: 256 samples (1 block). D=512: 512 samples (2 blocks).
                    let scratch_raw: &[f32] = scratch_buf.as_slice()
                        .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS scratch read L{layer_idx}: {e}"))?;

                    for blk in 0..norms_per_pos {
                        // Each block = 256 consecutive elements in scratch.
                        // D=256: blk=0, offset=0, 256 elements.
                        // D=512: blk=0 offset=0 (elements 0..255); blk=1 offset=256 (elements 256..511).
                        let blk_start = blk * 256;
                        let blk_end = (blk_start + 256).min(scratch_raw.len());
                        let samples: &[f32] = &scratch_raw[blk_start..blk_end];
                        // Compute RMS: divide by 256 (full block sample count).
                        let rms = if samples.len() == 256 {
                            let sum_sq: f32 = samples.iter().map(|v| v * v).sum();
                            (sum_sq / 256.0_f32).sqrt()
                        } else {
                            // Partial block (shouldn't happen, but guard):
                            let sum_sq: f32 = samples.iter().map(|v| v * v).sum();
                            if samples.is_empty() { 0.0 } else { (sum_sq / samples.len() as f32).sqrt() }
                        };
                        // iter-19 A2: band LOCKED at [0.8, 1.2] (catalog #11).
                        // This is the spec band for bare scale_factor=1.0 which is the iter-16 control.
                        // Only bare is valid (iter-16 result); sqrt256/sqrt512 are FALSIFIED (iter-16/18).
                        let status = if rms >= RMS_BAND_LOW && rms <= RMS_BAND_HIGH { "PASS" } else { "FAIL" };
                        // 2026-05-16 Gate-H investigation: RMS alone doesn't constrain
                        // kurtosis / outliers / shape — both Gaussian-N(0,1) and bimodal
                        // distributions can have RMS=1.0.  Add max-abs + percentile
                        // breakdown so we can see how far the actual distribution sits
                        // from N(0,1), where 8-bit codebook range is ±5.07σ.
                        let max_abs: f32 = samples.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
                        let mut sorted: Vec<f32> = samples.iter().map(|v| v.abs()).collect();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let p50 = sorted[sorted.len() / 2];
                        let p90 = sorted[(sorted.len() as f64 * 0.90) as usize];
                        let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];
                        // Kurtosis estimator (excess kurtosis: 0 for Gaussian, +3 for Laplace,
                        // higher for heavier-tail).  Uses sample fourth-moment / variance² − 3.
                        let mu: f32 = samples.iter().copied().sum::<f32>() / samples.len() as f32;
                        let m2: f64 = samples.iter().map(|&v| ((v - mu) as f64).powi(2)).sum::<f64>() / samples.len() as f64;
                        let m4: f64 = samples.iter().map(|&v| ((v - mu) as f64).powi(4)).sum::<f64>() / samples.len() as f64;
                        let excess_kurt = if m2 > 1e-12 { (m4 / (m2 * m2)) - 3.0 } else { 0.0 };
                        // Count clipped samples (beyond codebook range ±5.07).
                        let n_clipped: usize = samples.iter().filter(|&&v| v.abs() > 5.0652659).count();
                        eprintln!(
                            "[HF2Q_DEBUG_TQ_RMS] layer={layer_idx} kind={probe_kind} head=0 \
                             blk={blk} rms={rms:.4} max_abs={max_abs:.3} \
                             p50_abs={p50:.3} p90_abs={p90:.3} p99_abs={p99:.3} \
                             ex_kurt={excess_kurt:.2} clipped={n_clipped}/256 \
                             status={status} (band=[{RMS_BAND_LOW:.3},{RMS_BAND_HIGH:.3}])"
                        );
                    }

                }

                // C-1-unlock: post-hadamard_quantize pre-SDPA dump (decode step 1, layer 0).
                // Gate: dump_tq_state && layer_idx == 0 && kv_seq_len == 23 (one decode token
                // has been written into slot 22 of the TQ ring buffer).
                // Dumps full-capacity packed K/V + norms + Q (pre-FWHT) to post_quant subdir.
                if INVESTIGATION_ENV.dump_tq_state && layer_idx == 0 && kv_seq_len == 23 {
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("post_quant dump re-begin: {e}"))?).finish()
                        .map_err(|e| anyhow::anyhow!("post_quant dump finish L{layer_idx}: {e}"))?;
                    let hd_half = hd / 2;
                    let dump_dir = &INVESTIGATION_ENV.dump_dir;
                    let post_quant_dir = format!("{dump_dir}/post_quant");
                    std::fs::create_dir_all(&post_quant_dir)
                        .map_err(|e| anyhow::anyhow!("post_quant mkdir: {e}"))?;

                    // k_packed_post_quant.u8.bin — full [nkv, kv_capacity, hd/2] u8
                    {
                        let k_raw: &[u8] = self.kv_caches[layer_idx].k_packed.as_slice()
                            .map_err(|e| anyhow::anyhow!("post_quant k_packed read: {e}"))?;
                        let n_bytes = nkv * kv_capacity * hd_half;
                        let kp = format!("{post_quant_dir}/k_packed_post_quant.u8.bin");
                        std::fs::write(&kp, &k_raw[..n_bytes])
                            .map_err(|e| anyhow::anyhow!("write {kp}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] k_packed [{nkv},{kv_capacity},{hd_half}] u8 -> {kp}");
                    }

                    // v_packed_post_quant.u8.bin — full [nkv, kv_capacity, hd/2] u8
                    {
                        let v_raw: &[u8] = self.kv_caches[layer_idx].v_packed.as_slice()
                            .map_err(|e| anyhow::anyhow!("post_quant v_packed read: {e}"))?;
                        let n_bytes = nkv * kv_capacity * hd_half;
                        let vp = format!("{post_quant_dir}/v_packed_post_quant.u8.bin");
                        std::fs::write(&vp, &v_raw[..n_bytes])
                            .map_err(|e| anyhow::anyhow!("write {vp}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] v_packed [{nkv},{kv_capacity},{hd_half}] u8 -> {vp}");
                    }

                    // k_norms_post_quant.f32.bin — full [nkv, kv_capacity] f32
                    {
                        let kn_raw: &[f32] = self.kv_caches[layer_idx].k_norms.as_slice()
                            .map_err(|e| anyhow::anyhow!("post_quant k_norms read: {e}"))?;
                        let n_elems = nkv * kv_capacity;
                        let kn_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                kn_raw.as_ptr() as *const u8,
                                n_elems * std::mem::size_of::<f32>(),
                            )
                        };
                        let kn = format!("{post_quant_dir}/k_norms_post_quant.f32.bin");
                        std::fs::write(&kn, kn_bytes)
                            .map_err(|e| anyhow::anyhow!("write {kn}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] k_norms [{nkv},{kv_capacity}] f32 -> {kn}");
                    }

                    // v_norms_post_quant.f32.bin — full [nkv, kv_capacity] f32
                    {
                        let vn_raw: &[f32] = self.kv_caches[layer_idx].v_norms.as_slice()
                            .map_err(|e| anyhow::anyhow!("post_quant v_norms read: {e}"))?;
                        let n_elems = nkv * kv_capacity;
                        let vn_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                vn_raw.as_ptr() as *const u8,
                                n_elems * std::mem::size_of::<f32>(),
                            )
                        };
                        let vn = format!("{post_quant_dir}/v_norms_post_quant.f32.bin");
                        std::fs::write(&vn, vn_bytes)
                            .map_err(|e| anyhow::anyhow!("write {vn}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] v_norms [{nkv},{kv_capacity}] f32 -> {vn}");
                    }

                    // q_natural.f32.bin — Q pre-FWHT, shape [nh, hd] f32
                    {
                        let q_raw: &[f32] = self.activations.attn_q_normed.as_slice()
                            .map_err(|e| anyhow::anyhow!("post_quant q_normed read: {e}"))?;
                        let n_elems = nh * hd;
                        let q_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                q_raw.as_ptr() as *const u8,
                                n_elems * std::mem::size_of::<f32>(),
                            )
                        };
                        let qp = format!("{post_quant_dir}/q_natural.f32.bin");
                        std::fs::write(&qp, q_bytes)
                            .map_err(|e| anyhow::anyhow!("write {qp}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] q_natural [{nh},{hd}] f32 -> {qp}");
                    }

                    // meta_post_quant.json — production call-site params + provenance
                    {
                        // iter-25 Subtask B fix: use corrected ring_start formula (oldest slot).
                        let ring_start = if kv_is_sliding && kv_seq_len >= kv_capacity {
                            ((kv_write_pos + 1) % kv_capacity) as u32
                        } else {
                            0u32
                        };
                        let commit_sha = option_env!("GIT_COMMIT_SHA")
                            .unwrap_or("03bea75071a8b0fd43a47f1101a832e23317e429");
                        let meta = serde_json::json!({
                            "site": "post_hadamard_quantize_pre_sdpa",
                            "layer_idx": layer_idx,
                            "seq_pos": seq_pos,
                            "kv_seq_len": kv_seq_len,
                            "kv_capacity": kv_capacity,
                            "kv_write_pos": kv_write_pos,
                            "nkv": nkv,
                            "nh": nh,
                            "hd": hd,
                            "hd_half": hd_half,
                            "kv_is_sliding": kv_is_sliding,
                            "mask_type": if is_sliding { 2u32 } else { 1u32 },
                            "sliding_window": if is_sliding { self.sliding_window as u32 } else { 0u32 },
                            "ring_start": ring_start,
                            "k_packed_shape": [nkv, kv_capacity, hd_half],
                            "v_packed_shape": [nkv, kv_capacity, hd_half],
                            "k_norms_shape": [nkv, kv_capacity],
                            "v_norms_shape": [nkv, kv_capacity],
                            "q_natural_shape": [nh, hd],
                            "commit_sha": commit_sha,
                        });
                        let meta_str = serde_json::to_string_pretty(&meta)
                            .map_err(|e| anyhow::anyhow!("post_quant meta json: {e}"))?;
                        let mp = format!("{post_quant_dir}/meta_post_quant.json");
                        std::fs::write(&mp, meta_str.as_bytes())
                            .map_err(|e| anyhow::anyhow!("write {mp}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] meta -> {mp}");
                    }

                }

                // ADR-009 Phase 3A: dump Q,K,V before SDPA for the detail layer,
                // or ALL layers when HF2Q_DUMP_ALL_CACHE=1
                // W39 iter-112b: consult per-instance override first.
                let dump_all_cache = ctx.dump_all_cache_eff;
                if dump_layers && (dump_detail_layer == Some(layer_idx) || dump_all_cache) {
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("dump QKV re-begin L{layer_idx}: {e}"))?).finish()
                        .map_err(|e| anyhow::anyhow!("dump QKV finish L{layer_idx}: {e}"))?;
                    let dir_override = self.dump_dir_override.as_deref();
                    dumps::dump_f32_to(&self.activations.attn_q_normed, nh * hd,
                        "q_normed", Some(layer_idx), seq_pos, dir_override)?;
                    dumps::dump_f32_to(&self.activations.attn_k_normed, nkv * hd,
                        "k_normed", Some(layer_idx), seq_pos, dir_override)?;
                    dumps::dump_f32_to(v_src, nkv * hd,
                        "v_normed", Some(layer_idx), seq_pos, dir_override)?;
                }

                // -- SDPA: TQ (default) or DENSE opt-out --
                // ADR-007 CLOSED 2026-04-24, post-close correction 2026-04-24:
                // TQ-8-bit is the DEFAULT decode path (2× memory savings vs F16;
                // Gate A cosine 0.9998 exceeds TurboQuant paper 0.999; Gate B argmax
                // divergence 0.8% exceeds <1%; Gate C PPL delta 1.24% / 0.017 absolute
                // meets KIVI + KVQuant + AmesianX + vLLM + TurboQuant shippability gates).
                // Rationale: "TQ should be default if it is better" — user feedback on
                // iter-28's overly-conservative flip to dense. Byte-exact vs llama.cpp is
                // still achievable via HF2Q_USE_DENSE=1 (sourdough_gate.sh sets this).
                //
                // HF2Q_LAYER_POLICY values:
                //   unset OR "tq_all"         = DEFAULT: full TQ decode (8-bit native HB SDPA)
                //   "dense_all"               = dense everywhere (byte-exact vs llama.cpp)
                //   "tq_slide_dense_global"   = TQ for sliding layers, dense for global
                //   "dense_slide_tq_global"   = dense for sliding, TQ for global
                //
                // HF2Q_USE_DENSE=1 forces dense_all (explicit opt-out for byte-exact gates).
                //
                // W12 iter-108a blocker #3 (ADR-007 Gate H): per-call regime override
                // consulted BEFORE the env vars.  When the regime is `DecodeRegime::Default`
                // (the default for every existing call site), this path is bit-identical
                // to today's env-var-only logic.  When the regime is `ForceDense` /
                // `ForceTq`, the env vars are skipped entirely so a single process
                // can run both regimes against the same prompt without subprocess
                // fork.  The four-gate lockstep contract (W9's mapping —
                // forward_mlx.rs:1100/1234/<this gate>, forward_prefill.rs:330) is
                // preserved: only this SDPA-reader gate is overridden; the codebook-
                // bits gates remain env-driven because the codebook width is a
                // representation choice consistent across both regimes.  See
                // `MlxModelWeights::set_decode_regime` for the contract.
                // iter-108a-fix (W15, 2026-04-25): when `gate_h_inactive` is true
                // (the default — no Gate H env hooks armed and regime is Default),
                // skip the regime-match arm entirely and use the pre-iter-108a
                // env-var-only path verbatim. This restores the byte-identical
                // hot-path branch sequence to the iter-108a base commit
                // (`1bcf172`). When Gate H is active, the regime override is
                // consulted as before. Per-layer = ~30× per token; the saved
                // enum-field load + match across the layer loop is the bulk of
                // the W14b 5.6% regression.
                // ADR-005 wave-1 T1.2: HF2Q_USE_DENSE and HF2Q_LAYER_POLICY read from
                // INVESTIGATION_ENV LazyLock (parsed once at process start) instead of
                // calling std::env::var per-token per-layer. Behavior is bit-identical:
                // `use_dense` mirrors `== Ok("1")`; `layer_policy.as_deref()` mirrors
                // `as_deref()` on the Result, with None mapping to the former Err(_) arm.
                // iter-222 (ADR-005 closure, 2026-05-01): the iter-50
                // `None if force_dense_sdpa_on_tq_kv => true` arms that routed
                // iter-34's default to Branch A (dense_kvs) were deleted — see
                // file-level iter-222 closure note. Default now flows through
                // the inline-fused TQ-native path below as in pre-iter-34.
                let use_dense_sdpa = if self.dense_kvs.is_none() {
                    false
                } else if self.gate_h_inactive {
                    // Pre-iter-108a path: LazyLock-cached env values. Bit-identical to base.
                    if INVESTIGATION_ENV.use_dense {
                        true
                    } else {
                        match INVESTIGATION_ENV.layer_policy.as_deref() {
                            Some("dense_all") => true,
                            Some("tq_all") | None => false,
                            Some("tq_slide_dense_global") => !kv_is_sliding,
                            Some("dense_slide_tq_global") => kv_is_sliding,
                            Some(other) => {
                                static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                    eprintln!("[HF2Q_LAYER_POLICY] unknown value {:?}; defaulting to tq_all", other);
                                }
                                false
                            }
                        }
                    }
                } else {
                    match self.decode_regime {
                        DecodeRegime::ForceDense => true,
                        DecodeRegime::ForceTq => false,
                        DecodeRegime::Default => {
                            if INVESTIGATION_ENV.use_dense {
                                true
                            } else {
                                match INVESTIGATION_ENV.layer_policy.as_deref() {
                                    Some("dense_all") => true,
                                    Some("tq_all") | None => false,
                                    Some("tq_slide_dense_global") => !kv_is_sliding,
                                    Some("dense_slide_tq_global") => kv_is_sliding,
                                    Some(other) => {
                                        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                                        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                            eprintln!("[HF2Q_LAYER_POLICY] unknown value {:?}; defaulting to tq_all", other);
                                        }
                                        false
                                    }
                                }
                            }
                        }
                    }
                };

                if use_dense_sdpa {
                    // -- Dense decode SDPA (ADR-009 Track 3) --
                    // Copy this position's K,V into dense KV buffers.
                    // Uses F16 cast kernel when dense_kvs are F16, else F32 copy.
                    let dense_kvs = self.dense_kvs.as_ref().unwrap();
                    let dense_cap = dense_kvs[layer_idx].capacity;
                    let layer_is_ring = dense_kvs[layer_idx].is_sliding;
                    // ADR-017 Phase E.a iter-3.6: when LONG_RESUME is on
                    // and layer is sliding, the buffer is LINEAR (no
                    // wrap); decode writes go to slot=seq_pos. When OFF
                    // (default), sliding wraps via slot=seq_pos%cap.
                    let kv_lcp_long_resume_for_write = INVESTIGATION_ENV.kv_lcp_long_resume
                        && INVESTIGATION_ENV.kv_lcp_resume
                        && INVESTIGATION_ENV.use_dense;
                    let write_slot = if layer_is_ring && !kv_lcp_long_resume_for_write {
                        (seq_pos % dense_cap) as u32
                    } else {
                        seq_pos as u32
                    };
                    let kv_is_f16 = dense_kvs[layer_idx].k.dtype() == mlx_native::DType::F16;
                    session.barrier_between(
                        &[&self.activations.attn_k_normed, v_src],
                        &[&dense_kvs[layer_idx].k, &dense_kvs[layer_idx].v],
                    );
                    // ADR-028 iter-146: fused K+V single-position copy (default-on).
                    // HF2Q_KV_DUAL_LEGACY=1 forces 2-dispatch reference path for
                    // forensic A/B parity audit; matches W-5b.10/14 sunset cadence.
                    let use_legacy_2dispatch = INVESTIGATION_ENV.kv_dual_legacy;
                    if kv_is_f16 {
                        if use_legacy_2dispatch {
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_k_normed,
                                &dense_kvs[layer_idx].k,
                                nkv as u32, hd as u32,
                                dense_cap as u32, write_slot,
                            ).map_err(|e| anyhow::anyhow!("decode F16 K copy L{layer_idx}: {e}"))?;
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                                session.encoder_mut(), reg, metal_dev,
                                v_src,
                                &dense_kvs[layer_idx].v,
                                nkv as u32, hd as u32,
                                dense_cap as u32, write_slot,
                            ).map_err(|e| anyhow::anyhow!("decode F16 V copy L{layer_idx}: {e}"))?;
                            *total_dispatches += 2;
                        } else {
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16_kv_dual(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_k_normed, v_src,
                                &dense_kvs[layer_idx].k, &dense_kvs[layer_idx].v,
                                nkv as u32, hd as u32,
                                dense_cap as u32, write_slot,
                            ).map_err(|e| anyhow::anyhow!("decode F16 KV dual copy L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        }
                    } else if use_legacy_2dispatch {
                        // F32 batched: one dispatch per K, one per V (all heads at once).
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &dense_kvs[layer_idx].k,
                            nkv as u32, hd as u32,
                            dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("decode F32 K batch copy L{layer_idx}: {e}"))?;
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            session.encoder_mut(), reg, metal_dev,
                            v_src,
                            &dense_kvs[layer_idx].v,
                            nkv as u32, hd as u32,
                            dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("decode F32 V batch copy L{layer_idx}: {e}"))?;
                        *total_dispatches += 2;
                    } else {
                        // ADR-028 iter-146: fused F32 K+V into single dispatch.
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_kv_dual(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed, v_src,
                            &dense_kvs[layer_idx].k, &dense_kvs[layer_idx].v,
                            nkv as u32, hd as u32,
                            dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("decode F32 KV dual copy L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                    }

                    // ADR-009 Phase 3A: dump full cached K/V for the detail layer,
                    // or ALL layers when HF2Q_DUMP_ALL_CACHE=1
                    // W39 iter-112b: consult per-instance override first.
                    let dump_all_cache = ctx.dump_all_cache_eff;
                    if dump_layers && (dump_detail_layer == Some(layer_idx) || dump_all_cache) {
                        std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("dump cache re-begin L{layer_idx}: {e}"))?).finish()
                            .map_err(|e| anyhow::anyhow!("dump cache finish L{layer_idx}: {e}"))?;
                        // W39 iter-112b: per-instance dump dir override; falls back
                        // to INVESTIGATION_ENV.dump_dir when unset.
                        let dump_dir_override = self
                            .dump_dir_override
                            .as_ref()
                            .map(|p| p.to_string_lossy().into_owned());
                        let dump_dir: &str = dump_dir_override
                            .as_deref()
                            .unwrap_or(&INVESTIGATION_ENV.dump_dir);
                        // Pack [nkv, kv_seq_len, hd] into a tight F32 buffer for comparison
                        let valid_len = kv_seq_len;
                        let mut k_valid = vec![0.0f32; nkv * valid_len * hd];
                        let mut v_valid = vec![0.0f32; nkv * valid_len * hd];
                        if kv_is_f16 {
                            // Read F16 bits and convert to F32
                            let k_raw: &[u16] = dense_kvs[layer_idx].k.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump cache K L{layer_idx}: {e}"))?;
                            let v_raw: &[u16] = dense_kvs[layer_idx].v.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump cache V L{layer_idx}: {e}"))?;
                            for h in 0..nkv {
                                for p in 0..valid_len {
                                    let src = h * dense_cap * hd + p * hd;
                                    let dst = h * valid_len * hd + p * hd;
                                    for i in 0..hd {
                                        k_valid[dst+i] = half::f16::from_bits(k_raw[src+i]).to_f32();
                                        v_valid[dst+i] = half::f16::from_bits(v_raw[src+i]).to_f32();
                                    }
                                }
                            }
                        } else {
                            let k_data: &[f32] = dense_kvs[layer_idx].k.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump cache K L{layer_idx}: {e}"))?;
                            let v_data: &[f32] = dense_kvs[layer_idx].v.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump cache V L{layer_idx}: {e}"))?;
                            for h in 0..nkv {
                                for p in 0..valid_len {
                                    let src = h * dense_cap * hd + p * hd;
                                    let dst = h * valid_len * hd + p * hd;
                                    k_valid[dst..dst+hd].copy_from_slice(&k_data[src..src+hd]);
                                    v_valid[dst..dst+hd].copy_from_slice(&v_data[src..src+hd]);
                                }
                            }
                        }
                        let k_path = format!("{dump_dir}/hf2q_cache_k_layer{layer_idx:02}_pos{seq_pos}.bin");
                        let v_path = format!("{dump_dir}/hf2q_cache_v_layer{layer_idx:02}_pos{seq_pos}.bin");
                        let k_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(k_valid.as_ptr() as *const u8, k_valid.len() * 4)
                        };
                        let v_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(v_valid.as_ptr() as *const u8, v_valid.len() * 4)
                        };
                        std::fs::write(&k_path, k_bytes)
                            .map_err(|e| anyhow::anyhow!("write {k_path}: {e}"))?;
                        std::fs::write(&v_path, v_bytes)
                            .map_err(|e| anyhow::anyhow!("write {v_path}: {e}"))?;
                        let dtype_str = if kv_is_f16 { "F16→F32" } else { "F32" };
                        eprintln!("[DUMP] cache K layer {layer_idx:02} [{nkv},{valid_len},{hd}] {dtype_str} -> {k_path}");
                        eprintln!("[DUMP] cache V layer {layer_idx:02} [{nkv},{valid_len},{hd}] {dtype_str} -> {v_path}");
                    }

                    // Dense flash_attn_vec
                    let dense_sdpa_tmp = self.dense_sdpa_tmp.as_ref().unwrap();
                    session.barrier_between(
                        &[&self.activations.attn_q_normed,
                          &dense_kvs[layer_idx].k, &dense_kvs[layer_idx].v],
                        &[&self.activations.sdpa_out],
                    );
                    // kv_seq_len for the dense cache:
                    //   - Sliding (ring): min(seq_pos+1, capacity). The ring holds
                    //     at most `capacity=sliding_window` entries — the causal
                    //     mask then attends to exactly the populated slots.
                    //     Attention is permutation-invariant over cached K,V
                    //     (RoPE is baked in pre-cache), so slot order doesn't
                    //     matter for correctness.
                    //   - Global (linear): seq_pos + 1.
                    // In ring mode we use mask_type=1 (causal) since the ring
                    // itself applies the sliding-window constraint — the
                    // kernel's sliding-window mask would incorrectly mask slots
                    // whose logical positions don't equal their slot index.
                    // ADR-017 Phase E.a iter-3.6: when LONG_RESUME is on
                    // and layer is sliding, the buffer is LINEAR (cap >
                    // sliding_window, slot index = logical position),
                    // and the kernel masks via mask_type=2 +
                    // sliding_window=sw. When OFF (default), behavior is
                    // byte-identical to pre-iter-3.6 (ring + mask_type=1).
                    let kv_lcp_long_resume = INVESTIGATION_ENV.kv_lcp_long_resume
                        && INVESTIGATION_ENV.kv_lcp_resume
                        && INVESTIGATION_ENV.use_dense;
                    let use_linear_sliding = layer_is_ring && kv_lcp_long_resume;
                    let dense_kv_seq_len = if layer_is_ring && !use_linear_sliding {
                        ((seq_pos + 1).min(dense_cap)) as u32
                    } else {
                        (seq_pos + 1) as u32
                    };
                    let (mask_type_val, sliding_window_val) = if use_linear_sliding {
                        let model_sw = self.sliding_window.max(1);
                        (2u32, model_sw as u32)
                    } else {
                        (1u32, 0u32)
                    };
                    let p = mlx_native::ops::flash_attn_vec::FlashAttnVecParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: dense_kv_seq_len,
                        kv_capacity: dense_cap as u32,
                        scale: 1.0,
                        mask_type: mask_type_val,
                        sliding_window: sliding_window_val,
                        softcap: 0.0,
                        // ADR-034 task #89: decode path = single query.
                        q_seq_len:
                            mlx_native::ops::flash_attn_vec::FlashAttnVecParams::DEFAULT_Q_SEQ_LEN,
                    };
                    mlx_native::ops::flash_attn_vec::flash_attn_vec(
                        session.encoder_mut(), reg, dev,
                        &self.activations.attn_q_normed,
                        &dense_kvs[layer_idx].k,
                        &dense_kvs[layer_idx].v,
                        &self.activations.sdpa_out,
                        dense_sdpa_tmp,
                        &p,
                    ).map_err(|e| anyhow::anyhow!("dense flash_attn_vec L{layer_idx}: {e}"))?;
                    *total_dispatches += 2; // main + reduce
                // iter-222 (ADR-005 closure, 2026-05-01): the iter-20 Leg F /
                // iter-34 dense-on-shadow decode branch was deleted entirely
                // here (~170 LOC) — see file-level iter-222 closure note above
                // the (now-deleted) `dense_sdpa_on_tq_kv_enabled()` site for
                // rationale (Gate H regression + peer-impl research +
                // "no fallback" mantra). TQ-regime SDPA now flows through the
                // inline-fused `flash_attn_vec_tq_hb` (cb_bits>=5, default 8)
                // or `flash_attn_vec_tq` (cb_bits=4 legacy) branches below.
                } else if !INVESTIGATION_ENV.skip_tq_sdpa && ctx.use_native_hb_sdpa {
                    // ADR-028 Phase 10c (iter-348): hybrid path SDPA dispatcher
                    // not yet wired (Phase 10e).  When the user enables
                    // `HF2Q_HYBRID_KV=1` without 10e+10d (kernel) landed,
                    // hard-fail loud-not-silent rather than read stale F32
                    // SDPA-out from a previous decode token.  This is the
                    // intentional partial-stack failure mode signalled in
                    // Phase 10b's design (iter-347).
                    //
                    // ADR-028 Phase 10e (iter-350): live wiring lands here.
                    // K is stored F16 raw → Q stays raw (NO FWHT-pre dispatch),
                    // SDPA runs in raw domain (NO FWHT-undo dispatch), V comes
                    // from hybrid_kv[layer_idx].{v_packed, v_norms} (TQ-HB-encoded
                    // by the Phase 10c encode site at line ~3074).  Saves 60
                    // FWHT dispatches/decode-token at gemma4 30L on top of the
                    // K-side codebook elimination.
                    if INVESTIGATION_ENV.hybrid_kv {
                        // ADR-028 Phase 10e (iter-350): hybrid F16-K + TQ-HB-V SDPA.
                        //
                        // FWHT chain reasoning (Phase 10e initial wiring iter-350
                        // kept FWHT-undo because V was FWHT-rotated; Phase 10e.5
                        // iter-351 swapped V-encode to `kv_quantize_v_no_fwht`,
                        // which stores raw V — so output is now in raw domain):
                        //   * K stored RAW F16 → Q stays raw, NO fwht_sign_premult.
                        //   * V stored RAW (Phase 10e.5 V-encode dispatcher) → SDPA
                        //     output = softmax × V_raw → output IS raw → NO
                        //     fwht_sign_undo dispatch needed.
                        //
                        // Net dispatch saving: 60 dispatches/decode-token at gemma4
                        // 30L (the entire FWHT chain in attention is eliminated).
                        let hybrid_kv = self.hybrid_kv.as_ref().ok_or_else(|| anyhow::anyhow!(
                            "HF2Q_HYBRID_KV=1 but hybrid_kv buffers not allocated \
                             (gemma4 decode L{layer_idx}); should have been allocated \
                             by Phase 10c lazy-alloc gate. See ADR-028 §iter-350."
                        ))?;
                        let hb_cap = hybrid_kv[layer_idx].capacity;
                        let hb_is_ring = hybrid_kv[layer_idx].is_sliding;
                        let hb_kv_seq_len = if hb_is_ring {
                            ((kv_write_pos + 1).min(hb_cap)) as u32
                        } else {
                            (kv_write_pos + 1) as u32
                        };
                        let ring_start_hb = if hb_is_ring && hb_kv_seq_len as usize >= hb_cap {
                            ((kv_write_pos + 1) % hb_cap) as u32
                        } else {
                            0u32
                        };
                        session.barrier_between(
                            &[&self.activations.attn_q_normed,
                              &hybrid_kv[layer_idx].k,
                              &hybrid_kv[layer_idx].v_packed,
                              &hybrid_kv[layer_idx].v_norms],
                            &[&self.activations.sdpa_out],
                        );
                        let p_hyb = mlx_native::ops::flash_attn_vec_hybrid::FlashAttnVecTqHbParams {
                            num_heads: nh as u32,
                            num_kv_heads: nkv as u32,
                            head_dim: hd as u32,
                            kv_seq_len: hb_kv_seq_len,
                            kv_capacity: hb_cap as u32,
                            scale: 1.0,
                            mask_type: if is_sliding { 2 } else { 1 },
                            sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                            softcap: 0.0,
                            ring_start: ring_start_hb,
                            scale_factor_d512: ctx.tq_scale_factor_d512,
                            codebook_bits: ctx.tq_codebook_bits,
                            // Hybrid kernel: caller passes RAW Q (no rotation).
                            // fuse_fwht_pre=0 → kernel reads Q as-is.
                            fuse_fwht_pre: 0,
                            nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(hb_kv_seq_len),
                        };
                        // HF2Q_FA_PEER_PORT*: dispatch peer-port kernel variant instead of hybrid.
                        // Preconditions: head_dim==256, K dtype==F16, V dtype==F16.
                        //
                        // iter-137 — two variants:
                        //   HF2Q_FA_PEER_PORT       = NWG=1 verbatim port (iter-126).
                        //                             Falsified at tg5000 (-25%) because peer's
                        //                             actual runtime uses NWG=32 (iter-133 root
                        //                             cause). Kept for A/B + documentation;
                        //                             additionally gated on is_sliding so
                        //                             full-attn fallthrough to HYBRID.
                        //   HF2Q_FA_PEER_PORT_NWG32 = NWG=32 + reduce-kernel port (iters 134-137).
                        //                             Matches peer's actual runtime dispatch.
                        //                             Validated WIN +1.8-3.1pp at tg100/tg2000/tg5000
                        //                             vs HYBRID at PORT's f16-V regime (iter-138/140).
                        //                             Default-flipped ON iter-149 per operator
                        //                             approval: "best possible outcome for users —
                        //                             if coherent + TQ still enabled + marginally
                        //                             faster, of course default."
                        //                             Reuses existing sdpa_tmp buffer (identical
                        //                             size formula nrows*32*(dv+2)*4).
                        //
                        // PORT_NWG32 default ON; opt out via HF2Q_FA_PEER_PORT_NWG32=0.
                        // PORT (NWG=1, falsified) default OFF — explicit HF2Q_FA_PEER_PORT=1 only.
                        // The precondition `v_packed.dtype()==F16` means PORT_NWG32 ONLY fires when
                        // TQ-HB-V is bypassed (HF2Q_FULL_F16_KV=1 or otherwise F16-V regime).
                        // With default TQ-HB-V active, PORT_NWG32 gate falls through to hybrid —
                        // zero behavior change. With explicit F16-V request, PORT_NWG32 wins +2pp.
                        static FA_PEER_PORT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let use_peer_port = *FA_PEER_PORT.get_or_init(|| {
                            std::env::var("HF2Q_FA_PEER_PORT").map(|v| v == "1").unwrap_or(false)
                        });
                        static FA_PEER_PORT_NWG32: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let use_peer_port_nwg32 = *FA_PEER_PORT_NWG32.get_or_init(|| {
                            // env_default_true pattern (mirrors HF2Q_Q6K_MV_NR2 iter-326):
                            // unset → ON; "0"/"false"/"off" → OFF; "1"/"true"/"on" → ON.
                            match std::env::var("HF2Q_FA_PEER_PORT_NWG32").ok().as_deref() {
                                None => true,
                                Some(v) if v.eq_ignore_ascii_case("0")
                                    || v.eq_ignore_ascii_case("false")
                                    || v.eq_ignore_ascii_case("off") => false,
                                Some(_) => true,
                            }
                        });

                        if use_peer_port_nwg32
                            && hd == 256
                            && hybrid_kv[layer_idx].k.dtype() == mlx_native::DType::F16
                            && hybrid_kv[layer_idx].v_packed.dtype() == mlx_native::DType::F16
                        {
                            let p_peer = mlx_native::ops::flash_attn_vec_peer_port_f16::FlashAttnVecPeerPortParams {
                                num_heads: nh as u32,
                                num_kv_heads: nkv as u32,
                                head_dim: hd as u32,
                                kv_seq_len: hb_kv_seq_len,
                                kv_capacity: hb_cap as u32,
                                scale: 1.0,
                                mask_type: if is_sliding { 2 } else { 1 },
                                sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                                ring_start: ring_start_hb,
                            };
                            mlx_native::ops::flash_attn_vec_peer_port_f16::flash_attn_vec_peer_port_f16_nwg32(
                                session.encoder_mut(), reg, dev,
                                &self.activations.attn_q_normed,
                                &hybrid_kv[layer_idx].k,
                                &hybrid_kv[layer_idx].v_packed,
                                &self.activations.sdpa_tmp,
                                &self.activations.sdpa_out,
                                &p_peer,
                            ).map_err(|e| anyhow::anyhow!("flash_attn_vec_peer_port_f16_nwg32 L{layer_idx}: {e}"))?;
                            *total_dispatches += 2; // vec + reduce
                        } else if use_peer_port
                            && is_sliding
                            && hd == 256
                            && hybrid_kv[layer_idx].k.dtype() == mlx_native::DType::F16
                            && hybrid_kv[layer_idx].v_packed.dtype() == mlx_native::DType::F16
                        {
                            let p_peer = mlx_native::ops::flash_attn_vec_peer_port_f16::FlashAttnVecPeerPortParams {
                                num_heads: nh as u32,
                                num_kv_heads: nkv as u32,
                                head_dim: hd as u32,
                                kv_seq_len: hb_kv_seq_len,
                                kv_capacity: hb_cap as u32,
                                scale: 1.0,
                                mask_type: if is_sliding { 2 } else { 1 },
                                sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                                ring_start: ring_start_hb,
                            };
                            mlx_native::ops::flash_attn_vec_peer_port_f16::flash_attn_vec_peer_port_f16(
                                session.encoder_mut(), reg, dev,
                                &self.activations.attn_q_normed,
                                &hybrid_kv[layer_idx].k,
                                &hybrid_kv[layer_idx].v_packed,
                                &self.activations.sdpa_out,
                                &p_peer,
                            ).map_err(|e| anyhow::anyhow!("flash_attn_vec_peer_port_f16 L{layer_idx}: {e}"))?;
                            *total_dispatches += 1; // NWG=1: no reduce kernel
                        } else {
                            mlx_native::ops::flash_attn_vec_hybrid::flash_attn_vec_hybrid(
                                session.encoder_mut(), reg, dev,
                                &self.activations.attn_q_normed,
                                &hybrid_kv[layer_idx].k,
                                &hybrid_kv[layer_idx].v_packed,
                                &hybrid_kv[layer_idx].v_norms,
                                &self.activations.sdpa_out,
                                &self.activations.sdpa_tmp,
                                &p_hyb,
                            ).map_err(|e| anyhow::anyhow!("flash_attn_vec_hybrid L{layer_idx}: {e}"))?;
                            *total_dispatches += 2; // main + reduce (conservative)
                        }
                        // BUG-coherence fix (supersedes Phase 10e.5 iter-351):
                        // V is now FWHT-rotated then quantized (see V-encode site
                        // ~line 3724).  SDPA output is therefore in the FWHT domain
                        // and must be inverse-rotated to recover raw values before
                        // feeding o_proj.  Skipped when V is F16 (FULL_F16_KV or
                        // peer-port path) since no FWHT was applied during write.
                        //
                        // Why fwht_sign_undo and not fwht_sign_premult: the V-encode
                        // applies sign-premult + FWHT + /√d.  SDPA produces
                        //   Σ softmax * FWHT(sign*V)/√d = FWHT(sign * Σ softmax*V)/√d.
                        // fwht_sign_undo = (multiply by √d via output buffer scale,
                        // apply FWHT which is self-inverse for normalized H, then
                        // sign-undo) recovers `Σ softmax * V` exactly.  Mirrors the
                        // legacy TQ-HB SDPA caller at L4694.
                        if hybrid_kv[layer_idx].v_packed.dtype() != mlx_native::DType::F16 {
                            session.barrier_between(
                                &[&self.activations.sdpa_out],
                                &[&self.activations.sdpa_out],
                            );
                            mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.sdpa_out,
                                nh as u32, hd as u32,
                            ).map_err(|e| anyhow::anyhow!("hybrid FWHT sign-undo L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        }
                        // Hybrid path complete; fall through to o_proj/MLP without
                        // entering the legacy `if let Some(ref leg_hb_enc) ...` block
                        // below (`leg_hb_encoded` is `None` under hybrid_kv per the
                        // Phase 10c lazy-alloc mutex, so the `if let` is a no-op).
                    } else
                    // -- iter-24: native HB SDPA (5/6/8-bit byte-packed K/V) --
                    //
                    // K/V have been HB-encoded into leg_hb_encoded above.
                    // We dispatch flash_attn_vec_tq_hb which reads byte-packed K/V
                    // and applies the appropriate codebook inline — no dequant step needed.
                    if let Some(ref leg_hb_enc) = &self.leg_hb_encoded {
                        let hb_cap = leg_hb_enc[layer_idx].capacity;
                        let hb_is_ring = leg_hb_enc[layer_idx].is_sliding;

                        // ADR-028 iter-108: env-gated FWHT-pre fusion.
                        // HF2Q_TQ_FUSE_FWHT_PRE=1 skips the standalone FWHT-pre
                        // dispatch + its forced WAR barrier, instead asking the
                        // FA-vec-tq-hb kernel to apply sign-premult+FWHT+normalize
                        // internally before the K-loop. Iter-107 byte-parity test
                        // confirmed bit-identical output (max_abs_diff=0).
                        // Saves 1 dispatch + 1 barrier per layer × 30 = ~9% decode.
                        let fuse_fwht_pre_env = std::env::var("HF2Q_TQ_FUSE_FWHT_PRE")
                            .map(|v| v == "1").unwrap_or(false);

                        if !fuse_fwht_pre_env {
                            // Pre-rotate Q via FWHT with D1 sign pre-mult (same as 4-bit path).
                            session.barrier_between(
                                &[&self.activations.attn_q_normed],
                                &[&self.activations.attn_q_normed],
                            );
                            mlx_native::ops::fwht_standalone::dispatch_fwht_sign_premult_f32(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_q_normed,
                                nh as u32, hd as u32,
                            ).map_err(|e| anyhow::anyhow!("HB FWHT Q sign-premult L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        }

                        // Native HB SDPA (pre-rotated Q → rotated-domain output).
                        let hb_kv_seq_len = if hb_is_ring {
                            ((kv_write_pos + 1).min(hb_cap)) as u32
                        } else {
                            (kv_write_pos + 1) as u32
                        };
                        let ring_start_hb = if hb_is_ring && hb_kv_seq_len as usize >= hb_cap {
                            ((kv_write_pos + 1) % hb_cap) as u32
                        } else {
                            0u32
                        };
                        session.barrier_between(
                            &[&self.activations.attn_q_normed,
                              &leg_hb_enc[layer_idx].k_packed, &leg_hb_enc[layer_idx].k_norms,
                              &leg_hb_enc[layer_idx].v_packed, &leg_hb_enc[layer_idx].v_norms],
                            &[&self.activations.sdpa_out],
                        );
                        let p_hb = mlx_native::ops::flash_attn_vec_tq_hb::FlashAttnVecTqHbParams {
                            num_heads: nh as u32,
                            num_kv_heads: nkv as u32,
                            head_dim: hd as u32,
                            kv_seq_len: hb_kv_seq_len,
                            kv_capacity: hb_cap as u32,
                            scale: 1.0,
                            mask_type: if is_sliding { 2 } else { 1 },
                            sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                            softcap: 0.0,
                            ring_start: ring_start_hb,
                            scale_factor_d512: ctx.tq_scale_factor_d512,
                            codebook_bits: ctx.tq_codebook_bits,
                            fuse_fwht_pre: if fuse_fwht_pre_env { 1 } else { 0 },
                            // ADR-028 iter-127a Path D: NSG axis. Default 1 in
                            // iter-127a (byte-identical scaffold); compute_nsg
                            // lifts based on kL once kernel logic supports NSG > 1.
                            nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(hb_kv_seq_len),
                        };
                        // ADR-028 §iter-485 (Phase 7d H3): env-gated fused
                        // reduce + FWHT-sign-undo path. Saves 1 dispatch + 1
                        // forced memory_barrier per layer per decode-token
                        // (~30 of each at gemma4 30 layers). Parity test
                        // `reduce_tq_hb_undo_fused_vs_unfused_parity` confirmed
                        // byte-identical output (max_abs_diff=0, max_rel=0).
                        let tq_hb_out_fused = std::env::var("HF2Q_TQ_HB_OUT_FUSED")
                            .map(|v| v == "1").unwrap_or(false);

                        if tq_hb_out_fused {
                            mlx_native::ops::flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_with_fused_undo(
                                session.encoder_mut(), reg, dev,
                                &self.activations.attn_q_normed,
                                &leg_hb_enc[layer_idx].k_packed,
                                &leg_hb_enc[layer_idx].k_norms,
                                &leg_hb_enc[layer_idx].v_packed,
                                &leg_hb_enc[layer_idx].v_norms,
                                &self.activations.sdpa_out,
                                &self.activations.sdpa_tmp,
                                &p_hb,
                            ).map_err(|e| anyhow::anyhow!("flash_attn_vec_tq_hb_with_fused_undo L{layer_idx}: {e}"))?;
                            *total_dispatches += 2; // main + fused-reduce-undo
                            // Caller contract: no trailing fwht_sign_undo
                            // dispatch — the fused reduce already inverse-
                            // rotated the output.
                        } else {
                            mlx_native::ops::flash_attn_vec_tq_hb::flash_attn_vec_tq_hb(
                                session.encoder_mut(), reg, dev,
                                &self.activations.attn_q_normed,
                                &leg_hb_enc[layer_idx].k_packed,
                                &leg_hb_enc[layer_idx].k_norms,
                                &leg_hb_enc[layer_idx].v_packed,
                                &leg_hb_enc[layer_idx].v_norms,
                                &self.activations.sdpa_out,
                                &self.activations.sdpa_tmp,
                                &p_hb,
                            ).map_err(|e| anyhow::anyhow!("flash_attn_vec_tq_hb L{layer_idx}: {e}"))?;
                            *total_dispatches += 2; // main + reduce (conservative)

                            // Inverse-rotate SDPA output.
                            session.barrier_between(
                                &[&self.activations.sdpa_out],
                                &[&self.activations.sdpa_out],
                            );
                            mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.sdpa_out,
                                nh as u32, hd as u32,
                            ).map_err(|e| anyhow::anyhow!("HB FWHT sign-undo L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        }
                    }
                } else if !INVESTIGATION_ENV.skip_tq_sdpa {
                    // -- TQ-packed SDPA (original path) --
                    // Pre-rotate Q via FWHT with D1 sign pre-mult (ADR-007 iter-14 SRHT).
                    // Applies sign_j * Q_j before FWHT so Q_rotated = FWHT(sign*Q)/sqrt(d).
                    // K was encoded as FWHT(sign*K)/sqrt(d); dot product = (sign*Q)·(sign*K) = Q·K.
                    // Sign tables verbatim from AmesianX cpy-utils.cuh:158-163/211-220.
                    session.barrier_between(
                        &[&self.activations.attn_q_normed],
                        &[&self.activations.attn_q_normed],
                    );
                    mlx_native::ops::fwht_standalone::dispatch_fwht_sign_premult_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_q_normed,
                        nh as u32, hd as u32,
                    ).map_err(|e| anyhow::anyhow!("FWHT Q sign-premult pre-rotate L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;

                    // TQ SDPA (pre-rotated Q → rotated-domain output)
                    session.barrier_between(
                        &[&self.activations.attn_q_normed,
                          &self.kv_caches[layer_idx].k_packed, &self.kv_caches[layer_idx].k_norms,
                          &self.kv_caches[layer_idx].v_packed, &self.kv_caches[layer_idx].v_norms],
                        &[&self.activations.sdpa_out],
                    );
                    // iter-25 Subtask B fix: ring_start must be the physical slot of the OLDEST
                    // entry (not newest). kv_write_pos is pre-increment (the slot just written
                    // this step). After wrap: oldest = (kv_write_pos + 1) % capacity.
                    let ring_start = if kv_is_sliding && kv_seq_len >= kv_capacity {
                        ((kv_write_pos + 1) % kv_capacity) as u32
                    } else {
                        0
                    };
                    let p = FlashAttnVecTqParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: kv_seq_len as u32,
                        kv_capacity: kv_capacity as u32,
                        scale: 1.0,
                        mask_type: if is_sliding { 2 } else { 1 },
                        sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                        softcap: 0.0,
                        ring_start,
                        scale_factor_d512: ctx.tq_scale_factor_d512,
                    };
                    mlx_native::ops::flash_attn_vec_tq::flash_attn_vec_tq(
                        session.encoder_mut(), reg, dev,
                        &self.activations.attn_q_normed,
                        &self.kv_caches[layer_idx].k_packed,
                        &self.kv_caches[layer_idx].k_norms,
                        &self.kv_caches[layer_idx].v_packed,
                        &self.kv_caches[layer_idx].v_norms,
                        &self.activations.sdpa_out,
                        &self.activations.sdpa_tmp,
                        &p,
                    ).map_err(|e| anyhow::anyhow!("flash_attn_vec_tq L{layer_idx}: {e}"))?;

                    // Inverse-rotate SDPA output with D1 sign undo (ADR-007 iter-14 SRHT).
                    // Applies FWHT (= IWHT for normalized H) → sign_j * elem_j.
                    // Output accumulated sign*V_weighted; sign undo recovers V_weighted.
                    session.barrier_between(
                        &[&self.activations.sdpa_out],
                        &[&self.activations.sdpa_out],
                    );
                    mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.sdpa_out,
                        nh as u32, hd as u32,
                    ).map_err(|e| anyhow::anyhow!("FWHT sign-undo inv-rotate L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                    *total_dispatches += 2; // main + reduce
                }

                // ADR-009 Phase 3A: dump sdpa_out before O-proj for the detail layer,
                // or ALL layers when HF2Q_DUMP_ALL_CACHE=1
                // W39 iter-112b: route through dump_f32_to with the
                // per-instance dir override so Gate H's in-process harness
                // can redirect dumps after INVESTIGATION_ENV's LazyLock froze.
                if dump_layers && (dump_detail_layer == Some(layer_idx) || dump_all_cache) {
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("dump sdpa_out re-begin L{layer_idx}: {e}"))?).finish()
                        .map_err(|e| anyhow::anyhow!("dump sdpa_out finish L{layer_idx}: {e}"))?;
                    // [nh, 1, hd] flattened.
                    let dir_override = self.dump_dir_override.as_deref();
                    dumps::dump_f32_to(&self.activations.sdpa_out, nh * hd,
                        "sdpa_out", Some(layer_idx), seq_pos, dir_override)?;
                }

                // iter-18 S2C: first-divergence dump (layer=0, sliding, decode steps 1..=10).
                // kv_seq_len=23 = first decode step (prompt len 22 + 1), so steps 1..10 = seq_len 23..32.
                let s2c_step = if kv_seq_len >= 23 && kv_seq_len <= 32 { kv_seq_len - 22 } else { 0 };
                if dump_sliding_l0 && layer_idx == 0 && kv_is_sliding && s2c_step >= 1 {
                    if let Some(run_name) = dump_run_name {
                        std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("S2C re-begin step={s2c_step}: {e}"))?).finish()
                            .map_err(|e| anyhow::anyhow!("S2C dump finish step={s2c_step}: {e}"))?;
                        let dump_base = "/tmp/cfa-iter18/dumps";
                        std::fs::create_dir_all(dump_base)
                            .map_err(|e| anyhow::anyhow!("S2C mkdir: {e}"))?;
                        let p = s2c_step;
                        let run = run_name;
                        // Q (post-RoPE): [nh, hd] f32
                        {
                            let q_raw: &[f32] = self.activations.attn_q_normed.as_slice()
                                .map_err(|e| anyhow::anyhow!("S2C q read: {e}"))?;
                            let q_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                q_raw.as_ptr() as *const u8, nh * hd * 4) };
                            std::fs::write(format!("{dump_base}/pos-{p}-layer-0-q-{run}.bin"), q_bytes)
                                .map_err(|e| anyhow::anyhow!("S2C write q: {e}"))?;
                        }
                        // K cache slot 0: dense path reads from dense_kvs; TQ path reads from k_norms.
                        // We dump K_norms (f32) for TQ and the cache K for dense.
                        if use_dense_sdpa {
                            if let Some(ref dkvs) = self.dense_kvs {
                                let k_raw: &[f32] = dkvs[layer_idx].k.as_slice()
                                    .map_err(|e| anyhow::anyhow!("S2C dense k read: {e}"))?;
                                let slot0_bytes = nkv * hd * 4;
                                let k_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                    k_raw.as_ptr() as *const u8, slot0_bytes.min(k_raw.len() * 4)) };
                                std::fs::write(format!("{dump_base}/pos-{p}-layer-0-k-{run}.bin"), k_bytes)
                                    .map_err(|e| anyhow::anyhow!("S2C write dense k: {e}"))?;
                                let v_raw: &[f32] = dkvs[layer_idx].v.as_slice()
                                    .map_err(|e| anyhow::anyhow!("S2C dense v read: {e}"))?;
                                let v_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                    v_raw.as_ptr() as *const u8, slot0_bytes.min(v_raw.len() * 4)) };
                                std::fs::write(format!("{dump_base}/pos-{p}-layer-0-v-{run}.bin"), v_bytes)
                                    .map_err(|e| anyhow::anyhow!("S2C write dense v: {e}"))?;
                            }
                        } else {
                            // TQ path: dump k_norms + k_packed (representative)
                            let npp = (hd / 256).max(1);
                            let k_norms_raw: &[f32] = self.kv_caches[layer_idx].k_norms.as_slice()
                                .map_err(|e| anyhow::anyhow!("S2C tq k_norms read: {e}"))?;
                            let k_norms_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                k_norms_raw.as_ptr() as *const u8,
                                nkv * kv_capacity * npp * 4) };
                            std::fs::write(format!("{dump_base}/pos-{p}-layer-0-k-{run}.bin"), k_norms_bytes)
                                .map_err(|e| anyhow::anyhow!("S2C write tq k: {e}"))?;
                            let v_norms_raw: &[f32] = self.kv_caches[layer_idx].v_norms.as_slice()
                                .map_err(|e| anyhow::anyhow!("S2C tq v_norms read: {e}"))?;
                            let v_norms_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                v_norms_raw.as_ptr() as *const u8,
                                nkv * kv_capacity * npp * 4) };
                            std::fs::write(format!("{dump_base}/pos-{p}-layer-0-v-{run}.bin"), v_norms_bytes)
                                .map_err(|e| anyhow::anyhow!("S2C write tq v: {e}"))?;
                        }
                        // SDPA output: [nh, hd] f32
                        {
                            let sdpa_raw: &[f32] = self.activations.sdpa_out.as_slice()
                                .map_err(|e| anyhow::anyhow!("S2C sdpa read: {e}"))?;
                            let sdpa_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                sdpa_raw.as_ptr() as *const u8, nh * hd * 4) };
                            std::fs::write(format!("{dump_base}/pos-{p}-layer-0-sdpa-{run}.bin"), sdpa_bytes)
                                .map_err(|e| anyhow::anyhow!("S2C write sdpa: {e}"))?;
                        }
                        eprintln!("[HF2Q_S2C] pos={p} layer=0 dumped q/k/v/sdpa run={run}");
                    }
                }

                // -- O-proj --
                // ADR-028 iter-211: SKIP_O_PROJ bisect.  Sequential
                // single qmatmul on critical path after SDPA.
                if !INVESTIGATION_ENV.skip_o_proj {
                    session.barrier_between(
                        &[&self.activations.sdpa_out, &self.layers[layer_idx].attn.o_proj.buffer],
                        &[&self.activations.attn_out],
                    );
                    dispatch_qmatmul(session, reg, dev, &self.activations.sdpa_out,
                        &self.layers[layer_idx].attn.o_proj, &self.activations.attn_out, 1,
                        crate::quantize::imatrix::ImatrixHint::Layered { tag: "attn_output", layer: layer_idx })?;
                    *total_dispatches += 1;
                }

                // ADR-029 iter-9 — phase split at attn/ffn boundary.
                // HF2Q_PER_LAYER_PHASE_GPU_TIME=1 commits the attn portion
                // and reports its GPU time, then begins a new session for ffn.
                if std::env::var("HF2Q_PER_LAYER_PHASE_GPU_TIME").as_deref() == Ok("1") {
                    let gpu_ns: u64 = std::mem::replace(session, exec.begin().map_err(|e| anyhow::anyhow!("phase-attn begin L{layer_idx}: {e}"))?).finish_with_gpu_time()
                        .map_err(|e| anyhow::anyhow!("phase-attn finish L{layer_idx}: {e}"))?;
                    eprintln!("    [PHASE_ATTN L{:02} {}] gpu={:>6.1}µs",
                        layer_idx,
                        if is_sliding { "S" } else { "G" },
                        gpu_ns as f64 / 1000.0);
                    session.track_dispatch(&[],
                        &[&self.activations.hidden, &self.activations.attn_out]);
                }

                let num_experts = self.num_experts;
                let top_k = self.layers[layer_idx].moe.top_k;

                let dump_after_post_attn = dump_layers && dump_detail_layer == Some(layer_idx);

                // ADR-028 iter-186 — opt-in fused 4→1 kernel that combines:
                //   (a) post-attn norm+add (hidden + norm(attn_out, post_attn_w) → residual)
                //   (b) B8's three concurrent rms_norms over `residual` with weights
                //       {pre_feedforward_layernorm, pre_feedforward_layernorm_2,
                //        router_combined_weight} → {norm_out, moe_norm_out, router_norm_out}
                // Saves 3 dispatches/layer × 30 layers = 90 dispatches/token on gemma4.
                // Kernel `fused_post_attn_triple_norm_f32` already exists in mlx-native
                // (used by batched prefill).  Default-OFF until decode coherence proven.
                //
                // Disabled when dump_layers requires reading `residual` between
                // (a) and (b) — would need a CB split that defeats the fusion.
                if INVESTIGATION_ENV.fused_triple_norm && !dump_after_post_attn {
                    session.barrier_between(
                        &[&self.activations.hidden, &self.activations.attn_out],
                        &[&self.activations.residual,
                          &self.activations.norm_out,
                          &self.activations.moe_norm_out,
                          &self.activations.router_norm_out],
                    );
                    mlx_native::ops::rms_norm::dispatch_fused_post_attn_triple_norm_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.hidden,
                        &self.activations.attn_out,
                        &self.layers[layer_idx].norms.post_attention_layernorm,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                        &self.layers[layer_idx].moe.router_combined_weight,
                        &self.activations.residual,
                        &self.activations.norm_out,
                        &self.activations.moe_norm_out,
                        &self.activations.router_norm_out,
                        eps, 1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("fused post-attn+triple-norm L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                } else {
                    // -- Fused post-attention norm + residual add --
                    // ADR-028 iter-205: SKIP_POST_ATTN_NORM bisect — skip
                    // the fused_norm_add dispatch.  Sequential, 1 per layer.
                    // Produces garbage residual stream.
                    if !INVESTIGATION_ENV.skip_post_attn_norm {
                        // ADR-029 iter-107 H76 — env-gated SPLIT of the
                        // fused norm+add into 2 separate dispatches
                        // (rms_norm → norm_out; elementwise_add hidden+norm_out
                        // → residual). Tests the counter-fusion hypothesis
                        // (iter-105 confirmed: on Apple Metal scheduler, more
                        // smaller dispatches outperform fewer larger fused
                        // dispatches at decode shape).
                        let split_postattn = std::env::var("HF2Q_SPLIT_POSTATTN_NORM").as_deref() == Ok("1");
                        if split_postattn {
                            // Step 1: norm_out = rms_norm(attn_out, post_attn_weight)
                            session.barrier_between(
                                &[&self.activations.attn_out,
                                  &self.layers[layer_idx].norms.post_attention_layernorm],
                                &[&self.activations.norm_out],
                            );
                            session.rms_norm(
                                reg, metal_dev,
                                &self.activations.attn_out,
                                &self.layers[layer_idx].norms.post_attention_layernorm,
                                &self.activations.norm_out,
                                &self.activations.norm_params,
                                1, hs as u32,
                            ).map_err(|e| anyhow::anyhow!("split post-attn norm L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;

                            // Step 2: residual = hidden + norm_out
                            session.barrier_between(
                                &[&self.activations.hidden, &self.activations.norm_out],
                                &[&self.activations.residual],
                            );
                            mlx_native::ops::elementwise::elementwise_add(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.hidden,
                                &self.activations.norm_out,
                                &self.activations.residual,
                                hs,
                                mlx_native::DType::F32,
                            ).map_err(|e| anyhow::anyhow!("split post-attn add L{layer_idx}: {e}"))?;
                        } else {
                            session.barrier_between(
                                &[&self.activations.hidden, &self.activations.attn_out],
                                &[&self.activations.residual],
                            );
                            mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.hidden,
                                &self.activations.attn_out,
                                &self.layers[layer_idx].norms.post_attention_layernorm,
                                &self.activations.residual,
                                hs as u32, 1, eps,
                            ).map_err(|e| anyhow::anyhow!("fused post-attn norm+add L{layer_idx}: {e}"))?;
                        }
                    }

                    if dump_after_post_attn {
                        std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("dump post-attn re-begin L{layer_idx}: {e}"))?).finish()
                            .map_err(|e| anyhow::anyhow!("dump post-attn finish L{layer_idx}: {e}"))?;
                        dumps::dump_f32(&self.activations.residual, hs,
                            "attn_out", Some(layer_idx), seq_pos)?;
                    }
                    *total_dispatches += 1;

                    // ============================================================
                    // Dense MLP + MoE routing INTERLEAVED dispatch
                    // (ADR-006 Phase 4e: matches llama.cpp's graph reorder pattern)
                    //
                    // Group B8:  pre-FF norm1 + pre-FF norm2 + router norm  [3 concurrent]
                    // Group B9:  dense gate + dense up + router logits      [3 concurrent]
                    // Group B10: fused_gelu_mul + fused_moe_routing          [2 concurrent]
                    // Group B11: dense down + gate_up_id                     [2 concurrent]
                    //   ... then sequential MoE chain + post-processing
                    // ============================================================

                    // -- B8: pre-FF norm1 + pre-FF norm2 + router norm [3 CONCURRENT] --
                    session.barrier_between(
                        &[&self.activations.residual],
                        &[&self.activations.norm_out, &self.activations.moe_norm_out,
                          &self.activations.router_norm_out],
                    );
                    // ADR-029 iter-175 Step 1f — Q6_K_M rms_norm fast paths.
                    // All three norms share the same (F32, rows=1, dim=hs)
                    // bake; the shared `decode_record_rms_norm_f32_hs`
                    // OnceLock on `MlxModelWeights` populates on the first
                    // call and serves the remaining ~120 hs-norm dispatches/tok.
                    rms_norm_f32_hs_cached(
                        &self.decode_record_rms_norm_f32_hs,
                        session, reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                        &self.activations.norm_out,
                        &self.activations.norm_params,
                        hs as u32,
                    ).map_err(|e| anyhow::anyhow!("pre-FF norm L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;

                    rms_norm_f32_hs_cached(
                        &self.decode_record_rms_norm_f32_hs,
                        session, reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                        &self.activations.moe_norm_out,
                        &self.activations.norm_params,
                        hs as u32,
                    ).map_err(|e| anyhow::anyhow!("pre-FF norm 2 L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;

                    rms_norm_f32_hs_cached(
                        &self.decode_record_rms_norm_f32_hs,
                        session, reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].moe.router_combined_weight,
                        &self.activations.router_norm_out,
                        &self.activations.norm_params,
                        hs as u32,
                    ).map_err(|e| anyhow::anyhow!("router norm L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                }

                // ADR-029 iter-14 — FFN sub-phase split (HF2Q_FFN_SPLIT=1).
                // Boundary 1: end of "FFN_NORMS" sub-phase (post-attn norm +
                // B8 3 pre-FF norms or the fused_triple_norm equivalent).
                // Commits the CB so the next session's GPU time reports just
                // the FFN body (B9-B13) under the FFN_BODY label.
                if std::env::var("HF2Q_FFN_SPLIT").as_deref() == Ok("1") {
                    let gpu_ns: u64 = std::mem::replace(session, exec.begin().map_err(|e| anyhow::anyhow!("ffn-norms begin L{layer_idx}: {e}"))?).finish_with_gpu_time()
                        .map_err(|e| anyhow::anyhow!("ffn-norms finish L{layer_idx}: {e}"))?;
                    eprintln!("    [FFN_NORMS L{:02} {}] gpu={:>6.1}µs",
                        layer_idx,
                        if is_sliding { "S" } else { "G" },
                        gpu_ns as f64 / 1000.0);
                    session.track_dispatch(&[],
                        &[&self.activations.residual,
                          &self.activations.norm_out,
                          &self.activations.moe_norm_out,
                          &self.activations.router_norm_out]);
                }

                // -- B9: dense gate + dense up + router logits [3 CONCURRENT] --
                // gate/up read norm_out (from B8 norm1); router reads router_norm_out (from B8 router norm).
                // All write disjoint buffers. ONE barrier after B8, then 3 dispatches without barriers.
                session.barrier_between(
                    &[&self.activations.norm_out, &self.activations.router_norm_out],
                    &[&self.activations.mlp_gate, &self.activations.mlp_up,
                      &self.activations.moe_router_logits],
                );
                // ADR-029 iter-15 (H17 probe): HF2Q_B9_FORCE_SEQUENTIAL=1
                // inserts memory_barrier()s between B9's 3 concurrent qmatmuls
                // to test the "peer's more smaller serial dispatches" lever
                // class. M5 Max scheduler may favor sequential issue at this
                // shape (Q5_K 2816→5760 × 2 + 2816→128). Tracks no math
                // change — barriers ONLY affect timing/scheduling.
                let b9_sequential = std::env::var("HF2Q_B9_FORCE_SEQUENTIAL").as_deref() == Ok("1");
                // ADR-028 iter-200: SKIP_DENSE_MLP bisect — skip mlp_gate +
                // mlp_up dispatches.  Router proj must run (MoE depends on it).
                if !INVESTIGATION_ENV.skip_dense_mlp {
                    dispatch_qmatmul(session, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].mlp.gate_proj, &self.activations.mlp_gate, 1,
                        crate::quantize::imatrix::ImatrixHint::Layered { tag: "ffn_gate", layer: layer_idx })?;
                    *total_dispatches += 1;
                    if b9_sequential { session.encoder_mut().memory_barrier(); }
                    dispatch_qmatmul(session, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].mlp.up_proj, &self.activations.mlp_up, 1,
                        crate::quantize::imatrix::ImatrixHint::Layered { tag: "ffn_up", layer: layer_idx })?;
                    *total_dispatches += 1;
                    if b9_sequential { session.encoder_mut().memory_barrier(); }
                }
                // ADR-028 iter-213: SKIP_ROUTING bisect — skip router_proj qmatmul.
                if !INVESTIGATION_ENV.skip_routing {
                    dispatch_qmatmul(session, reg, dev, &self.activations.router_norm_out,
                        &self.layers[layer_idx].moe.router_proj,
                        &self.activations.moe_router_logits, 1,
                        crate::quantize::imatrix::ImatrixHint::Layered { tag: "ffn_gate_inp", layer: layer_idx })?;
                    *total_dispatches += 1;
                }

                // -- B10: fused_gelu_mul + fused_moe_routing [2 CONCURRENT] --
                // gelu_mul reads mlp_gate+mlp_up (from B9 gate/up), writes mlp_fused.
                // moe_routing reads moe_router_logits (from B9 router), writes expert_ids+weights.
                // Disjoint reads and writes — ONE barrier after B9, then both dispatch.
                session.barrier_between(
                    &[&self.activations.mlp_gate, &self.activations.mlp_up,
                      &self.activations.moe_router_logits],
                    &[&self.activations.mlp_fused,
                      &self.activations.moe_expert_ids, &self.activations.moe_routing_weights_gpu],
                );
                if !INVESTIGATION_ENV.skip_dense_mlp {
                    use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};
                    let n_elements_bytes = (self.intermediate_size as u32).to_ne_bytes();
                    let pipeline = reg.get_pipeline("fused_gelu_mul", metal_dev)?;
                    encode_with_args(
                        session.encoder_mut(), pipeline,
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
                    *total_dispatches += 1;
                }
                if !INVESTIGATION_ENV.skip_routing {
                    mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_router_logits,
                        &self.activations.moe_expert_ids,
                        &self.activations.moe_routing_weights_gpu,
                        &self.layers[layer_idx].moe.per_expert_scale,
                        num_experts as u32, top_k as u32,
                    ).map_err(|e| anyhow::anyhow!("fused MoE routing L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                }

                // ============================================================
                // MoE expert dispatches (was S4, now in same session)
                // ============================================================
                let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;
                let use_fused_id = self.layers[layer_idx].moe.stacked_gate_up.is_some()
                    && self.layers[layer_idx].moe.stacked_down.is_some();

                if use_fused_id {
                    let _ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                    let _ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;

                    // -- B11: dense down + gate_up_id [2 concurrent] --
                    // dense_down reads mlp_fused (from B10), gate_up_id reads moe_norm_out
                    // (from B8) + moe_expert_ids (from B10). Disjoint writes.
                    if !INVESTIGATION_ENV.skip_dense_mlp {
                        session.barrier_between(
                            &[&self.activations.mlp_fused, &self.layers[layer_idx].mlp.down_proj.buffer],
                            &[&self.activations.mlp_down],
                        );
                        dispatch_qmatmul(session, reg, dev, &self.activations.mlp_fused,
                            &self.layers[layer_idx].mlp.down_proj, &self.activations.mlp_down, 1,
                            crate::quantize::imatrix::ImatrixHint::Layered { tag: "ffn_down", layer: layer_idx })?;
                        *total_dispatches += 1;
                    }

                    let ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                    session.barrier_between(
                        &[&self.activations.moe_norm_out, &self.activations.moe_expert_ids,
                          self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap()],
                        &[&self.activations.moe_gate_up_id_out],
                    );
                    let gu_params = mlx_native::GgmlQuantizedMatmulIdParams {
                        n_tokens: 1,
                        top_k: top_k as u32,
                        n: (2 * moe_int) as u32,
                        k: hs as u32,
                        n_experts: num_experts as u32,
                        expert_stride: self.layers[layer_idx].moe.gate_up_expert_stride,
                        ggml_type: ggml_type_gu,
                    };
                    // ADR-029 iter-175 Step 1e — Q6_K_ID NR2 m=1 fast path.
                    // On gemma4 APEX-Q5_K_M (Q6_K gate_up) this hits ~30
                    // dispatches/decode-tok.  First call bakes the record
                    // via OnceLock::get_or_init; subsequent calls fire
                    // dispatch_record directly (saves HashMap lookup +
                    // MTLSize::new + GgmlMatvecIdGpuParams construction).
                    // Returns None when ggml_type != Q6_K or
                    // HF2Q_Q6K_ID_MV_NR2 is off → falls through to unbaked.
                    let q6k_id_record_opt = if matches!(gu_params.ggml_type, mlx_native::GgmlType::Q6_K) {
                        self.layers[layer_idx].moe.decode_record_q6k_id_m1_gateup.get_or_init(|| {
                            mlx_native::ops::quantized_matmul_id_ggml::build_q6k_id_nr2_m1_record(
                                reg,
                                dev.metal_device(),
                                gu_params.n,
                                gu_params.k,
                                gu_params.top_k,
                                gu_params.expert_stride,
                            )
                            .ok()
                            .flatten()
                        }).as_ref()
                    } else {
                        None
                    };
                    // ADR-028 iter-201: SKIP_MOE_EXPERTS bisect — skip
                    // gate_up_id + swiglu + down_id dispatches.  Produces
                    // garbage moe_down_id_out (stale buffer).
                    if !INVESTIGATION_ENV.skip_moe_experts {
                        if let Some(rec) = q6k_id_record_opt {
                            session.encoder_mut().dispatch_record(
                                rec,
                                &[
                                    self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap(),
                                    &self.activations.moe_norm_out,
                                    &self.activations.moe_gate_up_id_out,
                                    &self.activations.moe_expert_ids,
                                ],
                            );
                        } else {
                            session.quantized_matmul_id_ggml(
                                reg, dev,
                                &self.activations.moe_norm_out,
                                self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap(),
                                &self.activations.moe_expert_ids,
                                &self.activations.moe_gate_up_id_out,
                                &gu_params,
                            ).map_err(|e| anyhow::anyhow!("gate_up _id L{layer_idx}: {e}"))?;
                        }
                        *total_dispatches += 1;

                        // -- B12: swiglu (singleton) --
                        // ADR-028 iter-202: SKIP_MOE_SWIGLU isolates swiglu
                        // cost.  Skipping leaves moe_swiglu_id_out stale →
                        // down_id reads garbage.  Timing-only bisect.
                        if !INVESTIGATION_ENV.skip_moe_swiglu {
                            session.barrier_between(
                                &[&self.activations.moe_gate_up_id_out],
                                &[&self.activations.moe_swiglu_id_out],
                            );
                            mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.moe_gate_up_id_out,
                                &self.activations.moe_swiglu_id_out,
                                moe_int, top_k,
                            ).map_err(|e| anyhow::anyhow!("swiglu batch L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        }
                    }

                    // -- B13: down_id + post-FF norm1 [2 concurrent] --
                    // down_id reads moe_swiglu_id_out (from B12). post-FF norm1 reads
                    // mlp_down (from B11). Disjoint writes.
                    let ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;
                    session.barrier_between(
                        &[&self.activations.moe_swiglu_id_out, &self.activations.moe_expert_ids,
                          self.layers[layer_idx].moe.stacked_down.as_ref().unwrap()],
                        &[&self.activations.moe_down_id_out],
                    );
                    let dn_params = mlx_native::GgmlQuantizedMatmulIdParams {
                        n_tokens: top_k as u32,
                        top_k: 1,
                        n: hs as u32,
                        k: moe_int as u32,
                        n_experts: num_experts as u32,
                        expert_stride: self.layers[layer_idx].moe.down_expert_stride,
                        ggml_type: ggml_type_dn,
                    };
                    // ADR-029 iter-175 Step 1e2 — Q8_0_ID regular m=1 fast path.
                    // The down dispatch on gemma4 APEX-Q5_K_M is Q8_0 → ~30
                    // dispatches/decode-tok via kernel_mul_mv_id_q8_0_f32.
                    // Bake at first call; subsequent calls fire dispatch_record
                    // directly.  Returns None when ggml_type != Q8_0 or
                    // HF2Q_Q8_0_ID_MV_NR2=1 → falls through to unbaked.
                    let q8_0_id_record_opt = if matches!(dn_params.ggml_type, mlx_native::GgmlType::Q8_0) {
                        self.layers[layer_idx].moe.decode_record_q8_0_id_m1_down.get_or_init(|| {
                            mlx_native::ops::quantized_matmul_id_ggml::build_q8_0_id_decode_record(
                                reg,
                                dev.metal_device(),
                                dn_params.n,
                                dn_params.k,
                                dn_params.n_tokens, // = real_top_k for the down dispatch
                                dn_params.expert_stride,
                            )
                            .ok()
                            .flatten()
                        }).as_ref()
                    } else {
                        None
                    };
                    if !INVESTIGATION_ENV.skip_moe_experts {
                        if let Some(rec) = q8_0_id_record_opt {
                            session.encoder_mut().dispatch_record(
                                rec,
                                &[
                                    self.layers[layer_idx].moe.stacked_down.as_ref().unwrap(),
                                    &self.activations.moe_swiglu_id_out,
                                    &self.activations.moe_down_id_out,
                                    &self.activations.moe_expert_ids,
                                ],
                            );
                        } else {
                            session.quantized_matmul_id_ggml(
                                reg, dev,
                                &self.activations.moe_swiglu_id_out,
                                self.layers[layer_idx].moe.stacked_down.as_ref().unwrap(),
                                &self.activations.moe_expert_ids,
                                &self.activations.moe_down_id_out,
                                &dn_params,
                            ).map_err(|e| anyhow::anyhow!("down _id L{layer_idx}: {e}"))?;
                        }
                        *total_dispatches += 1;
                    }

                    // post-FF norm1: mlp_down → attn_out (concurrent with down_id)
                    session.barrier_between(
                        &[&self.activations.mlp_down],
                        &[&self.activations.attn_out],
                    );
                    // ADR-029 iter-175 Step 1f — fast path same shared bake.
                    rms_norm_f32_hs_cached(
                        &self.decode_record_rms_norm_f32_hs,
                        session, reg, metal_dev,
                        &self.activations.mlp_down,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                        &self.activations.attn_out,
                        &self.activations.norm_params,
                        hs as u32,
                    ).map_err(|e| anyhow::anyhow!("post-FF norm 1 L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;

                    // -- B14: weighted_sum (singleton) --
                    // ADR-028 iter-206: SKIP_WEIGHTED_SUM bisect.
                    // ADR-028 iter-367: fold moe_weighted_sum into the fused
                    // end-of-layer kernel (Path A only).  Default-ON.
                    let use_iter367_fusion = INVESTIGATION_ENV.fused_end_of_layer
                        && !INVESTIGATION_ENV.skip_end_of_layer
                        && INVESTIGATION_ENV.fused_moe_wsum_end_layer_v2
                        && (hs as u32) % 4 == 0;
                    if !INVESTIGATION_ENV.skip_weighted_sum && !use_iter367_fusion {
                        session.barrier_between(
                            &[&self.activations.moe_down_id_out, &self.activations.moe_routing_weights_gpu],
                            &[&self.activations.moe_accum],
                        );
                        mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.moe_down_id_out,
                            &self.activations.moe_routing_weights_gpu,
                            &self.activations.moe_accum,
                            hs, top_k,
                        ).map_err(|e| anyhow::anyhow!("weighted_sum L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                    }
                } else {
                    // Fallback: per-expert loop (all in same session)
                    mlx_native::ops::moe_dispatch::moe_zero_buffer_encode(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_accum, hs,
                    ).map_err(|e| anyhow::anyhow!("zero_buffer L{layer_idx}: {e}"))?;

                    // Note: fallback path still needs CPU to read expert_ids.
                    // For now, this path is unused (all layers have stacked weights).
                    // If needed, we'd add a finish/begin here, but the fused _id path
                    // is always available for Gemma4.
                    anyhow::bail!(
                        "Single-session forward requires fused _id path (stacked weights). \
                         Layer {layer_idx} missing stacked weights."
                    );
                }

                // ADR-029 iter-14 — FFN sub-phase split (HF2Q_FFN_SPLIT=1).
                // Boundary 2: end of "FFN_BODY" sub-phase (B9-B13: dense MLP +
                // MoE experts + interleaved post-FF norm 1).  Commits the CB
                // so the next session's GPU time reports just the end-of-layer
                // norm + add + scalar under the FFN_EOL label.
                if std::env::var("HF2Q_FFN_SPLIT").as_deref() == Ok("1") {
                    let gpu_ns: u64 = std::mem::replace(session, exec.begin().map_err(|e| anyhow::anyhow!("ffn-body begin L{layer_idx}: {e}"))?).finish_with_gpu_time()
                        .map_err(|e| anyhow::anyhow!("ffn-body finish L{layer_idx}: {e}"))?;
                    eprintln!("    [FFN_BODY  L{:02} {}] gpu={:>6.1}µs",
                        layer_idx,
                        if is_sliding { "S" } else { "G" },
                        gpu_ns as f64 / 1000.0);
                    session.track_dispatch(&[],
                        &[&self.activations.mlp_down, &self.activations.moe_accum,
                          &self.activations.attn_out, &self.activations.residual]);
                }

                // ============================================================
                // GPU post-MoE: norm, combine MLP+MoE, final norm, residual, scalar
                // ============================================================

                // ADR-028 iter-207: SKIP_END_OF_LAYER bisect — skip the
                // 2 sequential fused_norm_add dispatches at end-of-layer.
                if !INVESTIGATION_ENV.skip_end_of_layer {
                    let scalar_is_vector = self.layers[layer_idx].layer_scalar.element_count() > 1;

                    // ADR-028 iter-219: HF2Q_FUSED_END_OF_LAYER replaces
                    // the 2 sequential fused_norm_add dispatches with the
                    // single fused_post_ff_norm2_endlayer_f32 kernel.
                    // Bisect-confirmed +2.7% target (iter-208).  Parity test
                    // PASS (iter-218).  Default-OFF until production bench.
                    if INVESTIGATION_ENV.fused_end_of_layer {
                        // ADR-028 iter-367: HF2Q_FUSED_MOE_WSUM_END_LAYER_V2=1 fuses
                        // moe_weighted_sum INTO this end-of-layer kernel, eliminating
                        // 1 dispatch + moe_accum round-trip from gemma4 decode default.
                        // ADR-029 iter-175 Step 1au: cached INVESTIGATION_ENV field
                        // (parsed once at process start) — was per-layer per-token
                        // std::env::var call (~70 ns × 30 layers = 2.1 µs/tok savings).
                        let use_iter367_fusion = INVESTIGATION_ENV.fused_moe_wsum_end_layer_v2
                            && (hs as u32) % 4 == 0
                            && !INVESTIGATION_ENV.skip_weighted_sum;
                        if use_iter367_fusion {
                            // ADR-028 iter-371 (PROBE): explicit memory_barrier()
                            // forces a global Metal barrier even if the tracker
                            // doesn't detect a conflict.  Tests if iter-367's
                            // coherence regression under the iter-321 stack is
                            // caused by a missed barrier (tracker reset between
                            // distant write + read).
                            session.encoder_mut().memory_barrier();
                            session.barrier_between(
                                &[&self.activations.moe_down_id_out,
                                  &self.activations.moe_routing_weights_gpu,
                                  &self.activations.attn_out,
                                  &self.activations.residual,
                                  &self.layers[layer_idx].layer_scalar],
                                &[&self.activations.mlp_down, &self.activations.hidden],
                            );
                            mlx_native::ops::rms_norm::dispatch_fused_moe_wsum_post_ff_norm2_endlayer_f32_v2(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.moe_down_id_out,
                                &self.activations.moe_routing_weights_gpu,
                                &self.activations.attn_out,
                                &self.activations.residual,
                                &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                                &self.layers[layer_idx].norms.post_feedforward_layernorm,
                                &self.layers[layer_idx].layer_scalar,
                                &self.activations.mlp_down,
                                &self.activations.hidden,
                                eps, 1, hs as u32, top_k as u32,
                                scalar_is_vector,
                            ).map_err(|e| anyhow::anyhow!("iter-367 fused wsum+endlayer L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        } else {
                        session.barrier_between(
                            &[&self.activations.attn_out, &self.activations.moe_accum,
                              &self.activations.residual, &self.layers[layer_idx].layer_scalar],
                            &[&self.activations.mlp_down, &self.activations.hidden],
                        );
                        mlx_native::ops::rms_norm::dispatch_fused_post_ff_norm2_endlayer_f32(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_out,
                            &self.activations.moe_accum,
                            &self.activations.residual,
                            &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                            &self.layers[layer_idx].norms.post_feedforward_layernorm,
                            &self.layers[layer_idx].layer_scalar,
                            &self.activations.mlp_down,
                            &self.activations.hidden,
                            eps, 1, hs as u32,
                            scalar_is_vector,
                        ).map_err(|e| anyhow::anyhow!("fused end-of-layer L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                        }
                    } else {
                        // -- Fused post-FF norm 2 + combine MLP+MoE --
                        // ADR-029 iter-108 H77: env-gated SPLIT into 2 dispatches
                        // (rms_norm + elementwise_add). Same counter-fusion test
                        // class as H76; tests if STACKING multiple de-fusions
                        // produces measurable wall improvement (individual
                        // de-fusions are below noise floor per iter-107).
                        let split_postff_normadd = std::env::var("HF2Q_SPLIT_POSTFF_NORMADD").as_deref() == Ok("1");
                        if split_postff_normadd {
                            // Step 1: mlp_down = rms_norm(moe_accum, post_ff_norm_2)
                            session.barrier_between(
                                &[&self.activations.moe_accum,
                                  &self.layers[layer_idx].norms.post_feedforward_layernorm_2],
                                &[&self.activations.mlp_down],
                            );
                            session.rms_norm(
                                reg, metal_dev,
                                &self.activations.moe_accum,
                                &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                                &self.activations.mlp_down,
                                &self.activations.norm_params,
                                1, hs as u32,
                            ).map_err(|e| anyhow::anyhow!("split post-FF norm2 L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;

                            // Step 2: mlp_down = attn_out + mlp_down (in-place add)
                            session.barrier_between(
                                &[&self.activations.attn_out, &self.activations.mlp_down],
                                &[&self.activations.mlp_down],
                            );
                            mlx_native::ops::elementwise::elementwise_add(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_out,
                                &self.activations.mlp_down,
                                &self.activations.mlp_down,
                                hs,
                                mlx_native::DType::F32,
                            ).map_err(|e| anyhow::anyhow!("split post-FF add L{layer_idx}: {e}"))?;
                        } else {
                        session.barrier_between(
                            &[&self.activations.attn_out, &self.activations.moe_accum],
                            &[&self.activations.mlp_down],
                        );
                        mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_out,
                            &self.activations.moe_accum,
                            &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                            &self.activations.mlp_down,
                            hs as u32, 1, eps,
                        ).map_err(|e| anyhow::anyhow!("fused post-FF norm2+combine L{layer_idx}: {e}"))?;
                        }
                        *total_dispatches += 1;

                        // -- Fused end-of-layer: post-FF norm + residual add + scalar mul --
                        // ADR-028 iter-208 sub-bisect: SKIP_END_OF_LAYER_FINAL
                        // skips only this final dispatch (keeps post-FF norm 2).
                        if !INVESTIGATION_ENV.skip_end_of_layer_final {
                            // ADR-029 iter-108 H78: env-gated SPLIT of the
                            // 3-op fused end-of-layer into 3 separate dispatches
                            // (rms_norm + add + scalar_mul). Only enabled when
                            // scalar_is_vector (gemma4 default) — otherwise
                            // fall back to fused since no scalar_mul_f32 kernel
                            // for non-vector scalar exists.
                            let split_postff_normaddscalar = std::env::var("HF2Q_SPLIT_POSTFF_NORMADDSCALAR").as_deref() == Ok("1");
                            if split_postff_normaddscalar && scalar_is_vector {
                                // Step 1: norm_out = rms_norm(mlp_down, post_ff_norm)
                                session.barrier_between(
                                    &[&self.activations.mlp_down,
                                      &self.layers[layer_idx].norms.post_feedforward_layernorm],
                                    &[&self.activations.norm_out],
                                );
                                session.rms_norm(
                                    reg, metal_dev,
                                    &self.activations.mlp_down,
                                    &self.layers[layer_idx].norms.post_feedforward_layernorm,
                                    &self.activations.norm_out,
                                    &self.activations.norm_params,
                                    1, hs as u32,
                                ).map_err(|e| anyhow::anyhow!("split endlayer norm L{layer_idx}: {e}"))?;
                                *total_dispatches += 1;

                                // Step 2: norm_out = residual + norm_out
                                session.barrier_between(
                                    &[&self.activations.residual, &self.activations.norm_out],
                                    &[&self.activations.norm_out],
                                );
                                mlx_native::ops::elementwise::elementwise_add(
                                    session.encoder_mut(), reg, metal_dev,
                                    &self.activations.residual,
                                    &self.activations.norm_out,
                                    &self.activations.norm_out,
                                    hs,
                                    mlx_native::DType::F32,
                                ).map_err(|e| anyhow::anyhow!("split endlayer add L{layer_idx}: {e}"))?;
                                *total_dispatches += 1;

                                // Step 3: hidden = norm_out * layer_scalar (elementwise)
                                session.barrier_between(
                                    &[&self.activations.norm_out,
                                      &self.layers[layer_idx].layer_scalar],
                                    &[&self.activations.hidden],
                                );
                                mlx_native::ops::elementwise::elementwise_mul(
                                    session.encoder_mut(), reg, metal_dev,
                                    &self.activations.norm_out,
                                    &self.layers[layer_idx].layer_scalar,
                                    &self.activations.hidden,
                                    hs,
                                    mlx_native::DType::F32,
                                ).map_err(|e| anyhow::anyhow!("split endlayer scalar L{layer_idx}: {e}"))?;
                                // *total_dispatches += 1 happens via fall-through outside
                            } else {
                            session.barrier_between(
                                &[&self.activations.residual, &self.activations.mlp_down],
                                &[&self.activations.hidden],
                            );
                            mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_scalar_f32(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.residual,
                                &self.activations.mlp_down,
                                &self.layers[layer_idx].norms.post_feedforward_layernorm,
                                &self.activations.hidden,
                                &self.layers[layer_idx].layer_scalar,
                                1, hs as u32, eps,
                                scalar_is_vector,
                            ).map_err(|e| anyhow::anyhow!("fused end-of-layer L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                            }
                        }
                    }
                }

                if let Some(ref mut p) = profile {
                    // All layer ops in single session — attribute everything to S1
                    p.s1_dispatches[layer_idx] = *total_dispatches;
                }

                // ADR-029 iter-110 — CPU-encoding/GPU-execution overlap via
                // split CB. When HF2Q_DECODE_SPLIT_CB_AT_LAYER=N is set,
                // commit (non-blocking) the current session at end of layer
                // N-1 and start a new session for the remaining layers.
                // Mirrors peer's dispatch_apply overlap pattern at
                // /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:550
                // — peer encodes multi-CB in parallel during GPU execution.
                // We achieve the same overlap WITHOUT worker threads by
                // splitting into 2 CBs (non-blocking commit on CB1, encode
                // CB2 while GPU runs CB1, commit_and_wait CB2 at end).
                //
                // GraphSession::commit() returns the CommandEncoder which
                // we drop — Metal retains the committed CB and runs it to
                // completion. Cross-CB buffer dependencies (residual,
                // hidden) resolve via MTLCommandQueue's in-order execution.
                let split_at_layer: Option<usize> = {
                    static SPLIT_AT: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
                    *SPLIT_AT.get_or_init(|| {
                        std::env::var("HF2Q_DECODE_SPLIT_CB_AT_LAYER")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                    })
                };
                if let Some(n) = split_at_layer {
                    if layer_idx + 1 == n {
                        // End current session via commit (non-blocking).
                        // The returned encoder drops; Metal owns the
                        // committed CB and runs it to completion.
                        let prev_session = std::mem::replace(
                            session,
                            exec.begin().map_err(|e| anyhow::anyhow!("split CB begin: {e}"))?,
                        );
                        let _committed_enc = prev_session.commit();
                        // GPU begins executing CB1 immediately; CPU now
                        // proceeds to encode CB2 (layers n..num_layers + head).
                    }
                }

                // ADR-028 iter-292: per-layer dispatch attribution.
                if per_layer_disp_enabled {
                    let layer_disp_end = mlx_native::dispatch_count();
                    per_layer_disp_log.push((
                        layer_idx,
                        is_sliding,
                        layer_disp_end - layer_disp_start,
                    ));
                }

                // ADR-029 iter-9 — per-layer GPU TIME ground truth.
                // HF2Q_PER_LAYER_GPU_TIME=1 commits the session per-layer
                // and records GPU wall-clock via finish_with_gpu_time.
                // HF2Q_PER_LAYER_PHASE_GPU_TIME=1 also commits at the
                // attn/ffn boundary, so this commit at end-of-layer
                // reports just the FFN+EOL phase.
                let phase_split = std::env::var("HF2Q_PER_LAYER_PHASE_GPU_TIME").as_deref() == Ok("1");
                let per_layer = std::env::var("HF2Q_PER_LAYER_GPU_TIME").as_deref() == Ok("1");
                let ffn_split = std::env::var("HF2Q_FFN_SPLIT").as_deref() == Ok("1");
                if per_layer || phase_split || ffn_split {
                    let gpu_ns: u64 = std::mem::replace(session, exec.begin().map_err(|e| anyhow::anyhow!("per-layer-gpu-time begin L{layer_idx}: {e}"))?).finish_with_gpu_time()
                        .map_err(|e| anyhow::anyhow!("per-layer finish L{layer_idx}: {e}"))?;
                    let label = if ffn_split { "FFN_EOL  " }
                                else if phase_split { "PHASE_FFN" }
                                else { "PER_LAYER_GPU" };
                    eprintln!("    [{label} L{:02} {}] gpu={:>6.1}µs",
                        layer_idx,
                        if is_sliding { "S" } else { "G" },
                        gpu_ns as f64 / 1000.0);
                    session.track_dispatch(&[], &[&self.activations.hidden]);
                }

                // ADR-009 Phase 3A: per-layer hidden state dump.
                // Commits the session mid-forward to read hidden state, then re-starts.
                // Only active when HF2Q_DUMP_LAYERS=<seq_pos> matches.
                if dump_layers {
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("dump layer re-begin L{layer_idx}: {e}"))?).finish()
                        .map_err(|e| anyhow::anyhow!("dump layer finish L{layer_idx}: {e}"))?;
                    dumps::dump_f32(&self.activations.hidden, hs,
                        "l_out", Some(layer_idx), seq_pos)?;
                    // Re-start session for remaining layers
                }

                // Dual command buffer: commit buf0 after N layers, start buf1.
                // GPU begins executing buf0 immediately. CPU continues encoding
                // buf1 on the main thread — the overlap is implicit because Metal
                // command buffer execution is asynchronous.
                //
                // Tested and falsified:
                // - Sequential wait BEFORE encode: -5.6 tok/s (serialized pipeline)
                // - Threaded wait DURING encode:   -43 tok/s (thread spawn + Metal
                //   cross-thread synchronization overhead on command queue)
                // The async overlap without any wait is the correct approach.
                // ADR-028 iter-374: multi-split — commit at any of the
                // configured split points, not just the first.
                if dual_buffer_splits.contains(&(layer_idx + 1)) {
                    let b0_barriers = session.barrier_count();
                    let _b0_encoder = std::mem::replace(session, exec.begin().map_err(|e| anyhow::anyhow!("dual-buffer begin: {e}"))?).commit(); // commit current buf → GPU starts async
                    session.track_dispatch(&[], &[&self.activations.hidden]);
                    if INVESTIGATION_ENV.mlx_timing {
                        eprintln!("  [DUAL_BUFFER] split at layer {} — buf0: {} dispatches, {} barriers",
                            layer_idx + 1, *total_dispatches, b0_barriers);
                    }
                }
        Ok(())
    }

}

// ===========================================================================
// ADR-038 Step 4 — Gemma 4 EAGLE-3 tree-verify forward (G4-CFA-1/2)
//
// G4-CFA-1: Gemma4 tree-verify attention block (no sigmoid gate, per-layer
//           head_dim/rope_theta/freq_factors, fused head-norm+RoPE batch).
// G4-CFA-2: gemma4_tree_verify_full_layer_q — wraps G4-CFA-1 with
//           pre_feedforward_layernorm + Q4_0 dense SwiGLU +
//           post_feedforward_layernorm + layer_scalar.
// G4-CFA-3 (MlxModelWeights::forward_tree_verify_gpu) lives in model.rs.
// ===========================================================================

/// Shape parameters for one Gemma 4 full-attention or sliding-attention layer
/// in tree-verify mode.
///
/// Key differences from Qwen35TreeVerifyLayerShape:
/// - `head_dim`: 256 (sliding) or 512 (global) — varies per layer
/// - `num_kv_heads`: 16 (sliding) or 2 (global) for Gemma 4 31B dense
/// - No `attn_output_gate` — Gemma 4 does NOT have a sigmoid output gate
/// - `rope_theta`: `rope_theta_sliding` (10000) or `rope_theta_global` (1e6)
/// - `freq_factors_present`: true for global layers (drives freq_factors mask)
/// - `positions` buffer is U32 `[tree_seq_len]` (not 4×IMROPE format)
#[derive(Debug, Clone, Copy)]
pub struct Gemma4TreeVerifyLayerShape {
    pub hidden_size: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    /// 256 for sliding layers, 512 for global (full-attention) layers.
    pub head_dim: u32,
    pub tree_seq_len: u32,
    pub cache_prefix_len: u32,
    pub kv_capacity: u32,
    pub mask_stride: u32,
    pub rms_norm_eps: f32,
    /// RoPE base frequency: `rope_theta_sliding` (10000) or `rope_theta_global` (1e6).
    pub rope_theta: f32,
    /// True for global (full-attention) layers; triggers freq_factors mask application.
    pub freq_factors_present: bool,
}

impl Gemma4TreeVerifyLayerShape {
    pub fn validate(&self) -> Result<()> {
        use anyhow::ensure;
        ensure!(
            self.head_dim == 256 || self.head_dim == 512,
            "Gemma4TreeVerifyLayerShape: head_dim must be 256 (sliding) or 512 (global); got {}",
            self.head_dim
        );
        ensure!(self.tree_seq_len > 0, "Gemma4TreeVerifyLayerShape: tree_seq_len must be > 0");
        ensure!(self.hidden_size > 0, "Gemma4TreeVerifyLayerShape: hidden_size must be > 0");
        ensure!(self.num_q_heads > 0, "Gemma4TreeVerifyLayerShape: num_q_heads must be > 0");
        ensure!(self.num_kv_heads > 0, "Gemma4TreeVerifyLayerShape: num_kv_heads must be > 0");
        ensure!(
            self.num_q_heads % self.num_kv_heads == 0,
            "Gemma4TreeVerifyLayerShape: num_q_heads ({}) must be divisible by num_kv_heads ({})",
            self.num_q_heads, self.num_kv_heads
        );
        let kv_end = (self.cache_prefix_len as u64)
            .checked_add(self.tree_seq_len as u64)
            .ok_or_else(|| anyhow::anyhow!(
                "Gemma4TreeVerifyLayerShape: cache_prefix_len + tree_seq_len overflows u64"
            ))?;
        ensure!(
            kv_end <= self.kv_capacity as u64,
            "Gemma4TreeVerifyLayerShape: cache_prefix_len ({}) + tree_seq_len ({}) = {} \
             must be <= kv_capacity ({})",
            self.cache_prefix_len, self.tree_seq_len, kv_end, self.kv_capacity
        );
        ensure!(
            self.mask_stride >= kv_end as u32,
            "Gemma4TreeVerifyLayerShape: mask_stride ({}) must be >= cache_prefix_len + \
             tree_seq_len ({})",
            self.mask_stride, kv_end
        );
        Ok(())
    }
}

/// Dispatch Gemma 4 tree-attention (dk256 or dk512 based on `head_dim`).
///
/// Accepts both `head_dim=256` (sliding layers) and `head_dim=512` (global
/// layers) — both Metal kernels are shipped in `mlx-native`. This function
/// is the Gemma 4 counterpart of `dispatch_qwen35_tree_verify_attention`
/// (which is restricted to `head_dim=128` only).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gemma4_tree_verify_attention(
    enc: &mut mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_head_outer: &MlxBuffer,
    k_head_outer: &MlxBuffer,
    v_head_outer: &MlxBuffer,
    tree_mask: &MlxBuffer,
    shape: &Gemma4TreeVerifyLayerShape,
) -> Result<MlxBuffer> {
    use mlx_native::ops::tree_attention::{self as tree_attn_ops, TreeAttentionParams};
    if shape.head_dim != 256 && shape.head_dim != 512 {
        return Err(anyhow::anyhow!(
            "dispatch_gemma4_tree_verify_attention: head_dim must be 256 or 512; got {}",
            shape.head_dim
        ));
    }
    let q = shape.tree_seq_len as usize;
    let nq = shape.num_q_heads as usize;
    let nkv = shape.num_kv_heads as usize;
    let d = shape.head_dim as usize;
    let cap = shape.kv_capacity as usize;
    let kv_seq_len = (shape.cache_prefix_len + shape.tree_seq_len) as u32;
    let stride = shape.mask_stride as usize;

    let out_bytes = q
        .checked_mul(nq).and_then(|v| v.checked_mul(d))
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| anyhow::anyhow!("dispatch_gemma4_tree_verify_attention: out_bytes overflow"))?;
    let kv_req_bytes = nkv
        .checked_mul(cap).and_then(|v| v.checked_mul(d))
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| anyhow::anyhow!("dispatch_gemma4_tree_verify_attention: kv_bytes overflow"))?;
    let mask_req_bytes = q
        .checked_mul(stride)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| anyhow::anyhow!("dispatch_gemma4_tree_verify_attention: mask_bytes overflow"))?;

    if q_head_outer.byte_len() < out_bytes {
        return Err(anyhow::anyhow!(
            "dispatch_gemma4_tree_verify_attention: q buffer too small: have {} bytes, need >= {}",
            q_head_outer.byte_len(), out_bytes
        ));
    }
    if k_head_outer.byte_len() < kv_req_bytes {
        return Err(anyhow::anyhow!(
            "dispatch_gemma4_tree_verify_attention: k buffer too small: have {} bytes, need >= {}",
            k_head_outer.byte_len(), kv_req_bytes
        ));
    }
    if v_head_outer.byte_len() < kv_req_bytes {
        return Err(anyhow::anyhow!(
            "dispatch_gemma4_tree_verify_attention: v buffer too small: have {} bytes, need >= {}",
            v_head_outer.byte_len(), kv_req_bytes
        ));
    }
    if tree_mask.byte_len() < mask_req_bytes {
        return Err(anyhow::anyhow!(
            "dispatch_gemma4_tree_verify_attention: mask buffer too small: have {} bytes, need >= {}",
            tree_mask.byte_len(), mask_req_bytes
        ));
    }

    let scale = 1.0_f32 / (d as f32).sqrt();
    let tmp_bytes = tree_attn_ops::tmp_buffer_bytes(
        shape.num_q_heads,
        shape.head_dim,
        shape.tree_seq_len,
    );
    let output = device
        .alloc_buffer(out_bytes, mlx_native::DType::F32, vec![q, nq, d])
        .map_err(|e| anyhow::anyhow!("dispatch_gemma4_tree_verify_attention: alloc output: {e}"))?;
    let tmp = device
        .alloc_buffer(tmp_bytes, mlx_native::DType::F32, vec![tmp_bytes / 4])
        .map_err(|e| anyhow::anyhow!("dispatch_gemma4_tree_verify_attention: alloc tmp: {e}"))?;

    let tree_params = TreeAttentionParams {
        num_heads: shape.num_q_heads,
        num_kv_heads: shape.num_kv_heads,
        head_dim: shape.head_dim,
        kv_seq_len,
        kv_capacity: shape.kv_capacity,
        scale,
        q_seq_len: shape.tree_seq_len,
        mask_stride: shape.mask_stride,
    };

    enc.memory_barrier();

    tree_attn_ops::tree_attention(
        enc, registry, device,
        q_head_outer, k_head_outer, v_head_outer, tree_mask,
        &output, &tmp, &tree_params,
    )
    .context("dispatch_gemma4_tree_verify_attention: tree_attention")?;

    Ok(output)
}

/// Run one Gemma 4 attention sub-block in tree-verify mode.
///
/// # Op order (11 steps)
///
///  1. Validate shape + buffer invariants.
///  2. `input_layernorm`: RMSNorm(hidden_states_in, norms.input_layernorm).
///  3. Q/K/V projections (Q4_0).
///     `enc.memory_barrier()` — RAW: steps 4-5 read Q/K/V.
///  4. Per-head Q-norm + RoPE (fused); K-norm + RoPE (fused); V-norm (no RoPE).
///     `enc.memory_barrier()` — RAW: step 5 permute reads roped outputs.
///  5. permute_021_f32 × 3: head-outer Q, K_scratch, V_scratch.
///     `enc.memory_barrier()` + commit — CPU-side KV cache append.
///  6. KV-cache append (CPU memcpy).
///  7. Open enc2; `dispatch_gemma4_tree_verify_attention`.
///     `enc2.memory_barrier()` — RAW: step 8 reads attn_out.
///  8. O projection (Q4_0).
///     `enc2.memory_barrier()` — RAW: step 9.
///  9. `post_attention_layernorm` + residual add.
///    `enc2.commit_and_wait()` — terminal.
///
/// **No sigmoid gate** — Gemma 4 does not have `attn_output_gate`.
///
/// Returns: `hidden_states_out` F32 `[tree_seq_len, hidden_size]`.
#[allow(clippy::too_many_arguments)]
pub fn gemma4_tree_verify_attention_block(
    enc: mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    hidden_states_in: &MlxBuffer,
    tree_mask: &MlxBuffer,
    tree_positions: &MlxBuffer,
    k_cache: &mut MlxBuffer,
    v_cache: &mut MlxBuffer,
    layer_weights: &MlxDecoderLayerWeights,
    freq_factors_buf: Option<&MlxBuffer>,
    shape: Gemma4TreeVerifyLayerShape,
) -> Result<MlxBuffer> {
    shape.validate()?;

    let checked_mul = |a: usize, b: usize, ctx: &str| -> Result<usize> {
        a.checked_mul(b)
            .ok_or_else(|| anyhow::anyhow!("gemma4_tree_verify_attention_block: {ctx} overflows usize"))
    };

    let seq = shape.tree_seq_len as usize;
    let h = shape.hidden_size as usize;
    let nq = shape.num_q_heads as usize;
    let nkv = shape.num_kv_heads as usize;
    let d = shape.head_dim as usize;
    let cap = shape.kv_capacity as usize;
    let prefix = shape.cache_prefix_len as usize;

    if hidden_states_in.dtype() != mlx_native::DType::F32 {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_attention_block: hidden_states_in dtype must be F32, got {:?}",
            hidden_states_in.dtype()
        ));
    }
    let hs_elems = checked_mul(seq, h, "tree_seq_len * hidden_size")?;
    if hidden_states_in.element_count() != hs_elems {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_attention_block: hidden_states_in has {} elements, \
             expected {} (tree_seq_len={} * hidden_size={})",
            hidden_states_in.element_count(), hs_elems, seq, h
        ));
    }

    if tree_mask.dtype() != mlx_native::DType::F32 {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_attention_block: tree_mask dtype must be F32, got {:?}",
            tree_mask.dtype()
        ));
    }
    let mask_elems = checked_mul(seq, shape.mask_stride as usize, "tree_seq_len * mask_stride")?;
    if tree_mask.element_count() < mask_elems {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_attention_block: tree_mask has {} elements, \
             need >= {} (tree_seq_len={} * mask_stride={})",
            tree_mask.element_count(), mask_elems, seq, shape.mask_stride
        ));
    }

    if tree_positions.dtype() != mlx_native::DType::U32 {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_attention_block: tree_positions dtype must be U32 (got {:?}); \
             Gemma 4 uses standard RoPE positions, not 4×-IMROPE format",
            tree_positions.dtype()
        ));
    }
    if tree_positions.element_count() != seq {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_attention_block: tree_positions has {} elements, \
             need exactly {} (tree_seq_len)",
            tree_positions.element_count(), seq
        ));
    }

    if k_cache.dtype() != mlx_native::DType::F32 {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_attention_block: k_cache dtype must be F32, got {:?}",
            k_cache.dtype()
        ));
    }
    if v_cache.dtype() != mlx_native::DType::F32 {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_attention_block: v_cache dtype must be F32, got {:?}",
            v_cache.dtype()
        ));
    }
    let kv_req_elems = checked_mul(checked_mul(nkv, cap, "nkv * cap")?, d, "* head_dim")?;
    let kv_req_bytes = checked_mul(kv_req_elems, std::mem::size_of::<f32>(), "* sizeof f32")?;
    if k_cache.byte_len() < kv_req_bytes {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_attention_block: k_cache byte_len {} < required {}",
            k_cache.byte_len(), kv_req_bytes
        ));
    }
    if v_cache.byte_len() < kv_req_bytes {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_attention_block: v_cache byte_len {} < required {}",
            v_cache.byte_len(), kv_req_bytes
        ));
    }

    if layer_weights.norms.input_layernorm.element_count() != h {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_attention_block: input_layernorm has {} elements, expected {} (hidden_size)",
            layer_weights.norms.input_layernorm.element_count(), h
        ));
    }

    let mut enc = enc;

    let rms_out_bytes = hs_elems * std::mem::size_of::<f32>();
    let rms_params_bytes = 2 * std::mem::size_of::<f32>();
    let input_normed = device
        .alloc_buffer(rms_out_bytes, mlx_native::DType::F32, vec![seq, h])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: alloc input_normed: {e}"))?;
    let mut rms_params_buf = device
        .alloc_buffer(rms_params_bytes, mlx_native::DType::F32, vec![2])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: alloc rms_params: {e}"))?;
    {
        let s = rms_params_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: rms_params slice: {e}"))?;
        s[0] = shape.rms_norm_eps;
        s[1] = h as f32;
    }
    mlx_native::ops::rms_norm::dispatch_rms_norm(
        &mut enc, registry, device.metal_device(),
        hidden_states_in,
        &layer_weights.norms.input_layernorm,
        &input_normed,
        &rms_params_buf,
        shape.tree_seq_len,
        shape.hidden_size,
    )
    .context("gemma4_tree_verify_attention_block: step 1 input_layernorm")?;

    enc.memory_barrier();

    let apply_proj = crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32;

    let q_flat = apply_proj(
        &mut enc, registry, device,
        &input_normed, &layer_weights.attn.q_proj.buffer,
        shape.tree_seq_len, shape.hidden_size, (nq * d) as u32,
    )
    .context("gemma4_tree_verify_attention_block: step 2 Q proj")?;
    let k_flat = apply_proj(
        &mut enc, registry, device,
        &input_normed, &layer_weights.attn.k_proj.buffer,
        shape.tree_seq_len, shape.hidden_size, (nkv * d) as u32,
    )
    .context("gemma4_tree_verify_attention_block: step 2 K proj")?;
    let v_flat = {
        let v_proj_buf = match &layer_weights.attn.v_proj {
            Some(vp) => &vp.buffer,
            None => &layer_weights.attn.k_proj.buffer,
        };
        apply_proj(
            &mut enc, registry, device,
            &input_normed, v_proj_buf,
            shape.tree_seq_len, shape.hidden_size, (nkv * d) as u32,
        )
        .context("gemma4_tree_verify_attention_block: step 2 V proj")?
    };

    enc.memory_barrier();

    let half_rope = (d / 2) as u32;
    let q_roped_bytes = checked_mul(checked_mul(seq, nq, "seq*nq")?, d, "*d")?
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("gemma4_tree_verify_attention_block: q_roped bytes overflow"))?;
    let kv_roped_bytes = checked_mul(checked_mul(seq, nkv, "seq*nkv")?, d, "*d")?
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("gemma4_tree_verify_attention_block: kv_roped bytes overflow"))?;
    let q_roped = device
        .alloc_buffer(q_roped_bytes, mlx_native::DType::F32, vec![seq, nq, d])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: alloc q_roped: {e}"))?;
    let k_roped = device
        .alloc_buffer(kv_roped_bytes, mlx_native::DType::F32, vec![seq, nkv, d])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: alloc k_roped: {e}"))?;
    let v_normed = device
        .alloc_buffer(kv_roped_bytes, mlx_native::DType::F32, vec![seq, nkv, d])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: alloc v_normed: {e}"))?;

    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_batch_f32(
        &mut enc, registry, device.metal_device(),
        &q_flat, &q_roped,
        Some(&layer_weights.attn.q_norm_weight),
        tree_positions, freq_factors_buf,
        nq as u32, d as u32, half_rope,
        shape.tree_seq_len, shape.rms_norm_eps, shape.rope_theta,
    )
    .context("gemma4_tree_verify_attention_block: step 3 Q fused norm+RoPE")?;

    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_batch_f32(
        &mut enc, registry, device.metal_device(),
        &k_flat, &k_roped,
        Some(&layer_weights.attn.k_norm_weight),
        tree_positions, freq_factors_buf,
        nkv as u32, d as u32, half_rope,
        shape.tree_seq_len, shape.rms_norm_eps, shape.rope_theta,
    )
    .context("gemma4_tree_verify_attention_block: step 3 K fused norm+RoPE")?;

    {
        let v_norm_params_bytes = 2 * std::mem::size_of::<f32>();
        let mut v_norm_params_buf = device
            .alloc_buffer(v_norm_params_bytes, mlx_native::DType::F32, vec![2])
            .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: alloc v_norm_params: {e}"))?;
        {
            let s = v_norm_params_buf
                .as_mut_slice::<f32>()
                .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: v_norm_params slice: {e}"))?;
            s[0] = shape.rms_norm_eps;
            s[1] = d as f32;
        }
        dispatch_rms_norm_unit_perhead(
            &mut enc, registry, device.metal_device(),
            &RmsNormPerHeadArgs {
                input: &v_flat,
                output: &v_normed,
                params_buf: &v_norm_params_buf,
                rows: (seq * nkv) as u32,
                dim: d as u32,
            },
        )
        .context("gemma4_tree_verify_attention_block: step 3 V per-head norm")?;
    }

    enc.memory_barrier();

    let q_ho_bytes = q_roped_bytes;
    let kv_ho_bytes = kv_roped_bytes;
    let q_head_outer = device
        .alloc_buffer(q_ho_bytes, mlx_native::DType::F32, vec![nq, seq, d])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: alloc q_head_outer: {e}"))?;
    let k_scratch = device
        .alloc_buffer(kv_ho_bytes, mlx_native::DType::F32, vec![nkv, seq, d])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: alloc k_scratch: {e}"))?;
    let v_scratch = device
        .alloc_buffer(kv_ho_bytes, mlx_native::DType::F32, vec![nkv, seq, d])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: alloc v_scratch: {e}"))?;

    mlx_native::ops::transpose::permute_021_f32(
        &mut enc, registry, device.metal_device(),
        &q_roped, &q_head_outer, seq, nq, d,
    )
    .context("gemma4_tree_verify_attention_block: step 4 Q permute")?;
    mlx_native::ops::transpose::permute_021_f32(
        &mut enc, registry, device.metal_device(),
        &k_roped, &k_scratch, seq, nkv, d,
    )
    .context("gemma4_tree_verify_attention_block: step 4 K permute")?;
    mlx_native::ops::transpose::permute_021_f32(
        &mut enc, registry, device.metal_device(),
        &v_normed, &v_scratch, seq, nkv, d,
    )
    .context("gemma4_tree_verify_attention_block: step 4 V permute")?;

    enc.memory_barrier();

    enc.commit_and_wait()
        .context("gemma4_tree_verify_attention_block: step 5 commit before KV append")?;

    {
        let k_src = k_scratch
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: k_scratch as_slice: {e}"))?;
        let v_src = v_scratch
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: v_scratch as_slice: {e}"))?;
        let k_dst = k_cache
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: k_cache as_mut_slice: {e}"))?;
        let v_dst = v_cache
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: v_cache as_mut_slice: {e}"))?;

        for kv_head in 0..nkv {
            for pos in 0..seq {
                let src_off = kv_head
                    .checked_mul(seq).and_then(|x| x.checked_add(pos))
                    .and_then(|x| x.checked_mul(d))
                    .ok_or_else(|| anyhow::anyhow!("gemma4_tree_verify_attention_block: k_src offset overflow"))?;
                let dst_off = kv_head
                    .checked_mul(cap).and_then(|x| x.checked_add(prefix + pos))
                    .and_then(|x| x.checked_mul(d))
                    .ok_or_else(|| anyhow::anyhow!("gemma4_tree_verify_attention_block: k_dst offset overflow"))?;
                k_dst[dst_off..dst_off + d].copy_from_slice(&k_src[src_off..src_off + d]);
                v_dst[dst_off..dst_off + d].copy_from_slice(&v_src[src_off..src_off + d]);
            }
        }
    }

    let mut enc2 = device
        .command_encoder()
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: step 6 open enc2: {e}"))?;

    let attn_out = dispatch_gemma4_tree_verify_attention(
        &mut enc2, device, registry,
        &q_head_outer, k_cache, v_cache, tree_mask,
        &shape,
    )
    .context("gemma4_tree_verify_attention_block: step 6 tree_attention")?;

    enc2.memory_barrier();

    let o_out = crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32(
        &mut enc2, registry, device,
        &attn_out, &layer_weights.attn.o_proj.buffer,
        shape.tree_seq_len, (nq * d) as u32, shape.hidden_size,
    )
    .context("gemma4_tree_verify_attention_block: step 7 O proj")?;

    enc2.memory_barrier();

    let post_attn_normed = device
        .alloc_buffer(hs_elems * std::mem::size_of::<f32>(), mlx_native::DType::F32, vec![seq, h])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: alloc post_attn_normed: {e}"))?;
    mlx_native::ops::rms_norm::dispatch_rms_norm(
        &mut enc2, registry, device.metal_device(),
        &o_out,
        &layer_weights.norms.post_attention_layernorm,
        &post_attn_normed,
        &rms_params_buf,
        shape.tree_seq_len,
        shape.hidden_size,
    )
    .context("gemma4_tree_verify_attention_block: step 8 post_attention_layernorm")?;

    enc2.memory_barrier();

    let hidden_states_out = device
        .alloc_buffer(hs_elems * std::mem::size_of::<f32>(), mlx_native::DType::F32, vec![seq, h])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_attention_block: alloc hidden_states_out: {e}"))?;
    mlx_native::ops::elementwise::elementwise_add(
        &mut enc2, registry, device.metal_device(),
        hidden_states_in, &post_attn_normed, &hidden_states_out,
        hs_elems, mlx_native::DType::F32,
    )
    .context("gemma4_tree_verify_attention_block: step 8 residual add")?;

    enc2.commit_and_wait()
        .context("gemma4_tree_verify_attention_block: step 8 terminal commit")?;

    Ok(hidden_states_out)
}

/// Shape parameters for [`gemma4_tree_verify_full_layer_q`] (dense Q4_0 path).
#[derive(Debug, Clone, Copy)]
pub struct Gemma4TreeVerifyFullLayerShapeQ {
    pub attn: Gemma4TreeVerifyLayerShape,
    /// Dense FFN intermediate size. Gemma 4 31B: 21504.
    pub intermediate_size: u32,
}

impl Gemma4TreeVerifyFullLayerShapeQ {
    pub fn validate(&self) -> Result<()> {
        self.attn.validate()?;
        let h = self.attn.hidden_size as usize;
        let m = self.intermediate_size as usize;
        if self.intermediate_size == 0 {
            return Err(anyhow::anyhow!(
                "Gemma4TreeVerifyFullLayerShapeQ: intermediate_size must be > 0"
            ));
        }
        (m as u64)
            .checked_mul(h as u64)
            .ok_or_else(|| anyhow::anyhow!(
                "Gemma4TreeVerifyFullLayerShapeQ: intermediate_size ({}) * hidden_size ({}) \
                 overflows u64",
                m, h
            ))?;
        m.checked_mul(h).ok_or_else(|| anyhow::anyhow!(
            "Gemma4TreeVerifyFullLayerShapeQ: intermediate_size ({}) * hidden_size ({}) \
             overflows usize",
            m, h
        ))?;
        Ok(())
    }
}

/// Run one complete Gemma 4 dense transformer layer in tree-verify mode — Q4_0 production variant.
///
/// # Op order
///
///  A. `gemma4_tree_verify_attention_block` → attn_out [tree_seq_len, hidden_size].
///  B. `pre_feedforward_layernorm`: RMSNorm(attn_out).
///  C+D. gate_proj, up_proj (Q4_0).
///  E. silu_mul.
///  F. down_proj (Q4_0).
///  G. `post_feedforward_layernorm`.
///  H. residual add: attn_out + post_ff_normed.
///  I. `layer_scalar` multiply.
///
/// Returns: `[tree_seq_len, hidden_size]` F32.
#[allow(clippy::too_many_arguments)]
pub fn gemma4_tree_verify_full_layer_q(
    enc: mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    hidden_states_in: &MlxBuffer,
    tree_mask: &MlxBuffer,
    tree_positions: &MlxBuffer,
    k_cache: &mut MlxBuffer,
    v_cache: &mut MlxBuffer,
    layer_weights: &MlxDecoderLayerWeights,
    freq_factors_buf: Option<&MlxBuffer>,
    shape: Gemma4TreeVerifyFullLayerShapeQ,
) -> Result<MlxBuffer> {
    shape.validate()?;

    let seq = shape.attn.tree_seq_len as usize;
    let h = shape.attn.hidden_size as usize;
    let m = shape.intermediate_size as usize;

    let attn_out = gemma4_tree_verify_attention_block(
        enc, device, registry,
        hidden_states_in, tree_mask, tree_positions,
        k_cache, v_cache,
        layer_weights, freq_factors_buf,
        shape.attn,
    )
    .context("gemma4_tree_verify_full_layer_q: attention block")?;

    let mut enc2 = device
        .command_encoder()
        .context("gemma4_tree_verify_full_layer_q: alloc enc2")?;

    let rms_params_bytes = 2 * std::mem::size_of::<f32>();
    let mut rms_params_buf = device
        .alloc_buffer(rms_params_bytes, mlx_native::DType::F32, vec![2])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_full_layer_q: alloc rms_params: {e}"))?;
    {
        let s = rms_params_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_full_layer_q: rms_params slice: {e}"))?;
        s[0] = shape.attn.rms_norm_eps;
        s[1] = h as f32;
    }

    let rms_out_bytes = seq * h * std::mem::size_of::<f32>();
    let pre_ff_normed = device
        .alloc_buffer(rms_out_bytes, mlx_native::DType::F32, vec![seq, h])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_full_layer_q: alloc pre_ff_normed: {e}"))?;
    mlx_native::ops::rms_norm::dispatch_rms_norm(
        &mut enc2, registry, device.metal_device(),
        &attn_out,
        &layer_weights.norms.pre_feedforward_layernorm,
        &pre_ff_normed,
        &rms_params_buf,
        shape.attn.tree_seq_len,
        shape.attn.hidden_size,
    )
    .context("gemma4_tree_verify_full_layer_q: step C pre_feedforward_layernorm")?;

    enc2.memory_barrier();

    let gate_buf = {
        let out_bytes = seq * m * std::mem::size_of::<f32>();
        let apply_proj = crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32;
        apply_proj(
            &mut enc2, registry, device,
            &pre_ff_normed, &layer_weights.mlp.gate_proj.buffer,
            shape.attn.tree_seq_len, shape.attn.hidden_size,
            shape.intermediate_size,
        )
        .context("gemma4_tree_verify_full_layer_q: gate_proj")?
    };
    let up_buf = {
        let apply_proj = crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32;
        apply_proj(
            &mut enc2, registry, device,
            &pre_ff_normed, &layer_weights.mlp.up_proj.buffer,
            shape.attn.tree_seq_len, shape.attn.hidden_size,
            shape.intermediate_size,
        )
        .context("gemma4_tree_verify_full_layer_q: up_proj")?
    };

    enc2.memory_barrier();

    let n_silu_elems = seq * m;
    if n_silu_elems > (u32::MAX as usize) {
        return Err(anyhow::anyhow!(
            "gemma4_tree_verify_full_layer_q: seq ({}) * intermediate ({}) exceeds u32::MAX",
            seq, m
        ));
    }
    let n_silu = n_silu_elems as u32;
    let activated_bytes = n_silu_elems * std::mem::size_of::<f32>();
    let activated_buf = device
        .alloc_buffer(activated_bytes, mlx_native::DType::F32, vec![seq, m])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_full_layer_q: alloc activated_buf: {e}"))?;
    let mut silu_params_buf = device
        .alloc_buffer(4, mlx_native::DType::U32, vec![1])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_full_layer_q: alloc silu_params: {e}"))?;
    silu_params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_full_layer_q: silu_params slice: {e}"))?[0] = n_silu;
    mlx_native::ops::silu_mul::dispatch_silu_mul(
        &mut enc2, registry, device.metal_device(),
        &gate_buf, &up_buf, &activated_buf,
        &silu_params_buf, n_silu,
    )
    .context("gemma4_tree_verify_full_layer_q: step F silu_mul")?;

    enc2.memory_barrier();

    let down_out = crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32(
        &mut enc2, registry, device,
        &activated_buf, &layer_weights.mlp.down_proj.buffer,
        shape.attn.tree_seq_len, shape.intermediate_size,
        shape.attn.hidden_size,
    )
    .context("gemma4_tree_verify_full_layer_q: down_proj")?;

    enc2.memory_barrier();

    let post_ff_normed = device
        .alloc_buffer(rms_out_bytes, mlx_native::DType::F32, vec![seq, h])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_full_layer_q: alloc post_ff_normed: {e}"))?;
    mlx_native::ops::rms_norm::dispatch_rms_norm(
        &mut enc2, registry, device.metal_device(),
        &down_out,
        &layer_weights.norms.post_feedforward_layernorm,
        &post_ff_normed,
        &rms_params_buf,
        shape.attn.tree_seq_len,
        shape.attn.hidden_size,
    )
    .context("gemma4_tree_verify_full_layer_q: step H post_feedforward_layernorm")?;

    enc2.memory_barrier();

    let hs_elems = seq * h;
    let pre_scalar = device
        .alloc_buffer(hs_elems * std::mem::size_of::<f32>(), mlx_native::DType::F32, vec![seq, h])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_full_layer_q: alloc pre_scalar: {e}"))?;
    mlx_native::ops::elementwise::elementwise_add(
        &mut enc2, registry, device.metal_device(),
        &attn_out, &post_ff_normed, &pre_scalar,
        hs_elems, mlx_native::DType::F32,
    )
    .context("gemma4_tree_verify_full_layer_q: step I residual add")?;

    enc2.memory_barrier();

    let scalar_val = {
        let s = layer_weights
            .layer_scalar
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_full_layer_q: layer_scalar as_slice: {e}"))?;
        if s.is_empty() {
            return Err(anyhow::anyhow!(
                "gemma4_tree_verify_full_layer_q: layer_scalar buffer is empty"
            ));
        }
        s[0]
    };
    let hidden_states_out = device
        .alloc_buffer(hs_elems * std::mem::size_of::<f32>(), mlx_native::DType::F32, vec![seq, h])
        .map_err(|e| anyhow::anyhow!("gemma4_tree_verify_full_layer_q: alloc hidden_states_out: {e}"))?;
    mlx_native::ops::elementwise::scalar_mul_f32(
        &mut enc2, registry, device.metal_device(),
        &pre_scalar, &hidden_states_out,
        hs_elems, scalar_val,
    )
    .context("gemma4_tree_verify_full_layer_q: step J layer_scalar mul")?;

    enc2.commit_and_wait()
        .context("gemma4_tree_verify_full_layer_q: enc2 terminal commit")?;

    Ok(hidden_states_out)
}

/// Compute the divisor for kv_capacity from byte_len: `num_kv_heads * head_dim * sizeof(f32)`.
pub(super) fn nkv_capacity_divisor(num_kv_heads: usize, head_dim: usize) -> usize {
    num_kv_heads * head_dim * std::mem::size_of::<f32>()
}

#[cfg(test)]
mod g4_cfa_tests {
    use super::*;

    /// AC-G4-CFA-1.1 — Gemma4TreeVerifyLayerShape validates dk256 (sliding).
    #[test]
    fn g4_cfa1_layer_shape_dk256_validates_2026_05_22() {
        let shape = Gemma4TreeVerifyLayerShape {
            hidden_size: 5376,
            num_q_heads: 32,
            num_kv_heads: 16,
            head_dim: 256,
            tree_seq_len: 4,
            cache_prefix_len: 8,
            kv_capacity: 16,
            mask_stride: 12,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            freq_factors_present: false,
        };
        shape.validate().expect("dk256 sliding shape must validate");
    }

    /// AC-G4-CFA-1.2 — Gemma4TreeVerifyLayerShape validates dk512 (global).
    #[test]
    fn g4_cfa1_layer_shape_dk512_validates_2026_05_22() {
        let shape = Gemma4TreeVerifyLayerShape {
            hidden_size: 5376,
            num_q_heads: 32,
            num_kv_heads: 2,
            head_dim: 512,
            tree_seq_len: 4,
            cache_prefix_len: 8,
            kv_capacity: 16,
            mask_stride: 12,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            freq_factors_present: true,
        };
        shape.validate().expect("dk512 global shape must validate");
    }

    /// AC-G4-CFA-1.3 — head_dim != 256 or 512 is rejected.
    #[test]
    fn g4_cfa1_layer_shape_rejects_dk128_2026_05_22() {
        let shape = Gemma4TreeVerifyLayerShape {
            hidden_size: 128,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 128,
            tree_seq_len: 1,
            cache_prefix_len: 0,
            kv_capacity: 1,
            mask_stride: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            freq_factors_present: false,
        };
        assert!(shape.validate().is_err(), "head_dim=128 must be rejected");
    }

    /// AC-G4-CFA-1.4 — cache overflow is rejected.
    #[test]
    fn g4_cfa1_layer_shape_rejects_cache_overflow_2026_05_22() {
        let shape = Gemma4TreeVerifyLayerShape {
            hidden_size: 256,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 256,
            tree_seq_len: 8,
            cache_prefix_len: 10,
            kv_capacity: 16,
            mask_stride: 18,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            freq_factors_present: false,
        };
        assert!(shape.validate().is_err(), "cache overflow must be rejected");
    }

    /// AC-G4-CFA-2.1 — Gemma4TreeVerifyFullLayerShapeQ validates.
    #[test]
    fn g4_cfa2_full_layer_shape_validates_2026_05_22() {
        let shape = Gemma4TreeVerifyFullLayerShapeQ {
            attn: Gemma4TreeVerifyLayerShape {
                hidden_size: 5376,
                num_q_heads: 32,
                num_kv_heads: 16,
                head_dim: 256,
                tree_seq_len: 4,
                cache_prefix_len: 8,
                kv_capacity: 16,
                mask_stride: 12,
                rms_norm_eps: 1e-6,
                rope_theta: 10000.0,
                freq_factors_present: false,
            },
            intermediate_size: 21504,
        };
        shape.validate().expect("Gemma4 31B full layer shape must validate");
    }

    /// AC-G4-CFA-2.2 — intermediate_size=0 is rejected.
    #[test]
    fn g4_cfa2_full_layer_shape_rejects_zero_intermediate_2026_05_22() {
        let shape = Gemma4TreeVerifyFullLayerShapeQ {
            attn: Gemma4TreeVerifyLayerShape {
                hidden_size: 256,
                num_q_heads: 1,
                num_kv_heads: 1,
                head_dim: 256,
                tree_seq_len: 1,
                cache_prefix_len: 0,
                kv_capacity: 1,
                mask_stride: 1,
                rms_norm_eps: 1e-6,
                rope_theta: 10000.0,
                freq_factors_present: false,
            },
            intermediate_size: 0,
        };
        assert!(shape.validate().is_err(), "intermediate_size=0 must be rejected");
    }

    /// AC-G4-CFA-3.1 — nkv_capacity_divisor returns correct byte stride.
    #[test]
    fn g4_cfa3_nkv_capacity_divisor_dk256_2026_05_22() {
        assert_eq!(nkv_capacity_divisor(16, 256), 16 * 256 * 4);
    }

    /// AC-G4-CFA-3.2 — nkv_capacity_divisor dk512.
    #[test]
    fn g4_cfa3_nkv_capacity_divisor_dk512_2026_05_22() {
        assert_eq!(nkv_capacity_divisor(2, 512), 2 * 512 * 4);
    }
}
