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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::debug::INVESTIGATION_ENV;
use super::config::LayerType;
use super::forward_mlx::{
    DenseKvBuffers, MlxModelWeights, dispatch_qmatmul,
    dispatch_rms_norm_unit_perhead_dual,
};
use super::gpu::GpuContext;

// Wave P4.0 — env-gated per-kernel GPU-time profiling.  Enabled via
// HF2Q_PROFILE_FA / HF2Q_PROFILE_MOE / HF2Q_PROFILE_MM to break out the
// contribution of each kernel category.  Each adds 2 commit_and_wait per
// dispatch to isolate its session, so prefill total throughput will
// REGRESS substantially when on — never enable in production.
static PROFILE_FA_SW_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_FA_SW_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_FA_GL_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_FA_GL_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_MOE_GU_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MOE_GU_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_MOE_DN_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MOE_DN_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_MM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MM_COUNT: AtomicU64 = AtomicU64::new(0);

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
        // bf16 = 2 bytes/element (ADR-011 Phase 2 Wave 3 bf16 conversion).
        // Intermediate sublayer activations (Q/K/V, SDPA out, MLP/MoE expert
        // outputs) move to bf16 per the MLX-LM dtype convention; residual
        // stream stays f32. See docs/ADR-011-phase2-bf16-conversion-map.md.
        let bf16_sz: usize = 2;

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
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let nkv = layer.num_kv_heads;
            let hd = layer.head_dim;
            let layer_is_ring = layer.layer_type == LayerType::Sliding;
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
        let max_hd = self.layers.iter().map(|l| l.head_dim).max().unwrap_or(512);
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
        // bf16 allocation helper — used for intermediate sublayer activations
        // (Q/K/V post-qmatmul casts, head-normed + RoPE'd Q/K/V, permuted
        // Q/K/V for SDPA, SDPA output, MLP/MoE expert intermediates). The
        // residual stream (pf_hidden, pf_residual) stays f32.
        let alloc_bf16 = |n: usize, name: &str| -> Result<MlxBuffer> {
            dev.alloc_buffer(n * bf16_sz, DType::BF16, vec![n])
                .map_err(|e| anyhow::anyhow!("batched alloc {name}: {e}"))
        };
        let alloc_u32 = |n: usize, name: &str| -> Result<MlxBuffer> {
            dev.alloc_buffer(n * u32_sz, DType::U32, vec![n])
                .map_err(|e| anyhow::anyhow!("batched alloc {name}: {e}"))
        };

        let max_nkv = self.layers.iter().map(|l| l.num_kv_heads).max().unwrap_or(8);
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

        // ADR-011 Phase 2 Wave 3 (bf16 SDPA island):
        //
        // Q/K/V projections (qmatmul) and head-norm+RoPE remain f32 in this
        // stage — the MLX-LM convention calls for bf16 but the upstream f32
        // sources (quantized_matmul_ggml kernels, the f32 norm weights
        // loaded via `gguf.load_tensor_f32`) would require either
        // mlx-native kernel changes or a per-layer f32→bf16 weight cast.
        // Neither is in Wave 3's scope. Instead we introduce a *bf16 island*
        // spanning permute→SDPA→permute: cast f32 normed Q/K/V into bf16
        // buffers, run permute_021_bf16 + sdpa_bf16 + back-permute_021_bf16
        // on bf16 data, then cast bf16→f32 into the f32 `pf_sdpa_out` buffer
        // for the O-proj qmatmul (also f32-only). This matches the dtype
        // convention for the core attention compute and is exactly the
        // region Wave 4's `flash_attn_prefill` (bf16-only) will later wrap.
        let pf_q_normed_bf16 = alloc_bf16(seq_len * nh * max_hd, "pf_q_normed_bf16")?;
        let pf_k_normed_bf16 = alloc_bf16(seq_len * max_nkv * max_hd, "pf_k_normed_bf16")?;
        let pf_v_normed_bf16 = alloc_bf16(seq_len * max_nkv * max_hd, "pf_v_normed_bf16")?;
        let pf_q_perm = alloc_bf16(nh * seq_len * max_hd, "pf_q_perm")?;
        let pf_k_perm = alloc_bf16(max_nkv * seq_len * max_hd, "pf_k_perm")?;
        let pf_v_perm = alloc_bf16(max_nkv * seq_len * max_hd, "pf_v_perm")?;
        let mut pf_sdpa_out_perm = alloc_bf16(nh * seq_len * max_hd, "pf_sdpa_out_perm")?;
        // Wave P4.10 — pf_sdpa_out_bf16 (the intermediate bf16 buffer
        // between permute and cast) is no longer needed: the fused
        // permute_021_bf16_to_f32 dispatch writes f32 directly into
        // pf_sdpa_out from the bf16 source.
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

        // ADR-011 Phase 3 Wave P3b — scratch pooling for MoE mm_id path.
        //
        // The `quantized_matmul_id_ggml` mm_id branch (n_tokens > 8) needs
        // two small u32 scratch buffers per call (htpe: per-expert count,
        // hids: per-expert routed-token list).  Pool once here for the
        // whole prefill.
        //
        // Sizing: P3b-tensor.2 routes BOTH the gate_up call (top_k=8,
        // n_tokens=seq_len) AND the down call (top_k=1,
        // n_tokens=seq_len*top_k_max) through mm_id.  Size for the down
        // call's larger n_tokens.
        let max_id_n_tokens = (seq_len * top_k_max) as u32;
        let mut pf_moe_mm_scratch = mlx_native::IdMmScratch::alloc(
            dev, num_experts as u32, max_id_n_tokens,
        ).map_err(|e| anyhow::anyhow!("batched alloc IdMmScratch: {e}"))?;

        let mut pf_positions = alloc_u32(seq_len, "pf_positions")?;
        {
            let p: &mut [u32] = pf_positions.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("positions write: {e}"))?;
            for (i, slot) in p[..seq_len].iter_mut().enumerate() {
                *slot = i as u32;
            }
        }
        let mut pf_token_ids = alloc_u32(seq_len, "pf_token_ids")?;
        {
            let t: &mut [u32] = pf_token_ids.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("token_ids write: {e}"))?;
            for (i, &tok) in prompt_tokens.iter().enumerate() { t[i] = tok; }
        }

        // -------------------------------------------------------------------
        // SETUP SESSION (Wave P4.6 — merge embed + mask+blk into one session)
        //
        // Pre-P4.6 these ran as two separate `exec.begin()` / `s.finish()`
        // pairs, costing an extra commit_and_wait per prefill.  The two
        // workloads are independent (embed writes pf_hidden; mask+blk write
        // sliding_mask, global_mask, blk_sliding, blk_global — no overlap),
        // so they share the setup command buffer.  The blk dispatches still
        // need barrier_between against their masks (intra-session ordering)
        // but no inter-workload barrier is required.
        // -------------------------------------------------------------------
        //
        // ADR-011 Phase 2 Wave 4 mask + blk pre-pass docs (preserved):
        //   Build the two SWA/causal masks (sliding + global) and the two
        //   tile-skip pre-pass `blk` byte buffers ONCE per prefill, before
        //   the layer loop. Reused by 25 sliding (D=256) layers + 5 global
        //   (D=512) layers.  Mask layout: [seq_len, seq_len] bf16, single-
        //   plane.  blk layout: one byte per (qtile, ktile) at the tile
        //   shape used by each main kernel — (BQ=32, BK=16) for D=256,
        //   (BQ=8, BK=64) for D=512.  Constants — scale=1.0 (Q pre-scaled
        //   upstream), do_causal=false (mask carries causal), q_abs_offset=0.
        let prefill_start = Instant::now();
        let sliding_mask: MlxBuffer;
        let global_mask: MlxBuffer;
        let blk_sliding: MlxBuffer;
        let blk_global: MlxBuffer;
        {
            use mlx_native::ops::flash_attn_prefill_mask::{
                build_sdpa_mask_bf16, SdpaMaskParams,
            };
            use mlx_native::ops::flash_attn_prefill_blk::{
                alloc_blk_buffer, dispatch_flash_attn_prefill_blk, BlkParams,
            };

            // Pre-allocate the two blk byte buffers (Metal alloc, no kernel
            // dispatch) outside the session so &mut borrow stays confined.
            let blk_sliding_params = BlkParams {
                seq_len_q: seq_len as u32,
                seq_len_k: seq_len as u32,
                bq: 32,
                bk: 16,
            };
            blk_sliding = alloc_blk_buffer(dev, &blk_sliding_params)
                .map_err(|e| anyhow::anyhow!("alloc blk_sliding: {e}"))?;
            let blk_global_params = BlkParams {
                seq_len_q: seq_len as u32,
                seq_len_k: seq_len as u32,
                bq: 8,
                bk: 64,
            };
            blk_global = alloc_blk_buffer(dev, &blk_global_params)
                .map_err(|e| anyhow::anyhow!("alloc blk_global: {e}"))?;

            let mut s = exec.begin()
                .map_err(|e| anyhow::anyhow!("batched setup session: {e}"))?;

            // 1. Embedding: token_ids -> pf_hidden (no dependency on masks/blk).
            s.track_dispatch(&[&self.embed_weight, &pf_token_ids], &[&pf_hidden]);
            mlx_native::ops::elementwise::embedding_gather_scale_batch_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.embed_weight,
                &pf_token_ids,
                &pf_hidden,
                hs, seq_len,
                (hs as f32).sqrt(),
            ).map_err(|e| anyhow::anyhow!("batched embed: {e}"))?;

            // 2. Sliding-window causal mask — reused across all 25 sliding layers.
            sliding_mask = build_sdpa_mask_bf16(
                dev, reg, s.encoder_mut(),
                &SdpaMaskParams {
                    seq_len_q: seq_len as u32,
                    seq_len_k: seq_len as u32,
                    window_size: Some(self.sliding_window as u32),
                    causal: true,
                    q_abs_offset: 0,
                },
            ).map_err(|e| anyhow::anyhow!("build sliding_mask: {e}"))?;

            // 3. Global causal mask — reused across all 5 global layers.
            global_mask = build_sdpa_mask_bf16(
                dev, reg, s.encoder_mut(),
                &SdpaMaskParams {
                    seq_len_q: seq_len as u32,
                    seq_len_k: seq_len as u32,
                    window_size: None,
                    causal: true,
                    q_abs_offset: 0,
                },
            ).map_err(|e| anyhow::anyhow!("build global_mask: {e}"))?;

            // 4. Tile-skip classifiers — read mask, write blk bytes.
            s.barrier_between(
                &[&sliding_mask],
                &[&blk_sliding],
            );
            dispatch_flash_attn_prefill_blk(
                s.encoder_mut(), dev, reg,
                &sliding_mask, &blk_sliding,
                &blk_sliding_params,
            ).map_err(|e| anyhow::anyhow!("dispatch blk_sliding: {e}"))?;

            s.barrier_between(
                &[&global_mask],
                &[&blk_global],
            );
            dispatch_flash_attn_prefill_blk(
                s.encoder_mut(), dev, reg,
                &global_mask, &blk_global,
                &blk_global_params,
            ).map_err(|e| anyhow::anyhow!("dispatch blk_global: {e}"))?;

            s.finish()
                .map_err(|e| anyhow::anyhow!("batched setup finish: {e}"))?;
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

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_start = std::time::Instant::now();
            let hd = layer.head_dim;
            let nkv = layer.num_kv_heads;
            let is_sliding = layer.layer_type == LayerType::Sliding;
            let top_k = layer.moe.top_k;
            let moe_int = layer.moe.moe_intermediate_size;

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
                let profile_mm = std::env::var("HF2Q_PROFILE_MM").is_ok();
                let qkv_t0 = if profile_mm {
                    s.finish().map_err(|e| anyhow::anyhow!("MM-profile pre-finish (qkv) L{layer_idx}: {e}"))?;
                    let t0 = std::time::Instant::now();
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MM-profile begin (qkv) L{layer_idx}: {e}"))?;
                    Some(t0)
                } else { None };
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
                if let Some(t0) = qkv_t0 {
                    s.finish().map_err(|e| anyhow::anyhow!("MM-profile post-finish (qkv) L{layer_idx}: {e}"))?;
                    PROFILE_MM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_MM_COUNT.fetch_add(if v_is_k { 2 } else { 3 }, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MM-profile resume (qkv) L{layer_idx}: {e}"))?;
                }

                // 3. Batched fused head norm + RoPE on Q and K.
                //
                // ADR-011 Phase 3 Wave P3b.4 — use the `_with_bf16` variant
                // so the kernel co-writes the normed Q/K in both f32 AND
                // bf16 in a single dispatch.  The f32 pf_{q,k}_normed path
                // stays (KV cache copy + ADR-010 dump both read f32); the
                // bf16 pf_{q,k}_normed_bf16 path eliminates the two
                // otherwise-separate f32→bf16 cast dispatches that fed
                // the bf16 attention island.  Total: 60 cast dispatches /
                // prefill eliminated (2 per layer × 30 layers).
                s.barrier_between(
                    &[&pf_q, &pf_k],
                    &[&pf_q_normed, &pf_k_normed,
                      &pf_q_normed_bf16, &pf_k_normed_bf16],
                );
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_batch_f32_with_bf16(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_q,
                    &pf_q_normed,
                    Some(&pf_q_normed_bf16),
                    Some(&self.layers[layer_idx].attn.q_norm_weight),
                    &pf_positions,
                    ff_gpu,
                    nh as u32, hd as u32, half_rope,
                    seq_len as u32,
                    eps, theta,
                ).map_err(|e| anyhow::anyhow!("batched Q norm+RoPE+bf16 L{layer_idx}: {e}"))?;
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_batch_f32_with_bf16(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_k,
                    &pf_k_normed,
                    Some(&pf_k_normed_bf16),
                    Some(&self.layers[layer_idx].attn.k_norm_weight),
                    &pf_positions,
                    ff_gpu,
                    nkv as u32, hd as u32, half_rope,
                    seq_len as u32,
                    eps, theta,
                ).map_err(|e| anyhow::anyhow!("batched K norm+RoPE+bf16 L{layer_idx}: {e}"))?;

                // 4. V norm (unit RMS, no RoPE, per-head across seq_len)
                //    Layout: [seq_len * nkv, hd] — treat all positions' heads as rows
                //
                // ADR-011 Phase 3 Wave P3b-tensor.3 — use the dual-output
                // variant so the kernel co-writes both pf_v_normed (f32,
                // for the KV cache copy below) and pf_v_normed_bf16 (for
                // the bf16 attention island).  Eliminates the otherwise-
                // separate f32→bf16 cast dispatch (30 dispatches /
                // prefill).
                let v_input = if v_is_k { &pf_k } else { &pf_v };
                s.barrier_between(
                    &[v_input],
                    &[&pf_v_normed, &pf_v_normed_bf16],
                );
                dispatch_rms_norm_unit_perhead_dual(
                    s.encoder_mut(), reg, metal_dev,
                    v_input,
                    &pf_v_normed,
                    &pf_v_normed_bf16,
                    hd_norm_params,
                    (seq_len * nkv) as u32,
                    hd as u32,
                )?;

                // 5. Permute [seq_len, n_heads, hd] → [n_heads, seq_len, hd] (bf16).
                s.barrier_between(
                    &[&pf_q_normed_bf16, &pf_k_normed_bf16, &pf_v_normed_bf16],
                    &[&pf_q_perm, &pf_k_perm, &pf_v_perm],
                );
                mlx_native::ops::transpose::permute_021_bf16(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_q_normed_bf16, &pf_q_perm,
                    seq_len, nh, hd,
                ).map_err(|e| anyhow::anyhow!("batched permute Q L{layer_idx}: {e}"))?;
                mlx_native::ops::transpose::permute_021_bf16(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_k_normed_bf16, &pf_k_perm,
                    seq_len, nkv, hd,
                ).map_err(|e| anyhow::anyhow!("batched permute K L{layer_idx}: {e}"))?;
                mlx_native::ops::transpose::permute_021_bf16(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_v_normed_bf16, &pf_v_perm,
                    seq_len, nkv, hd,
                ).map_err(|e| anyhow::anyhow!("batched permute V L{layer_idx}: {e}"))?;

                // 6. Flash-attention tiled prefill (ADR-011 Phase 2 Wave 4):
                //    Q: [1, nh, seq_len, hd], K: [1, nkv, seq_len, hd], V: same
                //    scale = 1.0 for Gemma 4 (per llama.cpp oracle — Q is
                //      pre-scaled upstream in qmatmul).
                //    Global layers (head_dim=512): flash_attn_prefill_bf16_d512
                //      (llama.cpp-derived NSG=8 kernel). Consumes global_mask
                //      + blk_global built once per prefill above.
                //    Sliding layers (head_dim=256): flash_attn_prefill_bf16_d256
                //      (candle-derived BQ=32/BK=16 kernel). Consumes sliding_mask
                //      + blk_sliding — mask carries `q_abs - k_pos < window_size`
                //      AND causal constraint (Wave 2D SWA mask), so in-kernel
                //      do_causal=false avoids double-masking.
                //
                // History note: Wave 3 had a narrow bf16 SDPA island using
                // sdpa_bf16 (D=256) and sdpa (D=512) kernels; this was a
                // stepping-stone. Wave 4 replaces both with the flash-attention
                // tiled kernels that llama.cpp uses (flash_attn_ext_* family)
                // — single kernel for both sliding + global, with a single
                // mask representation. sdpa_sliding previously had a "dense
                // cap 1024" issue at pp=2455 (docs/spike-gate-a-prefill.md
                // §Addendum); flash_attn_prefill unblocks that sub-gate.
                s.barrier_between(
                    &[&pf_q_perm, &pf_k_perm, &pf_v_perm],
                    &[&pf_sdpa_out_perm],
                );
                // Wave P4.0 — env-gated FA isolation for per-kernel timing.
                // When HF2Q_PROFILE_FA=1: commit-and-wait the QKV+permute
                // work, restart with a session that holds ONLY the FA
                // dispatch, then commit-and-wait again to capture true
                // FA-only wall-clock GPU time.  The 2 extra syncs/layer
                // make the overall prefill slower but isolate FA's cost
                // from QKV mm and the post-FA permute/cast.
                let profile_fa = std::env::var("HF2Q_PROFILE_FA").is_ok();
                let fa_start = if profile_fa {
                    s.finish().map_err(|e| anyhow::anyhow!("FA-profile pre-finish L{layer_idx}: {e}"))?;
                    let t0 = std::time::Instant::now();
                    s = exec.begin().map_err(|e| anyhow::anyhow!("FA-profile begin L{layer_idx}: {e}"))?;
                    Some(t0)
                } else {
                    None
                };
                if is_sliding {
                    // ADR-011 Phase 2 Wave 4 Stage 2: flash_attn_prefill D=256
                    // replaces sdpa_sliding. Inputs:
                    //   - Q/K/V/O: bf16 [n_heads/n_kv_heads, seq_len, hd=256],
                    //     contiguous inner dim (already ensured by the
                    //     permute_021_bf16 pre-SDPA step above).
                    //   - mask: &sliding_mask, rank-2 [seq_len, seq_len] bf16
                    //     built once per prefill with window_size=sliding_window
                    //     and causal=true (Wave 2D). Post-Wave-4.1 dispatcher
                    //     detects rank-2 and emits strides [0,0,kL] so the
                    //     single plane broadcasts across all 16 heads.
                    //   - blk: &blk_sliding, (BQ=32, BK=16) tile-skip bytes
                    //     from Wave 2E classifier. Per-tile content matches
                    //     (sliding_mask tile); main kernel skips fully-masked
                    //     tiles entirely, saving work on rows where the
                    //     sliding window excludes most of the prefix.
                    //   - scale=1.0 (Gemma 4: Q is pre-scaled upstream in
                    //     qmatmul), do_causal=false (mask carries causal).
                    mlx_native::ops::flash_attn_prefill::
                        dispatch_flash_attn_prefill_bf16_d256_with_blk(
                        s.encoder_mut(), dev, reg,
                        &pf_q_perm, &pf_k_perm, &pf_v_perm,
                        Some(&sliding_mask),
                        Some(&blk_sliding),
                        &mut pf_sdpa_out_perm,
                        &mlx_native::ops::flash_attn_prefill::FlashAttnPrefillParams {
                            n_heads: nh as u32,
                            n_kv_heads: nkv as u32,
                            head_dim: hd as u32,
                            seq_len_q: seq_len as u32,
                            seq_len_k: seq_len as u32,
                            batch: 1,
                            scale: 1.0,
                            do_causal: false,
                        },
                    ).map_err(|e| anyhow::anyhow!("batched sliding flash_attn_prefill L{layer_idx}: {e}"))?;
                } else {
                    // ADR-011 Phase 2 Wave 4 Stage 3: flash_attn_prefill D=512
                    // (NSG=8 llama.cpp-derived kernel) replaces s.sdpa for
                    // Gemma 4's 5 global layers (head_dim=512).
                    //   - Q/K/V/O: bf16 [n_heads/n_kv_heads, seq_len, 512],
                    //     contiguous inner dim.
                    //   - mask: &global_mask, rank-2 [seq_len, seq_len] bf16
                    //     with window_size=None, causal=true (Wave 2D).
                    //   - blk: &blk_global, (BQ=8, BK=64) tile-skip bytes —
                    //     BK=64 matches the D=512 main kernel's ic0 loop
                    //     step (NCPSG=64). For fully-causal masks most
                    //     tiles are type-1 (mixed), so blk offers modest
                    //     savings mostly at (qtile, ktile) pairs beyond
                    //     the causal diagonal.
                    //   - scale=1.0, do_causal=false (same contract as D=256).
                    //
                    // History: an interim revert (commit 787f2fe) routed this
                    // branch back to s.sdpa after a kernel-level softmax bug
                    // in flash_attn_prefill_d512.metal flipped sourdough_gate
                    // argmaxes. The bug was a double application of log2(e)
                    // inside the online softmax (mlx-native commit f3abe0d);
                    // it is now fixed and this revert restores the original
                    // Wave 4 Stage 3 wiring. See
                    //   /opt/hf2q/docs/ADR-011-phase2-wave2c-d512-bug-fix.md
                    // for the per-line trace, the math, and the per-test
                    // tolerance numbers.
                    mlx_native::ops::flash_attn_prefill_d512::
                        dispatch_flash_attn_prefill_bf16_d512_with_blk(
                        s.encoder_mut(), dev, reg,
                        &pf_q_perm, &pf_k_perm, &pf_v_perm,
                        Some(&global_mask),
                        Some(&blk_global),
                        &mut pf_sdpa_out_perm,
                        &mlx_native::ops::flash_attn_prefill_d512::FlashAttnPrefillParams {
                            n_heads: nh as u32,
                            n_kv_heads: nkv as u32,
                            head_dim: hd as u32,
                            seq_len_q: seq_len as u32,
                            seq_len_k: seq_len as u32,
                            batch: 1,
                            scale: 1.0,
                            do_causal: false,
                        },
                    ).map_err(|e| anyhow::anyhow!("batched global flash_attn_prefill L{layer_idx}: {e}"))?;
                }
                // Wave P4.0 — close the FA-only session and accumulate
                // wall-clock time (commit_and_wait blocks until the GPU
                // finishes, so the measurement bounds true GPU work).
                if let Some(t0) = fa_start {
                    s.finish().map_err(|e| anyhow::anyhow!("FA-profile post-finish L{layer_idx}: {e}"))?;
                    let elapsed_ns = t0.elapsed().as_nanos() as u64;
                    if is_sliding {
                        PROFILE_FA_SW_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
                        PROFILE_FA_SW_COUNT.fetch_add(1, Ordering::Relaxed);
                    } else {
                        PROFILE_FA_GL_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
                        PROFILE_FA_GL_COUNT.fetch_add(1, Ordering::Relaxed);
                    }
                    s = exec.begin().map_err(|e| anyhow::anyhow!("FA-profile resume L{layer_idx}: {e}"))?;
                }

                // 7. Fused permute + bf16→f32 cast on sdpa_out:
                //    [n_heads, seq_len, hd] bf16 → [seq_len, n_heads, hd] f32.
                //    Wave P4.10 — replaces the prior two-pass sequence
                //    (permute_021_bf16 → cast_bf16_to_f32) with a single
                //    dispatch.  Halves global-memory traffic on the
                //    intermediate buffer (was 2 reads + 2 writes per layer,
                //    now 1 read + 1 write) and removes one dispatch per
                //    layer (30/prefill).
                s.barrier_between(
                    &[&pf_sdpa_out_perm],
                    &[&pf_sdpa_out],
                );
                mlx_native::ops::transpose::permute_021_bf16_to_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_sdpa_out_perm, &pf_sdpa_out,
                    nh, seq_len, hd,
                ).map_err(|e| anyhow::anyhow!("batched permute+cast SDPA L{layer_idx}: {e}"))?;

                // 8. O-proj (m = seq_len): [seq_len, nh*hd] → [seq_len, hs]
                s.barrier_between(
                    &[&pf_sdpa_out, &self.layers[layer_idx].attn.o_proj.buffer],
                    &[&pf_attn_out],
                );
                let o_t0 = if std::env::var("HF2Q_PROFILE_MM").is_ok() {
                    s.finish().map_err(|e| anyhow::anyhow!("MM-profile pre-finish (O) L{layer_idx}: {e}"))?;
                    let t0 = std::time::Instant::now();
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MM-profile begin (O) L{layer_idx}: {e}"))?;
                    Some(t0)
                } else { None };
                dispatch_qmatmul(&mut s, reg, dev, &pf_sdpa_out,
                    &self.layers[layer_idx].attn.o_proj,
                    &mut pf_attn_out, seq_len as u32)?;
                if let Some(t0) = o_t0 {
                    s.finish().map_err(|e| anyhow::anyhow!("MM-profile post-finish (O) L{layer_idx}: {e}"))?;
                    PROFILE_MM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_MM_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MM-profile resume (O) L{layer_idx}: {e}"))?;
                }

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

                // ------------------------------------------------------------
                // ADR-011 Phase 3 Wave P3b.2 — merge KV cache copy into
                // session A.  Pre-P3b.2 this ran as a separate "Session C"
                // with its own `exec.begin()` / `s.finish()` pair, costing
                // one commit_and_wait per layer (30 per prefill).  The
                // copy's inputs (pf_k_normed, pf_v_normed) are already
                // session-A internals; its outputs (dense_kvs_vec[layer].{k,v})
                // are not read again within this prefill.  Run as the final
                // dispatches of session A and the barrier-between check keeps
                // it correctly ordered against the V-norm that produced the
                // source buffers.
                // ------------------------------------------------------------
                let layer_cap = dense_kvs_vec[layer_idx].capacity;
                if !dense_kvs_vec[layer_idx].is_sliding && seq_len > layer_cap {
                    anyhow::bail!(
                        "batched prefill L{}: seq_len={} exceeds global dense cap={} — \
                         increase linear_capacity allocation",
                        layer_idx, seq_len, layer_cap);
                }
                let n_copy = seq_len.min(layer_cap);
                let src_tok_offset = (seq_len - n_copy) as u32;
                let dst_seq_pos_start = src_tok_offset;
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
                        layer_cap as u32,
                        dst_seq_pos_start, n_copy as u32,
                        src_tok_offset,
                    ).map_err(|e| anyhow::anyhow!("batched K cache copy L{layer_idx}: {e}"))?;
                    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_v_normed,
                        &dense_kvs_vec[layer_idx].v,
                        nkv as u32, hd as u32,
                        layer_cap as u32,
                        dst_seq_pos_start, n_copy as u32,
                        src_tok_offset,
                    ).map_err(|e| anyhow::anyhow!("batched V cache copy L{layer_idx}: {e}"))?;
                } else {
                    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_k_normed,
                        &dense_kvs_vec[layer_idx].k,
                        nkv as u32, hd as u32,
                        layer_cap as u32,
                        dst_seq_pos_start, n_copy as u32,
                        src_tok_offset,
                    ).map_err(|e| anyhow::anyhow!("batched K cache copy L{layer_idx}: {e}"))?;
                    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_v_normed,
                        &dense_kvs_vec[layer_idx].v,
                        nkv as u32, hd as u32,
                        layer_cap as u32,
                        dst_seq_pos_start, n_copy as u32,
                        src_tok_offset,
                    ).map_err(|e| anyhow::anyhow!("batched V cache copy L{layer_idx}: {e}"))?;
                }

                // ADR-011 Phase 3 Wave P3b.3 — MLP + MoE continue in the
                // same session `s` as attention + KV copy above.  Pre-P3b.3
                // this was a separate "Session B" with its own
                // `exec.begin()` / `s.finish()` pair — 30 CPU/GPU syncs
                // per prefill.  The sole cross-boundary input is
                // pf_residual (written by the post-attn fused_norm_add
                // above, read by the three pre-FF norms below); smart
                // barrier_between keeps it correctly ordered without
                // needing a CPU-visible sync.

                // ================================================================
                // MLP + MoE (merged into session A — Wave P3b.3)
                // ================================================================

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
                let gur_t0 = if std::env::var("HF2Q_PROFILE_MM").is_ok() {
                    s.finish().map_err(|e| anyhow::anyhow!("MM-profile pre-finish (gur) L{layer_idx}: {e}"))?;
                    let t0 = std::time::Instant::now();
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MM-profile begin (gur) L{layer_idx}: {e}"))?;
                    Some(t0)
                } else { None };
                dispatch_qmatmul(&mut s, reg, dev, &pf_norm_out,
                    &self.layers[layer_idx].mlp.gate_proj,
                    &mut pf_mlp_gate, seq_len as u32)?;
                dispatch_qmatmul(&mut s, reg, dev, &pf_norm_out,
                    &self.layers[layer_idx].mlp.up_proj,
                    &mut pf_mlp_up, seq_len as u32)?;
                dispatch_qmatmul(&mut s, reg, dev, &pf_router_norm_out,
                    &self.layers[layer_idx].moe.router_proj,
                    &mut pf_router_logits, seq_len as u32)?;
                if let Some(t0) = gur_t0 {
                    s.finish().map_err(|e| anyhow::anyhow!("MM-profile post-finish (gur) L{layer_idx}: {e}"))?;
                    PROFILE_MM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_MM_COUNT.fetch_add(3, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MM-profile resume (gur) L{layer_idx}: {e}"))?;
                }

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
                let dn_t0 = if std::env::var("HF2Q_PROFILE_MM").is_ok() {
                    s.finish().map_err(|e| anyhow::anyhow!("MM-profile pre-finish (mlp_dn) L{layer_idx}: {e}"))?;
                    let t0 = std::time::Instant::now();
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MM-profile begin (mlp_dn) L{layer_idx}: {e}"))?;
                    Some(t0)
                } else { None };
                dispatch_qmatmul(&mut s, reg, dev, &pf_mlp_fused,
                    &self.layers[layer_idx].mlp.down_proj,
                    &mut pf_mlp_down, seq_len as u32)?;
                if let Some(t0) = dn_t0 {
                    s.finish().map_err(|e| anyhow::anyhow!("MM-profile post-finish (mlp_dn) L{layer_idx}: {e}"))?;
                    PROFILE_MM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_MM_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MM-profile resume (mlp_dn) L{layer_idx}: {e}"))?;
                }

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
                let profile_moe = std::env::var("HF2Q_PROFILE_MOE").is_ok();
                let moe_gu_t0 = if profile_moe {
                    s.finish().map_err(|e| anyhow::anyhow!("MoE-profile pre-finish (gu) L{layer_idx}: {e}"))?;
                    let t0 = std::time::Instant::now();
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MoE-profile begin (gu) L{layer_idx}: {e}"))?;
                    Some(t0)
                } else { None };
                s.quantized_matmul_id_ggml_pooled(
                    reg, dev,
                    &pf_moe_norm_out,
                    self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap(),
                    &pf_expert_ids,
                    &mut pf_moe_gate_up,
                    &mut pf_moe_mm_scratch,
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
                if let Some(t0) = moe_gu_t0 {
                    s.finish().map_err(|e| anyhow::anyhow!("MoE-profile post-finish (gu) L{layer_idx}: {e}"))?;
                    PROFILE_MOE_GU_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_MOE_GU_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MoE-profile resume (gu) L{layer_idx}: {e}"))?;
                }

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
                let moe_dn_t0 = if std::env::var("HF2Q_PROFILE_MOE").is_ok() {
                    s.finish().map_err(|e| anyhow::anyhow!("MoE-profile pre-finish (dn) L{layer_idx}: {e}"))?;
                    let t0 = std::time::Instant::now();
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MoE-profile begin (dn) L{layer_idx}: {e}"))?;
                    Some(t0)
                } else { None };
                s.quantized_matmul_id_ggml_pooled(
                    reg, dev,
                    &pf_moe_swiglu,
                    self.layers[layer_idx].moe.stacked_down.as_ref().unwrap(),
                    &pf_expert_ids,
                    &mut pf_moe_down,
                    &mut pf_moe_mm_scratch,
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
                if let Some(t0) = moe_dn_t0 {
                    s.finish().map_err(|e| anyhow::anyhow!("MoE-profile post-finish (dn) L{layer_idx}: {e}"))?;
                    PROFILE_MOE_DN_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_MOE_DN_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MoE-profile resume (dn) L{layer_idx}: {e}"))?;
                }

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
                if std::env::var("HF2Q_PROFILE_LAYERS").is_ok() {
                    let kind = if is_sliding { "SW" } else { "GL" };
                    eprintln!("[LAYER_TIME] L{:02} {} {}us", layer_idx, kind,
                        layer_start.elapsed().as_micros());
                }
            }

            // ADR-011 Phase 3 Wave P3b.2 — Session C ("write K,V to dense
            // cache") was merged into the tail of Session A.  Only the
            // host-side metadata update survives here: it's pure CPU state
            // used by subsequent decode to locate the cache, and it doesn't
            // need to wait on the GPU copy (the copy is still in flight on
            // its command buffer, but decode won't run until that buffer
            // commits at the end of the prefill).
            self.kv_caches[layer_idx].write_pos = seq_len;
            self.kv_caches[layer_idx].seq_len = seq_len.min(self.kv_caches[layer_idx].capacity);

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

        // Wave P4.0 — dump per-kernel-category totals when profiling is on.
        let prefill_ms = elapsed.as_secs_f64() * 1000.0;
        if std::env::var("HF2Q_PROFILE_FA").is_ok() {
            let sw_ns = PROFILE_FA_SW_NS.swap(0, Ordering::Relaxed);
            let sw_n = PROFILE_FA_SW_COUNT.swap(0, Ordering::Relaxed);
            let gl_ns = PROFILE_FA_GL_NS.swap(0, Ordering::Relaxed);
            let gl_n = PROFILE_FA_GL_COUNT.swap(0, Ordering::Relaxed);
            let total_ms = (sw_ns + gl_ns) as f64 / 1_000_000.0;
            eprintln!(
                "[FA_PROFILE] D=256 (SW): {} calls, {:.2} ms total ({:.3} ms/call) | \
                 D=512 (GL): {} calls, {:.2} ms total ({:.3} ms/call) | \
                 FA total: {:.2} ms ({:.1}% of prefill {:.1} ms)",
                sw_n, sw_ns as f64 / 1_000_000.0,
                if sw_n > 0 { (sw_ns as f64 / sw_n as f64) / 1_000_000.0 } else { 0.0 },
                gl_n, gl_ns as f64 / 1_000_000.0,
                if gl_n > 0 { (gl_ns as f64 / gl_n as f64) / 1_000_000.0 } else { 0.0 },
                total_ms,
                if prefill_ms > 0.0 { 100.0 * total_ms / prefill_ms } else { 0.0 },
                prefill_ms,
            );
        }
        if std::env::var("HF2Q_PROFILE_MOE").is_ok() {
            let gu_ns = PROFILE_MOE_GU_NS.swap(0, Ordering::Relaxed);
            let gu_n = PROFILE_MOE_GU_COUNT.swap(0, Ordering::Relaxed);
            let dn_ns = PROFILE_MOE_DN_NS.swap(0, Ordering::Relaxed);
            let dn_n = PROFILE_MOE_DN_COUNT.swap(0, Ordering::Relaxed);
            let total_ms = (gu_ns + dn_ns) as f64 / 1_000_000.0;
            eprintln!(
                "[MOE_PROFILE] gate_up: {} calls, {:.2} ms total ({:.3} ms/call) | \
                 down: {} calls, {:.2} ms total ({:.3} ms/call) | \
                 MoE total: {:.2} ms ({:.1}% of prefill {:.1} ms)",
                gu_n, gu_ns as f64 / 1_000_000.0,
                if gu_n > 0 { (gu_ns as f64 / gu_n as f64) / 1_000_000.0 } else { 0.0 },
                dn_n, dn_ns as f64 / 1_000_000.0,
                if dn_n > 0 { (dn_ns as f64 / dn_n as f64) / 1_000_000.0 } else { 0.0 },
                total_ms,
                if prefill_ms > 0.0 { 100.0 * total_ms / prefill_ms } else { 0.0 },
                prefill_ms,
            );
        }
        if std::env::var("HF2Q_PROFILE_MM").is_ok() {
            let mm_ns = PROFILE_MM_NS.swap(0, Ordering::Relaxed);
            let mm_n = PROFILE_MM_COUNT.swap(0, Ordering::Relaxed);
            let total_ms = mm_ns as f64 / 1_000_000.0;
            eprintln!(
                "[MM_PROFILE] dense qmatmul: {} calls, {:.2} ms total ({:.3} ms/call) ({:.1}% of prefill {:.1} ms)",
                mm_n, total_ms,
                if mm_n > 0 { total_ms / mm_n as f64 } else { 0.0 },
                if prefill_ms > 0.0 { 100.0 * total_ms / prefill_ms } else { 0.0 },
                prefill_ms,
            );
        }

        // Store dense KV buffers so forward_decode can use them
        self.dense_kvs = Some(dense_kvs_vec);
        self.dense_sdpa_tmp = Some(sdpa_tmp);

        Ok(first_token)
    }
}
