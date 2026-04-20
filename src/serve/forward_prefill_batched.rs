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
    dispatch_rms_norm_unit_perhead_dual_perm,
};
use super::gpu::GpuContext;

// Wave P4.0 — env-gated per-kernel GPU-time profiling.  Enabled via
// HF2Q_PROFILE_FA / HF2Q_PROFILE_MOE / HF2Q_PROFILE_MM to break out the
// contribution of each kernel category.  Each adds 2 commit_and_wait per
// dispatch to isolate its session, so prefill total throughput will
// REGRESS substantially when on — never enable in production.
//
// Wave P4.17 — HF2Q_PROFILE_BUCKETS=1 is a super-flag that turns on
// per-op wall-clock isolation for EVERY category in the prefill forward
// pass: setup (embed, masks, blk), per-layer (pre-attn norm, QKV, head
// norm+RoPE, FA, post-FA permute, O, post-attn norm-add, KV copy, triple
// norm, MLP gate/up/router, gelu+routing, MLP down, MoE gate_up, MoE
// swiglu, MoE down, MoE wsum+dnorm+add, end-of-layer norm-add-scalar),
// and head session (copy+final_norm, lm_head, softcap, argmax).  With
// every op isolated, the sum of buckets accounts for the total prefill
// wall-clock (modulo CPU encode overhead), so the "unaccounted" gap to
// llama.cpp can be attributed category by category.
static PROFILE_FA_SW_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_FA_SW_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_FA_GL_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_FA_GL_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_MOE_GU_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MOE_GU_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_MOE_DN_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MOE_DN_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_MOE_POST_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MOE_POST_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_NORM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_NORM_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_PERMUTE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_PERMUTE_COUNT: AtomicU64 = AtomicU64::new(0);

// MM — split into four per-site atomics (was a single PROFILE_MM_NS).
// HF2Q_PROFILE_MM continues to work: its emission block reads the sum.
// HF2Q_PROFILE_BUCKETS reports them individually.
static PROFILE_QKV_MM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_QKV_MM_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_O_MM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_O_MM_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_MLP_GUR_MM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MLP_GUR_MM_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_MLP_DN_MM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MLP_DN_MM_COUNT: AtomicU64 = AtomicU64::new(0);

// Wave P4.17 bucket atomics — all gated on HF2Q_PROFILE_BUCKETS=1.
static PROFILE_B_EMBED_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_EMBED_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_MASK_SW_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_MASK_SW_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_MASK_GL_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_MASK_GL_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_BLK_SW_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_BLK_SW_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_BLK_GL_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_BLK_GL_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_PRE_ATTN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_PRE_ATTN_NORM_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_HEAD_NORM_ROPE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_HEAD_NORM_ROPE_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_POST_FA_PERMUTE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_POST_FA_PERMUTE_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_POST_ATTN_NORM_ADD_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_POST_ATTN_NORM_ADD_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_KV_COPY_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_KV_COPY_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_TRIPLE_NORM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_TRIPLE_NORM_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_GELU_MUL_ROUTING_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_GELU_MUL_ROUTING_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_MOE_WSUM_ADD_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_MOE_WSUM_ADD_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_END_LAYER_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_END_LAYER_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_FINAL_NORM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_FINAL_NORM_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_LM_HEAD_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_LM_HEAD_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_SOFTCAP_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_SOFTCAP_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_ARGMAX_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_ARGMAX_COUNT: AtomicU64 = AtomicU64::new(0);

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

        // Metal-1 — programmatic GPU capture.  Gated on
        // HF2Q_METAL_CAPTURE=path.gputrace.  Requires process env
        // MTL_CAPTURE_ENABLED=1 (Metal's own capture enablement).
        // HF2Q_METAL_CAPTURE_LAYERS="start-end" (inclusive, default
        // "0-0") bounds the capture to a single layer so the resulting
        // .gputrace is small enough to open in Xcode.  Start-capture is
        // deferred to the layer loop (see the matching begin/end calls
        // below).  We resolve paths + destination here.
        let capture_path = std::env::var("HF2Q_METAL_CAPTURE").ok();
        let (capture_layer_start, capture_layer_end) = std::env::var("HF2Q_METAL_CAPTURE_LAYERS")
            .ok()
            .and_then(|s| {
                let mut it = s.splitn(2, '-');
                let a = it.next()?.parse::<usize>().ok()?;
                let b = it.next()?.parse::<usize>().ok()?;
                Some((a, b))
            })
            .unwrap_or((0, 0));
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

        // Option 3 (Phase D) — HF2Q_NO_FA=1 swaps the flash-attention
        // prefill path for a tensor-mm attention path that mirrors
        // llama.cpp's `-fa 0` fast path:
        //   1. cast Q bf16 -> f32 (src1 dtype for our bf16 tensor-mm)
        //   2. Q @ K^T via hf2q_dense_mm_bf16_f32_tensor -> kq f32 [nh, seq, seq]
        //   3. scale_mask_softmax_f32 (scale is pre-applied in Q's norm
        //      weights; passes 1.0; reuses the existing bf16 sliding /
        //      global masks from the flash-attn path)
        //   4. transpose_last2_bf16 on V: [nkv, seq, hd] -> [nkv, hd, seq]
        //   5. scores @ V^T via hf2q_dense_mm_bf16_f32_tensor -> attn f32 [nh, seq, hd]
        //   6. permute_021_f32 -> pf_sdpa_out f32 [seq, nh, hd]
        // The FA path stays live; env flag swap lets us A/B without
        // code churn.  The non-FA path requires ~555 MB of extra
        // intermediate buffers at seq_len=2455; only allocate them if
        // the flag is set so the default-path footprint is unchanged.
        let use_no_fa = std::env::var("HF2Q_NO_FA").is_ok();

        // Wave P4.17 — super-flag: per-op isolation for bucket attribution.
        // When on, every dispatch is bracketed by s.finish()/s = exec.begin()
        // pairs so its wall-clock is measured; the individual bucket atomics
        // accumulate ns and counts.  Prefill throughput REGRESSES under this
        // flag (extra finish/begin per op ≈ 50-200 µs each, × ~15 ops/layer
        // × 30 layers ≈ 100-150 ms of pure overhead on top of normal work),
        // so it's a profiling-only knob.
        let profile_buckets_on = std::env::var("HF2Q_PROFILE_BUCKETS").is_ok();

        eprintln!("Batched prefill: KV={:?}, seq_len={}, path={}{}", kv_dtype, seq_len,
                  if use_no_fa { "tensor-mm (non-FA)" } else { "flash-attn" },
                  if profile_buckets_on { " [BUCKET_PROFILE]" } else { "" });

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
        let mut pf_hidden = alloc_f32(seq_len * hs, "pf_hidden")?;
        let pf_residual = alloc_f32(seq_len * hs, "pf_residual")?;
        let pf_norm_out = alloc_f32(seq_len * hs, "pf_norm_out")?;
        let pf_moe_norm_out = alloc_f32(seq_len * hs, "pf_moe_norm_out")?;
        let pf_router_norm_out = alloc_f32(seq_len * hs, "pf_router_norm_out")?;
        let mut pf_attn_out = alloc_f32(seq_len * hs, "pf_attn_out")?;
        // Wave P4.14 — pf_mlp_down_out (the intermediate normed MLP-down
        // buffer between post-FF norm 1 and the MoE wsum+add) is no longer
        // needed: the fused fused_moe_wsum_dnorm_add_f32 dispatch absorbs
        // both norms and the add into one pass.

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
        // Wave P4.15 — pf_q_normed_bf16 and pf_k_normed_bf16 removed.
        // The head_norm+RoPE dispatch now writes bf16 directly at permuted
        // layout into pf_q_perm and pf_k_perm, fusing the permute_021_bf16
        // pre-FA dispatch.
        // Wave P4.16 — pf_v_normed_bf16 removed.  V's dual-norm now writes
        // bf16 directly at permuted [nkv, seq_len, hd] layout into pf_v_perm.
        let pf_q_perm = alloc_bf16(nh * seq_len * max_hd, "pf_q_perm")?;
        let pf_k_perm = alloc_bf16(max_nkv * seq_len * max_hd, "pf_k_perm")?;
        let pf_v_perm = alloc_bf16(max_nkv * seq_len * max_hd, "pf_v_perm")?;
        let mut pf_sdpa_out_perm = alloc_bf16(nh * seq_len * max_hd, "pf_sdpa_out_perm")?;

        // HF2Q_NO_FA path buffers — only allocated when the env flag is
        // set.  pf_kq is the dominant footprint at seq_len=2455:
        //   nh × seq × seq × 4 = 16 × 2455² × 4 ≈ 386 MB.
        // Q / V-transposed / attn-out are each ~40-80 MB.
        let pf_q_perm_f32: Option<MlxBuffer> = if use_no_fa {
            Some(alloc_f32(nh * seq_len * max_hd, "pf_q_perm_f32")?)
        } else { None };
        let mut pf_kq: Option<MlxBuffer> = if use_no_fa {
            Some(alloc_f32(nh * seq_len * seq_len, "pf_kq")?)
        } else { None };
        let pf_v_perm_t: Option<MlxBuffer> = if use_no_fa {
            Some(alloc_bf16(max_nkv * max_hd * seq_len, "pf_v_perm_t")?)
        } else { None };
        let mut pf_attn_f32: Option<MlxBuffer> = if use_no_fa {
            Some(alloc_f32(nh * seq_len * max_hd, "pf_attn_f32")?)
        } else { None };
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

            // Wave P4.17 — each setup sub-step gets its own bucket timing
            // when HF2Q_PROFILE_BUCKETS=1.  The five sub-steps (embed,
            // sliding mask, global mask, sliding blk, global blk) are
            // independent in the default path (no barrier_between between
            // embed and the masks; the masks and blks serialize per-pair
            // via explicit barriers).  Splitting them adds 4 extra commit+
            // wait transitions (~50-200 µs each) under the profile flag;
            // normal runs keep the single session.

            // 1. Embedding: gather prompt rows from embed_weight into pf_hidden
            //    and scale by sqrt(hidden_size).
            //
            // Wave P4.18 — CPU-side gather.  The GPU kernel version cost
            // ~48 ms wall-clock at pp2455 on M5 Max (measured 2026-04-20
            // via HF2Q_SKIP_EMBED: prefill drops from 798 ms → 750 ms
            // when the GPU embed dispatch is skipped).  That was 50× more
            // than the ~1 ms a memcpy-speed bound would predict — the
            // op is a simple scatter/gather over 30 MB.  Root cause:
            // spawning 7.5 M GPU threads (one per output element) for a
            // purely-memory-bound op has high per-thread scheduling
            // latency that dominates the tiny per-thread work.
            //
            // The embed_weight buffer is StorageModeShared (CPU/GPU
            // unified) so the CPU can write pf_hidden directly; the
            // next GPU op (layer 0 pre-attn norm) will see the CPU
            // writes without any flush since MTLResourceOptions::
            // StorageModeShared is coherent.
            //
            // Per-row memcpy pattern: 2455 rows × 12 KB each.  On M5 Max
            // single-core memory copy runs at ~25 GB/s, so 30 MB → ~1.2 ms
            // pure data; with row-loop overhead, measure <5 ms in practice.
            // Saves ~45 ms vs the GPU dispatch.
            let t0_embed = if profile_buckets_on {
                Some(std::time::Instant::now())
            } else { None };
            {
                let scale = (hs as f32).sqrt();
                let embed_f32: &[f32] = self.embed_weight.as_slice()
                    .map_err(|e| anyhow::anyhow!("batched embed read: {e}"))?;
                let out: &mut [f32] = pf_hidden.as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("batched pf_hidden write: {e}"))?;
                // Two-pass: memcpy then scale.  copy_from_slice compiles
                // to a full-width memcpy (NEON on arm64) that streams at
                // ~50 GB/s; the subsequent scale loop auto-vectorizes to
                // NEON fmul, running at ~40 GB/s.  Total ~2-3 ms for 30 MB
                // on M5 Max vs the ~25 ms a per-element iterator-chain
                // version would spend on dependent-load stalls.
                for (tok_idx, &tok_id) in prompt_tokens.iter().enumerate() {
                    let src_off = (tok_id as usize) * hs;
                    let dst_off = tok_idx * hs;
                    out[dst_off..dst_off + hs]
                        .copy_from_slice(&embed_f32[src_off..src_off + hs]);
                }
                // Scale in-place; single contiguous pass over pf_hidden
                // hits the CPU prefetcher cleanly.
                for v in out[..seq_len * hs].iter_mut() {
                    *v *= scale;
                }
            }
            if let Some(t0) = t0_embed {
                // No s.finish() needed — no GPU dispatch was made; just
                // record the CPU-side wall-clock of the scatter+scale.
                PROFILE_B_EMBED_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                PROFILE_B_EMBED_COUNT.fetch_add(1, Ordering::Relaxed);
            }

            // 2. Sliding-window causal mask — reused across all 25 sliding layers.
            let t0_mask_sw = if profile_buckets_on {
                Some(std::time::Instant::now())
            } else { None };
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
            if let Some(t0) = t0_mask_sw {
                s.finish().map_err(|e| anyhow::anyhow!("bucket mask_sw finish: {e}"))?;
                PROFILE_B_MASK_SW_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                PROFILE_B_MASK_SW_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec.begin().map_err(|e| anyhow::anyhow!("bucket mask_sw resume: {e}"))?;
            }

            // 3. Global causal mask — reused across all 5 global layers.
            let t0_mask_gl = if profile_buckets_on {
                Some(std::time::Instant::now())
            } else { None };
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
            if let Some(t0) = t0_mask_gl {
                s.finish().map_err(|e| anyhow::anyhow!("bucket mask_gl finish: {e}"))?;
                PROFILE_B_MASK_GL_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                PROFILE_B_MASK_GL_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec.begin().map_err(|e| anyhow::anyhow!("bucket mask_gl resume: {e}"))?;
            }

            // 4. Tile-skip classifiers — read mask, write blk bytes.
            let t0_blk_sw = if profile_buckets_on {
                Some(std::time::Instant::now())
            } else { None };
            s.barrier_between(
                &[&sliding_mask],
                &[&blk_sliding],
            );
            dispatch_flash_attn_prefill_blk(
                s.encoder_mut(), dev, reg,
                &sliding_mask, &blk_sliding,
                &blk_sliding_params,
            ).map_err(|e| anyhow::anyhow!("dispatch blk_sliding: {e}"))?;
            if let Some(t0) = t0_blk_sw {
                s.finish().map_err(|e| anyhow::anyhow!("bucket blk_sw finish: {e}"))?;
                PROFILE_B_BLK_SW_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                PROFILE_B_BLK_SW_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec.begin().map_err(|e| anyhow::anyhow!("bucket blk_sw resume: {e}"))?;
            }

            let t0_blk_gl = if profile_buckets_on {
                Some(std::time::Instant::now())
            } else { None };
            s.barrier_between(
                &[&global_mask],
                &[&blk_global],
            );
            dispatch_flash_attn_prefill_blk(
                s.encoder_mut(), dev, reg,
                &global_mask, &blk_global,
                &blk_global_params,
            ).map_err(|e| anyhow::anyhow!("dispatch blk_global: {e}"))?;
            if let Some(t0) = t0_blk_gl {
                s.finish().map_err(|e| anyhow::anyhow!("bucket blk_gl finish: {e}"))?;
                PROFILE_B_BLK_GL_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                PROFILE_B_BLK_GL_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec.begin().map_err(|e| anyhow::anyhow!("bucket blk_gl resume: {e}"))?;
            }

            // Wave P4.18 — setup-session async commit.  Matches the
            // layer-boundary async-commit pattern (commit 9091b8c): the
            // setup outputs (sliding_mask, global_mask, blk_sliding,
            // blk_global, pf_hidden, pf_positions, pf_token_ids) are
            // written to shared buffers and read by layer 0's first
            // dispatches.  Metal guarantees in-order execution of CBs
            // submitted to the same queue, so those reads see the setup
            // writes without requiring a CPU-side wait.  The ~30-45 ms
            // of GPU idle at the setup→layer0 boundary (from the 2026-
            // 04-20 bucket profile) collapses: layer 0's CPU-encode
            // phase runs concurrently with the tail of setup's GPU work.
            //
            // Same fallback escape hatches as the layer boundary:
            //   * HF2Q_BATCHED_DUMP set — dump path CPU-reads setup
            //     outputs indirectly; keep sync for safety.
            //   * HF2Q_PROFILE_BUCKETS set — per-sub-step sync above
            //     has already flushed; a final finish is needed to
            //     close the last sub-step's measurement.
            //   * HF2Q_SYNC_PER_LAYER set — debug knob.
            let sync_setup = INVESTIGATION_ENV.batched_dump.is_some()
                || std::env::var("HF2Q_PROFILE_LAYERS").is_ok()
                || std::env::var("HF2Q_SYNC_PER_LAYER").is_ok()
                || profile_buckets_on;
            if sync_setup {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("batched setup finish: {e}"))?;
            } else {
                let _committed = s.commit();
                drop(_committed);
            }
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

        // Metal-1: tracks whether the GPU capture was successfully
        // started during the layer loop — needed so the teardown at the
        // end of the function knows whether to call stop_capture.
        let mut capture_active: bool = false;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Metal-1 — begin/end programmatic GPU capture around the
            // configured layer range (HF2Q_METAL_CAPTURE_LAYERS).
            if let Some(p) = capture_path.as_ref() {
                if layer_idx == capture_layer_start && !capture_active {
                    use mlx_native::metal::{CaptureDescriptor, MTLCaptureDestination};
                    let desc = CaptureDescriptor::new();
                    desc.set_capture_device(dev.metal_device());
                    desc.set_destination(MTLCaptureDestination::GpuTraceDocument);
                    desc.set_output_url(std::path::Path::new(p));
                    let mgr = mlx_native::metal::CaptureManager::shared();
                    match mgr.start_capture(&desc) {
                        Ok(()) => {
                            eprintln!(
                                "[METAL_CAPTURE] started at layer {} (through {}), writing to {}",
                                capture_layer_start, capture_layer_end, p
                            );
                            capture_active = true;
                        }
                        Err(e) => {
                            eprintln!(
                                "[METAL_CAPTURE] start_capture failed: {} \
                                 (make sure MTL_CAPTURE_ENABLED=1 is set in the env)",
                                e
                            );
                        }
                    }
                }
            }

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
                let t0_pre_attn_norm = if profile_buckets_on {
                    Some(std::time::Instant::now())
                } else { None };
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
                if let Some(t0) = t0_pre_attn_norm {
                    s.finish().map_err(|e| anyhow::anyhow!("bucket pre_attn_norm finish L{layer_idx}: {e}"))?;
                    PROFILE_B_PRE_ATTN_NORM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_B_PRE_ATTN_NORM_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("bucket pre_attn_norm resume L{layer_idx}: {e}"))?;
                }

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
                let profile_mm = std::env::var("HF2Q_PROFILE_MM").is_ok() || profile_buckets_on;
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
                    PROFILE_QKV_MM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_QKV_MM_COUNT.fetch_add(if v_is_k { 2 } else { 3 }, Ordering::Relaxed);
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
                // Wave P4.15 — head_norm+RoPE writes bf16 output DIRECTLY at
                // permuted layout [n_heads, seq_len, head_dim] into pf_q_perm/
                // pf_k_perm.  Eliminates the post-norm permute_021_bf16
                // dispatch for Q and K (60 dispatches/prefill saved) and the
                // intermediate pf_q_normed_bf16/pf_k_normed_bf16 buffers
                // (~50 MB at pp2455 × 30 = 1.5 GB of memory traffic).
                let t0_head_norm_rope = if profile_buckets_on {
                    Some(std::time::Instant::now())
                } else { None };
                s.barrier_between(
                    &[&pf_q, &pf_k],
                    &[&pf_q_normed, &pf_k_normed,
                      &pf_q_perm, &pf_k_perm],
                );
                // D.1 — in HF2Q_NO_FA mode, the head-norm+RoPE kernel
                // also co-writes f32 at the permuted [nh, seq, hd]
                // layout into pf_q_perm_f32, eliminating the separate
                // permute_021_bf16_to_f32 cast dispatch that otherwise
                // runs every layer to produce the src1 of the tensor-mm
                // Q@K^T.
                mlx_native::ops::fused_head_norm_rope::
                    dispatch_fused_head_norm_rope_batch_f32_with_bf16_f32_perm(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_q,
                    &pf_q_normed,
                    Some(&pf_q_perm),            // P4.15 bf16 permuted
                    pf_q_perm_f32.as_ref(),      // D.1 f32 permuted (None outside HF2Q_NO_FA)
                    Some(&self.layers[layer_idx].attn.q_norm_weight),
                    &pf_positions,
                    ff_gpu,
                    nh as u32, hd as u32, half_rope,
                    seq_len as u32,
                    eps, theta,
                    true,                // bf16_permuted
                ).map_err(|e| anyhow::anyhow!("batched Q norm+RoPE+permuted bf16/f32 L{layer_idx}: {e}"))?;
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_batch_f32_with_bf16(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_k,
                    &pf_k_normed,
                    Some(&pf_k_perm),    // P4.15: write bf16 at permuted [head, token, i]
                    Some(&self.layers[layer_idx].attn.k_norm_weight),
                    &pf_positions,
                    ff_gpu,
                    nkv as u32, hd as u32, half_rope,
                    seq_len as u32,
                    eps, theta,
                    true,                // bf16_permuted
                ).map_err(|e| anyhow::anyhow!("batched K norm+RoPE+permuted bf16 L{layer_idx}: {e}"))?;

                // 4. V norm (unit RMS, no RoPE, per-head across seq_len)
                //    Layout: [seq_len * nkv, hd] — treat all positions' heads as rows
                //
                // ADR-011 Phase 3 Wave P3b-tensor.3 — use the dual-output
                // variant so the kernel co-writes both pf_v_normed (f32,
                // for the KV cache copy below) and pf_v_normed_bf16 (for
                // the bf16 attention island).  Eliminates the otherwise-
                // separate f32→bf16 cast dispatch (30 dispatches /
                // prefill).
                // V norm with permuted bf16 output — Wave P4.16.
                // The kernel co-writes pf_v_normed at natural layout
                // (KV cache copy reads it) AND pf_v_perm at permuted
                // [nkv, seq_len, hd] layout (FA reads it).  Removes:
                //   - The previous separate dispatch_rms_norm_unit_
                //     perhead_dual + permute_021_bf16(V) pair
                //     (-30 dispatches/prefill).
                //   - pf_v_normed_bf16 intermediate buffer
                //     (~10 MB at pp2455 × 30 = ~300 MB of read+write
                //     traffic eliminated).
                let v_input = if v_is_k { &pf_k } else { &pf_v };
                s.barrier_between(
                    &[v_input],
                    &[&pf_v_normed, &pf_v_perm],
                );
                dispatch_rms_norm_unit_perhead_dual_perm(
                    s.encoder_mut(), reg, metal_dev,
                    v_input,
                    &pf_v_normed,
                    &pf_v_perm,
                    hd_norm_params,
                    nkv as u32,
                    seq_len as u32,
                    hd as u32,
                )?;
                if let Some(t0) = t0_head_norm_rope {
                    s.finish().map_err(|e| anyhow::anyhow!("bucket head_norm_rope finish L{layer_idx}: {e}"))?;
                    PROFILE_B_HEAD_NORM_ROPE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_B_HEAD_NORM_ROPE_COUNT.fetch_add(3, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("bucket head_norm_rope resume L{layer_idx}: {e}"))?;
                }

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
                if use_no_fa {
                    // ---- HF2Q_NO_FA path: tensor-mm attention ----
                    //
                    // Mirrors llama.cpp's `-fa 0` fast path (which on M5
                    // today measures 3410 vs 3217 for `-fa 1` at pp2455):
                    //
                    //   1. Q bf16 -> f32 (src1 dtype for our bf16 tensor-mm)
                    //   2. Q @ K^T -> kq f32  [nh, seq, seq]
                    //   3. scale-mask-softmax -> kq in-place
                    //   4. transpose_last2 V bf16 -> V_t bf16 [nkv, hd, seq]
                    //   5. scores @ V_t -> attn f32 [nh, seq, hd]
                    //   6. permute_021 attn -> pf_sdpa_out f32 [seq, nh, hd]
                    //
                    // Gemma 4 on this GGUF has no Q·K softcap (only the
                    // lm_head softcap), so step 3 skips the tanh-based
                    // logit cap that llama's graph inserts for Gemma 2/3.
                    // The mask buffer is the same bf16 sliding / global
                    // mask the FA path uses (reused verbatim).
                    let pf_q_perm_f32_ref = pf_q_perm_f32.as_ref()
                        .expect("pf_q_perm_f32 must be allocated in HF2Q_NO_FA mode");
                    let pf_kq_ref = pf_kq.as_mut()
                        .expect("pf_kq must be allocated in HF2Q_NO_FA mode");
                    let pf_v_perm_t_ref = pf_v_perm_t.as_ref()
                        .expect("pf_v_perm_t must be allocated in HF2Q_NO_FA mode");
                    let pf_attn_f32_ref = pf_attn_f32.as_mut()
                        .expect("pf_attn_f32 must be allocated in HF2Q_NO_FA mode");

                    // Step 1 removed (D.1): the Q bf16→f32 cast is fused
                    // into the head-norm+RoPE kernel via the optional
                    // output_f32_perm buffer.  pf_q_perm_f32 is already
                    // populated at permuted [nh, seq, hd] layout.

                    // Step 2: Q @ K^T via dense bf16×f32→f32 tensor-mm.
                    // src0 = K [nkv, seq, hd] bf16, src1 = Q [nh, seq, hd] f32.
                    // Output kq[h, q, k] = sum_d K[h/r2, k, d] * Q[h, q, d].
                    // r2 = nh / nkv for GQA head broadcast.
                    s.barrier_between(
                        &[&pf_k_perm, pf_q_perm_f32_ref],
                        &[pf_kq_ref],
                    );
                    mlx_native::ops::dense_mm_bf16::dense_matmul_bf16_f32_tensor(
                        s.encoder_mut(), reg, dev,
                        &pf_k_perm, pf_q_perm_f32_ref,
                        pf_kq_ref,
                        &mlx_native::ops::dense_mm_bf16::DenseMmBf16F32Params {
                            m: seq_len as u32,
                            n: seq_len as u32,
                            k: hd as u32,
                            src0_batch: nkv as u32,
                            src1_batch: nh as u32,
                        },
                    ).map_err(|e| anyhow::anyhow!("non-FA Q@K^T L{layer_idx}: {e}"))?;

                    // Step 3: scale + mask + softmax fused.  scale = 1.0
                    // because Q's norm-weight pre-scales by 1/sqrt(hd)
                    // (matches FA path's scale=1.0).
                    let mask_ref = if is_sliding { &sliding_mask } else { &global_mask };
                    s.barrier_between(
                        &[pf_kq_ref, mask_ref],
                        &[pf_kq_ref],
                    );
                    mlx_native::ops::scale_mask_softmax::dispatch_scale_mask_softmax_f32(
                        s.encoder_mut(), reg, dev,
                        pf_kq_ref, pf_kq_ref, mask_ref,
                        &mlx_native::ops::scale_mask_softmax::ScaleMaskSoftmaxParams {
                            rows: (nh * seq_len) as u32,
                            cols: seq_len as u32,
                            seq_q: seq_len as u32,
                            scale: 1.0,
                        },
                    ).map_err(|e| anyhow::anyhow!("non-FA scale_mask_softmax L{layer_idx}: {e}"))?;

                    // Step 4: transpose V [nkv, seq, hd] -> [nkv, hd, seq]
                    // so the scores@V matmul contracts on seq_kv (K-dim
                    // = inner-most dim of src0 per our kernel contract).
                    s.barrier_between(
                        &[&pf_v_perm],
                        &[pf_v_perm_t_ref],
                    );
                    mlx_native::ops::transpose::transpose_last2_bf16(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_v_perm, pf_v_perm_t_ref,
                        nkv, seq_len, hd,
                    ).map_err(|e| anyhow::anyhow!("non-FA V transpose L{layer_idx}: {e}"))?;

                    // Step 5: scores @ V_t via dense bf16×f32→f32 tensor-mm.
                    // src0 = V_t [nkv, hd, seq_kv] bf16, src1 = kq [nh, seq_q, seq_kv] f32.
                    // Output attn[h, q, d] = sum_k V_t[h/r2, d, k] * kq[h, q, k]
                    //                       = sum_k V[h/r2, k, d] * probs[h, q, k].
                    s.barrier_between(
                        &[pf_v_perm_t_ref, pf_kq_ref],
                        &[pf_attn_f32_ref],
                    );
                    mlx_native::ops::dense_mm_bf16::dense_matmul_bf16_f32_tensor(
                        s.encoder_mut(), reg, dev,
                        pf_v_perm_t_ref, pf_kq_ref,
                        pf_attn_f32_ref,
                        &mlx_native::ops::dense_mm_bf16::DenseMmBf16F32Params {
                            m: seq_len as u32,
                            n: hd as u32,
                            k: seq_len as u32,
                            src0_batch: nkv as u32,
                            src1_batch: nh as u32,
                        },
                    ).map_err(|e| anyhow::anyhow!("non-FA scores@V L{layer_idx}: {e}"))?;

                    // Step 6: permute attn [nh, seq, hd] f32 -> pf_sdpa_out
                    // [seq, nh, hd] f32 to match the O-proj input layout.
                    s.barrier_between(
                        &[pf_attn_f32_ref],
                        &[&pf_sdpa_out],
                    );
                    mlx_native::ops::transpose::permute_021_f32(
                        s.encoder_mut(), reg, metal_dev,
                        pf_attn_f32_ref, &pf_sdpa_out,
                        nh, seq_len, hd,
                    ).map_err(|e| anyhow::anyhow!("non-FA attn permute L{layer_idx}: {e}"))?;
                } else {
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
                let profile_fa = std::env::var("HF2Q_PROFILE_FA").is_ok() || profile_buckets_on;
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
                let t0_post_fa_permute = if profile_buckets_on {
                    Some(std::time::Instant::now())
                } else { None };
                s.barrier_between(
                    &[&pf_sdpa_out_perm],
                    &[&pf_sdpa_out],
                );
                mlx_native::ops::transpose::permute_021_bf16_to_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_sdpa_out_perm, &pf_sdpa_out,
                    nh, seq_len, hd,
                ).map_err(|e| anyhow::anyhow!("batched permute+cast SDPA L{layer_idx}: {e}"))?;
                if let Some(t0) = t0_post_fa_permute {
                    s.finish().map_err(|e| anyhow::anyhow!("bucket post_fa_permute finish L{layer_idx}: {e}"))?;
                    PROFILE_B_POST_FA_PERMUTE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_B_POST_FA_PERMUTE_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("bucket post_fa_permute resume L{layer_idx}: {e}"))?;
                }
                } // end of !use_no_fa branch

                // 8. O-proj (m = seq_len): [seq_len, nh*hd] → [seq_len, hs]
                s.barrier_between(
                    &[&pf_sdpa_out, &self.layers[layer_idx].attn.o_proj.buffer],
                    &[&pf_attn_out],
                );
                let o_t0 = if std::env::var("HF2Q_PROFILE_MM").is_ok() || profile_buckets_on {
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
                    PROFILE_O_MM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_O_MM_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MM-profile resume (O) L{layer_idx}: {e}"))?;
                }

                // 9. Post-attn fused norm + residual add (rows = seq_len)
                //    residual = (pre-attn hidden) + norm(attn_out, post_attn_norm)
                let t0_post_attn_norm_add = if profile_buckets_on {
                    Some(std::time::Instant::now())
                } else { None };
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
                if let Some(t0) = t0_post_attn_norm_add {
                    s.finish().map_err(|e| anyhow::anyhow!("bucket post_attn_norm_add finish L{layer_idx}: {e}"))?;
                    PROFILE_B_POST_ATTN_NORM_ADD_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_B_POST_ATTN_NORM_ADD_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("bucket post_attn_norm_add resume L{layer_idx}: {e}"))?;
                }

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
                let t0_kv_copy = if profile_buckets_on {
                    Some(std::time::Instant::now())
                } else { None };
                s.barrier_between(
                    &[&pf_k_normed, &pf_v_normed],
                    &[&dense_kvs_vec[layer_idx].k, &dense_kvs_vec[layer_idx].v],
                );
                // Wave P4.11 — fused K + V cache copy.  Both copies share
                // identical metadata + layout, only the source/dest buffers
                // differ.  One dispatch instead of two saves 30 dispatches/
                // prefill on Gemma 4.
                if use_f16_kv {
                    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16_dual(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_k_normed, &pf_v_normed,
                        &dense_kvs_vec[layer_idx].k, &dense_kvs_vec[layer_idx].v,
                        nkv as u32, hd as u32,
                        layer_cap as u32,
                        dst_seq_pos_start, n_copy as u32,
                        src_tok_offset,
                    ).map_err(|e| anyhow::anyhow!("batched KV cache copy (f16, dual) L{layer_idx}: {e}"))?;
                } else {
                    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_dual(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_k_normed, &pf_v_normed,
                        &dense_kvs_vec[layer_idx].k, &dense_kvs_vec[layer_idx].v,
                        nkv as u32, hd as u32,
                        layer_cap as u32,
                        dst_seq_pos_start, n_copy as u32,
                        src_tok_offset,
                    ).map_err(|e| anyhow::anyhow!("batched KV cache copy (f32, dual) L{layer_idx}: {e}"))?;
                }
                if let Some(t0) = t0_kv_copy {
                    s.finish().map_err(|e| anyhow::anyhow!("bucket kv_copy finish L{layer_idx}: {e}"))?;
                    PROFILE_B_KV_COPY_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_B_KV_COPY_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("bucket kv_copy resume L{layer_idx}: {e}"))?;
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

                // Pre-FF norm (for MLP), pre-FF norm 2 (for MoE input), router norm.
                //
                // Wave P4.9 — fused 3-output RMS norm: all three norms read the
                // same pf_residual input and apply different per-element
                // weights.  Using rms_norm_f32_triple computes RMS(pf_residual)
                // ONCE (instead of three times) and produces the three outputs
                // in one dispatch.  Saves 2 dispatches per layer (60/prefill)
                // and 2 reads of the [seq_len, hs] residual buffer per layer
                // (~40 MB at pp2455 × 30 layers = 1.2 GB of read traffic).
                let t0_triple_norm = if profile_buckets_on {
                    Some(std::time::Instant::now())
                } else { None };
                s.barrier_between(
                    &[&pf_residual],
                    &[&pf_norm_out, &pf_moe_norm_out, &pf_router_norm_out],
                );
                mlx_native::ops::rms_norm::dispatch_rms_norm_f32_triple(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_residual,
                    &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                    &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                    &self.layers[layer_idx].moe.router_combined_weight,
                    &pf_norm_out,
                    &pf_moe_norm_out,
                    &pf_router_norm_out,
                    &self.activations.norm_params,
                    seq_len as u32, hs as u32,
                ).map_err(|e| anyhow::anyhow!("batched pre-FF triple norm L{layer_idx}: {e}"))?;
                if let Some(t0) = t0_triple_norm {
                    s.finish().map_err(|e| anyhow::anyhow!("bucket triple_norm finish L{layer_idx}: {e}"))?;
                    PROFILE_B_TRIPLE_NORM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_B_TRIPLE_NORM_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("bucket triple_norm resume L{layer_idx}: {e}"))?;
                }

                // Dense MLP gate / up (m = seq_len); router proj (m = seq_len)
                s.barrier_between(
                    &[&pf_norm_out, &pf_router_norm_out],
                    &[&pf_mlp_gate, &pf_mlp_up, &pf_router_logits],
                );
                let gur_t0 = if std::env::var("HF2Q_PROFILE_MM").is_ok() || profile_buckets_on {
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
                    PROFILE_MLP_GUR_MM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_MLP_GUR_MM_COUNT.fetch_add(3, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MM-profile resume (gur) L{layer_idx}: {e}"))?;
                }

                // Fused GELU(gate) * up over [seq_len, intermediate]
                // + batched MoE routing over [seq_len, num_experts]
                let t0_gelu_mul_routing = if profile_buckets_on {
                    Some(std::time::Instant::now())
                } else { None };
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
                if let Some(t0) = t0_gelu_mul_routing {
                    s.finish().map_err(|e| anyhow::anyhow!("bucket gelu_mul_routing finish L{layer_idx}: {e}"))?;
                    PROFILE_B_GELU_MUL_ROUTING_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_B_GELU_MUL_ROUTING_COUNT.fetch_add(2, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("bucket gelu_mul_routing resume L{layer_idx}: {e}"))?;
                }

                // Dense MLP down
                s.barrier_between(
                    &[&pf_mlp_fused, &self.layers[layer_idx].mlp.down_proj.buffer],
                    &[&pf_mlp_down],
                );
                let dn_t0 = if std::env::var("HF2Q_PROFILE_MM").is_ok() || profile_buckets_on {
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
                    PROFILE_MLP_DN_MM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_MLP_DN_MM_COUNT.fetch_add(1, Ordering::Relaxed);
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
                let profile_moe = std::env::var("HF2Q_PROFILE_MOE").is_ok() || profile_buckets_on;
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
                let swiglu_t0 = if std::env::var("HF2Q_PROFILE_MOE_POST").is_ok() || profile_buckets_on {
                    s.finish().map_err(|e| anyhow::anyhow!("MoE-post-profile pre-finish (swiglu) L{layer_idx}: {e}"))?;
                    let t0 = std::time::Instant::now();
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MoE-post-profile begin (swiglu) L{layer_idx}: {e}"))?;
                    Some(t0)
                } else { None };
                mlx_native::ops::moe_dispatch::moe_swiglu_seq_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_moe_gate_up,
                    &pf_moe_swiglu,
                    moe_int, top_k, seq_len,
                ).map_err(|e| anyhow::anyhow!("batched MoE swiglu L{layer_idx}: {e}"))?;
                if let Some(t0) = swiglu_t0 {
                    s.finish().map_err(|e| anyhow::anyhow!("MoE-post-profile post-finish (swiglu) L{layer_idx}: {e}"))?;
                    PROFILE_MOE_POST_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_MOE_POST_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MoE-post-profile resume (swiglu) L{layer_idx}: {e}"))?;
                }

                // MoE down experts: quantized_matmul_id_ggml with n_tokens = seq_len*top_k, top_k=1
                let ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;
                s.barrier_between(
                    &[&pf_moe_swiglu, &pf_expert_ids,
                      self.layers[layer_idx].moe.stacked_down.as_ref().unwrap()],
                    &[&pf_moe_down],
                );
                let moe_dn_t0 = if std::env::var("HF2Q_PROFILE_MOE").is_ok() || profile_buckets_on {
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

                // Wave P4.14 — fully-fused post-MoE-down combine: in one
                // dispatch the kernel does
                //   normed_mlp = norm(pf_mlp_down,  post_FF_layernorm_1)
                //   weighted   = Σ_k pf_moe_down[k] * pf_routing_weights[k]
                //   normed_w   = norm(weighted,     post_FF_layernorm_2)
                //   pf_mlp_down (in-place) = normed_mlp + normed_w
                // The kernel runs two parallel sum-of-squares reductions
                // in threadgroup memory and stashes the per-row weighted
                // sum in shmem to avoid a global write+read.  Replaces
                // the previous two-dispatch sequence (RMS norm of
                // pf_mlp_down → pf_mlp_down_out, then fused_moe_wsum
                // _norm_add) — saves 1 more dispatch per layer
                // (30/prefill) and eliminates the pf_mlp_down_out
                // [seq_len, hs] write+read (~5 MB × 30 = 150 MB of
                // additional memory traffic on top of P4.13's saving).
                let t0_moe_wsum_add = if profile_buckets_on {
                    Some(std::time::Instant::now())
                } else { None };
                s.barrier_between(
                    &[&pf_moe_down, &pf_routing_weights, &pf_mlp_down],
                    &[&pf_mlp_down],
                );
                mlx_native::ops::fused_norm_add::dispatch_fused_moe_wsum_dnorm_add_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &pf_moe_down,
                    &pf_routing_weights,
                    &pf_mlp_down,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                    &pf_mlp_down,
                    hs as u32, top_k as u32, seq_len as u32, eps,
                ).map_err(|e| anyhow::anyhow!("batched fused MoE wsum+dnorm+add L{layer_idx}: {e}"))?;
                if let Some(t0) = t0_moe_wsum_add {
                    s.finish().map_err(|e| anyhow::anyhow!("bucket moe_wsum_add finish L{layer_idx}: {e}"))?;
                    PROFILE_B_MOE_WSUM_ADD_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_B_MOE_WSUM_ADD_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("bucket moe_wsum_add resume L{layer_idx}: {e}"))?;
                }

                // End-of-layer: output = (residual + norm(mlp_down, post_feedforward_layernorm)) * scalar
                let t0_end_layer = if profile_buckets_on {
                    Some(std::time::Instant::now())
                } else { None };
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
                if let Some(t0) = t0_end_layer {
                    s.finish().map_err(|e| anyhow::anyhow!("bucket end_layer finish L{layer_idx}: {e}"))?;
                    PROFILE_B_END_LAYER_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_B_END_LAYER_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("bucket end_layer resume L{layer_idx}: {e}"))?;
                }

                // Layer-boundary commit.  In production, we use `s.commit()`
                // (no wait) so the GPU runs this layer while the CPU encodes
                // the next — closes the ~2-3 ms per-layer GPU-idle burst that
                // `commit_and_wait` imposes and that the Metal System Trace
                // from 2026-04-20 showed as ~75 ms of 220 ms total idle in a
                // 916 ms pp2455 prefill.
                //
                // Metal guarantees in-order execution of command buffers
                // submitted to the same queue (see encoder.rs and
                // ADR-011 P4 notes), so pf_hidden written in layer N is
                // visible to layer N+1's first read automatically — the
                // CPU-side `wait_until_completed` is not required for GPU
                // correctness.  Memory residency is unchanged (activations
                // are shared buffers, weights are permanently resident).
                //
                // We still fall back to `s.finish()` (commit + wait) when:
                //   * HF2Q_BATCHED_DUMP is set — the dump path below
                //     CPU-reads activation buffers and needs the GPU to
                //     have finished.
                //   * HF2Q_PROFILE_LAYERS is set — the per-layer
                //     wall-clock print is only meaningful when we wait.
                //   * HF2Q_SYNC_PER_LAYER is set — explicit debug knob
                //     for bisecting cross-layer correctness issues.
                let sync_per_layer = batched_dump.is_some()
                    || std::env::var("HF2Q_PROFILE_LAYERS").is_ok()
                    || std::env::var("HF2Q_SYNC_PER_LAYER").is_ok()
                    || profile_buckets_on;
                if sync_per_layer {
                    s.finish()
                        .map_err(|e| anyhow::anyhow!("batched mlp finish L{layer_idx}: {e}"))?;
                    if std::env::var("HF2Q_PROFILE_LAYERS").is_ok() {
                        let kind = if is_sliding { "SW" } else { "GL" };
                        eprintln!("[LAYER_TIME] L{:02} {} {}us", layer_idx, kind,
                            layer_start.elapsed().as_micros());
                    }
                } else {
                    // Fire-and-forget commit — GPU continues executing
                    // while CPU moves on.  The returned CommandEncoder is
                    // dropped at end of scope; Metal's CommandBuffer is
                    // already committed and ref-counted into the queue,
                    // so it outlives the Rust handle until GPU completion.
                    let _committed = s.commit();
                    drop(_committed);
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

            // Metal-1 — stop capture once the target layer window closes
            // (before the final-norm + lm_head, so the .gputrace is
            // bounded to the per-layer scheduling we asked to inspect).
            if capture_active && layer_idx == capture_layer_end {
                mlx_native::metal::CaptureManager::shared().stop_capture();
                eprintln!(
                    "[METAL_CAPTURE] stopped after layer {}; .gputrace written to {}",
                    layer_idx,
                    capture_path.as_ref().map(|s| s.as_str()).unwrap_or("?")
                );
                capture_active = false;
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
            let t0_final_norm = if profile_buckets_on {
                Some(std::time::Instant::now())
            } else { None };
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
            if let Some(t0) = t0_final_norm {
                s.finish().map_err(|e| anyhow::anyhow!("bucket final_norm finish: {e}"))?;
                PROFILE_B_FINAL_NORM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                PROFILE_B_FINAL_NORM_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec.begin().map_err(|e| anyhow::anyhow!("bucket final_norm resume: {e}"))?;
            }

            // lm_head: whichever weight was loaded (Q8 for large vocab×hs, F16 otherwise).
            let t0_lm_head = if profile_buckets_on {
                Some(std::time::Instant::now())
            } else { None };
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
            if let Some(t0) = t0_lm_head {
                s.finish().map_err(|e| anyhow::anyhow!("bucket lm_head finish: {e}"))?;
                PROFILE_B_LM_HEAD_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                PROFILE_B_LM_HEAD_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec.begin().map_err(|e| anyhow::anyhow!("bucket lm_head resume: {e}"))?;
            }

            if let Some(cap) = self.final_logit_softcapping {
                let t0_softcap = if profile_buckets_on {
                    Some(std::time::Instant::now())
                } else { None };
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
                if let Some(t0) = t0_softcap {
                    s.finish().map_err(|e| anyhow::anyhow!("bucket softcap finish: {e}"))?;
                    PROFILE_B_SOFTCAP_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    PROFILE_B_SOFTCAP_COUNT.fetch_add(1, Ordering::Relaxed);
                    s = exec.begin().map_err(|e| anyhow::anyhow!("bucket softcap resume: {e}"))?;
                }
            }

            let t0_argmax = if profile_buckets_on {
                Some(std::time::Instant::now())
            } else { None };
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
            if let Some(t0) = t0_argmax {
                s.finish().map_err(|e| anyhow::anyhow!("bucket argmax finish: {e}"))?;
                PROFILE_B_ARGMAX_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                PROFILE_B_ARGMAX_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec.begin().map_err(|e| anyhow::anyhow!("bucket argmax resume: {e}"))?;
            }

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
            // Sum across the four per-site atomics.  These are also fed by
            // HF2Q_PROFILE_BUCKETS; if bucket profiling already swapped them,
            // this emission will see zeros and that's fine — both flags
            // don't cooperate, only one is meaningful per run.
            let qkv_ns = PROFILE_QKV_MM_NS.load(Ordering::Relaxed);
            let qkv_n  = PROFILE_QKV_MM_COUNT.load(Ordering::Relaxed);
            let o_ns   = PROFILE_O_MM_NS.load(Ordering::Relaxed);
            let o_n    = PROFILE_O_MM_COUNT.load(Ordering::Relaxed);
            let gur_ns = PROFILE_MLP_GUR_MM_NS.load(Ordering::Relaxed);
            let gur_n  = PROFILE_MLP_GUR_MM_COUNT.load(Ordering::Relaxed);
            let dn_ns  = PROFILE_MLP_DN_MM_NS.load(Ordering::Relaxed);
            let dn_n   = PROFILE_MLP_DN_MM_COUNT.load(Ordering::Relaxed);
            let mm_ns = qkv_ns + o_ns + gur_ns + dn_ns;
            let mm_n  = qkv_n + o_n + gur_n + dn_n;
            let total_ms = mm_ns as f64 / 1_000_000.0;
            eprintln!(
                "[MM_PROFILE] dense qmatmul: {} calls, {:.2} ms total ({:.3} ms/call) ({:.1}% of prefill {:.1} ms)",
                mm_n, total_ms,
                if mm_n > 0 { total_ms / mm_n as f64 } else { 0.0 },
                if prefill_ms > 0.0 { 100.0 * total_ms / prefill_ms } else { 0.0 },
                prefill_ms,
            );
        }
        if std::env::var("HF2Q_PROFILE_MOE_POST").is_ok() {
            let post_ns = PROFILE_MOE_POST_NS.swap(0, Ordering::Relaxed);
            let post_n = PROFILE_MOE_POST_COUNT.swap(0, Ordering::Relaxed);
            let total_ms = post_ns as f64 / 1_000_000.0;
            eprintln!(
                "[MOE_POST_PROFILE] swiglu+wsum: {} calls, {:.2} ms total ({:.3} ms/call) ({:.1}% of prefill {:.1} ms)",
                post_n, total_ms,
                if post_n > 0 { total_ms / post_n as f64 } else { 0.0 },
                if prefill_ms > 0.0 { 100.0 * total_ms / prefill_ms } else { 0.0 },
                prefill_ms,
            );
        }
        // Suppress dead_code warnings for as-yet-unwired profile counters
        // (PROFILE_NORM_*, PROFILE_PERMUTE_*) — staged for upcoming
        // sub-category profiling waves.
        let _ = (PROFILE_NORM_NS.load(Ordering::Relaxed), PROFILE_NORM_COUNT.load(Ordering::Relaxed),
                 PROFILE_PERMUTE_NS.load(Ordering::Relaxed), PROFILE_PERMUTE_COUNT.load(Ordering::Relaxed));

        // Wave P4.17 — comprehensive BUCKET_PROFILE breakdown.  Sum across
        // every instrumented category should approximately equal the
        // prefill wall-clock; the residual is CPU encode + commit/finish
        // overhead that the per-op sync pattern introduces.
        if profile_buckets_on {
            let fetch = |ns: &AtomicU64, cnt: &AtomicU64| -> (f64, u64) {
                let n_ns = ns.swap(0, Ordering::Relaxed);
                let n_cnt = cnt.swap(0, Ordering::Relaxed);
                (n_ns as f64 / 1_000_000.0, n_cnt)
            };
            let pct = |ms: f64| -> f64 {
                if prefill_ms > 0.0 { 100.0 * ms / prefill_ms } else { 0.0 }
            };
            let per = |ms: f64, n: u64| -> f64 {
                if n > 0 { ms / n as f64 } else { 0.0 }
            };

            // Startup sub-buckets.
            let (embed_ms, embed_n)       = fetch(&PROFILE_B_EMBED_NS, &PROFILE_B_EMBED_COUNT);
            let (mask_sw_ms, mask_sw_n)   = fetch(&PROFILE_B_MASK_SW_NS, &PROFILE_B_MASK_SW_COUNT);
            let (mask_gl_ms, mask_gl_n)   = fetch(&PROFILE_B_MASK_GL_NS, &PROFILE_B_MASK_GL_COUNT);
            let (blk_sw_ms, blk_sw_n)     = fetch(&PROFILE_B_BLK_SW_NS, &PROFILE_B_BLK_SW_COUNT);
            let (blk_gl_ms, blk_gl_n)     = fetch(&PROFILE_B_BLK_GL_NS, &PROFILE_B_BLK_GL_COUNT);
            let startup_ms = embed_ms + mask_sw_ms + mask_gl_ms + blk_sw_ms + blk_gl_ms;

            // Per-layer sub-buckets.
            let (pre_norm_ms, pre_norm_n) = fetch(&PROFILE_B_PRE_ATTN_NORM_NS, &PROFILE_B_PRE_ATTN_NORM_COUNT);
            let (qkv_ms, qkv_n)           = fetch(&PROFILE_QKV_MM_NS, &PROFILE_QKV_MM_COUNT);
            let (hnr_ms, hnr_n)           = fetch(&PROFILE_B_HEAD_NORM_ROPE_NS, &PROFILE_B_HEAD_NORM_ROPE_COUNT);
            let (fa_sw_ms, fa_sw_n)       = fetch(&PROFILE_FA_SW_NS, &PROFILE_FA_SW_COUNT);
            let (fa_gl_ms, fa_gl_n)       = fetch(&PROFILE_FA_GL_NS, &PROFILE_FA_GL_COUNT);
            let (post_fa_perm_ms, post_fa_perm_n) = fetch(&PROFILE_B_POST_FA_PERMUTE_NS, &PROFILE_B_POST_FA_PERMUTE_COUNT);
            let (o_ms, o_n)               = fetch(&PROFILE_O_MM_NS, &PROFILE_O_MM_COUNT);
            let (post_attn_na_ms, post_attn_na_n) = fetch(&PROFILE_B_POST_ATTN_NORM_ADD_NS, &PROFILE_B_POST_ATTN_NORM_ADD_COUNT);
            let (kv_copy_ms, kv_copy_n)   = fetch(&PROFILE_B_KV_COPY_NS, &PROFILE_B_KV_COPY_COUNT);
            let (triple_ms, triple_n)     = fetch(&PROFILE_B_TRIPLE_NORM_NS, &PROFILE_B_TRIPLE_NORM_COUNT);
            let (gur_ms, gur_n)           = fetch(&PROFILE_MLP_GUR_MM_NS, &PROFILE_MLP_GUR_MM_COUNT);
            let (gelu_mul_r_ms, gelu_mul_r_n) = fetch(&PROFILE_B_GELU_MUL_ROUTING_NS, &PROFILE_B_GELU_MUL_ROUTING_COUNT);
            let (mlp_dn_ms, mlp_dn_n)     = fetch(&PROFILE_MLP_DN_MM_NS, &PROFILE_MLP_DN_MM_COUNT);
            let (moe_gu_ms, moe_gu_n)     = fetch(&PROFILE_MOE_GU_NS, &PROFILE_MOE_GU_COUNT);
            let (moe_sw_ms, moe_sw_n)     = fetch(&PROFILE_MOE_POST_NS, &PROFILE_MOE_POST_COUNT);
            let (moe_dn_ms, moe_dn_n)     = fetch(&PROFILE_MOE_DN_NS, &PROFILE_MOE_DN_COUNT);
            let (moe_ws_ms, moe_ws_n)     = fetch(&PROFILE_B_MOE_WSUM_ADD_NS, &PROFILE_B_MOE_WSUM_ADD_COUNT);
            let (end_layer_ms, end_layer_n) = fetch(&PROFILE_B_END_LAYER_NS, &PROFILE_B_END_LAYER_COUNT);

            // Head sub-buckets.
            let (final_norm_ms, final_norm_n) = fetch(&PROFILE_B_FINAL_NORM_NS, &PROFILE_B_FINAL_NORM_COUNT);
            let (lm_head_ms, lm_head_n)   = fetch(&PROFILE_B_LM_HEAD_NS, &PROFILE_B_LM_HEAD_COUNT);
            let (softcap_ms, softcap_n)   = fetch(&PROFILE_B_SOFTCAP_NS, &PROFILE_B_SOFTCAP_COUNT);
            let (argmax_ms, argmax_n)     = fetch(&PROFILE_B_ARGMAX_NS, &PROFILE_B_ARGMAX_COUNT);
            let head_ms = final_norm_ms + lm_head_ms + softcap_ms + argmax_ms;

            let sum_ms = startup_ms
                + pre_norm_ms + qkv_ms + hnr_ms + fa_sw_ms + fa_gl_ms
                + post_fa_perm_ms + o_ms + post_attn_na_ms + kv_copy_ms
                + triple_ms + gur_ms + gelu_mul_r_ms + mlp_dn_ms
                + moe_gu_ms + moe_sw_ms + moe_dn_ms + moe_ws_ms + end_layer_ms
                + head_ms;
            let residual_ms = prefill_ms - sum_ms;

            let sep = "------------------------------------------------------------------";
            let row5 = |name: &str, ms: f64, n: u64| {
                eprintln!("{:<32} {:>8.3}  {:>5.1}%  {:>5}  {:>8.3}",
                    name, ms, pct(ms), n, per(ms, n));
            };
            let row3 = |name: &str, ms: f64, p: f64| {
                eprintln!("{:<32} {:>8.2}  {:>5.1}%", name, ms, p);
            };
            eprintln!("[BUCKET_PROFILE] pp={} prefill={:.2} ms (tok/s={:.1}) path={}",
                seq_len, prefill_ms, seq_len as f64 / (prefill_ms / 1000.0),
                if use_no_fa { "tensor-mm (non-FA)" } else { "flash-attn" });
            eprintln!("{:<32} {:>8}  {:>6}  {:>5}  {:>8}", "CATEGORY", "ms", "%", "calls", "ms/call");
            eprintln!("{sep}");
            let startup_n = embed_n + mask_sw_n + mask_gl_n + blk_sw_n + blk_gl_n;
            eprintln!("{:<32} {:>8.2}  {:>5.1}%  {:>5}  {:>8}",
                "STARTUP (setup total)", startup_ms, pct(startup_ms), startup_n, "—");
            row5("  embed",         embed_ms,   embed_n);
            row5("  mask_sliding",  mask_sw_ms, mask_sw_n);
            row5("  mask_global",   mask_gl_ms, mask_gl_n);
            row5("  blk_sliding",   blk_sw_ms,  blk_sw_n);
            row5("  blk_global",    blk_gl_ms,  blk_gl_n);
            eprintln!("{sep}");
            row5("PRE_ATTN_NORM",             pre_norm_ms,     pre_norm_n);
            row5("QKV_MM",                    qkv_ms,          qkv_n);
            row5("HEAD_NORM_ROPE (Q+K+V)",    hnr_ms,          hnr_n);
            row5("FA_SW (D=256)",             fa_sw_ms,        fa_sw_n);
            row5("FA_GL (D=512)",             fa_gl_ms,        fa_gl_n);
            row5("POST_FA_PERMUTE",           post_fa_perm_ms, post_fa_perm_n);
            row5("O_MM",                      o_ms,            o_n);
            row5("POST_ATTN_NORM_ADD",        post_attn_na_ms, post_attn_na_n);
            row5("KV_COPY",                   kv_copy_ms,      kv_copy_n);
            row5("TRIPLE_RMS_NORM",           triple_ms,       triple_n);
            row5("MLP_GUR_MM",                gur_ms,          gur_n);
            row5("GELU_MUL + MOE_ROUTING",    gelu_mul_r_ms,   gelu_mul_r_n);
            row5("MLP_DN_MM",                 mlp_dn_ms,       mlp_dn_n);
            row5("MOE_GATE_UP",               moe_gu_ms,       moe_gu_n);
            row5("MOE_SWIGLU",                moe_sw_ms,       moe_sw_n);
            row5("MOE_DOWN",                  moe_dn_ms,       moe_dn_n);
            row5("MOE_WSUM_DNORM_ADD",        moe_ws_ms,       moe_ws_n);
            row5("END_LAYER_NORM_ADD_SCALAR", end_layer_ms,    end_layer_n);
            eprintln!("{sep}");
            row5("FINAL_NORM (copy + rms)",   final_norm_ms, final_norm_n);
            row5("LM_HEAD",                   lm_head_ms,    lm_head_n);
            row5("SOFTCAP",                   softcap_ms,    softcap_n);
            row5("ARGMAX",                    argmax_ms,     argmax_n);
            eprintln!("{sep}");
            row3("SUM OF BUCKETS",        sum_ms,      pct(sum_ms));
            row3("PREFILL TOTAL",         prefill_ms,  100.0);
            row3("RESIDUAL (sync + CPU)", residual_ms, pct(residual_ms));
        }

        // Store dense KV buffers so forward_decode can use them
        self.dense_kvs = Some(dense_kvs_vec);
        self.dense_sdpa_tmp = Some(sdpa_tmp);

        // Metal-1 — safety-net stop in case the layer-range end was
        // beyond the actual layer count (shouldn't happen with valid
        // envs but the capture API must always be balanced).
        if capture_active {
            mlx_native::metal::CaptureManager::shared().stop_capture();
            if let Some(p) = capture_path.as_ref() {
                eprintln!("[METAL_CAPTURE] safety-net stop; .gputrace written to {}", p);
            }
        }

        Ok(first_token)
    }
}
