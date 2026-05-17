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
    DenseKvBuffers, HbKvBuffers, MlxModelWeights, dispatch_qmatmul,
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
// ADR-029 iter-81 H61: per-dispatch profiling of HF2Q_NO_FA's 5 dispatches
// per global-attn layer.  Used to localize where the NO_FA-vs-FA wall delta
// (1610 ms at 8K) actually goes: Q@K^T mm, scale-mask-softmax, V transpose,
// scores@V mm, or output permute_021.  All gated by HF2Q_PROFILE_BUCKETS=1.
static PROFILE_B_NOFA_QK_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_NOFA_QK_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_NOFA_SMS_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_NOFA_SMS_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_NOFA_VTRANS_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_NOFA_VTRANS_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_NOFA_SV_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_NOFA_SV_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_NOFA_PERM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_B_NOFA_PERM_COUNT: AtomicU64 = AtomicU64::new(0);
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

// perf-1 (cfa-20260420-172215-moe-parity-push):
// HF2Q_PROFILE_GPU_TS=1 — per-bucket GPU wall-clock instead of CPU
// wall-clock.  Every bucket's finish() pair switches to
// `finish_with_gpu_time()`, which reads MTLCommandBuffer.GPUStartTime /
// GPUEndTime after wait_until_completed.  The atomic accumulators
// receive pure GPU execution time (no CPU commit/wait overhead); the
// residual becomes the honest "commit+wait+CPU encode" component.
//
// `HF2Q_PROFILE_BUCKETS=1` keeps its old CPU-wall-clock semantics so
// existing logs stay comparable.  When both are set, GPU_TS wins.
//
// Zero runtime cost when unset — a single `load(Relaxed)` in the hot
// path selects between `finish()` and `finish_with_gpu_time()`.
static PROFILE_GPU_TS_ON: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Close the current bucket session and accumulate its elapsed time.
///
/// `$s`  — mutable `GraphSession` about to be closed (consumed by this macro).
/// `$exec` — `GraphExecutor` ref, used to begin the next session.
/// `$t0` — `Instant` captured before the first dispatch in the bucket.
/// `$ns` / `$cnt` — atomic accumulators to bump.
/// `$inc` — how many dispatches the bucket contains.
/// `$ctx` — string slice for the error paths.
///
/// When `PROFILE_GPU_TS_ON` is true, the GPU wall-clock read from
/// `MTLCommandBuffer.GPUEndTime - GPUStartTime` is used; else the CPU
/// wall-clock `t0.elapsed()`.  The macro leaves `$s` bound to a fresh
/// session so the caller can continue dispatching.
macro_rules! bucket_finish {
    ($s:ident, $exec:expr, $t0:expr, $ns:expr, $cnt:expr, $inc:expr, $ctx:expr) => {{
        let dt_ns = if PROFILE_GPU_TS_ON.load(Ordering::Relaxed) {
            $s.finish_with_gpu_time()
                .map_err(|e| anyhow::anyhow!("bucket {} finish_gpu: {}", $ctx, e))?
        } else {
            $s.finish()
                .map_err(|e| anyhow::anyhow!("bucket {} finish: {}", $ctx, e))?;
            $t0.elapsed().as_nanos() as u64
        };
        $ns.fetch_add(dt_ns, Ordering::Relaxed);
        $cnt.fetch_add($inc, Ordering::Relaxed);
        $s = $exec
            .begin()
            .map_err(|e| anyhow::anyhow!("bucket {} resume: {}", $ctx, e))?;
    }};
}

impl MlxModelWeights {
    /// True batched prefill with single-shot dense SDPA over the whole prompt.
    ///
    /// Returns the first decode token (greedy argmax of last-row logits).
    ///
    /// # ADR-028 iter-137 — append-mode parameter (Path A Phase 2 GPU step 2/7)
    ///
    /// `start_pos` lets the caller specify the absolute KV-cache position
    /// where this batch begins. Default callers pass `0` (cold prefill,
    /// matches pre-iter-137 behavior byte-for-byte). Future
    /// `forward_decode_verify_batched` (iter-139) calls with `start_pos =
    /// current_seq_pos` to append-mode the K/V writes.
    ///
    /// Internally this threads through `pf_positions[i] = start_pos + i`
    /// and `kv_caches[i].write_pos = start_pos + seq_len`. At
    /// `start_pos=0` both reduce to the original semantics.
    pub fn forward_prefill_batched(
        &mut self,
        prompt_tokens: &[u32],
        max_decode_tokens: usize,
        start_pos: usize,
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

        // ADR-032 (2026-05-17): HF2Q_NO_FA reduced to debug/diagnostic flag.
        //
        // History: this was previously default-on at seq_len>=32 as a
        // workaround for Bug A (FA-D=512 BF16-Q argmax drift, Hemoglobin
        // loop at decode-pos ~70 on enumeration prompts).  ADR-032 Phase 1
        // peer-kernel diff against llama.cpp's `kernel_flash_attn_ext`
        // showed that bug was specifically an *instantiation* deviation:
        // llama.cpp's default `kernel_flash_attn_ext_f16_dk512_dv512` uses
        // `FA_TYPES` (Q/K/V all `half`/F16 in shmem), and only the explicit
        // BF16-KV-cache instantiation (`FA_TYPES_BF`) uses `bfloat`.
        // Gemma 4's default KV cache is F16 in llama.cpp, so peer's
        // production path is F16-Q-in-shmem — NOT BF16.  Our prior NO_FA
        // default routed around the FA path entirely; ADR-032 instead
        // fixes the kernel-instantiation deviation (see `HF2Q_FA_F16`
        // below) so the FA path matches peer's algorithm.
        //
        // Tensor-mm fallback retained as `HF2Q_NO_FA=1` for diagnostic A/B
        // comparison only.  At seq_len<32 the dense matmul kernel
        // (`dense_matmul_bf16_f32_tensor`) returns a hard error because
        // its tile reduction is 32-aligned; we therefore force FA at
        // short seq regardless of operator request.
        //
        // Memory: the NO_FA path allocates `pf_kq` [n_heads, seq, seq]
        // F32 used by global D=512 layers only — ~4 MB at seq=246,
        // ~118 MB at seq=1359, ~555 MB at seq=2455.  The FA path
        // (default) scales O(seq) not O(seq²), so long-context workloads
        // benefit from the new default in addition to the algorithmic
        // alignment.
        let env_no_fa = match std::env::var("HF2Q_NO_FA").as_deref() {
            Ok("0") | Ok("false") | Ok("off") => Some(false),
            Ok("1") | Ok("true") | Ok("on") => Some(true),
            _ => None,
        };
        let use_no_fa = match env_no_fa {
            // Explicit override always honored.  Operator may force NO_FA
            // for diagnostic A/B (e.g. validating a future FA kernel
            // change against the tensor-mm reference).
            Some(b) => b && seq_len >= 32, // force-FA at seq<32 (kernel guard)
            // Default (ADR-032): use FA path.  F16-Q-in-shmem matches
            // peer algorithm; precision is sufficient at both D=256 sliding
            // and D=512 global layers.
            None => false,
        };

        // ADR-032 (2026-05-17): F16 FA path — peer-aligned default.
        //
        // Q/K/V are written F32 → F16 via `permute_021_f32_to_f16` (single
        // rounding step, mantissa-faithful) and the FA prefill kernel
        // instantiates with `T=half` (10-bit mantissa).  This matches
        // llama.cpp's default `kernel_flash_attn_ext_f16_dk{256,512}_dv*`
        // template `FA_TYPES` (see /opt/llama.cpp/ggml/src/ggml-metal/
        // ggml-metal.metal:6472, `half, half4, simdgroup_half8x8` for Q
        // shmem; F32 for accumulator, softmax, scale).
        //
        // Mantissa budget over a 512-element Q·K dot product accumulator:
        //   F16 × F16: sqrt(512) × 2^-11 ≈ 1.1% — below argmax-flip threshold
        //   BF16 × BF16: sqrt(512) × 2^-8  ≈ 9%  — above threshold for
        //     narrow-margin greedy decode (Bug A: Format: Hemoglobin loop)
        //
        // Default-flipped to TRUE (ADR-032 Phase 6).  Sliding D=256 layers
        // dispatch `dispatch_flash_attn_prefill_f16_d256_with_blk`; global
        // D=512 layers dispatch `dispatch_flash_attn_prefill_f16_d512_with_blk`.
        // F16 → BF16 cast on output preserves o_proj's BF16 input contract.
        //
        // Opt-out via `HF2Q_FA_F16=0|false|off` reverts to legacy BF16
        // instantiation (peer's `FA_TYPES_BF` path) for diagnostic A/B
        // comparison only.  BF16-Q is the known-buggy path on enumeration
        // prompts at D=512; the opt-out exists for kernel-bisection work,
        // not production use.
        let use_fa_f16 = !matches!(
            std::env::var("HF2Q_FA_F16").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        );

        // Wave P4.17 — super-flag: per-op isolation for bucket attribution.
        // When on, every dispatch is bracketed by s.finish()/s = exec.begin()
        // pairs so its wall-clock is measured; the individual bucket atomics
        // accumulate ns and counts.  Prefill throughput REGRESSES under this
        // flag (extra finish/begin per op ≈ 50-200 µs each, × ~15 ops/layer
        // × 30 layers ≈ 100-150 ms of pure overhead on top of normal work),
        // so it's a profiling-only knob.
        // perf-1 — `HF2Q_PROFILE_GPU_TS=1` implies bucket profiling (needs
        // the per-bucket sync boundaries) and additionally switches
        // every bucket's accumulator to use GPUStartTime/GPUEndTime
        // from the just-completed command buffer.  When off, behaviour
        // is unchanged (CPU wall-clock).
        let profile_gpu_ts_on = std::env::var("HF2Q_PROFILE_GPU_TS").is_ok();
        PROFILE_GPU_TS_ON.store(profile_gpu_ts_on, Ordering::Relaxed);
        let profile_buckets_on = profile_gpu_ts_on
            || std::env::var("HF2Q_PROFILE_BUCKETS").is_ok();

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
            dense_kvs_vec.push(DenseKvBuffers {
                k,
                v,
                capacity,
                is_sliding: layer_is_ring,
                // ADR-017 Phase E.a iter-3.5a — dtype invariant.
                dtype: kv_dtype,
            });
        }
        let max_nh = nh;
        let max_hd = self.layers.iter().map(|l| l.head_dim).max().unwrap_or(512);
        let tmp_bytes = mlx_native::ops::flash_attn_vec::tmp_buffer_bytes(
            max_nh as u32, max_hd as u32);
        let sdpa_tmp = dev.alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
            .map_err(|e| anyhow::anyhow!("batched sdpa_tmp: {e}"))?;

        // ADR-010 iter-64 — eager allocation of leg_hb_encoded for batched
        // prefill, mirrors per-token forward_prefill.rs:804-852. Without
        // this block, self.leg_hb_encoded was lazily allocated at the first
        // decode that needed HB cache (forward_mlx.rs:2314), AFTER batched-
        // prefill returned — leaving the buffers zero-initialized for the
        // decode SDPA reads → garbage attention → gibberish tokens. See
        // ADR-010 §Status Log 2026-05-09 iter-63 smoking-gun localization.
        let tq_codebook_bits_prefill: u32 = match std::env::var("HF2Q_TQ_CODEBOOK_BITS").as_deref() {
            Ok("4") => 0,
            Ok("5") => 5, Ok("6") => 6, Ok("8") => 8,
            _ => 8,  // DEFAULT: 8-bit (matches forward_prefill.rs)
        };
        let tq_scale_factor_d512: f32 = match std::env::var("HF2Q_SCALE_FORMULA").as_deref() {
            Ok("sqrt256") => 16.0_f32,
            Ok("sqrt512") => 512.0_f32.sqrt(),
            _ => 1.0_f32,  // bare (iter-16 default)
        };
        if tq_codebook_bits_prefill >= 5 {
            // ADR-028 Phase 10c (iter-348): hybrid F16-K + TQ-HB-V routing,
            // mirrors forward_mlx.rs decode lazy-alloc + forward_prefill.rs.
            //
            // ADR-030 iter-63 (extend-mode KV preservation): allocate only
            // if `self.hybrid_kv` / `self.leg_hb_encoded` is None.  This
            // matches forward_decode's lazy-alloc pattern at
            // forward_mlx.rs:3045 (`&& self.hybrid_kv.is_none()`) and is
            // load-bearing for spec-decode verify rounds, where the
            // orchestrator calls forward_prefill_batched repeatedly with
            // non-zero `start_pos` to APPEND to the existing cache.
            // Without this guard the second call zeroed prompt + accepted
            // K/V data, producing incoherent rounds despite
            // pf_positions / write_pos already being correctly offset by
            // iter-137/138.  All production callers pass `start_pos=0` on a
            // fresh MlxModelWeights instance (hybrid_kv == None), so this
            // is bit-identical to pre-iter-63 for the cmd_generate /
            // parity / engine flows.
            if INVESTIGATION_ENV.hybrid_kv {
                if self.hybrid_kv.is_none() {
                    eprintln!("[ADR-028 Phase 10c] Allocating hybrid_kv ({} layers, F16 K + TQ-HB V {}-bit) [batched]",
                        num_layers, tq_codebook_bits_prefill);
                    let mut hybrid_vec: Vec<crate::serve::forward_mlx::HybridKvBuffers> = Vec::with_capacity(num_layers);
                    for (layer_idx, layer) in self.layers.iter().enumerate() {
                        let nkv_l = layer.num_kv_heads;
                        let hd_l = layer.head_dim;
                        let layer_is_ring = layer.layer_type == LayerType::Sliding;
                        let capacity = if layer_is_ring { sw } else { linear_capacity };
                        hybrid_vec.push(crate::serve::forward_mlx::alloc_hybrid_kv_for_layer(
                            dev, layer_idx, nkv_l, hd_l, capacity, layer_is_ring)?);
                    }
                    self.hybrid_kv = Some(hybrid_vec);
                }
            } else if self.leg_hb_encoded.is_none() {
                eprintln!("[iter-21 Track B] Allocating leg_hb_encoded ({}-bit, {} layers) [batched]",
                          tq_codebook_bits_prefill, num_layers);
                let mut leg_hb_vec: Vec<HbKvBuffers> = Vec::with_capacity(num_layers);
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let nkv_l = layer.num_kv_heads;
                    let hd_l = layer.head_dim;
                    let layer_is_ring = layer.layer_type == LayerType::Sliding;
                    let capacity = if layer_is_ring { sw } else { linear_capacity };
                    let norms_per_pos = (hd_l / 256).max(1);
                    let norms_n = nkv_l * capacity * norms_per_pos;
                    let k_packed = dev.alloc_buffer(nkv_l * capacity * hd_l, mlx_native::DType::U8,
                        vec![nkv_l, capacity, hd_l])
                        .map_err(|e| anyhow::anyhow!("leg_hb batched K packed L{layer_idx}: {e}"))?;
                    let k_norms = dev.alloc_buffer(norms_n * 4, mlx_native::DType::F32,
                        if norms_per_pos == 1 { vec![nkv_l, capacity] } else { vec![nkv_l, capacity, norms_per_pos] })
                        .map_err(|e| anyhow::anyhow!("leg_hb batched K norms L{layer_idx}: {e}"))?;
                    let v_packed = dev.alloc_buffer(nkv_l * capacity * hd_l, mlx_native::DType::U8,
                        vec![nkv_l, capacity, hd_l])
                        .map_err(|e| anyhow::anyhow!("leg_hb batched V packed L{layer_idx}: {e}"))?;
                    let v_norms = dev.alloc_buffer(norms_n * 4, mlx_native::DType::F32,
                        if norms_per_pos == 1 { vec![nkv_l, capacity] } else { vec![nkv_l, capacity, norms_per_pos] })
                        .map_err(|e| anyhow::anyhow!("leg_hb batched V norms L{layer_idx}: {e}"))?;
                    leg_hb_vec.push(HbKvBuffers {
                        k_packed, k_norms, v_packed, v_norms,
                        capacity, is_sliding: layer_is_ring, norms_per_pos,
                    });
                }
                self.leg_hb_encoded = Some(leg_hb_vec);
                eprintln!("[iter-21 Track B] leg_hb_encoded ready ({} layers) [batched]", num_layers);
            }
        }

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
        // ADR-030 iter-77: F16 alloc helper for the cross-length verify
        // SDPA path (Q/output cast targets).  F16 width is 2 bytes
        // (matches BF16 in size; only mantissa/exponent split differs).
        let alloc_f16 = |n: usize, name: &str| -> Result<MlxBuffer> {
            dev.alloc_buffer(n * 2, DType::F16, vec![n])
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

        // F16 staging buffers for the HF2Q_FA_F16=1 path (kernel migration
        // step 3).  These mirror the BF16 perm buffers but in F16 dtype;
        // we cast BF16→F16 into them before the FA call and cast back
        // F16→BF16 from the output after.  Only allocated when the env
        // flag is set so the default-path memory footprint is unchanged
        // (matches HF2Q_NO_FA's allocation pattern).  Each buffer is the
        // same byte size as its BF16 sibling (both 2 bytes/element), so
        // the additional memory budget is 4× per-head F16 perm =
        // `(nh + max_nkv*2 + nh) * seq_len * max_hd * 2` bytes.  At
        // seq=246, max_hd=512 (D=512 layers): ~24 MB.  At seq=1359: ~134 MB.
        let pf_q_perm_f16 = if use_fa_f16 {
            Some(alloc_f16(nh * seq_len * max_hd, "pf_q_perm_f16")?)
        } else { None };
        let pf_k_perm_f16 = if use_fa_f16 {
            Some(alloc_f16(max_nkv * seq_len * max_hd, "pf_k_perm_f16")?)
        } else { None };
        let pf_v_perm_f16 = if use_fa_f16 {
            Some(alloc_f16(max_nkv * seq_len * max_hd, "pf_v_perm_f16")?)
        } else { None };
        let mut pf_sdpa_out_perm_f16 = if use_fa_f16 {
            Some(alloc_f16(nh * seq_len * max_hd, "pf_sdpa_out_perm_f16")?)
        } else { None };

        // ADR-030 iter-77 — cross-length-SDPA verify mode buffers.  Active
        // only when env flag HF2Q_DFLASH_XLEN_SDPA=1 AND we are in a verify
        // call (start_pos > 0 AND dflash_capture installed).  The cast
        // chain is Q BF16→F32→F16 (call resume kernel) → F16→F32→BF16,
        // staged via these scratch buffers.  At seq_len = K+1 = 8 these
        // are very small (8 × 16 × 256 × 4 bytes ≈ 130 KB each).
        let xlen_sdpa_mode = std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() == Ok("1")
            && start_pos > 0
            && self.dflash_capture.is_some();
        let pf_q_f32_xlen: Option<MlxBuffer> = if xlen_sdpa_mode {
            Some(alloc_f32(nh * seq_len * max_hd, "pf_q_f32_xlen")?)
        } else { None };
        let pf_q_f16_xlen: Option<MlxBuffer> = if xlen_sdpa_mode {
            Some(alloc_f16(nh * seq_len * max_hd, "pf_q_f16_xlen")?)
        } else { None };
        let pf_out_f16_xlen: Option<MlxBuffer> = if xlen_sdpa_mode {
            Some(alloc_f16(nh * seq_len * max_hd, "pf_out_f16_xlen")?)
        } else { None };
        let pf_out_f32_xlen: Option<MlxBuffer> = if xlen_sdpa_mode {
            Some(alloc_f32(nh * seq_len * max_hd, "pf_out_f32_xlen")?)
        } else { None };

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
            // ADR-028 iter-137 Path A Phase 2 GPU step 2/7 — append-mode positions.
            // start_pos=0 (production default callers): identical to pre-iter-137.
            // start_pos>0 (future verify_batched callers): append at offset.
            for (i, slot) in p[..seq_len].iter_mut().enumerate() {
                *slot = (start_pos + i) as u32;
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
        // F16 sliding mask for HF2Q_FA_F16=1.  Built once via cast_bf16_to_f16
        // immediately after sliding_mask is constructed, reused across all
        // 25 sliding-attention layers' F16 FA dispatches.
        let sliding_mask_f16: Option<MlxBuffer>;
        // F16 global mask for HF2Q_FA_F16=1.  Same shape and rationale as
        // sliding_mask_f16; reused across all 5 global D=512 layers' F16
        // FA dispatches.  Fixes Bug A (Hemoglobin loop) by bringing the
        // D=512 path to F16's 10-bit mantissa from BF16's 7-bit.
        let global_mask_f16: Option<MlxBuffer>;
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
                // NEON fmul.  Rayon parallel was tested (Wave P4.18 drafts)
                // and measured high-variance with no reliable speedup —
                // the gather is cold-page-touch bound, and spreading the
                // random reads across threads doesn't help when the
                // bottleneck is DRAM latency for cache-miss lines.
                for (tok_idx, &tok_id) in prompt_tokens.iter().enumerate() {
                    let src_off = (tok_id as usize) * hs;
                    let dst_off = tok_idx * hs;
                    out[dst_off..dst_off + hs]
                        .copy_from_slice(&embed_f32[src_off..src_off + hs]);
                }
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
            // HF2Q_FA_F16=1: cast the BF16 sliding mask to F16 once, reused
            // across all 25 sliding D=256 layers' F16 FA dispatches.
            // Must be rank-2 [seq_len, seq_len] so the FA dispatcher's
            // rank-2 broadcast detection (`mask.shape().len() == 2`) fires.
            sliding_mask_f16 = if use_fa_f16 {
                let mask_elems = (seq_len * seq_len) as usize;
                let m_f16 = dev.alloc_buffer(
                    mask_elems * 2, DType::F16, vec![seq_len, seq_len],
                ).map_err(|e| anyhow::anyhow!("alloc sliding_mask_f16: {e}"))?;
                s.barrier_between(&[&sliding_mask], &[&m_f16]);
                mlx_native::ops::elementwise::cast(
                    s.encoder_mut(), reg, metal_dev,
                    &sliding_mask, &m_f16, mask_elems,
                    mlx_native::ops::elementwise::CastDirection::BF16ToF16,
                ).map_err(|e| anyhow::anyhow!("cast sliding_mask BF16→F16: {e}"))?;
                Some(m_f16)
            } else { None };
            if let Some(t0) = t0_mask_sw {
                bucket_finish!(s, exec, t0, &PROFILE_B_MASK_SW_NS, &PROFILE_B_MASK_SW_COUNT, 1, "mask_sw");
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
            // HF2Q_FA_F16=1: cast the BF16 global mask to F16 once, reused
            // across all 5 global D=512 layers' F16 FA dispatches.
            // Must be rank-2 [seq_len, seq_len] so the FA dispatcher's
            // rank-2 broadcast detection (`mask.shape().len() == 2`) fires.
            global_mask_f16 = if use_fa_f16 {
                let mask_elems = (seq_len * seq_len) as usize;
                let m_f16 = dev.alloc_buffer(
                    mask_elems * 2, DType::F16, vec![seq_len, seq_len],
                ).map_err(|e| anyhow::anyhow!("alloc global_mask_f16: {e}"))?;
                s.barrier_between(&[&global_mask], &[&m_f16]);
                mlx_native::ops::elementwise::cast(
                    s.encoder_mut(), reg, metal_dev,
                    &global_mask, &m_f16, mask_elems,
                    mlx_native::ops::elementwise::CastDirection::BF16ToF16,
                ).map_err(|e| anyhow::anyhow!("cast global_mask BF16→F16: {e}"))?;
                Some(m_f16)
            } else { None };
            if let Some(t0) = t0_mask_gl {
                bucket_finish!(s, exec, t0, &PROFILE_B_MASK_GL_NS, &PROFILE_B_MASK_GL_COUNT, 1, "mask_gl");
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
                bucket_finish!(s, exec, t0, &PROFILE_B_BLK_SW_NS, &PROFILE_B_BLK_SW_COUNT, 1, "blk_sw");
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
                bucket_finish!(s, exec, t0, &PROFILE_B_BLK_GL_NS, &PROFILE_B_BLK_GL_COUNT, 1, "blk_gl");
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
            //
            // ADR-029 iter-39 H40 — `HF2Q_GRAPH_OPT_PREFILL=1` opts the
            // per-layer session into capture/record mode so the end-of-layer
            // commit can run the fusion pass over the captured graph before
            // emitting the command buffer.  Fusion replaces RMS_NORM→MUL
            // patterns with a single dispatch (see graph.rs `ComputeGraph::fuse`),
            // matching peer's `ggml_graph_optimize` pass.  Default off until
            // measured.  Incompatible with HF2Q_PROFILE_BUCKETS / HF2Q_PROFILE_MM
            // (those introduce mid-session finish/begin pairs that defeat the
            // per-layer capture); when both are set, the recording flag is
            // forced off for correctness.
            let graph_opt_prefill = std::env::var("HF2Q_GRAPH_OPT_PREFILL")
                .ok()
                .as_deref() == Some("1")
                && !profile_buckets_on
                && std::env::var("HF2Q_PROFILE_MM").is_err()
                && std::env::var("HF2Q_PROFILE_FA").is_err()
                && std::env::var("HF2Q_PROFILE_MOE").is_err();
            {
                let mut s = if graph_opt_prefill {
                    exec.begin_recorded()
                        .map_err(|e| anyhow::anyhow!("batched attn session (recorded) L{layer_idx}: {e}"))?
                } else {
                    exec.begin()
                        .map_err(|e| anyhow::anyhow!("batched attn session L{layer_idx}: {e}"))?
                };

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
                    bucket_finish!(s, exec, t0, &PROFILE_B_PRE_ATTN_NORM_NS, &PROFILE_B_PRE_ATTN_NORM_COUNT, 1, "pre_attn_norm");
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
                    bucket_finish!(s, exec, t0, &PROFILE_QKV_MM_NS, &PROFILE_QKV_MM_COUNT,
                        if v_is_k { 2 } else { 3 }, "qkv");
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
                    bucket_finish!(s, exec, t0, &PROFILE_B_HEAD_NORM_ROPE_NS, &PROFILE_B_HEAD_NORM_ROPE_COUNT, 3, "head_norm_rope");
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
                // ADR-029 iter-82 H62: gate NO_FA on global-attn layers
                // only.  Sliding layers' FA_SW already has sliding-window
                // K=1024 cap + Wave 2E tile-skip, costing only 6.55 ms/call
                // at pp8333 (iter-50).  Under NO_FA they'd compute the full
                // uncapped K=qL matmul = 1771 ms total across 25 layers
                // (iter-81 measurement).  Keeping FA_SW for sliding +
                // routing global through NO_FA saves ~52 ms wall at pp8333
                // per model.
                let route_through_nofa = use_no_fa && !is_sliding;
                if route_through_nofa {
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
                    // ADR-029 iter-81 H61: bracket with finish/begin under
                    // profile_buckets_on to measure per-dispatch GPU+CPU wall.
                    let nofa_qk_t0 = if profile_buckets_on {
                        s.finish().map_err(|e| anyhow::anyhow!("noFA QK pre-finish L{layer_idx}: {e}"))?;
                        let t0 = std::time::Instant::now();
                        s = exec.begin().map_err(|e| anyhow::anyhow!("noFA QK begin L{layer_idx}: {e}"))?;
                        Some(t0)
                    } else { None };
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
                    if let Some(t0) = nofa_qk_t0 {
                        s.finish().map_err(|e| anyhow::anyhow!("noFA QK post-finish L{layer_idx}: {e}"))?;
                        let dt_ns = t0.elapsed().as_nanos() as u64;
                        PROFILE_B_NOFA_QK_NS.fetch_add(dt_ns, Ordering::Relaxed);
                        PROFILE_B_NOFA_QK_COUNT.fetch_add(1, Ordering::Relaxed);
                        s = exec.begin().map_err(|e| anyhow::anyhow!("noFA QK reopen L{layer_idx}: {e}"))?;
                    }

                    // Step 3: scale + mask + softmax fused.  scale = 1.0
                    // because Q's norm-weight pre-scales by 1/sqrt(hd)
                    // (matches FA path's scale=1.0).
                    let nofa_sms_t0 = if profile_buckets_on {
                        s.finish().map_err(|e| anyhow::anyhow!("noFA SMS pre-finish L{layer_idx}: {e}"))?;
                        let t0 = std::time::Instant::now();
                        s = exec.begin().map_err(|e| anyhow::anyhow!("noFA SMS begin L{layer_idx}: {e}"))?;
                        Some(t0)
                    } else { None };
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
                    if let Some(t0) = nofa_sms_t0 {
                        s.finish().map_err(|e| anyhow::anyhow!("noFA SMS post-finish L{layer_idx}: {e}"))?;
                        let dt_ns = t0.elapsed().as_nanos() as u64;
                        PROFILE_B_NOFA_SMS_NS.fetch_add(dt_ns, Ordering::Relaxed);
                        PROFILE_B_NOFA_SMS_COUNT.fetch_add(1, Ordering::Relaxed);
                        s = exec.begin().map_err(|e| anyhow::anyhow!("noFA SMS reopen L{layer_idx}: {e}"))?;
                    }

                    // Step 4: transpose V [nkv, seq, hd] -> [nkv, hd, seq]
                    // so the scores@V matmul contracts on seq_kv (K-dim
                    // = inner-most dim of src0 per our kernel contract).
                    let nofa_vtrans_t0 = if profile_buckets_on {
                        s.finish().map_err(|e| anyhow::anyhow!("noFA Vtrans pre-finish L{layer_idx}: {e}"))?;
                        let t0 = std::time::Instant::now();
                        s = exec.begin().map_err(|e| anyhow::anyhow!("noFA Vtrans begin L{layer_idx}: {e}"))?;
                        Some(t0)
                    } else { None };
                    s.barrier_between(
                        &[&pf_v_perm],
                        &[pf_v_perm_t_ref],
                    );
                    mlx_native::ops::transpose::transpose_last2_bf16(
                        s.encoder_mut(), reg, metal_dev,
                        &pf_v_perm, pf_v_perm_t_ref,
                        nkv, seq_len, hd,
                    ).map_err(|e| anyhow::anyhow!("non-FA V transpose L{layer_idx}: {e}"))?;
                    if let Some(t0) = nofa_vtrans_t0 {
                        s.finish().map_err(|e| anyhow::anyhow!("noFA Vtrans post-finish L{layer_idx}: {e}"))?;
                        let dt_ns = t0.elapsed().as_nanos() as u64;
                        PROFILE_B_NOFA_VTRANS_NS.fetch_add(dt_ns, Ordering::Relaxed);
                        PROFILE_B_NOFA_VTRANS_COUNT.fetch_add(1, Ordering::Relaxed);
                        s = exec.begin().map_err(|e| anyhow::anyhow!("noFA Vtrans reopen L{layer_idx}: {e}"))?;
                    }

                    // Step 5: scores @ V_t via dense bf16×f32→f32 tensor-mm.
                    // src0 = V_t [nkv, hd, seq_kv] bf16, src1 = kq [nh, seq_q, seq_kv] f32.
                    // Output attn[h, q, d] = sum_k V_t[h/r2, d, k] * kq[h, q, k]
                    //                       = sum_k V[h/r2, k, d] * probs[h, q, k].
                    let nofa_sv_t0 = if profile_buckets_on {
                        s.finish().map_err(|e| anyhow::anyhow!("noFA SV pre-finish L{layer_idx}: {e}"))?;
                        let t0 = std::time::Instant::now();
                        s = exec.begin().map_err(|e| anyhow::anyhow!("noFA SV begin L{layer_idx}: {e}"))?;
                        Some(t0)
                    } else { None };
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
                    if let Some(t0) = nofa_sv_t0 {
                        s.finish().map_err(|e| anyhow::anyhow!("noFA SV post-finish L{layer_idx}: {e}"))?;
                        let dt_ns = t0.elapsed().as_nanos() as u64;
                        PROFILE_B_NOFA_SV_NS.fetch_add(dt_ns, Ordering::Relaxed);
                        PROFILE_B_NOFA_SV_COUNT.fetch_add(1, Ordering::Relaxed);
                        s = exec.begin().map_err(|e| anyhow::anyhow!("noFA SV reopen L{layer_idx}: {e}"))?;
                    }

                    // Step 6: permute attn [nh, seq, hd] f32 -> pf_sdpa_out
                    // [seq, nh, hd] f32 to match the O-proj input layout.
                    let nofa_perm_t0 = if profile_buckets_on {
                        s.finish().map_err(|e| anyhow::anyhow!("noFA perm pre-finish L{layer_idx}: {e}"))?;
                        let t0 = std::time::Instant::now();
                        s = exec.begin().map_err(|e| anyhow::anyhow!("noFA perm begin L{layer_idx}: {e}"))?;
                        Some(t0)
                    } else { None };
                    s.barrier_between(
                        &[pf_attn_f32_ref],
                        &[&pf_sdpa_out],
                    );
                    mlx_native::ops::transpose::permute_021_f32(
                        s.encoder_mut(), reg, metal_dev,
                        pf_attn_f32_ref, &pf_sdpa_out,
                        nh, seq_len, hd,
                    ).map_err(|e| anyhow::anyhow!("non-FA attn permute L{layer_idx}: {e}"))?;
                    if let Some(t0) = nofa_perm_t0 {
                        s.finish().map_err(|e| anyhow::anyhow!("noFA perm post-finish L{layer_idx}: {e}"))?;
                        let dt_ns = t0.elapsed().as_nanos() as u64;
                        PROFILE_B_NOFA_PERM_NS.fetch_add(dt_ns, Ordering::Relaxed);
                        PROFILE_B_NOFA_PERM_COUNT.fetch_add(1, Ordering::Relaxed);
                        s = exec.begin().map_err(|e| anyhow::anyhow!("noFA perm reopen L{layer_idx}: {e}"))?;
                    }
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
                    if xlen_sdpa_mode {
                        // ADR-030 iter-78 — pre-SDPA hybrid_kv write.
                        // The standard K/V copy to hybrid_kv (line 1572+) happens
                        // AFTER SDPA, but xlen SDPA needs hybrid_kv to be
                        // populated at THIS round's positions
                        // [start_pos..start_pos+seq_len) BEFORE reading.  Hoist
                        // the F32→F16 K/V writes here.  The post-SDPA copy at
                        // line 1572+ still runs and writes the SAME data
                        // (idempotent), wasteful but correct.
                        let hybrid_kv_vec = self.hybrid_kv.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("xlen SDPA L{layer_idx}: hybrid_kv not allocated — HF2Q_FULL_F16_KV=1 required"))?;
                        let layer_kv = &hybrid_kv_vec[layer_idx];
                        let xlen_dst_start = start_pos as u32;
                        let xlen_n_copy = seq_len as u32;
                        let xlen_src_off: u32 = 0;
                        let xlen_hb_cap = layer_kv.capacity as u32;
                        s.barrier_between(&[&pf_k_normed], &[&layer_kv.k]);
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            &pf_k_normed,
                            &layer_kv.k,
                            nkv as u32, hd as u32,
                            xlen_hb_cap, xlen_dst_start, xlen_n_copy, xlen_src_off,
                        ).map_err(|e| anyhow::anyhow!("xlen pre-SDPA K copy L{layer_idx}: {e}"))?;
                        // V is F16 when HF2Q_FULL_F16_KV=1 (validated above).
                        if layer_kv.v_packed.dtype() == mlx_native::DType::F16 {
                            s.barrier_between(&[&pf_v_normed], &[&layer_kv.v_packed]);
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16(
                                s.encoder_mut(), reg, metal_dev,
                                &pf_v_normed,
                                &layer_kv.v_packed,
                                nkv as u32, hd as u32,
                                xlen_hb_cap, xlen_dst_start, xlen_n_copy, xlen_src_off,
                            ).map_err(|e| anyhow::anyhow!("xlen pre-SDPA V copy L{layer_idx}: {e}"))?;
                        } else {
                            anyhow::bail!(
                                "xlen SDPA L{layer_idx}: V is not F16 (got {:?}); HF2Q_FULL_F16_KV=1 required",
                                layer_kv.v_packed.dtype()
                            );
                        }

                        // ADR-030 iter-99 — also write BF16 cache pre-SDPA so
                        // xlen branch sees verify positions populated when it
                        // reads (iter-100+ swap).  Writes same logical data
                        // as F16 hybrid_kv pre-SDPA write above but at BF16
                        // precision direct from pf_k_perm.  Idempotent with
                        // iter-98's post-SDPA bf16 hook.
                        if let (Some(ref bf16_k), Some(ref bf16_v)) =
                            (&layer_kv.bf16_xlen_k, &layer_kv.bf16_xlen_v) {
                            s.barrier_between(&[&pf_k_perm], &[bf16_k]);
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major(
                                s.encoder_mut(), reg, metal_dev,
                                &pf_k_perm, bf16_k,
                                nkv as u32, hd as u32,
                                xlen_hb_cap, xlen_dst_start, xlen_n_copy, xlen_src_off,
                                seq_len as u32,
                            ).map_err(|e| anyhow::anyhow!("xlen pre-SDPA bf16 K L{layer_idx}: {e}"))?;
                            s.barrier_between(&[&pf_v_perm], &[bf16_v]);
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major(
                                s.encoder_mut(), reg, metal_dev,
                                &pf_v_perm, bf16_v,
                                nkv as u32, hd as u32,
                                xlen_hb_cap, xlen_dst_start, xlen_n_copy, xlen_src_off,
                                seq_len as u32,
                            ).map_err(|e| anyhow::anyhow!("xlen pre-SDPA bf16 V L{layer_idx}: {e}"))?;
                        }

                        // ADR-030 iter-80 — hypothesis 2 (GPU ordering)
                        // FALSIFIED: inserting s.finish() between pre-SDPA
                        // K/V writes and SDPA dispatch yields IDENTICAL
                        // output, so the alternating-zeros bug is NOT a
                        // memory ordering race.  Determinism + identical
                        // output → bug is in cast chain, kernel params, or
                        // layout (hypothesis 1 or 3 from iter-79 plan).

                        // ADR-030 iter-77 — cross-length SDPA verify path.
                        // Cast pf_q_perm BF16 → F32 → F16, call
                        // dispatch_flash_attn_prefill_f16_d256_resume with
                        // K/V from hybrid_kv (already F16) at slot capacity,
                        // cast output F16 → F32 → BF16 back to pf_sdpa_out_perm.
                        let q_f32 = pf_q_f32_xlen.as_ref().expect("xlen Q f32 buf");
                        let q_f16 = pf_q_f16_xlen.as_ref().expect("xlen Q f16 buf");
                        let out_f16 = pf_out_f16_xlen.as_ref().expect("xlen out f16 buf");
                        let out_f32 = pf_out_f32_xlen.as_ref().expect("xlen out f32 buf");
                        let q_n_elems = nh * seq_len * hd;
                        s.barrier_between(&[&pf_q_perm], &[q_f32]);
                        mlx_native::ops::elementwise::cast(
                            s.encoder_mut(), reg, metal_dev,
                            &pf_q_perm, q_f32, q_n_elems,
                            mlx_native::ops::elementwise::CastDirection::BF16ToF32,
                        ).map_err(|e| anyhow::anyhow!("xlen Q BF16->F32 L{layer_idx}: {e}"))?;
                        s.barrier_between(&[q_f32], &[q_f16]);
                        mlx_native::ops::elementwise::cast(
                            s.encoder_mut(), reg, metal_dev,
                            q_f32, q_f16, q_n_elems,
                            mlx_native::ops::elementwise::CastDirection::F32ToF16,
                        ).map_err(|e| anyhow::anyhow!("xlen Q F32->F16 L{layer_idx}: {e}"))?;
                        // ADR-030 iter-82 — runtime cast-chain diagnostic
                        // (HF2Q_DFLASH_XLEN_DEBUG=1).  Commits the session,
                        // reads Q/K/V/out at L0 + final layer only, prints
                        // first 8 values each.  Lets us bisect between cast
                        // corruption (Q wrong), K/V data integrity (K/V
                        // wrong), kernel misconfig (Q/K/V fine, out wrong),
                        // AND downstream corruption (L0 fine, final wrong).
                        let xlen_debug = std::env::var("HF2Q_DFLASH_XLEN_DEBUG").as_deref() == Ok("1");
                        if xlen_debug {
                            s.finish().map_err(|e| anyhow::anyhow!("xlen debug pre-SDPA finish: {e}"))?;
                            let q_slice = q_f16.as_slice::<half::f16>()
                                .map_err(|e| anyhow::anyhow!("xlen debug Q slice: {e}"))?;
                            let k_slice = layer_kv.k.as_slice::<half::f16>()
                                .map_err(|e| anyhow::anyhow!("xlen debug K slice: {e}"))?;
                            let v_slice = layer_kv.v_packed.as_slice::<half::f16>()
                                .map_err(|e| anyhow::anyhow!("xlen debug V slice: {e}"))?;
                            let qstart = 0usize; // [h=0, t=0, d=0..8]
                            let kstart_0 = 0usize; // [h=0, p=0, d=0..8]
                            // ADR-030 iter-91: also dump position 10 (= persisted
                            // first_token K from R1 verify) AND position start_pos-1
                            // (= persisted committed_R1 K from previous round) to
                            // bisect cross-round K state.
                            let kstart_sp = (start_pos as usize) * (hd as usize); // [h=0, p=start_pos, d=0..8]
                            let kstart_p10 = 10usize * (hd as usize); // [h=0, p=10, d=0..8]
                            let kstart_sp_m1 = if start_pos > 0 {
                                (start_pos as usize - 1) * (hd as usize)
                            } else { 0 };
                            eprintln!(
                                "[XLEN_DEBUG sliding L{} verify start_pos={} seq_len={}]\n  \
                                 Q[h=0,t=0,d=0..8] = {:?}\n  \
                                 K[h=0,p=0,d=0..8] = {:?}\n  \
                                 K[h=0,p=10,d=0..8] = {:?}\n  \
                                 K[h=0,p={},d=0..8] = {:?}\n  \
                                 K[h=0,p={},d=0..8] = {:?}\n  \
                                 V[h=0,p=0,d=0..8] = {:?}\n  \
                                 V[h=0,p={},d=0..8] = {:?}",
                                layer_idx, start_pos, seq_len,
                                &q_slice[qstart..qstart + 8],
                                &k_slice[kstart_0..kstart_0 + 8],
                                &k_slice[kstart_p10..kstart_p10 + 8],
                                start_pos - 1, &k_slice[kstart_sp_m1..kstart_sp_m1 + 8],
                                start_pos, &k_slice[kstart_sp..kstart_sp + 8],
                                &v_slice[kstart_0..kstart_0 + 8],
                                start_pos, &v_slice[kstart_sp..kstart_sp + 8],
                            );
                            // ADR-030 iter-102 — BF16 cache readback parallel
                            // to F16 hybrid_kv readback above.  When iter-100's
                            // D=256 BF16-cache path is active, the SDPA reads
                            // from `bf16_xlen_k`, NOT `layer_kv.k`.  Comparing
                            // F16 vs BF16 cache at the SAME positions localises
                            // whether 6-tok failure is BF16-cache divergence or
                            // downstream of SDPA.  Indexing matches the head-
                            // major `[nkv, capacity, head_dim]` layout that
                            // `dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major`
                            // writes (`slot = dst_pos % capacity` for sliding).
                            if let Some(ref bf16_k) = layer_kv.bf16_xlen_k {
                                let bk_slice = bf16_k.as_slice::<half::bf16>()
                                    .map_err(|e| anyhow::anyhow!("xlen debug bf16 K slice: {e}"))?;
                                let cap = layer_kv.capacity;
                                let slot_p0    = 0usize;
                                let slot_p10   = 10 % cap;
                                let slot_sp_m1 = if start_pos > 0 { (start_pos as usize - 1) % cap } else { 0 };
                                let slot_sp    = (start_pos as usize) % cap;
                                let bk_p0    = slot_p0    * hd as usize;
                                let bk_p10   = slot_p10   * hd as usize;
                                let bk_sp_m1 = slot_sp_m1 * hd as usize;
                                let bk_sp    = slot_sp    * hd as usize;
                                eprintln!(
                                    "  BF16_K[h=0,p=0,d=0..8]   = {:?}\n  \
                                     BF16_K[h=0,p=10,d=0..8]  = {:?}\n  \
                                     BF16_K[h=0,p={},d=0..8]   = {:?}\n  \
                                     BF16_K[h=0,p={},d=0..8]   = {:?}",
                                    &bk_slice[bk_p0..bk_p0 + 8],
                                    &bk_slice[bk_p10..bk_p10 + 8],
                                    start_pos - 1, &bk_slice[bk_sp_m1..bk_sp_m1 + 8],
                                    start_pos,     &bk_slice[bk_sp..bk_sp + 8],
                                );
                            }
                            if let Some(ref bf16_v) = layer_kv.bf16_xlen_v {
                                let bv_slice = bf16_v.as_slice::<half::bf16>()
                                    .map_err(|e| anyhow::anyhow!("xlen debug bf16 V slice: {e}"))?;
                                let cap = layer_kv.capacity;
                                let slot_p0 = 0usize;
                                let slot_sp = (start_pos as usize) % cap;
                                let bv_p0 = slot_p0 * hd as usize;
                                let bv_sp = slot_sp * hd as usize;
                                eprintln!(
                                    "  BF16_V[h=0,p=0,d=0..8]   = {:?}\n  \
                                     BF16_V[h=0,p={},d=0..8]   = {:?}",
                                    &bv_slice[bv_p0..bv_p0 + 8],
                                    start_pos, &bv_slice[bv_sp..bv_sp + 8],
                                );
                            }
                            s = exec.begin().map_err(|e| anyhow::anyhow!("xlen debug post-dump reopen: {e}"))?;
                        }

                        // ADR-030 iter-100 — D=256 xlen SDPA via BF16 resume +
                        // bf16_xlen_k/v cache reads.  BF16 cache populated
                        // bit-identical to Option C's pf_k_perm (single
                        // F32→BF16 rounding at fused_head_norm_rope), so
                        // SDPA reads precision-equivalent K/V to Option C.
                        // Eliminates the F16-roundtrip precision drift
                        // root-caused at iter-92/93.  Env-gated to allow
                        // fallback to F16 path on regression.
                        let use_bf16_xlen = std::env::var("HF2Q_DFLASH_XLEN_BF16").as_deref() != Ok("0")
                            && layer_kv.bf16_xlen_k.is_some()
                            && layer_kv.bf16_xlen_v.is_some();
                        if use_bf16_xlen {
                            let bf16_k = layer_kv.bf16_xlen_k.as_ref().unwrap();
                            let bf16_v = layer_kv.bf16_xlen_v.as_ref().unwrap();
                            s.barrier_between(&[&pf_q_perm, bf16_k, bf16_v], &[&pf_sdpa_out_perm]);
                            mlx_native::ops::flash_attn_prefill::
                                dispatch_flash_attn_prefill_bf16_d256_resume(
                                s.encoder_mut(), dev, reg,
                                &pf_q_perm, bf16_k, bf16_v,
                                &pf_sdpa_out_perm,
                                &mlx_native::ops::flash_attn_prefill::FlashAttnPrefillResumeParams {
                                    n_heads: nh as u32,
                                    n_kv_heads: nkv as u32,
                                    head_dim: hd as u32,
                                    seq_len_q: seq_len as u32,
                                    seq_len_k: (start_pos + seq_len) as u32,
                                    batch: 1,
                                    scale: 1.0,
                                    do_causal: true,
                                    q_offset_in_k: start_pos as u32,
                                    kv_capacity: layer_kv.capacity as u32,
                                },
                            ).map_err(|e| anyhow::anyhow!("xlen sliding BF16 SDPA L{layer_idx}: {e}"))?;
                        } else {
                            s.barrier_between(&[q_f16, &layer_kv.k, &layer_kv.v_packed], &[out_f16]);
                            mlx_native::ops::flash_attn_prefill::
                                dispatch_flash_attn_prefill_f16_d256_resume(
                                s.encoder_mut(), dev, reg,
                                q_f16, &layer_kv.k, &layer_kv.v_packed,
                                out_f16,
                                &mlx_native::ops::flash_attn_prefill::FlashAttnPrefillResumeParams {
                                    n_heads: nh as u32,
                                    n_kv_heads: nkv as u32,
                                    head_dim: hd as u32,
                                    seq_len_q: seq_len as u32,
                                    seq_len_k: (start_pos + seq_len) as u32,
                                    batch: 1,
                                    scale: 1.0,
                                    do_causal: true,
                                    q_offset_in_k: start_pos as u32,
                                    kv_capacity: layer_kv.capacity as u32,
                                },
                            ).map_err(|e| anyhow::anyhow!("xlen sliding F16 SDPA L{layer_idx}: {e}"))?;
                        }
                        // F16-path output cast (BF16-path writes pf_sdpa_out_perm directly).
                        if !use_bf16_xlen {
                            if xlen_debug {
                                s.finish().map_err(|e| anyhow::anyhow!("xlen debug post-SDPA finish: {e}"))?;
                                let out_slice = out_f16.as_slice::<half::f16>()
                                    .map_err(|e| anyhow::anyhow!("xlen debug out slice: {e}"))?;
                                let n_used = (nh * seq_len * hd) as usize;
                                let nan_count = out_slice[..n_used].iter().filter(|x| x.is_nan()).count();
                                let inf_count = out_slice[..n_used].iter().filter(|x| x.is_infinite()).count();
                                let max_abs = out_slice[..n_used].iter()
                                    .filter(|x| x.is_finite())
                                    .map(|x| x.to_f32().abs())
                                    .fold(0.0f32, f32::max);
                                eprintln!(
                                    "  OUT[h=0,t=0,d=0..8]={:?} nan={} inf={} max_abs={:.4e}",
                                    &out_slice[0..8], nan_count, inf_count, max_abs,
                                );
                                s = exec.begin().map_err(|e| anyhow::anyhow!("xlen debug post-out reopen: {e}"))?;
                            }
                            s.barrier_between(&[out_f16], &[out_f32]);
                            mlx_native::ops::elementwise::cast(
                                s.encoder_mut(), reg, metal_dev,
                                out_f16, out_f32, q_n_elems,
                                mlx_native::ops::elementwise::CastDirection::F16ToF32,
                            ).map_err(|e| anyhow::anyhow!("xlen O F16->F32 L{layer_idx}: {e}"))?;
                            s.barrier_between(&[out_f32], &[&pf_sdpa_out_perm]);
                            mlx_native::ops::elementwise::cast(
                                s.encoder_mut(), reg, metal_dev,
                                out_f32, &pf_sdpa_out_perm, q_n_elems,
                                mlx_native::ops::elementwise::CastDirection::F32ToBF16,
                            ).map_err(|e| anyhow::anyhow!("xlen O F32->BF16 L{layer_idx}: {e}"))?;
                        }
                    } else {
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
                    if use_fa_f16 {
                        // HF2Q_FA_F16=1: F16 FA path (kernel migration step 4).
                        //
                        // Source F16 Q/K/V from the F32 *_normed buffers
                        // directly via permute_021_f32_to_f16 (mlx-native
                        // step 4 kernel) — bypassing the BF16 pf_*_perm
                        // round-trip that left the step-3 wiring with
                        // BF16 precision.  F16 is sourced from F32 with
                        // a single rounding step at the F16 store, giving
                        // F16's full 10-bit-mantissa precision (~8× the
                        // BF16 7-bit mantissa).
                        let q_f16 = pf_q_perm_f16.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_q_perm_f16 not allocated"))?;
                        let k_f16 = pf_k_perm_f16.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_k_perm_f16 not allocated"))?;
                        let v_f16 = pf_v_perm_f16.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_v_perm_f16 not allocated"))?;
                        let out_f16 = pf_sdpa_out_perm_f16.as_mut()
                            .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_sdpa_out_perm_f16 not allocated"))?;
                        let mask_f16 = sliding_mask_f16.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("FA_F16: sliding_mask_f16 not allocated"))?;

                        let q_elems = (nh * seq_len * hd) as usize;

                        // pf_q_normed / pf_k_normed / pf_v_normed are F32 in
                        // natural [seq_len, n_heads, head_dim] layout.
                        // permute_021_f32_to_f16 writes F16 in permuted
                        // [n_heads, seq_len, head_dim] layout which is what
                        // dispatch_flash_attn_prefill_f16_d256_with_blk expects.
                        s.barrier_between(&[&pf_q_normed], &[q_f16]);
                        mlx_native::ops::transpose::permute_021_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            &pf_q_normed, q_f16,
                            seq_len, nh, hd,
                        ).map_err(|e| anyhow::anyhow!("FA_F16 Q permute+cast L{layer_idx}: {e}"))?;
                        s.barrier_between(&[&pf_k_normed], &[k_f16]);
                        mlx_native::ops::transpose::permute_021_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            &pf_k_normed, k_f16,
                            seq_len, nkv, hd,
                        ).map_err(|e| anyhow::anyhow!("FA_F16 K permute+cast L{layer_idx}: {e}"))?;
                        s.barrier_between(&[&pf_v_normed], &[v_f16]);
                        mlx_native::ops::transpose::permute_021_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            &pf_v_normed, v_f16,
                            seq_len, nkv, hd,
                        ).map_err(|e| anyhow::anyhow!("FA_F16 V permute+cast L{layer_idx}: {e}"))?;

                        s.barrier_between(
                            &[q_f16, k_f16, v_f16, mask_f16, &blk_sliding],
                            &[out_f16],
                        );
                        mlx_native::ops::flash_attn_prefill::
                            dispatch_flash_attn_prefill_f16_d256_with_blk(
                            s.encoder_mut(), dev, reg,
                            q_f16, k_f16, v_f16,
                            Some(mask_f16),
                            Some(&blk_sliding),
                            out_f16,
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
                        ).map_err(|e| anyhow::anyhow!("FA_F16 sliding L{layer_idx}: {e}"))?;

                        // Cast F16 output back to BF16 for o_proj input compat.
                        s.barrier_between(&[out_f16], &[&pf_sdpa_out_perm]);
                        mlx_native::ops::elementwise::cast(
                            s.encoder_mut(), reg, metal_dev,
                            out_f16, &pf_sdpa_out_perm, q_elems,
                            mlx_native::ops::elementwise::CastDirection::F16ToBF16,
                        ).map_err(|e| anyhow::anyhow!("FA_F16 O cast L{layer_idx}: {e}"))?;
                    } else {
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
                    } // end FA_F16 vs BF16 branch
                    } // end sliding else (existing non-xlen path)
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
                    // ADR-030 iter-82 — env-gated D=512 xlen bypass for bisection.
                    // HF2Q_DFLASH_XLEN_D512_OFF=1 routes the full-attn (D=512)
                    // layers through the STANDARD non-xlen path while keeping
                    // sliding (D=256) layers on the xlen path.  Used for
                    // root-causing iter-82's L29 NaN; left in for safety
                    // valve.
                    let xlen_d512_disabled = std::env::var("HF2Q_DFLASH_XLEN_D512_OFF").as_deref() == Ok("1");
                    // ADR-030 iter-84 — F16 D=512 attention OVERFLOWS at the
                    // final gemma-4 layer (L29) because Q magnitudes are
                    // ~16× larger there and the F16 exponent caps at 2^15.
                    // BF16 has the same exponent range as F32 so it's safe.
                    // Cast hybrid_kv K, V from F16 to BF16 via F32, then
                    // dispatch the new bf16 D=512 resume kernel.  pf_q_perm
                    // is already BF16, so Q needs no cast.  Output goes
                    // directly to pf_sdpa_out_perm (BF16).
                    if xlen_sdpa_mode && !xlen_d512_disabled {
                        // ADR-030 iter-78 — pre-SDPA hybrid_kv write (D=512).
                        let hybrid_kv_vec = self.hybrid_kv.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("xlen SDPA L{layer_idx}: hybrid_kv not allocated — HF2Q_FULL_F16_KV=1 required"))?;
                        let layer_kv = &hybrid_kv_vec[layer_idx];
                        let xlen_dst_start = start_pos as u32;
                        let xlen_n_copy = seq_len as u32;
                        let xlen_src_off: u32 = 0;
                        let xlen_hb_cap = layer_kv.capacity as u32;
                        s.barrier_between(&[&pf_k_normed], &[&layer_kv.k]);
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            &pf_k_normed,
                            &layer_kv.k,
                            nkv as u32, hd as u32,
                            xlen_hb_cap, xlen_dst_start, xlen_n_copy, xlen_src_off,
                        ).map_err(|e| anyhow::anyhow!("xlen pre-SDPA K copy L{layer_idx}: {e}"))?;
                        if layer_kv.v_packed.dtype() == mlx_native::DType::F16 {
                            s.barrier_between(&[&pf_v_normed], &[&layer_kv.v_packed]);
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16(
                                s.encoder_mut(), reg, metal_dev,
                                &pf_v_normed,
                                &layer_kv.v_packed,
                                nkv as u32, hd as u32,
                                xlen_hb_cap, xlen_dst_start, xlen_n_copy, xlen_src_off,
                            ).map_err(|e| anyhow::anyhow!("xlen pre-SDPA V copy L{layer_idx}: {e}"))?;
                        } else {
                            anyhow::bail!(
                                "xlen SDPA L{layer_idx}: V is not F16 (got {:?}); HF2Q_FULL_F16_KV=1 required",
                                layer_kv.v_packed.dtype()
                            );
                        }

                        // ADR-030 iter-84 — D=512 BF16 cross-length verify
                        // (kept as-is; F16→F32→BF16 cast path).  iter-100's
                        // BF16-cache attempt regressed for D=512; reverted to
                        // proven iter-84 path.  D=256 uses BF16-cache.
                        let _q_n_elems = nh * seq_len * hd;
                        let kv_full_elems = nkv * (layer_kv.capacity) * hd;
                        let f32_kv_scratch = alloc_f32(kv_full_elems, "xlen_d512_kv_f32_scratch")?;
                        let bf16_k = alloc_bf16(kv_full_elems, "xlen_d512_bf16_k")?;
                        let bf16_v = alloc_bf16(kv_full_elems, "xlen_d512_bf16_v")?;
                        s.barrier_between(&[&layer_kv.k], &[&f32_kv_scratch]);
                        mlx_native::ops::elementwise::cast(
                            s.encoder_mut(), reg, metal_dev,
                            &layer_kv.k, &f32_kv_scratch, kv_full_elems,
                            mlx_native::ops::elementwise::CastDirection::F16ToF32,
                        ).map_err(|e| anyhow::anyhow!("xlen D=512 K F16->F32 L{layer_idx}: {e}"))?;
                        s.barrier_between(&[&f32_kv_scratch], &[&bf16_k]);
                        mlx_native::ops::elementwise::cast(
                            s.encoder_mut(), reg, metal_dev,
                            &f32_kv_scratch, &bf16_k, kv_full_elems,
                            mlx_native::ops::elementwise::CastDirection::F32ToBF16,
                        ).map_err(|e| anyhow::anyhow!("xlen D=512 K F32->BF16 L{layer_idx}: {e}"))?;
                        s.barrier_between(&[&layer_kv.v_packed], &[&f32_kv_scratch]);
                        mlx_native::ops::elementwise::cast(
                            s.encoder_mut(), reg, metal_dev,
                            &layer_kv.v_packed, &f32_kv_scratch, kv_full_elems,
                            mlx_native::ops::elementwise::CastDirection::F16ToF32,
                        ).map_err(|e| anyhow::anyhow!("xlen D=512 V F16->F32 L{layer_idx}: {e}"))?;
                        s.barrier_between(&[&f32_kv_scratch], &[&bf16_v]);
                        mlx_native::ops::elementwise::cast(
                            s.encoder_mut(), reg, metal_dev,
                            &f32_kv_scratch, &bf16_v, kv_full_elems,
                            mlx_native::ops::elementwise::CastDirection::F32ToBF16,
                        ).map_err(|e| anyhow::anyhow!("xlen D=512 V F32->BF16 L{layer_idx}: {e}"))?;
                        let xlen_debug_d512 = std::env::var("HF2Q_DFLASH_XLEN_DEBUG").as_deref() == Ok("1");
                        if xlen_debug_d512 {
                            eprintln!(
                                "[XLEN_DEBUG global L{} verify start_pos={} seq_len={} hd={} F16-cast path]",
                                layer_idx, start_pos, seq_len, hd,
                            );
                            // ADR-030 iter-102 — D=512 BF16 cache readback.
                            // For global layers the SDPA reads from `bf16_k`
                            // (local scratch, F16→F32→BF16 cast).  Dumping the
                            // persistent `bf16_xlen_k` (populated post-SDPA from
                            // pf_k_perm at iter-98+) lets the operator compare
                            // F16-cast-K vs BF16-direct-K bits at the same
                            // positions — localising whether the 6-tok failure
                            // is driven by the D=512 cast chain.
                            s.finish().map_err(|e| anyhow::anyhow!("xlen debug D512 pre-read finish: {e}"))?;
                            let f16_k_slice = layer_kv.k.as_slice::<half::f16>()
                                .map_err(|e| anyhow::anyhow!("xlen debug D512 F16 K slice: {e}"))?;
                            let cap_g = layer_kv.capacity;
                            let slot_p0    = 0usize;
                            let slot_sp_m1 = if start_pos > 0 { (start_pos as usize - 1) % cap_g } else { 0 };
                            let slot_sp    = (start_pos as usize) % cap_g;
                            let fk_p0    = slot_p0    * hd as usize;
                            let fk_sp_m1 = slot_sp_m1 * hd as usize;
                            let fk_sp    = slot_sp    * hd as usize;
                            eprintln!(
                                "  F16_K[h=0,p=0,d=0..8]    = {:?}\n  \
                                 F16_K[h=0,p={},d=0..8]    = {:?}\n  \
                                 F16_K[h=0,p={},d=0..8]    = {:?}",
                                &f16_k_slice[fk_p0..fk_p0 + 8],
                                start_pos - 1, &f16_k_slice[fk_sp_m1..fk_sp_m1 + 8],
                                start_pos,     &f16_k_slice[fk_sp..fk_sp + 8],
                            );
                            if let Some(ref bf16_kc) = layer_kv.bf16_xlen_k {
                                let bk_slice = bf16_kc.as_slice::<half::bf16>()
                                    .map_err(|e| anyhow::anyhow!("xlen debug D512 bf16 K slice: {e}"))?;
                                eprintln!(
                                    "  BF16_K[h=0,p=0,d=0..8]   = {:?}\n  \
                                     BF16_K[h=0,p={},d=0..8]   = {:?}\n  \
                                     BF16_K[h=0,p={},d=0..8]   = {:?}",
                                    &bk_slice[fk_p0..fk_p0 + 8],
                                    start_pos - 1, &bk_slice[fk_sp_m1..fk_sp_m1 + 8],
                                    start_pos,     &bk_slice[fk_sp..fk_sp + 8],
                                );
                            }
                            s = exec.begin().map_err(|e| anyhow::anyhow!("xlen debug D512 post-K-dump reopen: {e}"))?;
                        }
                        s.barrier_between(&[&pf_q_perm, &bf16_k, &bf16_v], &[&pf_sdpa_out_perm]);
                        mlx_native::ops::flash_attn_prefill_d512::
                            dispatch_flash_attn_prefill_bf16_d512_resume(
                            s.encoder_mut(), dev, reg,
                            &pf_q_perm, &bf16_k, &bf16_v,
                            &pf_sdpa_out_perm,
                            &mlx_native::ops::flash_attn_prefill::FlashAttnPrefillResumeParams {
                                n_heads: nh as u32,
                                n_kv_heads: nkv as u32,
                                head_dim: hd as u32,
                                seq_len_q: seq_len as u32,
                                seq_len_k: (start_pos + seq_len) as u32,
                                batch: 1,
                                scale: 1.0,
                                do_causal: true,
                                q_offset_in_k: start_pos as u32,
                                kv_capacity: layer_kv.capacity as u32,
                            },
                        ).map_err(|e| anyhow::anyhow!("xlen global BF16 SDPA L{layer_idx}: {e}"))?;
                        if xlen_debug_d512 {
                            s.finish().map_err(|e| anyhow::anyhow!("xlen debug D512 BF16 post-SDPA finish: {e}"))?;
                            let out_slice = pf_sdpa_out_perm.as_slice::<half::bf16>()
                                .map_err(|e| anyhow::anyhow!("xlen debug D512 BF16 out slice: {e}"))?;
                            let n_used = (nh * seq_len * hd) as usize;
                            let nan_count = out_slice[..n_used].iter().filter(|x| x.is_nan()).count();
                            let max_abs = out_slice[..n_used].iter()
                                .filter(|x| x.is_finite())
                                .map(|x| x.to_f32().abs())
                                .fold(0.0f32, f32::max);
                            eprintln!("  BF16 OUT[h=0,t=0,d=0..8]={:?} nan={} max_abs={:.4e}",
                                &out_slice[0..8], nan_count, max_abs);
                            s = exec.begin().map_err(|e| anyhow::anyhow!("xlen debug D512 BF16 reopen: {e}"))?;
                        }
                        // Bind unused buffers to silence unused-var warnings —
                        // these are leftover from the F16 path we replaced.
                        let _ = (&pf_q_f32_xlen, &pf_q_f16_xlen, &pf_out_f16_xlen, &pf_out_f32_xlen);
                    } else if use_fa_f16 {
                        // Bug A fix (2026-05-17): F16 D=512 FA path — keeps Q
                        // in F16 (10-bit mantissa) instead of BF16 (7-bit) for
                        // the global D=512 attention layers.  Worst-case
                        // accumulated relative error over the 512-element dot
                        // product drops from ~18% (BF16) to ~2.2% (F16) —
                        // below the empirical argmax-flip threshold that
                        // produced the `Format: Hemoglobin` greedy loop at
                        // decode-pos ~70 on 300-item enumeration probes.
                        //
                        // Source F16 Q/K/V from F32 *_normed buffers via
                        // permute_021_f32_to_f16 (single rounding step;
                        // bypasses the BF16 round-trip through pf_*_perm).
                        let q_f16 = pf_q_perm_f16.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_q_perm_f16 not allocated"))?;
                        let k_f16 = pf_k_perm_f16.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_k_perm_f16 not allocated"))?;
                        let v_f16 = pf_v_perm_f16.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_v_perm_f16 not allocated"))?;
                        let out_f16 = pf_sdpa_out_perm_f16.as_mut()
                            .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_sdpa_out_perm_f16 not allocated"))?;
                        let mask_f16 = global_mask_f16.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("FA_F16: global_mask_f16 not allocated"))?;

                        let q_elems = (nh * seq_len * hd) as usize;

                        s.barrier_between(&[&pf_q_normed], &[q_f16]);
                        mlx_native::ops::transpose::permute_021_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            &pf_q_normed, q_f16,
                            seq_len, nh, hd,
                        ).map_err(|e| anyhow::anyhow!("FA_F16 global Q permute+cast L{layer_idx}: {e}"))?;
                        s.barrier_between(&[&pf_k_normed], &[k_f16]);
                        mlx_native::ops::transpose::permute_021_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            &pf_k_normed, k_f16,
                            seq_len, nkv, hd,
                        ).map_err(|e| anyhow::anyhow!("FA_F16 global K permute+cast L{layer_idx}: {e}"))?;
                        s.barrier_between(&[&pf_v_normed], &[v_f16]);
                        mlx_native::ops::transpose::permute_021_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            &pf_v_normed, v_f16,
                            seq_len, nkv, hd,
                        ).map_err(|e| anyhow::anyhow!("FA_F16 global V permute+cast L{layer_idx}: {e}"))?;

                        s.barrier_between(
                            &[q_f16, k_f16, v_f16, mask_f16, &blk_global],
                            &[out_f16],
                        );
                        mlx_native::ops::flash_attn_prefill_d512::
                            dispatch_flash_attn_prefill_f16_d512_with_blk(
                            s.encoder_mut(), dev, reg,
                            q_f16, k_f16, v_f16,
                            Some(mask_f16),
                            Some(&blk_global),
                            out_f16,
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
                        ).map_err(|e| anyhow::anyhow!("FA_F16 global D=512 L{layer_idx}: {e}"))?;

                        // Cast F16 output back to BF16 for o_proj input compat.
                        s.barrier_between(&[out_f16], &[&pf_sdpa_out_perm]);
                        mlx_native::ops::elementwise::cast(
                            s.encoder_mut(), reg, metal_dev,
                            out_f16, &pf_sdpa_out_perm, q_elems,
                            mlx_native::ops::elementwise::CastDirection::F16ToBF16,
                        ).map_err(|e| anyhow::anyhow!("FA_F16 global O cast L{layer_idx}: {e}"))?;
                    } else {
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
                }
                // Wave P4.0 — close the FA-only session and accumulate
                // wall-clock time (commit_and_wait blocks until the GPU
                // finishes, so the measurement bounds true GPU work).
                if let Some(t0) = fa_start {
                    if is_sliding {
                        bucket_finish!(s, exec, t0, &PROFILE_FA_SW_NS, &PROFILE_FA_SW_COUNT, 1, "fa_sw");
                    } else {
                        bucket_finish!(s, exec, t0, &PROFILE_FA_GL_NS, &PROFILE_FA_GL_COUNT, 1, "fa_gl");
                    }
                }

                // Wave P4.19 — POST_FA_PERMUTE elimination.
                //
                // Pre-P4.19 we ran a dedicated `permute_021_bf16_to_f32`
                // dispatch here that transposed `pf_sdpa_out_perm`
                // ([n_heads, seq_len, head_dim] bf16, natively written by
                // flash-attention) into `pf_sdpa_out`
                // ([seq_len, n_heads*head_dim] f32) for the O-proj
                // matmul's f32 input contract.  The dispatch cost ~11 ms
                // in the bucket profile at pp2455 and ~30 MB of write+
                // read traffic per layer (~900 MB across the prefill).
                //
                // Wave P4.19 teaches the O-proj matmul kernel to read
                // `pf_sdpa_out_perm` DIRECTLY via a bf16-input perm021
                // variant (`kernel_mul_mm_q{4_0,6_K}_tensor_bf16_perm021`
                // in quantized_matmul_mm_tensor.metal).  The B-stage of
                // the kernel maps logical (m, k) to physical
                // (h = k/hd, t = m, f = k%hd) and converts bf16→half at
                // staging time.
                //
                // Byte-exact equivalence:
                //   old path: bf16 -> (cast dispatch) f32 -> (mm B-stage) half
                //   new path: bf16 -> (mm B-stage) half
                // Both produce identical half bits because bfloat->float
                // is pure bit-expansion and the float->half RNE round
                // drops the zero-pad low 16 bits without changing the
                // result.  Verified against sourdough gate.
                } // end of !use_no_fa branch

                // 8. O-proj (m = seq_len): [seq_len, nh*hd] -> [seq_len, hs]
                //
                // Branches on use_no_fa:
                //   * !use_no_fa (default FA path): O-proj reads pf_sdpa_out_perm
                //       [n_heads, seq_len, head_dim] bf16 directly via the
                //       perm021 tensor-mm variant (Wave P4.19) — no permute
                //       dispatch needed.
                //   * use_no_fa (experimental tensor-mm attention path):
                //       the NO_FA attention steps already permute the
                //       attention output into [seq_len, n_heads, head_dim]
                //       f32 at pf_sdpa_out, so O-proj reads pf_sdpa_out as
                //       before via the standard f32 tensor-mm path.
                let o_t0 = if std::env::var("HF2Q_PROFILE_MM").is_ok() || profile_buckets_on {
                    s.finish().map_err(|e| anyhow::anyhow!("MM-profile pre-finish (O) L{layer_idx}: {e}"))?;
                    let t0 = std::time::Instant::now();
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MM-profile begin (O) L{layer_idx}: {e}"))?;
                    Some(t0)
                } else { None };
                // ADR-029 iter-82 H62: O-proj reads pf_sdpa_out (NO_FA layout
                // f32) only when this layer actually routed through NO_FA
                // (i.e. !is_sliding AND use_no_fa).  Sliding layers under
                // FA_SW populated pf_sdpa_out_perm (FA layout bf16) and
                // need the FA-style O-proj branch.
                if route_through_nofa {
                    s.barrier_between(
                        &[&pf_sdpa_out, &self.layers[layer_idx].attn.o_proj.buffer],
                        &[&pf_attn_out],
                    );
                    dispatch_qmatmul(&mut s, reg, dev, &pf_sdpa_out,
                        &self.layers[layer_idx].attn.o_proj,
                        &mut pf_attn_out, seq_len as u32)?;
                } else {
                    let o_info = &self.layers[layer_idx].attn.o_proj.info;
                    let perm021_params = mlx_native::GgmlQuantizedMatmulPerm021Params {
                        m: seq_len as u32,
                        n: o_info.rows as u32,
                        k: o_info.cols as u32,
                        head_dim: hd as u32,
                        ggml_type: o_info.ggml_dtype,
                    };
                    // ADR-029 iter-36 H28-D — route O-proj through F16-weight
                    // perm021 variant when MlxQWeight.f16_shadow was populated
                    // at load (HF2Q_F16_SHADOW=1 by default per iter-31).
                    // The F16 kernel reads the half weight directly, bypassing
                    // the per-call quantized dequant.  Falls back to the
                    // quantized perm021 kernel when no shadow exists (env
                    // opt-out or no F16 buffer for this layer).
                    if let Some(f16_w) = self.layers[layer_idx].attn.o_proj.f16_shadow.as_ref() {
                        s.barrier_between(
                            &[&pf_sdpa_out_perm, f16_w],
                            &[&pf_attn_out],
                        );
                        mlx_native::quantized_matmul_mm_tensor_perm021_f16(
                            s.encoder_mut(), reg, dev,
                            &pf_sdpa_out_perm,
                            f16_w,
                            &mut pf_attn_out,
                            &perm021_params,
                        ).map_err(|e| anyhow::anyhow!("batched O-proj perm021_f16 L{layer_idx}: {e}"))?;
                    } else {
                        s.barrier_between(
                            &[&pf_sdpa_out_perm, &self.layers[layer_idx].attn.o_proj.buffer],
                            &[&pf_attn_out],
                        );
                        mlx_native::quantized_matmul_mm_tensor_perm021(
                            s.encoder_mut(), reg, dev,
                            &pf_sdpa_out_perm,
                            &self.layers[layer_idx].attn.o_proj.buffer,
                            &mut pf_attn_out,
                            &perm021_params,
                        ).map_err(|e| anyhow::anyhow!("batched O-proj perm021 L{layer_idx}: {e}"))?;
                    }
                }
                if let Some(t0) = o_t0 {
                    bucket_finish!(s, exec, t0, &PROFILE_O_MM_NS, &PROFILE_O_MM_COUNT, 1, "O_mm");
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
                    bucket_finish!(s, exec, t0, &PROFILE_B_POST_ATTN_NORM_ADD_NS, &PROFILE_B_POST_ATTN_NORM_ADD_COUNT, 1, "post_attn_norm_add");
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
                if !dense_kvs_vec[layer_idx].is_sliding && (start_pos + seq_len) > layer_cap {
                    anyhow::bail!(
                        "batched prefill L{}: start_pos={} + seq_len={} = {} exceeds global dense cap={} — \
                         increase linear_capacity allocation (max_decode_tokens param)",
                        layer_idx, start_pos, seq_len, start_pos + seq_len, layer_cap);
                }
                let n_copy = seq_len.min(layer_cap);
                let src_tok_offset = (seq_len - n_copy) as u32;
                // ADR-030 iter-64 (extend-mode write-position fix):
                // include start_pos in the K/V cache destination offset.
                // Before iter-64 `dst_seq_pos_start = src_tok_offset` which
                // is the CHUNK-INTERNAL offset (0 for single-chunk
                // prefill), causing all writes to go to position 0
                // regardless of `start_pos`.  iter-137/138 had fixed
                // pf_positions (RoPE) and write_pos (cache cursor) but
                // missed the kv_cache_copy/quantize destination offset.
                // For start_pos=0 (all production cmd_generate / parity /
                // engine callers) the expression reduces to src_tok_offset
                // — bit-identical behavior.
                let dst_seq_pos_start = (start_pos as u32) + src_tok_offset;
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
                    bucket_finish!(s, exec, t0, &PROFILE_B_KV_COPY_NS, &PROFILE_B_KV_COPY_COUNT, 1, "kv_copy");
                }

                // ADR-010 iter-64 — HB encode K/V into leg_hb_encoded for
                // batched prefill (mirrors per-token forward_prefill.rs:1234-1272).
                // Decode reads from leg_hb_encoded via flash_attn_vec_tq_hb;
                // without this block the buffers stay zero-initialized →
                // gibberish post-prefill. Use the SAME write-position
                // semantics as the dense KV copy above (dst_seq_pos_start /
                // n_copy / src_tok_offset) so dense and HB caches stay in
                // lockstep on sliding-window ring positions.
                if tq_codebook_bits_prefill >= 5 && !INVESTIGATION_ENV.skip_tq_encode {
                    if INVESTIGATION_ENV.hybrid_kv {
                        // ADR-028 Phase 10c (iter-348): hybrid F16-K + TQ-HB-V
                        // batched-prefill encode path. F32 K → F16 K (sequence
                        // copy) + V-only TQ-HB sequence encode.
                        if let Some(ref hybrid_kv) = self.hybrid_kv {
                            let hb_cap = hybrid_kv[layer_idx].capacity as u32;
                            let hb_is_ring = hybrid_kv[layer_idx].is_sliding;
                            s.barrier_between(
                                &[&pf_k_normed, &pf_v_normed],
                                &[&hybrid_kv[layer_idx].k,
                                  &hybrid_kv[layer_idx].v_packed, &hybrid_kv[layer_idx].v_norms],
                            );
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16(
                                s.encoder_mut(), reg, metal_dev,
                                &pf_k_normed,
                                &hybrid_kv[layer_idx].k,
                                nkv as u32, hd as u32,
                                hb_cap, dst_seq_pos_start, n_copy as u32, src_tok_offset,
                            ).map_err(|e| anyhow::anyhow!("batched hybrid F16 K L{layer_idx}: {e}"))?;
                            // ADR-029 iter-20 H27: if V buffer is F16-typed
                            // (HF2Q_FULL_F16_KV=1) → plain F32→F16 cast (no
                            // TQ-HB quantize).  Otherwise legacy V-only TQ-HB
                            // no-FWHT (Phase 10e.5).
                            if hybrid_kv[layer_idx].v_packed.dtype() == mlx_native::DType::F16 {
                                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16(
                                    s.encoder_mut(), reg, metal_dev,
                                    &pf_v_normed,
                                    &hybrid_kv[layer_idx].v_packed,
                                    nkv as u32, hd as u32,
                                    hb_cap, dst_seq_pos_start, n_copy as u32, src_tok_offset,
                                ).map_err(|e| anyhow::anyhow!("batched hybrid F16 V L{layer_idx}: {e}"))?;
                            } else {
                                // BUG-coherence fix (supersedes Phase 10e.5 iter-351):
                                // batched FWHT V quantize.  See forward_mlx.rs
                                // ~L3724 for empirical justification.  SDPA-side
                                // fwht_sign_undo at forward_mlx.rs's hybrid branch
                                // recovers raw output during decode.  Prefill SDPA
                                // operates on its own pf_k_normed/pf_v_normed (not
                                // the cache) so prefill output is unaffected.
                                mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq(
                                    s.encoder_mut(), reg, metal_dev,
                                    &pf_v_normed,
                                    &hybrid_kv[layer_idx].v_packed,
                                    &hybrid_kv[layer_idx].v_norms,
                                    nkv as u32, hd as u32,
                                    hb_cap, dst_seq_pos_start, n_copy as u32, src_tok_offset,
                                    hb_is_ring, tq_scale_factor_d512, tq_codebook_bits_prefill,
                                ).map_err(|e| anyhow::anyhow!("batched hybrid V FWHT quant L{layer_idx}: {e}"))?;
                            }
                            // ADR-030 iter-98 — populate BF16 xlen cache from
                            // pf_k_perm/pf_v_perm BF16 head-major (single
                            // F32→BF16 rounding at fused_head_norm_rope's
                            // output).  Bit-identical to what Option C's
                            // SDPA reads.  Kernel byte-identity verified at
                            // iter-97 (mlx-native commit bf1befd).
                            // Write-only this iteration — no SDPA reads yet
                            // (iter-99+ will swap xlen branch to use cache).
                            if let (Some(ref bf16_k), Some(ref bf16_v)) =
                                (&hybrid_kv[layer_idx].bf16_xlen_k,
                                 &hybrid_kv[layer_idx].bf16_xlen_v) {
                                s.barrier_between(&[&pf_k_perm], &[bf16_k]);
                                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major(
                                    s.encoder_mut(), reg, metal_dev,
                                    &pf_k_perm, bf16_k,
                                    nkv as u32, hd as u32,
                                    hb_cap, dst_seq_pos_start, n_copy as u32, src_tok_offset,
                                    seq_len as u32,
                                ).map_err(|e| anyhow::anyhow!("post-SDPA bf16 xlen K L{layer_idx}: {e}"))?;
                                s.barrier_between(&[&pf_v_perm], &[bf16_v]);
                                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major(
                                    s.encoder_mut(), reg, metal_dev,
                                    &pf_v_perm, bf16_v,
                                    nkv as u32, hd as u32,
                                    hb_cap, dst_seq_pos_start, n_copy as u32, src_tok_offset,
                                    seq_len as u32,
                                ).map_err(|e| anyhow::anyhow!("post-SDPA bf16 xlen V L{layer_idx}: {e}"))?;
                            }
                        }
                    } else if let Some(ref leg_hb_enc) = self.leg_hb_encoded {
                        let hb_cap = leg_hb_enc[layer_idx].capacity as u32;
                        let hb_is_ring = leg_hb_enc[layer_idx].is_sliding;
                        s.barrier_between(
                            &[&pf_k_normed],
                            &[&leg_hb_enc[layer_idx].k_packed, &leg_hb_enc[layer_idx].k_norms],
                        );
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq(
                            s.encoder_mut(), reg, metal_dev,
                            &pf_k_normed,
                            &leg_hb_enc[layer_idx].k_packed,
                            &leg_hb_enc[layer_idx].k_norms,
                            nkv as u32, hd as u32,
                            hb_cap, dst_seq_pos_start, n_copy as u32, src_tok_offset,
                            hb_is_ring, tq_scale_factor_d512, tq_codebook_bits_prefill,
                        ).map_err(|e| anyhow::anyhow!("batched HB encode K L{layer_idx}: {e}"))?;
                        s.barrier_between(
                            &[&pf_v_normed],
                            &[&leg_hb_enc[layer_idx].v_packed, &leg_hb_enc[layer_idx].v_norms],
                        );
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq(
                            s.encoder_mut(), reg, metal_dev,
                            &pf_v_normed,
                            &leg_hb_enc[layer_idx].v_packed,
                            &leg_hb_enc[layer_idx].v_norms,
                            nkv as u32, hd as u32,
                            hb_cap, dst_seq_pos_start, n_copy as u32, src_tok_offset,
                            hb_is_ring, tq_scale_factor_d512, tq_codebook_bits_prefill,
                        ).map_err(|e| anyhow::anyhow!("batched HB encode V L{layer_idx}: {e}"))?;
                    }
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
                    bucket_finish!(s, exec, t0, &PROFILE_B_TRIPLE_NORM_NS, &PROFILE_B_TRIPLE_NORM_COUNT, 1, "triple_norm");
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
                    bucket_finish!(s, exec, t0, &PROFILE_MLP_GUR_MM_NS, &PROFILE_MLP_GUR_MM_COUNT, 3, "gur_mm");
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
                    bucket_finish!(s, exec, t0, &PROFILE_B_GELU_MUL_ROUTING_NS, &PROFILE_B_GELU_MUL_ROUTING_COUNT, 2, "gelu_mul_routing");
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
                    bucket_finish!(s, exec, t0, &PROFILE_MLP_DN_MM_NS, &PROFILE_MLP_DN_MM_COUNT, 1, "mlp_dn");
                }

                // ADR-020 AC#5 Iter C2.3 — Gemma 4 MoE dispatch route
                // either through the legacy GGML id-mm pool (default) OR
                // the mlx-affine quantized_matmul_id_into kernel (when a
                // DWQ overlay was applied at load time).  The two are
                // mutually exclusive: overlay populates *_affine slots
                // and the legacy stacked_* buffers stay resident but
                // unused for the rest of the model lifetime.
                let gemma_moe_use_affine =
                    self.layers[layer_idx].moe.gate_up_affine.is_some()
                        && self.layers[layer_idx].moe.down_affine.is_some();
                if !gemma_moe_use_affine
                    && (self.layers[layer_idx].moe.stacked_gate_up.is_none()
                        || self.layers[layer_idx].moe.stacked_down.is_none())
                {
                    anyhow::bail!("batched prefill requires fused MoE _id path at L{layer_idx}");
                }
                let ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                let gu_w_buf: &mlx_native::MlxBuffer = if gemma_moe_use_affine {
                    &self.layers[layer_idx].moe.gate_up_affine.as_ref().unwrap().weight
                } else {
                    self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap()
                };
                s.barrier_between(
                    &[&pf_moe_norm_out, &pf_expert_ids, gu_w_buf],
                    &[&pf_moe_gate_up],
                );
                let profile_moe = std::env::var("HF2Q_PROFILE_MOE").is_ok() || profile_buckets_on;
                let moe_gu_t0 = if profile_moe {
                    s.finish().map_err(|e| anyhow::anyhow!("MoE-profile pre-finish (gu) L{layer_idx}: {e}"))?;
                    let t0 = std::time::Instant::now();
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MoE-profile begin (gu) L{layer_idx}: {e}"))?;
                    Some(t0)
                } else { None };
                if gemma_moe_use_affine {
                    let stack = self.layers[layer_idx].moe.gate_up_affine.as_ref().unwrap();
                    mlx_native::quantized_matmul_id_into(
                        s.encoder_mut(),
                        reg,
                        dev,
                        &pf_moe_norm_out,
                        &stack.weight,
                        &stack.scales,
                        &stack.biases,
                        &pf_expert_ids,
                        &pf_moe_gate_up,
                        &mlx_native::QuantizedMatmulIdParams {
                            m: seq_len as u32,
                            k: hs as u32,
                            n: (2 * moe_int) as u32,
                            group_size: stack.group_size,
                            bits: stack.bits,
                            n_expert_used: top_k as u32,
                            num_experts: num_experts as u32,
                        },
                    )
                    .map_err(|e| anyhow::anyhow!("batched gate_up_id (affine) L{layer_idx}: {e}"))?;
                    let _ = ggml_type_gu;
                } else {
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
                }
                if let Some(t0) = moe_gu_t0 {
                    bucket_finish!(s, exec, t0, &PROFILE_MOE_GU_NS, &PROFILE_MOE_GU_COUNT, 1, "moe_gu");
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
                    bucket_finish!(s, exec, t0, &PROFILE_MOE_POST_NS, &PROFILE_MOE_POST_COUNT, 1, "moe_swiglu");
                }

                // MoE down experts: same affine vs ggml routing as gate_up.
                let ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;
                let dn_w_buf: &mlx_native::MlxBuffer = if gemma_moe_use_affine {
                    &self.layers[layer_idx].moe.down_affine.as_ref().unwrap().weight
                } else {
                    self.layers[layer_idx].moe.stacked_down.as_ref().unwrap()
                };
                s.barrier_between(
                    &[&pf_moe_swiglu, &pf_expert_ids, dn_w_buf],
                    &[&pf_moe_down],
                );
                let moe_dn_t0 = if std::env::var("HF2Q_PROFILE_MOE").is_ok() || profile_buckets_on {
                    s.finish().map_err(|e| anyhow::anyhow!("MoE-profile pre-finish (dn) L{layer_idx}: {e}"))?;
                    let t0 = std::time::Instant::now();
                    s = exec.begin().map_err(|e| anyhow::anyhow!("MoE-profile begin (dn) L{layer_idx}: {e}"))?;
                    Some(t0)
                } else { None };
                if gemma_moe_use_affine {
                    let stack = self.layers[layer_idx].moe.down_affine.as_ref().unwrap();
                    mlx_native::quantized_matmul_id_into(
                        s.encoder_mut(),
                        reg,
                        dev,
                        &pf_moe_swiglu,
                        &stack.weight,
                        &stack.scales,
                        &stack.biases,
                        &pf_expert_ids,
                        &pf_moe_down,
                        &mlx_native::QuantizedMatmulIdParams {
                            m: (seq_len * top_k) as u32,
                            k: moe_int as u32,
                            n: hs as u32,
                            group_size: stack.group_size,
                            bits: stack.bits,
                            n_expert_used: 1,
                            num_experts: num_experts as u32,
                        },
                    )
                    .map_err(|e| anyhow::anyhow!("batched down_id (affine) L{layer_idx}: {e}"))?;
                    let _ = ggml_type_dn;
                } else {
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
                }
                if let Some(t0) = moe_dn_t0 {
                    bucket_finish!(s, exec, t0, &PROFILE_MOE_DN_NS, &PROFILE_MOE_DN_COUNT, 1, "moe_dn");
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
                    bucket_finish!(s, exec, t0, &PROFILE_B_MOE_WSUM_ADD_NS, &PROFILE_B_MOE_WSUM_ADD_COUNT, 1, "moe_wsum_add");
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
                    bucket_finish!(s, exec, t0, &PROFILE_B_END_LAYER_NS, &PROFILE_B_END_LAYER_COUNT, 1, "end_layer");
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
                // ADR-030 iter-65 (coherence-gate fix): when a DFlash
                // capture session is installed, the layer-loop hook at
                // line ~2216 CPU-reads `pf_hidden` via `as_slice()` to
                // populate the session.  With async `s.commit()` (the
                // fire-and-forget production default), the GPU has not
                // necessarily finished writing pf_hidden when the CPU
                // read happens — `as_slice` returns stale data from the
                // PRIOR layer (or initial zeros) → captured slab is for
                // the wrong layer → per_position_argmax produces wrong
                // values → coherence gate fails at pos 1.  Force
                // commit-and-wait at every layer when capture is active
                // so each layer's pf_hidden is GPU-flushed before the
                // hook reads.  Adds ~30 per-layer sync points (one per
                // layer per token-position) only when spec-decode is
                // running; production cmd_generate path is bit-identical.
                let sync_per_layer = batched_dump.is_some()
                    || std::env::var("HF2Q_PROFILE_LAYERS").is_ok()
                    || std::env::var("HF2Q_SYNC_PER_LAYER").is_ok()
                    || profile_buckets_on
                    || self.dflash_capture.is_some();
                if sync_per_layer {
                    if graph_opt_prefill {
                        // ADR-029 iter-39 H40 — sync-mode with fusion.  Mirrors
                        // commit_with_fusion's fuse+replay path, but commits-
                        // and-waits so HF2Q_PROFILE_LAYERS gets the same per-
                        // layer GPU-wall measurement it used pre-graph_opt.
                        let _f = s.finish_with_fusion(reg, dev.metal_device())
                            .map_err(|e| anyhow::anyhow!("batched mlp finish_with_fusion L{layer_idx}: {e}"))?;
                    } else {
                        s.finish()
                            .map_err(|e| anyhow::anyhow!("batched mlp finish L{layer_idx}: {e}"))?;
                    }
                    if std::env::var("HF2Q_PROFILE_LAYERS").is_ok() {
                        let kind = if is_sliding { "SW" } else { "GL" };
                        eprintln!("[LAYER_TIME] L{:02} {} {}us", layer_idx, kind,
                            layer_start.elapsed().as_micros());
                    }
                } else if graph_opt_prefill {
                    // ADR-029 iter-39 H40 — fusion + async commit.  The
                    // captured per-layer graph is fused (rms_norm→mul collapsed
                    // into single dispatch) then committed without waiting so
                    // the GPU pipelines with the next layer's CPU encode.
                    let (_committed, _fusions) = s.commit_with_fusion(reg, dev.metal_device())
                        .map_err(|e| anyhow::anyhow!("batched mlp commit_with_fusion L{layer_idx}: {e}"))?;
                    drop(_committed);
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
            //
            // ADR-028 iter-138 Path A Phase 2 GPU step 3/7 — append-mode
            // KV cursor advance. start_pos=0 (cold prefill, production
            // default): write_pos = seq_len, seq_len.min(capacity) —
            // identical to pre-iter-138. start_pos>0 (future verify):
            // write_pos = start_pos + seq_len, advancing past the existing
            // cache state. The K/V data at slots [0, start_pos) is
            // preserved (the kernel writes only at slots produced by
            // pf_positions, which iter-137 also offset by start_pos).
            let new_write_pos = start_pos + seq_len;
            self.kv_caches[layer_idx].write_pos = new_write_pos;
            self.kv_caches[layer_idx].seq_len = new_write_pos.min(self.kv_caches[layer_idx].capacity);

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

            // ADR-030 Phase 4 — DFlash hidden state capture hook.
            // When installed, captures pf_hidden (= this layer's output)
            // for layer indices matching dflash_capture.target_layer_ids.
            // Default-None preserves byte-identical legacy behavior.
            if self.dflash_capture.is_some() {
                let layer_idx_for_capture = layer_idx;
                let pf_data_opt: Option<Vec<f32>> = {
                    // Borrow pf_hidden read-only; copy out the slab so we
                    // don't hold a borrow into `self` when we then borrow
                    // self.dflash_capture mutably.
                    let pf_data: &[f32] = pf_hidden.as_slice()
                        .map_err(|e| anyhow::anyhow!("dflash capture pf_hidden L{layer_idx}: {e}"))?;
                    let needed = seq_len * hs;
                    // ADR-030 iter-82 — dump pf_hidden[t=0, d=0..8] at L0/Lfinal
                    // (env-gated) to bisect SDPA-correct vs hidden-state-wrong.
                    if std::env::var("HF2Q_DFLASH_XLEN_DEBUG").as_deref() == Ok("1")
                        && start_pos > 0
                    {
                        let n = 4.min(hs);
                        // Detect NaN/Inf and report magnitude summary.
                        let nan_count = pf_data[..needed].iter().filter(|x| x.is_nan()).count();
                        let inf_count = pf_data[..needed].iter().filter(|x| x.is_infinite()).count();
                        let max_abs = pf_data[..needed].iter()
                            .filter(|x| x.is_finite())
                            .map(|x| x.abs())
                            .fold(0.0f32, f32::max);
                        eprintln!(
                            "[XLEN_DEBUG capture L{} verify start_pos={} seq_len={} hs={}] \
                             pf_hidden[t=0,d=0..{}]={:?} nan={} inf={} max_abs={:.4e}",
                            layer_idx, start_pos, seq_len, hs, n,
                            &pf_data[..n], nan_count, inf_count, max_abs,
                        );
                    }
                    if pf_data.len() >= needed {
                        Some(pf_data[..needed].to_vec())
                    } else {
                        return Err(anyhow::anyhow!(
                            "dflash capture L{layer_idx}: pf_hidden len {} < seq_len*hs ({})",
                            pf_data.len(), needed
                        ));
                    }
                };
                if let Some(slab) = pf_data_opt {
                    let cap = self.dflash_capture.as_mut().unwrap();
                    if let Some(capture_idx) = cap.capture_index_for(layer_idx_for_capture) {
                        cap.write_layer_slab(capture_idx, &slab)
                            .map_err(|e| anyhow::anyhow!(
                                "dflash capture write_layer_slab L{layer_idx}: {e}"
                            ))?;
                    }
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
                bucket_finish!(s, exec, t0, &PROFILE_B_FINAL_NORM_NS, &PROFILE_B_FINAL_NORM_COUNT, 1, "final_norm");
            }

            // lm_head: whichever weight was loaded.  ADR-028 iter-345
            // adds Q6_K arm so HF2Q_LMHEAD_Q6K=1 (iter-188 lever, +2%
            // decode) coexists with HF2Q_BATCHED_PREFILL=1 (iter-344
            // default, 34× prefill).  Q6_K dispatched via the same
            // dispatch_qmatmul as Q8/Q4_0 (kernel_mul_mv_q6_K_f32_nr2
            // since iter-309 + iter-326 default).
            let t0_lm_head = if profile_buckets_on {
                Some(std::time::Instant::now())
            } else { None };
            if let Some(ref q6k) = self.lm_head_q6k {
                s.barrier_between(
                    &[&self.activations.norm_out, &q6k.buffer],
                    &[&self.activations.logits],
                );
                super::forward_mlx::dispatch_qmatmul(
                    &mut s, reg, dev,
                    &self.activations.norm_out,
                    q6k,
                    &mut self.activations.logits,
                    1,
                ).map_err(|e| anyhow::anyhow!("batched lm_head Q6_K: {e}"))?;
            } else if let Some(ref q8) = self.lm_head_q8 {
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
                anyhow::bail!("batched prefill requires GPU lm_head (Q6_K, F16, or Q8 weight)");
            }
            if let Some(t0) = t0_lm_head {
                bucket_finish!(s, exec, t0, &PROFILE_B_LM_HEAD_NS, &PROFILE_B_LM_HEAD_COUNT, 1, "lm_head");
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
                    bucket_finish!(s, exec, t0, &PROFILE_B_SOFTCAP_NS, &PROFILE_B_SOFTCAP_COUNT, 1, "softcap");
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
                bucket_finish!(s, exec, t0, &PROFILE_B_ARGMAX_NS, &PROFILE_B_ARGMAX_COUNT, 1, "argmax");
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
            // ADR-029 iter-81 H61: NO_FA per-dispatch buckets (only populated when use_no_fa is on).
            let (nofa_qk_ms, nofa_qk_n)         = fetch(&PROFILE_B_NOFA_QK_NS,     &PROFILE_B_NOFA_QK_COUNT);
            let (nofa_sms_ms, nofa_sms_n)       = fetch(&PROFILE_B_NOFA_SMS_NS,    &PROFILE_B_NOFA_SMS_COUNT);
            let (nofa_vtrans_ms, nofa_vtrans_n) = fetch(&PROFILE_B_NOFA_VTRANS_NS, &PROFILE_B_NOFA_VTRANS_COUNT);
            let (nofa_sv_ms, nofa_sv_n)         = fetch(&PROFILE_B_NOFA_SV_NS,     &PROFILE_B_NOFA_SV_COUNT);
            let (nofa_perm_ms, nofa_perm_n)     = fetch(&PROFILE_B_NOFA_PERM_NS,   &PROFILE_B_NOFA_PERM_COUNT);
            let nofa_total_ms = nofa_qk_ms + nofa_sms_ms + nofa_vtrans_ms + nofa_sv_ms + nofa_perm_ms;
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
                + head_ms
                + nofa_total_ms;
            let residual_ms = prefill_ms - sum_ms;

            let sep = "------------------------------------------------------------------";
            let row5 = |name: &str, ms: f64, n: u64| {
                eprintln!("{:<32} {:>8.3}  {:>5.1}%  {:>5}  {:>8.3}",
                    name, ms, pct(ms), n, per(ms, n));
            };
            let row3 = |name: &str, ms: f64, p: f64| {
                eprintln!("{:<32} {:>8.2}  {:>5.1}%", name, ms, p);
            };
            // [BUCKET_PROFILE] header is key=value pairs (pp, prefill, tok/s, path,
            // time_source). Parse by key, not position — time_source= was added when
            // HF2Q_PROFILE_GPU_TS=1 landed and distinguishes CPU vs GPU wall-clock.
            eprintln!("[BUCKET_PROFILE] pp={} prefill={:.2} ms (tok/s={:.1}) path={} time_source={}",
                seq_len, prefill_ms, seq_len as f64 / (prefill_ms / 1000.0),
                if use_no_fa { "tensor-mm (non-FA)" } else { "flash-attn" },
                if profile_gpu_ts_on { "GPU wall-clock (MTLCommandBuffer.GPUStartTime/GPUEndTime)" }
                else { "CPU wall-clock (includes commit+wait overhead)" });
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
            // ADR-029 iter-81 H61: NO_FA per-dispatch buckets (zero when use_no_fa=0).
            row5("NOFA_QK (Q@K^T)",           nofa_qk_ms,      nofa_qk_n);
            row5("NOFA_SMS (scale_mask_sm)",  nofa_sms_ms,     nofa_sms_n);
            row5("NOFA_VTRANS (V transpose)", nofa_vtrans_ms,  nofa_vtrans_n);
            row5("NOFA_SV (scores@V)",        nofa_sv_ms,      nofa_sv_n);
            row5("NOFA_PERM (perm021)",       nofa_perm_ms,    nofa_perm_n);
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

        // Store dense KV buffers so forward_decode can use them.
        //
        // ADR-017 Phase E.a iter-2.5: wrap each per-layer
        // `DenseKvBuffers` in an `Arc` at the prefill→decode handoff
        // (mirrors `forward_prefill.rs:1502`). Builder above keeps
        // `Vec<DenseKvBuffers>` so the in-flight kernel writes mutate
        // the buffers via `&mut`; Arc wrap fires once at end-of-prefill.
        self.dense_kvs = Some(
            dense_kvs_vec
                .into_iter()
                .map(std::sync::Arc::new)
                .collect(),
        );
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

    /// ADR-028 iter-134 Path A Phase 2 GPU — forward_decode_verify_batched scaffold.
    ///
    /// Speculative-decode verify: forward `tokens` through the model in a
    /// single batched pass (vs Shape S iter-123's K serial forward_decode
    /// calls). Returns one argmax per input token.
    ///
    /// **Shape B contract** (vs Shape S serial):
    /// - Single GraphSession encloses all K+1 token forwards (1
    ///   commit_and_wait, not K+1).
    /// - Append-mode KV: positions written are `[start_seq_pos,
    ///   start_seq_pos + tokens.len())`.
    /// - Per-position argmax: emits argmax at each token position, not
    ///   just last.
    ///
    /// At greedy temperature, output for a fixed prefix is byte-identical
    /// to calling `forward_decode` K+1 times serially (same numerics,
    /// just batched dispatch). This is the speedup Path A delivers.
    ///
    /// # Implementation status
    ///
    /// **iter-134**: scaffold only — returns
    /// `Err(NotYetImplemented)`. Subsequent iter-135+ will fill in:
    /// 1. Append-mode initialization (replace `kv_caches[i].write_pos =
    ///    seq_len` at line 1823 with `+= tokens.len()`).
    /// 2. Per-position LM head + argmax loop (replace last-row-only
    ///    emission at line 1966+ with a loop over each output position).
    /// 3. Output buffer for K+1 argmaxes.
    /// 4. Integration test: byte-identity vs K+1 serial forward_decode.
    ///
    /// # iter-135 scoping addendum (precise line-level diff for iter-136 to execute)
    ///
    /// The append-mode parameterization is a 4-site edit on this file
    /// (call counts and exact line numbers as of HEAD `972a2b4`):
    ///
    /// | Site | Line | Current                                                | After (param=start_pos)                       |
    /// |------|------|--------------------------------------------------------|------------------------------------------------|
    /// |  A   | 469  | `for (i, slot) in p[..seq_len].iter_mut().enumerate()` | (unchanged loop, body changes)                |
    /// |  A   | 470  | `*slot = i as u32`                                     | `*slot = (start_pos + i) as u32`              |
    /// |  B   | 1823 | `write_pos = seq_len`                                  | `write_pos = start_pos + seq_len`             |
    /// |  C   | 1824 | `seq_len = seq_len.min(capacity)`                      | `seq_len = (start_pos + seq_len).min(capacity)` |
    /// |  D   | (LM head) | `dispatch_argmax_f32` last-row-only             | per-position-loop (impl scope iter-137)       |
    ///
    /// **Refactor strategy**: factor out a `forward_batched_inner(
    /// tokens, start_pos: usize, capture_argmax: AllOrLast) -> Vec<u32>`
    /// helper. forward_prefill_batched becomes a thin wrapper calling
    /// `forward_batched_inner(tokens, 0, AllOrLast::Last)` and unwrapping
    /// the single argmax. forward_decode_verify_batched calls
    /// `forward_batched_inner(tokens, current_pos, AllOrLast::All)` and
    /// returns the full Vec.
    ///
    /// This refactor is multi-iter scope (requires comprehensive
    /// regression testing of forward_prefill_batched at start_pos=0,
    /// which is THE production decode/prefill path on qwen35). Per
    /// operator's "do it right, regression-gated" mantra, the refactor
    /// must be incremental:
    /// - iter-136: factor out the inner helper (NSG-equivalent
    ///   byte-identity gate + 8/8 sourdough byte-identity validation).
    /// - iter-137: thread `start_pos` parameter (default 0 → identical).
    /// - iter-138: thread `capture_argmax` parameter (Last default).
    /// - iter-139: implement Shape B body in
    ///   forward_decode_verify_batched (start_pos = current write_pos,
    ///   capture_argmax = All).
    /// - iter-140: spec-decode-loop integration test (proposer →
    ///   verify_batched → accept_prefix_argmax → rollback_kv).
    /// - iter-141+: production wire-up + acceptance-rate measurement.
    pub fn forward_decode_verify_batched(
        &mut self,
        tokens: &[u32],
        start_seq_pos: usize,
        gpu: &mut GpuContext,
    ) -> Result<Vec<u32>> {
        // ADR-030 Phase 4 (iter-47) — REAL BATCHED BODY.
        //
        // Previously a serial delegation (iter-139 TEMPORARY); now
        // implements the actual batched verify using:
        //  1. DFlashCaptureSession installed on self
        //  2. forward_prefill_batched with the layer-loop hook
        //     capturing the FINAL layer's pf_hidden
        //  3. per_position_argmax_from_hidden running final_norm +
        //     lm_head + softcap + argmax for each of the K+1 positions
        //
        // Byte-identity invariant: argmaxes[seq_len-1] MUST equal
        // forward_prefill_batched's returned first_token (the existing
        // last-row argmax). Both compute identical dispatches on the
        // identical hidden buffer. Debug-asserted at the bottom.
        //
        // Performance: 1 forward_prefill_batched + K+1 per-position
        // argmax sessions. The per-position argmax cost is ~5×K small
        // dispatches (vs the K× forward_decode the serial path did).
        // Total verify cost ≈ 1 batched forward + small constant.
        let seq_len = tokens.len() as u32;
        if seq_len == 0 {
            anyhow::bail!("forward_decode_verify_batched: empty tokens");
        }
        let hs = self.hidden_size;
        let num_layers = self.layers.len();
        let final_layer_idx = num_layers - 1;

        // Install a capture session targeting the FINAL layer only.
        // The Phase 4 orchestrator's drafter-input target_layer_ids
        // (= [1, 6, 11, 17, 22, 27] for gemma-4) is a separate concern
        // captured in a different session — that orchestrator wraps
        // this method and installs its own session in its own call.
        let session = crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession::new(
            vec![final_layer_idx],
            seq_len as usize,
            hs,
            true, // with per_position_argmaxes (not used here, but allocated for API consistency)
        );
        self.install_dflash_capture(session);

        // Run forward — the layer-loop hook captures pf_hidden into the
        // installed session at layer_idx = final_layer_idx.
        let first_token = self
            .forward_prefill_batched(tokens, 0, start_seq_pos, gpu)
            .map_err(|e| anyhow::anyhow!("forward_decode_verify_batched: forward: {e}"))?;

        // Take back the populated session.
        let session = self
            .take_dflash_capture()
            .ok_or_else(|| anyhow::anyhow!("forward_decode_verify_batched: session vanished"))?;

        // Extract final layer's [seq_len, hs] slab. With only one
        // target_layer_id, hidden_output is exactly [seq_len, hs] F32.
        let final_hidden = session.hidden_output;
        let expected_len = (seq_len as usize) * hs;
        if final_hidden.len() != expected_len {
            anyhow::bail!(
                "forward_decode_verify_batched: hidden_output len {} != seq_len({}) * hs({}) = {}",
                final_hidden.len(), seq_len, hs, expected_len
            );
        }

        // Compute per-position argmaxes from the captured hidden.
        let argmaxes = self.per_position_argmax_from_hidden(&final_hidden, seq_len, gpu)?;

        // Byte-identity guarantee: last-position argmax must match
        // forward_prefill_batched's first_token (same dispatchers,
        // same hidden row).
        debug_assert_eq!(
            argmaxes[(seq_len - 1) as usize],
            first_token,
            "forward_decode_verify_batched byte-identity: argmaxes[last] != first_token"
        );
        // In release, log a warning rather than panic if invariant
        // somehow violates — would indicate dispatch nondeterminism.
        if argmaxes[(seq_len - 1) as usize] != first_token {
            eprintln!(
                "[ADR-030 Phase 4 WARNING] forward_decode_verify_batched byte-identity \
                 violated: argmaxes[{}] = {} but first_token = {}. Coherence at risk.",
                seq_len - 1,
                argmaxes[(seq_len - 1) as usize],
                first_token
            );
        }

        Ok(argmaxes)
    }
}
