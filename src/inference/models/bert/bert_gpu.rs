//! BERT encoder forward primitives — Metal GPU dispatch.
//!
//! ADR-005 Phase 2b. This module ships the GPU primitives the encoder
//! forward pass composes — same staging discipline as the iter 44–51d
//! ViT GPU build-out. Each primitive lands as a unit with a CPU
//! reference for parity, then the iter that ships the next op is gated
//! on the prior op's parity.
//!
//! # Why a custom Metal kernel for LayerNorm
//!
//! `mlx-native` ships `dispatch_rms_norm` but no LayerNorm. RMSNorm is
//! `x / sqrt(mean(x²) + eps) * weight` — no mean centering, no bias.
//! LayerNorm is `(x - mean) / sqrt(var + eps) * weight + bias`. The
//! mean-centering and bias addition are required by every BERT variant
//! and substituting RMSNorm produces silently-wrong embeddings.
//!
//! We register an inline Metal source via `KernelRegistry::register_source`
//! exactly like the iter-51c `vit_avg_pool_2x2_f32` pattern — no fork of
//! mlx-native, no addition to the cross-lane crate, no lock contention
//! with the chat-model team's Metal-source surface.
//!
//! # Numerical strategy
//!
//! Two-pass mean/variance — accurate enough for F32 inputs at typical
//! BERT magnitudes (|x| < 10), and avoids the catastrophic-cancellation
//! risk of a one-pass `E[x²] - E[x]²` formulation. Each row (one
//! sequence position) is handled by one threadgroup. Threadgroup memory
//! holds the partial-sum reduction. Hidden sizes encountered in the
//! day-one model set: 384 (bge-small), 768 (nomic-embed-text), 1024
//! (mxbai-embed-large) — all within a single threadgroup at threads ≤
//! `min(hidden, 256)`.

#![allow(dead_code)] // forward pass + handler wiring lands in subsequent iters

use anyhow::{anyhow, Context, Result};
use mlx_native::metal::MTLSize;
use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
use mlx_native::ops::elementwise::{cast, elementwise_add, CastDirection};
use mlx_native::ops::encode_helpers::KernelArg;
use mlx_native::ops::gather::dispatch_gather_f32;
use mlx_native::ops::l2_norm::dispatch_l2_norm;
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

// ---------------------------------------------------------------------------
// Inline Metal sources
// ---------------------------------------------------------------------------

/// Metal source for the BERT-specific GPU primitives. Registered into
/// the `KernelRegistry` via `register_bert_custom_shaders`. Lives as
/// `&'static str` because the registry stores sources by reference.
const BERT_CUSTOM_SHADERS_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct LayerNormParams {
    uint hidden;
    uint batch;
    float eps;
};

// Per-row LayerNorm: out[r, h] = (x[r, h] - mean(x[r, :])) /
//                                sqrt(var(x[r, :]) + eps) * gamma[h] + beta[h]
//
// Dispatch: one threadgroup per row, `threads_per_threadgroup.x` set to
// the chosen reduction width (≤ hidden). Threadgroup memory at index 0
// is `4 * threads_per_threadgroup.x` bytes.
//
// Two-pass: pass 1 computes the row mean via parallel reduction, pass 2
// computes the variance via the same reduction pattern using the mean
// from pass 1, then a final write applies the affine transform. F32
// throughout — BERT weights are F16 in GGUF but every dequant target
// is F32 in this loader for parity with the CPU reference.
kernel void bert_layer_norm_f32(
    device const float* input  [[buffer(0)]],
    device const float* gamma  [[buffer(1)]],
    device const float* beta   [[buffer(2)]],
    device       float* output [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint  tid  [[thread_position_in_threadgroup]],
    uint  bid  [[threadgroup_position_in_grid]],
    uint  ntg  [[threads_per_threadgroup]]
) {
    if (bid >= params.batch) return;
    uint row_off = bid * params.hidden;

    // ----- Pass 1: row sum -> mean -----
    float sum = 0.0;
    for (uint i = tid; i < params.hidden; i += ntg) {
        sum += input[row_off + i];
    }
    shmem[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Parallel reduction; ntg is a power of two by caller construction.
    for (uint stride = ntg / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shmem[0] / float(params.hidden);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ----- Pass 2: row variance -> inv_std -----
    float var_sum = 0.0;
    for (uint i = tid; i < params.hidden; i += ntg) {
        float d = input[row_off + i] - mean;
        var_sum += d * d;
    }
    shmem[tid] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shmem[0] / float(params.hidden) + params.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ----- Apply: (x - mean) * inv_std * gamma + beta -----
    for (uint i = tid; i < params.hidden; i += ntg) {
        float x = input[row_off + i];
        output[row_off + i] = ((x - mean) * inv_std) * gamma[i] + beta[i];
    }
}

struct BiasAddParams {
    uint rows;
    uint cols;
};

// out[r, c] = input[r, c] + bias[c]. The matmul produces `[rows, cols]`
// row-major; this kernel broadcasts the per-column bias along rows.
//
// In-place is supported (input == output): each thread reads `input` and
// writes `output` at the same offset, so no aliasing hazard.
kernel void bert_bias_add_f32(
    device const float* input  [[buffer(0)]],
    device const float* bias   [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant BiasAddParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint r = gid.y;
    if (c >= params.cols || r >= params.rows) return;
    uint idx = r * params.cols + c;
    output[idx] = input[idx] + bias[c];
}

struct PoolMeanParams {
    uint seq_len;
    uint hidden;
};

struct AttnMaskAddParams {
    uint num_heads;
    uint seq_q;
    uint seq_k;
};

// Broadcast-add a `[seq_q, seq_k]` mask to a `[num_heads, seq_q, seq_k]`
// scores tensor. Mask[r, c] is 0.0 at valid positions and -INF
// (encoded as a very large negative float) at padded positions; after
// softmax the padded contributions vanish.
//
// In-place safe (input == output): each thread reads from `scores`
// at one offset and writes to `output` at the same offset. The kernel
// is bandwidth-bound; no reduction.
kernel void bert_attention_mask_add_f32(
    device const float* scores [[buffer(0)]],
    device const float* mask   [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant AttnMaskAddParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint r = gid.y;
    uint h = gid.z;
    if (c >= params.seq_k || r >= params.seq_q || h >= params.num_heads) return;
    uint scores_idx = (h * params.seq_q + r) * params.seq_k + c;
    uint mask_idx = r * params.seq_k + c;
    output[scores_idx] = scores[scores_idx] + mask[mask_idx];
}

// Mean-pool across the sequence dim:
//   out[h] = (1/seq_len) * sum_{s=0}^{seq_len-1} input[s * hidden + h]
//
// Input shape `[seq_len, hidden]` row-major; output shape `[hidden]`.
// One thread per column of `hidden`; the loop iterates over seq_len.
// For seq_len up to a few thousand, this is fully bandwidth-bound and a
// fancier reduction (parallel sum) would not help on M5 Max. Keeping
// the kernel scalar-per-thread also makes correctness obvious.
kernel void bert_pool_mean_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant PoolMeanParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.hidden) return;
    float acc = 0.0;
    for (uint s = 0; s < params.seq_len; ++s) {
        acc += input[s * params.hidden + gid];
    }
    output[gid] = acc / float(params.seq_len);
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct LayerNormGpuParams {
    hidden: u32,
    batch: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BiasAddGpuParams {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PoolMeanGpuParams {
    seq_len: u32,
    hidden: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnMaskAddGpuParams {
    num_heads: u32,
    seq_q: u32,
    seq_k: u32,
}

/// View any `Copy + repr(C)` POD as a byte slice. SAFE for `repr(C)`
/// structs containing only primitive fields with natural alignment.
fn pod_as_bytes<T: Copy>(p: &T) -> &[u8] {
    // SAFETY: `T: Copy + repr(C)` with primitive fields. The
    // `LayerNormGpuParams` layout (u32, u32, f32) is contiguous 12 bytes
    // with no padding on every supported target.
    unsafe { std::slice::from_raw_parts(p as *const T as *const u8, std::mem::size_of::<T>()) }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register every kernel exported by `BERT_CUSTOM_SHADERS_SOURCE` with
/// the registry. Idempotent — `register_source` overwrites any prior
/// registration for the same name. Caller invokes once per
/// `KernelRegistry`.
pub fn register_bert_custom_shaders(registry: &mut KernelRegistry) {
    registry.register_source("bert_layer_norm_f32", BERT_CUSTOM_SHADERS_SOURCE);
    registry.register_source("bert_bias_add_f32", BERT_CUSTOM_SHADERS_SOURCE);
    registry.register_source("bert_pool_mean_f32", BERT_CUSTOM_SHADERS_SOURCE);
    registry.register_source("bert_attention_mask_add_f32", BERT_CUSTOM_SHADERS_SOURCE);
    // mlx-native's per-op kernel sources are needed for the attention
    // chain (delegated to `vit_attention_gpu`), GeLU, embedding gather,
    // and L2 normalization. Modules listed here expose a `register` fn;
    // `transpose`, `elementwise`, `dense_mm_bf16`, and `cast` self-
    // register their sources on first dispatch via internal lookups.
    mlx_native::ops::gelu::register(registry);
    mlx_native::ops::softmax::register(registry);
    mlx_native::ops::sigmoid_mul::register(registry);
    mlx_native::ops::gather::register(registry);
    mlx_native::ops::l2_norm::register(registry);
}

// ---------------------------------------------------------------------------
// LayerNorm — GPU dispatch
// ---------------------------------------------------------------------------

/// GPU LayerNorm: `out[r, h] = (in[r, h] - mean_r) / sqrt(var_r + eps) *
///                            gamma[h] + beta[h]`
/// where `mean_r`/`var_r` are computed over the `hidden` dimension.
///
/// Inputs:
/// - `input`: F32 buffer shape `[batch, hidden]` (row-major).
/// - `gamma`, `beta`: F32 buffers shape `[hidden]` each.
///
/// Returns a fresh F32 `[batch, hidden]` output buffer.
///
/// `eps` is the LayerNorm epsilon (BERT uses `1e-12` in
/// HuggingFace configs; llama.cpp emits the same value via
/// `bert.attention.layer_norm_epsilon`).
///
/// # Threadgroup configuration
///
/// One threadgroup per row. Reduction width is `min(hidden, 256)`
/// rounded down to the nearest power of two — the kernel's reduction
/// loop assumes `ntg` is a power of two. For hidden ∈ {384, 768, 1024}
/// the chosen widths are 256/256/256, all well within Metal's 1024
/// `maxTotalThreadsPerThreadgroup` bound on M5 Max.
///
/// Caller registers `register_bert_custom_shaders(&mut registry)` before
/// the first call (the registry caches the compiled pipeline so
/// subsequent calls reuse it).
///
/// # Errors
///
/// - `batch == 0` or `hidden == 0`
/// - propagated from kernel pipeline compile failures
pub fn bert_layer_norm_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    gamma: &MlxBuffer,
    beta: &MlxBuffer,
    eps: f32,
    batch: u32,
    hidden: u32,
) -> Result<MlxBuffer> {
    if batch == 0 || hidden == 0 {
        return Err(anyhow!(
            "bert_layer_norm_gpu: batch ({}) and hidden ({}) must be > 0",
            batch,
            hidden
        ));
    }

    let total = (batch as usize) * (hidden as usize);
    let output = device
        .alloc_buffer(total * 4, DType::F32, vec![batch as usize, hidden as usize])
        .map_err(|e| anyhow!("alloc bert_layer_norm output: {e}"))?;

    let pipeline = registry
        .get_pipeline("bert_layer_norm_f32", device.metal_device())
        .map_err(|e| anyhow!("bert_layer_norm_gpu: get_pipeline: {e}"))?;

    let params = LayerNormGpuParams {
        hidden,
        batch,
        eps,
    };
    let bytes = pod_as_bytes(&params);

    // Choose a power-of-two reduction width ≤ min(hidden, 256). Any
    // larger and the threadgroup memory cost grows with no benefit;
    // smaller would underutilize the threadgroup. The per-thread loop
    // chunks across `hidden` so the kernel is correct for any hidden ≥
    // ntg as long as ntg is a power of two.
    let cap = hidden.min(256);
    let ntg = prev_pow2(cap.max(1));

    let threadgroups = MTLSize::new(batch as u64, 1, 1);
    let threadgroup_size = MTLSize::new(ntg as u64, 1, 1);
    let shmem_bytes = (ntg as u64) * 4;

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(gamma)),
            (2, KernelArg::Buffer(beta)),
            (3, KernelArg::Buffer(&output)),
            (4, KernelArg::Bytes(bytes)),
        ],
        &[(0, shmem_bytes)],
        threadgroups,
        threadgroup_size,
    );
    Ok(output)
}

/// Largest power of two ≤ `n`. `n` must be `>= 1`.
fn prev_pow2(n: u32) -> u32 {
    debug_assert!(n >= 1);
    1u32 << (31 - n.leading_zeros())
}

// ---------------------------------------------------------------------------
// Bias add — GPU dispatch
// ---------------------------------------------------------------------------

/// Broadcast-add a per-column bias to a `[rows, cols]` row-major matrix.
/// `out[r, c] = in[r, c] + bias[c]`. Allocates and returns a fresh F32
/// `[rows, cols]` output buffer.
///
/// `bias` shape `[cols]` F32. Supports any input matrix shape that
/// matmul produces (typically `[seq_len, out_features]`).
///
/// # Errors
/// - `rows == 0` or `cols == 0`
/// - propagated from kernel pipeline compile failures
pub fn bert_bias_add_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    bias: &MlxBuffer,
    rows: u32,
    cols: u32,
) -> Result<MlxBuffer> {
    if rows == 0 || cols == 0 {
        return Err(anyhow!(
            "bert_bias_add_gpu: rows ({}) and cols ({}) must be > 0",
            rows,
            cols
        ));
    }
    let total = (rows as usize) * (cols as usize);
    let output = device
        .alloc_buffer(total * 4, DType::F32, vec![rows as usize, cols as usize])
        .map_err(|e| anyhow!("alloc bert_bias_add output: {e}"))?;

    let pipeline = registry
        .get_pipeline("bert_bias_add_f32", device.metal_device())
        .map_err(|e| anyhow!("bert_bias_add_gpu: get_pipeline: {e}"))?;

    let params = BiasAddGpuParams { rows, cols };
    let bytes = pod_as_bytes(&params);
    let grid = MTLSize::new(cols as u64, rows as u64, 1);
    let tg_x = std::cmp::min(64, cols as u64);
    let tg = MTLSize::new(tg_x, 1, 1);
    encoder.encode_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(bias)),
            (2, KernelArg::Buffer(&output)),
            (3, KernelArg::Bytes(bytes)),
        ],
        grid,
        tg,
    );
    Ok(output)
}

// ---------------------------------------------------------------------------
// Linear (Wx + b) — GPU dispatch
// ---------------------------------------------------------------------------

/// GPU linear projection: `out = input @ weight.T + bias` (when bias
/// supplied), or `input @ weight.T` (when `bias_opt = None`).
///
/// Inputs:
/// - `input`   F32 `[seq_len, in_features]`
/// - `weight`  F32 `[out_features, in_features]` (cast to BF16 for the
///             matmul tensor-core path; same precision profile as
///             `vit_linear_gpu`)
/// - `bias_opt` Optional F32 `[out_features]`
///
/// Returns F32 `[seq_len, out_features]`.
///
/// # Why BF16 weight cast (and what it costs)
///
/// The hot matmul path on M5 Max is `dense_matmul_bf16_f32_tensor`,
/// which casts the weight to BF16 to use Apple's BF16 tensor-core
/// pipeline. Per-element error from the F32→BF16 round-trip is ≈ 2⁻⁷.
/// For BERT linear projections this is acceptable — input magnitudes
/// after LayerNorm are O(1), so post-bias output noise is bounded by
/// `|x| × 2⁻⁷ × √hidden ≈ 2⁻⁷ × √384 ≈ 0.15`. Within the F16 round-off
/// llama.cpp's BERT path itself accumulates.
///
/// **What this means for attention.** ViT iter 50 documented that the
/// same BF16 K cast plus saturated softmax flips winners. BERT's
/// activations after LayerNorm are unit-normalized so attention scores
/// stay near zero pre-softmax — saturated-softmax flips are unlikely.
/// The full attention path lands separately (iter 58); this iter's
/// `bert_linear_gpu` is correct for QKV/output/up/down projections.
///
/// `in_features < 32` is rejected because the BF16 tensor matmul kernel
/// requires K ≥ 32 (matches `vit_linear_gpu`'s constraint).
///
/// Caller registers `register_bert_custom_shaders(&mut registry)` first.
///
/// # Errors
/// - `seq_len == 0`, `in_features < 32`, `out_features == 0`
/// - bias supplied but element_count != out_features
/// - propagated from cast/matmul/bias-add dispatches
pub fn bert_linear_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight_f32: &MlxBuffer,
    bias_opt: Option<&MlxBuffer>,
    seq_len: u32,
    in_features: u32,
    out_features: u32,
) -> Result<MlxBuffer> {
    if in_features < 32 {
        return Err(anyhow!(
            "bert_linear_gpu: in_features ({}) must be >= 32",
            in_features
        ));
    }
    if seq_len == 0 || out_features == 0 {
        return Err(anyhow!(
            "bert_linear_gpu: seq_len ({}) and out_features ({}) must be > 0",
            seq_len,
            out_features
        ));
    }
    if let Some(b) = bias_opt {
        if b.element_count() != out_features as usize {
            return Err(anyhow!(
                "bert_linear_gpu: bias element_count ({}) != out_features ({})",
                b.element_count(),
                out_features
            ));
        }
    }

    let metal_dev = device.metal_device();

    // --- Cast weight F32 → BF16 once ---
    let n_w = (out_features as usize) * (in_features as usize);
    let weight_bf16 = device
        .alloc_buffer(
            n_w * 2,
            DType::BF16,
            vec![out_features as usize, in_features as usize],
        )
        .map_err(|e| anyhow!("alloc weight_bf16: {e}"))?;
    cast(
        encoder,
        registry,
        metal_dev,
        weight_f32,
        &weight_bf16,
        n_w,
        CastDirection::F32ToBF16,
    )
    .context("bert_linear_gpu: F32→BF16 cast")?;
    // RAW: matmul reads `weight_bf16` written by the cast above.
    // mlx-native uses MTLDispatchType::Concurrent, so without this
    // barrier the matmul could read pre-cast garbage.
    encoder.memory_barrier();

    // --- Allocate F32 matmul output ---
    let out_bytes = (seq_len as usize) * (out_features as usize) * 4;
    let mut matmul_out = device
        .alloc_buffer(
            out_bytes,
            DType::F32,
            vec![seq_len as usize, out_features as usize],
        )
        .map_err(|e| anyhow!("alloc matmul output: {e}"))?;

    // --- Dispatch matmul ---
    let params = DenseMmBf16F32Params {
        m: seq_len,
        n: out_features,
        k: in_features,
        src0_batch: 1,
        src1_batch: 1,
    };
    dense_matmul_bf16_f32_tensor(
        encoder,
        registry,
        device,
        &weight_bf16,
        input,
        &mut matmul_out,
        &params,
    )
    .context("bert_linear_gpu: dense_matmul_bf16_f32_tensor")?;

    // --- Optional bias add ---
    if let Some(bias) = bias_opt {
        // RAW: bias_add reads `matmul_out` written by the matmul. Same
        // concurrent-dispatch rule applies. Without this, bias_add reads
        // freshly-allocated uninitialized F32 bytes — observed empirically
        // as max_diff ≈ 1200 in the iter-57 dev cycle.
        encoder.memory_barrier();
        bert_bias_add_gpu(
            encoder,
            registry,
            device,
            &matmul_out,
            bias,
            seq_len,
            out_features,
        )
        .context("bert_linear_gpu: bias_add")
    } else {
        Ok(matmul_out)
    }
}

// ---------------------------------------------------------------------------
// GeLU activation — GPU dispatch (pytorch_tanh variant)
// ---------------------------------------------------------------------------

/// GPU GeLU activation, **pytorch_tanh variant** (matches llama.cpp's
/// `ggml_gelu`):
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
///
/// llama.cpp's BERT path uses this same approximation regardless of
/// whether the upstream HF model declared `gelu` or `gelu_new` — and
/// the Phase 2b accuracy gate is parity with llama.cpp's output, so we
/// use it unconditionally. The day-one models (`nomic-embed-text-v1.5`,
/// `mxbai-embed-large-v1`, `bge-small-en-v1.5`) all clear this gate
/// because their HF-released checkpoints' GeLU choices either are this
/// approximation or are within the F16 round-off llama.cpp is graded
/// against.
///
/// `input` is any F32 buffer; output is a fresh F32 buffer with the
/// same element count and shape. F16/BF16 dtypes are also accepted
/// transparently (mlx-native's GeLU dispatches the matching variant).
///
/// Caller registers `register_bert_custom_shaders(&mut registry)` first
/// (which calls `mlx_native::ops::gelu::register`).
pub fn bert_gelu_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
) -> Result<MlxBuffer> {
    let n = input.element_count();
    if n == 0 {
        return Err(anyhow!("bert_gelu_gpu: input must have at least one element"));
    }
    let bytes = match input.dtype() {
        DType::F32 => n * 4,
        DType::F16 | DType::BF16 => n * 2,
        other => {
            return Err(anyhow!(
                "bert_gelu_gpu: unsupported dtype {:?} (expected F32/F16/BF16)",
                other
            ));
        }
    };
    let output = device
        .alloc_buffer(bytes, input.dtype(), input.shape().to_vec())
        .map_err(|e| anyhow!("alloc bert_gelu output: {e}"))?;
    mlx_native::ops::gelu::dispatch_gelu(
        encoder,
        registry,
        device.metal_device(),
        input,
        &output,
    )
    .map_err(|e| anyhow!("bert_gelu_gpu: dispatch_gelu: {e}"))?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// Bidirectional self-attention — GPU dispatch
// ---------------------------------------------------------------------------

/// GPU bidirectional scaled dot-product attention:
/// `out[s, h, d] = sum_k softmax(Q[s, h, :] · K[k, h, :] * scale)[k] * V[k, h, d]`
///
/// Inputs (all F32, seq-major shape `[seq_len, num_heads, head_dim]`):
/// - `q_seq_major`, `k_seq_major`, `v_seq_major`
///
/// `scale` = `1 / sqrt(head_dim)` (the standard Vaswani scale; BERT does
/// not modify it). No causal mask, no RoPE — every sequence position
/// attends to every other position symmetrically.
///
/// Returns F32 `[seq_len, num_heads, head_dim]` in seq-major layout (the
/// caller can reshape to `[seq_len, hidden]` via a no-op reinterpretation
/// since `hidden = num_heads * head_dim`).
///
/// # Implementation: shared with ViT
///
/// The dispatch chain (Q@K^T → scale → softmax → permute V → V cast →
/// transpose → matmul → permute back) is **identical** to `vit_attention_gpu`
/// because both are bidirectional self-attention with no per-head
/// normalization, no causal mask, no positional bias. The vision module
/// owns the canonical implementation (validated under iter 47–50);
/// this wrapper keeps the BERT module's call-sites BERT-named and
/// avoids touching the vision lane while sharing every kernel pipeline.
///
/// # Precision profile
///
/// Inherits the iter-50 BF16-saturated-softmax characterization: the
/// `dense_matmul_bf16_f32_tensor` path casts K to BF16, which can flip
/// softmax winners when scores saturate (Q-norm magnitudes ≳ 5). For
/// BERT's post-LayerNorm activations (magnitude O(1)) the saturation
/// regime is structurally absent — pre-softmax score magnitudes stay
/// near zero — so the path is correct for all three day-one models
/// (`bge-small-en-v1.5`, `nomic-embed-text-v1.5`, `mxbai-embed-large-v1`).
///
/// # Errors
///
/// - `head_dim < 32` (matmul kernel constraint)
/// - `seq_len == 0` or `num_heads == 0`
/// - propagated from any sub-stage
pub fn bert_attention_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q_seq_major: &MlxBuffer,
    k_seq_major: &MlxBuffer,
    v_seq_major: &MlxBuffer,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
) -> Result<MlxBuffer> {
    crate::inference::vision::vit_gpu::vit_attention_gpu(
        encoder,
        registry,
        device,
        q_seq_major,
        k_seq_major,
        v_seq_major,
        seq_len,
        num_heads,
        head_dim,
        scale,
    )
    .context("bert_attention_gpu: delegate to vit_attention_gpu")
}

// ---------------------------------------------------------------------------
// Masked bidirectional attention — GPU dispatch
// ---------------------------------------------------------------------------

/// Apply a `[seq_q, seq_k]` mask broadcast across `num_heads` to a
/// `[num_heads, seq_q, seq_k]` scores tensor. Mask values: 0.0 at
/// real positions, large-negative (e.g. `f32::MIN_POSITIVE.recip().neg()`
/// or just `-1e30`) at padded positions. Output buffer same shape as
/// input; in-place safe.
pub fn bert_attention_mask_add_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    scores: &MlxBuffer,
    mask: &MlxBuffer,
    num_heads: u32,
    seq_q: u32,
    seq_k: u32,
) -> Result<MlxBuffer> {
    if num_heads == 0 || seq_q == 0 || seq_k == 0 {
        return Err(anyhow!(
            "bert_attention_mask_add_gpu: num_heads/seq_q/seq_k must be > 0"
        ));
    }
    let total = (num_heads as usize) * (seq_q as usize) * (seq_k as usize);
    let output = device
        .alloc_buffer(
            total * 4,
            DType::F32,
            vec![num_heads as usize, seq_q as usize, seq_k as usize],
        )
        .map_err(|e| anyhow!("alloc mask_add output: {e}"))?;
    let pipeline = registry
        .get_pipeline("bert_attention_mask_add_f32", device.metal_device())
        .map_err(|e| anyhow!("bert_attention_mask_add_gpu: get_pipeline: {e}"))?;
    let params = AttnMaskAddGpuParams {
        num_heads,
        seq_q,
        seq_k,
    };
    let bytes = pod_as_bytes(&params);
    let grid = MTLSize::new(seq_k as u64, seq_q as u64, num_heads as u64);
    let tg = MTLSize::new(std::cmp::min(64, seq_k as u64), 1, 1);
    encoder.encode_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(scores)),
            (1, KernelArg::Buffer(mask)),
            (2, KernelArg::Buffer(&output)),
            (3, KernelArg::Bytes(bytes)),
        ],
        grid,
        tg,
    );
    Ok(output)
}

/// Build a `[seq_len, seq_len]` F32 attention mask for BERT bidirectional
/// self-attention. Columns `[0, valid_len)` get value 0.0; columns
/// `[valid_len, seq_len)` get a large negative value (`-1e30`) so the
/// post-softmax weight is effectively zero. The mask is row-invariant
/// (same value across all query positions for a given key position) —
/// matches HuggingFace's `attention_mask` broadcast convention before
/// the `(1 - mask) * -10000` flip; here we precompute the flipped form
/// directly.
///
/// Allocated on the device's unified memory and populated via direct
/// CPU pointer write (Apple Silicon: same bytes the GPU sees).
pub fn alloc_bert_attention_mask(
    device: &MlxDevice,
    seq_len: u32,
    valid_len: u32,
) -> Result<MlxBuffer> {
    if seq_len == 0 {
        return Err(anyhow!("alloc_bert_attention_mask: seq_len must be > 0"));
    }
    let n = (seq_len as usize) * (seq_len as usize);
    let buf = device
        .alloc_buffer(n * 4, DType::F32, vec![seq_len as usize, seq_len as usize])
        .map_err(|e| anyhow!("alloc attention mask: {e}"))?;
    // SAFETY: just-allocated f32 buffer; exclusive access.
    let s: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut f32, n) };
    let valid = valid_len.min(seq_len) as usize;
    let seq = seq_len as usize;
    for r in 0..seq {
        for c in 0..seq {
            s[r * seq + c] = if c < valid { 0.0 } else { -1e30 };
        }
    }
    Ok(buf)
}

/// Bidirectional self-attention with an explicit padding mask.
///
/// Same dispatch chain as `vit_attention_gpu` — Q@K^T → scale → softmax
/// → permute V → cast → transpose → scores @ V → permute back — but
/// inserts a `mask_add` step **between scaling and softmax**. Required
/// for BERT inputs that are right-padded with `[PAD]` to satisfy the
/// matmul K floor: without the mask, softmax distributes weight across
/// padded positions and contaminates the output (cosine 0.58 on bge-
/// small at iter 65 short-prompt; 0.82 on long-prompt where no padding
/// is needed).
///
/// Caller supplies a `[seq_len, seq_len]` F32 mask (build via
/// `alloc_bert_attention_mask`). The mask is broadcast across `num_heads`.
///
/// # Errors
/// - `head_dim < 32` (matmul kernel constraint)
/// - any dim is 0
/// - propagated from any sub-stage
#[allow(clippy::too_many_arguments)]
pub fn bert_attention_with_mask_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q_seq_major: &MlxBuffer,
    k_seq_major: &MlxBuffer,
    v_seq_major: &MlxBuffer,
    mask: &MlxBuffer,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
) -> Result<MlxBuffer> {
    use crate::inference::vision::vit_gpu::{
        vit_attention_scores_gpu, vit_softmax_last_dim_gpu,
    };
    use mlx_native::ops::transpose::{permute_021_f32, transpose_last2_bf16};

    if head_dim < 32 {
        return Err(anyhow!(
            "bert_attention_with_mask_gpu: head_dim ({}) must be >= 32",
            head_dim
        ));
    }
    if seq_len == 0 || num_heads == 0 {
        return Err(anyhow!(
            "bert_attention_with_mask_gpu: seq_len/num_heads must be > 0"
        ));
    }
    let metal_dev = device.metal_device();

    // --- Stage 1: scores = (Q @ K^T) * scale, [num_heads, seq, seq] ---
    let scores = vit_attention_scores_gpu(
        encoder, registry, device, q_seq_major, k_seq_major, seq_len, num_heads, head_dim, scale,
    )?;
    encoder.memory_barrier();

    // --- Stage 2 (NEW): apply padding mask before softmax ---
    let masked_scores = bert_attention_mask_add_gpu(
        encoder, registry, device, &scores, mask, num_heads, seq_len, seq_len,
    )?;
    encoder.memory_barrier();

    // --- Stage 3: softmax along last dim. [num_heads * seq, seq] ---
    let n_rows = (num_heads as u64) * (seq_len as u64);
    let softmaxed =
        vit_softmax_last_dim_gpu(encoder, registry, device, &masked_scores, n_rows as u32, seq_len)?;
    encoder.memory_barrier();

    // --- Stage 4: V layout transforms (mirrors vit_attention_gpu) ---
    let n_v = (seq_len as usize) * (num_heads as usize) * (head_dim as usize);
    let v_perm = device
        .alloc_buffer(
            n_v * 4,
            DType::F32,
            vec![num_heads as usize, seq_len as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc v_perm: {e}"))?;
    permute_021_f32(
        encoder,
        registry,
        metal_dev,
        v_seq_major,
        &v_perm,
        seq_len as usize,
        num_heads as usize,
        head_dim as usize,
    )
    .context("permute V seq→head major")?;
    encoder.memory_barrier();

    let v_bf16 = device
        .alloc_buffer(
            n_v * 2,
            DType::BF16,
            vec![num_heads as usize, seq_len as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc v_bf16: {e}"))?;
    cast(
        encoder,
        registry,
        metal_dev,
        &v_perm,
        &v_bf16,
        n_v,
        CastDirection::F32ToBF16,
    )
    .context("cast V f32→bf16")?;
    encoder.memory_barrier();

    let v_t_bf16 = device
        .alloc_buffer(
            n_v * 2,
            DType::BF16,
            vec![num_heads as usize, head_dim as usize, seq_len as usize],
        )
        .map_err(|e| anyhow!("alloc v_t_bf16: {e}"))?;
    transpose_last2_bf16(
        encoder,
        registry,
        metal_dev,
        &v_bf16,
        &v_t_bf16,
        num_heads as usize,
        seq_len as usize,
        head_dim as usize,
    )
    .context("transpose V last 2")?;
    encoder.memory_barrier();

    // --- Stage 5: scores @ V matmul ---
    let n_attn = (num_heads as usize) * (seq_len as usize) * (head_dim as usize);
    let mut attn_head_major = device
        .alloc_buffer(
            n_attn * 4,
            DType::F32,
            vec![num_heads as usize, seq_len as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc attn_head_major: {e}"))?;
    let params = DenseMmBf16F32Params {
        m: seq_len,
        n: head_dim,
        k: seq_len,
        src0_batch: num_heads,
        src1_batch: num_heads,
    };
    dense_matmul_bf16_f32_tensor(
        encoder, registry, device, &v_t_bf16, &softmaxed, &mut attn_head_major, &params,
    )
    .context("attention scores @ V matmul")?;
    encoder.memory_barrier();

    // --- Stage 6: permute back to seq-major ---
    let attn_seq_major = device
        .alloc_buffer(
            n_attn * 4,
            DType::F32,
            vec![seq_len as usize, num_heads as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc attn_seq_major: {e}"))?;
    permute_021_f32(
        encoder,
        registry,
        metal_dev,
        &attn_head_major,
        &attn_seq_major,
        num_heads as usize,
        seq_len as usize,
        head_dim as usize,
    )
    .context("permute attn head→seq major")?;
    Ok(attn_seq_major)
}

// ---------------------------------------------------------------------------
// Residual add — GPU dispatch (used internally by the block forward)
// ---------------------------------------------------------------------------

/// GPU residual add: `out[i] = a[i] + b[i]` over `n_elements` F32 values.
/// Allocates a fresh F32 output. Used internally by the encoder-block
/// composer; exposed as `pub` so callers can also chain residuals
/// outside the block path (e.g., the embedding sum in iter 60).
///
/// # Errors
/// - `n_elements == 0`
pub fn bert_residual_add_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    a: &MlxBuffer,
    b: &MlxBuffer,
    n_elements: u32,
) -> Result<MlxBuffer> {
    if n_elements == 0 {
        return Err(anyhow!("bert_residual_add_gpu: n_elements must be > 0"));
    }
    let out = device
        .alloc_buffer(
            (n_elements as usize) * 4,
            DType::F32,
            vec![n_elements as usize],
        )
        .map_err(|e| anyhow!("alloc residual add output: {e}"))?;
    elementwise_add(
        encoder,
        registry,
        device.metal_device(),
        a,
        b,
        &out,
        n_elements as usize,
        DType::F32,
    )
    .context("bert_residual_add_gpu: elementwise_add")?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Encoder block — full forward pass composer
// ---------------------------------------------------------------------------

/// Per-layer tensor refs that drive `apply_bert_encoder_block_gpu`. Keeps
/// the call site declarative (caller pulls each tensor from
/// `LoadedBertWeights::block_required` / `block_optional`) and keeps the
/// composer function signature stable as more variants surface.
///
/// All weights are F32 on-device. Biases are optional — variants that
/// drop biases pass `None` and the corresponding `bert_linear_gpu` call
/// skips the bias-add dispatch.
pub struct BertEncoderBlockTensors<'a> {
    /// Q/K/V projection: `[hidden, hidden]` weight, `[hidden]` optional bias.
    pub q_w: &'a MlxBuffer,
    pub q_b: Option<&'a MlxBuffer>,
    pub k_w: &'a MlxBuffer,
    pub k_b: Option<&'a MlxBuffer>,
    pub v_w: &'a MlxBuffer,
    pub v_b: Option<&'a MlxBuffer>,
    /// Attention output projection: `[hidden, hidden]` + optional `[hidden]` bias.
    pub o_w: &'a MlxBuffer,
    pub o_b: Option<&'a MlxBuffer>,
    /// Post-attention LayerNorm γ, β. Both `[hidden]`.
    pub attn_norm_gamma: &'a MlxBuffer,
    pub attn_norm_beta: &'a MlxBuffer,
    /// FFN up: `[intermediate, hidden]` + optional `[intermediate]` bias.
    pub up_w: &'a MlxBuffer,
    pub up_b: Option<&'a MlxBuffer>,
    /// FFN down: `[hidden, intermediate]` + optional `[hidden]` bias.
    pub down_w: &'a MlxBuffer,
    pub down_b: Option<&'a MlxBuffer>,
    /// Post-FFN LayerNorm γ, β. Both `[hidden]`.
    pub ffn_norm_gamma: &'a MlxBuffer,
    pub ffn_norm_beta: &'a MlxBuffer,
}

/// One BERT encoder block forward pass on GPU.
///
/// **Post-norm topology** (the BERT-original layout, **not** the modern
/// pre-norm layout that ViT/Llama use):
///
/// ```text
///   attn_in  = x
///   q, k, v  = LinearWb(attn_in)           // 3 separate projections
///   y        = bidirectional_attn(q, k, v) // softmax(QK^T / √d) @ V
///   y        = LinearWb(y)                  // output projection
///   x'       = LayerNorm(x + y)             // residual + post-attn LN
///   h        = LinearWb(x')                 // FFN up
///   h        = GeLU(h)
///   h        = LinearWb(h)                  // FFN down
///   x''      = LayerNorm(x' + h)            // residual + post-FFN LN
///   return x''
/// ```
///
/// Inputs (F32 on device):
/// - `input`   `[seq_len, hidden]` row-major.
/// - `tensors` per-layer weight bundle.
///
/// Returns F32 `[seq_len, hidden]` row-major.
///
/// `seq_len ≥ 32` floor inherited from `bert_attention_gpu` (post-softmax
/// matmul has K = seq_len).
///
/// `head_dim = hidden / num_heads` — the caller passes both so the
/// composer doesn't recompute and the function is reusable with
/// uneven splits if a future variant requires them.
///
/// Caller registers `register_bert_custom_shaders(&mut registry)` first.
///
/// # Errors
/// Propagated from any sub-primitive; primarily shape mismatches.
#[allow(clippy::too_many_arguments)]
pub fn apply_bert_encoder_block_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    tensors: &BertEncoderBlockTensors<'_>,
    attention_mask: Option<&MlxBuffer>,
    seq_len: u32,
    hidden: u32,
    num_heads: u32,
    intermediate: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    if hidden == 0 || num_heads == 0 || hidden % num_heads != 0 {
        return Err(anyhow!(
            "apply_bert_encoder_block_gpu: hidden ({}) must be > 0 and divisible by num_heads ({})",
            hidden,
            num_heads
        ));
    }
    let head_dim = hidden / num_heads;
    let n_hidden = (seq_len as usize) * (hidden as usize);
    let n_inter = (seq_len as usize) * (intermediate as usize);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // --- Q/K/V projections ---
    let q = bert_linear_gpu(
        encoder, registry, device, input, tensors.q_w, tensors.q_b, seq_len, hidden, hidden,
    )
    .context("encoder block: Q projection")?;
    encoder.memory_barrier();
    let k = bert_linear_gpu(
        encoder, registry, device, input, tensors.k_w, tensors.k_b, seq_len, hidden, hidden,
    )
    .context("encoder block: K projection")?;
    encoder.memory_barrier();
    let v = bert_linear_gpu(
        encoder, registry, device, input, tensors.v_w, tensors.v_b, seq_len, hidden, hidden,
    )
    .context("encoder block: V projection")?;
    encoder.memory_barrier();

    // --- Bidirectional attention (with optional padding mask) ---
    // Q/K/V buffers are `[seq_len, hidden]` F32; the attention dispatch
    // reads them as `[seq_len, num_heads, head_dim]` row-major (same
    // bytes; hidden = num_heads × head_dim by construction above). No
    // copy needed.
    let attn_out = match attention_mask {
        Some(mask) => bert_attention_with_mask_gpu(
            encoder, registry, device, &q, &k, &v, mask, seq_len, num_heads, head_dim, scale,
        )
        .context("encoder block: masked bidirectional attention")?,
        None => bert_attention_gpu(
            encoder, registry, device, &q, &k, &v, seq_len, num_heads, head_dim, scale,
        )
        .context("encoder block: bidirectional attention")?,
    };
    encoder.memory_barrier();

    // --- Output projection (W_o + b_o) ---
    let o_proj = bert_linear_gpu(
        encoder,
        registry,
        device,
        &attn_out,
        tensors.o_w,
        tensors.o_b,
        seq_len,
        hidden,
        hidden,
    )
    .context("encoder block: attention output projection")?;
    encoder.memory_barrier();

    // --- Residual add: x + o_proj ---
    let attn_residual = bert_residual_add_gpu(
        encoder,
        registry,
        device,
        input,
        &o_proj,
        n_hidden as u32,
    )
    .context("encoder block: attn residual add")?;
    encoder.memory_barrier();

    // --- Post-attention LayerNorm ---
    let post_attn = bert_layer_norm_gpu(
        encoder,
        registry,
        device,
        &attn_residual,
        tensors.attn_norm_gamma,
        tensors.attn_norm_beta,
        eps,
        seq_len,
        hidden,
    )
    .context("encoder block: post-attn LayerNorm")?;
    encoder.memory_barrier();

    // --- FFN up: [seq, hidden] → [seq, intermediate] ---
    let ffn_up = bert_linear_gpu(
        encoder,
        registry,
        device,
        &post_attn,
        tensors.up_w,
        tensors.up_b,
        seq_len,
        hidden,
        intermediate,
    )
    .context("encoder block: FFN up projection")?;
    encoder.memory_barrier();

    // --- GeLU activation ---
    let ffn_act = bert_gelu_gpu(encoder, registry, device, &ffn_up)
        .context("encoder block: GeLU")?;
    encoder.memory_barrier();

    // --- FFN down: [seq, intermediate] → [seq, hidden] ---
    let ffn_down = bert_linear_gpu(
        encoder,
        registry,
        device,
        &ffn_act,
        tensors.down_w,
        tensors.down_b,
        seq_len,
        intermediate,
        hidden,
    )
    .context("encoder block: FFN down projection")?;
    let _ = n_inter; // documents the intermediate-hidden product; not used after this point.
    encoder.memory_barrier();

    // --- Residual: post_attn + ffn_down ---
    let ffn_residual = bert_residual_add_gpu(
        encoder,
        registry,
        device,
        &post_attn,
        &ffn_down,
        n_hidden as u32,
    )
    .context("encoder block: FFN residual add")?;
    encoder.memory_barrier();

    // --- Post-FFN LayerNorm — block output ---
    bert_layer_norm_gpu(
        encoder,
        registry,
        device,
        &ffn_residual,
        tensors.ffn_norm_gamma,
        tensors.ffn_norm_beta,
        eps,
        seq_len,
        hidden,
    )
    .context("encoder block: post-FFN LayerNorm")
}

// ---------------------------------------------------------------------------
// Embedding gather — F32 row lookup
// ---------------------------------------------------------------------------

/// GPU embedding gather: for each `i in 0..n_ids`, copy `table[ids[i]]`
/// (a row of `hidden` F32 values) into `output[i]`. Wraps mlx-native's
/// `dispatch_gather_f32`.
///
/// Inputs:
/// - `table`  F32 `[vocab, hidden]`
/// - `ids`    U32 `[n_ids]`
///
/// Returns F32 `[n_ids, hidden]`.
///
/// # Errors
/// - `vocab`, `hidden`, or `n_ids` is zero
/// - propagated from gather kernel dispatch
pub fn bert_embed_gather_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    table: &MlxBuffer,
    ids: &MlxBuffer,
    vocab: u32,
    hidden: u32,
    n_ids: u32,
) -> Result<MlxBuffer> {
    if vocab == 0 || hidden == 0 || n_ids == 0 {
        return Err(anyhow!(
            "bert_embed_gather_gpu: vocab/hidden/n_ids must all be > 0 (got {} / {} / {})",
            vocab,
            hidden,
            n_ids
        ));
    }
    let total = (n_ids as usize) * (hidden as usize);
    let output = device
        .alloc_buffer(total * 4, DType::F32, vec![n_ids as usize, hidden as usize])
        .map_err(|e| anyhow!("alloc embed_gather output: {e}"))?;
    dispatch_gather_f32(
        encoder,
        registry,
        device.metal_device(),
        table,
        ids,
        &output,
        vocab,
        hidden,
        n_ids,
    )
    .map_err(|e| anyhow!("bert_embed_gather_gpu: dispatch_gather_f32: {e}"))?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// Embeddings layer (token + position + token_types → sum → LayerNorm)
// ---------------------------------------------------------------------------

/// Build a U32 buffer holding `[0, 1, 2, ..., seq_len - 1]` for the
/// position-embed gather. Allocated on `device`'s unified memory and
/// populated via the CPU pointer view (Apple Silicon: same bytes the
/// GPU sees).
fn alloc_position_ids(device: &MlxDevice, seq_len: u32) -> Result<MlxBuffer> {
    let n = seq_len as usize;
    let buf = device
        .alloc_buffer(n * 4, DType::U32, vec![n])
        .map_err(|e| anyhow!("alloc position_ids: {e}"))?;
    // SAFETY: just-allocated u32 buffer; exclusive access.
    let slice: &mut [u32] =
        unsafe { std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut u32, n) };
    for (i, slot) in slice.iter_mut().enumerate() {
        *slot = i as u32;
    }
    Ok(buf)
}

/// Build a U32 buffer holding `[0, 0, ..., 0]` of length `seq_len` —
/// the default BERT segment ids for single-segment input. Used by
/// `apply_bert_full_forward_gpu` when the caller passes `type_ids = None`
/// but the model has a token_types embedding table to keep parity with
/// llama-embedding's default behavior.
fn alloc_zero_type_ids(device: &MlxDevice, seq_len: u32) -> Result<MlxBuffer> {
    let n = seq_len as usize;
    let buf = device
        .alloc_buffer(n * 4, DType::U32, vec![n])
        .map_err(|e| anyhow!("alloc zero type_ids: {e}"))?;
    // SAFETY: just-allocated u32 buffer; exclusive access. Zero-init
    // explicit (alloc_buffer doesn't guarantee zeroed contents).
    let slice: &mut [u32] =
        unsafe { std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut u32, n) };
    for slot in slice.iter_mut() {
        *slot = 0;
    }
    Ok(buf)
}

/// Full BERT embeddings layer:
/// `out = LayerNorm(token_embd[input_ids] + position_embd[0..seq] + maybe token_types[type_ids])`
///
/// Inputs:
/// - `input_ids`           U32 `[seq_len]`
/// - `type_ids_opt`        U32 `[seq_len]` — optional (None for single-segment)
/// - `token_embd`          F32 `[vocab, hidden]`
/// - `position_embd`       F32 `[max_pos, hidden]`
/// - `token_types_opt`     F32 `[type_vocab, hidden]` — must accompany
///                          `type_ids_opt` (one None implies the other None)
/// - `embed_norm_gamma`,
///   `embed_norm_beta`     F32 `[hidden]`
///
/// Returns F32 `[seq_len, hidden]`.
///
/// `seq_len` must be ≤ `max_pos` (caller's responsibility — silent
/// out-of-bounds reads in the gather kernel otherwise).
///
/// `vocab`, `max_pos`, `type_vocab` are passed explicitly so the gather
/// kernel can validate index range. Real BERT GGUFs encode these in
/// `BertConfig.{vocab_size, max_position_embeddings, type_vocab_size}`.
///
/// # Errors
/// - any zero dimension
/// - `type_ids_opt.is_some() != token_types_opt.is_some()` (must match)
/// - propagated from sub-dispatches
#[allow(clippy::too_many_arguments)]
pub fn bert_embeddings_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_ids: &MlxBuffer,
    type_ids_opt: Option<&MlxBuffer>,
    token_embd: &MlxBuffer,
    position_embd: &MlxBuffer,
    token_types_opt: Option<&MlxBuffer>,
    embed_norm_gamma: &MlxBuffer,
    embed_norm_beta: &MlxBuffer,
    eps: f32,
    seq_len: u32,
    hidden: u32,
    vocab: u32,
    max_pos: u32,
    type_vocab: u32,
) -> Result<MlxBuffer> {
    if seq_len == 0 || hidden == 0 {
        return Err(anyhow!(
            "bert_embeddings_gpu: seq_len ({}) and hidden ({}) must be > 0",
            seq_len,
            hidden
        ));
    }
    if seq_len > max_pos {
        return Err(anyhow!(
            "bert_embeddings_gpu: seq_len ({}) exceeds max_pos ({})",
            seq_len,
            max_pos
        ));
    }
    match (type_ids_opt.is_some(), token_types_opt.is_some()) {
        (true, true) | (false, false) => {}
        (a, b) => {
            return Err(anyhow!(
                "bert_embeddings_gpu: type_ids and token_types must both be Some or both None (got {} / {})",
                a, b
            ));
        }
    }

    let n_hidden = (seq_len as usize) * (hidden as usize);

    // Token gather.
    let tok = bert_embed_gather_gpu(
        encoder, registry, device, token_embd, input_ids, vocab, hidden, seq_len,
    )
    .context("embeddings: token gather")?;
    encoder.memory_barrier();

    // Position gather. Position ids are constructed as 0..seq.
    let pos_ids = alloc_position_ids(device, seq_len)?;
    let pos = bert_embed_gather_gpu(
        encoder,
        registry,
        device,
        position_embd,
        &pos_ids,
        max_pos,
        hidden,
        seq_len,
    )
    .context("embeddings: position gather")?;
    encoder.memory_barrier();

    // First sum: token + position.
    let tok_pos = bert_residual_add_gpu(encoder, registry, device, &tok, &pos, n_hidden as u32)
        .context("embeddings: token + position add")?;
    encoder.memory_barrier();

    // Second sum (optional): + token_types.
    let summed = if let (Some(type_ids), Some(token_types)) = (type_ids_opt, token_types_opt) {
        let typ = bert_embed_gather_gpu(
            encoder,
            registry,
            device,
            token_types,
            type_ids,
            type_vocab,
            hidden,
            seq_len,
        )
        .context("embeddings: type gather")?;
        encoder.memory_barrier();
        let s = bert_residual_add_gpu(
            encoder,
            registry,
            device,
            &tok_pos,
            &typ,
            n_hidden as u32,
        )
        .context("embeddings: + token_types add")?;
        encoder.memory_barrier();
        s
    } else {
        tok_pos
    };

    // LayerNorm finalize.
    bert_layer_norm_gpu(
        encoder,
        registry,
        device,
        &summed,
        embed_norm_gamma,
        embed_norm_beta,
        eps,
        seq_len,
        hidden,
    )
    .context("embeddings: post-sum LayerNorm")
}

// ---------------------------------------------------------------------------
// Pooling — Mean / CLS / Last
// ---------------------------------------------------------------------------

/// Pooling reduction kind used by `bert_pool_gpu`. Mirrors the runtime
/// choice from `BertConfig.pooling_type` minus the variants
/// `/v1/embeddings` cannot return (`None`/`Rank`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BertPoolKind {
    /// Average across the sequence dimension.
    Mean,
    /// Take row 0 (CLS token).
    Cls,
    /// Take row `seq_len - 1` (last token).
    Last,
}

/// GPU pool: reduce a `[seq_len, hidden]` row-major matrix to a single
/// `[hidden]` row per `kind`. Returns a fresh F32 `[hidden]` buffer.
///
/// # Errors
/// - `seq_len == 0` or `hidden == 0`
/// - propagated from kernel dispatch
pub fn bert_pool_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    kind: BertPoolKind,
    seq_len: u32,
    hidden: u32,
) -> Result<MlxBuffer> {
    if seq_len == 0 || hidden == 0 {
        return Err(anyhow!(
            "bert_pool_gpu: seq_len ({}) and hidden ({}) must be > 0",
            seq_len,
            hidden
        ));
    }
    match kind {
        BertPoolKind::Mean => {
            let output = device
                .alloc_buffer((hidden as usize) * 4, DType::F32, vec![hidden as usize])
                .map_err(|e| anyhow!("alloc pool_mean output: {e}"))?;
            let pipeline = registry
                .get_pipeline("bert_pool_mean_f32", device.metal_device())
                .map_err(|e| anyhow!("bert_pool_gpu: get_pipeline: {e}"))?;
            let params = PoolMeanGpuParams { seq_len, hidden };
            let bytes = pod_as_bytes(&params);
            let grid = MTLSize::new(hidden as u64, 1, 1);
            let tg_x = std::cmp::min(64, hidden as u64);
            let tg = MTLSize::new(tg_x, 1, 1);
            encoder.encode_with_args(
                pipeline,
                &[
                    (0, KernelArg::Buffer(input)),
                    (1, KernelArg::Buffer(&output)),
                    (2, KernelArg::Bytes(bytes)),
                ],
                grid,
                tg,
            );
            Ok(output)
        }
        BertPoolKind::Cls | BertPoolKind::Last => {
            // Implement as a 1-element gather: row index = 0 (CLS) or
            // seq_len - 1 (Last). The gather kernel handles the F32
            // row copy directly without needing a custom shader.
            let idx_val: u32 = match kind {
                BertPoolKind::Cls => 0,
                BertPoolKind::Last => seq_len - 1,
                BertPoolKind::Mean => unreachable!(),
            };
            let idx_buf = device
                .alloc_buffer(4, DType::U32, vec![1])
                .map_err(|e| anyhow!("alloc pool index buffer: {e}"))?;
            // SAFETY: just-allocated u32 buffer of length 1.
            let s: &mut [u32] =
                unsafe { std::slice::from_raw_parts_mut(idx_buf.contents_ptr() as *mut u32, 1) };
            s[0] = idx_val;
            bert_embed_gather_gpu(encoder, registry, device, input, &idx_buf, seq_len, hidden, 1)
        }
    }
}

// ---------------------------------------------------------------------------
// L2 normalize
// ---------------------------------------------------------------------------

/// GPU L2 normalize per row: `out[r, i] = in[r, i] / sqrt(sum_i in[r,i]² + eps)`.
/// Wraps `mlx_native::ops::l2_norm::dispatch_l2_norm`. Allocates a fresh
/// F32 output buffer.
///
/// For pooled embeddings (a single row), pass `rows = 1`. For
/// per-token output (no pooling), pass `rows = seq_len`.
///
/// # Errors
/// - `rows == 0` or `dim == 0`
/// - propagated from dispatch
pub fn bert_l2_normalize_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    eps: f32,
    rows: u32,
    dim: u32,
) -> Result<MlxBuffer> {
    if rows == 0 || dim == 0 {
        return Err(anyhow!(
            "bert_l2_normalize_gpu: rows ({}) and dim ({}) must be > 0",
            rows,
            dim
        ));
    }
    let total = (rows as usize) * (dim as usize);
    let output = device
        .alloc_buffer(total * 4, DType::F32, vec![rows as usize, dim as usize])
        .map_err(|e| anyhow!("alloc l2_norm output: {e}"))?;
    // dispatch_l2_norm wants a 2-element F32 params buf [eps, dim].
    let params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc l2_norm params: {e}"))?;
    {
        // SAFETY: just-allocated f32 buffer.
        let s: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(params_buf.contents_ptr() as *mut f32, 2)
        };
        s[0] = eps;
        s[1] = dim as f32;
    }
    dispatch_l2_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        &output,
        &params_buf,
        rows,
        dim,
    )
    .map_err(|e| anyhow!("bert_l2_normalize_gpu: dispatch_l2_norm: {e}"))?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// Full forward pass — embed → N×block → pool → L2 normalize
// ---------------------------------------------------------------------------

/// One BERT-family forward pass on GPU, end-to-end:
/// `out = L2norm(Pool(EncoderBlocks×N(Embed(input_ids, type_ids))))`
///
/// Inputs:
/// - `input_ids`        U32 `[seq_len]`
/// - `type_ids_opt`     U32 `[seq_len]` (None → no segment embedding)
/// - `weights`          parsed BERT GGUF tensors (F32 on device)
/// - `cfg`              architecture config (drives layer count, hidden
///                      sizes, pooling kind, eps)
///
/// Returns F32 `[hidden]` — a single L2-normalized embedding vector.
/// `cfg.pooling_type` decides which pooling reduction to apply.
///
/// `seq_len` must be ≥ 32 (post-softmax matmul K floor) and ≤
/// `cfg.max_position_embeddings` (silent OOB read otherwise).
///
/// Caller registers `register_bert_custom_shaders(&mut registry)` first.
///
/// # Errors
/// - `cfg.pooling_type` is `None` or `Rank` (no embedding output for those)
/// - `seq_len < 32` or `seq_len > cfg.max_position_embeddings`
/// - missing tensor (propagated from `LoadedBertWeights` accessors)
/// - propagated from any sub-dispatch
pub fn apply_bert_full_forward_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_ids: &MlxBuffer,
    type_ids_opt: Option<&MlxBuffer>,
    weights: &super::weights::LoadedBertWeights,
    cfg: &super::config::BertConfig,
    seq_len: u32,
    valid_token_count: u32,
) -> Result<MlxBuffer> {
    use super::config::PoolingType;

    let pool_kind = match cfg.pooling_type {
        PoolingType::Mean => BertPoolKind::Mean,
        PoolingType::Cls => BertPoolKind::Cls,
        PoolingType::Last => BertPoolKind::Last,
        PoolingType::None => {
            return Err(anyhow!(
                "apply_bert_full_forward_gpu: pooling_type=None is not a single-vector embedding output"
            ));
        }
        PoolingType::Rank => {
            return Err(anyhow!(
                "apply_bert_full_forward_gpu: pooling_type=Rank is reranker-only (out of scope for /v1/embeddings)"
            ));
        }
    };

    let hidden = cfg.hidden_size as u32;
    let num_heads = cfg.num_attention_heads as u32;
    let intermediate = cfg.intermediate_size as u32;
    let vocab = cfg.vocab_size as u32;
    let max_pos = cfg.max_position_embeddings as u32;
    let type_vocab = cfg.type_vocab_size as u32;
    let eps = cfg.layer_norm_eps;

    if seq_len < 32 {
        return Err(anyhow!(
            "apply_bert_full_forward_gpu: seq_len ({}) must be >= 32 (post-softmax matmul K floor)",
            seq_len
        ));
    }
    if seq_len > max_pos {
        return Err(anyhow!(
            "apply_bert_full_forward_gpu: seq_len ({}) > max_position_embeddings ({})",
            seq_len,
            max_pos
        ));
    }

    // --- Embeddings ---
    // BERT GGUFs serialize `token_types.weight` whether or not the
    // caller has explicit type_ids, and llama-embedding adds
    // `token_types[0]` for every position on single-segment input.
    // To keep parity with that behavior, when the caller passes
    // `type_ids_opt = None` and the model HAS a token_types table,
    // synthesize an all-zeros `[seq_len]` u32 buffer and pass it
    // through. That matches HuggingFace's BertEmbeddings default
    // behavior for single-segment inputs.
    let synthesized_type_ids: Option<MlxBuffer> = match (type_ids_opt, weights.token_types_weight()) {
        (None, Some(_)) => Some(alloc_zero_type_ids(device, seq_len)?),
        _ => None,
    };
    let effective_type_ids: Option<&MlxBuffer> = match (type_ids_opt, synthesized_type_ids.as_ref()) {
        (Some(b), _) => Some(b),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    let token_types_for_call = if effective_type_ids.is_some() {
        weights.token_types_weight()
    } else {
        None
    };

    let mut hidden_states = bert_embeddings_gpu(
        encoder,
        registry,
        device,
        input_ids,
        effective_type_ids,
        weights.token_embd_weight()?,
        weights.position_embd_weight()?,
        token_types_for_call,
        weights.embed_norm_weight()?,
        weights.embed_norm_bias()?,
        eps,
        seq_len,
        hidden,
        vocab,
        max_pos,
        type_vocab,
    )
    .context("full forward: embeddings")?;
    encoder.memory_barrier();

    // --- Build attention mask if any padding present ---
    // Mask is `[seq_len, seq_len]` F32: 0.0 at columns < valid_token_count,
    // -1e30 at columns ≥ valid_token_count. Broadcast across num_heads
    // by `bert_attention_mask_add_gpu`. If valid_token_count == seq_len,
    // every position is real and the mask add becomes a no-op — skip.
    let mask_opt: Option<MlxBuffer> = if valid_token_count < seq_len {
        Some(alloc_bert_attention_mask(device, seq_len, valid_token_count)?)
    } else {
        None
    };
    let mask_ref = mask_opt.as_ref();

    // --- N encoder blocks ---
    for layer_idx in 0..cfg.num_hidden_layers {
        let tensors = BertEncoderBlockTensors {
            q_w: weights.block_required(layer_idx, "attn_q.weight")?,
            q_b: weights.block_optional(layer_idx, "attn_q.bias"),
            k_w: weights.block_required(layer_idx, "attn_k.weight")?,
            k_b: weights.block_optional(layer_idx, "attn_k.bias"),
            v_w: weights.block_required(layer_idx, "attn_v.weight")?,
            v_b: weights.block_optional(layer_idx, "attn_v.bias"),
            o_w: weights.block_required(layer_idx, "attn_output.weight")?,
            o_b: weights.block_optional(layer_idx, "attn_output.bias"),
            attn_norm_gamma: weights.block_required(layer_idx, "attn_output_norm.weight")?,
            attn_norm_beta: weights.block_required(layer_idx, "attn_output_norm.bias")?,
            up_w: weights.block_required(layer_idx, "ffn_up.weight")?,
            up_b: weights.block_optional(layer_idx, "ffn_up.bias"),
            down_w: weights.block_required(layer_idx, "ffn_down.weight")?,
            down_b: weights.block_optional(layer_idx, "ffn_down.bias"),
            ffn_norm_gamma: weights.block_required(layer_idx, "layer_output_norm.weight")?,
            ffn_norm_beta: weights.block_required(layer_idx, "layer_output_norm.bias")?,
        };
        hidden_states = apply_bert_encoder_block_gpu(
            encoder,
            registry,
            device,
            &hidden_states,
            &tensors,
            mask_ref,
            seq_len,
            hidden,
            num_heads,
            intermediate,
            eps,
        )
        .with_context(|| format!("full forward: encoder block {}", layer_idx))?;
        encoder.memory_barrier();
    }

    // --- Pool ---
    let pooled =
        bert_pool_gpu(encoder, registry, device, &hidden_states, pool_kind, seq_len, hidden)
            .context("full forward: pool")?;
    encoder.memory_barrier();

    // --- L2 normalize ---
    bert_l2_normalize_gpu(encoder, registry, device, &pooled, eps, 1, hidden)
        .context("full forward: l2 normalize")
}

// ---------------------------------------------------------------------------
// CPU reference (parity oracle for tests)
// ---------------------------------------------------------------------------

/// CPU reference LayerNorm — used by tests only. F32 throughout. Output
/// shape `[batch, hidden]` row-major matches the GPU kernel.
/// CPU reference linear projection: `out[m, n] = sum_k input[m, k] *
/// weight[n, k] + bias[n]` (when bias supplied; zero otherwise).
/// Test-only; F32 throughout.
#[cfg(test)]
fn bert_linear_cpu_ref(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    seq: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), seq * in_features);
    assert_eq!(weight.len(), out_features * in_features);
    if let Some(b) = bias {
        assert_eq!(b.len(), out_features);
    }
    let mut out = vec![0.0f32; seq * out_features];
    for m in 0..seq {
        for n in 0..out_features {
            let mut acc = 0.0f64;
            for k in 0..in_features {
                acc += (input[m * in_features + k] as f64) * (weight[n * in_features + k] as f64);
            }
            let mut v = acc as f32;
            if let Some(b) = bias {
                v += b[n];
            }
            out[m * out_features + n] = v;
        }
    }
    out
}

/// CPU reference GeLU pytorch_tanh: `0.5 * x * (1 + tanh(sqrt(2/pi) *
/// (x + 0.044715 * x^3)))`. Test-only; F32 throughout. Constants match
/// llama.cpp's `ggml_gelu_f32` (`/opt/llama.cpp/ggml/src/ggml.c`).
#[cfg(test)]
fn bert_gelu_cpu_ref(input: &[f32]) -> Vec<f32> {
    const GELU_COEF_A: f32 = 0.044715;
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    input
        .iter()
        .map(|&x| {
            let inner = SQRT_2_OVER_PI * (x + GELU_COEF_A * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect()
}

/// CPU reference bidirectional self-attention: identical math to
/// `bert_attention_gpu`, F64 accumulators for parity-bar sharpness.
///
/// Inputs (seq-major, F32):
///   `q`, `k`, `v` each `[seq_len, num_heads, head_dim]`.
/// Returns `[seq_len, num_heads, head_dim]` F32.
#[cfg(test)]
fn bert_attention_cpu_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let n = seq_len * num_heads * head_dim;
    assert_eq!(q.len(), n);
    assert_eq!(k.len(), n);
    assert_eq!(v.len(), n);
    fn row(buf: &[f32], s: usize, h: usize, num_heads: usize, head_dim: usize) -> &[f32] {
        let off = (s * num_heads + h) * head_dim;
        &buf[off..off + head_dim]
    }
    let mut out = vec![0.0f32; n];
    for h in 0..num_heads {
        // Per-head scores `[seq_q, seq_k]`.
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for sq in 0..seq_len {
            for sk in 0..seq_len {
                let mut acc = 0.0f64;
                let qi = row(q, sq, h, num_heads, head_dim);
                let ki = row(k, sk, h, num_heads, head_dim);
                for d in 0..head_dim {
                    acc += qi[d] as f64 * ki[d] as f64;
                }
                scores[sq * seq_len + sk] = (acc as f32) * scale;
            }
        }
        // Softmax row-wise.
        for sq in 0..seq_len {
            let row_off = sq * seq_len;
            let mut m = scores[row_off];
            for sk in 1..seq_len {
                m = m.max(scores[row_off + sk]);
            }
            let mut sum = 0.0f64;
            for sk in 0..seq_len {
                let e = ((scores[row_off + sk] - m) as f64).exp();
                scores[row_off + sk] = e as f32;
                sum += e;
            }
            let inv = (1.0 / sum) as f32;
            for sk in 0..seq_len {
                scores[row_off + sk] *= inv;
            }
        }
        // Apply: out[sq, h, d] = sum_sk scores[sq, sk] * v[sk, h, d]
        for sq in 0..seq_len {
            let out_off = (sq * num_heads + h) * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for sk in 0..seq_len {
                    let vi = row(v, sk, h, num_heads, head_dim);
                    acc += (scores[sq * seq_len + sk] as f64) * (vi[d] as f64);
                }
                out[out_off + d] = acc as f32;
            }
        }
    }
    out
}

/// CPU reference for the BERT encoder block. Same math as
/// `apply_bert_encoder_block_gpu`, intended as a parity oracle for the
/// composer-level test. Composes the existing CPU references rather
/// than re-deriving the math.
///
/// Tensors are passed by value as `Vec<f32>` slices for test ergonomics.
/// This is test-only and slow (no SIMD), so it stays behind cfg(test).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn apply_bert_encoder_block_cpu_ref(
    input: &[f32],
    q_w: &[f32],
    q_b: Option<&[f32]>,
    k_w: &[f32],
    k_b: Option<&[f32]>,
    v_w: &[f32],
    v_b: Option<&[f32]>,
    o_w: &[f32],
    o_b: Option<&[f32]>,
    attn_gamma: &[f32],
    attn_beta: &[f32],
    up_w: &[f32],
    up_b: Option<&[f32]>,
    down_w: &[f32],
    down_b: Option<&[f32]>,
    ffn_gamma: &[f32],
    ffn_beta: &[f32],
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    intermediate: usize,
    eps: f32,
) -> Vec<f32> {
    let head_dim = hidden / num_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = bert_linear_cpu_ref(input, q_w, q_b, seq_len, hidden, hidden);
    let k = bert_linear_cpu_ref(input, k_w, k_b, seq_len, hidden, hidden);
    let v = bert_linear_cpu_ref(input, v_w, v_b, seq_len, hidden, hidden);
    let attn = bert_attention_cpu_ref(&q, &k, &v, seq_len, num_heads, head_dim, scale);
    let o = bert_linear_cpu_ref(&attn, o_w, o_b, seq_len, hidden, hidden);

    let n_hidden = seq_len * hidden;
    let mut residual = vec![0.0f32; n_hidden];
    for i in 0..n_hidden {
        residual[i] = input[i] + o[i];
    }
    let post_attn = bert_layer_norm_cpu_ref(&residual, attn_gamma, attn_beta, eps, seq_len, hidden);

    let ffn_up = bert_linear_cpu_ref(&post_attn, up_w, up_b, seq_len, hidden, intermediate);
    let ffn_act = bert_gelu_cpu_ref(&ffn_up);
    let ffn_down = bert_linear_cpu_ref(&ffn_act, down_w, down_b, seq_len, intermediate, hidden);

    let mut residual2 = vec![0.0f32; n_hidden];
    for i in 0..n_hidden {
        residual2[i] = post_attn[i] + ffn_down[i];
    }
    bert_layer_norm_cpu_ref(&residual2, ffn_gamma, ffn_beta, eps, seq_len, hidden)
}

#[cfg(test)]
fn bert_layer_norm_cpu_ref(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    batch: usize,
    hidden: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), batch * hidden);
    assert_eq!(gamma.len(), hidden);
    assert_eq!(beta.len(), hidden);
    let mut out = vec![0.0f32; batch * hidden];
    for r in 0..batch {
        let row = &input[r * hidden..(r + 1) * hidden];
        let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
        let var: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for h in 0..hidden {
            out[r * hidden + h] = (row[h] - mean) * inv_std * gamma[h] + beta[h];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::GraphExecutor;

    fn upload_f32(device: &MlxDevice, data: &[f32], shape: Vec<usize>) -> MlxBuffer {
        let bytes = data.len() * 4;
        let buf = device.alloc_buffer(bytes, DType::F32, shape).unwrap();
        // SAFETY: just allocated this buffer with f32 dtype; exclusive
        // access. Apple Silicon unified memory makes the contents_ptr
        // a CPU-visible pointer to the same bytes the GPU sees.
        let slice: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut f32, data.len())
        };
        slice.copy_from_slice(data);
        buf
    }

    fn readback_f32(buf: &MlxBuffer, expected_len: usize) -> Vec<f32> {
        let slice: &[f32] = buf.as_slice::<f32>().expect("readback as_slice");
        assert_eq!(slice.len(), expected_len, "readback length mismatch");
        slice.to_vec()
    }

    fn run_layer_norm(
        input_data: &[f32],
        gamma_data: &[f32],
        beta_data: &[f32],
        eps: f32,
        batch: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        // SAFETY: executor outlives session below; this raw-borrow trick
        // mirrors the vit_gpu test helpers' pattern. The aliasing is
        // safe because nothing else mutates the device for the duration
        // of the test, and the GPU dispatch only reads device.metal_device().
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        let input = upload_f32(device, input_data, vec![batch, hidden]);
        let gamma = upload_f32(device, gamma_data, vec![hidden]);
        let beta = upload_f32(device, beta_data, vec![hidden]);

        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let output = bert_layer_norm_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input,
            &gamma,
            &beta,
            eps,
            batch as u32,
            hidden as u32,
        )
        .expect("gpu dispatch");
        session.finish().expect("finish");
        readback_f32(&output, batch * hidden)
    }

    #[test]
    fn prev_pow2_table() {
        assert_eq!(prev_pow2(1), 1);
        assert_eq!(prev_pow2(2), 2);
        assert_eq!(prev_pow2(3), 2);
        assert_eq!(prev_pow2(255), 128);
        assert_eq!(prev_pow2(256), 256);
        assert_eq!(prev_pow2(257), 256);
        assert_eq!(prev_pow2(384), 256);
        assert_eq!(prev_pow2(1024), 1024);
    }

    #[test]
    fn cpu_ref_matches_known_value() {
        // Hand-computed: input = [1, 2, 3, 4], gamma=[1,1,1,1], beta=[0,0,0,0],
        // eps = 0. mean = 2.5. var = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²)/4
        //                          = (2.25 + 0.25 + 0.25 + 2.25)/4 = 5/4 = 1.25.
        // inv_std = 1/sqrt(1.25). out[i] = (x[i]-2.5) * inv_std.
        let out = bert_layer_norm_cpu_ref(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0, 0.0],
            0.0,
            1,
            4,
        );
        let inv_std = 1.0 / 1.25f32.sqrt();
        for (got, expected) in out.iter().zip([
            (1.0 - 2.5) * inv_std,
            (2.0 - 2.5) * inv_std,
            (3.0 - 2.5) * inv_std,
            (4.0 - 2.5) * inv_std,
        ]) {
            assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
        }
    }

    #[test]
    fn gpu_matches_cpu_on_synthetic_small_input() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let batch = 4usize;
        let hidden = 8usize;
        let input: Vec<f32> = (0..batch * hidden).map(|i| 0.1 * (i as f32) - 0.4).collect();
        let gamma: Vec<f32> = (0..hidden).map(|i| 1.0 + 0.05 * i as f32).collect();
        let beta: Vec<f32> = (0..hidden).map(|i| 0.01 * i as f32).collect();
        let eps = 1e-12;
        let cpu = bert_layer_norm_cpu_ref(&input, &gamma, &beta, eps, batch, hidden);
        let gpu = run_layer_norm(&input, &gamma, &beta, eps, batch, hidden);
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-5,
                "row {} col {}: gpu={} cpu={} diff={}",
                i / hidden,
                i % hidden,
                g,
                c,
                (g - c).abs()
            );
        }
    }

    #[test]
    fn gpu_constant_input_yields_bias_only_output() {
        // When input is constant per row, mean = x, var = 0 — so the
        // (x-mean) term zeroes and the output collapses to `beta`. eps is
        // irrelevant (var=0 + eps doesn't matter because the numerator
        // is zero). Validates that the two-pass reduction handles the
        // var=0 case without NaN.
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let batch = 2usize;
        let hidden = 16usize;
        let input = vec![3.5f32; batch * hidden];
        let gamma = vec![2.0f32; hidden]; // gamma irrelevant when (x-mean)=0
        let beta: Vec<f32> = (0..hidden).map(|i| 0.1 * i as f32 - 0.7).collect();
        let eps = 1e-5;
        let gpu = run_layer_norm(&input, &gamma, &beta, eps, batch, hidden);
        for r in 0..batch {
            for h in 0..hidden {
                let got = gpu[r * hidden + h];
                let want = beta[h];
                assert!(
                    (got - want).abs() < 1e-6,
                    "row {} col {}: got {} want {}",
                    r,
                    h,
                    got,
                    want
                );
            }
        }
    }

    #[test]
    fn gpu_matches_cpu_at_bge_small_hidden_384() {
        // Deterministic pseudo-random input to exercise the full
        // reduction width without depending on a real model.
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let batch = 32usize; // typical sequence length floor
        let hidden = 384usize; // bge-small-en-v1.5
        let input: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i.wrapping_mul(2654435761) % 1000) as f32) * 0.001 - 0.5)
            .collect();
        let gamma: Vec<f32> = (0..hidden).map(|i| 1.0 - 0.001 * i as f32).collect();
        let beta: Vec<f32> = (0..hidden).map(|i| 0.0001 * i as f32).collect();
        let eps = 1e-12;
        let cpu = bert_layer_norm_cpu_ref(&input, &gamma, &beta, eps, batch, hidden);
        let gpu = run_layer_norm(&input, &gamma, &beta, eps, batch, hidden);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        // F32 throughout, 384-wide reduction → at most ~1e-5 relative
        // error from accumulation order divergence between CPU and GPU.
        assert!(max_diff < 1e-4, "max_diff at bge-small shape: {max_diff}");
    }

    #[test]
    fn gpu_matches_cpu_at_mxbai_large_hidden_1024() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let batch = 8usize;
        let hidden = 1024usize; // mxbai-embed-large-v1
        let input: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i.wrapping_mul(2246822519) % 700) as f32) * 0.001 - 0.35)
            .collect();
        let gamma: Vec<f32> = (0..hidden).map(|i| 0.5 + 0.001 * i as f32).collect();
        let beta = vec![0.0f32; hidden];
        let eps = 1e-12;
        let cpu = bert_layer_norm_cpu_ref(&input, &gamma, &beta, eps, batch, hidden);
        let gpu = run_layer_norm(&input, &gamma, &beta, eps, batch, hidden);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        // Tighter tolerance — gamma magnitudes are O(1) so noise budget
        // stays at the F32 round-off line.
        assert!(max_diff < 2e-4, "max_diff at mxbai shape: {max_diff}");
    }

    #[test]
    fn gpu_rejects_zero_dimensions() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device available: {e}");
                return;
            }
        };
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        // SAFETY: executor owns the device; alias is stable for the duration of the test.
        let device: &MlxDevice = unsafe { &*device_ref };
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let buf = upload_f32(device, &[0.0; 4], vec![1, 4]);
        let err = bert_layer_norm_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &buf,
            &buf,
            &buf,
            1e-12,
            0,
            4,
        );
        assert!(err.is_err(), "batch=0 must error");
        let err = bert_layer_norm_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &buf,
            &buf,
            &buf,
            1e-12,
            4,
            0,
        );
        assert!(err.is_err(), "hidden=0 must error");
        // Drop the session without finishing — no real dispatch was issued
        // (both calls errored before encode). Dropping aborts the
        // command buffer cleanly.
        drop(session);
    }

    // -----------------------------------------------------------------
    // bert_linear_gpu — Wx + b on real Metal
    // -----------------------------------------------------------------

    fn run_linear(
        input_data: &[f32],
        weight_data: &[f32],
        bias_data: Option<&[f32]>,
        seq_len: usize,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        // SAFETY: executor outlives session.
        let device: &MlxDevice = unsafe { &*device_ref };
        let input = upload_f32(device, input_data, vec![seq_len, in_features]);
        let weight = upload_f32(device, weight_data, vec![out_features, in_features]);
        let bias_buf = bias_data.map(|b| upload_f32(device, b, vec![out_features]));
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let output = bert_linear_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input,
            &weight,
            bias_buf.as_ref(),
            seq_len as u32,
            in_features as u32,
            out_features as u32,
        )
        .expect("gpu dispatch");
        session.finish().expect("finish");
        readback_f32(&output, seq_len * out_features)
    }

    #[test]
    fn linear_no_bias_matches_cpu_on_small_input() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let seq = 4usize;
        let in_f = 64usize; // ≥ 32 (BF16 matmul kernel constraint)
        let out_f = 32usize;
        let input: Vec<f32> = (0..seq * in_f).map(|i| 0.01 * (i as f32) - 0.3).collect();
        let weight: Vec<f32> = (0..out_f * in_f)
            .map(|i| 0.005 * (i as f32) - 0.4)
            .collect();
        let cpu = bert_linear_cpu_ref(&input, &weight, None, seq, in_f, out_f);
        let gpu = run_linear(&input, &weight, None, seq, in_f, out_f);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        // BF16 weight cast: per-element bound is K × max(|x|) × max(|w|)
        // × 2⁻⁸ (BF16 half-ULP). For K=64, |x|<0.5, |w|<0.5: bound ≈
        // 64 × 0.5 × 0.5 × 4e-3 = 0.064. Observed 0.139 in dev — within
        // 2× of the worst-case bound; pick 0.20 for a stable envelope
        // that's still tight enough to catch a real correctness regression.
        assert!(max_diff < 0.20, "max_diff {} > 0.20", max_diff);
    }

    #[test]
    fn linear_with_bias_matches_cpu_on_small_input() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let seq = 4usize;
        let in_f = 64usize;
        let out_f = 32usize;
        let input: Vec<f32> = (0..seq * in_f).map(|i| 0.01 * (i as f32) - 0.3).collect();
        let weight: Vec<f32> = (0..out_f * in_f)
            .map(|i| 0.005 * (i as f32) - 0.4)
            .collect();
        let bias: Vec<f32> = (0..out_f).map(|i| 0.1 * i as f32 - 1.0).collect();
        let cpu = bert_linear_cpu_ref(&input, &weight, Some(&bias), seq, in_f, out_f);
        let gpu = run_linear(&input, &weight, Some(&bias), seq, in_f, out_f);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        // Bias add is exact (F32 add); same noise budget as no-bias
        // (~K × |x|×|w|×2⁻⁸ ≈ 0.064 worst-case for K=64). Same envelope.
        assert!(max_diff < 0.20, "max_diff {} > 0.20 (bias path)", max_diff);
    }

    #[test]
    fn linear_at_bge_small_qkv_shape() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        // Q-projection shape for bge-small-en-v1.5: seq=32, hidden=384.
        let seq = 32usize;
        let hidden = 384usize;
        let input: Vec<f32> = (0..seq * hidden)
            .map(|i| ((i.wrapping_mul(2654435761) % 1000) as f32) * 0.001 - 0.5)
            .collect();
        let weight: Vec<f32> = (0..hidden * hidden)
            .map(|i| ((i.wrapping_mul(40503) % 600) as f32) * 0.001 - 0.3)
            .collect();
        let bias: Vec<f32> = (0..hidden).map(|i| 0.0001 * i as f32).collect();
        let cpu = bert_linear_cpu_ref(&input, &weight, Some(&bias), seq, hidden, hidden);
        let gpu = run_linear(&input, &weight, Some(&bias), seq, hidden, hidden);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        // K=384 with |x|<0.5 |w|<0.3: budget ≈ 0.5 × 0.3 × 8e-3 × √384
        //                                    ≈ 0.024.
        assert!(
            max_diff < 5e-2,
            "max_diff {} > 0.05 at bge-small QKV shape",
            max_diff
        );
    }

    #[test]
    fn linear_rejects_in_features_below_32() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let input = upload_f32(device, &[0.0; 16], vec![1, 16]);
        let weight = upload_f32(device, &[0.0; 16 * 8], vec![8, 16]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let err = bert_linear_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input,
            &weight,
            None,
            1,
            16, // in_features < 32 → reject
            8,
        );
        assert!(err.is_err(), "in_features=16 must error");
        drop(session);
    }

    #[test]
    fn linear_rejects_bias_size_mismatch() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let input = upload_f32(device, &[0.0; 64], vec![1, 64]);
        let weight = upload_f32(device, &[0.0; 64 * 8], vec![8, 64]);
        let bias_wrong = upload_f32(device, &[0.0; 4], vec![4]); // expected 8
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let err = bert_linear_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input,
            &weight,
            Some(&bias_wrong),
            1,
            64,
            8,
        );
        assert!(err.is_err(), "bias size mismatch must error");
        drop(session);
    }

    // -----------------------------------------------------------------
    // bert_gelu_gpu — pytorch_tanh approximation
    // -----------------------------------------------------------------

    fn run_gelu(input_data: &[f32]) -> Vec<f32> {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let input = upload_f32(device, input_data, vec![input_data.len()]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let output =
            bert_gelu_gpu(session.encoder_mut(), &mut registry, device, &input).expect("gpu");
        session.finish().expect("finish");
        readback_f32(&output, input_data.len())
    }

    #[test]
    fn gelu_cpu_ref_known_values() {
        // gelu(0) = 0; gelu(1) ≈ 0.8412; gelu(-1) ≈ -0.1588.
        let out = bert_gelu_cpu_ref(&[0.0, 1.0, -1.0]);
        assert!(out[0].abs() < 1e-6);
        assert!((out[1] - 0.8411920071).abs() < 1e-4);
        assert!((out[2] - -0.1588079929).abs() < 1e-4);
    }

    #[test]
    fn gelu_gpu_matches_cpu_small_input() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let input: Vec<f32> = (-32..32).map(|i| 0.1 * i as f32).collect();
        let cpu = bert_gelu_cpu_ref(&input);
        let gpu = run_gelu(&input);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        assert!(max_diff < 1e-5, "gelu max_diff: {max_diff}");
    }

    #[test]
    fn gelu_gpu_matches_cpu_at_bge_small_ffn_shape() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        // bge-small FFN intermediate: seq=32, intermediate=1536. Total
        // 49152 elements — exercises the linear thread-grid sweep.
        let n: usize = 32 * 1536;
        let input: Vec<f32> = (0..n)
            .map(|i| ((i.wrapping_mul(2654435761usize) % 1000) as f32) * 0.005 - 2.5)
            .collect();
        let cpu = bert_gelu_cpu_ref(&input);
        let gpu = run_gelu(&input);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        assert!(max_diff < 1e-5, "gelu max_diff at bge FFN shape: {max_diff}");
    }

    // -----------------------------------------------------------------
    // bert_bias_add_gpu — per-column broadcast add
    // -----------------------------------------------------------------

    #[test]
    fn bias_add_gpu_matches_cpu_small() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let rows = 5usize;
        let cols = 7usize;
        let input: Vec<f32> = (0..rows * cols).map(|i| 0.01 * i as f32).collect();
        let bias: Vec<f32> = (0..cols).map(|i| 0.1 * i as f32 - 0.3).collect();
        let inp_buf = upload_f32(device, &input, vec![rows, cols]);
        let bias_buf = upload_f32(device, &bias, vec![cols]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let out_buf = bert_bias_add_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &inp_buf,
            &bias_buf,
            rows as u32,
            cols as u32,
        )
        .expect("dispatch");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let expected = input[r * cols + c] + bias[c];
                let actual = got[r * cols + c];
                assert!(
                    (expected - actual).abs() < 1e-6,
                    "row {} col {}: got {} expected {}",
                    r,
                    c,
                    actual,
                    expected
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // bert_attention_gpu — bidirectional self-attention parity
    // -----------------------------------------------------------------

    fn run_attention(
        q_data: &[f32],
        k_data: &[f32],
        v_data: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        // SAFETY: executor outlives session.
        let device: &MlxDevice = unsafe { &*device_ref };
        let shape = vec![seq_len, num_heads, head_dim];
        let q = upload_f32(device, q_data, shape.clone());
        let k = upload_f32(device, k_data, shape.clone());
        let v = upload_f32(device, v_data, shape);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let out = bert_attention_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &q,
            &k,
            &v,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
            scale,
        )
        .expect("dispatch");
        session.finish().expect("finish");
        readback_f32(&out, seq_len * num_heads * head_dim)
    }

    #[test]
    fn cpu_ref_attention_simple_softmax() {
        // Single-head, head_dim=4, seq=2. Q[0]=K[0], Q[1]=K[1], all
        // orthogonal → softmax should heavily prefer the matching key,
        // so out[s] ≈ V[s] for s in 0..2.
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = q.clone();
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let scale = 100.0; // large scale → softmax is essentially one-hot
        let out = bert_attention_cpu_ref(&q, &k, &v, 2, 1, 4, scale);
        for d in 0..4 {
            assert!(
                (out[d] - v[d]).abs() < 1e-3,
                "row 0 d={d}: got {}, want {}",
                out[d],
                v[d]
            );
            assert!(
                (out[4 + d] - v[4 + d]).abs() < 1e-3,
                "row 1 d={d}: got {}, want {}",
                out[4 + d],
                v[4 + d]
            );
        }
    }

    #[test]
    fn attention_gpu_matches_cpu_at_synthetic_small_input() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        // seq_len = 32 is the floor: the post-softmax `scores @ V`
        // matmul has K = seq_len, and dense_matmul_bf16_f32_tensor
        // rejects K < 32. Real BERT prompts always exceed this.
        let seq = 32usize;
        let num_heads = 1usize;
        let head_dim = 32usize; // ≥ 32 (matmul kernel constraint)
        let n = seq * num_heads * head_dim;
        let q: Vec<f32> = (0..n).map(|i| 0.05 * (i as f32) - 0.5).collect();
        let k: Vec<f32> = (0..n).map(|i| 0.04 * (i as f32) - 0.4).collect();
        let v: Vec<f32> = (0..n).map(|i| 0.03 * (i as f32) - 0.3).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let cpu = bert_attention_cpu_ref(&q, &k, &v, seq, num_heads, head_dim, scale);
        let gpu = run_attention(&q, &k, &v, seq, num_heads, head_dim, scale);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        // Two BF16 cast points: K in `Q@K^T` (head_dim=32 reduction) and
        // V in `scores@V` (seq=32 reduction). Per-element noise envelope
        // ≈ K × |q|×|k| × 2⁻⁸ on each. With |q|/|k|/|v| ≈ 1.0 here,
        // expect ≤ 32 × 1.0 × 1.0 × 4e-3 ≈ 0.128 worst-case. Observed
        // ≈ 0.091 in dev — well within. Use 0.20 envelope.
        assert!(max_diff < 0.20, "max_diff at synthetic small: {max_diff}");
    }

    #[test]
    fn attention_gpu_matches_cpu_at_bge_small_attention_shape() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        // bge-small-en-v1.5: hidden=384, num_heads=12, head_dim=32.
        // seq=32 satisfies the K≥32 floor of the post-softmax matmul.
        let seq = 32usize;
        let num_heads = 12usize;
        let head_dim = 32usize;
        let n = seq * num_heads * head_dim;
        let q: Vec<f32> = (0..n)
            .map(|i| ((i.wrapping_mul(2654435761usize) % 1000) as f32) * 0.001 - 0.5)
            .collect();
        let k: Vec<f32> = (0..n)
            .map(|i| ((i.wrapping_mul(40503usize) % 700) as f32) * 0.001 - 0.35)
            .collect();
        let v: Vec<f32> = (0..n)
            .map(|i| ((i.wrapping_mul(2246822519usize) % 800) as f32) * 0.001 - 0.4)
            .collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let cpu = bert_attention_cpu_ref(&q, &k, &v, seq, num_heads, head_dim, scale);
        let gpu = run_attention(&q, &k, &v, seq, num_heads, head_dim, scale);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        // Larger envelope: post-softmax matmul has K=seq=8 contraction
        // on BF16-cast V; per-element noise ≈ 8 × 0.4 × 1.0 × 4e-3 ≈
        // 0.013. Plus the BF16 K cast in scores can perturb softmax
        // slightly. Observed ~0.03 in dev; pick 0.10 with margin.
        assert!(max_diff < 0.10, "max_diff at bge-small attn: {max_diff}");
    }

    #[test]
    fn attention_gpu_rejects_small_head_dim() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let buf = upload_f32(device, &[0.0; 16], vec![1, 1, 16]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let err = bert_attention_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &buf,
            &buf,
            &buf,
            1,
            1,
            16, // head_dim < 32 → reject
            1.0,
        );
        assert!(err.is_err(), "head_dim=16 must error");
        drop(session);
    }

    // -----------------------------------------------------------------
    // apply_bert_encoder_block_gpu — full block parity vs CPU
    // -----------------------------------------------------------------

    #[test]
    fn encoder_block_gpu_matches_cpu_ref_at_minimal_shape() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        // Minimal shape that satisfies all constraints:
        //   seq_len  >= 32 (post-softmax matmul K floor)
        //   head_dim >= 32 (matmul kernel constraint)
        //   in_features >= 32 (linear projection constraint)
        // hidden = 64, num_heads = 2 → head_dim = 32.
        // intermediate = 128.
        let seq = 32usize;
        let hidden = 64usize;
        let num_heads = 2usize;
        let intermediate = 128usize;
        let eps = 1e-12f32;

        // Deterministic pseudo-random tensors. Magnitudes kept small so
        // softmax stays unsaturated and the CPU/GPU agree within the
        // BF16-cast envelope of the underlying linear and attention paths.
        let prand = |seed: usize, n: usize, scale: f32, offset: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    ((i.wrapping_mul(2654435761usize).wrapping_add(seed) % 1000) as f32) * scale
                        + offset
                })
                .collect()
        };
        let input = prand(1, seq * hidden, 0.001, -0.5);
        let q_w = prand(2, hidden * hidden, 0.0005, -0.15);
        let q_b = prand(3, hidden, 0.001, -0.05);
        let k_w = prand(4, hidden * hidden, 0.0005, -0.15);
        let k_b = prand(5, hidden, 0.001, -0.05);
        let v_w = prand(6, hidden * hidden, 0.0005, -0.15);
        let v_b = prand(7, hidden, 0.001, -0.05);
        let o_w = prand(8, hidden * hidden, 0.0005, -0.15);
        let o_b = prand(9, hidden, 0.001, -0.05);
        let attn_gamma = prand(10, hidden, 0.0001, 1.0); // close to 1.0
        let attn_beta = prand(11, hidden, 0.001, -0.5);
        let up_w = prand(12, intermediate * hidden, 0.0005, -0.15);
        let up_b = prand(13, intermediate, 0.001, -0.5);
        let down_w = prand(14, hidden * intermediate, 0.0005, -0.15);
        let down_b = prand(15, hidden, 0.001, -0.5);
        let ffn_gamma = prand(16, hidden, 0.0001, 1.0);
        let ffn_beta = prand(17, hidden, 0.001, -0.5);

        let cpu_out = apply_bert_encoder_block_cpu_ref(
            &input,
            &q_w,
            Some(&q_b),
            &k_w,
            Some(&k_b),
            &v_w,
            Some(&v_b),
            &o_w,
            Some(&o_b),
            &attn_gamma,
            &attn_beta,
            &up_w,
            Some(&up_b),
            &down_w,
            Some(&down_b),
            &ffn_gamma,
            &ffn_beta,
            seq,
            hidden,
            num_heads,
            intermediate,
            eps,
        );

        // GPU side: upload tensors, run apply_bert_encoder_block_gpu.
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let in_buf = upload_f32(device, &input, vec![seq, hidden]);
        let q_w_b = upload_f32(device, &q_w, vec![hidden, hidden]);
        let q_b_b = upload_f32(device, &q_b, vec![hidden]);
        let k_w_b = upload_f32(device, &k_w, vec![hidden, hidden]);
        let k_b_b = upload_f32(device, &k_b, vec![hidden]);
        let v_w_b = upload_f32(device, &v_w, vec![hidden, hidden]);
        let v_b_b = upload_f32(device, &v_b, vec![hidden]);
        let o_w_b = upload_f32(device, &o_w, vec![hidden, hidden]);
        let o_b_b = upload_f32(device, &o_b, vec![hidden]);
        let attn_gamma_b = upload_f32(device, &attn_gamma, vec![hidden]);
        let attn_beta_b = upload_f32(device, &attn_beta, vec![hidden]);
        let up_w_b = upload_f32(device, &up_w, vec![intermediate, hidden]);
        let up_b_b = upload_f32(device, &up_b, vec![intermediate]);
        let down_w_b = upload_f32(device, &down_w, vec![hidden, intermediate]);
        let down_b_b = upload_f32(device, &down_b, vec![hidden]);
        let ffn_gamma_b = upload_f32(device, &ffn_gamma, vec![hidden]);
        let ffn_beta_b = upload_f32(device, &ffn_beta, vec![hidden]);

        let tensors = BertEncoderBlockTensors {
            q_w: &q_w_b,
            q_b: Some(&q_b_b),
            k_w: &k_w_b,
            k_b: Some(&k_b_b),
            v_w: &v_w_b,
            v_b: Some(&v_b_b),
            o_w: &o_w_b,
            o_b: Some(&o_b_b),
            attn_norm_gamma: &attn_gamma_b,
            attn_norm_beta: &attn_beta_b,
            up_w: &up_w_b,
            up_b: Some(&up_b_b),
            down_w: &down_w_b,
            down_b: Some(&down_b_b),
            ffn_norm_gamma: &ffn_gamma_b,
            ffn_norm_beta: &ffn_beta_b,
        };

        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let out_buf = apply_bert_encoder_block_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &in_buf,
            &tensors,
            None, // no padding mask in synthetic block test
            seq as u32,
            hidden as u32,
            num_heads as u32,
            intermediate as u32,
            eps,
        )
        .expect("encoder block dispatch");
        session.finish().expect("finish");
        let gpu_out = readback_f32(&out_buf, seq * hidden);

        let mut max_diff = 0.0f32;
        let mut argmax = 0usize;
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
                argmax = i;
            }
        }
        // Total noise envelope for the whole block: every linear and
        // attention cast contributes BF16 round-off, then post-LN
        // re-normalizes which bounds the absolute output range. Each
        // element of the post-LN output is bounded by `|γ| + |β|` so the
        // raw envelope stays small. With our γ ≈ 1, β ≈ ±0.5 inputs:
        // expect |out| < ~3, max_diff observable around the 0.05–0.20
        // range. Tolerance set permissively but tight enough to catch
        // a real correctness regression.
        assert!(
            max_diff < 0.50,
            "encoder-block max_diff {} > 0.50 at i={} (gpu={}, cpu={})",
            max_diff,
            argmax,
            gpu_out[argmax],
            cpu_out[argmax]
        );
    }

    // -----------------------------------------------------------------
    // bert_embed_gather_gpu — F32 row lookup
    // -----------------------------------------------------------------

    fn upload_u32(device: &MlxDevice, data: &[u32], shape: Vec<usize>) -> MlxBuffer {
        let bytes = data.len() * 4;
        let buf = device.alloc_buffer(bytes, DType::U32, shape).unwrap();
        // SAFETY: just-allocated u32 buffer; exclusive access.
        let slice: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut u32, data.len())
        };
        slice.copy_from_slice(data);
        buf
    }

    #[test]
    fn embed_gather_gpu_picks_correct_rows() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let vocab = 5usize;
        let hidden = 8usize;
        // table[i, h] = i * 100 + h
        let table: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i / hidden) * 100 + (i % hidden)) as f32)
            .collect();
        let ids: Vec<u32> = vec![3, 0, 4, 1, 2];

        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        // SAFETY: executor outlives session.
        let device: &MlxDevice = unsafe { &*device_ref };
        let table_buf = upload_f32(device, &table, vec![vocab, hidden]);
        let ids_buf = upload_u32(device, &ids, vec![ids.len()]);

        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let out = bert_embed_gather_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &table_buf,
            &ids_buf,
            vocab as u32,
            hidden as u32,
            ids.len() as u32,
        )
        .expect("dispatch");
        session.finish().expect("finish");
        let got = readback_f32(&out, ids.len() * hidden);
        for (i, &id) in ids.iter().enumerate() {
            for h in 0..hidden {
                let expected = (id as usize * 100 + h) as f32;
                let actual = got[i * hidden + h];
                assert!(
                    (expected - actual).abs() < 1e-6,
                    "row {} h {}: got {}, want {}",
                    i,
                    h,
                    actual,
                    expected
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // bert_embeddings_gpu — full token+pos+type+LN composition
    // -----------------------------------------------------------------

    /// CPU reference for the embeddings layer. Independent recomposition
    /// of the same primitives so a regression in any sub-step surfaces.
    fn embeddings_cpu_ref(
        input_ids: &[u32],
        type_ids: Option<&[u32]>,
        token_embd: &[f32],
        position_embd: &[f32],
        token_types: Option<&[f32]>,
        embed_gamma: &[f32],
        embed_beta: &[f32],
        eps: f32,
        seq_len: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let mut summed = vec![0.0f32; seq_len * hidden];
        for s in 0..seq_len {
            let tid = input_ids[s] as usize;
            for h in 0..hidden {
                let v = token_embd[tid * hidden + h] + position_embd[s * hidden + h];
                let v = if let (Some(tids), Some(tt)) = (type_ids, token_types) {
                    let typ = tids[s] as usize;
                    v + tt[typ * hidden + h]
                } else {
                    v
                };
                summed[s * hidden + h] = v;
            }
        }
        bert_layer_norm_cpu_ref(&summed, embed_gamma, embed_beta, eps, seq_len, hidden)
    }

    #[test]
    fn embeddings_gpu_matches_cpu_with_token_types() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let seq = 32usize; // satisfies LN reduction; below max_pos
        let hidden = 64usize;
        let vocab = 100usize;
        let max_pos = 128usize;
        let type_vocab = 2usize;
        let eps = 1e-12f32;

        let prand_f32 = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    ((i.wrapping_mul(2654435761usize).wrapping_add(seed) % 1000) as f32) * 0.001
                        - 0.5
                })
                .collect()
        };
        let token_embd = prand_f32(1, vocab * hidden);
        let position_embd = prand_f32(2, max_pos * hidden);
        let token_types = prand_f32(3, type_vocab * hidden);
        let embed_gamma = prand_f32(4, hidden);
        let embed_beta = prand_f32(5, hidden);
        let input_ids: Vec<u32> = (0..seq).map(|i| (i.wrapping_mul(7) % vocab) as u32).collect();
        let type_ids: Vec<u32> = (0..seq).map(|i| (i & 1) as u32).collect();

        let cpu = embeddings_cpu_ref(
            &input_ids,
            Some(&type_ids),
            &token_embd,
            &position_embd,
            Some(&token_types),
            &embed_gamma,
            &embed_beta,
            eps,
            seq,
            hidden,
        );

        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        let token_embd_b = upload_f32(device, &token_embd, vec![vocab, hidden]);
        let position_embd_b = upload_f32(device, &position_embd, vec![max_pos, hidden]);
        let token_types_b = upload_f32(device, &token_types, vec![type_vocab, hidden]);
        let embed_gamma_b = upload_f32(device, &embed_gamma, vec![hidden]);
        let embed_beta_b = upload_f32(device, &embed_beta, vec![hidden]);
        let input_ids_b = upload_u32(device, &input_ids, vec![seq]);
        let type_ids_b = upload_u32(device, &type_ids, vec![seq]);

        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let out = bert_embeddings_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input_ids_b,
            Some(&type_ids_b),
            &token_embd_b,
            &position_embd_b,
            Some(&token_types_b),
            &embed_gamma_b,
            &embed_beta_b,
            eps,
            seq as u32,
            hidden as u32,
            vocab as u32,
            max_pos as u32,
            type_vocab as u32,
        )
        .expect("embeddings dispatch");
        session.finish().expect("finish");
        let gpu = readback_f32(&out, seq * hidden);

        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        // Embeddings is gather + add + LayerNorm — all F32, no BF16
        // cast. Tolerance tight at 1e-4 (LN reduction order is the
        // only divergence source).
        assert!(max_diff < 1e-4, "embeddings max_diff: {max_diff}");
    }

    #[test]
    fn embeddings_gpu_without_token_types_path_works() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let seq = 32usize;
        let hidden = 64usize;
        let vocab = 50usize;
        let max_pos = 64usize;
        let eps = 1e-12f32;

        let token_embd: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32) * 0.001).collect();
        let position_embd: Vec<f32> = (0..max_pos * hidden).map(|i| (i as f32) * 0.0005).collect();
        let embed_gamma = vec![1.0f32; hidden];
        let embed_beta = vec![0.0f32; hidden];
        let input_ids: Vec<u32> = (0..seq).map(|i| (i % vocab) as u32).collect();

        let cpu = embeddings_cpu_ref(
            &input_ids,
            None,
            &token_embd,
            &position_embd,
            None,
            &embed_gamma,
            &embed_beta,
            eps,
            seq,
            hidden,
        );

        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        let token_embd_b = upload_f32(device, &token_embd, vec![vocab, hidden]);
        let position_embd_b = upload_f32(device, &position_embd, vec![max_pos, hidden]);
        let embed_gamma_b = upload_f32(device, &embed_gamma, vec![hidden]);
        let embed_beta_b = upload_f32(device, &embed_beta, vec![hidden]);
        let input_ids_b = upload_u32(device, &input_ids, vec![seq]);

        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let out = bert_embeddings_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input_ids_b,
            None,
            &token_embd_b,
            &position_embd_b,
            None,
            &embed_gamma_b,
            &embed_beta_b,
            eps,
            seq as u32,
            hidden as u32,
            vocab as u32,
            max_pos as u32,
            0,
        )
        .expect("embeddings dispatch (no token_types)");
        session.finish().expect("finish");
        let gpu = readback_f32(&out, seq * hidden);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        assert!(max_diff < 1e-4, "embeddings (no types) max_diff: {max_diff}");
    }

    #[test]
    fn embeddings_rejects_inconsistent_token_types_args() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        let buf = upload_f32(device, &[0.0; 64], vec![64]);
        let id_buf = upload_u32(device, &[0; 32], vec![32]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let err = bert_embeddings_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &id_buf,
            Some(&id_buf), // type_ids supplied
            &buf,
            &buf,
            None, // but token_types not — must reject
            &buf,
            &buf,
            1e-12,
            32,
            1,
            10,
            64,
            2,
        );
        assert!(err.is_err(), "type_ids without token_types must error");
        drop(session);
    }

    // -----------------------------------------------------------------
    // bert_pool_gpu — Mean / CLS / Last
    // -----------------------------------------------------------------

    #[test]
    fn pool_mean_gpu_matches_cpu_average() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let seq = 5usize;
        let hidden = 8usize;
        // input[s, h] = s + h * 0.1
        let input: Vec<f32> = (0..seq * hidden)
            .map(|i| (i / hidden) as f32 + (i % hidden) as f32 * 0.1)
            .collect();
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let in_buf = upload_f32(device, &input, vec![seq, hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let out = bert_pool_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &in_buf,
            BertPoolKind::Mean,
            seq as u32,
            hidden as u32,
        )
        .expect("dispatch");
        session.finish().expect("finish");
        let got = readback_f32(&out, hidden);
        for h in 0..hidden {
            let mut expected = 0.0f32;
            for s in 0..seq {
                expected += s as f32 + h as f32 * 0.1;
            }
            expected /= seq as f32;
            assert!(
                (got[h] - expected).abs() < 1e-5,
                "h={}: got {}, want {}",
                h,
                got[h],
                expected
            );
        }
    }

    #[test]
    fn pool_cls_gpu_returns_first_row() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let seq = 5usize;
        let hidden = 8usize;
        let input: Vec<f32> = (0..seq * hidden).map(|i| i as f32).collect();
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let in_buf = upload_f32(device, &input, vec![seq, hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let out = bert_pool_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &in_buf,
            BertPoolKind::Cls,
            seq as u32,
            hidden as u32,
        )
        .expect("dispatch");
        session.finish().expect("finish");
        let got = readback_f32(&out, hidden);
        for h in 0..hidden {
            assert!(
                (got[h] - h as f32).abs() < 1e-6,
                "CLS h={}: got {}, want {}",
                h,
                got[h],
                h
            );
        }
    }

    #[test]
    fn pool_last_gpu_returns_last_row() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let seq = 5usize;
        let hidden = 8usize;
        let input: Vec<f32> = (0..seq * hidden).map(|i| i as f32).collect();
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let in_buf = upload_f32(device, &input, vec![seq, hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let out = bert_pool_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &in_buf,
            BertPoolKind::Last,
            seq as u32,
            hidden as u32,
        )
        .expect("dispatch");
        session.finish().expect("finish");
        let got = readback_f32(&out, hidden);
        for h in 0..hidden {
            let expected = ((seq - 1) * hidden + h) as f32;
            assert!(
                (got[h] - expected).abs() < 1e-6,
                "Last h={}: got {}, want {}",
                h,
                got[h],
                expected
            );
        }
    }

    // -----------------------------------------------------------------
    // bert_l2_normalize_gpu — unit L2 norm
    // -----------------------------------------------------------------

    #[test]
    fn l2_normalize_gpu_produces_unit_norm() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let rows = 1usize;
        let dim = 384usize; // bge-small hidden
        let input: Vec<f32> = (0..rows * dim).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        let in_buf = upload_f32(device, &input, vec![rows, dim]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let out = bert_l2_normalize_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &in_buf,
            1e-12,
            rows as u32,
            dim as u32,
        )
        .expect("dispatch");
        session.finish().expect("finish");
        let got = readback_f32(&out, rows * dim);

        // Each row should have L2 norm 1.
        for r in 0..rows {
            let mut sum_sq = 0.0f64;
            for d in 0..dim {
                let v = got[r * dim + d] as f64;
                sum_sq += v * v;
            }
            let norm = sum_sq.sqrt() as f32;
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "row {} norm {} != 1.0",
                r,
                norm
            );
        }
    }

    // -----------------------------------------------------------------
    // apply_bert_full_forward_gpu — end-to-end on synthetic weights
    // -----------------------------------------------------------------

    #[test]
    fn full_forward_gpu_produces_unit_norm_output_at_minimal_config() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        // Minimal config that satisfies all kernel constraints:
        //   seq_len             = 32 (post-softmax matmul K floor)
        //   hidden              = 64, num_heads = 2 → head_dim = 32
        //   intermediate        = 128
        //   vocab               = 100
        //   max_pos             = 64 (≥ seq_len)
        //   type_vocab          = 2
        //   num_hidden_layers   = 2
        //   pooling             = Mean
        let seq = 32usize;
        let hidden = 64usize;
        let num_heads = 2usize;
        let intermediate = 128usize;
        let vocab = 100usize;
        let max_pos = 64usize;
        let type_vocab = 2usize;
        let num_layers = 2usize;
        let cfg = super::super::config::BertConfig {
            hidden_size: hidden,
            num_attention_heads: num_heads,
            num_hidden_layers: num_layers,
            intermediate_size: intermediate,
            max_position_embeddings: max_pos,
            vocab_size: vocab,
            type_vocab_size: type_vocab,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
            pooling_type: super::super::config::PoolingType::Mean,
            causal_attention: false,
        };

        let prand = |seed: usize, n: usize, scale: f32, offset: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    ((i.wrapping_mul(2654435761usize).wrapping_add(seed) % 1000) as f32) * scale
                        + offset
                })
                .collect()
        };

        // Build LoadedBertWeights from synthetic tensors. Each weight
        // / bias is uploaded to device once so it lives across the
        // full-forward dispatch.
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        // SAFETY: executor outlives session below.
        let device: &MlxDevice = unsafe { &*device_ref };

        let mut tensors: std::collections::HashMap<String, MlxBuffer> =
            std::collections::HashMap::new();

        // Stem.
        tensors.insert(
            super::super::TENSOR_TOKEN_EMBD.into(),
            upload_f32(
                device,
                &prand(1, vocab * hidden, 0.001, -0.5),
                vec![vocab, hidden],
            ),
        );
        tensors.insert(
            super::super::TENSOR_POS_EMBD.into(),
            upload_f32(
                device,
                &prand(2, max_pos * hidden, 0.001, -0.5),
                vec![max_pos, hidden],
            ),
        );
        tensors.insert(
            super::super::TENSOR_TOKEN_TYPES.into(),
            upload_f32(
                device,
                &prand(3, type_vocab * hidden, 0.001, -0.5),
                vec![type_vocab, hidden],
            ),
        );
        tensors.insert(
            super::super::TENSOR_EMBED_NORM_WEIGHT.into(),
            upload_f32(device, &prand(4, hidden, 0.0001, 1.0), vec![hidden]),
        );
        tensors.insert(
            super::super::TENSOR_EMBED_NORM_BIAS.into(),
            upload_f32(device, &prand(5, hidden, 0.001, -0.5), vec![hidden]),
        );

        // Per-layer.
        for layer in 0..num_layers {
            let key =
                |s: &str| -> String { super::super::config::bert_layer_tensor(layer, s) };
            // Q/K/V/O linear layers — `[hidden, hidden]` weights + `[hidden]` biases.
            let qkvo_seed_base = 100 + layer * 100;
            for (i, name) in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
            ]
            .iter()
            .enumerate()
            {
                tensors.insert(
                    key(name),
                    upload_f32(
                        device,
                        &prand(qkvo_seed_base + i * 10, hidden * hidden, 0.0005, -0.15),
                        vec![hidden, hidden],
                    ),
                );
            }
            for (i, name) in [
                "attn_q.bias",
                "attn_k.bias",
                "attn_v.bias",
                "attn_output.bias",
            ]
            .iter()
            .enumerate()
            {
                tensors.insert(
                    key(name),
                    upload_f32(
                        device,
                        &prand(qkvo_seed_base + 50 + i * 10, hidden, 0.001, -0.05),
                        vec![hidden],
                    ),
                );
            }
            // Post-attn LN.
            tensors.insert(
                key("attn_output_norm.weight"),
                upload_f32(device, &prand(qkvo_seed_base + 60, hidden, 0.0001, 1.0), vec![hidden]),
            );
            tensors.insert(
                key("attn_output_norm.bias"),
                upload_f32(device, &prand(qkvo_seed_base + 61, hidden, 0.001, -0.5), vec![hidden]),
            );
            // FFN up/down.
            tensors.insert(
                key("ffn_up.weight"),
                upload_f32(
                    device,
                    &prand(qkvo_seed_base + 70, intermediate * hidden, 0.0005, -0.15),
                    vec![intermediate, hidden],
                ),
            );
            tensors.insert(
                key("ffn_up.bias"),
                upload_f32(
                    device,
                    &prand(qkvo_seed_base + 71, intermediate, 0.001, -0.5),
                    vec![intermediate],
                ),
            );
            tensors.insert(
                key("ffn_down.weight"),
                upload_f32(
                    device,
                    &prand(qkvo_seed_base + 72, hidden * intermediate, 0.0005, -0.15),
                    vec![hidden, intermediate],
                ),
            );
            tensors.insert(
                key("ffn_down.bias"),
                upload_f32(
                    device,
                    &prand(qkvo_seed_base + 73, hidden, 0.001, -0.5),
                    vec![hidden],
                ),
            );
            // Post-FFN LN.
            tensors.insert(
                key("layer_output_norm.weight"),
                upload_f32(device, &prand(qkvo_seed_base + 80, hidden, 0.0001, 1.0), vec![hidden]),
            );
            tensors.insert(
                key("layer_output_norm.bias"),
                upload_f32(device, &prand(qkvo_seed_base + 81, hidden, 0.001, -0.5), vec![hidden]),
            );
        }

        let weights = super::super::weights::LoadedBertWeights::from_tensors_for_test(
            tensors,
            MlxDevice::new().unwrap(),
        );

        // Synthetic input: arbitrary token ids modulo vocab.
        let input_ids: Vec<u32> = (0..seq).map(|i| (i.wrapping_mul(7) % vocab) as u32).collect();
        let type_ids: Vec<u32> = (0..seq).map(|i| (i & 1) as u32).collect();
        let input_ids_b = upload_u32(device, &input_ids, vec![seq]);
        let type_ids_b = upload_u32(device, &type_ids, vec![seq]);

        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let out = apply_bert_full_forward_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input_ids_b,
            Some(&type_ids_b),
            &weights,
            &cfg,
            seq as u32,
            seq as u32, // valid_token_count == seq_len → no mask
        )
        .expect("full forward dispatch");
        session.finish().expect("finish");
        let got = readback_f32(&out, hidden);

        // Two invariants the full forward must satisfy:
        // (1) every output element is finite (no NaN/Inf from the
        //     attention or LayerNorm chain).
        // (2) the output is L2-normalized (||y||₂ = 1 within F32 tol).
        for (i, &v) in got.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] not finite: {v}");
        }
        let mut sum_sq = 0.0f64;
        for &v in &got {
            sum_sq += (v as f64) * (v as f64);
        }
        let norm = sum_sq.sqrt() as f32;
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "full-forward output not unit L2 norm: {} (expected 1.0)",
            norm
        );
    }

    #[test]
    fn full_forward_rejects_pooling_type_none() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        // Minimal cfg with pooling_type = None — must error.
        let cfg = super::super::config::BertConfig {
            hidden_size: 64,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 128,
            max_position_embeddings: 64,
            vocab_size: 100,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
            pooling_type: super::super::config::PoolingType::None,
            causal_attention: false,
        };
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let weights = super::super::weights::LoadedBertWeights::empty(
            MlxDevice::new().unwrap(),
        );
        let input_ids = upload_u32(device, &[0u32; 32], vec![32]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let err = apply_bert_full_forward_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input_ids,
            None,
            &weights,
            &cfg,
            32,
            32,
        );
        assert!(err.is_err(), "pooling_type=None must error");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("pooling_type=None"),
            "error must name pooling_type=None: {msg}"
        );
        drop(session);
    }

    #[test]
    fn full_forward_rejects_seq_len_below_floor() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        let cfg = super::super::config::BertConfig {
            hidden_size: 64,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 128,
            max_position_embeddings: 64,
            vocab_size: 100,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
            pooling_type: super::super::config::PoolingType::Mean,
            causal_attention: false,
        };
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let weights = super::super::weights::LoadedBertWeights::empty(
            MlxDevice::new().unwrap(),
        );
        let input_ids = upload_u32(device, &[0u32; 16], vec![16]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let err = apply_bert_full_forward_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input_ids,
            None,
            &weights,
            &cfg,
            16, // < 32 floor
            16,
        );
        assert!(err.is_err(), "seq_len < 32 must error");
        drop(session);
    }

    #[test]
    fn encoder_block_rejects_hidden_not_divisible_by_num_heads() {
        if MlxDevice::new().is_err() {
            eprintln!("skipping: no Metal device available");
            return;
        }
        // hidden=65, num_heads=2 → 65 % 2 == 1 → must reject.
        let device = MlxDevice::new().unwrap();
        let executor = GraphExecutor::new(device);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let buf = upload_f32(device, &[0.0; 32 * 65], vec![32, 65]);
        let small = upload_f32(device, &[0.0; 65], vec![65]);
        let small_w = upload_f32(device, &[0.0; 65 * 65], vec![65, 65]);
        let tensors = BertEncoderBlockTensors {
            q_w: &small_w,
            q_b: None,
            k_w: &small_w,
            k_b: None,
            v_w: &small_w,
            v_b: None,
            o_w: &small_w,
            o_b: None,
            attn_norm_gamma: &small,
            attn_norm_beta: &small,
            up_w: &small_w,
            up_b: None,
            down_w: &small_w,
            down_b: None,
            ffn_norm_gamma: &small,
            ffn_norm_beta: &small,
        };
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_bert_custom_shaders(&mut registry);
        let err = apply_bert_encoder_block_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &buf,
            &tensors,
            None,
            32,
            65, // hidden
            2,  // num_heads → 65 % 2 != 0
            128,
            1e-12,
        );
        assert!(err.is_err(), "hidden % num_heads != 0 must error");
        drop(session);
    }
}
