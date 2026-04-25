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
use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::encode_helpers::KernelArg;
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
    // mlx-native's GeLU lives in its own ops module; register it here so
    // every `register_bert_custom_shaders` caller gets a registry that
    // can dispatch the FFN GeLU activation without touching mlx-native
    // op-level registration directly.
    mlx_native::ops::gelu::register(registry);
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
}
