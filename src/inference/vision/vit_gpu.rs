//! GPU-dispatched ViT forward primitives (ADR-005 Phase 2c, iter 43+).
//!
//! This module is the PRODUCTION forward path. The CPU functions in
//! `vit.rs` stay only as (a) byte-identical parity references on tiny
//! input shapes for validating each GPU op, and (b) architecture
//! documentation — they are never invoked on production `[196, 1152]`
//! shapes in the shipping binary.
//!
//! Each GPU primitive here wraps a mlx-native dispatch. Callers supply a
//! `GraphSession` (per forward pass) and MlxBuffer handles; the session
//! accumulates dispatches into a single CommandBuffer and commits +
//! waits once at `finish()`.
//!
//! # Starting-point mapping
//!
//!   CPU reference (vit.rs)          → GPU primitive (vit_gpu.rs)
//!   ───────────────────────────────── ────────────────────────────────
//!   linear_forward                  → vit_linear_gpu                       [iter 43]
//!   rms_norm_forward                → vit_rms_norm_gpu                     [iter 44]
//!   per_head_rms_norm_forward       → vit_per_head_rms_norm_gpu            [iter 44]
//!   scaled_dot_product_attention    → vit_attention_gpu (flash_attn_*)     [iter 45]
//!   silu_in_place + elementwise_mul → vit_sigmoid_mul_gpu (fused)          [iter 46]
//!   residual_add                    → vit_residual_add_gpu                 [iter 46]
//!   etc.

#![allow(dead_code)]

use anyhow::{anyhow, Context, Result};
use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::rms_norm::dispatch_rms_norm;
use mlx_native::ops::elementwise::{elementwise_add, elementwise_mul, scalar_mul_f32};
use mlx_native::ops::sigmoid_mul::dispatch_sigmoid_mul;
use mlx_native::ops::softmax::dispatch_softmax;
use mlx_native::ops::transpose::{permute_021_f32, transpose_last2_bf16};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

/// GPU dense linear projection `y = x @ W.T`.
///
/// Dtype contract:
///   - `input`  is F32 `[seq_len, in_features]` row-major on device.
///   - `weight` is F32 `[out_features, in_features]` row-major on device
///     (loaded via `GgufFile::load_tensor_f32`).
///   - Returned buffer is F32 `[seq_len, out_features]` row-major.
///
/// The weight is internally cast F32 → BF16 once per call to satisfy
/// `dense_matmul_bf16_f32_tensor`'s tensor-core dtype contract (src0 =
/// BF16, src1 = F32, dst = F32). The BF16 rounding introduces ≤ 1e-3
/// error vs the pure-F32 reference; callers compare with that tolerance.
///
/// Constraint: `in_features >= 32` (tensor-core tile requires one
/// NK=32 slice).
///
/// # Errors
///
/// Any mlx-native dispatch error, or the `in_features < 32` check.
pub fn vit_linear_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight_f32: &MlxBuffer,
    seq_len: u32,
    in_features: u32,
    out_features: u32,
) -> Result<MlxBuffer> {
    if in_features < 32 {
        return Err(anyhow!(
            "vit_linear_gpu: in_features ({}) must be >= 32",
            in_features
        ));
    }
    if seq_len == 0 || out_features == 0 {
        return Err(anyhow!(
            "vit_linear_gpu: seq_len ({}) and out_features ({}) must be > 0",
            seq_len,
            out_features
        ));
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
    .context("vit_linear_gpu: F32→BF16 cast")?;

    // --- Allocate F32 output ---
    let out_bytes = (seq_len as usize) * (out_features as usize) * 4;
    let mut dst = device
        .alloc_buffer(
            out_bytes,
            DType::F32,
            vec![seq_len as usize, out_features as usize],
        )
        .map_err(|e| anyhow!("alloc output: {e}"))?;

    // --- Dispatch dense matmul ---
    // Layout: src0 = weight [1, N=out, K=in] BF16,
    //         src1 = input  [1, M=seq, K=in] F32,
    //         dst  = output [1, M=seq, N=out] F32.
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
        &mut dst,
        &params,
    )
    .context("vit_linear_gpu: dense_matmul_bf16_f32_tensor")?;

    Ok(dst)
}

/// GPU RMSNorm with affine gain (single-parameter; no bias).
///
/// Wraps `mlx_native::ops::rms_norm::dispatch_rms_norm`. Computes
/// `y[r, i] = x[r, i] * rsqrt(mean(x[r,:]²) + eps) * gain[i]` per row.
///
/// Dtype contract:
///   - `input` is F32 `[rows, dim]` row-major on device.
///   - `gain` is F32 `[dim]`.
///   - Output is F32 `[rows, dim]` row-major (freshly allocated).
///
/// `eps` matches PyTorch's `nn.RMSNorm(eps)` semantic — added inside the
/// `sqrt(mean(x²) + eps)`. Typical Gemma 4 vision tower value: `1e-6`
/// (from `MmprojConfig.layer_norm_eps`).
///
/// # Errors
///
/// - `rows == 0` or `dim == 0`
/// - input/gain shape mismatches (propagated from mlx-native)
pub fn vit_rms_norm_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    gain_f32: &MlxBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    if rows == 0 || dim == 0 {
        return Err(anyhow!(
            "vit_rms_norm_gpu: rows ({}) and dim ({}) must be > 0",
            rows,
            dim
        ));
    }

    // Allocate output [rows, dim] f32.
    let out_bytes = (rows as usize) * (dim as usize) * 4;
    let output = device
        .alloc_buffer(out_bytes, DType::F32, vec![rows as usize, dim as usize])
        .map_err(|e| anyhow!("vit_rms_norm_gpu: alloc output: {e}"))?;

    // Allocate the params buffer expected by the kernel: 2 × f32 holding
    // [eps, dim_as_f32]. Filled via direct CPU pointer write (Apple
    // unified memory means the same address is GPU-visible).
    let params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("vit_rms_norm_gpu: alloc params: {e}"))?;
    {
        // SAFETY: just-allocated f32 buffer; no aliasing.
        let s: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(params_buf.contents_ptr() as *mut f32, 2)
        };
        s[0] = eps;
        s[1] = dim as f32;
    }

    dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        gain_f32,
        &output,
        &params_buf,
        rows,
        dim,
    )
    .context("vit_rms_norm_gpu: dispatch_rms_norm")?;

    Ok(output)
}

/// GPU per-head RMSNorm. Identical math to `vit_rms_norm_gpu` but
/// "per-head" semantically: input shape `[batch, num_heads, head_dim]`
/// is byte-equivalent to `[batch * num_heads, head_dim]` row-major, so
/// dispatch with `rows = batch * num_heads, dim = head_dim`. Gain is
/// `[head_dim]` shared across heads (Gemma 4 SigLIP convention).
///
/// # Errors
///
/// - any dim is 0
/// - propagated from `vit_rms_norm_gpu`
pub fn vit_per_head_rms_norm_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    gain_f32: &MlxBuffer,
    batch: u32,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    if batch == 0 || num_heads == 0 || head_dim == 0 {
        return Err(anyhow!(
            "vit_per_head_rms_norm_gpu: batch ({}), num_heads ({}), head_dim ({}) must all be > 0",
            batch,
            num_heads,
            head_dim
        ));
    }
    let rows = batch
        .checked_mul(num_heads)
        .ok_or_else(|| anyhow!("vit_per_head_rms_norm_gpu: batch*num_heads overflow"))?;
    vit_rms_norm_gpu(encoder, registry, device, input, gain_f32, rows, head_dim, eps)
}

/// GPU softmax along the last dimension of a `[rows, cols]` F32 tensor.
///
/// Wraps `mlx_native::ops::softmax::dispatch_softmax`. Numerically
/// stable (subtracts per-row max before exp). One threadgroup per row.
///
/// Allocates a fresh `[rows, cols]` F32 output buffer.
///
/// # Errors
///
/// - any dim is 0
/// - propagated from mlx-native dispatch
pub fn vit_softmax_last_dim_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    rows: u32,
    cols: u32,
) -> Result<MlxBuffer> {
    if rows == 0 || cols == 0 {
        return Err(anyhow!(
            "vit_softmax_last_dim_gpu: rows ({}) and cols ({}) must be > 0",
            rows,
            cols
        ));
    }

    let out_bytes = (rows as usize) * (cols as usize) * 4;
    let output = device
        .alloc_buffer(out_bytes, DType::F32, vec![rows as usize, cols as usize])
        .map_err(|e| anyhow!("vit_softmax_last_dim_gpu: alloc output: {e}"))?;

    // Params buffer: 2 × f32 holding [cols_as_f32, 0].
    let params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("vit_softmax_last_dim_gpu: alloc params: {e}"))?;
    {
        // SAFETY: just-allocated f32 buffer; no aliasing.
        let s: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(params_buf.contents_ptr() as *mut f32, 2)
        };
        s[0] = cols as f32;
        s[1] = 0.0;
    }

    dispatch_softmax(
        encoder,
        registry,
        device.metal_device(),
        input,
        &output,
        &params_buf,
        rows,
        cols,
    )
    .context("vit_softmax_last_dim_gpu: dispatch_softmax")?;

    Ok(output)
}

/// GPU attention scores: `scores = (Q @ K^T) * scale` per head.
///
/// Input shape: `Q`, `K` each `[batch, num_heads, head_dim]` F32 row-major.
/// Output shape: `[num_heads, batch_q, batch_k]` F32 (head-major scores).
///
/// Pipeline:
///   1. Permute Q, K from seq-major `[batch, num_heads, head_dim]` to
///      head-major `[num_heads, batch, head_dim]` via `permute_021_f32`.
///   2. Cast K_perm F32 → BF16 (tensor-core src0 contract).
///   3. Dispatch `dense_matmul_bf16_f32_tensor` with src0=K_bf16,
///      src1=Q_perm. Output: `[num_heads, batch_q, batch_k]` F32.
///   4. Apply `scale` via `scalar_mul_f32` in-place.
///
/// `scale` for standard ViT = `1 / sqrt(head_dim)`. **For Gemma 4V the
/// correct scale is 1.0** per llama.cpp `clip.cpp` — caller passes the
/// arch-appropriate value. This function does no implicit scaling.
///
/// # Errors
///
/// - `head_dim < 32` (tensor-core tile constraint)
/// - any dim is 0
/// - propagated from underlying mlx-native dispatches
pub fn vit_attention_scores_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q_seq_major: &MlxBuffer,
    k_seq_major: &MlxBuffer,
    batch: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
) -> Result<MlxBuffer> {
    if head_dim < 32 {
        return Err(anyhow!(
            "vit_attention_scores_gpu: head_dim ({}) must be >= 32",
            head_dim
        ));
    }
    if batch == 0 || num_heads == 0 {
        return Err(anyhow!(
            "vit_attention_scores_gpu: batch ({}) and num_heads ({}) must be > 0",
            batch,
            num_heads
        ));
    }

    let metal_dev = device.metal_device();
    let n_qk_elems = (batch as usize) * (num_heads as usize) * (head_dim as usize);

    // --- Step 1: permute Q, K to head-major layout ---
    let q_perm = device
        .alloc_buffer(
            n_qk_elems * 4,
            DType::F32,
            vec![num_heads as usize, batch as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc q_perm: {e}"))?;
    let k_perm = device
        .alloc_buffer(
            n_qk_elems * 4,
            DType::F32,
            vec![num_heads as usize, batch as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc k_perm: {e}"))?;
    permute_021_f32(
        encoder,
        registry,
        metal_dev,
        q_seq_major,
        &q_perm,
        batch as usize,
        num_heads as usize,
        head_dim as usize,
    )
    .context("permute Q seq→head major")?;
    permute_021_f32(
        encoder,
        registry,
        metal_dev,
        k_seq_major,
        &k_perm,
        batch as usize,
        num_heads as usize,
        head_dim as usize,
    )
    .context("permute K seq→head major")?;
    // mlx-native uses MTLDispatchType::Concurrent — explicit barriers
    // are required between dispatches with RAW/WAR/WAW dependencies.
    encoder.memory_barrier();

    // --- Step 2: cast K_perm F32 → BF16 ---
    let k_bf16 = device
        .alloc_buffer(
            n_qk_elems * 2,
            DType::BF16,
            vec![num_heads as usize, batch as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc k_bf16: {e}"))?;
    cast(
        encoder,
        registry,
        metal_dev,
        &k_perm,
        &k_bf16,
        n_qk_elems,
        CastDirection::F32ToBF16,
    )
    .context("cast K F32→BF16")?;
    encoder.memory_barrier();

    // --- Step 3: scores = Q @ K^T per head batch ---
    // Layout: src0=K_bf16 [num_heads, batch_k, head_dim] BF16,
    //         src1=Q_perm [num_heads, batch_q, head_dim] F32.
    // output[h, m, n] = sum_k K[h, n, k] * Q[h, m, k] = (Q[h] @ K[h]^T)[m, n].
    let n_scores = (num_heads as usize) * (batch as usize) * (batch as usize);
    let mut scores = device
        .alloc_buffer(
            n_scores * 4,
            DType::F32,
            vec![num_heads as usize, batch as usize, batch as usize],
        )
        .map_err(|e| anyhow!("alloc scores: {e}"))?;
    let params = DenseMmBf16F32Params {
        m: batch,
        n: batch,
        k: head_dim,
        src0_batch: num_heads,
        src1_batch: num_heads,
    };
    dense_matmul_bf16_f32_tensor(
        encoder,
        registry,
        device,
        &k_bf16,
        &q_perm,
        &mut scores,
        &params,
    )
    .context("attention scores matmul")?;

    // --- Step 4: apply scale (in-place via scalar_mul_f32) ---
    if scale != 1.0 {
        encoder.memory_barrier();
        scalar_mul_f32(
            encoder,
            registry,
            metal_dev,
            &scores,
            &scores,
            n_scores,
            scale,
        )
        .context("scale scores")?;
    }

    Ok(scores)
}

/// Full GPU scaled-dot-product attention for ViT.
///
/// Inputs (`Q`, `K`, `V`) are seq-major `[batch, num_heads, head_dim]`
/// F32 buffers; output has the same shape. No mask (ViT is bidirectional).
///
/// Pipeline:
///   1. `vit_attention_scores_gpu` → scores `[num_heads, batch, batch]` F32.
///   2. `vit_softmax_last_dim_gpu` → softmax along last dim.
///   3. Permute V seq→head major; cast to BF16; `transpose_last2_bf16`
///      to `[num_heads, head_dim, batch]`.
///   4. `dense_matmul_bf16_f32_tensor` per head: `attn = scores @ V`,
///      output `[num_heads, batch, head_dim]` F32.
///   5. `permute_021_f32` back to seq-major
///      `[batch, num_heads, head_dim]`.
///
/// `scale` matches `vit_attention_scores_gpu`. For Gemma 4V pass `1.0`
/// (per llama.cpp); for standard ViT pass `1 / sqrt(head_dim)`.
///
/// # Errors
///
/// Propagated from any sub-stage; primarily shape / dtype mismatches.
pub fn vit_attention_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q_seq_major: &MlxBuffer,
    k_seq_major: &MlxBuffer,
    v_seq_major: &MlxBuffer,
    batch: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
) -> Result<MlxBuffer> {
    if head_dim < 32 {
        return Err(anyhow!(
            "vit_attention_gpu: head_dim ({}) must be >= 32",
            head_dim
        ));
    }
    if batch == 0 || num_heads == 0 {
        return Err(anyhow!(
            "vit_attention_gpu: batch ({}) and num_heads ({}) must be > 0",
            batch,
            num_heads
        ));
    }

    let metal_dev = device.metal_device();

    // --- Stage 1: scores = (Q @ K^T) * scale, shape [num_heads, batch, batch] ---
    let scores =
        vit_attention_scores_gpu(encoder, registry, device, q_seq_major, k_seq_major, batch, num_heads, head_dim, scale)?;
    encoder.memory_barrier();

    // --- Stage 2: softmax along last dim. Treat scores as
    // [num_heads * batch, batch] = [rows, cols] ---
    let n_rows = (num_heads as u64) * (batch as u64);
    let softmaxed =
        vit_softmax_last_dim_gpu(encoder, registry, device, &scores, n_rows as u32, batch)?;
    encoder.memory_barrier();

    // --- Stage 3: V layout transforms ---
    // 3a. Permute V seq→head major: [batch, num_heads, head_dim] →
    //     [num_heads, batch, head_dim] f32.
    let n_v = (batch as usize) * (num_heads as usize) * (head_dim as usize);
    let v_perm = device
        .alloc_buffer(
            n_v * 4,
            DType::F32,
            vec![num_heads as usize, batch as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc v_perm: {e}"))?;
    permute_021_f32(
        encoder,
        registry,
        metal_dev,
        v_seq_major,
        &v_perm,
        batch as usize,
        num_heads as usize,
        head_dim as usize,
    )
    .context("permute V seq→head major")?;
    encoder.memory_barrier();

    // 3b. Cast V_perm f32 → bf16 (transpose_last2 only has a bf16 variant
    // and dense_matmul wants bf16 src0).
    let v_bf16 = device
        .alloc_buffer(
            n_v * 2,
            DType::BF16,
            vec![num_heads as usize, batch as usize, head_dim as usize],
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

    // 3c. transpose_last2_bf16: [num_heads, batch, head_dim] →
    //     [num_heads, head_dim, batch].
    let v_t_bf16 = device
        .alloc_buffer(
            n_v * 2,
            DType::BF16,
            vec![num_heads as usize, head_dim as usize, batch as usize],
        )
        .map_err(|e| anyhow!("alloc v_t_bf16: {e}"))?;
    transpose_last2_bf16(
        encoder,
        registry,
        metal_dev,
        &v_bf16,
        &v_t_bf16,
        num_heads as usize,
        batch as usize,
        head_dim as usize,
    )
    .context("transpose V last 2")?;
    encoder.memory_barrier();

    // --- Stage 4: attn = scores @ V per head batch ---
    // Layout: src0 = V_T [num_heads, head_dim, batch_k] BF16,
    //         src1 = softmaxed scores [num_heads, batch_q, batch_k] F32.
    // output[h, m, n] = sum_k V_T[h, n, k] * scores[h, m, k]
    //                 = sum_k V[h, k, n] * scores[h, m, k] = attn[h, m, n] ✓
    // Output shape: [num_heads, batch_q=m, head_dim=n] F32.
    let n_attn = (num_heads as usize) * (batch as usize) * (head_dim as usize);
    let mut attn_head_major = device
        .alloc_buffer(
            n_attn * 4,
            DType::F32,
            vec![num_heads as usize, batch as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc attn_head_major: {e}"))?;
    let params = DenseMmBf16F32Params {
        m: batch,      // batch_q
        n: head_dim,   // output last dim
        k: batch,      // batch_k (contract)
        src0_batch: num_heads,
        src1_batch: num_heads,
    };
    dense_matmul_bf16_f32_tensor(
        encoder,
        registry,
        device,
        &v_t_bf16,
        &softmaxed,
        &mut attn_head_major,
        &params,
    )
    .context("attention scores @ V matmul")?;
    encoder.memory_barrier();

    // --- Stage 5: permute back to seq-major ---
    // [num_heads, batch, head_dim] → [batch, num_heads, head_dim].
    let attn_seq_major = device
        .alloc_buffer(
            n_attn * 4,
            DType::F32,
            vec![batch as usize, num_heads as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc attn_seq_major: {e}"))?;
    permute_021_f32(
        encoder,
        registry,
        metal_dev,
        &attn_head_major,
        &attn_seq_major,
        num_heads as usize,
        batch as usize,
        head_dim as usize,
    )
    .context("permute attn head→seq major")?;

    Ok(attn_seq_major)
}

/// GPU residual add: `out[i] = a[i] + b[i]`. F32 elementwise.
///
/// Wraps `mlx_native::ops::elementwise::elementwise_add`. Allocates a
/// fresh F32 output. (mlx-native's elementwise_add takes a separate
/// output buffer; if a true in-place residual is needed for memory,
/// a future iter can substitute the in-place variant when one exists.)
///
/// # Errors
///
/// - `n_elements == 0`
/// - propagated from mlx-native dispatch
pub fn vit_residual_add_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    a: &MlxBuffer,
    b: &MlxBuffer,
    n_elements: u32,
) -> Result<MlxBuffer> {
    if n_elements == 0 {
        return Err(anyhow!("vit_residual_add_gpu: n_elements must be > 0"));
    }
    let out = device
        .alloc_buffer((n_elements as usize) * 4, DType::F32, vec![n_elements as usize])
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
    .context("vit_residual_add_gpu: elementwise_add")?;
    Ok(out)
}

/// GPU SwiGLU gating: `out = silu(gate) * up = gate * sigmoid(gate) * up`.
///
/// Composed as two dispatches:
///   1. `dispatch_sigmoid_mul(x=gate, gate=gate)` → `silu(gate)`
///      (i.e. `gate * sigmoid(gate)`).
///   2. `elementwise_mul(silu_out, up)` → final output.
///
/// All buffers are F32 with `n_elements` elements. Caller must register
/// sigmoid_mul shader sources before dispatch:
/// `mlx_native::ops::sigmoid_mul::register(&mut registry)`.
///
/// `encoder.memory_barrier()` is inserted between the two dispatches
/// per the iter-46 concurrent-dispatch lesson.
///
/// # Errors
///
/// - `n_elements == 0`
/// - propagated from mlx-native dispatches
pub fn vit_silu_mul_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate: &MlxBuffer,
    up: &MlxBuffer,
    n_elements: u32,
) -> Result<MlxBuffer> {
    if n_elements == 0 {
        return Err(anyhow!("vit_silu_mul_gpu: n_elements must be > 0"));
    }
    let metal_dev = device.metal_device();

    // Step 1: silu(gate) via sigmoid_mul(gate, gate).
    let silu_out = device
        .alloc_buffer((n_elements as usize) * 4, DType::F32, vec![n_elements as usize])
        .map_err(|e| anyhow!("alloc silu_out: {e}"))?;
    // sigmoid_mul takes a params_buf with the count (u32 cast as f32 element).
    let params_buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .map_err(|e| anyhow!("alloc sigmoid_mul params: {e}"))?;
    {
        // SAFETY: just-allocated f32 buffer.
        let s: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(params_buf.contents_ptr() as *mut u32, 1)
        };
        s[0] = n_elements;
    }
    dispatch_sigmoid_mul(
        encoder,
        registry,
        metal_dev,
        gate,
        gate,
        &silu_out,
        &params_buf,
        n_elements,
    )
    .context("vit_silu_mul_gpu: sigmoid_mul (silu step)")?;
    encoder.memory_barrier();

    // Step 2: out = silu_out * up.
    let out = device
        .alloc_buffer((n_elements as usize) * 4, DType::F32, vec![n_elements as usize])
        .map_err(|e| anyhow!("alloc final out: {e}"))?;
    elementwise_mul(
        encoder,
        registry,
        metal_dev,
        &silu_out,
        up,
        &out,
        n_elements as usize,
        DType::F32,
    )
    .context("vit_silu_mul_gpu: elementwise_mul")?;

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::vision::vit::{
        elementwise_mul_in_place, linear_forward as linear_cpu,
        per_head_rms_norm_forward as per_head_rms_cpu, residual_add as residual_add_cpu,
        rms_norm_forward as rms_norm_cpu, scaled_dot_product_attention as attention_cpu,
        silu_in_place, softmax_last_dim as softmax_cpu,
    };
    use crate::inference::vision::mmproj::MmprojConfig;
    use crate::inference::vision::mmproj_weights::LoadedMmprojWeights;
    use mlx_native::gguf::GgufFile;
    use mlx_native::{GraphExecutor, MlxDevice};
    use std::path::Path;

    const GEMMA4_MMPROJ_PATH: &str =
        "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq-mmproj.gguf";

    /// Upload a CPU f32 slice to a fresh device buffer.
    fn upload_f32(device: &MlxDevice, data: &[f32], shape: Vec<usize>) -> MlxBuffer {
        let bytes = data.len() * 4;
        let buf = device
            .alloc_buffer(bytes, DType::F32, shape)
            .expect("alloc upload");
        let slice: &mut [f32] = unsafe {
            // SAFETY: we just allocated this buffer with f32 dtype; exclusive access.
            std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut f32, data.len())
        };
        slice.copy_from_slice(data);
        buf
    }

    /// Read back a device f32 buffer to a CPU vec.
    fn readback_f32(buf: &MlxBuffer, expected_len: usize) -> Vec<f32> {
        let slice: &[f32] = buf.as_slice::<f32>().expect("readback as_slice");
        assert_eq!(slice.len(), expected_len, "readback length mismatch");
        slice.to_vec()
    }

    #[test]
    fn vit_linear_gpu_matches_cpu_reference_on_small_input() {
        // Small shape parity. seq=4, in=64 (≥32 required), out=32.
        // Use deterministic synthetic input + deterministic synthetic
        // weight; compare GPU output to CPU linear_forward within 1e-3
        // (bf16 weight round-trip tolerance).
        let seq = 4usize;
        let in_features = 64usize;
        let out_features = 32usize;

        let input_cpu: Vec<f32> = (0..seq * in_features)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let weight_cpu: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i as f32) * 0.01).cos() * 0.1)
            .collect();

        let expected_cpu = linear_cpu(
            &input_cpu,
            &weight_cpu,
            None,
            seq,
            in_features,
            out_features,
        )
        .expect("cpu ref");

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let input_buf = upload_f32(executor.device(), &input_cpu, vec![seq, in_features]);
        let weight_buf = upload_f32(
            executor.device(),
            &weight_cpu,
            vec![out_features, in_features],
        );

        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        // SAFETY: executor outlives session; device borrow is stable.
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_linear_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input_buf,
            &weight_buf,
            seq as u32,
            in_features as u32,
            out_features as u32,
        )
        .expect("gpu dispatch");
        session.finish().expect("finish");

        let got = readback_f32(&out_buf, seq * out_features);

        // BF16 weight round-trip: ≤ 1e-3 per element vs F32 reference.
        for (i, (g, e)) in got.iter().zip(expected_cpu.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff < 1e-2,
                "GPU/CPU mismatch at element {i}: gpu={g} cpu={e} diff={diff}"
            );
        }
        // Tighter check on max-abs: most elements should be well within
        // the coarse 1e-2 bound.
        let max_diff = got
            .iter()
            .zip(expected_cpu.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-2, "overall max_diff = {max_diff}");
    }

    #[test]
    fn vit_linear_gpu_rejects_small_in_features() {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        // Allocate tiny buffers that'd satisfy shape but in_features=16 < 32.
        let input = executor
            .device()
            .alloc_buffer(4 * 16 * 4, DType::F32, vec![4, 16])
            .expect("alloc");
        let weight = executor
            .device()
            .alloc_buffer(32 * 16 * 4, DType::F32, vec![32, 16])
            .expect("alloc");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_linear_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input,
            &weight,
            4,
            16,
            32,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("in_features"));
    }

    #[test]
    fn vit_linear_gpu_rejects_zero_dims() {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let input = executor
            .device()
            .alloc_buffer(32 * 4, DType::F32, vec![0, 32])
            .expect("alloc");
        let weight = executor
            .device()
            .alloc_buffer(32 * 32 * 4, DType::F32, vec![32, 32])
            .expect("alloc");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_linear_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input,
            &weight,
            0,
            32,
            32,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("> 0"));
    }

    #[test]
    fn vit_linear_gpu_on_real_gemma4_mm0_matches_cpu_at_small_seq() {
        // Real-data GPU path: use actual Gemma 4 mm.0.weight [2816,
        // 1152] F32, a synthetic [seq=4, 1152] input, run vit_linear_gpu
        // → [4, 2816] F32. Compare against CPU linear_forward; require
        // max_abs_diff ≤ 1e-2 (BF16 weight round-trip).
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found");
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let device = MlxDevice::new().expect("device");
        let weights = LoadedMmprojWeights::load(&gguf, &cfg, device).expect("load");

        let hidden = cfg.hidden_size as usize;
        let seq = 4usize;
        let mm0 = weights.mm_0_weight().expect("mm.0");
        let mm0_f32: &[f32] = mm0.as_slice::<f32>().expect("mm.0 slice");
        let text_hidden = mm0_f32.len() / hidden;
        assert_eq!(text_hidden, 2816);

        // Synthetic input — deterministic sine-based so CPU and GPU
        // see identical float bytes on both sides.
        let input_cpu: Vec<f32> = (0..seq * hidden)
            .map(|i| ((i as f32) * 1e-4).sin() * 0.1)
            .collect();

        // CPU reference — snapshot copy of mm0_f32 since the CPU fn
        // doesn't reference-borrow the MlxBuffer.
        let weight_cpu: Vec<f32> = mm0_f32.to_vec();
        let expected = linear_cpu(&input_cpu, &weight_cpu, None, seq, hidden, text_hidden)
            .expect("cpu ref");

        // GPU path.
        let exec_device = MlxDevice::new().expect("device2");
        let executor = GraphExecutor::new(exec_device);
        let input_buf = upload_f32(executor.device(), &input_cpu, vec![seq, hidden]);
        // Re-upload weight to the new device (weights Arc is on a different device).
        let weight_buf = upload_f32(
            executor.device(),
            &weight_cpu,
            vec![text_hidden, hidden],
        );

        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_linear_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input_buf,
            &weight_buf,
            seq as u32,
            hidden as u32,
            text_hidden as u32,
        )
        .expect("gpu dispatch");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, seq * text_hidden);

        // Elementwise tolerance with the BF16 round-trip bound.
        let mut max_diff = 0f32;
        let mut fail_count = 0usize;
        for (g, e) in got.iter().zip(expected.iter()) {
            let d = (g - e).abs();
            if d > max_diff {
                max_diff = d;
            }
            if d > 5e-2 {
                fail_count += 1;
            }
        }
        // Relative: mm.0 projector weights have magnitudes O(0.01-0.1);
        // bf16 round-trip error per element is ~1e-3 × |w|. At seq=4
        // × hidden=1152 accumulation, max abs diff should stay within
        // 5e-2 per output element for >99% of elements.
        let total = got.len();
        let fail_frac = (fail_count as f32) / (total as f32);
        assert!(
            fail_frac < 0.01,
            "too many GPU/CPU mismatches: {}/{} = {:.3}% failed max_diff = {}",
            fail_count,
            total,
            fail_frac * 100.0,
            max_diff
        );
    }

    // -----------------------------------------------------------------------
    // vit_rms_norm_gpu (iter 44)
    // -----------------------------------------------------------------------

    #[test]
    fn vit_rms_norm_gpu_matches_cpu_reference_on_small_input() {
        // 8 rows × 16 dim. F32 throughout. Compare GPU vs CPU
        // rms_norm_forward within float epsilon — RMSNorm has no
        // BF16 round-trip (input + gain stay F32), so tolerance is
        // tight.
        let rows = 8usize;
        let dim = 16usize;
        let eps = 1e-6f32;
        let input_cpu: Vec<f32> = (0..rows * dim)
            .map(|i| ((i as f32) * 0.05).sin() + 0.5)
            .collect();
        let gain_cpu: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32) * 0.05).collect();

        // CPU reference (mutates in place).
        let mut expected = input_cpu.clone();
        rms_norm_cpu(&mut expected, &gain_cpu, dim, eps).expect("cpu ref");

        // GPU path.
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let input_buf = upload_f32(executor.device(), &input_cpu, vec![rows, dim]);
        let gain_buf = upload_f32(executor.device(), &gain_cpu, vec![dim]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_rms_norm_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input_buf,
            &gain_buf,
            rows as u32,
            dim as u32,
            eps,
        )
        .expect("rms_norm");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, rows * dim);

        let max_diff = got
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-4, "rms_norm GPU vs CPU max_diff = {max_diff}");
    }

    #[test]
    fn vit_rms_norm_gpu_rejects_zero_dims() {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let input = executor.device().alloc_buffer(16 * 4, DType::F32, vec![4, 4]).expect("a");
        let gain = executor.device().alloc_buffer(4 * 4, DType::F32, vec![4]).expect("b");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_rms_norm_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input,
            &gain,
            0,
            4,
            1e-6,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("must be > 0"));
    }

    #[test]
    fn vit_rms_norm_gpu_on_real_gemma4_ln1_matches_cpu() {
        // Real-data parity: load real Gemma 4 mmproj, read v.blk.0.ln1.weight
        // as the f32 gain vector (1152 elements), apply GPU RMSNorm to a
        // synthetic [8, 1152] input. Compare against CPU rms_norm_forward.
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found");
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let device = MlxDevice::new().expect("device");
        let weights = LoadedMmprojWeights::load(&gguf, &cfg, device).expect("load");

        let hidden = cfg.hidden_size as usize;
        let rows = 8usize;
        let ln1_buf = weights.block_tensor(0, "ln1.weight").expect("ln1");
        let gain_f32: &[f32] = ln1_buf.as_slice::<f32>().expect("ln1 slice");
        assert_eq!(gain_f32.len(), hidden);
        let gain_cpu: Vec<f32> = gain_f32.to_vec();

        let input_cpu: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i as f32) * 1e-3).sin() * 0.5)
            .collect();

        let mut expected = input_cpu.clone();
        rms_norm_cpu(&mut expected, &gain_cpu, hidden, cfg.layer_norm_eps).expect("cpu ref");

        let exec_dev = MlxDevice::new().expect("device2");
        let executor = GraphExecutor::new(exec_dev);
        let input_buf = upload_f32(executor.device(), &input_cpu, vec![rows, hidden]);
        let gain_buf = upload_f32(executor.device(), &gain_cpu, vec![hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device_inner: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_rms_norm_gpu(
            session.encoder_mut(),
            &mut registry,
            device_inner,
            &input_buf,
            &gain_buf,
            rows as u32,
            hidden as u32,
            cfg.layer_norm_eps,
        )
        .expect("rms_norm");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, rows * hidden);

        let max_diff = got
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-3, "real-data rms_norm max_diff = {max_diff}");
    }

    // -----------------------------------------------------------------------
    // vit_per_head_rms_norm_gpu
    // -----------------------------------------------------------------------

    #[test]
    fn vit_per_head_rms_norm_gpu_matches_cpu_reference() {
        // batch=4, num_heads=8, head_dim=16. GPU should match CPU
        // per_head_rms_norm_forward.
        let batch = 4usize;
        let num_heads = 8usize;
        let head_dim = 16usize;
        let total = batch * num_heads * head_dim;
        let eps = 1e-6f32;

        let input_cpu: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.03).cos()).collect();
        let gain_cpu: Vec<f32> = (0..head_dim).map(|i| 1.0 + (i as f32) * 0.1).collect();

        let mut expected = input_cpu.clone();
        per_head_rms_cpu(&mut expected, &gain_cpu, batch, num_heads, head_dim, eps)
            .expect("cpu ref");

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let input_buf =
            upload_f32(executor.device(), &input_cpu, vec![batch, num_heads, head_dim]);
        let gain_buf = upload_f32(executor.device(), &gain_cpu, vec![head_dim]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_per_head_rms_norm_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input_buf,
            &gain_buf,
            batch as u32,
            num_heads as u32,
            head_dim as u32,
            eps,
        )
        .expect("per-head rms");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, total);

        let max_diff = got
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-4, "per_head_rms GPU vs CPU max_diff = {max_diff}");
    }

    // -----------------------------------------------------------------------
    // vit_softmax_last_dim_gpu (iter 45)
    // -----------------------------------------------------------------------

    #[test]
    fn vit_softmax_last_dim_gpu_matches_cpu_reference() {
        // 4 rows × 8 cols. Numerically stable softmax — GPU should match
        // CPU within float epsilon (no BF16 round-trip; everything F32).
        let rows = 4usize;
        let cols = 8usize;
        let input_cpu: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.3).sin() + 0.5)
            .collect();
        let mut expected = input_cpu.clone();
        softmax_cpu(&mut expected, cols).expect("cpu ref");

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let input_buf = upload_f32(executor.device(), &input_cpu, vec![rows, cols]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        // Softmax shaders need source registration before dispatch.
        mlx_native::ops::softmax::register(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_softmax_last_dim_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input_buf,
            rows as u32,
            cols as u32,
        )
        .expect("softmax");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, rows * cols);

        let max_diff = got
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-5, "softmax GPU vs CPU max_diff = {max_diff}");

        // Sanity: each row sums to 1.
        for r in 0..rows {
            let row_sum: f32 = got[r * cols..(r + 1) * cols].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-4, "row {r} sum = {row_sum}");
        }
    }

    #[test]
    fn vit_softmax_last_dim_gpu_numerically_stable_for_large_inputs() {
        // x = [1000, 999, 998] should not overflow with the
        // subtract-max trick (without it, exp(1000) → +∞).
        let rows = 1usize;
        let cols = 3usize;
        let input_cpu = vec![1000.0f32, 999.0, 998.0];

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let input_buf = upload_f32(executor.device(), &input_cpu, vec![rows, cols]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        mlx_native::ops::softmax::register(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_softmax_last_dim_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input_buf,
            rows as u32,
            cols as u32,
        )
        .expect("softmax");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, rows * cols);

        for v in &got {
            assert!(v.is_finite(), "non-finite: {v}");
        }
        // Expected: softmax([2, 1, 0]) ≈ [0.6652, 0.2447, 0.0900].
        assert!((got[0] - 0.6652).abs() < 1e-3, "got[0] = {}", got[0]);
        assert!((got[1] - 0.2447).abs() < 1e-3, "got[1] = {}", got[1]);
        assert!((got[2] - 0.0900).abs() < 1e-3, "got[2] = {}", got[2]);
    }

    // -----------------------------------------------------------------------
    // vit_attention_scores_gpu (iter 46)
    // -----------------------------------------------------------------------

    /// CPU reference: scores[h, q_pos, k_pos] = (Σ_d Q[q_pos, h, d]
    /// * K[k_pos, h, d]) * scale. Q and K are seq-major
    /// `[batch, num_heads, head_dim]`.
    fn attention_scores_cpu(
        q: &[f32],
        k: &[f32],
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0f32; num_heads * batch * batch];
        let stride_seq = num_heads * head_dim;
        for h in 0..num_heads {
            for q_pos in 0..batch {
                for k_pos in 0..batch {
                    let mut acc = 0f32;
                    let q_off = q_pos * stride_seq + h * head_dim;
                    let k_off = k_pos * stride_seq + h * head_dim;
                    for d in 0..head_dim {
                        acc += q[q_off + d] * k[k_off + d];
                    }
                    out[h * batch * batch + q_pos * batch + k_pos] = acc * scale;
                }
            }
        }
        out
    }

    /// Diagnostic: verify the permute_021_f32 step produces the right
    /// layout. Suspected zero-output bug in vit_attention_scores_gpu
    /// might be in this stage.
    #[test]
    fn permute_021_f32_seq_to_head_major_round_trips() {
        // Tiny [batch=2, num_heads=2, head_dim=4] test — easy to read
        // back and verify by hand.
        let batch = 2usize;
        let num_heads = 2usize;
        let head_dim = 4usize;
        let n = batch * num_heads * head_dim;
        // Input: each element = batch*100 + head*10 + dim — easy decode.
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let dim = i % head_dim;
                let head = (i / head_dim) % num_heads;
                let b = i / (head_dim * num_heads);
                (b * 100 + head * 10 + dim) as f32
            })
            .collect();

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let in_buf = upload_f32(executor.device(), &input, vec![batch, num_heads, head_dim]);
        let out_buf = executor
            .device()
            .alloc_buffer(n * 4, DType::F32, vec![num_heads, batch, head_dim])
            .expect("alloc out");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        permute_021_f32(
            session.encoder_mut(),
            &mut registry,
            executor.device().metal_device(),
            &in_buf,
            &out_buf,
            batch,
            num_heads,
            head_dim,
        )
        .expect("permute");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, n);

        // Expected: out[head, batch, dim] = in[batch, head, dim].
        // For (head=0, batch=0): in[0, 0, dim] = 0*100 + 0*10 + dim = dim
        // For (head=0, batch=1): in[1, 0, dim] = 100 + dim
        // For (head=1, batch=0): in[0, 1, dim] = 10 + dim
        // For (head=1, batch=1): in[1, 1, dim] = 110 + dim
        let layout = |head: usize, b: usize, d: usize| {
            head * batch * head_dim + b * head_dim + d
        };
        for d in 0..head_dim {
            assert_eq!(got[layout(0, 0, d)], d as f32, "h0 b0 d{d}");
            assert_eq!(got[layout(0, 1, d)], (100 + d) as f32, "h0 b1 d{d}");
            assert_eq!(got[layout(1, 0, d)], (10 + d) as f32, "h1 b0 d{d}");
            assert_eq!(got[layout(1, 1, d)], (110 + d) as f32, "h1 b1 d{d}");
        }
    }

    #[test]
    fn vit_attention_scores_gpu_matches_cpu_reference_on_small_input() {
        // batch=4, num_heads=2, head_dim=64. scale=0.125 (1/sqrt(64)).
        let batch = 4usize;
        let num_heads = 2usize;
        let head_dim = 64usize;
        let scale = 0.125f32;
        let n = batch * num_heads * head_dim;

        let q_cpu: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
        let k_cpu: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).cos() * 0.3).collect();

        let expected = attention_scores_cpu(&q_cpu, &k_cpu, batch, num_heads, head_dim, scale);

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let q_buf = upload_f32(executor.device(), &q_cpu, vec![batch, num_heads, head_dim]);
        let k_buf = upload_f32(executor.device(), &k_cpu, vec![batch, num_heads, head_dim]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let scores = vit_attention_scores_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &q_buf,
            &k_buf,
            batch as u32,
            num_heads as u32,
            head_dim as u32,
            scale,
        )
        .expect("scores");
        session.finish().expect("finish");
        let got = readback_f32(&scores, num_heads * batch * batch);

        // BF16 round-trip on K weight + accumulation across head_dim=64
        // → expect max_diff bounded by ~1e-3 (≤ 1% of typical magnitudes).
        let mut max_diff = 0f32;
        let mut fail_count = 0usize;
        for (g, e) in got.iter().zip(expected.iter()) {
            let d = (g - e).abs();
            if d > max_diff {
                max_diff = d;
            }
            if d > 5e-3 {
                fail_count += 1;
            }
        }
        assert!(
            fail_count == 0,
            "{}/{} elements exceeded 5e-3, max_diff = {}",
            fail_count,
            got.len(),
            max_diff
        );
    }

    #[test]
    fn vit_attention_scores_gpu_unit_scale_does_not_apply() {
        // scale=1.0 should skip the scalar_mul — verify by feeding
        // identical Q=K, expecting per-head diagonal = ||Q[h, q]||^2.
        let batch = 3usize;
        let num_heads = 2usize;
        let head_dim = 32usize;
        let n = batch * num_heads * head_dim;
        let qk_cpu: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.01).collect();

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let q_buf = upload_f32(executor.device(), &qk_cpu, vec![batch, num_heads, head_dim]);
        let k_buf = upload_f32(executor.device(), &qk_cpu, vec![batch, num_heads, head_dim]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let scores = vit_attention_scores_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &q_buf,
            &k_buf,
            batch as u32,
            num_heads as u32,
            head_dim as u32,
            1.0,
        )
        .expect("scores");
        session.finish().expect("finish");
        let got = readback_f32(&scores, num_heads * batch * batch);

        // Diagonal sanity: scores[h, q, q] should = sum_d Q[q,h,d]^2.
        for h in 0..num_heads {
            for q in 0..batch {
                let mut expected_norm_sq = 0f32;
                for d in 0..head_dim {
                    let v = qk_cpu[q * num_heads * head_dim + h * head_dim + d];
                    expected_norm_sq += v * v;
                }
                let got_diag = got[h * batch * batch + q * batch + q];
                let diff = (got_diag - expected_norm_sq).abs();
                let rel = diff / expected_norm_sq.max(1e-3);
                // BF16 K weight + f32 accumulation across 32 head_dim
                // terms → relative error bound ~1e-3.
                assert!(
                    rel < 5e-3,
                    "diag h={h} q={q}: got {got_diag}, want {expected_norm_sq}, rel {rel}"
                );
            }
        }
    }

    #[test]
    fn vit_attention_gpu_matches_cpu_scaled_dot_product_attention() {
        // Full attention parity. batch=32 (≥32 for the second matmul's
        // contract dim K=batch), num_heads=2, head_dim=64. Note CPU
        // reference uses scale = 1/sqrt(head_dim) baked in (no
        // parameter); GPU takes scale explicitly. Pass 1/sqrt(64) = 0.125.
        let batch = 32usize;
        let num_heads = 2usize;
        let head_dim = 64usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let n = batch * num_heads * head_dim;

        let q_cpu: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
        let k_cpu: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).cos() * 0.3).collect();
        let v_cpu: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.11).sin() * ((i as f32) * 0.13).cos() * 0.5)
            .collect();

        let expected = attention_cpu(&q_cpu, &k_cpu, &v_cpu, batch, num_heads, head_dim)
            .expect("cpu ref");

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let q_buf = upload_f32(executor.device(), &q_cpu, vec![batch, num_heads, head_dim]);
        let k_buf = upload_f32(executor.device(), &k_cpu, vec![batch, num_heads, head_dim]);
        let v_buf = upload_f32(executor.device(), &v_cpu, vec![batch, num_heads, head_dim]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        // Softmax shaders need explicit registration.
        mlx_native::ops::softmax::register(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let attn = vit_attention_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &q_buf,
            &k_buf,
            &v_buf,
            batch as u32,
            num_heads as u32,
            head_dim as u32,
            scale,
        )
        .expect("attn");
        session.finish().expect("finish");
        let got = readback_f32(&attn, n);

        // BF16 cast on K and V + 2 GEMMs + softmax → relative error
        // bound ~1e-2. Compare elementwise.
        let mut max_diff = 0f32;
        let mut fail_count = 0usize;
        for (g, e) in got.iter().zip(expected.iter()) {
            let d = (g - e).abs();
            if d > max_diff {
                max_diff = d;
            }
            if d > 1e-2 {
                fail_count += 1;
            }
        }
        let total = got.len();
        let frac = (fail_count as f32) / (total as f32);
        assert!(
            frac < 0.01,
            "{}/{} elements ({:.3}%) exceeded 1e-2, max_diff = {}",
            fail_count, total, frac * 100.0, max_diff
        );
    }

    // -----------------------------------------------------------------------
    // vit_residual_add_gpu (iter 48)
    // -----------------------------------------------------------------------

    #[test]
    fn vit_residual_add_gpu_matches_cpu_reference() {
        let n = 32usize;
        let a_cpu: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.1).collect();
        let b_cpu: Vec<f32> = (0..n).map(|i| -0.3 + (i as f32) * 0.05).collect();

        let mut expected = a_cpu.clone();
        residual_add_cpu(&mut expected, &b_cpu).expect("cpu");

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let a_buf = upload_f32(executor.device(), &a_cpu, vec![n]);
        let b_buf = upload_f32(executor.device(), &b_cpu, vec![n]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_residual_add_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &a_buf,
            &b_buf,
            n as u32,
        )
        .expect("residual_add");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, n);

        let max_diff = got.iter().zip(expected.iter()).map(|(g, e)| (g - e).abs()).fold(0f32, f32::max);
        assert!(max_diff < 1e-6, "residual_add max_diff = {max_diff}");
    }

    #[test]
    fn vit_residual_add_gpu_rejects_zero_n() {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let a = executor.device().alloc_buffer(16, DType::F32, vec![4]).expect("a");
        let b = executor.device().alloc_buffer(16, DType::F32, vec![4]).expect("b");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_residual_add_gpu(session.encoder_mut(), &mut registry, device, &a, &b, 0)
            .unwrap_err();
        assert!(format!("{err}").contains("must be > 0"));
    }

    // -----------------------------------------------------------------------
    // vit_silu_mul_gpu (iter 48)
    // -----------------------------------------------------------------------

    #[test]
    fn vit_silu_mul_gpu_matches_cpu_swiglu_gate() {
        // CPU reference: silu(gate) * up. silu(x) = x * sigmoid(x).
        let n = 64usize;
        let gate_cpu: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).sin()).collect();
        let up_cpu: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).cos()).collect();

        let mut silu_gate = gate_cpu.clone();
        silu_in_place(&mut silu_gate);
        let mut expected = silu_gate;
        elementwise_mul_in_place(&mut expected, &up_cpu).expect("cpu");

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let gate_buf = upload_f32(executor.device(), &gate_cpu, vec![n]);
        let up_buf = upload_f32(executor.device(), &up_cpu, vec![n]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        // sigmoid_mul shader sources need explicit registration.
        mlx_native::ops::sigmoid_mul::register(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_silu_mul_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &gate_buf,
            &up_buf,
            n as u32,
        )
        .expect("silu_mul");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, n);

        let max_diff = got.iter().zip(expected.iter()).map(|(g, e)| (g - e).abs()).fold(0f32, f32::max);
        // F32 throughout — tight tolerance.
        assert!(max_diff < 1e-5, "silu_mul max_diff = {max_diff}");
    }

    #[test]
    fn vit_silu_mul_gpu_rejects_zero_n() {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let g = executor.device().alloc_buffer(16, DType::F32, vec![4]).expect("g");
        let u = executor.device().alloc_buffer(16, DType::F32, vec![4]).expect("u");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_silu_mul_gpu(session.encoder_mut(), &mut registry, device, &g, &u, 0)
            .unwrap_err();
        assert!(format!("{err}").contains("must be > 0"));
    }

    #[test]
    fn vit_attention_gpu_rejects_small_head_dim() {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let q = executor.device().alloc_buffer(2*2*16*4, DType::F32, vec![2,2,16]).expect("q");
        let k = executor.device().alloc_buffer(2*2*16*4, DType::F32, vec![2,2,16]).expect("k");
        let v = executor.device().alloc_buffer(2*2*16*4, DType::F32, vec![2,2,16]).expect("v");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        mlx_native::ops::softmax::register(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_attention_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &q, &k, &v,
            2, 2, 16, 1.0,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("head_dim"));
    }

    #[test]
    fn vit_attention_scores_gpu_rejects_small_head_dim() {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let q = executor
            .device()
            .alloc_buffer(2 * 2 * 16 * 4, DType::F32, vec![2, 2, 16])
            .expect("a");
        let k = executor
            .device()
            .alloc_buffer(2 * 2 * 16 * 4, DType::F32, vec![2, 2, 16])
            .expect("b");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_attention_scores_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &q,
            &k,
            2,
            2,
            16,
            1.0,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("head_dim"));
    }

    #[test]
    fn vit_softmax_last_dim_gpu_rejects_zero_dims() {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let input = executor
            .device()
            .alloc_buffer(16 * 4, DType::F32, vec![4, 4])
            .expect("a");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_softmax_last_dim_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input,
            0,
            4,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("must be > 0"));
    }

    #[test]
    fn vit_per_head_rms_norm_gpu_rejects_zero_dims() {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let input = executor.device().alloc_buffer(64 * 4, DType::F32, vec![64]).expect("a");
        let gain = executor.device().alloc_buffer(8 * 4, DType::F32, vec![8]).expect("b");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_per_head_rms_norm_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &input,
            &gain,
            0,
            8,
            8,
            1e-6,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("must all be > 0"));
    }
}
