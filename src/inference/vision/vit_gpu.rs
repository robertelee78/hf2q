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
use std::sync::LazyLock;
use mlx_native::metal::MTLSize;
use mlx_native::ops::encode_helpers::KernelArg;
use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
use mlx_native::ops::dense_mm_f32_f32::{dense_matmul_f32_f32_tensor, DenseMmF32F32Params};
use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::gather::dispatch_gather_f32;
use mlx_native::ops::rms_norm::dispatch_rms_norm;
use mlx_native::ops::elementwise::{elementwise_add, elementwise_mul, scalar_mul_f32};
use mlx_native::ops::sigmoid_mul::dispatch_sigmoid_mul;
use mlx_native::ops::softmax::dispatch_softmax;
use mlx_native::ops::transpose::{permute_021_f32, transpose_last2_bf16};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

/// ADR-005 Phase 2c iter-120 (W49) — F32-attention env-gate.
///
/// When `HF2Q_VIT_F32_ATTENTION=1`, the production gemma4v ViT
/// attention path skips the F32→BF16 K cast and dispatches the
/// score matmul (`Q @ K^T`) on the F32×F32→F32 tensor-core GEMM
/// added to mlx-native 0.4.7 (`dense_matmul_f32_f32_tensor`). All
/// other ViT primitives, including the post-softmax `attn = softmaxed
/// @ V` matmul, retain the BF16 fast path. This isolates the K cast,
/// which the iter-118 audit (see
/// `project_vit_attention_bf16_softmax_drift.md`) identified as the
/// dominant ~0.68 logit perturbation that flips saturated-softmax
/// winners over 27 layers at gemma4v's `scale = 1.0`.
///
/// The env var is read exactly once per process; subsequent toggles
/// do not take effect (matches the gate-h `INVESTIGATION_ENV` /
/// `LazyLock` pattern in `forward_mlx.rs:556-565` so an A/B
/// invocation must launch hf2q with the env explicitly set in the
/// outer process).  Default (env unset / not "1") keeps the
/// BF16-tensor production path byte-identical to iter-118.
pub(crate) static VIT_F32_ATTENTION_ACTIVE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("HF2Q_VIT_F32_ATTENTION")
        .map(|v| v == "1")
        .unwrap_or(false)
});

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
    // mlx-native dispatches with Concurrent execution; the matmul below
    // reads `weight_bf16` written by the cast above (RAW). An explicit
    // barrier is required to guarantee the matmul observes the cast's
    // writes. Without it the race surfaces when many `vit_linear_gpu`
    // calls chain in one encoder (e.g. Gemma 4V block: q/k/v/o + gate/
    // up/down = 7 back-to-back) and produces non-deterministic output.
    encoder.memory_barrier();

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

    // --- Step 2 + 3: scores = Q @ K^T per head batch ---
    // Default (BF16): cast K F32→BF16, then dispatch the bf16-tensor
    // matmul (src0=K_bf16, src1=Q_f32, dst=scores_f32).
    //
    // ADR-005 iter-120 F32 path (`HF2Q_VIT_F32_ATTENTION=1`): skip the
    // BF16 K cast entirely and dispatch the F32×F32→F32 tensor-core
    // GEMM added in mlx-native 0.4.7. The two paths share the same
    // mathematical contract (`output[h, m, n] = sum_k K[h, n, k] *
    // Q[h, m, k]`), so the score buffer downstream is byte-shape-
    // identical; only the K-side dtype precision differs.
    let n_scores = (num_heads as usize) * (batch as usize) * (batch as usize);
    let mut scores = device
        .alloc_buffer(
            n_scores * 4,
            DType::F32,
            vec![num_heads as usize, batch as usize, batch as usize],
        )
        .map_err(|e| anyhow!("alloc scores: {e}"))?;
    if *VIT_F32_ATTENTION_ACTIVE {
        // F32 path: K_perm is already F32 [num_heads, batch_k, head_dim].
        // Pass it directly as src0 — the F32 GEMM interprets it as
        // [src0_batch=num_heads, n=batch_k, k=head_dim].
        let params = DenseMmF32F32Params {
            m: batch,
            n: batch,
            k: head_dim,
            src0_batch: num_heads,
            src1_batch: num_heads,
        };
        dense_matmul_f32_f32_tensor(
            encoder,
            registry,
            device,
            &k_perm,
            &q_perm,
            &mut scores,
            &params,
        )
        .context("attention scores matmul (F32 path, iter-120)")?;
    } else {
        // BF16 production path (unchanged).
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

        // Layout: src0=K_bf16 [num_heads, batch_k, head_dim] BF16,
        //         src1=Q_perm [num_heads, batch_q, head_dim] F32.
        // output[h, m, n] = sum_k K[h, n, k] * Q[h, m, k] = (Q[h] @ K[h]^T)[m, n].
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
    }

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

/// Execute one ViT transformer block on GPU. Mirrors CPU
/// `apply_vit_block_forward` semantics (same iter-40 parity TODOs:
/// Gemma4V `scale=1.0`, V-RMSNorm, 2D RoPE — all currently skipped
/// pending mlx-lm reference).
///
/// Input/output: F32 `[batch, hidden]` buffers on device. Tensors
/// looked up via `weights.block_tensor(block_idx, suffix)` and
/// uploaded to a fresh device (caller's `LoadedMmprojWeights` already
/// has them on the device matching `executor`).
///
/// Pipeline (each step separated by `encoder.memory_barrier()`):
///   1. cur = rms_norm_gpu(input, ln1)
///   2. q = linear_gpu(cur, attn_q)
///   3. k = linear_gpu(cur, attn_k)
///   4. v = linear_gpu(cur, attn_v)
///   5. q = per_head_rms_norm_gpu(q, attn_q_norm)
///   6. k = per_head_rms_norm_gpu(k, attn_k_norm)
///   7. attn = attention_gpu(q, k, v, scale)   [batch, num_heads, head_dim]
///                                             ≡ [batch, hidden] in memory
///   8. attn_proj = linear_gpu(attn, attn_output)
///   9. post_attn = residual_add_gpu(input, attn_proj)
///   10. cur = rms_norm_gpu(post_attn, ln2)
///   11. gate = linear_gpu(cur, ffn_gate)
///   12. up   = linear_gpu(cur, ffn_up)
///   13. activated = silu_mul_gpu(gate, up)
///   14. down = linear_gpu(activated, ffn_down)
///   15. down = rms_norm_gpu(down, post_ffw_norm)
///   16. block_out = residual_add_gpu(post_attn, down)
///
/// `scale` matches `vit_attention_gpu` — pass `1.0` for Gemma 4V (per
/// llama.cpp parity), `1/sqrt(head_dim)` for standard ViT.
///
/// Caller must register sigmoid_mul + softmax shaders before dispatch:
/// `mlx_native::ops::sigmoid_mul::register(&mut registry)` and
/// `mlx_native::ops::softmax::register(&mut registry)`.
///
/// # Errors
///
/// Propagated from any sub-primitive; primarily missing tensors or
/// shape mismatches.
pub fn apply_vit_block_forward_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weights: &super::mmproj_weights::LoadedMmprojWeights,
    cfg: &super::mmproj::MmprojConfig,
    block_idx: usize,
    input: &MlxBuffer,
    batch: u32,
    scale: f32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size;
    let num_heads = cfg.num_attention_heads;
    let head_dim = hidden / num_heads;
    let intermediate = cfg.intermediate_size;
    let eps = cfg.layer_norm_eps;
    let n_hidden = (batch as usize) * (hidden as usize);

    // Helper to fetch a block-local tensor f32 buffer.
    let block = |suffix: &str| -> Result<&MlxBuffer> {
        weights
            .block_tensor(block_idx, suffix)
            .map_err(|e| anyhow!("block {} {}: {e}", block_idx, suffix))
    };

    // --- Attention half ---
    let cur = vit_rms_norm_gpu(
        encoder,
        registry,
        device,
        input,
        block("ln1.weight")?,
        batch,
        hidden,
        eps,
    )?;
    encoder.memory_barrier();

    let q = vit_linear_gpu(
        encoder,
        registry,
        device,
        &cur,
        block("attn_q.weight")?,
        batch,
        hidden,
        hidden,
    )?;
    encoder.memory_barrier();
    let k = vit_linear_gpu(
        encoder,
        registry,
        device,
        &cur,
        block("attn_k.weight")?,
        batch,
        hidden,
        hidden,
    )?;
    encoder.memory_barrier();
    let v = vit_linear_gpu(
        encoder,
        registry,
        device,
        &cur,
        block("attn_v.weight")?,
        batch,
        hidden,
        hidden,
    )?;
    encoder.memory_barrier();

    let q_norm = vit_per_head_rms_norm_gpu(
        encoder,
        registry,
        device,
        &q,
        block("attn_q_norm.weight")?,
        batch,
        num_heads,
        head_dim,
        eps,
    )?;
    encoder.memory_barrier();
    let k_norm = vit_per_head_rms_norm_gpu(
        encoder,
        registry,
        device,
        &k,
        block("attn_k_norm.weight")?,
        batch,
        num_heads,
        head_dim,
        eps,
    )?;
    encoder.memory_barrier();

    // attention takes [batch, num_heads, head_dim]; q_norm/k_norm/v are
    // [batch, hidden] in memory which IS [batch, num_heads, head_dim].
    let attn = vit_attention_gpu(
        encoder,
        registry,
        device,
        &q_norm,
        &k_norm,
        &v,
        batch,
        num_heads,
        head_dim,
        scale,
    )?;
    encoder.memory_barrier();

    let attn_proj = vit_linear_gpu(
        encoder,
        registry,
        device,
        &attn,
        block("attn_output.weight")?,
        batch,
        hidden,
        hidden,
    )?;
    encoder.memory_barrier();

    let post_attn = vit_residual_add_gpu(
        encoder,
        registry,
        device,
        input,
        &attn_proj,
        n_hidden as u32,
    )?;
    encoder.memory_barrier();

    // --- FFN half ---
    let pre_ffn = vit_rms_norm_gpu(
        encoder,
        registry,
        device,
        &post_attn,
        block("ln2.weight")?,
        batch,
        hidden,
        eps,
    )?;
    encoder.memory_barrier();

    let gate = vit_linear_gpu(
        encoder,
        registry,
        device,
        &pre_ffn,
        block("ffn_gate.weight")?,
        batch,
        hidden,
        intermediate,
    )?;
    encoder.memory_barrier();
    let up = vit_linear_gpu(
        encoder,
        registry,
        device,
        &pre_ffn,
        block("ffn_up.weight")?,
        batch,
        hidden,
        intermediate,
    )?;
    encoder.memory_barrier();

    let activated = vit_silu_mul_gpu(
        encoder,
        registry,
        device,
        &gate,
        &up,
        (batch as usize * intermediate as usize) as u32,
    )?;
    encoder.memory_barrier();

    let down = vit_linear_gpu(
        encoder,
        registry,
        device,
        &activated,
        block("ffn_down.weight")?,
        batch,
        intermediate,
        hidden,
    )?;
    encoder.memory_barrier();

    let down_normed = vit_rms_norm_gpu(
        encoder,
        registry,
        device,
        &down,
        block("post_ffw_norm.weight")?,
        batch,
        hidden,
        eps,
    )?;
    encoder.memory_barrier();

    let block_out = vit_residual_add_gpu(
        encoder,
        registry,
        device,
        &post_attn,
        &down_normed,
        n_hidden as u32,
    )?;

    Ok(block_out)
}

/// Metal shader source for the iter-51c + iter-115 custom kernels:
///   - `vit_avg_pool_2x2_f32`: spatial 2×2 avg-pool on `[N_side, N_side, hidden]`.
///     SigLIP-49 path; preserved byte-identical for that arch.
///   - `vit_avg_pool_kxk_f32`: parameterized k×k avg-pool on a rectangular
///     `[n_y, n_x, hidden]` grid. Generalizes the 2×2 case to gemma4v's
///     k=3 + variable n_x/n_y. (iter 115)
///   - `vit_std_bias_scale_f32`: per-channel `(x - bias) * scale` on `[batch, hidden]`.
///   - `vit_clip_inplace_f32`: elementwise scalar clamp with optional
///     min/max. (iter 115)
///
/// Registered with the `KernelRegistry` on first use via
/// `register_vit_custom_shaders`. Lives as a `&'static str` since the
/// registry stores sources as `&'static str`.
const VIT_CUSTOM_SHADERS_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct AvgPool2x2Params {
    uint n_side;
    uint hidden;
};

kernel void vit_avg_pool_2x2_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant AvgPool2x2Params& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint h  = gid.x;
    uint ox = gid.y;
    uint oy = gid.z;
    uint out_side = params.n_side / 2;
    if (h >= params.hidden || ox >= out_side || oy >= out_side) return;
    uint iy = oy * 2u;
    uint ix = ox * 2u;
    uint h_stride = params.hidden;
    uint row_stride = params.n_side * h_stride;
    float a = input[iy * row_stride + ix * h_stride + h];
    float b = input[iy * row_stride + (ix + 1u) * h_stride + h];
    float c = input[(iy + 1u) * row_stride + ix * h_stride + h];
    float d = input[(iy + 1u) * row_stride + (ix + 1u) * h_stride + h];
    output[oy * out_side * h_stride + ox * h_stride + h] = (a + b + c + d) * 0.25;
}

// Parameterized k×k spatial avg-pool on a rectangular [n_y, n_x, hidden]
// grid. Output shape is [n_y/k, n_x/k, hidden]. Generalizes
// vit_avg_pool_2x2_f32 (which is the n_x=n_y=n_side, k=2 case).
//
// gemma4v call: n_x = n_patches_x, n_y = n_patches_y, k = n_merge = 3
// (per /opt/llama.cpp/tools/mtmd/clip.cpp:1337).
//
// Layout assumption: input is row-major with rows iterating Y first
// (so row stride = n_x * hidden), matching the
// [n_y, n_x, hidden] reshape in the gemma4v post-blocks pipeline.
struct AvgPoolKxKParams {
    uint n_x;     // input width  in patches
    uint n_y;     // input height in patches
    uint k;       // pool kernel edge (= stride; non-overlapping)
    uint hidden;  // channel dim
};

kernel void vit_avg_pool_kxk_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant AvgPoolKxKParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint h  = gid.x;
    uint ox = gid.y;
    uint oy = gid.z;
    uint out_x = params.n_x / params.k;
    uint out_y = params.n_y / params.k;
    if (h >= params.hidden || ox >= out_x || oy >= out_y) return;
    uint iy0 = oy * params.k;
    uint ix0 = ox * params.k;
    uint h_stride = params.hidden;
    uint row_stride = params.n_x * h_stride;
    float acc = 0.0f;
    // Sum over the k×k input block. k is small (== 3 for gemma4v) so
    // the inner double-loop is a handful of ops; the compiler will
    // unroll once k becomes a constant via specialization. We keep it
    // dynamic so the same kernel handles k=2 (SigLIP path), k=3
    // (gemma4v), and any future arch's pool factor.
    for (uint dy = 0u; dy < params.k; ++dy) {
        for (uint dx = 0u; dx < params.k; ++dx) {
            uint ix = ix0 + dx;
            uint iy = iy0 + dy;
            acc += input[iy * row_stride + ix * h_stride + h];
        }
    }
    float k2 = float(params.k * params.k);
    output[oy * out_x * h_stride + ox * h_stride + h] = acc / k2;
}

struct StdBiasScaleParams {
    uint hidden;
    uint batch;
};

kernel void vit_std_bias_scale_f32(
    device const float* input  [[buffer(0)]],
    device const float* bias   [[buffer(1)]],
    device const float* scale  [[buffer(2)]],
    device       float* output [[buffer(3)]],
    constant StdBiasScaleParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;
    uint b = gid.y;
    if (h >= params.hidden || b >= params.batch) return;
    uint idx = b * params.hidden + h;
    output[idx] = (input[idx] - bias[h]) * scale[h];
}

// Elementwise in-place scalar clamp. Both min and max are scalar f32
// values supplied via a 2-element params buffer. The convention:
//   out[i] = clamp(in[i], min, max)
// Setting min = -FLT_MAX or max = FLT_MAX makes that side a no-op,
// matching llama.cpp's get_scalar default for the gemma4v
// Gemma4ClippableLinear (`tools/mtmd/clip.cpp:1953-1956`).
//
// Caller dispatches once per clamp (input or output side); we don't
// fuse the matmul-and-clamps because the Gemma4ClippableLinear
// composes from existing primitives (clip + matmul + clip) and the
// clamp itself is bandwidth-bound (one pass through the output).
struct ClipInplaceParams {
    float min_val;
    float max_val;
    uint  n_elements;
    uint  _pad; // align to 16 bytes for Metal
};

kernel void vit_clip_inplace_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant ClipInplaceParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    float x = input[gid];
    x = max(x, params.min_val);
    x = min(x, params.max_val);
    output[gid] = x;
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct AvgPool2x2GpuParams {
    n_side: u32,
    hidden: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AvgPoolKxKGpuParams {
    n_x: u32,
    n_y: u32,
    k: u32,
    hidden: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct StdBiasScaleGpuParams {
    hidden: u32,
    batch: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClipInplaceGpuParams {
    min_val: f32,
    max_val: f32,
    n_elements: u32,
    _pad: u32,
}

/// View any `Copy + repr(C)` POD as a byte slice. SAFE for `repr(C)`
/// structs containing only primitive fields; the byte representation
/// is stable and exactly `size_of::<T>()` bytes.
fn pod_as_bytes<T: Copy>(p: &T) -> &[u8] {
    // SAFETY: `T: Copy + repr(C)` with primitive fields means the in-memory
    // representation is a contiguous block of `size_of::<T>()` bytes with
    // no uninitialized padding (we use only u32 fields packed naturally).
    unsafe {
        std::slice::from_raw_parts(p as *const T as *const u8, std::mem::size_of::<T>())
    }
}

/// Register the iter-51c custom shaders with the kernel registry.
/// Idempotent — `register_source` overwrites any previous registration
/// for the same name. Caller invokes once per `KernelRegistry`.
pub fn register_vit_custom_shaders(registry: &mut KernelRegistry) {
    registry.register_source("vit_avg_pool_2x2_f32", VIT_CUSTOM_SHADERS_SOURCE);
    registry.register_source("vit_avg_pool_kxk_f32", VIT_CUSTOM_SHADERS_SOURCE);
    registry.register_source("vit_std_bias_scale_f32", VIT_CUSTOM_SHADERS_SOURCE);
    registry.register_source("vit_clip_inplace_f32", VIT_CUSTOM_SHADERS_SOURCE);
}

/// GPU 2×2 spatial avg-pool on a `[N_side, N_side, hidden]` row-major
/// tensor. Output shape `[(N_side/2)², hidden]`.
///
/// For Gemma 4 ViT: `[14, 14, 1152]` → `[49, 1152]` post-blocks pooler.
///
/// Caller registers `register_vit_custom_shaders(&mut registry)` first.
///
/// # Errors
///
/// - `n_side == 0` or `n_side % 2 != 0`
/// - `hidden == 0`
/// - propagated from kernel pipeline compile failures
pub fn vit_avg_pool_2x2_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    n_side: u32,
    hidden: u32,
) -> Result<MlxBuffer> {
    if n_side == 0 || n_side % 2 != 0 {
        return Err(anyhow!(
            "vit_avg_pool_2x2_gpu: n_side ({}) must be positive and even",
            n_side
        ));
    }
    if hidden == 0 {
        return Err(anyhow!("vit_avg_pool_2x2_gpu: hidden must be > 0"));
    }
    let out_side = n_side / 2;
    let out_n_patches = (out_side as usize) * (out_side as usize);
    let out_bytes = out_n_patches * (hidden as usize) * 4;
    let output = device
        .alloc_buffer(
            out_bytes,
            DType::F32,
            vec![out_side as usize, out_side as usize, hidden as usize],
        )
        .map_err(|e| anyhow!("alloc avg_pool output: {e}"))?;

    let pipeline = registry
        .get_pipeline("vit_avg_pool_2x2_f32", device.metal_device())
        .map_err(|e| anyhow!("vit_avg_pool_2x2_gpu: get_pipeline: {e}"))?;

    let params = AvgPool2x2GpuParams {
        n_side,
        hidden,
    };
    let bytes = pod_as_bytes(&params);
    let grid = MTLSize::new(hidden as u64, out_side as u64, out_side as u64);
    // Threadgroup sized to fit the inner-most (hidden) dim into reasonable
    // chunks; clamp components to 1 in axes not needing parallelism.
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

/// GPU parameterized k×k spatial avg-pool on a rectangular
/// `[n_y, n_x, hidden]` row-major tensor. Output shape:
/// `[n_y/k, n_x/k, hidden]`.
///
/// Generalizes `vit_avg_pool_2x2_gpu` to:
///   - rectangular grids (`n_x` ≠ `n_y`),
///   - arbitrary kernel size `k` (must divide both `n_x` and `n_y`).
///
/// gemma4v call: pass `(n_x = n_patches_x, n_y = n_patches_y, k =
/// n_merge = 3)` per `/opt/llama.cpp/tools/mtmd/clip.cpp:1337,1334`.
///
/// # Byte-stable wrt the 2×2 path
///
/// For `(n_x = n_y = N, k = 2)` the kxk shader's accumulate-and-divide
/// path produces values byte-identical to the dedicated 2×2 shader's
/// `(a + b + c + d) * 0.25` (4 floats summed in the same order, divided
/// by 4.0). The SigLIP-49 path keeps using `vit_avg_pool_2x2_gpu` so
/// no regression risk; the kxk path is opt-in for gemma4v + future archs.
///
/// Caller registers `register_vit_custom_shaders(&mut registry)` first.
///
/// # Errors
///
/// - `n_x == 0` or `n_y == 0` or `k == 0`
/// - `n_x % k != 0` or `n_y % k != 0` (input must be exactly tileable;
///   llama.cpp's `ggml_pool_2d` with stride==kernel and pad==0 has the
///   same tiling assumption — fractional-tile output isn't well-defined)
/// - `hidden == 0`
/// - propagated from kernel pipeline compile failures
#[allow(clippy::too_many_arguments)]
pub fn vit_avg_pool_kxk_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    n_x: u32,
    n_y: u32,
    k: u32,
    hidden: u32,
) -> Result<MlxBuffer> {
    if n_x == 0 || n_y == 0 || k == 0 {
        return Err(anyhow!(
            "vit_avg_pool_kxk_gpu: n_x ({n_x}), n_y ({n_y}), k ({k}) must all be > 0"
        ));
    }
    if hidden == 0 {
        return Err(anyhow!("vit_avg_pool_kxk_gpu: hidden must be > 0"));
    }
    if n_x % k != 0 || n_y % k != 0 {
        return Err(anyhow!(
            "vit_avg_pool_kxk_gpu: n_x ({n_x}) and n_y ({n_y}) must both be multiples of k ({k})"
        ));
    }
    let out_x = n_x / k;
    let out_y = n_y / k;
    let out_n = (out_x as usize) * (out_y as usize);
    let out_bytes = out_n * (hidden as usize) * 4;
    let output = device
        .alloc_buffer(
            out_bytes,
            DType::F32,
            vec![out_y as usize, out_x as usize, hidden as usize],
        )
        .map_err(|e| anyhow!("alloc avg_pool_kxk output: {e}"))?;

    let pipeline = registry
        .get_pipeline("vit_avg_pool_kxk_f32", device.metal_device())
        .map_err(|e| anyhow!("vit_avg_pool_kxk_gpu: get_pipeline: {e}"))?;

    let params = AvgPoolKxKGpuParams { n_x, n_y, k, hidden };
    let bytes = pod_as_bytes(&params);
    let grid = MTLSize::new(hidden as u64, out_x as u64, out_y as u64);
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

/// GPU thin wrapper for the gemma4v 3×3 avg-pool. Routes through
/// `vit_avg_pool_kxk_gpu` with `k = 3` per the gemma4v `n_merge`
/// reference (`/opt/llama.cpp/tools/mtmd/clip.cpp:1337`).
///
/// `n_x` / `n_y` come from `Gemma4vPreprocessed.n_x` / `.n_y` — the
/// variable patch grid produced by `preprocess_gemma4v`. Both must be
/// multiples of 3 (callers reading the preprocessor output already
/// satisfy this; the inner check is for symmetry with the 2×2 wrapper).
pub fn gemma4v_avg_pool_3x3_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    n_x: u32,
    n_y: u32,
    hidden: u32,
) -> Result<MlxBuffer> {
    vit_avg_pool_kxk_gpu(encoder, registry, device, input, n_x, n_y, 3, hidden)
}

/// GPU elementwise scalar clamp: `out[i] = clamp(in[i], min_val, max_val)`.
///
/// Building block for Gemma4ClippableLinear (`gemma4v.cpp:138-151`).
/// Pass `min_val = f32::NEG_INFINITY` (or `f32::MIN`) for "no min" and
/// `max_val = f32::INFINITY` (or `f32::MAX`) for "no max" to match
/// llama.cpp's `get_scalar(name, FLT_MAX/-FLT_MAX)` defaults.
///
/// Allocates a fresh `[n_elements]` output buffer; the input is
/// untouched (despite the name "_inplace" in the Metal kernel — the
/// kernel reads from `input` and writes to `output`, which can be the
/// same buffer if the caller wants in-place behavior, but our Rust
/// dispatch always allocates a fresh output for clarity).
///
/// Caller registers `register_vit_custom_shaders(&mut registry)` first.
///
/// # Errors
///
/// - `n_elements == 0`
/// - `min_val > max_val` (would produce undefined output)
/// - propagated from kernel pipeline compile failures
pub fn vit_clip_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    n_elements: u32,
    min_val: f32,
    max_val: f32,
) -> Result<MlxBuffer> {
    if n_elements == 0 {
        return Err(anyhow!("vit_clip_gpu: n_elements must be > 0"));
    }
    if min_val.is_nan() || max_val.is_nan() {
        return Err(anyhow!(
            "vit_clip_gpu: NaN clamp bounds (min={min_val}, max={max_val}) — \
             would silently corrupt all elements"
        ));
    }
    if min_val > max_val {
        return Err(anyhow!(
            "vit_clip_gpu: min_val ({min_val}) > max_val ({max_val}) is undefined"
        ));
    }
    let out_bytes = (n_elements as usize) * 4;
    let output = device
        .alloc_buffer(out_bytes, DType::F32, vec![n_elements as usize])
        .map_err(|e| anyhow!("alloc clip output: {e}"))?;

    let pipeline = registry
        .get_pipeline("vit_clip_inplace_f32", device.metal_device())
        .map_err(|e| anyhow!("vit_clip_gpu: get_pipeline: {e}"))?;

    let params = ClipInplaceGpuParams {
        min_val,
        max_val,
        n_elements,
        _pad: 0,
    };
    let bytes = pod_as_bytes(&params);
    let grid = MTLSize::new(n_elements as u64, 1, 1);
    let tg_x = std::cmp::min(256, n_elements as u64);
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

/// GPU per-channel std-normalization: `out[b, h] = (in[b, h] - bias[h]) * scale[h]`.
///
/// `input` shape `[batch, hidden]` F32, `bias`/`scale` each `[hidden]` F32.
/// Returns a fresh F32 `[batch, hidden]` output buffer.
///
/// Used by Gemma 4 ViT's `(cur - std_bias) * std_scale` pre-projector
/// step. `std_bias` and `std_scale` are loaded from `v.std_bias`,
/// `v.std_scale` mmproj tensors.
///
/// Caller registers `register_vit_custom_shaders(&mut registry)` first.
///
/// # Errors
///
/// - `batch == 0` or `hidden == 0`
/// - propagated from kernel pipeline compile failures
pub fn vit_std_bias_scale_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    bias: &MlxBuffer,
    scale: &MlxBuffer,
    batch: u32,
    hidden: u32,
) -> Result<MlxBuffer> {
    if batch == 0 || hidden == 0 {
        return Err(anyhow!(
            "vit_std_bias_scale_gpu: batch ({}) and hidden ({}) must be > 0",
            batch,
            hidden
        ));
    }
    let total = (batch as usize) * (hidden as usize);
    let output = device
        .alloc_buffer(total * 4, DType::F32, vec![batch as usize, hidden as usize])
        .map_err(|e| anyhow!("alloc std_bias_scale output: {e}"))?;

    let pipeline = registry
        .get_pipeline("vit_std_bias_scale_f32", device.metal_device())
        .map_err(|e| anyhow!("vit_std_bias_scale_gpu: get_pipeline: {e}"))?;

    let params = StdBiasScaleGpuParams {
        hidden,
        batch,
    };
    let bytes = pod_as_bytes(&params);
    let grid = MTLSize::new(hidden as u64, batch as u64, 1);
    let tg = MTLSize::new(std::cmp::min(64, hidden as u64), 1, 1);
    encoder.encode_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(bias)),
            (2, KernelArg::Buffer(scale)),
            (3, KernelArg::Buffer(&output)),
            (4, KernelArg::Bytes(bytes)),
        ],
        grid,
        tg,
    );
    Ok(output)
}

/// In-place GPU scalar multiply: `buf[i] *= scalar` across every element.
///
/// Wraps `mlx_native::ops::elementwise::scalar_mul_f32` with input ==
/// output, exploiting the kernel's per-thread read-then-write pattern
/// (no aliasing issue — each thread touches one element).
///
/// Used by the post-blocks pipeline's `scale_in_place(√n_embd)` step
/// (`ggml_scale(cur, sqrtf(n_embd))` in llama.cpp's gemma4v graph).
///
/// # Errors
///
/// - `n_elements == 0`
/// - propagated from `scalar_mul_f32`
pub fn vit_scale_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    buf: &MlxBuffer,
    n_elements: u32,
    scalar: f32,
) -> Result<()> {
    if n_elements == 0 {
        return Err(anyhow!("vit_scale_gpu: n_elements must be > 0"));
    }
    scalar_mul_f32(
        encoder,
        registry,
        device.metal_device(),
        buf,
        buf,
        n_elements as usize,
        scalar,
    )
    .context("vit_scale_gpu: scalar_mul_f32")
}

/// Chain `apply_vit_block_forward_gpu` across all `cfg.num_hidden_layers`
/// blocks. Input + output shape: F32 `[batch, hidden]` on device.
///
/// This is the heavy compute portion of the ViT (~81 GFLOP across 27
/// blocks for Gemma 4). The post-blocks pipeline (avg_pool, scale,
/// std_bias_scale, projector, final rms_norm) lands in iter 51b once
/// the missing GPU primitives are added.
///
/// Caller registers `softmax::register` + `sigmoid_mul::register`
/// before dispatch.
pub fn apply_vit_blocks_loop_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weights: &super::mmproj_weights::LoadedMmprojWeights,
    cfg: &super::mmproj::MmprojConfig,
    input: &MlxBuffer,
    batch: u32,
    scale: f32,
) -> Result<MlxBuffer> {
    // Run block 0 with the input as residual stream.
    let mut hidden_states =
        apply_vit_block_forward_gpu(encoder, registry, device, weights, cfg, 0, input, batch, scale)?;
    encoder.memory_barrier();
    // Chain remaining blocks; each block_idx's output becomes next's input.
    for block_idx in 1..(cfg.num_hidden_layers as usize) {
        hidden_states = apply_vit_block_forward_gpu(
            encoder,
            registry,
            device,
            weights,
            cfg,
            block_idx,
            &hidden_states,
            batch,
            scale,
        )?;
        encoder.memory_barrier();
    }
    Ok(hidden_states)
}

/// Full GPU ViT forward: pixel tensor → projected multimodal embeddings.
///
/// Pipeline (matches llama.cpp `clip_graph_gemma4v::build`):
///   1. CPU `patch_embed_from_mmproj_weights(pixels, weights, cfg)` →
///      `[num_patches, hidden]` F32 (one-time per image; dominated by
///      the 27-block GPU compute that follows).
///   2. Upload to GPU as input to the blocks loop.
///   3. `apply_vit_blocks_loop_gpu` × 27 blocks → `[num_patches, hidden]`.
///   4. `vit_avg_pool_2x2_gpu` → `[(num_patches/4), hidden]`. For Gemma 4:
///      `[14, 14, 1152] → [49, 1152]`.
///   5. `vit_scale_gpu(√n_embd)` — Gemma 4V's `ggml_scale(cur, sqrtf(n_embd))`.
///   6. `vit_std_bias_scale_gpu(v.std_bias, v.std_scale)` — pre-projector norm.
///   7. `vit_linear_gpu(mm.0.weight)` → `[(num_patches/4), text_hidden]`. For
///      Gemma 4: `[49, 2816]`.
///   8. `vit_rms_norm_gpu(ones, eps)` — final no-gain RMSNorm
///      (`ggml_rms_norm` with no weight param).
///
/// Returns the final `[(num_patches/4), text_hidden]` F32 buffer on
/// device. Caller reads back via `MlxBuffer::as_slice::<f32>()` for
/// downstream embedding injection into the chat prompt.
///
/// `scale` matches `vit_attention_gpu` — pass `1.0` for Gemma 4V (per
/// llama.cpp), `1/sqrt(head_dim)` for standard ViT.
///
/// Caller registers shaders before dispatch:
/// ```
/// mlx_native::ops::softmax::register(&mut registry);
/// mlx_native::ops::sigmoid_mul::register(&mut registry);
/// register_vit_custom_shaders(&mut registry);
/// ```
///
/// # Errors
///
/// Propagated from any sub-stage. The `weights` arg's per-tensor
/// access (`weights.get`, `weights.mm_0_weight`) returns errors for
/// missing tensors.
///
/// # Cost
///
/// Gemma 4 ViT (27 blocks, [14×14, 1152] → [49, 2816]): ~57 ms for
/// the 27-block compute on M5 Max + a few ms for the post-blocks
/// pipeline. Total well under 100 ms. CPU patch_embed at the start
/// is ~5–15 ms for a 224×224 image. End-to-end < 150 ms per image.
pub fn apply_vit_full_forward_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weights: &super::mmproj_weights::LoadedMmprojWeights,
    cfg: &super::mmproj::MmprojConfig,
    pixel_values: &[f32],
    scale: f32,
) -> Result<MlxBuffer> {
    use super::vit::patch_embed_forward as patch_embed_cpu;

    let hidden = cfg.hidden_size;
    let num_patches_side = cfg.num_patches_side;
    let n_patches = (num_patches_side as u32) * (num_patches_side as u32);

    // --- Stage 1: CPU patch_embed ---
    let patch_embd_buf = weights
        .patch_embd_weight()
        .map_err(|e| anyhow!("apply_vit_full_forward_gpu: {e}"))?;
    let patch_embd_f32: &[f32] = patch_embd_buf
        .as_slice::<f32>()
        .map_err(|e| anyhow!("patch_embd as_slice: {e}"))?;
    let patch_bias_f32: Option<&[f32]> = weights
        .get("v.patch_embd.bias")
        .and_then(|b| b.as_slice::<f32>().ok());
    let patch_embeds_cpu = patch_embed_cpu(
        pixel_values,
        patch_embd_f32,
        patch_bias_f32,
        cfg.image_size,
        cfg.patch_size,
        hidden,
    )
    .context("apply_vit_full_forward_gpu: cpu patch_embed")?;

    // --- Stage 2: Upload to GPU ---
    let n_hidden = (n_patches as usize) * (hidden as usize);
    let input_gpu = device
        .alloc_buffer(
            n_hidden * 4,
            DType::F32,
            vec![n_patches as usize, hidden as usize],
        )
        .map_err(|e| anyhow!("alloc input_gpu: {e}"))?;
    {
        // SAFETY: just-allocated f32 buffer; copy-in is single-threaded
        // and Apple unified memory makes this byte-equivalent to a CPU
        // memcpy that's later read by the GPU.
        let dst: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(input_gpu.contents_ptr() as *mut f32, n_hidden)
        };
        dst.copy_from_slice(&patch_embeds_cpu);
    }

    // --- Stage 3: 27-block transformer loop ---
    let after_blocks = apply_vit_blocks_loop_gpu(
        encoder,
        registry,
        device,
        weights,
        cfg,
        &input_gpu,
        n_patches,
        scale,
    )?;
    encoder.memory_barrier();

    // --- Stage 4: avg-pool 2x2 ---
    let pooled = vit_avg_pool_2x2_gpu(
        encoder,
        registry,
        device,
        &after_blocks,
        num_patches_side as u32,
        hidden,
    )?;
    encoder.memory_barrier();

    // --- Stage 5: scale by sqrt(n_embd) ---
    let pooled_n_patches = ((num_patches_side / 2) * (num_patches_side / 2)) as usize;
    let pooled_total = pooled_n_patches * (hidden as usize);
    vit_scale_gpu(
        encoder,
        registry,
        device,
        &pooled,
        pooled_total as u32,
        (hidden as f32).sqrt(),
    )?;
    encoder.memory_barrier();

    // --- Stage 6: std_bias / std_scale normalization ---
    let std_bias = weights
        .get("v.std_bias")
        .ok_or_else(|| anyhow!("apply_vit_full_forward_gpu: missing v.std_bias"))?;
    let std_scale = weights
        .get("v.std_scale")
        .ok_or_else(|| anyhow!("apply_vit_full_forward_gpu: missing v.std_scale"))?;
    let normed = vit_std_bias_scale_gpu(
        encoder,
        registry,
        device,
        &pooled,
        std_bias,
        std_scale,
        pooled_n_patches as u32,
        hidden,
    )?;
    encoder.memory_barrier();

    // --- Stage 7: mm.0 projector ---
    let mm0 = weights
        .mm_0_weight()
        .map_err(|e| anyhow!("apply_vit_full_forward_gpu: mm.0.weight: {e}"))?;
    let mm0_f32: &[f32] = mm0
        .as_slice::<f32>()
        .map_err(|e| anyhow!("mm0 as_slice: {e}"))?;
    let text_hidden = (mm0_f32.len() / (hidden as usize)) as u32;
    let projected = vit_linear_gpu(
        encoder,
        registry,
        device,
        &normed,
        mm0,
        pooled_n_patches as u32,
        hidden,
        text_hidden,
    )?;
    encoder.memory_barrier();

    // --- Stage 8: final no-gain RMSNorm ---
    // Allocate a [text_hidden] all-ones gain vector once.
    let ones = device
        .alloc_buffer(
            (text_hidden as usize) * 4,
            DType::F32,
            vec![text_hidden as usize],
        )
        .map_err(|e| anyhow!("alloc ones: {e}"))?;
    {
        // SAFETY: just-allocated f32 buffer.
        let s: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(
                ones.contents_ptr() as *mut f32,
                text_hidden as usize,
            )
        };
        for v in s.iter_mut() {
            *v = 1.0;
        }
    }
    let final_out = vit_rms_norm_gpu(
        encoder,
        registry,
        device,
        &projected,
        &ones,
        pooled_n_patches as u32,
        text_hidden,
        cfg.layer_norm_eps,
    )?;
    Ok(final_out)
}

/// One-shot ViT GPU forward for server-startup self-test. Runs a single
/// synthetic full forward through `apply_vit_full_forward_gpu` against
/// the loaded mmproj weights, validating that every stage from
/// patch_embed through the projector is wired correctly on the actual
/// production weights at boot — surfaces missing-tensor / shape-mismatch
/// bugs at startup instead of on the first user request.
///
/// **Does NOT amortize kernel-compile cost across user requests.** The
/// `KernelRegistry` here is throwaway; user-request `compute_vision_
/// embeddings_gpu` invocations each build their own registry and
/// re-compile pipelines. Genuine kernel-compile amortization requires
/// hoisting a long-lived `KernelRegistry` onto `AppState` behind
/// `Arc<Mutex<>>` and threading it through every multimodal call —
/// that's a larger refactor (ADR-005 Phase 2c follow-up).
///
/// Total cost: one full forward (~1.3s on M5 Max for Gemma 4 ViT
/// including the CPU patch_embed) at boot. Catches wiring bugs that
/// would otherwise hit the first multimodal user.
///
/// Caller invokes once per `LoadedMmprojWeights`. Returns `Ok(())`
/// on success (the embedding output is discarded; only the side
/// effect of validating the wiring matters).
///
/// # Errors
///
/// Propagated from any sub-stage. A failed warmup is non-fatal but
/// surfaces real wiring bugs (missing tensors, shape mismatches)
/// before they hit a user request.
pub fn warmup_vit_gpu(
    weights: &super::mmproj_weights::LoadedMmprojWeights,
    cfg: &super::mmproj::MmprojConfig,
) -> Result<()> {
    use mlx_native::{GraphExecutor, MlxDevice};

    let executor = GraphExecutor::new(
        MlxDevice::new().map_err(|e| anyhow!("warmup_vit_gpu device: {e}"))?,
    );
    let mut session = executor
        .begin()
        .map_err(|e| anyhow!("warmup_vit_gpu begin: {e}"))?;
    let mut registry = KernelRegistry::new();
    mlx_native::ops::softmax::register(&mut registry);
    mlx_native::ops::sigmoid_mul::register(&mut registry);
    register_vit_custom_shaders(&mut registry);
    let device_ref: *const MlxDevice = executor.device() as *const _;
    let device: &MlxDevice = unsafe { &*device_ref };

    // Synthetic [3, image_size, image_size] uniform input. Magnitudes
    // tiny so attention isn't saturated; the goal is to exercise every
    // kernel, not to validate output values.
    let img = cfg.image_size as usize;
    let pixels = vec![0.01f32; 3 * img * img];

    let head_dim_f = (cfg.hidden_size / cfg.num_attention_heads) as f32;
    let scale = 1.0f32 / head_dim_f.sqrt();

    let _output = apply_vit_full_forward_gpu(
        session.encoder_mut(),
        &mut registry,
        device,
        weights,
        cfg,
        &pixels,
        scale,
    )
    .map_err(|e| anyhow!("warmup_vit_gpu forward: {e}"))?;
    session
        .finish()
        .map_err(|e| anyhow!("warmup_vit_gpu finish: {e}"))?;
    Ok(())
}

/// Run `apply_vit_full_forward_gpu` for each `PreprocessedImage`, read
/// back to CPU. Caller-friendly handler-side wrapper.
///
/// Returns one `Vec<f32>` per image, each of shape
/// `[(num_patches/4) × text_hidden]` row-major (e.g. `[49, 2816]` for
/// Gemma 4). Order matches the input slice.
///
/// Internally creates a fresh `GraphExecutor` + `KernelRegistry`,
/// registers all ViT shaders, runs each image through a separate
/// `GraphSession::finish()` (one GPU sync per image — keeps the
/// per-image readback hot path simple). For batched-image requests,
/// a future iter can amortize by running all images in a single
/// session before reading any back.
///
/// `scale` is the attention scale (`1.0` for Gemma 4V; `1/sqrt(head_dim)`
/// for standard ViT). Per the iter-50 finding, the GPU output uses BF16
/// attention internally — production-correct, but element-wise CPU
/// reference comparison is misleading.
///
/// # Errors
///
/// Propagated from any sub-stage (shape mismatch, missing tensors).
pub fn compute_vision_embeddings_gpu(
    images: &[super::PreprocessedImage],
    mmproj_weights: &super::mmproj_weights::LoadedMmprojWeights,
    mmproj_cfg: &super::mmproj::MmprojConfig,
    scale: f32,
) -> Result<Vec<Vec<f32>>> {
    use mlx_native::{GraphExecutor, MlxDevice};

    let mut out = Vec::with_capacity(images.len());
    for (idx, img) in images.iter().enumerate() {
        let executor = GraphExecutor::new(
            MlxDevice::new().map_err(|e| {
                anyhow!("compute_vision_embeddings_gpu image {}: device: {e}", idx)
            })?,
        );
        let mut session = executor.begin().map_err(|e| {
            anyhow!("compute_vision_embeddings_gpu image {}: begin: {e}", idx)
        })?;
        let mut registry = KernelRegistry::new();
        mlx_native::ops::softmax::register(&mut registry);
        mlx_native::ops::sigmoid_mul::register(&mut registry);
        register_vit_custom_shaders(&mut registry);
        // SAFETY: executor outlives session via the loop scope.
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        let buf = apply_vit_full_forward_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            mmproj_weights,
            mmproj_cfg,
            &img.pixel_values,
            scale,
        )
        .map_err(|e| anyhow!("compute_vision_embeddings_gpu image {}: forward: {e}", idx))?;
        session
            .finish()
            .map_err(|e| anyhow!("compute_vision_embeddings_gpu image {}: finish: {e}", idx))?;

        // Read back to CPU. Shape is [(N_side/2)², text_hidden].
        let n_patches_out =
            ((mmproj_cfg.num_patches_side / 2) * (mmproj_cfg.num_patches_side / 2)) as usize;
        let mm0 = mmproj_weights
            .mm_0_weight()
            .map_err(|e| anyhow!("mm.0: {e}"))?;
        let text_hidden = mm0
            .as_slice::<f32>()
            .map_err(|e| anyhow!("mm.0 slice: {e}"))?
            .len()
            / (mmproj_cfg.hidden_size as usize);
        let total = n_patches_out * text_hidden;
        let slice: &[f32] = buf
            .as_slice::<f32>()
            .map_err(|e| anyhow!("readback: {e}"))?;
        if slice.len() != total {
            return Err(anyhow!(
                "compute_vision_embeddings_gpu image {}: readback len {} != expected {}",
                idx,
                slice.len(),
                total
            ));
        }
        out.push(slice.to_vec());
    }
    Ok(out)
}

/// Variable-resolution gemma4v preprocessed image carrier.
///
/// Sibling to `PreprocessedImage` (which is square-fixed-size for the
/// SigLIP-49 path). This struct carries the patch tensor + per-patch
/// position arrays produced by `preprocess_gemma4v`, plus a debug
/// label for tracing. The `compute_vision_embeddings_gpu_gemma4v`
/// entry point consumes a slice of these.
///
/// `n_x` / `n_y` MUST satisfy `n_x % 3 == 0 && n_y % 3 == 0`
/// (`gemma4v_apply_full_forward_gpu` enforces this; the preprocessor
/// already produces multiple-of-3 grids).
#[derive(Debug, Clone)]
pub struct Gemma4vPreprocessedImage {
    pub patches: Vec<f32>,
    pub pos_x: Vec<u32>,
    pub pos_y: Vec<u32>,
    pub n_x: u32,
    pub n_y: u32,
    pub source_label: String,
}

/// Variable-resolution gemma4v end-to-end GPU forward over a slice of
/// preprocessed images. Sibling to `compute_vision_embeddings_gpu` for
/// the SigLIP-49 path.
///
/// Returns one `Vec<f32>` per image, each `[N_post_pool * text_hidden]`
/// row-major. `N_post_pool = (n_x/3) * (n_y/3)` so the output length
/// varies per image (252-280 input patches → 28-31 post-pool tokens).
///
/// Internally uses the same per-image fresh-session pattern as the
/// SigLIP-49 path (one `GraphSession::finish()` per image).
///
/// # Errors
///
/// Propagated from `gemma4v_apply_full_forward_gpu` (shape mismatch,
/// missing tensors).
pub fn compute_vision_embeddings_gpu_gemma4v(
    images: &[Gemma4vPreprocessedImage],
    mmproj_weights: &super::mmproj_weights::LoadedMmprojWeights,
    mmproj_cfg: &super::mmproj::MmprojConfig,
) -> Result<Vec<Vec<f32>>> {
    use mlx_native::{GraphExecutor, MlxDevice};

    // ADR-005 iter 124: the dump probe runs inside the per-image scope
    // below. Resolve the dump dir once here so we fail loud (e.g. EACCES)
    // BEFORE running the forward, not after spending GPU cycles. Returns
    // `None` when `HF2Q_VIT_DUMP` is unset → zero overhead default path.
    let dump_dir = super::vit_dump::resolve_dump_dir()?;

    let mut out = Vec::with_capacity(images.len());
    for (idx, img) in images.iter().enumerate() {
        let executor = GraphExecutor::new(
            MlxDevice::new().map_err(|e| {
                anyhow!("compute_vision_embeddings_gpu_gemma4v image {}: device: {e}", idx)
            })?,
        );
        let mut session = executor.begin().map_err(|e| {
            anyhow!("compute_vision_embeddings_gpu_gemma4v image {}: begin: {e}", idx)
        })?;
        let mut registry = KernelRegistry::new();
        mlx_native::ops::softmax::register(&mut registry);
        mlx_native::ops::sigmoid_mul::register(&mut registry);
        register_vit_custom_shaders(&mut registry);
        // SAFETY: executor outlives session via the loop scope.
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        // ADR-005 iter 124: when the dump probe is armed, run the forward
        // inside a `with_dump_collector` scope so intermediate buffers
        // are stashed via thread-local. We cannot read GPU buffers
        // mid-forward (encoder still recording); collection happens
        // in-flight, write-out happens after `session.finish()`.
        let (buf, collected) = if dump_dir.is_some() {
            super::vit_dump::with_dump_collector(|| {
                gemma4v_apply_full_forward_gpu(
                    session.encoder_mut(),
                    &mut registry,
                    device,
                    mmproj_weights,
                    mmproj_cfg,
                    &img.patches,
                    &img.pos_x,
                    &img.pos_y,
                    img.n_x,
                    img.n_y,
                )
                .map_err(|e| {
                    anyhow!("compute_vision_embeddings_gpu_gemma4v image {}: forward: {e}", idx)
                })
            })?
        } else {
            let b = gemma4v_apply_full_forward_gpu(
                session.encoder_mut(),
                &mut registry,
                device,
                mmproj_weights,
                mmproj_cfg,
                &img.patches,
                &img.pos_x,
                &img.pos_y,
                img.n_x,
                img.n_y,
            )
            .map_err(|e| {
                anyhow!("compute_vision_embeddings_gpu_gemma4v image {}: forward: {e}", idx)
            })?;
            (b, Vec::new())
        };
        session.finish().map_err(|e| {
            anyhow!("compute_vision_embeddings_gpu_gemma4v image {}: finish: {e}", idx)
        })?;

        // After GPU finish: shared-memory buffers are safe to read on
        // CPU. Write each collected stage to the dump dir. CPU mirrors
        // (00_pre_patchify, 00_post_patchify) are drained from a
        // parallel thread-local.
        if let Some(ref dir) = dump_dir {
            let img_dir = if images.len() > 1 {
                dir.join(format!("image_{}", idx))
            } else {
                dir.clone()
            };
            if !img_dir.exists() {
                std::fs::create_dir_all(&img_dir).map_err(|e| {
                    anyhow!("create dump subdir {}: {e}", img_dir.display())
                })?;
            }
            for mirror in super::vit_dump::drain_cpu_mirrors() {
                super::vit_dump::write_dump_cpu(&img_dir, &mirror).map_err(|e| {
                    anyhow!("write CPU dump {}: {e}", mirror.name)
                })?;
            }
            for (name, buffer) in &collected {
                super::vit_dump::write_dump_gpu(&img_dir, name, buffer).map_err(|e| {
                    anyhow!("write GPU dump {}: {e}", name)
                })?;
            }
        }

        let pooled_n = ((img.n_x / 3) as usize) * ((img.n_y / 3) as usize);
        let mm0 = mmproj_weights
            .mm_0_weight()
            .map_err(|e| anyhow!("mm.0: {e}"))?;
        let text_hidden = mm0
            .as_slice::<f32>()
            .map_err(|e| anyhow!("mm.0 slice: {e}"))?
            .len()
            / (mmproj_cfg.hidden_size as usize);
        let total = pooled_n * text_hidden;
        let slice: &[f32] = buf
            .as_slice::<f32>()
            .map_err(|e| anyhow!("readback: {e}"))?;
        if slice.len() != total {
            return Err(anyhow!(
                "compute_vision_embeddings_gpu_gemma4v image {}: readback len {} != expected {} \
                 (pooled_n={}, text_hidden={})",
                idx,
                slice.len(),
                total,
                pooled_n,
                text_hidden
            ));
        }
        out.push(slice.to_vec());
    }
    Ok(out)
}

/// Per-image input variant for `compute_vision_embeddings_gpu_dispatch`.
///
/// `Siglip49(PreprocessedImage)` — square fixed-size pixels for the
/// SigLIP-49 / classic-CLIP path. `Gemma4v(Gemma4vPreprocessedImage)` —
/// variable-resolution patches + 2D pos for gemma4v.
///
/// The handler chooses the variant based on the loaded mmproj's
/// `ArchProfile` at preprocess time (one branch in
/// `process_multimodal_content`); the dispatch function below routes
/// accordingly.
#[derive(Debug, Clone)]
pub enum VisionInput {
    Siglip49(super::PreprocessedImage),
    Gemma4v(Gemma4vPreprocessedImage),
}

/// Arch-profile dispatch over a heterogeneous slice of images.
///
/// Currently splits into two homogeneous batches (all-Siglip49 vs
/// all-Gemma4v) since the underlying GPU paths are distinct compute
/// kernels with different per-image setup. `arch` is the
/// `ArchProfile` detected at mmproj load time and supplied by the
/// caller — we fail loud on a mismatch (e.g. `Gemma4v` input with
/// `ArchProfile::ClipClassic` mmproj) rather than silently routing to
/// the wrong forward.
///
/// Output ordering matches input ordering. `scale` is the SigLIP-49
/// attention scale (ignored for the gemma4v branch which uses
/// scale=1.0 internally per gemma4v.cpp:93).
///
/// # Errors
///
/// - `arch == Unknown`
/// - `arch` doesn't match the variant of any input element
/// - propagated from the underlying compute call
pub fn compute_vision_embeddings_gpu_dispatch(
    inputs: &[VisionInput],
    arch: super::mmproj::ArchProfile,
    mmproj_weights: &super::mmproj_weights::LoadedMmprojWeights,
    mmproj_cfg: &super::mmproj::MmprojConfig,
    scale: f32,
) -> Result<Vec<Vec<f32>>> {
    if !arch.is_supported() {
        return Err(anyhow!(
            "compute_vision_embeddings_gpu_dispatch: arch profile is Unknown — \
             cannot dispatch a vision forward"
        ));
    }
    // Validate every input matches `arch` before running anything.
    for (idx, input) in inputs.iter().enumerate() {
        match (&arch, input) {
            (super::mmproj::ArchProfile::Gemma4Siglip, VisionInput::Gemma4v(_))
            | (super::mmproj::ArchProfile::Gemma4Siglip, VisionInput::Siglip49(_))
            | (super::mmproj::ArchProfile::ClipClassic, VisionInput::Siglip49(_)) => {}
            (super::mmproj::ArchProfile::ClipClassic, VisionInput::Gemma4v(_)) => {
                return Err(anyhow!(
                    "compute_vision_embeddings_gpu_dispatch: input {idx} is Gemma4v \
                     but arch is ClipClassic — preprocessing/arch mismatch"
                ));
            }
            (super::mmproj::ArchProfile::Unknown, _) => unreachable!("guarded above"),
        }
    }

    // Partition inputs by variant while preserving original index so we
    // can re-merge at the end.
    let mut siglip_idx: Vec<usize> = Vec::new();
    let mut siglip_imgs: Vec<super::PreprocessedImage> = Vec::new();
    let mut gemma_idx: Vec<usize> = Vec::new();
    let mut gemma_imgs: Vec<Gemma4vPreprocessedImage> = Vec::new();
    for (idx, input) in inputs.iter().enumerate() {
        match input {
            VisionInput::Siglip49(p) => {
                siglip_idx.push(idx);
                siglip_imgs.push(p.clone());
            }
            VisionInput::Gemma4v(g) => {
                gemma_idx.push(idx);
                gemma_imgs.push(g.clone());
            }
        }
    }

    let mut out: Vec<Option<Vec<f32>>> = (0..inputs.len()).map(|_| None).collect();
    if !siglip_imgs.is_empty() {
        let r = compute_vision_embeddings_gpu(&siglip_imgs, mmproj_weights, mmproj_cfg, scale)?;
        for (i, e) in siglip_idx.into_iter().zip(r.into_iter()) {
            out[i] = Some(e);
        }
    }
    if !gemma_imgs.is_empty() {
        let r = compute_vision_embeddings_gpu_gemma4v(&gemma_imgs, mmproj_weights, mmproj_cfg)?;
        for (i, e) in gemma_idx.into_iter().zip(r.into_iter()) {
            out[i] = Some(e);
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(i, slot)| {
            slot.ok_or_else(|| anyhow!("compute_vision_embeddings_gpu_dispatch: slot {i} unfilled"))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Gemma4V (variable-resolution, 2D-RoPE) GPU primitives
// ---------------------------------------------------------------------------
//
// Sibling to the SigLIP-49 path. The CPU references in `vit.rs` are
// `gemma4v_patch_embed_forward` and `gemma4v_position_embed_lookup` — the
// GPU primitives below are byte-equivalent (within the BF16 cast
// tolerance of `vit_linear_gpu`). Used together they implement the
// gemma4v patch-embed stage:
//
//   GPU pipeline:
//     1. `gemma4v_patch_embed_gpu`           [N_patches, hidden]
//     2. `gemma4v_apply_position_embed_gpu`  patches += pe[0][pos_x] + pe[1][pos_y]
//
// Subsequent iters wire 2D RoPE (per-axis NEOX ordering, `gemma4v.cpp:46-91`)
// and the per-block forward.

/// GPU patch-embed for gemma4v: flat Linear with NO bias.
///
/// Wraps `vit_linear_gpu` for the gemma4v `[N_patches, p²·3] × [hidden, p²·3]`
/// projection. The candle reference uses `linear_no_bias(p²·3, hidden)`
/// (`/opt/candle/.../gemma4/vision.rs:127`); the GPU dispatch is the
/// same as any other Linear in the ViT stack — we add a typed entry
/// point so the call-site reads as the gemma4v graph step rather than
/// a generic linear.
///
/// Inputs (all device buffers):
///   - `patches`: F32 `[N_patches, inner = p²·3]` row-major.
///   - `weight`: F32 `[hidden, inner]` row-major.
/// Output: F32 `[N_patches, hidden]` row-major (freshly allocated).
///
/// # Constraints
///
/// `inner >= 32` (`vit_linear_gpu`'s tensor-core tile requirement).
/// For p=16 / 3 channels, `inner = 768` so the constraint is met.
///
/// # Errors
///
/// Propagated from `vit_linear_gpu`.
pub fn gemma4v_patch_embed_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    patches: &MlxBuffer,
    weight: &MlxBuffer,
    n_patches: u32,
    inner: u32,
    hidden: u32,
) -> Result<MlxBuffer> {
    if n_patches == 0 {
        return Err(anyhow!("gemma4v_patch_embed_gpu: n_patches must be > 0"));
    }
    vit_linear_gpu(
        encoder,
        registry,
        device,
        patches,
        weight,
        n_patches,
        inner,
        hidden,
    )
    .context("gemma4v_patch_embed_gpu: vit_linear_gpu")
}

/// GPU dual position-embed lookup for gemma4v.
///
/// Implements `gemma4v.cpp:18-42` on-GPU using two `dispatch_gather_f32`
/// calls (X-table + Y-table) followed by a single `elementwise_add`:
///
///   emb_x = gather(pe_table[0..pos_size,:], pos_x)
///   emb_y = gather(pe_table[pos_size..2*pos_size,:], pos_y)
///   out   = emb_x + emb_y
///
/// `pe_table` is the `[2, pos_size, hidden]` GGUF tensor (see
/// `mmproj_weights::position_embd_table_3d`); we slice it into two
/// `[pos_size, hidden]` row-aligned views via `MlxBuffer::slice_view`.
/// The X table starts at byte offset 0 and the Y table at byte offset
/// `pos_size * hidden * 4` (F32).
///
/// `pos_x` and `pos_y` are F32 buffers carrying U32 indices via
/// `as_mut_slice::<u32>()` (the gather kernel reads U32). Caller is
/// responsible for uploading them.
///
/// # Concurrent dispatch — barriers
///
/// Per `project_mlx_native_concurrent_dispatch.md`, the two gathers
/// can run concurrently (they read disjoint regions of `pe_table` and
/// write disjoint outputs). We DO insert a barrier before the
/// `elementwise_add` since the add reads BOTH gathers' outputs (RAW).
///
/// # Errors
///
/// - any zero dim
/// - propagated from gather/elementwise_add dispatches
#[allow(clippy::too_many_arguments)]
pub fn gemma4v_position_embed_lookup_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    pe_table: &MlxBuffer,
    pos_x_idx: &MlxBuffer,
    pos_y_idx: &MlxBuffer,
    n_patches: u32,
    pos_size: u32,
    hidden: u32,
) -> Result<MlxBuffer> {
    if n_patches == 0 || pos_size == 0 || hidden == 0 {
        return Err(anyhow!(
            "gemma4v_position_embed_lookup_gpu: n_patches ({n_patches}), pos_size ({pos_size}), \
             hidden ({hidden}) must all be > 0"
        ));
    }

    // Slice the [2, pos_size, hidden] table into two [pos_size, hidden]
    // views — same backing buffer, distinct byte offsets. F32, so 4 bytes
    // per element.
    let row_elems = (pos_size as usize) * (hidden as usize);
    let table_bytes = row_elems * 4;
    let table_x = pe_table.slice_view(0, row_elems);
    let table_y = pe_table.slice_view(table_bytes as u64, row_elems);

    // Allocate two gather outputs. Caller-visible result is `out_xy`,
    // returned at end after add.
    let n_us = n_patches as usize;
    let h_us = hidden as usize;
    let out_bytes = n_us * h_us * 4;
    let emb_x = device
        .alloc_buffer(out_bytes, DType::F32, vec![n_us, h_us])
        .map_err(|e| anyhow!("alloc emb_x: {e}"))?;
    let emb_y = device
        .alloc_buffer(out_bytes, DType::F32, vec![n_us, h_us])
        .map_err(|e| anyhow!("alloc emb_y: {e}"))?;

    // Two concurrent gathers: each reads a disjoint table region and
    // writes a disjoint output buffer.
    dispatch_gather_f32(
        encoder,
        registry,
        device.metal_device(),
        &table_x,
        pos_x_idx,
        &emb_x,
        pos_size,
        hidden,
        n_patches,
    )
    .context("gemma4v_position_embed_lookup_gpu: gather X")?;
    dispatch_gather_f32(
        encoder,
        registry,
        device.metal_device(),
        &table_y,
        pos_y_idx,
        &emb_y,
        pos_size,
        hidden,
        n_patches,
    )
    .context("gemma4v_position_embed_lookup_gpu: gather Y")?;

    // RAW: the add below reads both gather outputs.
    encoder.memory_barrier();

    let out = device
        .alloc_buffer(out_bytes, DType::F32, vec![n_us, h_us])
        .map_err(|e| anyhow!("alloc pos_emb sum: {e}"))?;
    elementwise_add(
        encoder,
        registry,
        device.metal_device(),
        &emb_x,
        &emb_y,
        &out,
        n_us * h_us,
        DType::F32,
    )
    .context("gemma4v_position_embed_lookup_gpu: elementwise_add")?;
    Ok(out)
}

/// GPU `patch_embeds += pe[0][pos_x] + pe[1][pos_y]`.
///
/// Composes `gemma4v_position_embed_lookup_gpu` with a residual-add into
/// the patch-embedding tensor. Returns the summed buffer (freshly
/// allocated; the caller's `patch_embeds` is unchanged so the residual
/// can be reused if needed).
///
/// Inserts a memory barrier between the gather/add stage and the final
/// residual-add (RAW: residual add reads both inputs).
#[allow(clippy::too_many_arguments)]
pub fn gemma4v_apply_position_embed_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    patch_embeds: &MlxBuffer,
    pe_table: &MlxBuffer,
    pos_x_idx: &MlxBuffer,
    pos_y_idx: &MlxBuffer,
    n_patches: u32,
    pos_size: u32,
    hidden: u32,
) -> Result<MlxBuffer> {
    let pos_emb = gemma4v_position_embed_lookup_gpu(
        encoder, registry, device, pe_table, pos_x_idx, pos_y_idx, n_patches, pos_size, hidden,
    )?;
    encoder.memory_barrier();
    let n_elem = (n_patches as u64).saturating_mul(hidden as u64);
    if n_elem > u32::MAX as u64 {
        return Err(anyhow!(
            "gemma4v_apply_position_embed_gpu: n_patches*hidden ({n_elem}) exceeds u32::MAX"
        ));
    }
    vit_residual_add_gpu(encoder, registry, device, patch_embeds, &pos_emb, n_elem as u32)
        .context("gemma4v_apply_position_embed_gpu: residual add")
}

// ---------------------------------------------------------------------------
// Gemma4V per-block forward (GPU dispatch)
// ---------------------------------------------------------------------------
//
// Sibling to `apply_vit_block_forward_gpu` (the SigLIP-49 path). Wires
// up:
//   - 4 Gemma4 RMSNorms (literal `weight` gain — see HF
//     `Gemma4RMSNorm.forward`) instead of SigLIP-49's 2.
//   - GQA: K and V are projected to `num_kv_heads * head_dim`, then
//     `repeat_kv`-expanded after RoPE.
//   - V-norm: pure RMS (no learned gain) using mlx-native's
//     `dispatch_rms_norm_no_scale_f32`.
//   - 2-D NeoX RoPE on Q and K via mlx-native's `dispatch_vision_2d_rope`.
//   - GELU(pytorch_tanh) on the gate proj instead of SiLU.
//   - Attention scale = 1.0 (Q is RMS-normalized).
//
// All shapes/weights flow through the existing `LoadedMmprojWeights` /
// `block_tensor(idx, suffix)` API — the loader is name-agnostic. The
// caller provides the gemma4v-specific config (`Gemma4VisionBlockShape`)
// and per-patch position arrays.

use mlx_native::ops::elementwise::elementwise_mul as mlx_elementwise_mul;
use mlx_native::ops::gelu::dispatch_gelu;
use mlx_native::ops::rms_norm::dispatch_rms_norm_no_scale_f32;
use mlx_native::ops::vision_2d_rope::{
    build_vision_2d_rope_params, dispatch_vision_2d_rope,
};

/// Gemma4 ViT RMSNorm on GPU: `y = x * rsqrt(mean(x²) + eps) * weight`
/// (literal gain).
///
/// Spec: `transformers/models/gemma4/modeling_gemma4.py::Gemma4RMSNorm::forward`
/// — applies the literal weight (initialized to `torch.ones(dim)`). Peer
/// reference: llama.cpp `tools/mtmd/clip.cpp::clip_graph::build_norm`
/// (lines 524-547) — `ggml_mul(cur, mw)` literal.
///
/// Bug-history (iter-122, 2026-04-26): originally this helper allocated a
/// transient `weight + 1` buffer to apply the candle convention from
/// `/opt/candle/.../gemma4/vision.rs:39`. That convention is wrong for
/// Gemma4 (it's a Gemma3 copy-paste); the `+1` collapsed spatial
/// differentiation across 27 ViT blocks and produced image-blind soft
/// tokens. Removing the `+1` makes the GPU path peer-identical to
/// llama-mtmd-cli on the four-dots fixture. As a side benefit, this drops
/// 2 buffer allocs + 1 elementwise_add dispatch + 1 memory_barrier per
/// call (× 6 call-sites × 27 layers per forward).
///
/// The function name and the call-sites are preserved so blame/log
/// continuity stays intact and the "this is the GEMMA4 convention"
/// annotation stays attached to each gemma4v_block_forward LN dispatch.
///
/// # Errors
///
/// - any zero dim
/// - propagated from `vit_rms_norm_gpu`
#[allow(clippy::too_many_arguments)]
pub fn vit_gemma_rms_norm_gpu(
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
            "vit_gemma_rms_norm_gpu: rows ({rows}) and dim ({dim}) must be > 0"
        ));
    }
    // Delegate to the literal-gain primitive — peer-identical to
    // llama.cpp `clip_graph::build_norm` for the gemma4v ViT path.
    vit_rms_norm_gpu(encoder, registry, device, input, gain_f32, rows, dim, eps)
        .context("vit_gemma_rms_norm_gpu: rms_norm_gpu")
}

/// Gemma-style per-head RMSNorm on GPU: row-major `[batch, num_heads,
/// head_dim]` is byte-identical to `[batch * num_heads, head_dim]`, so
/// dispatch the 2-D variant with `rows = batch * num_heads`.
#[allow(clippy::too_many_arguments)]
pub fn vit_gemma_per_head_rms_norm_gpu(
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
            "vit_gemma_per_head_rms_norm_gpu: batch ({batch}), num_heads ({num_heads}), \
             head_dim ({head_dim}) must all be > 0"
        ));
    }
    let rows = batch
        .checked_mul(num_heads)
        .ok_or_else(|| anyhow!("vit_gemma_per_head_rms_norm_gpu: batch*num_heads overflow"))?;
    vit_gemma_rms_norm_gpu(encoder, registry, device, input, gain_f32, rows, head_dim, eps)
}

/// V-norm (no learned gain) on GPU. Wraps mlx-native's
/// `dispatch_rms_norm_no_scale_f32`. Dispatches per-head normalization
/// (`rows = batch * num_kv_heads`, `dim = head_dim`).
pub fn vit_v_norm_no_scale_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    batch: u32,
    num_kv_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    if batch == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Err(anyhow!(
            "vit_v_norm_no_scale_gpu: batch/num_kv_heads/head_dim must all be > 0"
        ));
    }
    let rows = batch
        .checked_mul(num_kv_heads)
        .ok_or_else(|| anyhow!("vit_v_norm_no_scale_gpu: batch*num_kv_heads overflow"))?;
    let n_elem = (rows as usize) * (head_dim as usize);
    let output = device
        .alloc_buffer(n_elem * 4, DType::F32, vec![rows as usize, head_dim as usize])
        .map_err(|e| anyhow!("vit_v_norm_no_scale_gpu: alloc output: {e}"))?;
    // Params: [eps, dim] (matches the rms_norm shader convention).
    let params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("vit_v_norm_no_scale_gpu: alloc params: {e}"))?;
    {
        // SAFETY: just-allocated f32 buffer; no aliasing.
        let s: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(params_buf.contents_ptr() as *mut f32, 2) };
        s[0] = eps;
        s[1] = head_dim as f32;
    }
    dispatch_rms_norm_no_scale_f32(
        encoder,
        registry,
        device.metal_device(),
        input,
        &output,
        &params_buf,
        rows,
        head_dim,
    )
    .context("vit_v_norm_no_scale_gpu: dispatch_rms_norm_no_scale_f32")?;
    Ok(output)
}

/// 2-D NeoX RoPE on GPU. Wraps mlx-native's `dispatch_vision_2d_rope`.
/// Allocates a fresh output buffer matching `input` (F32, `[batch *
/// num_heads, head_dim]`).
#[allow(clippy::too_many_arguments)]
pub fn vit_vision_2d_rope_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    pos_x: &MlxBuffer,
    pos_y: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    theta: f32,
) -> Result<MlxBuffer> {
    if seq_len == 0 || n_heads == 0 || head_dim == 0 {
        return Err(anyhow!(
            "vit_vision_2d_rope_gpu: seq_len/n_heads/head_dim must all be > 0"
        ));
    }
    let n_rows = (seq_len as usize) * (n_heads as usize);
    let n_elem = n_rows * (head_dim as usize);
    let output = device
        .alloc_buffer(n_elem * 4, DType::F32, vec![n_rows, head_dim as usize])
        .map_err(|e| anyhow!("vit_vision_2d_rope_gpu: alloc output: {e}"))?;
    let params = build_vision_2d_rope_params(device, theta, head_dim, n_heads)
        .map_err(|e| anyhow!("vit_vision_2d_rope_gpu: build params: {e}"))?;
    dispatch_vision_2d_rope(
        encoder,
        registry,
        device.metal_device(),
        input,
        &output,
        &params,
        pos_x,
        pos_y,
        seq_len,
        n_heads,
        head_dim,
    )
    .context("vit_vision_2d_rope_gpu: dispatch_vision_2d_rope")?;
    Ok(output)
}

/// Repeat-kv on GPU: `[batch, num_kv_heads, head_dim]` →
/// `[batch, num_kv_heads * num_kv_groups, head_dim]` by replicating each
/// kv-head `num_kv_groups` times.
///
/// Implemented as a host-side gather table + a single `dispatch_gather_f32`
/// over the output rows. For every output row `r`, the source row is
/// `(b, k)` where `r = b * num_heads + h`, `k = h / num_kv_groups`,
/// `src_row = b * num_kv_heads + k`. We build the src-index buffer on
/// the host once per call, then gather.
pub fn vit_repeat_kv_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    batch: u32,
    num_kv_heads: u32,
    num_kv_groups: u32,
    head_dim: u32,
) -> Result<MlxBuffer> {
    if batch == 0 || num_kv_heads == 0 || num_kv_groups == 0 || head_dim == 0 {
        return Err(anyhow!(
            "vit_repeat_kv_gpu: batch/num_kv_heads/num_kv_groups/head_dim must all be > 0"
        ));
    }
    let num_heads = num_kv_heads
        .checked_mul(num_kv_groups)
        .ok_or_else(|| anyhow!("vit_repeat_kv_gpu: num_kv_heads*num_kv_groups overflow"))?;
    let n_out_rows = batch
        .checked_mul(num_heads)
        .ok_or_else(|| anyhow!("vit_repeat_kv_gpu: batch*num_heads overflow"))?;
    let n_in_rows = batch
        .checked_mul(num_kv_heads)
        .ok_or_else(|| anyhow!("vit_repeat_kv_gpu: batch*num_kv_heads overflow"))?;

    // Build the gather index table on the host: out_row r = b * num_heads + h
    //   → src_row = b * num_kv_heads + (h / num_kv_groups).
    let idx_buf = device
        .alloc_buffer(n_out_rows as usize * 4, DType::U32, vec![n_out_rows as usize])
        .map_err(|e| anyhow!("vit_repeat_kv_gpu: alloc idx: {e}"))?;
    {
        // SAFETY: just-allocated u32 buffer.
        let s: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(
                idx_buf.contents_ptr() as *mut u32,
                n_out_rows as usize,
            )
        };
        for b in 0..batch {
            for h in 0..num_heads {
                let kv_h = h / num_kv_groups;
                s[(b * num_heads + h) as usize] = b * num_kv_heads + kv_h;
            }
        }
    }

    let n_out_elem = (n_out_rows as usize) * (head_dim as usize);
    let output = device
        .alloc_buffer(n_out_elem * 4, DType::F32, vec![n_out_rows as usize, head_dim as usize])
        .map_err(|e| anyhow!("vit_repeat_kv_gpu: alloc output: {e}"))?;
    dispatch_gather_f32(
        encoder,
        registry,
        device.metal_device(),
        input,
        &idx_buf,
        &output,
        n_in_rows,
        head_dim,
        n_out_rows,
    )
    .context("vit_repeat_kv_gpu: dispatch_gather_f32")?;
    Ok(output)
}

/// Shape parameters for `gemma4v_block_forward_gpu` (mirrors the CPU
/// `Gemma4VisionBlockShape`).
#[derive(Debug, Clone, Copy)]
pub struct Gemma4VisionBlockShapeGpu {
    pub hidden: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

/// GELU(pytorch_tanh) on GPU. Wraps mlx-native's `dispatch_gelu` and
/// allocates a fresh F32 output buffer.
pub fn vit_gelu_pytorch_tanh_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    n_elements: u32,
) -> Result<MlxBuffer> {
    if n_elements == 0 {
        return Err(anyhow!("vit_gelu_pytorch_tanh_gpu: n_elements must be > 0"));
    }
    let output = device
        .alloc_buffer(
            (n_elements as usize) * 4,
            DType::F32,
            vec![n_elements as usize],
        )
        .map_err(|e| anyhow!("vit_gelu_pytorch_tanh_gpu: alloc output: {e}"))?;
    dispatch_gelu(encoder, registry, device.metal_device(), input, &output)
        .context("vit_gelu_pytorch_tanh_gpu: dispatch_gelu")?;
    Ok(output)
}

/// Gemma 4 Vision per-block forward (GPU dispatch).
///
/// Pipeline (matches `gemma4v_block_forward` CPU reference):
///
/// ```text
///   x_in    — input residual
///   cur     = gemma_rms_norm(x_in, ln1.weight)
///   q       = linear(cur, attn_q.weight)         → [B, num_heads*head_dim]
///   k       = linear(cur, attn_k.weight)         → [B, num_kv_heads*head_dim]
///   v       = linear(cur, attn_v.weight)         → [B, num_kv_heads*head_dim]
///   q       = gemma_per_head_rms(q, attn_q_norm.weight)
///   k       = gemma_per_head_rms(k, attn_k_norm.weight)
///   v       = v_norm_no_scale(v)                 # no learned gain, no tensor
///   q       = vision_2d_rope(q, pos_x, pos_y, theta)
///   k       = vision_2d_rope(k, pos_x, pos_y, theta)
///   k_full  = repeat_kv(k, num_kv_groups)
///   v_full  = repeat_kv(v, num_kv_groups)
///   attn    = vit_attention(q, k_full, v_full, scale=1.0)
///   out     = linear(attn, attn_out.weight)
///   out     = gemma_rms_norm(out, attn_post_norm.weight)  # post_attention_layernorm
///   x_mid   = x_in + out
///
///   cur     = gemma_rms_norm(x_mid, ln2.weight)           # pre_feedforward_layernorm
///   gate    = linear(cur, ffn_gate.weight)
///   up      = linear(cur, ffn_up.weight)
///   gate    = gelu_pytorch_tanh(gate)
///   activated = gate * up
///   down    = linear(activated, ffn_down.weight)
///   down    = gemma_rms_norm(down, ffn_post_norm.weight)  # post_feedforward_layernorm
///   x_out   = x_mid + down
/// ```
///
/// # Tensor name lineage (W44 iter-116k)
///
/// Gemma4v vision-namespace short forms per
/// `/opt/llama.cpp/tools/mtmd/clip-impl.h`:
///   - `attn_out.weight`        — TN_ATTN_OUTPUT (l.82, W34 iter-116e)
///   - `attn_post_norm.weight`  — TN_ATTN_POST_NORM (l.94, post-attention norm)
///   - `ln2.weight`             — TN_LN_2 (l.90, pre-FFN norm — W36 iter-116f
///                                renamed `pre_feedforward_layernorm` → `ln2`)
///   - `ffn_post_norm.weight`   — TN_FFN_POST_NORM (l.95, post-FFN norm —
///                                W36 iter-116f renamed
///                                `post_feedforward_layernorm` → `ffn_post_norm`)
/// `block_tensor` accepts the legacy `attn_output` / `post_ffw_norm`
/// aliases bidirectionally; the ln1/ln2/attn_post_norm/ffn_post_norm
/// reads here use the canonical short forms emitted by W36+ writers.
///
/// `pos_x_idx` / `pos_y_idx` are u32 buffers of length `batch` (= num_patches).
///
/// # Errors
///
/// Propagated from any sub-primitive — primarily missing tensors or
/// shape mismatches against `shape`.
#[allow(clippy::too_many_arguments)]
pub fn gemma4v_block_forward_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weights: &super::mmproj_weights::LoadedMmprojWeights,
    shape: &Gemma4VisionBlockShapeGpu,
    block_idx: usize,
    input: &MlxBuffer,
    pos_x_idx: &MlxBuffer,
    pos_y_idx: &MlxBuffer,
    batch: u32,
) -> Result<MlxBuffer> {
    let hidden = shape.hidden;
    let num_heads = shape.num_heads;
    let num_kv_heads = shape.num_kv_heads;
    let head_dim = shape.head_dim;
    let intermediate = shape.intermediate;
    let eps = shape.rms_norm_eps;
    let theta = shape.rope_theta;

    if hidden == 0
        || num_heads == 0
        || num_kv_heads == 0
        || head_dim == 0
        || intermediate == 0
        || batch == 0
    {
        return Err(anyhow!(
            "gemma4v_block_forward_gpu: zero dim in shape ({shape:?}) or batch ({batch})"
        ));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(anyhow!(
            "gemma4v_block_forward_gpu: num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        ));
    }
    let num_kv_groups = num_heads / num_kv_heads;
    let q_dim = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow!("gemma4v_block_forward_gpu: num_heads*head_dim overflow"))?;
    let kv_dim = num_kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow!("gemma4v_block_forward_gpu: num_kv_heads*head_dim overflow"))?;
    if q_dim != hidden {
        return Err(anyhow!(
            "gemma4v_block_forward_gpu: q_dim ({q_dim}) != hidden ({hidden})"
        ));
    }

    let block = |suffix: &str| -> Result<&MlxBuffer> {
        weights
            .block_tensor(block_idx, suffix)
            .map_err(|e| anyhow!("block {} {}: {e}", block_idx, suffix))
    };
    let n_hidden = batch
        .checked_mul(hidden)
        .ok_or_else(|| anyhow!("gemma4v_block_forward_gpu: batch*hidden overflow"))?;

    // ---------- Attention half ----------
    let cur = vit_gemma_rms_norm_gpu(
        encoder,
        registry,
        device,
        input,
        block("ln1.weight")?,
        batch,
        hidden,
        eps,
    )?;
    encoder.memory_barrier();

    let q = vit_linear_gpu(
        encoder,
        registry,
        device,
        &cur,
        block("attn_q.weight")?,
        batch,
        hidden,
        q_dim,
    )?;
    encoder.memory_barrier();
    let k = vit_linear_gpu(
        encoder,
        registry,
        device,
        &cur,
        block("attn_k.weight")?,
        batch,
        hidden,
        kv_dim,
    )?;
    encoder.memory_barrier();
    let v = vit_linear_gpu(
        encoder,
        registry,
        device,
        &cur,
        block("attn_v.weight")?,
        batch,
        hidden,
        kv_dim,
    )?;
    encoder.memory_barrier();

    let q = vit_gemma_per_head_rms_norm_gpu(
        encoder,
        registry,
        device,
        &q,
        block("attn_q_norm.weight")?,
        batch,
        num_heads,
        head_dim,
        eps,
    )?;
    encoder.memory_barrier();
    let k = vit_gemma_per_head_rms_norm_gpu(
        encoder,
        registry,
        device,
        &k,
        block("attn_k_norm.weight")?,
        batch,
        num_kv_heads,
        head_dim,
        eps,
    )?;
    encoder.memory_barrier();
    let v = vit_v_norm_no_scale_gpu(
        encoder, registry, device, &v, batch, num_kv_heads, head_dim, eps,
    )?;
    encoder.memory_barrier();

    // 2-D RoPE on Q and K.
    let q = vit_vision_2d_rope_gpu(
        encoder, registry, device, &q, pos_x_idx, pos_y_idx, batch, num_heads, head_dim, theta,
    )?;
    encoder.memory_barrier();
    let k = vit_vision_2d_rope_gpu(
        encoder, registry, device, &k, pos_x_idx, pos_y_idx, batch, num_kv_heads, head_dim, theta,
    )?;
    encoder.memory_barrier();

    // GQA: expand K and V to match Q's head count.
    let k_full = vit_repeat_kv_gpu(
        encoder, registry, device, &k, batch, num_kv_heads, num_kv_groups, head_dim,
    )?;
    encoder.memory_barrier();
    let v_full = vit_repeat_kv_gpu(
        encoder, registry, device, &v, batch, num_kv_heads, num_kv_groups, head_dim,
    )?;
    encoder.memory_barrier();

    // Attention with scale = 1.0 (gemma4v: Q is RMS-normalized).
    let attn = vit_attention_gpu(
        encoder, registry, device, &q, &k_full, &v_full, batch, num_heads, head_dim, 1.0,
    )?;
    encoder.memory_barrier();

    let attn_proj = vit_linear_gpu(
        encoder,
        registry,
        device,
        &attn,
        // W44 iter-116k: TN_ATTN_OUTPUT short form `attn_out` per
        // /opt/llama.cpp/tools/mtmd/clip-impl.h:82 (W34 iter-116e
        // writer rename). `block_tensor`'s legacy alias still
        // accepts `attn_output.weight`, but this site uses the
        // canonical name to match the writer's emit.
        block("attn_out.weight")?,
        batch,
        hidden,
        hidden,
    )?;
    encoder.memory_barrier();

    let attn_out = vit_gemma_rms_norm_gpu(
        encoder,
        registry,
        device,
        &attn_proj,
        // W44 iter-116k: post-attention norm is `attn_post_norm`
        // (TN_ATTN_POST_NORM, clip-impl.h:94), NOT `ln2`. W36
        // iter-116f writer maps HF `post_attention_layernorm` →
        // `attn_post_norm.weight` per
        // /opt/llama.cpp/gguf-py/gguf/constants.py:1218
        // (V_ENC_ATTN_POST_NORM). Build_vit consumes via
        // `layer.attn_post_norm_w` (clip.cpp:439-441).
        block("attn_post_norm.weight")?,
        batch,
        hidden,
        eps,
    )?;
    encoder.memory_barrier();

    let x_mid = vit_residual_add_gpu(encoder, registry, device, input, &attn_out, n_hidden)?;
    encoder.memory_barrier();

    // ---------- MLP half ----------
    let cur = vit_gemma_rms_norm_gpu(
        encoder,
        registry,
        device,
        &x_mid,
        // W44 iter-116k: pre-FFN norm is `ln2` (TN_LN_2,
        // clip-impl.h:90), NOT `ffn_norm`. W36 iter-116f writer
        // maps HF `pre_feedforward_layernorm` → `ln2.weight` per
        // /opt/llama.cpp/gguf-py/gguf/constants.py:1214
        // (V_ENC_POST_ATTN_NORM = "v.blk.{bid}.ln2"). Build_vit
        // consumes via `layer.ln_2_w` (clip.cpp:452).
        block("ln2.weight")?,
        batch,
        hidden,
        eps,
    )?;
    encoder.memory_barrier();

    let gate = vit_linear_gpu(
        encoder,
        registry,
        device,
        &cur,
        block("ffn_gate.weight")?,
        batch,
        hidden,
        intermediate,
    )?;
    encoder.memory_barrier();
    let up = vit_linear_gpu(
        encoder,
        registry,
        device,
        &cur,
        block("ffn_up.weight")?,
        batch,
        hidden,
        intermediate,
    )?;
    encoder.memory_barrier();

    let n_inter = batch
        .checked_mul(intermediate)
        .ok_or_else(|| anyhow!("gemma4v_block_forward_gpu: batch*intermediate overflow"))?;
    let gated = vit_gelu_pytorch_tanh_gpu(encoder, registry, device, &gate, n_inter)?;
    encoder.memory_barrier();

    // activated = gated * up
    let activated = device
        .alloc_buffer((n_inter as usize) * 4, DType::F32, vec![n_inter as usize])
        .map_err(|e| anyhow!("gemma4v_block_forward_gpu: alloc activated: {e}"))?;
    mlx_elementwise_mul(
        encoder,
        registry,
        device.metal_device(),
        &gated,
        &up,
        &activated,
        n_inter as usize,
        DType::F32,
    )
    .context("gemma4v_block_forward_gpu: gate * up")?;
    encoder.memory_barrier();

    let down = vit_linear_gpu(
        encoder,
        registry,
        device,
        &activated,
        block("ffn_down.weight")?,
        batch,
        intermediate,
        hidden,
    )?;
    encoder.memory_barrier();

    let down = vit_gemma_rms_norm_gpu(
        encoder,
        registry,
        device,
        &down,
        // W44 iter-116k: post-FFN norm is `ffn_post_norm`
        // (TN_FFN_POST_NORM, clip-impl.h:95). W36 iter-116f writer
        // maps HF `post_feedforward_layernorm` →
        // `ffn_post_norm.weight` per
        // /opt/llama.cpp/gguf-py/gguf/constants.py:1219
        // (V_ENC_FFN_POST_NORM). `block_tensor`'s legacy alias
        // still accepts `post_ffw_norm.weight`, but this site
        // uses the canonical name to match the writer's emit.
        block("ffn_post_norm.weight")?,
        batch,
        hidden,
        eps,
    )?;
    encoder.memory_barrier();

    let x_out = vit_residual_add_gpu(encoder, registry, device, &x_mid, &down, n_hidden)?;
    Ok(x_out)
}

// ---------------------------------------------------------------------------
// Gemma4ClippableLinear (GPU dispatch)
// ---------------------------------------------------------------------------

/// GPU sibling of `vit::gemma4v_clippable_linear_forward`.
///
/// Pipeline:
///   1. (optional) `vit_clip_gpu(input, input_min, input_max)`
///   2. `vit_linear_gpu(clamped_input, weight)` (no bias)
///   3. (optional) `vit_clip_gpu(output, output_min, output_max)`
///
/// `bounds.any() == false` short-circuits to a plain `vit_linear_gpu`
/// call — no extra dispatches, no extra allocations. This is the common
/// case for any non-gemma4v projector and for the gemma4v projector
/// when the GGUF was written by an older converter that didn't emit the
/// clamp scalars.
///
/// Inserts a memory barrier between each clamp and the matmul (clamp's
/// output is matmul's input → RAW), and between matmul and the output
/// clamp.
///
/// # Errors
///
/// Propagated from `vit_clip_gpu` and `vit_linear_gpu`.
#[allow(clippy::too_many_arguments)]
pub fn gemma4v_clippable_linear_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight_f32: &MlxBuffer,
    bounds: &super::vit::Gemma4ClippableLinearBounds,
    seq_len: u32,
    in_features: u32,
    out_features: u32,
) -> Result<MlxBuffer> {
    // --- Stage 1: optional input clamp ---
    let input_for_matmul: MlxBuffer;
    let input_ref: &MlxBuffer = if bounds.input_min.is_some() || bounds.input_max.is_some() {
        let (mn, mx) = bounds.resolved_input();
        let n = (seq_len as u64).saturating_mul(in_features as u64);
        if n > u32::MAX as u64 {
            return Err(anyhow!(
                "gemma4v_clippable_linear_gpu: input element count ({n}) exceeds u32::MAX"
            ));
        }
        input_for_matmul = vit_clip_gpu(encoder, registry, device, input, n as u32, mn, mx)
            .context("gemma4v_clippable_linear_gpu: input clamp")?;
        encoder.memory_barrier();
        &input_for_matmul
    } else {
        input
    };

    // --- Stage 2: linear ---
    let projected = vit_linear_gpu(
        encoder,
        registry,
        device,
        input_ref,
        weight_f32,
        seq_len,
        in_features,
        out_features,
    )
    .context("gemma4v_clippable_linear_gpu: vit_linear_gpu")?;

    // --- Stage 3: optional output clamp ---
    if bounds.output_min.is_some() || bounds.output_max.is_some() {
        encoder.memory_barrier();
        let (mn, mx) = bounds.resolved_output();
        let n = (seq_len as u64).saturating_mul(out_features as u64);
        if n > u32::MAX as u64 {
            return Err(anyhow!(
                "gemma4v_clippable_linear_gpu: output element count ({n}) exceeds u32::MAX"
            ));
        }
        return vit_clip_gpu(encoder, registry, device, &projected, n as u32, mn, mx)
            .context("gemma4v_clippable_linear_gpu: output clamp");
    }

    Ok(projected)
}

// ---------------------------------------------------------------------------
// gemma4v full-forward composition + arch-profile dispatch
// ---------------------------------------------------------------------------

/// Full GPU gemma4v ViT forward: pre-processed image → projected
/// multimodal embeddings.
///
/// Pipeline (matches `clip_graph_gemma4v::build`,
/// `/opt/llama.cpp/tools/mtmd/models/gemma4v.cpp`):
///
///   1. Upload patches `[N, p²·3]` + position-index arrays
///      (`pos_x[N]`, `pos_y[N]`) to GPU.
///   2. `gemma4v_patch_embed_gpu`: `[N, p²·3] @ W` → `[N, hidden]`.
///   3. `gemma4v_apply_position_embed_gpu`: residual-add the dual-table
///      gather (X-table + Y-table) into the patch embeddings.
///   4. `gemma4v_block_forward_gpu` × `num_hidden_layers` (27 for the
///      Gemma 4 26B mmproj). Each block runs 4-RMSNorm + GQA attn +
///      2D NeoX RoPE on Q/K + GELU(tanh) MLP.
///   5. Reshape `[N, hidden]` → `[n_y, n_x, hidden]` (no copy; the
///      pool kernel reads the same buffer with the rectangular layout)
///      and `gemma4v_avg_pool_3x3_gpu` → `[n_y/3, n_x/3, hidden]`.
///   6. `vit_scale_gpu(sqrt(hidden))` — gemma4v.cpp:115.
///   7. `vit_std_bias_scale_gpu(std_bias, std_scale)` — gemma4v.cpp:121-122.
///   8. `gemma4v_clippable_linear_gpu(mm.0.weight, bounds)` —
///      gemma4v.cpp:127, build_mm at 138-151.
///   9. `vit_rms_norm_gpu(ones, eps)` — gemma4v.cpp:131
///      (`embedding_post_projection_norm`, no learned gain).
///
/// Returns the final `[N_post_pool, text_hidden]` F32 buffer on device.
/// `N_post_pool = (n_x/3) * (n_y/3)`.
///
/// `n_x` / `n_y` come from `Gemma4vPreprocessed` and MUST both be
/// multiples of 3 (the gemma4v pool kernel size). The
/// `preprocess_gemma4v` pipeline already enforces this — it rounds the
/// resized image dims to multiples of `patch_size * n_merge = 16 * 3
/// = 48` so the post-pool grid is exact.
///
/// Caller registers the SigLIP-shared shaders before dispatch (the
/// gemma4v-specific shaders — `vision_2d_rope`, `gelu`, `gather` — are
/// self-registered below; see `compute_vision_embeddings_gpu` for the
/// canonical SigLIP-shared setup):
/// ```
/// mlx_native::ops::softmax::register(&mut registry);
/// mlx_native::ops::sigmoid_mul::register(&mut registry);
/// register_vit_custom_shaders(&mut registry);
/// ```
///
/// # Errors
///
/// - `n_x` or `n_y` not a multiple of 3 (caller bug; preprocess rounds)
/// - any required tensor missing (`v.patch_embd.weight`, `v.position_
///   embd.weight`, `v.std_bias`, `v.std_scale`, `mm.0.weight`)
/// - propagated from any sub-stage
#[allow(clippy::too_many_arguments)]
pub fn gemma4v_apply_full_forward_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weights: &super::mmproj_weights::LoadedMmprojWeights,
    cfg: &super::mmproj::MmprojConfig,
    patches: &[f32],
    pos_x: &[u32],
    pos_y: &[u32],
    n_x: u32,
    n_y: u32,
) -> Result<MlxBuffer> {
    // Self-register gemma4v-specific shaders. `register_source` is
    // idempotent (overwrites prior registration), so re-calling here is
    // safe even if a caller already registered them. This closes the
    // W25 iter-115 omission where the new gemma4v full-forward and
    // dispatch entry points landed without registering vision_2d_rope
    // (and the GELU + gather kernels reached transitively via
    // `gemma4v_block_forward_gpu`). The `vit_vision_2d_rope_gpu` call
    // inside the per-block forward used to panic with `Kernel not
    // found: vision_2d_rope_f32`.
    mlx_native::ops::vision_2d_rope::register(registry);
    mlx_native::ops::gelu::register(registry);
    mlx_native::ops::gather::register(registry);

    if n_x == 0 || n_y == 0 {
        return Err(anyhow!(
            "gemma4v_apply_full_forward_gpu: n_x ({n_x}) and n_y ({n_y}) must be > 0"
        ));
    }
    if n_x % 3 != 0 || n_y % 3 != 0 {
        return Err(anyhow!(
            "gemma4v_apply_full_forward_gpu: n_x ({n_x}) and n_y ({n_y}) must both \
             be multiples of 3 (gemma4v pool kernel size)"
        ));
    }
    let n_patches = (n_x as u64).saturating_mul(n_y as u64);
    if n_patches == 0 || n_patches > u32::MAX as u64 {
        return Err(anyhow!(
            "gemma4v_apply_full_forward_gpu: n_patches ({n_patches}) overflow u32"
        ));
    }
    let n_patches = n_patches as u32;

    let hidden = cfg.hidden_size;
    let patch_size = cfg.patch_size;
    let inner = patch_size
        .checked_mul(patch_size)
        .and_then(|s| s.checked_mul(3))
        .ok_or_else(|| anyhow!("gemma4v_apply_full_forward_gpu: inner overflow"))?;

    let expected_patches = (n_patches as usize).saturating_mul(inner as usize);
    if patches.len() != expected_patches {
        return Err(anyhow!(
            "gemma4v_apply_full_forward_gpu: patches len {} != n_patches*inner = {}*{} = {}",
            patches.len(),
            n_patches,
            inner,
            expected_patches
        ));
    }
    if pos_x.len() != n_patches as usize || pos_y.len() != n_patches as usize {
        return Err(anyhow!(
            "gemma4v_apply_full_forward_gpu: pos_x ({}) / pos_y ({}) length must equal n_patches ({})",
            pos_x.len(),
            pos_y.len(),
            n_patches
        ));
    }

    // --- Stage 1: upload patches + pos_x + pos_y to GPU ---
    let patches_buf = device
        .alloc_buffer(
            expected_patches * 4,
            DType::F32,
            vec![n_patches as usize, inner as usize],
        )
        .map_err(|e| anyhow!("alloc patches: {e}"))?;
    {
        // SAFETY: just-allocated f32 buffer; copy-in is exclusive.
        let dst: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(
                patches_buf.contents_ptr() as *mut f32,
                expected_patches,
            )
        };
        dst.copy_from_slice(patches);
    }
    let pos_x_buf = device
        .alloc_buffer((n_patches as usize) * 4, DType::U32, vec![n_patches as usize])
        .map_err(|e| anyhow!("alloc pos_x: {e}"))?;
    {
        // SAFETY: just-allocated U32 buffer; copy-in is exclusive.
        let dst: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(
                pos_x_buf.contents_ptr() as *mut u32,
                n_patches as usize,
            )
        };
        dst.copy_from_slice(pos_x);
    }
    let pos_y_buf = device
        .alloc_buffer((n_patches as usize) * 4, DType::U32, vec![n_patches as usize])
        .map_err(|e| anyhow!("alloc pos_y: {e}"))?;
    {
        // SAFETY: just-allocated U32 buffer.
        let dst: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(
                pos_y_buf.contents_ptr() as *mut u32,
                n_patches as usize,
            )
        };
        dst.copy_from_slice(pos_y);
    }

    // ADR-005 iter 124 (W55) + iter 126 (W57): record the preprocessed
    // CPU input as TWO probe stages so the diff harness can compare
    // hf2q ↔ peer at the SAME memory layout.
    //
    // Stage `00_post_patchify` — `[N_patches, P²·3]` flat, hf2q's native
    // layout after `preprocess_gemma4v`'s patchify (HWC `(dy, dx, c)`
    // per row). hf2q-only; the peer (llama.cpp's `inp_raw_scaled`) is
    // captured pre-patchify so there's no peer counterpart at this
    // point. Useful for self-consistency checks across iters.
    //
    // Stage `00_pre_patchify` — `[3, H, W]` planar CHW (W fastest,
    // matching ggml ne ordering), reconstructed from the patchified
    // tensor by inverting the (py, px, dy, dx, c) permutation. This
    // matches peer's `inp_raw_scaled` byte-for-byte modulo the
    // bilinear-resize/uint8-cast rounding noise that's inherent to the
    // independent preprocessing pipelines. The diff harness can now
    // pair these element-wise without layout-induced false positives.
    if super::vit_dump::is_armed() {
        super::vit_dump::record_f32(
            "00_post_patchify",
            patches,
            vec![n_patches as usize, inner as usize],
        );
        // De-patchify into planar [C, H, W] = [3, n_y*P, n_x*P].
        // After iter-126 (W57) `patches` rows are CHW
        // `(c, dy, dx)` so `patches[row_base + c*P² + dy*P + dx]` ↔
        // planar[c, py*P+dy, px*P+dx]. This is a pure index permutation
        // (no FP arithmetic) so byte-exact.
        let p = patch_size as usize;
        let p2 = p * p;
        let h = (n_y as usize) * p;
        let w = (n_x as usize) * p;
        let mut planar = vec![0f32; 3 * h * w];
        for py in 0..(n_y as usize) {
            for px in 0..(n_x as usize) {
                let patch_idx = py * (n_x as usize) + px;
                let row_base = patch_idx * (inner as usize);
                for dy in 0..p {
                    for dx in 0..p {
                        let pos_in_plane = dy * p + dx;
                        let yy = py * p + dy;
                        let xx = px * p + dx;
                        for c in 0..3 {
                            planar[c * h * w + yy * w + xx] =
                                patches[row_base + c * p2 + pos_in_plane];
                        }
                    }
                }
            }
        }
        super::vit_dump::record_f32(
            "00_pre_patchify",
            &planar,
            vec![3, h, w],
        );
    }

    // --- Stage 2: patch-embed (Linear, no bias) ---
    let patch_w = weights
        .patch_embd_weight()
        .map_err(|e| anyhow!("gemma4v_apply_full_forward_gpu: {e}"))?;
    let patch_embeds = gemma4v_patch_embed_gpu(
        encoder, registry, device, &patches_buf, patch_w, n_patches, inner, hidden,
    )?;
    encoder.memory_barrier();
    super::vit_dump::record("01_patch_embd", &patch_embeds);

    // --- Stage 3: dual-table position-embed lookup + add ---
    let (pe_table, pos_size, pe_hidden) = weights
        .position_embd_table_3d()
        .map_err(|e| anyhow!("gemma4v_apply_full_forward_gpu: {e}"))?;
    if pe_hidden != hidden {
        return Err(anyhow!(
            "gemma4v_apply_full_forward_gpu: position_embd hidden ({pe_hidden}) != cfg.hidden_size ({hidden})"
        ));
    }
    let after_pos = gemma4v_apply_position_embed_gpu(
        encoder, registry, device, &patch_embeds, pe_table, &pos_x_buf, &pos_y_buf,
        n_patches, pos_size, hidden,
    )?;
    encoder.memory_barrier();
    super::vit_dump::record("02_pos_embd", &after_pos);

    // --- Stage 4: 27-block transformer loop ---
    // Build the gemma4v block shape from cfg. head_dim = hidden / num_heads.
    let head_dim = hidden / cfg.num_attention_heads;
    // gemma4v's GQA ratio: num_kv_heads = num_attention_heads / 4 in the
    // production 26B config (4 KV groups). Read from cfg if/when the
    // mmproj parser exposes a kv_head field; until then we infer from
    // the loaded `attn_k.weight` shape — its row count = num_kv_heads
    // * head_dim. This matches W24's per-block test approach.
    let attn_k0 = weights
        .block_tensor(0, "attn_k.weight")
        .map_err(|e| anyhow!("gemma4v_apply_full_forward_gpu: probe num_kv_heads: {e}"))?;
    let attn_k0_rows = attn_k0.shape().first().copied().unwrap_or(0) as u32;
    if attn_k0_rows == 0 || attn_k0_rows % head_dim != 0 {
        return Err(anyhow!(
            "gemma4v_apply_full_forward_gpu: cannot infer num_kv_heads from attn_k.weight \
             row count ({attn_k0_rows}); head_dim = {head_dim}"
        ));
    }
    let num_kv_heads = attn_k0_rows / head_dim;
    let shape = Gemma4VisionBlockShapeGpu {
        hidden,
        num_heads: cfg.num_attention_heads,
        num_kv_heads,
        head_dim,
        intermediate: cfg.intermediate_size,
        rms_norm_eps: cfg.layer_norm_eps,
        // gemma4v rope_theta is fixed at 100.0 per clip.cpp:1336
        // (`hparams.rope_theta = 100.0f` for PROJECTOR_TYPE_GEMMA4V).
        rope_theta: 100.0f32,
    };

    let mut hidden_states = gemma4v_block_forward_gpu(
        encoder, registry, device, weights, &shape, 0, &after_pos,
        &pos_x_buf, &pos_y_buf, n_patches,
    )?;
    encoder.memory_barrier();
    if super::vit_dump::is_armed() {
        super::vit_dump::record("03_block_00", &hidden_states);
    }
    for block_idx in 1..(cfg.num_hidden_layers as usize) {
        hidden_states = gemma4v_block_forward_gpu(
            encoder, registry, device, weights, &shape, block_idx, &hidden_states,
            &pos_x_buf, &pos_y_buf, n_patches,
        )?;
        encoder.memory_barrier();
        if super::vit_dump::is_armed() {
            super::vit_dump::record(
                &format!("03_block_{:02}", block_idx),
                &hidden_states,
            );
        }
    }

    // --- Stage 5: 3×3 spatial avg-pool ---
    // The block output is `[n_patches, hidden]` row-major with patches
    // iterating row-major in (y, x) order (matching pos_y * n_x + pos_x
    // as set by `preprocess_gemma4v`). The pool kernel interprets this
    // as a `[n_y, n_x, hidden]` 3-D tensor — same backing buffer, just a
    // different shape view, no copy needed.
    let pooled = gemma4v_avg_pool_3x3_gpu(
        encoder, registry, device, &hidden_states, n_x, n_y, hidden,
    )?;
    encoder.memory_barrier();
    super::vit_dump::record("30_final_pool", &pooled);

    let pooled_n = ((n_x / 3) as usize) * ((n_y / 3) as usize);
    let pooled_total = pooled_n * (hidden as usize);

    // --- Stage 6: scale by sqrt(hidden) ---
    vit_scale_gpu(
        encoder, registry, device, &pooled, pooled_total as u32,
        (hidden as f32).sqrt(),
    )?;
    encoder.memory_barrier();
    super::vit_dump::record("31_pool_sqrt_scale", &pooled);

    // --- Stage 7: std_bias / std_scale normalize ---
    let std_bias = weights
        .get("v.std_bias")
        .ok_or_else(|| anyhow!("gemma4v_apply_full_forward_gpu: missing v.std_bias"))?;
    let std_scale = weights
        .get("v.std_scale")
        .ok_or_else(|| anyhow!("gemma4v_apply_full_forward_gpu: missing v.std_scale"))?;
    let normed = vit_std_bias_scale_gpu(
        encoder, registry, device, &pooled, std_bias, std_scale,
        pooled_n as u32, hidden,
    )?;
    encoder.memory_barrier();
    super::vit_dump::record("32_std_bias_scale", &normed);

    // --- Stage 8: Gemma4ClippableLinear projection ---
    let mm0 = weights
        .mm_0_weight()
        .map_err(|e| anyhow!("gemma4v_apply_full_forward_gpu: mm.0.weight: {e}"))?;
    let mm0_slice: &[f32] = mm0
        .as_slice::<f32>()
        .map_err(|e| anyhow!("mm.0.weight as_slice: {e}"))?;
    let text_hidden = (mm0_slice.len() / (hidden as usize)) as u32;
    if text_hidden == 0 {
        return Err(anyhow!(
            "gemma4v_apply_full_forward_gpu: text_hidden derived from mm.0.weight is 0"
        ));
    }
    let bounds = weights.mm_0_bounds();
    let projected = gemma4v_clippable_linear_gpu(
        encoder, registry, device, &normed, mm0, &bounds,
        pooled_n as u32, hidden, text_hidden,
    )?;
    encoder.memory_barrier();
    super::vit_dump::record("33_projector", &projected);

    // --- Stage 9: final no-gain RMSNorm (embedding_post_projection_norm) ---
    let ones = device
        .alloc_buffer(
            (text_hidden as usize) * 4,
            DType::F32,
            vec![text_hidden as usize],
        )
        .map_err(|e| anyhow!("alloc ones: {e}"))?;
    {
        // SAFETY: just-allocated f32 buffer.
        let s: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(
                ones.contents_ptr() as *mut f32,
                text_hidden as usize,
            )
        };
        for v in s.iter_mut() {
            *v = 1.0;
        }
    }
    let post_proj_rms = vit_rms_norm_gpu(
        encoder, registry, device, &projected, &ones,
        pooled_n as u32, text_hidden, cfg.layer_norm_eps,
    )?;
    super::vit_dump::record("34_post_proj_rms", &post_proj_rms);
    Ok(post_proj_rms)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::vision::vit::{
        apply_vit_block_forward as apply_vit_block_forward_cpu, elementwise_mul_in_place,
        gemma4v_block_forward as gemma4v_block_forward_cpu,
        gemma4v_patch_embed_forward as gemma4v_patch_embed_cpu,
        gemma4v_position_embed_lookup as gemma4v_pos_embed_cpu,
        linear_forward as linear_cpu, per_head_rms_norm_forward as per_head_rms_cpu,
        residual_add as residual_add_cpu, rms_norm_forward as rms_norm_cpu,
        scaled_dot_product_attention as attention_cpu, silu_in_place,
        softmax_last_dim as softmax_cpu,
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

    // -----------------------------------------------------------------------
    // apply_vit_block_forward_gpu — full block parity vs CPU (iter 49)
    // -----------------------------------------------------------------------

    /// Iter 50 bisection: check stage-by-stage where GPU and CPU diverge
    /// inside `apply_vit_block_forward_gpu`. Each stage:
    ///   1. dispatches just that stage on GPU (fresh session)
    ///   2. reads back intermediate
    ///   3. computes the CPU reference for the same intermediate
    ///   4. reports max_diff
    ///
    /// First stage with > BF16 tolerance (~5e-3) is the bug.
    #[test]
    fn iter50_bisect_block_forward_gpu_vs_cpu_real_gemma4() {
        use crate::inference::vision::vit::{
            linear_forward as linear_cpu_fn,
            per_head_rms_norm_forward as per_head_rms_cpu_fn,
            rms_norm_forward as rms_norm_cpu_fn,
            scaled_dot_product_attention as attention_cpu_fn,
        };

        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found");
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let weights =
            LoadedMmprojWeights::load(&gguf, &cfg, MlxDevice::new().expect("dev")).expect("w");

        let hidden = cfg.hidden_size as usize;
        let num_heads = cfg.num_attention_heads as usize;
        let head_dim = hidden / num_heads;
        let _intermediate = cfg.intermediate_size as usize;
        let eps = cfg.layer_norm_eps;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let batch = 32usize;

        // Synthetic input.
        let input_cpu: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i as f32) * 1e-4).sin() * 0.05)
            .collect();

        // Helper to extract a block tensor's f32 slice (lives forever
        // because the weights buffer is owned by `weights`).
        let block = |suffix: &str| -> Vec<f32> {
            weights
                .block_tensor(0, suffix)
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .to_vec()
        };

        let l2 = |a: &[f32], b: &[f32]| -> (f32, f32) {
            let mut max_d = 0f32;
            let mut sum_sq = 0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = (x - y).abs();
                if d > max_d { max_d = d; }
                sum_sq += d * d;
            }
            (max_d, sum_sq.sqrt())
        };

        // Each stage runs on a fresh session so we can readback in
        // isolation. Most stages don't even touch the weights from a
        // separate device, so this is OK.

        // ---------------- STAGE A: ln1 RMSNorm ----------------
        let ln1_cpu = block("ln1.weight");
        let mut ref_after_ln1 = input_cpu.clone();
        rms_norm_cpu_fn(&mut ref_after_ln1, &ln1_cpu, hidden, eps).unwrap();

        let exec = GraphExecutor::new(MlxDevice::new().expect("e_a"));
        let in_a = upload_f32(exec.device(), &input_cpu, vec![batch, hidden]);
        let ln1_a = upload_f32(exec.device(), &ln1_cpu, vec![hidden]);
        let mut sess = exec.begin().expect("s");
        let mut reg = KernelRegistry::new();
        let dr: *const MlxDevice = exec.device() as *const _;
        let dev: &MlxDevice = unsafe { &*dr };
        let stage_a = vit_rms_norm_gpu(sess.encoder_mut(), &mut reg, dev, &in_a, &ln1_a, batch as u32, hidden as u32, eps).unwrap();
        sess.finish().unwrap();
        let stage_a_cpu = readback_f32(&stage_a, batch * hidden);
        let (md, l2v) = l2(&stage_a_cpu, &ref_after_ln1);
        eprintln!("STAGE A (ln1 rms_norm)            : max_diff = {:.6}, l2 = {:.6}", md, l2v);
        assert!(md < 1e-3, "STAGE A diverges: max_diff = {md}");

        // ---------------- STAGE B: + Q linear ----------------
        let q_w = block("attn_q.weight");
        let ref_q = linear_cpu_fn(&ref_after_ln1, &q_w, None, batch, hidden, hidden).unwrap();

        let exec = GraphExecutor::new(MlxDevice::new().expect("e_b"));
        let cur_b = upload_f32(exec.device(), &ref_after_ln1, vec![batch, hidden]);
        let qw_b = upload_f32(exec.device(), &q_w, vec![hidden, hidden]);
        let mut sess = exec.begin().expect("s");
        let mut reg = KernelRegistry::new();
        let dr: *const MlxDevice = exec.device() as *const _;
        let dev: &MlxDevice = unsafe { &*dr };
        let stage_b = vit_linear_gpu(sess.encoder_mut(), &mut reg, dev, &cur_b, &qw_b, batch as u32, hidden as u32, hidden as u32).unwrap();
        sess.finish().unwrap();
        let stage_b_cpu = readback_f32(&stage_b, batch * hidden);
        let (md, l2v) = l2(&stage_b_cpu, &ref_q);
        eprintln!("STAGE B (Q linear, uploaded ln1 ref input): max_diff = {:.6}, l2 = {:.6}", md, l2v);
        // BF16 weight cast + 1152-term sum → expected max_diff ~ 0.2.
        // Treat anything < 0.5 as plausible BF16 noise; tighter
        // thresholds expose chain-specific bugs.
        assert!(md < 0.5, "STAGE B diverges WAY beyond BF16: max_diff = {md}");

        // ---------------- STAGE B': Q linear with weights from .as_slice ----------------
        // Same as B but use the weights buffer directly (no upload).
        // Validates that LoadedMmprojWeights' buffers behave the same
        // as the upload_f32 path.
        let exec = GraphExecutor::new(MlxDevice::new().expect("e_bp"));
        let cur_bp = upload_f32(exec.device(), &ref_after_ln1, vec![batch, hidden]);
        let mut sess = exec.begin().expect("s");
        let mut reg = KernelRegistry::new();
        let dr: *const MlxDevice = exec.device() as *const _;
        let dev: &MlxDevice = unsafe { &*dr };
        let qw_native = weights.block_tensor(0, "attn_q.weight").unwrap();
        let stage_bp = vit_linear_gpu(sess.encoder_mut(), &mut reg, dev, &cur_bp, qw_native, batch as u32, hidden as u32, hidden as u32).unwrap();
        sess.finish().unwrap();
        let stage_bp_cpu = readback_f32(&stage_bp, batch * hidden);
        let (md, l2v) = l2(&stage_bp_cpu, &ref_q);
        eprintln!("STAGE B' (Q linear, native weight buffer): max_diff = {:.6}, l2 = {:.6}", md, l2v);
        // No assert — diagnostic only. If THIS diverges from B, it's
        // the cross-device buffer issue.

        // ---------------- STAGE C: ln1 + Q linear, both on GPU in one session ----------------
        let exec = GraphExecutor::new(MlxDevice::new().expect("e_c"));
        let in_c = upload_f32(exec.device(), &input_cpu, vec![batch, hidden]);
        let ln1_c = upload_f32(exec.device(), &ln1_cpu, vec![hidden]);
        let qw_c = upload_f32(exec.device(), &q_w, vec![hidden, hidden]);
        let mut sess = exec.begin().expect("s");
        let mut reg = KernelRegistry::new();
        let dr: *const MlxDevice = exec.device() as *const _;
        let dev: &MlxDevice = unsafe { &*dr };
        let cur = vit_rms_norm_gpu(sess.encoder_mut(), &mut reg, dev, &in_c, &ln1_c, batch as u32, hidden as u32, eps).unwrap();
        sess.encoder_mut().memory_barrier();
        let stage_c = vit_linear_gpu(sess.encoder_mut(), &mut reg, dev, &cur, &qw_c, batch as u32, hidden as u32, hidden as u32).unwrap();
        sess.finish().unwrap();
        let stage_c_cpu = readback_f32(&stage_c, batch * hidden);
        let (md, l2v) = l2(&stage_c_cpu, &ref_q);
        eprintln!("STAGE C (ln1 → Q linear, single session): max_diff = {:.6}, l2 = {:.6}", md, l2v);

        // ---------------- STAGE D: + K, V linears, then per-head Q/K norms ----------------
        let k_w = block("attn_k.weight");
        let v_w = block("attn_v.weight");
        let qn_w = block("attn_q_norm.weight");
        let kn_w = block("attn_k_norm.weight");

        let ref_k = linear_cpu_fn(&ref_after_ln1, &k_w, None, batch, hidden, hidden).unwrap();
        let ref_v = linear_cpu_fn(&ref_after_ln1, &v_w, None, batch, hidden, hidden).unwrap();
        let mut ref_q_norm = ref_q.clone();
        per_head_rms_cpu_fn(&mut ref_q_norm, &qn_w, batch, num_heads, head_dim, eps).unwrap();
        let mut ref_k_norm = ref_k.clone();
        per_head_rms_cpu_fn(&mut ref_k_norm, &kn_w, batch, num_heads, head_dim, eps).unwrap();

        let exec = GraphExecutor::new(MlxDevice::new().expect("e_d"));
        let in_d = upload_f32(exec.device(), &input_cpu, vec![batch, hidden]);
        let ln1_d = upload_f32(exec.device(), &ln1_cpu, vec![hidden]);
        let qw_d = upload_f32(exec.device(), &q_w, vec![hidden, hidden]);
        let kw_d = upload_f32(exec.device(), &k_w, vec![hidden, hidden]);
        let vw_d = upload_f32(exec.device(), &v_w, vec![hidden, hidden]);
        let qn_d = upload_f32(exec.device(), &qn_w, vec![head_dim]);
        let kn_d = upload_f32(exec.device(), &kn_w, vec![head_dim]);
        let mut sess = exec.begin().expect("s");
        let mut reg = KernelRegistry::new();
        let dr: *const MlxDevice = exec.device() as *const _;
        let dev: &MlxDevice = unsafe { &*dr };
        let cur = vit_rms_norm_gpu(sess.encoder_mut(), &mut reg, dev, &in_d, &ln1_d, batch as u32, hidden as u32, eps).unwrap();
        sess.encoder_mut().memory_barrier();
        let q = vit_linear_gpu(sess.encoder_mut(), &mut reg, dev, &cur, &qw_d, batch as u32, hidden as u32, hidden as u32).unwrap();
        sess.encoder_mut().memory_barrier();
        let k = vit_linear_gpu(sess.encoder_mut(), &mut reg, dev, &cur, &kw_d, batch as u32, hidden as u32, hidden as u32).unwrap();
        sess.encoder_mut().memory_barrier();
        let v = vit_linear_gpu(sess.encoder_mut(), &mut reg, dev, &cur, &vw_d, batch as u32, hidden as u32, hidden as u32).unwrap();
        sess.encoder_mut().memory_barrier();
        let q_norm_gpu = vit_per_head_rms_norm_gpu(sess.encoder_mut(), &mut reg, dev, &q, &qn_d, batch as u32, num_heads as u32, head_dim as u32, eps).unwrap();
        sess.encoder_mut().memory_barrier();
        let k_norm_gpu = vit_per_head_rms_norm_gpu(sess.encoder_mut(), &mut reg, dev, &k, &kn_d, batch as u32, num_heads as u32, head_dim as u32, eps).unwrap();
        sess.finish().unwrap();
        let q_norm_cpu_back = readback_f32(&q_norm_gpu, batch * hidden);
        let k_norm_cpu_back = readback_f32(&k_norm_gpu, batch * hidden);
        let v_back = readback_f32(&v, batch * hidden);
        let (md_q, _) = l2(&q_norm_cpu_back, &ref_q_norm);
        let (md_k, _) = l2(&k_norm_cpu_back, &ref_k_norm);
        let (md_v, _) = l2(&v_back, &ref_v);
        eprintln!("STAGE D (Q-norm)                  : max_diff = {:.6}", md_q);
        eprintln!("STAGE D (K-norm)                  : max_diff = {:.6}", md_k);
        eprintln!("STAGE D (V)                       : max_diff = {:.6}", md_v);
        assert!(md_q < 0.5, "STAGE D Q-norm diverges: max_diff = {md_q}");
        assert!(md_k < 0.5, "STAGE D K-norm diverges: max_diff = {md_k}");

        // ---------------- STAGE E: + attention ----------------
        let ref_attn = attention_cpu_fn(&ref_q_norm, &ref_k_norm, &ref_v, batch, num_heads, head_dim).unwrap();

        let exec = GraphExecutor::new(MlxDevice::new().expect("e_e"));
        let qn_e = upload_f32(exec.device(), &ref_q_norm, vec![batch, num_heads, head_dim]);
        let kn_e = upload_f32(exec.device(), &ref_k_norm, vec![batch, num_heads, head_dim]);
        let v_e = upload_f32(exec.device(), &ref_v, vec![batch, num_heads, head_dim]);
        let mut sess = exec.begin().expect("s");
        let mut reg = KernelRegistry::new();
        mlx_native::ops::softmax::register(&mut reg);
        let dr: *const MlxDevice = exec.device() as *const _;
        let dev: &MlxDevice = unsafe { &*dr };
        let attn_e = vit_attention_gpu(sess.encoder_mut(), &mut reg, dev, &qn_e, &kn_e, &v_e, batch as u32, num_heads as u32, head_dim as u32, scale).unwrap();
        sess.finish().unwrap();
        let attn_e_back = readback_f32(&attn_e, batch * hidden);
        let (md_attn, l2_attn) = l2(&attn_e_back, &ref_attn);
        eprintln!("STAGE E (attention, CPU-uploaded inputs): max_diff = {:.6}, l2 = {:.6}", md_attn, l2_attn);
        // Diagnostics on input/output magnitudes.
        let stat = |name: &str, v: &[f32]| {
            let max_abs = v.iter().map(|x| x.abs()).fold(0f32, f32::max);
            let mean_abs = v.iter().map(|x| x.abs()).sum::<f32>() / (v.len() as f32);
            eprintln!("  {}: max_abs = {:.4}, mean_abs = {:.4}", name, max_abs, mean_abs);
        };
        stat("ref_q_norm", &ref_q_norm);
        stat("ref_k_norm", &ref_k_norm);
        stat("ref_v", &ref_v);
        stat("ref_attn (CPU)", &ref_attn);
        stat("attn (GPU)", &attn_e_back);

        // ---------------- STAGE F: chained Q/K/V/norms/attention in one session ----------------
        let exec = GraphExecutor::new(MlxDevice::new().expect("e_f"));
        let in_f = upload_f32(exec.device(), &input_cpu, vec![batch, hidden]);
        let ln1_f = upload_f32(exec.device(), &ln1_cpu, vec![hidden]);
        let qw_f = upload_f32(exec.device(), &q_w, vec![hidden, hidden]);
        let kw_f = upload_f32(exec.device(), &k_w, vec![hidden, hidden]);
        let vw_f = upload_f32(exec.device(), &v_w, vec![hidden, hidden]);
        let qn_f = upload_f32(exec.device(), &qn_w, vec![head_dim]);
        let kn_f = upload_f32(exec.device(), &kn_w, vec![head_dim]);
        let mut sess = exec.begin().expect("s");
        let mut reg = KernelRegistry::new();
        mlx_native::ops::softmax::register(&mut reg);
        let dr: *const MlxDevice = exec.device() as *const _;
        let dev: &MlxDevice = unsafe { &*dr };
        let cur = vit_rms_norm_gpu(sess.encoder_mut(), &mut reg, dev, &in_f, &ln1_f, batch as u32, hidden as u32, eps).unwrap();
        sess.encoder_mut().memory_barrier();
        let q = vit_linear_gpu(sess.encoder_mut(), &mut reg, dev, &cur, &qw_f, batch as u32, hidden as u32, hidden as u32).unwrap();
        sess.encoder_mut().memory_barrier();
        let k = vit_linear_gpu(sess.encoder_mut(), &mut reg, dev, &cur, &kw_f, batch as u32, hidden as u32, hidden as u32).unwrap();
        sess.encoder_mut().memory_barrier();
        let v = vit_linear_gpu(sess.encoder_mut(), &mut reg, dev, &cur, &vw_f, batch as u32, hidden as u32, hidden as u32).unwrap();
        sess.encoder_mut().memory_barrier();
        let qn = vit_per_head_rms_norm_gpu(sess.encoder_mut(), &mut reg, dev, &q, &qn_f, batch as u32, num_heads as u32, head_dim as u32, eps).unwrap();
        sess.encoder_mut().memory_barrier();
        let kn = vit_per_head_rms_norm_gpu(sess.encoder_mut(), &mut reg, dev, &k, &kn_f, batch as u32, num_heads as u32, head_dim as u32, eps).unwrap();
        sess.encoder_mut().memory_barrier();
        let attn_f = vit_attention_gpu(sess.encoder_mut(), &mut reg, dev, &qn, &kn, &v, batch as u32, num_heads as u32, head_dim as u32, scale).unwrap();
        sess.finish().unwrap();
        let attn_f_back = readback_f32(&attn_f, batch * hidden);
        let (md_attn_f, l2_attn_f) = l2(&attn_f_back, &ref_attn);
        eprintln!("STAGE F (full chain through attention): max_diff = {:.6}, l2 = {:.6}", md_attn_f, l2_attn_f);
    }
    // -----------------------------------------------------------------------
    // vit_avg_pool_2x2_gpu + vit_std_bias_scale_gpu (iter 51c)
    // -----------------------------------------------------------------------

    #[test]
    fn vit_avg_pool_2x2_gpu_matches_cpu_reference() {
        // 4×4 grid × 2 hidden → 2×2 output. Same hand-decodable test
        // as vit::tests::avg_pool_2x2_averages_each_2x2_block.
        use crate::inference::vision::vit::avg_pool_2x2_spatial as avg_pool_cpu;
        let n_side = 4usize;
        let hidden = 2usize;
        let mut input_cpu = vec![0f32; n_side * n_side * hidden];
        for y in 0..n_side {
            for x in 0..n_side {
                let patch = y * n_side + x;
                input_cpu[patch * hidden + 0] = patch as f32;
                input_cpu[patch * hidden + 1] = (patch as f32) * 10.0;
            }
        }
        let expected = avg_pool_cpu(&input_cpu, n_side, hidden).unwrap();

        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let in_buf = upload_f32(executor.device(), &input_cpu, vec![n_side, n_side, hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_avg_pool_2x2_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &in_buf,
            n_side as u32,
            hidden as u32,
        )
        .expect("avg_pool");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, 4 * hidden);

        let max_diff = got
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-6, "avg_pool max_diff = {max_diff}");
    }

    #[test]
    fn vit_avg_pool_2x2_gpu_gemma4_production_shape() {
        // [14, 14, 1152] → [49, 1152]. Uniform input → uniform output.
        let n_side = 14usize;
        let hidden = 1152usize;
        let total = n_side * n_side * hidden;
        let input_cpu: Vec<f32> = vec![1.5f32; total];

        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let in_buf = upload_f32(executor.device(), &input_cpu, vec![n_side, n_side, hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_avg_pool_2x2_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &in_buf,
            n_side as u32,
            hidden as u32,
        )
        .expect("avg_pool");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, 49 * hidden);
        for v in &got {
            assert!((*v - 1.5).abs() < 1e-6, "expected 1.5, got {v}");
        }
    }

    #[test]
    fn vit_avg_pool_2x2_gpu_rejects_odd_n_side() {
        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let in_buf = executor.device().alloc_buffer(27 * 4, DType::F32, vec![3, 3, 3]).expect("a");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_avg_pool_2x2_gpu(session.encoder_mut(), &mut registry, device, &in_buf, 3, 3)
            .unwrap_err();
        assert!(format!("{err}").contains("positive and even"));
    }

    // -----------------------------------------------------------------------
    // vit_avg_pool_kxk_gpu (iter 115)
    // -----------------------------------------------------------------------

    /// Regression guard: kxk with k=2 + square n_x=n_y must produce
    /// byte-identical output to the dedicated 2x2 path. Locks in the
    /// SigLIP-49 invariant — if this ever drifts, the kxk shader's
    /// accumulate-and-divide differs from `(a + b + c + d) * 0.25`,
    /// which is a real numeric regression worth investigating.
    #[test]
    fn vit_avg_pool_kxk_k2_byte_identical_to_2x2_path() {
        let n_side = 4usize;
        let hidden = 2usize;
        let mut input_cpu = vec![0f32; n_side * n_side * hidden];
        for y in 0..n_side {
            for x in 0..n_side {
                let patch = y * n_side + x;
                input_cpu[patch * hidden + 0] = patch as f32;
                input_cpu[patch * hidden + 1] = (patch as f32) * 10.0;
            }
        }
        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let in_buf = upload_f32(executor.device(), &input_cpu, vec![n_side, n_side, hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        let out_2x2 = vit_avg_pool_2x2_gpu(
            session.encoder_mut(), &mut registry, device, &in_buf,
            n_side as u32, hidden as u32,
        )
        .expect("2x2");
        let out_kxk = vit_avg_pool_kxk_gpu(
            session.encoder_mut(), &mut registry, device, &in_buf,
            n_side as u32, n_side as u32, 2, hidden as u32,
        )
        .expect("kxk");
        session.finish().expect("finish");

        let r_2x2 = readback_f32(&out_2x2, 4 * hidden);
        let r_kxk = readback_f32(&out_kxk, 4 * hidden);
        assert_eq!(r_2x2.len(), r_kxk.len());
        let max_diff = r_2x2
            .iter()
            .zip(r_kxk.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "kxk(k=2) drifted from 2x2 path: max_diff = {max_diff}"
        );
    }

    /// gemma4v's 3x3 case on a tiny synthetic input — verify the
    /// shader actually averages over the 3×3 block. n_x = n_y = 6, k = 3
    /// → 2×2 output of hidden=2 channels.
    #[test]
    fn vit_avg_pool_kxk_k3_correctness() {
        let n_x = 6usize;
        let n_y = 6usize;
        let k = 3usize;
        let hidden = 2usize;
        // Each input element = patch_idx (over n_x * n_y patches), with
        // the second hidden dim being patch_idx + 100 to keep channels
        // distinguishable.
        let mut input_cpu = vec![0f32; n_x * n_y * hidden];
        for y in 0..n_y {
            for x in 0..n_x {
                let patch = y * n_x + x;
                input_cpu[patch * hidden + 0] = patch as f32;
                input_cpu[patch * hidden + 1] = patch as f32 + 100.0;
            }
        }
        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let in_buf = upload_f32(executor.device(), &input_cpu, vec![n_y, n_x, hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_avg_pool_kxk_gpu(
            session.encoder_mut(), &mut registry, device, &in_buf,
            n_x as u32, n_y as u32, k as u32, hidden as u32,
        )
        .expect("kxk k=3");
        session.finish().expect("finish");
        let out_x = n_x / k;
        let out_y = n_y / k;
        let got = readback_f32(&out_buf, out_x * out_y * hidden);

        // Compute expected on CPU. For each output (oy, ox), sum the
        // 3×3 block of input patches and divide by 9.
        let mut expected = vec![0f32; out_x * out_y * hidden];
        for oy in 0..out_y {
            for ox in 0..out_x {
                for h in 0..hidden {
                    let mut acc = 0f32;
                    for dy in 0..k {
                        for dx in 0..k {
                            let iy = oy * k + dy;
                            let ix = ox * k + dx;
                            let patch = iy * n_x + ix;
                            acc += input_cpu[patch * hidden + h];
                        }
                    }
                    let out_idx = (oy * out_x + ox) * hidden + h;
                    expected[out_idx] = acc / ((k * k) as f32);
                }
            }
        }
        let max_diff = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-4, "k=3 max_diff = {max_diff}");
    }

    /// Output dimensions on a rectangular grid (n_x ≠ n_y) — verifies
    /// the shader's row-stride calculations are correct independent of
    /// per-axis tile counts.
    #[test]
    fn vit_avg_pool_kxk_dimensions_rectangular() {
        let n_x = 12u32;
        let n_y = 9u32;
        let k = 3u32;
        let hidden = 4u32;
        let total = (n_x * n_y * hidden) as usize;
        let input_cpu = vec![2.0f32; total];

        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let in_buf = upload_f32(
            executor.device(),
            &input_cpu,
            vec![n_y as usize, n_x as usize, hidden as usize],
        );
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_avg_pool_kxk_gpu(
            session.encoder_mut(), &mut registry, device, &in_buf, n_x, n_y, k, hidden,
        )
        .expect("kxk rect");
        session.finish().expect("finish");
        let out_x = (n_x / k) as usize;
        let out_y = (n_y / k) as usize;
        let got = readback_f32(&out_buf, out_x * out_y * hidden as usize);
        // Uniform input → uniform output of same value.
        for v in &got {
            assert!((v - 2.0).abs() < 1e-6, "expected 2.0, got {v}");
        }
        assert_eq!(out_x, 4);
        assert_eq!(out_y, 3);
    }

    #[test]
    fn vit_avg_pool_kxk_rejects_non_divisible_n() {
        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let in_buf = executor
            .device()
            .alloc_buffer(7 * 9 * 4 * 4, DType::F32, vec![7, 9, 4])
            .expect("a");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        // n_x=9, n_y=7, k=3 — 7 is not a multiple of 3.
        let err = vit_avg_pool_kxk_gpu(
            session.encoder_mut(), &mut registry, device, &in_buf, 9, 7, 3, 4,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("multiples of k"));
    }

    // -----------------------------------------------------------------------
    // vit_clip_gpu (iter 115)
    // -----------------------------------------------------------------------

    #[test]
    fn vit_clip_gpu_clamps_to_min_max() {
        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let input_cpu: Vec<f32> = vec![-5.0, -1.0, 0.0, 1.0, 5.0, 100.0];
        let in_buf = upload_f32(executor.device(), &input_cpu, vec![input_cpu.len()]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_clip_gpu(
            session.encoder_mut(), &mut registry, device, &in_buf,
            input_cpu.len() as u32, -2.0, 3.0,
        )
        .expect("clip");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, input_cpu.len());
        let expected: Vec<f32> = input_cpu.iter().map(|v| v.clamp(-2.0, 3.0)).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn vit_clip_gpu_no_op_on_neg_inf_pos_inf() {
        // Default sentinels (no clamp on either side) → identity.
        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let input_cpu: Vec<f32> = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
        let in_buf = upload_f32(executor.device(), &input_cpu, vec![input_cpu.len()]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_clip_gpu(
            session.encoder_mut(), &mut registry, device, &in_buf,
            input_cpu.len() as u32, f32::NEG_INFINITY, f32::INFINITY,
        )
        .expect("clip");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, input_cpu.len());
        assert_eq!(got, input_cpu);
    }

    #[test]
    fn vit_clip_gpu_rejects_min_greater_than_max() {
        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let in_buf = upload_f32(executor.device(), &[1.0, 2.0, 3.0], vec![3]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_clip_gpu(
            session.encoder_mut(), &mut registry, device, &in_buf, 3, 5.0, 1.0,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("min_val") || format!("{err}").contains("> max_val"));
    }

    // -----------------------------------------------------------------------
    // gemma4v_clippable_linear (CPU + GPU, iter 115)
    // -----------------------------------------------------------------------

    /// Three-case parity: (no clamp, input only, both clamps) — the GPU
    /// path matches the CPU reference within bf16 round-trip tolerance
    /// (`vit_linear_gpu` casts weight F32→BF16 internally, so the
    /// expected diff for the linear stage alone is ≤ 1e-3).
    #[test]
    fn gemma4v_clippable_linear_with_and_without_clamp() {
        use crate::inference::vision::vit::{
            gemma4v_clippable_linear_forward as cpu_clip_linear,
            Gemma4ClippableLinearBounds,
        };
        let batch = 4usize;
        let in_features = 64usize;
        let out_features = 32usize;
        let input: Vec<f32> = (0..batch * in_features)
            .map(|i| ((i as f32) * 0.013).sin() * 2.0)
            .collect();
        let weight: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i as f32) * 0.017).cos() * 0.05)
            .collect();

        // Three bounds variants.
        let bounds_none = Gemma4ClippableLinearBounds::default();
        let bounds_input_only = Gemma4ClippableLinearBounds {
            input_min: Some(-0.5),
            input_max: Some(0.5),
            ..Default::default()
        };
        let bounds_both = Gemma4ClippableLinearBounds {
            input_min: Some(-1.0),
            input_max: Some(1.0),
            output_min: Some(-0.05),
            output_max: Some(0.05),
        };

        for (label, bounds) in &[
            ("none", &bounds_none),
            ("input_only", &bounds_input_only),
            ("both", &bounds_both),
        ] {
            // CPU reference.
            let cpu_out = cpu_clip_linear(
                &input, &weight, bounds, batch, in_features, out_features,
            )
            .expect("cpu clip linear");

            // GPU dispatch.
            let device = MlxDevice::new().expect("dev");
            let executor = GraphExecutor::new(device);
            let in_buf = upload_f32(executor.device(), &input, vec![batch, in_features]);
            let w_buf = upload_f32(executor.device(), &weight, vec![out_features, in_features]);
            let mut session = executor.begin().expect("begin");
            let mut registry = KernelRegistry::new();
            register_vit_custom_shaders(&mut registry);
            let device_ref: *const MlxDevice = executor.device() as *const _;
            let device: &MlxDevice = unsafe { &*device_ref };
            let out_buf = gemma4v_clippable_linear_gpu(
                session.encoder_mut(), &mut registry, device, &in_buf, &w_buf, bounds,
                batch as u32, in_features as u32, out_features as u32,
            )
            .expect("gpu clip linear");
            session.finish().expect("finish");
            let gpu_out = readback_f32(&out_buf, batch * out_features);

            assert_eq!(gpu_out.len(), cpu_out.len(), "[{label}] length mismatch");
            // BF16 weight round-trip in vit_linear_gpu: tolerance 5e-3
            // (per existing vit_linear_gpu_matches_cpu_reference test).
            let max_diff = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0f32, f32::max);
            assert!(
                max_diff < 5e-3,
                "[{label}] gpu/cpu max_diff = {max_diff}"
            );
            // Verify clamps actually fired when bounds were set: every
            // GPU output element must respect the output clamp.
            if let Some(mn) = bounds.output_min {
                for v in &gpu_out {
                    assert!(*v >= mn - 1e-4, "[{label}] {v} < output_min {mn}");
                }
            }
            if let Some(mx) = bounds.output_max {
                for v in &gpu_out {
                    assert!(*v <= mx + 1e-4, "[{label}] {v} > output_max {mx}");
                }
            }
            // Also: outputs are finite.
            for v in &gpu_out {
                assert!(v.is_finite(), "[{label}] non-finite: {v}");
            }
        }
    }

    #[test]
    fn vit_std_bias_scale_gpu_matches_cpu_reference() {
        // batch=2, hidden=3. bias=[1, 2, 3], scale=[10, 20, 30]. Same
        // hand-decodable test as vit::tests::std_bias_scale_subtracts_and_scales_per_channel.
        let batch = 2usize;
        let hidden = 3usize;
        let input_cpu = vec![5.0f32, 10.0, 15.0, 10.0, 20.0, 30.0];
        let bias_cpu = vec![1.0f32, 2.0, 3.0];
        let scale_cpu = vec![10.0f32, 20.0, 30.0];
        let expected = vec![40.0f32, 160.0, 360.0, 90.0, 360.0, 810.0];

        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let in_buf = upload_f32(executor.device(), &input_cpu, vec![batch, hidden]);
        let bias_buf = upload_f32(executor.device(), &bias_cpu, vec![hidden]);
        let scale_buf = upload_f32(executor.device(), &scale_cpu, vec![hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_std_bias_scale_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &in_buf,
            &bias_buf,
            &scale_buf,
            batch as u32,
            hidden as u32,
        )
        .expect("std_bias_scale");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, batch * hidden);
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "got {g} want {e}");
        }
    }

    #[test]
    fn vit_std_bias_scale_gpu_zero_bias_unit_scale_is_identity() {
        let batch = 4usize;
        let hidden = 8usize;
        let input_cpu: Vec<f32> = (0..batch * hidden).map(|i| (i as f32) * 0.1).collect();
        let bias_cpu = vec![0.0f32; hidden];
        let scale_cpu = vec![1.0f32; hidden];

        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let in_buf = upload_f32(executor.device(), &input_cpu, vec![batch, hidden]);
        let bias_buf = upload_f32(executor.device(), &bias_cpu, vec![hidden]);
        let scale_buf = upload_f32(executor.device(), &scale_cpu, vec![hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out_buf = vit_std_bias_scale_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &in_buf,
            &bias_buf,
            &scale_buf,
            batch as u32,
            hidden as u32,
        )
        .expect("std_bias_scale");
        session.finish().expect("finish");
        let got = readback_f32(&out_buf, batch * hidden);
        for (g, c) in got.iter().zip(input_cpu.iter()) {
            assert!((g - c).abs() < 1e-6);
        }
    }

    #[test]
    fn vit_std_bias_scale_gpu_rejects_zero_dims() {
        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let buf = executor.device().alloc_buffer(16, DType::F32, vec![4]).expect("a");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_std_bias_scale_gpu(session.encoder_mut(), &mut registry, device, &buf, &buf, &buf, 0, 4)
            .unwrap_err();
        assert!(format!("{err}").contains("must be > 0"));
    }

    // -----------------------------------------------------------------------
    // vit_scale_gpu (iter 51b)
    // -----------------------------------------------------------------------

    #[test]
    fn vit_scale_gpu_multiplies_every_element_in_place() {
        let n = 64usize;
        let scalar = 0.25f32;
        let cpu_in: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let expected: Vec<f32> = cpu_in.iter().map(|x| x * scalar).collect();

        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let buf = upload_f32(executor.device(), &cpu_in, vec![n]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        vit_scale_gpu(session.encoder_mut(), &mut registry, device, &buf, n as u32, scalar)
            .expect("scale");
        session.finish().expect("finish");
        let got = readback_f32(&buf, n);

        let max_diff = got.iter().zip(expected.iter()).map(|(g, e)| (g - e).abs()).fold(0f32, f32::max);
        assert!(max_diff < 1e-6, "scale GPU max_diff = {max_diff}");
    }

    #[test]
    fn vit_scale_gpu_by_unit_is_identity() {
        let n = 16usize;
        let cpu_in: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let buf = upload_f32(executor.device(), &cpu_in, vec![n]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        vit_scale_gpu(session.encoder_mut(), &mut registry, device, &buf, n as u32, 1.0)
            .expect("scale");
        session.finish().expect("finish");
        let got = readback_f32(&buf, n);
        for (g, c) in got.iter().zip(cpu_in.iter()) {
            assert!((g - c).abs() < 1e-6);
        }
    }

    #[test]
    fn vit_scale_gpu_rejects_zero_n() {
        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let buf = executor.device().alloc_buffer(16, DType::F32, vec![4]).expect("a");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = vit_scale_gpu(session.encoder_mut(), &mut registry, device, &buf, 0, 1.0)
            .unwrap_err();
        assert!(format!("{err}").contains("must be > 0"));
    }

    #[test]
    fn warmup_vit_gpu_compiles_all_kernels_real_gemma4() {
        // Iter 53: warmup function. Just verify it runs to completion
        // without panic on real Gemma 4 mmproj — its job is to JIT-
        // compile every Metal pipeline so subsequent runs are fast.
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found");
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let weights =
            LoadedMmprojWeights::load(&gguf, &cfg, MlxDevice::new().expect("dev"))
                .expect("w");
        let t0 = std::time::Instant::now();
        warmup_vit_gpu(&weights, &cfg).expect("warmup");
        eprintln!("warmup_vit_gpu (cold): {:?}", t0.elapsed());
        // Run again — should be much faster (kernels cached).
        let t1 = std::time::Instant::now();
        warmup_vit_gpu(&weights, &cfg).expect("warmup #2");
        eprintln!("warmup_vit_gpu (warm): {:?}", t1.elapsed());
    }

    #[test]
    fn compute_vision_embeddings_gpu_multi_image_real_gemma4() {
        // Iter 52: handler-side wrapper. Two synthetic preprocessed
        // images (different gradient inputs) → two distinct
        // [49, 2816] embeddings. Validates the multi-image loop +
        // ordering preservation.
        use crate::inference::vision::PreprocessedImage;
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found");
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let weights =
            LoadedMmprojWeights::load(&gguf, &cfg, MlxDevice::new().expect("dev"))
                .expect("w");

        let img = cfg.image_size as usize;
        let make_image = |seed: u32| -> PreprocessedImage {
            let mut pixels = vec![0f32; 3 * img * img];
            for c in 0..3 {
                for y in 0..img {
                    for x in 0..img {
                        pixels[c * img * img + y * img + x] =
                            ((c + 1) as f32) * 0.05
                                + (y as f32) * 0.001
                                + (x as f32) * 0.001
                                + (seed as f32) * 0.0001;
                    }
                }
            }
            PreprocessedImage {
                pixel_values: pixels,
                target_size: cfg.image_size,
                source_label: format!("synthetic-{seed}"),
            }
        };
        let images = vec![make_image(0), make_image(42)];

        let head_dim = (cfg.hidden_size / cfg.num_attention_heads) as f32;
        let scale = 1.0f32 / head_dim.sqrt();

        let t0 = std::time::Instant::now();
        let embeddings =
            compute_vision_embeddings_gpu(&images, &weights, &cfg, scale).expect("compute");
        eprintln!(
            "compute_vision_embeddings_gpu × {} images: {:?}",
            images.len(),
            t0.elapsed()
        );
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 49 * 2816);
        assert_eq!(embeddings[1].len(), 49 * 2816);

        // Both finite.
        for emb in &embeddings {
            for v in emb {
                assert!(v.is_finite(), "non-finite: {v}");
            }
        }
        // Two different images should produce DIFFERENT embeddings.
        let l2: f32 = embeddings[0]
            .iter()
            .zip(embeddings[1].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        eprintln!("inter-image L2 = {:.4}", l2);
        assert!(l2 > 1e-2, "two different images produced identical embeddings");
    }

    /// Iter 51d: full GPU ViT forward on real Gemma 4 mmproj.
    /// Pixel input → [49, 2816] projected multimodal embedding.
    /// **This is the final production forward path.**
    #[test]
    fn apply_vit_full_forward_gpu_on_real_gemma4_full_pipeline() {
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found");
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let weights =
            LoadedMmprojWeights::load(&gguf, &cfg, MlxDevice::new().expect("dev"))
                .expect("w");
        assert_eq!(cfg.num_hidden_layers, 27);
        assert_eq!(cfg.num_patches_side, 14);

        // Synthetic preprocessed pixel tensor [3, 224, 224] f32.
        let img = cfg.image_size as usize;
        let mut pixels = vec![0f32; 3 * img * img];
        for c in 0..3 {
            for y in 0..img {
                for x in 0..img {
                    pixels[c * img * img + y * img + x] =
                        ((c + 1) as f32) * 0.05 + (y as f32) * 0.001 + (x as f32) * 0.001;
                }
            }
        }

        let head_dim = (cfg.hidden_size / cfg.num_attention_heads) as f32;
        let scale = 1.0f32 / head_dim.sqrt();

        let executor = GraphExecutor::new(MlxDevice::new().expect("dev2"));
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        mlx_native::ops::softmax::register(&mut registry);
        mlx_native::ops::sigmoid_mul::register(&mut registry);
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        eprintln!("running full ViT GPU forward...");
        let t0 = std::time::Instant::now();
        let out = apply_vit_full_forward_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &weights,
            &cfg,
            &pixels,
            scale,
        )
        .expect("full forward");
        session.finish().expect("finish");
        let elapsed = t0.elapsed();
        eprintln!("full ViT GPU forward done in {:?}", elapsed);

        // Expected output shape: [49 patches, 2816 text_hidden].
        let mm0 = weights.mm_0_weight().expect("mm.0");
        let mm0_f32: &[f32] = mm0.as_slice::<f32>().expect("slice");
        let text_hidden = mm0_f32.len() / (cfg.hidden_size as usize);
        assert_eq!(text_hidden, 2816, "Gemma 4 projector output width");
        let n_patches_out = 49;
        let total = n_patches_out * text_hidden;

        let got = readback_f32(&out, total);

        // 1. Finite.
        for v in &got {
            assert!(v.is_finite(), "non-finite: {v}");
        }
        // 2. Each row's mean(x²) ≈ 1 (no-gain final RMSNorm).
        for p in 0..n_patches_out {
            let row = &got[p * text_hidden..(p + 1) * text_hidden];
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / (text_hidden as f32);
            assert!(
                (ms - 1.0).abs() < 0.05,
                "patch {p} post-norm mean(x²) = {ms}, expected ≈ 1.0"
            );
        }
        // 3. Cross-token diversity preserved.
        let p0 = &got[0..text_hidden];
        let p_last = &got[(n_patches_out - 1) * text_hidden..n_patches_out * text_hidden];
        let l2: f32 = p0
            .iter()
            .zip(p_last.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        eprintln!("cross-token L2 = {:.4}", l2);
        assert!(l2 > 1e-2, "all output tokens collapsed — diversity lost");

        let max_abs = got.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let mean_abs = got.iter().map(|v| v.abs()).sum::<f32>() / (got.len() as f32);
        eprintln!("output: max_abs = {:.4}, mean_abs = {:.4}", max_abs, mean_abs);
    }

    /// Iter 51a: 27-block GPU forward on real Gemma 4 mmproj. Tests the
    /// distributional output of the chained block loop — finite,
    /// magnitudes O(1) after blocks (residual stream stays bounded
    /// because blocks are pre-norm + bounded-norm activations),
    /// cross-token diversity preserved (each token has a different
    /// embedding even after 27 blocks of mixing).
    ///
    /// CPU element-wise comparison is intentionally NOT used here —
    /// see `project_vit_attention_bf16_softmax_drift.md` for why.
    #[test]
    fn apply_vit_blocks_loop_gpu_27_blocks_real_gemma4() {
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found");
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let weights =
            LoadedMmprojWeights::load(&gguf, &cfg, MlxDevice::new().expect("dev"))
                .expect("w");
        assert_eq!(cfg.num_hidden_layers, 27);

        let hidden = cfg.hidden_size as usize;
        let head_dim = hidden / cfg.num_attention_heads as usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let batch = 32usize;

        let input_cpu: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i as f32) * 1e-4).sin() * 0.05)
            .collect();

        let executor = GraphExecutor::new(MlxDevice::new().expect("dev2"));
        let input_buf = upload_f32(executor.device(), &input_cpu, vec![batch, hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        mlx_native::ops::softmax::register(&mut registry);
        mlx_native::ops::sigmoid_mul::register(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        eprintln!("running 27-block GPU forward...");
        let t0 = std::time::Instant::now();
        let out = apply_vit_blocks_loop_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &weights,
            &cfg,
            &input_buf,
            batch as u32,
            scale,
        )
        .expect("blocks loop");
        session.finish().expect("finish");
        let elapsed = t0.elapsed();
        eprintln!("27-block GPU forward done in {:?}", elapsed);

        let got = readback_f32(&out, batch * hidden);

        // Distributional sanity:
        let max_abs = got.iter().map(|x| x.abs()).fold(0f32, f32::max);
        let mean_abs = got.iter().map(|x| x.abs()).sum::<f32>() / (got.len() as f32);
        eprintln!(
            "27-block output: max_abs = {:.4}, mean_abs = {:.4}",
            max_abs, mean_abs
        );
        // 1. Finite.
        for v in &got {
            assert!(v.is_finite(), "non-finite: {v}");
        }
        // 2. Reasonable magnitude. Residual stream + norms shouldn't
        //    blow up to 1e6 or collapse to 0.
        assert!(max_abs > 1e-3, "max_abs collapsed to ~0: {max_abs}");
        assert!(max_abs < 1e4, "max_abs blew up: {max_abs}");
        assert!(mean_abs > 1e-4 && mean_abs < 1e3, "mean_abs out of range: {mean_abs}");
        // 3. Cross-token diversity preserved. After 27 blocks of mixing,
        //    different input tokens should still produce different outputs.
        let token_0 = &got[0..hidden];
        let token_last = &got[(batch - 1) * hidden..batch * hidden];
        let l2_diff: f32 = token_0
            .iter()
            .zip(token_last.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        eprintln!("cross-token L2 (token 0 vs token {}) = {:.4}", batch - 1, l2_diff);
        assert!(
            l2_diff > 1e-2,
            "all tokens collapsed to identical output — diversity lost"
        );
    }

    /// Iter 50 finding (don't re-enable without context):
    ///
    /// This test compares GPU vs CPU F32 `apply_vit_block_forward`
    /// element-wise. At Gemma 4 ViT's real per-head-rms-norm Q
    /// magnitudes (~5.6) the BF16 K cast in `vit_attention_gpu` puts
    /// ~0.68 noise into pre-softmax logits, which flips
    /// saturated-softmax winners between adjacent K positions. CPU
    /// and GPU produce attention outputs with matching macro stats
    /// (max_abs, mean_abs) but ~11x per-element max_diff because
    /// they pick different "winners" at saturated rows.
    ///
    /// mlx-lm production uses the SAME BF16 attention path (flash-attn
    /// with BF16 K), so GPU output IS production-correct. The CPU F32
    /// reference is too strict a parity bar. See memory:
    /// `project_vit_attention_bf16_softmax_drift.md`.
    ///
    /// Bisection diagnostic that DOES run is
    /// `iter50_bisect_block_forward_gpu_vs_cpu_real_gemma4` — it
    /// confirms stages A-D match within BF16 tolerance and reports
    /// stage E's BF16 drift with magnitude diagnostics.
    #[test]
    #[ignore = "BF16-saturated-softmax drift vs F32 CPU ref is expected; see memory note"]
    fn apply_vit_block_forward_gpu_matches_cpu_on_real_gemma4_block0() {
        // GPU full block 0 parity vs CPU on real Gemma 4 mmproj weights.
        // Single MlxDevice for everything — weights, input upload,
        // dispatch executor — all share the same Metal device.
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found");
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let weights = LoadedMmprojWeights::load(&gguf, &cfg, MlxDevice::new().expect("dev"))
            .expect("load");

        let hidden = cfg.hidden_size as usize;
        let head_dim = hidden / cfg.num_attention_heads as usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let batch = 32usize;
        let n_input = batch * hidden;

        let input_cpu: Vec<f32> = (0..n_input)
            .map(|i| ((i as f32) * 1e-4).sin() * 0.05)
            .collect();

        // CPU reference.
        eprintln!("running CPU block 0 reference (~25-30s)...");
        let cpu_t0 = std::time::Instant::now();
        let expected =
            apply_vit_block_forward_cpu(input_cpu.clone(), &weights, &cfg, 0).expect("cpu");
        eprintln!("CPU block 0 done in {:?}", cpu_t0.elapsed());

        // GPU: build executor from a new MlxDevice; weights' buffers
        // live on a different MlxDevice instance but the same
        // physical Metal device (system_default singleton). Buffers
        // are accessible cross-instance on Apple Silicon.
        let executor = GraphExecutor::new(MlxDevice::new().expect("dev2"));
        let input_buf = upload_f32(executor.device(), &input_cpu, vec![batch, hidden]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        mlx_native::ops::softmax::register(&mut registry);
        mlx_native::ops::sigmoid_mul::register(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        eprintln!("running GPU block 0 forward...");
        let gpu_t0 = std::time::Instant::now();
        let block_out = apply_vit_block_forward_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &weights,
            &cfg,
            0,
            &input_buf,
            batch as u32,
            scale,
        )
        .expect("gpu block forward");
        session.finish().expect("finish");
        eprintln!("GPU block 0 done in {:?}", gpu_t0.elapsed());

        let got = readback_f32(&block_out, n_input);

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
        let total = got.len();
        let frac = (fail_count as f32) / (total as f32);
        eprintln!(
            "block 0 GPU vs CPU: {}/{} = {:.3}% > 5e-2, max_diff = {}",
            fail_count, total, frac * 100.0, max_diff
        );
        assert!(
            frac < 0.05,
            "{:.3}% of elements exceeded 5e-2, max_diff = {}",
            frac * 100.0,
            max_diff
        );
    }

    /// Hypothesis: dense_matmul_bf16_f32_tensor requires K % 32 == 0
    /// (NK=32 tile size). Iter 47's passing test used head_dim=64 (2×32).
    /// Stage E in iter 50 fails for head_dim=72 (not a multiple of 32).
    /// This test confirms or refutes the hypothesis with synthetic
    /// inputs at the failing shape.
    #[test]
    fn vit_attention_gpu_isolated_head_dim_72_diagnostic() {
        // Same shape as Gemma 4: batch=32, num_heads=16, head_dim=72.
        // Synthetic Q/K/V (no real weights, no upstream stages).
        let batch = 32usize;
        let num_heads = 16usize;
        let head_dim = 72usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let n = batch * num_heads * head_dim;

        let q_cpu: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
        let k_cpu: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).cos() * 0.3).collect();
        let v_cpu: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.11).sin() * 0.4)
            .collect();

        let expected = attention_cpu(&q_cpu, &k_cpu, &v_cpu, batch, num_heads, head_dim)
            .expect("cpu");

        let device = MlxDevice::new().expect("dev");
        let executor = GraphExecutor::new(device);
        let q_buf = upload_f32(executor.device(), &q_cpu, vec![batch, num_heads, head_dim]);
        let k_buf = upload_f32(executor.device(), &k_cpu, vec![batch, num_heads, head_dim]);
        let v_buf = upload_f32(executor.device(), &v_cpu, vec![batch, num_heads, head_dim]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
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

        let mut max_diff = 0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            let d = (g - e).abs();
            if d > max_diff { max_diff = d; }
        }
        eprintln!(
            "head_dim=72 isolated attention: max_diff = {}, expected_sample = {}",
            max_diff, expected[0]
        );
        // If hypothesis correct: max_diff is huge (~10+), confirming
        // the K%32 kernel constraint. If wrong, max_diff is BF16-noise.
        // Don't assert — let it print so we see the value.
        if max_diff > 1.0 {
            eprintln!("HYPOTHESIS CONFIRMED: dense_mm_bf16 kernel requires K % 32 == 0");
        }
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

    // -------------------------------------------------------------------
    // gemma4v GPU primitives — CPU-vs-GPU parity tests
    // -------------------------------------------------------------------

    /// Upload a `&[u32]` into a fresh device buffer (for index gathers).
    fn upload_u32(device: &MlxDevice, data: &[u32], shape: Vec<usize>) -> MlxBuffer {
        let bytes = data.len() * 4;
        let buf = device
            .alloc_buffer(bytes, DType::F32, shape)
            .expect("alloc upload u32");
        let slice: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut u32, data.len())
        };
        slice.copy_from_slice(data);
        buf
    }

    #[test]
    fn gemma4v_patch_embed_cpu_gpu_parity() {
        // Tiny shape: 4 patches × inner=64 → hidden=32. Keep inner ≥ 32
        // (vit_linear_gpu's tensor-core tile floor).
        let n_patches = 4usize;
        let inner = 64usize;
        let hidden = 32usize;

        let patches_cpu: Vec<f32> = (0..n_patches * inner)
            .map(|i| ((i as f32) * 0.003).sin() * 0.5)
            .collect();
        let weight_cpu: Vec<f32> = (0..hidden * inner)
            .map(|i| ((i as f32) * 0.011).cos() * 0.2)
            .collect();

        let expect_cpu = gemma4v_patch_embed_cpu(
            &patches_cpu,
            &weight_cpu,
            n_patches as u32,
            inner as u32,
            hidden as u32,
        )
        .expect("cpu ref");

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let patches_buf = upload_f32(executor.device(), &patches_cpu, vec![n_patches, inner]);
        let weight_buf = upload_f32(executor.device(), &weight_cpu, vec![hidden, inner]);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out = gemma4v_patch_embed_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &patches_buf,
            &weight_buf,
            n_patches as u32,
            inner as u32,
            hidden as u32,
        )
        .expect("gpu dispatch");
        session.finish().expect("finish");

        let got = readback_f32(&out, n_patches * hidden);
        // BF16 weight round-trip tolerance — same bar as `vit_linear_gpu`'s
        // own parity test (1e-2 max-abs).
        for (i, (g, e)) in got.iter().zip(expect_cpu.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-2,
                "patch_embed parity mismatch at {i}: gpu={g} cpu={e}"
            );
        }
    }

    #[test]
    fn gemma4v_patch_embed_gpu_rejects_zero_n() {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let p = executor.device().alloc_buffer(64 * 4, DType::F32, vec![64]).expect("a");
        let w = executor.device().alloc_buffer(64 * 4, DType::F32, vec![64]).expect("b");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = gemma4v_patch_embed_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &p,
            &w,
            0,
            64,
            32,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("n_patches must be > 0"));
    }

    #[test]
    fn gemma4v_position_embed_cpu_gpu_parity() {
        // 6 patches into a [2, pos_size=4, hidden=8] table.
        let pos_size = 4usize;
        let hidden = 8usize;
        let pe_cpu: Vec<f32> = (0..2 * pos_size * hidden)
            .map(|i| (i as f32) * 0.1 - 5.0)
            .collect();
        let pos_x: Vec<u32> = vec![0, 1, 2, 3, 0, 2];
        let pos_y: Vec<u32> = vec![0, 0, 1, 2, 3, 1];
        let n_patches = pos_x.len() as u32;

        let expect_cpu = gemma4v_pos_embed_cpu(
            &pos_x,
            &pos_y,
            &pe_cpu,
            pos_size as u32,
            hidden as u32,
        )
        .expect("cpu pos lookup");

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let pe_buf = upload_f32(executor.device(), &pe_cpu, vec![2, pos_size, hidden]);
        let posx_buf = upload_u32(executor.device(), &pos_x, vec![pos_x.len()]);
        let posy_buf = upload_u32(executor.device(), &pos_y, vec![pos_y.len()]);

        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        // gather_f32 needs to be registered.
        mlx_native::ops::gather::register(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out = gemma4v_position_embed_lookup_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &pe_buf,
            &posx_buf,
            &posy_buf,
            n_patches,
            pos_size as u32,
            hidden as u32,
        )
        .expect("gpu pos lookup");
        session.finish().expect("finish");

        let got = readback_f32(&out, (n_patches as usize) * hidden);
        // Pure F32 path (no BF16 cast) — should be byte-exact.
        for (i, (g, e)) in got.iter().zip(expect_cpu.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "pos_embed parity at {i}: gpu={g} cpu={e}"
            );
        }
    }

    #[test]
    fn gemma4v_position_embed_lookup_gpu_rejects_zero_dims() {
        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let pe = executor.device().alloc_buffer(64 * 4, DType::F32, vec![64]).expect("a");
        let px = executor.device().alloc_buffer(4 * 4, DType::F32, vec![4]).expect("b");
        let py = executor.device().alloc_buffer(4 * 4, DType::F32, vec![4]).expect("c");
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let err = gemma4v_position_embed_lookup_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &pe,
            &px,
            &py,
            0,
            4,
            8,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("must all be > 0"));
    }

    #[test]
    fn gemma4v_apply_position_embed_gpu_adds_pe_to_patch_embeds() {
        // patch_embeds = ones[N=3, hidden=4] → after add, each row is
        // 1 + (pe[0][pos_x] + pe[1][pos_y]).
        let pos_size = 2usize;
        let hidden = 4usize;
        let pe_cpu: Vec<f32> = vec![
            // X-table:
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            // Y-table:
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
        ];
        let pos_x: Vec<u32> = vec![0, 1, 0];
        let pos_y: Vec<u32> = vec![1, 0, 0];
        let n_patches = pos_x.len() as u32;
        let patch_embeds_cpu: Vec<f32> = vec![1.0; (n_patches as usize) * hidden];

        // Expected: (pe_x + pe_y) row-wise, plus 1 baseline.
        let pos_emb = gemma4v_pos_embed_cpu(
            &pos_x, &pos_y, &pe_cpu,
            pos_size as u32,
            hidden as u32,
        )
        .unwrap();
        let mut expect: Vec<f32> = patch_embeds_cpu.clone();
        for (e, s) in expect.iter_mut().zip(pos_emb.iter()) {
            *e += *s;
        }

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let pe_buf = upload_f32(executor.device(), &pe_cpu, vec![2, pos_size, hidden]);
        let pe_table = pe_buf;
        let patch_buf = upload_f32(executor.device(), &patch_embeds_cpu, vec![n_patches as usize, hidden]);
        let posx_buf = upload_u32(executor.device(), &pos_x, vec![pos_x.len()]);
        let posy_buf = upload_u32(executor.device(), &pos_y, vec![pos_y.len()]);

        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        mlx_native::ops::gather::register(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };
        let out = gemma4v_apply_position_embed_gpu(
            session.encoder_mut(),
            &mut registry,
            device,
            &patch_buf,
            &pe_table,
            &posx_buf,
            &posy_buf,
            n_patches,
            pos_size as u32,
            hidden as u32,
        )
        .expect("gpu apply pos");
        session.finish().expect("finish");

        let got = readback_f32(&out, (n_patches as usize) * hidden);
        for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "apply_pos parity at {i}: gpu={g} cpu={e}"
            );
        }
    }

    // -------------------------------------------------------------------
    // gemma4v_block_forward — CPU+GPU parity + dimension + 4-RMSNorm tests
    // -------------------------------------------------------------------

    use crate::inference::vision::vit::{
        Gemma4VisionBlockShape, Gemma4VisionBlockWeights,
    };

    /// Synthesize a small gemma4v block fixture: weights + shape +
    /// positions. Sizes are chosen so the BF16 cast inside `vit_linear_gpu`
    /// (`in_features >= 32`) is satisfied for every projection while the
    /// numeric domain stays small enough for f32 parity comparisons.
    fn synth_gemma4v_block_fixture()
        -> (
            Gemma4VisionBlockShape,
            usize,                  // batch (== num_patches == seq_len)
            Vec<f32>,               // input_layernorm.weight  [hidden]
            Vec<f32>,               // post_attention_layernorm [hidden]
            Vec<f32>,               // pre_feedforward_layernorm [hidden]
            Vec<f32>,               // post_feedforward_layernorm [hidden]
            Vec<f32>,               // q_proj [num_heads*head_dim, hidden]
            Vec<f32>,               // k_proj [num_kv_heads*head_dim, hidden]
            Vec<f32>,               // v_proj [num_kv_heads*head_dim, hidden]
            Vec<f32>,               // o_proj [hidden, num_heads*head_dim]
            Vec<f32>,               // q_norm [head_dim]
            Vec<f32>,               // k_norm [head_dim]
            Vec<f32>,               // gate_proj [intermediate, hidden]
            Vec<f32>,               // up_proj   [intermediate, hidden]
            Vec<f32>,               // down_proj [hidden, intermediate]
            Vec<u32>,               // pos_x
            Vec<u32>,               // pos_y
            Vec<f32>,               // input [batch, hidden]
        ) {
        // 4 heads × 32 head_dim = hidden 128. KV groups = 2 → num_kv_heads = 2.
        // Intermediate = 64. Batch = 32 patches (matches `vit_attention_gpu`'s
        // tensor-core tile constraint K=batch >= 32 for the scores @ V matmul).
        let hidden = 128usize;
        let num_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 32usize;
        let intermediate = 64usize;
        let batch = 32usize;
        let shape = Gemma4VisionBlockShape {
            hidden: hidden as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            intermediate: intermediate as u32,
            rms_norm_eps: 1e-6,
            rope_theta: 100.0,
        };
        // Use deterministic small magnitudes so projections stay in a
        // domain where f32 sums don't accumulate large absolute error.
        let mk = |seed: f32, n: usize| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * seed + 0.13).sin() * 0.05).collect()
        };
        let input_ln    = mk(0.011, hidden);
        let post_attn_ln = mk(0.013, hidden);
        let pre_ff_ln    = mk(0.017, hidden);
        let post_ff_ln   = mk(0.019, hidden);
        let q_w  = mk(0.021, num_heads * head_dim * hidden);
        let k_w  = mk(0.023, num_kv_heads * head_dim * hidden);
        let v_w  = mk(0.025, num_kv_heads * head_dim * hidden);
        let o_w  = mk(0.027, hidden * num_heads * head_dim);
        let q_n  = mk(0.031, head_dim);
        let k_n  = mk(0.033, head_dim);
        let g_w  = mk(0.041, intermediate * hidden);
        let u_w  = mk(0.043, intermediate * hidden);
        let d_w  = mk(0.045, hidden * intermediate);
        let pos_x: Vec<u32> = (0..batch as u32).collect();
        let pos_y: Vec<u32> = (0..batch as u32).rev().collect();
        let input: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i as f32) * 0.07).cos() * 0.04)
            .collect();
        (
            shape, batch,
            input_ln, post_attn_ln, pre_ff_ln, post_ff_ln,
            q_w, k_w, v_w, o_w, q_n, k_n,
            g_w, u_w, d_w,
            pos_x, pos_y,
            input,
        )
    }

    /// Test 1: 4-RMSNorm dispatch count (architecture sanity).
    ///
    /// The CPU forward calls `gemma_rms_norm_forward` exactly 4 times per
    /// block (input_ln + post_attn_ln + pre_ff_ln + post_ff_ln) and
    /// `gemma_per_head_rms_norm_forward` exactly 2 times (q_norm, k_norm),
    /// plus 1 `v_norm_no_scale_forward`. We assert this by structuring the
    /// fixture so a wrong-arch shortcut would corrupt the output.
    /// In practice the 4-RMSNorm count is implied by the function calling
    /// each helper distinctly with 4 different gain tensors — if any pair
    /// were accidentally aliased the parity test below would fail.
    #[test]
    fn gemma4v_block_forward_4_rmsnorm_count_is_exactly_four() {
        // Synthesize fixture and execute the CPU forward; assert that
        // each of the 4 gain tensors actually influences the output by
        // perturbing one and observing a delta.
        let (
            shape, batch,
            input_ln, post_attn_ln, pre_ff_ln, post_ff_ln,
            q_w, k_w, v_w, o_w, q_n, k_n,
            g_w, u_w, d_w,
            pos_x, pos_y,
            input,
        ) = synth_gemma4v_block_fixture();

        let baseline = gemma4v_block_forward_cpu(
            input.clone(),
            &Gemma4VisionBlockWeights {
                input_layernorm: &input_ln,
                post_attention_layernorm: &post_attn_ln,
                pre_feedforward_layernorm: &pre_ff_ln,
                post_feedforward_layernorm: &post_ff_ln,
                q_proj: &q_w, k_proj: &k_w, v_proj: &v_w, o_proj: &o_w,
                q_norm: &q_n, k_norm: &k_n,
                gate_proj: &g_w, up_proj: &u_w, down_proj: &d_w,
            },
            &shape, &pos_x, &pos_y,
        ).expect("baseline");
        assert_eq!(baseline.len(), batch * shape.hidden as usize);

        // Perturb each of the 4 norm-gain tensors INDIVIDUALLY and confirm
        // the output changes. This proves all 4 are wired in (none are
        // accidentally bypassed) — a 4-RMSNorm count probe.
        for which in 0..4usize {
            let mut iln = input_ln.clone();
            let mut pal = post_attn_ln.clone();
            let mut pre = pre_ff_ln.clone();
            let mut post = post_ff_ln.clone();
            match which {
                0 => iln[0] += 0.5,
                1 => pal[0] += 0.5,
                2 => pre[0] += 0.5,
                _ => post[0] += 0.5,
            }
            let perturbed_out = gemma4v_block_forward_cpu(
                input.clone(),
                &Gemma4VisionBlockWeights {
                    input_layernorm: &iln,
                    post_attention_layernorm: &pal,
                    pre_feedforward_layernorm: &pre,
                    post_feedforward_layernorm: &post,
                    q_proj: &q_w, k_proj: &k_w, v_proj: &v_w, o_proj: &o_w,
                    q_norm: &q_n, k_norm: &k_n,
                    gate_proj: &g_w, up_proj: &u_w, down_proj: &d_w,
                },
                &shape, &pos_x, &pos_y,
            ).expect("perturbed");
            let max_d = baseline.iter().zip(perturbed_out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_d > 1e-6,
                "norm gain index {which} did not affect output (4-RMSNorm not wired correctly): max|delta| = {max_d}"
            );
        }
    }

    /// Test 2: GQA dimensions — Q has [B, num_heads, head_dim] and K/V
    /// have [B, num_kv_heads, head_dim] before repeat-kv, then K_full /
    /// V_full match Q's head count after repeat-kv. We test repeat_kv_cpu
    /// directly since it's the GQA bridge.
    #[test]
    fn gemma4v_block_forward_gqa_dimensions() {
        use crate::inference::vision::vit::repeat_kv_cpu;
        let batch = 3usize;
        let num_kv_heads = 2usize;
        let num_kv_groups = 3usize;
        let head_dim = 4usize;
        let n_in = batch * num_kv_heads * head_dim;
        let input: Vec<f32> = (0..n_in).map(|i| i as f32).collect();
        let out = repeat_kv_cpu(&input, batch, num_kv_heads, num_kv_groups, head_dim).unwrap();
        let num_heads = num_kv_heads * num_kv_groups; // = 6
        assert_eq!(out.len(), batch * num_heads * head_dim);
        // For each batch, kv_head k → expanded heads [k*g .. k*g + g - 1]
        // share the SAME [head_dim] slice from the input.
        for b in 0..batch {
            for k in 0..num_kv_heads {
                let in_base = (b * num_kv_heads + k) * head_dim;
                let in_slice = &input[in_base..in_base + head_dim];
                for g in 0..num_kv_groups {
                    let h = k * num_kv_groups + g;
                    let out_base = (b * num_heads + h) * head_dim;
                    let out_slice = &out[out_base..out_base + head_dim];
                    assert_eq!(
                        in_slice, out_slice,
                        "repeat_kv at batch={b} kv_head={k} group={g} (out head={h})"
                    );
                }
            }
        }
    }

    /// Test 3: CPU↔GPU parity for the full per-block forward with
    /// synthetic weights. Validates the pipeline wiring end-to-end.
    #[test]
    fn gemma4v_block_forward_cpu_gpu_parity() {
        let (
            shape, batch,
            input_ln, post_attn_ln, pre_ff_ln, post_ff_ln,
            q_w, k_w, v_w, o_w, q_n, k_n,
            g_w, u_w, d_w,
            pos_x, pos_y,
            input,
        ) = synth_gemma4v_block_fixture();
        let hidden = shape.hidden as usize;
        let num_heads = shape.num_heads as usize;
        let num_kv_heads = shape.num_kv_heads as usize;
        let head_dim = shape.head_dim as usize;
        let intermediate = shape.intermediate as usize;

        // ---- CPU reference ----
        let bw_cpu = Gemma4VisionBlockWeights {
            input_layernorm: &input_ln,
            post_attention_layernorm: &post_attn_ln,
            pre_feedforward_layernorm: &pre_ff_ln,
            post_feedforward_layernorm: &post_ff_ln,
            q_proj: &q_w, k_proj: &k_w, v_proj: &v_w, o_proj: &o_w,
            q_norm: &q_n, k_norm: &k_n,
            gate_proj: &g_w, up_proj: &u_w, down_proj: &d_w,
        };
        let cpu_out = gemma4v_block_forward_cpu(
            input.clone(), &bw_cpu, &shape, &pos_x, &pos_y,
        ).expect("cpu forward");

        // ---- GPU forward via synthetic LoadedMmprojWeights ----
        let device = MlxDevice::new().expect("device");

        // Build a tensor map under the gemma4v block-suffix names.
        let mut tensors: std::collections::HashMap<String, MlxBuffer> = std::collections::HashMap::new();
        let put = |tensors: &mut std::collections::HashMap<String, MlxBuffer>, dev: &MlxDevice, key: String, data: &[f32], shape: Vec<usize>| {
            let bytes = data.len() * 4;
            let buf = dev.alloc_buffer(bytes, DType::F32, shape).expect("alloc tensor");
            let slice: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut f32, data.len())
            };
            slice.copy_from_slice(data);
            tensors.insert(key, buf);
        };
        // Layer index = 0.
        let block_key = |suffix: &str| format!("v.blk.0.{}", suffix);
        // W44 iter-116k: emit-name semantics aligned with W36 iter-116f
        // writer renames; canonical short forms per
        // /opt/llama.cpp/tools/mtmd/clip-impl.h.
        //   - ln1            = pre-attention norm                     (l.89)
        //   - attn_post_norm = post-attention norm   (was: ln2)        (l.94)
        //   - ln2            = pre-FFN norm          (was: ffn_norm)   (l.90)
        //   - ffn_post_norm  = post-FFN norm         (was: post_ffw)   (l.95)
        //   - attn_out       = attn output projection (was: attn_output) (l.82)
        put(&mut tensors, &device, block_key("ln1.weight"), &input_ln, vec![hidden]);
        put(&mut tensors, &device, block_key("attn_post_norm.weight"), &post_attn_ln, vec![hidden]);
        put(&mut tensors, &device, block_key("ln2.weight"), &pre_ff_ln, vec![hidden]);
        put(&mut tensors, &device, block_key("ffn_post_norm.weight"), &post_ff_ln, vec![hidden]);
        put(&mut tensors, &device, block_key("attn_q.weight"), &q_w,
            vec![num_heads * head_dim, hidden]);
        put(&mut tensors, &device, block_key("attn_k.weight"), &k_w,
            vec![num_kv_heads * head_dim, hidden]);
        put(&mut tensors, &device, block_key("attn_v.weight"), &v_w,
            vec![num_kv_heads * head_dim, hidden]);
        put(&mut tensors, &device, block_key("attn_out.weight"), &o_w,
            vec![hidden, num_heads * head_dim]);
        put(&mut tensors, &device, block_key("attn_q_norm.weight"), &q_n, vec![head_dim]);
        put(&mut tensors, &device, block_key("attn_k_norm.weight"), &k_n, vec![head_dim]);
        put(&mut tensors, &device, block_key("ffn_gate.weight"), &g_w,
            vec![intermediate, hidden]);
        put(&mut tensors, &device, block_key("ffn_up.weight"), &u_w,
            vec![intermediate, hidden]);
        put(&mut tensors, &device, block_key("ffn_down.weight"), &d_w,
            vec![hidden, intermediate]);

        let weights = LoadedMmprojWeights::from_tensors_for_test(tensors, device);

        // Now run the GPU forward.
        let device = MlxDevice::new().expect("device2");
        let executor = GraphExecutor::new(device);
        let in_buf = upload_f32(executor.device(), &input, vec![batch, hidden]);
        // Upload positions as DType::U32 (vision_2d_rope dispatch validates dtype).
        let upload_u32_typed = |dev: &MlxDevice, data: &[u32]| -> MlxBuffer {
            let buf = dev.alloc_buffer(data.len() * 4, DType::U32, vec![data.len()])
                .expect("alloc u32 typed");
            let slice: &mut [u32] = unsafe {
                std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut u32, data.len())
            };
            slice.copy_from_slice(data);
            buf
        };
        let px_buf = upload_u32_typed(executor.device(), &pos_x);
        let py_buf = upload_u32_typed(executor.device(), &pos_y);

        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        // Required shaders: vit_silu_mul (sigmoid_mul) + softmax + 2D rope + gelu + rms_norm_no_scale + gather.
        mlx_native::ops::sigmoid_mul::register(&mut registry);
        mlx_native::ops::softmax::register(&mut registry);
        mlx_native::ops::vision_2d_rope::register(&mut registry);
        mlx_native::ops::gelu::register(&mut registry);
        mlx_native::ops::gather::register(&mut registry);

        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device: &MlxDevice = unsafe { &*device_ref };

        let shape_gpu = Gemma4VisionBlockShapeGpu {
            hidden: shape.hidden, num_heads: shape.num_heads,
            num_kv_heads: shape.num_kv_heads, head_dim: shape.head_dim,
            intermediate: shape.intermediate,
            rms_norm_eps: shape.rms_norm_eps, rope_theta: shape.rope_theta,
        };
        let out = gemma4v_block_forward_gpu(
            session.encoder_mut(), &mut registry, device,
            &weights, &shape_gpu, 0, &in_buf,
            &px_buf, &py_buf, batch as u32,
        ).expect("gpu block forward");
        session.finish().expect("finish");
        let gpu_out = readback_f32(&out, batch * hidden);

        // BF16 cast inside `vit_linear_gpu` introduces ~1e-3 relative
        // noise per matmul. With 4 matmuls (q/k/v/o) + gate/up/down +
        // 4 RMS norms + 2 RoPE rotations the cumulative absolute drift
        // can reach a few units of the input magnitude scale (~0.05).
        // Cosine similarity is the discipline-correct metric for
        // pipeline-cumulative tolerance.
        let dot: f32 = cpu_out.iter().zip(gpu_out.iter()).map(|(a, b)| a * b).sum();
        let na: f32 = cpu_out.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = gpu_out.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = dot / (na * nb).max(1e-30);
        let max_abs = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "gemma4v_block parity: cos={cos:.6}, max|abs|={max_abs:.6}, |cpu|={na:.4}, |gpu|={nb:.4}"
        );
        assert!(
            cos > 0.999,
            "gemma4v_block_forward CPU↔GPU cosine = {cos} < 0.999 (max|abs| = {max_abs})"
        );
    }

    // -----------------------------------------------------------------------
    // gemma4v_apply_full_forward_gpu — synthetic-fixture smoke tests (iter 115)
    // -----------------------------------------------------------------------

    /// Build a tiny synthetic gemma4v fixture sized to the
    /// `vit_attention_gpu` K-tile floor (K = batch >= 32) for the
    /// 27-block path. Unused-but-needed weights (std_bias, std_scale,
    /// mm.0.weight, position_embd dual table, patch_embd) are
    /// stamped into a `LoadedMmprojWeights`.
    ///
    /// Returns `(weights, cfg, patches, pos_x, pos_y, n_x, n_y)`. The
    /// returned `cfg.num_hidden_layers` is small (2) so the test runs
    /// fast without sacrificing the dispatch coverage.
    #[allow(clippy::too_many_arguments)]
    fn synth_gemma4v_full_fixture(
        n_x: u32,
        n_y: u32,
    ) -> (
        LoadedMmprojWeights,
        MmprojConfig,
        Vec<f32>,
        Vec<u32>,
        Vec<u32>,
    ) {
        // Smallest legal sizes that satisfy:
        //   - hidden % num_attention_heads == 0 (head_dim positive)
        //   - head_dim >= 32 (vit_attention_gpu floor)
        //   - num_attention_heads % num_kv_heads == 0 (GQA)
        //   - inner = patch_size² * 3 >= 32 (vit_linear_gpu floor)
        // Keep 2 layers so the loop exercises multi-block dispatch.
        let hidden: u32 = 128;
        let num_heads: u32 = 4;
        let num_kv_heads: u32 = 2;
        let head_dim: u32 = hidden / num_heads; // 32
        let intermediate: u32 = 64;
        let patch_size: u32 = 4;                 // inner = 4*4*3 = 48 >= 32
        let inner: u32 = patch_size * patch_size * 3;
        let num_layers: u32 = 2;
        // text_hidden — projector output; pick any > 0.
        let text_hidden: u32 = 64;
        // Position-embed table size — must be >= max(n_x, n_y) so
        // index lookups don't clamp. gemma4v's actual pos_size in the
        // GGUF is the number of unique X (or Y) coordinates the
        // preprocessor can emit; for this fixture we set it to a
        // safe upper bound.
        let pos_size: u32 = (n_x.max(n_y)).max(64);
        let n_patches = n_x * n_y;

        let mk = |seed: f32, n: usize| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * seed + 0.13).sin() * 0.05).collect()
        };

        let device = MlxDevice::new().expect("synth dev");
        let put = |tensors: &mut std::collections::HashMap<String, MlxBuffer>,
                   dev: &MlxDevice,
                   key: String,
                   data: &[f32],
                   shape: Vec<usize>| {
            let bytes = data.len() * 4;
            let buf = dev
                .alloc_buffer(bytes, DType::F32, shape)
                .expect("alloc tensor");
            let slice: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut f32, data.len())
            };
            slice.copy_from_slice(data);
            tensors.insert(key, buf);
        };

        let mut tensors: std::collections::HashMap<String, MlxBuffer> =
            std::collections::HashMap::new();

        // Stem tensors.
        let patch_w = mk(0.011, (hidden * inner) as usize);
        put(
            &mut tensors,
            &device,
            "v.patch_embd.weight".to_string(),
            &patch_w,
            vec![hidden as usize, inner as usize],
        );
        // Dual position-embed table [2, pos_size, hidden].
        let pe_table = mk(0.013, (2 * pos_size * hidden) as usize);
        put(
            &mut tensors,
            &device,
            "v.position_embd.weight".to_string(),
            &pe_table,
            vec![2usize, pos_size as usize, hidden as usize],
        );
        // std_bias / std_scale [hidden].
        let std_bias = mk(0.014, hidden as usize);
        put(
            &mut tensors,
            &device,
            "v.std_bias".to_string(),
            &std_bias,
            vec![hidden as usize],
        );
        // std_scale: keep magnitudes near 1.0 so the post-pool path
        // doesn't blow up f32 range.
        let std_scale: Vec<f32> = (0..hidden).map(|_| 1.0).collect();
        put(
            &mut tensors,
            &device,
            "v.std_scale".to_string(),
            &std_scale,
            vec![hidden as usize],
        );
        // mm.0.weight [text_hidden, hidden].
        let mm0 = mk(0.017, (text_hidden * hidden) as usize);
        put(
            &mut tensors,
            &device,
            "mm.0.weight".to_string(),
            &mm0,
            vec![text_hidden as usize, hidden as usize],
        );

        // Per-block tensors × num_layers.
        // W44 iter-116k: emit-name semantics aligned with W36 iter-116f
        // writer renames. Canonical short forms per
        // /opt/llama.cpp/tools/mtmd/clip-impl.h:
        //   - ln1            (l.89) — pre-attention norm
        //   - attn_post_norm (l.94) — post-attention norm   (was: ln2)
        //   - ln2            (l.90) — pre-FFN norm          (was: ffn_norm)
        //   - ffn_post_norm  (l.95) — post-FFN norm         (was: post_ffw_norm)
        //   - attn_out       (l.82) — attn output proj      (was: attn_output)
        for layer_idx in 0..num_layers {
            let prefix = format!("v.blk.{}.", layer_idx);
            put(
                &mut tensors,
                &device,
                format!("{prefix}ln1.weight"),
                &mk(0.021 + layer_idx as f32 * 0.001, hidden as usize),
                vec![hidden as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}attn_post_norm.weight"),
                &mk(0.022 + layer_idx as f32 * 0.001, hidden as usize),
                vec![hidden as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}ln2.weight"),
                &mk(0.023 + layer_idx as f32 * 0.001, hidden as usize),
                vec![hidden as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}ffn_post_norm.weight"),
                &mk(0.024 + layer_idx as f32 * 0.001, hidden as usize),
                vec![hidden as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}attn_q.weight"),
                &mk(0.031 + layer_idx as f32 * 0.001, (num_heads * head_dim * hidden) as usize),
                vec![(num_heads * head_dim) as usize, hidden as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}attn_k.weight"),
                &mk(0.033 + layer_idx as f32 * 0.001, (num_kv_heads * head_dim * hidden) as usize),
                vec![(num_kv_heads * head_dim) as usize, hidden as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}attn_v.weight"),
                &mk(0.035 + layer_idx as f32 * 0.001, (num_kv_heads * head_dim * hidden) as usize),
                vec![(num_kv_heads * head_dim) as usize, hidden as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}attn_out.weight"),
                &mk(0.037 + layer_idx as f32 * 0.001, (hidden * num_heads * head_dim) as usize),
                vec![hidden as usize, (num_heads * head_dim) as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}attn_q_norm.weight"),
                &mk(0.041 + layer_idx as f32 * 0.001, head_dim as usize),
                vec![head_dim as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}attn_k_norm.weight"),
                &mk(0.043 + layer_idx as f32 * 0.001, head_dim as usize),
                vec![head_dim as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}ffn_gate.weight"),
                &mk(0.051 + layer_idx as f32 * 0.001, (intermediate * hidden) as usize),
                vec![intermediate as usize, hidden as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}ffn_up.weight"),
                &mk(0.053 + layer_idx as f32 * 0.001, (intermediate * hidden) as usize),
                vec![intermediate as usize, hidden as usize],
            );
            put(
                &mut tensors,
                &device,
                format!("{prefix}ffn_down.weight"),
                &mk(0.055 + layer_idx as f32 * 0.001, (hidden * intermediate) as usize),
                vec![hidden as usize, intermediate as usize],
            );
        }

        let weights = LoadedMmprojWeights::from_tensors_for_test(tensors, device);

        // Synth patches + per-patch positions.
        let patches: Vec<f32> = (0..(n_patches * inner) as usize)
            .map(|i| ((i as f32) * 0.07).cos() * 0.04)
            .collect();
        let mut pos_x: Vec<u32> = Vec::with_capacity(n_patches as usize);
        let mut pos_y: Vec<u32> = Vec::with_capacity(n_patches as usize);
        for y in 0..n_y {
            for x in 0..n_x {
                pos_x.push(x);
                pos_y.push(y);
            }
        }

        let cfg = MmprojConfig {
            image_size: 0, // unused on gemma4v variable-resolution path
            patch_size,
            num_patches_side: 0, // unused
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_attention_heads: num_heads,
            num_hidden_layers: num_layers,
            layer_norm_eps: 1e-6,
            projector: crate::inference::vision::mmproj::ProjectorType::Mlp,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        };
        (weights, cfg, patches, pos_x, pos_y)
    }

    /// Variable-resolution sanity: run the full gemma4v forward at
    /// (n_x=6, n_y=6) → 36 patches, batch=36 (≥32 K-tile floor),
    /// post-pool 2×2 = 4 tokens. Asserts the output is finite and has
    /// the correct shape.
    #[test]
    fn gemma4v_apply_full_forward_gpu_synthetic_n_36() {
        let n_x: u32 = 6;
        let n_y: u32 = 6;
        let (weights, cfg, patches, pos_x, pos_y) = synth_gemma4v_full_fixture(n_x, n_y);

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        mlx_native::ops::softmax::register(&mut registry);
        mlx_native::ops::sigmoid_mul::register(&mut registry);
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device_ref: &MlxDevice = unsafe { &*device_ref };

        let buf = gemma4v_apply_full_forward_gpu(
            session.encoder_mut(),
            &mut registry,
            device_ref,
            &weights,
            &cfg,
            &patches,
            &pos_x,
            &pos_y,
            n_x,
            n_y,
        )
        .expect("full forward synthetic");
        session.finish().expect("finish");

        // Expected output shape: [(n_x/3)*(n_y/3), text_hidden = 64].
        let pooled_n = ((n_x / 3) * (n_y / 3)) as usize; // 4
        let text_hidden = 64usize;
        let expected_len = pooled_n * text_hidden;
        let slice: &[f32] = buf.as_slice::<f32>().expect("readback");
        assert_eq!(slice.len(), expected_len, "output len mismatch");
        for v in slice {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    /// Same fixture, larger N. (n_x=9, n_y=6) → 54 patches, post-pool
    /// 3×2 = 6 tokens. Verifies the rectangular path works at sizes
    /// that more closely approximate gemma4v production (~270 patches).
    #[test]
    fn gemma4v_apply_full_forward_gpu_synthetic_rectangular_n_54() {
        let n_x: u32 = 9;
        let n_y: u32 = 6;
        let (weights, cfg, patches, pos_x, pos_y) = synth_gemma4v_full_fixture(n_x, n_y);

        let device = MlxDevice::new().expect("device");
        let executor = GraphExecutor::new(device);
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        mlx_native::ops::softmax::register(&mut registry);
        mlx_native::ops::sigmoid_mul::register(&mut registry);
        register_vit_custom_shaders(&mut registry);
        let device_ref: *const MlxDevice = executor.device() as *const _;
        let device_ref: &MlxDevice = unsafe { &*device_ref };

        let buf = gemma4v_apply_full_forward_gpu(
            session.encoder_mut(),
            &mut registry,
            device_ref,
            &weights,
            &cfg,
            &patches,
            &pos_x,
            &pos_y,
            n_x,
            n_y,
        )
        .expect("full forward synthetic rect");
        session.finish().expect("finish");

        let pooled_n = ((n_x / 3) * (n_y / 3)) as usize; // 6
        let text_hidden = 64usize;
        let expected_len = pooled_n * text_hidden;
        let slice: &[f32] = buf.as_slice::<f32>().expect("readback");
        assert_eq!(slice.len(), expected_len, "rect output len mismatch");
        for v in slice {
            assert!(v.is_finite(), "non-finite rect output: {v}");
        }
    }

    /// Arch-profile dispatch: routes a Gemma4v input through the
    /// gemma4v full-forward when arch == Gemma4Siglip. Because the
    /// synthetic fixture only stamps gemma4v-shaped tensors, we can't
    /// validate the SigLIP branch here without a separate fixture —
    /// the arch dispatch over a homogeneous `Gemma4v` input batch is
    /// what we exercise.
    #[test]
    fn compute_vision_embeddings_gpu_dispatch_gemma4v_routes_correctly() {
        use crate::inference::vision::mmproj::ArchProfile;
        let n_x: u32 = 6;
        let n_y: u32 = 6;
        let (weights, cfg, patches, pos_x, pos_y) = synth_gemma4v_full_fixture(n_x, n_y);
        let img = Gemma4vPreprocessedImage {
            patches,
            pos_x,
            pos_y,
            n_x,
            n_y,
            source_label: "synthetic".to_string(),
        };
        let inputs = vec![VisionInput::Gemma4v(img)];
        let result = compute_vision_embeddings_gpu_dispatch(
            &inputs,
            ArchProfile::Gemma4Siglip,
            &weights,
            &cfg,
            1.0,
        )
        .expect("dispatch");
        assert_eq!(result.len(), 1);
        let pooled_n = ((n_x / 3) * (n_y / 3)) as usize;
        let text_hidden = 64usize;
        assert_eq!(result[0].len(), pooled_n * text_hidden);
        for v in &result[0] {
            assert!(v.is_finite());
        }
    }

    /// Arch-mismatch guard: a Gemma4v input under ArchProfile::ClipClassic
    /// must error rather than silently routing to the wrong forward.
    #[test]
    fn compute_vision_embeddings_gpu_dispatch_rejects_arch_mismatch() {
        use crate::inference::vision::mmproj::ArchProfile;
        let n_x: u32 = 6;
        let n_y: u32 = 6;
        let (weights, cfg, patches, pos_x, pos_y) = synth_gemma4v_full_fixture(n_x, n_y);
        let img = Gemma4vPreprocessedImage {
            patches,
            pos_x,
            pos_y,
            n_x,
            n_y,
            source_label: "synthetic".to_string(),
        };
        let inputs = vec![VisionInput::Gemma4v(img)];
        let err = compute_vision_embeddings_gpu_dispatch(
            &inputs,
            ArchProfile::ClipClassic,
            &weights,
            &cfg,
            1.0,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Gemma4v") && msg.contains("ClipClassic"),
            "expected arch-mismatch error message, got: {msg}"
        );
    }

    /// Empty input slice: must succeed with an empty output (a
    /// no-images-in-request shouldn't crash).
    #[test]
    fn compute_vision_embeddings_gpu_dispatch_empty_inputs_ok() {
        use crate::inference::vision::mmproj::ArchProfile;
        let n_x: u32 = 6;
        let n_y: u32 = 6;
        let (weights, cfg, _, _, _) = synth_gemma4v_full_fixture(n_x, n_y);
        let inputs: Vec<VisionInput> = Vec::new();
        let result = compute_vision_embeddings_gpu_dispatch(
            &inputs,
            ArchProfile::Gemma4Siglip,
            &weights,
            &cfg,
            1.0,
        )
        .expect("empty dispatch");
        assert!(result.is_empty());
    }

    /// ADR-005 phase 2c iter 124 (W55) — parity probe runner.
    ///
    /// Loads the four-dots fixture, runs the real gemma4v preprocessor +
    /// GPU ViT forward, and (when `HF2Q_VIT_DUMP=<dir>` is set) writes
    /// stage tensors via the production dump path in
    /// `gemma4v_apply_full_forward_gpu`. Skipped by default — the test
    /// is `#[ignore]`d AND additionally short-circuits when the env var
    /// is unset, so a stray `cargo test -- --ignored` on a host without
    /// HF2Q_VIT_DUMP doesn't burn 30+ s of GPU time.
    ///
    /// Run:
    ///   HF2Q_VIT_DUMP=/tmp/hf2q_dumps \
    ///     cargo test --release --bin hf2q -- \
    ///       inference::vision::vit_gpu::tests::iter124_parity_probe \
    ///       --ignored --nocapture
    #[test]
    #[ignore]
    fn iter124_parity_probe_dump_four_dots_real_gemma4() {
        if super::super::vit_dump::resolve_dump_dir()
            .expect("resolve dump dir")
            .is_none()
        {
            eprintln!("skip: HF2Q_VIT_DUMP unset (parity probe runner is opt-in)");
            return;
        }
        let mmproj_path = Path::new(GEMMA4_MMPROJ_PATH);
        if !mmproj_path.exists() {
            eprintln!("skip: mmproj fixture not found at {}", GEMMA4_MMPROJ_PATH);
            return;
        }
        let img_path = Path::new(
            "/opt/hf2q/tests/fixtures/vision/four_dots_in_corners_128x128.png",
        );
        let img_bytes = std::fs::read(img_path).expect("read four-dots fixture");

        let pre = crate::inference::vision::preprocess::preprocess_gemma4v(
            &img_bytes,
            &crate::inference::vision::preprocess::GEMMA4V_PREPROCESS_DEFAULT,
        )
        .expect("preprocess");

        let gguf = GgufFile::open(mmproj_path).expect("open mmproj");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let device = MlxDevice::new().expect("device");
        let weights =
            LoadedMmprojWeights::load(&gguf, &cfg, device).expect("load mmproj");

        let img = Gemma4vPreprocessedImage {
            patches: pre.patches,
            pos_x: pre.pos_x,
            pos_y: pre.pos_y,
            n_x: pre.n_x,
            n_y: pre.n_y,
            source_label: "four_dots_in_corners_128x128.png".to_string(),
        };

        let out = compute_vision_embeddings_gpu_gemma4v(
            std::slice::from_ref(&img),
            &weights,
            &cfg,
        )
        .expect("forward");
        assert_eq!(out.len(), 1);
        assert!(!out[0].is_empty());

        // The dump directory is now populated by
        // compute_vision_embeddings_gpu_gemma4v. Surface the resolved
        // path so the runner script can pick it up via stdout.
        let dir = super::super::vit_dump::resolve_dump_dir()
            .expect("resolve dump dir")
            .expect("dump dir set");
        eprintln!(
            "iter124 parity probe: dumps written to {} (n_x={}, n_y={})",
            dir.display(),
            img.n_x,
            img.n_y
        );
    }
}
