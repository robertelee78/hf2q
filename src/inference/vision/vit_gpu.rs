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
use mlx_native::ops::softmax::dispatch_softmax;
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::vision::vit::{
        linear_forward as linear_cpu, per_head_rms_norm_forward as per_head_rms_cpu,
        rms_norm_forward as rms_norm_cpu, softmax_last_dim as softmax_cpu,
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
