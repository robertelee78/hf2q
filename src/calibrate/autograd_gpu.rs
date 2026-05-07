//! **GPU-backed autograd primitives** on mlx-native (ADR-020 Track 1, iter 8b).
//!
//! This module is the production codepath for the autograd ops mlx-lm
//! `dynamic_quant.estimate_sensitivities` requires.  Each op is composed
//! from existing mlx-native Metal kernels (`dense_matmul_f32_f32_tensor`,
//! `transpose_2d`, `elementwise_add`, etc.) and parity-tested against
//! the iter-8a CPU correctness oracle in `crate::calibrate::autograd`.
//!
//! Per `~/Documents/mantra.txt`: production runs on GPU.  CPU code in
//! the sibling oracle module is reachable only from `#[cfg(test)]`
//! parity assertions — never from a runtime entry point.
//!
//! ## Iter 8b scope
//!
//! Standalone matmul forward + backward functions.  No `Tape` /
//! `Tensor` wrapping yet — that lands in iter 8b.1.  The matmul is the
//! computational backbone of the dynamic_quant gradient pipeline; the
//! other ops (add, mul, square, sum, mean) follow in 8c.
//!
//! ## Forward shape contract (matmul `Y = X @ W`)
//!
//! - `X` shape `[M, K]` row-major
//! - `W` shape `[K, N]` row-major
//! - `Y` shape `[M, N]` row-major
//!
//! mlx-native's `dense_matmul_f32_f32_tensor` computes
//! `dst[m, n] = sum_k src0[n, k] * src1[m, k]` — i.e. `src1 @ src0^T`.
//! So the forward call needs `src0 = W^T` (shape `[N, K]`) and
//! `src1 = X` (shape `[M, K]`).  We pre-transpose `W` via `transpose_2d`.
//!
//! ## Backward shape contract
//!
//! - `dX = dY @ W^T`  shape `[M, K]`
//! - `dW = X^T @ dY`  shape `[K, N]`
//!
//! For dense_matmul:
//! - `dX = sum_n dY[m, n] · W[k, n]`.  Map: `dst[m, k] = sum_n src0[k, n] · src1[m, n]`,
//!   so `src0 = W` (no transpose, shape `[K, N]`), `src1 = dY` (shape `[M, N]`),
//!   m_param=M, n_param=K, k_param=N.
//! - `dW = sum_m X[m, k] · dY[m, n]`.  Map:
//!   `dst[k, n] = sum_m src0[n, m] · src1[k, m]`, so `src0 = dY^T`
//!   (shape `[N, M]`), `src1 = X^T` (shape `[K, M]`), m_param=K,
//!   n_param=N, k_param=M.  Two transposes required.
//!
//! ## Constraints
//!
//! mlx-native's f32 matmul kernel requires `K >= 32` (one Metal
//! tensor-tile minimum).  Synthetic fixtures must pad K accordingly.
//! Production transformer dimensions (Qwen35 K=4096, Gemma4 K=3584)
//! are well above this floor.

use anyhow::{anyhow, Context, Result};
use mlx_native::{
    ops::dense_mm_f32_f32::{dense_matmul_f32_f32_tensor, DenseMmF32F32Params},
    ops::transpose::transpose_2d,
    DType, KernelRegistry, MlxBuffer, MlxDevice,
};

/// Allocate an `[rows, cols]` f32 GPU buffer and copy values in.
fn alloc_f32_2d(device: &MlxDevice, values: &[f32], rows: usize, cols: usize) -> Result<MlxBuffer> {
    if values.len() != rows * cols {
        return Err(anyhow!(
            "alloc_f32_2d: values.len()={} != rows*cols={}*{}={}",
            values.len(),
            rows,
            cols,
            rows * cols
        ));
    }
    let mut buf = device
        .alloc_buffer(rows * cols * 4, DType::F32, vec![rows, cols])
        .map_err(|e| anyhow!("alloc f32 [{rows}, {cols}]: {e}"))?;
    let dst = buf
        .as_mut_slice::<f32>()
        .map_err(|e| anyhow!("as_mut_slice f32: {e}"))?;
    dst.copy_from_slice(values);
    Ok(buf)
}

/// Allocate an uninitialised `[rows, cols]` f32 GPU buffer.
fn alloc_f32_2d_uninit(device: &MlxDevice, rows: usize, cols: usize) -> Result<MlxBuffer> {
    device
        .alloc_buffer(rows * cols * 4, DType::F32, vec![rows, cols])
        .map_err(|e| anyhow!("alloc f32 [{rows}, {cols}]: {e}"))
}

/// Read an MlxBuffer's f32 contents to an owned `Vec<f32>`.
fn read_f32(buf: &MlxBuffer) -> Result<Vec<f32>> {
    let s: &[f32] = buf.as_slice().map_err(|e| anyhow!("as_slice f32: {e}"))?;
    Ok(s.to_vec())
}

/// GPU forward matmul: `Y = X @ W` for f32 row-major inputs.
///
/// `X` shape `[m, k]`, `W` shape `[k, n]`, returns `Y` of shape `[m, n]`.
/// `k` must be `>= 32` (mlx-native f32 matmul kernel constraint).
///
/// Issues:
/// 1. transpose_2d to produce W^T (shape [n, k])
/// 2. dense_matmul_f32_f32_tensor with (src0=W^T, src1=X) → Y
/// 3. commit_and_wait
pub fn matmul_forward_f32(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    n: usize,
) -> Result<Vec<f32>> {
    if x.len() != m * k {
        return Err(anyhow!("x.len()={} != m*k={}*{}={}", x.len(), m, k, m * k));
    }
    if w.len() != k * n {
        return Err(anyhow!("w.len()={} != k*n={}*{}={}", w.len(), k, n, k * n));
    }
    if k < 32 {
        return Err(anyhow!(
            "matmul_forward_f32: k={k} but mlx-native f32 matmul kernel requires k >= 32"
        ));
    }

    let x_buf = alloc_f32_2d(device, x, m, k)?;
    let w_buf = alloc_f32_2d(device, w, k, n)?;
    let w_t_buf = alloc_f32_2d_uninit(device, n, k)?;
    let mut y_buf = alloc_f32_2d_uninit(device, m, n)?;

    let mut encoder = device.command_encoder().map_err(|e| anyhow!("encoder: {e}"))?;

    transpose_2d(
        &mut encoder,
        registry,
        device.metal_device(),
        &w_buf,
        &w_t_buf,
        k,
        n,
        DType::F32,
    )
    .context("matmul_forward_f32: transpose W → W^T")?;

    let params = DenseMmF32F32Params {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        src0_batch: 1,
        src1_batch: 1,
    };
    dense_matmul_f32_f32_tensor(
        &mut encoder,
        registry,
        device,
        &w_t_buf,
        &x_buf,
        &mut y_buf,
        &params,
    )
    .context("matmul_forward_f32: dense_matmul Y = X @ W")?;

    encoder
        .commit_and_wait()
        .map_err(|e| anyhow!("commit_and_wait: {e}"))?;

    read_f32(&y_buf)
}

/// GPU backward matmul.  Given inputs `X` shape `[m, k]`, `W` shape
/// `[k, n]`, and upstream gradient `dY` shape `[m, n]`, returns
/// `(dX, dW)` of shapes `[m, k]` and `[k, n]`.
///
/// `n` and `k` must both be `>= 32` (the dense_matmul kernel constraint
/// applies to the contract dim of each backward call).
///
/// Backward dispatches:
/// 1. `dX = dY @ W^T` — call dense_matmul with `src0 = W [k, n]`,
///    `src1 = dY [m, n]`, m=M, n=K, k=N.
/// 2. `dW = X^T @ dY` — needs `src0 = dY^T [n, m]` and
///    `src1 = X^T [k, m]`.  Two transposes.  Call dense_matmul with
///    m=K, n=N, k=M.  Then transpose result.  (Actually no — see below.)
///
/// Subtlety for (2): dense_matmul's constraint is `k >= 32`.  Here
/// k_param = M (the original batch dim).  For typical synthetic
/// fixtures M might be small (e.g. M=4).  To bypass this, we instead
/// compute `dW` as `(dY^T @ X)^T`:
///   - dst_temp [N, K] = sum_m dY[m, n] · X[m, k]
///     dense_matmul map: `dst_temp[n, k] = sum_m src0[k, m] · src1[n, m]`
///     so src0 = X^T [K, M], src1 = dY^T [N, M], m=N, n=K, k=M.
///     ALSO has k=M issue.
///
/// Pure-GPU mitigation: pad M by writing dY as `[m_padded, n]` with
/// trailing zeros where m_padded is the smallest multiple-of-32 >= m.
/// At iter 8b we instead require `m >= 32` for backward — synthetic
/// fixtures are sized accordingly.  Production transformer batches
/// (M = batch × seq = 4×512 = 2048) are well above the floor.
pub fn matmul_backward_f32(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    n: usize,
    dy: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    if x.len() != m * k {
        return Err(anyhow!("x.len()={} != m*k={}*{}={}", x.len(), m, k, m * k));
    }
    if w.len() != k * n {
        return Err(anyhow!("w.len()={} != k*n={}*{}={}", w.len(), k, n, k * n));
    }
    if dy.len() != m * n {
        return Err(anyhow!("dy.len()={} != m*n={}*{}={}", dy.len(), m, n, m * n));
    }
    if n < 32 {
        return Err(anyhow!(
            "matmul_backward_f32: n={n} but kernel requires n >= 32 for dX path"
        ));
    }
    if m < 32 {
        return Err(anyhow!(
            "matmul_backward_f32: m={m} but kernel requires m >= 32 for dW path \
             (k_param == M in the dW dispatch); pad m to a multiple of 32 for \
             smaller fixtures"
        ));
    }
    if k < 32 {
        return Err(anyhow!(
            "matmul_backward_f32: k={k} but transpose path requires k >= 32"
        ));
    }

    // Allocate input buffers + intermediates.
    let x_buf = alloc_f32_2d(device, x, m, k)?;
    let w_buf = alloc_f32_2d(device, w, k, n)?;
    let dy_buf = alloc_f32_2d(device, dy, m, n)?;

    let mut dx_buf = alloc_f32_2d_uninit(device, m, k)?;
    let dy_t_buf = alloc_f32_2d_uninit(device, n, m)?;
    let x_t_buf = alloc_f32_2d_uninit(device, k, m)?;
    let mut dw_buf = alloc_f32_2d_uninit(device, k, n)?;

    let mut encoder = device.command_encoder().map_err(|e| anyhow!("encoder: {e}"))?;

    // (1) dX = dY @ W^T = sum_n dY[m, n] · W[k, n]
    //     dense_matmul map: dst[m, k] = sum_n src0[k, n] · src1[m, n]
    //     → src0 = W [k, n] (no transpose), src1 = dY [m, n]
    //     params: m=M, n=K, k=N
    let dx_params = DenseMmF32F32Params {
        m: m as u32,
        n: k as u32,
        k: n as u32,
        src0_batch: 1,
        src1_batch: 1,
    };
    dense_matmul_f32_f32_tensor(
        &mut encoder,
        registry,
        device,
        &w_buf,
        &dy_buf,
        &mut dx_buf,
        &dx_params,
    )
    .context("matmul_backward_f32: dense_matmul dX = dY @ W^T")?;

    // (2) dW = X^T @ dY = sum_m X[m, k] · dY[m, n].
    //     We compute dW directly into [k, n] form by:
    //       dst[k, n] = sum_m src0[n, m] · src1[k, m]
    //     So src0 = dY^T [n, m], src1 = X^T [k, m].
    //     params: m=K, n=N, k=M.
    transpose_2d(
        &mut encoder,
        registry,
        device.metal_device(),
        &dy_buf,
        &dy_t_buf,
        m,
        n,
        DType::F32,
    )
    .context("matmul_backward_f32: transpose dY → dY^T")?;
    transpose_2d(
        &mut encoder,
        registry,
        device.metal_device(),
        &x_buf,
        &x_t_buf,
        m,
        k,
        DType::F32,
    )
    .context("matmul_backward_f32: transpose X → X^T")?;

    let dw_params = DenseMmF32F32Params {
        m: k as u32,
        n: n as u32,
        k: m as u32,
        src0_batch: 1,
        src1_batch: 1,
    };
    dense_matmul_f32_f32_tensor(
        &mut encoder,
        registry,
        device,
        &dy_t_buf,
        &x_t_buf,
        &mut dw_buf,
        &dw_params,
    )
    .context("matmul_backward_f32: dense_matmul dW = X^T @ dY")?;

    encoder
        .commit_and_wait()
        .map_err(|e| anyhow!("commit_and_wait: {e}"))?;

    let dx = read_f32(&dx_buf)?;
    let dw = read_f32(&dw_buf)?;
    Ok((dx, dw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::autograd::{backward, matmul as cpu_matmul, sum as cpu_sum, Tape, Tensor};

    /// Assert two f32 vectors are close in relative+absolute terms.
    fn assert_close(actual: &[f32], expected: &[f32], rel_tol: f32, abs_tol: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: len mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let scale = a.abs().max(e.abs()).max(1.0);
            assert!(
                diff <= abs_tol || diff / scale <= rel_tol,
                "{label}[{i}]: actual={a} expected={e} diff={diff} \
                 (abs_tol={abs_tol}, rel_tol={rel_tol})"
            );
        }
    }

    /// CPU-oracle reference: compute (Y, dX, dW) for `Y = X @ W; loss = sum(Y)`.
    /// Returns `(y, dx, dw)`.
    fn cpu_oracle_matmul_grad(
        x: &[f32],
        m: usize,
        k: usize,
        w: &[f32],
        n: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let tape = Tape::new();
        let xt = Tensor::from_vec(&tape, x.to_vec(), vec![m, k]).unwrap();
        let wt = Tensor::from_vec(&tape, w.to_vec(), vec![k, n]).unwrap();
        let y = cpu_matmul(&xt, &wt).unwrap();
        let y_vec = y.to_vec();
        // For backward parity we need dY = ones (so loss = sum(Y), dY = 1 everywhere).
        let loss = cpu_sum(&y).unwrap();
        let grads = backward(&loss).unwrap();
        let dx = grads[xt.node_idx()].clone().expect("grad x");
        let dw = grads[wt.node_idx()].clone().expect("grad w");
        (y_vec, dx, dw)
    }

    #[test]
    fn gpu_matmul_forward_parity_with_cpu_oracle_4_32_4() {
        // K=32 satisfies the kernel's K >= 32 minimum.
        let m = 4;
        let k = 32;
        let n = 4;
        // Deterministic synthetic inputs (fp32-clean values).
        let x: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let w: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.005 + 0.02).collect();

        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        let y_gpu = matmul_forward_f32(&device, &mut registry, &x, m, k, &w, n)
            .expect("gpu forward");
        let (y_cpu, _, _) = cpu_oracle_matmul_grad(&x, m, k, &w, n);

        // fp32 matmul on GPU vs CPU: tensor-cores may sum in different
        // order; allow a relative tolerance of 1e-4 (well within
        // single-precision fma accuracy at this size).
        assert_close(&y_gpu, &y_cpu, 1e-4, 1e-5, "matmul forward GPU↔CPU");
    }

    #[test]
    fn gpu_matmul_backward_parity_with_cpu_oracle_32_32_32() {
        // Per the doc-comments on matmul_backward_f32, M, K, N all need
        // to be >= 32 for the backward path.  Pick the smallest test
        // shape that satisfies this floor.
        let m = 32;
        let k = 32;
        let n = 32;
        let x: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.0007 - 0.1).collect();
        let w: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.0011 + 0.05).collect();
        // For the test, dY = ones (matches loss = sum(Y)).
        let dy = vec![1.0_f32; m * n];

        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        let (dx_gpu, dw_gpu) =
            matmul_backward_f32(&device, &mut registry, &x, m, k, &w, n, &dy)
                .expect("gpu backward");
        let (_, dx_cpu, dw_cpu) = cpu_oracle_matmul_grad(&x, m, k, &w, n);

        // Backward sums in different order vs forward; tolerance same.
        assert_close(&dx_gpu, &dx_cpu, 1e-4, 1e-5, "matmul backward dX GPU↔CPU");
        assert_close(&dw_gpu, &dw_cpu, 1e-4, 1e-5, "matmul backward dW GPU↔CPU");
    }

    #[test]
    fn gpu_matmul_forward_below_kernel_floor_errors() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        // K=8 below the 32 minimum.
        let r = matmul_forward_f32(&device, &mut registry, &[0.0; 8], 1, 8, &[0.0; 8], 1);
        let err = r.expect_err("k<32 must error");
        assert!(format!("{err}").contains("k >= 32"));
    }

    #[test]
    fn gpu_matmul_backward_below_kernel_floor_errors() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        // n=8 below 32.
        let r = matmul_backward_f32(
            &device,
            &mut registry,
            &[0.0; 32 * 32],
            32,
            32,
            &[0.0; 32 * 8],
            8,
            &[0.0; 32 * 8],
        );
        let err = r.expect_err("n<32 must error");
        assert!(format!("{err}").contains("n >= 32"));
    }

    #[test]
    fn gpu_matmul_forward_known_value_unit_x_diagonal_w() {
        // X = ones [m=2, k=32], W = 0.5 × I_32 padded to [k=32, n=32]
        // (pad columns 32 → just [k=32, n=32] identity scaled by 0.5).
        // Expected Y[i, j] = 0.5 * Σ_k X[i, k] · W[k, j]
        //                  = 0.5 * 1.0 (since W is 0.5 × I, only k=j contributes)
        //                  = 0.5
        let m = 2;
        let k = 32;
        let n = 32;
        let x = vec![1.0_f32; m * k];
        let mut w = vec![0.0_f32; k * n];
        for d in 0..32 {
            w[d * n + d] = 0.5;
        }

        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        let y = matmul_forward_f32(&device, &mut registry, &x, m, k, &w, n)
            .expect("gpu forward");
        let expected = vec![0.5_f32; m * n];
        assert_close(&y, &expected, 1e-6, 1e-7, "matmul I·X");
    }
}
