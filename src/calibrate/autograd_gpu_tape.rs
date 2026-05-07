//! **GPU autograd tape** for multi-op forward chains (ADR-020 Track 1, iter 8b.1).
//!
//! Builds on `autograd_gpu` (iter 8b standalone matmul) by wrapping the
//! GPU primitives in a tape structure analogous to the CPU oracle's
//! `Tape`/`Tensor`.  Enables multi-op composition like
//! `Z = matmul(matmul(X, W1), W2)` with `backward(Z, dZ)` producing
//! `(dX, dW1, dW2)` end-to-end on Metal.
//!
//! Per `~/Documents/mantra.txt`: production runs on GPU.  CPU oracle
//! (`crate::calibrate::autograd`) is reachable only from `#[cfg(test)]`
//! parity assertions.
//!
//! ## Design
//!
//! Tape: `Rc<GpuTapeInner>` with interior mutability via `RefCell`s.
//! Each tape owns the `MlxDevice` it was constructed with + a
//! `KernelRegistry` for kernel pipeline caching across ops.  Each
//! `GpuNode` records its `OpKind` (variant for op-specific reverse
//! dispatch), the forward `MlxBuffer` value, and its shape.
//!
//! Backward dispatches per-`OpKind` in a single encoder pass per
//! reverse-walk step — N ops produce N encoder commits.  Future
//! iterations can amortize this by holding one encoder open across
//! the full reverse pass; iter 8b.1 stays correct-first.
//!
//! ## What's NOT here yet
//!
//! - Activation ops (softmax, silu, gelu).  Iter 8c.
//! - Norms (RMSNorm, LayerNorm).  Iter 8d.
//! - Sum / mean reduction.  Iter 8e (when GPU sum kernel lands; until
//!   then the user supplies `output_grad` directly via the typed API).
//! - Higher-order autograd.  Forward-then-`backward` only.
//! - Operator overloads (`+`, `*`).  Explicit function calls only —
//!   keeps the tape side-effect locus inspectable.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{anyhow, Context, Result};
use mlx_native::{
    ops::dense_mm_f32_f32::{dense_matmul_f32_f32_tensor, DenseMmF32F32Params},
    ops::transpose::transpose_2d,
    DType, KernelRegistry, MlxBuffer, MlxDevice,
};

/// Op-specific data needed to dispatch a node's backward pass.  The
/// enum is closed: every op variant has a matching arm in
/// [`backward_dispatch`].
#[derive(Debug, Clone)]
enum OpKind {
    /// Constant input — no backward (no parents).
    Leaf,
    /// `Y = X @ W` for `X` shape `[m, k]`, `W` shape `[k, n]`,
    /// `Y` shape `[m, n]`.  `k >= 32` enforced at forward time.
    Matmul {
        lhs_idx: usize,
        rhs_idx: usize,
        m: usize,
        k: usize,
        n: usize,
    },
}

struct GpuNode {
    op: OpKind,
    /// Forward-computed value.  Cloning is a refcount bump.
    value: MlxBuffer,
    /// Retained for diagnostics + future shape-checking; not currently
    /// read by ops (each op reads `shape` directly off the `GpuTensor`).
    #[allow(dead_code)]
    shape: Vec<usize>,
}

struct GpuTapeInner {
    device: MlxDevice,
    registry: RefCell<KernelRegistry>,
    nodes: RefCell<Vec<GpuNode>>,
}

/// User-facing tape handle.  Cheap to clone (Rc bump).
#[derive(Clone)]
pub struct GpuTape(Rc<GpuTapeInner>);

impl GpuTape {
    pub fn new(device: MlxDevice) -> Self {
        Self(Rc::new(GpuTapeInner {
            device,
            registry: RefCell::new(KernelRegistry::new()),
            nodes: RefCell::new(Vec::new()),
        }))
    }

    pub fn device(&self) -> &MlxDevice {
        &self.0.device
    }

    fn ptr_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    fn push_node(&self, op: OpKind, value: MlxBuffer, shape: Vec<usize>) -> usize {
        let mut nodes = self.0.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(GpuNode { op, value, shape });
        idx
    }

    fn node_count(&self) -> usize {
        self.0.nodes.borrow().len()
    }

    fn with_node<R>(&self, idx: usize, f: impl FnOnce(&GpuNode) -> R) -> R {
        let nodes = self.0.nodes.borrow();
        f(&nodes[idx])
    }
}

/// User-facing GPU tensor.  Carries shape + a tape link.  Cheap to
/// clone.
#[derive(Clone)]
pub struct GpuTensor {
    tape: GpuTape,
    node_idx: usize,
    shape: Vec<usize>,
}

impl GpuTensor {
    /// Construct a leaf tensor on `tape` from CPU values + shape.
    pub fn from_vec(tape: &GpuTape, values: &[f32], shape: Vec<usize>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if values.len() != numel {
            return Err(anyhow!(
                "GpuTensor::from_vec: values.len()={} != numel={}",
                values.len(),
                numel
            ));
        }
        let mut buf = tape
            .device()
            .alloc_buffer(numel * 4, DType::F32, shape.clone())
            .map_err(|e| anyhow!("alloc leaf buffer: {e}"))?;
        let dst = buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("as_mut_slice f32 leaf: {e}"))?;
        dst.copy_from_slice(values);
        let node_idx = tape.push_node(OpKind::Leaf, buf, shape.clone());
        Ok(Self {
            tape: tape.clone(),
            node_idx,
            shape,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn node_idx(&self) -> usize {
        self.node_idx
    }

    /// Read forward-computed values to CPU.  Caller responsibility:
    /// any pending GPU work for this tensor must already be committed
    /// + waited (the public ops do `commit_and_wait` before returning).
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        self.tape.with_node(self.node_idx, |n| {
            let s: &[f32] = n.value.as_slice().map_err(|e| anyhow!("as_slice: {e}"))?;
            Ok(s.to_vec())
        })
    }

    fn assert_same_tape(&self, other: &Self) -> Result<()> {
        if self.tape.ptr_eq(&other.tape) {
            Ok(())
        } else {
            Err(anyhow!("GpuTensor: cross-tape op (lhs.tape != rhs.tape)"))
        }
    }
}

/// Allocate an uninitialised f32 GPU buffer with `[rows, cols]` shape.
fn alloc_f32_2d_uninit(device: &MlxDevice, rows: usize, cols: usize) -> Result<MlxBuffer> {
    device
        .alloc_buffer(rows * cols * 4, DType::F32, vec![rows, cols])
        .map_err(|e| anyhow!("alloc f32 [{rows}, {cols}]: {e}"))
}

/// `Y = X @ W` — wraps `dense_matmul_f32_f32_tensor` + `transpose_2d`
/// into a tape-aware op.  X shape `[m, k]`, W shape `[k, n]`,
/// Y shape `[m, n]`.  Requires `k >= 32` per the kernel constraint.
pub fn matmul(lhs: &GpuTensor, rhs: &GpuTensor) -> Result<GpuTensor> {
    lhs.assert_same_tape(rhs)?;
    let (m, k_l) = parse_2d_shape(&lhs.shape, "lhs")?;
    let (k_r, n) = parse_2d_shape(&rhs.shape, "rhs")?;
    if k_l != k_r {
        return Err(anyhow!(
            "matmul: shape mismatch lhs={:?} rhs={:?} (need lhs.last == rhs.first)",
            lhs.shape,
            rhs.shape
        ));
    }
    let k = k_l;
    if k < 32 {
        return Err(anyhow!(
            "matmul: k={k} but mlx-native f32 matmul kernel requires k >= 32"
        ));
    }

    let tape = &lhs.tape;
    let device = tape.device();
    let mut registry = tape.0.registry.borrow_mut();

    // Forward: Y = X @ W.
    // dense_matmul: dst[m,n] = sum_k src0[n,k] · src1[m,k]
    //   = src1 @ src0^T.  So src0 = W^T (shape [n, k]), src1 = X (shape [m, k]).
    let w_t_buf = alloc_f32_2d_uninit(device, n, k)?;
    let mut y_buf = alloc_f32_2d_uninit(device, m, n)?;

    let mut encoder = device
        .command_encoder()
        .map_err(|e| anyhow!("matmul: encoder: {e}"))?;

    // We need to access the input buffers from the tape.  Use a scoped
    // borrow to read them out as cheap MlxBuffer clones (Arc bump).
    let (x_buf, w_buf) = {
        let nodes = tape.0.nodes.borrow();
        (
            nodes[lhs.node_idx].value.clone(),
            nodes[rhs.node_idx].value.clone(),
        )
    };

    transpose_2d(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &w_buf,
        &w_t_buf,
        k,
        n,
        DType::F32,
    )
    .context("matmul: transpose W → W^T")?;

    let params = DenseMmF32F32Params {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        src0_batch: 1,
        src1_batch: 1,
    };
    dense_matmul_f32_f32_tensor(
        &mut encoder,
        &mut registry,
        device,
        &w_t_buf,
        &x_buf,
        &mut y_buf,
        &params,
    )
    .context("matmul: dense_matmul Y = X @ W")?;

    encoder
        .commit_and_wait()
        .map_err(|e| anyhow!("matmul: commit_and_wait: {e}"))?;

    drop(registry);

    let lhs_idx = lhs.node_idx;
    let rhs_idx = rhs.node_idx;
    let node_idx = tape.push_node(
        OpKind::Matmul {
            lhs_idx,
            rhs_idx,
            m,
            k,
            n,
        },
        y_buf,
        vec![m, n],
    );

    Ok(GpuTensor {
        tape: tape.clone(),
        node_idx,
        shape: vec![m, n],
    })
}

fn parse_2d_shape(shape: &[usize], label: &str) -> Result<(usize, usize)> {
    if shape.len() != 2 {
        return Err(anyhow!(
            "{label} must be 2D for matmul; got shape={:?}",
            shape
        ));
    }
    Ok((shape[0], shape[1]))
}

/// Run the reverse pass from `output` with seed gradient `output_grad`.
/// Returns `Vec<Option<MlxBuffer>>` indexed by node — `Some(grad)` for
/// nodes that participated in `output`'s subgraph, `None` otherwise.
///
/// `output_grad` must have the same shape + dtype as `output`'s value.
/// Caller supplies it explicitly; the typed API is preferable to an
/// implicit `sum` reduction (no GPU sum kernel yet — iter 8e adds one).
pub fn backward(
    output: &GpuTensor,
    output_grad: MlxBuffer,
) -> Result<Vec<Option<MlxBuffer>>> {
    let tape = &output.tape;
    let n_nodes = tape.node_count();

    let mut grads: Vec<Option<MlxBuffer>> = (0..n_nodes).map(|_| None).collect();
    grads[output.node_idx] = Some(output_grad);

    // Walk in reverse; for each node that has an accumulated grad,
    // dispatch backward into parents.
    for node_idx in (0..n_nodes).rev() {
        let Some(my_grad) = grads[node_idx].clone() else {
            continue;
        };
        let (op, parent_grads) = backward_dispatch(tape, node_idx, &my_grad)?;
        // Distribute `parent_grads` into the accumulator.
        match op {
            OpKind::Leaf => {} // no parents
            OpKind::Matmul { lhs_idx, rhs_idx, .. } => {
                accumulate(&mut grads, lhs_idx, &parent_grads[0], tape)?;
                accumulate(&mut grads, rhs_idx, &parent_grads[1], tape)?;
            }
        }
    }
    Ok(grads)
}

/// Add `contribution` into `grads[parent_idx]` (or initialize it if
/// `None`).  GPU elementwise add is composed via a single dispatch
/// using mlx-native's `elementwise_add`.
fn accumulate(
    grads: &mut [Option<MlxBuffer>],
    parent_idx: usize,
    contribution: &MlxBuffer,
    tape: &GpuTape,
) -> Result<()> {
    match grads[parent_idx].take() {
        None => {
            grads[parent_idx] = Some(contribution.clone());
            Ok(())
        }
        Some(existing) => {
            // GPU elementwise add: existing += contribution.
            let device = tape.device();
            let mut registry = tape.0.registry.borrow_mut();
            let n = existing.byte_len() / 4;
            let out = device
                .alloc_buffer(n * 4, DType::F32, existing.shape().to_vec())
                .map_err(|e| anyhow!("accumulate: alloc out: {e}"))?;
            let mut encoder = device
                .command_encoder()
                .map_err(|e| anyhow!("accumulate: encoder: {e}"))?;
            mlx_native::ops::elementwise::elementwise_add(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &existing,
                contribution,
                &out,
                n,
                DType::F32,
            )
            .map_err(|e| anyhow!("accumulate: elementwise_add: {e}"))?;
            encoder
                .commit_and_wait()
                .map_err(|e| anyhow!("accumulate: commit_and_wait: {e}"))?;
            grads[parent_idx] = Some(out);
            Ok(())
        }
    }
}

/// Dispatch the per-op backward, returning a fresh `OpKind` clone (for
/// the caller to match on parent indices) + a `Vec<MlxBuffer>` of
/// per-parent gradient contributions.
fn backward_dispatch(
    tape: &GpuTape,
    node_idx: usize,
    out_grad: &MlxBuffer,
) -> Result<(OpKind, Vec<MlxBuffer>)> {
    let op = tape.with_node(node_idx, |n| n.op.clone());
    match op.clone() {
        OpKind::Leaf => Ok((op, Vec::new())),
        OpKind::Matmul {
            lhs_idx,
            rhs_idx,
            m,
            k,
            n,
        } => {
            let device = tape.device();
            let mut registry = tape.0.registry.borrow_mut();

            // Backward floor enforcement (mirrors autograd_gpu).
            if m < 32 {
                return Err(anyhow!(
                    "backward matmul: m={m} but k_param=M in dW dispatch \
                     requires m >= 32"
                ));
            }
            if n < 32 {
                return Err(anyhow!(
                    "backward matmul: n={n} but k_param=N in dX dispatch \
                     requires n >= 32"
                ));
            }
            if k < 32 {
                return Err(anyhow!("backward matmul: k={k} but k >= 32 required"));
            }

            // Read parent buffers from tape.
            let (x_buf, w_buf) = {
                let nodes = tape.0.nodes.borrow();
                (
                    nodes[lhs_idx].value.clone(),
                    nodes[rhs_idx].value.clone(),
                )
            };

            let mut dx_buf = alloc_f32_2d_uninit(device, m, k)?;
            let dy_t_buf = alloc_f32_2d_uninit(device, n, m)?;
            let x_t_buf = alloc_f32_2d_uninit(device, k, m)?;
            let mut dw_buf = alloc_f32_2d_uninit(device, k, n)?;

            let mut encoder = device
                .command_encoder()
                .map_err(|e| anyhow!("backward matmul: encoder: {e}"))?;

            // (1) dX = dY @ W^T
            //     dense_matmul: dst[m, k] = sum_n src0[k, n] · src1[m, n]
            //     so src0 = W [k, n] (no transpose), src1 = dY [m, n]
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
                &mut registry,
                device,
                &w_buf,
                out_grad,
                &mut dx_buf,
                &dx_params,
            )
            .context("backward matmul: dX = dY @ W^T")?;

            // (2) dW = X^T @ dY
            //     dense_matmul: dst[k, n] = sum_m src0[n, m] · src1[k, m]
            //     so src0 = dY^T [n, m], src1 = X^T [k, m]
            //     params: m=K, n=N, k=M
            transpose_2d(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                out_grad,
                &dy_t_buf,
                m,
                n,
                DType::F32,
            )
            .context("backward matmul: transpose dY → dY^T")?;
            transpose_2d(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &x_buf,
                &x_t_buf,
                m,
                k,
                DType::F32,
            )
            .context("backward matmul: transpose X → X^T")?;

            let dw_params = DenseMmF32F32Params {
                m: k as u32,
                n: n as u32,
                k: m as u32,
                src0_batch: 1,
                src1_batch: 1,
            };
            dense_matmul_f32_f32_tensor(
                &mut encoder,
                &mut registry,
                device,
                &dy_t_buf,
                &x_t_buf,
                &mut dw_buf,
                &dw_params,
            )
            .context("backward matmul: dW = X^T @ dY")?;

            encoder
                .commit_and_wait()
                .map_err(|e| anyhow!("backward matmul: commit_and_wait: {e}"))?;

            Ok((op, vec![dx_buf, dw_buf]))
        }
    }
}

/// Build a GPU `MlxBuffer` of shape `[rows, cols]` filled with `1.0`
/// — convenient seed for `backward` when the conceptual loss is
/// `sum(output)` (i.e. dY = ones).  Returned buffer is on `tape.device()`.
pub fn ones_like(tape: &GpuTape, shape: &[usize]) -> Result<MlxBuffer> {
    let numel: usize = shape.iter().product();
    let mut buf = tape
        .device()
        .alloc_buffer(numel * 4, DType::F32, shape.to_vec())
        .map_err(|e| anyhow!("ones_like: alloc: {e}"))?;
    let dst = buf
        .as_mut_slice::<f32>()
        .map_err(|e| anyhow!("ones_like: as_mut_slice: {e}"))?;
    dst.fill(1.0);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::autograd::{
        backward as cpu_backward, matmul as cpu_matmul, sum as cpu_sum, Tape, Tensor,
    };

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

    #[test]
    fn gpu_tape_matmul_forward_parity() {
        let m = 32;
        let k = 32;
        let n = 32;
        let x: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.0009 + 0.01).collect();
        let w: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.0013 - 0.03).collect();

        // GPU
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &x, vec![m, k]).expect("xt");
        let wt = GpuTensor::from_vec(&tape, &w, vec![k, n]).expect("wt");
        let yt = matmul(&xt, &wt).expect("matmul");
        let y_gpu = yt.to_vec().expect("readback");

        // CPU oracle
        let cpu_tape = Tape::new();
        let cx = Tensor::from_vec(&cpu_tape, x.clone(), vec![m, k]).unwrap();
        let cw = Tensor::from_vec(&cpu_tape, w.clone(), vec![k, n]).unwrap();
        let cy = cpu_matmul(&cx, &cw).unwrap();
        let y_cpu = cy.to_vec();

        assert_close(&y_gpu, &y_cpu, 1e-4, 1e-5, "tape forward parity");
    }

    #[test]
    fn gpu_tape_matmul_backward_parity_with_dy_ones() {
        let m = 32;
        let k = 32;
        let n = 32;
        let x: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let w: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.002 + 0.04).collect();

        // GPU forward + backward via tape
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &x, vec![m, k]).expect("xt");
        let wt = GpuTensor::from_vec(&tape, &w, vec![k, n]).expect("wt");
        let yt = matmul(&xt, &wt).expect("matmul");
        let dy = ones_like(&tape, &[m, n]).expect("dy ones");
        let grads = backward(&yt, dy).expect("backward");
        let dx_gpu: &[f32] = grads[xt.node_idx()]
            .as_ref()
            .expect("grad x")
            .as_slice()
            .expect("dx slice");
        let dw_gpu: &[f32] = grads[wt.node_idx()]
            .as_ref()
            .expect("grad w")
            .as_slice()
            .expect("dw slice");

        // CPU oracle (loss = sum(Y), so dY = ones implicitly)
        let cpu_tape = Tape::new();
        let cx = Tensor::from_vec(&cpu_tape, x.clone(), vec![m, k]).unwrap();
        let cw = Tensor::from_vec(&cpu_tape, w.clone(), vec![k, n]).unwrap();
        let cy = cpu_matmul(&cx, &cw).unwrap();
        let loss = cpu_sum(&cy).unwrap();
        let cgrads = cpu_backward(&loss).unwrap();
        let dx_cpu = cgrads[cx.node_idx()].clone().unwrap();
        let dw_cpu = cgrads[cw.node_idx()].clone().unwrap();

        assert_close(dx_gpu, &dx_cpu, 1e-4, 1e-5, "tape backward dX");
        assert_close(dw_gpu, &dw_cpu, 1e-4, 1e-5, "tape backward dW");
    }

    #[test]
    fn gpu_tape_two_matmul_chain_backward_parity() {
        // Z = (X @ W1) @ W2; dZ = ones.
        // Verify dX, dW1, dW2 all match CPU oracle.
        let m = 32;
        let k1 = 32;
        let k2 = 32;
        let n = 32;
        let x: Vec<f32> = (0..(m * k1)).map(|i| (i as f32) * 0.0007 + 0.001).collect();
        let w1: Vec<f32> = (0..(k1 * k2)).map(|i| (i as f32) * 0.0011 - 0.02).collect();
        let w2: Vec<f32> = (0..(k2 * n)).map(|i| (i as f32) * 0.0015 + 0.03).collect();

        // GPU
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &x, vec![m, k1]).expect("xt");
        let w1t = GpuTensor::from_vec(&tape, &w1, vec![k1, k2]).expect("w1t");
        let w2t = GpuTensor::from_vec(&tape, &w2, vec![k2, n]).expect("w2t");
        let yt = matmul(&xt, &w1t).expect("matmul1");
        let zt = matmul(&yt, &w2t).expect("matmul2");
        let dz = ones_like(&tape, &[m, n]).expect("dz");
        let grads = backward(&zt, dz).expect("backward");
        let dx_gpu: &[f32] = grads[xt.node_idx()]
            .as_ref()
            .expect("grad x")
            .as_slice()
            .expect("dx slice");
        let dw1_gpu: &[f32] = grads[w1t.node_idx()]
            .as_ref()
            .expect("grad w1")
            .as_slice()
            .expect("dw1 slice");
        let dw2_gpu: &[f32] = grads[w2t.node_idx()]
            .as_ref()
            .expect("grad w2")
            .as_slice()
            .expect("dw2 slice");

        // CPU oracle
        let cpu_tape = Tape::new();
        let cx = Tensor::from_vec(&cpu_tape, x.clone(), vec![m, k1]).unwrap();
        let cw1 = Tensor::from_vec(&cpu_tape, w1.clone(), vec![k1, k2]).unwrap();
        let cw2 = Tensor::from_vec(&cpu_tape, w2.clone(), vec![k2, n]).unwrap();
        let cy = cpu_matmul(&cx, &cw1).unwrap();
        let cz = cpu_matmul(&cy, &cw2).unwrap();
        let cl = cpu_sum(&cz).unwrap();
        let cgrads = cpu_backward(&cl).unwrap();
        let dx_cpu = cgrads[cx.node_idx()].clone().unwrap();
        let dw1_cpu = cgrads[cw1.node_idx()].clone().unwrap();
        let dw2_cpu = cgrads[cw2.node_idx()].clone().unwrap();

        // Two-stage matmul accumulates more rounding error; loosen abs_tol
        // slightly while keeping rel_tol tight.
        assert_close(dx_gpu, &dx_cpu, 1e-4, 1e-4, "two-matmul dX");
        assert_close(dw1_gpu, &dw1_cpu, 1e-4, 1e-4, "two-matmul dW1");
        assert_close(dw2_gpu, &dw2_cpu, 1e-4, 1e-4, "two-matmul dW2");
    }

    #[test]
    fn gpu_tape_cross_tape_op_errors() {
        let device = MlxDevice::new().expect("device");
        let t1 = GpuTape::new(device);
        let device2 = MlxDevice::new().expect("device2");
        let t2 = GpuTape::new(device2);
        let a = GpuTensor::from_vec(&t1, &[0.0; 32 * 32], vec![32, 32]).unwrap();
        let b = GpuTensor::from_vec(&t2, &[0.0; 32 * 32], vec![32, 32]).unwrap();
        match matmul(&a, &b) {
            Err(e) => {
                let msg = format!("{e}");
                assert!(msg.contains("cross-tape"), "got: {msg}");
            }
            Ok(_) => panic!("cross-tape matmul must error"),
        }
    }

    #[test]
    fn gpu_tape_shape_mismatch_errors() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let a = GpuTensor::from_vec(&tape, &[0.0; 32 * 32], vec![32, 32]).unwrap();
        let b = GpuTensor::from_vec(&tape, &[0.0; 33 * 32], vec![33, 32]).unwrap();
        match matmul(&a, &b) {
            Err(e) => {
                let msg = format!("{e}");
                assert!(msg.contains("shape mismatch"), "got: {msg}");
            }
            Ok(_) => panic!("shape mismatch must error"),
        }
    }

    #[test]
    fn gpu_tape_backward_nonparticipating_node_grad_is_none() {
        // Construct two unrelated matmul subgraphs; backward from
        // subgraph A should leave subgraph B's nodes with grad=None.
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let m = 32;
        let xa: Vec<f32> = vec![0.5; m * m];
        let wa: Vec<f32> = vec![0.25; m * m];
        let xb: Vec<f32> = vec![1.5; m * m];
        let wb: Vec<f32> = vec![1.25; m * m];
        let xat = GpuTensor::from_vec(&tape, &xa, vec![m, m]).unwrap();
        let wat = GpuTensor::from_vec(&tape, &wa, vec![m, m]).unwrap();
        let xbt = GpuTensor::from_vec(&tape, &xb, vec![m, m]).unwrap();
        let wbt = GpuTensor::from_vec(&tape, &wb, vec![m, m]).unwrap();
        let _ya = matmul(&xat, &wat).unwrap();
        let yb = matmul(&xbt, &wbt).unwrap();
        let dy = ones_like(&tape, &[m, m]).unwrap();
        let grads = backward(&yb, dy).unwrap();

        // Subgraph B nodes (xbt, wbt, yb) should have grads.
        assert!(grads[xbt.node_idx()].is_some(), "xb grad expected");
        assert!(grads[wbt.node_idx()].is_some(), "wb grad expected");
        // Subgraph A nodes (xat, wat) should NOT have grads.
        assert!(grads[xat.node_idx()].is_none(), "xa grad must be None");
        assert!(grads[wat.node_idx()].is_none(), "wa grad must be None");
    }
}
