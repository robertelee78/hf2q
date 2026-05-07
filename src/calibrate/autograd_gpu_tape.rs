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
    metal,
    ops::dense_mm_f32_f32::{dense_matmul_f32_f32_tensor, DenseMmF32F32Params},
    ops::elementwise::{elementwise_add, elementwise_mul, scalar_mul_f32},
    ops::log_elementwise::{dispatch_log_backward_f32, dispatch_log_f32},
    ops::softmax::dispatch_softmax,
    ops::softmax_backward::dispatch_softmax_backward,
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
    /// Elementwise `Y = A + B`; A, B, Y all same shape.  Backward:
    /// `dA = dY`, `dB = dY` (both contributions are the upstream
    /// gradient itself; the accumulator handles the case `lhs_idx ==
    /// rhs_idx` by summing — i.e. `A + A = 2A → dA = 2·dY`).
    ElementwiseAdd { lhs_idx: usize, rhs_idx: usize },
    /// Elementwise `Y = A − B`; A, B, Y all same shape.  Backward:
    /// `dA = dY`, `dB = −dY` (computed via `scalar_mul_f32(-1, dy)`).
    ElementwiseSub { lhs_idx: usize, rhs_idx: usize },
    /// Elementwise `Y = A · B`; A, B, Y all same shape.  Backward:
    /// `dA = dY · B`, `dB = dY · A` (each via `elementwise_mul`).
    /// `lhs_idx == rhs_idx` (square via mul-self) is supported — the
    /// accumulator sums both contributions, yielding `2·A·dY`.
    ElementwiseMul { lhs_idx: usize, rhs_idx: usize },
    /// Row-wise softmax along the last dim of a 2-D tensor `[rows, cols]`.
    /// Backward via mlx-native's `dispatch_softmax_backward` kernel.
    Softmax {
        input_idx: usize,
        rows: usize,
        cols: usize,
    },
    /// Elementwise natural log.  Backward `dx = dy / x` via mlx-native's
    /// `dispatch_log_backward_f32`.  Caller must ensure forward input
    /// is strictly positive.
    Log { input_idx: usize },
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
            OpKind::Matmul { lhs_idx, rhs_idx, .. }
            | OpKind::ElementwiseAdd { lhs_idx, rhs_idx }
            | OpKind::ElementwiseSub { lhs_idx, rhs_idx }
            | OpKind::ElementwiseMul { lhs_idx, rhs_idx } => {
                accumulate(&mut grads, lhs_idx, &parent_grads[0], tape)?;
                accumulate(&mut grads, rhs_idx, &parent_grads[1], tape)?;
            }
            OpKind::Softmax { input_idx, .. } | OpKind::Log { input_idx } => {
                accumulate(&mut grads, input_idx, &parent_grads[0], tape)?;
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
        OpKind::ElementwiseAdd { .. } => {
            // dA = dY, dB = dY.  Cheapest: hand the same MlxBuffer to
            // both parents (Arc-share).  The accumulator copies on
            // first-touch and adds on subsequent touches; sharing a
            // single buffer between two parents that DON'T alias each
            // other is safe because each parent gets its own slot.
            // For lhs_idx == rhs_idx (a + a), the accumulator sums the
            // two contributions producing dA = 2·dY ✓.
            Ok((op, vec![out_grad.clone(), out_grad.clone()]))
        }
        OpKind::ElementwiseSub { .. } => {
            // dA = dY, dB = -dY.  -dY computed via scalar_mul_f32(-1, dY).
            // For lhs_idx == rhs_idx (a - a = 0 in forward): accumulator
            // sums dY + (-dY) = 0 ✓ (analytically correct).
            let device = tape.device();
            let mut registry = tape.0.registry.borrow_mut();
            let n = out_grad.byte_len() / 4;
            let neg_dy = device
                .alloc_buffer(n * 4, DType::F32, out_grad.shape().to_vec())
                .map_err(|e| anyhow!("backward sub: alloc neg_dy: {e}"))?;
            let mut encoder = device
                .command_encoder()
                .map_err(|e| anyhow!("backward sub: encoder: {e}"))?;
            scalar_mul_f32(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                out_grad,
                &neg_dy,
                n,
                -1.0,
            )
            .map_err(|e| anyhow!("backward sub: scalar_mul_f32(-1): {e}"))?;
            encoder
                .commit_and_wait()
                .map_err(|e| anyhow!("backward sub: commit_and_wait: {e}"))?;
            Ok((op, vec![out_grad.clone(), neg_dy]))
        }
        OpKind::ElementwiseMul { lhs_idx, rhs_idx } => {
            // dA = dY · B, dB = dY · A.  Each via elementwise_mul.
            // lhs_idx == rhs_idx (square via mul-self): accumulator
            // sums dY·a + dY·a = 2·a·dY ✓.
            let device = tape.device();
            let mut registry = tape.0.registry.borrow_mut();
            let n = out_grad.byte_len() / 4;

            let (a_buf, b_buf) = {
                let nodes = tape.0.nodes.borrow();
                (nodes[lhs_idx].value.clone(), nodes[rhs_idx].value.clone())
            };

            let da = device
                .alloc_buffer(n * 4, DType::F32, out_grad.shape().to_vec())
                .map_err(|e| anyhow!("backward mul: alloc dA: {e}"))?;
            let db = device
                .alloc_buffer(n * 4, DType::F32, out_grad.shape().to_vec())
                .map_err(|e| anyhow!("backward mul: alloc dB: {e}"))?;

            let mut encoder = device
                .command_encoder()
                .map_err(|e| anyhow!("backward mul: encoder: {e}"))?;
            elementwise_mul(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                out_grad,
                &b_buf,
                &da,
                n,
                DType::F32,
            )
            .map_err(|e| anyhow!("backward mul: dA = dY·B: {e}"))?;
            elementwise_mul(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                out_grad,
                &a_buf,
                &db,
                n,
                DType::F32,
            )
            .map_err(|e| anyhow!("backward mul: dB = dY·A: {e}"))?;
            encoder
                .commit_and_wait()
                .map_err(|e| anyhow!("backward mul: commit_and_wait: {e}"))?;

            Ok((op, vec![da, db]))
        }
        OpKind::Log { input_idx } => {
            // dx = dy / x, where x is the FORWARD INPUT (not output).
            let device = tape.device();
            let mut registry = tape.0.registry.borrow_mut();
            let x_buf = tape.with_node(input_idx, |n| n.value.clone());
            let n = x_buf.byte_len() / 4;
            let dx_buf = device
                .alloc_buffer(n * 4, DType::F32, x_buf.shape().to_vec())
                .map_err(|e| anyhow!("backward log: alloc dx: {e}"))?;
            let mut encoder = device
                .command_encoder()
                .map_err(|e| anyhow!("backward log: encoder: {e}"))?;
            dispatch_log_backward_f32(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &x_buf,
                out_grad,
                &dx_buf,
            )
            .map_err(|e| anyhow!("backward log: dispatch: {e}"))?;
            encoder
                .commit_and_wait()
                .map_err(|e| anyhow!("backward log: commit_and_wait: {e}"))?;
            Ok((op, vec![dx_buf]))
        }
        OpKind::Softmax { rows, cols, .. } => {
            // dx = y * (dy - sum_j(y · dy))  via mlx-native's
            // dispatch_softmax_backward kernel.  We need the FORWARD
            // softmax output `y` saved in this node's value.
            let device = tape.device();
            let mut registry = tape.0.registry.borrow_mut();

            let y_buf = tape.with_node(node_idx, |n| n.value.clone());
            let dx_buf = device
                .alloc_buffer(rows * cols * 4, DType::F32, vec![rows, cols])
                .map_err(|e| anyhow!("backward softmax: alloc dx: {e}"))?;
            // Params buffer: [cols_f, 0] as f32.
            let mut params_buf = device
                .alloc_buffer(8, DType::F32, vec![2])
                .map_err(|e| anyhow!("backward softmax: alloc params: {e}"))?;
            params_buf
                .as_mut_slice::<f32>()
                .map_err(|e| anyhow!("backward softmax: params write: {e}"))?
                .copy_from_slice(&[cols as f32, 0.0]);

            let mut encoder = device
                .command_encoder()
                .map_err(|e| anyhow!("backward softmax: encoder: {e}"))?;
            dispatch_softmax_backward(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &y_buf,
                out_grad,
                &dx_buf,
                &params_buf,
                rows as u32,
                cols as u32,
            )
            .map_err(|e| anyhow!("backward softmax: dispatch: {e}"))?;
            encoder
                .commit_and_wait()
                .map_err(|e| anyhow!("backward softmax: commit_and_wait: {e}"))?;

            Ok((op, vec![dx_buf]))
        }
    }
}

/// Elementwise natural log via mlx-native's `dispatch_log_f32`.
/// Caller must ensure values are strictly positive.
pub fn log(t: &GpuTensor) -> Result<GpuTensor> {
    let tape = &t.tape;
    let device = tape.device();
    let n: usize = t.shape.iter().product();
    let in_buf = tape.with_node(t.node_idx, |n| n.value.clone());
    let out = device
        .alloc_buffer(n * 4, DType::F32, t.shape.clone())
        .map_err(|e| anyhow!("log: alloc out: {e}"))?;
    let mut registry = tape.0.registry.borrow_mut();
    let mut encoder = device
        .command_encoder()
        .map_err(|e| anyhow!("log: encoder: {e}"))?;
    dispatch_log_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &in_buf,
        &out,
    )
    .map_err(|e| anyhow!("log: dispatch: {e}"))?;
    encoder
        .commit_and_wait()
        .map_err(|e| anyhow!("log: commit_and_wait: {e}"))?;
    drop(registry);
    let input_idx = t.node_idx;
    let node_idx = tape.push_node(OpKind::Log { input_idx }, out, t.shape.clone());
    Ok(GpuTensor {
        tape: tape.clone(),
        node_idx,
        shape: t.shape.clone(),
    })
}

/// Row-wise softmax along the last dim of a 2-D tensor `[rows, cols]`.
/// Numerically stable form via mlx-native's `dispatch_softmax`.
pub fn softmax(t: &GpuTensor) -> Result<GpuTensor> {
    if t.shape.len() != 2 {
        return Err(anyhow!(
            "softmax: input must be 2-D [rows, cols]; got shape={:?}",
            t.shape
        ));
    }
    let rows = t.shape[0];
    let cols = t.shape[1];
    let tape = &t.tape;
    let device = tape.device();

    let in_buf = tape.with_node(t.node_idx, |n| n.value.clone());
    let out = device
        .alloc_buffer(rows * cols * 4, DType::F32, vec![rows, cols])
        .map_err(|e| anyhow!("softmax: alloc out: {e}"))?;
    let mut params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("softmax: alloc params: {e}"))?;
    params_buf
        .as_mut_slice::<f32>()
        .map_err(|e| anyhow!("softmax: params write: {e}"))?
        .copy_from_slice(&[cols as f32, 0.0]);

    let mut registry = tape.0.registry.borrow_mut();
    let mut encoder = device
        .command_encoder()
        .map_err(|e| anyhow!("softmax: encoder: {e}"))?;
    dispatch_softmax(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &in_buf,
        &out,
        &params_buf,
        rows as u32,
        cols as u32,
    )
    .map_err(|e| anyhow!("softmax: dispatch: {e}"))?;
    encoder
        .commit_and_wait()
        .map_err(|e| anyhow!("softmax: commit_and_wait: {e}"))?;
    drop(registry);

    let input_idx = t.node_idx;
    let node_idx = tape.push_node(
        OpKind::Softmax {
            input_idx,
            rows,
            cols,
        },
        out,
        vec![rows, cols],
    );
    Ok(GpuTensor {
        tape: tape.clone(),
        node_idx,
        shape: vec![rows, cols],
    })
}

/// Elementwise `Y = A + B` via mlx-native's `elementwise_add`.
/// Inputs must share the same shape + tape.
pub fn add(lhs: &GpuTensor, rhs: &GpuTensor) -> Result<GpuTensor> {
    elementwise_op(
        lhs,
        rhs,
        "add",
        |encoder, registry, device, a, b, out, n| {
            elementwise_add(encoder, registry, device, a, b, out, n, DType::F32)
                .map_err(|e| anyhow!("elementwise_add: {e}"))
        },
        |lhs_idx, rhs_idx| OpKind::ElementwiseAdd { lhs_idx, rhs_idx },
    )
}

/// Elementwise `Y = A - B` composed from `scalar_mul_f32(-1, b)` +
/// `elementwise_add`.  Inputs must share the same shape + tape.
pub fn sub(lhs: &GpuTensor, rhs: &GpuTensor) -> Result<GpuTensor> {
    if !lhs.tape.ptr_eq(&rhs.tape) {
        return Err(anyhow!("sub: cross-tape op (lhs.tape != rhs.tape)"));
    }
    if lhs.shape != rhs.shape {
        return Err(anyhow!(
            "sub: shape mismatch lhs={:?} rhs={:?}",
            lhs.shape,
            rhs.shape
        ));
    }
    let tape = &lhs.tape;
    let device = tape.device();
    let n: usize = lhs.shape.iter().product();
    let neg_b = device
        .alloc_buffer(n * 4, DType::F32, lhs.shape.clone())
        .map_err(|e| anyhow!("sub: alloc neg_b: {e}"))?;
    let out = device
        .alloc_buffer(n * 4, DType::F32, lhs.shape.clone())
        .map_err(|e| anyhow!("sub: alloc out: {e}"))?;

    let (a_buf, b_buf) = {
        let nodes = tape.0.nodes.borrow();
        (
            nodes[lhs.node_idx].value.clone(),
            nodes[rhs.node_idx].value.clone(),
        )
    };
    let mut registry = tape.0.registry.borrow_mut();
    // Encoder 1: produce neg_b = -1·b.  We commit_and_wait before the
    // add so the elementwise_add reliably sees the negated values.
    // Single-encoder back-to-back compute dispatches that read each
    // other's outputs require an explicit memory barrier in Metal;
    // two encoders sequenced via commit_and_wait give equivalent
    // ordering without us needing to reach into the encoder's
    // memory_barrier API.
    {
        let mut encoder = device
            .command_encoder()
            .map_err(|e| anyhow!("sub: encoder1: {e}"))?;
        scalar_mul_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &b_buf,
            &neg_b,
            n,
            -1.0,
        )
        .map_err(|e| anyhow!("sub: scalar_mul_f32(-1): {e}"))?;
        encoder
            .commit_and_wait()
            .map_err(|e| anyhow!("sub: commit_and_wait scalar_mul: {e}"))?;
    }
    // Encoder 2: out = a + neg_b.
    {
        let mut encoder = device
            .command_encoder()
            .map_err(|e| anyhow!("sub: encoder2: {e}"))?;
        elementwise_add(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &a_buf,
            &neg_b,
            &out,
            n,
            DType::F32,
        )
        .map_err(|e| anyhow!("sub: elementwise_add(a, -b): {e}"))?;
        encoder
            .commit_and_wait()
            .map_err(|e| anyhow!("sub: commit_and_wait add: {e}"))?;
    }
    drop(registry);

    let lhs_idx = lhs.node_idx;
    let rhs_idx = rhs.node_idx;
    let node_idx = tape.push_node(
        OpKind::ElementwiseSub { lhs_idx, rhs_idx },
        out,
        lhs.shape.clone(),
    );
    Ok(GpuTensor {
        tape: tape.clone(),
        node_idx,
        shape: lhs.shape.clone(),
    })
}

/// Elementwise `Y = A · B` via mlx-native's `elementwise_mul`.
/// Inputs must share the same shape + tape.  When both args are the
/// same tensor (`mul(&x, &x)`) the result is `x²`; backward correctly
/// produces `2·x·dY` via the accumulator.
pub fn mul(lhs: &GpuTensor, rhs: &GpuTensor) -> Result<GpuTensor> {
    elementwise_op(
        lhs,
        rhs,
        "mul",
        |encoder, registry, device, a, b, out, n| {
            elementwise_mul(encoder, registry, device, a, b, out, n, DType::F32)
                .map_err(|e| anyhow!("elementwise_mul: {e}"))
        },
        |lhs_idx, rhs_idx| OpKind::ElementwiseMul { lhs_idx, rhs_idx },
    )
}

/// Convenience: `square(x) = mul(&x, &x)`.  Backward produces
/// `2·x·dY` correctly via the accumulator (both parent slots are
/// the same node, contributions sum).
pub fn square(t: &GpuTensor) -> Result<GpuTensor> {
    mul(t, t)
}

/// Internal helper: dispatch a binary elementwise op + record its node.
fn elementwise_op<F, K>(
    lhs: &GpuTensor,
    rhs: &GpuTensor,
    label: &str,
    dispatch: F,
    op_kind: K,
) -> Result<GpuTensor>
where
    F: FnOnce(
        &mut mlx_native::CommandEncoder,
        &mut KernelRegistry,
        &metal::DeviceRef,
        &MlxBuffer,
        &MlxBuffer,
        &MlxBuffer,
        usize,
    ) -> Result<()>,
    K: FnOnce(usize, usize) -> OpKind,
{
    if !lhs.tape.ptr_eq(&rhs.tape) {
        return Err(anyhow!("{label}: cross-tape op (lhs.tape != rhs.tape)"));
    }
    if lhs.shape != rhs.shape {
        return Err(anyhow!(
            "{label}: shape mismatch lhs={:?} rhs={:?}",
            lhs.shape,
            rhs.shape
        ));
    }
    let tape = &lhs.tape;
    let device = tape.device();
    let n: usize = lhs.shape.iter().product();
    let out = device
        .alloc_buffer(n * 4, DType::F32, lhs.shape.clone())
        .map_err(|e| anyhow!("{label}: alloc out: {e}"))?;

    let (a_buf, b_buf) = {
        let nodes = tape.0.nodes.borrow();
        (
            nodes[lhs.node_idx].value.clone(),
            nodes[rhs.node_idx].value.clone(),
        )
    };
    let mut registry = tape.0.registry.borrow_mut();
    let mut encoder = device
        .command_encoder()
        .map_err(|e| anyhow!("{label}: encoder: {e}"))?;
    dispatch(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &a_buf,
        &b_buf,
        &out,
        n,
    )?;
    encoder
        .commit_and_wait()
        .map_err(|e| anyhow!("{label}: commit_and_wait: {e}"))?;
    drop(registry);

    let lhs_idx = lhs.node_idx;
    let rhs_idx = rhs.node_idx;
    let node_idx = tape.push_node(op_kind(lhs_idx, rhs_idx), out, lhs.shape.clone());
    Ok(GpuTensor {
        tape: tape.clone(),
        node_idx,
        shape: lhs.shape.clone(),
    })
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

    use crate::calibrate::autograd::{
        add as cpu_add, mul as cpu_mul, square as cpu_square, sub as cpu_sub,
    };

    /// `Y = A + B`; loss = sum(Y).  Backward: dA = ones, dB = ones.
    /// GPU vs CPU oracle parity.
    #[test]
    fn gpu_tape_add_forward_backward_parity() {
        let m = 8;
        let n = 32; // shape m*n must satisfy any kernel constraints; elementwise has none
        let a: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.0011 + 0.05).collect();
        let b: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * -0.0007 + 0.1).collect();

        // GPU
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let at = GpuTensor::from_vec(&tape, &a, vec![m, n]).unwrap();
        let bt = GpuTensor::from_vec(&tape, &b, vec![m, n]).unwrap();
        let yt = add(&at, &bt).expect("gpu add");
        let y_gpu = yt.to_vec().unwrap();
        let dy = ones_like(&tape, &[m, n]).unwrap();
        let grads = backward(&yt, dy).unwrap();
        let da_gpu: &[f32] = grads[at.node_idx()].as_ref().unwrap().as_slice().unwrap();
        let db_gpu: &[f32] = grads[bt.node_idx()].as_ref().unwrap().as_slice().unwrap();

        // CPU oracle
        let cpu_tape = Tape::new();
        let ca = Tensor::from_vec(&cpu_tape, a.clone(), vec![m, n]).unwrap();
        let cb = Tensor::from_vec(&cpu_tape, b.clone(), vec![m, n]).unwrap();
        let cy = cpu_add(&ca, &cb).unwrap();
        let y_cpu = cy.to_vec();
        let cl = cpu_sum(&cy).unwrap();
        let cgrads = cpu_backward(&cl).unwrap();
        let da_cpu = cgrads[ca.node_idx()].clone().unwrap();
        let db_cpu = cgrads[cb.node_idx()].clone().unwrap();

        assert_close(&y_gpu, &y_cpu, 1e-6, 1e-7, "add forward");
        assert_close(da_gpu, &da_cpu, 1e-6, 1e-7, "add dA");
        assert_close(db_gpu, &db_cpu, 1e-6, 1e-7, "add dB");
    }

    /// `Y = A − B`; loss = sum(Y).  Backward: dA = ones, dB = −ones.
    #[test]
    fn gpu_tape_sub_forward_backward_parity() {
        let m = 8;
        let n = 32;
        let a: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.0009 + 0.03).collect();
        let b: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.0013 - 0.04).collect();

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let at = GpuTensor::from_vec(&tape, &a, vec![m, n]).unwrap();
        let bt = GpuTensor::from_vec(&tape, &b, vec![m, n]).unwrap();
        let yt = sub(&at, &bt).expect("gpu sub");
        let y_gpu = yt.to_vec().unwrap();
        let dy = ones_like(&tape, &[m, n]).unwrap();
        let grads = backward(&yt, dy).unwrap();
        let da_gpu: &[f32] = grads[at.node_idx()].as_ref().unwrap().as_slice().unwrap();
        let db_gpu: &[f32] = grads[bt.node_idx()].as_ref().unwrap().as_slice().unwrap();

        let cpu_tape = Tape::new();
        let ca = Tensor::from_vec(&cpu_tape, a.clone(), vec![m, n]).unwrap();
        let cb = Tensor::from_vec(&cpu_tape, b.clone(), vec![m, n]).unwrap();
        let cy = cpu_sub(&ca, &cb).unwrap();
        let y_cpu = cy.to_vec();
        let cl = cpu_sum(&cy).unwrap();
        let cgrads = cpu_backward(&cl).unwrap();
        let da_cpu = cgrads[ca.node_idx()].clone().unwrap();
        let db_cpu = cgrads[cb.node_idx()].clone().unwrap();

        assert_close(&y_gpu, &y_cpu, 1e-6, 1e-7, "sub forward");
        assert_close(da_gpu, &da_cpu, 1e-6, 1e-7, "sub dA = ones");
        assert_close(db_gpu, &db_cpu, 1e-6, 1e-7, "sub dB = -ones");
    }

    /// `Y = A · B`; loss = sum(Y).  Backward: dA = B, dB = A.
    #[test]
    fn gpu_tape_mul_forward_backward_parity() {
        let m = 8;
        let n = 32;
        let a: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.001 + 0.5).collect();
        let b: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.002 - 0.3).collect();

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let at = GpuTensor::from_vec(&tape, &a, vec![m, n]).unwrap();
        let bt = GpuTensor::from_vec(&tape, &b, vec![m, n]).unwrap();
        let yt = mul(&at, &bt).expect("gpu mul");
        let y_gpu = yt.to_vec().unwrap();
        let dy = ones_like(&tape, &[m, n]).unwrap();
        let grads = backward(&yt, dy).unwrap();
        let da_gpu: &[f32] = grads[at.node_idx()].as_ref().unwrap().as_slice().unwrap();
        let db_gpu: &[f32] = grads[bt.node_idx()].as_ref().unwrap().as_slice().unwrap();

        // Analytical: dA = B, dB = A (when loss = sum(A·B)).
        assert_close(da_gpu, &b, 1e-6, 1e-7, "mul analytical dA == B");
        assert_close(db_gpu, &a, 1e-6, 1e-7, "mul analytical dB == A");

        // Cross-check vs CPU oracle.
        let cpu_tape = Tape::new();
        let ca = Tensor::from_vec(&cpu_tape, a.clone(), vec![m, n]).unwrap();
        let cb = Tensor::from_vec(&cpu_tape, b.clone(), vec![m, n]).unwrap();
        let cy = cpu_mul(&ca, &cb).unwrap();
        let y_cpu = cy.to_vec();
        assert_close(&y_gpu, &y_cpu, 1e-6, 1e-7, "mul forward");
    }

    /// `Y = X²` via `mul(&x, &x)`; loss = sum(Y).  Backward: dX = 2X.
    /// Validates that `lhs_idx == rhs_idx` produces the correct
    /// accumulation (2 contributions to the same parent sum to 2·X·dY).
    #[test]
    fn gpu_tape_square_via_mul_self_finite_diff() {
        let m = 4;
        let n = 32;
        let x: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.0017 - 0.2).collect();

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &x, vec![m, n]).unwrap();
        let yt = square(&xt).expect("gpu square");
        let y_gpu = yt.to_vec().unwrap();
        let dy = ones_like(&tape, &[m, n]).unwrap();
        let grads = backward(&yt, dy).unwrap();
        let dx_gpu: &[f32] = grads[xt.node_idx()].as_ref().unwrap().as_slice().unwrap();

        // Analytical: dX = 2X.
        let analytical: Vec<f32> = x.iter().map(|v| 2.0 * v).collect();
        assert_close(
            dx_gpu,
            &analytical,
            1e-6,
            1e-7,
            "square via mul-self analytical dX == 2X",
        );

        // CPU oracle parity.
        let cpu_tape = Tape::new();
        let cx = Tensor::from_vec(&cpu_tape, x.clone(), vec![m, n]).unwrap();
        let cy = cpu_square(&cx).unwrap();
        let y_cpu = cy.to_vec();
        let cl = cpu_sum(&cy).unwrap();
        let cgrads = cpu_backward(&cl).unwrap();
        let dx_cpu = cgrads[cx.node_idx()].clone().unwrap();
        assert_close(&y_gpu, &y_cpu, 1e-6, 1e-7, "square forward");
        assert_close(dx_gpu, &dx_cpu, 1e-6, 1e-7, "square backward");
    }

    /// Composed: `Y = (X @ W) · S`; loss = sum(Y).
    /// Exercises matmul + elementwise mul end-to-end with backward
    /// flowing through both ops.  All gradients (dX, dW, dS) match
    /// CPU oracle — validates op composition.
    #[test]
    fn gpu_tape_matmul_then_mul_chain_backward_parity() {
        let m = 32;
        let k = 32;
        let n = 32;
        let x: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.0009 + 0.01).collect();
        let w: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.0011 - 0.02).collect();
        let s: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.0013 + 0.03).collect();

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &x, vec![m, k]).unwrap();
        let wt = GpuTensor::from_vec(&tape, &w, vec![k, n]).unwrap();
        let st = GpuTensor::from_vec(&tape, &s, vec![m, n]).unwrap();
        let xw = matmul(&xt, &wt).expect("matmul");
        let yt = mul(&xw, &st).expect("mul");
        let dy = ones_like(&tape, &[m, n]).unwrap();
        let grads = backward(&yt, dy).unwrap();
        let dx_gpu: &[f32] = grads[xt.node_idx()].as_ref().unwrap().as_slice().unwrap();
        let dw_gpu: &[f32] = grads[wt.node_idx()].as_ref().unwrap().as_slice().unwrap();
        let ds_gpu: &[f32] = grads[st.node_idx()].as_ref().unwrap().as_slice().unwrap();

        // CPU oracle
        let cpu_tape = Tape::new();
        let cx = Tensor::from_vec(&cpu_tape, x.clone(), vec![m, k]).unwrap();
        let cw = Tensor::from_vec(&cpu_tape, w.clone(), vec![k, n]).unwrap();
        let cs = Tensor::from_vec(&cpu_tape, s.clone(), vec![m, n]).unwrap();
        let cxw = cpu_matmul(&cx, &cw).unwrap();
        let cy = cpu_mul(&cxw, &cs).unwrap();
        let cl = cpu_sum(&cy).unwrap();
        let cgrads = cpu_backward(&cl).unwrap();
        let dx_cpu = cgrads[cx.node_idx()].clone().unwrap();
        let dw_cpu = cgrads[cw.node_idx()].clone().unwrap();
        let ds_cpu = cgrads[cs.node_idx()].clone().unwrap();

        assert_close(dx_gpu, &dx_cpu, 1e-4, 1e-4, "matmul+mul chain dX");
        assert_close(dw_gpu, &dw_cpu, 1e-4, 1e-4, "matmul+mul chain dW");
        assert_close(ds_gpu, &ds_cpu, 1e-4, 1e-4, "matmul+mul chain dS");
    }

    use crate::calibrate::autograd::log as cpu_log;
    use crate::calibrate::autograd::softmax as cpu_softmax;

    /// `Y = log(X)` for strictly positive X; backward via dispatch_log_backward_f32.
    #[test]
    fn gpu_tape_log_forward_backward_parity() {
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.05).collect();

        // GPU
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &x, vec![n]).unwrap();
        let yt = log(&xt).expect("gpu log");
        let y_gpu = yt.to_vec().unwrap();
        let dy = ones_like(&tape, &[n]).unwrap();
        let grads = backward(&yt, dy).unwrap();
        let dx_gpu: &[f32] = grads[xt.node_idx()].as_ref().unwrap().as_slice().unwrap();

        // Analytical: forward y = log(x), backward dx = 1/x.
        let y_expected: Vec<f32> = x.iter().map(|v| v.ln()).collect();
        let dx_expected: Vec<f32> = x.iter().map(|v| 1.0 / v).collect();
        assert_close(&y_gpu, &y_expected, 1e-5, 1e-6, "log forward");
        assert_close(dx_gpu, &dx_expected, 1e-5, 1e-6, "log backward dx = 1/x");

        // Cross-check vs CPU oracle.
        let cpu_tape = Tape::new();
        let cx = Tensor::from_vec(&cpu_tape, x.clone(), vec![n]).unwrap();
        let cy = cpu_log(&cx).unwrap();
        let y_cpu = cy.to_vec();
        assert_close(&y_gpu, &y_cpu, 1e-5, 1e-6, "log forward GPU↔CPU");
    }

    /// Composed: log_softmax(x) = log(softmax(x)).  Validates that
    /// the autograd composition through softmax + log produces the
    /// correct gradient (matches the analytical log_softmax backward).
    #[test]
    fn gpu_tape_log_softmax_via_composition_backward_parity() {
        let rows = 4;
        let cols = 32;
        let x: Vec<f32> = (0..(rows * cols))
            .map(|i| (i as f32) * 0.013 - 0.5)
            .collect();
        // Use a non-trivial weighted loss = sum(C · log_softmax(X)).
        let c: Vec<f32> = (0..(rows * cols))
            .map(|i| (i as f32) * 0.011 + 0.2)
            .collect();

        // GPU: y = log(softmax(x)); cy = c · y; backward(cy, ones) gives
        //      dC = y, dX = log_softmax_backward(C).
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &x, vec![rows, cols]).unwrap();
        let ct = GpuTensor::from_vec(&tape, &c, vec![rows, cols]).unwrap();
        let sm = softmax(&xt).expect("softmax");
        let lsm = log(&sm).expect("log");
        let weighted = mul(&ct, &lsm).expect("mul");
        let dy = ones_like(&tape, &[rows, cols]).unwrap();
        let grads = backward(&weighted, dy).expect("backward");
        let dx_gpu: &[f32] = grads[xt.node_idx()]
            .as_ref()
            .unwrap()
            .as_slice()
            .unwrap();

        // CPU oracle via the same composition.
        let cpu_tape = Tape::new();
        let cx = Tensor::from_vec(&cpu_tape, x.clone(), vec![rows, cols]).unwrap();
        let cc = Tensor::from_vec(&cpu_tape, c.clone(), vec![rows, cols]).unwrap();
        let csm = cpu_softmax(&cx).unwrap();
        let clsm = cpu_log(&csm).unwrap();
        let cw = crate::calibrate::autograd::mul(&cc, &clsm).unwrap();
        let cl = cpu_sum(&cw).unwrap();
        let cgrads = cpu_backward(&cl).unwrap();
        let dx_cpu = cgrads[cx.node_idx()].clone().unwrap();

        // log(softmax) at extreme negative values can produce small
        // numerical differences across CPU/GPU paths; use 1e-3 rel tol.
        assert_close(dx_gpu, &dx_cpu, 1e-3, 1e-4, "log_softmax via composition");
    }

    /// `Y = softmax(X)`; loss = sum(C · Y) for non-trivial C
    /// (a constant `loss = sum(softmax(X))` would have ∂L/∂X ≡ 0).
    #[test]
    fn gpu_tape_softmax_forward_parity_with_cpu_oracle() {
        let rows = 4;
        let cols = 32;
        let x: Vec<f32> = (0..(rows * cols))
            .map(|i| (i as f32) * 0.013 - 0.5)
            .collect();

        // GPU
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &x, vec![rows, cols]).unwrap();
        let yt = softmax(&xt).expect("gpu softmax");
        let y_gpu = yt.to_vec().unwrap();

        // CPU oracle
        let cpu_tape = Tape::new();
        let cx = Tensor::from_vec(&cpu_tape, x.clone(), vec![rows, cols]).unwrap();
        let cy = cpu_softmax(&cx).unwrap();
        let y_cpu = cy.to_vec();

        assert_close(&y_gpu, &y_cpu, 1e-5, 1e-6, "softmax forward");

        // Sanity: rows of y_gpu sum to 1.
        for b in 0..rows {
            let s: f32 = y_gpu[b * cols..(b + 1) * cols].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {b} sum={s}");
        }
    }

    #[test]
    fn gpu_tape_softmax_backward_parity_with_cpu_oracle() {
        // Use a non-trivial weighted loss = sum(C · softmax(X)) so dL/dX ≠ 0.
        let rows = 4;
        let cols = 32;
        let x: Vec<f32> = (0..(rows * cols))
            .map(|i| (i as f32) * 0.011 - 0.7)
            .collect();
        let c: Vec<f32> = (0..(rows * cols))
            .map(|i| (i as f32) * 0.017 + 0.3)
            .collect();

        // GPU: y = softmax(x); cy = c * y; dY = c (since loss = sum(c·y) ⇒ ∂loss/∂y = c).
        // Equivalently we can invoke backward(yt, dy=c).
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &x, vec![rows, cols]).unwrap();
        let yt = softmax(&xt).expect("gpu softmax");
        // Construct dY = C as a leaf MlxBuffer.
        let mut dy_buf = tape
            .device()
            .alloc_buffer(rows * cols * 4, DType::F32, vec![rows, cols])
            .unwrap();
        dy_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&c);
        let grads = backward(&yt, dy_buf).expect("gpu backward");
        let dx_gpu: &[f32] = grads[xt.node_idx()]
            .as_ref()
            .unwrap()
            .as_slice()
            .unwrap();

        // CPU oracle: same path via tape with mul(c, softmax(x)) then sum.
        let cpu_tape = Tape::new();
        let cx = Tensor::from_vec(&cpu_tape, x.clone(), vec![rows, cols]).unwrap();
        let cc = Tensor::from_vec(&cpu_tape, c.clone(), vec![rows, cols]).unwrap();
        let cy = cpu_softmax(&cx).unwrap();
        let cl = cpu_sum(&crate::calibrate::autograd::mul(&cc, &cy).unwrap()).unwrap();
        let cgrads = cpu_backward(&cl).unwrap();
        let dx_cpu = cgrads[cx.node_idx()].clone().unwrap();

        assert_close(dx_gpu, &dx_cpu, 1e-4, 1e-5, "softmax backward");
    }

    #[test]
    fn gpu_tape_softmax_non_2d_errors() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let x = GpuTensor::from_vec(&tape, &[1.0; 32], vec![32]).unwrap();
        match softmax(&x) {
            Err(e) => assert!(format!("{e}").contains("must be 2-D")),
            Ok(_) => panic!("non-2D softmax must error"),
        }
    }

    #[test]
    fn gpu_tape_add_shape_mismatch_errors() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let a = GpuTensor::from_vec(&tape, &[0.0; 8], vec![8]).unwrap();
        let b = GpuTensor::from_vec(&tape, &[0.0; 16], vec![16]).unwrap();
        match add(&a, &b) {
            Err(e) => assert!(format!("{e}").contains("shape mismatch")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn gpu_tape_mul_shape_mismatch_errors() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let a = GpuTensor::from_vec(&tape, &[0.0; 8], vec![8]).unwrap();
        let b = GpuTensor::from_vec(&tape, &[0.0; 16], vec![16]).unwrap();
        match mul(&a, &b) {
            Err(e) => assert!(format!("{e}").contains("shape mismatch")),
            Ok(_) => panic!("expected error"),
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
