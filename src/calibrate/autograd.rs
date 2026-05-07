//! **CPU correctness oracle** for the GPU autograd that ships in
//! production (ADR-020 Track 1, iter 8a).
//!
//! ## Role — NOT a production codepath
//!
//! This module exists ONLY as a finite-difference / analytical oracle
//! that the GPU-backed autograd (iter 8b+) gets falsifier-tested
//! against.  Production runs through mlx-native MlxBuffer + Metal
//! kernels.  The CPU forward + backward closures here are reachable
//! only from `#[cfg(test)]` cross-checks — never from a CLI, API,
//! convert pipeline, or any other runtime entry point.
//!
//! Per `~/Documents/mantra.txt`: "No fallback. No stub. Pure excellence."
//! CPU is *not* a fallback for GPU — it is a separate, smaller, slower
//! reference whose ONLY job is to verify mathematical correctness on
//! synthetic fixtures.  A user touching this code at runtime would be
//! a defect, not a feature.
//!
//! ## What ships in production (iter 8b+)
//!
//! mlx-native has zero autograd primitives today (verified iter-8
//! audit: `grep -rn "fn.*backward\|value_and_grad\|VJP\|JVP\|tape\|gradient\|autograd" /opt/mlx-native/src` empty).
//! mlx-lm `dynamic_quant.estimate_sensitivities`
//! (`/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py:38-106`) needs gradients
//! of a KL-divergence loss w.r.t. every quantizable Linear/Embedding's
//! weight.  Building that via mlx-native MlxBuffer + Metal kernels is
//! the work of iter 8b onward.
//!
//! ## Design (CPU oracle)
//!
//! Tape-based reverse mode (Wengert list).  Forward appends a `Node`
//! per op recording (parents, backward closure).  `backward()` walks
//! tape in reverse, accumulating per-node gradients in a parallel
//! `Vec<Option<Vec<f32>>>`.  Storage is owned `Vec<f32>` per node.
//! All `f32`, all CPU.  Performance is intentionally not a goal —
//! tests fix < 100 elements per op.
//!
//! ## Roadmap (iters 8b → 8h, all GPU)
//!
//! | Iter | Scope |
//! |------|-------|
//! | 8b | GPU tape on mlx-native: storage = `MlxBuffer`, ops = Metal kernels for {matmul, add, mul, sub, square, sum, mean}. Per-op finite-diff parity vs CPU oracle in this module. |
//! | 8c | GPU activations: {softmax, log_softmax, gelu, silu, relu} |
//! | 8d | GPU norms: {RMSNorm, LayerNorm} |
//! | 8e | GPU loss: {kl_div_loss, cross_entropy} |
//! | 8f | GPU Linear composite + 2-Linear MLP byte-parity vs oracle |
//! | 8g | GPU embedding + RoPE + attention block |
//! | 8h | QDQ-as-identity-for-codes (gradient flows through dequantize) |
//!
//! ## What's NOT here
//!
//! - Production use of any kind.  `pub` items are exported only for
//!   the GPU autograd module's `#[cfg(test)]` parity assertions.
//! - Higher-order autograd.  `value_and_grad` only.  No grad-of-grad.
//! - Broadcasting beyond what's needed for matmul + bias-add.
//! - Strided/view tensors — `Vec<f32>` is contiguous.
//! - Mutation: tensors are functional/immutable.

use std::cell::RefCell;
use std::rc::Rc;

use thiserror::Error;

/// Errors from autograd ops.
#[derive(Error, Debug, PartialEq)]
pub enum AutogradError {
    #[error("shape mismatch: lhs={lhs:?} rhs={rhs:?} for op {op}")]
    ShapeMismatch {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
        op: &'static str,
    },

    #[error("matmul: incompatible shapes lhs={lhs:?} rhs={rhs:?} (need lhs.last() == rhs.first())")]
    MatmulIncompatible { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("matmul: only 2-D × 2-D and (B, M, K) × (K, N) supported in iter 8a; got lhs.ndim={lhs_ndim}, rhs.ndim={rhs_ndim}")]
    MatmulRankUnsupported { lhs_ndim: usize, rhs_ndim: usize },

    #[error(
        "tensor on different tape: lhs.tape ≠ rhs.tape; tensors must share a tape for autograd"
    )]
    TapeMismatch,

    #[error("backward called on a tensor without grad source (constant, no parents)")]
    NoGradSource,

    #[error(
        "backward: requested grad for tensor that's not on this tape (was it created on another tape?)"
    )]
    GradTensorOffTape,

    #[error("empty tensor: numel={numel}, shape={shape:?}")]
    EmptyTensor { numel: usize, shape: Vec<usize> },
}

/// A node in the Wengert tape.  Each node knows its parent indices
/// + how to push gradient contributions back to those parents
/// given an `output_grad` flowing into THIS node.
struct Node {
    /// Indices of parent nodes in the tape (each `Tensor` linked here
    /// got its values from these).
    parents: Vec<usize>,
    /// Closure that pushes gradient contributions from this node's
    /// output to each parent's accumulator.  Called once per node
    /// during backward.  `output_grad` is the gradient w.r.t. THIS
    /// node's output (same shape as `forward_value`).
    /// `parent_grads[i]` is the per-parent accumulator slice
    /// (already-shaped Vec<f32>).
    backward: Box<dyn Fn(&[f32], &mut [Vec<f32>])>,
    /// Forward-computed value of this node (cached for backward).
    /// Owned per node; inputs are not aliased.
    forward_value: Vec<f32>,
    /// Shape of `forward_value` (last dim contiguous).
    shape: Vec<usize>,
}

/// The tape.  Holds Nodes + parallel grad accumulators.  Created via
/// `Tape::new()`; tensors live on a tape via `tape.constant(...)` or
/// derived ops.
pub struct TapeInner {
    nodes: Vec<Node>,
}

#[derive(Clone)]
pub struct Tape(Rc<RefCell<TapeInner>>);

impl Tape {
    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(TapeInner { nodes: Vec::new() })))
    }

    /// Register a leaf node (no parents — typically a model parameter
    /// or input).  Returns the node index.
    fn push_leaf(&self, value: Vec<f32>, shape: Vec<usize>) -> usize {
        let mut inner = self.0.borrow_mut();
        let idx = inner.nodes.len();
        inner.nodes.push(Node {
            parents: Vec::new(),
            backward: Box::new(|_grad, _parents| {}),
            forward_value: value,
            shape,
        });
        idx
    }

    fn push_node(
        &self,
        parents: Vec<usize>,
        backward: Box<dyn Fn(&[f32], &mut [Vec<f32>])>,
        value: Vec<f32>,
        shape: Vec<usize>,
    ) -> usize {
        let mut inner = self.0.borrow_mut();
        let idx = inner.nodes.len();
        inner.nodes.push(Node {
            parents,
            backward,
            forward_value: value,
            shape,
        });
        idx
    }

    fn ptr_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

/// User-facing tensor handle.  Carries shape + a tape link.  Cheap
/// to clone (it's an `Rc<TapeInner>` clone + a node index).
#[derive(Clone)]
pub struct Tensor {
    tape: Tape,
    node_idx: usize,
}

impl Tensor {
    /// Create a leaf tensor on the given tape.
    pub fn from_vec(tape: &Tape, value: Vec<f32>, shape: Vec<usize>) -> Result<Self, AutogradError> {
        let numel: usize = shape.iter().product();
        if numel == 0 || value.len() != numel {
            return Err(AutogradError::EmptyTensor { numel: value.len(), shape });
        }
        let node_idx = tape.push_leaf(value, shape);
        Ok(Self {
            tape: tape.clone(),
            node_idx,
        })
    }

    pub fn shape(&self) -> Vec<usize> {
        self.tape.0.borrow().nodes[self.node_idx].shape.clone()
    }

    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Read forward-computed values out (for inspection / asserts).
    pub fn to_vec(&self) -> Vec<f32> {
        self.tape.0.borrow().nodes[self.node_idx].forward_value.clone()
    }

    fn assert_same_tape(&self, other: &Self) -> Result<(), AutogradError> {
        if self.tape.ptr_eq(&other.tape) {
            Ok(())
        } else {
            Err(AutogradError::TapeMismatch)
        }
    }

    fn read_value<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[f32], &[usize]) -> R,
    {
        let inner = self.tape.0.borrow();
        let n = &inner.nodes[self.node_idx];
        f(&n.forward_value, &n.shape)
    }
}

/// Run backward from `loss` (a 0-D or 1-elem tensor) and return a
/// `Vec<Option<Vec<f32>>>` indexed by node — Some(grad) for nodes
/// that participated in `loss`'s subgraph, None otherwise.
///
/// To extract the gradient for a particular leaf (e.g. a parameter),
/// use `grads_at(loss, &[&param])` which returns `Vec<Vec<f32>>` one-
/// per-input-tensor in the requested order.
pub fn backward(loss: &Tensor) -> Result<Vec<Option<Vec<f32>>>, AutogradError> {
    let inner = loss.tape.0.borrow();
    if loss.numel() != 1 {
        return Err(AutogradError::ShapeMismatch {
            lhs: loss.shape(),
            rhs: vec![1],
            op: "backward",
        });
    }
    let n_nodes = inner.nodes.len();
    let mut grads: Vec<Option<Vec<f32>>> = vec![None; n_nodes];
    grads[loss.node_idx] = Some(vec![1.0]);

    // Walk in reverse-tape order; for each node with an accumulated
    // grad, run its backward closure to push contributions to parents.
    for node_idx in (0..n_nodes).rev() {
        let Some(my_grad) = grads[node_idx].clone() else {
            continue;
        };
        let node = &inner.nodes[node_idx];
        if node.parents.is_empty() {
            continue;
        }
        // Materialize parent-grad slots (zero-initialized to parent shapes).
        let mut parent_grads: Vec<Vec<f32>> = node
            .parents
            .iter()
            .map(|&p| {
                let p_node = &inner.nodes[p];
                vec![0.0_f32; p_node.forward_value.len()]
            })
            .collect();

        (node.backward)(&my_grad, &mut parent_grads);

        for (parent_idx, contribution) in node.parents.iter().zip(parent_grads.into_iter()) {
            match grads.get_mut(*parent_idx).expect("parent index in range") {
                slot @ None => *slot = Some(contribution),
                Some(existing) => {
                    debug_assert_eq!(existing.len(), contribution.len(), "grad shape mismatch");
                    for (a, b) in existing.iter_mut().zip(contribution.iter()) {
                        *a += *b;
                    }
                }
            }
        }
    }
    Ok(grads)
}

/// Convenience: `value_and_grad` for a closure of one parameter.
/// Returns `(loss_value_f32, grad_vec)`.
pub fn value_and_grad<F>(tape: &Tape, param: &Tensor, f: F) -> Result<(f32, Vec<f32>), AutogradError>
where
    F: FnOnce(&Tensor) -> Result<Tensor, AutogradError>,
{
    let loss = f(param)?;
    let loss_value = loss.to_vec()[0];
    let grads = backward(&loss)?;
    let g = grads
        .get(param.node_idx)
        .and_then(|o| o.clone())
        .ok_or(AutogradError::GradTensorOffTape)?;
    debug_assert!(tape.ptr_eq(&param.tape));
    Ok((loss_value, g))
}

// ============================================================================
// Op: add — elementwise, same shape
// ============================================================================

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, AutogradError> {
    lhs.assert_same_tape(rhs)?;
    let l_shape = lhs.shape();
    let r_shape = rhs.shape();
    if l_shape != r_shape {
        return Err(AutogradError::ShapeMismatch {
            lhs: l_shape,
            rhs: r_shape,
            op: "add",
        });
    }
    let l = lhs.to_vec();
    let r = rhs.to_vec();
    let out: Vec<f32> = l.iter().zip(r.iter()).map(|(a, b)| a + b).collect();
    let parents = vec![lhs.node_idx, rhs.node_idx];
    let backward = Box::new(|out_grad: &[f32], parent_grads: &mut [Vec<f32>]| {
        // ∂(a+b)/∂a = 1; ∂(a+b)/∂b = 1.
        parent_grads[0].copy_from_slice(out_grad);
        parent_grads[1].copy_from_slice(out_grad);
    });
    let node_idx = lhs.tape.push_node(parents, backward, out, l_shape);
    Ok(Tensor {
        tape: lhs.tape.clone(),
        node_idx,
    })
}

// ============================================================================
// Op: sub — elementwise, same shape
// ============================================================================

pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, AutogradError> {
    lhs.assert_same_tape(rhs)?;
    let l_shape = lhs.shape();
    let r_shape = rhs.shape();
    if l_shape != r_shape {
        return Err(AutogradError::ShapeMismatch {
            lhs: l_shape,
            rhs: r_shape,
            op: "sub",
        });
    }
    let l = lhs.to_vec();
    let r = rhs.to_vec();
    let out: Vec<f32> = l.iter().zip(r.iter()).map(|(a, b)| a - b).collect();
    let parents = vec![lhs.node_idx, rhs.node_idx];
    let backward = Box::new(|out_grad: &[f32], parent_grads: &mut [Vec<f32>]| {
        // ∂(a−b)/∂a = 1; ∂(a−b)/∂b = -1.
        parent_grads[0].copy_from_slice(out_grad);
        for (slot, g) in parent_grads[1].iter_mut().zip(out_grad.iter()) {
            *slot = -*g;
        }
    });
    let node_idx = lhs.tape.push_node(parents, backward, out, l_shape);
    Ok(Tensor {
        tape: lhs.tape.clone(),
        node_idx,
    })
}

// ============================================================================
// Op: mul — elementwise, same shape
// ============================================================================

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, AutogradError> {
    lhs.assert_same_tape(rhs)?;
    let l_shape = lhs.shape();
    let r_shape = rhs.shape();
    if l_shape != r_shape {
        return Err(AutogradError::ShapeMismatch {
            lhs: l_shape,
            rhs: r_shape,
            op: "mul",
        });
    }
    let l = lhs.to_vec();
    let r = rhs.to_vec();
    let out: Vec<f32> = l.iter().zip(r.iter()).map(|(a, b)| a * b).collect();
    // Capture l + r snapshots for backward (we need the FORWARD values
    // to compute ∂(a·b)/∂a = b and ∂(a·b)/∂b = a).
    let l_snapshot = l.clone();
    let r_snapshot = r.clone();
    let parents = vec![lhs.node_idx, rhs.node_idx];
    let backward = Box::new(move |out_grad: &[f32], parent_grads: &mut [Vec<f32>]| {
        for i in 0..out_grad.len() {
            parent_grads[0][i] = out_grad[i] * r_snapshot[i];
            parent_grads[1][i] = out_grad[i] * l_snapshot[i];
        }
    });
    let node_idx = lhs.tape.push_node(parents, backward, out, l_shape);
    Ok(Tensor {
        tape: lhs.tape.clone(),
        node_idx,
    })
}

// ============================================================================
// Op: square (elementwise x²)
// ============================================================================

pub fn square(t: &Tensor) -> Result<Tensor, AutogradError> {
    let v = t.to_vec();
    let out: Vec<f32> = v.iter().map(|x| x * x).collect();
    let v_snapshot = v.clone();
    let parents = vec![t.node_idx];
    let backward = Box::new(move |out_grad: &[f32], parent_grads: &mut [Vec<f32>]| {
        // ∂(x²)/∂x = 2x.
        for i in 0..out_grad.len() {
            parent_grads[0][i] = 2.0 * v_snapshot[i] * out_grad[i];
        }
    });
    let shape = t.shape();
    let node_idx = t.tape.push_node(parents, backward, out, shape);
    Ok(Tensor {
        tape: t.tape.clone(),
        node_idx,
    })
}

// ============================================================================
// Op: sum (reduce all elements to a scalar)
// ============================================================================

pub fn sum(t: &Tensor) -> Result<Tensor, AutogradError> {
    let v = t.to_vec();
    if v.is_empty() {
        return Err(AutogradError::EmptyTensor {
            numel: 0,
            shape: t.shape(),
        });
    }
    let s: f32 = v.iter().sum();
    let n = v.len();
    let parents = vec![t.node_idx];
    let backward = Box::new(move |out_grad: &[f32], parent_grads: &mut [Vec<f32>]| {
        // ∂(Σxᵢ)/∂xⱼ = 1 for all j; broadcast scalar grad to all elems.
        let g = out_grad[0];
        for slot in parent_grads[0][..n].iter_mut() {
            *slot = g;
        }
    });
    let node_idx = t.tape.push_node(parents, backward, vec![s], vec![1]);
    Ok(Tensor {
        tape: t.tape.clone(),
        node_idx,
    })
}

// ============================================================================
// Op: mean (reduce all elements to a scalar = sum/numel)
// ============================================================================

pub fn mean(t: &Tensor) -> Result<Tensor, AutogradError> {
    let v = t.to_vec();
    if v.is_empty() {
        return Err(AutogradError::EmptyTensor {
            numel: 0,
            shape: t.shape(),
        });
    }
    let n = v.len();
    let s: f32 = v.iter().sum::<f32>() / (n as f32);
    let parents = vec![t.node_idx];
    let backward = Box::new(move |out_grad: &[f32], parent_grads: &mut [Vec<f32>]| {
        // ∂(Σxᵢ/n)/∂xⱼ = 1/n.
        let g = out_grad[0] / (n as f32);
        for slot in parent_grads[0][..n].iter_mut() {
            *slot = g;
        }
    });
    let node_idx = t.tape.push_node(parents, backward, vec![s], vec![1]);
    Ok(Tensor {
        tape: t.tape.clone(),
        node_idx,
    })
}

// ============================================================================
// Op: matmul — supports 2D × 2D and (B, M, K) × (K, N).
// Sufficient for Linear forward (input @ W^T) at iter-8a.
// ============================================================================

pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, AutogradError> {
    lhs.assert_same_tape(rhs)?;
    let l_shape = lhs.shape();
    let r_shape = rhs.shape();
    let l_ndim = l_shape.len();
    let r_ndim = r_shape.len();
    if !((l_ndim == 2 && r_ndim == 2) || (l_ndim == 3 && r_ndim == 2)) {
        return Err(AutogradError::MatmulRankUnsupported {
            lhs_ndim: l_ndim,
            rhs_ndim: r_ndim,
        });
    }
    let k_l = l_shape[l_ndim - 1];
    let k_r = r_shape[0];
    if k_l != k_r {
        return Err(AutogradError::MatmulIncompatible {
            lhs: l_shape.clone(),
            rhs: r_shape.clone(),
        });
    }
    let n = r_shape[1];
    let l_flat = lhs.to_vec();
    let r_flat = rhs.to_vec();
    let l_snapshot = l_flat.clone();
    let r_snapshot = r_flat.clone();

    let (m, batch) = if l_ndim == 2 {
        (l_shape[0], 1usize)
    } else {
        (l_shape[1], l_shape[0])
    };
    let k = k_l;

    let mut out = vec![0.0_f32; batch * m * n];
    for b in 0..batch {
        let l_off = b * m * k;
        let o_off = b * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    acc += l_flat[l_off + i * k + kk] * r_flat[kk * n + j];
                }
                out[o_off + i * n + j] = acc;
            }
        }
    }
    let out_shape = if l_ndim == 2 {
        vec![m, n]
    } else {
        vec![batch, m, n]
    };

    let parents = vec![lhs.node_idx, rhs.node_idx];
    let backward = Box::new(move |out_grad: &[f32], parent_grads: &mut [Vec<f32>]| {
        // ∂(LR)/∂L = out_grad @ R^T;  ∂(LR)/∂R = L^T @ out_grad.
        // Per-batch for 3D L, then summed across batch into the same R.
        for b in 0..batch {
            let l_off = b * m * k;
            let o_off = b * m * n;
            // dL[b,i,kk] = Σⱼ out_grad[b,i,j] · R[kk,j]
            for i in 0..m {
                for kk in 0..k {
                    let mut acc = 0.0_f32;
                    for j in 0..n {
                        acc += out_grad[o_off + i * n + j] * r_snapshot[kk * n + j];
                    }
                    parent_grads[0][l_off + i * k + kk] = acc;
                }
            }
            // dR[kk,j] += Σᵢ L[b,i,kk] · out_grad[b,i,j]
            for kk in 0..k {
                for j in 0..n {
                    let mut acc = 0.0_f32;
                    for i in 0..m {
                        acc += l_snapshot[l_off + i * k + kk] * out_grad[o_off + i * n + j];
                    }
                    parent_grads[1][kk * n + j] += acc;
                }
            }
        }
    });
    let node_idx = lhs.tape.push_node(parents, backward, out, out_shape);
    Ok(Tensor {
        tape: lhs.tape.clone(),
        node_idx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finite-difference helper.  Computes numerical gradient of `f` at
    /// `param` element-by-element via central differences.  `f` must be
    /// scalar-valued.  Tolerance picked to balance fp32 truncation vs
    /// finite-diff error: `eps = 1e-3` gives ~1e-4 trunc + ~1e-6 fd
    /// error for well-behaved ops.
    fn finite_diff_grad<F>(param: &[f32], f: F, eps: f32) -> Vec<f32>
    where
        F: Fn(&[f32]) -> f32,
    {
        let mut g = vec![0.0_f32; param.len()];
        let mut perturbed: Vec<f32> = param.to_vec();
        for i in 0..param.len() {
            let orig = perturbed[i];
            perturbed[i] = orig + eps;
            let f_plus = f(&perturbed);
            perturbed[i] = orig - eps;
            let f_minus = f(&perturbed);
            perturbed[i] = orig;
            g[i] = (f_plus - f_minus) / (2.0 * eps);
        }
        g
    }

    fn assert_close(actual: &[f32], expected: &[f32], rel_tol: f32, abs_tol: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: len mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let scale = a.abs().max(e.abs()).max(1.0);
            assert!(
                diff <= abs_tol || diff / scale <= rel_tol,
                "{label}[{i}]: actual={a} expected={e} diff={diff} (abs_tol={abs_tol}, rel_tol={rel_tol})"
            );
        }
    }

    // ------------------------------------------------------------------
    // add
    // ------------------------------------------------------------------

    #[test]
    fn add_forward_correctness() {
        let tape = Tape::new();
        let a = Tensor::from_vec(&tape, vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(&tape, vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let c = add(&a, &b).unwrap();
        assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_backward_finite_diff_falsifier() {
        // f(a) = sum(a + b); ∂f/∂a = ones.
        let tape = Tape::new();
        let a_v = vec![1.5_f32, 2.5, -0.7, 4.0, 5.0, -1.3];
        let b_v = vec![0.1_f32, -0.2, 0.3, 0.4, 0.5, 0.6];
        let a = Tensor::from_vec(&tape, a_v.clone(), vec![6]).unwrap();
        let b = Tensor::from_vec(&tape, b_v.clone(), vec![6]).unwrap();
        let c = add(&a, &b).unwrap();
        let loss = sum(&c).unwrap();
        let grads = backward(&loss).unwrap();
        let analytical = grads[a.node_idx].clone().expect("got grad for a");

        let fd = finite_diff_grad(
            &a_v,
            |perturbed| {
                let inner_tape = Tape::new();
                let aa = Tensor::from_vec(&inner_tape, perturbed.to_vec(), vec![6]).unwrap();
                let bb = Tensor::from_vec(&inner_tape, b_v.clone(), vec![6]).unwrap();
                let cc = add(&aa, &bb).unwrap();
                sum(&cc).unwrap().to_vec()[0]
            },
            1e-3,
        );
        assert_close(&analytical, &fd, 1e-3, 1e-4, "add ∂f/∂a");
    }

    // ------------------------------------------------------------------
    // sub
    // ------------------------------------------------------------------

    #[test]
    fn sub_backward_finite_diff_falsifier() {
        let tape = Tape::new();
        let a_v = vec![1.5_f32, 2.5, -0.7, 4.0];
        let b_v = vec![0.1_f32, -0.2, 0.3, 0.4];
        let a = Tensor::from_vec(&tape, a_v.clone(), vec![4]).unwrap();
        let b = Tensor::from_vec(&tape, b_v.clone(), vec![4]).unwrap();
        let c = sub(&a, &b).unwrap();
        let loss = sum(&c).unwrap();
        let grads = backward(&loss).unwrap();
        let g_a = grads[a.node_idx].clone().unwrap();
        let g_b = grads[b.node_idx].clone().unwrap();

        let fd_a = finite_diff_grad(
            &a_v,
            |p| {
                let t = Tape::new();
                let aa = Tensor::from_vec(&t, p.to_vec(), vec![4]).unwrap();
                let bb = Tensor::from_vec(&t, b_v.clone(), vec![4]).unwrap();
                sum(&sub(&aa, &bb).unwrap()).unwrap().to_vec()[0]
            },
            1e-3,
        );
        let fd_b = finite_diff_grad(
            &b_v,
            |p| {
                let t = Tape::new();
                let aa = Tensor::from_vec(&t, a_v.clone(), vec![4]).unwrap();
                let bb = Tensor::from_vec(&t, p.to_vec(), vec![4]).unwrap();
                sum(&sub(&aa, &bb).unwrap()).unwrap().to_vec()[0]
            },
            1e-3,
        );
        assert_close(&g_a, &fd_a, 1e-3, 1e-4, "sub ∂f/∂a");
        assert_close(&g_b, &fd_b, 1e-3, 1e-4, "sub ∂f/∂b");
    }

    // ------------------------------------------------------------------
    // mul
    // ------------------------------------------------------------------

    #[test]
    fn mul_backward_finite_diff_falsifier() {
        let tape = Tape::new();
        let a_v = vec![1.5_f32, 2.5, -0.7, 4.0];
        let b_v = vec![0.1_f32, -0.2, 0.3, 0.4];
        let a = Tensor::from_vec(&tape, a_v.clone(), vec![4]).unwrap();
        let b = Tensor::from_vec(&tape, b_v.clone(), vec![4]).unwrap();
        let c = mul(&a, &b).unwrap();
        let loss = sum(&c).unwrap();
        let grads = backward(&loss).unwrap();
        let g_a = grads[a.node_idx].clone().unwrap();
        let g_b = grads[b.node_idx].clone().unwrap();
        // Analytical: ∂(Σaᵢbᵢ)/∂aⱼ = bⱼ
        assert_close(&g_a, &b_v, 1e-6, 1e-7, "mul analytical ∂f/∂a == b");
        assert_close(&g_b, &a_v, 1e-6, 1e-7, "mul analytical ∂f/∂b == a");

        // Finite-diff cross-check.
        let fd = finite_diff_grad(
            &a_v,
            |p| {
                let t = Tape::new();
                let aa = Tensor::from_vec(&t, p.to_vec(), vec![4]).unwrap();
                let bb = Tensor::from_vec(&t, b_v.clone(), vec![4]).unwrap();
                sum(&mul(&aa, &bb).unwrap()).unwrap().to_vec()[0]
            },
            1e-3,
        );
        assert_close(&g_a, &fd, 1e-3, 1e-4, "mul fd ∂f/∂a");
    }

    // ------------------------------------------------------------------
    // square
    // ------------------------------------------------------------------

    #[test]
    fn square_backward_finite_diff_falsifier() {
        let tape = Tape::new();
        let a_v = vec![1.5_f32, -2.5, 0.7, 4.0, -3.3];
        let a = Tensor::from_vec(&tape, a_v.clone(), vec![5]).unwrap();
        let b = square(&a).unwrap();
        let loss = sum(&b).unwrap();
        let grads = backward(&loss).unwrap();
        let g_a = grads[a.node_idx].clone().unwrap();
        // Analytical: ∂(Σxᵢ²)/∂xⱼ = 2xⱼ
        let analytical: Vec<f32> = a_v.iter().map(|x| 2.0 * x).collect();
        assert_close(&g_a, &analytical, 1e-6, 1e-7, "square analytical");

        let fd = finite_diff_grad(
            &a_v,
            |p| {
                let t = Tape::new();
                let aa = Tensor::from_vec(&t, p.to_vec(), vec![5]).unwrap();
                sum(&square(&aa).unwrap()).unwrap().to_vec()[0]
            },
            1e-3,
        );
        assert_close(&g_a, &fd, 1e-3, 1e-4, "square fd");
    }

    // ------------------------------------------------------------------
    // sum
    // ------------------------------------------------------------------

    #[test]
    fn sum_backward_uniform() {
        let tape = Tape::new();
        let a = Tensor::from_vec(&tape, vec![1.0_f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let s = sum(&a).unwrap();
        assert_eq!(s.to_vec(), vec![10.0]);
        let grads = backward(&s).unwrap();
        let g = grads[a.node_idx].clone().unwrap();
        assert_eq!(g, vec![1.0; 4]);
    }

    // ------------------------------------------------------------------
    // mean
    // ------------------------------------------------------------------

    #[test]
    fn mean_backward_uniform_one_over_n() {
        let tape = Tape::new();
        let a = Tensor::from_vec(&tape, vec![1.0_f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let m = mean(&a).unwrap();
        assert!((m.to_vec()[0] - 2.5).abs() < 1e-6);
        let grads = backward(&m).unwrap();
        let g = grads[a.node_idx].clone().unwrap();
        assert_eq!(g, vec![0.25; 4]);
    }

    // ------------------------------------------------------------------
    // matmul
    // ------------------------------------------------------------------

    #[test]
    fn matmul_2d_2d_forward() {
        let tape = Tape::new();
        // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        let l = Tensor::from_vec(&tape, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let r = Tensor::from_vec(&tape, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let out = matmul(&l, &r).unwrap();
        assert_eq!(out.to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_backward_2d_2d_finite_diff() {
        let tape = Tape::new();
        let l_v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r_v = vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let l = Tensor::from_vec(&tape, l_v.clone(), vec![2, 3]).unwrap();
        let r = Tensor::from_vec(&tape, r_v.clone(), vec![3, 2]).unwrap();
        let out = matmul(&l, &r).unwrap();
        let loss = sum(&out).unwrap();
        let grads = backward(&loss).unwrap();
        let g_l = grads[l.node_idx].clone().unwrap();
        let g_r = grads[r.node_idx].clone().unwrap();

        let fd_l = finite_diff_grad(
            &l_v,
            |p| {
                let t = Tape::new();
                let ll = Tensor::from_vec(&t, p.to_vec(), vec![2, 3]).unwrap();
                let rr = Tensor::from_vec(&t, r_v.clone(), vec![3, 2]).unwrap();
                sum(&matmul(&ll, &rr).unwrap()).unwrap().to_vec()[0]
            },
            1e-2,
        );
        let fd_r = finite_diff_grad(
            &r_v,
            |p| {
                let t = Tape::new();
                let ll = Tensor::from_vec(&t, l_v.clone(), vec![2, 3]).unwrap();
                let rr = Tensor::from_vec(&t, p.to_vec(), vec![3, 2]).unwrap();
                sum(&matmul(&ll, &rr).unwrap()).unwrap().to_vec()[0]
            },
            1e-2,
        );
        // matmul has bigger fp32 truncation; relax tolerance accordingly.
        assert_close(&g_l, &fd_l, 1e-3, 1e-2, "matmul ∂f/∂L");
        assert_close(&g_r, &fd_r, 1e-3, 1e-2, "matmul ∂f/∂R");
    }

    #[test]
    fn matmul_backward_3d_2d_finite_diff() {
        // (B, M, K) × (K, N) — the shape Linear forward uses.
        let tape = Tape::new();
        let l_v: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.3).collect(); // [2, 2, 3]
        let r_v: Vec<f32> = (0..6).map(|i| (i as f32) * 0.2 + 0.1).collect(); // [3, 2]
        let l = Tensor::from_vec(&tape, l_v.clone(), vec![2, 2, 3]).unwrap();
        let r = Tensor::from_vec(&tape, r_v.clone(), vec![3, 2]).unwrap();
        let out = matmul(&l, &r).unwrap();
        assert_eq!(out.shape(), vec![2, 2, 2]);
        let loss = sum(&out).unwrap();
        let grads = backward(&loss).unwrap();
        let g_r = grads[r.node_idx].clone().unwrap();

        let fd_r = finite_diff_grad(
            &r_v,
            |p| {
                let t = Tape::new();
                let ll = Tensor::from_vec(&t, l_v.clone(), vec![2, 2, 3]).unwrap();
                let rr = Tensor::from_vec(&t, p.to_vec(), vec![3, 2]).unwrap();
                sum(&matmul(&ll, &rr).unwrap()).unwrap().to_vec()[0]
            },
            1e-2,
        );
        assert_close(&g_r, &fd_r, 1e-3, 1e-2, "matmul 3D×2D ∂f/∂R (batch-summed)");
    }

    // ------------------------------------------------------------------
    // composed: f(W) = sum(square(input @ W)) — a Linear-like
    // gradient that exercises matmul + square + sum end-to-end.
    // This is the structural target for the final
    // estimate_sensitivities formula.
    // ------------------------------------------------------------------

    #[test]
    fn composed_linear_square_sum_finite_diff() {
        let tape = Tape::new();
        let x_v: Vec<f32> = (0..6).map(|i| (i as f32) * 0.1 + 0.5).collect(); // [2, 3]
        let w_v: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05 - 0.1).collect(); // [3, 4]
        let x = Tensor::from_vec(&tape, x_v.clone(), vec![2, 3]).unwrap();
        let w = Tensor::from_vec(&tape, w_v.clone(), vec![3, 4]).unwrap();
        let y = matmul(&x, &w).unwrap();
        let y2 = square(&y).unwrap();
        let loss = sum(&y2).unwrap();
        let grads = backward(&loss).unwrap();
        let g_w = grads[w.node_idx].clone().unwrap();

        let fd_w = finite_diff_grad(
            &w_v,
            |p| {
                let t = Tape::new();
                let xx = Tensor::from_vec(&t, x_v.clone(), vec![2, 3]).unwrap();
                let ww = Tensor::from_vec(&t, p.to_vec(), vec![3, 4]).unwrap();
                let yy = matmul(&xx, &ww).unwrap();
                sum(&square(&yy).unwrap()).unwrap().to_vec()[0]
            },
            1e-2,
        );
        assert_close(&g_w, &fd_w, 1e-3, 1e-2, "composed Linear+square+sum ∂L/∂W");
    }

    // ------------------------------------------------------------------
    // value_and_grad helper
    // ------------------------------------------------------------------

    #[test]
    fn value_and_grad_helper_matches_manual_path() {
        let tape = Tape::new();
        let w_v = vec![1.5_f32, -2.0, 0.7];
        let w = Tensor::from_vec(&tape, w_v.clone(), vec![3]).unwrap();
        let (loss_val, grad) =
            value_and_grad(&tape, &w, |p| sum(&square(p)?)).unwrap();
        assert!((loss_val - (1.5_f32 * 1.5 + 4.0 + 0.49)).abs() < 1e-6);
        let analytical: Vec<f32> = w_v.iter().map(|x| 2.0 * x).collect();
        assert_close(&grad, &analytical, 1e-6, 1e-7, "value_and_grad");
    }

    // ------------------------------------------------------------------
    // error paths
    // ------------------------------------------------------------------

    #[test]
    fn add_shape_mismatch_errors() {
        let tape = Tape::new();
        let a = Tensor::from_vec(&tape, vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(&tape, vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(matches!(
            add(&a, &b),
            Err(AutogradError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn matmul_incompatible_shapes_errors() {
        let tape = Tape::new();
        let l = Tensor::from_vec(&tape, vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let r = Tensor::from_vec(&tape, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        assert!(matches!(
            matmul(&l, &r),
            Err(AutogradError::MatmulIncompatible { .. })
        ));
    }

    #[test]
    fn cross_tape_op_errors() {
        let t1 = Tape::new();
        let t2 = Tape::new();
        let a = Tensor::from_vec(&t1, vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(&t2, vec![1.0, 2.0], vec![2]).unwrap();
        assert!(matches!(add(&a, &b), Err(AutogradError::TapeMismatch)));
    }

    #[test]
    fn empty_tensor_construction_errors() {
        let tape = Tape::new();
        let r = Tensor::from_vec(&tape, vec![], vec![0]);
        assert!(matches!(r, Err(AutogradError::EmptyTensor { .. })));
    }

    #[test]
    fn backward_on_non_scalar_loss_errors() {
        let tape = Tape::new();
        let a = Tensor::from_vec(&tape, vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        // `square` produces a [3] tensor — backward must reject it.
        let b = square(&a).unwrap();
        assert!(matches!(
            backward(&b),
            Err(AutogradError::ShapeMismatch { .. })
        ));
    }
}
