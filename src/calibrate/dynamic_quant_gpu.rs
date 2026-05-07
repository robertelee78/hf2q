//! Native GPU port of mlx-lm's `dynamic_quant.estimate_sensitivities`
//! (ADR-020 Track 1, iter 9).
//!
//! Source: `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py:38-106`.
//!
//! ## Algorithm (per researcher-2 §2)
//!
//! For each quantizable Linear layer, compute a signed first-order
//! Taylor estimate of the KL-divergence loss change incurred by
//! quantizing at `low_bits` instead of `high_bits`:
//!
//! ```text
//! sensitivity[W] = (∇_W KL · (W_low − W_high)).sum() / (numel(W) / 1e6)
//! ```
//!
//! where:
//! - `∇_W KL` is the gradient of the KL-divergence loss
//!   `KL(softmax(teacher_logits), softmax(student_logits))` with
//!   respect to the QDQ'd weight `W_low` in the student model;
//! - `W_low = qdq(W_orig, low_bits, low_group_size)` is the
//!   low-bit reconstruction (already replacing the original weight
//!   in the student forward pass);
//! - `W_high = qdq(W_orig, high_bits, high_group_size)` is the
//!   high-bit reconstruction (computed on demand for the formula);
//! - `numel(W) / 1e6` is the per-million normalization that makes
//!   scores comparable across small (output proj) and huge (MoE
//!   expert) tensors.
//!
//! ## What this module ships at iter 9
//!
//! - `SyntheticTwoLinearModel` — a minimal 2-Linear MLP fixture
//!   sufficient to validate the gradient flow through KL-div on
//!   GpuTape.  Uses ONLY the existing autograd primitives
//!   (matmul + softmax + log + sub + mul + row_sum).  No new
//!   mlx-native kernels.
//! - `kl_div_loss_per_row(logits_q, logits_p)` — the loss function:
//!   per-row KL-divergence built from existing primitives.  Returns
//!   shape `[batch]`; backward(loss, ones[batch]) gives gradient as
//!   if `loss = sum_per_row(KL)`.
//! - `estimate_sensitivities` — runs forward on the student model,
//!   computes KL vs teacher logits, runs backward, applies the
//!   sensitivity formula per Linear weight.  Returns a
//!   `BTreeMap<String, f64>` keyed by tensor path matching mlx-lm's
//!   convention (path with the `.weight` suffix already stripped).
//!
//! ## What's NOT here yet
//!
//! - Real qdq (quantize→dequantize round-trip) primitive.  Iter 9
//!   accepts pre-computed `(W_orig, W_low, W_high)` tuples from the
//!   caller; the qdq computation lands in iter 10 alongside hf2q's
//!   existing k-quant primitives or via a new mlx-native affine qdq
//!   kernel.
//! - Multi-batch streaming with `del grads + mx.eval(grad_accum)`
//!   memory rhythm — iter 9 ships single-batch; iter 10 adds
//!   per-batch accumulation.
//! - Real transformer architecture (attention, RMSNorm, RoPE,
//!   embeddings, MoE).  Iter 9's 2-Linear MLP validates the gradient
//!   flow + sensitivity formula correctness; iter 11+ wires into a
//!   real Qwen35-style block.

use std::collections::BTreeMap;

use anyhow::{anyhow, Result};

use crate::calibrate::autograd_gpu_tape::{
    backward, log, matmul, mul, ones_like, row_sum, softmax, sub, GpuTape, GpuTensor,
};

/// A synthetic 2-Linear MLP for autograd validation.  Forward:
///   `logits = X @ W1 @ W2`
/// where X is `[batch, h0]`, W1 is `[h0, h1]`, W2 is `[h1, vocab]`.
///
/// All dimensions must be `>= 32` to satisfy the matmul kernel's
/// `K, M, N >= 32` backward-pass constraint.  Production transformer
/// dimensions trivially exceed this floor.
pub struct SyntheticTwoLinearModel {
    pub w1: GpuTensor,
    pub w2: GpuTensor,
}

impl SyntheticTwoLinearModel {
    /// Construct from (CPU) weight values.  W1 shape `[h0, h1]`,
    /// W2 shape `[h1, vocab]`.  Tape ownership is shared via Rc.
    pub fn from_vecs(
        tape: &GpuTape,
        w1: &[f32],
        h0: usize,
        h1: usize,
        w2: &[f32],
        vocab: usize,
    ) -> Result<Self> {
        let w1 = GpuTensor::from_vec(tape, w1, vec![h0, h1])
            .map_err(|e| anyhow!("model: from_vec W1: {e}"))?;
        let w2 = GpuTensor::from_vec(tape, w2, vec![h1, vocab])
            .map_err(|e| anyhow!("model: from_vec W2: {e}"))?;
        Ok(Self { w1, w2 })
    }

    /// Forward: `logits = X @ W1 @ W2`.
    pub fn forward(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let h = matmul(x, &self.w1).map_err(|e| anyhow!("forward: X @ W1: {e}"))?;
        matmul(&h, &self.w2).map_err(|e| anyhow!("forward: H @ W2: {e}"))
    }
}

/// Per-row KL-divergence between `softmax(logits_q)` (student) and
/// `softmax(logits_p)` (teacher).  Returns shape `[batch]`.
///
/// `KL(P || Q) = Σ_i P_i · log(P_i / Q_i)`
///            = `Σ_i softmax(logits_p)_i · (log_softmax(logits_p)_i − log_softmax(logits_q)_i)`
///
/// Implementation: composes existing GpuTape primitives:
///
/// ```text
///     p     = softmax(logits_p)
///     log_p = log(softmax(logits_p))      // numerically log_softmax
///     log_q = log(softmax(logits_q))
///     diff  = log_p − log_q
///     w     = p · diff                    // shape [batch, vocab]
///     kl    = row_sum(w)                  // shape [batch]
/// ```
///
/// Backward via the autograd graph automatically produces the
/// analytical identity `∂loss/∂logits_q = softmax(logits_q) − softmax(logits_p)`
/// — verified by `gpu_tape_kl_div_via_composition_dq_equals_softmax_q_minus_p`
/// in `autograd_gpu_tape::tests`.
pub fn kl_div_loss_per_row(
    logits_q: &GpuTensor,
    logits_p: &GpuTensor,
) -> Result<GpuTensor> {
    let p = softmax(logits_p).map_err(|e| anyhow!("kl: softmax(p): {e}"))?;
    let log_p = log(&p).map_err(|e| anyhow!("kl: log(p): {e}"))?;
    let q = softmax(logits_q).map_err(|e| anyhow!("kl: softmax(q): {e}"))?;
    let log_q = log(&q).map_err(|e| anyhow!("kl: log(q): {e}"))?;
    let diff = sub(&log_p, &log_q).map_err(|e| anyhow!("kl: log_p − log_q: {e}"))?;
    let weighted = mul(&p, &diff).map_err(|e| anyhow!("kl: p · diff: {e}"))?;
    row_sum(&weighted).map_err(|e| anyhow!("kl: row_sum: {e}"))
}

/// Per-quantizable-tensor input to [`estimate_sensitivities`].  Each
/// entry carries the original (FP) weight + the low-bit and high-bit
/// reconstructions.  At iter 9 the qdq computation is the caller's
/// responsibility (e.g. via hf2q's k-quant primitives); iter 10
/// wires in a unified GPU qdq op.
pub struct QuantizableInput {
    /// Tensor path key (matches `dynamic_quant.py:104`'s convention,
    /// with the trailing `.weight` already stripped).
    pub path: String,
    /// Reference to the LOW-bit reconstruction tensor on the GpuTape.
    /// This is the weight already in the student model
    /// (`q_model` at `dynamic_quant.py:55-63`).
    pub w_low: GpuTensor,
    /// CPU-side bytes of `qdq(W_orig, high_bits, high_group_size)`,
    /// used to compute the per-element delta `(W_low − W_high)`.
    /// Shape must match `w_low.shape()`.
    pub w_high_values: Vec<f32>,
}

/// Run `estimate_sensitivities` on a forward graph that has already
/// been built on `student_logits` (computed via the student model
/// using `w_low` weights).  Compares against `teacher_logits`
/// (detached, treated as a constant target).
///
/// **Inputs:**
/// - `tape` — the GpuTape carrying `student_logits` and the input
///   tensors in `quantizables`.
/// - `student_logits` — `[batch, vocab]` GpuTensor produced by the
///   student model on the calibration batch.
/// - `teacher_logits` — `[batch, vocab]` GpuTensor produced by the
///   teacher model (constructed as a leaf — no gradient flows here).
/// - `quantizables` — list of `(path, w_low, w_high_values)` tuples,
///   one per Linear layer that should get a sensitivity score.
///
/// **Returns:**
/// `BTreeMap<String, f64>` keyed by `path`, with the sensitivity
/// score per the formula
/// `(∇_W KL · (W_low − W_high)).sum() / (numel(W) / 1e6)`.
pub fn estimate_sensitivities(
    tape: &GpuTape,
    student_logits: &GpuTensor,
    teacher_logits: &GpuTensor,
    quantizables: &[QuantizableInput],
) -> Result<BTreeMap<String, f64>> {
    let kl = kl_div_loss_per_row(student_logits, teacher_logits)
        .map_err(|e| anyhow!("estimate_sensitivities: KL forward: {e}"))?;
    let kl_shape = kl.shape().to_vec();
    let dy = ones_like(tape, &kl_shape)
        .map_err(|e| anyhow!("estimate_sensitivities: ones seed: {e}"))?;
    let grads = backward(&kl, dy)
        .map_err(|e| anyhow!("estimate_sensitivities: backward: {e}"))?;

    let mut out = BTreeMap::new();
    for q in quantizables {
        let grad_buf = grads
            .get(q.w_low.node_idx())
            .and_then(|g| g.as_ref())
            .ok_or_else(|| {
                anyhow!(
                    "estimate_sensitivities: no gradient for {} (node not in subgraph)",
                    q.path
                )
            })?;
        let grad_slice: &[f32] = grad_buf
            .as_slice()
            .map_err(|e| anyhow!("grad buf slice: {e}"))?;
        let w_low_buf: Vec<f32> = q
            .w_low
            .to_vec()
            .map_err(|e| anyhow!("w_low to_vec: {e}"))?;
        if grad_slice.len() != w_low_buf.len() || grad_slice.len() != q.w_high_values.len() {
            return Err(anyhow!(
                "estimate_sensitivities: shape mismatch for {} \
                 (grad len {}, w_low len {}, w_high len {})",
                q.path,
                grad_slice.len(),
                w_low_buf.len(),
                q.w_high_values.len()
            ));
        }
        // Apply the sensitivity formula:
        //   alignment = Σ_i grad[i] · (w_low[i] − w_high[i])
        //   sensitivity = alignment / (numel / 1e6)
        let mut alignment = 0.0_f64;
        for ((g, lo), hi) in grad_slice
            .iter()
            .zip(w_low_buf.iter())
            .zip(q.w_high_values.iter())
        {
            alignment += (*g as f64) * ((*lo as f64) - (*hi as f64));
        }
        let numel = grad_slice.len() as f64;
        let sensitivity = alignment / (numel / 1.0e6);
        out.insert(q.path.clone(), sensitivity);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::MlxDevice;

    fn assert_close_f64(actual: f64, expected: f64, rel_tol: f64, abs_tol: f64, label: &str) {
        let diff = (actual - expected).abs();
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            diff <= abs_tol || diff / scale <= rel_tol,
            "{label}: actual={actual} expected={expected} diff={diff} \
             (abs_tol={abs_tol}, rel_tol={rel_tol})"
        );
    }

    /// **The closing-the-loop iter-9 test.**
    ///
    /// Builds a synthetic 2-Linear MLP on GpuTape, computes KL-div
    /// loss vs synthetic teacher logits, runs backward, applies the
    /// sensitivity formula per Linear weight, then verifies the
    /// computed sensitivity scalar matches the analytically-derived
    /// reference on a hand-checked fixture.
    ///
    /// Reference computation:
    /// 1. Construct deterministic `X`, `W1`, `W2`, `W1_high`, `W2_high`.
    /// 2. Compute teacher logits via a separate forward pass with
    ///    `W1_high`/`W2_high` (treated as the teacher).
    /// 3. Compute student logits via forward with `W1`/`W2`.
    /// 4. Compute `dq = softmax(student) − softmax(teacher)` (the
    ///    analytical KL backward identity, verified at iter 8f).
    /// 5. Backprop `dq` through `W2`'s matmul to get `∇_W2 KL`:
    ///    `∇_W2 = H1^T @ dq` where `H1 = X @ W1`.
    /// 6. Backprop further through `W1`'s matmul:
    ///    `∇_W1 = X^T @ (dq @ W2^T)`.
    /// 7. Apply formula `(∇_W · (W_low − W_high)).sum() / (numel / 1e6)`.
    ///
    /// The test verifies GPU's `estimate_sensitivities` produces
    /// the same scalar as this hand-derived reference on a 32×32
    /// synthetic fixture (well within all matmul kernel constraints).
    #[test]
    fn iter9_estimate_sensitivities_two_linear_synthetic() {
        // Fixture dims (all >= 32 to satisfy matmul backward kernel).
        let batch = 32;
        let h0 = 32;
        let h1 = 32;
        let vocab = 32;

        let det = |i: usize, off: f32, scale: f32| (i as f32) * scale + off;
        let x: Vec<f32> = (0..(batch * h0)).map(|i| det(i, 0.0, 0.0011)).collect();
        let w1: Vec<f32> = (0..(h0 * h1)).map(|i| det(i, -0.05, 0.0007)).collect();
        let w2: Vec<f32> = (0..(h1 * vocab)).map(|i| det(i, 0.04, 0.0009)).collect();

        // "high-bit" reconstruction = w1 + small per-element noise.
        // For real quants this would be qdq(w_orig, high_bits) - qdq(w_orig, low_bits).
        // Synthetic deltas keep the test deterministic + hand-checkable.
        let w1_high: Vec<f32> = w1
            .iter()
            .enumerate()
            .map(|(i, v)| v + ((i % 7) as f32 - 3.0) * 1e-4)
            .collect();
        let w2_high: Vec<f32> = w2
            .iter()
            .enumerate()
            .map(|(i, v)| v + ((i % 5) as f32 - 2.0) * 1e-4)
            .collect();

        // Construct teacher + student forward graphs on the SAME tape.
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &x, vec![batch, h0]).unwrap();

        // Teacher: a separate set of weights; logits are detached
        // (no gradient flows back to teacher during student backward,
        // because the KL backward only uses softmax(p) as a constant).
        let teacher = SyntheticTwoLinearModel::from_vecs(
            &tape, &w1_high, h0, h1, &w2_high, vocab,
        )
        .unwrap();
        let teacher_logits = teacher.forward(&xt).unwrap();

        // Student: low-bit weights (we use `w1` and `w2` here as our
        // "w_low", since this is a synthetic).
        let student = SyntheticTwoLinearModel::from_vecs(
            &tape, &w1, h0, h1, &w2, vocab,
        )
        .unwrap();
        let student_logits = student.forward(&xt).unwrap();

        // Run estimate_sensitivities.
        let quantizables = vec![
            QuantizableInput {
                path: "model.layers.0.w1".to_string(),
                w_low: student.w1.clone(),
                w_high_values: w1_high.clone(),
            },
            QuantizableInput {
                path: "model.layers.0.w2".to_string(),
                w_low: student.w2.clone(),
                w_high_values: w2_high.clone(),
            },
        ];
        let sensitivities = estimate_sensitivities(
            &tape,
            &student_logits,
            &teacher_logits,
            &quantizables,
        )
        .expect("estimate_sensitivities");

        // Reference computation, hand-derived analytically:
        //   dq[b, v] = softmax(student_logits)[b, v] − softmax(teacher_logits)[b, v]
        //   ∇_W2 = H1^T @ dq        where H1 = X @ W1   (shape [h1, vocab])
        //   ∇_W1 = X^T @ (dq @ W2^T)                   (shape [h0, h1])
        // Then sensitivity[W] = Σ_i ∇_W[i] · (w_low[i] − w_high[i]) / (numel / 1e6).
        let student_logits_cpu = student_logits.to_vec().unwrap();
        let teacher_logits_cpu = teacher_logits.to_vec().unwrap();

        // Compute softmax(logits) on CPU (numerically stable).
        let row_softmax = |logits: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0_f32; batch * vocab];
            for b in 0..batch {
                let off = b * vocab;
                let mut max = f32::NEG_INFINITY;
                for i in 0..vocab {
                    let v = logits[off + i];
                    if v > max {
                        max = v;
                    }
                }
                let mut sum = 0.0_f32;
                for i in 0..vocab {
                    let e = (logits[off + i] - max).exp();
                    out[off + i] = e;
                    sum += e;
                }
                let inv = 1.0_f32 / sum;
                for i in 0..vocab {
                    out[off + i] *= inv;
                }
            }
            out
        };
        let p_t = row_softmax(&teacher_logits_cpu);
        let p_s = row_softmax(&student_logits_cpu);
        // dq = softmax(student) − softmax(teacher)
        let dq: Vec<f32> = p_s.iter().zip(p_t.iter()).map(|(s, t)| s - t).collect();

        // H1 = X @ W1  — shape [batch, h1].
        let mut h1_vals = vec![0.0_f32; batch * h1];
        for b in 0..batch {
            for j in 0..h1 {
                let mut acc = 0.0_f32;
                for k in 0..h0 {
                    acc += x[b * h0 + k] * w1[k * h1 + j];
                }
                h1_vals[b * h1 + j] = acc;
            }
        }

        // ∇_W2[h1, vocab] = H1^T @ dq:  ∇_W2[j, v] = Σ_b H1[b, j] · dq[b, v]
        let mut grad_w2 = vec![0.0_f32; h1 * vocab];
        for j in 0..h1 {
            for v in 0..vocab {
                let mut acc = 0.0_f32;
                for b in 0..batch {
                    acc += h1_vals[b * h1 + j] * dq[b * vocab + v];
                }
                grad_w2[j * vocab + v] = acc;
            }
        }
        // ∇_W1[h0, h1] = X^T @ (dq @ W2^T):
        // first compute dq_W2T[b, j] = Σ_v dq[b, v] · W2[j, v]    (since W2^T[v, j] = W2[j, v])
        let mut dq_w2t = vec![0.0_f32; batch * h1];
        for b in 0..batch {
            for j in 0..h1 {
                let mut acc = 0.0_f32;
                for v in 0..vocab {
                    acc += dq[b * vocab + v] * w2[j * vocab + v];
                }
                dq_w2t[b * h1 + j] = acc;
            }
        }
        let mut grad_w1 = vec![0.0_f32; h0 * h1];
        for k in 0..h0 {
            for j in 0..h1 {
                let mut acc = 0.0_f32;
                for b in 0..batch {
                    acc += x[b * h0 + k] * dq_w2t[b * h1 + j];
                }
                grad_w1[k * h1 + j] = acc;
            }
        }

        let formula = |grad: &[f32], lo: &[f32], hi: &[f32]| -> f64 {
            let mut alignment = 0.0_f64;
            for ((g, l), h) in grad.iter().zip(lo.iter()).zip(hi.iter()) {
                alignment += (*g as f64) * ((*l as f64) - (*h as f64));
            }
            alignment / (grad.len() as f64 / 1.0e6)
        };
        let s_w1_expected = formula(&grad_w1, &w1, &w1_high);
        let s_w2_expected = formula(&grad_w2, &w2, &w2_high);

        let s_w1_gpu = sensitivities["model.layers.0.w1"];
        let s_w2_gpu = sensitivities["model.layers.0.w2"];

        // Both gradient paths involve compositions through softmax + log
        // + sub + mul + row_sum + matmul; cumulative fp32 drift bounds
        // sit around 1e-3 rel tol on this 32×32 fixture.
        assert_close_f64(s_w1_gpu, s_w1_expected, 5e-3, 1e-4, "sensitivity W1");
        assert_close_f64(s_w2_gpu, s_w2_expected, 5e-3, 1e-4, "sensitivity W2");
    }

    #[test]
    fn iter9_kl_div_loss_per_row_shape() {
        let batch = 32;
        let vocab = 32;
        let q: Vec<f32> = (0..(batch * vocab)).map(|i| (i as f32) * 0.011).collect();
        let p: Vec<f32> = (0..(batch * vocab)).map(|i| (i as f32) * 0.013).collect();

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let qt = GpuTensor::from_vec(&tape, &q, vec![batch, vocab]).unwrap();
        let pt = GpuTensor::from_vec(&tape, &p, vec![batch, vocab]).unwrap();
        let kl = kl_div_loss_per_row(&qt, &pt).expect("kl forward");
        assert_eq!(kl.shape(), &[batch]);
        // Each per-row KL must be non-negative (Gibbs' inequality).
        let kl_vals = kl.to_vec().unwrap();
        for (i, v) in kl_vals.iter().enumerate() {
            assert!(*v >= -1e-5, "KL[{i}] should be ≥ 0; got {v}");
        }
    }

    #[test]
    fn iter9_estimate_sensitivities_returns_one_score_per_quantizable() {
        let batch = 32;
        let h0 = 32;
        let h1 = 32;
        let vocab = 32;
        let x: Vec<f32> = (0..(batch * h0)).map(|i| (i as f32) * 0.001).collect();
        let w1: Vec<f32> = (0..(h0 * h1)).map(|i| (i as f32) * 0.0007).collect();
        let w2: Vec<f32> = (0..(h1 * vocab)).map(|i| (i as f32) * 0.0009).collect();
        let w1_high: Vec<f32> = w1.iter().map(|v| v + 1e-4).collect();
        let w2_high: Vec<f32> = w2.iter().map(|v| v + 1e-4).collect();
        let teacher_logits: Vec<f32> =
            (0..(batch * vocab)).map(|i| (i as f32) * 0.005).collect();

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &x, vec![batch, h0]).unwrap();
        let student =
            SyntheticTwoLinearModel::from_vecs(&tape, &w1, h0, h1, &w2, vocab).unwrap();
        let teacher_logits_t =
            GpuTensor::from_vec(&tape, &teacher_logits, vec![batch, vocab]).unwrap();
        let student_logits = student.forward(&xt).unwrap();

        let quantizables = vec![
            QuantizableInput {
                path: "w1".to_string(),
                w_low: student.w1.clone(),
                w_high_values: w1_high,
            },
            QuantizableInput {
                path: "w2".to_string(),
                w_low: student.w2.clone(),
                w_high_values: w2_high,
            },
        ];
        let s = estimate_sensitivities(&tape, &student_logits, &teacher_logits_t, &quantizables)
            .expect("ok");
        assert_eq!(s.len(), 2);
        assert!(s.contains_key("w1"));
        assert!(s.contains_key("w2"));
        assert!(s["w1"].is_finite());
        assert!(s["w2"].is_finite());
    }
}
