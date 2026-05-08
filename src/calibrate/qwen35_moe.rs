//! ADR-020 iter-11h-e2 — MoE routing forward composition on GpuTape.
//!
//! Mirrors `mlx-lm/qwen3_next.py:Qwen3NextSparseMoeBlock.__call__`'s
//! routing chain:
//!
//! ```python
//!   gates = softmax(self.gate(x))                    # [n_tokens, n_experts]
//!   inds  = argpartition(gates, kth=-k)[..., -k:]    # [n_tokens, k] u32
//!   scores = take_along_axis(gates, inds, axis=-1)   # [n_tokens, k]
//!   if norm_topk_prob:
//!       scores = scores / scores.sum(axis=-1, keepdims=True)
//!   # ... downstream switch_mlp(x, inds) consumes (inds, scores)
//! ```
//!
//! ## Composition on GpuTape
//!
//! Forward (with backward auto-derived through every OpKind):
//!
//!   1. `gate_logits = matmul(input, gate_weight)`  // matmul has a
//!      32-floor on m, n, k — for production Qwen3.5MoE shapes
//!      (n_tokens >> 32, hidden >> 32, n_experts >> 32) this is met
//!      trivially; test fixtures must satisfy it.
//!   2. `probs = softmax(gate_logits)`              // along last axis
//!   3. **OFF TAPE**: argpartition `probs` per-row to get top-K
//!      indices.  Done CPU-side via `to_vec()` + `select_nth_unstable_by`.
//!      Indices are non-differentiable → no tape node.
//!   4. `scores = take_along_axis_topk(probs, indices, k)`  → `[n_tokens, k]`
//!   5. Optional renorm (`norm_topk_prob == true`):
//!         `sum_k = row_sum(scores)`             → `[n_tokens]`
//!         `recip = exp(scalar_mul(-1, log(sum_k)))`  → `[n_tokens]`
//!            (reciprocal composed from exp + log; no new kernel needed)
//!         `recip_broadcast = outer_product(recip, ones_k)` → `[n_tokens, k]`
//!         `normalized = mul(scores, recip_broadcast)`
//!
//! ## Returns
//!
//!   * `top_k_scores` — differentiable tape node with shape `[n_tokens, k]`
//!   * `top_k_indices` — `MlxBuffer` of u32 indices `[n_tokens, k]`
//!     (non-differentiable; used downstream by `switch_mlp` to dispatch
//!     each token to its top-K experts)

use anyhow::{anyhow, Result};

use mlx_native::{DType, MlxBuffer};

use super::autograd_gpu_tape::{
    exp, log, matmul, mul, outer_product, row_sum, scalar_mul, softmax,
    take_along_axis_topk, GpuTensor,
};

pub struct MoeRouteOutput {
    /// Differentiable per-row top-K scores, shape `[n_tokens, k]`.
    pub top_k_scores: GpuTensor,
    /// Non-differentiable per-row top-K expert indices, shape
    /// `[n_tokens, k]` u32.  Caller passes this to a downstream
    /// `switch_mlp(x, top_k_indices)` dispatch.
    pub top_k_indices: MlxBuffer,
}

/// Compose MoE routing forward on the tape.  Returns differentiable
/// scores + non-differentiable indices for downstream expert dispatch.
///
/// `ones_k` is a caller-supplied tape leaf of shape `[k]` containing
/// `1.0`s.  Reused across the renorm broadcast (avoids the cost of
/// recreating it inside the composition; same convention as
/// `gated_delta_step`'s `ones_dv` parameter).
pub fn moe_route(
    input: &GpuTensor,
    gate_weight: &GpuTensor,
    k: usize,
    norm_topk_prob: bool,
    ones_k: &GpuTensor,
) -> Result<MoeRouteOutput> {
    if input.shape().len() != 2 {
        return Err(anyhow!(
            "moe_route: input must be 2-D [n_tokens, hidden]; got {:?}",
            input.shape()
        ));
    }
    if gate_weight.shape().len() != 2 {
        return Err(anyhow!(
            "moe_route: gate_weight must be 2-D [hidden, n_experts]; got {:?}",
            gate_weight.shape()
        ));
    }
    let n_tokens = input.shape()[0];
    let hidden = input.shape()[1];
    if gate_weight.shape()[0] != hidden {
        return Err(anyhow!(
            "moe_route: gate_weight rows {} != input hidden {hidden}",
            gate_weight.shape()[0]
        ));
    }
    let n_experts = gate_weight.shape()[1];
    if k == 0 || k > n_experts {
        return Err(anyhow!(
            "moe_route: k must be in (0, n_experts={n_experts}]; got {k}"
        ));
    }
    if ones_k.shape() != &[k] {
        return Err(anyhow!(
            "moe_route: ones_k.shape {:?} != [k={k}]",
            ones_k.shape()
        ));
    }

    // Step 1: gate_logits = input @ gate_weight  → [n_tokens, n_experts]
    let gate_logits = matmul(input, gate_weight)?;

    // Step 2: probs = softmax(gate_logits)
    let probs = softmax(&gate_logits)?;

    // Step 3 (OFF TAPE): per-row argpartition for top-K indices.
    let probs_host = probs.to_vec()?;
    let mut top_k_idx_host: Vec<u32> = Vec::with_capacity(n_tokens * k);
    for r in 0..n_tokens {
        let row = &probs_host[r * n_experts..(r + 1) * n_experts];
        let mut indexed: Vec<(usize, f32)> =
            row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        // Partial sort: top-K by DESCENDING value.
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        for i in 0..k {
            top_k_idx_host.push(indexed[i].0 as u32);
        }
    }
    let tape = input.tape();
    let device = tape.device();
    let mut idx_buf = device
        .alloc_buffer(n_tokens * k * 4, DType::U32, vec![n_tokens, k])
        .map_err(|e| anyhow!("moe_route: alloc indices: {e}"))?;
    idx_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("moe_route: indices write: {e}"))?
        .copy_from_slice(&top_k_idx_host);

    // Step 4: scores = take_along_axis_topk(probs, indices)  → [n_tokens, k]
    // take_along_axis_topk consumes the indices buffer; clone for return.
    let idx_buf_for_op = idx_buf.clone();
    let scores = take_along_axis_topk(&probs, idx_buf_for_op, k)?;

    let final_scores = if norm_topk_prob {
        // Step 5a: sum_k = row_sum(scores)  → [n_tokens]
        let sum_k = row_sum(&scores)?;
        // Step 5b: reciprocal via exp(-log(sum)).  All sums are
        // positive (sum of softmax outputs), so log/exp are safe.
        let log_sum = log(&sum_k)?;
        let neg_log_sum = scalar_mul(&log_sum, -1.0)?;
        let recip = exp(&neg_log_sum)?;
        // Step 5c: broadcast recip to [n_tokens, k] via outer with ones_k.
        let recip_broadcast = outer_product(&recip, ones_k)?;
        // Step 5d: normalized = scores * recip_broadcast.
        mul(&scores, &recip_broadcast)?
    } else {
        scores
    };

    Ok(MoeRouteOutput {
        top_k_scores: final_scores,
        top_k_indices: idx_buf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::autograd_gpu_tape::{backward, ones_like, GpuTape};
    use mlx_native::MlxDevice;

    /// CPU reference for the moe_route forward path.
    /// Returns (top_k_scores, top_k_indices).
    fn moe_route_cpu(
        input: &[f32],
        gate_w: &[f32],
        n_tokens: usize,
        hidden: usize,
        n_experts: usize,
        k: usize,
        norm: bool,
    ) -> (Vec<f32>, Vec<u32>) {
        // Step 1 + 2: gate_logits = input @ gate_w; probs = softmax.
        let mut probs = vec![0.0f32; n_tokens * n_experts];
        for r in 0..n_tokens {
            // logits row
            let mut row = vec![0.0f64; n_experts];
            for j in 0..n_experts {
                let mut acc = 0.0f64;
                for h in 0..hidden {
                    acc += input[r * hidden + h] as f64 * gate_w[h * n_experts + j] as f64;
                }
                row[j] = acc;
            }
            // softmax (stable)
            let max = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f64> = row.iter().map(|&v| (v - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for j in 0..n_experts {
                probs[r * n_experts + j] = (exps[j] / sum) as f32;
            }
        }
        // Step 3: top-K indices per row.
        let mut indices = vec![0u32; n_tokens * k];
        for r in 0..n_tokens {
            let row = &probs[r * n_experts..(r + 1) * n_experts];
            let mut indexed: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            for j in 0..k {
                indices[r * k + j] = indexed[j].0 as u32;
            }
        }
        // Step 4: gather scores.
        let mut scores = vec![0.0f32; n_tokens * k];
        for r in 0..n_tokens {
            for j in 0..k {
                let idx = indices[r * k + j] as usize;
                scores[r * k + j] = probs[r * n_experts + idx];
            }
        }
        // Step 5: optional renorm.
        if norm {
            for r in 0..n_tokens {
                let sum: f64 =
                    (0..k).map(|j| scores[r * k + j] as f64).sum();
                for j in 0..k {
                    scores[r * k + j] = (scores[r * k + j] as f64 / sum) as f32;
                }
            }
        }
        (scores, indices)
    }

    /// Forward parity vs CPU oracle.  Uses dimensions ≥ 32 to satisfy
    /// matmul kernel's m/n/k floor.
    #[test]
    fn forward_matches_cpu_oracle_no_renorm() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);

        let n_tokens = 32usize;
        let hidden = 32usize;
        let n_experts = 32usize;
        let k = 4usize;
        let input_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.011 - 0.3).sin() * 0.5)
            .collect();
        let gate_w_data: Vec<f32> = (0..(hidden * n_experts))
            .map(|i| 0.05 + (i as f32) * 0.001)
            .collect();
        let ones_k_data: Vec<f32> = vec![1.0; k];

        let xt = GpuTensor::from_vec(&tape, &input_data, vec![n_tokens, hidden]).unwrap();
        let wt =
            GpuTensor::from_vec(&tape, &gate_w_data, vec![hidden, n_experts]).unwrap();
        let ot = GpuTensor::from_vec(&tape, &ones_k_data, vec![k]).unwrap();

        let out = moe_route(&xt, &wt, k, /* norm */ false, &ot).unwrap();
        let scores_host = out.top_k_scores.to_vec().unwrap();
        let indices_host = out
            .top_k_indices
            .as_slice::<u32>()
            .unwrap()
            .to_vec();

        let (s_cpu, i_cpu) = moe_route_cpu(
            &input_data, &gate_w_data, n_tokens, hidden, n_experts, k, false,
        );

        // Indices must match exactly (deterministic argpartition).
        assert_eq!(indices_host, i_cpu);

        // Scores match within 1e-3 (softmax + matmul accumulate FP errors).
        for r in 0..n_tokens {
            for j in 0..k {
                let g = scores_host[r * k + j];
                let c = s_cpu[r * k + j];
                assert!(
                    (g - c).abs() < 1e-3 * c.abs().max(1e-4),
                    "scores[{r},{j}]: gpu={} cpu={}",
                    g, c
                );
            }
        }
    }

    /// Forward parity with renorm enabled.
    #[test]
    fn forward_matches_cpu_oracle_with_renorm() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);

        let n_tokens = 32usize;
        let hidden = 32usize;
        let n_experts = 32usize;
        let k = 4usize;
        let input_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.013 - 0.4).cos() * 0.4)
            .collect();
        let gate_w_data: Vec<f32> = (0..(hidden * n_experts))
            .map(|i| 0.07 + (i as f32) * 0.0007)
            .collect();
        let ones_k_data: Vec<f32> = vec![1.0; k];

        let xt = GpuTensor::from_vec(&tape, &input_data, vec![n_tokens, hidden]).unwrap();
        let wt =
            GpuTensor::from_vec(&tape, &gate_w_data, vec![hidden, n_experts]).unwrap();
        let ot = GpuTensor::from_vec(&tape, &ones_k_data, vec![k]).unwrap();

        let out = moe_route(&xt, &wt, k, /* norm */ true, &ot).unwrap();
        let scores_host = out.top_k_scores.to_vec().unwrap();

        let (s_cpu, _i_cpu) = moe_route_cpu(
            &input_data, &gate_w_data, n_tokens, hidden, n_experts, k, true,
        );

        for r in 0..n_tokens {
            for j in 0..k {
                let g = scores_host[r * k + j];
                let c = s_cpu[r * k + j];
                assert!(
                    (g - c).abs() < 1e-3 * c.abs().max(1e-4),
                    "renormed scores[{r},{j}]: gpu={} cpu={}",
                    g, c
                );
            }
        }
        // Sanity: rows sum to 1.0 (renorm invariant).
        for r in 0..n_tokens {
            let s: f64 =
                (0..k).map(|j| scores_host[r * k + j] as f64).sum();
            assert!(
                (s - 1.0).abs() < 1e-4,
                "renorm row {r} sum={} != 1.0",
                s
            );
        }
    }

    /// FD falsifier on a sampled set of input + gate_weight elements.
    /// LOAD-BEARING: proves chain-rule routing through the entire
    /// composition (matmul → softmax → take_along_axis → row_sum →
    /// log → scalar_mul → exp → outer_product → mul = 9 OpKinds).
    /// 5% rel tolerance.
    #[test]
    fn backward_finite_diff_falsifier_with_renorm() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);

        let n_tokens = 32usize;
        let hidden = 32usize;
        let n_experts = 32usize;
        let k = 3usize;
        let input_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.0173 - 0.3).sin() * 0.3)
            .collect();
        let gate_w_data: Vec<f32> = (0..(hidden * n_experts))
            .map(|i| 0.04 + (i as f32) * 0.0005)
            .collect();
        let ones_k_data: Vec<f32> = vec![1.0; k];

        let forward_loss_and_grads = |xv: &[f32],
                                       wv: &[f32]|
         -> (f32, Vec<f32>, Vec<f32>) {
            tape.reset();
            let xt = GpuTensor::from_vec(&tape, xv, vec![n_tokens, hidden]).unwrap();
            let wt =
                GpuTensor::from_vec(&tape, wv, vec![hidden, n_experts]).unwrap();
            let ot =
                GpuTensor::from_vec(&tape, &ones_k_data, vec![k]).unwrap();
            let out = moe_route(&xt, &wt, k, true, &ot).unwrap();
            let host = out.top_k_scores.to_vec().unwrap();
            let loss = host.iter().map(|v| *v as f64).sum::<f64>() as f32;
            let dy = ones_like(&tape, out.top_k_scores.shape()).unwrap();
            let grads = backward(&out.top_k_scores, dy).unwrap();
            let g_x = grads[xt.node_idx()]
                .as_ref()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            let g_w = grads[wt.node_idx()]
                .as_ref()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            (loss, g_x, g_w)
        };

        let (_l0, g_x0, g_w0) = forward_loss_and_grads(&input_data, &gate_w_data);
        let h = 1e-3f32;
        // Spot-check 6 input + 6 gate_weight elements.
        for &i in &[0, 17, 100, 333, 700, 1023] {
            let mut p = input_data.clone();
            p[i] += h;
            let (lp, _, _) = forward_loss_and_grads(&p, &gate_w_data);
            let mut m = input_data.clone();
            m[i] -= h;
            let (lm, _, _) = forward_loss_and_grads(&m, &gate_w_data);
            let fd = (lp - lm) / (2.0 * h);
            let tol = 5e-2 * fd.abs().max(1.0);
            assert!(
                (g_x0[i] - fd).abs() < tol,
                "FD x[{i}]: analytic={} fd={}",
                g_x0[i], fd
            );
        }
        for &j in &[0, 11, 100, 333, 700, 1023] {
            let mut p = gate_w_data.clone();
            p[j] += h;
            let (lp, _, _) = forward_loss_and_grads(&input_data, &p);
            let mut m = gate_w_data.clone();
            m[j] -= h;
            let (lm, _, _) = forward_loss_and_grads(&input_data, &m);
            let fd = (lp - lm) / (2.0 * h);
            let tol = 5e-2 * fd.abs().max(1.0);
            assert!(
                (g_w0[j] - fd).abs() < tol,
                "FD w[{j}]: analytic={} fd={}",
                g_w0[j], fd
            );
        }
    }

    #[test]
    fn rejects_invalid_k() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt =
            GpuTensor::from_vec(&tape, &vec![0.0; 32 * 32], vec![32, 32]).unwrap();
        let wt =
            GpuTensor::from_vec(&tape, &vec![0.0; 32 * 32], vec![32, 32]).unwrap();
        let ot = GpuTensor::from_vec(&tape, &vec![1.0; 4], vec![4]).unwrap();
        // k=0
        assert!(moe_route(&xt, &wt, 0, false, &ot).is_err());
        // k > n_experts
        assert!(moe_route(&xt, &wt, 33, false, &ot).is_err());
    }
}
