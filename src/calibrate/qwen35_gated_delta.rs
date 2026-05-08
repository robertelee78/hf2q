//! ADR-020 iter-11h-c3 — single-step `gated_delta_step` recurrence
//! composition on GpuTape.  Required for differentiable Qwen3.5MoE
//! forward (the linear-attention `GatedDeltaNet` layer) — which
//! itself is the load-bearing forward path for full-model SOTA DWQ
//! training (iter-11h-f → iter-12d).
//!
//! ## Math (canonical reference)
//!
//! mirrors `mlx-lm/mlx_lm/models/gated_delta.py:_gated_delta_step_ops`
//! (mlx-lm canonical) for B=1, H=1, with `g` and `beta`
//! caller-pre-broadcasted to the operand shapes that GpuTape ops
//! accept (no broadcast op exists yet on tape, so the caller
//! materialises the broadcast explicitly):
//!
//! ```text
//!   state_dec   = state * g                            // [Dv, Dk]
//!   kv_mem      = row_sum(state_dec * outer(1_Dv, k))  // [Dv]
//!   delta       = beta * (v - kv_mem)                  // [Dv]
//!   state_out   = state_dec + outer(delta, k)          // [Dv, Dk]
//!   y           = row_sum(state_out * outer(1_Dv, q))  // [Dv]
//! ```
//!
//! The "row_sum(matrix * outer(ones, vec))" pattern replaces the
//! `matrix @ vec` matvec (matmul has a 32-floor on inner-dim that
//! can't accommodate inner-dim = 1; outer_product + row_sum has no
//! such constraint).
//!
//! ## Shape contract
//!
//!   * `q`, `k` : `[Dk]`
//!   * `v` : `[Dv]`
//!   * `g` : `[Dv, Dk]` (caller broadcasts mlx-lm's `g[..., None, None]`)
//!   * `beta` : `[Dv]` (caller broadcasts mlx-lm's `beta[..., None]`)
//!   * `state` : `[Dv, Dk]`
//!   * `ones_dv` : `[Dv]` (a tape leaf of all 1.0s — caller-supplied so
//!     it can be reused across timesteps without recreation overhead)
//!   * Returns `y : [Dv]` and `state_out : [Dv, Dk]`
//!
//! ## Why not full T-token recurrence here
//!
//! T-token recurrence requires sequential composition (per-step output
//! becomes next-step input).  GpuTape supports this via repeated calls
//! on the same tape — caller drives the per-token loop.  THIS function
//! is the per-step primitive.  iter-11h-d will derive the analytic
//! backward; until then, backward auto-derives via the chain rule
//! through the 11 composed OpKinds.

use anyhow::{anyhow, Result};

use super::autograd_gpu_tape::{
    add, mul, outer_product, row_sum, sub, GpuTensor,
};

pub struct GatedDeltaStepOutput {
    /// Per-token output, shape `[Dv]`.
    pub y: GpuTensor,
    /// Updated recurrent state, shape `[Dv, Dk]`.
    pub state_out: GpuTensor,
}

/// Compose one step of the gated DeltaNet recurrence on the tape.
/// Forward + backward via existing OpKind composition; no new
/// kernels.
pub fn gated_delta_step(
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    g: &GpuTensor,
    beta: &GpuTensor,
    state: &GpuTensor,
    ones_dv: &GpuTensor,
) -> Result<GatedDeltaStepOutput> {
    if q.shape().len() != 1 || k.shape().len() != 1 {
        return Err(anyhow!(
            "gated_delta_step: q and k must be 1-D [Dk]; got q={:?}, k={:?}",
            q.shape(), k.shape()
        ));
    }
    if q.shape()[0] != k.shape()[0] {
        return Err(anyhow!(
            "gated_delta_step: q.dim {} != k.dim {}",
            q.shape()[0], k.shape()[0]
        ));
    }
    let dk = q.shape()[0];
    if v.shape().len() != 1 {
        return Err(anyhow!(
            "gated_delta_step: v must be 1-D [Dv]; got {:?}",
            v.shape()
        ));
    }
    let dv = v.shape()[0];
    if state.shape() != &[dv, dk] {
        return Err(anyhow!(
            "gated_delta_step: state shape {:?} != [Dv={dv}, Dk={dk}]",
            state.shape()
        ));
    }
    if g.shape() != &[dv, dk] {
        return Err(anyhow!(
            "gated_delta_step: g shape {:?} != [Dv={dv}, Dk={dk}] (caller must broadcast)",
            g.shape()
        ));
    }
    if beta.shape() != &[dv] {
        return Err(anyhow!(
            "gated_delta_step: beta shape {:?} != [Dv={dv}]",
            beta.shape()
        ));
    }
    if ones_dv.shape() != &[dv] {
        return Err(anyhow!(
            "gated_delta_step: ones_dv shape {:?} != [Dv={dv}]",
            ones_dv.shape()
        ));
    }

    // Step 1: state_dec = state * g  (elementwise [Dv, Dk]).
    let state_dec = mul(state, g)?;

    // Step 2: k_broadcast = outer(1_Dv, k)  → [Dv, Dk] where each row is k.
    let k_broadcast = outer_product(ones_dv, k)?;

    // Step 3: state_k = state_dec * k_broadcast  → [Dv, Dk].
    let state_k = mul(&state_dec, &k_broadcast)?;

    // Step 4: kv_mem = row_sum(state_k)  → [Dv].
    let kv_mem = row_sum(&state_k)?;

    // Step 5: v - kv_mem  → [Dv].
    let v_minus_kv = sub(v, &kv_mem)?;

    // Step 6: delta = beta * (v - kv_mem)  → [Dv].
    let delta = mul(beta, &v_minus_kv)?;

    // Step 7: delta_outer = outer(delta, k)  → [Dv, Dk].
    let delta_outer = outer_product(&delta, k)?;

    // Step 8: state_out = state_dec + delta_outer  → [Dv, Dk].
    let state_out = add(&state_dec, &delta_outer)?;

    // Step 9: q_broadcast = outer(1_Dv, q)  → [Dv, Dk].
    let q_broadcast = outer_product(ones_dv, q)?;

    // Step 10: state_q = state_out * q_broadcast  → [Dv, Dk].
    let state_q = mul(&state_out, &q_broadcast)?;

    // Step 11: y = row_sum(state_q)  → [Dv].
    let y = row_sum(&state_q)?;

    Ok(GatedDeltaStepOutput { y, state_out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::autograd_gpu_tape::{
        backward, ones_like, GpuTape,
    };
    use mlx_native::MlxDevice;

    /// CPU reference per `mlx-lm/gated_delta.py:_gated_delta_step_ops`
    /// for a single (B=1, H=1, scalar-broadcasted-g/beta) step.
    fn gated_delta_step_cpu(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state: &[f32],
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        // state_dec = state * g
        let mut state_dec = vec![0.0f32; dv * dk];
        for i in 0..(dv * dk) {
            state_dec[i] = state[i] * g[i];
        }
        // kv_mem[d] = Σ_i state_dec[d, i] * k[i]
        let mut kv_mem = vec![0.0f32; dv];
        for d in 0..dv {
            let mut acc = 0.0f64;
            for i in 0..dk {
                acc += state_dec[d * dk + i] as f64 * k[i] as f64;
            }
            kv_mem[d] = acc as f32;
        }
        // delta = beta * (v - kv_mem)
        let mut delta = vec![0.0f32; dv];
        for d in 0..dv {
            delta[d] = beta[d] * (v[d] - kv_mem[d]);
        }
        // state_out = state_dec + outer(delta, k)
        let mut state_out = vec![0.0f32; dv * dk];
        for d in 0..dv {
            for i in 0..dk {
                state_out[d * dk + i] = state_dec[d * dk + i] + delta[d] * k[i];
            }
        }
        // y[d] = Σ_i state_out[d, i] * q[i]
        let mut y = vec![0.0f32; dv];
        for d in 0..dv {
            let mut acc = 0.0f64;
            for i in 0..dk {
                acc += state_out[d * dk + i] as f64 * q[i] as f64;
            }
            y[d] = acc as f32;
        }
        (y, state_out)
    }

    /// Forward parity vs CPU oracle.
    #[test]
    fn forward_matches_cpu_oracle() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);

        let dv = 8usize;
        let dk = 5usize;
        let q_data: Vec<f32> = (0..dk).map(|i| 0.3 + (i as f32) * 0.07).collect();
        let k_data: Vec<f32> = (0..dk).map(|i| ((i as f32) * 0.137 - 0.4).sin() * 0.6).collect();
        let v_data: Vec<f32> = (0..dv).map(|i| 0.4 + (i as f32) * 0.05).collect();
        let g_data: Vec<f32> = (0..(dv * dk))
            .map(|i| 0.85 + ((i as f32) * 0.013).sin() * 0.05)
            .collect();
        let beta_data: Vec<f32> = (0..dv).map(|i| 0.2 + (i as f32) * 0.03).collect();
        let state_data: Vec<f32> = (0..(dv * dk))
            .map(|i| ((i as f32) * 0.043 + 0.1).cos() * 0.3)
            .collect();
        let ones_dv: Vec<f32> = vec![1.0; dv];

        let q = GpuTensor::from_vec(&tape, &q_data, vec![dk]).unwrap();
        let k = GpuTensor::from_vec(&tape, &k_data, vec![dk]).unwrap();
        let v = GpuTensor::from_vec(&tape, &v_data, vec![dv]).unwrap();
        let g = GpuTensor::from_vec(&tape, &g_data, vec![dv, dk]).unwrap();
        let beta = GpuTensor::from_vec(&tape, &beta_data, vec![dv]).unwrap();
        let state = GpuTensor::from_vec(&tape, &state_data, vec![dv, dk]).unwrap();
        let ones = GpuTensor::from_vec(&tape, &ones_dv, vec![dv]).unwrap();

        let out = gated_delta_step(&q, &k, &v, &g, &beta, &state, &ones).unwrap();
        let y_host = out.y.to_vec().unwrap();
        let s_host = out.state_out.to_vec().unwrap();

        let (y_cpu, s_cpu) = gated_delta_step_cpu(
            &q_data, &k_data, &v_data, &g_data, &beta_data, &state_data, dv, dk,
        );
        for d in 0..dv {
            assert!(
                (y_host[d] - y_cpu[d]).abs() < 1e-4 * y_cpu[d].abs().max(1.0),
                "y[{d}]: gpu={} cpu={}",
                y_host[d], y_cpu[d]
            );
        }
        for i in 0..(dv * dk) {
            assert!(
                (s_host[i] - s_cpu[i]).abs() < 1e-5 * s_cpu[i].abs().max(1.0),
                "state_out[{i}]: gpu={} cpu={}",
                s_host[i], s_cpu[i]
            );
        }
    }

    /// FD falsifier on a sample of every input parameter.  Composed
    /// graph: loss = sum(y).  Asserts analytic gradient (from chain
    /// rule through 11 OpKinds) matches central-difference numerical
    /// gradient to within 5% rel tolerance.
    ///
    /// LOAD-BEARING — proves the entire chain (mul × 4, outer_product
    /// × 3, row_sum × 2, sub × 1, add × 1) routes gradients correctly
    /// to all 6 leaves (q, k, v, g, beta, state).  A reversed parent
    /// edge or wrong sign in any single OpKind would surface here.
    #[test]
    fn backward_finite_diff_falsifier() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);

        let dv = 4usize;
        let dk = 3usize;
        let q_data: Vec<f32> = (0..dk).map(|i| 0.3 + (i as f32) * 0.05).collect();
        let k_data: Vec<f32> = (0..dk).map(|i| 0.2 + (i as f32) * 0.07).collect();
        let v_data: Vec<f32> = (0..dv).map(|i| 0.4 + (i as f32) * 0.03).collect();
        let g_data: Vec<f32> = (0..(dv * dk)).map(|i| 0.85 + (i as f32) * 0.005).collect();
        let beta_data: Vec<f32> = (0..dv).map(|i| 0.25 + (i as f32) * 0.02).collect();
        let state_data: Vec<f32> = (0..(dv * dk)).map(|i| 0.1 + (i as f32) * 0.02).collect();
        let ones_dv: Vec<f32> = vec![1.0; dv];

        let forward_loss_and_grads = |q: &[f32],
                                       k: &[f32],
                                       v: &[f32],
                                       g: &[f32],
                                       beta: &[f32],
                                       state: &[f32]|
         -> (f32, [Vec<f32>; 6]) {
            tape.reset();
            let qt = GpuTensor::from_vec(&tape, q, vec![dk]).unwrap();
            let kt = GpuTensor::from_vec(&tape, k, vec![dk]).unwrap();
            let vt = GpuTensor::from_vec(&tape, v, vec![dv]).unwrap();
            let gt = GpuTensor::from_vec(&tape, g, vec![dv, dk]).unwrap();
            let bt = GpuTensor::from_vec(&tape, beta, vec![dv]).unwrap();
            let st = GpuTensor::from_vec(&tape, state, vec![dv, dk]).unwrap();
            let on = GpuTensor::from_vec(&tape, &ones_dv, vec![dv]).unwrap();
            let out = gated_delta_step(&qt, &kt, &vt, &gt, &bt, &st, &on).unwrap();
            let y_host = out.y.to_vec().unwrap();
            let loss = y_host.iter().map(|v| *v as f64).sum::<f64>() as f32;
            let dy = ones_like(&tape, out.y.shape()).unwrap();
            let grads = backward(&out.y, dy).unwrap();
            let read = |idx: usize| -> Vec<f32> {
                grads[idx]
                    .as_ref()
                    .unwrap()
                    .as_slice::<f32>()
                    .unwrap()
                    .to_vec()
            };
            (
                loss,
                [
                    read(qt.node_idx()),
                    read(kt.node_idx()),
                    read(vt.node_idx()),
                    read(gt.node_idx()),
                    read(bt.node_idx()),
                    read(st.node_idx()),
                ],
            )
        };

        let (_l0, grads0) = forward_loss_and_grads(
            &q_data, &k_data, &v_data, &g_data, &beta_data, &state_data,
        );

        let h = 1e-3f32;
        // Spot-check FD on every q/k/v/beta element + sampled g/state
        // elements (sweeping 12 + 12 = 24 total).
        let check = |name: &str,
                     baseline: &[f32],
                     analytic: &[f32],
                     mut perturb: Box<dyn FnMut(&mut Vec<f32>, usize, f32)>,
                     idxs: &[usize],
                     compute: Box<dyn Fn(&Vec<f32>) -> f32>| {
            for &idx in idxs {
                let mut p = baseline.to_vec();
                perturb(&mut p, idx, h);
                let lp = compute(&p);
                let mut m = baseline.to_vec();
                perturb(&mut m, idx, -h);
                let lm = compute(&m);
                let fd = (lp - lm) / (2.0 * h);
                let tol = 5e-2 * fd.abs().max(1.0);
                assert!(
                    (analytic[idx] - fd).abs() < tol,
                    "FD {name}[{idx}]: analytic={} fd={}",
                    analytic[idx],
                    fd
                );
            }
        };

        // q
        check(
            "q",
            &q_data,
            &grads0[0],
            Box::new(|v, i, dx| v[i] += dx),
            &(0..dk).collect::<Vec<_>>(),
            Box::new(|p| {
                forward_loss_and_grads(p, &k_data, &v_data, &g_data, &beta_data, &state_data).0
            }),
        );
        // k
        check(
            "k",
            &k_data,
            &grads0[1],
            Box::new(|v, i, dx| v[i] += dx),
            &(0..dk).collect::<Vec<_>>(),
            Box::new(|p| {
                forward_loss_and_grads(&q_data, p, &v_data, &g_data, &beta_data, &state_data).0
            }),
        );
        // v
        check(
            "v",
            &v_data,
            &grads0[2],
            Box::new(|v, i, dx| v[i] += dx),
            &(0..dv).collect::<Vec<_>>(),
            Box::new(|p| {
                forward_loss_and_grads(&q_data, &k_data, p, &g_data, &beta_data, &state_data).0
            }),
        );
        // beta
        check(
            "beta",
            &beta_data,
            &grads0[4],
            Box::new(|v, i, dx| v[i] += dx),
            &(0..dv).collect::<Vec<_>>(),
            Box::new(|p| {
                forward_loss_and_grads(&q_data, &k_data, &v_data, &g_data, p, &state_data).0
            }),
        );
        // g (sample 4 elements)
        check(
            "g",
            &g_data,
            &grads0[3],
            Box::new(|v, i, dx| v[i] += dx),
            &[0, 3, 7, 11],
            Box::new(|p| {
                forward_loss_and_grads(&q_data, &k_data, &v_data, p, &beta_data, &state_data).0
            }),
        );
        // state (sample 4 elements)
        check(
            "state",
            &state_data,
            &grads0[5],
            Box::new(|v, i, dx| v[i] += dx),
            &[0, 5, 8, 11],
            Box::new(|p| {
                forward_loss_and_grads(&q_data, &k_data, &v_data, &g_data, &beta_data, p).0
            }),
        );
    }

    #[test]
    fn rejects_shape_mismatches() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let dv = 4;
        let dk = 3;
        let q = GpuTensor::from_vec(&tape, &vec![0.0; dk], vec![dk]).unwrap();
        let k = GpuTensor::from_vec(&tape, &vec![0.0; dk], vec![dk]).unwrap();
        let v = GpuTensor::from_vec(&tape, &vec![0.0; dv], vec![dv]).unwrap();
        let g_bad = GpuTensor::from_vec(&tape, &vec![0.0; dv], vec![dv]).unwrap();
        let beta = GpuTensor::from_vec(&tape, &vec![0.0; dv], vec![dv]).unwrap();
        let state = GpuTensor::from_vec(&tape, &vec![0.0; dv * dk], vec![dv, dk]).unwrap();
        let ones = GpuTensor::from_vec(&tape, &vec![1.0; dv], vec![dv]).unwrap();
        // g has wrong shape (1-D instead of 2-D).
        let res = gated_delta_step(&q, &k, &v, &g_bad, &beta, &state, &ones);
        assert!(res.is_err());
    }
}
