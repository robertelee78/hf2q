//! Qwen 3.5 SwiGLU FFN — GpuTape composition for ADR-020 iter-11b.
//!
//! ```text
//!   gate     = matmul(x, W_gate)             [batch, intermediate]
//!   up       = matmul(x, W_up)               [batch, intermediate]
//!   silu_gate = silu(gate)
//!   prod     = silu_gate · up                (elementwise mul)
//!   y        = matmul(prod, W_down)          [batch, hidden]
//! ```
//!
//! All three projections (`W_gate`, `W_up`, `W_down`) are quantizable
//! Linears.  Backward routes gradients to all three weight leaves
//! plus the input `x`.

use anyhow::{anyhow, Result};

use crate::calibrate::autograd_gpu_tape::{matmul, mul, silu, GpuTape, GpuTensor};

/// Configuration for a SwiGLU FFN block.  All dims must be ≥ 32
/// (matmul kernel constraint).
#[derive(Debug, Clone, Copy)]
pub struct FfnConfig {
    pub batch: usize,
    pub hidden: usize,
    pub intermediate: usize,
}

impl FfnConfig {
    pub fn validate(&self) -> Result<()> {
        if self.batch < 32 || self.hidden < 32 || self.intermediate < 32 {
            return Err(anyhow!(
                "FfnConfig: all dims must be ≥ 32; got batch={} hidden={} intermediate={}",
                self.batch,
                self.hidden,
                self.intermediate
            ));
        }
        Ok(())
    }
}

/// Owned weights for a SwiGLU FFN block.  All matrices are row-major fp32.
#[derive(Debug, Clone)]
pub struct FfnWeights {
    /// `[hidden, intermediate]`
    pub w_gate: Vec<f32>,
    /// `[hidden, intermediate]`
    pub w_up: Vec<f32>,
    /// `[intermediate, hidden]`
    pub w_down: Vec<f32>,
}

impl FfnWeights {
    pub fn new(
        cfg: &FfnConfig,
        w_gate: Vec<f32>,
        w_up: Vec<f32>,
        w_down: Vec<f32>,
    ) -> Result<Self> {
        cfg.validate()?;
        let gate_up_len = cfg.hidden * cfg.intermediate;
        let down_len = cfg.intermediate * cfg.hidden;
        if w_gate.len() != gate_up_len {
            return Err(anyhow!(
                "w_gate length {} != hidden*intermediate {}",
                w_gate.len(),
                gate_up_len
            ));
        }
        if w_up.len() != gate_up_len {
            return Err(anyhow!(
                "w_up length {} != hidden*intermediate {}",
                w_up.len(),
                gate_up_len
            ));
        }
        if w_down.len() != down_len {
            return Err(anyhow!(
                "w_down length {} != intermediate*hidden {}",
                w_down.len(),
                down_len
            ));
        }
        Ok(Self {
            w_gate,
            w_up,
            w_down,
        })
    }
}

/// Tape leaves for the three FFN projection weights.
#[derive(Clone)]
pub struct FfnLeaves {
    pub w_gate: GpuTensor,
    pub w_up: GpuTensor,
    pub w_down: GpuTensor,
}

impl FfnLeaves {
    pub fn from_weights(tape: &GpuTape, cfg: &FfnConfig, w: &FfnWeights) -> Result<Self> {
        Ok(Self {
            w_gate: GpuTensor::from_vec(tape, &w.w_gate, vec![cfg.hidden, cfg.intermediate])?,
            w_up: GpuTensor::from_vec(tape, &w.w_up, vec![cfg.hidden, cfg.intermediate])?,
            w_down: GpuTensor::from_vec(tape, &w.w_down, vec![cfg.intermediate, cfg.hidden])?,
        })
    }

    pub fn from_weights_qdq<F>(
        tape: &GpuTape,
        cfg: &FfnConfig,
        w: &FfnWeights,
        qdq_fn: F,
    ) -> Result<Self>
    where
        F: Fn(&[f32]) -> Result<Vec<f32>>,
    {
        let w_gate_q = qdq_fn(&w.w_gate)?;
        let w_up_q = qdq_fn(&w.w_up)?;
        let w_down_q = qdq_fn(&w.w_down)?;
        Ok(Self {
            w_gate: GpuTensor::from_vec(tape, &w_gate_q, vec![cfg.hidden, cfg.intermediate])?,
            w_up: GpuTensor::from_vec(tape, &w_up_q, vec![cfg.hidden, cfg.intermediate])?,
            w_down: GpuTensor::from_vec(tape, &w_down_q, vec![cfg.intermediate, cfg.hidden])?,
        })
    }
}

/// Forward the SwiGLU FFN on the GpuTape.
pub fn forward(cfg: &FfnConfig, x: &GpuTensor, leaves: &FfnLeaves) -> Result<GpuTensor> {
    cfg.validate()?;
    if x.shape() != [cfg.batch, cfg.hidden] {
        return Err(anyhow!(
            "ffn forward: x shape {:?} != [batch={}, hidden={}]",
            x.shape(),
            cfg.batch,
            cfg.hidden
        ));
    }
    let gate = matmul(x, &leaves.w_gate)?; // [batch, intermediate]
    let up = matmul(x, &leaves.w_up)?; // [batch, intermediate]
    let silu_gate = silu(&gate)?; // [batch, intermediate]
    let prod = mul(&silu_gate, &up)?; // [batch, intermediate]
    let y = matmul(&prod, &leaves.w_down)?; // [batch, hidden]
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::autograd_gpu_tape::{backward, ones_like};
    use mlx_native::MlxDevice;

    fn deterministic_weights(cfg: &FfnConfig, seed: u64) -> FfnWeights {
        let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        let mut next = || {
            state ^= state >> 33;
            state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
            state ^= state >> 33;
            ((state as i64) as f32) / (i64::MAX as f32)
        };
        let gate_len = cfg.hidden * cfg.intermediate;
        let down_len = cfg.intermediate * cfg.hidden;
        let w_gate: Vec<f32> = (0..gate_len).map(|_| next() * 0.4).collect();
        let w_up: Vec<f32> = (0..gate_len).map(|_| next() * 0.4).collect();
        let w_down: Vec<f32> = (0..down_len).map(|_| next() * 0.4).collect();
        FfnWeights::new(cfg, w_gate, w_up, w_down).unwrap()
    }

    fn ffn_cpu_oracle(
        x: &[f32],
        w: &FfnWeights,
        batch: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Vec<f32> {
        // gate = x @ W_gate; up = x @ W_up
        let mut gate = vec![0f32; batch * intermediate];
        let mut up = vec![0f32; batch * intermediate];
        for b in 0..batch {
            for n in 0..intermediate {
                let mut g_acc = 0.0f32;
                let mut u_acc = 0.0f32;
                for k in 0..hidden {
                    g_acc += x[b * hidden + k] * w.w_gate[k * intermediate + n];
                    u_acc += x[b * hidden + k] * w.w_up[k * intermediate + n];
                }
                gate[b * intermediate + n] = g_acc;
                up[b * intermediate + n] = u_acc;
            }
        }
        // prod = silu(gate) * up
        let prod: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| {
                let s = 1.0 / (1.0 + (-g).exp());
                (g * s) * u
            })
            .collect();
        // y = prod @ W_down
        let mut y = vec![0f32; batch * hidden];
        for b in 0..batch {
            for n in 0..hidden {
                let mut acc = 0.0f32;
                for k in 0..intermediate {
                    acc += prod[b * intermediate + k] * w.w_down[k * hidden + n];
                }
                y[b * hidden + n] = acc;
            }
        }
        y
    }

    fn assert_close(label: &str, gpu: &[f32], cpu: &[f32], rel_tol: f32, abs_tol: f32) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            let scale = g.abs().max(c.abs()).max(1.0);
            assert!(
                diff <= abs_tol || diff / scale <= rel_tol,
                "{label}: i={i}: gpu={g} cpu={c} diff={diff}"
            );
        }
    }

    #[test]
    fn ffn_forward_parity_with_cpu() {
        let cfg = FfnConfig {
            batch: 32,
            hidden: 32,
            intermediate: 64,
        };
        let weights = deterministic_weights(&cfg, 4242);
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let x: Vec<f32> = (0..cfg.batch * cfg.hidden)
            .map(|i| (i as f32 * 0.013).sin() * 0.5)
            .collect();
        let xt = GpuTensor::from_vec(&tape, &x, vec![cfg.batch, cfg.hidden]).unwrap();
        let leaves = FfnLeaves::from_weights(&tape, &cfg, &weights).unwrap();
        let y = forward(&cfg, &xt, &leaves).unwrap();
        let y_gpu: Vec<f32> = y.to_vec().unwrap();
        let y_cpu = ffn_cpu_oracle(&x, &weights, cfg.batch, cfg.hidden, cfg.intermediate);
        assert_close("ffn forward", &y_gpu, &y_cpu, 1e-4, 1e-5);
    }

    #[test]
    fn ffn_backward_flows_to_all_three_weight_leaves() {
        let cfg = FfnConfig {
            batch: 32,
            hidden: 32,
            intermediate: 64,
        };
        let weights = deterministic_weights(&cfg, 31337);
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let x: Vec<f32> = (0..cfg.batch * cfg.hidden)
            .map(|i| (i as f32 * 0.011).cos() * 0.4)
            .collect();
        let xt = GpuTensor::from_vec(&tape, &x, vec![cfg.batch, cfg.hidden]).unwrap();
        let leaves = FfnLeaves::from_weights(&tape, &cfg, &weights).unwrap();
        let y = forward(&cfg, &xt, &leaves).unwrap();
        let dy = ones_like(&tape, &[cfg.batch, cfg.hidden]).unwrap();
        let grads = backward(&y, dy).unwrap();
        for (label, leaf, expected_numel) in [
            ("x", &xt, cfg.batch * cfg.hidden),
            ("w_gate", &leaves.w_gate, cfg.hidden * cfg.intermediate),
            ("w_up", &leaves.w_up, cfg.hidden * cfg.intermediate),
            ("w_down", &leaves.w_down, cfg.intermediate * cfg.hidden),
        ] {
            let g = grads[leaf.node_idx()]
                .as_ref()
                .unwrap_or_else(|| panic!("{label} grad missing"));
            assert_eq!(g.element_count(), expected_numel);
            let g_vec: Vec<f32> = g.as_slice::<f32>().unwrap().to_vec();
            for (i, v) in g_vec.iter().enumerate() {
                let val: f32 = *v;
                assert!(val.is_finite(), "{label} grad[{i}] = {val} not finite");
            }
        }
    }

    #[test]
    fn ffn_qdq_low_high_diff_nonzero() {
        // Sanity: qdq Q4_0 vs Q8_0 of FFN weights must produce
        // non-zero diffs (gradient-Taylor anchor sanity).
        use crate::calibrate::qdq_gpu::{qdq_q4_0_gpu, qdq_q8_0_gpu};
        let cfg = FfnConfig {
            batch: 32,
            hidden: 32,
            intermediate: 64,
        };
        let weights = deterministic_weights(&cfg, 7777);
        let device = MlxDevice::new().expect("device");
        let max_diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max)
        };
        for (label, ws) in [
            ("w_gate", &weights.w_gate),
            ("w_up", &weights.w_up),
            ("w_down", &weights.w_down),
        ] {
            let low = qdq_q4_0_gpu(&device, ws).unwrap();
            let high = qdq_q8_0_gpu(&device, ws).unwrap();
            let d = max_diff(&low, &high);
            assert!(
                d > 1e-4,
                "{label}: qdq Q4_0 vs Q8_0 max-diff {d} too small"
            );
        }
    }
}
