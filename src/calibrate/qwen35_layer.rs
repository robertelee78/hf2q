//! Qwen 3.5 transformer layer — GpuTape composition for ADR-020 iter-11c.
//!
//! Composes the standard pre-norm transformer layer:
//!
//! ```text
//!   normed_attn = rms_norm(x, w_attn_norm, eps)
//!   q           = matmul(normed_attn, W_q)         [batch, hidden]
//!   k           = matmul(normed_attn, W_k)
//!   v           = matmul(normed_attn, W_v)
//!   context     = multi_head_sdpa(q, k, v, n_heads, head_dim)
//!   attn_out    = matmul(context, W_o)             [batch, hidden]
//!   y_attn      = x + attn_out                     [residual]
//!   normed_ffn  = rms_norm(y_attn, w_ffn_norm, eps)
//!   ffn_out     = ffn::forward(normed_ffn, W_gate, W_up, W_down)
//!   y_final     = y_attn + ffn_out                 [residual]
//! ```
//!
//! Seven quantizable Linear weights per layer (4 attention + 3 FFN);
//! two RMSNorm weights are frozen.  This module is composition-only —
//! all underlying ops were already on the tape (rms_norm, matmul,
//! multi_head_sdpa, silu, mul, add).
//!
//! 1/√head_dim attention scale assumed folded into `W_q` at
//! construction (see [`LayerWeights::new`]).

use anyhow::{anyhow, Result};

use crate::calibrate::autograd_gpu_tape::{add, matmul, rms_norm, GpuTape, GpuTensor};
use crate::calibrate::qwen35_attention_block::multi_head_sdpa;
use crate::calibrate::qwen35_ffn::{self, FfnConfig, FfnLeaves};

/// Configuration for one transformer layer.  All dims must be ≥ 32
/// (matmul-backward kernel constraint).
#[derive(Debug, Clone, Copy)]
pub struct LayerConfig {
    pub batch: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub eps: f32,
}

impl LayerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.batch < 32
            || self.hidden < 32
            || self.head_dim < 32
            || self.intermediate < 32
        {
            return Err(anyhow!(
                "LayerConfig: all dims must be ≥ 32; got batch={} hidden={} head_dim={} intermediate={}",
                self.batch,
                self.hidden,
                self.head_dim,
                self.intermediate
            ));
        }
        if self.n_heads == 0 {
            return Err(anyhow!("LayerConfig: n_heads must be > 0"));
        }
        if self.n_heads * self.head_dim != self.hidden {
            return Err(anyhow!(
                "LayerConfig: n_heads({}) * head_dim({}) = {} != hidden({})",
                self.n_heads,
                self.head_dim,
                self.n_heads * self.head_dim,
                self.hidden
            ));
        }
        if !self.eps.is_finite() || self.eps < 0.0 {
            return Err(anyhow!(
                "LayerConfig: eps must be finite + non-negative; got {}",
                self.eps
            ));
        }
        Ok(())
    }
}

/// Owned weights for one transformer layer.  All matrices row-major fp32.
///
/// `W_q` has the 1/√head_dim attention scale folded in by
/// [`Self::new`] (caller passes the unscaled `W_q`).
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub w_attn_norm: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub w_ffn_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

impl LayerWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &LayerConfig,
        w_attn_norm: Vec<f32>,
        w_q_unscaled: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        w_o: Vec<f32>,
        w_ffn_norm: Vec<f32>,
        w_gate: Vec<f32>,
        w_up: Vec<f32>,
        w_down: Vec<f32>,
    ) -> Result<Self> {
        cfg.validate()?;
        let hidden_sq = cfg.hidden * cfg.hidden;
        let hidden_inter = cfg.hidden * cfg.intermediate;
        let inter_hidden = cfg.intermediate * cfg.hidden;
        if w_attn_norm.len() != cfg.hidden {
            return Err(anyhow!(
                "w_attn_norm length {} != hidden {}",
                w_attn_norm.len(),
                cfg.hidden
            ));
        }
        if w_ffn_norm.len() != cfg.hidden {
            return Err(anyhow!(
                "w_ffn_norm length {} != hidden {}",
                w_ffn_norm.len(),
                cfg.hidden
            ));
        }
        for (label, w, expected) in [
            ("w_q", &w_q_unscaled, hidden_sq),
            ("w_k", &w_k, hidden_sq),
            ("w_v", &w_v, hidden_sq),
            ("w_o", &w_o, hidden_sq),
            ("w_gate", &w_gate, hidden_inter),
            ("w_up", &w_up, hidden_inter),
            ("w_down", &w_down, inter_hidden),
        ] {
            if w.len() != expected {
                return Err(anyhow!("{label} length {} != expected {expected}", w.len()));
            }
        }
        let scale = 1.0_f32 / (cfg.head_dim as f32).sqrt();
        let w_q = w_q_unscaled.iter().map(|v| v * scale).collect();
        Ok(Self {
            w_attn_norm,
            w_q,
            w_k,
            w_v,
            w_o,
            w_ffn_norm,
            w_gate,
            w_up,
            w_down,
        })
    }
}

/// Tape leaves for the nine layer weights.
#[derive(Clone)]
pub struct LayerLeaves {
    pub w_attn_norm: GpuTensor,
    pub w_q: GpuTensor,
    pub w_k: GpuTensor,
    pub w_v: GpuTensor,
    pub w_o: GpuTensor,
    pub w_ffn_norm: GpuTensor,
    pub w_gate: GpuTensor,
    pub w_up: GpuTensor,
    pub w_down: GpuTensor,
}

impl LayerLeaves {
    pub fn from_weights(tape: &GpuTape, cfg: &LayerConfig, w: &LayerWeights) -> Result<Self> {
        Ok(Self {
            w_attn_norm: GpuTensor::from_vec(tape, &w.w_attn_norm, vec![cfg.hidden])?,
            w_q: GpuTensor::from_vec(tape, &w.w_q, vec![cfg.hidden, cfg.hidden])?,
            w_k: GpuTensor::from_vec(tape, &w.w_k, vec![cfg.hidden, cfg.hidden])?,
            w_v: GpuTensor::from_vec(tape, &w.w_v, vec![cfg.hidden, cfg.hidden])?,
            w_o: GpuTensor::from_vec(tape, &w.w_o, vec![cfg.hidden, cfg.hidden])?,
            w_ffn_norm: GpuTensor::from_vec(tape, &w.w_ffn_norm, vec![cfg.hidden])?,
            w_gate: GpuTensor::from_vec(tape, &w.w_gate, vec![cfg.hidden, cfg.intermediate])?,
            w_up: GpuTensor::from_vec(tape, &w.w_up, vec![cfg.hidden, cfg.intermediate])?,
            w_down: GpuTensor::from_vec(tape, &w.w_down, vec![cfg.intermediate, cfg.hidden])?,
        })
    }

    /// Same as [`from_weights`] but applies `qdq_fn` to the seven
    /// quantizable Linears (Q/K/V/O + FFN gate/up/down).  RMSNorm
    /// weights land verbatim (frozen, not quantized).
    pub fn from_weights_qdq<F>(
        tape: &GpuTape,
        cfg: &LayerConfig,
        w: &LayerWeights,
        qdq_fn: F,
    ) -> Result<Self>
    where
        F: Fn(&[f32]) -> Result<Vec<f32>>,
    {
        let w_q_q = qdq_fn(&w.w_q)?;
        let w_k_q = qdq_fn(&w.w_k)?;
        let w_v_q = qdq_fn(&w.w_v)?;
        let w_o_q = qdq_fn(&w.w_o)?;
        let w_gate_q = qdq_fn(&w.w_gate)?;
        let w_up_q = qdq_fn(&w.w_up)?;
        let w_down_q = qdq_fn(&w.w_down)?;
        Ok(Self {
            w_attn_norm: GpuTensor::from_vec(tape, &w.w_attn_norm, vec![cfg.hidden])?,
            w_q: GpuTensor::from_vec(tape, &w_q_q, vec![cfg.hidden, cfg.hidden])?,
            w_k: GpuTensor::from_vec(tape, &w_k_q, vec![cfg.hidden, cfg.hidden])?,
            w_v: GpuTensor::from_vec(tape, &w_v_q, vec![cfg.hidden, cfg.hidden])?,
            w_o: GpuTensor::from_vec(tape, &w_o_q, vec![cfg.hidden, cfg.hidden])?,
            w_ffn_norm: GpuTensor::from_vec(tape, &w.w_ffn_norm, vec![cfg.hidden])?,
            w_gate: GpuTensor::from_vec(tape, &w_gate_q, vec![cfg.hidden, cfg.intermediate])?,
            w_up: GpuTensor::from_vec(tape, &w_up_q, vec![cfg.hidden, cfg.intermediate])?,
            w_down: GpuTensor::from_vec(tape, &w_down_q, vec![cfg.intermediate, cfg.hidden])?,
        })
    }

    /// Build an [`FfnLeaves`] view over this layer's FFN tensors
    /// (cheap clone — `GpuTensor` is Rc-shared).
    fn ffn_leaves(&self) -> FfnLeaves {
        FfnLeaves {
            w_gate: self.w_gate.clone(),
            w_up: self.w_up.clone(),
            w_down: self.w_down.clone(),
        }
    }
}

/// Forward one full transformer layer on the tape:
///
/// `y_final = (x + attn(rms_norm(x))) + ffn(rms_norm(x + attn(...)))`
///
/// Output shape: `[batch, hidden]` (matches input).
pub fn forward(cfg: &LayerConfig, x: &GpuTensor, leaves: &LayerLeaves) -> Result<GpuTensor> {
    cfg.validate()?;
    if x.shape() != [cfg.batch, cfg.hidden] {
        return Err(anyhow!(
            "layer forward: x shape {:?} != [batch={}, hidden={}]",
            x.shape(),
            cfg.batch,
            cfg.hidden
        ));
    }
    // Attention sub-block (pre-norm).
    let normed_attn = rms_norm(x, &leaves.w_attn_norm, cfg.eps)?;
    let q = matmul(&normed_attn, &leaves.w_q)?;
    let k = matmul(&normed_attn, &leaves.w_k)?;
    let v = matmul(&normed_attn, &leaves.w_v)?;
    let context = multi_head_sdpa(&q, &k, &v, cfg.n_heads, cfg.head_dim)?;
    let attn_out = matmul(&context, &leaves.w_o)?;
    let y_attn = add(x, &attn_out)?;

    // FFN sub-block (pre-norm).
    let normed_ffn = rms_norm(&y_attn, &leaves.w_ffn_norm, cfg.eps)?;
    let ffn_cfg = FfnConfig {
        batch: cfg.batch,
        hidden: cfg.hidden,
        intermediate: cfg.intermediate,
    };
    let ffn_leaves = leaves.ffn_leaves();
    let ffn_out = qwen35_ffn::forward(&ffn_cfg, &normed_ffn, &ffn_leaves)?;
    let y_final = add(&y_attn, &ffn_out)?;
    Ok(y_final)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::autograd_gpu_tape::{backward, ones_like};
    use crate::calibrate::qdq_gpu::{qdq_q4_0_gpu, qdq_q8_0_gpu};
    use mlx_native::MlxDevice;

    fn deterministic_weights(cfg: &LayerConfig, seed: u64) -> LayerWeights {
        let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        let mut next = || {
            state ^= state >> 33;
            state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
            state ^= state >> 33;
            ((state as i64) as f32) / (i64::MAX as f32)
        };
        let hidden_sq = cfg.hidden * cfg.hidden;
        let hidden_inter = cfg.hidden * cfg.intermediate;
        let inter_hidden = cfg.intermediate * cfg.hidden;
        let w_attn_norm: Vec<f32> = (0..cfg.hidden).map(|_| 1.0 + next() * 0.05).collect();
        let w_ffn_norm: Vec<f32> = (0..cfg.hidden).map(|_| 1.0 + next() * 0.05).collect();
        let w_q: Vec<f32> = (0..hidden_sq).map(|_| next() * 0.4).collect();
        let w_k: Vec<f32> = (0..hidden_sq).map(|_| next() * 0.4).collect();
        let w_v: Vec<f32> = (0..hidden_sq).map(|_| next() * 0.4).collect();
        let w_o: Vec<f32> = (0..hidden_sq).map(|_| next() * 0.4).collect();
        let w_gate: Vec<f32> = (0..hidden_inter).map(|_| next() * 0.4).collect();
        let w_up: Vec<f32> = (0..hidden_inter).map(|_| next() * 0.4).collect();
        let w_down: Vec<f32> = (0..inter_hidden).map(|_| next() * 0.4).collect();
        LayerWeights::new(
            cfg,
            w_attn_norm,
            w_q,
            w_k,
            w_v,
            w_o,
            w_ffn_norm,
            w_gate,
            w_up,
            w_down,
        )
        .unwrap()
    }

    fn smallest_cfg() -> LayerConfig {
        LayerConfig {
            batch: 32,
            hidden: 64,
            n_heads: 2,
            head_dim: 32,
            intermediate: 128,
            eps: 1e-6,
        }
    }

    #[test]
    fn layer_forward_shape_and_finite() {
        let cfg = smallest_cfg();
        let weights = deterministic_weights(&cfg, 9001);
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let x: Vec<f32> = (0..cfg.batch * cfg.hidden)
            .map(|i| (i as f32 * 0.011).sin() * 0.3)
            .collect();
        let xt = GpuTensor::from_vec(&tape, &x, vec![cfg.batch, cfg.hidden]).unwrap();
        let leaves = LayerLeaves::from_weights(&tape, &cfg, &weights).unwrap();
        let y = forward(&cfg, &xt, &leaves).unwrap();
        assert_eq!(y.shape(), [cfg.batch, cfg.hidden]);
        let y_vec: Vec<f32> = y.to_vec().unwrap();
        for (i, v) in y_vec.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] = {v} not finite");
        }
    }

    #[test]
    fn layer_backward_flows_to_all_nine_leaves() {
        // Backward must produce gradients on EACH of:
        //   x (input residual contribution)
        //   w_attn_norm, w_ffn_norm (frozen, but still differentiable)
        //   W_q, W_k, W_v, W_o (4 quantizable attention)
        //   W_gate, W_up, W_down (3 quantizable FFN)
        // Total = 10 leaves (x + 9 layer weights).
        let cfg = smallest_cfg();
        let weights = deterministic_weights(&cfg, 31337);
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let x: Vec<f32> = (0..cfg.batch * cfg.hidden)
            .map(|i| (i as f32 * 0.013).cos() * 0.3)
            .collect();
        let xt = GpuTensor::from_vec(&tape, &x, vec![cfg.batch, cfg.hidden]).unwrap();
        let leaves = LayerLeaves::from_weights(&tape, &cfg, &weights).unwrap();
        let y = forward(&cfg, &xt, &leaves).unwrap();
        let dy = ones_like(&tape, &[cfg.batch, cfg.hidden]).unwrap();
        let grads = backward(&y, dy).unwrap();

        let leaves_to_check: [(&str, &GpuTensor, usize); 10] = [
            ("x", &xt, cfg.batch * cfg.hidden),
            ("w_attn_norm", &leaves.w_attn_norm, cfg.hidden),
            ("w_q", &leaves.w_q, cfg.hidden * cfg.hidden),
            ("w_k", &leaves.w_k, cfg.hidden * cfg.hidden),
            ("w_v", &leaves.w_v, cfg.hidden * cfg.hidden),
            ("w_o", &leaves.w_o, cfg.hidden * cfg.hidden),
            ("w_ffn_norm", &leaves.w_ffn_norm, cfg.hidden),
            ("w_gate", &leaves.w_gate, cfg.hidden * cfg.intermediate),
            ("w_up", &leaves.w_up, cfg.hidden * cfg.intermediate),
            ("w_down", &leaves.w_down, cfg.intermediate * cfg.hidden),
        ];
        for (label, leaf, expected_numel) in &leaves_to_check {
            let g = grads[leaf.node_idx()]
                .as_ref()
                .unwrap_or_else(|| panic!("{label} grad missing"));
            assert_eq!(
                g.element_count(),
                *expected_numel,
                "{label}: grad numel {} != expected {expected_numel}",
                g.element_count()
            );
            let g_vec: Vec<f32> = g.as_slice::<f32>().unwrap().to_vec();
            for (i, v) in g_vec.iter().enumerate() {
                let val: f32 = *v;
                assert!(val.is_finite(), "{label} grad[{i}] = {val} not finite");
            }
        }
    }

    #[test]
    fn layer_estimate_sensitivities_runs_for_all_seven_quantizable_leaves() {
        // End-to-end: build full-precision teacher + qdq-Q4_0 student
        // forward; run estimate_sensitivities over all 7 quantizable
        // weights (W_q/W_k/W_v/W_o/W_gate/W_up/W_down).  All scalars
        // must be finite + non-zero.
        use crate::calibrate::dynamic_quant_gpu::{
            estimate_sensitivities, QuantizableInput,
        };
        let cfg = smallest_cfg();
        let weights = deterministic_weights(&cfg, 4242);
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let x: Vec<f32> = (0..cfg.batch * cfg.hidden)
            .map(|i| (i as f32 * 0.0091).sin() * 0.35)
            .collect();
        let xt = GpuTensor::from_vec(&tape, &x, vec![cfg.batch, cfg.hidden]).unwrap();

        let teacher_leaves = LayerLeaves::from_weights(&tape, &cfg, &weights).unwrap();
        let teacher_out = forward(&cfg, &xt, &teacher_leaves).unwrap();

        let student_leaves =
            LayerLeaves::from_weights_qdq(&tape, &cfg, &weights, |w| qdq_q4_0_gpu(tape.device(), w))
                .unwrap();
        let student_out = forward(&cfg, &xt, &student_leaves).unwrap();

        let quantizables = vec![
            QuantizableInput {
                path: "W_q".to_string(),
                w_low: student_leaves.w_q.clone(),
                w_high_values: weights.w_q.clone(),
            },
            QuantizableInput {
                path: "W_k".to_string(),
                w_low: student_leaves.w_k.clone(),
                w_high_values: weights.w_k.clone(),
            },
            QuantizableInput {
                path: "W_v".to_string(),
                w_low: student_leaves.w_v.clone(),
                w_high_values: weights.w_v.clone(),
            },
            QuantizableInput {
                path: "W_o".to_string(),
                w_low: student_leaves.w_o.clone(),
                w_high_values: weights.w_o.clone(),
            },
            QuantizableInput {
                path: "W_gate".to_string(),
                w_low: student_leaves.w_gate.clone(),
                w_high_values: weights.w_gate.clone(),
            },
            QuantizableInput {
                path: "W_up".to_string(),
                w_low: student_leaves.w_up.clone(),
                w_high_values: weights.w_up.clone(),
            },
            QuantizableInput {
                path: "W_down".to_string(),
                w_low: student_leaves.w_down.clone(),
                w_high_values: weights.w_down.clone(),
            },
        ];

        let sens =
            estimate_sensitivities(&tape, &student_out, &teacher_out, &quantizables).unwrap();
        for k in &[
            "W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down",
        ] {
            let s = sens
                .get(*k)
                .copied()
                .unwrap_or_else(|| panic!("{k} sensitivity missing"));
            assert!(s.is_finite(), "{k} sensitivity {s} not finite");
            assert!(
                s.abs() > 1e-12,
                "{k} sensitivity {s} too small — autograd chain may be degenerate"
            );
        }
    }

    #[test]
    fn layer_q4_vs_q8_sensitivity_monotonicity() {
        // The Q4_0 (coarser) student should produce |sensitivity|
        // ≥ |sensitivity| under Q8_0 (finer) on the majority of
        // projections — same monotonicity check the iter-10c
        // attention block test pins, extended to all 7 quantizable
        // Linears in the layer.
        use crate::calibrate::dynamic_quant_gpu::{
            estimate_sensitivities, QuantizableInput,
        };
        let cfg = smallest_cfg();
        let weights = deterministic_weights(&cfg, 7777);

        let run = |qdq_fn: &dyn Fn(&[f32]) -> Result<Vec<f32>>| {
            let device = MlxDevice::new().expect("device");
            let tape = GpuTape::new(device);
            let x: Vec<f32> = (0..cfg.batch * cfg.hidden)
                .map(|i| (i as f32 * 0.011).cos() * 0.3)
                .collect();
            let xt =
                GpuTensor::from_vec(&tape, &x, vec![cfg.batch, cfg.hidden]).unwrap();
            let teacher_leaves = LayerLeaves::from_weights(&tape, &cfg, &weights).unwrap();
            let teacher_out = forward(&cfg, &xt, &teacher_leaves).unwrap();
            let student_leaves =
                LayerLeaves::from_weights_qdq(&tape, &cfg, &weights, qdq_fn).unwrap();
            let student_out = forward(&cfg, &xt, &student_leaves).unwrap();
            let quantizables = vec![
                QuantizableInput {
                    path: "W_q".into(),
                    w_low: student_leaves.w_q.clone(),
                    w_high_values: weights.w_q.clone(),
                },
                QuantizableInput {
                    path: "W_k".into(),
                    w_low: student_leaves.w_k.clone(),
                    w_high_values: weights.w_k.clone(),
                },
                QuantizableInput {
                    path: "W_v".into(),
                    w_low: student_leaves.w_v.clone(),
                    w_high_values: weights.w_v.clone(),
                },
                QuantizableInput {
                    path: "W_o".into(),
                    w_low: student_leaves.w_o.clone(),
                    w_high_values: weights.w_o.clone(),
                },
                QuantizableInput {
                    path: "W_gate".into(),
                    w_low: student_leaves.w_gate.clone(),
                    w_high_values: weights.w_gate.clone(),
                },
                QuantizableInput {
                    path: "W_up".into(),
                    w_low: student_leaves.w_up.clone(),
                    w_high_values: weights.w_up.clone(),
                },
                QuantizableInput {
                    path: "W_down".into(),
                    w_low: student_leaves.w_down.clone(),
                    w_high_values: weights.w_down.clone(),
                },
            ];
            estimate_sensitivities(&tape, &student_out, &teacher_out, &quantizables).unwrap()
        };

        let device_q4 = MlxDevice::new().expect("device q4 closure");
        let device_q8 = MlxDevice::new().expect("device q8 closure");
        let s4 = run(&|w| qdq_q4_0_gpu(&device_q4, w));
        let s8 = run(&|w| qdq_q8_0_gpu(&device_q8, w));

        let keys = ["W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"];
        let mut q4_larger = 0;
        for k in &keys {
            let a = s4[*k].abs();
            let b = s8[*k].abs();
            // 0.95 slack — Q8_0 grid is ~16x finer per-element so
            // |Q4_0 sens| should dominate with substantial margin.
            if a >= 0.95 * b {
                q4_larger += 1;
            }
        }
        assert!(
            q4_larger >= 5,
            "Q4_0-vs-Q8_0 monotonicity failed: only {q4_larger}/7 projections have \
             |sens(Q4_0)| ≥ 0.95·|sens(Q8_0)|.\n  s4={s4:?}\n  s8={s8:?}"
        );
    }

    #[test]
    fn layer_config_validates() {
        let mut cfg = smallest_cfg();
        cfg.n_heads = 3;
        // 3 * 32 = 96 ≠ 64
        let weights = deterministic_weights(&smallest_cfg(), 1);
        let result = LayerWeights::new(
            &cfg,
            weights.w_attn_norm.clone(),
            weights.w_q.clone(),
            weights.w_k.clone(),
            weights.w_v.clone(),
            weights.w_o.clone(),
            weights.w_ffn_norm.clone(),
            weights.w_gate.clone(),
            weights.w_up.clone(),
            weights.w_down.clone(),
        );
        match result {
            Err(e) => assert!(format!("{e}").contains("n_heads(3)")),
            Ok(_) => panic!("expected n_heads*head_dim != hidden error"),
        }
    }
}
