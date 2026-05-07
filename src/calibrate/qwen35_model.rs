//! Qwen 3.5 N-layer model — GpuTape composition for ADR-020 iter-11d.
//!
//! Composes the full text-generation forward pass:
//!
//! ```text
//!   x_0     = embedding(E, ids)          [batch, hidden]
//!   x_{i+1} = layer_i.forward(x_i)       for i ∈ [0, n_layers)
//!   y_norm  = rms_norm(x_n, w_final, eps)
//!   logits  = matmul(y_norm, W_lm_head)  [batch, vocab]
//! ```
//!
//! Quantizable Linears per model:
//!   N × 7 (Q/K/V/O + FFN gate/up/down per layer) + 1 (lm_head) = 7N + 1.
//!
//! RMSNorm weights (per-layer × 2 + final = 2N + 1) and the embedding
//! table are NOT quantized — they're frozen leaves on the tape.

use anyhow::{anyhow, Result};

use crate::calibrate::autograd_gpu_tape::{
    embedding, matmul, rms_norm, GpuTape, GpuTensor,
};
use crate::calibrate::qwen35_layer::{self, LayerConfig, LayerLeaves, LayerWeights};

/// N-layer model configuration.  All Linear dims must be ≥ 32.
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub eps: f32,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        if self.n_layers == 0 {
            return Err(anyhow!("ModelConfig: n_layers must be > 0"));
        }
        if self.vocab < 32 {
            return Err(anyhow!(
                "ModelConfig: vocab={} must be ≥ 32 (lm_head matmul backward requires k ≥ 32)",
                self.vocab
            ));
        }
        // Reuse LayerConfig validation for the per-layer dims.
        let layer_cfg = self.layer_config(/* batch */ 32);
        layer_cfg.validate()?;
        Ok(())
    }

    pub fn layer_config(&self, batch: usize) -> LayerConfig {
        LayerConfig {
            batch,
            hidden: self.hidden,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            intermediate: self.intermediate,
            eps: self.eps,
        }
    }
}

/// Owned weights for an N-layer model.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// `[vocab, hidden]` — token embedding table.
    pub embedding: Vec<f32>,
    /// One [`LayerWeights`] per transformer layer.
    pub layers: Vec<LayerWeights>,
    /// `[hidden]` — final RMSNorm weight.
    pub final_norm: Vec<f32>,
    /// `[hidden, vocab]` — language-model head projection.
    pub lm_head: Vec<f32>,
}

impl ModelWeights {
    pub fn new(
        cfg: &ModelConfig,
        embedding: Vec<f32>,
        layers: Vec<LayerWeights>,
        final_norm: Vec<f32>,
        lm_head: Vec<f32>,
    ) -> Result<Self> {
        cfg.validate()?;
        if embedding.len() != cfg.vocab * cfg.hidden {
            return Err(anyhow!(
                "embedding length {} != vocab*hidden {}",
                embedding.len(),
                cfg.vocab * cfg.hidden
            ));
        }
        if layers.len() != cfg.n_layers {
            return Err(anyhow!(
                "layers count {} != cfg.n_layers {}",
                layers.len(),
                cfg.n_layers
            ));
        }
        if final_norm.len() != cfg.hidden {
            return Err(anyhow!(
                "final_norm length {} != hidden {}",
                final_norm.len(),
                cfg.hidden
            ));
        }
        if lm_head.len() != cfg.hidden * cfg.vocab {
            return Err(anyhow!(
                "lm_head length {} != hidden*vocab {}",
                lm_head.len(),
                cfg.hidden * cfg.vocab
            ));
        }
        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
        })
    }
}

/// Tape leaves for an N-layer model.
#[derive(Clone)]
pub struct ModelLeaves {
    pub embedding: GpuTensor,
    pub layers: Vec<LayerLeaves>,
    pub final_norm: GpuTensor,
    pub lm_head: GpuTensor,
}

impl ModelLeaves {
    pub fn from_weights(
        tape: &GpuTape,
        cfg: &ModelConfig,
        w: &ModelWeights,
        batch: usize,
    ) -> Result<Self> {
        let layer_cfg = cfg.layer_config(batch);
        let embedding = GpuTensor::from_vec(tape, &w.embedding, vec![cfg.vocab, cfg.hidden])?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for lw in &w.layers {
            layers.push(LayerLeaves::from_weights(tape, &layer_cfg, lw)?);
        }
        let final_norm = GpuTensor::from_vec(tape, &w.final_norm, vec![cfg.hidden])?;
        let lm_head = GpuTensor::from_vec(tape, &w.lm_head, vec![cfg.hidden, cfg.vocab])?;
        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
        })
    }

    /// Apply `qdq_fn` to all 7N + 1 quantizable Linears (per-layer
    /// Q/K/V/O + FFN + lm_head).  Embedding table and RMSNorm weights
    /// land verbatim (frozen).
    pub fn from_weights_qdq<F>(
        tape: &GpuTape,
        cfg: &ModelConfig,
        w: &ModelWeights,
        batch: usize,
        qdq_fn: F,
    ) -> Result<Self>
    where
        F: Fn(&[f32]) -> Result<Vec<f32>>,
    {
        let layer_cfg = cfg.layer_config(batch);
        let embedding = GpuTensor::from_vec(tape, &w.embedding, vec![cfg.vocab, cfg.hidden])?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for lw in &w.layers {
            layers.push(LayerLeaves::from_weights_qdq(tape, &layer_cfg, lw, &qdq_fn)?);
        }
        let final_norm = GpuTensor::from_vec(tape, &w.final_norm, vec![cfg.hidden])?;
        let lm_head_q = qdq_fn(&w.lm_head)?;
        let lm_head = GpuTensor::from_vec(tape, &lm_head_q, vec![cfg.hidden, cfg.vocab])?;
        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
        })
    }
}

/// Forward the full N-layer model on the tape.
///
/// Output: `[batch, vocab]` logits (no softmax).
pub fn forward(
    cfg: &ModelConfig,
    ids: &[u32],
    leaves: &ModelLeaves,
) -> Result<GpuTensor> {
    cfg.validate()?;
    if ids.is_empty() {
        return Err(anyhow!("model forward: ids must not be empty"));
    }
    let batch = ids.len();
    let layer_cfg = cfg.layer_config(batch);

    // Embedding lookup → [batch, hidden].
    let mut x = embedding(&leaves.embedding, ids)?;

    // Layer stack.
    for (i, layer_leaves) in leaves.layers.iter().enumerate() {
        x = qwen35_layer::forward(&layer_cfg, &x, layer_leaves)
            .map_err(|e| anyhow!("model forward: layer {i}: {e}"))?;
    }

    // Final RMSNorm → [batch, hidden].
    let normed = rms_norm(&x, &leaves.final_norm, cfg.eps)?;

    // lm_head: [batch, hidden] @ [hidden, vocab] → [batch, vocab].
    let logits = matmul(&normed, &leaves.lm_head)?;
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::autograd_gpu_tape::{backward, ones_like};
    use crate::calibrate::qdq_gpu::{qdq_q4_0_gpu, qdq_q8_0_gpu};
    use mlx_native::MlxDevice;

    fn deterministic_layer_weights(cfg: &LayerConfig, seed: u64) -> LayerWeights {
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
        LayerWeights::new(
            cfg,
            (0..cfg.hidden).map(|_| 1.0 + next() * 0.05).collect(),
            (0..hidden_sq).map(|_| next() * 0.3).collect(),
            (0..hidden_sq).map(|_| next() * 0.3).collect(),
            (0..hidden_sq).map(|_| next() * 0.3).collect(),
            (0..hidden_sq).map(|_| next() * 0.3).collect(),
            (0..cfg.hidden).map(|_| 1.0 + next() * 0.05).collect(),
            (0..hidden_inter).map(|_| next() * 0.3).collect(),
            (0..hidden_inter).map(|_| next() * 0.3).collect(),
            (0..inter_hidden).map(|_| next() * 0.3).collect(),
        )
        .unwrap()
    }

    fn deterministic_model_weights(cfg: &ModelConfig, batch: usize, seed: u64) -> ModelWeights {
        let layer_cfg = cfg.layer_config(batch);
        let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        let mut next = || {
            state ^= state >> 33;
            state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
            state ^= state >> 33;
            ((state as i64) as f32) / (i64::MAX as f32)
        };
        let embedding_vec: Vec<f32> = (0..cfg.vocab * cfg.hidden)
            .map(|_| next() * 0.2)
            .collect();
        let layers: Vec<LayerWeights> = (0..cfg.n_layers)
            .map(|i| deterministic_layer_weights(&layer_cfg, seed.wrapping_add(i as u64 * 17)))
            .collect();
        let final_norm: Vec<f32> = (0..cfg.hidden).map(|_| 1.0 + next() * 0.03).collect();
        let lm_head: Vec<f32> = (0..cfg.hidden * cfg.vocab)
            .map(|_| next() * 0.2)
            .collect();
        ModelWeights::new(cfg, embedding_vec, layers, final_norm, lm_head).unwrap()
    }

    fn smallest_cfg() -> ModelConfig {
        ModelConfig {
            vocab: 32,
            hidden: 64,
            n_layers: 2,
            n_heads: 2,
            head_dim: 32,
            intermediate: 128,
            eps: 1e-6,
        }
    }

    #[test]
    fn model_forward_shape_and_finite() {
        let cfg = smallest_cfg();
        let batch = 32usize;
        let weights = deterministic_model_weights(&cfg, batch, 9001);
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let leaves = ModelLeaves::from_weights(&tape, &cfg, &weights, batch).unwrap();
        let ids: Vec<u32> = (0..batch).map(|i| (i * 3 + 1) as u32 % cfg.vocab as u32).collect();
        let logits = forward(&cfg, &ids, &leaves).unwrap();
        assert_eq!(logits.shape(), [batch, cfg.vocab]);
        let l_vec: Vec<f32> = logits.to_vec().unwrap();
        for (i, v) in l_vec.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    #[test]
    fn model_backward_flows_to_all_leaves() {
        // Backward must produce gradients on:
        //   embedding (1)
        //   per-layer 9 weights × n_layers (= 18 for 2 layers)
        //   final_norm (1)
        //   lm_head (1)
        // = 21 leaves total for n_layers=2.  Plus implicit ones on
        // intermediate tensors but those aren't reachable as user
        // leaves.
        let cfg = smallest_cfg();
        let batch = 32usize;
        let weights = deterministic_model_weights(&cfg, batch, 31337);
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let leaves = ModelLeaves::from_weights(&tape, &cfg, &weights, batch).unwrap();
        let ids: Vec<u32> = (0..batch).map(|i| (i * 5 + 2) as u32 % cfg.vocab as u32).collect();
        let logits = forward(&cfg, &ids, &leaves).unwrap();
        let dy = ones_like(&tape, &[batch, cfg.vocab]).unwrap();
        let grads = backward(&logits, dy).unwrap();

        // Embedding grad.
        let de = grads[leaves.embedding.node_idx()]
            .as_ref()
            .expect("embedding grad missing");
        assert_eq!(de.element_count(), cfg.vocab * cfg.hidden);
        // final_norm grad.
        let dfn = grads[leaves.final_norm.node_idx()]
            .as_ref()
            .expect("final_norm grad missing");
        assert_eq!(dfn.element_count(), cfg.hidden);
        // lm_head grad.
        let dlmh = grads[leaves.lm_head.node_idx()]
            .as_ref()
            .expect("lm_head grad missing");
        assert_eq!(dlmh.element_count(), cfg.hidden * cfg.vocab);

        // Per-layer 9 weight grads.
        for (i, layer_leaves) in leaves.layers.iter().enumerate() {
            for (label, leaf, expected) in [
                ("w_attn_norm", &layer_leaves.w_attn_norm, cfg.hidden),
                ("w_q", &layer_leaves.w_q, cfg.hidden * cfg.hidden),
                ("w_k", &layer_leaves.w_k, cfg.hidden * cfg.hidden),
                ("w_v", &layer_leaves.w_v, cfg.hidden * cfg.hidden),
                ("w_o", &layer_leaves.w_o, cfg.hidden * cfg.hidden),
                ("w_ffn_norm", &layer_leaves.w_ffn_norm, cfg.hidden),
                ("w_gate", &layer_leaves.w_gate, cfg.hidden * cfg.intermediate),
                ("w_up", &layer_leaves.w_up, cfg.hidden * cfg.intermediate),
                ("w_down", &layer_leaves.w_down, cfg.intermediate * cfg.hidden),
            ] {
                let g = grads[leaf.node_idx()]
                    .as_ref()
                    .unwrap_or_else(|| panic!("layer {i} {label} grad missing"));
                assert_eq!(
                    g.element_count(),
                    expected,
                    "layer {i} {label}: numel {} != {expected}",
                    g.element_count()
                );
            }
        }

        // All grads must be finite (sample a few).
        let de_vec: Vec<f32> = de.as_slice::<f32>().unwrap().to_vec();
        for (i, v) in de_vec.iter().enumerate().take(64) {
            let val: f32 = *v;
            assert!(val.is_finite(), "embedding grad[{i}] = {val} not finite");
        }
        let dlmh_vec: Vec<f32> = dlmh.as_slice::<f32>().unwrap().to_vec();
        for (i, v) in dlmh_vec.iter().enumerate().take(64) {
            let val: f32 = *v;
            assert!(val.is_finite(), "lm_head grad[{i}] = {val} not finite");
        }
    }

    #[test]
    fn model_estimate_sensitivities_runs_for_all_quantizable_linears() {
        // 2 layers × 7 quantizable Linears + 1 lm_head = 15 total.
        // End-to-end: full-precision teacher vs Q4_0 student;
        // estimate_sensitivities returns 15 finite + non-zero scalars.
        use crate::calibrate::dynamic_quant_gpu::{
            estimate_sensitivities, QuantizableInput,
        };
        let cfg = smallest_cfg();
        let batch = 32usize;
        let weights = deterministic_model_weights(&cfg, batch, 4242);
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let teacher_leaves = ModelLeaves::from_weights(&tape, &cfg, &weights, batch).unwrap();
        let ids: Vec<u32> = (0..batch).map(|i| (i * 7 + 3) as u32 % cfg.vocab as u32).collect();
        let teacher_logits = forward(&cfg, &ids, &teacher_leaves).unwrap();

        let student_leaves = ModelLeaves::from_weights_qdq(
            &tape,
            &cfg,
            &weights,
            batch,
            |w| qdq_q4_0_gpu(tape.device(), w),
        )
        .unwrap();
        let student_logits = forward(&cfg, &ids, &student_leaves).unwrap();

        let mut quantizables: Vec<QuantizableInput> = Vec::new();
        for (i, (lo, hi)) in student_leaves
            .layers
            .iter()
            .zip(weights.layers.iter())
            .enumerate()
        {
            for (label, low, high) in [
                ("W_q", &lo.w_q, &hi.w_q),
                ("W_k", &lo.w_k, &hi.w_k),
                ("W_v", &lo.w_v, &hi.w_v),
                ("W_o", &lo.w_o, &hi.w_o),
                ("W_gate", &lo.w_gate, &hi.w_gate),
                ("W_up", &lo.w_up, &hi.w_up),
                ("W_down", &lo.w_down, &hi.w_down),
            ] {
                quantizables.push(QuantizableInput {
                    path: format!("layer{i}.{label}"),
                    w_low: low.clone(),
                    w_high_values: high.clone(),
                });
            }
        }
        quantizables.push(QuantizableInput {
            path: "lm_head".to_string(),
            w_low: student_leaves.lm_head.clone(),
            w_high_values: weights.lm_head.clone(),
        });
        assert_eq!(quantizables.len(), cfg.n_layers * 7 + 1);

        let sens =
            estimate_sensitivities(&tape, &student_logits, &teacher_logits, &quantizables)
                .unwrap();
        assert_eq!(sens.len(), quantizables.len());
        for (k, v) in &sens {
            assert!(v.is_finite(), "{k} sensitivity {v} not finite");
            assert!(v.abs() > 1e-12, "{k} sensitivity {v} too small (degenerate)");
        }
    }

    #[test]
    fn model_q4_vs_q8_sensitivity_monotonicity_majority() {
        // Q4_0 (coarser) → larger |sens| than Q8_0 in MAJORITY of
        // quantizable Linears (≥ 2/3).  Some flips are expected from
        // the signed gradient-alignment formula but bulk monotonicity
        // must hold.
        use crate::calibrate::dynamic_quant_gpu::{
            estimate_sensitivities, QuantizableInput,
        };
        let cfg = smallest_cfg();
        let batch = 32usize;
        let weights = deterministic_model_weights(&cfg, batch, 7777);

        let run = |qdq_fn: &dyn Fn(&[f32]) -> Result<Vec<f32>>| {
            let device = MlxDevice::new().expect("device");
            let tape = GpuTape::new(device);
            let teacher_leaves =
                ModelLeaves::from_weights(&tape, &cfg, &weights, batch).unwrap();
            let ids: Vec<u32> = (0..batch)
                .map(|i| (i * 11 + 5) as u32 % cfg.vocab as u32)
                .collect();
            let teacher_logits = forward(&cfg, &ids, &teacher_leaves).unwrap();
            let student_leaves = ModelLeaves::from_weights_qdq(
                &tape,
                &cfg,
                &weights,
                batch,
                qdq_fn,
            )
            .unwrap();
            let student_logits = forward(&cfg, &ids, &student_leaves).unwrap();
            let mut quantizables: Vec<QuantizableInput> = Vec::new();
            for (i, (lo, hi)) in student_leaves
                .layers
                .iter()
                .zip(weights.layers.iter())
                .enumerate()
            {
                for (label, low, high) in [
                    ("W_q", &lo.w_q, &hi.w_q),
                    ("W_k", &lo.w_k, &hi.w_k),
                    ("W_v", &lo.w_v, &hi.w_v),
                    ("W_o", &lo.w_o, &hi.w_o),
                    ("W_gate", &lo.w_gate, &hi.w_gate),
                    ("W_up", &lo.w_up, &hi.w_up),
                    ("W_down", &lo.w_down, &hi.w_down),
                ] {
                    quantizables.push(QuantizableInput {
                        path: format!("layer{i}.{label}"),
                        w_low: low.clone(),
                        w_high_values: high.clone(),
                    });
                }
            }
            quantizables.push(QuantizableInput {
                path: "lm_head".to_string(),
                w_low: student_leaves.lm_head.clone(),
                w_high_values: weights.lm_head.clone(),
            });
            estimate_sensitivities(&tape, &student_logits, &teacher_logits, &quantizables)
                .unwrap()
        };

        let device_q4 = MlxDevice::new().expect("device q4");
        let device_q8 = MlxDevice::new().expect("device q8");
        let s4 = run(&|w| qdq_q4_0_gpu(&device_q4, w));
        let s8 = run(&|w| qdq_q8_0_gpu(&device_q8, w));

        let total = s4.len();
        let mut q4_dominant = 0;
        for k in s4.keys() {
            let a = s4[k].abs();
            let b = s8[k].abs();
            if a >= 0.95 * b {
                q4_dominant += 1;
            }
        }
        let threshold = (total * 2 + 2) / 3; // ceil(2/3 of total)
        assert!(
            q4_dominant >= threshold,
            "Q4_0/Q8_0 monotonicity failed: only {q4_dominant}/{total} Linears \
             have |sens(Q4_0)| ≥ 0.95·|sens(Q8_0)| (threshold ≥ {threshold}).\n\
             s4={s4:?}\n\
             s8={s8:?}"
        );
    }

    #[test]
    fn model_validates_dims() {
        let mut cfg = smallest_cfg();
        cfg.vocab = 16; // < 32 → reject
        match cfg.validate() {
            Err(e) => assert!(format!("{e}").contains("vocab=16")),
            Ok(_) => panic!("expected vocab too small error"),
        }
    }
}
