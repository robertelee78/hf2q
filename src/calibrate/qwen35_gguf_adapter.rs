//! GGUF tensor-map → [`ModelWeights`] adapter (ADR-020 iter-11e).
//!
//! Maps the standard llama.cpp/GGUF Qwen3-style tensor naming to
//! hf2q's GpuTape model structure.  The transformation includes a
//! transpose for every Linear weight because GGUF stores weights
//! as `[out, in]` (PyTorch `nn.Linear.weight` convention) while our
//! matmul expects `[in, out]` (the `Y = X @ W` form, where W is
//! `[k, n]`).
//!
//! This adapter is pure-Rust + framework-free — it operates on a
//! `BTreeMap<String, Vec<f32>>` produced by the caller (typically
//! by reading + dequantizing a GGUF file via hf2q's existing GGUF
//! reader).
//!
//! Scope (iter-11e): standard multi-head attention only.  Grouped-
//! query attention (GQA, where `n_kv_heads < n_heads`) is deferred to
//! iter-11g when actual Qwen3-0.6B-base weights are loaded — that
//! variant has different K/V output dims (`n_kv_heads * head_dim`
//! instead of `n_heads * head_dim`) and needs the multi-head SDPA
//! to broadcast K/V across the head ratio.

use std::collections::BTreeMap;

use anyhow::{anyhow, Result};

use crate::calibrate::qwen35_layer::LayerWeights;
use crate::calibrate::qwen35_model::{ModelConfig, ModelWeights};

/// GGUF-style tensor name → fp32 row-major buffer.  Caller typically
/// builds this from a GGUF file by reading + dequantizing each
/// tensor.
pub type GgufTensorMap = BTreeMap<String, Vec<f32>>;

/// Build a [`ModelWeights`] from a GGUF tensor map.  See module docs
/// for the GGUF→matmul shape transpose convention.
///
/// Required keys (per cfg.n_layers):
/// - `token_embd.weight` — `[vocab, hidden]`
/// - `output.weight` — `[vocab, hidden]` (lm_head, GGUF [out, in])
/// - `output_norm.weight` — `[hidden]`
/// - `blk.{i}.attn_norm.weight` — `[hidden]`
/// - `blk.{i}.post_attention_norm.weight` — `[hidden]`
/// - `blk.{i}.attn_q.weight` — `[hidden, hidden]` (GGUF [out, in])
/// - `blk.{i}.attn_k.weight` — `[hidden, hidden]`
/// - `blk.{i}.attn_v.weight` — `[hidden, hidden]`
/// - `blk.{i}.attn_output.weight` — `[hidden, hidden]`
/// - `blk.{i}.ffn_gate.weight` — `[intermediate, hidden]`
/// - `blk.{i}.ffn_up.weight` — `[intermediate, hidden]`
/// - `blk.{i}.ffn_down.weight` — `[hidden, intermediate]`
///
/// Returns the converted ModelWeights with all 7N+1 Linear weights
/// transposed to the `[in, out]` matmul convention.
pub fn weights_from_gguf_tensors(
    cfg: &ModelConfig,
    tensors: &GgufTensorMap,
) -> Result<ModelWeights> {
    cfg.validate()?;

    // Embedding [vocab, hidden] — already row-major in
    // (vocab, hidden) order; no transpose.
    let embedding = pull_required(tensors, "token_embd.weight", cfg.vocab * cfg.hidden)?;

    // Final RMSNorm [hidden] — vector, no transpose.
    let final_norm = pull_required(tensors, "output_norm.weight", cfg.hidden)?;

    // lm_head: GGUF stores as [vocab, hidden] (output, input);
    // our matmul wants [hidden, vocab] (input, output).
    let lm_head_oi = pull_required(tensors, "output.weight", cfg.vocab * cfg.hidden)?;
    let lm_head = transpose_2d_row_major(&lm_head_oi, cfg.vocab, cfg.hidden);

    // Per-layer.
    let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let key = |s: &str| format!("blk.{i}.{s}");
        let w_attn_norm = pull_required(tensors, &key("attn_norm.weight"), cfg.hidden)?;
        let w_ffn_norm =
            pull_required(tensors, &key("post_attention_norm.weight"), cfg.hidden)?;

        // Attention projections: GGUF [hidden_out, hidden_in] → matmul [hidden, hidden].
        // For standard MHA hidden_out == hidden_in == cfg.hidden.
        let h_sq = cfg.hidden * cfg.hidden;
        let w_q_oi = pull_required(tensors, &key("attn_q.weight"), h_sq)?;
        let w_k_oi = pull_required(tensors, &key("attn_k.weight"), h_sq)?;
        let w_v_oi = pull_required(tensors, &key("attn_v.weight"), h_sq)?;
        let w_o_oi = pull_required(tensors, &key("attn_output.weight"), h_sq)?;
        let w_q = transpose_2d_row_major(&w_q_oi, cfg.hidden, cfg.hidden);
        let w_k = transpose_2d_row_major(&w_k_oi, cfg.hidden, cfg.hidden);
        let w_v = transpose_2d_row_major(&w_v_oi, cfg.hidden, cfg.hidden);
        let w_o = transpose_2d_row_major(&w_o_oi, cfg.hidden, cfg.hidden);

        // FFN projections.
        // ffn_gate / ffn_up: GGUF [intermediate, hidden] → matmul [hidden, intermediate].
        // ffn_down:           GGUF [hidden, intermediate] → matmul [intermediate, hidden].
        let h_inter = cfg.intermediate * cfg.hidden;
        let w_gate_oi = pull_required(tensors, &key("ffn_gate.weight"), h_inter)?;
        let w_up_oi = pull_required(tensors, &key("ffn_up.weight"), h_inter)?;
        let w_down_oi = pull_required(tensors, &key("ffn_down.weight"), h_inter)?;
        let w_gate =
            transpose_2d_row_major(&w_gate_oi, cfg.intermediate, cfg.hidden);
        let w_up = transpose_2d_row_major(&w_up_oi, cfg.intermediate, cfg.hidden);
        let w_down = transpose_2d_row_major(&w_down_oi, cfg.hidden, cfg.intermediate);

        // The layer constructor folds the 1/√head_dim scale into W_q
        // when given the *unscaled* weight.  GGUF stores the unscaled
        // version (the scaling is normally done at attention compute
        // time via a separate scalar).  So we pass the transposed
        // unscaled W_q.
        let lw = LayerWeights::new(
            &cfg.layer_config(/* batch dummy */ 32),
            w_attn_norm,
            w_q,
            w_k,
            w_v,
            w_o,
            w_ffn_norm,
            w_gate,
            w_up,
            w_down,
        )?;
        layers.push(lw);
    }

    ModelWeights::new(cfg, embedding, layers, final_norm, lm_head)
}

fn pull_required(tensors: &GgufTensorMap, key: &str, expected_len: usize) -> Result<Vec<f32>> {
    let v = tensors
        .get(key)
        .ok_or_else(|| anyhow!("GGUF tensor missing: {key}"))?;
    if v.len() != expected_len {
        return Err(anyhow!(
            "GGUF tensor {key} length {} != expected {expected_len}",
            v.len()
        ));
    }
    Ok(v.clone())
}

/// Transpose a 2D row-major tensor `src[rows, cols]` → `dst[cols, rows]`.
/// Pure-CPU; used only at load time.
fn transpose_2d_row_major(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut dst = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::autograd_gpu_tape::{GpuTape, GpuTensor};
    use crate::calibrate::qwen35_model::{forward, ModelLeaves};
    use mlx_native::MlxDevice;

    fn make_synthetic_gguf_map(cfg: &ModelConfig, seed: u64) -> GgufTensorMap {
        let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        let mut next = || {
            state ^= state >> 33;
            state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
            state ^= state >> 33;
            ((state as i64) as f32) / (i64::MAX as f32)
        };
        let mut m = GgufTensorMap::new();
        m.insert(
            "token_embd.weight".into(),
            (0..cfg.vocab * cfg.hidden).map(|_| next() * 0.2).collect(),
        );
        m.insert(
            "output.weight".into(),
            (0..cfg.vocab * cfg.hidden).map(|_| next() * 0.2).collect(),
        );
        m.insert(
            "output_norm.weight".into(),
            (0..cfg.hidden).map(|_| 1.0 + next() * 0.05).collect(),
        );
        for i in 0..cfg.n_layers {
            m.insert(
                format!("blk.{i}.attn_norm.weight"),
                (0..cfg.hidden).map(|_| 1.0 + next() * 0.05).collect(),
            );
            m.insert(
                format!("blk.{i}.post_attention_norm.weight"),
                (0..cfg.hidden).map(|_| 1.0 + next() * 0.05).collect(),
            );
            for name in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                m.insert(
                    format!("blk.{i}.{name}.weight"),
                    (0..cfg.hidden * cfg.hidden).map(|_| next() * 0.2).collect(),
                );
            }
            m.insert(
                format!("blk.{i}.ffn_gate.weight"),
                (0..cfg.intermediate * cfg.hidden).map(|_| next() * 0.2).collect(),
            );
            m.insert(
                format!("blk.{i}.ffn_up.weight"),
                (0..cfg.intermediate * cfg.hidden).map(|_| next() * 0.2).collect(),
            );
            m.insert(
                format!("blk.{i}.ffn_down.weight"),
                (0..cfg.hidden * cfg.intermediate).map(|_| next() * 0.2).collect(),
            );
        }
        m
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
    fn weights_from_gguf_constructs_valid_model() {
        let cfg = smallest_cfg();
        let map = make_synthetic_gguf_map(&cfg, 31337);
        let weights = weights_from_gguf_tensors(&cfg, &map).expect("convert");
        // Spot-check shapes.
        assert_eq!(weights.embedding.len(), cfg.vocab * cfg.hidden);
        assert_eq!(weights.lm_head.len(), cfg.hidden * cfg.vocab);
        assert_eq!(weights.final_norm.len(), cfg.hidden);
        assert_eq!(weights.layers.len(), cfg.n_layers);
        for lw in &weights.layers {
            assert_eq!(lw.w_attn_norm.len(), cfg.hidden);
            assert_eq!(lw.w_ffn_norm.len(), cfg.hidden);
            assert_eq!(lw.w_q.len(), cfg.hidden * cfg.hidden);
            assert_eq!(lw.w_k.len(), cfg.hidden * cfg.hidden);
            assert_eq!(lw.w_v.len(), cfg.hidden * cfg.hidden);
            assert_eq!(lw.w_o.len(), cfg.hidden * cfg.hidden);
            assert_eq!(lw.w_gate.len(), cfg.hidden * cfg.intermediate);
            assert_eq!(lw.w_up.len(), cfg.hidden * cfg.intermediate);
            assert_eq!(lw.w_down.len(), cfg.intermediate * cfg.hidden);
        }
    }

    #[test]
    fn weights_from_gguf_e2e_forward_finite() {
        // Build the ModelWeights from synthetic GGUF, place on tape,
        // run forward — output must be [batch, vocab] all finite.
        let cfg = smallest_cfg();
        let batch = 32usize;
        let map = make_synthetic_gguf_map(&cfg, 4242);
        let weights = weights_from_gguf_tensors(&cfg, &map).unwrap();
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let leaves = ModelLeaves::from_weights(&tape, &cfg, &weights, batch).unwrap();
        let ids: Vec<u32> = (0..batch)
            .map(|i| (i * 3 + 1) as u32 % cfg.vocab as u32)
            .collect();
        let logits = forward(&cfg, &ids, &leaves).unwrap();
        assert_eq!(logits.shape(), [batch, cfg.vocab]);
        let l_vec: Vec<f32> = logits.to_vec().unwrap();
        for (i, v) in l_vec.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    #[test]
    fn weights_from_gguf_missing_tensor_errors_with_clear_message() {
        let cfg = smallest_cfg();
        let mut map = make_synthetic_gguf_map(&cfg, 1);
        map.remove("blk.0.attn_v.weight");
        let err = weights_from_gguf_tensors(&cfg, &map).expect_err("expected missing-tensor error");
        let msg = format!("{err}");
        assert!(
            msg.contains("blk.0.attn_v.weight") && msg.contains("missing"),
            "wrong error message: {msg}"
        );
    }

    #[test]
    fn weights_from_gguf_wrong_shape_errors_with_clear_message() {
        let cfg = smallest_cfg();
        let mut map = make_synthetic_gguf_map(&cfg, 1);
        // Truncate output_norm to wrong length.
        map.insert("output_norm.weight".into(), vec![1.0; cfg.hidden - 1]);
        let err = weights_from_gguf_tensors(&cfg, &map)
            .expect_err("expected wrong-shape error");
        let msg = format!("{err}");
        assert!(
            msg.contains("output_norm.weight") && msg.contains("length"),
            "wrong error message: {msg}"
        );
    }

    #[test]
    fn transpose_2d_round_trip_identity() {
        // Transpose twice → identity.
        let rows = 7usize;
        let cols = 11usize;
        let src: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.13 - 0.5).collect();
        let t1 = transpose_2d_row_major(&src, rows, cols);
        assert_eq!(t1.len(), rows * cols);
        let t2 = transpose_2d_row_major(&t1, cols, rows);
        for (i, (a, b)) in src.iter().zip(t2.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "round-trip mismatch at {i}");
        }
    }

    #[test]
    fn transpose_2d_correctness_3x4() {
        // src[3, 4] row-major:
        //   [ 0  1  2  3]
        //   [ 4  5  6  7]
        //   [ 8  9 10 11]
        // transpose [4, 3]:
        //   [ 0  4  8]
        //   [ 1  5  9]
        //   [ 2  6 10]
        //   [ 3  7 11]
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = transpose_2d_row_major(&src, 3, 4);
        let expected: Vec<f32> = vec![
            0.0, 4.0, 8.0, // col 0
            1.0, 5.0, 9.0, // col 1
            2.0, 6.0, 10.0, // col 2
            3.0, 7.0, 11.0, // col 3
        ];
        assert_eq!(t, expected);
    }
}
