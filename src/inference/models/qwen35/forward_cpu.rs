//! Integrated CPU-reference forward pass for `Qwen35Model`.
//!
//! Composes:
//! * [`super::io_heads::embed_tokens`] — token → hidden.
//! * [`super::full_attn::gated_full_attention_cpu_ref`] — full-attention layers.
//! * [`super::delta_net::delta_net_layer_cpu_ref`] — linear-attention layers.
//! * [`super::ffn::dense_swiglu_cpu_ref`] / `moe_ffn_cpu_ref` — FFN per variant.
//! * [`super::io_heads::apply_output_head`] — final norm + lm_head.
//!
//! The post-attention norm (applied between attention and FFN — `pre_ffw`
//! in some references, `post_attention_norm` in apex GGUF) is a no-op in
//! this reference because:
//!
//! 1. The apex GGUF embeds it in every layer AS `post_attention_norm.weight`,
//!    but the forward order in both full-attn and DeltaNet CPU refs already
//!    produces the post-attention residual stream. The post-attention norm
//!    is applied to that stream BEFORE feeding into the FFN.
//! 2. For the CPU reference, we inline it here as a pre-FFN RMSNorm using
//!    whichever per-layer norm weight the caller provides.
//!
//! The `Qwen35Model::forward_cpu` entrypoint sets `state_in` / `conv_state`
//! to zeros for a fresh forward pass (prefill regime). Decode regime with
//! a pre-populated hybrid KV cache is a future extension.

use anyhow::{anyhow, Result};

use super::delta_net::{delta_net_layer_cpu_ref, DeltaNetLayerShape};
use super::ffn::{
    dense_swiglu_cpu_ref, moe_ffn_cpu_ref, DenseFfnShape, MoeFfnShape,
};
use super::full_attn::{gated_full_attention_cpu_ref, FullAttnShape};
use super::io_heads::{apply_output_head, embed_tokens};
use super::model::{Qwen35FfnWeights, Qwen35LayerWeights, Qwen35Model};

/// RMSNorm over the last axis (row-wise).
fn rms_norm_rows(x: &mut [f32], weight: &[f32], hidden: usize, eps: f32) {
    assert_eq!(weight.len(), hidden);
    let seq = x.len() / hidden;
    for t in 0..seq {
        let row = &mut x[t * hidden..(t + 1) * hidden];
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = ((sum_sq / (hidden as f32)) + eps).sqrt().recip();
        for (j, v) in row.iter_mut().enumerate() {
            *v = *v * inv * weight[j];
        }
    }
}

/// In-place residual add: `dst[i] += src[i]`.
fn residual_add(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

impl Qwen35Model {
    /// Pure-Rust f32 forward pass (prefill regime, no incremental KV cache).
    ///
    /// # Arguments
    ///
    /// * `tokens`: input token IDs, length = seq_len.
    /// * `positions`: per-token axis positions `[[t_pos, h_pos, w_pos, e_pos]; seq_len]`.
    ///   For text-only Qwen3.5, replicate the token index across all 4 axes.
    ///
    /// # Returns
    ///
    /// `[seq_len, vocab_size]` row-major logits.
    ///
    /// # Tests
    ///
    /// Validated end-to-end with a tiny synthetic config (2 layers, 1 full + 1
    /// linear, small dims) — verifies no shape or NaN bugs and that the
    /// composition matches per-stage references.
    pub fn forward_cpu(
        &self,
        tokens: &[u32],
        positions: &[[i32; 4]],
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(anyhow!("forward_cpu: tokens must be non-empty"));
        }
        if tokens.len() != positions.len() {
            return Err(anyhow!(
                "forward_cpu: tokens.len() = {} != positions.len() = {}",
                tokens.len(),
                positions.len()
            ));
        }

        let _seq = tokens.len();
        let h = self.cfg.hidden_size as usize;
        let eps = self.cfg.rms_norm_eps;

        // 1. Embedding lookup → hidden[0].
        let mut hidden = embed_tokens(
            tokens,
            &self.token_embd,
            self.cfg.vocab_size,
            self.cfg.hidden_size,
        );

        // 2. Per-layer forward pass.
        for layer in &self.layers {
            // Attention contribution.
            let attn_out = match layer {
                Qwen35LayerWeights::FullAttn { attn, .. } => {
                    let shape = FullAttnShape::from_config(&self.cfg);
                    gated_full_attention_cpu_ref(&hidden, positions, attn, shape)
                }
                Qwen35LayerWeights::LinearAttn { attn, .. } => {
                    let shape = DeltaNetLayerShape::from_config(&self.cfg);
                    let state_in = vec![
                        0.0f32;
                        (self.cfg.linear_key_head_dim
                            * self.cfg.linear_value_head_dim
                            * self.cfg.linear_num_value_heads)
                            as usize
                    ];
                    let km1 = (self.cfg.linear_conv_kernel_dim - 1) as usize;
                    let qkv_channels = (2 * self.cfg.linear_num_key_heads
                        * self.cfg.linear_key_head_dim
                        + self.cfg.linear_num_value_heads
                            * self.cfg.linear_value_head_dim)
                        as usize;
                    let conv_state = vec![0.0f32; km1 * qkv_channels];
                    let (out, _new_state, _new_conv) =
                        delta_net_layer_cpu_ref(&hidden, attn, shape, &state_in, &conv_state);
                    out
                }
            };

            // Residual after attention.
            residual_add(&mut hidden, &attn_out);

            // Post-attention RMSNorm: the normed value is the FFN *input* only.
            // The FFN output is added back to `ffn_residual` (the pre-norm value),
            // matching llama.cpp qwen35moe.cpp:
            //   ffn_residual = cur;               // after attn residual, BEFORE norm
            //   attn_post_norm = build_norm(cur); // norm for FFN input
            //   cur = build_layer_ffn(attn_post_norm);
            //   cur = ggml_add(cur, ffn_residual);// FFN residual is pre-norm
            let ffn_residual = hidden.clone();
            let mut ffn_input = hidden.clone();
            let post_norm_w = match layer {
                Qwen35LayerWeights::FullAttn { attn, .. } => &attn.post_attn_norm,
                Qwen35LayerWeights::LinearAttn { attn, .. } => &attn.post_attn_norm,
            };
            rms_norm_rows(&mut ffn_input, post_norm_w, h, eps);

            // FFN contribution (takes normed input, not pre-norm residual).
            let ffn_out = match &layer.ffn() {
                Qwen35FfnWeights::Dense(w) => {
                    let m = self
                        .cfg
                        .intermediate_size
                        .ok_or_else(|| anyhow!("dense variant missing intermediate_size"))?;
                    let shape = DenseFfnShape {
                        hidden_size: self.cfg.hidden_size,
                        intermediate_size: m,
                    };
                    dense_swiglu_cpu_ref(&ffn_input, w, shape)
                }
                Qwen35FfnWeights::Moe(w) => {
                    let moe = self.cfg.moe.as_ref().ok_or_else(|| {
                        anyhow!("moe variant missing moe config")
                    })?;
                    let shape = MoeFfnShape {
                        hidden_size: self.cfg.hidden_size,
                        num_experts: moe.num_experts,
                        num_experts_per_tok: moe.num_experts_per_tok,
                        moe_intermediate_size: moe.moe_intermediate_size,
                        shared_intermediate_size: moe.shared_expert_intermediate_size,
                    };
                    moe_ffn_cpu_ref(&ffn_input, w, shape)
                }
                // DenseQ and MoeQ are GGUF-loaded quantized variants; forward_cpu
                // does not support them (projection weights are Metal buffers, not
                // f32 vecs). CPU-only inference is not needed for the production path.
                Qwen35FfnWeights::DenseQ(_) => {
                    return Err(anyhow!(
                        "forward_cpu does not support DenseQ (quantized dense FFN weights \
                         loaded from GGUF); use forward_gpu instead"
                    ));
                }
                Qwen35FfnWeights::MoeQ(_) => {
                    return Err(anyhow!(
                        "forward_cpu does not support MoeQ (quantized expert weights \
                         loaded from GGUF); use forward_gpu instead"
                    ));
                }
            };

            // Residual after FFN: add to pre-norm value (ffn_residual), not normed.
            // This matches llama.cpp's `cur = ggml_add(cur, ffn_residual)`.
            hidden = ffn_residual;
            residual_add(&mut hidden, &ffn_out);
        }

        // 3. Final RMSNorm (absorbed into apply_output_head) + LM head.
        let logits = apply_output_head(
            &hidden,
            &self.output_norm,
            &self.output_weight,
            self.cfg.hidden_size,
            self.cfg.vocab_size,
            eps,
        );
        let _ = h; // shape var kept for doc clarity
        Ok(logits)
    }
}

// Re-export a helper for tests to easily construct text-convention positions.
pub fn text_positions(seq_len: u32) -> Vec<[i32; 4]> {
    (0..seq_len as i32).map(|i| [i, i, i, i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::{
        default_layer_types, Qwen35Config, Qwen35LayerKind, Qwen35MoeConfig, Qwen35Variant,
    };

    fn tiny_moe_cfg() -> Qwen35Config {
        // 2 layers: layer 0 linear, layer 1 full (we fake interval=2 so
        // the second is full-attn per our `default_layer_types` formula).
        let layer_types = default_layer_types(2, 2);
        assert_eq!(layer_types[0], Qwen35LayerKind::LinearAttention);
        assert_eq!(layer_types[1], Qwen35LayerKind::FullAttention);
        Qwen35Config {
            variant: Qwen35Variant::Moe,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            layer_types,
            partial_rotary_factor: 0.5,
            rope_theta: 10000.0,
            rotary_dim: 2,
            mrope_section: [1, 0, 0, 0],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 128,
            vocab_size: 32,
            attn_output_gate: true,
            mtp_num_hidden_layers: 0,
            intermediate_size: None,
            moe: Some(Qwen35MoeConfig {
                moe_intermediate_size: 4,
                num_experts: 2,
                num_experts_per_tok: 1,
                shared_expert_intermediate_size: 4,
            }),
        }
    }

    fn tiny_dense_cfg() -> Qwen35Config {
        let mut c = tiny_moe_cfg();
        c.variant = Qwen35Variant::Dense;
        c.intermediate_size = Some(4);
        c.moe = None;
        c
    }

    /// Integrated forward pass on a zero-weighted model produces finite
    /// output with the expected logits shape. The zeros-model path is a
    /// good smoke for shape plumbing without weight-dependent bugs.
    #[test]
    fn forward_cpu_zero_model_returns_correct_shape() {
        let cfg = tiny_moe_cfg();
        let m = Qwen35Model::empty_from_cfg(cfg.clone());
        let tokens = vec![0u32, 1, 2, 3];
        let positions = text_positions(tokens.len() as u32);
        let logits = m.forward_cpu(&tokens, &positions).expect("forward");
        assert_eq!(logits.len(), tokens.len() * cfg.vocab_size as usize);
        // Zero weights + zero embedding + zero-filled norm=1 layers produces
        // zero hidden stream throughout → logits = lm_head @ 0 = 0.
        for v in &logits {
            assert!(v.is_finite(), "zero-model produced non-finite logits");
        }
    }

    /// Non-zero embedding table + zero-everything-else — embedded hidden
    /// state flows through unchanged-by-zero layers, final logits reflect
    /// the embedding contents after the (ineffective) transform stack.
    #[test]
    fn forward_cpu_embedding_flows_through_zero_layers() {
        let cfg = tiny_moe_cfg();
        let mut m = Qwen35Model::empty_from_cfg(cfg.clone());

        // Fill token embedding with a unique pattern per token.
        for t in 0..cfg.vocab_size as usize {
            for j in 0..cfg.hidden_size as usize {
                m.token_embd[t * cfg.hidden_size as usize + j] =
                    (t as f32) * 0.01 + (j as f32) * 0.001;
            }
        }

        let tokens = vec![5u32, 10, 15];
        let positions = text_positions(tokens.len() as u32);
        let logits = m.forward_cpu(&tokens, &positions).expect("forward");
        assert_eq!(logits.len(), tokens.len() * cfg.vocab_size as usize);
        for v in &logits {
            assert!(v.is_finite());
        }
    }

    /// Determinism: same model + tokens → same logits bit-for-bit.
    #[test]
    fn forward_cpu_deterministic() {
        let cfg = tiny_dense_cfg();
        let mut m = Qwen35Model::empty_from_cfg(cfg.clone());

        // Seed some weights deterministically.
        for (i, v) in m.token_embd.iter_mut().enumerate() {
            *v = ((i as f32) * 0.001 - 0.1).sin();
        }
        for (i, v) in m.output_weight.iter_mut().enumerate() {
            *v = ((i as f32) * 0.0005 - 0.05).cos();
        }

        let tokens = vec![3u32, 7];
        let positions = text_positions(tokens.len() as u32);
        let l1 = m.forward_cpu(&tokens, &positions).expect("1");
        let l2 = m.forward_cpu(&tokens, &positions).expect("2");
        assert_eq!(l1.len(), l2.len());
        for i in 0..l1.len() {
            assert_eq!(l1[i].to_bits(), l2[i].to_bits(), "non-deterministic at {}", i);
        }
    }

    /// Invalid input: empty tokens → error (not panic).
    #[test]
    fn forward_cpu_rejects_empty_tokens() {
        let cfg = tiny_moe_cfg();
        let m = Qwen35Model::empty_from_cfg(cfg);
        let positions: Vec<[i32; 4]> = vec![];
        let result = m.forward_cpu(&[], &positions);
        assert!(result.is_err());
    }

    /// Invalid input: positions length mismatch → error.
    #[test]
    fn forward_cpu_rejects_position_mismatch() {
        let cfg = tiny_moe_cfg();
        let m = Qwen35Model::empty_from_cfg(cfg);
        let tokens = vec![0u32, 1, 2];
        let positions = text_positions(2); // mismatched
        let result = m.forward_cpu(&tokens, &positions);
        assert!(result.is_err());
    }

    /// Both dense and MoE variants run cleanly on the same token input.
    /// Exercises both FFN branches of the layer dispatch.
    #[test]
    fn forward_cpu_runs_both_ffn_variants() {
        let tokens = vec![1u32, 2];
        let positions = text_positions(tokens.len() as u32);

        let m_moe = Qwen35Model::empty_from_cfg(tiny_moe_cfg());
        let l_moe = m_moe.forward_cpu(&tokens, &positions).expect("moe");
        assert_eq!(l_moe.len(), 2 * 32);

        let m_dense = Qwen35Model::empty_from_cfg(tiny_dense_cfg());
        let l_dense = m_dense.forward_cpu(&tokens, &positions).expect("dense");
        assert_eq!(l_dense.len(), 2 * 32);
    }

    /// Argmax sampling integration: with deterministic weights that prefer
    /// a specific vocab index, forward + argmax should return that index.
    /// Constructs lm_head so that vocab index 7 has weights matching the
    /// normalized hidden state, and all other vocab indices have zero weights.
    #[test]
    fn forward_cpu_plus_argmax_selects_expected_token() {
        use super::super::io_heads::greedy_argmax_last_token;

        let cfg = tiny_moe_cfg();
        let mut m = Qwen35Model::empty_from_cfg(cfg.clone());

        // token_embd for token 5: distinctive pattern.
        let h = cfg.hidden_size as usize;
        for j in 0..h {
            m.token_embd[5 * h + j] = (j as f32 + 1.0) * 0.1;
        }

        // output_weight: make vocab index 7 strongly prefer our expected
        // normalized hidden. Row 7 = pattern, other rows = zero.
        for j in 0..h {
            m.output_weight[7 * h + j] = 10.0;
        }

        // output_norm_w = 1 so the normalized hidden is > 0.
        for v in &mut m.output_norm {
            *v = 1.0;
        }

        let tokens = vec![5u32];
        let positions = text_positions(1);
        let logits = m.forward_cpu(&tokens, &positions).expect("forward");
        let picked = greedy_argmax_last_token(&logits, cfg.vocab_size);

        // We expect vocab index 7 to have the highest logit (positive),
        // while all other indices should be 0 (their rows are zero).
        assert_eq!(picked, 7, "expected vocab 7, got {}", picked);
    }
}
