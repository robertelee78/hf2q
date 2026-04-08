//! Minimal transformer forward pass using Candle.
//!
//! Loads weights from hf2q's IR TensorMap using HuggingFace naming conventions,
//! then runs a decoder-only transformer forward pass. Supports activation capture
//! for DWQ calibration and quality measurement.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor, D};
#[cfg(test)]
use candle_core::IndexOp;
use candle_nn::{Embedding, Linear, Module};
use tracing::debug;

use crate::ir::{ModelMetadata, TensorMap};

use super::tensor_from_ir;

/// Output of a forward pass through the transformer.
pub struct ForwardOutput {
    /// Logits tensor: [seq_len, vocab_size].
    pub logits: Tensor,
    /// Per-layer hidden states, present only when activation capture is enabled.
    pub hidden_states: Option<Vec<Tensor>>,
}

/// RMSNorm layer (used by LLaMA, Gemma, Mistral, etc.).
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        // Upcast to f32 for numerical stability
        let x_f32 = x.to_dtype(DType::F32)?;
        let sq = x_f32.sqr()?;
        let mean_sq = sq.mean_keepdim(D::Minus1)?;
        let eps_t = mean_sq.ones_like()?.affine(0.0, self.eps)?;
        let rms = (mean_sq + eps_t)?.sqrt()?.recip()?;
        let normed = x_f32.broadcast_mul(&rms)?;
        let weight_f32 = self.weight.to_dtype(DType::F32)?;
        let result = normed.broadcast_mul(&weight_f32)?;
        result.to_dtype(dtype).map_err(Into::into)
    }
}

/// A single transformer layer (decoder block).
struct TransformerLayer {
    attn_norm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    ffn_norm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl TransformerLayer {
    /// Run attention + FFN for this layer.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;

        // Pre-attention norm
        let normed = self.attn_norm.forward(x)?;

        // Self-attention (no KV cache -- calibration only)
        let attn_out = self.attention(&normed)?;
        let x = (residual + attn_out)?;

        // Pre-FFN norm
        let residual = &x;
        let normed = self.ffn_norm.forward(&x)?;

        // SwiGLU FFN
        let ffn_out = self.ffn(&normed)?;
        (residual + ffn_out).map_err(Into::into)
    }

    /// Multi-head (grouped-query) attention without KV cache.
    fn attention(&self, x: &Tensor) -> Result<Tensor> {
        let (seq_len, _hidden) = x.dims2()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape: [seq_len, hidden] -> [num_heads, seq_len, head_dim]
        let q = q
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;
        let k = k
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(0, 1)?;
        let v = v
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(0, 1)?;

        // Apply RoPE (simplified: use relative position encoding via rotation)
        let q = self.apply_rope(&q, seq_len)?;
        let k = self.apply_rope(&k, seq_len)?;

        // GQA: repeat KV heads to match Q heads
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt().recip();
        let attn_weights = q
            .matmul(&k.transpose(D::Minus2, D::Minus1)?)?
            .affine(scale, 0.0)?;

        // Causal mask
        let attn_weights = self.apply_causal_mask(&attn_weights, seq_len)?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Attention output
        let attn_out = attn_weights.matmul(&v)?;

        // Reshape back: [num_heads, seq_len, head_dim] -> [seq_len, hidden]
        let attn_out = attn_out
            .transpose(0, 1)?
            .reshape((seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_out).map_err(Into::into)
    }

    /// Simplified RoPE: rotary position embedding.
    fn apply_rope(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let device = x.device();
        let dtype = x.dtype();
        let half_dim = self.head_dim / 2;

        // Frequency basis: theta_i = 1 / 10000^(2i/d)
        let theta: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / self.head_dim as f32))
            .collect();

        let theta = Tensor::from_vec(theta, (1, 1, half_dim), device)?;

        // Positions: [0, 1, ..., seq_len-1]
        let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (1, seq_len, 1), device)?;

        // Angles: [1, seq_len, half_dim]
        let angles = positions.broadcast_mul(&theta)?;

        let cos = angles.cos()?.to_dtype(dtype)?;
        let sin = angles.sin()?.to_dtype(dtype)?;

        // Split x into two halves along last dim
        let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;

        // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rotated_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[&rotated_x1, &rotated_x2], D::Minus1).map_err(Into::into)
    }

    /// Repeat KV heads to match the number of Q heads (for GQA).
    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_kv_heads == self.num_heads {
            return Ok(x.clone());
        }

        let repeats = self.num_heads / self.num_kv_heads;
        let (_kv_heads, seq_len, head_dim) = x.dims3()?;

        // Expand: [kv_heads, seq, dim] -> [kv_heads, repeats, seq, dim] -> [heads, seq, dim]
        let x = x.unsqueeze(1)?; // [kv_heads, 1, seq, dim]
        let x = x.expand((self.num_kv_heads, repeats, seq_len, head_dim))?;
        x.reshape((self.num_heads, seq_len, head_dim))
            .map_err(Into::into)
    }

    /// Apply causal (lower-triangular) attention mask.
    fn apply_causal_mask(&self, attn_weights: &Tensor, seq_len: usize) -> Result<Tensor> {
        if seq_len <= 1 {
            return Ok(attn_weights.clone());
        }

        let device = attn_weights.device();
        let dtype = attn_weights.dtype();

        // Build lower-triangular mask: 0 where allowed, -inf where masked
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }

        let mask = Tensor::from_vec(mask_data, (1, seq_len, seq_len), device)?
            .to_dtype(dtype)?;

        attn_weights.broadcast_add(&mask).map_err(Into::into)
    }

    /// SwiGLU FFN: down_proj(silu(gate_proj(x)) * up_proj(x)).
    fn ffn(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::Activation::Silu.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        let fused = (gate * up)?;
        self.down_proj.forward(&fused).map_err(Into::into)
    }
}

/// Full transformer forward pass for decoder-only models.
pub struct TransformerForward {
    embed: Embedding,
    layers: Vec<TransformerLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
}

impl TransformerForward {
    /// Load a transformer from the IR TensorMap using HuggingFace naming conventions.
    pub fn load(
        tensor_map: &TensorMap,
        metadata: &ModelMetadata,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let hidden_size = metadata.hidden_size as usize;
        let num_heads = metadata.num_attention_heads as usize;
        let num_kv_heads = metadata.num_kv_heads.unwrap_or(metadata.num_attention_heads) as usize;
        let head_dim = hidden_size / num_heads;
        let num_layers = metadata.num_layers as usize;
        let eps = Self::extract_rms_eps(metadata);

        debug!(
            hidden_size,
            num_heads, num_kv_heads, head_dim, num_layers, "Loading transformer"
        );

        // Helper to load a tensor by name
        let get = |name: &str| -> Result<Tensor> {
            let tref = tensor_map
                .tensors
                .get(name)
                .with_context(|| format!("Missing tensor: {}", name))?;
            tensor_from_ir(tref, device)
        };

        // Embedding
        let embed_weight = get("model.embed_tokens.weight")?;
        let embed = Embedding::new(embed_weight, hidden_size);

        // Layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("model.layers.{}", i);

            let attn_norm = RmsNorm::new(get(&format!("{}.input_layernorm.weight", prefix))?, eps);

            let q_proj = Self::load_linear(&get, &format!("{}.self_attn.q_proj", prefix))?;
            let k_proj = Self::load_linear(&get, &format!("{}.self_attn.k_proj", prefix))?;
            let v_proj = Self::load_linear(&get, &format!("{}.self_attn.v_proj", prefix))?;
            let o_proj = Self::load_linear(&get, &format!("{}.self_attn.o_proj", prefix))?;

            let ffn_norm = RmsNorm::new(
                get(&format!("{}.post_attention_layernorm.weight", prefix))?,
                eps,
            );

            let gate_proj = Self::load_linear(&get, &format!("{}.mlp.gate_proj", prefix))?;
            let up_proj = Self::load_linear(&get, &format!("{}.mlp.up_proj", prefix))?;
            let down_proj = Self::load_linear(&get, &format!("{}.mlp.down_proj", prefix))?;

            layers.push(TransformerLayer {
                attn_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                ffn_norm,
                gate_proj,
                up_proj,
                down_proj,
                num_heads,
                num_kv_heads,
                head_dim,
            });

            debug!("Loaded layer {}/{}", i + 1, num_layers);
        }

        // Final norm
        let final_norm = RmsNorm::new(get("model.norm.weight")?, eps);

        // LM head -- some models tie embed and lm_head weights
        let lm_head_weight = if tensor_map.tensors.contains_key("lm_head.weight") {
            get("lm_head.weight")?
        } else {
            debug!("lm_head.weight not found, using tied embed_tokens.weight");
            get("model.embed_tokens.weight")?
        };
        let lm_head = Linear::new(lm_head_weight, None);

        Ok(Self {
            embed,
            layers,
            final_norm,
            lm_head,
        })
    }

    /// Run the forward pass, returning logits.
    pub fn forward(&self, token_ids: &[u32]) -> Result<ForwardOutput> {
        self.forward_inner(token_ids, false)
    }

    /// Run the forward pass, capturing per-layer hidden states.
    pub fn forward_with_activations(
        &self,
        token_ids: &[u32],
    ) -> Result<ForwardOutput> {
        self.forward_inner(token_ids, true)
    }

    /// Internal forward pass implementation.
    fn forward_inner(&self, token_ids: &[u32], capture: bool) -> Result<ForwardOutput> {
        if token_ids.is_empty() {
            bail!("Cannot run forward pass with empty token sequence");
        }

        let device = self.embed.embeddings().device();
        let seq_len = token_ids.len();

        // Convert token IDs to tensor
        let ids: Vec<u32> = token_ids.to_vec();
        let input = Tensor::from_vec(ids, (seq_len,), device)?;

        // Embedding lookup
        let mut hidden = self.embed.forward(&input)?;

        // Collect activations if requested
        let mut hidden_states = if capture {
            Some(Vec::with_capacity(self.layers.len() + 1))
        } else {
            None
        };

        if let Some(ref mut hs) = hidden_states {
            hs.push(hidden.clone());
        }

        // Transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;

            if let Some(ref mut hs) = hidden_states {
                hs.push(hidden.clone());
            }
        }

        // Final norm
        hidden = self.final_norm.forward(&hidden)?;

        // LM head: [seq_len, hidden] -> [seq_len, vocab_size]
        let logits = self.lm_head.forward(&hidden)?;

        Ok(ForwardOutput {
            logits,
            hidden_states,
        })
    }

    /// Load a Linear layer (weight + optional bias) from the tensor map.
    fn load_linear(
        get: &dyn Fn(&str) -> Result<Tensor>,
        prefix: &str,
    ) -> Result<Linear> {
        let weight = get(&format!("{}.weight", prefix))?;
        let bias_key = format!("{}.bias", prefix);
        // Try to load bias, but it's optional
        let bias = get(&bias_key).ok();
        Ok(Linear::new(weight, bias))
    }

    /// Extract RMSNorm epsilon from model metadata.
    fn extract_rms_eps(metadata: &ModelMetadata) -> f64 {
        metadata
            .raw_config
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6)
    }

    /// Number of transformer layers.
    #[allow(dead_code)]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_identity() {
        // RMSNorm with weight=1 should normalize but preserve direction
        let device = candle_core::Device::Cpu;
        let weight = Tensor::ones((4,), DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let result = norm.forward(&x).unwrap();

        // Should be normalized: each element / rms(x)
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals.len(), 4);

        // rms = sqrt(mean(1^2 + 2^2 + 3^2 + 4^2)) = sqrt(30/4) = sqrt(7.5)
        let rms = (7.5f32).sqrt();
        for (i, v) in vals.iter().enumerate() {
            let expected = (i as f32 + 1.0) / rms;
            assert!(
                (v - expected).abs() < 1e-4,
                "Element {}: got {}, expected {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_forward_output_struct() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::zeros((5, 100), DType::F32, &device).unwrap();
        let output = ForwardOutput {
            logits,
            hidden_states: None,
        };
        assert_eq!(output.logits.dims(), &[5, 100]);
        assert!(output.hidden_states.is_none());
    }

    #[test]
    fn test_causal_mask() {
        let device = candle_core::Device::Cpu;
        let layer = TransformerLayer {
            attn_norm: RmsNorm::new(Tensor::ones((4,), DType::F32, &device).unwrap(), 1e-6),
            q_proj: Linear::new(
                Tensor::zeros((4, 4), DType::F32, &device).unwrap(),
                None,
            ),
            k_proj: Linear::new(
                Tensor::zeros((4, 4), DType::F32, &device).unwrap(),
                None,
            ),
            v_proj: Linear::new(
                Tensor::zeros((4, 4), DType::F32, &device).unwrap(),
                None,
            ),
            o_proj: Linear::new(
                Tensor::zeros((4, 4), DType::F32, &device).unwrap(),
                None,
            ),
            ffn_norm: RmsNorm::new(Tensor::ones((4,), DType::F32, &device).unwrap(), 1e-6),
            gate_proj: Linear::new(
                Tensor::zeros((8, 4), DType::F32, &device).unwrap(),
                None,
            ),
            up_proj: Linear::new(
                Tensor::zeros((8, 4), DType::F32, &device).unwrap(),
                None,
            ),
            down_proj: Linear::new(
                Tensor::zeros((4, 8), DType::F32, &device).unwrap(),
                None,
            ),
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
        };

        // Test causal mask shape
        let attn = Tensor::zeros((2, 3, 3), DType::F32, &device).unwrap();
        let masked = layer.apply_causal_mask(&attn, 3).unwrap();
        assert_eq!(masked.dims(), &[2, 3, 3]);

        // Upper triangle should be -inf
        let vals: Vec<f32> = masked.i(0).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        // Position (0,1) should be -inf
        assert!(vals[1].is_infinite() && vals[1] < 0.0);
        // Position (0,0) should be 0
        assert_eq!(vals[0], 0.0);
        // Position (1,1) should be 0
        assert_eq!(vals[4], 0.0);
    }
}
