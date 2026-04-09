//! Minimal transformer forward pass using Candle.
//!
//! Loads weights from hf2q's IR TensorMap using HuggingFace naming conventions,
//! then runs a decoder-only transformer forward pass. Supports activation capture
//! for DWQ calibration and quality measurement.
//!
//! Supports both generic LLaMA-like models and Gemma4 A4B (MoE) architecture.

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

/// Unit RMSNorm (no learned weight, just normalize). Used for V in Gemma4.
fn rms_norm_unit(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq = x_f32.sqr()?;
    let mean_sq = sq.mean_keepdim(D::Minus1)?;
    let eps_t = mean_sq.ones_like()?.affine(0.0, 1e-6)?;
    let rms = (mean_sq + eps_t)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rms)?;
    normed.to_dtype(dtype).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Activation {
    Silu,
    Gelu,
}

impl Activation {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Activation::Silu => candle_nn::Activation::Silu.forward(x).map_err(Into::into),
            Activation::Gelu => candle_nn::Activation::Gelu.forward(x).map_err(Into::into),
        }
    }
}

// ---------------------------------------------------------------------------
// Rotary Embedding (configurable)
// ---------------------------------------------------------------------------

/// Rotary position embedding supporting standard (full) and partial rotation.
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    /// Standard RoPE: rotates all dimensions.
    fn new_standard(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        device: &candle_core::Device,
    ) -> Result<Self> {
        Self::build(head_dim, head_dim, max_seq_len, rope_theta, device)
    }

    /// Partial RoPE: only rotates a fraction of dimensions.
    /// IMPORTANT: frequency denominator uses full head_dim, not rotary_dim.
    fn new_partial(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let rope_angles =
            ((partial_rotary_factor * head_dim as f64 / 2.0).floor() as usize).min(head_dim / 2);
        let rotary_dim = rope_angles * 2;
        Self::build(head_dim, rotary_dim, max_seq_len, rope_theta, device)
    }

    fn build(
        head_dim: usize,
        rotary_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let half = rotary_dim / 2;
        // Frequency basis uses full head_dim for denominator
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / rope_theta.powf(2.0 * i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        Ok(Self {
            sin,
            cos,
            rotary_dim,
        })
    }

    /// Apply RoPE to Q and K tensors.
    /// Input shape: [num_heads, seq_len, head_dim]
    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_h, seq_len, head_dim) = q.dims3()?;
        let cos = self.cos.narrow(0, 0, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, 0, seq_len)?.to_dtype(q.dtype())?;

        if self.rotary_dim == head_dim {
            let q_rot = Self::rope_apply(&q.contiguous()?, &cos, &sin)?;
            let k_rot = Self::rope_apply(&k.contiguous()?, &cos, &sin)?;
            Ok((q_rot, k_rot))
        } else {
            // Partial rotation: rotate first rotary_dim dims, pass through the rest
            let pass_len = head_dim - self.rotary_dim;
            let q_rot_part = q.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
            let q_pass = q.narrow(D::Minus1, self.rotary_dim, pass_len)?;
            let k_rot_part = k.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
            let k_pass = k.narrow(D::Minus1, self.rotary_dim, pass_len)?;

            let q_rot = Self::rope_apply(&q_rot_part, &cos, &sin)?;
            let k_rot = Self::rope_apply(&k_rot_part, &cos, &sin)?;

            let q_out = Tensor::cat(&[q_rot, q_pass.contiguous()?], D::Minus1)?;
            let k_out = Tensor::cat(&[k_rot, k_pass.contiguous()?], D::Minus1)?;
            Ok((q_out, k_out))
        }
    }

    /// Standard RoPE rotation: split x into (x1,x2), rotate.
    fn rope_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let half = x.dim(D::Minus1)? / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[r1, r2], D::Minus1).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// MoE Block (Mixture of Experts with Sigmoid Routing)
// ---------------------------------------------------------------------------

/// Mixture of Experts block with sigmoid routing for Gemma4.
struct MoeBlock {
    /// Router projection: hidden_size -> num_experts
    router_proj: Linear,
    /// Per-hidden-dim scale applied before routing
    router_scale: Tensor,
    /// Per-expert scale/bias
    per_expert_scale: Tensor,
    /// Expert weights: gate_up [num_experts, intermediate*2, hidden] (safetensors layout)
    expert_gate_up: Tensor,
    /// Expert weights: down [num_experts, hidden, intermediate] (safetensors layout)
    expert_down: Tensor,
    #[allow(dead_code)]
    num_experts: usize,
    top_k: usize,
    moe_intermediate_size: usize,
    activation: Activation,
}

impl MoeBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq_len, hidden) = x.dims2()?;

        // Router: h_scaled = h * scale, logits = h_scaled @ proj, weighted by per_expert_scale
        let x_scaled = x.broadcast_mul(&self.router_scale)?;
        let logits = self.router_proj.forward(&x_scaled)?; // [seq_len, num_experts]
        let logits = logits.broadcast_mul(&self.per_expert_scale)?;

        // Sigmoid gating (not softmax)
        let weights = candle_nn::ops::sigmoid(&logits)?; // [seq_len, num_experts]

        // Process tokens — loop over tokens for correct top-k selection
        let weights_cpu: Vec<Vec<f32>> = weights.to_dtype(DType::F32)?.to_vec2()?;
        let device = x.device();
        let dtype = x.dtype();
        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);

        for tok_idx in 0..seq_len {
            let token_weights = &weights_cpu[tok_idx];

            // Select top-k experts
            let mut indexed: Vec<(usize, f32)> =
                token_weights.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(self.top_k);

            let sum: f32 = indexed.iter().map(|&(_, w)| w).sum();
            let token_vec = x.narrow(0, tok_idx, 1)?; // [1, hidden]

            let mut combined = Tensor::zeros((1, hidden), dtype, device)?;

            for &(expert_idx, weight) in &indexed {
                // gate_up_proj: [num_experts, intermediate*2, hidden]
                // Select expert e -> [intermediate*2, hidden]
                let gate_up_w = self
                    .expert_gate_up
                    .narrow(0, expert_idx, 1)?
                    .squeeze(0)?
                    .contiguous()?; // [intermediate*2, hidden]
                // token_vec [1, hidden] @ gate_up_w^T [hidden, intermediate*2] -> [1, intermediate*2]
                let gate_up = token_vec.matmul(&gate_up_w.t()?)?;

                let gate = gate_up.narrow(1, 0, self.moe_intermediate_size)?;
                let up = gate_up.narrow(
                    1,
                    self.moe_intermediate_size,
                    self.moe_intermediate_size,
                )?;

                let gate_act = self.activation.forward(&gate)?;
                let fused = (gate_act * up)?;

                // down_proj: [num_experts, hidden, intermediate]
                // Select expert e -> [hidden, intermediate]
                let down_w = self
                    .expert_down
                    .narrow(0, expert_idx, 1)?
                    .squeeze(0)?
                    .contiguous()?; // [hidden, intermediate]
                // fused [1, intermediate] @ down_w^T [intermediate, hidden] -> [1, hidden]
                let expert_out = fused.matmul(&down_w.t()?)?;

                let scaled = (expert_out * (weight / sum) as f64)?;
                combined = (combined + scaled)?;
            }

            outputs.push(combined);
        }

        let result = Tensor::cat(&outputs, 0)?;
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Gemma4 layer configuration (extracted per-layer)
// ---------------------------------------------------------------------------

/// Per-layer Gemma4 configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
enum LayerType {
    Sliding,
    Full,
}

/// Gemma4-specific configuration extracted from model metadata.
struct Gemma4Config {
    head_dim: usize,          // sliding head_dim (256)
    global_head_dim: usize,   // full attention head_dim (512)
    num_kv_heads: usize,      // sliding kv heads (8)
    num_global_kv_heads: usize, // full attention kv heads (2)
    num_attention_heads: usize, // 16 for both
    rope_theta_sliding: f64,
    rope_theta_global: f64,
    partial_rotary_factor: f64,
    attention_k_eq_v: bool,
    moe_intermediate_size: usize,
    num_experts: usize,
    top_k_experts: usize,
    final_logit_softcapping: Option<f64>,
    layer_types: Vec<LayerType>,
    hidden_size: usize,
    intermediate_size: usize,
}

impl Gemma4Config {
    /// Extract Gemma4 config from raw_config JSON.
    fn from_metadata(metadata: &ModelMetadata) -> Result<Self> {
        let rc = &metadata.raw_config;

        // Try top-level or nested under text_config
        let text_cfg = rc.get("text_config").unwrap_or(rc);

        let get_usize = |key: &str, default: usize| -> usize {
            text_cfg
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let get_bool = |key: &str, default: bool| -> bool {
            text_cfg
                .get(key)
                .and_then(|v| v.as_bool())
                .unwrap_or(default)
        };

        // Extract rope parameters from nested structure
        let rope_params = text_cfg.get("rope_parameters");
        let rope_theta_sliding = rope_params
            .and_then(|rp| rp.get("sliding_attention"))
            .and_then(|sa| sa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        let rope_theta_global = rope_params
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1000000.0);
        let partial_rotary_factor = rope_params
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25);

        // Layer types from metadata or raw config
        let layer_types: Vec<LayerType> = if !metadata.layer_types.is_empty() {
            metadata
                .layer_types
                .iter()
                .map(|s| {
                    if s == "full_attention" {
                        LayerType::Full
                    } else {
                        LayerType::Sliding
                    }
                })
                .collect()
        } else {
            // Extract from text_config.layer_types
            text_cfg
                .get("layer_types")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|v| {
                            if v.as_str() == Some("full_attention") {
                                LayerType::Full
                            } else {
                                LayerType::Sliding
                            }
                        })
                        .collect()
                })
                .unwrap_or_else(|| {
                    // Default: every 6th layer is full attention
                    let n = metadata.num_layers as usize;
                    (0..n)
                        .map(|i| {
                            if (i + 1) % 6 == 0 {
                                LayerType::Full
                            } else {
                                LayerType::Sliding
                            }
                        })
                        .collect()
                })
        };

        let final_logit_softcapping = text_cfg
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64());

        Ok(Self {
            head_dim: get_usize("head_dim", 256),
            global_head_dim: get_usize("global_head_dim", 512),
            num_kv_heads: get_usize("num_key_value_heads", 8),
            num_global_kv_heads: get_usize("num_global_key_value_heads", 2),
            num_attention_heads: get_usize("num_attention_heads", 16),
            rope_theta_sliding,
            rope_theta_global,
            partial_rotary_factor,
            attention_k_eq_v: get_bool("attention_k_eq_v", true),
            moe_intermediate_size: get_usize("moe_intermediate_size", 704),
            num_experts: metadata.num_experts.unwrap_or(128) as usize,
            top_k_experts: metadata.top_k_experts.unwrap_or(8) as usize,
            final_logit_softcapping,
            layer_types,
            hidden_size: metadata.hidden_size as usize,
            intermediate_size: metadata.intermediate_size.unwrap_or(2112) as usize,
        })
    }

    fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .copied()
            == Some(LayerType::Full)
    }

    fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_full_attention(layer_idx) {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    fn num_kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_full_attention(layer_idx) {
            self.num_global_kv_heads
        } else {
            self.num_kv_heads
        }
    }
}

// ---------------------------------------------------------------------------
// Generic transformer layer (LLaMA-like)
// ---------------------------------------------------------------------------

/// A single transformer layer (decoder block) — generic path.
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
        let attn_weights = apply_causal_mask(&attn_weights, seq_len)?;

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
        repeat_kv_heads(x, self.num_heads, self.num_kv_heads)
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

// ---------------------------------------------------------------------------
// Gemma4 transformer layer
// ---------------------------------------------------------------------------

/// A single Gemma4 decoder layer with attention + dense MLP + MoE.
struct Gemma4Layer {
    // Attention projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    // Q/K norms (applied after projection, before transpose)
    q_norm: RmsNorm,
    k_norm: RmsNorm,

    // Attention config
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    k_eq_v: bool,

    // RoPE for this layer
    rotary_emb: RotaryEmbedding,

    // Dense MLP
    mlp_gate_proj: Linear,
    mlp_up_proj: Linear,
    mlp_down_proj: Linear,

    // MoE block
    moe: MoeBlock,

    // All norm layers (9 total per layer)
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    pre_feedforward_layernorm_2: RmsNorm,
    post_feedforward_layernorm_1: RmsNorm,
    post_feedforward_layernorm_2: RmsNorm,

    // Layer scalar (multiplied at end)
    layer_scalar: Option<Tensor>,

    // Activation function
    activation: Activation,
}

impl Gemma4Layer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Attention block
        let residual = x;
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.attention(&normed)?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let xs = (residual + attn_out)?;

        // 2. Dense MLP block
        let residual = &xs;
        let normed = self.pre_feedforward_layernorm.forward(&xs)?;
        let mlp_out = self.mlp(&normed)?;
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        let xs = (residual + mlp_out)?;

        // 3. MoE block
        let residual = &xs;
        let normed = self.pre_feedforward_layernorm_2.forward(&xs)?;
        let moe_out = self.moe.forward(&normed)?;
        let moe_out = self.post_feedforward_layernorm_1.forward(&moe_out)?;
        let moe_out = self.post_feedforward_layernorm_2.forward(&moe_out)?;
        let xs = (residual + moe_out)?;

        // 4. Layer scalar
        match &self.layer_scalar {
            Some(scalar) => xs.broadcast_mul(scalar).map_err(Into::into),
            None => Ok(xs),
        }
    }

    /// Multi-head attention with Q/K norms, V=K tying, unit V norm.
    fn attention(&self, x: &Tensor) -> Result<Tensor> {
        let (seq_len, _hidden) = x.dims2()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;

        // Reshape to [seq_len, num_heads/num_kv_heads, head_dim]
        let q = q.reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        // V: either clone K or use v_proj
        let v = if self.k_eq_v {
            k.clone()
        } else {
            self.v_proj
                .forward(x)?
                .reshape((seq_len, self.num_kv_heads, self.head_dim))?
        };

        // Apply Q/K norms before transpose (norms operate on last dim = head_dim)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Apply unit RMSNorm on V
        let v = rms_norm_unit(&v)?;

        // Transpose to [heads, seq_len, head_dim]
        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;

        // Apply RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k)?;

        // GQA: repeat KV heads to match Q heads
        let k = repeat_kv_heads(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv_heads(&v, self.num_heads, self.num_kv_heads)?;

        // Attention scores — no scale when Q/K norms are present (scale = 1.0)
        let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;

        // Causal mask
        let attn_weights = apply_causal_mask(&attn_weights, seq_len)?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Attention output
        let attn_out = attn_weights.matmul(&v)?;

        // Reshape back: [num_heads, seq_len, head_dim] -> [seq_len, num_heads*head_dim]
        let attn_out = attn_out
            .transpose(0, 1)?
            .reshape((seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_out).map_err(Into::into)
    }

    /// Dense MLP: GELU-gated.
    fn mlp(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.mlp_gate_proj.forward(x)?;
        let gate = self.activation.forward(&gate)?;
        let up = self.mlp_up_proj.forward(x)?;
        let fused = (gate * up)?;
        self.mlp_down_proj.forward(&fused).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Repeat KV heads to match Q heads (for GQA).
fn repeat_kv_heads(x: &Tensor, num_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_kv_heads == num_heads {
        return Ok(x.clone());
    }

    let repeats = num_heads / num_kv_heads;
    let (_kv_heads, seq_len, head_dim) = x.dims3()?;

    // Expand: [kv_heads, seq, dim] -> [kv_heads, repeats, seq, dim] -> [heads, seq, dim]
    let x = x.unsqueeze(1)?; // [kv_heads, 1, seq, dim]
    let x = x.expand((num_kv_heads, repeats, seq_len, head_dim))?;
    x.reshape((num_heads, seq_len, head_dim))
        .map_err(Into::into)
}

/// Apply causal (lower-triangular) attention mask.
fn apply_causal_mask(attn_weights: &Tensor, seq_len: usize) -> Result<Tensor> {
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

    let mask =
        Tensor::from_vec(mask_data, (1, seq_len, seq_len), device)?.to_dtype(dtype)?;

    attn_weights.broadcast_add(&mask).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Full transformer forward pass
// ---------------------------------------------------------------------------

/// Which architecture variant is loaded.
enum ModelVariant {
    /// Generic LLaMA-like transformer.
    Generic {
        layers: Vec<TransformerLayer>,
    },
    /// Gemma4 A4B with MoE.
    Gemma4 {
        layers: Vec<Gemma4Layer>,
        final_logit_softcapping: Option<f64>,
        hidden_size: usize,
    },
}

/// Full transformer forward pass for decoder-only models.
pub struct TransformerForward {
    embed: Embedding,
    variant: ModelVariant,
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
        let num_layers = metadata.num_layers as usize;
        let eps = Self::extract_rms_eps(metadata);

        // Detect naming convention: some models (e.g. Gemma 4 multimodal) use
        // "model.language_model.*" instead of "model.*".
        let model_prefix =
            if tensor_map
                .tensors
                .contains_key("model.language_model.embed_tokens.weight")
            {
                "model.language_model"
            } else {
                "model"
            };
        debug!(model_prefix, "Detected tensor naming prefix");

        // Helper to load a tensor by name
        let get = |name: &str| -> Result<Tensor> {
            let tref = tensor_map
                .tensors
                .get(name)
                .with_context(|| format!("Missing tensor: {}", name))?;
            tensor_from_ir(tref, device)
        };

        // Optional tensor loader (returns None if missing)
        let try_get = |name: &str| -> Option<Tensor> {
            tensor_map
                .tensors
                .get(name)
                .and_then(|tref| tensor_from_ir(tref, device).ok())
        };

        // Embedding
        let embed_weight = get(&format!("{}.embed_tokens.weight", model_prefix))?;
        let embed = Embedding::new(embed_weight, hidden_size);

        // Detect Gemma4 by model_type
        let is_gemma4 = metadata.model_type == "gemma4"
            || metadata.architecture.contains("Gemma4");

        if is_gemma4 {
            Self::load_gemma4(
                &get,
                &try_get,
                tensor_map,
                metadata,
                device,
                model_prefix,
                embed,
                hidden_size,
                num_layers,
                eps,
            )
        } else {
            Self::load_generic(
                &get,
                metadata,
                model_prefix,
                embed,
                hidden_size,
                num_layers,
                eps,
            )
        }
    }

    /// Load generic LLaMA-like transformer.
    fn load_generic(
        get: &dyn Fn(&str) -> Result<Tensor>,
        metadata: &ModelMetadata,
        model_prefix: &str,
        embed: Embedding,
        hidden_size: usize,
        num_layers: usize,
        eps: f64,
    ) -> Result<Self> {
        let num_heads = metadata.num_attention_heads as usize;
        let num_kv_heads =
            metadata.num_kv_heads.unwrap_or(metadata.num_attention_heads) as usize;
        let head_dim = hidden_size / num_heads;

        debug!(
            hidden_size,
            num_heads, num_kv_heads, head_dim, num_layers, "Loading generic transformer"
        );

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("{}.layers.{}", model_prefix, i);

            let attn_norm =
                RmsNorm::new(get(&format!("{}.input_layernorm.weight", prefix))?, eps);

            let q_proj =
                Self::load_linear(get, &format!("{}.self_attn.q_proj", prefix))?;
            let k_proj =
                Self::load_linear(get, &format!("{}.self_attn.k_proj", prefix))?;
            let v_proj =
                Self::load_linear(get, &format!("{}.self_attn.v_proj", prefix))?;
            let o_proj =
                Self::load_linear(get, &format!("{}.self_attn.o_proj", prefix))?;

            let ffn_norm = RmsNorm::new(
                get(&format!("{}.post_attention_layernorm.weight", prefix))?,
                eps,
            );

            let gate_proj =
                Self::load_linear(get, &format!("{}.mlp.gate_proj", prefix))?;
            let up_proj =
                Self::load_linear(get, &format!("{}.mlp.up_proj", prefix))?;
            let down_proj =
                Self::load_linear(get, &format!("{}.mlp.down_proj", prefix))?;

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
        let final_norm =
            RmsNorm::new(get(&format!("{}.norm.weight", model_prefix))?, eps);

        // LM head -- some models tie embed and lm_head weights
        let lm_head_weight = if tensor_map_has_key_prefix(get, "lm_head.weight") {
            get("lm_head.weight")?
        } else {
            debug!("lm_head.weight not found, using tied embed_tokens.weight");
            get(&format!("{}.embed_tokens.weight", model_prefix))?
        };
        let lm_head = Linear::new(lm_head_weight, None);

        Ok(Self {
            embed,
            variant: ModelVariant::Generic { layers },
            final_norm,
            lm_head,
        })
    }

    /// Load Gemma4 A4B transformer with MoE.
    fn load_gemma4(
        get: &dyn Fn(&str) -> Result<Tensor>,
        try_get: &dyn Fn(&str) -> Option<Tensor>,
        tensor_map: &TensorMap,
        metadata: &ModelMetadata,
        device: &candle_core::Device,
        model_prefix: &str,
        embed: Embedding,
        hidden_size: usize,
        num_layers: usize,
        eps: f64,
    ) -> Result<Self> {
        let cfg = Gemma4Config::from_metadata(metadata)?;

        debug!(
            hidden_size,
            num_heads = cfg.num_attention_heads,
            head_dim = cfg.head_dim,
            global_head_dim = cfg.global_head_dim,
            num_experts = cfg.num_experts,
            top_k = cfg.top_k_experts,
            num_layers,
            "Loading Gemma4 transformer"
        );

        // Cap max seq len for RoPE tables
        let max_rope_len = 8192usize;

        // Build shared RoPE embeddings
        let rope_sliding = RotaryEmbedding::new_standard(
            cfg.head_dim,
            max_rope_len,
            cfg.rope_theta_sliding,
            device,
        )?;
        let rope_global = RotaryEmbedding::new_partial(
            cfg.global_head_dim,
            max_rope_len,
            cfg.rope_theta_global,
            cfg.partial_rotary_factor,
            device,
        )?;

        // Pre-compute sin/cos once, clone per layer
        let rope_sliding_sin = rope_sliding.sin.clone();
        let rope_sliding_cos = rope_sliding.cos.clone();
        let rope_sliding_rotary_dim = rope_sliding.rotary_dim;
        let rope_global_sin = rope_global.sin.clone();
        let rope_global_cos = rope_global.cos.clone();
        let rope_global_rotary_dim = rope_global.rotary_dim;

        let activation = Activation::Gelu;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("{}.layers.{}", model_prefix, i);
            let is_full = cfg.is_full_attention(i);
            let head_dim = cfg.head_dim_for_layer(i);
            let num_kv_heads = cfg.num_kv_heads_for_layer(i);
            let k_eq_v = is_full && cfg.attention_k_eq_v;

            // Select RoPE for this layer
            let rotary_emb = if is_full {
                RotaryEmbedding {
                    sin: rope_global_sin.clone(),
                    cos: rope_global_cos.clone(),
                    rotary_dim: rope_global_rotary_dim,
                }
            } else {
                RotaryEmbedding {
                    sin: rope_sliding_sin.clone(),
                    cos: rope_sliding_cos.clone(),
                    rotary_dim: rope_sliding_rotary_dim,
                }
            };

            // Attention projections
            let q_proj =
                Self::load_linear(get, &format!("{}.self_attn.q_proj", prefix))?;
            let k_proj =
                Self::load_linear(get, &format!("{}.self_attn.k_proj", prefix))?;
            let v_proj = if k_eq_v {
                // V is tied to K — create a dummy (never used in forward)
                let k_w = get(&format!("{}.self_attn.k_proj.weight", prefix))?;
                let k_b_key = format!("{}.self_attn.k_proj.bias", prefix);
                let k_b = try_get(&k_b_key);
                Linear::new(k_w, k_b)
            } else {
                Self::load_linear(get, &format!("{}.self_attn.v_proj", prefix))?
            };
            let o_proj =
                Self::load_linear(get, &format!("{}.self_attn.o_proj", prefix))?;

            // Q/K norms
            let q_norm = RmsNorm::new(
                get(&format!("{}.self_attn.q_norm.weight", prefix))?,
                eps,
            );
            let k_norm = RmsNorm::new(
                get(&format!("{}.self_attn.k_norm.weight", prefix))?,
                eps,
            );

            // Dense MLP
            let mlp_gate_proj =
                Self::load_linear(get, &format!("{}.mlp.gate_proj", prefix))?;
            let mlp_up_proj =
                Self::load_linear(get, &format!("{}.mlp.up_proj", prefix))?;
            let mlp_down_proj =
                Self::load_linear(get, &format!("{}.mlp.down_proj", prefix))?;

            // MoE block
            let router_proj =
                Self::load_linear(get, &format!("{}.router.proj", prefix))?;
            let router_scale =
                get(&format!("{}.router.scale", prefix))?;
            let per_expert_scale =
                get(&format!("{}.router.per_expert_scale", prefix))?;
            // Safetensors layout: [num_experts, intermediate*2, hidden] for gate_up
            //                    [num_experts, hidden, intermediate] for down
            let expert_gate_up = get(&format!(
                "{}.experts.gate_up_proj",
                prefix
            ))?;
            let expert_down = get(&format!(
                "{}.experts.down_proj",
                prefix
            ))?;

            let moe = MoeBlock {
                router_proj,
                router_scale,
                per_expert_scale,
                expert_gate_up,
                expert_down,
                num_experts: cfg.num_experts,
                top_k: cfg.top_k_experts,
                moe_intermediate_size: cfg.moe_intermediate_size,
                activation,
            };

            // All 7 norm layers
            let input_layernorm = RmsNorm::new(
                get(&format!("{}.input_layernorm.weight", prefix))?,
                eps,
            );
            let post_attention_layernorm = RmsNorm::new(
                get(&format!("{}.post_attention_layernorm.weight", prefix))?,
                eps,
            );
            let pre_feedforward_layernorm = RmsNorm::new(
                get(&format!("{}.pre_feedforward_layernorm.weight", prefix))?,
                eps,
            );
            let post_feedforward_layernorm = RmsNorm::new(
                get(&format!("{}.post_feedforward_layernorm.weight", prefix))?,
                eps,
            );
            let pre_feedforward_layernorm_2 = RmsNorm::new(
                get(&format!("{}.pre_feedforward_layernorm_2.weight", prefix))?,
                eps,
            );
            let post_feedforward_layernorm_1 = RmsNorm::new(
                get(&format!("{}.post_feedforward_layernorm_1.weight", prefix))?,
                eps,
            );
            let post_feedforward_layernorm_2 = RmsNorm::new(
                get(&format!("{}.post_feedforward_layernorm_2.weight", prefix))?,
                eps,
            );

            // Layer scalar (optional)
            let layer_scalar = try_get(&format!(
                "{}.layer_scalar.weight",
                prefix
            ));

            layers.push(Gemma4Layer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                num_heads: cfg.num_attention_heads,
                num_kv_heads,
                head_dim,
                k_eq_v,
                rotary_emb,
                mlp_gate_proj,
                mlp_up_proj,
                mlp_down_proj,
                moe,
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
                pre_feedforward_layernorm_2,
                post_feedforward_layernorm_1,
                post_feedforward_layernorm_2,
                layer_scalar,
                activation,
            });

            debug!("Loaded Gemma4 layer {}/{}", i + 1, num_layers);
        }

        // Final norm
        let final_norm =
            RmsNorm::new(get(&format!("{}.norm.weight", model_prefix))?, eps);

        // LM head — Gemma4 ties embed and lm_head
        let lm_head_weight =
            if tensor_map.tensors.contains_key("lm_head.weight") {
                get("lm_head.weight")?
            } else {
                debug!(
                    "lm_head.weight not found, using tied embed_tokens.weight"
                );
                get(&format!("{}.embed_tokens.weight", model_prefix))?
            };
        let lm_head = Linear::new(lm_head_weight, None);

        Ok(Self {
            embed,
            variant: ModelVariant::Gemma4 {
                layers,
                final_logit_softcapping: cfg.final_logit_softcapping,
                hidden_size: cfg.hidden_size,
            },
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
    fn forward_inner(
        &self,
        token_ids: &[u32],
        capture: bool,
    ) -> Result<ForwardOutput> {
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

        // Gemma4: scale embeddings by sqrt(hidden_size)
        let is_gemma4 = matches!(&self.variant, ModelVariant::Gemma4 { .. });
        if is_gemma4 {
            let hs = match &self.variant {
                ModelVariant::Gemma4 { hidden_size, .. } => *hidden_size,
                _ => unreachable!(),
            };
            hidden = (hidden * (hs as f64).sqrt())?;
        }

        // Collect activations if requested
        let mut hidden_states = if capture {
            Some(Vec::with_capacity(self.num_layers() + 1))
        } else {
            None
        };

        if let Some(ref mut hs) = hidden_states {
            hs.push(hidden.clone());
        }

        // Transformer layers
        match &self.variant {
            ModelVariant::Generic { layers } => {
                for layer in layers {
                    hidden = layer.forward(&hidden)?;
                    if let Some(ref mut hs) = hidden_states {
                        hs.push(hidden.clone());
                    }
                }
            }
            ModelVariant::Gemma4 { layers, .. } => {
                for layer in layers {
                    hidden = layer.forward(&hidden)?;
                    if let Some(ref mut hs) = hidden_states {
                        hs.push(hidden.clone());
                    }
                }
            }
        }

        // Final norm
        hidden = self.final_norm.forward(&hidden)?;

        // LM head: [seq_len, hidden] -> [seq_len, vocab_size]
        let logits = self.lm_head.forward(&hidden)?;

        // Gemma4: apply logit softcapping
        let logits = match &self.variant {
            ModelVariant::Gemma4 {
                final_logit_softcapping: Some(sc),
                ..
            } => {
                let sc_val = *sc;
                ((logits / sc_val)?.tanh()? * sc_val)?
            }
            _ => logits,
        };

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
        let rc = &metadata.raw_config;
        // Try text_config first (Gemma4 nests it)
        rc.get("text_config")
            .and_then(|tc| tc.get("rms_norm_eps"))
            .and_then(|v| v.as_f64())
            .or_else(|| rc.get("rms_norm_eps").and_then(|v| v.as_f64()))
            .unwrap_or(1e-6)
    }

    /// Number of transformer layers.
    #[allow(dead_code)]
    pub fn num_layers(&self) -> usize {
        match &self.variant {
            ModelVariant::Generic { layers } => layers.len(),
            ModelVariant::Gemma4 { layers, .. } => layers.len(),
        }
    }
}

/// Check if a key exists in the tensor map by attempting to load it.
/// Returns true if the get succeeds.
fn tensor_map_has_key_prefix(
    get: &dyn Fn(&str) -> Result<Tensor>,
    key: &str,
) -> bool {
    get(key).is_ok()
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
    fn test_unit_rms_norm() {
        let device = candle_core::Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let result = rms_norm_unit(&x).unwrap();

        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals.len(), 4);

        // Same as RMSNorm with weight=1
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

        // Test causal mask shape
        let attn = Tensor::zeros((2, 3, 3), DType::F32, &device).unwrap();
        let masked = apply_causal_mask(&attn, 3).unwrap();
        assert_eq!(masked.dims(), &[2, 3, 3]);

        // Upper triangle should be -inf
        let vals: Vec<f32> = masked
            .i(0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // Position (0,1) should be -inf
        assert!(vals[1].is_infinite() && vals[1] < 0.0);
        // Position (0,0) should be 0
        assert_eq!(vals[0], 0.0);
        // Position (1,1) should be 0
        assert_eq!(vals[4], 0.0);
    }

    #[test]
    fn test_activation_silu() {
        let device = candle_core::Device::Cpu;
        let x = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], (1, 3), &device).unwrap();
        let result = Activation::Silu.forward(&x).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // silu(0) = 0
        assert!((vals[0]).abs() < 1e-6);
    }

    #[test]
    fn test_activation_gelu() {
        let device = candle_core::Device::Cpu;
        let x = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], (1, 3), &device).unwrap();
        let result = Activation::Gelu.forward(&x).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // gelu(0) = 0
        assert!((vals[0]).abs() < 1e-6);
    }

    #[test]
    fn test_rotary_embedding_standard() {
        let device = candle_core::Device::Cpu;
        let rope = RotaryEmbedding::new_standard(4, 16, 10000.0, &device).unwrap();
        assert_eq!(rope.rotary_dim, 4);
    }

    #[test]
    fn test_rotary_embedding_partial() {
        let device = candle_core::Device::Cpu;
        // partial_rotary_factor=0.25 on head_dim=512 -> rotary_dim = 2 * floor(0.25 * 512 / 2) = 2 * 64 = 128
        let rope =
            RotaryEmbedding::new_partial(512, 16, 1000000.0, 0.25, &device).unwrap();
        assert_eq!(rope.rotary_dim, 128);
    }

    #[test]
    fn test_repeat_kv_heads_no_repeat() {
        let device = candle_core::Device::Cpu;
        let x = Tensor::zeros((4, 3, 8), DType::F32, &device).unwrap();
        let result = repeat_kv_heads(&x, 4, 4).unwrap();
        assert_eq!(result.dims(), &[4, 3, 8]);
    }

    #[test]
    fn test_repeat_kv_heads_with_repeat() {
        let device = candle_core::Device::Cpu;
        let x = Tensor::zeros((2, 3, 8), DType::F32, &device).unwrap();
        let result = repeat_kv_heads(&x, 8, 2).unwrap();
        assert_eq!(result.dims(), &[8, 3, 8]);
    }

    #[test]
    fn test_gemma4_config_layer_types() {
        let metadata = ModelMetadata {
            architecture: "Gemma4ForConditionalGeneration".to_string(),
            model_type: "gemma4".to_string(),
            param_count: 26_000_000_000,
            hidden_size: 2816,
            num_layers: 12,
            layer_types: vec![
                "sliding_attention".to_string(),
                "sliding_attention".to_string(),
                "sliding_attention".to_string(),
                "sliding_attention".to_string(),
                "sliding_attention".to_string(),
                "full_attention".to_string(),
                "sliding_attention".to_string(),
                "sliding_attention".to_string(),
                "sliding_attention".to_string(),
                "sliding_attention".to_string(),
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            num_attention_heads: 16,
            num_kv_heads: Some(8),
            vocab_size: 262144,
            dtype: "bfloat16".to_string(),
            shard_count: 1,
            num_experts: Some(128),
            top_k_experts: Some(8),
            intermediate_size: Some(2112),
            raw_config: serde_json::json!({
                "text_config": {
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "num_global_key_value_heads": 2,
                    "attention_k_eq_v": true,
                    "moe_intermediate_size": 704
                }
            }),
        };

        let cfg = Gemma4Config::from_metadata(&metadata).unwrap();
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.global_head_dim, 512);
        assert_eq!(cfg.num_global_kv_heads, 2);
        assert!(cfg.attention_k_eq_v);
        assert_eq!(cfg.moe_intermediate_size, 704);
        assert!(cfg.is_full_attention(5));
        assert!(!cfg.is_full_attention(0));
        assert_eq!(cfg.head_dim_for_layer(5), 512);
        assert_eq!(cfg.head_dim_for_layer(0), 256);
        assert_eq!(cfg.num_kv_heads_for_layer(5), 2);
        assert_eq!(cfg.num_kv_heads_for_layer(0), 8);
    }
}
