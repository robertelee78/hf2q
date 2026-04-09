//! Gemma 4 A4B model — decoder-only transformer with Mixture of Experts.
//!
//! Architecture: attention + dense MLP + SigMoE (128 experts, top-8) per layer.
//! Dual attention: sliding (head_dim=256) and global (head_dim=512) layers.
//! RoPE: standard for sliding, partial for global.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Embedding, Module};
use std::sync::Arc;

use super::config::Gemma4Config;
use super::gguf_loader::GgufModel;

const MODEL_DTYPE: DType = DType::BF16;

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Rotary Embedding
// ---------------------------------------------------------------------------

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new_standard(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        Self::build(head_dim, head_dim, max_seq_len, rope_theta, dev)
    }

    fn new_partial(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        dev: &Device,
    ) -> Result<Self> {
        let rope_angles = ((partial_rotary_factor * head_dim as f64 / 2.0).floor() as usize).min(head_dim / 2);
        let rotary_dim = rope_angles * 2;
        Self::build(head_dim, rotary_dim, max_seq_len, rope_theta, dev)
    }

    fn build(
        head_dim: usize,
        rotary_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let half = rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / rope_theta.powf(2.0 * i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        Ok(Self { sin, cos, rotary_dim })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, head_dim) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?.to_dtype(q.dtype())?;

        if self.rotary_dim == head_dim {
            let q_rot = Self::rope_apply(&q.contiguous()?, &cos, &sin)?;
            let k_rot = Self::rope_apply(&k.contiguous()?, &cos, &sin)?;
            Ok((q_rot, k_rot))
        } else {
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
// KV Cache (simple concat)
// ---------------------------------------------------------------------------

struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    #[allow(dead_code)]
    max_seq_len: usize,
}

impl KvCache {
    fn new(max_seq_len: usize) -> Self {
        Self { k: None, v: None, max_seq_len }
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k_out, v_out) = match (&self.k, &self.v) {
            (Some(pk), Some(pv)) => {
                let k_cat = Tensor::cat(&[pk, k], 2)?;
                let v_cat = Tensor::cat(&[pv, v], 2)?;
                (k_cat, v_cat)
            }
            _ => (k.clone(), v.clone()),
        };
        self.k = Some(k_out.clone());
        self.v = Some(v_out.clone());
        Ok((k_out, v_out))
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

// ---------------------------------------------------------------------------
// Linear layer (dequantized weights — MVP, no QMatMul yet)
// ---------------------------------------------------------------------------

struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // GGUF stores weights as [in_features, out_features]
        // Handle batched input: reshape [b, s, h] → [b*s, h], matmul, reshape back
        let dims = x.dims().to_vec();
        let result = if dims.len() == 3 {
            let (b, s, h) = (dims[0], dims[1], dims[2]);
            let flat = x.reshape((b * s, h))?;
            let out = flat.matmul(&self.weight)?;
            let out_dim = out.dim(1)?;
            out.reshape((b, s, out_dim))?
        } else {
            x.matmul(&self.weight)?
        };
        match &self.bias {
            Some(b) => result.broadcast_add(b).map_err(Into::into),
            None => Ok(result),
        }
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    k_eq_v: bool,
}

impl Attention {
    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        Gemma4Model::nan_check(xs, "    attn input")?;
        let q_raw = self.q_proj.forward(xs)?;
        Gemma4Model::nan_check(&q_raw, "    q_proj out")?;
        let q = q_raw.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let k_raw = self.k_proj.forward(xs)?;
        Gemma4Model::nan_check(&k_raw, "    k_proj out")?;
        let k = k_raw.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
        let v = if self.k_eq_v {
            k.clone()
        } else {
            self.v_proj.forward(xs)?
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
        };

        // Apply Q/K norms before transpose
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        // V gets a unit RMSNorm (just normalize, no learned weight)
        let v = Self::rms_norm_unit(&v)?.transpose(1, 2)?;

        // RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        // KV cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // GQA: repeat KV heads
        let k = Self::repeat_kv(&k, self.num_heads / self.num_kv_heads)?;
        let v = Self::repeat_kv(&v, self.num_heads / self.num_kv_heads)?;

        // Attention scores
        let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;

        // Apply mask
        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_out = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_out = attn_out
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_out)
    }

    fn repeat_kv(x: &Tensor, repeats: usize) -> Result<Tensor> {
        if repeats == 1 {
            return Ok(x.clone());
        }
        let (b, kv_heads, seq, dim) = x.dims4()?;
        let x = x.unsqueeze(2)?;
        let x = x.expand((b, kv_heads, repeats, seq, dim))?;
        x.reshape((b, kv_heads * repeats, seq, dim)).map_err(Into::into)
    }

    /// Unit RMSNorm (no learned weight, just normalize).
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

    #[allow(dead_code)]
    fn clear_cache(&mut self) {
        self.kv_cache.reset();
    }
}

// ---------------------------------------------------------------------------
// Dense MLP (SwiGLU)
// ---------------------------------------------------------------------------

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::Activation::Gelu.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        let fused = (gate * up)?;
        self.down_proj.forward(&fused)
    }
}

// ---------------------------------------------------------------------------
// MoE (Mixture of Experts) with Sigmoid Routing
// ---------------------------------------------------------------------------

struct MoeBlock {
    /// Router projection: hidden_size → num_experts
    router_proj: Linear,
    /// Per-hidden-dim scale applied before routing
    router_scale: Tensor,
    /// Per-expert scale/bias
    per_expert_scale: Tensor,
    /// Expert weights: gate_up [hidden, intermediate*2, num_experts] dequantized
    expert_gate_up: Tensor,
    /// Expert weights: down [intermediate, hidden, num_experts] dequantized
    expert_down: Tensor,
    #[allow(dead_code)]
    num_experts: usize,
    top_k: usize,
    moe_intermediate_size: usize,
}

impl MoeBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = x.dims3()?;
        let x_flat = x.reshape((b_sz * seq_len, hidden))?;

        // Router: h_scaled = h * scale, logits = h_scaled @ proj, weighted by per_expert_scale
        let x_scaled = x_flat.broadcast_mul(&self.router_scale)?;
        let logits = self.router_proj.forward(&x_scaled)?; // [tokens, num_experts]
        let logits = logits.broadcast_mul(&self.per_expert_scale)?;

        // Sigmoid gating (not softmax)
        let weights = candle_nn::ops::sigmoid(&logits)?; // [tokens, num_experts]

        // Process tokens — for MVP, loop over tokens
        let weights_cpu: Vec<Vec<f32>> = weights.to_dtype(DType::F32)?.to_vec2()?;
        let num_tokens = b_sz * seq_len;
        let device = x.device();
        let dtype = x.dtype();
        let mut outputs: Vec<Tensor> = Vec::with_capacity(num_tokens);

        for tok_idx in 0..num_tokens {
            let token_weights = &weights_cpu[tok_idx];

            // Select top-k experts
            let mut indexed: Vec<(usize, f32)> = token_weights.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(self.top_k);

            let sum: f32 = indexed.iter().map(|&(_, w)| w).sum();
            let token_vec = x_flat.narrow(0, tok_idx, 1)?; // [1, hidden]

            let mut combined = Tensor::zeros((1, hidden), dtype, device)?;

            for &(expert_idx, weight) in &indexed {
                // Expert gate_up_proj: [hidden, intermediate*2, num_experts]
                // Slice and squeeze to get [hidden, intermediate*2] as contiguous
                let gate_up_w = self.expert_gate_up.narrow(2, expert_idx, 1)?
                    .squeeze(2)?
                    .contiguous()?; // [hidden, intermediate*2]
                let gate_up = token_vec.matmul(&gate_up_w)?; // [1, intermediate*2]

                let gate = gate_up.narrow(1, 0, self.moe_intermediate_size)?;
                let up = gate_up.narrow(1, self.moe_intermediate_size, self.moe_intermediate_size)?;

                let gate_act = candle_nn::Activation::Gelu.forward(&gate)?;
                let fused = (gate_act * up)?;

                // down_proj: [intermediate, hidden, num_experts] → [intermediate, hidden]
                let down_w = self.expert_down.narrow(2, expert_idx, 1)?
                    .squeeze(2)?
                    .contiguous()?; // [intermediate, hidden]
                let expert_out = fused.matmul(&down_w)?; // [1, hidden]

                let scaled = (expert_out * (weight / sum) as f64)?;
                combined = (combined + scaled)?;
            }

            outputs.push(combined);
        }

        let result = Tensor::cat(&outputs, 0)?;
        result.reshape((b_sz, seq_len, hidden)).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

struct DecoderLayer {
    self_attn: Attention,
    // Dense MLP
    mlp: Mlp,
    // MoE
    moe: MoeBlock,
    // Norms
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    pre_feedforward_layernorm_2: RmsNorm,
    post_feedforward_layernorm_1: RmsNorm,
    post_feedforward_layernorm_2: RmsNorm,
    layer_scalar: Tensor,
}

impl DecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // 1. Attention block
        let residual = xs;
        let normed = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward(&normed, mask, seqlen_offset)?;
        Gemma4Model::nan_check(&attn_out, "  attn_out")?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let xs = (residual + attn_out)?;

        // 2. Dense MLP block
        let residual = &xs;
        let normed = self.pre_feedforward_layernorm.forward(&xs)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Gemma4Model::nan_check(&mlp_out, "  mlp_out")?;
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        let xs = (residual + mlp_out)?;

        // 3. MoE block
        let residual = &xs;
        let normed = self.pre_feedforward_layernorm_2.forward(&xs)?;
        let moe_out = self.moe.forward(&normed)?;
        Gemma4Model::nan_check(&moe_out, "  moe_out")?;
        let moe_out = self.post_feedforward_layernorm_1.forward(&moe_out)?;
        let moe_out = self.post_feedforward_layernorm_2.forward(&moe_out)?;
        let xs = (residual + moe_out)?;

        // 4. Layer scalar
        xs.broadcast_mul(&self.layer_scalar).map_err(Into::into)
    }

    #[allow(dead_code)]
    fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }
}

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

pub struct Gemma4Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head_weight: Tensor,
    hidden_size: usize,
    final_logit_softcapping: Option<f64>,
    device: Device,
}

impl Gemma4Model {
    /// Load model from GGUF + config.
    pub fn load(cfg: &Gemma4Config, gguf: &GgufModel, device: &Device) -> Result<Self> {
        // Embedding — GGUF stores as [hidden, vocab]; Candle Embedding wants [vocab, hidden]
        let embed_w = gguf.get_tensor("token_embd.weight", MODEL_DTYPE)?;
        let embed_w = if embed_w.dim(0)? == cfg.hidden_size && embed_w.dim(1)? == cfg.vocab_size {
            embed_w.t()?.contiguous()?
        } else {
            embed_w
        };
        let embed = Embedding::new(embed_w.clone(), cfg.hidden_size);

        // Rotary embeddings (shared across same-type layers)
        // Cap max seq len for RoPE tables to something reasonable for startup
        let max_rope_len = cfg.max_position_embeddings.min(8192);
        let rope_sliding = Arc::new(RotaryEmbedding::new_standard(
            cfg.head_dim, max_rope_len, cfg.rope_theta_sliding, device,
        )?);
        let rope_global = Arc::new(RotaryEmbedding::new_partial(
            cfg.global_head_dim, max_rope_len, cfg.rope_theta_global,
            cfg.partial_rotary_factor_global, device,
        )?);

        // Layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            eprint!("\r  Loading layer {}/{}...", i + 1, cfg.num_hidden_layers);
            let lp = format!("blk.{}", i);
            let is_full = cfg.is_full_attention(i);
            let head_dim = cfg.head_dim_for_layer(i);
            let num_kv_heads = cfg.num_kv_heads_for_layer(i);

            let rotary = if is_full { rope_global.clone() } else { rope_sliding.clone() };

            // Attention
            let k_eq_v = is_full && cfg.attention_k_eq_v;
            let k_proj = load_linear(gguf, &format!("{}.attn_k", lp))?;
            let v_proj = if k_eq_v {
                // V is tied to K — create a dummy (never used in forward)
                Linear::new(k_proj.weight.clone(), k_proj.bias.clone())
            } else {
                load_linear(gguf, &format!("{}.attn_v", lp))?
            };

            let attn = Attention {
                q_proj: load_linear(gguf, &format!("{}.attn_q", lp))?,
                k_proj,
                v_proj,
                o_proj: load_linear(gguf, &format!("{}.attn_output", lp))?,
                q_norm: load_rms_norm(gguf, &format!("{}.attn_q_norm", lp), cfg.rms_norm_eps)?,
                k_norm: load_rms_norm(gguf, &format!("{}.attn_k_norm", lp), cfg.rms_norm_eps)?,
                num_heads: cfg.num_attention_heads,
                num_kv_heads,
                head_dim,
                rotary_emb: rotary,
                kv_cache: KvCache::new(cfg.max_position_embeddings),
                k_eq_v,
            };

            // Dense MLP
            let mlp = Mlp {
                gate_proj: load_linear(gguf, &format!("{}.ffn_gate", lp))?,
                up_proj: load_linear(gguf, &format!("{}.ffn_up", lp))?,
                down_proj: load_linear(gguf, &format!("{}.ffn_down", lp))?,
            };

            // MoE
            let expert_gate_up = gguf.get_tensor(
                &format!("{}.ffn_gate_up_exps.weight", lp), MODEL_DTYPE,
            )?;
            let expert_down = gguf.get_tensor(
                &format!("{}.ffn_down_exps.weight", lp), MODEL_DTYPE,
            )?;
            let moe = MoeBlock {
                router_proj: load_linear(gguf, &format!("{}.ffn_gate_inp", lp))?,
                router_scale: gguf.get_tensor(&format!("{}.ffn_gate_inp.scale", lp), MODEL_DTYPE)?,
                per_expert_scale: gguf.get_tensor(&format!("{}.ffn_down_exps.scale", lp), MODEL_DTYPE)?,
                expert_gate_up,
                expert_down,
                num_experts: cfg.num_experts,
                top_k: cfg.top_k_experts,
                moe_intermediate_size: cfg.moe_intermediate_size,
            };

            let layer = DecoderLayer {
                self_attn: attn,
                mlp,
                moe,
                input_layernorm: load_rms_norm(gguf, &format!("{}.attn_norm", lp), cfg.rms_norm_eps)?,
                post_attention_layernorm: load_rms_norm(gguf, &format!("{}.ffn_norm", lp), cfg.rms_norm_eps)?,
                pre_feedforward_layernorm: load_rms_norm(gguf, &format!("{}.pre_ffw_norm", lp), cfg.rms_norm_eps)?,
                post_feedforward_layernorm: load_rms_norm(gguf, &format!("{}.post_ffw_norm", lp), cfg.rms_norm_eps)?,
                pre_feedforward_layernorm_2: load_rms_norm(gguf, &format!("{}.pre_ffw_norm_2", lp), cfg.rms_norm_eps)?,
                post_feedforward_layernorm_1: load_rms_norm(gguf, &format!("{}.post_ffw_norm_1", lp), cfg.rms_norm_eps)?,
                post_feedforward_layernorm_2: load_rms_norm(gguf, &format!("{}.post_ffw_norm_2", lp), cfg.rms_norm_eps)?,
                layer_scalar: gguf.get_tensor(&format!("{}.layer_output_scale.weight", lp), MODEL_DTYPE)?,
            };

            layers.push(layer);
        }
        eprintln!("\r  Loaded {}/{} layers.    ", cfg.num_hidden_layers, cfg.num_hidden_layers);

        // Final norm
        let norm = load_rms_norm(gguf, "output_norm", cfg.rms_norm_eps)?;

        // lm_head is tied to embed_tokens
        let lm_head_weight = embed_w;

        Ok(Self {
            embed_tokens: embed,
            layers,
            norm,
            lm_head_weight,
            hidden_size: cfg.hidden_size,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: device.clone(),
        })
    }

    /// Forward pass: [batch, seq_len] → logits [batch, 1, vocab_size].
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

        // Embed and scale
        let mut xs = self.embed_tokens.forward(input_ids)?;
        xs = (xs * (self.hidden_size as f64).sqrt())?;
        Self::nan_check(&xs, "post-embed")?;

        // Build causal mask for prefill (skip for single-token decode)
        let mask = if seq_len > 1 {
            Some(Self::causal_mask(seq_len, seqlen_offset, &self.device, xs.dtype())?)
        } else {
            None
        };

        // Transformer layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(&xs, mask.as_ref(), seqlen_offset)?;
            if i < 3 || Self::has_nan(&xs) {
                Self::nan_check(&xs, &format!("layer-{}", i))?;
            }
        }

        // Final norm + lm_head (last token only)
        let last_hidden = xs.narrow(1, seq_len - 1, 1)?;
        let normed = self.norm.forward(&last_hidden)?;
        // lm_head is tied to embed_tokens which is [vocab, hidden]
        // logits = normed @ lm_head.T  →  [1,1,hidden] @ [hidden, vocab]
        let normed_2d = normed.reshape((1, self.hidden_size))?;
        let logits = normed_2d.matmul(&self.lm_head_weight.t()?)?;
        let logits = logits.unsqueeze(0)?; // [1, 1, vocab]

        // Softcapping
        match self.final_logit_softcapping {
            Some(sc) => ((logits / sc)?.tanh()? * sc).map_err(Into::into),
            None => Ok(logits),
        }
    }

    #[allow(dead_code)]
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }

    fn has_nan(t: &Tensor) -> bool {
        t.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map(|v| v.iter().any(|x| x.is_nan()))
            .unwrap_or(true)
    }

    fn nan_check(t: &Tensor, label: &str) -> Result<()> {
        let flat = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let nan_count = flat.iter().filter(|v| v.is_nan()).count();
        let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
        let sample: Vec<f32> = flat.iter().take(5).copied().collect();
        eprintln!("[NAN] {}: shape={:?} nan={}/{} inf={} sample={:?}",
            label, t.shape(), nan_count, flat.len(), inf_count, sample);
        Ok(())
    }

    fn causal_mask(seq_len: usize, offset: usize, dev: &Device, dtype: DType) -> Result<Tensor> {
        let kv_len = seq_len + offset;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j > i + offset { f32::NEG_INFINITY } else { 0.0 }
                })
            })
            .collect();
        Tensor::from_vec(mask, (1, 1, seq_len, kv_len), dev)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_linear(gguf: &GgufModel, prefix: &str) -> Result<Linear> {
    let weight = gguf.get_tensor(&format!("{}.weight", prefix), MODEL_DTYPE)?;
    let bias = gguf.try_get_tensor(&format!("{}.bias", prefix), MODEL_DTYPE)?;
    Ok(Linear::new(weight, bias))
}

fn load_rms_norm(gguf: &GgufModel, prefix: &str, eps: f64) -> Result<RmsNorm> {
    let weight = gguf.get_tensor(&format!("{}.weight", prefix), MODEL_DTYPE)?;
    Ok(RmsNorm::new(weight, eps))
}
