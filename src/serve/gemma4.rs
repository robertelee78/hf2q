//! Gemma 4 A4B model — decoder-only transformer with Mixture of Experts.
//!
//! Architecture: attention + dense MLP + SigMoE (128 experts, top-8) per layer.
//! Dual attention: sliding (head_dim=256) and global (head_dim=512) layers.
//! RoPE: standard for sliding, partial for global.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_core::quantized::QMatMul;
use candle_nn::{Embedding, Module};
use std::sync::Arc;

use super::config::Gemma4Config;
use super::gguf_loader::GgufModel;

const MODEL_DTYPE: DType = DType::F32;

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

    /// Phase 1b.10: Fused residual add + RmsNorm.
    /// Computes `sum = x + residual` then `norm(sum)`, returning both.
    fn forward_with_residual(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        let sum = (x + residual)?;
        let normed = self.forward(&sum)?;
        Ok((normed, sum))
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
    k: Tensor,
    v: Tensor,
    cache_size: usize,
    current_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
}

const KV_CACHE_INITIAL_SIZE: usize = 4096;

impl KvCache {
    fn new(
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let cache_size = match sliding_window {
            Some(w) => KV_CACHE_INITIAL_SIZE.min(w),
            None => KV_CACHE_INITIAL_SIZE,
        };
        let k = Tensor::zeros((1, num_kv_heads, cache_size, head_dim), MODEL_DTYPE, device)?;
        let v = Tensor::zeros((1, num_kv_heads, cache_size, head_dim), MODEL_DTYPE, device)?;
        Ok(Self { k, v, cache_size, current_len: 0, num_kv_heads, head_dim, sliding_window })
    }

    fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let new_len = k_new.dim(2)?;
        let needed = self.current_len + new_len;

        if needed > self.cache_size {
            let mut new_size = self.cache_size;
            while new_size < needed { new_size *= 2; }
            let device = self.k.device().clone();
            let new_k = Tensor::zeros((1, self.num_kv_heads, new_size, self.head_dim), MODEL_DTYPE, &device)?;
            let new_v = Tensor::zeros((1, self.num_kv_heads, new_size, self.head_dim), MODEL_DTYPE, &device)?;
            if self.current_len > 0 {
                let active_k = self.k.narrow(2, 0, self.current_len)?;
                let active_v = self.v.narrow(2, 0, self.current_len)?;
                self.k = new_k.slice_scatter(&active_k, 2, 0)?;
                self.v = new_v.slice_scatter(&active_v, 2, 0)?;
            } else {
                self.k = new_k;
                self.v = new_v;
            }
            self.cache_size = new_size;
        }

        self.k = self.k.slice_scatter(k_new, 2, self.current_len)?;
        self.v = self.v.slice_scatter(v_new, 2, self.current_len)?;
        self.current_len = needed;

        // Sliding window truncation — expose only the last W tokens for sliding layers.
        // Global layers (sliding_window=None) see the full history.
        let visible_start = match self.sliding_window {
            Some(w) if self.current_len > w => self.current_len - w,
            _ => 0,
        };
        let visible_len = self.current_len - visible_start;

        // slice_scatter on dim != 0 uses a transpose trick that leaves the
        // returned tensor with non-standard strides — the memory layout is
        // [seq, heads, 1, hd] but the shape is [1, heads, seq, hd], so the
        // position stride is heads*hd instead of hd. SDPA's vector kernel
        // assumes positions are contiguous (stride = hd), so we must return
        // a contiguous view. This copies only the active portion per step.
        let k_active = self.k.narrow(2, visible_start, visible_len)?.contiguous()?;
        let v_active = self.v.narrow(2, visible_start, visible_len)?.contiguous()?;
        Ok((k_active, v_active))
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.current_len = 0;
    }
}

// ---------------------------------------------------------------------------
// QLinear layer (quantized weights via candle QMatMul)
// ---------------------------------------------------------------------------

struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
}

impl QLinear {
    fn new(qmatmul: QMatMul, bias: Option<Tensor>) -> Self {
        Self { inner: qmatmul, bias }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Metal QMatMul kernels require F32 input; cast and cast back
        let in_dtype = x.dtype();
        let x_f32 = if in_dtype != DType::F32 { x.to_dtype(DType::F32)? } else { x.clone() };
        let out = self.inner.forward(&x_f32)?;
        let out = if in_dtype != DType::F32 { out.to_dtype(in_dtype)? } else { out };
        match &self.bias {
            Some(b) => out.broadcast_add(b).map_err(Into::into),
            None => Ok(out),
        }
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
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

        let q_raw = self.q_proj.forward(xs)?;
        let q = q_raw.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let k_raw = self.k_proj.forward(xs)?;
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

        // SDPA for single-token decode (native GQA); manual attention for prefill
        // (candle's SDPA full kernel exceeds 32KB threadgroup mem for head_dim=512).
        let attn_out = if q_len == 1 {
            candle_nn::ops::sdpa(&q, &k, &v, None, false, 1.0, 1.0)?
        } else {
            let k_exp = Self::repeat_kv(&k, self.num_heads / self.num_kv_heads)?;
            let v_exp = Self::repeat_kv(&v, self.num_heads / self.num_kv_heads)?;
            let attn_weights = q.matmul(&k_exp.transpose(D::Minus2, D::Minus1)?)?;
            let attn_weights = match mask {
                Some(m) => attn_weights.broadcast_add(m)?,
                None => attn_weights,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v_exp)?
        };

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
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        let fused = (gate * up)?;
        self.down_proj.forward(&fused)
    }
}

// ---------------------------------------------------------------------------
// MoE (Mixture of Experts) with Softmax Routing
// ---------------------------------------------------------------------------

struct MoeBlock {
    /// Router projection: hidden_size → num_experts
    router_proj: QLinear,
    /// Per-hidden-dim learned scale for router input
    router_scale: Tensor,
    /// Per-expert scale applied to selected weights after softmax
    #[allow(dead_code)]
    per_expert_scale: Tensor,
    /// Cached CPU copy of per_expert_scale (avoids GPU→CPU sync every forward)
    per_expert_scale_cpu: Vec<f32>,
    /// Per-expert gate_up QMatMul (TEST: QMatMul from byte-sliced 3D QTensor)
    expert_gate_up: Vec<QMatMul>,
    expert_down: Vec<QMatMul>,
    #[allow(dead_code)]
    num_experts: usize,
    top_k: usize,
    moe_intermediate_size: usize,
    hidden_size: usize,
}

impl MoeBlock {
    /// Forward pass. `x` is the pre-normed expert input, `router_input` is
    /// the raw residual for the router (router applies its own RMS norm).
    fn forward(&self, x: &Tensor, router_input: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = x.dims3()?;
        let x_flat = x.reshape((b_sz * seq_len, hidden))?;
        let router_flat = router_input.reshape((b_sz * seq_len, hidden))?;

        // Router: unit_rms_norm(residual) * learned_scale * (1/sqrt(hidden)) → project → softmax
        let router_normed = Attention::rms_norm_unit(&router_flat)?;
        let scale_factor = (self.hidden_size as f64).powf(-0.5);
        let router_scaled = (router_normed.broadcast_mul(&self.router_scale)? * scale_factor)?;
        let logits = self.router_proj.forward(&router_scaled)?; // [tokens, num_experts]

        // Softmax gating
        let probs = candle_nn::ops::softmax_last_dim(&logits)?; // [tokens, num_experts]

        // --- GPU-side top-k selection (avoids full probs GPU→CPU sync) ---
        let probs_f32 = probs.to_dtype(DType::F32)?;
        // arg_sort descending on GPU: first k indices are the top-k experts
        let sorted_indices = probs_f32.contiguous()?.arg_sort_last_dim(false)?;
        let top_k_indices = sorted_indices.narrow(D::Minus1, 0, self.top_k)?.contiguous()?;
        // Gather the top-k probabilities on GPU
        let top_k_probs = probs_f32.contiguous()?.gather(&top_k_indices, D::Minus1)?;
        // Normalize: divide by sum of top-k probs (on GPU)
        let top_k_sum = top_k_probs.sum_keepdim(D::Minus1)?;
        let top_k_weights = top_k_probs.broadcast_div(&top_k_sum)?;
        // Pull ONLY the tiny top-k results to CPU (top_k values per token, not 128)
        let top_k_indices_cpu: Vec<Vec<u32>> = top_k_indices.to_vec2()?;
        let top_k_weights_cpu: Vec<Vec<f32>> = top_k_weights.to_vec2()?;
        let per_expert_scale_cpu = &self.per_expert_scale_cpu;
        let num_tokens = b_sz * seq_len;
        let device = x.device();
        let dtype = x.dtype();
        let mut outputs: Vec<Tensor> = Vec::with_capacity(num_tokens);

        for tok_idx in 0..num_tokens {
            let token_vec = x_flat.narrow(0, tok_idx, 1)?; // [1, hidden], BF16

            // Cast input to F32 for QMatMul (Metal kernels require F32)
            let token_f32 = token_vec.to_dtype(DType::F32)?;

            // Accumulate weighted expert outputs for this token
            let mut combined = Tensor::zeros((1, hidden), DType::F32, device)?;
            for k in 0..self.top_k {
                let eid = top_k_indices_cpu[tok_idx][k] as usize;
                let w = top_k_weights_cpu[tok_idx][k] * per_expert_scale_cpu[eid];

                // gate_up: [1, hidden] @ W^T -> [1, intermediate*2]
                let gate_up_out = self.expert_gate_up[eid].forward(&token_f32)?;
                let gate = gate_up_out.narrow(1, 0, self.moe_intermediate_size)?;
                let up = gate_up_out.narrow(1, self.moe_intermediate_size, self.moe_intermediate_size)?;
                let gate_act = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;
                let fused = (gate_act * up)?;

                // down: [1, intermediate] @ W^T -> [1, hidden]
                let expert_out = self.expert_down[eid].forward(&fused)?;

                // Scalar weight
                let w_t = Tensor::new(&[w], device)?;
                combined = (combined + expert_out.broadcast_mul(&w_t)?)?;
            }

            // Cast back to original dtype (BF16)
            outputs.push(combined.to_dtype(dtype)?);
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
        let normed = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward(&normed, mask, seqlen_offset)?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        // Phase 1b.10: fused residual-add + pre-FFW norm
        let (normed, xs) = self.pre_feedforward_layernorm.forward_with_residual(&attn_out, xs)?;

        // 2. Dense MLP and MoE run in PARALLEL from the same residual
        let residual = &xs;

        // Dense MLP branch (uses the already-normed input from the fused op above)
        let mlp_out = self.mlp.forward(&normed)?;
        let mlp_normed = self.post_feedforward_layernorm_1.forward(&mlp_out)?;

        // MoE branch (router takes raw residual; experts take pre-normed residual)
        let normed_moe = self.pre_feedforward_layernorm_2.forward(&xs)?;
        let moe_out = self.moe.forward(&normed_moe, &xs)?;
        let moe_normed = self.post_feedforward_layernorm_2.forward(&moe_out)?;

        // Sum MLP and MoE outputs, apply final post-FFW norm
        let combined = (mlp_normed + moe_normed)?;
        let combined = self.post_feedforward_layernorm.forward(&combined)?;
        let xs = (residual + combined)?;

        // 3. Layer scalar
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
            let k_proj = load_qlinear(gguf, &format!("{}.attn_k", lp))?;
            let v_proj = if k_eq_v {
                // V is tied to K — create a dummy (never used in forward)
                QLinear::new(k_proj.inner.clone(), k_proj.bias.clone())
            } else {
                load_qlinear(gguf, &format!("{}.attn_v", lp))?
            };

            let attn = Attention {
                q_proj: load_qlinear(gguf, &format!("{}.attn_q", lp))?,
                k_proj,
                v_proj,
                o_proj: load_qlinear(gguf, &format!("{}.attn_output", lp))?,
                q_norm: load_rms_norm(gguf, &format!("{}.attn_q_norm", lp), cfg.rms_norm_eps)?,
                k_norm: load_rms_norm(gguf, &format!("{}.attn_k_norm", lp), cfg.rms_norm_eps)?,
                num_heads: cfg.num_attention_heads,
                num_kv_heads,
                head_dim,
                rotary_emb: rotary,
                kv_cache: KvCache::new(
                    num_kv_heads, head_dim, device,
                    if is_full { None } else { Some(cfg.sliding_window) },
                )?,
                k_eq_v,
            };

            // Dense MLP
            let mlp = Mlp {
                gate_proj: load_qlinear(gguf, &format!("{}.ffn_gate", lp))?,
                up_proj: load_qlinear(gguf, &format!("{}.ffn_up", lp))?,
                down_proj: load_qlinear(gguf, &format!("{}.ffn_down", lp))?,
            };

            // MoE: load per-expert QMatMul from byte-sliced 3D QTensor.
            // GGUF stores expert weights as 3D: candle reads dims as [num_experts, out, in].
            // We extract each expert's raw quantized bytes and construct a 2D QMatMul.
            let gu_qt = gguf.get_qtensor(&format!("{}.ffn_gate_up_exps.weight", lp))?;
            let dn_qt = gguf.get_qtensor(&format!("{}.ffn_down_exps.weight", lp))?;

            if i == 0 {
                tracing::info!(
                    "Expert gate_up: {:?} {:?}, down: {:?} {:?}",
                    gu_qt.shape(), gu_qt.dtype(), dn_qt.shape(), dn_qt.dtype()
                );
            }

            let gu_data = gu_qt.data()?;
            let dn_data = dn_qt.data()?;
            let gu_dtype = gu_qt.dtype();
            let dn_dtype = dn_qt.dtype();
            let gu_shape = gu_qt.shape();
            let dn_shape = dn_qt.shape();
            let gu_bytes_per_expert = gu_data.len() / cfg.num_experts;
            let dn_bytes_per_expert = dn_data.len() / cfg.num_experts;
            // Per-expert 2D shape: [out_features, in_features]
            let gu_expert_shape = (gu_shape.dims()[1], gu_shape.dims()[2]);
            let dn_expert_shape = (dn_shape.dims()[1], dn_shape.dims()[2]);

            let mut expert_gate_up = Vec::with_capacity(cfg.num_experts);
            let mut expert_down = Vec::with_capacity(cfg.num_experts);
            for e in 0..cfg.num_experts {
                let gu_slice = &gu_data[e * gu_bytes_per_expert..(e + 1) * gu_bytes_per_expert];
                let gu_storage = candle_core::quantized::QStorage::from_data(
                    std::borrow::Cow::Borrowed(gu_slice), device, gu_dtype,
                )?;
                let gu_qtensor = Arc::new(candle_core::quantized::QTensor::new(
                    gu_storage, gu_expert_shape,
                )?);
                expert_gate_up.push(QMatMul::from_arc(gu_qtensor)?);

                let dn_slice = &dn_data[e * dn_bytes_per_expert..(e + 1) * dn_bytes_per_expert];
                let dn_storage = candle_core::quantized::QStorage::from_data(
                    std::borrow::Cow::Borrowed(dn_slice), device, dn_dtype,
                )?;
                let dn_qtensor = Arc::new(candle_core::quantized::QTensor::new(
                    dn_storage, dn_expert_shape,
                )?);
                expert_down.push(QMatMul::from_arc(dn_qtensor)?);
            }

            let moe = MoeBlock {
                router_proj: load_qlinear(gguf, &format!("{}.ffn_gate_inp", lp))?,
                router_scale: gguf.get_tensor(&format!("{}.ffn_gate_inp.scale", lp), MODEL_DTYPE)?,
                per_expert_scale: gguf.get_tensor(&format!("{}.ffn_down_exps.scale", lp), MODEL_DTYPE)?,
                per_expert_scale_cpu: gguf.get_tensor(&format!("{}.ffn_down_exps.scale", lp), MODEL_DTYPE)?
                    .to_dtype(DType::F32)?.to_vec1::<f32>()?,
                expert_gate_up,
                expert_down,
                num_experts: cfg.num_experts,
                top_k: cfg.top_k_experts,
                moe_intermediate_size: cfg.moe_intermediate_size,
                hidden_size: cfg.hidden_size,
            };

            let layer = DecoderLayer {
                self_attn: attn,
                mlp,
                moe,
                input_layernorm: load_rms_norm(gguf, &format!("{}.attn_norm", lp), cfg.rms_norm_eps)?,
                post_attention_layernorm: load_rms_norm(gguf, &format!("{}.post_attention_norm", lp), cfg.rms_norm_eps)?,
                pre_feedforward_layernorm: load_rms_norm(gguf, &format!("{}.ffn_norm", lp), cfg.rms_norm_eps)?,
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

        // Build causal mask for prefill (skip for single-token decode)
        let mask = if seq_len > 1 {
            Some(Self::causal_mask(seq_len, seqlen_offset, &self.device, xs.dtype())?)
        } else {
            None
        };

        // Transformer layers
        for (_i, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(&xs, mask.as_ref(), seqlen_offset)?;
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

    #[allow(dead_code)]
    fn has_nan(t: &Tensor) -> bool {
        t.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map(|v| v.iter().any(|x| x.is_nan()))
            .unwrap_or(true)
    }

    #[allow(dead_code)]
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

fn load_qlinear(gguf: &GgufModel, prefix: &str) -> Result<QLinear> {
    let qt = gguf.get_qtensor(&format!("{}.weight", prefix))?;
    let qmm = QMatMul::from_arc(qt)?;
    let bias = gguf.try_get_tensor(&format!("{}.bias", prefix), MODEL_DTYPE)?;
    Ok(QLinear::new(qmm, bias))
}

fn load_rms_norm(gguf: &GgufModel, prefix: &str, eps: f64) -> Result<RmsNorm> {
    let weight = gguf.get_tensor(&format!("{}.weight", prefix), MODEL_DTYPE)?;
    Ok(RmsNorm::new(weight, eps))
}

// ---------------------------------------------------------------------------
// Debug Forward-Pass Component Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod forward_tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_core::quantized::QMatMul;
    use std::path::Path;

    const GGUF_PATH: &str =
        "models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf";
    const CONFIG_PATH: &str = "models/gemma4/config.json";

    /// Helper: print first N f32 values from a tensor (flattened).
    fn first_n(t: &Tensor, n: usize) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map(|v| v.into_iter().take(n).collect())
            .unwrap_or_default()
    }

    /// Helper: max absolute difference between two tensors.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a_flat: Vec<f32> = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let b_flat: Vec<f32> = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        a_flat.iter().zip(b_flat.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn load_test_fixtures() -> (GgufModel, Gemma4Config, Device) {
        let device = Device::new_metal(0).expect("Metal device required for these tests");
        let gguf = GgufModel::load(Path::new(GGUF_PATH), &device)
            .expect("Failed to load GGUF model");
        let cfg = Gemma4Config::from_config_json(Path::new(CONFIG_PATH))
            .expect("Failed to load config");
        (gguf, cfg, device)
    }

    // -----------------------------------------------------------------------
    // Test 1: Embedding lookup + scale
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_01_embedding() {
        let (gguf, cfg, _device) = load_test_fixtures();

        let embed_w = gguf.get_tensor("token_embd.weight", MODEL_DTYPE).unwrap();
        // Transpose if GGUF stored as [hidden, vocab]
        let embed_w = if embed_w.dim(0).unwrap() == cfg.hidden_size
            && embed_w.dim(1).unwrap() == cfg.vocab_size
        {
            embed_w.t().unwrap().contiguous().unwrap()
        } else {
            embed_w
        };

        println!("\n=== Test 1: Embedding ===");
        println!("token_embd.weight shape: {:?}", embed_w.shape());

        // Look up token ID 2 (BOS)
        let embed = candle_nn::Embedding::new(embed_w, cfg.hidden_size);
        let token_ids = Tensor::new(&[2u32], embed.embeddings().device()).unwrap();
        let token_ids = token_ids.unsqueeze(0).unwrap(); // [1, 1]
        let emb = embed.forward(&token_ids).unwrap(); // [1, 1, hidden]

        let scale = (cfg.hidden_size as f64).sqrt();
        let scaled = (&emb * scale).unwrap();

        println!("BOS embedding (raw) first 5: {:?}", first_n(&emb, 5));
        println!("Scale factor: {}", scale);
        println!("BOS embedding (scaled) first 5: {:?}", first_n(&scaled, 5));
        println!("Embedding dtype: {:?}", scaled.dtype());
    }

    // -----------------------------------------------------------------------
    // Test 2: QMatMul vs dequantized matmul
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_02_qmatmul_vs_dequant() {
        let (gguf, _cfg, device) = load_test_fixtures();

        println!("\n=== Test 2: QMatMul vs Dequantize ===");

        // Load as QMatMul
        let qt = gguf.get_qtensor("blk.0.attn_q.weight").unwrap();
        let qmm = QMatMul::from_arc(qt).unwrap();

        // Load as dequantized tensor
        let deq_w = gguf.get_tensor("blk.0.attn_q.weight", DType::F32).unwrap();
        println!("Dequantized weight shape: {:?}", deq_w.shape());

        // Input: ones(1, 2816)
        let input = Tensor::ones((1, 2816), DType::F32, &device).unwrap();

        // QMatMul forward (does x @ W^T internally)
        let qmm_out = qmm.forward(&input).unwrap();
        println!("QMatMul output shape: {:?}", qmm_out.shape());
        println!("QMatMul first 5: {:?}", first_n(&qmm_out, 5));

        // Manual: input @ deq_w.T
        let manual_out = input.matmul(&deq_w.t().unwrap()).unwrap();
        println!("Dequant+matmul output shape: {:?}", manual_out.shape());
        println!("Dequant+matmul first 5: {:?}", first_n(&manual_out, 5));

        let diff = max_abs_diff(&qmm_out, &manual_out);
        println!("Max abs diff: {}", diff);
        assert!(diff < 1.0, "QMatMul vs dequant+matmul differ by more than 1.0: {}", diff);
    }

    // -----------------------------------------------------------------------
    // Test 3: RmsNorm
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_03_rmsnorm() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 3: RmsNorm ===");

        let norm_w = gguf.get_tensor("blk.0.attn_norm.weight", MODEL_DTYPE).unwrap();
        println!("attn_norm.weight shape: {:?}", norm_w.shape());
        println!("attn_norm.weight first 5: {:?}", first_n(&norm_w, 5));

        let norm = RmsNorm::new(norm_w, cfg.rms_norm_eps);

        // Input: ones(1, 2816)
        let input = Tensor::ones((1, 2816), MODEL_DTYPE, &device).unwrap();
        let output = norm.forward(&input).unwrap();
        println!("RmsNorm(ones) first 5: {:?}", first_n(&output, 5));
        println!("  (should equal the norm weight values since ones are normalized to 1.0)");

        // Input: arange for non-trivial test
        let arange = Tensor::arange(0u32, 2816, &device).unwrap()
            .to_dtype(DType::F32).unwrap()
            .unsqueeze(0).unwrap()
            .to_dtype(MODEL_DTYPE).unwrap();
        let output2 = norm.forward(&arange).unwrap();
        println!("RmsNorm(arange) first 5: {:?}", first_n(&output2, 5));
    }

    // -----------------------------------------------------------------------
    // Test 4: Single attention layer Q/K/V projections
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_04_attention_projections() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 4: Attention Projections (layer 0) ===");

        // Layer 0 is sliding attention
        let is_full = cfg.is_full_attention(0);
        let head_dim = cfg.head_dim_for_layer(0);
        let num_kv_heads = cfg.num_kv_heads_for_layer(0);
        println!("Layer 0: is_full={}, head_dim={}, num_heads={}, num_kv_heads={}",
            is_full, head_dim, cfg.num_attention_heads, num_kv_heads);

        // Load Q, K, V projections
        let q_proj = load_qlinear(&gguf, "blk.0.attn_q").unwrap();
        let k_proj = load_qlinear(&gguf, "blk.0.attn_k").unwrap();
        let v_proj = load_qlinear(&gguf, "blk.0.attn_v").unwrap();

        // Get the BOS embedding as input
        let embed_w = gguf.get_tensor("token_embd.weight", MODEL_DTYPE).unwrap();
        let embed_w = if embed_w.dim(0).unwrap() == cfg.hidden_size
            && embed_w.dim(1).unwrap() == cfg.vocab_size
        {
            embed_w.t().unwrap().contiguous().unwrap()
        } else {
            embed_w
        };
        let embed = candle_nn::Embedding::new(embed_w, cfg.hidden_size);
        let token_ids = Tensor::new(&[2u32], &device).unwrap().unsqueeze(0).unwrap();
        let emb = embed.forward(&token_ids).unwrap();
        let scale = (cfg.hidden_size as f64).sqrt();
        let input = (&emb * scale).unwrap(); // [1, 1, hidden]
        println!("Input (scaled BOS) shape: {:?}, first 5: {:?}", input.shape(), first_n(&input, 5));

        // Q projection
        let q_out = q_proj.forward(&input.squeeze(0).unwrap()).unwrap();
        let expected_q_dim = cfg.num_attention_heads * head_dim;
        println!("Q projection output shape: {:?} (expected [1, {}])", q_out.shape(), expected_q_dim);
        println!("Q first 5: {:?}", first_n(&q_out, 5));

        // K projection
        let k_out = k_proj.forward(&input.squeeze(0).unwrap()).unwrap();
        let expected_k_dim = num_kv_heads * head_dim;
        println!("K projection output shape: {:?} (expected [1, {}])", k_out.shape(), expected_k_dim);
        println!("K first 5: {:?}", first_n(&k_out, 5));

        // V projection
        let v_out = v_proj.forward(&input.squeeze(0).unwrap()).unwrap();
        println!("V projection output shape: {:?} (expected [1, {}])", v_out.shape(), expected_k_dim);
        println!("V first 5: {:?}", first_n(&v_out, 5));

        // Q/K norms
        let q_norm = load_rms_norm(&gguf, "blk.0.attn_q_norm", cfg.rms_norm_eps).unwrap();
        let k_norm = load_rms_norm(&gguf, "blk.0.attn_k_norm", cfg.rms_norm_eps).unwrap();

        // Reshape Q to [1, 1, num_heads, head_dim] for norm
        let q_reshaped = q_out.reshape((1, 1, cfg.num_attention_heads, head_dim)).unwrap();
        let q_normed = q_norm.forward(&q_reshaped).unwrap();
        println!("Q after norm, first 5: {:?}", first_n(&q_normed, 5));

        let k_reshaped = k_out.reshape((1, 1, num_kv_heads, head_dim)).unwrap();
        let k_normed = k_norm.forward(&k_reshaped).unwrap();
        println!("K after norm, first 5: {:?}", first_n(&k_normed, 5));
    }

    // -----------------------------------------------------------------------
    // Test 5: MoE router
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_05_moe_router() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 5: MoE Router (layer 0) ===");

        // Load router weights
        let router_proj = load_qlinear(&gguf, "blk.0.ffn_gate_inp").unwrap();
        let router_scale = gguf.get_tensor("blk.0.ffn_gate_inp.scale", MODEL_DTYPE).unwrap();
        println!("Router scale shape: {:?}, first 5: {:?}", router_scale.shape(), first_n(&router_scale, 5));

        let per_expert_scale = gguf.get_tensor("blk.0.ffn_down_exps.scale", MODEL_DTYPE).unwrap();
        println!("Per-expert scale shape: {:?}, first 5: {:?}", per_expert_scale.shape(), first_n(&per_expert_scale, 5));

        // Create a simple hidden state: ones(1, hidden)
        let input = Tensor::ones((1, cfg.hidden_size), MODEL_DTYPE, &device).unwrap();

        // Router pipeline: unit_rms_norm → scale → (1/sqrt(hidden)) → project → softmax
        let normed = Attention::rms_norm_unit(&input).unwrap();
        println!("After unit rms_norm first 5: {:?}", first_n(&normed, 5));

        let scale_factor = (cfg.hidden_size as f64).powf(-0.5);
        println!("Scale factor (1/sqrt({})): {}", cfg.hidden_size, scale_factor);

        let scaled = (normed.broadcast_mul(&router_scale).unwrap() * scale_factor).unwrap();
        println!("After scale first 5: {:?}", first_n(&scaled, 5));

        let logits = router_proj.forward(&scaled).unwrap();
        println!("Router logits shape: {:?}", logits.shape());
        println!("Router logits first 5: {:?}", first_n(&logits, 5));

        let probs = candle_nn::ops::softmax_last_dim(&logits).unwrap();
        println!("Router probs first 5: {:?}", first_n(&probs, 5));

        // Find top-8 experts
        let probs_vec: Vec<f32> = probs.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let per_expert_vec: Vec<f32> = per_expert_scale.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();

        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Top-8 experts:");
        let sum: f32 = indexed.iter().take(cfg.top_k_experts).map(|(_, w)| w).sum();
        for &(idx, weight) in indexed.iter().take(cfg.top_k_experts) {
            let normalized = weight / sum;
            let expert_scale = if idx < per_expert_vec.len() { per_expert_vec[idx] } else { 1.0 };
            println!("  Expert {}: raw_prob={:.6}, normalized={:.6}, per_expert_scale={:.6}",
                idx, weight, normalized, expert_scale);
        }
        println!("Sum of top-{} probs: {:.6}", cfg.top_k_experts, sum);
    }

    // -----------------------------------------------------------------------
    // Test 6: Expert forward (expert 0, layer 0)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_06_expert() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 6: Expert Forward (expert 0, layer 0) ===");

        // Load 3D expert weights
        let gate_up_3d = gguf.get_tensor("blk.0.ffn_gate_up_exps.weight", MODEL_DTYPE).unwrap();
        let down_3d = gguf.get_tensor("blk.0.ffn_down_exps.weight", MODEL_DTYPE).unwrap();

        println!("gate_up_exps raw shape (candle): {:?}", gate_up_3d.shape());
        println!("down_exps raw shape (candle): {:?}", down_3d.shape());

        // Slice expert 0, transpose to [in, out] for matmul
        let gate_up_e0 = gate_up_3d.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        println!("Expert 0 gate_up slice shape: {:?}", gate_up_e0.shape());
        let gate_up_w = gate_up_e0.t().unwrap().contiguous().unwrap();
        println!("Expert 0 gate_up (transposed) shape: {:?}", gate_up_w.shape());

        let down_e0 = down_3d.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        println!("Expert 0 down slice shape: {:?}", down_e0.shape());
        let down_w = down_e0.t().unwrap().contiguous().unwrap();
        println!("Expert 0 down (transposed) shape: {:?}", down_w.shape());

        // Input: ones(1, hidden)
        let input = Tensor::ones((1, cfg.hidden_size), MODEL_DTYPE, &device).unwrap();

        // gate_up = input @ gate_up_w
        let gate_up = input.matmul(&gate_up_w).unwrap();
        println!("gate_up output shape: {:?} (expected [1, {}])",
            gate_up.shape(), cfg.moe_intermediate_size * 2);
        println!("gate_up first 5: {:?}", first_n(&gate_up, 5));

        // Split into gate and up
        let gate = gate_up.narrow(1, 0, cfg.moe_intermediate_size).unwrap();
        let up = gate_up.narrow(1, cfg.moe_intermediate_size, cfg.moe_intermediate_size).unwrap();
        println!("gate shape: {:?}, first 5: {:?}", gate.shape(), first_n(&gate, 5));
        println!("up shape: {:?}, first 5: {:?}", up.shape(), first_n(&up, 5));

        // Apply GELU to gate, multiply by up
        let gate_act = candle_nn::Activation::GeluPytorchTanh.forward(&gate).unwrap();
        println!("gate after GELU first 5: {:?}", first_n(&gate_act, 5));

        let fused = (&gate_act * &up).unwrap();
        println!("fused (gate*up) shape: {:?}, first 5: {:?}", fused.shape(), first_n(&fused, 5));

        // down = fused @ down_w
        let down_out = fused.matmul(&down_w).unwrap();
        println!("down output shape: {:?} (expected [1, {}])", down_out.shape(), cfg.hidden_size);
        println!("down first 5: {:?}", first_n(&down_out, 5));

        // Check for NaN/Inf
        let flat: Vec<f32> = down_out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let nan_count = flat.iter().filter(|v| v.is_nan()).count();
        let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
        println!("NaN count: {}, Inf count: {}", nan_count, inf_count);
    }

    // -----------------------------------------------------------------------
    // Test 7: Full layer 0 forward pass (dense MLP branch only)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_07_dense_mlp() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 7: Dense MLP (layer 0) ===");

        // Load MLP weights
        let gate_proj = load_qlinear(&gguf, "blk.0.ffn_gate").unwrap();
        let up_proj = load_qlinear(&gguf, "blk.0.ffn_up").unwrap();
        let down_proj = load_qlinear(&gguf, "blk.0.ffn_down").unwrap();

        let mlp = Mlp { gate_proj, up_proj, down_proj };

        // Input: ones(1, hidden)
        let input = Tensor::ones((1, cfg.hidden_size), MODEL_DTYPE, &device).unwrap();

        let gate_out = mlp.gate_proj.forward(&input).unwrap();
        println!("gate_proj output shape: {:?}, first 5: {:?}", gate_out.shape(), first_n(&gate_out, 5));

        let up_out = mlp.up_proj.forward(&input).unwrap();
        println!("up_proj output shape: {:?}, first 5: {:?}", up_out.shape(), first_n(&up_out, 5));

        let mlp_out = mlp.forward(&input).unwrap();
        println!("Dense MLP output shape: {:?}, first 5: {:?}", mlp_out.shape(), first_n(&mlp_out, 5));

        // NaN/Inf check
        let flat: Vec<f32> = mlp_out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let nan_count = flat.iter().filter(|v| v.is_nan()).count();
        let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
        println!("NaN count: {}, Inf count: {}", nan_count, inf_count);
    }

    // -----------------------------------------------------------------------
    // Test 8: Layer scalar + norms pipeline
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_08_layer_scalar_and_norms() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 8: Layer Scalar and Post-Norms (layer 0) ===");

        let layer_scalar = gguf.get_tensor("blk.0.layer_output_scale.weight", MODEL_DTYPE).unwrap();
        println!("layer_scalar shape: {:?}, first 5: {:?}", layer_scalar.shape(), first_n(&layer_scalar, 5));

        let post_attn_norm = load_rms_norm(&gguf, "blk.0.post_attention_norm", cfg.rms_norm_eps).unwrap();
        let post_ffw_norm = load_rms_norm(&gguf, "blk.0.post_ffw_norm", cfg.rms_norm_eps).unwrap();
        let post_ffw_norm_1 = load_rms_norm(&gguf, "blk.0.post_ffw_norm_1", cfg.rms_norm_eps).unwrap();
        let post_ffw_norm_2 = load_rms_norm(&gguf, "blk.0.post_ffw_norm_2", cfg.rms_norm_eps).unwrap();

        // Test post-attention norm with ones
        let input = Tensor::ones((1, cfg.hidden_size), MODEL_DTYPE, &device).unwrap();
        let normed = post_attn_norm.forward(&input).unwrap();
        println!("post_attention_norm(ones) first 5: {:?}", first_n(&normed, 5));

        // Test layer scalar multiplication
        let scaled = input.broadcast_mul(&layer_scalar).unwrap();
        println!("ones * layer_scalar first 5: {:?}", first_n(&scaled, 5));

        // Print all post-FFW norm weights for comparison
        println!("post_ffw_norm weight first 5: {:?}", first_n(&post_ffw_norm.weight, 5));
        println!("post_ffw_norm_1 weight first 5: {:?}", first_n(&post_ffw_norm_1.weight, 5));
        println!("post_ffw_norm_2 weight first 5: {:?}", first_n(&post_ffw_norm_2.weight, 5));
    }

    // -----------------------------------------------------------------------
    // Test 9: End-to-end single token (BOS) through full model
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_09_single_token_e2e() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 9: Single Token (BOS) End-to-End ===");

        let mut model = Gemma4Model::load(&cfg, &gguf, &device)
            .expect("Failed to load full model");

        let input_ids = Tensor::new(&[2u32], &device).unwrap().unsqueeze(0).unwrap();
        println!("Input token IDs: [2] (BOS)");

        let logits = model.forward(&input_ids, 0).unwrap();
        println!("Logits shape: {:?}", logits.shape());

        let logits_flat: Vec<f32> = logits.to_dtype(DType::F32).unwrap()
            .flatten_all().unwrap().to_vec1().unwrap();

        println!("Logits first 5: {:?}", &logits_flat[..5.min(logits_flat.len())]);

        // Find top-5 token predictions
        let mut indexed: Vec<(usize, f32)> = logits_flat.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("Top-5 predicted tokens:");
        for &(idx, logit) in indexed.iter().take(5) {
            println!("  token {}: logit {:.4}", idx, logit);
        }

        // NaN/Inf check
        let nan_count = logits_flat.iter().filter(|v| v.is_nan()).count();
        let inf_count = logits_flat.iter().filter(|v| v.is_infinite()).count();
        println!("NaN count: {}, Inf count: {} (out of {})", nan_count, inf_count, logits_flat.len());
    }
}
