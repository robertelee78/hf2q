//! Gemma 4 forward pass implementation.
//!
//! Assembles mlx-native GPU ops into a complete 30-layer transformer forward
//! pass for the Gemma 4 architecture (MoE, dual attention types, tied embeddings).
//!
//! ## Architecture Summary
//!
//! - 30 layers with two attention types (25 sliding, 5 global)
//! - Pattern: 5 sliding + 1 global, repeating
//! - Sliding: 16 Q heads, 8 KV heads, head_dim 256, window 1024, RoPE theta 10000
//! - Global: 16 Q heads, 2 KV heads, head_dim 512, RoPE theta 1000000, partial rotary 0.25
//! - MoE FFN: 128 experts, top-8, intermediate_size 704
//! - GELU pytorch_tanh activation
//! - RMS norm eps 1e-6
//! - Tied embeddings (embed_tokens = lm_head)
//! - Final logit softcap at 30.0

use mlx_native::ops::{
    elementwise, embedding, quantized_matmul as qmatmul,
    rms_norm, sdpa, sdpa_sliding, softcap,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice, QuantizedMatmulParams};
use tracing::{debug, info};

use crate::inference::kv_cache::{CacheType, KvCache, KvCacheError};
use crate::inference::weight_loader::{LoadedWeight, WeightMap};

/// Errors from the Gemma 4 forward pass.
#[derive(Debug, thiserror::Error)]
pub enum Gemma4Error {
    #[error("Missing weight tensor: {name}")]
    MissingWeight { name: String },

    #[error("GPU kernel error: {0}")]
    GpuError(#[from] mlx_native::MlxError),

    #[error("KV cache error: {0}")]
    KvCacheError(#[from] KvCacheError),

    #[error("Configuration error: {reason}")]
    ConfigError { reason: String },

    #[error("Forward pass error: {reason}")]
    ForwardError { reason: String },
}

/// Attention layer type for Gemma 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerAttentionType {
    /// Sliding window attention: 8 KV heads, head_dim 256, window 1024.
    Sliding,
    /// Global (full) attention: 2 KV heads, head_dim 512.
    Global,
}

/// Parsed Gemma 4 model configuration.
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    /// Number of transformer layers (30).
    pub num_layers: usize,
    /// Hidden size (2816).
    pub hidden_size: usize,
    /// Vocabulary size (262144).
    pub vocab_size: usize,
    /// Number of query attention heads (16).
    pub num_attention_heads: usize,
    /// Number of KV heads for sliding layers (8).
    pub num_kv_heads_sliding: usize,
    /// Number of KV heads for global layers (2).
    pub num_kv_heads_global: usize,
    /// Head dimension for sliding layers (256).
    pub head_dim_sliding: usize,
    /// Head dimension for global layers (512).
    pub head_dim_global: usize,
    /// Sliding window size (1024).
    pub sliding_window: usize,
    /// RoPE theta for sliding layers (10000.0).
    pub rope_theta_sliding: f32,
    /// RoPE theta for global layers (1000000.0).
    pub rope_theta_global: f32,
    /// Partial rotary factor for global attention (0.25).
    pub partial_rotary_factor_global: f32,
    /// Number of MoE experts (128).
    pub num_experts: usize,
    /// Number of experts selected per token (8).
    pub top_k_experts: usize,
    /// MoE intermediate size per expert (704).
    pub moe_intermediate_size: usize,
    /// Dense FFN intermediate size (2112, for non-MoE layers if any).
    pub intermediate_size: usize,
    /// RMS norm epsilon (1e-6).
    pub rms_norm_eps: f32,
    /// Final logit softcap value (30.0).
    pub final_logit_softcapping: f32,
    /// Per-layer attention type.
    pub layer_types: Vec<LayerAttentionType>,
    /// Whether K and V share projection (attention_k_eq_v).
    pub attention_k_eq_v: bool,
    /// Whether word embeddings are tied to lm_head.
    pub tie_word_embeddings: bool,
    /// Maximum sequence length for global KV cache pre-allocation.
    pub max_seq_len: usize,
}

impl Gemma4Config {
    /// Parse Gemma 4 config from the raw JSON value in `ModelConfig`.
    pub fn from_model_config(
        raw: &serde_json::Value,
    ) -> Result<Self, Gemma4Error> {
        // Helper to get from text_config or root
        let text_cfg = raw.get("text_config");
        let get = |key: &str| -> Option<&serde_json::Value> {
            text_cfg.and_then(|tc| tc.get(key)).or_else(|| raw.get(key))
        };
        let get_u64 = |key: &str| -> Option<u64> { get(key).and_then(|v| v.as_u64()) };
        let get_f64 = |key: &str| -> Option<f64> { get(key).and_then(|v| v.as_f64()) };
        let get_bool = |key: &str| -> Option<bool> { get(key).and_then(|v| v.as_bool()) };

        let num_layers = get_u64("num_hidden_layers").ok_or_else(|| Gemma4Error::ConfigError {
            reason: "Missing num_hidden_layers".into(),
        })? as usize;

        let hidden_size = get_u64("hidden_size").ok_or_else(|| Gemma4Error::ConfigError {
            reason: "Missing hidden_size".into(),
        })? as usize;

        let vocab_size = get_u64("vocab_size").ok_or_else(|| Gemma4Error::ConfigError {
            reason: "Missing vocab_size".into(),
        })? as usize;

        let num_attention_heads =
            get_u64("num_attention_heads").ok_or_else(|| Gemma4Error::ConfigError {
                reason: "Missing num_attention_heads".into(),
            })? as usize;

        let num_kv_heads_sliding =
            get_u64("num_key_value_heads").ok_or_else(|| Gemma4Error::ConfigError {
                reason: "Missing num_key_value_heads".into(),
            })? as usize;

        let num_kv_heads_global =
            get_u64("num_global_key_value_heads").unwrap_or(num_kv_heads_sliding as u64) as usize;

        let head_dim_sliding = get_u64("head_dim").unwrap_or(256) as usize;
        let head_dim_global = get_u64("global_head_dim").unwrap_or(512) as usize;

        let sliding_window = get_u64("sliding_window").unwrap_or(1024) as usize;

        // Parse rope parameters from the nested structure
        let rope_params = get("rope_parameters");
        let rope_theta_sliding = rope_params
            .and_then(|rp| rp.get("sliding_attention"))
            .and_then(|sa| sa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32;

        let rope_theta_global = rope_params
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1000000.0) as f32;

        let partial_rotary_factor_global = rope_params
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let num_experts = get_u64("num_experts").unwrap_or(128) as usize;
        let top_k_experts = get_u64("top_k_experts")
            .or_else(|| get_u64("num_experts_per_tok"))
            .unwrap_or(8) as usize;

        let moe_intermediate_size = get_u64("moe_intermediate_size").unwrap_or(704) as usize;
        let intermediate_size = get_u64("intermediate_size").unwrap_or(2112) as usize;

        let rms_norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-6) as f32;
        let final_logit_softcapping =
            get_f64("final_logit_softcapping").unwrap_or(30.0) as f32;

        let attention_k_eq_v = get_bool("attention_k_eq_v").unwrap_or(false);
        let tie_word_embeddings = raw
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .or_else(|| get_bool("tie_word_embeddings"))
            .unwrap_or(true);

        // Parse layer_types array
        let layer_types_json = get("layer_types").and_then(|v| v.as_array());
        let layer_types = if let Some(arr) = layer_types_json {
            arr.iter()
                .map(|v| {
                    let s = v.as_str().unwrap_or("sliding_attention");
                    match s {
                        "full_attention" | "global_attention" => LayerAttentionType::Global,
                        _ => LayerAttentionType::Sliding,
                    }
                })
                .collect()
        } else {
            // Default pattern: 5 sliding + 1 global, repeating
            (0..num_layers)
                .map(|i| {
                    if (i + 1) % 6 == 0 {
                        LayerAttentionType::Global
                    } else {
                        LayerAttentionType::Sliding
                    }
                })
                .collect()
        };

        // Max sequence length for global cache: use a reasonable default
        // (can be overridden at inference time)
        let max_seq_len =
            get_u64("max_position_embeddings").unwrap_or(8192) as usize;

        Ok(Self {
            num_layers,
            hidden_size,
            vocab_size,
            num_attention_heads,
            num_kv_heads_sliding,
            num_kv_heads_global,
            head_dim_sliding,
            head_dim_global,
            sliding_window,
            rope_theta_sliding,
            rope_theta_global,
            partial_rotary_factor_global,
            num_experts,
            top_k_experts,
            moe_intermediate_size,
            intermediate_size,
            rms_norm_eps,
            final_logit_softcapping,
            layer_types,
            attention_k_eq_v,
            tie_word_embeddings,
            max_seq_len,
        })
    }

    /// Get the attention type for a specific layer.
    pub fn layer_type(&self, layer_idx: usize) -> LayerAttentionType {
        self.layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerAttentionType::Sliding)
    }

    /// Get the number of KV heads for a specific layer.
    pub fn n_kv_heads(&self, layer_idx: usize) -> usize {
        match self.layer_type(layer_idx) {
            LayerAttentionType::Sliding => self.num_kv_heads_sliding,
            LayerAttentionType::Global => self.num_kv_heads_global,
        }
    }

    /// Get the head dimension for a specific layer.
    pub fn head_dim(&self, layer_idx: usize) -> usize {
        match self.layer_type(layer_idx) {
            LayerAttentionType::Sliding => self.head_dim_sliding,
            LayerAttentionType::Global => self.head_dim_global,
        }
    }

    /// Get the RoPE theta for a specific layer.
    pub fn rope_theta(&self, layer_idx: usize) -> f32 {
        match self.layer_type(layer_idx) {
            LayerAttentionType::Sliding => self.rope_theta_sliding,
            LayerAttentionType::Global => self.rope_theta_global,
        }
    }

    /// Get the number of dimensions that receive RoPE for a specific layer.
    ///
    /// For sliding layers, all head_dim dimensions are rotated.
    /// For global layers, only partial_rotary_factor * head_dim are rotated.
    pub fn rope_dim(&self, layer_idx: usize) -> usize {
        match self.layer_type(layer_idx) {
            LayerAttentionType::Sliding => self.head_dim_sliding,
            LayerAttentionType::Global => {
                (self.head_dim_global as f32 * self.partial_rotary_factor_global) as usize
            }
        }
    }
}

/// Gemma 4 model holding weights, KV cache, and configuration.
pub struct Gemma4Model {
    /// Loaded model weights.
    weights: WeightMap,
    /// Pre-allocated KV cache.
    kv_cache: KvCache,
    /// Model configuration.
    config: Gemma4Config,
    /// Metal device for buffer allocation and command dispatch.
    device: MlxDevice,
    /// Kernel registry for shader compilation.
    registry: KernelRegistry,
}

impl Gemma4Model {
    /// Build the model from loaded weights and a parsed config.
    ///
    /// Pre-allocates the KV cache for all 30 layers.
    pub fn from_weights(
        weights: WeightMap,
        config: Gemma4Config,
        device: MlxDevice,
    ) -> Result<Self, Gemma4Error> {
        // Validate we have the embedding weight
        require_weight(&weights, "model.embed_tokens.weight")?;

        // Validate we have the final norm weight
        require_weight(&weights, "model.norm.weight")?;

        // Build KV cache layout
        let mut cache_types = Vec::with_capacity(config.num_layers);
        let mut kv_configs = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            let (cache_type, n_kv, hd) = match config.layer_type(i) {
                LayerAttentionType::Sliding => (
                    CacheType::Sliding {
                        window: config.sliding_window,
                    },
                    config.num_kv_heads_sliding,
                    config.head_dim_sliding,
                ),
                LayerAttentionType::Global => (
                    CacheType::Global {
                        max_seq_len: config.max_seq_len,
                    },
                    config.num_kv_heads_global,
                    config.head_dim_global,
                ),
            };
            cache_types.push(cache_type);
            kv_configs.push((n_kv, hd));
        }

        let kv_cache = KvCache::new(&device, &cache_types, &kv_configs)?;

        info!(
            num_layers = config.num_layers,
            hidden_size = config.hidden_size,
            vocab_size = config.vocab_size,
            num_experts = config.num_experts,
            "Gemma4Model initialized"
        );

        let registry = KernelRegistry::new();

        Ok(Self {
            weights,
            kv_cache,
            config,
            device,
            registry,
        })
    }

    /// Run a forward pass on a sequence of token IDs.
    ///
    /// Returns a flat f32 buffer of shape `[seq_len, vocab_size]` containing logits.
    ///
    /// Uses CommandEncoder batching: all ops for one layer are encoded into a
    /// single command buffer, then committed before proceeding to the next layer.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Gemma4Error> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(Gemma4Error::ForwardError {
                reason: "Empty token sequence".into(),
            });
        }

        let hidden_size = self.config.hidden_size;
        let diag = std::env::var("HF2Q_DIAG").is_ok();

        if diag {
            eprintln!("[DIAG] === Forward pass: seq_len={}, hidden_size={}, vocab_size={} ===",
                seq_len, hidden_size, self.config.vocab_size);
            eprintln!("[DIAG] Input tokens: {:?}", &tokens[..tokens.len().min(20)]);
        }

        // Step 1: Embedding gather (returns f32), then convert to bf16 pipeline
        debug!(seq_len = seq_len, "Embedding gather");
        let hidden_f32 = self.embedding_gather(tokens)?;

        if diag {
            let s: &[f32] = hidden_f32.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read embed: {e}"),
            })?;
            let n = s.len().min(5);
            eprintln!("[DIAG] After embedding_gather (first {}): {:?}", n, &s[..n]);
            let nonzero = s.iter().filter(|v| **v != 0.0).count();
            eprintln!("[DIAG]   total_elements={}, nonzero={}", s.len(), nonzero);
        }

        // Convert embedding output to bf16 for the bf16 pipeline
        let embed_f32_data = read_buffer_f32(&hidden_f32)?;
        let mut hidden = f32_vec_to_bf16_buffer(
            &self.device, &embed_f32_data, vec![seq_len, hidden_size],
        )?;

        // Step 2: Scale embeddings by sqrt(hidden_size) as Gemma convention
        // Gemma models scale embeddings by sqrt(hidden_size)
        // Operate on bf16 data: read as u16, convert to f32, scale, convert back
        let scale = (hidden_size as f32).sqrt();
        {
            let slice: &mut [u16] = hidden
                .as_mut_slice()
                .map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("Embedding scale bf16: {e}"),
                })?;
            for v in slice.iter_mut() {
                let f = half::bf16::from_bits(*v).to_f32() * scale;
                *v = half::bf16::from_f32(f).to_bits();
            }

            if diag {
                let n = slice.len().min(5);
                let diag_vals: Vec<f32> = slice[..n].iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
                eprintln!("[DIAG] After embedding scale bf16 (factor={:.4}), first {}: {:?}",
                    scale, n, diag_vals);
            }
        }

        // Step 3: Process each transformer layer
        for layer_idx in 0..self.config.num_layers {
            debug!(layer = layer_idx, "Processing transformer layer");
            hidden = self.forward_layer(layer_idx, &hidden, seq_len)?;

            if diag {
                let s = read_buffer_f32(&hidden)?;
                // Print L2 norm of last token's hidden state
                let last_start = (seq_len - 1) * hidden_size;
                let last_token = &s[last_start..last_start + hidden_size];
                let l2 = last_token.iter().map(|v| v * v).sum::<f32>().sqrt();
                let first5 = &last_token[..5.min(last_token.len())];
                let nan_count = s.iter().filter(|v| v.is_nan()).count();
                let layer_type = self.config.layer_type(layer_idx);
                eprintln!("[DIAG] After layer {:2} ({:?}): L2={:.4}, first5={:?}{}",
                    layer_idx, layer_type, l2, first5,
                    if nan_count > 0 { format!(", NaN={}", nan_count) } else { String::new() });
            }
        }

        // Step 4: Final RMS norm (bf16 in, bf16 out)
        debug!("Final RMS norm");
        let normed = self.apply_rms_norm(&hidden, "model.norm.weight", seq_len)?;

        if diag {
            let s = read_buffer_f32(&normed)?;
            let n = s.len().min(5);
            eprintln!("[DIAG] After final rms_norm, first {}: {:?}", n, &s[..n]);
            // Also last token's first 5 values (this is what gets projected to logits)
            if seq_len > 1 {
                let last_start = (seq_len - 1) * hidden_size;
                let last_end = (last_start + 5).min(s.len());
                eprintln!("[DIAG]   last token first 5: {:?}", &s[last_start..last_end]);
            }
        }

        // Step 5: lm_head (tied embeddings -- transpose embed_tokens and matmul)
        // lm_head takes bf16 input and returns f32 logits
        debug!("lm_head projection (tied embeddings)");
        let logits = self.lm_head_projection(&normed, seq_len)?;

        if diag {
            let s: &[f32] = logits.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read logits: {e}"),
            })?;
            // Print top-10 logits for the LAST token position
            let vocab_size = self.config.vocab_size;
            let last_logits_start = (seq_len - 1) * vocab_size;
            let last_logits = &s[last_logits_start..last_logits_start + vocab_size];
            let mut indexed: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[DIAG] After lm_head, top-10 logit (idx, val) for last token:");
            for (i, (idx, val)) in indexed.iter().take(10).enumerate() {
                eprintln!("[DIAG]   #{}: token_id={}, logit={:.6}", i, idx, val);
            }
            // Also print min/max/mean
            let min = last_logits.iter().copied().fold(f32::INFINITY, f32::min);
            let max = last_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean = last_logits.iter().sum::<f32>() / vocab_size as f32;
            let std_dev = (last_logits.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vocab_size as f32).sqrt();
            eprintln!("[DIAG]   logit stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", min, max, mean, std_dev);
        }

        // Step 6: Softcap (CPU path — the GPU softcap kernel produces
        // incorrect results for very large dispatch grids on some Metal
        // implementations; CPU is fast enough for this element-wise op).
        debug!(cap = self.config.final_logit_softcapping, "Softcap");
        let capped_logits = self.apply_softcap_cpu(&logits, seq_len)?;

        if diag {
            let s: &[f32] = capped_logits.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read capped logits: {e}"),
            })?;
            let vocab_size = self.config.vocab_size;
            let last_logits_start = (seq_len - 1) * vocab_size;
            let last_logits = &s[last_logits_start..last_logits_start + vocab_size];
            let mut indexed: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[DIAG] After softcap (cap={}), top-10 logit (idx, val) for last token:",
                self.config.final_logit_softcapping);
            for (i, (idx, val)) in indexed.iter().take(10).enumerate() {
                eprintln!("[DIAG]   #{}: token_id={}, logit={:.6}", i, idx, val);
            }
            let min = last_logits.iter().copied().fold(f32::INFINITY, f32::min);
            let max = last_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean = last_logits.iter().sum::<f32>() / vocab_size as f32;
            eprintln!("[DIAG]   capped stats: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);
        }

        // Read logits to CPU
        let output: &[f32] = capped_logits
            .as_slice()
            .map_err(|e| Gemma4Error::ForwardError {
                reason: format!("Read logits: {e}"),
            })?;

        Ok(output.to_vec())
    }

    /// Run a forward pass returning the final hidden states (after the last
    /// RMS norm but before the lm_head projection).
    ///
    /// Returns a flat f32 buffer of shape `[seq_len, hidden_size]`. This is
    /// used by the embeddings endpoint which needs the hidden representation
    /// rather than logits.
    pub fn forward_hidden_states(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Gemma4Error> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(Gemma4Error::ForwardError {
                reason: "Empty token sequence".into(),
            });
        }

        let hidden_size = self.config.hidden_size;

        // Step 1: Embedding gather (f32) -> bf16
        debug!(seq_len = seq_len, "Embedding gather (hidden states)");
        let hidden_f32 = self.embedding_gather(tokens)?;
        let mut hidden = f32_vec_to_bf16_buffer(
            &self.device,
            &read_buffer_f32(&hidden_f32)?,
            vec![seq_len, hidden_size],
        )?;

        // Step 2: Scale embeddings by sqrt(hidden_size) as Gemma convention (in bf16)
        let scale = (hidden_size as f32).sqrt();
        {
            let slice: &mut [u16] = hidden
                .as_mut_slice()
                .map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("Embedding scale bf16: {e}"),
                })?;
            for v in slice.iter_mut() {
                let f = half::bf16::from_bits(*v).to_f32() * scale;
                *v = half::bf16::from_f32(f).to_bits();
            }
        }

        // Step 3: Process each transformer layer (bf16 pipeline)
        for layer_idx in 0..self.config.num_layers {
            debug!(layer = layer_idx, "Processing transformer layer (hidden states)");
            hidden = self.forward_layer(layer_idx, &hidden, seq_len)?;
        }

        // Step 4: Final RMS norm (bf16 in, bf16 out)
        debug!("Final RMS norm (hidden states)");
        let normed = self.apply_rms_norm(&hidden, "model.norm.weight", seq_len)?;

        // Read bf16 hidden states to CPU as f32
        let output = read_buffer_f32(&normed)?;

        Ok(output)
    }

    /// Run a forward pass with vision token injection.
    ///
    /// This is the multimodal path: after computing text embeddings and scaling,
    /// the vision features are scattered into positions where `token_ids[i] == image_token_id`.
    /// The vision features are already projected to the text model's hidden dimension
    /// by the vision encoder.
    ///
    /// `vision_features` is a flat f32 buffer of shape `[num_vision_tokens, hidden_size]`
    /// where num_vision_tokens equals the count of image_token_id occurrences in `tokens`.
    ///
    /// Returns logits of shape `[seq_len, vocab_size]`.
    pub fn forward_with_vision(
        &mut self,
        tokens: &[u32],
        image_token_id: u32,
        vision_features: &[f32],
    ) -> Result<Vec<f32>, Gemma4Error> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(Gemma4Error::ForwardError {
                reason: "Empty token sequence".into(),
            });
        }

        let hidden_size = self.config.hidden_size;

        // Step 1: Embedding gather (replace image tokens with PAD to avoid OOV)
        let pad_token_id = 0u32; // PAD token
        let safe_tokens: Vec<u32> = tokens
            .iter()
            .map(|&t| if t == image_token_id { pad_token_id } else { t })
            .collect();
        debug!(seq_len = seq_len, "Embedding gather (with vision)");
        let hidden_f32 = self.embedding_gather(&safe_tokens)?;

        // Convert to bf16 and scale
        let mut hidden_data = read_buffer_f32(&hidden_f32)?;
        let scale = (hidden_size as f32).sqrt();
        for v in hidden_data.iter_mut() {
            *v *= scale;
        }

        // Step 3: Scatter vision features into image token positions (in f32 space)
        // The vision features are already in the text model's hidden dimension
        // and have been processed by the vision encoder + projection.
        {
            let mut vision_idx = 0usize;
            for (pos, &token_id) in tokens.iter().enumerate() {
                if token_id == image_token_id {
                    let src_offset = vision_idx * hidden_size;
                    let dst_offset = pos * hidden_size;
                    if src_offset + hidden_size <= vision_features.len() {
                        hidden_data[dst_offset..dst_offset + hidden_size]
                            .copy_from_slice(&vision_features[src_offset..src_offset + hidden_size]);
                    }
                    vision_idx += 1;
                }
            }
            debug!(
                vision_tokens = vision_idx,
                "Vision features scattered into sequence"
            );
        }

        // Convert to bf16 for the pipeline
        let mut hidden = f32_vec_to_bf16_buffer(
            &self.device,
            &hidden_data,
            vec![seq_len, hidden_size],
        )?;

        // Step 4: Process each transformer layer (bf16 pipeline)
        for layer_idx in 0..self.config.num_layers {
            debug!(layer = layer_idx, "Processing transformer layer");
            hidden = self.forward_layer(layer_idx, &hidden, seq_len)?;
        }

        // Step 5: Final RMS norm (bf16 in, bf16 out)
        debug!("Final RMS norm");
        let normed = self.apply_rms_norm(&hidden, "model.norm.weight", seq_len)?;

        // Step 6: lm_head (bf16 in, f32 logits out)
        debug!("lm_head projection (tied embeddings)");
        let logits = self.lm_head_projection(&normed, seq_len)?;

        // Step 7: Softcap (CPU path to avoid GPU kernel issues with large buffers)
        debug!(cap = self.config.final_logit_softcapping, "Softcap");
        let capped_logits = self.apply_softcap_cpu(&logits, seq_len)?;

        // Read logits to CPU
        let output: &[f32] = capped_logits
            .as_slice()
            .map_err(|e| Gemma4Error::ForwardError {
                reason: format!("Read logits: {e}"),
            })?;

        Ok(output.to_vec())
    }

    /// Reset the KV cache (call between different sequences).
    pub fn reset_cache(&mut self) {
        self.kv_cache.reset();
    }

    /// Restore the KV cache write positions to a previously valid point.
    ///
    /// Used by prompt caching: after finding that the first `position`
    /// tokens match a cached prefix, the KV cache is rewound so the next
    /// forward pass appends starting at `position`. The KV buffer data
    /// for positions `0..position` is already valid from the previous
    /// generation and is reused in-place (no copies needed).
    pub fn restore_cache_position(&mut self, position: usize) -> Result<(), Gemma4Error> {
        self.kv_cache
            .restore_all_positions(position)
            .map_err(|e| Gemma4Error::KvCacheError(e))
    }

    /// Total allocated byte size of all KV cache buffers.
    pub fn kv_cache_allocated_bytes(&self) -> usize {
        self.kv_cache.total_allocated_bytes()
    }

    /// Get a reference to the model config.
    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    /// Get a reference to the weight map.
    pub fn weights(&self) -> &WeightMap {
        &self.weights
    }

    // ---- Private implementation methods ----

    /// Gather embeddings for the given token IDs.
    ///
    /// Uses mlx-native `embedding_gather` for quantized embedding tables,
    /// or a CPU fallback for unquantized weights.
    fn embedding_gather(&mut self, tokens: &[u32]) -> Result<MlxBuffer, Gemma4Error> {
        let embed_weight = require_weight(&self.weights, "model.embed_tokens.weight")?;
        let hidden_size = self.config.hidden_size;
        let diag = std::env::var("HF2Q_DIAG").is_ok();

        if diag {
            eprintln!("[DIAG] embed_tokens.weight: dtype={}, shape={:?}, byte_len={}",
                embed_weight.buffer.dtype(), embed_weight.buffer.shape(), embed_weight.buffer.byte_len());
            if let Some(ref qmeta) = embed_weight.quant_meta {
                eprintln!("[DIAG]   quant_meta: bits={}, group_size={}", qmeta.bits, qmeta.group_size);
            } else {
                eprintln!("[DIAG]   quant_meta: None (unquantized)");
            }
        }

        // Check if embedding is quantized
        if let Some(ref qmeta) = embed_weight.quant_meta {
            if qmeta.bits == 4 || qmeta.bits == 6 {
                if diag {
                    eprintln!("[DIAG] Using QUANTIZED embedding gather (bits={})", qmeta.bits);
                }
                return self.quantized_embedding_gather(tokens, qmeta.bits, qmeta.group_size);
            }
        }

        // Unquantized path: CPU gather from the f32/f16 embedding table
        // Read embedding weight to CPU, gather rows, write to a new f32 buffer
        if diag {
            eprintln!("[DIAG] Using CPU embedding gather (unquantized)");
        }
        self.cpu_embedding_gather(tokens, hidden_size)
    }

    /// Quantized embedding gather via GPU kernel.
    fn quantized_embedding_gather(
        &mut self,
        tokens: &[u32],
        bits: u8,
        group_size: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let n_tokens = tokens.len();

        // Need: weight_packed, scales, biases, token_ids buffers
        let embed_packed =
            require_weight(&self.weights, "model.embed_tokens.weight")?;
        let embed_scales =
            require_weight(&self.weights, "model.embed_tokens.scales")?;
        let embed_biases =
            require_weight(&self.weights, "model.embed_tokens.biases")?;

        // Create token_ids buffer
        let mut token_buf = self.device.alloc_buffer(
            n_tokens * std::mem::size_of::<u32>(),
            DType::U32,
            vec![n_tokens],
        )?;
        {
            let slice: &mut [u32] = token_buf.as_mut_slice().map_err(|e| {
                Gemma4Error::ForwardError {
                    reason: format!("Token buffer write: {e}"),
                }
            })?;
            slice.copy_from_slice(tokens);
        }

        // Allocate output
        let output_bytes = n_tokens * hidden_size * std::mem::size_of::<f32>();
        let output = self.device.alloc_buffer(
            output_bytes,
            DType::F32,
            vec![n_tokens, hidden_size],
        )?;

        let params = embedding::EmbeddingGatherParams {
            embed_dim: hidden_size,
            group_size,
            bits,
            n_tokens,
        };

        let mut encoder = self.device.command_encoder()?;
        embedding::embedding_gather(
            &mut encoder,
            &mut self.registry,
            self.device.metal_device(),
            &embed_packed.buffer,
            &embed_scales.buffer,
            &embed_biases.buffer,
            &token_buf,
            &output,
            &params,
        )?;
        encoder.commit_and_wait()?;

        Ok(output)
    }

    /// CPU fallback embedding gather for unquantized weights.
    fn cpu_embedding_gather(
        &self,
        tokens: &[u32],
        hidden_size: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let embed_weight = require_weight(&self.weights, "model.embed_tokens.weight")?;
        let n_tokens = tokens.len();

        // Read the embedding weight as f32 (may need conversion from f16/bf16)
        let embed_data = read_weight_as_f32(embed_weight)?;

        // Gather rows
        let mut output_data = vec![0.0f32; n_tokens * hidden_size];
        for (i, &token) in tokens.iter().enumerate() {
            let token_idx = token as usize;
            let src_start = token_idx * hidden_size;
            let src_end = src_start + hidden_size;
            if src_end > embed_data.len() {
                return Err(Gemma4Error::ForwardError {
                    reason: format!(
                        "Token ID {} out of bounds for embedding table of size {}",
                        token_idx,
                        embed_data.len() / hidden_size
                    ),
                });
            }
            output_data[i * hidden_size..(i + 1) * hidden_size]
                .copy_from_slice(&embed_data[src_start..src_end]);
        }

        // Write to Metal buffer
        let mut buf = self.device.alloc_buffer(
            output_data.len() * std::mem::size_of::<f32>(),
            DType::F32,
            vec![n_tokens, hidden_size],
        )?;
        {
            let slice: &mut [f32] = buf.as_mut_slice().map_err(|e| {
                Gemma4Error::ForwardError {
                    reason: format!("Embedding output write: {e}"),
                }
            })?;
            slice.copy_from_slice(&output_data);
        }

        Ok(buf)
    }

    /// Process a single transformer layer.
    ///
    /// 1. Pre-attention RMS norm
    /// 2. Q/K/V projection via quantized_matmul
    /// 3. RoPE on Q and K
    /// 4. KV cache append
    /// 5. Attention (sdpa or sdpa_sliding)
    /// 6. Output projection
    /// 7. Residual add
    /// 8. Post-attention RMS norm
    /// 9. MoE FFN (gate -> dispatch -> residual add)
    fn forward_layer(
        &mut self,
        layer_idx: usize,
        input: &MlxBuffer,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let layer_type = self.config.layer_type(layer_idx);
        let n_heads = self.config.num_attention_heads;
        let n_kv_heads = self.config.n_kv_heads(layer_idx);
        let head_dim = self.config.head_dim(layer_idx);
        let diag = std::env::var("HF2Q_DIAG").is_ok() && layer_idx == 0;

        // --- Step 1: Pre-attention RMS norm ---
        let norm_name = format!(
            "model.layers.{layer_idx}.input_layernorm.weight"
        );
        let normed = self.apply_rms_norm(input, &norm_name, seq_len)?;

        if diag {
            let s = read_buffer_f32(&normed)?;
            let n = s.len().min(5);
            eprintln!("[DIAG] Layer 0 after rms_norm, first {}: {:?}", n, &s[..n]);
        }

        // --- Step 2: Q/K/V projections ---
        // For Gemma 4: Q has n_heads * head_dim output dims,
        // K and V each have n_kv_heads * head_dim output dims.
        // If attention_k_eq_v, K and V share the same projection weight.
        let q_out_dim = n_heads * head_dim;
        let kv_out_dim = n_kv_heads * head_dim;
        let eps = self.config.rms_norm_eps;

        let q_proj = self.quantized_projection(
            &normed,
            &format!("model.layers.{layer_idx}.self_attn.q_proj"),
            seq_len,
            hidden_size,
            q_out_dim,
        )?;

        if diag {
            let s = read_buffer_f32(&q_proj)?;
            let n = s.len().min(5);
            eprintln!("[DIAG] Layer 0 after Q proj (dim={}), first {}: {:?}", q_out_dim, n, &s[..n]);
        }

        let k_proj = self.quantized_projection(
            &normed,
            &format!("model.layers.{layer_idx}.self_attn.k_proj"),
            seq_len,
            hidden_size,
            kv_out_dim,
        )?;

        // K-eq-V applies to ALL layers when the config flag is set.
        // When attention_k_eq_v is true, V is derived from the K projection
        // (not a separate v_proj) for every layer type.
        let use_k_eq_v = self.config.attention_k_eq_v;

        let v_proj = if use_k_eq_v {
            // K == V: reuse the K projection output as V
            // Note: we need to create a separate buffer copy because they might
            // diverge after norms (k_norm vs v_norm, RoPE only on K)
            self.quantized_projection(
                &normed,
                &format!("model.layers.{layer_idx}.self_attn.k_proj"),
                seq_len,
                hidden_size,
                kv_out_dim,
            )?
        } else {
            self.quantized_projection(
                &normed,
                &format!("model.layers.{layer_idx}.self_attn.v_proj"),
                seq_len,
                hidden_size,
                kv_out_dim,
            )?
        };

        // --- Step 2b: QK norms (per-head RMS norm on Q and K) ---
        // Gemma 4 applies RMS norm per-head to Q and K after projection, before RoPE.
        // Weight shape is [head_dim], shared across all heads.
        let q_norm_name = format!("model.layers.{layer_idx}.self_attn.q_norm.weight");
        let k_norm_name = format!("model.layers.{layer_idx}.self_attn.k_norm.weight");

        let q_norm_weight = read_weight_as_f32(require_weight(&self.weights, &q_norm_name)?)?;
        let k_norm_weight = read_weight_as_f32(require_weight(&self.weights, &k_norm_name)?)?;

        // read_buffer_f32 handles bf16 -> f32 conversion automatically
        let q_data = read_buffer_f32(&q_proj)?;
        let q_normed_data = cpu_per_head_rms_norm(&q_data, &q_norm_weight, seq_len, n_heads, head_dim, eps);
        let q_normed = f32_vec_to_bf16_buffer(&self.device, &q_normed_data, vec![seq_len, q_out_dim])?;

        let k_data_raw = read_buffer_f32(&k_proj)?;
        let k_normed_data = cpu_per_head_rms_norm(&k_data_raw, &k_norm_weight, seq_len, n_kv_heads, head_dim, eps);
        let k_normed = f32_vec_to_bf16_buffer(&self.device, &k_normed_data, vec![seq_len, kv_out_dim])?;

        // --- Step 2c: V norm (scaleless per-head RMS norm on V) ---
        // Gemma 4 applies RMSNormNoScale to V (normalize without learned weight).
        let v_data_raw = read_buffer_f32(&v_proj)?;
        let v_normed_data = cpu_per_head_rms_norm_no_scale(&v_data_raw, seq_len, n_kv_heads, head_dim, eps);
        let v_normed_buf = f32_vec_to_bf16_buffer(&self.device, &v_normed_data, vec![seq_len, kv_out_dim])?;

        if diag {
            // Compare K and V raw projections (should be IDENTICAL for k_eq_v)
            eprintln!("[DIAG] Layer 0 K raw (before k_norm), pos 0 first 5: {:?}", &k_data_raw[..5.min(k_data_raw.len())]);
            eprintln!("[DIAG] Layer 0 V raw (before v_norm), pos 0 first 5: {:?}", &v_data_raw[..5.min(v_data_raw.len())]);
            let k_v_match = k_data_raw.len() == v_data_raw.len() &&
                k_data_raw.iter().zip(v_data_raw.iter()).take(10).all(|(a, b)| (a - b).abs() < 1e-6);
            eprintln!("[DIAG]   K==V? {} (k_eq_v={}, k_len={}, v_len={})", k_v_match, use_k_eq_v, k_data_raw.len(), v_data_raw.len());

            let n = q_normed_data.len().min(5);
            eprintln!("[DIAG] Layer 0 after QK norms, Q first {}: {:?}", n, &q_normed_data[..n]);
            let n = k_normed_data.len().min(5);
            eprintln!("[DIAG] Layer 0 after QK norms, K first {}: {:?}", n, &k_normed_data[..n]);

            // V diagnostic: raw projection and after norm
            eprintln!("[DIAG] Layer 0 V raw (k_proj, before v_norm), pos 0 first 5: {:?}", &v_data_raw[..5.min(v_data_raw.len())]);
            eprintln!("[DIAG] Layer 0 V normed (after v_norm_no_scale), pos 0 first 5: {:?}", &v_normed_data[..5.min(v_normed_data.len())]);
            // Compute RMS of V raw for head 0, pos 0 to verify norm
            let v_head0 = &v_data_raw[..head_dim];
            let v_rms = (v_head0.iter().map(|v| v * v).sum::<f32>() / head_dim as f32).sqrt();
            eprintln!("[DIAG]   V head 0 pos 0: RMS={:.6}, len={}, k_eq_v={}", v_rms, head_dim, use_k_eq_v);
        }

        // --- Step 3: RoPE on Q and K ---
        // For global attention with partial_rotary_factor, only apply RoPE to
        // the first rope_dim dimensions of each head, leaving the rest unchanged.
        let rope_dim = self.config.rope_dim(layer_idx);
        let theta = self.config.rope_theta(layer_idx);

        let q_roped = self.apply_rope_to_heads(
            &q_normed, seq_len, n_heads, head_dim, rope_dim, theta,
        )?;

        let k_roped = self.apply_rope_to_heads(
            &k_normed, seq_len, n_kv_heads, head_dim, rope_dim, theta,
        )?;

        // --- Step 4: KV cache append ---
        // Read K and V (bf16) to CPU as f32, append to cache (f32 storage)
        let k_data = read_buffer_f32(&k_roped)?;
        let v_data = read_buffer_f32(&v_normed_buf)?; // V does NOT get RoPE, but does get v_norm

        {
            let layer_cache = self.kv_cache.layer_mut(layer_idx)?;
            layer_cache.append(&k_data, &v_data)?;
        }

        // --- Step 5: Attention ---
        // Gemma 4 uses scale=1.0 for attention (QK norms handle the scaling).
        // However, the SDPA kernel hardcodes scale = 1/sqrt(head_dim).
        // To compensate, we pre-scale Q by sqrt(head_dim) so the kernel's
        // division cancels out, yielding an effective scale of 1.0.
        // Q is bf16; read as f32, scale, write back as bf16 for SDPA.
        let q_prescaled = {
            let q_data = read_buffer_f32(&q_roped)?;
            let sqrt_hd = (head_dim as f32).sqrt();
            let scaled: Vec<f32> = q_data.iter().map(|v| v * sqrt_hd).collect();
            f32_vec_to_bf16_buffer(&self.device, &scaled, vec![seq_len, n_heads * head_dim])?
        };

        let attn_out = self.compute_attention(
            layer_idx,
            &q_prescaled,
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            layer_type,
        )?;

        if diag {
            // Print V values before attention (what we stored in the cache)
            eprintln!("[DIAG] Layer 0 V stored in cache (first 5 of pos 0): {:?}", &v_data[..5.min(v_data.len())]);
            eprintln!("[DIAG]   V total elements: {}, expected: {}x{}={}", v_data.len(), seq_len, n_kv_heads * head_dim, seq_len * n_kv_heads * head_dim);

            // Read back from cache to verify
            let layer_cache_read = self.kv_cache.layer(layer_idx)?;
            let (_k_cb, v_cb, kv_slen) = layer_cache_read.keys_values();
            let v_cache_data: &[f32] = v_cb.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read v cache: {e}"),
            })?;
            let v_first5 = &v_cache_data[..5.min(v_cache_data.len())];
            eprintln!("[DIAG] Layer 0 V cache readback pos 0 first 5: {:?}, kv_seq_len={}", v_first5, kv_slen);

            // Print reshaped V that goes into SDPA
            let v_valid = &v_cache_data[..kv_slen * n_kv_heads * head_dim];
            let v_reshaped = reshape_qkv_for_sdpa(v_valid, kv_slen, n_kv_heads, head_dim);
            eprintln!("[DIAG] Layer 0 V reshaped for SDPA (head 0, pos 0, first 5): {:?}", &v_reshaped[..5.min(v_reshaped.len())]);

            // Print Q pre-scaled values
            let q_ps = read_buffer_f32(&q_prescaled)?;
            eprintln!("[DIAG] Layer 0 Q pre-scaled (pos 0, first 5): {:?}", &q_ps[..5.min(q_ps.len())]);

            let s = read_buffer_f32(&attn_out)?;
            let n = s.len().min(5);
            eprintln!("[DIAG] Layer 0 after attention output, first {}: {:?}", n, &s[..n]);
            let nan_count = s.iter().filter(|v| v.is_nan()).count();
            let inf_count = s.iter().filter(|v| v.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                eprintln!("[DIAG]   WARNING attn: NaN={}, Inf={}", nan_count, inf_count);
            }
        }

        // --- Step 6: Output projection ---
        let o_proj = self.quantized_projection(
            &attn_out,
            &format!("model.layers.{layer_idx}.self_attn.o_proj"),
            seq_len,
            n_heads * head_dim,
            hidden_size,
        )?;

        // --- Step 7: Post-attention layernorm and residual add ---
        // Python: h = self.post_attention_layernorm(attn_output)
        //         h = residual + h
        // The norm is applied to the attention output BEFORE adding the residual.
        let post_attn_norm_name = format!(
            "model.layers.{layer_idx}.post_attention_layernorm.weight"
        );
        let post_attn_normed = self.apply_rms_norm(&o_proj, &post_attn_norm_name, seq_len)?;
        let residual1 = self.elementwise_add(input, &post_attn_normed, seq_len * hidden_size)?;

        // --- Step 8: Feedforward with pre/post norms ---
        // Gemma 4 MoE layers have TWO parallel paths:
        //   Path 1 (dense MLP): pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm_1
        //   Path 2 (MoE experts): router(h), pre_feedforward_layernorm_2 -> experts -> post_feedforward_layernorm_2
        //   Combined: h = path1 + path2
        //   Final: post_feedforward_layernorm(combined)
        // Non-MoE layers: pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm

        // Check if this layer has MoE (router exists)
        let router_weight_name = format!("model.layers.{layer_idx}.router.proj.weight");
        let has_moe_router = self.weights.get(&router_weight_name).is_some()
            || self.weights.get(&format!("language_model.{}", router_weight_name)).is_some();

        let ffn_out = if has_moe_router {
            // --- MoE path: two parallel branches ---

            // Path 1: dense MLP
            let pre_ff_norm_name = format!(
                "model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
            );
            let pre_ff_normed = self.apply_rms_norm(&residual1, &pre_ff_norm_name, seq_len)?;
            let mlp_out = self.dense_ffn(layer_idx, &pre_ff_normed, seq_len)?;
            let post_ff_norm1_name = format!(
                "model.layers.{layer_idx}.post_feedforward_layernorm_1.weight"
            );
            let mlp_normed = self.apply_rms_norm(&mlp_out, &post_ff_norm1_name, seq_len)?;

            // Path 2: MoE experts
            let pre_ff_norm2_name = format!(
                "model.layers.{layer_idx}.pre_feedforward_layernorm_2.weight"
            );
            let pre_ff_normed2 = self.apply_rms_norm(&residual1, &pre_ff_norm2_name, seq_len)?;
            // Router gets the un-normed residual1 (it does its own internal norm)
            let moe_out = self.moe_ffn(layer_idx, &residual1, &pre_ff_normed2, seq_len)?;
            let post_ff_norm2_name = format!(
                "model.layers.{layer_idx}.post_feedforward_layernorm_2.weight"
            );
            let moe_normed = self.apply_rms_norm(&moe_out, &post_ff_norm2_name, seq_len)?;

            // Sum the two paths
            let combined = self.elementwise_add(&mlp_normed, &moe_normed, seq_len * hidden_size)?;

            // Final post-feedforward norm
            let post_ff_norm_name = format!(
                "model.layers.{layer_idx}.post_feedforward_layernorm.weight"
            );
            self.apply_rms_norm(&combined, &post_ff_norm_name, seq_len)?
        } else {
            // --- Non-MoE path: simple MLP with pre/post norms ---
            let pre_ff_norm_name = format!(
                "model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
            );
            let pre_ff_normed = self.apply_rms_norm(&residual1, &pre_ff_norm_name, seq_len)?;
            let mlp_out = self.dense_ffn(layer_idx, &pre_ff_normed, seq_len)?;
            let post_ff_norm_name = format!(
                "model.layers.{layer_idx}.post_feedforward_layernorm.weight"
            );
            self.apply_rms_norm(&mlp_out, &post_ff_norm_name, seq_len)?
        };

        if diag {
            let s = read_buffer_f32(&ffn_out)?;
            let n = s.len().min(5);
            eprintln!("[DIAG] Layer 0 after FFN+norms output, first {}: {:?}", n, &s[..n]);
            let nan_count = s.iter().filter(|v| v.is_nan()).count();
            let inf_count = s.iter().filter(|v| v.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                eprintln!("[DIAG]   WARNING ffn: NaN={}, Inf={}", nan_count, inf_count);
            }
        }

        // --- Step 9: Residual add ---
        let pre_scalar = self.elementwise_add(&residual1, &ffn_out, seq_len * hidden_size)?;

        // --- Step 10: Apply layer_scalar to the ENTIRE layer output ---
        // Python: h = h * self.layer_scalar (applied to the full output, not sub-parts)
        let output = self.apply_layer_scalar(layer_idx, &pre_scalar, seq_len * hidden_size)?;

        Ok(output)
    }

    /// Apply RMS normalization using a named weight tensor.
    ///
    /// The output dtype matches the input dtype. When input is bf16, the bf16
    /// RMS norm kernel is dispatched and the weight is kept as bf16 (or cast
    /// to bf16 if stored as f32). When input is f32, the f32 kernel is used.
    fn apply_rms_norm(
        &mut self,
        input: &MlxBuffer,
        weight_name: &str,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let input_dtype = input.dtype();

        // Clone the weight buffer so we can release the immutable borrow on self
        let weight = require_weight(&self.weights, weight_name)?;
        let norm_weight_buf = clone_buffer(&self.device, &weight.buffer)?;

        // Output matches input dtype
        let elem_size = input_dtype.size_of();
        let output_bytes = seq_len * hidden_size * elem_size;
        let output = self.device.alloc_buffer(
            output_bytes,
            input_dtype,
            vec![seq_len, hidden_size],
        )?;

        // Create params buffer with [eps, dim]
        let eps = self.config.rms_norm_eps;
        let dim = hidden_size as f32;
        let mut params_buf = self.device.alloc_buffer(
            std::mem::size_of::<[f32; 2]>(),
            DType::F32,
            vec![2],
        )?;
        {
            let slice: &mut [f32] = params_buf.as_mut_slice().map_err(|e| {
                Gemma4Error::ForwardError {
                    reason: format!("RMS norm params write: {e}"),
                }
            })?;
            slice[0] = eps;
            slice[1] = dim;
        }

        // Prepare the norm weight to match input dtype.
        // For bf16 pipeline: if weight is already bf16, use as-is; if f32, cast to bf16.
        // For f32 pipeline: ensure weight is f32.
        let norm_weight = match input_dtype {
            DType::BF16 => {
                match norm_weight_buf.dtype() {
                    DType::BF16 => clone_buffer(&self.device, &norm_weight_buf)?,
                    DType::F32 => {
                        self.cast_f32_to_bf16(&norm_weight_buf, hidden_size)?
                    }
                    DType::F16 => {
                        // f16 -> f32 -> bf16
                        let f32_buf = self.cast_to_f32(&norm_weight_buf, hidden_size)?;
                        self.cast_f32_to_bf16(&f32_buf, hidden_size)?
                    }
                    _ => {
                        return Err(Gemma4Error::ForwardError {
                            reason: format!("Unsupported norm weight dtype for bf16 pipeline: {}", norm_weight_buf.dtype()),
                        });
                    }
                }
            }
            _ => {
                // f32 pipeline: ensure weight is f32
                self.ensure_f32_weight(&norm_weight_buf, hidden_size, weight_name)?
            }
        };

        let mut encoder = self.device.command_encoder()?;
        rms_norm::dispatch_rms_norm(
            &mut encoder,
            &mut self.registry,
            self.device.metal_device(),
            input,
            &norm_weight,
            &output,
            &params_buf,
            seq_len as u32,
            hidden_size as u32,
        )?;
        encoder.commit_and_wait()?;

        Ok(output)
    }

    /// Ensure an input buffer is f32. Returns a reference to the same buffer if
    /// already f32, or casts from f16/bf16 to a new f32 buffer.
    fn ensure_f32_input(
        &mut self,
        buffer: &MlxBuffer,
        n_elements: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        match buffer.dtype() {
            DType::F32 => {
                // Already f32 -- clone to satisfy ownership
                clone_buffer(&self.device, buffer)
                    .map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("clone f32 input: {e}"),
                    })
            }
            DType::F16 => {
                self.cast_to_f32(buffer, n_elements)
            }
            DType::BF16 => {
                // Cast bf16 -> f32
                let out_bytes = n_elements * std::mem::size_of::<f32>();
                let output = self.device.alloc_buffer(
                    out_bytes, DType::F32, vec![n_elements],
                )?;
                let mut encoder = self.device.command_encoder()?;
                elementwise::cast(
                    &mut encoder,
                    &mut self.registry,
                    self.device.metal_device(),
                    buffer,
                    &output,
                    n_elements,
                    elementwise::CastDirection::BF16ToF32,
                )?;
                encoder.commit_and_wait()?;
                Ok(output)
            }
            other => Err(Gemma4Error::ForwardError {
                reason: format!("Unsupported input dtype for f32 cast: {other}"),
            }),
        }
    }

    /// Ensure a weight buffer is in f32 format (convert from f16/bf16 if needed).
    fn ensure_f32_weight(
        &mut self,
        buffer: &MlxBuffer,
        n_elements: usize,
        _name: &str,
    ) -> Result<MlxBuffer, Gemma4Error> {
        match buffer.dtype() {
            DType::F32 => {
                // Already f32 -- we need to return a reference-compatible buffer.
                // For simplicity, copy it to a new buffer (the norm weights are small).
                let data = read_buffer_f32(buffer)?;
                let mut out = self.device.alloc_buffer(
                    data.len() * std::mem::size_of::<f32>(),
                    DType::F32,
                    vec![n_elements],
                )?;
                {
                    let slice: &mut [f32] =
                        out.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                            reason: format!("Weight copy: {e}"),
                        })?;
                    slice.copy_from_slice(&data);
                }
                Ok(out)
            }
            DType::F16 => {
                // Cast f16 -> f32 on GPU
                let out_bytes = n_elements * std::mem::size_of::<f32>();
                let output =
                    self.device
                        .alloc_buffer(out_bytes, DType::F32, vec![n_elements])?;

                let mut encoder = self.device.command_encoder()?;
                elementwise::cast(
                    &mut encoder,
                    &mut self.registry,
                    self.device.metal_device(),
                    buffer,
                    &output,
                    n_elements,
                    elementwise::CastDirection::F16ToF32,
                )?;
                encoder.commit_and_wait()?;
                Ok(output)
            }
            DType::BF16 => {
                let out_bytes = n_elements * std::mem::size_of::<f32>();
                let output =
                    self.device
                        .alloc_buffer(out_bytes, DType::F32, vec![n_elements])?;

                let mut encoder = self.device.command_encoder()?;
                elementwise::cast(
                    &mut encoder,
                    &mut self.registry,
                    self.device.metal_device(),
                    buffer,
                    &output,
                    n_elements,
                    elementwise::CastDirection::BF16ToF32,
                )?;
                encoder.commit_and_wait()?;
                Ok(output)
            }
            other => Err(Gemma4Error::ForwardError {
                reason: format!("Unsupported weight dtype: {other}"),
            }),
        }
    }

    /// Perform a quantized matmul projection (e.g., Q/K/V/O projections).
    ///
    /// Looks up weight, scales, biases tensors by the base name pattern.
    /// Input may be f32 or bf16. Output is bf16 to keep the pipeline in bf16.
    /// The GPU qmatmul kernel produces f32 output which is then cast to bf16.
    /// The CPU path accumulates in f32 and writes bf16.
    fn quantized_projection(
        &mut self,
        input: &MlxBuffer,
        weight_base: &str,
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let weight_name = format!("{weight_base}.weight");
        let scales_name = format!("{weight_base}.scales");
        let biases_name = format!("{weight_base}.biases");

        // Extract quant metadata before borrowing mutably
        let loaded = require_weight(&self.weights, &weight_name)?;
        let quant_info = loaded.quant_meta.as_ref().map(|qm| (qm.bits, qm.group_size));

        // Check if weight is quantized
        if let Some((bits, _group_size)) = quant_info {
            // GPU quantized matmul: uses native bf16 dequantization on Metal,
            // matching MLX's precision exactly. The kernel dequantizes weights
            // to bf16, truncates input to bf16, and accumulates in f32.
            let use_cpu_qmatmul = std::env::var("HF2Q_CPU_QMATMUL").is_ok();
            if !use_cpu_qmatmul && (bits == 4 || bits == 8) {
                // GPU path (default — matches MLX's bf16 precision)
                // qmatmul kernel expects f32 input; convert bf16 if needed
                let input_f32 = self.ensure_f32_input(input, seq_len * in_dim)?;

                let loaded = require_weight(&self.weights, &weight_name)?;
                let scales = require_weight(&self.weights, &scales_name)?;
                let biases = require_weight(&self.weights, &biases_name)?;

                let params = QuantizedMatmulParams {
                    m: seq_len as u32,
                    k: in_dim as u32,
                    n: out_dim as u32,
                    group_size: _group_size as u32,
                    bits: bits as u32,
                };

                let mut encoder = self.device.command_encoder()?;
                let output_f32 = qmatmul::quantized_matmul(
                    &mut encoder,
                    &mut self.registry,
                    &self.device,
                    &input_f32,
                    &loaded.buffer,
                    &scales.buffer,
                    &biases.buffer,
                    &params,
                )?;
                encoder.commit_and_wait()?;

                // Cast f32 output to bf16 for the bf16 pipeline
                let output_bf16 = self.cast_f32_to_bf16(&output_f32, seq_len * out_dim)?;
                return Ok(output_bf16);
            }

            // CPU dequantize + bf16-precision matmul.
            // MLX operates entirely in bf16: dequantized weights are bf16,
            // inputs are bf16, products are f32 (accumulated in f32).
            // We match this by rounding both operands to bf16 precision
            // before multiplying. This is critical for MoE models where
            // small precision differences cascade through expert routing.
            let weight_f32 = self.dequantize_2d_weight(weight_base, out_dim, in_dim)?;
            let input_data = read_buffer_f32(input)?;

            let weight_bf16 = round_to_bf16(&weight_f32);
            let input_bf16 = round_to_bf16(&input_data);

            let mut output_data = vec![0.0f32; seq_len * out_dim];
            for row in 0..seq_len {
                for col in 0..out_dim {
                    let mut sum = 0.0f32;
                    for k in 0..in_dim {
                        sum += input_bf16[row * in_dim + k] * weight_bf16[col * in_dim + k];
                    }
                    output_data[row * out_dim + col] = sum;
                }
            }

            // Write as bf16 for the bf16 pipeline
            return f32_vec_to_bf16_buffer(&self.device, &output_data, vec![seq_len, out_dim]);
        }

        // Non-quantized path: CPU matmul fallback, output as bf16
        let weight_buf = clone_buffer(
            &self.device,
            &require_weight(&self.weights, &weight_name)?.buffer,
        )?;
        let f32_result = self.cpu_matmul(input, &weight_buf, seq_len, in_dim, out_dim)?;
        // Cast to bf16
        self.cast_f32_to_bf16(&f32_result, seq_len * out_dim)
    }

    /// Cast an f32 buffer to f16 on the GPU.
    fn cast_to_f16(
        &mut self,
        input: &MlxBuffer,
        n_elements: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let out_bytes = n_elements * DType::F16.size_of();
        let output =
            self.device
                .alloc_buffer(out_bytes, DType::F16, vec![n_elements])?;

        let mut encoder = self.device.command_encoder()?;
        elementwise::cast(
            &mut encoder,
            &mut self.registry,
            self.device.metal_device(),
            input,
            &output,
            n_elements,
            elementwise::CastDirection::F32ToF16,
        )?;
        encoder.commit_and_wait()?;
        Ok(output)
    }

    /// Cast an f16 buffer to f32 on the GPU.
    fn cast_to_f32(
        &mut self,
        input: &MlxBuffer,
        n_elements: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let out_bytes = n_elements * DType::F32.size_of();
        let output =
            self.device
                .alloc_buffer(out_bytes, DType::F32, vec![n_elements])?;

        let mut encoder = self.device.command_encoder()?;
        elementwise::cast(
            &mut encoder,
            &mut self.registry,
            self.device.metal_device(),
            input,
            &output,
            n_elements,
            elementwise::CastDirection::F16ToF32,
        )?;
        encoder.commit_and_wait()?;
        Ok(output)
    }

    /// Cast an f32 buffer to bf16 on the GPU.
    fn cast_f32_to_bf16(
        &mut self,
        input: &MlxBuffer,
        n_elements: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let out_bytes = n_elements * 2; // bf16 = 2 bytes
        let output =
            self.device
                .alloc_buffer(out_bytes, DType::BF16, vec![n_elements])?;

        let mut encoder = self.device.command_encoder()?;
        elementwise::cast(
            &mut encoder,
            &mut self.registry,
            self.device.metal_device(),
            input,
            &output,
            n_elements,
            elementwise::CastDirection::F32ToBF16,
        )?;
        encoder.commit_and_wait()?;
        Ok(output)
    }

    /// Cast a bf16 buffer to f32 on the GPU.
    fn cast_bf16_to_f32(
        &mut self,
        input: &MlxBuffer,
        n_elements: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let out_bytes = n_elements * std::mem::size_of::<f32>();
        let output =
            self.device
                .alloc_buffer(out_bytes, DType::F32, vec![n_elements])?;

        let mut encoder = self.device.command_encoder()?;
        elementwise::cast(
            &mut encoder,
            &mut self.registry,
            self.device.metal_device(),
            input,
            &output,
            n_elements,
            elementwise::CastDirection::BF16ToF32,
        )?;
        encoder.commit_and_wait()?;
        Ok(output)
    }

    /// CPU fallback matmul for unquantized weights.
    fn cpu_matmul(
        &self,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let input_data = read_weight_as_f32_from_buffer(input)?;
        let weight_data = read_weight_as_f32_from_buffer(weight)?;

        // weight is [out_dim, in_dim] row-major; input is [seq_len, in_dim]
        // output = input @ weight^T => [seq_len, out_dim]
        let mut output_data = vec![0.0f32; seq_len * out_dim];
        for row in 0..seq_len {
            for col in 0..out_dim {
                let mut sum = 0.0f32;
                for k in 0..in_dim {
                    sum += input_data[row * in_dim + k] * weight_data[col * in_dim + k];
                }
                output_data[row * out_dim + col] = sum;
            }
        }

        let mut buf = self.device.alloc_buffer(
            output_data.len() * std::mem::size_of::<f32>(),
            DType::F32,
            vec![seq_len, out_dim],
        )?;
        {
            let slice: &mut [f32] = buf.as_mut_slice().map_err(|e| {
                Gemma4Error::ForwardError {
                    reason: format!("CPU matmul output write: {e}"),
                }
            })?;
            slice.copy_from_slice(&output_data);
        }
        Ok(buf)
    }

    /// Apply RoPE to Q or K tensor organized as [seq_len, n_heads * head_dim].
    ///
    /// For partial rotary (global attention), only the first `rope_dim` of each
    /// head's dimensions are rotated. The remaining dimensions pass through unchanged.
    fn apply_rope_to_heads(
        &mut self,
        input: &MlxBuffer,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        theta: f32,
    ) -> Result<MlxBuffer, Gemma4Error> {
        if rope_dim == 0 || seq_len == 0 {
            // No rotation needed
            return Ok(clone_buffer(&self.device, input)?);
        }

        let input_data = read_buffer_f32(input)?;

        // Compute starting position from KV cache
        // The first layer that matches this config gives us the position base
        // We need to figure out the position offset from the KV cache state.
        // For now, use 0 as the start position for the first call.
        // In decode mode (seq_len=1), position = kv_cache.seq_len().
        // In prefill mode (seq_len>1), positions = [0, 1, 2, ..., seq_len-1].
        // We compute positions based on how much is already in the cache.

        // Actually, we need the position to be the absolute position in the sequence.
        // For the first forward pass, this is [0..seq_len).
        // For subsequent decode steps, position starts at the current cache length
        // (before this layer's append, but we already appended above, so subtract seq_len).
        // However, position tracking is cleaner if we look at the first layer.
        // Since all layers share the same position tracking, we use a simple approach:
        // the position for the current tokens is based on what was in the cache before
        // this forward call minus the tokens we just appended.
        // Actually the append already happened for this layer's KV cache.
        // The position base is: total_written - seq_len (what was before this call's append).
        // But this gets called per-layer... Let's use the layer's write_position - seq_len.

        // The simplest correct approach: use the write_position of the current layer's
        // cache, which reflects the total positions written including the current tokens.
        // So the positions for the current tokens are:
        // [write_position - seq_len, ..., write_position - 1]

        // But wait -- we call apply_rope BEFORE appending to cache in the layer's flow.
        // Looking at forward_layer: we compute Q,K,V, apply RoPE, THEN append.
        // Actually no -- looking more carefully at the flow above in forward_layer:
        // Steps 2-3 happen before step 4 (KV cache append).
        // So at the time of RoPE, the cache has NOT yet been updated for this token.
        // The correct position base is: cache.write_position() (before append).
        // But we need to know which layer we're talking about...

        // For correctness, we'll do the RoPE on CPU where we have full control
        // over positions. The GPU RoPE dispatch expects [seq_len, head_dim] input
        // with a positions buffer.

        let mut output_data = input_data.clone();
        let total_dim = n_heads * head_dim;

        // We need the position base. We'll look at the layer cache for the first
        // layer (all layers advance positions in lockstep, but the cache types differ).
        // However, this function doesn't know which layer it's being called for.
        // We need to pass position info in. Let's use a simpler approach:
        // compute positions as [0..seq_len) for the initial prefill.
        // For decode mode, the caller would need to track positions.
        // For now, we compute based on whether seq_len > 1 (prefill vs decode).

        // A better approach: accept an explicit position_offset parameter.
        // But to avoid changing the signature right now, we look at the "total_written"
        // of any sliding cache layer (they all advance together).
        // IMPORTANT: This is called BEFORE the append for this layer, so the cache
        // reflects the state before this forward call's tokens.

        // Find the first layer's cache to get position offset
        // (all layers are at the same sequence position)
        let pos_offset = if self.kv_cache.num_layers() > 0 {
            // Use layer 0's write_position as the reference.
            // But wait: in forward_layer, we process layers sequentially.
            // Layer 0 would have already been appended by the time we get to layer 1.
            // So for layer N, layers 0..N have already appended, but layer N has not.
            // The correct position offset is layer N's write_position before append.

            // Actually, we need to rethink. Let me look at forward_layer flow again:
            // 1. norm
            // 2. Q/K/V projections
            // 3. RoPE (we are here)
            // 4. KV cache append
            // So for this specific layer, append hasn't happened yet.
            // But for earlier layers (0..layer_idx), append HAS happened.
            // However, the position offset should be the same for all layers in a
            // given forward() call: it's the number of tokens processed before this
            // forward() invocation.

            // The cleanest way: the first layer that hasn't been appended yet
            // has write_position equal to the number of tokens before this forward().
            // Since we process layers 0, 1, 2, ..., and we're currently in layer N,
            // layer N hasn't been appended yet. So its write_position gives the
            // correct base.
            //
            // But we don't know layer_idx here! Let's just pass it as a parameter.
            // For now, use the last layer's cache (it hasn't been appended yet if
            // we process sequentially).

            // Actually, the simplest correct approach: all layers' positions are
            // determined by the overall sequence position, not per-layer cache state.
            // In the first forward() call, positions are [0..seq_len).
            // In subsequent calls, positions are [prev_total..prev_total+seq_len).
            // The "prev_total" is the last layer's write_position (since it's the
            // one that hasn't been touched yet when we start a new forward() call).

            // For layer 0 during the current forward, its write_position after append
            // includes the current tokens. For the last layer, write_position is still
            // at the pre-forward state.

            // Use the last layer: it hasn't been appended yet during this forward().
            let last_layer = self.kv_cache.num_layers() - 1;
            match self.kv_cache.layer(last_layer) {
                Ok(cache) => cache.write_position(),
                Err(_) => 0,
            }
        } else {
            0
        };

        // Apply RoPE per head
        for pos in 0..seq_len {
            let abs_pos = pos_offset + pos;
            for head in 0..n_heads {
                let base = pos * total_dim + head * head_dim;
                // Apply rotation to the first rope_dim dimensions (in pairs)
                for i in (0..rope_dim).step_by(2) {
                    let freq = 1.0 / theta.powf(i as f32 / rope_dim as f32);
                    let angle = abs_pos as f32 * freq;
                    let cos_val = angle.cos();
                    let sin_val = angle.sin();

                    let idx0 = base + i;
                    let idx1 = base + i + 1;
                    if idx1 < output_data.len() {
                        let x0 = input_data[idx0];
                        let x1 = input_data[idx1];
                        output_data[idx0] = x0 * cos_val - x1 * sin_val;
                        output_data[idx1] = x0 * sin_val + x1 * cos_val;
                    }
                }
                // Dimensions [rope_dim..head_dim] pass through unchanged
                // (already copied from input_data.clone())
            }
        }

        // Write to bf16 Metal buffer for the bf16 pipeline
        f32_vec_to_bf16_buffer(&self.device, &output_data, input.shape().to_vec())
    }

    /// Compute attention for a single layer.
    ///
    /// Q is bf16 [seq_len, n_heads * head_dim]. K/V are read from the f32 KV cache
    /// and converted to bf16 at the SDPA boundary. The SDPA bf16 kernel is dispatched
    /// based on the Q buffer dtype. Output is bf16 [seq_len, n_heads * head_dim].
    fn compute_attention(
        &mut self,
        layer_idx: usize,
        q: &MlxBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        layer_type: LayerAttentionType,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let layer_cache = self.kv_cache.layer(layer_idx)?;
        let (k_cache_buf, v_cache_buf, kv_seq_len) = layer_cache.keys_values();

        if kv_seq_len == 0 {
            return Err(Gemma4Error::ForwardError {
                reason: format!("KV cache is empty for layer {layer_idx}"),
            });
        }

        // Q is bf16 [seq_len, n_heads * head_dim].
        // SDPA expects [batch, n_heads, seq_len, head_dim].
        // Read Q as f32 for reshape, then write as bf16.
        let q_data = read_buffer_f32(q)?;
        let q_reshaped = reshape_qkv_for_sdpa(&q_data, seq_len, n_heads, head_dim);

        // K/V cache is f32 [kv_seq_len, n_kv_heads * head_dim].
        // Read, reshape, then write as bf16 for the bf16 SDPA kernel.
        let k_data: &[f32] = k_cache_buf.as_slice().map_err(|e| {
            Gemma4Error::ForwardError {
                reason: format!("K cache read: {e}"),
            }
        })?;
        let k_valid = &k_data[..kv_seq_len * n_kv_heads * head_dim];
        let k_reshaped = reshape_qkv_for_sdpa(k_valid, kv_seq_len, n_kv_heads, head_dim);

        let v_data: &[f32] = v_cache_buf.as_slice().map_err(|e| {
            Gemma4Error::ForwardError {
                reason: format!("V cache read: {e}"),
            }
        })?;
        let v_valid = &v_data[..kv_seq_len * n_kv_heads * head_dim];
        let v_reshaped = reshape_qkv_for_sdpa(v_valid, kv_seq_len, n_kv_heads, head_dim);

        // Allocate bf16 GPU buffers for SDPA
        let q_buf = f32_vec_to_bf16_buffer(
            &self.device,
            &q_reshaped,
            vec![1, n_heads, seq_len, head_dim],
        )?;
        let k_buf = f32_vec_to_bf16_buffer(
            &self.device,
            &k_reshaped,
            vec![1, n_kv_heads, kv_seq_len, head_dim],
        )?;
        let v_buf = f32_vec_to_bf16_buffer(
            &self.device,
            &v_reshaped,
            vec![1, n_kv_heads, kv_seq_len, head_dim],
        )?;

        // Output is bf16 to match the pipeline
        let output_elements = 1 * n_heads * seq_len * head_dim;
        let output = self.device.alloc_buffer(
            output_elements * 2, // bf16 = 2 bytes
            DType::BF16,
            vec![1, n_heads, seq_len, head_dim],
        )?;

        let mut encoder = self.device.command_encoder()?;

        match layer_type {
            LayerAttentionType::Sliding => {
                let params = sdpa_sliding::SdpaSlidingParams {
                    n_heads: n_heads as u32,
                    n_kv_heads: n_kv_heads as u32,
                    head_dim: head_dim as u32,
                    seq_len: seq_len as u32,
                    kv_seq_len: kv_seq_len as u32,
                    window_size: self.config.sliding_window as u32,
                };
                sdpa_sliding::sdpa_sliding(
                    &mut encoder,
                    &mut self.registry,
                    &self.device,
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &output,
                    &params,
                    1, // batch_size
                )?;
            }
            LayerAttentionType::Global => {
                let params = sdpa::SdpaParams {
                    n_heads: n_heads as u32,
                    n_kv_heads: n_kv_heads as u32,
                    head_dim: head_dim as u32,
                    seq_len: seq_len as u32,
                    kv_seq_len: kv_seq_len as u32,
                };
                sdpa::sdpa(
                    &mut encoder,
                    &mut self.registry,
                    &self.device,
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &output,
                    &params,
                    1, // batch_size
                )?;
            }
        }

        encoder.commit_and_wait()?;

        // Reshape output from [1, n_heads, seq_len, head_dim] back to [seq_len, n_heads * head_dim]
        // Read bf16 output as f32, reshape, write back as bf16
        let out_data = read_buffer_f32(&output)?;
        let out_flat = reshape_sdpa_output(&out_data, seq_len, n_heads, head_dim);

        f32_vec_to_bf16_buffer(
            &self.device,
            &out_flat,
            vec![seq_len, n_heads * head_dim],
        )
    }

    /// Apply per-layer scalar: output = layer_scalar * input.
    /// Gemma 4 has a learned scalar per layer that scales residual contributions.
    /// Input/output are bf16 in the bf16 pipeline.
    fn apply_layer_scalar(
        &mut self,
        layer_idx: usize,
        input: &MlxBuffer,
        n_elements: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let scalar_name = format!("model.layers.{layer_idx}.layer_scalar");
        match require_weight(&self.weights, &scalar_name) {
            Ok(w) => {
                // Read the scalar value (bf16 -> f32)
                let scalar_val = if w.buffer.byte_len() >= 4 && w.buffer.dtype() == DType::F32 {
                    let s: &[f32] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("read layer_scalar: {e}"),
                    })?;
                    s[0]
                } else {
                    // bf16 (2 bytes) — read as u16 and convert
                    let raw: &[u16] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("read layer_scalar bf16: {e}"),
                    })?;
                    half::bf16::from_bits(raw[0]).to_f32()
                };

                // Read bf16 input as f32, multiply, write back as bf16
                let input_data = read_buffer_f32(input)?;
                let scaled: Vec<f32> = input_data.iter().map(|v| v * scalar_val).collect();
                f32_vec_to_bf16_buffer(&self.device, &scaled, vec![n_elements])
                    .map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("alloc scaled buffer: {e}"),
                    })
            }
            Err(_) => {
                // No layer_scalar for this layer — return input unchanged (clone it)
                clone_buffer(&self.device, input)
                    .map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("clone for missing layer_scalar: {e}"),
                    })
            }
        }
    }

    /// Elementwise add of two buffers (f32 or bf16).
    ///
    /// Both inputs must have the same dtype. Output matches the input dtype.
    fn elementwise_add(
        &mut self,
        a: &MlxBuffer,
        b: &MlxBuffer,
        n_elements: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let dtype = a.dtype();
        let elem_size = dtype.size_of();
        let output_bytes = n_elements * elem_size;
        let output = self.device.alloc_buffer(
            output_bytes,
            dtype,
            vec![n_elements],
        )?;

        let mut encoder = self.device.command_encoder()?;
        elementwise::elementwise_add(
            &mut encoder,
            &mut self.registry,
            self.device.metal_device(),
            a,
            b,
            &output,
            n_elements,
            dtype,
        )?;
        encoder.commit_and_wait()?;

        Ok(output)
    }

    /// MoE FFN for a single layer.
    ///
    /// `router_input` is the un-normed hidden state used for routing (the router
    /// applies its own internal RMS norm).
    /// `expert_input` is the pre-normed hidden state (after pre_feedforward_layernorm_2)
    /// that gets fed through the selected experts.
    ///
    /// For each token in the sequence:
    /// 1. Router: RMS-norm input, project to expert scores, top-K with softmax
    /// 2. Dispatch to selected experts using expert_input
    /// 3. Weighted sum of expert outputs
    fn moe_ffn(
        &mut self,
        layer_idx: usize,
        router_input: &MlxBuffer,
        expert_input: &MlxBuffer,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let n_experts = self.config.num_experts;
        let top_k = self.config.top_k_experts;
        let eps = self.config.rms_norm_eps;
        let diag = std::env::var("HF2Q_DIAG").is_ok() && layer_idx == 0;

        let router_data = read_buffer_f32(router_input)?;
        let expert_data = read_buffer_f32(expert_input)?;
        let mut output_data = vec![0.0f32; seq_len * hidden_size];

        if diag {
            let router_weight_name = format!("model.layers.{layer_idx}.router.proj.weight");
            let rw = require_weight(&self.weights, &router_weight_name)?;
            eprintln!("[DIAG] router weight: dtype={}, shape={:?}, byte_len={}",
                rw.buffer.dtype(), rw.buffer.shape(), rw.buffer.byte_len());
            if let Some(ref qm) = rw.quant_meta {
                eprintln!("[DIAG]   router quant: bits={}, group_size={}", qm.bits, qm.group_size);
            }
        }

        // Dequantize the router weight once for all tokens in this layer call.
        // Router weight is [n_experts, hidden_size], possibly quantized.
        // Round to bf16 to match MLX's precision.
        let router_f32_raw = self.dequantize_2d_weight(
            &format!("model.layers.{layer_idx}.router.proj"),
            n_experts,
            hidden_size,
        )?;
        let router_bf16 = round_to_bf16(&router_f32_raw);

        // Read router.scale [hidden_size] and per_expert_scale [n_experts]
        // The router's scale weight is used as the RMS norm weight, scaled by hidden_size^(-0.5).
        // Python: x = rms_norm(x, self.scale * self._root_size, eps)
        let router_scale_vec = self.read_1d_weight(
            &format!("model.layers.{layer_idx}.router.scale"),
            hidden_size,
        ).unwrap_or_else(|_| vec![1.0f32; hidden_size]);

        let per_expert_scale = self.read_1d_weight(
            &format!("model.layers.{layer_idx}.router.per_expert_scale"),
            n_experts,
        ).unwrap_or_else(|_| vec![1.0f32; n_experts]);

        // Pre-scale the router norm weight: scale * hidden_size^(-0.5)
        let root_size = (hidden_size as f32).powf(-0.5);
        let router_norm_weight: Vec<f32> = router_scale_vec.iter().map(|s| s * root_size).collect();

        if diag {
            eprintln!("[DIAG] router_norm_weight first 5: {:?}, per_expert_scale first 5: {:?}",
                &router_norm_weight[..5.min(router_norm_weight.len())],
                &per_expert_scale[..5.min(per_expert_scale.len())]);
            let n = router_bf16.len().min(10);
            eprintln!("[DIAG] router_f32 first {}: {:?}", n, &router_bf16[..n]);
        }

        let intermediate_size = self.config.moe_intermediate_size;

        // Process token by token for MoE (each token routes to different experts)
        for t in 0..seq_len {
            let token_router = &router_data[t * hidden_size..(t + 1) * hidden_size];
            let token_expert = &expert_data[t * hidden_size..(t + 1) * hidden_size];

            // Step 1: RMS-normalize the router input, then project to expert scores
            // Python: x = rms_norm(x, self.scale * root_size, eps); scores = self.proj(x)
            let normed_router = cpu_rms_norm(token_router, &router_norm_weight, eps);

            // Round to bf16 to match MLX's precision for router scoring
            let normed_bf16 = round_to_bf16(&normed_router);

            let mut logits = vec![0.0f32; n_experts];
            for e in 0..n_experts {
                let mut sum = 0.0f32;
                for k in 0..hidden_size {
                    sum += router_bf16[e * hidden_size + k] * normed_bf16[k];
                }
                logits[e] = sum;
            }

            if diag && t == 0 {
                let mut sorted: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                eprintln!("[DIAG] MoE L0 T0 router logits (pre-softmax) top-5: {:?}",
                    &sorted[..5.min(sorted.len())]);
                let nan_count = logits.iter().filter(|v| v.is_nan()).count();
                if nan_count > 0 {
                    eprintln!("[DIAG]   WARNING: router logits have {} NaN values", nan_count);
                }
            }

            // Step 2: Top-K with softmax, then apply per_expert_scale
            // Python: argpartition -> take top-K scores -> softmax -> * per_expert_scale
            let (expert_ids, routing_weights) =
                top_k_softmax_with_scale(&logits, &per_expert_scale, top_k);

            if diag && t == 0 {
                eprintln!("[DIAG] MoE L0 T0 selected experts: {:?}, weights: {:?}",
                    expert_ids, routing_weights);
            }

            // Step 3: Dispatch to selected experts and accumulate
            // Expert input is the pre-normed tensor (after pre_feedforward_layernorm_2)
            let mut token_output = vec![0.0f32; hidden_size];

            for (rank, &expert_id) in expert_ids.iter().enumerate() {
                let weight = routing_weights[rank];
                if weight.abs() < 1e-10 {
                    continue;
                }

                let expert_output = self.expert_ffn_3d(
                    layer_idx, expert_id, token_expert,
                    hidden_size, intermediate_size,
                )?;

                if diag && t == 0 && rank == 0 {
                    let n = expert_output.len().min(5);
                    eprintln!("[DIAG] MoE L0 T0 expert {} output first {}: {:?}",
                        expert_id, n, &expert_output[..n]);
                    let nan_count = expert_output.iter().filter(|v| v.is_nan()).count();
                    if nan_count > 0 {
                        eprintln!("[DIAG]   WARNING: expert output has {} NaN values out of {}",
                            nan_count, expert_output.len());
                    }
                }

                for (j, &val) in expert_output.iter().enumerate() {
                    token_output[j] += weight * val;
                }
            }

            if diag && t == 0 {
                let n = token_output.len().min(5);
                eprintln!("[DIAG] MoE L0 T0 final accumulated output first {}: {:?}", n, &token_output[..n]);
            }

            output_data[t * hidden_size..(t + 1) * hidden_size]
                .copy_from_slice(&token_output);
        }

        f32_vec_to_bf16_buffer(
            &self.device,
            &output_data,
            vec![seq_len, hidden_size],
        )
    }

    /// Dequantize a 2D weight tensor (possibly quantized) and return as flat f32 [rows * cols].
    ///
    /// `weight_base` is the base name without `.weight` suffix; we look up
    /// `.weight`, `.scales`, `.biases` as needed.
    fn dequantize_2d_weight(
        &self,
        weight_base: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, Gemma4Error> {
        let weight_name = format!("{weight_base}.weight");
        let loaded = require_weight(&self.weights, &weight_name)?;

        // If not quantized, just read as float
        if loaded.quant_meta.is_none() || loaded.buffer.dtype() == DType::F32
            || loaded.buffer.dtype() == DType::F16 || loaded.buffer.dtype() == DType::BF16
        {
            return read_weight_as_f32(loaded);
        }

        // Quantized path: dequantize on CPU
        let qmeta = loaded.quant_meta.as_ref().unwrap();
        let bits = qmeta.bits as usize;
        let group_size = qmeta.group_size;

        let scales_name = format!("{weight_base}.scales");
        let biases_name = format!("{weight_base}.biases");
        let scales_w = require_weight(&self.weights, &scales_name)?;
        let biases_w = require_weight(&self.weights, &biases_name)?;

        let packed_u32: &[u32] = loaded.buffer.as_slice().map_err(|e| {
            Gemma4Error::ForwardError { reason: format!("read packed weight u32: {e}") }
        })?;
        let scales_f32 = read_weight_as_f32(scales_w)?;
        let biases_f32 = read_weight_as_f32(biases_w)?;

        cpu_dequantize_flat(packed_u32, &scales_f32, &biases_f32, rows, cols, bits, group_size)
    }

    /// Read a scalar weight (bf16 or f32) as f32.
    fn read_scalar_weight(&self, name: &str) -> Result<f32, Gemma4Error> {
        let loaded = require_weight(&self.weights, name)?;
        if loaded.buffer.byte_len() >= 4 && loaded.buffer.dtype() == DType::F32 {
            let s: &[f32] = loaded.buffer.as_slice().map_err(|e| {
                Gemma4Error::ForwardError { reason: format!("read scalar f32: {e}") }
            })?;
            Ok(s[0])
        } else {
            // bf16 (2 bytes)
            let raw: &[u16] = loaded.buffer.as_slice().map_err(|e| {
                Gemma4Error::ForwardError { reason: format!("read scalar bf16: {e}") }
            })?;
            Ok(half::bf16::from_bits(raw[0]).to_f32())
        }
    }

    /// Read a 1D weight vector as f32, handling bf16/f16/f32.
    fn read_1d_weight(
        &self,
        name: &str,
        expected_len: usize,
    ) -> Result<Vec<f32>, Gemma4Error> {
        let loaded = require_weight(&self.weights, name)?;
        match loaded.buffer.dtype() {
            DType::F32 => {
                let s: &[f32] = loaded.buffer.as_slice().map_err(|e| {
                    Gemma4Error::ForwardError { reason: format!("read 1d f32: {e}") }
                })?;
                Ok(s[..expected_len].to_vec())
            }
            DType::F16 => {
                let s: &[u16] = loaded.buffer.as_slice().map_err(|e| {
                    Gemma4Error::ForwardError { reason: format!("read 1d f16: {e}") }
                })?;
                Ok(s[..expected_len].iter().map(|&b| f16_to_f32(b)).collect())
            }
            DType::BF16 => {
                let s: &[u16] = loaded.buffer.as_slice().map_err(|e| {
                    Gemma4Error::ForwardError { reason: format!("read 1d bf16: {e}") }
                })?;
                Ok(s[..expected_len].iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect())
            }
            other => Err(Gemma4Error::ForwardError {
                reason: format!("Unsupported dtype for 1d weight '{}': {}", name, other),
            }),
        }
    }

    /// Run a single expert's FFN by slicing 3D packed weight tensors.
    ///
    /// Expert weights are stored as 3D tensors `[n_experts, rows, packed_cols]`
    /// where `packed_cols = cols * bits / 32`. This method slices the expert's
    /// 2D plane `[rows, packed_cols]` along dim 0, dequantizes, and computes
    /// the gated FFN: `down_proj(GELU(gate_proj(x)) * up_proj(x))`.
    fn expert_ffn_3d(
        &self,
        layer_idx: usize,
        expert_id: usize,
        token_hidden: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Vec<f32>, Gemma4Error> {
        // Only diagnose for first call on layer 0
        let diag = std::env::var("HF2Q_DIAG").is_ok() && layer_idx == 0;
        static DIAG_PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let diag = diag && !DIAG_PRINTED.swap(true, std::sync::atomic::Ordering::Relaxed);

        let base = format!("model.layers.{layer_idx}.experts.switch_glu");

        let gate_name = format!("{base}.gate_proj");
        let up_name = format!("{base}.up_proj");
        let down_name = format!("{base}.down_proj");

        if diag {
            let gw = require_weight(&self.weights, &format!("{gate_name}.weight"))?;
            eprintln!("[DIAG] expert_ffn_3d L0 E{}: gate_proj dtype={}, shape={:?}, byte_len={}",
                expert_id, gw.buffer.dtype(), gw.buffer.shape(), gw.buffer.byte_len());
            if let Some(ref qm) = gw.quant_meta {
                eprintln!("[DIAG]   gate_proj quant: bits={}, group_size={}", qm.bits, qm.group_size);
            }
            let dw = require_weight(&self.weights, &format!("{down_name}.weight"))?;
            eprintln!("[DIAG]   down_proj dtype={}, shape={:?}", dw.buffer.dtype(), dw.buffer.shape());
        }

        // gate_proj: [n_experts, intermediate_size, packed_k] where packed_k = hidden_size * bits / 32
        let gate_data = self.dequantize_expert_slice(
            &gate_name, expert_id, intermediate_size, hidden_size,
        )?;
        // up_proj: same shape as gate_proj
        let up_data = self.dequantize_expert_slice(
            &up_name, expert_id, intermediate_size, hidden_size,
        )?;
        // down_proj: [n_experts, hidden_size, packed_k] where packed_k = intermediate_size * bits / 32
        let down_data = self.dequantize_expert_slice(
            &down_name, expert_id, hidden_size, intermediate_size,
        )?;

        if diag {
            eprintln!("[DIAG] expert_ffn_3d L0 E{}: gate_data len={} (expect {}x{}={}), first 5: {:?}",
                expert_id, gate_data.len(), intermediate_size, hidden_size,
                intermediate_size * hidden_size, &gate_data[..5.min(gate_data.len())]);
            eprintln!("[DIAG]   down_data len={} (expect {}x{}={}), first 5: {:?}",
                down_data.len(), hidden_size, intermediate_size,
                hidden_size * intermediate_size, &down_data[..5.min(down_data.len())]);
            let gate_nan = gate_data.iter().filter(|v| v.is_nan()).count();
            let up_nan = up_data.iter().filter(|v| v.is_nan()).count();
            let down_nan = down_data.iter().filter(|v| v.is_nan()).count();
            if gate_nan > 0 || up_nan > 0 || down_nan > 0 {
                eprintln!("[DIAG]   NaN in dequantized weights! gate={}, up={}, down={}",
                    gate_nan, up_nan, down_nan);
            }
            eprintln!("[DIAG]   input token_hidden first 5: {:?}, len={}",
                &token_hidden[..5.min(token_hidden.len())], token_hidden.len());
        }

        // Round weights and input to bf16 to match MLX's precision
        let gate_bf16 = round_to_bf16(&gate_data);
        let up_bf16 = round_to_bf16(&up_data);
        let down_bf16 = round_to_bf16(&down_data);
        let input_bf16 = round_to_bf16(token_hidden);

        // gate_out = gate_proj @ x   [intermediate_size]
        let mut gate_out = vec![0.0f32; intermediate_size];
        for i in 0..intermediate_size {
            let mut sum = 0.0f32;
            for k in 0..hidden_size {
                sum += gate_bf16[i * hidden_size + k] * input_bf16[k];
            }
            gate_out[i] = sum;
        }

        if diag {
            let n = gate_out.len().min(5);
            eprintln!("[DIAG] expert_ffn_3d L0 E{}: gate_out first {}: {:?}",
                expert_id, n, &gate_out[..n]);
        }

        // up_out = up_proj @ x   [intermediate_size]
        let mut up_out = vec![0.0f32; intermediate_size];
        for i in 0..intermediate_size {
            let mut sum = 0.0f32;
            for k in 0..hidden_size {
                sum += up_bf16[i * hidden_size + k] * input_bf16[k];
            }
            up_out[i] = sum;
        }

        // hidden = GELU(gate_out) * up_out
        let mut hidden = vec![0.0f32; intermediate_size];
        for i in 0..intermediate_size {
            hidden[i] = gelu_pytorch_tanh(gate_out[i]) * up_out[i];
        }

        if diag {
            let n = hidden.len().min(5);
            eprintln!("[DIAG] expert_ffn_3d L0 E{}: after GELU*up first {}: {:?}",
                expert_id, n, &hidden[..n]);
        }

        // Round hidden to bf16 for down_proj matmul
        let hidden_bf16 = round_to_bf16(&hidden);

        // expert_out = down_proj @ hidden   [hidden_size]
        let mut expert_out = vec![0.0f32; hidden_size];
        for i in 0..hidden_size {
            let mut sum = 0.0f32;
            for k in 0..intermediate_size {
                sum += down_bf16[i * intermediate_size + k] * hidden_bf16[k];
            }
            expert_out[i] = sum;
        }

        if diag {
            let n = expert_out.len().min(5);
            eprintln!("[DIAG] expert_ffn_3d L0 E{}: final expert_out first {}: {:?}",
                expert_id, n, &expert_out[..n]);
            let nan_count = expert_out.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                eprintln!("[DIAG]   WARNING: expert_out has {} NaN out of {}", nan_count, expert_out.len());
            }
        }

        Ok(expert_out)
    }

    /// Dequantize a single expert's 2D weight slice from a 3D packed tensor.
    ///
    /// The weight tensor has shape `[n_experts, rows, packed_cols]` where
    /// `packed_cols = cols * bits / 32`. Scales and biases have shape
    /// `[n_experts, rows, n_groups]` where `n_groups = cols / group_size`.
    ///
    /// This method extracts `expert_id`'s plane and dequantizes to `[rows, cols]` f32.
    fn dequantize_expert_slice(
        &self,
        weight_base: &str,
        expert_id: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, Gemma4Error> {
        let weight_name = format!("{weight_base}.weight");
        let loaded = require_weight(&self.weights, &weight_name)?;

        // If not quantized (float dtype), handle the 3D float case
        if loaded.quant_meta.is_none() || loaded.buffer.dtype() == DType::F32
            || loaded.buffer.dtype() == DType::F16 || loaded.buffer.dtype() == DType::BF16
        {
            let all_data = read_weight_as_f32(loaded)?;
            let shape = loaded.buffer.shape();
            if shape.len() == 3 {
                // 3D float tensor: [n_experts, rows, cols]
                let expert_stride = shape[1] * shape[2];
                let start = expert_id * expert_stride;
                let end = start + rows * cols;
                return Ok(all_data[start..end].to_vec());
            }
            // 2D: already per-expert or shared (return as-is)
            return Ok(all_data);
        }

        // Quantized 3D tensor path
        let qmeta = loaded.quant_meta.as_ref().unwrap();
        let bits = qmeta.bits as usize;
        let group_size = qmeta.group_size;

        let scales_name = format!("{weight_base}.scales");
        let biases_name = format!("{weight_base}.biases");
        let scales_w = require_weight(&self.weights, &scales_name)?;
        let biases_w = require_weight(&self.weights, &biases_name)?;

        let packed_u32: &[u32] = loaded.buffer.as_slice().map_err(|e| {
            Gemma4Error::ForwardError { reason: format!("read 3D packed weight u32: {e}") }
        })?;
        let scales_f32 = read_weight_as_f32(scales_w)?;
        let biases_f32 = read_weight_as_f32(biases_w)?;

        let shape = loaded.buffer.shape();
        let scales_shape = scales_w.buffer.shape();

        // Determine packed_cols and n_groups from the actual tensor shape
        let packed_cols = if shape.len() == 3 { shape[2] } else { shape[1] };
        let n_groups = if scales_shape.len() == 3 { scales_shape[2] } else { scales_shape[1] };

        // Stride for one expert in the packed weight tensor
        let weight_expert_stride = rows * packed_cols;
        let scales_expert_stride = rows * n_groups;

        let w_start = expert_id * weight_expert_stride;
        let w_end = w_start + weight_expert_stride;
        let s_start = expert_id * scales_expert_stride;
        let s_end = s_start + scales_expert_stride;

        let expert_packed = &packed_u32[w_start..w_end];
        let expert_scales = &scales_f32[s_start..s_end];
        let expert_biases = &biases_f32[s_start..s_end];

        cpu_dequantize_flat(expert_packed, expert_scales, expert_biases, rows, cols, bits, group_size)
    }

    /// Dense (non-MoE) FFN fallback.
    fn dense_ffn(
        &mut self,
        layer_idx: usize,
        input: &MlxBuffer,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let diag = std::env::var("HF2Q_DIAG").is_ok() && layer_idx == 0;

        if diag {
            eprintln!("[DIAG] dense_ffn L0: hidden_size={}, intermediate_size={}", hidden_size, intermediate_size);
            let gw_name = format!("model.layers.{layer_idx}.mlp.gate_proj.weight");
            let gw = require_weight(&self.weights, &gw_name)?;
            eprintln!("[DIAG]   gate_proj.weight: dtype={}, shape={:?}, byte_len={}",
                gw.buffer.dtype(), gw.buffer.shape(), gw.buffer.byte_len());
            if let Some(ref qm) = gw.quant_meta {
                eprintln!("[DIAG]   gate_proj quant: bits={}, group_size={}", qm.bits, qm.group_size);
                let packed_cols = hidden_size * qm.bits as usize / 32;
                eprintln!("[DIAG]   expected packed shape for [{},{}] at {} bits: [{},{}]",
                    intermediate_size, hidden_size, qm.bits, intermediate_size, packed_cols);
            }
        }

        // gate_proj
        let gate = self.quantized_projection(
            input,
            &format!("model.layers.{layer_idx}.mlp.gate_proj"),
            seq_len,
            hidden_size,
            intermediate_size,
        )?;

        if diag {
            let s = read_buffer_f32(&gate)?;
            let n = s.len().min(5);
            eprintln!("[DIAG] dense_ffn L0: gate output first {}: {:?}", n, &s[..n]);
            let nan_count = s.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                eprintln!("[DIAG]   WARNING dense gate: NaN={}/{}", nan_count, s.len());
            }
        }

        // up_proj
        let up = self.quantized_projection(
            input,
            &format!("model.layers.{layer_idx}.mlp.up_proj"),
            seq_len,
            hidden_size,
            intermediate_size,
        )?;

        if diag {
            let s = read_buffer_f32(&up)?;
            let n = s.len().min(5);
            eprintln!("[DIAG] dense_ffn L0: up output first {}: {:?}", n, &s[..n]);
            let nan_count = s.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                eprintln!("[DIAG]   WARNING dense up: NaN={}/{}", nan_count, s.len());
            }
        }

        // GELU(gate) * up
        let gate_data = read_buffer_f32(&gate)?;
        let up_data = read_buffer_f32(&up)?;
        let mut hidden_data = vec![0.0f32; seq_len * intermediate_size];
        for i in 0..hidden_data.len() {
            hidden_data[i] = gelu_pytorch_tanh(gate_data[i]) * up_data[i];
        }

        if diag {
            let n = hidden_data.len().min(5);
            eprintln!("[DIAG] dense_ffn L0: GELU*up first {}: {:?}", n, &hidden_data[..n]);
        }

        let hidden_buf = f32_vec_to_bf16_buffer(
            &self.device,
            &hidden_data,
            vec![seq_len, intermediate_size],
        )?;

        // down_proj (bf16 in, bf16 out)
        let result = self.quantized_projection(
            &hidden_buf,
            &format!("model.layers.{layer_idx}.mlp.down_proj"),
            seq_len,
            intermediate_size,
            hidden_size,
        )?;

        if diag {
            let s = read_buffer_f32(&result)?;
            let n = s.len().min(5);
            eprintln!("[DIAG] dense_ffn L0: down output first {}: {:?}", n, &s[..n]);
            let nan_count = s.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                eprintln!("[DIAG]   WARNING dense down: NaN={}/{}", nan_count, s.len());
            }
        }

        Ok(result)
    }

    /// Compute the lm_head projection using tied embeddings.
    ///
    /// The embedding weight is [vocab_size, hidden_size].
    /// lm_head = hidden_state @ embed_weight^T => [seq_len, vocab_size].
    ///
    /// Input is bf16 (final normed hidden state). Output is f32 logits
    /// (for softcap and sampling). This is where we transition back to f32.
    fn lm_head_projection(
        &mut self,
        input: &MlxBuffer,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        let embed_weight = require_weight(&self.weights, "model.embed_tokens.weight")?;

        // Check if embedding is quantized
        if let Some(ref qmeta) = embed_weight.quant_meta {
            if qmeta.bits == 4 || qmeta.bits == 6 {
                // quantized_projection now returns bf16. For lm_head we need f32 logits.
                // Use the GPU qmatmul directly which returns f32, or convert after.
                let bf16_result = self.quantized_projection(
                    input,
                    "model.embed_tokens",
                    seq_len,
                    hidden_size,
                    vocab_size,
                )?;
                // Cast bf16 logits back to f32 for softcap
                return self.cast_bf16_to_f32(&bf16_result, seq_len * vocab_size);
            }
        }

        // Unquantized: CPU matmul (input is bf16, read as f32 by cpu_matmul)
        // embed_weight is [vocab_size, hidden_size]
        // We want: input @ embed_weight^T => [seq_len, vocab_size]
        // cpu_matmul reads buffers as f32 (handles bf16), produces f32 output
        self.cpu_matmul(input, &embed_weight.buffer, seq_len, hidden_size, vocab_size)
    }

    /// Apply softcap: cap * tanh(logits / cap).
    fn apply_softcap(
        &mut self,
        logits: &MlxBuffer,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let vocab_size = self.config.vocab_size;
        let cap = self.config.final_logit_softcapping;
        let n_elements = seq_len * vocab_size;

        let output_bytes = n_elements * std::mem::size_of::<f32>();
        let output = self.device.alloc_buffer(
            output_bytes,
            DType::F32,
            vec![seq_len, vocab_size],
        )?;

        // Create params buffer with cap value
        let mut params_buf = self.device.alloc_buffer(
            std::mem::size_of::<f32>(),
            DType::F32,
            vec![1],
        )?;
        {
            let slice: &mut [f32] = params_buf.as_mut_slice().map_err(|e| {
                Gemma4Error::ForwardError {
                    reason: format!("Softcap params write: {e}"),
                }
            })?;
            slice[0] = cap;
        }

        let mut encoder = self.device.command_encoder()?;
        softcap::dispatch_softcap(
            &mut encoder,
            &mut self.registry,
            self.device.metal_device(),
            logits,
            &output,
            &params_buf,
            cap,
        )?;
        encoder.commit_and_wait()?;

        Ok(output)
    }

    /// Apply softcap on CPU: cap * tanh(logits / cap).
    ///
    /// This is a fallback for the GPU kernel which can produce incorrect results
    /// on some Metal implementations with very large dispatch grids.
    fn apply_softcap_cpu(
        &self,
        logits: &MlxBuffer,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let vocab_size = self.config.vocab_size;
        let cap = self.config.final_logit_softcapping;
        let n_elements = seq_len * vocab_size;

        let input_data: &[f32] = logits.as_slice().map_err(|e| Gemma4Error::ForwardError {
            reason: format!("softcap_cpu read input: {e}"),
        })?;

        let capped: Vec<f32> = input_data[..n_elements]
            .iter()
            .map(|&x| (x / cap).tanh() * cap)
            .collect();

        f32_vec_to_buffer(&self.device, &capped, vec![seq_len, vocab_size])
            .map_err(|e| Gemma4Error::ForwardError {
                reason: format!("softcap_cpu alloc output: {e}"),
            })
    }
}

// ---- Helper functions ----

/// Require a weight to exist in the weight map.
/// Tries the given name first, then with `language_model.` prefix (MLX-format models).
fn require_weight<'a>(
    weights: &'a WeightMap,
    name: &str,
) -> Result<&'a LoadedWeight, Gemma4Error> {
    if let Some(w) = weights.get(name) {
        return Ok(w);
    }
    // Try with language_model. prefix (MLX-format Gemma 4 models use this)
    let prefixed = format!("language_model.{}", name);
    if let Some(w) = weights.get(&prefixed) {
        return Ok(w);
    }
    Err(Gemma4Error::MissingWeight {
        name: name.to_string(),
    })
}

/// Read a Metal buffer's contents as f32.
fn read_buffer_f32(buffer: &MlxBuffer) -> Result<Vec<f32>, Gemma4Error> {
    match buffer.dtype() {
        DType::F32 => {
            let slice: &[f32] = buffer.as_slice().map_err(|e| {
                Gemma4Error::ForwardError {
                    reason: format!("Buffer read f32: {e}"),
                }
            })?;
            Ok(slice.to_vec())
        }
        DType::F16 => {
            let slice: &[u16] = buffer.as_slice().map_err(|e| {
                Gemma4Error::ForwardError {
                    reason: format!("Buffer read f16: {e}"),
                }
            })?;
            Ok(slice.iter().map(|&bits| f16_to_f32(bits)).collect())
        }
        DType::BF16 => {
            let slice: &[u16] = buffer.as_slice().map_err(|e| {
                Gemma4Error::ForwardError {
                    reason: format!("Buffer read bf16: {e}"),
                }
            })?;
            Ok(slice.iter().map(|&bits| half::bf16::from_bits(bits).to_f32()).collect())
        }
        other => Err(Gemma4Error::ForwardError {
            reason: format!("Unsupported buffer dtype for read: {other}"),
        }),
    }
}

/// Read a loaded weight tensor as f32, handling f16/bf16 conversion.
fn read_weight_as_f32(weight: &LoadedWeight) -> Result<Vec<f32>, Gemma4Error> {
    read_weight_as_f32_from_buffer(&weight.buffer)
}

/// Read a buffer as f32, handling dtype conversion.
fn read_weight_as_f32_from_buffer(buffer: &MlxBuffer) -> Result<Vec<f32>, Gemma4Error> {
    read_buffer_f32(buffer)
}

/// Clone a Metal buffer (allocate new buffer, copy contents).
fn clone_buffer(device: &MlxDevice, src: &MlxBuffer) -> Result<MlxBuffer, Gemma4Error> {
    let byte_len = src.byte_len();
    let dst = device.alloc_buffer(byte_len, src.dtype(), src.shape().to_vec())?;

    // Copy via CPU (both buffers are StorageModeShared)
    let src_ptr = src.contents_ptr();
    let dst_ptr = dst.contents_ptr();
    if !src_ptr.is_null() && !dst_ptr.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_ptr as *const u8,
                dst_ptr as *mut u8,
                byte_len,
            );
        }
    }
    Ok(dst)
}

/// Create an f32 Metal buffer from a Vec.
fn f32_vec_to_buffer(
    device: &MlxDevice,
    data: &[f32],
    shape: Vec<usize>,
) -> Result<MlxBuffer, Gemma4Error> {
    let byte_len = data.len() * std::mem::size_of::<f32>();
    let mut buf = device.alloc_buffer(byte_len, DType::F32, shape)?;
    {
        let slice: &mut [f32] = buf.as_mut_slice().map_err(|e| {
            Gemma4Error::ForwardError {
                reason: format!("f32 buffer write: {e}"),
            }
        })?;
        slice.copy_from_slice(data);
    }
    Ok(buf)
}

/// Create a bf16 Metal buffer from f32 data.
///
/// Each f32 value is rounded to bf16 precision and stored as u16.
fn f32_vec_to_bf16_buffer(
    device: &MlxDevice,
    data: &[f32],
    shape: Vec<usize>,
) -> Result<MlxBuffer, Gemma4Error> {
    let byte_len = data.len() * 2; // bf16 = 2 bytes
    let mut buf = device.alloc_buffer(byte_len, DType::BF16, shape)?;
    {
        let slice: &mut [u16] = buf.as_mut_slice().map_err(|e| {
            Gemma4Error::ForwardError {
                reason: format!("bf16 buffer write: {e}"),
            }
        })?;
        for (i, &v) in data.iter().enumerate() {
            slice[i] = half::bf16::from_f32(v).to_bits();
        }
    }
    Ok(buf)
}

/// Reshape Q/K/V from [seq_len, n_heads * head_dim] to [n_heads, seq_len, head_dim].
///
/// Input layout (row-major):
///   for each seq position s:
///     for each head h:
///       head_dim values at input[s * (n_heads * head_dim) + h * head_dim .. + head_dim]
///
/// Output layout: [n_heads, seq_len, head_dim]
///   for each head h:
///     for each seq position s:
///       head_dim values
fn reshape_qkv_for_sdpa(
    data: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let total_dim = n_heads * head_dim;
    let mut out = vec![0.0f32; n_heads * seq_len * head_dim];

    for s in 0..seq_len {
        for h in 0..n_heads {
            let src_base = s * total_dim + h * head_dim;
            let dst_base = h * seq_len * head_dim + s * head_dim;
            out[dst_base..dst_base + head_dim]
                .copy_from_slice(&data[src_base..src_base + head_dim]);
        }
    }

    out
}

/// Reshape SDPA output from [n_heads, seq_len, head_dim] to [seq_len, n_heads * head_dim].
fn reshape_sdpa_output(
    data: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let total_dim = n_heads * head_dim;
    let mut out = vec![0.0f32; seq_len * total_dim];

    for h in 0..n_heads {
        for s in 0..seq_len {
            let src_base = h * seq_len * head_dim + s * head_dim;
            let dst_base = s * total_dim + h * head_dim;
            out[dst_base..dst_base + head_dim]
                .copy_from_slice(&data[src_base..src_base + head_dim]);
        }
    }

    out
}

/// CPU RMS normalization of a single vector with a learned weight.
///
/// `output[i] = weight[i] * x[i] / sqrt(mean(x^2) + eps)`
fn cpu_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = x.len();
    debug_assert_eq!(dim, weight.len());
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| wi * xi * scale)
        .collect()
}

/// CPU RMS normalization without a learnable scale (unit weight).
///
/// `output[i] = x[i] / sqrt(mean(x^2) + eps)`
fn cpu_rms_norm_no_scale(x: &[f32], eps: f32) -> Vec<f32> {
    let dim = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|&xi| xi * scale).collect()
}

/// Apply per-head RMS normalization to Q or K.
///
/// Input layout: `[seq_len, n_heads * head_dim]` (flat f32).
/// Weight: `[head_dim]` (one weight vector shared across all heads).
///
/// For each sequence position and each head, independently normalizes the
/// `head_dim`-sized vector using the given weight.
fn cpu_per_head_rms_norm(
    data: &[f32],
    weight: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let total_dim = n_heads * head_dim;
    let mut output = vec![0.0f32; seq_len * total_dim];

    for s in 0..seq_len {
        for h in 0..n_heads {
            let offset = s * total_dim + h * head_dim;
            let head_slice = &data[offset..offset + head_dim];
            let normed = cpu_rms_norm(head_slice, weight, eps);
            output[offset..offset + head_dim].copy_from_slice(&normed);
        }
    }

    output
}

/// Apply scaleless per-head RMS normalization to V (no learnable weight).
///
/// Input layout: `[seq_len, n_heads * head_dim]` (flat f32).
///
/// For each sequence position and each head, independently normalizes the
/// `head_dim`-sized vector without a learned scale.
fn cpu_per_head_rms_norm_no_scale(
    data: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let total_dim = n_heads * head_dim;
    let mut output = vec![0.0f32; seq_len * total_dim];

    for s in 0..seq_len {
        for h in 0..n_heads {
            let offset = s * total_dim + h * head_dim;
            let head_slice = &data[offset..offset + head_dim];
            let normed = cpu_rms_norm_no_scale(head_slice, eps);
            output[offset..offset + head_dim].copy_from_slice(&normed);
        }
    }

    output
}

/// CPU RMS normalization of a full `[seq_len, dim]` tensor with a learned weight.
///
/// This is like `apply_rms_norm` but works on arbitrary `dim` (not just `hidden_size`),
/// using CPU for simplicity and correctness.
#[allow(dead_code)]
fn cpu_rms_norm_2d(
    data: &[f32],
    weight: &[f32],
    seq_len: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * dim];
    for s in 0..seq_len {
        let offset = s * dim;
        let row = &data[offset..offset + dim];
        let normed = cpu_rms_norm(row, weight, eps);
        output[offset..offset + dim].copy_from_slice(&normed);
    }
    output
}

/// GELU activation with PyTorch tanh approximation.
///
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
fn gelu_pytorch_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = 0.7978845608028654;
    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Top-K selection with softmax normalization over the selected experts.
///
/// Used by models that employ softmax-based routing.
#[allow(dead_code)]
fn top_k_softmax(
    logits: &[f32],
    _n_experts: usize,
    top_k: usize,
) -> (Vec<usize>, Vec<f32>) {
    // Find top-K indices by value (descending)
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let selected: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

    // Softmax over selected logits
    let max_logit = selected
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = selected.iter().map(|(_, v)| (v - max_logit).exp()).sum();

    let expert_ids: Vec<usize> = selected.iter().map(|(id, _)| *id).collect();
    let weights: Vec<f32> = selected
        .iter()
        .map(|(_, v)| (v - max_logit).exp() / exp_sum)
        .collect();

    (expert_ids, weights)
}

/// Top-K selection with sigmoid routing (legacy, kept for tests).
///
/// For each expert: routing_weight = sigmoid(logit) * per_expert_scale[expert].
/// Then select the top-K experts by routing_weight. The returned weights are the
/// raw sigmoid * scale values (NOT renormalized).
#[allow(dead_code)]
fn top_k_sigmoid(
    logits: &[f32],
    per_expert_scale: &[f32],
    top_k: usize,
) -> (Vec<usize>, Vec<f32>) {
    let n = logits.len();
    // Compute sigmoid(logit) * per_expert_scale for each expert
    let mut scored: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let sig = 1.0 / (1.0 + (-logits[i]).exp());
            (i, sig * per_expert_scale[i])
        })
        .collect();

    // Sort by score descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let selected: Vec<(usize, f32)> = scored.into_iter().take(top_k).collect();
    let expert_ids: Vec<usize> = selected.iter().map(|(id, _)| *id).collect();
    let weights: Vec<f32> = selected.iter().map(|(_, w)| *w).collect();

    (expert_ids, weights)
}

/// Top-K selection with softmax normalization and per-expert scaling (Gemma 4 router).
///
/// Matches the Python reference:
///   1. Find top-K expert indices by raw logit score
///   2. Softmax over only those K logits
///   3. Multiply by per_expert_scale for selected experts
fn top_k_softmax_with_scale(
    logits: &[f32],
    per_expert_scale: &[f32],
    top_k: usize,
) -> (Vec<usize>, Vec<f32>) {
    // Find top-K indices by raw logit value (descending)
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let selected: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

    // Softmax over selected logits
    let max_logit = selected
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = selected.iter().map(|(_, v)| (v - max_logit).exp()).sum();

    let expert_ids: Vec<usize> = selected.iter().map(|(id, _)| *id).collect();
    // Apply softmax then multiply by per_expert_scale
    let weights: Vec<f32> = selected
        .iter()
        .map(|(id, v)| {
            let softmax_w = (v - max_logit).exp() / exp_sum;
            softmax_w * per_expert_scale[*id]
        })
        .collect();

    (expert_ids, weights)
}

/// CPU dequantization of a flat packed uint32 weight array.
///
/// Supports power-of-2 bit widths (2, 4, 8) and non-power-of-2 (3, 6).
///
/// For power-of-2 bits: each u32 contains `32/bits` quantized values.
/// For 3-bit/6-bit: MLX uses 3-byte (24-bit) triplet packing:
///   - 6-bit: 4 values per 3 bytes (4 * 6 = 24 bits)
///   - 3-bit: 8 values per 3 bytes (8 * 3 = 24 bits)
///   The byte stream is stored as u32 in safetensors.
///
/// Layout: weight `[rows, packed_cols]`, scales/biases `[rows, n_groups]`
/// where `n_groups = cols / group_size`, `packed_cols = cols * bits / 32`.
///
/// NOTE: MLX's GPU dequantization returns bf16, causing slight differences
/// from this f32 implementation. These differences cascade through MoE
/// routing across 30 layers. For exact MLX match, the GPU quantized
/// matmul kernel should be used instead.
fn cpu_dequantize_flat(
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    bits: usize,
    group_size: usize,
) -> Result<Vec<f32>, Gemma4Error> {
    let packed_cols = (cols * bits + 31) / 32; // ceil division
    let n_groups = (cols + group_size - 1) / group_size;

    let mut output = vec![0.0f32; rows * cols];

    if bits == 3 || bits == 6 {
        // Non-power-of-2: 3-byte triplet packing.
        // The packed u32 array is actually a byte stream reinterpreted as u32.
        // We need to process it as bytes.
        // Reinterpret the u32 array as a byte array (safe for little-endian)
        let packed_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                packed.as_ptr() as *const u8,
                packed.len() * 4,
            )
        };
        let vals_per_triplet = if bits == 3 { 8 } else { 4 }; // 24/3=8 or 24/6=4
        let mask = (1u32 << bits) - 1;
        let bytes_per_row = packed_cols * 4; // u32 count * 4 bytes

        for row in 0..rows {
            let row_bytes = &packed_bytes[row * bytes_per_row..];
            let scale_row = &scales[row * n_groups..];
            let bias_row = &biases[row * n_groups..];

            let mut col = 0usize;
            let mut byte_idx = 0usize;
            while col < cols {
                // Read a 3-byte triplet as a 24-bit value
                let b0 = row_bytes.get(byte_idx).copied().unwrap_or(0) as u32;
                let b1 = row_bytes.get(byte_idx + 1).copied().unwrap_or(0) as u32;
                let b2 = row_bytes.get(byte_idx + 2).copied().unwrap_or(0) as u32;
                let pack = b0 | (b1 << 8) | (b2 << 16);
                byte_idx += 3;

                for k in 0..vals_per_triplet {
                    if col >= cols {
                        break;
                    }
                    let quant_val = (pack >> (k * bits)) & mask;
                    let group_idx = col / group_size;
                    let scale = scale_row[group_idx];
                    let bias = bias_row[group_idx];
                    output[row * cols + col] = scale * (quant_val as f32) + bias;
                    col += 1;
                }
            }
        }
    } else {
        // Power-of-2 bits (2, 4, 8): standard u32 packing
        let vals_per_u32 = 32 / bits;
        let mask = (1u32 << bits) - 1;

        for row in 0..rows {
            let packed_row = &packed[row * packed_cols..];
            let scale_row = &scales[row * n_groups..];
            let bias_row = &biases[row * n_groups..];

            for col in 0..cols {
                let word_idx = col / vals_per_u32;
                let bit_offset = (col % vals_per_u32) * bits;
                let quant_val = (packed_row[word_idx] >> bit_offset) & mask;

                let group_idx = col / group_size;
                let scale = scale_row[group_idx];
                let bias = bias_row[group_idx];

                output[row * cols + col] = scale * (quant_val as f32) + bias;
            }
        }
    }

    Ok(output)
}

/// Round an f32 slice to bf16 precision.
///
/// MLX operates entirely in bf16 for quantized models. To match its behavior,
/// both matmul operands (input and dequantized weight) must be rounded to bf16
/// before multiplication. This ensures `bf16 * bf16 → f32` accumulation matches
/// MLX's fused quantized matmul kernel exactly.
fn round_to_bf16(data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect()
}

/// Convert f16 bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32 & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let mut m = mantissa;
        let mut e: i32 = -14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        let f32_exp = ((e + 127) as u32) << 23;
        let f32_mantissa = m << 13;
        return f32::from_bits(sign | f32_exp | f32_mantissa);
    }

    if exp == 31 {
        let m = if mantissa != 0 { 0x007F_FFFF } else { 0 };
        return f32::from_bits(sign | 0x7F80_0000 | m);
    }

    let f32_exp = ((exp as i32 - 15 + 127) as u32) << 23;
    let f32_mantissa = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_mantissa)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma4_config_parse() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{
                "text_config": {
                    "num_hidden_layers": 30,
                    "hidden_size": 2816,
                    "vocab_size": 262144,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "num_global_key_value_heads": 2,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "sliding_window": 1024,
                    "num_experts": 128,
                    "top_k_experts": 8,
                    "moe_intermediate_size": 704,
                    "intermediate_size": 2112,
                    "rms_norm_eps": 1e-6,
                    "final_logit_softcapping": 30.0,
                    "attention_k_eq_v": true,
                    "rope_parameters": {
                        "sliding_attention": { "rope_theta": 10000.0 },
                        "full_attention": {
                            "rope_theta": 1000000.0,
                            "partial_rotary_factor": 0.25
                        }
                    },
                    "layer_types": [
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention"
                    ]
                },
                "tie_word_embeddings": true
            }"#,
        )
        .unwrap();

        let config = Gemma4Config::from_model_config(&json).unwrap();
        assert_eq!(config.num_layers, 30);
        assert_eq!(config.hidden_size, 2816);
        assert_eq!(config.vocab_size, 262144);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_kv_heads_sliding, 8);
        assert_eq!(config.num_kv_heads_global, 2);
        assert_eq!(config.head_dim_sliding, 256);
        assert_eq!(config.head_dim_global, 512);
        assert_eq!(config.sliding_window, 1024);
        assert_eq!(config.num_experts, 128);
        assert_eq!(config.top_k_experts, 8);
        assert_eq!(config.moe_intermediate_size, 704);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.final_logit_softcapping, 30.0);
        assert!(config.attention_k_eq_v);
        assert!(config.tie_word_embeddings);

        // RoPE
        assert!((config.rope_theta_sliding - 10000.0).abs() < 1.0);
        assert!((config.rope_theta_global - 1000000.0).abs() < 1.0);
        assert!((config.partial_rotary_factor_global - 0.25).abs() < 0.01);

        // Layer types
        assert_eq!(config.layer_types.len(), 6);
        assert_eq!(config.layer_types[0], LayerAttentionType::Sliding);
        assert_eq!(config.layer_types[5], LayerAttentionType::Global);

        // Per-layer accessors
        assert_eq!(config.n_kv_heads(0), 8);
        assert_eq!(config.n_kv_heads(5), 2);
        assert_eq!(config.head_dim(0), 256);
        assert_eq!(config.head_dim(5), 512);
        assert_eq!(config.rope_dim(0), 256); // full rotation
        assert_eq!(config.rope_dim(5), 128); // 0.25 * 512
    }

    #[test]
    fn test_layer_type_default_pattern() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{
                "text_config": {
                    "num_hidden_layers": 12,
                    "hidden_size": 64,
                    "vocab_size": 100,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2
                }
            }"#,
        )
        .unwrap();

        let config = Gemma4Config::from_model_config(&json).unwrap();
        assert_eq!(config.layer_types.len(), 12);
        // Pattern: 5 sliding + 1 global, repeating
        assert_eq!(config.layer_type(0), LayerAttentionType::Sliding);
        assert_eq!(config.layer_type(4), LayerAttentionType::Sliding);
        assert_eq!(config.layer_type(5), LayerAttentionType::Global);
        assert_eq!(config.layer_type(6), LayerAttentionType::Sliding);
        assert_eq!(config.layer_type(11), LayerAttentionType::Global);
    }

    #[test]
    fn test_gelu_pytorch_tanh_known_values() {
        // GELU(0) = 0
        assert!((gelu_pytorch_tanh(0.0) - 0.0).abs() < 1e-6);

        // GELU(1) ~ 0.8412
        assert!((gelu_pytorch_tanh(1.0) - 0.8412).abs() < 1e-3);

        // GELU(-1) ~ -0.1588
        assert!((gelu_pytorch_tanh(-1.0) - (-0.1588)).abs() < 1e-3);
    }

    #[test]
    fn test_top_k_softmax() {
        let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let (ids, weights) = top_k_softmax(&logits, 5, 2);

        assert_eq!(ids.len(), 2);
        assert_eq!(weights.len(), 2);
        assert_eq!(ids[0], 1); // index of 5.0
        assert_eq!(ids[1], 4); // index of 4.0

        // Weights should sum to 1.0
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // First weight should be larger (higher logit)
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn test_reshape_qkv_roundtrip() {
        let seq_len = 3;
        let n_heads = 2;
        let head_dim = 4;
        let total_dim = n_heads * head_dim;

        // Create input [seq_len, n_heads * head_dim]
        let input: Vec<f32> = (0..(seq_len * total_dim))
            .map(|i| i as f32)
            .collect();

        let reshaped = reshape_qkv_for_sdpa(&input, seq_len, n_heads, head_dim);
        assert_eq!(reshaped.len(), n_heads * seq_len * head_dim);

        let back = reshape_sdpa_output(&reshaped, seq_len, n_heads, head_dim);
        assert_eq!(back.len(), input.len());
        assert_eq!(back, input);
    }

    #[test]
    fn test_f16_conversion_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 65504.0, -0.001];
        for &v in &values {
            let bits = f32_to_f16_bits(v);
            let back = f16_to_f32(bits);
            let tol = if v.abs() > 1000.0 { v.abs() * 0.002 } else { 0.002 };
            assert!(
                (back - v).abs() < tol,
                "f16 roundtrip failed: {v} -> bits={bits:#06x} -> {back}"
            );
        }
    }

    /// Helper to convert f32 to f16 bits.
    fn f32_to_f16_bits(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x007F_FFFF;

        if exp == 255 {
            let m = if mantissa != 0 { 0x0200 } else { 0 };
            return (sign | 0x7C00 | m) as u16;
        }

        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            return (sign | 0x7C00) as u16;
        }
        if new_exp <= 0 {
            if new_exp < -10 {
                return sign as u16;
            }
            let m = (mantissa | 0x0080_0000) >> (1 - new_exp + 13);
            return (sign | m) as u16;
        }

        let m = mantissa >> 13;
        let round_bit = (mantissa >> 12) & 1;
        let sticky = if (mantissa & 0xFFF) != 0 { 1u32 } else { 0 };
        let round_up = round_bit & (sticky | m);
        (sign | ((new_exp as u32) << 10) | m + round_up) as u16
    }

    #[test]
    fn test_top_k_sigmoid_basic() {
        // 5 experts, select top 2
        let logits = vec![0.0, 2.0, -1.0, 1.0, 3.0];
        let per_expert_scale = vec![1.0; 5];
        let (ids, weights) = top_k_sigmoid(&logits, &per_expert_scale, 2);

        assert_eq!(ids.len(), 2);
        assert_eq!(weights.len(), 2);
        // Expert 4 (logit=3.0) and expert 1 (logit=2.0) should be top 2
        assert_eq!(ids[0], 4);
        assert_eq!(ids[1], 1);
        // sigmoid(3.0) ~ 0.9526, sigmoid(2.0) ~ 0.8808
        assert!((weights[0] - 0.9526).abs() < 0.01);
        assert!((weights[1] - 0.8808).abs() < 0.01);
    }

    #[test]
    fn test_top_k_sigmoid_with_scale() {
        // per_expert_scale can reorder experts
        let logits = vec![2.0, 1.0];
        let per_expert_scale = vec![0.1, 10.0]; // scale suppresses expert 0, boosts expert 1
        let (ids, weights) = top_k_sigmoid(&logits, &per_expert_scale, 1);

        assert_eq!(ids[0], 1); // expert 1 wins despite lower logit
        // sigmoid(1.0) * 10.0 ~ 7.31
        assert!(weights[0] > 5.0);
    }

    #[test]
    fn test_cpu_dequantize_flat_4bit() {
        // 2 rows, 8 cols, 4-bit, group_size=4
        // Each u32 holds 8 x 4-bit values = 8 values
        // packed_cols = 8 * 4 / 32 = 1 u32 per row
        // n_groups = 8 / 4 = 2
        let rows = 2;
        let cols = 8;
        let bits = 4;
        let group_size = 4;

        // Pack values 0..7 into row 0, and 8..15 into row 1
        // 4-bit packing: val[0] in bits 0-3, val[1] in bits 4-7, etc.
        let row0_packed: u32 = 0 | (1 << 4) | (2 << 8) | (3 << 12)
            | (4 << 16) | (5 << 20) | (6 << 24) | (7 << 28);
        let row1_packed: u32 = 8 | (9 << 4) | (10 << 8) | (11 << 12)
            | (12 << 16) | (13 << 20) | (14 << 24) | (15 << 28);
        let packed = vec![row0_packed, row1_packed];

        // scale=1.0, bias=0.0 for identity transform
        let scales = vec![1.0f32; rows * 2]; // 2 groups per row
        let biases = vec![0.0f32; rows * 2];

        let result = cpu_dequantize_flat(&packed, &scales, &biases, rows, cols, bits, group_size).unwrap();
        assert_eq!(result.len(), 16);
        for i in 0..16 {
            assert!(
                (result[i] - i as f32).abs() < 1e-6,
                "result[{}] = {}, expected {}",
                i, result[i], i,
            );
        }
    }

    #[test]
    fn test_cpu_dequantize_flat_affine() {
        // Test scale=2.0, bias=-3.0
        let rows = 1;
        let cols = 4;
        let bits = 4;
        let group_size = 4;

        // Pack values [1, 2, 3, 4]
        let packed = vec![1u32 | (2 << 4) | (3 << 8) | (4 << 12)];
        let scales = vec![2.0f32]; // 1 group
        let biases = vec![-3.0f32];

        let result = cpu_dequantize_flat(&packed, &scales, &biases, rows, cols, bits, group_size).unwrap();
        // 2.0 * 1 + (-3.0) = -1.0
        // 2.0 * 2 + (-3.0) = 1.0
        // 2.0 * 3 + (-3.0) = 3.0
        // 2.0 * 4 + (-3.0) = 5.0
        assert!((result[0] - (-1.0)).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
        assert!((result[3] - 5.0).abs() < 1e-6);
    }
}
