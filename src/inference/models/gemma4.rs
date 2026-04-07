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

        // Step 1: Embedding gather
        debug!(seq_len = seq_len, "Embedding gather");
        let mut hidden = self.embedding_gather(tokens)?;

        if diag {
            let s: &[f32] = hidden.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read embed: {e}"),
            })?;
            let n = s.len().min(5);
            eprintln!("[DIAG] After embedding_gather (first {}): {:?}", n, &s[..n]);
            let nonzero = s.iter().filter(|v| **v != 0.0).count();
            eprintln!("[DIAG]   total_elements={}, nonzero={}", s.len(), nonzero);
        }

        // Step 2: Scale embeddings by sqrt(hidden_size) as Gemma convention
        // Gemma models scale embeddings by sqrt(hidden_size)
        let scale = (hidden_size as f32).sqrt();
        {
            let slice: &mut [f32] = hidden
                .as_mut_slice()
                .map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("Embedding scale: {e}"),
                })?;
            for v in slice.iter_mut() {
                *v *= scale;
            }

            if diag {
                let n = slice.len().min(5);
                eprintln!("[DIAG] After embedding scale (factor={:.4}), first {}: {:?}",
                    scale, n, &slice[..n]);
            }
        }

        // Step 3: Process each transformer layer
        for layer_idx in 0..self.config.num_layers {
            debug!(layer = layer_idx, "Processing transformer layer");
            hidden = self.forward_layer(layer_idx, &hidden, seq_len)?;

            if diag && layer_idx == 0 {
                let s: &[f32] = hidden.as_slice().map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("diag read layer0: {e}"),
                })?;
                let n = s.len().min(5);
                eprintln!("[DIAG] After layer 0 output, first {}: {:?}", n, &s[..n]);
                // Check for NaN/Inf
                let nan_count = s.iter().filter(|v| v.is_nan()).count();
                let inf_count = s.iter().filter(|v| v.is_infinite()).count();
                if nan_count > 0 || inf_count > 0 {
                    eprintln!("[DIAG]   WARNING: NaN={}, Inf={}", nan_count, inf_count);
                }
                // Print last 5 too for diversity check
                let last_n = s.len().min(5);
                eprintln!("[DIAG]   last {}: {:?}", last_n, &s[s.len()-last_n..]);
            }
        }

        // Step 4: Final RMS norm
        debug!("Final RMS norm");
        let normed = self.apply_rms_norm(&hidden, "model.norm.weight", seq_len)?;

        if diag {
            let s: &[f32] = normed.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read final norm: {e}"),
            })?;
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

        // Step 6: Softcap
        debug!(cap = self.config.final_logit_softcapping, "Softcap");
        let capped_logits = self.apply_softcap(&logits, seq_len)?;

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

        // Step 1: Embedding gather
        debug!(seq_len = seq_len, "Embedding gather (hidden states)");
        let mut hidden = self.embedding_gather(tokens)?;

        // Step 2: Scale embeddings by sqrt(hidden_size) as Gemma convention
        let scale = (hidden_size as f32).sqrt();
        {
            let slice: &mut [f32] = hidden
                .as_mut_slice()
                .map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("Embedding scale: {e}"),
                })?;
            for v in slice.iter_mut() {
                *v *= scale;
            }
        }

        // Step 3: Process each transformer layer
        for layer_idx in 0..self.config.num_layers {
            debug!(layer = layer_idx, "Processing transformer layer (hidden states)");
            hidden = self.forward_layer(layer_idx, &hidden, seq_len)?;
        }

        // Step 4: Final RMS norm (stop here -- do NOT project to vocab)
        debug!("Final RMS norm (hidden states)");
        let normed = self.apply_rms_norm(&hidden, "model.norm.weight", seq_len)?;

        // Read hidden states to CPU
        let output: &[f32] = normed
            .as_slice()
            .map_err(|e| Gemma4Error::ForwardError {
                reason: format!("Read hidden states: {e}"),
            })?;

        Ok(output.to_vec())
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
        let mut hidden = self.embedding_gather(&safe_tokens)?;

        // Step 2: Scale embeddings by sqrt(hidden_size) as Gemma convention
        let scale = (hidden_size as f32).sqrt();
        {
            let slice: &mut [f32] = hidden
                .as_mut_slice()
                .map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("Embedding scale: {e}"),
                })?;
            for v in slice.iter_mut() {
                *v *= scale;
            }
        }

        // Step 3: Scatter vision features into image token positions
        // The vision features are already in the text model's hidden dimension
        // and have been processed by the vision encoder + projection.
        {
            let slice: &mut [f32] = hidden
                .as_mut_slice()
                .map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("Vision scatter: {e}"),
                })?;
            let mut vision_idx = 0usize;
            for (pos, &token_id) in tokens.iter().enumerate() {
                if token_id == image_token_id {
                    let src_offset = vision_idx * hidden_size;
                    let dst_offset = pos * hidden_size;
                    if src_offset + hidden_size <= vision_features.len() {
                        slice[dst_offset..dst_offset + hidden_size]
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

        // Step 4: Process each transformer layer
        for layer_idx in 0..self.config.num_layers {
            debug!(layer = layer_idx, "Processing transformer layer");
            hidden = self.forward_layer(layer_idx, &hidden, seq_len)?;
        }

        // Step 5: Final RMS norm
        debug!("Final RMS norm");
        let normed = self.apply_rms_norm(&hidden, "model.norm.weight", seq_len)?;

        // Step 6: lm_head (tied embeddings)
        debug!("lm_head projection (tied embeddings)");
        let logits = self.lm_head_projection(&normed, seq_len)?;

        // Step 7: Softcap
        debug!(cap = self.config.final_logit_softcapping, "Softcap");
        let capped_logits = self.apply_softcap(&logits, seq_len)?;

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
            let s: &[f32] = normed.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read norm L0: {e}"),
            })?;
            let n = s.len().min(5);
            eprintln!("[DIAG] Layer 0 after rms_norm, first {}: {:?}", n, &s[..n]);
        }

        // --- Step 2: Q/K/V projections ---
        // For Gemma 4: Q has n_heads * head_dim output dims,
        // K and V each have n_kv_heads * head_dim output dims.
        // If attention_k_eq_v, K and V share the same projection weight.
        let q_out_dim = n_heads * head_dim;
        let kv_out_dim = n_kv_heads * head_dim;

        let q_proj = self.quantized_projection(
            &normed,
            &format!("model.layers.{layer_idx}.self_attn.q_proj"),
            seq_len,
            hidden_size,
            q_out_dim,
        )?;

        if diag {
            let s: &[f32] = q_proj.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read q_proj L0: {e}"),
            })?;
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

        let v_proj = if self.config.attention_k_eq_v {
            // K == V: reuse the K projection output as V
            // Note: we need to create a separate buffer copy because they might
            // diverge after RoPE (RoPE only applies to K, not V)
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

        // --- Step 3: RoPE on Q and K ---
        // For global attention with partial_rotary_factor, only apply RoPE to
        // the first rope_dim dimensions of each head, leaving the rest unchanged.
        let rope_dim = self.config.rope_dim(layer_idx);
        let theta = self.config.rope_theta(layer_idx);

        let q_roped = self.apply_rope_to_heads(
            &q_proj, seq_len, n_heads, head_dim, rope_dim, theta,
        )?;

        let k_roped = self.apply_rope_to_heads(
            &k_proj, seq_len, n_kv_heads, head_dim, rope_dim, theta,
        )?;

        // --- Step 4: KV cache append ---
        // Read K and V to CPU, append to cache, then read back cache contents
        let k_data = read_buffer_f32(&k_roped)?;
        let v_data = read_buffer_f32(&v_proj)?; // V does NOT get RoPE

        {
            let layer_cache = self.kv_cache.layer_mut(layer_idx)?;
            layer_cache.append(&k_data, &v_data)?;
        }

        // --- Step 5: Attention ---
        let attn_out = self.compute_attention(
            layer_idx,
            &q_roped,
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            layer_type,
        )?;

        if diag {
            let s: &[f32] = attn_out.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read attn L0: {e}"),
            })?;
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

        // --- Step 7: Apply layer_scalar and residual add ---
        // Gemma 4 has a per-layer scalar: hidden = input + layer_scalar * attn_output
        let scaled_attn = self.apply_layer_scalar(layer_idx, &o_proj, seq_len * hidden_size)?;
        let residual1 = self.elementwise_add(input, &scaled_attn, seq_len * hidden_size)?;

        // --- Step 8: Post-attention RMS norm ---
        let post_norm_name = format!(
            "model.layers.{layer_idx}.post_attention_layernorm.weight"
        );
        let post_normed = self.apply_rms_norm(&residual1, &post_norm_name, seq_len)?;

        // --- Step 9: MoE FFN ---
        let ffn_out = self.moe_ffn(layer_idx, &post_normed, seq_len)?;

        if diag {
            let s: &[f32] = ffn_out.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read ffn L0: {e}"),
            })?;
            let n = s.len().min(5);
            eprintln!("[DIAG] Layer 0 after MoE FFN output, first {}: {:?}", n, &s[..n]);
            let nan_count = s.iter().filter(|v| v.is_nan()).count();
            let inf_count = s.iter().filter(|v| v.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                eprintln!("[DIAG]   WARNING ffn: NaN={}, Inf={}", nan_count, inf_count);
            }
        }

        // --- Step 10: Apply layer_scalar to FFN output and residual add ---
        let scaled_ffn = self.apply_layer_scalar(layer_idx, &ffn_out, seq_len * hidden_size)?;
        let output = self.elementwise_add(&residual1, &scaled_ffn, seq_len * hidden_size)?;

        Ok(output)
    }

    /// Apply RMS normalization using a named weight tensor.
    fn apply_rms_norm(
        &mut self,
        input: &MlxBuffer,
        weight_name: &str,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let hidden_size = self.config.hidden_size;

        // Clone the weight buffer so we can release the immutable borrow on self
        let weight = require_weight(&self.weights, weight_name)?;
        let norm_weight_buf = clone_buffer(&self.device, &weight.buffer)?;

        // RMS norm needs: input [rows, dim], weight [dim], output [rows, dim], params_buf [eps, dim]
        let output_bytes = seq_len * hidden_size * std::mem::size_of::<f32>();
        let output = self.device.alloc_buffer(
            output_bytes,
            DType::F32,
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

        // Norm weights may be f32 or need conversion
        let norm_weight = self.ensure_f32_weight(&norm_weight_buf, hidden_size, weight_name)?;

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
    /// Input is f32 [seq_len, in_dim]; output is f16 [seq_len, out_dim] (converted to f32).
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
        if let Some((bits, group_size)) = quant_info {
            if bits == 4 || bits == 6 {
                // Cast input to f16 first (releases immutable borrow on self.weights)
                let input_f16 = self.cast_to_f16(input, seq_len * in_dim)?;

                // Now re-borrow the weights we need
                let loaded = require_weight(&self.weights, &weight_name)?;
                let scales = require_weight(&self.weights, &scales_name)?;
                let biases = require_weight(&self.weights, &biases_name)?;

                let params = QuantizedMatmulParams {
                    m: seq_len as u32,
                    k: in_dim as u32,
                    n: out_dim as u32,
                    group_size: group_size as u32,
                    bits: bits as u32,
                };

                let mut encoder = self.device.command_encoder()?;
                let output_f16 = qmatmul::quantized_matmul(
                    &mut encoder,
                    &mut self.registry,
                    &self.device,
                    &input_f16,
                    &loaded.buffer,
                    &scales.buffer,
                    &biases.buffer,
                    &params,
                )?;
                encoder.commit_and_wait()?;

                // Cast output back to f32
                let output_f32 = self.cast_to_f32(&output_f16, seq_len * out_dim)?;
                return Ok(output_f32);
            }
        }

        // Non-quantized path: CPU matmul fallback
        // Need to clone the weight buffer to avoid borrow conflict with self.cpu_matmul
        let weight_buf = clone_buffer(
            &self.device,
            &require_weight(&self.weights, &weight_name)?.buffer,
        )?;
        self.cpu_matmul(input, &weight_buf, seq_len, in_dim, out_dim)
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

        // Write to Metal buffer
        let mut buf = self.device.alloc_buffer(
            output_data.len() * std::mem::size_of::<f32>(),
            DType::F32,
            input.shape().to_vec(),
        )?;
        {
            let slice: &mut [f32] = buf.as_mut_slice().map_err(|e| {
                Gemma4Error::ForwardError {
                    reason: format!("RoPE output write: {e}"),
                }
            })?;
            slice.copy_from_slice(&output_data);
        }
        Ok(buf)
    }

    /// Compute attention for a single layer.
    ///
    /// Reshapes Q from [seq_len, n_heads * head_dim] to [1, n_heads, seq_len, head_dim],
    /// reads K and V from the KV cache, dispatches sdpa or sdpa_sliding.
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

        // Q is [seq_len, n_heads * head_dim] in memory.
        // SDPA expects [batch, n_heads, seq_len, head_dim].
        // We need to transpose: reshape to [seq_len, n_heads, head_dim] then
        // permute to [n_heads, seq_len, head_dim] and add batch dim.
        // Since batch=1, the layout is [1, n_heads, seq_len, head_dim].

        // Do the reshape/transpose on CPU for correctness:
        let q_data = read_buffer_f32(q)?;
        let q_reshaped = reshape_qkv_for_sdpa(&q_data, seq_len, n_heads, head_dim);

        // K cache is [kv_seq_len, n_kv_heads * head_dim]
        // Read and reshape to [1, n_kv_heads, kv_seq_len, head_dim]
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

        // Allocate GPU buffers
        let q_buf = f32_vec_to_buffer(
            &self.device,
            &q_reshaped,
            vec![1, n_heads, seq_len, head_dim],
        )?;
        let k_buf = f32_vec_to_buffer(
            &self.device,
            &k_reshaped,
            vec![1, n_kv_heads, kv_seq_len, head_dim],
        )?;
        let v_buf = f32_vec_to_buffer(
            &self.device,
            &v_reshaped,
            vec![1, n_kv_heads, kv_seq_len, head_dim],
        )?;

        let output_elements = 1 * n_heads * seq_len * head_dim;
        let output = self.device.alloc_buffer(
            output_elements * std::mem::size_of::<f32>(),
            DType::F32,
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
        let out_data = read_buffer_f32(&output)?;
        let out_flat = reshape_sdpa_output(&out_data, seq_len, n_heads, head_dim);

        f32_vec_to_buffer(
            &self.device,
            &out_flat,
            vec![seq_len, n_heads * head_dim],
        )
    }

    /// Apply per-layer scalar: output = layer_scalar * input.
    /// Gemma 4 has a learned scalar per layer that scales residual contributions.
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
                let scalar_val = if w.buffer.byte_len() >= 4 {
                    // f32
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

                // Multiply input by scalar
                let input_data: &[f32] = input.as_slice().map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("read input for layer_scalar: {e}"),
                })?;
                let scaled: Vec<f32> = input_data.iter().map(|v| v * scalar_val).collect();
                f32_vec_to_buffer(&self.device, &scaled, vec![n_elements])
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

    /// Elementwise add of two f32 buffers.
    fn elementwise_add(
        &mut self,
        a: &MlxBuffer,
        b: &MlxBuffer,
        n_elements: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let output_bytes = n_elements * std::mem::size_of::<f32>();
        let output = self.device.alloc_buffer(
            output_bytes,
            DType::F32,
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
            DType::F32,
        )?;
        encoder.commit_and_wait()?;

        Ok(output)
    }

    /// MoE FFN for a single layer.
    ///
    /// For each token in the sequence:
    /// 1. Router: matmul hidden_state with gate weight to get [n_experts] logits
    /// 2. Top-K selection with softmax routing weights
    /// 3. Dispatch to selected experts (gate_proj, up_proj, down_proj with GELU)
    /// 4. Weighted sum of expert outputs
    fn moe_ffn(
        &mut self,
        layer_idx: usize,
        input: &MlxBuffer,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let n_experts = self.config.num_experts;
        let top_k = self.config.top_k_experts;
        let diag = std::env::var("HF2Q_DIAG").is_ok() && layer_idx == 0;

        // Router weight: [n_experts, hidden_size]
        let router_weight_name =
            format!("model.layers.{layer_idx}.router.proj.weight");

        let input_data = read_buffer_f32(input)?;
        let mut output_data = vec![0.0f32; seq_len * hidden_size];

        // Check if router weight exists -- if not, check for a dense FFN
        let has_moe_router = self.weights.get(&router_weight_name).is_some();

        if diag {
            // Also check with the language_model prefix
            let alt_name = format!("language_model.{}", router_weight_name);
            let has_alt = self.weights.get(&alt_name).is_some();
            eprintln!("[DIAG] MoE layer 0: has_moe_router={} (checked '{}'), alt_prefix={}",
                has_moe_router, router_weight_name, has_alt);
            if !has_moe_router && !has_alt {
                // List some available weight keys for this layer
                let prefix = format!("model.layers.{layer_idx}.");
                let alt_prefix = format!("language_model.model.layers.{layer_idx}.");
                let matching: Vec<_> = self.weights.weights.keys()
                    .filter(|k| k.starts_with(&prefix) || k.starts_with(&alt_prefix))
                    .take(20)
                    .collect();
                eprintln!("[DIAG]   Available weight keys for layer {}: {:?}", layer_idx, matching);
            }
        }

        if !has_moe_router {
            if diag {
                eprintln!("[DIAG] Falling back to dense FFN for layer {}", layer_idx);
            }
            // Dense FFN fallback (some layers might not have MoE)
            return self.dense_ffn(layer_idx, input, seq_len);
        }

        if diag {
            // Check router weight info
            let rw = require_weight(&self.weights, &router_weight_name)?;
            eprintln!("[DIAG] router weight: dtype={}, shape={:?}, byte_len={}",
                rw.buffer.dtype(), rw.buffer.shape(), rw.buffer.byte_len());
        }

        // Process token by token for MoE (each token routes to different experts)
        for t in 0..seq_len {
            let token_hidden = &input_data[t * hidden_size..(t + 1) * hidden_size];

            // Step 1: Router -- CPU matmul for the gate
            let router_data = self.cpu_router_forward(layer_idx, token_hidden)?;

            if diag && t == 0 {
                let mut router_sorted: Vec<(usize, f32)> = router_data.iter().copied().enumerate().collect();
                router_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                eprintln!("[DIAG] MoE L0 T0 router top-5: {:?}", &router_sorted[..5.min(router_sorted.len())]);
                let nan_count = router_data.iter().filter(|v| v.is_nan()).count();
                if nan_count > 0 {
                    eprintln!("[DIAG]   WARNING: router has {} NaN values", nan_count);
                }
            }

            // Step 2: Top-K selection and softmax
            let (expert_ids, routing_weights) =
                top_k_softmax(&router_data, n_experts, top_k);

            if diag && t == 0 {
                eprintln!("[DIAG] MoE L0 T0 selected experts: {:?}, weights: {:?}",
                    expert_ids, routing_weights);
            }

            // Step 3: Dispatch to selected experts and accumulate
            let mut token_output = vec![0.0f32; hidden_size];

            for (rank, &expert_id) in expert_ids.iter().enumerate() {
                let weight = routing_weights[rank];
                if weight.abs() < 1e-10 {
                    continue;
                }

                let expert_output =
                    self.expert_ffn(layer_idx, expert_id, token_hidden)?;

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

        f32_vec_to_buffer(
            &self.device,
            &output_data,
            vec![seq_len, hidden_size],
        )
    }

    /// CPU router forward: matmul hidden_state with gate weight.
    fn cpu_router_forward(
        &self,
        layer_idx: usize,
        token_hidden: &[f32],
    ) -> Result<Vec<f32>, Gemma4Error> {
        let router_name =
            format!("model.layers.{layer_idx}.router.proj.weight");
        let router_weight = require_weight(&self.weights, &router_name)?;
        let router_data = read_weight_as_f32(router_weight)?;

        let hidden_size = self.config.hidden_size;
        let n_experts = self.config.num_experts;

        // router_weight is [n_experts, hidden_size]
        let mut logits = vec![0.0f32; n_experts];
        for e in 0..n_experts {
            let mut sum = 0.0f32;
            for k in 0..hidden_size {
                sum += router_data[e * hidden_size + k] * token_hidden[k];
            }
            logits[e] = sum;
        }

        Ok(logits)
    }

    /// Run a single expert's FFN: gate_proj(x) * GELU(up_proj(x)), then down_proj.
    fn expert_ffn(
        &self,
        layer_idx: usize,
        expert_id: usize,
        token_hidden: &[f32],
    ) -> Result<Vec<f32>, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.moe_intermediate_size;
        // Only diagnose for first call on layer 0
        let diag = std::env::var("HF2Q_DIAG").is_ok() && layer_idx == 0;
        // Use a simple static to only print once
        static DIAG_PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let diag = diag && !DIAG_PRINTED.swap(true, std::sync::atomic::Ordering::Relaxed);

        // Weight names for expert
        let base = format!(
            "model.layers.{layer_idx}.experts.switch_glu"
        );

        let gate_name = format!("{base}.gate_proj.weight");
        let up_name = format!("{base}.up_proj.weight");
        let down_name = format!("{base}.down_proj.weight");

        if diag {
            // Check what the weight looks like
            let gw = require_weight(&self.weights, &gate_name)?;
            eprintln!("[DIAG] expert_ffn L0 E{}: gate_proj dtype={}, shape={:?}, byte_len={}",
                expert_id, gw.buffer.dtype(), gw.buffer.shape(), gw.buffer.byte_len());
            if let Some(ref qm) = gw.quant_meta {
                eprintln!("[DIAG]   gate_proj quant: bits={}, group_size={}", qm.bits, qm.group_size);
            }
            let uw = require_weight(&self.weights, &up_name)?;
            eprintln!("[DIAG]   up_proj dtype={}, shape={:?}", uw.buffer.dtype(), uw.buffer.shape());
            let dw = require_weight(&self.weights, &down_name)?;
            eprintln!("[DIAG]   down_proj dtype={}, shape={:?}", dw.buffer.dtype(), dw.buffer.shape());
        }

        let gate_data = self.read_expert_weight(&gate_name,
            intermediate_size, hidden_size)?;
        let up_data = self.read_expert_weight(&up_name,
            intermediate_size, hidden_size)?;
        let down_data = self.read_expert_weight(&down_name,
            hidden_size, intermediate_size)?;

        if diag {
            eprintln!("[DIAG] expert_ffn L0 E{}: gate_data len={} (expect {}x{}={}), first 5: {:?}",
                expert_id, gate_data.len(), intermediate_size, hidden_size,
                intermediate_size * hidden_size, &gate_data[..5.min(gate_data.len())]);
            eprintln!("[DIAG]   up_data len={}, first 5: {:?}",
                up_data.len(), &up_data[..5.min(up_data.len())]);
            eprintln!("[DIAG]   down_data len={} (expect {}x{}={}), first 5: {:?}",
                down_data.len(), hidden_size, intermediate_size,
                hidden_size * intermediate_size, &down_data[..5.min(down_data.len())]);
            let gate_nan = gate_data.iter().filter(|v| v.is_nan()).count();
            let up_nan = up_data.iter().filter(|v| v.is_nan()).count();
            let down_nan = down_data.iter().filter(|v| v.is_nan()).count();
            if gate_nan > 0 || up_nan > 0 || down_nan > 0 {
                eprintln!("[DIAG]   NaN in weights! gate={}, up={}, down={}",
                    gate_nan, up_nan, down_nan);
            }
            eprintln!("[DIAG]   input token_hidden first 5: {:?}, len={}",
                &token_hidden[..5.min(token_hidden.len())], token_hidden.len());
        }

        // gate_out = gate_proj @ x   [intermediate_size]
        let mut gate_out = vec![0.0f32; intermediate_size];
        for i in 0..intermediate_size {
            let mut sum = 0.0f32;
            for k in 0..hidden_size {
                sum += gate_data[i * hidden_size + k] * token_hidden[k];
            }
            gate_out[i] = sum;
        }

        if diag {
            let n = gate_out.len().min(5);
            eprintln!("[DIAG] expert_ffn L0 E{}: gate_out first {}: {:?}",
                expert_id, n, &gate_out[..n]);
            let nan_count = gate_out.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                eprintln!("[DIAG]   WARNING: gate_out has {} NaN", nan_count);
            }
        }

        // up_out = up_proj @ x   [intermediate_size]
        let mut up_out = vec![0.0f32; intermediate_size];
        for i in 0..intermediate_size {
            let mut sum = 0.0f32;
            for k in 0..hidden_size {
                sum += up_data[i * hidden_size + k] * token_hidden[k];
            }
            up_out[i] = sum;
        }

        if diag {
            let n = up_out.len().min(5);
            eprintln!("[DIAG] expert_ffn L0 E{}: up_out first {}: {:?}",
                expert_id, n, &up_out[..n]);
            let nan_count = up_out.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                eprintln!("[DIAG]   WARNING: up_out has {} NaN", nan_count);
            }
        }

        // hidden = GELU(gate_out) * up_out
        let mut hidden = vec![0.0f32; intermediate_size];
        for i in 0..intermediate_size {
            hidden[i] = gelu_pytorch_tanh(gate_out[i]) * up_out[i];
        }

        if diag {
            let n = hidden.len().min(5);
            eprintln!("[DIAG] expert_ffn L0 E{}: after GELU*up first {}: {:?}",
                expert_id, n, &hidden[..n]);
        }

        // expert_out = down_proj @ hidden   [hidden_size]
        let mut expert_out = vec![0.0f32; hidden_size];
        for i in 0..hidden_size {
            let mut sum = 0.0f32;
            for k in 0..intermediate_size {
                sum += down_data[i * intermediate_size + k] * hidden[k];
            }
            expert_out[i] = sum;
        }

        if diag {
            let n = expert_out.len().min(5);
            eprintln!("[DIAG] expert_ffn L0 E{}: final expert_out first {}: {:?}",
                expert_id, n, &expert_out[..n]);
            let nan_count = expert_out.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                eprintln!("[DIAG]   WARNING: expert_out has {} NaN out of {}", nan_count, expert_out.len());
            }
        }

        Ok(expert_out)
    }

    /// Read an expert weight tensor as f32 (handling quantized or float formats).
    fn read_expert_weight(
        &self,
        name: &str,
        _rows: usize,
        _cols: usize,
    ) -> Result<Vec<f32>, Gemma4Error> {
        let loaded = require_weight(&self.weights, name)?;
        read_weight_as_f32(loaded)
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
            let s: &[f32] = gate.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag dense gate: {e}"),
            })?;
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
            let s: &[f32] = up.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag dense up: {e}"),
            })?;
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

        let hidden_buf = f32_vec_to_buffer(
            &self.device,
            &hidden_data,
            vec![seq_len, intermediate_size],
        )?;

        // down_proj
        let result = self.quantized_projection(
            &hidden_buf,
            &format!("model.layers.{layer_idx}.mlp.down_proj"),
            seq_len,
            intermediate_size,
            hidden_size,
        )?;

        if diag {
            let s: &[f32] = result.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag dense down: {e}"),
            })?;
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
                // For quantized lm_head, use quantized_matmul
                // embed_weight is [vocab_size, hidden_size] quantized
                // We want: input [seq_len, hidden_size] @ embed_weight^T => [seq_len, vocab_size]
                // quantized_matmul computes: output = input @ dequant(weight)^T
                // where weight is [N, K] = [vocab_size, hidden_size]
                // So N=vocab_size, K=hidden_size, M=seq_len. This matches.
                return self.quantized_projection(
                    input,
                    "model.embed_tokens",
                    seq_len,
                    hidden_size,
                    vocab_size,
                );
            }
        }

        // Unquantized: CPU matmul
        // embed_weight is [vocab_size, hidden_size]
        // We want: input @ embed_weight^T => [seq_len, vocab_size]
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

/// GELU activation with PyTorch tanh approximation.
///
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
fn gelu_pytorch_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = 0.7978845608028654;
    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Top-K selection with softmax normalization over the selected experts.
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
}
