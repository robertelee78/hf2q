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
    elementwise, embedding, fused_head_norm_rope, fused_residual_norm, gelu, kv_cache_copy,
    moe_dispatch as moe_ops, moe_gate, quantized_matmul as qmatmul, rms_norm, rope, sdpa,
    sdpa_sliding, softcap, transpose,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice, QuantizedMatmulParams};
use std::collections::HashMap;
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
    /// Number of KV-shared layers (last N layers reuse K/V from donor layers).
    /// Default 0 (no sharing). For 27B models, typically 20.
    pub num_kv_shared_layers: usize,
    /// MoE intermediate size for KV-shared layers (double-wide).
    /// Default = moe_intermediate_size * 2 when num_kv_shared_layers > 0.
    pub moe_intermediate_size_shared: usize,
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

        // KV sharing: last N layers reuse K/V from earlier donor layers
        let num_kv_shared_layers = get_u64("num_kv_shared_layers").unwrap_or(0) as usize;

        // Double-wide MoE intermediate size for KV-shared layers
        let moe_intermediate_size_shared = get_u64("moe_intermediate_size_shared")
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                if num_kv_shared_layers > 0 {
                    moe_intermediate_size * 2
                } else {
                    moe_intermediate_size
                }
            });

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
            num_kv_shared_layers,
            moe_intermediate_size_shared,
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

    /// Returns true if this layer is a KV-shared layer (reuses K/V from a donor).
    ///
    /// KV-shared layers are the last `num_kv_shared_layers` layers. They skip
    /// K/V projection and read K/V from the most recent earlier (donor) layer
    /// of the same attention type that is NOT itself a shared layer.
    pub fn is_kv_shared_layer(&self, layer_idx: usize) -> bool {
        self.num_kv_shared_layers > 0
            && layer_idx >= self.num_layers.saturating_sub(self.num_kv_shared_layers)
    }

    /// For a KV-shared layer, find the donor layer index.
    ///
    /// The donor is the most recent earlier layer with the same attention type
    /// (sliding/global) that is NOT itself a KV-shared layer.
    /// Returns `None` if `layer_idx` is not a shared layer or no donor exists.
    pub fn donor_layer_for(&self, layer_idx: usize) -> Option<usize> {
        if !self.is_kv_shared_layer(layer_idx) {
            return None;
        }
        let my_type = self.layer_type(layer_idx);
        let first_shared = self.num_layers.saturating_sub(self.num_kv_shared_layers);
        // Search backwards from the layer just before the shared region
        // (or from layer_idx - 1 if layer_idx < first_shared, which shouldn't happen)
        let search_end = layer_idx.min(first_shared);
        for i in (0..search_end).rev() {
            if self.layer_type(i) == my_type {
                return Some(i);
            }
        }
        // If no non-shared donor of the same type exists before the shared region,
        // search within the shared region for an earlier layer of the same type
        // that has a donor (chain to its donor). But the simplest correct approach:
        // search backwards from layer_idx - 1, skipping shared layers.
        None
    }

    /// Get the MoE intermediate size for a specific layer.
    ///
    /// KV-shared layers use double-wide MoE FFN (intermediate_size * 2).
    pub fn moe_intermediate_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_kv_shared_layer(layer_idx) {
            self.moe_intermediate_size_shared
        } else {
            self.moe_intermediate_size
        }
    }
}

/// Pre-allocated GPU buffer pool for MoE expert FFN computation.
///
/// Since experts execute sequentially within a command encoder (Metal guarantees
/// ordering), the same buffers can be reused for each expert. This eliminates
/// ~1200 buffer allocations per forward pass (5 buffers x 8 experts x 30 layers).
pub struct MoeBufferPool {
    /// gate_proj output: [moe_intermediate_size] f32
    gate_out: Option<MlxBuffer>,
    /// up_proj output: [moe_intermediate_size] f32
    up_out: Option<MlxBuffer>,
    /// GELU activation output: [moe_intermediate_size] f32
    gelu_out: Option<MlxBuffer>,
    /// hidden = gelu_out * up_out: [moe_intermediate_size] f32
    hidden_buf: Option<MlxBuffer>,
    /// down_proj output: [hidden_size] f32
    down_out: Option<MlxBuffer>,
    /// Accumulator per token: [hidden_size] f32
    accum_buf: Option<MlxBuffer>,
    /// Final bf16 output: [hidden_size] bf16
    bf16_output: Option<MlxBuffer>,
    /// Cached sizes so we know if reallocation is needed.
    intermediate_size: usize,
    hidden_size: usize,
}

impl MoeBufferPool {
    /// Create an empty pool. Buffers are allocated lazily on first use.
    fn new() -> Self {
        Self {
            gate_out: None,
            up_out: None,
            gelu_out: None,
            hidden_buf: None,
            down_out: None,
            accum_buf: None,
            bf16_output: None,
            intermediate_size: 0,
            hidden_size: 0,
        }
    }

    /// Ensure all buffers are allocated to the required sizes. Returns true if
    /// buffers were already valid and did not need reallocation.
    fn ensure_allocated(
        &mut self,
        device: &MlxDevice,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> Result<(), Gemma4Error> {
        if self.intermediate_size == intermediate_size
            && self.hidden_size == hidden_size
            && self.gate_out.is_some()
        {
            return Ok(());
        }

        self.intermediate_size = intermediate_size;
        self.hidden_size = hidden_size;

        let f32_inter_bytes = intermediate_size * std::mem::size_of::<f32>();
        let f32_hidden_bytes = hidden_size * std::mem::size_of::<f32>();

        self.gate_out = Some(device.alloc_buffer(
            f32_inter_bytes, DType::F32, vec![1, intermediate_size],
        )?);
        self.up_out = Some(device.alloc_buffer(
            f32_inter_bytes, DType::F32, vec![1, intermediate_size],
        )?);
        self.gelu_out = Some(device.alloc_buffer(
            f32_inter_bytes, DType::F32, vec![1, intermediate_size],
        )?);
        self.hidden_buf = Some(device.alloc_buffer(
            f32_inter_bytes, DType::F32, vec![1, intermediate_size],
        )?);
        self.down_out = Some(device.alloc_buffer(
            f32_hidden_bytes, DType::F32, vec![1, hidden_size],
        )?);
        self.accum_buf = Some(device.alloc_buffer(
            f32_hidden_bytes, DType::F32, vec![1, hidden_size],
        )?);
        self.bf16_output = Some(device.alloc_buffer(
            hidden_size * 2, DType::BF16, vec![1, hidden_size],
        )?);

        Ok(())
    }

    /// Get a reference to the gate_out buffer (panics if not allocated).
    #[inline]
    fn gate_out(&self) -> &MlxBuffer { self.gate_out.as_ref().unwrap() }
    /// Get a reference to the up_out buffer.
    #[inline]
    fn up_out(&self) -> &MlxBuffer { self.up_out.as_ref().unwrap() }
    /// Get a reference to the gelu_out buffer.
    #[inline]
    fn gelu_out(&self) -> &MlxBuffer { self.gelu_out.as_ref().unwrap() }
    /// Get a reference to the hidden_buf buffer.
    #[inline]
    fn hidden_buf(&self) -> &MlxBuffer { self.hidden_buf.as_ref().unwrap() }
}

/// Gemma 4 model holding weights, KV cache, and configuration.
/// Handle to an in-flight GPU forward pass.
///
/// Returned by [`Gemma4Model::forward_start`].  The GPU is executing
/// asynchronously; call [`Gemma4Model::forward_wait`] to block until
/// completion and read back the logits.
///
/// The handle owns the committed `CommandEncoder` (which in turn owns the
/// Metal command buffer) and the GPU output buffer, preventing use-after-free.
pub struct ForwardHandle {
    /// The Metal command encoder that was committed (not yet waited on).
    encoder: CommandEncoder,
    /// Pre-allocated output buffer for capped logits (f32, on GPU).
    logits_buffer: MlxBuffer,
    /// Vocab size for the last token's logits slice.
    vocab_size: usize,
    /// Sequence length used in this forward pass (needed to locate last-token logits).
    seq_len: usize,
}

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
    /// Cache: layer_idx -> dequantized + bf16-rounded router projection weights.
    /// Avoids re-dequantizing 360K elements (128 experts x 2816 hidden) on every
    /// moe_ffn call. ~43MB for 30 layers, but eliminates ~30x redundant work per step.
    router_proj_cache: HashMap<usize, Vec<f32>>,
    /// Cache: layer_idx -> pre-scaled router norm weight (scale * hidden_size^(-0.5)).
    router_norm_cache: HashMap<usize, Vec<f32>>,
    /// Cache: layer_idx -> per-expert scale factors.
    per_expert_scale_cache: HashMap<usize, Vec<f32>>,
    /// GPU buffer cache: layer_idx -> router projection weights as f32 GPU buffer.
    router_proj_gpu_cache: HashMap<usize, MlxBuffer>,
    /// GPU buffer cache: layer_idx -> router norm weight as f32 GPU buffer.
    router_norm_gpu_cache: HashMap<usize, MlxBuffer>,
    /// GPU buffer cache: layer_idx -> per-expert scale as f32 GPU buffer.
    per_expert_scale_gpu_cache: HashMap<usize, MlxBuffer>,
    /// GPU buffer cache: layer_idx -> (expert_ids, expert_weights) output buffers.
    moe_gate_output_cache: HashMap<usize, (MlxBuffer, MlxBuffer)>,
    /// Pre-allocated buffer pool for MoE expert FFN computation.
    /// Eliminates ~1200 GPU buffer allocations per forward pass.
    moe_buffer_pool: MoeBufferPool,
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

        // Build KV cache layout with donor mapping for KV-shared layers
        let mut cache_types = Vec::with_capacity(config.num_layers);
        let mut kv_configs = Vec::with_capacity(config.num_layers);
        let mut donor_layers = Vec::with_capacity(config.num_layers);

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
            donor_layers.push(config.donor_layer_for(i));
        }

        if config.num_kv_shared_layers > 0 {
            let shared_count = donor_layers.iter().filter(|d| d.is_some()).count();
            info!(
                num_kv_shared_layers = config.num_kv_shared_layers,
                actual_shared = shared_count,
                moe_intermediate_shared = config.moe_intermediate_size_shared,
                "KV sharing enabled"
            );
        }

        let kv_cache = KvCache::new_with_donors(
            &device, &cache_types, &kv_configs, &donor_layers,
        )?;

        info!(
            num_layers = config.num_layers,
            hidden_size = config.hidden_size,
            vocab_size = config.vocab_size,
            num_experts = config.num_experts,
            "Gemma4Model initialized"
        );

        let mut registry = KernelRegistry::new();
        gelu::register(&mut registry);
        kv_cache_copy::register(&mut registry);
        fused_head_norm_rope::register(&mut registry);
        fused_residual_norm::register(&mut registry);

        Ok(Self {
            weights,
            kv_cache,
            config,
            device,
            registry,
            router_proj_cache: HashMap::new(),
            router_norm_cache: HashMap::new(),
            per_expert_scale_cache: HashMap::new(),
            router_proj_gpu_cache: HashMap::new(),
            router_norm_gpu_cache: HashMap::new(),
            per_expert_scale_gpu_cache: HashMap::new(),
            moe_gate_output_cache: HashMap::new(),
            moe_buffer_pool: MoeBufferPool::new(),
        })
    }

    /// Run a forward pass on a sequence of token IDs.
    ///
    /// Returns a flat f32 buffer containing logits for the **last** sequence
    /// position only (shape `[vocab_size]`).  For prefill (`seq_len > 1`) the
    /// C1 optimisation extracts only the last token's hidden state before the
    /// lm_head projection, avoiding ~2 GB of wasted GPU memory.
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

        // Convert embedding output to bf16 for the bf16 pipeline (GPU cast, no CPU round-trip)
        let mut hidden = self.cast_f32_to_bf16(&hidden_f32, seq_len * hidden_size)?;

        // Step 2: Scale embeddings by sqrt(hidden_size) as Gemma convention
        // MLX stores this as bf16: embed_scale = bf16(sqrt(hidden_size))
        // Using bf16 scale matches MLX's precision exactly.
        // GPU scalar_mul_bf16 for the entire buffer.
        let scale = half::bf16::from_f32((hidden_size as f32).sqrt()).to_f32();
        let n_embed_elements = seq_len * hidden_size;
        let scaled_embed = self.device.alloc_buffer(
            n_embed_elements * 2, // bf16
            DType::BF16,
            vec![seq_len, hidden_size],
        )?;
        {
            let mut encoder = self.device.command_encoder()?;
            elementwise::scalar_mul_bf16(
                &mut encoder,
                &mut self.registry,
                self.device.metal_device(),
                &hidden,
                &scaled_embed,
                n_embed_elements,
                scale,
            )?;
            encoder.commit_and_wait()?;
        }
        hidden = scaled_embed;

        if diag {
            let s = read_buffer_f32(&hidden)?;
            let n = s.len().min(5);
            eprintln!("[DIAG] After embedding scale bf16 (factor={:.4}), first {}: {:?}",
                scale, n, &s[..n]);
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

        // C1 optimisation: for prefill (seq_len > 1), extract only the last
        // token's hidden state before the expensive lm_head projection.  This
        // avoids computing vocab_size logits for every position when only the
        // last position is used for sampling — saving ~2 GB for a 2048-token
        // prefill.  For decode (seq_len == 1) the buffer already contains a
        // single token so no extraction is needed.
        let (lm_input, lm_seq_len) = if seq_len > 1 {
            debug!("Extracting last token hidden state for lm_head (prefill optimisation)");
            let mut last_token_buf = self.device.alloc_buffer(
                hidden_size * 2, // bf16
                DType::BF16,
                vec![1, hidden_size],
            )?;
            {
                let src: &[u16] = normed.as_slice().map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("Read normed bf16 for last-token extraction: {e}"),
                })?;
                let dst: &mut [u16] = last_token_buf.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("Write last-token bf16 buffer: {e}"),
                })?;
                let last_start = (seq_len - 1) * hidden_size;
                dst.copy_from_slice(&src[last_start..last_start + hidden_size]);
            }
            (last_token_buf, 1usize)
        } else {
            (normed, seq_len)
        };

        // Step 5: lm_head (tied embeddings -- transpose embed_tokens and matmul)
        // lm_head takes bf16 input and returns f32 logits.
        // After C1, lm_seq_len is always 1 — we project only the last token.
        debug!("lm_head projection (tied embeddings)");
        let logits = self.lm_head_projection(&lm_input, lm_seq_len)?;

        if diag {
            let s: &[f32] = logits.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read logits: {e}"),
            })?;
            // After C1, logits are always [1, vocab_size] — last token only
            let vocab_size = self.config.vocab_size;
            let last_logits = &s[..vocab_size];
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

        // Step 6: Softcap on GPU — keeps logits on-device, single readback at
        // the end.  The kernel now includes bounds checking and uses explicit
        // threadgroup dispatch to avoid the previous out-of-bounds bug with
        // non-uniform grid tails.
        debug!(cap = self.config.final_logit_softcapping, "Softcap (GPU)");
        let capped_logits = self.apply_softcap(&logits, lm_seq_len)?;

        if diag {
            let s: &[f32] = capped_logits.as_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("diag read capped logits: {e}"),
            })?;
            let vocab_size = self.config.vocab_size;
            let last_logits = &s[..vocab_size];
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

        // Read logits to CPU — after C1 this is always [vocab_size] f32s
        let output: &[f32] = capped_logits
            .as_slice()
            .map_err(|e| Gemma4Error::ForwardError {
                reason: format!("Read logits: {e}"),
            })?;

        Ok(output.to_vec())
    }

    /// Run a greedy forward pass that returns just the argmax token ID and its
    /// logit value, avoiding the full vocab-sized logits readback to CPU.
    ///
    /// For greedy decoding (temperature ~0), this saves reading ~1MB of logits
    /// per step. Instead, argmax runs on GPU and only 8 bytes (u32 index + f32
    /// value) are read back.
    ///
    /// Falls back to full `forward()` + CPU argmax when the GPU argmax kernel
    /// is not available.
    pub fn forward_greedy(&mut self, tokens: &[u32]) -> Result<(u32, f32), Gemma4Error> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(Gemma4Error::ForwardError {
                reason: "Empty token sequence".into(),
            });
        }

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        // Steps 1-3: Same as forward() — embedding, scale, transformer layers
        let hidden_f32 = self.embedding_gather(tokens)?;
        let mut hidden = self.cast_f32_to_bf16(&hidden_f32, seq_len * hidden_size)?;

        let scale = half::bf16::from_f32((hidden_size as f32).sqrt()).to_f32();
        let n_embed_elements = seq_len * hidden_size;
        let scaled_embed = self.device.alloc_buffer(
            n_embed_elements * 2, DType::BF16, vec![seq_len, hidden_size],
        )?;
        {
            let mut encoder = self.device.command_encoder()?;
            elementwise::scalar_mul_bf16(
                &mut encoder, &mut self.registry, self.device.metal_device(),
                &hidden, &scaled_embed, n_embed_elements, scale,
            )?;
            encoder.commit_and_wait()?;
        }
        hidden = scaled_embed;

        for layer_idx in 0..self.config.num_layers {
            hidden = self.forward_layer(layer_idx, &hidden, seq_len)?;
        }

        // Step 4: Final RMS norm
        let normed = self.apply_rms_norm(&hidden, "model.norm.weight", seq_len)?;

        // Step 5: lm_head projection
        let logits = self.lm_head_projection(&normed, seq_len)?;

        // Step 6: Softcap
        let capped_logits = self.apply_softcap(&logits, seq_len)?;

        // Step 7: GPU argmax on last position's logits
        // TODO(kernel-agent): Wire dispatch_argmax_f32 when available.
        // For now, read only the last position's logits (vocab_size floats)
        // instead of all seq_len * vocab_size.
        let use_gpu_argmax = std::env::var("HF2Q_GPU_ARGMAX").is_ok();
        if use_gpu_argmax {
            // TODO: When mlx_native::ops::argmax::dispatch_argmax_f32 lands:
            //
            // let last_logits_offset = (seq_len - 1) * vocab_size;
            // let logits_slice = capped_logits.sub_buffer(
            //     last_logits_offset * std::mem::size_of::<f32>(),
            //     vocab_size * std::mem::size_of::<f32>(),
            // )?;
            // let out_index = self.device.alloc_buffer(4, DType::U32, vec![1])?;
            // let out_value = self.device.alloc_buffer(4, DType::F32, vec![1])?;
            // let mut encoder = self.device.command_encoder()?;
            // mlx_native::ops::argmax::dispatch_argmax_f32(
            //     &mut encoder, &logits_slice, &out_index, &out_value,
            //     vocab_size as u32, &mut self.registry,
            // )?;
            // encoder.commit_and_wait()?;
            // let idx: &[u32] = out_index.as_slice().map_err(|e| Gemma4Error::ForwardError {
            //     reason: format!("read argmax index: {e}"),
            // })?;
            // let val: &[f32] = out_value.as_slice().map_err(|e| Gemma4Error::ForwardError {
            //     reason: format!("read argmax value: {e}"),
            // })?;
            // return Ok((idx[0], val[0]));
        }

        // Fallback: read last position logits to CPU and argmax there
        let all_logits: &[f32] = capped_logits.as_slice().map_err(|e| Gemma4Error::ForwardError {
            reason: format!("Read logits for greedy: {e}"),
        })?;
        let last_start = (seq_len - 1) * vocab_size;
        let last_logits = &all_logits[last_start..last_start + vocab_size];

        let (best_idx, best_val) = last_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i as u32, v))
            .unwrap_or((0, 0.0));

        Ok((best_idx, best_val))
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

        // Step 2: Scale embeddings by sqrt(hidden_size) as Gemma convention (bf16)
        let scale = half::bf16::from_f32((hidden_size as f32).sqrt()).to_f32();
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
        let diag5 = std::env::var("HF2Q_DIAG5").is_ok() && layer_idx == 5;
        let timing = std::env::var("HF2Q_TIMING").is_ok() && layer_idx == 0;
        let t_layer_start = if timing { Some(std::time::Instant::now()) } else { None };

        // KV sharing: shared layers skip K/V projection and read from donor cache
        let is_kv_shared = self.config.is_kv_shared_layer(layer_idx);

        // =================================================================
        // BATCH 1: Pre-attention norm + Q/K/V projections
        //
        // All GPU dispatches (rms_norm, casts, qmatmuls) are encoded into a
        // single command buffer. One commit_and_wait replaces ~10 individual
        // commits that the unbatched path would have used.
        // =================================================================
        let q_out_dim = n_heads * head_dim;
        let kv_out_dim = n_kv_heads * head_dim;
        let eps = self.config.rms_norm_eps;

        let (normed, q_proj, k_proj, v_proj);

        // K-eq-V applies to GLOBAL (full_attention) layers only.
        let use_k_eq_v_check = self.config.attention_k_eq_v
            && layer_type == LayerAttentionType::Global;

        // Check which projections can use GPU qmatmul. If any projection would
        // take the CPU fallback path (which commits the encoder internally),
        // we cannot batch it into a shared encoder.
        let q_base = format!("model.layers.{layer_idx}.self_attn.q_proj");
        let k_base = format!("model.layers.{layer_idx}.self_attn.k_proj");
        let v_base = if use_k_eq_v_check {
            k_base.clone()
        } else {
            format!("model.layers.{layer_idx}.self_attn.v_proj")
        };
        let q_gpu = weight_uses_gpu_qmatmul(&self.weights, &q_base);
        let k_gpu = weight_uses_gpu_qmatmul(&self.weights, &k_base);
        let v_gpu = weight_uses_gpu_qmatmul(&self.weights, &v_base);
        let all_gpu = q_gpu && k_gpu && v_gpu;

        // =================================================================
        // MERGED BATCH 1+1b: Pre-attention norm + Q/K/V projections +
        // QK/V per-head norms + RoPE — ALL in ONE command encoder when
        // all projections use GPU. Replaces ~2 commit_and_wait() with 1.
        // =================================================================
        let q_norm_name = format!("model.layers.{layer_idx}.self_attn.q_norm.weight");
        let k_norm_name = format!("model.layers.{layer_idx}.self_attn.k_norm.weight");
        let rope_dim = self.config.rope_dim(layer_idx);
        let theta = self.config.rope_theta(layer_idx);

        let (q_roped, k_roped, v_normed_buf);

        // Check if O-proj also uses GPU (needed for Mega-encoder A).
        let o_base = format!("model.layers.{layer_idx}.self_attn.o_proj");
        let o_gpu = weight_uses_gpu_qmatmul(&self.weights, &o_base);
        let mega_a = all_gpu && o_gpu;

        // Declare attn/O-proj/residual outputs here so they are visible after
        // the mega-encoder block.
        let (attn_out, o_proj, residual1);

        if mega_a {
            // =============================================================
            // MEGA-ENCODER A: norm + QKV proj + head norms + RoPE +
            // KV cache append + SDPA + permutes + O proj + post-attn norm +
            // residual add — ALL in ONE command encoder (~25 dispatches).
            // Reduces ~4 commit_and_wait() to 1.
            // =============================================================
            let mut encoder = self.device.command_encoder()?;
            let norm_name = format!(
                "model.layers.{layer_idx}.input_layernorm.weight"
            );
            normed = self.apply_rms_norm_with_encoder(&mut encoder, input, &norm_name, seq_len)?;
            // Q projection always happens (shared layers still compute their own Q)
            q_proj = self.quantized_projection_with_encoder(
                &mut encoder, &normed, &q_base, seq_len, hidden_size, q_out_dim,
            )?;

            if is_kv_shared {
                // KV-shared layer: skip K/V projection entirely.
                // Placeholder buffers to satisfy the let binding.
                k_proj = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
                v_proj = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
            } else {
                k_proj = self.quantized_projection_with_encoder(
                    &mut encoder, &normed, &k_base, seq_len, hidden_size, kv_out_dim,
                )?;
                v_proj = self.quantized_projection_with_encoder(
                    &mut encoder, &normed, &v_base, seq_len, hidden_size, kv_out_dim,
                )?;
            }

            // Per-head Q norm (always needed) + K/V norms (skipped for shared)
            let mut norm_params = self.device.alloc_buffer(
                std::mem::size_of::<[f32; 2]>(), DType::F32, vec![2],
            )?;
            {
                let s: &mut [f32] = norm_params.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("per-head norm params: {e}"),
                })?;
                s[0] = eps;
                s[1] = head_dim as f32;
            }

            let q_nw = require_weight(&self.weights, &q_norm_name)?;
            let q_nw_bf16 = match q_nw.buffer.dtype() {
                DType::BF16 => clone_buffer(&self.device, &q_nw.buffer)?,
                _ => {
                    let f32_data = read_weight_as_f32(q_nw)?;
                    f32_vec_to_bf16_buffer(&self.device, &f32_data, vec![head_dim])?
                }
            };

            let q_normed = self.device.alloc_buffer(
                seq_len * q_out_dim * 2, DType::BF16, vec![seq_len, q_out_dim],
            )?;
            rms_norm::dispatch_rms_norm(
                &mut encoder, &mut self.registry, self.device.metal_device(),
                &q_proj, &q_nw_bf16, &q_normed, &norm_params,
                (seq_len * n_heads) as u32, head_dim as u32,
            )?;

            // K norm + V norm: only for non-shared layers
            #[allow(unused_assignments)]
            let mut k_normed_tmp = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
            if is_kv_shared {
                // KV-shared: skip K/V norms. K/V are in donor cache already.
                v_normed_buf = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
            } else {
                let k_nw = require_weight(&self.weights, &k_norm_name)?;
                let k_nw_bf16 = match k_nw.buffer.dtype() {
                    DType::BF16 => clone_buffer(&self.device, &k_nw.buffer)?,
                    _ => {
                        let f32_data = read_weight_as_f32(k_nw)?;
                        f32_vec_to_bf16_buffer(&self.device, &f32_data, vec![head_dim])?
                    }
                };

                k_normed_tmp = self.device.alloc_buffer(
                    seq_len * kv_out_dim * 2, DType::BF16, vec![seq_len, kv_out_dim],
                )?;
                rms_norm::dispatch_rms_norm(
                    &mut encoder, &mut self.registry, self.device.metal_device(),
                    &k_proj, &k_nw_bf16, &k_normed_tmp, &norm_params,
                    (seq_len * n_kv_heads) as u32, head_dim as u32,
                )?;

                let v_out = self.device.alloc_buffer(
                    seq_len * kv_out_dim * 2, DType::BF16, vec![seq_len, kv_out_dim],
                )?;
                rms_norm::dispatch_rms_norm_no_scale_bf16(
                    &mut encoder, &mut self.registry, self.device.metal_device(),
                    &v_proj, &v_out, &norm_params,
                    (seq_len * n_kv_heads) as u32, head_dim as u32,
                )?;
                v_normed_buf = v_out;
            }

            // RoPE: Q always, K only for non-shared layers
            if rope_dim > 0 && seq_len > 0 {
                let pos_offset = if self.kv_cache.num_layers() > 0 {
                    let last_layer = self.kv_cache.num_layers() - 1;
                    match self.kv_cache.layer(last_layer) {
                        Ok(cache) => cache.write_position(),
                        Err(_) => 0,
                    }
                } else {
                    0
                };

                let positions_vec: Vec<u32> = (0..seq_len).map(|i| (pos_offset + i) as u32).collect();
                let mut positions_buf = self.device.alloc_buffer(
                    seq_len * std::mem::size_of::<u32>(), DType::U32, vec![seq_len],
                )?;
                {
                    let s: &mut [u32] = positions_buf.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("RoPE positions: {e}"),
                    })?;
                    s.copy_from_slice(&positions_vec);
                }

                let mut rope_params = self.device.alloc_buffer(
                    4 * std::mem::size_of::<f32>(), DType::F32, vec![4],
                )?;
                {
                    let s: &mut [f32] = rope_params.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("RoPE params: {e}"),
                    })?;
                    s[0] = theta;
                    s[1] = head_dim as f32;
                    s[2] = rope_dim as f32;
                    s[3] = 0.0;
                }

                let q_rope_out = self.device.alloc_buffer(
                    seq_len * q_out_dim * 2, DType::BF16, vec![seq_len, q_out_dim],
                )?;
                rope::dispatch_rope_neox_bf16(
                    &mut encoder, &mut self.registry, self.device.metal_device(),
                    &q_normed, &q_rope_out, &rope_params, &positions_buf,
                    seq_len as u32, n_heads as u32, head_dim as u32, rope_dim as u32,
                )?;
                q_roped = q_rope_out;

                if is_kv_shared {
                    // K already in donor cache with RoPE applied
                    k_roped = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
                } else {
                    let k_rope_out = self.device.alloc_buffer(
                        seq_len * kv_out_dim * 2, DType::BF16, vec![seq_len, kv_out_dim],
                    )?;
                    rope::dispatch_rope_neox_bf16(
                        &mut encoder, &mut self.registry, self.device.metal_device(),
                        &k_normed_tmp, &k_rope_out, &rope_params, &positions_buf,
                        seq_len as u32, n_kv_heads as u32, head_dim as u32, rope_dim as u32,
                    )?;
                    k_roped = k_rope_out;
                }
            } else {
                q_roped = q_normed;
                if is_kv_shared {
                    k_roped = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
                } else {
                    k_roped = k_normed_tmp;
                }
            }

            // --- KV cache append (same encoder, no commit) ---
            // KV-shared layers skip: they read from donor layer's cache.
            if !is_kv_shared {
                let layer_cache = self.kv_cache.layer_mut(layer_idx)?;
                layer_cache.append_gpu(
                    &k_roped,
                    &v_normed_buf,
                    &mut encoder,
                    &mut self.registry,
                    self.device.metal_device(),
                )?;
            }

            // --- Attention: permutes + SDPA + output permute (same encoder) ---
            attn_out = self.compute_attention_with_encoder(
                &mut encoder,
                layer_idx,
                &q_roped,
                seq_len,
                n_heads,
                n_kv_heads,
                head_dim,
                layer_type,
            )?;

            // --- O projection + post-attn fused(norm + residual) (same encoder) ---
            o_proj = self.quantized_projection_with_encoder(
                &mut encoder, &attn_out, &o_base,
                seq_len, n_heads * head_dim, hidden_size,
            )?;
            let post_attn_norm_name = format!(
                "model.layers.{layer_idx}.post_attention_layernorm.weight"
            );
            residual1 = self.fused_norm_add_with_encoder(
                &mut encoder, input, &o_proj, &post_attn_norm_name, seq_len,
            )?;

            // =============================================================
            // DECODE FAST PATH (seq_len == 1): Merge attention + FFN into
            // minimal commits.  MoE layers: 2 commits.  Non-MoE: 1 commit.
            // Falls through to existing multi-commit path for prefill.
            // =============================================================
            if seq_len == 1 {
                let router_weight_name_fp = format!("model.layers.{layer_idx}.router.proj.weight");
                let has_moe_fp = self.weights.get(&router_weight_name_fp).is_some()
                    || self.weights.get(&format!("language_model.{}", router_weight_name_fp)).is_some();

                if has_moe_fp {
                    // --- MoE decode: 2 commits total ---
                    // Pre-FFN norms (same MEGA-A encoder)
                    let pre_ff_norm_name = format!(
                        "model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
                    );
                    let pre_ff_normed = self.apply_rms_norm_with_encoder(
                        &mut encoder, &residual1, &pre_ff_norm_name, seq_len,
                    )?;
                    let pre_ff_norm2_name = format!(
                        "model.layers.{layer_idx}.pre_feedforward_layernorm_2.weight"
                    );
                    let pre_ff_normed2 = self.apply_rms_norm_with_encoder(
                        &mut encoder, &residual1, &pre_ff_norm2_name, seq_len,
                    )?;

                    // MoE gate routing (same encoder — GPU dispatches only)
                    let use_cpu_route = std::env::var("HF2Q_CPU_MOE_ROUTE").is_ok();
                    let (route_ids_buf, route_weights_buf) = if !use_cpu_route {
                        let (ids, wts) = self.gpu_moe_route_encode(
                            &mut encoder, layer_idx, &residual1, seq_len,
                        )?;
                        (Some(ids), Some(wts))
                    } else {
                        (None, None)
                    };

                    // COMMIT 1: MEGA-A + pre-FFN norms + MoE gate
                    let t_c1_pre = if timing { Some(std::time::Instant::now()) } else { None };
                    encoder.commit_and_wait()?;
                    if let (Some(start), Some(pre)) = (t_layer_start, t_c1_pre) {
                        let encode_ms = pre.duration_since(start).as_secs_f64() * 1000.0;
                        let wait_ms = pre.elapsed().as_secs_f64() * 1000.0;
                        eprintln!("[TIMING] L0 decode-fast commit1 (attn+norms+gate): encode={:.2}ms, wait={:.2}ms",
                            encode_ms, wait_ms);
                    }

                    // Read back routing results (64 bytes for top-8)
                    let (all_expert_ids, all_routing_weights) = if let (Some(ref ids), Some(ref wts)) =
                        (route_ids_buf, route_weights_buf)
                    {
                        self.gpu_moe_route_readback(ids, wts, seq_len)?
                    } else {
                        self.cpu_moe_route(layer_idx, &residual1, seq_len)?
                    };

                    // COMMIT 2: Dense FFN + expert dispatch + post-FFN norms + combine + residual + scalar
                    let t_c2_pre = if timing { Some(std::time::Instant::now()) } else { None };
                    let mut enc2 = self.device.command_encoder()?;

                    // Dense FFN
                    let dense_result = self.dense_ffn_with_encoder(
                        &mut enc2, layer_idx, &pre_ff_normed, seq_len,
                    )?;
                    let mlp_out = match dense_result {
                        Some(buf) => buf,
                        None => {
                            // CPU fallback: dense_ffn with its own commits
                            drop(enc2);
                            let mlp = self.dense_ffn(layer_idx, &pre_ff_normed, seq_len)?;
                            enc2 = self.device.command_encoder()?;
                            mlp
                        }
                    };

                    // Expert dispatch (inlined from moe_ffn_gpu_with_encoder)
                    let intermediate_size_fp = self.config.moe_intermediate_for_layer(layer_idx);
                    let gate_name_fp = format!("model.layers.{layer_idx}.experts.switch_glu.gate_proj.weight");
                    let gate_w_fp = require_weight(&self.weights, &gate_name_fp)?;
                    let qm_fp = gate_w_fp.quant_meta.as_ref().ok_or_else(|| Gemma4Error::ForwardError {
                        reason: "decode fast path: expert weights lack quant_meta".into(),
                    })?;
                    let bits_fp = qm_fp.bits as u32;
                    let group_size_fp = qm_fp.group_size as u32;

                    let mut gpu_output_bufs: Vec<(f32, MlxBuffer)> = Vec::with_capacity(
                        self.config.top_k_experts,
                    );
                    for (rank, &expert_id) in all_expert_ids[0].iter().enumerate() {
                        let w = all_routing_weights[0][rank];
                        if w.abs() < 1e-10 { continue; }
                        let out_buf = self.expert_ffn_3d_gpu_buf_with_encoder(
                            &mut enc2, layer_idx, expert_id, &pre_ff_normed2,
                            0, hidden_size, intermediate_size_fp, bits_fp, group_size_fp,
                        )?;
                        gpu_output_bufs.push((w, out_buf));
                    }

                    // MoE accumulation (same encoder)
                    let accum_buf = self.device.alloc_buffer(
                        hidden_size * std::mem::size_of::<f32>(),
                        DType::F32, vec![1, hidden_size],
                    )?;
                    moe_ops::moe_zero_buffer_encode(
                        &mut enc2, &mut self.registry,
                        self.device.metal_device(),
                        &accum_buf, hidden_size,
                    )?;
                    for (weight, buf) in &gpu_output_bufs {
                        moe_ops::moe_accumulate_encode(
                            &mut enc2, &mut self.registry,
                            self.device.metal_device(),
                            &accum_buf, buf, *weight, hidden_size,
                        )?;
                    }
                    let moe_out = self.device.alloc_buffer(
                        hidden_size * 2, DType::BF16, vec![1, hidden_size],
                    )?;
                    elementwise::cast(
                        &mut enc2, &mut self.registry,
                        self.device.metal_device(),
                        &accum_buf, &moe_out, hidden_size,
                        elementwise::CastDirection::F32ToBF16,
                    )?;

                    // Post-FFN norms + combine + residual + scalar (MEGA-D, same encoder)
                    let post_ff_norm1_name = format!(
                        "model.layers.{layer_idx}.post_feedforward_layernorm_1.weight"
                    );
                    let mlp_normed = self.apply_rms_norm_with_encoder(
                        &mut enc2, &mlp_out, &post_ff_norm1_name, seq_len,
                    )?;
                    let post_ff_norm2_name = format!(
                        "model.layers.{layer_idx}.post_feedforward_layernorm_2.weight"
                    );
                    let moe_normed = self.apply_rms_norm_with_encoder(
                        &mut enc2, &moe_out, &post_ff_norm2_name, seq_len,
                    )?;
                    let combined = self.elementwise_add_with_encoder(
                        &mut enc2, &mlp_normed, &moe_normed, hidden_size,
                    )?;
                    let post_ff_norm_name = format!(
                        "model.layers.{layer_idx}.post_feedforward_layernorm.weight"
                    );
                    let pre_scalar = self.fused_norm_add_with_encoder(
                        &mut enc2, &residual1, &combined, &post_ff_norm_name, seq_len,
                    )?;

                    let scalar_name_fp = format!("model.layers.{layer_idx}.layer_scalar");
                    let mega_output = match require_weight(&self.weights, &scalar_name_fp) {
                        Ok(w) => {
                            let sv = if w.buffer.byte_len() >= 4 && w.buffer.dtype() == DType::F32 {
                                let s: &[f32] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                                    reason: format!("read layer_scalar: {e}"),
                                })?;
                                s[0]
                            } else {
                                let raw: &[u16] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                                    reason: format!("read layer_scalar bf16: {e}"),
                                })?;
                                half::bf16::from_bits(raw[0]).to_f32()
                            };
                            let scaled = self.device.alloc_buffer(
                                hidden_size * 2, DType::BF16, vec![hidden_size],
                            )?;
                            elementwise::scalar_mul_bf16(
                                &mut enc2, &mut self.registry, self.device.metal_device(),
                                &pre_scalar, &scaled, hidden_size, sv,
                            )?;
                            scaled
                        }
                        Err(_) => pre_scalar,
                    };

                    enc2.commit_and_wait()?;
                    if let (Some(start), Some(pre)) = (t_layer_start, t_c2_pre) {
                        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
                        let c2_ms = pre.elapsed().as_secs_f64() * 1000.0;
                        eprintln!("[TIMING] L0 decode-fast commit2 (experts+postFFN): {:.2}ms, total: {:.2}ms",
                            c2_ms, total_ms);
                    }

                    return Ok(mega_output);

                } else {
                    // --- Non-MoE decode: 1 commit total ---
                    let pre_ff_norm_name = format!(
                        "model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
                    );
                    let pre_ff_normed = self.apply_rms_norm_with_encoder(
                        &mut encoder, &residual1, &pre_ff_norm_name, seq_len,
                    )?;

                    let dense_result = self.dense_ffn_with_encoder(
                        &mut encoder, layer_idx, &pre_ff_normed, seq_len,
                    )?;
                    if let Some(mlp_out) = dense_result {
                        let post_ff_norm_name = format!(
                            "model.layers.{layer_idx}.post_feedforward_layernorm.weight"
                        );
                        let pre_scalar = self.fused_norm_add_with_encoder(
                            &mut encoder, &residual1, &mlp_out, &post_ff_norm_name, seq_len,
                        )?;

                        let scalar_name_fp = format!("model.layers.{layer_idx}.layer_scalar");
                        let layer_output = match require_weight(&self.weights, &scalar_name_fp) {
                            Ok(w) => {
                                let sv = if w.buffer.byte_len() >= 4 && w.buffer.dtype() == DType::F32 {
                                    let s: &[f32] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                                        reason: format!("read layer_scalar: {e}"),
                                    })?;
                                    s[0]
                                } else {
                                    let raw: &[u16] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                                        reason: format!("read layer_scalar bf16: {e}"),
                                    })?;
                                    half::bf16::from_bits(raw[0]).to_f32()
                                };
                                let scaled = self.device.alloc_buffer(
                                    hidden_size * 2, DType::BF16, vec![hidden_size],
                                )?;
                                elementwise::scalar_mul_bf16(
                                    &mut encoder, &mut self.registry, self.device.metal_device(),
                                    &pre_scalar, &scaled, hidden_size, sv,
                                )?;
                                scaled
                            }
                            Err(_) => pre_scalar,
                        };

                        // ONE commit for the entire non-MoE layer
                        let t_nm_pre = if timing { Some(std::time::Instant::now()) } else { None };
                        encoder.commit_and_wait()?;
                        if let (Some(start), Some(pre)) = (t_layer_start, t_nm_pre) {
                            let total_ms = start.elapsed().as_secs_f64() * 1000.0;
                            let wait_ms = pre.elapsed().as_secs_f64() * 1000.0;
                            eprintln!("[TIMING] L0 decode-fast non-MoE: wait={:.2}ms, total={:.2}ms",
                                wait_ms, total_ms);
                        }

                        return Ok(layer_output);
                    }
                    // Fall through if dense_ffn needs CPU fallback
                }
            }

            // Original prefill path: commit MEGA-A and continue to multi-commit FFN below
            let t_attn_pre = if timing { Some(std::time::Instant::now()) } else { None };
            encoder.commit_and_wait()?;
            if let (Some(start), Some(pre)) = (t_layer_start, t_attn_pre) {
                let encode_ms = pre.duration_since(start).as_secs_f64() * 1000.0;
                let wait_ms = pre.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[TIMING] L0 mega-A: encode={:.2}ms, commit_wait={:.2}ms", encode_ms, wait_ms);
            }

            if diag {
                let k_raw = read_buffer_f32(&k_proj)?;
                let v_raw = read_buffer_f32(&v_proj)?;
                let use_k_eq_v = self.config.attention_k_eq_v
                    && layer_type == LayerAttentionType::Global;
                eprintln!("[DIAG] Layer 0 K raw (before k_norm), pos 0 first 5: {:?}", &k_raw[..5.min(k_raw.len())]);
                eprintln!("[DIAG] Layer 0 V raw (before v_norm), pos 0 first 5: {:?}", &v_raw[..5.min(v_raw.len())]);
                let k_v_match = k_raw.len() == v_raw.len() &&
                    k_raw.iter().zip(v_raw.iter()).take(10).all(|(a, b)| (a - b).abs() < 1e-6);
                eprintln!("[DIAG]   K==V? {} (k_eq_v={}, k_len={}, v_len={})", k_v_match, use_k_eq_v, k_raw.len(), v_raw.len());

                let q_rd = read_buffer_f32(&q_roped)?;
                let n = q_rd.len().min(5);
                eprintln!("[DIAG] Layer 0 after QK norms+RoPE, Q first {}: {:?}", n, &q_rd[..n]);
                let k_rd = read_buffer_f32(&k_roped)?;
                let n = k_rd.len().min(5);
                eprintln!("[DIAG] Layer 0 after QK norms+RoPE, K first {}: {:?}", n, &k_rd[..n]);

                let v_nd = read_buffer_f32(&v_normed_buf)?;
                eprintln!("[DIAG] Layer 0 V normed (after v_norm_no_scale), pos 0 first 5: {:?}", &v_nd[..5.min(v_nd.len())]);
                let v_head0 = &v_raw[..head_dim];
                let v_rms = (v_head0.iter().map(|v| v * v).sum::<f32>() / head_dim as f32).sqrt();
                eprintln!("[DIAG]   V head 0 pos 0: RMS={:.6}, len={}, k_eq_v={}", v_rms, head_dim, use_k_eq_v);

                let v_src_u16: &[u16] = v_normed_buf.as_slice().map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("diag read v_normed: {e}"),
                })?;
                let v_data_f32: Vec<f32> = v_src_u16.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
                eprintln!("[DIAG] Layer 0 V stored in cache (first 5 of pos 0): {:?}", &v_data_f32[..5.min(v_data_f32.len())]);
                eprintln!("[DIAG]   V total elements: {}, expected: {}x{}={}", v_data_f32.len(), seq_len, n_kv_heads * head_dim, seq_len * n_kv_heads * head_dim);

                let layer_cache_read = self.kv_cache.layer(layer_idx)?;
                let (_k_cb, v_cb, kv_slen) = layer_cache_read.keys_values();
                let v_cache_u16: &[u16] = v_cb.as_slice().map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("diag read v cache: {e}"),
                })?;
                let v_first5: Vec<f32> = v_cache_u16[..5.min(v_cache_u16.len())].iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
                eprintln!("[DIAG] Layer 0 V cache readback pos 0 first 5: {:?}, kv_seq_len={}", v_first5, kv_slen);

                let v_valid_u16 = &v_cache_u16[..kv_slen * n_kv_heads * head_dim];
                let v_valid_f32: Vec<f32> = v_valid_u16.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
                let v_reshaped = reshape_qkv_for_sdpa(&v_valid_f32, kv_slen, n_kv_heads, head_dim);
                eprintln!("[DIAG] Layer 0 V reshaped for SDPA (head 0, pos 0, first 5): {:?}", &v_reshaped[..5.min(v_reshaped.len())]);

                let q_ps = read_buffer_f32(&q_roped)?;
                eprintln!("[DIAG] Layer 0 Q for SDPA (pos 0, first 5): {:?}", &q_ps[..5.min(q_ps.len())]);

                let s = read_buffer_f32(&attn_out)?;
                let n = s.len().min(5);
                eprintln!("[DIAG] Layer 0 after attention output, first {}: {:?}", n, &s[..n]);
                let last_attn_start = (seq_len - 1) * n_heads * head_dim;
                let last_attn_5 = &s[last_attn_start..last_attn_start + 5.min(s.len() - last_attn_start)];
                eprintln!("[DIAG] Layer 0 attn output LAST token (head 0, first 5): {:?}", last_attn_5);
                let q_data_all = read_buffer_f32(&q_roped)?;
                let last_q_start = (seq_len - 1) * n_heads * head_dim;
                let last_q_5 = &q_data_all[last_q_start..last_q_start + 5.min(q_data_all.len() - last_q_start)];
                eprintln!("[DIAG] Layer 0 Q LAST token (head 0, first 5): {:?}", last_q_5);
                let nan_count = s.iter().filter(|v| v.is_nan()).count();
                let inf_count = s.iter().filter(|v| v.is_infinite()).count();
                if nan_count > 0 || inf_count > 0 {
                    eprintln!("[DIAG]   WARNING attn: NaN={}, Inf={}", nan_count, inf_count);
                }

                let o_data = read_buffer_f32(&o_proj)?;
                let last_o = (seq_len - 1) * hidden_size;
                eprintln!("[DIAG] Layer 0 O proj LAST token first 5: {:?}", &o_data[last_o..last_o + 5]);
                let r1_data = read_buffer_f32(&residual1)?;
                eprintln!("[DIAG] Layer 0 residual1 LAST token first 5: {:?}", &r1_data[last_o..last_o + 5]);
            }

            // Deep diagnostics for layer 5 (first Global)
            if diag5 {
                let hd = head_dim;
                let nk = n_kv_heads;
                let nh = n_heads;

                // Q after RoPE: shape [seq_len, n_heads * head_dim], read last token, head 0
                let q_all = read_buffer_f32(&q_roped)?;
                let last_q = (seq_len - 1) * nh * hd;
                let q_h0 = &q_all[last_q..last_q + 20.min(nh * hd)];
                eprintln!("[DIAG5] L5 Q after RoPE (head 0, last pos, first 20): {:?}", q_h0);

                // K after RoPE: shape [seq_len, n_kv_heads * head_dim], read last token, head 0
                let k_all = read_buffer_f32(&k_roped)?;
                let last_k = (seq_len - 1) * nk * hd;
                let k_h0 = &k_all[last_k..last_k + 20.min(nk * hd)];
                eprintln!("[DIAG5] L5 K after RoPE (head 0, last pos, first 20): {:?}", k_h0);

                // V normed: shape [seq_len, n_kv_heads * head_dim]
                let v_all = read_buffer_f32(&v_normed_buf)?;
                let last_v = (seq_len - 1) * nk * hd;
                let v_h0 = &v_all[last_v..last_v + 20.min(nk * hd)];
                eprintln!("[DIAG5] L5 V normed (head 0, last pos, first 20): {:?}", v_h0);

                // K before RoPE (after k_norm): to isolate RoPE vs norm diff
                // k_normed is the output of rms_norm on k_proj
                // But k_normed was consumed by RoPE... we only have k_roped.
                // Print attention output too
                let a_all = read_buffer_f32(&attn_out)?;
                let last_a = (seq_len - 1) * nh * hd;
                let a_h0 = &a_all[last_a..last_a + 5.min(nh * hd)];
                eprintln!("[DIAG5] L5 attn output (head 0, last pos, first 5): {:?}", a_h0);

                // O projection output
                let o_all = read_buffer_f32(&o_proj)?;
                let last_o_start = (seq_len - 1) * hidden_size;
                let o_5 = &o_all[last_o_start..last_o_start + 5];
                eprintln!("[DIAG5] L5 O proj (last pos, first 5): {:?}", o_5);

                // Residual after attention
                let r1_all = read_buffer_f32(&residual1)?;
                let r1_5 = &r1_all[last_o_start..last_o_start + 5];
                eprintln!("[DIAG5] L5 residual1 (last pos, first 5): {:?}", r1_5);
            }
        } else {
            // Slow path: at least one projection needs CPU fallback.
            {
                let mut encoder = self.device.command_encoder()?;
                let norm_name = format!(
                    "model.layers.{layer_idx}.input_layernorm.weight"
                );
                normed = self.apply_rms_norm_with_encoder(&mut encoder, input, &norm_name, seq_len)?;
                encoder.commit_and_wait()?;
            }
            {
                let mut encoder = self.device.command_encoder()?;
                q_proj = self.quantized_projection_with_encoder(
                    &mut encoder, &normed, &q_base, seq_len, hidden_size, q_out_dim,
                )?;
                if q_gpu { encoder.commit_and_wait()?; }
            }
            if is_kv_shared {
                // KV-shared layer: skip K/V projection
                k_proj = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
                v_proj = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
            } else {
                {
                    let mut encoder = self.device.command_encoder()?;
                    k_proj = self.quantized_projection_with_encoder(
                        &mut encoder, &normed, &k_base, seq_len, hidden_size, kv_out_dim,
                    )?;
                    if k_gpu { encoder.commit_and_wait()?; }
                }
                {
                    let mut encoder = self.device.command_encoder()?;
                    v_proj = self.quantized_projection_with_encoder(
                        &mut encoder, &normed, &v_base, seq_len, hidden_size, kv_out_dim,
                    )?;
                    if v_gpu { encoder.commit_and_wait()?; }
                }
            }

            // Per-head norms + RoPE in one encoder
            let mut encoder = self.device.command_encoder()?;
            let mut norm_params = self.device.alloc_buffer(
                std::mem::size_of::<[f32; 2]>(), DType::F32, vec![2],
            )?;
            {
                let s: &mut [f32] = norm_params.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("per-head norm params: {e}"),
                })?;
                s[0] = eps;
                s[1] = head_dim as f32;
            }

            let q_nw = require_weight(&self.weights, &q_norm_name)?;
            let q_nw_bf16 = match q_nw.buffer.dtype() {
                DType::BF16 => clone_buffer(&self.device, &q_nw.buffer)?,
                _ => {
                    let f32_data = read_weight_as_f32(q_nw)?;
                    f32_vec_to_bf16_buffer(&self.device, &f32_data, vec![head_dim])?
                }
            };

            let q_normed = self.device.alloc_buffer(
                seq_len * q_out_dim * 2, DType::BF16, vec![seq_len, q_out_dim],
            )?;
            rms_norm::dispatch_rms_norm(
                &mut encoder, &mut self.registry, self.device.metal_device(),
                &q_proj, &q_nw_bf16, &q_normed, &norm_params,
                (seq_len * n_heads) as u32, head_dim as u32,
            )?;

            // K norm + V norm: only for non-shared layers
            #[allow(unused_assignments)]
            let mut k_normed_slow = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
            if is_kv_shared {
                v_normed_buf = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
            } else {
                let k_nw = require_weight(&self.weights, &k_norm_name)?;
                let k_nw_bf16 = match k_nw.buffer.dtype() {
                    DType::BF16 => clone_buffer(&self.device, &k_nw.buffer)?,
                    _ => {
                        let f32_data = read_weight_as_f32(k_nw)?;
                        f32_vec_to_bf16_buffer(&self.device, &f32_data, vec![head_dim])?
                    }
                };

                k_normed_slow = self.device.alloc_buffer(
                    seq_len * kv_out_dim * 2, DType::BF16, vec![seq_len, kv_out_dim],
                )?;
                rms_norm::dispatch_rms_norm(
                    &mut encoder, &mut self.registry, self.device.metal_device(),
                    &k_proj, &k_nw_bf16, &k_normed_slow, &norm_params,
                    (seq_len * n_kv_heads) as u32, head_dim as u32,
                )?;

                let v_out = self.device.alloc_buffer(
                    seq_len * kv_out_dim * 2, DType::BF16, vec![seq_len, kv_out_dim],
                )?;
                rms_norm::dispatch_rms_norm_no_scale_bf16(
                    &mut encoder, &mut self.registry, self.device.metal_device(),
                    &v_proj, &v_out, &norm_params,
                    (seq_len * n_kv_heads) as u32, head_dim as u32,
                )?;
                v_normed_buf = v_out;
            }

            if rope_dim > 0 && seq_len > 0 {
                let pos_offset = if self.kv_cache.num_layers() > 0 {
                    let last_layer = self.kv_cache.num_layers() - 1;
                    match self.kv_cache.layer(last_layer) {
                        Ok(cache) => cache.write_position(),
                        Err(_) => 0,
                    }
                } else {
                    0
                };

                let positions_vec: Vec<u32> = (0..seq_len).map(|i| (pos_offset + i) as u32).collect();
                let mut positions_buf = self.device.alloc_buffer(
                    seq_len * std::mem::size_of::<u32>(), DType::U32, vec![seq_len],
                )?;
                {
                    let s: &mut [u32] = positions_buf.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("RoPE positions: {e}"),
                    })?;
                    s.copy_from_slice(&positions_vec);
                }

                let mut rope_params = self.device.alloc_buffer(
                    4 * std::mem::size_of::<f32>(), DType::F32, vec![4],
                )?;
                {
                    let s: &mut [f32] = rope_params.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("RoPE params: {e}"),
                    })?;
                    s[0] = theta;
                    s[1] = head_dim as f32;
                    s[2] = rope_dim as f32;
                    s[3] = 0.0;
                }

                let q_rope_out = self.device.alloc_buffer(
                    seq_len * q_out_dim * 2, DType::BF16, vec![seq_len, q_out_dim],
                )?;
                rope::dispatch_rope_neox_bf16(
                    &mut encoder, &mut self.registry, self.device.metal_device(),
                    &q_normed, &q_rope_out, &rope_params, &positions_buf,
                    seq_len as u32, n_heads as u32, head_dim as u32, rope_dim as u32,
                )?;
                q_roped = q_rope_out;

                if is_kv_shared {
                    k_roped = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
                } else {
                    let k_rope_out = self.device.alloc_buffer(
                        seq_len * kv_out_dim * 2, DType::BF16, vec![seq_len, kv_out_dim],
                    )?;
                    rope::dispatch_rope_neox_bf16(
                        &mut encoder, &mut self.registry, self.device.metal_device(),
                        &k_normed_slow, &k_rope_out, &rope_params, &positions_buf,
                        seq_len as u32, n_kv_heads as u32, head_dim as u32, rope_dim as u32,
                    )?;
                    k_roped = k_rope_out;
                }
            } else {
                q_roped = q_normed;
                if is_kv_shared {
                    k_roped = self.device.alloc_buffer(4, DType::BF16, vec![1, 1])?;
                } else {
                    k_roped = k_normed_slow;
                }
            }

            // =============================================================
            // DECODE FAST PATH (slow attn): Merge head norms + RoPE +
            // KV append + attention + O proj + post-attn norm into ONE
            // commit instead of 4 separate commits.
            // =============================================================
            if seq_len == 1 && o_gpu {
                // Continue using the head norms + RoPE encoder
                // KV cache append (same encoder)
                if !is_kv_shared {
                    let layer_cache = self.kv_cache.layer_mut(layer_idx)?;
                    layer_cache.append_gpu(
                        &k_roped,
                        &v_normed_buf,
                        &mut encoder,
                        &mut self.registry,
                        self.device.metal_device(),
                    )?;
                }

                // Attention (same encoder)
                attn_out = self.compute_attention_with_encoder(
                    &mut encoder,
                    layer_idx,
                    &q_roped,
                    seq_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    layer_type,
                )?;

                // O projection + post-attn norm (same encoder)
                o_proj = self.quantized_projection_with_encoder(
                    &mut encoder, &attn_out, &o_base,
                    seq_len, n_heads * head_dim, hidden_size,
                )?;
                let post_attn_norm_name = format!(
                    "model.layers.{layer_idx}.post_attention_layernorm.weight"
                );
                residual1 = self.fused_norm_add_with_encoder(
                    &mut encoder, input, &o_proj, &post_attn_norm_name, seq_len,
                )?;

                // ONE commit: head norms + RoPE + KV + attention + O + post-attn
                encoder.commit_and_wait()?;
            } else {
                // Prefill or non-GPU O-proj: original multi-commit path
                encoder.commit_and_wait()?;

                // Non-mega: KV append, attention, O-proj, residual as separate encoders
                let use_k_eq_v = self.config.attention_k_eq_v
                    && layer_type == LayerAttentionType::Global;

                if diag && !is_kv_shared {
                    let k_raw = read_buffer_f32(&k_proj)?;
                    let v_raw = read_buffer_f32(&v_proj)?;
                    eprintln!("[DIAG] Layer 0 K raw (before k_norm), pos 0 first 5: {:?}", &k_raw[..5.min(k_raw.len())]);
                    eprintln!("[DIAG] Layer 0 V raw (before v_norm), pos 0 first 5: {:?}", &v_raw[..5.min(v_raw.len())]);
                    let k_v_match = k_raw.len() == v_raw.len() &&
                        k_raw.iter().zip(v_raw.iter()).take(10).all(|(a, b)| (a - b).abs() < 1e-6);
                    eprintln!("[DIAG]   K==V? {} (k_eq_v={}, k_len={}, v_len={})", k_v_match, use_k_eq_v, k_raw.len(), v_raw.len());

                    let q_rd = read_buffer_f32(&q_roped)?;
                    let n = q_rd.len().min(5);
                    eprintln!("[DIAG] Layer 0 after QK norms+RoPE, Q first {}: {:?}", n, &q_rd[..n]);
                    let k_rd = read_buffer_f32(&k_roped)?;
                    let n = k_rd.len().min(5);
                    eprintln!("[DIAG] Layer 0 after QK norms+RoPE, K first {}: {:?}", n, &k_rd[..n]);

                    let v_nd = read_buffer_f32(&v_normed_buf)?;
                    eprintln!("[DIAG] Layer 0 V normed (after v_norm_no_scale), pos 0 first 5: {:?}", &v_nd[..5.min(v_nd.len())]);
                    let v_head0 = &v_raw[..head_dim];
                    let v_rms = (v_head0.iter().map(|v| v * v).sum::<f32>() / head_dim as f32).sqrt();
                    eprintln!("[DIAG]   V head 0 pos 0: RMS={:.6}, len={}, k_eq_v={}", v_rms, head_dim, use_k_eq_v);
                }

                // --- Step 4: KV cache append --- (GPU direct copy, no CPU round-trip)
                if !is_kv_shared {
                    let mut kv_encoder = self.device.command_encoder()?;
                    let layer_cache = self.kv_cache.layer_mut(layer_idx)?;
                    layer_cache.append_gpu(
                        &k_roped,
                        &v_normed_buf,
                        &mut kv_encoder,
                        &mut self.registry,
                        self.device.metal_device(),
                    )?;
                    kv_encoder.commit_and_wait()?;
                }

                // --- Step 5: Attention ---
                attn_out = self.compute_attention(
                    layer_idx,
                    &q_roped,
                    seq_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    layer_type,
                )?;

                if diag {
                    let v_src_u16: &[u16] = v_normed_buf.as_slice().map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("diag read v_normed: {e}"),
                    })?;
                    let v_data_f32: Vec<f32> = v_src_u16.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
                    eprintln!("[DIAG] Layer 0 V stored in cache (first 5 of pos 0): {:?}", &v_data_f32[..5.min(v_data_f32.len())]);
                    eprintln!("[DIAG]   V total elements: {}, expected: {}x{}={}", v_data_f32.len(), seq_len, n_kv_heads * head_dim, seq_len * n_kv_heads * head_dim);

                    let layer_cache_read = self.kv_cache.layer(layer_idx)?;
                    let (_k_cb, v_cb, kv_slen) = layer_cache_read.keys_values();
                    let v_cache_u16: &[u16] = v_cb.as_slice().map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("diag read v cache: {e}"),
                    })?;
                    let v_first5: Vec<f32> = v_cache_u16[..5.min(v_cache_u16.len())].iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
                    eprintln!("[DIAG] Layer 0 V cache readback pos 0 first 5: {:?}, kv_seq_len={}", v_first5, kv_slen);

                    let v_valid_u16 = &v_cache_u16[..kv_slen * n_kv_heads * head_dim];
                    let v_valid_f32: Vec<f32> = v_valid_u16.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
                    let v_reshaped = reshape_qkv_for_sdpa(&v_valid_f32, kv_slen, n_kv_heads, head_dim);
                    eprintln!("[DIAG] Layer 0 V reshaped for SDPA (head 0, pos 0, first 5): {:?}", &v_reshaped[..5.min(v_reshaped.len())]);

                    let q_ps = read_buffer_f32(&q_roped)?;
                    eprintln!("[DIAG] Layer 0 Q for SDPA (pos 0, first 5): {:?}", &q_ps[..5.min(q_ps.len())]);

                    let s = read_buffer_f32(&attn_out)?;
                    let n = s.len().min(5);
                    eprintln!("[DIAG] Layer 0 after attention output, first {}: {:?}", n, &s[..n]);
                    let last_attn_start = (seq_len - 1) * n_heads * head_dim;
                    let last_attn_5 = &s[last_attn_start..last_attn_start + 5.min(s.len() - last_attn_start)];
                    eprintln!("[DIAG] Layer 0 attn output LAST token (head 0, first 5): {:?}", last_attn_5);
                    let q_data_all = read_buffer_f32(&q_roped)?;
                    let last_q_start = (seq_len - 1) * n_heads * head_dim;
                    let last_q_5 = &q_data_all[last_q_start..last_q_start + 5.min(q_data_all.len() - last_q_start)];
                    eprintln!("[DIAG] Layer 0 Q LAST token (head 0, first 5): {:?}", last_q_5);
                    let nan_count = s.iter().filter(|v| v.is_nan()).count();
                    let inf_count = s.iter().filter(|v| v.is_infinite()).count();
                    if nan_count > 0 || inf_count > 0 {
                        eprintln!("[DIAG]   WARNING attn: NaN={}, Inf={}", nan_count, inf_count);
                    }
                }

                // O projection + post-attention norm + residual add
                {
                    let mut encoder = self.device.command_encoder()?;

                    let o_proj_tmp = if o_gpu {
                        self.quantized_projection_with_encoder(
                            &mut encoder, &attn_out, &o_base,
                            seq_len, n_heads * head_dim, hidden_size,
                        )?
                    } else {
                        encoder.commit_and_wait()?;
                        let mut o_enc = self.device.command_encoder()?;
                        let result = self.quantized_projection_with_encoder(
                            &mut o_enc, &attn_out, &o_base,
                            seq_len, n_heads * head_dim, hidden_size,
                        )?;
                        encoder = self.device.command_encoder()?;
                        result
                    };

                    let post_attn_norm_name = format!(
                        "model.layers.{layer_idx}.post_attention_layernorm.weight"
                    );
                    residual1 = self.fused_norm_add_with_encoder(
                        &mut encoder, input, &o_proj_tmp, &post_attn_norm_name, seq_len,
                    )?;

                    encoder.commit_and_wait()?;
                    o_proj = o_proj_tmp;
                }

                if diag {
                    let o_data = read_buffer_f32(&o_proj)?;
                    let last_o = (seq_len - 1) * hidden_size;
                    eprintln!("[DIAG] Layer 0 O proj LAST token first 5: {:?}", &o_data[last_o..last_o + 5]);
                    let r1_data = read_buffer_f32(&residual1)?;
                    eprintln!("[DIAG] Layer 0 residual1 LAST token first 5: {:?}", &r1_data[last_o..last_o + 5]);
                }
            }
        } // end !mega_a

        // --- Step 8: Feedforward with pre/post norms ---
        // Check if this layer has MoE (router exists)
        let router_weight_name = format!("model.layers.{layer_idx}.router.proj.weight");
        let has_moe_router = self.weights.get(&router_weight_name).is_some()
            || self.weights.get(&format!("language_model.{}", router_weight_name)).is_some();

        if has_moe_router {
            // --- MoE path: two parallel branches ---
            // dense_ffn and moe_ffn both do CPU reads internally, so we batch
            // what we can around them.
            let t_moe_start = if timing { Some(std::time::Instant::now()) } else { None };

            // =============================================================
            // DECODE FAST PATH (Step 8): Merge pre-FFN norms + MoE gate
            // into 1 commit, then dense + experts + post-FFN into 1 commit.
            // Saves 2 commits per MoE layer vs the original path.
            // =============================================================
            if seq_len == 1 {
                // Pre-FFN norms + MoE gate in ONE encoder
                let mut norms_gate_enc = self.device.command_encoder()?;
                let pre_ff_norm_name = format!(
                    "model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
                );
                let pre_ff_normed = self.apply_rms_norm_with_encoder(
                    &mut norms_gate_enc, &residual1, &pre_ff_norm_name, seq_len,
                )?;
                let pre_ff_norm2_name = format!(
                    "model.layers.{layer_idx}.pre_feedforward_layernorm_2.weight"
                );
                let pre_ff_normed2 = self.apply_rms_norm_with_encoder(
                    &mut norms_gate_enc, &residual1, &pre_ff_norm2_name, seq_len,
                )?;

                // MoE gate routing in the same encoder
                let use_cpu_route = std::env::var("HF2Q_CPU_MOE_ROUTE").is_ok();
                let (route_ids_buf, route_weights_buf) = if !use_cpu_route {
                    let (ids, wts) = self.gpu_moe_route_encode(
                        &mut norms_gate_enc, layer_idx, &residual1, seq_len,
                    )?;
                    (Some(ids), Some(wts))
                } else {
                    (None, None)
                };

                // COMMIT: norms + gate
                norms_gate_enc.commit_and_wait()?;

                // Read back routing results
                let (all_expert_ids, all_routing_weights) = if let (Some(ref ids), Some(ref wts)) =
                    (route_ids_buf, route_weights_buf)
                {
                    self.gpu_moe_route_readback(ids, wts, seq_len)?
                } else {
                    self.cpu_moe_route(layer_idx, &residual1, seq_len)?
                };

                // Dense FFN + expert dispatch + MEGA-D in ONE encoder
                let mut merged_enc = self.device.command_encoder()?;
                let dense_result = self.dense_ffn_with_encoder(
                    &mut merged_enc, layer_idx, &pre_ff_normed, seq_len,
                )?;
                let mlp_out = match dense_result {
                    Some(buf) => buf,
                    None => {
                        drop(merged_enc);
                        let mlp = self.dense_ffn(layer_idx, &pre_ff_normed, seq_len)?;
                        merged_enc = self.device.command_encoder()?;
                        mlp
                    }
                };

                // Expert dispatch (inlined)
                let intermediate_size_s8 = self.config.moe_intermediate_for_layer(layer_idx);
                let gate_name_s8 = format!("model.layers.{layer_idx}.experts.switch_glu.gate_proj.weight");
                let gate_w_s8 = require_weight(&self.weights, &gate_name_s8)?;
                let qm_s8 = gate_w_s8.quant_meta.as_ref().ok_or_else(|| Gemma4Error::ForwardError {
                    reason: "step8 decode: expert weights lack quant_meta".into(),
                })?;
                let bits_s8 = qm_s8.bits as u32;
                let group_size_s8 = qm_s8.group_size as u32;

                let mut gpu_output_bufs_s8: Vec<(f32, MlxBuffer)> = Vec::with_capacity(
                    self.config.top_k_experts,
                );
                for (rank, &expert_id) in all_expert_ids[0].iter().enumerate() {
                    let w = all_routing_weights[0][rank];
                    if w.abs() < 1e-10 { continue; }
                    let out_buf = self.expert_ffn_3d_gpu_buf_with_encoder(
                        &mut merged_enc, layer_idx, expert_id, &pre_ff_normed2,
                        0, hidden_size, intermediate_size_s8, bits_s8, group_size_s8,
                    )?;
                    gpu_output_bufs_s8.push((w, out_buf));
                }

                // MoE accumulation
                let accum_buf_s8 = self.device.alloc_buffer(
                    hidden_size * std::mem::size_of::<f32>(),
                    DType::F32, vec![1, hidden_size],
                )?;
                moe_ops::moe_zero_buffer_encode(
                    &mut merged_enc, &mut self.registry,
                    self.device.metal_device(),
                    &accum_buf_s8, hidden_size,
                )?;
                for (weight, buf) in &gpu_output_bufs_s8 {
                    moe_ops::moe_accumulate_encode(
                        &mut merged_enc, &mut self.registry,
                        self.device.metal_device(),
                        &accum_buf_s8, buf, *weight, hidden_size,
                    )?;
                }
                let moe_out = self.device.alloc_buffer(
                    hidden_size * 2, DType::BF16, vec![1, hidden_size],
                )?;
                elementwise::cast(
                    &mut merged_enc, &mut self.registry,
                    self.device.metal_device(),
                    &accum_buf_s8, &moe_out, hidden_size,
                    elementwise::CastDirection::F32ToBF16,
                )?;

                // Post-FFN norms + combine + residual + scalar (MEGA-D in same encoder)
                let post_ff_norm1_name = format!(
                    "model.layers.{layer_idx}.post_feedforward_layernorm_1.weight"
                );
                let mlp_normed = self.apply_rms_norm_with_encoder(
                    &mut merged_enc, &mlp_out, &post_ff_norm1_name, seq_len,
                )?;
                let post_ff_norm2_name = format!(
                    "model.layers.{layer_idx}.post_feedforward_layernorm_2.weight"
                );
                let moe_normed = self.apply_rms_norm_with_encoder(
                    &mut merged_enc, &moe_out, &post_ff_norm2_name, seq_len,
                )?;
                let combined = self.elementwise_add_with_encoder(
                    &mut merged_enc, &mlp_normed, &moe_normed, hidden_size,
                )?;
                let post_ff_norm_name = format!(
                    "model.layers.{layer_idx}.post_feedforward_layernorm.weight"
                );
                let pre_scalar = self.fused_norm_add_with_encoder(
                    &mut merged_enc, &residual1, &combined, &post_ff_norm_name, seq_len,
                )?;

                let scalar_name_s8 = format!("model.layers.{layer_idx}.layer_scalar");
                let mega_d_output = match require_weight(&self.weights, &scalar_name_s8) {
                    Ok(w) => {
                        let sv = if w.buffer.byte_len() >= 4 && w.buffer.dtype() == DType::F32 {
                            let s: &[f32] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                                reason: format!("read layer_scalar: {e}"),
                            })?;
                            s[0]
                        } else {
                            let raw: &[u16] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                                reason: format!("read layer_scalar bf16: {e}"),
                            })?;
                            half::bf16::from_bits(raw[0]).to_f32()
                        };
                        let scaled = self.device.alloc_buffer(
                            hidden_size * 2, DType::BF16, vec![hidden_size],
                        )?;
                        elementwise::scalar_mul_bf16(
                            &mut merged_enc, &mut self.registry, self.device.metal_device(),
                            &pre_scalar, &scaled, hidden_size, sv,
                        )?;
                        scaled
                    }
                    Err(_) => pre_scalar,
                };

                // COMMIT: dense + experts + post-FFN
                merged_enc.commit_and_wait()?;

                if let Some(moe_start) = t_moe_start {
                    eprintln!("[TIMING] L0 decode-fast-s8: {:.2}ms", moe_start.elapsed().as_secs_f64() * 1000.0);
                }
                if let Some(start) = t_layer_start {
                    eprintln!("[TIMING] L0 total forward_layer: {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
                }
                return Ok(mega_d_output);
            }

            // --- Prefill (seq_len > 1) path below ---

            // BATCH 3a: Pre-FFN norms for both paths
            let (pre_ff_normed, pre_ff_normed2);
            {
                let mut encoder = self.device.command_encoder()?;
                let pre_ff_norm_name = format!(
                    "model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
                );
                pre_ff_normed = self.apply_rms_norm_with_encoder(
                    &mut encoder, &residual1, &pre_ff_norm_name, seq_len,
                )?;
                let pre_ff_norm2_name = format!(
                    "model.layers.{layer_idx}.pre_feedforward_layernorm_2.weight"
                );
                pre_ff_normed2 = self.apply_rms_norm_with_encoder(
                    &mut encoder, &residual1, &pre_ff_norm2_name, seq_len,
                )?;
                encoder.commit_and_wait()?;
            }

            // =============================================================
            // MEGA-ENCODER C: Dense FFN + MoE expert dispatch + accumulation
            // in ONE command encoder when all dense FFN weights are GPU.
            // Otherwise falls back to separate encoders.
            //
            // The CPU MoE routing (Phase 1) reads from buffers committed by
            // Mega-B above. It runs interleaved with GPU dispatch encoding.
            // =============================================================

            // Try GPU-only dense FFN path first
            let mut mega_c_encoder = self.device.command_encoder()?;
            let dense_result = self.dense_ffn_with_encoder(
                &mut mega_c_encoder, layer_idx, &pre_ff_normed, seq_len,
            )?;

            let (mlp_out, moe_out);
            if let Some(dense_buf) = dense_result {
                // GPU fast path: dense FFN dispatched into mega_c_encoder.
                // Now do MoE CPU routing (reads from Mega-B committed buffers),
                // then dispatch MoE experts into the SAME encoder.
                mlp_out = dense_buf;
                // Multi-token prefill: commit dense FFN first, then run moe_ffn
                // with its own encoder (it needs CPU readback for accumulation).
                mega_c_encoder.commit_and_wait()?;
                moe_out = self.moe_ffn(layer_idx, &residual1, &pre_ff_normed2, seq_len)?;
            } else {
                // Slow path: dense FFN needs CPU fallback.
                // Drop the encoder (no dispatches were added since dense returned None).
                drop(mega_c_encoder);
                mlp_out = self.dense_ffn(layer_idx, &pre_ff_normed, seq_len)?;
                moe_out = self.moe_ffn(layer_idx, &residual1, &pre_ff_normed2, seq_len)?;
            }

            // =============================================================
            // MEGA-ENCODER D (MoE): Post-FFN norms + combine + final norm +
            // residual add + layer scalar — ALL in ONE command encoder.
            // Reduces 2 commit_and_wait() to 1.
            // =============================================================
            let n_elements = seq_len * hidden_size;
            let scalar_name = format!("model.layers.{layer_idx}.layer_scalar");
            let scalar_val_moe = match require_weight(&self.weights, &scalar_name) {
                Ok(w) => {
                    if w.buffer.byte_len() >= 4 && w.buffer.dtype() == DType::F32 {
                        let s: &[f32] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                            reason: format!("read layer_scalar: {e}"),
                        })?;
                        Some(s[0])
                    } else {
                        let raw: &[u16] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                            reason: format!("read layer_scalar bf16: {e}"),
                        })?;
                        Some(half::bf16::from_bits(raw[0]).to_f32())
                    }
                }
                Err(_) => None,
            };

            let mega_d_output;
            {
                let mut encoder = self.device.command_encoder()?;
                let post_ff_norm1_name = format!(
                    "model.layers.{layer_idx}.post_feedforward_layernorm_1.weight"
                );
                let mlp_normed = self.apply_rms_norm_with_encoder(
                    &mut encoder, &mlp_out, &post_ff_norm1_name, seq_len,
                )?;
                let post_ff_norm2_name = format!(
                    "model.layers.{layer_idx}.post_feedforward_layernorm_2.weight"
                );
                let moe_normed = self.apply_rms_norm_with_encoder(
                    &mut encoder, &moe_out, &post_ff_norm2_name, seq_len,
                )?;

                let combined = self.elementwise_add_with_encoder(
                    &mut encoder, &mlp_normed, &moe_normed, seq_len * hidden_size,
                )?;

                let post_ff_norm_name = format!(
                    "model.layers.{layer_idx}.post_feedforward_layernorm.weight"
                );

                // Fused norm + residual add (same encoder)
                let pre_scalar = self.fused_norm_add_with_encoder(
                    &mut encoder, &residual1, &combined, &post_ff_norm_name, seq_len,
                )?;

                if let Some(sv) = scalar_val_moe {
                    let scaled = self.device.alloc_buffer(
                        n_elements * 2, DType::BF16, vec![n_elements],
                    )?;
                    elementwise::scalar_mul_bf16(
                        &mut encoder, &mut self.registry, self.device.metal_device(),
                        &pre_scalar, &scaled, n_elements, sv,
                    )?;
                    mega_d_output = scaled;
                } else {
                    mega_d_output = pre_scalar;
                }

                encoder.commit_and_wait()?;

                if let Some(moe_start) = t_moe_start {
                    eprintln!("[TIMING] L0 FFN+MoE total: {:.2}ms", moe_start.elapsed().as_secs_f64() * 1000.0);
                }

                if diag {
                    let d = read_buffer_f32(&mlp_normed)?;
                    let last = (seq_len - 1) * hidden_size;
                    eprintln!("[DIAG] Layer 0 dense_mlp_normed LAST token first 5: {:?}", &d[last..last + 5]);
                    let d = read_buffer_f32(&moe_normed)?;
                    eprintln!("[DIAG] Layer 0 moe_normed LAST token first 5: {:?}", &d[last..last + 5]);
                }
            }

            if diag {
                let o = read_buffer_f32(&mega_d_output)?;
                let last = (seq_len - 1) * hidden_size;
                eprintln!("[DIAG] Layer 0 FINAL output LAST token first 5: {:?}", &o[last..last + 5]);
            }

            if let Some(start) = t_layer_start {
                eprintln!("[TIMING] L0 total forward_layer: {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
            }

            return Ok(mega_d_output);
        } else {
            // --- Non-MoE path: simple MLP with pre/post norms ---
            // Pre-FFN norm (batched alone since dense_ffn follows with its own commits)
            let pre_ff_normed;
            {
                let mut encoder = self.device.command_encoder()?;
                let pre_ff_norm_name = format!(
                    "model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
                );
                pre_ff_normed = self.apply_rms_norm_with_encoder(
                    &mut encoder, &residual1, &pre_ff_norm_name, seq_len,
                )?;
                encoder.commit_and_wait()?;
            }

            let mlp_out = self.dense_ffn(layer_idx, &pre_ff_normed, seq_len)?;

            // =============================================================
            // MEGA-ENCODER D (non-MoE): Post-FFN norm + residual add +
            // layer scalar — ALL in ONE command encoder.
            // Reduces 2 commit_and_wait() to 1.
            // =============================================================
            let n_elements = seq_len * hidden_size;
            let scalar_name = format!("model.layers.{layer_idx}.layer_scalar");
            let scalar_val = match require_weight(&self.weights, &scalar_name) {
                Ok(w) => {
                    if w.buffer.byte_len() >= 4 && w.buffer.dtype() == DType::F32 {
                        let s: &[f32] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                            reason: format!("read layer_scalar: {e}"),
                        })?;
                        Some(s[0])
                    } else {
                        let raw: &[u16] = w.buffer.as_slice().map_err(|e| Gemma4Error::ForwardError {
                            reason: format!("read layer_scalar bf16: {e}"),
                        })?;
                        Some(half::bf16::from_bits(raw[0]).to_f32())
                    }
                }
                Err(_) => None,
            };

            let output;
            {
                let mut encoder = self.device.command_encoder()?;
                let post_ff_norm_name = format!(
                    "model.layers.{layer_idx}.post_feedforward_layernorm.weight"
                );

                // Fused norm + residual add (same encoder)
                let pre_scalar = self.fused_norm_add_with_encoder(
                    &mut encoder, &residual1, &mlp_out, &post_ff_norm_name, seq_len,
                )?;

                if let Some(sv) = scalar_val {
                    let scaled = self.device.alloc_buffer(
                        n_elements * 2, DType::BF16, vec![n_elements],
                    )?;
                    elementwise::scalar_mul_bf16(
                        &mut encoder, &mut self.registry, self.device.metal_device(),
                        &pre_scalar, &scaled, n_elements, sv,
                    )?;
                    output = scaled;
                } else {
                    output = pre_scalar;
                }

                encoder.commit_and_wait()?;
            }

            if diag {
                let o = read_buffer_f32(&output)?;
                let last = (seq_len - 1) * hidden_size;
                eprintln!("[DIAG] Layer 0 FINAL output LAST token first 5: {:?}", &o[last..last + 5]);
            }

            return Ok(output);
        }
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

    /// Apply RMS normalization using a shared command encoder (batched variant).
    ///
    /// Same as [`apply_rms_norm`] but dispatches into the caller's encoder
    /// instead of creating and committing its own. The caller is responsible
    /// for calling `commit_and_wait()` on the encoder after all batched
    /// dispatches are encoded.
    fn apply_rms_norm_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        input: &MlxBuffer,
        weight_name: &str,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let input_dtype = input.dtype();

        let weight = require_weight(&self.weights, weight_name)?;
        let norm_weight_buf = clone_buffer(&self.device, &weight.buffer)?;

        let elem_size = input_dtype.size_of();
        let output_bytes = seq_len * hidden_size * elem_size;
        let output = self.device.alloc_buffer(
            output_bytes,
            input_dtype,
            vec![seq_len, hidden_size],
        )?;

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

        let norm_weight = match input_dtype {
            DType::BF16 => {
                match norm_weight_buf.dtype() {
                    DType::BF16 => clone_buffer(&self.device, &norm_weight_buf)?,
                    DType::F32 => {
                        self.cast_f32_to_bf16_with_encoder(encoder, &norm_weight_buf, hidden_size)?
                    }
                    DType::F16 => {
                        let f32_buf = self.cast_with_encoder(
                            encoder, &norm_weight_buf, hidden_size,
                            DType::F32, elementwise::CastDirection::F16ToF32,
                        )?;
                        self.cast_f32_to_bf16_with_encoder(encoder, &f32_buf, hidden_size)?
                    }
                    _ => {
                        return Err(Gemma4Error::ForwardError {
                            reason: format!("Unsupported norm weight dtype for bf16 pipeline: {}", norm_weight_buf.dtype()),
                        });
                    }
                }
            }
            _ => {
                self.ensure_f32_weight_with_encoder(encoder, &norm_weight_buf, hidden_size, weight_name)?
            }
        };

        rms_norm::dispatch_rms_norm(
            encoder,
            &mut self.registry,
            self.device.metal_device(),
            input,
            &norm_weight,
            &output,
            &params_buf,
            seq_len as u32,
            hidden_size as u32,
        )?;

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
            if !use_cpu_qmatmul && (bits == 4 || bits == 6 || bits == 8) {
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
                let output_f32 = qmatmul::quantized_matmul_simd(
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

    /// Generic cast dispatch into a shared command encoder.
    ///
    /// Encodes a cast operation without committing. The caller must commit the
    /// encoder after all dispatches are encoded.
    fn cast_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        input: &MlxBuffer,
        n_elements: usize,
        out_dtype: DType,
        direction: elementwise::CastDirection,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let out_bytes = n_elements * out_dtype.size_of();
        let output = self.device.alloc_buffer(out_bytes, out_dtype, vec![n_elements])?;
        elementwise::cast(
            encoder,
            &mut self.registry,
            self.device.metal_device(),
            input,
            &output,
            n_elements,
            direction,
        )?;
        Ok(output)
    }

    /// Cast f32 to bf16 using a shared command encoder (batched variant).
    fn cast_f32_to_bf16_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        input: &MlxBuffer,
        n_elements: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        self.cast_with_encoder(
            encoder, input, n_elements,
            DType::BF16, elementwise::CastDirection::F32ToBF16,
        )
    }

    /// Cast bf16 to f32 using a shared command encoder (batched variant).
    fn cast_bf16_to_f32_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        input: &MlxBuffer,
        n_elements: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        self.cast_with_encoder(
            encoder, input, n_elements,
            DType::F32, elementwise::CastDirection::BF16ToF32,
        )
    }

    /// Ensure f32 input using a shared command encoder (batched variant).
    fn ensure_f32_input_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        buffer: &MlxBuffer,
        n_elements: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        match buffer.dtype() {
            DType::F32 => {
                clone_buffer(&self.device, buffer)
                    .map_err(|e| Gemma4Error::ForwardError {
                        reason: format!("clone f32 input: {e}"),
                    })
            }
            DType::F16 => {
                self.cast_with_encoder(
                    encoder, buffer, n_elements,
                    DType::F32, elementwise::CastDirection::F16ToF32,
                )
            }
            DType::BF16 => {
                self.cast_with_encoder(
                    encoder, buffer, n_elements,
                    DType::F32, elementwise::CastDirection::BF16ToF32,
                )
            }
            other => Err(Gemma4Error::ForwardError {
                reason: format!("Unsupported input dtype for f32 cast: {other}"),
            }),
        }
    }

    /// Ensure weight is f32 using a shared command encoder (batched variant).
    fn ensure_f32_weight_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        buffer: &MlxBuffer,
        n_elements: usize,
        _name: &str,
    ) -> Result<MlxBuffer, Gemma4Error> {
        match buffer.dtype() {
            DType::F32 => {
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
                self.cast_with_encoder(
                    encoder, buffer, n_elements,
                    DType::F32, elementwise::CastDirection::F16ToF32,
                )
            }
            DType::BF16 => {
                self.cast_with_encoder(
                    encoder, buffer, n_elements,
                    DType::F32, elementwise::CastDirection::BF16ToF32,
                )
            }
            other => Err(Gemma4Error::ForwardError {
                reason: format!("Unsupported weight dtype: {other}"),
            }),
        }
    }

    /// Elementwise add using a shared command encoder (batched variant).
    fn elementwise_add_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
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

        elementwise::elementwise_add(
            encoder,
            &mut self.registry,
            self.device.metal_device(),
            a,
            b,
            &output,
            n_elements,
            dtype,
        )?;

        Ok(output)
    }

    /// Fused RMS norm + residual add using a shared command encoder.
    ///
    /// Computes: output = residual + rms_norm(input, weight, eps)
    ///
    /// When the fused_norm_add_bf16 kernel is available, this dispatches a single
    /// GPU kernel instead of separate norm + add. Falls back to the two-step
    /// pattern when the kernel is not yet built.
    fn fused_norm_add_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        residual: &MlxBuffer,
        input: &MlxBuffer,
        weight_name: &str,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let _eps = self.config.rms_norm_eps; // used by fused kernel when available
        let n_elements = seq_len * hidden_size;

        // TODO(kernel-agent): Wire fused_norm_add_bf16 when available.
        // Check env var to enable once the kernel lands.
        let use_fused = std::env::var("HF2Q_FUSED_NORM_ADD").is_ok();
        if use_fused {
            // Get norm weight as bf16
            let weight = require_weight(&self.weights, weight_name)?;
            let norm_weight_buf = match weight.buffer.dtype() {
                DType::BF16 => clone_buffer(&self.device, &weight.buffer)?,
                _ => {
                    let f32_data = read_weight_as_f32(weight)?;
                    f32_vec_to_bf16_buffer(&self.device, &f32_data, vec![hidden_size])?
                }
            };

            let output = self.device.alloc_buffer(
                n_elements * 2, DType::BF16, vec![seq_len, hidden_size],
            )?;

            // TODO: Uncomment when kernel is available:
            // mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_bf16(
            //     encoder, residual, input, &norm_weight_buf, &output,
            //     hidden_size as u32, seq_len as u32, eps,
            //     &mut self.registry,
            // )?;
            // return Ok(output);

            // Fallback: two-step norm + add (same encoder, still batched)
            let _ = (output, norm_weight_buf); // suppress unused warnings
        }

        // Two-step fallback: rms_norm then elementwise_add
        let normed = self.apply_rms_norm_with_encoder(encoder, input, weight_name, seq_len)?;
        self.elementwise_add_with_encoder(encoder, residual, &normed, n_elements)
    }

    /// Quantized projection using a shared command encoder (batched variant).
    ///
    /// Batches the ensure_f32_input cast, qmatmul, and cast_f32_to_bf16 into
    /// the caller's encoder instead of creating 3 separate encoders.
    fn quantized_projection_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        input: &MlxBuffer,
        weight_base: &str,
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let weight_name = format!("{weight_base}.weight");
        let scales_name = format!("{weight_base}.scales");
        let biases_name = format!("{weight_base}.biases");

        let loaded = require_weight(&self.weights, &weight_name)?;
        let quant_info = loaded.quant_meta.as_ref().map(|qm| (qm.bits, qm.group_size));

        if let Some((bits, _group_size)) = quant_info {
            let use_cpu_qmatmul = std::env::var("HF2Q_CPU_QMATMUL").is_ok();
            let use_bf16_qmatmul = std::env::var("HF2Q_BF16_QMATMUL").is_ok();
            if !use_cpu_qmatmul && use_bf16_qmatmul && (bits == 4 || bits == 6 || bits == 8) {
                // GPU path: direct bf16 qmatmul — no cast chain needed.
                // Input is bf16, output is bf16. Eliminates 2 cast dispatches per projection.
                // NOTE: Gated behind HF2Q_BF16_QMATMUL=1 until kernel is validated.
                //
                // Clone weight buffer refs upfront to avoid borrow conflict with &mut self
                // methods (cast_f32_to_bf16_with_encoder needs &mut self).
                let weight_buf = clone_buffer(&self.device, &require_weight(&self.weights, &weight_name)?.buffer)?;
                let scales_buf = clone_buffer(&self.device, &require_weight(&self.weights, &scales_name)?.buffer)?;
                let biases_buf = clone_buffer(&self.device, &require_weight(&self.weights, &biases_name)?.buffer)?;

                let params = QuantizedMatmulParams {
                    m: seq_len as u32,
                    k: in_dim as u32,
                    n: out_dim as u32,
                    group_size: _group_size as u32,
                    bits: bits as u32,
                };

                // Ensure input is bf16 for the bf16 kernel (it should already be).
                let bf16_input = if input.dtype() == DType::BF16 {
                    clone_buffer(&self.device, input)?
                } else {
                    self.cast_f32_to_bf16_with_encoder(encoder, input, seq_len * in_dim)?
                };

                let output_bf16 = qmatmul::dispatch_quantized_matmul_simd_bf16(
                    encoder,
                    &mut self.registry,
                    &self.device,
                    &bf16_input,
                    &weight_buf,
                    &scales_buf,
                    &biases_buf,
                    &params,
                )?;

                return Ok(output_bf16);
            }
            if !use_cpu_qmatmul && (bits == 4 || bits == 6 || bits == 8) {
                // GPU path: f32 qmatmul with cast chain (default, proven correct).
                // Cast bf16 input -> f32, run f32 qmatmul, cast f32 output -> bf16.
                let weight_buf = clone_buffer(&self.device, &require_weight(&self.weights, &weight_name)?.buffer)?;
                let scales_buf = clone_buffer(&self.device, &require_weight(&self.weights, &scales_name)?.buffer)?;
                let biases_buf = clone_buffer(&self.device, &require_weight(&self.weights, &biases_name)?.buffer)?;

                let params = QuantizedMatmulParams {
                    m: seq_len as u32,
                    k: in_dim as u32,
                    n: out_dim as u32,
                    group_size: _group_size as u32,
                    bits: bits as u32,
                };

                // Ensure input is f32 for the f32 kernel
                let f32_input = self.ensure_f32_input_with_encoder(encoder, input, seq_len * in_dim)?;

                let output_f32 = qmatmul::quantized_matmul_simd(
                    encoder, &mut self.registry, &self.device,
                    &f32_input, &weight_buf, &scales_buf, &biases_buf, &params,
                )?;

                // Cast f32 output to bf16
                let output_bf16 = self.cast_f32_to_bf16_with_encoder(encoder, &output_f32, seq_len * out_dim)?;
                return Ok(output_bf16);
            }

            // CPU dequantize + bf16-precision matmul (same as non-batched path).
            // Must commit the encoder first to ensure any prior GPU work is done,
            // then do CPU work, then the encoder is "used up" -- caller will need
            // a new one after this returns.
            // NOTE: We flush the encoder to ensure GPU data is available for CPU reads.
            encoder.commit_and_wait()?;
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

            return f32_vec_to_bf16_buffer(&self.device, &output_data, vec![seq_len, out_dim]);
        }

        // Non-quantized path: CPU matmul fallback -- same commit-first strategy
        encoder.commit_and_wait()?;
        let weight_buf = clone_buffer(
            &self.device,
            &require_weight(&self.weights, &weight_name)?.buffer,
        )?;
        let f32_result = self.cpu_matmul(input, &weight_buf, seq_len, in_dim, out_dim)?;
        self.cast_f32_to_bf16(&f32_result, seq_len * out_dim)
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

        // Compute position offset from the last layer's KV cache (it hasn't
        // been appended yet during this forward() call, so write_position
        // reflects the pre-forward state).
        let pos_offset = if self.kv_cache.num_layers() > 0 {
            let last_layer = self.kv_cache.num_layers() - 1;
            match self.kv_cache.layer(last_layer) {
                Ok(cache) => cache.write_position(),
                Err(_) => 0,
            }
        } else {
            0
        };

        // Build positions buffer: [pos_offset, pos_offset+1, ..., pos_offset+seq_len-1]
        let positions_vec: Vec<u32> = (0..seq_len).map(|i| (pos_offset + i) as u32).collect();
        let mut positions_buf = self.device.alloc_buffer(
            seq_len * std::mem::size_of::<u32>(),
            DType::U32,
            vec![seq_len],
        )?;
        {
            let s: &mut [u32] = positions_buf.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("RoPE positions write: {e}"),
            })?;
            s.copy_from_slice(&positions_vec);
        }

        // Build params buffer: [theta, head_dim, rope_dim, 0] as f32
        let mut params_buf = self.device.alloc_buffer(
            4 * std::mem::size_of::<f32>(),
            DType::F32,
            vec![4],
        )?;
        {
            let s: &mut [f32] = params_buf.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                reason: format!("RoPE params write: {e}"),
            })?;
            s[0] = theta;
            s[1] = head_dim as f32;
            s[2] = rope_dim as f32;
            s[3] = 0.0;
        }

        // Output buffer: same shape as input (bf16)
        let total_elements = seq_len * n_heads * head_dim;
        let output = self.device.alloc_buffer(
            total_elements * 2, // bf16
            DType::BF16,
            input.shape().to_vec(),
        )?;

        // GPU dispatch: rope_neox_bf16
        // Input layout [seq_len, n_heads * head_dim] == [seq_len * n_heads, head_dim]
        let mut encoder = self.device.command_encoder()?;
        rope::dispatch_rope_neox_bf16(
            &mut encoder,
            &mut self.registry,
            self.device.metal_device(),
            input,
            &output,
            &params_buf,
            &positions_buf,
            seq_len as u32,
            n_heads as u32,
            head_dim as u32,
            rope_dim as u32,
        )?;
        encoder.commit_and_wait()?;

        Ok(output)
    }

    /// Compute attention for a single layer.
    ///
    /// Q is bf16 [seq_len, n_heads * head_dim]. K/V are read from the bf16 KV cache
    /// and reshaped directly as bf16 — no f32 conversion needed.
    /// Output is bf16 [seq_len, n_heads * head_dim].
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
        let mut encoder = self.device.command_encoder()?;
        let out_flat = self.compute_attention_with_encoder(
            &mut encoder, layer_idx, q, seq_len, n_heads, n_kv_heads, head_dim, layer_type,
        )?;
        encoder.commit_and_wait()?;
        Ok(out_flat)
    }

    /// Encode attention dispatches (permutes + SDPA + output permute) into an
    /// external command encoder WITHOUT committing. The caller is responsible
    /// for calling `commit_and_wait()` on the encoder after all batched
    /// dispatches are encoded.
    fn compute_attention_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        layer_idx: usize,
        q: &MlxBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        layer_type: LayerAttentionType,
    ) -> Result<MlxBuffer, Gemma4Error> {
        // For KV-shared layers, read K/V from the donor layer's cache.
        let (k_cache_buf, v_cache_buf, kv_seq_len) = self.kv_cache.keys_values_for(layer_idx)?;

        if kv_seq_len == 0 {
            let effective = self.kv_cache.donor_for(layer_idx).unwrap_or(layer_idx);
            return Err(Gemma4Error::ForwardError {
                reason: format!("KV cache is empty for layer {layer_idx} (effective cache layer {effective})"),
            });
        }

        // Q is bf16 [seq_len, n_heads * head_dim] == [seq_len, n_heads, head_dim].
        // SDPA expects [batch, n_heads, seq_len, head_dim].
        // GPU permute_021: [seq_len, n_heads, head_dim] -> [n_heads, seq_len, head_dim]
        let q_buf = self.device.alloc_buffer(
            seq_len * n_heads * head_dim * 2,
            DType::BF16,
            vec![1, n_heads, seq_len, head_dim],
        )?;
        transpose::permute_021_bf16(
            encoder, &mut self.registry, self.device.metal_device(),
            q, &q_buf,
            seq_len, n_heads, head_dim,
        )?;

        // K/V cache is bf16 [capacity, n_kv_heads * head_dim] (Metal shared buffer).
        // GPU permute_021: [kv_seq_len, n_kv_heads, head_dim] -> [n_kv_heads, kv_seq_len, head_dim]
        let k_buf = self.device.alloc_buffer(
            kv_seq_len * n_kv_heads * head_dim * 2,
            DType::BF16,
            vec![1, n_kv_heads, kv_seq_len, head_dim],
        )?;
        let v_buf = self.device.alloc_buffer(
            kv_seq_len * n_kv_heads * head_dim * 2,
            DType::BF16,
            vec![1, n_kv_heads, kv_seq_len, head_dim],
        )?;
        transpose::permute_021_bf16(
            encoder, &mut self.registry, self.device.metal_device(),
            k_cache_buf, &k_buf,
            kv_seq_len, n_kv_heads, head_dim,
        )?;
        transpose::permute_021_bf16(
            encoder, &mut self.registry, self.device.metal_device(),
            v_cache_buf, &v_buf,
            kv_seq_len, n_kv_heads, head_dim,
        )?;

        // Output is bf16 to match the pipeline
        let output_elements = 1 * n_heads * seq_len * head_dim;
        let output = self.device.alloc_buffer(
            output_elements * 2, // bf16 = 2 bytes
            DType::BF16,
            vec![1, n_heads, seq_len, head_dim],
        )?;

        match layer_type {
            LayerAttentionType::Sliding => {
                let params = sdpa_sliding::SdpaSlidingParams {
                    n_heads: n_heads as u32,
                    n_kv_heads: n_kv_heads as u32,
                    head_dim: head_dim as u32,
                    seq_len: seq_len as u32,
                    kv_seq_len: kv_seq_len as u32,
                    window_size: self.config.sliding_window as u32,
                    scale: 1.0, // Gemma 4: QK norms handle scaling
                };
                sdpa_sliding::sdpa_sliding(
                    encoder,
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
                    scale: 1.0, // Gemma 4: QK norms handle scaling
                };
                sdpa::sdpa(
                    encoder,
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

        // Reshape output from [1, n_heads, seq_len, head_dim] back to [seq_len, n_heads * head_dim]
        // GPU permute_021: [n_heads, seq_len, head_dim] -> [seq_len, n_heads, head_dim]
        let out_flat = self.device.alloc_buffer(
            n_heads * seq_len * head_dim * 2,
            DType::BF16,
            vec![seq_len, n_heads * head_dim],
        )?;
        transpose::permute_021_bf16(
            encoder, &mut self.registry, self.device.metal_device(),
            &output, &out_flat,
            n_heads, seq_len, head_dim,
        )?;

        Ok(out_flat)
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

                // GPU scalar_mul_bf16: multiply all elements by the scalar on GPU
                let output = self.device.alloc_buffer(
                    n_elements * 2, // bf16
                    DType::BF16,
                    vec![n_elements],
                )?;
                let mut encoder = self.device.command_encoder()?;
                elementwise::scalar_mul_bf16(
                    &mut encoder,
                    &mut self.registry,
                    self.device.metal_device(),
                    input,
                    &output,
                    n_elements,
                    scalar_val,
                )?;
                encoder.commit_and_wait()?;
                Ok(output)
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

    /// Ensure GPU buffers for MoE router are cached for the given layer.
    ///
    /// Populates `router_proj_gpu_cache`, `router_norm_gpu_cache`, and
    /// `per_expert_scale_gpu_cache` with f32 GPU buffers. Also ensures the
    /// CPU caches are populated (used as source data for the GPU upload).
    fn ensure_moe_gate_gpu_caches(
        &mut self,
        layer_idx: usize,
    ) -> Result<(), Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let n_experts = self.config.num_experts;

        // Ensure CPU caches are populated first
        if !self.router_proj_cache.contains_key(&layer_idx) {
            let router_f32_raw = self.dequantize_2d_weight(
                &format!("model.layers.{layer_idx}.router.proj"),
                n_experts,
                hidden_size,
            )?;
            self.router_proj_cache.insert(layer_idx, round_to_bf16(&router_f32_raw));
        }
        if !self.router_norm_cache.contains_key(&layer_idx) {
            let router_scale_vec = self.read_1d_weight(
                &format!("model.layers.{layer_idx}.router.scale"),
                hidden_size,
            ).unwrap_or_else(|_| vec![1.0f32; hidden_size]);
            let root_size = (hidden_size as f32).powf(-0.5);
            let norm_weight: Vec<f32> = router_scale_vec.iter().map(|s| s * root_size).collect();
            self.router_norm_cache.insert(layer_idx, norm_weight);
        }
        if !self.per_expert_scale_cache.contains_key(&layer_idx) {
            let pes = self.read_1d_weight(
                &format!("model.layers.{layer_idx}.router.per_expert_scale"),
                n_experts,
            ).unwrap_or_else(|_| vec![1.0f32; n_experts]);
            self.per_expert_scale_cache.insert(layer_idx, pes);
        }

        // Upload to GPU if not already cached
        if !self.router_proj_gpu_cache.contains_key(&layer_idx) {
            let cpu_data = self.router_proj_cache.get(&layer_idx).unwrap();
            let byte_len = cpu_data.len() * std::mem::size_of::<f32>();
            let mut buf = self.device.alloc_buffer(
                byte_len, DType::F32, vec![n_experts, hidden_size],
            )?;
            {
                let slice: &mut [f32] = buf.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("upload router_proj GPU: {e}"),
                })?;
                slice.copy_from_slice(cpu_data);
            }
            self.router_proj_gpu_cache.insert(layer_idx, buf);
        }
        if !self.router_norm_gpu_cache.contains_key(&layer_idx) {
            let cpu_data = self.router_norm_cache.get(&layer_idx).unwrap();
            let byte_len = cpu_data.len() * std::mem::size_of::<f32>();
            let mut buf = self.device.alloc_buffer(
                byte_len, DType::F32, vec![hidden_size],
            )?;
            {
                let slice: &mut [f32] = buf.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("upload router_norm GPU: {e}"),
                })?;
                slice.copy_from_slice(cpu_data);
            }
            self.router_norm_gpu_cache.insert(layer_idx, buf);
        }
        if !self.per_expert_scale_gpu_cache.contains_key(&layer_idx) {
            let cpu_data = self.per_expert_scale_cache.get(&layer_idx).unwrap();
            let byte_len = cpu_data.len() * std::mem::size_of::<f32>();
            let mut buf = self.device.alloc_buffer(
                byte_len, DType::F32, vec![n_experts],
            )?;
            {
                let slice: &mut [f32] = buf.as_mut_slice().map_err(|e| Gemma4Error::ForwardError {
                    reason: format!("upload per_expert_scale GPU: {e}"),
                })?;
                slice.copy_from_slice(cpu_data);
            }
            self.per_expert_scale_gpu_cache.insert(layer_idx, buf);
        }

        Ok(())
    }

    /// GPU MoE routing via moe_gate kernel. Returns (expert_ids, expert_weights)
    /// as CPU vectors after reading back the small output buffers.
    fn gpu_moe_route(
        &mut self,
        layer_idx: usize,
        router_input: &MlxBuffer,
        seq_len: usize,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>), Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let n_experts = self.config.num_experts;
        let top_k = self.config.top_k_experts;
        let eps = self.config.rms_norm_eps;

        self.ensure_moe_gate_gpu_caches(layer_idx)?;

        // Allocate output buffers for expert_ids and expert_weights
        let ids_buf = self.device.alloc_buffer(
            seq_len * top_k * std::mem::size_of::<u32>(),
            DType::U32,
            vec![seq_len, top_k],
        )?;
        let weights_buf = self.device.alloc_buffer(
            seq_len * top_k * std::mem::size_of::<f32>(),
            DType::F32,
            vec![seq_len, top_k],
        )?;

        let params = moe_gate::MoeGateParams {
            hidden_dim: hidden_size,
            n_experts,
            top_k,
            seq_len,
            rms_eps: eps,
        };

        {
            let mut encoder = self.device.command_encoder()?;
            let router_proj_buf = self.router_proj_gpu_cache.get(&layer_idx).unwrap();
            let router_norm_buf = self.router_norm_gpu_cache.get(&layer_idx).unwrap();
            let per_expert_scale_buf = self.per_expert_scale_gpu_cache.get(&layer_idx).unwrap();

            moe_gate::moe_gate(
                &mut encoder,
                &mut self.registry,
                self.device.metal_device(),
                router_input,
                router_proj_buf,
                router_norm_buf,
                per_expert_scale_buf,
                &ids_buf,
                &weights_buf,
                &params,
            )?;
            encoder.commit_and_wait()?;
        }

        // Read back small output buffers (seq_len * top_k * 4 bytes each, e.g. 32 bytes for decode)
        let ids_raw: &[u32] = ids_buf.as_slice().map_err(|e| Gemma4Error::ForwardError {
            reason: format!("read moe_gate expert_ids: {e}"),
        })?;
        let weights_raw: &[f32] = weights_buf.as_slice().map_err(|e| Gemma4Error::ForwardError {
            reason: format!("read moe_gate expert_weights: {e}"),
        })?;

        let mut all_expert_ids: Vec<Vec<usize>> = Vec::with_capacity(seq_len);
        let mut all_routing_weights: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let offset = t * top_k;
            let ids: Vec<usize> = ids_raw[offset..offset + top_k]
                .iter()
                .map(|&id| id as usize)
                .collect();
            let weights: Vec<f32> = weights_raw[offset..offset + top_k].to_vec();
            all_expert_ids.push(ids);
            all_routing_weights.push(weights);
        }

        Ok((all_expert_ids, all_routing_weights))
    }

    /// Dispatch MoE gate routing into an external command encoder WITHOUT
    /// committing. Returns the output buffers (expert IDs and weights) which
    /// must be read back AFTER the caller commits the encoder.
    ///
    /// This enables merging the MoE gate dispatch into a larger encoder
    /// (e.g., MEGA-A) to reduce commit_and_wait count.
    fn gpu_moe_route_encode(
        &mut self,
        encoder: &mut CommandEncoder,
        layer_idx: usize,
        router_input: &MlxBuffer,
        seq_len: usize,
    ) -> Result<(MlxBuffer, MlxBuffer), Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let n_experts = self.config.num_experts;
        let top_k = self.config.top_k_experts;
        let eps = self.config.rms_norm_eps;

        self.ensure_moe_gate_gpu_caches(layer_idx)?;

        let ids_buf = self.device.alloc_buffer(
            seq_len * top_k * std::mem::size_of::<u32>(),
            DType::U32,
            vec![seq_len, top_k],
        )?;
        let weights_buf = self.device.alloc_buffer(
            seq_len * top_k * std::mem::size_of::<f32>(),
            DType::F32,
            vec![seq_len, top_k],
        )?;

        let params = moe_gate::MoeGateParams {
            hidden_dim: hidden_size,
            n_experts,
            top_k,
            seq_len,
            rms_eps: eps,
        };

        let router_proj_buf = self.router_proj_gpu_cache.get(&layer_idx).unwrap();
        let router_norm_buf = self.router_norm_gpu_cache.get(&layer_idx).unwrap();
        let per_expert_scale_buf = self.per_expert_scale_gpu_cache.get(&layer_idx).unwrap();

        moe_gate::moe_gate(
            encoder,
            &mut self.registry,
            self.device.metal_device(),
            router_input,
            router_proj_buf,
            router_norm_buf,
            per_expert_scale_buf,
            &ids_buf,
            &weights_buf,
            &params,
        )?;

        Ok((ids_buf, weights_buf))
    }

    /// Read back MoE routing results from GPU buffers after commit.
    /// Must be called AFTER the encoder containing the moe_gate dispatch
    /// has been committed.
    fn gpu_moe_route_readback(
        &self,
        ids_buf: &MlxBuffer,
        weights_buf: &MlxBuffer,
        seq_len: usize,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>), Gemma4Error> {
        let top_k = self.config.top_k_experts;

        let ids_raw: &[u32] = ids_buf.as_slice().map_err(|e| Gemma4Error::ForwardError {
            reason: format!("read moe_gate expert_ids: {e}"),
        })?;
        let weights_raw: &[f32] = weights_buf.as_slice().map_err(|e| Gemma4Error::ForwardError {
            reason: format!("read moe_gate expert_weights: {e}"),
        })?;

        let mut all_expert_ids: Vec<Vec<usize>> = Vec::with_capacity(seq_len);
        let mut all_routing_weights: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let offset = t * top_k;
            let ids: Vec<usize> = ids_raw[offset..offset + top_k]
                .iter()
                .map(|&id| id as usize)
                .collect();
            let weights: Vec<f32> = weights_raw[offset..offset + top_k].to_vec();
            all_expert_ids.push(ids);
            all_routing_weights.push(weights);
        }

        Ok((all_expert_ids, all_routing_weights))
    }

    /// CPU fallback MoE routing. Used when GPU routing is unavailable or fails.
    fn cpu_moe_route(
        &mut self,
        layer_idx: usize,
        router_input: &MlxBuffer,
        seq_len: usize,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>), Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let n_experts = self.config.num_experts;
        let top_k = self.config.top_k_experts;
        let eps = self.config.rms_norm_eps;

        let router_data = read_buffer_f32(router_input)?;

        // Ensure CPU caches are populated
        if !self.router_proj_cache.contains_key(&layer_idx) {
            let router_f32_raw = self.dequantize_2d_weight(
                &format!("model.layers.{layer_idx}.router.proj"),
                n_experts,
                hidden_size,
            )?;
            self.router_proj_cache.insert(layer_idx, round_to_bf16(&router_f32_raw));
        }
        if !self.router_norm_cache.contains_key(&layer_idx) {
            let router_scale_vec = self.read_1d_weight(
                &format!("model.layers.{layer_idx}.router.scale"),
                hidden_size,
            ).unwrap_or_else(|_| vec![1.0f32; hidden_size]);
            let root_size = (hidden_size as f32).powf(-0.5);
            let norm_weight: Vec<f32> = router_scale_vec.iter().map(|s| s * root_size).collect();
            self.router_norm_cache.insert(layer_idx, norm_weight);
        }
        if !self.per_expert_scale_cache.contains_key(&layer_idx) {
            let pes = self.read_1d_weight(
                &format!("model.layers.{layer_idx}.router.per_expert_scale"),
                n_experts,
            ).unwrap_or_else(|_| vec![1.0f32; n_experts]);
            self.per_expert_scale_cache.insert(layer_idx, pes);
        }
        let router_bf16 = self.router_proj_cache.get(&layer_idx).unwrap();
        let router_norm_weight = self.router_norm_cache.get(&layer_idx).unwrap();
        let per_expert_scale = self.per_expert_scale_cache.get(&layer_idx).unwrap();

        let mut all_expert_ids: Vec<Vec<usize>> = Vec::with_capacity(seq_len);
        let mut all_routing_weights: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let token_router = &router_data[t * hidden_size..(t + 1) * hidden_size];
            let normed_router = cpu_rms_norm(token_router, router_norm_weight, eps);
            let normed_bf16 = round_to_bf16(&normed_router);

            let mut logits = vec![0.0f32; n_experts];
            for e in 0..n_experts {
                let mut sum = 0.0f32;
                for k in 0..hidden_size {
                    sum += router_bf16[e * hidden_size + k] * normed_bf16[k];
                }
                logits[e] = sum;
            }

            let (expert_ids, routing_weights) =
                top_k_softmax_with_scale(&logits, per_expert_scale, top_k);

            all_expert_ids.push(expert_ids);
            all_routing_weights.push(routing_weights);
        }

        Ok((all_expert_ids, all_routing_weights))
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
        let diag = std::env::var("HF2Q_DIAG").is_ok() && layer_idx == 0;

        let expert_data = read_buffer_f32(expert_input)?;

        if diag {
            let router_weight_name = format!("model.layers.{layer_idx}.router.proj.weight");
            let rw = require_weight(&self.weights, &router_weight_name)?;
            eprintln!("[DIAG] router weight: dtype={}, shape={:?}, byte_len={}",
                rw.buffer.dtype(), rw.buffer.shape(), rw.buffer.byte_len());
            if let Some(ref qm) = rw.quant_meta {
                eprintln!("[DIAG]   router quant: bits={}, group_size={}", qm.bits, qm.group_size);
            }
        }

        let intermediate_size = self.config.moe_intermediate_for_layer(layer_idx);

        // ===================================================================
        // Phase 1: GPU MoE routing via moe_gate kernel.
        // Replaces CPU RMS norm + matmul + top-K softmax with a single GPU dispatch.
        // Falls back to CPU routing if GPU routing fails (e.g. unsupported config).
        // ===================================================================
        let use_cpu_route = std::env::var("HF2Q_CPU_MOE_ROUTE").is_ok();
        let (all_expert_ids, all_routing_weights) = if !use_cpu_route {
            match self.gpu_moe_route(layer_idx, router_input, seq_len) {
                Ok((ids, weights)) => {
                    if diag {
                        let top_k = self.config.top_k_experts;
                        eprintln!("[DIAG] MoE L{} GPU routing: {} tokens, top-{}", layer_idx, seq_len, top_k);
                        if !ids.is_empty() {
                            eprintln!("[DIAG] MoE L{} T0 selected experts: {:?}, weights: {:?}",
                                layer_idx, ids[0], weights[0]);
                        }
                        if seq_len > 1 && ids.len() == seq_len {
                            eprintln!("[DIAG] MoE L{} T{} (LAST) selected experts: {:?}, weights: {:?}",
                                layer_idx, seq_len - 1, ids[seq_len - 1], weights[seq_len - 1]);
                        }
                    }
                    (ids, weights)
                }
                Err(e) => {
                    debug!(layer = layer_idx, error = %e, "GPU MoE routing failed, falling back to CPU");
                    self.cpu_moe_route(layer_idx, router_input, seq_len)?
                }
            }
        } else {
            self.cpu_moe_route(layer_idx, router_input, seq_len)?
        };

        // ===================================================================
        // Phase 2: GPU expert dispatch — ALL tokens, ALL experts, ONE encoder
        // Check if GPU path is available, otherwise fall back to CPU per-expert.
        // ===================================================================
        let use_cpu = std::env::var("HF2Q_CPU_QMATMUL").is_ok();
        let gate_name = format!("model.layers.{layer_idx}.experts.switch_glu.gate_proj.weight");
        let gate_w = require_weight(&self.weights, &gate_name)?;
        let gpu_expert = !use_cpu && gate_w.quant_meta.as_ref().map_or(false, |qm| {
            qm.bits == 4 || qm.bits == 8
        });

        let mut output_data = vec![0.0f32; seq_len * hidden_size];

        if gpu_expert {
            let qm = gate_w.quant_meta.as_ref().unwrap();
            let bits = qm.bits as u32;
            let group_size = qm.group_size as u32;

            // Collect all (token, expert) dispatch requests, then batch into ONE encoder
            struct ExpertDispatch {
                token_idx: usize,
                expert_id: usize,
                weight: f32,
            }
            let mut dispatches: Vec<ExpertDispatch> = Vec::new();
            for t in 0..seq_len {
                for (rank, &expert_id) in all_expert_ids[t].iter().enumerate() {
                    let w = all_routing_weights[t][rank];
                    if w.abs() < 1e-10 { continue; }
                    dispatches.push(ExpertDispatch { token_idx: t, expert_id, weight: w });
                }
            }

            // ---------------------------------------------------------------
            // MERGED: Expert dispatch + accumulation + cast in ONE encoder.
            // Metal guarantees ordered execution within a command buffer,
            // so accumulation reads from buffers written by dispatch safely.
            // This eliminates 1 commit_and_wait() per MoE layer.
            // ---------------------------------------------------------------
            let mut encoder = self.device.command_encoder()?;
            let mut gpu_output_bufs: Vec<(usize, f32, MlxBuffer)> = Vec::with_capacity(dispatches.len());

            for d in &dispatches {
                let token_expert = &expert_data[d.token_idx * hidden_size..(d.token_idx + 1) * hidden_size];
                let out_buf = self.expert_ffn_3d_gpu_with_encoder(
                    &mut encoder, layer_idx, d.expert_id, token_expert,
                    hidden_size, intermediate_size, bits, group_size,
                )?;
                gpu_output_bufs.push((d.token_idx, d.weight, out_buf));
            }

            // Continue in the SAME encoder: zero + accumulate + cast
            // (no commit between dispatch and accumulation)

            // Allocate per-token f32 accumulator buffers on GPU
            let mut token_accum_bufs: Vec<MlxBuffer> = Vec::with_capacity(seq_len);
            for _t in 0..seq_len {
                token_accum_bufs.push(self.device.alloc_buffer(
                    hidden_size * std::mem::size_of::<f32>(),
                    DType::F32,
                    vec![1, hidden_size],
                )?);
            }

            // Allocate bf16 output buffer for final result
            let bf16_output = self.device.alloc_buffer(
                seq_len * hidden_size * 2, // bf16 = 2 bytes
                DType::BF16,
                vec![seq_len, hidden_size],
            )?;

            // Zero all per-token accumulators (same encoder)
            for accum_buf in &token_accum_bufs {
                moe_ops::moe_zero_buffer_encode(
                    &mut encoder, &mut self.registry,
                    self.device.metal_device(),
                    accum_buf, hidden_size,
                )?;
            }

            // Weighted accumulation: accum[token] += weight * expert_output
            for (_i, (token_idx, weight, buf)) in gpu_output_bufs.iter().enumerate() {
                moe_ops::moe_accumulate_encode(
                    &mut encoder, &mut self.registry,
                    self.device.metal_device(),
                    &token_accum_bufs[*token_idx], buf,
                    *weight, hidden_size,
                )?;
            }

            // Cast each token's f32 accumulator to bf16 in the output buffer.
            // For single-token decode (seq_len=1), cast directly — fully on GPU.
            // For multi-token prefill, fall back to per-token CPU assembly
            // (rare path, but still avoids per-expert readback).
            if seq_len == 1 {
                elementwise::cast(
                    &mut encoder, &mut self.registry,
                    self.device.metal_device(),
                    &token_accum_bufs[0], &bf16_output,
                    hidden_size,
                    elementwise::CastDirection::F32ToBF16,
                )?;

                encoder.commit_and_wait()?;

                if diag {
                    let diag_data = read_buffer_f32(&token_accum_bufs[0])?;
                    let n = diag_data.len().min(5);
                    eprintln!("[DIAG] MoE L0 final accumulated output first {}: {:?}", n, &diag_data[..n]);
                }

                return Ok(bf16_output);
            } else {
                // Multi-token: commit dispatch+accumulation, read back f32 accumulators
                encoder.commit_and_wait()?;

                for t in 0..seq_len {
                    let token_f32 = read_buffer_f32(&token_accum_bufs[t])?;
                    output_data[t * hidden_size..(t + 1) * hidden_size]
                        .copy_from_slice(&token_f32);
                }

                if diag {
                    let n = output_data.len().min(5);
                    eprintln!("[DIAG] MoE L0 final accumulated output first {}: {:?}", n, &output_data[..n]);
                }

                return f32_vec_to_bf16_buffer(
                    &self.device,
                    &output_data,
                    vec![seq_len, hidden_size],
                );
            }
        } else {
            // CPU fallback path: per-token, per-expert dispatch
            for t in 0..seq_len {
                let token_expert = &expert_data[t * hidden_size..(t + 1) * hidden_size];
                let mut token_output = vec![0.0f32; hidden_size];

                for (rank, &expert_id) in all_expert_ids[t].iter().enumerate() {
                    let weight = all_routing_weights[t][rank];
                    if weight.abs() < 1e-10 { continue; }

                    let expert_output = self.expert_ffn_3d(
                        layer_idx, expert_id, token_expert,
                        hidden_size, intermediate_size,
                    )?;

                    if diag && t == 0 && rank == 0 {
                        let n = expert_output.len().min(5);
                        eprintln!("[DIAG] MoE L0 T0 expert {} output first {}: {:?}",
                            expert_id, n, &expert_output[..n]);
                    }

                    for (j, &val) in expert_output.iter().enumerate() {
                        token_output[j] += weight * val;
                    }
                }

                output_data[t * hidden_size..(t + 1) * hidden_size]
                    .copy_from_slice(&token_output);
            }
        }

        if diag {
            let n = output_data.len().min(5);
            eprintln!("[DIAG] MoE L0 final accumulated output first {}: {:?}", n, &output_data[..n]);
        }

        f32_vec_to_bf16_buffer(
            &self.device,
            &output_data,
            vec![seq_len, hidden_size],
        )
    }

    /// MoE FFN with external encoder (GPU fast path only).
    ///
    /// Does CPU routing (Phase 1) internally, then dispatches expert FFNs +
    /// accumulation into the provided encoder WITHOUT committing.
    /// Returns None if expert weights are not GPU-capable (caller should
    /// fall back to `moe_ffn`).
    fn moe_ffn_gpu_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        layer_idx: usize,
        router_input: &MlxBuffer,
        expert_input: &MlxBuffer,
        seq_len: usize,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let diag = std::env::var("HF2Q_DIAG").is_ok() && layer_idx == 0;

        let intermediate_size = self.config.moe_intermediate_for_layer(layer_idx);

        // Phase 1: GPU MoE routing (falls back to CPU if GPU fails)
        // NOTE: gpu_moe_route creates its own encoder and syncs — this is a
        // known bottleneck (64-byte readback). Future: merge into shared encoder.
        let use_cpu_route = std::env::var("HF2Q_CPU_MOE_ROUTE").is_ok();
        let (all_expert_ids, all_routing_weights) = if !use_cpu_route {
            match self.gpu_moe_route(layer_idx, router_input, seq_len) {
                Ok(result) => result,
                Err(e) => {
                    debug!(layer = layer_idx, error = %e, "GPU MoE routing failed in encoder path, falling back to CPU");
                    self.cpu_moe_route(layer_idx, router_input, seq_len)?
                }
            }
        } else {
            self.cpu_moe_route(layer_idx, router_input, seq_len)?
        };

        if diag && !all_expert_ids.is_empty() {
            eprintln!("[DIAG] MoE L0 T0 selected experts: {:?}, weights: {:?}",
                all_expert_ids[0], all_routing_weights[0]);
        }

        // Phase 2: GPU expert dispatch into external encoder
        let gate_name = format!("model.layers.{layer_idx}.experts.switch_glu.gate_proj.weight");
        let gate_w = require_weight(&self.weights, &gate_name)?;
        let qm = gate_w.quant_meta.as_ref().ok_or_else(|| Gemma4Error::ForwardError {
            reason: "moe_ffn_gpu_with_encoder called but expert weights lack quant_meta".into(),
        })?;
        let bits = qm.bits as u32;
        let group_size = qm.group_size as u32;

        struct ExpertDispatch {
            token_idx: usize,
            expert_id: usize,
            weight: f32,
        }
        let mut dispatches: Vec<ExpertDispatch> = Vec::new();
        for t in 0..seq_len {
            for (rank, &expert_id) in all_expert_ids[t].iter().enumerate() {
                let w = all_routing_weights[t][rank];
                if w.abs() < 1e-10 { continue; }
                dispatches.push(ExpertDispatch { token_idx: t, expert_id, weight: w });
            }
        }

        // For single-token decode, pass the GPU buffer directly to expert FFN
        // instead of reading to CPU first. This eliminates the expensive
        // read_buffer_f32 + f32_vec_to_buffer round-trip per expert.
        let expert_data_cpu;
        let use_gpu_input = seq_len == 1;

        if !use_gpu_input {
            expert_data_cpu = Some(read_buffer_f32(expert_input)?);
        } else {
            expert_data_cpu = None;
        }

        let mut gpu_output_bufs: Vec<(usize, f32, MlxBuffer)> = Vec::with_capacity(dispatches.len());
        for d in &dispatches {
            let out_buf = if use_gpu_input {
                // GPU path: expert_input is already on GPU as bf16, pass directly
                self.expert_ffn_3d_gpu_buf_with_encoder(
                    encoder, layer_idx, d.expert_id, expert_input,
                    d.token_idx, hidden_size, intermediate_size, bits, group_size,
                )?
            } else {
                // CPU path: read from CPU buffer (prefill with multiple tokens)
                let expert_data = expert_data_cpu.as_ref().unwrap();
                let token_expert = &expert_data[d.token_idx * hidden_size..(d.token_idx + 1) * hidden_size];
                self.expert_ffn_3d_gpu_with_encoder(
                    encoder, layer_idx, d.expert_id, token_expert,
                    hidden_size, intermediate_size, bits, group_size,
                )?
            };
            gpu_output_bufs.push((d.token_idx, d.weight, out_buf));
        }

        // Accumulation in same encoder
        let mut token_accum_bufs: Vec<MlxBuffer> = Vec::with_capacity(seq_len);
        for _t in 0..seq_len {
            token_accum_bufs.push(self.device.alloc_buffer(
                hidden_size * std::mem::size_of::<f32>(),
                DType::F32,
                vec![1, hidden_size],
            )?);
        }

        let bf16_output = self.device.alloc_buffer(
            seq_len * hidden_size * 2,
            DType::BF16,
            vec![seq_len, hidden_size],
        )?;

        for accum_buf in &token_accum_bufs {
            moe_ops::moe_zero_buffer_encode(
                encoder, &mut self.registry,
                self.device.metal_device(),
                accum_buf, hidden_size,
            )?;
        }

        for (_i, (token_idx, weight, buf)) in gpu_output_bufs.iter().enumerate() {
            moe_ops::moe_accumulate_encode(
                encoder, &mut self.registry,
                self.device.metal_device(),
                &token_accum_bufs[*token_idx], buf,
                *weight, hidden_size,
            )?;
        }

        if seq_len == 1 {
            elementwise::cast(
                encoder, &mut self.registry,
                self.device.metal_device(),
                &token_accum_bufs[0], &bf16_output,
                hidden_size,
                elementwise::CastDirection::F32ToBF16,
            )?;
            // Caller will commit
            return Ok(bf16_output);
        }

        // Multi-token: need to commit to read back accumulators.
        // This is a rare path (prefill), add a commit here.
        // The caller's encoder is consumed by this commit.
        encoder.commit_and_wait()?;

        let mut output_data = vec![0.0f32; seq_len * hidden_size];
        for t in 0..seq_len {
            let token_f32 = read_buffer_f32(&token_accum_bufs[t])?;
            output_data[t * hidden_size..(t + 1) * hidden_size]
                .copy_from_slice(&token_f32);
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
    ///
    /// When GPU quantized matmul is available (not forced to CPU by
    /// `HF2Q_CPU_QMATMUL`), the expert FFN is dispatched entirely on GPU
    /// using `quantized_matmul_simd` + `gelu` + `elementwise_mul` kernels.
    fn expert_ffn_3d(
        &mut self,
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

        // --- GPU path: quantized matmul directly on packed expert weights ---
        let use_cpu = std::env::var("HF2Q_CPU_QMATMUL").is_ok();
        if !use_cpu {
            let gate_w = require_weight(&self.weights, &format!("{gate_name}.weight"))?;
            if let Some(ref qm) = gate_w.quant_meta {
                let bits = qm.bits as u32;
                let group_size = qm.group_size as u32;
                if bits == 4 || bits == 6 || bits == 8 {
                    return self.expert_ffn_3d_gpu(
                        layer_idx, expert_id, token_hidden,
                        hidden_size, intermediate_size,
                        bits, group_size, diag,
                    );
                }
            }
        }

        // --- CPU fallback path ---
        // gate_proj: [n_experts, intermediate_size, packed_k]
        let gate_data = self.dequantize_expert_slice(
            &gate_name, expert_id, intermediate_size, hidden_size,
        )?;
        let up_data = self.dequantize_expert_slice(
            &up_name, expert_id, intermediate_size, hidden_size,
        )?;
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
        }

        // Round weights and input to bf16 to match MLX's precision
        let gate_bf16 = round_to_bf16(&gate_data);
        let up_bf16 = round_to_bf16(&up_data);
        let down_bf16 = round_to_bf16(&down_data);
        let input_bf16 = round_to_bf16(token_hidden);

        let mut gate_out = vec![0.0f32; intermediate_size];
        for i in 0..intermediate_size {
            let mut sum = 0.0f32;
            for k in 0..hidden_size {
                sum += gate_bf16[i * hidden_size + k] * input_bf16[k];
            }
            gate_out[i] = sum;
        }

        let mut up_out = vec![0.0f32; intermediate_size];
        for i in 0..intermediate_size {
            let mut sum = 0.0f32;
            for k in 0..hidden_size {
                sum += up_bf16[i * hidden_size + k] * input_bf16[k];
            }
            up_out[i] = sum;
        }

        let mut hidden = vec![0.0f32; intermediate_size];
        for i in 0..intermediate_size {
            hidden[i] = gelu_pytorch_tanh(gate_out[i]) * up_out[i];
        }

        let hidden_bf16 = round_to_bf16(&hidden);

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
        }

        Ok(expert_out)
    }

    /// GPU path for expert FFN: dispatch quantized matmul + GELU + elementwise_mul
    /// on Metal, avoiding CPU dequantization entirely.
    ///
    /// For each projection (gate, up, down), we:
    /// 1. Extract the expert's sub-slice from the 3D packed weight tensor
    /// 2. Copy the sub-slice into a GPU buffer
    /// 3. Dispatch `quantized_matmul_simd` on GPU
    ///
    /// Then GELU and elementwise_mul are dispatched on GPU as well.
    /// All ops are batched into a single command encoder.
    #[allow(clippy::too_many_arguments)]
    fn expert_ffn_3d_gpu(
        &mut self,
        layer_idx: usize,
        expert_id: usize,
        token_hidden: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        bits: u32,
        group_size: u32,
        diag: bool,
    ) -> Result<Vec<f32>, Gemma4Error> {
        let mut encoder = self.device.command_encoder()?;
        let down_out = self.expert_ffn_3d_gpu_with_encoder(
            &mut encoder, layer_idx, expert_id, token_hidden,
            hidden_size, intermediate_size, bits, group_size,
        )?;
        encoder.commit_and_wait()?;
        let result = read_buffer_f32(&down_out)?;

        if diag {
            let n = result.len().min(5);
            eprintln!("[DIAG] expert_ffn_3d_gpu L0 E{}: output first {}: {:?}",
                expert_id, n, &result[..n]);
        }

        Ok(result)
    }

    /// GPU expert FFN using an external command encoder (batched variant).
    ///
    /// Dispatches gate_proj, up_proj, GELU, mul, down_proj into the caller's
    /// encoder. Returns the f32 output buffer on GPU — caller must commit and
    /// read back when ready.
    #[allow(clippy::too_many_arguments)]
    fn expert_ffn_3d_gpu_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        layer_idx: usize,
        expert_id: usize,
        token_hidden: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        bits: u32,
        group_size: u32,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let base = format!("model.layers.{layer_idx}.experts.switch_glu");

        // Upload token input as f32 buffer [1, hidden_size]
        let input_buf = f32_vec_to_buffer(
            &self.device, token_hidden, vec![1, hidden_size],
        )?;

        // Extract expert sub-buffers for gate_proj, up_proj, down_proj
        let gate_bufs = self.extract_expert_quant_buffers(
            &format!("{base}.gate_proj"), expert_id,
            intermediate_size, hidden_size,
        )?;
        let up_bufs = self.extract_expert_quant_buffers(
            &format!("{base}.up_proj"), expert_id,
            intermediate_size, hidden_size,
        )?;
        let down_bufs = self.extract_expert_quant_buffers(
            &format!("{base}.down_proj"), expert_id,
            hidden_size, intermediate_size,
        )?;

        // gate_out = gate_proj @ input  [1, intermediate_size]
        let gate_params = QuantizedMatmulParams {
            m: 1,
            k: hidden_size as u32,
            n: intermediate_size as u32,
            group_size,
            bits,
        };
        let gate_out = qmatmul::quantized_matmul_simd(
            encoder, &mut self.registry, &self.device,
            &input_buf, &gate_bufs.0, &gate_bufs.1, &gate_bufs.2,
            &gate_params,
        )?;

        // up_out = up_proj @ input  [1, intermediate_size]
        let up_out = qmatmul::quantized_matmul_simd(
            encoder, &mut self.registry, &self.device,
            &input_buf, &up_bufs.0, &up_bufs.1, &up_bufs.2,
            &gate_params, // same dimensions as gate
        )?;

        // GELU on gate_out (f32 -> f32) — use pooled buffer
        self.moe_buffer_pool.ensure_allocated(
            &self.device, intermediate_size, hidden_size,
        )?;
        let pool_gelu_out = self.moe_buffer_pool.gelu_out();
        gelu::dispatch_gelu(
            encoder, &mut self.registry,
            self.device.metal_device(),
            &gate_out, pool_gelu_out,
        )?;

        // hidden = gelu_out * up_out (f32 elementwise mul) — use pooled buffer
        let pool_hidden_buf = self.moe_buffer_pool.hidden_buf();
        elementwise::elementwise_mul(
            encoder, &mut self.registry,
            self.device.metal_device(),
            pool_gelu_out, &up_out, pool_hidden_buf,
            intermediate_size, DType::F32,
        )?;

        // down_out = down_proj @ hidden  [1, hidden_size]
        let down_params = QuantizedMatmulParams {
            m: 1,
            k: intermediate_size as u32,
            n: hidden_size as u32,
            group_size,
            bits,
        };
        let down_out = qmatmul::quantized_matmul_simd(
            encoder, &mut self.registry, &self.device,
            pool_hidden_buf, &down_bufs.0, &down_bufs.1, &down_bufs.2,
            &down_params,
        )?;

        Ok(down_out)
    }

    /// Expert FFN that reads input directly from a GPU bf16 buffer.
    ///
    /// Same computation as `expert_ffn_3d_gpu_with_encoder` but avoids the
    /// CPU round-trip by casting bf16→f32 on GPU within the same encoder.
    /// Used for single-token decode where `expert_input` is [1, hidden_size] bf16.
    #[allow(clippy::too_many_arguments)]
    fn expert_ffn_3d_gpu_buf_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        layer_idx: usize,
        expert_id: usize,
        expert_input_bf16: &MlxBuffer,
        token_idx: usize,
        hidden_size: usize,
        intermediate_size: usize,
        bits: u32,
        group_size: u32,
    ) -> Result<MlxBuffer, Gemma4Error> {
        let base = format!("model.layers.{layer_idx}.experts.switch_glu");

        // Cast bf16 input to f32 on GPU (no CPU round-trip).
        // For seq_len=1, token_idx is always 0 and expert_input is [1, hidden_size] bf16.
        // For multi-token, we'd need to extract the token slice — but this path
        // is only called for seq_len=1.
        let _ = token_idx; // always 0 for seq_len=1
        let n_elems = hidden_size;
        let input_f32 = self.device.alloc_buffer(
            n_elems * std::mem::size_of::<f32>(),
            DType::F32,
            vec![1, hidden_size],
        )?;
        elementwise::cast(
            encoder, &mut self.registry, self.device.metal_device(),
            expert_input_bf16, &input_f32, n_elems,
            elementwise::CastDirection::BF16ToF32,
        )?;

        // From here, same as expert_ffn_3d_gpu_with_encoder but using input_f32
        let gate_bufs = self.extract_expert_quant_buffers(
            &format!("{base}.gate_proj"), expert_id,
            intermediate_size, hidden_size,
        )?;
        let up_bufs = self.extract_expert_quant_buffers(
            &format!("{base}.up_proj"), expert_id,
            intermediate_size, hidden_size,
        )?;
        let down_bufs = self.extract_expert_quant_buffers(
            &format!("{base}.down_proj"), expert_id,
            hidden_size, intermediate_size,
        )?;

        let gate_params = QuantizedMatmulParams {
            m: 1, k: hidden_size as u32, n: intermediate_size as u32, group_size, bits,
        };
        let gate_out = qmatmul::quantized_matmul_simd(
            encoder, &mut self.registry, &self.device,
            &input_f32, &gate_bufs.0, &gate_bufs.1, &gate_bufs.2, &gate_params,
        )?;
        let up_out = qmatmul::quantized_matmul_simd(
            encoder, &mut self.registry, &self.device,
            &input_f32, &up_bufs.0, &up_bufs.1, &up_bufs.2, &gate_params,
        )?;

        self.moe_buffer_pool.ensure_allocated(&self.device, intermediate_size, hidden_size)?;
        let pool_gelu_out = self.moe_buffer_pool.gelu_out();
        gelu::dispatch_gelu(encoder, &mut self.registry, self.device.metal_device(), &gate_out, pool_gelu_out)?;

        let pool_hidden_buf = self.moe_buffer_pool.hidden_buf();
        elementwise::elementwise_mul(
            encoder, &mut self.registry, self.device.metal_device(),
            pool_gelu_out, &up_out, pool_hidden_buf, intermediate_size, DType::F32,
        )?;

        let down_params = QuantizedMatmulParams {
            m: 1, k: intermediate_size as u32, n: hidden_size as u32, group_size, bits,
        };
        let down_out = qmatmul::quantized_matmul_simd(
            encoder, &mut self.registry, &self.device,
            pool_hidden_buf, &down_bufs.0, &down_bufs.1, &down_bufs.2, &down_params,
        )?;

        Ok(down_out)
    }

    /// Extract an expert's quantized sub-buffers from a 3D packed weight tensor.
    ///
    /// Returns (weight_buf, scales_buf, biases_buf) for the given expert_id.
    /// The weight buffer is u32 packed, scales and biases match the original dtype.
    fn extract_expert_quant_buffers(
        &self,
        weight_base: &str,
        expert_id: usize,
        rows: usize,
        _cols: usize,
    ) -> Result<(MlxBuffer, MlxBuffer, MlxBuffer), Gemma4Error> {
        let weight_name = format!("{weight_base}.weight");
        let scales_name = format!("{weight_base}.scales");
        let biases_name = format!("{weight_base}.biases");

        let loaded = require_weight(&self.weights, &weight_name)?;
        let scales_w = require_weight(&self.weights, &scales_name)?;
        let biases_w = require_weight(&self.weights, &biases_name)?;

        let shape = loaded.buffer.shape();
        let scales_shape = scales_w.buffer.shape();

        // Determine packed_cols and n_groups from the actual tensor shape
        let packed_cols = if shape.len() == 3 { shape[2] } else { shape[1] };
        let n_groups = if scales_shape.len() == 3 { scales_shape[2] } else { scales_shape[1] };

        // --- Extract expert's packed weight sub-slice (u32) ---
        let weight_expert_stride = rows * packed_cols;
        let w_start = expert_id * weight_expert_stride;
        let packed_u32: &[u32] = loaded.buffer.as_slice().map_err(|e| {
            Gemma4Error::ForwardError { reason: format!("read 3D packed weight u32: {e}") }
        })?;
        let expert_packed = &packed_u32[w_start..w_start + weight_expert_stride];

        // Create GPU buffer for expert's packed weights
        let w_byte_len = weight_expert_stride * std::mem::size_of::<u32>();
        let mut w_buf = self.device.alloc_buffer(
            w_byte_len, DType::U32, vec![rows, packed_cols],
        )?;
        {
            let dst: &mut [u32] = w_buf.as_mut_slice().map_err(|e| {
                Gemma4Error::ForwardError { reason: format!("write expert weight buf: {e}") }
            })?;
            dst.copy_from_slice(expert_packed);
        }

        // --- Extract expert's scales sub-slice ---
        let scales_expert_stride = rows * n_groups;
        let s_start = expert_id * scales_expert_stride;
        let scales_dtype = scales_w.buffer.dtype();
        let scales_elem_size = scales_dtype.size_of();
        let s_byte_len = scales_expert_stride * scales_elem_size;

        let s_buf = self.device.alloc_buffer(
            s_byte_len, scales_dtype, vec![rows, n_groups],
        )?;
        {
            let src_ptr = scales_w.buffer.contents_ptr() as *const u8;
            let dst_ptr = s_buf.contents_ptr() as *mut u8;
            let src_offset = s_start * scales_elem_size;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr.add(src_offset),
                    dst_ptr,
                    s_byte_len,
                );
            }
        }

        // --- Extract expert's biases sub-slice ---
        let biases_dtype = biases_w.buffer.dtype();
        let biases_elem_size = biases_dtype.size_of();
        let b_byte_len = scales_expert_stride * biases_elem_size;

        let b_buf = self.device.alloc_buffer(
            b_byte_len, biases_dtype, vec![rows, n_groups],
        )?;
        {
            let src_ptr = biases_w.buffer.contents_ptr() as *const u8;
            let dst_ptr = b_buf.contents_ptr() as *mut u8;
            let src_offset = s_start * biases_elem_size;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr.add(src_offset),
                    dst_ptr,
                    b_byte_len,
                );
            }
        }

        Ok((w_buf, s_buf, b_buf))
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

        // Merge gate_proj + up_proj + GELU*up + down_proj into minimal commits.
        // When all weights use GPU qmatmul: gate+up+GELU*up in ONE encoder,
        // down_proj in a second encoder. Total: 2 commits instead of 3.
        let gate_base = format!("model.layers.{layer_idx}.mlp.gate_proj");
        let up_base = format!("model.layers.{layer_idx}.mlp.up_proj");
        let down_base = format!("model.layers.{layer_idx}.mlp.down_proj");
        let n_hidden = seq_len * intermediate_size;

        let all_ffn_gpu = weight_uses_gpu_qmatmul(&self.weights, &gate_base)
            && weight_uses_gpu_qmatmul(&self.weights, &up_base);

        let hidden_buf;
        if all_ffn_gpu {
            // Fast path: gate+up projections + GELU*up all in ONE encoder
            let mut encoder = self.device.command_encoder()?;
            let gate = self.quantized_projection_with_encoder(
                &mut encoder, input, &gate_base, seq_len, hidden_size, intermediate_size,
            )?;
            let up = self.quantized_projection_with_encoder(
                &mut encoder, input, &up_base, seq_len, hidden_size, intermediate_size,
            )?;

            // Cast gate and up from bf16 to f32, GELU, mul, cast back — same encoder
            let gate_f32 = self.cast_bf16_to_f32_with_encoder(&mut encoder, &gate, n_hidden)?;
            let up_f32 = self.cast_bf16_to_f32_with_encoder(&mut encoder, &up, n_hidden)?;

            let gelu_out = self.device.alloc_buffer(
                n_hidden * 4, DType::F32, vec![seq_len, intermediate_size],
            )?;
            gelu::dispatch_gelu(
                &mut encoder, &mut self.registry, self.device.metal_device(),
                &gate_f32, &gelu_out,
            )?;

            let mul_out = self.device.alloc_buffer(
                n_hidden * 4, DType::F32, vec![seq_len, intermediate_size],
            )?;
            elementwise::elementwise_mul(
                &mut encoder, &mut self.registry, self.device.metal_device(),
                &gelu_out, &up_f32, &mul_out, n_hidden, DType::F32,
            )?;

            hidden_buf = self.cast_f32_to_bf16_with_encoder(&mut encoder, &mul_out, n_hidden)?;
            encoder.commit_and_wait()?;
        } else {
            // Slow path: separate encoders for CPU fallback projections
            let (gate, up);
            {
                let mut enc = self.device.command_encoder()?;
                gate = self.quantized_projection_with_encoder(
                    &mut enc, input, &gate_base, seq_len, hidden_size, intermediate_size,
                )?;
                if weight_uses_gpu_qmatmul(&self.weights, &gate_base) { enc.commit_and_wait()?; }
            }
            {
                let mut enc = self.device.command_encoder()?;
                up = self.quantized_projection_with_encoder(
                    &mut enc, input, &up_base, seq_len, hidden_size, intermediate_size,
                )?;
                if weight_uses_gpu_qmatmul(&self.weights, &up_base) { enc.commit_and_wait()?; }
            }

            let mut encoder = self.device.command_encoder()?;
            let gate_f32 = self.cast_bf16_to_f32_with_encoder(&mut encoder, &gate, n_hidden)?;
            let up_f32 = self.cast_bf16_to_f32_with_encoder(&mut encoder, &up, n_hidden)?;

            let gelu_out = self.device.alloc_buffer(
                n_hidden * 4, DType::F32, vec![seq_len, intermediate_size],
            )?;
            gelu::dispatch_gelu(
                &mut encoder, &mut self.registry, self.device.metal_device(),
                &gate_f32, &gelu_out,
            )?;

            let mul_out = self.device.alloc_buffer(
                n_hidden * 4, DType::F32, vec![seq_len, intermediate_size],
            )?;
            elementwise::elementwise_mul(
                &mut encoder, &mut self.registry, self.device.metal_device(),
                &gelu_out, &up_f32, &mul_out, n_hidden, DType::F32,
            )?;

            hidden_buf = self.cast_f32_to_bf16_with_encoder(&mut encoder, &mul_out, n_hidden)?;
            encoder.commit_and_wait()?;
        }

        if diag {
            let hidden_data = read_buffer_f32(&hidden_buf)?;
            let n = hidden_data.len().min(5);
            eprintln!("[DIAG] dense_ffn L0: GELU*up first {}: {:?}", n, &hidden_data[..n]);
        }

        // down_proj (bf16 in, bf16 out)
        let result;
        {
            let mut encoder = self.device.command_encoder()?;
            result = self.quantized_projection_with_encoder(
                &mut encoder, &hidden_buf, &down_base,
                seq_len, intermediate_size, hidden_size,
            )?;
            if weight_uses_gpu_qmatmul(&self.weights, &down_base) {
                encoder.commit_and_wait()?;
            }
        }

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

    /// Dense FFN with external encoder (GPU fast path only).
    ///
    /// Encodes gate+up+GELU*up+down into an external command encoder WITHOUT
    /// committing. Returns None if any projection is not GPU-capable (caller
    /// should fall back to `dense_ffn` which manages its own encoders).
    fn dense_ffn_with_encoder(
        &mut self,
        encoder: &mut CommandEncoder,
        layer_idx: usize,
        input: &MlxBuffer,
        seq_len: usize,
    ) -> Result<Option<MlxBuffer>, Gemma4Error> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        let gate_base = format!("model.layers.{layer_idx}.mlp.gate_proj");
        let up_base = format!("model.layers.{layer_idx}.mlp.up_proj");
        let down_base = format!("model.layers.{layer_idx}.mlp.down_proj");
        let n_hidden = seq_len * intermediate_size;

        let all_ffn_gpu = weight_uses_gpu_qmatmul(&self.weights, &gate_base)
            && weight_uses_gpu_qmatmul(&self.weights, &up_base)
            && weight_uses_gpu_qmatmul(&self.weights, &down_base);

        if !all_ffn_gpu {
            return Ok(None);
        }

        // gate+up projections
        let gate = self.quantized_projection_with_encoder(
            encoder, input, &gate_base, seq_len, hidden_size, intermediate_size,
        )?;
        let up = self.quantized_projection_with_encoder(
            encoder, input, &up_base, seq_len, hidden_size, intermediate_size,
        )?;

        // Cast gate and up from bf16 to f32, GELU, mul, cast back — same encoder
        let gate_f32 = self.cast_bf16_to_f32_with_encoder(encoder, &gate, n_hidden)?;
        let up_f32 = self.cast_bf16_to_f32_with_encoder(encoder, &up, n_hidden)?;

        let gelu_out = self.device.alloc_buffer(
            n_hidden * 4, DType::F32, vec![seq_len, intermediate_size],
        )?;
        gelu::dispatch_gelu(
            encoder, &mut self.registry, self.device.metal_device(),
            &gate_f32, &gelu_out,
        )?;

        let mul_out = self.device.alloc_buffer(
            n_hidden * 4, DType::F32, vec![seq_len, intermediate_size],
        )?;
        elementwise::elementwise_mul(
            encoder, &mut self.registry, self.device.metal_device(),
            &gelu_out, &up_f32, &mul_out, n_hidden, DType::F32,
        )?;

        let hidden_buf = self.cast_f32_to_bf16_with_encoder(encoder, &mul_out, n_hidden)?;

        // down_proj in the same encoder
        let result = self.quantized_projection_with_encoder(
            encoder, &hidden_buf, &down_base,
            seq_len, intermediate_size, hidden_size,
        )?;

        Ok(Some(result))
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

        // Create params buffer with [cap, n_elements_as_bits].
        // The kernel reads params[1] via as_type<uint> to get n_elements
        // for bounds checking, avoiding out-of-bounds access when
        // threadgroup_count * threadgroup_size > n_elements.
        let mut params_buf = self.device.alloc_buffer(
            2 * std::mem::size_of::<f32>(),
            DType::F32,
            vec![2],
        )?;
        {
            let slice: &mut [f32] = params_buf.as_mut_slice().map_err(|e| {
                Gemma4Error::ForwardError {
                    reason: format!("Softcap params write: {e}"),
                }
            })?;
            slice[0] = cap;
            slice[1] = f32::from_bits(n_elements as u32);
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

    /// Apply softcap on GPU, committing the command buffer non-blocking.
    ///
    /// Returns `(CommandEncoder, MlxBuffer)` where the encoder has been
    /// committed but NOT waited on.  The caller must call
    /// `encoder.wait_until_completed()` before reading the output buffer.
    fn apply_softcap_start(
        &mut self,
        logits: &MlxBuffer,
        seq_len: usize,
    ) -> Result<(CommandEncoder, MlxBuffer), Gemma4Error> {
        let vocab_size = self.config.vocab_size;
        let cap = self.config.final_logit_softcapping;
        let n_elements = seq_len * vocab_size;

        let output_bytes = n_elements * std::mem::size_of::<f32>();
        let output = self.device.alloc_buffer(
            output_bytes,
            DType::F32,
            vec![seq_len, vocab_size],
        )?;

        let mut params_buf = self.device.alloc_buffer(
            2 * std::mem::size_of::<f32>(),
            DType::F32,
            vec![2],
        )?;
        {
            let slice: &mut [f32] = params_buf.as_mut_slice().map_err(|e| {
                Gemma4Error::ForwardError {
                    reason: format!("Softcap params write: {e}"),
                }
            })?;
            slice[0] = cap;
            slice[1] = f32::from_bits(n_elements as u32);
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
        // Non-blocking commit — GPU begins executing immediately but CPU continues.
        encoder.commit();

        Ok((encoder, output))
    }

    /// Begin an async forward pass: encode all GPU work and commit
    /// non-blocking.
    ///
    /// This performs the full forward pass (embedding, transformer layers,
    /// final norm, lm_head projection, softcap) but the **last** GPU command
    /// buffer commit is non-blocking.  The GPU executes asynchronously while
    /// the CPU can do other work (e.g. sampling the previous token).
    ///
    /// Call [`forward_wait`] on the returned [`ForwardHandle`] to block until
    /// the GPU finishes and read back the logits.
    ///
    /// # Errors
    ///
    /// Returns `Gemma4Error` if any step of the forward pass fails to encode
    /// or if a synchronous mid-pass commit fails (per-layer commits remain
    /// synchronous for correctness).
    pub fn forward_start(&mut self, tokens: &[u32]) -> Result<ForwardHandle, Gemma4Error> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(Gemma4Error::ForwardError {
                reason: "Empty token sequence".into(),
            });
        }

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        // Step 1: Embedding gather (returns f32), then convert to bf16
        let hidden_f32 = self.embedding_gather(tokens)?;
        let mut hidden = self.cast_f32_to_bf16(&hidden_f32, seq_len * hidden_size)?;

        // Step 2: Scale embeddings by sqrt(hidden_size) (Gemma convention)
        let scale = half::bf16::from_f32((hidden_size as f32).sqrt()).to_f32();
        let n_embed_elements = seq_len * hidden_size;
        let scaled_embed = self.device.alloc_buffer(
            n_embed_elements * 2, // bf16
            DType::BF16,
            vec![seq_len, hidden_size],
        )?;
        {
            let mut encoder = self.device.command_encoder()?;
            elementwise::scalar_mul_bf16(
                &mut encoder,
                &mut self.registry,
                self.device.metal_device(),
                &hidden,
                &scaled_embed,
                n_embed_elements,
                scale,
            )?;
            encoder.commit_and_wait()?;
        }
        hidden = scaled_embed;

        // Step 3: Process each transformer layer (synchronous per-layer commits)
        for layer_idx in 0..self.config.num_layers {
            hidden = self.forward_layer(layer_idx, &hidden, seq_len)?;
        }

        // Step 4: Final RMS norm
        let normed = self.apply_rms_norm(&hidden, "model.norm.weight", seq_len)?;

        // Step 5: lm_head projection (tied embeddings)
        let logits = self.lm_head_projection(&normed, seq_len)?;

        // Step 6: Softcap — non-blocking commit
        let (encoder, capped_logits) = self.apply_softcap_start(&logits, seq_len)?;

        Ok(ForwardHandle {
            encoder,
            logits_buffer: capped_logits,
            vocab_size,
            seq_len,
        })
    }

    /// Wait for an in-flight forward pass to complete and read back logits.
    ///
    /// Blocks until the GPU finishes executing the command buffer that was
    /// committed by [`forward_start`], then copies the logits from the GPU
    /// buffer to a CPU `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns `Gemma4Error` if the GPU reports an error or if the buffer
    /// readback fails.
    pub fn forward_wait(&mut self, handle: ForwardHandle) -> Result<Vec<f32>, Gemma4Error> {
        // Block until GPU finishes the softcap command buffer.
        handle.encoder.wait_until_completed().map_err(|e| {
            Gemma4Error::ForwardError {
                reason: format!("GPU async forward wait failed: {e}"),
            }
        })?;

        // Read logits from GPU buffer to CPU.
        let output: &[f32] = handle.logits_buffer.as_slice().map_err(|e| {
            Gemma4Error::ForwardError {
                reason: format!("Read async logits: {e}"),
            }
        })?;

        Ok(output.to_vec())
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

/// Create a bf16 Metal buffer from u16 (bf16 bits) data directly.
///
/// No conversion needed — data is already bf16 bits.
fn u16_vec_to_bf16_buffer(
    device: &MlxDevice,
    data: &[u16],
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

/// Reshape bf16 (u16) data from [seq_len, n_heads * head_dim] to [n_heads, seq_len, head_dim].
///
/// Same transpose as `reshape_qkv_for_sdpa` but operates on raw bf16 bits (u16)
/// to avoid bf16↔f32 conversion overhead on the KV cache hot path.
fn reshape_qkv_for_sdpa_bf16(
    data: &[u16],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<u16> {
    let total_dim = n_heads * head_dim;
    let mut out = vec![0u16; n_heads * seq_len * head_dim];

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

/// Check whether a weight at `weight_base` (e.g. `model.layers.0.self_attn.v_proj`)
/// would take the GPU quantized matmul path in `quantized_projection_with_encoder`.
///
/// Returns `true` if the weight is 4-bit or 8-bit quantized and the CPU override
/// is not set. Returns `false` if the weight would need the CPU fallback (6-bit,
/// non-quantized, or CPU forced).
fn weight_uses_gpu_qmatmul(weights: &WeightMap, weight_base: &str) -> bool {
    let weight_name = format!("{weight_base}.weight");
    let loaded = match weights.get(&weight_name) {
        Some(w) => w,
        None => {
            let prefixed = format!("language_model.{}", weight_name);
            match weights.get(&prefixed) {
                Some(w) => w,
                None => return false,
            }
        }
    };
    if std::env::var("HF2Q_CPU_QMATMUL").is_ok() {
        return false;
    }
    loaded.quant_meta.as_ref()
        .map(|qm| qm.bits == 4 || qm.bits == 6 || qm.bits == 8)
        .unwrap_or(false)
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

        // No KV sharing for 12B config (num_kv_shared_layers not set)
        assert_eq!(config.num_kv_shared_layers, 0);
        assert!(!config.is_kv_shared_layer(0));
        assert!(!config.is_kv_shared_layer(29));
        assert_eq!(config.donor_layer_for(0), None);
        assert_eq!(config.moe_intermediate_for_layer(0), 704);
    }

    #[test]
    fn test_kv_sharing_27b() {
        // Simulate a 27B config with 30 layers and 20 KV-shared layers
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
                    "num_kv_shared_layers": 20,
                    "layer_types": [
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention"
                    ]
                }
            }"#,
        )
        .unwrap();

        let config = Gemma4Config::from_model_config(&json).unwrap();
        assert_eq!(config.num_kv_shared_layers, 20);

        // First 10 layers (30 - 20 = 10) are NOT shared
        for i in 0..10 {
            assert!(!config.is_kv_shared_layer(i), "layer {i} should NOT be shared");
        }
        // Last 20 layers (10..30) ARE shared
        for i in 10..30 {
            assert!(config.is_kv_shared_layer(i), "layer {i} should be shared");
        }

        // Donor layer tests:
        // Layer 10 is sliding, first shared. Donor should be the most recent
        // non-shared sliding layer before it (searching 0..10).
        // Layer 9 is sliding -> donor = 9
        assert_eq!(config.donor_layer_for(10), Some(9));

        // Layer 11 is full_attention (global). Donor should be the most recent
        // non-shared global layer. In the pattern, layers 5 are global in 0..10.
        assert_eq!(config.donor_layer_for(11), Some(5));

        // Layer 0 is not shared -> no donor
        assert_eq!(config.donor_layer_for(0), None);

        // Double-wide MoE for shared layers
        assert_eq!(config.moe_intermediate_size_shared, 1408); // 704 * 2
        assert_eq!(config.moe_intermediate_for_layer(0), 704);  // non-shared
        assert_eq!(config.moe_intermediate_for_layer(10), 1408); // shared
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
