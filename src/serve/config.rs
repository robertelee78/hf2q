//! Gemma4 A4B model configuration — parsed from config.json.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Layer attention type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    Sliding,
    Full,
}

/// Configuration for Gemma 4 A4B (MoE variant).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_global_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta_sliding: f64,
    pub rope_theta_global: f64,
    pub partial_rotary_factor_global: f64,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub final_logit_softcapping: Option<f64>,
    pub attention_bias: bool,
    pub attention_k_eq_v: bool,
    pub tie_word_embeddings: bool,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub layer_types: Vec<LayerType>,
}

impl Gemma4Config {
    pub fn from_config_json(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read {}", path.display()))?;
        let raw: RawConfig = serde_json::from_str(&content)
            .with_context(|| format!("Cannot parse {}", path.display()))?;
        let tc = raw.text_config.context("Missing text_config in config.json")?;

        let layer_types: Vec<LayerType> = tc.layer_types
            .unwrap_or_default()
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" => LayerType::Full,
                _ => LayerType::Sliding,
            })
            .collect();

        let num_layers = tc.num_hidden_layers.unwrap_or(30);
        let layer_types = if layer_types.is_empty() {
            // Default: every 6th layer is full attention
            (0..num_layers)
                .map(|i| if (i + 1) % 6 == 0 { LayerType::Full } else { LayerType::Sliding })
                .collect()
        } else {
            layer_types
        };

        let rope = tc.rope_parameters.unwrap_or_default();
        let sliding_rope = rope.sliding_attention.unwrap_or_default();
        let full_rope = rope.full_attention.unwrap_or_default();

        Ok(Self {
            vocab_size: tc.vocab_size.unwrap_or(262144),
            hidden_size: tc.hidden_size.unwrap_or(2816),
            intermediate_size: tc.intermediate_size.unwrap_or(2112),
            moe_intermediate_size: tc.moe_intermediate_size.unwrap_or(704),
            num_hidden_layers: num_layers,
            num_attention_heads: tc.num_attention_heads.unwrap_or(16),
            num_key_value_heads: tc.num_key_value_heads.unwrap_or(8),
            num_global_key_value_heads: tc.num_global_key_value_heads.unwrap_or(2),
            head_dim: tc.head_dim.unwrap_or(256),
            global_head_dim: tc.global_head_dim.unwrap_or(512),
            rms_norm_eps: tc.rms_norm_eps.unwrap_or(1e-6),
            rope_theta_sliding: sliding_rope.rope_theta.unwrap_or(10000.0),
            rope_theta_global: full_rope.rope_theta.unwrap_or(1000000.0),
            partial_rotary_factor_global: full_rope.partial_rotary_factor.unwrap_or(0.25),
            sliding_window: tc.sliding_window.unwrap_or(1024),
            max_position_embeddings: tc.max_position_embeddings.unwrap_or(262144),
            final_logit_softcapping: tc.final_logit_softcapping,
            attention_bias: tc.attention_bias.unwrap_or(false),
            attention_k_eq_v: tc.attention_k_eq_v.unwrap_or(true),
            tie_word_embeddings: tc.tie_word_embeddings.unwrap_or(true),
            num_experts: tc.num_experts.unwrap_or(128),
            top_k_experts: tc.top_k_experts.unwrap_or(8),
            layer_types,
        })
    }

    /// Is layer `idx` a global (full) attention layer?
    pub fn is_full_attention(&self, idx: usize) -> bool {
        self.layer_types.get(idx).copied() == Some(LayerType::Full)
    }

    /// Head dim for the given layer.
    pub fn head_dim_for_layer(&self, idx: usize) -> usize {
        if self.is_full_attention(idx) { self.global_head_dim } else { self.head_dim }
    }

    /// Number of KV heads for the given layer.
    pub fn num_kv_heads_for_layer(&self, idx: usize) -> usize {
        if self.is_full_attention(idx) { self.num_global_key_value_heads } else { self.num_key_value_heads }
    }
}

// --- Raw serde types ---

#[derive(Debug, Deserialize)]
struct RawConfig {
    text_config: Option<TextConfig>,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    moe_intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    num_global_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    global_head_dim: Option<usize>,
    rms_norm_eps: Option<f64>,
    sliding_window: Option<usize>,
    max_position_embeddings: Option<usize>,
    final_logit_softcapping: Option<f64>,
    attention_bias: Option<bool>,
    attention_k_eq_v: Option<bool>,
    tie_word_embeddings: Option<bool>,
    num_experts: Option<usize>,
    top_k_experts: Option<usize>,
    layer_types: Option<Vec<String>>,
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Deserialize, Default)]
struct RopeParameters {
    sliding_attention: Option<RopeEntry>,
    full_attention: Option<RopeEntry>,
}

#[derive(Debug, Deserialize, Default)]
struct RopeEntry {
    rope_theta: Option<f64>,
    partial_rotary_factor: Option<f64>,
}
