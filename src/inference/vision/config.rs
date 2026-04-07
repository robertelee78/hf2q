//! Vision encoder configuration parsed from the model's `config.json`.
//!
//! Corresponds to the `vision_config` section of the Gemma 4 model config.

/// Parsed vision encoder configuration.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Hidden size of the vision encoder (1152 for Gemma 4).
    pub hidden_size: usize,
    /// Number of transformer layers (27 for Gemma 4).
    pub num_hidden_layers: usize,
    /// Number of attention heads (16 for Gemma 4).
    pub num_attention_heads: usize,
    /// Number of key/value heads (16 for Gemma 4, MHA not GQA).
    pub num_key_value_heads: usize,
    /// Intermediate size for MLP (4304 for Gemma 4).
    pub intermediate_size: usize,
    /// Patch size in pixels (16 for Gemma 4).
    pub patch_size: usize,
    /// Attention head dimension (72 for Gemma 4).
    pub head_dim: usize,
    /// RMS norm epsilon (1e-6).
    pub rms_norm_eps: f32,
    /// RoPE theta for vision encoder (100.0).
    pub rope_theta: f32,
    /// Spatial pooling kernel size (3 for Gemma 4).
    pub pooling_kernel_size: usize,
    /// Maximum position embedding table size (10240).
    pub position_embedding_size: usize,
    /// Default number of output soft tokens per image (280).
    pub default_output_length: usize,
    /// Whether to apply standardization (bias/scale) after pooling.
    pub standardize: bool,
    /// Hidden size of the text model (for projection target).
    pub text_hidden_size: usize,
    /// Image token ID in the tokenizer vocabulary.
    pub image_token_id: u32,
    /// Begin-of-image token ID.
    pub boi_token_id: u32,
    /// End-of-image token ID.
    pub eoi_token_id: u32,
    /// Number of vision soft tokens per image.
    pub vision_soft_tokens_per_image: usize,
}

impl VisionConfig {
    /// Parse vision config from the raw model config JSON.
    ///
    /// Reads from `vision_config` sub-object and top-level fields.
    pub fn from_model_config(raw: &serde_json::Value) -> Option<Self> {
        let vc = raw.get("vision_config")?;

        // If vision_config is null, the model has no vision support
        if vc.is_null() {
            return None;
        }

        let get_u64 = |obj: &serde_json::Value, key: &str| obj.get(key).and_then(|v| v.as_u64());
        let get_f64 = |obj: &serde_json::Value, key: &str| obj.get(key).and_then(|v| v.as_f64());
        let get_bool = |obj: &serde_json::Value, key: &str| obj.get(key).and_then(|v| v.as_bool());

        let hidden_size = get_u64(vc, "hidden_size")? as usize;
        let num_hidden_layers = get_u64(vc, "num_hidden_layers")? as usize;
        let num_attention_heads = get_u64(vc, "num_attention_heads")? as usize;
        let num_key_value_heads =
            get_u64(vc, "num_key_value_heads").unwrap_or(num_attention_heads as u64) as usize;
        let intermediate_size = get_u64(vc, "intermediate_size")? as usize;
        let patch_size = get_u64(vc, "patch_size").unwrap_or(16) as usize;
        let head_dim = get_u64(vc, "head_dim")
            .unwrap_or((hidden_size / num_attention_heads) as u64) as usize;
        let rms_norm_eps = get_f64(vc, "rms_norm_eps").unwrap_or(1e-6) as f32;

        // RoPE parameters for vision encoder
        let rope_theta = vc
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(100.0) as f32;

        let pooling_kernel_size = get_u64(vc, "pooling_kernel_size").unwrap_or(3) as usize;
        let position_embedding_size =
            get_u64(vc, "position_embedding_size").unwrap_or(10240) as usize;
        let default_output_length =
            get_u64(vc, "default_output_length").unwrap_or(280) as usize;
        let standardize = get_bool(vc, "standardize").unwrap_or(true);

        // Text model hidden size (for projection)
        let text_hidden_size = raw
            .get("text_config")
            .and_then(|tc| tc.get("hidden_size"))
            .and_then(|v| v.as_u64())
            .or_else(|| raw.get("hidden_size").and_then(|v| v.as_u64()))
            .unwrap_or(2816) as usize;

        // Special token IDs
        let image_token_id = raw
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(258880) as u32;
        let boi_token_id = raw
            .get("boi_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(255999) as u32;
        let eoi_token_id = raw
            .get("eoi_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(258882) as u32;
        let vision_soft_tokens_per_image = raw
            .get("vision_soft_tokens_per_image")
            .and_then(|v| v.as_u64())
            .unwrap_or(280) as usize;

        Some(Self {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            patch_size,
            head_dim,
            rms_norm_eps,
            rope_theta,
            pooling_kernel_size,
            position_embedding_size,
            default_output_length,
            standardize,
            text_hidden_size,
            image_token_id,
            boi_token_id,
            eoi_token_id,
            vision_soft_tokens_per_image,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vision_config() {
        let raw: serde_json::Value = serde_json::json!({
            "vision_config": {
                "hidden_size": 1152,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "intermediate_size": 4304,
                "patch_size": 16,
                "head_dim": 72,
                "rms_norm_eps": 1e-6,
                "pooling_kernel_size": 3,
                "position_embedding_size": 10240,
                "default_output_length": 280,
                "standardize": true,
                "rope_parameters": {
                    "rope_theta": 100.0,
                    "rope_type": "default"
                }
            },
            "text_config": {
                "hidden_size": 2816
            },
            "image_token_id": 258880,
            "boi_token_id": 255999,
            "eoi_token_id": 258882,
            "vision_soft_tokens_per_image": 280
        });

        let config = VisionConfig::from_model_config(&raw).unwrap();
        assert_eq!(config.hidden_size, 1152);
        assert_eq!(config.num_hidden_layers, 27);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.intermediate_size, 4304);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.head_dim, 72);
        assert_eq!(config.pooling_kernel_size, 3);
        assert_eq!(config.position_embedding_size, 10240);
        assert_eq!(config.default_output_length, 280);
        assert!(config.standardize);
        assert_eq!(config.text_hidden_size, 2816);
        assert_eq!(config.image_token_id, 258880);
        assert_eq!(config.vision_soft_tokens_per_image, 280);
        assert!((config.rope_theta - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_vision_config_returns_none() {
        let raw: serde_json::Value = serde_json::json!({
            "text_config": {
                "hidden_size": 2816
            }
        });
        assert!(VisionConfig::from_model_config(&raw).is_none());
    }

    #[test]
    fn test_null_vision_config_returns_none() {
        let raw: serde_json::Value = serde_json::json!({
            "vision_config": null
        });
        assert!(VisionConfig::from_model_config(&raw).is_none());
    }
}
