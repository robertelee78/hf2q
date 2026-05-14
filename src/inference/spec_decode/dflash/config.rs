//! DFlash drafter configuration (ADR-030 §3.0).
//!
//! Mirrors `/opt/dflash/dflash/model_mlx.py:DFlashConfig` (lines 29-48).
//! Loaded from HuggingFace `config.json` of a z-lab DFlash draft model.
//!
//! ## Key shape facts (from `z-lab/gemma-4-26B-A4B-it-DFlash`, verified iter-2)
//!
//! - `num_hidden_layers = 5` (small — 4 sliding + 1 full attention)
//! - `hidden_size = 2816` (matches target gemma-4-26B-A4B-it's hidden_size)
//! - `num_attention_heads = 32` (vs target's 16)
//! - `num_key_value_heads = 8` (matches target; GQA factor 4 for draft, 2 for target)
//! - `head_dim = 128` (vs target's 256)
//! - `intermediate_size = 5632` (vs target's 2112)
//! - `block_size = 16` (training default; runtime override via `HF2Q_SPEC_DFLASH_BLOCK_SIZE`)
//! - `target_layer_ids = [1, 6, 11, 17, 22, 27]` — 6 hidden states from target's 30 layers
//! - `mask_token_id = 4`
//! - `final_logit_softcapping = 30.0`
//! - `rope_theta = 1_000_000`
//! - `sliding_window = 2048` (drafter; target uses 1024)

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Layer attention type — matches HF transformers `layer_types` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    SlidingAttention,
}

/// Nested `dflash_config` block in HF config.json — holds drafter-specific
/// fields not native to qwen3 architecture.
#[derive(Debug, Clone, Deserialize)]
struct DFlashSpecificConfig {
    #[serde(default)]
    mask_token_id: i64,
    target_layer_ids: Vec<usize>,
}

/// Raw HF config.json layout for a DFlash draft model.
///
/// We deserialize this then convert into [`DFlashConfig`] (the validated
/// runtime view). This indirection lets us reject malformed configs at
/// load time rather than at first-forward.
#[derive(Debug, Clone, Deserialize)]
struct RawDFlashConfigJson {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    max_position_embeddings: usize,
    block_size: usize,
    num_target_layers: usize,
    dflash_config: DFlashSpecificConfig,

    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    layer_types: Option<Vec<LayerType>>,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    final_logit_softcapping: Option<f32>,
}

/// Validated drafter config (runtime view).
///
/// Constructed via [`DFlashConfig::from_json_path`] or
/// [`DFlashConfig::from_json_str`]. All invariants checked at construction:
///
/// - `layer_types.len() == num_hidden_layers`
/// - `sliding_window` present if any `SlidingAttention` layer
/// - `target_layer_ids` strictly increasing, all `< num_target_layers`
/// - `block_size >= 2` (need at least 1 mask token)
/// - `num_target_layers > 0`
#[derive(Debug, Clone)]
pub struct DFlashConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub block_size: usize,
    pub target_layer_ids: Vec<usize>,
    pub num_target_layers: usize,
    pub mask_token_id: u32,
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    pub layer_types: Vec<LayerType>,
    pub sliding_window: Option<usize>,
    pub final_logit_softcapping: Option<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("dflash config IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("dflash config JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("dflash config validation: {0}")]
    Invalid(String),
}

impl DFlashConfig {
    /// Load + validate config from a JSON file path. Mirrors the
    /// `path / "config.json"` step of `load_draft` in `model_mlx.py:208`.
    pub fn from_json_path<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let raw = std::fs::read_to_string(path)?;
        Self::from_json_str(&raw)
    }

    /// Load + validate config from JSON text.
    pub fn from_json_str(s: &str) -> Result<Self, ConfigError> {
        let raw: RawDFlashConfigJson = serde_json::from_str(s)?;

        let layer_types = raw.layer_types.unwrap_or_else(|| {
            vec![LayerType::FullAttention; raw.num_hidden_layers]
        });

        // Validation mirrors `load_draft` lines 209-216 in model_mlx.py.
        if layer_types.len() != raw.num_hidden_layers {
            return Err(ConfigError::Invalid(format!(
                "layer_types length {} != num_hidden_layers {}",
                layer_types.len(),
                raw.num_hidden_layers
            )));
        }
        if layer_types.iter().any(|t| matches!(t, LayerType::SlidingAttention))
            && raw.sliding_window.is_none()
        {
            return Err(ConfigError::Invalid(
                "sliding_window required when any sliding_attention layer is present".into(),
            ));
        }
        if raw.block_size < 2 {
            return Err(ConfigError::Invalid(format!(
                "block_size must be >= 2 (at least 1 mask token); got {}",
                raw.block_size
            )));
        }
        if raw.num_target_layers == 0 {
            return Err(ConfigError::Invalid("num_target_layers must be > 0".into()));
        }
        if raw.dflash_config.target_layer_ids.is_empty() {
            return Err(ConfigError::Invalid(
                "target_layer_ids must be non-empty".into(),
            ));
        }
        // strictly increasing + bounds-checked
        for w in raw.dflash_config.target_layer_ids.windows(2) {
            if w[0] >= w[1] {
                return Err(ConfigError::Invalid(format!(
                    "target_layer_ids must be strictly increasing; got {:?}",
                    raw.dflash_config.target_layer_ids
                )));
            }
        }
        if let Some(&max) = raw.dflash_config.target_layer_ids.iter().max() {
            if max >= raw.num_target_layers {
                return Err(ConfigError::Invalid(format!(
                    "target_layer_ids contains {} >= num_target_layers {}",
                    max, raw.num_target_layers
                )));
            }
        }
        if raw.dflash_config.mask_token_id < 0 || raw.dflash_config.mask_token_id > u32::MAX as i64 {
            return Err(ConfigError::Invalid(format!(
                "mask_token_id {} not in u32 range",
                raw.dflash_config.mask_token_id
            )));
        }

        let rope_scaling = raw.rope_scaling.and_then(|v| match v {
            serde_json::Value::Null => None,
            serde_json::Value::Object(m) => Some(m.into_iter().collect()),
            _ => None,
        });

        Ok(DFlashConfig {
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            intermediate_size: raw.intermediate_size,
            vocab_size: raw.vocab_size,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw.rope_theta,
            max_position_embeddings: raw.max_position_embeddings,
            block_size: raw.block_size,
            target_layer_ids: raw.dflash_config.target_layer_ids,
            num_target_layers: raw.num_target_layers,
            mask_token_id: raw.dflash_config.mask_token_id as u32,
            rope_scaling,
            layer_types,
            sliding_window: raw.sliding_window,
            final_logit_softcapping: raw.final_logit_softcapping,
        })
    }

    /// True if the given layer index uses sliding-window attention.
    /// Mirrors `model_mlx.py:DFlashAttention.is_sliding` (line 73).
    pub fn is_sliding(&self, layer_idx: usize) -> bool {
        matches!(self.layer_types[layer_idx], LayerType::SlidingAttention)
    }

    /// Sliding-window size for a given layer, or `None` for full attention.
    pub fn layer_sliding_window(&self, layer_idx: usize) -> Option<usize> {
        if self.is_sliding(layer_idx) {
            self.sliding_window
        } else {
            None
        }
    }

    /// FC projection input size = `num_target_layers_used * hidden_size`.
    /// Mirrors `model_mlx.py:DFlashDraftModel.__init__` line 138-139:
    ///   `concat_dim = len(config.target_layer_ids) * config.hidden_size`
    pub fn fc_input_dim(&self) -> usize {
        self.target_layer_ids.len() * self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim JSON from `~/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/.../config.json`
    /// (confirmed iter-2 of ADR-030 mission). This is the actual production
    /// drafter config that hf2q must consume.
    const GEMMA4_26B_A4B_DFLASH_CONFIG: &str = r#"{
        "architectures": ["DFlashDraftModel"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "block_size": 16,
        "bos_token_id": 2,
        "dflash_config": {
            "mask_token_id": 4,
            "target_layer_ids": [1, 6, 11, 17, 22, 27]
        },
        "dtype": "bfloat16",
        "eos_token_id": 1,
        "final_logit_softcapping": 30.0,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2816,
        "intermediate_size": 5632,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ],
        "max_position_embeddings": 262144,
        "max_window_layers": 5,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
        "num_target_layers": 30,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-06,
        "sliding_window": 2048,
        "tie_word_embeddings": false,
        "use_cache": true,
        "use_sliding_window": true,
        "vocab_size": 262144,
        "rope_theta": 1000000,
        "rope_scaling": null
    }"#;

    #[test]
    fn loads_gemma4_26b_a4b_dflash_config() {
        let cfg = DFlashConfig::from_json_str(GEMMA4_26B_A4B_DFLASH_CONFIG)
            .expect("valid config should parse");

        assert_eq!(cfg.hidden_size, 2816);
        assert_eq!(cfg.num_hidden_layers, 5);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 5632);
        assert_eq!(cfg.vocab_size, 262144);
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.target_layer_ids, vec![1, 6, 11, 17, 22, 27]);
        assert_eq!(cfg.num_target_layers, 30);
        assert_eq!(cfg.mask_token_id, 4);
        assert_eq!(cfg.sliding_window, Some(2048));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1e-3);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-9);

        assert_eq!(cfg.layer_types.len(), 5);
        assert_eq!(cfg.layer_types[0], LayerType::SlidingAttention);
        assert_eq!(cfg.layer_types[4], LayerType::FullAttention);

        assert!(cfg.is_sliding(0));
        assert!(!cfg.is_sliding(4));
        assert_eq!(cfg.layer_sliding_window(0), Some(2048));
        assert_eq!(cfg.layer_sliding_window(4), None);

        assert_eq!(cfg.fc_input_dim(), 6 * 2816);
    }

    #[test]
    fn rejects_block_size_below_2() {
        let bad = GEMMA4_26B_A4B_DFLASH_CONFIG.replace("\"block_size\": 16", "\"block_size\": 1");
        let err = DFlashConfig::from_json_str(&bad).unwrap_err();
        match err {
            ConfigError::Invalid(msg) => assert!(msg.contains("block_size")),
            e => panic!("expected Invalid block_size, got {e:?}"),
        }
    }

    #[test]
    fn rejects_target_layer_ids_out_of_bounds() {
        let bad = GEMMA4_26B_A4B_DFLASH_CONFIG.replace("[1, 6, 11, 17, 22, 27]", "[1, 6, 11, 17, 22, 30]");
        let err = DFlashConfig::from_json_str(&bad).unwrap_err();
        match err {
            ConfigError::Invalid(msg) => assert!(msg.contains("target_layer_ids")),
            e => panic!("expected Invalid target_layer_ids, got {e:?}"),
        }
    }

    #[test]
    fn rejects_target_layer_ids_not_monotonic() {
        let bad = GEMMA4_26B_A4B_DFLASH_CONFIG.replace("[1, 6, 11, 17, 22, 27]", "[1, 6, 17, 11, 22, 27]");
        let err = DFlashConfig::from_json_str(&bad).unwrap_err();
        match err {
            ConfigError::Invalid(msg) => assert!(msg.contains("strictly increasing")),
            e => panic!("expected Invalid monotonicity, got {e:?}"),
        }
    }

    #[test]
    fn rejects_layer_types_length_mismatch() {
        let bad = GEMMA4_26B_A4B_DFLASH_CONFIG.replace(
            "\"sliding_attention\",\n            \"full_attention\"",
            "\"full_attention\"",
        );
        let err = DFlashConfig::from_json_str(&bad).unwrap_err();
        match err {
            ConfigError::Invalid(msg) => assert!(msg.contains("layer_types")),
            e => panic!("expected Invalid layer_types length, got {e:?}"),
        }
    }

    #[test]
    fn rejects_sliding_layer_without_window() {
        // Remove sliding_window but keep sliding_attention layers
        let bad = GEMMA4_26B_A4B_DFLASH_CONFIG.replace("\"sliding_window\": 2048,", "");
        let err = DFlashConfig::from_json_str(&bad).unwrap_err();
        match err {
            ConfigError::Invalid(msg) => assert!(msg.contains("sliding_window")),
            e => panic!("expected Invalid sliding_window, got {e:?}"),
        }
    }
}
