//! Validated DFlash2 checkpoint descriptors (ADR-067 D0).

use serde::Deserialize;
use std::path::Path;

pub const MAX_DFLASH2_BLOCK_ROWS: usize = 16;
pub const DFLASH2_CONVOLUTION_STAGES: usize = 2;
pub const DFLASH2_SELECTOR_RANK: usize = 256;
pub const DFLASH2_SELECTOR_TOP_K: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DFlash2LayerType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, Deserialize)]
struct RawDFlash2SpecificConfig {
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    mask_token_id: i64,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
    #[serde(default)]
    final_logit_softcapping: Option<f32>,
    #[serde(default)]
    output_multiplier: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawRopeParameters {
    rope_theta: f64,
    rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RawDFlash2Config {
    architectures: Vec<String>,
    attention_bias: bool,
    is_causal: bool,
    dtype: String,
    dflash_config: RawDFlash2SpecificConfig,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    layer_types: Vec<DFlash2LayerType>,
    max_position_embeddings: usize,
    max_window_layers: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    num_target_layers: usize,
    rms_norm_eps: f32,
    rope_parameters: RawRopeParameters,
    sliding_window: usize,
    tie_word_embeddings: bool,
    use_sliding_window: bool,
    vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct RawSpeculatorsVerifier {
    architectures: Vec<String>,
    name_or_path: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RawSpeculatorsConfig {
    algorithm: String,
    verifier: RawSpeculatorsVerifier,
}

#[derive(Debug, Clone, Deserialize)]
struct RawNestedTransformerConfig {
    attention_bias: bool,
    #[serde(default)]
    dtype: Option<String>,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    layer_types: Vec<DFlash2LayerType>,
    max_position_embeddings: usize,
    #[serde(rename = "max_window_layers")]
    _target_max_window_layers: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    rope_parameters: RawRopeParameters,
    sliding_window: usize,
    tie_word_embeddings: bool,
    use_sliding_window: bool,
    vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct RawNestedDFlash2Config {
    architectures: Vec<String>,
    aux_hidden_state_layer_ids: Vec<usize>,
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    draft_vocab_size: usize,
    dtype: String,
    mask_token_id: i64,
    sample_from_anchor: bool,
    selector_rank: usize,
    selector_top_k: usize,
    sliding_window_non_causal: bool,
    speculators_config: RawSpeculatorsConfig,
    speculators_model_type: String,
    tie_word_embeddings: bool,
    transformer_layer_config: RawNestedTransformerConfig,
}

impl RawNestedDFlash2Config {
    fn normalize(self) -> Result<RawDFlash2Config, DFlash2ConfigError> {
        let target_layers = match self.speculators_config.verifier.name_or_path.as_str() {
            "google/gemma-4-E2B-it" => 35,
            "google/gemma-4-31B-it" => 60,
            target => {
                return Err(DFlash2ConfigError::Invalid(format!(
                    "nested speculators schema has unsupported target identity {target}"
                )));
            }
        };
        if self.speculators_config.algorithm != "dflash2"
            || self.speculators_model_type != "dflash2"
            || self.speculators_config.verifier.architectures.as_slice()
                != ["Gemma4ForConditionalGeneration"]
            || self.sample_from_anchor
            || !self.sliding_window_non_causal
            || self.draft_vocab_size != self.transformer_layer_config.vocab_size
            || self
                .transformer_layer_config
                .dtype
                .as_deref()
                .is_some_and(|dtype| dtype != self.dtype.as_str())
            || self.tie_word_embeddings
            || self.transformer_layer_config.tie_word_embeddings
        {
            return Err(DFlash2ConfigError::Invalid(
                "nested speculators DFlash2 envelope is incompatible".into(),
            ));
        }
        let transformer = self.transformer_layer_config;
        Ok(RawDFlash2Config {
            architectures: self.architectures,
            attention_bias: transformer.attention_bias,
            is_causal: false,
            dtype: self.dtype,
            dflash_config: RawDFlash2SpecificConfig {
                block_size: self.block_size,
                conv_group_size: self.conv_group_size,
                conv_kernel_size: self.conv_kernel_size,
                mask_token_id: self.mask_token_id,
                selector_rank: self.selector_rank,
                selector_top_k: self.selector_top_k,
                target_layer_ids: self.aux_hidden_state_layer_ids,
                final_logit_softcapping: None,
                output_multiplier: None,
            },
            head_dim: transformer.head_dim,
            hidden_act: transformer.hidden_act,
            hidden_size: transformer.hidden_size,
            intermediate_size: transformer.intermediate_size,
            layer_types: transformer.layer_types,
            max_position_embeddings: transformer.max_position_embeddings,
            max_window_layers: transformer.num_hidden_layers,
            model_type: transformer.model_type,
            num_attention_heads: transformer.num_attention_heads,
            num_hidden_layers: transformer.num_hidden_layers,
            num_key_value_heads: transformer.num_key_value_heads,
            num_target_layers: target_layers,
            rms_norm_eps: transformer.rms_norm_eps,
            rope_parameters: transformer.rope_parameters,
            sliding_window: transformer.sliding_window,
            tie_word_embeddings: false,
            use_sliding_window: transformer.use_sliding_window,
            vocab_size: transformer.vocab_size,
        })
    }
}

/// Runtime descriptor shared by every family adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct DFlash2Config {
    pub draft_model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_target_layers: usize,
    pub target_layer_ids: Vec<usize>,
    pub layer_types: Vec<DFlash2LayerType>,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub block_size: usize,
    pub conv_group_size: usize,
    pub conv_kernel_size: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub mask_token_id: u32,
    pub final_logit_softcapping: Option<f32>,
    pub output_multiplier: Option<f32>,
}

/// Target metadata supplied by a family adapter. Checkpoint identity is bound
/// separately because the Hugging Face config does not name the target revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2TargetShape<'a> {
    pub architecture: &'a str,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum DFlash2ConfigError {
    #[error("DFlash2 config IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("DFlash2 config JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("DFlash2 config validation: {0}")]
    Invalid(String),
    #[error("DFlash2 target {architecture} is incompatible: {reason}")]
    Target {
        architecture: String,
        reason: String,
    },
}

impl DFlash2Config {
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self, DFlash2ConfigError> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }

    pub fn from_json_str(json: &str) -> Result<Self, DFlash2ConfigError> {
        let value: serde_json::Value = serde_json::from_str(json)?;
        let raw = if value.get("transformer_layer_config").is_some() {
            serde_json::from_value::<RawNestedDFlash2Config>(value)?.normalize()?
        } else {
            serde_json::from_value::<RawDFlash2Config>(value)?
        };
        let specific = &raw.dflash_config;
        let invalid = |message: String| DFlash2ConfigError::Invalid(message);

        if raw.architectures.as_slice() != ["DFlash2DraftModel"] {
            return Err(invalid(format!(
                "architectures must be [\"DFlash2DraftModel\"], got {:?}",
                raw.architectures
            )));
        }
        if raw.dtype != "bfloat16" {
            return Err(invalid(format!(
                "dtype must be bfloat16, got {}",
                raw.dtype
            )));
        }
        if raw.attention_bias || raw.is_causal || raw.tie_word_embeddings {
            return Err(invalid(
                "attention_bias, is_causal and tie_word_embeddings must be false".into(),
            ));
        }
        if raw.hidden_act != "silu" {
            return Err(invalid(format!(
                "hidden_act must be silu, got {}",
                raw.hidden_act
            )));
        }
        for (name, value) in [
            ("hidden_size", raw.hidden_size),
            ("intermediate_size", raw.intermediate_size),
            ("num_hidden_layers", raw.num_hidden_layers),
            ("num_attention_heads", raw.num_attention_heads),
            ("num_key_value_heads", raw.num_key_value_heads),
            ("head_dim", raw.head_dim),
            ("vocab_size", raw.vocab_size),
            ("num_target_layers", raw.num_target_layers),
            ("max_position_embeddings", raw.max_position_embeddings),
            ("sliding_window", raw.sliding_window),
        ] {
            if value == 0 {
                return Err(invalid(format!("{name} must be nonzero")));
            }
        }
        if raw.num_attention_heads % raw.num_key_value_heads != 0 {
            return Err(invalid(format!(
                "num_attention_heads {} must be divisible by num_key_value_heads {}",
                raw.num_attention_heads, raw.num_key_value_heads
            )));
        }
        if raw.layer_types.len() != raw.num_hidden_layers {
            return Err(invalid(format!(
                "layer_types length {} != num_hidden_layers {}",
                raw.layer_types.len(),
                raw.num_hidden_layers
            )));
        }
        if raw.max_window_layers > raw.num_hidden_layers {
            return Err(invalid(format!(
                "max_window_layers {} exceeds num_hidden_layers {}",
                raw.max_window_layers, raw.num_hidden_layers
            )));
        }
        if !raw.use_sliding_window
            || !raw
                .layer_types
                .iter()
                .any(|layer| *layer == DFlash2LayerType::SlidingAttention)
        {
            return Err(invalid(
                "DFlash2 requires use_sliding_window and at least one sliding_attention layer"
                    .into(),
            ));
        }
        if raw.rms_norm_eps <= 0.0 || !raw.rms_norm_eps.is_finite() {
            return Err(invalid("rms_norm_eps must be finite and positive".into()));
        }
        if raw.rope_parameters.rope_type != "default"
            || raw.rope_parameters.rope_theta <= 0.0
            || !raw.rope_parameters.rope_theta.is_finite()
        {
            return Err(invalid(
                "rope_parameters must have rope_type=default and finite positive rope_theta".into(),
            ));
        }
        if !(2..=MAX_DFLASH2_BLOCK_ROWS).contains(&specific.block_size) {
            return Err(invalid(format!(
                "block_size must be in 2..={MAX_DFLASH2_BLOCK_ROWS}, got {}",
                specific.block_size
            )));
        }
        if specific.conv_group_size == 0 || raw.hidden_size % specific.conv_group_size != 0 {
            return Err(invalid(format!(
                "conv_group_size {} must divide hidden_size {}",
                specific.conv_group_size, raw.hidden_size
            )));
        }
        if specific.conv_kernel_size != 2 {
            return Err(invalid(format!(
                "conv_kernel_size must be 2, got {}",
                specific.conv_kernel_size
            )));
        }
        if specific.selector_rank != DFLASH2_SELECTOR_RANK
            || specific.selector_top_k != DFLASH2_SELECTOR_TOP_K
        {
            return Err(invalid(format!(
                "selector must be rank {DFLASH2_SELECTOR_RANK}, top-k {DFLASH2_SELECTOR_TOP_K}; got rank {}, top-k {}",
                specific.selector_rank, specific.selector_top_k
            )));
        }
        if specific.target_layer_ids.is_empty() {
            return Err(invalid("target_layer_ids must be non-empty".into()));
        }
        for pair in specific.target_layer_ids.windows(2) {
            if pair[0] >= pair[1] {
                return Err(invalid(format!(
                    "target_layer_ids must be strictly increasing, got {:?}",
                    specific.target_layer_ids
                )));
            }
        }
        if let Some(layer) = specific.target_layer_ids.last() {
            if *layer >= raw.num_target_layers {
                return Err(invalid(format!(
                    "target layer {layer} is outside num_target_layers {}",
                    raw.num_target_layers
                )));
            }
        }
        if specific.mask_token_id < 0 || specific.mask_token_id as usize >= raw.vocab_size {
            return Err(invalid(format!(
                "mask_token_id {} is outside vocabulary {}",
                specific.mask_token_id, raw.vocab_size
            )));
        }
        for (name, value) in [
            ("final_logit_softcapping", specific.final_logit_softcapping),
            ("output_multiplier", specific.output_multiplier),
        ] {
            if let Some(value) = value {
                if value <= 0.0 || !value.is_finite() {
                    return Err(invalid(format!("{name} must be finite and positive")));
                }
            }
        }

        raw.hidden_size
            .checked_mul(specific.target_layer_ids.len())
            .ok_or_else(|| invalid("fc input dimension overflows usize".into()))?;
        raw.num_attention_heads
            .checked_mul(raw.head_dim)
            .ok_or_else(|| invalid("query projection dimension overflows usize".into()))?;
        raw.num_key_value_heads
            .checked_mul(raw.head_dim)
            .ok_or_else(|| invalid("KV projection dimension overflows usize".into()))?;

        Ok(Self {
            draft_model_type: raw.model_type,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            vocab_size: raw.vocab_size,
            num_target_layers: raw.num_target_layers,
            target_layer_ids: specific.target_layer_ids.clone(),
            layer_types: raw.layer_types,
            max_position_embeddings: raw.max_position_embeddings,
            sliding_window: raw.sliding_window,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw.rope_parameters.rope_theta,
            block_size: specific.block_size,
            conv_group_size: specific.conv_group_size,
            conv_kernel_size: specific.conv_kernel_size,
            selector_rank: specific.selector_rank,
            selector_top_k: specific.selector_top_k,
            mask_token_id: specific.mask_token_id as u32,
            final_logit_softcapping: specific.final_logit_softcapping,
            output_multiplier: specific.output_multiplier,
        })
    }

    pub fn proposal_tokens(&self) -> usize {
        self.block_size - 1
    }

    pub fn fc_input_dim(&self) -> usize {
        self.hidden_size * self.target_layer_ids.len()
    }

    pub fn convolution_groups(&self) -> usize {
        self.hidden_size / self.conv_group_size
    }

    pub fn dynamic_projection_width(&self) -> usize {
        DFLASH2_CONVOLUTION_STAGES * self.conv_kernel_size * self.convolution_groups()
    }

    pub fn validate_target(
        &self,
        target: DFlash2TargetShape<'_>,
    ) -> Result<(), DFlash2ConfigError> {
        let mismatch = if target.hidden_size != self.hidden_size {
            Some(format!(
                "hidden_size {} != draft {}",
                target.hidden_size, self.hidden_size
            ))
        } else if target.vocab_size != self.vocab_size {
            Some(format!(
                "vocab_size {} != draft {}",
                target.vocab_size, self.vocab_size
            ))
        } else if target.num_hidden_layers != self.num_target_layers {
            Some(format!(
                "num_hidden_layers {} != draft num_target_layers {}",
                target.num_hidden_layers, self.num_target_layers
            ))
        } else {
            None
        };
        if let Some(reason) = mismatch {
            return Err(DFlash2ConfigError::Target {
                architecture: target.architecture.into(),
                reason,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub const QWEN38_CONFIG: &str = r#"{
      "architectures":["DFlash2DraftModel"], "attention_bias":false,
      "is_causal":false, "dtype":"bfloat16", "head_dim":128,
      "hidden_act":"silu", "hidden_size":5120, "intermediate_size":17408,
      "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
      "max_position_embeddings":262144, "max_window_layers":5,
      "model_type":"qwen3", "num_attention_heads":32, "num_hidden_layers":5,
      "num_key_value_heads":8, "num_target_layers":64, "rms_norm_eps":0.000001,
      "rope_parameters":{"rope_theta":10000000.0,"rope_type":"default"},
      "sliding_window":2048, "tie_word_embeddings":false,
      "use_sliding_window":true, "vocab_size":248320,
      "dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,
        "mask_token_id":248070,"selector_rank":256,"selector_top_k":16,
        "target_layer_ids":[5,19,33,47,61]}
    }"#;

    const GEMMA4_E2B_CONFIG: &str = r#"{
      "architectures":["DFlash2DraftModel"],
      "aux_hidden_state_layer_ids":[3,13,23,33],
      "block_size":8,"conv_group_size":16,"conv_kernel_size":2,
      "draft_vocab_size":262144,"dtype":"bfloat16","mask_token_id":4,
      "sample_from_anchor":false,"selector_rank":256,"selector_top_k":16,
      "sliding_window_non_causal":true,"speculators_model_type":"dflash2",
      "tie_word_embeddings":false,
      "speculators_config":{"algorithm":"dflash2","verifier":{
        "architectures":["Gemma4ForConditionalGeneration"],
        "name_or_path":"google/gemma-4-E2B-it"}},
      "transformer_layer_config":{"attention_bias":false,
        "head_dim":128,"hidden_act":"silu","hidden_size":1536,
        "intermediate_size":6144,
        "layer_types":["sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","full_attention"],
        "max_position_embeddings":131072,"max_window_layers":28,
        "model_type":"qwen3","num_attention_heads":12,"num_hidden_layers":5,
        "num_key_value_heads":4,"rms_norm_eps":0.000001,
        "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},
        "sliding_window":2048,"tie_word_embeddings":false,
        "use_sliding_window":true,"vocab_size":262144}
    }"#;

    #[test]
    fn parses_pinned_qwen38_contract() {
        let cfg = DFlash2Config::from_json_str(QWEN38_CONFIG).unwrap();
        assert_eq!(cfg.proposal_tokens(), 7);
        assert_eq!(cfg.fc_input_dim(), 25_600);
        assert_eq!(cfg.convolution_groups(), 320);
        assert_eq!(cfg.dynamic_projection_width(), 1_280);
        assert_eq!(cfg.target_layer_ids, [5, 19, 33, 47, 61]);
        cfg.validate_target(DFlash2TargetShape {
            architecture: "qwen35",
            hidden_size: 5_120,
            vocab_size: 248_320,
            num_hidden_layers: 64,
        })
        .unwrap();
    }

    #[test]
    fn parses_nested_gemma4_speculators_contract() {
        let cfg = DFlash2Config::from_json_str(GEMMA4_E2B_CONFIG).unwrap();
        assert_eq!(cfg.hidden_size, 1_536);
        assert_eq!(cfg.num_target_layers, 35);
        assert_eq!(cfg.target_layer_ids, [3, 13, 23, 33]);
        assert_eq!(cfg.fc_input_dim(), 6_144);
        assert_eq!(cfg.dynamic_projection_width(), 384);
        assert_eq!(
            cfg.layer_types.last(),
            Some(&DFlash2LayerType::FullAttention)
        );
        cfg.validate_target(DFlash2TargetShape {
            architecture: "gemma4",
            hidden_size: 1_536,
            vocab_size: 262_144,
            num_hidden_layers: 35,
        })
        .unwrap();
    }

    #[test]
    fn rejects_cross_family_shape_instead_of_approximating() {
        let cfg = DFlash2Config::from_json_str(QWEN38_CONFIG).unwrap();
        let error = cfg
            .validate_target(DFlash2TargetShape {
                architecture: "gemma4",
                hidden_size: 5_120,
                vocab_size: 262_144,
                num_hidden_layers: 64,
            })
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("vocab_size 262144 != draft 248320"));
    }

    #[test]
    fn rejects_invalid_dynamic_convolution_shape() {
        let json = QWEN38_CONFIG.replace("\"conv_group_size\":16", "\"conv_group_size\":17");
        let error = DFlash2Config::from_json_str(&json).unwrap_err();
        assert!(error.to_string().contains("must divide hidden_size"));
    }
}
