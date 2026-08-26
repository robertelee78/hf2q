//! Shared Hugging Face `config.json` architecture detection.
//!
//! Conversion and calibration must interpret the same exact config bytes.
//! Keeping the closed mapping here prevents calibration from requiring an
//! optional `architectures` entry when conversion would correctly use the
//! canonical `model_type` discriminator.

use crate::quantize::ggml_quants::ArchName;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedModelArch {
    pub observed: String,
}

pub fn detect_model_arch(config: &serde_json::Value) -> Result<ArchName, UnsupportedModelArch> {
    let model_type = config.get("model_type").and_then(|value| value.as_str());
    let architectures: Vec<&str> = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .map(|values| values.iter().filter_map(|value| value.as_str()).collect())
        .unwrap_or_default();

    // Standalone Qwen vision wrappers are a retired architecture surface.
    // Reject either discriminator before positive fallback so a conflicting
    // config cannot be smuggled through a supported text-family alias. Nested
    // `vision_config.model_type=qwen3_vl` remains a valid projector schema;
    // this boundary intentionally inspects only the root model identity.
    const RETIRED_MODEL_TYPES: &[&str] = &[
        "qwen3_vl",
        "qwen3vl",
        "qwen3_vl_moe",
        "qwen3vlmoe",
        "qwen3_vl_text",
    ];
    const RETIRED_ARCHITECTURES: &[&str] = &[
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3VLTextForCausalLM",
    ];
    if let Some(retired) = model_type.filter(|value| RETIRED_MODEL_TYPES.contains(value)) {
        return Err(UnsupportedModelArch {
            observed: retired.to_owned(),
        });
    }
    if let Some(retired) = architectures
        .iter()
        .copied()
        .find(|value| RETIRED_ARCHITECTURES.contains(value))
    {
        return Err(UnsupportedModelArch {
            observed: retired.to_owned(),
        });
    }

    if let Some(model_type) = model_type {
        let arch = match model_type {
            "llama" => Some(ArchName::Llama3),
            "gemma3" | "gemma" | "gemma4" | "gemma4_text" => Some(ArchName::Gemma4),
            "bert" => Some(ArchName::Bert),
            "nomic_bert" => Some(ArchName::NomicBert),
            "qwen3_moe" => Some(ArchName::Qwen35Moe),
            "qwen3_5" | "qwen3_5_text" => Some(ArchName::Qwen35),
            "qwen3_5_moe" | "qwen3_5_moe_text" => Some(ArchName::Qwen35MoeFull),
            "minimax_m2" => Some(ArchName::MiniMaxM2),
            "deepseek_v4" => Some(ArchName::Deepseek4),
            _ => None,
        };
        if let Some(arch) = arch {
            return Ok(arch);
        }
    }

    for architecture in &architectures {
        let arch = match *architecture {
            "LlamaForCausalLM" => Some(ArchName::Llama3),
            name if name.starts_with("Gemma3")
                || name.starts_with("Gemma2")
                || name.starts_with("Gemma4")
                || name == "GemmaForCausalLM" =>
            {
                Some(ArchName::Gemma4)
            }
            "BertForMaskedLM" | "BertModel" => Some(ArchName::Bert),
            "NomicBertModel" => Some(ArchName::NomicBert),
            "Qwen3MoeForCausalLM" => Some(ArchName::Qwen35Moe),
            "Qwen3_5MoeForCausalLM" | "Qwen3_5MoeForConditionalGeneration" => {
                Some(ArchName::Qwen35MoeFull)
            }
            "Qwen3_5ForCausalLM" | "Qwen3_5ForConditionalGeneration" => Some(ArchName::Qwen35),
            "MiniMaxM2ForCausalLM" => Some(ArchName::MiniMaxM2),
            "DeepseekV4ForCausalLM" => Some(ArchName::Deepseek4),
            _ => None,
        };
        if let Some(arch) = arch {
            return Ok(arch);
        }
    }

    Err(UnsupportedModelArch {
        observed: model_type
            .map(str::to_owned)
            .or_else(|| architectures.first().map(|value| (*value).to_owned()))
            .unwrap_or_else(|| "<missing model_type and architectures>".into()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_type_is_authoritative_and_architectures_is_a_fallback() {
        let config = serde_json::json!({
            "model_type": "qwen3_5",
            "architectures": ["LlamaForCausalLM"]
        });
        assert_eq!(detect_model_arch(&config).unwrap(), ArchName::Qwen35);

        let config = serde_json::json!({"architectures": ["Qwen3_5ForCausalLM"]});
        assert_eq!(detect_model_arch(&config).unwrap(), ArchName::Qwen35);
    }

    #[test]
    fn standalone_qwen_vision_is_explicitly_unsupported() {
        for config in [
            serde_json::json!({"model_type": "qwen3_vl"}),
            serde_json::json!({"model_type": "qwen3vl"}),
            serde_json::json!({"model_type": "qwen3_vl_moe"}),
            serde_json::json!({"model_type": "qwen3vlmoe"}),
            serde_json::json!({"model_type": "qwen3_vl_text"}),
            serde_json::json!({"architectures": ["Qwen3VLForConditionalGeneration"]}),
            serde_json::json!({"architectures": ["Qwen3VLMoeForConditionalGeneration"]}),
            serde_json::json!({"architectures": ["Qwen3VLTextForCausalLM"]}),
            serde_json::json!({
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3_5ForConditionalGeneration"]
            }),
            serde_json::json!({
                "model_type": "qwen3_5",
                "architectures": ["Qwen3VLForConditionalGeneration"]
            }),
        ] {
            assert!(detect_model_arch(&config).is_err());
        }
    }
}
