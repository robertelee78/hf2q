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

    if let Some(model_type) = model_type {
        let arch = match model_type {
            "llama" => Some(ArchName::Llama3),
            "gemma3" | "gemma" | "gemma4" | "gemma4_text" => Some(ArchName::Gemma4),
            "bert" => Some(ArchName::Bert),
            "nomic_bert" => Some(ArchName::NomicBert),
            "qwen3_moe" => Some(ArchName::Qwen35Moe),
            "qwen3_5" | "qwen3_5_text" => Some(ArchName::Qwen35),
            "qwen3_5_moe" | "qwen3_5_moe_text" => Some(ArchName::Qwen35MoeFull),
            "qwen3_vl" | "qwen3_vl_moe" | "qwen3_vl_text" => Some(ArchName::Qwen3VlText),
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
            "Qwen3VLForConditionalGeneration"
            | "Qwen3VLMoeForConditionalGeneration"
            | "Qwen3VLTextForCausalLM" => Some(ArchName::Qwen3VlText),
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
}
