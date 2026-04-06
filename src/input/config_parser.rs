//! HuggingFace config.json parser.
//!
//! Parses arbitrary HF model configurations using serde_json::Value.
//! No hardcoded model families — handles any config.json structure.

use std::path::Path;

use serde_json::Value;
use thiserror::Error;
use tracing::warn;

use crate::ir::ModelMetadata;

/// Errors from config parsing.
#[derive(Error, Debug)]
pub enum ConfigParseError {
    #[error("Failed to read config file '{path}': {source}")]
    ReadError {
        path: String,
        source: std::io::Error,
    },

    #[error("Failed to parse config JSON from '{path}': {source}")]
    JsonError {
        path: String,
        source: serde_json::Error,
    },

    #[error("Config missing required field: {field}")]
    MissingField { field: String },
}

/// Parse a HuggingFace config.json into ModelMetadata.
///
/// This parser handles arbitrary architectures by:
/// 1. Looking for standard fields in the top-level config
/// 2. Falling back to nested text_config (for multimodal models like Gemma4)
/// 3. Warning on missing optional fields rather than failing
pub fn parse_config(config_path: &Path) -> Result<ModelMetadata, ConfigParseError> {
    let content = std::fs::read_to_string(config_path).map_err(|e| ConfigParseError::ReadError {
        path: config_path.display().to_string(),
        source: e,
    })?;

    let config: Value = serde_json::from_str(&content).map_err(|e| ConfigParseError::JsonError {
        path: config_path.display().to_string(),
        source: e,
    })?;

    parse_config_value(&config, config_path)
}

/// Parse a config.json Value into ModelMetadata.
fn parse_config_value(config: &Value, config_path: &Path) -> Result<ModelMetadata, ConfigParseError> {
    // Architecture name — required
    let architecture = extract_architecture(config)
        .ok_or_else(|| ConfigParseError::MissingField {
            field: "architectures".to_string(),
        })?;

    let model_type = extract_string(config, "model_type")
        .unwrap_or_else(|| "unknown".to_string());

    // For multimodal models (like Gemma4), the text config is nested
    let text_config = config.get("text_config");
    let primary = text_config.unwrap_or(config);

    let hidden_size = extract_u64(primary, "hidden_size").unwrap_or_else(|| {
        warn!("config.json missing hidden_size — defaulting to 0");
        0
    });

    let num_layers = extract_u64(primary, "num_hidden_layers").unwrap_or_else(|| {
        warn!("config.json missing num_hidden_layers — defaulting to 0");
        0
    }) as u32;

    let num_attention_heads = extract_u64(primary, "num_attention_heads").unwrap_or_else(|| {
        warn!("config.json missing num_attention_heads — defaulting to 0");
        0
    }) as u32;

    let num_kv_heads = extract_u64(primary, "num_key_value_heads").map(|v| v as u32);

    let vocab_size = extract_u64(primary, "vocab_size").unwrap_or_else(|| {
        warn!("config.json missing vocab_size — defaulting to 0");
        0
    });

    let dtype = extract_string(primary, "dtype")
        .or_else(|| extract_string(config, "dtype"))
        .unwrap_or_else(|| {
            warn!("config.json missing dtype — defaulting to 'float16'");
            "float16".to_string()
        });

    let layer_types = extract_layer_types(primary);

    let num_experts = extract_u64(primary, "num_experts").map(|v| v as u32);
    let top_k_experts = extract_u64(primary, "top_k_experts")
        .or_else(|| extract_u64(primary, "num_experts_per_tok"))
        .map(|v| v as u32);

    let intermediate_size = extract_u64(primary, "intermediate_size");

    // Count shards by looking at the directory
    let shard_count = count_shards(config_path);

    // Estimate parameter count from the index.json or config
    let param_count = estimate_param_count(config_path, config);

    Ok(ModelMetadata {
        architecture,
        model_type,
        param_count,
        hidden_size,
        num_layers,
        layer_types,
        num_attention_heads,
        num_kv_heads,
        vocab_size,
        dtype,
        shard_count,
        num_experts,
        top_k_experts,
        intermediate_size,
        raw_config: config.clone(),
    })
}

/// Extract the architecture name from config.
fn extract_architecture(config: &Value) -> Option<String> {
    config
        .get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Extract a string field.
fn extract_string(config: &Value, field: &str) -> Option<String> {
    config.get(field).and_then(|v| v.as_str()).map(|s| s.to_string())
}

/// Extract a u64 field.
fn extract_u64(config: &Value, field: &str) -> Option<u64> {
    config.get(field).and_then(|v| v.as_u64())
}

/// Extract layer types from config.
fn extract_layer_types(config: &Value) -> Vec<String> {
    if let Some(types) = config.get("layer_types").and_then(|v| v.as_array()) {
        return types
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
    }

    // If no explicit layer_types, generate a default list
    let num_layers = extract_u64(config, "num_hidden_layers").unwrap_or(0) as usize;
    if num_layers > 0 {
        // Check for MoE indicators
        let has_moe = config.get("enable_moe_block")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || config.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) > 1;

        let layer_type = if has_moe { "moe_attention" } else { "attention" };
        return vec![layer_type.to_string(); num_layers];
    }

    Vec::new()
}

/// Count safetensors shards in the model directory.
fn count_shards(config_path: &Path) -> u32 {
    let dir = match config_path.parent() {
        Some(d) => d,
        None => return 0,
    };

    // Check for sharded model (index.json)
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&index_path) {
            if let Ok(index) = serde_json::from_str::<Value>(&content) {
                if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                    let mut shard_names: Vec<String> = weight_map
                        .values()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    shard_names.sort();
                    shard_names.dedup();
                    return shard_names.len() as u32;
                }
            }
        }
    }

    // Check for single shard
    if dir.join("model.safetensors").exists() {
        return 1;
    }

    // Count any .safetensors files
    let count = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "safetensors")
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0);

    count as u32
}

/// Estimate total parameter count.
fn estimate_param_count(config_path: &Path, _config: &Value) -> u64 {
    let dir = match config_path.parent() {
        Some(d) => d,
        None => return 0,
    };

    // Try to read from index.json metadata
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&index_path) {
            if let Ok(index) = serde_json::from_str::<Value>(&content) {
                if let Some(total_params) = index
                    .get("metadata")
                    .and_then(|m| m.get("total_parameters"))
                    .and_then(|v| v.as_u64())
                {
                    return total_params;
                }
                // Fallback: estimate from total_size (assume bf16 = 2 bytes per param)
                if let Some(total_size) = index
                    .get("metadata")
                    .and_then(|m| m.get("total_size"))
                    .and_then(|v| v.as_u64())
                {
                    return total_size / 2;
                }
            }
        }
    }

    0
}

/// Format model metadata for human-readable terminal display.
pub fn format_info(metadata: &ModelMetadata) -> String {
    use crate::progress::format_param_count;

    let mut lines = Vec::new();

    lines.push(format!(
        "  Architecture: {}",
        console::style(&metadata.architecture).bold()
    ));
    lines.push(format!("  Model type:   {}", metadata.model_type));
    lines.push(format!(
        "  Parameters:   {}",
        format_param_count(metadata.param_count)
    ));
    lines.push(format!("  Hidden size:  {}", metadata.hidden_size));
    lines.push(format!("  Layers:       {}", metadata.num_layers));

    let unique_types = metadata.unique_layer_types();
    if !unique_types.is_empty() {
        lines.push(format!("  Layer types:  {}", unique_types.join(", ")));
    }

    lines.push(format!("  Attn heads:   {}", metadata.num_attention_heads));
    if let Some(kv) = metadata.num_kv_heads {
        lines.push(format!("  KV heads:     {}", kv));
    }
    lines.push(format!("  Vocab size:   {}", metadata.vocab_size));
    lines.push(format!("  Dtype:        {}", metadata.dtype));
    lines.push(format!("  Shards:       {}", metadata.shard_count));

    if let Some(experts) = metadata.num_experts {
        lines.push(format!("  Experts:      {}", experts));
    }
    if let Some(top_k) = metadata.top_k_experts {
        lines.push(format!("  Top-k:        {}", top_k));
    }
    if let Some(ffn) = metadata.intermediate_size {
        lines.push(format!("  FFN size:     {}", ffn));
    }

    if metadata.is_moe() {
        lines.push(format!(
            "  Type:         {} (Mixture of Experts)",
            console::style("MoE").yellow().bold()
        ));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_config() {
        let config: Value = serde_json::from_str(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "intermediate_size": 11008,
                "dtype": "bfloat16"
            }"#,
        )
        .unwrap();

        let meta = parse_config_value(&config, Path::new("/nonexistent/config.json")).unwrap();
        assert_eq!(meta.architecture, "LlamaForCausalLM");
        assert_eq!(meta.model_type, "llama");
        assert_eq!(meta.hidden_size, 4096);
        assert_eq!(meta.num_layers, 32);
        assert_eq!(meta.num_attention_heads, 32);
        assert_eq!(meta.num_kv_heads, Some(8));
        assert_eq!(meta.vocab_size, 32000);
        assert!(!meta.is_moe());
    }

    #[test]
    fn test_parse_nested_text_config() {
        let config: Value = serde_json::from_str(
            r#"{
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "dtype": "bfloat16",
                "text_config": {
                    "hidden_size": 2816,
                    "num_hidden_layers": 30,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "vocab_size": 262144,
                    "num_experts": 128,
                    "top_k_experts": 8,
                    "intermediate_size": 2112,
                    "layer_types": ["sliding_attention", "full_attention"]
                }
            }"#,
        )
        .unwrap();

        let meta = parse_config_value(&config, Path::new("/nonexistent/config.json")).unwrap();
        assert_eq!(meta.architecture, "Gemma4ForConditionalGeneration");
        assert_eq!(meta.hidden_size, 2816);
        assert_eq!(meta.num_layers, 30);
        assert_eq!(meta.num_experts, Some(128));
        assert_eq!(meta.top_k_experts, Some(8));
        assert!(meta.is_moe());
        assert_eq!(meta.layer_types.len(), 2);
    }

    #[test]
    fn test_parse_missing_optional_fields() {
        let config: Value = serde_json::from_str(
            r#"{
                "architectures": ["MinimalModel"],
                "model_type": "minimal"
            }"#,
        )
        .unwrap();

        let meta = parse_config_value(&config, Path::new("/nonexistent/config.json")).unwrap();
        assert_eq!(meta.architecture, "MinimalModel");
        assert_eq!(meta.hidden_size, 0);
        assert_eq!(meta.num_layers, 0);
        assert!(!meta.is_moe());
    }

    #[test]
    fn test_missing_architecture_fails() {
        let config: Value = serde_json::from_str(r#"{"model_type": "test"}"#).unwrap();

        let result = parse_config_value(&config, Path::new("/nonexistent/config.json"));
        assert!(result.is_err());
    }
}
