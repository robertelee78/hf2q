//! HuggingFace config.json parser.
//!
//! Parses arbitrary HF model configurations using serde_json::Value.
//! No hardcoded model families — handles any config.json structure.
//!
//! ADR-012 Decision 2: extended with 18 Qwen3.5-family fields (all Option<T>).
//! Chesterton's fence: all new fields default to None — Gemma4 parsing is
//! byte-identical before and after this change.

use std::path::Path;

use serde_json::Value;
use thiserror::Error;
use tracing::warn;

use crate::ir::{ModelMetadata, RopeParameters};

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

    #[error("Config field '{field}' has invalid value '{value}': {reason}")]
    InvalidFieldValue {
        field: String,
        value: String,
        reason: String,
    },
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

    // --- ADR-012 Decision 2: Qwen3.5-family extended fields ---
    // All are Option<T> with None defaults — Gemma4 AST is unchanged.

    // Explicit layer_types from JSON (preferred over interval derivation).
    // Note: extract_layer_types() above returns Vec (used for the legacy field).
    // Here we separately capture the raw JSON array — None means absent from JSON.
    let explicit_layer_types = extract_explicit_layer_types(primary);

    let full_attention_interval = extract_u64(primary, "full_attention_interval").map(|v| v as u32);

    let attn_output_gate = extract_bool(primary, "attn_output_gate");

    // head_dim: explicitly parsed — never derived. Decoupled from hidden_size/num_heads.
    let head_dim = extract_u64(primary, "head_dim").map(|v| v as u32);

    let partial_rotary_factor = extract_f64(primary, "partial_rotary_factor").map(|v| v as f32);

    // Nested rope_parameters object
    let rope_parameters = extract_rope_parameters(primary);

    // Linear-attention (Gated DeltaNet) kernel dimensions
    let linear_conv_kernel_dim = extract_u64(primary, "linear_conv_kernel_dim").map(|v| v as u32);
    let linear_key_head_dim = extract_u64(primary, "linear_key_head_dim").map(|v| v as u32);
    let linear_num_key_heads = extract_u64(primary, "linear_num_key_heads").map(|v| v as u32);
    let linear_value_head_dim = extract_u64(primary, "linear_value_head_dim").map(|v| v as u32);
    let linear_num_value_heads = extract_u64(primary, "linear_num_value_heads").map(|v| v as u32);

    // mamba_ssm_dtype: validated enum-like — only three accepted values.
    let mamba_ssm_dtype = extract_mamba_ssm_dtype(primary)?;

    // MoE sizing
    let moe_intermediate_size = extract_u64(primary, "moe_intermediate_size").map(|v| v as u32);
    let shared_expert_intermediate_size =
        extract_u64(primary, "shared_expert_intermediate_size").map(|v| v as u32);

    // MTP fields
    let mtp_num_hidden_layers = extract_u64(primary, "mtp_num_hidden_layers").map(|v| v as u32);
    let mtp_use_dedicated_embeddings = extract_bool(primary, "mtp_use_dedicated_embeddings");

    // Router fields
    let output_router_logits = extract_bool(primary, "output_router_logits");
    let router_aux_loss_coef = extract_f64(primary, "router_aux_loss_coef").map(|v| v as f32);

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
        // ADR-012 P1 fields
        explicit_layer_types,
        full_attention_interval,
        attn_output_gate,
        head_dim,
        partial_rotary_factor,
        rope_parameters,
        linear_conv_kernel_dim,
        linear_key_head_dim,
        linear_num_key_heads,
        linear_value_head_dim,
        linear_num_value_heads,
        mamba_ssm_dtype,
        moe_intermediate_size,
        shared_expert_intermediate_size,
        mtp_num_hidden_layers,
        mtp_use_dedicated_embeddings,
        output_router_logits,
        router_aux_loss_coef,
    })
}

/// Validate that all required Qwen3.5-MoE fields are present, returning an error
/// identifying the first missing field by name. Called by P2+ when arch == "qwen35moe".
///
/// ADR-012 Decision 2: "missing required-for-qwen35moe fields … returns an error
/// with a clear message identifying the missing field — not silent zero-filling."
pub fn validate_required_qwen35moe_fields(
    metadata: &ModelMetadata,
) -> Result<(), ConfigParseError> {
    macro_rules! require {
        ($field:expr, $name:literal) => {
            if $field.is_none() {
                return Err(ConfigParseError::MissingField {
                    field: $name.to_string(),
                });
            }
        };
    }

    require!(metadata.rope_parameters, "rope_parameters");
    // rope_parameters is present; verify mrope_section is non-empty
    if let Some(ref rp) = metadata.rope_parameters {
        if rp.mrope_section.is_empty() {
            return Err(ConfigParseError::MissingField {
                field: "rope_parameters.mrope_section".to_string(),
            });
        }
    }

    require!(metadata.full_attention_interval, "full_attention_interval");
    require!(metadata.head_dim, "head_dim");
    require!(metadata.linear_conv_kernel_dim, "linear_conv_kernel_dim");
    require!(metadata.linear_key_head_dim, "linear_key_head_dim");
    require!(metadata.linear_num_key_heads, "linear_num_key_heads");
    require!(metadata.linear_value_head_dim, "linear_value_head_dim");
    require!(metadata.linear_num_value_heads, "linear_num_value_heads");
    require!(metadata.mamba_ssm_dtype, "mamba_ssm_dtype");
    require!(metadata.moe_intermediate_size, "moe_intermediate_size");
    require!(
        metadata.shared_expert_intermediate_size,
        "shared_expert_intermediate_size"
    );
    require!(metadata.mtp_num_hidden_layers, "mtp_num_hidden_layers");

    Ok(())
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

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

/// Extract a bool field.
fn extract_bool(config: &Value, field: &str) -> Option<bool> {
    config.get(field).and_then(|v| v.as_bool())
}

/// Extract an f64 field.
fn extract_f64(config: &Value, field: &str) -> Option<f64> {
    config.get(field).and_then(|v| v.as_f64())
}

/// Extract layer types from config (legacy path used for `layer_types` in ModelMetadata).
///
/// Returns a Vec<String> because the existing field is `Vec<String>`.
/// If the JSON contains an explicit array, use that; otherwise derive.
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

/// Extract the explicit layer_types array as Option<Vec<String>>.
///
/// Returns None when the field is absent (Gemma4 path — None preserves existing AST).
/// Returns Some(...) only when the JSON explicitly contains "layer_types".
fn extract_explicit_layer_types(config: &Value) -> Option<Vec<String>> {
    config.get("layer_types").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    })
}

/// Extract nested rope config.
///
/// Accepts EITHER of two HF config formats (some models ship one,
/// some the other — both are "valid" HF; Qwen3.6 uses `rope_parameters`,
/// older models use `rope_scaling`):
///
///   - `rope_parameters` (Qwen3.6 / Qwen3.5-MoE convention) — nested
///     object with explicit `rope_theta`, `mrope_section`, etc.
///   - `rope_scaling` (older Qwen / Llama convention) — same nested
///     object shape but keyed under a different name; `type` field
///     maps to our `rope_type` field. `rope_theta` may live on the
///     parent config rather than inside the object.
///
/// Returns None when neither is present (Gemma4, LLaMA, etc.).
/// Per mantra (Chesterton's fence): the caller in P2+ was silently
/// missing rope metadata emission when configs used `rope_scaling`;
/// this accepts both forms to eliminate that silent fallback.
fn extract_rope_parameters(config: &Value) -> Option<RopeParameters> {
    let (obj, is_scaling_form) = match config.get("rope_parameters").and_then(|v| v.as_object()) {
        Some(o) => (o, false),
        None => match config.get("rope_scaling").and_then(|v| v.as_object()) {
            Some(o) => (o, true),
            None => return None,
        },
    };

    let mrope_interleaved = obj
        .get("mrope_interleaved")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mrope_section = obj
        .get("mrope_section")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect::<Vec<u32>>()
        })
        .unwrap_or_default();

    // `rope_theta` can live inside the nested object (preferred) OR
    // on the parent config (legacy `rope_scaling` form). Check both.
    let rope_theta = obj
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .or_else(|| config.get("rope_theta").and_then(|v| v.as_f64()))
        .unwrap_or(0.0);

    // Rope type naming: `rope_type` in the new form, `type` in the
    // legacy `rope_scaling` form.
    let rope_type = obj
        .get("rope_type")
        .or_else(|| if is_scaling_form { obj.get("type") } else { None })
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let partial_rotary_factor = obj
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .or_else(|| config.get("partial_rotary_factor").and_then(|v| v.as_f64()))
        .unwrap_or(0.0) as f32;

    Some(RopeParameters {
        mrope_interleaved,
        mrope_section,
        rope_theta,
        rope_type,
        partial_rotary_factor,
    })
}

/// Extract and validate `mamba_ssm_dtype`.
///
/// Accepted values: "float32", "bfloat16", "float16".
/// Returns Ok(None) when absent. Returns Err when present but invalid.
/// Silent acceptance of arbitrary strings is a broken window (ADR-012 mantra).
fn extract_mamba_ssm_dtype(config: &Value) -> Result<Option<String>, ConfigParseError> {
    match config.get("mamba_ssm_dtype") {
        None => Ok(None),
        Some(v) => {
            let s = v.as_str().ok_or_else(|| ConfigParseError::InvalidFieldValue {
                field: "mamba_ssm_dtype".to_string(),
                value: v.to_string(),
                reason: "must be a string".to_string(),
            })?;
            match s {
                "float32" | "bfloat16" | "float16" => Ok(Some(s.to_string())),
                other => Err(ConfigParseError::InvalidFieldValue {
                    field: "mamba_ssm_dtype".to_string(),
                    value: other.to_string(),
                    reason: "must be one of: float32, bfloat16, float16".to_string(),
                }),
            }
        }
    }
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

    // ---------------------------------------------------------------------------
    // Pre-existing tests (must remain byte-identical — Chesterton's fence)
    // ---------------------------------------------------------------------------

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

    // ---------------------------------------------------------------------------
    // ADR-012 P1 new tests
    // ---------------------------------------------------------------------------

    /// Load the apex Qwen3.5-MoE config.json and assert all 18 new fields.
    ///
    /// Values cross-checked against the on-disk fixture at:
    /// /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/config.json
    #[test]
    fn apex_config_parses_all_18_fields() {
        let config_path = Path::new(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/config.json",
        );
        // Skip gracefully if fixture is not present in CI
        if !config_path.exists() {
            eprintln!("SKIP apex_config_parses_all_18_fields: fixture not found");
            return;
        }

        let meta = parse_config(config_path).expect("apex config.json should parse without error");

        // Core fields
        assert_eq!(meta.vocab_size, 248_320, "vocab_size");
        assert_eq!(meta.num_layers, 40, "num_hidden_layers");
        assert_eq!(meta.num_attention_heads, 16, "num_attention_heads");

        // Field 1: layer_types (explicit)
        let explicit = meta
            .explicit_layer_types
            .as_ref()
            .expect("layer_types should be present");
        assert_eq!(explicit.len(), 40, "layer_types.len()");
        assert_eq!(explicit[0], "linear_attention", "layer_types[0]");
        assert_eq!(explicit[3], "full_attention", "layer_types[3]");

        // Field 2: full_attention_interval
        assert_eq!(
            meta.full_attention_interval,
            Some(4),
            "full_attention_interval"
        );

        // Field 3: attn_output_gate
        assert_eq!(meta.attn_output_gate, Some(true), "attn_output_gate");

        // Field 4: head_dim — explicitly parsed, not derived
        assert_eq!(meta.head_dim, Some(256), "head_dim");

        // Field 5: partial_rotary_factor (top-level)
        let prf = meta.partial_rotary_factor.expect("partial_rotary_factor");
        assert!((prf - 0.25).abs() < 1e-6, "partial_rotary_factor ≈ 0.25");

        // Field 6: rope_parameters (nested)
        let rp = meta.rope_parameters.as_ref().expect("rope_parameters");
        assert!(rp.mrope_interleaved, "mrope_interleaved");
        assert_eq!(rp.mrope_section, vec![11u32, 11, 10], "mrope_section");
        assert!(
            (rp.rope_theta - 10_000_000.0).abs() < 1.0,
            "rope_theta ≈ 10_000_000 (got {})",
            rp.rope_theta
        );
        assert_eq!(rp.rope_type, "default", "rope_type");
        assert!(
            (rp.partial_rotary_factor - 0.25).abs() < 1e-6,
            "rope_parameters.partial_rotary_factor"
        );

        // Fields 7-11: linear-attention kernel dims
        assert_eq!(
            meta.linear_conv_kernel_dim,
            Some(4),
            "linear_conv_kernel_dim"
        );
        assert_eq!(meta.linear_key_head_dim, Some(128), "linear_key_head_dim");
        assert_eq!(meta.linear_num_key_heads, Some(16), "linear_num_key_heads");
        assert_eq!(
            meta.linear_value_head_dim,
            Some(128),
            "linear_value_head_dim"
        );
        assert_eq!(
            meta.linear_num_value_heads,
            Some(32),
            "linear_num_value_heads"
        );

        // Field 12: mamba_ssm_dtype
        assert_eq!(
            meta.mamba_ssm_dtype,
            Some("float32".to_string()),
            "mamba_ssm_dtype"
        );

        // Fields 13-14: MoE sizing
        assert_eq!(meta.moe_intermediate_size, Some(512), "moe_intermediate_size");
        assert_eq!(
            meta.shared_expert_intermediate_size,
            Some(512),
            "shared_expert_intermediate_size"
        );

        // Fields 15-16: MTP
        assert_eq!(meta.mtp_num_hidden_layers, Some(1), "mtp_num_hidden_layers");
        assert_eq!(
            meta.mtp_use_dedicated_embeddings,
            Some(false),
            "mtp_use_dedicated_embeddings"
        );

        // Fields 17-18: router
        assert_eq!(
            meta.output_router_logits,
            Some(false),
            "output_router_logits"
        );
        let coef = meta.router_aux_loss_coef.expect("router_aux_loss_coef");
        assert!((coef - 0.001).abs() < 1e-6, "router_aux_loss_coef ≈ 0.001");
    }

    /// Regression guard: Gemma4 parsing is byte-identical after P1 additions.
    ///
    /// All 18 new fields must be None for the Gemma4 path. No AST change.
    #[test]
    fn gemma4_config_regression() {
        // Minimal inline Gemma4-like config — mirrors the nested text_config shape.
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

        let meta = parse_config_value(&config, Path::new("/nonexistent/config.json"))
            .expect("Gemma4 config should parse without error");

        // Pre-existing fields unchanged
        assert_eq!(meta.architecture, "Gemma4ForConditionalGeneration");
        assert_eq!(meta.hidden_size, 2816);
        assert_eq!(meta.num_layers, 30);
        assert_eq!(meta.num_attention_heads, 16);
        assert_eq!(meta.num_kv_heads, Some(8));
        assert_eq!(meta.vocab_size, 262_144);
        assert_eq!(meta.num_experts, Some(128));
        assert_eq!(meta.top_k_experts, Some(8));
        assert_eq!(meta.intermediate_size, Some(2112));
        assert_eq!(meta.layer_types.len(), 2);
        assert!(meta.is_moe());

        // All 18 new P1 fields must be None — no semantic change for Gemma4
        assert!(meta.full_attention_interval.is_none(), "full_attention_interval must be None");
        assert!(meta.attn_output_gate.is_none(), "attn_output_gate must be None");
        assert!(meta.head_dim.is_none(), "head_dim must be None");
        assert!(meta.partial_rotary_factor.is_none(), "partial_rotary_factor must be None");
        assert!(meta.rope_parameters.is_none(), "rope_parameters must be None");
        assert!(meta.linear_conv_kernel_dim.is_none(), "linear_conv_kernel_dim must be None");
        assert!(meta.linear_key_head_dim.is_none(), "linear_key_head_dim must be None");
        assert!(meta.linear_num_key_heads.is_none(), "linear_num_key_heads must be None");
        assert!(meta.linear_value_head_dim.is_none(), "linear_value_head_dim must be None");
        assert!(meta.linear_num_value_heads.is_none(), "linear_num_value_heads must be None");
        assert!(meta.mamba_ssm_dtype.is_none(), "mamba_ssm_dtype must be None");
        assert!(meta.moe_intermediate_size.is_none(), "moe_intermediate_size must be None");
        assert!(
            meta.shared_expert_intermediate_size.is_none(),
            "shared_expert_intermediate_size must be None"
        );
        assert!(meta.mtp_num_hidden_layers.is_none(), "mtp_num_hidden_layers must be None");
        assert!(
            meta.mtp_use_dedicated_embeddings.is_none(),
            "mtp_use_dedicated_embeddings must be None"
        );
        assert!(meta.output_router_logits.is_none(), "output_router_logits must be None");
        assert!(meta.router_aux_loss_coef.is_none(), "router_aux_loss_coef must be None");

        // explicit_layer_types: the text_config has "layer_types" so this will be Some —
        // the explicit_layer_types captures whatever is in the JSON.
        // What must NOT change is that the legacy layer_types Vec is still length 2.
        assert_eq!(
            meta.layer_types.len(),
            2,
            "legacy layer_types Vec length unchanged"
        );
    }

    /// A qwen35moe config missing mrope_section must error with the field name.
    #[test]
    fn malformed_qwen35moe_errors_with_field_name() {
        // Config with rope_parameters present but mrope_section omitted.
        let config: Value = serde_json::from_str(
            r#"{
                "architectures": ["Qwen3_5MoeForCausalLM"],
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "vocab_size": 248320,
                "dtype": "bfloat16",
                "full_attention_interval": 4,
                "head_dim": 256,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "rope_theta": 10000000,
                    "rope_type": "default",
                    "partial_rotary_factor": 0.25
                },
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_value_head_dim": 128,
                "linear_num_value_heads": 32,
                "mamba_ssm_dtype": "float32",
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "mtp_num_hidden_layers": 1
            }"#,
        )
        .unwrap();

        let meta = parse_config_value(&config, Path::new("/nonexistent/config.json"))
            .expect("initial parse should succeed");

        // Now validate required qwen35moe fields — should fail on mrope_section
        let err = validate_required_qwen35moe_fields(&meta)
            .expect_err("validation must fail with missing mrope_section");

        let msg = err.to_string();
        assert!(
            msg.contains("mrope_section"),
            "error must name the missing field 'mrope_section', got: {msg}"
        );
    }

    /// head_dim is read from JSON explicitly — not derived from hidden_size/num_heads.
    ///
    /// Qwen3.5-MoE: hidden_size=2048, num_attention_heads=16 → naive derivation = 128.
    /// But head_dim in config.json = 256. The parser must return 256.
    #[test]
    fn head_dim_not_derived_when_present() {
        let config: Value = serde_json::from_str(
            r#"{
                "architectures": ["Qwen3_5MoeForCausalLM"],
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "vocab_size": 248320,
                "dtype": "bfloat16",
                "head_dim": 256
            }"#,
        )
        .unwrap();

        let meta = parse_config_value(&config, Path::new("/nonexistent/config.json")).unwrap();

        // Naive derivation: 2048 / 16 = 128. The parser must return 256 (explicit).
        assert_eq!(
            meta.head_dim,
            Some(256),
            "head_dim must come from JSON, not hidden_size/num_heads"
        );
    }

    /// When both layer_types and full_attention_interval are present,
    /// explicit_layer_types (the direct JSON parse) wins in resolved_layer_types().
    #[test]
    fn layer_types_prefers_explicit_over_interval() {
        let config: Value = serde_json::from_str(
            r#"{
                "architectures": ["Qwen3_5MoeForCausalLM"],
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "num_hidden_layers": 8,
                "num_attention_heads": 16,
                "vocab_size": 248320,
                "dtype": "bfloat16",
                "full_attention_interval": 2,
                "layer_types": [
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention"
                ]
            }"#,
        )
        .unwrap();

        let meta = parse_config_value(&config, Path::new("/nonexistent/config.json")).unwrap();

        // Explicit: 3 linear + 1 full repeating (interval=4 pattern from explicit)
        // interval=2 would give: linear, full, linear, full, …
        let resolved = meta.resolved_layer_types();
        assert_eq!(resolved.len(), 8);
        // Key assertion: explicit wins over interval=2.
        // With interval=2: resolved[1] would be "full_attention" (pos 1, (1+1)%2==0).
        // With explicit: resolved[1] = "linear_attention".
        assert_eq!(
            resolved[1],
            "linear_attention",
            "resolved[1] must be linear_attention from explicit (interval=2 would give full_attention)"
        );
        assert_eq!(
            resolved[3],
            "full_attention",
            "resolved[3] must be full_attention from explicit"
        );

        // Gemma case: neither explicit nor interval — falls back to layer_types Vec
        let gemma_config: Value = serde_json::from_str(
            r#"{
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "vocab_size": 1000,
                "dtype": "bfloat16"
            }"#,
        )
        .unwrap();

        let gemma_meta =
            parse_config_value(&gemma_config, Path::new("/nonexistent/config.json")).unwrap();
        assert!(gemma_meta.explicit_layer_types.is_none());
        assert!(gemma_meta.full_attention_interval.is_none());
        // resolved_layer_types falls back to layer_types (derived as "attention" * 4)
        let gemma_resolved = gemma_meta.resolved_layer_types();
        assert_eq!(gemma_resolved.len(), 4);
        assert_eq!(gemma_resolved[0], "attention");
    }

    /// mamba_ssm_dtype must reject values outside the allowed set.
    #[test]
    fn mamba_ssm_dtype_rejects_invalid() {
        let config: Value = serde_json::from_str(
            r#"{
                "architectures": ["Qwen3_5MoeForCausalLM"],
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "vocab_size": 248320,
                "dtype": "bfloat16",
                "mamba_ssm_dtype": "int8"
            }"#,
        )
        .unwrap();

        let err = parse_config_value(&config, Path::new("/nonexistent/config.json"))
            .expect_err("int8 mamba_ssm_dtype should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("mamba_ssm_dtype"),
            "error must name the offending field, got: {msg}"
        );
        assert!(
            msg.contains("int8"),
            "error must include the invalid value, got: {msg}"
        );
    }

    /// mamba_ssm_dtype accepts all three valid values.
    #[test]
    fn mamba_ssm_dtype_accepts_valid_values() {
        for dtype in &["float32", "bfloat16", "float16"] {
            let json = format!(
                r#"{{
                    "architectures": ["Qwen3_5MoeForCausalLM"],
                    "model_type": "qwen3_5_moe_text",
                    "hidden_size": 2048,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 16,
                    "vocab_size": 248320,
                    "dtype": "bfloat16",
                    "mamba_ssm_dtype": "{dtype}"
                }}"#
            );
            let config: Value = serde_json::from_str(&json).unwrap();
            let meta = parse_config_value(&config, Path::new("/nonexistent/config.json"))
                .unwrap_or_else(|e| panic!("mamba_ssm_dtype={dtype} should parse OK, got: {e}"));
            assert_eq!(
                meta.mamba_ssm_dtype,
                Some(dtype.to_string()),
                "mamba_ssm_dtype={dtype}"
            );
        }
    }

    /// resolved_layer_types() derives from full_attention_interval when explicit absent.
    #[test]
    fn resolved_layer_types_derives_from_interval() {
        let config: Value = serde_json::from_str(
            r#"{
                "architectures": ["Qwen3_5MoeForCausalLM"],
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "num_hidden_layers": 8,
                "num_attention_heads": 16,
                "vocab_size": 248320,
                "dtype": "bfloat16",
                "full_attention_interval": 4
            }"#,
        )
        .unwrap();

        let meta = parse_config_value(&config, Path::new("/nonexistent/config.json")).unwrap();

        assert!(meta.explicit_layer_types.is_none());
        assert_eq!(meta.full_attention_interval, Some(4));

        let resolved = meta.resolved_layer_types();
        assert_eq!(resolved.len(), 8);
        // interval=4: layers at 1-indexed positions 4,8 are full_attention (i=3,7)
        assert_eq!(resolved[0], "linear_attention");
        assert_eq!(resolved[1], "linear_attention");
        assert_eq!(resolved[2], "linear_attention");
        assert_eq!(resolved[3], "full_attention");
        assert_eq!(resolved[4], "linear_attention");
        assert_eq!(resolved[7], "full_attention");
    }

    // ------------------------------------------------------------------
    // Rope schema flexibility tests (2026-04-24)
    // ------------------------------------------------------------------
    // Real-world HF configs use either `rope_parameters` (Qwen3.6) or
    // `rope_scaling` (older Qwen / Llama). Both must produce a populated
    // `RopeParameters` so Decision 7's rope.* keys emit.

    #[test]
    fn rope_parameters_nested_form_parses() {
        let config: Value = serde_json::from_str(
            r#"{"rope_parameters": {
                "mrope_section": [3, 3, 2],
                "rope_theta": 10000000.0,
                "rope_type": "mrope",
                "mrope_interleaved": true,
                "partial_rotary_factor": 0.25
            }}"#,
        )
        .unwrap();
        let rp = extract_rope_parameters(&config).expect("parsed");
        assert_eq!(rp.rope_theta, 10_000_000.0);
        assert_eq!(rp.mrope_section, vec![3, 3, 2]);
        assert_eq!(rp.rope_type, "mrope");
        assert!(rp.mrope_interleaved);
        assert_eq!(rp.partial_rotary_factor, 0.25);
    }

    #[test]
    fn rope_scaling_legacy_form_parses_with_rope_theta_on_parent() {
        // Older HF convention: `rope_scaling` carries the mrope_section
        // but rope_theta lives on the parent config. Verify both values
        // flow through to the emitter.
        let config: Value = serde_json::from_str(
            r#"{
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "rope_scaling": {
                    "mrope_section": [11, 11, 10],
                    "type": "mrope"
                }
            }"#,
        )
        .unwrap();
        let rp = extract_rope_parameters(&config).expect("legacy form parsed");
        assert_eq!(rp.rope_theta, 10_000_000.0, "theta from parent");
        assert_eq!(rp.mrope_section, vec![11, 11, 10]);
        assert_eq!(rp.rope_type, "mrope", "legacy `type` maps to rope_type");
        assert_eq!(rp.partial_rotary_factor, 0.25, "factor from parent");
    }

    #[test]
    fn rope_parameters_takes_precedence_over_rope_scaling() {
        // If BOTH are present, the newer `rope_parameters` form wins
        // — matches Qwen's own migration path (newer configs ship both
        // keys during transition periods).
        let config: Value = serde_json::from_str(
            r#"{
                "rope_parameters": {
                    "mrope_section": [3, 3, 2],
                    "rope_theta": 999.0
                },
                "rope_scaling": {
                    "mrope_section": [99, 99, 99],
                    "type": "mrope"
                }
            }"#,
        )
        .unwrap();
        let rp = extract_rope_parameters(&config).expect("parsed");
        assert_eq!(rp.mrope_section, vec![3, 3, 2], "prefers rope_parameters");
        assert_eq!(rp.rope_theta, 999.0);
    }

    #[test]
    fn rope_absent_returns_none() {
        let config: Value = serde_json::from_str(r#"{"hidden_size": 64}"#).unwrap();
        assert!(extract_rope_parameters(&config).is_none());
    }

    #[test]
    fn rope_scaling_without_parent_theta_defaults_to_zero() {
        // Safety net — ensures the parser doesn't panic when theta
        // is truly missing. Decision 7 emitter fallbacks to 10_000_000
        // for qwen35 when rp.rope_theta is zero.
        let config: Value = serde_json::from_str(
            r#"{"rope_scaling": {"mrope_section": [1, 2, 3], "type": "mrope"}}"#,
        )
        .unwrap();
        let rp = extract_rope_parameters(&config).expect("parsed");
        assert_eq!(rp.rope_theta, 0.0);
        assert_eq!(rp.mrope_section, vec![1, 2, 3]);
    }
}
