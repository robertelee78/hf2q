//! Architecture detection and model registry.
//!
//! Reads config.json to determine the model architecture and validates
//! that it is supported. Currently only Gemma4ForConditionalGeneration
//! is supported.

use std::path::Path;

use serde::Deserialize;
use thiserror::Error;
use tracing::info;

/// Errors from architecture detection.
#[derive(Error, Debug)]
pub enum RegistryError {
    #[error(
        "Unsupported architecture: {architecture}. \
         Supported: Gemma4ForConditionalGeneration"
    )]
    UnsupportedArchitecture { architecture: String },

    #[error("config.json missing 'architectures' field")]
    MissingArchitectures,

    #[error("config.json 'architectures' field is empty")]
    EmptyArchitectures,

    #[error("Failed to read config.json at '{path}': {reason}")]
    ReadError { path: String, reason: String },

    #[error("Failed to parse config.json at '{path}': {reason}")]
    ParseError { path: String, reason: String },
}

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// Gemma 4 (multimodal, MoE-based conditional generation)
    Gemma4,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArchitecture::Gemma4 => write!(f, "Gemma4ForConditionalGeneration"),
        }
    }
}

impl ModelArchitecture {
    /// Map an architecture string from config.json to an enum variant.
    fn from_str(arch: &str) -> Option<Self> {
        match arch {
            "Gemma4ForConditionalGeneration" => Some(ModelArchitecture::Gemma4),
            _ => None,
        }
    }
}

/// Minimal config structure for architecture detection.
///
/// We only need the `architectures` field from config.json.
#[derive(Deserialize)]
struct ConfigArchitectures {
    architectures: Option<Vec<String>>,
}

/// Parsed model configuration extracted from config.json.
///
/// Contains the full config needed for weight loading and memory estimation.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Detected model architecture.
    pub architecture: ModelArchitecture,
    /// Raw architecture string from config.json.
    pub architecture_str: String,
    /// Full parsed config.json as a JSON value.
    pub raw: serde_json::Value,
}

impl ModelConfig {
    /// Get a value from the text_config sub-object, falling back to top level.
    pub fn text_config_or_root(&self, key: &str) -> Option<&serde_json::Value> {
        self.raw
            .get("text_config")
            .and_then(|tc| tc.get(key))
            .or_else(|| self.raw.get(key))
    }

    /// Get a u64 from text_config or root.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.text_config_or_root(key).and_then(|v| v.as_u64())
    }

    /// Get hidden_size from the config.
    pub fn hidden_size(&self) -> u64 {
        self.get_u64("hidden_size").unwrap_or(0)
    }

    /// Get num_hidden_layers from the config.
    pub fn num_hidden_layers(&self) -> u64 {
        self.get_u64("num_hidden_layers").unwrap_or(0)
    }

    /// Get num_attention_heads from the config.
    pub fn num_attention_heads(&self) -> u64 {
        self.get_u64("num_attention_heads").unwrap_or(0)
    }

    /// Get num_key_value_heads from the config.
    pub fn num_kv_heads(&self) -> u64 {
        self.get_u64("num_key_value_heads").unwrap_or(0)
    }

    /// Get vocab_size from the config.
    pub fn vocab_size(&self) -> u64 {
        self.get_u64("vocab_size").unwrap_or(0)
    }

    /// Get head_dim from the config, falling back to hidden_size / num_heads.
    pub fn head_dim(&self) -> u64 {
        self.get_u64("head_dim").unwrap_or_else(|| {
            let heads = self.num_attention_heads();
            if heads > 0 {
                self.hidden_size() / heads
            } else {
                0
            }
        })
    }

    /// Get intermediate_size from the config.
    pub fn intermediate_size(&self) -> u64 {
        self.get_u64("intermediate_size").unwrap_or(0)
    }

    /// Get num_experts from the config.
    pub fn num_experts(&self) -> u64 {
        self.get_u64("num_experts").unwrap_or(0)
    }

    /// Get top_k_experts (num_experts_per_tok) from the config.
    pub fn top_k_experts(&self) -> u64 {
        self.get_u64("top_k_experts")
            .or_else(|| self.get_u64("num_experts_per_tok"))
            .unwrap_or(0)
    }
}

/// Detect the model architecture from a config.json file.
///
/// Reads the `architectures` array from config.json and maps the first entry
/// to a supported [`ModelArchitecture`]. Returns an error if the architecture
/// is not supported.
pub fn detect_architecture(config_path: &Path) -> Result<ModelConfig, RegistryError> {
    let content = std::fs::read_to_string(config_path).map_err(|e| RegistryError::ReadError {
        path: config_path.display().to_string(),
        reason: e.to_string(),
    })?;

    let raw: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| RegistryError::ParseError {
            path: config_path.display().to_string(),
            reason: e.to_string(),
        })?;

    let config: ConfigArchitectures =
        serde_json::from_value(raw.clone()).map_err(|e| RegistryError::ParseError {
            path: config_path.display().to_string(),
            reason: e.to_string(),
        })?;

    let architectures = config
        .architectures
        .ok_or(RegistryError::MissingArchitectures)?;

    if architectures.is_empty() {
        return Err(RegistryError::EmptyArchitectures);
    }

    let arch_str = &architectures[0];
    let architecture =
        ModelArchitecture::from_str(arch_str).ok_or_else(|| RegistryError::UnsupportedArchitecture {
            architecture: arch_str.clone(),
        })?;

    info!(
        architecture = %arch_str,
        "Detected model architecture"
    );

    Ok(ModelConfig {
        architecture,
        architecture_str: arch_str.clone(),
        raw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_config(dir: &Path, json: &str) -> std::path::PathBuf {
        let path = dir.join("config.json");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_detect_gemma4_architecture() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = write_config(
            tmp.path(),
            r#"{
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_config": {
                    "hidden_size": 2816,
                    "num_hidden_layers": 30,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "vocab_size": 262144,
                    "num_experts": 128,
                    "intermediate_size": 2112
                }
            }"#,
        );

        let config = detect_architecture(&config_path).unwrap();
        assert_eq!(config.architecture, ModelArchitecture::Gemma4);
        assert_eq!(config.architecture_str, "Gemma4ForConditionalGeneration");
        assert_eq!(config.hidden_size(), 2816);
        assert_eq!(config.num_hidden_layers(), 30);
        assert_eq!(config.num_attention_heads(), 16);
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.vocab_size(), 262144);
        assert_eq!(config.num_experts(), 128);
        assert_eq!(config.intermediate_size(), 2112);
    }

    #[test]
    fn test_detect_unsupported_architecture() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = write_config(
            tmp.path(),
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama"
            }"#,
        );

        let result = detect_architecture(&config_path);
        assert!(result.is_err());
        match result.unwrap_err() {
            RegistryError::UnsupportedArchitecture { architecture } => {
                assert_eq!(architecture, "LlamaForCausalLM");
            }
            other => panic!("Expected UnsupportedArchitecture, got: {}", other),
        }
    }

    #[test]
    fn test_detect_missing_architectures_field() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = write_config(
            tmp.path(),
            r#"{"model_type": "test"}"#,
        );

        let result = detect_architecture(&config_path);
        assert!(result.is_err());
        match result.unwrap_err() {
            RegistryError::MissingArchitectures => {}
            other => panic!("Expected MissingArchitectures, got: {}", other),
        }
    }

    #[test]
    fn test_detect_empty_architectures() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = write_config(
            tmp.path(),
            r#"{"architectures": []}"#,
        );

        let result = detect_architecture(&config_path);
        assert!(result.is_err());
        match result.unwrap_err() {
            RegistryError::EmptyArchitectures => {}
            other => panic!("Expected EmptyArchitectures, got: {}", other),
        }
    }

    #[test]
    fn test_detect_nonexistent_file() {
        let result = detect_architecture(Path::new("/nonexistent/config.json"));
        assert!(result.is_err());
        match result.unwrap_err() {
            RegistryError::ReadError { .. } => {}
            other => panic!("Expected ReadError, got: {}", other),
        }
    }

    #[test]
    fn test_detect_invalid_json() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = write_config(tmp.path(), "not valid json");

        let result = detect_architecture(&config_path);
        assert!(result.is_err());
        match result.unwrap_err() {
            RegistryError::ParseError { .. } => {}
            other => panic!("Expected ParseError, got: {}", other),
        }
    }

    #[test]
    fn test_unsupported_architecture_error_message() {
        let err = RegistryError::UnsupportedArchitecture {
            architecture: "FooBarModel".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("FooBarModel"));
        assert!(msg.contains("Gemma4ForConditionalGeneration"));
    }

    #[test]
    fn test_model_config_text_config_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        // Config with hidden_size at top level (no text_config)
        let config_path = write_config(
            tmp.path(),
            r#"{
                "architectures": ["Gemma4ForConditionalGeneration"],
                "hidden_size": 4096,
                "num_hidden_layers": 32
            }"#,
        );

        let config = detect_architecture(&config_path).unwrap();
        assert_eq!(config.hidden_size(), 4096);
        assert_eq!(config.num_hidden_layers(), 32);
    }

    #[test]
    fn test_model_architecture_display() {
        assert_eq!(
            format!("{}", ModelArchitecture::Gemma4),
            "Gemma4ForConditionalGeneration"
        );
    }
}
