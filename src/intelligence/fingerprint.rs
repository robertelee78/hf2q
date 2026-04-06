//! Model fingerprinting from config.json metadata.
//!
//! Produces a stable, hashable identifier for RuVector lookups based on
//! model architecture properties that don't change between runs.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ir::ModelMetadata;

/// Errors from model fingerprinting.
#[derive(Error, Debug)]
pub enum FingerprintError {
    #[error("Model fingerprinting failed: {reason}")]
    FingerprintFailed { reason: String },

    #[error("Missing required metadata for fingerprinting: {field}")]
    MissingField { field: String },
}

/// A stable fingerprint of a model's architecture.
///
/// Contains the key properties that determine how a model should be quantized.
/// Two instances of the same model architecture produce the same fingerprint
/// regardless of where they're stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelFingerprint {
    /// Architecture name (e.g., "Gemma4ForConditionalGeneration")
    pub architecture: String,
    /// Total parameter count
    pub total_params: u64,
    /// Number of transformer layers
    pub layer_count: u32,
    /// Number of experts (0 for non-MoE models)
    pub expert_count: u32,
    /// Attention types present in the model (deduplicated, sorted)
    pub attention_types: Vec<String>,
    /// Hidden size of the model
    pub hidden_size: u64,
    /// Original dtype (e.g., "bfloat16")
    pub dtype: String,
    /// Intermediate (FFN) size, if known
    pub intermediate_size: Option<u64>,
    /// Number of attention heads
    pub num_attention_heads: u32,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: Option<u32>,
    /// Vocabulary size
    pub vocab_size: u64,
}

impl ModelFingerprint {
    /// Create a fingerprint from model metadata.
    ///
    /// This extracts the architecture-defining properties and normalizes them
    /// into a stable representation suitable for hashing and comparison.
    pub fn from_metadata(metadata: &ModelMetadata) -> Result<Self, FingerprintError> {
        if metadata.architecture.is_empty() {
            return Err(FingerprintError::MissingField {
                field: "architecture".to_string(),
            });
        }

        // Deduplicate and sort attention/layer types for stable hashing
        let mut attention_types = metadata.unique_layer_types();
        attention_types.sort();

        Ok(Self {
            architecture: metadata.architecture.clone(),
            total_params: metadata.param_count,
            layer_count: metadata.num_layers,
            expert_count: metadata.num_experts.unwrap_or(0),
            attention_types,
            hidden_size: metadata.hidden_size,
            dtype: metadata.dtype.clone(),
            intermediate_size: metadata.intermediate_size,
            num_attention_heads: metadata.num_attention_heads,
            num_kv_heads: metadata.num_kv_heads,
            vocab_size: metadata.vocab_size,
        })
    }

    /// Produce a stable, hashable identifier string for RuVector lookups.
    ///
    /// Based on architecture name, param count, layer count, and hidden size.
    /// This is the primary key for finding stored conversion results.
    pub fn stable_id(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.architecture.hash(&mut hasher);
        self.total_params.hash(&mut hasher);
        self.layer_count.hash(&mut hasher);
        self.hidden_size.hash(&mut hasher);
        self.expert_count.hash(&mut hasher);
        self.vocab_size.hash(&mut hasher);
        format!("model-{:016x}", hasher.finish())
    }

    /// Estimated model size in bytes at f16 precision.
    ///
    /// This is an approximation: total_params * 2 bytes per f16 parameter.
    /// Used by heuristics to determine if a model fits in memory.
    pub fn estimated_f16_size_bytes(&self) -> u64 {
        self.total_params * 2
    }

    /// Estimated model size in bytes at the given bit width.
    pub fn estimated_size_bytes(&self, bits: u8) -> u64 {
        // Approximate: total_params * bits / 8
        // This doesn't account for scales/metadata overhead but is close enough
        // for heuristic memory fitting calculations.
        (self.total_params as f64 * bits as f64 / 8.0) as u64
    }

    /// Whether this is a Mixture of Experts model.
    pub fn is_moe(&self) -> bool {
        self.expert_count > 1
    }
}

impl std::fmt::Display for ModelFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}B params, {} layers, hidden={}{})",
            self.architecture,
            self.total_params as f64 / 1_000_000_000.0,
            self.layer_count,
            self.hidden_size,
            if self.is_moe() {
                format!(", {} experts", self.expert_count)
            } else {
                String::new()
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "LlamaForCausalLM".to_string(),
            model_type: "llama".to_string(),
            param_count: 8_000_000_000,
            hidden_size: 4096,
            num_layers: 32,
            layer_types: vec!["attention".to_string(); 32],
            num_attention_heads: 32,
            num_kv_heads: Some(8),
            vocab_size: 128256,
            dtype: "bfloat16".to_string(),
            shard_count: 4,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some(14336),
            raw_config: serde_json::Value::Null,
        }
    }

    fn make_moe_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "Gemma4ForConditionalGeneration".to_string(),
            model_type: "gemma4".to_string(),
            param_count: 27_000_000_000,
            hidden_size: 2816,
            num_layers: 30,
            layer_types: vec![
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            num_attention_heads: 16,
            num_kv_heads: Some(8),
            vocab_size: 262144,
            dtype: "bfloat16".to_string(),
            shard_count: 12,
            num_experts: Some(128),
            top_k_experts: Some(8),
            intermediate_size: Some(2112),
            raw_config: serde_json::Value::Null,
        }
    }

    #[test]
    fn test_fingerprint_from_metadata() {
        let metadata = make_test_metadata();
        let fp = ModelFingerprint::from_metadata(&metadata).unwrap();

        assert_eq!(fp.architecture, "LlamaForCausalLM");
        assert_eq!(fp.total_params, 8_000_000_000);
        assert_eq!(fp.layer_count, 32);
        assert_eq!(fp.expert_count, 0);
        assert_eq!(fp.hidden_size, 4096);
        assert!(!fp.is_moe());
    }

    #[test]
    fn test_fingerprint_from_moe_metadata() {
        let metadata = make_moe_metadata();
        let fp = ModelFingerprint::from_metadata(&metadata).unwrap();

        assert_eq!(fp.expert_count, 128);
        assert!(fp.is_moe());
        assert_eq!(fp.attention_types.len(), 2);
        assert!(fp.attention_types.contains(&"full_attention".to_string()));
        assert!(fp.attention_types.contains(&"sliding_attention".to_string()));
    }

    #[test]
    fn test_stable_id_deterministic() {
        let metadata = make_test_metadata();
        let fp = ModelFingerprint::from_metadata(&metadata).unwrap();

        let id1 = fp.stable_id();
        let id2 = fp.stable_id();
        assert_eq!(id1, id2);
        assert!(id1.starts_with("model-"));
    }

    #[test]
    fn test_stable_id_different_models() {
        let fp1 = ModelFingerprint::from_metadata(&make_test_metadata()).unwrap();
        let fp2 = ModelFingerprint::from_metadata(&make_moe_metadata()).unwrap();

        assert_ne!(fp1.stable_id(), fp2.stable_id());
    }

    #[test]
    fn test_estimated_sizes() {
        let metadata = make_test_metadata();
        let fp = ModelFingerprint::from_metadata(&metadata).unwrap();

        // 8B params * 2 bytes = 16 GB at f16
        assert_eq!(fp.estimated_f16_size_bytes(), 16_000_000_000);

        // 8B params * 4 bits / 8 = 4 GB at q4
        assert_eq!(fp.estimated_size_bytes(4), 4_000_000_000);
    }

    #[test]
    fn test_empty_architecture_fails() {
        let mut metadata = make_test_metadata();
        metadata.architecture = String::new();

        let result = ModelFingerprint::from_metadata(&metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_display() {
        let metadata = make_test_metadata();
        let fp = ModelFingerprint::from_metadata(&metadata).unwrap();
        let display = format!("{}", fp);
        assert!(display.contains("LlamaForCausalLM"));
        assert!(display.contains("8"));
        assert!(display.contains("32 layers"));
    }

    #[test]
    fn test_attention_types_sorted_for_stability() {
        // Even if layer_types come in different order, the fingerprint should be the same
        let mut meta1 = make_moe_metadata();
        meta1.layer_types = vec![
            "full_attention".to_string(),
            "sliding_attention".to_string(),
        ];

        let mut meta2 = make_moe_metadata();
        meta2.layer_types = vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ];

        let fp1 = ModelFingerprint::from_metadata(&meta1).unwrap();
        let fp2 = ModelFingerprint::from_metadata(&meta2).unwrap();

        assert_eq!(fp1.attention_types, fp2.attention_types);
        assert_eq!(fp1.stable_id(), fp2.stable_id());
    }
}
