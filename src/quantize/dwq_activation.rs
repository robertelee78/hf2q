//! Activation-based DWQ calibration.
//!
//! ADR-008: the candle-based GPU forward pass has been removed.  Activation-based
//! calibration requires an inference forward pass which now lives exclusively in the
//! mlx-native serve path.  This module returns an error so callers fall back to
//! weight-space DWQ calibration.

use std::path::Path;

use crate::ir::{ModelMetadata, QuantizedModel, TensorMap};
use crate::progress::ProgressReporter;

use super::dwq::{DwqConfig, DwqError};

/// Run activation-based DWQ calibration.
///
/// ADR-008: returns an error because the candle GPU forward pass has been removed.
/// Callers should fall back to weight-space DWQ calibration via `run_dwq_calibration`.
pub fn run_dwq_activation_calibration(
    _tensor_map: &TensorMap,
    _metadata: &ModelMetadata,
    _config: &DwqConfig,
    _model_dir: &Path,
    _progress: &ProgressReporter,
) -> Result<QuantizedModel, DwqError> {
    Err(DwqError::GpuError {
        reason: "Activation-based DWQ requires an inference forward pass; \
                 the candle backend has been removed (ADR-008). \
                 Falling back to weight-space calibration."
            .to_string(),
    })
}

/// Extract the layer index from a tensor name.
///
/// Looks for patterns like ".layers.N." and returns N.
#[cfg(test)]
fn extract_layer_index(name: &str) -> Option<usize> {
    let marker = ".layers.";
    let start = name.find(marker)? + marker.len();
    let rest = &name[start..];
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(
            extract_layer_index("model.layers.0.self_attn.q_proj.weight"),
            Some(0)
        );
        assert_eq!(
            extract_layer_index("model.layers.15.mlp.gate_proj.weight"),
            Some(15)
        );
        assert_eq!(
            extract_layer_index("model.layers.123.post_attention_layernorm.weight"),
            Some(123)
        );
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
        assert_eq!(extract_layer_index("model.norm.weight"), None);
    }

    #[test]
    fn test_activation_calibration_returns_error() {
        let tensor_map = TensorMap::new();
        let metadata = ModelMetadata {
            architecture: "test".to_string(),
            model_type: "test".to_string(),
            param_count: 0,
            hidden_size: 0,
            num_layers: 0,
            layer_types: vec![],
            num_attention_heads: 0,
            num_kv_heads: None,
            vocab_size: 0,
            dtype: "float16".to_string(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
            raw_config: serde_json::Value::Null,
        };
        let config = DwqConfig::default();
        let progress = ProgressReporter::new();

        let result = run_dwq_activation_calibration(
            &tensor_map,
            &metadata,
            &config,
            Path::new("/nonexistent"),
            &progress,
        );
        assert!(result.is_err());
    }
}
