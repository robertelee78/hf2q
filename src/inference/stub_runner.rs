//! Stub inference runner — returns UnsupportedPlatform on non-Apple targets
//! or when the mlx-backend feature is not enabled.

use crate::inference::{ForwardOutput, InferenceError, InferenceRunner, TokenInput};
use crate::ir::{ModelMetadata, TensorMap};

/// Stub runner for platforms without MLX support or when mlx-backend feature is disabled.
pub struct StubRunner;

impl InferenceRunner for StubRunner {
    fn name(&self) -> &str {
        "stub"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn load(
        &mut self,
        _tensor_map: &TensorMap,
        _metadata: &ModelMetadata,
        _memory_budget_bytes: usize,
    ) -> Result<(), InferenceError> {
        Err(InferenceError::MlxBackendRequired)
    }

    fn forward(&self, _input: &TokenInput) -> Result<ForwardOutput, InferenceError> {
        Err(InferenceError::MlxBackendRequired)
    }

    fn logits(&self, _input: &TokenInput) -> Result<Vec<Vec<f32>>, InferenceError> {
        Err(InferenceError::MlxBackendRequired)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_not_available() {
        let runner = StubRunner;
        assert!(!runner.is_available());
        assert_eq!(runner.name(), "stub");
    }

    #[test]
    fn test_stub_load_returns_error() {
        let mut runner = StubRunner;
        let tensor_map = TensorMap::new();
        let metadata = ModelMetadata {
            architecture: "Test".to_string(),
            model_type: "test".to_string(),
            param_count: 0,
            hidden_size: 0,
            num_layers: 0,
            layer_types: vec![],
            num_attention_heads: 0,
            num_kv_heads: None,
            vocab_size: 0,
            dtype: "float16".to_string(),
            shard_count: 0,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
            raw_config: serde_json::Value::Null,
        };
        let result = runner.load(&tensor_map, &metadata, 1024);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("mlx-backend"),
            "Error should mention mlx-backend feature: {}",
            err
        );
    }

    #[test]
    fn test_stub_forward_returns_error() {
        let runner = StubRunner;
        let input = TokenInput::single(vec![1, 2, 3]);
        let result = runner.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_stub_logits_returns_error() {
        let runner = StubRunner;
        let input = TokenInput::single(vec![1, 2, 3]);
        let result = runner.logits(&input);
        assert!(result.is_err());
    }
}
