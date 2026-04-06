//! Inference module — wraps mlx-rs for forward passes.
//!
//! Used by both DWQ calibration and quality measurement.
//! Platform-specific #[cfg] is isolated here.
//! No other module imports mlx_rs directly.

#[cfg(feature = "mlx-backend")]
pub mod mlx_runner;
#[cfg(not(feature = "mlx-backend"))]
pub mod mlx_runner {
    //! Placeholder when mlx-backend feature is not enabled.
}

pub mod stub_runner;

use thiserror::Error;

/// Errors from inference operations.
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Unsupported platform: MLX requires Apple Silicon with the mlx-backend feature enabled")]
    UnsupportedPlatform,

    #[error("Quality measurement requires the mlx-backend feature. Rebuild with: cargo build --features mlx-backend")]
    MlxBackendRequired,

    #[error("Model loading failed: {reason}")]
    LoadFailed { reason: String },

    #[error("Forward pass failed: {reason}")]
    ForwardFailed { reason: String },

    #[error("Logits extraction failed: {reason}")]
    LogitsFailed { reason: String },

    #[error("Runner not loaded — call load() before forward() or logits()")]
    NotLoaded,
}

/// Token input for a forward pass — a batch of token ID sequences.
#[derive(Debug, Clone)]
pub struct TokenInput {
    /// Token IDs: shape [batch_size, seq_len]
    pub token_ids: Vec<Vec<u32>>,
}

impl TokenInput {
    /// Create a single-sequence input.
    pub fn single(tokens: Vec<u32>) -> Self {
        Self {
            token_ids: vec![tokens],
        }
    }

    /// Batch size.
    pub fn batch_size(&self) -> usize {
        self.token_ids.len()
    }

    /// Sequence length (of first sequence).
    pub fn seq_len(&self) -> usize {
        self.token_ids.first().map(|s| s.len()).unwrap_or(0)
    }
}

/// Output from a forward pass.
#[derive(Debug, Clone)]
pub struct ForwardOutput {
    /// Logits: shape [batch_size, seq_len, vocab_size], stored as f32 in row-major order.
    pub logits: Vec<f32>,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Per-layer activations (optional, for quality measurement).
    /// Each entry is the hidden state after that layer, flattened as f32.
    pub layer_activations: Option<Vec<Vec<f32>>>,
}

impl ForwardOutput {
    /// Get logits for a specific batch item at the last sequence position.
    /// Returns a slice of length vocab_size.
    pub fn last_token_logits(&self, batch_idx: usize) -> Option<&[f32]> {
        if batch_idx >= self.batch_size {
            return None;
        }
        let seq_offset = (self.seq_len - 1) * self.vocab_size;
        let batch_offset = batch_idx * self.seq_len * self.vocab_size;
        let start = batch_offset + seq_offset;
        let end = start + self.vocab_size;
        if end <= self.logits.len() {
            Some(&self.logits[start..end])
        } else {
            None
        }
    }

    /// Get all logits for a specific batch item.
    /// Returns a slice of length seq_len * vocab_size.
    pub fn batch_logits(&self, batch_idx: usize) -> Option<&[f32]> {
        if batch_idx >= self.batch_size {
            return None;
        }
        let start = batch_idx * self.seq_len * self.vocab_size;
        let end = start + self.seq_len * self.vocab_size;
        if end <= self.logits.len() {
            Some(&self.logits[start..end])
        } else {
            None
        }
    }
}

/// Trait for inference runners.
///
/// All implementations must be Send + Sync.
/// The runner provides load(), forward(), and logits() for quality measurement and DWQ.
pub trait InferenceRunner: Send + Sync {
    /// Human-readable name.
    fn name(&self) -> &str;

    /// Whether this runner is available on the current platform.
    fn is_available(&self) -> bool;

    /// Load model weights from a TensorMap into the runner.
    /// After this call, the runner is ready for forward passes.
    ///
    /// The `memory_budget_bytes` parameter constrains how much memory the runner
    /// may use for loaded weights (to respect NFR4).
    fn load(
        &mut self,
        tensor_map: &crate::ir::TensorMap,
        metadata: &crate::ir::ModelMetadata,
        memory_budget_bytes: usize,
    ) -> Result<(), InferenceError>;

    /// Run a forward pass on the given token input.
    /// Returns output including logits and optionally per-layer activations.
    fn forward(&self, input: &TokenInput) -> Result<ForwardOutput, InferenceError>;

    /// Extract logits from the model for the given input.
    /// Convenience method — equivalent to forward() but only returns the logits
    /// at the last token position for each batch item.
    ///
    /// Returns a Vec of length batch_size, where each inner Vec has length vocab_size.
    fn logits(&self, input: &TokenInput) -> Result<Vec<Vec<f32>>, InferenceError>;
}

/// Create the appropriate inference runner for the current platform.
///
/// Returns the MlxRunner if the mlx-backend feature is enabled,
/// otherwise returns the StubRunner.
pub fn create_runner() -> Box<dyn InferenceRunner> {
    #[cfg(feature = "mlx-backend")]
    {
        Box::new(mlx_runner::MlxRunner::new())
    }
    #[cfg(not(feature = "mlx-backend"))]
    {
        Box::new(stub_runner::StubRunner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_input_single() {
        let input = TokenInput::single(vec![1, 2, 3]);
        assert_eq!(input.batch_size(), 1);
        assert_eq!(input.seq_len(), 3);
    }

    #[test]
    fn test_forward_output_last_token_logits() {
        let output = ForwardOutput {
            logits: vec![
                // batch 0, seq pos 0: [1.0, 2.0, 3.0]
                1.0, 2.0, 3.0,
                // batch 0, seq pos 1: [4.0, 5.0, 6.0]
                4.0, 5.0, 6.0,
            ],
            batch_size: 1,
            seq_len: 2,
            vocab_size: 3,
            layer_activations: None,
        };

        let last = output.last_token_logits(0).unwrap();
        assert_eq!(last, &[4.0, 5.0, 6.0]);
        assert!(output.last_token_logits(1).is_none());
    }

    #[test]
    fn test_forward_output_batch_logits() {
        let output = ForwardOutput {
            logits: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
            ],
            batch_size: 2,
            seq_len: 2,
            vocab_size: 3,
            layer_activations: None,
        };

        let b0 = output.batch_logits(0).unwrap();
        assert_eq!(b0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b1 = output.batch_logits(1).unwrap();
        assert_eq!(b1, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_create_runner_returns_stub_without_feature() {
        // Without mlx-backend feature, this should return a stub
        let runner = create_runner();
        // On non-feature builds it's a stub; on feature builds it's MlxRunner
        // Either way, the runner should have a name
        assert!(!runner.name().is_empty());
    }
}
