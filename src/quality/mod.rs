//! Quality measurement module (Epic 4).
//!
//! Measures KL divergence, perplexity delta, and cosine similarity
//! between pre-quant and post-quant model outputs.

pub mod cosine_sim;
pub mod kl_divergence;
pub mod perplexity;

use thiserror::Error;

/// Errors from quality measurement.
#[derive(Error, Debug)]
pub enum QualityError {
    #[error("Quality measurement is not yet implemented (Epic 4)")]
    NotImplemented,

    #[error("Inference required for quality measurement but runner not available: {reason}")]
    NoInferenceRunner { reason: String },
}

/// Quality measurement report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualityReport {
    /// Overall KL divergence
    pub kl_divergence: Option<f64>,
    /// Per-layer KL divergence
    pub per_layer_kl: Option<Vec<f64>>,
    /// Pre-quantization perplexity
    pub perplexity_pre: Option<f64>,
    /// Post-quantization perplexity
    pub perplexity_post: Option<f64>,
    /// Perplexity delta
    pub perplexity_delta: Option<f64>,
    /// Per-layer cosine similarity
    pub per_layer_cosine_sim: Option<Vec<f64>>,
}
