//! Quality measurement module.
//!
//! Measures KL divergence, perplexity delta, and cosine similarity
//! between pre-quant and post-quant model outputs.
//!
//! Requires an InferenceRunner — if no backend is available,
//! quality measurement returns a clear error.

pub mod cosine_sim;
pub mod kl_divergence;
pub mod perplexity;

use thiserror::Error;

use crate::inference::{InferenceError, InferenceRunner, TokenInput};
use crate::ir::{ModelMetadata, TensorMap};
use crate::progress::ProgressReporter;

/// Errors from quality measurement.
#[derive(Error, Debug)]
pub enum QualityError {
    #[error("Quality measurement requires an inference backend. No backend is currently available.")]
    BackendRequired,

    #[error("Inference runner not available: {reason}")]
    NoInferenceRunner { reason: String },

    #[error("Inference error during quality measurement: {0}")]
    Inference(#[from] InferenceError),

    #[error("KL divergence computation failed: {0}")]
    KlDivergence(#[from] kl_divergence::KlDivergenceError),

    #[error("Perplexity computation failed: {0}")]
    Perplexity(#[from] perplexity::PerplexityError),

    #[error("Cosine similarity computation failed: {0}")]
    CosineSimilarity(#[from] cosine_sim::CosineSimilarityError),
}

/// Quality measurement report — the complete results of quality analysis.
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
    /// Average cosine similarity
    pub cosine_sim_average: Option<f64>,
    /// Minimum cosine similarity (worst layer)
    pub cosine_sim_min: Option<f64>,
    /// Index of worst layer by cosine similarity
    pub cosine_sim_min_layer: Option<usize>,
}

impl QualityReport {
    /// Create an empty report (when quality measurement is skipped).
    pub fn empty() -> Self {
        Self {
            kl_divergence: None,
            per_layer_kl: None,
            perplexity_pre: None,
            perplexity_post: None,
            perplexity_delta: None,
            per_layer_cosine_sim: None,
            cosine_sim_average: None,
            cosine_sim_min: None,
            cosine_sim_min_layer: None,
        }
    }

    /// Whether any metrics are populated.
    pub fn has_metrics(&self) -> bool {
        self.kl_divergence.is_some()
            || self.perplexity_pre.is_some()
            || self.cosine_sim_average.is_some()
    }
}

/// Measure quality by comparing original and quantized model outputs.
///
/// This function:
/// 1. Loads the original model weights into the InferenceRunner
/// 2. Runs a forward pass on calibration tokens to get original logits
/// 3. Loads the quantized weights
/// 4. Runs the same forward pass to get quantized logits
/// 5. Computes KL divergence, perplexity delta, and cosine similarity
///
/// If the runner is not available, this returns a clear error.
pub fn measure_quality(
    runner: &mut dyn InferenceRunner,
    original_tensors: &TensorMap,
    quantized_tensors: &TensorMap,
    metadata: &ModelMetadata,
    progress: &ProgressReporter,
) -> Result<QualityReport, QualityError> {
    if !runner.is_available() {
        return Err(QualityError::BackendRequired);
    }

    let num_layers = metadata.num_layers as u64;
    let pb = progress.bar(num_layers * 2 + 3, "Measuring quality");

    // Generate calibration token sequence
    // Use a simple sequence of token IDs for calibration
    let vocab_size = metadata.vocab_size as u32;
    let seq_len = 32; // Short sequence for quality measurement
    let calibration_tokens: Vec<u32> = (0..seq_len)
        .map(|i| (i as u32 * 7 + 13) % vocab_size.max(1))
        .collect();
    let input = TokenInput::single(calibration_tokens.clone());

    // Memory budget: allow up to 2x the original model size
    let memory_budget = original_tensors.total_size_bytes() * 2;

    // Step 1: Load original model and run forward pass
    pb.set_message("Loading original model");
    runner.load(original_tensors, metadata, memory_budget)?;
    pb.inc(1);

    pb.set_message("Running original forward pass");
    let original_output = runner.forward(&input)?;
    pb.inc(1);

    // Extract original logits and activations
    let original_logits = extract_per_position_logits(&original_output);
    let original_activations = original_output.layer_activations.clone();

    // Step 2: Load quantized model and run forward pass
    // Note: we need to reconstruct a TensorMap-like structure for quantized weights
    pb.set_message("Loading quantized model");
    runner.load(quantized_tensors, metadata, memory_budget)?;
    pb.inc(1);

    pb.set_message("Running quantized forward pass");
    let quantized_output = runner.forward(&input)?;

    let quantized_logits = extract_per_position_logits(&quantized_output);
    let quantized_activations = quantized_output.layer_activations.clone();

    // Step 3: Compute KL divergence
    let mut report = QualityReport::empty();

    // KL divergence from logits
    if !original_logits.is_empty() && !quantized_logits.is_empty() {
        let min_len = original_logits.len().min(quantized_logits.len());
        let orig_flat: Vec<f32> = original_logits[..min_len]
            .iter()
            .flat_map(|v| v.iter())
            .copied()
            .collect();
        let quant_flat: Vec<f32> = quantized_logits[..min_len]
            .iter()
            .flat_map(|v| v.iter())
            .copied()
            .collect();

        if !orig_flat.is_empty() && orig_flat.len() == quant_flat.len() {
            match kl_divergence::kl_divergence(&orig_flat, &quant_flat) {
                Ok(kl) => report.kl_divergence = Some(kl),
                Err(e) => tracing::warn!("KL divergence computation failed: {}", e),
            }
        }
    }

    // Per-layer KL divergence from activations
    if let (Some(ref orig_acts), Some(ref quant_acts)) =
        (&original_activations, &quantized_activations)
    {
        match kl_divergence::per_layer_kl_divergence(orig_acts, quant_acts) {
            Ok(kl_result) => {
                report.per_layer_kl = Some(kl_result.per_layer);
                // Use per-layer overall if we couldn't compute from logits
                if report.kl_divergence.is_none() {
                    report.kl_divergence = Some(kl_result.overall);
                }
            }
            Err(e) => tracing::warn!("Per-layer KL divergence failed: {}", e),
        }

        for i in 0..num_layers {
            pb.set_message(format!("Measuring quality: layer {}/{}", i + 1, num_layers));
            pb.inc(1);
        }
    }

    // Perplexity delta
    if !original_logits.is_empty() && !quantized_logits.is_empty() {
        // Use token IDs shifted by 1 as targets (standard next-token prediction)
        let targets: Vec<u32> = calibration_tokens[1..].to_vec();
        let min_len = targets.len().min(original_logits.len() - 1);

        if min_len > 0 {
            match perplexity::perplexity_delta(
                &original_logits[..min_len],
                &quantized_logits[..min_len],
                &targets[..min_len],
            ) {
                Ok(ppl_result) => {
                    report.perplexity_pre = Some(ppl_result.pre_quant);
                    report.perplexity_post = Some(ppl_result.post_quant);
                    report.perplexity_delta = Some(ppl_result.delta);
                }
                Err(e) => tracing::warn!("Perplexity computation failed: {}", e),
            }
        }
    }

    // Cosine similarity of activations
    if let (Some(ref orig_acts), Some(ref quant_acts)) =
        (&original_activations, &quantized_activations)
    {
        match cosine_sim::per_layer_cosine_similarity(orig_acts, quant_acts) {
            Ok(cos_result) => {
                report.per_layer_cosine_sim = Some(cos_result.per_layer);
                report.cosine_sim_average = Some(cos_result.average);
                report.cosine_sim_min = Some(cos_result.min);
                report.cosine_sim_min_layer = Some(cos_result.min_layer_idx);
            }
            Err(e) => tracing::warn!("Cosine similarity computation failed: {}", e),
        }

        for i in 0..num_layers {
            pb.set_message(format!(
                "Cosine similarity: layer {}/{}",
                i + 1,
                num_layers
            ));
            pb.inc(1);
        }
    }

    pb.finish_with_message("Quality measurement complete");

    Ok(report)
}

/// Extract per-position logit vectors from a ForwardOutput.
fn extract_per_position_logits(
    output: &crate::inference::ForwardOutput,
) -> Vec<Vec<f32>> {
    let mut result = Vec::new();
    if output.vocab_size == 0 || output.seq_len == 0 {
        return result;
    }

    // Extract logits for batch 0, each sequence position
    for pos in 0..output.seq_len {
        let start = pos * output.vocab_size;
        let end = start + output.vocab_size;
        if end <= output.logits.len() {
            result.push(output.logits[start..end].to_vec());
        }
    }

    result
}

/// Print a terminal summary of quality metrics.
pub fn print_quality_summary(report: &QualityReport) {
    use console::style;

    if !report.has_metrics() {
        return;
    }

    eprintln!();
    eprintln!("{}", style("Quality Metrics").bold().cyan());
    eprintln!("{}", style("───────────────").dim());

    if let Some(kl) = report.kl_divergence {
        let kl_style = if kl < 0.01 {
            style(format!("{:.6}", kl)).green()
        } else if kl < 0.1 {
            style(format!("{:.6}", kl)).yellow()
        } else {
            style(format!("{:.6}", kl)).red()
        };
        eprintln!("  KL Divergence: {}", kl_style);
    }

    if let Some(pre) = report.perplexity_pre {
        eprintln!("  Perplexity (pre):  {:.2}", pre);
    }
    if let Some(post) = report.perplexity_post {
        eprintln!("  Perplexity (post): {:.2}", post);
    }
    if let Some(delta) = report.perplexity_delta {
        let delta_style = if delta.abs() < 0.5 {
            style(format!("{:+.2}", delta)).green()
        } else if delta.abs() < 2.0 {
            style(format!("{:+.2}", delta)).yellow()
        } else {
            style(format!("{:+.2}", delta)).red()
        };
        eprintln!("  Perplexity delta:  {}", delta_style);
    }

    if let Some(avg) = report.cosine_sim_average {
        let avg_style = if avg > 0.99 {
            style(format!("{:.4}", avg)).green()
        } else if avg > 0.95 {
            style(format!("{:.4}", avg)).yellow()
        } else {
            style(format!("{:.4}", avg)).red()
        };
        eprintln!("  Cosine sim (avg):  {}", avg_style);
    }

    if let (Some(min), Some(idx)) = (report.cosine_sim_min, report.cosine_sim_min_layer) {
        eprintln!("  Cosine sim (min):  {:.4} (layer {})", min, idx);
    }

    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_report_empty() {
        let report = QualityReport::empty();
        assert!(!report.has_metrics());
        assert!(report.kl_divergence.is_none());
    }

    #[test]
    fn test_quality_report_has_metrics() {
        let mut report = QualityReport::empty();
        assert!(!report.has_metrics());

        report.kl_divergence = Some(0.05);
        assert!(report.has_metrics());
    }

    #[test]
    fn test_extract_per_position_logits() {
        let output = crate::inference::ForwardOutput {
            logits: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            batch_size: 1,
            seq_len: 2,
            vocab_size: 3,
            layer_activations: None,
        };

        let logits = extract_per_position_logits(&output);
        assert_eq!(logits.len(), 2);
        assert_eq!(logits[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(logits[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_extract_empty_output() {
        let output = crate::inference::ForwardOutput {
            logits: vec![],
            batch_size: 0,
            seq_len: 0,
            vocab_size: 0,
            layer_activations: None,
        };

        let logits = extract_per_position_logits(&output);
        assert!(logits.is_empty());
    }

    #[test]
    fn test_measure_quality_requires_runner() {
        let mut runner = crate::inference::stub_runner::StubRunner;
        let original = TensorMap::new();
        let quantized = TensorMap::new();
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
        let progress = ProgressReporter::new();

        let result = measure_quality(&mut runner, &original, &quantized, &metadata, &progress);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("inference backend"), "Error should mention inference backend: {}", err);
    }
}
