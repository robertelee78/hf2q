//! Quality measurement module.
//!
//! Computes weight-level cosine similarity between original and quantized tensors.
//! KL divergence and perplexity require an inference forward pass and are populated
//! only when an external runner provides logit data.

pub mod cosine_sim;
pub mod kl_divergence;
pub mod perplexity;
pub mod regression;

use std::path::Path;

use thiserror::Error;
use tracing::{info, warn};

use crate::ir::{DType, ModelMetadata, QuantizedModel, QuantizedTensor, TensorMap, TensorRef};
use crate::progress::ProgressReporter;

/// Errors from quality measurement.
#[derive(Error, Debug)]
pub enum QualityError {
    #[error("GPU error during quality measurement: {0}")]
    Gpu(#[from] anyhow::Error),

    #[error("Tokenizer not available: {reason}")]
    TokenizerUnavailable { reason: String },
}

/// Quality measurement report -- the complete results of quality analysis.
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

/// Quality threshold configuration.
pub struct QualityThresholds {
    /// Maximum acceptable KL divergence
    pub max_kl_divergence: f64,
    /// Maximum acceptable perplexity delta
    pub max_perplexity_delta: f64,
    /// Minimum acceptable cosine similarity
    pub min_cosine_similarity: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            max_kl_divergence: 0.1,
            max_perplexity_delta: 2.0,
            min_cosine_similarity: 0.95,
        }
    }
}

/// Check if quality metrics pass thresholds.
///
/// Returns a list of violation messages. An empty list means all thresholds pass.
pub fn check_thresholds(report: &QualityReport, thresholds: &QualityThresholds) -> Vec<String> {
    let mut violations = Vec::new();

    if let Some(kl) = report.kl_divergence {
        if kl > thresholds.max_kl_divergence {
            violations.push(format!(
                "KL divergence {:.6} exceeds threshold {:.6}",
                kl, thresholds.max_kl_divergence
            ));
        }
    }

    if let Some(delta) = report.perplexity_delta {
        if delta > thresholds.max_perplexity_delta {
            violations.push(format!(
                "Perplexity delta {:.2} exceeds threshold {:.2}",
                delta, thresholds.max_perplexity_delta
            ));
        }
    }

    if let Some(avg) = report.cosine_sim_average {
        if avg < thresholds.min_cosine_similarity {
            violations.push(format!(
                "Cosine similarity {:.4} below threshold {:.4}",
                avg, thresholds.min_cosine_similarity
            ));
        }
    }

    violations
}

/// Measure quality by comparing original and quantized tensor maps.
///
/// Computes weight-level cosine similarity by comparing matching weight tensors
/// between the original and quantized (dequantized) tensor maps.
///
/// KL divergence and perplexity are not computed (they require an inference
/// forward pass which is handled by the mlx-native serve path).
pub fn measure_quality(
    original_tensors: &TensorMap,
    quantized_tensors: &TensorMap,
    _metadata: &ModelMetadata,
    _model_dir: &Path,
    progress: &ProgressReporter,
) -> Result<QualityReport, QualityError> {
    let pb = progress.bar(2, "Quality measurement");

    let mut report = QualityReport::empty();

    // Compute weight-level cosine similarity between original and quantized tensors
    pb.set_message("Computing weight cosine similarity");

    let mut orig_vecs: Vec<Vec<f32>> = Vec::new();
    let mut quant_vecs: Vec<Vec<f32>> = Vec::new();

    // Collect matching weight tensor pairs, sorted by name for deterministic ordering
    let mut weight_names: Vec<&String> = original_tensors
        .tensors
        .keys()
        .filter(|name| {
            original_tensors.tensors[*name].is_weight()
                && quantized_tensors.tensors.contains_key(*name)
        })
        .collect();
    weight_names.sort();

    for name in &weight_names {
        let orig_tensor = &original_tensors.tensors[*name];
        let quant_tensor = &quantized_tensors.tensors[*name];

        let orig_f32 = tensor_ref_to_f32(orig_tensor);
        let quant_f32 = tensor_ref_to_f32(quant_tensor);

        if orig_f32.is_empty() || quant_f32.is_empty() {
            continue;
        }

        // Use the shorter length (they should match, but be safe)
        let len = orig_f32.len().min(quant_f32.len());
        orig_vecs.push(orig_f32[..len].to_vec());
        quant_vecs.push(quant_f32[..len].to_vec());
    }

    pb.inc(1);

    if !orig_vecs.is_empty() {
        match cosine_sim::per_layer_cosine_similarity(&orig_vecs, &quant_vecs) {
            Ok(cos_result) => {
                report.per_layer_cosine_sim = Some(cos_result.per_layer.clone());
                report.cosine_sim_average = Some(cos_result.average);
                report.cosine_sim_min = Some(cos_result.min);
                report.cosine_sim_min_layer = Some(cos_result.min_layer_idx);
            }
            Err(e) => {
                warn!("Cosine similarity computation failed: {}", e);
            }
        }
    } else {
        warn!("No matching weight tensors found for cosine similarity");
    }

    info!(
        weight_pairs = orig_vecs.len(),
        "Weight-level quality measurement complete"
    );

    pb.inc(1);
    pb.finish_with_message("Quality measurement complete");

    Ok(report)
}

/// Convert a TensorRef to f32 values for quality comparison.
fn tensor_ref_to_f32(tensor: &TensorRef) -> Vec<f32> {
    match tensor.dtype {
        DType::F32 => tensor
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F16 => tensor
            .data
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        DType::BF16 => tensor
            .data
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        _ => Vec::new(),
    }
}

/// Dequantize a single QuantizedTensor back to a TensorRef for comparison.
///
/// Preserved/f16 tensors are returned as-is (data clone). Quantized tensors
/// are unpacked and dequantized using their stored scales back to f16.
///
/// This is the per-tensor primitive used by both [`dequantize_to_tensor_map`]
/// (the legacy whole-model variant kept for tests) and
/// [`measure_quality_streaming`] (the bounded-memory variant used in
/// Phase 4.5).
pub fn dequantize_single_tensor(name: &str, qt: &QuantizedTensor) -> TensorRef {
    if qt.quant_info.preserved {
        let dtype = match qt.quant_info.bits {
            32 => DType::F32,
            _ => DType::F16,
        };
        return TensorRef {
            name: name.to_string(),
            shape: qt.shape.clone(),
            dtype,
            data: qt.data.clone(),
        };
    }

    if qt.quant_info.method == "f16" {
        return TensorRef {
            name: name.to_string(),
            shape: qt.shape.clone(),
            dtype: DType::F16,
            data: qt.data.clone(),
        };
    }

    let numel: usize = qt.shape.iter().product();
    let bits = qt.quant_info.bits;
    let group_size = qt.quant_info.group_size;

    let scales: Vec<f32> = match &qt.quant_info.scales {
        Some(s) => s
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect(),
        None => {
            warn!("No scales for tensor '{}', treating as passthrough", name);
            let data: Vec<u8> = vec![0u8; numel * 2];
            return TensorRef {
                name: name.to_string(),
                shape: qt.shape.clone(),
                dtype: DType::F16,
                data,
            };
        }
    };

    let quantized_values = unpack_quantized(&qt.data, numel, bits);
    let effective_group_size = if numel < group_size {
        numel
    } else {
        group_size
    };

    let mut f16_data = Vec::with_capacity(numel * 2);
    for (i, &q_val) in quantized_values.iter().enumerate() {
        let group_idx = i / effective_group_size;
        let scale = if group_idx < scales.len() {
            scales[group_idx]
        } else {
            0.0
        };
        let dequantized = q_val as f32 * scale;
        let f16_val = half::f16::from_f32(dequantized);
        f16_data.extend_from_slice(&f16_val.to_le_bytes());
    }

    TensorRef {
        name: name.to_string(),
        shape: qt.shape.clone(),
        dtype: DType::F16,
        data: f16_data,
    }
}

/// Dequantize a QuantizedModel back into a TensorMap for forward pass comparison.
///
/// Preserved (passthrough) tensors are returned as-is. Quantized tensors are
/// unpacked and dequantized using their stored scales back to f16.
///
/// **Memory characteristic:** holds the entire dequantized model in RAM. For
/// 27B+ DWQ models this peaks near `whole_model × 2` (alongside the original
/// `tensor_map`) which OOMs on a 128 GB M5 Max — see Task #13 / ADR-012.  Use
/// [`measure_quality_streaming`] for production conversion of large models;
/// this variant is retained for tests and small-model paths.
pub fn dequantize_to_tensor_map(model: &QuantizedModel) -> TensorMap {
    let mut tensor_map = TensorMap::new();
    for (name, qt) in &model.tensors {
        tensor_map.insert(dequantize_single_tensor(name, qt));
    }
    tensor_map
}

/// Streaming weight-cosine quality measurement.
///
/// Iterates the quantized model tensor-by-tensor, dequantizing each weight
/// just-in-time and dropping the F32 buffers before the next iteration.
/// Bounds peak memory to roughly `max_tensor_F32 × 2` instead of the
/// `whole_model × 2` footprint of [`measure_quality`] +
/// [`dequantize_to_tensor_map`].
///
/// This is the path used by Phase 4.5 of the convert pipeline (main.rs)
/// for real-model DWQ conversions where the dense path otherwise OOMs at
/// ~170 GB peak on a 128 GB M5 Max — see Task #13 / ADR-012.
///
/// Returns the same shape of [`QualityReport`] as the legacy variant
/// (per-layer cosine, average, min, min layer index), so downstream
/// consumers ([`print_quality_summary`], [`check_thresholds`]) work
/// unchanged.
pub fn measure_quality_streaming(
    original_tensors: &TensorMap,
    quantized: &QuantizedModel,
    _metadata: &ModelMetadata,
    _model_dir: &Path,
    progress: &ProgressReporter,
) -> Result<QualityReport, QualityError> {
    let mut weight_names: Vec<&String> = quantized
        .tensors
        .keys()
        .filter(|name| {
            original_tensors
                .tensors
                .get(*name)
                .map(|t| t.is_weight())
                .unwrap_or(false)
        })
        .collect();
    weight_names.sort();

    let pb = progress.bar(weight_names.len() as u64, "Quality measurement (streaming)");

    let mut report = QualityReport::empty();
    let mut per_layer: Vec<f64> = Vec::with_capacity(weight_names.len());
    let mut min_val: f64 = 1.0;
    let mut min_idx: usize = 0;

    for (idx, name) in weight_names.iter().enumerate() {
        let orig_tensor = &original_tensors.tensors[*name];
        // SAFETY: weight_names was filtered against quantized.tensors above.
        let qt = &quantized.tensors[*name];

        // Dequantize one tensor at a time; drop after the cosine pair is computed.
        let dequant_ref = dequantize_single_tensor(name, qt);

        let orig_f32 = tensor_ref_to_f32(orig_tensor);
        let dequant_f32 = tensor_ref_to_f32(&dequant_ref);
        // Drop the dequant TensorRef byte buffer as soon as we have F32.
        drop(dequant_ref);

        if orig_f32.is_empty() || dequant_f32.is_empty() {
            pb.inc(1);
            continue;
        }

        let len = orig_f32.len().min(dequant_f32.len());
        let sim = match cosine_sim::cosine_similarity(&orig_f32[..len], &dequant_f32[..len]) {
            Ok(v) => v,
            Err(e) => {
                warn!("Cosine similarity failed for tensor '{}': {}", name, e);
                pb.inc(1);
                continue;
            }
        };

        if sim < min_val {
            min_val = sim;
            min_idx = idx;
        }
        per_layer.push(sim);
        // Explicit drops to free F32 staging before the next tensor.
        drop(orig_f32);
        drop(dequant_f32);
        pb.inc(1);
    }

    pb.finish_with_message("Quality measurement complete");

    if !per_layer.is_empty() {
        let average = per_layer.iter().sum::<f64>() / per_layer.len() as f64;
        report.cosine_sim_average = Some(average);
        report.cosine_sim_min = Some(min_val);
        report.cosine_sim_min_layer = Some(min_idx);
        report.per_layer_cosine_sim = Some(per_layer);
    } else {
        warn!("No matching weight tensors found for streaming quality measurement");
    }

    info!(
        weight_pairs = report
            .per_layer_cosine_sim
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(0),
        "Streaming weight-level quality measurement complete"
    );

    Ok(report)
}

/// Unpack bit-packed quantized values back to i8.
fn unpack_quantized(data: &[u8], num_elements: usize, bits: u8) -> Vec<i8> {
    match bits {
        8 => data.iter().take(num_elements).map(|&v| v as i8).collect(),
        4 => {
            let mut values = Vec::with_capacity(num_elements);
            for &byte in data {
                if values.len() >= num_elements {
                    break;
                }
                // Low nibble first
                let lo = (byte & 0x0F) as i8;
                // Sign-extend from 4 bits
                let lo = if lo >= 8 { lo - 16 } else { lo };
                values.push(lo);

                if values.len() >= num_elements {
                    break;
                }
                // High nibble
                let hi = ((byte >> 4) & 0x0F) as i8;
                let hi = if hi >= 8 { hi - 16 } else { hi };
                values.push(hi);
            }
            values
        }
        2 => {
            let mut values = Vec::with_capacity(num_elements);
            for &byte in data {
                for shift in (0..8).step_by(2) {
                    if values.len() >= num_elements {
                        break;
                    }
                    let val = ((byte >> shift) & 0x03) as i8;
                    // Sign-extend from 2 bits
                    let val = if val >= 2 { val - 4 } else { val };
                    values.push(val);
                }
            }
            values
        }
        _ => data.iter().take(num_elements).map(|&v| v as i8).collect(),
    }
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
    fn test_thresholds_default() {
        let thresholds = QualityThresholds::default();
        assert!((thresholds.max_kl_divergence - 0.1).abs() < f64::EPSILON);
        assert!((thresholds.max_perplexity_delta - 2.0).abs() < f64::EPSILON);
        assert!((thresholds.min_cosine_similarity - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_check_thresholds_pass() {
        let report = QualityReport {
            kl_divergence: Some(0.05),
            per_layer_kl: None,
            perplexity_pre: Some(5.0),
            perplexity_post: Some(5.5),
            perplexity_delta: Some(0.5),
            per_layer_cosine_sim: None,
            cosine_sim_average: Some(0.98),
            cosine_sim_min: None,
            cosine_sim_min_layer: None,
        };
        let thresholds = QualityThresholds::default();
        let violations = check_thresholds(&report, &thresholds);
        assert!(violations.is_empty(), "Expected no violations: {:?}", violations);
    }

    #[test]
    fn test_check_thresholds_fail_kl() {
        let report = QualityReport {
            kl_divergence: Some(0.5),
            per_layer_kl: None,
            perplexity_pre: None,
            perplexity_post: None,
            perplexity_delta: None,
            per_layer_cosine_sim: None,
            cosine_sim_average: None,
            cosine_sim_min: None,
            cosine_sim_min_layer: None,
        };
        let thresholds = QualityThresholds::default();
        let violations = check_thresholds(&report, &thresholds);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("KL divergence"));
    }

    #[test]
    fn test_check_thresholds_fail_all() {
        let report = QualityReport {
            kl_divergence: Some(0.5),
            per_layer_kl: None,
            perplexity_pre: Some(5.0),
            perplexity_post: Some(10.0),
            perplexity_delta: Some(5.0),
            per_layer_cosine_sim: None,
            cosine_sim_average: Some(0.80),
            cosine_sim_min: None,
            cosine_sim_min_layer: None,
        };
        let thresholds = QualityThresholds::default();
        let violations = check_thresholds(&report, &thresholds);
        assert_eq!(violations.len(), 3);
    }

    #[test]
    fn test_check_thresholds_empty_report() {
        let report = QualityReport::empty();
        let thresholds = QualityThresholds::default();
        let violations = check_thresholds(&report, &thresholds);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_unpack_8bit() {
        let data = vec![127u8, 0, 255]; // as i8: 127, 0, -1
        let result = unpack_quantized(&data, 3, 8);
        assert_eq!(result, vec![127i8, 0, -1]);
    }

    #[test]
    fn test_unpack_4bit() {
        // Byte 0xE3 = lo nibble 3, hi nibble 0xE (14, sign-extend -> -2)
        let data = vec![0xE3u8];
        let result = unpack_quantized(&data, 2, 4);
        assert_eq!(result[0], 3);
        assert_eq!(result[1], -2);
    }

    #[test]
    fn test_unpack_2bit() {
        // Byte with values: 0b_11_10_01_00 = positions 0=0, 1=1, 2=-2, 3=-1
        let byte = 0b_11_10_01_00u8;
        let result = unpack_quantized(&[byte], 4, 2);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
        assert_eq!(result[2], -2);
        assert_eq!(result[3], -1);
    }

    #[test]
    fn test_dequantize_preserved_tensor() {
        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        let f16_data: Vec<u8> = [1.0f32, 2.0]
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        tensors.insert(
            "test.weight".to_string(),
            crate::ir::QuantizedTensor {
                name: "test.weight".to_string(),
                shape: vec![2],
                original_dtype: DType::F16,
                data: f16_data.clone(),
                quant_info: crate::ir::TensorQuantInfo {
                    method: "passthrough".to_string(),
                    bits: 16,
                    group_size: 0,
                    preserved: true,
                    scales: None,
                    biases: None,
                    ggml_type: None,
                },
            },
        );

        let model = QuantizedModel {
            metadata: ModelMetadata {
                architecture: "test".to_string(),
                model_type: "test".to_string(),
                param_count: 2,
                hidden_size: 2,
                num_layers: 1,
                layer_types: vec![],
                num_attention_heads: 1,
                num_kv_heads: None,
                vocab_size: 32000,
                dtype: "float16".to_string(),
                shard_count: 1,
                num_experts: None,
                top_k_experts: None,
                intermediate_size: None,
                raw_config: serde_json::Value::Null,
                explicit_layer_types: None,
                full_attention_interval: None,
                attn_output_gate: None,
                head_dim: None,
                partial_rotary_factor: None,
                rope_parameters: None,
                linear_conv_kernel_dim: None,
                linear_key_head_dim: None,
                linear_num_key_heads: None,
                linear_value_head_dim: None,
                linear_num_value_heads: None,
                mamba_ssm_dtype: None,
                moe_intermediate_size: None,
                shared_expert_intermediate_size: None,
                mtp_num_hidden_layers: None,
                mtp_use_dedicated_embeddings: None,
                output_router_logits: None,
                router_aux_loss_coef: None,
            },
            tensors,
            quant_method: "passthrough".to_string(),
            group_size: 64,
            bits: 16,
        };

        let result = dequantize_to_tensor_map(&model);
        assert_eq!(result.len(), 1);
        let t = result.tensors.get("test.weight").unwrap();
        assert_eq!(t.dtype, DType::F16);
        assert_eq!(t.data, f16_data);
    }

    #[test]
    fn test_tensor_ref_to_f32() {
        let f16_data: Vec<u8> = [1.0f32, -2.0, 3.5]
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        let tensor = TensorRef {
            name: "test".to_string(),
            shape: vec![3],
            dtype: DType::F16,
            data: f16_data,
        };

        let vals = tensor_ref_to_f32(&tensor);
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 0.01);
        assert!((vals[1] + 2.0).abs() < 0.01);
        assert!((vals[2] - 3.5).abs() < 0.01);
    }

    /// Build a small QuantizedModel + matching original TensorMap that has
    /// a passthrough weight (identical orig/quant) and a quantized weight
    /// (identical orig/quant within float roundtrip).  Returns
    /// (orig_tensor_map, quantized_model, expected_min_layer_name).
    fn build_streaming_test_model() -> (TensorMap, QuantizedModel) {
        use std::collections::HashMap;

        // Need shape with row_dim (last dim) ≥ 32 for is_weight() to return
        // true (per ADR-012 P9b: Q4_0/Q5_0/Q8_0 require row_dim divisible
        // by 32).  Use 1×32 tensors.
        const ROW: usize = 32;

        // Quantized "ffn_up.weight": 32 elements at 8-bit, group_size=32.
        // Lossless dequant when scale = 1.0 and values are in [-128, 127].
        let orig_q_vals: Vec<f32> = (0..ROW as i32)
            .map(|i| (i - 16) as f32) // -16..15
            .collect();
        let q_data: Vec<u8> = orig_q_vals.iter().map(|&v| v as i8 as u8).collect();
        let scale_f16 = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let q_scales: Vec<u8> = scale_f16.to_vec();
        let orig_q_f16: Vec<u8> = orig_q_vals
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        // Preserved tensor: 1×32 of varying values.
        let preserved_vals: Vec<f32> = (0..ROW as i32).map(|i| (i + 1) as f32 * 0.5).collect();
        let preserved_f16: Vec<u8> = preserved_vals
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        let mut orig_map = TensorMap::new();
        orig_map.insert(TensorRef {
            name: "blk.0.ffn_up.weight".to_string(),
            shape: vec![1, ROW],
            dtype: DType::F16,
            data: orig_q_f16,
        });
        orig_map.insert(TensorRef {
            name: "blk.0.ffn_down.weight".to_string(),
            shape: vec![1, ROW],
            dtype: DType::F16,
            data: preserved_f16.clone(),
        });

        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.ffn_up.weight".to_string(),
            crate::ir::QuantizedTensor {
                name: "blk.0.ffn_up.weight".to_string(),
                shape: vec![1, ROW],
                original_dtype: DType::F16,
                data: q_data,
                quant_info: crate::ir::TensorQuantInfo {
                    method: "q8".to_string(),
                    bits: 8,
                    group_size: 32,
                    preserved: false,
                    scales: Some(q_scales),
                    biases: None,
                    ggml_type: None,
                },
            },
        );
        tensors.insert(
            "blk.0.ffn_down.weight".to_string(),
            crate::ir::QuantizedTensor {
                name: "blk.0.ffn_down.weight".to_string(),
                shape: vec![1, ROW],
                original_dtype: DType::F16,
                data: preserved_f16,
                quant_info: crate::ir::TensorQuantInfo {
                    method: "passthrough".to_string(),
                    bits: 16,
                    group_size: 0,
                    preserved: true,
                    scales: None,
                    biases: None,
                    ggml_type: None,
                },
            },
        );

        let metadata = ModelMetadata {
            architecture: "test".to_string(),
            model_type: "test".to_string(),
            param_count: 64,
            hidden_size: 32,
            num_layers: 1,
            layer_types: vec![],
            num_attention_heads: 1,
            num_kv_heads: None,
            vocab_size: 32000,
            dtype: "float16".to_string(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        };

        let model = QuantizedModel {
            metadata,
            tensors,
            quant_method: "q8".to_string(),
            group_size: 8,
            bits: 8,
        };

        (orig_map, model)
    }

    #[test]
    fn measure_quality_streaming_matches_legacy_for_lossless_dequant() {
        // Lossless q8 + passthrough → both paths must report cosine ≈ 1.0
        // for every weight.  Streaming and legacy must agree on the per-layer
        // vector length, average, min, and min_layer_idx.
        let (orig, model) = build_streaming_test_model();
        let progress = ProgressReporter::new();
        let metadata = model.metadata.clone();
        let tmp = std::path::PathBuf::from("/tmp");

        let streaming = measure_quality_streaming(&orig, &model, &metadata, &tmp, &progress)
            .expect("streaming quality should succeed");

        let legacy_dequant = dequantize_to_tensor_map(&model);
        let legacy = measure_quality(&orig, &legacy_dequant, &metadata, &tmp, &progress)
            .expect("legacy quality should succeed");

        // Both must produce the same number of per-layer entries.
        assert_eq!(
            streaming.per_layer_cosine_sim.as_ref().map(|v| v.len()),
            legacy.per_layer_cosine_sim.as_ref().map(|v| v.len()),
            "streaming and legacy must report the same number of weight tensors",
        );

        // Average and min must agree within float tolerance.
        let s_avg = streaming.cosine_sim_average.expect("streaming avg");
        let l_avg = legacy.cosine_sim_average.expect("legacy avg");
        assert!(
            (s_avg - l_avg).abs() < 1e-9,
            "streaming avg {} must match legacy avg {} within 1e-9",
            s_avg,
            l_avg,
        );

        let s_min = streaming.cosine_sim_min.expect("streaming min");
        let l_min = legacy.cosine_sim_min.expect("legacy min");
        assert!(
            (s_min - l_min).abs() < 1e-9,
            "streaming min {} must match legacy min {} within 1e-9",
            s_min,
            l_min,
        );

        // For lossless q8 + passthrough, every cosine should be ~1.0.
        for (i, &v) in streaming
            .per_layer_cosine_sim
            .as_ref()
            .expect("per-layer")
            .iter()
            .enumerate()
        {
            assert!(
                (v - 1.0).abs() < 1e-3,
                "layer {} streaming cosine {} should be ~1.0 for lossless dequant",
                i,
                v,
            );
        }
    }

    #[test]
    fn measure_quality_streaming_handles_empty_quantized_model() {
        // Empty tensor map → empty per_layer + None metrics, no panic.
        let progress = ProgressReporter::new();
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
            shard_count: 0,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        };
        let model = QuantizedModel {
            metadata: metadata.clone(),
            tensors: std::collections::HashMap::new(),
            quant_method: "q8".to_string(),
            group_size: 8,
            bits: 8,
        };
        let report = measure_quality_streaming(
            &TensorMap::new(),
            &model,
            &metadata,
            std::path::Path::new("/tmp"),
            &progress,
        )
        .expect("empty model streaming quality should succeed");
        assert!(report.cosine_sim_average.is_none());
        assert!(report.per_layer_cosine_sim.is_none());
    }
}
