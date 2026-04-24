//! Apex quantization: importance-matrix calibrated, per-tensor optimal precision.
//!
//! Two-pass pipeline:
//! Pass 1: Run forward passes on calibration data, compute importance matrix (imatrix).
//!         Falls back to weight-magnitude-only scoring when GPU is unavailable.
//! Pass 2: Use imatrix to select optimal GGML K-quant type per tensor,
//!         then quantize each tensor at the chosen precision.
//!
//! The importance matrix measures how much each weight tensor contributes to output
//! quality. More important tensors get higher precision K-quant types while less
//! important tensors are compressed more aggressively, all subject to a target
//! average bits-per-weight constraint.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use tracing::{debug, info};

use crate::ir::{ModelMetadata, QuantizedModel, TensorMap, TensorRef};
use crate::progress::ProgressReporter;
use crate::quantize::static_quant::StaticQuantizer;
use crate::quantize::{LayerQuantConfig, Quantizer};

/// Configuration for Apex quantization.
#[derive(Debug, Clone)]
pub struct ApexConfig {
    /// Number of calibration tokens for importance matrix computation.
    pub calibration_tokens: usize,
    /// Target average bits per weight (e.g., 4.5 for balanced quality/size).
    pub target_bpw: f32,
    /// Minimum K-quant type (floor).
    pub min_type: String,
    /// Maximum K-quant type (ceiling).
    pub max_type: String,
}

impl Default for ApexConfig {
    fn default() -> Self {
        Self {
            calibration_tokens: 512,
            target_bpw: 4.5,
            min_type: "Q3_K_S".to_string(),
            max_type: "Q6_K".to_string(),
        }
    }
}

/// K-quant type descriptor with its approximate bits-per-weight and the bit width
/// used for our IR quantization pass.
#[derive(Debug, Clone)]
struct KQuantSpec {
    name: &'static str,
    bpw: f32,
    /// Bit width to pass to StaticQuantizer (nearest supported: 2, 4, or 8).
    ir_bits: u8,
}

/// Ordered spectrum of K-quant types from lowest to highest precision.
const KQUANT_SPECTRUM: &[KQuantSpec] = &[
    KQuantSpec { name: "Q2_K",   bpw: 2.5, ir_bits: 2 },
    KQuantSpec { name: "Q3_K_S", bpw: 3.0, ir_bits: 4 },
    KQuantSpec { name: "Q3_K_M", bpw: 3.5, ir_bits: 4 },
    KQuantSpec { name: "Q4_K_S", bpw: 4.2, ir_bits: 4 },
    KQuantSpec { name: "Q4_K_M", bpw: 4.5, ir_bits: 4 },
    KQuantSpec { name: "Q5_K_S", bpw: 5.0, ir_bits: 8 },
    KQuantSpec { name: "Q5_K_M", bpw: 5.5, ir_bits: 8 },
    KQuantSpec { name: "Q6_K",   bpw: 6.5, ir_bits: 8 },
];

/// Look up a K-quant spec by name, returning its index in the spectrum.
fn kquant_index(name: &str) -> Option<usize> {
    KQUANT_SPECTRUM.iter().position(|s| s.name == name)
}

/// Compute the importance matrix for all weight tensors.
///
/// For each weight tensor W the importance of element W\[i,j\] is approximated as:
///   importance\[i,j\] = mean(|activation_input\[j\]|) * |W\[i,j\]|
///
/// When GPU activations are available (Pass 1 forward pass succeeds), activation
/// magnitudes per hidden dimension are collected and multiplied element-wise with
/// weight magnitudes. When GPU is unavailable, only weight magnitudes are used
/// (all activations treated as 1.0).
///
/// Returns a map of weight tensor name to a single scalar importance score
/// (the mean of element-wise importances).
pub fn compute_importance_matrix(
    tensor_map: &TensorMap,
    metadata: &ModelMetadata,
    model_dir: &Path,
    calibration_tokens: usize,
    progress: &ProgressReporter,
) -> Result<HashMap<String, f64>> {
    // Attempt GPU-based activation capture
    let activation_magnitudes = attempt_activation_capture(tensor_map, metadata, model_dir, calibration_tokens);

    let weight_names: Vec<&String> = tensor_map.tensors.keys()
        .filter(|n| tensor_map.tensors[*n].is_weight())
        .collect();

    let pb = progress.bar(weight_names.len() as u64, "Computing importance");
    let mut importance = HashMap::new();

    for name in &weight_names {
        let tensor = &tensor_map.tensors[*name];
        let f32_vals = tensor_to_f32_values(tensor);

        let score = if let Some(ref act_mag) = activation_magnitudes {
            // Use activation magnitudes if the tensor's inner dimension matches
            let inner_dim = *tensor.shape.last().unwrap_or(&1);
            if let Some(act) = act_mag.get(*name) {
                // Element-wise: importance = |weight| * activation_magnitude
                // act is per-column (length = inner_dim); broadcast across rows
                let mut total = 0.0f64;
                for (i, &w) in f32_vals.iter().enumerate() {
                    let col = i % inner_dim;
                    let a = if col < act.len() { act[col] } else { 1.0 };
                    total += (w.abs() as f64) * a;
                }
                total / f32_vals.len().max(1) as f64
            } else {
                // Tensor not in activation map — use weight magnitude only
                weight_only_importance(&f32_vals)
            }
        } else {
            weight_only_importance(&f32_vals)
        };

        importance.insert((*name).clone(), score);
        pb.inc(1);
    }

    pb.finish_with_message(format!("Scored {} weight tensors", importance.len()));
    Ok(importance)
}

/// Simple weight-magnitude importance: mean(|W|).
fn weight_only_importance(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().map(|v| v.abs() as f64).sum();
    sum / values.len() as f64
}

/// Attempt to collect per-tensor activation magnitudes.
///
/// ADR-008: the candle-based GPU forward pass has been removed.  Activation
/// capture now requires the mlx-native inference path which is not wired into
/// the quantization pipeline yet.  Returns None so callers fall back to
/// weight-magnitude-only importance scoring.
fn attempt_activation_capture(
    _tensor_map: &TensorMap,
    _metadata: &ModelMetadata,
    _model_dir: &Path,
    _calibration_tokens: usize,
) -> Option<HashMap<String, Vec<f64>>> {
    info!("Apex: using weight-magnitude-only importance (activation capture requires mlx-native serve path)");
    None
}

/// Select optimal GGML K-quant type per tensor based on importance scores.
///
/// Strategy:
/// 1. Rank tensors by importance score (descending).
/// 2. Assign K-quant types to meet target_bpw:
///    - Most important tensors get the highest allowed type.
///    - Least important tensors get the lowest allowed type.
/// 3. Special rules: embedding and lm_head always get max type.
/// 4. Iteratively adjusts the allocation boundary to hit the target bpw.
pub fn select_kquant_types(
    importance: &HashMap<String, f64>,
    config: &ApexConfig,
) -> HashMap<String, String> {
    let min_idx = kquant_index(&config.min_type).unwrap_or(1); // Q3_K_S default
    let max_idx = kquant_index(&config.max_type).unwrap_or(KQUANT_SPECTRUM.len() - 1);

    // Identify which spectrum entries are in the allowed range
    let allowed: Vec<usize> = (min_idx..=max_idx).collect();
    if allowed.is_empty() {
        // Fallback: assign everything to Q4_K_M
        return importance.keys().map(|k| (k.clone(), "Q4_K_M".to_string())).collect();
    }

    // Sort tensors by importance (descending)
    let mut ranked: Vec<(&String, &f64)> = importance.iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    let total_tensors = ranked.len();
    if total_tensors == 0 {
        return HashMap::new();
    }

    // Assign types by partitioning into equal bands across the allowed spectrum.
    // Then adjust the partition boundaries to meet the target bpw.
    let num_types = allowed.len();
    let mut assignments: HashMap<String, String> = HashMap::with_capacity(total_tensors);

    // Initial assignment: spread evenly across allowed types (most important get highest)
    for (i, (name, _score)) in ranked.iter().enumerate() {
        // Map tensor rank to type index: rank 0 -> max_idx, rank last -> min_idx
        let fraction = i as f32 / total_tensors.max(1) as f32;
        let type_offset = (fraction * num_types as f32).min((num_types - 1) as f32) as usize;
        let spectrum_idx = allowed[num_types - 1 - type_offset];
        assignments.insert((*name).clone(), KQUANT_SPECTRUM[spectrum_idx].name.to_string());
    }

    // Special rules: embedding and lm_head always get max type
    let max_type_name = KQUANT_SPECTRUM[max_idx].name;
    for name in importance.keys() {
        if name.contains("embed_tokens") || name.contains("lm_head") {
            assignments.insert(name.clone(), max_type_name.to_string());
        }
    }

    // Compute current average bpw and iteratively adjust
    for _iter in 0..10 {
        let avg_bpw = compute_avg_bpw(&assignments);
        let delta = avg_bpw - config.target_bpw;

        if delta.abs() < 0.1 {
            break;
        }

        // Need to adjust: if too high, downgrade least important; if too low, upgrade most important
        let mut mutable_ranked = ranked.clone();
        if delta > 0.0 {
            // Too many bits — downgrade from the least important end
            mutable_ranked.reverse();
        }

        let mut adjusted = false;
        for (name, _score) in &mutable_ranked {
            let current = assignments.get(*name).cloned().unwrap_or_default();
            let current_idx = kquant_index(&current).unwrap_or(min_idx);

            let new_idx = if delta > 0.0 {
                // Downgrade (lower index = fewer bits)
                if current_idx > min_idx { current_idx - 1 } else { continue }
            } else {
                // Upgrade
                if current_idx < max_idx { current_idx + 1 } else { continue }
            };

            if new_idx >= min_idx && new_idx <= max_idx {
                // Don't touch embed/lm_head
                if name.contains("embed_tokens") || name.contains("lm_head") {
                    continue;
                }
                assignments.insert((*name).clone(), KQUANT_SPECTRUM[new_idx].name.to_string());
                adjusted = true;
                break;
            }
        }

        if !adjusted {
            break;
        }
    }

    debug!(
        avg_bpw = compute_avg_bpw(&assignments),
        target_bpw = config.target_bpw,
        "K-quant type selection complete"
    );

    assignments
}

/// Compute average bits-per-weight from a K-quant type assignment map.
fn compute_avg_bpw(assignments: &HashMap<String, String>) -> f32 {
    if assignments.is_empty() {
        return 0.0;
    }
    let total_bpw: f32 = assignments.values()
        .map(|t| kquant_index(t).map(|i| KQUANT_SPECTRUM[i].bpw).unwrap_or(4.5))
        .sum();
    total_bpw / assignments.len() as f32
}

/// Get the IR bit width for a K-quant type name.
fn ir_bits_for_kquant(kquant_name: &str) -> u8 {
    kquant_index(kquant_name)
        .map(|i| KQUANT_SPECTRUM[i].ir_bits)
        .unwrap_or(4)
}

/// Convert a tensor to f32 values for importance scoring.
fn tensor_to_f32_values(tensor: &TensorRef) -> Vec<f32> {
    use crate::ir::DType;
    match tensor.dtype {
        DType::F32 => tensor.data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F16 => tensor.data.chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        DType::BF16 => tensor.data.chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        _ => Vec::new(),
    }
}

/// Run the full Apex two-pass quantization pipeline.
///
/// Pass 1: Compute importance matrix (uses GPU forward pass if available).
/// Pass 2: For each tensor, quantize using the assigned K-quant type's bit width
///         and set `quant_info.ggml_type` so the GGUF backend writes the correct type.
pub fn run_apex_quantization(
    tensor_map: &TensorMap,
    metadata: &ModelMetadata,
    model_dir: &Path,
    config: &ApexConfig,
    progress: &ProgressReporter,
) -> Result<QuantizedModel> {
    info!(
        target_bpw = config.target_bpw,
        calibration_tokens = config.calibration_tokens,
        min_type = %config.min_type,
        max_type = %config.max_type,
        "Starting Apex quantization"
    );

    // Pass 1: Compute importance matrix
    let importance = compute_importance_matrix(
        tensor_map, metadata, model_dir, config.calibration_tokens, progress,
    ).context("Apex Pass 1: importance matrix computation failed")?;

    info!(scored_tensors = importance.len(), "Apex Pass 1 complete");

    // Select K-quant types based on importance
    let kquant_types = select_kquant_types(&importance, config);

    // Log allocation summary
    let mut type_counts: HashMap<&str, usize> = HashMap::new();
    for t in kquant_types.values() {
        *type_counts.entry(t.as_str()).or_insert(0) += 1;
    }
    for (t, count) in &type_counts {
        info!(kquant_type = t, count = count, "Apex allocation");
    }

    // Pass 2: Quantize each tensor using the assigned type
    let quantizer = StaticQuantizer::new("q4")
        .expect("q4 quantizer should always be valid");

    let total = tensor_map.len() as u64;
    let pb = progress.bar(total, "Apex quantizing");

    let mut quantized_tensors = HashMap::new();
    let mut tensor_names: Vec<&String> = tensor_map.tensors.keys().collect();
    tensor_names.sort();

    for name in tensor_names {
        let tensor = &tensor_map.tensors[name];

        let quantized = if !tensor.is_weight() {
            // Non-weight tensors: preserve at full precision
            let config = LayerQuantConfig {
                bits: 16,
                group_size: 0,
                preserve: true,
            };
            quantizer.quantize_tensor(tensor, &config)
                .map_err(|e| anyhow::anyhow!("Failed to preserve tensor '{}': {}", name, e))?
        } else if let Some(kquant_name) = kquant_types.get(name) {
            // Weight tensor with assigned K-quant type
            let bits = ir_bits_for_kquant(kquant_name);
            let config = LayerQuantConfig {
                bits,
                group_size: 64,
                preserve: false,
            };
            let mut qt = quantizer.quantize_tensor(tensor, &config)
                .map_err(|e| anyhow::anyhow!("Failed to quantize tensor '{}': {}", name, e))?;

            // Set the GGML type override so the GGUF backend uses the correct K-quant type
            qt.quant_info.ggml_type = Some(kquant_name.clone());
            qt.quant_info.method = format!("apex-{}", kquant_name);
            qt
        } else {
            // Weight tensor without importance score (shouldn't happen, but fallback to Q4_K_M)
            let config = LayerQuantConfig {
                bits: 4,
                group_size: 64,
                preserve: false,
            };
            let mut qt = quantizer.quantize_tensor(tensor, &config)
                .map_err(|e| anyhow::anyhow!("Failed to quantize tensor '{}': {}", name, e))?;
            qt.quant_info.ggml_type = Some("Q4_K_M".to_string());
            qt.quant_info.method = "apex-Q4_K_M".to_string();
            qt
        };

        quantized_tensors.insert(name.clone(), quantized);
        pb.inc(1);
    }

    let avg_bpw = compute_avg_bpw(&kquant_types);
    pb.finish_with_message(format!(
        "Apex: {} tensors quantized, avg {:.1} bpw",
        quantized_tensors.len(),
        avg_bpw
    ));

    Ok(QuantizedModel {
        metadata: metadata.clone(),
        tensors: quantized_tensors,
        quant_method: "apex".to_string(),
        group_size: 64,
        bits: avg_bpw.round() as u8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;

    fn make_f16_tensor(name: &str, shape: Vec<usize>, values: &[f32]) -> TensorRef {
        let data: Vec<u8> = values
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        TensorRef {
            name: name.to_string(),
            shape,
            dtype: DType::F16,
            data,
        }
    }

    fn make_test_tensor_map() -> TensorMap {
        let mut map = TensorMap::new();

        // High-magnitude weight (should be more important)
        map.insert(make_f16_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            vec![4, 4],
            &[10.0, -8.0, 5.0, -3.0, 7.0, -6.0, 4.0, -2.0,
              9.0, -7.0, 6.0, -4.0, 8.0, -5.0, 3.0, -1.0],
        ));

        // Low-magnitude weight (should be less important)
        map.insert(make_f16_tensor(
            "model.layers.1.mlp.down_proj.weight",
            vec![4, 4],
            &[0.1, -0.05, 0.02, -0.01, 0.08, -0.04, 0.03, -0.02,
              0.06, -0.03, 0.01, -0.005, 0.04, -0.02, 0.01, -0.005],
        ));

        // Non-weight tensor (should be preserved)
        map.insert(TensorRef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            shape: vec![4],
            dtype: DType::F16,
            data: vec![0u8; 8],
        });

        // Embedding (special rule: always gets max type)
        map.insert(make_f16_tensor(
            "model.embed_tokens.weight",
            vec![4, 4],
            &[1.0; 16],
        ));

        map
    }

    fn make_test_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "TestModel".to_string(),
            model_type: "test".to_string(),
            param_count: 1000,
            hidden_size: 4,
            num_layers: 2,
            layer_types: vec!["attention".to_string()],
            num_attention_heads: 1,
            num_kv_heads: Some(1),
            vocab_size: 100,
            dtype: "float16".to_string(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some(8),
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
        }
    }

    #[test]
    fn test_weight_only_importance() {
        let values = vec![1.0, -2.0, 3.0, -4.0];
        let score = weight_only_importance(&values);
        assert!((score - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_weight_only_importance_empty() {
        let score = weight_only_importance(&[]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_importance_ranking() {
        // Higher magnitude weights should score higher
        let tensor_map = make_test_tensor_map();
        let metadata = make_test_metadata();
        let progress = ProgressReporter::new();

        let importance = compute_importance_matrix(
            &tensor_map,
            &metadata,
            Path::new("/nonexistent"),
            64,
            &progress,
        )
        .unwrap();

        let high = importance.get("model.layers.0.self_attn.q_proj.weight").unwrap();
        let low = importance.get("model.layers.1.mlp.down_proj.weight").unwrap();

        assert!(high > low, "High-magnitude tensor should have higher importance: {} vs {}", high, low);

        // Non-weight tensors should not be in the importance map
        assert!(!importance.contains_key("model.layers.0.input_layernorm.weight"));
    }

    #[test]
    fn test_kquant_index() {
        assert_eq!(kquant_index("Q2_K"), Some(0));
        assert_eq!(kquant_index("Q4_K_M"), Some(4));
        assert_eq!(kquant_index("Q6_K"), Some(7));
        assert_eq!(kquant_index("INVALID"), None);
    }

    #[test]
    fn test_select_kquant_types_basic() {
        let mut importance = HashMap::new();
        importance.insert("model.layers.0.self_attn.q_proj.weight".to_string(), 10.0);
        importance.insert("model.layers.1.mlp.down_proj.weight".to_string(), 1.0);

        let config = ApexConfig {
            target_bpw: 4.5,
            min_type: "Q3_K_S".to_string(),
            max_type: "Q6_K".to_string(),
            ..Default::default()
        };

        let types = select_kquant_types(&importance, &config);
        assert_eq!(types.len(), 2);

        // More important tensor should get a higher type
        let high_type = types.get("model.layers.0.self_attn.q_proj.weight").unwrap();
        let low_type = types.get("model.layers.1.mlp.down_proj.weight").unwrap();

        let high_idx = kquant_index(high_type).unwrap();
        let low_idx = kquant_index(low_type).unwrap();
        assert!(high_idx >= low_idx, "Higher importance should get >= type: {} vs {}", high_type, low_type);
    }

    #[test]
    fn test_select_kquant_types_embed_lm_head_get_max() {
        let mut importance = HashMap::new();
        importance.insert("model.embed_tokens.weight".to_string(), 0.001); // Even low importance
        importance.insert("lm_head.weight".to_string(), 0.001);
        importance.insert("model.layers.0.self_attn.q_proj.weight".to_string(), 100.0);

        let config = ApexConfig {
            target_bpw: 4.5,
            min_type: "Q3_K_S".to_string(),
            max_type: "Q6_K".to_string(),
            ..Default::default()
        };

        let types = select_kquant_types(&importance, &config);
        assert_eq!(types.get("model.embed_tokens.weight").unwrap(), "Q6_K");
        assert_eq!(types.get("lm_head.weight").unwrap(), "Q6_K");
    }

    #[test]
    fn test_select_kquant_types_empty() {
        let importance: HashMap<String, f64> = HashMap::new();
        let config = ApexConfig::default();
        let types = select_kquant_types(&importance, &config);
        assert!(types.is_empty());
    }

    #[test]
    fn test_compute_avg_bpw() {
        let mut assignments = HashMap::new();
        assignments.insert("a".to_string(), "Q4_K_M".to_string()); // 4.5
        assignments.insert("b".to_string(), "Q6_K".to_string());   // 6.5
        let avg = compute_avg_bpw(&assignments);
        assert!((avg - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_ir_bits_for_kquant() {
        assert_eq!(ir_bits_for_kquant("Q2_K"), 2);
        assert_eq!(ir_bits_for_kquant("Q3_K_S"), 4);
        assert_eq!(ir_bits_for_kquant("Q4_K_M"), 4);
        assert_eq!(ir_bits_for_kquant("Q5_K_M"), 8);
        assert_eq!(ir_bits_for_kquant("Q6_K"), 8);
        assert_eq!(ir_bits_for_kquant("UNKNOWN"), 4); // fallback
    }

    #[test]
    fn test_apex_config_default() {
        let config = ApexConfig::default();
        assert_eq!(config.target_bpw, 4.5);
        assert_eq!(config.min_type, "Q3_K_S");
        assert_eq!(config.max_type, "Q6_K");
        assert_eq!(config.calibration_tokens, 512);
    }

    #[test]
    fn test_tensor_to_f32_values() {
        let tensor = make_f16_tensor("test", vec![2, 2], &[1.0, -2.0, 3.0, -4.0]);
        let vals = tensor_to_f32_values(&tensor);
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 1.0).abs() < 0.01);
        assert!((vals[1] + 2.0).abs() < 0.01);
    }

    #[test]
    fn test_full_pipeline_mock() {
        // Test the full pipeline with mock data (no GPU, falls back to weight-only importance)
        let tensor_map = make_test_tensor_map();
        let metadata = make_test_metadata();
        let progress = ProgressReporter::new();
        let config = ApexConfig::default();

        let result = run_apex_quantization(
            &tensor_map,
            &metadata,
            Path::new("/nonexistent"),
            &config,
            &progress,
        )
        .unwrap();

        assert_eq!(result.quant_method, "apex");

        // All tensors should be in the output
        assert_eq!(result.tensors.len(), tensor_map.len());

        // Weight tensors should have ggml_type set
        let q_proj = result.tensors.get("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert!(q_proj.quant_info.ggml_type.is_some(), "Weight tensor should have ggml_type set");
        assert!(q_proj.quant_info.method.starts_with("apex-"), "Method should start with 'apex-'");

        // Non-weight tensors should be preserved
        let norm = result.tensors.get("model.layers.0.input_layernorm.weight").unwrap();
        assert!(norm.quant_info.preserved, "Non-weight tensor should be preserved");
    }
}
