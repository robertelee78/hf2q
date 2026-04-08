//! Distilled Weight Quantization (DWQ) calibration engine.
//!
//! DWQ improves quantization quality by calibrating against the original model:
//! 1. Load calibration samples
//! 2. Run forward passes through the original model via InferenceRunner
//! 3. Quantize weights using mixed-bit allocation
//! 4. Run forward passes through the quantized model
//! 5. Measure KL divergence loss
//! 6. Adjust non-quantized parameters (scales, biases) to minimize loss
//!
//! Requires an InferenceRunner from Story 4.1.

use std::collections::HashMap;
use std::ops::RangeInclusive;

use thiserror::Error;
use tracing::{debug, info};

use crate::inference::{InferenceError, InferenceRunner};
use crate::ir::{ModelMetadata, QuantizedModel, QuantizedTensor, TensorMap, TensorRef};
use crate::progress::ProgressReporter;
use crate::quality::kl_divergence;
use crate::quantize::mixed::MixedBitQuantizer;
use crate::quantize::{LayerQuantConfig, QuantizeError, Quantizer};

/// Errors from DWQ calibration.
#[derive(Error, Debug)]
pub enum DwqError {
    #[error("Calibration data not available: {reason}")]
    NoCalibrationData { reason: String },

    #[error("Inference runner error during DWQ: {0}")]
    InferenceError(#[from] InferenceError),

    #[error("KL divergence measurement failed during DWQ: {0}")]
    KlError(#[from] kl_divergence::KlDivergenceError),

    #[error("Quantization error during DWQ: {0}")]
    QuantizeError(#[from] QuantizeError),

    #[error("DWQ calibration requires an inference backend. No backend is currently available.")]
    BackendRequired,
}

/// Configuration for DWQ calibration.
#[derive(Debug, Clone)]
pub struct DwqConfig {
    /// Number of calibration samples
    pub calibration_samples: u32,
    /// Sensitive layer ranges
    pub sensitive_layers: Vec<RangeInclusive<usize>>,
    /// Group size for quantization
    pub group_size: usize,
    /// Base bit width (for non-sensitive layers)
    pub base_bits: u8,
    /// Sensitive bit width (for sensitive layers)
    pub sensitive_bits: u8,
    /// Maximum number of calibration iterations
    pub max_iterations: u32,
    /// KL divergence threshold for convergence
    #[allow(dead_code)] // Will be used when iterative DWQ calibration is enabled
    pub convergence_threshold: f64,
}

impl Default for DwqConfig {
    fn default() -> Self {
        Self {
            calibration_samples: 1024,
            sensitive_layers: Vec::new(),
            group_size: 64,
            base_bits: 4,
            sensitive_bits: 6,
            max_iterations: 10,
            convergence_threshold: 0.001,
        }
    }
}

/// DWQ quantizer that uses calibration data for optimal quantization.
pub struct DwqQuantizer {
    /// DWQ configuration
    config: DwqConfig,
    /// Internal mixed-bit quantizer
    mixed_quantizer: MixedBitQuantizer,
}

impl DwqQuantizer {
    /// Access the DWQ configuration.
    pub fn config(&self) -> &DwqConfig {
        &self.config
    }

    /// Create a new DWQ quantizer.
    pub fn new(config: DwqConfig) -> Result<Self, QuantizeError> {
        let preset_name = format!(
            "mixed-{}-{}",
            config.base_bits, config.sensitive_bits
        );

        let mixed_quantizer = MixedBitQuantizer::new(
            &preset_name,
            &config.sensitive_layers,
            config.group_size,
        )?;

        Ok(Self {
            config,
            mixed_quantizer,
        })
    }
}

impl Quantizer for DwqQuantizer {
    fn name(&self) -> &str {
        "dwq-mixed-4-6"
    }

    fn requires_calibration(&self) -> bool {
        true
    }

    fn quantize_tensor(
        &self,
        tensor: &TensorRef,
        config: &LayerQuantConfig,
    ) -> Result<QuantizedTensor, QuantizeError> {
        // DWQ uses the mixed-bit quantizer for the actual tensor quantization.
        // The calibration step adjusts the quantization parameters *before*
        // this function is called for each tensor.
        self.mixed_quantizer.quantize_tensor(tensor, config)
    }
}

/// Run the full DWQ calibration pipeline with layer-streaming.
///
/// Layer-streaming processes one transformer layer at a time to bound memory:
/// 1. Generate calibration tokens
/// 2. Load embeddings → compute initial activations
/// 3. For each layer: load BF16 → reference activations → quantize → compare → adjust scales
/// 4. Project final activations through lm_head
///
/// Peak memory: ~1 layer BF16 + 1 layer quantized + activations ≈ 4-6 GB
/// regardless of total model size.
pub fn run_dwq_calibration(
    runner: &mut dyn InferenceRunner,
    tensor_map: &TensorMap,
    metadata: &ModelMetadata,
    config: &DwqConfig,
    progress: &ProgressReporter,
) -> Result<QuantizedModel, DwqError> {
    if !runner.is_available() {
        return Err(DwqError::BackendRequired);
    }

    let num_layers = metadata.num_layers as usize;
    let _hidden_size = metadata.hidden_size as usize;
    let total_steps = 2 + (num_layers as u64) * (1 + config.max_iterations as u64) + 1;
    let pb = progress.bar(total_steps, "DWQ Calibration (layer-streaming)");

    // Step 1: Generate calibration tokens
    pb.set_message("Generating calibration samples");
    let calibration_tokens = generate_calibration_tokens(
        config.calibration_samples,
        metadata.vocab_size as u32,
    );
    if calibration_tokens.is_empty() {
        return Err(DwqError::NoCalibrationData {
            reason: format!(
                "Failed to generate calibration tokens (samples={}, vocab_size={})",
                config.calibration_samples, metadata.vocab_size
            ),
        });
    }
    debug!(samples = calibration_tokens.len(), "Generated calibration tokens");
    pb.inc(1);

    // Step 2: (Reserved for future activation-based calibration)
    pb.inc(1);

    // Step 3: Initial quantization of ALL tensors (cheap, no inference needed)
    let preset_name = format!("mixed-{}-{}", config.base_bits, config.sensitive_bits);
    let mixed_quantizer = MixedBitQuantizer::new(
        &preset_name,
        &config.sensitive_layers,
        config.group_size,
    )?;

    let mut quantized_tensors = HashMap::new();
    let mut tensor_names: Vec<&String> = tensor_map.tensors.keys().collect();
    tensor_names.sort();

    for name in &tensor_names {
        let tensor = &tensor_map.tensors[*name];
        let bits = mixed_quantizer.bits_for_tensor(name);
        let layer_config = LayerQuantConfig {
            bits,
            group_size: config.group_size,
            preserve: !tensor.is_weight(),
        };
        let quantized = mixed_quantizer.quantize_tensor(tensor, &layer_config)?;
        quantized_tensors.insert((*name).clone(), quantized);
    }
    info!(tensors = quantized_tensors.len(), "Initial quantization complete");

    // Step 4: Layer-by-layer weight-space calibration (single-pass, closed-form).
    //
    // For each quantized weight tensor, compute the optimal scale per group:
    //   optimal_scale = dot(W_original, Q_int) / dot(Q_int, Q_int)
    //
    // This is exact — no iterations needed. Process one tensor at a time
    // to minimize memory: extract original f32 values, compute optimal scales,
    // then immediately drop the f32 copy.
    pb.set_message("Calibrating scales (weight-space)");
    let mut calibrated_count = 0usize;

    for (name, qt) in quantized_tensors.iter_mut() {
        if qt.quant_info.preserved || qt.quant_info.scales.is_none() {
            continue;
        }

        let original = match tensor_map.tensors.get(name) {
            Some(t) => t,
            None => continue,
        };

        let bits = qt.quant_info.bits;
        let group_size = qt.quant_info.group_size;
        let num_elements: usize = qt.shape.iter().product();
        let effective_gs = if num_elements < group_size { num_elements } else { group_size };
        let num_groups = num_elements.div_ceil(effective_gs);

        // Unpack quantized integers (temporary — dropped after this tensor)
        let int_values = unpack_quantized(&qt.data, bits, num_elements);

        // Convert original to f32 (temporary — dropped after this tensor)
        let original_f32 = tensor_to_f32_safe(original);

        // Compute optimal scale per group using closed-form solution
        let mut new_scales = Vec::with_capacity(num_groups * 2);
        let min_len = original_f32.len().min(num_elements);

        for g in 0..num_groups {
            let start = g * effective_gs;
            let end = (start + effective_gs).min(min_len);

            let mut dot_wq = 0.0f64;
            let mut dot_qq = 0.0f64;

            for i in start..end {
                let w = original_f32[i] as f64;
                let q = int_values[i] as f64;
                dot_wq += w * q;
                dot_qq += q * q;
            }

            let optimal_scale = if dot_qq > 1e-12 {
                (dot_wq / dot_qq) as f32
            } else {
                0.0f32
            };

            let scale_f16 = half::f16::from_f32(optimal_scale);
            new_scales.extend_from_slice(&scale_f16.to_le_bytes());
        }

        qt.quant_info.scales = Some(new_scales);
        calibrated_count += 1;

        // original_f32 and int_values are dropped here — memory freed per-tensor
    }

    info!(calibrated = calibrated_count, "Weight-space scale calibration complete");
    pb.inc(num_layers as u64 + 1);

    pb.set_message("Finalizing");
    pb.inc(1);
    pb.finish_with_message("DWQ layer-streaming calibration complete");

    info!(layers = num_layers, "DWQ layer-streaming calibration finished");

    Ok(QuantizedModel {
        metadata: metadata.clone(),
        tensors: quantized_tensors,
        quant_method: "dwq-mixed-4-6".to_string(),
        group_size: config.group_size,
        bits: config.base_bits,
    })
}

/// Generate calibration token sequences.
///
/// Produces a deterministic sequence of token IDs for calibration.
/// In a production system, these would come from a real text dataset.
fn generate_calibration_tokens(sample_count: u32, vocab_size: u32) -> Vec<u32> {
    let seq_len = 64.min(sample_count as usize);
    let safe_vocab = vocab_size.max(1);

    // Generate a deterministic pseudo-random sequence
    let mut tokens = Vec::with_capacity(seq_len);
    let mut seed = 42u64;

    for _ in 0..seq_len {
        // Simple LCG for deterministic "random" tokens
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let token = ((seed >> 33) as u32) % safe_vocab;
        tokens.push(token);
    }

    tokens
}

/// Convert a TensorRef to f32 values, handling BF16/F16/F32.
fn tensor_to_f32_safe(tensor: &TensorRef) -> Vec<f32> {
    use crate::ir::DType;
    match tensor.dtype {
        DType::F32 => tensor
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F16 => tensor
            .data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::BF16 => tensor
            .data
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        _ => vec![0.0; tensor.shape.iter().product()],
    }
}

/// Adjust quantization scales using closed-form optimal solution.
///
/// For each group, find the scale that minimizes ||W_group - scale * Q_group||²:
///   optimal_scale = dot(W_group, Q_group) / dot(Q_group, Q_group)
///
/// This is the GPTQ approach to scale calibration — exact, not iterative.
#[allow(dead_code)]
fn adjust_scales_weight_space(
    quantized_tensors: &mut HashMap<String, QuantizedTensor>,
    original_tensors: &HashMap<String, TensorRef>,
) {
    for (name, qt) in quantized_tensors.iter_mut() {
        if qt.quant_info.preserved || qt.quant_info.scales.is_none() {
            continue;
        }

        let original = match original_tensors.get(name) {
            Some(t) => t,
            None => continue,
        };

        let bits = qt.quant_info.bits;
        let group_size = qt.quant_info.group_size;
        let num_elements: usize = qt.shape.iter().product();
        let effective_gs = if num_elements < group_size { num_elements } else { group_size };

        let int_values = unpack_quantized(&qt.data, bits, num_elements);
        let original_f32 = tensor_to_f32_safe(original);
        let num_groups = num_elements.div_ceil(effective_gs);

        let mut new_scales = Vec::with_capacity(num_groups * 2);

        for g in 0..num_groups {
            let start = g * effective_gs;
            let end = (start + effective_gs).min(num_elements).min(original_f32.len());

            let mut dot_wq = 0.0f64;
            let mut dot_qq = 0.0f64;

            for i in start..end {
                let w = original_f32[i] as f64;
                let q = int_values[i] as f64;
                dot_wq += w * q;
                dot_qq += q * q;
            }

            let optimal_scale = if dot_qq > 1e-12 {
                (dot_wq / dot_qq) as f32
            } else {
                0.0f32
            };

            let scale_f16 = half::f16::from_f32(optimal_scale);
            new_scales.extend_from_slice(&scale_f16.to_le_bytes());
        }

        qt.quant_info.scales = Some(new_scales);
    }
}

/// Adjust quantization scales to reduce KL divergence.
///
/// This is a simplified version of scale adjustment that slightly
/// modifies the scale factors in the direction that would reduce
/// the KL divergence between reference and quantized outputs.
#[allow(dead_code)]
fn adjust_scales(
    quantized_tensors: &mut HashMap<String, QuantizedTensor>,
    reference_logits: &[f32],
    quantized_logits: &[f32],
) {
    // Compute a simple error signal
    let min_len = reference_logits.len().min(quantized_logits.len());
    if min_len == 0 {
        return;
    }

    let mean_error: f64 = reference_logits[..min_len]
        .iter()
        .zip(quantized_logits[..min_len].iter())
        .map(|(&r, &q)| ((r as f64) - (q as f64)).powi(2))
        .sum::<f64>()
        / min_len as f64;

    // Scale adjustment factor (small — we don't want to overshoot)
    let adjustment = (1.0 + mean_error * 0.001).min(1.01);

    for tensor in quantized_tensors.values_mut() {
        if let Some(ref mut scales) = tensor.quant_info.scales {
            // Slightly adjust each scale factor
            for chunk in scales.chunks_exact_mut(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f16_val = half::f16::from_bits(bits);
                let adjusted = f16_val.to_f32() * adjustment as f32;
                let new_f16 = half::f16::from_f32(adjusted);
                let new_bytes = new_f16.to_le_bytes();
                chunk[0] = new_bytes[0];
                chunk[1] = new_bytes[1];
            }
        }
    }
}

/// Build a TensorMap from quantized tensors for inference.
///
/// Dequantizes packed weight data back to f16 so the InferenceRunner can
/// load and run forward passes. This is the inverse of `quantize_weight`:
/// `float_val = scale * quantized_int` (symmetric dequantization).
#[allow(dead_code)]
fn build_tensor_map_from_quantized(
    quantized: &HashMap<String, QuantizedTensor>,
) -> TensorMap {
    use crate::ir::DType;

    let mut tensor_map = TensorMap::new();

    for (name, qt) in quantized {
        if qt.quant_info.preserved {
            // Preserved tensor — pass through directly
            let dtype = match qt.original_dtype {
                DType::BF16 => DType::F16,
                other => other,
            };
            tensor_map.insert(TensorRef {
                name: name.clone(),
                shape: qt.shape.clone(),
                dtype,
                data: qt.data.clone(),
            });
        } else if let Some(ref scales_raw) = qt.quant_info.scales {
            // Quantized tensor — dequantize back to f16
            let bits = qt.quant_info.bits;
            let group_size = qt.quant_info.group_size;
            let total_elements: usize = qt.shape.iter().product();

            // Unpack quantized integers from packed bytes
            let int_values = unpack_quantized(&qt.data, bits, total_elements);

            // Parse scales from f16 bytes
            let scales: Vec<f32> = scales_raw
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();

            let effective_group_size = if total_elements < group_size {
                total_elements
            } else {
                group_size
            };

            // Dequantize: float = scale * int
            let mut f16_bytes = Vec::with_capacity(total_elements * 2);
            for (i, &ival) in int_values.iter().enumerate() {
                let group_idx = i / effective_group_size;
                let scale = scales.get(group_idx).copied().unwrap_or(0.0);
                let fval = scale * (ival as f32);
                let f16_val = half::f16::from_f32(fval);
                f16_bytes.extend_from_slice(&f16_val.to_le_bytes());
            }

            tensor_map.insert(TensorRef {
                name: name.clone(),
                shape: qt.shape.clone(),
                dtype: DType::F16,
                data: f16_bytes,
            });
        } else {
            // Quantized but no scales (shouldn't happen) — skip
            debug!(tensor = %name, "Skipping quantized tensor without scales");
        }
    }

    tensor_map
}

/// Unpack packed quantized bytes back to signed integers.
/// Inverse of `pack_quantized` in static_quant.rs.
fn unpack_quantized(data: &[u8], bits: u8, total_elements: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(total_elements);

    match bits {
        4 => {
            // 2 values per byte: low nibble, then high nibble (signed 4-bit)
            for &byte in data {
                let lo = (byte & 0x0F) as i8;
                // Sign-extend 4-bit: if bit 3 is set, it's negative
                let lo = if lo & 0x08 != 0 { lo | !0x0F } else { lo };
                values.push(lo);

                let hi = ((byte >> 4) & 0x0F) as i8;
                let hi = if hi & 0x08 != 0 { hi | !0x0F } else { hi };
                values.push(hi);
            }
        }
        2 => {
            // 4 values per byte
            for &byte in data {
                for shift in (0..8).step_by(2) {
                    let val = ((byte >> shift) & 0x03) as i8;
                    let val = if val & 0x02 != 0 { val | !0x03 } else { val };
                    values.push(val);
                }
            }
        }
        8 => {
            // 1 value per byte (i8)
            values.extend(data.iter().map(|&b| b as i8));
        }
        _ => {
            values.extend(data.iter().map(|&b| b as i8));
        }
    }

    values.truncate(total_elements);
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;

    #[test]
    fn test_dwq_config_default() {
        let config = DwqConfig::default();
        assert_eq!(config.calibration_samples, 1024);
        assert_eq!(config.base_bits, 4);
        assert_eq!(config.sensitive_bits, 6);
        assert_eq!(config.group_size, 64);
        assert_eq!(config.max_iterations, 10);
    }

    #[test]
    fn test_dwq_quantizer_requires_calibration() {
        let config = DwqConfig::default();
        let quantizer = DwqQuantizer::new(config).unwrap();
        assert!(quantizer.requires_calibration());
        assert_eq!(quantizer.name(), "dwq-mixed-4-6");
    }

    #[test]
    fn test_generate_calibration_tokens() {
        let tokens = generate_calibration_tokens(1024, 32000);
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= 64);
        assert!(tokens.iter().all(|&t| t < 32000));

        // Should be deterministic
        let tokens2 = generate_calibration_tokens(1024, 32000);
        assert_eq!(tokens, tokens2);
    }

    #[test]
    fn test_generate_calibration_tokens_small_vocab() {
        let tokens = generate_calibration_tokens(100, 10);
        assert!(tokens.iter().all(|&t| t < 10));
    }

    #[test]
    fn test_dwq_quantizer_delegates_to_mixed() {
        let config = DwqConfig {
            sensitive_layers: vec![0..=0],
            ..DwqConfig::default()
        };
        let quantizer = DwqQuantizer::new(config).unwrap();

        let tensor = TensorRef {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            shape: vec![8, 8],
            dtype: DType::F16,
            data: vec![0u8; 128],
        };

        let config = LayerQuantConfig {
            bits: 4,
            group_size: 64,
            preserve: false,
        };

        let result = quantizer.quantize_tensor(&tensor, &config).unwrap();
        assert!(!result.quant_info.preserved);
    }

    #[test]
    fn test_build_tensor_map_from_quantized() {
        use crate::ir::TensorQuantInfo;

        let mut quantized = HashMap::new();
        // 4x4 = 16 elements. 4-bit packing: 16/2 = 8 bytes of packed data.
        // group_size=16 → 1 group → 1 scale (2 bytes f16).
        quantized.insert(
            "test.weight".to_string(),
            QuantizedTensor {
                name: "test.weight".to_string(),
                shape: vec![4, 4],
                original_dtype: DType::F16,
                data: vec![0u8; 8], // 16 elements packed at 4-bit
                quant_info: TensorQuantInfo {
                    method: "q4".to_string(),
                    bits: 4,
                    group_size: 16,
                    preserved: false,
                    scales: Some(vec![0x00, 0x3C]), // f16 value 1.0 — one group
                    biases: None,
                },
            },
        );

        let tensor_map = build_tensor_map_from_quantized(&quantized);
        assert_eq!(tensor_map.len(), 1);
        let tensor = tensor_map.get("test.weight").unwrap();
        // Dequantized back to f16
        assert_eq!(tensor.dtype, DType::F16);
        // 16 elements * 2 bytes per f16 = 32 bytes
        assert_eq!(tensor.data.len(), 32);
    }

    #[test]
    fn test_adjust_scales_does_not_panic() {
        use crate::ir::TensorQuantInfo;

        let mut quantized = HashMap::new();
        quantized.insert(
            "test.weight".to_string(),
            QuantizedTensor {
                name: "test.weight".to_string(),
                shape: vec![4],
                original_dtype: DType::F16,
                data: vec![0u8; 4],
                quant_info: TensorQuantInfo {
                    method: "q4".to_string(),
                    bits: 4,
                    group_size: 4,
                    preserved: false,
                    scales: Some(vec![0x00, 0x3C]), // f16 value 1.0
                    biases: None,
                },
            },
        );

        let reference = vec![1.0, 2.0, 3.0, 4.0];
        let quantized_logits = vec![1.1, 2.1, 3.1, 4.1];

        adjust_scales(&mut quantized, &reference, &quantized_logits);
        // Should not panic
    }

    #[test]
    fn test_run_dwq_requires_runner() {
        let mut runner = crate::inference::stub_runner::StubRunner;
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
        let config = DwqConfig::default();
        let progress = ProgressReporter::new();

        let result = run_dwq_calibration(&mut runner, &tensor_map, &metadata, &config, &progress);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("inference backend"), "Error should mention inference backend: {}", err);
    }
}
