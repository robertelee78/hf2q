//! Activation-based DWQ calibration via Candle GPU forward passes.
//!
//! Extends the weight-space DWQ calibration engine with activation-based
//! sensitivity analysis. Uses per-layer hidden-state statistics to drive
//! a smooth mixed-bit allocation, then applies weight-space scale refinement.

use std::collections::HashMap;
use std::path::Path;

use tracing::{debug, info};

use crate::ir::{ModelMetadata, QuantizedModel, TensorMap};
use crate::progress::ProgressReporter;
use crate::quantize::mixed::MixedBitQuantizer;
use crate::quantize::sensitivity;
use crate::quantize::{LayerQuantConfig, Quantizer};

use super::dwq::{DwqConfig, DwqError};

/// Run activation-based DWQ calibration using GPU forward passes.
///
/// 1. Load tokenizer and encode calibration text
/// 2. Load model into GPU via Candle TransformerForward
/// 3. Run forward pass on calibration text, capture per-layer activations
/// 4. Compute per-layer sensitivity from activation variance and magnitude
/// 5. Use sensitivity scores to drive smooth mixed-bit allocation
/// 6. Quantize with sensitivity-aware bit assignment
/// 7. Apply weight-space scale refinement on top
///
/// Falls back to weight-space calibration if GPU or tokenizer is unavailable.
pub fn run_dwq_activation_calibration(
    tensor_map: &TensorMap,
    metadata: &ModelMetadata,
    config: &DwqConfig,
    model_dir: &Path,
    progress: &ProgressReporter,
) -> Result<QuantizedModel, DwqError> {
    let num_layers = metadata.num_layers as usize;
    let total_steps = 5 + num_layers as u64;
    let pb = progress.bar(total_steps, "DWQ Activation Calibration");

    // Step 1: Load tokenizer
    pb.set_message("Loading tokenizer");
    let tokenizer = match crate::gpu::tokenizer::load_tokenizer(model_dir) {
        Ok(t) => t,
        Err(e) => {
            return Err(DwqError::TokenizerError {
                reason: format!("{}", e),
            });
        }
    };
    pb.inc(1);

    // Step 2: Encode calibration text
    pb.set_message("Encoding calibration text");
    let cal_text = crate::gpu::tokenizer::default_calibration_text();
    let token_ids = crate::gpu::tokenizer::encode_calibration_text(&tokenizer, cal_text)
        .map_err(|e| DwqError::TokenizerError {
            reason: format!("Failed to encode calibration text: {}", e),
        })?;

    if token_ids.is_empty() {
        return Err(DwqError::NoCalibrationData {
            reason: "Tokenizer produced empty token sequence from calibration text".to_string(),
        });
    }
    info!(tokens = token_ids.len(), "Calibration text encoded");
    pb.inc(1);

    // Step 3: Select GPU device and load model
    pb.set_message("Loading model onto GPU");
    let (device, gpu_kind) = crate::gpu::select_device().map_err(|e| DwqError::GpuError {
        reason: format!("Device selection failed: {}", e),
    })?;
    info!(device = %gpu_kind, "GPU device selected for activation calibration");

    let transformer =
        crate::gpu::forward::TransformerForward::load(tensor_map, metadata, &device).map_err(
            |e| DwqError::GpuError {
                reason: format!("Failed to load transformer: {}", e),
            },
        )?;
    pb.inc(1);

    // Step 4: Run forward pass with activation capture
    pb.set_message("Running forward pass (capturing activations)");
    let output =
        transformer
            .forward_with_activations(&token_ids)
            .map_err(|e| DwqError::GpuError {
                reason: format!("Forward pass failed: {}", e),
            })?;

    let activations = output.hidden_states.ok_or_else(|| DwqError::GpuError {
        reason: "Forward pass did not capture hidden states".to_string(),
    })?;
    info!(
        layers_captured = activations.len(),
        "Activation capture complete"
    );
    pb.inc(1);

    // Step 5: Compute per-layer sensitivity scores
    pb.set_message("Computing layer sensitivity");
    let layer_sensitivity =
        sensitivity::compute_layer_sensitivity(&activations).map_err(|e| DwqError::GpuError {
            reason: format!("Sensitivity computation failed: {}", e),
        })?;

    // Log sensitivity distribution
    if !layer_sensitivity.is_empty() {
        let min_score = layer_sensitivity
            .iter()
            .map(|s| s.score)
            .fold(f64::INFINITY, f64::min);
        let max_score = layer_sensitivity
            .iter()
            .map(|s| s.score)
            .fold(f64::NEG_INFINITY, f64::max);
        info!(
            layers = layer_sensitivity.len(),
            min_score = format!("{:.4}", min_score),
            max_score = format!("{:.4}", max_score),
            "Layer sensitivity distribution"
        );
    }

    // Step 6: Allocate bits per layer based on sensitivity
    let per_layer_bits = sensitivity::allocate_bits_by_sensitivity(
        &layer_sensitivity,
        config.base_bits,
        config.sensitive_bits,
    );

    for (i, &bits) in per_layer_bits.iter().enumerate() {
        debug!(layer = i, bits = bits, "Sensitivity-driven bit allocation");
    }
    pb.inc(1);

    // Step 7: Quantize all tensors using per-layer bit allocation
    pb.set_message("Quantizing with sensitivity-driven bits");

    // Build a mapping from layer index to bit width
    let layer_bits_map: HashMap<usize, u8> = per_layer_bits
        .iter()
        .enumerate()
        .map(|(i, &b)| (i, b))
        .collect();

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

        // Determine bit width: use sensitivity-driven allocation for weight tensors
        let bits = if tensor.is_weight() {
            extract_layer_index(name)
                .and_then(|idx| layer_bits_map.get(&idx).copied())
                .unwrap_or(config.base_bits)
        } else {
            // Non-weight tensors (norms, biases) are preserved
            config.sensitive_bits
        };

        let layer_config = LayerQuantConfig {
            bits,
            group_size: config.group_size,
            preserve: !tensor.is_weight(),
        };

        let quantized = mixed_quantizer.quantize_tensor(tensor, &layer_config)?;
        quantized_tensors.insert((*name).clone(), quantized);
    }

    info!(
        tensors = quantized_tensors.len(),
        "Activation-based quantization complete"
    );

    // Step 8: Apply weight-space scale calibration on top
    pb.set_message("Calibrating scales (weight-space refinement)");
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
        let effective_gs = if num_elements < group_size {
            num_elements
        } else {
            group_size
        };
        let num_groups = num_elements.div_ceil(effective_gs);

        let int_values = super::dwq::unpack_quantized(&qt.data, bits, num_elements);
        let original_f32 = super::dwq::tensor_to_f32_safe(original);

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
    }

    info!(
        calibrated = calibrated_count,
        "Weight-space scale refinement complete"
    );

    pb.set_message("Finalizing");
    pb.inc(num_layers as u64);
    pb.finish_with_message("DWQ activation-based calibration complete");

    info!(
        layers = num_layers,
        "DWQ activation-based calibration finished"
    );

    Ok(QuantizedModel {
        metadata: metadata.clone(),
        tensors: quantized_tensors,
        quant_method: "dwq-mixed-4-6".to_string(),
        group_size: config.group_size,
        bits: config.base_bits,
    })
}

/// Extract the layer index from a tensor name.
///
/// Looks for patterns like ".layers.N." and returns N.
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
}
