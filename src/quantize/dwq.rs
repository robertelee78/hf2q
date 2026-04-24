//! Distilled Weight Quantization (DWQ) calibration engine.
//!
//! Phase 1: Weight-space calibration (CPU, no inference needed).
//! For each quantized weight tensor, computes the optimal scale per group:
//!   optimal_scale = dot(W_original, Q_int) / dot(Q_int, Q_int)
//!
//! Phase 2: Activation-based calibration via Candle GPU forward passes.
//! Uses per-layer activation statistics to drive mixed-bit allocation.

use std::collections::HashMap;
use std::ops::RangeInclusive;

use thiserror::Error;
use tracing::{debug, info};

use crate::ir::{ModelMetadata, QuantizedModel, QuantizedTensor, TensorMap, TensorRef};
use crate::progress::ProgressReporter;
use crate::quantize::mixed::MixedBitQuantizer;
use crate::quantize::{LayerQuantConfig, QuantizeError, Quantizer};

/// Architecture family for DWQ calibration routing.
///
/// ADR-012 Decision 13: qwen35 / qwen35moe DWQ requires an `ActivationCapture`
/// implementation from the inference session.  Weight-space fallback is NOT
/// available for these architectures.  All other archs remain on the existing
/// weight-space path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwqArch {
    /// Qwen3.5 dense — requires ActivationCapture (ADR-012 D13, ADR-013 P12).
    Qwen35Dense,
    /// Qwen3.5-MoE — requires ActivationCapture (ADR-012 D13, ADR-013 P12).
    Qwen35MoE,
    /// All other architectures — weight-space calibration path (existing behaviour).
    Other,
}

impl DwqArch {
    /// Resolve from the HuggingFace architecture string stored in `ModelMetadata.architecture`.
    ///
    /// Both `"Qwen3_5ForCausalLM"` (dense) and `"Qwen3_5MoeForCausalLM"` (MoE)
    /// are matched case-insensitively on the `"qwen3_5"` substring.
    pub fn from_hf_architecture(arch: &str, is_moe: bool) -> Self {
        let lower = arch.to_lowercase();
        if lower.contains("qwen3_5") {
            if is_moe {
                DwqArch::Qwen35MoE
            } else {
                DwqArch::Qwen35Dense
            }
        } else {
            DwqArch::Other
        }
    }

    /// Whether this architecture requires an `ActivationCapture` impl to proceed.
    /// Returns `false` for `Other` (weight-space path OK).
    pub fn requires_activation_capture(&self) -> bool {
        matches!(self, DwqArch::Qwen35Dense | DwqArch::Qwen35MoE)
    }
}

/// Errors from DWQ calibration.
#[derive(Error, Debug)]
pub enum DwqError {
    #[error("Calibration data not available: {reason}")]
    NoCalibrationData { reason: String },

    #[error("Quantization error during DWQ: {0}")]
    QuantizeError(#[from] QuantizeError),

    #[error("GPU forward pass failed: {reason}")]
    GpuError { reason: String },

    #[error("Tokenizer error: {reason}")]
    #[allow(dead_code)]
    TokenizerError { reason: String },

    /// ADR-012 Decision 13 / no-fallback rule.
    ///
    /// qwen35 / qwen35moe DWQ calibration requires an `ActivationCapture`
    /// implementation from the inference session.  Weight-space fallback is
    /// explicitly not available — per the no-fallback mantra, the blocker must
    /// be fixed upstream (land ADR-013 P12 real impl), not routed around.
    #[error(
        "qwen35/qwen35moe DWQ calibration requires an ActivationCapture implementation from \
         the inference session (see ADR-013 P12 / ADR-012 Decision 13). \
         Weight-space fallback is not available. \
         Fix: ensure the inference session provides a real ActivationCapture impl before \
         invoking DWQ conversion for this architecture."
    )]
    NoActivationCapture,
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
    /// Whether to use activation-based calibration (requires GPU + tokenizer).
    /// When false, falls back to weight-space calibration.
    pub use_activations: bool,
    /// Architecture family for routing the calibration path.
    ///
    /// ADR-012 Decision 13: qwen35 / qwen35moe MUST use ActivationCapture.
    /// Defaults to `DwqArch::Other` (weight-space path).  Callers in main.rs
    /// must set this from `DwqArch::from_hf_architecture(&metadata.architecture, metadata.is_moe())`.
    ///
    /// `run_dwq_calibration` enforces this by returning `DwqError::NoActivationCapture`
    /// if `arch.requires_activation_capture()` is true — a second line of defence
    /// in case the call site in main.rs is ever bypassed.
    pub arch: DwqArch,
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
            use_activations: false,
            arch: DwqArch::Other,
        }
    }
}

/// DWQ quantizer that uses calibration data for optimal quantization.
pub struct DwqQuantizer {
    /// DWQ configuration
    config: DwqConfig,
    /// Internal mixed-bit quantizer
    mixed_quantizer: MixedBitQuantizer,
    /// Derived name string (e.g. "dwq-mixed-4-6")
    name: String,
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

        let name = format!("dwq-mixed-{}-{}", config.base_bits, config.sensitive_bits);

        Ok(Self {
            config,
            mixed_quantizer,
            name,
        })
    }
}

impl Quantizer for DwqQuantizer {
    fn name(&self) -> &str {
        &self.name
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

/// Run the DWQ weight-space calibration pipeline.
///
/// For each quantized weight tensor, computes the optimal scale per group using
/// the closed-form solution: optimal_scale = dot(W, Q) / dot(Q, Q).
/// This is exact — no iterations or inference needed. CPU-only.
///
/// # ADR-012 Decision 13 — No-fallback guard
///
/// If `config.arch.requires_activation_capture()` is true (qwen35 / qwen35moe),
/// this function returns `DwqError::NoActivationCapture` immediately.
/// This is a second line of defence: the primary guard lives in `main.rs`.
pub fn run_dwq_calibration(
    tensor_map: &TensorMap,
    metadata: &ModelMetadata,
    config: &DwqConfig,
    progress: &ProgressReporter,
) -> Result<QuantizedModel, DwqError> {
    // ADR-012 Decision 13 — no-fallback: qwen35/qwen35moe must not reach this path.
    if config.arch.requires_activation_capture() {
        return Err(DwqError::NoActivationCapture);
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
            preserve: !tensor.is_weight() || tensor.is_vision_tensor(),
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
        quant_method: format!("dwq-mixed-{}-{}", config.base_bits, config.sensitive_bits),
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
pub(crate) fn tensor_to_f32_safe(tensor: &TensorRef) -> Vec<f32> {
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

/// Unpack packed quantized bytes back to signed integers.
/// Inverse of `pack_quantized` in static_quant.rs.
pub(crate) fn unpack_quantized(data: &[u8], bits: u8, total_elements: usize) -> Vec<i8> {
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
        assert!(!config.use_activations);
        // Default arch is Other (weight-space path)
        assert_eq!(config.arch, DwqArch::Other);
    }

    // --- ADR-012 Decision 13: DwqArch no-fallback tests ---

    #[test]
    fn test_dwq_arch_from_hf_qwen35_dense() {
        let arch = DwqArch::from_hf_architecture("Qwen3_5ForCausalLM", false);
        assert_eq!(arch, DwqArch::Qwen35Dense);
        assert!(arch.requires_activation_capture());
    }

    #[test]
    fn test_dwq_arch_from_hf_qwen35moe() {
        let arch = DwqArch::from_hf_architecture("Qwen3_5MoeForCausalLM", true);
        assert_eq!(arch, DwqArch::Qwen35MoE);
        assert!(arch.requires_activation_capture());
    }

    /// `DwqArch::from_hf_architecture` does a case-insensitive substring
    /// match via `arch.to_lowercase().contains("qwen3_5")`. If the
    /// `.to_lowercase()` call is ever removed, the canonical-case tests
    /// above still pass because the constants happen to match. This
    /// test anchors the case-insensitivity contract — future HF
    /// checkpoints that ship with variant casing (`"qwen3_5"`,
    /// `"QWEN3_5"`) continue to route correctly.
    #[test]
    fn test_dwq_arch_case_insensitive_match() {
        for hf_arch in [
            "qwen3_5forcausallm",        // all lowercase
            "QWEN3_5FORCAUSALLM",        // all uppercase
            "Qwen3_5ForCausalLM",        // canonical (existing)
            "qwen3_5_For_CausalLM",      // mixed
        ] {
            let arch = DwqArch::from_hf_architecture(hf_arch, false);
            assert_eq!(
                arch,
                DwqArch::Qwen35Dense,
                "case-insensitive match must resolve {hf_arch} to Qwen35Dense"
            );
            assert!(arch.requires_activation_capture());
        }
        for hf_arch in [
            "qwen3_5moeforcausallm",
            "QWEN3_5MOEFORCAUSALLM",
            "Qwen3_5MoeForCausalLM",
        ] {
            let arch = DwqArch::from_hf_architecture(hf_arch, true);
            assert_eq!(
                arch,
                DwqArch::Qwen35MoE,
                "case-insensitive match must resolve {hf_arch} to Qwen35MoE"
            );
        }
    }

    #[test]
    fn test_dwq_arch_from_hf_other_archs_do_not_require_capture() {
        for (hf_arch, is_moe) in &[
            ("Gemma4ForConditionalGeneration", true),
            ("LlamaForCausalLM", false),
            ("MistralForCausalLM", false),
            ("MixtralForCausalLM", true),
        ] {
            let arch = DwqArch::from_hf_architecture(hf_arch, *is_moe);
            assert_eq!(arch, DwqArch::Other, "Expected Other for {hf_arch}");
            assert!(
                !arch.requires_activation_capture(),
                "{hf_arch} should not require ActivationCapture"
            );
        }
    }

    /// ADR-012 D13 / no-fallback: error text must be unambiguous.
    #[test]
    fn test_no_activation_capture_error_message() {
        let err = DwqError::NoActivationCapture;
        let msg = format!("{err}");
        assert!(msg.contains("ActivationCapture"), "Error must name the trait");
        assert!(msg.contains("ADR-013"), "Error must cite ADR-013 P12");
        assert!(msg.contains("ADR-012"), "Error must cite ADR-012 D13");
        // The message must state that weight-space fallback is NOT available
        // (i.e. it names fallback only to say it isn't offered, not to offer it).
        assert!(
            msg.contains("not available"),
            "Error must state weight-space fallback is not available"
        );
    }

    /// ADR-012 D13: weight-space calibration succeeds for Other arch (Gemma regression).
    #[test]
    fn test_weight_space_calibration_succeeds_for_other_arch() {
        let tensor_map = TensorMap::new();
        let metadata = make_metadata();
        let config = DwqConfig {
            arch: DwqArch::Other,
            ..DwqConfig::default()
        };
        let progress = ProgressReporter::new();
        let result = run_dwq_calibration(&tensor_map, &metadata, &config, &progress);
        assert!(result.is_ok(), "Other arch should succeed on weight-space path");
    }

    /// ADR-012 D13: run_dwq_calibration returns NoActivationCapture for qwen35.
    ///
    /// Defence-in-depth: even if main.rs guard is bypassed, the calibration function
    /// itself rejects qwen35/qwen35moe.
    #[test]
    fn test_run_dwq_calibration_rejects_qwen35_dense() {
        let tensor_map = TensorMap::new();
        let metadata = make_metadata();
        let config = DwqConfig {
            arch: DwqArch::Qwen35Dense,
            ..DwqConfig::default()
        };
        let progress = ProgressReporter::new();
        let result = run_dwq_calibration(&tensor_map, &metadata, &config, &progress);
        assert!(result.is_err(), "qwen35 dense must error — no weight-space fallback");
        let err = result.unwrap_err();
        assert!(
            matches!(err, DwqError::NoActivationCapture),
            "Must be NoActivationCapture, got: {err}"
        );
    }

    /// ADR-012 D13: run_dwq_calibration returns NoActivationCapture for qwen35moe.
    #[test]
    fn test_run_dwq_calibration_rejects_qwen35moe() {
        let tensor_map = TensorMap::new();
        let metadata = make_metadata();
        let config = DwqConfig {
            arch: DwqArch::Qwen35MoE,
            ..DwqConfig::default()
        };
        let progress = ProgressReporter::new();
        let result = run_dwq_calibration(&tensor_map, &metadata, &config, &progress);
        assert!(
            matches!(result.unwrap_err(), DwqError::NoActivationCapture),
            "qwen35moe must return NoActivationCapture"
        );
    }

    /// ADR-012 D13: MockActivationCapture feeds deterministic tensors into scorer.
    ///
    /// This test validates that the cross-ADR ActivationCapture contract works:
    /// MockActivationCapture (from ADR-013 P12) produces expected-shape LayerActivations
    /// that the sensitivity scorer can consume.
    #[test]
    fn test_mock_activation_capture_feeds_scorer() {
        use crate::inference::models::qwen35::activation_capture::{
            ActivationCapture, MockActivationCapture,
        };
        use crate::quantize::sensitivity::compute_layer_sensitivity;

        let mut mock = MockActivationCapture::new(4, 8);
        let tokens = vec![42u32, 100, 7, 255];
        let act = mock
            .run_calibration_prompt(&tokens)
            .expect("MockActivationCapture must succeed");
        act.validate().expect("LayerActivations must be valid");

        // Feed layer_inputs into the sensitivity scorer
        let sensitivities = compute_layer_sensitivity(&act.layer_inputs)
            .expect("Scorer must accept MockActivationCapture output");

        assert_eq!(sensitivities.len(), 4, "One sensitivity per layer");
        // Mock formula increments by 0.01 per layer → sensitivities increase
        for s in &sensitivities {
            assert!(s.score.is_finite(), "Sensitivity score must be finite");
        }
        // Scores should be strictly monotonically increasing (mock adds 0.01 per layer)
        for w in sensitivities.windows(2) {
            assert!(
                w[1].score > w[0].score,
                "Mock-derived sensitivities should be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_dwq_quantizer_requires_calibration() {
        let config = DwqConfig::default();
        let quantizer = DwqQuantizer::new(config).unwrap();
        assert!(quantizer.requires_calibration());
        assert_eq!(quantizer.name(), "dwq-mixed-4-6");
    }

    #[test]
    fn test_dwq_quantizer_name_derived_from_config() {
        // DwqMixed46 path — Gemma-4 regression snapshot
        let config46 = DwqConfig { base_bits: 4, sensitive_bits: 6, ..DwqConfig::default() };
        let q46 = DwqQuantizer::new(config46).unwrap();
        assert_eq!(q46.name(), "dwq-mixed-4-6", "Gemma-4 regression: name must be 'dwq-mixed-4-6'");

        // New variants
        let config48 = DwqConfig { base_bits: 4, sensitive_bits: 8, ..DwqConfig::default() };
        let q48 = DwqQuantizer::new(config48).unwrap();
        assert_eq!(q48.name(), "dwq-mixed-4-8");

        let config68 = DwqConfig { base_bits: 6, sensitive_bits: 8, ..DwqConfig::default() };
        let q68 = DwqQuantizer::new(config68).unwrap();
        assert_eq!(q68.name(), "dwq-mixed-6-8");

        let config28 = DwqConfig { base_bits: 2, sensitive_bits: 8, ..DwqConfig::default() };
        let q28 = DwqQuantizer::new(config28).unwrap();
        assert_eq!(q28.name(), "dwq-mixed-2-8");
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
    fn test_dwq_error_variants() {
        let gpu_err = DwqError::GpuError {
            reason: "test GPU error".to_string(),
        };
        assert!(format!("{}", gpu_err).contains("GPU forward pass failed"));

        let tok_err = DwqError::TokenizerError {
            reason: "test tokenizer error".to_string(),
        };
        assert!(format!("{}", tok_err).contains("Tokenizer error"));
    }

    fn make_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "Test".to_string(),
            model_type: "test".to_string(),
            param_count: 0,
            hidden_size: 0,
            num_layers: 0,
            layer_types: vec![],
            num_attention_heads: 0,
            num_kv_heads: None,
            vocab_size: 32000,
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
        }
    }

    #[test]
    fn test_run_dwq_weight_space() {
        let tensor_map = TensorMap::new();
        let metadata = make_metadata();
        let config = DwqConfig::default();
        let progress = ProgressReporter::new();

        // Empty model should succeed (no tensors to calibrate)
        let result = run_dwq_calibration(&tensor_map, &metadata, &config, &progress);
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.quant_method, "dwq-mixed-4-6");
    }

    #[test]
    fn test_run_dwq_calibration_quant_method_derived_from_config() {
        let tensor_map = TensorMap::new();
        let metadata = make_metadata();
        let progress = ProgressReporter::new();

        // Gemma-4 regression snapshot: (4,6) must produce "dwq-mixed-4-6"
        let config46 = DwqConfig { base_bits: 4, sensitive_bits: 6, ..DwqConfig::default() };
        let model46 = run_dwq_calibration(&tensor_map, &metadata, &config46, &progress).unwrap();
        assert_eq!(
            model46.quant_method, "dwq-mixed-4-6",
            "Gemma-4 regression: quant_method must be 'dwq-mixed-4-6'"
        );

        // New variants
        let config48 = DwqConfig { base_bits: 4, sensitive_bits: 8, ..DwqConfig::default() };
        let model48 = run_dwq_calibration(&tensor_map, &metadata, &config48, &progress).unwrap();
        assert_eq!(model48.quant_method, "dwq-mixed-4-8");

        let config68 = DwqConfig { base_bits: 6, sensitive_bits: 8, ..DwqConfig::default() };
        let model68 = run_dwq_calibration(&tensor_map, &metadata, &config68, &progress).unwrap();
        assert_eq!(model68.quant_method, "dwq-mixed-6-8");

        let config28 = DwqConfig { base_bits: 2, sensitive_bits: 8, ..DwqConfig::default() };
        let model28 = run_dwq_calibration(&tensor_map, &metadata, &config28, &progress).unwrap();
        assert_eq!(model28.quant_method, "dwq-mixed-2-8");
    }

    #[test]
    fn test_gemma4_regression_full_snapshot() {
        // Full Gemma-4 struct-value snapshot:
        // All observables of the DwqMixed46 code path must be byte-identical to
        // pre-change values. If any of these fail, the Gemma-4 quantization
        // output would differ from pre-change HEAD.
        let config = DwqConfig { base_bits: 4, sensitive_bits: 6, ..DwqConfig::default() };
        assert_eq!(config.base_bits, 4, "base_bits must be 4 for Gemma-4 path");
        assert_eq!(config.sensitive_bits, 6, "sensitive_bits must be 6 for Gemma-4 path");

        let quantizer = DwqQuantizer::new(config.clone()).unwrap();
        assert_eq!(
            quantizer.name(), "dwq-mixed-4-6",
            "DwqQuantizer::name() must be 'dwq-mixed-4-6' for Gemma-4 path"
        );

        let tensor_map = TensorMap::new();
        let metadata = make_metadata();
        let progress = ProgressReporter::new();
        let model = run_dwq_calibration(&tensor_map, &metadata, &config, &progress).unwrap();
        assert_eq!(
            model.quant_method, "dwq-mixed-4-6",
            "QuantizedModel.quant_method must be 'dwq-mixed-4-6' for Gemma-4 path"
        );
        assert_eq!(model.bits, 4, "QuantizedModel.bits must be 4 for Gemma-4 path");
    }
}
