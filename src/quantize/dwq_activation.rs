//! Activation-aware DWQ calibration (ADR-012 Decision 13 + P9b.3a).
//!
//! Replaces the ADR-008 stub that returned `DwqError::GpuError` with a real
//! implementation that consumes any [`ActivationCapture`] impl. Required for
//! qwen35 / qwen35moe per Decision 13's no-fallback rule — those architectures
//! do not have a valid weight-space-only calibration path.
//!
//! # Pipeline
//!
//! 1. Generate calibration tokens (`generate_calibration_tokens`, shared with
//!    the weight-space path).
//! 2. Run a single forward pass via `ActivationCapture::run_calibration_prompt`
//!    to capture per-layer residual streams.
//! 3. Compute per-layer sensitivity from the captured `layer_inputs` via
//!    [`compute_layer_sensitivity`].
//! 4. Allocate a base/sensitive bit choice per layer via
//!    [`allocate_bits_by_sensitivity`] using the preset's `(base_bits,
//!    sensitive_bits)` pair, then derive the `Vec<RangeInclusive<usize>>`
//!    of layers that should receive `sensitive_bits`.
//! 5. Build a derived `DwqConfig` whose `sensitive_layers` field is the
//!    activation-derived range list (replacing whatever the user passed).
//! 6. Delegate to [`run_dwq_calibration_internal`] for the actual
//!    initial-quant + closed-form scale calibration. The internal entry
//!    bypasses the Decision-13 no-fallback guard because the contract has
//!    been satisfied (capture has just run).
//!
//! # Shape contract
//!
//! `LayerActivations.layer_inputs.len() == metadata.num_layers`. If they
//! diverge (mock misconfigured / model-config mismatch) we error out
//! rather than truncating. This is a load-bearing invariant that
//! `LayerActivations::validate` already enforces — we re-check it here so
//! the message is actionable at the calibration call site.

use std::ops::RangeInclusive;

use tracing::{debug, info};

use crate::inference::models::qwen35::activation_capture::ActivationCapture;
use crate::ir::{ModelMetadata, QuantizedModel, TensorMap};
use crate::progress::ProgressReporter;

use super::dwq::{
    generate_calibration_tokens, run_dwq_calibration_internal, DwqConfig, DwqError,
};
use super::sensitivity::{allocate_bits_by_sensitivity, compute_layer_sensitivity};

/// Run activation-aware DWQ calibration.
///
/// `capture` is the runtime-supplied [`ActivationCapture`] — typically a
/// `RealActivationCapture` constructed from an intermediate F16 GGUF (per
/// ADR-012 P9b two-pass conversion). For tests, `MockActivationCapture`
/// works.
///
/// **No fallback.** If capture, sensitivity, or downstream calibration
/// fails, return the error. qwen35 / qwen35moe have no valid weight-space
/// fallback per Decision 13.
///
/// Wired through `src/main.rs` for qwen35/qwen35moe DWQ as of P9b.3b
/// (commit landing this fn into the convert-pipeline two-pass branch).
pub fn run_dwq_activation_calibration(
    tensor_map: &TensorMap,
    metadata: &ModelMetadata,
    config: &DwqConfig,
    capture: &mut dyn ActivationCapture,
    progress: &ProgressReporter,
) -> Result<QuantizedModel, DwqError> {
    info!(
        arch = %metadata.architecture,
        layers = metadata.num_layers,
        base_bits = config.base_bits,
        sensitive_bits = config.sensitive_bits,
        "Activation-aware DWQ calibration starting"
    );

    // Step 1: generate calibration tokens (deterministic — same path as
    // weight-space calibration).
    let calibration_tokens =
        generate_calibration_tokens(config.calibration_samples, metadata.vocab_size as u32);
    if calibration_tokens.is_empty() {
        return Err(DwqError::NoCalibrationData {
            reason: format!(
                "Failed to generate calibration tokens (samples={}, vocab_size={})",
                config.calibration_samples, metadata.vocab_size
            ),
        });
    }
    debug!(
        samples = calibration_tokens.len(),
        "Generated activation-calibration tokens"
    );

    // Step 2: capture activations.
    let activations = capture
        .run_calibration_prompt(&calibration_tokens)
        .map_err(|e| DwqError::GpuError {
            reason: format!("ActivationCapture::run_calibration_prompt failed: {e}"),
        })?;
    activations.validate().map_err(|e| DwqError::GpuError {
        reason: format!("LayerActivations shape invariant: {e}"),
    })?;
    if (activations.num_layers as usize) != (metadata.num_layers as usize) {
        return Err(DwqError::GpuError {
            reason: format!(
                "LayerActivations.num_layers ({}) != metadata.num_layers ({}); \
                 capture and config disagree on model depth",
                activations.num_layers, metadata.num_layers
            ),
        });
    }

    // Step 3: per-layer sensitivity (variance × log2 magnitude).
    let sensitivities = compute_layer_sensitivity(&activations.layer_inputs).map_err(|e| {
        DwqError::GpuError {
            reason: format!("compute_layer_sensitivity: {e}"),
        }
    })?;

    // Step 4: bit allocation → "sensitive" layer index list. The
    // `allocate_bits_by_sensitivity` helper interpolates within
    // `[base_bits, sensitive_bits]`; we collapse anything strictly above
    // the midpoint to "sensitive". For the bit-pair presets (4-6, 4-8,
    // 6-8, 2-8) this gives us a clean two-level split driven by activation
    // magnitudes rather than user-supplied `--sensitive-layers`.
    let allocated = allocate_bits_by_sensitivity(
        &sensitivities,
        config.base_bits,
        config.sensitive_bits,
    );
    let midpoint = (config.base_bits as f32 + config.sensitive_bits as f32) / 2.0;
    let sensitive_indices: Vec<usize> = allocated
        .iter()
        .enumerate()
        .filter(|(_, &b)| (b as f32) > midpoint)
        .map(|(i, _)| i)
        .collect();

    info!(
        sensitive_count = sensitive_indices.len(),
        total = allocated.len(),
        "Activation-derived bit allocation: {} of {} layers above {:.1}-bit midpoint",
        sensitive_indices.len(),
        allocated.len(),
        midpoint
    );

    let derived_sensitive_ranges = indices_to_ranges(&sensitive_indices);

    // Step 5: build a derived DwqConfig with the activation-derived
    // sensitive layers. Everything else (group_size, bit pair, arch) is
    // preserved.
    let derived_config = DwqConfig {
        sensitive_layers: derived_sensitive_ranges,
        ..config.clone()
    };

    // Step 6: delegate to the internal calibration entry (bypasses the
    // Decision-13 no-fallback guard because the contract has just been
    // satisfied).
    run_dwq_calibration_internal(tensor_map, metadata, &derived_config, progress)
}

/// Convert a sorted (or unsorted) index list into a minimum-cardinality
/// `Vec<RangeInclusive<usize>>` covering exactly those indices.
///
/// Adjacent indices coalesce: `[3, 4, 5, 9]` → `[3..=5, 9..=9]`.
/// Duplicates are deduplicated.
#[allow(dead_code)]
fn indices_to_ranges(indices: &[usize]) -> Vec<RangeInclusive<usize>> {
    if indices.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<usize> = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut ranges = Vec::new();
    let mut start = sorted[0];
    let mut end = sorted[0];
    for &i in &sorted[1..] {
        if i == end + 1 {
            end = i;
        } else {
            ranges.push(start..=end);
            start = i;
            end = i;
        }
    }
    ranges.push(start..=end);
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::activation_capture::MockActivationCapture;
    use crate::ir::{DType, TensorRef};
    use crate::quantize::dwq::DwqArch;

    fn tiny_metadata(num_layers: u32, hidden_size: u32) -> ModelMetadata {
        ModelMetadata {
            architecture: "TestArch".into(),
            model_type: "test".into(),
            param_count: 1_000_000,
            hidden_size: hidden_size as u64,
            num_layers,
            layer_types: (0..num_layers).map(|_| "attention".into()).collect(),
            num_attention_heads: 4,
            num_kv_heads: Some(4),
            vocab_size: 256,
            dtype: "float16".into(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some((hidden_size as u64) * 2),
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: Some(hidden_size / 4),
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

    fn tiny_tensor_map(num_layers: u32, hidden_size: u32) -> TensorMap {
        let mut tm = TensorMap::new();
        let h = hidden_size as usize;
        for layer in 0..num_layers {
            // One weight tensor per layer; quantization will route each
            // through MixedBitQuantizer (base or sensitive bits depending on
            // activation-derived sensitive_indices).
            let name = format!("model.layers.{}.self_attn.q_proj.weight", layer);
            tm.insert(TensorRef {
                name,
                shape: vec![h, h],
                dtype: DType::F16,
                data: vec![0u8; h * h * 2],
            });
            // One norm (preserved).
            let norm_name = format!("model.layers.{}.input_layernorm.weight", layer);
            tm.insert(TensorRef {
                name: norm_name,
                shape: vec![h],
                dtype: DType::F16,
                data: vec![0u8; h * 2],
            });
        }
        // Embeddings + output (always preserved).
        tm.insert(TensorRef {
            name: "model.embed_tokens.weight".into(),
            shape: vec![256, h],
            dtype: DType::F16,
            data: vec![0u8; 256 * h * 2],
        });
        tm
    }

    #[test]
    fn indices_to_ranges_groups_adjacent() {
        assert_eq!(indices_to_ranges(&[]), Vec::<RangeInclusive<usize>>::new());
        assert_eq!(indices_to_ranges(&[3]), vec![3..=3]);
        assert_eq!(indices_to_ranges(&[3, 4, 5]), vec![3..=5]);
        assert_eq!(indices_to_ranges(&[3, 4, 5, 9]), vec![3..=5, 9..=9]);
        assert_eq!(
            indices_to_ranges(&[1, 2, 4, 5, 7, 9, 10, 11]),
            vec![1..=2, 4..=5, 7..=7, 9..=11]
        );
        // Unsorted input
        assert_eq!(indices_to_ranges(&[5, 3, 4]), vec![3..=5]);
        // Duplicate input
        assert_eq!(indices_to_ranges(&[3, 3, 4, 4, 5]), vec![3..=5]);
    }

    #[test]
    fn activation_calibration_succeeds_with_mock() {
        let num_layers = 4u32;
        let hidden_size = 16u32;
        let metadata = tiny_metadata(num_layers, hidden_size);
        let tensor_map = tiny_tensor_map(num_layers, hidden_size);
        let mut mock = MockActivationCapture::new(num_layers, hidden_size);
        let progress = ProgressReporter::new();
        let config = DwqConfig {
            calibration_samples: 4,
            base_bits: 4,
            sensitive_bits: 6,
            arch: DwqArch::Other,
            ..DwqConfig::default()
        };

        let model = run_dwq_activation_calibration(
            &tensor_map,
            &metadata,
            &config,
            &mut mock,
            &progress,
        )
        .expect("activation calibration should succeed with mock");

        // Output model must contain at least one tensor per input tensor.
        assert!(model.tensors.len() >= tensor_map.tensors.len());
        // Method tag should reflect the bit pair.
        assert_eq!(model.quant_method, "dwq-mixed-4-6");
    }

    #[test]
    fn activation_calibration_rejects_layer_mismatch() {
        // Mock with 3 layers, metadata with 4 — must error rather than
        // silently truncate.
        let metadata = tiny_metadata(4, 16);
        let tensor_map = tiny_tensor_map(4, 16);
        let mut mock = MockActivationCapture::new(3, 16);
        let progress = ProgressReporter::new();
        let config = DwqConfig {
            calibration_samples: 2,
            base_bits: 4,
            sensitive_bits: 6,
            arch: DwqArch::Other,
            ..DwqConfig::default()
        };

        let result = run_dwq_activation_calibration(
            &tensor_map,
            &metadata,
            &config,
            &mut mock,
            &progress,
        );
        assert!(result.is_err(), "layer count mismatch must error");
        let err = result.unwrap_err();
        assert!(
            format!("{err}").contains("num_layers"),
            "error should mention num_layers, got: {err}"
        );
    }

    #[test]
    fn activation_calibration_rejects_zero_samples() {
        let metadata = tiny_metadata(2, 16);
        let tensor_map = tiny_tensor_map(2, 16);
        let mut mock = MockActivationCapture::new(2, 16);
        let progress = ProgressReporter::new();
        let config = DwqConfig {
            calibration_samples: 0,
            base_bits: 4,
            sensitive_bits: 6,
            arch: DwqArch::Other,
            ..DwqConfig::default()
        };
        let result = run_dwq_activation_calibration(
            &tensor_map,
            &metadata,
            &config,
            &mut mock,
            &progress,
        );
        assert!(result.is_err(), "zero calibration samples must error");
    }

    /// Sensitivity-driven bit allocation must produce SOME sensitive layers
    /// for a non-uniform activation distribution. MockActivationCapture's
    /// formula `tokens[t]*0.001 + l*0.01 + j*0.0001` makes later layers
    /// have higher activation values, which scores them as more sensitive.
    #[test]
    fn activation_calibration_promotes_high_sensitivity_layers() {
        let num_layers = 8u32;
        let hidden_size = 32u32;
        let metadata = tiny_metadata(num_layers, hidden_size);
        let tensor_map = tiny_tensor_map(num_layers, hidden_size);
        let mut mock = MockActivationCapture::new(num_layers, hidden_size);
        let progress = ProgressReporter::new();
        let config = DwqConfig {
            calibration_samples: 4,
            base_bits: 4,
            sensitive_bits: 6,
            arch: DwqArch::Other,
            ..DwqConfig::default()
        };

        // Run twice with different random seeds via different tokens.
        // The higher layer indices in the mock have larger magnitudes.
        let model = run_dwq_activation_calibration(
            &tensor_map,
            &metadata,
            &config,
            &mut mock,
            &progress,
        )
        .expect("calibration");
        // Check that at least one layer's q_proj was quantized at the
        // sensitive bits (6); concretely, the highest-magnitude layer
        // (highest index) should be among the sensitive set.
        let last_layer_q_proj = format!(
            "model.layers.{}.self_attn.q_proj.weight",
            num_layers - 1
        );
        let qt = model
            .tensors
            .get(&last_layer_q_proj)
            .expect("last-layer q_proj tensor must be present");
        // For mock activations, last layer has the highest activation
        // magnitude → should be allocated sensitive_bits=6.
        assert_eq!(
            qt.quant_info.bits, 6,
            "last layer (highest activation magnitude) should receive sensitive_bits=6"
        );
    }

    /// Extract the layer index from a tensor name. Used by tests.
    #[allow(dead_code)]
    fn extract_layer_index(name: &str) -> Option<usize> {
        let marker = ".layers.";
        let start = name.find(marker)? + marker.len();
        let rest = &name[start..];
        let end = rest.find('.')?;
        rest[..end].parse().ok()
    }

    #[test]
    fn extract_layer_index_smoke() {
        assert_eq!(
            extract_layer_index("model.layers.0.self_attn.q_proj.weight"),
            Some(0)
        );
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
    }

    /// **End-to-end validation (2026-04-25)** — drive the calibrator with
    /// a real `Qwen35Model` (zero-weighted, in-memory via `empty_from_cfg`)
    /// wrapped in `RealActivationCapture::from_model`, not the
    /// `MockActivationCapture` used by the other tests. This proves the
    /// CPU forward pass on a live qwen35 hybrid model (1 linear-attn +
    /// 1 full-attn layer) flows through `run_calibration_prompt` →
    /// `LayerActivations` → `compute_layer_sensitivity` →
    /// `allocate_bits_by_sensitivity` → `MixedBitQuantizer` →
    /// `QuantizedModel` end-to-end.
    ///
    /// On a zero-weighted model every layer's residual stream is also
    /// zero, so all sensitivities tie. `allocate_bits_by_sensitivity`
    /// returns the midpoint bit count for uniform sensitivity (per its
    /// own contract: see `sensitivity::tests::test_allocate_bits_uniform_sensitivity`).
    /// We assert exactly that — it's a load-bearing contract: a
    /// degenerate (all-zero) input must not crash the calibrator and
    /// must produce a well-defined output that's sensible for the
    /// activation distribution.
    #[test]
    fn activation_calibration_with_real_model_wrapper_succeeds() {
        use crate::inference::models::qwen35::activation_capture_real::RealActivationCapture;
        use crate::inference::models::qwen35::model::Qwen35Model;
        use crate::inference::models::qwen35::{
            Qwen35Config, Qwen35LayerKind, Qwen35Variant,
        };

        // Tiny hybrid qwen35 dense config, identical shape to
        // activation_capture_real::tests::tiny_dense_cfg() so the
        // forward path is exercised against the same surface that's
        // already pinned by 9 unit tests in that module.
        let cfg = Qwen35Config {
            variant: Qwen35Variant::Dense,
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: Some(16),
            rope_theta: 10_000.0,
            rotary_dim: 4,
            mrope_section: [1, 1, 1, 1],
            mrope_interleaved: true,
            partial_rotary_factor: 1.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            attn_output_gate: true,
            layer_types: vec![
                Qwen35LayerKind::LinearAttention,
                Qwen35LayerKind::FullAttention,
            ],
            full_attention_interval: 2,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            moe: None,
            mtp_num_hidden_layers: 0,
        };
        let num_layers = cfg.num_hidden_layers;
        let hidden_size = cfg.hidden_size;

        let model_vocab_size: u64 = 16; // matches cfg.vocab_size above (cast to u64 for ModelMetadata)
        let model = Qwen35Model::empty_from_cfg(cfg);
        let mut capture = RealActivationCapture::from_model(model);

        // Tensor_map is independent of the model — it's the conversion-
        // side data feeding the downstream quantizer. Match num_layers,
        // hidden_size, AND vocab_size so the calibration token generator
        // (which samples up to metadata.vocab_size) doesn't emit token
        // ids above the model's embedding table.
        let mut metadata = tiny_metadata(num_layers, hidden_size);
        metadata.vocab_size = model_vocab_size;
        let tensor_map = tiny_tensor_map(num_layers, hidden_size);

        let progress = ProgressReporter::new();
        let config = DwqConfig {
            // 2 calibration samples is enough — we just need the forward
            // pass to execute and capture residuals.
            calibration_samples: 2,
            base_bits: 4,
            sensitive_bits: 6,
            arch: DwqArch::Other,
            ..DwqConfig::default()
        };

        // Wall-time visibility — print without asserting a bound (CI-safe).
        // At apex MoE scale, expect ~60–180s for the calibration step
        // alone (1024 tokens × 40-layer hybrid forward, CPU). This tiny
        // synthetic gives a sub-millisecond baseline.
        let t0 = std::time::Instant::now();
        let model_out = run_dwq_activation_calibration(
            &tensor_map,
            &metadata,
            &config,
            &mut capture,
            &progress,
        )
        .expect(
            "real-Qwen35Model-wrapped activation calibration must succeed \
             on a zero-weighted hybrid model",
        );
        let elapsed = t0.elapsed();
        eprintln!(
            "[bench] run_dwq_activation_calibration (real Qwen35Model, \
             num_layers={num_layers}, hidden_size={hidden_size}, \
             samples={}): {:?}",
            config.calibration_samples, elapsed
        );

        // Method tag reflects the configured bit pair.
        assert_eq!(model_out.quant_method, "dwq-mixed-4-6");

        // Output preserves all input tensors.
        assert!(
            model_out.tensors.len() >= tensor_map.tensors.len(),
            "output tensor count ({}) must cover input ({})",
            model_out.tensors.len(),
            tensor_map.tensors.len()
        );

        // For zero-weighted model: all layer residuals are zero, so
        // sensitivities are uniform → `allocate_bits_by_sensitivity`
        // returns the midpoint (4+6)/2 = 5 for every layer (per its
        // documented contract). With midpoint = 5.0, our `> midpoint`
        // filter yields zero sensitive layers, so every weight gets
        // base_bits = 4. Anchor that contract here.
        let layer0_q = "model.layers.0.self_attn.q_proj.weight";
        let layer1_q = "model.layers.1.self_attn.q_proj.weight";
        for name in [layer0_q, layer1_q] {
            let qt = model_out
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("expected tensor {name} in output"));
            assert_eq!(
                qt.quant_info.bits, 4,
                "{name}: zero-weight model produces uniform sensitivity → \
                 every layer gets base_bits=4 (no sensitive promotion)"
            );
        }
    }
}
