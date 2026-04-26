//! Mixed-bit quantizer with --sensitive-layers support.
//!
//! Applies different bit widths to different layers:
//! - Sensitive layers (specified by --sensitive-layers) get higher bits
//! - All other layers get lower bits
//!
//! Presets:
//! - mixed-2-6: 2-bit base, 6-bit sensitive
//! - mixed-3-6: 3-bit base, 6-bit sensitive
//! - mixed-4-6: 4-bit base, 6-bit sensitive

use std::collections::HashSet;
use std::ops::RangeInclusive;

use thiserror::Error;

use crate::ir::{QuantizedTensor, TensorRef};
use crate::quantize::static_quant::StaticQuantizer;
use crate::quantize::{LayerQuantConfig, QuantizeError, Quantizer};

/// Errors from mixed-bit quantization.
#[derive(Error, Debug)]
pub enum MixedQuantError {
    #[error("Invalid sensitive layer range: {0}")]
    InvalidLayerRange(String),

    #[error("Invalid mixed-bit preset: {0}. Valid presets: mixed-2-6, mixed-3-6, mixed-4-6, mixed-2-8, mixed-4-8, mixed-6-8")]
    InvalidPreset(String),
}

/// Configuration for a mixed-bit quantization preset.
#[derive(Debug, Clone)]
pub struct MixedBitPreset {
    /// Base bit width for non-sensitive layers
    pub base_bits: u8,
    /// Higher bit width for sensitive layers
    pub sensitive_bits: u8,
    /// Human-readable name
    pub name: String,
}

impl MixedBitPreset {
    /// Parse a preset name into a MixedBitPreset.
    pub fn from_name(name: &str) -> Result<Self, MixedQuantError> {
        match name {
            "mixed-2-6" => Ok(Self {
                base_bits: 2,
                sensitive_bits: 6,
                name: name.to_string(),
            }),
            "mixed-3-6" => Ok(Self {
                base_bits: 3,
                sensitive_bits: 6,
                name: name.to_string(),
            }),
            "mixed-4-6" => Ok(Self {
                base_bits: 4,
                sensitive_bits: 6,
                name: name.to_string(),
            }),
            "mixed-4-8" => Ok(Self {
                base_bits: 4,
                sensitive_bits: 8,
                name: name.to_string(),
            }),
            "mixed-6-8" => Ok(Self {
                base_bits: 6,
                sensitive_bits: 8,
                name: name.to_string(),
            }),
            "mixed-2-8" => Ok(Self {
                base_bits: 2,
                sensitive_bits: 8,
                name: name.to_string(),
            }),
            _ => Err(MixedQuantError::InvalidPreset(name.to_string())),
        }
    }
}

/// Mixed-bit quantizer that applies different bit widths per layer.
///
/// Uses StaticQuantizer internally for the actual quantization math,
/// but routes each tensor to the appropriate bit width based on
/// the sensitive layers configuration.
pub struct MixedBitQuantizer {
    /// The preset being used
    preset: MixedBitPreset,
    /// Set of layer indices considered sensitive (get higher bits)
    sensitive_indices: HashSet<usize>,
    /// Group size for quantization
    group_size: usize,
    /// Internal low-bit quantizer
    base_quantizer: StaticQuantizer,
    /// Internal high-bit quantizer
    sensitive_quantizer: StaticQuantizer,
}

impl MixedBitQuantizer {
    /// Create a new mixed-bit quantizer.
    ///
    /// `preset_name`: One of "mixed-2-6", "mixed-3-6", "mixed-4-6"
    /// `sensitive_layers`: Ranges of layer indices to protect at higher precision
    /// `group_size`: Group size for block quantization
    pub fn new(
        preset_name: &str,
        sensitive_layers: &[RangeInclusive<usize>],
        group_size: usize,
    ) -> Result<Self, QuantizeError> {
        let preset = MixedBitPreset::from_name(preset_name).map_err(|e| {
            QuantizeError::UnsupportedMethod {
                method: e.to_string(),
            }
        })?;

        // Validate sensitive layer ranges
        for range in sensitive_layers {
            if range.start() > range.end() {
                return Err(QuantizeError::UnsupportedMethod {
                    method: MixedQuantError::InvalidLayerRange(format!(
                        "start ({}) > end ({})",
                        range.start(),
                        range.end()
                    ))
                    .to_string(),
                });
            }
        }

        // Build the set of sensitive layer indices
        let mut sensitive_indices = HashSet::new();
        for range in sensitive_layers {
            for idx in range.clone() {
                sensitive_indices.insert(idx);
            }
        }

        // Create internal quantizers
        let base_method = format!("q{}", preset.base_bits);
        let sensitive_method = format!("q{}", preset.sensitive_bits);

        // For bits not directly supported by StaticQuantizer (e.g., q3, q6),
        // we use the quantize_weight function directly with custom bits
        let base_quantizer = StaticQuantizer::new(&base_method).unwrap_or_else(|_| {
            // Fallback: use q4 as base and override bits
            StaticQuantizer::new("q4").expect("q4 quantizer should always be valid")
        });

        let sensitive_quantizer = StaticQuantizer::new(&sensitive_method).unwrap_or_else(|_| {
            // Fallback: use q8 as base and override bits
            StaticQuantizer::new("q8").expect("q8 quantizer should always be valid")
        });

        Ok(Self {
            preset,
            sensitive_indices,
            group_size,
            base_quantizer,
            sensitive_quantizer,
        })
    }

    /// Determine if a tensor belongs to a sensitive layer.
    ///
    /// Extracts the layer index from tensor names like:
    /// "model.layers.13.self_attn.q_proj.weight" -> layer 13
    fn is_sensitive_tensor(&self, tensor_name: &str) -> bool {
        if self.sensitive_indices.is_empty() {
            return false;
        }

        extract_layer_index(tensor_name)
            .map(|idx| self.sensitive_indices.contains(&idx))
            .unwrap_or(false)
    }

    /// Get the bit width for a specific tensor.
    pub fn bits_for_tensor(&self, tensor_name: &str) -> u8 {
        if self.is_sensitive_tensor(tensor_name) {
            self.preset.sensitive_bits
        } else {
            self.preset.base_bits
        }
    }
}

impl Quantizer for MixedBitQuantizer {
    fn name(&self) -> &str {
        &self.preset.name
    }

    fn requires_calibration(&self) -> bool {
        false
    }

    fn quantize_tensor(
        &self,
        tensor: &TensorRef,
        config: &LayerQuantConfig,
    ) -> Result<QuantizedTensor, QuantizeError> {
        // Non-weight tensors are always preserved at full precision
        if config.preserve || !tensor.is_weight() {
            return self.base_quantizer.quantize_tensor(
                tensor,
                &LayerQuantConfig {
                    bits: config.bits,
                    group_size: config.group_size,
                    preserve: true,
                },
            );
        }

        // Determine bit width based on layer membership
        let bits = self.bits_for_tensor(&tensor.name);

        let layer_config = LayerQuantConfig {
            bits,
            group_size: self.group_size,
            preserve: false,
        };

        // Route to the appropriate internal quantizer
        if self.is_sensitive_tensor(&tensor.name) {
            self.sensitive_quantizer
                .quantize_tensor(tensor, &layer_config)
        } else {
            self.base_quantizer
                .quantize_tensor(tensor, &layer_config)
        }
    }
}

/// Extract the layer index from a tensor name.
///
/// Handles common naming patterns:
/// - "model.layers.13.self_attn.q_proj.weight" -> Some(13)
/// - "model.language_model.layers.5.mlp.down_proj.weight" -> Some(5)
/// - "model.embed_tokens.weight" -> None (not a layer tensor)
fn extract_layer_index(name: &str) -> Option<usize> {
    // Look for "layers.N" pattern
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" {
            if let Some(num_str) = parts.get(i + 1) {
                return num_str.parse::<usize>().ok();
            }
        }
    }
    None
}

/// Build a per-layer bit allocation map for the quantization_config.json.
///
/// Returns a map of tensor name -> bit width for all weight tensors.
pub fn build_per_layer_bits_map(
    quantizer: &MixedBitQuantizer,
    tensor_names: &[String],
) -> std::collections::HashMap<String, u8> {
    tensor_names
        .iter()
        .map(|name| (name.clone(), quantizer.bits_for_tensor(name)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;

    fn make_f16_tensor(name: &str, shape: Vec<usize>) -> TensorRef {
        let numel: usize = shape.iter().product();
        TensorRef {
            name: name.to_string(),
            shape,
            dtype: DType::F16,
            data: vec![0u8; numel * 2],
        }
    }

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(
            extract_layer_index("model.layers.13.self_attn.q_proj.weight"),
            Some(13)
        );
        assert_eq!(
            extract_layer_index("model.layers.0.mlp.down_proj.weight"),
            Some(0)
        );
        assert_eq!(
            extract_layer_index("model.language_model.layers.5.self_attn.weight"),
            Some(5)
        );
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
    }

    #[test]
    fn test_mixed_bit_preset_parsing() {
        let preset = MixedBitPreset::from_name("mixed-4-6").unwrap();
        assert_eq!(preset.base_bits, 4);
        assert_eq!(preset.sensitive_bits, 6);

        let preset = MixedBitPreset::from_name("mixed-2-6").unwrap();
        assert_eq!(preset.base_bits, 2);
        assert_eq!(preset.sensitive_bits, 6);

        let preset = MixedBitPreset::from_name("mixed-3-6").unwrap();
        assert_eq!(preset.base_bits, 3);
        assert_eq!(preset.sensitive_bits, 6);

        // New DWQ-backing presets
        let preset = MixedBitPreset::from_name("mixed-4-8").unwrap();
        assert_eq!(preset.base_bits, 4);
        assert_eq!(preset.sensitive_bits, 8);
        assert_eq!(preset.name, "mixed-4-8");

        let preset = MixedBitPreset::from_name("mixed-6-8").unwrap();
        assert_eq!(preset.base_bits, 6);
        assert_eq!(preset.sensitive_bits, 8);
        assert_eq!(preset.name, "mixed-6-8");

        let preset = MixedBitPreset::from_name("mixed-2-8").unwrap();
        assert_eq!(preset.base_bits, 2);
        assert_eq!(preset.sensitive_bits, 8);
        assert_eq!(preset.name, "mixed-2-8");
    }

    #[test]
    fn test_invalid_preset() {
        assert!(MixedBitPreset::from_name("mixed-5-8").is_err());
        assert!(MixedBitPreset::from_name("q4").is_err());
    }

    #[test]
    fn test_invalid_preset_error_message_lists_all_valid_keys() {
        let err = MixedBitPreset::from_name("mixed-5-7").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("mixed-2-6"), "error must mention mixed-2-6");
        assert!(msg.contains("mixed-3-6"), "error must mention mixed-3-6");
        assert!(msg.contains("mixed-4-6"), "error must mention mixed-4-6");
        assert!(msg.contains("mixed-4-8"), "error must mention mixed-4-8");
        assert!(msg.contains("mixed-6-8"), "error must mention mixed-6-8");
        assert!(msg.contains("mixed-2-8"), "error must mention mixed-2-8");
    }

    #[test]
    fn test_sensitive_layer_detection() {
        let quantizer = MixedBitQuantizer::new(
            "mixed-4-6",
            &[13..=24],
            64,
        )
        .unwrap();

        assert!(quantizer.is_sensitive_tensor("model.layers.13.self_attn.q_proj.weight"));
        assert!(quantizer.is_sensitive_tensor("model.layers.24.mlp.down_proj.weight"));
        assert!(!quantizer.is_sensitive_tensor("model.layers.0.self_attn.q_proj.weight"));
        assert!(!quantizer.is_sensitive_tensor("model.layers.25.self_attn.q_proj.weight"));
        assert!(!quantizer.is_sensitive_tensor("model.embed_tokens.weight"));
    }

    #[test]
    fn test_bits_for_tensor() {
        let quantizer = MixedBitQuantizer::new(
            "mixed-4-6",
            &[13..=24],
            64,
        )
        .unwrap();

        assert_eq!(
            quantizer.bits_for_tensor("model.layers.13.weight"),
            6
        );
        assert_eq!(
            quantizer.bits_for_tensor("model.layers.0.weight"),
            4
        );
    }

    #[test]
    fn test_mixed_quantizer_trait() {
        let quantizer = MixedBitQuantizer::new(
            "mixed-4-6",
            &[0..=0],
            64,
        )
        .unwrap();

        assert_eq!(quantizer.name(), "mixed-4-6");
        assert!(!quantizer.requires_calibration());
    }

    #[test]
    fn test_quantize_sensitive_tensor() {
        let quantizer = MixedBitQuantizer::new(
            "mixed-4-6",
            &[0..=0],
            64,
        )
        .unwrap();

        // ADR-012 P9b is_weight() invariant: inner_dim ≥ 32 required for the
        // tensor to be classified as a weight (Q4_0 block alignment).  Shape
        // bumped from [8,8] → [32,32] post-2026-04-25 ir.rs change.
        let tensor = make_f16_tensor("model.layers.0.self_attn.q_proj.weight", vec![32, 32]);
        let config = LayerQuantConfig {
            bits: 4,
            group_size: 64,
            preserve: false,
        };

        let result = quantizer.quantize_tensor(&tensor, &config).unwrap();
        // Sensitive layer should get 6 bits (or the sensitive quantizer's bit width)
        // The actual bits depend on the quantizer routing
        assert!(!result.quant_info.preserved);
    }

    #[test]
    fn test_quantize_non_sensitive_tensor() {
        let quantizer = MixedBitQuantizer::new(
            "mixed-4-6",
            &[10..=20],
            64,
        )
        .unwrap();

        // ADR-012 P9b is_weight() invariant: inner_dim ≥ 32 required for the
        // tensor to be classified as a weight (Q4_0 block alignment).  Shape
        // bumped from [8,8] → [32,32] post-2026-04-25 ir.rs change.
        let tensor = make_f16_tensor("model.layers.0.self_attn.q_proj.weight", vec![32, 32]);
        let config = LayerQuantConfig {
            bits: 4,
            group_size: 64,
            preserve: false,
        };

        let result = quantizer.quantize_tensor(&tensor, &config).unwrap();
        assert!(!result.quant_info.preserved);
    }

    #[test]
    fn test_preserved_tensors_always_preserved() {
        let quantizer = MixedBitQuantizer::new(
            "mixed-4-6",
            &[0..=30],
            64,
        )
        .unwrap();

        let tensor = TensorRef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            shape: vec![8],
            dtype: DType::F16,
            data: vec![0u8; 16],
        };
        let config = LayerQuantConfig {
            bits: 4,
            group_size: 64,
            preserve: true,
        };

        let result = quantizer.quantize_tensor(&tensor, &config).unwrap();
        assert!(result.quant_info.preserved);
    }

    #[test]
    fn test_build_per_layer_bits_map() {
        let quantizer = MixedBitQuantizer::new(
            "mixed-4-6",
            &[1..=1],
            64,
        )
        .unwrap();

        let names = vec![
            "model.layers.0.self_attn.weight".to_string(),
            "model.layers.1.self_attn.weight".to_string(),
            "model.layers.2.self_attn.weight".to_string(),
        ];

        let map = build_per_layer_bits_map(&quantizer, &names);
        assert_eq!(*map.get("model.layers.0.self_attn.weight").unwrap(), 4);
        assert_eq!(*map.get("model.layers.1.self_attn.weight").unwrap(), 6);
        assert_eq!(*map.get("model.layers.2.self_attn.weight").unwrap(), 4);
    }

    #[test]
    fn test_comma_separated_sensitive_layers() {
        let quantizer = MixedBitQuantizer::new(
            "mixed-4-6",
            &[1..=1, 5..=5, 13..=24],
            64,
        )
        .unwrap();

        assert_eq!(quantizer.bits_for_tensor("model.layers.1.weight"), 6);
        assert_eq!(quantizer.bits_for_tensor("model.layers.5.weight"), 6);
        assert_eq!(quantizer.bits_for_tensor("model.layers.13.weight"), 6);
        assert_eq!(quantizer.bits_for_tensor("model.layers.24.weight"), 6);
        assert_eq!(quantizer.bits_for_tensor("model.layers.0.weight"), 4);
        assert_eq!(quantizer.bits_for_tensor("model.layers.6.weight"), 4);
        assert_eq!(quantizer.bits_for_tensor("model.layers.25.weight"), 4);
    }

    #[test]
    fn test_no_sensitive_layers() {
        // When no sensitive layers specified, everything gets base bits
        let quantizer = MixedBitQuantizer::new(
            "mixed-4-6",
            &[],
            64,
        )
        .unwrap();

        assert_eq!(quantizer.bits_for_tensor("model.layers.0.weight"), 4);
        assert_eq!(quantizer.bits_for_tensor("model.layers.13.weight"), 4);
    }
}
