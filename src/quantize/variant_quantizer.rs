//! `VariantKQuantizer` — variant-aware K-quant Quantizer that picks
//! per-tensor targets via [`crate::quantize::layer_mix::target_for`]
//! (ADR-014 P7 iter-3s).
//!
//! Composes:
//! - `layer_mix` (iter-3r) for the per-tensor variant policy.
//! - `k_quant_codec` (iter-3g/3j) for the calibration-aware codec.
//! - `KQuantCodecQuantizer` (iter-3m) for the underlying Quantizer
//!   trait machinery.
//!
//! ## API
//!
//! ```ignore
//! use crate::quantize::{
//!     layer_mix::KQuantVariant,
//!     variant_quantizer::VariantKQuantizer,
//! };
//!
//! let variant = KQuantVariant::Q4_K_M;
//! let calibration = CalibrationData::from_imatrix_gguf(&imatrix)?;
//! let q = VariantKQuantizer::new(variant, calibration, 32 /* n_layers */);
//!
//! // q implements Quantizer; cmd_convert calls q.quantize_tensor(...)
//! // for every tensor; the right per-tensor target is picked
//! // automatically (Q6_K for output/token_embd; Q5_K/Q6_K bumps for
//! // some attn_v / ffn_down layers; Q4_K base elsewhere).
//! ```
//!
//! ## Layer-index parsing
//!
//! The Quantizer trait's `quantize_tensor(&self, tensor, config)`
//! signature doesn't pass `i_layer` explicitly — the variant policy
//! needs to extract it from `tensor.name`. This module's
//! [`parse_block_index`] handles the standard `blk.<N>.<role>.<dtype>`
//! pattern. Tensors that don't match (output, token_embd, norms with
//! no block prefix) get `i_layer = 0` (irrelevant for the policy
//! since those categories don't index into `use_more_bits`).

use crate::calibrate::calibrator::CalibrationData;
use crate::ir::{DType, QuantizedTensor, TensorQuantInfo, TensorRef};
use crate::quantize::k_quant_codec::{quantize_tensor_2d_to_bytes, KQuantTarget};
use crate::quantize::k_quant_codec_quantizer::METHOD_K_QUANT_CODEC_DIRECT;
use crate::quantize::layer_mix::{target_for, KQuantVariant};
use crate::quantize::{LayerQuantConfig, Quantizer, QuantizeError};

/// Variant-aware K-quant Quantizer. Each tensor gets the target the
/// `layer_mix` policy assigns for the chosen variant.
pub struct VariantKQuantizer {
    variant: KQuantVariant,
    calibration: CalibrationData,
    n_layers: usize,
}

impl VariantKQuantizer {
    /// Construct with the variant, calibration, and the model's total
    /// transformer depth (used by `use_more_bits` per-layer-position
    /// policy).
    pub fn new(variant: KQuantVariant, calibration: CalibrationData, n_layers: usize) -> Self {
        Self {
            variant,
            calibration,
            n_layers,
        }
    }

    /// The variant this quantizer emits.
    pub fn variant(&self) -> KQuantVariant {
        self.variant
    }

    /// Total layer count (n_layers) baked into the policy lookup.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
}

/// Extract the `<N>` from a tensor name following the `blk.<N>.…`
/// pattern. Returns `None` for non-block tensors (output.weight,
/// token_embd.weight, norms without block prefix).
///
/// ```ignore
/// assert_eq!(parse_block_index("blk.5.attn_v.weight"), Some(5));
/// assert_eq!(parse_block_index("output.weight"), None);
/// assert_eq!(parse_block_index("blk.x.attn_q.weight"), None); // bad N
/// ```
pub fn parse_block_index(tensor_name: &str) -> Option<usize> {
    let after_blk = tensor_name.strip_prefix("blk.")?;
    let dot = after_blk.find('.')?;
    after_blk[..dot].parse().ok()
}

/// Convert `TensorRef` data to `Vec<f32>` (F32/F16/BF16).
fn tensor_to_f32(tensor: &TensorRef) -> Result<Vec<f32>, QuantizeError> {
    match tensor.dtype {
        DType::F32 => Ok(tensor
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        DType::F16 => Ok(tensor
            .data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect()),
        DType::BF16 => Ok(tensor
            .data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::bf16::from_bits(bits).to_f32()
            })
            .collect()),
        _ => Err(QuantizeError::TensorQuantizeFailed {
            tensor: tensor.name.clone(),
            reason: format!("variant-quantizer: cannot convert dtype {} to f32", tensor.dtype),
        }),
    }
}

/// Map a [`KQuantTarget`] back to its canonical GGUF name string.
fn target_to_ggml_name(target: KQuantTarget) -> String {
    match target {
        KQuantTarget::Q4K => "Q4_K".to_string(),
        KQuantTarget::Q5K => "Q5_K".to_string(),
        KQuantTarget::Q6K => "Q6_K".to_string(),
        KQuantTarget::Q4Legacy => "Q4_0".to_string(),
        KQuantTarget::Q4Legacy1 => "Q4_1".to_string(),
        KQuantTarget::Q5Legacy0 => "Q5_0".to_string(),
        KQuantTarget::Q5Legacy1 => "Q5_1".to_string(),
        KQuantTarget::Q8Legacy => "Q8_0".to_string(),
    }
}

impl Quantizer for VariantKQuantizer {
    fn name(&self) -> &str {
        self.variant.name()
    }

    fn requires_calibration(&self) -> bool {
        false
    }

    fn quantize_tensor(
        &self,
        tensor: &TensorRef,
        config: &LayerQuantConfig,
    ) -> Result<QuantizedTensor, QuantizeError> {
        // Preserve path matches StaticQuantizer / KQuantCodecQuantizer:
        // norms, biases, vision tensors, explicit-preserve flags pass
        // through at original precision.
        if config.preserve {
            let (data, dtype) = if tensor.dtype == DType::BF16 {
                match tensor.to_f16() {
                    Ok(converted) => (converted.data, DType::F16),
                    Err(_) => (tensor.data.clone(), tensor.dtype),
                }
            } else {
                (tensor.data.clone(), tensor.dtype)
            };
            return Ok(QuantizedTensor {
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                original_dtype: tensor.dtype,
                data,
                quant_info: TensorQuantInfo {
                    method: "passthrough".to_string(),
                    bits: dtype.element_size() as u8 * 8,
                    group_size: 0,
                    preserved: true,
                    scales: None,
                    biases: None,
                    ggml_type: None,
                },
            });
        }

        // Pick the per-tensor target via the policy.
        let i_layer = parse_block_index(&tensor.name).unwrap_or(0);
        let target = target_for(self.variant, &tensor.name, i_layer, self.n_layers);

        // Convert to F32 for the codec.
        let f32_values = tensor_to_f32(tensor)?;

        // Determine row layout.
        let (n_rows, row_len) = match tensor.shape.len() {
            0 => {
                return Err(QuantizeError::TensorQuantizeFailed {
                    tensor: tensor.name.clone(),
                    reason: "variant-quantizer: 0-D tensor not supported".into(),
                });
            }
            1 => (1, tensor.shape[0]),
            _ => {
                let row_len = *tensor.shape.last().unwrap();
                let n_rows = tensor.numel() / row_len;
                (n_rows, row_len)
            }
        };

        let bytes = quantize_tensor_2d_to_bytes(
            &f32_values,
            n_rows,
            row_len,
            target,
            &self.calibration,
            &tensor.name,
        )
        .map_err(|e| QuantizeError::TensorQuantizeFailed {
            tensor: tensor.name.clone(),
            reason: format!("variant-quantizer ({}, target={target:?}): {e}", self.variant.name()),
        })?;

        Ok(QuantizedTensor {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            original_dtype: tensor.dtype,
            data: bytes,
            quant_info: TensorQuantInfo {
                method: METHOD_K_QUANT_CODEC_DIRECT.to_string(),
                bits: 0,
                group_size: 0,
                preserved: false,
                scales: None,
                biases: None,
                ggml_type: Some(target_to_ggml_name(target)),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;

    fn make_f32_tensor(name: &str, shape: Vec<usize>, values: &[f32]) -> TensorRef {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        TensorRef {
            name: name.to_string(),
            shape,
            dtype: DType::F32,
            data,
        }
    }

    #[test]
    fn parse_block_index_canonical() {
        assert_eq!(parse_block_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(parse_block_index("blk.5.attn_v.weight"), Some(5));
        assert_eq!(parse_block_index("blk.31.ffn_down.weight"), Some(31));
        assert_eq!(parse_block_index("blk.100.x.weight"), Some(100));
    }

    #[test]
    fn parse_block_index_non_block_tensors() {
        assert_eq!(parse_block_index("output.weight"), None);
        assert_eq!(parse_block_index("token_embd.weight"), None);
        assert_eq!(parse_block_index("rope_freqs.weight"), None);
    }

    #[test]
    fn parse_block_index_malformed_returns_none() {
        assert_eq!(parse_block_index("blk.abc.weight"), None);
        assert_eq!(parse_block_index("blk."), None);
        assert_eq!(parse_block_index("blk"), None);
    }

    /// `Q4_K_M` variant on a [4, 256] attn_q tensor at layer 5
    /// (middle of 32 layers): policy picks Q4_K (base) — not the
    /// Q6_K bump, since attn_q isn't in the bump list.
    #[test]
    fn variant_q4km_attn_q_uses_q4k_base() {
        const QK_K: usize = 256;
        let row: Vec<f32> = (0..4 * QK_K).map(|i| (i as f32 - 512.0) / 512.0).collect();
        let tensor = make_f32_tensor("blk.5.attn_q.weight", vec![4, QK_K], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, 32);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.data.len(), 4 * 144); // Q4_K block = 144 bytes
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));
    }

    /// `Q4_K_M` on `output.weight` bumps to Q6_K (210 bytes/block).
    #[test]
    fn variant_q4km_output_bumps_to_q6k() {
        const QK_K: usize = 256;
        let row: Vec<f32> = (0..QK_K).map(|i| (i as f32) / QK_K as f32).collect();
        let tensor = make_f32_tensor("output.weight", vec![QK_K], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, 32);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.data.len(), 210); // Q6_K block size
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q6_K"));
    }

    /// `Q4_K_M` on `blk.0.attn_v.weight` (layer 0, in first 1/8 →
    /// use_more_bits) bumps to Q6_K.
    #[test]
    fn variant_q4km_attn_v_first_layer_bumps_to_q6k() {
        const QK_K: usize = 256;
        let row: Vec<f32> = (0..QK_K).map(|i| (i as f32) / QK_K as f32).collect();
        let tensor = make_f32_tensor("blk.0.attn_v.weight", vec![QK_K], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, 32);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.data.len(), 210);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q6_K"));
    }

    /// `Q4_K_M` on `blk.5.attn_v.weight` (layer 5, NOT in
    /// use_more_bits range for n_layers=32) stays at Q4_K.
    #[test]
    fn variant_q4km_attn_v_middle_layer_stays_q4k() {
        const QK_K: usize = 256;
        let row: Vec<f32> = (0..QK_K).map(|i| (i as f32) / QK_K as f32).collect();
        let tensor = make_f32_tensor("blk.5.attn_v.weight", vec![QK_K], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, 32);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.data.len(), 144);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));
    }

    /// `Q5_K_S` variant: output stays at Q5_K (no _M bump on _S).
    #[test]
    fn variant_q5ks_output_stays_q5k() {
        const QK_K: usize = 256;
        let row: Vec<f32> = (0..QK_K).map(|i| (i as f32) / QK_K as f32).collect();
        let tensor = make_f32_tensor("output.weight", vec![QK_K], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = VariantKQuantizer::new(KQuantVariant::Q5_K_S, CalibrationData::None, 32);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.data.len(), 176); // Q5_K block size
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q5_K"));
    }

    /// `Q6_K` variant: every tensor at Q6_K.
    #[test]
    fn variant_q6k_all_tensors_q6k() {
        const QK_K: usize = 256;
        let row: Vec<f32> = (0..QK_K).map(|i| (i as f32) / QK_K as f32).collect();
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = VariantKQuantizer::new(KQuantVariant::Q6_K, CalibrationData::None, 32);
        for name in [
            "output.weight",
            "blk.0.attn_v.weight",
            "blk.5.ffn_down.weight",
            "blk.10.attn_q.weight",
        ] {
            let tensor = make_f32_tensor(name, vec![QK_K], &row);
            let out = q.quantize_tensor(&tensor, &cfg).unwrap();
            assert_eq!(out.data.len(), 210, "{name}");
            assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q6_K"), "{name}");
        }
    }

    /// `preserve = true` returns passthrough regardless of variant.
    #[test]
    fn variant_preserve_passthrough() {
        let row: Vec<f32> = vec![0.5_f32; 16];
        let tensor = make_f32_tensor("blk.0.attn_norm.weight", vec![16], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: true,
        };
        let q = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, 32);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.method, "passthrough");
        assert!(out.quant_info.preserved);
        assert_eq!(out.data, tensor.data);
    }

    /// `Quantizer::name` reports the variant name.
    #[test]
    fn variant_name_introspection() {
        assert_eq!(
            VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, 32).name(),
            "Q4_K_M"
        );
        assert_eq!(
            VariantKQuantizer::new(KQuantVariant::Q5_K_S, CalibrationData::None, 32).name(),
            "Q5_K_S"
        );
    }

    /// A full **end-to-end variant pipeline test**: build calibration,
    /// quantize multiple tensors with the variant, verify the policy
    /// picked the right per-tensor target.
    #[test]
    fn variant_end_to_end_pipeline() {
        use crate::calibrate::imatrix::ImatrixCollector;

        const QK_K: usize = 256;

        // Calibration with biased weights on first 16 cols.
        let mut col = ImatrixCollector::new();
        let acts: Vec<f32> = (0..QK_K).map(|i| if i < 16 { 10.0 } else { 0.1 }).collect();
        for tname in [
            "output.weight",
            "blk.0.attn_v.weight",
            "blk.5.attn_q.weight",
            "blk.10.ffn_down.weight",
        ] {
            col.accumulate_dense(tname, &acts, 1, QK_K).unwrap();
        }
        col.record_chunk();

        let calibration = CalibrationData::from_imatrix_collector(&col);
        let q = VariantKQuantizer::new(KQuantVariant::Q4_K_M, calibration, 32);

        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };

        let row: Vec<f32> = (0..QK_K).map(|i| (i as f32) / QK_K as f32).collect();

        // output.weight → Q6_K (M variant bump)
        let tensor = make_f32_tensor("output.weight", vec![QK_K], &row);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q6_K"));

        // blk.0.attn_v.weight → Q6_K (use_more_bits at layer 0)
        let tensor = make_f32_tensor("blk.0.attn_v.weight", vec![QK_K], &row);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q6_K"));

        // blk.5.attn_q.weight → Q4_K (attn_q is not in the bump set)
        let tensor = make_f32_tensor("blk.5.attn_q.weight", vec![QK_K], &row);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));

        // blk.10.ffn_down.weight → Q4_K (layer 10 of 32; (10-4)%3=0,
        // so use_more_bits=false; ffn_down stays at base).
        let tensor = make_f32_tensor("blk.10.ffn_down.weight", vec![QK_K], &row);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));
    }
}
