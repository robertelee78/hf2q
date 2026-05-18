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
use crate::quantize::k_quant_codec_quantizer::{f16_passthrough, METHOD_K_QUANT_CODEC_DIRECT};
use crate::quantize::layer_mix::{
    is_kquant_row_misaligned, is_vision_tensor_pattern, kquant_misalignment_fallback, target_for,
    KQuantVariant,
};
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
        KQuantTarget::Q2K => "Q2_K".to_string(),
        KQuantTarget::Q3K => "Q3_K".to_string(),
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
        // through at original precision.  Also catches the layer_mix
        // skip rule for `ffn_gate_inp.weight` (`llama-quant.cpp:307`).
        //
        // **Ordering note (Iter D)**: same as `KQuantCodecQuantizer` —
        // explicit `config.preserve` (and the `ffn_gate_inp` skip rule)
        // wins over the implicit Iter D skip so the legacy
        // `method = "passthrough"` shape is unchanged at every existing
        // preserve site.  Iter D's skip runs only when neither preserve
        // signal triggered.
        let must_preserve = config.preserve
            || crate::quantize::layer_mix::should_skip_quantization(&tensor.name);
        if must_preserve {
            let (data, dtype) = if tensor.dtype == DType::BF16 {
                match tensor.to_f16() {
                    Ok(converted) => (std::sync::Arc::unwrap_or_clone(converted.data), DType::F16),
                    Err(_) => ((*tensor.data).clone(), tensor.dtype),
                }
            } else {
                ((*tensor.data).clone(), tensor.dtype)
            };
            return Ok(QuantizedTensor {
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                original_dtype: tensor.dtype,
                data: std::sync::Arc::new(data),
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

        // Vision-tensor F16 passthrough is intentional policy (vision
        // encoder weights ship as F16 in production GGUFs regardless
        // of the surrounding quantization target — see
        // `is_vision_tensor_pattern` doc).  This arm fires BEFORE the
        // K-quant misalignment arm below so vision tensors land at F16
        // even when their inner dim is 32-aligned.
        let row_len_for_skip = tensor.shape.last().copied().unwrap_or(0);
        if is_vision_tensor_pattern(&tensor.name) {
            tracing::info!(
                tensor = %tensor.name,
                row_len = row_len_for_skip,
                variant = self.variant.name(),
                "vision tensor → F16 passthrough"
            );
            return f16_passthrough(tensor, &tensor.name);
        }

        // Pick the per-tensor target via the policy.
        let i_layer = parse_block_index(&tensor.name).unwrap_or(0);
        let target = target_for(self.variant, &tensor.name, i_layer, self.n_layers);

        // K-quant misalignment fallback (mirrors llama.cpp's
        // `tensor_type_fallback` at `llama-quant.cpp:362-408`).
        // When the tensor's inner dim isn't a multiple of 256 (the
        // K-quant super-block size, `QK_K`), downshift to the
        // canonical 32-aligned legacy quant rather than F16
        // passthrough — keeps the output GGUF byte-compatible with
        // bartowski / unsloth conventions AND keeps hf2q's runtime
        // happy (its 3D MoE expert dispatcher `dispatch_id_mm_for_test`
        // only accepts Q-family block types).  If the legacy fallback
        // itself can't fit (ncols % 32 != 0), drop to F16 per
        // llama.cpp's `(WARNING: must use F16 due to unusual shape)`
        // branch.
        let target = if is_kquant_row_misaligned(row_len_for_skip) {
            let Some(fb) = kquant_misalignment_fallback(target) else {
                // `target` is already a 32-aligned legacy type and the
                // row is still misaligned to its block — the tensor is
                // genuinely un-quantizable.  Surface a typed error
                // rather than silently degrade to F16 (the no-fallback
                // mantra: degraded output is worse than a loud failure).
                return Err(QuantizeError::TensorQuantizeFailed {
                    tensor: tensor.name.clone(),
                    reason: format!(
                        "variant-quantizer ({}): row_len={row_len_for_skip} not aligned to \
                         any K-quant or legacy 32-aligned block, and target={target:?} \
                         has no further downshift",
                        self.variant.name()
                    ),
                });
            };
            if row_len_for_skip % 32 != 0 {
                // Even the 32-aligned legacy fallback can't represent
                // this row.  Production models (Gemma, Qwen, Llama)
                // never hit this — every tensor's inner dim is div 32.
                // If a future model breaks that, surface a typed error
                // so the operator addresses it explicitly.
                return Err(QuantizeError::TensorQuantizeFailed {
                    tensor: tensor.name.clone(),
                    reason: format!(
                        "variant-quantizer ({}): row_len={row_len_for_skip} not div 32 — \
                         no GGML block format can encode this tensor (Q-family blocks need \
                         div 32, K-family need div 256).  Refusing to silently emit F16.",
                        self.variant.name()
                    ),
                });
            }
            tracing::info!(
                tensor = %tensor.name,
                row_len = row_len_for_skip,
                variant = self.variant.name(),
                from = ?target,
                to = ?fb,
                "K-quant row not div 256 — switching target to 32-aligned legacy"
            );
            fb
        } else {
            target
        };

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
            data: bytes.into(),
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
            data: std::sync::Arc::new(data),
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

    /// **Iter D**: vision-tensor blocker shape (1152-dim
    /// `model.visual.…`) emits F16 passthrough through the variant
    /// dispatcher — pre-empts the codec's row-alignment rejection.
    #[test]
    fn variant_q4km_vision_tensor_emits_f16_passthrough() {
        let row: Vec<f32> = (0..1152).map(|i| (i as f32 - 576.0) / 576.0).collect();
        let tensor = make_f32_tensor(
            "model.visual.blocks.0.attn.proj.weight",
            vec![1152],
            &row,
        );
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, 32);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.method, "f16");
        assert_eq!(out.quant_info.bits, 16);
        assert!(out.quant_info.preserved);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("F16"));
        assert_eq!(out.data.len(), 1152 * 2);
    }

    /// **Iter D**: aligned language tensor under the same variant still
    /// routes through the codec — proves the skip is narrowly scoped.
    #[test]
    fn variant_q4km_aligned_language_tensor_routes_through_codec() {
        const QK_K: usize = 256;
        let row: Vec<f32> = (0..QK_K).map(|i| (i as f32) / QK_K as f32).collect();
        let tensor = make_f32_tensor("blk.5.attn_q.weight", vec![QK_K], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = VariantKQuantizer::new(KQuantVariant::Q4_K_M, CalibrationData::None, 32);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(
            out.quant_info.method,
            crate::quantize::k_quant_codec_quantizer::METHOD_K_QUANT_CODEC_DIRECT
        );
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));
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
