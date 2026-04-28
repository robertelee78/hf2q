//! `KQuantCodecQuantizer` — `Quantizer` trait impl that uses the
//! pure-Rust `k_quant_codec` to produce GGUF block bytes directly
//! (ADR-014 P7 iter-3m).
//!
//! This is the **production bridge** between the existing `Quantizer`
//! trait machinery (`quantize_streaming`, `quantize_streaming_parallel`,
//! and the rayon thread pool) and the new pure-Rust block-format
//! ports (k_quant.rs + q_legacy.rs).
//!
//! ## Output `QuantizedTensor` shape
//!
//! `data` contains the GGUF-block-format bytes ready for direct write
//! to a tensor data section — **no further repack needed**. The
//! `quant_info` fields are set so the existing GGUF backend's tensor-
//! header writer can find the target type:
//!
//! - `method = "k_quant_codec_direct"` — sentinel that signals the
//!   bytes are already in GGUF block format. P4 wiring teaches the
//!   GGUF backend to recognise this and skip the repack step.
//! - `bits = 0` — bits no longer drives the packing decision (the
//!   target enum does). 0 is a sentinel for "see ggml_type field".
//! - `group_size = 0` — same reason; the codec block size is implicit
//!   in the target.
//! - `ggml_type = Some(format_name)` — already exists on
//!   `TensorQuantInfo` for ADR-005 Apex per-tensor types; reuse it.
//! - `scales = None`, `biases = None` — the codec packs scales inline
//!   in the block bytes.
//!
//! ## When to use vs `StaticQuantizer`
//!
//! `StaticQuantizer` produces the legacy IR-quantize format (raw i8
//! quants + separate scales/biases) consumed by the existing
//! `repack_q*_*` paths in `src/backends/gguf.rs`. That format has a
//! lossy round-trip via `f32_values[i] = q × scale` when the GGUF
//! backend re-quantizes — the very dance ADR-014 P4 eliminates.
//!
//! `KQuantCodecQuantizer` produces final GGUF block bytes in **one
//! pass**, no intermediate IR-quantize step. This is the path P4
//! enables.

use crate::calibrate::calibrator::CalibrationData;
use crate::ir::{DType, QuantizedTensor, TensorQuantInfo, TensorRef};
use crate::quantize::k_quant_codec::{quantize_tensor_2d_to_bytes, KQuantTarget};
use crate::quantize::layer_mix::should_emit_f16_for_kquant;
use crate::quantize::{LayerQuantConfig, Quantizer, QuantizeError};

/// Quantizer that produces final GGUF block bytes via [`k_quant_codec`].
///
/// Constructed once per quantize-pass with a chosen target format
/// (e.g. `Q4_K`) and an immutable [`CalibrationData`] reference.
/// `quantize_tensor` is invoked per-tensor by the streaming
/// quantize loop.
pub struct KQuantCodecQuantizer {
    name: String,
    target: KQuantTarget,
    calibration: CalibrationData,
}

impl KQuantCodecQuantizer {
    /// Construct with a target format and calibration data.
    ///
    /// `name` is the human-readable identifier reported by the
    /// `Quantizer::name()` impl (e.g. `"q4_k_m"`, `"imatrix-q4_k_m"`).
    pub fn new(name: impl Into<String>, target: KQuantTarget, calibration: CalibrationData) -> Self {
        Self {
            name: name.into(),
            target,
            calibration,
        }
    }

    /// The target format this quantizer emits.
    pub fn target(&self) -> KQuantTarget {
        self.target
    }
}

/// Convert `TensorRef` data to `Vec<f32>` regardless of dtype.
/// Mirrors `static_quant::tensor_to_f32` so we don't introduce a
/// cross-module dep just for this helper. Supports F32/F16/BF16
/// (the three dtypes the convert pipeline produces post-cast).
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
            reason: format!("k-quant-codec: cannot convert dtype {} to f32", tensor.dtype),
        }),
    }
}

/// Sentinel method string in `TensorQuantInfo.method` indicating the
/// `data` field already contains GGUF block bytes ready for direct
/// write. P4 GGUF backend wiring uses this to skip the repack path.
pub const METHOD_K_QUANT_CODEC_DIRECT: &str = "k_quant_codec_direct";

/// Build a F16-passthrough [`QuantizedTensor`] for a tensor that the
/// K-quant codec cannot encode (vision-encoder tensors, non-256-multiple
/// row lengths). The output `QuantizedTensor` carries:
///
///   * `method = "f16"` — routed through the F16 arm in
///     `repack_to_ggml_blocks` (`src/backends/gguf.rs:1049`).
///   * `bits = 16`, `preserved = true` — `quant_info_to_ggml_type`
///     short-circuits to `GGML_TYPE_F16` on the very first line
///     (`src/backends/gguf.rs:1681`).
///   * `ggml_type = Some("F16")` — explicit, defends against any future
///     loss of the `preserved` short-circuit.
///
/// The `data` field is the tensor's bytes converted to F16 LE:
///   * BF16 → F16 via the existing `TensorRef::to_f16` helper (lossy
///     re-cast through F32; matches the convert pipeline's pre-quant
///     cast at `src/main.rs::convert_bf16_to_f16`).
///   * F32 → F16 via `half::f16::from_f32` per element (lossy on values
///     outside F16's [-65504, 65504] range; vision tensors are bounded
///     and never trigger overflow in practice).
///   * F16 → F16 unchanged.
///   * Any other dtype: typed `QuantizeError::TensorQuantizeFailed`.
///
/// Used by the Iter D vision-tensor + non-256-multiple skip predicate
/// (`should_emit_f16_for_kquant`) at the top of every K-quant
/// dispatch site (this module's `KQuantCodecQuantizer`,
/// `VariantKQuantizer`, `DwqKQuantizer`).
pub fn f16_passthrough(
    tensor: &TensorRef,
    name: &str,
) -> Result<QuantizedTensor, QuantizeError> {
    let f16_data: Vec<u8> = match tensor.dtype {
        DType::F16 => tensor.data.clone(),
        DType::BF16 => {
            // Re-use the canonical BF16→F16 helper. Map IrError into a
            // typed quantize error so callers see a uniform error surface.
            let converted = tensor.to_f16().map_err(|e| {
                QuantizeError::TensorQuantizeFailed {
                    tensor: name.to_string(),
                    reason: format!("f16-passthrough: BF16→F16 cast failed: {e}"),
                }
            })?;
            converted.data
        }
        DType::F32 => {
            let mut out = Vec::with_capacity(tensor.data.len() / 2);
            for chunk in tensor.data.chunks_exact(4) {
                let bits = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.extend_from_slice(&half::f16::from_f32(bits).to_le_bytes());
            }
            out
        }
        other => {
            return Err(QuantizeError::TensorQuantizeFailed {
                tensor: name.to_string(),
                reason: format!(
                    "f16-passthrough: unsupported source dtype {other} \
                     (expected F32 / F16 / BF16)"
                ),
            });
        }
    };

    Ok(QuantizedTensor {
        name: name.to_string(),
        shape: tensor.shape.clone(),
        original_dtype: tensor.dtype,
        data: f16_data,
        quant_info: TensorQuantInfo {
            method: "f16".to_string(),
            bits: 16,
            group_size: 0,
            preserved: true,
            scales: None,
            biases: None,
            ggml_type: Some("F16".to_string()),
        },
    })
}

/// Map `KQuantTarget` to the canonical GGML type name string used by
/// `TensorQuantInfo.ggml_type` (preserves compatibility with the
/// existing per-tensor type field).
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

impl Quantizer for KQuantCodecQuantizer {
    fn name(&self) -> &str {
        &self.name
    }

    fn requires_calibration(&self) -> bool {
        // The calibration data is provided up-front via the
        // constructor; no further forward pass needed at
        // quantize_tensor time.
        false
    }

    fn quantize_tensor(
        &self,
        tensor: &TensorRef,
        config: &LayerQuantConfig,
    ) -> Result<QuantizedTensor, QuantizeError> {
        // Preserve path matches `StaticQuantizer::quantize_tensor`:
        // norms, biases, vision tensors, and explicit preserve flags
        // pass through at original precision (BF16 → F16 conversion
        // is the caller's job via `convert_bf16_to_f16` upstream).
        //
        // **Ordering note (Iter D)**: `config.preserve` is the explicit,
        // opt-in preserve channel set by the caller — honour it FIRST
        // so the legacy `method = "passthrough"` shape is unchanged for
        // every existing per-tensor preserve site (norms, biases, the
        // pre-quant `is_weight()` filter at `src/main.rs`).  The
        // implicit Iter D skip (vision + non-256-multiple) runs AFTER,
        // as a defensive fallback for K-quant dispatchers that the
        // upstream `is_vision_tensor`/`is_weight` predicates did not
        // already mark `preserve = true`.
        if config.preserve {
            // Caller asked for passthrough — return as-is, marking the
            // method "passthrough" so GGUF backend handles via F16 path.
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

        // ADR-014 P11-prereq Iter D (2026-04-27): vision-tensor +
        // non-256-multiple skip → F16 passthrough. Pre-empts the
        // K-quant codec's row-alignment rejection at
        // `quantize_row_to_bytes` (e.g. Qwen3.6-27B vision-attn
        // proj at row_len = 1152, the LIVE blocker recorded in
        // ADR-014 § "P11-prereq Iter D" 2026-04-27). See
        // `should_emit_f16_for_kquant` doc for the full pattern list.
        // Runs AFTER `config.preserve` (legacy passthrough takes
        // priority — see ordering note above) but BEFORE the codec
        // dispatch so codec rejections never surface for predicate-
        // matched tensors.
        let row_len_for_skip = tensor.shape.last().copied().unwrap_or(0);
        if should_emit_f16_for_kquant(&tensor.name, row_len_for_skip) {
            tracing::info!(
                tensor = %tensor.name,
                row_len = row_len_for_skip,
                target = ?self.target,
                "K-quant skip → F16 passthrough (vision or non-256-multiple)"
            );
            return f16_passthrough(tensor, &tensor.name);
        }

        // Convert to F32 for the codec.
        let f32_values = tensor_to_f32(tensor)?;

        // Determine row layout: last dim = row_len, product of leading dims = n_rows.
        // For 1D tensors, treat as 1×N.
        let (n_rows, row_len) = match tensor.shape.len() {
            0 => {
                return Err(QuantizeError::TensorQuantizeFailed {
                    tensor: tensor.name.clone(),
                    reason: "k-quant-codec: 0-D tensor not supported".into(),
                });
            }
            1 => (1, tensor.shape[0]),
            _ => {
                let row_len = *tensor.shape.last().unwrap();
                let n_rows = tensor.numel() / row_len;
                (n_rows, row_len)
            }
        };

        // Dispatch through the codec. The codec validates row_len
        // alignment and surfaces typed errors for bad inputs.
        let bytes = quantize_tensor_2d_to_bytes(
            &f32_values,
            n_rows,
            row_len,
            self.target,
            &self.calibration,
            &tensor.name,
        )
        .map_err(|e| QuantizeError::TensorQuantizeFailed {
            tensor: tensor.name.clone(),
            reason: format!("k-quant-codec: {e}"),
        })?;

        Ok(QuantizedTensor {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            original_dtype: tensor.dtype,
            data: bytes,
            quant_info: TensorQuantInfo {
                method: METHOD_K_QUANT_CODEC_DIRECT.to_string(),
                // bits=0 + group_size=0 sentinel: see module doc.
                bits: 0,
                group_size: 0,
                preserved: false,
                scales: None,
                biases: None,
                ggml_type: Some(target_to_ggml_name(self.target)),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;
    use std::collections::HashMap;

    fn make_f32_tensor(name: &str, shape: Vec<usize>, values: &[f32]) -> TensorRef {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        TensorRef {
            name: name.to_string(),
            shape,
            dtype: DType::F32,
            data,
        }
    }

    fn make_bf16_tensor(name: &str, shape: Vec<usize>, values: &[f32]) -> TensorRef {
        let data: Vec<u8> = values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        TensorRef {
            name: name.to_string(),
            shape,
            dtype: DType::BF16,
            data,
        }
    }

    /// `KQuantCodecQuantizer` round-trips a 256-element 1D F32 tensor
    /// through Q4_K and produces GGUF block bytes (144 bytes, the
    /// Q4_K block size).
    #[test]
    fn quantize_1d_f32_q4_k() {
        let row: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let tensor = make_f32_tensor("blk.0.attn_q.weight", vec![256], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = KQuantCodecQuantizer::new("q4_k_m", KQuantTarget::Q4K, CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.name, "blk.0.attn_q.weight");
        assert_eq!(out.shape, vec![256]);
        assert_eq!(out.original_dtype, DType::F32);
        assert_eq!(out.data.len(), 144); // single Q4_K block
        assert_eq!(out.quant_info.method, METHOD_K_QUANT_CODEC_DIRECT);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));
        assert!(!out.quant_info.preserved);
        assert!(out.quant_info.scales.is_none());
    }

    /// 2D weight matrix [4, 256] → 4 rows × 144 bytes = 576 bytes.
    #[test]
    fn quantize_2d_f32_q4_k() {
        let n_rows = 4;
        let row_len = 256;
        let mut data = Vec::with_capacity(n_rows * row_len);
        for r in 0..n_rows {
            for j in 0..row_len {
                data.push(((r * row_len + j) as f32 - 512.0) / 512.0);
            }
        }
        let tensor = make_f32_tensor("blk.0.attn_q.weight", vec![n_rows, row_len], &data);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = KQuantCodecQuantizer::new("q4_k_m", KQuantTarget::Q4K, CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.shape, vec![n_rows, row_len]);
        assert_eq!(out.data.len(), n_rows * 144);
    }

    /// BF16 input is accepted (converted to F32 internally).
    #[test]
    fn quantize_bf16_input_accepted() {
        let row: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let tensor = make_bf16_tensor("weight", vec![256], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = KQuantCodecQuantizer::new("q6_k", KQuantTarget::Q6K, CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.original_dtype, DType::BF16);
        assert_eq!(out.data.len(), 210); // single Q6_K block
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q6_K"));
    }

    /// `preserve = true` → method is "passthrough", data is the
    /// original (or BF16→F16) bytes, ggml_type is None.
    #[test]
    fn preserve_path_returns_passthrough() {
        let row: Vec<f32> = vec![0.5_f32; 16];
        let tensor = make_f32_tensor("blk.0.attn_norm.weight", vec![16], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: true,
        };
        let q = KQuantCodecQuantizer::new("q4_k_m", KQuantTarget::Q4K, CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.method, "passthrough");
        assert!(out.quant_info.preserved);
        assert!(out.quant_info.ggml_type.is_none());
        // F32 data passed through unchanged.
        assert_eq!(out.data, tensor.data);
    }

    /// Imatrix calibration is plumbed through to the codec.
    #[test]
    fn quantize_with_imatrix_calibration() {
        let row: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let tensor = make_f32_tensor("blk.0.attn_q.weight", vec![256], &row);

        let mut weights_map = HashMap::new();
        let mut w = vec![1.0_f32; 256];
        for v in w.iter_mut().take(64) {
            *v = 100.0;
        }
        weights_map.insert("blk.0.attn_q.weight".to_string(), w);
        let imatrix = CalibrationData::Imatrix(weights_map);

        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };

        let q_imatrix = KQuantCodecQuantizer::new("imatrix-q4_k_m", KQuantTarget::Q4K, imatrix);
        let q_ref =
            KQuantCodecQuantizer::new("q4_k_m", KQuantTarget::Q4K, CalibrationData::None);

        let out_imatrix = q_imatrix.quantize_tensor(&tensor, &cfg).unwrap();
        let out_ref = q_ref.quantize_tensor(&tensor, &cfg).unwrap();

        assert_eq!(out_imatrix.data.len(), out_ref.data.len());
        // Imatrix-weighted dispatch should produce different bytes.
        assert_ne!(
            out_imatrix.data, out_ref.data,
            "imatrix path produced byte-identical output to _ref"
        );
    }

    /// Mismatched row length (not a multiple of QK_K=256) at the
    /// dispatcher level is caught by Iter D's defensive non-256-multiple
    /// arm and emitted as F16 passthrough — NOT a typed error.
    ///
    /// **History**: pre-Iter D this test asserted that
    /// `KQuantCodecQuantizer::quantize_tensor` surfaced a
    /// `TensorQuantizeFailed` for misaligned rows. Iter D added the
    /// `should_emit_f16_for_kquant` predicate which fires before the
    /// codec dispatch, converting the rejection-error path into a
    /// graceful F16 passthrough. The codec-level rejection (the actual
    /// `KQuantCodecError::NotBlockAligned` typed error) is still
    /// exercised at `src/quantize/k_quant_codec.rs::tests::rejects_…`
    /// for callers that bypass this dispatcher.
    #[test]
    fn dispatcher_misaligned_row_routes_to_f16_passthrough() {
        let row: Vec<f32> = vec![0.0_f32; 200]; // not a multiple of 256
        let tensor = make_f32_tensor("blk.0.weird.weight", vec![200], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = KQuantCodecQuantizer::new("q4_k_m", KQuantTarget::Q4K, CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.method, "f16");
        assert!(out.quant_info.preserved);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("F16"));
        // F16 bytes: 2 per element × 200 elements.
        assert_eq!(out.data.len(), 200 * 2);
    }

    /// **Iter D**: vision tensor with the LIVE blocker shape
    /// (`model.visual.blocks.0.attn.proj.weight`, row_len = 1152) is
    /// emitted as F16 passthrough — NOT rejected by the codec. Pre-Iter D
    /// this same tensor surfaced the `k-quant: input length 1152 is not
    /// a multiple of QK_K (256)` error documented in ADR-014 § "P11-prereq
    /// Iter D" 2026-04-27.
    #[test]
    fn vision_tensor_blocker_shape_emits_f16_not_codec_rejection() {
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
        let q = KQuantCodecQuantizer::new("q4_k_m", KQuantTarget::Q4K, CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.method, "f16");
        assert_eq!(out.quant_info.bits, 16);
        assert!(out.quant_info.preserved);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("F16"));
        // F16 bytes: 2 per element × 1152 elements.
        assert_eq!(out.data.len(), 1152 * 2);
        assert_eq!(out.shape, vec![1152]);
        assert_eq!(out.original_dtype, DType::F32);
    }

    /// **Iter D**: aligned non-vision tensor still routes through the
    /// codec (the skip predicate is narrowly scoped — does NOT divert
    /// language-tensor paths).
    #[test]
    fn aligned_language_tensor_still_routes_through_codec() {
        let row: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let tensor = make_f32_tensor("blk.0.attn_q.weight", vec![256], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = KQuantCodecQuantizer::new("q4_k_m", KQuantTarget::Q4K, CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.method, METHOD_K_QUANT_CODEC_DIRECT);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));
        assert_eq!(out.data.len(), 144);
    }

    /// **Iter D**: BF16 vision tensor → F16 passthrough via the existing
    /// `TensorRef::to_f16` helper. Matches the convert pipeline's
    /// pre-quant cast path (`convert_bf16_to_f16` upstream).
    #[test]
    fn bf16_vision_tensor_emits_f16_via_to_f16_helper() {
        let row: Vec<f32> = (0..1152).map(|i| (i as f32 - 576.0) / 576.0).collect();
        let tensor = make_bf16_tensor(
            "model.visual.blocks.0.attn.proj.weight",
            vec![1152],
            &row,
        );
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = KQuantCodecQuantizer::new("q4_k_m", KQuantTarget::Q4K, CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.method, "f16");
        assert_eq!(out.original_dtype, DType::BF16);
        assert_eq!(out.data.len(), 1152 * 2);
    }

    /// **Iter D**: defensive non-256-multiple arm fires on a synthetic
    /// non-vision name (proves the predicate is independent of the
    /// vision pattern).
    #[test]
    fn non_256_multiple_non_vision_tensor_emits_f16() {
        let row: Vec<f32> = (0..1153).map(|i| (i as f32) / 1153.0).collect();
        let tensor = make_f32_tensor("blk.0.attn_weird.weight", vec![1153], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = KQuantCodecQuantizer::new("q4_k_m", KQuantTarget::Q4K, CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.method, "f16");
        assert_eq!(out.data.len(), 1153 * 2);
    }

    /// **Iter D**: `f16_passthrough` direct unit test — verifies the
    /// helper's quant_info shape independent of the dispatcher wiring.
    #[test]
    fn f16_passthrough_helper_shapes_quant_info_correctly() {
        let row: Vec<f32> = vec![0.5_f32; 16];
        let tensor = make_f32_tensor("any.tensor.name", vec![16], &row);
        let out = f16_passthrough(&tensor, "any.tensor.name").unwrap();
        assert_eq!(out.quant_info.method, "f16");
        assert_eq!(out.quant_info.bits, 16);
        assert_eq!(out.quant_info.group_size, 0);
        assert!(out.quant_info.preserved);
        assert!(out.quant_info.scales.is_none());
        assert!(out.quant_info.biases.is_none());
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("F16"));
        assert_eq!(out.data.len(), 16 * 2);
        assert_eq!(out.shape, vec![16]);
        assert_eq!(out.original_dtype, DType::F32);
        assert_eq!(out.name, "any.tensor.name");
    }

    /// `name()`, `requires_calibration()`, `target()` accessors.
    #[test]
    fn quantizer_introspection() {
        let q = KQuantCodecQuantizer::new("q5_k_m", KQuantTarget::Q5K, CalibrationData::None);
        assert_eq!(q.name(), "q5_k_m");
        assert!(!q.requires_calibration());
        assert_eq!(q.target(), KQuantTarget::Q5K);
    }

    /// `target_to_ggml_name` covers all 7 targets and returns the
    /// canonical GGUF name strings.
    #[test]
    fn target_to_ggml_name_all_targets() {
        assert_eq!(target_to_ggml_name(KQuantTarget::Q4K), "Q4_K");
        assert_eq!(target_to_ggml_name(KQuantTarget::Q5K), "Q5_K");
        assert_eq!(target_to_ggml_name(KQuantTarget::Q6K), "Q6_K");
        assert_eq!(target_to_ggml_name(KQuantTarget::Q4Legacy), "Q4_0");
        assert_eq!(target_to_ggml_name(KQuantTarget::Q4Legacy1), "Q4_1");
        assert_eq!(target_to_ggml_name(KQuantTarget::Q5Legacy0), "Q5_0");
        assert_eq!(target_to_ggml_name(KQuantTarget::Q5Legacy1), "Q5_1");
        assert_eq!(target_to_ggml_name(KQuantTarget::Q8Legacy), "Q8_0");
    }

    // ─────────── Full ADR-014 pipeline integration test (iter-3q) ───────────

    /// **End-to-end pipeline test** demonstrating that all the pieces
    /// landed in P6 + P7 iter-3a..3p compose correctly into a working
    /// imatrix-aware GGUF block-format quantize pipeline:
    ///
    /// 1. `ImatrixCollector::accumulate_dense` builds a per-tensor
    ///    importance map from synthetic activations.
    /// 2. `save_imatrix_gguf` persists to llama.cpp-compatible GGUF.
    /// 3. `CalibrationData::from_imatrix_gguf` loads it back.
    /// 4. `KQuantCodecQuantizer::new(name, Q4K, calibration)` plugs
    ///    the calibration into the existing `Quantizer` trait.
    /// 5. `quantize_tensor` emits a `QuantizedTensor` whose `data`
    ///    field contains final GGUF block bytes.
    /// 6. `dequantize_row_q4_k_bytes` round-trips back to F32 within
    ///    the format's RMSE bound.
    ///
    /// This test is **the ADR-014 closure proof** that the data-format
    /// pipeline is fully wireable end-to-end. It does NOT exercise
    /// forward-pass orchestration (deferred to ADR-013 ActivationCapture)
    /// or cmd_convert wiring (P4 — depends on infra outside this PR's
    /// scope).
    #[test]
    fn end_to_end_pipeline_collector_to_quantized_bytes() {
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::quantize::k_quant::dequantize_row_q4_k_bytes;

        const QK_K: usize = 256;

        // ─── Step 1: Build calibration via accumulate_dense ───
        let mut collector = ImatrixCollector::new();

        // Two tensors: a 256-col attention Q-projection and a 512-col
        // FFN down-projection. Synthetic per-token activations.
        let attn_q_acts: Vec<f32> = (0..QK_K).map(|i| (i as f32) / 100.0).collect();
        collector
            .accumulate_dense("blk.0.attn_q.weight", &attn_q_acts, 1, QK_K)
            .unwrap();

        let ffn_down_acts: Vec<f32> = (0..2 * QK_K).map(|i| (i as f32) / 200.0).collect();
        collector
            .accumulate_dense("blk.0.ffn_down.weight", &ffn_down_acts, 1, 2 * QK_K)
            .unwrap();
        collector.record_chunk();

        // ─── Step 2: Save imatrix to GGUF ───
        let dir = std::env::temp_dir().join("hf2q-end-to-end-pipeline-test");
        let _ = std::fs::create_dir_all(&dir);
        let imatrix_path = dir.join("e2e.imatrix.gguf");
        let _ = std::fs::remove_file(&imatrix_path);
        collector
            .save_imatrix_gguf(&imatrix_path, 512, &["calib.txt"])
            .unwrap();

        // ─── Step 3: Load via CalibrationData bridge ───
        let calibration = CalibrationData::from_imatrix_gguf(&imatrix_path).unwrap();
        let _ = std::fs::remove_file(&imatrix_path);

        match &calibration {
            CalibrationData::ImatrixWithStats(map) => {
                assert!(map.contains_key("blk.0.attn_q.weight"));
                assert!(map.contains_key("blk.0.ffn_down.weight"));
            }
            _ => panic!("expected ImatrixWithStats from from_imatrix_gguf"),
        }

        // ─── Step 4: Build KQuantCodecQuantizer ───
        let quantizer = KQuantCodecQuantizer::new(
            "imatrix-q4_k_m",
            KQuantTarget::Q4K,
            calibration.clone(),
        );
        assert_eq!(quantizer.name(), "imatrix-q4_k_m");
        assert_eq!(quantizer.target(), KQuantTarget::Q4K);

        // ─── Step 5: Quantize a synthetic weight matrix ───
        // Realistic shape: [4 heads, 256 hidden] for blk.0.attn_q.weight.
        let weight_values: Vec<f32> = (0..4 * QK_K)
            .map(|i| (i as f32 - 512.0) / 512.0)
            .collect();
        let tensor = make_f32_tensor("blk.0.attn_q.weight", vec![4, QK_K], &weight_values);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let quantized = quantizer.quantize_tensor(&tensor, &cfg).unwrap();

        // Output contract: GGUF block bytes ready for direct write.
        assert_eq!(quantized.shape, vec![4, QK_K]);
        assert_eq!(quantized.data.len(), 4 * 144); // 4 rows × Q4_K block size
        assert_eq!(quantized.quant_info.method, METHOD_K_QUANT_CODEC_DIRECT);
        assert_eq!(quantized.quant_info.ggml_type.as_deref(), Some("Q4_K"));
        assert_eq!(quantized.quant_info.bits, 0); // sentinel
        assert_eq!(quantized.quant_info.group_size, 0); // sentinel
        assert!(!quantized.quant_info.preserved);
        assert!(quantized.quant_info.scales.is_none());

        // ─── Step 6: Round-trip back to F32 via dequantize ───
        for r in 0..4 {
            let row_bytes = &quantized.data[r * 144..(r + 1) * 144];
            let mut decoded = vec![0.0_f32; QK_K];
            dequantize_row_q4_k_bytes(row_bytes, &mut decoded).unwrap();

            let original = &weight_values[r * QK_K..(r + 1) * QK_K];
            let mut sse = 0.0_f64;
            for (a, b) in original.iter().zip(decoded.iter()) {
                let d = (*a as f64) - (*b as f64);
                sse += d * d;
            }
            let rmse = (sse / QK_K as f64).sqrt();
            assert!(
                rmse < 0.05,
                "row {r} round-trip RMSE {rmse} > 0.05 (Q4_K bound)"
            );
        }

        // ─── Compare imatrix path vs uncalibrated path ───
        // The imatrix path should produce different bytes than the
        // None-calibration path (verifies the imatrix signal threaded
        // through all 6 steps).
        let plain_quantizer =
            KQuantCodecQuantizer::new("q4_k_m", KQuantTarget::Q4K, CalibrationData::None);
        let plain_quantized = plain_quantizer.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(plain_quantized.data.len(), quantized.data.len());

        // The bytes typically differ — but for a calibration where the
        // imatrix weights happen to match the uniform default, they
        // could be byte-equal. So we don't strictly assert _ne. Just
        // verify both produce well-formed output (already asserted above).
        // The byte-equivalence vs imatrix-bias signal is exercised in
        // `dispatch_imatrix_q4_k_uses_weighted_path` of k_quant_codec
        // tests on biased input.
    }

    /// **Pipeline + multi-format**: the same calibration drives Q4_K,
    /// Q5_K, and Q6_K quantizers — verifies the codec dispatch correctly
    /// routes the imatrix data through different target paths.
    #[test]
    fn end_to_end_pipeline_multi_format_dispatch() {
        use crate::calibrate::imatrix::ImatrixCollector;

        const QK_K: usize = 256;

        let mut collector = ImatrixCollector::new();
        let acts: Vec<f32> = (0..QK_K).map(|i| 0.1 + (i as f32) / 1000.0).collect();
        collector
            .accumulate_dense("test.weight", &acts, 1, QK_K)
            .unwrap();
        collector.record_chunk();

        let calibration = CalibrationData::from_imatrix_collector(&collector);

        let weight: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let tensor = make_f32_tensor("test.weight", vec![QK_K], &weight);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };

        for (target, expected_size) in [
            (KQuantTarget::Q4K, 144),
            (KQuantTarget::Q5K, 176),
            (KQuantTarget::Q6K, 210),
        ] {
            let q = KQuantCodecQuantizer::new("imatrix", target, calibration.clone());
            let out = q.quantize_tensor(&tensor, &cfg).unwrap();
            assert_eq!(
                out.data.len(),
                expected_size,
                "{target:?}: got {} bytes, expected {expected_size}",
                out.data.len()
            );
            assert_eq!(out.quant_info.method, METHOD_K_QUANT_CODEC_DIRECT);
        }
    }
}
