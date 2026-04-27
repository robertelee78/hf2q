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

use std::collections::HashMap;

use crate::calibrate::calibrator::CalibrationData;
use crate::ir::{DType, QuantizedTensor, TensorQuantInfo, TensorRef};
use crate::quantize::k_quant_codec::{quantize_tensor_2d_to_bytes, KQuantTarget};
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

/// Map `KQuantTarget` to the canonical GGML type name string used by
/// `TensorQuantInfo.ggml_type` (preserves compatibility with the
/// existing per-tensor type field).
fn target_to_ggml_name(target: KQuantTarget) -> String {
    match target {
        KQuantTarget::Q4K => "Q4_K".to_string(),
        KQuantTarget::Q5K => "Q5_K".to_string(),
        KQuantTarget::Q6K => "Q6_K".to_string(),
        KQuantTarget::Q4Legacy => "Q4_0".to_string(),
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

    /// Mismatched row length (not a multiple of QK_K=256) surfaces
    /// a typed `TensorQuantizeFailed` error.
    #[test]
    fn rejects_misaligned_row() {
        let row: Vec<f32> = vec![0.0_f32; 200]; // not a multiple of 256
        let tensor = make_f32_tensor("blk.0.weird.weight", vec![200], &row);
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        };
        let q = KQuantCodecQuantizer::new("q4_k_m", KQuantTarget::Q4K, CalibrationData::None);
        let err = q.quantize_tensor(&tensor, &cfg).unwrap_err();
        match err {
            QuantizeError::TensorQuantizeFailed { tensor: name, reason } => {
                assert_eq!(name, "blk.0.weird.weight");
                assert!(reason.contains("k-quant-codec"));
            }
            _ => panic!("expected TensorQuantizeFailed"),
        }
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
        assert_eq!(target_to_ggml_name(KQuantTarget::Q5Legacy0), "Q5_0");
        assert_eq!(target_to_ggml_name(KQuantTarget::Q5Legacy1), "Q5_1");
        assert_eq!(target_to_ggml_name(KQuantTarget::Q8Legacy), "Q8_0");
    }
}
