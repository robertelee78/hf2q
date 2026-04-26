//! `k_quant_codec` — calibration-aware dispatch over the k-quant
//! codebook ports (ADR-014 P7 Decision 11). Bridges
//! [`crate::calibrate::calibrator::CalibrationData`] (None / Imatrix /
//! Dwq) with the per-target-format flat-bytes wrappers in
//! [`crate::quantize::k_quant`].
//!
//! ## Why this layer exists
//!
//! P7 iters 3a–3f land the **algorithmic** k-quant codebook (block
//! layouts, dequantize, _ref quantize, imatrix quantize, flat-bytes
//! wrappers). What's missing is the **dispatch shim** that:
//!
//! 1. Maps a target GGUF format ([`KQuantTarget`]) to the right
//!    quantize entry point (Q4_K vs Q5_K vs Q6_K).
//! 2. Decides whether to use the `_ref` or `_imatrix` variant based on
//!    [`CalibrationData`].
//! 3. Looks up per-tensor imatrix weights from the calibrator's
//!    `HashMap<String, Vec<f32>>` keyed by tensor name.
//!
//! Once this lands, the streaming convert pipeline's tensor-emit loop
//! (P4 wiring) can call `quantize_row_to_bytes(row, target, calib,
//! name)` and not care about the target-format-specific quantize
//! function names. The calibrator stays orthogonal: same
//! [`CalibrationData`] payload routes to any of the three k-quant
//! formats.
//!
//! ## Sovereignty
//!
//! Pure Rust. No FFI, no runtime link to libggml. Dispatches into
//! [`crate::quantize::k_quant`] which itself is pure Rust.

use thiserror::Error;

use crate::calibrate::calibrator::CalibrationData;
use crate::quantize::k_quant::{
    quantize_row_q4_k_imatrix_to_bytes, quantize_row_q4_k_to_bytes,
    quantize_row_q5_k_imatrix_to_bytes, quantize_row_q5_k_to_bytes,
    quantize_row_q6_k_imatrix_to_bytes, quantize_row_q6_k_to_bytes, KQuantError,
};
use crate::quantize::q_legacy::{
    quantize_row_q4_0_to_bytes, quantize_row_q5_0_to_bytes, quantize_row_q5_1_to_bytes,
    quantize_row_q8_0_to_bytes, QLegacyError,
};

/// Target GGUF block format. Spans the K-family (Q4_K/Q5_K/Q6_K) plus
/// the legacy fallback chain (Q4_0/Q5_0/Q5_1/Q8_0). Maps to llama.cpp's
/// `GGML_TYPE_*` constants.
///
/// The `_M`/`_S` K-family variants share a block layout (and thus a
/// target here) — the suffix only affects which tensors get which
/// target in the layer-mix policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KQuantTarget {
    // K-family (256-element super-blocks).
    /// `Q4_K_M` / `Q4_K_S` / `Q4_K`. 4.5 bpw, super-block scale +
    /// per-sub-block min term, 8 sub-blocks of 32 elements.
    Q4K,
    /// `Q5_K_M` / `Q5_K_S` / `Q5_K`. 5.5 bpw, same shape as Q4_K.
    Q5K,
    /// `Q6_K`. 6.5625 bpw, signed sub-block scales, 16 sub-blocks of
    /// 16 elements (no per-sub-block min term).
    Q6K,
    // Legacy fallback chain (32-element blocks).
    /// `Q4_0`. 4.5 bpw, symmetric (`max / -8`), 32-element blocks.
    /// Fallback for `Q3_K`/`Q2_K` rows that aren't 256-aligned.
    Q4Legacy,
    /// `Q5_0`. 5.5 bpw, symmetric (`max / -16`), 32-element blocks.
    /// Fallback for `Q4_K` rows.
    Q5Legacy0,
    /// `Q5_1`. 6.0 bpw, asymmetric (per-block min), 32-element blocks.
    /// Fallback for `Q5_K` rows.
    Q5Legacy1,
    /// `Q8_0`. 8.5 bpw, symmetric, 32-element blocks. Fallback for
    /// `Q6_K` rows.
    Q8Legacy,
}

impl KQuantTarget {
    /// Construct from a llama.cpp `GGML_TYPE_*` constant. Returns
    /// `None` for non-block types (F32/F16/IQ family, etc).
    pub fn from_ggml_type(ggml_type: u32) -> Option<Self> {
        match ggml_type {
            2 => Some(Self::Q4Legacy),  // GGML_TYPE_Q4_0
            6 => Some(Self::Q5Legacy0), // GGML_TYPE_Q5_0
            7 => Some(Self::Q5Legacy1), // GGML_TYPE_Q5_1
            8 => Some(Self::Q8Legacy),  // GGML_TYPE_Q8_0
            12 => Some(Self::Q4K),      // GGML_TYPE_Q4_K
            13 => Some(Self::Q5K),      // GGML_TYPE_Q5_K
            14 => Some(Self::Q6K),      // GGML_TYPE_Q6_K
            _ => None,
        }
    }

    /// llama.cpp's `GGML_TYPE_*` constant for this target.
    pub fn ggml_type(&self) -> u32 {
        match self {
            Self::Q4Legacy => 2,
            Self::Q5Legacy0 => 6,
            Self::Q5Legacy1 => 7,
            Self::Q8Legacy => 8,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
        }
    }

    /// Bytes per block on disk.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::Q4K => crate::quantize::k_quant::BLOCK_Q4_K_SIZE,
            Self::Q5K => crate::quantize::k_quant::BLOCK_Q5_K_SIZE,
            Self::Q6K => crate::quantize::k_quant::BLOCK_Q6_K_SIZE,
            Self::Q4Legacy => crate::quantize::q_legacy::BLOCK_Q4_0_SIZE,
            Self::Q5Legacy0 => crate::quantize::q_legacy::BLOCK_Q5_0_SIZE,
            Self::Q5Legacy1 => crate::quantize::q_legacy::BLOCK_Q5_1_SIZE,
            Self::Q8Legacy => crate::quantize::q_legacy::BLOCK_Q8_0_SIZE,
        }
    }

    /// Number of elements per block. K-family: 256. Legacy: 32.
    pub fn elements_per_block(&self) -> usize {
        match self {
            Self::Q4K | Self::Q5K | Self::Q6K => crate::quantize::k_quant::QK_K,
            Self::Q4Legacy => crate::quantize::q_legacy::QK4_0,
            Self::Q5Legacy0 => crate::quantize::q_legacy::QK5_0,
            Self::Q5Legacy1 => crate::quantize::q_legacy::QK5_1,
            Self::Q8Legacy => crate::quantize::q_legacy::QK8_0,
        }
    }

    /// Whether this target supports imatrix-weighted codebook search.
    /// Only K-family does — legacy formats use the `_ref` path.
    pub fn supports_imatrix(&self) -> bool {
        matches!(self, Self::Q4K | Self::Q5K | Self::Q6K)
    }

    /// Bits per weight (effective storage cost). Includes per-block
    /// metadata overhead.
    pub fn bpw(&self) -> f32 {
        match self {
            Self::Q4K => 4.5,
            Self::Q5K => 5.5,
            Self::Q6K => 6.5625,
            Self::Q4Legacy => 4.5,
            Self::Q5Legacy0 => 5.5,
            Self::Q5Legacy1 => 6.0,
            Self::Q8Legacy => 8.5,
        }
    }
}

/// Errors from the k-quant codec dispatch.
#[derive(Error, Debug)]
pub enum KQuantCodecError {
    /// Underlying K-family quantize error.
    #[error("k-quant codec: {0}")]
    KQuant(#[from] KQuantError),

    /// Underlying legacy-format quantize error.
    #[error("k-quant codec: {0}")]
    QLegacy(#[from] QLegacyError),

    /// Calibration data has imatrix weights for a tensor, but the
    /// length doesn't match the row length expected by the codebook.
    #[error(
        "k-quant codec: tensor '{tensor}' has imatrix weights of length {weights_len}, \
         expected {row_len}"
    )]
    ImatrixWeightsLengthMismatch {
        tensor: String,
        weights_len: usize,
        row_len: usize,
    },

    /// `CalibrationData::Dwq` was passed but the caller didn't expect
    /// it. DWQ sensitivity is per-layer, applied at the bit-pair-
    /// allocation phase, not at the row-quantize phase. The codec
    /// surfaces this as an explicit error rather than silently
    /// downgrading to `_ref`.
    #[error(
        "k-quant codec: tensor '{tensor}' was given DWQ calibration \
         but DWQ doesn't apply at row-quantize time"
    )]
    DwqAtRowQuantize { tensor: String },
}

/// Look up imatrix weights for `tensor_name` in `calibration`. Returns
/// `Some(weights)` when the calibrator has them, `None` for the
/// `_ref` (no calibration) path.
fn lookup_imatrix_weights<'a>(
    calibration: &'a CalibrationData,
    tensor_name: &str,
) -> Option<&'a [f32]> {
    match calibration {
        CalibrationData::None => None,
        CalibrationData::Imatrix(map) => map.get(tensor_name).map(|v| v.as_slice()),
        CalibrationData::ImatrixWithStats(map) => {
            map.get(tensor_name).map(|s| s.values.as_slice())
        }
        CalibrationData::Dwq(_) => None,
    }
}

/// Quantize a row of F32 to k-quant block bytes, dispatching by
/// target format and the presence of imatrix calibration.
///
/// - `row.len()` must be a multiple of `QK_K` (256).
/// - When [`CalibrationData::None`]: dispatches to the `_ref` quantize
///   variant.
/// - When [`CalibrationData::Imatrix`] / [`CalibrationData::ImatrixWithStats`]:
///   looks up `tensor_name` in the calibrator's map. If found, uses
///   the `_imatrix` variant; if not found (tensor has no imatrix
///   data — e.g. embedding layers in some imatrix runs), falls back
///   to `_ref`.
/// - When [`CalibrationData::Dwq`]: returns
///   [`KQuantCodecError::DwqAtRowQuantize`] (DWQ is per-layer
///   sensitivity; doesn't apply at the row-quantize codebook search).
///
/// `tensor_name` is the GGUF tensor name (post-`map_tensor_name_to_gguf`)
/// — the same key the imatrix calibrator uses.
pub fn quantize_row_to_bytes(
    row: &[f32],
    target: KQuantTarget,
    calibration: &CalibrationData,
    tensor_name: &str,
) -> Result<Vec<u8>, KQuantCodecError> {
    if let CalibrationData::Dwq(_) = calibration {
        return Err(KQuantCodecError::DwqAtRowQuantize {
            tensor: tensor_name.to_string(),
        });
    }

    let weights = lookup_imatrix_weights(calibration, tensor_name);
    if let Some(w) = weights {
        if w.len() != row.len() {
            return Err(KQuantCodecError::ImatrixWeightsLengthMismatch {
                tensor: tensor_name.to_string(),
                weights_len: w.len(),
                row_len: row.len(),
            });
        }
    }

    let bytes = match (target, weights) {
        // K-family: imatrix-weighted when available, else _ref.
        (KQuantTarget::Q4K, Some(w)) => quantize_row_q4_k_imatrix_to_bytes(row, w)?,
        (KQuantTarget::Q4K, None) => quantize_row_q4_k_to_bytes(row)?,
        (KQuantTarget::Q5K, Some(w)) => quantize_row_q5_k_imatrix_to_bytes(row, w)?,
        (KQuantTarget::Q5K, None) => quantize_row_q5_k_to_bytes(row)?,
        (KQuantTarget::Q6K, Some(w)) => quantize_row_q6_k_imatrix_to_bytes(row, w)?,
        (KQuantTarget::Q6K, None) => quantize_row_q6_k_to_bytes(row)?,
        // Legacy: always _ref. Imatrix weights silently ignored
        // (legacy formats don't support per-element-weighted codebook search).
        (KQuantTarget::Q4Legacy, _) => quantize_row_q4_0_to_bytes(row)?,
        (KQuantTarget::Q5Legacy0, _) => quantize_row_q5_0_to_bytes(row)?,
        (KQuantTarget::Q5Legacy1, _) => quantize_row_q5_1_to_bytes(row)?,
        (KQuantTarget::Q8Legacy, _) => quantize_row_q8_0_to_bytes(row)?,
    };
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::imatrix::Stats as ImatrixStats;
    use std::collections::HashMap;

    const QK_K: usize = 256;

    fn smooth_ramp(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = (i as f32) / (n as f32 - 1.0);
                -2.0 + 4.0 * t
            })
            .collect()
    }

    /// `KQuantTarget::from_ggml_type` round-trips through `ggml_type()`.
    #[test]
    fn target_ggml_type_round_trip() {
        for t in [
            KQuantTarget::Q4K,
            KQuantTarget::Q5K,
            KQuantTarget::Q6K,
            KQuantTarget::Q4Legacy,
            KQuantTarget::Q5Legacy0,
            KQuantTarget::Q5Legacy1,
            KQuantTarget::Q8Legacy,
        ] {
            let g = t.ggml_type();
            assert_eq!(KQuantTarget::from_ggml_type(g), Some(t), "round trip {t:?}");
        }
        // Non-block types yield None.
        assert_eq!(KQuantTarget::from_ggml_type(0), None); // F32
        assert_eq!(KQuantTarget::from_ggml_type(1), None); // F16
        assert_eq!(KQuantTarget::from_ggml_type(99), None); // unknown
    }

    /// `KQuantTarget::bytes_per_block` matches the llama.cpp
    /// `static_assert`s for every supported target.
    #[test]
    fn target_bytes_per_block() {
        // K-family
        assert_eq!(KQuantTarget::Q4K.bytes_per_block(), 144);
        assert_eq!(KQuantTarget::Q5K.bytes_per_block(), 176);
        assert_eq!(KQuantTarget::Q6K.bytes_per_block(), 210);
        // Legacy
        assert_eq!(KQuantTarget::Q4Legacy.bytes_per_block(), 18);
        assert_eq!(KQuantTarget::Q5Legacy0.bytes_per_block(), 22);
        assert_eq!(KQuantTarget::Q5Legacy1.bytes_per_block(), 24);
        assert_eq!(KQuantTarget::Q8Legacy.bytes_per_block(), 34);
    }

    /// `KQuantTarget::elements_per_block`: 256 for K-family, 32 for legacy.
    #[test]
    fn target_elements_per_block() {
        assert_eq!(KQuantTarget::Q4K.elements_per_block(), 256);
        assert_eq!(KQuantTarget::Q5K.elements_per_block(), 256);
        assert_eq!(KQuantTarget::Q6K.elements_per_block(), 256);
        assert_eq!(KQuantTarget::Q4Legacy.elements_per_block(), 32);
        assert_eq!(KQuantTarget::Q5Legacy0.elements_per_block(), 32);
        assert_eq!(KQuantTarget::Q5Legacy1.elements_per_block(), 32);
        assert_eq!(KQuantTarget::Q8Legacy.elements_per_block(), 32);
    }

    /// `supports_imatrix`: only K-family supports imatrix-weighted
    /// codebook search.
    #[test]
    fn target_supports_imatrix() {
        assert!(KQuantTarget::Q4K.supports_imatrix());
        assert!(KQuantTarget::Q5K.supports_imatrix());
        assert!(KQuantTarget::Q6K.supports_imatrix());
        assert!(!KQuantTarget::Q4Legacy.supports_imatrix());
        assert!(!KQuantTarget::Q5Legacy0.supports_imatrix());
        assert!(!KQuantTarget::Q5Legacy1.supports_imatrix());
        assert!(!KQuantTarget::Q8Legacy.supports_imatrix());
    }

    /// `bpw` returns documented values for all 7 targets.
    #[test]
    fn target_bpw() {
        assert!((KQuantTarget::Q4K.bpw() - 4.5).abs() < 1e-4);
        assert!((KQuantTarget::Q5K.bpw() - 5.5).abs() < 1e-4);
        assert!((KQuantTarget::Q6K.bpw() - 6.5625).abs() < 1e-4);
        assert!((KQuantTarget::Q4Legacy.bpw() - 4.5).abs() < 1e-4);
        assert!((KQuantTarget::Q5Legacy0.bpw() - 5.5).abs() < 1e-4);
        assert!((KQuantTarget::Q5Legacy1.bpw() - 6.0).abs() < 1e-4);
        assert!((KQuantTarget::Q8Legacy.bpw() - 8.5).abs() < 1e-4);
    }

    /// `quantize_row_to_bytes` with `CalibrationData::None` dispatches
    /// to the `_ref` path; output length matches single-block size.
    #[test]
    fn dispatch_none_q4_k() {
        let row = smooth_ramp(QK_K);
        let bytes = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4K,
            &CalibrationData::None,
            "blk.0.attn_q.weight",
        )
        .unwrap();
        assert_eq!(bytes.len(), 144);
    }

    /// `quantize_row_to_bytes` with `CalibrationData::None` for Q5_K.
    #[test]
    fn dispatch_none_q5_k() {
        let row = smooth_ramp(QK_K);
        let bytes = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q5K,
            &CalibrationData::None,
            "blk.0.attn_q.weight",
        )
        .unwrap();
        assert_eq!(bytes.len(), 176);
    }

    /// `quantize_row_to_bytes` with `CalibrationData::None` for Q6_K.
    #[test]
    fn dispatch_none_q6_k() {
        let row = smooth_ramp(QK_K);
        let bytes = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q6K,
            &CalibrationData::None,
            "blk.0.attn_q.weight",
        )
        .unwrap();
        assert_eq!(bytes.len(), 210);
    }

    /// `quantize_row_to_bytes` with `CalibrationData::Imatrix` and
    /// matching tensor key dispatches to the `_imatrix` path.
    /// Verifies the bytes differ from the `_ref` path (different
    /// codebook search yields different bytes).
    #[test]
    fn dispatch_imatrix_q4_k_uses_weighted_path() {
        let row = smooth_ramp(QK_K);
        let mut weights_map = HashMap::new();
        weights_map.insert("blk.0.attn_q.weight".to_string(), vec![1.0_f32; QK_K]);
        let calib = CalibrationData::Imatrix(weights_map);

        let bytes_imatrix = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4K,
            &calib,
            "blk.0.attn_q.weight",
        )
        .unwrap();
        let bytes_none = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4K,
            &CalibrationData::None,
            "blk.0.attn_q.weight",
        )
        .unwrap();

        assert_eq!(bytes_imatrix.len(), bytes_none.len());
        // Different codebook search → bytes typically differ. We don't
        // require they MUST differ on every input (could collide), but
        // for a generic ramp + uniform weights they're expected to.
        assert_ne!(
            bytes_imatrix, bytes_none,
            "imatrix path produced byte-identical output to _ref path on smooth ramp"
        );
    }

    /// `CalibrationData::Imatrix` without an entry for the requested
    /// tensor name falls back to the `_ref` path.
    #[test]
    fn dispatch_imatrix_missing_tensor_falls_back_to_ref() {
        let row = smooth_ramp(QK_K);
        let mut weights_map = HashMap::new();
        weights_map.insert("blk.0.attn_q.weight".to_string(), vec![1.0_f32; QK_K]);
        let calib = CalibrationData::Imatrix(weights_map);

        let bytes_missing = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4K,
            &calib,
            "blk.99.unknown_tensor.weight", // not in map
        )
        .unwrap();
        let bytes_ref = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4K,
            &CalibrationData::None,
            "blk.99.unknown_tensor.weight",
        )
        .unwrap();

        assert_eq!(bytes_missing, bytes_ref, "fallback to _ref on missing tensor");
    }

    /// `CalibrationData::ImatrixWithStats` is consumed identically to
    /// `Imatrix` (uses `Stats::values` field). Verify a wired tensor
    /// produces the same bytes as the equivalent `Imatrix` payload.
    #[test]
    fn dispatch_imatrix_with_stats_equiv() {
        let row = smooth_ramp(QK_K);

        let mut imatrix_map = HashMap::new();
        imatrix_map.insert("blk.0.attn_q.weight".to_string(), vec![1.0_f32; QK_K]);
        let calib_simple = CalibrationData::Imatrix(imatrix_map);

        let mut stats_map = HashMap::new();
        stats_map.insert(
            "blk.0.attn_q.weight".to_string(),
            ImatrixStats {
                values: vec![1.0_f32; QK_K],
                counts: vec![1; QK_K], // placeholder; codec doesn't read counts
            },
        );
        let calib_stats = CalibrationData::ImatrixWithStats(stats_map);

        let bytes_simple = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4K,
            &calib_simple,
            "blk.0.attn_q.weight",
        )
        .unwrap();
        let bytes_stats = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4K,
            &calib_stats,
            "blk.0.attn_q.weight",
        )
        .unwrap();
        assert_eq!(bytes_simple, bytes_stats);
    }

    /// `CalibrationData::Dwq` returns `DwqAtRowQuantize` — DWQ
    /// sensitivity is per-layer, not per-row-quantize.
    #[test]
    fn dispatch_dwq_rejected() {
        let row = smooth_ramp(QK_K);
        let mut dwq_map = HashMap::new();
        dwq_map.insert("blk.0.attn_q.weight".to_string(), vec![0.5_f32]);
        let calib = CalibrationData::Dwq(dwq_map);

        let err = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4K,
            &calib,
            "blk.0.attn_q.weight",
        )
        .unwrap_err();
        match err {
            KQuantCodecError::DwqAtRowQuantize { tensor } => {
                assert_eq!(tensor, "blk.0.attn_q.weight");
            }
            _ => panic!("expected DwqAtRowQuantize"),
        }
    }

    /// Imatrix weights with mismatched length surface a typed error
    /// (length validation prevents downstream silent corruption).
    #[test]
    fn dispatch_imatrix_length_mismatch() {
        let row = smooth_ramp(QK_K);
        let mut weights_map = HashMap::new();
        weights_map.insert(
            "blk.0.attn_q.weight".to_string(),
            vec![1.0_f32; QK_K - 1], // one short
        );
        let calib = CalibrationData::Imatrix(weights_map);

        let err = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4K,
            &calib,
            "blk.0.attn_q.weight",
        )
        .unwrap_err();
        match err {
            KQuantCodecError::ImatrixWeightsLengthMismatch {
                weights_len, row_len, ..
            } => {
                assert_eq!(weights_len, QK_K - 1);
                assert_eq!(row_len, QK_K);
            }
            _ => panic!("expected ImatrixWeightsLengthMismatch"),
        }
    }

    /// Misaligned input rejected with the underlying k-quant error.
    #[test]
    fn dispatch_misaligned_input() {
        let bad = vec![0.0_f32; QK_K - 5];
        let err = quantize_row_to_bytes(
            &bad,
            KQuantTarget::Q4K,
            &CalibrationData::None,
            "blk.0.attn_q.weight",
        )
        .unwrap_err();
        match err {
            KQuantCodecError::KQuant(KQuantError::NotBlockAligned { actual }) => {
                assert_eq!(actual, QK_K - 5);
            }
            _ => panic!("expected KQuant(NotBlockAligned)"),
        }
    }

    // ─────────────── Legacy-format dispatch tests (iter-3j) ───────────────

    /// Q4_0 dispatch: 32 elements → 18 bytes.
    #[test]
    fn dispatch_none_q4_0() {
        let row: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 16.0).collect();
        let bytes = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4Legacy,
            &CalibrationData::None,
            "blk.0.attn_norm.weight",
        )
        .unwrap();
        assert_eq!(bytes.len(), 18);
    }

    /// Q5_0 dispatch: 32 elements → 22 bytes.
    #[test]
    fn dispatch_none_q5_0() {
        let row: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 16.0).collect();
        let bytes = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q5Legacy0,
            &CalibrationData::None,
            "blk.0.attn_norm.weight",
        )
        .unwrap();
        assert_eq!(bytes.len(), 22);
    }

    /// Q5_1 dispatch: 32 elements → 24 bytes.
    #[test]
    fn dispatch_none_q5_1() {
        let row: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0).collect();
        let bytes = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q5Legacy1,
            &CalibrationData::None,
            "blk.0.attn_norm.weight",
        )
        .unwrap();
        assert_eq!(bytes.len(), 24);
    }

    /// Q8_0 dispatch: 32 elements → 34 bytes.
    #[test]
    fn dispatch_none_q8_0() {
        let row: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 16.0).collect();
        let bytes = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q8Legacy,
            &CalibrationData::None,
            "blk.0.attn_norm.weight",
        )
        .unwrap();
        assert_eq!(bytes.len(), 34);
    }

    /// Legacy targets ignore imatrix calibration (no `_impl` variant
    /// in our pure-Rust port). Bytes produced with imatrix data
    /// match bytes produced with `None` for legacy targets.
    #[test]
    fn dispatch_legacy_ignores_imatrix() {
        let row: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 16.0).collect();

        let mut weights_map = HashMap::new();
        weights_map.insert("blk.0.x.weight".to_string(), vec![1.0_f32; 32]);
        let imatrix = CalibrationData::Imatrix(weights_map);

        for target in [
            KQuantTarget::Q4Legacy,
            KQuantTarget::Q5Legacy0,
            KQuantTarget::Q5Legacy1,
            KQuantTarget::Q8Legacy,
        ] {
            let bytes_imatrix =
                quantize_row_to_bytes(&row, target, &imatrix, "blk.0.x.weight").unwrap();
            let bytes_none =
                quantize_row_to_bytes(&row, target, &CalibrationData::None, "blk.0.x.weight")
                    .unwrap();
            assert_eq!(
                bytes_imatrix, bytes_none,
                "legacy target {target:?} should silently ignore imatrix data"
            );
        }
    }

    /// Legacy targets reject misaligned (not multiple-of-32) input
    /// with the underlying QLegacy error.
    #[test]
    fn dispatch_legacy_misaligned() {
        let bad = vec![0.0_f32; 33];
        let err = quantize_row_to_bytes(
            &bad,
            KQuantTarget::Q8Legacy,
            &CalibrationData::None,
            "blk.0.x.weight",
        )
        .unwrap_err();
        match err {
            KQuantCodecError::QLegacy(QLegacyError::NotBlockAligned { actual, qk }) => {
                assert_eq!(actual, 33);
                assert_eq!(qk, 32);
            }
            _ => panic!("expected QLegacy(NotBlockAligned)"),
        }
    }

    /// DWQ calibration is rejected for legacy targets too.
    #[test]
    fn dispatch_dwq_rejected_for_legacy() {
        let row = vec![0.5_f32; 32];
        let mut dwq_map = HashMap::new();
        dwq_map.insert("blk.0.x.weight".to_string(), vec![0.7_f32]);
        let calib = CalibrationData::Dwq(dwq_map);

        for target in [
            KQuantTarget::Q4Legacy,
            KQuantTarget::Q5Legacy0,
            KQuantTarget::Q5Legacy1,
            KQuantTarget::Q8Legacy,
        ] {
            let err = quantize_row_to_bytes(&row, target, &calib, "blk.0.x.weight").unwrap_err();
            match err {
                KQuantCodecError::DwqAtRowQuantize { .. } => {}
                _ => panic!("expected DwqAtRowQuantize for {target:?}"),
            }
        }
    }

    /// End-to-end round-trip via codec for each legacy format.
    #[test]
    fn dispatch_legacy_round_trip() {
        use crate::quantize::q_legacy::{
            dequantize_row_q4_0_bytes, dequantize_row_q5_0_bytes, dequantize_row_q5_1_bytes,
            dequantize_row_q8_0_bytes,
        };

        let row: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();

        // Pairs of (target, RMSE bound).
        for (target, rmse_bound) in [
            (KQuantTarget::Q8Legacy, 0.01),
            (KQuantTarget::Q5Legacy0, 0.04),
            (KQuantTarget::Q5Legacy1, 0.04),
            (KQuantTarget::Q4Legacy, 0.07),
        ] {
            let bytes = quantize_row_to_bytes(
                &row,
                target,
                &CalibrationData::None,
                "blk.0.x.weight",
            )
            .unwrap();
            let mut decoded = vec![0.0_f32; row.len()];
            match target {
                KQuantTarget::Q8Legacy => {
                    dequantize_row_q8_0_bytes(&bytes, &mut decoded).unwrap();
                }
                KQuantTarget::Q5Legacy0 => {
                    dequantize_row_q5_0_bytes(&bytes, &mut decoded).unwrap();
                }
                KQuantTarget::Q5Legacy1 => {
                    dequantize_row_q5_1_bytes(&bytes, &mut decoded).unwrap();
                }
                KQuantTarget::Q4Legacy => {
                    dequantize_row_q4_0_bytes(&bytes, &mut decoded).unwrap();
                }
                _ => unreachable!(),
            }

            let mut sse = 0.0_f64;
            for (a, b) in row.iter().zip(decoded.iter()) {
                let d = (*a as f64) - (*b as f64);
                sse += d * d;
            }
            let rmse = (sse / row.len() as f64).sqrt();
            assert!(
                rmse < rmse_bound,
                "round-trip via codec for {target:?}: RMSE {rmse} > {rmse_bound}"
            );
        }
    }
}
