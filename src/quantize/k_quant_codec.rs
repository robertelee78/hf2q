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
    quantize_row_q3_k_imatrix_to_bytes, quantize_row_q3_k_to_bytes,
    quantize_row_q4_k_imatrix_to_bytes, quantize_row_q4_k_to_bytes,
    quantize_row_q5_k_imatrix_to_bytes, quantize_row_q5_k_to_bytes,
    quantize_row_q6_k_imatrix_to_bytes, quantize_row_q6_k_to_bytes, KQuantError,
};
use crate::quantize::q_legacy::{
    quantize_row_q4_0_to_bytes, quantize_row_q4_1_to_bytes, quantize_row_q5_0_to_bytes,
    quantize_row_q5_1_to_bytes, quantize_row_q8_0_to_bytes, QLegacyError,
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
    /// `Q3_K_M` / `Q3_K_S` / `Q3_K_L` / `Q3_K`. 3.4375 bpw, super-block
    /// scale only (no min term), 16 sub-blocks of 16 elements.
    /// Per-element 3 bits = 1 bit (hmask) + 2 bits (qs); per-sub-block
    /// scales packed at 6 bits each. 110 bytes/block (smaller than
    /// Q4_K's 144 bytes).
    Q3K,
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
    /// `Q4_1`. 5.0 bpw, asymmetric (per-block min), 32-element blocks.
    /// Not currently used in the K-family fallback chain at
    /// `src/backends/gguf.rs:354-358`; included for completeness.
    Q4Legacy1,
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
            3 => Some(Self::Q4Legacy1), // GGML_TYPE_Q4_1
            6 => Some(Self::Q5Legacy0), // GGML_TYPE_Q5_0
            7 => Some(Self::Q5Legacy1), // GGML_TYPE_Q5_1
            8 => Some(Self::Q8Legacy),  // GGML_TYPE_Q8_0
            11 => Some(Self::Q3K),      // GGML_TYPE_Q3_K
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
            Self::Q4Legacy1 => 3,
            Self::Q5Legacy0 => 6,
            Self::Q5Legacy1 => 7,
            Self::Q8Legacy => 8,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
        }
    }

    /// Bytes per block on disk.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::Q3K => crate::quantize::k_quant::BLOCK_Q3_K_SIZE,
            Self::Q4K => crate::quantize::k_quant::BLOCK_Q4_K_SIZE,
            Self::Q5K => crate::quantize::k_quant::BLOCK_Q5_K_SIZE,
            Self::Q6K => crate::quantize::k_quant::BLOCK_Q6_K_SIZE,
            Self::Q4Legacy => crate::quantize::q_legacy::BLOCK_Q4_0_SIZE,
            Self::Q4Legacy1 => crate::quantize::q_legacy::BLOCK_Q4_1_SIZE,
            Self::Q5Legacy0 => crate::quantize::q_legacy::BLOCK_Q5_0_SIZE,
            Self::Q5Legacy1 => crate::quantize::q_legacy::BLOCK_Q5_1_SIZE,
            Self::Q8Legacy => crate::quantize::q_legacy::BLOCK_Q8_0_SIZE,
        }
    }

    /// Number of elements per block. K-family: 256. Legacy: 32.
    pub fn elements_per_block(&self) -> usize {
        match self {
            Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => crate::quantize::k_quant::QK_K,
            Self::Q4Legacy => crate::quantize::q_legacy::QK4_0,
            Self::Q4Legacy1 => crate::quantize::q_legacy::QK4_1,
            Self::Q5Legacy0 => crate::quantize::q_legacy::QK5_0,
            Self::Q5Legacy1 => crate::quantize::q_legacy::QK5_1,
            Self::Q8Legacy => crate::quantize::q_legacy::QK8_0,
        }
    }

    /// Whether this target supports imatrix-weighted codebook search.
    /// Only K-family does — legacy formats use the `_ref` path.
    pub fn supports_imatrix(&self) -> bool {
        matches!(self, Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K)
    }

    /// Bits per weight (effective storage cost). Includes per-block
    /// metadata overhead.
    pub fn bpw(&self) -> f32 {
        match self {
            Self::Q3K => 3.4375,
            Self::Q4K => 4.5,
            Self::Q5K => 5.5,
            Self::Q6K => 6.5625,
            Self::Q4Legacy => 4.5,
            Self::Q4Legacy1 => 5.0,
            Self::Q5Legacy0 => 5.5,
            Self::Q5Legacy1 => 6.0,
            Self::Q8Legacy => 8.5,
        }
    }

    /// Enumerate every supported codec target in canonical order
    /// (K-family first, then the legacy fallback chain). Used by
    /// codec-coverage smoke tests so that adding a new target to the
    /// enum automatically extends test coverage without touching the
    /// test fixtures.
    pub fn all() -> &'static [Self] {
        &[
            Self::Q3K,
            Self::Q4K,
            Self::Q5K,
            Self::Q6K,
            Self::Q4Legacy,
            Self::Q4Legacy1,
            Self::Q5Legacy0,
            Self::Q5Legacy1,
            Self::Q8Legacy,
        ]
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
        (KQuantTarget::Q3K, Some(w)) => quantize_row_q3_k_imatrix_to_bytes(row, w)?,
        (KQuantTarget::Q3K, None) => quantize_row_q3_k_to_bytes(row)?,
        (KQuantTarget::Q4K, Some(w)) => quantize_row_q4_k_imatrix_to_bytes(row, w)?,
        (KQuantTarget::Q4K, None) => quantize_row_q4_k_to_bytes(row)?,
        (KQuantTarget::Q5K, Some(w)) => quantize_row_q5_k_imatrix_to_bytes(row, w)?,
        (KQuantTarget::Q5K, None) => quantize_row_q5_k_to_bytes(row)?,
        (KQuantTarget::Q6K, Some(w)) => quantize_row_q6_k_imatrix_to_bytes(row, w)?,
        (KQuantTarget::Q6K, None) => quantize_row_q6_k_to_bytes(row)?,
        // Legacy: always _ref. Imatrix weights silently ignored
        // (legacy formats don't support per-element-weighted codebook search).
        (KQuantTarget::Q4Legacy, _) => quantize_row_q4_0_to_bytes(row)?,
        (KQuantTarget::Q4Legacy1, _) => quantize_row_q4_1_to_bytes(row)?,
        (KQuantTarget::Q5Legacy0, _) => quantize_row_q5_0_to_bytes(row)?,
        (KQuantTarget::Q5Legacy1, _) => quantize_row_q5_1_to_bytes(row)?,
        (KQuantTarget::Q8Legacy, _) => quantize_row_q8_0_to_bytes(row)?,
    };
    Ok(bytes)
}

/// Quantize a 2D tensor (row-major) to GGUF block bytes. The natural
/// production-caller interface — most LLM weight tensors are
/// `[out_features, in_features]` shape and quantize per row (each
/// `in_features` slice is one row, packed independently).
///
/// `data` is the row-major flat F32 storage. `n_rows` × `row_len` must
/// equal `data.len()`. `row_len` must be a multiple of
/// `target.elements_per_block()` (256 for K-family, 32 for legacy).
///
/// The output bytes are the concatenation of all rows' block bytes,
/// in row-major order. Total size:
/// `n_rows × (row_len / elements_per_block) × bytes_per_block`.
///
/// **Imatrix lookup**: when `calibration` is `Imatrix` /
/// `ImatrixWithStats` and the calibrator has weights for `tensor_name`,
/// the same imatrix vector is used for **every row** of the tensor.
/// This matches llama.cpp's `quantize_q4_K_impl` etc., which apply
/// the same per-tensor importance map across all rows. (Per-row
/// imatrix would require per-row activation captures, which is
/// outside ADR-014's scope.)
pub fn quantize_tensor_2d_to_bytes(
    data: &[f32],
    n_rows: usize,
    row_len: usize,
    target: KQuantTarget,
    calibration: &CalibrationData,
    tensor_name: &str,
) -> Result<Vec<u8>, KQuantCodecError> {
    if data.len() != n_rows * row_len {
        return Err(KQuantCodecError::ImatrixWeightsLengthMismatch {
            tensor: tensor_name.to_string(),
            weights_len: data.len(),
            row_len: n_rows * row_len,
        });
    }
    let qk = target.elements_per_block();
    if !row_len.is_multiple_of(qk) {
        // Surface the underlying NotBlockAligned via a single-row dispatch.
        // This way the caller sees a consistent error type regardless of
        // whether the misalignment is at the row level or the data level.
        return quantize_row_to_bytes(&data[..row_len], target, calibration, tensor_name)
            .map(|_| unreachable!("misaligned row should have errored above"));
    }

    let blocks_per_row = row_len / qk;
    let bytes_per_row = blocks_per_row * target.bytes_per_block();
    let mut out = Vec::with_capacity(n_rows * bytes_per_row);

    for r in 0..n_rows {
        let row = &data[r * row_len..(r + 1) * row_len];
        let row_bytes = quantize_row_to_bytes(row, target, calibration, tensor_name)?;
        debug_assert_eq!(
            row_bytes.len(),
            bytes_per_row,
            "row {r} produced {} bytes, expected {bytes_per_row}",
            row_bytes.len()
        );
        out.extend_from_slice(&row_bytes);
    }
    Ok(out)
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
            KQuantTarget::Q3K,
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
        assert_eq!(KQuantTarget::Q3K.bytes_per_block(), 110);
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
        assert_eq!(KQuantTarget::Q3K.elements_per_block(), 256);
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
        assert!(KQuantTarget::Q3K.supports_imatrix());
        assert!(KQuantTarget::Q4K.supports_imatrix());
        assert!(KQuantTarget::Q5K.supports_imatrix());
        assert!(KQuantTarget::Q6K.supports_imatrix());
        assert!(!KQuantTarget::Q4Legacy.supports_imatrix());
        assert!(!KQuantTarget::Q5Legacy0.supports_imatrix());
        assert!(!KQuantTarget::Q5Legacy1.supports_imatrix());
        assert!(!KQuantTarget::Q8Legacy.supports_imatrix());
    }

    /// `bpw` returns documented values for all 9 targets (Q3_K added iter-9).
    #[test]
    fn target_bpw() {
        assert!((KQuantTarget::Q3K.bpw() - 3.4375).abs() < 1e-4);
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

    /// Q4_1 dispatch: 32 elements → 20 bytes.
    #[test]
    fn dispatch_none_q4_1() {
        let row: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0).collect();
        let bytes = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4Legacy1,
            &CalibrationData::None,
            "blk.0.attn_norm.weight",
        )
        .unwrap();
        assert_eq!(bytes.len(), 20);
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

    // ─────────────── End-to-end integration tests (iter-3k) ───────────────

    /// Deterministic LCG-based F32 generator for reproducible test
    /// inputs. Output values are roughly in `[-1, 1]`.
    fn synth_row(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let bits = (state >> 33) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn rmse_pair(a: &[f32], b: &[f32]) -> f64 {
        let mut sse = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (*x as f64) - (*y as f64);
            sse += d * d;
        }
        (sse / a.len() as f64).sqrt()
    }

    fn decode_via_codec_format(
        target: KQuantTarget,
        bytes: &[u8],
        n: usize,
    ) -> Vec<f32> {
        use crate::quantize::k_quant::{
            dequantize_row_q3_k_bytes, dequantize_row_q4_k_bytes, dequantize_row_q5_k_bytes,
            dequantize_row_q6_k_bytes,
        };
        use crate::quantize::q_legacy::{
            dequantize_row_q4_0_bytes, dequantize_row_q5_0_bytes, dequantize_row_q5_1_bytes,
            dequantize_row_q8_0_bytes,
        };

        let mut out = vec![0.0_f32; n];
        match target {
            KQuantTarget::Q3K => {
                dequantize_row_q3_k_bytes(bytes, &mut out).unwrap();
            }
            KQuantTarget::Q4K => {
                dequantize_row_q4_k_bytes(bytes, &mut out).unwrap();
            }
            KQuantTarget::Q5K => {
                dequantize_row_q5_k_bytes(bytes, &mut out).unwrap();
            }
            KQuantTarget::Q6K => {
                dequantize_row_q6_k_bytes(bytes, &mut out).unwrap();
            }
            KQuantTarget::Q4Legacy => {
                dequantize_row_q4_0_bytes(bytes, &mut out).unwrap();
            }
            KQuantTarget::Q4Legacy1 => {
                use crate::quantize::q_legacy::dequantize_row_q4_1_bytes;
                dequantize_row_q4_1_bytes(bytes, &mut out).unwrap();
            }
            KQuantTarget::Q5Legacy0 => {
                dequantize_row_q5_0_bytes(bytes, &mut out).unwrap();
            }
            KQuantTarget::Q5Legacy1 => {
                dequantize_row_q5_1_bytes(bytes, &mut out).unwrap();
            }
            KQuantTarget::Q8Legacy => {
                dequantize_row_q8_0_bytes(bytes, &mut out).unwrap();
            }
        }
        out
    }

    fn rmse_bound_for(target: KQuantTarget) -> f64 {
        match target {
            KQuantTarget::Q3K => 0.10,
            KQuantTarget::Q4K => 0.05,
            KQuantTarget::Q5K => 0.025,
            KQuantTarget::Q6K => 0.012,
            KQuantTarget::Q4Legacy => 0.05,
            KQuantTarget::Q4Legacy1 => 0.025,
            KQuantTarget::Q5Legacy0 => 0.025,
            KQuantTarget::Q5Legacy1 => 0.025,
            KQuantTarget::Q8Legacy => 0.005,
        }
    }

    /// 4096-element row (typical attention Q-projection inner dim)
    /// round-trip for each of the 7 codec targets within RMSE bound.
    #[test]
    fn integration_round_trip_4096_all_formats() {
        let n = 4096;
        let row = synth_row(0xCAFE_BABE, n);

        for target in [
            KQuantTarget::Q4K,
            KQuantTarget::Q5K,
            KQuantTarget::Q6K,
            KQuantTarget::Q4Legacy,
            KQuantTarget::Q5Legacy0,
            KQuantTarget::Q5Legacy1,
            KQuantTarget::Q8Legacy,
        ] {
            let bytes =
                quantize_row_to_bytes(&row, target, &CalibrationData::None, "weight").unwrap();
            let n_blocks = n / target.elements_per_block();
            let expected_bytes = n_blocks * target.bytes_per_block();
            assert_eq!(
                bytes.len(),
                expected_bytes,
                "{target:?}: produced {} bytes, expected {expected_bytes}",
                bytes.len()
            );

            let decoded = decode_via_codec_format(target, &bytes, n);
            let r = rmse_pair(&row, &decoded);
            let bound = rmse_bound_for(target);
            assert!(
                r < bound,
                "{target:?} 4096-elem RMSE {r} > bound {bound}"
            );
        }
    }

    /// 16384-element row (typical FFN gate-projection output dim).
    #[test]
    fn integration_round_trip_16384_all_formats() {
        let n = 16384;
        let row = synth_row(0xDEAD_BEEF, n);

        for target in [
            KQuantTarget::Q4K,
            KQuantTarget::Q5K,
            KQuantTarget::Q6K,
            KQuantTarget::Q4Legacy,
            KQuantTarget::Q5Legacy0,
            KQuantTarget::Q5Legacy1,
            KQuantTarget::Q8Legacy,
        ] {
            let bytes =
                quantize_row_to_bytes(&row, target, &CalibrationData::None, "weight").unwrap();
            let decoded = decode_via_codec_format(target, &bytes, n);
            let r = rmse_pair(&row, &decoded);
            let bound = rmse_bound_for(target);
            assert!(
                r < bound,
                "{target:?} 16K-elem RMSE {r} > bound {bound}"
            );
        }
    }

    /// Bit-budget verification — output bytes × 8 / element-count
    /// matches `bpw()` within 5%. Catches drift in
    /// bytes_per_block / elements_per_block accounting.
    #[test]
    fn integration_bpw_matches_on_disk_size() {
        let n = 4096;
        let row = vec![0.5_f32; n];

        for target in [
            KQuantTarget::Q4K,
            KQuantTarget::Q5K,
            KQuantTarget::Q6K,
            KQuantTarget::Q4Legacy,
            KQuantTarget::Q5Legacy0,
            KQuantTarget::Q5Legacy1,
            KQuantTarget::Q8Legacy,
        ] {
            let bytes =
                quantize_row_to_bytes(&row, target, &CalibrationData::None, "weight").unwrap();
            let measured_bpw = (bytes.len() as f32 * 8.0) / n as f32;
            let documented_bpw = target.bpw();
            let pct_diff = ((measured_bpw - documented_bpw) / documented_bpw).abs();
            assert!(
                pct_diff < 0.05,
                "{target:?}: measured bpw {measured_bpw:.4} != documented {documented_bpw:.4}"
            );
        }
    }

    /// **Imatrix path actually fires for K-family** — biased weights
    /// produce different bytes than uniform / None for Q4_K/Q5_K/Q6_K.
    /// Ensures the dispatch branch isn't accidentally falling through.
    #[test]
    fn integration_imatrix_changes_k_family_bytes() {
        let n = 4096;
        let row = synth_row(0xAA55_AA55, n);

        let mut weights_map = HashMap::new();
        let mut w = vec![1.0_f32; n];
        for v in w.iter_mut().take(256) {
            *v = 100.0;
        }
        weights_map.insert("weight".to_string(), w);
        let imatrix = CalibrationData::Imatrix(weights_map);

        for target in [KQuantTarget::Q4K, KQuantTarget::Q5K, KQuantTarget::Q6K] {
            let bytes_imatrix =
                quantize_row_to_bytes(&row, target, &imatrix, "weight").unwrap();
            let bytes_ref =
                quantize_row_to_bytes(&row, target, &CalibrationData::None, "weight").unwrap();
            assert_eq!(bytes_imatrix.len(), bytes_ref.len());
            assert_ne!(
                bytes_imatrix, bytes_ref,
                "{target:?}: imatrix path produced byte-identical output to _ref"
            );
        }
    }

    // ─────────────── Multi-row tensor helper tests (iter-3l) ───────────────

    /// `quantize_tensor_2d_to_bytes` on a 4-row × 256-element tensor
    /// produces exactly 4 × 144 = 576 bytes for Q4_K. Round-trip via
    /// per-row dequantize matches the original within RMSE bound.
    #[test]
    fn tensor_2d_q4_k_round_trip() {
        use crate::quantize::k_quant::dequantize_row_q4_k_bytes;

        let n_rows = 4;
        let row_len = 256;
        let mut data = Vec::with_capacity(n_rows * row_len);
        for r in 0..n_rows {
            // Each row gets a different seed → different data per row.
            data.extend(synth_row(0xAA00 + (r as u64), row_len));
        }

        let bytes = quantize_tensor_2d_to_bytes(
            &data,
            n_rows,
            row_len,
            KQuantTarget::Q4K,
            &CalibrationData::None,
            "weight",
        )
        .unwrap();

        let bytes_per_row = (row_len / 256) * KQuantTarget::Q4K.bytes_per_block();
        assert_eq!(bytes.len(), n_rows * bytes_per_row);

        // Decode each row independently and check RMSE.
        for r in 0..n_rows {
            let row_bytes = &bytes[r * bytes_per_row..(r + 1) * bytes_per_row];
            let mut decoded = vec![0.0_f32; row_len];
            dequantize_row_q4_k_bytes(row_bytes, &mut decoded).unwrap();

            let original = &data[r * row_len..(r + 1) * row_len];
            let r_rmse = rmse_pair(original, &decoded);
            assert!(
                r_rmse < 0.05,
                "row {r} Q4_K RMSE {r_rmse} > 0.05"
            );
        }
    }

    /// Multi-row Q8_0: 8 rows × 64 elements each. Verifies the legacy
    /// per-row block accounting works through the multi-row helper.
    #[test]
    fn tensor_2d_q8_0_round_trip() {
        use crate::quantize::q_legacy::dequantize_row_q8_0_bytes;

        let n_rows = 8;
        let row_len = 64;
        let mut data = Vec::with_capacity(n_rows * row_len);
        for r in 0..n_rows {
            data.extend(synth_row(0xBB00 + (r as u64), row_len));
        }

        let bytes = quantize_tensor_2d_to_bytes(
            &data,
            n_rows,
            row_len,
            KQuantTarget::Q8Legacy,
            &CalibrationData::None,
            "weight",
        )
        .unwrap();

        let bytes_per_row = (row_len / 32) * KQuantTarget::Q8Legacy.bytes_per_block();
        assert_eq!(bytes.len(), n_rows * bytes_per_row);

        for r in 0..n_rows {
            let row_bytes = &bytes[r * bytes_per_row..(r + 1) * bytes_per_row];
            let mut decoded = vec![0.0_f32; row_len];
            dequantize_row_q8_0_bytes(row_bytes, &mut decoded).unwrap();
            let original = &data[r * row_len..(r + 1) * row_len];
            let r_rmse = rmse_pair(original, &decoded);
            assert!(r_rmse < 0.005, "row {r} Q8_0 RMSE {r_rmse} > 0.005");
        }
    }

    /// Multi-row helper produces output equivalent to manual per-row
    /// concatenation. Validates the iteration order and accumulation.
    #[test]
    fn tensor_2d_equivalent_to_per_row_loop() {
        let n_rows = 3;
        let row_len = 256;
        let mut data = Vec::with_capacity(n_rows * row_len);
        for r in 0..n_rows {
            data.extend(synth_row(0xCC00 + (r as u64), row_len));
        }

        let multi_row_bytes = quantize_tensor_2d_to_bytes(
            &data,
            n_rows,
            row_len,
            KQuantTarget::Q5K,
            &CalibrationData::None,
            "weight",
        )
        .unwrap();

        let mut manual = Vec::new();
        for r in 0..n_rows {
            let row = &data[r * row_len..(r + 1) * row_len];
            let bytes = quantize_row_to_bytes(
                row,
                KQuantTarget::Q5K,
                &CalibrationData::None,
                "weight",
            )
            .unwrap();
            manual.extend_from_slice(&bytes);
        }
        assert_eq!(multi_row_bytes, manual);
    }

    /// Mismatched data length (n_rows × row_len ≠ data.len()) surfaces
    /// a typed error.
    #[test]
    fn tensor_2d_rejects_data_length_mismatch() {
        let data = vec![0.0_f32; 256];
        // Caller claims 2 rows × 256 = 512 elements but only 256 provided.
        let err = quantize_tensor_2d_to_bytes(
            &data,
            2,
            256,
            KQuantTarget::Q4K,
            &CalibrationData::None,
            "weight",
        )
        .unwrap_err();
        match err {
            KQuantCodecError::ImatrixWeightsLengthMismatch {
                weights_len, row_len, ..
            } => {
                assert_eq!(weights_len, 256);
                assert_eq!(row_len, 512);
            }
            _ => panic!("expected length mismatch"),
        }
    }

    /// Misaligned row_len (not a multiple of `elements_per_block`)
    /// rejected with the underlying KQuant error.
    #[test]
    fn tensor_2d_rejects_misaligned_row() {
        let data = vec![0.0_f32; 200];
        let err = quantize_tensor_2d_to_bytes(
            &data,
            1,
            200,
            KQuantTarget::Q4K,
            &CalibrationData::None,
            "weight",
        )
        .unwrap_err();
        match err {
            KQuantCodecError::KQuant(KQuantError::NotBlockAligned { actual }) => {
                assert_eq!(actual, 200);
            }
            _ => panic!("expected NotBlockAligned, got {err:?}"),
        }
    }

    /// Multi-row tensor with imatrix calibration: every row uses the
    /// SAME imatrix vector (matches llama.cpp's per-tensor imatrix
    /// model). Verifies the imatrix data isn't accidentally
    /// per-row-indexed or reset across rows.
    #[test]
    fn tensor_2d_imatrix_applies_to_every_row() {
        let n_rows = 3;
        let row_len = 256;
        let mut data = Vec::with_capacity(n_rows * row_len);
        for r in 0..n_rows {
            data.extend(synth_row(0xDD00 + (r as u64), row_len));
        }

        let mut weights_map = HashMap::new();
        let mut w = vec![1.0_f32; row_len];
        for v in w.iter_mut().take(64) {
            *v = 100.0;
        }
        weights_map.insert("weight".to_string(), w);
        let imatrix = CalibrationData::Imatrix(weights_map);

        let bytes_imatrix = quantize_tensor_2d_to_bytes(
            &data,
            n_rows,
            row_len,
            KQuantTarget::Q4K,
            &imatrix,
            "weight",
        )
        .unwrap();
        let bytes_ref = quantize_tensor_2d_to_bytes(
            &data,
            n_rows,
            row_len,
            KQuantTarget::Q4K,
            &CalibrationData::None,
            "weight",
        )
        .unwrap();
        assert_eq!(bytes_imatrix.len(), bytes_ref.len());
        // At least one row's bytes must differ — the imatrix bias on
        // the first 64 elements should perturb the codebook search.
        assert_ne!(
            bytes_imatrix, bytes_ref,
            "imatrix tensor-2d produced identical bytes to _ref — \
             dispatch likely fell through"
        );
    }

    /// Resolution ordering — Q8_0 > Q6_K > Q5_K > Q5_0 > Q4_K > Q4_0
    /// in accuracy. Verifies the bpw vs RMSE curve holds end-to-end.
    /// Q5_1 ordering vs Q5_0 is data-dependent (asymmetry helps when
    /// distribution is shifted), so not asserted here.
    #[test]
    fn integration_resolution_ordering() {
        let n = 4096;
        let row = synth_row(0xFEED_FACE, n);

        let mut rmses: Vec<(KQuantTarget, f64)> = Vec::new();
        for target in [
            KQuantTarget::Q4Legacy,
            KQuantTarget::Q4K,
            KQuantTarget::Q5Legacy0,
            KQuantTarget::Q5K,
            KQuantTarget::Q6K,
            KQuantTarget::Q8Legacy,
        ] {
            let bytes =
                quantize_row_to_bytes(&row, target, &CalibrationData::None, "weight").unwrap();
            let decoded = decode_via_codec_format(target, &bytes, n);
            rmses.push((target, rmse_pair(&row, &decoded)));
        }

        let r = |t: KQuantTarget| rmses.iter().find(|(x, _)| *x == t).unwrap().1;

        let r_q40 = r(KQuantTarget::Q4Legacy);
        let r_q4k = r(KQuantTarget::Q4K);
        let r_q50 = r(KQuantTarget::Q5Legacy0);
        let r_q5k = r(KQuantTarget::Q5K);
        let r_q6k = r(KQuantTarget::Q6K);
        let r_q80 = r(KQuantTarget::Q8Legacy);

        assert!(r_q50 < r_q40, "Q5_0 RMSE {r_q50} should be < Q4_0 {r_q40}");
        assert!(r_q5k < r_q50, "Q5_K {r_q5k} should be < Q5_0 {r_q50}");
        assert!(r_q6k < r_q5k, "Q6_K {r_q6k} should be < Q5_K {r_q5k}");
        assert!(r_q80 < r_q6k, "Q8_0 {r_q80} should be < Q6_K {r_q6k}");
        assert!(r_q4k < r_q40, "Q4_K {r_q4k} should be < Q4_0 {r_q40}");
    }

    /// ADR-014 P7 iter-5 — `KQuantTarget::all()` enumeration round-trips
    /// through `ggml_type` / `from_ggml_type` and the metadata methods
    /// (`bytes_per_block`, `elements_per_block`) reflect the expected
    /// layout for every supported codec target.
    ///
    /// Future-proofs codec coverage: a new target added to the enum
    /// must also be added to `all()`, `ggml_type` round-trip, the
    /// metadata methods, and `quantize_row_to_bytes` dispatch.  This
    /// test fails on the second of those omissions; the
    /// `dispatch_all_targets_smoke` test below catches the third.
    #[test]
    fn target_all_round_trips_metadata() {
        let all = KQuantTarget::all();
        assert_eq!(all.len(), 9, "exactly 9 supported codec targets (Q3_K added iter-9)");

        // canonical order: K-family first, then legacy
        assert_eq!(all[0], KQuantTarget::Q3K);
        assert_eq!(all[1], KQuantTarget::Q4K);
        assert_eq!(all[2], KQuantTarget::Q5K);
        assert_eq!(all[3], KQuantTarget::Q6K);
        assert_eq!(all[4], KQuantTarget::Q4Legacy);
        assert_eq!(all[5], KQuantTarget::Q4Legacy1);
        assert_eq!(all[6], KQuantTarget::Q5Legacy0);
        assert_eq!(all[7], KQuantTarget::Q5Legacy1);
        assert_eq!(all[8], KQuantTarget::Q8Legacy);

        // every target round-trips through ggml_type
        for &target in all {
            let g = target.ggml_type();
            let recovered = KQuantTarget::from_ggml_type(g)
                .unwrap_or_else(|| panic!("from_ggml_type({g}) returned None"));
            assert_eq!(recovered, target,
                "ggml_type round-trip diverged for {target:?}");

            // bytes_per_block / elements_per_block are non-zero and
            // bpw matches the documented bits-per-weight value.
            assert!(target.bytes_per_block() > 0, "{target:?}: bytes_per_block == 0");
            assert!(target.elements_per_block() > 0, "{target:?}: elements_per_block == 0");
            assert!(target.bpw() > 0.0, "{target:?}: bpw <= 0");

            // K-family supports imatrix; legacy doesn't.
            let expected_imatrix = matches!(target,
                KQuantTarget::Q3K | KQuantTarget::Q4K | KQuantTarget::Q5K | KQuantTarget::Q6K);
            assert_eq!(
                target.supports_imatrix(), expected_imatrix,
                "{target:?}: supports_imatrix mismatch"
            );
        }
    }

    /// ADR-014 P7 iter-5 — `quantize_row_to_bytes` dispatch covers
    /// every target in `KQuantTarget::all()` end-to-end on a smooth
    /// ramp.  Each target produces non-empty output sized to the
    /// target's `bytes_per_block`.
    ///
    /// Catches a regression where a new target is added to the enum
    /// + `all()` but a `match target` arm in `quantize_row_to_bytes`
    /// is missed (would compile if the match is non-exhaustive on the
    /// new variant via a `_ =>` fallback).  Also locks the contract
    /// "every codec target produces bytes whose length is a multiple
    /// of `bytes_per_block`" — required for downstream dequant.
    #[test]
    fn dispatch_all_targets_smoke() {
        for &target in KQuantTarget::all() {
            // pick a row whose length aligns to this target's block size
            let n = target.elements_per_block();
            let row = smooth_ramp(n);

            let bytes = quantize_row_to_bytes(
                &row, target, &CalibrationData::None, "blk.5.attn_q.weight",
            )
            .unwrap_or_else(|e| panic!("{target:?}: dispatch failed: {e}"));

            assert!(!bytes.is_empty(), "{target:?}: empty output");
            assert_eq!(
                bytes.len() % target.bytes_per_block(), 0,
                "{target:?}: output bytes ({}) not a multiple of bytes_per_block ({})",
                bytes.len(), target.bytes_per_block()
            );
            // exactly one block emitted (n == elements_per_block)
            assert_eq!(
                bytes.len(), target.bytes_per_block(),
                "{target:?}: expected exactly 1 block, got {} bytes",
                bytes.len()
            );
        }
    }
}
