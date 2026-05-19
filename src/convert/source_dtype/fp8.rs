//! `float8_e4m3fn` (1-bit sign + 4-bit exponent + 3-bit mantissa, no
//! Inf, single NaN encoding) dequantizer with HF block-wise scaling.
//!
//! Used by [`crate::convert::source_reader::HfModelSource::open`] when
//! `config.json::quantization_config.quant_method == "fp8"`. This is the
//! source format DeepSeek-V3, MiniMax-M2.7, and similar large-MoE
//! releases publish; the convert pipeline upcasts to F32 in-memory
//! before quantizing to the user-chosen GGUF type.
//!
//! Wire format (per the HF / PyTorch `float8_e4m3fn` spec — distinct
//! from `e4m3fnuz` and `e5m2`):
//!
//! ```text
//!  bit:  7    6  5  4  3    2  1  0
//!       [s]  [   exp   ]   [ mant ]
//! ```
//!
//! - `bias = 7` (exponent bias).
//! - Normal numbers: `(-1)^s * 2^(exp - 7) * 1.mant_frac`.
//! - Subnormals (exp == 0): `(-1)^s * 2^(1 - 7) * 0.mant_frac` =
//!   `(-1)^s * 2^-6 * mant/8`.
//! - **No Infinity**: the all-ones exponent slot (`exp == 0b1111`) is
//!   NOT used for Inf. Instead, the single NaN encoding is `0x7f`
//!   (`s=0, exp=1111, mant=111`) and `0xff` (`s=1, exp=1111, mant=111`).
//!   Every other `exp == 1111` pattern is a finite normal number with
//!   the largest representable magnitude (`+/- 448` for `mant=110`).
//! - Zero: `0x00` (+0) and `0x80` (-0) — both decode to 0.0.
//!
//! Block scaling (HF convention; mirrors `transformers/models/deepseek/`
//! and the per-tensor `weight_scale_inv` sidecar in MiniMax-M2.7
//! releases):
//!
//! - The on-disk FP8 tensor has PyTorch shape `[rows, cols]`.
//! - The sibling `<name>.weight_scale_inv` tensor is F32, shape
//!   `[ceil(rows / block_rows), ceil(cols / block_cols)]`.
//! - Each block of `block_rows × block_cols` FP8 elements shares ONE
//!   F32 inverse-scale.
//! - Dequant rule per element at `(r, c)`:
//!   `f32 = decode_e4m3fn(byte) * scale_inv[r / block_rows, c / block_cols]`.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: every length
//! mismatch is a typed error — no silent truncation, no NaN-fills, no
//! zero-padding. Callers either supply correctly-sized scales or get
//! [`Fp8Error`].

use std::fmt;

/// One element of `block_size` describes a `(rows, cols)` block shape
/// shared by one F32 inverse-scale entry. The HF schema names this
/// field `weight_block_size`; common values include `[128, 128]`
/// (DeepSeek-V3, MiniMax-M2.7) and `[1, 128]` (per-row variants).
pub type BlockSize = (usize, usize);

/// Errors raised by [`dequantize_fp8_block`]. Each variant carries
/// enough context to identify the offending tensor at the caller side
/// without re-deriving the math.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fp8Error {
    /// `block_size` has a zero component — division by zero would
    /// follow.
    InvalidBlockSize { block: BlockSize },
    /// `shape = [rows, cols]` doesn't have exactly 2 dims, or has a
    /// zero dim.
    InvalidShape { shape: Vec<usize> },
    /// `payload.len() != rows * cols`. FP8 is exactly 1 byte per
    /// element so the byte count must equal the element count.
    PayloadLengthMismatch {
        expected_bytes: usize,
        got_bytes: usize,
    },
    /// `scale_inv.len() != ceil_div(rows, block_rows) * ceil_div(cols, block_cols)`.
    ScaleLengthMismatch {
        expected: usize,
        got: usize,
        rows: usize,
        cols: usize,
        block: BlockSize,
    },
}

impl fmt::Display for Fp8Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fp8Error::InvalidBlockSize { block } => {
                write!(f, "fp8/block_size: invalid block {:?} (no zero dims)", block)
            }
            Fp8Error::InvalidShape { shape } => {
                write!(
                    f,
                    "fp8/shape: expected 2-D [rows, cols] with non-zero dims, got {:?}",
                    shape
                )
            }
            Fp8Error::PayloadLengthMismatch {
                expected_bytes,
                got_bytes,
            } => write!(
                f,
                "fp8/payload: expected {} bytes (1 per element), got {}",
                expected_bytes, got_bytes
            ),
            Fp8Error::ScaleLengthMismatch {
                expected,
                got,
                rows,
                cols,
                block,
            } => write!(
                f,
                "fp8/scale_inv: expected {} F32 entries for [{rows}, {cols}] with block {:?}, got {}",
                expected, block, got
            ),
        }
    }
}

impl std::error::Error for Fp8Error {}

/// Decode one `float8_e4m3fn` byte to F32.
///
/// Returns `f32::NAN` for the two NaN encodings (`0x7f`, `0xff`).
/// Returns signed-zero (`0.0` or `-0.0`) for `0x00` / `0x80`.
/// Otherwise: normal `(-1)^s * 2^(exp-7) * 1.mant_frac` or subnormal
/// `(-1)^s * 2^-6 * mant/8`.
#[inline]
pub fn decode_e4m3fn(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x01;
    let exp = ((byte >> 3) & 0x0f) as i32;
    let mant = (byte & 0x07) as u32;

    // NaN: exp==1111 AND mant==111. Spec is "single NaN encoding" but
    // both sign bits are valid (0x7f, 0xff); we collapse to one NaN.
    if exp == 0b1111 && mant == 0b111 {
        return f32::NAN;
    }

    let sign_mul: f32 = if sign == 0 { 1.0 } else { -1.0 };

    if exp == 0 {
        // Subnormal — including +0 / -0 when mant == 0.
        // value = sign * 2^(1 - bias) * (mant / 2^mant_bits)
        //       = sign * 2^-6 * (mant / 8)
        let mant_f = mant as f32 / 8.0;
        sign_mul * mant_f * (1.0_f32 / 64.0) // 2^-6
    } else {
        // Normal — value = sign * 2^(exp - 7) * (1 + mant / 8).
        let mant_f = 1.0 + (mant as f32 / 8.0);
        // 2^(exp - 7), exp ∈ [1, 15] → power ∈ [-6, 8]. Use bit-fiddle
        // to avoid `powi` overhead: build the F32 directly with exp
        // field `(exp - 7) + 127`. exp - 7 + 127 ∈ [121, 135] which is
        // a valid normal F32 exponent.
        let pow_bits: u32 = ((exp - 7 + 127) as u32) << 23;
        let pow = f32::from_bits(pow_bits);
        sign_mul * pow * mant_f
    }
}

#[inline]
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Dequantize a 2-D `float8_e4m3fn` tensor with HF block-wise inverse
/// scales to a flat row-major F32 buffer.
///
/// # Arguments
///
/// * `payload` — `rows * cols` FP8 bytes in row-major order.
/// * `scale_inv` — F32 inverse-scales, row-major over the block grid.
///   Length must equal `ceil_div(rows, block.0) * ceil_div(cols, block.1)`.
/// * `shape` — `[rows, cols]` in PyTorch order.
/// * `block` — `(block_rows, block_cols)` per HF `weight_block_size`.
///
/// # Returns
///
/// A `Vec<f32>` of length `rows * cols`, row-major. Each element is
/// `decode_e4m3fn(byte) * scale_inv[row / block_rows, col / block_cols]`.
///
/// # Errors
///
/// Returns [`Fp8Error`] for any length / shape mismatch; never panics
/// on user-controlled inputs. Per
/// [[feedback-no-loop-suppression-2026-05-17]]: no silent fallback.
pub fn dequantize_fp8_block(
    payload: &[u8],
    scale_inv: &[f32],
    shape: &[usize],
    block: BlockSize,
) -> Result<Vec<f32>, Fp8Error> {
    // ----- validate -------------------------------------------------------
    if block.0 == 0 || block.1 == 0 {
        return Err(Fp8Error::InvalidBlockSize { block });
    }
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(Fp8Error::InvalidShape {
            shape: shape.to_vec(),
        });
    }
    let rows = shape[0];
    let cols = shape[1];

    let expected_bytes = rows * cols;
    if payload.len() != expected_bytes {
        return Err(Fp8Error::PayloadLengthMismatch {
            expected_bytes,
            got_bytes: payload.len(),
        });
    }

    let scale_rows = ceil_div(rows, block.0);
    let scale_cols = ceil_div(cols, block.1);
    let expected_scales = scale_rows * scale_cols;
    if scale_inv.len() != expected_scales {
        return Err(Fp8Error::ScaleLengthMismatch {
            expected: expected_scales,
            got: scale_inv.len(),
            rows,
            cols,
            block,
        });
    }

    // ----- dequant --------------------------------------------------------
    // Row-major iteration; per row we cache the row-block index. Inner
    // loop walks columns and recomputes the col-block index only when
    // it crosses a block boundary (avoids `/` per element).
    let mut out: Vec<f32> = Vec::with_capacity(expected_bytes);
    for r in 0..rows {
        let row_block = r / block.0;
        let row_scale_base = row_block * scale_cols;
        let mut col_block = 0usize;
        let mut col_block_end = block.1.min(cols);
        let mut scale = scale_inv[row_scale_base];
        for c in 0..cols {
            if c >= col_block_end {
                col_block += 1;
                col_block_end = ((col_block + 1) * block.1).min(cols);
                scale = scale_inv[row_scale_base + col_block];
            }
            let byte = payload[r * cols + c];
            out.push(decode_e4m3fn(byte) * scale);
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============== decode_e4m3fn unit tests ==============

    /// `0x00` is +0.0; `0x80` is -0.0.
    #[test]
    fn decode_zero_signed() {
        assert_eq!(decode_e4m3fn(0x00), 0.0);
        let neg_zero = decode_e4m3fn(0x80);
        // Compare bit-pattern to distinguish -0.0 from +0.0.
        assert_eq!(neg_zero.to_bits(), (-0.0_f32).to_bits());
    }

    /// `0x38` = sign=0, exp=0111=7, mant=000 → normal 2^(7-7) * 1.0 = 1.0.
    /// `0x3c` = sign=0, exp=0111=7, mant=100 → 2^0 * (1 + 4/8) = 1.5.
    /// `0xb8` = sign=1, exp=0111=7, mant=000 → -1.0.
    #[test]
    fn decode_normal_numbers() {
        assert_eq!(decode_e4m3fn(0x38), 1.0);
        assert_eq!(decode_e4m3fn(0x3c), 1.5);
        assert_eq!(decode_e4m3fn(0xb8), -1.0);
        // Largest finite: 0x7e = sign=0, exp=1111, mant=110 → 2^8 * (1 + 6/8) = 448.0.
        assert_eq!(decode_e4m3fn(0x7e), 448.0);
        // Smallest positive normal: 0x08 = exp=0001, mant=000 → 2^-6 * 1.0 = 1/64.
        assert_eq!(decode_e4m3fn(0x08), 1.0 / 64.0);
        // 0x40 = sign=0, exp=1000, mant=000 → 2^(8-7) * 1.0 = 2.0.
        assert_eq!(decode_e4m3fn(0x40), 2.0);
    }

    /// Subnormals: `0x01` = sign=0, exp=0, mant=001 → 2^-6 * (1/8) = 1/512.
    /// `0x07` = exp=0, mant=111 → 2^-6 * (7/8) = 7/512.
    #[test]
    fn decode_subnormals() {
        assert!((decode_e4m3fn(0x01) - 1.0 / 512.0).abs() < 1e-7);
        assert!((decode_e4m3fn(0x07) - 7.0 / 512.0).abs() < 1e-7);
        // Negative subnormal.
        assert!((decode_e4m3fn(0x81) + 1.0 / 512.0).abs() < 1e-7);
    }

    /// `0x7f` and `0xff` are NaN per the e4m3fn spec ("single NaN encoding";
    /// both sign bits are valid in practice).
    #[test]
    fn decode_nan() {
        assert!(decode_e4m3fn(0x7f).is_nan());
        assert!(decode_e4m3fn(0xff).is_nan());
    }

    // ============== dequantize_fp8_block tests ==============

    /// Spec test 1 — all-zero FP8 input produces all-zero F32 output.
    #[test]
    fn dequant_zero() {
        let payload = vec![0u8; 8]; // 2x4 zeros
        let scale_inv = vec![1.0_f32]; // 1 block covers everything
        let out = dequantize_fp8_block(&payload, &scale_inv, &[2, 4], (2, 4)).unwrap();
        assert_eq!(out, vec![0.0_f32; 8]);
    }

    /// Spec test 2 — hand-crafted FP8 bit patterns vs known F32 values.
    /// Covers: +1.0, -0.0, NaN, +1.5, +2.0, +0.5, -1.0.
    #[test]
    fn dequant_normal_numbers() {
        // 0x38 = +1.0
        // 0x80 = -0.0
        // 0xff = NaN
        // 0x3c = +1.5
        // 0x40 = +2.0
        // 0x30 = sign=0, exp=0110=6, mant=000 → 2^-1 * 1.0 = 0.5
        // 0xb8 = -1.0
        // 0x00 = +0.0  (pad to round shape)
        let payload = vec![0x38, 0x80, 0xff, 0x3c, 0x40, 0x30, 0xb8, 0x00];
        let scale_inv = vec![1.0_f32];
        let out =
            dequantize_fp8_block(&payload, &scale_inv, &[1, 8], (1, 8)).expect("dequant");

        assert_eq!(out[0], 1.0);
        // -0.0 by bit pattern.
        assert_eq!(out[1].to_bits(), (-0.0_f32).to_bits());
        assert!(out[2].is_nan());
        assert_eq!(out[3], 1.5);
        assert_eq!(out[4], 2.0);
        assert_eq!(out[5], 0.5);
        assert_eq!(out[6], -1.0);
        assert_eq!(out[7], 0.0);
    }

    /// Spec test 3 — 2x2 FP8 with single 2x2 inverse-scale = 0.5 → all
    /// values halved.
    #[test]
    fn dequant_block_scale_applied() {
        // 2x2 of +1.0, +2.0, -1.0, +1.5.
        let payload = vec![0x38, 0x40, 0xb8, 0x3c];
        let scale_inv = vec![0.5_f32];
        let out = dequantize_fp8_block(&payload, &scale_inv, &[2, 2], (2, 2)).unwrap();
        assert_eq!(out, vec![0.5, 1.0, -0.5, 0.75]);
    }

    /// Block scaling with a 4x4 tensor and 2x2 blocks: 4 distinct
    /// inverse-scales (one per quadrant). Confirms the per-element
    /// scale lookup honors the (row, col) block grid.
    #[test]
    fn dequant_multi_block_scaling() {
        // 4x4 tensor, every byte = 0x38 (+1.0).
        let payload = vec![0x38; 16];
        // 2x2 grid of inverse-scales:
        //   top-left = 1, top-right = 2,
        //   bot-left = 3, bot-right = 4.
        let scale_inv = vec![1.0, 2.0, 3.0, 4.0];
        let out =
            dequantize_fp8_block(&payload, &scale_inv, &[4, 4], (2, 2)).expect("dequant");
        // Row 0, 1 use scale row 0 (1.0 for cols 0-1, 2.0 for cols 2-3).
        // Row 2, 3 use scale row 1 (3.0 for cols 0-1, 4.0 for cols 2-3).
        let want = [
            1.0, 1.0, 2.0, 2.0, // r0
            1.0, 1.0, 2.0, 2.0, // r1
            3.0, 3.0, 4.0, 4.0, // r2
            3.0, 3.0, 4.0, 4.0, // r3
        ];
        for (i, (got, w)) in out.iter().zip(want.iter()).enumerate() {
            assert_eq!(*got, *w, "element {i}: got {got}, want {w}");
        }
    }

    /// Block-grid ceil-div: 3x3 tensor with 2x2 block. The block grid
    /// is `ceil_div(3, 2)=2` × `ceil_div(3, 2)=2 = 4` blocks; the
    /// "edge" blocks cover the partial 1-element strip.
    #[test]
    fn dequant_block_grid_ceil_div() {
        let payload = vec![0x38; 9]; // 3x3 of +1.0
        let scale_inv = vec![10.0, 20.0, 30.0, 40.0];
        let out = dequantize_fp8_block(&payload, &scale_inv, &[3, 3], (2, 2)).unwrap();
        // (0,0)..(1,1) → 10; (0,2)..(1,2) → 20; (2,0)..(2,1) → 30; (2,2) → 40.
        let want = [
            10.0, 10.0, 20.0, // r0
            10.0, 10.0, 20.0, // r1
            30.0, 30.0, 40.0, // r2
        ];
        assert_eq!(out, want);
    }

    // ============== error-path tests ==============

    #[test]
    fn invalid_block_size_errors() {
        let payload = vec![0u8; 4];
        let scale_inv = vec![1.0];
        let err = dequantize_fp8_block(&payload, &scale_inv, &[2, 2], (0, 2))
            .expect_err("must error");
        matches!(err, Fp8Error::InvalidBlockSize { .. });
    }

    #[test]
    fn invalid_shape_errors() {
        let payload = vec![0u8; 4];
        let scale_inv = vec![1.0];
        let err = dequantize_fp8_block(&payload, &scale_inv, &[2, 2, 2], (2, 2))
            .expect_err("3-D rejected");
        matches!(err, Fp8Error::InvalidShape { .. });
        let err2 = dequantize_fp8_block(&payload, &scale_inv, &[0, 4], (2, 2))
            .expect_err("zero dim rejected");
        matches!(err2, Fp8Error::InvalidShape { .. });
    }

    #[test]
    fn payload_length_mismatch_errors() {
        let payload = vec![0u8; 3];
        let scale_inv = vec![1.0];
        let err = dequantize_fp8_block(&payload, &scale_inv, &[2, 2], (2, 2))
            .expect_err("must error");
        match err {
            Fp8Error::PayloadLengthMismatch {
                expected_bytes,
                got_bytes,
            } => {
                assert_eq!(expected_bytes, 4);
                assert_eq!(got_bytes, 3);
            }
            other => panic!("expected PayloadLengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn scale_length_mismatch_errors() {
        let payload = vec![0u8; 16]; // 4x4
        let scale_inv = vec![1.0, 2.0]; // need 4 entries for 2x2 block grid
        let err = dequantize_fp8_block(&payload, &scale_inv, &[4, 4], (2, 2))
            .expect_err("must error");
        match err {
            Fp8Error::ScaleLengthMismatch {
                expected, got, ..
            } => {
                assert_eq!(expected, 4);
                assert_eq!(got, 2);
            }
            other => panic!("expected ScaleLengthMismatch, got {other:?}"),
        }
    }
}
