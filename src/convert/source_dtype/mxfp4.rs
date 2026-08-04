//! DeepSeek-V4's official Hugging Face routed-expert source format.
//!
//! Each source byte stores two E2M1 values (low nibble first), while
//! one unsigned E8M0 exponent byte scales every 32 logical values.
//! Conversion expands this source representation to F32 in-process;
//! the selected GGUF quantizer then owns the destination encoding.

use std::fmt;

/// OCP MX E2M1 values in the exact nibble order used by the official
/// DeepSeek-V4 converter.
pub const E2M1_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mxfp4Error {
    InvalidShape {
        shape: Vec<usize>,
    },
    LogicalRowNotBlockAligned {
        logical_cols: usize,
    },
    PayloadLengthMismatch {
        expected: usize,
        got: usize,
    },
    ScaleShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    ScaleLengthMismatch {
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for Mxfp4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape { shape } => write!(
                f,
                "mxfp4/shape: expected non-empty 2-D [rows, packed_cols], got {shape:?}"
            ),
            Self::LogicalRowNotBlockAligned { logical_cols } => write!(
                f,
                "mxfp4/shape: logical row {logical_cols} is not a multiple of 32"
            ),
            Self::PayloadLengthMismatch { expected, got } => write!(
                f,
                "mxfp4/payload: expected {expected} packed bytes, got {got}"
            ),
            Self::ScaleShapeMismatch { expected, got } => {
                write!(f, "mxfp4/scale: expected shape {expected:?}, got {got:?}")
            }
            Self::ScaleLengthMismatch { expected, got } => {
                write!(f, "mxfp4/scale: expected {expected} E8M0 bytes, got {got}")
            }
        }
    }
}

impl std::error::Error for Mxfp4Error {}

/// Decode the official UE8M0/E8M0 scale convention: `2^(bits - 127)`.
/// This intentionally mirrors the model's converter, including 0xff
/// overflowing to positive infinity in F32.
#[inline]
pub fn decode_e8m0(bits: u8) -> f32 {
    (bits as f32 - 127.0).exp2()
}

/// Expand packed E2M1 routed-expert weights to row-major F32.
pub fn dequantize_e2m1(
    payload: &[u8],
    packed_shape: &[usize],
    scale_bits: &[u8],
    scale_shape: &[usize],
) -> Result<Vec<f32>, Mxfp4Error> {
    if packed_shape.len() != 2 || packed_shape[0] == 0 || packed_shape[1] == 0 {
        return Err(Mxfp4Error::InvalidShape {
            shape: packed_shape.to_vec(),
        });
    }
    let rows = packed_shape[0];
    let packed_cols = packed_shape[1];
    let logical_cols = packed_cols * 2;
    if logical_cols % 32 != 0 {
        return Err(Mxfp4Error::LogicalRowNotBlockAligned { logical_cols });
    }
    let expected_payload = rows * packed_cols;
    if payload.len() != expected_payload {
        return Err(Mxfp4Error::PayloadLengthMismatch {
            expected: expected_payload,
            got: payload.len(),
        });
    }
    let expected_scale_shape = vec![rows, logical_cols / 32];
    if scale_shape != expected_scale_shape {
        return Err(Mxfp4Error::ScaleShapeMismatch {
            expected: expected_scale_shape,
            got: scale_shape.to_vec(),
        });
    }
    let expected_scales = rows * (logical_cols / 32);
    if scale_bits.len() != expected_scales {
        return Err(Mxfp4Error::ScaleLengthMismatch {
            expected: expected_scales,
            got: scale_bits.len(),
        });
    }

    let blocks_per_row = logical_cols / 32;
    let mut out = Vec::with_capacity(rows * logical_cols);
    for row in 0..rows {
        for packed_col in 0..packed_cols {
            let byte = payload[row * packed_cols + packed_col];
            let logical_col = packed_col * 2;
            let scale = decode_e8m0(scale_bits[row * blocks_per_row + logical_col / 32]);
            out.push(E2M1_TABLE[(byte & 0x0f) as usize] * scale);
            out.push(E2M1_TABLE[(byte >> 4) as usize] * scale);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_matches_official_power_of_two_rule() {
        assert_eq!(decode_e8m0(127), 1.0);
        assert_eq!(decode_e8m0(128), 2.0);
        assert_eq!(decode_e8m0(126), 0.5);
        assert!(decode_e8m0(255).is_infinite());
    }

    #[test]
    fn low_nibble_precedes_high_nibble_and_scales_per_32() {
        let mut payload = vec![0x21; 32];
        payload[16] = 0xa9; // -0.5, -1.0 at start of block two
        let got = dequantize_e2m1(&payload, &[1, 32], &[127, 128], &[1, 2]).unwrap();
        assert_eq!(&got[..4], &[0.5, 1.0, 0.5, 1.0]);
        assert_eq!(&got[32..34], &[-1.0, -2.0]);
        assert_eq!(got.len(), 64);
    }

    #[test]
    fn rejects_malformed_group_shapes_and_lengths() {
        assert!(matches!(
            dequantize_e2m1(&[0; 15], &[1, 15], &[127], &[1, 1]),
            Err(Mxfp4Error::LogicalRowNotBlockAligned { .. })
        ));
        assert!(matches!(
            dequantize_e2m1(&[0; 16], &[1, 16], &[127], &[2, 1]),
            Err(Mxfp4Error::ScaleShapeMismatch { .. })
        ));
        assert!(matches!(
            dequantize_e2m1(&[0; 15], &[1, 16], &[127], &[1, 1]),
            Err(Mxfp4Error::PayloadLengthMismatch { .. })
        ));
    }
}
