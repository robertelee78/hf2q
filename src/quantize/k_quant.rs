//! K-quant codebooks — Q4_K_M / Q5_K_M / Q6_K (ADR-014 P7 Decision 11).
//!
//! Pure-Rust port of llama.cpp's k-quant codebook logic from
//! `/opt/llama.cpp/ggml/src/ggml-quants.c`. The k-quant family stores
//! 256 elements per super-block, organised into sub-blocks (8 × 32 for
//! Q4_K and Q5_K; 16 × 16 for Q6_K) with per-sub-block scales/mins
//! and a single per-super-block scale.
//!
//! ## ADR-014 Decision 11 round-2: NEON path on aarch64-apple-darwin
//!
//! The byte-identical gate against llama.cpp targets the **NEON code
//! path** (`quantize_row_q4_K` on `aarch64-apple-darwin`), not the
//! scalar reference. The NEON path is what M-series users actually
//! run; matching the scalar reference would not constitute peer
//! parity in production. The pure-Rust port either uses
//! `std::arch::aarch64` intrinsics or scalar code that replicates
//! NEON's reduction order (associativity-sensitive horizontal sums in
//! `make_qkx2_quants` must match).
//!
//! ## This iter — Q4_K dequantize only
//!
//! P7 iter-3a lands the **dequantize** direction for Q4_K only. The
//! quantize direction (which depends on `make_qkx2_quants` /
//! `make_qkx3_quants` codebook search routines + per-column-weighted
//! MSE for the imatrix variant) is the harder half and lands in
//! subsequent P7 iters. Q5_K and Q6_K land at the same time as Q4_K
//! quantize.
//!
//! ## Block layouts (verbatim from `ggml-common.h`)
//!
//! ```text
//! block_q4_K (144 bytes, 4.5 bpw):
//!   d:        f16            // super-block scale for quantized scales
//!   dmin:     f16            // super-block scale for quantized mins
//!   scales:   [u8; 12]       // 6-bit packed: 8 sub-block scales + 8 sub-block mins
//!   qs:       [u8; 128]      // 4-bit packed quants (256 elements / 2)
//!
//! block_q5_K (176 bytes, 5.5 bpw):
//!   d:        f16
//!   dmin:     f16
//!   scales:   [u8; 12]       // 6-bit packed
//!   qh:       [u8; 32]       // high bit (256 / 8)
//!   qs:       [u8; 128]      // low 4 bits
//!
//! block_q6_K (210 bytes, 6.5625 bpw):
//!   ql:       [u8; 128]      // lower 4 bits
//!   qh:       [u8; 64]       // upper 2 bits (256 / 4)
//!   scales:   [i8; 16]       // 8-bit signed (16 sub-blocks of 16 elements)
//!   d:        f16
//! ```
//!
//! ## Sovereignty
//!
//! Pure Rust. No `cc`/`cmake` link to libggml. The reference impl at
//! `/opt/llama.cpp/ggml/src/ggml-quants.c` is read-only; this port
//! reproduces the algorithm in safe Rust without runtime linkage.

use thiserror::Error;

/// Super-block size (256 elements per k-quant block).
pub const QK_K: usize = 256;

/// `block_q4_K` size on disk (matches `sizeof(block_q4_K)` in C).
/// 2 × f16 (d, dmin) + 12 byte scales + 128 byte quants = 144 bytes.
pub const BLOCK_Q4_K_SIZE: usize = 2 * 2 + 12 + QK_K / 2;

/// Errors from k-quant operations.
#[derive(Error, Debug)]
pub enum KQuantError {
    #[error("k-quant: input length {actual} is not a multiple of QK_K ({QK_K})")]
    NotBlockAligned { actual: usize },

    #[error("k-quant: input byte length {actual} disagrees with declared block count {n_blocks} × {bytes_per_block}")]
    BlockSizeMismatch {
        actual: usize,
        n_blocks: usize,
        bytes_per_block: usize,
    },
}

/// Q4_K super-block — 256 elements stored at 4.5 bpw.
///
/// Layout matches `block_q4_K` in `/opt/llama.cpp/ggml/src/ggml-common.h:317`
/// byte-for-byte. The struct is `repr(C)` so Rust types decoded
/// from a slice over a 144-byte buffer match the C struct layout
/// position-for-position; this is what the byte-identical gate
/// against llama.cpp's NEON path tests at iter-3b.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4K {
    /// Super-block scale for the 6-bit per-sub-block scales. F16.
    pub d_bits: u16,
    /// Super-block scale for the 6-bit per-sub-block mins. F16.
    pub dmin_bits: u16,
    /// 12 bytes encoding 8 sub-block scales + 8 sub-block mins
    /// (6 bits each). The packing matches
    /// `get_scale_min_k4` in `ggml-quants.c:818`:
    ///
    /// ```text
    /// for j < 4: scales[j] holds 6-bit scale; scales[j+4] holds 6-bit min.
    /// for j >= 4: nibble-packed across scales[j+4] (low 4 bits) and
    ///             the upper 2 bits of scales[j-4] (for the scale)
    ///             / scales[j-0] (for the min).
    /// ```
    pub scales: [u8; 12],
    /// 4-bit packed quants — 256 elements stored as 128 bytes
    /// (low nibble + high nibble per byte). The decode order matches
    /// llama.cpp's `dequantize_row_q4_K` (lines 1467+):
    /// 32 elements from low nibbles, then 32 from high nibbles, per
    /// 64-element half-sub-block.
    pub qs: [u8; QK_K / 2],
}

impl BlockQ4K {
    /// Read a `BlockQ4K` from a 144-byte buffer (little-endian).
    /// Returns `None` if the buffer is shorter than `BLOCK_Q4_K_SIZE`.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < BLOCK_Q4_K_SIZE {
            return None;
        }
        let d_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let dmin_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&bytes[4..16]);
        let mut qs = [0u8; QK_K / 2];
        qs.copy_from_slice(&bytes[16..16 + QK_K / 2]);
        Some(Self {
            d_bits,
            dmin_bits,
            scales,
            qs,
        })
    }

    /// Serialise to 144-byte little-endian buffer (matches the C
    /// memory layout on `aarch64-apple-darwin`). Used by the
    /// byte-identical-to-llama.cpp gate.
    pub fn to_bytes(&self) -> [u8; BLOCK_Q4_K_SIZE] {
        let mut out = [0u8; BLOCK_Q4_K_SIZE];
        out[0..2].copy_from_slice(&self.d_bits.to_le_bytes());
        out[2..4].copy_from_slice(&self.dmin_bits.to_le_bytes());
        out[4..16].copy_from_slice(&self.scales);
        out[16..16 + QK_K / 2].copy_from_slice(&self.qs);
        out
    }

    /// Super-block scale `d` as F32.
    pub fn d(&self) -> f32 {
        half::f16::from_bits(self.d_bits).to_f32()
    }

    /// Super-block min scale `dmin` as F32.
    pub fn dmin(&self) -> f32 {
        half::f16::from_bits(self.dmin_bits).to_f32()
    }
}

/// Decode the 6-bit packed `(scale, min)` pair for sub-block `j`
/// from the 12-byte `scales` array. Mirrors
/// [`get_scale_min_k4`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c#L818)
/// byte-for-byte.
///
/// Returns `(sc, m)` where each is in `[0, 63]` (6-bit values).
#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 0x3F, q[j + 4] & 0x3F)
    } else {
        let sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (sc, m)
    }
}

/// Dequantize a sequence of `block_q4_K` blocks to F32. Pure-Rust
/// port of `dequantize_row_q4_K` (`ggml-quants.c:1467`).
///
/// For each super-block `[256 elements]`:
/// 1. Decode super-block scales `d`, `dmin` from F16 → F32.
/// 2. For each pair of sub-blocks `(j, j+1)` (covering 64 elements):
///    a. Decode the 6-bit `(scale, min)` for sub-block `j` and
///       sub-block `j+1` via [`get_scale_min_k4`].
///    b. Compute per-sub-block dequant scales:
///       `d1 = d * sc1`, `m1 = dmin * m1`,
///       `d2 = d * sc2`, `m2 = dmin * m2`.
///    c. For each of 32 elements in sub-block `j`: low nibble of
///       `qs[l]`, output `d1 * (q & 0xF) - m1`.
///    d. For each of 32 elements in sub-block `j+1`: high nibble of
///       `qs[l]`, output `d2 * (q >> 4) - m2`.
///    e. Advance `qs` pointer by 32 bytes.
///
/// The output buffer must have length exactly `n_blocks × QK_K`.
/// Returns the number of F32 elements written (always
/// `n_blocks × QK_K`).
///
/// **Byte order / endianness**: F16 fields decoded little-endian
/// (matches `aarch64-apple-darwin` / `x86_64-linux-gnu` native
/// layout). Cross-platform big-endian portability is documented but
/// not in scope for ADR-014.
pub fn dequantize_row_q4_k(
    blocks: &[BlockQ4K],
    out: &mut [f32],
) -> Result<usize, KQuantError> {
    let expected = blocks.len() * QK_K;
    if out.len() < expected {
        return Err(KQuantError::BlockSizeMismatch {
            actual: out.len(),
            n_blocks: blocks.len(),
            bytes_per_block: QK_K,
        });
    }

    let mut written = 0usize;
    for block in blocks {
        let d = block.d();
        let dmin = block.dmin();

        // Walk 64 elements at a time (two consecutive sub-blocks of 32).
        // qs offset advances by 32 per pair.
        let mut qs_off = 0usize;
        for j_pair in 0..(QK_K / 64) {
            let is = j_pair * 2;
            let (sc1, m1) = get_scale_min_k4(is, &block.scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &block.scales);
            let d1 = d * (sc1 as f32);
            let m1f = dmin * (m1 as f32);
            let d2 = d * (sc2 as f32);
            let m2f = dmin * (m2 as f32);

            // 32 low-nibble elements (sub-block `is`).
            for l in 0..32 {
                let q = block.qs[qs_off + l] & 0x0F;
                out[written] = d1 * (q as f32) - m1f;
                written += 1;
            }
            // 32 high-nibble elements (sub-block `is + 1`).
            for l in 0..32 {
                let q = block.qs[qs_off + l] >> 4;
                out[written] = d2 * (q as f32) - m2f;
                written += 1;
            }
            qs_off += 32;
        }
    }
    Ok(written)
}

/// Read a packed `block_q4_K` byte buffer (multiple super-blocks
/// concatenated) and decode each into F32. Convenience wrapper over
/// [`BlockQ4K::from_bytes`] + [`dequantize_row_q4_k`].
///
/// `data.len()` must be exactly `n_blocks × BLOCK_Q4_K_SIZE` for some
/// integer `n_blocks > 0`. Output buffer length must be
/// `n_blocks × QK_K`.
pub fn dequantize_row_q4_k_bytes(
    data: &[u8],
    out: &mut [f32],
) -> Result<usize, KQuantError> {
    if data.len() % BLOCK_Q4_K_SIZE != 0 {
        return Err(KQuantError::BlockSizeMismatch {
            actual: data.len(),
            n_blocks: data.len() / BLOCK_Q4_K_SIZE,
            bytes_per_block: BLOCK_Q4_K_SIZE,
        });
    }
    let n_blocks = data.len() / BLOCK_Q4_K_SIZE;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let start = i * BLOCK_Q4_K_SIZE;
        let end = start + BLOCK_Q4_K_SIZE;
        let block = BlockQ4K::from_bytes(&data[start..end]).ok_or(
            KQuantError::BlockSizeMismatch {
                actual: data.len(),
                n_blocks,
                bytes_per_block: BLOCK_Q4_K_SIZE,
            },
        )?;
        blocks.push(block);
    }
    dequantize_row_q4_k(&blocks, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Block size matches llama.cpp's static_assert
    /// `sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2`.
    #[test]
    fn block_q4_k_size_matches_c_struct() {
        assert_eq!(BLOCK_Q4_K_SIZE, 4 + 12 + 128);
        assert_eq!(BLOCK_Q4_K_SIZE, 144);
    }

    /// `get_scale_min_k4` decoded byte-for-byte against the C
    /// reference. Hand-computed expected values for a deterministic
    /// scales array.
    #[test]
    fn get_scale_min_k4_low_branch() {
        // For j < 4: sc = q[j] & 63, m = q[j+4] & 63.
        // q[0..4] = [10, 20, 30, 40] (all < 63); q[4..8] = [50, 60, 5, 15]
        // q[8..12] = anything (only used for j >= 4 path)
        let q = [10u8, 20, 30, 40, 50, 60, 5, 15, 0xAB, 0xCD, 0xEF, 0x12];
        for (j, expected_sc, expected_m) in &[
            (0, 10, 50),
            (1, 20, 60),
            (2, 30, 5),
            (3, 40, 15),
        ] {
            let (sc, m) = get_scale_min_k4(*j, &q);
            assert_eq!(sc, *expected_sc, "j={j} sc");
            assert_eq!(m, *expected_m, "j={j} m");
        }
    }

    /// `get_scale_min_k4` j >= 4 branch — the nibble-packed encoding
    /// reading the scale across `scales[j+4]` (low 4 bits) +
    /// `scales[j-4]` (upper 2 bits as bits 5-4 of result).
    ///
    /// Hand-encode a known (sc, m) pair into the layout, then decode.
    #[test]
    fn get_scale_min_k4_high_branch() {
        // Want for j=4: sc = 0x2A (= 0b101010 = 42), m = 0x35 (= 53).
        // Encoding rule (j >= 4):
        //   sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
        //     low 4 bits in scales[j+4] = sc & 0x0F = 0x0A
        //     upper 2 bits in scales[j-4] >> 6 = (sc >> 4) & 3 = 0x02
        //     so scales[j-4] = ... | (0x02 << 6) = 0x80
        //   m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
        //     low 4 bits in scales[j+4] >> 4 = m & 0x0F = 0x05
        //     upper 2 bits in scales[j] >> 6 = (m >> 4) & 3 = 0x03
        //     so scales[j] = ... | (0x03 << 6) = 0xC0
        // For j=4 specifically: scales[j-4] = scales[0]; scales[j+4] = scales[8]; scales[j] = scales[4].
        let mut q = [0u8; 12];
        q[0] = 0x80;                       // upper 2 bits of sc
        q[8] = 0x0A | (0x05 << 4);         // = 0x5A: low 4 bits of sc + low 4 bits of m
        q[4] = 0xC0;                       // upper 2 bits of m

        let (sc, m) = get_scale_min_k4(4, &q);
        assert_eq!(sc, 0x2A, "sc (got {sc:#x}, want 0x2A)");
        assert_eq!(m, 0x35, "m (got {m:#x}, want 0x35)");
    }

    /// Dequantize an all-zeros block (every quant nibble = 0):
    /// each output element should be `0 * d * sc - dmin * m = -dmin * m`.
    /// With dmin = 0, every output is 0.
    #[test]
    fn dequantize_zero_block_with_zero_min() {
        let block = BlockQ4K {
            d_bits: half::f16::from_f32(1.0).to_bits(),
            dmin_bits: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; 12], // sc=0 for j<4, m=0 too; high-branch reads 0s.
            qs: [0u8; QK_K / 2],
        };

        let mut out = vec![0.0_f32; QK_K];
        let n = dequantize_row_q4_k(std::slice::from_ref(&block), &mut out).unwrap();
        assert_eq!(n, QK_K);
        // All scales 0 + all quants 0 → all outputs 0.
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "out[{i}] = {v}");
        }
    }

    /// Dequantize a block with sc=10, m=0 for sub-block 0; quants
    /// alternating 0..15. Expected output:
    /// `out[i] = d * sc * (q & 0xF) - dmin * m`
    /// With d=1, sc=10, m=0: `out[i] = 10 * (q & 0xF)`.
    #[test]
    fn dequantize_known_sub_block_low_nibble() {
        let mut scales = [0u8; 12];
        // Sub-block 0: sc=10, m=0 (j=0 path, scales[0]=10, scales[4]=0).
        scales[0] = 10;
        scales[4] = 0;
        // Other sub-blocks: keep sc=0, m=0 → output zero.

        // qs[0..32]: low nibbles encode 0..15 (cycle).
        // qs[l] = (l & 0xF) | (next_high << 4); but we don't care about high.
        let mut qs = [0u8; QK_K / 2];
        for l in 0..32 {
            qs[l] = (l & 0xF) as u8; // high nibble = 0
        }

        let block = BlockQ4K {
            d_bits: half::f16::from_f32(1.0).to_bits(),
            dmin_bits: half::f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        };

        let mut out = vec![0.0_f32; QK_K];
        dequantize_row_q4_k(std::slice::from_ref(&block), &mut out).unwrap();

        // First 32 elements (sub-block 0): out[l] = 10 * (l & 0xF).
        for l in 0..32 {
            let expected = 10.0 * ((l & 0xF) as f32);
            assert!(
                (out[l] - expected).abs() < 1e-5,
                "low nibble out[{l}] = {} (want {expected})",
                out[l]
            );
        }
    }

    /// Round-trip BlockQ4K via to_bytes / from_bytes preserves the
    /// struct byte-for-byte.
    #[test]
    fn block_q4_k_byte_round_trip() {
        let block = BlockQ4K {
            d_bits: 0xABCD,
            dmin_bits: 0x1234,
            scales: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            qs: {
                let mut a = [0u8; QK_K / 2];
                for (i, slot) in a.iter_mut().enumerate() {
                    *slot = (i & 0xFF) as u8;
                }
                a
            },
        };
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), BLOCK_Q4_K_SIZE);
        let decoded = BlockQ4K::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.d_bits, block.d_bits);
        assert_eq!(decoded.dmin_bits, block.dmin_bits);
        assert_eq!(decoded.scales, block.scales);
        assert_eq!(decoded.qs, block.qs);
    }

    /// `dequantize_row_q4_k_bytes` rejects mis-aligned input length.
    #[test]
    fn dequantize_bytes_rejects_misaligned_length() {
        // 144 bytes per block — 200 is a partial block.
        let bad = vec![0u8; 200];
        let mut out = vec![0.0_f32; QK_K];
        let err = dequantize_row_q4_k_bytes(&bad, &mut out).unwrap_err();
        match err {
            KQuantError::BlockSizeMismatch {
                bytes_per_block, ..
            } => {
                assert_eq!(bytes_per_block, BLOCK_Q4_K_SIZE);
            }
            _ => panic!("expected BlockSizeMismatch"),
        }
    }

    /// `dequantize_row_q4_k_bytes` decodes a multi-block buffer and
    /// produces N × QK_K outputs.
    #[test]
    fn dequantize_bytes_multi_block() {
        let block = BlockQ4K {
            d_bits: half::f16::from_f32(0.0).to_bits(),
            dmin_bits: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; 12],
            qs: [0u8; QK_K / 2],
        };
        // Concatenate 3 blocks → 432 bytes.
        let mut buf = Vec::with_capacity(3 * BLOCK_Q4_K_SIZE);
        for _ in 0..3 {
            buf.extend_from_slice(&block.to_bytes());
        }
        assert_eq!(buf.len(), 3 * BLOCK_Q4_K_SIZE);

        let mut out = vec![0.0_f32; 3 * QK_K];
        let n = dequantize_row_q4_k_bytes(&buf, &mut out).unwrap();
        assert_eq!(n, 3 * QK_K);
        // d_bits = 0 → d = 0 → every output is 0 (regardless of scales/qs).
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    /// Output buffer too small surfaces a typed error.
    #[test]
    fn dequantize_rejects_short_output_buffer() {
        let block = BlockQ4K {
            d_bits: 0,
            dmin_bits: 0,
            scales: [0u8; 12],
            qs: [0u8; QK_K / 2],
        };
        let mut out = vec![0.0_f32; QK_K - 1]; // one too short
        let err = dequantize_row_q4_k(std::slice::from_ref(&block), &mut out).unwrap_err();
        match err {
            KQuantError::BlockSizeMismatch { actual, .. } => {
                assert_eq!(actual, QK_K - 1);
            }
            _ => panic!("expected BlockSizeMismatch"),
        }
    }
}
