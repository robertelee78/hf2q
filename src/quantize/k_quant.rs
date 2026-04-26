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
//! ## This iter — dequantize for Q4_K, Q5_K, Q6_K
//!
//! P7 iter-3a landed Q4_K dequantize. P7 iter-3c (this iter) extends
//! the dequantize side to **Q5_K** and **Q6_K**, completing the
//! decode-direction port of llama.cpp's k-quant family. Each block
//! type is `repr(C)` and round-trips byte-for-byte through the
//! corresponding C struct on `aarch64-apple-darwin`.
//!
//! The **quantize** direction (depends on `make_qkx2_quants` /
//! `make_qkx3_quants` codebook search routines + per-column-weighted
//! MSE for the imatrix variant) is the harder half and lands in
//! subsequent P7 iter-3b.
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

/// `block_q5_K` size on disk (matches `sizeof(block_q5_K)` in C).
/// 2 × f16 (d, dmin) + 12 byte scales + 32 byte qh (high bit) + 128
/// byte qs (low 4 bits) = 176 bytes.
pub const BLOCK_Q5_K_SIZE: usize = 2 * 2 + 12 + QK_K / 8 + QK_K / 2;

/// `block_q6_K` size on disk (matches `sizeof(block_q6_K)` in C).
/// 128 byte ql (lower 4 bits) + 64 byte qh (upper 2 bits) + 16 byte
/// signed scales + 1 × f16 d = 210 bytes.
pub const BLOCK_Q6_K_SIZE: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;

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

/// Q5_K super-block — 256 elements stored at 5.5 bpw.
///
/// Layout matches `block_q5_K` in `/opt/llama.cpp/ggml/src/ggml-common.h:333`
/// byte-for-byte. The struct is `repr(C)` so Rust types decoded
/// from a slice over a 176-byte buffer match the C struct layout
/// position-for-position.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5K {
    /// Super-block scale for the 6-bit per-sub-block scales. F16.
    pub d_bits: u16,
    /// Super-block scale for the 6-bit per-sub-block mins. F16.
    pub dmin_bits: u16,
    /// 12 bytes encoding 8 sub-block scales + 8 sub-block mins
    /// (6 bits each). Same packing as Q4_K (`get_scale_min_k4`).
    pub scales: [u8; 12],
    /// 32 bytes — high bit of each 5-bit quant (256 elements / 8).
    /// Bit `b` of `qh[l]` selects element `l + b * 32` mod 256.
    pub qh: [u8; QK_K / 8],
    /// 128 bytes — low 4 bits of each 5-bit quant (256 elements / 2).
    pub qs: [u8; QK_K / 2],
}

impl BlockQ5K {
    /// Read a `BlockQ5K` from a 176-byte buffer (little-endian).
    /// Returns `None` if the buffer is shorter than `BLOCK_Q5_K_SIZE`.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < BLOCK_Q5_K_SIZE {
            return None;
        }
        let d_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let dmin_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&bytes[4..16]);
        let mut qh = [0u8; QK_K / 8];
        qh.copy_from_slice(&bytes[16..16 + QK_K / 8]);
        let mut qs = [0u8; QK_K / 2];
        qs.copy_from_slice(&bytes[16 + QK_K / 8..16 + QK_K / 8 + QK_K / 2]);
        Some(Self {
            d_bits,
            dmin_bits,
            scales,
            qh,
            qs,
        })
    }

    /// Serialise to a 176-byte little-endian buffer (matches the C
    /// memory layout on `aarch64-apple-darwin`).
    pub fn to_bytes(&self) -> [u8; BLOCK_Q5_K_SIZE] {
        let mut out = [0u8; BLOCK_Q5_K_SIZE];
        out[0..2].copy_from_slice(&self.d_bits.to_le_bytes());
        out[2..4].copy_from_slice(&self.dmin_bits.to_le_bytes());
        out[4..16].copy_from_slice(&self.scales);
        out[16..16 + QK_K / 8].copy_from_slice(&self.qh);
        out[16 + QK_K / 8..16 + QK_K / 8 + QK_K / 2].copy_from_slice(&self.qs);
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

/// Dequantize a sequence of `block_q5_K` blocks to F32. Pure-Rust
/// port of `dequantize_row_q5_K` (`ggml-quants.c:1669`).
///
/// For each super-block `[256 elements]`:
/// 1. Decode super-block scales `d`, `dmin` from F16 → F32.
/// 2. Walk 64 elements at a time (a pair of 32-element sub-blocks),
///    advancing the per-pair `(u1, u2)` bit-mask shifters. `u1`/`u2`
///    select the high bit from `qh[l]`:
///    - element `l` (low nibble): high bit = `(qh[l] & u1) ? 16 : 0`
///    - element `l+32` (high nibble): high bit = `(qh[l] & u2) ? 16 : 0`
///
///    After the pair completes, both shift left by 2 (`u1 <<= 2`,
///    `u2 <<= 2`).
/// 3. For each pair, decode `(sc, m)` for sub-blocks `is` and `is+1`
///    via [`get_scale_min_k4`].
///
/// The output buffer must have length at least `n_blocks × QK_K`.
pub fn dequantize_row_q5_k(
    blocks: &[BlockQ5K],
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

        let mut qs_off = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j_pair in 0..(QK_K / 64) {
            let is = j_pair * 2;
            let (sc1, m1) = get_scale_min_k4(is, &block.scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &block.scales);
            let d1 = d * (sc1 as f32);
            let m1f = dmin * (m1 as f32);
            let d2 = d * (sc2 as f32);
            let m2f = dmin * (m2 as f32);

            // 32 low-nibble + high-bit-from-u1 elements.
            for l in 0..32 {
                let q_low = (block.qs[qs_off + l] & 0x0F) as i32;
                let q_high = if (block.qh[l] & u1) != 0 { 16 } else { 0 };
                let q = q_low + q_high;
                out[written] = d1 * (q as f32) - m1f;
                written += 1;
            }
            // 32 high-nibble + high-bit-from-u2 elements.
            for l in 0..32 {
                let q_low = (block.qs[qs_off + l] >> 4) as i32;
                let q_high = if (block.qh[l] & u2) != 0 { 16 } else { 0 };
                let q = q_low + q_high;
                out[written] = d2 * (q as f32) - m2f;
                written += 1;
            }
            qs_off += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(written)
}

/// Read a packed `block_q5_K` byte buffer (multiple super-blocks
/// concatenated) and decode each into F32. Convenience wrapper over
/// [`BlockQ5K::from_bytes`] + [`dequantize_row_q5_k`].
pub fn dequantize_row_q5_k_bytes(
    data: &[u8],
    out: &mut [f32],
) -> Result<usize, KQuantError> {
    if data.len() % BLOCK_Q5_K_SIZE != 0 {
        return Err(KQuantError::BlockSizeMismatch {
            actual: data.len(),
            n_blocks: data.len() / BLOCK_Q5_K_SIZE,
            bytes_per_block: BLOCK_Q5_K_SIZE,
        });
    }
    let n_blocks = data.len() / BLOCK_Q5_K_SIZE;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let start = i * BLOCK_Q5_K_SIZE;
        let end = start + BLOCK_Q5_K_SIZE;
        let block = BlockQ5K::from_bytes(&data[start..end]).ok_or(
            KQuantError::BlockSizeMismatch {
                actual: data.len(),
                n_blocks,
                bytes_per_block: BLOCK_Q5_K_SIZE,
            },
        )?;
        blocks.push(block);
    }
    dequantize_row_q5_k(&blocks, out)
}

/// Q6_K super-block — 256 elements stored at 6.5625 bpw.
///
/// Layout matches `block_q6_K` in `/opt/llama.cpp/ggml/src/ggml-common.h:352`
/// byte-for-byte. Unlike Q4_K/Q5_K (8 sub-blocks of 32 elements with
/// 6-bit packed scale+min), Q6_K uses 16 sub-blocks of 16 elements
/// with 8-bit signed scales and **no per-sub-block min**. Quantised
/// values are stored sign-shifted so `q ∈ [-32, 31]`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6K {
    /// 128 bytes — lower 4 bits of each 6-bit quant.
    pub ql: [u8; QK_K / 2],
    /// 64 bytes — upper 2 bits of each 6-bit quant (4 elements / byte).
    pub qh: [u8; QK_K / 4],
    /// 16 signed-byte scales (one per 16-element sub-block).
    pub scales: [i8; QK_K / 16],
    /// Super-block scale. F16.
    pub d_bits: u16,
}

impl BlockQ6K {
    /// Read a `BlockQ6K` from a 210-byte buffer (little-endian).
    /// Returns `None` if the buffer is shorter than `BLOCK_Q6_K_SIZE`.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < BLOCK_Q6_K_SIZE {
            return None;
        }
        let mut ql = [0u8; QK_K / 2];
        ql.copy_from_slice(&bytes[0..QK_K / 2]);
        let mut qh = [0u8; QK_K / 4];
        qh.copy_from_slice(&bytes[QK_K / 2..QK_K / 2 + QK_K / 4]);
        let mut scales = [0i8; QK_K / 16];
        for (i, slot) in scales.iter_mut().enumerate() {
            *slot = bytes[QK_K / 2 + QK_K / 4 + i] as i8;
        }
        let d_off = QK_K / 2 + QK_K / 4 + QK_K / 16;
        let d_bits = u16::from_le_bytes([bytes[d_off], bytes[d_off + 1]]);
        Some(Self {
            ql,
            qh,
            scales,
            d_bits,
        })
    }

    /// Serialise to 210-byte little-endian buffer (matches the C
    /// memory layout on `aarch64-apple-darwin`).
    pub fn to_bytes(&self) -> [u8; BLOCK_Q6_K_SIZE] {
        let mut out = [0u8; BLOCK_Q6_K_SIZE];
        out[0..QK_K / 2].copy_from_slice(&self.ql);
        out[QK_K / 2..QK_K / 2 + QK_K / 4].copy_from_slice(&self.qh);
        for (i, &s) in self.scales.iter().enumerate() {
            out[QK_K / 2 + QK_K / 4 + i] = s as u8;
        }
        let d_off = QK_K / 2 + QK_K / 4 + QK_K / 16;
        out[d_off..d_off + 2].copy_from_slice(&self.d_bits.to_le_bytes());
        out
    }

    /// Super-block scale `d` as F32.
    pub fn d(&self) -> f32 {
        half::f16::from_bits(self.d_bits).to_f32()
    }
}

/// Dequantize a sequence of `block_q6_K` blocks to F32. Pure-Rust
/// port of `dequantize_row_q6_K` (`ggml-quants.c:1877`).
///
/// For each super-block `[256 elements]`:
/// 1. Decode super-block scale `d` from F16 → F32.
/// 2. Walk in 128-element halves (`n = 0`, then `n = 128`).
///    For `l ∈ 0..32`, decode four quants from `(ql[l..l+32], qh[l])`:
///    - `q1 = ((ql[l]      & 0xF) | (qh[l] & 3)         << 4) - 32`
///    - `q2 = ((ql[l + 32] & 0xF) | ((qh[l] >> 2) & 3)  << 4) - 32`
///    - `q3 = ((ql[l]       >> 4) | ((qh[l] >> 4) & 3)  << 4) - 32`
///    - `q4 = ((ql[l + 32]  >> 4) | ((qh[l] >> 6) & 3)  << 4) - 32`
///
///    The `is = l / 16` index walks 0..2 over the 32 inner iterations,
///    selecting `scales[is + 0..is + 6]` step-2 for the four lanes.
/// 3. Outputs at `y[l]`, `y[l+32]`, `y[l+64]`, `y[l+96]` are
///    `d * scales[is+k] * qk` for `k = 0,2,4,6`.
/// 4. After the half completes, `y += 128`, `ql += 64`, `qh += 32`,
///    `sc += 8`.
pub fn dequantize_row_q6_k(
    blocks: &[BlockQ6K],
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

        // Two halves of 128 elements each.
        for half in 0..(QK_K / 128) {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;

            for l in 0..32 {
                let is = l / 16;
                let qh_byte = block.qh[qh_off + l];
                let q1 = ((block.ql[ql_off + l] & 0x0F) as i32
                    | ((qh_byte & 0x3) as i32) << 4)
                    - 32;
                let q2 = ((block.ql[ql_off + l + 32] & 0x0F) as i32
                    | (((qh_byte >> 2) & 0x3) as i32) << 4)
                    - 32;
                let q3 = ((block.ql[ql_off + l] >> 4) as i32
                    | (((qh_byte >> 4) & 0x3) as i32) << 4)
                    - 32;
                let q4 = ((block.ql[ql_off + l + 32] >> 4) as i32
                    | (((qh_byte >> 6) & 0x3) as i32) << 4)
                    - 32;

                let s0 = block.scales[sc_off + is] as f32;
                let s2 = block.scales[sc_off + is + 2] as f32;
                let s4 = block.scales[sc_off + is + 4] as f32;
                let s6 = block.scales[sc_off + is + 6] as f32;

                out[written + l] = d * s0 * (q1 as f32);
                out[written + l + 32] = d * s2 * (q2 as f32);
                out[written + l + 64] = d * s4 * (q3 as f32);
                out[written + l + 96] = d * s6 * (q4 as f32);
            }
            written += 128;
        }
    }
    Ok(written)
}

/// Read a packed `block_q6_K` byte buffer (multiple super-blocks
/// concatenated) and decode each into F32. Convenience wrapper over
/// [`BlockQ6K::from_bytes`] + [`dequantize_row_q6_k`].
pub fn dequantize_row_q6_k_bytes(
    data: &[u8],
    out: &mut [f32],
) -> Result<usize, KQuantError> {
    if data.len() % BLOCK_Q6_K_SIZE != 0 {
        return Err(KQuantError::BlockSizeMismatch {
            actual: data.len(),
            n_blocks: data.len() / BLOCK_Q6_K_SIZE,
            bytes_per_block: BLOCK_Q6_K_SIZE,
        });
    }
    let n_blocks = data.len() / BLOCK_Q6_K_SIZE;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let start = i * BLOCK_Q6_K_SIZE;
        let end = start + BLOCK_Q6_K_SIZE;
        let block = BlockQ6K::from_bytes(&data[start..end]).ok_or(
            KQuantError::BlockSizeMismatch {
                actual: data.len(),
                n_blocks,
                bytes_per_block: BLOCK_Q6_K_SIZE,
            },
        )?;
        blocks.push(block);
    }
    dequantize_row_q6_k(&blocks, out)
}

// ────────────────────────── Quantize side ──────────────────────────

/// Round `fval` to the nearest integer using llama.cpp's bit-trick
/// (`ggml-quants.c:559`). Adds the binary-fixed offset
/// `12582912.f = 0x4B400000` (IEEE-754 single-precision: exponent 23,
/// mantissa = 0), which forces the mantissa's lowest bits to align
/// with the integer boundary; the masked mantissa minus 0x00400000
/// yields the round-to-nearest-even result.
///
/// Equivalent to `fval.round() as i32` for finite inputs in
/// `[-4194303, 4194303]`, with the bit-trick guaranteeing the
/// **same FPU rounding** as the reference (matters for byte-identity
/// against the NEON path).
#[inline]
pub fn nearest_int(fval: f32) -> i32 {
    debug_assert!(
        fval.abs() <= 4_194_303.0,
        "nearest_int: |fval| > 4194303 (got {fval})"
    );
    let val = fval + 12_582_912.0;
    let bits = val.to_bits() as i32;
    (bits & 0x007F_FFFF) - 0x0040_0000
}

/// Codebook search for `Q4_K` / `Q5_K` per-sub-block scale + min.
/// Pure-Rust port of `make_qkx2_quants` (`ggml-quants.c:737`).
///
/// Given `n` source values `x[0..n]` and `n` per-element weights
/// `weights[0..n]`, finds the best `(scale, -min)` pair that
/// minimises `sum_i weights[i] * f(scale * L[i] + min - x[i])`
/// where `L[i] ∈ [0, nmax]` and `f` is square (when `use_mad=false`)
/// or absolute (when `use_mad=true`). Returns the chosen `scale`,
/// writes quants into `L`, and writes `-min` into `the_min` (note
/// the **negated** sign convention — same as llama.cpp).
///
/// `Laux` is a `[u8; n]` scratch buffer for the inner per-step
/// quant set (avoids re-allocation in the search loop).
///
/// `nstep` ≥ 1 enables the iterative scale-min refinement search;
/// `nstep < 1` returns after the initial scale guess. `rmin` and
/// `rdelta` parameterise the search step (typically `-1.0` and `0.1`).
///
/// **Sign convention**: `the_min` receives `-min` (so callers see
/// the negative-min convention used in `quantize_row_q4_K_ref`).
///
/// Tested against scalar reference behavior; NEON-byte-identity
/// is a separate gate at iter-3b2 (test fixture against
/// pre-recorded llama.cpp output).
#[allow(clippy::too_many_arguments)]
pub fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> (f32, f32) {
    debug_assert_eq!(x.len(), n, "x.len() must equal n");
    debug_assert_eq!(weights.len(), n, "weights.len() must equal n");
    debug_assert!(l.len() >= n, "L buffer must be ≥ n");
    debug_assert!(laux.len() >= n, "Laux buffer must be ≥ n");

    // Initial pass: min, max, sum_w, sum_x.
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        for slot in l.iter_mut().take(n) {
            *slot = 0;
        }
        return (0.0, -min);
    }

    let mut iscale = (nmax as f32) / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let lq = nearest_int(iscale * (x[i] - min));
        let lq = lq.max(0).min(nmax);
        l[i] = lq as u8;
        let mut diff = scale * (l[i] as f32) + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }

    // Refinement loop.
    for is in 0..=nstep {
        iscale = (rmin + rdelta * (is as f32) + (nmax as f32)) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let lq = nearest_int(iscale * (x[i] - min));
            let lq = lq.max(0).min(nmax);
            laux[i] = lq as u8;
            let w = weights[i];
            let lf = lq as f32;
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let det = sum_w * sum_l2 - sum_l * sum_l;
        if det > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * (laux[i] as f32) + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                l[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// Quantize `row` (length `n_blocks × QK_K`) to a sequence of
/// `BlockQ4K` super-blocks. Pure-Rust port of `quantize_row_q4_K_ref`
/// (`ggml-quants.c:1395`).
///
/// Per super-block:
/// 1. For each of 8 sub-blocks of 32 elements:
///    - Compute the per-sub-block weight `weights[l] = av_x + |x[l]|`
///      where `av_x = sqrt(sum(x²)/32)`.
///    - Run `make_qkx2_quants` with `nmax=15, rmin=-1.0, rdelta=0.1,
///      nstep=20, use_mad=false` to get `scales[j]` and `mins[j]`.
/// 2. Find super-block max scale + max min, compute super-block
///    inverse scales `inv_scale = 63/max_scale`, `inv_min = 63/max_min`.
/// 3. Pack 6-bit per-sub-block scales/mins into `scales[12]` per
///    `get_scale_min_k4`'s inverse.
/// 4. Re-derive 4-bit quants from final `(d, dmin, sc, m)` for each
///    sub-block: `L[i] = round((x[i] + dm) / d)` clamped to `[0, 15]`.
/// 5. Pack low-nibble + high-nibble into `qs[128]`.
///
/// Output `blocks.len()` must equal `row.len() / QK_K` and `row.len()`
/// must be a multiple of `QK_K`.
pub fn quantize_row_q4_k(row: &[f32], blocks: &mut [BlockQ4K]) -> Result<(), KQuantError> {
    if !row.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { actual: row.len() });
    }
    let nb = row.len() / QK_K;
    if blocks.len() < nb {
        return Err(KQuantError::BlockSizeMismatch {
            actual: blocks.len(),
            n_blocks: nb,
            bytes_per_block: BLOCK_Q4_K_SIZE,
        });
    }

    let mut l_buf = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut weights_buf = [0.0f32; 32];
    let mut mins = [0.0f32; QK_K / 32];
    let mut scales = [0.0f32; QK_K / 32];

    for i in 0..nb {
        let x = &row[i * QK_K..(i + 1) * QK_K];
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;

        // Per-sub-block: compute weights, run codebook search.
        for j in 0..(QK_K / 32) {
            let xj = &x[32 * j..32 * j + 32];
            let mut sum_x2 = 0.0f32;
            for &v in xj.iter() {
                sum_x2 += v * v;
            }
            let av_x = (sum_x2 / 32.0).sqrt();
            for (l, &v) in weights_buf.iter_mut().zip(xj.iter()) {
                *l = av_x + v.abs();
            }
            let (scale, the_min) = make_qkx2_quants(
                32,
                15,
                xj,
                &weights_buf,
                &mut l_buf[32 * j..32 * j + 32],
                &mut laux,
                -1.0,
                0.1,
                20,
                false,
            );
            scales[j] = scale;
            mins[j] = the_min;
            if scale > max_scale {
                max_scale = scale;
            }
            if the_min > max_min {
                max_min = the_min;
            }
        }

        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut packed_scales = [0u8; 12];

        // Pack per-sub-block 6-bit scales + mins into the 12-byte scales array.
        for j in 0..(QK_K / 32) {
            let ls = nearest_int(inv_scale * scales[j]).max(0).min(63) as u8;
            let lm = nearest_int(inv_min * mins[j]).max(0).min(63) as u8;
            if j < 4 {
                packed_scales[j] = ls;
                packed_scales[j + 4] = lm;
            } else {
                packed_scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                packed_scales[j - 4] |= (ls >> 4) << 6;
                packed_scales[j] |= (lm >> 4) << 6;
            }
        }

        let d_f = max_scale / 63.0;
        let dmin_f = max_min / 63.0;
        let d_bits = half::f16::from_f32(d_f).to_bits();
        let dmin_bits = half::f16::from_f32(dmin_f).to_bits();
        let d = half::f16::from_bits(d_bits).to_f32();
        let dmin = half::f16::from_bits(dmin_bits).to_f32();

        // Re-quantize using the final (d, dmin, sc, m) for each sub-block.
        for j in 0..(QK_K / 32) {
            let (sc, m) = get_scale_min_k4(j, &packed_scales);
            let d_eff = d * (sc as f32);
            if d_eff == 0.0 {
                continue;
            }
            let dm = dmin * (m as f32);
            for ii in 0..32 {
                let lq = nearest_int((x[32 * j + ii] + dm) / d_eff)
                    .max(0)
                    .min(15);
                l_buf[32 * j + ii] = lq as u8;
            }
        }

        // Pack 4-bit quants: low nibble at l, high nibble at l+32.
        let mut qs = [0u8; QK_K / 2];
        let mut qoff = 0usize;
        for j_pair in 0..(QK_K / 64) {
            for l in 0..32 {
                qs[qoff + l] = l_buf[64 * j_pair + l] | (l_buf[64 * j_pair + l + 32] << 4);
            }
            qoff += 32;
        }

        blocks[i] = BlockQ4K {
            d_bits,
            dmin_bits,
            scales: packed_scales,
            qs,
        };
    }
    Ok(())
}

/// Quantize `row` (length `n_blocks × QK_K`) to a sequence of
/// `BlockQ5K` super-blocks. Pure-Rust port of `quantize_row_q5_K_ref`
/// (`ggml-quants.c:1582`).
///
/// Structurally parallel to [`quantize_row_q4_k`] with the following
/// differences:
/// - `make_qkx2_quants` is called with `nmax=31` (5-bit), `rmin=-0.5`,
///   `rdelta=0.1`, `nstep=15` (vs `nmax=15, rmin=-1.0, nstep=20` for Q4_K).
/// - Re-derived per-element `L` is clamped to `[0, 31]` (5-bit).
/// - Pack: `L > 15` triggers the high-bit pack into `qh[32]`; the
///   per-pair mask shifters `(m1, m2)` start at `(1, 2)` and shift left
///   by 2 per 64-element pair (mirrors the dequantize-side `(u1, u2)`).
pub fn quantize_row_q5_k(row: &[f32], blocks: &mut [BlockQ5K]) -> Result<(), KQuantError> {
    if !row.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { actual: row.len() });
    }
    let nb = row.len() / QK_K;
    if blocks.len() < nb {
        return Err(KQuantError::BlockSizeMismatch {
            actual: blocks.len(),
            n_blocks: nb,
            bytes_per_block: BLOCK_Q5_K_SIZE,
        });
    }

    let mut l_buf = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut weights_buf = [0.0f32; 32];
    let mut mins = [0.0f32; QK_K / 32];
    let mut scales = [0.0f32; QK_K / 32];

    for i in 0..nb {
        let x = &row[i * QK_K..(i + 1) * QK_K];
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;

        // Per-sub-block: codebook search at nmax=31.
        for j in 0..(QK_K / 32) {
            let xj = &x[32 * j..32 * j + 32];
            let mut sum_x2 = 0.0f32;
            for &v in xj.iter() {
                sum_x2 += v * v;
            }
            let av_x = (sum_x2 / 32.0).sqrt();
            for (l, &v) in weights_buf.iter_mut().zip(xj.iter()) {
                *l = av_x + v.abs();
            }
            let (scale, the_min) = make_qkx2_quants(
                32,
                31,
                xj,
                &weights_buf,
                &mut l_buf[32 * j..32 * j + 32],
                &mut laux,
                -0.5,
                0.1,
                15,
                false,
            );
            scales[j] = scale;
            mins[j] = the_min;
            if scale > max_scale {
                max_scale = scale;
            }
            if the_min > max_min {
                max_min = the_min;
            }
        }

        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut packed_scales = [0u8; 12];

        for j in 0..(QK_K / 32) {
            let ls = nearest_int(inv_scale * scales[j]).max(0).min(63) as u8;
            let lm = nearest_int(inv_min * mins[j]).max(0).min(63) as u8;
            if j < 4 {
                packed_scales[j] = ls;
                packed_scales[j + 4] = lm;
            } else {
                packed_scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                packed_scales[j - 4] |= (ls >> 4) << 6;
                packed_scales[j] |= (lm >> 4) << 6;
            }
        }

        let d_f = max_scale / 63.0;
        let dmin_f = max_min / 63.0;
        let d_bits = half::f16::from_f32(d_f).to_bits();
        let dmin_bits = half::f16::from_f32(dmin_f).to_bits();
        let d = half::f16::from_bits(d_bits).to_f32();
        let dmin = half::f16::from_bits(dmin_bits).to_f32();

        // Re-quantize L using final (d, dmin, sc, m) — clamp to 5-bit [0,31].
        for j in 0..(QK_K / 32) {
            let (sc, m) = get_scale_min_k4(j, &packed_scales);
            let d_eff = d * (sc as f32);
            if d_eff == 0.0 {
                continue;
            }
            let dm = dmin * (m as f32);
            for ii in 0..32 {
                let lq = nearest_int((x[32 * j + ii] + dm) / d_eff)
                    .max(0)
                    .min(31);
                l_buf[32 * j + ii] = lq as u8;
            }
        }

        // Pack 5-bit quants: low 4 bits in qs, high bit in qh.
        let mut qs = [0u8; QK_K / 2];
        let mut qh = [0u8; QK_K / 8];
        let mut m1: u8 = 1;
        let mut m2: u8 = 2;
        let mut ql_off = 0usize;
        for j_pair in 0..(QK_K / 64) {
            let n = j_pair * 64;
            for jj in 0..32 {
                let mut l1 = l_buf[n + jj] as i32;
                if l1 > 15 {
                    l1 -= 16;
                    qh[jj] |= m1;
                }
                let mut l2 = l_buf[n + jj + 32] as i32;
                if l2 > 15 {
                    l2 -= 16;
                    qh[jj] |= m2;
                }
                qs[ql_off + jj] = (l1 as u8) | ((l2 as u8) << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
            ql_off += 32;
        }

        blocks[i] = BlockQ5K {
            d_bits,
            dmin_bits,
            scales: packed_scales,
            qh,
            qs,
        };
    }
    Ok(())
}

/// Threshold below which `make_qx_quants` treats the entire group as
/// zero. Mirrors `GROUP_MAX_EPS` in `ggml-quants.c:16` (1e-15).
const GROUP_MAX_EPS: f32 = 1e-15;

/// Symmetric codebook quantizer for Q6_K (and Q3_K, IQ ranges).
/// Pure-Rust port of `make_qx_quants` (`ggml-quants.c:566`).
///
/// Unlike [`make_qkx2_quants`] (which subtracts a per-sub-block min
/// before quantizing — `q ∈ [0, nmax]`), this one is **symmetric**:
/// quants live in `[-nmax, nmax-1]` and are stored at offset `+nmax`
/// in `L[i] ∈ [0, 2*nmax-1]`. Used by Q6_K (and Q3_K) where there's
/// no per-sub-block min.
///
/// Algorithm:
/// 1. Find `max = x[argmax_abs]`. Early-return zero if `|max| < eps`.
/// 2. Initial guess `iscale = -nmax / max`.
/// 3. If `rmse_type == 0`: write quants and return `1/iscale`.
/// 4. Else: refine with weighted moments. The weight is selected by
///    `rmse_type`:
///    - `1`: `w = x²` (importance ≈ value magnitude squared)
///    - `2`: `w = 1` (uniform)
///    - `3`: `w = |x|`
///    - other: `w = sqrt(|x|)`
///    Custom weights via `qw` override the rmse_type-derived weights.
///    Iterate `is ∈ [-9, 9] \ {0}` shifting `iscale` by `0.1*is`.
///    Pick the `is` with maximum `sumlx² / suml2` ratio.
/// 5. Return final `scale = sumlx / suml2`.
///
/// `rmse_type < 0` means "return early after the initial weighted
/// scale" — used by Q3_K's quick path. Not currently exercised by Q6_K.
pub fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [i8],
    rmse_type: i32,
    qw: Option<&[f32]>,
) -> f32 {
    debug_assert_eq!(x.len(), n);
    debug_assert!(l.len() >= n);

    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &v in x.iter().take(n) {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        for slot in l.iter_mut().take(n) {
            *slot = 0;
        }
        return 0.0;
    }

    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let lq = nearest_int(iscale * x[i]);
            l[i] = (nmax + lq.max(-nmax).min(nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }

    let mut return_early = false;
    let mut rmse = rmse_type;
    if rmse < 0 {
        rmse = -rmse;
        return_early = true;
    }

    let weight_at = |i: usize| -> f32 {
        if let Some(qw) = qw {
            qw[i]
        } else if rmse == 1 {
            x[i] * x[i]
        } else if rmse == 2 {
            1.0
        } else if rmse == 3 {
            x[i].abs()
        } else {
            x[i].abs().sqrt()
        }
    };

    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let lq = nearest_int(iscale * x[i]).max(-nmax).min(nmax - 1);
        l[i] = (lq + nmax) as i8;
        let w = weight_at(i);
        sumlx += w * x[i] * (lq as f32);
        suml2 += w * (lq as f32) * (lq as f32);
    }
    let mut scale = if suml2 > 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9i32..=9 {
        if is == 0 {
            continue;
        }
        iscale = -((nmax as f32) + 0.1 * (is as f32)) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let lq = nearest_int(iscale * x[i]).max(-nmax).min(nmax - 1);
            let w = weight_at(i);
            sumlx += w * x[i] * (lq as f32);
            suml2 += w * (lq as f32) * (lq as f32);
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let lq = nearest_int(iscale * x[i]).max(-nmax).min(nmax - 1);
                l[i] = (lq + nmax) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// Quantize `row` (length `n_blocks × QK_K`) to a sequence of
/// `BlockQ6K` super-blocks. Pure-Rust port of `quantize_row_q6_K_ref`
/// (`ggml-quants.c:1807`).
///
/// Per super-block (16 sub-blocks of 16 elements each, **not** 8×32):
/// 1. Run [`make_qx_quants`] per sub-block with `nmax=32, rmse_type=1`
///    to get F32 `scales[16]`. Track the `max_scale` (signed) at the
///    `argmax_abs(scale)` over sub-blocks.
/// 2. If `|max_scale| < GROUP_MAX_EPS`: emit a zeroed block (matches
///    the C `memset` early-return).
/// 3. Else: super-block scale `iscale = -128 / max_scale`. Per
///    sub-block, store `scales[ib] = MIN(127, round(iscale * scales[ib]))`.
///    Note the i8 scales accept negative values — Q6_K is the only
///    k-quant with **signed** sub-block scales.
/// 4. Re-derive `L[256]` per sub-block from `(d, scale[ib])`:
///    `l = round(x / d).clamp(-32, 31); L = l + 32`.
/// 5. Pack: `ql[128]` holds the lower 4 bits, `qh[64]` the upper
///    2 bits. The pack groups four 32-element strides per
///    128-element half: `q1, q2, q3, q4` from `L[l], L[l+32],
///    L[l+64], L[l+96]`. Lower nibbles pack `(q1, q3) → ql[l]` and
///    `(q2, q4) → ql[l+32]`. Upper-2-bits pack `((L>>4) & 3)` for
///    each of the four strides into `qh[l]` shifted by `(0, 2, 4, 6)`.
pub fn quantize_row_q6_k(row: &[f32], blocks: &mut [BlockQ6K]) -> Result<(), KQuantError> {
    if !row.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { actual: row.len() });
    }
    let nb = row.len() / QK_K;
    if blocks.len() < nb {
        return Err(KQuantError::BlockSizeMismatch {
            actual: blocks.len(),
            n_blocks: nb,
            bytes_per_block: BLOCK_Q6_K_SIZE,
        });
    }

    let mut l_buf = [0i8; QK_K];
    let mut scales = [0.0f32; QK_K / 16];

    for i in 0..nb {
        let x = &row[i * QK_K..(i + 1) * QK_K];

        // Per-sub-block: run symmetric quantizer at nmax=32, rmse_type=1.
        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;
        for ib in 0..(QK_K / 16) {
            let xib = &x[16 * ib..16 * ib + 16];
            let scale = make_qx_quants(16, 32, xib, &mut l_buf[16 * ib..16 * ib + 16], 1, None);
            scales[ib] = scale;
            let abs_scale = scale.abs();
            if abs_scale > max_abs_scale {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        // Early-return zero block.
        if max_abs_scale < GROUP_MAX_EPS {
            blocks[i] = BlockQ6K {
                ql: [0u8; QK_K / 2],
                qh: [0u8; QK_K / 4],
                scales: [0i8; QK_K / 16],
                d_bits: half::f16::from_f32(0.0).to_bits(),
            };
            continue;
        }

        let iscale = -128.0 / max_scale;
        let d_f = 1.0 / iscale;
        let d_bits = half::f16::from_f32(d_f).to_bits();
        let d = half::f16::from_bits(d_bits).to_f32();

        let mut block_scales = [0i8; QK_K / 16];
        for ib in 0..(QK_K / 16) {
            // i8 range: nearest_int output then min(127, x); the i8 cast
            // saturates negative side naturally since scales are typically negative
            // (max_scale < 0 case → iscale > 0 → most products go positive;
            //  max_scale > 0 case → iscale < 0 → products go negative).
            let lq = nearest_int(iscale * scales[ib]).min(127);
            block_scales[ib] = lq as i8;
        }

        // Re-derive 6-bit quants from final (d, scales[ib]) per sub-block.
        for j in 0..(QK_K / 16) {
            let d_eff = d * (block_scales[j] as f32);
            if d_eff == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let lq = nearest_int(x[16 * j + ii] / d_eff).max(-32).min(31);
                l_buf[16 * j + ii] = (lq + 32) as i8;
            }
        }

        // Pack 6-bit quants: ql holds low 4 bits, qh holds upper 2 bits.
        let mut ql = [0u8; QK_K / 2];
        let mut qh = [0u8; QK_K / 4];
        for half in 0..(QK_K / 128) {
            let l_off = half * 128;
            let ql_off = half * 64;
            let qh_off = half * 32;
            for l in 0..32 {
                let q1 = (l_buf[l_off + l] as u8) & 0x0F;
                let q2 = (l_buf[l_off + l + 32] as u8) & 0x0F;
                let q3 = (l_buf[l_off + l + 64] as u8) & 0x0F;
                let q4 = (l_buf[l_off + l + 96] as u8) & 0x0F;
                ql[ql_off + l] = q1 | (q3 << 4);
                ql[ql_off + l + 32] = q2 | (q4 << 4);
                qh[qh_off + l] = ((l_buf[l_off + l] as u8) >> 4)
                    | (((l_buf[l_off + l + 32] as u8) >> 4) << 2)
                    | (((l_buf[l_off + l + 64] as u8) >> 4) << 4)
                    | (((l_buf[l_off + l + 96] as u8) >> 4) << 6);
            }
        }

        blocks[i] = BlockQ6K {
            ql,
            qh,
            scales: block_scales,
            d_bits,
        };
    }
    Ok(())
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

    // ─────────────────── Q5_K tests ───────────────────

    /// Block size matches llama.cpp's static_assert
    /// `sizeof(block_q5_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8`.
    #[test]
    fn block_q5_k_size_matches_c_struct() {
        assert_eq!(BLOCK_Q5_K_SIZE, 4 + 12 + 32 + 128);
        assert_eq!(BLOCK_Q5_K_SIZE, 176);
    }

    /// Round-trip BlockQ5K via to_bytes / from_bytes preserves the
    /// struct byte-for-byte.
    #[test]
    fn block_q5_k_byte_round_trip() {
        let block = BlockQ5K {
            d_bits: 0xFEDC,
            dmin_bits: 0x4321,
            scales: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8],
            qh: {
                let mut a = [0u8; QK_K / 8];
                for (i, slot) in a.iter_mut().enumerate() {
                    *slot = ((i * 7 + 11) & 0xFF) as u8;
                }
                a
            },
            qs: {
                let mut a = [0u8; QK_K / 2];
                for (i, slot) in a.iter_mut().enumerate() {
                    *slot = ((i * 13 + 5) & 0xFF) as u8;
                }
                a
            },
        };
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), BLOCK_Q5_K_SIZE);
        let decoded = BlockQ5K::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.d_bits, block.d_bits);
        assert_eq!(decoded.dmin_bits, block.dmin_bits);
        assert_eq!(decoded.scales, block.scales);
        assert_eq!(decoded.qh, block.qh);
        assert_eq!(decoded.qs, block.qs);
    }

    /// All-zeros block (zero d, zero dmin, zero scales, zero qs/qh)
    /// produces all-zero output.
    #[test]
    fn dequantize_q5_k_zero_block() {
        let block = BlockQ5K {
            d_bits: half::f16::from_f32(0.0).to_bits(),
            dmin_bits: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; 12],
            qh: [0u8; QK_K / 8],
            qs: [0u8; QK_K / 2],
        };

        let mut out = vec![0.0_f32; QK_K];
        let n = dequantize_row_q5_k(std::slice::from_ref(&block), &mut out).unwrap();
        assert_eq!(n, QK_K);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "out[{i}] = {v}");
        }
    }

    /// Q5_K dequantize with high-bit set: sub-block 0 with sc=10, m=0,
    /// d=1, dmin=0. qs[l] low nibble = 0, qh[l] bit-0 = 1 for l < 32.
    /// Expected: out[l] = 10 * (0 + 16) - 0 = 160 for l in 0..32.
    #[test]
    fn dequantize_q5_k_high_bit_only() {
        let mut scales = [0u8; 12];
        scales[0] = 10; // sc for sub-block 0 (j=0 path)
        scales[4] = 0;  // m for sub-block 0

        let mut qh = [0u8; QK_K / 8];
        for slot in qh.iter_mut().take(32) {
            *slot = 0x01; // bit 0 set for first 32 qh bytes
        }

        let block = BlockQ5K {
            d_bits: half::f16::from_f32(1.0).to_bits(),
            dmin_bits: half::f16::from_f32(0.0).to_bits(),
            scales,
            qh,
            qs: [0u8; QK_K / 2],
        };

        let mut out = vec![0.0_f32; QK_K];
        dequantize_row_q5_k(std::slice::from_ref(&block), &mut out).unwrap();

        // First 32 elements: 10 * 16 = 160.
        for l in 0..32 {
            assert!(
                (out[l] - 160.0).abs() < 1e-5,
                "out[{l}] = {} (want 160)",
                out[l]
            );
        }
        // Next 32 (high nibble) — qh bit 1 not set → q_high = 0; sc=0 (sub-block 1) → 0.
        for l in 32..64 {
            assert_eq!(out[l], 0.0, "out[{l}] = {}", out[l]);
        }
    }

    /// `dequantize_row_q5_k_bytes` rejects misaligned input length.
    #[test]
    fn dequantize_q5_k_bytes_rejects_misaligned() {
        let bad = vec![0u8; 200]; // not a multiple of 176
        let mut out = vec![0.0_f32; QK_K];
        let err = dequantize_row_q5_k_bytes(&bad, &mut out).unwrap_err();
        match err {
            KQuantError::BlockSizeMismatch {
                bytes_per_block, ..
            } => {
                assert_eq!(bytes_per_block, BLOCK_Q5_K_SIZE);
            }
            _ => panic!("expected BlockSizeMismatch"),
        }
    }

    // ─────────────────── Q6_K tests ───────────────────

    /// Block size matches llama.cpp's static_assert
    /// `sizeof(block_q6_K) == sizeof(ggml_half) + QK_K/16 + 3*QK_K/4`.
    #[test]
    fn block_q6_k_size_matches_c_struct() {
        assert_eq!(BLOCK_Q6_K_SIZE, 128 + 64 + 16 + 2);
        assert_eq!(BLOCK_Q6_K_SIZE, 210);
    }

    /// Round-trip BlockQ6K via to_bytes / from_bytes preserves the
    /// struct byte-for-byte. Includes negative scales (i8) to verify
    /// sign preservation.
    #[test]
    fn block_q6_k_byte_round_trip() {
        let block = BlockQ6K {
            ql: {
                let mut a = [0u8; QK_K / 2];
                for (i, slot) in a.iter_mut().enumerate() {
                    *slot = ((i * 17 + 3) & 0xFF) as u8;
                }
                a
            },
            qh: {
                let mut a = [0u8; QK_K / 4];
                for (i, slot) in a.iter_mut().enumerate() {
                    *slot = ((i * 5 + 9) & 0xFF) as u8;
                }
                a
            },
            scales: [
                -32, -16, -1, 0, 1, 16, 31, 64, -127, 127, -64, 32, -8, 8, 0, -100,
            ],
            d_bits: 0xBEEF,
        };
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), BLOCK_Q6_K_SIZE);
        let decoded = BlockQ6K::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.ql, block.ql);
        assert_eq!(decoded.qh, block.qh);
        assert_eq!(decoded.scales, block.scales);
        assert_eq!(decoded.d_bits, block.d_bits);
    }

    /// All-zero ql + qh + scales=0 + d=0 → all-zero output.
    #[test]
    fn dequantize_q6_k_zero_block() {
        let block = BlockQ6K {
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [0i8; QK_K / 16],
            d_bits: half::f16::from_f32(0.0).to_bits(),
        };

        let mut out = vec![0.0_f32; QK_K];
        let n = dequantize_row_q6_k(std::slice::from_ref(&block), &mut out).unwrap();
        assert_eq!(n, QK_K);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    /// Q6_K with `d=1`, `ql[l] & 0xF = 0`, `qh = 0` produces
    /// `q1 = 0 - 32 = -32`, so output lanes get `1 * scales[is+k] * -32`.
    /// Per the reference (`ggml-quants.c:1895-1898`):
    ///   `y[l+0]`  = d * scales[is+0] * q1
    ///   `y[l+32]` = d * scales[is+2] * q2
    ///   `y[l+64]` = d * scales[is+4] * q3
    ///   `y[l+96]` = d * scales[is+6] * q4
    ///
    /// Set `scales[0] = 1` and `scales[4] = 2` (both use `is=0` for
    /// l < 16). Verify `out[0..16] = -32` and `out[64..80] = -64`.
    #[test]
    fn dequantize_q6_k_min_quant_baseline() {
        let mut scales = [0i8; QK_K / 16];
        scales[0] = 1;  // out[l + 0]  uses scales[is+0] = scales[0]
        scales[4] = 2;  // out[l + 64] uses scales[is+4] = scales[4]
        // ql=0, qh=0 → q1=q2=q3=q4 = 0 - 32 = -32
        let block = BlockQ6K {
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales,
            d_bits: half::f16::from_f32(1.0).to_bits(),
        };
        let mut out = vec![0.0_f32; QK_K];
        dequantize_row_q6_k(std::slice::from_ref(&block), &mut out).unwrap();

        // out[0..16] = 1 * 1 * -32 = -32.
        for l in 0..16 {
            assert_eq!(out[l], -32.0, "out[{l}] = {}", out[l]);
        }
        // out[64..80] = 1 * 2 * -32 = -64.
        for l in 0..16 {
            assert_eq!(out[64 + l], -64.0, "out[{}]={}", 64 + l, out[64 + l]);
        }
    }

    /// `dequantize_row_q6_k_bytes` decodes a multi-block buffer.
    #[test]
    fn dequantize_q6_k_bytes_multi_block() {
        let block = BlockQ6K {
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [0i8; QK_K / 16],
            d_bits: 0,
        };
        let mut buf = Vec::with_capacity(2 * BLOCK_Q6_K_SIZE);
        for _ in 0..2 {
            buf.extend_from_slice(&block.to_bytes());
        }
        assert_eq!(buf.len(), 2 * BLOCK_Q6_K_SIZE);
        let mut out = vec![0.0_f32; 2 * QK_K];
        let n = dequantize_row_q6_k_bytes(&buf, &mut out).unwrap();
        assert_eq!(n, 2 * QK_K);
        // d=0 → all output = 0.
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    /// `dequantize_row_q6_k_bytes` rejects misaligned input.
    #[test]
    fn dequantize_q6_k_bytes_rejects_misaligned() {
        let bad = vec![0u8; 300]; // 300 % 210 != 0
        let mut out = vec![0.0_f32; QK_K];
        let err = dequantize_row_q6_k_bytes(&bad, &mut out).unwrap_err();
        match err {
            KQuantError::BlockSizeMismatch {
                bytes_per_block, ..
            } => {
                assert_eq!(bytes_per_block, BLOCK_Q6_K_SIZE);
            }
            _ => panic!("expected BlockSizeMismatch"),
        }
    }

    // ─────────────────── nearest_int + make_qkx2_quants tests ───────────────────

    /// `nearest_int` round-to-nearest-even: 0.5 → 0, 1.5 → 2, 2.5 → 2,
    /// 3.5 → 4 (banker's rounding); negative values: -0.5 → 0, -1.5 → -2.
    /// Matches IEEE 754 round-half-to-even. Verified for byte-identity
    /// against llama.cpp's bit-trick implementation.
    #[test]
    fn nearest_int_round_to_even() {
        assert_eq!(nearest_int(0.0), 0);
        assert_eq!(nearest_int(0.49), 0);
        assert_eq!(nearest_int(0.5), 0); // half-to-even
        assert_eq!(nearest_int(0.51), 1);
        assert_eq!(nearest_int(1.5), 2); // half-to-even
        assert_eq!(nearest_int(2.5), 2); // half-to-even
        assert_eq!(nearest_int(3.5), 4); // half-to-even
        assert_eq!(nearest_int(-0.5), 0); // half-to-even
        assert_eq!(nearest_int(-1.5), -2);
        assert_eq!(nearest_int(-2.5), -2);
        assert_eq!(nearest_int(7.0), 7);
        assert_eq!(nearest_int(-7.0), -7);
        assert_eq!(nearest_int(15.0), 15);
        assert_eq!(nearest_int(15.999), 16);
    }

    /// `make_qkx2_quants` on all-zero input returns scale=0, mins=0,
    /// and L[i]=0 for all i — matching the `max == min` early return.
    #[test]
    fn make_qkx2_quants_all_zero_input() {
        let n = 32;
        let x = vec![0.0_f32; n];
        let weights = vec![1.0_f32; n];
        let mut l = vec![0u8; n];
        let mut laux = vec![0u8; n];
        let (scale, the_min) =
            make_qkx2_quants(n, 15, &x, &weights, &mut l, &mut laux, -1.0, 0.1, 20, false);
        assert_eq!(scale, 0.0);
        assert_eq!(the_min, 0.0);
        for &q in l.iter() {
            assert_eq!(q, 0);
        }
    }

    /// `make_qkx2_quants` on a constant non-zero input: max == min after
    /// the `min = 0` clamp doesn't apply (since min < 0 isn't true), so
    /// the early-return path triggers when max == min is true. With
    /// constant `x = c > 0`, we get min=0 (clamped from c > 0 → min=c
    /// clamped to 0; wait, min stays at c since we only clamp `if min > 0`,
    /// then min=0). Actually re-reading: if min > 0 then min = 0. So with
    /// all values equal to 1.0: initial min=1.0, then min = 0 (clamped),
    /// max stays 1.0. max != min, so search proceeds.
    /// Just verify it returns a finite scale + L values in [0, 15].
    #[test]
    fn make_qkx2_quants_constant_positive_input() {
        let n = 32;
        let x = vec![1.0_f32; n];
        let weights = vec![1.0_f32; n];
        let mut l = vec![0u8; n];
        let mut laux = vec![0u8; n];
        let (scale, _the_min) =
            make_qkx2_quants(n, 15, &x, &weights, &mut l, &mut laux, -1.0, 0.1, 20, false);
        assert!(scale.is_finite(), "scale not finite: {scale}");
        for &q in l.iter() {
            assert!(q <= 15, "L[i] out of [0,15]: {q}");
        }
    }

    /// `make_qkx2_quants` on a deterministic ramp. Verifies the algorithm
    /// produces a near-monotonic L sequence.
    #[test]
    fn make_qkx2_quants_monotonic_ramp() {
        let n = 32;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let weights = vec![1.0_f32; n];
        let mut l = vec![0u8; n];
        let mut laux = vec![0u8; n];
        let (scale, the_min) =
            make_qkx2_quants(n, 15, &x, &weights, &mut l, &mut laux, -1.0, 0.1, 20, false);
        assert!(scale > 0.0, "scale must be positive for monotonic ramp");
        // Expect L[i] non-decreasing for monotonic input.
        for i in 1..n {
            assert!(
                l[i] >= l[i - 1],
                "expected non-decreasing L, got L[{}]={} < L[{}]={}",
                i,
                l[i],
                i - 1,
                l[i - 1]
            );
        }
        // L[0] should be 0 or 1 (smallest input near codebook bottom).
        assert!(l[0] <= 1, "L[0] = {} (expected 0 or 1 for ramp[0]=0)", l[0]);
        // L[n-1] should be near nmax (largest input maps near top).
        assert!(
            l[n - 1] >= 14,
            "L[n-1] = {} (expected ≥14 for ramp's max)",
            l[n - 1]
        );
        // the_min is the negated min from the search (post-refinement).
        // The search may pick a slightly negative `min` to balance
        // round-off across all elements, yielding `the_min ≥ 0`.
        assert!(
            the_min >= 0.0 && the_min.is_finite(),
            "the_min = {the_min} (expected non-negative finite)"
        );
    }

    /// `make_qkx2_quants` with `nstep < 1` skips refinement and returns
    /// the initial scale.
    #[test]
    fn make_qkx2_quants_no_refinement() {
        let n = 32;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let weights = vec![1.0_f32; n];
        let mut l = vec![0u8; n];
        let mut laux = vec![0u8; n];
        let (scale_with_refine, _) = make_qkx2_quants(
            n,
            15,
            &x,
            &weights,
            &mut l.clone(),
            &mut laux.clone(),
            -1.0,
            0.1,
            20,
            false,
        );
        let (scale_no_refine, _) =
            make_qkx2_quants(n, 15, &x, &weights, &mut l, &mut laux, -1.0, 0.1, 0, false);
        // Both finite; different (refinement may shift scale).
        assert!(scale_with_refine.is_finite());
        assert!(scale_no_refine.is_finite());
        // The no-refinement scale = (max - min) / nmax = 31 / 15 ≈ 2.0667.
        assert!(
            (scale_no_refine - 31.0 / 15.0).abs() < 1e-4,
            "no-refine scale {scale_no_refine} != 31/15"
        );
    }

    // ─────────────────── quantize_row_q4_k tests ───────────────────

    /// All-zero input quantized to a Q4_K block: every output byte = 0,
    /// d/dmin = 0. Round-trip dequantize must produce all zeros.
    #[test]
    fn quantize_q4_k_zero_input() {
        let row = vec![0.0_f32; QK_K];
        let mut blocks = vec![
            BlockQ4K {
                d_bits: 0xFFFF, // pre-fill to confirm overwrite
                dmin_bits: 0xFFFF,
                scales: [0xFFu8; 12],
                qs: [0xFFu8; QK_K / 2],
            };
            1
        ];
        quantize_row_q4_k(&row, &mut blocks).unwrap();
        // d, dmin should be 0.
        assert_eq!(blocks[0].d_bits, 0); // F16(0.0)
        assert_eq!(blocks[0].dmin_bits, 0);
        // scales packed all zeros.
        assert_eq!(blocks[0].scales, [0u8; 12]);
        // qs all zeros.
        for &b in blocks[0].qs.iter() {
            assert_eq!(b, 0);
        }
        // Round-trip dequantize → all zeros.
        let mut decoded = vec![0.0_f32; QK_K];
        dequantize_row_q4_k(&blocks, &mut decoded).unwrap();
        for &v in decoded.iter() {
            assert_eq!(v, 0.0);
        }
    }

    /// Quantize a deterministic synthetic row, dequantize, verify the
    /// round-trip MSE is within a reasonable Q4_K tolerance. This is
    /// the algorithmic correctness gate (separate from byte-identity
    /// gate against llama.cpp NEON which lands at iter-3b2).
    #[test]
    fn quantize_q4_k_round_trip_synthetic() {
        // Synthetic row: gentle ramp across multiple sub-blocks for
        // coverage of all 8 sub-blocks. Range [-2, 2].
        let row: Vec<f32> = (0..QK_K)
            .map(|i| {
                let t = (i as f32) / (QK_K as f32 - 1.0); // 0..1
                -2.0 + 4.0 * t // -2..2
            })
            .collect();

        let mut blocks = vec![
            BlockQ4K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qs: [0u8; QK_K / 2],
            };
            1
        ];
        quantize_row_q4_k(&row, &mut blocks).unwrap();

        let mut decoded = vec![0.0_f32; QK_K];
        dequantize_row_q4_k(&blocks, &mut decoded).unwrap();

        // RMSE bound for Q4_K on a smooth ramp: empirically << 0.05 since
        // 4.5 bpw covers ~32 levels per super-block with min subtraction.
        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(
            rmse < 0.05,
            "Q4_K round-trip RMSE {rmse} exceeds 0.05 threshold for smooth ramp"
        );
    }

    /// Quantize with all-equal positive input. Q4_K should pack a
    /// near-constant decoded value.
    #[test]
    fn quantize_q4_k_constant_positive() {
        let row = vec![0.5_f32; QK_K];
        let mut blocks = vec![
            BlockQ4K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qs: [0u8; QK_K / 2],
            };
            1
        ];
        quantize_row_q4_k(&row, &mut blocks).unwrap();
        let mut decoded = vec![0.0_f32; QK_K];
        dequantize_row_q4_k(&blocks, &mut decoded).unwrap();

        // Each decoded element should be very close to 0.5.
        for (i, &v) in decoded.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 0.05,
                "constant-input element [{i}] = {v} (expected ~0.5)"
            );
        }
    }

    /// `quantize_row_q4_k` rejects misaligned input lengths.
    #[test]
    fn quantize_q4_k_rejects_misaligned_length() {
        let row = vec![0.0_f32; QK_K + 7]; // not a multiple of QK_K
        let mut blocks = vec![
            BlockQ4K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qs: [0u8; QK_K / 2],
            };
            2
        ];
        let err = quantize_row_q4_k(&row, &mut blocks).unwrap_err();
        match err {
            KQuantError::NotBlockAligned { actual } => {
                assert_eq!(actual, QK_K + 7);
            }
            _ => panic!("expected NotBlockAligned"),
        }
    }

    /// `quantize_row_q4_k` rejects too-small output buffer.
    #[test]
    fn quantize_q4_k_rejects_short_blocks_buffer() {
        let row = vec![0.0_f32; 2 * QK_K]; // need 2 blocks
        let mut blocks = vec![
            BlockQ4K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qs: [0u8; QK_K / 2],
            };
            1
        ]; // only 1 slot
        let err = quantize_row_q4_k(&row, &mut blocks).unwrap_err();
        match err {
            KQuantError::BlockSizeMismatch { n_blocks, .. } => {
                assert_eq!(n_blocks, 2);
            }
            _ => panic!("expected BlockSizeMismatch"),
        }
    }

    // ─────────────────── quantize_row_q5_k tests ───────────────────

    /// Q5_K all-zero input → blocks all zero; dequantize back to zeros.
    #[test]
    fn quantize_q5_k_zero_input() {
        let row = vec![0.0_f32; QK_K];
        let mut blocks = vec![
            BlockQ5K {
                d_bits: 0xFFFF,
                dmin_bits: 0xFFFF,
                scales: [0xFFu8; 12],
                qh: [0xFFu8; QK_K / 8],
                qs: [0xFFu8; QK_K / 2],
            };
            1
        ];
        quantize_row_q5_k(&row, &mut blocks).unwrap();
        assert_eq!(blocks[0].d_bits, 0);
        assert_eq!(blocks[0].dmin_bits, 0);
        assert_eq!(blocks[0].scales, [0u8; 12]);
        // qh re-zeroed even though we pre-filled.
        for &b in blocks[0].qh.iter() {
            assert_eq!(b, 0);
        }
        for &b in blocks[0].qs.iter() {
            assert_eq!(b, 0);
        }
        let mut decoded = vec![0.0_f32; QK_K];
        dequantize_row_q5_k(&blocks, &mut decoded).unwrap();
        for &v in decoded.iter() {
            assert_eq!(v, 0.0);
        }
    }

    /// Q5_K round-trip RMSE bound on a smooth ramp [-2, 2]. Q5_K's
    /// 5.5-bpw codebook should yield RMSE ≤ 0.025 — about half of
    /// the Q4_K bound, because we double the codebook density.
    #[test]
    fn quantize_q5_k_round_trip_synthetic() {
        let row: Vec<f32> = (0..QK_K)
            .map(|i| {
                let t = (i as f32) / (QK_K as f32 - 1.0);
                -2.0 + 4.0 * t
            })
            .collect();

        let mut blocks = vec![
            BlockQ5K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qh: [0u8; QK_K / 8],
                qs: [0u8; QK_K / 2],
            };
            1
        ];
        quantize_row_q5_k(&row, &mut blocks).unwrap();

        let mut decoded = vec![0.0_f32; QK_K];
        dequantize_row_q5_k(&blocks, &mut decoded).unwrap();

        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(
            rmse < 0.025,
            "Q5_K round-trip RMSE {rmse} > 0.025 threshold for smooth ramp"
        );
    }

    /// Q5_K vs Q4_K: on the same input, Q5_K must produce strictly
    /// lower RMSE than Q4_K (more bits → lower error). Verifies the
    /// 5-bit codebook isn't accidentally degraded.
    #[test]
    fn quantize_q5_k_lower_rmse_than_q4_k() {
        let row: Vec<f32> = (0..QK_K)
            .map(|i| {
                let t = (i as f32) / (QK_K as f32 - 1.0);
                -3.0 + 6.0 * t
            })
            .collect();

        let mut q4 = vec![
            BlockQ4K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qs: [0u8; QK_K / 2],
            };
            1
        ];
        let mut q5 = vec![
            BlockQ5K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qh: [0u8; QK_K / 8],
                qs: [0u8; QK_K / 2],
            };
            1
        ];
        quantize_row_q4_k(&row, &mut q4).unwrap();
        quantize_row_q5_k(&row, &mut q5).unwrap();

        let mut q4_decoded = vec![0.0_f32; QK_K];
        let mut q5_decoded = vec![0.0_f32; QK_K];
        dequantize_row_q4_k(&q4, &mut q4_decoded).unwrap();
        dequantize_row_q5_k(&q5, &mut q5_decoded).unwrap();

        let rmse = |a: &[f32], b: &[f32]| -> f64 {
            let mut sse = 0.0;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = (*x as f64) - (*y as f64);
                sse += d * d;
            }
            (sse / a.len() as f64).sqrt()
        };
        let r4 = rmse(&row, &q4_decoded);
        let r5 = rmse(&row, &q5_decoded);
        assert!(
            r5 < r4,
            "Q5_K RMSE {r5} should be < Q4_K RMSE {r4} (more bits → less error)"
        );
    }

    /// Q5_K with input that needs the high bit to be set (values that
    /// quantize above 15). Verifies the qh[] packing actually fires.
    #[test]
    fn quantize_q5_k_high_bit_packed() {
        // Use a wide-range input so quants exceed 15 in some sub-blocks.
        let row: Vec<f32> = (0..QK_K)
            .map(|i| if i < QK_K / 2 { 0.0 } else { 10.0 })
            .collect();

        let mut blocks = vec![
            BlockQ5K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qh: [0u8; QK_K / 8],
                qs: [0u8; QK_K / 2],
            };
            1
        ];
        quantize_row_q5_k(&row, &mut blocks).unwrap();

        // qh must have at least one bit set (high-bit pack triggered).
        let any_qh_set = blocks[0].qh.iter().any(|&b| b != 0);
        assert!(
            any_qh_set,
            "qh has no bits set — Q5_K high-bit pack missing"
        );
    }

    // ─────────────────── make_qx_quants tests ───────────────────

    /// `make_qx_quants` on all-zero input early-returns scale=0, L=[0; n].
    #[test]
    fn make_qx_quants_all_zero_input() {
        let n = 16;
        let x = vec![0.0_f32; n];
        let mut l = vec![0i8; n];
        let scale = make_qx_quants(n, 32, &x, &mut l, 1, None);
        assert_eq!(scale, 0.0);
        for &q in l.iter() {
            assert_eq!(q, 0);
        }
    }

    /// `make_qx_quants` on a symmetric ramp [-1, 1] with rmse_type=1
    /// returns a finite scale and L values in [0, 2*nmax-1].
    #[test]
    fn make_qx_quants_symmetric_ramp() {
        let n = 16;
        let x: Vec<f32> = (0..n)
            .map(|i| -1.0 + 2.0 * (i as f32) / ((n - 1) as f32))
            .collect();
        let mut l = vec![0i8; n];
        let scale = make_qx_quants(n, 32, &x, &mut l, 1, None);
        assert!(scale.is_finite());
        // L stored at offset +32, so L ∈ [0, 63]. Verify all in range.
        for &q in l.iter() {
            assert!(q >= 0, "L[i] negative: {q}");
            assert!(q <= 63, "L[i] > 2*nmax-1: {q}");
        }
    }

    /// `make_qx_quants` with `rmse_type == 0` skips refinement and
    /// returns the initial scale `1/iscale = -max/nmax`.
    #[test]
    fn make_qx_quants_no_refinement() {
        let n = 16;
        // Single non-zero value → max = 1.0; iscale = -32/1 = -32; scale = -1/32.
        let mut x = vec![0.0_f32; n];
        x[5] = 1.0;
        let mut l = vec![0i8; n];
        let scale = make_qx_quants(n, 32, &x, &mut l, 0, None);
        assert!(
            (scale - (-1.0 / 32.0)).abs() < 1e-5,
            "scale = {scale}, expected -1/32"
        );
    }

    // ─────────────────── quantize_row_q6_k tests ───────────────────

    /// Q6_K all-zero input → blocks all zero.
    #[test]
    fn quantize_q6_k_zero_input() {
        let row = vec![0.0_f32; QK_K];
        let mut blocks = vec![
            BlockQ6K {
                ql: [0xFFu8; QK_K / 2],
                qh: [0xFFu8; QK_K / 4],
                scales: [127i8; QK_K / 16],
                d_bits: 0xFFFF,
            };
            1
        ];
        quantize_row_q6_k(&row, &mut blocks).unwrap();
        assert_eq!(blocks[0].d_bits, 0);
        for &s in blocks[0].scales.iter() {
            assert_eq!(s, 0);
        }
        for &b in blocks[0].ql.iter() {
            assert_eq!(b, 0);
        }
        for &b in blocks[0].qh.iter() {
            assert_eq!(b, 0);
        }

        let mut decoded = vec![0.0_f32; QK_K];
        dequantize_row_q6_k(&blocks, &mut decoded).unwrap();
        for &v in decoded.iter() {
            assert_eq!(v, 0.0);
        }
    }

    /// Q6_K round-trip RMSE bound on a smooth ramp [-2, 2]. Q6_K's
    /// 6.5625-bpw codebook should yield RMSE ≤ 0.012 — about half
    /// of Q5_K's bound.
    #[test]
    fn quantize_q6_k_round_trip_synthetic() {
        let row: Vec<f32> = (0..QK_K)
            .map(|i| {
                let t = (i as f32) / (QK_K as f32 - 1.0);
                -2.0 + 4.0 * t
            })
            .collect();

        let mut blocks = vec![
            BlockQ6K {
                ql: [0u8; QK_K / 2],
                qh: [0u8; QK_K / 4],
                scales: [0i8; QK_K / 16],
                d_bits: 0,
            };
            1
        ];
        quantize_row_q6_k(&row, &mut blocks).unwrap();

        let mut decoded = vec![0.0_f32; QK_K];
        dequantize_row_q6_k(&blocks, &mut decoded).unwrap();

        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(
            rmse < 0.012,
            "Q6_K round-trip RMSE {rmse} > 0.012 threshold for smooth ramp"
        );
    }

    /// Q6_K vs Q5_K vs Q4_K on **zero-mean symmetric** input: each
    /// should produce strictly lower RMSE than the previous. Note:
    /// for shifted-mean per-sub-block distributions (like a linear
    /// ramp), Q6_K's symmetric codebook can waste precision and lose
    /// to Q5_K_M's min-term advantage. A sine wave keeps each
    /// 16-element sub-block roughly zero-mean, which is where Q6_K's
    /// extra bit pays off.
    #[test]
    fn quantize_q6_k_lowest_rmse() {
        // Sine wave amplitude 2 over QK_K elements covering 4 cycles.
        // Each 16-element sub-block sees ~half a cycle → near-zero mean.
        let row: Vec<f32> = (0..QK_K)
            .map(|i| {
                let theta = 8.0 * std::f32::consts::PI * (i as f32) / (QK_K as f32);
                2.0 * theta.sin()
            })
            .collect();

        let mut q4 = vec![
            BlockQ4K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qs: [0u8; QK_K / 2],
            };
            1
        ];
        let mut q5 = vec![
            BlockQ5K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qh: [0u8; QK_K / 8],
                qs: [0u8; QK_K / 2],
            };
            1
        ];
        let mut q6 = vec![
            BlockQ6K {
                ql: [0u8; QK_K / 2],
                qh: [0u8; QK_K / 4],
                scales: [0i8; QK_K / 16],
                d_bits: 0,
            };
            1
        ];
        quantize_row_q4_k(&row, &mut q4).unwrap();
        quantize_row_q5_k(&row, &mut q5).unwrap();
        quantize_row_q6_k(&row, &mut q6).unwrap();

        let mut q4_decoded = vec![0.0_f32; QK_K];
        let mut q5_decoded = vec![0.0_f32; QK_K];
        let mut q6_decoded = vec![0.0_f32; QK_K];
        dequantize_row_q4_k(&q4, &mut q4_decoded).unwrap();
        dequantize_row_q5_k(&q5, &mut q5_decoded).unwrap();
        dequantize_row_q6_k(&q6, &mut q6_decoded).unwrap();

        let rmse = |a: &[f32], b: &[f32]| -> f64 {
            let mut sse = 0.0;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = (*x as f64) - (*y as f64);
                sse += d * d;
            }
            (sse / a.len() as f64).sqrt()
        };
        let r4 = rmse(&row, &q4_decoded);
        let r5 = rmse(&row, &q5_decoded);
        let r6 = rmse(&row, &q6_decoded);
        assert!(r5 < r4, "Q5_K RMSE {r5} should be < Q4_K RMSE {r4}");
        assert!(r6 < r5, "Q6_K RMSE {r6} should be < Q5_K RMSE {r5}");
    }

    /// Q6_K rejects misaligned input.
    #[test]
    fn quantize_q6_k_rejects_misaligned() {
        let row = vec![0.0_f32; QK_K + 1];
        let mut blocks = vec![
            BlockQ6K {
                ql: [0u8; QK_K / 2],
                qh: [0u8; QK_K / 4],
                scales: [0i8; QK_K / 16],
                d_bits: 0,
            };
            1
        ];
        let err = quantize_row_q6_k(&row, &mut blocks).unwrap_err();
        match err {
            KQuantError::NotBlockAligned { actual } => {
                assert_eq!(actual, QK_K + 1);
            }
            _ => panic!("expected NotBlockAligned"),
        }
    }

    /// `quantize_row_q5_k` rejects misaligned input.
    #[test]
    fn quantize_q5_k_rejects_misaligned() {
        let row = vec![0.0_f32; QK_K + 3];
        let mut blocks = vec![
            BlockQ5K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qh: [0u8; QK_K / 8],
                qs: [0u8; QK_K / 2],
            };
            1
        ];
        let err = quantize_row_q5_k(&row, &mut blocks).unwrap_err();
        match err {
            KQuantError::NotBlockAligned { actual } => {
                assert_eq!(actual, QK_K + 3);
            }
            _ => panic!("expected NotBlockAligned"),
        }
    }

    /// Multi-block: quantize 3 super-blocks, dequantize, RMSE bound
    /// should still hold (verifies the per-block loop and accounting).
    #[test]
    fn quantize_q4_k_multi_block_round_trip() {
        let n_blocks = 3;
        let row: Vec<f32> = (0..n_blocks * QK_K)
            .map(|i| {
                let t = (i as f32) / ((n_blocks * QK_K) as f32 - 1.0);
                -1.5 + 3.0 * t
            })
            .collect();

        let mut blocks = vec![
            BlockQ4K {
                d_bits: 0,
                dmin_bits: 0,
                scales: [0u8; 12],
                qs: [0u8; QK_K / 2],
            };
            n_blocks
        ];
        quantize_row_q4_k(&row, &mut blocks).unwrap();

        let mut decoded = vec![0.0_f32; n_blocks * QK_K];
        dequantize_row_q4_k(&blocks, &mut decoded).unwrap();

        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(rmse < 0.05, "multi-block RMSE {rmse} > 0.05");
    }
}
