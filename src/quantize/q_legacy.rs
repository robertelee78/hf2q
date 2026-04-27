//! Legacy GGML block-32 formats — Q8_0, Q4_0, Q5_0, Q5_1, Q4_1
//! (ADR-014 P7 — fallback path for K-family tensors that aren't
//! 256-element-row-aligned).
//!
//! These formats predate the K-quant family and use 32-element blocks
//! with a single per-block F16 scale (and optionally a per-block F16
//! min for the `_1` variants). They're the GGUF fallback chain for
//! tensors whose innermost dimension isn't divisible by `QK_K = 256`:
//! ssm_conv1d (4-element conv kernel), some bias-like tensors, etc.
//!
//! Per `tensor_type_fallback` in `/opt/llama.cpp/src/llama-quant.cpp`
//! (mirrored at `src/backends/gguf.rs:353-358`):
//!
//! - `Q6_K` → `Q8_0`
//! - `Q5_K` → `Q5_1`
//! - `Q4_K` → `Q5_0`
//! - `Q3_K`/`Q2_K` → `Q4_0`
//!
//! ## This iter — Q8_0 only
//!
//! P7 iter-3h lands `Q8_0` in both directions (quantize + dequantize)
//! plus flat-bytes wrappers. Q4_0, Q5_0, Q5_1 follow in subsequent
//! iters (each is similar shape but with per-block scales packed
//! differently).
//!
//! ## Sovereignty
//!
//! Pure Rust. No `cc`/`cmake` link to libggml. References to
//! `/opt/llama.cpp/ggml/src/ggml-quants.c` are read-only.

use thiserror::Error;

/// Block size for Q8_0 (and Q4_0 / Q5_0 / Q5_1 / Q4_1). Mirrors
/// `QK8_0` / `QK4_0` etc in `ggml-common.h`.
pub const QK8_0: usize = 32;

/// Block size for Q4_0 (matches `QK8_0` — both are 32-element blocks).
pub const QK4_0: usize = 32;

/// Block size in bytes — F16 d (2) + 32 × i8 quants (32) = 34 bytes.
/// Matches `sizeof(block_q8_0)` per `ggml-common.h:246`.
pub const BLOCK_Q8_0_SIZE: usize = 2 + QK8_0;

/// Block size in bytes — F16 d (2) + 16 packed nibbles (`QK4_0/2`) = 18 bytes.
/// Matches `sizeof(block_q4_0)` per `ggml-common.h:189`.
pub const BLOCK_Q4_0_SIZE: usize = 2 + QK4_0 / 2;

/// Block size for Q4_1 (32-element block).
pub const QK4_1: usize = 32;

/// Block size in bytes — F16 d (2) + F16 m (2) + 16 packed nibbles
/// (`QK4_1/2`) = 20 bytes. Matches `sizeof(block_q4_1)` per
/// `ggml-common.h:202`.
pub const BLOCK_Q4_1_SIZE: usize = 2 + 2 + QK4_1 / 2;

/// Block size for Q5_0 / Q5_1 (32-element blocks).
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;

/// Block size in bytes — F16 d (2) + qh u32 (4) + 16 packed nibbles
/// (`QK5_0/2`) = 22 bytes. Matches `sizeof(block_q5_0)` per
/// `ggml-common.h:225`.
pub const BLOCK_Q5_0_SIZE: usize = 2 + 4 + QK5_0 / 2;

/// Block size in bytes — F16 d (2) + F16 m (2) + qh u32 (4) +
/// 16 packed nibbles (`QK5_1/2`) = 24 bytes. Matches `sizeof(block_q5_1)`
/// per `ggml-common.h:239`.
pub const BLOCK_Q5_1_SIZE: usize = 2 + 2 + 4 + QK5_1 / 2;

/// Errors from legacy block-32 operations.
#[derive(Error, Debug)]
pub enum QLegacyError {
    /// Input length isn't a multiple of the format's block size.
    #[error("q_legacy: input length {actual} is not a multiple of QK ({qk})")]
    NotBlockAligned { actual: usize, qk: usize },

    /// Output buffer is shorter than required.
    #[error(
        "q_legacy: output buffer length {actual} insufficient for {n_blocks} blocks of {bytes_per_block} bytes"
    )]
    BlockSizeMismatch {
        actual: usize,
        n_blocks: usize,
        bytes_per_block: usize,
    },
}

/// Q8_0 block — 32 elements stored at 8 bpw + F16 super-block scale.
///
/// Layout matches `block_q8_0` in `/opt/llama.cpp/ggml/src/ggml-common.h:243`
/// byte-for-byte.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    /// Block scale. F16.
    pub d_bits: u16,
    /// 32 signed-byte quants in `[-128, 127]`.
    pub qs: [i8; QK8_0],
}

impl BlockQ8_0 {
    /// Read a `BlockQ8_0` from a 34-byte buffer (little-endian F16
    /// scale + 32 i8 quants). Returns `None` if the buffer is shorter
    /// than `BLOCK_Q8_0_SIZE`.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < BLOCK_Q8_0_SIZE {
            return None;
        }
        let d_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut qs = [0i8; QK8_0];
        for (i, slot) in qs.iter_mut().enumerate() {
            *slot = bytes[2 + i] as i8;
        }
        Some(Self { d_bits, qs })
    }

    /// Serialise to 34-byte little-endian buffer (matches the C
    /// memory layout on `aarch64-apple-darwin`).
    pub fn to_bytes(&self) -> [u8; BLOCK_Q8_0_SIZE] {
        let mut out = [0u8; BLOCK_Q8_0_SIZE];
        out[0..2].copy_from_slice(&self.d_bits.to_le_bytes());
        for (i, &q) in self.qs.iter().enumerate() {
            out[2 + i] = q as u8;
        }
        out
    }

    /// Block scale `d` as F32.
    pub fn d(&self) -> f32 {
        half::f16::from_bits(self.d_bits).to_f32()
    }
}

/// Quantize a row of F32 to a sequence of `BlockQ8_0` blocks.
/// Pure-Rust port of `quantize_row_q8_0_ref` (`ggml-quants.c:234`).
///
/// Per-block algorithm (mirrors C reference):
/// 1. Compute `amax = max(|x[j]|)` over the 32-element block.
/// 2. Compute scale `d = amax / 127`.
/// 3. Convert to F16 (round-to-nearest-even via `half::f16`).
/// 4. Quantize each element as `q[j] = round(x[j] / d)` clamped to
///    `[-128, 127]`. (Note the inverse-scale `id = 1/d` is used in
///    the C code for performance; we use the equivalent direct
///    division — the rounding behavior matches because both go
///    through `roundf`/`f32::round`.)
///
/// `row.len()` must be a multiple of `QK8_0` (32). The output
/// `blocks` must have at least `row.len() / QK8_0` slots.
pub fn quantize_row_q8_0(row: &[f32], blocks: &mut [BlockQ8_0]) -> Result<(), QLegacyError> {
    if !row.len().is_multiple_of(QK8_0) {
        return Err(QLegacyError::NotBlockAligned {
            actual: row.len(),
            qk: QK8_0,
        });
    }
    let nb = row.len() / QK8_0;
    if blocks.len() < nb {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: blocks.len(),
            n_blocks: nb,
            bytes_per_block: BLOCK_Q8_0_SIZE,
        });
    }

    for i in 0..nb {
        let x = &row[i * QK8_0..(i + 1) * QK8_0];
        let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let d_bits = half::f16::from_f32(d).to_bits();

        let mut qs = [0i8; QK8_0];
        for (j, &v) in x.iter().enumerate() {
            let q = (v * id).round() as i32;
            qs[j] = q.clamp(-128, 127) as i8;
        }

        blocks[i] = BlockQ8_0 { d_bits, qs };
    }
    Ok(())
}

/// Dequantize a sequence of `BlockQ8_0` blocks back to F32.
/// Pure-Rust port of `dequantize_row_q8_0` (`ggml-quants.c:491`).
///
/// Per-block: `y[j] = qs[j] × d` where `d = F16_TO_F32(x.d_bits)`.
pub fn dequantize_row_q8_0(blocks: &[BlockQ8_0], out: &mut [f32]) -> Result<usize, QLegacyError> {
    let expected = blocks.len() * QK8_0;
    if out.len() < expected {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: out.len(),
            n_blocks: blocks.len(),
            bytes_per_block: QK8_0,
        });
    }
    let mut written = 0usize;
    for block in blocks {
        let d = block.d();
        for &q in &block.qs {
            out[written] = (q as f32) * d;
            written += 1;
        }
    }
    Ok(written)
}

/// Quantize F32 to flat Q8_0 block bytes (GGUF tensor data section
/// format). Convenience wrapper over [`quantize_row_q8_0`] +
/// [`BlockQ8_0::to_bytes`].
///
/// Output length: `(row.len() / QK8_0) × BLOCK_Q8_0_SIZE` bytes.
pub fn quantize_row_q8_0_to_bytes(row: &[f32]) -> Result<Vec<u8>, QLegacyError> {
    if !row.len().is_multiple_of(QK8_0) {
        return Err(QLegacyError::NotBlockAligned {
            actual: row.len(),
            qk: QK8_0,
        });
    }
    let nb = row.len() / QK8_0;
    let mut blocks = vec![
        BlockQ8_0 {
            d_bits: 0,
            qs: [0i8; QK8_0],
        };
        nb
    ];
    quantize_row_q8_0(row, &mut blocks)?;
    let mut out = Vec::with_capacity(nb * BLOCK_Q8_0_SIZE);
    for b in &blocks {
        out.extend_from_slice(&b.to_bytes());
    }
    Ok(out)
}

/// Decode flat Q8_0 block bytes back to F32. Convenience wrapper
/// over [`BlockQ8_0::from_bytes`] + [`dequantize_row_q8_0`].
///
/// `data.len()` must be a multiple of `BLOCK_Q8_0_SIZE`.
pub fn dequantize_row_q8_0_bytes(data: &[u8], out: &mut [f32]) -> Result<usize, QLegacyError> {
    if data.len() % BLOCK_Q8_0_SIZE != 0 {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: data.len(),
            n_blocks: data.len() / BLOCK_Q8_0_SIZE,
            bytes_per_block: BLOCK_Q8_0_SIZE,
        });
    }
    let n_blocks = data.len() / BLOCK_Q8_0_SIZE;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let start = i * BLOCK_Q8_0_SIZE;
        let end = start + BLOCK_Q8_0_SIZE;
        let block = BlockQ8_0::from_bytes(&data[start..end]).ok_or(
            QLegacyError::BlockSizeMismatch {
                actual: data.len(),
                n_blocks,
                bytes_per_block: BLOCK_Q8_0_SIZE,
            },
        )?;
        blocks.push(block);
    }
    dequantize_row_q8_0(&blocks, out)
}

// ────────────────────────── Q4_0 ──────────────────────────

/// Q4_0 block — 32 elements stored at 4 bpw + F16 super-block scale.
///
/// Layout matches `block_q4_0` in `/opt/llama.cpp/ggml/src/ggml-common.h:185`
/// byte-for-byte. Quants live in `[-8, 7]` symmetric around zero,
/// stored offset by `+8` in the 4-bit nibbles `[0, 15]`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    /// Block scale. F16. Note: signed (Q4_0's scale is `max_signed / -8`,
    /// which can be negative if `max > 0`).
    pub d_bits: u16,
    /// 16 bytes packing 32 × 4-bit quants.
    /// `qs[j]` low nibble is element `[j]`; high nibble is element
    /// `[j + qk/2]`. **Not interleaved** like K-quant.
    pub qs: [u8; QK4_0 / 2],
}

impl BlockQ4_0 {
    /// Read a `BlockQ4_0` from an 18-byte buffer (little-endian F16
    /// scale + 16 packed-nibble bytes).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < BLOCK_Q4_0_SIZE {
            return None;
        }
        let d_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut qs = [0u8; QK4_0 / 2];
        qs.copy_from_slice(&bytes[2..2 + QK4_0 / 2]);
        Some(Self { d_bits, qs })
    }

    /// Serialise to 18-byte little-endian buffer.
    pub fn to_bytes(&self) -> [u8; BLOCK_Q4_0_SIZE] {
        let mut out = [0u8; BLOCK_Q4_0_SIZE];
        out[0..2].copy_from_slice(&self.d_bits.to_le_bytes());
        out[2..2 + QK4_0 / 2].copy_from_slice(&self.qs);
        out
    }

    /// Block scale `d` as F32.
    pub fn d(&self) -> f32 {
        half::f16::from_bits(self.d_bits).to_f32()
    }
}

/// Quantize a row of F32 to a sequence of `BlockQ4_0` blocks.
/// Pure-Rust port of `quantize_row_q4_0_ref` (`ggml-quants.c:71`).
///
/// Per-block algorithm (mirrors C reference):
/// 1. Find `(amax, max)` — the absolute max and the **signed** value
///    of the element with that abs value.
/// 2. Block scale `d = max / -8` (negative when `max > 0`, positive
///    when `max < 0`).
/// 3. Inverse scale `id = 1/d` (or 0 if `d == 0`).
/// 4. For each `j ∈ [0, qk/2)`, quantize two elements:
///    - low nibble: `x0 = x[j] × id`; encode as `clamp(int8(x0 + 8.5), 0, 15)`
///    - high nibble: same shape for `x[j + qk/2]`
///    Pack: `qs[j] = low | (high << 4)`.
///
/// **Note**: the `+ 8.5` then `int8` cast is the reference's way to
/// shift `[-8, 7]` → `[0, 15]` with round-half-toward-positive. Since
/// `int8(positive)` truncates toward zero in C, `+ 8.5` gives correct
/// rounding for typical values; we replicate via `(x + 8.5) as i32`
/// then clamp to `[0, 15]`.
pub fn quantize_row_q4_0(row: &[f32], blocks: &mut [BlockQ4_0]) -> Result<(), QLegacyError> {
    if !row.len().is_multiple_of(QK4_0) {
        return Err(QLegacyError::NotBlockAligned {
            actual: row.len(),
            qk: QK4_0,
        });
    }
    let nb = row.len() / QK4_0;
    if blocks.len() < nb {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: blocks.len(),
            n_blocks: nb,
            bytes_per_block: BLOCK_Q4_0_SIZE,
        });
    }

    for i in 0..nb {
        let x = &row[i * QK4_0..(i + 1) * QK4_0];
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in x.iter() {
            let av = v.abs();
            if av > amax {
                amax = av;
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let d_bits = half::f16::from_f32(d).to_bits();

        let mut qs = [0u8; QK4_0 / 2];
        for j in 0..(QK4_0 / 2) {
            let x0 = x[j] * id;
            let x1 = x[j + QK4_0 / 2] * id;
            let xi0 = ((x0 + 8.5) as i32).clamp(0, 15) as u8;
            let xi1 = ((x1 + 8.5) as i32).clamp(0, 15) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }

        blocks[i] = BlockQ4_0 { d_bits, qs };
    }
    Ok(())
}

/// Dequantize a sequence of `BlockQ4_0` blocks back to F32.
/// Pure-Rust port of `dequantize_row_q4_0` (`ggml-quants.c:397`).
///
/// Per-block: for each `j ∈ [0, qk/2)`:
/// - `y[i*qk + j]      = ((qs[j] & 0xF) - 8) × d`
/// - `y[i*qk + j + qk/2] = ((qs[j] >> 4)  - 8) × d`
pub fn dequantize_row_q4_0(blocks: &[BlockQ4_0], out: &mut [f32]) -> Result<usize, QLegacyError> {
    let expected = blocks.len() * QK4_0;
    if out.len() < expected {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: out.len(),
            n_blocks: blocks.len(),
            bytes_per_block: QK4_0,
        });
    }
    for (i, block) in blocks.iter().enumerate() {
        let d = block.d();
        for j in 0..(QK4_0 / 2) {
            let x0 = ((block.qs[j] & 0x0F) as i32 - 8) as f32;
            let x1 = ((block.qs[j] >> 4) as i32 - 8) as f32;
            out[i * QK4_0 + j] = x0 * d;
            out[i * QK4_0 + j + QK4_0 / 2] = x1 * d;
        }
    }
    Ok(blocks.len() * QK4_0)
}

/// Quantize F32 to flat Q4_0 block bytes.
pub fn quantize_row_q4_0_to_bytes(row: &[f32]) -> Result<Vec<u8>, QLegacyError> {
    if !row.len().is_multiple_of(QK4_0) {
        return Err(QLegacyError::NotBlockAligned {
            actual: row.len(),
            qk: QK4_0,
        });
    }
    let nb = row.len() / QK4_0;
    let mut blocks = vec![
        BlockQ4_0 {
            d_bits: 0,
            qs: [0u8; QK4_0 / 2],
        };
        nb
    ];
    quantize_row_q4_0(row, &mut blocks)?;
    let mut out = Vec::with_capacity(nb * BLOCK_Q4_0_SIZE);
    for b in &blocks {
        out.extend_from_slice(&b.to_bytes());
    }
    Ok(out)
}

/// Decode flat Q4_0 block bytes to F32.
pub fn dequantize_row_q4_0_bytes(data: &[u8], out: &mut [f32]) -> Result<usize, QLegacyError> {
    if data.len() % BLOCK_Q4_0_SIZE != 0 {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: data.len(),
            n_blocks: data.len() / BLOCK_Q4_0_SIZE,
            bytes_per_block: BLOCK_Q4_0_SIZE,
        });
    }
    let n_blocks = data.len() / BLOCK_Q4_0_SIZE;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let start = i * BLOCK_Q4_0_SIZE;
        let end = start + BLOCK_Q4_0_SIZE;
        let block = BlockQ4_0::from_bytes(&data[start..end]).ok_or(
            QLegacyError::BlockSizeMismatch {
                actual: data.len(),
                n_blocks,
                bytes_per_block: BLOCK_Q4_0_SIZE,
            },
        )?;
        blocks.push(block);
    }
    dequantize_row_q4_0(&blocks, out)
}

// ────────────────────────── Q4_1 ──────────────────────────

/// Q4_1 block — 32 elements at 5 bpw (4-bit nibbles + per-block min)
/// + per-block F16 scale and F16 min.
///
/// Layout matches `block_q4_1` in `/opt/llama.cpp/ggml/src/ggml-common.h:198`
/// byte-for-byte. Asymmetric (with min term) — quants live in
/// `[0, 15]` and dequantize as `q × d + m`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_1 {
    /// Block scale `d = (max - min) / 15`. F16, always non-negative.
    pub d_bits: u16,
    /// Block min. F16.
    pub m_bits: u16,
    /// 16 bytes packing 32 × 4-bit low-nibbles (same packing as Q4_0).
    pub qs: [u8; QK4_1 / 2],
}

impl BlockQ4_1 {
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < BLOCK_Q4_1_SIZE {
            return None;
        }
        let d_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let m_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut qs = [0u8; QK4_1 / 2];
        qs.copy_from_slice(&bytes[4..4 + QK4_1 / 2]);
        Some(Self {
            d_bits,
            m_bits,
            qs,
        })
    }

    pub fn to_bytes(&self) -> [u8; BLOCK_Q4_1_SIZE] {
        let mut out = [0u8; BLOCK_Q4_1_SIZE];
        out[0..2].copy_from_slice(&self.d_bits.to_le_bytes());
        out[2..4].copy_from_slice(&self.m_bits.to_le_bytes());
        out[4..4 + QK4_1 / 2].copy_from_slice(&self.qs);
        out
    }

    pub fn d(&self) -> f32 {
        half::f16::from_bits(self.d_bits).to_f32()
    }

    pub fn m(&self) -> f32 {
        half::f16::from_bits(self.m_bits).to_f32()
    }
}

/// Quantize F32 row to `BlockQ4_1` blocks. Pure-Rust port of
/// `quantize_row_q4_1_ref` (`ggml-quants.c:108`).
///
/// Per-block:
/// 1. Track `(min, max)` over the 32 elements.
/// 2. `d = (max - min) / 15`.
/// 3. `id = 1/d` (or 0 if `d == 0`).
/// 4. For each `j ∈ [0, qk/2)`:
///    - `xi0 = clamp((x[j] - min) × id + 0.5, 0, 15) as u8`
///    - `xi1 = clamp((x[j + qk/2] - min) × id + 0.5, 0, 15) as u8`
///    - `qs[j] = xi0 | (xi1 << 4)`
pub fn quantize_row_q4_1(row: &[f32], blocks: &mut [BlockQ4_1]) -> Result<(), QLegacyError> {
    if !row.len().is_multiple_of(QK4_1) {
        return Err(QLegacyError::NotBlockAligned {
            actual: row.len(),
            qk: QK4_1,
        });
    }
    let nb = row.len() / QK4_1;
    if blocks.len() < nb {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: blocks.len(),
            n_blocks: nb,
            bytes_per_block: BLOCK_Q4_1_SIZE,
        });
    }

    for i in 0..nb {
        let x = &row[i * QK4_1..(i + 1) * QK4_1];
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in x.iter() {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let d = (max - min) / 15.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let d_bits = half::f16::from_f32(d).to_bits();
        let m_bits = half::f16::from_f32(min).to_bits();

        let mut qs = [0u8; QK4_1 / 2];
        for j in 0..(QK4_1 / 2) {
            let x0 = (x[j] - min) * id;
            let x1 = (x[j + QK4_1 / 2] - min) * id;
            // C: `MIN(15, (int8_t)(x0 + 0.5f))` — i8 truncation
            // toward zero, then min with 15. We replicate via
            // `as i32` then clamp.
            let xi0 = ((x0 + 0.5) as i32).clamp(0, 15) as u8;
            let xi1 = ((x1 + 0.5) as i32).clamp(0, 15) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }

        blocks[i] = BlockQ4_1 {
            d_bits,
            m_bits,
            qs,
        };
    }
    Ok(())
}

/// Dequantize `BlockQ4_1` blocks. Pure-Rust port of
/// `dequantize_row_q4_1` (`ggml-quants.c:417`).
///
/// Per-block: for each `j ∈ [0, qk/2)`:
/// - `y[j]      = (qs[j] & 0xF) × d + m`
/// - `y[j+qk/2] = (qs[j] >> 4)  × d + m`
pub fn dequantize_row_q4_1(blocks: &[BlockQ4_1], out: &mut [f32]) -> Result<usize, QLegacyError> {
    let expected = blocks.len() * QK4_1;
    if out.len() < expected {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: out.len(),
            n_blocks: blocks.len(),
            bytes_per_block: QK4_1,
        });
    }
    for (i, block) in blocks.iter().enumerate() {
        let d = block.d();
        let m = block.m();
        for j in 0..(QK4_1 / 2) {
            let x0 = (block.qs[j] & 0x0F) as i32;
            let x1 = (block.qs[j] >> 4) as i32;
            out[i * QK4_1 + j] = (x0 as f32) * d + m;
            out[i * QK4_1 + j + QK4_1 / 2] = (x1 as f32) * d + m;
        }
    }
    Ok(blocks.len() * QK4_1)
}

/// Quantize F32 to flat Q4_1 block bytes.
pub fn quantize_row_q4_1_to_bytes(row: &[f32]) -> Result<Vec<u8>, QLegacyError> {
    if !row.len().is_multiple_of(QK4_1) {
        return Err(QLegacyError::NotBlockAligned {
            actual: row.len(),
            qk: QK4_1,
        });
    }
    let nb = row.len() / QK4_1;
    let mut blocks = vec![
        BlockQ4_1 {
            d_bits: 0,
            m_bits: 0,
            qs: [0u8; QK4_1 / 2],
        };
        nb
    ];
    quantize_row_q4_1(row, &mut blocks)?;
    let mut out = Vec::with_capacity(nb * BLOCK_Q4_1_SIZE);
    for b in &blocks {
        out.extend_from_slice(&b.to_bytes());
    }
    Ok(out)
}

/// Decode flat Q4_1 block bytes to F32.
pub fn dequantize_row_q4_1_bytes(data: &[u8], out: &mut [f32]) -> Result<usize, QLegacyError> {
    if data.len() % BLOCK_Q4_1_SIZE != 0 {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: data.len(),
            n_blocks: data.len() / BLOCK_Q4_1_SIZE,
            bytes_per_block: BLOCK_Q4_1_SIZE,
        });
    }
    let n_blocks = data.len() / BLOCK_Q4_1_SIZE;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let start = i * BLOCK_Q4_1_SIZE;
        let end = start + BLOCK_Q4_1_SIZE;
        let block = BlockQ4_1::from_bytes(&data[start..end]).ok_or(
            QLegacyError::BlockSizeMismatch {
                actual: data.len(),
                n_blocks,
                bytes_per_block: BLOCK_Q4_1_SIZE,
            },
        )?;
        blocks.push(block);
    }
    dequantize_row_q4_1(&blocks, out)
}

// ────────────────────────── Q5_0 ──────────────────────────

/// Q5_0 block — 32 elements stored at 5 bpw (4-bit nibbles + 1-bit
/// high bit packed in `qh: u32`) + F16 super-block scale.
///
/// Layout matches `block_q5_0` in `/opt/llama.cpp/ggml/src/ggml-common.h:222`
/// byte-for-byte. Quants live in `[-16, 15]` symmetric around zero,
/// stored offset by `+16` in the 5-bit unsigned form `[0, 31]`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_0 {
    /// Block scale. F16. Sign mirrors Q4_0 (`max_signed / -16`).
    pub d_bits: u16,
    /// 32 high bits packed into a u32 — bit `j` = high bit of element
    /// `[j]` (for `j ∈ [0, 16)`) or element `[j + 16]` (for the upper
    /// 16 bits — actually the C code packs the upper-half element's
    /// high bit at `j + qk/2`).
    pub qh: u32,
    /// 16 bytes packing 32 × 4-bit low-nibbles. Same packing as Q4_0
    /// (low nibble = `[j]`, high nibble = `[j + qk/2]`).
    pub qs: [u8; QK5_0 / 2],
}

impl BlockQ5_0 {
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < BLOCK_Q5_0_SIZE {
            return None;
        }
        let d_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let qh = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
        let mut qs = [0u8; QK5_0 / 2];
        qs.copy_from_slice(&bytes[6..6 + QK5_0 / 2]);
        Some(Self { d_bits, qh, qs })
    }

    pub fn to_bytes(&self) -> [u8; BLOCK_Q5_0_SIZE] {
        let mut out = [0u8; BLOCK_Q5_0_SIZE];
        out[0..2].copy_from_slice(&self.d_bits.to_le_bytes());
        out[2..6].copy_from_slice(&self.qh.to_le_bytes());
        out[6..6 + QK5_0 / 2].copy_from_slice(&self.qs);
        out
    }

    pub fn d(&self) -> f32 {
        half::f16::from_bits(self.d_bits).to_f32()
    }
}

/// Quantize F32 row to `BlockQ5_0` blocks. Pure-Rust port of
/// `quantize_row_q5_0_ref` (`ggml-quants.c:145`).
pub fn quantize_row_q5_0(row: &[f32], blocks: &mut [BlockQ5_0]) -> Result<(), QLegacyError> {
    if !row.len().is_multiple_of(QK5_0) {
        return Err(QLegacyError::NotBlockAligned {
            actual: row.len(),
            qk: QK5_0,
        });
    }
    let nb = row.len() / QK5_0;
    if blocks.len() < nb {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: blocks.len(),
            n_blocks: nb,
            bytes_per_block: BLOCK_Q5_0_SIZE,
        });
    }

    for i in 0..nb {
        let x = &row[i * QK5_0..(i + 1) * QK5_0];
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in x.iter() {
            let av = v.abs();
            if av > amax {
                amax = av;
                max = v;
            }
        }
        let d = max / -16.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let d_bits = half::f16::from_f32(d).to_bits();

        let mut qh: u32 = 0;
        let mut qs = [0u8; QK5_0 / 2];
        for j in 0..(QK5_0 / 2) {
            let x0 = x[j] * id;
            let x1 = x[j + QK5_0 / 2] * id;
            let xi0 = ((x0 + 16.5) as i32).clamp(0, 31) as u8;
            let xi1 = ((x1 + 16.5) as i32).clamp(0, 31) as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            // 5th bit goes into qh at positions j (low half) and j + qk/2 (high half).
            qh |= (((xi0 & 0x10) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10) >> 4) as u32) << (j + QK5_0 / 2);
        }

        blocks[i] = BlockQ5_0 { d_bits, qh, qs };
    }
    Ok(())
}

/// Dequantize `BlockQ5_0` blocks to F32. Pure-Rust port of
/// `dequantize_row_q5_0` (`ggml-quants.c:438`).
pub fn dequantize_row_q5_0(blocks: &[BlockQ5_0], out: &mut [f32]) -> Result<usize, QLegacyError> {
    let expected = blocks.len() * QK5_0;
    if out.len() < expected {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: out.len(),
            n_blocks: blocks.len(),
            bytes_per_block: QK5_0,
        });
    }
    for (i, block) in blocks.iter().enumerate() {
        let d = block.d();
        let qh = block.qh;
        for j in 0..(QK5_0 / 2) {
            // High-bit reconstruction matches `dequantize_row_q5_0` exactly.
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((block.qs[j] & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((block.qs[j] >> 4) | xh_1) as i32 - 16;
            out[i * QK5_0 + j] = (x0 as f32) * d;
            out[i * QK5_0 + j + QK5_0 / 2] = (x1 as f32) * d;
        }
    }
    Ok(blocks.len() * QK5_0)
}

/// Quantize F32 to flat Q5_0 block bytes.
pub fn quantize_row_q5_0_to_bytes(row: &[f32]) -> Result<Vec<u8>, QLegacyError> {
    if !row.len().is_multiple_of(QK5_0) {
        return Err(QLegacyError::NotBlockAligned {
            actual: row.len(),
            qk: QK5_0,
        });
    }
    let nb = row.len() / QK5_0;
    let mut blocks = vec![
        BlockQ5_0 {
            d_bits: 0,
            qh: 0,
            qs: [0u8; QK5_0 / 2],
        };
        nb
    ];
    quantize_row_q5_0(row, &mut blocks)?;
    let mut out = Vec::with_capacity(nb * BLOCK_Q5_0_SIZE);
    for b in &blocks {
        out.extend_from_slice(&b.to_bytes());
    }
    Ok(out)
}

/// Decode flat Q5_0 block bytes to F32.
pub fn dequantize_row_q5_0_bytes(data: &[u8], out: &mut [f32]) -> Result<usize, QLegacyError> {
    if data.len() % BLOCK_Q5_0_SIZE != 0 {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: data.len(),
            n_blocks: data.len() / BLOCK_Q5_0_SIZE,
            bytes_per_block: BLOCK_Q5_0_SIZE,
        });
    }
    let n_blocks = data.len() / BLOCK_Q5_0_SIZE;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let start = i * BLOCK_Q5_0_SIZE;
        let end = start + BLOCK_Q5_0_SIZE;
        let block = BlockQ5_0::from_bytes(&data[start..end]).ok_or(
            QLegacyError::BlockSizeMismatch {
                actual: data.len(),
                n_blocks,
                bytes_per_block: BLOCK_Q5_0_SIZE,
            },
        )?;
        blocks.push(block);
    }
    dequantize_row_q5_0(&blocks, out)
}

// ────────────────────────── Q5_1 ──────────────────────────

/// Q5_1 block — same shape as Q5_0 plus a per-block min term.
/// Layout matches `block_q5_1` (`ggml-common.h:236`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_1 {
    /// Block scale `d = (max - min) / 31`. F16, always non-negative.
    pub d_bits: u16,
    /// Block min. F16.
    pub m_bits: u16,
    /// 32 high bits packed (same encoding as Q5_0).
    pub qh: u32,
    /// 16 bytes packing 32 × 4-bit low-nibbles.
    pub qs: [u8; QK5_1 / 2],
}

impl BlockQ5_1 {
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < BLOCK_Q5_1_SIZE {
            return None;
        }
        let d_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let m_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let qh = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let mut qs = [0u8; QK5_1 / 2];
        qs.copy_from_slice(&bytes[8..8 + QK5_1 / 2]);
        Some(Self {
            d_bits,
            m_bits,
            qh,
            qs,
        })
    }

    pub fn to_bytes(&self) -> [u8; BLOCK_Q5_1_SIZE] {
        let mut out = [0u8; BLOCK_Q5_1_SIZE];
        out[0..2].copy_from_slice(&self.d_bits.to_le_bytes());
        out[2..4].copy_from_slice(&self.m_bits.to_le_bytes());
        out[4..8].copy_from_slice(&self.qh.to_le_bytes());
        out[8..8 + QK5_1 / 2].copy_from_slice(&self.qs);
        out
    }

    pub fn d(&self) -> f32 {
        half::f16::from_bits(self.d_bits).to_f32()
    }

    pub fn m(&self) -> f32 {
        half::f16::from_bits(self.m_bits).to_f32()
    }
}

/// Quantize F32 row to `BlockQ5_1` blocks. Pure-Rust port of
/// `quantize_row_q5_1_ref` (`ggml-quants.c:189`).
pub fn quantize_row_q5_1(row: &[f32], blocks: &mut [BlockQ5_1]) -> Result<(), QLegacyError> {
    if !row.len().is_multiple_of(QK5_1) {
        return Err(QLegacyError::NotBlockAligned {
            actual: row.len(),
            qk: QK5_1,
        });
    }
    let nb = row.len() / QK5_1;
    if blocks.len() < nb {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: blocks.len(),
            n_blocks: nb,
            bytes_per_block: BLOCK_Q5_1_SIZE,
        });
    }

    for i in 0..nb {
        let x = &row[i * QK5_1..(i + 1) * QK5_1];
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in x.iter() {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let d = (max - min) / 31.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let d_bits = half::f16::from_f32(d).to_bits();
        let m_bits = half::f16::from_f32(min).to_bits();

        let mut qh: u32 = 0;
        let mut qs = [0u8; QK5_1 / 2];
        for j in 0..(QK5_1 / 2) {
            let x0 = (x[j] - min) * id;
            let x1 = (x[j + QK5_1 / 2] - min) * id;
            // C: `(uint8_t)(x0 + 0.5f)` — no MIN clamp; trusts that
            // `(x0 + 0.5f).trunc()` falls in [0, 31] for in-range input.
            // We reproduce the trunc-toward-zero behaviour via `as u8`.
            let xi0 = (x0 + 0.5) as u8;
            let xi1 = (x1 + 0.5) as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= (((xi0 & 0x10) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10) >> 4) as u32) << (j + QK5_1 / 2);
        }

        blocks[i] = BlockQ5_1 {
            d_bits,
            m_bits,
            qh,
            qs,
        };
    }
    Ok(())
}

/// Dequantize `BlockQ5_1` blocks to F32. Pure-Rust port of
/// `dequantize_row_q5_1` (`ggml-quants.c:464`).
pub fn dequantize_row_q5_1(blocks: &[BlockQ5_1], out: &mut [f32]) -> Result<usize, QLegacyError> {
    let expected = blocks.len() * QK5_1;
    if out.len() < expected {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: out.len(),
            n_blocks: blocks.len(),
            bytes_per_block: QK5_1,
        });
    }
    for (i, block) in blocks.iter().enumerate() {
        let d = block.d();
        let m = block.m();
        let qh = block.qh;
        for j in 0..(QK5_1 / 2) {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((block.qs[j] & 0x0F) | xh_0) as i32;
            let x1 = ((block.qs[j] >> 4) | xh_1) as i32;
            out[i * QK5_1 + j] = (x0 as f32) * d + m;
            out[i * QK5_1 + j + QK5_1 / 2] = (x1 as f32) * d + m;
        }
    }
    Ok(blocks.len() * QK5_1)
}

/// Quantize F32 to flat Q5_1 block bytes.
pub fn quantize_row_q5_1_to_bytes(row: &[f32]) -> Result<Vec<u8>, QLegacyError> {
    if !row.len().is_multiple_of(QK5_1) {
        return Err(QLegacyError::NotBlockAligned {
            actual: row.len(),
            qk: QK5_1,
        });
    }
    let nb = row.len() / QK5_1;
    let mut blocks = vec![
        BlockQ5_1 {
            d_bits: 0,
            m_bits: 0,
            qh: 0,
            qs: [0u8; QK5_1 / 2],
        };
        nb
    ];
    quantize_row_q5_1(row, &mut blocks)?;
    let mut out = Vec::with_capacity(nb * BLOCK_Q5_1_SIZE);
    for b in &blocks {
        out.extend_from_slice(&b.to_bytes());
    }
    Ok(out)
}

/// Decode flat Q5_1 block bytes to F32.
pub fn dequantize_row_q5_1_bytes(data: &[u8], out: &mut [f32]) -> Result<usize, QLegacyError> {
    if data.len() % BLOCK_Q5_1_SIZE != 0 {
        return Err(QLegacyError::BlockSizeMismatch {
            actual: data.len(),
            n_blocks: data.len() / BLOCK_Q5_1_SIZE,
            bytes_per_block: BLOCK_Q5_1_SIZE,
        });
    }
    let n_blocks = data.len() / BLOCK_Q5_1_SIZE;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let start = i * BLOCK_Q5_1_SIZE;
        let end = start + BLOCK_Q5_1_SIZE;
        let block = BlockQ5_1::from_bytes(&data[start..end]).ok_or(
            QLegacyError::BlockSizeMismatch {
                actual: data.len(),
                n_blocks,
                bytes_per_block: BLOCK_Q5_1_SIZE,
            },
        )?;
        blocks.push(block);
    }
    dequantize_row_q5_1(&blocks, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Block size matches llama.cpp's static_assert
    /// `sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0`.
    #[test]
    fn block_q8_0_size_matches_c_struct() {
        assert_eq!(BLOCK_Q8_0_SIZE, 2 + 32);
        assert_eq!(BLOCK_Q8_0_SIZE, 34);
    }

    /// Round-trip BlockQ8_0 via to_bytes / from_bytes preserves the
    /// struct byte-for-byte.
    #[test]
    fn block_q8_0_byte_round_trip() {
        let block = BlockQ8_0 {
            d_bits: 0xABCD,
            qs: {
                let mut a = [0i8; QK8_0];
                for (i, slot) in a.iter_mut().enumerate() {
                    *slot = (i as i8).wrapping_mul(7).wrapping_sub(64);
                }
                a
            },
        };
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), BLOCK_Q8_0_SIZE);
        let decoded = BlockQ8_0::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.d_bits, block.d_bits);
        assert_eq!(decoded.qs, block.qs);
    }

    /// Quantizing all-zero F32 input yields a block with `d=0` and
    /// `qs=[0; 32]`. Round-trip dequantize back to all zeros.
    #[test]
    fn quantize_q8_0_zero_input() {
        let row = vec![0.0_f32; QK8_0];
        let mut blocks = vec![BlockQ8_0 {
            d_bits: 0xFFFF, // pre-fill to confirm overwrite
            qs: [127i8; QK8_0],
        }; 1];
        quantize_row_q8_0(&row, &mut blocks).unwrap();
        assert_eq!(blocks[0].d_bits, 0);
        for &q in &blocks[0].qs {
            assert_eq!(q, 0);
        }
        let mut decoded = vec![0.0_f32; QK8_0];
        dequantize_row_q8_0(&blocks, &mut decoded).unwrap();
        for &v in &decoded {
            assert_eq!(v, 0.0);
        }
    }

    /// Quantize a smooth ramp [-1, 1] over 32 elements; round-trip
    /// RMSE bound. Q8_0's 8-bit codebook over `[-127, 127]` levels
    /// covers a range of `[-amax, amax]` with step `amax/127`. For
    /// amax=1, expected step = 1/127 ≈ 0.0079, so RMSE < 0.005 is
    /// well within the bound.
    #[test]
    fn quantize_q8_0_round_trip_synthetic() {
        let row: Vec<f32> = (0..QK8_0)
            .map(|i| -1.0 + 2.0 * (i as f32) / ((QK8_0 - 1) as f32))
            .collect();
        let mut blocks = vec![BlockQ8_0 {
            d_bits: 0,
            qs: [0i8; QK8_0],
        }; 1];
        quantize_row_q8_0(&row, &mut blocks).unwrap();

        let mut decoded = vec![0.0_f32; QK8_0];
        dequantize_row_q8_0(&blocks, &mut decoded).unwrap();

        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(rmse < 0.005, "Q8_0 round-trip RMSE {rmse} > 0.005");
    }

    /// Single-spike input (one nonzero, rest zero) — verifies the
    /// scale-from-amax behavior. With amax=10, d=10/127≈0.0787,
    /// so qs[5]=round(10/0.0787)=127 (saturated), and decoded[5]=10.
    #[test]
    fn quantize_q8_0_single_spike() {
        let mut row = vec![0.0_f32; QK8_0];
        row[5] = 10.0;
        let mut blocks = vec![BlockQ8_0 {
            d_bits: 0,
            qs: [0i8; QK8_0],
        }; 1];
        quantize_row_q8_0(&row, &mut blocks).unwrap();

        // qs[5] should saturate at 127.
        assert_eq!(blocks[0].qs[5], 127);
        // d ≈ 10/127 ≈ 0.0787 (F16 representation).
        let d = blocks[0].d();
        assert!(d > 0.0 && d < 0.1, "d = {d}");
    }

    /// Multi-block: 4 super-blocks of 32 elements each. Verifies the
    /// per-block loop.
    #[test]
    fn quantize_q8_0_multi_block() {
        let n_blocks = 4;
        let row: Vec<f32> = (0..n_blocks * QK8_0)
            .map(|i| {
                let t = (i as f32) / ((n_blocks * QK8_0) as f32 - 1.0);
                -1.5 + 3.0 * t
            })
            .collect();
        let mut blocks = vec![BlockQ8_0 {
            d_bits: 0,
            qs: [0i8; QK8_0],
        }; n_blocks];
        quantize_row_q8_0(&row, &mut blocks).unwrap();

        let mut decoded = vec![0.0_f32; n_blocks * QK8_0];
        dequantize_row_q8_0(&blocks, &mut decoded).unwrap();

        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(rmse < 0.01, "Q8_0 multi-block RMSE {rmse} > 0.01");
    }

    /// Flat-bytes wrapper produces exactly `nb × BLOCK_Q8_0_SIZE`
    /// bytes; round-trip via `dequantize_row_q8_0_bytes` matches.
    #[test]
    fn quantize_q8_0_to_bytes_round_trip() {
        let row: Vec<f32> = (0..2 * QK8_0)
            .map(|i| (i as f32 - 32.0) / 32.0)
            .collect();
        let bytes = quantize_row_q8_0_to_bytes(&row).unwrap();
        assert_eq!(bytes.len(), 2 * BLOCK_Q8_0_SIZE);

        let mut decoded = vec![0.0_f32; 2 * QK8_0];
        dequantize_row_q8_0_bytes(&bytes, &mut decoded).unwrap();

        // RMSE bound (looser than per-block since two blocks, one
        // each side of zero).
        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(rmse < 0.01, "Q8_0 bytes round-trip RMSE {rmse} > 0.01");
    }

    /// Misaligned input rejected with typed error.
    #[test]
    fn quantize_q8_0_rejects_misaligned() {
        let bad = vec![0.0_f32; QK8_0 - 1];
        let mut blocks = vec![BlockQ8_0 {
            d_bits: 0,
            qs: [0i8; QK8_0],
        }; 1];
        let err = quantize_row_q8_0(&bad, &mut blocks).unwrap_err();
        match err {
            QLegacyError::NotBlockAligned { actual, qk } => {
                assert_eq!(actual, QK8_0 - 1);
                assert_eq!(qk, QK8_0);
            }
            _ => panic!("expected NotBlockAligned"),
        }
    }

    /// `dequantize_row_q8_0_bytes` rejects misaligned input length.
    #[test]
    fn dequantize_q8_0_bytes_rejects_misaligned() {
        let bad = vec![0u8; 100]; // not a multiple of 34
        let mut out = vec![0.0_f32; QK8_0];
        let err = dequantize_row_q8_0_bytes(&bad, &mut out).unwrap_err();
        match err {
            QLegacyError::BlockSizeMismatch {
                bytes_per_block, ..
            } => {
                assert_eq!(bytes_per_block, BLOCK_Q8_0_SIZE);
            }
            _ => panic!("expected BlockSizeMismatch"),
        }
    }

    /// Output buffer too small surfaces typed error.
    #[test]
    fn dequantize_q8_0_rejects_short_output() {
        let block = BlockQ8_0 {
            d_bits: 0,
            qs: [0i8; QK8_0],
        };
        let mut out = vec![0.0_f32; QK8_0 - 1];
        let err = dequantize_row_q8_0(std::slice::from_ref(&block), &mut out).unwrap_err();
        match err {
            QLegacyError::BlockSizeMismatch { actual, .. } => {
                assert_eq!(actual, QK8_0 - 1);
            }
            _ => panic!("expected BlockSizeMismatch"),
        }
    }

    // ─────────────────── Q4_0 tests ───────────────────

    /// Block size matches llama.cpp's static_assert.
    #[test]
    fn block_q4_0_size_matches_c_struct() {
        assert_eq!(BLOCK_Q4_0_SIZE, 2 + 16);
        assert_eq!(BLOCK_Q4_0_SIZE, 18);
    }

    /// Round-trip BlockQ4_0 via to_bytes / from_bytes.
    #[test]
    fn block_q4_0_byte_round_trip() {
        let block = BlockQ4_0 {
            d_bits: 0xCAFE,
            qs: [
                0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55,
                0x66, 0x77, 0x88,
            ],
        };
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), BLOCK_Q4_0_SIZE);
        let decoded = BlockQ4_0::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.d_bits, block.d_bits);
        assert_eq!(decoded.qs, block.qs);
    }

    /// All-zero input → d=±0 (sign depends on `max / -8` IEEE
    /// behavior — for `max = +0.0`, the result is `-0.0` with F16
    /// bit-pattern 0x8000, matching the C reference exactly).
    /// qs each = `0x88` (each nibble = `(0 + 8.5).trunc() = 8`,
    /// packed: `8 | (8 << 4) = 0x88`). Round-trip yields all zeros
    /// since both `+0.0` and `-0.0` produce 0 in `(q-8) × d`.
    #[test]
    fn quantize_q4_0_zero_input() {
        let row = vec![0.0_f32; QK4_0];
        let mut blocks = vec![BlockQ4_0 {
            d_bits: 0xFFFF,
            qs: [0xFFu8; QK4_0 / 2],
        }; 1];
        quantize_row_q4_0(&row, &mut blocks).unwrap();
        // d_bits is F16(±0). Both 0x0000 (+0) and 0x8000 (-0) are
        // valid; the C reference produces -0 via `0.0 / -8.0`.
        assert!(
            blocks[0].d_bits == 0 || blocks[0].d_bits == 0x8000,
            "d_bits = {:#06x}, expected F16(±0)",
            blocks[0].d_bits
        );
        assert_eq!(blocks[0].d().abs(), 0.0);
        // Each nibble = clamp(int32(0 + 8.5), 0, 15) = 8.
        // Packed byte: 8 | (8 << 4) = 0x88.
        for &b in &blocks[0].qs {
            assert_eq!(b, 0x88);
        }
        let mut decoded = vec![0.0_f32; QK4_0];
        dequantize_row_q4_0(&blocks, &mut decoded).unwrap();
        // d=±0 → all decoded = 0 regardless of qs.
        for &v in &decoded {
            assert_eq!(v, 0.0);
        }
    }

    /// Smooth ramp [-1, 1] over 32 elements: round-trip RMSE bound.
    /// Q4_0's 4-bit codebook over `[-amax, amax]` with 16 levels
    /// gives step = amax/8 = 0.125, so RMSE ~ 0.125/sqrt(12) ≈ 0.036.
    #[test]
    fn quantize_q4_0_round_trip_synthetic() {
        let row: Vec<f32> = (0..QK4_0)
            .map(|i| -1.0 + 2.0 * (i as f32) / ((QK4_0 - 1) as f32))
            .collect();
        let mut blocks = vec![BlockQ4_0 {
            d_bits: 0,
            qs: [0u8; QK4_0 / 2],
        }; 1];
        quantize_row_q4_0(&row, &mut blocks).unwrap();

        let mut decoded = vec![0.0_f32; QK4_0];
        dequantize_row_q4_0(&blocks, &mut decoded).unwrap();

        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        // 4-bit symmetric Q4_0 expected RMSE ~0.04 on smooth ramp.
        assert!(rmse < 0.06, "Q4_0 round-trip RMSE {rmse} > 0.06");
    }

    /// `quantize_row_q4_0_to_bytes` produces 18-byte output for one
    /// block; round-trip via `dequantize_row_q4_0_bytes` recovers F32
    /// within the RMSE bound.
    #[test]
    fn quantize_q4_0_to_bytes_round_trip() {
        let row: Vec<f32> = (0..2 * QK4_0)
            .map(|i| (i as f32 - 32.0) / 32.0)
            .collect();
        let bytes = quantize_row_q4_0_to_bytes(&row).unwrap();
        assert_eq!(bytes.len(), 2 * BLOCK_Q4_0_SIZE);

        let mut decoded = vec![0.0_f32; 2 * QK4_0];
        dequantize_row_q4_0_bytes(&bytes, &mut decoded).unwrap();

        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(rmse < 0.07, "Q4_0 bytes round-trip RMSE {rmse} > 0.07");
    }

    /// `quantize_row_q4_0` rejects misaligned input length.
    #[test]
    fn quantize_q4_0_rejects_misaligned() {
        let bad = vec![0.0_f32; QK4_0 + 5];
        let mut blocks = vec![BlockQ4_0 {
            d_bits: 0,
            qs: [0u8; QK4_0 / 2],
        }; 2];
        let err = quantize_row_q4_0(&bad, &mut blocks).unwrap_err();
        match err {
            QLegacyError::NotBlockAligned { actual, qk } => {
                assert_eq!(actual, QK4_0 + 5);
                assert_eq!(qk, QK4_0);
            }
            _ => panic!("expected NotBlockAligned"),
        }
    }

    // ─────────────────── Q5_0 tests ───────────────────

    /// Block size matches llama.cpp's static_assert
    /// `sizeof(block_q5_0) == sizeof(ggml_half) + sizeof(uint32_t) + QK5_0/2`.
    #[test]
    fn block_q5_0_size_matches_c_struct() {
        assert_eq!(BLOCK_Q5_0_SIZE, 2 + 4 + 16);
        assert_eq!(BLOCK_Q5_0_SIZE, 22);
    }

    /// Round-trip BlockQ5_0 via to_bytes / from_bytes.
    #[test]
    fn block_q5_0_byte_round_trip() {
        let block = BlockQ5_0 {
            d_bits: 0xBEEF,
            qh: 0x12345678,
            qs: [
                0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55,
                0x66, 0x77, 0x88,
            ],
        };
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), BLOCK_Q5_0_SIZE);
        let decoded = BlockQ5_0::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.d_bits, block.d_bits);
        assert_eq!(decoded.qh, block.qh);
        assert_eq!(decoded.qs, block.qs);
    }

    /// All-zero input: d=±0, qh=0, qs=[16|16<<4]=0x... wait,
    /// nibble pack is `(xi0 & 0x0F) | ((xi1 & 0x0F) << 4)`. With
    /// `xi0 = (0 + 16.5).trunc() = 16`, the low 4 bits of 16 = 0,
    /// and `(xi0 & 0x10) >> 4 = 1` goes into qh. So qs = 0x00, but
    /// qh has all 32 bits set.
    #[test]
    fn quantize_q5_0_zero_input() {
        let row = vec![0.0_f32; QK5_0];
        let mut blocks = vec![BlockQ5_0 {
            d_bits: 0xFFFF,
            qh: 0xFFFFFFFF,
            qs: [0xFFu8; QK5_0 / 2],
        }; 1];
        quantize_row_q5_0(&row, &mut blocks).unwrap();
        // d = 0/-16 = -0.0 → F16 ±0.
        assert!(
            blocks[0].d_bits == 0 || blocks[0].d_bits == 0x8000,
            "d_bits = {:#06x}",
            blocks[0].d_bits
        );
        // Each xi nibble = 16 → low 4 bits = 0 → qs = 0; bit 4 → qh.
        for &b in &blocks[0].qs {
            assert_eq!(b, 0);
        }
        // qh: bit j and bit j+16 set for each j ∈ [0, 16) → all 32 bits.
        assert_eq!(blocks[0].qh, 0xFFFFFFFF);

        // Dequantize: xh_0 = (qh >> j << 4) & 0x10 = 0x10 (since qh bit j=1).
        // (qs[j] & 0xF | xh_0) - 16 = (0 | 16) - 16 = 0. Output = 0 * d = 0.
        let mut decoded = vec![0.0_f32; QK5_0];
        dequantize_row_q5_0(&blocks, &mut decoded).unwrap();
        for &v in &decoded {
            assert_eq!(v, 0.0);
        }
    }

    /// Q5_0 round-trip RMSE on a smooth ramp [-1, 1]. 5-bit symmetric
    /// codebook gives 32 levels over `[-amax, amax]`, expected RMSE
    /// ~0.018.
    #[test]
    fn quantize_q5_0_round_trip_synthetic() {
        let row: Vec<f32> = (0..QK5_0)
            .map(|i| -1.0 + 2.0 * (i as f32) / ((QK5_0 - 1) as f32))
            .collect();
        let mut blocks = vec![BlockQ5_0 {
            d_bits: 0,
            qh: 0,
            qs: [0u8; QK5_0 / 2],
        }; 1];
        quantize_row_q5_0(&row, &mut blocks).unwrap();
        let mut decoded = vec![0.0_f32; QK5_0];
        dequantize_row_q5_0(&blocks, &mut decoded).unwrap();
        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(rmse < 0.025, "Q5_0 RMSE {rmse} > 0.025");
    }

    /// Q5_0 vs Q4_0: more bits → less RMSE on smooth ramp.
    #[test]
    fn quantize_q5_0_lower_rmse_than_q4_0() {
        let row: Vec<f32> = (0..QK5_0)
            .map(|i| -2.0 + 4.0 * (i as f32) / ((QK5_0 - 1) as f32))
            .collect();
        let mut q4 = vec![BlockQ4_0 {
            d_bits: 0,
            qs: [0u8; QK4_0 / 2],
        }; 1];
        let mut q5 = vec![BlockQ5_0 {
            d_bits: 0,
            qh: 0,
            qs: [0u8; QK5_0 / 2],
        }; 1];
        quantize_row_q4_0(&row, &mut q4).unwrap();
        quantize_row_q5_0(&row, &mut q5).unwrap();
        let mut d4 = vec![0.0_f32; QK4_0];
        let mut d5 = vec![0.0_f32; QK5_0];
        dequantize_row_q4_0(&q4, &mut d4).unwrap();
        dequantize_row_q5_0(&q5, &mut d5).unwrap();
        let rmse = |a: &[f32], b: &[f32]| -> f64 {
            let mut sse = 0.0;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = (*x as f64) - (*y as f64);
                sse += d * d;
            }
            (sse / a.len() as f64).sqrt()
        };
        let r4 = rmse(&row, &d4);
        let r5 = rmse(&row, &d5);
        assert!(r5 < r4, "Q5_0 RMSE {r5} should be < Q4_0 RMSE {r4}");
    }

    /// Q5_0 flat-bytes round-trip.
    #[test]
    fn quantize_q5_0_to_bytes_round_trip() {
        let row: Vec<f32> = (0..2 * QK5_0)
            .map(|i| (i as f32 - 32.0) / 32.0)
            .collect();
        let bytes = quantize_row_q5_0_to_bytes(&row).unwrap();
        assert_eq!(bytes.len(), 2 * BLOCK_Q5_0_SIZE);
        let mut decoded = vec![0.0_f32; 2 * QK5_0];
        dequantize_row_q5_0_bytes(&bytes, &mut decoded).unwrap();
        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(rmse < 0.04, "Q5_0 bytes round-trip RMSE {rmse} > 0.04");
    }

    // ─────────────────── Q5_1 tests ───────────────────

    /// Block size matches llama.cpp's static_assert.
    #[test]
    fn block_q5_1_size_matches_c_struct() {
        assert_eq!(BLOCK_Q5_1_SIZE, 2 + 2 + 4 + 16);
        assert_eq!(BLOCK_Q5_1_SIZE, 24);
    }

    /// Round-trip BlockQ5_1.
    #[test]
    fn block_q5_1_byte_round_trip() {
        let block = BlockQ5_1 {
            d_bits: 0xCAFE,
            m_bits: 0xBEEF,
            qh: 0x87654321,
            qs: [
                0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
                0x77, 0x88, 0x99,
            ],
        };
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), BLOCK_Q5_1_SIZE);
        let decoded = BlockQ5_1::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.d_bits, block.d_bits);
        assert_eq!(decoded.m_bits, block.m_bits);
        assert_eq!(decoded.qh, block.qh);
        assert_eq!(decoded.qs, block.qs);
    }

    /// Q5_1 round-trip on smooth ramp [0, 1] (asymmetric — Q5_1
    /// shines here since min term offsets the codebook).
    /// 5-bit codebook over `[min, max]` with 32 levels, expected
    /// RMSE ~0.005.
    #[test]
    fn quantize_q5_1_round_trip_asymmetric() {
        let row: Vec<f32> = (0..QK5_1)
            .map(|i| (i as f32) / ((QK5_1 - 1) as f32)) // [0, 1]
            .collect();
        let mut blocks = vec![BlockQ5_1 {
            d_bits: 0,
            m_bits: 0,
            qh: 0,
            qs: [0u8; QK5_1 / 2],
        }; 1];
        quantize_row_q5_1(&row, &mut blocks).unwrap();
        let mut decoded = vec![0.0_f32; QK5_1];
        dequantize_row_q5_1(&blocks, &mut decoded).unwrap();
        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(rmse < 0.012, "Q5_1 asymmetric RMSE {rmse} > 0.012");
    }

    /// Q5_1 vs Q5_0 on asymmetric input [0, 2]: Q5_1's min term gives
    /// it an advantage when the distribution is shifted away from
    /// zero. Both should be finite; Q5_1 should be ≤ Q5_0 RMSE.
    #[test]
    fn quantize_q5_1_lower_rmse_than_q5_0_on_asymmetric() {
        let row: Vec<f32> = (0..QK5_0)
            .map(|i| 2.0 * (i as f32) / ((QK5_0 - 1) as f32)) // [0, 2]
            .collect();
        let mut q50 = vec![BlockQ5_0 {
            d_bits: 0,
            qh: 0,
            qs: [0u8; QK5_0 / 2],
        }; 1];
        let mut q51 = vec![BlockQ5_1 {
            d_bits: 0,
            m_bits: 0,
            qh: 0,
            qs: [0u8; QK5_1 / 2],
        }; 1];
        quantize_row_q5_0(&row, &mut q50).unwrap();
        quantize_row_q5_1(&row, &mut q51).unwrap();
        let mut d50 = vec![0.0_f32; QK5_0];
        let mut d51 = vec![0.0_f32; QK5_1];
        dequantize_row_q5_0(&q50, &mut d50).unwrap();
        dequantize_row_q5_1(&q51, &mut d51).unwrap();
        let rmse = |a: &[f32], b: &[f32]| -> f64 {
            let mut sse = 0.0;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = (*x as f64) - (*y as f64);
                sse += d * d;
            }
            (sse / a.len() as f64).sqrt()
        };
        let r50 = rmse(&row, &d50);
        let r51 = rmse(&row, &d51);
        assert!(
            r51 <= r50,
            "Q5_1 RMSE {r51} should be ≤ Q5_0 RMSE {r50} on asymmetric input"
        );
    }

    /// Q5_1 flat-bytes round-trip.
    #[test]
    fn quantize_q5_1_to_bytes_round_trip() {
        let row: Vec<f32> = (0..2 * QK5_1)
            .map(|i| (i as f32) / 32.0) // [0, 2)
            .collect();
        let bytes = quantize_row_q5_1_to_bytes(&row).unwrap();
        assert_eq!(bytes.len(), 2 * BLOCK_Q5_1_SIZE);
        let mut decoded = vec![0.0_f32; 2 * QK5_1];
        dequantize_row_q5_1_bytes(&bytes, &mut decoded).unwrap();
        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(rmse < 0.04, "Q5_1 bytes round-trip RMSE {rmse} > 0.04");
    }

    // ─────────────────── Q4_1 tests (iter-3n) ───────────────────

    /// Block size matches llama.cpp's static_assert
    /// `sizeof(block_q4_1) == 2*sizeof(ggml_half) + QK4_1/2`.
    #[test]
    fn block_q4_1_size_matches_c_struct() {
        assert_eq!(BLOCK_Q4_1_SIZE, 2 + 2 + 16);
        assert_eq!(BLOCK_Q4_1_SIZE, 20);
    }

    /// Round-trip BlockQ4_1.
    #[test]
    fn block_q4_1_byte_round_trip() {
        let block = BlockQ4_1 {
            d_bits: 0xCAFE,
            m_bits: 0xBEEF,
            qs: [
                0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55,
                0x66, 0x77, 0x88,
            ],
        };
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), BLOCK_Q4_1_SIZE);
        let decoded = BlockQ4_1::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.d_bits, block.d_bits);
        assert_eq!(decoded.m_bits, block.m_bits);
        assert_eq!(decoded.qs, block.qs);
    }

    /// Q4_1 round-trip on asymmetric ramp [0, 1] — Q4_1's strength.
    /// 4-bit codebook over `[min, max]` with 16 levels gives step
    /// = 1/15 ≈ 0.067, so RMSE ~ 0.067/sqrt(12) ≈ 0.019.
    #[test]
    fn quantize_q4_1_round_trip_asymmetric() {
        let row: Vec<f32> = (0..QK4_1)
            .map(|i| (i as f32) / ((QK4_1 - 1) as f32))
            .collect();
        let mut blocks = vec![BlockQ4_1 {
            d_bits: 0,
            m_bits: 0,
            qs: [0u8; QK4_1 / 2],
        }; 1];
        quantize_row_q4_1(&row, &mut blocks).unwrap();
        let mut decoded = vec![0.0_f32; QK4_1];
        dequantize_row_q4_1(&blocks, &mut decoded).unwrap();
        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(rmse < 0.025, "Q4_1 asymmetric RMSE {rmse} > 0.025");
    }

    /// Q4_1 vs Q4_0 on asymmetric input [0, 2]: Q4_1's min term gives
    /// it an advantage when the distribution is shifted from zero.
    #[test]
    fn quantize_q4_1_lower_rmse_than_q4_0_on_asymmetric() {
        let row: Vec<f32> = (0..QK4_0)
            .map(|i| 2.0 * (i as f32) / ((QK4_0 - 1) as f32))
            .collect();
        let mut q40 = vec![BlockQ4_0 {
            d_bits: 0,
            qs: [0u8; QK4_0 / 2],
        }; 1];
        let mut q41 = vec![BlockQ4_1 {
            d_bits: 0,
            m_bits: 0,
            qs: [0u8; QK4_1 / 2],
        }; 1];
        quantize_row_q4_0(&row, &mut q40).unwrap();
        quantize_row_q4_1(&row, &mut q41).unwrap();
        let mut d40 = vec![0.0_f32; QK4_0];
        let mut d41 = vec![0.0_f32; QK4_1];
        dequantize_row_q4_0(&q40, &mut d40).unwrap();
        dequantize_row_q4_1(&q41, &mut d41).unwrap();
        let rmse = |a: &[f32], b: &[f32]| -> f64 {
            let mut sse = 0.0;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = (*x as f64) - (*y as f64);
                sse += d * d;
            }
            (sse / a.len() as f64).sqrt()
        };
        let r40 = rmse(&row, &d40);
        let r41 = rmse(&row, &d41);
        assert!(
            r41 <= r40,
            "Q4_1 RMSE {r41} should be ≤ Q4_0 RMSE {r40} on asymmetric input"
        );
    }

    /// Q4_1 flat-bytes round-trip.
    #[test]
    fn quantize_q4_1_to_bytes_round_trip() {
        let row: Vec<f32> = (0..2 * QK4_1)
            .map(|i| (i as f32) / 32.0)
            .collect();
        let bytes = quantize_row_q4_1_to_bytes(&row).unwrap();
        assert_eq!(bytes.len(), 2 * BLOCK_Q4_1_SIZE);
        let mut decoded = vec![0.0_f32; 2 * QK4_1];
        dequantize_row_q4_1_bytes(&bytes, &mut decoded).unwrap();
        let mut sse = 0.0_f64;
        for (a, b) in row.iter().zip(decoded.iter()) {
            let d = (*a as f64) - (*b as f64);
            sse += d * d;
        }
        let rmse = (sse / row.len() as f64).sqrt();
        assert!(rmse < 0.07, "Q4_1 bytes round-trip RMSE {rmse} > 0.07");
    }

    /// Q4_1 rejects misaligned input.
    #[test]
    fn quantize_q4_1_rejects_misaligned() {
        let bad = vec![0.0_f32; QK4_1 + 7];
        let mut blocks = vec![BlockQ4_1 {
            d_bits: 0,
            m_bits: 0,
            qs: [0u8; QK4_1 / 2],
        }; 2];
        let err = quantize_row_q4_1(&bad, &mut blocks).unwrap_err();
        match err {
            QLegacyError::NotBlockAligned { actual, qk } => {
                assert_eq!(actual, QK4_1 + 7);
                assert_eq!(qk, QK4_1);
            }
            _ => panic!("expected NotBlockAligned"),
        }
    }

    /// Q4_0 vs Q8_0 RMSE on the same input: Q8_0 (8-bit) should be
    /// strictly more accurate than Q4_0 (4-bit) on a smooth ramp.
    #[test]
    fn quantize_q4_0_lower_resolution_than_q8_0() {
        let row: Vec<f32> = (0..QK4_0)
            .map(|i| -2.0 + 4.0 * (i as f32) / ((QK4_0 - 1) as f32))
            .collect();
        let mut q4 = vec![BlockQ4_0 {
            d_bits: 0,
            qs: [0u8; QK4_0 / 2],
        }; 1];
        let mut q8 = vec![BlockQ8_0 {
            d_bits: 0,
            qs: [0i8; QK8_0],
        }; 1];
        quantize_row_q4_0(&row, &mut q4).unwrap();
        quantize_row_q8_0(&row, &mut q8).unwrap();
        let mut d4 = vec![0.0_f32; QK4_0];
        let mut d8 = vec![0.0_f32; QK8_0];
        dequantize_row_q4_0(&q4, &mut d4).unwrap();
        dequantize_row_q8_0(&q8, &mut d8).unwrap();
        let rmse = |a: &[f32], b: &[f32]| -> f64 {
            let mut sse = 0.0;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = (*x as f64) - (*y as f64);
                sse += d * d;
            }
            (sse / a.len() as f64).sqrt()
        };
        let r4 = rmse(&row, &d4);
        let r8 = rmse(&row, &d8);
        assert!(r8 < r4, "Q8_0 RMSE {r8} should be < Q4_0 RMSE {r4}");
    }
}
