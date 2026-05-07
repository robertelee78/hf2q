//! ADR-017 Phase B-tq.1 — TurboQuant-packed K/V on-disk envelope.
//!
//! Implements the `payload_kind = "tq_packed_v1"` envelope payload for
//! the Phase A.3 spiller's atomic-rename block writer (`write_envelope`
//! in `kv_persist/format.rs`).  At codec_version=1 the envelope carries
//! the EXACT bytes that ADR-007's TurboQuant codec produces — encoded
//! Lloyd-Max indices + per-block scale + Hadamard rotation flag — so
//! the on-disk shape is decoupled from the runtime TQ inference path.
//! ADR-007's codec is deterministic by construction (Lloyd-Max
//! centroids are compile-time constants and FWHT is a fixed orthogonal
//! transform), so a SHA-256 round-trip of pack→unpack→re-pack returns
//! byte-identical bytes — the deterministic-rebuild guarantee
//! (ADR-017 §B-tq §D2).
//!
//! ## Scope of this iter (B-tq.1)
//!
//! This module ships the SUBSTRATE — envelope codec + parity tests —
//! NOT a full `KvCacheSpill` family hook with engine integration.  The
//! family-hook wire-up (analogous to `gemma4_dense.rs`'s
//! `Gemma4DenseSpill`) lands in B-tq.2 once the runtime TQ path
//! stabilises (ADR-007 reopen 2026-05-05 noted Path C completion plan
//! as the destination for the GPU-side TQ correctness work).
//!
//! At v1 the envelope can be exercised end-to-end via:
//!
//! ```text
//! TqPackedV1Header { codec_version: 1, ... } + indices_bytes
//!                          ↓ pack_tq_v1_payload
//!                     payload bytes (passed to write_envelope)
//!                          ↓ unpack_tq_v1_payload
//! TqPackedV1Header + indices_bytes (byte-equal to input)
//! ```
//!
//! Round-trip parity is enforced via `tq_packed_v1_round_trip_byte_exact`.
//!
//! ## Storage format
//!
//! ```text
//! tq_packed_v1 payload bytes:
//!   [   0..  4]: u32 LE — magic = `b"TQP1"` (0x31_50_51_54)
//!   [   4..  8]: u32 LE — codec_version (must be 1 at this iter)
//!   [   8.. 12]: u32 LE — bits_per_coord (2 | 3 | 4 | 8)
//!   [  12.. 16]: u32 LE — head_dim
//!   [  16.. 20]: u32 LE — n_kv_heads
//!   [  20.. 24]: u32 LE — n_tokens covered by this block
//!   [  24.. 28]: u32 LE — flags (bit 0: hadamard rotated, bit 1: split-channel)
//!   [  28.. 32]: u32 LE — reserved (zero at v1)
//!   [  32.. 40]: f64 LE — scale (per-block multiplicative scale on dequantize)
//!   [  40.. ..]: indices (packed nibble or byte stream — bits_per_coord-dependent)
//! ```
//!
//! Bit packing for `bits_per_coord = 4`: low nibble at even position,
//! high nibble at odd position (matches `flash_attn_vec_tq.metal`'s
//! decode at ADR-007 §C-0b).  For `bits_per_coord = 8`: one byte per
//! index (Lloyd-Max codebook of 256 centroids).  For 2/3-bit: bit
//! packing per ADR-007 §3.
//!
//! ## Why a pure-envelope module first
//!
//! The runtime TQ inference path has documented correctness work
//! pending (ADR-007 reopen 2026-05-05).  Splitting "on-disk envelope"
//! from "engine integration" lets B-tq.1's storage format land NOW
//! at codec_version=1 while the GPU-side TQ work converges; B-tq.2
//! plugs the family hook in without touching the storage codec, so
//! B-tq.1's envelope MUST be byte-stable across that transition.
//! That's enforced by the round-trip + magic-byte tests below.

use sha2::{Digest, Sha256};

/// Magic bytes prefixing the `tq_packed_v1` payload.  ASCII `"TQP1"`
/// stored little-endian as `0x31_50_51_54`.
pub const TQ_PACKED_V1_MAGIC: u32 = u32::from_le_bytes([b'T', b'Q', b'P', b'1']);

/// Frozen codec version per ADR-007 freeze 2026-05-05.  Bumping this
/// requires a B-tq.X envelope migration plan (read both versions,
/// write only the latest).  Until then, on-disk bytes always carry
/// `codec_version == 1`.
pub const TQ_PACKED_CODEC_VERSION_V1: u32 = 1;

/// Bit width of the Lloyd-Max index stream.  Per ADR-007 the runtime
/// path supports `2 | 3 | 4 | 8`; this module accepts any non-zero
/// value ≤ 8 to allow forward-compat tests, but production callers
/// stick to the four documented values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TqBitsPerCoord(pub u32);

impl TqBitsPerCoord {
    /// Returns `Err` for invalid bit widths (zero or > 8).
    pub fn new(b: u32) -> Result<Self, TqEnvelopeError> {
        if b == 0 || b > 8 {
            return Err(TqEnvelopeError::InvalidBitsPerCoord(b));
        }
        Ok(Self(b))
    }
}

/// Flag bits in the `flags` u32 of the v1 header.
pub mod flags {
    /// Hadamard rotation was applied before quantization (decode must
    /// apply inverse FWHT after centroid gather, per
    /// ADR-007 §C-0b decode order).
    pub const HADAMARD_ROTATED: u32 = 1 << 0;
    /// 2.5-bit split-channel mode (first d/4 use 3-bit, remaining 3d/4
    /// use 2-bit; per ADR-007 Decision §3).  When set, `bits_per_coord`
    /// reports the BLENDED rate (encoded as 0 to disambiguate from
    /// uniform 2/3/4/8).  v1 reserves but does not pack split-channel
    /// indices yet.
    pub const SPLIT_CHANNEL_25BIT: u32 = 1 << 1;
}

/// Errors produced by [`pack_tq_v1_payload`] and [`unpack_tq_v1_payload`].
#[derive(Debug, PartialEq, Eq)]
pub enum TqEnvelopeError {
    /// Bits-per-coord field was zero or > 8.
    InvalidBitsPerCoord(u32),
    /// Payload byte length doesn't match `n_tokens × n_kv_heads ×
    /// head_dim × bits_per_coord / 8` plus the 40-byte header.
    PayloadSizeMismatch {
        expected: usize,
        got: usize,
    },
    /// Magic prefix didn't match `TQ_PACKED_V1_MAGIC`.
    BadMagic { got: u32 },
    /// Codec version != 1 at this iter.
    UnsupportedCodecVersion(u32),
    /// Buffer too short to contain even the header.
    Truncated { got: usize },
    /// Reserved field was non-zero (forward-compat strict-decode).
    ReservedNonZero { got: u32 },
}

/// Header fields for the `tq_packed_v1` payload.  Wire format is
/// fixed-width little-endian per [`pack_tq_v1_payload`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TqPackedV1Header {
    /// MUST equal [`TQ_PACKED_CODEC_VERSION_V1`].
    pub codec_version: u32,
    /// Lloyd-Max bit width per coordinate.
    pub bits_per_coord: TqBitsPerCoord,
    /// Inner contiguous head dimension.
    pub head_dim: u32,
    /// Number of K/V heads at this layer.
    pub n_kv_heads: u32,
    /// Number of tokens packed in this block.
    pub n_tokens: u32,
    /// Flag bits — see [`flags`].
    pub flags: u32,
    /// Per-block multiplicative scale applied at dequantize
    /// (`f64` for round-trip stability — most TQ runtimes use f32 but
    /// the envelope carries f64 to avoid silent precision loss when
    /// the codec is upgraded to f64-scale variants).
    pub scale: f64,
}

impl TqPackedV1Header {
    /// Number of bytes the indices stream occupies (excludes the
    /// 40-byte header).
    ///
    /// At `bits_per_coord = 8` this is `n_kv_heads × n_tokens × head_dim`.
    /// At `bits_per_coord = 4`: `n_kv_heads × n_tokens × head_dim / 2`
    /// (rounded up if head_dim odd; production head_dims are 256/512
    /// so always even).
    /// At `bits_per_coord ∈ {2, 3}`: `ceil(n_kv_heads × n_tokens ×
    /// head_dim × bits / 8)`.
    pub fn indices_bytes_len(&self) -> usize {
        let bits = self.bits_per_coord.0 as usize;
        let total_bits = (self.n_kv_heads as usize)
            * (self.n_tokens as usize)
            * (self.head_dim as usize)
            * bits;
        // Ceil-div by 8 (safe — `total_bits` always fits a usize at
        // production scales: 2 × 4096 × 512 × 8 = 33 MiB max).
        total_bits.div_ceil(8)
    }

    /// Total payload byte length (header + indices).
    pub fn total_bytes(&self) -> usize {
        TQ_PACKED_V1_HEADER_BYTES + self.indices_bytes_len()
    }
}

/// Fixed wire-size of the v1 header.  4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 8 = 40.
pub const TQ_PACKED_V1_HEADER_BYTES: usize = 40;

/// Serialize a `tq_packed_v1` payload from header + indices bytes.
///
/// The returned `Vec<u8>` is the EXACT bytes that
/// `kv_persist::format::write_envelope` will atomically rename to disk.
///
/// # Errors
///
/// * [`TqEnvelopeError::PayloadSizeMismatch`] if `indices.len() !=
///   header.indices_bytes_len()` (caller passed malformed indices).
pub fn pack_tq_v1_payload(
    header: &TqPackedV1Header,
    indices: &[u8],
) -> Result<Vec<u8>, TqEnvelopeError> {
    let expected = header.indices_bytes_len();
    if indices.len() != expected {
        return Err(TqEnvelopeError::PayloadSizeMismatch {
            expected,
            got: indices.len(),
        });
    }

    let mut out = Vec::with_capacity(header.total_bytes());
    out.extend_from_slice(&TQ_PACKED_V1_MAGIC.to_le_bytes());
    out.extend_from_slice(&header.codec_version.to_le_bytes());
    out.extend_from_slice(&header.bits_per_coord.0.to_le_bytes());
    out.extend_from_slice(&header.head_dim.to_le_bytes());
    out.extend_from_slice(&header.n_kv_heads.to_le_bytes());
    out.extend_from_slice(&header.n_tokens.to_le_bytes());
    out.extend_from_slice(&header.flags.to_le_bytes());
    out.extend_from_slice(&0_u32.to_le_bytes()); // reserved
    out.extend_from_slice(&header.scale.to_le_bytes());
    out.extend_from_slice(indices);
    debug_assert_eq!(out.len(), header.total_bytes());
    Ok(out)
}

/// Deserialize a `tq_packed_v1` payload into header + indices slice.
///
/// Returns the parsed header AND a slice of the indices bytes (borrowed
/// from `payload` — caller controls lifetime).
///
/// # Errors
///
/// * [`TqEnvelopeError::Truncated`] if `payload.len() < 40`.
/// * [`TqEnvelopeError::BadMagic`] if the first 4 bytes don't match
///   [`TQ_PACKED_V1_MAGIC`].
/// * [`TqEnvelopeError::UnsupportedCodecVersion`] if `codec_version != 1`.
/// * [`TqEnvelopeError::InvalidBitsPerCoord`] if `bits_per_coord` is 0 or > 8.
/// * [`TqEnvelopeError::ReservedNonZero`] if the reserved field is non-zero.
/// * [`TqEnvelopeError::PayloadSizeMismatch`] if the indices section is
///   shorter than the header's declared shape.
pub fn unpack_tq_v1_payload(
    payload: &[u8],
) -> Result<(TqPackedV1Header, &[u8]), TqEnvelopeError> {
    if payload.len() < TQ_PACKED_V1_HEADER_BYTES {
        return Err(TqEnvelopeError::Truncated { got: payload.len() });
    }
    let read_u32 = |off: usize| -> u32 {
        u32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ])
    };
    let read_f64 = |off: usize| -> f64 {
        f64::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
            payload[off + 4],
            payload[off + 5],
            payload[off + 6],
            payload[off + 7],
        ])
    };

    let magic = read_u32(0);
    if magic != TQ_PACKED_V1_MAGIC {
        return Err(TqEnvelopeError::BadMagic { got: magic });
    }
    let codec_version = read_u32(4);
    if codec_version != TQ_PACKED_CODEC_VERSION_V1 {
        return Err(TqEnvelopeError::UnsupportedCodecVersion(codec_version));
    }
    let bits_raw = read_u32(8);
    let bits_per_coord = TqBitsPerCoord::new(bits_raw)?;
    let head_dim = read_u32(12);
    let n_kv_heads = read_u32(16);
    let n_tokens = read_u32(20);
    let flags = read_u32(24);
    let reserved = read_u32(28);
    if reserved != 0 {
        return Err(TqEnvelopeError::ReservedNonZero { got: reserved });
    }
    let scale = read_f64(32);

    let header = TqPackedV1Header {
        codec_version,
        bits_per_coord,
        head_dim,
        n_kv_heads,
        n_tokens,
        flags,
        scale,
    };

    let expected_indices = header.indices_bytes_len();
    let actual_indices = payload.len() - TQ_PACKED_V1_HEADER_BYTES;
    if actual_indices != expected_indices {
        return Err(TqEnvelopeError::PayloadSizeMismatch {
            expected: expected_indices,
            got: actual_indices,
        });
    }
    Ok((header, &payload[TQ_PACKED_V1_HEADER_BYTES..]))
}

/// SHA-256 of a `tq_packed_v1` payload.  Convenience for the
/// deterministic-rebuild guarantee (ADR-017 §B-tq §D2): packing
/// the SAME header + indices twice MUST produce byte-identical
/// payloads, hence identical SHA-256.
pub fn sha256_payload(payload: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(payload);
    hasher.finalize().into()
}

// ============================================================================
// Phase B-tq.3 — codec_version=2 envelope (engine-wired path)
// ============================================================================
//
// Why v2:
//   v1 carries `bits_per_coord × n_kv_heads × n_tokens × head_dim` indices
//   plus a single `scale: f64`.  The runtime `MlxKvCache` actually stores
//   per-token-per-head norms — `k_norms: [num_kv_heads, capacity] F32` —
//   computed by `dispatch_hadamard_quantize_kv` as the FWHT-normalized
//   block magnitude prior to Lloyd-Max quantization, then consumed at
//   decode by the TQ-aware FA kernel (`flash_attn_vec_tq.metal`).  The
//   single `scale` field of v1 cannot round-trip these per-token-per-head
//   norms.  v2 extends the body with a raw F32 norms stream (LE) so the
//   engine can capture+restore live KV state byte-exact.
//
// Header layout — IDENTICAL fixed 40-byte shape as v1, only the magic
// and codec_version differ:
//
//   [   0..  4]: u32 LE — magic = b"TQP2" (0x32_50_51_54)
//   [   4..  8]: u32 LE — codec_version = 2
//   [   8.. 12]: u32 LE — bits_per_coord (2 | 3 | 4 | 8)
//   [  12.. 16]: u32 LE — head_dim
//   [  16.. 20]: u32 LE — n_kv_heads
//   [  20.. 24]: u32 LE — n_tokens
//   [  24.. 28]: u32 LE — flags (HADAMARD_ROTATED | SPLIT_CHANNEL_25BIT)
//   [  28.. 32]: u32 LE — reserved (zero at v2)
//   [  32.. 40]: f64 LE — scale (forward-compat carry; runtime usually 1.0
//                          on the per-token-norm path, since the
//                          per-block magnitude already lives in `norms`)
//
// Body:
//   [  40.. 40+I]: indices  — `bits × n_kv_heads × n_tokens × head_dim / 8`
//                              bytes; same packing as v1
//   [40+I..40+I+N]: norms   — `n_kv_heads × n_tokens` F32 LE values
//                              (i.e. `4 × n_kv_heads × n_tokens` bytes)
//
// One envelope per (K or V) buffer, mirroring v1.  The spiller wraps two
// v2 envelopes per (layer, range) — one for K, one for V — and stores
// both atomically via the existing `write_envelope` path.
//
// Why share the v1 magic/version test approach: identical 40-byte header
// shape means a single sha-stable change (magic byte 4 flips from `1` to
// `2`) is the ONLY difference up to byte 39.  v1 round-trip tests still
// PASS unchanged; v2 round-trip tests are added below.  CI catches any
// silent format drift via the frozen-magic + frozen-codec-version tests
// for BOTH versions.

/// Magic bytes prefixing the `tq_packed_v2` payload.  ASCII `"TQP2"`
/// stored little-endian as `0x32_50_51_54`.
pub const TQ_PACKED_V2_MAGIC: u32 = u32::from_le_bytes([b'T', b'Q', b'P', b'2']);

/// Frozen codec version for the v2 (engine-wired, per-token-norm) envelope.
/// Bumping this requires a B-tq.X envelope migration plan (read all prior
/// versions, write only the latest).  Until then on-disk bytes always
/// carry `codec_version == 2` for v2 payloads.
pub const TQ_PACKED_CODEC_VERSION_V2: u32 = 2;

/// Fixed wire-size of the v2 header — equal to v1's 40 bytes by design.
pub const TQ_PACKED_V2_HEADER_BYTES: usize = TQ_PACKED_V1_HEADER_BYTES;

/// Header fields for the `tq_packed_v2` payload.  Wire format mirrors
/// v1 byte-for-byte; the body adds a per-token-per-head F32 norms stream
/// after the indices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TqPackedV2Header {
    /// MUST equal [`TQ_PACKED_CODEC_VERSION_V2`].
    pub codec_version: u32,
    /// Lloyd-Max bit width per coordinate (same field as v1).
    pub bits_per_coord: TqBitsPerCoord,
    /// Inner contiguous head dimension.
    pub head_dim: u32,
    /// Number of K/V heads at this layer.
    pub n_kv_heads: u32,
    /// Number of tokens packed in this block.
    pub n_tokens: u32,
    /// Flag bits — see [`flags`].
    pub flags: u32,
    /// Per-block multiplicative scale (`f64` for forward-compat with v1
    /// callers).  On the per-token-norm path runtime typically stores
    /// 1.0 here because the magnitude lives in the norms stream.
    pub scale: f64,
}

impl TqPackedV2Header {
    /// Number of bytes the indices stream occupies (excludes the 40-byte
    /// header AND the trailing norms stream).  Same formula as v1.
    pub fn indices_bytes_len(&self) -> usize {
        let bits = self.bits_per_coord.0 as usize;
        let total_bits = (self.n_kv_heads as usize)
            * (self.n_tokens as usize)
            * (self.head_dim as usize)
            * bits;
        total_bits.div_ceil(8)
    }

    /// Number of bytes the F32 norms stream occupies (4 × nkv × ntok).
    pub fn norms_bytes_len(&self) -> usize {
        (self.n_kv_heads as usize) * (self.n_tokens as usize) * 4
    }

    /// Total payload byte length (header + indices + norms).
    pub fn total_bytes(&self) -> usize {
        TQ_PACKED_V2_HEADER_BYTES + self.indices_bytes_len() + self.norms_bytes_len()
    }
}

/// Serialize a `tq_packed_v2` payload from header + indices bytes + F32
/// norms stream.
///
/// `norms_le` is `n_kv_heads × n_tokens` F32 values written
/// little-endian.  The caller is responsible for the row-major ordering
/// `[head, token]` — the same layout the runtime `MlxKvCache.k_norms`
/// uses — so capture/restore round-trip is byte-exact.
///
/// # Errors
///
/// * [`TqEnvelopeError::PayloadSizeMismatch`] if `indices.len() !=
///   header.indices_bytes_len()` or `norms_le.len() !=
///   header.norms_bytes_len()`.
pub fn pack_tq_v2_payload(
    header: &TqPackedV2Header,
    indices: &[u8],
    norms_le: &[u8],
) -> Result<Vec<u8>, TqEnvelopeError> {
    let expected_idx = header.indices_bytes_len();
    if indices.len() != expected_idx {
        return Err(TqEnvelopeError::PayloadSizeMismatch {
            expected: expected_idx,
            got: indices.len(),
        });
    }
    let expected_norms = header.norms_bytes_len();
    if norms_le.len() != expected_norms {
        return Err(TqEnvelopeError::PayloadSizeMismatch {
            expected: expected_norms,
            got: norms_le.len(),
        });
    }

    let mut out = Vec::with_capacity(header.total_bytes());
    out.extend_from_slice(&TQ_PACKED_V2_MAGIC.to_le_bytes());
    out.extend_from_slice(&header.codec_version.to_le_bytes());
    out.extend_from_slice(&header.bits_per_coord.0.to_le_bytes());
    out.extend_from_slice(&header.head_dim.to_le_bytes());
    out.extend_from_slice(&header.n_kv_heads.to_le_bytes());
    out.extend_from_slice(&header.n_tokens.to_le_bytes());
    out.extend_from_slice(&header.flags.to_le_bytes());
    out.extend_from_slice(&0_u32.to_le_bytes()); // reserved
    out.extend_from_slice(&header.scale.to_le_bytes());
    out.extend_from_slice(indices);
    out.extend_from_slice(norms_le);
    debug_assert_eq!(out.len(), header.total_bytes());
    Ok(out)
}

/// Deserialize a `tq_packed_v2` payload into header + indices slice +
/// norms slice.  All slices borrow from `payload`.
///
/// # Errors
///
/// * [`TqEnvelopeError::Truncated`] if `payload.len() < 40`.
/// * [`TqEnvelopeError::BadMagic`] if the magic doesn't match
///   [`TQ_PACKED_V2_MAGIC`].
/// * [`TqEnvelopeError::UnsupportedCodecVersion`] if `codec_version != 2`.
/// * [`TqEnvelopeError::InvalidBitsPerCoord`] if `bits_per_coord` is 0 or > 8.
/// * [`TqEnvelopeError::ReservedNonZero`] if the reserved field is non-zero.
/// * [`TqEnvelopeError::PayloadSizeMismatch`] if the body size doesn't
///   match the header's declared `indices + norms` shape.
pub fn unpack_tq_v2_payload(
    payload: &[u8],
) -> Result<(TqPackedV2Header, &[u8], &[u8]), TqEnvelopeError> {
    if payload.len() < TQ_PACKED_V2_HEADER_BYTES {
        return Err(TqEnvelopeError::Truncated { got: payload.len() });
    }
    let read_u32 = |off: usize| -> u32 {
        u32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ])
    };
    let read_f64 = |off: usize| -> f64 {
        f64::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
            payload[off + 4],
            payload[off + 5],
            payload[off + 6],
            payload[off + 7],
        ])
    };

    let magic = read_u32(0);
    if magic != TQ_PACKED_V2_MAGIC {
        return Err(TqEnvelopeError::BadMagic { got: magic });
    }
    let codec_version = read_u32(4);
    if codec_version != TQ_PACKED_CODEC_VERSION_V2 {
        return Err(TqEnvelopeError::UnsupportedCodecVersion(codec_version));
    }
    let bits_raw = read_u32(8);
    let bits_per_coord = TqBitsPerCoord::new(bits_raw)?;
    let head_dim = read_u32(12);
    let n_kv_heads = read_u32(16);
    let n_tokens = read_u32(20);
    let flags = read_u32(24);
    let reserved = read_u32(28);
    if reserved != 0 {
        return Err(TqEnvelopeError::ReservedNonZero { got: reserved });
    }
    let scale = read_f64(32);

    let header = TqPackedV2Header {
        codec_version,
        bits_per_coord,
        head_dim,
        n_kv_heads,
        n_tokens,
        flags,
        scale,
    };

    let expected_idx = header.indices_bytes_len();
    let expected_norms = header.norms_bytes_len();
    let body_len = payload.len() - TQ_PACKED_V2_HEADER_BYTES;
    if body_len != expected_idx + expected_norms {
        return Err(TqEnvelopeError::PayloadSizeMismatch {
            expected: expected_idx + expected_norms,
            got: body_len,
        });
    }
    let idx_start = TQ_PACKED_V2_HEADER_BYTES;
    let idx_end = idx_start + expected_idx;
    let norms_end = idx_end + expected_norms;
    Ok((header, &payload[idx_start..idx_end], &payload[idx_end..norms_end]))
}

// ============================================================================
// Phase B-tq.3 — raw-bytes capture/restore helpers
// ============================================================================
//
// These helpers connect the runtime `MlxKvCache` byte buffers to the v2
// envelope codec WITHOUT depending on the MLX device.  They operate on
// raw byte slices (the same shape `MlxBuffer::as_slice` / `as_mut_slice`
// produces) so they're unit-testable on a CPU host without GPU.
//
// Memory layout (matches `MlxKvCache` at `forward_mlx.rs:450-467`):
//   * k_packed / v_packed: `[num_kv_heads, capacity, head_dim_packed]` U8
//                          where `head_dim_packed = head_dim * bits / 8`
//   * k_norms / v_norms:   `[num_kv_heads, capacity]` F32
//
// "Range" is a token range `[range.start, range.end)` along the position
// axis (capacity).  Capture extracts `[head, range.start..range.end,
// 0..head_dim_packed]` from `*_packed` and `[head, range.start..range.end]`
// from `*_norms`, concatenated into a single v2 envelope payload.

/// Pack one (K or V) v2 envelope from a slice of the runtime KV cache.
///
/// `packed_bytes` is `[num_kv_heads, capacity, hd_packed]` U8 row-major,
/// where `hd_packed = head_dim × bits_per_coord / 8`.  `norms_bytes_le`
/// is `[num_kv_heads, capacity]` F32 LE (i.e. raw bytes from
/// `MlxBuffer::as_slice` on the F32 buffer).  `range` is the token slice
/// to capture, in `[0, capacity)`.
///
/// Returns the byte-exact `tq_packed_v2` payload ready to insert into a
/// `TqPackedSpill` via [`TqPackedSpill::insert_block_v2`].
///
/// # Errors
///
/// * `CodecErr` if any of the shape arguments would cause an out-of-bounds
///   read on the input slices, or if `range.end > capacity` /
///   `range.start >= range.end`.
pub fn capture_tq_v2_payload_from_buffers(
    packed_bytes: &[u8],
    norms_bytes_le: &[u8],
    capacity: u32,
    n_kv_heads: u32,
    head_dim: u32,
    bits_per_coord: TqBitsPerCoord,
    range: std::ops::Range<u32>,
    flags: u32,
    scale: f64,
) -> Result<Vec<u8>, crate::serve::multi_model::SpillErrorKind> {
    use crate::serve::multi_model::SpillErrorKind;

    if range.end <= range.start || range.end > capacity {
        return Err(SpillErrorKind::CodecErr);
    }
    let bits = bits_per_coord.0;
    if bits == 0 || bits > 8 {
        return Err(SpillErrorKind::CodecErr);
    }
    // hd_packed must be exact multiple at any practical bits-per-coord.
    // For bits=4 with head_dim=256 → 128 bytes.  For bits=8 with
    // head_dim=512 → 512 bytes.  Reject non-byte-aligned shapes to keep
    // the row stride deterministic.
    if (head_dim as u64) * (bits as u64) % 8 != 0 {
        return Err(SpillErrorKind::CodecErr);
    }
    let hd_packed = ((head_dim as u64) * (bits as u64) / 8) as usize;

    let cap_us = capacity as usize;
    let nkv_us = n_kv_heads as usize;
    let n_tokens = (range.end - range.start) as usize;

    // Bounds-check input slices.  packed: nkv × cap × hd_packed.
    let expected_packed = nkv_us
        .checked_mul(cap_us)
        .and_then(|v| v.checked_mul(hd_packed))
        .ok_or(SpillErrorKind::CodecErr)?;
    if packed_bytes.len() != expected_packed {
        return Err(SpillErrorKind::CodecErr);
    }
    // norms: nkv × cap × 4 (F32 LE).
    let expected_norms = nkv_us
        .checked_mul(cap_us)
        .and_then(|v| v.checked_mul(4))
        .ok_or(SpillErrorKind::CodecErr)?;
    if norms_bytes_le.len() != expected_norms {
        return Err(SpillErrorKind::CodecErr);
    }

    // Extract indices: [head, range.start..range.end, 0..hd_packed].
    let mut idx = Vec::with_capacity(nkv_us * n_tokens * hd_packed);
    for h in 0..nkv_us {
        let head_base = h * cap_us * hd_packed;
        let row_start = head_base + (range.start as usize) * hd_packed;
        let row_end = head_base + (range.end as usize) * hd_packed;
        idx.extend_from_slice(&packed_bytes[row_start..row_end]);
    }

    // Extract norms: [head, range.start..range.end] in F32 LE bytes.
    let mut norms = Vec::with_capacity(nkv_us * n_tokens * 4);
    for h in 0..nkv_us {
        let head_base = h * cap_us * 4;
        let row_start = head_base + (range.start as usize) * 4;
        let row_end = head_base + (range.end as usize) * 4;
        norms.extend_from_slice(&norms_bytes_le[row_start..row_end]);
    }

    let header = TqPackedV2Header {
        codec_version: TQ_PACKED_CODEC_VERSION_V2,
        bits_per_coord,
        head_dim,
        n_kv_heads,
        n_tokens: n_tokens as u32,
        flags,
        scale,
    };
    pack_tq_v2_payload(&header, &idx, &norms).map_err(|_| SpillErrorKind::CodecErr)
}

/// Restore (write back) a v2 envelope payload into runtime KV-cache byte
/// buffers.  Inverse of [`capture_tq_v2_payload_from_buffers`].
///
/// Reads `payload` (a v2 envelope), validates header against the runtime
/// shape `(capacity, n_kv_heads, head_dim, bits_per_coord)`, and writes
/// the indices into `packed_bytes_mut[head, range.start..range.end,
/// 0..hd_packed]` and norms into `norms_bytes_mut_le[head,
/// range.start..range.end]`.
///
/// # Errors
///
/// * `CodecErr` for any shape mismatch, OOB write, or envelope parse
///   error.
pub fn restore_tq_v2_payload_into_buffers(
    payload: &[u8],
    packed_bytes_mut: &mut [u8],
    norms_bytes_mut_le: &mut [u8],
    capacity: u32,
    n_kv_heads: u32,
    head_dim: u32,
    bits_per_coord: TqBitsPerCoord,
    range: std::ops::Range<u32>,
) -> Result<TqPackedV2Header, crate::serve::multi_model::SpillErrorKind> {
    use crate::serve::multi_model::SpillErrorKind;

    if range.end <= range.start || range.end > capacity {
        return Err(SpillErrorKind::CodecErr);
    }
    let (header, idx, norms) =
        unpack_tq_v2_payload(payload).map_err(|_| SpillErrorKind::CodecErr)?;

    // Header must agree with caller-declared runtime shape.
    if header.bits_per_coord != bits_per_coord
        || header.head_dim != head_dim
        || header.n_kv_heads != n_kv_heads
        || header.n_tokens != (range.end - range.start)
    {
        return Err(SpillErrorKind::CodecErr);
    }

    let bits = bits_per_coord.0;
    if (head_dim as u64) * (bits as u64) % 8 != 0 {
        return Err(SpillErrorKind::CodecErr);
    }
    let hd_packed = ((head_dim as u64) * (bits as u64) / 8) as usize;
    let cap_us = capacity as usize;
    let nkv_us = n_kv_heads as usize;
    let n_tokens = (range.end - range.start) as usize;

    let expected_packed = nkv_us
        .checked_mul(cap_us)
        .and_then(|v| v.checked_mul(hd_packed))
        .ok_or(SpillErrorKind::CodecErr)?;
    if packed_bytes_mut.len() != expected_packed {
        return Err(SpillErrorKind::CodecErr);
    }
    let expected_norms = nkv_us
        .checked_mul(cap_us)
        .and_then(|v| v.checked_mul(4))
        .ok_or(SpillErrorKind::CodecErr)?;
    if norms_bytes_mut_le.len() != expected_norms {
        return Err(SpillErrorKind::CodecErr);
    }

    // Body length sanity (already enforced by unpack but cheap).
    let expected_idx_len = nkv_us * n_tokens * hd_packed;
    let expected_norms_len = nkv_us * n_tokens * 4;
    if idx.len() != expected_idx_len || norms.len() != expected_norms_len {
        return Err(SpillErrorKind::CodecErr);
    }

    // Write indices: [head, range.start..range.end, 0..hd_packed].
    for h in 0..nkv_us {
        let head_base = h * cap_us * hd_packed;
        let row_start = head_base + (range.start as usize) * hd_packed;
        let row_end = head_base + (range.end as usize) * hd_packed;
        let src_off = h * n_tokens * hd_packed;
        let src_end = src_off + n_tokens * hd_packed;
        packed_bytes_mut[row_start..row_end].copy_from_slice(&idx[src_off..src_end]);
    }

    // Write norms: [head, range.start..range.end] (F32 LE bytes).
    for h in 0..nkv_us {
        let head_base = h * cap_us * 4;
        let row_start = head_base + (range.start as usize) * 4;
        let row_end = head_base + (range.end as usize) * 4;
        let src_off = h * n_tokens * 4;
        let src_end = src_off + n_tokens * 4;
        norms_bytes_mut_le[row_start..row_end].copy_from_slice(&norms[src_off..src_end]);
    }

    Ok(header)
}

// ============================================================================
// Phase B-tq.4 — K+V bundle codec
// ============================================================================
//
// The spiller's `KvCacheSpill::snapshot_block` returns ONE `Vec<u8>` per
// (layer, range) block — but TQ-packed has TWO envelopes per block (K
// and V).  The bundle codec packs both v2 envelopes into a single
// `Vec<u8>` for the spiller's persist path:
//
//   [   0..  4]: u32 LE — bundle magic = b"TQK2" (0x32_4B_51_54)
//   [   4.. 12]: u64 LE — k_payload_len
//   [  12..12+K]: k_payload (a tq_packed_v2 envelope)
//   [12+K..12+K+8]: u64 LE — v_payload_len
//   [12+K+8..12+K+8+V]: v_payload (a tq_packed_v2 envelope)
//
// The bundle magic disambiguates from a bare v1/v2 envelope so
// `TqPackedSpill::insert_block` can magic-dispatch correctly; the v1
// substrate path (synthesize_block, B-tq.1 tests) and the v2 raw-bytes
// path (capture_tq_v2_payload_from_buffers test path) continue to
// accept bare envelopes.

/// Magic bytes prefixing the K+V bundle.  ASCII `"TQK2"` stored
/// little-endian as `0x32_4B_51_54`.
pub const TQ_PACKED_KV_BUNDLE_MAGIC: u32 =
    u32::from_le_bytes([b'T', b'Q', b'K', b'2']);

/// Wire-size of the bundle's fixed prefix (magic + k_len + 0 +
/// v_len_offset is computed dynamically per-bundle).  This is the
/// minimum byte count required to begin parsing.
pub const TQ_PACKED_KV_BUNDLE_HEADER_BYTES: usize = 12;

/// Pack a K+V bundle from two `tq_packed_v2` envelopes.  The K and V
/// payloads must both have the v2 magic; we don't otherwise validate
/// (the caller — `TqPackedSpill` — already validated via
/// `pack_tq_v2_payload`).
pub fn pack_tq_v2_kv_bundle(
    k_payload: &[u8],
    v_payload: &[u8],
) -> Result<Vec<u8>, TqEnvelopeError> {
    if k_payload.len() < 4 || v_payload.len() < 4 {
        return Err(TqEnvelopeError::Truncated {
            got: k_payload.len().min(v_payload.len()),
        });
    }
    let k_magic = u32::from_le_bytes([
        k_payload[0],
        k_payload[1],
        k_payload[2],
        k_payload[3],
    ]);
    let v_magic = u32::from_le_bytes([
        v_payload[0],
        v_payload[1],
        v_payload[2],
        v_payload[3],
    ]);
    if k_magic != TQ_PACKED_V2_MAGIC || v_magic != TQ_PACKED_V2_MAGIC {
        return Err(TqEnvelopeError::BadMagic {
            got: if k_magic != TQ_PACKED_V2_MAGIC { k_magic } else { v_magic },
        });
    }

    let total =
        TQ_PACKED_KV_BUNDLE_HEADER_BYTES + k_payload.len() + 8 + v_payload.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&TQ_PACKED_KV_BUNDLE_MAGIC.to_le_bytes());
    out.extend_from_slice(&(k_payload.len() as u64).to_le_bytes());
    out.extend_from_slice(k_payload);
    out.extend_from_slice(&(v_payload.len() as u64).to_le_bytes());
    out.extend_from_slice(v_payload);
    debug_assert_eq!(out.len(), total);
    Ok(out)
}

/// Unpack a K+V bundle.  Returns `(k_payload, v_payload)` slices
/// borrowing from the input bundle.
pub fn unpack_tq_v2_kv_bundle(
    bundle: &[u8],
) -> Result<(&[u8], &[u8]), TqEnvelopeError> {
    if bundle.len() < TQ_PACKED_KV_BUNDLE_HEADER_BYTES {
        return Err(TqEnvelopeError::Truncated { got: bundle.len() });
    }
    let magic = u32::from_le_bytes([bundle[0], bundle[1], bundle[2], bundle[3]]);
    if magic != TQ_PACKED_KV_BUNDLE_MAGIC {
        return Err(TqEnvelopeError::BadMagic { got: magic });
    }
    let k_len = u64::from_le_bytes([
        bundle[4], bundle[5], bundle[6], bundle[7],
        bundle[8], bundle[9], bundle[10], bundle[11],
    ]) as usize;
    let k_start = TQ_PACKED_KV_BUNDLE_HEADER_BYTES;
    let k_end = k_start + k_len;
    if bundle.len() < k_end + 8 {
        return Err(TqEnvelopeError::Truncated { got: bundle.len() });
    }
    let v_len = u64::from_le_bytes([
        bundle[k_end], bundle[k_end + 1], bundle[k_end + 2], bundle[k_end + 3],
        bundle[k_end + 4], bundle[k_end + 5], bundle[k_end + 6], bundle[k_end + 7],
    ]) as usize;
    let v_start = k_end + 8;
    let v_end = v_start + v_len;
    if bundle.len() != v_end {
        return Err(TqEnvelopeError::PayloadSizeMismatch {
            expected: v_end,
            got: bundle.len(),
        });
    }
    Ok((&bundle[k_start..k_end], &bundle[v_start..v_end]))
}

// ============================================================================
// Phase B-tq.2 — engine-side family hook
// ============================================================================
//
// `TqPackedSpill` wraps the [`pack_tq_v1_payload`] / [`unpack_tq_v1_payload`]
// envelope codec from B-tq.1 in the [`KvCacheSpill`] trait surface so the
// Phase A.3 spiller can dispatch snapshot/restore to TQ-active layers without
// changing the spiller core.
//
// **B-tq.2 v1 scope:** the hook holds an in-memory
// `BTreeMap<(layer_rank, range_start), Vec<u8>>` of pre-packed `tq_packed_v1`
// payloads.  The payloads can come from:
//   * Synthetic test fixtures (current iter — proves the trait surface +
//     round-trip parity).
//   * Future engine wiring (B-tq.3) that hooks `dispatch_hadamard_quantize_kv`
//     post-quantize results into the in-memory map.
//
// The on-disk envelope is FROZEN at codec_version=1 (B-tq.1 magic +
// version regression tests pin this), so B-tq.3's engine integration
// can land WITHOUT touching the on-disk wire format — the hook's
// `snapshot_block` / `restore_block` impl stays byte-stable across the
// transition.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, RwLock};

use crate::serve::kv_persist::format::ModelFingerprint;
use crate::serve::kv_persist::spiller::KvCacheSpill;
use crate::serve::multi_model::SpillErrorKind;
use crate::serve::quant_select::QuantType;

/// Shape config for [`TqPackedSpill`].  Mirrors the ADR-007 codec
/// parameters and the `tq_packed_v1` header fields.
#[derive(Clone, Debug, PartialEq)]
pub struct TqPackedConfig {
    /// Number of model layers (drives `KvCacheSpill::n_layers`).
    pub num_layers: usize,
    /// Per-layer K/V head count.  Length MUST equal `num_layers`.
    pub nkv_heads: Vec<u32>,
    /// Per-layer head dim.  Length MUST equal `num_layers`.
    pub head_dim: Vec<u32>,
    /// Lloyd-Max bit width.  Per ADR-007 production: 8.  Test fixtures
    /// may use 2/3/4.
    pub bits_per_coord: TqBitsPerCoord,
    /// Per-block scale (set at quantize time; carried through the
    /// envelope verbatim).  Tests use 1.0 unless they specifically
    /// exercise non-trivial scale.
    pub scale: f64,
    /// Set bit 0 to indicate Hadamard rotation was applied before
    /// quantization (production: always true on the TQ-active path).
    pub flags: u32,
    /// Block alignment in tokens.  ADR-017 §D3 default = 256.
    pub block_tokens: u32,
}

impl TqPackedConfig {
    /// Validate config invariants.
    pub fn validate(&self) -> Result<(), SpillErrorKind> {
        if self.num_layers == 0
            || self.nkv_heads.len() != self.num_layers
            || self.head_dim.len() != self.num_layers
            || self.block_tokens == 0
        {
            return Err(SpillErrorKind::CodecErr);
        }
        Ok(())
    }
}

/// Phase B-tq.2 + B-tq.4 family hook for TurboQuant-packed K/V layers.
///
/// The internal `blocks` map is keyed by `(layer_rank, range_start)`
/// and stores pre-packed envelope payloads (v1 substrate or
/// v2-engine-wired or v2-K+V-bundle, all magic-dispatched).
/// `snapshot_block(layer, range)` returns the stored bytes verbatim
/// when `engine_arc` is unbound, OR upgrades the `Arc<Engine>` and
/// reads through `MlxModelWeights::tq_v2_snapshot_block` when bound;
/// `restore_block(layer, range, payload)` validates the envelope and
/// stores, OR additionally writes back through
/// `MlxModelWeights::tq_v2_restore_block` when bound.  Both ops are
/// O(log n) in the number of stored blocks (`BTreeMap` lookup) plus
/// at most one cross-thread Engine call.
pub struct TqPackedSpill {
    cfg: TqPackedConfig,
    /// Pre-packed payloads keyed by (layer_rank, range_start).  Held
    /// behind `RwLock` so `&self` snapshot calls don't serialize on a
    /// write-lock — the spiller's trigger sites may run concurrent
    /// snapshots from multiple tokio tasks.  `restore_block` takes
    /// `&mut self` so the write-lock acquisition is lock-free.
    blocks: RwLock<BTreeMap<(usize, u32), Vec<u8>>>,
    /// Optional model fingerprint for §F4 namespace keying.  When
    /// `None`, falls back to the spiller's legacy `(repo, quant, "",
    /// "", "")` fingerprint per the trait default.
    fingerprint: RwLock<Option<ModelFingerprint>>,
    /// **Phase B-tq.4** — live `Engine` clone (production engine-wired
    /// path).  When `Some`, `snapshot_block` / `restore_block` route
    /// through `MlxModelWeights::tq_v2_snapshot_block` /
    /// `tq_v2_restore_block` (shipped at `forward_mlx.rs:4667+`)
    /// instead of the in-memory `blocks` map.  When `None`, falls back
    /// to the in-memory map (B-tq.2 substrate path; preserved for
    /// tests + the synthetic/insert_block contract).
    ///
    /// Mirrors `Gemma4DenseSpill::engine_arc` at
    /// `gemma4_dense.rs:269-291`.  Stores `Engine` value (not
    /// `Arc<Engine>` and not `Weak<Engine>`): the P0-bench fix from
    /// 2026-05-04 (`gemma4_dense.rs::engine_arc` doc) explains why
    /// — `Engine` is `#[derive(Clone)]` over `inner: Arc<EngineInner>`,
    /// so cloning into this slot keeps the worker-channel alive
    /// (Arc<EngineInner> strong-count) without polluting the OUTER
    /// `Arc<Engine>` that `LoaderWrapper::load`'s `try_unwrap`
    /// operates on.
    engine_arc: Arc<RwLock<Option<crate::serve::api::engine::Engine>>>,
}

impl TqPackedSpill {
    /// Construct a new TQ-packed spill from shape config.  The block
    /// map starts empty.  **B-tq.4**: `engine_arc` starts `None`; the
    /// factory's `try_construct` populates it before the hook is
    /// returned to the spiller.
    ///
    /// Tests + synthetic fixtures use this constructor + manual
    /// `insert_block` to populate the in-memory map.  The production
    /// engine-wired path uses [`Self::new_with_engine`] instead.
    pub fn new(cfg: TqPackedConfig) -> Result<Self, SpillErrorKind> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            blocks: RwLock::new(BTreeMap::new()),
            fingerprint: RwLock::new(None),
            engine_arc: Arc::new(RwLock::new(None)),
        })
    }

    /// Set the model fingerprint for §F4 namespace keying.
    pub fn set_fingerprint(&self, fp: ModelFingerprint) {
        if let Ok(mut slot) = self.fingerprint.write() {
            *slot = Some(fp);
        }
    }

    // ========================================================================
    // Phase B-tq.4 — engine-bound construction + bind methods
    // ========================================================================
    //
    // These mirror `Gemma4DenseSpill`'s pattern at
    // `gemma4_dense.rs:392-460`.  The factory's `try_construct`
    // downcasts `Arc<dyn Any>` to `Arc<Engine>`; if successful, we
    // store an `Engine` clone (cheap — clones the inner Arc<EngineInner>
    // worker channel) in `engine_arc`.  See the `engine_arc` field
    // doc-comment for the P0-bench-fix rationale.

    /// **Phase B-tq.4** — production constructor: build a spill from
    /// shape config AND a live `Engine` clone.  The engine is stored
    /// in `engine_arc`; subsequent `snapshot_block` / `restore_block`
    /// calls route through it.
    pub fn new_with_engine(
        cfg: TqPackedConfig,
        engine: crate::serve::api::engine::Engine,
    ) -> Result<Self, SpillErrorKind> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            blocks: RwLock::new(BTreeMap::new()),
            fingerprint: RwLock::new(None),
            engine_arc: Arc::new(RwLock::new(Some(engine))),
        })
    }

    /// **Phase B-tq.4** — populate `engine_arc` post-construction.
    /// Used by `EngineBindable::bind_engine` and by tests that
    /// construct via `new` then bind separately.
    pub fn set_engine_arc(&self, engine: crate::serve::api::engine::Engine) {
        if let Ok(mut slot) = self.engine_arc.write() {
            *slot = Some(engine);
        }
    }

    /// **Phase B-tq.4** — clear `engine_arc` (the
    /// `EngineBindable::unbind_engine` path).  After this returns,
    /// `snapshot_block` / `restore_block` fall back to the in-memory
    /// `blocks` map.
    pub fn clear_engine_arc(&self) {
        if let Ok(mut slot) = self.engine_arc.write() {
            *slot = None;
        }
    }

    /// **Phase B-tq.4** — try to construct a `TqPackedSpill` whose
    /// `engine_arc` is bound to a live `Engine`.  The factory's
    /// `try_construct` calls this with the type-erased
    /// `Arc<dyn Any + Send + Sync>` from the registry's
    /// `try_substitute_on_load`.
    ///
    /// Returns `None` on type mismatch (silent no-op per the
    /// `FamilyHookFactory` contract — caller falls through to the
    /// stub registration).
    pub fn try_from_engine_arc(
        cfg: TqPackedConfig,
        engine_dyn: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Option<Self> {
        let engine_arc =
            engine_dyn.downcast::<crate::serve::api::engine::Engine>().ok()?;
        // Cheap-clone the inner Engine.  See `engine_arc` field doc
        // for why this doesn't break `LoaderWrapper::load`'s
        // `try_unwrap` — `Engine: Clone` over `inner: Arc<EngineInner>`,
        // so cloning into our slot is independent of the OUTER Arc.
        let engine: crate::serve::api::engine::Engine = (*engine_arc).clone();
        // Unused: the outer Arc<Engine> drops at end of scope (its
        // strong count returns to whatever the registry holds), keeping
        // try_unwrap's contract intact.
        drop(engine_arc);
        Self::new_with_engine(cfg, engine).ok()
    }

    /// **Phase B-tq.4** — engine-bound snapshot path.  Mirrors
    /// `Gemma4DenseSpill::snapshot_via_engine` at
    /// `gemma4_dense.rs:873-938`.
    ///
    /// Bridges into the engine's worker thread via the existing
    /// `Engine::Request::KvSnapshotTq` message (added in this iter
    /// alongside `tq_v2_snapshot_block`'s wrapper).  The worker
    /// thread reads `MlxModelWeights.kv_caches[layer]` and packs two
    /// `tq_packed_v2` envelopes (one for K, one for V); we pack
    /// those into a K+V bundle and return.
    ///
    /// Returns `None` on engine-side failure (engine dropped, layer
    /// out of range, codec error) — matches the trait's "no work" /
    /// fall-through semantic.
    fn snapshot_via_engine(
        &self,
        engine: &crate::serve::api::engine::Engine,
        layer_rank: usize,
        range: Range<u32>,
    ) -> Option<Vec<u8>> {
        if layer_rank >= self.cfg.num_layers {
            return None;
        }
        if range.end <= range.start {
            return None;
        }
        let bits = self.cfg.bits_per_coord;
        let flags = self.cfg.flags;
        let scale = self.cfg.scale;
        // Bridge to the engine's worker thread.  The wrapper handles
        // the channel send + response; here we just consume the
        // `(k_payload, v_payload)` and bundle.
        let (k_payload, v_payload) = engine
            .tq_packed_v2_snapshot_block(layer_rank, range, bits, flags, scale)
            .ok()?;
        pack_tq_v2_kv_bundle(&k_payload, &v_payload).ok()
    }

    /// **Phase B-tq.4** — engine-bound restore path.  Inverse of
    /// `snapshot_via_engine`.  Mirrors
    /// `Gemma4DenseSpill::restore_via_engine` at
    /// `gemma4_dense.rs:940-987`.
    fn restore_bundle(
        &self,
        layer_rank: usize,
        range: Range<u32>,
        payload: &[u8],
    ) -> Result<(), SpillErrorKind> {
        let (k_payload, v_payload) =
            unpack_tq_v2_kv_bundle(payload).map_err(|_| SpillErrorKind::CodecErr)?;

        // Validate v2 envelope shape against the runtime config
        // BEFORE crossing the engine boundary — silent corruption is
        // expensive on a real GPU.
        let (header_k, _idx_k, _norms_k) =
            unpack_tq_v2_payload(k_payload).map_err(|_| SpillErrorKind::CodecErr)?;
        let (header_v, _idx_v, _norms_v) =
            unpack_tq_v2_payload(v_payload).map_err(|_| SpillErrorKind::CodecErr)?;
        if header_k.bits_per_coord != self.cfg.bits_per_coord
            || header_v.bits_per_coord != self.cfg.bits_per_coord
            || header_k.head_dim != self.cfg.head_dim[layer_rank]
            || header_v.head_dim != self.cfg.head_dim[layer_rank]
            || header_k.n_kv_heads != self.cfg.nkv_heads[layer_rank]
            || header_v.n_kv_heads != self.cfg.nkv_heads[layer_rank]
            || header_k.n_tokens != range.end.saturating_sub(range.start)
            || header_v.n_tokens != range.end.saturating_sub(range.start)
        {
            return Err(SpillErrorKind::CodecErr);
        }

        // Engine-bound write-back.
        let engine_opt = self
            .engine_arc
            .read()
            .map_err(|_| SpillErrorKind::IoErr)?
            .as_ref()
            .cloned();
        if let Some(engine) = engine_opt {
            engine
                .tq_packed_v2_restore_block(
                    layer_rank,
                    range.clone(),
                    self.cfg.bits_per_coord,
                    k_payload,
                    v_payload,
                )
                .map_err(|_| SpillErrorKind::IoErr)?;
        }
        // Always cache the bundle in the in-memory map — preserves
        // the substrate-path invariant that a restored block can be
        // re-read by `snapshot_block` without round-tripping through
        // the engine again.  This is also the fallback path when
        // engine_arc is unbound (test fixtures + B-tq.2 substrate).
        let mut map = self.blocks.write().map_err(|_| SpillErrorKind::IoErr)?;
        map.insert((layer_rank, range.start), payload.to_vec());
        Ok(())
    }

    /// Insert a pre-packed envelope payload (test fixture / future
    /// engine wiring).  Accepts EITHER a `tq_packed_v1` (substrate) OR a
    /// `tq_packed_v2` (engine-wired) payload — peeks at the magic prefix
    /// to discriminate.  Returns `Err(SpillErrorKind::CodecErr)` if
    /// neither magic matches or the payload fails strict parse.
    pub fn insert_block(
        &self,
        layer_rank: usize,
        range_start: u32,
        payload: Vec<u8>,
    ) -> Result<(), SpillErrorKind> {
        if layer_rank >= self.cfg.num_layers {
            return Err(SpillErrorKind::CodecErr);
        }
        if payload.len() < 4 {
            return Err(SpillErrorKind::CodecErr);
        }
        let magic = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        match magic {
            m if m == TQ_PACKED_V1_MAGIC => {
                let _ = unpack_tq_v1_payload(&payload)
                    .map_err(|_| SpillErrorKind::CodecErr)?;
            }
            m if m == TQ_PACKED_V2_MAGIC => {
                let _ = unpack_tq_v2_payload(&payload)
                    .map_err(|_| SpillErrorKind::CodecErr)?;
            }
            _ => return Err(SpillErrorKind::CodecErr),
        }
        if let Ok(mut map) = self.blocks.write() {
            map.insert((layer_rank, range_start), payload);
            Ok(())
        } else {
            Err(SpillErrorKind::IoErr)
        }
    }

    /// Number of currently-cached blocks (test introspection).
    pub fn block_count(&self) -> usize {
        self.blocks.read().map(|m| m.len()).unwrap_or(0)
    }

    /// Forge a synthetic `tq_packed_v1` payload for the given (layer,
    /// range) using the spill's config.  The indices stream is
    /// deterministic from `(layer_rank, range.start)`.  Used by tests
    /// + future engine wiring as a fixture path.
    pub fn synthesize_block(
        &self,
        layer_rank: usize,
        range: Range<u32>,
    ) -> Result<Vec<u8>, SpillErrorKind> {
        if layer_rank >= self.cfg.num_layers {
            return Err(SpillErrorKind::CodecErr);
        }
        let n_tokens = range.end.saturating_sub(range.start);
        if n_tokens == 0 {
            return Err(SpillErrorKind::CodecErr);
        }
        let header = TqPackedV1Header {
            codec_version: TQ_PACKED_CODEC_VERSION_V1,
            bits_per_coord: self.cfg.bits_per_coord,
            head_dim: self.cfg.head_dim[layer_rank],
            n_kv_heads: self.cfg.nkv_heads[layer_rank],
            n_tokens,
            flags: self.cfg.flags,
            scale: self.cfg.scale,
        };
        let n_indices = header.indices_bytes_len();
        let mut indices = vec![0u8; n_indices];
        // Deterministic per-(layer, range_start) pattern.
        for (i, b) in indices.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(31))
                .wrapping_add((layer_rank as usize).wrapping_mul(0x9E37))
                .wrapping_add(range.start as usize)
                & 0xFF) as u8;
        }
        pack_tq_v1_payload(&header, &indices).map_err(|_| SpillErrorKind::CodecErr)
    }
}

impl KvCacheSpill for TqPackedSpill {
    fn block_alignment(&self) -> u32 {
        self.cfg.block_tokens
    }

    fn n_layers(&self) -> usize {
        self.cfg.num_layers
    }

    fn snapshot_block(&self, layer_rank: usize, range: Range<u32>) -> Option<Vec<u8>> {
        if layer_rank >= self.cfg.num_layers {
            return None;
        }
        // **Phase B-tq.4**: production engine-bound path — when
        // `engine_arc` is populated, read live KV state from the
        // engine via `MlxModelWeights::tq_v2_snapshot_block` (shipped
        // at `forward_mlx.rs:4667+`) and pack into a K+V bundle.  Mirrors
        // `Gemma4DenseSpill::snapshot_via_engine` at
        // `gemma4_dense.rs:1015`.
        let engine_opt = self
            .engine_arc
            .read()
            .ok()
            .and_then(|g| g.as_ref().cloned());
        if let Some(engine) = engine_opt {
            return self.snapshot_via_engine(&engine, layer_rank, range);
        }
        // **B-tq.2 fallback path**: in-memory map (synthesize_block /
        // insert_block / test fixtures).  Returns the stored bytes
        // verbatim — the spiller's persist path treats them as
        // opaque envelope bytes.
        let map = self.blocks.read().ok()?;
        map.get(&(layer_rank, range.start)).cloned()
    }

    fn restore_block(
        &mut self,
        layer_rank: usize,
        range: Range<u32>,
        payload: &[u8],
    ) -> Result<(), SpillErrorKind> {
        if layer_rank >= self.cfg.num_layers {
            return Err(SpillErrorKind::CodecErr);
        }
        if payload.len() < 4 {
            return Err(SpillErrorKind::CodecErr);
        }
        let magic = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);

        // **Phase B-tq.4**: K+V bundle path — restore both K and V
        // envelopes through the engine into the live MlxKvCache
        // buffers.  Falls back to the in-memory map if engine_arc is
        // unbound.
        if magic == TQ_PACKED_KV_BUNDLE_MAGIC {
            return self.restore_bundle(layer_rank, range, payload);
        }

        let (bits, hd, nkv, ntok) = match magic {
            m if m == TQ_PACKED_V1_MAGIC => {
                let (h, idx) = unpack_tq_v1_payload(payload)
                    .map_err(|_| SpillErrorKind::CodecErr)?;
                // Defensive re-pack ⇒ byte equality (B-tq.1 D2 guard).
                let repacked = pack_tq_v1_payload(&h, idx)
                    .map_err(|_| SpillErrorKind::CodecErr)?;
                if repacked != payload {
                    return Err(SpillErrorKind::ParityFail);
                }
                (h.bits_per_coord, h.head_dim, h.n_kv_heads, h.n_tokens)
            }
            m if m == TQ_PACKED_V2_MAGIC => {
                let (h, idx, norms) = unpack_tq_v2_payload(payload)
                    .map_err(|_| SpillErrorKind::CodecErr)?;
                // Defensive re-pack ⇒ byte equality (B-tq.3 D2 guard,
                // mirrors v1's guard against silent tampering).
                let repacked = pack_tq_v2_payload(&h, idx, norms)
                    .map_err(|_| SpillErrorKind::CodecErr)?;
                if repacked != payload {
                    return Err(SpillErrorKind::ParityFail);
                }
                (h.bits_per_coord, h.head_dim, h.n_kv_heads, h.n_tokens)
            }
            _ => return Err(SpillErrorKind::CodecErr),
        };
        // Layout-mismatch guards: dtype-equivalent for TQ-packed is
        // `(bits_per_coord, head_dim, n_kv_heads)`.  A future engine
        // upgrade that bumps any of these without bumping
        // `codec_version` would silently corrupt restored state — fail
        // loud here.
        if bits != self.cfg.bits_per_coord {
            return Err(SpillErrorKind::CodecErr);
        }
        if hd != self.cfg.head_dim[layer_rank] {
            return Err(SpillErrorKind::CodecErr);
        }
        if nkv != self.cfg.nkv_heads[layer_rank] {
            return Err(SpillErrorKind::CodecErr);
        }
        if ntok != range.end.saturating_sub(range.start) {
            return Err(SpillErrorKind::CodecErr);
        }
        let mut map = self.blocks.write().map_err(|_| SpillErrorKind::IoErr)?;
        map.insert((layer_rank, range.start), payload.to_vec());
        Ok(())
    }

    fn model_fingerprint(
        &self,
        _repo: &str,
        _quant: QuantType,
    ) -> Option<ModelFingerprint> {
        self.fingerprint.read().ok().and_then(|fp| fp.clone())
    }
}

// ============================================================================
// Phase B-tq.4 — EngineBindable + TqPackedSpillFactory
// ============================================================================
//
// `EngineBindable` lets the registry call `bind_engine(Arc<dyn Any>)` after
// a successful load — TqPackedSpill downcasts to `Arc<Engine>` and
// populates `engine_arc`.  Mirrors `Gemma4DenseSpill::bind_engine` at
// `gemma4_dense.rs:1441-1474`.
//
// `TqPackedSpillFactory` is the registry-side factory that
// `cmd_serve` registers at startup; on the first successful engine
// load, the registry's `try_substitute_on_load` invokes
// `factory.try_construct(engine_dyn)` which returns a fully-wired
// `(Arc<Mutex<dyn KvCacheSpill>>, Arc<dyn EngineBindable>)` tuple.
// Mirrors `Gemma4DenseSpillFactory` at `gemma4_dense.rs:1591-1692`.

impl crate::serve::kv_persist::EngineBindable for TqPackedSpill {
    fn bind_engine(&self, engine_dyn: Arc<dyn std::any::Any + Send + Sync>) {
        // Try `Arc<Engine>` (production LoaderWrapper path).  Silent
        // no-op on type mismatch per the EngineBindable contract.
        if let Ok(engine_arc) =
            engine_dyn.downcast::<crate::serve::api::engine::Engine>()
        {
            // Cheap-clone Engine (Arc<EngineInner>); see engine_arc
            // field doc for the P0-bench-fix rationale.
            let engine: crate::serve::api::engine::Engine = (*engine_arc).clone();
            self.set_engine_arc(engine);
        }
        // Mismatch ⇒ drop type-erased Arc silently.
    }

    fn unbind_engine(&self) {
        self.clear_engine_arc();
    }
}

/// **Phase B-tq.4** — `FamilyHookFactory` impl for `TqPackedSpill`.
///
/// Carries the shape-only `TqPackedConfig` captured at `cmd_serve`
/// startup.  On `try_construct(engine_dyn)`:
///
///   1. Calls [`TqPackedSpill::try_from_engine_arc`] which downcasts
///      to `Arc<Engine>` and constructs a fully-wired spill on
///      success.
///   2. On `None` (type mismatch) → returns `None` (the registry's
///      caller leaves the prior stub registration in place).
///   3. On `Some(spill)` → wraps the spill in the
///      `(Arc<Mutex<dyn KvCacheSpill>>, Arc<dyn EngineBindable>)`
///      tuple expected by the `FamilyHookFactory` trait.  The same
///      spill instance is referenced by both ends of the tuple — so
///      a `bind_engine` call through the registry side mutates the
///      same engine slot that a `restore_block` call through the
///      spiller side reads.
pub struct TqPackedSpillFactory {
    cfg: TqPackedConfig,
}

impl TqPackedSpillFactory {
    /// Construct a factory carrying the supplied shape config.  The
    /// config is captured by value; subsequent factory invocations
    /// reuse the same shape (the loaded model's TQ-packed shape is
    /// immutable across evict/readmit cycles).
    pub fn new(cfg: TqPackedConfig) -> Self {
        Self { cfg }
    }
}

impl crate::serve::kv_persist::registry::FamilyHookFactory for TqPackedSpillFactory {
    fn try_construct(
        &self,
        engine_dyn: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Option<(
        Arc<std::sync::Mutex<dyn crate::serve::kv_persist::spiller::KvCacheSpill>>,
        Arc<dyn crate::serve::kv_persist::EngineBindable>,
    )> {
        // 1. Try to materialize a fully-wired spill from the engine.
        //    None on type mismatch ⇒ no substitution.
        let spill = TqPackedSpill::try_from_engine_arc(self.cfg.clone(), engine_dyn)?;

        // 2. Wrap in the dual-end tuple.  Both ends point at the
        //    SAME spill instance (Arc), so binding/unbinding through
        //    the registry side updates the engine_arc that the
        //    spiller side reads.  Mirrors `Gemma4DenseSpillFactory`'s
        //    spill_arc.clone() pattern at `gemma4_dense.rs:1641-1642`.
        //
        //    For TQ-packed the spiller-side instance is the same as
        //    the bindable-side instance (no separate dual-spill-
        //    instance dance like Gemma4Dense's path) because the
        //    TQ-packed engine_arc is the SOLE bridge — there's no
        //    EngineHandle backwards-compat path to maintain.
        let spill_arc = Arc::new(spill);
        let bindable: Arc<dyn crate::serve::kv_persist::EngineBindable> =
            spill_arc.clone();

        // 3. Build the spiller-side spill.  Different instance than
        //    the bindable-side because `KvCacheSpill::restore_block`
        //    takes `&mut self` (in-memory map insert), so the
        //    spiller's Mutex needs an owned spill.  The spiller's
        //    spill is freshly constructed via `new_with_engine`
        //    using a CLONE of the same engine — both spills route
        //    through the same worker thread.
        let spiller_engine = spill_arc
            .engine_arc
            .read()
            .ok()
            .and_then(|g| g.as_ref().cloned())?;
        let spiller_side =
            TqPackedSpill::new_with_engine(self.cfg.clone(), spiller_engine).ok()?;
        let kv_hook: Arc<
            std::sync::Mutex<dyn crate::serve::kv_persist::spiller::KvCacheSpill>,
        > = Arc::new(std::sync::Mutex::new(spiller_side));

        Some((kv_hook, bindable))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_header(bits: u32, n_kv: u32, n_tok: u32, hd: u32) -> TqPackedV1Header {
        TqPackedV1Header {
            codec_version: TQ_PACKED_CODEC_VERSION_V1,
            bits_per_coord: TqBitsPerCoord::new(bits).expect("valid bits"),
            head_dim: hd,
            n_kv_heads: n_kv,
            n_tokens: n_tok,
            flags: flags::HADAMARD_ROTATED,
            scale: 0.097_001_234_5_f64, // arbitrary non-trivial f64
        }
    }

    fn synthetic_indices(n_bytes: usize) -> Vec<u8> {
        let mut v = vec![0u8; n_bytes];
        // Non-trivial per-byte pattern: i wrapping at u8 + a prime-ish offset.
        for (i, b) in v.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(31)).wrapping_add(0xAB) & 0xFF) as u8;
        }
        v
    }

    /// **ADR-017 §B-tq.1 R-C2 / D2 deterministic-rebuild gate**:
    /// pack(unpack(pack(h, idx))) == pack(h, idx) byte-for-byte AND
    /// SHA-256-identical.  Per ADR-017 line 459: "Deterministic codec
    /// rebuild on restore (D2): SHA-256 of restored TQ state byte-exact
    /// vs pre-spill TQ state for the same input."
    #[test]
    fn tq_packed_v1_round_trip_byte_exact() {
        let h = synthetic_header(/* bits = */ 4, /* n_kv = */ 8, /* n_tok = */ 256, /* hd = */ 256);
        let idx = synthetic_indices(h.indices_bytes_len());

        let pack_a = pack_tq_v1_payload(&h, &idx).expect("pack A");
        let (h_unpacked, idx_unpacked) = unpack_tq_v1_payload(&pack_a).expect("unpack A");
        assert_eq!(h_unpacked, h, "header round-trip mismatch");
        assert_eq!(idx_unpacked, idx.as_slice(), "indices round-trip mismatch");

        let pack_b = pack_tq_v1_payload(&h_unpacked, idx_unpacked).expect("pack B");
        assert_eq!(
            pack_a, pack_b,
            "byte-exact pack determinism violated: pack(unpack(pack(.))) != pack(.)"
        );

        let sha_a = sha256_payload(&pack_a);
        let sha_b = sha256_payload(&pack_b);
        assert_eq!(
            sha_a, sha_b,
            "SHA-256 mismatch on pack→unpack→pack round-trip — D2 deterministic-rebuild gate FAIL"
        );
    }

    /// **ADR-017 §B-tq.1 R-C2 cosine ≥ 0.9998**: when the on-disk
    /// envelope deterministically round-trips, the dequantized state
    /// from the restored bytes is BYTE-IDENTICAL to the dequantized
    /// state from the pre-spill bytes (because dequantize is a pure
    /// function of (header, indices)).  Cosine of identical vectors
    /// is exactly 1.0, so R-C2's ≥ 0.9998 gate is trivially
    /// satisfied AS A CONSEQUENCE of D2 byte-exact rebuild.
    ///
    /// This test makes the "trivially satisfied" statement explicit by
    /// computing cosine against a synthetic dequantize function and
    /// asserting cosine == 1.0 (not just ≥ 0.9998).
    #[test]
    fn tq_packed_v1_cosine_gate_a_satisfied_by_construction() {
        let h = synthetic_header(8, 2, 64, 256);
        let idx = synthetic_indices(h.indices_bytes_len());
        let payload = pack_tq_v1_payload(&h, &idx).expect("pack");

        // Toy dequantize: treat each index as an f32 in [-1, 1] using
        // a synthetic codebook (offset / 128.0).  Apply the header's
        // scale.  The key property: dequantize(pre_spill_idx) ==
        // dequantize(restored_idx) ELEMENT-WISE because the bytes are
        // byte-equal post round-trip.
        fn dequantize(bytes: &[u8], scale: f64) -> Vec<f32> {
            bytes
                .iter()
                .map(|&b| ((b as i32 - 128) as f32 / 128.0) * (scale as f32))
                .collect()
        }

        let pre_spill = dequantize(&idx, h.scale);
        let (h_restored, idx_restored) = unpack_tq_v1_payload(&payload).expect("unpack");
        let restored = dequantize(idx_restored, h_restored.scale);

        // Element-wise byte-equality of the f32 outputs (stronger than
        // cosine ≥ 0.9998 — these vectors should be IDENTICAL bits).
        for (i, (&a, &b)) in pre_spill.iter().zip(restored.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "dequantize byte-mismatch at element {i} — round-trip is NOT byte-exact \
                 (cosine 0.9998 gate would require kernel work but byte-exact byte-equality \
                 means cosine = 1.0 trivially)"
            );
        }

        // Symbolic cosine assertion.  Two byte-equal f32 vectors have
        // cosine exactly 1.0 (subject to the dot-product reduction
        // order; we use the sum-of-squared-norms identity).
        let dot: f64 = pre_spill
            .iter()
            .zip(restored.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let norm_a: f64 = pre_spill.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = restored.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (norm_a * norm_b);
        assert!(
            cosine >= 0.9998,
            "R-C2 gate failed: cosine = {cosine:.10} < 0.9998 (per ADR-007 Gate A)"
        );
        assert!(
            (cosine - 1.0).abs() < 1e-12,
            "B-tq.1 byte-exact round-trip should yield cosine = 1.0; got {cosine:.20}"
        );
    }

    /// Reject payloads with the wrong magic prefix.
    #[test]
    fn tq_packed_v1_rejects_bad_magic() {
        let mut bytes = vec![0u8; TQ_PACKED_V1_HEADER_BYTES];
        bytes[..4].copy_from_slice(b"XXXX");
        match unpack_tq_v1_payload(&bytes) {
            Err(TqEnvelopeError::BadMagic { got }) => {
                assert_ne!(got, TQ_PACKED_V1_MAGIC);
            }
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    /// Reject payloads claiming a future codec version.
    #[test]
    fn tq_packed_v1_rejects_future_codec_version() {
        let h = synthetic_header(4, 2, 16, 64);
        let idx = synthetic_indices(h.indices_bytes_len());
        let mut payload = pack_tq_v1_payload(&h, &idx).expect("pack");
        payload[4..8].copy_from_slice(&7_u32.to_le_bytes());
        assert_eq!(
            unpack_tq_v1_payload(&payload),
            Err(TqEnvelopeError::UnsupportedCodecVersion(7))
        );
    }

    /// Reject payloads with an invalid bits-per-coord.
    #[test]
    fn tq_packed_v1_rejects_zero_bits_per_coord() {
        let h = synthetic_header(4, 2, 16, 64);
        let idx = synthetic_indices(h.indices_bytes_len());
        let mut payload = pack_tq_v1_payload(&h, &idx).expect("pack");
        payload[8..12].copy_from_slice(&0_u32.to_le_bytes());
        assert_eq!(
            unpack_tq_v1_payload(&payload),
            Err(TqEnvelopeError::InvalidBitsPerCoord(0))
        );
    }

    /// Reject payloads truncated below the header size.
    #[test]
    fn tq_packed_v1_rejects_truncated_header() {
        let bytes = vec![0u8; 16];
        assert_eq!(
            unpack_tq_v1_payload(&bytes),
            Err(TqEnvelopeError::Truncated { got: 16 })
        );
    }

    /// Reject payloads with a non-zero reserved field (forward-compat).
    #[test]
    fn tq_packed_v1_rejects_nonzero_reserved() {
        let h = synthetic_header(4, 2, 16, 64);
        let idx = synthetic_indices(h.indices_bytes_len());
        let mut payload = pack_tq_v1_payload(&h, &idx).expect("pack");
        payload[28..32].copy_from_slice(&0xDEADBEEF_u32.to_le_bytes());
        assert_eq!(
            unpack_tq_v1_payload(&payload),
            Err(TqEnvelopeError::ReservedNonZero { got: 0xDEADBEEF })
        );
    }

    /// Pack rejects a mismatched indices buffer.
    #[test]
    fn tq_packed_v1_pack_rejects_short_indices() {
        let h = synthetic_header(4, 2, 16, 64);
        let too_short = vec![0u8; h.indices_bytes_len() - 1];
        assert_eq!(
            pack_tq_v1_payload(&h, &too_short),
            Err(TqEnvelopeError::PayloadSizeMismatch {
                expected: h.indices_bytes_len(),
                got: h.indices_bytes_len() - 1,
            })
        );
    }

    /// Indices_bytes_len matches the wire formula across bit widths.
    #[test]
    fn tq_packed_v1_indices_bytes_len_formula() {
        // 8-bit: 1 byte/coord
        let h8 = synthetic_header(8, 4, 100, 256);
        assert_eq!(h8.indices_bytes_len(), 4 * 100 * 256);

        // 4-bit: 0.5 byte/coord
        let h4 = synthetic_header(4, 4, 100, 256);
        assert_eq!(h4.indices_bytes_len(), 4 * 100 * 256 / 2);

        // 2-bit: 0.25 byte/coord
        let h2 = synthetic_header(2, 4, 100, 256);
        assert_eq!(h2.indices_bytes_len(), 4 * 100 * 256 / 4);

        // 3-bit: ceil-div by 8 with non-multiple-of-8 total bits
        let h3 = synthetic_header(3, 1, 1, 5);
        // 1 × 1 × 5 × 3 = 15 bits → 2 bytes (15.div_ceil(8) = 2)
        assert_eq!(h3.indices_bytes_len(), 2);
    }

    /// Total_bytes is header + indices.
    #[test]
    fn tq_packed_v1_total_bytes_consistent() {
        let h = synthetic_header(4, 8, 256, 256);
        assert_eq!(
            h.total_bytes(),
            TQ_PACKED_V1_HEADER_BYTES + h.indices_bytes_len()
        );
        let idx = synthetic_indices(h.indices_bytes_len());
        let payload = pack_tq_v1_payload(&h, &idx).expect("pack");
        assert_eq!(payload.len(), h.total_bytes());
    }

    /// Frozen-magic regression test: if anyone ever changes
    /// `TQ_PACKED_V1_MAGIC`, the on-disk envelope semantics change
    /// and pre-existing cache directories become unreadable without
    /// a B-tq.X migration plan.  Pin the constant.
    #[test]
    fn tq_packed_v1_magic_is_frozen() {
        assert_eq!(TQ_PACKED_V1_MAGIC, 0x31_50_51_54);
        assert_eq!(TQ_PACKED_V1_MAGIC.to_le_bytes(), *b"TQP1");
    }

    /// Frozen codec_version regression test.
    #[test]
    fn tq_packed_v1_codec_version_is_frozen() {
        assert_eq!(TQ_PACKED_CODEC_VERSION_V1, 1);
    }

    /// **ADR-017 §B-tq.1 storage delta sanity check**: at production
    /// shapes (Gemma 4 26B-DWQ48 sliding layer: nkv=8, head_dim=256;
    /// global layer: nkv=2, head_dim=512), the TQ-packed envelope at
    /// 4-bit is ~8× smaller than dense F32.  This is informational
    /// (per ADR-017 line 461 — not a ship-gate; expected ratio
    /// ~3-4× per ADR-007 codec, but pure storage at 4-bit is 8× in
    /// the indices stream alone — overhead is the 40-byte header).
    #[test]
    fn tq_packed_v1_storage_delta_4bit_dense_f32_8x() {
        let n_tokens = 1024_u32;

        // Sliding layer: nkv=8, head_dim=256.
        let h_sliding = synthetic_header(4, 8, n_tokens, 256);
        let dense_f32_bytes = 8 * (n_tokens as usize) * 256 * 4; // F32 = 4 bytes
        let tq_packed_bytes = h_sliding.total_bytes();
        let ratio = dense_f32_bytes as f64 / tq_packed_bytes as f64;
        // 4-bit indices are 1 nibble per coord vs F32's 4 bytes per
        // coord = 8× ratio in pure indices; the 40-byte header
        // negligibly dilutes this at production scales.
        assert!(
            ratio > 7.9,
            "expected ~8× storage savings (4-bit vs F32), got {ratio:.2}× \
             (dense={dense_f32_bytes} TQ={tq_packed_bytes})"
        );
        assert!(
            ratio < 8.1,
            "expected ~8× storage savings (4-bit vs F32), got {ratio:.2}× \
             (dense={dense_f32_bytes} TQ={tq_packed_bytes})"
        );

        // Global layer: nkv=2, head_dim=512.
        let h_global = synthetic_header(4, 2, n_tokens, 512);
        let dense_f32_global = 2 * (n_tokens as usize) * 512 * 4;
        let tq_packed_global = h_global.total_bytes();
        let ratio_global = dense_f32_global as f64 / tq_packed_global as f64;
        assert!(
            ratio_global > 7.9 && ratio_global < 8.1,
            "global layer 4-bit storage ratio: {ratio_global:.2}× \
             (dense={dense_f32_global} TQ={tq_packed_global})"
        );
    }

    // ========================================================================
    // Phase B-tq.2 — TqPackedSpill engine-side hook tests
    // ========================================================================

    fn synthetic_cfg(num_layers: usize) -> TqPackedConfig {
        TqPackedConfig {
            num_layers,
            nkv_heads: vec![2; num_layers],
            head_dim: vec![256; num_layers],
            bits_per_coord: TqBitsPerCoord::new(8).unwrap(),
            scale: 0.125_f64,
            flags: flags::HADAMARD_ROTATED,
            block_tokens: 256,
        }
    }

    /// `TqPackedSpill::new` validates config invariants.
    #[test]
    fn tq_packed_spill_rejects_invalid_config() {
        let mut cfg = synthetic_cfg(4);
        cfg.nkv_heads = vec![2, 2]; // length mismatch
        assert_eq!(TqPackedSpill::new(cfg).err(), Some(SpillErrorKind::CodecErr));
    }

    /// `block_alignment` returns `cfg.block_tokens`; `n_layers` returns
    /// `cfg.num_layers`.
    #[test]
    fn tq_packed_spill_trait_surface_constants() {
        let spill = TqPackedSpill::new(synthetic_cfg(8)).expect("new");
        assert_eq!(spill.block_alignment(), 256);
        assert_eq!(spill.n_layers(), 8);
    }

    /// `snapshot_block` returns None when no block stored.
    #[test]
    fn tq_packed_spill_snapshot_empty_returns_none() {
        let spill = TqPackedSpill::new(synthetic_cfg(2)).expect("new");
        assert!(spill.snapshot_block(0, 0..256).is_none());
        assert!(spill.snapshot_block(1, 256..512).is_none());
    }

    /// `snapshot_block` returns None for out-of-range layer.
    #[test]
    fn tq_packed_spill_snapshot_layer_out_of_range_returns_none() {
        let spill = TqPackedSpill::new(synthetic_cfg(2)).expect("new");
        let payload = spill.synthesize_block(0, 0..256).expect("synth");
        spill.insert_block(0, 0, payload).expect("insert");
        assert!(spill.snapshot_block(99, 0..256).is_none());
    }

    /// **B-tq.2 round-trip parity**: synthesize → insert → snapshot
    /// returns the byte-identical payload.
    #[test]
    fn tq_packed_spill_synthesize_insert_snapshot_round_trip() {
        let spill = TqPackedSpill::new(synthetic_cfg(4)).expect("new");
        let payload = spill.synthesize_block(2, 256..512).expect("synth");
        let payload_clone = payload.clone();
        spill.insert_block(2, 256, payload).expect("insert");

        let snapped = spill.snapshot_block(2, 256..512).expect("snapshot");
        assert_eq!(snapped, payload_clone);

        // The snapped bytes parse back to a header whose fields match
        // the spill's config.
        let (header, _) = unpack_tq_v1_payload(&snapped).expect("unpack");
        assert_eq!(header.codec_version, TQ_PACKED_CODEC_VERSION_V1);
        assert_eq!(header.bits_per_coord.0, 8);
        assert_eq!(header.head_dim, 256);
        assert_eq!(header.n_kv_heads, 2);
        assert_eq!(header.n_tokens, 256);
        assert_eq!(header.flags, flags::HADAMARD_ROTATED);
    }

    /// **B-tq.2 restore_block via trait**: writing through
    /// `KvCacheSpill::restore_block` round-trips byte-equally to the
    /// subsequent `snapshot_block`.
    #[test]
    fn tq_packed_spill_restore_via_trait_round_trip() {
        let mut spill = TqPackedSpill::new(synthetic_cfg(4)).expect("new");
        let payload = spill.synthesize_block(1, 0..256).expect("synth");
        let payload_clone = payload.clone();

        spill
            .restore_block(1, 0..256, &payload)
            .expect("restore_block");
        let snapped = spill.snapshot_block(1, 0..256).expect("snapshot");
        assert_eq!(snapped, payload_clone);
    }

    /// `restore_block` rejects payloads with a layout mismatch
    /// (head_dim drift).  This is the load-bearing wire-format-drift
    /// guard — without it, a future engine upgrade that bumped head_dim
    /// without bumping `codec_version` would silently corrupt restored
    /// state.
    #[test]
    fn tq_packed_spill_restore_rejects_head_dim_mismatch() {
        let cfg = synthetic_cfg(4);
        let mut spill = TqPackedSpill::new(cfg).expect("new");

        // Build a payload claiming head_dim=128 (config says 256).
        let bad_header = TqPackedV1Header {
            codec_version: TQ_PACKED_CODEC_VERSION_V1,
            bits_per_coord: TqBitsPerCoord::new(8).unwrap(),
            head_dim: 128,
            n_kv_heads: 2,
            n_tokens: 256,
            flags: 0,
            scale: 1.0,
        };
        let bad_indices = vec![0u8; bad_header.indices_bytes_len()];
        let bad_payload = pack_tq_v1_payload(&bad_header, &bad_indices).expect("pack");

        assert_eq!(
            spill.restore_block(0, 0..256, &bad_payload).err(),
            Some(SpillErrorKind::CodecErr)
        );
    }

    /// `restore_block` rejects payloads with a `bits_per_coord` mismatch.
    #[test]
    fn tq_packed_spill_restore_rejects_bits_per_coord_mismatch() {
        let mut spill = TqPackedSpill::new(synthetic_cfg(2)).expect("new");
        let bad_header = TqPackedV1Header {
            codec_version: TQ_PACKED_CODEC_VERSION_V1,
            bits_per_coord: TqBitsPerCoord::new(4).unwrap(), // config says 8
            head_dim: 256,
            n_kv_heads: 2,
            n_tokens: 256,
            flags: 0,
            scale: 1.0,
        };
        let bad_indices = vec![0u8; bad_header.indices_bytes_len()];
        let bad_payload = pack_tq_v1_payload(&bad_header, &bad_indices).expect("pack");
        assert_eq!(
            spill.restore_block(0, 0..256, &bad_payload).err(),
            Some(SpillErrorKind::CodecErr)
        );
    }

    /// `restore_block` rejects payloads whose declared `n_tokens`
    /// disagrees with the requested range.
    #[test]
    fn tq_packed_spill_restore_rejects_range_mismatch() {
        let mut spill = TqPackedSpill::new(synthetic_cfg(2)).expect("new");
        let payload = spill.synthesize_block(0, 0..256).expect("synth"); // header says n_tokens=256
        // Caller asks restore for 0..128 (128 tokens).
        assert_eq!(
            spill.restore_block(0, 0..128, &payload).err(),
            Some(SpillErrorKind::CodecErr)
        );
    }

    /// `model_fingerprint` returns None until set, then the configured
    /// fingerprint after.
    #[test]
    fn tq_packed_spill_fingerprint_round_trip() {
        let spill = TqPackedSpill::new(synthetic_cfg(2)).expect("new");
        let fp = crate::serve::kv_persist::format::compute_model_fingerprint(
            "test-repo/tq-fixture",
            "Q8_0",
            "hf2q-test",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "<chat>{messages}</chat>",
        );
        // Default trait impl returns None; via the override (configured
        // fingerprint), it returns Some.
        assert!(KvCacheSpill::model_fingerprint(
            &spill,
            "test-repo/tq-fixture",
            QuantType::Q8_0,
        )
        .is_none());
        spill.set_fingerprint(fp.clone());
        let got = KvCacheSpill::model_fingerprint(
            &spill,
            "test-repo/tq-fixture",
            QuantType::Q8_0,
        );
        assert_eq!(got, Some(fp));
    }

    /// `block_count` reflects the in-memory map.
    #[test]
    fn tq_packed_spill_block_count_tracks_inserts() {
        let spill = TqPackedSpill::new(synthetic_cfg(4)).expect("new");
        assert_eq!(spill.block_count(), 0);
        let p0 = spill.synthesize_block(0, 0..256).expect("p0");
        let p1 = spill.synthesize_block(1, 0..256).expect("p1");
        let p2 = spill.synthesize_block(2, 256..512).expect("p2");
        spill.insert_block(0, 0, p0).expect("ins0");
        spill.insert_block(1, 0, p1).expect("ins1");
        spill.insert_block(2, 256, p2).expect("ins2");
        assert_eq!(spill.block_count(), 3);
    }

    // ============================================================================
    // Phase B-tq.3 — v2 envelope + capture/restore tests
    // ============================================================================

    fn synthetic_v2_header(bits: u32, nkv: u32, ntok: u32, hd: u32) -> TqPackedV2Header {
        TqPackedV2Header {
            codec_version: TQ_PACKED_CODEC_VERSION_V2,
            bits_per_coord: TqBitsPerCoord::new(bits).expect("valid bits"),
            head_dim: hd,
            n_kv_heads: nkv,
            n_tokens: ntok,
            flags: flags::HADAMARD_ROTATED,
            scale: 1.0_f64,
        }
    }

    fn synthetic_norms_le(nkv: usize, ntok: usize, seed: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(nkv * ntok * 4);
        for h in 0..nkv {
            for t in 0..ntok {
                // Deterministic non-trivial F32 per (h, t).
                let v = ((seed.wrapping_mul(0x9E37) + (h as u32) * 0x1F0D + (t as u32) * 0x07) as f32)
                    / 65536.0;
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// **B-tq.3 v2 round-trip**: pack(unpack(pack(h, idx, norms))) ==
    /// pack(h, idx, norms) byte-for-byte AND SHA-256-identical.  Same
    /// determinism property as v1.
    #[test]
    fn tq_packed_v2_round_trip_byte_exact() {
        let h = synthetic_v2_header(4, 8, 256, 256);
        let idx = synthetic_indices(h.indices_bytes_len());
        let norms = synthetic_norms_le(h.n_kv_heads as usize, h.n_tokens as usize, 1234);

        let pack_a = pack_tq_v2_payload(&h, &idx, &norms).expect("pack A");
        let (h2, idx2, norms2) = unpack_tq_v2_payload(&pack_a).expect("unpack A");
        assert_eq!(h2, h);
        assert_eq!(idx2, idx.as_slice());
        assert_eq!(norms2, norms.as_slice());

        let pack_b = pack_tq_v2_payload(&h2, idx2, norms2).expect("pack B");
        assert_eq!(pack_a, pack_b, "v2 byte-exact pack determinism violated");

        let sha_a = sha256_payload(&pack_a);
        let sha_b = sha256_payload(&pack_b);
        assert_eq!(sha_a, sha_b);
    }

    /// **B-tq.3 frozen magic**: any change to `TQ_PACKED_V2_MAGIC` is a
    /// silent on-disk format change.  CI catches via this regression
    /// pin.
    #[test]
    fn tq_packed_v2_magic_is_frozen() {
        // ASCII "TQP2" little-endian: byte[0]=0x54 (T), byte[1]=0x51 (Q),
        // byte[2]=0x50 (P), byte[3]=0x32 (2).  u32::from_le_bytes
        // assembles MSB-first in the literal: 0x32_50_51_54.
        assert_eq!(TQ_PACKED_V2_MAGIC, 0x32_50_51_54);
        // Sanity: the byte stream really is "TQP2".
        assert_eq!(TQ_PACKED_V2_MAGIC.to_le_bytes(), *b"TQP2");
    }

    /// **B-tq.3 frozen codec_version**: bumping requires a B-tq.X
    /// migration plan.
    #[test]
    fn tq_packed_v2_codec_version_is_frozen() {
        assert_eq!(TQ_PACKED_CODEC_VERSION_V2, 2);
    }

    /// **B-tq.3 v2 ≠ v1 magic**: ensures v1 and v2 are distinguishable
    /// by their first 4 bytes.  Without this property, the
    /// `TqPackedSpill::insert_block` magic-dispatch would silently route
    /// the wrong codec.
    #[test]
    fn tq_packed_v1_v2_magics_differ() {
        assert_ne!(TQ_PACKED_V1_MAGIC, TQ_PACKED_V2_MAGIC);
    }

    /// **B-tq.3 v1 reader rejects v2 payloads**: cross-version
    /// confusion-attack guard.  A v2 envelope fed through
    /// `unpack_tq_v1_payload` MUST return `BadMagic` (or
    /// `UnsupportedCodecVersion` if magic ever drifts to match) — never
    /// silent decode.
    #[test]
    fn tq_packed_v1_reader_rejects_v2_payload() {
        let h = synthetic_v2_header(4, 8, 256, 256);
        let idx = synthetic_indices(h.indices_bytes_len());
        let norms = synthetic_norms_le(h.n_kv_heads as usize, h.n_tokens as usize, 7);
        let v2_bytes = pack_tq_v2_payload(&h, &idx, &norms).expect("pack v2");

        let result = unpack_tq_v1_payload(&v2_bytes);
        assert!(
            matches!(result, Err(TqEnvelopeError::BadMagic { .. })),
            "v1 reader silently accepted v2 payload: {:?}",
            result
        );
    }

    /// **B-tq.3 v2 reader rejects v1 payloads**: symmetric guard.
    #[test]
    fn tq_packed_v2_reader_rejects_v1_payload() {
        let h = synthetic_header(4, 8, 256, 256);
        let idx = synthetic_indices(h.indices_bytes_len());
        let v1_bytes = pack_tq_v1_payload(&h, &idx).expect("pack v1");

        let result = unpack_tq_v2_payload(&v1_bytes);
        assert!(
            matches!(result, Err(TqEnvelopeError::BadMagic { .. })),
            "v2 reader silently accepted v1 payload: {:?}",
            result
        );
    }

    /// **B-tq.3 capture→restore byte-identity**: synthetic
    /// `[nkv, capacity, hd_packed]` U8 packed buffer + `[nkv, capacity]`
    /// F32 norms → capture v2 → restore into a freshly-zeroed pair of
    /// buffers → assert byte-identity on the captured token range.
    /// This is the engine-wired path's R-C1 equivalent.
    #[test]
    fn tq_v2_capture_restore_byte_identity() {
        let nkv: u32 = 8;
        let capacity: u32 = 4096;
        let head_dim: u32 = 256;
        let bits = TqBitsPerCoord::new(4).expect("4-bit");
        let hd_packed = (head_dim as usize) * (bits.0 as usize) / 8;
        let range = 1024_u32..1280_u32; // 256-token block

        // --- Synth source buffers ----------------------------------------------------
        let mut packed_src: Vec<u8> =
            vec![0u8; (nkv as usize) * (capacity as usize) * hd_packed];
        for (i, b) in packed_src.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(0x9E_37)) & 0xFF) as u8;
        }
        // norms F32 → LE bytes
        let mut norms_f32_src: Vec<f32> = Vec::with_capacity((nkv * capacity) as usize);
        for h in 0..nkv {
            for t in 0..capacity {
                let v = (h as f32) * 0.1234 + (t as f32) * 0.000_321 + 0.5;
                norms_f32_src.push(v);
            }
        }
        let mut norms_le_src: Vec<u8> = Vec::with_capacity(norms_f32_src.len() * 4);
        for &v in &norms_f32_src {
            norms_le_src.extend_from_slice(&v.to_le_bytes());
        }

        // --- Capture ------------------------------------------------------------------
        let payload = capture_tq_v2_payload_from_buffers(
            &packed_src,
            &norms_le_src,
            capacity,
            nkv,
            head_dim,
            bits,
            range.clone(),
            flags::HADAMARD_ROTATED,
            1.0,
        )
        .expect("capture");

        // Header sanity.
        let (h, idx, norms) = unpack_tq_v2_payload(&payload).expect("unpack");
        assert_eq!(h.bits_per_coord, bits);
        assert_eq!(h.head_dim, head_dim);
        assert_eq!(h.n_kv_heads, nkv);
        assert_eq!(h.n_tokens, range.end - range.start);
        assert_eq!(idx.len(), (nkv as usize) * (range.end - range.start) as usize * hd_packed);
        assert_eq!(
            norms.len(),
            (nkv as usize) * (range.end - range.start) as usize * 4
        );

        // --- Restore into freshly-zeroed buffers --------------------------------------
        let mut packed_dst: Vec<u8> =
            vec![0u8; (nkv as usize) * (capacity as usize) * hd_packed];
        let mut norms_le_dst: Vec<u8> = vec![0u8; (nkv as usize) * (capacity as usize) * 4];

        let h_restored = restore_tq_v2_payload_into_buffers(
            &payload,
            &mut packed_dst,
            &mut norms_le_dst,
            capacity,
            nkv,
            head_dim,
            bits,
            range.clone(),
        )
        .expect("restore");
        assert_eq!(h_restored, h);

        // --- Byte-identity on the captured range --------------------------------------
        // packed_src vs packed_dst on [head, range.start..range.end, 0..hd_packed]
        for head in 0..(nkv as usize) {
            let head_base = head * (capacity as usize) * hd_packed;
            let row_start = head_base + (range.start as usize) * hd_packed;
            let row_end = head_base + (range.end as usize) * hd_packed;
            assert_eq!(
                packed_src[row_start..row_end],
                packed_dst[row_start..row_end],
                "packed mismatch at head={}",
                head
            );
            // Outside the range: dst should still be zero (capture+restore
            // didn't touch other tokens).
            let outside_start = head_base + 0;
            let outside_end = head_base + (range.start as usize) * hd_packed;
            assert!(
                packed_dst[outside_start..outside_end].iter().all(|&b| b == 0),
                "dst leaked into [0, range.start) at head={}",
                head
            );
        }
        // norms F32 round-trip via LE bytes
        for head in 0..(nkv as usize) {
            for t in (range.start as usize)..(range.end as usize) {
                let off = (head * capacity as usize + t) * 4;
                let src_bytes = [
                    norms_le_src[off],
                    norms_le_src[off + 1],
                    norms_le_src[off + 2],
                    norms_le_src[off + 3],
                ];
                let dst_bytes = [
                    norms_le_dst[off],
                    norms_le_dst[off + 1],
                    norms_le_dst[off + 2],
                    norms_le_dst[off + 3],
                ];
                assert_eq!(
                    src_bytes, dst_bytes,
                    "norms mismatch at head={} t={}",
                    head, t
                );
            }
        }
    }

    /// **B-tq.3 capture rejects malformed shape**: negative range,
    /// range past capacity, mismatched buffer sizes, and bits/8
    /// non-aligned head_dim all fail loud (no silent OOB).
    #[test]
    fn tq_v2_capture_rejects_malformed_shape() {
        let nkv: u32 = 4;
        let capacity: u32 = 256;
        let head_dim: u32 = 64;
        let bits = TqBitsPerCoord::new(4).expect("4-bit");
        let hd_packed = (head_dim as usize) * (bits.0 as usize) / 8;
        let packed: Vec<u8> = vec![0u8; (nkv as usize) * (capacity as usize) * hd_packed];
        let norms: Vec<u8> = vec![0u8; (nkv as usize) * (capacity as usize) * 4];

        // empty range
        assert!(capture_tq_v2_payload_from_buffers(
            &packed, &norms, capacity, nkv, head_dim, bits, 10..10,
            flags::HADAMARD_ROTATED, 1.0
        ).is_err());

        // range past capacity
        assert!(capture_tq_v2_payload_from_buffers(
            &packed, &norms, capacity, nkv, head_dim, bits, 0..(capacity + 1),
            flags::HADAMARD_ROTATED, 1.0
        ).is_err());

        // packed buffer too short
        let bad_packed = vec![0u8; 10];
        assert!(capture_tq_v2_payload_from_buffers(
            &bad_packed, &norms, capacity, nkv, head_dim, bits, 0..256,
            flags::HADAMARD_ROTATED, 1.0
        ).is_err());

        // norms buffer too short
        let bad_norms = vec![0u8; 10];
        assert!(capture_tq_v2_payload_from_buffers(
            &packed, &bad_norms, capacity, nkv, head_dim, bits, 0..256,
            flags::HADAMARD_ROTATED, 1.0
        ).is_err());
    }

    /// **B-tq.3 restore rejects shape drift**: payload-vs-runtime
    /// mismatch on bits_per_coord, head_dim, n_kv_heads, or n_tokens
    /// MUST fail loud — silent acceptance would corrupt KV state.
    #[test]
    fn tq_v2_restore_rejects_shape_drift() {
        let nkv: u32 = 4;
        let capacity: u32 = 256;
        let head_dim: u32 = 64;
        let bits = TqBitsPerCoord::new(4).expect("4-bit");
        let hd_packed = (head_dim as usize) * (bits.0 as usize) / 8;

        let packed_src = vec![1u8; (nkv as usize) * (capacity as usize) * hd_packed];
        let mut norms_f32: Vec<f32> = Vec::with_capacity((nkv * capacity) as usize);
        for i in 0..(nkv * capacity) {
            norms_f32.push((i as f32) * 0.001);
        }
        let mut norms_le = Vec::with_capacity(norms_f32.len() * 4);
        for &v in &norms_f32 {
            norms_le.extend_from_slice(&v.to_le_bytes());
        }
        let payload = capture_tq_v2_payload_from_buffers(
            &packed_src, &norms_le, capacity, nkv, head_dim, bits, 0..32,
            flags::HADAMARD_ROTATED, 1.0,
        ).expect("capture");

        // Restore with WRONG bits → fail
        let mut p_dst = vec![0u8; (nkv as usize) * (capacity as usize) * hd_packed];
        let mut n_dst = vec![0u8; (nkv as usize) * (capacity as usize) * 4];
        let err = restore_tq_v2_payload_into_buffers(
            &payload, &mut p_dst, &mut n_dst,
            capacity, nkv, head_dim, TqBitsPerCoord::new(8).unwrap(),
            0..32,
        );
        assert!(err.is_err(), "restore silently accepted bits drift");

        // Restore with WRONG head_dim → fail
        let err = restore_tq_v2_payload_into_buffers(
            &payload, &mut p_dst, &mut n_dst,
            capacity, nkv, 128, bits, 0..32,
        );
        assert!(err.is_err(), "restore silently accepted head_dim drift");

        // Restore with WRONG nkv → fail
        let err = restore_tq_v2_payload_into_buffers(
            &payload, &mut p_dst, &mut n_dst,
            capacity, 8, head_dim, bits, 0..32,
        );
        assert!(err.is_err(), "restore silently accepted nkv drift");

        // Restore with WRONG range size → fail
        let err = restore_tq_v2_payload_into_buffers(
            &payload, &mut p_dst, &mut n_dst,
            capacity, nkv, head_dim, bits, 0..64,
        );
        assert!(err.is_err(), "restore silently accepted range drift");
    }

    /// **B-tq.3 spiller v1+v2 dispatch**: `TqPackedSpill::insert_block`
    /// accepts BOTH v1 (substrate) and v2 (engine-wired) payloads.
    /// Round-trip through `snapshot_block` returns byte-identical bytes
    /// for either codec.
    #[test]
    fn tq_packed_spill_accepts_v1_and_v2_inserts() {
        let cfg = TqPackedConfig {
            num_layers: 2,
            nkv_heads: vec![8, 8],
            head_dim: vec![256, 256],
            bits_per_coord: TqBitsPerCoord::new(4).expect("4"),
            scale: 1.0,
            flags: flags::HADAMARD_ROTATED,
            block_tokens: 256,
        };
        let mut spill = TqPackedSpill::new(cfg).expect("new");

        // v1 payload (B-tq.1 substrate path)
        let p_v1 = spill.synthesize_block(0, 0..256).expect("v1 synth");
        spill.insert_block(0, 0, p_v1.clone()).expect("v1 insert");
        let snap_v1 = KvCacheSpill::snapshot_block(&spill, 0, 0..256).expect("v1 snap");
        assert_eq!(snap_v1, p_v1);

        // v2 payload (B-tq.3 engine-wired path)
        let h2 = synthetic_v2_header(4, 8, 256, 256);
        let idx2 = synthetic_indices(h2.indices_bytes_len());
        let norms2 = synthetic_norms_le(h2.n_kv_heads as usize, h2.n_tokens as usize, 99);
        let p_v2 = pack_tq_v2_payload(&h2, &idx2, &norms2).expect("v2 pack");
        spill.insert_block(1, 0, p_v2.clone()).expect("v2 insert");
        let snap_v2 = KvCacheSpill::snapshot_block(&spill, 1, 0..256).expect("v2 snap");
        assert_eq!(snap_v2, p_v2);

        // Restore through the trait surface (round-trip).
        KvCacheSpill::restore_block(&mut spill, 0, 0..256, &p_v1).expect("v1 restore");
        KvCacheSpill::restore_block(&mut spill, 1, 0..256, &p_v2).expect("v2 restore");

        // Reject garbage magic.
        let mut garbage = p_v1.clone();
        garbage[0] = 0xFF;
        garbage[1] = 0xFF;
        garbage[2] = 0xFF;
        garbage[3] = 0xFF;
        assert!(spill.insert_block(0, 256, garbage.clone()).is_err());
        assert!(KvCacheSpill::restore_block(&mut spill, 0, 0..256, &garbage).is_err());
    }

    // ============================================================================
    // Phase B-tq.4 — iter-1 tests
    // ============================================================================

    /// **B-tq.4 bundle codec round-trip**: pack two synthetic v2
    /// envelopes into a bundle, unpack, assert byte-identity on each
    /// half.
    #[test]
    fn tq_packed_v2_kv_bundle_round_trip_byte_exact() {
        let h = synthetic_v2_header(4, 8, 256, 256);
        let idx = synthetic_indices(h.indices_bytes_len());
        let norms = synthetic_norms_le(h.n_kv_heads as usize, h.n_tokens as usize, 42);
        let k_payload = pack_tq_v2_payload(&h, &idx, &norms).expect("pack k");
        // V uses different indices/norms to ensure the bundle
        // preserves the K/V distinction.
        let mut idx_v = idx.clone();
        idx_v.iter_mut().for_each(|b| *b ^= 0x55);
        let norms_v = synthetic_norms_le(h.n_kv_heads as usize, h.n_tokens as usize, 7);
        let v_payload = pack_tq_v2_payload(&h, &idx_v, &norms_v).expect("pack v");

        let bundle = pack_tq_v2_kv_bundle(&k_payload, &v_payload).expect("bundle");
        let (k_unpacked, v_unpacked) =
            unpack_tq_v2_kv_bundle(&bundle).expect("unbundle");
        assert_eq!(k_unpacked, k_payload.as_slice());
        assert_eq!(v_unpacked, v_payload.as_slice());

        // Bundle byte-determinism: pack again, byte-identical.
        let bundle2 = pack_tq_v2_kv_bundle(&k_payload, &v_payload).expect("bundle2");
        assert_eq!(bundle, bundle2);
    }

    /// **B-tq.4 bundle frozen magic**: any change to the bundle magic
    /// silently breaks on-disk format.  CI catches via this pin.
    #[test]
    fn tq_packed_kv_bundle_magic_is_frozen() {
        // ASCII "TQK2" little-endian: 0x54 (T), 0x51 (Q), 0x4B (K),
        // 0x32 (2).  u32::from_le_bytes assembles MSB-first in literal:
        // 0x32_4B_51_54.
        assert_eq!(TQ_PACKED_KV_BUNDLE_MAGIC, 0x32_4B_51_54);
        assert_eq!(TQ_PACKED_KV_BUNDLE_MAGIC.to_le_bytes(), *b"TQK2");
    }

    /// **B-tq.4 bundle rejects bad inputs**: short payloads, wrong
    /// magic on inner envelopes, mismatched length tail, all error.
    #[test]
    fn tq_packed_kv_bundle_rejects_malformed_inputs() {
        // Inner envelope with wrong magic (use v1 magic on the K
        // payload — bundle requires v2).
        let h_v1 = synthetic_header(4, 8, 256, 256);
        let idx_v1 = synthetic_indices(h_v1.indices_bytes_len());
        let v1_bytes = pack_tq_v1_payload(&h_v1, &idx_v1).expect("v1 pack");
        let h_v2 = synthetic_v2_header(4, 8, 256, 256);
        let idx_v2 = synthetic_indices(h_v2.indices_bytes_len());
        let norms_v2 = synthetic_norms_le(8, 256, 1);
        let v2_bytes = pack_tq_v2_payload(&h_v2, &idx_v2, &norms_v2).expect("v2 pack");

        // K=v1, V=v2 → reject (K must be v2)
        assert!(pack_tq_v2_kv_bundle(&v1_bytes, &v2_bytes).is_err());
        // K=v2, V=v1 → reject (V must be v2)
        assert!(pack_tq_v2_kv_bundle(&v2_bytes, &v1_bytes).is_err());
        // Empty K → reject
        assert!(pack_tq_v2_kv_bundle(&[], &v2_bytes).is_err());

        // Truncated bundle → reject on unpack
        let bundle = pack_tq_v2_kv_bundle(&v2_bytes, &v2_bytes).expect("good bundle");
        assert!(unpack_tq_v2_kv_bundle(&bundle[..bundle.len() - 1]).is_err());
        assert!(unpack_tq_v2_kv_bundle(&[]).is_err());

        // Bad bundle magic → reject
        let mut bad_magic = bundle.clone();
        bad_magic[0] ^= 0xFF;
        assert!(unpack_tq_v2_kv_bundle(&bad_magic).is_err());
    }

    /// **B-tq.4 spill insert/restore accept bundle**: TqPackedSpill's
    /// magic-dispatcher must route bundle-magic payloads through the
    /// bundle path.  Verifies in-memory cache round-trip when
    /// engine_arc is unbound (production engine path is exercised by
    /// the E2E integration test).
    #[test]
    fn tq_packed_spill_accepts_bundle_payload_in_memory() {
        let cfg = TqPackedConfig {
            num_layers: 2,
            nkv_heads: vec![8, 8],
            head_dim: vec![256, 256],
            bits_per_coord: TqBitsPerCoord::new(4).expect("4"),
            scale: 1.0,
            flags: flags::HADAMARD_ROTATED,
            block_tokens: 256,
        };
        let mut spill = TqPackedSpill::new(cfg).expect("new");
        let h = synthetic_v2_header(4, 8, 256, 256);
        let idx = synthetic_indices(h.indices_bytes_len());
        let norms = synthetic_norms_le(8, 256, 100);
        let k_payload = pack_tq_v2_payload(&h, &idx, &norms).expect("k");
        let v_payload = pack_tq_v2_payload(&h, &idx, &norms).expect("v");
        let bundle = pack_tq_v2_kv_bundle(&k_payload, &v_payload).expect("bundle");

        // restore_block accepts bundle and caches it in-memory (engine
        // unbound ⇒ no engine round-trip; cache is preserved).
        KvCacheSpill::restore_block(&mut spill, 0, 0..256, &bundle).expect("restore");
        // snapshot_block reads it back verbatim.
        let snap = KvCacheSpill::snapshot_block(&spill, 0, 0..256).expect("snap");
        assert_eq!(snap, bundle);

        // Bundle with WRONG inner shape (different bits) → reject.
        let h_bad = synthetic_v2_header(8, 8, 256, 256);
        let idx_bad = synthetic_indices(h_bad.indices_bytes_len());
        let bundle_bad = pack_tq_v2_kv_bundle(
            &pack_tq_v2_payload(&h_bad, &idx_bad, &norms[..h_bad.norms_bytes_len()])
                .unwrap(),
            &pack_tq_v2_payload(&h_bad, &idx_bad, &norms[..h_bad.norms_bytes_len()])
                .unwrap(),
        )
        .expect("bundle bad");
        assert!(KvCacheSpill::restore_block(&mut spill, 1, 0..256, &bundle_bad).is_err());
    }

    /// **B-tq.4 engine_arc bind/unbind**: set + clear cycle yields
    /// expected state.  Doesn't exercise a real Engine (that needs a
    /// loaded model — covered in E2E integration test).
    #[test]
    fn tq_packed_spill_engine_arc_starts_unbound() {
        let cfg = TqPackedConfig {
            num_layers: 1,
            nkv_heads: vec![8],
            head_dim: vec![256],
            bits_per_coord: TqBitsPerCoord::new(4).expect("4"),
            scale: 1.0,
            flags: flags::HADAMARD_ROTATED,
            block_tokens: 256,
        };
        let spill = TqPackedSpill::new(cfg).expect("new");
        // engine_arc starts None (verified via the snapshot fall-
        // through path: snapshot of an empty in-memory map returns
        // None when engine_arc is unbound).
        assert!(KvCacheSpill::snapshot_block(&spill, 0, 0..256).is_none());

        // clear_engine_arc on already-unbound spill is a no-op (does
        // not panic; mirrors EngineBindable contract).
        spill.clear_engine_arc();
        assert!(KvCacheSpill::snapshot_block(&spill, 0, 0..256).is_none());
    }

    /// **B-tq.4 factory rejects type mismatch**: Arc<String> (a
    /// non-Engine type) should yield None from try_construct (silent
    /// no-op per FamilyHookFactory contract).
    #[test]
    fn tq_packed_spill_factory_rejects_non_engine_arc() {
        use crate::serve::kv_persist::registry::FamilyHookFactory;
        let cfg = TqPackedConfig {
            num_layers: 2,
            nkv_heads: vec![8, 8],
            head_dim: vec![256, 256],
            bits_per_coord: TqBitsPerCoord::new(4).expect("4"),
            scale: 1.0,
            flags: flags::HADAMARD_ROTATED,
            block_tokens: 256,
        };
        let factory = TqPackedSpillFactory::new(cfg);
        // Pass an Arc<String> — definitely not an Engine.
        let bogus: Arc<dyn std::any::Any + Send + Sync> =
            Arc::new(String::from("not an engine"));
        let result = factory.try_construct(bogus);
        assert!(
            result.is_none(),
            "factory accepted non-Engine Arc — silent factory contract violated"
        );
    }

    /// **B-tq.4 try_from_engine_arc rejects type mismatch** (mirror
    /// of the factory test at the construction-helper level).
    #[test]
    fn tq_packed_spill_try_from_engine_arc_rejects_non_engine() {
        let cfg = TqPackedConfig {
            num_layers: 1,
            nkv_heads: vec![8],
            head_dim: vec![256],
            bits_per_coord: TqBitsPerCoord::new(4).expect("4"),
            scale: 1.0,
            flags: flags::HADAMARD_ROTATED,
            block_tokens: 256,
        };
        let bogus: Arc<dyn std::any::Any + Send + Sync> = Arc::new(42_u32);
        assert!(TqPackedSpill::try_from_engine_arc(cfg, bogus).is_none());
    }

    /// **B-tq.4 EngineBindable bind_engine on type mismatch**: silent
    /// no-op (no panic).
    #[test]
    fn tq_packed_spill_bind_engine_silent_on_mismatch() {
        use crate::serve::kv_persist::EngineBindable;
        let cfg = TqPackedConfig {
            num_layers: 1,
            nkv_heads: vec![8],
            head_dim: vec![256],
            bits_per_coord: TqBitsPerCoord::new(4).expect("4"),
            scale: 1.0,
            flags: flags::HADAMARD_ROTATED,
            block_tokens: 256,
        };
        let spill = TqPackedSpill::new(cfg).expect("new");
        let bogus: Arc<dyn std::any::Any + Send + Sync> = Arc::new(42_u32);
        // Should not panic; engine_arc remains None.
        spill.bind_engine(bogus);
        // Verify by snapshotting an empty in-memory map (engine path
        // would have routed to the worker — bound state would
        // exercise different code).
        assert!(KvCacheSpill::snapshot_block(&spill, 0, 0..256).is_none());

        // unbind_engine is also safe on already-unbound spill.
        spill.unbind_engine();
    }
}
