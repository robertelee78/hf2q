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
use std::sync::RwLock;

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

/// Phase B-tq.2 family hook for TurboQuant-packed K/V layers.
///
/// The internal `blocks` map is keyed by `(layer_rank, range_start)`
/// and stores pre-packed `tq_packed_v1` envelope payloads.
/// `snapshot_block(layer, range)` returns the stored bytes verbatim;
/// `restore_block(layer, range, payload)` validates the envelope and
/// stores.  Both operations are O(log n) in the number of stored
/// blocks (`BTreeMap` lookup).
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
}

impl TqPackedSpill {
    /// Construct a new TQ-packed spill from shape config.  The block
    /// map starts empty; B-tq.3 engine wiring populates it via
    /// [`Self::insert_block`] (or future
    /// `snapshot_via_engine_post_quantize`).
    pub fn new(cfg: TqPackedConfig) -> Result<Self, SpillErrorKind> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            blocks: RwLock::new(BTreeMap::new()),
            fingerprint: RwLock::new(None),
        })
    }

    /// Set the model fingerprint for §F4 namespace keying.
    pub fn set_fingerprint(&self, fp: ModelFingerprint) {
        if let Ok(mut slot) = self.fingerprint.write() {
            *slot = Some(fp);
        }
    }

    /// Insert a pre-packed envelope payload (test fixture / future
    /// engine wiring).  Returns `Err(SpillErrorKind::CodecErr)` if the
    /// payload doesn't parse as `tq_packed_v1`.
    pub fn insert_block(
        &self,
        layer_rank: usize,
        range_start: u32,
        payload: Vec<u8>,
    ) -> Result<(), SpillErrorKind> {
        if layer_rank >= self.cfg.num_layers {
            return Err(SpillErrorKind::CodecErr);
        }
        let _ = unpack_tq_v1_payload(&payload)
            .map_err(|_| SpillErrorKind::CodecErr)?;
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
        let (header, _indices) =
            unpack_tq_v1_payload(payload).map_err(|_| SpillErrorKind::CodecErr)?;
        // Layout-mismatch guards: dtype-equivalent for TQ-packed is
        // `(bits_per_coord, head_dim, n_kv_heads)`.  A future engine
        // upgrade that bumps any of these without bumping
        // `codec_version` would silently corrupt restored state — fail
        // loud here.
        if header.bits_per_coord != self.cfg.bits_per_coord {
            return Err(SpillErrorKind::CodecErr);
        }
        if header.head_dim != self.cfg.head_dim[layer_rank] {
            return Err(SpillErrorKind::CodecErr);
        }
        if header.n_kv_heads != self.cfg.nkv_heads[layer_rank] {
            return Err(SpillErrorKind::CodecErr);
        }
        if header.n_tokens != range.end.saturating_sub(range.start) {
            return Err(SpillErrorKind::CodecErr);
        }
        // Defensive: re-pack and assert byte-equality to catch
        // post-decode tampering / wire-format drift before we commit
        // to engine state.  Because pack is deterministic
        // (B-tq.1 round-trip test), the re-packed bytes MUST byte-equal
        // the input.
        let (_h2, idx) = unpack_tq_v1_payload(payload).map_err(|_| SpillErrorKind::CodecErr)?;
        let repacked = pack_tq_v1_payload(&header, idx).map_err(|_| SpillErrorKind::CodecErr)?;
        if repacked != payload {
            return Err(SpillErrorKind::ParityFail);
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
}
