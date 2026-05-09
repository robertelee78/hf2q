//! ADR-027 Phase A — Qwen3.5/3.6 hybrid KV-cache snapshot persistor.
//!
//! Wraps the existing `HybridKvCache::snapshot()` substrate (already shipped
//! by ADR-017 Phase E.a B.2) with a serialize/deserialize codec so cold
//! processes can resume the in-memory `LcpRegistry<HybridKvCacheSnapshot>`
//! from disk.
//!
//! **NOT a `KvCacheSpill` impl** — qwen35's hybrid cache shape (full-attn
//! ring buffers + DeltaNet ping-pong scratch state) doesn't fit the
//! spiller's `(layer_rank, range)` block contract. See `families/mod.rs:15-23`
//! and ADR-027 §2.0 for the Chesterton's-fence rationale; the snapshot-
//! based path mirrors what `LcpRegistry` already does in memory.
//!
//! # Iter sequence (per ADR-027 §6)
//!
//! - **Iter 2 (this commit)**: serialize/deserialize for full-attn slots
//!   only. Linear-attn + MTP slots are accepted by the envelope but the
//!   round-trip is a no-op for them (deserialize allocates them as
//!   freshly-zeroed buffers matching the config; this is correct but
//!   unhelpful — iter 3+4 fill in the real bytes).
//! - **Iter 3**: linear-attn slot bytes (with swap_parity hint).
//! - **Iter 4**: MTP slot bytes.
//! - **Iter 5**: `Qwen35DiskPersistor` LcpRegistry write-through to disk.
//! - **Iter 11**: `full_attn_codec_tag = 1` → TQ-v2 encoded full-attn bytes.
//!
//! # On-disk envelope (codec_version=1)
//!
//! ```text
//! [magic: 4 bytes "QH35"]
//! [codec_version: u32 LE = 1]
//! [n_full_attn: u32 LE]
//! [n_linear_attn: u32 LE]            # iter-2: matches config but bytes are zero
//! [mtp_present: u8]                  # iter-2: matches config but bytes are zero
//! [full_attn_codec_tag: u8]          # iter-2: 0 = F32 dense
//! [n_seqs: u32 LE]
//! [reserved: u16 LE = 0]
//!
//! Per full-attn slot (n_full_attn iterations):
//!   [slot_idx: u32 LE]
//!   [shape: [u64; 4] LE]             # K and V share shape
//!   [k_byte_len: u64 LE]
//!   [v_byte_len: u64 LE]
//!   [current_len: u32 × n_seqs LE]
//!   [k_bytes: k_byte_len]
//!   [v_bytes: v_byte_len]
//!
//! Per linear-attn slot (iter 3 will populate; iter 2 reserves layout):
//!   ... (codec_version stays 1; iter 3 adds these slots without bumping)
//!
//! MTP slot (iter 4):
//!   ... (same)
//! ```
//!
//! Iter 3/4 add slots within `codec_version=1`; iter 11's TQ tag bump is
//! still version 1 (the tag byte is the discriminator, frozen-by-position).

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

use crate::inference::models::qwen35::kv_cache::{
    HybridKvCacheSnapshot, MtpKvSnapshot,
};

/// Magic bytes prefixing every QH35 (Qwen3.5 Hybrid) envelope. ASCII for
/// "QH35" to make hex dumps trivial to spot-read.
pub const QH35_MAGIC: [u8; 4] = *b"QH35";

/// Current codec version. Iter-2 ships v1; iter-11 reserves a follow-up
/// to extend the full_attn_codec_tag namespace if the tag byte runs out
/// of room (8 codec variants is more than we will ever need).
pub const QH35_CODEC_VERSION: u32 = 1;

/// Full-attn codec discriminator. Iter-2 ships only F32Dense; iter-11
/// wires TqV2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FullAttnCodec {
    F32Dense = 0,
    // TqV2 = 1,  // ADR-027 iter 11
}

impl FullAttnCodec {
    fn from_u8(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::F32Dense),
            other => Err(anyhow!(
                "QH35 envelope: unknown full_attn_codec_tag = {other} (expected 0)"
            )),
        }
    }
}

/// Shape configuration for the persistor — the values that must match
/// between serialize-time and deserialize-time. Captured from
/// `Qwen35Config` + `HybridKvCache` at engine load and threaded through
/// to deserialize so we can validate (a) the envelope on disk was
/// written by a compatible producer, and (b) the live cache is allocated
/// to receive the bytes.
#[derive(Clone, Debug)]
pub struct Qwen35HybridConfig {
    /// Number of full-attention layers.
    pub n_full_attn: u32,
    /// Number of linear-attention (DeltaNet) layers.
    pub n_linear_attn: u32,
    /// `true` iff the model has an MTP head (nextn_predict_layers > 0).
    pub has_mtp: bool,
    /// Number of sequences in the cache (1 for single-request inference).
    pub n_seqs: u32,
    /// Per-full-attn-slot shape `[n_seqs, n_kv_heads, max_seq_len, head_dim]`.
    pub full_attn_shape: [u64; 4],
    /// Encoder's choice of full-attn codec. Iter 2 always uses F32Dense.
    pub full_attn_codec: FullAttnCodec,
}

impl Qwen35HybridConfig {
    /// Validate that an inbound config matches this one in every shape
    /// field. Used at deserialize to fail-fast on producer/consumer
    /// drift before allocating buffers.
    fn assert_matches(&self, other: &Self) -> Result<()> {
        ensure!(
            self.n_full_attn == other.n_full_attn,
            "QH35 config drift: n_full_attn = {} vs {}",
            self.n_full_attn,
            other.n_full_attn
        );
        ensure!(
            self.n_linear_attn == other.n_linear_attn,
            "QH35 config drift: n_linear_attn = {} vs {}",
            self.n_linear_attn,
            other.n_linear_attn
        );
        ensure!(
            self.has_mtp == other.has_mtp,
            "QH35 config drift: has_mtp = {} vs {}",
            self.has_mtp,
            other.has_mtp
        );
        ensure!(
            self.n_seqs == other.n_seqs,
            "QH35 config drift: n_seqs = {} vs {}",
            self.n_seqs,
            other.n_seqs
        );
        ensure!(
            self.full_attn_shape == other.full_attn_shape,
            "QH35 config drift: full_attn_shape = {:?} vs {:?}",
            self.full_attn_shape,
            other.full_attn_shape
        );
        ensure!(
            self.full_attn_codec == other.full_attn_codec,
            "QH35 config drift: full_attn_codec = {:?} vs {:?}",
            self.full_attn_codec,
            other.full_attn_codec
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tiny LE write helpers (mirror format.rs's read_u32 style — keep deps low).
// ---------------------------------------------------------------------------

#[inline]
fn write_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

#[inline]
fn write_u16_le(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_u32_le(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_u64_le(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn read_u8(buf: &[u8], cursor: &mut usize) -> Result<u8> {
    let pos = *cursor;
    let v = *buf
        .get(pos)
        .ok_or_else(|| anyhow!("QH35 read_u8 OOB at offset {pos} (len={})", buf.len()))?;
    *cursor = pos + 1;
    Ok(v)
}

#[inline]
fn read_u16_le(buf: &[u8], cursor: &mut usize) -> Result<u16> {
    let pos = *cursor;
    let end = pos
        .checked_add(2)
        .ok_or_else(|| anyhow!("QH35 read_u16_le offset overflow"))?;
    ensure!(
        end <= buf.len(),
        "QH35 read_u16_le OOB ({end} > {})",
        buf.len()
    );
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&buf[pos..end]);
    *cursor = end;
    Ok(u16::from_le_bytes(bytes))
}

#[inline]
fn read_u32_le(buf: &[u8], cursor: &mut usize) -> Result<u32> {
    let pos = *cursor;
    let end = pos
        .checked_add(4)
        .ok_or_else(|| anyhow!("QH35 read_u32_le offset overflow"))?;
    ensure!(
        end <= buf.len(),
        "QH35 read_u32_le OOB ({end} > {})",
        buf.len()
    );
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&buf[pos..end]);
    *cursor = end;
    Ok(u32::from_le_bytes(bytes))
}

#[inline]
fn read_u64_le(buf: &[u8], cursor: &mut usize) -> Result<u64> {
    let pos = *cursor;
    let end = pos
        .checked_add(8)
        .ok_or_else(|| anyhow!("QH35 read_u64_le offset overflow"))?;
    ensure!(
        end <= buf.len(),
        "QH35 read_u64_le OOB ({end} > {})",
        buf.len()
    );
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&buf[pos..end]);
    *cursor = end;
    Ok(u64::from_le_bytes(bytes))
}

#[inline]
fn read_bytes<'a>(buf: &'a [u8], cursor: &mut usize, n: usize) -> Result<&'a [u8]> {
    let pos = *cursor;
    let end = pos
        .checked_add(n)
        .ok_or_else(|| anyhow!("QH35 read_bytes offset overflow"))?;
    ensure!(
        end <= buf.len(),
        "QH35 read_bytes OOB ({end} > {})",
        buf.len()
    );
    *cursor = end;
    Ok(&buf[pos..end])
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Serialize a `HybridKvCacheSnapshot` into the QH35 envelope.
///
/// Iter-2 scope: serializes full-attn slot bytes verbatim; linear-attn and
/// MTP slot bytes are NOT emitted yet (the envelope reserves their slot
/// counts so iter-3/4 add them without breaking compat). Round-trip on
/// linear/MTP-only snapshots in iter-2 yields freshly-zeroed buffers on
/// the deserialize side — correct envelope but unhelpful payload, which
/// is the whole point of the per-iter sequencing.
pub fn serialize_hybrid_snapshot(
    snapshot: &HybridKvCacheSnapshot,
    cfg: &Qwen35HybridConfig,
) -> Result<Vec<u8>> {
    // Validate snapshot shape against config before emitting any bytes.
    ensure!(
        snapshot.full_attn_k.len() == cfg.n_full_attn as usize,
        "QH35 serialize: snapshot.full_attn_k.len() = {} but cfg.n_full_attn = {}",
        snapshot.full_attn_k.len(),
        cfg.n_full_attn
    );
    ensure!(
        snapshot.full_attn_v.len() == cfg.n_full_attn as usize,
        "QH35 serialize: snapshot.full_attn_v.len() = {} but cfg.n_full_attn = {}",
        snapshot.full_attn_v.len(),
        cfg.n_full_attn
    );
    ensure!(
        snapshot.full_attn_current_len.len() == cfg.n_full_attn as usize,
        "QH35 serialize: snapshot.full_attn_current_len.len() = {} but cfg.n_full_attn = {}",
        snapshot.full_attn_current_len.len(),
        cfg.n_full_attn
    );
    ensure!(
        snapshot.linear_conv.len() == cfg.n_linear_attn as usize,
        "QH35 serialize: snapshot.linear_conv.len() = {} but cfg.n_linear_attn = {}",
        snapshot.linear_conv.len(),
        cfg.n_linear_attn
    );
    ensure!(
        snapshot.linear_recurrent.len() == cfg.n_linear_attn as usize,
        "QH35 serialize: snapshot.linear_recurrent.len() = {} but cfg.n_linear_attn = {}",
        snapshot.linear_recurrent.len(),
        cfg.n_linear_attn
    );
    ensure!(
        snapshot.mtp.is_some() == cfg.has_mtp,
        "QH35 serialize: snapshot.mtp.is_some() = {} but cfg.has_mtp = {}",
        snapshot.mtp.is_some(),
        cfg.has_mtp
    );

    let mut out: Vec<u8> = Vec::new();

    // --- Header (16 bytes including the magic) ---
    out.extend_from_slice(&QH35_MAGIC);
    write_u32_le(&mut out, QH35_CODEC_VERSION);
    write_u32_le(&mut out, cfg.n_full_attn);
    write_u32_le(&mut out, cfg.n_linear_attn);
    write_u8(&mut out, if cfg.has_mtp { 1 } else { 0 });
    write_u8(&mut out, cfg.full_attn_codec as u8);
    write_u32_le(&mut out, cfg.n_seqs);
    write_u16_le(&mut out, 0); // reserved

    // --- Per full-attn slot ---
    for slot_idx in 0..cfg.n_full_attn as usize {
        let k = &snapshot.full_attn_k[slot_idx];
        let v = &snapshot.full_attn_v[slot_idx];
        let current_len = &snapshot.full_attn_current_len[slot_idx];

        // Per-slot validation against config.
        ensure!(
            k.shape().len() == 4,
            "QH35 serialize: full_attn[{slot_idx}].k shape rank {} != 4",
            k.shape().len()
        );
        ensure!(
            v.shape().len() == 4,
            "QH35 serialize: full_attn[{slot_idx}].v shape rank {} != 4",
            v.shape().len()
        );
        let k_shape: [u64; 4] = [
            k.shape()[0] as u64,
            k.shape()[1] as u64,
            k.shape()[2] as u64,
            k.shape()[3] as u64,
        ];
        ensure!(
            k_shape == cfg.full_attn_shape,
            "QH35 serialize: full_attn[{slot_idx}].k shape {:?} != cfg.full_attn_shape {:?}",
            k_shape,
            cfg.full_attn_shape
        );
        let v_shape: [u64; 4] = [
            v.shape()[0] as u64,
            v.shape()[1] as u64,
            v.shape()[2] as u64,
            v.shape()[3] as u64,
        ];
        ensure!(
            v_shape == cfg.full_attn_shape,
            "QH35 serialize: full_attn[{slot_idx}].v shape {:?} != cfg.full_attn_shape {:?}",
            v_shape,
            cfg.full_attn_shape
        );
        ensure!(
            current_len.len() == cfg.n_seqs as usize,
            "QH35 serialize: full_attn[{slot_idx}].current_len.len() = {} != n_seqs = {}",
            current_len.len(),
            cfg.n_seqs
        );

        let k_bytes: &[u8] = k
            .as_slice::<u8>()
            .map_err(|e| anyhow!("QH35 serialize: full_attn[{slot_idx}].k as_slice: {e}"))?;
        let v_bytes: &[u8] = v
            .as_slice::<u8>()
            .map_err(|e| anyhow!("QH35 serialize: full_attn[{slot_idx}].v as_slice: {e}"))?;
        ensure!(
            k_bytes.len() == k.byte_len(),
            "QH35 serialize: full_attn[{slot_idx}].k as_slice.len() = {} != byte_len = {}",
            k_bytes.len(),
            k.byte_len()
        );
        ensure!(
            v_bytes.len() == v.byte_len(),
            "QH35 serialize: full_attn[{slot_idx}].v as_slice.len() = {} != byte_len = {}",
            v_bytes.len(),
            v.byte_len()
        );

        write_u32_le(&mut out, slot_idx as u32);
        for &dim in &k_shape {
            write_u64_le(&mut out, dim);
        }
        write_u64_le(&mut out, k_bytes.len() as u64);
        write_u64_le(&mut out, v_bytes.len() as u64);
        for &cl in current_len.iter() {
            write_u32_le(&mut out, cl);
        }
        out.extend_from_slice(k_bytes);
        out.extend_from_slice(v_bytes);
    }

    // Linear-attn slot bytes: ITER-3 SCOPE. The envelope's
    // `n_linear_attn` field already says how many slots exist; readers
    // built before iter-3 reach EOF after the last full-attn slot and
    // synthesize zeroed linear/mtp buffers (deserialize_hybrid_snapshot
    // below handles this honestly — see its iter-2 note).

    // MTP: same — ITER-4 SCOPE.

    Ok(out)
}

/// Deserialize a QH35 envelope back into a `HybridKvCacheSnapshot` against
/// a freshly-allocated set of buffers via `device`.
///
/// Iter-2 scope: full-attn slot bytes are copied back verbatim; linear-attn
/// and MTP buffers are allocated freshly-zeroed (no payload bytes to read).
/// Iter-3 + iter-4 extend this to read real payload bytes for those slots.
pub fn deserialize_hybrid_snapshot(
    bytes: &[u8],
    cfg: &Qwen35HybridConfig,
    device: &MlxDevice,
) -> Result<HybridKvCacheSnapshot> {
    let mut cursor = 0usize;

    let magic = read_bytes(bytes, &mut cursor, 4)?;
    ensure!(
        magic == QH35_MAGIC,
        "QH35 deserialize: bad magic {:?} (expected {:?})",
        magic,
        QH35_MAGIC
    );

    let codec_version = read_u32_le(bytes, &mut cursor)?;
    ensure!(
        codec_version == QH35_CODEC_VERSION,
        "QH35 deserialize: unsupported codec_version {} (expected {})",
        codec_version,
        QH35_CODEC_VERSION
    );

    let header_cfg = Qwen35HybridConfig {
        n_full_attn: read_u32_le(bytes, &mut cursor)?,
        n_linear_attn: read_u32_le(bytes, &mut cursor)?,
        has_mtp: read_u8(bytes, &mut cursor)? != 0,
        full_attn_codec: FullAttnCodec::from_u8(read_u8(bytes, &mut cursor)?)?,
        n_seqs: read_u32_le(bytes, &mut cursor)?,
        full_attn_shape: cfg.full_attn_shape, // shape is per-slot in body; populated below
    };
    let _reserved = read_u16_le(bytes, &mut cursor)?;

    // Validate header against expected config (shape is checked per-slot).
    {
        let mut h = header_cfg.clone();
        h.full_attn_shape = cfg.full_attn_shape;
        h.assert_matches(cfg)
            .context("QH35 deserialize: envelope header / runtime config mismatch")?;
    }

    let mut full_attn_k: Vec<MlxBuffer> = Vec::with_capacity(cfg.n_full_attn as usize);
    let mut full_attn_v: Vec<MlxBuffer> = Vec::with_capacity(cfg.n_full_attn as usize);
    let mut full_attn_current_len: Vec<Vec<u32>> = Vec::with_capacity(cfg.n_full_attn as usize);

    for expected_slot in 0..cfg.n_full_attn as usize {
        let slot_idx = read_u32_le(bytes, &mut cursor)? as usize;
        ensure!(
            slot_idx == expected_slot,
            "QH35 deserialize: full_attn slot order mismatch — got {} expected {}",
            slot_idx,
            expected_slot
        );

        let mut shape_arr = [0u64; 4];
        for dim in &mut shape_arr {
            *dim = read_u64_le(bytes, &mut cursor)?;
        }
        ensure!(
            shape_arr == cfg.full_attn_shape,
            "QH35 deserialize: full_attn[{slot_idx}] shape on disk {:?} != cfg {:?}",
            shape_arr,
            cfg.full_attn_shape
        );
        let k_byte_len = read_u64_le(bytes, &mut cursor)? as usize;
        let v_byte_len = read_u64_le(bytes, &mut cursor)? as usize;

        let mut current_len = Vec::with_capacity(cfg.n_seqs as usize);
        for _ in 0..cfg.n_seqs {
            current_len.push(read_u32_le(bytes, &mut cursor)?);
        }

        let k_src = read_bytes(bytes, &mut cursor, k_byte_len)?;
        let v_src = read_bytes(bytes, &mut cursor, v_byte_len)?;

        let mlx_shape: Vec<usize> = shape_arr.iter().map(|d| *d as usize).collect();
        let dtype = full_attn_dtype_for_codec(header_cfg.full_attn_codec);
        let mut k_buf = device
            .alloc_buffer(k_byte_len, dtype, mlx_shape.clone())
            .map_err(|e| anyhow!("QH35 deserialize: alloc full_attn[{slot_idx}].k: {e}"))?;
        let mut v_buf = device
            .alloc_buffer(v_byte_len, dtype, mlx_shape)
            .map_err(|e| anyhow!("QH35 deserialize: alloc full_attn[{slot_idx}].v: {e}"))?;
        {
            let k_dst = k_buf
                .as_mut_slice::<u8>()
                .map_err(|e| anyhow!("QH35 deserialize: full_attn[{slot_idx}].k mut_slice: {e}"))?;
            ensure!(
                k_dst.len() == k_src.len(),
                "QH35 deserialize: full_attn[{slot_idx}].k dst.len() = {} != src.len() = {}",
                k_dst.len(),
                k_src.len()
            );
            k_dst.copy_from_slice(k_src);
        }
        {
            let v_dst = v_buf
                .as_mut_slice::<u8>()
                .map_err(|e| anyhow!("QH35 deserialize: full_attn[{slot_idx}].v mut_slice: {e}"))?;
            ensure!(
                v_dst.len() == v_src.len(),
                "QH35 deserialize: full_attn[{slot_idx}].v dst.len() = {} != src.len() = {}",
                v_dst.len(),
                v_src.len()
            );
            v_dst.copy_from_slice(v_src);
        }
        full_attn_k.push(k_buf);
        full_attn_v.push(v_buf);
        full_attn_current_len.push(current_len);
    }

    // Linear-attn slots: iter-2 reads NO bytes; allocates fresh-zeroed
    // buffers matching the config. Iter-3 fills in the per-slot read
    // path. Until then this is the honest, mantra-aligned behavior:
    // the envelope says how many slots exist, but iter-2 cannot
    // restore their values — we emit empty placeholders rather than
    // pretend we have them.
    let linear_conv: Vec<MlxBuffer> = Vec::new(); // iter-2: empty;  iter-3 populates
    let linear_recurrent: Vec<MlxBuffer> = Vec::new();

    // MTP: same — iter-4 scope.
    let mtp: Option<MtpKvSnapshot> = None;

    // If iter-3 / iter-4 have NOT shipped yet but this envelope was
    // produced with non-zero linear / mtp counts, the snapshot we
    // return is inconsistent with `cfg`. iter-2 deliberately surfaces
    // this rather than silently returning an empty snapshot — callers
    // get a hard error so they don't ship broken cache state to the
    // worker thread.
    if cfg.n_linear_attn > 0 {
        return Err(anyhow!(
            "QH35 deserialize: cfg.n_linear_attn = {} but linear-attn read \
             path is iter-3 scope (not yet implemented). Until iter-3 ships, \
             do not call this function with linear-attn slots present.",
            cfg.n_linear_attn
        ));
    }
    if cfg.has_mtp {
        return Err(anyhow!(
            "QH35 deserialize: cfg.has_mtp = true but MTP read path is iter-4 \
             scope (not yet implemented). Until iter-4 ships, do not call this \
             function with MTP present."
        ));
    }

    Ok(HybridKvCacheSnapshot {
        full_attn_k,
        full_attn_v,
        full_attn_current_len,
        mtp,
        linear_conv,
        linear_recurrent,
    })
}

/// dtype the persistor expects to find in the buffer for a given codec.
/// Iter-2: F32Dense → DType::F32. Iter-11 will branch for TqV2.
fn full_attn_dtype_for_codec(codec: FullAttnCodec) -> DType {
    match codec {
        FullAttnCodec::F32Dense => DType::F32,
    }
}

// ---------------------------------------------------------------------------
// Tests (synthetic-state round-trip — AC-A1)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small synthetic full-attn-only `HybridKvCacheSnapshot` with
    /// deterministic byte patterns so the round-trip is byte-equality
    /// verifiable without GPU-state randomness.
    fn synth_full_attn_only_snapshot(
        device: &MlxDevice,
        cfg: &Qwen35HybridConfig,
    ) -> HybridKvCacheSnapshot {
        let elems_per_slot: usize = cfg.full_attn_shape.iter().product::<u64>() as usize;
        let bytes_per_slot = elems_per_slot * std::mem::size_of::<f32>();
        let shape_usize: Vec<usize> =
            cfg.full_attn_shape.iter().map(|d| *d as usize).collect();

        let mut full_attn_k: Vec<MlxBuffer> = Vec::with_capacity(cfg.n_full_attn as usize);
        let mut full_attn_v: Vec<MlxBuffer> = Vec::with_capacity(cfg.n_full_attn as usize);
        let mut full_attn_current_len: Vec<Vec<u32>> =
            Vec::with_capacity(cfg.n_full_attn as usize);

        for slot in 0..cfg.n_full_attn as usize {
            let mut k = device
                .alloc_buffer(bytes_per_slot, DType::F32, shape_usize.clone())
                .expect("alloc k");
            let mut v = device
                .alloc_buffer(bytes_per_slot, DType::F32, shape_usize.clone())
                .expect("alloc v");
            // Deterministic byte pattern: K filled with (slot * 7 + i) mod 251,
            // V filled with (slot * 11 + i) mod 251.
            {
                let k_dst = k.as_mut_slice::<u8>().expect("k mut_slice");
                for (i, b) in k_dst.iter_mut().enumerate() {
                    *b = ((slot * 7 + i) % 251) as u8;
                }
            }
            {
                let v_dst = v.as_mut_slice::<u8>().expect("v mut_slice");
                for (i, b) in v_dst.iter_mut().enumerate() {
                    *b = ((slot * 11 + i) % 251) as u8;
                }
            }
            full_attn_k.push(k);
            full_attn_v.push(v);
            // current_len: per seq, deterministic.
            let cl: Vec<u32> = (0..cfg.n_seqs)
                .map(|s| (slot as u32) * 100 + s)
                .collect();
            full_attn_current_len.push(cl);
        }

        HybridKvCacheSnapshot {
            full_attn_k,
            full_attn_v,
            full_attn_current_len,
            mtp: None,
            linear_conv: Vec::new(),
            linear_recurrent: Vec::new(),
        }
    }

    fn synth_cfg(n_full_attn: u32, n_seqs: u32) -> Qwen35HybridConfig {
        // Tiny shape: [n_seqs, n_kv_heads=2, max_seq_len=8, head_dim=4]
        // → 2 * 8 * 4 = 64 elems/seq; 64 * n_seqs total per K (or V).
        Qwen35HybridConfig {
            n_full_attn,
            n_linear_attn: 0,
            has_mtp: false,
            n_seqs,
            full_attn_shape: [n_seqs as u64, 2, 8, 4],
            full_attn_codec: FullAttnCodec::F32Dense,
        }
    }

    fn snapshots_byte_equal(a: &HybridKvCacheSnapshot, b: &HybridKvCacheSnapshot) -> bool {
        if a.full_attn_k.len() != b.full_attn_k.len() {
            return false;
        }
        if a.full_attn_v.len() != b.full_attn_v.len() {
            return false;
        }
        if a.full_attn_current_len != b.full_attn_current_len {
            return false;
        }
        for i in 0..a.full_attn_k.len() {
            let ak = a.full_attn_k[i].as_slice::<u8>().expect("ak slice");
            let bk = b.full_attn_k[i].as_slice::<u8>().expect("bk slice");
            if ak != bk {
                return false;
            }
            let av = a.full_attn_v[i].as_slice::<u8>().expect("av slice");
            let bv = b.full_attn_v[i].as_slice::<u8>().expect("bv slice");
            if av != bv {
                return false;
            }
        }
        if a.mtp.is_some() != b.mtp.is_some() {
            return false;
        }
        if a.linear_conv.len() != b.linear_conv.len() {
            return false;
        }
        if a.linear_recurrent.len() != b.linear_recurrent.len() {
            return false;
        }
        true
    }

    #[test]
    fn qh35_round_trip_full_attn_only_byte_equal() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg(3, 1);
        let snap = synth_full_attn_only_snapshot(&device, &cfg);
        let bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize");
        // Envelope size sanity. Header: magic(4) + codec_version(4) +
        // n_full_attn(4) + n_linear_attn(4) + mtp_present(1) +
        // full_attn_codec_tag(1) + n_seqs(4) + reserved(2) = 24 bytes.
        // Per-slot overhead: slot_idx(4) + shape(32) + k_byte_len(8) +
        // v_byte_len(8) + current_len(4 * n_seqs=1) = 56. Per-slot body:
        // K(256) + V(256) = 512. Per-slot total = 568. n_full_attn=3.
        assert_eq!(bytes.len(), 24 + 3 * 568);
        let restored =
            deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
        assert!(snapshots_byte_equal(&snap, &restored));
    }

    #[test]
    fn qh35_round_trip_two_seqs_byte_equal() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg(2, 2);
        let snap = synth_full_attn_only_snapshot(&device, &cfg);
        let bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize");
        let restored =
            deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
        assert!(snapshots_byte_equal(&snap, &restored));
    }

    #[test]
    fn qh35_serialize_rejects_shape_mismatch() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg(1, 1);
        let snap = synth_full_attn_only_snapshot(&device, &cfg);
        // Pass a different cfg with a different shape: serialize should error.
        let mut bad_cfg = cfg.clone();
        bad_cfg.full_attn_shape = [1, 4, 8, 4]; // n_kv_heads=4 instead of 2
        let err = serialize_hybrid_snapshot(&snap, &bad_cfg).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("shape"),
            "expected shape-mismatch error, got: {msg}"
        );
    }

    #[test]
    fn qh35_deserialize_rejects_bad_magic() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg(1, 1);
        let mut bytes = serialize_hybrid_snapshot(
            &synth_full_attn_only_snapshot(&device, &cfg),
            &cfg,
        )
        .expect("serialize");
        bytes[0] = b'X';
        let err = deserialize_hybrid_snapshot(&bytes, &cfg, &device).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("bad magic"), "expected magic error, got: {msg}");
    }

    #[test]
    fn qh35_deserialize_rejects_codec_version_drift() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg(1, 1);
        let mut bytes = serialize_hybrid_snapshot(
            &synth_full_attn_only_snapshot(&device, &cfg),
            &cfg,
        )
        .expect("serialize");
        // codec_version is at offset 4..8 (LE u32). Bump to 99.
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = deserialize_hybrid_snapshot(&bytes, &cfg, &device).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("codec_version"),
            "expected codec_version error, got: {msg}"
        );
    }

    #[test]
    fn qh35_deserialize_rejects_iter3_scope_linear_attn() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        // Build a config that claims linear-attn slots but pass an empty
        // snapshot (which serialize will reject — so we hand-craft bytes
        // by using a config that says n_linear_attn=2 with no slots, then
        // patch the count after-the-fact for a deserialize-only test).
        let mut cfg = synth_cfg(0, 1);
        let snap = synth_full_attn_only_snapshot(&device, &cfg);
        let mut bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize");
        // n_linear_attn at offset 12..16 — bump to 2.
        bytes[12..16].copy_from_slice(&2u32.to_le_bytes());
        cfg.n_linear_attn = 2;
        let err = deserialize_hybrid_snapshot(&bytes, &cfg, &device).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("iter-3 scope"),
            "expected iter-3 scope error, got: {msg}"
        );
    }
}
