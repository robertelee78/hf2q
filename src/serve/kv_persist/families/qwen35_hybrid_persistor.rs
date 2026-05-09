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
    /// Per-linear-attn-slot conv-state shape `[conv_channels, K-1, n_seqs]`
    /// (matches `LinearAttnStateSlot::conv_state`'s allocation in
    /// kv_cache.rs's `HybridKvCache::new`). Iter-3 scope.
    pub linear_conv_shape: [u64; 3],
    /// Per-linear-attn-slot recurrent-state shape
    /// `[D_k, D_v, num_v_heads, n_seqs]`. Iter-3 scope.
    pub linear_recurrent_shape: [u64; 4],
    /// MTP slot shape `[n_seqs, n_kv_heads, max_seq_len, head_dim]` (iter-4).
    /// Same rank as `full_attn_shape` but allowed to differ — Qwen3.6
    /// MTP block has its own head-count config.  Ignored when has_mtp = false.
    pub mtp_shape: [u64; 4],
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
        ensure!(
            self.linear_conv_shape == other.linear_conv_shape,
            "QH35 config drift: linear_conv_shape = {:?} vs {:?}",
            self.linear_conv_shape,
            other.linear_conv_shape
        );
        ensure!(
            self.linear_recurrent_shape == other.linear_recurrent_shape,
            "QH35 config drift: linear_recurrent_shape = {:?} vs {:?}",
            self.linear_recurrent_shape,
            other.linear_recurrent_shape
        );
        ensure!(
            self.mtp_shape == other.mtp_shape,
            "QH35 config drift: mtp_shape = {:?} vs {:?}",
            self.mtp_shape,
            other.mtp_shape
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

    // --- Per linear-attn slot (iter-3) ---
    // Active conv_state + recurrent only; scratch is intentionally NOT
    // serialized per the existing HybridKvCache::snapshot semantics
    // (kv_cache.rs:624-659). See ADR-027 §4.4 for the Chesterton's-
    // fence rationale (no swap_parity field; "active" IS the canonical
    // state).
    let conv_elems: u64 = cfg.linear_conv_shape.iter().product();
    let recurrent_elems: u64 = cfg.linear_recurrent_shape.iter().product();
    let expected_conv_bytes = (conv_elems as usize) * std::mem::size_of::<f32>();
    let expected_recurrent_bytes = (recurrent_elems as usize) * std::mem::size_of::<f32>();
    for slot_idx in 0..cfg.n_linear_attn as usize {
        let conv = &snapshot.linear_conv[slot_idx];
        let rec = &snapshot.linear_recurrent[slot_idx];
        let conv_bytes: &[u8] = conv.as_slice::<u8>().map_err(|e| {
            anyhow!("QH35 serialize: linear_conv[{slot_idx}] as_slice: {e}")
        })?;
        let rec_bytes: &[u8] = rec.as_slice::<u8>().map_err(|e| {
            anyhow!("QH35 serialize: linear_recurrent[{slot_idx}] as_slice: {e}")
        })?;
        ensure!(
            conv_bytes.len() == expected_conv_bytes,
            "QH35 serialize: linear_conv[{slot_idx}].byte_len = {} != expected {}",
            conv_bytes.len(),
            expected_conv_bytes
        );
        ensure!(
            rec_bytes.len() == expected_recurrent_bytes,
            "QH35 serialize: linear_recurrent[{slot_idx}].byte_len = {} != expected {}",
            rec_bytes.len(),
            expected_recurrent_bytes
        );
        write_u32_le(&mut out, slot_idx as u32);
        write_u64_le(&mut out, conv_bytes.len() as u64);
        write_u64_le(&mut out, rec_bytes.len() as u64);
        out.extend_from_slice(conv_bytes);
        out.extend_from_slice(rec_bytes);
    }

    // --- MTP slot (iter-4) ---
    // MtpKvSnapshot has the same field shape as a single FullAttnKvSlot
    // snapshot (k, v, current_len). Layout mirrors per-full-attn-slot
    // exactly except the shape is from cfg.mtp_shape (Qwen3.6 MTP can
    // declare its own head count independent of regular full-attn).
    if cfg.has_mtp {
        let mtp = snapshot.mtp.as_ref().expect("mtp present per cfg + assert above");
        ensure!(
            mtp.k.shape().len() == 4,
            "QH35 serialize: mtp.k shape rank {} != 4",
            mtp.k.shape().len()
        );
        let mk_shape: [u64; 4] = [
            mtp.k.shape()[0] as u64,
            mtp.k.shape()[1] as u64,
            mtp.k.shape()[2] as u64,
            mtp.k.shape()[3] as u64,
        ];
        ensure!(
            mk_shape == cfg.mtp_shape,
            "QH35 serialize: mtp.k shape {:?} != cfg.mtp_shape {:?}",
            mk_shape,
            cfg.mtp_shape
        );
        let mv_shape: [u64; 4] = [
            mtp.v.shape()[0] as u64,
            mtp.v.shape()[1] as u64,
            mtp.v.shape()[2] as u64,
            mtp.v.shape()[3] as u64,
        ];
        ensure!(
            mv_shape == cfg.mtp_shape,
            "QH35 serialize: mtp.v shape {:?} != cfg.mtp_shape {:?}",
            mv_shape,
            cfg.mtp_shape
        );
        ensure!(
            mtp.current_len.len() == cfg.n_seqs as usize,
            "QH35 serialize: mtp.current_len.len() = {} != n_seqs = {}",
            mtp.current_len.len(),
            cfg.n_seqs
        );
        let mk_bytes: &[u8] = mtp
            .k
            .as_slice::<u8>()
            .map_err(|e| anyhow!("QH35 serialize: mtp.k as_slice: {e}"))?;
        let mv_bytes: &[u8] = mtp
            .v
            .as_slice::<u8>()
            .map_err(|e| anyhow!("QH35 serialize: mtp.v as_slice: {e}"))?;
        for &dim in &mk_shape {
            write_u64_le(&mut out, dim);
        }
        write_u64_le(&mut out, mk_bytes.len() as u64);
        write_u64_le(&mut out, mv_bytes.len() as u64);
        for &cl in mtp.current_len.iter() {
            write_u32_le(&mut out, cl);
        }
        out.extend_from_slice(mk_bytes);
        out.extend_from_slice(mv_bytes);
    }

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
        // Shapes are per-(slot, family) and not in the header — copy from
        // runtime cfg so the assert_matches comparison treats them as
        // equal (per-slot shape is validated against cfg in the body
        // loops below).
        full_attn_shape: cfg.full_attn_shape,
        linear_conv_shape: cfg.linear_conv_shape,
        linear_recurrent_shape: cfg.linear_recurrent_shape,
        mtp_shape: cfg.mtp_shape,
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

    // --- Per linear-attn slot (iter-3) ---
    let conv_shape_usize: Vec<usize> =
        cfg.linear_conv_shape.iter().map(|d| *d as usize).collect();
    let recurrent_shape_usize: Vec<usize> = cfg
        .linear_recurrent_shape
        .iter()
        .map(|d| *d as usize)
        .collect();
    let expected_conv_bytes = (cfg.linear_conv_shape.iter().product::<u64>() as usize)
        * std::mem::size_of::<f32>();
    let expected_recurrent_bytes =
        (cfg.linear_recurrent_shape.iter().product::<u64>() as usize)
            * std::mem::size_of::<f32>();
    let mut linear_conv: Vec<MlxBuffer> = Vec::with_capacity(cfg.n_linear_attn as usize);
    let mut linear_recurrent: Vec<MlxBuffer> =
        Vec::with_capacity(cfg.n_linear_attn as usize);
    for expected_slot in 0..cfg.n_linear_attn as usize {
        let slot_idx = read_u32_le(bytes, &mut cursor)? as usize;
        ensure!(
            slot_idx == expected_slot,
            "QH35 deserialize: linear_attn slot order mismatch — got {} expected {}",
            slot_idx,
            expected_slot
        );
        let conv_byte_len = read_u64_le(bytes, &mut cursor)? as usize;
        let rec_byte_len = read_u64_le(bytes, &mut cursor)? as usize;
        ensure!(
            conv_byte_len == expected_conv_bytes,
            "QH35 deserialize: linear_conv[{slot_idx}] on-disk byte_len = {} != \
             cfg-derived {}",
            conv_byte_len,
            expected_conv_bytes
        );
        ensure!(
            rec_byte_len == expected_recurrent_bytes,
            "QH35 deserialize: linear_recurrent[{slot_idx}] on-disk byte_len = {} \
             != cfg-derived {}",
            rec_byte_len,
            expected_recurrent_bytes
        );
        let conv_src = read_bytes(bytes, &mut cursor, conv_byte_len)?;
        let rec_src = read_bytes(bytes, &mut cursor, rec_byte_len)?;
        let mut conv_buf = device
            .alloc_buffer(conv_byte_len, DType::F32, conv_shape_usize.clone())
            .map_err(|e| {
                anyhow!("QH35 deserialize: alloc linear_conv[{slot_idx}]: {e}")
            })?;
        let mut rec_buf = device
            .alloc_buffer(rec_byte_len, DType::F32, recurrent_shape_usize.clone())
            .map_err(|e| {
                anyhow!("QH35 deserialize: alloc linear_recurrent[{slot_idx}]: {e}")
            })?;
        {
            let conv_dst = conv_buf.as_mut_slice::<u8>().map_err(|e| {
                anyhow!("QH35 deserialize: linear_conv[{slot_idx}] mut_slice: {e}")
            })?;
            conv_dst.copy_from_slice(conv_src);
        }
        {
            let rec_dst = rec_buf.as_mut_slice::<u8>().map_err(|e| {
                anyhow!("QH35 deserialize: linear_recurrent[{slot_idx}] mut_slice: {e}")
            })?;
            rec_dst.copy_from_slice(rec_src);
        }
        linear_conv.push(conv_buf);
        linear_recurrent.push(rec_buf);
    }

    // --- MTP slot (iter-4) ---
    let mtp: Option<MtpKvSnapshot> = if cfg.has_mtp {
        let mut mk_shape_arr = [0u64; 4];
        for dim in &mut mk_shape_arr {
            *dim = read_u64_le(bytes, &mut cursor)?;
        }
        ensure!(
            mk_shape_arr == cfg.mtp_shape,
            "QH35 deserialize: mtp shape on disk {:?} != cfg.mtp_shape {:?}",
            mk_shape_arr,
            cfg.mtp_shape
        );
        let mk_byte_len = read_u64_le(bytes, &mut cursor)? as usize;
        let mv_byte_len = read_u64_le(bytes, &mut cursor)? as usize;
        let mut mtp_current_len = Vec::with_capacity(cfg.n_seqs as usize);
        for _ in 0..cfg.n_seqs {
            mtp_current_len.push(read_u32_le(bytes, &mut cursor)?);
        }
        let mk_src = read_bytes(bytes, &mut cursor, mk_byte_len)?;
        let mv_src = read_bytes(bytes, &mut cursor, mv_byte_len)?;
        let mtp_shape_usize: Vec<usize> =
            mk_shape_arr.iter().map(|d| *d as usize).collect();
        let dtype = full_attn_dtype_for_codec(header_cfg.full_attn_codec);
        let mut mk_buf = device
            .alloc_buffer(mk_byte_len, dtype, mtp_shape_usize.clone())
            .map_err(|e| anyhow!("QH35 deserialize: alloc mtp.k: {e}"))?;
        let mut mv_buf = device
            .alloc_buffer(mv_byte_len, dtype, mtp_shape_usize)
            .map_err(|e| anyhow!("QH35 deserialize: alloc mtp.v: {e}"))?;
        {
            let dst = mk_buf
                .as_mut_slice::<u8>()
                .map_err(|e| anyhow!("QH35 deserialize: mtp.k mut_slice: {e}"))?;
            ensure!(
                dst.len() == mk_src.len(),
                "QH35 deserialize: mtp.k dst.len() = {} != src.len() = {}",
                dst.len(),
                mk_src.len()
            );
            dst.copy_from_slice(mk_src);
        }
        {
            let dst = mv_buf
                .as_mut_slice::<u8>()
                .map_err(|e| anyhow!("QH35 deserialize: mtp.v mut_slice: {e}"))?;
            ensure!(
                dst.len() == mv_src.len(),
                "QH35 deserialize: mtp.v dst.len() = {} != src.len() = {}",
                dst.len(),
                mv_src.len()
            );
            dst.copy_from_slice(mv_src);
        }
        Some(MtpKvSnapshot {
            k: mk_buf,
            v: mv_buf,
            current_len: mtp_current_len,
        })
    } else {
        None
    };

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
        synth_cfg_with_linear(n_full_attn, 0, n_seqs)
    }

    /// Same as `synth_cfg` but allows `n_linear_attn > 0` for iter-3
    /// round-trip tests. Linear conv shape `[conv_channels=4, K-1=3, n_seqs]`
    /// (DELTA_NET_CONV_K = 4 → K-1 = 3); recurrent `[D_k=4, D_v=8, num_v_heads=2, n_seqs]`.
    fn synth_cfg_with_linear(
        n_full_attn: u32,
        n_linear_attn: u32,
        n_seqs: u32,
    ) -> Qwen35HybridConfig {
        synth_cfg_full(n_full_attn, n_linear_attn, false, n_seqs)
    }

    /// Iter-4: same as synth_cfg_with_linear but allows toggling MTP.
    fn synth_cfg_full(
        n_full_attn: u32,
        n_linear_attn: u32,
        has_mtp: bool,
        n_seqs: u32,
    ) -> Qwen35HybridConfig {
        Qwen35HybridConfig {
            n_full_attn,
            n_linear_attn,
            has_mtp,
            n_seqs,
            full_attn_shape: [n_seqs as u64, 2, 8, 4],
            full_attn_codec: FullAttnCodec::F32Dense,
            linear_conv_shape: [4, 3, n_seqs as u64],
            linear_recurrent_shape: [4, 8, 2, n_seqs as u64],
            // MTP at a slightly different head_dim so the per-cfg shape
            // path is exercised — Qwen3.6 MTP block does declare its own
            // head_count in production.
            mtp_shape: [n_seqs as u64, 4, 8, 4],
        }
    }

    /// Build a snapshot containing both full-attn AND linear-attn slots
    /// with deterministic byte patterns for round-trip verification.
    fn synth_full_plus_linear_snapshot(
        device: &MlxDevice,
        cfg: &Qwen35HybridConfig,
    ) -> HybridKvCacheSnapshot {
        let mut snap = synth_full_attn_only_snapshot(device, cfg);
        let conv_elems: usize = cfg.linear_conv_shape.iter().product::<u64>() as usize;
        let conv_bytes_len = conv_elems * std::mem::size_of::<f32>();
        let conv_shape_usize: Vec<usize> =
            cfg.linear_conv_shape.iter().map(|d| *d as usize).collect();
        let rec_elems: usize =
            cfg.linear_recurrent_shape.iter().product::<u64>() as usize;
        let rec_bytes_len = rec_elems * std::mem::size_of::<f32>();
        let rec_shape_usize: Vec<usize> = cfg
            .linear_recurrent_shape
            .iter()
            .map(|d| *d as usize)
            .collect();
        for slot in 0..cfg.n_linear_attn as usize {
            let mut conv = device
                .alloc_buffer(conv_bytes_len, DType::F32, conv_shape_usize.clone())
                .expect("alloc conv");
            {
                let dst = conv.as_mut_slice::<u8>().expect("conv mut_slice");
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((slot * 13 + i) % 251) as u8;
                }
            }
            snap.linear_conv.push(conv);

            let mut rec = device
                .alloc_buffer(rec_bytes_len, DType::F32, rec_shape_usize.clone())
                .expect("alloc rec");
            {
                let dst = rec.as_mut_slice::<u8>().expect("rec mut_slice");
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((slot * 17 + i) % 251) as u8;
                }
            }
            snap.linear_recurrent.push(rec);
        }
        snap
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
        for i in 0..a.linear_conv.len() {
            let ac = a.linear_conv[i].as_slice::<u8>().expect("ac slice");
            let bc = b.linear_conv[i].as_slice::<u8>().expect("bc slice");
            if ac != bc {
                return false;
            }
            let ar = a.linear_recurrent[i].as_slice::<u8>().expect("ar slice");
            let br = b.linear_recurrent[i].as_slice::<u8>().expect("br slice");
            if ar != br {
                return false;
            }
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
    fn qh35_round_trip_with_linear_attn_byte_equal() {
        // ADR-027 Phase A iter-3: linear-attn slot round-trip.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg_with_linear(2, 3, 1);
        let snap = synth_full_plus_linear_snapshot(&device, &cfg);
        let bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize");
        let restored =
            deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
        assert!(snapshots_byte_equal(&snap, &restored));
        // Per-linear-slot overhead = slot_idx(4) + conv_len(8) + rec_len(8) = 20.
        // Per-linear-slot body = conv(4*3*1*4) + rec(4*8*2*1*4) = 48 + 256 = 304.
        // Per-slot total = 324. With 3 linear slots = 972 bytes.
        // Plus header(24) + 2 full-attn slots @ 568 each = 24 + 1136 = 1160.
        // Total = 1160 + 972 = 2132 bytes.
        assert_eq!(bytes.len(), 24 + 2 * 568 + 3 * 324);
    }

    /// Iter-4: extends a snapshot with an MTP slot containing
    /// deterministic byte patterns at a possibly-different shape than
    /// the regular full-attn slots (per cfg.mtp_shape).
    fn synth_full_plus_linear_plus_mtp_snapshot(
        device: &MlxDevice,
        cfg: &Qwen35HybridConfig,
    ) -> HybridKvCacheSnapshot {
        let mut snap = synth_full_plus_linear_snapshot(device, cfg);
        if cfg.has_mtp {
            let elems: usize = cfg.mtp_shape.iter().product::<u64>() as usize;
            let bytes_len = elems * std::mem::size_of::<f32>();
            let shape_usize: Vec<usize> =
                cfg.mtp_shape.iter().map(|d| *d as usize).collect();
            let mut k = device
                .alloc_buffer(bytes_len, DType::F32, shape_usize.clone())
                .expect("alloc mtp.k");
            {
                let dst = k.as_mut_slice::<u8>().expect("mtp.k mut_slice");
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((19 * i + 5) % 251) as u8;
                }
            }
            let mut v = device
                .alloc_buffer(bytes_len, DType::F32, shape_usize)
                .expect("alloc mtp.v");
            {
                let dst = v.as_mut_slice::<u8>().expect("mtp.v mut_slice");
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((23 * i + 7) % 251) as u8;
                }
            }
            let current_len: Vec<u32> = (0..cfg.n_seqs).map(|s| 99 + s).collect();
            snap.mtp = Some(MtpKvSnapshot {
                k,
                v,
                current_len,
            });
        }
        snap
    }

    #[test]
    fn qh35_round_trip_with_mtp_byte_equal() {
        // ADR-027 Phase A iter-4: full + linear + MTP round-trip.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg_full(2, 3, true, 1);
        let snap = synth_full_plus_linear_plus_mtp_snapshot(&device, &cfg);
        let bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize");
        let restored =
            deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
        // Compare full + linear + MTP byte-equal.
        assert!(snapshots_byte_equal(&snap, &restored));
        assert!(restored.mtp.is_some());
        let r_mtp = restored.mtp.as_ref().unwrap();
        let s_mtp = snap.mtp.as_ref().unwrap();
        let r_k = r_mtp.k.as_slice::<u8>().expect("rk slice");
        let s_k = s_mtp.k.as_slice::<u8>().expect("sk slice");
        assert_eq!(r_k, s_k);
        let r_v = r_mtp.v.as_slice::<u8>().expect("rv slice");
        let s_v = s_mtp.v.as_slice::<u8>().expect("sv slice");
        assert_eq!(r_v, s_v);
        assert_eq!(r_mtp.current_len, s_mtp.current_len);
    }

    #[test]
    fn qh35_round_trip_mtp_only_no_linear_byte_equal() {
        // Edge case: MTP present but no linear-attn slots.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg_full(1, 0, true, 1);
        let snap = synth_full_plus_linear_plus_mtp_snapshot(&device, &cfg);
        let bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize");
        let restored =
            deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
        assert!(snapshots_byte_equal(&snap, &restored));
    }

    #[test]
    fn qh35_serialize_rejects_linear_conv_shape_mismatch() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg_with_linear(0, 2, 1);
        let snap = synth_full_plus_linear_snapshot(&device, &cfg);
        // Claim a different conv_channels count than the snapshot's
        // buffers actually carry — serialize should error.
        let mut bad_cfg = cfg.clone();
        bad_cfg.linear_conv_shape = [8, 3, 1]; // conv_channels=8 instead of 4
        let err = serialize_hybrid_snapshot(&snap, &bad_cfg).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("linear_conv") && msg.contains("byte_len"),
            "expected linear_conv byte_len error, got: {msg}"
        );
    }
}
