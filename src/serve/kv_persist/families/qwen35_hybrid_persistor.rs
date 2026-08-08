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
    HybridKvCache, HybridKvCacheSnapshot, MtpKvSnapshot, TqKvSnapshot,
};
use crate::serve::kv_persist::format::ModelFingerprint;

/// Magic bytes prefixing every QH35 (Qwen3.5 Hybrid) envelope. ASCII for
/// "QH35" to make hex dumps trivial to spot-read.
pub const QH35_MAGIC: [u8; 4] = *b"QH35";

/// Current codec version.
/// - v1 (iter-2): always-Some K/V per slot.
/// - v2 (iter-23a-γ): per-slot `kv_present: u8` byte before shape so
///   None entries can round-trip without K/V payload (the F32-drop
///   precondition for iter-23c+). v2 deserializer also accepts v1
///   envelopes via fallback (every slot treated as present).
/// - v3 (iter-36 = sub-iter 23d-β): per-slot `tq_present: u8`
///   byte after the v2 body, with optional TQ payload (`norms_per_pos`
///   + 4 byte_len-prefixed buffer blobs). Closes the iter-34 cross-
///   process replay gap: in TQ-only mode (slot.k=None) the codec
///   round-trips slot.tq state so cold-start hydrate restores both
///   the F32 absence AND the TQ presence. v3 deserializer accepts v1
///   AND v2 envelopes via fallback (every slot treated as tq_absent).
/// - v4 (sub-iter 23d-γ, 2026-08-03): per-MTP `kv_present: u8` byte
///   before the MTP shape block, mirroring the full-attn slots' v2
///   byte. v3 and earlier always emitted MTP K/V (Some); in TQ-only
///   mode the MTP slot also drops its F32 backing (`mtp_slot.k=None`)
///   and the v3 serializer hard-panicked on the `expect` — same gap
///   class as the full-attn slots had pre-v2. v4 deserializer accepts
///   v1..v3 envelopes via fallback (MTP treated as kv_present).
/// - v5: full-attention and MTP payloads may have a shorter sequence axis
///   than the live request cache. TQ blobs carry their four-dimensional
///   packed shape explicitly. Readers still accept v1..v4 full-capacity
///   snapshots. This permits compact LCP checkpoints without changing the
///   live-cache fingerprint or exact prompt-cache snapshot semantics.
pub const QH35_CODEC_VERSION: u32 = 5;

/// Per-slot `kv_present` byte values (v2 / v3).
pub const QH35_KV_PRESENT: u8 = 1;
pub const QH35_KV_ABSENT: u8 = 0;

/// Per-slot `tq_present` byte values (v3 only). `KV_PRESENT`/`ABSENT`
/// constants intentionally NOT shared — `kv` and `tq` are orthogonal
/// presence flags (a slot may have only F32, only TQ, both, or neither).
pub const QH35_TQ_PRESENT: u8 = 1;
pub const QH35_TQ_ABSENT: u8 = 0;

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

/// ADR-027 sub-iter 23d-γ (2026-08-03) — which KV substrate the source
/// cache's full-attn-family slots actually carry. Derived by
/// `cfg_from_cache` from the LIVE cache and mixed into the disk
/// fingerprint (`compute_config_fingerprint_hex`) so a snapshot written
/// under one substrate NEVER hydrates into a cache allocated under a
/// different substrate. Cross-substrate restore is incorrect by
/// construction: the (snapshot-substrate, cache-substrate) pairs that
/// restore nothing would leave the resumed prefix state zeroed (silent
/// coherence corruption) — namespacing by substrate turns those pairs
/// into clean cache misses instead.
///
/// NOT serialized into the QH35 envelope header (the v2/v3/v4
/// `kv_present` / `tq_present` bytes make every payload
/// self-describing); it exists purely to namespace the on-disk
/// cfg-fingerprint directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum KvSubstrate {
    /// Legacy F32-only: `slot.k`/`v` Some, `slot.tq` None
    /// (`HF2Q_TQ_KV=0`).
    F32Only = 0,
    /// Production TQ-only: `slot.k`/`v` None, `slot.tq` Some
    /// (default, `tq_kv_active=true`, iter-34 F32-drop).
    TqOnly = 1,
    /// Shadow mode: both F32 and TQ populated (iter-8 bridge; not a
    /// production runtime config).
    Both = 2,
}

impl KvSubstrate {
    /// Classify one full-attn-family slot's substrate from its
    /// `k` / `tq` presence. `None` when the slot has NEITHER substrate
    /// (degenerate — surfaces as a hard error in `cfg_from_cache`).
    fn classify(k_present: bool, tq_present: bool) -> Option<Self> {
        match (k_present, tq_present) {
            (true, false) => Some(Self::F32Only),
            (false, true) => Some(Self::TqOnly),
            (true, true) => Some(Self::Both),
            (false, false) => None,
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
    /// KV substrate of the source cache's full-attn-family slots
    /// (ADR-027 sub-iter 23d-γ). Fingerprint-only — see [`KvSubstrate`].
    pub kv_substrate: KvSubstrate,
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

fn ensure_prefix_shape_compatible(actual: [u64; 4], live: [u64; 4], what: &str) -> Result<()> {
    ensure!(
        actual[0] == live[0] && actual[1] == live[1] && actual[3] == live[3],
        "QH35 {what}: non-sequence shape {:?} incompatible with live cache {:?}",
        actual,
        live
    );
    ensure!(
        actual[2] > 0 && actual[2] <= live[2],
        "QH35 {what}: sequence capacity {} outside 1..={} (live cache)",
        actual[2],
        live[2]
    );
    Ok(())
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

/// Derive a `Qwen35HybridConfig` from a live `HybridKvCache`. The cache
/// is the runtime authority on shape (`HybridKvCache::new(cfg, dev,
/// max_seq_len, n_seqs)` at kv_cache.rs:347+ allocates per-prefill
/// from the input config + runtime params); reading the actually-
/// allocated buffers gives the QH35 envelope writer the exact dims to
/// stamp.
///
/// `codec` is the operator's choice of full-attn codec for THIS
/// snapshot — iter-6b.2 always passes `FullAttnCodec::F32Dense`;
/// Phase B iter-11 introduces `TqV2`.
///
/// Errors: empty full_attn AND empty mtp (degenerate cache shape;
/// indicates the model has no full-attention layers + no MTP, which
/// cannot happen for any in-tree qwen35 / qwen35moe variant).
pub fn cfg_from_cache(cache: &HybridKvCache, codec: FullAttnCodec) -> Result<Qwen35HybridConfig> {
    let n_full_attn = cache.full_attn.len() as u32;
    let n_linear_attn = cache.linear_attn.len() as u32;
    let has_mtp = cache.mtp_slot.is_some();
    let n_seqs = cache.n_seqs;

    // ADR-027 sub-iter 23d-γ (2026-08-03): derive one full-attn-family
    // slot's logical `[n_seqs, n_kv_heads, max_seq_len, head_dim]` shape
    // from WHICHEVER substrate is populated — F32 `k` when present
    // (legacy / HF2Q_TQ_KV=0), else the TQ `k_packed` (production
    // TQ-only mode, iter-34 F32-drop). TQ packed buffers carry the SAME
    // four logical dims as F32 (U8 dtype, one byte per element — see
    // `alloc_tq_full_attn_buffers`), so the derived shape is identical
    // across substrates for the same model + max_seq_len. This closes
    // the iter-23d TODO that previously hard-panicked in TQ-only mode.
    let shape4_from_slot = |slot: &crate::inference::models::qwen35::kv_cache::FullAttnKvSlot,
                            what: &str|
     -> Result<[u64; 4]> {
        let (buf, src): (&MlxBuffer, &str) = if let Some(k) = slot.k.as_ref() {
            (k, "k")
        } else if let Some(tq) = slot.tq.as_ref() {
            (&tq.k_packed, "tq.k_packed")
        } else {
            return Err(anyhow!(
                "cfg_from_cache: {what} has neither F32 k nor TQ k_packed \
                 (degenerate cache — neither substrate populated)"
            ));
        };
        let s = buf.shape();
        ensure!(
            s.len() == 4,
            "cfg_from_cache: {what}.{src} shape rank {} != 4",
            s.len()
        );
        let shape = [s[0] as u64, s[1] as u64, s[2] as u64, s[3] as u64];
        ensure!(
            shape[0] == n_seqs as u64,
            "cfg_from_cache: {what}.{src} shape[0] {} != n_seqs {}",
            shape[0],
            n_seqs
        );
        Ok(shape)
    };

    // full_attn_shape: prefer a real full_attn slot; fall back to mtp
    // (same rank/role) when full_attn is empty.
    let full_attn_shape: [u64; 4] = if let Some(slot) = cache.full_attn.first() {
        shape4_from_slot(slot, "full_attn[0]")?
    } else if let Some(slot) = cache.mtp_slot.as_ref() {
        shape4_from_slot(slot, "mtp_slot")?
    } else {
        return Err(anyhow!(
            "cfg_from_cache: cache has no full_attn slots AND no mtp_slot \
             (impossible for any in-tree qwen35 model)"
        ));
    };

    // ADR-027 sub-iter 23d-γ: classify the cache's KV substrate and
    // verify it is UNIFORM across every full-attn-family slot (the
    // `new_with_options` allocator invariant — a mixed-substrate cache
    // indicates an alloc-path bug that must not silently fingerprint).
    let mut kv_substrate: Option<KvSubstrate> = None;
    for (i, slot) in cache
        .full_attn
        .iter()
        .chain(cache.mtp_slot.as_ref())
        .enumerate()
    {
        let s = KvSubstrate::classify(slot.k.is_some(), slot.tq.is_some()).ok_or_else(|| {
            anyhow!(
                "cfg_from_cache: full-attn-family slot #{i} has neither F32 k nor TQ \
                 k_packed (degenerate cache — neither substrate populated)"
            )
        })?;
        match kv_substrate {
            None => kv_substrate = Some(s),
            Some(prev) => ensure!(
                prev == s,
                "cfg_from_cache: mixed KV substrates across full-attn-family slots \
                 (slot #0 = {prev:?}, slot #{i} = {s:?}) — allocator invariant violated"
            ),
        }
    }
    let kv_substrate = kv_substrate.ok_or_else(|| {
        anyhow!("cfg_from_cache: no full-attn-family slots to classify (unreachable)")
    })?;

    // MTP shape: prefer mtp_slot (either substrate, same derivation as
    // full-attn); otherwise use full_attn_shape (same rank/role; ignored
    // at write/read time when has_mtp = false).
    let mtp_shape: [u64; 4] = if let Some(slot) = cache.mtp_slot.as_ref() {
        shape4_from_slot(slot, "mtp_slot")?
    } else {
        full_attn_shape
    };

    // linear_conv_shape / linear_recurrent_shape: prefer first real
    // slot; fall back to small-but-valid sentinel shapes when no
    // linear-attn layers exist (the field is ignored at serialize
    // time when n_linear_attn == 0; sentinel keeps assert_matches
    // stable across runs).
    let (linear_conv_shape, linear_recurrent_shape) = if let Some(slot) = cache.linear_attn.first()
    {
        let cs = slot.conv_state.shape();
        ensure!(
            cs.len() == 3,
            "cfg_from_cache: linear_attn[0].conv_state shape rank {} != 3",
            cs.len()
        );
        let rs = slot.recurrent.shape();
        ensure!(
            rs.len() == 4,
            "cfg_from_cache: linear_attn[0].recurrent shape rank {} != 4",
            rs.len()
        );
        (
            [cs[0] as u64, cs[1] as u64, cs[2] as u64],
            [rs[0] as u64, rs[1] as u64, rs[2] as u64, rs[3] as u64],
        )
    } else {
        // Sentinel: the values are unused when n_linear_attn = 0; the
        // assert_matches comparison still requires equality between
        // serialize-time and deserialize-time cfgs, so the sentinel
        // must be deterministic. All-zeros chosen to be obviously
        // sentinel-shaped in any debug log.
        ([0, 0, 0], [0, 0, 0, 0])
    };

    Ok(Qwen35HybridConfig {
        n_full_attn,
        n_linear_attn,
        has_mtp,
        n_seqs,
        full_attn_shape,
        full_attn_codec: codec,
        linear_conv_shape,
        linear_recurrent_shape,
        mtp_shape,
        kv_substrate,
    })
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
    //
    // ADR-027 sub-sub-iter 23a-γ (this iter, codec v2): each slot is
    // prefixed with a `kv_present: u8` byte. When 0 (TQ-only mode,
    // iter-23c+), the slot writes ONLY slot_idx + kv_present + current_len
    // (no shape, no byte_lens, no K/V payload). When 1 (F32 path), the
    // slot writes the full v1-equivalent payload after the kv_present
    // byte. Both K and V must have the same kv_present state per slot.
    for slot_idx in 0..cfg.n_full_attn as usize {
        let k_opt = snapshot.full_attn_k[slot_idx].as_ref();
        let v_opt = snapshot.full_attn_v[slot_idx].as_ref();
        let current_len = &snapshot.full_attn_current_len[slot_idx];

        // K and V must agree on Some/None (mismatched is a producer bug).
        ensure!(
            k_opt.is_some() == v_opt.is_some(),
            "QH35 serialize: full_attn[{slot_idx}] K Some={} but V Some={} \
             (mismatch — producer bug)",
            k_opt.is_some(),
            v_opt.is_some()
        );
        ensure!(
            current_len.len() == cfg.n_seqs as usize,
            "QH35 serialize: full_attn[{slot_idx}].current_len.len() = {} != n_seqs = {}",
            current_len.len(),
            cfg.n_seqs
        );

        write_u32_le(&mut out, slot_idx as u32);

        match (k_opt, v_opt) {
            (Some(k), Some(v)) => {
                // Per-slot validation against config (same as v1).
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
                ensure_prefix_shape_compatible(
                    k_shape,
                    cfg.full_attn_shape,
                    &format!("serialize full_attn[{slot_idx}].k"),
                )?;
                let v_shape: [u64; 4] = [
                    v.shape()[0] as u64,
                    v.shape()[1] as u64,
                    v.shape()[2] as u64,
                    v.shape()[3] as u64,
                ];
                ensure!(
                    v_shape == k_shape,
                    "QH35 serialize: full_attn[{slot_idx}].v shape {:?} != k shape {:?}",
                    v_shape,
                    k_shape
                );

                let k_bytes: &[u8] = k.as_slice::<u8>().map_err(|e| {
                    anyhow!("QH35 serialize: full_attn[{slot_idx}].k as_slice: {e}")
                })?;
                let v_bytes: &[u8] = v.as_slice::<u8>().map_err(|e| {
                    anyhow!("QH35 serialize: full_attn[{slot_idx}].v as_slice: {e}")
                })?;
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

                write_u8(&mut out, QH35_KV_PRESENT);
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
            (None, None) => {
                // TQ-only mode (iter-23c+): no K/V payload to serialize.
                // Just the kv_present=0 byte + current_len for snapshot
                // bookkeeping.
                write_u8(&mut out, QH35_KV_ABSENT);
                for &cl in current_len.iter() {
                    write_u32_le(&mut out, cl);
                }
            }
            _ => unreachable!("k_opt.is_some() == v_opt.is_some() asserted above"),
        }

        // ADR-027 Phase B iter-36 (sub-iter 23d-β): per-slot tq_present
        // byte + optional TQ payload. v3 only — earlier versions
        // implicitly emit no TQ.
        let tq_opt = snapshot.full_attn_tq.get(slot_idx).and_then(|t| t.as_ref());
        match tq_opt {
            Some(tq) => {
                write_u8(&mut out, QH35_TQ_PRESENT);
                serialize_tq_blob(&mut out, tq, slot_idx, "full_attn", cfg.full_attn_shape)?;
            }
            None => {
                write_u8(&mut out, QH35_TQ_ABSENT);
            }
        }
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
        let conv_bytes: &[u8] = conv
            .as_slice::<u8>()
            .map_err(|e| anyhow!("QH35 serialize: linear_conv[{slot_idx}] as_slice: {e}"))?;
        let rec_bytes: &[u8] = rec
            .as_slice::<u8>()
            .map_err(|e| anyhow!("QH35 serialize: linear_recurrent[{slot_idx}] as_slice: {e}"))?;
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
        let mtp = snapshot
            .mtp
            .as_ref()
            .expect("mtp present per cfg + assert above");
        ensure!(
            mtp.current_len.len() == cfg.n_seqs as usize,
            "QH35 serialize: mtp.current_len.len() = {} != n_seqs = {}",
            mtp.current_len.len(),
            cfg.n_seqs
        );
        // ADR-027 sub-iter 23d-γ (codec v4): per-MTP `kv_present: u8`
        // byte mirroring the full-attn slots' v2 byte. In TQ-only mode
        // the MTP slot drops its F32 backing exactly like the full-attn
        // slots (`alloc_full_attn_slot` with tq_kv_active=true); the
        // pre-v4 serializer's `expect("mtp.k is None")` hard-panicked —
        // the same gap class the v2 byte closed for full-attn slots.
        // K and V must agree on Some/None (mismatched = producer bug).
        ensure!(
            mtp.k.is_some() == mtp.v.is_some(),
            "QH35 serialize: mtp.k Some={} but mtp.v Some={} (mismatch — producer bug)",
            mtp.k.is_some(),
            mtp.v.is_some()
        );
        match (mtp.k.as_ref(), mtp.v.as_ref()) {
            (Some(mtp_k), Some(mtp_v)) => {
                ensure!(
                    mtp_k.shape().len() == 4,
                    "QH35 serialize: mtp.k shape rank {} != 4",
                    mtp_k.shape().len()
                );
                let mk_shape: [u64; 4] = [
                    mtp_k.shape()[0] as u64,
                    mtp_k.shape()[1] as u64,
                    mtp_k.shape()[2] as u64,
                    mtp_k.shape()[3] as u64,
                ];
                ensure_prefix_shape_compatible(mk_shape, cfg.mtp_shape, "serialize mtp.k")?;
                let mv_shape: [u64; 4] = [
                    mtp_v.shape()[0] as u64,
                    mtp_v.shape()[1] as u64,
                    mtp_v.shape()[2] as u64,
                    mtp_v.shape()[3] as u64,
                ];
                ensure!(
                    mv_shape == mk_shape,
                    "QH35 serialize: mtp.v shape {:?} != mtp.k shape {:?}",
                    mv_shape,
                    mk_shape
                );
                let mk_bytes: &[u8] = mtp_k
                    .as_slice::<u8>()
                    .map_err(|e| anyhow!("QH35 serialize: mtp.k as_slice: {e}"))?;
                let mv_bytes: &[u8] = mtp_v
                    .as_slice::<u8>()
                    .map_err(|e| anyhow!("QH35 serialize: mtp.v as_slice: {e}"))?;
                write_u8(&mut out, QH35_KV_PRESENT);
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
            (None, None) => {
                // TQ-only mode: kv_present=0 + current_len only (the
                // MTP TQ payload rides the tq_present block below).
                write_u8(&mut out, QH35_KV_ABSENT);
                for &cl in mtp.current_len.iter() {
                    write_u32_le(&mut out, cl);
                }
            }
            _ => unreachable!("mtp.k.is_some() == mtp.v.is_some() asserted above"),
        }

        // ADR-027 Phase B iter-36 (sub-iter 23d-β): MTP TQ payload — same
        // format as full-attn slot TQ (see `serialize_tq_blob`).
        let mtp_tq_opt = mtp.tq.as_ref();
        match mtp_tq_opt {
            Some(tq) => {
                write_u8(&mut out, QH35_TQ_PRESENT);
                serialize_tq_blob(&mut out, tq, 0, "mtp", cfg.mtp_shape)?;
            }
            None => {
                write_u8(&mut out, QH35_TQ_ABSENT);
            }
        }
    }

    Ok(out)
}

/// ADR-027 Phase B iter-36 (sub-iter 23d-β): serialize one TQ payload
/// blob. Layout:
///
/// ```text
/// [norms_per_pos: u32 LE]
/// [k_packed_byte_len: u64 LE] [k_packed_bytes...]
/// [k_norms_byte_len:  u64 LE] [k_norms_bytes...]
/// [v_packed_byte_len: u64 LE] [v_packed_bytes...]
/// [v_norms_byte_len:  u64 LE] [v_norms_bytes...]
/// ```
///
/// Caller emits the per-slot `tq_present: u8` byte BEFORE invoking
/// this function (so v3 readers can skip the entire blob on
/// `tq_present == 0`).
fn serialize_tq_blob(
    out: &mut Vec<u8>,
    tq: &TqKvSnapshot,
    slot_idx: usize,
    family: &str,
    live_shape: [u64; 4],
) -> Result<()> {
    let packed_shape = |buf: &MlxBuffer, name: &str| -> Result<[u64; 4]> {
        ensure!(
            buf.shape().len() == 4,
            "QH35 serialize: {family}[{slot_idx}].tq.{name} rank {} != 4",
            buf.shape().len()
        );
        Ok([
            buf.shape()[0] as u64,
            buf.shape()[1] as u64,
            buf.shape()[2] as u64,
            buf.shape()[3] as u64,
        ])
    };
    let shape = packed_shape(&tq.k_packed, "k_packed")?;
    ensure_prefix_shape_compatible(
        shape,
        live_shape,
        &format!("serialize {family}[{slot_idx}].tq.k_packed"),
    )?;
    ensure!(
        packed_shape(&tq.v_packed, "v_packed")? == shape,
        "QH35 serialize: {family}[{slot_idx}].tq.v_packed shape differs from k_packed"
    );
    let norms_shape = [shape[0], shape[1], shape[2], tq.norms_per_pos as u64];
    ensure!(
        packed_shape(&tq.k_norms, "k_norms")? == norms_shape,
        "QH35 serialize: {family}[{slot_idx}].tq.k_norms shape mismatch"
    );
    ensure!(
        packed_shape(&tq.v_norms, "v_norms")? == norms_shape,
        "QH35 serialize: {family}[{slot_idx}].tq.v_norms shape mismatch"
    );
    let kp = tq
        .k_packed
        .as_slice::<u8>()
        .map_err(|e| anyhow!("QH35 serialize: {family}[{slot_idx}].tq.k_packed as_slice: {e}"))?;
    let kn = tq
        .k_norms
        .as_slice::<u8>()
        .map_err(|e| anyhow!("QH35 serialize: {family}[{slot_idx}].tq.k_norms as_slice: {e}"))?;
    let vp = tq
        .v_packed
        .as_slice::<u8>()
        .map_err(|e| anyhow!("QH35 serialize: {family}[{slot_idx}].tq.v_packed as_slice: {e}"))?;
    let vn = tq
        .v_norms
        .as_slice::<u8>()
        .map_err(|e| anyhow!("QH35 serialize: {family}[{slot_idx}].tq.v_norms as_slice: {e}"))?;

    for dim in shape {
        write_u64_le(out, dim);
    }
    write_u32_le(out, tq.norms_per_pos);
    write_u64_le(out, kp.len() as u64);
    out.extend_from_slice(kp);
    write_u64_le(out, kn.len() as u64);
    out.extend_from_slice(kn);
    write_u64_le(out, vp.len() as u64);
    out.extend_from_slice(vp);
    write_u64_le(out, vn.len() as u64);
    out.extend_from_slice(vn);
    Ok(())
}

/// ADR-027 Phase B iter-36 (sub-iter 23d-β): deserialize one TQ payload
/// blob. Mirror of [`serialize_tq_blob`]. Allocates 4 fresh `MlxBuffer`s
/// via `device` and copies the bytes verbatim.
///
/// Caller has already consumed the `tq_present: u8` byte and confirmed
/// it equals `QH35_TQ_PRESENT`.
///
/// ADR-027 sub-iter 23d-γ: buffers are allocated with their LOGICAL
/// 4-rank shapes (derived from `packed_shape4` = the cfg's
/// `[n_seqs, n_kv_heads, max_seq_len, head_dim]`), NOT as flat rank-1
/// blobs. The pre-23d-γ flat allocation surfaced as a hard
/// `partial_copy_slot` rank error the first time a disk-hydrated
/// snapshot was LCP-resumed (`HybridKvCache::restore_partial` requires
/// rank-4 on both sides for the per-head prefix copy). Every blob's
/// byte_len is validated against the cfg-derived shape product, so a
/// shape drift between writer and reader still fails fast HERE rather
/// than at restore time.
fn deserialize_tq_blob(
    bytes: &[u8],
    cursor: &mut usize,
    device: &MlxDevice,
    slot_idx: usize,
    family: &str,
    live_shape4: [u64; 4],
    codec_v5: bool,
) -> Result<TqKvSnapshot> {
    let packed_shape4 = if codec_v5 {
        let mut shape = [0u64; 4];
        for dim in &mut shape {
            *dim = read_u64_le(bytes, cursor)?;
        }
        ensure_prefix_shape_compatible(
            shape,
            live_shape4,
            &format!("deserialize {family}[{slot_idx}].tq"),
        )?;
        shape
    } else {
        live_shape4
    };
    let norms_per_pos = read_u32_le(bytes, cursor)?;
    ensure!(
        norms_per_pos > 0,
        "QH35 deserialize: {family}[{slot_idx}].tq norms_per_pos = 0 (invalid)"
    );
    let packed_shape: Vec<usize> = packed_shape4.iter().map(|d| *d as usize).collect();
    let norms_shape: Vec<usize> = vec![
        packed_shape4[0] as usize,
        packed_shape4[1] as usize,
        packed_shape4[2] as usize,
        norms_per_pos as usize,
    ];
    let expected_packed_bytes: usize = packed_shape.iter().product();
    let expected_norms_bytes: usize = norms_shape.iter().product::<usize>() * 4;

    // Helper closure to read one byte_len-prefixed buffer into a fresh,
    // CORRECTLY-SHAPED MlxBuffer (packed = U8, norms = F32).
    let read_blob = |bytes: &[u8],
                     cursor: &mut usize,
                     device: &MlxDevice,
                     name: &str,
                     dtype: DType,
                     shape: &[usize],
                     expected_bytes: usize|
     -> Result<MlxBuffer> {
        let n = read_u64_le(bytes, cursor)? as usize;
        ensure!(
            n == expected_bytes,
            "QH35 deserialize: {family}[{slot_idx}].tq.{name} on-disk byte_len {} != \
             cfg-derived {} (shape {:?}) — writer/reader shape drift",
            n,
            expected_bytes,
            shape
        );
        let src = read_bytes(bytes, cursor, n)?;
        let mut buf = device
            .alloc_buffer(n, dtype, shape.to_vec())
            .map_err(|e| anyhow!("QH35 deserialize: alloc {family}[{slot_idx}].tq.{name}: {e}"))?;
        let dst = buf.as_mut_slice::<u8>().map_err(|e| {
            anyhow!("QH35 deserialize: {family}[{slot_idx}].tq.{name} mut_slice: {e}")
        })?;
        ensure!(
            dst.len() == src.len(),
            "QH35 deserialize: {family}[{slot_idx}].tq.{name} dst.len {} != src.len {}",
            dst.len(),
            src.len()
        );
        dst.copy_from_slice(src);
        Ok(buf)
    };

    let k_packed = read_blob(
        bytes,
        cursor,
        device,
        "k_packed",
        DType::U8,
        &packed_shape,
        expected_packed_bytes,
    )?;
    let k_norms = read_blob(
        bytes,
        cursor,
        device,
        "k_norms",
        DType::F32,
        &norms_shape,
        expected_norms_bytes,
    )?;
    let v_packed = read_blob(
        bytes,
        cursor,
        device,
        "v_packed",
        DType::U8,
        &packed_shape,
        expected_packed_bytes,
    )?;
    let v_norms = read_blob(
        bytes,
        cursor,
        device,
        "v_norms",
        DType::F32,
        &norms_shape,
        expected_norms_bytes,
    )?;

    Ok(TqKvSnapshot {
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        norms_per_pos,
    })
}

/// Deserialize a QH35 envelope back into a `HybridKvCacheSnapshot` against
/// a freshly-allocated set of buffers via `device`. Cursor=0; the entire
/// envelope is consumed.
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
    let snap = deserialize_hybrid_snapshot_at_cursor(bytes, &mut cursor, cfg, device)?;
    Ok(snap)
}

/// Iter-6b.3 — cursor-aware variant of `deserialize_hybrid_snapshot`. The
/// caller controls the read position so a sidecar metadata block can be
/// composed onto the tail of the envelope (see
/// `deserialize_hybrid_with_sidecar`).
///
/// The cursor advances past the consumed snapshot bytes; the caller is
/// responsible for either asserting end-of-buffer (no sidecar expected)
/// or invoking `deserialize_lcp_sidecar(bytes, cursor)` to consume the
/// sidecar block that immediately follows.
pub fn deserialize_hybrid_snapshot_at_cursor(
    bytes: &[u8],
    cursor: &mut usize,
    cfg: &Qwen35HybridConfig,
    device: &MlxDevice,
) -> Result<HybridKvCacheSnapshot> {
    let magic = read_bytes(bytes, cursor, 4)?;
    ensure!(
        magic == QH35_MAGIC,
        "QH35 deserialize: bad magic {:?} (expected {:?})",
        magic,
        QH35_MAGIC
    );

    let codec_version = read_u32_le(bytes, cursor)?;
    // ADR-027 sub-sub-iter 23a-γ: v2 adds per-slot kv_present byte;
    // accept BOTH v1 (legacy, always-Some) and v2 (Optional). v1
    // envelopes are read with implicit kv_present=1 for every slot.
    // iter-36 (sub-iter 23d-β): v3 adds tq_present:u8 per slot. v1/v2
    // envelopes are read with implicit tq_present=0.
    // sub-iter 23d-γ: v4 adds per-MTP kv_present:u8. v1..v3 envelopes
    // are read with implicit MTP kv_present=1.
    ensure!(
        codec_version >= 1 && codec_version <= 5,
        "QH35 deserialize: unsupported codec_version {} (expected 1 through 5)",
        codec_version
    );
    let codec_v2 = codec_version >= 2;
    let codec_v3 = codec_version >= 3;
    let codec_v4 = codec_version >= 4;
    let codec_v5 = codec_version >= 5;

    let header_cfg = Qwen35HybridConfig {
        n_full_attn: read_u32_le(bytes, cursor)?,
        n_linear_attn: read_u32_le(bytes, cursor)?,
        has_mtp: read_u8(bytes, cursor)? != 0,
        full_attn_codec: FullAttnCodec::from_u8(read_u8(bytes, cursor)?)?,
        n_seqs: read_u32_le(bytes, cursor)?,
        // Shapes are per-(slot, family) and not in the header — copy from
        // runtime cfg so the assert_matches comparison treats them as
        // equal (per-slot shape is validated against cfg in the body
        // loops below). kv_substrate is likewise fingerprint-only (not
        // serialized) — copy from runtime cfg.
        full_attn_shape: cfg.full_attn_shape,
        linear_conv_shape: cfg.linear_conv_shape,
        linear_recurrent_shape: cfg.linear_recurrent_shape,
        mtp_shape: cfg.mtp_shape,
        kv_substrate: cfg.kv_substrate,
    };
    let _reserved = read_u16_le(bytes, cursor)?;

    // Validate header against expected config (shape is checked per-slot).
    {
        let mut h = header_cfg.clone();
        h.full_attn_shape = cfg.full_attn_shape;
        h.assert_matches(cfg)
            .context("QH35 deserialize: envelope header / runtime config mismatch")?;
    }

    let mut full_attn_k: Vec<Option<MlxBuffer>> = Vec::with_capacity(cfg.n_full_attn as usize);
    let mut full_attn_v: Vec<Option<MlxBuffer>> = Vec::with_capacity(cfg.n_full_attn as usize);
    let mut full_attn_current_len: Vec<Vec<u32>> = Vec::with_capacity(cfg.n_full_attn as usize);
    // iter-36 (sub-iter 23d-β): TQ snapshot per slot. v3 deserializes
    // from the envelope; v1/v2 leave all entries None.
    let mut full_attn_tq: Vec<Option<TqKvSnapshot>> =
        (0..cfg.n_full_attn as usize).map(|_| None).collect();

    for expected_slot in 0..cfg.n_full_attn as usize {
        let slot_idx = read_u32_le(bytes, cursor)? as usize;
        ensure!(
            slot_idx == expected_slot,
            "QH35 deserialize: full_attn slot order mismatch — got {} expected {}",
            slot_idx,
            expected_slot
        );

        // ADR-027 sub-sub-iter 23a-γ: per-slot kv_present byte (v2 only).
        // v1 envelopes implicitly treat every slot as kv_present=1.
        let kv_present: u8 = if codec_v2 {
            let b = read_u8(bytes, cursor)?;
            ensure!(
                b == QH35_KV_PRESENT || b == QH35_KV_ABSENT,
                "QH35 deserialize: invalid kv_present byte {} at full_attn[{slot_idx}] \
                 (expected 0 or 1)",
                b
            );
            b
        } else {
            QH35_KV_PRESENT
        };

        if kv_present == QH35_KV_ABSENT {
            // TQ-only slot: no shape, no byte_lens, no payload — just
            // current_len for snapshot bookkeeping.
            let mut current_len = Vec::with_capacity(cfg.n_seqs as usize);
            for _ in 0..cfg.n_seqs {
                current_len.push(read_u32_le(bytes, cursor)?);
            }
            full_attn_k.push(None);
            full_attn_v.push(None);
            full_attn_current_len.push(current_len);
            // ADR-027 Phase B iter-36 (sub-iter 23d-β): v3 tq_present
            // byte AFTER the kv block, regardless of kv_present state.
            // v1/v2 envelopes don't have this byte — leave full_attn_tq[i]
            // at its initial None.
            if codec_v3 {
                let tq_byte = read_u8(bytes, cursor)?;
                ensure!(
                    tq_byte == QH35_TQ_PRESENT || tq_byte == QH35_TQ_ABSENT,
                    "QH35 deserialize: invalid tq_present byte {} at full_attn[{slot_idx}] \
                     (expected 0 or 1)",
                    tq_byte
                );
                if tq_byte == QH35_TQ_PRESENT {
                    full_attn_tq[slot_idx] = Some(deserialize_tq_blob(
                        bytes,
                        cursor,
                        device,
                        slot_idx,
                        "full_attn",
                        cfg.full_attn_shape,
                        codec_v5,
                    )?);
                }
            }
            continue;
        }

        let mut shape_arr = [0u64; 4];
        for dim in &mut shape_arr {
            *dim = read_u64_le(bytes, cursor)?;
        }
        if codec_v5 {
            ensure_prefix_shape_compatible(
                shape_arr,
                cfg.full_attn_shape,
                &format!("deserialize full_attn[{slot_idx}]"),
            )?;
        } else {
            ensure!(
                shape_arr == cfg.full_attn_shape,
                "QH35 deserialize: full_attn[{slot_idx}] shape on disk {:?} != cfg {:?}",
                shape_arr,
                cfg.full_attn_shape
            );
        }
        let k_byte_len = read_u64_le(bytes, cursor)? as usize;
        let v_byte_len = read_u64_le(bytes, cursor)? as usize;

        let mut current_len = Vec::with_capacity(cfg.n_seqs as usize);
        for _ in 0..cfg.n_seqs {
            current_len.push(read_u32_le(bytes, cursor)?);
        }

        let k_src = read_bytes(bytes, cursor, k_byte_len)?;
        let v_src = read_bytes(bytes, cursor, v_byte_len)?;

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
        // ADR-027 sub-sub-iter 23a-β: codec emits Some today (iter-23d
        // will branch on a kv_present byte to support None for TQ-only).
        full_attn_k.push(Some(k_buf));
        full_attn_v.push(Some(v_buf));
        full_attn_current_len.push(current_len);
        // iter-36 (sub-iter 23d-β): tq_present byte after kv block on v3.
        if codec_v3 {
            let tq_byte = read_u8(bytes, cursor)?;
            ensure!(
                tq_byte == QH35_TQ_PRESENT || tq_byte == QH35_TQ_ABSENT,
                "QH35 deserialize: invalid tq_present byte {} at full_attn[{slot_idx}] \
                 (expected 0 or 1)",
                tq_byte
            );
            if tq_byte == QH35_TQ_PRESENT {
                full_attn_tq[slot_idx] = Some(deserialize_tq_blob(
                    bytes,
                    cursor,
                    device,
                    slot_idx,
                    "full_attn",
                    cfg.full_attn_shape,
                    codec_v5,
                )?);
            }
        }
    }

    // --- Per linear-attn slot (iter-3) ---
    let conv_shape_usize: Vec<usize> = cfg.linear_conv_shape.iter().map(|d| *d as usize).collect();
    let recurrent_shape_usize: Vec<usize> = cfg
        .linear_recurrent_shape
        .iter()
        .map(|d| *d as usize)
        .collect();
    let expected_conv_bytes =
        (cfg.linear_conv_shape.iter().product::<u64>() as usize) * std::mem::size_of::<f32>();
    let expected_recurrent_bytes =
        (cfg.linear_recurrent_shape.iter().product::<u64>() as usize) * std::mem::size_of::<f32>();
    let mut linear_conv: Vec<MlxBuffer> = Vec::with_capacity(cfg.n_linear_attn as usize);
    let mut linear_recurrent: Vec<MlxBuffer> = Vec::with_capacity(cfg.n_linear_attn as usize);
    for expected_slot in 0..cfg.n_linear_attn as usize {
        let slot_idx = read_u32_le(bytes, cursor)? as usize;
        ensure!(
            slot_idx == expected_slot,
            "QH35 deserialize: linear_attn slot order mismatch — got {} expected {}",
            slot_idx,
            expected_slot
        );
        let conv_byte_len = read_u64_le(bytes, cursor)? as usize;
        let rec_byte_len = read_u64_le(bytes, cursor)? as usize;
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
        let conv_src = read_bytes(bytes, cursor, conv_byte_len)?;
        let rec_src = read_bytes(bytes, cursor, rec_byte_len)?;
        let mut conv_buf = device
            .alloc_buffer(conv_byte_len, DType::F32, conv_shape_usize.clone())
            .map_err(|e| anyhow!("QH35 deserialize: alloc linear_conv[{slot_idx}]: {e}"))?;
        let mut rec_buf = device
            .alloc_buffer(rec_byte_len, DType::F32, recurrent_shape_usize.clone())
            .map_err(|e| anyhow!("QH35 deserialize: alloc linear_recurrent[{slot_idx}]: {e}"))?;
        {
            let conv_dst = conv_buf
                .as_mut_slice::<u8>()
                .map_err(|e| anyhow!("QH35 deserialize: linear_conv[{slot_idx}] mut_slice: {e}"))?;
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
        // sub-iter 23d-γ (codec v4): per-MTP kv_present byte mirroring
        // the full-attn slots' v2 byte. v1..v3 envelopes implicitly
        // treat MTP K/V as present (the only shape they could emit).
        let mtp_kv_present: u8 = if codec_v4 {
            let b = read_u8(bytes, cursor)?;
            ensure!(
                b == QH35_KV_PRESENT || b == QH35_KV_ABSENT,
                "QH35 deserialize: invalid mtp kv_present byte {} \
                 (expected 0 or 1)",
                b
            );
            b
        } else {
            QH35_KV_PRESENT
        };

        let (mk_buf, mv_buf, mtp_current_len) = if mtp_kv_present == QH35_KV_PRESENT {
            let mut mk_shape_arr = [0u64; 4];
            for dim in &mut mk_shape_arr {
                *dim = read_u64_le(bytes, cursor)?;
            }
            if codec_v5 {
                ensure_prefix_shape_compatible(mk_shape_arr, cfg.mtp_shape, "deserialize mtp")?;
            } else {
                ensure!(
                    mk_shape_arr == cfg.mtp_shape,
                    "QH35 deserialize: mtp shape on disk {:?} != cfg.mtp_shape {:?}",
                    mk_shape_arr,
                    cfg.mtp_shape
                );
            }
            let mk_byte_len = read_u64_le(bytes, cursor)? as usize;
            let mv_byte_len = read_u64_le(bytes, cursor)? as usize;
            let mut current_len = Vec::with_capacity(cfg.n_seqs as usize);
            for _ in 0..cfg.n_seqs {
                current_len.push(read_u32_le(bytes, cursor)?);
            }
            let mk_src = read_bytes(bytes, cursor, mk_byte_len)?;
            let mv_src = read_bytes(bytes, cursor, mv_byte_len)?;
            let mtp_shape_usize: Vec<usize> = mk_shape_arr.iter().map(|d| *d as usize).collect();
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
            (Some(mk_buf), Some(mv_buf), current_len)
        } else {
            // TQ-only mode: current_len only, K/V reconstruct as None
            // (the MTP TQ payload is read from the tq block below).
            let mut current_len = Vec::with_capacity(cfg.n_seqs as usize);
            for _ in 0..cfg.n_seqs {
                current_len.push(read_u32_le(bytes, cursor)?);
            }
            (None, None, current_len)
        };

        // iter-36 (sub-iter 23d-β): MTP TQ payload after kv block on v3.
        let mtp_tq = if codec_v3 {
            let tq_byte = read_u8(bytes, cursor)?;
            ensure!(
                tq_byte == QH35_TQ_PRESENT || tq_byte == QH35_TQ_ABSENT,
                "QH35 deserialize: invalid tq_present byte {} at mtp \
                 (expected 0 or 1)",
                tq_byte
            );
            if tq_byte == QH35_TQ_PRESENT {
                Some(deserialize_tq_blob(
                    bytes,
                    cursor,
                    device,
                    0,
                    "mtp",
                    cfg.mtp_shape,
                    codec_v5,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        Some(MtpKvSnapshot {
            // sub-iter 23d-γ (codec v4): None in TQ-only mode (payload
            // rides the TQ block); Some in every earlier-version mode.
            k: mk_buf,
            v: mv_buf,
            current_len: mtp_current_len,
            tq: mtp_tq,
        })
    } else {
        None
    };

    Ok(HybridKvCacheSnapshot {
        full_attn_k,
        full_attn_v,
        full_attn_current_len,
        // iter-36 (sub-iter 23d-β): full_attn_tq populated by the
        // per-slot loop above (v3 reads tq_present + payload; v1/v2
        // leave entries at the initial-None state set at vec-init).
        full_attn_tq,
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
// ADR-027 Phase A iter-6b.3 — LCP sidecar metadata
// ---------------------------------------------------------------------------
//
// The QH35 envelope (iter-2..iter-4) carries the snapshot bytes only.  The
// in-memory `LcpRegistry::store` entry that produced a snapshot also needs
// `(LcpKey, prompt_tokens, sliding_window, linear_capacity)` to be
// reinserted on cold start — none of which are derivable from the snapshot
// payload or the on-disk filename (the filename is a one-way SHA hex of
// the LcpKey, not the key itself).
//
// Iter-6b.3 introduces a SIDECAR block that lives at the tail of the QH35
// envelope, marked with its own magic so a v1 reader (just `deserialize_
// hybrid_snapshot`) can ignore the trailing bytes safely.  The sidecar
// codec is intentionally orthogonal to the snapshot codec — adding /
// changing sidecar fields does not require bumping `QH35_CODEC_VERSION`,
// preserving Chesterton's fence around the snapshot codec.
//
// On-disk sidecar layout:
// ```text
// [magic: 4 bytes "QH3M"]
// [version: u32 LE = 1]
// [model_fingerprint: 32 bytes]    # ModelFingerprint([u8; 32])
// [tenant_id_len: u32 LE]
// [tenant_id: tenant_id_len bytes]
// [params_hash: u64 LE]
// [prompt_tokens_count: u64 LE]
// [prompt_tokens: u32 × prompt_tokens_count LE]
// [sliding_window: u64 LE]
// [linear_capacity: u64 LE]
// ```

/// Magic bytes prefixing every QH3M (Qwen3.5 Hybrid sidecar Metadata) block.
/// ASCII for "QH3M" so hex dumps spot the boundary visually.
pub const QH3M_SIDECAR_MAGIC: [u8; 4] = *b"QH3M";

/// Sidecar codec version. v1 ships in iter-6b.3.
pub const QH3M_SIDECAR_VERSION: u32 = 1;

/// Sidecar metadata appended to a QH35 envelope so a cold-start hydrate
/// path can reconstruct the full `LcpRegistry::store(...)` call without
/// re-deriving any field from the live request.
///
/// Held + emitted by `Qwen35DiskPersistor::write` and consumed by
/// `Qwen35DiskPersistor::read` / `hydrate_for_cfg`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LcpSidecarMetadata {
    /// Same `ModelFingerprint` as the in-memory `LcpKey.model_fingerprint`.
    pub model_fingerprint: ModelFingerprint,
    /// Tenant identifier — same value passed at store-time.
    pub tenant_id: String,
    /// Sampling-params hash — same value passed at store-time.
    pub params_hash: u64,
    /// Full prompt token IDs (the SOURCE prompt that produced the snapshot).
    /// Required so `LcpRegistry::lookup` can compute LCP against new prompts.
    pub prompt_tokens: Vec<u32>,
    /// Sliding-window size at store-time. Stored as u64 (over-aligned for
    /// portability; production sliding_window fits in 32 bits).
    pub sliding_window: u64,
    /// Linear-attn capacity at store-time. Same u64 rationale.
    pub linear_capacity: u64,
}

/// Serialize a sidecar metadata block (no QH35 envelope around it). The
/// disk persistor calls this and APPENDS the bytes to the QH35 envelope's
/// tail.
pub fn serialize_lcp_sidecar(sidecar: &LcpSidecarMetadata) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        4 + 4 + 32 + 4 + sidecar.tenant_id.len() + 8 + 8 + sidecar.prompt_tokens.len() * 4 + 8 + 8,
    );
    out.extend_from_slice(&QH3M_SIDECAR_MAGIC);
    write_u32_le(&mut out, QH3M_SIDECAR_VERSION);
    out.extend_from_slice(&sidecar.model_fingerprint.0);
    write_u32_le(&mut out, sidecar.tenant_id.len() as u32);
    out.extend_from_slice(sidecar.tenant_id.as_bytes());
    write_u64_le(&mut out, sidecar.params_hash);
    write_u64_le(&mut out, sidecar.prompt_tokens.len() as u64);
    for &tok in &sidecar.prompt_tokens {
        write_u32_le(&mut out, tok);
    }
    write_u64_le(&mut out, sidecar.sliding_window);
    write_u64_le(&mut out, sidecar.linear_capacity);
    out
}

/// Deserialize a sidecar metadata block from `bytes` starting at `*cursor`.
/// Advances `*cursor` past the consumed bytes.
pub fn deserialize_lcp_sidecar(bytes: &[u8], cursor: &mut usize) -> Result<LcpSidecarMetadata> {
    let magic = read_bytes(bytes, cursor, 4)?;
    ensure!(
        magic == QH3M_SIDECAR_MAGIC,
        "QH3M sidecar deserialize: bad magic {:?} (expected {:?})",
        magic,
        QH3M_SIDECAR_MAGIC
    );
    let version = read_u32_le(bytes, cursor)?;
    ensure!(
        version == QH3M_SIDECAR_VERSION,
        "QH3M sidecar deserialize: unsupported version {} (expected {})",
        version,
        QH3M_SIDECAR_VERSION
    );
    let mut fp_bytes = [0u8; 32];
    fp_bytes.copy_from_slice(read_bytes(bytes, cursor, 32)?);
    let model_fingerprint = ModelFingerprint(fp_bytes);
    let tenant_len = read_u32_le(bytes, cursor)? as usize;
    // Hard cap on tenant_id len to bound the alloc + reject obviously
    // corrupt envelopes early. Production tenant_ids are short
    // identifiers (e.g. "default", "qwen35:lcp_chunk:64").
    ensure!(
        tenant_len <= 64 * 1024,
        "QH3M sidecar deserialize: tenant_id length {} exceeds 64 KiB cap",
        tenant_len
    );
    let tenant_bytes = read_bytes(bytes, cursor, tenant_len)?;
    let tenant_id = std::str::from_utf8(tenant_bytes)
        .map_err(|e| anyhow!("QH3M sidecar deserialize: tenant_id not UTF-8: {e}"))?
        .to_string();
    let params_hash = read_u64_le(bytes, cursor)?;
    let prompt_tokens_count = read_u64_le(bytes, cursor)? as usize;
    // Prompt-token cap matches the largest reasonable single-prompt size
    // (production prompts < 2 MiB tokens × 4 = 8 MiB).
    ensure!(
        prompt_tokens_count <= 16 * 1024 * 1024,
        "QH3M sidecar deserialize: prompt_tokens count {} exceeds 16M cap",
        prompt_tokens_count
    );
    let mut prompt_tokens = Vec::with_capacity(prompt_tokens_count);
    for _ in 0..prompt_tokens_count {
        prompt_tokens.push(read_u32_le(bytes, cursor)?);
    }
    let sliding_window = read_u64_le(bytes, cursor)?;
    let linear_capacity = read_u64_le(bytes, cursor)?;
    Ok(LcpSidecarMetadata {
        model_fingerprint,
        tenant_id,
        params_hash,
        prompt_tokens,
        sliding_window,
        linear_capacity,
    })
}

/// Compose the snapshot envelope + sidecar block into a single Vec<u8>.
/// The disk persistor uses this as its write codec; the sidecar tail
/// allows the cold-start hydrate path to reinsert the snapshot back into
/// `LcpRegistry` with the original key + prompt + capacity fields.
pub fn serialize_hybrid_with_sidecar(
    snapshot: &HybridKvCacheSnapshot,
    cfg: &Qwen35HybridConfig,
    sidecar: &LcpSidecarMetadata,
) -> Result<Vec<u8>> {
    let mut out = serialize_hybrid_snapshot(snapshot, cfg)?;
    out.extend_from_slice(&serialize_lcp_sidecar(sidecar));
    Ok(out)
}

/// Round-trip pair of `serialize_hybrid_with_sidecar`. Returns the
/// reconstructed snapshot AND the sidecar metadata; the disk persistor
/// uses the sidecar to re-insert into the in-memory LcpRegistry.
pub fn deserialize_hybrid_with_sidecar(
    bytes: &[u8],
    cfg: &Qwen35HybridConfig,
    device: &MlxDevice,
) -> Result<(HybridKvCacheSnapshot, LcpSidecarMetadata)> {
    let mut cursor = 0usize;
    let snap = deserialize_hybrid_snapshot_at_cursor(bytes, &mut cursor, cfg, device)?;
    let sidecar = deserialize_lcp_sidecar(bytes, &mut cursor)
        .context("QH35-with-sidecar: sidecar block missing or invalid at envelope tail")?;
    Ok((snap, sidecar))
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
        let shape_usize: Vec<usize> = cfg.full_attn_shape.iter().map(|d| *d as usize).collect();

        let mut full_attn_k: Vec<Option<MlxBuffer>> = Vec::with_capacity(cfg.n_full_attn as usize);
        let mut full_attn_v: Vec<Option<MlxBuffer>> = Vec::with_capacity(cfg.n_full_attn as usize);
        let mut full_attn_current_len: Vec<Vec<u32>> = Vec::with_capacity(cfg.n_full_attn as usize);

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
            // ADR-027 sub-sub-iter 23a-β: test fixture wraps in Some.
            full_attn_k.push(Some(k));
            full_attn_v.push(Some(v));
            // current_len: per seq, deterministic.
            let cl: Vec<u32> = (0..cfg.n_seqs).map(|s| (slot as u32) * 100 + s).collect();
            full_attn_current_len.push(cl);
        }

        let n_full_attn = full_attn_k.len();
        HybridKvCacheSnapshot {
            full_attn_k,
            full_attn_v,
            full_attn_current_len,
            // iter-35 (sub-iter 23d-α): test fixture, no TQ.
            full_attn_tq: (0..n_full_attn).map(|_| None).collect(),
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
            kv_substrate: KvSubstrate::F32Only,
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
        let rec_elems: usize = cfg.linear_recurrent_shape.iter().product::<u64>() as usize;
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
            // ADR-027 sub-sub-iter 23a-β: Optional full-attn K/V — compare
            // Some-to-Some byte-equal (None-to-None test path lands iter-23c+).
            let ak = a.full_attn_k[i]
                .as_ref()
                .expect("a.k some")
                .as_slice::<u8>()
                .expect("ak slice");
            let bk = b.full_attn_k[i]
                .as_ref()
                .expect("b.k some")
                .as_slice::<u8>()
                .expect("bk slice");
            if ak != bk {
                return false;
            }
            let av = a.full_attn_v[i]
                .as_ref()
                .expect("a.v some")
                .as_slice::<u8>()
                .expect("av slice");
            let bv = b.full_attn_v[i]
                .as_ref()
                .expect("b.v some")
                .as_slice::<u8>()
                .expect("bv slice");
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
        // ADR-027 sub-sub-iter 23a-γ (codec v2): per-slot kv_present(1)
        // byte adds +1 byte per slot. Per-slot overhead: slot_idx(4) +
        // kv_present(1) + shape(32) + k_byte_len(8) + v_byte_len(8) +
        // current_len(4 * n_seqs=1) = 57. Per-slot body: K(256) +
        // V(256) = 512. Per-slot total = 569. n_full_attn=3.
        // iter-36 (sub-iter 23d-β): v3 adds tq_present:u8 per slot.
        // Per-slot overhead is now 569 (v2 body) + 1 (tq_present) = 570.
        assert_eq!(bytes.len(), 24 + 3 * 570);
        let restored = deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
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
        let restored = deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
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
        let mut bytes =
            serialize_hybrid_snapshot(&synth_full_attn_only_snapshot(&device, &cfg), &cfg)
                .expect("serialize");
        bytes[0] = b'X';
        let err = deserialize_hybrid_snapshot(&bytes, &cfg, &device).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("bad magic"),
            "expected magic error, got: {msg}"
        );
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
        let mut bytes =
            serialize_hybrid_snapshot(&synth_full_attn_only_snapshot(&device, &cfg), &cfg)
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
        let restored = deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
        assert!(snapshots_byte_equal(&snap, &restored));
        // Per-linear-slot overhead = slot_idx(4) + conv_len(8) + rec_len(8) = 20.
        // Per-linear-slot body = conv(4*3*1*4) + rec(4*8*2*1*4) = 48 + 256 = 304.
        // Per-slot total = 324. With 3 linear slots = 972 bytes.
        // Plus header(24) + 2 full-attn slots @ 569 each (codec v2
        // adds per-slot kv_present byte) = 24 + 1138 = 1162.
        // Total = 1160 + 972 = 2132 bytes.
        // iter-36 (sub-iter 23d-β): v3 adds tq_present:u8 per slot.
        // Per-full-attn-slot overhead 569 → 570; linear-attn unchanged
        // (TQ doesn't apply to linear-attn slots).
        assert_eq!(bytes.len(), 24 + 2 * 570 + 3 * 324);
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
            let shape_usize: Vec<usize> = cfg.mtp_shape.iter().map(|d| *d as usize).collect();
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
                // ADR-027 sub-sub-iter 23a-α: test fixture wraps in Some.
                k: Some(k),
                v: Some(v),
                current_len,
                // iter-35 (sub-iter 23d-α): test fixture, no TQ.
                tq: None,
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
        let restored = deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
        // Compare full + linear + MTP byte-equal.
        assert!(snapshots_byte_equal(&snap, &restored));
        assert!(restored.mtp.is_some());
        let r_mtp = restored.mtp.as_ref().unwrap();
        let s_mtp = snap.mtp.as_ref().unwrap();
        // ADR-027 sub-sub-iter 23a-α: Optional MTP K/V — codec round-trip
        // produces Some today (iter-23d adds None support).
        let r_k = r_mtp
            .k
            .as_ref()
            .expect("r_mtp.k some")
            .as_slice::<u8>()
            .expect("rk slice");
        let s_k = s_mtp
            .k
            .as_ref()
            .expect("s_mtp.k some")
            .as_slice::<u8>()
            .expect("sk slice");
        assert_eq!(r_k, s_k);
        let r_v = r_mtp
            .v
            .as_ref()
            .expect("r_mtp.v some")
            .as_slice::<u8>()
            .expect("rv slice");
        let s_v = s_mtp
            .v
            .as_ref()
            .expect("s_mtp.v some")
            .as_slice::<u8>()
            .expect("sv slice");
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
        let restored = deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
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

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase A iter-6b.3 — sidecar codec round-trip tests
    // ──────────────────────────────────────────────────────────────────

    fn synth_sidecar() -> LcpSidecarMetadata {
        // Deterministic byte pattern across the fingerprint so tests are
        // reproducible.
        let mut fp = [0u8; 32];
        for (i, b) in fp.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(13).wrapping_add(17);
        }
        LcpSidecarMetadata {
            model_fingerprint: ModelFingerprint(fp),
            tenant_id: "qwen35:lcp_chunk:64".to_string(),
            params_hash: 0xDEADBEEF_CAFEBABE,
            prompt_tokens: vec![1, 2, 3, 4, 5, 100, 200, 300, 400, 500],
            sliding_window: 8192,
            linear_capacity: 4096,
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 sub-sub-iter 23a-γ — codec v2 None-K/V round-trip tests
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn qh35_codec_v2_round_trip_none_full_attn_k_v_byte_equal() {
        // **iter-23a-γ deliverable test**: synthetic snapshot with
        // None K/V on every full-attn slot (TQ-only mode that iter-23c+
        // will produce). Round-trips through serialize/deserialize via
        // the v2 codec's kv_present=0 byte. Restored snapshot keeps
        // None entries; current_len bookkeeping is preserved.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg(3, 1);
        // Build a None-K/V snapshot directly.
        let n_full_attn = cfg.n_full_attn as usize;
        let snap = HybridKvCacheSnapshot {
            full_attn_k: (0..n_full_attn).map(|_| None).collect(),
            full_attn_v: (0..n_full_attn).map(|_| None).collect(),
            full_attn_current_len: (0..n_full_attn)
                .map(|slot| (0..cfg.n_seqs).map(|s| (slot as u32) * 100 + s).collect())
                .collect(),
            // iter-35 (sub-iter 23d-α): test fixture, no TQ.
            full_attn_tq: (0..n_full_attn).map(|_| None).collect(),
            mtp: None,
            linear_conv: Vec::new(),
            linear_recurrent: Vec::new(),
        };
        let bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize");
        // iter-36 (sub-iter 23d-β): v3 envelope.
        // Per-slot None-K/V overhead = slot_idx(4) + kv_present(1) +
        // current_len(4 * n_seqs=1) + tq_present(1) = 10 bytes per slot.
        // 3 slots × 10 = 30. Total = 24 + 30 = 54.
        assert_eq!(bytes.len(), 24 + 3 * 10);

        let restored = deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
        // Every full-attn slot is None.
        for i in 0..n_full_attn {
            assert!(
                restored.full_attn_k[i].is_none(),
                "slot[{i}].k expected None"
            );
            assert!(
                restored.full_attn_v[i].is_none(),
                "slot[{i}].v expected None"
            );
        }
        // current_len bookkeeping preserved.
        assert_eq!(restored.full_attn_current_len, snap.full_attn_current_len);
    }

    #[test]
    fn qh35_codec_v2_rejects_invalid_kv_present_byte() {
        // Defensive: if a corrupt envelope has kv_present=2 (neither
        // 0 nor 1), the deserializer rejects with a clear error.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg(1, 1);
        let snap = synth_full_attn_only_snapshot(&device, &cfg);
        let mut bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize");
        // Locate the kv_present byte: header(24) + slot_idx(4) = offset 28.
        bytes[28] = 99;
        let err = deserialize_hybrid_snapshot(&bytes, &cfg, &device).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("invalid kv_present byte"),
            "expected kv_present validation error, got: {msg}"
        );
    }

    #[test]
    fn qh3m_sidecar_codec_byte_round_trip() {
        let sidecar = synth_sidecar();
        let bytes = serialize_lcp_sidecar(&sidecar);
        let mut cursor = 0usize;
        let restored =
            deserialize_lcp_sidecar(&bytes, &mut cursor).expect("deserialize_lcp_sidecar");
        assert_eq!(cursor, bytes.len(), "sidecar codec must consume all bytes");
        assert_eq!(restored, sidecar);
    }

    /// ADR-027 Phase B iter-36 (sub-iter 23d-β): synthetic snapshot
    /// with `full_attn_tq` populated round-trips byte-equal through
    /// the v3 codec. THE LOAD-BEARING TEST for cross-process replay
    /// in TQ-only mode (the iter-34 production deployment scenario
    /// where slot.k=None at restore time).
    #[test]
    fn qh35_codec_v3_round_trip_with_tq_payload_byte_equal() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let n_full_attn: u32 = 2;
        let cfg = synth_cfg(n_full_attn, 1); // n_seqs=1

        // Build a TQ-only snapshot: K/V are None per slot, but TQ is
        // Some(_) with synthetic deterministic byte content.
        // TQ shape (synth_cfg full_attn_shape = [1,2,8,4]): packed 64 B U8;
        // norms [1,2,8,1] F32 = 64 B. Lengths must match the 23d-γ
        // cfg-derived shape validation in deserialize_tq_blob.
        let mut full_attn_tq: Vec<Option<TqKvSnapshot>> = Vec::new();
        for slot in 0..n_full_attn as usize {
            let packed_shape = vec![1, 2, 8, 4];
            let norms_shape = vec![1, 2, 8, 1];
            let mut k_packed = device
                .alloc_buffer(64, DType::U8, packed_shape.clone())
                .unwrap();
            let mut k_norms = device
                .alloc_buffer(64, DType::F32, norms_shape.clone())
                .unwrap();
            let mut v_packed = device.alloc_buffer(64, DType::U8, packed_shape).unwrap();
            let mut v_norms = device.alloc_buffer(64, DType::F32, norms_shape).unwrap();
            for (i, b) in k_packed
                .as_mut_slice::<u8>()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *b = ((slot * 31 + i * 7) % 251) as u8;
            }
            for (i, b) in v_packed
                .as_mut_slice::<u8>()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *b = ((slot * 13 + i * 11) % 251) as u8;
            }
            for (i, f) in k_norms
                .as_mut_slice::<f32>()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *f = (slot as f32) * 0.5 + (i as f32) * 0.125;
            }
            for (i, f) in v_norms
                .as_mut_slice::<f32>()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *f = (slot as f32) * 0.25 + (i as f32) * 0.0625;
            }
            full_attn_tq.push(Some(TqKvSnapshot {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
                norms_per_pos: 1,
            }));
        }

        let snap = HybridKvCacheSnapshot {
            full_attn_k: (0..n_full_attn as usize).map(|_| None).collect(),
            full_attn_v: (0..n_full_attn as usize).map(|_| None).collect(),
            full_attn_current_len: (0..n_full_attn as usize)
                .map(|s| vec![s as u32 + 1])
                .collect(),
            full_attn_tq,
            mtp: None,
            linear_conv: Vec::new(),
            linear_recurrent: Vec::new(),
        };

        let bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize v3");

        let restored = deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize v3");

        // K/V remain None per slot (TQ-only mode).
        for i in 0..n_full_attn as usize {
            assert!(
                restored.full_attn_k[i].is_none(),
                "slot[{i}].k expected None (TQ-only)"
            );
            assert!(
                restored.full_attn_v[i].is_none(),
                "slot[{i}].v expected None (TQ-only)"
            );
        }
        // TQ payload byte-equal across all 4 buffers per slot.
        for i in 0..n_full_attn as usize {
            let src = snap.full_attn_tq[i].as_ref().unwrap();
            let dst = restored.full_attn_tq[i]
                .as_ref()
                .expect(&format!("restored.full_attn_tq[{i}] must be Some"));
            assert_eq!(
                src.norms_per_pos, dst.norms_per_pos,
                "slot[{i}].norms_per_pos mismatch"
            );
            assert_eq!(
                src.k_packed.as_slice::<u8>().unwrap(),
                dst.k_packed.as_slice::<u8>().unwrap(),
                "slot[{i}].k_packed bytes mismatch"
            );
            assert_eq!(
                src.k_norms.as_slice::<u8>().unwrap(),
                dst.k_norms.as_slice::<u8>().unwrap(),
                "slot[{i}].k_norms bytes mismatch"
            );
            assert_eq!(
                src.v_packed.as_slice::<u8>().unwrap(),
                dst.v_packed.as_slice::<u8>().unwrap(),
                "slot[{i}].v_packed bytes mismatch"
            );
            assert_eq!(
                src.v_norms.as_slice::<u8>().unwrap(),
                dst.v_norms.as_slice::<u8>().unwrap(),
                "slot[{i}].v_norms bytes mismatch"
            );
        }
        assert_eq!(restored.full_attn_current_len, snap.full_attn_current_len);
    }

    /// ADR-027 Phase B iter-36 — backward compatibility: v3 deserializer
    /// accepts v2 envelopes (no tq_present byte). full_attn_tq is all-None.
    #[test]
    fn qh35_codec_v3_deserializer_accepts_v2_envelope_with_implicit_no_tq() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let n_full_attn: u32 = 3;
        let cfg = synth_cfg(n_full_attn, 1); // n_seqs=1
        let snap = synth_full_attn_only_snapshot(&device, &cfg);

        // Synthesize a v3 envelope, then HACK the codec_version byte
        // back to 2 + STRIP the tq_present:u8 bytes per slot to simulate
        // a real v2 envelope on disk.
        let mut bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize v3");
        // Header: magic(4) + codec_version(4) — write 2 in place of 3.
        bytes[4] = 2;
        bytes[5] = 0;
        bytes[6] = 0;
        bytes[7] = 0;
        // Strip the trailing tq_present:u8 from each full-attn slot.
        // v3 per-slot byte size = 570 (per the byte-count test). v2 = 569.
        // Slot 0 starts at offset 24+2 (skip 2-byte _reserved). After
        // header(24) + n_full_attn(4) + n_linear_attn(4) + has_mtp(1) +
        // codec_tag(1) + n_seqs(4) + _reserved(2) = 16 bytes header tail,
        // slot 0 begins at offset 40. Stripping is fragile — easier to
        // skip the test's surgery and just round-trip the v3 directly.
        // For this test, we'll instead build a v2 by writing minimal v2
        // envelope BYTES and verify deserialize works.
        //
        // Practical compromise: use the v2 deserializer's known behavior:
        // codec_version=3 with empty tq is byte-equivalent to codec_version=2
        // EXCEPT for the trailing tq_present=0 byte per slot.
        // The simplest backward-compat assertion: an authentic v2 envelope
        // reconstructed manually. Skip the surgery; rely on v3 round-trip
        // and the v2 round-trip test (already passing post-iter-36 byte
        // assertion update) to prove backward compat.
        let _ = bytes;

        // Build a v2-style envelope by serializing then truncating tq_present
        // bytes. Locate them at the END of each slot's body in v3 layout.
        // The v3 layout for a Some-K/V slot in synth_full_attn_only_snapshot
        // (head_dim=4, max_seq=2, n_kv_heads=1, n_seqs=1):
        //   slot_idx(4) + kv_present(1) + shape(4*8=32) + k_byte_len(8) +
        //   v_byte_len(8) + current_len(4) + k_bytes(32) + v_bytes(32) +
        //   tq_present(1)
        // = 122 bytes per slot in v3. Actually we relied on synth helper
        // sizing — use bytes_per_slot from v2 which we know is 569 vs v3 570.
        // The 1-byte delta is ALWAYS at the slot's tail.
        //
        // To reconstruct v2: take v3 bytes, write codec_version=2, then
        // delete the 1-byte tq_present at each slot's tail.
        let mut v3_bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize");
        // Confirm v3 size matches expectation (header 24 + 3 * 570).
        assert_eq!(v3_bytes.len(), 24 + (n_full_attn as usize) * 570);
        // Header: codec_version → 2.
        v3_bytes[4] = 2;
        // Walk slots in REVERSE and remove the tq_present byte at each tail.
        // V3 slot size = 570; tq_present is the LAST byte of each slot.
        for slot_rev in (0..n_full_attn as usize).rev() {
            // V3 layout: header(24) + slot[0..i] each at full v3 size.
            // After header: slot 0 at offset 24, ends at 24+570.
            // The tq_present byte for slot s is at offset
            // 24 + (s+1) * 570 - 1.
            let tq_byte_offset = 24 + (slot_rev + 1) * 570 - 1;
            v3_bytes.remove(tq_byte_offset);
        }
        // Now v3_bytes has length 24 + 3 * 569 — a valid v2 envelope.
        let v2_bytes = v3_bytes;
        assert_eq!(v2_bytes.len(), 24 + (n_full_attn as usize) * 569);

        // V3 deserializer must accept this v2 envelope; full_attn_tq all None.
        let restored = deserialize_hybrid_snapshot(&v2_bytes, &cfg, &device)
            .expect("v3 deserializer must accept v2 envelope");
        assert_eq!(restored.full_attn_tq.len(), n_full_attn as usize);
        for i in 0..n_full_attn as usize {
            assert!(
                restored.full_attn_tq[i].is_none(),
                "v2 envelope must yield None TQ per slot (got Some at {i})"
            );
            // K/V restored as Some via v2 path.
            assert!(restored.full_attn_k[i].is_some());
            assert!(restored.full_attn_v[i].is_some());
        }
    }

    /// ADR-027 Phase B iter-36 — defensive: corrupt tq_present byte
    /// rejected with clear error.
    #[test]
    fn qh35_codec_v3_rejects_invalid_tq_present_byte() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg(1, 1); // n_seqs=1
        let snap = synth_full_attn_only_snapshot(&device, &cfg);
        let mut bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize");
        // tq_present byte is the last byte of slot 0's body.
        // Slot size = 570 bytes; offset = header(24) + 570 - 1 = 593.
        let tq_byte_offset = 24 + 570 - 1;
        bytes[tq_byte_offset] = 99;
        let err = deserialize_hybrid_snapshot(&bytes, &cfg, &device).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("invalid tq_present byte"),
            "expected tq_present validation error, got: {msg}"
        );
    }

    #[test]
    fn qh3m_sidecar_codec_rejects_bad_magic() {
        let sidecar = synth_sidecar();
        let mut bytes = serialize_lcp_sidecar(&sidecar);
        bytes[0] = b'Z';
        let mut cursor = 0usize;
        let err = deserialize_lcp_sidecar(&bytes, &mut cursor).unwrap_err();
        assert!(
            format!("{err:#}").contains("bad magic"),
            "expected bad-magic error"
        );
    }

    #[test]
    fn qh3m_sidecar_codec_rejects_version_drift() {
        let sidecar = synth_sidecar();
        let mut bytes = serialize_lcp_sidecar(&sidecar);
        // version is at offset 4..8 (LE u32). Bump to 99.
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let mut cursor = 0usize;
        let err = deserialize_lcp_sidecar(&bytes, &mut cursor).unwrap_err();
        assert!(
            format!("{err:#}").contains("version"),
            "expected version-drift error, got: {err:#}"
        );
    }

    #[test]
    fn qh35_with_sidecar_round_trip_byte_equal() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg_full(2, 3, true, 1);
        let snap = synth_full_plus_linear_plus_mtp_snapshot(&device, &cfg);
        let sidecar = synth_sidecar();
        let bytes = serialize_hybrid_with_sidecar(&snap, &cfg, &sidecar)
            .expect("serialize_hybrid_with_sidecar");
        let (restored_snap, restored_sidecar) =
            deserialize_hybrid_with_sidecar(&bytes, &cfg, &device)
                .expect("deserialize_hybrid_with_sidecar");
        assert!(snapshots_byte_equal(&snap, &restored_snap));
        assert_eq!(restored_sidecar, sidecar);
    }

    #[test]
    fn qh35_with_sidecar_full_attn_only_round_trip() {
        // Edge case: the lightest snapshot (full-attn only, no linear, no
        // MTP) still round-trips with sidecar appended.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg(1, 1);
        let snap = synth_full_attn_only_snapshot(&device, &cfg);
        let sidecar = synth_sidecar();
        let bytes = serialize_hybrid_with_sidecar(&snap, &cfg, &sidecar)
            .expect("serialize_hybrid_with_sidecar");
        let (restored_snap, restored_sidecar) =
            deserialize_hybrid_with_sidecar(&bytes, &cfg, &device)
                .expect("deserialize_hybrid_with_sidecar");
        assert!(snapshots_byte_equal(&snap, &restored_snap));
        assert_eq!(restored_sidecar, sidecar);
    }

    #[test]
    fn qh35_with_sidecar_rejects_truncated_envelope() {
        // Truncate the bytes BEFORE the sidecar magic — deserialize must
        // fail loudly rather than silently returning a default sidecar.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg(1, 1);
        let snap = synth_full_attn_only_snapshot(&device, &cfg);
        let sidecar = synth_sidecar();
        let bytes = serialize_hybrid_with_sidecar(&snap, &cfg, &sidecar).expect("serialize");
        // Drop the sidecar tail entirely — only the snapshot remains.
        let truncated = &bytes[..bytes.len() - serialize_lcp_sidecar(&sidecar).len()];
        let err = deserialize_hybrid_with_sidecar(truncated, &cfg, &device).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("sidecar block missing")
                || msg.contains("OOB")
                || msg.contains("bad magic"),
            "expected truncated-envelope error, got: {msg}"
        );
    }

    #[test]
    fn qh3m_sidecar_empty_prompt_tokens_round_trip() {
        // Defensive: sidecar with empty prompt_tokens still round-trips.
        // (LcpRegistry::store rejects empty prompts at insert-time, so
        // production never writes such a sidecar — but the codec must
        // handle it cleanly so a bad write doesn't poison the cache dir.)
        let mut sidecar = synth_sidecar();
        sidecar.prompt_tokens.clear();
        let bytes = serialize_lcp_sidecar(&sidecar);
        let mut cursor = 0usize;
        let restored = deserialize_lcp_sidecar(&bytes, &mut cursor).expect("deserialize");
        assert_eq!(cursor, bytes.len());
        assert_eq!(restored, sidecar);
    }

    // ---------------------------------------------------------------------
    // ADR-027 sub-iter 23d-gamma (2026-08-03) — TQ-only mode coverage:
    // cfg_from_cache shape/substrate derivation, codec v4 MTP kv_present,
    // substrate-namespaced fingerprint, v3 back-compat.
    // ---------------------------------------------------------------------

    /// Minimal real-cache config fixture (mirrors kv_cache.rs's
    /// `tiny_dense_cfg_4layer_for_multi_seq_tests` — duplicated here to
    /// keep the serve/persist layer's tests free of an inference-layer
    /// test-helper dependency).
    fn tiny_qwen35_cfg(mtp: bool) -> crate::inference::models::qwen35::Qwen35Config {
        use crate::inference::models::qwen35::{default_layer_types, Qwen35Config, Qwen35Variant};
        Qwen35Config {
            variant: Qwen35Variant::Dense,
            hidden_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 32,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            layer_types: default_layer_types(4, 2),
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rotary_dim: 8,
            mrope_section: [2, 2, 2, 2],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 4096,
            vocab_size: 256,
            attn_output_gate: true,
            mtp_num_hidden_layers: if mtp { 1 } else { 0 },
            mtp_use_dedicated_embeddings: true,
            intermediate_size: Some(128),
            moe: None,
        }
    }

    #[test]
    fn cfg_from_cache_tq_only_derives_shape_substrate_and_mtp() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_qwen35_cfg(true);
        let cache = HybridKvCache::new_with_options(&cfg, &device, 64, 1, true)
            .expect("alloc TQ-mode cache");
        assert!(cache.tq_kv_active && cache.full_attn[0].k.is_none());

        let derived = cfg_from_cache(&cache, FullAttnCodec::F32Dense)
            .expect("cfg_from_cache must NOT panic in TQ-only mode (23d-gamma)");
        // Shape derived from tq.k_packed == the F32 logical layout.
        assert_eq!(derived.full_attn_shape, [1, 2, 64, 32]);
        assert_eq!(derived.mtp_shape, [1, 2, 64, 32]);
        assert_eq!(derived.n_full_attn, 2);
        assert_eq!(derived.n_linear_attn, 2);
        assert!(derived.has_mtp);
        assert_eq!(derived.n_seqs, 1);
        assert_eq!(derived.kv_substrate, KvSubstrate::TqOnly);
    }

    #[test]
    fn cfg_from_cache_f32_only_substrate() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_qwen35_cfg(true);
        let cache = HybridKvCache::new(&cfg, &device, 64, 1).expect("alloc legacy cache");
        assert!(!cache.tq_kv_active && cache.full_attn[0].tq.is_none());

        let derived = cfg_from_cache(&cache, FullAttnCodec::F32Dense).expect("cfg_from_cache");
        assert_eq!(derived.full_attn_shape, [1, 2, 64, 32]);
        assert_eq!(derived.kv_substrate, KvSubstrate::F32Only);
    }

    #[test]
    fn cfg_from_cache_mixed_substrate_hard_errors() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = tiny_qwen35_cfg(false);
        let mut cache = HybridKvCache::new(&cfg, &device, 64, 1).expect("alloc legacy cache");
        // Graft TQ onto slot 1 only (slot 0 stays F32) — violates the
        // allocator's uniform-substrate invariant.
        cache.full_attn[1].k = None;
        cache.full_attn[1].v = None;
        cache.full_attn[1].tq = Some(
            crate::inference::models::qwen35::kv_cache::alloc_tq_full_attn_buffers(
                &cfg, &device, 64, 1,
            )
            .expect("alloc tq buffers"),
        );
        let err = cfg_from_cache(&cache, FullAttnCodec::F32Dense).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("mixed KV substrates"),
            "expected mixed-substrate error, got: {msg}"
        );
    }

    /// Build a fully TQ-populated snapshot (full-attn + MTP, k/v=None,
    /// tq=Some) with deterministic byte patterns.
    fn synth_tq_only_snapshot_with_mtp(
        device: &MlxDevice,
        cfg: &Qwen35HybridConfig,
    ) -> HybridKvCacheSnapshot {
        let mut snap = synth_full_attn_only_snapshot(device, cfg);
        // Flip full-attn slots to TQ-only: k/v=None, tq=Some.
        let n_full = cfg.n_full_attn as usize;
        snap.full_attn_k = (0..n_full).map(|_| None).collect();
        snap.full_attn_v = (0..n_full).map(|_| None).collect();
        snap.full_attn_tq = (0..n_full)
            .map(|slot| {
                let packed_shape: Vec<usize> =
                    cfg.full_attn_shape.iter().map(|d| *d as usize).collect();
                let norms_shape = vec![packed_shape[0], packed_shape[1], packed_shape[2], 1];
                let packed_bytes = packed_shape.iter().product();
                let norms_bytes = norms_shape.iter().product::<usize>() * 4;
                let mut k_packed = device
                    .alloc_buffer(packed_bytes, DType::U8, packed_shape.clone())
                    .unwrap();
                let mut k_norms = device
                    .alloc_buffer(norms_bytes, DType::F32, norms_shape.clone())
                    .unwrap();
                let mut v_packed = device
                    .alloc_buffer(packed_bytes, DType::U8, packed_shape)
                    .unwrap();
                let mut v_norms = device
                    .alloc_buffer(norms_bytes, DType::F32, norms_shape)
                    .unwrap();
                for (i, b) in k_packed
                    .as_mut_slice::<u8>()
                    .unwrap()
                    .iter_mut()
                    .enumerate()
                {
                    *b = ((slot * 41 + i * 13) % 251) as u8;
                }
                for (i, b) in v_packed
                    .as_mut_slice::<u8>()
                    .unwrap()
                    .iter_mut()
                    .enumerate()
                {
                    *b = ((slot * 17 + i * 23) % 251) as u8;
                }
                for (i, f) in k_norms
                    .as_mut_slice::<f32>()
                    .unwrap()
                    .iter_mut()
                    .enumerate()
                {
                    *f = (slot as f32) * 0.75 + (i as f32) * 0.25;
                }
                for (i, f) in v_norms
                    .as_mut_slice::<f32>()
                    .unwrap()
                    .iter_mut()
                    .enumerate()
                {
                    *f = (slot as f32) * 0.5 + (i as f32) * 0.125;
                }
                Some(TqKvSnapshot {
                    k_packed,
                    k_norms,
                    v_packed,
                    v_norms,
                    norms_per_pos: 1,
                })
            })
            .collect();
        // MTP in TQ-only mode too.
        let mtp_packed_shape: Vec<usize> = cfg.mtp_shape.iter().map(|d| *d as usize).collect();
        let mtp_norms_shape = vec![
            mtp_packed_shape[0],
            mtp_packed_shape[1],
            mtp_packed_shape[2],
            1,
        ];
        let mtp_packed_bytes = mtp_packed_shape.iter().product();
        let mtp_norms_bytes = mtp_norms_shape.iter().product::<usize>() * 4;
        let mut mk_packed = device
            .alloc_buffer(mtp_packed_bytes, DType::U8, mtp_packed_shape.clone())
            .unwrap();
        let mut mk_norms = device
            .alloc_buffer(mtp_norms_bytes, DType::F32, mtp_norms_shape.clone())
            .unwrap();
        let mut mv_packed = device
            .alloc_buffer(mtp_packed_bytes, DType::U8, mtp_packed_shape)
            .unwrap();
        let mut mv_norms = device
            .alloc_buffer(mtp_norms_bytes, DType::F32, mtp_norms_shape)
            .unwrap();
        for (i, b) in mk_packed
            .as_mut_slice::<u8>()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *b = ((97 + i * 7) % 251) as u8;
        }
        for (i, b) in mv_packed
            .as_mut_slice::<u8>()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *b = ((53 + i * 11) % 251) as u8;
        }
        for (i, f) in mk_norms
            .as_mut_slice::<f32>()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *f = 0.5 + (i as f32) * 0.5;
        }
        for (i, f) in mv_norms
            .as_mut_slice::<f32>()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *f = 0.25 + (i as f32) * 0.25;
        }
        snap.mtp = Some(MtpKvSnapshot {
            k: None,
            v: None,
            current_len: (0..cfg.n_seqs).map(|s| 7 + s).collect(),
            tq: Some(TqKvSnapshot {
                k_packed: mk_packed,
                k_norms: mk_norms,
                v_packed: mv_packed,
                v_norms: mv_norms,
                norms_per_pos: 1,
            }),
        });
        snap
    }

    #[test]
    fn qh35_v5_tq_only_with_mtp_round_trip_byte_exact() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut cfg = synth_cfg_full(2, 0, true, 1);
        cfg.kv_substrate = KvSubstrate::TqOnly;
        let snap = synth_tq_only_snapshot_with_mtp(&device, &cfg);

        let bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize v5 TQ-only");
        // Header: magic + version.
        assert_eq!(&bytes[..4], &QH35_MAGIC);
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(version, 5, "serializer must emit codec v5");

        let restored =
            deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize v5 TQ-only");

        // full-attn: k/v None restored; TQ blobs byte-exact.
        for slot in 0..cfg.n_full_attn as usize {
            assert!(restored.full_attn_k[slot].is_none());
            assert!(restored.full_attn_v[slot].is_none());
            let src = snap.full_attn_tq[slot].as_ref().unwrap();
            let dst = restored.full_attn_tq[slot].as_ref().expect("tq restored");
            assert_eq!(
                dst.k_packed.as_slice::<u8>().unwrap(),
                src.k_packed.as_slice::<u8>().unwrap(),
                "slot {slot} k_packed"
            );
            assert_eq!(
                dst.v_packed.as_slice::<u8>().unwrap(),
                src.v_packed.as_slice::<u8>().unwrap(),
                "slot {slot} v_packed"
            );
            assert_eq!(
                dst.k_norms.as_slice::<u8>().unwrap(),
                src.k_norms.as_slice::<u8>().unwrap(),
                "slot {slot} k_norms"
            );
            assert_eq!(
                dst.v_norms.as_slice::<u8>().unwrap(),
                src.v_norms.as_slice::<u8>().unwrap(),
                "slot {slot} v_norms"
            );
            assert_eq!(dst.norms_per_pos, src.norms_per_pos);
            assert_eq!(
                restored.full_attn_current_len[slot],
                snap.full_attn_current_len[slot]
            );
        }
        // MTP: k/v None restored (v4+ kv_present=0); TQ blob byte-exact.
        let msrc = snap.mtp.as_ref().unwrap();
        let mdst = restored.mtp.as_ref().expect("mtp restored");
        assert!(
            mdst.k.is_none() && mdst.v.is_none(),
            "v5 must restore MTP k/v as None in TQ-only mode"
        );
        let (stq, dtq) = (
            msrc.tq.as_ref().unwrap(),
            mdst.tq.as_ref().expect("mtp tq restored"),
        );
        assert_eq!(
            dtq.k_packed.as_slice::<u8>().unwrap(),
            stq.k_packed.as_slice::<u8>().unwrap()
        );
        assert_eq!(
            dtq.v_packed.as_slice::<u8>().unwrap(),
            stq.v_packed.as_slice::<u8>().unwrap()
        );
        assert_eq!(
            dtq.k_norms.as_slice::<u8>().unwrap(),
            stq.k_norms.as_slice::<u8>().unwrap()
        );
        assert_eq!(
            dtq.v_norms.as_slice::<u8>().unwrap(),
            stq.v_norms.as_slice::<u8>().unwrap()
        );
        assert_eq!(mdst.current_len, msrc.current_len);
    }

    #[test]
    fn qh35_v3_envelope_deserializes_with_implicit_mtp_kv_present() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = synth_cfg_full(1, 0, true, 1);

        // Hand-craft a v3 envelope: 1 full-attn F32 slot (kv_present=1,
        // tq_present=0), no linear slots, MTP F32 present WITHOUT a
        // kv_present byte (v3 had none), MTP tq_present=0.
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&QH35_MAGIC);
        write_u32_le(&mut out, 3); // codec_version = 3
        write_u32_le(&mut out, cfg.n_full_attn);
        write_u32_le(&mut out, cfg.n_linear_attn);
        write_u8(&mut out, 1); // has_mtp
        write_u8(&mut out, FullAttnCodec::F32Dense as u8);
        write_u32_le(&mut out, cfg.n_seqs);
        write_u16_le(&mut out, 0); // reserved
                                   // full-attn slot 0: kv_present=1 + shape + lens + current_len + payload
        let elems: usize = cfg.full_attn_shape.iter().product::<u64>() as usize;
        let byte_len = elems * 4;
        write_u32_le(&mut out, 0); // slot_idx
        write_u8(&mut out, QH35_KV_PRESENT);
        for &d in &cfg.full_attn_shape {
            write_u64_le(&mut out, d);
        }
        write_u64_le(&mut out, byte_len as u64);
        write_u64_le(&mut out, byte_len as u64);
        write_u32_le(&mut out, 5); // current_len[0]
        let k_bytes: Vec<u8> = (0..byte_len).map(|i| (i % 251) as u8).collect();
        let v_bytes: Vec<u8> = (0..byte_len).map(|i| ((i * 3) % 251) as u8).collect();
        out.extend_from_slice(&k_bytes);
        out.extend_from_slice(&v_bytes);
        write_u8(&mut out, QH35_TQ_ABSENT);
        // MTP: NO kv_present byte (v3 layout) — shape + lens + current_len + payload.
        let melems: usize = cfg.mtp_shape.iter().product::<u64>() as usize;
        let mbyte_len = melems * 4;
        for &d in &cfg.mtp_shape {
            write_u64_le(&mut out, d);
        }
        write_u64_le(&mut out, mbyte_len as u64);
        write_u64_le(&mut out, mbyte_len as u64);
        write_u32_le(&mut out, 5);
        let mk_bytes: Vec<u8> = (0..mbyte_len).map(|i| ((i * 7) % 251) as u8).collect();
        out.extend_from_slice(&mk_bytes);
        out.extend_from_slice(&mk_bytes); // v == k for this pin
        write_u8(&mut out, QH35_TQ_ABSENT);

        let restored = deserialize_hybrid_snapshot(&out, &cfg, &device)
            .expect("v3 envelope must still deserialize (implicit MTP kv_present)");
        assert!(
            restored.mtp.as_ref().expect("mtp").k.is_some(),
            "v3 MTP restores as Some"
        );
        assert_eq!(
            restored
                .mtp
                .as_ref()
                .unwrap()
                .k
                .as_ref()
                .unwrap()
                .as_slice::<u8>()
                .unwrap(),
            mk_bytes.as_slice(),
            "v3 MTP k bytes must round-trip"
        );
        assert_eq!(restored.full_attn_current_len[0], vec![5]);
        let fa_k = restored.full_attn_k[0].as_ref().expect("fa k some");
        assert_eq!(fa_k.as_slice::<u8>().unwrap(), k_bytes.as_slice());
    }

    #[test]
    fn qh35_v4_full_capacity_tq_envelope_remains_readable() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut cfg = synth_cfg(1, 1);
        cfg.kv_substrate = KvSubstrate::TqOnly;

        // Hand-craft the final pre-v5 layout: one TQ-only full-attention
        // slot. v4 TQ blobs had no shape field and therefore always used
        // the live cache's full capacity.
        let mut out = Vec::new();
        out.extend_from_slice(&QH35_MAGIC);
        write_u32_le(&mut out, 4);
        write_u32_le(&mut out, 1); // n_full_attn
        write_u32_le(&mut out, 0); // n_linear_attn
        write_u8(&mut out, 0); // has_mtp
        write_u8(&mut out, FullAttnCodec::F32Dense as u8);
        write_u32_le(&mut out, 1); // n_seqs
        write_u16_le(&mut out, 0);
        write_u32_le(&mut out, 0); // slot index
        write_u8(&mut out, QH35_KV_ABSENT);
        write_u32_le(&mut out, 6); // current_len
        write_u8(&mut out, QH35_TQ_PRESENT);
        write_u32_le(&mut out, 1); // norms_per_pos

        let packed_len = cfg.full_attn_shape.iter().product::<u64>() as usize;
        let norms_len =
            (cfg.full_attn_shape[0] * cfg.full_attn_shape[1] * cfg.full_attn_shape[2] * 4) as usize;
        let payloads = [
            (packed_len, 3u8),
            (norms_len, 7u8),
            (packed_len, 11u8),
            (norms_len, 13u8),
        ];
        for (len, seed) in payloads {
            write_u64_le(&mut out, len as u64);
            out.extend((0..len).map(|i| seed.wrapping_add((i % 239) as u8)));
        }

        let restored = deserialize_hybrid_snapshot(&out, &cfg, &device)
            .expect("v4 full-capacity TQ envelope must remain readable");
        let tq = restored.full_attn_tq[0].as_ref().expect("restored tq");
        assert_eq!(tq.k_packed.shape(), &[1, 2, 8, 4]);
        assert_eq!(tq.k_norms.shape(), &[1, 2, 8, 1]);
        assert_eq!(restored.full_attn_current_len[0], vec![6]);
    }

    #[test]
    fn fingerprint_namespaces_by_kv_substrate() {
        let mut a = synth_cfg(1, 1);
        a.kv_substrate = KvSubstrate::F32Only;
        let mut b = a.clone();
        b.kv_substrate = KvSubstrate::TqOnly;
        let ha = crate::serve::kv_persist::families::qwen35_disk_persistor::Qwen35DiskPersistor::fingerprint_hex_for(&a);
        let hb = crate::serve::kv_persist::families::qwen35_disk_persistor::Qwen35DiskPersistor::fingerprint_hex_for(&b);
        assert_ne!(ha, hb, "substrate must namespace the fingerprint (cross-mode hydrate would silently zero-restore)");
        let ha2 = crate::serve::kv_persist::families::qwen35_disk_persistor::Qwen35DiskPersistor::fingerprint_hex_for(&a);
        assert_eq!(ha, ha2, "fingerprint must be deterministic");
    }

    /// ADR-027 sub-iter 23d-γ — the exact production path that 500'd at
    /// the live serve gate: a TQ-only snapshot written to disk
    /// (serialize), hydrated back (deserialize), then LCP-resumed into a
    /// freshly-allocated TQ-mode live cache via
    /// `HybridKvCache::restore_partial`. Pre-23d-γ this chain hard-failed
    /// with `partial_copy_slot: src shape rank 1 != 4` because the
    /// deserializer emitted flat blobs. Pin: prefix bytes byte-exact +
    /// tail zero + current_len advanced.
    #[test]
    fn qh35_v5_hydrated_tq_snapshot_restores_into_live_cache() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let live_cfg = tiny_qwen35_cfg(true);
        let max_seq_len = 64u32;
        let n_tokens = 40usize;

        // 1) Source cache with planted patterns → snapshot (production
        //    producer shape: live TQ-mode cache).
        let mut src = HybridKvCache::new_with_options(&live_cfg, &device, max_seq_len, 1, true)
            .expect("alloc src");
        for (i, slot) in src.full_attn.iter_mut().enumerate() {
            let tq = slot.tq.as_mut().expect("tq");
            for (bi, buf) in [
                &mut tq.k_packed,
                &mut tq.k_norms,
                &mut tq.v_packed,
                &mut tq.v_norms,
            ]
            .into_iter()
            .enumerate()
            {
                let s = buf.as_mut_slice::<u8>().unwrap();
                for (j, b) in s.iter_mut().enumerate() {
                    *b = ((i * 31 + bi * 7 + j) % 251) as u8;
                }
            }
            slot.current_len[0] = n_tokens as u32;
        }
        {
            let tq = src
                .mtp_slot
                .as_mut()
                .expect("mtp")
                .tq
                .as_mut()
                .expect("mtp tq");
            for (bi, buf) in [
                &mut tq.k_packed,
                &mut tq.k_norms,
                &mut tq.v_packed,
                &mut tq.v_norms,
            ]
            .into_iter()
            .enumerate()
            {
                let s = buf.as_mut_slice::<u8>().unwrap();
                for (j, b) in s.iter_mut().enumerate() {
                    *b = ((97 + bi * 11 + j) % 251) as u8;
                }
            }
            src.mtp_slot.as_mut().expect("mtp").current_len[0] = n_tokens as u32;
        }
        for slot in &mut src.full_attn {
            slot.current_len[0] = max_seq_len;
        }
        src.mtp_slot.as_mut().expect("mtp").current_len[0] = max_seq_len;
        let full_snap = src
            .snapshot(&device)
            .expect("fully initialized capacity snapshot");
        for slot in &mut src.full_attn {
            slot.current_len[0] = n_tokens as u32;
        }
        src.mtp_slot.as_mut().expect("mtp").current_len[0] = n_tokens as u32;
        let snap = src
            .snapshot_prefix(&device, n_tokens)
            .expect("compact prefix snapshot");
        let cfg = cfg_from_cache(&src, FullAttnCodec::F32Dense).expect("cfg_from_cache");
        assert_eq!(cfg.kv_substrate, KvSubstrate::TqOnly);

        // 2) Disk round-trip (serialize → deserialize) — the hydrate arm.
        let full_bytes = serialize_hybrid_snapshot(&full_snap, &cfg).expect("serialize full");
        let bytes = serialize_hybrid_snapshot(&snap, &cfg).expect("serialize");
        assert!(
            bytes.len() < full_bytes.len(),
            "compact disk envelope must be smaller (compact={} full={})",
            bytes.len(),
            full_bytes.len()
        );
        let restored_snap =
            deserialize_hybrid_snapshot(&bytes, &cfg, &device).expect("deserialize");
        assert_eq!(
            restored_snap.full_attn_tq[0]
                .as_ref()
                .unwrap()
                .k_packed
                .shape()[2],
            n_tokens,
            "v5 hydrate must retain compact sequence capacity"
        );

        // 3) LCP resume into a fresh TQ-mode cache — THE 500 PATH.
        let mut dst = HybridKvCache::new_with_options(&live_cfg, &device, max_seq_len, 1, true)
            .expect("alloc dst");
        dst.restore_partial(&restored_snap, n_tokens)
            .expect("restore_partial on hydrated snapshot (was the live-gate 500)");

        // 4) Pin: per-head prefix byte-exact on all four TQ buffers,
        //    every full-attn slot + the MTP slot; tail zero; cursor set.
        let check_slot =
            |src_tq: &crate::inference::models::qwen35::kv_cache::TqFullAttnKvBuffers,
             dst_tq: &crate::inference::models::qwen35::kv_cache::TqFullAttnKvBuffers,
             what: &str| {
                for (name, s, d) in [
                    ("k_packed", &src_tq.k_packed, &dst_tq.k_packed),
                    ("k_norms", &src_tq.k_norms, &dst_tq.k_norms),
                    ("v_packed", &src_tq.v_packed, &dst_tq.v_packed),
                    ("v_norms", &src_tq.v_norms, &dst_tq.v_norms),
                ] {
                    let shape = d.shape();
                    let (n_kv, max_seq, inner) = (shape[1], shape[2], shape[3]);
                    let elem = d.dtype().size_of();
                    let head_stride = max_seq * inner * elem;
                    let (sb, db) = (s.as_slice::<u8>().unwrap(), d.as_slice::<u8>().unwrap());
                    for head in 0..n_kv {
                        let off = head * head_stride;
                        let n = n_tokens * inner * elem;
                        assert_eq!(
                            &db[off..off + n],
                            &sb[off..off + n],
                            "{what}.tq.{name}[head {head}] prefix diverged (23d-γ)"
                        );
                        let tail = &db[off + n..off + head_stride];
                        assert!(
                            tail.iter().all(|&b| b == 0),
                            "{what}.tq.{name}[head {head}] tail not zero (23d-γ)"
                        );
                    }
                }
            };
        for (i, slot) in dst.full_attn.iter().enumerate() {
            check_slot(
                src.full_attn[i].tq.as_ref().unwrap(),
                slot.tq.as_ref().expect("dst tq"),
                &format!("full_attn[{i}]"),
            );
            assert_eq!(slot.current_len[0] as usize, n_tokens);
        }
        check_slot(
            src.mtp_slot.as_ref().unwrap().tq.as_ref().unwrap(),
            dst.mtp_slot
                .as_ref()
                .unwrap()
                .tq
                .as_ref()
                .expect("dst mtp tq"),
            "mtp",
        );
        assert_eq!(
            dst.mtp_slot.as_ref().unwrap().current_len[0] as usize,
            n_tokens
        );
    }
}
