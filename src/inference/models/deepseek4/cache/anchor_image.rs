//! ADR-062 D4 — byte codec for the DeepSeek-V4 recovery-anchor prefix
//! image (one image per conversation, persisted by the serve-side
//! `kv_persist::families::deepseek4_anchor` disk persistor).
//!
//! ## What the image carries (designed from the actual resume path)
//!
//! The in-memory recovery anchor (`Deepseek4CacheSnapshot` + its exact
//! rendered token ledger) only owns the state that decode after the
//! anchor boundary can overwrite: the 128-row circular window and the
//! recurrent compressor pools. The compressed/indexer KV rows are
//! append-only, so the in-memory anchor relies on the LIVE cache still
//! holding the rows below the anchor position. A restart drops those
//! rows, so the disk image must carry, per layer:
//!
//! - the anchor snapshot's circular-window rows (valid rows at the
//!   anchor position),
//! - the anchor snapshot's recurrent compressor state buffers
//!   (`main_kv_state` / `main_score_state` / indexer twins), and
//! - the append-only `compressed_kv` / `indexer_kv` rows valid at the
//!   anchor position, read from the live cache at commit time.
//!
//! Hydration rebuilds a logical cache at the image's plan, writes those
//! rows back, sets the logical position, and captures a fresh
//! `Deepseek4CacheSnapshot` from the rebuilt cache — the exact state
//! the in-memory RecoveryAnchor resume path would have held.
//!
//! ## Compatibility + corruption
//!
//! The file embeds the anchor position, token ledger, layer count,
//! per-layer compression schedule, and the plan's context length; a
//! SHA-256 checksum trails the file. Hydration re-derives the plan from
//! the live model config, validates every serialized byte count against
//! it, and refuses on any mismatch — a missing, corrupt, or
//! incompatible image therefore only causes a cold prefill at the
//! engine layer (ADR-062 D4).
//!
//! Graft discipline (ADR-059 gate 5): a bound `--kv-graft` is refused
//! at boot when `--kv-persist` is set, so images are only written with
//! `graft_slots == 0`. Both directions re-check: serialization refuses
//! a grafted cache and hydration refuses an image that records graft
//! rows.

mod serialize;

use mlx_native::MlxDevice;
use sha2::{Digest, Sha256};

use super::{
    CacheError, CacheKind, Deepseek4Cache, Deepseek4CachePlan, Deepseek4CacheSnapshot,
};
use crate::inference::models::deepseek4::Deepseek4Config;

/// File magic — `DeepSeek-V4 anchor image, codec 1`.
const ANCHOR_IMAGE_MAGIC: [u8; 8] = *b"DSK4AIM1";
/// Fixed header length in bytes (see the write/read helpers below).
const ANCHOR_IMAGE_HEADER_LEN: usize = 40;
/// Trailing SHA-256 checksum length in bytes.
const ANCHOR_IMAGE_CHECKSUM_LEN: usize = 32;
/// Current codec version.
const ANCHOR_IMAGE_CODEC_VERSION: u32 = 1;
/// BF16 row storage size in bytes (window / compressed / indexer rows).
const BF16_SIZE: usize = 2;

/// Successfully hydrated recovery-anchor state.
pub struct AnchorImageHydration {
    /// Live cache rebuilt at the image's plan with the image's rows and
    /// logical position.
    pub cache: Deepseek4Cache,
    /// Fresh recovery-anchor snapshot captured from `cache`.
    pub snapshot: Deepseek4CacheSnapshot,
    /// Exact rendered token ledger for the anchor position.
    pub tokens: Vec<u32>,
}

/// Header facts for a persisted image (used by the serve-side startup
/// scan to build the hydrate index without reading whole files).
pub struct AnchorImageHeader {
    pub anchor_position: usize,
    pub token_count: usize,
    pub layer_count: usize,
    pub context_length: usize,
    pub graft_slots: usize,
}

impl AnchorImageHeader {
    /// Parse the fixed 40-byte header (plus the token ledger when the
    /// input carries it — `None` otherwise, which the serve-side
    /// startup scan uses to index a file without reading its payload).
    /// Validates magic, version, graft discipline, and the
    /// anchor/token invariant.
    pub fn parse_prefix(bytes: &[u8]) -> Result<(Self, Option<Vec<u32>>), CacheError> {
        let corrupt = |reason: &'static str| CacheError::AnchorImageCorrupt { reason };
        if bytes.len() < ANCHOR_IMAGE_HEADER_LEN {
            return Err(corrupt("truncated header"));
        }
        if bytes[..8] != ANCHOR_IMAGE_MAGIC {
            return Err(corrupt("bad magic"));
        }
        let codec_version = read_u32(bytes, 8);
        if codec_version != ANCHOR_IMAGE_CODEC_VERSION {
            return Err(CacheError::AnchorImageIncompatible {
                reason: "codec version mismatch",
            });
        }
        let anchor_position = read_u32(bytes, 12) as usize;
        let token_count = read_u32(bytes, 16) as usize;
        let layer_count = read_u32(bytes, 20) as usize;
        let context_length = read_u32(bytes, 24) as usize;
        let graft_slots = read_u32(bytes, 28) as usize;
        if graft_slots != 0 {
            return Err(CacheError::AnchorImageIncompatible {
                reason: "image records graft rows (graft + persist refused at boot)",
            });
        }
        if anchor_position == 0 || anchor_position != token_count {
            return Err(corrupt("anchor position disagrees with token count"));
        }
        if layer_count == 0 || context_length == 0 {
            return Err(corrupt("zero layer count or context length"));
        }
        let token_bytes = token_count
            .checked_mul(4)
            .ok_or(corrupt("token ledger overflow"))?;
        let expected = ANCHOR_IMAGE_HEADER_LEN
            .checked_add(token_bytes)
            .ok_or(corrupt("token ledger overflow"))?;
        let tokens = if bytes.len() >= expected {
            let mut tokens = Vec::with_capacity(token_count);
            for index in 0..token_count {
                tokens.push(read_u32(bytes, ANCHOR_IMAGE_HEADER_LEN + index * 4));
            }
            Some(tokens)
        } else {
            None
        };
        Ok((
            AnchorImageHeader {
                anchor_position,
                token_count,
                layer_count,
                context_length,
                graft_slots,
            },
            tokens,
        ))
    }
}

impl Deepseek4Cache {
    /// Serialize the exact recovery-anchor prefix image for `anchor`
    /// (captured from this cache) with its token ledger. The anchor's
    /// window rows and recurrent state come from the snapshot; the
    /// append-only compressed/indexer rows valid at the anchor position
    /// are read from this live cache (they are never rewritten below
    /// the anchor, so they are exact at commit time).
    pub fn serialize_anchor_image(
        &self,
        anchor: &Deepseek4CacheSnapshot,
        tokens: &[u32],
    ) -> Result<Vec<u8>, CacheError> {
        if tokens.is_empty() || anchor.position() != tokens.len() {
            return Err(CacheError::AnchorImageIncompatible {
                reason: "anchor position disagrees with token ledger",
            });
        }
        if anchor.plan != self.plan {
            return Err(CacheError::SnapshotPlanMismatch);
        }
        if self.graft_slots != 0 {
            // ADR-059 gate 5: graft + persist refuse at boot, so this is
            // unreachable in production; refuse anyway rather than write
            // an image whose graft rows cannot be re-spliced at hydrate.
            return Err(CacheError::AnchorImageIncompatible {
                reason: "cache carries bound graft rows",
            });
        }
        serialize::serialize(self, anchor, tokens)
    }

    /// Rebuild `(cache, snapshot, tokens)` from a persisted anchor-image
    /// file. The plan is re-derived from `cfg` at the image's context
    /// length and every serialized byte count is validated against it;
    /// the trailing SHA-256 must match. Any corrupt or incompatible
    /// input is an error the caller maps to a cold prefill.
    pub fn hydrate_anchor_image(
        bytes: &[u8],
        cfg: &Deepseek4Config,
        device: &MlxDevice,
    ) -> Result<AnchorImageHydration, CacheError> {
        let corrupt = |reason: &'static str| CacheError::AnchorImageCorrupt { reason };
        if bytes.len() < ANCHOR_IMAGE_HEADER_LEN + ANCHOR_IMAGE_CHECKSUM_LEN {
            return Err(corrupt("truncated image"));
        }
        let body_end = bytes.len() - ANCHOR_IMAGE_CHECKSUM_LEN;
        let expected_checksum: [u8; 32] = Sha256::digest(&bytes[..body_end]).into();
        if expected_checksum != bytes[body_end..] {
            return Err(corrupt("checksum mismatch"));
        }
        // The header parse re-validates magic, version, graft, and the
        // anchor/token invariant; the full file must carry the ledger.
        let (header, tokens) = AnchorImageHeader::parse_prefix(bytes)?;
        let tokens = tokens.ok_or_else(|| corrupt("truncated token ledger"))?;
        let plan = Deepseek4CachePlan::for_context(cfg, header.context_length)?;
        if header.layer_count != plan.layers.len() {
            return Err(CacheError::AnchorImageIncompatible {
                reason: "layer count disagrees with live model config",
            });
        }
        let mut cache = Deepseek4Cache::allocate_logical(&plan, device.clone())?;
        let mut cursor = ANCHOR_IMAGE_HEADER_LEN + tokens.len() * 4;
        for (layer_index, layer_plan) in plan.layers.iter().enumerate() {
            let incompatible = |reason: &'static str| CacheError::AnchorImageIncompatible { reason };
            if body_end.saturating_sub(cursor) < 8 * 8 {
                return Err(corrupt("truncated layer section header"));
            }
            let layer_index_field = read_u32(bytes, cursor) as usize;
            let ratio_field = read_u32(bytes, cursor + 4);
            let window_bytes = read_u64(bytes, cursor + 8) as usize;
            let compressed_bytes = read_u64(bytes, cursor + 16) as usize;
            let indexer_bytes = read_u64(bytes, cursor + 24) as usize;
            let main_kv_state_bytes = read_u64(bytes, cursor + 32) as usize;
            let main_score_state_bytes = read_u64(bytes, cursor + 40) as usize;
            let indexer_kv_state_bytes = read_u64(bytes, cursor + 48) as usize;
            let indexer_score_state_bytes = read_u64(bytes, cursor + 56) as usize;
            cursor += 64;
            if layer_index_field != layer_plan.layer_index || ratio_field != layer_plan.compress_ratio
            {
                return Err(incompatible("layer schedule disagrees with live model config"));
            }
            let position = header.anchor_position;
            let window_rows = position.min(layer_plan.window_kv.shape[0]);
            if window_bytes != row_bytes(layer_plan.window_kv.shape[1], window_rows)? {
                return Err(incompatible("window row count disagrees with plan"));
            }
            let expected_compressed = if layer_plan.compress_ratio == 0 {
                0
            } else {
                row_bytes(
                    layer_plan
                        .compressed_kv
                        .as_ref()
                        .map_or(0, |plan| plan.shape[1]),
                    position / layer_plan.compress_ratio as usize,
                )?
            };
            if compressed_bytes != expected_compressed {
                return Err(incompatible("compressed row count disagrees with plan"));
            }
            let expected_indexer = if layer_plan.compress_ratio == 4 {
                row_bytes(
                    layer_plan.indexer_kv.as_ref().map_or(0, |plan| plan.shape[1]),
                    position / 4,
                )?
            } else {
                0
            };
            if indexer_bytes != expected_indexer {
                return Err(incompatible("indexer row count disagrees with plan"));
            }
            for (actual, expected, kind) in [
                (main_kv_state_bytes, layer_plan.main_kv_state.as_ref().map_or(0, |p| p.bytes as usize), CacheKind::MainKvState),
                (main_score_state_bytes, layer_plan.main_score_state.as_ref().map_or(0, |p| p.bytes as usize), CacheKind::MainScoreState),
                (indexer_kv_state_bytes, layer_plan.indexer_kv_state.as_ref().map_or(0, |p| p.bytes as usize), CacheKind::IndexerKvState),
                (indexer_score_state_bytes, layer_plan.indexer_score_state.as_ref().map_or(0, |p| p.bytes as usize), CacheKind::IndexerScoreState),
            ] {
                if actual != expected {
                    return Err(CacheError::SnapshotBufferMismatch { layer: layer_index, kind });
                }
            }
            let layer = &mut cache.layers[layer_index];
            let payload = |cursor: &mut usize, len: usize| -> Result<&[u8], CacheError> {
                if len == 0 {
                    return Ok(&[]);
                }
                let end = cursor
                    .checked_add(len)
                    .filter(|end| *end <= body_end)
                    .ok_or(corrupt("layer payload overruns image"))?;
                let slice = &bytes[*cursor..end];
                *cursor = end;
                Ok(slice)
            };
            let window_payload = payload(&mut cursor, window_bytes)?;
            if window_bytes > 0 {
                let destination = layer
                    .window_kv
                    .as_mut_slice::<u8>()
                    .map_err(|source| CacheError::SnapshotCopy { layer: layer_index, kind: CacheKind::WindowKv, source })?;
                destination[..window_bytes].copy_from_slice(window_payload);
            }
            let compressed_payload = payload(&mut cursor, compressed_bytes)?;
            if compressed_bytes > 0 {
                let destination = layer
                    .compressed_kv
                    .as_mut()
                    .expect("compressed presence validated against plan")
                    .as_mut_slice::<u8>()
                    .map_err(|source| CacheError::SnapshotCopy { layer: layer_index, kind: CacheKind::CompressedKv, source })?;
                destination[..compressed_bytes].copy_from_slice(compressed_payload);
            }
            let indexer_payload = payload(&mut cursor, indexer_bytes)?;
            if indexer_bytes > 0 {
                let destination = layer
                    .indexer_kv
                    .as_mut()
                    .expect("indexer presence validated against plan")
                    .as_mut_slice::<u8>()
                    .map_err(|source| CacheError::SnapshotCopy { layer: layer_index, kind: CacheKind::IndexerKv, source })?;
                destination[..indexer_bytes].copy_from_slice(indexer_payload);
            }
            for (len, buffer, kind) in [
                (main_kv_state_bytes, layer.main_kv_state.as_mut(), CacheKind::MainKvState),
                (main_score_state_bytes, layer.main_score_state.as_mut(), CacheKind::MainScoreState),
                (indexer_kv_state_bytes, layer.indexer_kv_state.as_mut(), CacheKind::IndexerKvState),
                (indexer_score_state_bytes, layer.indexer_score_state.as_mut(), CacheKind::IndexerScoreState),
            ] {
                let state_payload = payload(&mut cursor, len)?;
                if len == 0 {
                    continue;
                }
                let destination = buffer
                    .expect("state presence validated against plan")
                    .as_mut_slice::<u8>()
                    .map_err(|source| CacheError::SnapshotCopy { layer: layer_index, kind, source })?;
                destination.copy_from_slice(state_payload);
            }
        }
        if cursor != body_end {
            return Err(corrupt("trailing bytes after layer sections"));
        }
        cache.next_position = header.anchor_position;
        cache.poisoned = false;
        let snapshot = cache.snapshot()?;
        Ok(AnchorImageHydration {
            cache,
            snapshot,
            tokens,
        })
    }
}

/// Row-payload byte count for `rows` rows of `row_width` elements stored
/// as BF16.
fn row_bytes(row_width: usize, rows: usize) -> Result<usize, CacheError> {
    rows.checked_mul(row_width)
        .and_then(|elements| elements.checked_mul(BF16_SIZE))
        .ok_or(CacheError::ByteOverflow {
            layer: 0,
            kind: CacheKind::AttentionKv,
        })
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("header offset bounds validated by caller"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("header offset bounds validated by caller"),
    )
}

