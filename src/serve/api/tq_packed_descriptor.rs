//! ADR-017 §B-tq.4 — `TqPackedSpillDescriptor`: cached shape config for
//! the TQ-active KV-persist hook bridge between `Engine` and the
//! `TqPackedSpillFactory`.
//!
//! ## Why this lives in `serve::api` (sibling to `KvSpillDescriptor`)
//!
//! Same threading contract as `KvSpillDescriptor` at
//! `src/serve/api/kv_spill_descriptor.rs`: the `Engine` worker thread
//! owns the `MlxModelWeights` (with its `kv_caches[i].{k_packed,
//! k_norms, v_packed, v_norms}` and per-layer shape).  The
//! `TqPackedSpill` hook cannot see those internals across the worker-
//! thread boundary.  This descriptor snapshots the immutable TQ-cache
//! shape at spawn time (synchronous, before the move into the worker
//! thread) so the factory can read it without round-tripping through
//! the request channel.
//!
//! ## Scope
//!
//! TQ-active K/V layouts on Gemma 4 dense layers only.  Qwen3.5/3.6
//! hybrid (DeltaNet) is out of B-tq.4 scope per the family-scoping
//! discipline established by B-dense.1 (each family gets its own
//! descriptor + spill + factory).
//!
//! ## Read-only on `MlxModelWeights`
//!
//! `from_gemma_loaded_model_tq` reads `weights.kv_caches[i].{k_packed
//! shape}` for shape derivation and the operator-time
//! `HF2Q_TQ_CODEBOOK_BITS` env for the bit-width.  It does NOT mutate
//! the weights and does NOT touch `forward_mlx.rs` — the only
//! references it keeps are the read-only shape fields.

use crate::serve::api::kv_spill_descriptor::KvSpillProvenance;
use crate::serve::kv_persist::families::tq_packed::{flags, TqBitsPerCoord};

/// Cached TQ-packed KV-cache shape config for one Gemma 4 model.
/// Captured by `Engine::spawn` from the `MlxModelWeights` before the
/// move into the worker thread; read-only thereafter.
///
/// The fields here are sufficient to drive
/// [`crate::serve::kv_persist::families::tq_packed::TqPackedConfig`]
/// construction without round-tripping through the worker channel.
#[derive(Debug, Clone)]
pub struct TqPackedSpillDescriptor {
    /// Lloyd-Max bit width per coordinate.  Reads
    /// `HF2Q_TQ_CODEBOOK_BITS` env at engine spawn (default `8` per
    /// ADR-007 §1234 production default).
    pub bits_per_coord: TqBitsPerCoord,
    /// Number of attention layers — equals
    /// `weights.kv_caches.len()`.
    pub num_layers: usize,
    /// Per-layer KV-head count derived from `kv_caches[i].k_packed`
    /// shape `[nkv, capacity, hd_packed]`.  Length == `num_layers`.
    pub nkv_heads: Vec<u32>,
    /// Per-layer head dim derived from
    /// `(kv_caches[i].k_packed.shape()[2] * 8 / bits_per_coord)`.
    /// Length == `num_layers`.
    pub head_dim: Vec<u32>,
    /// Block alignment in tokens.  Always equals
    /// [`crate::serve::kv_persist::block_store::BLOCK_TOKENS`] (256).
    pub block_tokens: u32,
    /// Header `flags` field passed verbatim into every `tq_packed_v2`
    /// envelope.  Production runtime always sets
    /// [`flags::HADAMARD_ROTATED`] because
    /// `dispatch_hadamard_quantize_kv` applies FWHT before
    /// quantization.
    pub flags: u32,
    /// Per-block `scale: f64` field passed verbatim into envelopes.
    /// On the per-token-norm v2 path the magnitude lives in the
    /// per-token-per-head norms stream, so the production default is
    /// `1.0`.
    pub scale: f64,
    /// ADR-017 §F4 — provenance bits used to derive the
    /// per-`(repo, quant)` `ModelFingerprint` namespace key.  Same
    /// shape and semantics as `KvSpillDescriptor.provenance`.
    pub provenance: KvSpillProvenance,
}

impl TqPackedSpillDescriptor {
    /// Construct a descriptor from a Gemma 4 `MlxModelWeights` snapshot
    /// when the runtime is in TQ-active mode (`HF2Q_TQ_KV=1`).  Returns
    /// `None` if any layer's `k_packed` buffer doesn't expose a 3-D
    /// `[nkv, capacity, hd_packed]` shape (defensive — current Gemma 4
    /// runtime always shapes the buffer this way at
    /// `forward_mlx.rs:1285+1679`).
    ///
    /// `bits_per_coord` is read from the operator-time
    /// `HF2Q_TQ_CODEBOOK_BITS` env at spawn time (default `8`).  The
    /// descriptor is captured ONCE at spawn (not per-request) so
    /// toggling the env mid-process doesn't re-shape the descriptor.
    /// This matches `KvSpillDescriptor`'s `kv_dtype` capture pattern.
    ///
    /// `provenance` is captured by the caller from
    /// `serve::provenance::detect(&gguf)` and the GGUF-embedded
    /// `tokenizer.chat_template` (same source as
    /// `KvSpillDescriptor.provenance`).  ADR-017 §F4 — drives the
    /// `ModelFingerprint` namespace key.
    pub fn from_gemma_loaded_model_tq(
        weights: &crate::serve::forward_mlx::MlxModelWeights,
        provenance: KvSpillProvenance,
    ) -> Option<Self> {
        let bits = read_tq_codebook_bits_env();
        let bits_per_coord = TqBitsPerCoord::new(bits).ok()?;

        // **B-tq.7** — at bits >= 5 the runtime stores K/V in
        // `leg_hb_encoded[layer]` (1 byte per coord, shape
        // `[nkv, capacity, head_dim]` per `HbKvBuffers` doc); at
        // bits == 4 it stores in the legacy `kv_caches[layer].k_packed`
        // (nibble-packed, shape `[nkv, capacity, head_dim/2]`).
        //
        // **Note**: `leg_hb_encoded` is lazily allocated at FIRST
        // `forward_decode` (warmup), so at `Engine::spawn` time it's
        // `None`.  We can't read its shape directly here.  Instead,
        // derive `head_dim` and `n_kv_heads` from the per-layer model
        // config (`weights.layers[i].{head_dim, num_kv_heads}`) which
        // is stable from GGUF-load time.  At restore-time the
        // snapshot/restore code reads the actual buffer (which is
        // populated by then).
        let num_layers = weights.kv_caches.len();
        if num_layers == 0 || weights.layers.len() != num_layers {
            return None;
        }

        let mut nkv_heads = Vec::with_capacity(num_layers);
        let mut head_dim = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let layer_cfg = &weights.layers[layer_idx];
            nkv_heads.push(layer_cfg.num_kv_heads as u32);
            head_dim.push(layer_cfg.head_dim as u32);
        }

        Some(Self {
            bits_per_coord,
            num_layers,
            nkv_heads,
            head_dim,
            block_tokens: crate::serve::kv_persist::format::BLOCK_TOKENS,
            flags: flags::HADAMARD_ROTATED,
            scale: 1.0_f64,
            provenance,
        })
    }
}

/// Pure parsing logic for `HF2Q_TQ_CODEBOOK_BITS`: takes the env
/// value (as `Option<&str>`) and returns the validated bit width, or
/// the default `8` per ADR-007 §1234.  Pure → unit-testable without
/// env contention.
pub(crate) fn parse_tq_codebook_bits(env: Option<&str>) -> u32 {
    const DEFAULT_BITS: u32 = 8;
    match env {
        Some(v) => match v.trim().parse::<u32>() {
            Ok(n) if matches!(n, 2 | 3 | 4 | 5 | 6 | 8) => n,
            _ => DEFAULT_BITS,
        },
        None => DEFAULT_BITS,
    }
}

/// Read `HF2Q_TQ_CODEBOOK_BITS` env (default `8` per ADR-007 §1234
/// production default).  Validates the value is one of {2, 3, 4, 5, 6,
/// 8}; on parse failure or out-of-range, falls back to `8`.
///
/// Thin wrapper around [`parse_tq_codebook_bits`] for production use.
pub(crate) fn read_tq_codebook_bits_env() -> u32 {
    parse_tq_codebook_bits(std::env::var("HF2Q_TQ_CODEBOOK_BITS").ok().as_deref())
}

/// Pure parsing logic for `HF2Q_TQ_KV`: takes the env value (as
/// `Option<&str>`) and returns whether TQ-active mode is enabled.
/// Pure → unit-testable without env contention.
pub fn parse_tq_active_mode(env: Option<&str>) -> bool {
    match env {
        Some(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        None => false,
    }
}

/// Returns `true` iff the operator has enabled TQ-active KV mode via
/// `HF2Q_TQ_KV=1`.  Engine spawn uses this to decide whether to emit a
/// `TqPackedSpillDescriptor` alongside the dense `KvSpillDescriptor`.
///
/// Empty / `0` / unset → `false`.  Any non-`0` non-empty value → `true`.
/// Matches the `kv-persist` startup-disable predicate's permissive
/// truthiness convention.
pub fn is_tq_active_mode() -> bool {
    parse_tq_active_mode(std::env::var("HF2Q_TQ_KV").ok().as_deref())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_tq_codebook_bits` defaults to 8 when env unset.
    /// Documents the ADR-007 §1234 production default.  Pure-function
    /// test — no env contention.
    #[test]
    fn parse_tq_codebook_bits_defaults_to_8_when_unset() {
        assert_eq!(parse_tq_codebook_bits(None), 8);
    }

    /// Valid bit widths {2, 3, 4, 5, 6, 8} round-trip through the
    /// parser.
    #[test]
    fn parse_tq_codebook_bits_accepts_documented_widths() {
        for bits in [2u32, 3, 4, 5, 6, 8] {
            let s = bits.to_string();
            assert_eq!(
                parse_tq_codebook_bits(Some(s.as_str())),
                bits,
                "bits={}",
                bits
            );
        }
    }

    /// Invalid widths fall back to 8.  Falsifier: returning the parsed
    /// out-of-range value would silently mis-shape the descriptor.
    #[test]
    fn parse_tq_codebook_bits_invalid_falls_back_to_default() {
        for bad in ["", "0", "1", "7", "9", "16", "garbage", "  "] {
            assert_eq!(
                parse_tq_codebook_bits(Some(bad)),
                8,
                "bad={:?}",
                bad
            );
        }
    }

    /// `parse_tq_active_mode` honors permissive truthiness — pure-fn
    /// test, no env contention.
    #[test]
    fn parse_tq_active_mode_permissive_truthiness() {
        // Unset / empty / 0 → false.
        assert!(!parse_tq_active_mode(None));
        for falsy in ["", "0", "  0  ", "  "] {
            assert!(!parse_tq_active_mode(Some(falsy)), "falsy={:?}", falsy);
        }
        // Any non-zero non-empty trimmed → true.
        for truthy in ["1", "yes", "true", "on", "anything"] {
            assert!(parse_tq_active_mode(Some(truthy)), "truthy={:?}", truthy);
        }
    }
}
