//! ADR-017 §B-dense / §B-hybrid / §B-tq — per-family `KvCacheSpill`
//! implementations.
//!
//! The Phase A.3 spiller (`crate::serve::kv_persist::spiller`) owns the
//! envelope / chain-hash / atomic-rename / LRU-evict lifecycle and
//! delegates the per-family snapshot/restore byte codec to one
//! `KvCacheSpill` impl per loaded `(repo, quant)` pair. This `families`
//! module is the home for those impls — one file per architecture
//! family so the per-family Chesterton's-fence reasoning stays
//! co-located with the code:
//!
//! - [`gemma4_dense`] (B-dense.1) — Gemma 4's dense F32/F16
//!   K/V cache (`MlxModelWeights.dense_kvs`). Sliding layers use
//!   ring-buffer mode; full-attention layers use a linear buffer.
//! - Qwen 3.5/3.6 hybrid (B-hybrid via Phase E.a B.2-B.5, 2026-05-05) —
//!   Qwen 3.5's interleaved full-attention + DeltaNet recurrent state.
//!   The hybrid family does NOT need a sibling `KvCacheSpill` impl;
//!   ADR-017 Phase E.a Phase B.2 ships the LCP partial-prefill resume
//!   substrate via `Qwen35LoadedModel::lcp_registry` directly (see
//!   `engine_qwen35.rs`), keying full-attn + DeltaNet snapshots under
//!   chunk-position-keyed `LcpKey`s.  Phase D's spiller layer is
//!   side-stepped because hybrid-MoE ring-buffer slot accounting
//!   doesn't fit the `(layer_rank, range)` block contract.
//! - [`tq_packed`] (B-tq.1, this iter 2026-05-05) — TurboQuant-packed
//!   K/V codec.  Provides `payload_kind = "tq_packed_v1"` envelope
//!   serialization at codec_version=1 frozen with deterministic round-
//!   trip + R-C2 cosine ≥ 0.9998 (trivially satisfied by byte-exact
//!   rebuild).  Family hook wire-up (TqPackedSpill, analogous to
//!   Gemma4DenseSpill) is B-tq.2 and lands once the runtime TQ
//!   inference path stabilises (ADR-007 reopen 2026-05-05 Path C).
//!
//! Each family hook is registered with the spiller via
//! `BlockPrefixCacheSpiller::register_family((repo, quant), hook)` at
//! engine load (Phase C.1). The hook is held behind
//! `Arc<Mutex<dyn KvCacheSpill>>` so the spiller's two trigger sites
//! (`pre_evict` reads / `post_admit` writes) share a single
//! registration across the engine's lifetime.

pub mod gemma4_dense;
pub mod tq_packed;
