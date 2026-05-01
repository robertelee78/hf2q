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
//! - [`gemma4_dense`] (B-dense.1, this iter) — Gemma 4's dense F32/F16
//!   K/V cache (`MlxModelWeights.dense_kvs`). Sliding layers use
//!   ring-buffer mode; full-attention layers use a linear buffer.
//! - `qwen35_hybrid` (B-hybrid.1, future) — Qwen 3.5's interleaved full-
//!   attention + DeltaNet recurrent state. Out of scope here.
//! - `tq_packed` (B-tq.1, future) — TurboQuant-packed K/V codec. Out of
//!   scope here.
//!
//! Each family hook is registered with the spiller via
//! `BlockPrefixCacheSpiller::register_family((repo, quant), hook)` at
//! engine load (Phase C.1). The hook is held behind
//! `Arc<Mutex<dyn KvCacheSpill>>` so the spiller's two trigger sites
//! (`pre_evict` reads / `post_admit` writes) share a single
//! registration across the engine's lifetime.

pub mod gemma4_dense;
