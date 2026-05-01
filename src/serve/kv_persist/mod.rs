//! ADR-017 §A.1 — persistent block prefix cache, format + index module.
//!
//! This module ships the lowest layer of the ADR-017 persistence stack:
//!
//!   * [`format`] — on-disk envelope (byte-compatible with oMLX
//!     `paged_ssd_cache.py:246-297`), chain-hash identity per ADR-017
//!     §D4, and the `EnvelopeHeader` JSON schema (ADR-017 §D10).
//!   * [`index`] — in-memory `HashMap<BlockHash, BlockMeta>` with
//!     restart-recovery scan (ADR-017 §D8) and quarantine of corrupted
//!     files (ADR-017 §R-F9).
//!
//! Phase A.2 lands `block_store` + `writer` + `recovery` on top of these
//! primitives; Phase A.3 lands the `BlockPrefixCacheSpiller<E>` impl
//! that wires `KvSpiller<E>` (ADR-005 Phase 4 iter-212) into the
//! HotSwapManager.

pub mod block_store;
pub mod cache_ops;
pub mod families;
pub mod format;
pub mod index;
pub mod loader_wrapper;
pub mod recovery;
pub mod registry;
pub mod spiller;
pub mod writer;

#[allow(unused_imports)]
pub use block_store::{DiskBlockStore, WriteJob, MAX_BLOCK_BYTES};
#[allow(unused_imports)]
pub use loader_wrapper::LoaderWrapper;
#[allow(unused_imports)]
pub use recovery::{
    quarantine_corrupted_block, recover_from_disk, QuarantineReason, RecoveryReport,
};
#[allow(unused_imports)]
pub use registry::KvPersistRegistry;
#[allow(unused_imports)]
pub use spiller::{BlockPrefixCacheSpiller, KvCacheSpill, StubGemma4Spill};
#[allow(unused_imports)]
pub use writer::{AsyncWriterHandle, DEFAULT_CHANNEL_CAPACITY};
#[allow(unused_imports)]
pub use format::{
    compute_block_hash, compute_model_fingerprint, read_envelope_body, read_envelope_header,
    write_envelope, BlockHash, CacheFormatVersion, EnvelopeHeader, ModelFingerprint,
    ParentBlockHash, BLOCK_TOKENS, CURRENT_FORMAT_VERSION,
};
#[allow(unused_imports)]
pub use index::{BlockIndex, BlockMeta};

// ---------------------------------------------------------------------------
// EngineBindable — Phase C.1 additive trait surface for binding the live
// engine reference into per-family `KvCacheSpill` hooks BEFORE the
// `KvSpiller::post_admit` trigger fires.
//
// ## Why a separate trait (not a method on `KvCacheSpill`)
//
// `KvCacheSpill` is the per-family payload codec — its surface (`block_alignment`,
// `snapshot_block`, `restore_block`) is stable per the iter-212 ship contract +
// the Phase A.3 wiring discipline. Adding `bind_engine` directly to
// `KvCacheSpill` would force every existing impl (including the Phase A.3
// `StubGemma4Spill` and any test mocks) to handle a concept they don't need.
//
// Splitting the engine-binding seam into its own trait keeps `KvCacheSpill`
// payload-only and lets stateless hooks (e.g. `StubGemma4Spill`,
// `MockKvCacheSpill`) impl it as a no-op. Hooks that DO need engine state
// (Gemma 4 dense, future Qwen 3.5 hybrid) impl it with the real downcast.
//
// ## Why `Arc<dyn Any + Send + Sync>`
//
// Phase A.3's `BlockPrefixCacheSpiller<E>` is generic over the engine type
// `E` so the trait surface stays engine-agnostic. The Phase C.1
// `LoaderWrapper<E>` is also generic over `E`. Neither knows about
// concrete types like `MlxModelWeights` or `EngineHandle`. Type-erasing
// the engine ref through `Arc<dyn Any>` lets the wrapper deliver the
// freshly-loaded engine to the hook without the wrapper depending on the
// hook's concrete state shape — the hook performs the downcast itself
// (`Arc::downcast`) and silently no-ops on type mismatch.
// ---------------------------------------------------------------------------

/// Per-family hook extension for binding the live engine reference at
/// load time. Phase C.1 calls
/// [`Self::bind_engine`] from inside `LoaderWrapper::load` BEFORE
/// `KvSpiller::post_admit` fires, and [`Self::unbind_engine`] from
/// the manager's evict path so stale handles don't outlive the
/// engine.
///
/// **Send + Sync** because the `KvPersistRegistry` holds the hook
/// behind `Arc<dyn EngineBindable>` and the bind/unbind trigger sites
/// may run from concurrent tokio tasks (per the
/// `HotSwapManager<E>` concurrency model in `multi_model.rs:887-892`).
///
/// **Object safety**: both methods take `&self` (no generic params,
/// no Self return), so `Arc<dyn EngineBindable>` is well-formed.
///
/// **Failure handling**: hooks must NEVER panic on type mismatch. The
/// canonical impl uses `Arc::downcast::<ConcreteHandle>()` which
/// returns `Result<Arc<ConcreteHandle>, Arc<dyn Any>>`; the `Err`
/// branch must silently discard the engine ref and return without
/// further state mutation.
pub trait EngineBindable: Send + Sync {
    /// Bind a live engine reference to the hook. The hook is
    /// responsible for downcasting `engine_dyn` to its expected
    /// concrete type and silently no-opping on mismatch.
    ///
    /// Called by [`KvPersistRegistry::bind_for`] from inside
    /// `LoaderWrapper::load` after a successful `loader.load(...)`.
    /// Multiple `bind_engine` calls overwrite the prior binding (the
    /// freshest load wins — matches the registry's overwrite
    /// semantic).
    fn bind_engine(&self, engine_dyn: std::sync::Arc<dyn std::any::Any + Send + Sync>);

    /// Drop the live engine reference. Called by
    /// [`KvPersistRegistry::unbind_for`] when the manager evicts the
    /// engine. Subsequent `KvCacheSpill::snapshot_block` /
    /// `restore_block` calls on the underlying hook return
    /// `None` / `Skipped` (the per-family contract is "no engine
    /// handle ⇒ no work").
    fn unbind_engine(&self);
}
