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
pub mod format;
pub mod index;
pub mod recovery;
pub mod spiller;
pub mod writer;

#[allow(unused_imports)]
pub use block_store::{DiskBlockStore, WriteJob, MAX_BLOCK_BYTES};
#[allow(unused_imports)]
pub use recovery::{
    quarantine_corrupted_block, recover_from_disk, QuarantineReason, RecoveryReport,
};
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
