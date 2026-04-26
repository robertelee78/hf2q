//! Arch-registry scaffolding — ADR-012 Decision 20 (landed within P8).
//!
//! Single source of truth for per-architecture conformance knobs:
//! tensor catalogs, quality thresholds, smoke prompts, MTP/vision flags.
//! Consumed by `hf2q smoke` (Decision 16), the PPL/KL eval helper
//! (Decision 17), the MTP round-trip gate (Decision 19), and the
//! vision-tower emitter (Decision 18).
//!
//! Design rule (mantra: no stubs): only arches that are fully
//! populated ship an entry. `qwen35` and `qwen35moe` are the two
//! entries ADR-012 adds. Gemma4 parity, Ministral (ADR-015), and
//! DeepSeek-V3 (ADR-016) each register in their own ADR when opened.
//!
//! A request for an unregistered arch returns a uniform structured
//! `ArchError::UnknownArch` (no per-arch `todo!()` branches).

pub mod catalog;
pub mod conformance;
pub mod entries;
pub mod registry;
pub mod smoke;

// Public re-export surface — intentional, used by callers outside the arch
// module once P9/P10/P11 land their own source files. rustc warns until
// those callers appear; the re-exports are load-bearing per the ADR.
#[allow(unused_imports)]
pub use catalog::{TensorCatalog, TensorCatalogEntry, TensorDtype};
#[allow(unused_imports)]
pub use registry::{
    ArchEntry, ArchError, ArchRegistry, EvalCorpus, QualityThresholds,
};
