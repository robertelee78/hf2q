//! GGUF backend — seek-back writer for the convert-v2 pipeline.
//!
//! Post-ADR-033 P6: this subdir is the surviving GGUF writer surface.
//! The legacy two-pass `backends/gguf.rs` (and the `OutputBackend`
//! trait implementation it carried) was retired alongside the rest
//! of the legacy convert pipeline.

pub mod types;
pub mod writer;
