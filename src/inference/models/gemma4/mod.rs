//! Per-architecture Gemma4 helpers.
//!
//! Today: GGUF-embedded tokenizer construction (ADR-022 P1.11).
//! Future: dedicated Gemma4 forward graph if/when the
//! `src/serve/forward_mlx.rs` monolith is split per-arch.

pub mod tokenizer;
