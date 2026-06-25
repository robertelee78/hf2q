//! Per-architecture Gemma 4 inference support.
//!
//! Owned by ADR-008 (mlx-native forward path) and ADR-017 (TQ-packed KV
//! persistence). Variant detection in `serve::api::engine::LoadedModel::load`.
//!
//! Migration history: ADR-038 split this from monolithic `src/serve/forward_mlx.rs`.
//!
//! # Module layout
//!
//! * `model.rs` — `MlxModelWeights` struct, GGUF loader, DWQ overlay, embed_tokens.
//! * `kv_cache.rs` — KV-buffer structs + `DecodeRegime` (Step 2).
//! * `gpu_full_attn.rs` — per-layer attention dispatch (`encode_one_layer`).
//! * `gpu_ffn.rs` — Path A stub; Path B follow-up will extract from `encode_one_layer`.
//! * `forward_gpu.rs` — outer `forward_decode` + ADR-031 parallel worker.
//! * `io_heads.rs` — argmax/logits/NLL surface.
//! * `kv_persist.rs` — ADR-017 TQ snapshot/restore.
//! * `profile.rs` — kernel/token profile + accumulator.
//! * `tokenizer.rs` — GGUF-embedded tokenizer (unchanged).

pub mod model;
pub mod kv_cache;
pub mod gpu_full_attn;
pub mod gpu_ffn;
pub mod forward_gpu;
pub mod batched_head;
pub mod batched_body;
pub mod io_heads;
pub mod kv_persist;
pub mod profile;
pub mod tokenizer;

// Re-exports collapse import-site churn for the most-touched types.
pub use model::MlxModelWeights;
pub use profile::{ProfileAccumulator, TokenProfile, KernelTypeProfile};
pub use kv_cache::{DenseKvBuffers, HbKvBuffers, HybridKvBuffers, MlxKvCache, DecodeRegime};
