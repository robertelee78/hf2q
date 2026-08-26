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

pub mod batched_body;
pub mod batched_head;
pub mod expert_dispatch;
pub mod forward_gpu;
pub mod gpu_ffn;
pub mod gpu_full_attn;
pub mod io_heads;
pub mod kv_cache;
pub mod kv_persist;
pub mod model;
pub mod native_matrix;
pub mod profile;
pub(crate) mod rectangular_prefill;
pub mod tokenizer;

/// Physical widths emitted by Gemma's current slot-aware scheduler and its
/// bounded prompt transaction. Activation and preflight consume this single
/// source so a scheduler-width change cannot silently lose calibrated routes.
pub(crate) const GEMMA4_MAX_PHYSICAL_DECODE_WIDTH: u32 = 8;
pub(crate) const GEMMA4_SLOT_PREFILL_CHUNK_TOKENS: u32 = 4_096;

pub(crate) fn native_expert_activation_widths() -> Vec<u32> {
    (1..=GEMMA4_MAX_PHYSICAL_DECODE_WIDTH)
        .chain([GEMMA4_SLOT_PREFILL_CHUNK_TOKENS])
        .collect()
}

#[cfg(test)]
mod native_route_width_tests {
    #[test]
    fn scalar_expert_widths_cover_scheduler_decode_and_prompt_boundaries() {
        let widths = super::native_expert_activation_widths();
        assert_eq!(&widths[..8], &(1..=8).collect::<Vec<_>>());
        assert_eq!(
            widths.last(),
            Some(&super::GEMMA4_SLOT_PREFILL_CHUNK_TOKENS)
        );
    }
}

// Re-exports collapse import-site churn for the most-touched types.
pub use kv_cache::{
    DecodeRegime, DenseKvBuffers, GemmaLcpLayerKv, HbKvBuffers, HybridKvBuffers, MlxKvCache,
};
pub use model::{MlxModelWeights, MultiSeqPrefillOutput};
pub use profile::{KernelTypeProfile, ProfileAccumulator, TokenProfile};
