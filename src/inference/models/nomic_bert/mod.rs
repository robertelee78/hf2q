//! nomic-bert embedding-model architecture (ADR-005 Phase 2b, Task #16).
//!
//! Variant of the BERT family that diverges in three load-bearing places:
//!
//! 1. **Position encoding**: RoPE (NeoX convention) applied to Q/K inside
//!    each attention block. No `position_embd.weight` table.
//! 2. **MLP**: SwiGLU (parallel) — `silu(ffn_up(x)) * ffn_gate(x)` →
//!    `ffn_down`. Plain BERT uses GeLU on `ffn_up`.
//! 3. **Tensor manifest**: fused `blk.{i}.attn_qkv.weight` instead of the
//!    separate `attn_q/k/v` triple. No `cls.*` pooler tensors. New
//!    per-layer `blk.{i}.ffn_gate.weight`.
//!
//! Reference: llama.cpp's `llm_build_bert` (src/models/bert.cpp); the
//! `LLM_ARCH_NOMIC_BERT` branch is the spec we conform to. Day-one
//! supported model: `nomic-embed-text-v1.5` (137M, hidden=768, layers=12,
//! head_count=12, n_ff=3072, rope_freq_base=1000.0).
//!
//! # Module layout
//!
//!   - `mod.rs` (this file) — `ARCH_NOMIC_BERT` const + re-exports.
//!   - `config.rs` — `NomicBertConfig` GGUF metadata parser.
//!   - `weights.rs` — `LoadedNomicBertWeights` GGUF tensor loader.
//!   - `tokenizer.rs` — re-exports `BertWpmTokenizer` (same WPM scheme).
//!   - `forward.rs` — encoder forward pass + pooling. Lands in iter 76.
//!
//! # Why a parallel module instead of extending `bert/`
//!
//! The existing `bert::weights::validate_tensor_set` hardcodes
//! `position_embd.weight` as required and the per-layer suffixes as
//! `attn_q/k/v.weight` + `attn_output.weight` (no `ffn_gate`). Both are
//! correct for the day-one BERT models (`bge-small-en-v1.5`,
//! `mxbai-embed-large-v1`) and changing them would either break the
//! existing accuracy gate or grow `BertConfig` into an arch-aware union.
//! Per the project convention "all model-specific code in per-model files
//! under `models/`", `nomic_bert` is its own bounded context. Primitives
//! (`bert_linear_gpu`, `bert_layer_norm_gpu`, `bert_attention_gpu`,
//! `bert_residual_add_gpu`, `bert_pool_gpu`, `bert_l2_normalize_gpu`) are
//! reused via `crate::inference::models::bert::bert_gpu` since they're
//! arch-agnostic compute primitives — only the composer differs.

#![allow(dead_code)] // forward pass + handler wiring lands in later iters

pub mod config;
pub mod forward;
pub mod tokenizer;
pub mod weights;

#[allow(unused_imports)]
pub use config::{nomic_bert_layer_tensor, NomicBertConfig, NOMIC_BERT_TENSOR_TOKEN_EMBD};
#[allow(unused_imports)]
pub use forward::{
    apply_nomic_bert_encoder_block_gpu, apply_nomic_bert_full_forward_gpu,
    nomic_bert_embeddings_gpu, register_nomic_bert_kernels, NomicBertEncoderBlockTensors,
};
#[allow(unused_imports)]
pub use tokenizer::build_nomic_wordpiece_tokenizer;
#[allow(unused_imports)]
pub use weights::{
    validate_tensor_set, LoadedNomicBertWeights, NOMIC_BERT_BLOCK_OPTIONAL_SUFFIXES,
    NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES,
};

/// GGUF architecture identifier for the nomic-bert family.
///
/// Spelled lowercase with a hyphen to match
/// `/opt/llama.cpp/src/llama-arch.cpp:25`:
///   `{ LLM_ARCH_NOMIC_BERT, "nomic-bert" }`.
pub const ARCH_NOMIC_BERT: &str = "nomic-bert";
