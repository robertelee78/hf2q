//! BERT-family embedding models (ADR-005 Phase 2b, Task #13).
//!
//! Encoder-only, bidirectional attention, no KV cache, pooling per
//! `bert.pooling_type` GGUF metadata (NONE / MEAN / CLS / LAST / RANK).
//! Day-one supported: `nomic-embed-text-v1.5`, `mxbai-embed-large-v1`,
//! `bge-small-en-v1.5`.
//!
//! # Module layout
//!
//!   - `mod.rs` (this file) — shared: `BertConfig`, `PoolingType`, GGUF
//!     metadata parser, tensor-name table.
//!   - (future) `forward.rs` — encoder forward pass + pooling. Lands when
//!     the `embed_forward` API is plumbed (blocked on live-model
//!     validation).
//!
//! # GGUF metadata keys (llama.cpp convention)
//!
//!   - `general.architecture = "bert"`
//!   - `bert.embedding_length`      → hidden_size
//!   - `bert.attention.head_count`  → num_attention_heads
//!   - `bert.block_count`           → num_hidden_layers
//!   - `bert.feed_forward_length`   → intermediate_size
//!   - `bert.attention.layer_norm_epsilon` → layer_norm_eps
//!   - `bert.context_length`        → max_position_embeddings
//!   - `bert.pooling_type`          → pooling method (enum below)
//!   - `bert.causal_attention`      → false for encoder-only (BERT default)
//!
//! # Tensor-name table (llama.cpp GGUF BERT convention)
//!
//! Every tensor the encoder forward pass needs is declared here as a
//! const so the loader + forward code share a single source of truth.
//! These are plain `&'static str`; the per-layer variants use a helper
//! that formats `blk.{n}.{suffix}` for layer index `n`.

#![allow(dead_code)] // forward pass lands in a later iter

pub mod bert_gpu;
pub mod config;
pub mod tokenizer;
pub mod weights;
#[allow(unused_imports)]
pub use config::{
    bert_layer_tensor, BertConfig, PoolingType, TENSOR_EMBED_NORM_BIAS, TENSOR_EMBED_NORM_WEIGHT,
    TENSOR_POS_EMBD, TENSOR_TOKEN_EMBD, TENSOR_TOKEN_TYPES,
};
#[allow(unused_imports)]
pub use tokenizer::{
    build_token_to_id_map, build_wordpiece_tokenizer, BertSpecialTokens, BertVocab,
    BertWpmTokenizer,
};
#[allow(unused_imports)]
pub use weights::{
    validate_tensor_set, LoadedBertWeights, BERT_BLOCK_OPTIONAL_SUFFIXES,
    BERT_BLOCK_REQUIRED_SUFFIXES,
};

/// GGUF architecture identifier for the BERT family.
pub const ARCH_BERT: &str = "bert";
