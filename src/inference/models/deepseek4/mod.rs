//! DeepSeek-V4-Flash inference primitives.
//!
//! This module is deliberately architecture-specific. DeepSeek-V4's
//! sqrt-softplus router and Hyper-Connection residual mixer are not
//! interchangeable with the Qwen or Gemma graphs.

pub mod attention;
mod attention_entry;
mod attention_forward;
mod attention_weights;
pub mod cache;
mod cache_buffers;
mod cache_window_write;
mod compressed_attention;
mod compressed_attention_common;
mod compressed_attention_indexer;
mod compressed_attention_main;
mod compressed_attention_weights;
pub mod compressor;
mod compressor_tail;
pub mod config;
mod decode_cohort;
mod ffn_forward;
pub mod forward;
mod forward_support;
pub mod hyper_connection;
mod joint_decode;
mod joint_decode_core;
mod joint_decode_routes;
pub mod model;
mod output_forward;
mod prefill_annotations;
mod prefill_flash_attention;
pub mod residency;
pub mod rope;
pub mod routing;
mod row_matmul;
mod submission;
pub mod tokenizer;
mod verifier_forward;
pub mod weights;

pub use config::Deepseek4Config;
pub(crate) use forward_support::{
    decode_scratch_stats, prefill_scratch_stats, release_decode_scratch, release_prefill_scratch,
    TransientScratchStats,
};
pub(crate) use joint_decode::MAX_JOINT_DECODE_ROWS;
pub use model::Deepseek4Model;
pub use residency::Deepseek4Weights;
pub(crate) use verifier_forward::{
    matrix_prefill_chunk_len, MAX_COOPERATIVE_PREFILL_ROWS, MIN_MATRIX_APPEND_TOKENS,
};

#[cfg(test)]
mod attention_forward_tests;
#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod cooperative_fault;
#[cfg(test)]
mod ffn_forward_tests;
#[cfg(test)]
mod forward_tests;
#[cfg(test)]
pub(crate) mod mixed_step_tests;
#[cfg(test)]
mod model_tests;
#[cfg(test)]
mod real_artifact_cohort_tests;
#[cfg(test)]
mod real_artifact_decode_cohort_tests;
#[cfg(test)]
mod real_artifact_joint_decode_tests;
#[cfg(test)]
mod real_artifact_mixed_step_tests;
#[cfg(test)]
mod real_artifact_tests;
#[cfg(test)]
mod residency_tests;
#[cfg(test)]
mod row_matmul_tests;

pub const ARCH_DEEPSEEK4: &str = "deepseek4";

#[cfg(test)]
mod bounded_map_ledger;
