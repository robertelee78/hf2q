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
mod compressed_attention;
mod compressed_attention_common;
mod compressed_attention_indexer;
mod compressed_attention_main;
mod compressed_attention_weights;
pub mod compressor;
pub mod config;
mod ffn_forward;
pub mod forward;
mod forward_support;
pub mod hyper_connection;
pub mod model;
pub mod residency;
pub mod rope;
pub mod routing;
mod verifier_forward;
pub mod weights;

pub use config::Deepseek4Config;
pub use model::Deepseek4Model;
pub use residency::Deepseek4Weights;

#[cfg(test)]
mod attention_forward_tests;
#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod ffn_forward_tests;
#[cfg(test)]
mod forward_tests;
#[cfg(test)]
mod model_tests;
#[cfg(test)]
mod real_artifact_tests;
#[cfg(test)]
mod residency_tests;

pub const ARCH_DEEPSEEK4: &str = "deepseek4";
