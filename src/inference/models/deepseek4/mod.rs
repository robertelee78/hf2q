//! DeepSeek-V4-Flash inference primitives.
//!
//! This module is deliberately architecture-specific. DeepSeek-V4's
//! sqrt-softplus router and Hyper-Connection residual mixer are not
//! interchangeable with the Qwen or Gemma graphs.

pub mod attention;
pub mod cache;
pub mod compressor;
pub mod config;
pub mod hyper_connection;
pub mod rope;
pub mod residency;
pub mod routing;
pub mod weights;

pub use config::Deepseek4Config;
pub use residency::Deepseek4Weights;

#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod residency_tests;

pub const ARCH_DEEPSEEK4: &str = "deepseek4";
