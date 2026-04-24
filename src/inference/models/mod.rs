//! Per-architecture model implementations.
//!
//! Each sub-module owns its own config parser, tensor-name table,
//! forward-pass graph builder, and KV-cache management. Modules do NOT
//! depend on each other; they share mlx-native ops only.

pub mod bert;
pub mod qwen35;
