//! Per-architecture model implementations.
//!
//! Each sub-module owns its own config parser, tensor-name table,
//! forward-pass graph builder, and KV-cache management. Modules do NOT
//! depend on each other in their composers; primitives in
//! `bert::bert_gpu` are arch-agnostic GPU kernels reused by `nomic_bert`
//! to avoid duplication of the linear / layer-norm / attention building
//! blocks.

pub mod bert;
pub mod nomic_bert;
pub mod qwen35;
