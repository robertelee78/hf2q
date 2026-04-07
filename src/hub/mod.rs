//! Hub module — HuggingFace Hub model download and resolution for inference.
//!
//! This module provides model download and cache management for the inference
//! engine. It reuses patterns from `input::hf_download` but uses a persistent
//! cache at `~/.cache/hf2q/models/` so downloaded inference models survive
//! across runs.

#[cfg(feature = "mlx-native")]
pub mod download;
