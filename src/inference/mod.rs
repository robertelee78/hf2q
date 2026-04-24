//! Per-model inference modules.
//!
//! This is the home for architecture-specific inference code that does not
//! fit inside `src/serve/forward_mlx.rs` (which is Gemma-4-shaped).
//!
//! Per `project_model_class_split.md`: all model-specific code lives in a
//! per-model file under `models/`, not in generic infra.
//!
//! Populated by ADR-013 (Qwen3.5 / Qwen3.5-MoE).

pub mod models;
