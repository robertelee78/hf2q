//! Model-specific conversion logic.
//!
//! Per `project_model_class_split.md`: all arch-specific code lives here,
//! never in generic infrastructure (ir.rs, backends/, etc.).
//!
//! ADR-012: Qwen3.5-family conversion module (P2).
//! ADR-013: inference-side models live in src/inference/models/ — do NOT mix.

pub mod qwen35;

/// ADR-012 P10: pure-Rust mmproj vision-tower emitter.
pub mod vit;
