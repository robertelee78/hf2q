//! ADR-033 P3 — convert orchestration scaffolding.
//!
//! Wires `StandardPolicy::target_for` + `GgmlQuantizer::quantize` +
//! `GgufWriter` into a single self-contained pipeline driver, so the
//! integration shape can be exercised end-to-end on synthetic tensors
//! before the per-arch safetensors mappers (P4+) come online.
//!
//! This module is intentionally **not** wired into `cmd_convert` (the
//! legacy two-pass pipeline at `src/main.rs::cmd_convert`); the legacy
//! pipeline stays load-bearing until P6 deletes it. Per
//! [[feedback-no-backwards-compat-2026-05-18]]: this is new code — no
//! migration shims.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: the orchestrator
//! returns typed errors; the only place where a tensor escapes the
//! `StandardPolicy` → `GgmlQuantizer` pipeline is the
//! vision/audio-pattern gate in [`crate::quantize::ggml_quants::vision`].
//! Matched tensors emit F16 directly; unmatched tensors that fail the
//! policy / quantizer surface as `OrchestratorError::*` — never silent
//! F16 demotion.

pub mod orchestrator;

pub use orchestrator::{ConvertOrchestrator, OrchestratorError};
