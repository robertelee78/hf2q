//! Quantization module — pure-Rust GGML kernel ports for convert-v2.
//!
//! Post-P6 (ADR-033 §P6) the legacy two-pass quantize/calibrate/quality
//! stack is gone; only the [`ggml_quants`] subdir survives. It hosts
//! the pure-Rust kernel ports for every standard ggml ftype the new
//! convert-v2 pipeline emits, plus the APEX algorithmic tiers.
//!
//! Callers should import directly from `crate::quantize::ggml_quants`
//! (e.g. `quantize::ggml_quants::quantizer::Quantizer`,
//! `quantize::ggml_quants::LlamaFtype`,
//! `quantize::ggml_quants::apex::ApexPolicy`).
//!
//! The `imatrix` submodule implements ADR-033 §Pi — in-tree imatrix
//! generation + `.imatrix.gguf` read/write. Phase A (corpus loader,
//! accumulator, gguf writer/loader, file-based `--imatrix <path>`)
//! AND Phase B (forward-pass driver wired through the convert
//! orchestrator, exposed as `--imatrix-corpus <name>`) BOTH
//! SHIPPED 2026-05-19. Gemma 4 is the Stage 3.0 driver arch; other
//! arches consume pre-computed `.imatrix.gguf` via `--imatrix
//! <path>` until Stage 3b.4 adds Qwen35Moe driver wiring.

pub mod ggml_quants;
pub mod imatrix;
