//! Qwen3.5 27B dense-variant inference.
//!
//! Ownership:
//! - Dense SwiGLU FFN (`gate_proj`, `up_proj`, `down_proj`).
//! - Dense tensor-name table resolution.
//! - Dense forward entry point.
//!
//! The linear-attention branch, gated full-attention branch, MROPE
//! dispatch, hybrid KV cache, and tokenizer are shared with
//! [`super::moe`] and live in [`super`].
//!
//! Implementation lands in phases P9 (FFN), P11 (forward wire-up), and
//! P13 (correctness gate) — see ADR-013.
