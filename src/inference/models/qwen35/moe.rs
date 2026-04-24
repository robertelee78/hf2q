//! Qwen3.5-MoE 35B-A3B mixture-of-experts inference.
//!
//! Ownership:
//! - 256-expert top-8 routing (`ffn_gate_inp`, `ffn_gate_up_exps`,
//!   `ffn_down_exps`).
//! - Gated shared expert (`ffn_gate_inp_shexp` sigmoid-gated onto
//!   `ffn_{gate,up,down}_shexp`) — ADR-013 Decision 13.
//! - MoE tensor-name table resolution.
//! - MoE forward entry point.
//!
//! The linear-attention branch, gated full-attention branch, MROPE
//! dispatch, hybrid KV cache, and tokenizer are shared with
//! [`super::dense`] and live in [`super`].
//!
//! Implementation lands in phases P9 (FFN), P11 (forward wire-up), and
//! P13 (correctness gate) — see ADR-013.
