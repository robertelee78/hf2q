//! OpenAI-compatible HTTP API server for hf2q (ADR-005 Phase 2).
//!
//! This module hosts the spec layer (request/response types, SSE encoding,
//! router assembly) and — in later loop iterations — the handlers, AppState,
//! middleware, grammar stack, prompt cache, and embedding path.
//!
//! Every submodule below is engine-agnostic or mlx-native-wired per Decision
//! in ADR-005's 2026-04-23 Phase 2 scope refinement. No candle, no MLX-rs.
//!
//! Submodule layout (progressive — future iterations add to this list):
//!   - `schema` — OpenAI wire-format types (Tiers 1+2+3+4, reasoning split,
//!                overflow policy, logprobs, embeddings). Engine-agnostic.

#![allow(dead_code)] // handlers and downstream wiring land in later iterations

pub mod schema;
