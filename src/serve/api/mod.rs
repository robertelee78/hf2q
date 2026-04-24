//! OpenAI-compatible HTTP API server for hf2q (ADR-005 Phase 2).
//!
//! This module hosts the spec layer (request/response types, SSE encoding,
//! router assembly, middleware, handlers) and — in later loop iterations —
//! the engine wiring (model load + warmup + forward pass bridge + prompt
//! cache + grammar + embeddings).
//!
//! Every submodule below is engine-agnostic or mlx-native-wired per Decision
//! in ADR-005's 2026-04-23 Phase 2 scope refinement. No candle, no MLX-rs.
//!
//! Submodule layout (progressive — future iterations add to this list):
//!   - `schema`      — OpenAI wire-format types (Tiers 1+2+3+4, reasoning
//!                     split, overflow policy, logprobs, embeddings).
//!                     Engine-agnostic.
//!   - `sse`         — SSE stream encoder over the generation event
//!                     protocol. Grammar-free; tool-call deltas come from
//!                     the grammar sampler upstream.
//!   - `state`       — `AppState` + `ServerConfig` threaded through axum.
//!   - `middleware`  — CORS, optional Bearer auth, request-id.
//!   - `handlers`    — `/health`, `/readyz`, `/v1/models`, `/v1/models/:id`.
//!   - `router`      — axum router assembly, layered middleware, 404 fallback.

#![allow(dead_code)] // some handlers + state helpers land with the engine iter

pub mod schema;
pub mod sse;
pub mod state;
pub mod middleware;
pub mod handlers;
pub mod router;

pub use router::build_router;
#[allow(unused_imports)]
pub use state::{AppState, ServerConfig};
