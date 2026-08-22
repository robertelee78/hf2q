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

pub(crate) const DIAGNOSTIC_NO_EVICT_HEADER: &str = "x-hf2q-diagnostic-no-evict";
pub(crate) const DIAGNOSTIC_NO_EVICT_VALUE: &str = "1";

pub mod artifact_catalog;
pub mod cancellation;
pub mod control;
pub mod embedding_pool;
pub mod engine;
pub mod engine_deepseek4;
pub mod engine_qwen35;
pub mod engine_qwen3vl;
mod engine_supervisor;
pub mod grammar;
pub mod handlers;
pub mod kv_spill_descriptor;
pub mod lifecycle;
pub mod local_artifacts;
pub mod middleware;
pub mod qwen35_speculation;
mod qwen35_anchor_store;
mod qwen_thinking_policy;
pub mod registry;
pub mod router;
pub mod schema;
pub mod sse;
pub mod state;
pub mod tq_packed_descriptor;

#[cfg(test)]
mod qwen36_watchdog_fixture_tests;

pub use router::build_router;
#[allow(unused_imports)]
pub use state::{AppState, ServerConfig};
