//! Route definitions for the OpenAI-compatible API server.
//!
//! Assembles the axum Router with all endpoint handlers, CORS middleware,
//! and the fallback handler for unmatched routes.

use axum::routing::{get, post};
use axum::Router;

use super::embeddings;
use super::handlers;
use super::middleware;
use super::AppState;

/// Build the complete axum router with all routes and middleware.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/{model_id}", get(handlers::get_model))
        .route(
            "/v1/chat/completions",
            post(handlers::chat_completions),
        )
        .route("/v1/embeddings", post(embeddings::embeddings))
        .fallback(handlers::fallback)
        .layer(middleware::cors_layer())
        .with_state(state)
}
