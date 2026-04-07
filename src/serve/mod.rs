//! OpenAI-compatible API server for hf2q inference.
//!
//! Exposes the following endpoints matching the OpenAI API specification:
//! - `/v1/models` -- list loaded models
//! - `/v1/chat/completions` -- text generation with tool calling support
//! - `/v1/embeddings` -- text embeddings via prefill-only forward pass
//! - `/health` -- health check
//!
//! The entire module is gated behind the `serve` feature flag.
//!
//! Architecture:
//! - axum 0.8 for HTTP routing and handler extraction
//! - tower-http for CORS middleware
//! - tokio::task::spawn_blocking for Metal GPU compute isolation
//! - tokio::sync::mpsc for sync-to-async token streaming
//! - Dual-lane concurrency: Semaphore for generation queue, separate Semaphore
//!   for embedding requests (embeddings never block behind generation)

pub mod embeddings;
mod handlers;
mod middleware;
mod router;
pub mod schema;
mod sse;
pub mod tool_parser;

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use tokio::sync::Semaphore;
use tracing::info;

use crate::inference::engine::InferenceEngine;
use crate::tokenizer::chat_template::ChatTemplate;
use crate::tokenizer::HfTokenizer;

use embeddings::EmbeddingLane;
use sse::unix_timestamp;

// ---------------------------------------------------------------------------
// Generation queue
// ---------------------------------------------------------------------------

/// Semaphore-based generation queue providing backpressure.
///
/// Each permit represents one active generation slot. When all permits are
/// held, new requests receive HTTP 503 immediately via `try_acquire()`.
/// Permits are released automatically when the generation future completes
/// (or errors), thanks to RAII on the `OwnedSemaphorePermit`.
#[derive(Debug)]
pub struct GenerationQueue {
    semaphore: Arc<Semaphore>,
    capacity: usize,
}

impl GenerationQueue {
    /// Create a new generation queue with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(capacity)),
            capacity,
        }
    }

    /// Try to acquire a generation slot without blocking.
    ///
    /// Returns an owned permit on success; the caller holds it for the
    /// duration of generation. Returns an error if the queue is full.
    pub fn try_acquire(
        &self,
    ) -> Result<tokio::sync::OwnedSemaphorePermit, tokio::sync::TryAcquireError> {
        self.semaphore.clone().try_acquire_owned()
    }

    /// Number of currently active generations.
    pub fn active_count(&self) -> usize {
        self.capacity - self.semaphore.available_permits()
    }
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

/// Shared application state passed to all axum handlers via `State<AppState>`.
#[derive(Clone)]
pub struct AppState {
    /// Model name (derived from the model directory basename).
    pub model_name: String,
    /// Unix timestamp when the server started (used in model object).
    pub created_at: i64,
    /// The inference engine (behind a Mutex because `generate` takes &mut self).
    pub engine: Arc<Mutex<InferenceEngine>>,
    /// Tokenizer for prompt token counting (separate from engine to avoid
    /// holding the engine Mutex during validation).
    pub tokenizer: Arc<Mutex<HfTokenizer>>,
    /// Chat template for rendering messages into prompts.
    pub chat_template: Arc<ChatTemplate>,
    /// Generation queue for backpressure.
    pub queue: Arc<GenerationQueue>,
    /// Maximum sequence length supported by the model.
    pub max_seq_len: usize,
    /// Embedding lane for concurrent embedding requests (independent of generation).
    pub embedding_lane: Arc<EmbeddingLane>,
}

/// Configuration for the serve command.
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub queue_depth: usize,
    /// Maximum concurrent embedding requests (default 4).
    pub embedding_concurrency: usize,
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

/// Start the API server.
///
/// Loads the model, builds the router, and serves HTTP requests until
/// a shutdown signal (SIGINT/SIGTERM) is received.
pub async fn run(
    engine: InferenceEngine,
    tokenizer: HfTokenizer,
    chat_template: ChatTemplate,
    model_name: String,
    max_seq_len: usize,
    config: ServeConfig,
) -> Result<()> {
    let state = AppState {
        model_name,
        created_at: unix_timestamp(),
        engine: Arc::new(Mutex::new(engine)),
        tokenizer: Arc::new(Mutex::new(tokenizer)),
        chat_template: Arc::new(chat_template),
        queue: Arc::new(GenerationQueue::new(config.queue_depth)),
        max_seq_len,
        embedding_lane: Arc::new(EmbeddingLane::new(config.embedding_concurrency)),
    };

    let app = router::build_router(state);

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("Failed to bind to {}", addr))?;

    info!("Listening on {}", addr);
    eprintln!("Listening on {}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Server error")?;

    info!("Server shut down");
    Ok(())
}

/// Wait for a shutdown signal (Ctrl+C or SIGTERM).
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("Shutdown signal received, stopping server...");
    eprintln!("\nShutting down...");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_queue_new() {
        let queue = GenerationQueue::new(4);
        assert_eq!(queue.active_count(), 0);
        assert_eq!(queue.capacity, 4);
    }

    #[tokio::test]
    async fn test_generation_queue_acquire_and_release() {
        let queue = GenerationQueue::new(2);
        assert_eq!(queue.active_count(), 0);

        let permit1 = queue.try_acquire().unwrap();
        assert_eq!(queue.active_count(), 1);

        let permit2 = queue.try_acquire().unwrap();
        assert_eq!(queue.active_count(), 2);

        // Queue is full
        assert!(queue.try_acquire().is_err());

        // Release one permit
        drop(permit1);
        assert_eq!(queue.active_count(), 1);

        // Can acquire again
        let permit3 = queue.try_acquire().unwrap();
        assert_eq!(queue.active_count(), 2);

        drop(permit2);
        drop(permit3);
        assert_eq!(queue.active_count(), 0);
    }

    #[tokio::test]
    async fn test_generation_queue_full_returns_error() {
        let queue = GenerationQueue::new(1);
        let _permit = queue.try_acquire().unwrap();
        assert!(queue.try_acquire().is_err());
    }
}
