//! Shared application state (`AppState`) for the hf2q HTTP API server.
//!
//! The axum router threads a single `AppState` through every handler. This
//! type is deliberately thin: it holds immutable per-server config, the
//! startup timestamp (for `/health` uptime), a warmup-ready atomic (for
//! `/readyz`), and a monotonic request counter (for request-id generation
//! and `/metrics`).
//!
//! In this iteration the inference engine itself is NOT part of `AppState` —
//! the engine + model load + warmup pipeline lands in the next iteration
//! alongside `/v1/chat/completions` and `/v1/embeddings`. The `ready_for_gen`
//! flag is currently always `true` because no generation endpoint is routed
//! yet; when the engine is introduced, the flag will start `false`, flip to
//! `true` after warmup completes, and the chat/embedding handlers will gate
//! on it per ADR-005 Decision #16 (`503 + Retry-After` during warmup).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use super::engine::Engine;
use super::schema::OverflowPolicy;

/// Server-level configuration, captured at startup from CLI flags + defaults.
///
/// All fields are immutable for the lifetime of the server. A restart is
/// required to change any of them (matching ollama / llama.cpp conventions).
///
/// Decision numbers reference ADR-005 Phase 2 refinement (2026-04-23).
#[derive(Debug, Clone)]
pub struct ServerConfig {
    // --- Networking ---
    /// Bind address. Defaults to `127.0.0.1` (Decision #7).
    pub host: String,
    /// TCP port. Defaults to `8080`.
    pub port: u16,

    // --- Auth (Decision #8) ---
    /// Optional Bearer token. When `Some(token)`, every request must carry
    /// `Authorization: Bearer <token>` or receive 401. When `None`, no auth.
    pub auth_token: Option<String>,

    // --- CORS (Decision #9) ---
    /// Allowed origins for CORS. Empty = wide-open `*` (localhost dev
    /// default); populated = restrictive allowlist.
    pub cors_allowed_origins: Vec<String>,

    // --- Queue (Decision #19) ---
    /// Hard cap on the FIFO generation queue. Overflow returns 429 +
    /// `Retry-After`. Applies only to generation endpoints.
    pub queue_capacity: usize,

    // --- Rate limits (Decision #10) ---
    /// Max concurrent in-flight HTTP requests per bind. 0 = unlimited
    /// (bounded only by the queue cap + OS).
    pub max_concurrent_requests: usize,

    // --- Timeouts ---
    /// Per-request timeout (applies to the whole request including queue wait
    /// and generation). 0 = no timeout.
    pub request_timeout_seconds: u64,

    // --- Overflow policy (Decision #23) ---
    /// Default context-overflow policy. Per-request `hf2q_overflow_policy`
    /// overrides this.
    pub default_overflow_policy: OverflowPolicy,

    // --- Model catalog ---
    /// Directory to scan for `/v1/models` listing. Per Decision #26 this is
    /// `~/.cache/hf2q/`; overridable for tests / bring-your-own-cache.
    pub cache_dir: Option<PathBuf>,

    // --- Server identity ---
    /// Optional system fingerprint advertised via `ChatCompletionResponse.
    /// system_fingerprint`. Defaults to `None`; production can set to
    /// `"hf2q-<short-git-sha>-mlx-native"`.
    pub system_fingerprint: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        // Defaults are aligned with Decision #7 (localhost bind) + conservative
        // queue + no auth. Tests construct with defaults + per-test overrides.
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            auth_token: None,
            cors_allowed_origins: Vec::new(),
            queue_capacity: 32,
            max_concurrent_requests: 0,
            request_timeout_seconds: 0,
            default_overflow_policy: OverflowPolicy::Summarize,
            cache_dir: default_cache_dir(),
            system_fingerprint: None,
        }
    }
}

/// Resolve the default HF2Q cache directory (`$HOME/.cache/hf2q`).
///
/// Returns `None` if `$HOME` is unset (test / hermetic CI envs).
pub fn default_cache_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|h| h.join(".cache").join("hf2q"))
}

/// Process-wide metric counters surfaced via `/metrics` in Prometheus text
/// format (Decision #11). Cheap atomics; handler code bumps them inline.
#[derive(Debug, Default)]
pub struct ServerMetrics {
    /// Total number of HTTP requests that reached a handler (post-auth).
    pub requests_total: AtomicU64,
    /// Total number of chat-completion generations started.
    pub chat_completions_started: AtomicU64,
    /// Total number of chat-completion generations that completed
    /// successfully.
    pub chat_completions_completed: AtomicU64,
    /// Total number of chat-completion generations that hit
    /// `queue_full` (FIFO at capacity → 429).
    pub chat_completions_queue_full: AtomicU64,
    /// Total number of SSE stream cancellations (client dropped the
    /// connection mid-generation; Decision #18).
    pub sse_cancellations: AtomicU64,
    /// Total tokens decoded across all completions (cumulative counter).
    pub decode_tokens_total: AtomicU64,
    /// Total prompt tokens ingested across all completions.
    pub prompt_tokens_total: AtomicU64,
    /// Total requests rejected at handler boundary (auth, malformed, etc.).
    pub requests_rejected_total: AtomicU64,
}

/// Shared runtime state threaded through axum handlers.
///
/// Cheap to clone (every field is behind `Arc` or is a plain atomic wrapper).
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub started_at: Arc<Instant>,
    /// `true` once the server is ready to serve generation. When `engine` is
    /// `None`, generation endpoints return 400 `model_not_loaded`. When
    /// `engine` is `Some` but `ready_for_gen` is `false`, they return 503
    /// `not_ready` + `Retry-After: 1` (Decision #16, warmup-in-progress).
    pub ready_for_gen: Arc<AtomicBool>,
    /// Monotonic counter for request-id generation + metrics.
    pub request_counter: Arc<AtomicU64>,
    /// Inference engine. `None` means no `--model` was supplied at startup
    /// (the HTTP backbone still serves `/health`, `/readyz`, `/v1/models`).
    /// `Some(engine)` means a model is loaded and the worker thread is up;
    /// generation endpoints gate on `ready_for_gen` for warmup status.
    pub engine: Option<Engine>,
    /// Process-wide metric counters surfaced via `/metrics`.
    pub metrics: Arc<ServerMetrics>,
}

impl AppState {
    /// Construct `AppState` without an engine. Only `/health`, `/readyz`,
    /// `/v1/models` will serve real data; generation endpoints return
    /// `model_not_loaded`. Used by the iter-2 backbone path and by tests.
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config: Arc::new(config),
            started_at: Arc::new(Instant::now()),
            // No engine → no warmup needed → ready.
            ready_for_gen: Arc::new(AtomicBool::new(true)),
            request_counter: Arc::new(AtomicU64::new(0)),
            engine: None,
            metrics: Arc::new(ServerMetrics::default()),
        }
    }

    /// Construct `AppState` with an engine. The engine worker thread is
    /// already running at this point; `ready_for_gen` starts `false` until
    /// the warmup task flips it.
    pub fn with_engine(config: ServerConfig, engine: Engine) -> Self {
        Self {
            config: Arc::new(config),
            started_at: Arc::new(Instant::now()),
            ready_for_gen: Arc::new(AtomicBool::new(false)),
            request_counter: Arc::new(AtomicU64::new(0)),
            engine: Some(engine),
            metrics: Arc::new(ServerMetrics::default()),
        }
    }

    /// Seconds since the server started.
    pub fn uptime_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    /// Mark the server ready for generation (called after warmup).
    pub fn mark_ready_for_gen(&self) {
        self.ready_for_gen.store(true, Ordering::Release);
    }

    /// Mark the server NOT ready (e.g. during graceful shutdown drain).
    pub fn mark_not_ready(&self) {
        self.ready_for_gen.store(false, Ordering::Release);
    }

    pub fn is_ready_for_gen(&self) -> bool {
        self.ready_for_gen.load(Ordering::Acquire)
    }

    /// Allocate the next request counter value.
    pub fn next_request_seq(&self) -> u64 {
        self.request_counter.fetch_add(1, Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_localhost() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 8080);
        assert!(cfg.auth_token.is_none());
        assert!(cfg.cors_allowed_origins.is_empty());
        assert_eq!(cfg.queue_capacity, 32);
        assert_eq!(cfg.default_overflow_policy, OverflowPolicy::Summarize);
    }

    #[test]
    fn app_state_starts_ready_in_iter_2() {
        let state = AppState::new(ServerConfig::default());
        assert!(state.is_ready_for_gen());
        assert_eq!(state.uptime_seconds(), 0);
    }

    #[test]
    fn mark_not_ready_flips_to_false_then_back() {
        let state = AppState::new(ServerConfig::default());
        assert!(state.is_ready_for_gen());
        state.mark_not_ready();
        assert!(!state.is_ready_for_gen());
        state.mark_ready_for_gen();
        assert!(state.is_ready_for_gen());
    }

    #[test]
    fn request_seq_is_monotonic() {
        let state = AppState::new(ServerConfig::default());
        let a = state.next_request_seq();
        let b = state.next_request_seq();
        let c = state.next_request_seq();
        assert_eq!(a + 1, b);
        assert_eq!(b + 1, c);
    }
}
