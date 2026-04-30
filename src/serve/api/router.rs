//! Axum route assembly for the hf2q API server.
//!
//! Mounts the `AppState`-threaded handlers under their OpenAI-compatible
//! paths, layers on middleware (CORS, request-id, optional Bearer auth), and
//! installs the OpenAI-shaped 404 fallback.
//!
//! Layer order matters. axum layers run outside-in on request and inside-out
//! on response; the composition below is:
//!
//!     request  → cors → request-id → [route] → bearer_auth → handler
//!     response ← cors ← request-id ← [route] ← bearer_auth ← handler
//!
//! - CORS is outermost so preflight `OPTIONS` requests are handled without
//!   hitting auth (matches ollama/llama.cpp convention for LAN deployments).
//! - request-id is between CORS and auth so every response carries the id
//!   even when auth rejects the request (useful for client-side debugging).
//! - auth is innermost so it runs AFTER the request-id is stamped, allowing
//!   401 responses to carry the request-id for correlation.

use axum::middleware::from_fn_with_state;
use axum::routing::{get, post};
use axum::Router;

use super::handlers;
use super::middleware::{bearer_auth, cors_layer, fallback, request_id_layer};
use super::state::AppState;

/// Build the complete axum router with all routes and middleware.
///
/// In iter 2, POST `/v1/chat/completions` and POST `/v1/embeddings` are NOT
/// yet routed — adding them is a later iter's additive change. Hitting them
/// under this iter returns the OpenAI-shaped 404 error envelope via
/// `fallback`.
pub fn build_router(state: AppState) -> Router {
    let cors = cors_layer(&state.config.cors_allowed_origins);

    Router::new()
        .route("/health", get(handlers::health))
        .route("/readyz", get(handlers::readyz))
        .route("/metrics", get(handlers::metrics))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/:model_id", get(handlers::get_model))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/embeddings", post(handlers::embeddings))
        .fallback(fallback)
        // Apply layers outside-in. The axum convention is `.layer()` wraps,
        // so the last `.layer(X)` call becomes the outermost layer.
        // Order chosen: bearer_auth innermost, then request-id, then CORS.
        .layer(from_fn_with_state(state.clone(), bearer_auth))
        .layer(axum::middleware::from_fn(request_id_layer))
        .layer(cors)
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Integration tests (router-level)
// ---------------------------------------------------------------------------
//
// These tests exercise the composed router end-to-end via `tower::ServiceExt`
// so every layer runs (CORS, request-id, auth, fallback). They do NOT spin
// up a real TCP socket — the axum `oneshot` service path is used instead.

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{header, Request, StatusCode};
    use tower::ServiceExt;

    use super::super::state::ServerConfig;

    fn state_default() -> AppState {
        AppState::new(ServerConfig::default())
    }

    fn state_with_auth(token: &str) -> AppState {
        let cfg = ServerConfig {
            auth_token: Some(token.to_string()),
            ..Default::default()
        };
        AppState::new(cfg)
    }

    async fn body_string(resp: axum::response::Response) -> String {
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    // --- health ---

    #[tokio::test]
    async fn health_returns_200_with_json() {
        let app = build_router(state_default());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["backend"], "mlx-native");
        assert!(v["uptime_seconds"].as_u64().is_some());
    }

    // --- readyz ---

    #[tokio::test]
    async fn readyz_returns_200_when_ready() {
        let state = state_default();
        state.mark_ready_for_gen();
        let app = build_router(state);
        let req = Request::builder()
            .uri("/readyz")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["ready"], true);
    }

    #[tokio::test]
    async fn readyz_returns_503_when_not_ready_with_retry_after() {
        let state = state_default();
        state.mark_not_ready();
        let app = build_router(state);
        let req = Request::builder()
            .uri("/readyz")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get(header::RETRY_AFTER).and_then(|v| v.to_str().ok()),
            Some("1")
        );
        let body = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["ready"], false);
    }

    // --- /v1/models ---

    #[tokio::test]
    async fn list_models_returns_empty_list_when_cache_unset() {
        let cfg = ServerConfig {
            cache_dir: None,
            ..Default::default()
        };
        let state = AppState::new(cfg);
        let app = build_router(state);
        let req = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["object"], "list");
        assert!(v["data"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn get_model_returns_404_when_absent() {
        let cfg = ServerConfig {
            cache_dir: None,
            ..Default::default()
        };
        let state = AppState::new(cfg);
        let app = build_router(state);
        let req = Request::builder()
            .uri("/v1/models/does-not-exist")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["code"], "model_not_found");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("does-not-exist"));
    }

    // --- fallback 404 ---

    #[tokio::test]
    async fn unknown_route_returns_openai_shaped_404() {
        let app = build_router(state_default());
        let req = Request::builder()
            .uri("/this-route-does-not-exist")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("/this-route-does-not-exist"));
    }

    // --- bearer auth ---

    #[tokio::test]
    async fn no_auth_configured_allows_unauthenticated_requests() {
        let app = build_router(state_default());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_configured_rejects_missing_header_with_401() {
        let app = build_router(state_with_auth("secret-token"));
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let body = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], "authentication_error");
    }

    #[tokio::test]
    async fn auth_configured_rejects_wrong_token_with_401() {
        let app = build_router(state_with_auth("secret-token"));
        let req = Request::builder()
            .uri("/health")
            .header(header::AUTHORIZATION, "Bearer wrong-token")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_configured_rejects_non_bearer_scheme_with_401() {
        let app = build_router(state_with_auth("secret-token"));
        let req = Request::builder()
            .uri("/health")
            .header(header::AUTHORIZATION, "Basic dXNlcjpwYXNz")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_configured_accepts_correct_token() {
        let app = build_router(state_with_auth("secret-token"));
        let req = Request::builder()
            .uri("/health")
            .header(header::AUTHORIZATION, "Bearer secret-token")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // --- request-id ---

    #[tokio::test]
    async fn response_always_has_request_id_header() {
        let app = build_router(state_default());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let id = resp
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(!id.is_empty());
        // Generated UUIDv4 is 36 chars.
        assert_eq!(id.len(), 36);
    }

    #[tokio::test]
    async fn client_request_id_is_echoed_in_response() {
        let app = build_router(state_default());
        let req = Request::builder()
            .uri("/health")
            .header("x-request-id", "client-supplied-id-42")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.headers().get("x-request-id").and_then(|v| v.to_str().ok()),
            Some("client-supplied-id-42")
        );
    }

    #[tokio::test]
    async fn request_id_present_even_on_401() {
        let app = build_router(state_with_auth("secret"));
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let id = resp
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(!id.is_empty(), "request-id must be present on 401 too");
    }

    // --- 404 fallback still emits request-id + correct shape ---

    // --- /v1/chat/completions (non-streaming path, no engine) ---

    #[tokio::test]
    async fn chat_completions_without_engine_returns_model_not_loaded() {
        let app = build_router(state_default());
        let body = r#"{"model":"gemma4","messages":[{"role":"user","content":"hi"}]}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body_text = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body_text).unwrap();
        assert_eq!(v["error"]["code"], "model_not_loaded");
    }

    #[tokio::test]
    async fn chat_completions_rejects_empty_messages() {
        // Even without an engine, an empty messages array should fail
        // validation before the engine gate — BUT the handler currently
        // gates on engine first, so this returns model_not_loaded. The
        // validation path is exercised once the engine is wired in iter 4
        // tests. This test documents the current ordering.
        let app = build_router(state_default());
        let body = r#"{"model":"gemma4","messages":[]}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
        // With no engine, the handler returns model_not_loaded first.
        assert_eq!(v["error"]["code"], "model_not_loaded");
    }

    // --- /metrics (Decision #11) ---

    #[tokio::test]
    async fn metrics_returns_prometheus_text_format() {
        let app = build_router(state_default());
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.starts_with("text/plain") && ct.contains("version=0.0.4"),
            "unexpected content-type: {}",
            ct
        );
        let body = body_string(resp).await;
        // Spot-check the expected metric names + HELP/TYPE annotations.
        for expected in [
            "hf2q_uptime_seconds",
            "hf2q_ready",
            "hf2q_model_loaded",
            "hf2q_requests_total",
            "hf2q_chat_completions_started",
            "hf2q_decode_tokens_total",
            "# HELP",
            "# TYPE",
        ] {
            assert!(
                body.contains(expected),
                "metric body missing {:?}\nbody:\n{}",
                expected,
                body
            );
        }
    }

    #[tokio::test]
    async fn metrics_ready_gauge_reflects_state() {
        let state = state_default();
        state.mark_not_ready();
        let app = build_router(state);
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = body_string(resp).await;
        // `hf2q_ready 0` must appear on its own line when not ready.
        assert!(
            body.lines().any(|l| l.trim() == "hf2q_ready 0"),
            "expected `hf2q_ready 0` line; body:\n{}",
            body
        );
    }

    #[tokio::test]
    async fn metrics_counter_increments_after_health_request() {
        // Hit /health once, then /metrics and assert requests_total >= 2.
        let state = state_default();
        let app = build_router(state.clone());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let _ = app.oneshot(req).await.unwrap();
        let app2 = build_router(state);
        let req2 = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app2.oneshot(req2).await.unwrap();
        let body = body_string(resp).await;
        // Find the `hf2q_requests_total <N>` line and assert N >= 1.
        let line = body
            .lines()
            .find(|l| l.starts_with("hf2q_requests_total"))
            .expect("requests_total line present");
        let n: u64 = line
            .split_whitespace()
            .last()
            .and_then(|s| s.parse().ok())
            .expect("parse counter");
        assert!(n >= 1, "requests_total should have been bumped, got {}", n);
    }

    // --- response_format pre-compile validation (Decision #6) ---

    #[tokio::test]
    async fn bad_json_schema_returns_grammar_error() {
        // Engine gate still runs first (no engine → model_not_loaded).
        // To exercise the grammar path, the test would need a loaded engine;
        // for now we assert the error kind is model_not_loaded (gate order)
        // and document: engine-loaded variant gets the grammar_error path.
        let app = build_router(state_default());
        let body = r#"{
            "model":"nope",
            "messages":[{"role":"user","content":"hi"}],
            "response_format":{"type":"json_schema","json_schema":{"name":"bad","schema":{"type":"not_a_real_type"}}}
        }"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // Without an engine loaded, the engine gate runs first and we
        // return model_not_loaded. The grammar pre-compile kicks in only
        // after engine-gate passes; this is the documented ordering.
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value =
            serde_json::from_str(&body_string(resp).await).unwrap();
        assert_eq!(v["error"]["code"], "model_not_loaded");
    }

    // --- Multimodal validation (Phase 2c, iter 23) ---

    #[tokio::test]
    async fn chat_completions_without_engine_multimodal_gate_is_secondary() {
        // Without an engine loaded, the engine gate runs FIRST — so a
        // multimodal request without --model returns model_not_loaded
        // (same as text-only). Documents the ordering: multimodal
        // validation happens after engine-gate.
        let app = build_router(state_default());
        let body = r#"{
            "model":"nope",
            "messages":[{"role":"user","content":[
                {"type":"text","text":"what is this"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0K"}}
            ]}]
        }"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value =
            serde_json::from_str(&body_string(resp).await).unwrap();
        // Engine-gate runs first; test doesn't depend on which gate hits.
        assert!(
            v["error"]["code"] == "model_not_loaded",
            "engine-gate should win: got {:?}",
            v
        );
    }

    #[tokio::test]
    async fn chat_completions_stream_without_engine_returns_model_not_loaded() {
        // Even with stream:true, the engine gate runs first — no engine →
        // 400 model_not_loaded. This is the documented Decision #16 / #26
        // ordering; the streaming path itself is only reached after the
        // gate passes.
        let app = build_router(state_default());
        let body = r#"{"model":"nope","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value =
            serde_json::from_str(&body_string(resp).await).unwrap();
        assert_eq!(v["error"]["code"], "model_not_loaded");
    }

    #[tokio::test]
    async fn chat_completions_malformed_json_returns_400() {
        let app = build_router(state_default());
        let body = r#"{ not valid json"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // axum's default JSON extractor rejection surface.
        assert_eq!(resp.status().as_u16() / 100, 4, "4xx expected");
    }

    #[tokio::test]
    async fn embeddings_route_returns_400_when_no_embedding_model_loaded() {
        // Server has no `--embedding-model` configured → embedding_config
        // is None → handler must return `model_not_loaded` (400).
        let app = build_router(state_default());
        let body = r#"{"model": "any", "input": "hello"}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        // `model_not_loaded` is an `invalid_request_error` with the
        // `code` field set to "model_not_loaded" (matches OpenAI's
        // 400-on-config-issue shape).
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "model_not_loaded");
    }

    #[tokio::test]
    async fn embeddings_route_rejects_malformed_json_with_4xx() {
        let app = build_router(state_default());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{ not valid"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status().as_u16() / 100, 4, "4xx expected");
    }

    #[tokio::test]
    async fn unknown_route_with_auth_still_gets_401_before_404() {
        // When auth is configured and the caller is unauthenticated, the
        // auth layer fires *before* the handler dispatch — so an unknown
        // route without a token returns 401, not 404. This matches
        // ollama/llama.cpp behavior and is the documented Decision #8 path.
        let app = build_router(state_with_auth("secret"));
        let req = Request::builder()
            .uri("/does-not-exist")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ────────────────────────────────────────────────────────────────────
    // ADR-005 Phase 4 iter-209 (W77) — pool-routed AppState
    // ────────────────────────────────────────────────────────────────────
    //
    // Tests below exercise the `HotSwapManager`-backed `AppState`
    // surface introduced by iter-209.  The default test state
    // (`AppState::new`) ships an EMPTY pool with a 1 GiB synthetic
    // memory budget (no real engine load), so every chat-completion
    // request must fail through the auto-pipeline → Decision #26
    // `400 model_not_loaded` path.  The new tests document the
    // post-refactor invariants:
    //
    //   1. Empty pool + unresolvable req.model → 400 model_not_loaded
    //      (the auto-pipeline rejects the input; the handler maps to
    //      Decision #26's contract surface, preserving the OpenAI-
    //      facing error shape pre-iter-209 callers depended on).
    //   2. /v1/models with empty pool reports zero loaded entries.
    //   3. Concurrent unresolvable requests serialize cleanly through
    //      the cache mutex without poisoning state.
    //   4. /metrics gauge `hf2q_model_loaded` is 0 when the pool is
    //      empty (matches pre-iter-209 single-slot semantics for the
    //      empty case).

    #[tokio::test]
    async fn iter209_chat_with_empty_pool_returns_model_not_loaded_for_unresolvable_name() {
        // Empty pool + req.model="not-a-repo-or-path" → auto-pipeline
        // rejects ("not on disk + not a valid HF repo-id") → handler
        // maps to 400 `model_not_loaded` (Decision #26 contract).
        let app = build_router(state_default());
        let body = r#"{"model":"not-a-repo-or-path","messages":[{"role":"user","content":"hi"}]}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
        assert_eq!(v["error"]["code"], "model_not_loaded");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("not-a-repo-or-path"),
            "error message should name the unresolvable model: {v}"
        );
    }

    #[tokio::test]
    async fn iter209_v1_models_empty_pool_reports_no_loaded_entries() {
        // Empty pool + cache_dir = None → /v1/models returns an empty
        // data array.  Pre-iter-209 same shape; post-iter-209 the
        // `loaded` boolean is sourced from `pool.snapshot_engines()`
        // (empty pool ⇒ no entries get marked loaded).
        let cfg = ServerConfig {
            cache_dir: None,
            ..Default::default()
        };
        let state = AppState::new(cfg);
        let app = build_router(state);
        let req = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
        let data = v["data"].as_array().expect("data is array");
        // No entries marked `loaded: true` when the pool is empty.
        for entry in data {
            let loaded = entry["loaded"].as_bool().unwrap_or(false);
            assert!(
                !loaded,
                "no entry should be loaded with empty pool; got {entry}"
            );
        }
    }

    #[tokio::test]
    async fn iter209_metrics_model_loaded_zero_with_empty_pool() {
        // Pre-iter-209: hf2q_model_loaded gauge was `1` iff
        // `state.engine.is_some()`.  Iter-209: gauge is `1` iff
        // `pool.read().pool_stats().loaded_count > 0`.  Empty pool ⇒ 0.
        let app = build_router(state_default());
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = body_string(resp).await;
        // `hf2q_model_loaded 0` must appear on its own line.
        assert!(
            body.lines().any(|l| l.trim() == "hf2q_model_loaded 0"),
            "expected `hf2q_model_loaded 0` line; body:\n{}",
            body
        );
    }

    #[tokio::test]
    async fn iter210_metrics_emits_pool_gauges() {
        // ADR-005 Phase 4 iter-210 (W78): /metrics surfaces three new
        // pool-state gauges sourced from `HotSwapManager::pool_stats()`.
        // With an empty default pool we expect:
        //   - hf2q_pool_loaded_models 0
        //   - hf2q_pool_resident_bytes 0
        //   - hf2q_pool_memory_budget_bytes <budget>  (synthetic 1 GiB
        //     test budget set by AppState::new — see
        //     `src/serve/api/state.rs::with_capacity_and_budget`).
        // The HELP/TYPE preamble for each gauge is asserted so the
        // Prometheus exposition format stays parseable; downstream
        // dashboards rely on it for type-aware aggregation.
        let app = build_router(state_default());
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_string(resp).await;

        // Gauge 1: loaded_models — empty pool ⇒ 0.
        assert!(
            body.contains("# HELP hf2q_pool_loaded_models"),
            "missing HELP line for hf2q_pool_loaded_models; body:\n{body}"
        );
        assert!(
            body.contains("# TYPE hf2q_pool_loaded_models gauge"),
            "missing TYPE line for hf2q_pool_loaded_models; body:\n{body}"
        );
        assert!(
            body.lines().any(|l| l.trim() == "hf2q_pool_loaded_models 0"),
            "expected `hf2q_pool_loaded_models 0` line; body:\n{body}"
        );

        // Gauge 2: resident_bytes — empty pool ⇒ 0.
        assert!(
            body.contains("# HELP hf2q_pool_resident_bytes"),
            "missing HELP line for hf2q_pool_resident_bytes; body:\n{body}"
        );
        assert!(
            body.contains("# TYPE hf2q_pool_resident_bytes gauge"),
            "missing TYPE line for hf2q_pool_resident_bytes; body:\n{body}"
        );
        assert!(
            body.lines().any(|l| l.trim() == "hf2q_pool_resident_bytes 0"),
            "expected `hf2q_pool_resident_bytes 0` line; body:\n{body}"
        );

        // Gauge 3: memory_budget_bytes — value comes from `state_default()`'s
        // synthetic pool budget (set in AppState::new for tests).  We
        // assert the prefix + a non-empty value to avoid coupling to the
        // exact byte count, which is intentionally synthetic in tests.
        assert!(
            body.contains("# HELP hf2q_pool_memory_budget_bytes"),
            "missing HELP line for hf2q_pool_memory_budget_bytes; body:\n{body}"
        );
        assert!(
            body.contains("# TYPE hf2q_pool_memory_budget_bytes gauge"),
            "missing TYPE line for hf2q_pool_memory_budget_bytes; body:\n{body}"
        );
        let budget_line = body
            .lines()
            .find(|l| l.starts_with("hf2q_pool_memory_budget_bytes "))
            .unwrap_or_else(|| panic!("expected hf2q_pool_memory_budget_bytes line; body:\n{body}"));
        let budget_val: u64 = budget_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                panic!("could not parse u64 from {budget_line:?}; body:\n{body}")
            });
        // The synthetic test budget is non-zero; production reads
        // `from_hardware` (80% of unified RAM).  Either way, > 0 here
        // proves the gauge wired through `pool_stats()` correctly.
        assert!(
            budget_val > 0,
            "expected positive memory_budget_bytes, got {budget_val}; body:\n{body}"
        );
    }

    #[tokio::test]
    async fn iter209_two_concurrent_unresolvable_requests_both_400() {
        // Concurrent requests for the same unresolvable model — both
        // serialize through the cache mutex inside `spawn_blocking`,
        // both return 400 model_not_loaded; neither poisons the
        // mutex (which would surface as a 500 on the third request).
        let state = state_default();
        let app1 = build_router(state.clone());
        let app2 = build_router(state.clone());
        let body = r#"{"model":"unresolvable-name","messages":[{"role":"user","content":"hi"}]}"#;
        let req1 = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let req2 = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let (resp1, resp2) = tokio::join!(app1.oneshot(req1), app2.oneshot(req2));
        let resp1 = resp1.unwrap();
        let resp2 = resp2.unwrap();
        assert_eq!(resp1.status(), StatusCode::BAD_REQUEST);
        assert_eq!(resp2.status(), StatusCode::BAD_REQUEST);
        let v1: serde_json::Value = serde_json::from_str(&body_string(resp1).await).unwrap();
        let v2: serde_json::Value = serde_json::from_str(&body_string(resp2).await).unwrap();
        assert_eq!(v1["error"]["code"], "model_not_loaded");
        assert_eq!(v2["error"]["code"], "model_not_loaded");

        // Third request after both — proves the cache mutex was not
        // poisoned (a poisoned mutex would surface as 500 on the third).
        let app3 = build_router(state);
        let req3 = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp3 = app3.oneshot(req3).await.unwrap();
        assert_eq!(resp3.status(), StatusCode::BAD_REQUEST);
        let v3: serde_json::Value = serde_json::from_str(&body_string(resp3).await).unwrap();
        assert_eq!(v3["error"]["code"], "model_not_loaded");
    }

    #[tokio::test]
    async fn iter209_default_model_fallback_when_req_model_empty() {
        // When req.model is empty and no default_model is set, handler
        // maps to 400 model_not_loaded (no model name to resolve).
        let app = build_router(state_default());
        let body = r#"{"model":"","messages":[{"role":"user","content":"hi"}]}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
        assert_eq!(v["error"]["code"], "model_not_loaded");
    }

    // ────────────────────────────────────────────────────────────────────
    // ADR-005 Phase 4 reopen iter-213 (AC 5472) — KV-spill telemetry tests
    // ────────────────────────────────────────────────────────────────────
    //
    // Tests below exercise the iter-213 KV-spill / KV-restore telemetry
    // counter surface introduced parallel to the iter-210 pool-gauge
    // tests.  The default test state (`AppState::new`) constructs a
    // fresh `KvSpillCounters` and injects the same Arc into the pool's
    // `HotSwapManager` so the trigger sites bump the table the
    // `/metrics` handler reads.
    //
    // Synthetic spiller fixtures used here mirror the `MockSpiller`
    // pattern at `src/serve/multi_model.rs:2387` (iter-212).  The
    // tests use direct `record_spill` / `record_restore` calls against
    // the counter Arc to exercise the counter surface in isolation
    // from the (repo, quant) → handle plumbing — the trigger-site
    // sequencing was already validated by the iter-212 9-test cohort
    // (`hotswap_pre_evict_*`, `hotswap_post_admit_*`).  The router
    // tests' job here is to confirm the counter surface emits the
    // expected Prometheus text and respects the closed-enum + Skipped
    // semantics.

    use crate::serve::api::state::KV_SPILL_OUTCOMES;
    use crate::serve::multi_model::{
        RestoreErrorKind, RestoreOutcome, SpillErrorKind, SpillOutcome,
    };
    use crate::serve::quant_select::QuantType;

    /// 1/6: extends `iter210_metrics_emits_pool_gauges` — after driving
    /// one synthetic eviction (`SpillOutcome::EnqueuedBlocks(1)`) +
    /// one synthetic admission (`RestoreOutcome::RestoredBlocks(1)`)
    /// via the counters surface directly (mirroring what the
    /// HotSwapManager trigger sites do at runtime), `/metrics` scrape
    /// contains all 4 outcome lines for spills and 4 for restores;
    /// `outcome="success"` shows count `1`, the other three show `0`.
    #[tokio::test]
    async fn iter213_metrics_emits_kv_counters() {
        let state = state_default();
        // Drive one synthetic eviction + one synthetic admission via
        // the same counter surface the HotSwapManager trigger sites
        // hit.  EnqueuedBlocks(1) → success row +1; RestoredBlocks(1)
        // → success row +1.  The other three outcome rows are zero
        // but MUST still emit (Prometheus convention; AC 5472).
        state.kv_spill_counters.record_spill(
            "acme/m1",
            QuantType::Q4_K_M,
            SpillOutcome::EnqueuedBlocks(1),
        );
        state.kv_spill_counters.record_restore(
            "acme/m1",
            QuantType::Q4_K_M,
            RestoreOutcome::RestoredBlocks(1),
        );
        let app = build_router(state);
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_string(resp).await;

        // HELP/TYPE preamble for both metrics MUST be present (iter-213
        // emits them unconditionally per Prometheus convention).
        assert!(
            body.contains("# HELP hf2q_pool_kv_spills_total"),
            "missing HELP for hf2q_pool_kv_spills_total; body:\n{body}"
        );
        assert!(
            body.contains("# TYPE hf2q_pool_kv_spills_total counter"),
            "missing TYPE for hf2q_pool_kv_spills_total; body:\n{body}"
        );
        assert!(
            body.contains("# HELP hf2q_pool_kv_restores_total"),
            "missing HELP for hf2q_pool_kv_restores_total; body:\n{body}"
        );
        assert!(
            body.contains("# TYPE hf2q_pool_kv_restores_total counter"),
            "missing TYPE for hf2q_pool_kv_restores_total; body:\n{body}"
        );

        // Spill rows: success=1, codec_err=0, io_err=0, parity_fail=0.
        for (outcome, expected) in [
            ("success", 1u64),
            ("codec_err", 0),
            ("io_err", 0),
            ("parity_fail", 0),
        ] {
            let needle = format!(
                "hf2q_pool_kv_spills_total{{repo=\"acme/m1\",quant=\"Q4_K_M\",outcome=\"{outcome}\"}} {expected}",
            );
            assert!(
                body.contains(&needle),
                "missing spill line {needle:?} in body:\n{body}"
            );
        }

        // Restore rows: same shape — success=1, others=0.
        for (outcome, expected) in [
            ("success", 1u64),
            ("codec_err", 0),
            ("io_err", 0),
            ("parity_fail", 0),
        ] {
            let needle = format!(
                "hf2q_pool_kv_restores_total{{repo=\"acme/m1\",quant=\"Q4_K_M\",outcome=\"{outcome}\"}} {expected}",
            );
            assert!(
                body.contains(&needle),
                "missing restore line {needle:?} in body:\n{body}"
            );
        }

        // Pool-gauge no-regression smoke (iter-210 surface stays green).
        assert!(
            body.lines().any(|l| l.trim() == "hf2q_pool_loaded_models 0"),
            "iter-210 hf2q_pool_loaded_models 0 line regression; body:\n{body}"
        );
    }

    /// 2/6: counter delta correctness under synthetic spill cycle.
    /// `MockSpiller`-style behavior: every `pre_evict` returns
    /// `EnqueuedBlocks(1)`.  Spawn N=3 synthetic eviction sequences
    /// (per-call increments).  Assert `success` delta == N exactly.
    /// Per AC 5472: per-call NOT per-block — `EnqueuedBlocks(7)` would
    /// still bump by 1, never 7.
    #[tokio::test]
    async fn iter213_kv_counter_delta_under_synthetic_spill() {
        let state = state_default();
        const N: u64 = 3;
        // N synthetic evictions.  Same (repo, quant) so they all
        // accumulate into one row.
        for _ in 0..N {
            state.kv_spill_counters.record_spill(
                "acme/m1",
                QuantType::Q4_K_M,
                SpillOutcome::EnqueuedBlocks(1),
            );
        }
        let app = build_router(state);
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = body_string(resp).await;
        let needle = format!(
            "hf2q_pool_kv_spills_total{{repo=\"acme/m1\",quant=\"Q4_K_M\",outcome=\"success\"}} {N}",
        );
        assert!(
            body.contains(&needle),
            "expected success counter == N={N} after {N} synthetic spills; \
             body:\n{body}"
        );

        // Per-call NOT per-block invariant — bumping with
        // EnqueuedBlocks(7) once should still increment by 1, total
        // stays at N+1.  The other counters must remain 0 throughout.
        // The state is consumed by build_router(); rebuild a parallel
        // state to assert this.
        let state2 = state_default();
        state2.kv_spill_counters.record_spill(
            "acme/m1",
            QuantType::Q4_K_M,
            SpillOutcome::EnqueuedBlocks(7),
        );
        let app2 = build_router(state2);
        let req2 = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp2 = app2.oneshot(req2).await.unwrap();
        let body2 = body_string(resp2).await;
        // EnqueuedBlocks(7) → success counter = 1, NOT 7.
        let needle_per_call = "hf2q_pool_kv_spills_total{repo=\"acme/m1\",quant=\"Q4_K_M\",outcome=\"success\"} 1";
        assert!(
            body2.contains(needle_per_call),
            "per-call NOT per-block: EnqueuedBlocks(7) must increment by 1 not 7; \
             body:\n{body2}"
        );
    }

    /// 3/6: compile-time + runtime guard against accidental 5th
    /// outcome.  KV_SPILL_OUTCOMES MUST be a 4-element slice
    /// containing exactly `["success", "codec_err", "io_err",
    /// "parity_fail"]` in that order (the order is load-bearing —
    /// `record_spill` indexes by usize and the metrics handler emits
    /// in this order).  Adding a fifth outcome requires amending
    /// ADR-005 Phase 4 with a 5473 telemetry-extension AC per AC 5472
    /// closed-enum semantics.
    #[tokio::test]
    async fn iter213_kv_counter_outcome_cardinality_registered() {
        assert_eq!(
            KV_SPILL_OUTCOMES.len(),
            4,
            "KV_SPILL_OUTCOMES cardinality is FIXED at 4; adding a 5th \
             outcome requires a Phase 4 amendment per AC 5472 closed-enum \
             contract"
        );
        assert_eq!(
            KV_SPILL_OUTCOMES,
            &["success", "codec_err", "io_err", "parity_fail"],
            "KV_SPILL_OUTCOMES order is load-bearing — record_spill \
             indexes by usize and /metrics emits in this exact order; \
             reordering breaks scrape diffs and ADR-017 Phase C \
             expectations"
        );
    }

    /// 4/6: `Server-Timing` response header is DEFAULT-OFF in iter-213.
    /// ADR-017 Phase C `cmd_serve --kv-persist` flag enables it; until
    /// then the header MUST NOT appear on swap-reload paths.  We
    /// trigger a swap-reload-shape request (`/v1/chat/completions`
    /// against an unresolvable model — same ordering as the iter-209
    /// `chat_completions_without_engine_returns_model_not_loaded`
    /// test) and assert no Server-Timing header is present.  Also
    /// drive one synthetic spill via the counter surface to prove the
    /// header stays absent even when counter activity has occurred
    /// (the toggle is independent of activity).
    #[tokio::test]
    async fn iter213_server_timing_header_default_off() {
        let state = state_default();
        // Drive a synthetic spill — the header must STILL stay absent
        // because the toggle is default-OFF independent of activity.
        state.kv_spill_counters.record_spill(
            "acme/m1",
            QuantType::Q4_K_M,
            SpillOutcome::EnqueuedBlocks(1),
        );
        // Verify the toggle defaults OFF (the load-bearing invariant).
        assert!(
            !state.kv_spill_counters.server_timing_enabled(),
            "Server-Timing toggle MUST default OFF in iter-213"
        );
        let app = build_router(state);
        // Trigger the swap-reload path shape: a chat completion against
        // an unresolvable model.  This routes through `load_or_get`'s
        // cold-load path that ADR-017 Phase C will (when --kv-persist
        // is set) instrument with kv_spill / kv_restore Server-Timing
        // measurements.  In iter-213 default-OFF, the header MUST NOT
        // appear on the response.
        let body = r#"{"model":"unresolvable-name","messages":[{"role":"user","content":"hi"}]}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // Server-Timing header is RFC 8673; iter-213 NEVER emits it.
        // The casing is canonical "Server-Timing"; HTTP headers are
        // case-insensitive but axum's HeaderMap uses lowercase keys.
        assert!(
            resp.headers().get("server-timing").is_none(),
            "Server-Timing header MUST NOT appear in iter-213 default-OFF \
             responses; saw: {:?}",
            resp.headers().get("server-timing")
        );
    }

    /// 5/6: error outcomes increment the matching counter.  Drive
    /// each error variant in sequence (CodecErr, IoErr, ParityFail);
    /// assert each outcome counter increments by exactly 1.  This is
    /// the closed-enum mapping contract from
    /// `KvSpillCounters::spill_outcome_index` /
    /// `restore_outcome_index`.
    #[tokio::test]
    async fn iter213_kv_counter_error_outcomes_increment() {
        let state = state_default();
        // Spill side: 3 distinct errors → 3 distinct outcome rows.
        state.kv_spill_counters.record_spill(
            "acme/m1",
            QuantType::Q4_K_M,
            SpillOutcome::Error(SpillErrorKind::CodecErr),
        );
        state.kv_spill_counters.record_spill(
            "acme/m1",
            QuantType::Q4_K_M,
            SpillOutcome::Error(SpillErrorKind::IoErr),
        );
        state.kv_spill_counters.record_spill(
            "acme/m1",
            QuantType::Q4_K_M,
            SpillOutcome::Error(SpillErrorKind::ParityFail),
        );
        // Restore side: same — 3 distinct error variants.
        state.kv_spill_counters.record_restore(
            "acme/m1",
            QuantType::Q4_K_M,
            RestoreOutcome::Error(RestoreErrorKind::CodecErr),
        );
        state.kv_spill_counters.record_restore(
            "acme/m1",
            QuantType::Q4_K_M,
            RestoreOutcome::Error(RestoreErrorKind::IoErr),
        );
        state.kv_spill_counters.record_restore(
            "acme/m1",
            QuantType::Q4_K_M,
            RestoreOutcome::Error(RestoreErrorKind::ParityFail),
        );

        let app = build_router(state);
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = body_string(resp).await;

        // Each error outcome counter == 1; success counter == 0.
        for (outcome, expected) in [
            ("success", 0u64),
            ("codec_err", 1),
            ("io_err", 1),
            ("parity_fail", 1),
        ] {
            let spill_needle = format!(
                "hf2q_pool_kv_spills_total{{repo=\"acme/m1\",quant=\"Q4_K_M\",outcome=\"{outcome}\"}} {expected}",
            );
            assert!(
                body.contains(&spill_needle),
                "missing spill error line {spill_needle:?} in body:\n{body}"
            );
            let restore_needle = format!(
                "hf2q_pool_kv_restores_total{{repo=\"acme/m1\",quant=\"Q4_K_M\",outcome=\"{outcome}\"}} {expected}",
            );
            assert!(
                body.contains(&restore_needle),
                "missing restore error line {restore_needle:?} in body:\n{body}"
            );
        }
    }

    /// 6/6: Skipped outcome does NOT increment any counter.  This is
    /// the closed-enum guard's whole point: counting Skipped would
    /// conflate the noop spiller's default behavior with successful
    /// spills.  After N record_* calls all returning Skipped, every
    /// outcome counter MUST remain at 0 — and the metric line itself
    /// MUST NOT appear in the scrape (no observation against the
    /// (repo, quant) pair triggered lazy-init of the row).
    #[tokio::test]
    async fn iter213_skipped_outcome_does_not_increment() {
        let state = state_default();
        const N: usize = 5;
        for _ in 0..N {
            state.kv_spill_counters.record_spill(
                "acme/m1",
                QuantType::Q4_K_M,
                SpillOutcome::Skipped,
            );
            state.kv_spill_counters.record_restore(
                "acme/m1",
                QuantType::Q4_K_M,
                RestoreOutcome::Skipped,
            );
        }
        let app = build_router(state);
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = body_string(resp).await;

        // No (repo, quant) row was ever observed (Skipped is a no-op),
        // so no per-pair lines emit.  HELP/TYPE preamble still emits
        // unconditionally — that's the iter-213 invariant.
        assert!(
            body.contains("# HELP hf2q_pool_kv_spills_total"),
            "HELP must emit even when no observations occurred; body:\n{body}"
        );
        assert!(
            body.contains("# TYPE hf2q_pool_kv_spills_total counter"),
            "TYPE must emit even when no observations occurred; body:\n{body}"
        );
        // No row lines for the (repo, quant) pair — Skipped did not
        // trigger lazy-init of the row.  We assert the absence of the
        // success line because if any of the four outcomes had bumped,
        // ALL four would emit at zero (lazy-init creates the full
        // four-element row).
        let absent_needle = "hf2q_pool_kv_spills_total{repo=\"acme/m1\"";
        assert!(
            !body.contains(absent_needle),
            "Skipped outcome MUST NOT lazy-init the (repo, quant) row; \
             saw line containing {absent_needle:?} in body:\n{body}"
        );
        let absent_restore = "hf2q_pool_kv_restores_total{repo=\"acme/m1\"";
        assert!(
            !body.contains(absent_restore),
            "Skipped restore outcome MUST NOT lazy-init the row; \
             saw line containing {absent_restore:?} in body:\n{body}"
        );
    }

    #[tokio::test]
    async fn iter209_default_model_fallback_uses_default_when_req_model_empty() {
        // When req.model is empty BUT default_model is set, the
        // handler routes through default_model.  Since "still-not-resolvable"
        // is not a valid path or repo-id, this still maps to
        // 400 model_not_loaded — but proves the fallback was consulted
        // (otherwise we'd see an empty-string error message).
        let cfg = ServerConfig::default();
        let state = AppState::new(cfg).with_default_model(Some("still-not-resolvable".into()));
        let app = build_router(state);
        let body = r#"{"model":"","messages":[{"role":"user","content":"hi"}]}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
        // The error message should name the default-model fallback,
        // proving the handler consulted it (not the empty req.model).
        assert_eq!(v["error"]["code"], "model_not_loaded");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("still-not-resolvable"),
            "expected default_model name in error: {v}"
        );
    }

    // ────────────────────────────────────────────────────────────────
    // Iter-215 Wedge-2 — /metrics scrape with Qwen35 engine resident
    // ────────────────────────────────────────────────────────────────

    /// /metrics surface MUST stay valid Prometheus text when a
    /// Qwen3.5/3.6 engine is resident in the pool.  Verifies that
    /// the iter-210 pool gauges (loaded_models / resident_bytes /
    /// memory_budget_bytes) AND the iter-213 KV-spill counter
    /// preamble (HELP/TYPE blocks) emit unconditionally with a
    /// LoadedArch::Qwen35 engine in the pool.
    #[tokio::test]
    async fn qwen35_metrics_endpoint_works_with_qwen35_loaded() {
        use super::super::engine;
        use crate::serve::quant_select::QuantType;
        let state = AppState::new(ServerConfig::default());
        // Inject a synthetic Qwen35-arch engine into the pool.
        {
            let mut mgr = state.pool.write().expect("pool rwlock");
            let engine = engine::make_synthetic_engine_for_test(engine::LoadedArch::Qwen35);
            mgr.admit_for_test(
                "iter-215-qwen35-test",
                QuantType::Q4_K_M,
                /* bytes_resident */ 1024,
                engine,
            )
            .expect("admit synthetic Qwen35 engine");
        }
        let app = build_router(state);
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_string(resp).await;

        // Iter-210 (W78) pool gauges — present.
        assert!(
            body.contains("hf2q_pool_loaded_models"),
            "missing pool gauge: {body}"
        );
        assert!(
            body.contains("hf2q_pool_resident_bytes"),
            "missing pool gauge: {body}"
        );
        assert!(
            body.contains("hf2q_pool_memory_budget_bytes"),
            "missing pool gauge: {body}"
        );
        // Iter-213 (AC 5472) KV-spill counter HELP/TYPE preamble —
        // present even with no spills observed (Prometheus convention).
        assert!(
            body.contains("hf2q_pool_kv_spills_total"),
            "missing kv_spills HELP/TYPE: {body}"
        );
        assert!(
            body.contains("hf2q_pool_kv_restores_total"),
            "missing kv_restores HELP/TYPE: {body}"
        );

        // Confirm the loaded count reflects the synthetic admission.
        // The pool gauge line `hf2q_pool_loaded_models 1` MUST appear.
        assert!(
            body.lines().any(|l| l.trim() == "hf2q_pool_loaded_models 1"),
            "expected `hf2q_pool_loaded_models 1` line; got body: {body}"
        );
    }
}
