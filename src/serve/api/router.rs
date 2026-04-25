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
        let mut cfg = ServerConfig::default();
        cfg.auth_token = Some(token.to_string());
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
        let mut cfg = ServerConfig::default();
        cfg.cache_dir = None;
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
        let mut cfg = ServerConfig::default();
        cfg.cache_dir = None;
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
}
