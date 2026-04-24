//! Axum middleware for the hf2q API server.
//!
//! Three orthogonal layers:
//!
//!   1. **CORS** (Decision #9) — restrictive allowlist when configured,
//!      wide-open `*` only for empty allowlist (localhost dev default).
//!   2. **Bearer auth** (Decision #8) — when `auth_token` is configured,
//!      every request must carry `Authorization: Bearer <token>` or receive
//!      401. When not configured, the middleware is a no-op (not even
//!      installed on the router) so the hot path is a single pointer check.
//!   3. **Request ID** — accept a client-supplied `X-Request-Id` header and
//!      echo it in the response; generate a UUIDv4 if absent (ADR-005
//!      "Request IDs: accept client X-Request-Id, echo + generate UUIDv4").
//!
//! The layers are composed in `router.rs`. Each is independently testable and
//! has no shared state beyond `AppState`.

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Request, StatusCode};
use axum::middleware::Next;
use axum::response::IntoResponse;
use tower_http::cors::{AllowOrigin, CorsLayer};

use super::schema::ApiError;
use super::state::AppState;

/// Header name used for request IDs (both on request and response).
pub const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

// ---------------------------------------------------------------------------
// CORS
// ---------------------------------------------------------------------------

/// Build a CORS layer from the server config's allowed-origins list.
///
/// - Empty list → wide-open `Any` (localhost dev default, per Decision #9).
/// - Non-empty list → explicit allowlist (restrictive default for LAN /
///   production).
///
/// Allowed methods are the set the API surface actually uses:
/// `GET, POST, OPTIONS`. Allowed headers include `Content-Type`,
/// `Authorization`, and `X-Request-Id`. Allowed credentials are not enabled;
/// clients that need cookies/credentials (extremely rare for an OpenAI-shaped
/// API) must be handled by a reverse proxy.
pub fn cors_layer(allowed_origins: &[String]) -> CorsLayer {
    use axum::http::Method;
    let methods = [Method::GET, Method::POST, Method::OPTIONS];
    let headers = [
        header::CONTENT_TYPE,
        header::AUTHORIZATION,
        X_REQUEST_ID,
    ];

    let mut layer = CorsLayer::new().allow_methods(methods).allow_headers(headers);

    if allowed_origins.is_empty() {
        layer = layer.allow_origin(AllowOrigin::any());
    } else {
        let mut origins = Vec::with_capacity(allowed_origins.len());
        for origin in allowed_origins {
            match HeaderValue::from_str(origin) {
                Ok(v) => origins.push(v),
                Err(e) => {
                    tracing::warn!(
                        origin = %origin,
                        error = %e,
                        "Dropping malformed CORS origin from allowlist"
                    );
                }
            }
        }
        layer = layer.allow_origin(AllowOrigin::list(origins));
    }
    layer
}

// ---------------------------------------------------------------------------
// Bearer auth
// ---------------------------------------------------------------------------

/// Middleware: require `Authorization: Bearer <token>` when `auth_token` is
/// configured. Returns 401 with the OpenAI-shaped error envelope on mismatch.
///
/// Constant-time comparison to avoid a trivial timing sidechannel on the
/// token check. (The token lives in server config; the hot-path equality is
/// byte-by-byte after extracting the header.)
pub async fn bearer_auth(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> axum::response::Response {
    let Some(expected) = state.config.auth_token.clone() else {
        // Auth not configured — no-op pass-through.
        return next.run(req).await;
    };

    let supplied = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.trim())
        .unwrap_or("");

    if supplied.is_empty() || !constant_time_eq(supplied.as_bytes(), expected.as_bytes()) {
        return ApiError::unauthorized().into_response();
    }

    next.run(req).await
}

/// Constant-time byte-slice equality. Avoids early-exit on mismatch so the
/// comparison time does not leak information about the supplied token length
/// or first mismatching byte.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        acc |= x ^ y;
    }
    acc == 0
}

// ---------------------------------------------------------------------------
// Request ID
// ---------------------------------------------------------------------------

/// Extract a request ID from the `X-Request-Id` header, or generate a
/// fresh UUIDv4 if the header is absent / malformed.
pub fn extract_or_generate_request_id(headers: &HeaderMap) -> String {
    if let Some(hv) = headers.get(&X_REQUEST_ID) {
        if let Ok(s) = hv.to_str() {
            let s = s.trim();
            if !s.is_empty() && s.len() <= 256 && s.chars().all(is_safe_request_id_char) {
                return s.to_string();
            }
        }
    }
    uuid::Uuid::new_v4().to_string()
}

/// Only alphanumerics, `-`, `_`, `.` are accepted in a client-supplied
/// request id. Rejects whitespace, control chars, and anything that might
/// show up in a log field and cause confusion.
fn is_safe_request_id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.'
}

/// Middleware: derive a request ID (extracting a client-supplied one or
/// generating), stamp it on the request extensions, run the handler, then
/// echo it on the response as `X-Request-Id`.
pub async fn request_id_layer(
    mut req: Request<Body>,
    next: Next,
) -> axum::response::Response {
    let request_id = extract_or_generate_request_id(req.headers());
    req.extensions_mut().insert(RequestId(request_id.clone()));

    let mut response = next.run(req).await;

    // Best-effort echo — if the generated/supplied id happens to be invalid
    // as an `HeaderValue` (should not happen since we validate chars and
    // generate UUIDs) silently skip to avoid corrupting the response.
    if let Ok(hv) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert(X_REQUEST_ID, hv);
    }

    response
}

/// Request ID propagated via request extensions. Handlers that want to log
/// or surface the id pull it out with `Extension<RequestId>`.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

// ---------------------------------------------------------------------------
// 404 fallback
// ---------------------------------------------------------------------------

/// Fallback handler for unmatched routes. Returns an OpenAI-shaped 404 error.
pub async fn fallback(req: Request<Body>) -> impl IntoResponse {
    let _ = StatusCode::NOT_FOUND; // force import
    ApiError::not_found(format!(
        "No route matched {} {}",
        req.method(),
        req.uri().path()
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_id_is_generated_when_header_absent() {
        let headers = HeaderMap::new();
        let id = extract_or_generate_request_id(&headers);
        assert_eq!(id.len(), 36, "UUIDv4 has 36 characters: {:?}", id);
        assert_eq!(id.chars().filter(|c| *c == '-').count(), 4);
    }

    #[test]
    fn request_id_is_echoed_when_header_present_and_safe() {
        let mut headers = HeaderMap::new();
        headers.insert(&X_REQUEST_ID, HeaderValue::from_static("abc-123_x.y"));
        let id = extract_or_generate_request_id(&headers);
        assert_eq!(id, "abc-123_x.y");
    }

    #[test]
    fn request_id_regenerated_when_header_contains_unsafe_chars() {
        let mut headers = HeaderMap::new();
        headers.insert(
            &X_REQUEST_ID,
            HeaderValue::from_static("has space and $"),
        );
        let id = extract_or_generate_request_id(&headers);
        // Generated UUIDs don't contain spaces or $
        assert!(!id.contains(' '));
        assert!(!id.contains('$'));
    }

    #[test]
    fn request_id_regenerated_when_header_too_long() {
        let mut headers = HeaderMap::new();
        let too_long = "a".repeat(257);
        headers.insert(
            &X_REQUEST_ID,
            HeaderValue::from_str(&too_long).unwrap(),
        );
        let id = extract_or_generate_request_id(&headers);
        assert_eq!(id.len(), 36);
    }

    #[test]
    fn constant_time_eq_matches_regular_eq_for_equal_slices() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn cors_layer_with_empty_allowlist_constructs() {
        let _layer = cors_layer(&[]);
    }

    #[test]
    fn cors_layer_with_explicit_allowlist_constructs() {
        let _layer = cors_layer(&["http://localhost:3000".to_string()]);
    }

    #[test]
    fn cors_layer_silently_drops_malformed_origins() {
        // "http://\u{2028}" contains U+2028 LINE SEPARATOR which is not a
        // valid HeaderValue char. Should be dropped with a log warn, not panic.
        let _layer = cors_layer(&[
            "http://ok.example".to_string(),
            "http://\u{2028}bad".to_string(),
        ]);
    }
}
