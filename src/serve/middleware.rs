//! Middleware configuration for the API server.
//!
//! CORS is configured permissively to allow any origin, matching
//! local inference use cases where the server is accessed from
//! browser-based tools like Open WebUI.

use tower_http::cors::CorsLayer;

/// Build the CORS middleware layer.
///
/// Allows all origins, methods, and headers. This matches the local
/// inference use case where the server is accessed from various tools
/// (Open WebUI, Cursor, Continue) running on localhost.
pub fn cors_layer() -> CorsLayer {
    CorsLayer::permissive()
}
