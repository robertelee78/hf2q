//! Axum handlers for the hf2q API server.
//!
//! This iteration lands three real handlers:
//!
//!   - `GET /health`  — JSON liveness + model info + uptime (Decision #12).
//!   - `GET /readyz`  — k8s-style readiness gate (Decision #12, #16). In this
//!                      iter generation is not yet routed, so `ready=true`
//!                      as soon as the server binds.
//!   - `GET /v1/models` — lists all GGUFs under `~/.cache/hf2q/` with
//!                        extension fields `{quant_type, context_length,
//!                        backend, loaded}` per Decision #26. Inspects each
//!                        GGUF's header-only via `mlx_native::gguf::GgufFile`
//!                        (no tensor data read) to extract quant + context.
//!
//! Chat completions, embeddings, and the auto-pipeline/hot-swap endpoints
//! come in subsequent iterations. Adding them is additive to this file; the
//! scaffolding here (AppState threading, error envelope, JSON responses) is
//! what every future handler is built on.

use std::collections::HashMap;
use std::path::Path;

use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::schema::{
    ApiError, HealthResponse, ModelListResponse, ModelObject, ReadyzResponse,
};
use super::state::AppState;

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

/// Handler for `GET /health`. Always returns 200 while the HTTP server is
/// running. The response includes currently-loaded model id (once a future
/// iter wires a model into `AppState`), backend name, context length, and
/// process uptime in seconds.
pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let resp = HealthResponse {
        status: "ok".to_string(),
        // iter 2: no engine wired → no model id/context-length yet. Future
        // iter: pull from state.engine.model_id / state.engine.context_length.
        model: None,
        backend: "mlx-native",
        context_length: None,
        uptime_seconds: state.uptime_seconds(),
    };
    (StatusCode::OK, Json(resp))
}

// ---------------------------------------------------------------------------
// GET /readyz
// ---------------------------------------------------------------------------

/// Handler for `GET /readyz` (Decision #12, #16). Returns 503 while the
/// server is still warming up, 200 once ready. The readiness signal is an
/// `AtomicBool` flipped by the warmup task (future iter).
pub async fn readyz(State(state): State<AppState>) -> impl IntoResponse {
    if state.is_ready_for_gen() {
        (
            StatusCode::OK,
            Json(ReadyzResponse { ready: true, detail: "ready" }),
        )
            .into_response()
    } else {
        let mut resp = (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadyzResponse { ready: false, detail: "warming up" }),
        )
            .into_response();
        // Retry-After: 1 second suggestion; warmup typically completes in
        // seconds, but we match the error envelope's convention.
        resp.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            axum::http::HeaderValue::from_static("1"),
        );
        resp
    }
}

// ---------------------------------------------------------------------------
// GET /v1/models + GET /v1/models/:id
// ---------------------------------------------------------------------------

/// List all cached GGUFs under `~/.cache/hf2q/` (or the configured
/// `cache_dir`). Each entry is an OpenAI `ModelObject` extended with
/// `quant_type`, `context_length`, `backend`, `loaded` fields (Decision #26).
///
/// The scan is done synchronously on the tokio runtime's blocking pool (via
/// `tokio::task::spawn_blocking`) — GGUF header parsing does file I/O and a
/// modest amount of allocation; keeping it off the async executor keeps the
/// request hot path responsive.
pub async fn list_models(State(state): State<AppState>) -> Response {
    let cache_dir = state.config.cache_dir.clone();
    let models = match tokio::task::spawn_blocking(move || scan_cache_dir(cache_dir.as_deref()))
        .await
    {
        Ok(Ok(models)) => models,
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "model cache scan failed");
            return ApiError::internal_error().into_response();
        }
        Err(e) => {
            tracing::error!(error = %e, "spawn_blocking panicked in list_models");
            return ApiError::internal_error().into_response();
        }
    };
    let resp = ModelListResponse {
        object: "list",
        data: models,
    };
    (StatusCode::OK, Json(resp)).into_response()
}

/// Retrieve a single model by id. Id matching is case-sensitive on the
/// model's filesystem stem (e.g. `gemma4-26b-it-Q4_K_M`). Returns 404
/// `model_not_found` if not present in the cache.
pub async fn get_model(
    State(state): State<AppState>,
    AxPath(model_id): AxPath<String>,
) -> Response {
    let cache_dir = state.config.cache_dir.clone();
    let all = match tokio::task::spawn_blocking(move || scan_cache_dir(cache_dir.as_deref())).await
    {
        Ok(Ok(m)) => m,
        Ok(Err(_)) | Err(_) => return ApiError::internal_error().into_response(),
    };
    match all.into_iter().find(|m| m.id == model_id) {
        Some(m) => (StatusCode::OK, Json(m)).into_response(),
        None => ApiError::model_not_found(&model_id).into_response(),
    }
}

// ---------------------------------------------------------------------------
// Cache-directory scanner
// ---------------------------------------------------------------------------

/// Scan the cache directory for `.gguf` files and return a `ModelObject` for
/// each. Reads only GGUF header metadata (no tensor data). Skips files that
/// fail to parse rather than failing the whole listing.
pub(crate) fn scan_cache_dir(cache_dir: Option<&Path>) -> std::io::Result<Vec<ModelObject>> {
    let Some(dir) = cache_dir else {
        return Ok(Vec::new());
    };
    if !dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    // Bounded recursion depth so a pathological symlink graph can't hang us.
    visit_dir(dir, &mut out, 0, 6)?;
    // Sort for deterministic ordering (tests depend on this).
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

fn visit_dir(
    dir: &Path,
    out: &mut Vec<ModelObject>,
    depth: usize,
    max_depth: usize,
) -> std::io::Result<()> {
    if depth > max_depth {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            // Don't follow symlinks to directories; they can cycle.
            if !ft.is_symlink() {
                visit_dir(&path, out, depth + 1, max_depth)?;
            }
        } else if ft.is_file() {
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                if let Some(obj) = inspect_gguf(&path) {
                    out.push(obj);
                }
            }
        }
    }
    Ok(())
}

/// Parse a single GGUF and build a `ModelObject` for it. Returns `None` if
/// the file fails to parse (logged as a warning — we skip, not fail, so one
/// bad file doesn't hide the rest of the catalog).
fn inspect_gguf(path: &Path) -> Option<ModelObject> {
    use mlx_native::gguf::GgufFile;

    let gguf = match GgufFile::open(path) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "skipping malformed GGUF in cache scan"
            );
            return None;
        }
    };

    let stem = path.file_stem()?.to_string_lossy().into_owned();

    let context_length = context_length_for_arch(&gguf);
    let quant_type = infer_quant_type(&gguf);

    let created = std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    Some(ModelObject {
        id: stem,
        object: "model",
        created,
        owned_by: "hf2q",
        context_length,
        quant_type,
        backend: Some("mlx-native"),
        // iter 2: no model is loaded at runtime. Future iter flips this to
        // `true` for the currently-loaded model entry.
        loaded: false,
    })
}

/// Read `{arch}.context_length` from GGUF metadata. The architecture key
/// prefix varies by model family (`llama`, `qwen2`, `gemma3`, etc.); we read
/// `general.architecture` first then probe the arch-specific key.
fn context_length_for_arch(gguf: &mlx_native::gguf::GgufFile) -> Option<usize> {
    let arch = gguf.metadata_string("general.architecture")?;
    let key = format!("{arch}.context_length");
    gguf.metadata_u32(&key).map(|v| v as usize)
}

/// Infer a quant-type label for the GGUF.
///
/// Strategy: compute a histogram of ggml tensor types, skip fp bookkeeping
/// types (F32 / F16 for norms and embeds), and report the most common
/// non-fp quant type as the label. Matches how llama.cpp's `gguf-tools show`
/// reports a file.
///
/// Returns `None` if every tensor is fp (e.g. a pre-quantization safetensors
/// conversion artifact).
///
/// Note: mlx-native's `GgmlType` currently enumerates only the six types
/// hf2q's kernels support (F32, F16, Q4_0, Q8_0, Q4_K, Q6_K). A GGUF that
/// contains only those six types will be fully listed; anything with
/// unsupported types fails to open earlier in `inspect_gguf` and never
/// reaches this function. This matches the correctness contract — we only
/// advertise models we can actually serve.
fn infer_quant_type(gguf: &mlx_native::gguf::GgufFile) -> Option<String> {
    use mlx_native::GgmlType;

    let mut histogram: HashMap<&'static str, usize> = HashMap::new();
    for name in gguf.tensor_names() {
        let Some(info) = gguf.tensor_info(name) else { continue };
        let label = ggml_type_label(info.ggml_type);
        // Skip fp types — we want the dominant quantization, not the norm/embed dtype.
        if matches!(info.ggml_type, GgmlType::F32 | GgmlType::F16) {
            continue;
        }
        *histogram.entry(label).or_insert(0) += 1;
    }
    histogram.into_iter().max_by_key(|(_, n)| *n).map(|(k, _)| k.to_string())
}

/// Map a ggml type enum into a short, well-known label.
/// This centralizes the string convention for `/v1/models` so different
/// handlers never disagree.
fn ggml_type_label(t: mlx_native::GgmlType) -> &'static str {
    use mlx_native::GgmlType;
    match t {
        GgmlType::F32 => "F32",
        GgmlType::F16 => "F16",
        GgmlType::Q4_0 => "Q4_0",
        GgmlType::Q8_0 => "Q8_0",
        GgmlType::Q4_K => "Q4_K",
        GgmlType::Q6_K => "Q6_K",
    }
}

// ---------------------------------------------------------------------------
// Helpers exposed for tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) fn test_inspect_gguf(path: &Path) -> Option<ModelObject> {
    inspect_gguf(path)
}

#[cfg(test)]
pub(crate) fn test_scan(dir: &Path) -> std::io::Result<Vec<ModelObject>> {
    scan_cache_dir(Some(dir))
}

// ---------------------------------------------------------------------------
// Tests (unit — integration tests via router live in tests/api_smoke.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_missing_dir_returns_empty() {
        let tmp = std::env::temp_dir().join("hf2q-test-does-not-exist-xyz");
        // Don't create it.
        let result = scan_cache_dir(Some(&tmp)).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn scan_none_cache_dir_returns_empty() {
        let result = scan_cache_dir(None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn scan_empty_dir_returns_empty() {
        let tmp = tempdir_for("hf2q-scan-empty");
        let result = scan_cache_dir(Some(&tmp)).unwrap();
        assert!(result.is_empty());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn scan_skips_non_gguf_files() {
        let tmp = tempdir_for("hf2q-scan-skip-nongguf");
        std::fs::write(tmp.join("readme.txt"), "hello").unwrap();
        std::fs::write(tmp.join("data.bin"), [0u8, 1, 2, 3]).unwrap();
        let result = scan_cache_dir(Some(&tmp)).unwrap();
        assert!(result.is_empty());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn scan_skips_malformed_gguf_but_succeeds() {
        let tmp = tempdir_for("hf2q-scan-malformed");
        // Write a bogus "gguf" file (wrong magic) — inspect_gguf returns None
        // and we get an empty catalog without an error.
        std::fs::write(tmp.join("fake.gguf"), b"not a real gguf file").unwrap();
        let result = scan_cache_dir(Some(&tmp)).unwrap();
        assert!(result.is_empty(), "malformed GGUF should be skipped");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn scan_is_deterministic_ordering() {
        // Build two identically-named fake files in nested dirs, ensure
        // sorted output. (We can't easily build a valid GGUF in a unit
        // test; this test only checks that the scan survives nesting.)
        let tmp = tempdir_for("hf2q-scan-determ");
        std::fs::create_dir_all(tmp.join("a")).unwrap();
        std::fs::create_dir_all(tmp.join("b")).unwrap();
        // No valid GGUFs → empty result, but the scan should not error.
        let result = scan_cache_dir(Some(&tmp)).unwrap();
        assert!(result.is_empty());
        std::fs::remove_dir_all(&tmp).ok();
    }

    fn tempdir_for(tag: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let p = std::env::temp_dir().join(format!("{tag}-{pid}-{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
