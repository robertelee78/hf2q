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

use super::engine::{self, SamplingParams};
use super::schema::{
    ApiError, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    HealthResponse, MessageContent, ModelListResponse, ModelObject, PromptTokensDetails,
    ReadyzResponse, UsageStats,
};
use super::state::AppState;

// ---------------------------------------------------------------------------
// POST /v1/chat/completions (non-streaming)
// ---------------------------------------------------------------------------

/// Handler for `POST /v1/chat/completions` — non-streaming path.
///
/// Flow:
///   1. If no engine is loaded: return 400 `model_not_loaded` (Decision #26).
///   2. If not warmed up yet: return 503 `not_ready` + `Retry-After: 1`.
///   3. If `request.model` doesn't match the loaded engine: 400
///      `model_not_loaded` naming the mismatched id.
///   4. Render the chat template over the OpenAI messages array.
///   5. Tokenize, enforce context budget (max_tokens capped by
///      `engine.context_length - prompt_len`; TODO iter 4: overflow policy).
///   6. Build `SamplingParams` from request tier 1+2+3 fields.
///   7. Call `engine.generate(...)` — returns after the worker thread
///      completes the decode.
///   8. Wrap result in OpenAI `ChatCompletionResponse`.
///
/// Streaming (SSE) lands in the next iter; this path only handles
/// `stream: false` or absent. A request with `stream: true` returns 400
/// pointing to the iter-4 placeholder until streaming lands.
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Shared prelude: engine gate, model-id match, chat-template render,
    // tokenize, sampling-params build. Returns either a prepared context or
    // an error response to return directly.
    let prepared = match prepare_chat_generation(&state, &req) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    // --- Dispatch streaming vs non-streaming ---
    if req.stream.unwrap_or(false) {
        return chat_completions_stream(state.clone(), req, prepared).await;
    }

    let PreparedChatContext {
        engine: _engine_ref,
        prompt_tokens,
        params,
    } = prepared;
    let engine = state.engine.as_ref().expect("engine gate above ensures Some");

    // --- Generate (non-streaming) ---
    let gen_started = std::time::Instant::now();
    let result = match engine.generate(prompt_tokens, params).await {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("{e}");
            // Distinguish queue_full (→ 429) from other engine errors (→ 500).
            if msg.contains("queue_full") {
                return ApiError::queue_full().into_response();
            }
            tracing::error!(error = %msg, "chat_completion generation failed");
            return ApiError::generation_error(msg).into_response();
        }
    };
    let total_time = gen_started.elapsed();

    // --- Wrap in OpenAI response envelope ---
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono_seconds();
    let system_fingerprint = state.config.system_fingerprint.clone();

    let prefill_time_secs = result.prefill_duration.as_secs_f64();
    let decode_time_secs = result.decode_duration.as_secs_f64();
    let total_time_secs = total_time.as_secs_f64();
    let prefill_tokens_per_sec = if prefill_time_secs > 0.0 {
        result.prompt_tokens as f64 / prefill_time_secs
    } else {
        0.0
    };
    let decode_tokens_per_sec = if decode_time_secs > 0.0 {
        result.completion_tokens as f64 / decode_time_secs
    } else {
        0.0
    };
    let ttft_ms = prefill_time_secs * 1000.0;
    let timing = super::schema::TimingInfo {
        prefill_time_secs,
        decode_time_secs,
        total_time_secs,
        time_to_first_token_ms: ttft_ms,
        prefill_tokens_per_sec,
        decode_tokens_per_sec,
        gpu_sync_count: 0,
        gpu_dispatch_count: 0,
    };

    let resp = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created,
        model: req.model.clone(),
        system_fingerprint,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: Some(MessageContent::Text(result.text)),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            finish_reason: result.finish_reason.to_string(),
            logprobs: None,
        }],
        usage: UsageStats {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.prompt_tokens + result.completion_tokens,
            // Prompt caching (Decision #24) lands with Task #7 — no cached
            // tokens reported until then.
            prompt_tokens_details: Some(PromptTokensDetails { cached_tokens: 0 }),
            completion_tokens_details: None,
        },
        x_hf2q_timing: Some(timing),
    };
    (StatusCode::OK, Json(resp)).into_response()
}

/// Everything the non-streaming + streaming paths both need to start the
/// generation on the worker thread.
struct PreparedChatContext {
    /// Kept for symmetry; the actual engine ref is re-fetched from `state`
    /// at the call site because the async streaming body needs to take a
    /// `'static` clone of the handle.
    engine: (),
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
}

/// Run every validation + rendering + tokenization step common to the
/// streaming and non-streaming chat-completion paths. Returns the prepared
/// context on success, or an `axum::Response` to return directly on failure.
fn prepare_chat_generation(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> std::result::Result<PreparedChatContext, Response> {
    // --- Engine gate ---
    let Some(engine) = state.engine.as_ref() else {
        return Err(ApiError::model_not_loaded(&req.model).into_response());
    };
    if !state.is_ready_for_gen() {
        return Err(ApiError::not_ready().into_response());
    }
    if req.model != engine.model_id() {
        return Err(ApiError::model_not_loaded(&req.model).into_response());
    }
    if req.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "messages must contain at least one entry",
            Some("messages".into()),
        )
        .into_response());
    }
    let rendered =
        match engine::render_chat_prompt(engine.chat_template(), &req.messages) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "chat template render failed");
                return Err(ApiError::invalid_request(
                    format!("chat template render failed: {e}"),
                    Some("messages".into()),
                )
                .into_response());
            }
        };
    let encoding = match engine.tokenizer().encode(rendered.as_str(), false) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(error = %e, "tokenization failed");
            return Err(ApiError::internal_error().into_response());
        }
    };
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();
    if let Some(ctx_len) = engine.context_length() {
        if prompt_len >= ctx_len {
            return Err(
                ApiError::context_length_exceeded(ctx_len, prompt_len).into_response(),
            );
        }
    }
    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(SamplingParams::default().max_tokens);
    let stop_strings = req
        .stop
        .clone()
        .map(|s| s.into_vec())
        .unwrap_or_default();
    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.map(|v| v as usize).unwrap_or(0),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
        max_tokens,
        stop_strings,
    };
    Ok(PreparedChatContext {
        engine: (),
        prompt_tokens,
        params,
    })
}

/// Streaming chat-completion path. Opens an `mpsc::channel(64)`, hands the
/// sender to the engine worker, and wraps the receiver in the SSE encoder
/// built at `src/serve/api/sse.rs::generation_events_to_sse`. If the engine
/// queue is full, returns 429 + Retry-After instead of starting the stream.
async fn chat_completions_stream(
    state: AppState,
    req: ChatCompletionRequest,
    prepared: PreparedChatContext,
) -> Response {
    use super::sse::{generation_events_to_sse, SseStreamOptions};

    let engine = match state.engine.as_ref() {
        Some(e) => e.clone(),
        None => return ApiError::model_not_loaded(&req.model).into_response(),
    };

    let (events_tx, events_rx) = tokio::sync::mpsc::channel(64);
    if let Err(e) = engine
        .generate_stream(prepared.prompt_tokens, prepared.params, events_tx)
        .await
    {
        let msg = format!("{e}");
        if msg.contains("queue_full") {
            return ApiError::queue_full().into_response();
        }
        tracing::error!(error = %msg, "chat_completions_stream enqueue failed");
        return ApiError::generation_error(msg).into_response();
    }

    // SSE options: include_usage follows Decision #22 (Tier 2 — `stream_options.include_usage`).
    // logprobs follow Tier 4 — the grammar-aware sampler will feed Logprobs
    // events in the iter that lands top_logprobs.
    let opts = SseStreamOptions {
        include_usage: req
            .stream_options
            .as_ref()
            .and_then(|s| s.include_usage)
            .unwrap_or(false),
        logprobs: req.logprobs.unwrap_or(false),
        system_fingerprint: state.config.system_fingerprint.clone(),
    };

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono_seconds();
    let sse = generation_events_to_sse(events_rx, request_id, req.model, created, opts);
    sse.into_response()
}

/// Current Unix epoch seconds (used for response `created` field).
fn chrono_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

/// Handler for `GET /health`. Always returns 200 while the HTTP server is
/// running. The response includes currently-loaded model id (when an engine
/// is wired into `AppState`), backend name, context length, and process
/// uptime in seconds.
pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let (model, context_length) = match state.engine.as_ref() {
        Some(e) => (Some(e.model_id().to_string()), e.context_length()),
        None => (None, None),
    };
    let resp = HealthResponse {
        status: "ok".to_string(),
        model,
        backend: "mlx-native",
        context_length,
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
    let mut models =
        match tokio::task::spawn_blocking(move || scan_cache_dir(cache_dir.as_deref())).await {
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
    // Mark the currently-loaded model as loaded=true. If the loaded model
    // isn't in the cache catalog (e.g. loaded from a path outside
    // ~/.cache/hf2q/), prepend it.
    if let Some(engine) = state.engine.as_ref() {
        let loaded_id = engine.model_id().to_string();
        match models.iter_mut().find(|m| m.id == loaded_id) {
            Some(m) => m.loaded = true,
            None => {
                models.insert(
                    0,
                    ModelObject {
                        id: loaded_id,
                        object: "model",
                        created: chrono_seconds(),
                        owned_by: "hf2q",
                        context_length: engine.context_length(),
                        quant_type: engine.quant_type().map(|s| s.to_string()),
                        backend: Some("mlx-native"),
                        loaded: true,
                    },
                );
            }
        }
    }
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
        GgmlType::Q5_K => "Q5_K",
        GgmlType::Q6_K => "Q6_K",
        GgmlType::I16 => "I16",
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
