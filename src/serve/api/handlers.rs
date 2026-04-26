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
use super::grammar;
use super::schema::{
    ApiError, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    EmbeddingObject, EmbeddingPayload, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    HealthResponse, MessageContent, ModelListResponse, ModelObject, PromptTokensDetails,
    ReadyzResponse, ResponseFormat, UsageStats,
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
    use std::sync::atomic::Ordering;
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);

    // Shared prelude: engine gate, model-id match, chat-template render,
    // tokenize, sampling-params build. Returns either a prepared context or
    // an error response to return directly.
    let prepared = match prepare_chat_generation(&state, &req).await {
        Ok(p) => p,
        Err(resp) => {
            state.metrics.requests_rejected_total.fetch_add(1, Ordering::Relaxed);
            return resp;
        }
    };
    state.metrics.chat_completions_started.fetch_add(1, Ordering::Relaxed);

    // --- Dispatch streaming vs non-streaming ---
    if req.stream.unwrap_or(false) {
        return chat_completions_stream(state.clone(), req, prepared).await;
    }

    let PreparedChatContext {
        engine: _engine_ref,
        prompt_tokens,
        params,
        summarized_messages,
        summary_tokens,
        soft_tokens,
        vit_forward_ms,
        vit_images,
    } = prepared;
    let engine = state.engine.as_ref().expect("engine gate above ensures Some");

    // --- Generate (non-streaming) ---
    // Snapshot mlx-native process-global GPU counters pre-generation so we
    // can report the per-request delta in x_hf2q_timing.
    let pre_dispatches = mlx_native::dispatch_count();
    let pre_syncs = mlx_native::sync_count();
    let gen_started = std::time::Instant::now();
    // Vision dispatch fork (Phase 2c Task #17 / iter-99): when the
    // multimodal preflight produced soft-token overrides, route through
    // `Engine::generate_with_soft_tokens` so the worker plugs the
    // projected ViT embeddings into the prefill at every placeholder
    // position.  Empty `soft_tokens` ⇒ pure-text path (unchanged
    // `Engine::generate`).
    let gen_outcome = if soft_tokens.is_empty() {
        engine.generate(prompt_tokens, params).await
    } else {
        engine
            .generate_with_soft_tokens(prompt_tokens, soft_tokens, params)
            .await
    };
    let result = match gen_outcome {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("{e}");
            // Distinguish queue_full (→ 429) from other engine errors (→ 500).
            if msg.contains("queue_full") {
                state.metrics.chat_completions_queue_full.fetch_add(1, Ordering::Relaxed);
                return queue_full_with_rate_limit_headers(&state);
            }
            tracing::error!(error = %msg, "chat_completion generation failed");
            return ApiError::generation_error(msg).into_response();
        }
    };
    let total_time = gen_started.elapsed();
    state.metrics.chat_completions_completed.fetch_add(1, Ordering::Relaxed);
    state.metrics.prompt_tokens_total.fetch_add(result.prompt_tokens as u64, Ordering::Relaxed);
    state.metrics.decode_tokens_total.fetch_add(result.completion_tokens as u64, Ordering::Relaxed);

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
    let post_dispatches = mlx_native::dispatch_count();
    let post_syncs = mlx_native::sync_count();
    let timing = super::schema::TimingInfo {
        prefill_time_secs,
        decode_time_secs,
        total_time_secs,
        time_to_first_token_ms: ttft_ms,
        prefill_tokens_per_sec,
        decode_tokens_per_sec,
        gpu_sync_count: post_syncs.saturating_sub(pre_syncs),
        gpu_dispatch_count: post_dispatches.saturating_sub(pre_dispatches),
    };

    // Per-token reasoning count comes straight from the engine (it runs the
    // same ReasoningSplitter the streaming path uses). Fall back to None
    // when the model has no reasoning markers registered.
    let reasoning_tokens = result.reasoning_tokens;
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
                reasoning_content: result.reasoning_text,
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
            // Prompt caching (Decision #24, Task #7, iter-96).  > 0 on a
            // cache hit (full-equality, greedy decode); 0 on miss.
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: result.cached_tokens,
            }),
            completion_tokens_details: reasoning_tokens
                .map(|reasoning_tokens| super::schema::CompletionTokensDetails { reasoning_tokens }),
        },
        x_hf2q_timing: Some(timing),
    };
    let mut response = (StatusCode::OK, Json(resp)).into_response();
    apply_transparency_headers(&state, &req, &mut response, summarized_messages, summary_tokens);
    apply_vit_transparency_headers(&mut response, vit_forward_ms, vit_images);
    response
}

/// Apply Decision #23 transparency headers on a chat-completion response.
///
///   - `X-HF2Q-Overflow-Policy` — the policy the server applied for THIS
///     request (request's override if present, else the server default).
///   - `X-HF2Q-Summarized-Messages` — count of messages that were replaced
///     by a synthetic summary (Decision #23 summarize path; when the
///     summarization path is wired in a later iter, handlers set this to
///     the count).
///   - `X-HF2Q-Summary-Tokens` — token count of the synthetic summary.
///
/// Present on EVERY chat-completion response so clients don't have to
/// re-derive the applied policy from config.
fn apply_transparency_headers(
    state: &AppState,
    req: &ChatCompletionRequest,
    resp: &mut Response,
    summarized_messages: Option<usize>,
    summary_tokens: Option<usize>,
) {
    use axum::http::{header::HeaderName, HeaderValue};
    use super::schema::OverflowPolicy;
    let policy = req
        .hf2q_overflow_policy
        .unwrap_or(state.config.default_overflow_policy);
    let policy_str: &'static str = match policy {
        OverflowPolicy::Reject => "reject",
        OverflowPolicy::TruncateLeft => "truncate_left",
        OverflowPolicy::Summarize => "summarize",
    };
    let headers = resp.headers_mut();
    headers.insert(
        HeaderName::from_static("x-hf2q-overflow-policy"),
        HeaderValue::from_static(policy_str),
    );
    if let Some(n) = summarized_messages {
        if let Ok(v) = HeaderValue::from_str(&n.to_string()) {
            headers.insert(HeaderName::from_static("x-hf2q-summarized-messages"), v);
        }
    }
    if let Some(n) = summary_tokens {
        if let Ok(v) = HeaderValue::from_str(&n.to_string()) {
            headers.insert(HeaderName::from_static("x-hf2q-summary-tokens"), v);
        }
    }
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
    /// Decision #23 transparency counters. `Some` only when the
    /// Summarize policy actually ran a forward pass to produce a
    /// synthetic summary system message.
    summarized_messages: Option<usize>,
    summary_tokens: Option<usize>,
    /// Per-position embedding overrides for the multimodal vision path
    /// (Phase 2c Task #17 / iter-99).  Empty for text-only requests.
    /// When non-empty, `chat_completions` dispatches via
    /// `Engine::generate_with_soft_tokens` instead of `Engine::generate`.
    soft_tokens: Vec<engine::SoftTokenData>,
    /// ViT forward wall-clock for the vision branch (transparency
    /// header `x-hf2q-vit-forward-ms`).  `None` for text-only.
    vit_forward_ms: Option<u64>,
    /// Image count for the vision branch (transparency header
    /// `x-hf2q-vit-images`).  `None` for text-only.
    vit_images: Option<usize>,
}

/// Run every validation + rendering + tokenization step common to the
/// streaming and non-streaming chat-completion paths. Returns the prepared
/// context on success, or an `axum::Response` to return directly on failure.
async fn prepare_chat_generation(
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

    // --- Multimodal content pipeline (Phase 2c — Decision #1, iter-99) ---
    // Scan messages for `image_url` content parts; if any are present,
    // require a mmproj, parse each URL, load the bytes, and preprocess
    // (resize/normalize/CHW) to a ViT-ready pixel tensor. Error taxonomy:
    //   images absent                       → text-only flow (no soft-tokens)
    //   images present, mmproj missing      → 400 no_mmproj_loaded
    //   image URL malformed                 → 400 invalid_request (per part)
    //   image load/decode/preprocess fails  → 400 invalid_request (per part)
    //   ViT GPU forward fails               → 500 generation_error
    //   embedding row count not multiple of hidden → 500 generation_error
    //
    // On success we hold `Vec<Vec<f32>>` projected embeddings (one
    // tensor per image) plus a rewritten messages vector with one
    // `<|image|>` placeholder text token inserted in place of each
    // image content part.  The actual soft-token expansion (placeholder
    // → N image tokens + MlxBuffer alloc + range computation) runs
    // after the standard render/tokenize/overflow pipeline so its
    // inputs are exactly the prompt tokens the engine sees.
    let preprocessed_images =
        match process_multimodal_content(&req.messages, state.mmproj.as_ref()) {
            Ok(imgs) => imgs,
            Err(resp) => return Err(resp),
        };
    let (messages_for_render, vision_embeddings, vit_forward_ms_v, vit_images_v) =
        if preprocessed_images.is_empty() {
            (req.messages.clone(), Vec::new(), None, None)
        } else {
            let n_images = preprocessed_images.len();
            let mmproj = state
                .mmproj
                .as_ref()
                .expect("mmproj checked in process_multimodal_content");
            let head_dim_f =
                (mmproj.config.hidden_size / mmproj.config.num_attention_heads) as f32;
            let scale = 1.0f32 / head_dim_f.sqrt();
            let t0 = std::time::Instant::now();
            let embeddings =
                match crate::inference::vision::vit_gpu::compute_vision_embeddings_gpu(
                    &preprocessed_images,
                    &mmproj.weights,
                    &mmproj.config,
                    scale,
                ) {
                    Ok(e) => e,
                    Err(e) => {
                        return Err(ApiError::generation_error(format!(
                            "ViT forward failed: {e}"
                        ))
                        .into_response());
                    }
                };
            let elapsed_ms = t0.elapsed().as_millis() as u64;
            tracing::info!(
                n_images,
                embed_dim = embeddings.first().map(|e| e.len()).unwrap_or(0),
                forward_ms = elapsed_ms,
                "Vision embeddings computed via GPU ViT forward"
            );
            // Validate every embedding's element count is a positive
            // multiple of the chat-model hidden size so the soft-token
            // expansion has a unique answer for `N_image_tokens`.
            let hidden = engine.hidden_size();
            for (i, e) in embeddings.iter().enumerate() {
                if hidden == 0 || e.is_empty() || e.len() % hidden != 0 {
                    return Err(ApiError::generation_error(format!(
                        "vision embedding [{i}] length {} is not a positive multiple of hidden_size {hidden}",
                        e.len()
                    ))
                    .into_response());
                }
            }
            let rewritten = rewrite_messages_for_vision_placeholders(&req.messages);
            (rewritten, embeddings, Some(elapsed_ms), Some(n_images))
        };

    // Compile the response_format grammar (Decision #6).  Iter-95 wires
    // the parsed grammar into `SamplingParams.grammar` so the decode loop
    // can mask invalid tokens at every step.  Pre-iter-95 the grammar
    // was discarded after parse-validation; bad requests still fail
    // fast in <1 ms but valid grammars now constrain decoding instead
    // of being silently ignored.
    let response_grammar: Option<grammar::Grammar> = match req.response_format.as_ref() {
        Some(rf) => match compile_response_format(rf) {
            Ok(g) => g,
            Err(resp) => return Err(resp),
        },
        None => None,
    };

    // Render + tokenize + apply context-overflow policy (Decision #23).
    // `apply_overflow_policy` handles the three policies:
    //   Reject        → 400 context_length_exceeded when prompt ≥ ctx_len.
    //   TruncateLeft  → iteratively drop oldest non-system messages until
    //                   the prompt fits under ctx_len, then re-tokenize.
    //   Summarize     → currently returns 501 (not implemented yet; needs
    //                   internal engine recursion which lands when
    //                   forward_decode is refactored for logit exposure).
    // The 80% budget trigger from Decision #23 applies only to Summarize
    // (its whole point is to preemptively shrink the context so summary +
    // completion fit in the remaining 20%); Reject + TruncateLeft trigger
    // strictly on prompt ≥ ctx_len.
    let policy = req
        .hf2q_overflow_policy
        .unwrap_or(state.config.default_overflow_policy);
    let (prompt_tokens, _prompt_len, summarized_messages, summary_tokens) =
        match apply_overflow_policy(engine, &messages_for_render, policy).await {
            Ok(r) => r,
            Err(resp) => return Err(resp),
        };
    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(SamplingParams::default().max_tokens);
    let stop_strings = req
        .stop
        .clone()
        .map(|s| s.into_vec())
        .unwrap_or_default();
    // Translate request logit_bias (HashMap<String, f32>) → HashMap<u32, f32>.
    // OpenAI keys are stringified token ids; we accept any string that parses
    // as an unsigned int. Malformed keys are silently dropped + warned.
    let logit_bias: std::collections::HashMap<u32, f32> = req
        .logit_bias
        .as_ref()
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| k.parse::<u32>().ok().map(|id| (id, *v)))
                .collect()
        })
        .unwrap_or_default();
    // When the request carries a grammar (response_format=json_object /
    // json_schema), attach the per-vocab token-bytes table so the worker
    // can mask invalid tokens at every decode step.  The table is
    // lazily built + cached on the Engine — first grammar request pays
    // ~50-200 ms (CPU work, off the worker thread); every subsequent
    // request gets a free Arc clone.  No grammar ⇒ no table attachment ⇒
    // zero extra cost.
    let grammar_token_bytes: Option<std::sync::Arc<Vec<Vec<u8>>>> = if response_grammar.is_some() {
        Some(engine.token_bytes_table())
    } else {
        None
    };
    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.map(|v| v as usize).unwrap_or(0),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
        max_tokens,
        stop_strings,
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        seed: req.seed,
        min_p: req.min_p.unwrap_or(0.0),
        logit_bias,
        logprobs: req.logprobs.unwrap_or(false),
        top_logprobs: req.top_logprobs.unwrap_or(0),
        parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
        grammar: response_grammar,
        token_bytes: grammar_token_bytes,
    };
    // Vision soft-token expansion (Phase 2c Task #17 / iter-99).  When
    // the multimodal preflight produced ViT embeddings, locate the
    // single `<|image|>` placeholder token per image in the rendered
    // prompt and EXPAND each into a contiguous run of `N_image_tokens`
    // copies so the prefill iterates one model-side step per image
    // patch position.  `forward_prefill_with_soft_tokens` then bypasses
    // the embed-table gather at those positions and copies the
    // projected vision embeddings into the hidden state instead.
    //
    // Empty `vision_embeddings` ⇒ pure-text path: skip everything,
    // return `Vec::new()` for `soft_tokens`.
    let (final_prompt_tokens, soft_tokens) = if vision_embeddings.is_empty() {
        (prompt_tokens, Vec::<engine::SoftTokenData>::new())
    } else {
        match expand_image_placeholders(engine, &prompt_tokens, &vision_embeddings) {
            Ok(pair) => pair,
            Err(resp) => return Err(resp),
        }
    };
    // Post-expansion context-budget guard.  The pre-expansion overflow
    // policy ran on the un-expanded prompt (which carries one
    // placeholder token per image, not `N_image_tokens` per image), so
    // the expanded prompt may push past `context_length` even when the
    // un-expanded prompt fit.  Fail clean with the standard 400 envelope
    // when that happens; richer image-aware overflow handling is a
    // follow-up iter (would need to drop *images*, which is a content
    // decision, not a tokenization concern).
    if let Some(ctx_len) = engine.context_length() {
        if final_prompt_tokens.len() >= ctx_len {
            return Err(
                ApiError::context_length_exceeded(ctx_len, final_prompt_tokens.len())
                    .into_response(),
            );
        }
    }
    Ok(PreparedChatContext {
        engine: (),
        prompt_tokens: final_prompt_tokens,
        params,
        summarized_messages,
        summary_tokens,
        soft_tokens,
        vit_forward_ms: vit_forward_ms_v,
        vit_images: vit_images_v,
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

    // Vision streaming is not supported in this iter — the soft-token
    // path runs through `Engine::generate_with_soft_tokens` (non-
    // streaming).  When the streaming variant lands, this gate reverts
    // to the standard `generate_stream` call but with the soft tokens
    // attached.  Until then a vision request with `stream: true`
    // returns a clean 400 instead of silently dropping the image
    // embeddings.
    if !prepared.soft_tokens.is_empty() {
        return ApiError::invalid_request(
            "stream: true with image content parts is not yet supported. \
             Send the same request with stream: false to exercise the \
             vision soft-token path.",
            Some("stream".into()),
        )
        .into_response();
    }

    // Decision #23: capture transparency counters BEFORE we move
    // `prepared` into `engine.generate_stream`. They flow into the
    // response headers below.
    let summarized_messages = prepared.summarized_messages;
    let summary_tokens = prepared.summary_tokens;

    let (events_tx, events_rx) = tokio::sync::mpsc::channel(64);
    // Worker bumps this counter if it aborts because the SSE receiver was
    // dropped (client disconnect per Decision #18). Shared atomic lives on
    // ServerMetrics so /metrics surfaces it.
    let cancellation_counter =
        Some(state.metrics.sse_cancellations_counter_arc());
    if let Err(e) = engine
        .generate_stream(
            prepared.prompt_tokens,
            prepared.params,
            events_tx,
            cancellation_counter,
        )
        .await
    {
        let msg = format!("{e}");
        if msg.contains("queue_full") {
            state
                .metrics
                .chat_completions_queue_full
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return queue_full_with_rate_limit_headers(&state);
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
    let sse =
        generation_events_to_sse(events_rx, request_id, req.model.clone(), created, opts);
    let mut response = sse.into_response();
    apply_transparency_headers(&state, &req, &mut response, summarized_messages, summary_tokens);
    response
}

/// Pure-compute core of `apply_overflow_policy`. Parameterized by an
/// abstract `tokenize` callback so it can be unit-tested without a live
/// Engine + tokenizer + chat template.
///
/// Returns `Ok((possibly-shrunk messages, tokens, length))` on success.
/// On any failure the caller receives a `TruncateOutcome` describing the
/// failure kind; the HTTP-level `apply_overflow_policy` maps it to the
/// right 4xx/5xx response envelope.
#[derive(Debug, PartialEq)]
pub(crate) enum TruncateOutcome<E> {
    /// Prompt fits within `ctx_len`; no mutation was needed.
    Fits(Vec<super::schema::ChatMessage>, usize),
    /// Shrunk successfully via left-truncation.
    Truncated(Vec<super::schema::ChatMessage>, usize, usize),
    /// Could not shrink further (only system + last-user remained and the
    /// prompt still overflowed).
    CannotShrink { ctx_len: usize, actual: usize },
    /// Caller-tokenizer returned an error.
    TokenizeErr(E),
}

/// Left-truncate iteratively. Drops the oldest non-system message that
/// isn't the triggering last-user turn, re-tokenizes via `tokenize`, and
/// stops when the prompt fits or when nothing more can be dropped.
pub(crate) fn truncate_left<F, E>(
    messages: &[super::schema::ChatMessage],
    ctx_len: usize,
    mut tokenize: F,
) -> TruncateOutcome<E>
where
    F: FnMut(&[super::schema::ChatMessage]) -> Result<usize, E>,
{
    // Initial size check.
    let initial_n = match tokenize(messages) {
        Ok(n) => n,
        Err(e) => return TruncateOutcome::TokenizeErr(e),
    };
    if initial_n < ctx_len {
        return TruncateOutcome::Fits(messages.to_vec(), initial_n);
    }

    let mut msgs = messages.to_vec();
    let mut iterations = 0usize;
    loop {
        // Identify the last-user index on EACH iteration since msgs mutates.
        let last_user_idx = msgs.iter().rposition(|m| m.role == "user");
        // Find the first non-system-and-not-last-user index.
        let drop_idx = msgs
            .iter()
            .position(|m| m.role != "system")
            .filter(|idx| Some(*idx) != last_user_idx);
        let Some(drop_idx) = drop_idx else {
            // Only system + last user remain.
            let final_n = tokenize(&msgs).unwrap_or(usize::MAX);
            return TruncateOutcome::CannotShrink {
                ctx_len,
                actual: final_n,
            };
        };
        if Some(drop_idx) == last_user_idx {
            // Defensive: the filter above already excludes this, but leave
            // the guard for future refactor safety.
            return TruncateOutcome::CannotShrink {
                ctx_len,
                actual: initial_n,
            };
        }
        msgs.remove(drop_idx);
        let n = match tokenize(&msgs) {
            Ok(n) => n,
            Err(e) => return TruncateOutcome::TokenizeErr(e),
        };
        iterations += 1;
        if n < ctx_len {
            return TruncateOutcome::Truncated(msgs, n, iterations);
        }
    }
}

/// Number of most-recent non-system messages to keep verbatim during
/// summarize. Anything older than this (excluding system messages) gets
/// folded into a single synthetic summary system message. K=4 is two
/// user-assistant turns — enough for the model to maintain local
/// coherence without dragging the entire history.
const SUMMARIZE_KEEP_RECENT_MSGS: usize = 4;

/// Cap on completion tokens for a summarization forward pass. ~120-160
/// tokens is enough for a 2-3 sentence summary; anything longer defeats
/// the purpose of context compression.
const SUMMARIZE_MAX_TOKENS: usize = 160;

/// Pure-compute split of a message list into (system_prefix,
/// summary_window, recent_window) for the Decision #23 summarize path.
///
/// - `system_prefix`: every leading `role == "system"` message, in
///   order. Preserved verbatim because the system prompt sets behavior.
/// - `recent_window`: the last `keep_recent_count` non-system messages.
///   The model needs these for local context.
/// - `summary_window`: everything else (the oldest non-system messages).
///   These get summarized into a single synthetic system message.
///
/// When `recent_keep_count >= num_non_system`, the summary window is
/// empty — caller must fall back to truncate_left.
pub(crate) fn split_for_summarize(
    messages: &[super::schema::ChatMessage],
    keep_recent_count: usize,
) -> SummarySplit {
    // System prefix = leading run of role="system".
    let prefix_end = messages
        .iter()
        .position(|m| m.role != "system")
        .unwrap_or(messages.len());
    let system_prefix: Vec<_> = messages[..prefix_end].to_vec();
    let non_system: &[super::schema::ChatMessage] = &messages[prefix_end..];

    // Anywhere in the tail might also be system messages (rare — most
    // chat templates only support one leading system); they're treated
    // as part of the summary/recent split. Keep_recent_count is counted
    // against ALL tail messages, not non-system-only.
    let n_tail = non_system.len();
    let recent_count = keep_recent_count.min(n_tail);
    let summary_end = n_tail - recent_count;

    SummarySplit {
        system_prefix,
        summary_window: non_system[..summary_end].to_vec(),
        recent_window: non_system[summary_end..].to_vec(),
    }
}

/// Output of `split_for_summarize`. Sum of all three vectors' lengths
/// equals the input length.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SummarySplit {
    pub system_prefix: Vec<super::schema::ChatMessage>,
    pub summary_window: Vec<super::schema::ChatMessage>,
    pub recent_window: Vec<super::schema::ChatMessage>,
}

/// Build the user-side text for a summarization request. Each message
/// in `summary_window` becomes a single line `"<ROLE>: <content>"`.
/// Non-text content parts are silently dropped (image_url etc. — the
/// summarizer can't see images anyway).
pub(crate) fn build_summary_user_text(
    summary_window: &[super::schema::ChatMessage],
) -> String {
    use super::schema::MessageContent;
    let mut buf = String::with_capacity(summary_window.len() * 80);
    buf.push_str(
        "Summarize the following conversation in 2-3 sentences. \
         Be concise; preserve key facts and decisions:\n\n",
    );
    for m in summary_window {
        let text = match &m.content {
            None => "".to_string(),
            Some(MessageContent::Text(s)) => s.clone(),
            Some(MessageContent::Parts(parts)) => parts
                .iter()
                .filter_map(|p| match p {
                    super::schema::ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        };
        if text.trim().is_empty() {
            continue;
        }
        buf.push_str(&m.role.to_uppercase());
        buf.push_str(": ");
        buf.push_str(&text);
        buf.push('\n');
    }
    buf
}

/// Build the synthetic system message that replaces `summary_window` in
/// the rewritten message list. Wraps the model's summary in a marker
/// prefix so downstream consumers can recognize and (if they want)
/// re-expand from cached history.
pub(crate) fn build_synthetic_summary_message(
    summary_text: &str,
) -> super::schema::ChatMessage {
    super::schema::ChatMessage {
        role: "system".to_string(),
        content: Some(super::schema::MessageContent::Text(format!(
            "[Summary of prior conversation]: {}",
            summary_text.trim()
        ))),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

/// Apply Decision #23 context-overflow policy. Renders the chat template,
/// tokenizes, and checks against `engine.context_length()`:
///
///   - `Reject`        → 400 context_length_exceeded if `prompt ≥ ctx_len`.
///   - `TruncateLeft`  → drop oldest non-system messages until prompt fits
///                       (via the pure-compute `truncate_left` helper).
///   - `Summarize`     → 501 not-implemented in this iter (needs internal
///                       engine recursion; lands with forward_decode
///                       refactor).
///
/// Returns `(prompt_tokens, prompt_len, summarized_messages_count,
/// summary_tokens)` on success. The last two are `Some` only when the
/// Summarize policy actually ran a summarization pass — they flow into
/// `X-HF2Q-Summarized-Messages` / `X-HF2Q-Summary-Tokens` transparency
/// headers per Decision #23.
async fn apply_overflow_policy(
    engine: &engine::Engine,
    messages: &[super::schema::ChatMessage],
    policy: super::schema::OverflowPolicy,
) -> std::result::Result<(Vec<u32>, usize, Option<usize>, Option<usize>), Response> {
    use super::schema::OverflowPolicy;

    let (tokens, n) = render_and_tokenize_for_overflow(engine, messages)?;
    let ctx_len = match engine.context_length() {
        Some(c) => c,
        None => return Ok((tokens, n, None, None)), // no advertised ctx → trust the caller
    };
    if n < ctx_len {
        return Ok((tokens, n, None, None));
    }

    // Prompt does not fit. Dispatch on policy.
    match policy {
        OverflowPolicy::Reject => {
            Err(ApiError::context_length_exceeded(ctx_len, n).into_response())
        }
        OverflowPolicy::TruncateLeft => {
            let (t, n) = apply_truncate_left(engine, messages, ctx_len, tokens, n)?;
            Ok((t, n, None, None))
        }
        OverflowPolicy::Summarize => apply_summarize(engine, messages, ctx_len).await,
    }
}

/// Render the chat template + tokenize. Returns `(tokens, n_tokens)` or
/// the response to bail with on failure. Pulled out of
/// `apply_overflow_policy` so the summarize path can reuse it.
fn render_and_tokenize_for_overflow(
    engine: &engine::Engine,
    msgs: &[super::schema::ChatMessage],
) -> Result<(Vec<u32>, usize), Response> {
    let rendered = match engine::render_chat_prompt(engine.chat_template(), msgs) {
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
    let tokens: Vec<u32> = encoding.get_ids().to_vec();
    let n = tokens.len();
    Ok((tokens, n))
}

/// TruncateLeft branch helper. Iteratively drop oldest non-system
/// messages until the prompt fits; map the outcome to the right HTTP
/// envelope.
fn apply_truncate_left(
    engine: &engine::Engine,
    messages: &[super::schema::ChatMessage],
    ctx_len: usize,
    initial_tokens: Vec<u32>,
    initial_n: usize,
) -> Result<(Vec<u32>, usize), Response> {
    let outcome = truncate_left(messages, ctx_len, |msgs| {
        render_and_tokenize_for_overflow(engine, msgs)
            .map(|(_, n)| n)
            .map_err(|r| r)
    });
    match outcome {
        TruncateOutcome::Fits(_, _) => Ok((initial_tokens, initial_n)),
        TruncateOutcome::Truncated(shrunk, _, _) => render_and_tokenize_for_overflow(engine, &shrunk),
        TruncateOutcome::CannotShrink { ctx_len, actual } => {
            Err(ApiError::context_length_exceeded(ctx_len, actual).into_response())
        }
        TruncateOutcome::TokenizeErr(resp) => Err(resp),
    }
}

/// Summarize branch — Decision #23. Splits the message list into
/// (system_prefix, summary_window, recent_window), runs a forward pass
/// on the summary window through the loaded chat engine, replaces the
/// window with a synthetic system message, and re-tokenizes. If the
/// summary window is empty (too few messages to compress) or the
/// resulting prompt still doesn't fit, falls through to truncate_left.
///
/// Returns the same shape as `apply_overflow_policy`: when the
/// summarize path actually ran, the last two `Option<usize>` are
/// populated with the count of messages collapsed into the summary
/// and the summary's completion-token count.
async fn apply_summarize(
    engine: &engine::Engine,
    messages: &[super::schema::ChatMessage],
    ctx_len: usize,
) -> Result<(Vec<u32>, usize, Option<usize>, Option<usize>), Response> {
    let split = split_for_summarize(messages, SUMMARIZE_KEEP_RECENT_MSGS);
    if split.summary_window.is_empty() {
        // Nothing to summarize — fall through to truncate_left without
        // populating the summarize-specific transparency counters.
        let (initial_tokens, initial_n) = render_and_tokenize_for_overflow(engine, messages)?;
        let (t, n) = apply_truncate_left(engine, messages, ctx_len, initial_tokens, initial_n)?;
        return Ok((t, n, None, None));
    }

    // Build the summarization request: a 2-message conversation that
    // reuses the engine's chat template. System sets the role; user
    // delivers the rendered window verbatim.
    let summary_user_text = build_summary_user_text(&split.summary_window);
    let summary_request_msgs = vec![
        super::schema::ChatMessage {
            role: "system".to_string(),
            content: Some(super::schema::MessageContent::Text(
                "You are a concise summarization assistant. Produce 2-3 \
                 sentence summaries of conversations. Preserve key facts \
                 and decisions; drop pleasantries."
                    .to_string(),
            )),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        super::schema::ChatMessage {
            role: "user".to_string(),
            content: Some(super::schema::MessageContent::Text(summary_user_text)),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ];

    // Tokenize the summary request through the engine's chat template.
    let (summary_prompt_tokens, summary_prompt_n) =
        render_and_tokenize_for_overflow(engine, &summary_request_msgs)?;

    // Sanity: the summary prompt + completion budget must itself fit.
    // If not, the summary window is too long for a one-shot summary
    // (e.g. a single 50k-token user dump) — fall back to truncate_left
    // on the original message list.
    if summary_prompt_n + SUMMARIZE_MAX_TOKENS >= ctx_len {
        let (initial_tokens, initial_n) = render_and_tokenize_for_overflow(engine, messages)?;
        let (t, n) = apply_truncate_left(engine, messages, ctx_len, initial_tokens, initial_n)?;
        return Ok((t, n, None, None));
    }

    // Run the summarization pass. T=0 for determinism; max 160 tokens.
    let params = super::engine::SamplingParams {
        temperature: 0.0,
        max_tokens: SUMMARIZE_MAX_TOKENS,
        ..super::engine::SamplingParams::default()
    };
    let summary_text = match engine.generate(summary_prompt_tokens, params).await {
        Ok(result) => {
            let text = result.text.trim().to_string();
            if text.is_empty() {
                tracing::warn!("summarize produced empty text; falling back to truncate_left");
                let (initial_tokens, initial_n) =
                    render_and_tokenize_for_overflow(engine, messages)?;
                let (t, n) =
                    apply_truncate_left(engine, messages, ctx_len, initial_tokens, initial_n)?;
                return Ok((t, n, None, None));
            }
            (text, result.completion_tokens)
        }
        Err(e) => {
            // Engine refused the request (e.g. queue full). Surface as
            // an internal error — the user asked for summarize and we
            // could not deliver. Don't silently fall through, that hides
            // overload from the operator.
            tracing::warn!(error = %e, "summarize engine.generate failed");
            return Err(ApiError::generation_error(format!(
                "summarize forward pass failed: {e}"
            ))
            .into_response());
        }
    };
    let (summary_text, summary_completion_tokens) = summary_text;

    // Build the rewritten message list with the synthetic summary
    // inserted after the system prefix.
    let synthetic = build_synthetic_summary_message(&summary_text);
    let mut new_messages = split.system_prefix.clone();
    new_messages.push(synthetic);
    new_messages.extend(split.recent_window.iter().cloned());

    let summarized_count = split.summary_window.len();

    let (new_tokens, new_n) = render_and_tokenize_for_overflow(engine, &new_messages)?;
    if new_n < ctx_len {
        return Ok((
            new_tokens,
            new_n,
            Some(summarized_count),
            Some(summary_completion_tokens),
        ));
    }

    // Summary collapsed the window but the result still doesn't fit
    // (e.g. recent_window alone is too long). Truncate from the recent
    // end. Keep the summarize-counters populated so the operator can
    // see in transparency headers what the policy attempted.
    let outcome = truncate_left(&new_messages, ctx_len, |msgs| {
        render_and_tokenize_for_overflow(engine, msgs)
            .map(|(_, n)| n)
            .map_err(|r| r)
    });
    match outcome {
        TruncateOutcome::Fits(_, _) => Ok((
            new_tokens,
            new_n,
            Some(summarized_count),
            Some(summary_completion_tokens),
        )),
        TruncateOutcome::Truncated(shrunk, _, _) => {
            let (t, n) = render_and_tokenize_for_overflow(engine, &shrunk)?;
            Ok((
                t,
                n,
                Some(summarized_count),
                Some(summary_completion_tokens),
            ))
        }
        TruncateOutcome::CannotShrink { ctx_len, actual } => {
            Err(ApiError::context_length_exceeded(ctx_len, actual).into_response())
        }
        TruncateOutcome::TokenizeErr(resp) => Err(resp),
    }
}

// ---------------------------------------------------------------------------
// Tests for the pure-compute truncate helper
// ---------------------------------------------------------------------------

#[cfg(test)]
mod truncate_tests {
    use super::*;
    use super::super::schema::{ChatMessage, MessageContent};

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: Some(MessageContent::Text(content.into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Fake tokenizer: N tokens per message. Chosen so tests can reason
    /// about token counts precisely without needing a real tokenizer.
    fn toks_per_msg(msgs: &[ChatMessage], per_msg: usize) -> Result<usize, ()> {
        Ok(msgs.len() * per_msg)
    }

    #[test]
    fn truncate_left_fits_returns_initial_count() {
        let msgs = vec![msg("system", "s"), msg("user", "u")];
        let out = truncate_left::<_, ()>(&msgs, 100, |m| toks_per_msg(m, 10));
        match out {
            TruncateOutcome::Fits(m, n) => {
                assert_eq!(m.len(), 2);
                assert_eq!(n, 20);
            }
            other => panic!("expected Fits, got {:?}", other),
        }
    }

    #[test]
    fn truncate_left_drops_oldest_nonsystem_until_fits() {
        // 10 tokens/msg. 5 messages = 50 tokens. ctx_len=25 → must drop
        // down to 2 messages. But system+last_user are both pinned, so
        // the only droppables are the 3 middle. Dropping 3 → 2 msgs = 20
        // tokens < 25. Final: 2 messages retained.
        let msgs = vec![
            msg("system", "s"),
            msg("user", "u1"),
            msg("assistant", "a1"),
            msg("user", "u2"),
            msg("assistant", "a2"),
            msg("user", "final"),
        ];
        let out = truncate_left::<_, ()>(&msgs, 25, |m| toks_per_msg(m, 10));
        match out {
            TruncateOutcome::Truncated(retained, n, iters) => {
                assert_eq!(retained.len(), 2);
                assert_eq!(retained[0].role, "system");
                assert_eq!(retained[1].role, "user");
                // Last user should be the ORIGINAL last-user (the "final" msg).
                assert_eq!(retained[1].content.as_ref().map(|c| c.text()).unwrap(), "final");
                assert_eq!(n, 20);
                assert_eq!(iters, 4); // dropped u1, a1, u2, a2
            }
            other => panic!("expected Truncated, got {:?}", other),
        }
    }

    #[test]
    fn truncate_left_cannot_shrink_when_system_plus_last_user_overflow() {
        // ctx_len=15 but system+last_user = 20 tokens. Cannot shrink further.
        let msgs = vec![msg("system", "s"), msg("user", "u")];
        let out = truncate_left::<_, ()>(&msgs, 15, |m| toks_per_msg(m, 10));
        assert!(
            matches!(out, TruncateOutcome::CannotShrink { .. }),
            "got {:?}",
            out
        );
    }

    #[test]
    fn truncate_left_no_system_keeps_last_user_only() {
        // No system message. messages: user, assistant, user(final).
        // 30 tokens total, ctx=15 → drop user and assistant → 1 msg = 10
        // tokens < 15. Retained: just the last user.
        let msgs = vec![
            msg("user", "u1"),
            msg("assistant", "a1"),
            msg("user", "final"),
        ];
        let out = truncate_left::<_, ()>(&msgs, 15, |m| toks_per_msg(m, 10));
        match out {
            TruncateOutcome::Truncated(retained, n, _) => {
                assert_eq!(retained.len(), 1);
                assert_eq!(retained[0].role, "user");
                assert_eq!(retained[0].content.as_ref().map(|c| c.text()).unwrap(), "final");
                assert_eq!(n, 10);
            }
            other => panic!("expected Truncated, got {:?}", other),
        }
    }

    #[test]
    fn truncate_left_multiple_system_messages_all_preserved() {
        // Two system messages + several history turns. Both systems must
        // survive truncation.
        let msgs = vec![
            msg("system", "s1"),
            msg("system", "s2"),
            msg("user", "u1"),
            msg("assistant", "a1"),
            msg("user", "final"),
        ];
        let out = truncate_left::<_, ()>(&msgs, 35, |m| toks_per_msg(m, 10));
        match out {
            TruncateOutcome::Truncated(retained, _, _) => {
                assert!(retained.iter().filter(|m| m.role == "system").count() == 2);
                assert_eq!(retained.last().unwrap().role, "user");
            }
            other => panic!("expected Truncated, got {:?}", other),
        }
    }

    #[test]
    fn truncate_left_propagates_tokenizer_error() {
        let msgs = vec![msg("user", "u")];
        let out = truncate_left::<_, &'static str>(&msgs, 100, |_| Err("fake tokenize error"));
        assert_eq!(out, TruncateOutcome::TokenizeErr("fake tokenize error"));
    }

    #[test]
    fn truncate_left_is_deterministic() {
        // Same input → same output. Regression-guard against accidental
        // randomness via HashMap iteration or similar.
        let msgs = vec![
            msg("system", "s"),
            msg("user", "u1"),
            msg("assistant", "a1"),
            msg("user", "final"),
        ];
        let a = truncate_left::<_, ()>(&msgs, 25, |m| toks_per_msg(m, 10));
        let b = truncate_left::<_, ()>(&msgs, 25, |m| toks_per_msg(m, 10));
        assert_eq!(format!("{:?}", a), format!("{:?}", b));
    }

    // ----- split_for_summarize -----

    #[test]
    fn split_for_summarize_no_messages_yields_empty_split() {
        let s = split_for_summarize(&[], 4);
        assert!(s.system_prefix.is_empty());
        assert!(s.summary_window.is_empty());
        assert!(s.recent_window.is_empty());
    }

    #[test]
    fn split_for_summarize_only_system_prefix() {
        let msgs = vec![msg("system", "s1"), msg("system", "s2")];
        let s = split_for_summarize(&msgs, 4);
        assert_eq!(s.system_prefix.len(), 2);
        assert!(s.summary_window.is_empty());
        assert!(s.recent_window.is_empty());
    }

    #[test]
    fn split_for_summarize_keep_count_exceeds_tail_means_no_summary() {
        // 1 system + 2 non-system, keep_recent=4 → all non-system in recent.
        let msgs = vec![
            msg("system", "s"),
            msg("user", "u"),
            msg("assistant", "a"),
        ];
        let s = split_for_summarize(&msgs, 4);
        assert_eq!(s.system_prefix.len(), 1);
        assert!(s.summary_window.is_empty());
        assert_eq!(s.recent_window.len(), 2);
    }

    #[test]
    fn split_for_summarize_5_messages_keep_2_recent() {
        let msgs = vec![
            msg("system", "s"),
            msg("user", "u1"),
            msg("assistant", "a1"),
            msg("user", "u2"),
            msg("assistant", "a2"),
            msg("user", "u3"),
        ];
        let s = split_for_summarize(&msgs, 2);
        // System prefix: 1
        // Tail = 5 (u1 a1 u2 a2 u3). keep_recent=2 → recent = [a2, u3], summary = [u1, a1, u2].
        assert_eq!(s.system_prefix.len(), 1);
        assert_eq!(s.summary_window.len(), 3);
        assert_eq!(s.summary_window[0].content,
            Some(MessageContent::Text("u1".into())));
        assert_eq!(s.summary_window[2].content,
            Some(MessageContent::Text("u2".into())));
        assert_eq!(s.recent_window.len(), 2);
        assert_eq!(s.recent_window[1].content,
            Some(MessageContent::Text("u3".into())));
    }

    #[test]
    fn split_for_summarize_no_system_prefix() {
        let msgs = vec![
            msg("user", "u1"),
            msg("assistant", "a1"),
            msg("user", "u2"),
        ];
        let s = split_for_summarize(&msgs, 2);
        assert!(s.system_prefix.is_empty());
        assert_eq!(s.summary_window.len(), 1);
        assert_eq!(s.summary_window[0].content,
            Some(MessageContent::Text("u1".into())));
        assert_eq!(s.recent_window.len(), 2);
    }

    // ----- build_summary_user_text -----

    #[test]
    fn build_summary_user_text_skips_empty_content() {
        let window = vec![
            msg("user", "hello"),
            msg("assistant", ""),
            msg("user", "world"),
        ];
        let out = build_summary_user_text(&window);
        assert!(out.contains("USER: hello"));
        assert!(out.contains("USER: world"));
        // Empty assistant message should NOT produce an "ASSISTANT:" line.
        assert!(!out.contains("ASSISTANT:"));
    }

    #[test]
    fn build_summary_user_text_uppercases_role() {
        let window = vec![msg("user", "x"), msg("tool", "y")];
        let out = build_summary_user_text(&window);
        assert!(out.contains("USER: x"));
        assert!(out.contains("TOOL: y"));
    }

    #[test]
    fn build_summary_user_text_handles_multipart_content() {
        use super::super::schema::{ContentPart, ImageUrl};
        let m = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: "see this:".into() },
                ContentPart::ImageUrl {
                    image_url: ImageUrl { url: "data:img...".into(), detail: None },
                },
                ContentPart::Text { text: "and this".into() },
            ])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let out = build_summary_user_text(&[m]);
        // Text parts joined; image_url silently dropped.
        assert!(out.contains("USER: see this: and this"));
        assert!(!out.contains("data:img"));
    }

    // ----- build_synthetic_summary_message -----

    #[test]
    fn build_synthetic_summary_message_wraps_with_marker() {
        let m = build_synthetic_summary_message("user discussed deployment");
        assert_eq!(m.role, "system");
        let content = match m.content {
            Some(MessageContent::Text(s)) => s,
            _ => panic!("expected text content"),
        };
        assert!(content.starts_with("[Summary of prior conversation]:"));
        assert!(content.contains("user discussed deployment"));
    }

    #[test]
    fn build_synthetic_summary_message_trims_whitespace() {
        let m = build_synthetic_summary_message("  trimmed text  \n");
        let content = match m.content {
            Some(MessageContent::Text(s)) => s,
            _ => unreachable!(),
        };
        // Surrounding whitespace stripped from the model output.
        assert!(content.ends_with("trimmed text"));
        assert!(!content.ends_with("trimmed text  "));
    }
}

/// Top-level multimodal content pipeline for a chat-completion request.
///
/// Behavior matrix:
///
/// | images present | mmproj loaded | outcome                              |
/// |----------------|---------------|--------------------------------------|
/// | no             | either        | Ok(vec![]) — normal text-only flow.  |
/// | yes            | no            | Err(400 no_mmproj_loaded).           |
/// | yes            | yes           | preprocess each → Ok(Vec<Preprocessed>)    |
/// | yes (bad URL)  | either        | Err(400 invalid_request on that part).     |
///
/// Returns the preprocessed image tensors in message order. Empty vec when
/// the request is text-only. The caller is responsible for deciding what
/// to do with a non-empty vec: until the ViT forward pass lands (Task #15),
/// the handler short-circuits with a 501.
///
/// Per mantra ("no stubs"): we refuse to silently strip images from a
/// request that supplied them, and we refuse to claim the ViT forward
/// pass is ready when it isn't. The preprocessing stage IS done for real
/// so the request exercises the full load→decode→normalize→CHW pipeline.
fn process_multimodal_content(
    messages: &[super::schema::ChatMessage],
    mmproj: Option<&super::state::LoadedMmproj>,
) -> std::result::Result<Vec<crate::inference::vision::PreprocessedImage>, Response> {
    use super::schema::{ContentPart, MessageContent};
    // Pass 1: gather (msg_idx, part_idx, &ImageUrl) refs without parsing.
    // A separate pass keeps the early-exit policy decisions (mmproj absent)
    // from wasting work on the load/decode pipeline.
    let mut image_refs: Vec<(usize, usize, &super::schema::ImageUrl)> = Vec::new();
    for (mi, msg) in messages.iter().enumerate() {
        if let Some(MessageContent::Parts(parts)) = msg.content.as_ref() {
            for (pi, p) in parts.iter().enumerate() {
                if let ContentPart::ImageUrl { image_url } = p {
                    image_refs.push((mi, pi, image_url));
                }
            }
        }
    }
    if image_refs.is_empty() {
        return Ok(Vec::new());
    }
    // Images present — confirm we have an mmproj before running any I/O.
    let mmproj = match mmproj {
        Some(m) => m,
        None => return Err(ApiError::no_mmproj_loaded().into_response()),
    };
    let preprocess_cfg = mmproj.config.preprocess_config();

    let mut out: Vec<crate::inference::vision::PreprocessedImage> =
        Vec::with_capacity(image_refs.len());
    for (mi, pi, image_url) in image_refs {
        // Parse the URL (data URI / file:// / bare path / http).
        let parsed = crate::inference::vision::parse_image_url(&image_url.url)
            .map_err(|e| {
                ApiError::invalid_request(
                    format!(
                        "messages[{}].content[{}].image_url parse failed: {}",
                        mi, pi, e
                    ),
                    Some(format!("messages[{}].content[{}]", mi, pi)),
                )
                .into_response()
            })?;
        // Compute a source_label before consuming `parsed` into the loader.
        let source_label = match &parsed {
            crate::inference::vision::ImageInput::DataUri { mime_type, .. } => mime_type.clone(),
            crate::inference::vision::ImageInput::FilePath(p) => p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("file")
                .to_string(),
            crate::inference::vision::ImageInput::HttpUrl(u) => u.clone(),
        };
        // Load raw bytes (base64 decode / file read / http reject).
        let bytes = crate::inference::vision::load_image_bytes(&parsed).map_err(|e| {
            ApiError::invalid_request(
                format!(
                    "messages[{}].content[{}].image_url load failed: {}",
                    mi, pi, e
                ),
                Some(format!("messages[{}].content[{}]", mi, pi)),
            )
            .into_response()
        })?;
        // Decode + resize + normalize + CHW. This is CPU-heavy per image
        // (896×896 bilinear resize + 2.4M float normalizes); future work
        // can move it to a rayon pool if a multi-image request becomes a
        // hot path.
        let pixel_values = crate::inference::vision::preprocess_rgb_chw(&bytes, &preprocess_cfg)
            .map_err(|e| {
                ApiError::invalid_request(
                    format!(
                        "messages[{}].content[{}].image_url preprocess failed: {}",
                        mi, pi, e
                    ),
                    Some(format!("messages[{}].content[{}]", mi, pi)),
                )
                .into_response()
            })?;
        out.push(crate::inference::vision::PreprocessedImage {
            pixel_values,
            target_size: preprocess_cfg.target_size,
            source_label,
        });
    }
    Ok(out)
}

/// Emit the 501 for "images preprocessed successfully, ViT forward pass
/// not yet wired". Centralized so the streaming and non-streaming paths
/// phrase it identically.
fn vit_forward_pending_response(n_images: usize) -> Response {
    let err = ApiError::invalid_request(
        format!(
            "Request carries {n} image(s); all parsed + preprocessed \
             successfully into ViT pixel tensors. The ViT forward pass \
             that consumes them lands in a later hf2q iter (ADR-005 \
             Phase 2c, Task #15). Send a text-only message to exercise \
             the chat path today.",
            n = n_images
        ),
        Some("messages".into()),
    );
    let mut resp = err.into_response();
    *resp.status_mut() = StatusCode::NOT_IMPLEMENTED;
    resp
}

/// Iter 52: 501 emitted AFTER the GPU ViT forward succeeded. Reports
/// the per-request vision-forward timing so clients can verify the
/// path is real even though the engine doesn't yet consume the
/// embeddings.
/// Vision-path transparency headers (Phase 2c Task #17 / iter-99).  Set
/// on every successful chat-completion that routed through the soft-
/// token path:
///
///   - `X-HF2Q-ViT-Forward-Ms` — wall-clock for the ViT GPU forward.
///   - `X-HF2Q-ViT-Images`     — count of images consumed.
///
/// Lets clients see the per-request vision cost without parsing the
/// body.  No-op when both fields are `None` (text-only request).
fn apply_vit_transparency_headers(
    resp: &mut Response,
    forward_ms: Option<u64>,
    n_images: Option<usize>,
) {
    use axum::http::{header::HeaderName, HeaderValue};
    let headers = resp.headers_mut();
    if let Some(ms) = forward_ms {
        if let Ok(v) = HeaderValue::from_str(&ms.to_string()) {
            headers.insert(HeaderName::from_static("x-hf2q-vit-forward-ms"), v);
        }
    }
    if let Some(n) = n_images {
        if let Ok(v) = HeaderValue::from_str(&n.to_string()) {
            headers.insert(HeaderName::from_static("x-hf2q-vit-images"), v);
        }
    }
}

/// Convert each `MessageContent::Parts` containing image content parts
/// into a `MessageContent::Text` whose payload preserves the original
/// part order, with each image part substituted by the literal token
/// text `<|image|>` (Gemma family) — one placeholder per image.  Pure-
/// text messages and pure-text Parts are passed through unchanged.
///
/// The chat template (rendered by `engine::render_chat_prompt`) wraps
/// the message content in role-specific markers; the placeholder text
/// flows through verbatim and the tokenizer maps each `<|image|>`
/// sequence to its special-token id (Gemma 4: 258880).  At soft-token
/// expansion time those single placeholder positions are replaced by
/// `N_image_tokens` consecutive copies of the same id (the model never
/// reads them — the embed step is short-circuited per
/// `SoftTokenInjection`'s contract — but keeping them as literal image
/// tokens means OpenAI usage shape and hf2q's own token-counting stay
/// consistent).
///
/// Phase 2c Task #17 / iter-99.
fn rewrite_messages_for_vision_placeholders(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    use super::schema::ContentPart;
    messages
        .iter()
        .map(|msg| {
            let new_content = msg.content.as_ref().map(|c| match c {
                MessageContent::Text(_) => c.clone(),
                MessageContent::Parts(parts) => {
                    let any_image = parts
                        .iter()
                        .any(|p| matches!(p, ContentPart::ImageUrl { .. }));
                    if !any_image {
                        c.clone()
                    } else {
                        let mut buf = String::new();
                        for part in parts {
                            match part {
                                ContentPart::Text { text } => buf.push_str(text),
                                ContentPart::ImageUrl { .. } => buf.push_str("<|image|>"),
                            }
                        }
                        MessageContent::Text(buf)
                    }
                }
            });
            ChatMessage {
                role: msg.role.clone(),
                content: new_content,
                reasoning_content: msg.reasoning_content.clone(),
                tool_calls: msg.tool_calls.clone(),
                tool_call_id: msg.tool_call_id.clone(),
                name: msg.name.clone(),
            }
        })
        .collect()
}

/// Reported by `compute_soft_token_layout` when the prompt's
/// placeholder-token count doesn't match the supplied per-image
/// vector.  Carries both observed counts so the caller can build a
/// useful error message (the chat-template emitted the wrong number
/// of image markers, or the request's image count drifted from what
/// the renderer saw).
///
/// Phase 2c Task #17 / iter-100.
#[derive(Debug, PartialEq)]
pub(crate) struct PlaceholderCountMismatch {
    pub placeholder_positions_found: usize,
    pub n_image_tokens_supplied: usize,
}

/// Pure compute: given the placeholder token id, the input prompt
/// tokens, and the per-image `N_image_tokens` counts, build the
/// post-expansion prompt vector + per-image post-expansion ranges.
///
/// Splits out of `expand_image_placeholders` so it's unit-testable
/// without a live `Engine` (which carries the tokenizer and the
/// `MlxDevice`).  The caller of this helper then allocs + populates
/// one `MlxBuffer` per range and builds the `SoftTokenData` entries.
///
/// Returns:
///   * `Ok((expanded_tokens, ranges))` where `ranges[i]` is the
///     post-expansion `[start, end)` slot for image `i`'s embedding.
///   * `Err(PlaceholderCountMismatch)` when the prompt holds a
///     different number of `img_token_id` placeholders than the
///     supplied `n_image_tokens_per_image` count vector.
///
/// Phase 2c Task #17 / iter-100.
pub(crate) fn compute_soft_token_layout(
    img_token_id: u32,
    prompt_tokens: &[u32],
    n_image_tokens_per_image: &[usize],
) -> std::result::Result<(Vec<u32>, Vec<std::ops::Range<usize>>), PlaceholderCountMismatch> {
    let placeholder_positions: Vec<usize> = prompt_tokens
        .iter()
        .enumerate()
        .filter_map(|(p, t)| if *t == img_token_id { Some(p) } else { None })
        .collect();
    if placeholder_positions.len() != n_image_tokens_per_image.len() {
        return Err(PlaceholderCountMismatch {
            placeholder_positions_found: placeholder_positions.len(),
            n_image_tokens_supplied: n_image_tokens_per_image.len(),
        });
    }
    let total_extra: usize = n_image_tokens_per_image
        .iter()
        .copied()
        .sum::<usize>()
        .saturating_sub(placeholder_positions.len()); // each placeholder already counts once
    let mut prompt_expanded: Vec<u32> = Vec::with_capacity(prompt_tokens.len() + total_extra);
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(placeholder_positions.len());
    let mut last_pos = 0usize;
    for (i, &pos) in placeholder_positions.iter().enumerate() {
        prompt_expanded.extend_from_slice(&prompt_tokens[last_pos..pos]);
        let n = n_image_tokens_per_image[i];
        let start = prompt_expanded.len();
        for _ in 0..n {
            prompt_expanded.push(img_token_id);
        }
        let end = prompt_expanded.len();
        ranges.push(start..end);
        last_pos = pos + 1;
    }
    prompt_expanded.extend_from_slice(&prompt_tokens[last_pos..]);
    Ok((prompt_expanded, ranges))
}

/// Locate `<|image|>` placeholder positions in the rendered prompt
/// tokens and EXPAND each into a contiguous run of `N_image_tokens`
/// copies (where `N_image_tokens = embeddings[i].len() / hidden`).  For
/// each expanded run, allocate an `MlxBuffer` shaped `[N_image_tokens,
/// hidden]` F32 and copy the projected vision embedding row-major,
/// then return `(prompt_tokens_expanded, soft_tokens)` where
/// `soft_tokens[i].range` indexes the post-expansion vector.
///
/// Errors (each mapped to a 500 `generation_error`):
///   * tokenizer has no `<|image|>` special-token id
///   * placeholder count != image count (template emitted wrong number)
///   * MlxDevice / buffer allocation failed
///
/// Phase 2c Task #17 / iter-99.
fn expand_image_placeholders(
    engine: &engine::Engine,
    prompt_tokens: &[u32],
    embeddings: &[Vec<f32>],
) -> std::result::Result<(Vec<u32>, Vec<engine::SoftTokenData>), Response> {
    let n_images = embeddings.len();
    let hidden = engine.hidden_size();
    let img_token_id: u32 = match engine.tokenizer().token_to_id("<|image|>") {
        Some(id) => id,
        None => {
            return Err(ApiError::generation_error(
                "tokenizer has no `<|image|>` special-token id; the loaded chat \
                 model does not support vision input through hf2q's soft-token \
                 path",
            )
            .into_response());
        }
    };
    let placeholder_positions: Vec<usize> = prompt_tokens
        .iter()
        .enumerate()
        .filter_map(|(p, t)| if *t == img_token_id { Some(p) } else { None })
        .collect();
    if placeholder_positions.len() != n_images {
        return Err(ApiError::generation_error(format!(
            "rendered prompt has {} `<|image|>` placeholder(s) but request \
             carries {} image(s); the chat template likely dropped or \
             duplicated image markers — check `tokenizer_config.json` and \
             the GGUF chat template",
            placeholder_positions.len(),
            n_images
        ))
        .into_response());
    }
    // Apple Silicon: MlxDevice::new() returns the singleton Metal
    // device.  Buffers it allocates are usable by any other GpuContext
    // / device handle in this process via shared-memory semantics, so
    // the handler can alloc + populate the soft-token buffers off the
    // worker thread.
    let mlx_dev = match mlx_native::MlxDevice::new() {
        Ok(d) => d,
        Err(e) => {
            return Err(
                ApiError::generation_error(format!("MlxDevice init failed: {e}")).into_response(),
            );
        }
    };
    let total_extra: usize = embeddings
        .iter()
        .map(|e| e.len() / hidden)
        .sum::<usize>()
        .saturating_sub(n_images); // each placeholder already counts once
    let mut prompt_expanded: Vec<u32> = Vec::with_capacity(prompt_tokens.len() + total_extra);
    let mut soft_tokens: Vec<engine::SoftTokenData> = Vec::with_capacity(n_images);
    let mut last_pos = 0usize;
    for (i, &pos) in placeholder_positions.iter().enumerate() {
        prompt_expanded.extend_from_slice(&prompt_tokens[last_pos..pos]);
        let n_image_tokens = embeddings[i].len() / hidden;
        let start = prompt_expanded.len();
        for _ in 0..n_image_tokens {
            prompt_expanded.push(img_token_id);
        }
        let end = prompt_expanded.len();
        let byte_len = n_image_tokens * hidden * std::mem::size_of::<f32>();
        let mut buf = match mlx_dev.alloc_buffer(
            byte_len,
            mlx_native::DType::F32,
            vec![n_image_tokens, hidden],
        ) {
            Ok(b) => b,
            Err(e) => {
                return Err(ApiError::generation_error(format!(
                    "soft-token buffer alloc failed (image {i}): {e}"
                ))
                .into_response());
            }
        };
        match buf.as_mut_slice::<f32>() {
            Ok(dst) => {
                debug_assert_eq!(dst.len(), embeddings[i].len());
                dst.copy_from_slice(&embeddings[i]);
            }
            Err(e) => {
                return Err(ApiError::generation_error(format!(
                    "soft-token buffer mut slice failed (image {i}): {e}"
                ))
                .into_response());
            }
        }
        soft_tokens.push(engine::SoftTokenData {
            range: start..end,
            embeddings: buf,
        });
        last_pos = pos + 1;
    }
    prompt_expanded.extend_from_slice(&prompt_tokens[last_pos..]);
    Ok((prompt_expanded, soft_tokens))
}

/// Pre-compile the request's `response_format` to a parsed GBNF grammar.
///
/// Returns `Ok(None)` for `ResponseFormat::Text` (no constraint applies).
/// Returns `Ok(Some(grammar))` for `JsonObject` / `JsonSchema` — caller
/// attaches the parsed grammar to `SamplingParams` so the decode loop's
/// mask step (iter-95+) can constrain token selection to grammar-valid
/// completions.
///
/// Returns `Err(Response)` with the OpenAI-shaped 400 `grammar_error`
/// envelope on a malformed JSON-Schema or unparseable GBNF — fails fast
/// in <1ms instead of producing garbage after N tokens.
fn compile_response_format(
    rf: &ResponseFormat,
) -> std::result::Result<Option<grammar::Grammar>, Response> {
    let gbnf = match rf {
        ResponseFormat::Text => return Ok(None),
        ResponseFormat::JsonObject => {
            // Unconstrained JSON object grammar — same shape as
            // llama.cpp's built-in json_object.gbnf.
            static JSON_OBJECT_GRAMMAR: &str = r#"root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws
ws ::= | " " | "\n" [ \t]{0,20}
"#;
            JSON_OBJECT_GRAMMAR.to_string()
        }
        ResponseFormat::JsonSchema { json_schema } => {
            // Compile json_schema → GBNF via iter-8's translator.
            match grammar::json_schema::schema_to_gbnf(&json_schema.schema) {
                Ok(g) => g,
                Err(e) => {
                    return Err(ApiError::grammar_error(format!(
                        "json_schema → GBNF failed: {}",
                        e
                    ))
                    .into_response())
                }
            }
        }
    };
    match grammar::parser::parse(&gbnf) {
        Ok(g) => Ok(Some(g)),
        Err(e) => Err(ApiError::grammar_error(format!("GBNF parse failed: {}", e))
            .into_response()),
    }
}

/// Build a 429 `queue_full` response with OpenAI-convention rate-limit
/// headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`).
/// The OpenAI error envelope is preserved; these headers supplement
/// `Retry-After` with the advisory values the client can use to pace
/// retries.
///
/// Values:
///   - `Limit`    = configured queue capacity.
///   - `Remaining` = 0 when we're returning 429 (queue is at capacity).
///   - `Reset`    = seconds until retry is likely to succeed (same as
///                  Retry-After — 1 second heuristic for queue-based backoff).
fn queue_full_with_rate_limit_headers(state: &AppState) -> Response {
    use axum::http::{header::HeaderName, HeaderValue};
    let err = ApiError::queue_full();
    let mut resp = err.into_response();
    let cap = state.config.queue_capacity as u64;
    let headers = resp.headers_mut();
    // OpenAI convention uses dashed Title-Case header names.
    if let Ok(v) = HeaderValue::from_str(&cap.to_string()) {
        headers.insert(HeaderName::from_static("x-ratelimit-limit"), v);
    }
    headers.insert(
        HeaderName::from_static("x-ratelimit-remaining"),
        HeaderValue::from_static("0"),
    );
    headers.insert(
        HeaderName::from_static("x-ratelimit-reset"),
        HeaderValue::from_static("1"),
    );
    resp
}

/// Current Unix epoch seconds (used for response `created` field).
fn chrono_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// POST /v1/embeddings  (Phase 2b — BERT-family embedding model)
// ---------------------------------------------------------------------------

/// Minimum sequence length for the BERT forward path. The BF16 tensor-
/// core matmul kernel rejects K < 32; the post-softmax `scores @ V`
/// matmul has K = seq_len, so seq_len must be ≥ 32. Tokenized inputs
/// shorter than this are right-padded with `[PAD]`, and the attention
/// mask (built from `valid_token_count` in `apply_bert_full_forward_gpu`)
/// keeps the padded positions from contaminating the embedding.
///
/// Note: an earlier `dense_mm_bf16_tensor.metal` bug required padding
/// seq_len up to a multiple of 32, but mlx-native iter 68 fixed the
/// kernel to handle partial K-tiles correctly. The handler now pads
/// only to the minimum and trusts the kernel.
const BERT_MIN_SEQ_LEN: usize = 32;

/// Handler for `POST /v1/embeddings`. Tokenizes each input string,
/// runs the full BERT forward pass on GPU, and returns OpenAI-shaped
/// `{object: "list", data: [{object: "embedding", embedding, index}],
/// model, usage}`.
///
/// Failure modes:
///   - No `--embedding-model` was supplied at startup → 503
///     `model_not_loaded` naming the requested id.
///   - `req.model` doesn't match the loaded embedding model id → 400
///     `model_not_loaded`.
///   - `encoding_format = "base64"` → 400 `invalid_request_error`
///     (only `"float"` is supported today).
///   - Tokenization or forward-pass error → 500 `generation_error`.
pub async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    // --- Chat-model embeddings fast path (Phase 2a Task #8, iter-92) ---
    //
    // When no `--embedding-model` was supplied at startup but a chat
    // model is loaded AND the request's `model` matches the chat
    // engine's id, route through `Engine::embed` (Last pooling on the
    // chat-model's last-layer hidden state).  This lets users embed
    // through the SAME loaded model they chat with — the standard
    // OpenAI-compatible workflow for any client that wants embeddings
    // and doesn't carry a separate embedding model.
    //
    // Order matters: the dedicated `--embedding-model` path takes
    // precedence when present (it's a purpose-built BERT/nomic-bert
    // encoder, faster + more accurate than chat-model Last pool).  The
    // chat-model path is the fallback for users who only loaded a chat
    // model.
    if state.embedding_config.is_none() {
        if let Some(engine) = state.engine.as_ref() {
            if req.model == engine.model_id() {
                return chat_model_embeddings(engine.clone(), req).await;
            }
            return ApiError::model_not_loaded(&req.model).into_response();
        }
        return ApiError::model_not_loaded(&req.model).into_response();
    }

    // --- Dedicated embedding model path (--embedding-model) ---
    let em = match state.embedding_config.as_ref() {
        Some(em) => em.clone(),
        None => {
            return ApiError::model_not_loaded(&req.model).into_response();
        }
    };
    if req.model != em.model_id {
        return ApiError::model_not_loaded(&req.model).into_response();
    }
    let arch = match em.arch.as_ref() {
        Some(a) => a.clone(),
        None => {
            return ApiError::generation_error(
                "embedding model has no loaded weights (server startup did not eagerly load)"
                    .to_string(),
            )
            .into_response();
        }
    };
    let hidden_size_native = arch.hidden_size();
    let max_pos = arch.max_position_embeddings();

    // --- Validate request format ---
    // OpenAI accepts {"float", "base64"}. The Python SDK defaults to
    // base64 to reduce JSON payload size; we support both. Anything
    // else is a 400.
    let want_base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return ApiError::invalid_request(
                format!("encoding_format='{other}' not supported (only 'float' or 'base64')"),
                Some("encoding_format".into()),
            )
            .into_response();
        }
    };

    // OpenAI's `dimensions` parameter is only supported on Matryoshka-
    // trained models (text-embedding-3 family). bge/mxbai are not
    // trained for arbitrary-prefix truncation; honoring `dimensions`
    // by returning the first N dims would silently degrade quality.
    // Per OpenAI's published behavior, models without Matryoshka
    // support reject the parameter with 400. We do the same: accept
    // `dimensions == em.config.hidden_size` (no-op), reject anything
    // else.
    if let Some(d) = req.dimensions {
        if d != hidden_size_native {
            return ApiError::invalid_request(
                format!(
                    "model '{}' does not support custom output dimensions (native dim is {}; \
                     only the text-embedding-3 family supports `dimensions`)",
                    em.model_id, hidden_size_native
                ),
                Some("dimensions".into()),
            )
            .into_response();
        }
    }
    let inputs = req.input.into_vec();
    if inputs.is_empty() {
        return ApiError::invalid_request(
            "input must be a non-empty string or array of strings".to_string(),
            Some("input".into()),
        )
        .into_response();
    }

    // --- Run the forward pass per input ---
    // The blocking GPU dispatch runs inside spawn_blocking so the tokio
    // runtime is not held up while the per-input forward (~ms-scale)
    // executes. Each input gets its own GraphSession + KernelRegistry —
    // amortizing kernel-compile across requests is iter-63's perf work.
    let model_id = em.model_id.clone();
    let tokenizer = em.tokenizer.clone();
    let shared_registry = state.embedding_registry.clone();

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<EmbeddingResponse> {
        use crate::inference::models::bert::bert_gpu::apply_bert_full_forward_gpu;
        use crate::inference::models::nomic_bert::apply_nomic_bert_full_forward_gpu;
        use crate::serve::api::state::EmbeddingArch;
        use mlx_native::{GraphExecutor, MlxDevice};

        // Per-call device handle. The underlying Metal device is shared
        // with the eager-loaded weights (Apple Silicon: same hardware
        // visible through every MlxDevice handle). `MlxDevice::new()` is
        // sub-millisecond; not a hot-path concern per iter-83 timing.
        let device = MlxDevice::new()
            .map_err(|e| anyhow::anyhow!("create MlxDevice for embedding forward: {e}"))?;
        let executor = GraphExecutor::new(device);

        // Lock the shared, pre-warmed kernel registry. Held for the
        // full forward pass; on a single-stream embedding workload the
        // serialization cost is negligible (forward is ~7 ms post-fix).
        // No fallback path: the registry MUST be populated before any
        // /v1/embeddings request reaches this handler — populated by
        // `cmd_serve` immediately after weight load.
        let registry_arc = shared_registry.ok_or_else(|| {
            anyhow::anyhow!(
                "embedding registry not pre-warmed (server boot did not initialize it)"
            )
        })?;

        let mut data: Vec<EmbeddingObject> = Vec::with_capacity(inputs.len());
        let mut total_tokens: usize = 0;

        for (i, input) in inputs.into_iter().enumerate() {
            // Tokenize via the llama.cpp-compatible WPM tokenizer.
            // `add_special_tokens=true` wraps the output in
            // `[CLS] ... [SEP]` — without that the embedding diverges
            // from llama-embedding's reference output.
            let raw_ids: Vec<u32> = tokenizer.encode(input.as_str(), true);
            total_tokens += raw_ids.len();

            // Pad short inputs up to the kernel's K floor (32) so the
            // post-softmax `scores @ V` matmul is dispatchable. The
            // attention mask (built from valid_token_count in the
            // forward) keeps padded positions from contaminating the
            // embedding output.
            let mut ids: Vec<u32> = raw_ids.to_vec();
            if ids.len() < BERT_MIN_SEQ_LEN {
                ids.resize(BERT_MIN_SEQ_LEN, 0u32);
            }
            if ids.len() > max_pos {
                ids.truncate(max_pos);
            }
            let seq_len = ids.len() as u32;

            // Upload ids to the device (per-request — small buffers).
            let device_ref: *const MlxDevice = executor.device() as *const _;
            // SAFETY: executor outlives this closure scope.
            let device: &MlxDevice = unsafe { &*device_ref };
            let ids_buf = device
                .alloc_buffer(
                    ids.len() * 4,
                    mlx_native::DType::U32,
                    vec![ids.len()],
                )
                .map_err(|e| anyhow::anyhow!("alloc ids buf: {e}"))?;
            // SAFETY: just-allocated u32 buffer; exclusive access.
            let s: &mut [u32] = unsafe {
                std::slice::from_raw_parts_mut(ids_buf.contents_ptr() as *mut u32, ids.len())
            };
            s.copy_from_slice(&ids);

            // Dispatch the forward pass — arch-aware. Uses the shared,
            // pre-warmed KernelRegistry: every `get_pipeline()` call
            // hits the cache (no shader compile in the hot path).
            let mut session = executor
                .begin()
                .map_err(|e| anyhow::anyhow!("begin session: {e}"))?;
            let mut registry_guard = registry_arc.lock().map_err(|e| {
                anyhow::anyhow!("embedding registry mutex poisoned: {e}")
            })?;
            // valid_token_count = the count BEFORE [PAD] padding. The
            // forward pass uses this to build the attention mask so
            // padded positions don't contaminate the embedding.
            let valid_token_count = raw_ids.len().min(ids.len()) as u32;

            let out = match &arch {
                EmbeddingArch::Bert { config, weights } => apply_bert_full_forward_gpu(
                    session.encoder_mut(),
                    &mut registry_guard,
                    device,
                    &ids_buf,
                    None, // type_ids: single-segment input
                    weights,
                    config,
                    seq_len,
                    valid_token_count,
                )?,
                EmbeddingArch::NomicBert { config, weights } => {
                    apply_nomic_bert_full_forward_gpu(
                        session.encoder_mut(),
                        &mut registry_guard,
                        device,
                        &ids_buf,
                        None, // type_ids: single-segment input
                        weights,
                        config,
                        seq_len,
                        valid_token_count,
                    )?
                }
            };
            session
                .finish()
                .map_err(|e| anyhow::anyhow!("session finish: {e}"))?;
            // Drop the registry guard early so subsequent requests can
            // begin dispatching while we serialize the response.
            drop(registry_guard);

            // Read back the [hidden] vector and encode per the request's
            // encoding_format. OpenAI's spec: float = list of f32; base64
            // = standard base64 of little-endian F32 bytes.
            let slice = out
                .as_slice::<f32>()
                .map_err(|e| anyhow::anyhow!("readback: {e}"))?;
            let payload = if want_base64 {
                let mut bytes = Vec::with_capacity(slice.len() * 4);
                for v in slice {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                use base64::Engine;
                EmbeddingPayload::Base64(base64::engine::general_purpose::STANDARD.encode(&bytes))
            } else {
                EmbeddingPayload::Float(slice.to_vec())
            };

            data.push(EmbeddingObject {
                object: "embedding",
                embedding: payload,
                index: i,
            });
        }

        Ok(EmbeddingResponse {
            object: "list",
            data,
            model: model_id,
            usage: EmbeddingUsage {
                prompt_tokens: total_tokens,
                total_tokens,
            },
        })
    })
    .await;

    match result {
        Ok(Ok(resp)) => (StatusCode::OK, Json(resp)).into_response(),
        Ok(Err(e)) => {
            // Format the full anyhow chain with `:#` so the client gets
            // every nested context, not just the topmost. Otherwise a
            // dispatch failure four contexts deep is invisible.
            ApiError::generation_error(format!("embedding forward: {e:#}")).into_response()
        }
        Err(join_err) => {
            ApiError::generation_error(format!("embedding worker panicked: {join_err}"))
                .into_response()
        }
    }
}

/// Chat-model pooled-embedding handler — invoked from `embeddings` when
/// no `--embedding-model` is loaded but the request's `model` matches
/// the chat engine's id.  Phase 2a Task #8 (iter-92).
///
/// Pooling: **Last** (final-token hidden state, post-final-RMS-norm,
/// L2-normalized).  Chat models use causal attention, so the last
/// token's hidden state is a function of the entire sequence — the
/// natural pooling for autoregressive embedders.  See
/// `MlxModelWeights::forward_embed_last` for the GPU-side semantics.
///
/// Mean / CLS pooling is intentionally NOT supported on the chat-model
/// path; users who need those should load a dedicated BERT-family
/// encoder via `--embedding-model`.
///
/// Tokenization uses the chat model's own tokenizer (`engine.tokenizer()`).
/// Inputs run sequentially through the engine's FIFO worker queue —
/// concurrent embedding requests get the same 429 + Retry-After
/// behaviour as concurrent chat completions when the queue fills.
async fn chat_model_embeddings(
    engine: super::engine::Engine,
    req: EmbeddingRequest,
) -> Response {
    let want_base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return ApiError::invalid_request(
                format!("encoding_format='{other}' not supported (only 'float' or 'base64')"),
                Some("encoding_format".into()),
            )
            .into_response();
        }
    };

    // OpenAI's `dimensions` parameter is only valid for Matryoshka-trained
    // models (text-embedding-3 family). Chat models are not Matryoshka;
    // accept dimensions == hidden_size as a no-op, reject any other value.
    let hidden_size = engine.hidden_size();
    if let Some(d) = req.dimensions {
        if d != hidden_size {
            return ApiError::invalid_request(
                format!(
                    "model '{}' does not support custom output dimensions (native dim is {}; \
                     only the text-embedding-3 family supports `dimensions`)",
                    engine.model_id(),
                    hidden_size
                ),
                Some("dimensions".into()),
            )
            .into_response();
        }
    }

    let inputs = req.input.into_vec();
    if inputs.is_empty() {
        return ApiError::invalid_request(
            "input must be a non-empty string or array of strings".to_string(),
            Some("input".into()),
        )
        .into_response();
    }

    let model_id = engine.model_id().to_string();
    let mut data: Vec<EmbeddingObject> = Vec::with_capacity(inputs.len());
    let mut total_tokens: usize = 0;

    // Resolve the model's BOS token id once per request.  llama-embedding
    // (the byte-for-byte parity bar for Phase 2a Task #8) prepends BOS
    // unconditionally for embedding inputs — that's what aligns the
    // embedding's last-token hidden state with the model's training-time
    // sequence-prefix convention.  hf2q's chat tokenizer (HuggingFace
    // `tokenizers` crate) does NOT auto-add BOS for Gemma 4 even with
    // `add_special_tokens=true`, because the Gemma tokenizer.json
    // post-processor leaves bos handling to the chat-template render
    // layer.  Probe the tokenizer's special-token map for the BOS
    // string used by the major chat-model families and prepend its id
    // manually.
    //
    // Probe order — first match wins:
    //   "<bos>"             — Gemma 1/2/3/4
    //   "<|begin_of_text|>" — Llama 3 / 3.1 / 3.2
    //   "<s>"               — Llama 1/2, Mistral
    //   "<|im_start|>"      — Qwen 2/2.5 (used as start-of-turn, treated
    //                         as bos by llama-embedding)
    //
    // Models without a recognised BOS token skip the prepend silently.
    // The `tokenizer.ggml.bos_token_id` GGUF key would be the
    // authoritative source; surfacing it through Engine (rather than
    // probing the HF tokenizer) is iter-94+ work.
    let bos_id: Option<u32> = ["<bos>", "<|begin_of_text|>", "<s>", "<|im_start|>"]
        .iter()
        .find_map(|t| engine.tokenizer().token_to_id(t));

    for (i, input) in inputs.into_iter().enumerate() {
        // Tokenize without auto-special-tokens — we manually prepend BOS
        // above to match llama-embedding's wire format byte-for-byte.
        let encoded = match engine.tokenizer().encode(input.as_str(), false) {
            Ok(e) => e,
            Err(e) => {
                return ApiError::invalid_request(
                    format!("input[{i}] tokenization failed: {e}"),
                    Some("input".into()),
                )
                .into_response();
            }
        };
        let mut prompt_tokens: Vec<u32> = encoded.get_ids().to_vec();
        if let Some(b) = bos_id {
            prompt_tokens.insert(0, b);
        }
        if prompt_tokens.is_empty() {
            return ApiError::invalid_request(
                format!("input[{i}] tokenized to zero tokens (empty after preprocessing)"),
                Some("input".into()),
            )
            .into_response();
        }
        total_tokens += prompt_tokens.len();

        // Dispatch through the engine's FIFO worker queue.  queue_full
        // maps to 429 + Retry-After in the same convention as chat
        // completions.
        let embedding: Vec<f32> = match engine.embed(prompt_tokens).await {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("{e}");
                if msg.contains("queue_full") {
                    return (
                        StatusCode::TOO_MANY_REQUESTS,
                        [(axum::http::header::RETRY_AFTER, "1")],
                        Json(serde_json::json!({
                            "error": {
                                "message": "engine queue full; retry shortly",
                                "type": "rate_limit_exceeded",
                                "param": null,
                                "code": "queue_full"
                            }
                        })),
                    )
                        .into_response();
                }
                return ApiError::generation_error(format!("chat-model embedding: {e:#}"))
                    .into_response();
            }
        };

        if embedding.len() != hidden_size {
            return ApiError::generation_error(format!(
                "chat-model embedding length {} != hidden_size {}",
                embedding.len(),
                hidden_size
            ))
            .into_response();
        }

        let payload = if want_base64 {
            let mut bytes = Vec::with_capacity(embedding.len() * 4);
            for v in &embedding {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            use base64::Engine;
            EmbeddingPayload::Base64(base64::engine::general_purpose::STANDARD.encode(&bytes))
        } else {
            EmbeddingPayload::Float(embedding)
        };

        data.push(EmbeddingObject {
            object: "embedding",
            embedding: payload,
            index: i,
        });
    }

    let resp = EmbeddingResponse {
        object: "list",
        data,
        model: model_id,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };
    (StatusCode::OK, Json(resp)).into_response()
}

// ---------------------------------------------------------------------------
// GET /metrics  (Prometheus text exposition format, Decision #11)
// ---------------------------------------------------------------------------

/// Handler for `GET /metrics`. Emits Prometheus text-exposition format
/// (version 0.0.4) directly — no Prometheus client library dependency; the
/// output is plain-text key/value with `# HELP` and `# TYPE` annotations.
///
/// Counters / gauges surfaced:
///   - `hf2q_uptime_seconds`          (gauge)
///   - `hf2q_ready`                   (gauge — 0/1)
///   - `hf2q_model_loaded`            (gauge — 0/1)
///   - `hf2q_requests_total`          (counter)
///   - `hf2q_requests_rejected_total` (counter)
///   - `hf2q_chat_completions_started`   (counter)
///   - `hf2q_chat_completions_completed` (counter)
///   - `hf2q_chat_completions_queue_full`(counter)
///   - `hf2q_sse_cancellations`       (counter)
///   - `hf2q_decode_tokens_total`     (counter)
///   - `hf2q_prompt_tokens_total`     (counter)
pub async fn metrics(State(state): State<AppState>) -> Response {
    use std::sync::atomic::Ordering;
    let m = &state.metrics;
    let ready = if state.is_ready_for_gen() { 1 } else { 0 };
    let model_loaded = if state.engine.is_some() { 1 } else { 0 };
    let body = format!(
        "\
# HELP hf2q_uptime_seconds Process uptime in seconds since bind.\n\
# TYPE hf2q_uptime_seconds gauge\n\
hf2q_uptime_seconds {uptime}\n\
# HELP hf2q_ready 1 if generation endpoints are ready, 0 during warmup.\n\
# TYPE hf2q_ready gauge\n\
hf2q_ready {ready}\n\
# HELP hf2q_model_loaded 1 if a model is loaded, 0 if HTTP-only backbone.\n\
# TYPE hf2q_model_loaded gauge\n\
hf2q_model_loaded {model}\n\
# HELP hf2q_requests_total Total HTTP requests reaching a handler (post-auth).\n\
# TYPE hf2q_requests_total counter\n\
hf2q_requests_total {req_total}\n\
# HELP hf2q_requests_rejected_total HTTP requests rejected at handler (auth/malformed).\n\
# TYPE hf2q_requests_rejected_total counter\n\
hf2q_requests_rejected_total {req_rej}\n\
# HELP hf2q_chat_completions_started Chat completion generations started.\n\
# TYPE hf2q_chat_completions_started counter\n\
hf2q_chat_completions_started {chat_start}\n\
# HELP hf2q_chat_completions_completed Chat completion generations completed successfully.\n\
# TYPE hf2q_chat_completions_completed counter\n\
hf2q_chat_completions_completed {chat_done}\n\
# HELP hf2q_chat_completions_queue_full Chat completions rejected with 429 queue_full.\n\
# TYPE hf2q_chat_completions_queue_full counter\n\
hf2q_chat_completions_queue_full {chat_429}\n\
# HELP hf2q_sse_cancellations SSE streams cancelled by client drop mid-generation.\n\
# TYPE hf2q_sse_cancellations counter\n\
hf2q_sse_cancellations {sse_cancel}\n\
# HELP hf2q_decode_tokens_total Tokens decoded across all completions.\n\
# TYPE hf2q_decode_tokens_total counter\n\
hf2q_decode_tokens_total {decode_tok}\n\
# HELP hf2q_prompt_tokens_total Prompt tokens ingested across all completions.\n\
# TYPE hf2q_prompt_tokens_total counter\n\
hf2q_prompt_tokens_total {prompt_tok}\n\
",
        uptime = state.uptime_seconds(),
        ready = ready,
        model = model_loaded,
        req_total = m.requests_total.load(Ordering::Relaxed),
        req_rej = m.requests_rejected_total.load(Ordering::Relaxed),
        chat_start = m.chat_completions_started.load(Ordering::Relaxed),
        chat_done = m.chat_completions_completed.load(Ordering::Relaxed),
        chat_429 = m.chat_completions_queue_full.load(Ordering::Relaxed),
        sse_cancel = m.sse_cancellations.load(Ordering::Relaxed),
        decode_tok = m.decode_tokens_total.load(Ordering::Relaxed),
        prompt_tok = m.prompt_tokens_total.load(Ordering::Relaxed),
    );
    // Prometheus exposition format: text/plain with a versioned content-type.
    let mut resp = (StatusCode::OK, body).into_response();
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    resp
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

/// Handler for `GET /health`. Always returns 200 while the HTTP server is
/// running. The response includes currently-loaded model id (when an engine
/// is wired into `AppState`), backend name, context length, and process
/// uptime in seconds.
pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    state
        .metrics
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    state
        .metrics
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    state
        .metrics
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    // Prepend the embedding model if one was supplied via --embedding-model.
    // Listed with `loaded: true` since the config is resident in memory even
    // if the forward-pass path isn't wired yet.
    if let Some(em) = state.embedding_config.as_ref() {
        if !models.iter().any(|m| m.id == em.model_id) {
            models.insert(
                0,
                ModelObject {
                    id: em.model_id.clone(),
                    object: "model",
                    created: chrono_seconds(),
                    owned_by: "hf2q",
                    context_length: em.arch.as_ref().map(|a| a.max_position_embeddings()),
                    // Embedding GGUFs are typically F16/F32 — we don't
                    // run infer_quant_type for them because they're
                    // identified via the --embedding-model flag, not the
                    // cache scan.
                    quant_type: None,
                    backend: Some("mlx-native"),
                    loaded: true,
                },
            );
        }
    }
    // Prepend the mmproj if one was supplied via --mmproj. Context length
    // doesn't apply to a vision-tower projector, so it's left `None`;
    // clients that care can inspect the id (file stem) to correlate with
    // their chat model. Listed with `loaded: true` — header+config are
    // resident in memory, weight loading happens on first multimodal
    // request in a later iter.
    if let Some(m) = state.mmproj.as_ref() {
        if !models.iter().any(|existing| existing.id == m.model_id) {
            models.insert(
                0,
                ModelObject {
                    id: m.model_id.clone(),
                    object: "model",
                    created: chrono_seconds(),
                    owned_by: "hf2q",
                    context_length: None,
                    quant_type: None,
                    backend: Some("mlx-native"),
                    loaded: true,
                },
            );
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

// ---------------------------------------------------------------------------
// Multimodal content pipeline tests (iter 26 — Task #14 extraction layer).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod multimodal_tests {
    use super::*;
    use crate::inference::vision::mmproj::{MmprojConfig, ProjectorType};
    use crate::serve::api::schema::{ChatMessage, ContentPart, ImageUrl, MessageContent};
    use crate::serve::api::state::LoadedMmproj;

    /// Build a tiny (4×4) PNG and base64-encode it into an OpenAI-style
    /// data URI. Keeps the test payload under 200 bytes and doesn't touch
    /// the filesystem.
    fn synthetic_png_data_uri() -> String {
        use base64::Engine;
        use image::{ImageBuffer, ImageFormat, Rgb, RgbImage};
        use std::io::Cursor;
        let img: RgbImage = ImageBuffer::from_fn(4, 4, |_x, _y| Rgb([200u8, 100, 50]));
        let mut buf: Vec<u8> = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
            .expect("encode png");
        let b64 = base64::engine::general_purpose::STANDARD.encode(&buf);
        format!("data:image/png;base64,{b64}")
    }

    /// Build a synthetic `LoadedMmproj` matching a shrunken 8×8 Gemma-4
    /// vision tower for cheap preprocess in tests. `target_size` is 8
    /// so the 4×4 synthetic PNGs resize cheaply. Uses the empty
    /// `LoadedMmprojWeights` — multimodal tests here only exercise
    /// preprocessing + config routing, not forward-pass weight math.
    fn synthetic_mmproj() -> LoadedMmproj {
        use std::sync::Arc;
        let cfg = MmprojConfig {
            image_size: 8,
            patch_size: 1,
            num_patches_side: 8,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            layer_norm_eps: 1e-6,
            projector: ProjectorType::Mlp,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        };
        let device = mlx_native::MlxDevice::new().expect("create device");
        let weights = crate::inference::vision::mmproj_weights::LoadedMmprojWeights::empty(device);
        LoadedMmproj {
            gguf_path: "/tmp/synthetic-mmproj.gguf".into(),
            config: cfg,
            arch: crate::inference::vision::mmproj::ArchProfile::Gemma4Siglip,
            weights: Arc::new(weights),
            model_id: "synthetic-mmproj".into(),
        }
    }

    fn user_text(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text(text.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn user_with_image(text: &str, image_url: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: text.into() },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: image_url.into(),
                        detail: None,
                    },
                },
            ])),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn text_only_returns_empty_and_does_not_need_mmproj() {
        // Text-only request, no mmproj configured — must succeed with an
        // empty vec so the handler proceeds to the text-only flow.
        let msgs = vec![user_text("hi")];
        let got = process_multimodal_content(&msgs, None).expect("ok");
        assert!(got.is_empty());
    }

    #[test]
    fn images_without_mmproj_return_400_no_mmproj_loaded() {
        let uri = synthetic_png_data_uri();
        let msgs = vec![user_with_image("describe this", &uri)];
        let resp = process_multimodal_content(&msgs, None).expect_err("image without mmproj should 400");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn single_image_preprocesses_to_chw_f32_tensor() {
        // With a synthetic 8×8 mmproj, a 4×4 PNG resizes to 8×8 → 3×8×8
        // f32 = 192 floats. Normalized with mean/std = 0.5 → values in
        // [-1, 1] ish range.
        let uri = synthetic_png_data_uri();
        let msgs = vec![user_with_image("describe", &uri)];
        let mmproj = synthetic_mmproj();
        let got = process_multimodal_content(&msgs, Some(&mmproj)).expect("ok");
        assert_eq!(got.len(), 1);
        let img = &got[0];
        assert_eq!(img.target_size, 8);
        assert_eq!(img.pixel_values.len(), 3 * 8 * 8);
        assert_eq!(img.source_label, "image/png");
        // Source pixel [200, 100, 50] / 255 then (x-0.5)/0.5 range:
        //   R: (200/255-0.5)/0.5 ≈ 0.569
        //   G: (100/255-0.5)/0.5 ≈ -0.216
        //   B: (50/255-0.5)/0.5 ≈ -0.608
        // With bilinear resize of a solid fill, interior pixels keep the
        // source value ± tiny filter error.
        let r = img.pixel_values[0];
        let g = img.pixel_values[64];
        let b = img.pixel_values[128];
        assert!((r - 0.569).abs() < 0.05, "R approx 0.569, got {r}");
        assert!((g - (-0.216)).abs() < 0.05, "G approx -0.216, got {g}");
        assert!((b - (-0.608)).abs() < 0.05, "B approx -0.608, got {b}");
    }

    #[test]
    fn multiple_images_preserve_message_order() {
        let uri = synthetic_png_data_uri();
        let msgs = vec![
            user_with_image("first", &uri),
            user_text("middle"),
            user_with_image("second", &uri),
        ];
        let mmproj = synthetic_mmproj();
        let got = process_multimodal_content(&msgs, Some(&mmproj)).expect("ok");
        assert_eq!(got.len(), 2);
        // Both from the same PNG source, so source_label is identical —
        // what we care about is ordering survived the scan→preprocess split.
        assert_eq!(got[0].source_label, "image/png");
        assert_eq!(got[1].source_label, "image/png");
    }

    #[test]
    fn malformed_url_returns_400_with_location() {
        let msgs = vec![user_with_image("x", "not-a-url")];
        let mmproj = synthetic_mmproj();
        let resp = process_multimodal_content(&msgs, Some(&mmproj))
            .expect_err("bad URL should 400");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn unsupported_mime_type_returns_400() {
        // GIF is deliberately rejected by parse_image_url.
        let msgs = vec![user_with_image("x", "data:image/gif;base64,R0lGODlh")];
        let mmproj = synthetic_mmproj();
        let resp = process_multimodal_content(&msgs, Some(&mmproj))
            .expect_err("gif should 400");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn malformed_png_bytes_returns_400() {
        use base64::Engine;
        // base64 of gibberish bytes → valid base64 but invalid PNG.
        let payload = base64::engine::general_purpose::STANDARD.encode(b"this is not a png");
        let uri = format!("data:image/png;base64,{payload}");
        let msgs = vec![user_with_image("x", &uri)];
        let mmproj = synthetic_mmproj();
        let resp = process_multimodal_content(&msgs, Some(&mmproj))
            .expect_err("bad PNG payload should 400");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn vit_forward_pending_response_is_501_with_messages_param() {
        let resp = vit_forward_pending_response(2);
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    }

    // -----------------------------------------------------------------------
    // ADR-005 Phase 2c iter-99: vision soft-token wiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn rewrite_messages_for_vision_placeholders_passthrough_text() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text("hello".into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let out = rewrite_messages_for_vision_placeholders(&msgs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, msgs[0].content);
    }

    #[test]
    fn rewrite_messages_for_vision_placeholders_pure_text_parts() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: "hello".into() },
                ContentPart::Text { text: " world".into() },
            ])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let out = rewrite_messages_for_vision_placeholders(&msgs);
        match out[0].content.as_ref().expect("content") {
            MessageContent::Parts(parts) => assert_eq!(parts.len(), 2),
            other => panic!("expected Parts, got {:?}", other),
        }
    }

    #[test]
    fn rewrite_messages_for_vision_placeholders_one_image() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: "see this:".into() },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,XXX".into(),
                        detail: None,
                    },
                },
            ])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let out = rewrite_messages_for_vision_placeholders(&msgs);
        match out[0].content.as_ref().expect("content") {
            MessageContent::Text(t) => {
                assert!(t.contains("<|image|>"), "got: {t}");
                assert!(t.starts_with("see this:"));
            }
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn rewrite_messages_for_vision_placeholders_two_images_two_placeholders() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: "a:".into() },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,A".into(),
                        detail: None,
                    },
                },
                ContentPart::Text { text: " b:".into() },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,B".into(),
                        detail: None,
                    },
                },
            ])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let out = rewrite_messages_for_vision_placeholders(&msgs);
        match out[0].content.as_ref().expect("content") {
            MessageContent::Text(t) => {
                let n_marks = t.matches("<|image|>").count();
                assert_eq!(n_marks, 2, "expected 2 placeholders, got {n_marks} in {t:?}");
                assert!(t.starts_with("a:"));
            }
            other => panic!("expected Text, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // compute_soft_token_layout — pure expansion math (iter-100)
    // -----------------------------------------------------------------------

    /// Empty prompt + zero images is a valid no-op identity.
    #[test]
    fn compute_soft_token_layout_empty_prompt_zero_images_returns_empty() {
        let (out, ranges) = compute_soft_token_layout(258880, &[], &[]).expect("ok");
        assert!(out.is_empty());
        assert!(ranges.is_empty());
    }

    /// Pure-text prompt (no placeholders) is passed through unchanged
    /// and yields zero ranges.
    #[test]
    fn compute_soft_token_layout_no_placeholders_passes_through() {
        let prompt = vec![1u32, 2, 3, 4, 5];
        let (out, ranges) = compute_soft_token_layout(258880, &prompt, &[]).expect("ok");
        assert_eq!(out, prompt);
        assert!(ranges.is_empty());
    }

    /// Single placeholder in the middle expands into N consecutive
    /// copies of the image token id, growing the prompt by N-1.
    #[test]
    fn compute_soft_token_layout_single_placeholder_expands_to_n_copies() {
        const IMG: u32 = 258880;
        let prompt = vec![10u32, 20, IMG, 30, 40];
        let (out, ranges) = compute_soft_token_layout(IMG, &prompt, &[4]).expect("ok");
        assert_eq!(out, vec![10, 20, IMG, IMG, IMG, IMG, 30, 40]);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 2..6);
        for slot in &out[ranges[0].clone()] {
            assert_eq!(*slot, IMG);
        }
    }

    /// Placeholder at the very start (position 0) expands cleanly.
    #[test]
    fn compute_soft_token_layout_placeholder_at_start() {
        const IMG: u32 = 258880;
        let prompt = vec![IMG, 1, 2, 3];
        let (out, ranges) = compute_soft_token_layout(IMG, &prompt, &[3]).expect("ok");
        assert_eq!(out, vec![IMG, IMG, IMG, 1, 2, 3]);
        assert_eq!(ranges[0], 0..3);
    }

    /// Placeholder at the very end (last position) expands cleanly.
    #[test]
    fn compute_soft_token_layout_placeholder_at_end() {
        const IMG: u32 = 258880;
        let prompt = vec![1, 2, 3, IMG];
        let (out, ranges) = compute_soft_token_layout(IMG, &prompt, &[2]).expect("ok");
        assert_eq!(out, vec![1, 2, 3, IMG, IMG]);
        assert_eq!(ranges[0], 3..5);
    }

    /// Two placeholders, possibly with different per-image token
    /// counts, expand independently.  Range bounds are computed in the
    /// post-expansion vector.
    #[test]
    fn compute_soft_token_layout_two_placeholders_independent_ranges() {
        const IMG: u32 = 258880;
        // Prompt: [a, IMG, b, IMG, c]; image 0 → 2 tokens, image 1 → 4 tokens.
        let prompt = vec![100u32, IMG, 200, IMG, 300];
        let (out, ranges) = compute_soft_token_layout(IMG, &prompt, &[2, 4]).expect("ok");
        assert_eq!(out.len(), 5 - 2 + 2 + 4);
        assert_eq!(ranges[0], 1..3);
        assert_eq!(ranges[1], 4..8);
        assert_eq!(out[0], 100);
        assert_eq!(out[3], 200);
        assert_eq!(out[8], 300);
    }

    /// Mismatched placeholder count vs image count returns the
    /// `PlaceholderCountMismatch` error variant carrying both counts.
    #[test]
    fn compute_soft_token_layout_mismatch_reports_both_counts() {
        const IMG: u32 = 258880;
        let prompt = vec![1, IMG, 2, IMG, 3];
        let err = compute_soft_token_layout(IMG, &prompt, &[5]).expect_err("must reject");
        assert_eq!(
            err,
            PlaceholderCountMismatch {
                placeholder_positions_found: 2,
                n_image_tokens_supplied: 1,
            }
        );
        let err = compute_soft_token_layout(IMG, &[1, 2, 3], &[5, 5]).expect_err("must reject");
        assert_eq!(
            err,
            PlaceholderCountMismatch {
                placeholder_positions_found: 0,
                n_image_tokens_supplied: 2,
            }
        );
    }

    /// Placeholder with N=0 image tokens is a degenerate case (drops
    /// the placeholder entirely without inserting copies).
    #[test]
    fn compute_soft_token_layout_zero_image_tokens_drops_placeholder() {
        const IMG: u32 = 258880;
        let prompt = vec![10u32, IMG, 20];
        let (out, ranges) = compute_soft_token_layout(IMG, &prompt, &[0]).expect("ok");
        assert_eq!(out, vec![10, 20]);
        assert_eq!(ranges[0], 1..1);
    }

    /// For every range, the corresponding slot count equals the
    /// requested count.
    #[test]
    fn compute_soft_token_layout_ranges_match_n_image_tokens() {
        const IMG: u32 = 258880;
        let prompt = vec![1u32, IMG, 2, 3, IMG, 4, IMG];
        let n_per = vec![3usize, 7, 1];
        let (_, ranges) = compute_soft_token_layout(IMG, &prompt, &n_per).expect("ok");
        for (i, range) in ranges.iter().enumerate() {
            assert_eq!(range.len(), n_per[i], "range {i} len");
        }
    }

    #[test]
    fn rewrite_messages_for_vision_placeholders_only_touches_image_messages() {
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: Some(MessageContent::Text("be helpful".into())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,X".into(),
                        detail: None,
                    },
                }])),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: Some(MessageContent::Text("ack".into())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];
        let out = rewrite_messages_for_vision_placeholders(&msgs);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].content, msgs[0].content);
        assert_eq!(out[2].content, msgs[2].content);
        match out[1].content.as_ref().expect("content") {
            MessageContent::Text(t) => assert_eq!(t, "<|image|>"),
            other => panic!("expected Text, got {:?}", other),
        }
    }
}
