//! Request handlers for the OpenAI-compatible API endpoints.
//!
//! Each handler extracts state and request data via axum extractors,
//! validates the input, and dispatches to the inference engine.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio::sync::mpsc;
use tracing::{error, warn};

use super::schema::*;
use super::sse::{generation_events_to_sse, unix_timestamp, GenerationEvent};
use super::AppState;

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

/// Health check endpoint.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
    })
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

/// List loaded models in OpenAI format.
pub async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".into(),
        data: vec![ModelObject {
            id: state.model_name.clone(),
            object: "model".into(),
            created: state.created_at,
            owned_by: "hf2q".into(),
        }],
    })
}

// ---------------------------------------------------------------------------
// GET /v1/models/:model_id
// ---------------------------------------------------------------------------

/// Retrieve a specific model by ID.
pub async fn get_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelObject>, ApiError> {
    if model_id == state.model_name {
        Ok(Json(ModelObject {
            id: state.model_name.clone(),
            object: "model".into(),
            created: state.created_at,
            owned_by: "hf2q".into(),
        }))
    } else {
        Err(ApiError::model_not_found(&model_id))
    }
}

// ---------------------------------------------------------------------------
// Fallback (404)
// ---------------------------------------------------------------------------

/// Fallback handler for unmatched routes.
pub async fn fallback() -> ApiError {
    ApiError::not_found("Not found")
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

/// Chat completions handler supporting both streaming (SSE) and non-streaming
/// response modes.
pub async fn chat_completions(
    State(state): State<AppState>,
    body: Result<Json<ChatCompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    // Handle JSON parse failures with OpenAI error format
    let Json(req) = body.map_err(|rejection| {
        warn!(error = %rejection, "Invalid JSON in chat completion request");
        ApiError::invalid_request(
            format!("Invalid request body: {}", rejection),
            None,
        )
    })?;

    // Validate the request before acquiring a queue permit (fail fast)
    validate_chat_request(&req, &state)?;

    // Acquire a generation queue permit (non-blocking)
    let permit = state.queue.try_acquire().map_err(|_| {
        warn!(
            active = state.queue.active_count(),
            "Generation queue full, rejecting request"
        );
        ApiError::queue_full()
    })?;

    // Format the prompt using the chat template
    let prompt = format_prompt(&req.messages, &state)?;

    // Tokenize to get prompt token count and check context length
    let prompt_tokens = {
        let tokenizer = state.tokenizer.lock().unwrap();
        tokenizer
            .encode_with_special_tokens(&prompt)
            .map_err(|e| {
                ApiError::invalid_request(format!("Tokenization failed: {}", e), None)
            })?
    };
    let prompt_token_count = prompt_tokens.len();

    // Check context length before starting generation
    if prompt_token_count >= state.max_seq_len {
        return Err(ApiError::context_length_exceeded(
            state.max_seq_len,
            prompt_token_count,
        ));
    }

    // Build engine config for this request
    let temperature = req.temperature.unwrap_or(1.0);
    let top_p = req.top_p.unwrap_or(1.0);
    let max_tokens = req
        .max_tokens
        .unwrap_or(state.max_seq_len.saturating_sub(prompt_token_count));
    let stop_sequences = req
        .stop
        .map(|s| s.into_vec())
        .unwrap_or_default();

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = unix_timestamp();
    let model_name = state.model_name.clone();
    let is_streaming = req.stream.unwrap_or(false);

    if is_streaming {
        // Streaming mode: spawn_blocking + mpsc -> SSE
        let (tx, rx) = mpsc::channel::<GenerationEvent>(32);

        let engine = state.engine.clone();
        let cancellation = Arc::new(AtomicBool::new(false));
        let cancel_token = cancellation.clone();

        // Move the permit into the spawned task so it is held for the
        // duration of generation and released on completion/error.
        let _permit = permit;

        tokio::task::spawn_blocking(move || {
            run_generation_blocking(
                engine,
                prompt,
                temperature,
                top_p,
                max_tokens,
                stop_sequences,
                cancel_token,
                tx,
            );
            // _permit drops here, releasing the queue slot
            drop(_permit);
        });

        let sse = generation_events_to_sse(rx, request_id, model_name, created);
        Ok(sse.into_response())
    } else {
        // Non-streaming mode: collect all tokens, return complete response
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(32);

        let engine = state.engine.clone();
        let cancellation = Arc::new(AtomicBool::new(false));
        let cancel_token = cancellation.clone();

        let _permit_guard = permit;

        let handle = tokio::task::spawn_blocking(move || {
            run_generation_blocking(
                engine,
                prompt,
                temperature,
                top_p,
                max_tokens,
                stop_sequences,
                cancel_token,
                tx,
            );
            drop(_permit_guard);
        });

        // Collect all tokens
        let mut full_text = String::new();
        let mut completion_tokens = 0usize;
        let mut generation_error: Option<String> = None;

        while let Some(event) = rx.recv().await {
            match event {
                GenerationEvent::Token(text) => {
                    full_text.push_str(&text);
                    completion_tokens += 1;
                }
                GenerationEvent::Done {
                    prompt_tokens: _,
                    completion_tokens: ct,
                } => {
                    // Use the accurate counts from the engine
                    completion_tokens = ct;
                    break;
                }
                GenerationEvent::Error(msg) => {
                    generation_error = Some(msg);
                    break;
                }
            }
        }

        // Wait for the blocking task to finish and handle panics
        if let Err(join_err) = handle.await {
            error!(error = %join_err, "Generation task panicked");
            return Err(ApiError::internal_error());
        }

        // Check for generation errors
        if let Some(err_msg) = generation_error {
            return Err(ApiError::generation_error(err_msg));
        }

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".into(),
            created,
            model: model_name,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Some(full_text),
                    tool_calls: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: UsageStats {
                prompt_tokens: prompt_token_count,
                completion_tokens,
                total_tokens: prompt_token_count + completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

// ---------------------------------------------------------------------------
// Request validation
// ---------------------------------------------------------------------------

/// Validate a chat completion request before processing.
///
/// Checks are ordered to fail fast: empty messages, invalid roles, model
/// mismatch, and parameter ranges. Context length is checked later after
/// tokenization.
fn validate_chat_request(req: &ChatCompletionRequest, state: &AppState) -> Result<(), ApiError> {
    // Messages must be non-empty
    if req.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "'messages' must contain at least one message",
            Some("messages".into()),
        ));
    }

    // Validate roles
    for msg in &req.messages {
        match msg.role.as_str() {
            "system" | "user" | "assistant" | "tool" => {}
            other => {
                return Err(ApiError::invalid_request(
                    format!(
                        "Invalid role '{}'. Must be one of: system, user, assistant",
                        other
                    ),
                    Some("messages".into()),
                ));
            }
        }
    }

    // Validate model name (accept empty or matching)
    if !req.model.is_empty() && req.model != state.model_name {
        return Err(ApiError::model_not_found(&req.model));
    }

    // Validate temperature range
    if let Some(temp) = req.temperature {
        if !(0.0..=2.0).contains(&temp) {
            return Err(ApiError::invalid_request(
                format!(
                    "Invalid temperature {}. Must be between 0.0 and 2.0.",
                    temp
                ),
                Some("temperature".into()),
            ));
        }
    }

    // Validate max_tokens
    if let Some(max_tokens) = req.max_tokens {
        if max_tokens == 0 {
            return Err(ApiError::invalid_request(
                "max_tokens must be greater than 0",
                Some("max_tokens".into()),
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Prompt formatting
// ---------------------------------------------------------------------------

/// Apply the chat template to format messages into a single prompt string.
fn format_prompt(messages: &[ChatMessage], state: &AppState) -> Result<String, ApiError> {
    use crate::tokenizer::chat_template::Message;

    let template_messages: Vec<Message> = messages
        .iter()
        .map(|m| Message {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
        })
        .collect();

    let tokenizer = state.tokenizer.lock().unwrap();
    let bos_token = tokenizer
        .bos_id()
        .and_then(|id| tokenizer.decode(&[id]).ok())
        .unwrap_or_default();
    let eos_token = tokenizer
        .eos_id()
        .and_then(|id| tokenizer.decode(&[id]).ok())
        .unwrap_or_default();

    state
        .chat_template
        .render(&template_messages, &bos_token, &eos_token)
        .map_err(|e| {
            ApiError::invalid_request(
                format!("Chat template rendering failed: {}", e),
                Some("messages".into()),
            )
        })
}

// ---------------------------------------------------------------------------
// Blocking generation bridge
// ---------------------------------------------------------------------------

/// Run the inference engine generation loop inside a blocking thread.
///
/// Sends `GenerationEvent` messages through the mpsc channel. On completion
/// sends `Done` with usage stats; on error sends `Error`. The channel sender
/// is always dropped at the end (either explicitly or via drop), ensuring the
/// async receiver sees the channel close.
fn run_generation_blocking(
    engine: Arc<std::sync::Mutex<crate::inference::engine::InferenceEngine>>,
    prompt: String,
    temperature: f32,
    top_p: f32,
    max_tokens: usize,
    stop_sequences: Vec<String>,
    cancellation: Arc<AtomicBool>,
    tx: mpsc::Sender<GenerationEvent>,
) {
    use crate::inference::engine::EngineConfig;
    use crate::inference::sampler::SamplerConfig;

    // Wrap in catch_unwind for panic safety
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let mut engine_guard = engine.lock().unwrap();

        // Reconfigure the engine for this request's parameters.
        // The engine is behind a Mutex so we can mutate its config.
        // We create a new sampler config for each request.
        let sampler_config = SamplerConfig {
            temperature,
            top_p,
            top_k: 0,
            repetition_penalty: 1.0,
        };

        // We need to reconfigure the engine for this specific request.
        // Temporarily update the engine's config.
        engine_guard.config = EngineConfig {
            max_tokens,
            sampler: sampler_config,
            stop_sequences,
        };

        let tx_clone = tx.clone();
        let cancel_ref = cancellation.clone();

        let result = engine_guard.generate_from_formatted_prompt(&prompt, |token_text| {
            // Check cancellation
            if cancel_ref.load(Ordering::Relaxed) {
                return false;
            }
            // Send the token through the channel; if the receiver is dropped
            // (client disconnected), stop generating.
            tx_clone
                .blocking_send(GenerationEvent::Token(token_text.to_string()))
                .is_ok()
        });

        result
    }));

    match result {
        Ok(Ok((_, stats))) => {
            let _ = tx.blocking_send(GenerationEvent::Done {
                prompt_tokens: stats.prompt_tokens,
                completion_tokens: stats.generated_tokens,
            });
        }
        Ok(Err(engine_err)) => {
            error!(error = %engine_err, "Generation engine error");
            let _ = tx.blocking_send(GenerationEvent::Error(engine_err.to_string()));
        }
        Err(panic_err) => {
            let msg = if let Some(s) = panic_err.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_err.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic during generation".into()
            };
            error!(error = %msg, "Generation task panicked");
            let _ = tx.blocking_send(GenerationEvent::Error(msg));
        }
    }

    // tx is dropped here, closing the channel
}
