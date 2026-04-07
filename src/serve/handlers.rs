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
use super::sse::{generation_events_to_sse_with_tools, unix_timestamp, GenerationEvent};
use super::tool_parser::{generate_tool_call_id, ToolCallParser, ToolParserEvent};
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
            context_length: Some(state.max_seq_len),
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
            context_length: Some(state.max_seq_len),
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

    // Parse tool_choice
    let tool_choice = ToolChoiceValue::parse(req.tool_choice.as_ref());

    // Determine which tools to pass to the template (respecting tool_choice)
    let effective_tools: Option<&Vec<Tool>> = match &tool_choice {
        ToolChoiceValue::None => None, // "none" -- omit tools from template
        ToolChoiceValue::Function(_) => {
            // Pass all tools to template; the parser will only accept the named one
            req.tools.as_ref()
        }
        _ => req.tools.as_ref(),
    };
    // Suppress warning: effective_tools is used intentionally in both branches
    let _ = &tool_choice;

    // Acquire a generation queue permit (non-blocking)
    let permit = state.queue.try_acquire().map_err(|_| {
        warn!(
            active = state.queue.active_count(),
            "Generation queue full, rejecting request"
        );
        ApiError::queue_full()
    })?;

    // Extract and preprocess images from multimodal content (if any)
    let vision_features = extract_and_encode_images(&req.messages, &state)?;

    // Format the prompt using the chat template (with tools if present)
    let prompt = format_prompt(&req.messages, effective_tools, &state)?;

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

    // Determine whether tool parsing is active
    let tools_active = matches!(tool_choice, ToolChoiceValue::Auto | ToolChoiceValue::Required | ToolChoiceValue::Function(_))
        && req.tools.is_some();
    let forced_function = match &tool_choice {
        ToolChoiceValue::Function(name) => Some(name.clone()),
        _ => None,
    };

    if is_streaming {
        // Streaming mode: spawn_blocking + mpsc -> SSE
        let (tx, rx) = mpsc::channel::<GenerationEvent>(32);

        let engine = state.engine.clone();
        let cancellation = Arc::new(AtomicBool::new(false));
        let cancel_token = cancellation.clone();

        // Move the permit into the spawned task so it is held for the
        // duration of generation and released on completion/error.
        let _permit = permit;
        let vision_features_clone = vision_features.clone();

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
                vision_features_clone,
            );
            // _permit drops here, releasing the queue slot
            drop(_permit);
        });

        let sse = generation_events_to_sse_with_tools(
            rx,
            request_id,
            model_name,
            created,
            tools_active,
            forced_function,
        );
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
                vision_features,
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

        // Parse tool calls from the full text if tools are active
        let (content, tool_calls_vec, finish_reason) = if tools_active {
            parse_tool_calls_from_text(&full_text, forced_function.as_deref())
        } else {
            (Some(full_text), None, "stop".to_string())
        };

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".into(),
            created,
            model: model_name,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: content.map(MessageContent::Text),
                    tool_calls: tool_calls_vec,
                    tool_call_id: None,
                },
                finish_reason,
            }],
            usage: UsageStats {
                prompt_tokens: prompt_token_count,
                completion_tokens,
                total_tokens: prompt_token_count + completion_tokens,
                prompt_tokens_details: None,
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

    // Validate roles and tool message requirements
    for msg in &req.messages {
        match msg.role.as_str() {
            "system" | "user" | "assistant" => {}
            "tool" => {
                // Tool role messages require a tool_call_id
                if msg.tool_call_id.is_none() || msg.tool_call_id.as_deref() == Some("") {
                    return Err(ApiError::invalid_request(
                        "Messages with role 'tool' must include a non-empty 'tool_call_id'",
                        Some("messages".into()),
                    ));
                }
            }
            other => {
                return Err(ApiError::invalid_request(
                    format!(
                        "Invalid role '{}'. Must be one of: system, user, assistant, tool",
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
///
/// When `tools` is provided (and not empty), they are passed to the template
/// via the `tools` variable so that tool-aware templates (like Gemma 4's)
/// can include tool definitions in the prompt.
fn format_prompt(
    messages: &[ChatMessage],
    tools: Option<&Vec<Tool>>,
    state: &AppState,
) -> Result<String, ApiError> {
    use crate::tokenizer::chat_template::Message;

    let template_messages: Vec<Message> = messages
        .iter()
        .map(|m| {
            // Convert tool_calls from ChatMessage to a JSON value for the template
            let tool_calls_value = m.tool_calls.as_ref().map(|calls| {
                serde_json::to_value(calls).unwrap_or(serde_json::Value::Null)
            });

            Message {
                role: m.role.clone(),
                content: m.content.as_ref().map(|c| c.text()).unwrap_or_default(),
                tool_call_id: m.tool_call_id.clone(),
                tool_calls: tool_calls_value,
            }
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

    // Convert tools to a JSON value for the template
    let tools_value = tools
        .filter(|t| !t.is_empty())
        .map(|t| serde_json::to_value(t).unwrap_or(serde_json::Value::Null));

    state
        .chat_template
        .render_with_tools(
            &template_messages,
            &bos_token,
            &eos_token,
            tools_value.as_ref(),
        )
        .map_err(|e| {
            ApiError::invalid_request(
                format!("Chat template rendering failed: {}", e),
                Some("messages".into()),
            )
        })
}

// ---------------------------------------------------------------------------
// Non-streaming tool call parsing
// ---------------------------------------------------------------------------

/// Parse tool calls from the complete generated text.
///
/// Runs the text through the `ToolCallParser` after generation completes.
/// Returns `(content, tool_calls, finish_reason)`.
/// - `content` is `Some(text)` for any text before tool calls (or `None` if only tool calls).
/// - `tool_calls` is `Some(vec)` if tool calls were detected.
/// - `finish_reason` is `"tool_calls"` if tool calls were found, otherwise `"stop"`.
fn parse_tool_calls_from_text(
    text: &str,
    forced_function: Option<&str>,
) -> (Option<String>, Option<Vec<ToolCall>>, String) {
    let mut parser = ToolCallParser::new(true, forced_function.map(|s| s.to_string()));

    let mut content_parts = String::new();
    let mut tool_calls: Vec<ToolCallBuilder> = Vec::new();

    // Feed the entire text character by character
    for ch in text.chars() {
        for event in parser.feed(&ch.to_string()) {
            match event {
                ToolParserEvent::ContentDelta(text) => {
                    content_parts.push_str(&text);
                }
                ToolParserEvent::ToolCallStart { index } => {
                    // Ensure we have a builder for this index
                    while tool_calls.len() <= index {
                        tool_calls.push(ToolCallBuilder::new());
                    }
                }
                ToolParserEvent::NameDelta { index, text } => {
                    if let Some(tc) = tool_calls.get_mut(index) {
                        tc.name.push_str(&text);
                    }
                }
                ToolParserEvent::ArgumentsDelta { index, text } => {
                    if let Some(tc) = tool_calls.get_mut(index) {
                        tc.arguments.push_str(&text);
                    }
                }
                ToolParserEvent::ToolCallEnd { .. } => {}
            }
        }
    }

    // Finalize
    for event in parser.finalize() {
        match event {
            ToolParserEvent::ContentDelta(text) => content_parts.push_str(&text),
            ToolParserEvent::ArgumentsDelta { index, text } => {
                if let Some(tc) = tool_calls.get_mut(index) {
                    tc.arguments.push_str(&text);
                }
            }
            _ => {}
        }
    }

    if parser.has_tool_calls() && !tool_calls.is_empty() {
        let calls: Vec<ToolCall> = tool_calls
            .into_iter()
            .filter(|tc| !tc.name.is_empty())
            .map(|tc| ToolCall {
                id: generate_tool_call_id(),
                call_type: "function".to_string(),
                function: ToolCallFunction {
                    name: tc.name,
                    arguments: tc.arguments,
                },
            })
            .collect();

        let content = if content_parts.trim().is_empty() {
            None
        } else {
            Some(content_parts)
        };

        (content, Some(calls), "tool_calls".to_string())
    } else {
        (Some(text.to_string()), None, "stop".to_string())
    }
}

/// Builder for accumulating tool call parts during non-streaming parsing.
struct ToolCallBuilder {
    name: String,
    arguments: String,
}

impl ToolCallBuilder {
    fn new() -> Self {
        Self {
            name: String::new(),
            arguments: String::new(),
        }
    }
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
    vision_features: Option<VisionFeatures>,
) {
    use crate::inference::engine::EngineConfig;
    use crate::inference::sampler::SamplerConfig;

    // Wrap in catch_unwind for panic safety
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let mut engine_guard = engine.lock().unwrap();

        // Reconfigure the engine for this request's parameters.
        let sampler_config = SamplerConfig {
            temperature,
            top_p,
            top_k: 0,
            repetition_penalty: 1.0,
        };

        engine_guard.config = EngineConfig {
            max_tokens,
            sampler: sampler_config,
            stop_sequences,
        };

        let tx_clone = tx.clone();
        let cancel_ref = cancellation.clone();

        let on_token = |token_text: &str| -> bool {
            if cancel_ref.load(Ordering::Relaxed) {
                return false;
            }
            tx_clone
                .blocking_send(GenerationEvent::Token(token_text.to_string()))
                .is_ok()
        };

        let result = if let Some(ref vf) = vision_features {
            engine_guard.generate_from_formatted_prompt_with_vision(
                &prompt,
                vf.image_token_id,
                &vf.features,
                on_token,
            )
        } else {
            engine_guard.generate_from_formatted_prompt(&prompt, on_token)
        };

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

// ---------------------------------------------------------------------------
// Vision image extraction and encoding
// ---------------------------------------------------------------------------

/// Pre-computed vision features ready for injection into the text model.
#[derive(Clone)]
struct VisionFeatures {
    /// Flat f32 buffer of shape [num_vision_tokens, text_hidden_size].
    features: Vec<f32>,
    /// The token ID that marks image placeholder positions.
    image_token_id: u32,
}

/// Extract images from multimodal messages, preprocess, and encode them.
///
/// Returns `None` if no images are present in any message.
/// Returns an error if images are present but cannot be processed.
fn extract_and_encode_images(
    messages: &[ChatMessage],
    state: &AppState,
) -> Result<Option<VisionFeatures>, ApiError> {
    use crate::inference::vision::preprocessing::{
        decode_base64_image, load_image_from_path, preprocess_image,
    };
    use crate::inference::vision::config::VisionConfig;
    use crate::inference::vision::encoder::VisionEncoder;

    // Collect all image URLs from all messages
    let mut image_urls: Vec<&str> = Vec::new();
    for msg in messages {
        if let Some(ref content) = msg.content {
            let urls = content.image_urls();
            image_urls.extend(urls);
        }
    }

    if image_urls.is_empty() {
        return Ok(None);
    }

    // Parse vision config from the model config
    // We need to access the model's raw config JSON. For now, read it from
    // the engine's config path stored in AppState. Since we don't have direct
    // access to the raw config JSON from the handler, we use the vision config
    // that should be parsed during model loading.
    //
    // If the model doesn't have vision weights, return an error.
    let vision_config = state.vision_config.as_ref().ok_or_else(|| {
        ApiError::invalid_request(
            "This model does not support vision/image inputs",
            Some("messages".into()),
        )
    })?;

    // Load and preprocess each image
    let mut all_vision_features: Vec<f32> = Vec::new();

    for url in &image_urls {
        // Extract image bytes
        let image_bytes = if url.starts_with("data:") || !url.contains("://") && !url.starts_with("/") {
            // Base64 data URL or raw base64
            decode_base64_image(url).map_err(|e| {
                ApiError::invalid_request(
                    format!("Invalid image: {e}"),
                    Some("messages".into()),
                )
            })?
        } else if url.starts_with("file://") || url.starts_with("/") {
            // Local file path
            load_image_from_path(url).map_err(|e| {
                ApiError::invalid_request(
                    format!("Invalid image: {e}"),
                    Some("messages".into()),
                )
            })?
        } else {
            return Err(ApiError::invalid_request(
                format!("Unsupported image URL scheme: {url}. Supported: data: (base64), file://, or local path"),
                Some("messages".into()),
            ));
        };

        // Preprocess: decode, resize, patchify
        let preprocessed = preprocess_image(&image_bytes, vision_config).map_err(|e| {
            ApiError::invalid_request(
                format!("Invalid image: {e}"),
                Some("messages".into()),
            )
        })?;

        // Encode: run through vision encoder + projection
        let encoder = VisionEncoder::new(VisionConfig::clone(&vision_config));
        let engine_guard = state.engine.lock().unwrap();
        let weights = engine_guard.weights();

        // Check if vision weights are present
        if !VisionEncoder::has_vision_weights(weights) {
            return Err(ApiError::invalid_request(
                "Model weights do not include vision encoder weights. \
                 Use a multimodal model checkpoint.",
                Some("messages".into()),
            ));
        }

        let features = encoder.encode_image(&preprocessed, weights).map_err(|e| {
            error!(error = %e, "Vision encoder error");
            ApiError::generation_error(format!("Vision encoding failed: {e}"))
        })?;

        all_vision_features.extend_from_slice(&features);
    }

    Ok(Some(VisionFeatures {
        features: all_vision_features,
        image_token_id: vision_config.image_token_id,
    }))
}
