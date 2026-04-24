//! OpenAI-compatible request/response types for the hf2q API server.
//!
//! Restored from git `fe54bc2~1:src/serve/schema.rs` (the commit preceding the
//! MLX divorce at `fe54bc2`) after engine-agnostic review confirmed the file
//! contains only wire-format types with no inference-engine dependencies. The
//! restore was extended in-place to cover the OpenAI parameter surface agreed
//! in ADR-005 Phase 2 party-mode session `adr_005_phase_2` (2026-04-23):
//! Tiers 1+2+3+4, `response_format`, `stream_options`, `logprobs` +
//! `top_logprobs`, `logit_bias`, `parallel_tool_calls`, reasoning-content
//! split, and the `hf2q_overflow_policy` per-request extension.
//!
//! All types here match the OpenAI API specification so that OpenAI SDKs,
//! Open WebUI, Continue, Cursor, and other clients can speak to hf2q without
//! modification. Fields outside the OpenAI surface that hf2q needs (timings,
//! overflow policy) use the `x_hf2q_*` / `hf2q_*` naming prefix so they
//! round-trip cleanly through strict clients.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Error types (Decision #24 — OpenAI-compliant `{error: {...}}` envelope)
// ---------------------------------------------------------------------------

/// Top-level error wrapper matching the OpenAI `{"error": {...}}` format.
#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
    /// HTTP status code (not serialized in the JSON body).
    #[serde(skip)]
    pub status: StatusCode,
    /// Optional `Retry-After` header value (seconds). Populated on 429 / 503.
    #[serde(skip)]
    pub retry_after_seconds: Option<u64>,
}

/// The inner error object within the OpenAI error envelope.
///
/// Matches OpenAI's documented schema: `{message, type, param, code}`.
#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    fn bare(status: StatusCode, message: impl Into<String>, error_type: &str, code: Option<&str>, param: Option<String>) -> Self {
        Self {
            status,
            retry_after_seconds: None,
            error: ApiErrorBody {
                message: message.into(),
                error_type: error_type.into(),
                param,
                code: code.map(String::from),
            },
        }
    }

    /// Generic invalid request error (HTTP 400).
    pub fn invalid_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self::bare(StatusCode::BAD_REQUEST, message, "invalid_request_error", None, param)
    }

    /// Model not found error (HTTP 404).
    pub fn model_not_found(model_name: &str) -> Self {
        Self::bare(
            StatusCode::NOT_FOUND,
            format!("The model '{}' does not exist", model_name),
            "invalid_request_error",
            Some("model_not_found"),
            Some("model".into()),
        )
    }

    /// Model not loaded — used when a `/v1/models` entry is cached on disk but
    /// not the currently-loaded model. Phase 4 hot-swap replaces this with an
    /// auto-swap without a contract change (Decision #26).
    pub fn model_not_loaded(model_name: &str) -> Self {
        Self::bare(
            StatusCode::BAD_REQUEST,
            format!("The model '{}' is cached but not currently loaded. Start the server with `--model <path>` for this model.", model_name),
            "invalid_request_error",
            Some("model_not_loaded"),
            Some("model".into()),
        )
    }

    /// Context length exceeded error (HTTP 400). Used only when the overflow
    /// policy is `reject`; `truncate_left` and `summarize` handle it silently.
    pub fn context_length_exceeded(max_tokens: usize, actual_tokens: usize) -> Self {
        Self::bare(
            StatusCode::BAD_REQUEST,
            format!(
                "This model's maximum context length is {} tokens. However, your messages resulted in {} tokens.",
                max_tokens, actual_tokens
            ),
            "invalid_request_error",
            Some("context_length_exceeded"),
            Some("messages".into()),
        )
    }

    /// Queue full (HTTP 429) — serialized FIFO queue at hard cap (Decision #19).
    /// `Retry-After` is populated with a conservative 1-second suggestion.
    pub fn queue_full() -> Self {
        let mut e = Self::bare(
            StatusCode::TOO_MANY_REQUESTS,
            "Server is at capacity. Too many pending requests.",
            "server_error",
            Some("queue_full"),
            None,
        );
        e.retry_after_seconds = Some(1);
        e
    }

    /// Server is still warming up (HTTP 503). Emitted before `/readyz` flips to
    /// 200 (Decision #15, #16). Includes `Retry-After: 1`.
    pub fn not_ready() -> Self {
        let mut e = Self::bare(
            StatusCode::SERVICE_UNAVAILABLE,
            "Model is still warming up; please retry shortly.",
            "server_error",
            Some("not_ready"),
            None,
        );
        e.retry_after_seconds = Some(1);
        e
    }

    /// Generation error (HTTP 500) — Metal failure, decoder panic caught, etc.
    pub fn generation_error(detail: impl Into<String>) -> Self {
        Self::bare(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Generation failed: {}", detail.into()),
            "server_error",
            Some("generation_error"),
            None,
        )
    }

    /// Generic internal server error (HTTP 500).
    pub fn internal_error() -> Self {
        Self::bare(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error",
            "server_error",
            Some("internal_error"),
            None,
        )
    }

    /// Unauthorized (HTTP 401) — missing or invalid bearer token when auth is
    /// configured (Decision #8).
    pub fn unauthorized() -> Self {
        Self::bare(
            StatusCode::UNAUTHORIZED,
            "Missing or invalid authorization header.",
            "authentication_error",
            Some("invalid_api_key"),
            None,
        )
    }

    /// No mmproj configured (HTTP 400) — the request contains `image_url`
    /// content parts but the server was started without `--mmproj`.
    /// Lands in the 400 class (not 501) because the request is malformed
    /// against THIS server configuration: the client needs to either omit
    /// images or use a server instance that has a mmproj loaded.
    pub fn no_mmproj_loaded() -> Self {
        Self::bare(
            StatusCode::BAD_REQUEST,
            "Request includes image_url content parts but this server \
             was started without a multimodal projector. Start with \
             `--mmproj <path>` or send a text-only request.",
            "invalid_request_error",
            Some("no_mmproj_loaded"),
            Some("messages".into()),
        )
    }

    /// Grammar-rejection (HTTP 400) — a malformed JSON schema or GBNF grammar
    /// was supplied in `response_format` or `tools` (Decision #6).
    pub fn grammar_error(detail: impl Into<String>) -> Self {
        Self::bare(
            StatusCode::BAD_REQUEST,
            format!("Grammar compilation failed: {}", detail.into()),
            "invalid_request_error",
            Some("grammar_error"),
            Some("response_format".into()),
        )
    }

    /// Not found error (HTTP 404) — used for unmatched routes.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::bare(StatusCode::NOT_FOUND, message, "invalid_request_error", None, None)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        use axum::http::{header, HeaderValue};

        let status = self.status;
        let retry_after = self.retry_after_seconds;
        let body = serde_json::to_string(&self).unwrap_or_else(|_| {
            r#"{"error":{"message":"Internal serialization error","type":"server_error","param":null,"code":null}}"#.into()
        });

        let mut response = (
            status,
            [(header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response();

        if let Some(secs) = retry_after {
            if let Ok(val) = HeaderValue::from_str(&secs.to_string()) {
                response.headers_mut().insert(header::RETRY_AFTER, val);
            }
        }

        response
    }
}

// ---------------------------------------------------------------------------
// Health / Readyz
// ---------------------------------------------------------------------------

/// Response for `GET /health` — JSON liveness with model info (Decision #12).
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    /// "ok" when the process is alive; "error" if a core component has failed.
    pub status: String,
    /// Currently-loaded model id (path basename or user-supplied alias).
    pub model: Option<String>,
    /// Backend name (`mlx-native` under ADR-008).
    pub backend: &'static str,
    /// Model context length in tokens.
    pub context_length: Option<usize>,
    /// Process uptime in seconds.
    pub uptime_seconds: u64,
}

/// Response for `GET /readyz` — k8s-style readiness (Decision #12, #16).
#[derive(Debug, Clone, Serialize)]
pub struct ReadyzResponse {
    pub ready: bool,
    pub detail: &'static str,
}

// ---------------------------------------------------------------------------
// Models (Decision #26)
// ---------------------------------------------------------------------------

/// A single model object in the OpenAI format.
///
/// Extended with hf2q-specific fields: `quant_type`, `context_length`,
/// `backend`, `loaded`. These survive round-tripping through OpenAI SDKs
/// because SDKs preserve unknown fields in the deserialized object.
#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: &'static str,
    /// Maximum context length in tokens (non-standard, widely supported).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<usize>,
    /// GGUF quant type (`Q4_K_M`, `Q6_K`, `Q8_0`, `F16`, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant_type: Option<String>,
    /// Inference backend. Always `mlx-native` today.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<&'static str>,
    /// Whether this model is the currently-loaded one. Phase 4 hot-swap will
    /// allow multiple `loaded: true` entries; Phase 2 is exactly one.
    pub loaded: bool,
}

/// Response for `GET /v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

// ---------------------------------------------------------------------------
// Chat Completions — request
// ---------------------------------------------------------------------------

/// A single message in the chat conversation.
///
/// `content` supports both the simple string format and the OpenAI Vision API
/// array format. `reasoning_content` (Decision #21) is the OpenAI-o1-style
/// split for thinking-model reasoning traces; it is separate from `content`
/// both on input (history echo-back) and output (model response).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    /// Message content, either a plain string or an array of content parts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// Reasoning-content (OpenAI-o1-style split; Decision #21). On request
    /// echo-back, clients send it as a sibling field to `content`. On
    /// response it carries the model's pre-answer reasoning trace,
    /// delimited by per-model boundary markers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool calls made by an assistant message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID for a tool-role message (references the prior tool call).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional name (for `system` / `user` messages in OpenAI).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message content: either a plain string or an array of content parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain string content.
    Text(String),
    /// Array of content parts (for multimodal messages).
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Concatenate all text parts into a single string.
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Extract image URLs from multimodal content parts.
    pub fn image_urls(&self) -> Vec<&str> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url.url.as_str()),
                    _ => None,
                })
                .collect(),
        }
    }

    /// True if the message contains at least one image content part.
    pub fn has_images(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts.iter().any(|p| matches!(p, ContentPart::ImageUrl { .. })),
        }
    }

    /// `Some(text)` if non-empty, else `None`.
    pub fn as_text_opt(&self) -> Option<String> {
        let text = self.text();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }
}

/// A single content part within a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content (base64 data URL, file path, or HTTP URL).
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// Image URL within a content part.
///
/// Supported formats:
/// - `data:image/{format};base64,{data}` — inline base64 (Open WebUI default)
/// - `file:///path/to/image.jpg` — local file
/// - `/path/to/image.jpg` — local file (shorthand)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
    /// Optional detail level (`auto` / `low` / `high`). hf2q accepts for
    /// compatibility but does not currently branch on it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A tool call object (assistant-generated; grammar-constrained so the
/// `arguments` string is guaranteed well-formed JSON by construction).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

/// Function details within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    /// Arguments as a JSON string (per OpenAI; the string contains a JSON
    /// document, not the parsed object).
    pub arguments: String,
}

/// A tool definition in the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing the function's parameters. hf2q converts this
    /// to GBNF via the ported `json-schema-to-grammar` (Decision #6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Stop sequence: OpenAI supports either a single string or array of strings.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequence {
    /// Convert to a `Vec<String>` regardless of variant.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSequence::Single(s) => vec![s],
            StopSequence::Multiple(v) => v,
        }
    }
}

/// `response_format` parameter (Decision #6; Tier 1 surface).
///
/// Three shapes are accepted:
///   `{"type": "text"}`         — unconstrained (default).
///   `{"type": "json_object"}`  — legacy "any valid JSON" constraint.
///   `{"type": "json_schema",
///     "json_schema": {"name": "...", "schema": {...}, "strict": true}}`
///                              — schema-constrained JSON via the ported
///                                 `json-schema-to-grammar` + GBNF sampler.
///
/// All three compile down to a grammar the sampler applies token-by-token.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema {
        json_schema: JsonSchemaSpec,
    },
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct JsonSchemaSpec {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub schema: serde_json::Value,
    /// If `true`, the schema must match exactly (OpenAI `strict: true`).
    /// hf2q treats `null`/`false`/absent as the same "not strict" mode.
    #[serde(default)]
    pub strict: Option<bool>,
}

/// `stream_options` parameter (Tier 2 surface).
///
/// Currently only `include_usage` is specified by OpenAI.
#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

/// `logit_bias` parameter (Tier 4 surface) — raw OpenAI shape is
/// `{token_id_string: bias_float}`; we parse into a typed map.
pub type LogitBiasMap = std::collections::HashMap<String, f32>;

/// Per-request overflow-policy override (hf2q extension, Decision #23).
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OverflowPolicy {
    /// Return HTTP 400 `context_length_exceeded` — classic behavior.
    Reject,
    /// Drop oldest non-system messages until the prompt fits.
    TruncateLeft,
    /// Summarize oldest non-system messages; replace in-place with a
    /// synthetic `system` message "[Summary of prior conversation]: ...".
    #[default]
    Summarize,
}

/// Request body for `POST /v1/chat/completions` — the full Phase 2a surface
/// (Tiers 1+2+3+4 per Decision #22).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    // --- Tier 1: core ---
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// OpenAI's newer replacement for `max_tokens`. When both are set,
    /// `max_completion_tokens` wins.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,

    // --- Tier 2: important ---
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    // --- Tier 3: llama.cpp / ollama extensions ---
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,

    // --- Tier 4: power-user ---
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub logit_bias: Option<LogitBiasMap>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    // --- hf2q extensions ---
    /// Per-request overflow policy override (Decision #23).
    #[serde(default)]
    pub hf2q_overflow_policy: Option<OverflowPolicy>,
}

// ---------------------------------------------------------------------------
// Chat Completions — response (non-streaming)
// ---------------------------------------------------------------------------

/// Extended timing information returned alongside chat completions (hf2q).
///
/// Serialized into `x_hf2q_timing`. Omitted when not populated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimingInfo {
    pub prefill_time_secs: f64,
    pub decode_time_secs: f64,
    pub total_time_secs: f64,
    /// Time from request start to the first sampled token (milliseconds).
    pub time_to_first_token_ms: f64,
    pub prefill_tokens_per_sec: f64,
    pub decode_tokens_per_sec: f64,
    /// Number of GPU command-buffer commits during the request. Useful for
    /// cross-run perf regression detection.
    pub gpu_sync_count: u64,
    /// Number of GPU dispatches (kernel launches) during the request.
    pub gpu_dispatch_count: u64,
}

/// Full chat completion response (non-streaming).
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    /// Optional OpenAI system fingerprint (sampler+engine identity); hf2q
    /// sets `hf2q-<short-git-sha>-<mlx-native>` or omits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: UsageStats,
    /// Extended timing information (hf2q-specific). Omitted when unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_hf2q_timing: Option<TimingInfo>,
}

/// A single choice in a non-streaming response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
    /// Per-token logprobs; populated only when the request set `logprobs: true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Details about prompt token processing (OpenAI-compatible).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Breakdown of prompt-token processing (OpenAI-compatible).
#[derive(Debug, Clone, Serialize)]
pub struct PromptTokensDetails {
    /// Number of prompt tokens served from the prompt cache (Decision #24).
    pub cached_tokens: usize,
}

/// Breakdown of completion-token processing (OpenAI-compatible).
#[derive(Debug, Clone, Serialize)]
pub struct CompletionTokensDetails {
    /// Number of reasoning tokens (the portion of the completion between the
    /// model's reasoning-open and reasoning-close markers; Decision #21).
    pub reasoning_tokens: usize,
}

// ---------------------------------------------------------------------------
// Logprobs — Tier 4
// ---------------------------------------------------------------------------

/// Top-level `logprobs` object on a chat-completion choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprob>,
}

/// Per-token logprob entry.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    /// Raw token bytes (UTF-8 byte values). Useful for tokens that straddle
    /// UTF-8 boundaries and thus don't cleanly fit `token`.
    pub bytes: Option<Vec<u8>>,
    /// Top-K alternatives at this position. Empty vec if `top_logprobs` was 0.
    pub top_logprobs: Vec<TopLogprobEntry>,
}

/// A single top-K alternative logprob entry.
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

// ---------------------------------------------------------------------------
// Chat Completions — streaming (SSE chunks)
// ---------------------------------------------------------------------------

/// A streaming chunk for SSE responses.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChunkChoice>,
    /// Usage stats. Included only in the final chunk when the request set
    /// `stream_options.include_usage: true` (Tier 2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageStats>,
}

/// A single choice in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
    /// Per-token logprobs for this chunk's delta. Populated only when the
    /// request set `logprobs: true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// The delta content in a streaming chunk (Decision #21 — reasoning split).
#[derive(Debug, Clone, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// OpenAI-o1-style reasoning delta, streamed separately from `content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool call deltas for streaming tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// A single tool call delta in a streaming chunk.
///
/// The first delta for a given index includes `id`, `type`, and
/// `function.name`. Subsequent deltas for the same index append to
/// `function.arguments`.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

/// Partial function details within a streaming tool call delta.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool choice (parsed enum)
// ---------------------------------------------------------------------------

/// Parsed `tool_choice` value from the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoiceValue {
    /// "auto" or absent — the model decides.
    Auto,
    /// "none" — skip tool calling entirely.
    None,
    /// "required" — the model must emit a tool call.
    Required,
    /// Force a specific function by name.
    Function(String),
}

impl ToolChoiceValue {
    pub fn parse(value: Option<&serde_json::Value>) -> Self {
        match value {
            None => ToolChoiceValue::Auto,
            Some(serde_json::Value::String(s)) => match s.as_str() {
                "none" => ToolChoiceValue::None,
                "required" => ToolChoiceValue::Required,
                _ => ToolChoiceValue::Auto,
            },
            Some(serde_json::Value::Object(obj)) => {
                if let Some(func_obj) = obj.get("function") {
                    if let Some(name) = func_obj.get("name").and_then(|n| n.as_str()) {
                        return ToolChoiceValue::Function(name.to_string());
                    }
                }
                ToolChoiceValue::Auto
            }
            _ => ToolChoiceValue::Auto,
        }
    }
}

// ---------------------------------------------------------------------------
// Embeddings (Decision #4)
// ---------------------------------------------------------------------------

/// Input for the embeddings endpoint — accepts a single string or an array.
///
/// OpenAI also accepts `Vec<Vec<u32>>` (pre-tokenized inputs); hf2q rejects
/// that path with `invalid_request_error` until a concrete client asks for it.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Multiple(v) => v,
        }
    }
}

/// Request body for `POST /v1/embeddings`.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    /// Encoding format. hf2q supports `"float"` (default); `"base64"` returns
    /// 400 `invalid_request_error` until a concrete client needs it.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// OpenAI dimensions cap. hf2q treats as advisory — the returned vector
    /// is whatever the model produces; truncation is not performed silently.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// Optional user identifier (accepted, ignored — matches OpenAI behavior).
    #[serde(default)]
    pub user: Option<String>,
}

/// A single embedding object in the response.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// Response for `POST /v1/embeddings`.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// Token usage stats specific to the embeddings endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_error_serialization() {
        let err = ApiError::invalid_request("Something went wrong", None);
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["message"], "Something went wrong");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(json["error"]["param"].is_null());
        assert!(json["error"]["code"].is_null());
    }

    #[test]
    fn test_api_error_with_param() {
        let err = ApiError::invalid_request("Bad field", Some("messages".into()));
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["param"], "messages");
    }

    #[test]
    fn test_model_not_found_error() {
        let err = ApiError::model_not_found("gpt-5");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["code"], "model_not_found");
        assert!(json["error"]["message"].as_str().unwrap().contains("gpt-5"));
        assert_eq!(err.status, StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_model_not_loaded_error() {
        let err = ApiError::model_not_loaded("qwen3.6-27b");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.error.code.as_deref(), Some("model_not_loaded"));
        assert!(err.error.message.contains("qwen3.6-27b"));
        assert_eq!(err.error.param.as_deref(), Some("model"));
    }

    #[test]
    fn test_context_length_exceeded_error() {
        let err = ApiError::context_length_exceeded(8192, 9000);
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["code"], "context_length_exceeded");
        let msg = json["error"]["message"].as_str().unwrap();
        assert!(msg.contains("8192"));
        assert!(msg.contains("9000"));
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_queue_full_error_is_429_with_retry_after() {
        let err = ApiError::queue_full();
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response.headers().get("retry-after").and_then(|v| v.to_str().ok()),
            Some("1")
        );
    }

    #[test]
    fn test_not_ready_is_503_with_retry_after() {
        let err = ApiError::not_ready();
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.headers().get("retry-after").and_then(|v| v.to_str().ok()),
            Some("1")
        );
    }

    #[test]
    fn test_unauthorized_error() {
        let err = ApiError::unauthorized();
        assert_eq!(err.status, StatusCode::UNAUTHORIZED);
        assert_eq!(err.error.error_type, "authentication_error");
    }

    #[test]
    fn test_grammar_error() {
        let err = ApiError::grammar_error("unclosed brace at pos 42");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "grammar_error");
        assert_eq!(json["error"]["param"], "response_format");
        assert!(json["error"]["message"].as_str().unwrap().contains("unclosed brace at pos 42"));
    }

    #[test]
    fn test_generation_error() {
        let err = ApiError::generation_error("Metal command buffer error");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["code"], "generation_error");
        assert!(json["error"]["message"].as_str().unwrap().contains("Metal command buffer error"));
        assert_eq!(err.status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_internal_error() {
        let err = ApiError::internal_error();
        assert_eq!(err.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.error.code, Some("internal_error".into()));
    }

    #[test]
    fn test_model_list_response_serialization() {
        let resp = ModelListResponse {
            object: "list",
            data: vec![ModelObject {
                id: "test-model".into(),
                object: "model",
                created: 1234567890,
                owned_by: "hf2q",
                context_length: Some(262144),
                quant_type: Some("Q4_K_M".into()),
                backend: Some("mlx-native"),
                loaded: true,
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "test-model");
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["created"], 1234567890);
        assert_eq!(json["data"][0]["owned_by"], "hf2q");
        assert_eq!(json["data"][0]["context_length"], 262144);
        assert_eq!(json["data"][0]["quant_type"], "Q4_K_M");
        assert_eq!(json["data"][0]["backend"], "mlx-native");
        assert_eq!(json["data"][0]["loaded"], true);
    }

    #[test]
    fn test_health_response_serialization() {
        let resp = HealthResponse {
            status: "ok".into(),
            model: Some("gemma4-26b".into()),
            backend: "mlx-native",
            context_length: Some(262144),
            uptime_seconds: 42,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["model"], "gemma4-26b");
        assert_eq!(json["backend"], "mlx-native");
        assert_eq!(json["uptime_seconds"], 42);
    }

    #[test]
    fn test_readyz_response_serialization() {
        let resp = ReadyzResponse { ready: false, detail: "warming up" };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["ready"], false);
        assert_eq!(json["detail"], "warming up");
    }

    #[test]
    fn test_chat_completion_response_serialization() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".into(),
            object: "chat.completion",
            created: 1700000000,
            model: "test-model".into(),
            system_fingerprint: Some("hf2q-deadbeef-mlx-native".into()),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Some(MessageContent::Text("Hello!".into())),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: "stop".into(),
                logprobs: None,
            }],
            usage: UsageStats {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            },
            x_hf2q_timing: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "chatcmpl-123");
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["system_fingerprint"], "hf2q-deadbeef-mlx-native");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["prompt_tokens"], 10);
        assert_eq!(json["usage"]["completion_tokens"], 5);
        assert_eq!(json["usage"]["total_tokens"], 15);
    }

    #[test]
    fn test_chat_completion_request_all_tiers_deserialize() {
        let json = r#"{
            "model": "gemma4-26b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 100,
            "max_completion_tokens": 200,
            "temperature": 0.7,
            "stop": "END",
            "response_format": {"type": "json_object"},
            "top_p": 0.9,
            "seed": 42,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stream_options": {"include_usage": true},
            "top_k": 40,
            "repetition_penalty": 1.05,
            "min_p": 0.05,
            "logprobs": true,
            "top_logprobs": 5,
            "logit_bias": {"1234": -100.0, "5678": 100.0},
            "parallel_tool_calls": false,
            "hf2q_overflow_policy": "summarize"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gemma4-26b");
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.max_completion_tokens, Some(200));
        assert_eq!(req.temperature, Some(0.7));
        assert!(matches!(req.response_format, Some(ResponseFormat::JsonObject)));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.seed, Some(42));
        assert_eq!(req.frequency_penalty, Some(0.1));
        assert_eq!(req.presence_penalty, Some(0.2));
        assert_eq!(req.stream_options.as_ref().unwrap().include_usage, Some(true));
        assert_eq!(req.top_k, Some(40));
        assert_eq!(req.repetition_penalty, Some(1.05));
        assert_eq!(req.min_p, Some(0.05));
        assert_eq!(req.logprobs, Some(true));
        assert_eq!(req.top_logprobs, Some(5));
        assert_eq!(req.logit_bias.as_ref().unwrap().len(), 2);
        assert_eq!(req.parallel_tool_calls, Some(false));
        assert_eq!(req.hf2q_overflow_policy, Some(OverflowPolicy::Summarize));
    }

    #[test]
    fn test_response_format_json_schema_deserialize() {
        let json = r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "description": "A typed answer",
                    "schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                    "strict": true
                }
            }
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        match req.response_format {
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                assert_eq!(json_schema.name, "answer");
                assert_eq!(json_schema.description.as_deref(), Some("A typed answer"));
                assert_eq!(json_schema.strict, Some(true));
                assert!(json_schema.schema.is_object());
            }
            other => panic!("expected JsonSchema, got {:?}", other),
        }
    }

    #[test]
    fn test_chat_completion_request_minimal() {
        let json = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream.is_none());
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.max_completion_tokens.is_none());
        assert!(req.stop.is_none());
        assert!(req.response_format.is_none());
        assert!(req.seed.is_none());
        assert!(req.top_k.is_none());
        assert!(req.logprobs.is_none());
        assert!(req.hf2q_overflow_policy.is_none());
    }

    #[test]
    fn test_stop_sequence_single() {
        let json = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stop":"END"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let stops = req.stop.unwrap().into_vec();
        assert_eq!(stops, vec!["END"]);
    }

    #[test]
    fn test_stop_sequence_multiple() {
        let json =
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stop":["A","B"]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let stops = req.stop.unwrap().into_vec();
        assert_eq!(stops, vec!["A", "B"]);
    }

    #[test]
    fn test_final_chunk_with_usage() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-789".into(),
            object: "chat.completion.chunk",
            created: 1700000000,
            model: "test-model".into(),
            system_fingerprint: None,
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
                logprobs: None,
            }],
            usage: Some(UsageStats {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 30);
    }

    #[test]
    fn test_tool_choice_parse_auto() {
        assert_eq!(ToolChoiceValue::parse(None), ToolChoiceValue::Auto);
        let val = serde_json::json!("auto");
        assert_eq!(ToolChoiceValue::parse(Some(&val)), ToolChoiceValue::Auto);
    }

    #[test]
    fn test_tool_choice_parse_none() {
        let val = serde_json::json!("none");
        assert_eq!(ToolChoiceValue::parse(Some(&val)), ToolChoiceValue::None);
    }

    #[test]
    fn test_tool_choice_parse_required() {
        let val = serde_json::json!("required");
        assert_eq!(ToolChoiceValue::parse(Some(&val)), ToolChoiceValue::Required);
    }

    #[test]
    fn test_tool_choice_parse_forced_function() {
        let val = serde_json::json!({"type": "function", "function": {"name": "get_weather"}});
        match ToolChoiceValue::parse(Some(&val)) {
            ToolChoiceValue::Function(name) => assert_eq!(name, "get_weather"),
            other => panic!("Expected Function, got {:?}", other),
        }
    }

    #[test]
    fn test_tool_call_delta_serialization() {
        let delta = ToolCallDelta {
            index: 0,
            id: Some("call_abc123".to_string()),
            call_type: Some("function".to_string()),
            function: Some(ToolCallFunctionDelta {
                name: Some("get_weather".to_string()),
                arguments: None,
            }),
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert_eq!(json["index"], 0);
        assert_eq!(json["id"], "call_abc123");
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "get_weather");
        assert!(json["function"].get("arguments").is_none());
    }

    #[test]
    fn test_chat_message_with_tool_call_id() {
        let json = r#"{"role":"tool","content":"sunny","tool_call_id":"call_123"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.content.as_ref().map(|c| c.text()).as_deref(), Some("sunny"));
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_123"));
    }

    #[test]
    fn test_chat_message_reasoning_content_round_trip() {
        let msg = ChatMessage {
            role: "assistant".into(),
            content: Some(MessageContent::Text("final answer".into())),
            reasoning_content: Some("let me think step by step...".into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["reasoning_content"], "let me think step by step...");
        assert_eq!(json["content"], "final answer");
        let round_trip: ChatMessage = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, msg);
    }

    #[test]
    fn test_chunk_delta_with_reasoning_only() {
        let delta = ChunkDelta {
            role: None,
            content: None,
            reasoning_content: Some("wait...".into()),
            tool_calls: None,
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert!(json.get("content").is_none());
        assert_eq!(json["reasoning_content"], "wait...");
    }

    #[test]
    fn test_embedding_input_single_string() {
        let json = r#""hello world""#;
        let input: EmbeddingInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.into_vec(), vec!["hello world"]);
    }

    #[test]
    fn test_embedding_input_array() {
        let json = r#"["hello", "world"]"#;
        let input: EmbeddingInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.into_vec(), vec!["hello", "world"]);
    }

    #[test]
    fn test_embedding_request_deserialize() {
        let json = r#"{"model": "gemma4", "input": "test input"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gemma4");
        assert!(matches!(req.input, EmbeddingInput::Single(_)));
        assert!(req.encoding_format.is_none());
        assert!(req.dimensions.is_none());
    }

    #[test]
    fn test_embedding_response_schema() {
        let resp = EmbeddingResponse {
            object: "list",
            data: vec![
                EmbeddingObject { object: "embedding", embedding: vec![0.1, 0.2], index: 0 },
                EmbeddingObject { object: "embedding", embedding: vec![0.3, 0.4], index: 1 },
            ],
            model: "test".to_string(),
            usage: EmbeddingUsage { prompt_tokens: 10, total_tokens: 10 },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
        assert_eq!(json["data"][0]["object"], "embedding");
        assert_eq!(json["data"][0]["index"], 0);
        assert_eq!(json["data"][1]["index"], 1);
        assert_eq!(json["usage"]["prompt_tokens"], 10);
        assert_eq!(json["usage"]["total_tokens"], 10);
    }

    #[test]
    fn test_chat_request_with_tools_deserialize() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                    }
                }
            ],
            "tool_choice": "auto"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert!(req.tool_choice.is_some());
    }

    #[test]
    fn test_overflow_policy_deserialize_each_variant() {
        for (raw, expected) in [
            ("\"reject\"", OverflowPolicy::Reject),
            ("\"truncate_left\"", OverflowPolicy::TruncateLeft),
            ("\"summarize\"", OverflowPolicy::Summarize),
        ] {
            let p: OverflowPolicy = serde_json::from_str(raw).unwrap();
            assert_eq!(p, expected);
        }
    }

    #[test]
    fn test_overflow_policy_default_is_summarize() {
        // Decision #23 — `summarize` is the default. If this test breaks on a
        // future refactor, the docs + ADR must change in lockstep.
        assert_eq!(OverflowPolicy::default(), OverflowPolicy::Summarize);
    }

    #[test]
    fn test_message_content_text_string() {
        let json = r#"{"role":"user","content":"Hello"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.as_ref().unwrap().text(), "Hello");
        assert!(!msg.content.as_ref().unwrap().has_images());
        assert!(msg.content.as_ref().unwrap().image_urls().is_empty());
    }

    #[test]
    fn test_message_content_null() {
        let json = r#"{"role":"assistant","content":null}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert!(msg.content.is_none());
    }

    #[test]
    fn test_message_content_vision_array() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        let content = msg.content.as_ref().unwrap();
        assert_eq!(content.text(), "What's in this image?");
        assert!(content.has_images());
        let urls = content.image_urls();
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "data:image/png;base64,abc123");
    }

    #[test]
    fn test_message_content_multiple_images() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,img1"}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,img2"}}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        let content = msg.content.as_ref().unwrap();
        assert_eq!(content.text(), "Compare these:");
        let urls = content.image_urls();
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "data:image/png;base64,img1");
        assert_eq!(urls[1], "data:image/jpeg;base64,img2");
    }

    #[test]
    fn test_message_content_image_url_with_detail() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "file:///tmp/test.png", "detail": "high"}}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        let content = msg.content.as_ref().unwrap();
        assert!(content.has_images());
        assert_eq!(content.image_urls()[0], "file:///tmp/test.png");
    }

    #[test]
    fn test_message_content_text_only_array() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "First part"},
                {"type": "text", "text": " second part"}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        let content = msg.content.as_ref().unwrap();
        assert_eq!(content.text(), "First part second part");
        assert!(!content.has_images());
    }

    #[test]
    fn test_message_content_as_text_opt() {
        let content = MessageContent::Text("hello".to_string());
        assert_eq!(content.as_text_opt(), Some("hello".to_string()));

        let content = MessageContent::Text("".to_string());
        assert_eq!(content.as_text_opt(), None);
    }

    #[test]
    fn test_message_content_serialization_round_trip_text() {
        let msg = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text("Hello".into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"], "Hello");
    }

    #[test]
    fn test_message_content_serialization_round_trip_parts() {
        let msg = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: "Look at this:".into() },
                ContentPart::ImageUrl {
                    image_url: ImageUrl { url: "data:image/png;base64,abc".into(), detail: None },
                },
            ])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert!(json["content"].is_array());
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "Look at this:");
        assert_eq!(json["content"][1]["type"], "image_url");
        assert_eq!(json["content"][1]["image_url"]["url"], "data:image/png;base64,abc");
    }

    #[test]
    fn test_logprobs_serialization() {
        let lp = ChoiceLogprobs {
            content: vec![TokenLogprob {
                token: "Hello".into(),
                logprob: -0.5,
                bytes: Some(vec![72, 101, 108, 108, 111]),
                top_logprobs: vec![TopLogprobEntry {
                    token: "Hi".into(),
                    logprob: -1.2,
                    bytes: Some(vec![72, 105]),
                }],
            }],
        };
        let json = serde_json::to_value(&lp).unwrap();
        assert_eq!(json["content"][0]["token"], "Hello");
        assert!((json["content"][0]["logprob"].as_f64().unwrap() - -0.5).abs() < 1e-6);
        assert_eq!(json["content"][0]["top_logprobs"][0]["token"], "Hi");
    }
}
