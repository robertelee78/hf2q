//! OpenAI-compatible request/response types for the API server.
//!
//! All types here match the OpenAI API specification so that clients like
//! Open WebUI, Cursor, and Continue can speak to hf2q without modification.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Top-level error wrapper matching the OpenAI `{"error": {...}}` format.
#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
    /// HTTP status code (not serialized in the JSON body).
    #[serde(skip)]
    pub status: StatusCode,
}

/// The inner error object within the OpenAI error envelope.
#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    /// Generic invalid request error (HTTP 400).
    pub fn invalid_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error: ApiErrorBody {
                message: message.into(),
                error_type: "invalid_request_error".into(),
                param,
                code: None,
            },
        }
    }

    /// Model not found error (HTTP 404).
    pub fn model_not_found(model_name: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error: ApiErrorBody {
                message: format!("The model '{}' does not exist", model_name),
                error_type: "invalid_request_error".into(),
                param: Some("model".into()),
                code: Some("model_not_found".into()),
            },
        }
    }

    /// Context length exceeded error (HTTP 400).
    pub fn context_length_exceeded(max_tokens: usize, actual_tokens: usize) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error: ApiErrorBody {
                message: format!(
                    "This model's maximum context length is {} tokens. \
                     However, your messages resulted in {} tokens.",
                    max_tokens, actual_tokens
                ),
                error_type: "invalid_request_error".into(),
                param: Some("messages".into()),
                code: Some("context_length_exceeded".into()),
            },
        }
    }

    /// Queue full error (HTTP 503).
    pub fn queue_full() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            error: ApiErrorBody {
                message: "Server overloaded, generation queue full".into(),
                error_type: "server_error".into(),
                param: None,
                code: Some("queue_full".into()),
            },
        }
    }

    /// Generation error (HTTP 500) -- Metal failure, etc.
    pub fn generation_error(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error: ApiErrorBody {
                message: format!("Generation failed: {}", detail.into()),
                error_type: "server_error".into(),
                param: None,
                code: Some("generation_error".into()),
            },
        }
    }

    /// Internal server error (HTTP 500) -- panics, unexpected failures.
    pub fn internal_error() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error: ApiErrorBody {
                message: "Internal server error".into(),
                error_type: "server_error".into(),
                param: None,
                code: Some("internal_error".into()),
            },
        }
    }

    /// Not found error (HTTP 404).
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error: ApiErrorBody {
                message: message.into(),
                error_type: "invalid_request_error".into(),
                param: None,
                code: None,
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        use axum::http::header;

        let status = self.status;
        let body = serde_json::to_string(&self).unwrap_or_else(|_| {
            r#"{"error":{"message":"Internal serialization error","type":"server_error","param":null,"code":null}}"#.into()
        });

        let mut response = (
            status,
            [(header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response();

        // Add Retry-After header for 503 responses
        if status == StatusCode::SERVICE_UNAVAILABLE {
            response.headers_mut().insert(
                header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("1"),
            );
        }

        response
    }
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/// Response for `GET /health`.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// A single model object in the OpenAI format.
///
/// Includes non-standard `context_length` field used by Open WebUI,
/// Continue, and other clients to discover the model's context window.
#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    /// Maximum context length in tokens (non-standard, widely supported).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<usize>,
}

/// Response for `GET /v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

// ---------------------------------------------------------------------------
// Chat Completions -- Request
// ---------------------------------------------------------------------------

/// A single message in the chat conversation.
///
/// The `content` field supports both the simple string format and the OpenAI
/// Vision API array format (with `text` and `image_url` content parts).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Message content, either as a plain string or as an array of content parts
    /// (for multimodal messages with images).
    #[serde(default)]
    pub content: Option<MessageContent>,
    /// Tool calls made by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID for tool-role messages (references a previous tool call).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Message content: either a plain string or an array of content parts.
///
/// Supports the OpenAI Vision API format:
/// ```json
/// [
///   {"type": "text", "text": "What's in this image?"},
///   {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
/// ]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain string content (most common).
    Text(String),
    /// Array of content parts (for multimodal messages).
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract the text content from this message, concatenating all text parts
    /// if this is a multimodal message.
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => {
                parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("")
            }
        }
    }

    /// Extract image URLs from multimodal content parts.
    /// Returns an empty vec for plain text content.
    pub fn image_urls(&self) -> Vec<&str> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => {
                parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::ImageUrl { image_url } => Some(image_url.url.as_str()),
                        _ => None,
                    })
                    .collect()
            }
        }
    }

    /// Check if this content contains any images.
    #[allow(dead_code)]
    pub fn has_images(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => {
                parts.iter().any(|p| matches!(p, ContentPart::ImageUrl { .. }))
            }
        }
    }

    /// Get as Option<String> for backward compatibility with code expecting
    /// the old `Option<String>` content field.
    #[allow(dead_code)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content.
    #[serde(rename = "text")]
    Text {
        text: String,
    },
    /// Image URL content (base64 data URL, file path, or HTTP URL).
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: ImageUrl,
    },
}

/// Image URL within a content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// The image URL. Supported formats:
    /// - `data:image/{format};base64,{data}` -- inline base64
    /// - `file:///path/to/image.jpg` -- local file
    /// - `/path/to/image.jpg` -- local file (shorthand)
    pub url: String,
    /// Optional detail level (not used by hf2q but accepted for compatibility).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A tool call object (forward compatibility for Epic 4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

/// Function details within a tool call (forward compatibility for Epic 4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// A tool definition (forward compatibility for Epic 4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

/// Function definition within a tool (forward compatibility for Epic 4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Stop sequence: OpenAI supports either a single string or array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequence {
    /// Convert to a Vec<String> regardless of variant.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSequence::Single(s) => vec![s],
            StopSequence::Multiple(v) => v,
        }
    }
}

/// Request body for `POST /v1/chat/completions`.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    /// Tools for function calling (forward compatibility for Epic 4).
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// Tool choice (forward compatibility for Epic 4).
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Frequency penalty [-2.0, 2.0], default 0.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty [-2.0, 2.0], default 0.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
}

// ---------------------------------------------------------------------------
// Chat Completions -- Response (non-streaming)
// ---------------------------------------------------------------------------

/// Full chat completion response (non-streaming).
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: UsageStats,
}

/// A single choice in a non-streaming response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
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
}

/// Breakdown of prompt token processing.
#[derive(Debug, Clone, Serialize)]
pub struct PromptTokensDetails {
    /// Number of prompt tokens served from the prompt cache.
    pub cached_tokens: usize,
}

// ---------------------------------------------------------------------------
// Chat Completions -- Streaming (SSE chunks)
// ---------------------------------------------------------------------------

/// A streaming chunk for SSE responses.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Usage stats, included only in the final chunk if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageStats>,
}

/// A single choice in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

/// The delta content in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool call deltas for streaming tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// A single tool call delta in a streaming chunk.
///
/// The first delta for a given index includes the `id`, `type`, and
/// `function.name`. Subsequent deltas for the same index append to
/// `function.arguments`.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallDelta {
    /// Zero-based index identifying which tool call this delta belongs to.
    pub index: usize,
    /// Unique tool call ID (present only in the first delta for this index).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Always "function" (present only in the first delta).
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    /// Function name and/or arguments fragment.
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
// Tool Choice (parsed enum)
// ---------------------------------------------------------------------------

/// Parsed tool_choice value from the request.
#[derive(Debug, Clone)]
pub enum ToolChoiceValue {
    /// "auto" or absent -- model decides.
    Auto,
    /// "none" -- skip tool call parsing entirely.
    None,
    /// "required" -- model must make a tool call.
    Required,
    /// Force a specific function by name.
    Function(String),
}

impl ToolChoiceValue {
    /// Parse from the raw serde_json::Value in the request.
    pub fn parse(value: Option<&serde_json::Value>) -> Self {
        match value {
            None => ToolChoiceValue::Auto,
            Some(serde_json::Value::String(s)) => match s.as_str() {
                "none" => ToolChoiceValue::None,
                "required" => ToolChoiceValue::Required,
                _ => ToolChoiceValue::Auto,
            },
            Some(serde_json::Value::Object(obj)) => {
                // {"type": "function", "function": {"name": "X"}}
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
// Embeddings
// ---------------------------------------------------------------------------

/// Input for the embeddings endpoint -- accepts a single string or array.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    /// Convert to a Vec<String> regardless of variant.
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
    /// Encoding format (currently only "float" is supported).
    #[serde(default)]
    pub encoding_format: Option<String>,
}

/// A single embedding object in the response.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// Response for `POST /v1/embeddings`.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
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
    fn test_queue_full_error() {
        let err = ApiError::queue_full();
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["code"], "queue_full");
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_queue_full_has_retry_after_header() {
        let err = ApiError::queue_full();
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.headers().get("retry-after").map(|v| v.to_str().unwrap()),
            Some("1")
        );
    }

    #[test]
    fn test_generation_error() {
        let err = ApiError::generation_error("Metal command buffer error");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["code"], "generation_error");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Metal command buffer error"));
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
            object: "list".into(),
            data: vec![ModelObject {
                id: "test-model".into(),
                object: "model".into(),
                created: 1234567890,
                owned_by: "hf2q".into(),
                context_length: Some(262144),
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "test-model");
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["created"], 1234567890);
        assert_eq!(json["data"][0]["owned_by"], "hf2q");
    }

    #[test]
    fn test_health_response_serialization() {
        let resp = HealthResponse {
            status: "ok".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert_eq!(json, r#"{"status":"ok"}"#);
    }

    #[test]
    fn test_chat_completion_response_serialization() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".into(),
            object: "chat.completion".into(),
            created: 1700000000,
            model: "test-model".into(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Some(MessageContent::Text("Hello!".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: UsageStats {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_tokens_details: None,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "chatcmpl-123");
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["prompt_tokens"], 10);
        assert_eq!(json["usage"]["completion_tokens"], 5);
        assert_eq!(json["usage"]["total_tokens"], 15);
    }

    #[test]
    fn test_chat_completion_chunk_serialization() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-456".into(),
            object: "chat.completion.chunk".into(),
            created: 1700000000,
            model: "test-model".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some("Hello".into()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["object"], "chat.completion.chunk");
        assert_eq!(json["choices"][0]["delta"]["content"], "Hello");
        assert!(json["choices"][0]["finish_reason"].is_null());
        // usage should be absent when None
        assert!(json.get("usage").is_none());
    }

    #[test]
    fn test_chat_completion_request_deserialization() {
        let json = r#"{
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "stream": true,
            "temperature": 0.7,
            "max_tokens": 256
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test-model");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content.as_ref().map(|c| c.text()).as_deref(), Some("Hello"));
        assert_eq!(req.stream, Some(true));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(256));
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
    }

    #[test]
    fn test_chat_completion_request_minimal() {
        let json = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream.is_none());
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.stop.is_none());
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
            object: "chat.completion.chunk".into(),
            created: 1700000000,
            model: "test-model".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(UsageStats {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                prompt_tokens_details: None,
            }),
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 30);
    }

    // --- Epic 4 tests ---

    #[test]
    fn test_tool_choice_parse_auto() {
        assert!(matches!(ToolChoiceValue::parse(None), ToolChoiceValue::Auto));
        let val = serde_json::json!("auto");
        assert!(matches!(ToolChoiceValue::parse(Some(&val)), ToolChoiceValue::Auto));
    }

    #[test]
    fn test_tool_choice_parse_none() {
        let val = serde_json::json!("none");
        assert!(matches!(ToolChoiceValue::parse(Some(&val)), ToolChoiceValue::None));
    }

    #[test]
    fn test_tool_choice_parse_required() {
        let val = serde_json::json!("required");
        assert!(matches!(ToolChoiceValue::parse(Some(&val)), ToolChoiceValue::Required));
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
    fn test_tool_call_delta_arguments_only() {
        let delta = ToolCallDelta {
            index: 0,
            id: None,
            call_type: None,
            function: Some(ToolCallFunctionDelta {
                name: None,
                arguments: Some("{\"city\":".to_string()),
            }),
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert_eq!(json["index"], 0);
        assert!(json.get("id").is_none());
        assert!(json.get("type").is_none());
        assert_eq!(json["function"]["arguments"], "{\"city\":");
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
    fn test_chat_message_with_tool_calls() {
        let json = r#"{
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
                }
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
        let calls = msg.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].function.name, "get_weather");
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
    }

    #[test]
    fn test_embedding_response_schema() {
        let resp = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![
                EmbeddingObject {
                    object: "embedding".to_string(),
                    embedding: vec![0.1, 0.2],
                    index: 0,
                },
                EmbeddingObject {
                    object: "embedding".to_string(),
                    embedding: vec![0.3, 0.4],
                    index: 1,
                },
            ],
            model: "test".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 10,
                total_tokens: 10,
            },
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
    fn test_chunk_delta_with_tool_calls_serialization() {
        let delta = ChunkDelta {
            role: None,
            content: None,
            tool_calls: Some(vec![ToolCallDelta {
                index: 0,
                id: Some("call_xyz".to_string()),
                call_type: Some("function".to_string()),
                function: Some(ToolCallFunctionDelta {
                    name: Some("search".to_string()),
                    arguments: None,
                }),
            }]),
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert!(json.get("content").is_none());
        let tc = &json["tool_calls"][0];
        assert_eq!(tc["index"], 0);
        assert_eq!(tc["id"], "call_xyz");
        assert_eq!(tc["function"]["name"], "search");
    }

    // --- Epic 5 Vision API tests ---

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
        // Non-empty text
        let content = MessageContent::Text("hello".to_string());
        assert_eq!(content.as_text_opt(), Some("hello".to_string()));

        // Empty text
        let content = MessageContent::Text("".to_string());
        assert_eq!(content.as_text_opt(), None);
    }

    #[test]
    fn test_message_content_serialization_round_trip() {
        // Plain text should serialize as a string
        let msg = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text("Hello".into())),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"], "Hello");

        // Array content should serialize as array
        let msg = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Look at this:".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,abc".into(),
                        detail: None,
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert!(json["content"].is_array());
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "Look at this:");
        assert_eq!(json["content"][1]["type"], "image_url");
        assert_eq!(json["content"][1]["image_url"]["url"], "data:image/png;base64,abc");
    }
}
