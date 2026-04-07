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
#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    /// Tool calls made by the assistant (forward compatibility for Epic 4).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
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
                    content: Some("Hello!".into()),
                    tool_calls: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: UsageStats {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
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
        assert_eq!(req.messages[0].content.as_deref(), Some("Hello"));
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
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(UsageStats {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 30);
    }
}
