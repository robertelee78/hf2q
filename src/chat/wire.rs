use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct Message {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    pub(crate) fn text(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.to_owned(),
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolFunction,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct ToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ThinkingMode {
    #[default]
    Auto,
    On,
    Off,
}

impl ThinkingMode {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "on" => Some(Self::On),
            "off" => Some(Self::Off),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }

    fn request_override(self) -> Option<bool> {
        match self {
            Self::Auto => None,
            Self::On => Some(true),
            Self::Off => Some(false),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RequestOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
    pub reasoning_effort: Option<String>,
    pub thinking: ThinkingMode,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatRequest<'a> {
    pub model: &'a str,
    pub messages: &'a [Message],
    pub stream: bool,
    pub stream_options: StreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf2q_enable_thinking: Option<bool>,
}

impl<'a> ChatRequest<'a> {
    pub(crate) fn new(
        model: &'a str,
        messages: &'a [Message],
        options: &'a RequestOptions,
    ) -> Self {
        Self {
            model,
            messages,
            stream: true,
            stream_options: StreamOptions {
                include_usage: true,
            },
            temperature: options.temperature,
            top_p: options.top_p,
            max_tokens: options.max_tokens,
            seed: options.seed,
            reasoning_effort: options.reasoning_effort.as_deref(),
            hf2q_enable_thinking: options.thinking.request_override(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatChunk {
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub x_hf2q_timing: Option<Timing>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkChoice {
    #[serde(default)]
    pub delta: ChunkDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct ChunkDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ToolCallDelta {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    pub call_type: Option<String>,
    #[serde(default)]
    pub function: Option<ToolFunctionDelta>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ToolFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokenDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokenDetails>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct PromptTokenDetails {
    pub cached_tokens: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct CompletionTokenDetails {
    pub reasoning_tokens: usize,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct Timing {
    #[serde(default)]
    pub prefill_time_secs: Option<f64>,
    #[serde(default)]
    pub decode_time_secs: Option<f64>,
    #[serde(default)]
    pub total_time_secs: Option<f64>,
    #[serde(default)]
    pub time_to_first_token_ms: Option<f64>,
    #[serde(default)]
    pub prefill_tokens_per_sec: Option<f64>,
    #[serde(default)]
    pub decode_tokens_per_sec: Option<f64>,
    #[serde(default)]
    pub gpu_sync_count: Option<u64>,
    #[serde(default)]
    pub gpu_dispatch_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelList {
    pub data: Vec<Model>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Model {
    pub id: String,
    #[serde(default)]
    pub loaded: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_request_has_no_hidden_generation_parameters() {
        let messages = vec![Message::text("user", "hello")];
        let options = RequestOptions::default();
        let request = ChatRequest::new("model-a", &messages, &options);
        let value = serde_json::to_value(request).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "model": "model-a",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
                "stream_options": {"include_usage": true}
            })
        );
    }

    #[test]
    fn thinking_auto_is_omitted_while_explicit_modes_are_serialized() {
        let messages = vec![Message::text("user", "hello")];
        let mut options = RequestOptions::default();
        options.thinking = ThinkingMode::On;
        let on = serde_json::to_value(ChatRequest::new("m", &messages, &options)).unwrap();
        assert_eq!(on["hf2q_enable_thinking"], true);
        options.thinking = ThinkingMode::Off;
        let off = serde_json::to_value(ChatRequest::new("m", &messages, &options)).unwrap();
        assert_eq!(off["hf2q_enable_thinking"], false);
    }
}
