use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};

use super::wire::{ChatChunk, Timing, ToolCall, ToolFunction, Usage};

const MAX_SSE_FRAME_BYTES: usize = 8 * 1024 * 1024;
const MAX_SSE_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StreamUpdate {
    Reasoning(String),
    Content(String),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompletedResponse {
    pub content: String,
    pub reasoning_content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
    pub usage: Option<Usage>,
    pub timing: Option<Timing>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    call_type: String,
    name: String,
    arguments: String,
}

/// Incremental SSE decoder that waits for a complete frame before decoding
/// UTF-8. Network chunks may divide a multibyte scalar or any SSE delimiter.
#[derive(Debug, Default)]
pub(crate) struct SseDecoder {
    bytes: Vec<u8>,
    content: String,
    reasoning: String,
    tools: BTreeMap<usize, PartialToolCall>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
    timing: Option<Timing>,
    saw_done: bool,
    received_bytes: usize,
}

impl SseDecoder {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Result<Vec<StreamUpdate>> {
        if self.saw_done && !bytes.iter().all(u8::is_ascii_whitespace) {
            bail!("received data after the [DONE] terminator");
        }
        self.received_bytes = self.received_bytes.saturating_add(bytes.len());
        if self.received_bytes > MAX_SSE_RESPONSE_BYTES {
            bail!("chat SSE response exceeded the 64 MiB diagnostic-client limit");
        }
        self.bytes.extend_from_slice(bytes);
        let mut updates = Vec::new();
        while let Some((frame_end, separator_len)) = next_frame(&self.bytes) {
            if frame_end > MAX_SSE_FRAME_BYTES {
                bail!("chat SSE frame exceeded the 8 MiB diagnostic-client limit");
            }
            let frame = self.bytes[..frame_end].to_vec();
            self.bytes.drain(..frame_end + separator_len);
            self.decode_frame(&frame, &mut updates)?;
        }
        if self.bytes.len() > MAX_SSE_FRAME_BYTES {
            bail!("chat SSE frame exceeded the 8 MiB diagnostic-client limit");
        }
        Ok(updates)
    }

    fn decode_frame(&mut self, frame: &[u8], updates: &mut Vec<StreamUpdate>) -> Result<()> {
        let text = std::str::from_utf8(frame).context("SSE frame was not valid UTF-8")?;
        let mut data = String::new();
        for raw_line in text.split('\n') {
            let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
            if line.starts_with(':') || line.is_empty() {
                continue;
            }
            if let Some(value) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(value.strip_prefix(' ').unwrap_or(value));
            }
        }
        if data.is_empty() {
            return Ok(());
        }
        if data == "[DONE]" {
            self.saw_done = true;
            return Ok(());
        }
        if self.saw_done {
            bail!("received an SSE event after the [DONE] terminator");
        }

        let chunk: ChatChunk = serde_json::from_str(&data).context("decode chat SSE JSON")?;
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }
        if let Some(timing) = chunk.x_hf2q_timing {
            self.timing = Some(timing);
        }
        for choice in chunk.choices {
            if let Some(fragment) = choice.delta.reasoning_content {
                self.reasoning.push_str(&fragment);
                updates.push(StreamUpdate::Reasoning(fragment));
            }
            if let Some(fragment) = choice.delta.content {
                self.content.push_str(&fragment);
                updates.push(StreamUpdate::Content(fragment));
            }
            for delta in choice.delta.tool_calls.unwrap_or_default() {
                let tool = self.tools.entry(delta.index).or_default();
                if let Some(id) = delta.id {
                    tool.id.push_str(&id);
                }
                if let Some(call_type) = delta.call_type {
                    tool.call_type.push_str(&call_type);
                }
                if let Some(function) = delta.function {
                    if let Some(name) = function.name {
                        tool.name.push_str(&name);
                    }
                    if let Some(arguments) = function.arguments {
                        tool.arguments.push_str(&arguments);
                    }
                }
            }
            if choice.finish_reason.is_some() {
                self.finish_reason = choice.finish_reason;
            }
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<CompletedResponse> {
        if !self.bytes.iter().all(u8::is_ascii_whitespace) {
            bail!("chat stream ended with an incomplete SSE frame");
        }
        if !self.saw_done {
            bail!("chat stream ended before the [DONE] terminator");
        }
        let finish_reason = self
            .finish_reason
            .context("chat stream ended without a finish reason")?;
        if finish_reason == "error" {
            bail!("server ended generation with finish_reason=error");
        }
        let mut tool_calls = Vec::with_capacity(self.tools.len());
        for (_, tool) in self.tools {
            if tool.id.is_empty() || tool.name.is_empty() {
                bail!("chat stream ended with an incomplete structured tool call");
            }
            tool_calls.push(ToolCall {
                id: tool.id,
                call_type: if tool.call_type.is_empty() {
                    "function".to_owned()
                } else {
                    tool.call_type
                },
                function: ToolFunction {
                    name: tool.name,
                    arguments: tool.arguments,
                },
            });
        }
        Ok(CompletedResponse {
            content: self.content,
            reasoning_content: self.reasoning,
            tool_calls,
            finish_reason,
            usage: self.usage,
            timing: self.timing,
        })
    }
}

fn next_frame(bytes: &[u8]) -> Option<(usize, usize)> {
    for index in 0..bytes.len().saturating_sub(1) {
        if bytes[index..].starts_with(b"\r\n\r\n") {
            return Some((index, 4));
        }
        if bytes[index..].starts_with(b"\n\n") {
            return Some((index, 2));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstructs_every_split_utf8_reasoning_tools_usage_and_timing() {
        let wire = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"café \"}}]}\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"done\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"inspect\",\"arguments\":\"{\\\"x\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6},\"x_hf2q_timing\":{\"time_to_first_token_ms\":12.5}}\n\n",
            "data: [DONE]\n\n"
        );
        let assert_completed = |completed: CompletedResponse| {
            assert_eq!(completed.reasoning_content, "café ");
            assert_eq!(completed.content, "done");
            assert_eq!(completed.tool_calls[0].function.arguments, "{\"x\":1}");
            assert_eq!(completed.usage.unwrap().total_tokens, 6);
            assert_eq!(completed.timing.unwrap().time_to_first_token_ms, Some(12.5));
        };

        for split in 0..=wire.len() {
            let mut decoder = SseDecoder::default();
            decoder.push(&wire.as_bytes()[..split]).unwrap();
            decoder.push(&wire.as_bytes()[split..]).unwrap();
            assert_completed(decoder.finish().unwrap());
        }

        let mut bytewise = SseDecoder::default();
        for byte in wire.as_bytes() {
            bytewise.push(std::slice::from_ref(byte)).unwrap();
        }
        assert_completed(bytewise.finish().unwrap());
    }

    #[test]
    fn rejects_truncated_or_error_streams() {
        let mut truncated = SseDecoder::default();
        truncated
            .push(b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")
            .unwrap();
        assert!(truncated
            .finish()
            .unwrap_err()
            .to_string()
            .contains("[DONE]"));

        let mut failed = SseDecoder::default();
        failed
            .push(concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"boom\"},\"finish_reason\":\"error\"}]}\n\n",
                "data: [DONE]\n\n"
            ).as_bytes())
            .unwrap();
        assert!(failed
            .finish()
            .unwrap_err()
            .to_string()
            .contains("finish_reason=error"));
    }

    #[test]
    fn rejects_cumulative_streams_above_the_session_bound() {
        let mut decoder = SseDecoder {
            received_bytes: MAX_SSE_RESPONSE_BYTES,
            ..Default::default()
        };
        assert!(decoder
            .push(b"x")
            .unwrap_err()
            .to_string()
            .contains("64 MiB"));
    }
}
