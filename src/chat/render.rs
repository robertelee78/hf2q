//! Scrollback-preserving rendering for diagnostic chat streams.

use std::io::Write;

use anyhow::Result;
use console::style;

use super::sse::{CompletedResponse, StreamUpdate};

pub(super) struct StreamRenderer<'a, W> {
    output: &'a mut W,
    model: String,
    saw_reasoning: bool,
    saw_content: bool,
}

impl<'a, W: Write> StreamRenderer<'a, W> {
    pub(super) fn new(output: &'a mut W, model: String) -> Self {
        Self {
            output,
            model,
            saw_reasoning: false,
            saw_content: false,
        }
    }

    pub(super) fn render(&mut self, update: StreamUpdate) -> Result<()> {
        match update {
            StreamUpdate::Reasoning(fragment) => {
                if !self.saw_reasoning {
                    writeln!(self.output, "[reasoning]")?;
                    self.saw_reasoning = true;
                }
                write!(self.output, "{}", style(fragment).dim())?;
            }
            StreamUpdate::Content(fragment) => {
                if !self.saw_content {
                    if self.saw_reasoning {
                        writeln!(self.output)?;
                    }
                    writeln!(self.output, "[answer]")?;
                    self.saw_content = true;
                }
                write!(self.output, "{fragment}")?;
            }
        }
        self.output.flush()?;
        Ok(())
    }

    pub(super) fn complete(&mut self, response: &CompletedResponse) -> Result<()> {
        if self.saw_reasoning || self.saw_content {
            writeln!(self.output)?;
        }
        for tool in &response.tool_calls {
            writeln!(self.output, "[tool call — not executed]")?;
            writeln!(self.output, "{}", serde_json::to_string_pretty(tool)?)?;
        }
        write_telemetry(self.output, &self.model, response)
    }

    pub(super) fn fail(&mut self, error: &anyhow::Error) -> Result<()> {
        if self.saw_reasoning || self.saw_content {
            writeln!(self.output)?;
        }
        writeln!(self.output, "[request not saved] {error:#}")?;
        Ok(())
    }
}

fn write_telemetry(
    output: &mut impl Write,
    model: &str,
    response: &CompletedResponse,
) -> Result<()> {
    writeln!(
        output,
        "[finish] model={} reason={}",
        model, response.finish_reason
    )?;
    if let Some(usage) = &response.usage {
        write!(
            output,
            "[usage] prompt={} completion={} total={}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        )?;
        if let Some(details) = &usage.prompt_tokens_details {
            write!(output, " cached={}", details.cached_tokens)?;
        }
        if let Some(details) = &usage.completion_tokens_details {
            write!(output, " reasoning={}", details.reasoning_tokens)?;
        }
        writeln!(output)?;
    }
    if let Some(timing) = &response.timing {
        let mut values = Vec::new();
        if let Some(value) = timing.time_to_first_token_ms {
            values.push(format!("ttft={value:.1}ms"));
        }
        if let Some(value) = timing.prefill_time_secs {
            values.push(format!("prefill_time={value:.3}s"));
        }
        if let Some(value) = timing.decode_time_secs {
            values.push(format!("decode_time={value:.3}s"));
        }
        if let Some(value) = timing.prefill_tokens_per_sec {
            values.push(format!("prefill={value:.1} tok/s"));
        }
        if let Some(value) = timing.decode_tokens_per_sec {
            values.push(format!("decode={value:.1} tok/s"));
        }
        if let Some(value) = timing.total_time_secs {
            values.push(format!("total={value:.3}s"));
        }
        if let Some(value) = timing.gpu_sync_count {
            values.push(format!("gpu_sync={value}"));
        }
        if let Some(value) = timing.gpu_dispatch_count {
            values.push(format!("gpu_dispatch={value}"));
        }
        if !values.is_empty() {
            writeln!(output, "[timing] {}", values.join(" "))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::wire::{Timing, ToolCall, ToolFunction};

    #[test]
    fn tool_call_is_displayed_as_data_and_labeled_not_executed() {
        let response = CompletedResponse {
            content: String::new(),
            reasoning_content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                call_type: "function".into(),
                function: ToolFunction {
                    name: "inspect".into(),
                    arguments: "{\"path\":\"src\"}".into(),
                },
            }],
            finish_reason: "tool_calls".into(),
            usage: None,
            timing: Some(Timing::default()),
        };
        let mut output = Vec::new();
        StreamRenderer::new(&mut output, "model-a".into())
            .complete(&response)
            .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("[tool call — not executed]"));
        assert!(output.contains("\"name\": \"inspect\""));
        assert!(output.contains("reason=tool_calls"));
    }
}
