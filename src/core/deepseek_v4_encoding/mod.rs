//! Pure-Rust DeepSeek-V4-Flash-0731 conversation encoding.
//!
//! This is a direct behavioral port of the model repository's
//! `encoding/encoding_dsv4.py`. It intentionally does not execute Python or
//! delegate prompt construction to another process. The encoder owns the
//! stateful pieces that a conventional chat Jinja template cannot express
//! reliably: merging OpenAI tool messages, preserving tool-call order,
//! dropping old reasoning, reasoning effort, and quick-instruction tasks.

mod parser;
#[cfg(test)]
mod tests;
mod types;

pub use parser::{parse_completion, ParsedAssistant, ParsedToolCall, ParsedToolCallFunction};
pub use types::{Message, OrderedValue, ToolCall, ToolCallFunction, ToolDefinition};

use std::collections::HashMap;

use types::ContentBlock;

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const USER: &str = "<｜User｜>";
pub const ASSISTANT: &str = "<｜Assistant｜>";
pub const LATEST_REMINDER: &str = "<｜latest_reminder｜>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const DSML: &str = "｜DSML｜";

const EFFORT_HIGH: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
const EFFORT_MAX: &str = "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

const TOOLS_HEAD: &str = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n";
const TOOLS_FOOT: &str = "\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

#[derive(Debug, thiserror::Error)]
pub enum EncodingError {
    #[error("invalid DeepSeek-V4 message JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("DeepSeek-V4 message {index} has unsupported role {role:?}")]
    Role { index: usize, role: String },
    #[error("DeepSeek-V4 message {index}: {detail}")]
    Invalid { index: usize, detail: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningEffort {
    #[default]
    Low,
    High,
    Max,
}

#[derive(Debug, Clone, Copy)]
pub struct EncodeOptions {
    pub thinking_mode: ThinkingMode,
    pub drop_thinking: bool,
    pub add_bos: bool,
    pub reasoning_effort: ReasoningEffort,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            thinking_mode: ThinkingMode::Chat,
            drop_thinking: true,
            add_bos: true,
            reasoning_effort: ReasoningEffort::Low,
        }
    }
}

pub fn messages_from_json(json: &str) -> Result<Vec<Message>, EncodingError> {
    Ok(serde_json::from_str(json)?)
}

/// Encode official-message JSON without a Python runtime.
pub fn encode_json(json: &str, options: EncodeOptions) -> Result<String, EncodingError> {
    encode_messages(&messages_from_json(json)?, options)
}

pub fn encode_messages(
    messages: &[Message],
    options: EncodeOptions,
) -> Result<String, EncodingError> {
    let mut messages = merge_tool_messages(messages);
    sort_tool_results(&mut messages);
    let mut drop_thinking = options.drop_thinking;
    if messages.iter().any(|m| !m.tools.is_empty()) {
        drop_thinking = false;
    }
    if options.thinking_mode == ThinkingMode::Thinking && drop_thinking {
        messages = drop_old_thinking(messages);
    }

    let mut out = String::new();
    if options.add_bos {
        out.push_str(BOS);
    }
    for i in 0..messages.len() {
        render_message(&mut out, i, &messages, options, drop_thinking)?;
    }
    Ok(out)
}

fn merge_tool_messages(messages: &[Message]) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::new();
    for original in messages {
        let mut msg = original.clone();
        match msg.role.as_str() {
            "tool" => {
                let block = ContentBlock::ToolResult {
                    id: msg.tool_call_id.clone().unwrap_or_default(),
                    content: tool_content_text(&msg),
                };
                if out
                    .last()
                    .is_some_and(|m| m.role == "user" && !m.content_blocks.is_empty())
                {
                    out.last_mut().unwrap().content_blocks.push(block);
                } else {
                    msg.role = "user".into();
                    msg.content = None;
                    msg.content_blocks = vec![block];
                    out.push(msg);
                }
            }
            "user" => {
                let text = content_text(&msg);
                let block = ContentBlock::Text(text);
                if out.last().is_some_and(|m| {
                    m.role == "user" && !m.content_blocks.is_empty() && m.task.is_none()
                }) {
                    out.last_mut().unwrap().content_blocks.push(block);
                } else {
                    msg.content_blocks = vec![block];
                    out.push(msg);
                }
            }
            _ => out.push(msg),
        }
    }
    out
}

fn sort_tool_results(messages: &mut [Message]) {
    let mut order: HashMap<String, usize> = HashMap::new();
    for msg in messages {
        if msg.role == "assistant" && !msg.tool_calls.is_empty() {
            order.clear();
            for (i, tc) in msg.tool_calls.iter().enumerate() {
                if let Some(id) = &tc.id {
                    order.insert(id.clone(), i);
                }
            }
        } else if msg.role == "user"
            && msg
                .content_blocks
                .iter()
                .filter(|b| matches!(b, ContentBlock::ToolResult { .. }))
                .count()
                > 1
        {
            let mut results: Vec<_> = msg
                .content_blocks
                .iter()
                .filter_map(|b| {
                    if matches!(b, ContentBlock::ToolResult { .. }) {
                        Some(b.clone())
                    } else {
                        None
                    }
                })
                .collect();
            results.sort_by_key(|b| match b {
                ContentBlock::ToolResult { id, .. } => order.get(id).copied().unwrap_or(0),
                _ => 0,
            });
            let mut it = results.into_iter();
            for b in &mut msg.content_blocks {
                if matches!(b, ContentBlock::ToolResult { .. }) {
                    *b = it.next().expect("same tool block count");
                }
            }
        }
    }
}

fn drop_old_thinking(messages: Vec<Message>) -> Vec<Message> {
    let last = last_user_index(&messages);
    messages
        .into_iter()
        .enumerate()
        .filter_map(|(i, mut m)| {
            if matches!(
                m.role.as_str(),
                "user" | "system" | "tool" | "latest_reminder" | "direct_search_results"
            ) || i as isize >= last
            {
                Some(m)
            } else if m.role == "assistant" {
                m.reasoning_content = None;
                Some(m)
            } else {
                None
            }
        })
        .collect()
}

fn last_user_index(messages: &[Message]) -> isize {
    messages
        .iter()
        .rposition(|m| matches!(m.role.as_str(), "user" | "developer"))
        .map(|i| i as isize)
        .unwrap_or(-1)
}

fn content_text(msg: &Message) -> String {
    match msg.content.as_ref() {
        Some(OrderedValue::String(v)) => v.clone(),
        Some(OrderedValue::Array(parts)) => parts
            .iter()
            .filter_map(|part| ordered_object_string(part, "text"))
            .collect::<String>(),
        _ => String::new(),
    }
}

fn tool_content_text(msg: &Message) -> String {
    match msg.content.as_ref() {
        Some(OrderedValue::String(v)) => v.clone(),
        Some(OrderedValue::Array(parts)) => parts
            .iter()
            .map(|part| {
                let kind = ordered_object_string(part, "type").unwrap_or("unknown");
                if kind == "text" {
                    ordered_object_string(part, "text")
                        .unwrap_or("")
                        .to_string()
                } else {
                    format!("[Unsupported {kind}]")
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

fn ordered_object_string<'a>(value: &'a OrderedValue, key: &str) -> Option<&'a str> {
    let OrderedValue::Object(fields) = value else {
        return None;
    };
    fields
        .iter()
        .find(|(name, _)| name == key)
        .and_then(|(_, value)| value.text())
}

fn render_message(
    out: &mut String,
    index: usize,
    messages: &[Message],
    options: EncodeOptions,
    drop_thinking: bool,
) -> Result<(), EncodingError> {
    let msg = &messages[index];
    let last_user = last_user_index(messages);
    if index == 0 && options.thinking_mode == ThinkingMode::Thinking {
        out.push_str(match options.reasoning_effort {
            ReasoningEffort::Low => "",
            ReasoningEffort::High => EFFORT_HIGH,
            ReasoningEffort::Max => EFFORT_MAX,
        });
    }
    match msg.role.as_str() {
        "system" => {
            out.push_str(&content_text(msg));
            append_directives(out, msg);
        }
        "developer" => {
            if content_text(msg).is_empty() {
                return Err(invalid(index, "developer content is empty"));
            }
            out.push_str(USER);
            out.push_str(&content_text(msg));
            append_directives(out, msg);
        }
        "user" => {
            out.push_str(USER);
            if msg.content_blocks.is_empty() {
                out.push_str(&content_text(msg));
            } else {
                for (i, b) in msg.content_blocks.iter().enumerate() {
                    if i > 0 {
                        out.push_str("\n\n");
                    }
                    match b {
                        ContentBlock::Text(s) => out.push_str(s),
                        ContentBlock::ToolResult { content, .. } => {
                            out.push_str("<tool_result>");
                            out.push_str(content);
                            out.push_str("</tool_result>");
                        }
                    }
                }
            }
        }
        "latest_reminder" => {
            out.push_str(LATEST_REMINDER);
            out.push_str(&content_text(msg));
        }
        "assistant" => {
            let prev_has_task = index > 0 && messages[index - 1].task.is_some();
            if options.thinking_mode == ThinkingMode::Thinking
                && !prev_has_task
                && (!drop_thinking || index as isize > last_user)
            {
                out.push_str(msg.reasoning_content.as_deref().unwrap_or(""));
                out.push_str(THINK_END);
            }
            out.push_str(&content_text(msg));
            if !msg.tool_calls.is_empty() {
                render_tool_calls(out, &msg.tool_calls);
            }
            if !msg.wo_eos {
                out.push_str(EOS);
            }
        }
        role => {
            return Err(EncodingError::Role {
                index,
                role: role.into(),
            })
        }
    }

    if index + 1 < messages.len()
        && !matches!(
            messages[index + 1].role.as_str(),
            "assistant" | "latest_reminder"
        )
    {
        return Ok(());
    }
    if let Some(task) = &msg.task {
        let token =
            task_token(task).ok_or_else(|| invalid(index, format!("invalid task {task:?}")))?;
        if task == "action" {
            out.push_str(ASSISTANT);
            out.push_str(if options.thinking_mode == ThinkingMode::Thinking {
                THINK_START
            } else {
                THINK_END
            });
        }
        out.push_str(token);
    } else if matches!(msg.role.as_str(), "user" | "developer") {
        out.push_str(ASSISTANT);
        let open = options.thinking_mode == ThinkingMode::Thinking
            && (!drop_thinking || index as isize >= last_user);
        out.push_str(if open { THINK_START } else { THINK_END });
    }
    Ok(())
}

fn append_directives(out: &mut String, msg: &Message) {
    if !msg.tools.is_empty() {
        out.push_str("\n\n");
        out.push_str(TOOLS_HEAD);
        out.push_str(
            &msg.tools
                .iter()
                .map(ToolDefinition::schema_json)
                .collect::<Vec<_>>()
                .join("\n"),
        );
        out.push('\n');
        out.push_str(TOOLS_FOOT);
    }
    if let Some(v) = &msg.response_format {
        out.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
        out.push_str(&v.python_json());
    }
}

fn render_tool_calls(out: &mut String, calls: &[ToolCall]) {
    out.push_str("\n\n<｜DSML｜tool_calls>\n");
    for call in calls {
        out.push_str("<｜DSML｜invoke name=\"");
        out.push_str(&call.function.name);
        out.push_str("\">\n");
        let args =
            serde_json::from_str::<OrderedValue>(&call.function.arguments).unwrap_or_else(|_| {
                OrderedValue::Object(vec![(
                    "arguments".into(),
                    OrderedValue::String(call.function.arguments.clone()),
                )])
            });
        if let OrderedValue::Object(fields) = args {
            let empty = fields.is_empty();
            for (k, v) in fields {
                out.push_str("<｜DSML｜parameter name=\"");
                out.push_str(&k);
                out.push_str("\" string=\"");
                match v {
                    OrderedValue::String(s) => {
                        out.push_str("true\">");
                        out.push_str(&s);
                    }
                    other => {
                        out.push_str("false\">");
                        out.push_str(&other.python_json());
                    }
                }
                out.push_str("</｜DSML｜parameter>\n");
            }
            if empty {
                out.push('\n');
            }
        }
        out.push_str("</｜DSML｜invoke>\n");
    }
    out.push_str("</｜DSML｜tool_calls>");
}

fn task_token(task: &str) -> Option<&'static str> {
    Some(match task {
        "action" => "<｜action｜>",
        "query" => "<｜query｜>",
        "authority" => "<｜authority｜>",
        "domain" => "<｜domain｜>",
        "title" => "<｜title｜>",
        "read_url" => "<｜read_url｜>",
        _ => return None,
    })
}
fn invalid(index: usize, detail: impl Into<String>) -> EncodingError {
    EncodingError::Invalid {
        index,
        detail: detail.into(),
    }
}
