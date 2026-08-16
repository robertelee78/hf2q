//! Strict DSML completion parser matching the official DeepSeek-V4 encoder.

use super::{ThinkingMode, DSML, EOS, THINK_END};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub call_type: &'static str,
    pub function: ParsedToolCallFunction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedAssistant {
    pub role: &'static str,
    pub content: String,
    pub reasoning_content: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("invalid DeepSeek-V4 completion: {0}")]
    Invalid(String),
}

pub fn parse_completion(text: &str, mode: ThinkingMode) -> Result<ParsedAssistant, ParseError> {
    let mut rest = text;
    let reasoning_content = if mode == ThinkingMode::Thinking {
        let (value, tail) = split_once(rest, THINK_END, "missing </think>")?;
        rest = tail;
        value.to_string()
    } else {
        String::new()
    };

    let marker = format!("\n\n<{DSML}tool_calls>");
    let (content, tool_calls) = if let Some(pos) = rest.find(&marker) {
        let content = rest[..pos].to_string();
        let after = &rest[pos + marker.len()..];
        let end = format!("</{DSML}tool_calls>");
        let (body, tail) = split_once(after, &end, "missing DSML tool_calls end")?;
        if tail != EOS {
            return Err(ParseError::Invalid(
                "content after DSML tool calls or missing EOS".into(),
            ));
        }
        (content, parse_tool_calls_body(body)?)
    } else {
        let content = rest
            .strip_suffix(EOS)
            .ok_or_else(|| ParseError::Invalid("missing EOS".into()))?
            .to_string();
        (content, Vec::new())
    };

    for token in [super::BOS, EOS, super::THINK_START, THINK_END, DSML] {
        if content.contains(token) || reasoning_content.contains(token) {
            return Err(ParseError::Invalid(format!(
                "unexpected special token {token:?} in content"
            )));
        }
    }
    Ok(ParsedAssistant {
        role: "assistant",
        content,
        reasoning_content,
        tool_calls,
    })
}

/// Parse the contents of one DSML `tool_calls` block without requiring the
/// surrounding completion/EOS. Serving uses this when the streaming boundary
/// splitter has already consumed the outer markers.
pub fn parse_tool_calls_body(mut body: &str) -> Result<Vec<ParsedToolCall>, ParseError> {
    let invoke = format!("<{DSML}invoke name=\"");
    let invoke_end = format!("</{DSML}invoke>");
    let parameter = format!("<{DSML}parameter name=\"");
    let parameter_end = format!("</{DSML}parameter>");
    let mut calls = Vec::new();
    while !body.is_empty() {
        body = body.strip_prefix('\n').unwrap_or(body);
        if body.is_empty() {
            break;
        }
        body = body
            .strip_prefix(&invoke)
            .ok_or_else(|| ParseError::Invalid("malformed DSML invoke".into()))?;
        let (name, tail) = split_once(body, "\">\n", "malformed DSML invoke name")?;
        body = tail;
        let mut args: Vec<(String, String)> = Vec::new();
        while body.starts_with(&parameter) {
            body = &body[parameter.len()..];
            let (key, tail) = split_once(body, "\" string=\"", "malformed DSML parameter name")?;
            let (is_string, tail) = split_once(tail, "\">", "malformed DSML string flag")?;
            if is_string != "true" && is_string != "false" {
                return Err(ParseError::Invalid(
                    "DSML string flag must be true or false".into(),
                ));
            }
            let (value, tail) = split_once(tail, &parameter_end, "missing DSML parameter end")?;
            if args.iter().any(|(k, _)| k == key) {
                return Err(ParseError::Invalid(format!("duplicate parameter {key:?}")));
            }
            let encoded = if is_string == "true" {
                serde_json::to_string(value).expect("string JSON")
            } else {
                serde_json::from_str::<serde_json::Value>(value)
                    .map_err(|error| {
                        ParseError::Invalid(format!(
                            "DSML non-string parameter {key:?} is not valid JSON: {error}"
                        ))
                    })?
                    .to_string()
            };
            args.push((key.to_string(), encoded));
            body = tail.strip_prefix('\n').ok_or_else(|| {
                ParseError::Invalid("DSML parameter must end with newline".into())
            })?;
        }
        // Empty argument maps are encoded with a blank line between the
        // invoke opener and closer by the official formatter.
        body = body.strip_prefix('\n').unwrap_or(body);
        body = body
            .strip_prefix(&invoke_end)
            .ok_or_else(|| ParseError::Invalid("missing DSML invoke end".into()))?;
        let arguments = format!(
            "{{{}}}",
            args.into_iter()
                .map(|(k, v)| format!("{}: {v}", serde_json::to_string(&k).expect("key JSON")))
                .collect::<Vec<_>>()
                .join(", ")
        );
        calls.push(ParsedToolCall {
            call_type: "function",
            function: ParsedToolCallFunction {
                name: name.to_string(),
                arguments,
            },
        });
        body = body.strip_prefix('\n').unwrap_or(body);
    }
    Ok(calls)
}

fn split_once<'a>(
    text: &'a str,
    marker: &str,
    error: &str,
) -> Result<(&'a str, &'a str), ParseError> {
    text.split_once(marker)
        .ok_or_else(|| ParseError::Invalid(error.into()))
}
