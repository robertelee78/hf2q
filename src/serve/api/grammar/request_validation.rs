use std::collections::HashSet;

use super::RequestGrammarError;
use crate::serve::api::schema::{ChatCompletionRequest, LlamaGrammarTriggerType, ToolChoiceValue};

fn valid_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

/// Validate tools and normalize the OpenAI `tool_choice` union before model
/// resolution. This prevents malformed choices or declarations from falling
/// through to automatic, unconstrained generation.
pub fn validate_tool_request(
    request: &ChatCompletionRequest,
) -> Result<ToolChoiceValue, RequestGrammarError> {
    let choice = ToolChoiceValue::try_parse(request.tool_choice.as_ref())
        .map_err(|message| RequestGrammarError::new("tool_choice", message))?;

    let tools = match request.tools.as_deref() {
        Some([]) => {
            return Err(RequestGrammarError::new(
                "tools",
                "tools must contain at least one function when specified",
            ));
        }
        Some(tools) => tools,
        None => {
            if request.tool_choice.is_some()
                && matches!(
                    choice,
                    ToolChoiceValue::Auto
                        | ToolChoiceValue::Required
                        | ToolChoiceValue::Function(_)
                )
            {
                return Err(RequestGrammarError::new(
                    "tool_choice",
                    "tool_choice requires a non-empty tools array",
                ));
            }
            return Ok(choice);
        }
    };

    let mut names = HashSet::with_capacity(tools.len());
    for (index, tool) in tools.iter().enumerate() {
        if tool.tool_type != "function" {
            return Err(RequestGrammarError::new(
                format!("tools[{index}].type"),
                "only type='function' tools are supported",
            ));
        }
        if !valid_tool_name(&tool.function.name) {
            return Err(RequestGrammarError::new(
                format!("tools[{index}].function.name"),
                "function name must be 1-64 ASCII letters, digits, underscores, or hyphens",
            ));
        }
        if !names.insert(tool.function.name.as_str()) {
            return Err(RequestGrammarError::new(
                format!("tools[{index}].function.name"),
                format!("duplicate function name {:?}", tool.function.name),
            ));
        }
        if let Some(parameters) = tool.function.parameters.as_ref() {
            if !parameters.is_object() && !parameters.is_boolean() {
                return Err(RequestGrammarError::new(
                    format!("tools[{index}].function.parameters"),
                    "function parameters must be an object or boolean JSON Schema",
                ));
            }
        }
    }

    if let ToolChoiceValue::Function(name) = &choice {
        if !valid_tool_name(name) {
            return Err(RequestGrammarError::new(
                "tool_choice.function.name",
                "function name must be 1-64 ASCII letters, digits, underscores, or hyphens",
            ));
        }
        if !names.contains(name.as_str()) {
            return Err(RequestGrammarError::new(
                "tool_choice.function.name",
                format!("named function {name:?} is not present in tools"),
            ));
        }
    }

    Ok(choice)
}

pub(super) fn first_lazy_param(request: &ChatCompletionRequest) -> Option<&'static str> {
    if request.grammar_lazy.is_some() {
        Some("grammar_lazy")
    } else if request.preserved_tokens.is_some() {
        Some("preserved_tokens")
    } else if request.grammar_triggers.is_some() {
        Some("grammar_triggers")
    } else {
        None
    }
}

pub(super) fn validate_lazy_fields(
    request: &ChatCompletionRequest,
    has_constraint: bool,
) -> Result<(), RequestGrammarError> {
    let Some(first_param) = first_lazy_param(request) else {
        return Ok(());
    };
    if !has_constraint {
        return Err(RequestGrammarError::new(
            first_param,
            "lazy grammar fields require an output constraint",
        ));
    }
    if request.grammar_lazy != Some(true) {
        let param = if request.grammar_triggers.is_some() {
            "grammar_triggers"
        } else if request.preserved_tokens.is_some() {
            "preserved_tokens"
        } else {
            return Ok(());
        };
        return Err(RequestGrammarError::new(
            param,
            "grammar_triggers and preserved_tokens require grammar_lazy=true",
        ));
    }

    let triggers = request
        .grammar_triggers
        .as_deref()
        .filter(|triggers| !triggers.is_empty())
        .ok_or_else(|| {
            RequestGrammarError::new(
                "grammar_triggers",
                "grammar_lazy=true requires at least one grammar trigger",
            )
        })?;
    for (index, trigger) in triggers.iter().enumerate() {
        if trigger.value.is_empty() {
            return Err(RequestGrammarError::new(
                format!("grammar_triggers[{index}].value"),
                "trigger value must not be empty",
            ));
        }
        if trigger.trigger_type == LlamaGrammarTriggerType::Token
            && trigger.token.is_some_and(|token| token < 0)
        {
            return Err(RequestGrammarError::new(
                format!("grammar_triggers[{index}].token"),
                "token id must be non-negative",
            ));
        }
    }

    if let Some(tokens) = request.preserved_tokens.as_deref() {
        if tokens.is_empty() {
            return Err(RequestGrammarError::new(
                "preserved_tokens",
                "preserved_tokens must contain at least one token string when specified",
            ));
        }
        let mut unique = HashSet::with_capacity(tokens.len());
        for (index, token) in tokens.iter().enumerate() {
            if token.is_empty() {
                return Err(RequestGrammarError::new(
                    format!("preserved_tokens[{index}]"),
                    "preserved token must not be empty",
                ));
            }
            if !unique.insert(token.as_str()) {
                return Err(RequestGrammarError::new(
                    format!("preserved_tokens[{index}]"),
                    format!("duplicate preserved token {token:?}"),
                ));
            }
        }
    }

    Ok(())
}
