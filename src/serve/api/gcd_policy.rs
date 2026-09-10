//! ADR-057: resolve trusted server policy before request constraints can change it.

use super::schema::{ApiError, ChatCompletionRequest, ToolChoiceValue};
use super::state::ServerConfig;

fn explicit_constraint_surface(request: &ChatCompletionRequest) -> Option<&'static str> {
    if request.grammar.is_some() {
        Some("grammar")
    } else if request.response_format.is_some() {
        Some("response_format")
    } else if request.structured_outputs.is_some() {
        Some("structured_outputs")
    } else if request.json_schema.is_some() {
        Some("json_schema")
    } else {
        None
    }
}

fn locked_request_error(surface: &str, reason: &str) -> ApiError {
    ApiError::invalid_request(
        format!("--gcd-schema-locked: {reason}; request field `{surface}` is not allowed"),
        Some(surface.into()),
    )
}

/// Check the caller's unmodified request, then inject the trusted default.
/// Keeping both operations here prevents an injected schema from being
/// mistaken for a caller override. Rejections leave the request unchanged.
///
/// Locked mode owns response output. Tool definitions require `tool_choice:
/// "none"`; required/named tool choices cannot replace the schema with a
/// native tool grammar. Lazy-grammar modifiers cannot defer policy activation.
/// GLP is deliberately absent from this decision: it never selects a grammar.
pub(super) fn apply_gcd_policy(
    config: &ServerConfig,
    request: &mut ChatCompletionRequest,
) -> Result<(), ApiError> {
    if config.gcd_schema_locked {
        // The CLI prevents this configuration, but internal callers must also
        // fail closed rather than silently operate without a locked schema.
        if config.gcd || config.gcd_schema_grammar.is_none() {
            tracing::error!("locked GCD policy requires a schema and excludes the prose default");
            return Err(ApiError::internal_error());
        }
        if let Some(surface) = explicit_constraint_surface(request) {
            return Err(locked_request_error(
                surface,
                "the server schema is mandatory and cannot be overridden",
            ));
        }
        let modifier = if request.grammar_lazy.is_some() {
            Some("grammar_lazy")
        } else if request.preserved_tokens.is_some() {
            Some("preserved_tokens")
        } else if request.grammar_triggers.is_some() {
            Some("grammar_triggers")
        } else {
            None
        };
        if let Some(surface) = modifier {
            return Err(locked_request_error(
                surface,
                "the server schema must remain active from the first generated token",
            ));
        }
        // Preserve the normal tool contract before applying the additional
        // locked-output restrictions or injecting the trusted schema.
        let tool_choice = super::grammar::request::validate_tool_request(request)
            .map_err(|error| ApiError::invalid_request(error.message, Some(error.param)))?;
        match tool_choice {
            ToolChoiceValue::Required | ToolChoiceValue::Function(_) => {
                return Err(locked_request_error(
                    "tool_choice",
                    "tool calls cannot replace the server response schema; use tool_choice='none'",
                ));
            }
            ToolChoiceValue::Auto
                if request
                    .tools
                    .as_ref()
                    .is_some_and(|tools| !tools.is_empty()) =>
            {
                return Err(locked_request_error(
                    "tools",
                    "tool definitions require tool_choice='none' with a locked response schema",
                ));
            }
            ToolChoiceValue::Auto | ToolChoiceValue::None => {}
        }
    }

    // Unlocked defaults defer to every supported explicit output surface.
    if explicit_constraint_surface(request).is_none() {
        if config.gcd {
            request.grammar = Some(include_str!("grammar/gcd_w1.gbnf").to_owned());
        } else if let Some(grammar) = &config.gcd_schema_grammar {
            request.grammar = Some(grammar.clone());
        }
        if request.grammar.is_some() {
            // The prose anchor or schema must engage at token zero; template
            // thinking output before it would be outside the language.
            request.hf2q_enable_thinking = Some(false);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "gcd_policy_tests.rs"]
mod tests;
