//! Request-surface normalization for OpenAI, vLLM, and llama.cpp grammar
//! controls. Every accepted surface is lowered to the one hf2q GBNF runtime;
//! ambiguous or unsupported combinations fail before generation starts.

use serde_json::Value;

use super::{json_schema, lark, parser, regex_gbnf, structural_tag, Grammar};
use crate::serve::api::schema::{
    ChatCompletionRequest, ResponseFormat, StructuredOutputJson, StructuredOutputs, ToolChoiceValue,
};

#[path = "request_validation.rs"]
mod validation;
pub use validation::validate_tool_request;
use validation::{first_lazy_param, validate_lazy_fields};

const JSON_OBJECT_GRAMMAR: &str = r#"root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws
ws ::= | " " | "\n" [ \t]{0,20}
"#;

const MAX_RAW_CONSTRAINT_BYTES: usize = 1024 * 1024;
const MAX_CHOICES: usize = 1_024;
const MAX_CHOICE_LITERAL_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestGrammarError {
    pub param: String,
    pub message: String,
}

impl RequestGrammarError {
    fn new(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            param: param.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for RequestGrammarError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.param, self.message)
    }
}

impl std::error::Error for RequestGrammarError {}

fn parse_gbnf(source: &str, param: &'static str) -> Result<Grammar, RequestGrammarError> {
    parser::parse(source)
        .map_err(|error| RequestGrammarError::new(param, format!("GBNF parse failed: {error}")))
}

fn parse_vllm_grammar(source: &str, param: &'static str) -> Result<Grammar, RequestGrammarError> {
    if source.len() > MAX_RAW_CONSTRAINT_BYTES {
        return Err(RequestGrammarError::new(
            param,
            format!("grammar input exceeds the {MAX_RAW_CONSTRAINT_BYTES}-byte resource limit"),
        ));
    }
    let normalized = lark::normalize_for_gbnf(source)
        .map_err(|error| RequestGrammarError::new(param, error.to_string()))?;
    parse_gbnf(&normalized, param)
}

fn compile_structural_tag(
    payload: &Value,
    param: &'static str,
) -> Result<Grammar, RequestGrammarError> {
    let bytes = serde_json::to_vec(payload).map_err(|error| {
        RequestGrammarError::new(
            param,
            format!("structural tag serialization failed: {error}"),
        )
    })?;
    if bytes.len() > MAX_RAW_CONSTRAINT_BYTES {
        return Err(RequestGrammarError::new(
            param,
            format!(
                "serialized structural tag exceeds the {MAX_RAW_CONSTRAINT_BYTES}-byte resource limit"
            ),
        ));
    }
    structural_tag::compile(payload)
        .map_err(|error| RequestGrammarError::new(param, error.to_string()))
}

fn compile_schema(
    schema: &Value,
    param: &'static str,
    whitespace_pattern: Option<&str>,
) -> Result<Grammar, RequestGrammarError> {
    let bytes = serde_json::to_vec(schema).map_err(|error| {
        RequestGrammarError::new(param, format!("JSON Schema serialization failed: {error}"))
    })?;
    if bytes.len() > MAX_RAW_CONSTRAINT_BYTES {
        return Err(RequestGrammarError::new(
            param,
            format!(
                "serialized JSON Schema exceeds the {MAX_RAW_CONSTRAINT_BYTES}-byte resource limit"
            ),
        ));
    }
    let source = json_schema::schema_to_gbnf_with_whitespace(schema, whitespace_pattern).map_err(
        |error| RequestGrammarError::new(param, format!("JSON Schema compilation failed: {error}")),
    )?;
    parser::parse_generated(&source)
        .map_err(|error| RequestGrammarError::new(param, format!("GBNF parse failed: {error}")))
}

fn configured_whitespace(
    structured: &StructuredOutputs,
) -> Result<Option<&str>, RequestGrammarError> {
    if structured.disable_any_whitespace == Some(true) && structured.whitespace_pattern.is_some() {
        return Err(RequestGrammarError::new(
            "structured_outputs",
            "disable_any_whitespace and whitespace_pattern are mutually exclusive",
        ));
    }
    Ok(if structured.disable_any_whitespace == Some(true) {
        Some("")
    } else {
        structured.whitespace_pattern.as_deref()
    })
}

fn json_object_grammar(whitespace_pattern: Option<&str>) -> Result<String, RequestGrammarError> {
    let ws = match whitespace_pattern {
        None => return Ok(JSON_OBJECT_GRAMMAR.to_string()),
        Some("") => r#""""#.to_string(),
        Some(pattern) => regex_gbnf::regex_to_gbnf_full_match(
            pattern,
            regex_gbnf::Surface::RawOutput,
        )
        .map_err(|error| {
            RequestGrammarError::new("structured_outputs.whitespace_pattern", error.to_string())
        })?,
    };
    Ok(JSON_OBJECT_GRAMMAR.replace(
        r#"ws ::= | " " | "\n" [ \t]{0,20}"#,
        &format!("ws ::= {ws}"),
    ))
}

fn close_implicit_objects(schema: &mut Value) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };
    let object_schema = object.get("type").and_then(Value::as_str) == Some("object")
        || object.contains_key("properties")
        || object.contains_key("required");
    if object_schema && !object.contains_key("additionalProperties") {
        object.insert("additionalProperties".into(), Value::Bool(false));
    }
    for keyword in ["$defs", "definitions", "properties"] {
        if let Some(children) = object.get_mut(keyword).and_then(Value::as_object_mut) {
            for child in children.values_mut() {
                close_implicit_objects(child);
            }
        }
    }
    for keyword in ["items", "additionalProperties", "if", "then", "else"] {
        if let Some(child) = object.get_mut(keyword) {
            close_implicit_objects(child);
        }
    }
    for keyword in ["prefixItems", "anyOf", "oneOf", "allOf"] {
        if let Some(children) = object.get_mut(keyword).and_then(Value::as_array_mut) {
            for child in children {
                close_implicit_objects(child);
            }
        }
    }
}

fn structured_schema(value: &StructuredOutputJson) -> Result<Value, RequestGrammarError> {
    match value {
        StructuredOutputJson::Object(object) => Ok(Value::Object(object.clone())),
        StructuredOutputJson::String(source) => serde_json::from_str(source).map_err(|error| {
            RequestGrammarError::new(
                "structured_outputs.json",
                format!("serialized JSON Schema is invalid: {error}"),
            )
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstraintKind {
    Choice,
    Regex,
    Json,
    JsonObject,
    Grammar,
    StructuralTag,
}

fn structured_constraint_kind(
    structured: &StructuredOutputs,
) -> Result<ConstraintKind, RequestGrammarError> {
    structured
        .validate_exactly_one_constraint()
        .map_err(|error| RequestGrammarError::new("structured_outputs", error.to_string()))?;
    Ok(if structured.choice.is_some() {
        ConstraintKind::Choice
    } else if structured.regex.is_some() {
        ConstraintKind::Regex
    } else if structured.json.is_some() {
        ConstraintKind::Json
    } else if structured.json_object.is_some() {
        ConstraintKind::JsonObject
    } else if structured.grammar.is_some() {
        ConstraintKind::Grammar
    } else {
        ConstraintKind::StructuralTag
    })
}

fn response_constraint_kind(response: &ResponseFormat) -> Option<ConstraintKind> {
    match response {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(ConstraintKind::JsonObject),
        ResponseFormat::JsonSchema { .. } => Some(ConstraintKind::Json),
        ResponseFormat::StructuralTag { .. } => Some(ConstraintKind::StructuralTag),
    }
}

pub(crate) fn compile_response_format(
    response: &ResponseFormat,
    structured_options: Option<&StructuredOutputs>,
) -> Result<Option<Grammar>, RequestGrammarError> {
    let whitespace_pattern = structured_options
        .map(configured_whitespace)
        .transpose()?
        .flatten();
    match response {
        ResponseFormat::Text => Ok(None),
        ResponseFormat::JsonObject => {
            if structured_options
                .is_some_and(|options| options.disable_additional_properties.is_some())
            {
                return Err(RequestGrammarError::new(
                    "structured_outputs.disable_additional_properties",
                    "disable_additional_properties requires an explicit JSON Schema",
                ));
            }
            parse_gbnf(&json_object_grammar(whitespace_pattern)?, "response_format").map(Some)
        }
        ResponseFormat::JsonSchema { json_schema: spec } => {
            let mut schema = spec.schema.clone();
            if structured_options
                .is_some_and(|options| options.disable_additional_properties == Some(true))
            {
                close_implicit_objects(&mut schema);
            }
            compile_schema(&schema, "response_format", whitespace_pattern).map(Some)
        }
        ResponseFormat::StructuralTag { spec } => {
            if structured_options.is_some_and(|options| {
                options.disable_any_whitespace.is_some()
                    || options.disable_additional_properties.is_some()
                    || options.whitespace_pattern.is_some()
            }) {
                return Err(RequestGrammarError::new(
                    "structured_outputs",
                    "JSON backend options cannot modify a structural_tag constraint",
                ));
            }
            let mut payload = spec.clone();
            payload.insert("type".into(), Value::String("structural_tag".into()));
            compile_structural_tag(&Value::Object(payload), "response_format").map(Some)
        }
    }
}

pub fn compile_structured_outputs(
    structured: &StructuredOutputs,
) -> Result<Grammar, RequestGrammarError> {
    structured
        .validate_exactly_one_constraint()
        .map_err(|error| RequestGrammarError::new("structured_outputs", error.to_string()))?;
    if structured.json.is_none()
        && structured.json_object.is_none()
        && (structured.disable_any_whitespace.is_some() || structured.whitespace_pattern.is_some())
    {
        return Err(RequestGrammarError::new(
            "structured_outputs",
            "disable_any_whitespace and whitespace_pattern apply only to JSON constraints",
        ));
    }
    let whitespace_pattern = configured_whitespace(structured)?;

    if let Some(choices) = structured.choice.as_ref() {
        if choices.is_empty() {
            return Err(RequestGrammarError::new(
                "structured_outputs.choice",
                "choice must contain at least one string",
            ));
        }
        if choices.len() > MAX_CHOICES {
            return Err(RequestGrammarError::new(
                "structured_outputs.choice",
                format!("choice exceeds the {MAX_CHOICES}-entry resource limit"),
            ));
        }
        let literal_bytes = choices.iter().try_fold(0usize, |total, choice| {
            total.checked_add(choice.len()).ok_or_else(|| {
                RequestGrammarError::new(
                    "structured_outputs.choice",
                    "choice literal byte count overflowed the resource counter",
                )
            })
        })?;
        if literal_bytes > MAX_CHOICE_LITERAL_BYTES {
            return Err(RequestGrammarError::new(
                "structured_outputs.choice",
                format!(
                    "choice literals exceed the {MAX_CHOICE_LITERAL_BYTES}-byte aggregate resource limit"
                ),
            ));
        }
        if structured.disable_additional_properties.is_some() {
            return Err(RequestGrammarError::new(
                "structured_outputs.disable_additional_properties",
                "disable_additional_properties applies only to JSON Schema constraints",
            ));
        }
        let alternatives = choices
            .iter()
            .map(|choice| json_schema::format_literal(choice))
            .collect::<Vec<_>>()
            .join(" | ");
        return parse_gbnf(
            &format!("root ::= ( {alternatives} )\n"),
            "structured_outputs.choice",
        );
    }
    if let Some(expression) = structured.regex.as_ref() {
        if expression.contains('\0') {
            return Err(RequestGrammarError::new(
                "structured_outputs.regex",
                "regex must not contain NUL",
            ));
        }
        if expression.len() > MAX_RAW_CONSTRAINT_BYTES {
            return Err(RequestGrammarError::new(
                "structured_outputs.regex",
                format!("regex exceeds the {MAX_RAW_CONSTRAINT_BYTES}-byte resource limit"),
            ));
        }
        if structured.disable_additional_properties.is_some() {
            return Err(RequestGrammarError::new(
                "structured_outputs.disable_additional_properties",
                "disable_additional_properties applies only to JSON Schema constraints",
            ));
        }
        let body = regex_gbnf::regex_to_gbnf_full_match(expression, regex_gbnf::Surface::RawOutput)
            .map_err(|error| {
                RequestGrammarError::new("structured_outputs.regex", error.to_string())
            })?;
        return parse_gbnf(&format!("root ::= {body}\n"), "structured_outputs.regex");
    }
    if let Some(schema) = structured.json.as_ref() {
        let mut schema = structured_schema(schema)?;
        if structured.disable_additional_properties == Some(true) {
            close_implicit_objects(&mut schema);
        }
        return compile_schema(&schema, "structured_outputs.json", whitespace_pattern);
    }
    if structured.json_object == Some(true) {
        if structured.disable_additional_properties.is_some() {
            return Err(RequestGrammarError::new(
                "structured_outputs.disable_additional_properties",
                "disable_additional_properties requires an explicit JSON Schema",
            ));
        }
        return parse_gbnf(
            &json_object_grammar(whitespace_pattern)?,
            "structured_outputs.json_object",
        );
    }
    if let Some(source) = structured.grammar.as_ref() {
        if source.trim().is_empty() {
            return Err(RequestGrammarError::new(
                "structured_outputs.grammar",
                "grammar must not be empty",
            ));
        }
        if structured.disable_additional_properties.is_some() {
            return Err(RequestGrammarError::new(
                "structured_outputs.disable_additional_properties",
                "disable_additional_properties applies only to JSON Schema constraints",
            ));
        }
        return parse_vllm_grammar(source, "structured_outputs.grammar");
    }
    if structured.disable_additional_properties.is_some() {
        return Err(RequestGrammarError::new(
            "structured_outputs.disable_additional_properties",
            "disable_additional_properties applies only to JSON Schema constraints",
        ));
    }
    let source = structured
        .structural_tag
        .as_deref()
        .expect("validated structural_tag constraint");
    if source.len() > MAX_RAW_CONSTRAINT_BYTES {
        return Err(RequestGrammarError::new(
            "structured_outputs.structural_tag",
            format!(
                "serialized structural tag exceeds the {MAX_RAW_CONSTRAINT_BYTES}-byte resource limit"
            ),
        ));
    }
    let payload = serde_json::from_str(source).map_err(|error| {
        RequestGrammarError::new(
            "structured_outputs.structural_tag",
            format!("serialized structural tag is invalid JSON: {error}"),
        )
    })?;
    compile_structural_tag(&payload, "structured_outputs.structural_tag")
}

/// Resolve all non-tool request surfaces to one grammar. A non-text OpenAI
/// response format replaces the same vLLM structured-output slot while
/// retaining backend options; selecting a different slot is a conflict.
/// `text` contributes no constraint and preserves another explicit surface.
pub fn compile_request_constraint(
    request: &ChatCompletionRequest,
) -> Result<Option<Grammar>, RequestGrammarError> {
    let structured_kind = request
        .structured_outputs
        .as_ref()
        .map(structured_constraint_kind)
        .transpose()?;
    let response_kind = request
        .response_format
        .as_ref()
        .and_then(response_constraint_kind);

    if request.grammar.is_some() && request.json_schema.is_some() {
        return Err(RequestGrammarError::new(
            "grammar",
            "grammar and json_schema are mutually exclusive",
        ));
    }
    if response_kind.is_some() && (request.grammar.is_some() || request.json_schema.is_some()) {
        return Err(RequestGrammarError::new(
            "response_format",
            "response_format cannot be combined with top-level grammar or json_schema",
        ));
    }
    if structured_kind.is_some() && (request.grammar.is_some() || request.json_schema.is_some()) {
        return Err(RequestGrammarError::new(
            "structured_outputs",
            "structured_outputs cannot be combined with top-level grammar or json_schema",
        ));
    }
    if let (Some(response), Some(structured)) = (response_kind, structured_kind) {
        if response != structured {
            return Err(RequestGrammarError::new(
                "response_format",
                "response_format and structured_outputs select a different constraint type",
            ));
        }
    }

    let has_constraint = response_kind.is_some()
        || structured_kind.is_some()
        || request.grammar.is_some()
        || request.json_schema.is_some();
    validate_lazy_fields(request, has_constraint)?;

    if let Some(response) = request.response_format.as_ref() {
        if response_kind.is_some() {
            return compile_response_format(response, request.structured_outputs.as_ref());
        }
    }
    if let Some(structured) = request.structured_outputs.as_ref() {
        return compile_structured_outputs(structured).map(Some);
    }
    if let Some(schema) = request.json_schema.as_ref() {
        return compile_schema(&schema.as_value(), "json_schema", None).map(Some);
    }
    if let Some(source) = request.grammar.as_ref() {
        if source.trim().is_empty() {
            return Err(RequestGrammarError::new(
                "grammar",
                "grammar must not be empty",
            ));
        }
        return parse_gbnf(source, "grammar").map(Some);
    }
    Ok(None)
}

/// Apply native tool-call precedence before compiling an output constraint.
/// Required and named choices use the model-family tool grammar, so an
/// otherwise-unused response constraint is deliberately not compiled.
pub fn compile_request_output_constraint(
    request: &ChatCompletionRequest,
    tool_choice: &ToolChoiceValue,
) -> Result<Option<Grammar>, RequestGrammarError> {
    if matches!(
        tool_choice,
        ToolChoiceValue::Required | ToolChoiceValue::Function(_)
    ) {
        if let Some(param) = first_lazy_param(request) {
            return Err(RequestGrammarError::new(
                param,
                "llama.cpp lazy grammar fields cannot modify a required native tool grammar",
            ));
        }
        return Ok(None);
    }
    compile_request_constraint(request)
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;
