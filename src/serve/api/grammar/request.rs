//! Request-surface normalization for OpenAI, vLLM, and llama.cpp grammar
//! controls. Every accepted surface is lowered to the one hf2q GBNF runtime;
//! ambiguous or unsupported combinations fail before generation starts.

use serde_json::Value;

use super::{json_schema, lark, parser, regex_gbnf, structural_tag, Grammar, LazyGrammarConfig};
use crate::serve::api::schema::{
    ChatCompletionRequest, LlamaGrammarTriggerType, ResponseFormat, StopSequence,
    StructuredOutputJson, StructuredOutputs, ToolChoiceValue,
};

#[path = "request_validation.rs"]
mod validation;
pub use validation::validate_tool_request;
use validation::{first_lazy_param, validate_lazy_fields};

const MAX_LAZY_TRIGGER_COUNT: usize = 1024;
const MAX_LAZY_PATTERN_BYTES: usize = 1024 * 1024;

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
    deferred_to_tokenizer: bool,
}

impl RequestGrammarError {
    fn new(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            param: param.into(),
            message: message.into(),
            deferred_to_tokenizer: false,
        }
    }

    fn deferred_to_tokenizer(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            param: param.into(),
            message: message.into(),
            deferred_to_tokenizer: true,
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

fn parse_gbnf_bound(
    source: &str,
    param: &'static str,
    tokenizer: Option<&tokenizers::Tokenizer>,
) -> Result<Grammar, RequestGrammarError> {
    match tokenizer {
        Some(tokenizer) => parser::parse_with_tokenizer(source, tokenizer).map_err(|error| {
            RequestGrammarError::new(param, format!("GBNF parse failed: {error}"))
        }),
        None => parser::parse(source).map_err(|error| {
            let message = format!("GBNF parse failed: {error}");
            if error.requires_tokenizer() {
                RequestGrammarError::deferred_to_tokenizer(param, message)
            } else {
                RequestGrammarError::new(param, message)
            }
        }),
    }
}

fn parse_vllm_grammar(
    source: &str,
    param: &'static str,
    tokenizer: Option<&tokenizers::Tokenizer>,
) -> Result<Grammar, RequestGrammarError> {
    if source.len() > MAX_RAW_CONSTRAINT_BYTES {
        return Err(RequestGrammarError::new(
            param,
            format!("grammar input exceeds the {MAX_RAW_CONSTRAINT_BYTES}-byte resource limit"),
        ));
    }
    let normalized = lark::normalize_for_gbnf(source)
        .map_err(|error| RequestGrammarError::new(param, error.to_string()))?;
    parse_gbnf_bound(&normalized, param, tokenizer)
}

fn compile_structural_tag(
    payload: &Value,
    param: &'static str,
    tokenizer: Option<&tokenizers::Tokenizer>,
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
    let grammar = match tokenizer {
        Some(tokenizer) => structural_tag::compile_with_token_resolver(payload, |token| {
            resolve_single_token(token, param, tokenizer).map_err(|error| error.message)
        }),
        None => structural_tag::compile(payload),
    }
    .map_err(|error| match error {
        structural_tag::StructuralTagError::NeedsTokenVocabulary(_) if tokenizer.is_none() => {
            RequestGrammarError::deferred_to_tokenizer(param, error.to_string())
        }
        _ => RequestGrammarError::new(param, error.to_string()),
    })?;
    if let Some(tokenizer) = tokenizer {
        parser::validate_token_ids(&grammar, tokenizer).map_err(|error| {
            RequestGrammarError::new(param, format!("token binding failed: {error}"))
        })?;
    }
    Ok(grammar)
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
        ResponseFormat::JsonObject { schema: Some(_) } => Some(ConstraintKind::Json),
        ResponseFormat::JsonObject { schema: None } => Some(ConstraintKind::JsonObject),
        ResponseFormat::JsonSchema { .. } => Some(ConstraintKind::Json),
        ResponseFormat::StructuralTag { .. } => Some(ConstraintKind::StructuralTag),
    }
}

fn compile_response_format_impl(
    response: &ResponseFormat,
    structured_options: Option<&StructuredOutputs>,
    tokenizer: Option<&tokenizers::Tokenizer>,
) -> Result<Option<Grammar>, RequestGrammarError> {
    let whitespace_pattern = structured_options
        .map(configured_whitespace)
        .transpose()?
        .flatten();
    match response {
        ResponseFormat::Text => Ok(None),
        ResponseFormat::JsonObject { schema } => {
            if let Some(schema) = schema {
                let mut schema = schema.as_value();
                if structured_options
                    .is_some_and(|options| options.disable_additional_properties == Some(true))
                {
                    close_implicit_objects(&mut schema);
                }
                compile_schema(&schema, "response_format.schema", whitespace_pattern).map(Some)
            } else {
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
            compile_structural_tag(&Value::Object(payload), "response_format", tokenizer).map(Some)
        }
    }
}

pub(crate) fn compile_response_format(
    response: &ResponseFormat,
    structured_options: Option<&StructuredOutputs>,
) -> Result<Option<Grammar>, RequestGrammarError> {
    compile_response_format_impl(response, structured_options, None)
}

fn compile_structured_outputs_impl(
    structured: &StructuredOutputs,
    tokenizer: Option<&tokenizers::Tokenizer>,
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
        return parse_vllm_grammar(source, "structured_outputs.grammar", tokenizer);
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
    compile_structural_tag(&payload, "structured_outputs.structural_tag", tokenizer)
}

pub fn compile_structured_outputs(
    structured: &StructuredOutputs,
) -> Result<Grammar, RequestGrammarError> {
    compile_structured_outputs_impl(structured, None)
}

fn resolve_single_token(
    value: &str,
    param: &'static str,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<u32, RequestGrammarError> {
    let encoding = tokenizer.encode(value, false).map_err(|error| {
        RequestGrammarError::new(param, format!("tokenizer rejected value: {error}"))
    })?;
    let [token_id] = encoding.get_ids() else {
        return Err(RequestGrammarError::new(
            param,
            format!(
                "value must resolve to exactly one token id, resolved {}",
                encoding.get_ids().len()
            ),
        ));
    };
    if tokenizer.id_to_token(*token_id).is_none() {
        return Err(RequestGrammarError::new(
            param,
            format!("resolved token id {token_id} is outside the model vocabulary"),
        ));
    }
    Ok(*token_id)
}

fn compile_lazy_config(
    request: &ChatCompletionRequest,
    grammar_present: bool,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Option<LazyGrammarConfig>, RequestGrammarError> {
    if request.grammar_lazy != Some(true) {
        if request.preserved_tokens.is_some() || request.grammar_triggers.is_some() {
            return Err(RequestGrammarError::new(
                "grammar_lazy",
                "preserved_tokens and grammar_triggers require grammar_lazy=true",
            ));
        }
        return Ok(None);
    }
    if !grammar_present {
        return Err(RequestGrammarError::new(
            "grammar_lazy",
            "grammar_lazy=true requires a grammar constraint",
        ));
    }

    let triggers = request.grammar_triggers.as_deref().ok_or_else(|| {
        RequestGrammarError::new(
            "grammar_triggers",
            "grammar_lazy=true requires at least one grammar trigger",
        )
    })?;
    if triggers.is_empty() {
        return Err(RequestGrammarError::new(
            "grammar_triggers",
            "grammar_lazy=true requires at least one grammar trigger",
        ));
    }
    if triggers.len() > MAX_LAZY_TRIGGER_COUNT {
        return Err(RequestGrammarError::new(
            "grammar_triggers",
            format!("at most {MAX_LAZY_TRIGGER_COUNT} grammar triggers are allowed"),
        ));
    }

    let preserved = request.preserved_tokens.as_deref().unwrap_or_default();
    if preserved.len() > MAX_LAZY_TRIGGER_COUNT {
        return Err(RequestGrammarError::new(
            "preserved_tokens",
            format!("at most {MAX_LAZY_TRIGGER_COUNT} preserved tokens are allowed"),
        ));
    }
    let mut preserved_tokens = preserved
        .iter()
        .map(|value| resolve_single_token(value, "preserved_tokens", tokenizer))
        .collect::<Result<Vec<_>, _>>()?;
    preserved_tokens.sort_unstable();
    preserved_tokens.dedup();

    let mut token_triggers = Vec::new();
    let mut trigger_patterns = Vec::new();
    let mut pattern_bytes = 0usize;
    for trigger in triggers {
        match trigger.trigger_type {
            LlamaGrammarTriggerType::Token => {
                let signed = trigger.token.expect("schema validates token trigger");
                let token_id = u32::try_from(signed).map_err(|_| {
                    RequestGrammarError::new(
                        "grammar_triggers",
                        format!("token trigger id {signed} must be non-negative"),
                    )
                })?;
                if tokenizer.id_to_token(token_id).is_none() {
                    return Err(RequestGrammarError::new(
                        "grammar_triggers",
                        format!("token trigger id {token_id} is outside the model vocabulary"),
                    ));
                }
                token_triggers.push(token_id);
            }
            LlamaGrammarTriggerType::Word => {
                let encoding =
                    tokenizer
                        .encode(trigger.value.as_str(), false)
                        .map_err(|error| {
                            RequestGrammarError::new(
                                "grammar_triggers",
                                format!("tokenizer rejected word trigger: {error}"),
                            )
                        })?;
                if let [token_id] = encoding.get_ids() {
                    if !preserved_tokens.contains(token_id) {
                        return Err(RequestGrammarError::new(
                            "grammar_triggers",
                            format!(
                                "single-token word trigger {:?} must also appear in preserved_tokens",
                                trigger.value
                            ),
                        ));
                    }
                    token_triggers.push(*token_id);
                } else {
                    trigger_patterns.push(regex::escape(&trigger.value));
                }
            }
            LlamaGrammarTriggerType::Pattern => {
                trigger_patterns.push(trigger.value.clone());
            }
            LlamaGrammarTriggerType::PatternFull => {
                let pattern = if trigger.value.is_empty() {
                    "^$".to_string()
                } else {
                    format!(
                        "{}{}{}",
                        if trigger.value.starts_with('^') {
                            ""
                        } else {
                            "^"
                        },
                        trigger.value,
                        if trigger.value.ends_with('$') {
                            ""
                        } else {
                            "$"
                        }
                    )
                };
                trigger_patterns.push(pattern);
            }
        }
    }

    token_triggers.sort_unstable();
    token_triggers.dedup();
    for pattern in &trigger_patterns {
        pattern_bytes = pattern_bytes.checked_add(pattern.len()).ok_or_else(|| {
            RequestGrammarError::new("grammar_triggers", "grammar trigger patterns are too large")
        })?;
        if pattern_bytes > MAX_LAZY_PATTERN_BYTES {
            return Err(RequestGrammarError::new(
                "grammar_triggers",
                format!("grammar trigger patterns exceed the {MAX_LAZY_PATTERN_BYTES}-byte limit"),
            ));
        }
        regex::bytes::Regex::new(pattern).map_err(|error| {
            RequestGrammarError::new(
                "grammar_triggers",
                format!("unsupported grammar trigger pattern {pattern:?}: {error}"),
            )
        })?;
    }

    Ok(Some(LazyGrammarConfig {
        token_triggers,
        trigger_patterns,
        preserved_tokens,
    }))
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledRequestConstraint {
    pub grammar: Option<Grammar>,
    pub lazy: Option<LazyGrammarConfig>,
}

/// Resolve all non-tool request surfaces to one grammar. vLLM's documented
/// response-format conversion wins over `structured_outputs`, except `text`,
/// which preserves the explicit structured-output request.
fn compile_request_constraint_impl(
    request: &ChatCompletionRequest,
    tokenizer: Option<&tokenizers::Tokenizer>,
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
            return compile_response_format_impl(
                response,
                request.structured_outputs.as_ref(),
                tokenizer,
            );
        }
    }
    if let Some(structured) = request.structured_outputs.as_ref() {
        return compile_structured_outputs_impl(structured, tokenizer).map(Some);
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
        return parse_gbnf_bound(source, "grammar", tokenizer).map(Some);
    }
    Ok(None)
}

/// Compile a request before a model/tokenizer has been selected. Lazy fields
/// are rejected on this boundary because accepting them without authoritative
/// token resolution would silently change their meaning.
pub fn compile_request_constraint(
    request: &ChatCompletionRequest,
) -> Result<Option<Grammar>, RequestGrammarError> {
    let grammar = compile_request_constraint_impl(request, None)?;
    if first_lazy_param(request).is_some() {
        return Err(RequestGrammarError::new(
            "grammar_lazy",
            "lazy grammar fields require the tokenizer-bound compiler",
        ));
    }
    Ok(grammar)
}

/// Compile every public structured-output surface after model resolution.
/// Raw GBNF token terminals and lazy trigger strings are resolved against the
/// selected model's authoritative tokenizer, including numeric OOV checks.
pub fn compile_request_constraint_with_tokenizer(
    request: &ChatCompletionRequest,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<CompiledRequestConstraint, RequestGrammarError> {
    let grammar = compile_request_constraint_impl(request, Some(tokenizer))?;
    let lazy = compile_lazy_config(request, grammar.is_some(), tokenizer)?;
    Ok(CompiledRequestConstraint { grammar, lazy })
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

/// Validate model-independent structured-output semantics before resolving or
/// loading a model. Textual token terminals are the sole deferred condition:
/// their syntax and vocabulary membership are proved later by the
/// tokenizer-bound compiler. This keeps malformed schemas, conflicting
/// surfaces, and invalid lazy controls attributable to their request fields
/// instead of letting model lookup mask them.
pub fn validate_request_output_constraint_before_model(
    request: &ChatCompletionRequest,
    tool_choice: &ToolChoiceValue,
) -> Result<(), RequestGrammarError> {
    let result = if matches!(
        tool_choice,
        ToolChoiceValue::Required | ToolChoiceValue::Function(_)
    ) {
        compile_request_output_constraint(request, tool_choice).map(|_| ())
    } else {
        compile_request_constraint_impl(request, None).map(|_| ())
    };
    match result {
        Err(error) if error.deferred_to_tokenizer => Ok(()),
        other => other,
    }
}

/// Tokenizer-bound counterpart of [`compile_request_output_constraint`].
/// This preserves native tool precedence while resolving public token
/// terminals and lazy triggers against the selected model.
pub fn compile_request_output_constraint_with_tokenizer(
    request: &ChatCompletionRequest,
    tool_choice: &ToolChoiceValue,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<CompiledRequestConstraint, RequestGrammarError> {
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
        return Ok(CompiledRequestConstraint {
            grammar: None,
            lazy: None,
        });
    }
    compile_request_constraint_with_tokenizer(request, tokenizer)
}

/// Stop-string stripping can remove bytes that were part of the accepted
/// grammar language. Until every unary/SSE family buffers and validates the
/// stripped suffix identically, reject the ambiguous combination explicitly.
pub fn validate_stop_with_constraint(
    request: &ChatCompletionRequest,
    constraint_present: bool,
) -> Result<(), RequestGrammarError> {
    let has_stop = match request.stop.as_ref() {
        None => false,
        Some(StopSequence::Single(_)) => true,
        Some(StopSequence::Multiple(values)) => !values.is_empty(),
    };
    if constraint_present && has_stop {
        return Err(RequestGrammarError::new(
            "stop",
            "stop cannot be combined with a grammar-constrained response because stripping the stop sequence could invalidate the constrained output",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;
    use tokenizers::{models::bpe::BPE, AddedToken, Tokenizer};

    fn accepts(grammar: &Grammar, bytes: &[u8]) -> bool {
        let root = grammar.rule_id("root").expect("root");
        let mut runtime =
            super::super::GrammarRuntime::new(grammar.clone(), root).expect("runtime");
        runtime.accept_bytes(bytes) && runtime.is_accepted()
    }

    fn tokenizer_with_specials(values: &[&str]) -> Tokenizer {
        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.add_special_tokens(
            &values
                .iter()
                .map(|value| AddedToken::from((*value).to_string(), true))
                .collect::<Vec<_>>(),
        );
        tokenizer
    }

    fn request(fields: serde_json::Value) -> ChatCompletionRequest {
        let mut value = serde_json::json!({"model":"fixture", "messages":[]});
        value
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn vllm_choice_regex_json_and_raw_grammar_compile_and_enforce() {
        let choice = StructuredOutputs {
            choice: Some(vec!["allow".into(), "deny".into()]),
            ..Default::default()
        };
        let grammar = compile_structured_outputs(&choice).unwrap();
        assert!(accepts(&grammar, b"allow"));
        assert!(!accepts(&grammar, b"allowed"));

        let regex = StructuredOutputs {
            regex: Some("[A-Z]{2}[0-9]{2}".into()),
            ..Default::default()
        };
        let grammar = compile_structured_outputs(&regex).unwrap();
        assert!(accepts(&grammar, b"AB12"));
        assert!(!accepts(&grammar, b"xAB12y"));

        let json = StructuredOutputs {
            json: Some(StructuredOutputJson::String(
                r#"{"type":"string","enum":["ok"]}"#.into(),
            )),
            ..Default::default()
        };
        let grammar = compile_structured_outputs(&json).unwrap();
        assert!(accepts(&grammar, br#""ok""#));
        assert!(!accepts(&grammar, br#""no""#));

        let raw = StructuredOutputs {
            grammar: Some("root ::= \"yes\" | \"no\"\n".into()),
            ..Default::default()
        };
        let grammar = compile_structured_outputs(&raw).unwrap();
        assert!(accepts(&grammar, b"yes"));
        assert!(!accepts(&grammar, b"maybe"));
    }

    #[test]
    fn vllm_invalid_and_ambiguous_constraints_fail_closed() {
        for structured in [
            StructuredOutputs::default(),
            StructuredOutputs {
                choice: Some(Vec::new()),
                ..Default::default()
            },
            StructuredOutputs {
                grammar: Some("  ".into()),
                ..Default::default()
            },
            StructuredOutputs {
                regex: Some("x\0y".into()),
                ..Default::default()
            },
            StructuredOutputs {
                choice: Some(vec!["x".into()]),
                regex: Some("x".into()),
                ..Default::default()
            },
        ] {
            assert!(compile_structured_outputs(&structured).is_err());
        }
    }

    #[test]
    fn disable_additional_properties_closes_implicit_nested_objects() {
        let structured = StructuredOutputs {
            json: Some(StructuredOutputJson::Object(
                serde_json::json!({
                    "type":"object",
                    "properties":{"nested":{"type":"object","properties":{"x":{"type":"integer"}}}}
                })
                .as_object()
                .unwrap()
                .clone(),
            )),
            disable_additional_properties: Some(true),
            ..Default::default()
        };
        let grammar = compile_structured_outputs(&structured).unwrap();
        assert!(accepts(&grammar, br#"{"nested":{"x":1}}"#));
        assert!(!accepts(&grammar, br#"{"nested":{"x":1,"y":2}}"#));
    }

    #[test]
    fn whitespace_options_are_enforced_for_json_constraints() {
        let compact = StructuredOutputs {
            json: Some(StructuredOutputJson::Object(
                serde_json::json!({
                    "type":"object",
                    "properties":{"x":{"type":"integer"}},
                    "required":["x"],
                    "additionalProperties":false
                })
                .as_object()
                .unwrap()
                .clone(),
            )),
            disable_any_whitespace: Some(true),
            ..Default::default()
        };
        let grammar = compile_structured_outputs(&compact).unwrap();
        assert!(accepts(&grammar, br#"{"x":1}"#));
        assert!(!accepts(&grammar, br#"{ "x": 1 }"#));

        let custom = StructuredOutputs {
            json_object: Some(true),
            whitespace_pattern: Some("[ ]*".into()),
            ..Default::default()
        };
        let grammar = compile_structured_outputs(&custom).unwrap();
        assert!(accepts(&grammar, br#"{ "x": 1 }"#));
        assert!(!accepts(&grammar, b"{\n\"x\":1}"));
    }

    #[test]
    fn tokenizer_bound_raw_grammar_resolves_text_and_rejects_every_oov_form() {
        let tokenizer = tokenizer_with_specials(&["<tok>"]);
        let token_id = tokenizer.token_to_id("<tok>").unwrap();
        let compiled = compile_request_constraint_with_tokenizer(
            &request(serde_json::json!({"grammar":"root ::= <tok>\n"})),
            &tokenizer,
        )
        .unwrap();
        assert_eq!(compiled.grammar.unwrap().rules[0][0].value, token_id);

        for source in [
            "root ::= <[999]>\n",
            "root ::= !<[999]>\n",
            "root ::= !<[0,999]>\n",
        ] {
            let error = compile_request_constraint_with_tokenizer(
                &request(serde_json::json!({"grammar":source})),
                &tokenizer,
            )
            .expect_err("numeric token ids outside the selected vocab must fail");
            assert_eq!(error.param, "grammar");
            assert!(error.message.contains("not present"), "{error}");
        }
    }

    #[test]
    fn lazy_fields_compile_all_llama_trigger_types_after_tokenizer_binding() {
        let tokenizer = tokenizer_with_specials(&["<tok>", "<a>", "<b>"]);
        let token_id = tokenizer.token_to_id("<tok>").unwrap();
        let compiled = compile_request_constraint_with_tokenizer(
            &request(serde_json::json!({
                "grammar":"root ::= \"BODY\"\n",
                "grammar_lazy":true,
                "preserved_tokens":["<tok>"],
                "grammar_triggers":[
                    {"type":0,"value":"ignored-by-token-kind","token":token_id},
                    {"type":1,"value":"<tok>"},
                    {"type":1,"value":"<a><b>"},
                    {"type":2,"value":"tool:(BODY)"},
                    {"type":3,"value":"whole"}
                ]
            })),
            &tokenizer,
        )
        .unwrap();

        let lazy = compiled.lazy.unwrap();
        assert_eq!(lazy.preserved_tokens, vec![token_id]);
        assert_eq!(lazy.token_triggers, vec![token_id]);
        assert_eq!(
            lazy.trigger_patterns,
            vec![
                regex::escape("<a><b>"),
                "tool:(BODY)".to_string(),
                "^whole$".to_string()
            ]
        );
    }

    #[test]
    fn lazy_fields_reject_non_atomic_preserved_oov_and_silent_combinations() {
        let tokenizer = tokenizer_with_specials(&["<a>", "<b>"]);
        let cases = [
            serde_json::json!({
                "grammar":"root ::= \"x\"\n",
                "grammar_lazy":true,
                "preserved_tokens":["<a><b>"],
                "grammar_triggers":[{"type":2,"value":"x"}]
            }),
            serde_json::json!({
                "grammar":"root ::= \"x\"\n",
                "grammar_lazy":true,
                "grammar_triggers":[{"type":0,"value":"x","token":999}]
            }),
            serde_json::json!({
                "grammar":"root ::= \"x\"\n",
                "preserved_tokens":["<a>"]
            }),
            serde_json::json!({
                "grammar":"root ::= \"x\"\n",
                "grammar_lazy":true,
                "grammar_triggers":[]
            }),
            serde_json::json!({
                "grammar_lazy":true,
                "grammar_triggers":[{"type":2,"value":"x"}]
            }),
            serde_json::json!({
                "grammar":"root ::= \"x\"\n",
                "grammar_lazy":true,
                "grammar_triggers":[{"type":2,"value":"(?=unsupported-lookahead)"}]
            }),
        ];
        for fields in cases {
            assert!(
                compile_request_constraint_with_tokenizer(&request(fields), &tokenizer).is_err()
            );
        }
    }

    #[test]
    fn single_token_word_trigger_requires_preservation() {
        let tokenizer = tokenizer_with_specials(&["<word>"]);
        let error = compile_request_constraint_with_tokenizer(
            &request(serde_json::json!({
                "grammar":"root ::= \"x\"\n",
                "grammar_lazy":true,
                "grammar_triggers":[{"type":1,"value":"<word>"}]
            })),
            &tokenizer,
        )
        .unwrap_err();
        assert_eq!(error.param, "grammar_triggers");
        assert!(error.message.contains("preserved_tokens"));
    }

    #[test]
    fn structural_tag_tokens_bind_to_the_selected_model_vocabulary() {
        let tokenizer = tokenizer_with_specials(&["<open>"]);
        let token_id = tokenizer.token_to_id("<open>").unwrap();
        let compiled = compile_request_constraint_with_tokenizer(
            &request(serde_json::json!({
                "response_format": {
                    "type":"structural_tag",
                    "format":{"type":"token","token":"<open>"}
                }
            })),
            &tokenizer,
        )
        .unwrap();
        let grammar = compiled.grammar.unwrap();
        assert!(grammar
            .rules
            .iter()
            .flatten()
            .any(|element| { element.ty == parser::GretType::Token && element.value == token_id }));

        let error = compile_request_constraint_with_tokenizer(
            &request(serde_json::json!({
                "response_format": {
                    "type":"structural_tag",
                    "format":{"type":"token","token":999}
                }
            })),
            &tokenizer,
        )
        .unwrap_err();
        assert_eq!(error.param, "response_format");
        assert!(error.message.contains("not present"), "{error}");
    }
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;
