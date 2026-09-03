//! Request-surface normalization for OpenAI, vLLM, and llama.cpp grammar
//! controls. Every accepted surface is lowered to the one hf2q GBNF runtime;
//! ambiguous or unsupported combinations fail before generation starts.

use serde_json::Value;

use super::{json_schema, parser, regex_gbnf, Grammar};
use crate::serve::api::schema::{
    ChatCompletionRequest, ResponseFormat, StructuredOutputJson, StructuredOutputs,
};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestGrammarError {
    pub param: &'static str,
    pub message: String,
}

impl RequestGrammarError {
    fn new(param: &'static str, message: impl Into<String>) -> Self {
        Self {
            param,
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

fn compile_schema(
    schema: &Value,
    param: &'static str,
    whitespace_pattern: Option<&str>,
) -> Result<Grammar, RequestGrammarError> {
    let source = json_schema::schema_to_gbnf_with_whitespace(schema, whitespace_pattern).map_err(
        |error| RequestGrammarError::new(param, format!("JSON Schema compilation failed: {error}")),
    )?;
    parse_gbnf(&source, param)
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

pub fn compile_structured_outputs(
    structured: &StructuredOutputs,
) -> Result<Grammar, RequestGrammarError> {
    structured
        .validate_exactly_one_constraint()
        .map_err(|error| RequestGrammarError::new("structured_outputs", error.to_string()))?;
    let whitespace_pattern = configured_whitespace(structured)?;

    if let Some(choices) = structured.choice.as_ref() {
        if choices.is_empty() {
            return Err(RequestGrammarError::new(
                "structured_outputs.choice",
                "choice must contain at least one string",
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
        return parse_gbnf(source, "structured_outputs.grammar");
    }
    Err(RequestGrammarError::new(
        "structured_outputs.structural_tag",
        "structural_tag compilation is not yet implemented",
    ))
}

/// Resolve all non-tool request surfaces to one grammar. vLLM's documented
/// response-format conversion wins over `structured_outputs`, except `text`,
/// which preserves the explicit structured-output request.
pub fn compile_request_constraint(
    request: &ChatCompletionRequest,
) -> Result<Option<Grammar>, RequestGrammarError> {
    if request.grammar.is_some() && request.json_schema.is_some() {
        return Err(RequestGrammarError::new(
            "grammar",
            "grammar and json_schema are mutually exclusive",
        ));
    }
    let response_overrides = request
        .response_format
        .as_ref()
        .is_some_and(|format| !matches!(format, ResponseFormat::Text));
    if response_overrides && (request.grammar.is_some() || request.json_schema.is_some()) {
        return Err(RequestGrammarError::new(
            "response_format",
            "response_format cannot be combined with top-level grammar or json_schema",
        ));
    }
    if !response_overrides
        && request.structured_outputs.is_some()
        && (request.grammar.is_some() || request.json_schema.is_some())
    {
        return Err(RequestGrammarError::new(
            "structured_outputs",
            "structured_outputs cannot be combined with top-level grammar or json_schema",
        ));
    }

    match request.response_format.as_ref() {
        Some(ResponseFormat::JsonObject) => {
            return parse_gbnf(JSON_OBJECT_GRAMMAR, "response_format").map(Some)
        }
        Some(ResponseFormat::JsonSchema { json_schema: spec }) => {
            return compile_schema(&spec.schema, "response_format", None).map(Some)
        }
        Some(ResponseFormat::StructuralTag { .. }) => {
            return Err(RequestGrammarError::new(
                "response_format",
                "structural_tag compilation is not yet implemented",
            ))
        }
        Some(ResponseFormat::Text) | None => {}
    }
    if let Some(structured) = request.structured_outputs.as_ref() {
        return compile_structured_outputs(structured).map(Some);
    }
    if let Some(schema) = request.json_schema.as_ref() {
        return compile_schema(&Value::Object(schema.clone()), "json_schema", None).map(Some);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts(grammar: &Grammar, bytes: &[u8]) -> bool {
        let root = grammar.rule_id("root").expect("root");
        let mut runtime =
            super::super::GrammarRuntime::new(grammar.clone(), root).expect("runtime");
        runtime.accept_bytes(bytes) && runtime.is_accepted()
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
}
