use super::*;
use crate::serve::api::schema::ToolChoiceValue;

fn request_with(fields: Value) -> ChatCompletionRequest {
    let mut request = serde_json::json!({
        "model": "gemma4-27b-it",
        "messages": [{"role": "user", "content": "hi"}]
    });
    request
        .as_object_mut()
        .expect("request object")
        .extend(fields.as_object().expect("fields object").clone());
    serde_json::from_value(request).expect("request fixture")
}

fn accepts(grammar: &Grammar, bytes: &[u8]) -> bool {
    let root = grammar.rule_id("root").expect("root");
    let mut runtime = super::super::GrammarRuntime::new(grammar.clone(), root).expect("runtime");
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

    let lark = StructuredOutputs {
        grammar: Some("start: \"yes\" | \"no\"".into()),
        ..Default::default()
    };
    let grammar = compile_structured_outputs(&lark).unwrap();
    assert!(accepts(&grammar, b"yes"));
    assert!(!accepts(&grammar, b"maybe"));
}

#[test]
fn current_and_legacy_structural_tag_surfaces_compile_and_enforce() {
    let current = request_with(serde_json::json!({
        "structured_outputs": {
            "structural_tag": serde_json::json!({
                "type":"structural_tag",
                "format":{"type":"const_string","value":"ok"}
            }).to_string()
        }
    }));
    let grammar = compile_request_constraint(&current)
        .unwrap()
        .expect("current structural tag grammar");
    assert!(accepts(&grammar, b"ok"));
    assert!(!accepts(&grammar, b"no"));

    let legacy = request_with(serde_json::json!({
        "response_format": {
            "type":"structural_tag",
            "structures":[{
                "begin":"<call>",
                "schema":{"type":"string","const":"ok"},
                "end":"</call>"
            }],
            "triggers":["<call>"]
        }
    }));
    let grammar = compile_request_constraint(&legacy)
        .unwrap()
        .expect("legacy structural tag grammar");
    assert!(accepts(&grammar, b"text<call>\"ok\"</call>tail"));
    assert!(!accepts(&grammar, b"text<call>\"no\"</call>tail"));
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

    for structured in [
        StructuredOutputs {
            choice: Some(vec!["x".into()]),
            whitespace_pattern: Some("[ ]*".into()),
            ..Default::default()
        },
        StructuredOutputs {
            grammar: Some("root ::= \"x\"".into()),
            disable_any_whitespace: Some(true),
            ..Default::default()
        },
    ] {
        assert!(
            compile_structured_outputs(&structured).is_err(),
            "JSON whitespace options must not be silently ignored"
        );
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
fn response_format_replaces_same_structured_slot_and_keeps_backend_options() {
    let request = request_with(serde_json::json!({
        "structured_outputs": {
            "json": {"type":"string", "const":"structured"},
            "disable_any_whitespace": true
        },
        "response_format": {
            "type":"json_schema",
            "json_schema": {
                "name":"response",
                "schema": {"type":"string", "const":"response"}
            }
        }
    }));
    let grammar = compile_request_constraint(&request)
        .unwrap()
        .expect("response grammar");
    assert!(accepts(&grammar, br#""response""#));
    assert!(!accepts(&grammar, br#""structured""#));

    let request = request_with(serde_json::json!({
        "structured_outputs": {
            "json_object": true,
            "disable_any_whitespace": true
        },
        "response_format": {"type":"json_object"}
    }));
    let grammar = compile_request_constraint(&request)
        .unwrap()
        .expect("json object grammar");
    assert!(accepts(&grammar, br#"{"x":1}"#));
    assert!(!accepts(&grammar, br#"{ "x": 1 }"#));
}

#[test]
fn response_format_conflicts_with_a_different_structured_slot() {
    let request = request_with(serde_json::json!({
        "structured_outputs": {"regex":"[a-z]+"},
        "response_format": {"type":"json_object"}
    }));
    let error = compile_request_constraint(&request).unwrap_err();
    assert_eq!(error.param, "response_format");
    assert!(error.message.contains("different constraint"));
}

#[test]
fn top_level_boolean_json_schema_compiles() {
    let request = request_with(serde_json::json!({"json_schema": true}));
    let grammar = compile_request_constraint(&request)
        .unwrap()
        .expect("boolean schema grammar");
    assert!(accepts(&grammar, br#"{"anything":[1,true,null]}"#));
}

#[test]
fn tool_request_matrix_and_definitions_fail_closed() {
    let absent = request_with(serde_json::json!({}));
    assert_eq!(
        validate_tool_request(&absent).unwrap(),
        ToolChoiceValue::Auto
    );

    let explicit_none = request_with(serde_json::json!({"tool_choice":"none"}));
    assert_eq!(
        validate_tool_request(&explicit_none).unwrap(),
        ToolChoiceValue::None
    );

    for (fields, param) in [
        (serde_json::json!({"tool_choice":"auto"}), "tool_choice"),
        (serde_json::json!({"tool_choice":"required"}), "tool_choice"),
        (serde_json::json!({"tools":[]}), "tools"),
        (
            serde_json::json!({
                "tools":[{"type":"other","function":{"name":"lookup"}}]
            }),
            "tools[0].type",
        ),
        (
            serde_json::json!({
                "tools":[{"type":"function","function":{"name":"bad name"}}]
            }),
            "tools[0].function.name",
        ),
        (
            serde_json::json!({
                "tools":[{"type":"function","function":{"name":"lookup","parameters":7}}]
            }),
            "tools[0].function.parameters",
        ),
        (
            serde_json::json!({
                "tools":[
                    {"type":"function","function":{"name":"lookup"}},
                    {"type":"function","function":{"name":"lookup"}}
                ]
            }),
            "tools[1].function.name",
        ),
        (
            serde_json::json!({
                "tools":[{"type":"function","function":{"name":"lookup"}}],
                "tool_choice":{"type":"function","function":{"name":"missing"}}
            }),
            "tool_choice.function.name",
        ),
    ] {
        let request = request_with(fields);
        let error = validate_tool_request(&request).unwrap_err();
        assert_eq!(error.param, param);
    }

    for parameters in [
        serde_json::json!({"type":"object"}),
        serde_json::json!(true),
    ] {
        let request = request_with(serde_json::json!({
            "tools":[{
                "type":"function",
                "function":{"name":"lookup_1", "parameters":parameters}
            }],
            "tool_choice":{"type":"function","function":{"name":"lookup_1"}}
        }));
        assert_eq!(
            validate_tool_request(&request).unwrap(),
            ToolChoiceValue::Function("lookup_1".into())
        );
    }
}

#[test]
fn malformed_tool_choice_is_attributed_to_tool_choice() {
    let request = request_with(serde_json::json!({"tool_choice":"sometimes"}));
    let error = validate_tool_request(&request).unwrap_err();
    assert_eq!(error.param, "tool_choice");
    assert!(error.message.contains("sometimes"));
}

#[test]
fn required_tool_choice_precedes_unused_output_constraint() {
    let request = request_with(serde_json::json!({
        "tools":[{"type":"function","function":{"name":"lookup"}}],
        "tool_choice":"required",
        "structured_outputs":{"grammar":"not valid gbnf"},
        "response_format":{"type":"structural_tag", "format":{}}
    }));
    let tool_choice = validate_tool_request(&request).unwrap();
    assert!(
        compile_request_output_constraint(&request, &tool_choice)
            .unwrap()
            .is_none()
    );
}

#[test]
fn automatic_tools_do_not_erase_an_explicit_output_constraint() {
    let request = request_with(serde_json::json!({
        "tools":[{"type":"function","function":{"name":"lookup"}}],
        "response_format":{"type":"json_object"}
    }));
    let tool_choice = validate_tool_request(&request).unwrap();
    assert_eq!(tool_choice, ToolChoiceValue::Auto);
    assert!(
        compile_request_output_constraint(&request, &tool_choice)
            .unwrap()
            .is_some()
    );
}

#[test]
fn llama_lazy_fields_validate_consistently_and_fail_if_runtime_is_unavailable() {
    for (fields, param) in [
        (serde_json::json!({"grammar_lazy":false}), "grammar_lazy"),
        (
            serde_json::json!({"preserved_tokens":[]}),
            "preserved_tokens",
        ),
        (
            serde_json::json!({"grammar_triggers":[]}),
            "grammar_triggers",
        ),
        (
            serde_json::json!({
                "grammar":"root ::= \"ok\"",
                "grammar_triggers":[{"type":1,"value":"<tool>"}]
            }),
            "grammar_triggers",
        ),
        (
            serde_json::json!({
                "grammar":"root ::= \"ok\"",
                "grammar_lazy":true
            }),
            "grammar_triggers",
        ),
        (
            serde_json::json!({
                "grammar":"root ::= \"ok\"",
                "grammar_lazy":true,
                "grammar_triggers":[{"type":1,"value":""}]
            }),
            "grammar_triggers[0].value",
        ),
        (
            serde_json::json!({
                "grammar":"root ::= \"ok\"",
                "grammar_lazy":true,
                "grammar_triggers":[{"type":0,"value":"<tool>","token":-1}]
            }),
            "grammar_triggers[0].token",
        ),
        (
            serde_json::json!({
                "grammar":"root ::= \"ok\"",
                "grammar_lazy":true,
                "grammar_triggers":[{"type":1,"value":"<tool>"}],
                "preserved_tokens":["<tool>","<tool>"]
            }),
            "preserved_tokens[1]",
        ),
    ] {
        let request = request_with(fields);
        let error = compile_request_constraint(&request).unwrap_err();
        assert_eq!(error.param, param);
    }

    let valid_but_not_yet_executable = request_with(serde_json::json!({
        "grammar":"root ::= \"ok\"",
        "grammar_lazy":true,
        "grammar_triggers":[{"type":1,"value":"<tool>"}],
        "preserved_tokens":["<tool>"]
    }));
    let error = compile_request_constraint(&valid_but_not_yet_executable).unwrap_err();
    assert_eq!(error.param, "grammar_lazy");
    assert!(error.message.contains("not yet implemented"));
}

#[test]
fn native_required_tool_choice_rejects_external_lazy_modifiers() {
    let request = request_with(serde_json::json!({
        "tools":[{"type":"function","function":{"name":"lookup"}}],
        "tool_choice":"required",
        "grammar_lazy":true,
        "grammar_triggers":[{"type":1,"value":"<tool>"}]
    }));
    let choice = validate_tool_request(&request).unwrap();
    let error = compile_request_output_constraint(&request, &choice).unwrap_err();
    assert_eq!(error.param, "grammar_lazy");
}
