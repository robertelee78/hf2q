//! Non-strict function-tool hints use the production request deserializer.
use super::ChatCompletionRequest;
use serde_json::{json, Value};

fn request(strict: Option<Value>) -> Value {
    let tools: Vec<Value> = (0..12)
        .map(|index| {
            let mut function = json!({
                "name": format!("tool_{index}"),
                "description": "A public fixture tool",
                "parameters": {
                    "type": "object",
                    "properties": {"strict": {"type": "boolean"}},
                    "required": ["strict"],
                    "additionalProperties": false
                }
            });
            if let Some(value) = &strict {
                function["strict"] = value.clone();
            }
            json!({"type": "function", "function": function})
        })
        .collect();
    json!({"model": "fixture", "messages": [{"role": "user", "content": "Use a tool"}], "tools": tools})
}

fn normalized_tools(value: Value) -> Value {
    let request: ChatCompletionRequest = serde_json::from_value(value).unwrap();
    serde_json::to_value(request.tools.unwrap()).unwrap()
}

#[test]
fn tool_strict_nonstrict_hints_normalize_twelve_tools_without_changing_schemas() {
    let omitted = request(None);
    let expected = omitted["tools"].clone();
    for strict in [None, Some(json!(false)), Some(Value::Null)] {
        let normalized = normalized_tools(request(strict));
        assert_eq!(normalized, expected);
        for tool in normalized.as_array().unwrap() {
            assert!(tool["function"].get("strict").is_none());
            assert_eq!(
                tool["function"]["parameters"]["properties"]["strict"],
                json!({"type": "boolean"})
            );
        }
    }
}

#[test]
fn tool_strict_true_rejects_the_whole_request_even_in_the_last_tool() {
    let mut value = request(Some(json!(false)));
    value["tools"][11]["function"]["strict"] = json!(true);
    let error = serde_json::from_value::<ChatCompletionRequest>(value).unwrap_err();
    assert!(error
        .to_string()
        .contains("unsupported strict tool mode: tools[].function.strict=true"));
}

#[test]
fn tool_strict_wrong_types_and_unknown_tool_fields_reject() {
    for strict in [json!("false"), json!(0), json!(1), json!({}), json!([])] {
        assert!(serde_json::from_value::<ChatCompletionRequest>(request(Some(strict))).is_err());
    }
    for inner in [false, true] {
        let mut value = request(Some(json!(false)));
        if inner {
            value["tools"][0]["function"]["unknown"] = json!(true);
        } else {
            value["tools"][0]["unknown"] = json!(true);
        }
        assert!(serde_json::from_value::<ChatCompletionRequest>(value).is_err());
    }
}

#[test]
fn tool_strict_duplicate_fields_reject_in_raw_production_requests() {
    // Value fixtures cannot preserve duplicate keys; exercise raw JSON instead.
    for function in [
        r#"{"name":"lookup","strict":false,"strict":false}"#,
        r#"{"name":"lookup","strict":null,"strict":false}"#,
        r#"{"name":"lookup","name":"other","strict":false}"#,
        r#"{"name":"lookup","parameters":null,"parameters":{}}"#,
    ] {
        let raw = format!(
            r#"{{"model":"fixture","messages":[],"tools":[{{"type":"function","function":{function}}}]}}"#
        );
        let error = serde_json::from_str::<ChatCompletionRequest>(&raw).unwrap_err();
        assert!(error.to_string().contains("duplicate field"), "{error}");
    }
}

#[test]
fn tool_strict_preserves_optional_field_defaults_and_nulls() {
    for function in [
        json!({"name": "lookup", "strict": false}),
        json!({"name": "lookup", "description": null, "parameters": null, "strict": null}),
    ] {
        let value = json!({"model":"fixture", "messages":[], "tools":[{"type":"function","function":function}]});
        assert_eq!(
            normalized_tools(value),
            json!([{"type":"function","function":{"name":"lookup"}}])
        );
    }
}
