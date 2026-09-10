use super::apply_gcd_policy;
use crate::serve::api::grammar::{self, request::compile_request_output_constraint};
use crate::serve::api::schema::{ChatCompletionRequest, ToolChoiceValue};
use crate::serve::api::state::{AppState, ServerConfig};
use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use tower::ServiceExt;

fn schema_config(locked: bool) -> ServerConfig {
    let schema = json!({
        "type": "object",
        "properties": {"decision": {"type": "string", "enum": ["allow"]}},
        "required": ["decision"],
        "additionalProperties": false
    });
    ServerConfig {
        gcd_schema_grammar: Some(grammar::json_schema::schema_to_gbnf(&schema).unwrap()),
        gcd_schema_locked: locked,
        ..ServerConfig::default()
    }
}

fn tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}}
        }
    })
}

fn request_value(extra: Value, stream: bool) -> Value {
    let mut value = json!({
        "model": "locked-schema-test",
        "messages": [{"role": "user", "content": "Return the decision."}],
        "stream": stream,
        "hf2q_enable_thinking": true
    });
    value
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    value
}

fn request(extra: Value) -> ChatCompletionRequest {
    serde_json::from_value(request_value(extra, false)).unwrap()
}

fn explicit_constraints() -> Vec<(&'static str, Value)> {
    vec![
        ("grammar", json!("root ::= \"caller\"")),
        ("response_format", json!({"type": "json_object"})),
        ("json_schema", json!({"type": "object"})),
        ("structured_outputs", json!({"choice": ["caller"]})),
    ]
}

fn allowed_requests() -> Vec<Value> {
    vec![
        json!({}),
        json!({"tool_choice": "none"}),
        json!({"tool_choice": "none", "tools": [tool()]}),
    ]
}

fn rejected_tool_requests() -> Vec<(&'static str, Value)> {
    vec![
        ("tools", json!({"tools": [tool()]})),
        ("tools", json!({"tools": [tool()], "tool_choice": "auto"})),
        (
            "tool_choice",
            json!({"tools": [tool()], "tool_choice": "required"}),
        ),
        (
            "tool_choice",
            json!({
                "tools": [tool()],
                "tool_choice": {"type": "function", "function": {"name": "lookup"}}
            }),
        ),
    ]
}

fn invalid_tool_requests() -> Vec<(&'static str, Value)> {
    vec![
        ("tool_choice", json!({"tool_choice": "auto"})),
        ("tools", json!({"tool_choice": "auto", "tools": []})),
        ("tools", json!({"tool_choice": "none", "tools": []})),
        ("tools", json!({"tools": []})),
        ("tool_choice", json!({"tool_choice": "required"})),
        ("tool_choice", json!({"tool_choice": "sometimes"})),
        ("tool_choice", json!({"tool_choice": {"type": "function"}})),
    ]
}

fn grammar_modifiers() -> Vec<(&'static str, Value)> {
    vec![
        ("grammar_lazy", json!(true)),
        ("grammar_lazy", json!(false)),
        ("preserved_tokens", json!([])),
        ("grammar_triggers", json!([])),
    ]
}

#[test]
fn locked_policy_admits_text_and_keeps_the_schema_through_request_compilation() {
    let config = schema_config(true);
    for extra in allowed_requests() {
        let mut req = request(extra.clone());
        // Exercise the same composition as the HTTP handler, not separately
        // ordered gate/injection helpers (which missed the original defect).
        apply_gcd_policy(&config, &mut req).unwrap();
        assert_eq!(req.grammar, config.gcd_schema_grammar, "{extra}");
        assert_eq!(req.hf2q_enable_thinking, Some(false), "{extra}");
        let choice = grammar::request::validate_tool_request(&req).unwrap();
        let compiled = compile_request_output_constraint(&req, &choice)
            .unwrap()
            .expect("every admitted locked request must retain its response grammar");
        let root = compiled.rule_id("root").unwrap();
        for (bytes, expected) in [
            (br#"{"decision":"allow"}"#.as_slice(), true),
            (br#"{"decision":"deny"}"#.as_slice(), false),
            (br#"{"name":"lookup","arguments":{}}"#.as_slice(), false),
        ] {
            let mut runtime = grammar::GrammarRuntime::new(compiled.clone(), root).unwrap();
            let accepted = runtime.accept_bytes(bytes) && runtime.is_terminally_accepted();
            assert_eq!(accepted, expected, "request {extra}, output {bytes:?}");
        }
    }
}

#[test]
fn locked_policy_rejects_caller_constraints_and_activation_modifiers_before_mutation() {
    let config = schema_config(true);
    for (param, value) in explicit_constraints()
        .into_iter()
        .chain(grammar_modifiers())
    {
        let mut req = request(json!({param: value}));
        let original_grammar = req.grammar.clone();
        let error = apply_gcd_policy(&config, &mut req).unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.error.param.as_deref(), Some(param));
        assert!(error.error.message.contains("--gcd-schema-locked"));
        assert_eq!(
            req.grammar, original_grammar,
            "rejection must not inject a grammar"
        );
        assert_eq!(req.hf2q_enable_thinking, Some(true));
    }
}

#[test]
fn locked_policy_rejects_tool_precedence_and_malformed_choices() {
    let config = schema_config(true);
    for (param, extra) in rejected_tool_requests()
        .into_iter()
        .chain(invalid_tool_requests())
    {
        let mut req = request(extra.clone());
        let validation_error = grammar::request::validate_tool_request(&req).err();
        let error = apply_gcd_policy(&config, &mut req).unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST, "{extra}");
        assert_eq!(error.error.param.as_deref(), Some(param), "{extra}");
        if let Some(expected) = validation_error {
            assert_eq!(error.error.message, expected.message, "{extra}");
        }
        assert!(req.grammar.is_none(), "{extra}");
        assert_eq!(req.hf2q_enable_thinking, Some(true), "{extra}");
    }
}

#[test]
fn unlocked_defaults_preserve_explicit_constraints_and_native_tool_precedence() {
    for config in [
        schema_config(false),
        ServerConfig {
            gcd: true,
            ..ServerConfig::default()
        },
    ] {
        for (param, value) in explicit_constraints() {
            let mut req = request(json!({param: value}));
            let original = req.grammar.clone();
            apply_gcd_policy(&config, &mut req).unwrap();
            assert_eq!(
                req.grammar, original,
                "explicit {param} must suppress injection"
            );
            assert_eq!(req.hf2q_enable_thinking, Some(true));
        }
    }
    for (_, extra) in rejected_tool_requests() {
        let mut req = request(extra);
        apply_gcd_policy(&schema_config(false), &mut req).unwrap();
        let choice = ToolChoiceValue::try_parse(req.tool_choice.as_ref()).unwrap();
        let grammar = compile_request_output_constraint(&req, &choice).unwrap();
        assert_eq!(
            grammar.is_none(),
            matches!(
                choice,
                ToolChoiceValue::Required | ToolChoiceValue::Function(_)
            )
        );
    }
}

#[test]
fn defaults_remain_orthogonal_to_steering_and_locked_configuration_fails_closed() {
    let mut req = request(json!({}));
    apply_gcd_policy(
        &ServerConfig {
            gcd: true,
            ..ServerConfig::default()
        },
        &mut req,
    )
    .unwrap();
    assert_eq!(
        req.grammar.as_deref(),
        Some(include_str!("grammar/gcd_w1.gbnf"))
    );
    assert_eq!(req.hf2q_enable_thinking, Some(false));

    let mut req = request(json!({}));
    apply_gcd_policy(&ServerConfig::default(), &mut req).unwrap();
    assert!(req.grammar.is_none());
    assert_eq!(req.hf2q_enable_thinking, Some(true));

    for config in [
        ServerConfig {
            gcd_schema_locked: true,
            ..ServerConfig::default()
        },
        ServerConfig {
            gcd: true,
            ..schema_config(true)
        },
    ] {
        assert_eq!(
            apply_gcd_policy(&config, &mut req).unwrap_err().status,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert!(req.grammar.is_none());
    }
}

/// Drive the actual router, JSON extraction, and production handler. Readiness
/// stops admitted requests before model resolution, so these tests cannot load
/// weights or open SSE. Separate real-model gates cover successful generation.
async fn routed_request(config: ServerConfig, extra: Value, stream: bool) -> (StatusCode, Value) {
    let state = AppState::new(config);
    state.mark_not_ready();
    let app = crate::serve::api::build_router(state.clone());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request_value(extra, stream).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    assert!(response.headers()[header::CONTENT_TYPE]
        .to_str()
        .unwrap()
        .starts_with("application/json"));
    assert_eq!(
        state
            .metrics
            .chat_completions_started
            .load(Ordering::Relaxed),
        0
    );
    assert_eq!(state.metrics.requests_total.load(Ordering::Relaxed), 1);
    assert_eq!(
        state
            .metrics
            .requests_rejected_total
            .load(Ordering::Relaxed),
        1
    );
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn router_admits_locked_text_requests_to_readiness_for_unary_and_sse() {
    for stream in [false, true] {
        for extra in allowed_requests() {
            let (status, body) = routed_request(schema_config(true), extra.clone(), stream).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{extra}: {body}");
            assert_eq!(body["error"]["code"], "not_ready", "{body}");
        }
    }
}

#[tokio::test]
async fn router_rejects_locked_constraint_and_lazy_overrides_before_unary_or_sse() {
    for stream in [false, true] {
        for (param, value) in explicit_constraints()
            .into_iter()
            .chain(grammar_modifiers())
        {
            // A secondary unknown field must not hide the policy rejection.
            let extra = json!({param: value, "beam_search": true});
            let (status, body) = routed_request(schema_config(true), extra, stream).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(body["error"]["param"], param, "{body}");
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert!(body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("--gcd-schema-locked"));
        }
    }
}

#[tokio::test]
async fn router_rejects_locked_tool_requests_before_unary_or_sse() {
    for stream in [false, true] {
        for (param, extra) in rejected_tool_requests()
            .into_iter()
            .chain(invalid_tool_requests())
        {
            let validation_error =
                grammar::request::validate_tool_request(&request(extra.clone())).err();
            let (status, body) = routed_request(schema_config(true), extra, stream).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(body["error"]["param"], param, "{body}");
            if let Some(expected) = validation_error {
                assert_eq!(body["error"]["message"], expected.message, "{body}");
            } else {
                assert!(body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("--gcd-schema-locked"));
            }
        }
    }
}

#[tokio::test]
async fn router_keeps_unlocked_request_overrides_for_unary_and_sse() {
    for stream in [false, true] {
        let explicit = explicit_constraints()
            .into_iter()
            .map(|(param, value)| json!({param: value}));
        let tools = rejected_tool_requests().into_iter().map(|(_, extra)| extra);
        for extra in explicit.chain(tools) {
            let (status, body) = routed_request(schema_config(false), extra, stream).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
            assert_eq!(body["error"]["code"], "not_ready", "{body}");
        }
    }
}
