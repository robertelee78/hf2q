//! Issue #251: real agentic tool schemas with length bounds above 2,000
//! (for example OpenCode MCP `argv` items with `maxLength: 4096`) MUST
//! compile into every native tool-call grammar instead of failing the whole
//! chat request with HTTP 400.

use super::{combine_function_grammars, compile_tool_grammar_with_registration, registry};
use crate::serve::api::schema::{ChatCompletionRequest, ToolChoiceValue};
use serde_json::json;

fn request(tools: serde_json::Value) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "run it"}],
        "tools": tools
    }))
    .expect("chat request fixture")
}

fn argv_tool(name: &str) -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string", "maxLength": 4096}}
                },
                "required": ["argv"]
            }
        }
    })
}

#[test]
fn argv_items_max_length_4096_compiles_for_every_native_tool_family() {
    let other = json!({
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 8192}},
                "required": ["path"]
            }
        }
    });
    for model in FAMILY_MODELS {
        let registration = registry::find_for(model)
            .unwrap_or_else(|| panic!("{model} must resolve to a registered family"));
        for tools in [
            json!([argv_tool("bash")]),
            json!([argv_tool("bash"), other.clone()]),
        ] {
            let req = request(tools);
            for choice in [ToolChoiceValue::Required, ToolChoiceValue::Auto] {
                let grammar =
                    compile_tool_grammar_with_registration(&req, &choice, Some(&registration))
                        .unwrap_or_else(|response| {
                            panic!(
                                "{model} ({}) {choice:?} must compile, got HTTP {}",
                                registration.family,
                                response.status()
                            )
                        });
                assert!(grammar.is_some(), "{model} {choice:?} produced no grammar");
            }
        }
    }
}

const FAMILY_MODELS: [&str; 4] = [
    "gemma4-27b-it",
    "Qwen3.8 27B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "DeepSeek-V4-Flash-0731",
];

fn bounded_tool(name: &str, bound: u64) -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": bound},
                    "argv": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": bound},
                        "maxItems": bound
                    }
                },
                "required": ["text"]
            }
        }
    })
}

/// The shared-block composition keeps per-field cost at O(S + N / S), so a
/// large agentic tool list (40 tools, each with two 4,096 length bounds and
/// a 4,096 maxItems) and several tools at the 65,536 ceiling stay within the
/// combined-grammar limits for every family.
#[test]
fn many_bounded_tools_fit_combined_grammar_limits() {
    let forty = (0..40)
        .map(|index| bounded_tool(&format!("t{index}"), 4096))
        .collect::<Vec<_>>();
    let ceiling = (0..3)
        .map(|index| bounded_tool(&format!("c{index}"), 65_536))
        .collect::<Vec<_>>();
    for model in FAMILY_MODELS {
        let registration = registry::find_for(model).expect("registered family");
        for tools in [&forty, &ceiling] {
            let req = request(json!(tools));
            let grammar = compile_tool_grammar_with_registration(
                &req,
                &ToolChoiceValue::Auto,
                Some(&registration),
            )
            .unwrap_or_else(|response| {
                let body = futures::executor::block_on(axum::body::to_bytes(
                    response.into_body(),
                    usize::MAX,
                ))
                .unwrap_or_default();
                panic!("{model}: {}", String::from_utf8_lossy(&body))
            });
            assert!(grammar.is_some(), "{model} produced no grammar");
        }
    }
}

/// Exceeding the combined grammar byte limit is a request error, not a
/// combiner invariant panic.
#[test]
fn combined_grammar_resource_exhaustion_is_an_error() {
    let large = format!("root ::= \"{}\"\n", "a".repeat(2_500_000));
    let error = combine_function_grammars(vec![large.clone(), large], false, "")
        .expect_err("combined grammar above the raw byte limit");
    assert!(error.message.contains("resource limit"), "{error}");
}
