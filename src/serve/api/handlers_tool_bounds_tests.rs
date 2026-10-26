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

#[test]
fn private_dsml_names_do_not_capture_public_properties_or_function_names() {
    // Both schema property sanitizers replace '_' with '-'. Private markers
    // deliberately retain '_' so neither legacy nor private-looking user keys
    // can acquire the runtime-only DSML semantics through generated suffixes.
    let keys = [
        "dsml-data-raw",
        "dsml-data-json-raw",
        "dsml-data-raw-end",
        "hf2q_dsml_data_raw",
        "hf2q_dsml_json_raw",
        "hf2q_dsml_raw_end",
    ];
    let properties = keys
        .iter()
        .map(|key| ((*key).to_owned(), json!({"type":"string","maxLength":4})))
        .collect::<serde_json::Map<_, _>>();
    let tools = keys
        .iter()
        .skip(3)
        .map(|key| {
            json!({"type":"function","function":{
        "name":key,"parameters":{"type":"object",
        "properties":properties,"required":keys}}})
        })
        .collect::<Vec<_>>();
    for model in FAMILY_MODELS {
        let registration = registry::find_for(model).unwrap();
        for choice in [ToolChoiceValue::Required, ToolChoiceValue::Auto] {
            let req = request(json!(tools));
            let grammar =
                compile_tool_grammar_with_registration(&req, &choice, Some(&registration))
                    .unwrap_or_else(|response| panic!("{model}: {}", response.status()))
                    .unwrap();
            let root = grammar.rule_id("root").unwrap();
            let mut runtime = super::grammar::sampler::GrammarRuntime::new(grammar, root)
                .unwrap_or_else(|| {
                    panic!("public property key became private DSML constraint: {model}")
                });
            // For non-DS families the exact NONE fast path is part of this
            // control, rather than inferring absence from successful creation.
            assert_eq!(
                runtime.has_dsml_guard_for_test(),
                registration.family == "deepseek4"
            );
            runtime.set_awaiting_trigger(false);
        }
        let name = "hf2q_dsml_data_raw";
        let schema = json!({"type":"object","properties":properties,"required":keys});
        let source = registration
            .tool_call_gbnf(name, &schema, registry::GrammarShape::SingleBody)
            .unwrap();
        let parsed = super::grammar::parser::parse(&source).unwrap();
        let root = parsed.rule_id("root").unwrap();
        let mut runtime = super::grammar::sampler::GrammarRuntime::new(parsed, root).unwrap();
        let wire=match registration.family {
            "gemma4"=>format!("call:{name}{{{}}}",keys.iter().map(|key|format!("{key}:<|\"|>ok<|\"|>")).collect::<Vec<_>>().join(",")),
            "qwen35"=>format!("<function={name}>\n{}</function>",keys.iter().map(|key|format!("<parameter={key}>\nok\n</parameter>\n")).collect::<String>()),
            "qwen3vl"=>json!({"name":name,"arguments":keys.iter().map(|k|((*k).to_owned(),json!("ok"))).collect::<serde_json::Map<_,_>>()}).to_string(),
            "deepseek4"=>format!("<｜DSML｜invoke name=\"{name}\">\n{}</｜DSML｜invoke>",keys.iter().map(|key|format!("<｜DSML｜parameter name=\"{key}\" string=\"true\">ok</｜DSML｜parameter>\n")).collect::<String>()),
            family=>panic!("unexpected test family {family}"),
        };
        assert!(
            runtime.accept_bytes(wire.as_bytes()) && runtime.is_accepted(),
            "private-looking public keys rejected actual {model} body"
        );
    }
}
