//! Deterministic, privacy-safe reproduction fixture for the Qwen 3.6
//! long-prefill watchdog incident.
//!
//! The original OpenCode request logged only its rendered token and tool
//! counts; its private message text and dynamically assembled tool schemas
//! were not persisted. This test deliberately generates a public equivalent
//! and verifies it through the exact GGUF chat template and tokenizer used by
//! production serving. No model tensors or Metal kernels are loaded.

use super::engine::render_chat_prompt_with_tools;
use super::schema::{ChatMessage, MessageContent, Tool, ToolFunction};
use crate::inference::models::qwen35::tokenizer::build_tokenizer_from_gguf;
use mlx_native::gguf::GgufFile;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

const MODEL_ENV: &str = "HF2Q_QWEN36_WATCHDOG_FIXTURE_MODEL";
const OUTPUT_ENV: &str = "HF2Q_QWEN36_WATCHDOG_FIXTURE_OUTPUT";
const SHORT_OUTPUT_ENV: &str = "HF2Q_QWEN36_WATCHDOG_SHORT_FIXTURE_OUTPUT";
const MODEL_ID: &str = "qwen36-abliterix-t63-APEX";
const TOOL_COUNT: usize = 347;
const TARGET_PROMPT_TOKENS: usize = 87_972;
const SHORT_PROMPT_TOKENS: usize = 552;
const LONG_PADDING_REPETITIONS: usize = 56_122;
const SHORT_PADDING_REPETITIONS: usize = 496;
const REQUEST_SHA256: &str = "6671a0c89b8d4935caa4b87bee08361c5b8727ec557e9edb05947ad90c94c13d";
const TOOLS_SHA256: &str = "586e09658c8d4d69b1ad451c8218199e405eeb72de4e550741730e83ed653766";
const SHORT_REQUEST_SHA256: &str =
    "7aeddea35e6363c698ea0bcb4934b9f2cf1e0c48fb2045fa9db3272461e54004";
const TEMPLATE_SHA256: &str = "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259";
const SYSTEM_TEXT: &str = "You are exercising a deterministic public Qwen 3.6 serving reliability fixture. Use only the declared fixture tools when a tool is required.";
const USER_PREFIX: &str =
    "Reliability fixture payload follows. Preserve service responsiveness while processing it.\n";

fn fixture_tools() -> Vec<Tool> {
    (0..TOOL_COUNT)
        .map(|index| Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: format!("fixture_tool_{index:03}"),
                description: Some(format!(
                    "Deterministic public workspace inspection fixture tool {index:03}."
                )),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Public workspace-relative path"
                        },
                        "line": {
                            "type": "integer",
                            "description": "One-based source line"
                        },
                        "include_hidden": {
                            "type": "boolean",
                            "description": "Whether hidden entries are included"
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                })),
            },
        })
        .collect()
}

fn fixture_messages(repetitions: usize) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".to_string(),
            content: Some(MessageContent::Text(SYSTEM_TEXT.to_string())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text(format!(
                "{USER_PREFIX}{}\nCall fixture_tool_346 with path src/serve/api/engine.rs.",
                " x".repeat(repetitions)
            ))),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ]
}

fn short_fixture_messages(repetitions: usize) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".to_string(),
            content: Some(MessageContent::Text(
                "You are the short live lane in a deterministic Qwen scheduler reliability fixture."
                    .to_string(),
            )),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: Some(MessageContent::Text(
                "Ready for the bounded scheduler check.".to_string(),
            )),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text(format!(
                "Short-lane padding follows.{}\nRespond with exactly OK.",
                " x".repeat(repetitions)
            ))),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ]
}

fn fixture_request(messages: &[ChatMessage], tools: &[Tool]) -> Value {
    json!({
        "hf2q_enable_thinking": false,
        "max_tokens": 8192,
        "messages": messages,
        "model": MODEL_ID,
        "parallel_tool_calls": false,
        "repetition_penalty": 1.0,
        "seed": 1234,
        "stream": true,
        "temperature": 0,
        "tool_choice": {
            "type": "function",
            "function": {"name": "fixture_tool_346"}
        },
        "tools": tools
    })
}

fn short_fixture_request(messages: &[ChatMessage]) -> Value {
    json!({
        "hf2q_enable_thinking": false,
        "max_tokens": 64,
        "messages": messages,
        "model": MODEL_ID,
        "repetition_penalty": 1.0,
        "seed": 1234,
        "stream": true,
        "temperature": 0
    })
}

fn canonical_json(value: &Value) -> Vec<u8> {
    fn sort_keys(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut keys: Vec<_> = object.keys().collect();
                keys.sort_unstable();
                let mut sorted = serde_json::Map::with_capacity(keys.len());
                for key in keys {
                    sorted.insert(key.clone(), sort_keys(&object[key]));
                }
                Value::Object(sorted)
            }
            Value::Array(values) => Value::Array(values.iter().map(sort_keys).collect()),
            other => other.clone(),
        }
    }

    serde_json::to_vec(&sort_keys(value)).expect("serialize canonical fixture JSON")
}

fn sha256_json(value: &Value) -> String {
    hex::encode(Sha256::digest(canonical_json(value)))
}

#[test]
fn public_watchdog_fixture_bytes_are_stable_without_a_model() {
    let tools = fixture_tools();
    let request = fixture_request(&fixture_messages(LONG_PADDING_REPETITIONS), &tools);
    let short_request = short_fixture_request(&short_fixture_messages(SHORT_PADDING_REPETITIONS));
    assert_eq!(sha256_json(&request), REQUEST_SHA256);
    assert_eq!(
        sha256_json(&serde_json::to_value(&tools).expect("serialize tools")),
        TOOLS_SHA256
    );
    assert_eq!(sha256_json(&short_request), SHORT_REQUEST_SHA256);
}

/// Header/tokenizer-only pin for the public 347-tool reproduction. The test
/// is ignored because it requires the local 25 GiB GGUF file, although it
/// reads only metadata and tokenizer tables.
#[test]
#[ignore = "requires HF2Q_QWEN36_WATCHDOG_FIXTURE_MODEL pointing at the canonical GGUF"]
fn public_347_tool_fixture_renders_to_exact_87972_tokens() {
    let model = std::env::var(MODEL_ENV).expect("set fixture GGUF path");
    let model = Path::new(&model);
    assert!(model.is_file(), "fixture GGUF does not exist: {model:?}");
    let gguf = GgufFile::open(model).expect("open fixture GGUF metadata");
    let tokenizer = build_tokenizer_from_gguf(&gguf).expect("build GGUF tokenizer");
    let template = gguf
        .metadata_string("tokenizer.chat_template")
        .expect("canonical GGUF chat template");
    let tools = fixture_tools();

    let token_count = |repetitions: usize| {
        let messages = fixture_messages(repetitions);
        let rendered =
            render_chat_prompt_with_tools(template, &messages, Some(&tools), false, None)
                .expect("render fixture through production template");
        tokenizer
            .encode(rendered, false)
            .expect("tokenize fixture through production tokenizer")
            .len()
    };

    let base_tokens = token_count(0);
    assert!(
        base_tokens < TARGET_PROMPT_TOKENS,
        "347-tool fixture already exceeds target: {base_tokens}"
    );
    // The first ` x` merges with USER_PREFIX's trailing newline boundary;
    // after that boundary every repetition contributes exactly one token.
    let repetitions = TARGET_PROMPT_TOKENS - base_tokens - 1;
    assert_eq!(repetitions, LONG_PADDING_REPETITIONS);
    assert_eq!(
        token_count(repetitions),
        TARGET_PROMPT_TOKENS,
        "the fixed ` x` tail must contribute exactly one token per repetition"
    );

    let messages = fixture_messages(repetitions);
    let tools_value = serde_json::to_value(&tools).expect("serialize tools");
    let request = fixture_request(&messages, &tools);
    if let Some(output) = std::env::var_os(OUTPUT_ENV) {
        std::fs::write(&output, canonical_json(&request)).expect("write fixture request JSON");
    }

    let short_token_count = |repetitions: usize| {
        let messages = short_fixture_messages(repetitions);
        let rendered = render_chat_prompt_with_tools(template, &messages, None, false, None)
            .expect("render short fixture through production template");
        tokenizer
            .encode(rendered, false)
            .expect("tokenize short fixture through production tokenizer")
            .len()
    };
    let short_base_tokens = short_token_count(0);
    assert!(short_base_tokens < SHORT_PROMPT_TOKENS);
    let short_repetitions = SHORT_PROMPT_TOKENS - short_base_tokens;
    assert_eq!(short_repetitions, SHORT_PADDING_REPETITIONS);
    assert_eq!(
        short_token_count(short_repetitions),
        SHORT_PROMPT_TOKENS,
        "the short public fixture must render to the original 552-token shape"
    );
    let short_request = short_fixture_request(&short_fixture_messages(short_repetitions));
    if let Some(output) = std::env::var_os(SHORT_OUTPUT_ENV) {
        std::fs::write(&output, canonical_json(&short_request))
            .expect("write short fixture request JSON");
    }
    let template_sha256 = hex::encode(Sha256::digest(template.as_bytes()));
    assert_eq!(sha256_json(&request), REQUEST_SHA256);
    assert_eq!(sha256_json(&tools_value), TOOLS_SHA256);
    assert_eq!(sha256_json(&short_request), SHORT_REQUEST_SHA256);
    assert_eq!(template_sha256, TEMPLATE_SHA256);
    eprintln!(
        "qwen36_watchdog_fixture repetitions={repetitions} prompt_tokens={TARGET_PROMPT_TOKENS} tools={TOOL_COUNT} request_sha256={} tools_sha256={} short_repetitions={short_repetitions} short_prompt_tokens={SHORT_PROMPT_TOKENS} short_request_sha256={} template_sha256={template_sha256}",
        sha256_json(&request),
        sha256_json(&tools_value),
        sha256_json(&short_request),
    );
}
