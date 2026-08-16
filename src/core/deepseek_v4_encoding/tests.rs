use super::*;
use serde::Deserialize;

const INPUT_1: &str = include_str!("../../../tests/fixtures/deepseek_v4_encoding/input_1.json");
const OUTPUT_1: &str = include_str!("../../../tests/fixtures/deepseek_v4_encoding/output_1.txt");
const INPUT_2: &str = include_str!("../../../tests/fixtures/deepseek_v4_encoding/input_2.json");
const OUTPUT_2: &str = include_str!("../../../tests/fixtures/deepseek_v4_encoding/output_2.txt");
const INPUT_3: &str = include_str!("../../../tests/fixtures/deepseek_v4_encoding/input_3.json");
const OUTPUT_3: &str = include_str!("../../../tests/fixtures/deepseek_v4_encoding/output_3.txt");
const INPUT_4: &str = include_str!("../../../tests/fixtures/deepseek_v4_encoding/input_4.json");
const OUTPUT_4: &str = include_str!("../../../tests/fixtures/deepseek_v4_encoding/output_4.txt");

fn options(mode: ThinkingMode) -> EncodeOptions {
    EncodeOptions {
        thinking_mode: mode,
        ..EncodeOptions::default()
    }
}

fn official_bytes(fixture: &'static str) -> &'static str {
    // The upstream .txt files omit a final newline. Repository text files
    // retain one; that transport-only byte is not part of the vector.
    fixture.strip_suffix('\n').unwrap_or(fixture)
}

#[derive(Deserialize)]
struct ToolFixture {
    tools: Vec<ToolDefinition>,
    messages: Vec<Message>,
}

#[test]
fn official_case_1_thinking_with_tools_and_parse() {
    let mut fixture: ToolFixture = serde_json::from_str(INPUT_1).unwrap();
    fixture.messages[0].tools = fixture.tools;
    let prompt = encode_messages(&fixture.messages, options(ThinkingMode::Thinking)).unwrap();
    assert_eq!(prompt, official_bytes(OUTPUT_1));

    let marker = format!("{ASSISTANT}{THINK_START}");
    let start = prompt.find(&marker).unwrap() + marker.len();
    let end = prompt[start..].find(USER).unwrap() + start;
    let parsed = parse_completion(&prompt[start..end], ThinkingMode::Thinking).unwrap();
    assert_eq!(
        parsed.reasoning_content,
        "The user wants to know the weather in Beijing. I should use the get_weather tool."
    );
    assert_eq!(parsed.content, "");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].call_type, "function");
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&parsed.tool_calls[0].function.arguments)
            .unwrap(),
        serde_json::json!({"location":"Beijing","unit":"celsius"})
    );
}

#[test]
fn official_case_2_drops_old_thinking() {
    let prompt = encode_json(INPUT_2, options(ThinkingMode::Thinking)).unwrap();
    assert_eq!(prompt, official_bytes(OUTPUT_2));
    assert!(!prompt.contains("The user said hello"));

    let marker = format!("{ASSISTANT}{THINK_START}");
    let start = prompt.rfind(&marker).unwrap() + marker.len();
    let parsed = parse_completion(&prompt[start..], ThinkingMode::Thinking).unwrap();
    assert_eq!(
        parsed.reasoning_content,
        "The user asks about the capital of France. It is Paris."
    );
    assert_eq!(parsed.content, "The capital of France is Paris.");
    assert!(parsed.tool_calls.is_empty());
}

#[test]
fn official_case_3_developer_tools_and_latest_reminder() {
    assert_eq!(
        encode_json(INPUT_3, options(ThinkingMode::Thinking)).unwrap(),
        official_bytes(OUTPUT_3)
    );
}

#[test]
fn official_case_4_chat_quick_instruction() {
    assert_eq!(
        encode_json(INPUT_4, options(ThinkingMode::Chat)).unwrap(),
        official_bytes(OUTPUT_4)
    );
}

#[test]
fn reasoning_effort_is_thinking_only() {
    let input = r#"[{"role":"user","content":"Why?"}]"#;
    let high = encode_json(
        input,
        EncodeOptions {
            thinking_mode: ThinkingMode::Thinking,
            reasoning_effort: ReasoningEffort::High,
            ..EncodeOptions::default()
        },
    )
    .unwrap();
    assert!(high.starts_with(&format!("{BOS}{EFFORT_HIGH}")));
    let chat = encode_json(
        input,
        EncodeOptions {
            thinking_mode: ThinkingMode::Chat,
            reasoning_effort: ReasoningEffort::Max,
            ..EncodeOptions::default()
        },
    )
    .unwrap();
    assert_eq!(chat, format!("{BOS}{USER}Why?{ASSISTANT}{THINK_END}"));
}

#[test]
fn malformed_completion_fails_closed() {
    let err = parse_completion("unfinished", ThinkingMode::Thinking).unwrap_err();
    assert!(err.to_string().contains("missing </think>"));
}

#[test]
fn empty_tool_arguments_round_trip_with_official_blank_line() {
    let input = r#"[{"role":"user","content":"Run it"},{"role":"assistant","tool_calls":[{"id":"x","type":"function","function":{"name":"ping","arguments":"{}"}}]}]"#;
    let prompt = encode_json(input, options(ThinkingMode::Chat)).unwrap();
    assert!(prompt.contains("<｜DSML｜invoke name=\"ping\">\n\n</｜DSML｜invoke>"));
    let start = prompt.find(THINK_END).unwrap() + THINK_END.len();
    let parsed = parse_completion(&prompt[start..], ThinkingMode::Chat).unwrap();
    assert_eq!(parsed.tool_calls[0].function.arguments, "{}");
}

#[test]
fn malformed_non_string_parameter_fails_closed() {
    let body = "\n<｜DSML｜invoke name=\"question\">\n<｜DSML｜parameter name=\"questions\" string=\"false\">[{\"header\":null,\"options\":{\"label\":\"Movies\",}</｜DSML｜parameter>\n</｜DSML｜invoke>\n";
    let error = parse_tool_calls_body(body).expect_err("invalid JSON must not become tool args");
    assert!(error.to_string().contains("is not valid JSON"), "{error}");
}
