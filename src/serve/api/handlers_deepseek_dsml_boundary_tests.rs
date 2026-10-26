//! Real request compiler + token-acceptance regression; no model or HTTP.
use super::*;
use crate::serve::api::engine::{
    accept_grammar_token, grammar_runtime_for_request, GrammarKind, SamplingParams, ToolCallPolicy,
};
use crate::serve::api::schema::{ChatCompletionRequest, ToolChoiceValue};
use std::sync::Arc;

fn request() -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model":"DeepSeek V4 Flash 0731", "messages":[{"role":"user","content":"fix ports"}],
        "tool_choice":"auto", "parallel_tool_calls":false, "temperature":0,
        "reasoning_effort":"max", "max_completion_tokens":8192,
        "tools":[
            {"type":"function","function":{"name":"read_file","strict":false,
                "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
            {"type":"function","function":{"name":"write_file","strict":false,
                "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
            {"type":"function","function":{"name":"run_tests","strict":false,
                "parameters":{"type":"object","properties":{"target":{"type":"string","enum":["ports","all"]}},"required":["target"]}}}
        ]
    })).unwrap()
}

fn invoke(path: &str) -> String {
    format!("<｜DSML｜invoke name=\"read_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">{path}</｜DSML｜parameter>\n</｜DSML｜invoke>")
}

#[test]
fn deepseek_auto_nonparallel_compiled_grammar_rejects_two_parsed_invokes() {
    let registration = registry::find_for("DeepSeek-V4-Flash-0731").unwrap();
    let req = request();
    assert!(!effective_parallel_tool_calls(req.parallel_tool_calls));
    let grammar =
        compile_tool_grammar_for_request(&req, &ToolChoiceValue::Auto, Some(&registration), false)
            .expect("actual request compiler")
            .expect("native auto tool grammar");
    for count in [1, 2] {
        let body = if count == 1 {
            invoke("src/ports.rs")
        } else {
            format!("{}\n{}", invoke("src/ports.rs"), invoke("tests/ports.rs"))
        };
        assert_eq!(
            registry::parse_tool_call_bodies(&registration, &body)
                .unwrap()
                .len(),
            count
        );
        // Token fragments exercise the actual lazy marker scanner, including a
        // split marker and body suffix, rather than manually waking the runtime.
        let table = Arc::new(vec![
            b"reasoning preamble<".to_vec(),
            "｜DSML｜tool_calls>\n".as_bytes().to_vec(),
            format!("{body}\n</｜DSML｜tool_calls>").into_bytes(),
        ]);
        let params = SamplingParams {
            grammar: Some(grammar.clone()),
            token_bytes: Some(Arc::clone(&table)),
            grammar_kind: GrammarKind::ToolCallBodyAuto,
            tool_call_policy: ToolCallPolicy::AutoLazyGrammar,
            parallel_tool_calls: false,
            ..Default::default()
        };
        let mut runtime = grammar_runtime_for_request(&params, Some(&registration)).unwrap();
        assert!(runtime.as_ref().unwrap().is_awaiting_trigger());
        accept_grammar_token(&mut runtime, Some(table.as_slice()), &[], 0).unwrap();
        accept_grammar_token(&mut runtime, Some(table.as_slice()), &[], 1).unwrap();
        assert!(
            !runtime.as_ref().unwrap().is_awaiting_trigger(),
            "actual token path already activates unary"
        );
        let result = accept_grammar_token(&mut runtime, Some(table.as_slice()), &[], 2);
        if count == 1 {
            result.expect("one real parsed call must remain valid");
            assert!(runtime.as_ref().unwrap().is_accepted());
        } else {
            assert!(
                result.is_err(),
                "nonparallel grammar must not swallow protocol boundaries as raw argument text"
            );
        }
    }
}

fn compiled_accepts(schema: serde_json::Value, value: &str, parallel: bool) -> bool {
    compiled_accepts_wire(schema, value, parallel, true)
}

fn compiled_accepts_wire(
    schema: serde_json::Value,
    value: &str,
    parallel: bool,
    is_string: bool,
) -> bool {
    let registration = registry::find_for("DeepSeek-V4-Flash-0731").unwrap();
    let mut req = request();
    req.parallel_tool_calls = Some(parallel);
    req.tools.as_mut().unwrap()[0]
        .function
        .parameters
        .as_mut()
        .unwrap()["properties"]["path"] = schema;
    let grammar =
        compile_tool_grammar_for_request(&req, &ToolChoiceValue::Auto, Some(&registration), false)
            .expect("supported delimiter-safe schema")
            .unwrap();
    let call = if is_string {
        invoke(value)
    } else {
        invoke(value).replace("string=\"true\"", "string=\"false\"")
    };
    let body = format!("{call}\n</｜DSML｜tool_calls>");
    let table = Arc::new(vec![
        "<｜DSML｜tool_calls>\n".as_bytes().to_vec(),
        body.into_bytes(),
    ]);
    let params = SamplingParams {
        grammar: Some(grammar),
        token_bytes: Some(Arc::clone(&table)),
        grammar_kind: GrammarKind::ToolCallBodyAuto,
        tool_call_policy: ToolCallPolicy::AutoLazyGrammar,
        parallel_tool_calls: parallel,
        ..Default::default()
    };
    let mut runtime = grammar_runtime_for_request(&params, Some(&registration)).unwrap();
    accept_grammar_token(&mut runtime, Some(table.as_slice()), &[], 0).unwrap();
    accept_grammar_token(&mut runtime, Some(table.as_slice()), &[], 1).is_ok()
        && runtime.as_ref().unwrap().is_accepted()
}

#[test]
fn deepseek_raw_strings_preserve_code_and_reject_exact_reserved_delimiter() {
    for code in [
        "fmt::Formatter<'_>",
        "Vec::<T>",
        "a < b && b > c",
        "line one\nline two",
        r"C:\tmp\file",
        "</｜DSML｜paramete",
        "<<</x>",
    ] {
        assert!(
            compiled_accepts(serde_json::json!({"type":"string"}), code, false),
            "{code:?}"
        );
    }
    for code in [
        "</｜DSML｜parameter>",
        r"\</｜DSML｜parameter>",
        "before </｜DSML｜parameter> after",
    ] {
        assert!(
            !compiled_accepts(serde_json::json!({"type":"string"}), code, false),
            "{code:?}"
        );
    }
}

#[test]
fn deepseek_raw_length_counter_preserves_unicode_bounds_and_delimiter_exclusion() {
    let schema = serde_json::json!({"type":"string","minLength":2,"maxLength":4});
    for code in ["<>", "é<>", r"\x", "abcd"] {
        assert!(compiled_accepts(schema.clone(), code, false), "{code:?}");
    }
    for code in ["a", "abcde", r"\"] {
        assert!(!compiled_accepts(schema.clone(), code, false), "{code:?}");
    }
    assert!(!compiled_accepts(
        serde_json::json!({"type":"string","maxLength":24}),
        "</｜DSML｜parameter>",
        false
    ));
}

#[test]
fn deepseek_safe_enum_const_literals_and_parallel_calls_remain_supported() {
    assert!(compiled_accepts(
        serde_json::json!({"type":"string","enum":["fmt::Formatter<'_>"]}),
        "fmt::Formatter<'_>",
        false
    ));
    assert!(compiled_accepts(
        serde_json::json!({"const":"Vec::<T>"}),
        "Vec::<T>",
        false
    ));
    let registration = registry::find_for("DeepSeek-V4-Flash-0731").unwrap();
    let mut req = request();
    req.parallel_tool_calls = Some(true);
    let grammar =
        compile_tool_grammar_for_request(&req, &ToolChoiceValue::Auto, Some(&registration), false)
            .unwrap()
            .unwrap();
    let body = format!(
        "{}\n{}\n</｜DSML｜tool_calls>",
        invoke("src/ports.rs"),
        invoke("tests/ports.rs")
    );
    let table = Arc::new(vec![
        "<｜DSML｜tool_calls>\n".as_bytes().to_vec(),
        body.into_bytes(),
    ]);
    let params = SamplingParams {
        grammar: Some(grammar),
        token_bytes: Some(Arc::clone(&table)),
        grammar_kind: GrammarKind::ToolCallBodyAuto,
        tool_call_policy: ToolCallPolicy::AutoLazyGrammar,
        parallel_tool_calls: true,
        ..Default::default()
    };
    let mut runtime = grammar_runtime_for_request(&params, Some(&registration)).unwrap();
    for token in 0..2 {
        accept_grammar_token(&mut runtime, Some(table.as_slice()), &[], token).unwrap();
    }
    assert!(runtime.as_ref().unwrap().is_accepted());
}

#[test]
fn deepseek_unsafe_literals_uncomposed_patterns_and_resources_fail_closed() {
    let registration = registry::find_for("DeepSeek-V4-Flash-0731").unwrap();
    for schema in [
        serde_json::json!({"enum":["</｜DSML｜parameter>"]}),
        serde_json::json!({"const":"</｜DSML｜parameter>"}),
        serde_json::json!({"const":{"x":"</｜DSML｜parameter>"}}),
        serde_json::json!({"type":"string","pattern":".*"}),
        serde_json::json!({"type":"string","format":"not-a-known-format"}),
        serde_json::json!({"type":"string","maxLength":u64::MAX}),
    ] {
        let mut req = request();
        req.tools.as_mut().unwrap()[0]
            .function
            .parameters
            .as_mut()
            .unwrap()["properties"]["path"] = schema.clone();
        assert!(
            compile_tool_grammar_for_request(
                &req,
                &ToolChoiceValue::Auto,
                Some(&registration),
                false
            )
            .is_err(),
            "{schema}"
        );
    }
}

#[test]
fn deepseek_nested_json_strings_keys_and_unicode_bounds_remain_delimiter_safe() {
    let schema = serde_json::json!({"type":"object","properties":{
        "content":{"type":"string"}, "items":{"type":"array","items":{"type":"string"}}
    },"required":["content","items"],"additionalProperties":false});
    assert!(compiled_accepts_wire(
        schema.clone(),
        r#"{"content":"fmt::Formatter<'_>","items":["Vec::<T>"]}"#,
        false,
        false
    ));
    assert!(!compiled_accepts_wire(
        schema,
        r#"{"content":"</｜DSML｜parameter>","items":[]}"#,
        false,
        false
    ));
    for schema in [
        serde_json::json!({}),
        serde_json::json!(true),
        serde_json::json!({"type":"object"}),
    ] {
        assert!(compiled_accepts_wire(
            schema.clone(),
            r#"{"x":"Vec::<T>"}"#,
            false,
            false
        ));
        assert!(!compiled_accepts_wire(
            schema,
            r#"{"</｜DSML｜parameter>":"x"}"#,
            false,
            false
        ));
    }
    let open = serde_json::json!({"type":"object","properties":{"known":{"type":"string"}}});
    assert!(compiled_accepts_wire(
        open.clone(),
        r#"{"known":"fmt::<T>","other":"<"}"#,
        false,
        false
    ));
    assert!(compiled_accepts_wire(
        open.clone(),
        r#"{"\u003ckey":"value"}"#,
        false,
        false
    ));
    assert!(!compiled_accepts_wire(
        open,
        r#"{"bad</｜DSML｜parameter>":"x"}"#,
        false,
        false
    ));
    let bounded =
        serde_json::json!({"type":"array","items":{"type":"string","minLength":2,"maxLength":4}});
    assert!(compiled_accepts_wire(
        bounded.clone(),
        r#"["é<>","\u003c>"]"#,
        false,
        false
    ));
    assert!(!compiled_accepts_wire(
        bounded,
        r#"["abcde"]"#,
        false,
        false
    ));
}

#[test]
fn deepseek_safe_compiled_pattern_and_format_alphabets_preserve_constraints() {
    let identifier = serde_json::json!({"type":"string","pattern":"^[a-z][a-z0-9-]*$"});
    assert!(compiled_accepts(identifier.clone(), "port-name", false));
    assert!(!compiled_accepts(identifier, "Port_Name", false));
    let date = serde_json::json!({"type":"string","format":"date"});
    assert!(compiled_accepts(date.clone(), "2026-10-26", false));
    assert!(!compiled_accepts(date, "2026-19-26", false));
    assert!(compiled_accepts_wire(
        serde_json::json!({"type":"array","items":{
        "type":"string","pattern":"^CWE-[0-9]+$"}}),
        r#"["CWE-79"]"#,
        false,
        false
    ));
}

#[test]
fn deepseek_bounded_data_guard_survives_actual_three_tool_combiner() {
    for n in [4096, 65_536] {
        let schema=serde_json::json!({"type":"string","minLength":n,"maxLength":n});
        assert!(compiled_accepts_wire(schema.clone(),&"a".repeat(n),false,true));
        assert!(!compiled_accepts_wire(schema.clone(),&"a".repeat(n-1),false,true));
        assert!(!compiled_accepts_wire(schema,&"a".repeat(n+1),false,true));
    }
    let pair=serde_json::json!({"type":"string","minLength":2,"maxLength":2});
    assert!(compiled_accepts_wire(pair.clone(),r"\x",false,true));
    assert!(compiled_accepts_wire(pair,"😀é",false,true));
    assert!(!compiled_accepts_wire(serde_json::json!({"type":"string","maxLength":1}),r"\x",false,true));
    assert!(!compiled_accepts_wire(serde_json::json!({"type":"string","maxLength":64}),
        "x</｜DSML｜parameter>y",false,true));
}

#[test]
fn deepseek_literal_trailing_backslash_survives_other_raw_tool_and_union() {
    // The real three-tool request always includes generic raw write_file
    // content. A literal branch must not inherit that other branch's escape
    // completion constraint. PR402's literal emitter allowed these values.
    let value="x\\";
    for schema in [
        serde_json::json!({"type":"string","const":value}),
        serde_json::json!({"type":"string","enum":[value]}),
        serde_json::json!({"anyOf":[{"type":"string","const":value},{"type":"string","maxLength":8}]}),
    ] {
        assert!(compiled_accepts_wire(schema,value,false,true),
            "literal escape ending must not inherit another raw branch's end guard");
    }
}
