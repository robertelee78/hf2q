//! Native tool catalogues are prompt bytes: preserve insertion order recursively.
use super::*;
#[test]
fn tool_catalogue_preserves_nested_request_key_order() {
    let raw = r#"{"type":"function","function":{"name":"write_file","description":"Edit fixture","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"],"additionalProperties":false}}}"#;
    let tool: super::super::schema::Tool = serde_json::from_str(raw).unwrap();
    let actual = render_chat_prompt_with_tools_generation_prompt(
        "{% for tool in tools %}{{ tool | tojson }}{% endfor %}",
        &[],
        Some(&[tool]),
        false,
        None,
        true,
    )
    .unwrap();
    assert_eq!(
        actual,
        r#"{"type": "function", "function": {"name": "write_file", "description": "Edit fixture", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": false}}}"#
    );
}
#[test]
fn explicit_template_sort_still_controls_order() {
    let arguments: serde_json::Value = serde_json::from_str(r#"{"z":1,"a":2}"#).unwrap();
    let mut env = crate::serve::build_chat_template_env(crate::serve::RaisePolicy::Strict);
    env.add_template(
        "test",
        "{% for key, value in arguments | dictsort %}{{ key }}={{ value }};{% endfor %}",
    )
    .unwrap();
    assert_eq!(
        env.get_template("test")
            .unwrap()
            .render(minijinja::context! { arguments => arguments })
            .unwrap(),
        "a=2;z=1;"
    );
}
