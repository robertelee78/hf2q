//! Issue #251: JSON Schema length/count bounds above the GBNF parser's
//! per-operator repetition limit compile exactly.

use super::super::parser::parse_generated;
use super::super::repetition::MAX_SCHEMA_REPETITION_BOUND;
use super::super::sampler::GrammarRuntime;
use super::schema_to_gbnf;
use serde_json::{json, Value};

fn accepts(schema: &Value, instance: &Value) -> bool {
    let source = schema_to_gbnf(schema).unwrap_or_else(|error| panic!("compile: {error}"));
    let grammar = parse_generated(&source).unwrap_or_else(|error| panic!("parse: {error}"));
    let root = grammar.rule_id("root").expect("root rule");
    let mut runtime = GrammarRuntime::new(grammar, root).expect("runtime");
    let bytes = serde_json::to_vec(instance).expect("serialize instance");
    runtime.accept_bytes(&bytes) && runtime.is_accepted()
}

fn text(length: usize) -> Value {
    Value::String("a".repeat(length))
}

fn integers(count: usize) -> Value {
    Value::Array(vec![json!(1); count])
}

#[test]
fn string_max_length_4096_is_exact() {
    let schema = json!({"type": "string", "maxLength": 4096});
    assert!(accepts(&schema, &text(0)));
    assert!(accepts(&schema, &text(4096)));
    assert!(!accepts(&schema, &text(4097)));
}

#[test]
fn string_length_boundaries_2000_and_2001_are_exact() {
    for bound in [2000usize, 2001] {
        let schema = json!({"type": "string", "minLength": bound, "maxLength": bound});
        assert!(accepts(&schema, &text(bound)), "rejects exactly {bound}");
        assert!(!accepts(&schema, &text(bound - 1)), "accepts {}", bound - 1);
        assert!(!accepts(&schema, &text(bound + 1)), "accepts {}", bound + 1);
    }
    let schema = json!({"type": "string", "minLength": 2001});
    assert!(!accepts(&schema, &text(2000)));
    assert!(accepts(&schema, &text(2001)));
    assert!(accepts(&schema, &text(4500)));
}

#[test]
fn array_items_max_length_4096_is_exact() {
    let schema = json!({
        "type": "object",
        "properties": {"argv": {"type": "array", "items": {"type": "string", "maxLength": 4096}}},
        "required": ["argv"]
    });
    assert!(accepts(&schema, &json!({"argv": []})));
    assert!(accepts(
        &schema,
        &json!({"argv": [text(4096), "ls", text(4096)]})
    ));
    assert!(!accepts(&schema, &json!({"argv": ["ls", text(4097)]})));
}

#[test]
fn array_item_counts_above_2000_are_exact() {
    let schema =
        json!({"type": "array", "items": {"type": "integer"}, "minItems": 2001, "maxItems": 2500});
    assert!(!accepts(&schema, &integers(2000)));
    assert!(accepts(&schema, &integers(2001)));
    assert!(accepts(&schema, &integers(2500)));
    assert!(!accepts(&schema, &integers(2501)));

    for bound in [2000usize, 2001] {
        let schema = json!({"type": "array", "items": {"type": "integer"}, "maxItems": bound});
        assert!(accepts(&schema, &integers(0)));
        assert!(
            accepts(&schema, &integers(bound)),
            "rejects exactly {bound}"
        );
        assert!(
            !accepts(&schema, &integers(bound + 1)),
            "accepts {}",
            bound + 1
        );
    }
}

#[test]
fn nested_bounded_strings_in_bounded_arrays_compile() {
    // Before #251 the inline `char{0,1000}` inside the comma group made the
    // parser's n_prev_rules * total_rules check fail for maxItems >= 3.
    let schema = json!({
        "type": "array",
        "items": {"type": "string", "maxLength": 1000},
        "maxItems": 3
    });
    assert!(accepts(
        &schema,
        &json!([text(1000), text(1000), text(1000)])
    ));
    assert!(!accepts(
        &schema,
        &json!([text(1000), text(1000), text(1000), "a"])
    ));
    assert!(!accepts(&schema, &json!([text(1001)])));
}

#[test]
fn property_counts_above_2000_are_exact() {
    let schema = json!({"type": "object", "additionalProperties": {"type": "integer"}, "maxProperties": 2001});
    let object = |count: usize| {
        Value::Object(
            (0..count)
                .map(|index| (format!("k{index}"), json!(index)))
                .collect(),
        )
    };
    assert!(accepts(&schema, &object(2001)));
    assert!(!accepts(&schema, &object(2002)));
}

#[test]
fn repetition_ceiling_still_fails_closed() {
    let at_ceiling = json!({"type": "string", "maxLength": MAX_SCHEMA_REPETITION_BOUND});
    let source = schema_to_gbnf(&at_ceiling).expect("ceiling compiles");
    parse_generated(&source).expect("ceiling parses");

    for keyword in ["minLength", "maxLength"] {
        let schema = json!({"type": "string", keyword: MAX_SCHEMA_REPETITION_BOUND + 1});
        let error = schema_to_gbnf(&schema).expect_err("above ceiling must fail closed");
        let message = error.to_string();
        assert!(message.contains(&format!("/{keyword}")), "{message}");
        assert!(
            message.contains("exceeds the supported repetition ceiling 65536"),
            "{message}"
        );
    }
    let schema = json!({"type": "array", "maxItems": MAX_SCHEMA_REPETITION_BOUND + 1});
    assert!(schema_to_gbnf(&schema).is_err());
}

#[test]
fn counted_object_grammar_keeps_its_bound() {
    let schema = json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "maxProperties": 2001
    });
    let error = schema_to_gbnf(&schema).expect_err("counted object above 2000 fails closed");
    assert!(error.to_string().contains("above 2000"), "{error}");
}
