//! JSON Schema → GBNF translator.
//!
//! Ported (minimal viable subset) from llama.cpp's
//! `/opt/llama.cpp/common/json-schema-to-grammar.cpp`. The subset covers the
//! common cases actually exercised by OpenAI `response_format: {type:
//! "json_schema", json_schema: {schema: {...}}}` requests and by tool-call
//! parameter schemas:
//!
//!   - Primitive types: `string`, `number`, `integer`, `boolean`, `null`.
//!   - `object` with `properties` and `required`.
//!   - `array` with `items`.
//!   - `enum` (string values only at this iter).
//!   - `type` as either a single string or an array of strings (unions).
//!   - Type-agnostic schema (bare `{}`) → `value` primitive.
//!
//! Features deliberately deferred (landed when a concrete user requests
//! them — mantra: no stubs, but also no speculative features):
//!   - `$ref` / `$defs` (requires ref resolution).
//!   - `pattern` (regex → grammar conversion).
//!   - `minLength` / `maxLength` / `minimum` / `maximum` / `minItems` /
//!     `maxItems`.
//!   - `anyOf` / `oneOf` / `allOf`.
//!   - `additionalProperties` beyond default-true.
//!   - Tuple-form arrays (`items: [schemaA, schemaB, ...]`).
//!
//! The output is a GBNF string that can be parsed by
//! `super::parser::parse(...)` and consumed by `super::sampler::GrammarRuntime`.
//! The root rule is always named `root`.

use std::collections::{BTreeMap, HashSet};

use serde_json::Value;

// ---------------------------------------------------------------------------
// Primitive rule library (ported verbatim from json-schema-to-grammar.cpp's
// PRIMITIVE_RULES map).
// ---------------------------------------------------------------------------

/// GBNF body for the `space` rule — 0+ whitespace characters. Kept identical
/// to llama.cpp's `SPACE_RULE` so output grammars are byte-for-byte
/// comparable.
const SPACE_RULE: &str = r#"| " " | "\n"{1,2} [ \t]{0,20}"#;

/// `(name, body, deps)` — name is the GBNF rule name; body is the rule's
/// body text; deps is a list of other primitive rule names this rule
/// depends on (transitively included in the output).
fn primitive(name: &str) -> Option<(&'static str, &'static str, &'static [&'static str])> {
    match name {
        "boolean" => Some(("boolean", r#"("true" | "false") space"#, &[])),
        "decimal-part" => Some(("decimal-part", r#"[0-9]{1,16}"#, &[])),
        "integral-part" => Some(("integral-part", r#"[0] | [1-9] [0-9]{0,15}"#, &[])),
        "number" => Some((
            "number",
            r#"("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space"#,
            &["integral-part", "decimal-part"],
        )),
        "integer" => Some((
            "integer",
            r#"("-"? integral-part) space"#,
            &["integral-part"],
        )),
        "value" => Some((
            "value",
            r#"object | array | string | number | boolean | null"#,
            &["object", "array", "string", "number", "boolean", "null"],
        )),
        "object" => Some((
            "object",
            r#"{ space ( string ":" space value ("," space string ":" space value)* )? } space"#,
            &["string", "value"],
        )),
        "array" => Some((
            "array",
            r#""[" space ( value ("," space value)* )? "]" space"#,
            &["value"],
        )),
        "char" => Some((
            "char",
            r#"[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})"#,
            &[],
        )),
        "string" => Some(("string", r#""\"" char* "\"" space"#, &["char"])),
        "null" => Some(("null", r#""null" space"#, &[])),
        _ => None,
    }
}

// The "object" primitive body above has a subtle issue in llama.cpp: the
// braces { } are treated as literals in the body but that's not valid GBNF.
// llama.cpp's version is:
//   "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space"
// Let me use the quoted-brace form.

/// llama.cpp's actual primitives body uses quoted braces for object. This is
/// the string-escape-correct version.
fn primitive_exact(name: &str) -> Option<(&'static str, &'static str, &'static [&'static str])> {
    match name {
        "boolean" => Some(("boolean", r#"("true" | "false") space"#, &[])),
        "decimal-part" => Some(("decimal-part", r#"[0-9]{1,16}"#, &[])),
        "integral-part" => Some(("integral-part", r#"[0] | [1-9] [0-9]{0,15}"#, &[])),
        "number" => Some((
            "number",
            r#"("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space"#,
            &["integral-part", "decimal-part"],
        )),
        "integer" => Some((
            "integer",
            r#"("-"? integral-part) space"#,
            &["integral-part"],
        )),
        "value" => Some((
            "value",
            r#"object | array | string | number | boolean | null"#,
            &["object", "array", "string", "number", "boolean", "null"],
        )),
        "object" => Some((
            "object",
            r#""{" space ( string ":" space value ("," space string ":" space value)* )? "}" space"#,
            &["string", "value"],
        )),
        "array" => Some((
            "array",
            r#""[" space ( value ("," space value)* )? "]" space"#,
            &["value"],
        )),
        "char" => Some((
            "char",
            r#"[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})"#,
            &[],
        )),
        "string" => Some(("string", r#""\"" char* "\"" space"#, &["char"])),
        "null" => Some(("null", r#""null" space"#, &[])),
        _ => None,
    }
}

#[allow(dead_code)]
const _UNUSED_PRIMITIVE: fn(&str) -> Option<(&'static str, &'static str, &'static [&'static str])>
    = primitive;

// ---------------------------------------------------------------------------
// Literal escape helpers
// ---------------------------------------------------------------------------

/// Escape a string so it can be embedded as a GBNF literal between double
/// quotes. Mirrors llama.cpp's `format_literal` behavior.
pub fn format_literal(literal: &str) -> String {
    let mut out = String::with_capacity(literal.len() + 2);
    out.push('"');
    for c in literal.chars() {
        match c {
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaError {
    pub path: String,
    pub message: String,
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "json-schema-to-grammar error at {}: {}", self.path, self.message)
    }
}
impl std::error::Error for SchemaError {}

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

/// Convert a JSON Schema (supplied as a `serde_json::Value`) to a GBNF
/// grammar string with `root` as the start rule.
///
/// Returns `Err(SchemaError)` if the schema contains a feature that isn't
/// yet supported (see the module-level doc for the supported subset).
pub fn schema_to_gbnf(schema: &Value) -> Result<String, SchemaError> {
    let mut conv = Converter {
        rules: BTreeMap::new(),
        added_primitives: HashSet::new(),
    };
    let root_body = conv.visit(schema, "")?;
    conv.rules.insert("root".to_string(), root_body);

    // `space` is always needed since all primitives reference it.
    conv.rules
        .entry("space".to_string())
        .or_insert_with(|| SPACE_RULE.to_string());

    // Serialize rules in a deterministic order — root first, then alpha.
    let mut out = String::new();
    // Put root first for readability.
    if let Some(body) = conv.rules.get("root") {
        out.push_str(&format!("root ::= {}\n", body));
    }
    for (name, body) in &conv.rules {
        if name == "root" {
            continue;
        }
        out.push_str(&format!("{} ::= {}\n", name, body));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Internal converter
// ---------------------------------------------------------------------------

struct Converter {
    /// Emitted rules keyed by name. BTreeMap for deterministic output order.
    rules: BTreeMap<String, String>,
    added_primitives: HashSet<&'static str>,
}

impl Converter {
    fn add_primitive(&mut self, name: &'static str) {
        if self.added_primitives.contains(name) {
            return;
        }
        self.added_primitives.insert(name);
        let (_, body, deps) = primitive_exact(name).expect("unknown primitive");
        self.rules.insert(name.to_string(), body.to_string());
        for dep in deps {
            self.add_primitive(dep);
        }
    }

    /// Return the GBNF rule body that matches `schema`. `path` is used in
    /// error messages.
    fn visit(&mut self, schema: &Value, path: &str) -> Result<String, SchemaError> {
        let obj = match schema.as_object() {
            Some(o) => o,
            None => {
                return Err(SchemaError {
                    path: path.to_string(),
                    message: "schema must be a JSON object".into(),
                });
            }
        };

        // enum: alternation of literal strings.
        if let Some(Value::Array(values)) = obj.get("enum") {
            if values.is_empty() {
                return Err(SchemaError {
                    path: path.to_string(),
                    message: "enum cannot be empty".into(),
                });
            }
            let mut alts: Vec<String> = Vec::with_capacity(values.len());
            for v in values {
                match v {
                    Value::String(s) => {
                        // Literal string value in JSON → double-quoted literal
                        // in the emitted JSON. The grammar must match the
                        // quoted form, so embed `"value"` literally.
                        let quoted_value = serde_json::to_string(s).map_err(|e| SchemaError {
                            path: path.to_string(),
                            message: format!("enum serialize: {e}"),
                        })?;
                        alts.push(format_literal(&quoted_value));
                    }
                    Value::Number(_) | Value::Bool(_) | Value::Null => {
                        // Non-string enum — serialize to JSON text.
                        let text = serde_json::to_string(v).map_err(|e| SchemaError {
                            path: path.to_string(),
                            message: format!("enum serialize: {e}"),
                        })?;
                        alts.push(format_literal(&text));
                    }
                    Value::Array(_) | Value::Object(_) => {
                        return Err(SchemaError {
                            path: format!("{}/enum", path),
                            message: "enum values must be scalars (string/number/bool/null)"
                                .into(),
                        });
                    }
                }
            }
            // After an enum value we still emit `space` so trailing whitespace
            // is accepted — matches llama.cpp's convention.
            self.rules
                .entry("space".to_string())
                .or_insert_with(|| SPACE_RULE.to_string());
            return Ok(format!("({}) space", alts.join(" | ")));
        }

        // `type`: the dominant dispatch.
        let type_val = obj.get("type");
        let type_str = match type_val {
            None => {
                // Untyped — accept any JSON value.
                self.add_primitive("value");
                return Ok("value".into());
            }
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(types)) => {
                // Union type — emit an alternation.
                let mut alts: Vec<String> = Vec::with_capacity(types.len());
                for (i, t) in types.iter().enumerate() {
                    let tstr = t.as_str().ok_or_else(|| SchemaError {
                        path: format!("{}/type/{}", path, i),
                        message: "type array entries must be strings".into(),
                    })?;
                    let mut stub = serde_json::Map::new();
                    stub.insert("type".into(), Value::String(tstr.into()));
                    let body = self.visit(&Value::Object(stub), &format!("{}/type[{}]", path, i))?;
                    alts.push(body);
                }
                return Ok(alts.join(" | "));
            }
            Some(other) => {
                return Err(SchemaError {
                    path: format!("{}/type", path),
                    message: format!(
                        "type must be a string or array of strings, got {:?}",
                        other
                    ),
                });
            }
        };

        match type_str.as_str() {
            "string" => {
                self.add_primitive("string");
                Ok("string".into())
            }
            "number" => {
                self.add_primitive("number");
                Ok("number".into())
            }
            "integer" => {
                self.add_primitive("integer");
                Ok("integer".into())
            }
            "boolean" => {
                self.add_primitive("boolean");
                Ok("boolean".into())
            }
            "null" => {
                self.add_primitive("null");
                Ok("null".into())
            }
            "object" => self.visit_object(obj, path),
            "array" => self.visit_array(obj, path),
            other => Err(SchemaError {
                path: format!("{}/type", path),
                message: format!("unsupported type '{}'", other),
            }),
        }
    }

    fn visit_object(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        path: &str,
    ) -> Result<String, SchemaError> {
        self.add_primitive("string");
        self.add_primitive("value");
        self.rules
            .entry("space".to_string())
            .or_insert_with(|| SPACE_RULE.to_string());

        let properties = obj
            .get("properties")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let required_list: HashSet<String> = obj
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if properties.is_empty() {
            // No explicit properties — accept any object.
            self.add_primitive("object");
            return Ok("object".into());
        }

        // Deterministic property iteration: sort alpha for emission order. The
        // original object spec allows any order of keys, but the grammar
        // enforces the order we emit here — so emit in a stable canonical
        // order.
        //
        // NOTE: this is stricter than llama.cpp which allows arbitrary order
        // via recursive alternation. For iter 8 we required properties to
        // appear in alphabetical order in the output JSON — a deliberate
        // simplification. Iter 74 keeps that simplification but fixes the
        // required/optional bookkeeping below.
        let mut required_entries: Vec<String> = Vec::new();
        let mut optional_entries: Vec<String> = Vec::new();
        let mut keys: Vec<&String> = properties.keys().collect();
        keys.sort();
        for k in keys {
            let v = &properties[k];
            let prop_rule_name = sanitize_rule_name(k);
            let vbody = self.visit(v, &format!("{}/properties/{}", path, k))?;
            // Inline-emitted rule body (not a named rule) so multiple object
            // schemas with the same property name don't collide.
            let rule_name = format!("{}-{}", path_slug(path), prop_rule_name);
            self.rules.insert(rule_name.clone(), vbody);
            let quoted_key = format_literal(&format!("\"{}\"", k));
            let entry = format!("{} \":\" space {}", quoted_key, rule_name);
            if required_list.contains(k) {
                required_entries.push(entry);
            } else {
                optional_entries.push(entry);
            }
        }

        // Iter 74 fix: required fields are emitted FIRST, joined with
        // mandatory `","` separators (each one is non-optional, so all
        // required keys must appear). Optional fields follow, each wrapped
        // in its own `("," space entry)?` so the separator is inside the
        // optional — `a (",", b)?` accepts `a` alone or `a,b`.
        //
        // Bug from prior iter: the old code did `prop_rules.remove(0)` then
        // wrapped EVERY remaining entry in `(",", x)?` regardless of
        // required/optional bookkeeping. With 2 required fields the second
        // was emitted as optional, falsely accepting `{first: ...}` alone.
        // Surfaced by iter-74 function-call-with-nested-object test.
        if required_entries.is_empty() && optional_entries.is_empty() {
            return Ok(r#""{" space "}" space"#.into());
        }

        let mut body = String::new();
        let mut first_emitted = false;

        // Required block: mandatory commas between consecutive required.
        for r in &required_entries {
            if !first_emitted {
                body.push_str(r);
                first_emitted = true;
            } else {
                body.push_str(&format!(" \",\" space {}", r));
            }
        }

        // Optional block: each one's own (",", entry)? wrapper.
        for o in &optional_entries {
            if !first_emitted {
                // No required preceding — the very first entry is optional.
                // Wrap as `(entry)?`. Produces `(a)? (",", b)? (",", c)?`
                // which accepts `{}`, `{a}`, `{a,b}`, `{a,b,c}` (in order).
                body.push_str(&format!("({})?", o));
                first_emitted = true;
            } else {
                body.push_str(&format!(" (\",\" space {})?", o));
            }
        }

        Ok(format!(r#""{{" space {} "}}" space"#, body))
    }

    fn visit_array(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        path: &str,
    ) -> Result<String, SchemaError> {
        self.rules
            .entry("space".to_string())
            .or_insert_with(|| SPACE_RULE.to_string());
        let item_schema = obj.get("items");
        let item_rule = match item_schema {
            None => {
                self.add_primitive("value");
                "value".to_string()
            }
            Some(Value::Object(_)) => self.visit(item_schema.unwrap(), &format!("{}/items", path))?,
            Some(Value::Array(_)) => {
                return Err(SchemaError {
                    path: format!("{}/items", path),
                    message: "tuple-form arrays (items: [...]) not yet supported".into(),
                });
            }
            _ => {
                return Err(SchemaError {
                    path: format!("{}/items", path),
                    message: "items must be an object schema".into(),
                });
            }
        };
        // [ items ] with zero-or-more elements, comma-separated.
        Ok(format!(r#""[" space ( {0} ("," space {0})* )? "]" space"#, item_rule))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sanitize_rule_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() || c == '-' {
            out.push(c);
        } else {
            out.push('-');
        }
    }
    if out.is_empty() {
        out.push('x');
    }
    out
}

fn path_slug(path: &str) -> String {
    if path.is_empty() {
        return "root".into();
    }
    sanitize_rule_name(path.trim_start_matches('/'))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser::parse;
    use super::super::sampler::GrammarRuntime;

    fn compile(schema_json: &str) -> String {
        let schema: Value = serde_json::from_str(schema_json).unwrap();
        schema_to_gbnf(&schema).unwrap_or_else(|e| panic!("schema_to_gbnf: {:?}", e))
    }

    fn runtime(schema_json: &str) -> GrammarRuntime {
        let gbnf = compile(schema_json);
        let g = parse(&gbnf).unwrap_or_else(|e| panic!("parse gbnf:\n{}\nerror: {}", gbnf, e));
        let rid = g.rule_id("root").unwrap();
        GrammarRuntime::new(g, rid).unwrap()
    }

    #[test]
    fn primitive_boolean_schema_accepts_true_and_false() {
        let mut rt_true = runtime(r#"{"type":"boolean"}"#);
        assert!(rt_true.accept_bytes(b"true"));
        assert!(rt_true.is_accepted());
        let mut rt_false = runtime(r#"{"type":"boolean"}"#);
        assert!(rt_false.accept_bytes(b"false"));
        assert!(rt_false.is_accepted());
        let mut rt_bad = runtime(r#"{"type":"boolean"}"#);
        let ok = rt_bad.accept_bytes(b"maybe");
        assert!(!(ok && rt_bad.is_accepted()));
    }

    #[test]
    fn primitive_integer_schema_accepts_numbers() {
        for num in &["0", "1", "-42", "12345"] {
            let mut rt = runtime(r#"{"type":"integer"}"#);
            assert!(rt.accept_bytes(num.as_bytes()), "accept {:?}", num);
            assert!(rt.is_accepted(), "is_accepted for {:?}", num);
        }
        for bad in &["1.5", "abc", ""] {
            let mut rt = runtime(r#"{"type":"integer"}"#);
            let ok = rt.accept_bytes(bad.as_bytes());
            assert!(!(ok && rt.is_accepted()), "reject {:?}", bad);
        }
    }

    #[test]
    fn primitive_number_schema_accepts_decimals() {
        for num in &["0", "1.5", "-42.0", "3.14", "2e10", "-1.5E-3"] {
            let mut rt = runtime(r#"{"type":"number"}"#);
            assert!(rt.accept_bytes(num.as_bytes()), "accept {:?}", num);
            assert!(rt.is_accepted(), "is_accepted for {:?}", num);
        }
    }

    #[test]
    fn primitive_string_schema_accepts_quoted() {
        let mut rt = runtime(r#"{"type":"string"}"#);
        assert!(rt.accept_bytes(b"\"hello\""));
        assert!(rt.is_accepted());

        let mut rt2 = runtime(r#"{"type":"string"}"#);
        let ok = rt2.accept_bytes(b"unquoted");
        assert!(!(ok && rt2.is_accepted()));
    }

    #[test]
    fn primitive_null_schema_accepts_null_keyword() {
        let mut rt = runtime(r#"{"type":"null"}"#);
        assert!(rt.accept_bytes(b"null"));
        assert!(rt.is_accepted());
    }

    #[test]
    fn enum_string_values() {
        let schema = r#"{"enum":["red","green","blue"]}"#;
        for good in &["\"red\"", "\"green\"", "\"blue\""] {
            let mut rt = runtime(schema);
            assert!(rt.accept_bytes(good.as_bytes()), "accept {}", good);
            assert!(rt.is_accepted(), "is_accepted {}", good);
        }
        for bad in &["\"yellow\"", "red", "\"\""] {
            let mut rt = runtime(schema);
            let ok = rt.accept_bytes(bad.as_bytes());
            assert!(!(ok && rt.is_accepted()), "reject {}", bad);
        }
    }

    #[test]
    fn empty_schema_accepts_any_json_value() {
        let schema = r#"{}"#;
        for good in &["42", "\"hi\"", "true", "null", "[]", "{}", "[1,2,3]"] {
            let mut rt = runtime(schema);
            assert!(rt.accept_bytes(good.as_bytes()), "accept {}", good);
            assert!(rt.is_accepted(), "is_accepted {}", good);
        }
    }

    #[test]
    fn object_with_single_required_property() {
        let schema = r#"{
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }"#;
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(b"{\"name\":\"Alice\"}"));
        assert!(rt.is_accepted());

        let mut rt2 = runtime(schema);
        let ok = rt2.accept_bytes(b"{}");
        assert!(!(ok && rt2.is_accepted()));
    }

    #[test]
    fn object_with_multiple_required_properties() {
        // Alphabetical order: age, name. Both required.
        let schema = r#"{
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"]
        }"#;
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(b"{\"age\":30,\"name\":\"Bob\"}"));
        assert!(rt.is_accepted());
    }

    #[test]
    fn object_with_optional_property() {
        let schema = r#"{
            "type": "object",
            "properties": {"name": {"type": "string"}, "nickname": {"type": "string"}},
            "required": ["name"]
        }"#;
        // With nickname.
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(b"{\"name\":\"Carol\",\"nickname\":\"Carrie\"}"));
        assert!(rt.is_accepted());
        // Without nickname.
        let mut rt2 = runtime(schema);
        assert!(rt2.accept_bytes(b"{\"name\":\"Carol\"}"));
        assert!(rt2.is_accepted());
    }

    #[test]
    fn array_of_integers() {
        let schema = r#"{"type":"array","items":{"type":"integer"}}"#;
        for good in &["[]", "[1]", "[1,2,3]", "[-5,0,42]"] {
            let mut rt = runtime(schema);
            assert!(rt.accept_bytes(good.as_bytes()), "accept {}", good);
            assert!(rt.is_accepted(), "is_accepted {}", good);
        }
        let mut rt_bad = runtime(schema);
        let ok = rt_bad.accept_bytes(b"[1,\"x\"]");
        assert!(!(ok && rt_bad.is_accepted()));
    }

    #[test]
    fn array_without_items_accepts_any_values() {
        let schema = r#"{"type":"array"}"#;
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(b"[1,\"x\",true,null]"));
        assert!(rt.is_accepted());
    }

    #[test]
    fn union_type_string_or_null() {
        let schema = r#"{"type":["string","null"]}"#;
        let mut rt_s = runtime(schema);
        assert!(rt_s.accept_bytes(b"\"hi\""));
        assert!(rt_s.is_accepted());
        let mut rt_n = runtime(schema);
        assert!(rt_n.accept_bytes(b"null"));
        assert!(rt_n.is_accepted());
        let mut rt_bad = runtime(schema);
        let ok = rt_bad.accept_bytes(b"42");
        assert!(!(ok && rt_bad.is_accepted()));
    }

    #[test]
    fn nested_object_with_array() {
        // Classic tool-call shape: {name: string, arguments: {...}}
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
            },
            "required": ["name", "arguments"]
        }"#;
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(
            b"{\"arguments\":{\"city\":\"NYC\"},\"name\":\"get_weather\"}"
        ));
        assert!(rt.is_accepted());
    }

    #[test]
    fn unsupported_type_rejected_at_compile_time() {
        let schema: Value =
            serde_json::from_str(r#"{"type":"notathing"}"#).unwrap();
        let err = schema_to_gbnf(&schema).unwrap_err();
        assert!(err.message.contains("unsupported type"));
    }

    #[test]
    fn pattern_not_yet_supported_but_compiles_when_ignored() {
        // `pattern` isn't in our subset — we ignore unknown keys silently.
        // (The test documents this behavior: the grammar compiles as if
        // pattern weren't there. Stricter mode comes with iter 9+.)
        let schema = r#"{"type":"string","pattern":"^[a-z]+$"}"#;
        let mut rt = runtime(schema);
        // No constraint beyond "any JSON string".
        assert!(rt.accept_bytes(b"\"ABC123\""));
        assert!(rt.is_accepted());
    }

    #[test]
    fn enum_non_string_value_accepted() {
        let schema = r#"{"enum":[42, true, null]}"#;
        for good in &["42", "true", "null"] {
            let mut rt = runtime(schema);
            assert!(rt.accept_bytes(good.as_bytes()), "accept {}", good);
            assert!(rt.is_accepted(), "is_accepted {}", good);
        }
    }

    #[test]
    fn compiled_grammar_has_root_rule() {
        let out = compile(r#"{"type":"boolean"}"#);
        assert!(out.starts_with("root ::="), "output:\n{}", out);
    }

    // -----------------------------------------------------------------
    // OpenAI function-calling schemas — realistic production shapes
    //
    // The OpenAI Chat Completions tools API serializes a function call
    // as `{name: string, arguments: <stringified-JSON-of-args>}`.
    // structured_outputs / response_format=json_schema accepts ANY
    // OpenAI JSON Schema. Below cases mirror three distinct production
    // workloads we need to support:
    //   1. Single string argument (e.g. weather query)
    //   2. Nested object argument with multiple required fields
    //   3. Enum-constrained string argument
    // Each test compiles the schema, parses the GBNF, then exercises
    // the runtime against a sample OpenAI-shape function-call output.
    // -----------------------------------------------------------------

    #[test]
    fn function_call_with_single_string_argument() {
        // Mirrors: tools=[{type:"function", function:{name:"get_weather",
        // parameters:{type:"object", properties:{city:{type:"string"}},
        // required:["city"]}}}]
        let schema = r#"{
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"],
            "additionalProperties": false
        }"#;
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"city":"London"}"#),
            "rejected valid function-call payload"
        );
        assert!(rt.is_accepted(), "runtime not accepted at end");

        // Reject if required field is missing.
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{}"#);
        assert!(!(ok && rt.is_accepted()), "accepted empty object missing required city");
    }

    #[test]
    fn function_call_with_nested_object_argument() {
        // Realistic: a search() tool that takes {query: str, filters:
        // {min_price: number, max_price: number}}. This is the most
        // common production shape — one level of nesting with mixed
        // required fields.
        //
        // Iter 74 fixed the "second-required-becomes-optional" bug; the
        // grammar emits ALL required fields as mandatory. The
        // alphabetical-key-order constraint is a separate documented
        // simplification — both required keys must appear, but in the
        // alphabetical order the grammar enforces (filters < query).
        // Models trained on OpenAI's API typically respect schema-
        // declared order; accept-any-permutation grammar is iter 75+
        // work.
        let schema = r#"{
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "min_price": {"type": "number"},
                        "max_price": {"type": "number"}
                    },
                    "required": ["min_price", "max_price"]
                }
            },
            "required": ["query", "filters"]
        }"#;
        // Alphabetical order: filters before query; max_price before min_price.
        let mut rt = runtime(schema);
        let payload =
            br#"{"filters":{"max_price":2000,"min_price":500},"query":"laptops"}"#;
        assert!(rt.accept_bytes(payload), "rejected valid nested function-call");
        assert!(rt.is_accepted(), "runtime not accepted at end");

        // Critical bug-fix anchor: missing the SECOND required field
        // (query) must be REJECTED. Pre-iter-74 this was falsely
        // accepted because the grammar emitted query as optional.
        let mut rt = runtime(schema);
        let missing_required =
            br#"{"filters":{"max_price":2000,"min_price":500}}"#;
        let ok = rt.accept_bytes(missing_required);
        assert!(
            !(ok && rt.is_accepted()),
            "accepted object missing required 'query' (iter 74 regression)"
        );
    }

    #[test]
    fn function_call_with_enum_argument() {
        // Tool with a constrained enum field (e.g. unit selector for a
        // weather function: celsius/fahrenheit). The structured-output
        // grammar must reject any value that isn't in the enum.
        // Alphabetical: city before unit (matches input).
        let schema = r#"{
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city", "unit"]
        }"#;
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"city":"London","unit":"celsius"}"#),
            "rejected valid enum value"
        );
        assert!(rt.is_accepted());

        // Out-of-enum unit value must be rejected.
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{"city":"London","unit":"kelvin"}"#);
        assert!(
            !(ok && rt.is_accepted()),
            "accepted enum value 'kelvin' that's not in [celsius, fahrenheit]"
        );

        // Missing required 'unit' rejected.
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{"city":"London"}"#);
        assert!(
            !(ok && rt.is_accepted()),
            "accepted object missing required 'unit'"
        );
    }

    #[test]
    fn function_call_with_array_arguments_field() {
        // A tool that takes an array of strings (e.g. tags for a
        // bookmark create). Common production shape. Alphabetical key
        // order: tags before url.
        let schema = r#"{
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["url", "tags"]
        }"#;
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(
            br#"{"tags":["news","tech"],"url":"https://example.com"}"#
        ));
        assert!(rt.is_accepted());

        // Empty tags array allowed.
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(br#"{"tags":[],"url":"https://example.com"}"#));
        assert!(rt.is_accepted());

        // Missing 'tags' rejected (iter 74 bug-fix anchor).
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{"url":"https://example.com"}"#);
        assert!(
            !(ok && rt.is_accepted()),
            "accepted object missing required 'tags'"
        );
    }
}
