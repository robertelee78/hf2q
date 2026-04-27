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
//!   - `additionalProperties: {schema}` (schema-typed additional props).
//!   - Tuple-form arrays (`items: [schemaA, schemaB, ...]`).
//!
//! `additionalProperties: false` IS enforced (iter 75): the grammar rejects
//! any key not declared in `properties`. `additionalProperties: true` or
//! unset (the default per JSON Schema spec) allows extra keys permissively.
//! `additionalProperties: {schema}` is deferred (treated as permissive).
//!
//! Object key order (iter 75): grammar accepts keys in ANY order. Previously
//! keys were required alphabetically (a deliberate simplification in iter 8
//! that was never a feature — just a coincidence of BTreeMap iteration order).
//! The new algorithm generates O(2^N) permutation sub-rules for N required
//! keys and O(N^2) optional-chain sub-rules for optional keys.
//!
//! The output is a GBNF string that can be parsed by
//! `super::parser::parse(...)` and consumed by `super::sampler::GrammarRuntime`.
//! The root rule is always named `root`.

use std::collections::{BTreeMap, HashMap, HashSet};

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

        // additionalProperties handling (iter 75):
        //   - unset or true  → permissive: accept any extra keys (JSON Schema
        //     default). The grammar allows unknown kv-pairs via a wildcard
        //     `string ":" space value` rule in the optional chain.
        //   - false          → closed: grammar rejects keys not in properties.
        //     Implemented by omitting the wildcard rule from the optional
        //     chain — only declared property keys can appear, so extra keys
        //     cause the grammar stack to die.
        //   - {schema}       → deferred (module docstring). Treated as
        //     permissive for now.
        let additional_props = obj.get("additionalProperties");
        let additional_closed = matches!(additional_props, Some(Value::Bool(false)));

        if properties.is_empty() {
            if additional_closed {
                // additionalProperties:false + no declared properties means
                // only the empty object {} is valid.
                return Ok(r#""{" space "}" space"#.into());
            }
            // No explicit properties — accept any object.
            self.add_primitive("object");
            return Ok("object".into());
        }

        // ---------------------------------------------------------------
        // Build per-property kv rules.
        //
        // Each kv rule is named `<slug>-<prop>-kv` and captures:
        //   "\"key\"" ":" space VALUE_RULE
        //
        // Keys are sorted here for deterministic rule-name generation only.
        // The grammar itself accepts them in ANY order (iter 75 fix).
        //
        // WHY alphabetical-only was the original choice (Chesterton note):
        // iter 8 took the simplest subset — BTreeMap::iter() produces
        // sorted order, so "emit in iteration order" silently became
        // "enforce alphabetical key order in JSON output". The comment
        // said "deliberate simplification" but the consequence was that
        // any model emitting non-alphabetical keys had valid JSON rejected
        // by the grammar mask. This was never a feature — it was a
        // coincidence of implementation. Iter 75 fixes it.
        // ---------------------------------------------------------------
        let mut all_keys: Vec<&String> = properties.keys().collect();
        all_keys.sort();

        let slug = path_slug(path);

        // Map: property name → kv rule name.
        let mut kv_rule_name: HashMap<String, String> = HashMap::new();
        let mut required_keys: Vec<String> = Vec::new();
        let mut optional_keys: Vec<String> = Vec::new();

        for k in &all_keys {
            let v = &properties[*k];
            let vbody = self.visit(v, &format!("{}/properties/{}", path, k))?;
            // Value rule: path-slug prefix avoids collisions when two
            // different object schemas share a property name.
            let val_rule = format!("{}-{}", slug, sanitize_rule_name(k));
            self.rules.insert(val_rule.clone(), vbody);

            // kv rule: literal key + ":" + space + value-rule.
            let quoted_key = format_literal(&format!("\"{}\"", k));
            let kv_body = format!("{} \":\" space {}", quoted_key, val_rule);
            let kv_name = format!("{}-{}-kv", slug, sanitize_rule_name(k));
            self.rules.insert(kv_name.clone(), kv_body);
            kv_rule_name.insert((*k).clone(), kv_name);

            if required_list.contains(*k) {
                required_keys.push((*k).clone());
            } else {
                optional_keys.push((*k).clone());
            }
        }

        if required_keys.is_empty() && optional_keys.is_empty() {
            return Ok(r#""{" space "}" space"#.into());
        }

        // Cap: total properties (required + optional) capped at 32.
        // For schemas > 32 properties, return a clear error rather than
        // generating an exponentially large (or infinite) grammar.
        let n_total = required_keys.len() + optional_keys.len();
        if n_total > 32 {
            return Err(SchemaError {
                path: path.to_string(),
                message: format!(
                    "object schema has {} properties (required={} + optional={}); \
                     max supported for any-position grammar is 32",
                    n_total,
                    required_keys.len(),
                    optional_keys.len(),
                ),
            });
        }

        // Any-order threshold for required keys.
        //
        // The bitmask-based any-order algorithm generates O(2^N_req) unique
        // grammar rules — one per subset of remaining required keys.  This is
        // practical for small N_req but becomes intractable at N_req > ~16.
        // Threshold = 8 keeps the worst-case to 2^8 = 256 rules, which compiles
        // in microseconds while covering the common case (tools schemas rarely
        // have more than 8 required keys at the same level).
        //
        // For N_req > threshold we fall back to sequential (sorted) required-key
        // order — the same behaviour as llama.cpp's json-schema-to-grammar.cpp.
        // Optional keys still accept any order via the opt-chain.
        const ANY_ORDER_MAX_REQUIRED: usize = 8;

        // Build extra-kv wildcard rule (shared across all states if allowed).
        if !additional_closed {
            let extra_kv_name = format!("{}-extra-kv", slug);
            self.rules
                .entry(extra_kv_name)
                .or_insert_with(|| "string \":\" space value".to_string());
        }

        // Compute the inner rule reference (the first key-value pair and all
        // subsequent ones).
        let inner = if required_keys.is_empty() {
            // No required keys: the whole object body is optional.
            // Build an opt-chain for the possible keys and wrap it in `( ... )?`
            // so `{}` is also accepted.
            let mut entries: Vec<(String, bool)> = optional_keys
                .iter()
                .map(|k| (kv_rule_name[k].clone(), false))
                .collect();
            if !additional_closed {
                let extra_kv_name = format!("{}-extra-kv", slug);
                entries.push((extra_kv_name, true));
            }
            if entries.is_empty() {
                return Ok(r#""{" space "}" space"#.into());
            }
            let chain = self.build_optional_chain(&slug, &entries);
            format!("( {} )?", chain)
        } else if required_keys.len() <= ANY_ORDER_MAX_REQUIRED {
            // Few required keys: use bitmask any-order (full permutation grammar).
            //
            // Bitmask seeds — exactly n bits set for n keys in 1..=32.
            // Use `u32::MAX >> (32 - n)` to avoid shift-by-32 UB.
            let n_req = required_keys.len(); // 1..=ANY_ORDER_MAX_REQUIRED
            let req_full: u32 = u32::MAX >> (32 - n_req);
            let opt_full: u32 = if optional_keys.is_empty() {
                0
            } else {
                let n_opt = optional_keys.len(); // 1..=32
                u32::MAX >> (32 - n_opt)
            };
            self.build_unified_inner(
                &slug,
                req_full,
                opt_full,
                &required_keys,
                &optional_keys,
                &kv_rule_name,
                !additional_closed,
            )
        } else {
            // Many required keys: fall back to sequential (sorted) order.
            //
            // Required keys are emitted in sorted order (deterministic).
            // Optional keys still accept any order via the opt-chain.
            // Extra keys are accepted before any required key, between required
            // keys, and after the last required key — provided
            // additionalProperties is permissive.
            self.build_sequential_required(
                &slug,
                &required_keys,
                &optional_keys,
                &kv_rule_name,
                !additional_closed,
            )
        };

        Ok(format!(r#""{{" space {} "}}" space"#, inner))
    }

    /// Build the unified any-position inner rule for state
    /// `(req_remaining, opt_remaining)`.  Returns the name of the emitted
    /// GBNF rule.
    ///
    /// # Contract
    ///
    /// `req_remaining` MUST be non-zero on entry (the caller uses
    /// `build_optional_chain` for the all-optional case).
    ///
    /// Rule semantics: emits the first key-value pair of the current slot,
    /// then either:
    ///   - A comma + space + the next state rule, OR
    ///   - Nothing (closes the object) — only when req_remaining has a single
    ///     bit set AND opt_remaining/extra are handled by the opt-suffix tail.
    ///
    /// # Naming
    ///
    /// `{slug}-up-r{req_remaining:08x}-o{opt_remaining:08x}` — "up" for
    /// Unified Permutation; hex bitmasks are fixed-width for readability.
    fn build_unified_inner(
        &mut self,
        slug: &str,
        req_remaining: u32,
        opt_remaining: u32,
        required_keys: &[String],
        optional_keys: &[String],
        kv_rule_name: &HashMap<String, String>,
        allow_extra_kv: bool,
    ) -> String {
        let rule_name = format!(
            "{}-up-r{:08x}-o{:08x}",
            slug, req_remaining, opt_remaining
        );

        if self.rules.contains_key(&rule_name) {
            return rule_name;
        }

        // Placeholder prevents re-entrant infinite loops (defensive; the
        // state strictly decrements so no true cycles except via the extra-kv
        // self-loop which is inlined, not recursive on the same state).
        self.rules.insert(rule_name.clone(), String::new());

        let mut alts: Vec<String> = Vec::new();

        // --- Alternatives: emit one required key ---
        for (i, k) in required_keys.iter().enumerate() {
            if req_remaining & (1u32 << i) == 0 {
                continue; // already emitted
            }
            let kv = kv_rule_name[k].clone();
            let new_req = req_remaining & !(1u32 << i);

            if new_req == 0 {
                // Last required key: after it, object may close or continue
                // with optional / extra keys.  Build the optional tail suffix.
                let opt_suffix = self.build_optional_suffix_masked(
                    slug,
                    opt_remaining,
                    optional_keys,
                    kv_rule_name,
                    allow_extra_kv,
                );
                let alt = if opt_suffix.is_empty() {
                    kv.clone()
                } else {
                    format!("{} {}", kv, opt_suffix)
                };
                alts.push(alt);
            } else {
                // More required keys remain: comma is mandatory.
                let next = self.build_unified_inner(
                    slug,
                    new_req,
                    opt_remaining,
                    required_keys,
                    optional_keys,
                    kv_rule_name,
                    allow_extra_kv,
                );
                alts.push(format!("{} \",\" space {}", kv, next));
            }
        }

        // --- Alternatives: emit one optional key before all required are done ---
        for (j, o) in optional_keys.iter().enumerate() {
            if opt_remaining & (1u32 << j) == 0 {
                continue; // already emitted
            }
            let kv = kv_rule_name[o].clone();
            let new_opt = opt_remaining & !(1u32 << j);
            // Required state unchanged; comma mandatory (required keys still remain).
            let next = self.build_unified_inner(
                slug,
                req_remaining,
                new_opt,
                required_keys,
                optional_keys,
                kv_rule_name,
                allow_extra_kv,
            );
            alts.push(format!("{} \",\" space {}", kv, next));
        }

        // --- Alternative: emit one extra key before all required are done ---
        // Extra keys can repeat (wildcard), so this creates a self-loop via
        // the same state.  We inline this as an alternative that references
        // the current rule_name so GBNF handles the Kleene-star semantics.
        if allow_extra_kv {
            let extra_kv_name = format!("{}-extra-kv", slug);
            // Self-referential: extra-kv "," space <this rule>
            alts.push(format!("{} \",\" space {}", extra_kv_name, rule_name));
        }

        let body = alts.join(" | ");
        self.rules.insert(rule_name.clone(), body);
        rule_name
    }

    /// Build the optional suffix for the tail AFTER the last required key has
    /// been emitted.  Only optional keys still in `opt_mask` are considered.
    ///
    /// Returns a GBNF fragment of the form `( "," space <chain> )?` or an
    /// empty string when there are no optional keys and no extra-kv.
    fn build_optional_suffix_masked(
        &mut self,
        slug: &str,
        opt_mask: u32,
        optional_keys: &[String],
        kv_rule_name: &HashMap<String, String>,
        allow_extra_kv: bool,
    ) -> String {
        let mut entries: Vec<(String, bool)> = Vec::new();
        for (j, o) in optional_keys.iter().enumerate() {
            if opt_mask & (1u32 << j) != 0 {
                entries.push((kv_rule_name[o].clone(), false));
            }
        }
        if allow_extra_kv {
            let extra_kv_name = format!("{}-extra-kv", slug);
            entries.push((extra_kv_name, true));
        }
        if entries.is_empty() {
            return String::new();
        }
        let chain = self.build_optional_chain(slug, &entries);
        format!("( \",\" space {} )?", chain)
    }

    /// Recursively build the optional-chain rule for `entries`.
    /// Returns the name of the emitted rule.
    ///
    /// For entries [a, b] this produces:
    ///   slug-opt-<fp> ::= a-kv ( "," space slug-opt-<fp-b> )?
    ///                   | b-kv ( "," space slug-opt-<fp-a> )?
    ///
    /// The rule is keyed by a fingerprint of the sorted entry names so
    /// the same optional set encountered in different contexts shares the
    /// same rule (safe because the body is purely a function of the
    /// entry set).
    fn build_optional_chain(&mut self, slug: &str, entries: &[(String, bool)]) -> String {
        // Fingerprint: sorted kv-rule names joined and sanitized.
        let mut names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
        names.sort_unstable();
        let fp = sanitize_rule_name(&names.join("-"));
        let rule_name = format!("{}-opt-{}", slug, fp);

        if self.rules.contains_key(&rule_name) {
            return rule_name;
        }

        // Placeholder to prevent re-entrant emission (defensive).
        self.rules.insert(rule_name.clone(), String::new());

        let mut alts: Vec<String> = Vec::new();
        for (i, (kv, _is_wildcard)) in entries.iter().enumerate() {
            let remaining: Vec<(String, bool)> = entries
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, e)| e.clone())
                .collect();
            let alt = if remaining.is_empty() {
                kv.clone()
            } else {
                let rest = self.build_optional_chain(slug, &remaining);
                format!("{} ( \",\" space {} )?", kv, rest)
            };
            alts.push(alt);
        }

        let body = alts.join(" | ");
        self.rules.insert(rule_name.clone(), body);
        rule_name
    }

    /// Build a sequential (fixed-order) required-key inner rule for schemas
    /// with many required keys where the bitmask any-order algorithm would
    /// produce an exponentially large grammar.
    ///
    /// Required keys are emitted in the sorted order they were passed.
    /// Optional keys still accept any order via `build_optional_suffix_masked`.
    /// Extra keys (when `allow_extra_kv`) may appear before any required key
    /// and between consecutive required keys — via a recursive `extra-star`
    /// rule — as well as after the last required key via the opt-suffix.
    ///
    /// Returns a GBNF inline fragment (not a separate named rule) because the
    /// body is uniquely determined by the key list and has exactly one call site.
    fn build_sequential_required(
        &mut self,
        slug: &str,
        required_keys: &[String],
        optional_keys: &[String],
        kv_rule_name: &HashMap<String, String>,
        allow_extra_kv: bool,
    ) -> String {
        // n_opt bitmask seed for the optional suffix (all optional keys remain).
        let opt_full: u32 = if optional_keys.is_empty() {
            0
        } else {
            let n_opt = optional_keys.len();
            u32::MAX >> (32 - n_opt)
        };

        // Optional tail after all required keys have been emitted.
        let opt_suffix = self.build_optional_suffix_masked(
            slug,
            opt_full,
            optional_keys,
            kv_rule_name,
            allow_extra_kv,
        );

        // Zero-or-more extra keys (if additionalProperties is permissive):
        // `extra-star` is a recursive GBNF rule that matches
        // (extra-kv "," space)* — i.e., any number of extra key-value pairs
        // each followed by a mandatory comma before the next required key.
        let extra_star_prefix: String = if allow_extra_kv {
            let extra_kv_name = format!("{}-extra-kv", slug);
            let star_name = format!("{}-extra-star", slug);
            self.rules.entry(star_name.clone()).or_insert_with(|| {
                // star ::= ( extra-kv "," space star )?
                format!(r#"( {} "," space {} )?"#, extra_kv_name, star_name)
            });
            // Include a trailing space so the prefix can be directly
            // concatenated with the kv rule name.
            format!("{} ", star_name)
        } else {
            String::new()
        };

        // Build the body as a flat string:
        //   extra-star? req0-kv "," space extra-star? req1-kv ... reqN-1-kv opt-suffix?
        let mut body = String::new();
        for (idx, k) in required_keys.iter().enumerate() {
            let kv = &kv_rule_name[k];
            if idx > 0 {
                // Mandatory comma between required keys.
                body.push_str(" \",\" space ");
            }
            // Optional run of extra keys before this required key.
            body.push_str(&extra_star_prefix);
            body.push_str(kv);
        }

        if !opt_suffix.is_empty() {
            body.push(' ');
            body.push_str(&opt_suffix);
        }

        body
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
        // Both key orders must now be accepted (iter 75 fix).
        let schema = r#"{
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"]
        }"#;
        // age first (alphabetical).
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(b"{\"age\":30,\"name\":\"Bob\"}"), "age-first rejected");
        assert!(rt.is_accepted());
        // name first (non-alphabetical — was broken before iter 75).
        let mut rt2 = runtime(schema);
        assert!(rt2.accept_bytes(b"{\"name\":\"Bob\",\"age\":30}"), "name-first rejected");
        assert!(rt2.is_accepted());
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
        // arguments-first (alphabetical).
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(
            b"{\"arguments\":{\"city\":\"NYC\"},\"name\":\"get_weather\"}"
        ));
        assert!(rt.is_accepted());
        // name-first (non-alphabetical — iter 75 fix).
        let mut rt2 = runtime(schema);
        assert!(rt2.accept_bytes(
            b"{\"name\":\"get_weather\",\"arguments\":{\"city\":\"NYC\"}}"
        ));
        assert!(rt2.is_accepted());
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
        // Iter 75 fix: both required key orders accepted at every level.
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
        // filters before query (alphabetical — old behavior).
        let mut rt = runtime(schema);
        let payload =
            br#"{"filters":{"max_price":2000,"min_price":500},"query":"laptops"}"#;
        assert!(rt.accept_bytes(payload), "rejected nested (filters-first)");
        assert!(rt.is_accepted());

        // query before filters (non-alphabetical — iter 75 fix).
        let mut rt2 = runtime(schema);
        let payload2 =
            br#"{"query":"laptops","filters":{"max_price":2000,"min_price":500}}"#;
        assert!(rt2.accept_bytes(payload2), "rejected nested (query-first)");
        assert!(rt2.is_accepted());

        // Critical bug-fix anchor: missing the SECOND required field
        // (query) must be REJECTED. Pre-iter-74 this was falsely accepted.
        let mut rt = runtime(schema);
        let missing =
            br#"{"filters":{"max_price":2000,"min_price":500}}"#;
        let ok = rt.accept_bytes(missing);
        assert!(
            !(ok && rt.is_accepted()),
            "accepted object missing required 'query' (iter 74 regression)"
        );
    }

    #[test]
    fn function_call_with_enum_argument() {
        // Tool with a constrained enum field.
        // Iter 75: both key orders accepted.
        let schema = r#"{
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city", "unit"]
        }"#;
        // city first (alphabetical).
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"city":"London","unit":"celsius"}"#),
            "rejected enum value (city-first)"
        );
        assert!(rt.is_accepted());

        // unit first (non-alphabetical — iter 75).
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"unit":"celsius","city":"London"}"#),
            "rejected enum value (unit-first)"
        );
        assert!(rt.is_accepted());

        // Out-of-enum unit value must be rejected.
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{"city":"London","unit":"kelvin"}"#);
        assert!(
            !(ok && rt.is_accepted()),
            "accepted 'kelvin' not in [celsius, fahrenheit]"
        );

        // Missing required 'unit' rejected.
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{"city":"London"}"#);
        assert!(!(ok && rt.is_accepted()), "accepted object missing required 'unit'");
    }

    #[test]
    fn function_call_with_array_arguments_field() {
        // A tool that takes an array of strings.
        // Iter 75: both key orders accepted.
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
        // tags first (alphabetical).
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(
            br#"{"tags":["news","tech"],"url":"https://example.com"}"#
        ));
        assert!(rt.is_accepted());

        // url first (non-alphabetical — iter 75).
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(
            br#"{"url":"https://example.com","tags":["news","tech"]}"#
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

    // -----------------------------------------------------------------
    // PREREQ 1 — Any-order key acceptance (iter 75)
    // -----------------------------------------------------------------

    #[test]
    fn object_keys_accepted_in_any_order_three_required() {
        // Three required properties: a, b, c.
        // All 6 permutations must be accepted.
        // Objects missing any required key must be rejected.
        let schema = r#"{
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
                "c": {"type": "integer"}
            },
            "required": ["a", "b", "c"]
        }"#;

        let perms: &[&[u8]] = &[
            br#"{"a":1,"b":2,"c":3}"#,
            br#"{"a":1,"c":3,"b":2}"#,
            br#"{"b":2,"a":1,"c":3}"#,
            br#"{"b":2,"c":3,"a":1}"#,
            br#"{"c":3,"a":1,"b":2}"#,
            br#"{"c":3,"b":2,"a":1}"#,
        ];
        for perm in perms {
            let mut rt = runtime(schema);
            assert!(
                rt.accept_bytes(perm),
                "rejected permutation: {}",
                std::str::from_utf8(perm).unwrap()
            );
            assert!(
                rt.is_accepted(),
                "not accepted after: {}",
                std::str::from_utf8(perm).unwrap()
            );
        }

        // Missing required 'c'.
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{"a":1,"b":2}"#);
        assert!(!(ok && rt.is_accepted()), "accepted object missing required 'c'");

        // Missing required 'a'.
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{"b":2,"c":3}"#);
        assert!(!(ok && rt.is_accepted()), "accepted object missing required 'a'");
    }

    // -----------------------------------------------------------------
    // PREREQ 2 — additionalProperties handling (iter 75)
    // -----------------------------------------------------------------

    #[test]
    fn additional_properties_false_rejects_extra_keys() {
        // additionalProperties:false — only declared keys are accepted.
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age":  {"type": "integer"}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        }"#;

        // Valid — only declared keys.
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(br#"{"name":"Alice","age":30}"#));
        assert!(rt.is_accepted());

        // Valid in reverse order.
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(br#"{"age":30,"name":"Alice"}"#));
        assert!(rt.is_accepted());

        // Extra key "extra" not in properties — must be rejected.
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{"name":"Alice","age":30,"extra":"xxx"}"#);
        assert!(
            !(ok && rt.is_accepted()),
            "accepted extra key when additionalProperties:false"
        );
    }

    #[test]
    fn additional_properties_true_accepts_extra_keys() {
        // additionalProperties:true (explicit) — extra keys allowed.
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"],
            "additionalProperties": true
        }"#;

        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(br#"{"name":"Alice"}"#));
        assert!(rt.is_accepted());

        // Extra key must be accepted.
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"name":"Alice","extra":"xxx"}"#),
            "rejected extra key when additionalProperties:true"
        );
        assert!(rt.is_accepted());
    }

    #[test]
    fn additional_properties_unset_accepts_extra_keys() {
        // additionalProperties unset → JSON Schema default is permissive.
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }"#;

        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(br#"{"name":"Alice"}"#));
        assert!(rt.is_accepted());

        // Extra key must be accepted (default permissive).
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"name":"Alice","extra":"xxx"}"#),
            "rejected extra key when additionalProperties unset (must be permissive)"
        );
        assert!(rt.is_accepted());
    }

    // -----------------------------------------------------------------
    // Wave-2.5 W-δ C2 additions — T1.8 prereq + large-schema guard
    // -----------------------------------------------------------------

    /// T1.8 prereq object: optional key BEFORE required key.
    ///
    /// When a schema has both optional and required properties the
    /// grammar must accept any interleaving, including optional-first.
    #[test]
    fn prereq_optional_key_before_required() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name":  {"type": "string"},
                "title": {"type": "string"}
            },
            "required": ["name"]
        }"#;
        // title (optional) before name (required).
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"title":"Dr","name":"Alice"}"#),
            "optional key before required key was rejected"
        );
        assert!(rt.is_accepted(), "not accepted after optional-before-required");
    }

    /// T1.8 prereq object: extra keys interspersed around required key.
    ///
    /// extras-then-required-then-extras (with additionalProperties unset
    /// so extra keys are permissive).
    #[test]
    fn prereq_extras_surrounding_required_key() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "id": {"type": "integer"}
            },
            "required": ["id"]
        }"#;
        // extra before, id in middle, extra after.
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"before":"x","id":1,"after":"y"}"#),
            "extras-then-required-then-extras was rejected (additionalProperties unset)"
        );
        assert!(rt.is_accepted());
    }

    /// T1.8 prereq object: only extra keys, no required keys → accept when
    /// additionalProperties is unset (permissive) and there are no required fields.
    #[test]
    fn prereq_only_extra_keys_no_required() {
        let schema = r#"{
            "type": "object",
            "properties": {}
        }"#;
        // No required fields; extra keys must be accepted.
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"anything":"goes"}"#),
            "no-required-keys object with only extra keys was rejected"
        );
        assert!(rt.is_accepted());
    }

    /// T1.8 prereq object: multiple extra keys before and after multiple
    /// required keys.
    #[test]
    fn prereq_extras_before_and_after_two_required() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }"#;
        // extra, a, extra, b, extra.
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"z":0,"a":1,"y":0,"b":2,"x":0}"#),
            "extras interspersed between two required keys was rejected"
        );
        assert!(rt.is_accepted());
    }

    /// T1.8 prereq object: extra keys before all required keys.
    #[test]
    fn prereq_multiple_extras_then_required() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }"#;
        // Two extra keys, then the required key.
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"x":"v1","y":"v2","name":"Alice"}"#),
            "multiple extras before required key was rejected"
        );
        assert!(rt.is_accepted());
    }

    /// B4 — extras BEFORE required key with additionalProperties:false → reject.
    ///
    /// When additionalProperties is false the grammar is closed: only declared
    /// property keys are accepted.  An extra key appearing before the required
    /// key must cause the grammar to reject the input.
    #[test]
    fn extras_before_required_additional_properties_false_rejects() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"],
            "additionalProperties": false
        }"#;
        // Extra key before required key — must be rejected.
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{"extra":"x","name":"Alice"}"#);
        assert!(
            !(ok && rt.is_accepted()),
            "accepted extra key before required when additionalProperties:false"
        );
        // Just the required key — must be accepted.
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(br#"{"name":"Alice"}"#));
        assert!(rt.is_accepted());
    }

    /// B4 — key duplication → reject (one-time semantics).
    ///
    /// An optional key that has already been emitted must not be re-emittable
    /// at a later position.  We verify this by checking that a duplicate
    /// optional key causes the runtime to fail (either accept_bytes returns
    /// false or is_accepted returns false after the whole input).
    #[test]
    fn duplicate_optional_key_rejected() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name":  {"type": "string"},
                "title": {"type": "string"}
            },
            "required": ["name"],
            "additionalProperties": false
        }"#;
        // Duplicate optional key "title" — grammar must reject.
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{"name":"Alice","title":"Dr","title":"Prof"}"#);
        assert!(
            !(ok && rt.is_accepted()),
            "accepted duplicate optional key 'title'"
        );
        // Unique keys — must be accepted.
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(br#"{"name":"Alice","title":"Dr"}"#));
        assert!(rt.is_accepted());
    }

    /// T1.8 large-schema guard: a schema with 33 properties must return
    /// `Err(SchemaError)` because the any-position grammar state-machine
    /// cap is 32 (n_total > 32 check at json_schema.rs:484).
    ///
    /// This validates W-γ2 B4: the emitter rejects oversize schemas with a
    /// clear error rather than generating an exponentially large grammar.
    #[test]
    fn large_schema_33_properties_returns_error() {
        // Build a schema with 33 properties, all required.
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..33usize {
            let key = format!("prop{:02}", i);
            props.insert(key.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(key));
        }
        let schema = serde_json::Value::Object({
            let mut m = serde_json::Map::new();
            m.insert("type".into(), serde_json::json!("object"));
            m.insert("properties".into(), serde_json::Value::Object(props));
            m.insert("required".into(), serde_json::Value::Array(required));
            m
        });

        let err = schema_to_gbnf(&schema).unwrap_err();
        assert!(
            err.message.contains("33") || err.message.contains("max supported"),
            "expected error mentioning property count or 'max supported'; got: {:?}",
            err.message
        );
    }

    /// T1.8 large-schema guard: a schema with exactly 32 properties must
    /// compile successfully (boundary condition: 32 is the last allowed value).
    #[test]
    fn large_schema_32_properties_compiles_ok() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..32usize {
            let key = format!("prop{:02}", i);
            props.insert(key.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(key));
        }
        let schema = serde_json::Value::Object({
            let mut m = serde_json::Map::new();
            m.insert("type".into(), serde_json::json!("object"));
            m.insert("properties".into(), serde_json::Value::Object(props));
            m.insert("required".into(), serde_json::Value::Array(required));
            m
        });

        // Must not error — 32 is exactly the cap.
        let result = schema_to_gbnf(&schema);
        assert!(
            result.is_ok(),
            "32-property schema failed to compile (must be <= cap=32): {:?}",
            result.err()
        );
    }
}
